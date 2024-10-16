import Mathlib

namespace NUMINAMATH_CALUDE_odd_sum_probability_l585_58535

/-- Represents a skewed six-sided die where rolling an odd number
    is twice as likely as rolling an even number -/
structure SkewedDie where
  /-- Probability of rolling an even number -/
  prob_even : ℝ
  /-- Probability of rolling an odd number -/
  prob_odd : ℝ
  /-- The die is six-sided -/
  is_six_sided : Nat
  /-- Probability of odd is twice the probability of even -/
  odd_twice_even : prob_odd = 2 * prob_even
  /-- Total probability is 1 -/
  total_prob : prob_even + prob_odd = 1
  /-- Probabilities are non-negative -/
  prob_even_nonneg : prob_even ≥ 0
  prob_odd_nonneg : prob_odd ≥ 0

/-- The probability of obtaining an odd sum when rolling the die twice -/
def prob_odd_sum (d : SkewedDie) : ℝ :=
  2 * d.prob_even * d.prob_odd

theorem odd_sum_probability (d : SkewedDie) :
  prob_odd_sum d = 4/9 := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_probability_l585_58535


namespace NUMINAMATH_CALUDE_students_not_liking_sports_l585_58520

theorem students_not_liking_sports (total : ℕ) (basketball : ℕ) (tableTennis : ℕ) (both : ℕ) :
  total = 30 →
  basketball = 15 →
  tableTennis = 10 →
  both = 3 →
  total - (basketball + tableTennis - both) = 8 :=
by sorry

end NUMINAMATH_CALUDE_students_not_liking_sports_l585_58520


namespace NUMINAMATH_CALUDE_tommys_balloons_l585_58558

theorem tommys_balloons (initial_balloons : ℕ) (final_balloons : ℕ) : 
  initial_balloons = 26 → final_balloons = 60 → final_balloons - initial_balloons = 34 := by
  sorry

end NUMINAMATH_CALUDE_tommys_balloons_l585_58558


namespace NUMINAMATH_CALUDE_fibonacci_fraction_bound_l585_58569

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

theorem fibonacci_fraction_bound (a b n : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : n ≥ 2) :
  ((fib n / fib (n - 1) < a / b ∧ a / b < fib (n + 1) / fib n) ∨
   (fib (n + 1) / fib n < a / b ∧ a / b < fib n / fib (n - 1))) →
  b ≥ fib (n + 1) :=
by sorry

end NUMINAMATH_CALUDE_fibonacci_fraction_bound_l585_58569


namespace NUMINAMATH_CALUDE_units_digit_of_17_times_27_l585_58581

theorem units_digit_of_17_times_27 : (17 * 27) % 10 = 9 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_17_times_27_l585_58581


namespace NUMINAMATH_CALUDE_equation_solution_l585_58536

theorem equation_solution : ∃ f : ℝ, 
  ((10 * 0.3 + 2) / 4 - (3 * 0.3 - 6) / f = (2 * 0.3 + 4) / 3) ∧ 
  (abs (f - 18) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l585_58536


namespace NUMINAMATH_CALUDE_books_sold_is_24_l585_58545

/-- Calculates the number of books sold to buy a clarinet -/
def books_sold_for_clarinet (initial_savings : ℕ) (clarinet_cost : ℕ) (book_price : ℕ) : ℕ :=
  let additional_needed := clarinet_cost - initial_savings
  let halfway_savings := additional_needed / 2
  let total_to_save := halfway_savings + additional_needed
  total_to_save / book_price

theorem books_sold_is_24 :
  books_sold_for_clarinet 10 90 5 = 24 := by
  sorry

#eval books_sold_for_clarinet 10 90 5

end NUMINAMATH_CALUDE_books_sold_is_24_l585_58545


namespace NUMINAMATH_CALUDE_road_repair_hours_l585_58507

/-- Given that 39 persons can repair a road in 12 days working h hours a day,
    and 30 persons working 6 hours a day can complete the same work in 26 days,
    prove that h = 10. -/
theorem road_repair_hours (h : ℝ) : 
  39 * h * 12 = 30 * 6 * 26 → h = 10 := by
sorry

end NUMINAMATH_CALUDE_road_repair_hours_l585_58507


namespace NUMINAMATH_CALUDE_unique_intersection_l585_58537

-- Define the line equation
def line (x b : ℝ) : ℝ := 2 * x + b

-- Define the parabola equation
def parabola (x b : ℝ) : ℝ := x^2 + b * x + 1

-- Define the y-intercept of the parabola
def y_intercept (b : ℝ) : ℝ := parabola 0 b

-- Theorem statement
theorem unique_intersection :
  ∃! b : ℝ, line 0 b = y_intercept b := by sorry

end NUMINAMATH_CALUDE_unique_intersection_l585_58537


namespace NUMINAMATH_CALUDE_intersection_M_N_l585_58572

-- Define the sets M and N
def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 < 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | 2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l585_58572


namespace NUMINAMATH_CALUDE_root_sum_equation_l585_58502

theorem root_sum_equation (a b : ℝ) : 
  (Complex.I + 1) ^ 2 * a + (Complex.I + 1) * b + 2 = 0 → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_equation_l585_58502


namespace NUMINAMATH_CALUDE_interval_covering_theorem_l585_58503

/-- Definition of the interval I_k -/
def I (a : ℝ → ℝ) (k : ℕ) : Set ℝ := {x | a k ≤ x ∧ x ≤ a k + 1}

/-- The main theorem stating the minimum and maximum values of N -/
theorem interval_covering_theorem (N : ℕ) (a : ℝ → ℝ) : 
  (∀ x ∈ Set.Icc 0 100, ∃ k ∈ Finset.range N, x ∈ I a k) →
  (∀ k ∈ Finset.range N, ∃ x ∈ Set.Icc 0 100, ∀ i ∈ Finset.range N, i ≠ k → x ∉ I a i) →
  100 ≤ N ∧ N ≤ 200 := by
  sorry

end NUMINAMATH_CALUDE_interval_covering_theorem_l585_58503


namespace NUMINAMATH_CALUDE_probability_of_three_ones_l585_58571

def probability_of_sum_three (n : ℕ) (sides : ℕ) (target_sum : ℕ) : ℚ :=
  if n = 3 ∧ sides = 6 ∧ target_sum = 3 then 1 / 216 else 0

theorem probability_of_three_ones :
  probability_of_sum_three 3 6 3 = 1 / 216 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_three_ones_l585_58571


namespace NUMINAMATH_CALUDE_similar_not_congruent_l585_58596

/-- Two triangles with sides a1, b1, c1 and a2, b2, c2 respectively -/
structure Triangle :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Definition of similar triangles -/
def similar (t1 t2 : Triangle) : Prop :=
  t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c

/-- Definition of congruent triangles -/
def congruent (t1 t2 : Triangle) : Prop :=
  t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c

/-- Theorem: There exist two triangles with 3 equal angles (similar) 
    and 2 equal sides that are not congruent -/
theorem similar_not_congruent : ∃ (t1 t2 : Triangle), 
  similar t1 t2 ∧ t1.c = t2.c ∧ t1.a = t2.a ∧ ¬congruent t1 t2 := by
  sorry


end NUMINAMATH_CALUDE_similar_not_congruent_l585_58596


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l585_58557

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > b ∧ b > 0 → 1/a < 1/b) ∧
  (∃ a b : ℝ, 1/a < 1/b ∧ ¬(a > b ∧ b > 0)) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l585_58557


namespace NUMINAMATH_CALUDE_intersection_A_B_complement_B_union_P_intersection_AB_complement_P_l585_58551

open Set

-- Define the sets
def U : Set ℝ := univ
def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def P : Set ℝ := {x | x ≤ 0 ∨ 5 ≤ x}

-- Theorems to prove
theorem intersection_A_B : A ∩ B = {x | -1 < x ∧ x < 2} := by sorry

theorem complement_B_union_P : (U \ B) ∪ P = {x | x ≤ 0 ∨ 3 < x} := by sorry

theorem intersection_AB_complement_P : (A ∩ B) ∩ (U \ P) = {x | 0 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_complement_B_union_P_intersection_AB_complement_P_l585_58551


namespace NUMINAMATH_CALUDE_two_pizzas_not_enough_l585_58506

/-- Represents a pizza with its toppings -/
structure Pizza where
  hasTomatoes : Bool
  hasMushrooms : Bool
  hasSausage : Bool

/-- Represents a child's pizza preference -/
structure Preference where
  wantsTomatoes : Option Bool
  wantsMushrooms : Option Bool
  wantsSausage : Option Bool

/-- Checks if a pizza satisfies a child's preference -/
def satisfiesPreference (pizza : Pizza) (pref : Preference) : Bool :=
  (pref.wantsTomatoes.isNone || pref.wantsTomatoes == some pizza.hasTomatoes) &&
  (pref.wantsMushrooms.isNone || pref.wantsMushrooms == some pizza.hasMushrooms) &&
  (pref.wantsSausage.isNone || pref.wantsSausage == some pizza.hasSausage)

def masha : Preference := { wantsTomatoes := some true, wantsMushrooms := none, wantsSausage := some false }
def vanya : Preference := { wantsTomatoes := none, wantsMushrooms := some true, wantsSausage := none }
def dasha : Preference := { wantsTomatoes := some false, wantsMushrooms := none, wantsSausage := none }
def nikita : Preference := { wantsTomatoes := some true, wantsMushrooms := some false, wantsSausage := none }
def igor : Preference := { wantsTomatoes := none, wantsMushrooms := some false, wantsSausage := some true }

theorem two_pizzas_not_enough : 
  ∀ (pizza1 pizza2 : Pizza), 
  ¬(satisfiesPreference pizza1 masha ∨ satisfiesPreference pizza2 masha) ∨
  ¬(satisfiesPreference pizza1 vanya ∨ satisfiesPreference pizza2 vanya) ∨
  ¬(satisfiesPreference pizza1 dasha ∨ satisfiesPreference pizza2 dasha) ∨
  ¬(satisfiesPreference pizza1 nikita ∨ satisfiesPreference pizza2 nikita) ∨
  ¬(satisfiesPreference pizza1 igor ∨ satisfiesPreference pizza2 igor) :=
sorry

end NUMINAMATH_CALUDE_two_pizzas_not_enough_l585_58506


namespace NUMINAMATH_CALUDE_b_third_place_four_times_l585_58567

-- Define the structure for a contestant
structure Contestant where
  name : String
  firstPlace : Nat
  secondPlace : Nat
  thirdPlace : Nat

-- Define the competition parameters
def numCompetitions : Nat := 6
def firstPlaceScore : Nat := 5
def secondPlaceScore : Nat := 2
def thirdPlaceScore : Nat := 1

-- Define the contestants
def contestantA : Contestant := ⟨"A", 4, 1, 1⟩
def contestantB : Contestant := ⟨"B", 1, 0, 4⟩
def contestantC : Contestant := ⟨"C", 0, 3, 2⟩

-- Define the score calculation function
def calculateScore (c : Contestant) : Nat :=
  c.firstPlace * firstPlaceScore + c.secondPlace * secondPlaceScore + c.thirdPlace * thirdPlaceScore

-- Theorem to prove
theorem b_third_place_four_times :
  (calculateScore contestantA = 26) ∧
  (calculateScore contestantB = 11) ∧
  (calculateScore contestantC = 11) ∧
  (contestantB.firstPlace = 1) ∧
  (contestantA.firstPlace + contestantB.firstPlace + contestantC.firstPlace +
   contestantA.secondPlace + contestantB.secondPlace + contestantC.secondPlace +
   contestantA.thirdPlace + contestantB.thirdPlace + contestantC.thirdPlace = numCompetitions) →
  contestantB.thirdPlace = 4 := by
  sorry


end NUMINAMATH_CALUDE_b_third_place_four_times_l585_58567


namespace NUMINAMATH_CALUDE_large_ball_radius_l585_58530

theorem large_ball_radius (num_small_balls : ℕ) (small_radius : ℝ) (large_radius : ℝ) : 
  num_small_balls = 12 →
  small_radius = 2 →
  (4 / 3) * Real.pi * large_radius ^ 3 = num_small_balls * ((4 / 3) * Real.pi * small_radius ^ 3) →
  large_radius = (96 : ℝ) ^ (1 / 3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_large_ball_radius_l585_58530


namespace NUMINAMATH_CALUDE_inequality_solution_set_l585_58531

theorem inequality_solution_set :
  let S := {x : ℝ | (1 / 3 : ℝ) < x ∧ x < (5 / 3 : ℝ) ∧ x ≠ 1}
  ∀ x : ℝ, x ∈ S ↔ (1 / |x - 1| : ℝ) > (3 / 2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l585_58531


namespace NUMINAMATH_CALUDE_number_selection_theorem_l585_58548

def number_pairs : List (ℕ × ℕ) := [
  (1, 36), (2, 35), (3, 34), (4, 33),
  (5, 32), (6, 31), (7, 30), (8, 29),
  (9, 28), (10, 27), (11, 26), (12, 25)
]

def number_pairs_reduced : List (ℕ × ℕ) := [
  (1, 36), (2, 35), (3, 34), (4, 33),
  (5, 32), (6, 31), (7, 30), (8, 29),
  (9, 28), (10, 27)
]

def is_valid_selection (pairs : List (ℕ × ℕ)) (selection : List Bool) : Prop :=
  selection.length = pairs.length ∧
  (selection.zip pairs).foldl (λ sum (b, (x, y)) => sum + if b then x else y) 0 =
  (selection.zip pairs).foldl (λ sum (b, (x, y)) => sum + if b then y else x) 0

theorem number_selection_theorem :
  (∃ selection, is_valid_selection number_pairs selection) ∧
  (¬ ∃ selection, is_valid_selection number_pairs_reduced selection) := by sorry

end NUMINAMATH_CALUDE_number_selection_theorem_l585_58548


namespace NUMINAMATH_CALUDE_tens_digit_of_36_pow_12_l585_58593

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem tens_digit_of_36_pow_12 : tens_digit (36^12) = 3 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_36_pow_12_l585_58593


namespace NUMINAMATH_CALUDE_max_value_of_expression_l585_58587

theorem max_value_of_expression (x y : Real) 
  (hx : 0 < x ∧ x < π/2) (hy : 0 < y ∧ y < π/2) : 
  (Real.sqrt (Real.sqrt (Real.sin x * Real.sin y))) / 
  (Real.sqrt (Real.sqrt (Real.tan x)) + Real.sqrt (Real.sqrt (Real.tan y))) 
  ≤ Real.sqrt (Real.sqrt 8) / 4 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l585_58587


namespace NUMINAMATH_CALUDE_car_distance_l585_58584

theorem car_distance (time : ℝ) (cyclist_distance : ℝ) (speed_difference : ℝ) :
  time = 8 →
  cyclist_distance = 88 →
  speed_difference = 5 →
  let cyclist_speed := cyclist_distance / time
  let car_speed := cyclist_speed + speed_difference
  car_speed * time = 128 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_l585_58584


namespace NUMINAMATH_CALUDE_complex_number_location_l585_58515

theorem complex_number_location :
  let z : ℂ := Complex.I / (3 - 3 * Complex.I)
  (z.re < 0) ∧ (z.im > 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l585_58515


namespace NUMINAMATH_CALUDE_tile_pricing_problem_l585_58553

/-- Represents the price and discount information for tiles --/
structure TileInfo where
  basePrice : ℝ
  discountRate : ℝ
  discountThreshold : ℕ

/-- Calculates the price for a given quantity of tiles --/
def calculatePrice (info : TileInfo) (quantity : ℕ) : ℝ :=
  if quantity ≥ info.discountThreshold
  then info.basePrice * (1 - info.discountRate) * quantity
  else info.basePrice * quantity

/-- Theorem statement for the tile pricing problem --/
theorem tile_pricing_problem
  (redInfo bluInfo : TileInfo)
  (h1 : calculatePrice redInfo 4000 + calculatePrice bluInfo 6000 = 86000)
  (h2 : calculatePrice redInfo 10000 + calculatePrice bluInfo 3500 = 99000)
  (h3 : redInfo.discountRate = 0.2)
  (h4 : bluInfo.discountRate = 0.1)
  (h5 : redInfo.discountThreshold = 5000)
  (h6 : bluInfo.discountThreshold = 5000) :
  redInfo.basePrice = 8 ∧ bluInfo.basePrice = 10 ∧
  (∃ (redQty bluQty : ℕ),
    redQty + bluQty = 12000 ∧
    bluQty ≥ redQty / 2 ∧
    bluQty ≤ 6000 ∧
    calculatePrice redInfo redQty + calculatePrice bluInfo bluQty = 89800 ∧
    ∀ (r b : ℕ), r + b = 12000 → b ≥ r / 2 → b ≤ 6000 →
      calculatePrice redInfo r + calculatePrice bluInfo b ≥ 89800) :=
sorry

end NUMINAMATH_CALUDE_tile_pricing_problem_l585_58553


namespace NUMINAMATH_CALUDE_max_t_and_solution_set_l585_58592

open Real

noncomputable def f (x : ℝ) := 9 / (sin x)^2 + 4 / (cos x)^2

theorem max_t_and_solution_set :
  (∃ (t : ℝ), ∀ (x : ℝ), 0 < x ∧ x < π/2 → f x ≥ t) ∧
  (∀ (x : ℝ), 0 < x ∧ x < π/2 → f x ≥ 25) ∧
  (∀ (t : ℝ), (∀ (x : ℝ), 0 < x ∧ x < π/2 → f x ≥ t) → t ≤ 25) ∧
  ({x : ℝ | |x + 5| + |2*x - 1| ≤ 6} = {x : ℝ | 0 ≤ x ∧ x ≤ 2/3}) := by
  sorry

end NUMINAMATH_CALUDE_max_t_and_solution_set_l585_58592


namespace NUMINAMATH_CALUDE_problem_solution_l585_58501

def A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a*x + 1 = 0}

theorem problem_solution :
  (B (-1/3) ⊆ A) ∧
  (∀ a : ℝ, A ∪ B a = A ↔ a = 0 ∨ a = -1/3 ∨ a = -1/5) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l585_58501


namespace NUMINAMATH_CALUDE_a_equals_one_m_geq_two_l585_58564

/-- The function f defined as f(x) = |x + 2a| -/
def f (a : ℝ) (x : ℝ) : ℝ := |x + 2*a|

/-- Theorem stating that given the conditions, a must equal 1 -/
theorem a_equals_one (a : ℝ) : 
  (∀ x, f a x < 4 - 2*a ↔ -4 < x ∧ x < 0) → a = 1 := by sorry

/-- The function g defined as g(x) = |x + 2| -/
def g (x : ℝ) : ℝ := |x + 2|

/-- Theorem stating that given the conditions, m must be greater than or equal to 2 -/
theorem m_geq_two (m : ℝ) :
  (∀ x, g x - g (-2*x) ≤ x + m) → m ≥ 2 := by sorry

end NUMINAMATH_CALUDE_a_equals_one_m_geq_two_l585_58564


namespace NUMINAMATH_CALUDE_no_valid_n_l585_58543

theorem no_valid_n : ¬∃ (n : ℕ), n > 0 ∧ 
  (100 ≤ n / 4 ∧ n / 4 ≤ 999) ∧ 
  (100 ≤ 4 * n ∧ 4 * n ≤ 999) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_n_l585_58543


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l585_58528

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≥ 0) ↔ (∃ x : ℝ, x^2 + x + 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l585_58528


namespace NUMINAMATH_CALUDE_traffic_light_change_probability_l585_58532

/-- Represents a traffic light cycle -/
structure TrafficLightCycle where
  total_duration : ℕ
  change_interval : ℕ

/-- Calculates the probability of observing a color change -/
def probability_of_change (cycle : TrafficLightCycle) : ℚ :=
  cycle.change_interval / cycle.total_duration

theorem traffic_light_change_probability :
  let cycle : TrafficLightCycle := ⟨93, 15⟩
  probability_of_change cycle = 5 / 31 := by
  sorry

end NUMINAMATH_CALUDE_traffic_light_change_probability_l585_58532


namespace NUMINAMATH_CALUDE_student_arrangements_l585_58538

theorem student_arrangements (n : ℕ) (h : n = 5) : 
  (n - 1) * Nat.factorial (n - 1) = 96 := by
  sorry

end NUMINAMATH_CALUDE_student_arrangements_l585_58538


namespace NUMINAMATH_CALUDE_representation_inequality_l585_58512

/-- The smallest number of 1s needed to represent a positive integer using only 1s, +, ×, and brackets -/
noncomputable def f (n : ℕ) : ℕ := sorry

/-- The inequality holds for all n > 1 -/
theorem representation_inequality (n : ℕ) (hn : n > 1) :
  3 * Real.log n ≤ Real.log 3 * (f n : ℝ) ∧ Real.log 3 * (f n : ℝ) ≤ 5 * Real.log n := by
  sorry

end NUMINAMATH_CALUDE_representation_inequality_l585_58512


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_N_div_100_l585_58547

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·+1) 1

def sum_of_fractions : ℚ :=
  1 / (factorial 2 * factorial 17) +
  1 / (factorial 3 * factorial 16) +
  1 / (factorial 4 * factorial 15) +
  1 / (factorial 5 * factorial 14) +
  1 / (factorial 6 * factorial 13) +
  1 / (factorial 7 * factorial 12) +
  1 / (factorial 8 * factorial 11) +
  1 / (factorial 9 * factorial 10)

def N : ℚ := sum_of_fractions * factorial 18

theorem greatest_integer_less_than_N_div_100 :
  ⌊N / 100⌋ = 137 :=
sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_N_div_100_l585_58547


namespace NUMINAMATH_CALUDE_min_y_is_e_l585_58576

-- Define the function representing the given equation
def f (x y : ℝ) : Prop := Real.exp x = y * Real.log x + y * Real.log y

-- Theorem stating the minimum value of y
theorem min_y_is_e :
  ∃ (y_min : ℝ), y_min = Real.exp 1 ∧
  ∀ (x y : ℝ), x > 0 → y > 0 → f x y → y ≥ y_min :=
sorry

end NUMINAMATH_CALUDE_min_y_is_e_l585_58576


namespace NUMINAMATH_CALUDE_units_digit_of_4569_pow_804_l585_58585

theorem units_digit_of_4569_pow_804 : (4569^804) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_4569_pow_804_l585_58585


namespace NUMINAMATH_CALUDE_unequal_grandchildren_probability_l585_58554

def num_grandchildren : ℕ := 12

def prob_male : ℚ := 1/2

def prob_female : ℚ := 1/2

theorem unequal_grandchildren_probability :
  let total_outcomes := 2^num_grandchildren
  let equal_outcomes := (num_grandchildren.choose (num_grandchildren / 2))
  (total_outcomes - equal_outcomes) / total_outcomes = 3172/4096 :=
sorry

end NUMINAMATH_CALUDE_unequal_grandchildren_probability_l585_58554


namespace NUMINAMATH_CALUDE_polar_to_rectangular_min_a_for_inequality_l585_58598

-- Part A
theorem polar_to_rectangular (ρ θ : ℝ) (x y : ℝ) :
  ρ^2 * Real.cos θ - ρ = 0 ↔ x = 1 :=
sorry

-- Part B
theorem min_a_for_inequality (a : ℝ) :
  (∀ x ∈ Set.Icc 0 5, |2 - x| + |x + 1| ≤ a) ↔ a ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_min_a_for_inequality_l585_58598


namespace NUMINAMATH_CALUDE_m_plus_2n_equals_neg_one_l585_58522

theorem m_plus_2n_equals_neg_one (m n : ℝ) (h : |m - 3| + (n + 2)^2 = 0) : m + 2*n = -1 := by
  sorry

end NUMINAMATH_CALUDE_m_plus_2n_equals_neg_one_l585_58522


namespace NUMINAMATH_CALUDE_factory_production_l585_58521

/-- Calculates the total television production in the second year given the daily production rate in the first year and the reduction percentage. -/
def secondYearProduction (dailyRate : ℕ) (reductionPercent : ℕ) : ℕ :=
  let firstYearTotal := dailyRate * 365
  let reduction := firstYearTotal * reductionPercent / 100
  firstYearTotal - reduction

/-- Theorem stating that for a factory producing 10 televisions per day in the first year
    and reducing production by 10% in the second year, the total production in the second year is 3285. -/
theorem factory_production :
  secondYearProduction 10 10 = 3285 := by
  sorry

end NUMINAMATH_CALUDE_factory_production_l585_58521


namespace NUMINAMATH_CALUDE_polynomial_roots_problem_l585_58573

/-- 
Given two real numbers u and v that are roots of the polynomial r(x) = x^3 + cx + d,
and u+3 and v-2 are roots of another polynomial s(x) = x^3 + cx + d + 153,
prove that the only possible value for d is 0.
-/
theorem polynomial_roots_problem (u v c d : ℝ) : 
  (u^3 + c*u + d = 0) →
  (v^3 + c*v + d = 0) →
  ((u+3)^3 + c*(u+3) + d + 153 = 0) →
  ((v-2)^3 + c*(v-2) + d + 153 = 0) →
  d = 0 := by
  sorry


end NUMINAMATH_CALUDE_polynomial_roots_problem_l585_58573


namespace NUMINAMATH_CALUDE_sum_in_base4_l585_58540

/-- Represents a number in base 4 -/
def Base4 : Type := List (Fin 4)

/-- Addition of two Base4 numbers -/
def add_base4 : Base4 → Base4 → Base4 := sorry

/-- Conversion from a natural number to Base4 -/
def nat_to_base4 : ℕ → Base4 := sorry

/-- Conversion from Base4 to a natural number -/
def base4_to_nat : Base4 → ℕ := sorry

theorem sum_in_base4 :
  let a : Base4 := nat_to_base4 211
  let b : Base4 := nat_to_base4 332
  let c : Base4 := nat_to_base4 123
  let result : Base4 := nat_to_base4 1120
  add_base4 (add_base4 a b) c = result := by sorry

end NUMINAMATH_CALUDE_sum_in_base4_l585_58540


namespace NUMINAMATH_CALUDE_ratio_bounds_l585_58510

theorem ratio_bounds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a ≤ b + c) (h2 : b + c ≤ 2 * a) (h3 : b ≤ a + c) (h4 : a + c ≤ 2 * b) :
  2 / 3 ≤ b / a ∧ b / a ≤ 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_ratio_bounds_l585_58510


namespace NUMINAMATH_CALUDE_root_in_interval_implies_m_range_l585_58586

theorem root_in_interval_implies_m_range (m : ℝ) :
  (∃ x ∈ Set.Icc (-2) 1, 2 * m * x + 4 = 0) →
  m ∈ Set.Iic (-2) ∪ Set.Ici 1 := by
sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_m_range_l585_58586


namespace NUMINAMATH_CALUDE_addition_and_multiplication_of_integers_l585_58509

theorem addition_and_multiplication_of_integers : 
  (-3 + 2 = -1) ∧ ((-3) * 2 = -6) := by sorry

end NUMINAMATH_CALUDE_addition_and_multiplication_of_integers_l585_58509


namespace NUMINAMATH_CALUDE_product_simplification_l585_58578

theorem product_simplification (y : ℝ) : (16 * y^3) * (12 * y^5) * (1 / (4 * y)^3) = 3 * y^5 := by
  sorry

end NUMINAMATH_CALUDE_product_simplification_l585_58578


namespace NUMINAMATH_CALUDE_trace_of_matrix_minus_inverse_zero_l585_58583

/-- Given a 2x2 matrix A with real entries a, 2, -3, and d,
    if A - A^(-1) is the zero matrix, then the trace of A is a + d. -/
theorem trace_of_matrix_minus_inverse_zero (a d : ℝ) :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![a, 2; -3, d]
  (A - A⁻¹ = 0) → Matrix.trace A = a + d := by
  sorry

end NUMINAMATH_CALUDE_trace_of_matrix_minus_inverse_zero_l585_58583


namespace NUMINAMATH_CALUDE_price_difference_l585_58579

theorem price_difference (P : ℝ) (h : P > 0) :
  let new_price := P * 1.2
  let discounted_price := new_price * 0.8
  new_price - discounted_price = P * 0.24 := by
sorry

end NUMINAMATH_CALUDE_price_difference_l585_58579


namespace NUMINAMATH_CALUDE_stratified_sampling_management_l585_58546

theorem stratified_sampling_management (total_employees : ℕ) (management : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 150)
  (h2 : management = 15)
  (h3 : sample_size = 30) :
  (management * sample_size) / total_employees = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_management_l585_58546


namespace NUMINAMATH_CALUDE_inequality_proof_l585_58595

theorem inequality_proof (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) :
  (1/2) * (a + b)^2 + (1/4) * (a + b) ≥ a * Real.sqrt b + b * Real.sqrt a :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l585_58595


namespace NUMINAMATH_CALUDE_sixth_number_tenth_row_l585_58563

/-- Represents a triangular number array with specific properties -/
structure TriangularArray where
  -- The first number of each row forms an arithmetic sequence
  first_term : ℚ
  common_difference : ℚ
  -- The numbers in each row form a geometric sequence
  common_ratio : ℚ

/-- Get the nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  a₁ + (n - 1) * d

/-- Get the nth term of a geometric sequence -/
def geometricSequenceTerm (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n - 1)

/-- The main theorem -/
theorem sixth_number_tenth_row (arr : TriangularArray) 
  (h1 : arr.first_term = 1/4)
  (h2 : arr.common_difference = 1/4)
  (h3 : arr.common_ratio = 1/2) :
  let first_number_tenth_row := arithmeticSequenceTerm arr.first_term arr.common_difference 10
  geometricSequenceTerm first_number_tenth_row arr.common_ratio 6 = 5/64 := by
  sorry

end NUMINAMATH_CALUDE_sixth_number_tenth_row_l585_58563


namespace NUMINAMATH_CALUDE_tetrahedron_volume_and_surface_area_l585_58542

/-- A regular tetrahedron with given height and base edge length -/
structure RegularTetrahedron where
  height : ℝ
  base_edge : ℝ

/-- Volume of a regular tetrahedron -/
def volume (t : RegularTetrahedron) : ℝ := sorry

/-- Surface area of a regular tetrahedron -/
def surface_area (t : RegularTetrahedron) : ℝ := sorry

/-- Theorem stating the volume and surface area of a specific regular tetrahedron -/
theorem tetrahedron_volume_and_surface_area :
  let t : RegularTetrahedron := ⟨1, 2 * Real.sqrt 6⟩
  volume t = 2 * Real.sqrt 3 ∧
  surface_area t = 9 * Real.sqrt 2 + 6 * Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_and_surface_area_l585_58542


namespace NUMINAMATH_CALUDE_hyperbola_equation_l585_58562

/-- A hyperbola is defined by its equation in the form ax^2 + by^2 = c,
    where a, b, and c are real numbers and a and b have opposite signs. -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_opposite_signs : a * b < 0

/-- The point (x, y) in ℝ² -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a hyperbola -/
def point_on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  h.a * p.x^2 + h.b * p.y^2 = h.c

/-- Two hyperbolas have the same asymptotes if their equations are proportional -/
def same_asymptotes (h1 h2 : Hyperbola) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ h1.a = k * h2.a ∧ h1.b = k * h2.b

theorem hyperbola_equation (h1 : Hyperbola) (h2 : Hyperbola) (p : Point) :
  same_asymptotes h1 { a := 1, b := -1/4, c := 1, h_opposite_signs := sorry } →
  point_on_hyperbola h2 { x := 2, y := 0 } →
  h2 = { a := 1/4, b := -1/16, c := 1, h_opposite_signs := sorry } :=
sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l585_58562


namespace NUMINAMATH_CALUDE_symmetry_axis_l585_58549

-- Define a function f with the given symmetry property
def f : ℝ → ℝ := sorry

-- State the symmetry property of f
axiom f_symmetry (x : ℝ) : f x = f (3 - x)

-- Define what it means for a vertical line to be an axis of symmetry
def is_axis_of_symmetry (a : ℝ) : Prop :=
  ∀ x y : ℝ, f x = y → f (2 * a - x) = y

-- Theorem stating that x = 1.5 is an axis of symmetry
theorem symmetry_axis :
  is_axis_of_symmetry 1.5 :=
sorry

end NUMINAMATH_CALUDE_symmetry_axis_l585_58549


namespace NUMINAMATH_CALUDE_max_voters_with_95_percent_support_l585_58517

/-- Represents the election scenario with an initial poll and subsequent groups -/
structure ElectionPoll where
  initial_voters : ℕ
  initial_support : ℕ
  group_size : ℕ
  group_support : ℕ

/-- Calculates the total number of voters and supporters for a given number of additional groups -/
def totalVoters (poll : ElectionPoll) (additional_groups : ℕ) : ℕ × ℕ :=
  (poll.initial_voters + poll.group_size * additional_groups,
   poll.initial_support + poll.group_support * additional_groups)

/-- Checks if the support percentage is at least 95% -/
def isSupportAboveThreshold (total : ℕ) (support : ℕ) : Prop :=
  (support : ℚ) / (total : ℚ) ≥ 95 / 100

/-- Theorem stating the maximum number of voters while maintaining 95% support -/
theorem max_voters_with_95_percent_support :
  ∃ (poll : ElectionPoll) (max_groups : ℕ),
    poll.initial_voters = 100 ∧
    poll.initial_support = 98 ∧
    poll.group_size = 10 ∧
    poll.group_support = 9 ∧
    (let (total, support) := totalVoters poll max_groups
     isSupportAboveThreshold total support) ∧
    (∀ g > max_groups,
      let (total, support) := totalVoters poll g
      ¬(isSupportAboveThreshold total support)) ∧
    poll.initial_voters + poll.group_size * max_groups = 160 :=
  sorry

end NUMINAMATH_CALUDE_max_voters_with_95_percent_support_l585_58517


namespace NUMINAMATH_CALUDE_artist_painting_hours_l585_58525

/-- Represents the number of hours an artist spends painting per week. -/
def hours_per_week (hours_per_painting : ℕ) (paintings_in_four_weeks : ℕ) : ℕ :=
  (hours_per_painting * paintings_in_four_weeks) / 4

/-- Theorem stating that an artist who takes 3 hours to complete a painting
    and can make 40 paintings in four weeks spends 30 hours painting every week. -/
theorem artist_painting_hours :
  hours_per_week 3 40 = 30 := by
  sorry

end NUMINAMATH_CALUDE_artist_painting_hours_l585_58525


namespace NUMINAMATH_CALUDE_percentage_difference_l585_58597

theorem percentage_difference (x y : ℝ) (h : x = 12 * y) :
  (x - y) / x * 100 = 91.67 :=
sorry

end NUMINAMATH_CALUDE_percentage_difference_l585_58597


namespace NUMINAMATH_CALUDE_smallest_rectangle_cover_l585_58568

/-- The width of the rectangle -/
def rectangle_width : ℕ := 3

/-- The height of the rectangle -/
def rectangle_height : ℕ := 4

/-- The area of a single rectangle -/
def rectangle_area : ℕ := rectangle_width * rectangle_height

/-- The side length of the smallest square that can be covered by whole rectangles -/
def square_side : ℕ := lcm rectangle_width rectangle_height

/-- The area of the square -/
def square_area : ℕ := square_side * square_side

/-- The number of rectangles needed to cover the square -/
def num_rectangles : ℕ := square_area / rectangle_area

theorem smallest_rectangle_cover :
  num_rectangles = 12 ∧
  ∀ n : ℕ, n < num_rectangles → 
    ¬ (∃ s : ℕ, s * s = n * rectangle_area) :=
sorry

end NUMINAMATH_CALUDE_smallest_rectangle_cover_l585_58568


namespace NUMINAMATH_CALUDE_nominal_rate_for_given_ear_l585_58552

/-- Given an effective annual rate and compounding frequency, 
    calculate the nominal rate of interest per annum. -/
def nominal_rate (ear : ℝ) (n : ℕ) : ℝ :=
  n * ((1 + ear) ^ (1 / n) - 1)

/-- Theorem stating that for an effective annual rate of 12.36% 
    with half-yearly compounding, the nominal rate is approximately 11.66% -/
theorem nominal_rate_for_given_ear :
  let ear := 0.1236
  let n := 2
  abs (nominal_rate ear n - 0.1166) < 0.0001 := by sorry

end NUMINAMATH_CALUDE_nominal_rate_for_given_ear_l585_58552


namespace NUMINAMATH_CALUDE_max_value_sin_cos_sum_l585_58574

theorem max_value_sin_cos_sum (a b : ℝ) :
  ∃ (M : ℝ), M = Real.sqrt (a^2 + b^2) ∧
  (∀ t : ℝ, 0 < t ∧ t < 2 * Real.pi → a * Real.sin t + b * Real.cos t ≤ M) ∧
  (∃ t : ℝ, 0 < t ∧ t < 2 * Real.pi ∧ a * Real.sin t + b * Real.cos t = M) :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_cos_sum_l585_58574


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l585_58566

theorem fraction_sum_equality : (18 : ℚ) / 42 - 2 / 9 + 1 / 14 = 5 / 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l585_58566


namespace NUMINAMATH_CALUDE_quadratic_sum_l585_58527

/-- A quadratic function passing through two given points -/
structure QuadraticFunction where
  p : ℝ
  q : ℝ
  r : ℝ
  point1 : p + q + r = 5
  point2 : 4*p + 2*q + r = 3

/-- The theorem stating that p+q+2r equals 10 for the given quadratic function -/
theorem quadratic_sum (g : QuadraticFunction) : g.p + g.q + 2*g.r = 10 := by
  sorry

#check quadratic_sum

end NUMINAMATH_CALUDE_quadratic_sum_l585_58527


namespace NUMINAMATH_CALUDE_students_without_A_l585_58560

theorem students_without_A (total : ℕ) (science_A : ℕ) (english_A : ℕ) (both_A : ℕ) : 
  total - (science_A + english_A - both_A) = 18 :=
by
  sorry

#check students_without_A 40 10 18 6

end NUMINAMATH_CALUDE_students_without_A_l585_58560


namespace NUMINAMATH_CALUDE_geometric_figure_x_length_l585_58570

theorem geometric_figure_x_length 
  (total_area : ℝ)
  (square1_side : ℝ → ℝ)
  (square2_side : ℝ → ℝ)
  (triangle_leg1 : ℝ → ℝ)
  (triangle_leg2 : ℝ → ℝ)
  (h1 : total_area = 1000)
  (h2 : ∀ x, square1_side x = 3 * x)
  (h3 : ∀ x, square2_side x = 4 * x)
  (h4 : ∀ x, triangle_leg1 x = 3 * x)
  (h5 : ∀ x, triangle_leg2 x = 4 * x)
  (h6 : ∀ x, (square1_side x)^2 + (square2_side x)^2 + 1/2 * (triangle_leg1 x) * (triangle_leg2 x) = total_area) :
  ∃ x : ℝ, x = 10 * Real.sqrt 31 / 31 := by
  sorry

end NUMINAMATH_CALUDE_geometric_figure_x_length_l585_58570


namespace NUMINAMATH_CALUDE_same_speed_problem_l585_58575

theorem same_speed_problem (x : ℝ) : 
  let jack_speed := x^2 - 11*x - 22
  let jill_distance := x^2 - 5*x - 60
  let jill_time := x + 6
  let jill_speed := jill_distance / jill_time
  jack_speed = jill_speed → jack_speed = 4 :=
by sorry

end NUMINAMATH_CALUDE_same_speed_problem_l585_58575


namespace NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l585_58529

theorem greatest_perimeter_of_special_triangle :
  ∀ (a b c : ℕ),
    a > 0 ∧ b > 0 ∧ c > 0 →
    b = 4 * a →
    c = 20 →
    a + b > c ∧ a + c > b ∧ b + c > a →
    a + b + c ≤ 50 :=
by sorry

end NUMINAMATH_CALUDE_greatest_perimeter_of_special_triangle_l585_58529


namespace NUMINAMATH_CALUDE_zoo_birds_count_l585_58514

theorem zoo_birds_count (non_bird_animals : ℕ) : 
  (5 * non_bird_animals = non_bird_animals + 360) → 
  (5 * non_bird_animals = 450) := by
sorry

end NUMINAMATH_CALUDE_zoo_birds_count_l585_58514


namespace NUMINAMATH_CALUDE_binomial_expansion_problem_l585_58541

def binomial_sum (n : ℕ) : ℕ := 2^(n-1)

def rational_terms (n : ℕ) : List (ℕ × ℕ) :=
  [(5, 1), (4, 210)]

def coefficient_x_squared (n : ℕ) : ℕ :=
  (Finset.range (n - 2)).sum (λ k => Nat.choose (k + 3) 2)

theorem binomial_expansion_problem (n : ℕ) 
  (h : binomial_sum n = 512) : 
  n = 10 ∧ 
  rational_terms n = [(5, 1), (4, 210)] ∧
  coefficient_x_squared n = 164 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_problem_l585_58541


namespace NUMINAMATH_CALUDE_no_simultaneous_squares_l585_58523

theorem no_simultaneous_squares (x y : ℕ) : ¬(∃ (a b : ℕ), x^2 + y = a^2 ∧ y^2 + x = b^2) := by
  sorry

end NUMINAMATH_CALUDE_no_simultaneous_squares_l585_58523


namespace NUMINAMATH_CALUDE_no_integer_points_in_sphere_intersection_l585_58555

theorem no_integer_points_in_sphere_intersection : 
  ¬∃ (x y z : ℤ), (x^2 + y^2 + (z - 10)^2 ≤ 9) ∧ (x^2 + y^2 + (z - 2)^2 ≤ 16) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_points_in_sphere_intersection_l585_58555


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l585_58519

theorem triangle_abc_properties (A B C : ℝ) (h_obtuse : π / 2 < C ∧ C < π) 
  (h_sin_2c : Real.sin (2 * C) = Real.sqrt 3 * Real.cos C) 
  (h_b : Real.sqrt (A^2 + C^2 - 2*A*C*Real.cos B) = 6) 
  (h_area : 1/2 * Real.sqrt (A^2 + C^2 - 2*A*C*Real.cos B) * 
    Real.sqrt (B^2 + C^2 - 2*B*C*Real.cos A) * Real.sin C = 6 * Real.sqrt 3) : 
  C = 2 * π / 3 ∧ 
  Real.sqrt (A^2 + C^2 - 2*A*C*Real.cos B) + 
  Real.sqrt (B^2 + C^2 - 2*B*C*Real.cos A) + 
  Real.sqrt (A^2 + B^2 - 2*A*B*Real.cos C) = 10 + 2 * Real.sqrt 19 := by
  sorry


end NUMINAMATH_CALUDE_triangle_abc_properties_l585_58519


namespace NUMINAMATH_CALUDE_greatest_common_factor_36_54_81_l585_58565

theorem greatest_common_factor_36_54_81 : Nat.gcd 36 (Nat.gcd 54 81) = 9 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_factor_36_54_81_l585_58565


namespace NUMINAMATH_CALUDE_odometer_difference_l585_58533

theorem odometer_difference (initial_reading final_reading : ℝ) 
  (h1 : initial_reading = 212.3)
  (h2 : final_reading = 584.3) :
  final_reading - initial_reading = 372 := by
  sorry

end NUMINAMATH_CALUDE_odometer_difference_l585_58533


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l585_58505

-- Define a function to check if a number is a perfect square
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Define what it means for a quadratic radical to be in its simplest form
def isSimplestQuadraticRadical (x : ℝ) : Prop :=
  x > 0 ∧ ¬∃ (a b : ℕ), (b > 1) ∧ (¬isPerfectSquare b) ∧ (x = (a : ℝ) * Real.sqrt b)

-- Theorem statement
theorem simplest_quadratic_radical :
  isSimplestQuadraticRadical (Real.sqrt 7) ∧
  ¬isSimplestQuadraticRadical (Real.sqrt 12) ∧
  ¬isSimplestQuadraticRadical (Real.sqrt (2/3)) ∧
  ¬isSimplestQuadraticRadical (Real.sqrt 0.3) :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l585_58505


namespace NUMINAMATH_CALUDE_best_fit_line_slope_for_given_data_l585_58513

/-- Data point representing height and weight measurements -/
structure DataPoint where
  height : ℝ
  weight : ℝ

/-- Calculate the slope of the best-fit line for given data points -/
def bestFitLineSlope (data : List DataPoint) : ℝ :=
  sorry

/-- Theorem stating that the slope of the best-fit line for the given data is 0.525 -/
theorem best_fit_line_slope_for_given_data :
  let data := [
    DataPoint.mk 150 50,
    DataPoint.mk 160 55,
    DataPoint.mk 170 60.5
  ]
  bestFitLineSlope data = 0.525 := by
  sorry

end NUMINAMATH_CALUDE_best_fit_line_slope_for_given_data_l585_58513


namespace NUMINAMATH_CALUDE_sqrt_less_than_5x_iff_l585_58591

theorem sqrt_less_than_5x_iff (x : ℝ) (h : x > 0) :
  Real.sqrt x < 5 * x ↔ x > 1 / 25 := by sorry

end NUMINAMATH_CALUDE_sqrt_less_than_5x_iff_l585_58591


namespace NUMINAMATH_CALUDE_inverse_f_zero_solution_l585_58516

noncomputable section

variables (a b c : ℝ)
variable (f : ℝ → ℝ)

-- Define the function f
def f_def : f = λ x => 1 / (a * x^2 + b * x + c) := by sorry

-- Conditions: a, b, and c are nonzero
axiom a_nonzero : a ≠ 0
axiom b_nonzero : b ≠ 0
axiom c_nonzero : c ≠ 0

-- Theorem: The only solution to f^(-1)(x) = 0 is x = 1/c
theorem inverse_f_zero_solution :
  ∀ x : ℝ, (Function.invFun f) x = 0 ↔ x = 1 / c := by sorry

end

end NUMINAMATH_CALUDE_inverse_f_zero_solution_l585_58516


namespace NUMINAMATH_CALUDE_smallest_positive_angle_same_terminal_side_l585_58559

/-- Given an angle α = 2012°, the smallest positive angle θ with the same terminal side is 212°. -/
theorem smallest_positive_angle_same_terminal_side :
  let α : Real := 2012
  ∃ θ : Real,
    0 < θ ∧ 
    θ ≤ 360 ∧
    ∃ k : Int, α = k * 360 + θ ∧
    ∀ φ : Real, (0 < φ ∧ φ ≤ 360 ∧ ∃ m : Int, α = m * 360 + φ) → θ ≤ φ ∧
    θ = 212 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_same_terminal_side_l585_58559


namespace NUMINAMATH_CALUDE_smallest_solution_floor_equation_l585_58534

theorem smallest_solution_floor_equation :
  ∃ (x : ℝ), x = 3 * Real.sqrt 2 ∧
  (∀ y : ℝ, y > 0 → ⌊y^2⌋ - ⌊y⌋^2 = 12 → y ≥ x) ∧
  ⌊x^2⌋ - ⌊x⌋^2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_floor_equation_l585_58534


namespace NUMINAMATH_CALUDE_ball_attendance_l585_58508

theorem ball_attendance :
  ∀ (n m : ℕ),
  n + m < 50 →
  (3 * n) / 4 = (5 * m) / 7 →
  n + m = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_attendance_l585_58508


namespace NUMINAMATH_CALUDE_coefficient_x5y3_in_binomial_expansion_l585_58582

theorem coefficient_x5y3_in_binomial_expansion :
  (Finset.range 9).sum (fun k => Nat.choose 8 k * (1 : ℕ)^(8 - k) * (1 : ℕ)^k) = 256 ∧
  Nat.choose 8 3 = 56 :=
by sorry

end NUMINAMATH_CALUDE_coefficient_x5y3_in_binomial_expansion_l585_58582


namespace NUMINAMATH_CALUDE_binomial_150_150_l585_58580

theorem binomial_150_150 : (150 : ℕ).choose 150 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_150_150_l585_58580


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l585_58599

theorem nested_fraction_equality : 2 + 1 / (2 + 1 / (2 + 2)) = 22 / 9 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l585_58599


namespace NUMINAMATH_CALUDE_jakes_balloons_l585_58526

/-- Given information about balloons brought to a park, prove how many balloons Jake brought initially. -/
theorem jakes_balloons (allan_initial : ℕ) (allan_bought : ℕ) (total : ℕ) 
  (h1 : allan_initial = 3)
  (h2 : allan_bought = 2)
  (h3 : total = 10) :
  total - (allan_initial + allan_bought) = 5 := by
sorry

end NUMINAMATH_CALUDE_jakes_balloons_l585_58526


namespace NUMINAMATH_CALUDE_jake_watched_19_hours_on_friday_l585_58518

/-- Represents the number of hours in a day -/
def hours_per_day : ℕ := 24

/-- Represents the total length of the show in hours -/
def show_length : ℕ := 52

/-- Calculates the hours Jake watched on Monday -/
def monday_hours : ℕ := hours_per_day / 2

/-- Represents the hours Jake watched on Tuesday -/
def tuesday_hours : ℕ := 4

/-- Calculates the hours Jake watched on Wednesday -/
def wednesday_hours : ℕ := hours_per_day / 4

/-- Calculates the total hours Jake watched from Monday to Wednesday -/
def mon_to_wed_total : ℕ := monday_hours + tuesday_hours + wednesday_hours

/-- Calculates the hours Jake watched on Thursday -/
def thursday_hours : ℕ := mon_to_wed_total / 2

/-- Calculates the total hours Jake watched from Monday to Thursday -/
def mon_to_thu_total : ℕ := mon_to_wed_total + thursday_hours

/-- Represents the hours Jake watched on Friday -/
def friday_hours : ℕ := show_length - mon_to_thu_total

theorem jake_watched_19_hours_on_friday : friday_hours = 19 := by
  sorry

end NUMINAMATH_CALUDE_jake_watched_19_hours_on_friday_l585_58518


namespace NUMINAMATH_CALUDE_equation_solution_l585_58524

theorem equation_solution (x : ℝ) : 3 * x + 2 = 11 → 5 * x - 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l585_58524


namespace NUMINAMATH_CALUDE_twenty_second_visits_l585_58550

/-- Represents the tanning salon scenario --/
structure TanningSalon where
  total_customers : ℕ
  first_visit_charge : ℕ
  subsequent_visit_charge : ℕ
  third_visit_customers : ℕ
  total_revenue : ℕ

/-- Calculates the number of customers who made a second visit --/
def second_visit_customers (ts : TanningSalon) : ℕ :=
  (ts.total_revenue - ts.total_customers * ts.first_visit_charge - ts.third_visit_customers * ts.subsequent_visit_charge) / ts.subsequent_visit_charge

/-- Theorem stating that 20 customers made a second visit --/
theorem twenty_second_visits (ts : TanningSalon) 
  (h1 : ts.total_customers = 100)
  (h2 : ts.first_visit_charge = 10)
  (h3 : ts.subsequent_visit_charge = 8)
  (h4 : ts.third_visit_customers = 10)
  (h5 : ts.total_revenue = 1240) :
  second_visit_customers ts = 20 := by
  sorry

end NUMINAMATH_CALUDE_twenty_second_visits_l585_58550


namespace NUMINAMATH_CALUDE_shift_repeating_segment_2011th_digit_6_l585_58590

/-- Represents a repeating decimal with an initial non-repeating part and a repeating segment. -/
structure RepeatingDecimal where
  initial : ℚ
  repeating : List ℕ

/-- Shifts the repeating segment of a repeating decimal. -/
def shiftRepeatingSegment (d : RepeatingDecimal) (n : ℕ) : RepeatingDecimal :=
  sorry

/-- Gets the nth digit after the decimal point in a repeating decimal. -/
def nthDigitAfterDecimal (d : RepeatingDecimal) (n : ℕ) : ℕ :=
  sorry

/-- The main theorem about shifting the repeating segment. -/
theorem shift_repeating_segment_2011th_digit_6 (d : RepeatingDecimal) :
  d.initial = 0.1 ∧ d.repeating = [2, 3, 4, 5, 6, 7, 8] →
  ∃ (k : ℕ), 
    let d' := shiftRepeatingSegment d k
    nthDigitAfterDecimal d' 2011 = 6 ∧
    d'.initial = 0.1 ∧ d'.repeating = [2, 3, 4, 5, 6, 7, 8] :=
  sorry

end NUMINAMATH_CALUDE_shift_repeating_segment_2011th_digit_6_l585_58590


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l585_58561

theorem arithmetic_sequence_length
  (a : ℕ) (d : ℕ) (last : ℕ) (n : ℕ)
  (h1 : a = 3)
  (h2 : d = 4)
  (h3 : last = 47)
  (h4 : last = a + (n - 1) * d) :
  n = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l585_58561


namespace NUMINAMATH_CALUDE_line_x_coordinate_l585_58589

/-- Given a line passing through (x, -4) and (10, 3) with x-intercept 4, 
    prove that x = -4 -/
theorem line_x_coordinate (x : ℝ) : 
  (∃ (m b : ℝ), (∀ (t : ℝ), -4 = m * x + b) ∧ 
                 (3 = m * 10 + b) ∧ 
                 (0 = m * 4 + b)) →
  x = -4 := by
  sorry

end NUMINAMATH_CALUDE_line_x_coordinate_l585_58589


namespace NUMINAMATH_CALUDE_least_positive_integer_mod_l585_58504

theorem least_positive_integer_mod (n : ℕ) : 
  ∃ x : ℕ, x > 0 ∧ (x + 7237 : ℤ) ≡ 5017 [ZMOD 12] ∧ 
  ∀ y : ℕ, y > 0 ∧ (y + 7237 : ℤ) ≡ 5017 [ZMOD 12] → x ≤ y :=
by
  use 12
  sorry

end NUMINAMATH_CALUDE_least_positive_integer_mod_l585_58504


namespace NUMINAMATH_CALUDE_wendy_trip_miles_l585_58511

theorem wendy_trip_miles : 
  let day1_miles : ℕ := 125
  let day2_miles : ℕ := 223
  let day3_miles : ℕ := 145
  day1_miles + day2_miles + day3_miles = 493 := by sorry

end NUMINAMATH_CALUDE_wendy_trip_miles_l585_58511


namespace NUMINAMATH_CALUDE_perfect_squares_difference_l585_58577

theorem perfect_squares_difference (n : ℕ) : 
  (∃ a : ℕ, n - 52 = a^2) ∧ (∃ b : ℕ, n + 37 = b^2) → n = 1988 := by
  sorry

end NUMINAMATH_CALUDE_perfect_squares_difference_l585_58577


namespace NUMINAMATH_CALUDE_log_like_function_72_l585_58539

/-- A function satisfying f(ab) = f(a) + f(b) for all a and b -/
def LogLikeFunction (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f (a * b) = f a + f b

/-- Theorem: If f is a LogLikeFunction with f(2) = m and f(3) = n, then f(72) = 3m + 2n -/
theorem log_like_function_72 (f : ℝ → ℝ) (m n : ℝ) 
  (h_log_like : LogLikeFunction f) (h_2 : f 2 = m) (h_3 : f 3 = n) : 
  f 72 = 3 * m + 2 * n := by
sorry

end NUMINAMATH_CALUDE_log_like_function_72_l585_58539


namespace NUMINAMATH_CALUDE_difference_of_squares_51_50_l585_58588

-- Define the function for squaring a number
def square (n : ℕ) : ℕ := n * n

-- State the theorem
theorem difference_of_squares_51_50 : square 51 - square 50 = 101 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_51_50_l585_58588


namespace NUMINAMATH_CALUDE_vector_ratio_theorem_l585_58594

/-- Given points O, A, B, C in a Cartesian coordinate system where O is the origin,
    prove that the ratio of the magnitudes of BC to AC is 3,
    given that OC is a weighted sum of OA and OB. -/
theorem vector_ratio_theorem (O A B C : ℝ × ℝ) :
  O = (0, 0) →
  C - O = 3/4 • (A - O) + 1/4 • (B - O) →
  ‖C - B‖ / ‖C - A‖ = 3 := by
  sorry

end NUMINAMATH_CALUDE_vector_ratio_theorem_l585_58594


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l585_58500

theorem diophantine_equation_solutions :
  ∀ x y : ℤ, 9*x*y - x^2 - 8*y^2 = 2005 ↔ 
    (x = 63 ∧ y = 58) ∨ (x = -63 ∧ y = -58) ∨ 
    (x = 459 ∧ y = 58) ∨ (x = -459 ∧ y = -58) := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l585_58500


namespace NUMINAMATH_CALUDE_robotics_club_neither_l585_58544

/-- The number of students in the robotics club who take neither computer science nor electronics -/
theorem robotics_club_neither (total : ℕ) (cs : ℕ) (elec : ℕ) (both : ℕ) 
  (h_total : total = 60)
  (h_cs : cs = 42)
  (h_elec : elec = 35)
  (h_both : both = 25) :
  total - (cs + elec - both) = 8 := by
  sorry

end NUMINAMATH_CALUDE_robotics_club_neither_l585_58544


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l585_58556

def A : Set ℝ := {x | x^2 ≤ 1}
def B : Set ℝ := {x | x > 0}

theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l585_58556
