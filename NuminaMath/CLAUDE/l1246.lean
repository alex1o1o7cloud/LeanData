import Mathlib

namespace fraction_change_l1246_124634

theorem fraction_change (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (0.6 * x) / (0.4 * y) = 1.5 * (x / y) := by
sorry

end fraction_change_l1246_124634


namespace second_month_bill_l1246_124665

/-- Represents Elvin's monthly telephone bill -/
structure TelephoneBill where
  callCharge : ℝ
  internetCharge : ℝ

/-- Calculates the total bill -/
def TelephoneBill.total (bill : TelephoneBill) : ℝ :=
  bill.callCharge + bill.internetCharge

theorem second_month_bill
  (firstMonth secondMonth : TelephoneBill)
  (h1 : firstMonth.total = 46)
  (h2 : secondMonth.total = 76)
  (h3 : secondMonth.callCharge = 2 * firstMonth.callCharge)
  (h4 : firstMonth.internetCharge = secondMonth.internetCharge) :
  secondMonth.total = 76 := by
  sorry

#check second_month_bill

end second_month_bill_l1246_124665


namespace magic_square_y_zero_l1246_124632

/-- Represents a 3x3 magic square -/
structure MagicSquare where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  g : ℤ
  h : ℤ
  i : ℤ
  is_magic : 
    a + b + c = d + e + f ∧
    a + b + c = g + h + i ∧
    a + b + c = a + d + g ∧
    a + b + c = b + e + h ∧
    a + b + c = c + f + i ∧
    a + b + c = a + e + i ∧
    a + b + c = c + e + g

/-- The theorem stating that y must be 0 in the given magic square configuration -/
theorem magic_square_y_zero (ms : MagicSquare) 
  (h1 : ms.a = y)
  (h2 : ms.b = 17)
  (h3 : ms.c = 124)
  (h4 : ms.d = 9) :
  y = 0 := by
  sorry


end magic_square_y_zero_l1246_124632


namespace perpendicular_vectors_dot_product_l1246_124645

/-- Given two vectors m and n in ℝ², where m = (2, 5) and n = (-5, t),
    if m is perpendicular to n, then (m + n) · (m - 2n) = -29 -/
theorem perpendicular_vectors_dot_product (t : ℝ) :
  let m : Fin 2 → ℝ := ![2, 5]
  let n : Fin 2 → ℝ := ![-5, t]
  (m • n = 0) →  -- m is perpendicular to n
  (m + n) • (m - 2 • n) = -29 :=
by sorry

end perpendicular_vectors_dot_product_l1246_124645


namespace travel_agency_comparison_l1246_124601

/-- Calculates the total cost for Travel Agency A -/
def costA (fullPrice : ℕ) (numStudents : ℕ) : ℕ :=
  fullPrice + numStudents * (fullPrice / 2)

/-- Calculates the total cost for Travel Agency B -/
def costB (fullPrice : ℕ) (numPeople : ℕ) : ℕ :=
  numPeople * (fullPrice * 60 / 100)

theorem travel_agency_comparison (fullPrice : ℕ) :
  (fullPrice = 240) →
  (costA fullPrice 5 < costB fullPrice 6) ∧
  (costB fullPrice 3 < costA fullPrice 2) := by
  sorry


end travel_agency_comparison_l1246_124601


namespace square_property_iff_4_or_100_l1246_124666

/-- The number of positive divisors of n -/
def d (n : ℕ+) : ℕ := sorry

/-- The condition for n to satisfy the square property -/
def is_square_property (n : ℕ+) : Prop :=
  ∃ k : ℕ, (n ^ (d n + 1) * (n + 21) ^ (d n) : ℕ) = k ^ 2

/-- The main theorem -/
theorem square_property_iff_4_or_100 :
  ∀ n : ℕ+, is_square_property n ↔ n = 4 ∨ n = 100 := by sorry

end square_property_iff_4_or_100_l1246_124666


namespace complex_sum_zero_l1246_124655

theorem complex_sum_zero (b a : ℝ) : 
  let z₁ : ℂ := 2 + b * Complex.I
  let z₂ : ℂ := a + Complex.I
  z₁ + z₂ = 0 → a + b * Complex.I = -2 - Complex.I :=
by
  sorry

end complex_sum_zero_l1246_124655


namespace stamp_solution_l1246_124603

def stamp_problem (one_cent two_cent five_cent eight_cent : ℕ) : Prop :=
  two_cent = (3 * one_cent) / 4 ∧
  five_cent = (3 * two_cent) / 4 ∧
  eight_cent = 5 ∧
  one_cent * 1 + two_cent * 2 + five_cent * 5 + eight_cent * 8 = 100000

theorem stamp_solution :
  ∃ (one_cent two_cent five_cent eight_cent : ℕ),
    stamp_problem one_cent two_cent five_cent eight_cent ∧
    one_cent = 18816 ∧
    two_cent = 14112 ∧
    five_cent = 10584 ∧
    eight_cent = 5 :=
  sorry

end stamp_solution_l1246_124603


namespace cube_division_equality_l1246_124682

def cube_edge_lengths : List ℕ := List.range 16

def group1 : List ℕ := [1, 4, 6, 7, 10, 11, 13, 16]
def group2 : List ℕ := [2, 3, 5, 8, 9, 12, 14, 15]

def volume (a : ℕ) : ℕ := a^3
def lateral_surface_area (a : ℕ) : ℕ := 4 * a^2
def edge_length (a : ℕ) : ℕ := 12 * a

theorem cube_division_equality :
  (group1.length = group2.length) ∧
  (group1.sum = group2.sum) ∧
  ((group1.map lateral_surface_area).sum = (group2.map lateral_surface_area).sum) ∧
  ((group1.map volume).sum = (group2.map volume).sum) ∧
  ((group1.map edge_length).sum = (group2.map edge_length).sum) :=
by sorry

end cube_division_equality_l1246_124682


namespace final_S_value_l1246_124661

def S : ℕ → ℕ
  | 0 => 1
  | n + 1 => S n + 2

theorem final_S_value : S 4 = 9 := by
  sorry

end final_S_value_l1246_124661


namespace total_stamps_l1246_124629

theorem total_stamps (stamps_AJ : ℕ) (stamps_KJ : ℕ) (stamps_CJ : ℕ) : 
  stamps_AJ = 370 →
  stamps_KJ = stamps_AJ / 2 →
  stamps_CJ = 2 * stamps_KJ + 5 →
  stamps_AJ + stamps_KJ + stamps_CJ = 930 :=
by
  sorry

end total_stamps_l1246_124629


namespace smallest_three_digit_congruence_l1246_124688

theorem smallest_three_digit_congruence :
  ∃ (n : ℕ), 
    n = 113 ∧ 
    100 ≤ n ∧ n < 1000 ∧ 
    (77 * n) % 385 = 231 % 385 ∧
    ∀ (m : ℕ), 100 ≤ m ∧ m < n → (77 * m) % 385 ≠ 231 % 385 := by
  sorry

end smallest_three_digit_congruence_l1246_124688


namespace investor_initial_investment_l1246_124687

/-- Compound interest calculation --/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proof of the investor's initial investment --/
theorem investor_initial_investment :
  let principal : ℝ := 7000
  let rate : ℝ := 0.10
  let time : ℕ := 2
  let final_amount : ℝ := 8470
  compound_interest principal rate time = final_amount := by
  sorry

end investor_initial_investment_l1246_124687


namespace eight_bead_bracelet_arrangements_l1246_124610

/-- The number of distinct arrangements of beads on a bracelet -/
def distinct_bracelet_arrangements (n : ℕ) : ℕ :=
  Nat.factorial n / (n * 2)

/-- Theorem: The number of distinct arrangements of 8 beads on a bracelet is 2520 -/
theorem eight_bead_bracelet_arrangements :
  distinct_bracelet_arrangements 8 = 2520 := by
  sorry

end eight_bead_bracelet_arrangements_l1246_124610


namespace hyperbola_equation_l1246_124672

/-- A hyperbola with right focus at (5, 0) and an asymptote with equation 2x - y = 0 
    has the standard equation x²/5 - y²/20 = 1 -/
theorem hyperbola_equation (x y : ℝ) : 
  let right_focus : ℝ × ℝ := (5, 0)
  let asymptote (x y : ℝ) : Prop := 2 * x - y = 0
  x^2 / 5 - y^2 / 20 = 1 :=
by sorry


end hyperbola_equation_l1246_124672


namespace expression_equalities_l1246_124616

theorem expression_equalities :
  (-2^3 = (-2)^3) ∧ 
  (2^3 ≠ 2 * 3) ∧ 
  (-(-2)^2 ≠ (-2)^2) ∧ 
  (-3^2 ≠ 3^2) := by
sorry

end expression_equalities_l1246_124616


namespace athlete_running_time_l1246_124677

/-- Proof that an athlete spends 35 minutes running given the conditions -/
theorem athlete_running_time 
  (calories_per_minute_running : ℕ) 
  (calories_per_minute_walking : ℕ)
  (total_calories_burned : ℕ)
  (total_time : ℕ)
  (h1 : calories_per_minute_running = 10)
  (h2 : calories_per_minute_walking = 4)
  (h3 : total_calories_burned = 450)
  (h4 : total_time = 60) :
  ∃ (running_time : ℕ), 
    running_time = 35 ∧ 
    running_time + (total_time - running_time) = total_time ∧
    calories_per_minute_running * running_time + 
    calories_per_minute_walking * (total_time - running_time) = total_calories_burned :=
by
  sorry


end athlete_running_time_l1246_124677


namespace fraction_of_percentages_l1246_124678

theorem fraction_of_percentages (P R M N : ℝ) 
  (hM : M = 0.4 * R)
  (hR : R = 0.25 * P)
  (hN : N = 0.6 * P)
  (hP : P ≠ 0) : 
  M / N = 1 / 6 := by
sorry

end fraction_of_percentages_l1246_124678


namespace equation_solution_l1246_124690

theorem equation_solution (p q : ℝ) (h : p^2*q = p*q + p^2) : 
  p = 0 ∨ (q ≠ 1 ∧ p = q / (q - 1)) := by sorry

end equation_solution_l1246_124690


namespace area_covered_is_56_l1246_124699

/-- The total area covered by five rectangular strips arranged in a specific pattern. -/
def total_area_covered (strip_length : ℝ) (strip_width : ℝ) (center_overlap : ℝ) : ℝ :=
  let single_strip_area := strip_length * strip_width
  let total_area_without_overlap := 5 * single_strip_area
  let center_overlap_area := 4 * (center_overlap * center_overlap)
  let fifth_strip_overlap_area := 2 * (center_overlap * center_overlap)
  total_area_without_overlap - (center_overlap_area + fifth_strip_overlap_area)

/-- Theorem stating that the total area covered by the strips is 56. -/
theorem area_covered_is_56 :
  total_area_covered 8 2 2 = 56 := by
  sorry

end area_covered_is_56_l1246_124699


namespace bug_distance_is_28_l1246_124617

def bug_crawl (start end1 end2 end3 : Int) : Int :=
  |end1 - start| + |end2 - end1| + |end3 - end2|

theorem bug_distance_is_28 :
  bug_crawl 3 (-4) 8 (-1) = 28 := by
  sorry

end bug_distance_is_28_l1246_124617


namespace exists_good_not_next_good_l1246_124685

/-- The digit sum of a natural number -/
def digitSum (n : ℕ) : ℕ := sorry

/-- The function f(n) = n - S(n) where S(n) is the digit sum of n -/
def f (n : ℕ) : ℕ := n - digitSum n

/-- f^k is f applied k times iteratively -/
def fIterate (k : ℕ) : ℕ → ℕ :=
  match k with
  | 0 => id
  | k+1 => f ∘ fIterate k

/-- A number x is k-good if there exists a y such that f^k(y) = x -/
def isGood (k : ℕ) (x : ℕ) : Prop :=
  ∃ y, fIterate k y = x

/-- The main theorem: for all n, there exists an x that is n-good but not (n+1)-good -/
theorem exists_good_not_next_good :
  ∀ n : ℕ, ∃ x : ℕ, isGood n x ∧ ¬isGood (n + 1) x := sorry

end exists_good_not_next_good_l1246_124685


namespace tangent_circle_radius_l1246_124628

/-- A configuration of tangents to a circle -/
structure TangentConfiguration where
  r : ℝ  -- radius of the circle
  AB : ℝ  -- length of tangent AB
  CD : ℝ  -- length of tangent CD
  EF : ℝ  -- length of EF

/-- The theorem stating the radius of the circle given the tangent configuration -/
theorem tangent_circle_radius (config : TangentConfiguration) 
  (h1 : config.AB = 12)
  (h2 : config.CD = 20)
  (h3 : config.EF = 8) :
  config.r = 6 := by
  sorry

#check tangent_circle_radius

end tangent_circle_radius_l1246_124628


namespace habitable_land_area_l1246_124681

/-- Calculates the area of habitable land in a rectangular field with a circular pond. -/
theorem habitable_land_area (length width diagonal pond_radius : ℝ) 
  (h_length : length = 23)
  (h_diagonal : diagonal = 33)
  (h_width : width^2 = diagonal^2 - length^2)
  (h_pond_radius : pond_radius = 3) : 
  ∃ (area : ℝ), abs (area - 515.91) < 0.01 ∧ 
  area = length * width - π * pond_radius^2 := by
sorry

end habitable_land_area_l1246_124681


namespace bobby_candy_theorem_l1246_124664

def candy_problem (initial : ℕ) (first_eaten : ℕ) (second_eaten : ℕ) : ℕ :=
  initial - first_eaten - second_eaten

theorem bobby_candy_theorem :
  candy_problem 21 5 9 = 7 := by
  sorry

end bobby_candy_theorem_l1246_124664


namespace same_terminal_side_l1246_124671

/-- Proves that given an angle of -3π/10 radians, 306° has the same terminal side when converted to degrees -/
theorem same_terminal_side : ∃ (β : ℝ), β = 306 ∧ ∃ (k : ℤ), β = (-3/10 * π) * (180/π) + 360 * k :=
sorry

end same_terminal_side_l1246_124671


namespace equation_solution_l1246_124621

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * f y - y * f x) = f (x * y) - x * y

/-- The theorem stating that functions satisfying the equation are either the identity function or the absolute value function -/
theorem equation_solution (f : ℝ → ℝ) (h : SatisfiesEquation f) :
    (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = |x|) := by
  sorry


end equation_solution_l1246_124621


namespace janet_waiting_time_l1246_124696

/-- Proves that the waiting time for Janet is 3 hours given the conditions of the problem -/
theorem janet_waiting_time (lake_width : ℝ) (speedboat_speed : ℝ) (sailboat_speed : ℝ)
  (h1 : lake_width = 60)
  (h2 : speedboat_speed = 30)
  (h3 : sailboat_speed = 12) :
  sailboat_speed * (lake_width / speedboat_speed) - lake_width = 3 * speedboat_speed := by
  sorry


end janet_waiting_time_l1246_124696


namespace mutually_exclusive_but_not_converse_l1246_124695

/-- Represents the outcome of rolling a single die -/
inductive DieOutcome
  | One
  | Two
  | Three
  | Four
  | Five
  | Six

/-- Represents the outcome of rolling two dice -/
def TwoDiceOutcome := DieOutcome × DieOutcome

/-- Checks if a die outcome is odd -/
def isOdd (outcome : DieOutcome) : Bool :=
  match outcome with
  | DieOutcome.One => true
  | DieOutcome.Three => true
  | DieOutcome.Five => true
  | _ => false

/-- Event: Exactly one odd number -/
def exactlyOneOdd (outcome : TwoDiceOutcome) : Prop :=
  (isOdd outcome.1 && !isOdd outcome.2) || (!isOdd outcome.1 && isOdd outcome.2)

/-- Event: Exactly two odd numbers -/
def exactlyTwoOdd (outcome : TwoDiceOutcome) : Prop :=
  isOdd outcome.1 && isOdd outcome.2

/-- The sample space of all possible outcomes when rolling two fair six-sided dice -/
def sampleSpace : Set TwoDiceOutcome := sorry

theorem mutually_exclusive_but_not_converse :
  (∀ (outcome : TwoDiceOutcome), ¬(exactlyOneOdd outcome ∧ exactlyTwoOdd outcome)) ∧
  (∃ (outcome : TwoDiceOutcome), ¬exactlyOneOdd outcome ∧ ¬exactlyTwoOdd outcome) :=
sorry

end mutually_exclusive_but_not_converse_l1246_124695


namespace smallest_k_for_sum_4n_plus_1_l1246_124673

/-- Given a positive integer n, M is the set of integers from 1 to 2n -/
def M (n : ℕ+) : Finset ℕ := Finset.range (2 * n) \ {0}

/-- A function that checks if a subset of M contains 4 distinct elements summing to 4n + 1 -/
def has_sum_4n_plus_1 (n : ℕ+) (S : Finset ℕ) : Prop :=
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    a + b + c + d = 4 * n + 1

theorem smallest_k_for_sum_4n_plus_1 (n : ℕ+) :
  (∀ (S : Finset ℕ), S ⊆ M n → S.card = n + 3 → has_sum_4n_plus_1 n S) ∧
  (∃ (T : Finset ℕ), T ⊆ M n ∧ T.card = n + 2 ∧ ¬has_sum_4n_plus_1 n T) :=
sorry

end smallest_k_for_sum_4n_plus_1_l1246_124673


namespace max_value_sin_cos_l1246_124649

theorem max_value_sin_cos (x y : ℝ) (h : Real.sin x + Real.sin y = 1/3) :
  (∀ z w : ℝ, Real.sin z + Real.sin w = 1/3 → 
    Real.sin y - Real.cos x ^ 2 ≤ Real.sin w - Real.cos z ^ 2) →
  Real.sin y - Real.cos x ^ 2 = 4/9 :=
sorry

end max_value_sin_cos_l1246_124649


namespace unique_solution_condition_l1246_124639

theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, Real.sqrt (x + 1/2 + Real.sqrt (x + 1/4)) + x = a) ↔ a ≥ 1/4 := by
sorry

end unique_solution_condition_l1246_124639


namespace oh_squared_equals_526_l1246_124630

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the circumcenter O and orthocenter H
def circumcenter (t : Triangle) : ℝ × ℝ := sorry
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the circumradius R
def circumradius (t : Triangle) : ℝ := sorry

-- Define side lengths a, b, c
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ := sorry

theorem oh_squared_equals_526 (t : Triangle) :
  let O := circumcenter t
  let H := orthocenter t
  let R := circumradius t
  let (a, b, c) := side_lengths t
  R = 8 →
  2 * a^2 + b^2 + c^2 = 50 →
  (O.1 - H.1)^2 + (O.2 - H.2)^2 = 526 := by
  sorry

end oh_squared_equals_526_l1246_124630


namespace quadratic_real_roots_condition_l1246_124698

/-- 
For a quadratic equation (k-1)x^2 + 4x + 2 = 0 to have real roots,
k must satisfy the condition k ≤ 3 and k ≠ 1.
-/
theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 4 * x + 2 = 0) ↔ (k ≤ 3 ∧ k ≠ 1) :=
sorry

end quadratic_real_roots_condition_l1246_124698


namespace solve_linear_equations_l1246_124662

theorem solve_linear_equations :
  (∃ y : ℚ, 8 * y - 4 * (3 * y + 2) = 6 ∧ y = -7/2) ∧
  (∃ x : ℚ, 2 - (x + 2) / 3 = x - (x - 1) / 6 ∧ x = 1) :=
by sorry

end solve_linear_equations_l1246_124662


namespace expand_a_plus_one_a_plus_two_expand_three_a_plus_b_three_a_minus_b_square_of_101_expand_and_simplify_l1246_124686

-- 1. Prove that (a+1)(a+2) = a^2 + 3a + 2
theorem expand_a_plus_one_a_plus_two (a : ℝ) : 
  (a + 1) * (a + 2) = a^2 + 3*a + 2 := by sorry

-- 2. Prove that (3a+b)(3a-b) = 9a^2 - b^2
theorem expand_three_a_plus_b_three_a_minus_b (a b : ℝ) : 
  (3*a + b) * (3*a - b) = 9*a^2 - b^2 := by sorry

-- 3. Prove that 101^2 = 10201
theorem square_of_101 : 
  (101 : ℕ)^2 = 10201 := by sorry

-- 4. Prove that (y+2)(y-2)-(y-1)(y+5) = -4y + 1
theorem expand_and_simplify (y : ℝ) : 
  (y + 2) * (y - 2) - (y - 1) * (y + 5) = -4*y + 1 := by sorry

end expand_a_plus_one_a_plus_two_expand_three_a_plus_b_three_a_minus_b_square_of_101_expand_and_simplify_l1246_124686


namespace problem_ratio_is_three_to_one_l1246_124694

/-- The number of math problems composed by Bill -/
def bill_problems : ℕ := 20

/-- The number of math problems composed by Ryan -/
def ryan_problems : ℕ := 2 * bill_problems

/-- The number of different types of math problems -/
def problem_types : ℕ := 4

/-- The number of problems Frank composes for each type -/
def frank_problems_per_type : ℕ := 30

/-- The total number of problems Frank composes -/
def frank_problems : ℕ := frank_problems_per_type * problem_types

/-- The ratio of problems Frank composes to problems Ryan composes -/
def problem_ratio : ℚ := frank_problems / ryan_problems

theorem problem_ratio_is_three_to_one : problem_ratio = 3 := by
  sorry

end problem_ratio_is_three_to_one_l1246_124694


namespace belinda_pages_per_day_l1246_124658

/-- Given that Janet reads 80 pages a day and 2100 more pages than Belinda in 6 weeks,
    prove that Belinda reads 30 pages a day. -/
theorem belinda_pages_per_day :
  let janet_pages_per_day : ℕ := 80
  let weeks : ℕ := 6
  let days_in_week : ℕ := 7
  let extra_pages : ℕ := 2100
  let belinda_pages_per_day : ℕ := 30
  janet_pages_per_day * (weeks * days_in_week) = 
    belinda_pages_per_day * (weeks * days_in_week) + extra_pages :=
by
  sorry

#check belinda_pages_per_day

end belinda_pages_per_day_l1246_124658


namespace min_distance_to_perpendicular_bisector_l1246_124637

open Complex

theorem min_distance_to_perpendicular_bisector (z : ℂ) :
  (abs z = abs (z + 2 + 2*I)) →
  (∃ (min_val : ℝ), ∀ (w : ℂ), abs w = abs (w + 2 + 2*I) → abs (w - 1 + I) ≥ min_val) ∧
  (∃ (z₀ : ℂ), abs z₀ = abs (z₀ + 2 + 2*I) ∧ abs (z₀ - 1 + I) = Real.sqrt 2) :=
by sorry

end min_distance_to_perpendicular_bisector_l1246_124637


namespace minimum_percentage_bad_work_l1246_124609

theorem minimum_percentage_bad_work (total_works : ℝ) (h_total_positive : total_works > 0) :
  let bad_works := 0.2 * total_works
  let good_works := 0.8 * total_works
  let misclassified_good := 0.1 * good_works
  let misclassified_bad := 0.1 * bad_works
  let rechecked_works := bad_works - misclassified_bad + misclassified_good
  let actual_bad_rechecked := bad_works - misclassified_bad
  ⌊(actual_bad_rechecked / rechecked_works * 100)⌋ = 69 :=
by sorry

end minimum_percentage_bad_work_l1246_124609


namespace smallest_number_with_two_thirds_prob_l1246_124618

/-- The smallest number that can be drawn in the lottery -/
def minNumber : ℕ := 1

/-- The largest number in the first range of the lottery -/
def rangeEnd : ℕ := 15

/-- The probability of drawing a number between minNumber and rangeEnd, inclusive -/
def probFirstRange : ℚ := 1/3

/-- The probability of drawing a number less than or equal to rangeEnd -/
def probUpToRangeEnd : ℚ := 2/3

/-- The probability of drawing a number larger than the target number -/
def probLargerThanTarget : ℚ := 2/3

theorem smallest_number_with_two_thirds_prob :
  ∃ N : ℕ, N = rangeEnd + 1 ∧
    (∀ k : ℕ, k > N → (probFirstRange + (k - rangeEnd : ℚ) * probFirstRange = probLargerThanTarget)) ∧
    (∀ m : ℕ, m < N → (probFirstRange + (m - rangeEnd : ℚ) * probFirstRange < probLargerThanTarget)) :=
sorry

end smallest_number_with_two_thirds_prob_l1246_124618


namespace abc_inequality_l1246_124683

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
  sorry

end abc_inequality_l1246_124683


namespace range_of_sum_l1246_124602

theorem range_of_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : a + b + 1/a + 1/b = 5) : 1 ≤ a + b ∧ a + b ≤ 4 := by
  sorry

end range_of_sum_l1246_124602


namespace sequence_properties_l1246_124642

def a (n : ℕ) : ℤ := n^2 - 7*n + 6

theorem sequence_properties :
  (a 4 = -6) ∧
  (a 16 = 150) ∧
  (∀ n : ℕ, n ≥ 7 → a n > 0) := by
sorry

end sequence_properties_l1246_124642


namespace no_perfect_square_2007_plus_4n_l1246_124625

theorem no_perfect_square_2007_plus_4n :
  ¬ ∃ (n : ℕ), ∃ (k : ℕ), 2007 + 4^n = k^2 := by
sorry

end no_perfect_square_2007_plus_4n_l1246_124625


namespace r_value_when_n_is_3_l1246_124622

theorem r_value_when_n_is_3 (n : ℕ) (s r : ℕ) 
  (h1 : s = 3^n - 1) 
  (h2 : r = 3^s + s) 
  (h3 : n = 3) : 
  r = 3^26 + 26 := by
  sorry

end r_value_when_n_is_3_l1246_124622


namespace a_2n_is_perfect_square_a_2n_specific_form_l1246_124613

/-- The number of natural numbers with digit sum n, using only digits 1, 3, and 4 -/
def a (n : ℕ) : ℕ :=
  sorry

/-- The main theorem: a₂ₙ is a perfect square for all natural numbers n -/
theorem a_2n_is_perfect_square (n : ℕ) : ∃ k : ℕ, a (2 * n) = k ^ 2 := by
  sorry

/-- The specific form of a₂ₙ as (aₙ + aₙ₋₂)² -/
theorem a_2n_specific_form (n : ℕ) : a (2 * n) = (a n + a (n - 2)) ^ 2 := by
  sorry

end a_2n_is_perfect_square_a_2n_specific_form_l1246_124613


namespace men_entered_room_l1246_124644

theorem men_entered_room (initial_men initial_women : ℕ) 
  (men_entered : ℕ) (h1 : initial_men * 5 = initial_women * 4) 
  (h2 : initial_men + men_entered = 14) 
  (h3 : 2 * (initial_women - 3) = 24) : men_entered = 2 := by
  sorry

end men_entered_room_l1246_124644


namespace quadratic_inequality_condition_l1246_124669

theorem quadratic_inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) → (0 < a ∧ a < 1) :=
by sorry

end quadratic_inequality_condition_l1246_124669


namespace smallest_m_for_cube_sum_inequality_l1246_124646

theorem smallest_m_for_cube_sum_inequality :
  ∃ (m : ℝ), m = 27 ∧
  (∀ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 →
    m * (a^3 + b^3 + c^3) ≥ 6 * (a^2 + b^2 + c^2) + 1) ∧
  (∀ (m' : ℝ), m' < m →
    ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 1 ∧
      m' * (a^3 + b^3 + c^3) < 6 * (a^2 + b^2 + c^2) + 1) :=
by sorry

end smallest_m_for_cube_sum_inequality_l1246_124646


namespace smallest_prime_8_less_than_perfect_square_l1246_124640

/-- A number is a perfect square if it's the square of some integer. -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- A number is prime if it's greater than 1 and its only divisors are 1 and itself. -/
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem smallest_prime_8_less_than_perfect_square :
  ∃ (n : ℕ), is_prime n ∧ (∃ (m : ℕ), is_perfect_square m ∧ n = m - 8) ∧
  (∀ (k : ℕ), k < n → ¬(is_prime k ∧ ∃ (m : ℕ), is_perfect_square m ∧ k = m - 8)) :=
by
  -- The proof goes here
  sorry

end smallest_prime_8_less_than_perfect_square_l1246_124640


namespace scientific_notation_proof_l1246_124653

/-- Proves that 470,000,000 is equal to 4.7 × 10^8 in scientific notation -/
theorem scientific_notation_proof :
  (470000000 : ℝ) = 4.7 * (10 ^ 8) := by
  sorry

end scientific_notation_proof_l1246_124653


namespace probability_one_red_one_white_l1246_124606

/-- The probability of selecting one red ball and one white ball from a bag -/
theorem probability_one_red_one_white (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) :
  total_balls = red_balls + white_balls →
  red_balls = 3 →
  white_balls = 2 →
  (red_balls.choose 1 * white_balls.choose 1 : ℚ) / total_balls.choose 2 = 3 / 5 :=
by sorry

end probability_one_red_one_white_l1246_124606


namespace mary_has_ten_marbles_l1246_124631

/-- The number of blue marbles Dan has -/
def dan_marbles : ℕ := 5

/-- The factor by which Mary has more marbles than Dan -/
def mary_factor : ℕ := 2

/-- The number of blue marbles Mary has -/
def mary_marbles : ℕ := mary_factor * dan_marbles

theorem mary_has_ten_marbles : mary_marbles = 10 := by
  sorry

end mary_has_ten_marbles_l1246_124631


namespace real_roots_condition_l1246_124663

theorem real_roots_condition (p q : ℝ) : 
  (∃ x : ℝ, x^4 + p*x^2 + q = 0) → 65*p^2 ≥ 4*q ∧ 
  ¬(∀ p q : ℝ, 65*p^2 ≥ 4*q → ∃ x : ℝ, x^4 + p*x^2 + q = 0) :=
by sorry

end real_roots_condition_l1246_124663


namespace james_golden_retrievers_l1246_124611

/-- Represents the number of dogs James has for each breed -/
structure DogCounts where
  huskies : Nat
  pitbulls : Nat
  golden_retrievers : Nat

/-- Represents the number of pups each breed has -/
structure PupCounts where
  husky_pups : Nat
  pitbull_pups : Nat
  golden_retriever_pups : Nat

/-- The problem statement -/
theorem james_golden_retrievers (dogs : DogCounts) (pups : PupCounts) : 
  dogs.huskies = 5 →
  dogs.pitbulls = 2 →
  pups.husky_pups = 3 →
  pups.pitbull_pups = 3 →
  pups.golden_retriever_pups = pups.husky_pups + 2 →
  dogs.huskies * pups.husky_pups + 
  dogs.pitbulls * pups.pitbull_pups + 
  dogs.golden_retrievers * pups.golden_retriever_pups = 
  (dogs.huskies + dogs.pitbulls + dogs.golden_retrievers) + 30 →
  dogs.golden_retrievers = 4 := by
sorry

end james_golden_retrievers_l1246_124611


namespace circle_radius_in_rectangle_l1246_124615

/-- A rectangle with length 10 and width 6 -/
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (length_eq : length = 10)
  (width_eq : width = 6)

/-- A circle passing through two vertices of the rectangle and tangent to the opposite side -/
structure Circle (rect : Rectangle) :=
  (radius : ℝ)
  (passes_through_vertices : Bool)
  (tangent_to_opposite_side : Bool)

/-- The theorem stating that the radius of the circle is 3 -/
theorem circle_radius_in_rectangle (rect : Rectangle) (circ : Circle rect) :
  circ.passes_through_vertices ∧ circ.tangent_to_opposite_side → circ.radius = 3 := by
  sorry

end circle_radius_in_rectangle_l1246_124615


namespace prime_solution_equation_l1246_124689

theorem prime_solution_equation : 
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    p^2 - 6*p*q + q^2 + 3*q - 1 = 0 → 
    (p = 17 ∧ q = 3) := by
  sorry

end prime_solution_equation_l1246_124689


namespace not_in_range_iff_a_in_interval_l1246_124691

/-- The function g(x) defined as x^2 + ax + 3 -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 3

/-- Theorem stating that -3 is not in the range of g(x) if and only if a is in the open interval (-√24, √24) -/
theorem not_in_range_iff_a_in_interval (a : ℝ) :
  (∀ x : ℝ, g a x ≠ -3) ↔ a ∈ Set.Ioo (-Real.sqrt 24) (Real.sqrt 24) :=
sorry

end not_in_range_iff_a_in_interval_l1246_124691


namespace raise_calculation_l1246_124660

-- Define the original weekly earnings
def original_earnings : ℚ := 60

-- Define the percentage increase
def percentage_increase : ℚ := 33.33 / 100

-- Define the new weekly earnings
def new_earnings : ℚ := original_earnings * (1 + percentage_increase)

-- Theorem to prove
theorem raise_calculation :
  new_earnings = 80 := by sorry

end raise_calculation_l1246_124660


namespace complex_fraction_simplification_l1246_124693

theorem complex_fraction_simplification :
  let z₁ : ℂ := 3 + 5*I
  let z₂ : ℂ := -2 + 3*I
  z₁ / z₂ = 9/13 - (19/13)*I := by
sorry

end complex_fraction_simplification_l1246_124693


namespace softball_team_ratio_l1246_124638

/-- Proves that for a co-ed softball team with 6 more women than men and 24 total players, 
    the ratio of men to women is 3:5 -/
theorem softball_team_ratio : 
  ∀ (men women : ℕ), 
  women = men + 6 →
  men + women = 24 →
  (men : ℚ) / women = 3 / 5 := by
sorry

end softball_team_ratio_l1246_124638


namespace soda_quarters_l1246_124651

/-- Represents the number of quarters in a dollar -/
def quarters_per_dollar : ℕ := 4

/-- Represents the total amount paid in dollars -/
def total_paid : ℕ := 4

/-- Represents the number of quarters paid for chips -/
def quarters_for_chips : ℕ := 4

/-- Calculates the number of quarters paid for soda -/
def quarters_for_soda : ℕ := (total_paid - quarters_for_chips / quarters_per_dollar) * quarters_per_dollar

theorem soda_quarters : quarters_for_soda = 12 := by
  sorry

end soda_quarters_l1246_124651


namespace inverse_of_complex_expression_l1246_124636

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem inverse_of_complex_expression :
  i ^ 2 = -1 → (3 * i - 3 * i⁻¹)⁻¹ = -i / 6 := by
  sorry

end inverse_of_complex_expression_l1246_124636


namespace normal_equation_for_given_conditions_l1246_124684

def normal_equation (p : ℝ) (α : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x * Real.cos α + y * Real.sin α - p = 0

theorem normal_equation_for_given_conditions :
  let p : ℝ := 3
  let α₁ : ℝ := π / 4  -- 45°
  let α₂ : ℝ := 7 * π / 4  -- 315°
  (∀ x y, normal_equation p α₁ x y ↔ Real.sqrt 2 / 2 * x + Real.sqrt 2 / 2 * y - 3 = 0) ∧
  (∀ x y, normal_equation p α₂ x y ↔ Real.sqrt 2 / 2 * x - Real.sqrt 2 / 2 * y - 3 = 0) :=
by sorry

end normal_equation_for_given_conditions_l1246_124684


namespace evaluate_complex_expression_l1246_124626

theorem evaluate_complex_expression :
  let M := (Real.sqrt (Real.sqrt 7 + 3) + Real.sqrt (Real.sqrt 7 - 3)) / Real.sqrt (Real.sqrt 7 + 2) + Real.sqrt (5 - 2 * Real.sqrt 6)
  M = 2 * Real.sqrt 2 - 1 := by
  sorry

end evaluate_complex_expression_l1246_124626


namespace quadratic_function_properties_l1246_124620

/-- A quadratic function passing through (-2, 1) with exactly one root -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- The function g derived from f -/
def g (a b k : ℝ) (x : ℝ) : ℝ := f a b x - k * x

/-- The theorem stating the properties of f and g -/
theorem quadratic_function_properties (a b : ℝ) (h_a : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, 1 ≤ x₁ ∧ x₁ < x₂ ∧ f a b (-2) = 1) →
  (∃! x : ℝ, f a b x = 0) →
  (∀ x : ℝ, f a b x = (x + 1)^2) ∧
  (∀ k : ℝ, (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → Monotone (g a b k)) ↔ k ≤ 0 ∨ k ≥ 6) :=
sorry

end quadratic_function_properties_l1246_124620


namespace solution_exists_l1246_124697

theorem solution_exists (x y b : ℝ) : 
  (4 * x + 2 * y = b) →
  (3 * x + 7 * y = 3 * b) →
  (x = -1) →
  b = -22 :=
by sorry

end solution_exists_l1246_124697


namespace hyperbola_point_distance_to_x_axis_l1246_124612

/-- The distance from a point on a hyperbola to the x-axis, given specific conditions -/
theorem hyperbola_point_distance_to_x_axis 
  (P : ℝ × ℝ) 
  (F₁ F₂ : ℝ × ℝ) 
  (h_hyperbola : (P.1^2 / 16) - (P.2^2 / 9) = 1) 
  (h_on_hyperbola : P ∈ {p : ℝ × ℝ | (p.1^2 / 16) - (p.2^2 / 9) = 1}) 
  (h_focal_points : F₁ ∈ {f : ℝ × ℝ | (f.1^2 / 16) - (f.2^2 / 9) = 1} ∧ 
                    F₂ ∈ {f : ℝ × ℝ | (f.1^2 / 16) - (f.2^2 / 9) = 1}) 
  (h_perpendicular : (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0) : 
  |P.2| = 9/5 := by
sorry

end hyperbola_point_distance_to_x_axis_l1246_124612


namespace function_value_at_three_l1246_124623

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x, f x + 3 * f (1 - x) = 4 * x^2

theorem function_value_at_three 
  (f : ℝ → ℝ) 
  (h : FunctionalEquation f) : 
  f 3 = 3/2 := by
  sorry

end function_value_at_three_l1246_124623


namespace deck_card_count_l1246_124680

theorem deck_card_count : ∀ (r n : ℕ), 
  (n = 2 * r) →                           -- Initially, black cards are twice red cards
  (n + 4 = 3 * r) →                       -- After adding 4 black cards, black is triple red
  (r + n = 12) :=                         -- Initial total number of cards is 12
by
  sorry

end deck_card_count_l1246_124680


namespace stability_comparison_l1246_124679

/-- Represents a student's scores in the competition -/
structure StudentScores where
  scores : List ℝ
  mean : ℝ
  variance : ℝ

/-- The competition has 5 rounds -/
def num_rounds : ℕ := 5

/-- Stability comparison of two students' scores -/
def more_stable (a b : StudentScores) : Prop :=
  a.variance < b.variance

theorem stability_comparison (a b : StudentScores) 
  (h1 : a.scores.length = num_rounds)
  (h2 : b.scores.length = num_rounds)
  (h3 : a.mean = 90)
  (h4 : b.mean = 90)
  (h5 : a.variance = 15)
  (h6 : b.variance = 3) :
  more_stable b a :=
sorry

end stability_comparison_l1246_124679


namespace prob_yellow_twice_is_one_ninth_l1246_124643

/-- A fair 12-sided die with 4 yellow faces -/
structure YellowDie :=
  (sides : ℕ)
  (yellow_faces : ℕ)
  (is_fair : sides = 12)
  (yellow_count : yellow_faces = 4)

/-- The probability of rolling yellow twice with a YellowDie -/
def prob_yellow_twice (d : YellowDie) : ℚ :=
  (d.yellow_faces : ℚ) / (d.sides : ℚ) * (d.yellow_faces : ℚ) / (d.sides : ℚ)

/-- Theorem: The probability of rolling yellow twice with a YellowDie is 1/9 -/
theorem prob_yellow_twice_is_one_ninth (d : YellowDie) :
  prob_yellow_twice d = 1 / 9 := by
  sorry

end prob_yellow_twice_is_one_ninth_l1246_124643


namespace sum_of_digits_five_pow_eq_two_pow_l1246_124670

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The only natural number n for which the sum of digits of 5^n equals 2^n is 3 -/
theorem sum_of_digits_five_pow_eq_two_pow :
  ∃! n : ℕ, sum_of_digits (5^n) = 2^n ∧ n = 3 := by sorry

end sum_of_digits_five_pow_eq_two_pow_l1246_124670


namespace expression_value_at_three_l1246_124654

theorem expression_value_at_three :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 4 * x + 2
  f 3 = 17 := by
  sorry

end expression_value_at_three_l1246_124654


namespace hot_dogs_remainder_l1246_124604

theorem hot_dogs_remainder : 25197629 % 4 = 1 := by
  sorry

end hot_dogs_remainder_l1246_124604


namespace angle_is_rational_multiple_of_360_degrees_l1246_124648

/-- A point moving on two intersecting lines -/
structure JumpingPoint where
  angle : ℝ  -- The angle between the lines in radians
  position : ℕ × Bool  -- The position as (jump number, which line)

/-- The condition that the point returns to its starting position -/
def returnsToStart (jp : JumpingPoint) (n : ℕ) : Prop :=
  ∃ k : ℕ, n * jp.angle = k * (2 * Real.pi)

/-- The main theorem -/
theorem angle_is_rational_multiple_of_360_degrees 
  (jp : JumpingPoint) 
  (returns : ∃ n : ℕ, returnsToStart jp n) 
  (h_angle : 0 < jp.angle ∧ jp.angle < 2 * Real.pi) :
  ∃ q : ℚ, jp.angle = q * (2 * Real.pi) :=
sorry

end angle_is_rational_multiple_of_360_degrees_l1246_124648


namespace parallel_vectors_theorem_l1246_124667

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def Parallel (v w : V) : Prop := ∃ (c : ℝ), v = c • w

theorem parallel_vectors_theorem (e₁ e₂ a b : V) (m : ℝ) 
  (h_non_collinear : ¬ Parallel e₁ e₂)
  (h_a : a = 2 • e₁ - e₂)
  (h_b : b = m • e₁ + 3 • e₂)
  (h_parallel : Parallel a b) :
  m = -6 := by sorry

end parallel_vectors_theorem_l1246_124667


namespace manufacturing_cost_of_shoe_l1246_124607

/-- The manufacturing cost of a shoe given transportation cost, selling price, and profit margin. -/
theorem manufacturing_cost_of_shoe (transportation_cost : ℚ) (selling_price : ℚ) (profit_margin : ℚ) :
  transportation_cost = 5 →
  selling_price = 234 →
  profit_margin = 1/5 →
  ∃ (manufacturing_cost : ℚ), 
    selling_price = (manufacturing_cost + transportation_cost) * (1 + profit_margin) ∧
    manufacturing_cost = 190 :=
by sorry

end manufacturing_cost_of_shoe_l1246_124607


namespace perpendicular_line_equation_l1246_124627

/-- Given a line L1 with equation 4x + 5y - 8 = 0 and a point A(3,2),
    the line L2 passing through A and perpendicular to L1 has equation 4y - 5x + 7 = 0 -/
theorem perpendicular_line_equation (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 4 * x + 5 * y - 8 = 0
  let A : ℝ × ℝ := (3, 2)
  let L2 : ℝ → ℝ → Prop := λ x y => 4 * y - 5 * x + 7 = 0
  (∀ x y, L2 x y ↔ (y - A.2) = -(4/5) * (x - A.1)) ∧
  (∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L1 x₂ y₂ → (y₂ - y₁) / (x₂ - x₁) = 4/5) →
  L2 A.1 A.2 ∧ ∀ x₁ y₁ x₂ y₂, L1 x₁ y₁ → L2 x₂ y₂ → (y₂ - y₁) / (x₂ - x₁) = -5/4 :=
by
  sorry


end perpendicular_line_equation_l1246_124627


namespace pure_imaginary_square_l1246_124674

theorem pure_imaginary_square (a : ℝ) (z : ℂ) : 
  z = a + (1 + a) * Complex.I → 
  (∃ b : ℝ, z = b * Complex.I) → 
  z^2 = -1 := by
sorry

end pure_imaginary_square_l1246_124674


namespace stream_speed_l1246_124676

/-- Proves that given a boat with a speed of 8 kmph in standing water,
    traveling a round trip of 420 km (210 km each way) in 56 hours,
    the speed of the stream is 2 kmph. -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) :
  boat_speed = 8 →
  distance = 210 →
  total_time = 56 →
  ∃ (stream_speed : ℝ),
    stream_speed = 2 ∧
    (distance / (boat_speed - stream_speed) + distance / (boat_speed + stream_speed) = total_time) :=
by sorry

end stream_speed_l1246_124676


namespace profit_growth_rate_l1246_124600

theorem profit_growth_rate (initial_profit target_profit : ℝ) (growth_rate : ℝ) (months : ℕ) :
  initial_profit * (1 + growth_rate / 100) ^ months = target_profit →
  growth_rate = 25 :=
by
  intro h
  -- Proof goes here
  sorry

#check profit_growth_rate 1.6 2.5 25 2

end profit_growth_rate_l1246_124600


namespace roots_equation_problem_l1246_124668

theorem roots_equation_problem (x₁ x₂ m : ℝ) :
  (2 * x₁^2 - 3 * x₁ + m = 0) →
  (2 * x₂^2 - 3 * x₂ + m = 0) →
  (8 * x₁ - 2 * x₂ = 7) →
  m = 1 := by
sorry

end roots_equation_problem_l1246_124668


namespace rational_function_value_l1246_124657

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_linear : ∃ a b : ℝ, ∀ x, p x = a * x + b
  q_quadratic : ∃ a b c : ℝ, ∀ x, q x = a * x^2 + b * x + c
  q_roots : q (-4) = 0 ∧ q 1 = 0
  point_zero : p 0 = 0 ∧ q 0 ≠ 0
  point_neg_one : p (-1) / q (-1) = -2

/-- The main theorem -/
theorem rational_function_value (f : RationalFunction) : f.p 2 / f.q 2 = 4 := by
  sorry

end rational_function_value_l1246_124657


namespace curve_self_intersection_l1246_124635

/-- The x-coordinate of a point on the curve as a function of t -/
def x (t : ℝ) : ℝ := t^2 - 4

/-- The y-coordinate of a point on the curve as a function of t -/
def y (t : ℝ) : ℝ := t^3 - 6*t + 3

/-- Theorem stating that (2,3) is the only self-intersection point of the curve -/
theorem curve_self_intersection :
  ∃! p : ℝ × ℝ, p.1 = 2 ∧ p.2 = 3 ∧
  ∃ a b : ℝ, a ≠ b ∧ 
    x a = x b ∧ y a = y b ∧
    x a = p.1 ∧ y a = p.2 :=
sorry

end curve_self_intersection_l1246_124635


namespace chase_blue_jays_count_l1246_124614

/-- The number of blue jays Chase saw -/
def chase_blue_jays : ℕ := 3

/-- The number of robins Gabrielle saw -/
def gabrielle_robins : ℕ := 5

/-- The number of cardinals Gabrielle saw -/
def gabrielle_cardinals : ℕ := 4

/-- The number of blue jays Gabrielle saw -/
def gabrielle_blue_jays : ℕ := 3

/-- The number of robins Chase saw -/
def chase_robins : ℕ := 2

/-- The number of cardinals Chase saw -/
def chase_cardinals : ℕ := 5

/-- The percentage more birds Gabrielle saw compared to Chase -/
def percentage_difference : ℚ := 1/5

theorem chase_blue_jays_count :
  (gabrielle_robins + gabrielle_cardinals + gabrielle_blue_jays : ℚ) =
  (chase_robins + chase_cardinals + chase_blue_jays : ℚ) * (1 + percentage_difference) :=
by sorry

end chase_blue_jays_count_l1246_124614


namespace range_of_a_l1246_124605

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : A ∩ B a = B a → -2 ≤ a ∧ a ≤ 1 := by
  sorry

end range_of_a_l1246_124605


namespace susanna_purchase_l1246_124641

/-- The cost of each item in pounds and pence -/
structure ItemCost where
  pounds : ℕ
  pence : Fin 100
  pence_eq : pence = 99

/-- The total amount spent by Susanna in pence -/
def total_spent : ℕ := 65 * 100 + 76

/-- The number of items Susanna bought -/
def items_bought : ℕ := 24

theorem susanna_purchase :
  ∀ (cost : ItemCost),
  (cost.pounds * 100 + cost.pence) * items_bought = total_spent :=
sorry

end susanna_purchase_l1246_124641


namespace movie_ticket_price_is_30_l1246_124647

/-- The price of a movie ticket -/
def movie_ticket_price : ℝ := sorry

/-- The price of a football game ticket -/
def football_ticket_price : ℝ := sorry

/-- Eight movie tickets cost 2 times as much as one football game ticket -/
axiom ticket_price_relation : 8 * movie_ticket_price = 2 * football_ticket_price

/-- The total amount paid for 8 movie tickets and 5 football game tickets is $840 -/
axiom total_cost : 8 * movie_ticket_price + 5 * football_ticket_price = 840

theorem movie_ticket_price_is_30 : movie_ticket_price = 30 := by sorry

end movie_ticket_price_is_30_l1246_124647


namespace cards_given_away_l1246_124692

theorem cards_given_away (original_cards : ℕ) (remaining_cards : ℕ) 
  (h1 : original_cards = 350) 
  (h2 : remaining_cards = 248) : 
  original_cards - remaining_cards = 102 := by
  sorry

end cards_given_away_l1246_124692


namespace problem_2011_l1246_124633

theorem problem_2011 : (2011^2 - 2011) / 2011 = 2010 := by sorry

end problem_2011_l1246_124633


namespace nancy_grew_six_potatoes_l1246_124656

/-- The number of potatoes Sandy grew -/
def sandy_potatoes : ℕ := 7

/-- The total number of potatoes Nancy and Sandy grew together -/
def total_potatoes : ℕ := 13

/-- The number of potatoes Nancy grew -/
def nancy_potatoes : ℕ := total_potatoes - sandy_potatoes

theorem nancy_grew_six_potatoes : nancy_potatoes = 6 := by
  sorry

end nancy_grew_six_potatoes_l1246_124656


namespace triangle_perimeter_l1246_124624

/-- Given a triangle with sides satisfying specific conditions, prove its perimeter. -/
theorem triangle_perimeter (a b : ℝ) : 
  let side1 := a + b
  let side2 := side1 + (a + 2)
  let side3 := side2 - 3
  side1 + side2 + side3 = 5*a + 3*b + 1 := by
  sorry

end triangle_perimeter_l1246_124624


namespace twelve_eat_both_l1246_124652

/-- Represents the eating habits in a family -/
structure FamilyEatingHabits where
  only_veg : ℕ
  only_non_veg : ℕ
  total_veg : ℕ

/-- Calculates the number of people who eat both veg and non-veg -/
def both_veg_and_non_veg (habits : FamilyEatingHabits) : ℕ :=
  habits.total_veg - habits.only_veg

/-- Theorem: In the given family, 12 people eat both veg and non-veg -/
theorem twelve_eat_both (habits : FamilyEatingHabits) 
    (h1 : habits.only_veg = 19)
    (h2 : habits.only_non_veg = 9)
    (h3 : habits.total_veg = 31) :
    both_veg_and_non_veg habits = 12 := by
  sorry

#eval both_veg_and_non_veg ⟨19, 9, 31⟩

end twelve_eat_both_l1246_124652


namespace normal_vector_of_l_l1246_124608

/-- Definition of the line l: 2x - 3y + 4 = 0 -/
def l (x y : ℝ) : Prop := 2 * x - 3 * y + 4 = 0

/-- Definition of a normal vector to a line -/
def is_normal_vector (v : ℝ × ℝ) (line : ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ v = (k * 2, k * (-3))

/-- Theorem: (4, -6) is a normal vector to the line l -/
theorem normal_vector_of_l : is_normal_vector (4, -6) l := by
  sorry

end normal_vector_of_l_l1246_124608


namespace twice_probability_possible_l1246_124619

/-- Represents the schedule of trains in one direction -/
structure TrainSchedule :=
  (interval : ℝ)
  (offset : ℝ)

/-- Represents the metro system with two directions -/
structure MetroSystem :=
  (direction1 : TrainSchedule)
  (direction2 : TrainSchedule)

/-- Calculates the probability of taking a train in a given direction -/
def probability_of_taking_train (metro : MetroSystem) (direction : ℕ) : ℝ :=
  sorry

/-- Theorem stating that the probability of taking one train can be twice the other -/
theorem twice_probability_possible (metro : MetroSystem) :
  ∃ (direction1 direction2 : ℕ),
    direction1 ≠ direction2 ∧
    probability_of_taking_train metro direction1 = 2 * probability_of_taking_train metro direction2 :=
  sorry

end twice_probability_possible_l1246_124619


namespace probability_six_consecutive_heads_l1246_124675

def coin_flips : ℕ := 8

def favorable_outcomes : ℕ := 17

def total_outcomes : ℕ := 2^coin_flips

theorem probability_six_consecutive_heads :
  (favorable_outcomes : ℚ) / total_outcomes = 17 / 256 :=
sorry

end probability_six_consecutive_heads_l1246_124675


namespace geometric_series_equality_l1246_124650

/-- Given real numbers p, q, and r, if the infinite geometric series
    (p/q) + (p/q^2) + (p/q^3) + ... equals 9, then the infinite geometric series
    (p/(p+r)) + (p/(p+r)^2) + (p/(p+r)^3) + ... equals 9(q-1) / (9q + r - 10) -/
theorem geometric_series_equality (p q r : ℝ) 
  (h : ∑' n, p / q^n = 9) :
  ∑' n, p / (p + r)^n = 9 * (q - 1) / (9 * q + r - 10) := by
  sorry

end geometric_series_equality_l1246_124650


namespace expand_product_l1246_124659

theorem expand_product (x : ℝ) : (x + 2) * (x^2 - 4*x + 1) = x^3 - 2*x^2 - 7*x + 2 := by
  sorry

end expand_product_l1246_124659
