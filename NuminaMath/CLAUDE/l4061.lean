import Mathlib

namespace parking_cost_proof_l4061_406122

-- Define the initial cost for up to 2 hours
def initial_cost : ℝ := 9

-- Define the total parking duration in hours
def total_hours : ℝ := 9

-- Define the average cost per hour for the total duration
def average_cost_per_hour : ℝ := 2.361111111111111

-- Define the cost for each hour in excess of 2 hours
def excess_hour_cost : ℝ := 1.75

-- Theorem statement
theorem parking_cost_proof :
  excess_hour_cost * (total_hours - 2) + initial_cost = average_cost_per_hour * total_hours :=
by sorry

end parking_cost_proof_l4061_406122


namespace g_neg_two_equals_neg_seventeen_l4061_406165

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define g in terms of f
def g (x : ℝ) : ℝ := f x + 1

-- State the theorem
theorem g_neg_two_equals_neg_seventeen
  (h1 : ∀ x, f x + 2 * x^2 = -(f (-x) + 2 * (-x)^2)) -- y = f(x) + 2x^2 is odd
  (h2 : f 2 = 2) -- f(2) = 2
  : g f (-2) = -17 := by
  sorry

end g_neg_two_equals_neg_seventeen_l4061_406165


namespace show_attendance_l4061_406195

theorem show_attendance (adult_price child_price total_cost : ℕ) 
  (num_children : ℕ) (h1 : adult_price = 12) (h2 : child_price = 10) 
  (h3 : num_children = 3) (h4 : total_cost = 66) : 
  ∃ (num_adults : ℕ), num_adults = 3 ∧ 
    adult_price * num_adults + child_price * num_children = total_cost :=
by
  sorry

end show_attendance_l4061_406195


namespace inequality_proof_l4061_406151

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a^3 / (a^3 + 2*b^2)) + (b^3 / (b^3 + 2*c^2)) + (c^3 / (c^3 + 2*a^2)) ≥ 1 := by
  sorry

end inequality_proof_l4061_406151


namespace distinct_paths_6x4_l4061_406147

/-- The number of rows in the grid -/
def rows : ℕ := 4

/-- The number of columns in the grid -/
def cols : ℕ := 6

/-- The total number of steps needed to reach from top-left to bottom-right -/
def total_steps : ℕ := rows + cols - 2

/-- The number of down steps needed -/
def down_steps : ℕ := rows - 1

/-- The number of distinct paths from top-left to bottom-right in a 6x4 grid -/
theorem distinct_paths_6x4 : Nat.choose total_steps down_steps = 56 := by
  sorry

end distinct_paths_6x4_l4061_406147


namespace rhombus_perimeter_l4061_406132

/-- A rhombus with given diagonal lengths has the specified perimeter -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) :
  let side_length := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side_length = 52 := by sorry

end rhombus_perimeter_l4061_406132


namespace molecular_weight_5_moles_AlBr3_l4061_406166

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Bromine in g/mol -/
def atomic_weight_Br : ℝ := 79.90

/-- The number of Aluminum atoms in AlBr3 -/
def num_Al : ℕ := 1

/-- The number of Bromine atoms in AlBr3 -/
def num_Br : ℕ := 3

/-- The number of moles of AlBr3 -/
def num_moles : ℝ := 5

/-- The molecular weight of AlBr3 in g/mol -/
def molecular_weight_AlBr3 : ℝ :=
  num_Al * atomic_weight_Al + num_Br * atomic_weight_Br

/-- Theorem stating that the molecular weight of 5 moles of AlBr3 is 1333.40 grams -/
theorem molecular_weight_5_moles_AlBr3 :
  num_moles * molecular_weight_AlBr3 = 1333.40 := by
  sorry

end molecular_weight_5_moles_AlBr3_l4061_406166


namespace difference_of_largest_prime_factors_l4061_406103

theorem difference_of_largest_prime_factors : ∃ (p q : Nat), 
  Nat.Prime p ∧ Nat.Prime q ∧ p * q = 172081 ∧ 
  ∀ (r : Nat), Nat.Prime r ∧ r ∣ 172081 → r ≤ max p q ∧
  max p q - min p q = 13224 := by
  sorry

end difference_of_largest_prime_factors_l4061_406103


namespace intersection_A_B_l4061_406145

-- Define set A
def A : Set ℝ := {x | (x + 1) * (4 - x) > 0}

-- Define set B
def B : Set ℝ := {x | 0 < x ∧ x < 9}

-- Define the open interval (0, 4)
def open_interval_0_4 : Set ℝ := {x | 0 < x ∧ x < 4}

-- Theorem statement
theorem intersection_A_B : A ∩ B = open_interval_0_4 := by
  sorry

end intersection_A_B_l4061_406145


namespace jason_seashells_l4061_406173

/-- Given that Jason initially had 49 seashells and gave away 13 seashells,
    prove that he now has 36 seashells. -/
theorem jason_seashells (initial : ℕ) (given_away : ℕ) (remaining : ℕ) 
    (h1 : initial = 49)
    (h2 : given_away = 13)
    (h3 : remaining = initial - given_away) :
  remaining = 36 := by
  sorry

end jason_seashells_l4061_406173


namespace new_tv_width_l4061_406186

theorem new_tv_width (first_tv_width : ℝ) (first_tv_height : ℝ) (first_tv_cost : ℝ)
                     (new_tv_height : ℝ) (new_tv_cost : ℝ) :
  first_tv_width = 24 →
  first_tv_height = 16 →
  first_tv_cost = 672 →
  new_tv_height = 32 →
  new_tv_cost = 1152 →
  (first_tv_cost / (first_tv_width * first_tv_height)) =
    (new_tv_cost / (new_tv_height * (new_tv_cost / (new_tv_height * (first_tv_cost / (first_tv_width * first_tv_height) - 1))))) + 1 →
  new_tv_cost / (new_tv_height * (first_tv_cost / (first_tv_width * first_tv_height) - 1)) = 48 :=
by
  sorry

#check new_tv_width

end new_tv_width_l4061_406186


namespace perpendicular_lines_a_values_l4061_406199

theorem perpendicular_lines_a_values (a : ℝ) : 
  let l1 := {(x, y) : ℝ × ℝ | a * x + 2 * y + 1 = 0}
  let l2 := {(x, y) : ℝ × ℝ | (3 - a) * x - y + a = 0}
  let slope1 := -a / 2
  let slope2 := 3 - a
  (slope1 * slope2 = -1) → (a = 1 ∨ a = 2) := by
sorry

end perpendicular_lines_a_values_l4061_406199


namespace trapezoid_constructible_l4061_406149

/-- A trapezoid with given bases and diagonals -/
structure Trapezoid where
  a : ℝ  -- length of one base
  c : ℝ  -- length of the other base
  e : ℝ  -- length of one diagonal
  f : ℝ  -- length of the other diagonal

/-- Condition for trapezoid constructibility -/
def is_constructible (t : Trapezoid) : Prop :=
  t.e + t.f > t.a + t.c ∧
  t.e + (t.a + t.c) > t.f ∧
  t.f + (t.a + t.c) > t.e

/-- Theorem stating that a trapezoid with bases 8 and 4, and diagonals 9 and 15 is constructible -/
theorem trapezoid_constructible : 
  is_constructible { a := 8, c := 4, e := 9, f := 15 } := by
  sorry


end trapezoid_constructible_l4061_406149


namespace valid_domains_for_range_l4061_406134

def f (x : ℝ) := x^2 - 2*x + 2

theorem valid_domains_for_range (a b : ℝ) (h : a < b) :
  (∀ x ∈ Set.Icc a b, 1 ≤ f x ∧ f x ≤ 2) →
  (∀ y ∈ Set.Icc 1 2, ∃ x ∈ Set.Icc a b, f x = y) →
  (a = 0 ∧ b = 1) ∨ (a = 1/4 ∧ b = 2) :=
by sorry

end valid_domains_for_range_l4061_406134


namespace be_length_is_fourth_root_three_l4061_406190

/-- A rhombus with specific properties and internal rectangles -/
structure SpecialRhombus where
  -- The side length of the rhombus
  side_length : ℝ
  -- The length of one diagonal of the rhombus
  diagonal_length : ℝ
  -- The length of the side BE of the internal rectangle EBCF
  be_length : ℝ
  -- The side length is 2
  side_length_eq : side_length = 2
  -- One diagonal measures 2√3
  diagonal_eq : diagonal_length = 2 * Real.sqrt 3
  -- EBCF is a square (implied by equal sides along BC and BE)
  ebcf_square : be_length * be_length = be_length * be_length
  -- Area of EBCF + Area of JKHG = Area of rhombus (since they are congruent and fit within the rhombus)
  area_eq : 2 * (be_length * be_length) = (1 / 2) * diagonal_length * side_length

/-- The length of BE in the special rhombus is ∜3 -/
theorem be_length_is_fourth_root_three (r : SpecialRhombus) : r.be_length = Real.sqrt (Real.sqrt 3) := by
  sorry


end be_length_is_fourth_root_three_l4061_406190


namespace distance_to_line_l4061_406116

/-- Given two perpendicular lines and a point, prove the distance to a third line -/
theorem distance_to_line (m : ℝ) : 
  (∀ x y, 2*x + y - 2 = 0 → x + m*y - 1 = 0 → (2 : ℝ) * (-1/m) = -1) →
  let P := (m, m)
  (abs (P.1 + P.2 + 3) / Real.sqrt 2 : ℝ) = Real.sqrt 2 / 2 := by
  sorry

end distance_to_line_l4061_406116


namespace items_distribution_count_l4061_406193

-- Define the number of items and bags
def num_items : ℕ := 5
def num_bags : ℕ := 4

-- Define a function to calculate the number of ways to distribute items
def distribute_items (items : ℕ) (bags : ℕ) : ℕ :=
  sorry

-- Theorem statement
theorem items_distribution_count :
  distribute_items num_items num_bags = 52 := by
  sorry

end items_distribution_count_l4061_406193


namespace sin_75_deg_l4061_406177

/-- Proves that the sine of 75 degrees is equal to (√6 + √2) / 4 -/
theorem sin_75_deg : Real.sin (75 * π / 180) = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end sin_75_deg_l4061_406177


namespace no_primes_in_range_l4061_406154

theorem no_primes_in_range (n : ℕ) (h : n > 1) :
  ∀ k, n! + 1 < k ∧ k < n! + 2*n → ¬ Nat.Prime k := by
  sorry

end no_primes_in_range_l4061_406154


namespace two_round_trips_time_l4061_406196

/-- Represents the time for a round trip given the time for one-way trip at normal speed -/
def round_trip_time (one_way_time : ℝ) : ℝ := one_way_time + 2 * one_way_time

/-- Proves that two round trips take 6 hours when one-way trip takes 1 hour -/
theorem two_round_trips_time : round_trip_time 1 * 2 = 6 := by
  sorry

end two_round_trips_time_l4061_406196


namespace minimum_at_two_implies_a_twelve_l4061_406160

/-- Given a function f(x) = x^3 - ax, prove that if f takes its minimum value at x = 2, then a = 12 -/
theorem minimum_at_two_implies_a_twelve (a : ℝ) : 
  (∀ x : ℝ, x^3 - a*x ≥ 2^3 - a*2) → a = 12 := by
  sorry

end minimum_at_two_implies_a_twelve_l4061_406160


namespace sequence_a_formula_l4061_406123

def sequence_a : ℕ → ℤ
  | 0 => 1
  | 1 => 5
  | (n + 2) => (2 * (sequence_a (n + 1))^2 - 3 * sequence_a (n + 1) - 9) / (2 * sequence_a n)

theorem sequence_a_formula : ∀ n : ℕ, sequence_a n = 2^(n + 2) - 3 := by
  sorry

end sequence_a_formula_l4061_406123


namespace additional_triangles_for_hexagon_l4061_406108

/-- The number of vertices in a hexagon -/
def hexagon_vertices : ℕ := 6

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of triangles in the original shape -/
def original_triangles : ℕ := 36

/-- The number of additional triangles needed for each vertex -/
def triangles_per_vertex : ℕ := 2

/-- The number of additional triangles needed for each side -/
def triangles_per_side : ℕ := 1

/-- The theorem stating the smallest number of additional triangles needed -/
theorem additional_triangles_for_hexagon :
  hexagon_vertices * triangles_per_vertex + hexagon_sides * triangles_per_side = 18 :=
sorry

end additional_triangles_for_hexagon_l4061_406108


namespace all_equal_cyclic_inequality_l4061_406157

theorem all_equal_cyclic_inequality (a : Fin 100 → ℝ) 
  (h : ∀ i : Fin 100, a i - 3 * a (i + 1) + 2 * a (i + 2) ≥ 0) :
  ∀ i j : Fin 100, a i = a j :=
sorry

end all_equal_cyclic_inequality_l4061_406157


namespace jake_final_balance_l4061_406153

/-- Represents Jake's bitcoin transactions and calculates his final balance --/
def jake_bitcoin_balance (initial_fortune : ℚ) (investment : ℚ) (first_donation : ℚ) 
  (brother_return : ℚ) (second_donation : ℚ) : ℚ :=
  let after_investment := initial_fortune - investment
  let after_first_donation := after_investment - (first_donation / 2)
  let after_giving_to_brother := after_first_donation / 2
  let after_brother_return := after_giving_to_brother + brother_return
  let after_quadrupling := after_brother_return * 4
  after_quadrupling - (second_donation * 4)

/-- Theorem stating that Jake ends up with 95 bitcoins --/
theorem jake_final_balance : 
  jake_bitcoin_balance 120 40 25 5 15 = 95 := by
  sorry

end jake_final_balance_l4061_406153


namespace jellybean_problem_l4061_406155

theorem jellybean_problem (x : ℕ) : 
  x ≥ 150 ∧ 
  x % 15 = 14 ∧ 
  x % 17 = 16 ∧ 
  (∀ y : ℕ, y ≥ 150 ∧ y % 15 = 14 ∧ y % 17 = 16 → x ≤ y) → 
  x = 254 := by
sorry

end jellybean_problem_l4061_406155


namespace christines_speed_l4061_406150

/-- Given a distance of 20 miles and a time of 5 hours, the speed is 4 miles per hour. -/
theorem christines_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
  (h1 : distance = 20)
  (h2 : time = 5)
  (h3 : speed = distance / time) :
  speed = 4 := by
  sorry

end christines_speed_l4061_406150


namespace consecutive_integers_sum_l4061_406194

theorem consecutive_integers_sum (x : ℤ) :
  x * (x + 1) * (x + 2) = 2730 → x + (x + 1) + (x + 2) = 42 := by
  sorry

end consecutive_integers_sum_l4061_406194


namespace ladder_length_is_twice_h_l4061_406182

/-- The length of a ladder resting against two walls in an alley -/
def ladder_length (w h k : ℝ) : ℝ :=
  2 * h

/-- Theorem: The length of the ladder is twice the height at point Q -/
theorem ladder_length_is_twice_h (w h k : ℝ) (hw : w > 0) (hh : h > 0) (hk : k > 0) :
  ladder_length w h k = 2 * h :=
by
  sorry

#check ladder_length_is_twice_h

end ladder_length_is_twice_h_l4061_406182


namespace base_difference_not_divisible_by_three_l4061_406109

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The difference of (2021)_b and (221)_b in base 10 -/
def baseDifference (b : Nat) : Nat :=
  toBase10 [2, 0, 2, 1] b - toBase10 [2, 2, 1] b

theorem base_difference_not_divisible_by_three (b : Nat) :
  b > 0 → (baseDifference b % 3 ≠ 0 ↔ b % 3 = 2) := by
  sorry

end base_difference_not_divisible_by_three_l4061_406109


namespace simplify_expression_l4061_406143

theorem simplify_expression (a b : ℝ) : a * b - (a^2 - a * b + b^2) = -a^2 + 2 * a * b - b^2 := by
  sorry

end simplify_expression_l4061_406143


namespace isabelle_ticket_cost_l4061_406156

def brothers_ticket_cost : ℕ := 20
def brothers_savings : ℕ := 5
def isabelle_savings : ℕ := 5
def isabelle_earnings : ℕ := 30

def total_amount_needed : ℕ := isabelle_earnings + isabelle_savings + brothers_savings

theorem isabelle_ticket_cost :
  total_amount_needed - brothers_ticket_cost = 15 := by sorry

end isabelle_ticket_cost_l4061_406156


namespace least_N_for_probability_l4061_406172

def P (N : ℕ) : ℚ := 2 * (N / 3 + 1) / (N + 2)

def is_multiple_of_seven (N : ℕ) : Prop := ∃ k, N = 7 * k

theorem least_N_for_probability (N : ℕ) :
  is_multiple_of_seven N →
  (∀ M, is_multiple_of_seven M → M < N → P M ≥ 7/10) →
  P N < 7/10 →
  N = 700 :=
sorry

end least_N_for_probability_l4061_406172


namespace f_12345_equals_12345_l4061_406100

/-- The set of all non-zero real-valued functions satisfying the given functional equation. -/
def S : Set (ℝ → ℝ) :=
  {f | ∀ x y z : ℝ, f ≠ 0 ∧ f (x^2 + y * f z) = x * f x + z * f y}

/-- Theorem stating that for any function in S, f(12345) = 12345. -/
theorem f_12345_equals_12345 (f : ℝ → ℝ) (hf : f ∈ S) : f 12345 = 12345 := by
  sorry

end f_12345_equals_12345_l4061_406100


namespace apple_pear_puzzle_l4061_406158

theorem apple_pear_puzzle (apples pears : ℕ) : 
  (apples : ℚ) / 3 = (pears : ℚ) / 2 + 1 →
  (apples : ℚ) / 5 = (pears : ℚ) / 4 - 3 →
  apples = 23 ∧ pears = 16 := by
sorry

end apple_pear_puzzle_l4061_406158


namespace simplify_and_rationalize_l4061_406198

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 8) * (Real.sqrt 6 / Real.sqrt 9) = Real.sqrt 35 / 14 := by
  sorry

end simplify_and_rationalize_l4061_406198


namespace complex_equation_solution_l4061_406142

def complex_i : ℂ := Complex.I

theorem complex_equation_solution (a : ℝ) :
  (2 : ℂ) / (a + complex_i) = 1 - complex_i → a = 1 := by
  sorry

end complex_equation_solution_l4061_406142


namespace cauchy_equation_solution_l4061_406102

theorem cauchy_equation_solution (f : ℝ → ℝ) 
  (h_cauchy : ∀ x y : ℝ, f (x + y) = f x + f y) 
  (h_condition : f 1 ^ 2 = f 1) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x) := by
  sorry

end cauchy_equation_solution_l4061_406102


namespace sum_of_squares_l4061_406168

theorem sum_of_squares : 17^2 + 19^2 + 23^2 + 29^2 = 2020 := by
  sorry

end sum_of_squares_l4061_406168


namespace boys_not_in_varsity_clubs_l4061_406144

theorem boys_not_in_varsity_clubs (total_students : ℕ) (girls_percentage : ℚ) (boys_in_clubs_fraction : ℚ) :
  total_students = 150 →
  girls_percentage = 60 / 100 →
  boys_in_clubs_fraction = 1 / 3 →
  (total_students : ℚ) * (1 - girls_percentage) * (1 - boys_in_clubs_fraction) = 40 :=
by sorry

end boys_not_in_varsity_clubs_l4061_406144


namespace tv_watching_weeks_l4061_406124

/-- Represents Flynn's TV watching habits and total time --/
structure TVWatching where
  weekdayMinutes : ℕ  -- Minutes watched per weekday night
  weekendHours : ℕ    -- Additional hours watched on weekends
  totalHours : ℕ      -- Total hours watched

/-- Calculates the number of weeks based on TV watching habits --/
def calculateWeeks (tw : TVWatching) : ℚ :=
  let weekdayHours : ℚ := (tw.weekdayMinutes * 5 : ℚ) / 60
  let totalWeeklyHours : ℚ := weekdayHours + tw.weekendHours
  tw.totalHours / totalWeeklyHours

/-- Theorem stating that 234 hours of TV watching corresponds to 52 weeks --/
theorem tv_watching_weeks (tw : TVWatching) 
  (h1 : tw.weekdayMinutes = 30)
  (h2 : tw.weekendHours = 2)
  (h3 : tw.totalHours = 234) :
  calculateWeeks tw = 52 := by
  sorry

end tv_watching_weeks_l4061_406124


namespace polynomial_division_l4061_406184

theorem polynomial_division (z : ℝ) :
  6 * z^5 - 5 * z^4 + 2 * z^3 - 8 * z^2 + 7 * z - 3 = 
  (z^2 - 1) * (6 * z^3 + z^2 + 3 * z) + 6 := by
  sorry

end polynomial_division_l4061_406184


namespace sequence_length_l4061_406110

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

theorem sequence_length : 
  ∃ n : ℕ, n = 757 ∧ arithmetic_sequence 2 4 n = 3026 := by sorry

end sequence_length_l4061_406110


namespace max_distance_ellipse_to_line_l4061_406197

/-- The maximum distance from a point on the ellipse x^2/4 + y^2 = 1 to the line x + 2y = 0 -/
theorem max_distance_ellipse_to_line :
  let ellipse := {P : ℝ × ℝ | P.1^2/4 + P.2^2 = 1}
  let line := {P : ℝ × ℝ | P.1 + 2*P.2 = 0}
  ∃ (d : ℝ), d = 2*Real.sqrt 10/5 ∧
    ∀ P ∈ ellipse, ∀ Q ∈ line,
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ d ∧
      ∃ P' ∈ ellipse, ∃ Q' ∈ line,
        Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) = d :=
by sorry

end max_distance_ellipse_to_line_l4061_406197


namespace omelet_problem_l4061_406175

/-- The number of people that can be served omelets given the conditions -/
def number_of_people (eggs_per_dozen : ℕ) (eggs_per_omelet : ℕ) (omelets_per_person : ℕ) : ℕ :=
  let total_eggs := 3 * eggs_per_dozen
  let total_omelets := total_eggs / eggs_per_omelet
  total_omelets / omelets_per_person

/-- Theorem stating that under the given conditions, the number of people is 3 -/
theorem omelet_problem : number_of_people 12 4 3 = 3 := by
  sorry

end omelet_problem_l4061_406175


namespace coffee_shop_lattes_l4061_406164

/-- The number of teas sold -/
def T : ℕ := 6

/-- The number of lattes sold -/
def L : ℕ := 4 * T + 8

theorem coffee_shop_lattes : L = 32 := by
  sorry

end coffee_shop_lattes_l4061_406164


namespace blue_paint_cans_l4061_406120

/-- Given a paint mixture with a blue to green ratio of 4:1 and a total of 40 cans,
    prove that 32 cans of blue paint are required. -/
theorem blue_paint_cans (total_cans : ℕ) (blue_ratio green_ratio : ℕ) 
  (h1 : total_cans = 40)
  (h2 : blue_ratio = 4)
  (h3 : green_ratio = 1) :
  (blue_ratio * total_cans) / (blue_ratio + green_ratio) = 32 := by
sorry


end blue_paint_cans_l4061_406120


namespace dividend_calculation_l4061_406112

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 21) 
  (h2 : quotient = 14) 
  (h3 : remainder = 7) : 
  divisor * quotient + remainder = 301 := by
  sorry

end dividend_calculation_l4061_406112


namespace inequality_proof_l4061_406183

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 0 > c) :
  (c - a < c - b) ∧ (a⁻¹ * c > b⁻¹ * c) := by
  sorry

end inequality_proof_l4061_406183


namespace a_11_value_l4061_406115

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem a_11_value (a : ℕ → ℝ) 
    (h_arithmetic : arithmetic_sequence a)
    (h_a1 : a 1 = 1)
    (h_diff : ∀ n : ℕ, a (n + 2) - a n = 6) :
  a 11 = 31 := by
sorry

end a_11_value_l4061_406115


namespace remainder_70_div_17_l4061_406161

theorem remainder_70_div_17 : 70 % 17 = 2 := by sorry

end remainder_70_div_17_l4061_406161


namespace pens_sold_correct_solve_paul_pens_problem_l4061_406140

/-- Calculates the number of pens sold given the initial and final counts -/
def pens_sold (initial : ℕ) (final : ℕ) : ℕ :=
  initial - final

theorem pens_sold_correct (initial final : ℕ) (h : initial ≥ final) :
  pens_sold initial final = initial - final :=
by sorry

/-- The specific problem instance -/
def paul_pens_problem : Prop :=
  pens_sold 106 14 = 92

theorem solve_paul_pens_problem : paul_pens_problem :=
by sorry

end pens_sold_correct_solve_paul_pens_problem_l4061_406140


namespace equilateral_triangle_perimeter_l4061_406136

theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) : 
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by
  sorry

end equilateral_triangle_perimeter_l4061_406136


namespace cardinality_of_C_l4061_406159

def A : Finset ℕ := {0, 2, 3, 4, 5, 7}
def B : Finset ℕ := {1, 2, 3, 4, 6}
def C : Finset ℕ := A \ B

theorem cardinality_of_C : Finset.card C = 3 := by
  sorry

end cardinality_of_C_l4061_406159


namespace power_fraction_simplification_l4061_406126

theorem power_fraction_simplification :
  (5^2022)^2 - (5^2020)^2 / (5^2021)^2 - (5^2019)^2 = 5^2 := by
  sorry

end power_fraction_simplification_l4061_406126


namespace johns_trip_duration_l4061_406138

/-- The duration of John's trip given his travel conditions -/
def trip_duration (first_country_duration : ℕ) (num_countries : ℕ) : ℕ :=
  first_country_duration + 2 * first_country_duration * (num_countries - 1)

/-- Theorem stating that John's trip duration is 10 weeks -/
theorem johns_trip_duration :
  trip_duration 2 3 = 10 := by
  sorry

end johns_trip_duration_l4061_406138


namespace min_value_sum_reciprocals_l4061_406106

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (pos_p : p > 0) (pos_q : q > 0) (pos_r : r > 0) 
  (pos_s : s > 0) (pos_t : t > 0) (pos_u : u > 0)
  (sum_eq_10 : p + q + r + s + t + u = 10) : 
  (1/p + 9/q + 4/r + 1/s + 16/t + 25/u) ≥ 25.6 := by
  sorry

end min_value_sum_reciprocals_l4061_406106


namespace remaining_quantity_average_l4061_406121

theorem remaining_quantity_average (total : ℕ) (avg_all : ℚ) (avg_five : ℚ) (avg_two : ℚ) :
  total = 8 ∧ avg_all = 15 ∧ avg_five = 10 ∧ avg_two = 22 →
  (total * avg_all - 5 * avg_five - 2 * avg_two) = 26 := by
sorry

end remaining_quantity_average_l4061_406121


namespace card_problem_l4061_406118

theorem card_problem (x y : ℕ) : 
  x - 1 = y + 1 → 
  x + 1 = 2 * (y - 1) → 
  x + y = 12 := by
sorry

end card_problem_l4061_406118


namespace new_car_distance_l4061_406171

theorem new_car_distance (old_car_speed : ℝ) (old_car_distance : ℝ) (new_car_speed : ℝ) : 
  old_car_distance = 150 →
  new_car_speed = old_car_speed * 1.3 →
  new_car_speed * (old_car_distance / old_car_speed) = 195 := by
sorry

end new_car_distance_l4061_406171


namespace gcd_of_large_powers_l4061_406131

theorem gcd_of_large_powers (n m : ℕ) : 
  Nat.gcd (2^1050 - 1) (2^1062 - 1) = 2^12 - 1 :=
by sorry

end gcd_of_large_powers_l4061_406131


namespace even_function_implies_m_eq_neg_one_l4061_406119

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The quadratic function f(x) = (m - 1)x² - (m² - 1)x + m + 2 -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (m - 1) * x^2 - (m^2 - 1) * x + m + 2

theorem even_function_implies_m_eq_neg_one :
  ∀ m : ℝ, EvenFunction (f m) → m = -1 := by
  sorry

end even_function_implies_m_eq_neg_one_l4061_406119


namespace solution_set_inequality_l4061_406114

theorem solution_set_inequality (f : ℝ → ℝ) (hf : ∀ x, f x + (deriv f) x > 1) (hf0 : f 0 = 4) :
  {x : ℝ | f x > 3 / Real.exp x + 1} = {x : ℝ | x > 0} := by
  sorry

end solution_set_inequality_l4061_406114


namespace jason_bookcase_weight_difference_l4061_406129

/-- Represents the bookcase and Jason's collection of items -/
structure Bookcase :=
  (shelves : Nat)
  (shelf_weight_limit : Nat)
  (hardcover_books : Nat)
  (textbooks : Nat)
  (knick_knacks : Nat)
  (max_hardcover_weight : Real)
  (max_textbook_weight : Real)
  (max_knick_knack_weight : Real)

/-- Calculates the maximum weight of the collection minus the bookcase's weight limit -/
def weight_difference (b : Bookcase) : Real :=
  b.hardcover_books * b.max_hardcover_weight +
  b.textbooks * b.max_textbook_weight +
  b.knick_knacks * b.max_knick_knack_weight -
  b.shelves * b.shelf_weight_limit

/-- Theorem stating that the weight difference for Jason's collection is 195 pounds -/
theorem jason_bookcase_weight_difference :
  ∃ (b : Bookcase),
    b.shelves = 4 ∧
    b.shelf_weight_limit = 20 ∧
    b.hardcover_books = 70 ∧
    b.textbooks = 30 ∧
    b.knick_knacks = 10 ∧
    b.max_hardcover_weight = 1.5 ∧
    b.max_textbook_weight = 3 ∧
    b.max_knick_knack_weight = 8 ∧
    weight_difference b = 195 :=
  sorry

end jason_bookcase_weight_difference_l4061_406129


namespace holly_throws_five_times_l4061_406117

/-- Represents the Frisbee throwing scenario -/
structure FrisbeeScenario where
  bess_throw_distance : ℕ
  bess_throw_count : ℕ
  holly_throw_distance : ℕ
  total_distance : ℕ

/-- Calculates the number of times Holly throws the Frisbee -/
def holly_throw_count (scenario : FrisbeeScenario) : ℕ :=
  (scenario.total_distance - 2 * scenario.bess_throw_distance * scenario.bess_throw_count) / scenario.holly_throw_distance

/-- Theorem stating that Holly throws the Frisbee 5 times in the given scenario -/
theorem holly_throws_five_times (scenario : FrisbeeScenario) 
  (h1 : scenario.bess_throw_distance = 20)
  (h2 : scenario.bess_throw_count = 4)
  (h3 : scenario.holly_throw_distance = 8)
  (h4 : scenario.total_distance = 200) :
  holly_throw_count scenario = 5 := by
  sorry

#eval holly_throw_count { bess_throw_distance := 20, bess_throw_count := 4, holly_throw_distance := 8, total_distance := 200 }

end holly_throws_five_times_l4061_406117


namespace even_function_implies_a_eq_neg_one_l4061_406189

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function y = (x+1)(x-a) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

/-- If f(a) is an even function, then a = -1 -/
theorem even_function_implies_a_eq_neg_one (a : ℝ) :
  IsEven (f a) → a = -1 := by sorry

end even_function_implies_a_eq_neg_one_l4061_406189


namespace extreme_value_implies_b_l4061_406104

/-- Given a function f(x) = x³ + ax² + bx + a² where a and b are real numbers,
    if f(x) has an extreme value of 10 at x = 1, then b = -11 -/
theorem extreme_value_implies_b (a b : ℝ) : 
  let f := fun (x : ℝ) => x^3 + a*x^2 + b*x + a^2
  (∃ (ε : ℝ), ∀ (x : ℝ), x ≠ 1 → |x - 1| < ε → f x ≤ f 1) ∧ 
  (f 1 = 10) →
  b = -11 := by sorry

end extreme_value_implies_b_l4061_406104


namespace sqrt_450_equals_15_l4061_406141

theorem sqrt_450_equals_15 : Real.sqrt 450 = 15 := by
  sorry

end sqrt_450_equals_15_l4061_406141


namespace scarlet_savings_l4061_406146

theorem scarlet_savings (initial_savings : ℕ) (earrings_cost : ℕ) (necklace_cost : ℕ) : 
  initial_savings = 80 → earrings_cost = 23 → necklace_cost = 48 → 
  initial_savings - (earrings_cost + necklace_cost) = 9 := by
sorry

end scarlet_savings_l4061_406146


namespace triangle_angle_identity_l4061_406178

theorem triangle_angle_identity (α : Real) 
  (h1 : 0 < α ∧ α < Real.pi)  -- α is an internal angle of a triangle
  (h2 : Real.sin α * Real.cos α = 1/8) :  -- given condition
  Real.cos α + Real.sin α = Real.sqrt 5 / 2 := by
  sorry

end triangle_angle_identity_l4061_406178


namespace marble_arrangement_theorem_l4061_406111

/-- Represents the colors of marbles --/
inductive Color
| Green
| Red

/-- Represents a circular arrangement of marbles --/
def CircularArrangement := List Color

/-- Counts the number of marbles with same-color neighbors --/
def countSameColorNeighbors (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Counts the number of marbles with different-color neighbors --/
def countDifferentColorNeighbors (arrangement : CircularArrangement) : Nat :=
  sorry

/-- Checks if an arrangement satisfies the neighbor color condition --/
def isValidArrangement (arrangement : CircularArrangement) : Prop :=
  countSameColorNeighbors arrangement = countDifferentColorNeighbors arrangement

/-- Counts the number of valid arrangements --/
def countValidArrangements (greenMarbles redMarbles : Nat) : Nat :=
  sorry

/-- The main theorem --/
theorem marble_arrangement_theorem :
  let greenMarbles : Nat := 7
  let maxRedMarbles : Nat := 14
  (countValidArrangements greenMarbles maxRedMarbles) % 1000 = 432 :=
sorry

end marble_arrangement_theorem_l4061_406111


namespace no_real_solutions_l4061_406135

theorem no_real_solutions : ¬∃ (x : ℝ), (3*x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1) := by
  sorry

end no_real_solutions_l4061_406135


namespace partnership_investment_l4061_406180

/-- Represents a partnership with three partners -/
structure Partnership where
  investmentA : ℝ
  investmentB : ℝ
  investmentC : ℝ
  totalProfit : ℝ
  cProfit : ℝ

/-- Calculates the total investment of the partnership -/
def totalInvestment (p : Partnership) : ℝ :=
  p.investmentA + p.investmentB + p.investmentC

/-- Theorem stating that if the given conditions are met, 
    then Partner C's investment is 36000 -/
theorem partnership_investment 
  (p : Partnership) 
  (h1 : p.investmentA = 24000)
  (h2 : p.investmentB = 32000)
  (h3 : p.totalProfit = 92000)
  (h4 : p.cProfit = 36000)
  (h5 : p.cProfit / p.totalProfit = p.investmentC / totalInvestment p) :
  p.investmentC = 36000 :=
sorry

end partnership_investment_l4061_406180


namespace second_machine_time_l4061_406167

theorem second_machine_time (t1 t_combined : ℝ) (h1 : t1 = 9) (h2 : t_combined = 4.235294117647059) : 
  let t2 := (t1 * t_combined) / (t1 - t_combined)
  t2 = 8 := by sorry

end second_machine_time_l4061_406167


namespace mersenne_primes_less_than_1000_are_3_7_31_127_l4061_406187

def is_mersenne_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ n : ℕ, Nat.Prime n ∧ p = 2^n - 1

def mersenne_primes_less_than_1000 : Set ℕ :=
  {p : ℕ | is_mersenne_prime p ∧ p < 1000}

theorem mersenne_primes_less_than_1000_are_3_7_31_127 :
  mersenne_primes_less_than_1000 = {3, 7, 31, 127} :=
by sorry

end mersenne_primes_less_than_1000_are_3_7_31_127_l4061_406187


namespace combined_weight_in_pounds_l4061_406113

-- Define the weight of the elephant in tons
def elephant_weight_tons : ℝ := 3

-- Define the conversion factor from tons to pounds
def tons_to_pounds : ℝ := 2000

-- Define the weight ratio of the donkey compared to the elephant
def donkey_weight_ratio : ℝ := 0.1

-- Theorem statement
theorem combined_weight_in_pounds :
  let elephant_weight_pounds := elephant_weight_tons * tons_to_pounds
  let donkey_weight_pounds := elephant_weight_pounds * donkey_weight_ratio
  elephant_weight_pounds + donkey_weight_pounds = 6600 := by
sorry

end combined_weight_in_pounds_l4061_406113


namespace circle_and_symmetry_l4061_406162

-- Define the circle
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define the chord line
def ChordLine (x : ℝ) := x + 1

-- Define the variable line
def VariableLine (k : ℝ) (x : ℝ) := k * (x - 1)

-- Define the fixed point N
def N : ℝ × ℝ := (4, 0)

-- Main theorem
theorem circle_and_symmetry :
  -- The chord intercepted by y = x + 1 has length √14
  (∃ (a b : ℝ), (a, ChordLine a) ∈ Circle ∧ (b, ChordLine b) ∈ Circle ∧ (b - a)^2 + (ChordLine b - ChordLine a)^2 = 14) →
  -- The equation of the circle is x² + y² = 4
  (∀ (x y : ℝ), (x, y) ∈ Circle ↔ x^2 + y^2 = 4) ∧
  -- N is the fixed point of symmetry
  (∀ (k : ℝ) (A B : ℝ × ℝ),
    k ≠ 0 →
    A ∈ Circle →
    B ∈ Circle →
    A.2 = VariableLine k A.1 →
    B.2 = VariableLine k B.1 →
    (A.2 / (A.1 - N.1) + B.2 / (B.1 - N.1) = 0)) :=
by sorry


end circle_and_symmetry_l4061_406162


namespace language_coverage_probability_l4061_406128

def total_students : ℕ := 40
def french_students : ℕ := 30
def spanish_students : ℕ := 32
def german_students : ℕ := 10
def german_and_other : ℕ := 26

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

theorem language_coverage_probability :
  let french_and_spanish : ℕ := french_students + spanish_students - total_students
  let french_only : ℕ := french_students - french_and_spanish
  let spanish_only : ℕ := spanish_students - french_and_spanish
  let remaining : ℕ := total_students - (french_only + spanish_only + french_and_spanish)
  let total_combinations : ℕ := choose total_students 2
  let unfavorable_outcomes : ℕ := choose french_only 2 + choose spanish_only 2 + choose remaining 2
  (total_combinations - unfavorable_outcomes : ℚ) / total_combinations = 353 / 390 :=
sorry

end language_coverage_probability_l4061_406128


namespace sqrt_simplification_l4061_406130

theorem sqrt_simplification (a : ℝ) (ha : a ≥ 0) :
  Real.sqrt (a^(1/2) * Real.sqrt (a^(1/2) * Real.sqrt a)) = a^(1/2) := by
  sorry

end sqrt_simplification_l4061_406130


namespace sixtieth_pair_is_five_seven_l4061_406174

/-- Definition of the sequence of pairs -/
def pair_sequence : ℕ → ℕ × ℕ
| n => let group := (n + 1).sqrt
       let position := n - (group * (group - 1)) / 2
       (position, group + 1 - position)

/-- Theorem stating that the 60th pair is (5, 7) -/
theorem sixtieth_pair_is_five_seven :
  pair_sequence 60 = (5, 7) := by
  sorry


end sixtieth_pair_is_five_seven_l4061_406174


namespace insurance_cost_over_decade_l4061_406169

/-- The amount spent on car insurance in a year -/
def yearly_insurance_cost : ℕ := 4000

/-- The number of years in a decade -/
def years_in_decade : ℕ := 10

/-- The total cost of car insurance over a decade -/
def decade_insurance_cost : ℕ := yearly_insurance_cost * years_in_decade

theorem insurance_cost_over_decade : 
  decade_insurance_cost = 40000 := by sorry

end insurance_cost_over_decade_l4061_406169


namespace function_value_at_negative_one_l4061_406170

/-- Given a function f(x) = ax³ + b*sin(x) + 1 where f(1) = 5, prove that f(-1) = -3 -/
theorem function_value_at_negative_one 
  (a b : ℝ) 
  (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * Real.sin x + 1) 
  (h2 : f 1 = 5) : 
  f (-1) = -3 := by
sorry

end function_value_at_negative_one_l4061_406170


namespace max_sides_equal_longest_diagonal_l4061_406176

/-- A convex polygon is a polygon where all interior angles are less than or equal to 180 degrees. -/
def ConvexPolygon (P : Set (ℝ × ℝ)) : Prop := sorry

/-- The longest diagonal of a polygon is the longest line segment connecting any two non-adjacent vertices. -/
def LongestDiagonal (P : Set (ℝ × ℝ)) : ℝ := sorry

/-- A side of a polygon is a line segment connecting two adjacent vertices. -/
def Side (P : Set (ℝ × ℝ)) (s : ℝ × ℝ → ℝ × ℝ → Prop) : Prop := sorry

/-- Count of sides equal to the longest diagonal -/
def CountSidesEqualToLongestDiagonal (P : Set (ℝ × ℝ)) : ℕ := sorry

/-- An equilateral triangle is a triangle with all sides of equal length. -/
def EquilateralTriangle (P : Set (ℝ × ℝ)) : Prop := sorry

theorem max_sides_equal_longest_diagonal 
  (P : Set (ℝ × ℝ)) 
  (h_convex : ConvexPolygon P) :
  (CountSidesEqualToLongestDiagonal P ≤ 2) ∨ 
  (CountSidesEqualToLongestDiagonal P = 3 ∧ EquilateralTriangle P) := by
  sorry

end max_sides_equal_longest_diagonal_l4061_406176


namespace grasshopper_visits_all_integers_l4061_406179

def grasshopper_jump (k : ℕ) : ℤ :=
  if k % 2 = 0 then -k else k + 1

def grasshopper_position (n : ℕ) : ℤ :=
  (List.range n).foldl (λ acc k => acc + grasshopper_jump k) 0

theorem grasshopper_visits_all_integers :
  ∀ (z : ℤ), ∃ (n : ℕ), grasshopper_position n = z :=
sorry

end grasshopper_visits_all_integers_l4061_406179


namespace division_problem_l4061_406152

theorem division_problem (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  a = 13 * b + 6 ∧ 
  a + b + 13 + 6 = 137 → 
  a = 110 ∧ b = 8 := by
sorry

end division_problem_l4061_406152


namespace runners_meeting_point_l4061_406105

/-- Represents the meeting point on the circular track -/
inductive MeetingPoint
| S -- Boundary of A and D
| A
| B
| C
| D

/-- Represents a runner on the circular track -/
structure Runner where
  startPosition : ℝ -- Position in meters from point S
  direction : Bool -- true for counterclockwise, false for clockwise
  distanceRun : ℝ -- Total distance run in meters

/-- Theorem stating where Alice and Bob meet on the circular track -/
theorem runners_meeting_point 
  (trackCircumference : ℝ)
  (alice : Runner)
  (bob : Runner)
  (h1 : trackCircumference = 60)
  (h2 : alice.startPosition = 0)
  (h3 : alice.direction = true)
  (h4 : alice.distanceRun = 7200)
  (h5 : bob.startPosition = 30)
  (h6 : bob.direction = false)
  (h7 : bob.distanceRun = alice.distanceRun) :
  MeetingPoint.S = 
    (let aliceFinalPosition := alice.startPosition + (alice.distanceRun % trackCircumference)
     let bobFinalPosition := bob.startPosition - (bob.distanceRun % trackCircumference)
     if aliceFinalPosition = bobFinalPosition 
     then MeetingPoint.S 
     else MeetingPoint.A) :=
by sorry

end runners_meeting_point_l4061_406105


namespace integral_sqrt_one_minus_x_squared_plus_x_l4061_406185

theorem integral_sqrt_one_minus_x_squared_plus_x (f : ℝ → ℝ) :
  (∫ x in (-1)..1, (Real.sqrt (1 - x^2) + x)) = π / 2 := by
  sorry

end integral_sqrt_one_minus_x_squared_plus_x_l4061_406185


namespace cucumber_weight_after_evaporation_l4061_406127

/-- Calculates the new weight of cucumbers after water evaporation -/
theorem cucumber_weight_after_evaporation 
  (initial_weight : ℝ) 
  (initial_water_percent : ℝ) 
  (final_water_percent : ℝ) : 
  initial_weight = 100 → 
  initial_water_percent = 99 / 100 → 
  final_water_percent = 96 / 100 → 
  ∃ (new_weight : ℝ), 
    new_weight = 25 ∧ 
    (1 - initial_water_percent) * initial_weight = (1 - final_water_percent) * new_weight :=
by sorry

end cucumber_weight_after_evaporation_l4061_406127


namespace sarah_picked_45_apples_l4061_406137

/-- The number of apples Sarah's brother picked -/
def brother_apples : ℕ := 9

/-- The factor by which Sarah picked more apples than her brother -/
def sarah_factor : ℕ := 5

/-- The number of apples Sarah picked -/
def sarah_apples : ℕ := sarah_factor * brother_apples

theorem sarah_picked_45_apples : sarah_apples = 45 := by
  sorry

end sarah_picked_45_apples_l4061_406137


namespace range_of_a_l4061_406133

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (x - a + Real.log (x / a)) * (-2 * x^2 + a * x + 10) ≤ 0) → 
  a = Real.sqrt 10 := by
sorry

end range_of_a_l4061_406133


namespace tomatoes_for_family_of_eight_l4061_406107

/-- The number of tomatoes needed to feed a family for a single meal -/
def tomatoes_needed (slices_per_tomato : ℕ) (slices_per_person : ℕ) (family_size : ℕ) : ℕ :=
  (slices_per_person * family_size) / slices_per_tomato

theorem tomatoes_for_family_of_eight :
  tomatoes_needed 8 20 8 = 20 := by
  sorry

end tomatoes_for_family_of_eight_l4061_406107


namespace standard_of_living_purchasing_power_correlated_l4061_406188

/-- Represents a person's standard of living -/
def StandardOfLiving : Type := ℝ

/-- Represents a person's purchasing power -/
def PurchasingPower : Type := ℝ

/-- Definition of correlation as a statistical relationship between two random variables -/
def Correlated (X Y : Type) : Prop := sorry

/-- Theorem stating that standard of living and purchasing power are correlated -/
theorem standard_of_living_purchasing_power_correlated :
  Correlated StandardOfLiving PurchasingPower :=
sorry

end standard_of_living_purchasing_power_correlated_l4061_406188


namespace rug_coverage_theorem_l4061_406101

/-- The total floor area covered by three overlapping rugs -/
def totalFloorArea (combinedArea twoLayerArea threeLayerArea : ℝ) : ℝ :=
  combinedArea - (twoLayerArea + 2 * threeLayerArea)

/-- Theorem stating that the total floor area covered by the rugs is 140 square meters -/
theorem rug_coverage_theorem :
  totalFloorArea 200 22 19 = 140 := by
  sorry

end rug_coverage_theorem_l4061_406101


namespace inequality_system_solution_l4061_406125

/-- Given a real number m, proves that if the solution to the system of linear inequalities
    (2x - 1 > 3(x - 2) and x < m) is x < 5, then m ≥ 5. -/
theorem inequality_system_solution (m : ℝ) : 
  (∀ x : ℝ, (2*x - 1 > 3*(x - 2) ∧ x < m) ↔ x < 5) → m ≥ 5 :=
by sorry

end inequality_system_solution_l4061_406125


namespace vector_calculation_l4061_406148

def a : Fin 3 → ℝ := ![(-3 : ℝ), 5, 2]
def b : Fin 3 → ℝ := ![(6 : ℝ), -1, -3]
def c : Fin 3 → ℝ := ![(1 : ℝ), 2, 3]

theorem vector_calculation :
  a - (4 • b) + (2 • c) = ![(-25 : ℝ), 13, 20] := by sorry

end vector_calculation_l4061_406148


namespace log_product_equality_l4061_406192

theorem log_product_equality : Real.log 3 / Real.log 8 * (Real.log 32 / Real.log 9) = 5 / 6 := by
  sorry

end log_product_equality_l4061_406192


namespace same_color_sock_pairs_l4061_406163

def total_socks : ℕ := 12
def red_socks : ℕ := 5
def green_socks : ℕ := 3
def blue_socks : ℕ := 4

theorem same_color_sock_pairs :
  (Nat.choose red_socks 2) + (Nat.choose green_socks 2) + (Nat.choose blue_socks 2) = 19 := by
  sorry

end same_color_sock_pairs_l4061_406163


namespace f_inequality_solution_f_minimum_value_l4061_406181

def f (x : ℝ) : ℝ := |2*x + 1| - |x - 4|

theorem f_inequality_solution (x : ℝ) : 
  f x > 2 ↔ x < -7 ∨ (5/3 < x ∧ x < 4) ∨ x > 7 := by sorry

theorem f_minimum_value : 
  ∃ (x : ℝ), f x = -9/2 ∧ ∀ (y : ℝ), f y ≥ -9/2 := by sorry

end f_inequality_solution_f_minimum_value_l4061_406181


namespace shane_sandwich_problem_l4061_406139

/-- The number of slices in each package of sliced bread -/
def slices_per_bread_package : ℕ := 20

/-- The number of packages of sliced bread Shane buys -/
def bread_packages : ℕ := 2

/-- The number of packages of sliced ham Shane buys -/
def ham_packages : ℕ := 2

/-- The number of ham slices in each package -/
def ham_slices_per_package : ℕ := 8

/-- The number of bread slices needed for each sandwich -/
def bread_slices_per_sandwich : ℕ := 2

/-- The number of bread slices leftover after making sandwiches -/
def leftover_bread_slices : ℕ := 8

theorem shane_sandwich_problem :
  slices_per_bread_package * bread_packages = 
    (ham_packages * ham_slices_per_package * bread_slices_per_sandwich) + leftover_bread_slices := by
  sorry

end shane_sandwich_problem_l4061_406139


namespace sufficient_condition_range_l4061_406191

theorem sufficient_condition_range (a : ℝ) : 
  (∀ x : ℝ, |x + 1| ≤ 2 → x ≤ a) ∧ 
  (∃ x : ℝ, x ≤ a ∧ |x + 1| > 2) → 
  a ≥ 1 := by
sorry

end sufficient_condition_range_l4061_406191
