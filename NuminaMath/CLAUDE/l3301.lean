import Mathlib

namespace S_intersect_T_eq_T_l3301_330129

def S : Set Int := {s | ∃ n : Int, s = 2 * n + 1}
def T : Set Int := {t | ∃ n : Int, t = 4 * n + 1}

theorem S_intersect_T_eq_T : S ∩ T = T := by
  sorry

end S_intersect_T_eq_T_l3301_330129


namespace find_number_l3301_330111

theorem find_number : ∃ x : ℝ, 11 * x + 1 = 45 ∧ x = 4 := by
  sorry

end find_number_l3301_330111


namespace max_k_value_l3301_330187

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 3 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) :
  k ≤ (Real.sqrt 7 - 1) / 2 := by sorry

end max_k_value_l3301_330187


namespace smallest_number_with_55_divisors_l3301_330182

def number_of_divisors (n : ℕ) : ℕ := sorry

theorem smallest_number_with_55_divisors :
  ∀ n : ℕ, number_of_divisors n = 55 → n ≥ 3^4 * 2^10 :=
sorry

end smallest_number_with_55_divisors_l3301_330182


namespace increase_and_subtract_l3301_330190

theorem increase_and_subtract (initial : ℝ) (increase_percent : ℝ) (subtract : ℝ) : 
  initial = 75 → increase_percent = 150 → subtract = 40 →
  initial * (1 + increase_percent / 100) - subtract = 147.5 := by
  sorry

end increase_and_subtract_l3301_330190


namespace women_to_total_ratio_in_salem_l3301_330192

/-- The population of Leesburg -/
def leesburg_population : ℕ := 58940

/-- The original population of Salem before people moved out -/
def salem_original_population : ℕ := 15 * leesburg_population

/-- The number of people who moved out of Salem -/
def people_moved_out : ℕ := 130000

/-- The new population of Salem after people moved out -/
def salem_new_population : ℕ := salem_original_population - people_moved_out

/-- The number of women living in Salem after the population change -/
def women_in_salem : ℕ := 377050

/-- The theorem stating the ratio of women to the total population in Salem -/
theorem women_to_total_ratio_in_salem :
  (women_in_salem : ℚ) / salem_new_population = 377050 / 754100 := by sorry

end women_to_total_ratio_in_salem_l3301_330192


namespace lcm_hcf_problem_l3301_330110

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 192 →
  Nat.gcd A B = 16 →
  A = 48 →
  B = 64 := by sorry

end lcm_hcf_problem_l3301_330110


namespace jared_tom_age_ratio_l3301_330102

theorem jared_tom_age_ratio : 
  ∀ (tom_future_age jared_current_age : ℕ),
    tom_future_age = 30 →
    jared_current_age = 48 →
    ∃ (jared_past_age tom_past_age : ℕ),
      jared_past_age = jared_current_age - 2 ∧
      tom_past_age = tom_future_age - 7 ∧
      jared_past_age = 2 * tom_past_age :=
by sorry

end jared_tom_age_ratio_l3301_330102


namespace ellipse_equation_and_fixed_point_l3301_330158

structure Ellipse where
  center : ℝ × ℝ
  a : ℝ
  b : ℝ

def pointOnEllipse (E : Ellipse) (p : ℝ × ℝ) : Prop :=
  (p.1 - E.center.1)^2 / E.a^2 + (p.2 - E.center.2)^2 / E.b^2 = 1

def Line (p q : ℝ × ℝ) := {r : ℝ × ℝ | ∃ t, r = (1 - t) • p + t • q}

theorem ellipse_equation_and_fixed_point 
  (E : Ellipse)
  (h_center : E.center = (0, 0))
  (h_A : pointOnEllipse E (0, -2))
  (h_B : pointOnEllipse E (3/2, -1)) :
  (E.a^2 = 3 ∧ E.b^2 = 4) ∧
  ∀ (P M N T H : ℝ × ℝ),
    P = (1, -2) →
    pointOnEllipse E M →
    pointOnEllipse E N →
    M ∈ Line P N →
    T.1 = M.1 ∧ T ∈ Line (0, -2) (3/2, -1) →
    H.1 - T.1 = T.1 - M.1 ∧ H.2 = T.2 →
    (0, -2) ∈ Line H N :=
by sorry

end ellipse_equation_and_fixed_point_l3301_330158


namespace vegetable_seedling_price_l3301_330100

theorem vegetable_seedling_price (base_price : ℚ) : 
  (300 / base_price - 300 / (5/4 * base_price) = 3) → base_price = 20 := by
  sorry

end vegetable_seedling_price_l3301_330100


namespace eighteen_power_equality_l3301_330157

theorem eighteen_power_equality (m n : ℤ) (P Q : ℝ) 
  (hP : P = 2^m) (hQ : Q = 3^n) : 
  18^(m+n) = P^(m+n) * Q^(2*(m+n)) := by
  sorry

end eighteen_power_equality_l3301_330157


namespace hyperbola_focus_asymptote_distance_l3301_330186

/-- Hyperbola C with equation x²/a² - y²/b² = 1 (a > 0, b > 0) -/
structure Hyperbola (a b : ℝ) : Prop where
  a_pos : a > 0
  b_pos : b > 0

/-- Line l with equation y = 2x - 2 -/
def Line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = 2 * p.1 - 2}

/-- The asymptote of the hyperbola C -/
def asymptote (h : Hyperbola a b) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = (b / a) * p.1 ∨ p.2 = -(b / a) * p.1}

/-- Predicate to check if a line is parallel to an asymptote -/
def is_parallel_to_asymptote (h : Hyperbola a b) : Prop :=
  b / a = 2

/-- Predicate to check if a line passes through a vertex of the hyperbola -/
def passes_through_vertex (h : Hyperbola a b) : Prop :=
  (1, 0) ∈ Line

/-- The distance from the focus of the hyperbola to its asymptote -/
def focus_asymptote_distance (h : Hyperbola a b) : ℝ := b

/-- Main theorem -/
theorem hyperbola_focus_asymptote_distance 
  (h : Hyperbola a b) 
  (parallel : is_parallel_to_asymptote h) 
  (through_vertex : passes_through_vertex h) : 
  focus_asymptote_distance h = 2 := by
    sorry

end hyperbola_focus_asymptote_distance_l3301_330186


namespace total_sum_lent_is_2769_l3301_330130

/-- Calculates the total sum lent given the conditions of the problem -/
def totalSumLent (secondPart : ℕ) : ℕ :=
  let firstPart := (secondPart * 5) / 8
  firstPart + secondPart

/-- Proves that the total sum lent is 2769 given the problem conditions -/
theorem total_sum_lent_is_2769 :
  totalSumLent 1704 = 2769 := by
  sorry

#eval totalSumLent 1704

end total_sum_lent_is_2769_l3301_330130


namespace x_less_than_y_less_than_zero_l3301_330196

theorem x_less_than_y_less_than_zero (x y : ℝ) 
  (h1 : x^2 - y^2 > 2*x) 
  (h2 : x*y < y) : 
  x < y ∧ y < 0 := by
sorry

end x_less_than_y_less_than_zero_l3301_330196


namespace egyptian_fraction_sum_l3301_330160

theorem egyptian_fraction_sum : ∃! (b₂ b₃ b₄ b₅ : ℤ),
  (3 : ℚ) / 5 = (b₂ : ℚ) / 2 + (b₃ : ℚ) / 6 + (b₄ : ℚ) / 24 + (b₅ : ℚ) / 120 ∧
  (0 ≤ b₂ ∧ b₂ < 2) ∧
  (0 ≤ b₃ ∧ b₃ < 3) ∧
  (0 ≤ b₄ ∧ b₄ < 4) ∧
  (0 ≤ b₅ ∧ b₅ < 5) ∧
  b₂ + b₃ + b₄ + b₅ = 5 := by
sorry

end egyptian_fraction_sum_l3301_330160


namespace total_interest_after_trebling_l3301_330120

/-- 
Given a principal amount and an interest rate, if the simple interest 
on the principal for 10 years is 700, and the principal is trebled after 5 years, 
then the total interest at the end of the tenth year is 1750.
-/
theorem total_interest_after_trebling (P R : ℝ) : 
  (P * R * 10) / 100 = 700 → 
  ((P * R * 5) / 100) + (((3 * P) * R * 5) / 100) = 1750 := by
  sorry

#check total_interest_after_trebling

end total_interest_after_trebling_l3301_330120


namespace base_value_l3301_330154

theorem base_value (some_base : ℕ) : 
  (1/2)^16 * (1/81)^8 = 1/(some_base^16) → some_base = 18 := by sorry

end base_value_l3301_330154


namespace diamond_computation_l3301_330148

-- Define the set of elements
inductive Element : Type
  | one : Element
  | two : Element
  | three : Element
  | four : Element
  | five : Element

-- Define the operation
def diamond : Element → Element → Element
  | Element.one, Element.one => Element.two
  | Element.one, Element.two => Element.one
  | Element.one, Element.three => Element.three
  | Element.one, Element.four => Element.five
  | Element.one, Element.five => Element.four
  | Element.two, Element.one => Element.one
  | Element.two, Element.two => Element.five
  | Element.two, Element.three => Element.four
  | Element.two, Element.four => Element.three
  | Element.two, Element.five => Element.two
  | Element.three, Element.one => Element.three
  | Element.three, Element.two => Element.four
  | Element.three, Element.three => Element.two
  | Element.three, Element.four => Element.one
  | Element.three, Element.five => Element.five
  | Element.four, Element.one => Element.five
  | Element.four, Element.two => Element.two
  | Element.four, Element.three => Element.one
  | Element.four, Element.four => Element.four
  | Element.four, Element.five => Element.three
  | Element.five, Element.one => Element.four
  | Element.five, Element.two => Element.three
  | Element.five, Element.three => Element.five
  | Element.five, Element.four => Element.two
  | Element.five, Element.five => Element.one

theorem diamond_computation :
  diamond (diamond Element.four Element.five) (diamond Element.one Element.three) = Element.two :=
by sorry

end diamond_computation_l3301_330148


namespace fruit_distribution_ways_l3301_330153

def num_apples : ℕ := 2
def num_pears : ℕ := 3
def num_days : ℕ := 5

theorem fruit_distribution_ways :
  (Nat.choose num_days num_apples) = 10 := by
  sorry

end fruit_distribution_ways_l3301_330153


namespace complement_intersection_equals_set_l3301_330133

open Set

def U : Set ℕ := {1,2,3,4,5,6,7,8}
def P : Set ℕ := {3,4,5}
def Q : Set ℕ := {1,3,6}

theorem complement_intersection_equals_set : (U \ P) ∩ (U \ Q) = {2,7,8} := by sorry

end complement_intersection_equals_set_l3301_330133


namespace rotation_180_maps_points_l3301_330188

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Rotates a point 180 degrees clockwise around the origin -/
def rotate180 (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- The theorem stating that rotating the given points 180 degrees clockwise
    results in the expected transformed points -/
theorem rotation_180_maps_points :
  let C : Point := { x := 3, y := -2 }
  let D : Point := { x := 2, y := -5 }
  let C' : Point := { x := -3, y := 2 }
  let D' : Point := { x := -2, y := 5 }
  rotate180 C = C' ∧ rotate180 D = D' := by
  sorry

end rotation_180_maps_points_l3301_330188


namespace rectangle_width_decrease_l3301_330170

theorem rectangle_width_decrease (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let L' := 1.4 * L
  let W' := W * L / L'
  (W - W') / W = 2 / 7 := by sorry

end rectangle_width_decrease_l3301_330170


namespace expand_product_l3301_330131

theorem expand_product (y : ℝ) : 5 * (y - 3) * (y + 10) = 5 * y^2 + 35 * y - 150 := by
  sorry

end expand_product_l3301_330131


namespace first_train_length_l3301_330143

/-- The length of a train given its speed, the speed and length of an oncoming train, and the time they take to cross each other. -/
def trainLength (speed1 : ℝ) (speed2 : ℝ) (length2 : ℝ) (crossTime : ℝ) : ℝ :=
  (speed1 + speed2) * crossTime - length2

/-- Theorem stating the length of the first train given the problem conditions -/
theorem first_train_length :
  let speed1 := 120 * (1000 / 3600)  -- Convert 120 km/hr to m/s
  let speed2 := 80 * (1000 / 3600)   -- Convert 80 km/hr to m/s
  let length2 := 230.04              -- Length of the second train in meters
  let crossTime := 9                 -- Time to cross in seconds
  trainLength speed1 speed2 length2 crossTime = 269.96 := by
  sorry

end first_train_length_l3301_330143


namespace geometric_sequence_third_term_l3301_330127

/-- For a geometric sequence with positive real terms, if a_1 = 1 and a_5 = 9, then a_3 = 3 -/
theorem geometric_sequence_third_term 
  (a : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_pos : ∀ n, a n > 0) 
  (h_a1 : a 1 = 1) 
  (h_a5 : a 5 = 9) : 
  a 3 = 3 := by
sorry

end geometric_sequence_third_term_l3301_330127


namespace coefficient_x4_proof_l3301_330119

/-- The coefficient of x^4 in the expansion of (x - 1/(2x))^10 -/
def coefficient_x4 : ℤ := -15

/-- The binomial coefficient (n choose k) -/
def binomial (n k : ℕ) : ℕ := sorry

theorem coefficient_x4_proof :
  coefficient_x4 = binomial 10 3 * (-1/2)^3 := by sorry

end coefficient_x4_proof_l3301_330119


namespace balloons_given_to_fred_l3301_330105

/-- Given that Tom initially had 30 balloons and now has 14 balloons,
    prove that he gave 16 balloons to Fred. -/
theorem balloons_given_to_fred 
  (initial_balloons : ℕ) 
  (remaining_balloons : ℕ) 
  (h1 : initial_balloons = 30) 
  (h2 : remaining_balloons = 14) : 
  initial_balloons - remaining_balloons = 16 :=
by sorry

end balloons_given_to_fred_l3301_330105


namespace marble_difference_l3301_330191

theorem marble_difference (total : ℕ) (bag_a : ℕ) (bag_b : ℕ) : 
  total = 72 → bag_a = 42 → bag_b = total - bag_a → bag_a - bag_b = 12 := by
  sorry

end marble_difference_l3301_330191


namespace statue_selling_price_l3301_330122

/-- The selling price of a statue given its cost and profit percentage -/
def selling_price (cost : ℝ) (profit_percentage : ℝ) : ℝ :=
  cost * (1 + profit_percentage)

/-- Theorem: The selling price of a statue that costs $536 and is sold at a 25% profit is $670 -/
theorem statue_selling_price : 
  selling_price 536 0.25 = 670 := by
  sorry

end statue_selling_price_l3301_330122


namespace necessary_condition_not_sufficient_l3301_330128

def f (x : ℝ) := |x - 2| + |x + 3|

def proposition_p (a : ℝ) := ∃ x, f x < a

theorem necessary_condition (a : ℝ) :
  (¬ proposition_p a) → a ≥ 5 := by sorry

theorem not_sufficient (a : ℝ) :
  ∃ a, a ≥ 5 ∧ proposition_p a := by sorry

end necessary_condition_not_sufficient_l3301_330128


namespace problem_statement_l3301_330198

open Set

def p (m : ℝ) : Prop := ∀ x, ∃ y, y = Real.log (m * x^2 - m * x + 1)

def q (m : ℝ) : Prop := ∃ x₀ ∈ Icc 0 3, x₀^2 - 2*x₀ - m ≥ 0

theorem problem_statement (m : ℝ) :
  (q m ↔ m ∈ Iic 3) ∧
  ((p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ m ∈ Iio 0 ∪ Ioo 3 4) :=
sorry

end problem_statement_l3301_330198


namespace polynomial_evaluation_l3301_330121

theorem polynomial_evaluation (y : ℝ) (h1 : y > 0) (h2 : y^2 - 3*y - 9 = 0) :
  y^3 - 3*y^2 - 9*y + 5 = 5 := by
sorry

end polynomial_evaluation_l3301_330121


namespace tan_product_seventh_roots_l3301_330137

theorem tan_product_seventh_roots : 
  Real.tan (π / 7) * Real.tan (2 * π / 7) * Real.tan (3 * π / 7) = Real.sqrt 7 := by
  sorry

end tan_product_seventh_roots_l3301_330137


namespace number_subtraction_problem_l3301_330169

theorem number_subtraction_problem (x y : ℝ) 
  (h1 : (x - 5) / 7 = 7)
  (h2 : (x - y) / 10 = 5) : 
  y = 4 := by
  sorry

end number_subtraction_problem_l3301_330169


namespace fixed_point_of_arithmetic_sequence_l3301_330147

/-- If k, -1, and b form an arithmetic sequence, then the line y = kx + b passes through the point (1, -2). -/
theorem fixed_point_of_arithmetic_sequence (k b : ℝ) : 
  (k - (-1) = (-1) - b) → 
  ∃ (y : ℝ), y = k * 1 + b ∧ y = -2 :=
by sorry

end fixed_point_of_arithmetic_sequence_l3301_330147


namespace negative_seven_x_is_product_l3301_330179

theorem negative_seven_x_is_product : 
  ∀ x : ℝ, -7 * x = (-7) * x :=
by
  sorry

end negative_seven_x_is_product_l3301_330179


namespace solve_system_l3301_330126

theorem solve_system (A B C D : ℤ) 
  (eq1 : A + C = 15)
  (eq2 : A - B = 1)
  (eq3 : C + C = A)
  (eq4 : B - D = 2)
  (diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  D = 7 := by
sorry

end solve_system_l3301_330126


namespace tina_shoe_expense_l3301_330108

def savings_june : ℕ := 27
def savings_july : ℕ := 14
def savings_august : ℕ := 21
def spent_on_books : ℕ := 5
def amount_left : ℕ := 40

theorem tina_shoe_expense : 
  savings_june + savings_july + savings_august - spent_on_books - amount_left = 17 := by
  sorry

end tina_shoe_expense_l3301_330108


namespace bug_movement_l3301_330183

/-- Probability of the bug being at the starting vertex after n moves -/
def Q (n : ℕ) : ℚ :=
  if n = 0 then 1
  else (1 / 3) * (1 - Q (n - 1))

/-- The bug's movement on a square -/
theorem bug_movement :
  Q 8 = 547 / 2187 :=
sorry

end bug_movement_l3301_330183


namespace decreasing_function_implies_a_leq_neg_three_l3301_330197

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

-- State the theorem
theorem decreasing_function_implies_a_leq_neg_three :
  ∀ a : ℝ, (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ 4 → f a x₁ > f a x₂) → a ≤ -3 :=
by sorry

end decreasing_function_implies_a_leq_neg_three_l3301_330197


namespace daniels_noodles_l3301_330166

def noodles_problem (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : Prop :=
  initial = given_away + remaining

theorem daniels_noodles : ∃ initial : ℕ, noodles_problem initial 12 54 ∧ initial = 66 := by
  sorry

end daniels_noodles_l3301_330166


namespace right_triangle_BD_length_l3301_330145

-- Define the triangle and its properties
structure RightTriangle where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  D : ℝ  -- Represents the position of D on BC
  hAB : AB = 45
  hAC : AC = 60
  hBC : BC^2 = AB^2 + AC^2
  hD : 0 < D ∧ D < BC

-- Define the theorem
theorem right_triangle_BD_length (t : RightTriangle) : 
  let AD := (t.AB * t.AC) / t.BC
  let BD := Real.sqrt (t.AB^2 - AD^2)
  BD = 27 := by sorry

end right_triangle_BD_length_l3301_330145


namespace symmetric_f_inequality_solution_l3301_330178

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + 1| + |x - a|

-- State the theorem
theorem symmetric_f_inequality_solution (a : ℝ) :
  (∀ x : ℝ, f a x = f a (2 - x)) →
  {x : ℝ | f a (x^2 - 3) < f a (x - 1)} = {x : ℝ | -3 < x ∧ x < -1} := by
  sorry

end symmetric_f_inequality_solution_l3301_330178


namespace terrys_test_score_l3301_330124

theorem terrys_test_score (total_problems : ℕ) (total_score : ℕ) 
  (correct_points : ℕ) (incorrect_points : ℕ) :
  total_problems = 25 →
  total_score = 85 →
  correct_points = 4 →
  incorrect_points = 1 →
  ∃ (correct incorrect : ℕ),
    correct + incorrect = total_problems ∧
    correct_points * correct - incorrect_points * incorrect = total_score ∧
    incorrect = 3 := by
  sorry

end terrys_test_score_l3301_330124


namespace chemical_solution_replacement_exists_l3301_330162

theorem chemical_solution_replacement_exists : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ 
  (1 - x)^5 * 0.5 + x * (0.6 + 0.65 + 0.7 + 0.75 + 0.8) = 0.63 := by
  sorry

end chemical_solution_replacement_exists_l3301_330162


namespace doris_babysitting_earnings_l3301_330174

/-- Represents the problem of calculating how many weeks Doris needs to earn enough for her monthly expenses --/
theorem doris_babysitting_earnings :
  let hourly_rate : ℚ := 20
  let weekday_hours : ℚ := 3
  let saturday_hours : ℚ := 5
  let monthly_expense : ℚ := 1200
  let weekly_hours := weekday_hours * 5 + saturday_hours
  let weekly_earnings := hourly_rate * weekly_hours
  let weeks_needed := monthly_expense / weekly_earnings
  weeks_needed = 3 := by sorry

end doris_babysitting_earnings_l3301_330174


namespace bus_schedule_theorem_l3301_330115

def is_valid_interval (T : ℚ) : Prop :=
  T < 30 ∧
  T > 0 ∧
  ∀ k : ℤ, ∀ t₀ : ℚ, 0 ≤ t₀ ∧ t₀ < T →
    (¬ (0 ≤ (t₀ + k * T) % 60 ∧ (t₀ + k * T) % 60 < 5)) ∧
    (¬ (38 ≤ (t₀ + k * T) % 60 ∧ (t₀ + k * T) % 60 < 43))

def valid_intervals : Set ℚ := {20, 15, 12, 10, 7.5, 5 + 5/11}

theorem bus_schedule_theorem :
  ∀ T : ℚ, is_valid_interval T ↔ T ∈ valid_intervals :=
sorry

end bus_schedule_theorem_l3301_330115


namespace power_fraction_minus_one_l3301_330134

theorem power_fraction_minus_one : (5 / 3 : ℚ) ^ 4 - 1 = 544 / 81 := by sorry

end power_fraction_minus_one_l3301_330134


namespace cone_volume_from_half_sector_l3301_330164

/-- The volume of a cone formed by rolling up a half-sector of a circle --/
theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) : 
  let base_radius : ℝ := r / 2
  let cone_height : ℝ := Real.sqrt (r^2 - base_radius^2)
  (1/3 : ℝ) * Real.pi * base_radius^2 * cone_height = 9 * Real.pi * Real.sqrt 3 :=
by sorry

end cone_volume_from_half_sector_l3301_330164


namespace complex_equation_solution_l3301_330189

theorem complex_equation_solution (z : ℂ) : z * Complex.I = 1 + Complex.I → z = 1 - Complex.I := by
  sorry

end complex_equation_solution_l3301_330189


namespace smallest_square_area_l3301_330140

/-- The smallest square area containing two non-overlapping rectangles -/
theorem smallest_square_area (r1_width r1_height r2_width r2_height : ℕ) 
  (h1 : r1_width = 2 ∧ r1_height = 4)
  (h2 : r2_width = 3 ∧ r2_height = 5) : 
  (max (r1_width + r2_width) (max r1_height r2_height))^2 = 36 := by
  sorry

#check smallest_square_area

end smallest_square_area_l3301_330140


namespace solve_proportion_l3301_330161

theorem solve_proportion (x y : ℚ) (h1 : x / y = 8 / 3) (h2 : y = 27) : x = 72 := by
  sorry

end solve_proportion_l3301_330161


namespace intersection_of_M_and_N_l3301_330159

def M : Set ℕ := {1, 2, 3, 5}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by
  sorry

end intersection_of_M_and_N_l3301_330159


namespace solution_transformation_l3301_330138

theorem solution_transformation (k : ℤ) (x y : ℤ) 
  (h1 : ∃ n : ℤ, 15 * k = n) 
  (h2 : x^2 - 2*y^2 = k) : 
  ∃ t u : ℤ, t^2 - 2*u^2 = -k ∧ 
  ((t = x + 2*y ∧ u = x + y) ∨ (t = x - 2*y ∧ u = x - y)) := by
  sorry

end solution_transformation_l3301_330138


namespace range_of_a_for_unique_integer_solution_l3301_330163

/-- Given a system of inequalities, prove the range of a for which there is exactly one integer solution. -/
theorem range_of_a_for_unique_integer_solution (a : ℝ) : 
  (∃! (x : ℤ), (x^3 + 3*x^2 - x - 3 > 0) ∧ 
                (x^2 - 2*a*x - 1 ≤ 0) ∧ 
                (a > 0)) ↔ 
  (3/4 ≤ a ∧ a < 4/3) := by
sorry

end range_of_a_for_unique_integer_solution_l3301_330163


namespace fraction_meaningful_l3301_330184

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = (x - 1) / (2 - x)) ↔ x ≠ 2 := by sorry

end fraction_meaningful_l3301_330184


namespace complex_magnitude_l3301_330185

/-- Given two complex numbers z₁ and z₂, where z₁/z₂ is purely imaginary,
    prove that the magnitude of z₁ is 10/3. -/
theorem complex_magnitude (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 3 - 4*I
  (∃ (b : ℝ), z₁ / z₂ = b*I) → Complex.abs z₁ = 10/3 := by
sorry

end complex_magnitude_l3301_330185


namespace no_valid_n_l3301_330116

theorem no_valid_n : ¬∃ (n : ℕ), 
  (n > 0) ∧ 
  (1000 ≤ n / 4) ∧ (n / 4 ≤ 9999) ∧ 
  (1000 ≤ 4 * n) ∧ (4 * n ≤ 9999) := by
  sorry

end no_valid_n_l3301_330116


namespace arrangement_theorem_l3301_330142

/-- The number of ways to arrange 2 men and 4 women in a row, 
    such that no two men or two women are adjacent -/
def arrangement_count : ℕ := 240

/-- The number of positions between and at the ends of the men -/
def women_positions : ℕ := 5

/-- The number of men -/
def num_men : ℕ := 2

/-- The number of women -/
def num_women : ℕ := 4

theorem arrangement_theorem : 
  arrangement_count = women_positions.choose num_women * num_women.factorial * num_men.factorial :=
sorry

end arrangement_theorem_l3301_330142


namespace box_dimensions_solution_l3301_330152

/-- Represents the dimensions of a box --/
structure BoxDimensions where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : a + c = 17
  h2 : a + b = 13
  h3 : b + c = 20
  h4 : a < b
  h5 : b < c

/-- Proves that the dimensions of the box are 5, 8, and 12 --/
theorem box_dimensions_solution (box : BoxDimensions) : 
  box.a = 5 ∧ box.b = 8 ∧ box.c = 12 := by
  sorry

end box_dimensions_solution_l3301_330152


namespace negation_of_universal_proposition_l3301_330167

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x ≥ 0 → x^3 + x ≥ 0) ↔ ∃ x : ℝ, x ≥ 0 ∧ x^3 + x < 0 := by
  sorry

end negation_of_universal_proposition_l3301_330167


namespace correct_forecast_interpretation_l3301_330112

-- Define the probability of rainfall
def rainfall_probability : ℝ := 0.9

-- Define the event of getting wet when going out without rain gear
def might_get_wet (p : ℝ) : Prop :=
  p > 0 ∧ p < 1

-- Theorem statement
theorem correct_forecast_interpretation :
  might_get_wet rainfall_probability := by
  sorry

end correct_forecast_interpretation_l3301_330112


namespace card_game_combinations_l3301_330125

theorem card_game_combinations : Nat.choose 52 10 = 158200242220 := by sorry

end card_game_combinations_l3301_330125


namespace sarah_trip_distance_l3301_330150

/-- Represents Sarah's trip to the airport -/
structure AirportTrip where
  initial_speed : ℝ
  initial_time : ℝ
  final_speed : ℝ
  early_arrival : ℝ
  total_distance : ℝ

/-- The theorem stating the total distance of Sarah's trip -/
theorem sarah_trip_distance (trip : AirportTrip) : 
  trip.initial_speed = 15 ∧ 
  trip.initial_time = 1 ∧ 
  trip.final_speed = 60 ∧ 
  trip.early_arrival = 0.5 →
  trip.total_distance = 45 := by
  sorry

#check sarah_trip_distance

end sarah_trip_distance_l3301_330150


namespace set_relationship_l3301_330151

def P : Set ℝ := {y | ∃ x, y = -x^2 + 1}
def Q : Set ℝ := {y | ∃ x, y = 2^x}

theorem set_relationship : ∀ y : ℝ, y > 1 → y ∈ Q := by
  sorry

end set_relationship_l3301_330151


namespace fraction_inequality_l3301_330113

theorem fraction_inequality (x : ℝ) (h : x ≠ -5) :
  (x - 2) / (x + 5) ≥ 0 ↔ x < -5 ∨ x ≥ 2 := by
  sorry

end fraction_inequality_l3301_330113


namespace bread_needed_for_field_trip_bread_needed_proof_l3301_330149

/-- Calculates the number of pieces of bread needed for a field trip --/
theorem bread_needed_for_field_trip 
  (sandwiches_per_student : ℕ) 
  (students_per_group : ℕ) 
  (number_of_groups : ℕ) 
  (bread_per_sandwich : ℕ) : ℕ :=
  let total_students := students_per_group * number_of_groups
  let total_sandwiches := total_students * sandwiches_per_student
  let total_bread := total_sandwiches * bread_per_sandwich
  total_bread

/-- Proves that 120 pieces of bread are needed for the field trip --/
theorem bread_needed_proof :
  bread_needed_for_field_trip 2 6 5 2 = 120 := by
  sorry

end bread_needed_for_field_trip_bread_needed_proof_l3301_330149


namespace train_length_calculation_l3301_330156

/-- Given a train that crosses a platform in 54 seconds and a signal pole in 18 seconds,
    where the platform length is 600.0000000000001 meters, prove that the length of the train
    is 300.00000000000005 meters. -/
theorem train_length_calculation (platform_crossing_time : ℝ) (pole_crossing_time : ℝ) (platform_length : ℝ)
    (h1 : platform_crossing_time = 54)
    (h2 : pole_crossing_time = 18)
    (h3 : platform_length = 600.0000000000001) :
    ∃ (train_length : ℝ), train_length = 300.00000000000005 :=
by sorry

end train_length_calculation_l3301_330156


namespace special_sequence_common_difference_l3301_330139

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  first_term : ℝ
  last_term : ℝ
  sum : ℝ
  num_terms : ℕ
  common_difference : ℝ

/-- Properties of the specific arithmetic sequence -/
def special_sequence : ArithmeticSequence where
  first_term := 5
  last_term := 50
  sum := 275
  num_terms := 10  -- Derived from the solution, but could be proved
  common_difference := 5  -- This is what we want to prove

/-- Theorem stating that the common difference of the special sequence is 5 -/
theorem special_sequence_common_difference :
  (special_sequence.common_difference = 5) ∧
  (special_sequence.first_term = 5) ∧
  (special_sequence.last_term = 50) ∧
  (special_sequence.sum = 275) := by
  sorry

#check special_sequence_common_difference

end special_sequence_common_difference_l3301_330139


namespace two_numbers_difference_l3301_330199

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 80) :
  |x - y| = 8 := by sorry

end two_numbers_difference_l3301_330199


namespace min_value_reciprocal_sum_l3301_330155

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 1) :
  (1 / x + 1 / y) ≥ 5 + 3 * Real.sqrt 3 ∧
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 3 * y₀ = 1 ∧ 1 / x₀ + 1 / y₀ = 5 + 3 * Real.sqrt 3 :=
sorry

end min_value_reciprocal_sum_l3301_330155


namespace clock_rings_count_l3301_330175

def clock_rings (hour : ℕ) : Bool :=
  if hour ≤ 12 then
    hour % 2 = 1
  else
    hour % 4 = 1

def total_rings : ℕ :=
  (List.range 24).filter (λ h => clock_rings (h + 1)) |>.length

theorem clock_rings_count : total_rings = 10 := by
  sorry

end clock_rings_count_l3301_330175


namespace point_in_second_quadrant_l3301_330177

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate. -/
def second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

/-- The point P with coordinates (-3, a^2 + 1) lies in the second quadrant for any real number a. -/
theorem point_in_second_quadrant (a : ℝ) : second_quadrant (-3, a^2 + 1) := by
  sorry

end point_in_second_quadrant_l3301_330177


namespace ceiling_minus_x_zero_l3301_330180

theorem ceiling_minus_x_zero (x : ℝ) (h : ⌈x⌉ - ⌊x⌋ = 0) : ⌈x⌉ - x = 0 := by
  sorry

end ceiling_minus_x_zero_l3301_330180


namespace f_monotone_increasing_l3301_330135

-- Define the function f and its derivative
noncomputable def f : ℝ → ℝ := sorry

-- State the derivative condition
axiom f_derivative (x : ℝ) : deriv f x = x * (1 - x)

-- Theorem statement
theorem f_monotone_increasing : MonotoneOn f (Set.Icc 0 1) := by sorry

end f_monotone_increasing_l3301_330135


namespace trigonometric_identity_angle_relation_l3301_330176

-- Part 1
theorem trigonometric_identity :
  Real.sin (120 * π / 180) ^ 2 + Real.cos (180 * π / 180) + Real.tan (45 * π / 180) -
  Real.cos (-330 * π / 180) ^ 2 + Real.sin (-210 * π / 180) = 1 / 2 := by sorry

-- Part 2
theorem angle_relation (α β : Real) (h1 : 0 < α) (h2 : α < π) (h3 : 0 < β) (h4 : β < π)
  (h5 : Real.tan (α - β) = 1 / 2) (h6 : Real.tan β = -1 / 7) :
  2 * α - β = -3 * π / 4 := by sorry

end trigonometric_identity_angle_relation_l3301_330176


namespace dollar_op_five_neg_two_l3301_330136

def dollar_op (x y : ℤ) : ℤ := x * (2 * y - 1) + 2 * x * y

theorem dollar_op_five_neg_two : dollar_op 5 (-2) = -45 := by
  sorry

end dollar_op_five_neg_two_l3301_330136


namespace point_movement_on_number_line_l3301_330168

theorem point_movement_on_number_line (m : ℝ) : 
  (|m - 3 + 5| = 6) → (m = -8 ∨ m = 4) := by
  sorry

end point_movement_on_number_line_l3301_330168


namespace relay_race_time_l3301_330195

-- Define the runners and their properties
structure Runner where
  base_time : ℝ
  obstacle_time : ℝ
  handicap : ℝ

def rhonda : Runner :=
  { base_time := 24
  , obstacle_time := 2
  , handicap := 0.95 }

def sally : Runner :=
  { base_time := 26
  , obstacle_time := 5
  , handicap := 0.90 }

def diane : Runner :=
  { base_time := 21
  , obstacle_time := 21 * 0.1
  , handicap := 1.05 }

-- Calculate the final time for a runner
def final_time (runner : Runner) : ℝ :=
  (runner.base_time + runner.obstacle_time) * runner.handicap

-- Calculate the total time for the relay race
def relay_time : ℝ :=
  final_time rhonda + final_time sally + final_time diane

-- Theorem statement
theorem relay_race_time : relay_time = 76.855 := by
  sorry

end relay_race_time_l3301_330195


namespace existence_of_x0_l3301_330146

theorem existence_of_x0 (a b : ℝ) :
  ∃ x₀ ∈ Set.Icc 1 9, |a * x₀ + b + 9 / x₀| ≥ 2 := by
  sorry

end existence_of_x0_l3301_330146


namespace square_equation_solution_l3301_330173

theorem square_equation_solution : 
  ∀ x y : ℕ+, x^2 = y^2 + 7*y + 6 ↔ x = 6 ∧ y = 3 := by sorry

end square_equation_solution_l3301_330173


namespace product_of_repeating_decimal_and_nine_l3301_330106

theorem product_of_repeating_decimal_and_nine (q : ℚ) : 
  (∃ (n : ℕ), q * (100 : ℚ) - q = (45 : ℚ) + n * (100 : ℚ)) → q * 9 = 45 / 11 := by
  sorry

end product_of_repeating_decimal_and_nine_l3301_330106


namespace parabola_focus_l3301_330144

/-- A parabola is defined by its equation in the form y = ax^2, where a is a non-zero real number. -/
structure Parabola where
  a : ℝ
  a_nonzero : a ≠ 0

/-- The focus of a parabola is a point (h, k) where h and k are real numbers. -/
structure Focus where
  h : ℝ
  k : ℝ

/-- Given a parabola y = -1/8 * x^2, its focus is at the point (0, -2). -/
theorem parabola_focus (p : Parabola) (h : p.a = -1/8) : 
  ∃ (f : Focus), f.h = 0 ∧ f.k = -2 := by
  sorry

end parabola_focus_l3301_330144


namespace segments_in_proportion_l3301_330123

/-- A set of four line segments is in proportion if the product of the extremes
    equals the product of the means. -/
def is_in_proportion (a b c d : ℝ) : Prop :=
  a * d = b * c

/-- The set of line segments (2, 3, 4, 6) is in proportion. -/
theorem segments_in_proportion :
  is_in_proportion 2 3 4 6 := by
  sorry

end segments_in_proportion_l3301_330123


namespace product_sum_equality_l3301_330118

theorem product_sum_equality : 25 * 13 * 2 + 15 * 13 * 7 = 2015 := by
  sorry

end product_sum_equality_l3301_330118


namespace expression_value_l3301_330181

theorem expression_value (x : ℝ) (h : x^2 + 2*x = 1) :
  (1 - x)^2 - (x + 3)*(3 - x) - (x - 3)*(x - 1) = -10 := by
  sorry

end expression_value_l3301_330181


namespace smallest_of_seven_consecutive_evens_l3301_330109

/-- Given seven consecutive even integers whose sum is 448, 
    the smallest of these numbers is 58. -/
theorem smallest_of_seven_consecutive_evens (a : ℤ) : 
  (∃ n : ℤ, a = 2*n ∧ 
   (a + (a+2) + (a+4) + (a+6) + (a+8) + (a+10) + (a+12) = 448)) → 
  a = 58 := by
sorry

end smallest_of_seven_consecutive_evens_l3301_330109


namespace hyperbola_eccentricity_l3301_330104

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptotes x ± 2y = 0 is √5/2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a ≠ 0) (k : b ≠ 0) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  let asymptote (x y : ℝ) := x = 2 * y ∨ x = -2 * y
  asymptote x y ∧ x^2 / a^2 - y^2 / b^2 = 1 → e = Real.sqrt 5 / 2 :=
by sorry

end hyperbola_eccentricity_l3301_330104


namespace intersection_complement_equals_set_l3301_330141

def I : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 6}
def N : Set ℕ := {2, 3, 4}

theorem intersection_complement_equals_set : M ∩ (I \ N) = {1, 6} := by
  sorry

end intersection_complement_equals_set_l3301_330141


namespace inequality_theorem_l3301_330103

theorem inequality_theorem (n : ℕ) (hn : n > 0) :
  (∀ x : ℝ, x > 0 → x + (n^n : ℝ) / x^n ≥ n + 1) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → x + a / x^n ≥ n + 1) → a = n^n) :=
by sorry

end inequality_theorem_l3301_330103


namespace solution_is_121_l3301_330194

/-- Sum of first n odd numbers -/
def sumOddNumbers (n : ℕ) : ℕ := n^2

/-- Sum of first n even numbers -/
def sumEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The equation from the problem -/
def equation (n : ℕ) : Prop :=
  (sumOddNumbers n : ℚ) / (sumEvenNumbers n : ℚ) = 121 / 122

theorem solution_is_121 : ∃ (n : ℕ), n > 0 ∧ equation n ∧ n = 121 := by
  sorry

end solution_is_121_l3301_330194


namespace parallelogram_height_l3301_330132

theorem parallelogram_height (area base height : ℝ) : 
  area = 96 ∧ base = 12 ∧ area = base * height → height = 8 := by sorry

end parallelogram_height_l3301_330132


namespace cheese_slices_lcm_l3301_330193

theorem cheese_slices_lcm : 
  let cheddar_slices : ℕ := 12
  let swiss_slices : ℕ := 28
  let gouda_slices : ℕ := 18
  Nat.lcm (Nat.lcm cheddar_slices swiss_slices) gouda_slices = 252 := by
  sorry

end cheese_slices_lcm_l3301_330193


namespace joan_grew_29_carrots_l3301_330114

/-- The number of carrots Joan grew -/
def joans_carrots : ℕ := sorry

/-- The number of watermelons Joan grew -/
def joans_watermelons : ℕ := 14

/-- The number of carrots Jessica grew -/
def jessicas_carrots : ℕ := 11

/-- The total number of carrots grown by Joan and Jessica -/
def total_carrots : ℕ := 40

/-- Theorem stating that Joan grew 29 carrots -/
theorem joan_grew_29_carrots : joans_carrots = 29 := by
  sorry

end joan_grew_29_carrots_l3301_330114


namespace f_range_l3301_330107

-- Define the closest multiple function
def closestMultiple (k : ℤ) (n : ℕ) : ℤ :=
  let m := (2 * n + 1 : ℤ)
  m * ((k + m / 2) / m)

-- Define the function f
def f (k : ℤ) : ℤ :=
  closestMultiple k 1 + closestMultiple (2 * k) 2 + closestMultiple (3 * k) 3 - 6 * k

-- State the theorem
theorem f_range :
  ∀ k : ℤ, -6 ≤ f k ∧ f k ≤ 6 :=
sorry

end f_range_l3301_330107


namespace total_days_calculation_l3301_330171

/-- Represents the total number of days needed to listen to a record collection. -/
def total_days (x y z t : ℕ) : ℕ := (x + y + z) * t

/-- Theorem stating that the total days needed to listen to the entire collection
    is the product of the total number of records and the time per record. -/
theorem total_days_calculation (x y z t : ℕ) : 
  total_days x y z t = (x + y + z) * t := by sorry

end total_days_calculation_l3301_330171


namespace intersection_equals_sqrt_set_l3301_330117

-- Define the square S
def S : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

-- Define the set Ct for a given t
def C (t : ℝ) : Set (ℝ × ℝ) := {p ∈ S | p.1 / t + p.2 / (1 - t) ≥ 1}

-- Define the intersection of all Ct
def intersectionC : Set (ℝ × ℝ) := ⋂ t ∈ {t | 0 < t ∧ t < 1}, C t

-- Define the set of points (x, y) in S such that √x + √y ≥ 1
def sqrtSet : Set (ℝ × ℝ) := {p ∈ S | Real.sqrt p.1 + Real.sqrt p.2 ≥ 1}

-- State the theorem
theorem intersection_equals_sqrt_set : intersectionC = sqrtSet := by
  sorry

end intersection_equals_sqrt_set_l3301_330117


namespace ellipse_hyperbola_foci_l3301_330165

theorem ellipse_hyperbola_foci (a b : ℝ) : 
  (∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 → (x = 7 ∧ y = 0) ∨ (x = -7 ∧ y = 0)) →
  |a * b| = Real.sqrt 444 :=
by sorry

end ellipse_hyperbola_foci_l3301_330165


namespace f_has_min_value_neg_ten_l3301_330172

-- Define the function
def f (x : ℝ) : ℝ := 4 * x^2 - 12 * x - 1

-- Theorem statement
theorem f_has_min_value_neg_ten :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = -10 :=
by sorry

end f_has_min_value_neg_ten_l3301_330172


namespace ten_people_handshakes_l3301_330101

/-- Represents the number of handshakes in a group where each person shakes hands only with those taller than themselves. -/
def handshakes (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

/-- Theorem stating that in a group of 10 people where each person shakes hands only with those taller than themselves, the total number of handshakes is 45. -/
theorem ten_people_handshakes :
  handshakes 10 = 45 := by
  sorry

#eval handshakes 10  -- Should output 45

end ten_people_handshakes_l3301_330101
