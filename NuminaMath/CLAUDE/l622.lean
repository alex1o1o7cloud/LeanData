import Mathlib

namespace function_evaluation_l622_62283

theorem function_evaluation (f : ℝ → ℝ) (h : ∀ x, f x = 4*x - 2) : f (-3) = -14 := by
  sorry

end function_evaluation_l622_62283


namespace odd_even_f_l622_62216

def f (n : ℕ) : ℕ := (n * (Nat.totient n)) / 2

theorem odd_even_f (n : ℕ) (h : n > 1) :
  (Odd (f n) ∧ Even (f (2015 * n))) ↔ Odd n ∧ n > 1 := by sorry

end odd_even_f_l622_62216


namespace childrens_ticket_cost_l622_62284

theorem childrens_ticket_cost :
  let adult_ticket_cost : ℚ := 25
  let total_receipts : ℚ := 7200
  let total_attendance : ℕ := 400
  let adult_attendance : ℕ := 280
  let child_attendance : ℕ := 120
  let child_ticket_cost : ℚ := (total_receipts - (adult_ticket_cost * adult_attendance)) / child_attendance
  child_ticket_cost = 5/3 := by sorry

end childrens_ticket_cost_l622_62284


namespace complement_of_A_in_U_l622_62229

-- Define the universe U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x < 1}

-- Theorem statement
theorem complement_of_A_in_U : Set.compl A = {x : ℝ | x ≥ 1} := by sorry

end complement_of_A_in_U_l622_62229


namespace neon_signs_blink_together_l622_62234

theorem neon_signs_blink_together (a b : ℕ) (ha : a = 9) (hb : b = 15) :
  Nat.lcm a b = 45 := by
  sorry

end neon_signs_blink_together_l622_62234


namespace point_P_coordinates_l622_62218

def M : ℝ × ℝ := (-2, 7)
def N : ℝ × ℝ := (10, -2)

def vector (A B : ℝ × ℝ) : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

theorem point_P_coordinates :
  ∃ P : ℝ × ℝ,
    (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (M.1 + t * (N.1 - M.1), M.2 + t * (N.2 - M.2))) ∧
    vector P N = (-2 : ℝ) • (vector P M) ∧
    P = (2, 4) := by
  sorry

end point_P_coordinates_l622_62218


namespace solution_set_f_leq_4_range_of_m_f_gt_m_squared_plus_m_l622_62269

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x - 3|

-- Theorem for the solution set of f(x) ≤ 4
theorem solution_set_f_leq_4 :
  {x : ℝ | f x ≤ 4} = {x : ℝ | 0 ≤ x ∧ x ≤ 4} := by sorry

-- Theorem for the range of m where f(x) > m^2 + m always holds
theorem range_of_m_f_gt_m_squared_plus_m :
  {m : ℝ | ∀ x, f x > m^2 + m} = {m : ℝ | -2 < m ∧ m < 1} := by sorry

end solution_set_f_leq_4_range_of_m_f_gt_m_squared_plus_m_l622_62269


namespace S_min_at_24_l622_62267

/-- The sequence term a_n as a function of n -/
def a (n : ℕ) : ℤ := 2 * n - 49

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n * (2 * a 1 + (n - 1) * 2) / 2

/-- Theorem stating that S reaches its minimum value when n = 24 -/
theorem S_min_at_24 : ∀ k : ℕ, S 24 ≤ S k :=
sorry

end S_min_at_24_l622_62267


namespace ratio_s4_s5_l622_62223

/-- An arithmetic sequence with a given ratio of second to third term -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n
  ratio_condition : a 2 / a 3 = 1 / 3

/-- Sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (seq.a 1 + seq.a n) * n / 2

/-- The main theorem: ratio of S_4 to S_5 is 8/15 -/
theorem ratio_s4_s5 (seq : ArithmeticSequence) :
  sum_n seq 4 / sum_n seq 5 = 8 / 15 := by
  sorry

end ratio_s4_s5_l622_62223


namespace maple_leaf_high_basketball_score_l622_62258

theorem maple_leaf_high_basketball_score :
  ∀ (x : ℚ) (y : ℕ),
    x > 0 →
    (1/3 : ℚ) * x + (3/8 : ℚ) * x + 18 + y = x →
    10 ≤ y →
    y ≤ 30 →
    y = 21 :=
by
  sorry

end maple_leaf_high_basketball_score_l622_62258


namespace smallest_a1_l622_62251

/-- Given a sequence of positive real numbers {aₙ} where aₙ = 15aₙ₋₁ - 2n for all n > 1,
    the smallest possible value of a₁ is 29/98. -/
theorem smallest_a1 (a : ℕ → ℝ) :
  (∀ n, a n > 0) →
  (∀ n > 1, a n = 15 * a (n - 1) - 2 * n) →
  ∀ x, (∀ n, a n > 0) → (∀ n > 1, a n = 15 * a (n - 1) - 2 * n) → a 1 ≥ x →
  x ≤ 29 / 98 :=
by sorry

end smallest_a1_l622_62251


namespace range_of_a_l622_62286

open Real

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Define the theorem
theorem range_of_a (a : ℝ) (h : (¬p a) ∧ q a) : a > 1 := by
  sorry

end range_of_a_l622_62286


namespace tan_45_degrees_l622_62255

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end tan_45_degrees_l622_62255


namespace imaginary_part_of_complex_product_l622_62285

theorem imaginary_part_of_complex_product : Complex.im ((5 + Complex.I) * (1 - Complex.I)) = -4 := by
  sorry

end imaginary_part_of_complex_product_l622_62285


namespace solve_for_a_l622_62275

def U (a : ℝ) : Set ℝ := {2, 4, 1-a}
def A (a : ℝ) : Set ℝ := {2, a^2-a+2}

theorem solve_for_a (a : ℝ) : 
  (U a \ A a = {-1}) → a = 2 := by
  sorry

end solve_for_a_l622_62275


namespace actual_distance_traveled_l622_62210

/-- Proves that the actual distance traveled is 50 km given the conditions of the problem -/
theorem actual_distance_traveled (slow_speed fast_speed additional_distance : ℝ) 
  (h1 : slow_speed = 10)
  (h2 : fast_speed = 14)
  (h3 : additional_distance = 20)
  (h4 : ∀ d : ℝ, d / slow_speed = (d + additional_distance) / fast_speed) :
  ∃ d : ℝ, d = 50 ∧ d / slow_speed = (d + additional_distance) / fast_speed :=
by sorry

end actual_distance_traveled_l622_62210


namespace min_value_quadratic_roots_l622_62201

theorem min_value_quadratic_roots (a b c : ℤ) (α β : ℝ) : 
  a > 0 → 
  (∃ x : ℝ, a * x^2 + b * x + c = 0) → 
  (α * α * a + b * α + c = 0) →
  (β * β * a + b * β + c = 0) →
  0 < α → α < β → β < 1 → 
  a ≥ 5 := by
sorry

end min_value_quadratic_roots_l622_62201


namespace cheryl_mms_after_dinner_l622_62262

/-- The number of m&m's Cheryl had at the beginning -/
def initial_mms : ℕ := 25

/-- The number of m&m's Cheryl ate after lunch -/
def after_lunch : ℕ := 7

/-- The number of m&m's Cheryl gave to her sister -/
def given_to_sister : ℕ := 13

/-- The number of m&m's Cheryl ate after dinner -/
def after_dinner : ℕ := 5

theorem cheryl_mms_after_dinner : 
  initial_mms - after_lunch - after_dinner - given_to_sister = 0 :=
by sorry

end cheryl_mms_after_dinner_l622_62262


namespace lcm_problem_l622_62244

theorem lcm_problem (n : ℕ+) (h1 : Nat.lcm 40 n = 200) (h2 : Nat.lcm n 45 = 180) : n = 100 := by
  sorry

end lcm_problem_l622_62244


namespace tomorrow_is_saturday_l622_62230

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

-- Define a function to get the next day
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

-- Define a function to advance a day by n days
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

-- Theorem statement
theorem tomorrow_is_saturday 
  (h : advanceDay (advanceDay DayOfWeek.Wednesday 2) 5 = DayOfWeek.Monday) :
  nextDay DayOfWeek.Friday = DayOfWeek.Saturday :=
by
  sorry


end tomorrow_is_saturday_l622_62230


namespace multiplication_table_odd_fraction_l622_62270

/-- The size of the multiplication table (16 x 16) -/
def tableSize : ℕ := 16

/-- The number of odd numbers from 0 to 15 -/
def oddCount : ℕ := 8

/-- The total number of entries in the multiplication table -/
def totalEntries : ℕ := tableSize * tableSize

/-- The number of odd entries in the multiplication table -/
def oddEntries : ℕ := oddCount * oddCount

theorem multiplication_table_odd_fraction :
  (oddEntries : ℚ) / totalEntries = 1 / 4 := by
  sorry

end multiplication_table_odd_fraction_l622_62270


namespace root_exists_in_interval_l622_62291

def f (x : ℝ) := x^3 + x + 1

theorem root_exists_in_interval :
  ∃ r ∈ Set.Ioo (-1 : ℝ) 0, f r = 0 :=
sorry

end root_exists_in_interval_l622_62291


namespace fraction_evaluation_l622_62235

theorem fraction_evaluation : (1 - 1/4) / (1 - 1/3) = 9/8 := by
  sorry

end fraction_evaluation_l622_62235


namespace clara_age_multiple_of_anna_l622_62288

theorem clara_age_multiple_of_anna (anna_current_age clara_current_age : ℕ) 
  (h1 : anna_current_age = 54)
  (h2 : clara_current_age = 80) :
  ∃ (years_ago : ℕ), 
    clara_current_age - years_ago = 3 * (anna_current_age - years_ago) ∧ 
    years_ago = 41 := by
  sorry

end clara_age_multiple_of_anna_l622_62288


namespace unique_positive_number_l622_62261

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 17 = 60 * (1 / x) := by
  sorry

end unique_positive_number_l622_62261


namespace square_triangle_perimeter_l622_62274

theorem square_triangle_perimeter (square_perimeter : ℝ) :
  square_perimeter = 160 →
  let side_length := square_perimeter / 4
  let diagonal_length := side_length * Real.sqrt 2
  let triangle_perimeter := 2 * side_length + diagonal_length
  triangle_perimeter = 80 + 40 * Real.sqrt 2 := by
  sorry

end square_triangle_perimeter_l622_62274


namespace mary_story_characters_l622_62259

theorem mary_story_characters (total : ℕ) (a b c g d e f h : ℕ) : 
  total = 360 →
  a = total / 3 →
  b = (total - a) / 4 →
  c = (total - a - b) / 5 →
  g = (total - a - b - c) / 6 →
  d + e + f + h = total - a - b - c - g →
  d = 3 * e →
  f = 2 * e →
  h = f →
  d = 45 :=
by sorry

end mary_story_characters_l622_62259


namespace expression_value_l622_62256

theorem expression_value : 
  let x : ℝ := 5
  (x^2 - 3*x - 4) / (x - 4) = 6 := by
  sorry

end expression_value_l622_62256


namespace ball_probabilities_l622_62213

theorem ball_probabilities (total_balls : ℕ) (p_red p_black p_yellow : ℚ) :
  total_balls = 12 →
  p_red + p_black + p_yellow = 1 →
  p_red = 1/3 →
  p_black - p_yellow = 1/6 →
  p_black = 5/12 ∧ p_yellow = 1/4 := by
  sorry

end ball_probabilities_l622_62213


namespace base_n_multiple_of_five_l622_62254

theorem base_n_multiple_of_five (n : ℕ) : 
  let count := Finset.filter (fun n => (2*n^5 + 3*n^4 + 5*n^3 + 2*n^2 + 3*n + 6) % 5 = 0) 
    (Finset.range 99 ∪ {100})
  (2 ≤ n) → (n ≤ 100) → Finset.card count = 40 := by
  sorry

end base_n_multiple_of_five_l622_62254


namespace optimal_output_l622_62231

noncomputable section

/-- The defective rate as a function of daily output -/
def defective_rate (c : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ c then 1 / (6 - x) else 2 / 3

/-- The daily profit as a function of daily output -/
def daily_profit (c : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ c
  then (3 * (9 * x - 2 * x^2)) / (2 * (6 - x))
  else 0

/-- The theorem stating the optimal daily output for maximum profit -/
theorem optimal_output (c : ℝ) (h : 0 < c ∧ c < 6) :
  (∃ (x : ℝ), ∀ (y : ℝ), daily_profit c y ≤ daily_profit c x) →
  ((0 < c ∧ c < 3 → ∃ (x : ℝ), x = c ∧ ∀ (y : ℝ), daily_profit c y ≤ daily_profit c x) ∧
   (3 ≤ c ∧ c < 6 → ∃ (x : ℝ), x = 3 ∧ ∀ (y : ℝ), daily_profit c y ≤ daily_profit c x)) :=
by sorry

end

end optimal_output_l622_62231


namespace artist_payment_multiple_l622_62245

theorem artist_payment_multiple : ∃ (x : ℕ+) (D : ℕ+), 
  D + (x * D + 1000) = 50000 ∧ 
  ∀ (y : ℕ+), y > x → ¬∃ (E : ℕ+), E + (y * E + 1000) = 50000 := by
  sorry

end artist_payment_multiple_l622_62245


namespace wine_consumption_equations_l622_62280

/-- Represents the wine consumption and intoxication scenario from the Ming Dynasty poem --/
theorem wine_consumption_equations :
  ∃ (x y : ℚ),
    (x + y = 19) ∧
    (3 * x + (1/3) * y = 33) ∧
    (x ≥ 0) ∧ (y ≥ 0) :=
by sorry

end wine_consumption_equations_l622_62280


namespace root_implies_h_value_l622_62219

theorem root_implies_h_value (h : ℝ) :
  (3 : ℝ)^3 - 2*h*3 + 15 = 0 → h = 7 := by
  sorry

end root_implies_h_value_l622_62219


namespace probability_inconsistency_l622_62247

-- Define the probability measure
variable (p : Set ℝ → ℝ)

-- Define events a and b
variable (a b : Set ℝ)

-- State the given probabilities
axiom pa : p a = 0.18
axiom pb : p b = 0.5
axiom pab : p (a ∩ b) = 0.36

-- Theorem to prove the inconsistency
theorem probability_inconsistency :
  ¬(0 ≤ p a ∧ p a ≤ 1 ∧
    0 ≤ p b ∧ p b ≤ 1 ∧
    0 ≤ p (a ∩ b) ∧ p (a ∩ b) ≤ 1 ∧
    p (a ∩ b) ≤ p a ∧ p (a ∩ b) ≤ p b) :=
by sorry

end probability_inconsistency_l622_62247


namespace horseshoe_profit_800_sets_l622_62272

/-- Calculates the profit for horseshoe manufacturing given the specified conditions --/
def horseshoe_profit (
  initial_outlay : ℕ)
  (cost_first_300 : ℕ)
  (cost_beyond_300 : ℕ)
  (price_first_400 : ℕ)
  (price_beyond_400 : ℕ)
  (total_sets : ℕ) : ℕ :=
  let manufacturing_cost := initial_outlay +
    (min total_sets 300) * cost_first_300 +
    (max (total_sets - 300) 0) * cost_beyond_300
  let revenue := (min total_sets 400) * price_first_400 +
    (max (total_sets - 400) 0) * price_beyond_400
  revenue - manufacturing_cost

theorem horseshoe_profit_800_sets :
  horseshoe_profit 10000 20 15 50 45 800 = 14500 := by
  sorry

end horseshoe_profit_800_sets_l622_62272


namespace matrix_determinant_l622_62265

theorem matrix_determinant : 
  let A : Matrix (Fin 3) (Fin 3) ℤ := !![2, 4, -2; 0, 3, -1; 5, -1, 2]
  Matrix.det A = 20 := by
  sorry

end matrix_determinant_l622_62265


namespace additive_multiplicative_inverses_problem_l622_62209

theorem additive_multiplicative_inverses_problem (a b c d m : ℝ) 
  (h1 : a + b = 0)  -- a and b are additive inverses
  (h2 : c * d = 1)  -- c and d are multiplicative inverses
  (h3 : abs m = 1)  -- absolute value of m is 1
  : (a + b) * c * d - 2009 * m = -2009 ∨ (a + b) * c * d - 2009 * m = 2009 := by
  sorry

end additive_multiplicative_inverses_problem_l622_62209


namespace fraction_reduction_l622_62248

theorem fraction_reduction (a b c : ℝ) 
  (h : (3*a^2 + 6*a*c - 3*c^2 - 6*a*b) ≠ 0) : 
  (4*a^2 + 2*c^2 - 4*b^2 - 8*b*c) / (3*a^2 + 6*a*c - 3*c^2 - 6*a*b) = 
  (4/3) * ((a-2*b+c)*(a-c)) / ((a-b+c)*(a-b-c)) := by
  sorry

end fraction_reduction_l622_62248


namespace completing_square_transformation_l622_62287

theorem completing_square_transformation :
  ∃ (m n : ℝ), (∀ x : ℝ, x^2 - 4*x - 4 = 0 ↔ (x + m)^2 = n) ∧ m = -2 ∧ n = 8 := by
  sorry

end completing_square_transformation_l622_62287


namespace parabola_translation_l622_62226

/-- The initial parabola function -/
def initial_parabola (x : ℝ) : ℝ := -3 * (x + 1)^2 - 2

/-- The final parabola function -/
def final_parabola (x : ℝ) : ℝ := -3 * x^2

/-- Translation function that moves a point 1 unit right and 2 units up -/
def translate (x y : ℝ) : ℝ × ℝ := (x - 1, y + 2)

theorem parabola_translation :
  ∀ x : ℝ, final_parabola x = (initial_parabola (x - 1) + 2) :=
by sorry

end parabola_translation_l622_62226


namespace initial_concentration_is_40_percent_l622_62215

-- Define the capacities and concentrations
def vessel1_capacity : ℝ := 2
def vessel2_capacity : ℝ := 6
def vessel2_concentration : ℝ := 0.6
def total_liquid : ℝ := 8
def new_vessel_capacity : ℝ := 10
def final_concentration : ℝ := 0.44

-- Define the unknown initial concentration of vessel 1
def vessel1_concentration : ℝ := sorry

-- Theorem statement
theorem initial_concentration_is_40_percent :
  vessel1_concentration * vessel1_capacity + 
  vessel2_concentration * vessel2_capacity = 
  final_concentration * new_vessel_capacity := by
  sorry

end initial_concentration_is_40_percent_l622_62215


namespace functional_equation_solution_l622_62297

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 - y^2) = (x - y) * (f x + f y)) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x := by
sorry

end functional_equation_solution_l622_62297


namespace freds_allowance_l622_62264

/-- Fred's weekly allowance problem -/
theorem freds_allowance (allowance : ℝ) : 
  (allowance / 2 + 11 = 20) → allowance = 18 := by
  sorry

end freds_allowance_l622_62264


namespace yuna_has_biggest_number_l622_62260

def yoongi_number : ℕ := 7
def jungkook_number : ℕ := 6
def yuna_number : ℕ := 9

theorem yuna_has_biggest_number :
  yuna_number = max yoongi_number (max jungkook_number yuna_number) :=
by sorry

end yuna_has_biggest_number_l622_62260


namespace max_value_trig_expression_l622_62232

theorem max_value_trig_expression (a b c : ℝ) :
  (⨆ θ : ℝ, a * Real.cos θ + b * Real.sin θ + c * Real.sin (2 * θ)) = Real.sqrt (a^2 + b^2 + 4 * c^2) := by
  sorry

end max_value_trig_expression_l622_62232


namespace odd_function_value_l622_62240

/-- A function f: ℝ → ℝ is odd if f(-x) = -f(x) for all x ∈ ℝ -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_value (f : ℝ → ℝ) (b : ℝ) 
    (h_odd : IsOdd f)
    (h_def : ∀ x ≥ 0, f x = x^2 - 3*x + b) :
  f (-2) = 2 := by
  sorry

end odd_function_value_l622_62240


namespace square_area_l622_62298

-- Define the parabola function
def parabola (x : ℝ) : ℝ := x^2 + 5*x + 6

-- Define the horizontal line
def horizontal_line : ℝ := 10

-- Theorem statement
theorem square_area : ∃ (x₁ x₂ : ℝ), 
  parabola x₁ = horizontal_line ∧ 
  parabola x₂ = horizontal_line ∧ 
  (x₂ - x₁)^2 = 41 := by
  sorry

end square_area_l622_62298


namespace pasta_calculation_l622_62243

/-- Given a recipe that uses 2 pounds of pasta to serve 7 people,
    calculate the amount of pasta needed to serve 35 people. -/
theorem pasta_calculation (original_pasta : ℝ) (original_servings : ℕ) 
    (target_servings : ℕ) (h1 : original_pasta = 2) 
    (h2 : original_servings = 7) (h3 : target_servings = 35) : 
    (original_pasta * target_servings / original_servings : ℝ) = 10 := by
  sorry

#check pasta_calculation

end pasta_calculation_l622_62243


namespace fourth_root_equation_solution_l622_62237

theorem fourth_root_equation_solution :
  ∃! x : ℝ, (2 - x / 2) ^ (1/4 : ℝ) = 2 ∧ x = -28 := by
  sorry

end fourth_root_equation_solution_l622_62237


namespace special_quadrilateral_area_sum_l622_62250

/-- A convex quadrilateral with specific side lengths and angle -/
structure ConvexQuadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  angleCDA : ℝ
  convex : AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0
  angleCondition : 0 < angleCDA ∧ angleCDA < π

/-- The area of the quadrilateral can be expressed in the form √a + b√c -/
def hasSpecialAreaForm (q : ConvexQuadrilateral) (a b c : ℕ) : Prop :=
  ∃ (area : ℝ), area = Real.sqrt a + b * Real.sqrt c ∧
  area = q.AB * q.BC * Real.sin q.angleCDA / 2 + q.CD * q.DA * Real.sin q.angleCDA / 2 ∧
  ∀ k : ℕ, k > 1 → (k * k ∣ a → k = 1) ∧ (k * k ∣ c → k = 1)

/-- Main theorem -/
theorem special_quadrilateral_area_sum (q : ConvexQuadrilateral) 
    (h1 : q.AB = 8) (h2 : q.BC = 4) (h3 : q.CD = 10) (h4 : q.DA = 10) 
    (h5 : q.angleCDA = π/3) (a b c : ℕ) (h6 : hasSpecialAreaForm q a b c) : 
    a + b + c = 259 := by
  sorry

end special_quadrilateral_area_sum_l622_62250


namespace work_increase_percentage_l622_62222

theorem work_increase_percentage (p : ℕ) (W : ℝ) (h : p > 0) : 
  let absent_ratio : ℝ := 1 / 6
  let present_ratio : ℝ := 1 - absent_ratio
  let original_work_per_person : ℝ := W / p
  let new_work_per_person : ℝ := W / (p * present_ratio)
  let work_increase : ℝ := new_work_per_person - original_work_per_person
  let percentage_increase : ℝ := (work_increase / original_work_per_person) * 100
  percentage_increase = 20 := by sorry

end work_increase_percentage_l622_62222


namespace positive_reals_inequality_l622_62202

theorem positive_reals_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (1 + a * b) / (1 + a) + (1 + b * c) / (1 + b) + (1 + c * a) / (1 + c) ≥ 3 := by
  sorry

end positive_reals_inequality_l622_62202


namespace range_of_s_l622_62233

/-- Definition of the function s for composite positive integers -/
def s (n : ℕ) : ℕ :=
  if n.Prime then 0
  else (n.factors.map (λ p => p ^ 2)).sum

/-- The range of s is the set of integers greater than 11 -/
theorem range_of_s :
  ∀ m : ℕ, m > 11 → ∃ n : ℕ, ¬n.Prime ∧ s n = m ∧
  ∀ k : ℕ, ¬k.Prime → s k > 11 :=
sorry

end range_of_s_l622_62233


namespace rectangular_field_length_l622_62295

/-- Represents a rectangular field with a given width and area. -/
structure RectangularField where
  width : ℝ
  area : ℝ

/-- The length of a rectangular field is 10 meters more than its width. -/
def length (field : RectangularField) : ℝ := field.width + 10

/-- The theorem stating that a rectangular field with an area of 171 square meters
    and length 10 meters more than its width has a length of 19 meters. -/
theorem rectangular_field_length (field : RectangularField) 
  (h1 : field.area = 171)
  (h2 : field.area = field.width * (field.width + 10)) :
  length field = 19 := by
  sorry

#check rectangular_field_length

end rectangular_field_length_l622_62295


namespace cos_225_degrees_l622_62293

theorem cos_225_degrees : Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end cos_225_degrees_l622_62293


namespace quadratic_root_form_l622_62217

theorem quadratic_root_form (m n p : ℕ+) (h_gcd : Nat.gcd m.val (Nat.gcd n.val p.val) = 1) :
  (∀ x : ℝ, 3 * x^2 - 8 * x + 2 = 0 ↔ x = (m.val + Real.sqrt n.val) / p.val ∨ x = (m.val - Real.sqrt n.val) / p.val) →
  n = 10 := by
  sorry

end quadratic_root_form_l622_62217


namespace divisible_by_2power10000_within_day_l622_62252

/-- Represents a card with a natural number -/
structure Card where
  value : ℕ

/-- Represents the state of the table at any given time -/
structure TableState where
  cards : List Card
  time : ℕ

/-- Checks if a number is divisible by 2^10000 -/
def isDivisibleBy2Power10000 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * (2^10000)

/-- The process of adding a new card every minute -/
def addNewCard (state : TableState) : TableState :=
  sorry

/-- The main theorem to be proved -/
theorem divisible_by_2power10000_within_day
  (initial_cards : List Card)
  (h1 : initial_cards.length = 100)
  (h2 : (initial_cards.filter (fun c => c.value % 2 = 1)).length = 43) :
  ∃ (final_state : TableState),
    final_state.time ≤ 1440 ∧
    ∃ (c : Card), c ∈ final_state.cards ∧ isDivisibleBy2Power10000 c.value :=
  sorry

end divisible_by_2power10000_within_day_l622_62252


namespace trigonometric_identities_l622_62273

theorem trigonometric_identities :
  (∀ x : Real, (1 + Real.tan (1 * π / 180)) * (1 + Real.tan (44 * π / 180)) = 2) ∧
  (∀ x : Real, (3 - Real.sin (70 * π / 180)) / (2 - Real.cos (10 * π / 180)^2) = 2) := by
  sorry

end trigonometric_identities_l622_62273


namespace median_sum_ge_four_times_circumradius_l622_62282

/-- A triangle in a 2D Euclidean space --/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The radius of the circumscribed circle of a triangle --/
def circumradius (t : Triangle) : ℝ := sorry

/-- The length of a median of a triangle --/
def median_length (t : Triangle) (vertex : Fin 3) : ℝ := sorry

/-- Predicate to check if a triangle is not obtuse --/
def is_not_obtuse (t : Triangle) : Prop := sorry

/-- Theorem: For any non-obtuse triangle, the sum of its three medians
    is greater than or equal to four times the radius of its circumscribed circle --/
theorem median_sum_ge_four_times_circumradius (t : Triangle) :
  is_not_obtuse t →
  (median_length t 0) + (median_length t 1) + (median_length t 2) ≥ 4 * (circumradius t) := by
  sorry

end median_sum_ge_four_times_circumradius_l622_62282


namespace game_probability_l622_62249

theorem game_probability (p_lose p_tie : ℚ) 
  (h_lose : p_lose = 3/7)
  (h_tie : p_tie = 2/21)
  (h_outcomes : ∃ (p_win : ℚ), p_win + p_lose + p_tie = 1) :
  ∃ (p_win : ℚ), p_win = 10/21 := by
sorry

end game_probability_l622_62249


namespace base3_to_base10_20123_l622_62211

/-- Converts a base 3 number to base 10 --/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

/-- The base 3 representation of the number --/
def base3Number : List Nat := [3, 2, 1, 0, 2]

/-- Theorem stating that the base 10 equivalent of 20123 (base 3) is 180 --/
theorem base3_to_base10_20123 :
  base3ToBase10 base3Number = 180 := by
  sorry

end base3_to_base10_20123_l622_62211


namespace minimum_value_implies_m_equals_one_l622_62205

-- Define the domain D
def D : Set ℝ := Set.Icc 1 2

-- Define the function g
def g (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*m*x - m^2

-- Theorem statement
theorem minimum_value_implies_m_equals_one :
  ∀ m : ℝ, (∀ x ∈ D, g m x ≥ 2) ∧ (∃ x ∈ D, g m x = 2) → m = 1 := by
  sorry

end minimum_value_implies_m_equals_one_l622_62205


namespace total_jelly_beans_l622_62207

theorem total_jelly_beans (vanilla : ℕ) (grape : ℕ) : 
  vanilla = 120 → 
  grape = 5 * vanilla + 50 → 
  vanilla + grape = 770 := by
sorry

end total_jelly_beans_l622_62207


namespace prob_consecutive_prob_sum_divisible_by_3_l622_62241

-- Define the type for ball labels
inductive BallLabel : Type
  | one : BallLabel
  | two : BallLabel
  | three : BallLabel
  | four : BallLabel

-- Define a function to convert BallLabel to natural number
def ballLabelToNat (b : BallLabel) : ℕ :=
  match b with
  | BallLabel.one => 1
  | BallLabel.two => 2
  | BallLabel.three => 3
  | BallLabel.four => 4

-- Define the type for a pair of drawn balls
def DrawnPair := BallLabel × BallLabel

-- Define the sample space
def sampleSpace : Finset DrawnPair := sorry

-- Define the event of drawing consecutive numbers
def consecutiveEvent : Finset DrawnPair := sorry

-- Define the event of drawing numbers with sum divisible by 3
def sumDivisibleBy3Event : Finset DrawnPair := sorry

-- Theorem for the probability of drawing consecutive numbers
theorem prob_consecutive : 
  (consecutiveEvent.card : ℚ) / sampleSpace.card = 3 / 8 := sorry

-- Theorem for the probability of drawing numbers with sum divisible by 3
theorem prob_sum_divisible_by_3 : 
  (sumDivisibleBy3Event.card : ℚ) / sampleSpace.card = 5 / 16 := sorry

end prob_consecutive_prob_sum_divisible_by_3_l622_62241


namespace opposite_minus_six_l622_62289

theorem opposite_minus_six (a : ℤ) : a = -(-6) → 1 - a = -5 := by
  sorry

end opposite_minus_six_l622_62289


namespace inequality_equivalence_l622_62236

theorem inequality_equivalence :
  {x : ℝ | |(6 - 2*x + 5) / 4| < 3} = {x : ℝ | -1/2 < x ∧ x < 23/2} := by
  sorry

end inequality_equivalence_l622_62236


namespace xyz_value_l622_62224

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 25)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 7) : 
  x * y * z = 6 := by
  sorry

end xyz_value_l622_62224


namespace bob_pennies_bob_pennies_proof_l622_62212

theorem bob_pennies : ℕ → ℕ → Prop :=
  fun a b =>
    (b + 1 = 4 * (a - 1)) ∧
    (b - 1 = 3 * (a + 1)) →
    b = 31

-- The proof is omitted
theorem bob_pennies_proof : bob_pennies 9 31 := by sorry

end bob_pennies_bob_pennies_proof_l622_62212


namespace circplus_not_commutative_l622_62263

/-- Definition of the ⊕ operation -/
def circplus (a b : ℚ) : ℚ := a * b + 2 * a

/-- Theorem stating that ⊕ is not commutative -/
theorem circplus_not_commutative : ¬ (∀ a b : ℚ, circplus a b = circplus b a) := by
  sorry

end circplus_not_commutative_l622_62263


namespace youngest_sibling_age_l622_62246

theorem youngest_sibling_age 
  (siblings : Fin 4 → ℕ) 
  (age_differences : ∀ i : Fin 4, siblings i = siblings 0 + [0, 2, 7, 11].get i) 
  (average_age : (siblings 0 + siblings 1 + siblings 2 + siblings 3) / 4 = 25) : 
  siblings 0 = 20 := by
  sorry

end youngest_sibling_age_l622_62246


namespace trig_special_angles_sum_l622_62292

theorem trig_special_angles_sum : 
  4 * Real.sin (30 * π / 180) - Real.sqrt 2 * Real.cos (45 * π / 180) - 
  Real.sqrt 3 * Real.tan (30 * π / 180) + 2 * Real.sin (60 * π / 180) = Real.sqrt 3 := by
  sorry

end trig_special_angles_sum_l622_62292


namespace power_equation_solutions_l622_62242

theorem power_equation_solutions (a b : ℕ) (ha : a ≥ 1) (hb : b ≥ 1) :
  a^(b^2) = b^a → (a = 1 ∧ b = 1) ∨ (a = 2 ∧ b = 16) ∨ (a = 3 ∧ b = 27) := by
  sorry

end power_equation_solutions_l622_62242


namespace midpoint_trajectory_l622_62227

/-- The trajectory of the midpoint M of a line segment PP', where P is on a circle
    with center (0,0) and radius 2, and P' is the projection of P on the x-axis. -/
theorem midpoint_trajectory (x y : ℝ) : 
  (∃ x₀ y₀ : ℝ, 
    x₀^2 + y₀^2 = 4 ∧   -- P is on the circle
    x = x₀ ∧            -- M's x-coordinate is same as P's
    2 * y = y₀) →       -- M's y-coordinate is half of P's
  x^2 / 4 + y^2 = 1 := by
sorry

end midpoint_trajectory_l622_62227


namespace tangent_line_and_extreme_values_l622_62279

-- Define the function f(x)
def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b*x - 4

-- Define the derivative of f(x)
def f' (a b x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

-- Theorem statement
theorem tangent_line_and_extreme_values :
  ∃ (a b : ℝ),
  (f a b 2 = -2) ∧
  (f' a b 2 = 1) ∧
  (a = -4 ∧ b = 5) ∧
  (∀ x : ℝ, f (-4) 5 x ≤ -2) ∧
  (f (-4) 5 1 = -2) ∧
  (∀ x : ℝ, f (-4) 5 x ≥ -58/27) ∧
  (f (-4) 5 (5/3) = -58/27) := by
sorry

end tangent_line_and_extreme_values_l622_62279


namespace sara_wins_731_l622_62220

/-- Represents the state of a wall in the brick removal game -/
def Wall := List Nat

/-- Calculates the nim-value of a single wall -/
def nimValue (wall : Nat) : Nat :=
  sorry

/-- Calculates the nim-sum (XOR) of a list of natural numbers -/
def nimSum (values : List Nat) : Nat :=
  sorry

/-- Determines if a given game state is a winning position for the second player -/
def isWinningForSecondPlayer (state : Wall) : Prop :=
  nimSum (state.map nimValue) = 0

/-- The main theorem stating that (7, 3, 1) is a winning position for the second player -/
theorem sara_wins_731 : isWinningForSecondPlayer [7, 3, 1] := by
  sorry

end sara_wins_731_l622_62220


namespace unique_solution_for_n_l622_62238

theorem unique_solution_for_n : ∃! n : ℚ, (1 / (n + 2)) + (2 / (n + 2)) + ((n + 1) / (n + 2)) = 2 := by
  sorry

end unique_solution_for_n_l622_62238


namespace keaton_annual_earnings_l622_62277

/-- Represents Keaton's farm and calculates annual earnings --/
def farm_earnings : ℕ :=
  let months_per_year : ℕ := 12
  let orange_harvest_interval : ℕ := 2
  let apple_harvest_interval : ℕ := 3
  let orange_harvest_value : ℕ := 50
  let apple_harvest_value : ℕ := 30
  let orange_harvests_per_year : ℕ := months_per_year / orange_harvest_interval
  let apple_harvests_per_year : ℕ := months_per_year / apple_harvest_interval
  let orange_earnings : ℕ := orange_harvests_per_year * orange_harvest_value
  let apple_earnings : ℕ := apple_harvests_per_year * apple_harvest_value
  orange_earnings + apple_earnings

/-- Theorem stating that Keaton's annual farm earnings are $420 --/
theorem keaton_annual_earnings : farm_earnings = 420 := by
  sorry

end keaton_annual_earnings_l622_62277


namespace towel_rate_proof_l622_62200

/-- Proves that given the specified towel purchases and average price, the unknown rate must be 250. -/
theorem towel_rate_proof (num_towels_1 num_towels_2 num_towels_unknown : ℕ)
  (price_1 price_2 avg_price : ℚ) :
  num_towels_1 = 3 →
  num_towels_2 = 5 →
  num_towels_unknown = 2 →
  price_1 = 100 →
  price_2 = 150 →
  avg_price = 155 →
  let total_towels := num_towels_1 + num_towels_2 + num_towels_unknown
  let total_cost := num_towels_1 * price_1 + num_towels_2 * price_2 + num_towels_unknown * avg_price
  (total_cost / total_towels : ℚ) = avg_price →
  (((total_cost - (num_towels_1 * price_1 + num_towels_2 * price_2)) / num_towels_unknown) : ℚ) = 250 :=
by sorry

end towel_rate_proof_l622_62200


namespace linear_function_quadrants_l622_62206

def linear_function (k : ℝ) (x : ℝ) : ℝ := k * x - k

def decreasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ < x₂ → f x₁ > f x₂

def passes_through_quadrant (f : ℝ → ℝ) (quadrant : ℕ) : Prop :=
  match quadrant with
  | 1 => ∃ x > 0, f x > 0
  | 2 => ∃ x < 0, f x > 0
  | 3 => ∃ x < 0, f x < 0
  | 4 => ∃ x > 0, f x < 0
  | _ => False

theorem linear_function_quadrants (k : ℝ) :
  decreasing_function (linear_function k) →
  (passes_through_quadrant (linear_function k) 1 ∧
   passes_through_quadrant (linear_function k) 2 ∧
   passes_through_quadrant (linear_function k) 4) :=
by sorry

end linear_function_quadrants_l622_62206


namespace initially_calculated_average_is_175_l622_62228

/-- The initially calculated average height of a class, given:
  * The class has 20 students
  * One student's height was incorrectly recorded as 40 cm more than their actual height
  * The actual average height of the students is 173 cm
-/
def initiallyCalculatedAverage (numStudents : ℕ) (heightError : ℕ) (actualAverage : ℕ) : ℕ :=
  actualAverage + heightError / numStudents

/-- Theorem stating that the initially calculated average height is 175 cm -/
theorem initially_calculated_average_is_175 :
  initiallyCalculatedAverage 20 40 173 = 175 := by
  sorry

end initially_calculated_average_is_175_l622_62228


namespace inserted_numbers_sum_l622_62271

theorem inserted_numbers_sum (a b c : ℝ) : 
  (∃ r : ℝ, a = 3 * r ∧ b = 3 * r^2) →  -- Geometric progression condition
  (∃ d : ℝ, b = a + d ∧ c = b + d ∧ 27 = c + d) →  -- Arithmetic progression condition
  a + b + c = 161 / 3 := by
sorry

end inserted_numbers_sum_l622_62271


namespace tv_price_increase_l622_62208

theorem tv_price_increase (x : ℝ) : 
  (1 + x / 100) * 1.2 = 1 + 56.00000000000001 / 100 → x = 30 := by
  sorry

end tv_price_increase_l622_62208


namespace complex_coordinate_proof_l622_62296

theorem complex_coordinate_proof (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) :
  z = 1 + Complex.I := by
  sorry

end complex_coordinate_proof_l622_62296


namespace reflected_light_ray_equation_l622_62239

/-- Given a point P and its reflection P' across the x-axis, and another point Q,
    this function returns true if the given equation represents the line through P' and Q -/
def is_reflected_line_equation (P Q : ℝ × ℝ) (equation : ℝ → ℝ → ℝ) : Prop :=
  let P' := (P.1, -P.2)  -- Reflection of P across x-axis
  (equation P'.1 P'.2 = 0) ∧ (equation Q.1 Q.2 = 0)

/-- The main theorem stating that 4x + y - 5 = 0 is the equation of the 
    reflected light ray for the given points -/
theorem reflected_light_ray_equation :
  is_reflected_line_equation (2, 3) (1, 1) (fun x y => 4*x + y - 5) := by
  sorry

#check reflected_light_ray_equation

end reflected_light_ray_equation_l622_62239


namespace triangle_point_inequalities_l622_62294

/-- Given a triangle ABC and a point P, prove two inequalities involving side lengths and distances --/
theorem triangle_point_inequalities 
  (A B C P : ℝ × ℝ) -- Points in 2D plane
  (a b c : ℝ) -- Side lengths of triangle ABC
  (α β γ : ℝ) -- Distances from P to A, B, C respectively
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0) -- Triangle inequality
  (h_a : a = Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)) -- Definition of side length a
  (h_b : b = Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)) -- Definition of side length b
  (h_c : c = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) -- Definition of side length c
  (h_α : α = Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)) -- Definition of distance α
  (h_β : β = Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)) -- Definition of distance β
  (h_γ : γ = Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)) -- Definition of distance γ
  : (a * β * γ + b * γ * α + c * α * β ≥ a * b * c) ∧ 
    (α * b * c + β * c * a + γ * a * b ≥ Real.sqrt 3 * a * b * c) := by
  sorry

end triangle_point_inequalities_l622_62294


namespace integer_solutions_of_polynomial_l622_62214

theorem integer_solutions_of_polynomial (n : ℤ) : 
  n^5 - 2*n^4 - 7*n^2 - 7*n + 3 = 0 ↔ n = -1 ∨ n = 3 := by
  sorry

end integer_solutions_of_polynomial_l622_62214


namespace cow_count_l622_62257

theorem cow_count (ducks cows : ℕ) : 
  (2 * ducks + 4 * cows = 2 * (ducks + cows) + 16) → cows = 8 := by
sorry

end cow_count_l622_62257


namespace ourNumber_decimal_l622_62299

/-- Represents a number in millions, thousands, and ones -/
structure LargeNumber where
  millions : Nat
  thousands : Nat
  ones : Nat

/-- Converts a LargeNumber to its decimal representation -/
def toDecimal (n : LargeNumber) : Nat :=
  n.millions * 1000000 + n.thousands * 1000 + n.ones

/-- The specific large number we're working with -/
def ourNumber : LargeNumber :=
  { millions := 10
  , thousands := 300
  , ones := 50 }

theorem ourNumber_decimal : toDecimal ourNumber = 10300050 := by
  sorry

end ourNumber_decimal_l622_62299


namespace intersection_point_on_line_and_x_axis_l622_62253

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := 5 * y - 7 * x = 35

/-- The point of intersection -/
def intersection_point : ℝ × ℝ := (-5, 0)

/-- Theorem: The intersection point satisfies the line equation and lies on the x-axis -/
theorem intersection_point_on_line_and_x_axis :
  line_equation intersection_point.1 intersection_point.2 ∧ intersection_point.2 = 0 := by
  sorry

end intersection_point_on_line_and_x_axis_l622_62253


namespace third_divisor_l622_62290

theorem third_divisor (x : ℕ) 
  (h1 : x - 16 = 136)
  (h2 : 4 ∣ x)
  (h3 : 6 ∣ x)
  (h4 : 10 ∣ x)
  (h5 : ∀ y, y - 16 = 136 ∧ 4 ∣ y ∧ 6 ∣ y ∧ 10 ∣ y → x ≤ y) :
  19 ∣ x ∧ 19 ≠ 4 ∧ 19 ≠ 6 ∧ 19 ≠ 10 :=
sorry

end third_divisor_l622_62290


namespace highest_frequency_last_3_groups_l622_62278

/-- Represents the frequency distribution of a sample -/
structure FrequencyDistribution where
  total_sample : ℕ
  num_groups : ℕ
  cumulative_freq_7 : ℚ
  last_3_geometric : Bool
  common_ratio_gt_2 : Bool

/-- Theorem stating the highest frequency in the last 3 groups -/
theorem highest_frequency_last_3_groups
  (fd : FrequencyDistribution)
  (h1 : fd.total_sample = 100)
  (h2 : fd.num_groups = 10)
  (h3 : fd.cumulative_freq_7 = 79/100)
  (h4 : fd.last_3_geometric)
  (h5 : fd.common_ratio_gt_2) :
  ∃ (a r : ℕ),
    r > 2 ∧
    a + a * r + a * r^2 = 21 ∧
    (∀ x : ℕ, x ∈ [a, a * r, a * r^2] → x ≤ 16) ∧
    16 ∈ [a, a * r, a * r^2] :=
sorry

end highest_frequency_last_3_groups_l622_62278


namespace simplify_trig_expression_l622_62281

theorem simplify_trig_expression :
  (Real.sin (40 * π / 180) + Real.sin (80 * π / 180)) /
  (Real.cos (40 * π / 180) + Real.cos (80 * π / 180)) =
  Real.tan (60 * π / 180) := by
sorry

end simplify_trig_expression_l622_62281


namespace distinct_polygons_count_l622_62268

/-- The number of points marked on the circle -/
def n : ℕ := 15

/-- The total number of subsets of n points -/
def total_subsets : ℕ := 2^n

/-- The number of subsets with 0, 1, 2, or 3 members -/
def small_subsets : ℕ := (n.choose 0) + (n.choose 1) + (n.choose 2) + (n.choose 3)

/-- The maximum number of points that can lie on a semicircle -/
def max_semicircle : ℕ := n / 2 + 1

/-- The number of subsets that lie on a semicircle -/
def semicircle_subsets : ℕ := 2^max_semicircle - 1

/-- Conservative estimate of subsets to exclude due to lying on the same semicircle -/
def conservative_exclusion : ℕ := 500

/-- The number of distinct convex polygons with 4 or more sides -/
def distinct_polygons : ℕ := total_subsets - small_subsets - semicircle_subsets - conservative_exclusion

theorem distinct_polygons_count :
  distinct_polygons = 31437 :=
sorry

end distinct_polygons_count_l622_62268


namespace arithmetic_sequence_298_l622_62203

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + d * (n - 1)

theorem arithmetic_sequence_298 :
  ∃ n : ℕ, arithmetic_sequence 1 3 n = 298 ∧ n = 100 :=
by
  sorry

end arithmetic_sequence_298_l622_62203


namespace parabola_properties_l622_62266

-- Define the parabola equation
def parabola_equation (x : ℝ) : ℝ := -3 * x^2 + 18 * x - 29

-- Define the vertex of the parabola
def vertex : ℝ × ℝ := (3, -2)

-- Define the point that the parabola passes through
def point_on_parabola : ℝ × ℝ := (4, -5)

-- Theorem statement
theorem parabola_properties :
  -- The parabola passes through the given point
  parabola_equation point_on_parabola.1 = point_on_parabola.2 ∧
  -- The vertex of the parabola is at the given point
  (∀ x : ℝ, parabola_equation x ≥ parabola_equation vertex.1) ∧
  -- The axis of symmetry is vertical (x = vertex.1)
  (∀ x : ℝ, parabola_equation (2 * vertex.1 - x) = parabola_equation x) := by
  sorry

end parabola_properties_l622_62266


namespace friday_ice_cream_amount_l622_62225

/-- The amount of ice cream eaten on Friday night, given the total amount eaten over two nights and the amount eaten on Saturday night. -/
theorem friday_ice_cream_amount (total : ℝ) (saturday : ℝ) (h1 : total = 3.5) (h2 : saturday = 0.25) :
  total - saturday = 3.25 := by
  sorry

end friday_ice_cream_amount_l622_62225


namespace students_in_class_l622_62221

theorem students_in_class (b : ℕ) : 
  100 < b ∧ b < 200 ∧ 
  b % 3 = 1 ∧ 
  b % 4 = 1 ∧ 
  b % 5 = 1 → 
  b = 101 ∨ b = 161 := by sorry

end students_in_class_l622_62221


namespace positive_integer_value_l622_62276

def first_seven_multiples_of_four : List Nat := [4, 8, 12, 16, 20, 24, 28]

def a : ℚ := (first_seven_multiples_of_four.sum : ℚ) / 7

def b (n : ℕ) : ℚ := 2 * n

theorem positive_integer_value (n : ℕ) (h : n > 0) :
  a^2 - (b n)^2 = 0 → n = 8 := by
  sorry

end positive_integer_value_l622_62276


namespace polygon_interior_exterior_angle_relation_l622_62204

theorem polygon_interior_exterior_angle_relation :
  ∀ n : ℕ, 
  n > 2 →
  (n - 2) * 180 = 2 * 360 →
  n = 6 :=
by
  sorry

end polygon_interior_exterior_angle_relation_l622_62204
