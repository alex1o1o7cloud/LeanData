import Mathlib

namespace rudy_total_running_time_l3762_376204

/-- Calculates the total running time for Rudy given his running segments -/
theorem rudy_total_running_time :
  let segment1 : ℝ := 5 * 10  -- 5 miles at 10 minutes per mile
  let segment2 : ℝ := 4 * 9.5 -- 4 miles at 9.5 minutes per mile
  let segment3 : ℝ := 3 * 8.5 -- 3 miles at 8.5 minutes per mile
  let segment4 : ℝ := 2 * 12  -- 2 miles at 12 minutes per mile
  segment1 + segment2 + segment3 + segment4 = 137.5 := by
sorry

end rudy_total_running_time_l3762_376204


namespace range_x_and_a_l3762_376215

def P (x a : ℝ) : Prop := -x^2 + 4*a*x - 3*a^2 > 0

def q (x : ℝ) : Prop := (x - 3) / (x - 2) < 0

theorem range_x_and_a (a : ℝ) (h : a > 0) :
  (∀ x, P x 1 ∧ q x → x > 2 ∧ x < 3) ∧
  (∀ a, (∀ x, 2 < x ∧ x < 3 → a < x ∧ x < 3*a) ↔ 1 ≤ a ∧ a ≤ 2) := by
  sorry

end range_x_and_a_l3762_376215


namespace negation_of_squared_nonnegative_l3762_376257

theorem negation_of_squared_nonnegative :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) := by sorry

end negation_of_squared_nonnegative_l3762_376257


namespace min_value_of_expression_min_value_achievable_l3762_376239

theorem min_value_of_expression (x y : ℝ) : 
  (x^2*y - 1)^2 + (x^2 + y)^2 ≥ 1 :=
sorry

theorem min_value_achievable : 
  ∃ x y : ℝ, (x^2*y - 1)^2 + (x^2 + y)^2 = 1 :=
sorry

end min_value_of_expression_min_value_achievable_l3762_376239


namespace urn_probability_l3762_376200

/-- Represents the color of a ball -/
inductive Color
| Red
| Blue

/-- Represents the state of the urn -/
structure UrnState :=
  (red : ℕ)
  (blue : ℕ)

/-- Represents a sequence of ball draws -/
def DrawSequence := List Color

/-- The initial state of the urn -/
def initial_state : UrnState := ⟨2, 1⟩

/-- The number of operations performed -/
def num_operations : ℕ := 5

/-- The final number of balls in the urn -/
def final_total_balls : ℕ := 8

/-- The desired final state of the urn -/
def target_state : UrnState := ⟨3, 3⟩

/-- Calculates the probability of a specific draw sequence -/
def sequence_probability (seq : DrawSequence) : ℚ :=
  sorry

/-- Calculates the number of valid sequences that result in the target state -/
def num_valid_sequences : ℕ :=
  sorry

/-- The main theorem to prove -/
theorem urn_probability : 
  (num_valid_sequences * sequence_probability (List.replicate num_operations Color.Red)) = 8 / 105 :=
sorry

end urn_probability_l3762_376200


namespace solution_count_l3762_376227

-- Define the function f
def f (n : ℤ) : ℤ := ⌈(149 * n : ℚ) / 150⌉ - ⌊(150 * n : ℚ) / 151⌋

-- State the theorem
theorem solution_count : 
  (∃ (S : Finset ℤ), (∀ n ∈ S, 1 + ⌊(150 * n : ℚ) / 151⌋ = ⌈(149 * n : ℚ) / 150⌉) ∧ S.card = 15150) :=
sorry

end solution_count_l3762_376227


namespace unique_solution_l3762_376216

/-- The equation holds for all real x -/
def equation_holds (k : ℕ) : Prop :=
  ∀ x : ℝ, (Real.sin x)^k * Real.sin (k * x) + (Real.cos x)^k * Real.cos (k * x) = (Real.cos (2 * x))^k

/-- k = 3 is the only positive integer solution -/
theorem unique_solution :
  ∃! k : ℕ, k > 0 ∧ equation_holds k :=
by sorry

end unique_solution_l3762_376216


namespace gamma_less_than_delta_l3762_376238

open Real

theorem gamma_less_than_delta (α β γ δ : Real) 
  (h1 : 0 < α) (h2 : α < β) (h3 : β < π/2)
  (h4 : 0 < γ) (h5 : γ < π/2)
  (h6 : 0 < δ) (h7 : δ < π/2)
  (h8 : tan γ = (tan α + tan β) / 2)
  (h9 : 1/cos δ = (1/cos α + 1/cos β) / 2) :
  γ < δ := by
sorry


end gamma_less_than_delta_l3762_376238


namespace min_sum_squares_l3762_376289

theorem min_sum_squares (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x + 2*y + 3*z = 2) : 
  x^2 + y^2 + z^2 ≥ 2/7 ∧ 
  (x^2 + y^2 + z^2 = 2/7 ↔ x = 1/7 ∧ y = 2/7 ∧ z = 3/7) :=
sorry

end min_sum_squares_l3762_376289


namespace range_of_a_l3762_376274

def prop_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def prop_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a (a : ℝ) :
  (prop_p a ∧ prop_q a) → (a ≤ -2 ∨ a = 1) :=
sorry

end range_of_a_l3762_376274


namespace late_average_speed_l3762_376298

/-- Proves that the late average speed is 50 kmph given the problem conditions -/
theorem late_average_speed (journey_length : ℝ) (on_time_speed : ℝ) (late_time : ℝ) :
  journey_length = 225 →
  on_time_speed = 60 →
  late_time = 0.75 →
  ∃ v : ℝ, journey_length / on_time_speed + late_time = journey_length / v ∧ v = 50 :=
by sorry

end late_average_speed_l3762_376298


namespace sum_of_distinct_prime_factors_396_l3762_376271

def sum_of_distinct_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

theorem sum_of_distinct_prime_factors_396 :
  sum_of_distinct_prime_factors 396 = 16 := by
  sorry

end sum_of_distinct_prime_factors_396_l3762_376271


namespace charles_whistle_count_l3762_376256

/-- The number of whistles Sean has -/
def sean_whistles : ℕ := 45

/-- The difference between Sean's and Charles' whistles -/
def whistle_difference : ℕ := 32

/-- The number of whistles Charles has -/
def charles_whistles : ℕ := sean_whistles - whistle_difference

theorem charles_whistle_count : charles_whistles = 13 := by
  sorry

end charles_whistle_count_l3762_376256


namespace salad_dressing_ratio_l3762_376233

theorem salad_dressing_ratio (bowl_capacity : ℝ) (oil_density : ℝ) (vinegar_density : ℝ) 
  (total_weight : ℝ) (oil_fraction : ℝ) :
  bowl_capacity = 150 →
  oil_fraction = 2/3 →
  oil_density = 5 →
  vinegar_density = 4 →
  total_weight = 700 →
  (total_weight - oil_fraction * bowl_capacity * oil_density) / vinegar_density / bowl_capacity = 1/3 :=
by sorry

end salad_dressing_ratio_l3762_376233


namespace absolute_value_relation_l3762_376260

theorem absolute_value_relation :
  let p : ℝ → Prop := λ x ↦ |x| = 2
  let q : ℝ → Prop := λ x ↦ x = 2
  (∀ x, q x → p x) ∧ ¬(∀ x, p x → q x) := by
  sorry

end absolute_value_relation_l3762_376260


namespace cos_product_equality_l3762_376297

theorem cos_product_equality : 
  Real.cos (π / 9) * Real.cos (2 * π / 9) * Real.cos (-23 * π / 9) = 1 / 16 := by
  sorry

end cos_product_equality_l3762_376297


namespace not_p_and_not_q_range_l3762_376249

-- Define proposition p
def p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 / (1 - 2*m) + y^2 / (m + 2) = 1

-- Define proposition q
def q (m : ℝ) : Prop := ∃ x₀ : ℝ, x₀^2 + 2*m*x₀ + 2 - m = 0

-- Theorem statement
theorem not_p_and_not_q_range (m : ℝ) :
  (¬p m ∧ ¬q m) ↔ (m > -2 ∧ m ≤ 1/2) :=
sorry

end not_p_and_not_q_range_l3762_376249


namespace carol_rectangle_length_l3762_376280

theorem carol_rectangle_length 
  (carol_width jordan_length jordan_width : ℕ) 
  (h1 : carol_width = 24)
  (h2 : jordan_length = 8)
  (h3 : jordan_width = 15)
  (h4 : carol_width * carol_length = jordan_length * jordan_width) :
  carol_length = 5 :=
by
  sorry

end carol_rectangle_length_l3762_376280


namespace peggy_stamps_to_add_l3762_376254

/-- Given the number of stamps each person has, calculates how many stamps Peggy needs to add to have as many as Bert. -/
def stamps_to_add (peggy_stamps : ℕ) (ernie_multiplier : ℕ) (bert_multiplier : ℕ) : ℕ :=
  bert_multiplier * (ernie_multiplier * peggy_stamps) - peggy_stamps

/-- Proves that Peggy needs to add 825 stamps to have as many as Bert. -/
theorem peggy_stamps_to_add : 
  stamps_to_add 75 3 4 = 825 := by sorry

end peggy_stamps_to_add_l3762_376254


namespace converse_and_inverse_false_l3762_376244

-- Define the types
variable (Quadrilateral : Type)
variable (isRhombus : Quadrilateral → Prop)
variable (isParallelogram : Quadrilateral → Prop)

-- Define the original statement
axiom original_statement : ∀ q : Quadrilateral, isRhombus q → isParallelogram q

-- State the theorem to be proved
theorem converse_and_inverse_false :
  (∀ q : Quadrilateral, isParallelogram q → isRhombus q) = False ∧
  (∀ q : Quadrilateral, ¬isRhombus q → ¬isParallelogram q) = False :=
by sorry

end converse_and_inverse_false_l3762_376244


namespace prob_five_largest_l3762_376218

def card_set : Finset ℕ := Finset.range 6

def selection_size : ℕ := 4

def prob_not_select_6 : ℚ :=
  (5 : ℚ) / 6 * 4 / 5 * 3 / 4 * 2 / 3

def prob_not_select_5_or_6 : ℚ :=
  (4 : ℚ) / 6 * 3 / 5 * 2 / 4 * 1 / 3

theorem prob_five_largest (card_set : Finset ℕ) (selection_size : ℕ) 
  (prob_not_select_6 : ℚ) (prob_not_select_5_or_6 : ℚ) :
  card_set = Finset.range 6 →
  selection_size = 4 →
  prob_not_select_6 = (5 : ℚ) / 6 * 4 / 5 * 3 / 4 * 2 / 3 →
  prob_not_select_5_or_6 = (4 : ℚ) / 6 * 3 / 5 * 2 / 4 * 1 / 3 →
  prob_not_select_6 - prob_not_select_5_or_6 = 4 / 15 := by
  sorry

end prob_five_largest_l3762_376218


namespace distance_to_x_axis_l3762_376286

/-- The distance from a point on the line y = 2x + 1 to the x-axis -/
theorem distance_to_x_axis (k : ℝ) : 
  let M : ℝ × ℝ := (-2, k)
  let line_eq : ℝ → ℝ := λ x => 2 * x + 1
  k = line_eq (-2) →
  |k| = 3 := by
sorry

end distance_to_x_axis_l3762_376286


namespace min_people_liking_both_l3762_376263

theorem min_people_liking_both (total : ℕ) (chopin : ℕ) (beethoven : ℕ) 
  (h1 : total = 120) (h2 : chopin = 95) (h3 : beethoven = 80) :
  ∃ both : ℕ, both ≥ 55 ∧ chopin + beethoven - both ≤ total := by
  sorry

end min_people_liking_both_l3762_376263


namespace divides_prime_expression_l3762_376229

theorem divides_prime_expression (p : Nat) (h1 : p.Prime) (h2 : p > 3) :
  (42 * p) ∣ (3^p - 2^p - 1) := by
  sorry

end divides_prime_expression_l3762_376229


namespace max_min_values_of_f_l3762_376224

-- Define the function
def f (x : ℝ) : ℝ := x^2 - 6*x + 8

-- Define the interval
def interval : Set ℝ := { x | 2 ≤ x ∧ x ≤ 6 }

-- State the theorem
theorem max_min_values_of_f :
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f x ≤ f y) ∧
  (∃ x ∈ interval, f x = 8) ∧
  (∃ x ∈ interval, f x = -1) :=
sorry

end max_min_values_of_f_l3762_376224


namespace existence_of_critical_point_and_upper_bound_l3762_376296

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x - a * x^2 - 2 * x - 1

theorem existence_of_critical_point_and_upper_bound (a : ℝ) (h : 1 < a ∧ a < 2) :
  ∃ x₀ : ℝ, x₀ ∈ Set.Ioo (-1/a) (-1/4) ∧ 
    (deriv (f a)) x₀ = 0 ∧ 
    f a x₀ < 15/16 := by
  sorry

end existence_of_critical_point_and_upper_bound_l3762_376296


namespace pigeonhole_friends_l3762_376237

/-- Represents a class of students -/
structure ClassOfStudents where
  n : ℕ  -- number of students
  h : n > 0  -- ensures the class is not empty

/-- Represents the number of friends each student has -/
def FriendCount (c : ClassOfStudents) := Fin c.n → ℕ

/-- The property that if a student has 0 friends, no one has n-1 friends -/
def ValidFriendCount (c : ClassOfStudents) (f : FriendCount c) : Prop :=
  (∃ i, f i = 0) → ∀ j, f j < c.n - 1

theorem pigeonhole_friends (c : ClassOfStudents) (f : FriendCount c) 
    (hf : ValidFriendCount c f) : 
    ∃ i j, i ≠ j ∧ f i = f j :=
  sorry


end pigeonhole_friends_l3762_376237


namespace equal_squares_count_l3762_376220

/-- Represents a square grid -/
structure Grid (n : ℕ) where
  cells : Fin n → Fin n → Bool

/-- Counts squares with equal black and white cells in a 5x5 grid -/
def count_equal_squares (g : Grid 5) : ℕ :=
  let valid_2x2 : ℕ := 14  -- 16 total - 2 invalid
  let valid_4x4 : ℕ := 2
  valid_2x2 + valid_4x4

/-- Theorem: The number of squares with equal black and white cells is 16 -/
theorem equal_squares_count (g : Grid 5) : count_equal_squares g = 16 := by
  sorry

end equal_squares_count_l3762_376220


namespace travis_potato_probability_l3762_376293

/-- Represents a player in the hot potato game -/
inductive Player : Type
  | George : Player
  | Jeff : Player
  | Brian : Player
  | Travis : Player

/-- The game state after each turn -/
structure GameState :=
  (george_potatoes : Nat)
  (jeff_potatoes : Nat)
  (brian_potatoes : Nat)
  (travis_potatoes : Nat)

/-- The initial game state -/
def initial_state : GameState :=
  ⟨1, 1, 0, 0⟩

/-- The probability of passing a potato to a specific player -/
def pass_probability : ℚ := 1 / 3

/-- The probability of Travis having at least one hot potato after one round -/
def travis_has_potato_probability : ℚ := 5 / 27

/-- Theorem stating the probability of Travis having at least one hot potato after one round -/
theorem travis_potato_probability :
  travis_has_potato_probability = 5 / 27 :=
by sorry


end travis_potato_probability_l3762_376293


namespace triangle_inequality_l3762_376294

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  a + b > c ∧ b + c > a ∧ c + a > b :=
by
  sorry

end triangle_inequality_l3762_376294


namespace stock_worth_l3762_376221

def total_modules : ℕ := 11
def cheap_modules : ℕ := 10
def expensive_modules : ℕ := total_modules - cheap_modules
def cheap_cost : ℚ := 3.5
def expensive_cost : ℚ := 10

def total_worth : ℚ := cheap_modules * cheap_cost + expensive_modules * expensive_cost

theorem stock_worth : total_worth = 45 := by
  sorry

end stock_worth_l3762_376221


namespace max_bouquets_sara_l3762_376253

def red_flowers : ℕ := 47
def yellow_flowers : ℕ := 63
def blue_flowers : ℕ := 54
def orange_flowers : ℕ := 29
def pink_flowers : ℕ := 36

theorem max_bouquets_sara :
  ∀ n : ℕ,
    n ≤ red_flowers ∧
    n ≤ yellow_flowers ∧
    n ≤ blue_flowers ∧
    n ≤ orange_flowers ∧
    n ≤ pink_flowers →
    n ≤ 1 :=
by sorry

end max_bouquets_sara_l3762_376253


namespace julio_lime_cost_l3762_376266

/-- Represents the cost of limes for Julio's mocktails over 30 days -/
def lime_cost (mocktails_per_day : ℕ) (lime_juice_per_mocktail : ℚ) (juice_per_lime : ℚ) (days : ℕ) (limes_per_dollar : ℕ) : ℚ :=
  let limes_needed := (mocktails_per_day * lime_juice_per_mocktail * days) / juice_per_lime
  let lime_sets := (limes_needed / limes_per_dollar).ceil
  lime_sets

theorem julio_lime_cost :
  lime_cost 1 (1/2) 2 30 3 = 5 := by
  sorry

#eval lime_cost 1 (1/2) 2 30 3

end julio_lime_cost_l3762_376266


namespace set_with_gcd_property_has_power_of_two_elements_l3762_376248

theorem set_with_gcd_property_has_power_of_two_elements (S : Finset ℕ+) 
  (h : ∀ (s : ℕ+) (d : ℕ+), s ∈ S → d ∣ s → ∃! (t : ℕ+), t ∈ S ∧ Nat.gcd s.val t.val = d.val) :
  ∃ (k : ℕ), S.card = 2^k :=
sorry

end set_with_gcd_property_has_power_of_two_elements_l3762_376248


namespace part_one_part_two_l3762_376214

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def B (m : ℝ) : Set ℝ := {x | m - 2 ≤ x ∧ x ≤ 2 * m}

-- Theorem for part 1
theorem part_one :
  A ∩ (U \ B 3) = {x | 0 ≤ x ∧ x < 1} := by sorry

-- Theorem for part 2
theorem part_two :
  ∀ m : ℝ, A ∪ B m = B m ↔ 3/2 ≤ m ∧ m ≤ 2 := by sorry

end part_one_part_two_l3762_376214


namespace first_term_of_arithmetic_sequence_l3762_376258

/-- An arithmetic sequence with given properties -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem first_term_of_arithmetic_sequence
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_a3 : a 3 = 2)
  (h_d : ∃ d : ℚ, d = -1/2 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  a 1 = 3 :=
sorry

end first_term_of_arithmetic_sequence_l3762_376258


namespace trig_equation_solution_l3762_376251

theorem trig_equation_solution (x : ℝ) : 
  12 * Real.sin x - 5 * Real.cos x = 13 ↔ 
  ∃ k : ℤ, x = π / 2 + Real.arctan (5 / 12) + 2 * k * π :=
sorry

end trig_equation_solution_l3762_376251


namespace closest_value_is_112_l3762_376205

def original_value : ℝ := 50.5
def increase_percentage : ℝ := 0.05
def additional_value : ℝ := 0.15
def multiplier : ℝ := 2.1

def options : List ℝ := [105, 110, 112, 115, 120]

def calculated_value : ℝ := multiplier * ((original_value * (1 + increase_percentage)) + additional_value)

theorem closest_value_is_112 : 
  (options.argmin (λ x => |x - calculated_value|)) = some 112 := by sorry

end closest_value_is_112_l3762_376205


namespace import_tax_problem_l3762_376206

theorem import_tax_problem (tax_rate : ℝ) (tax_threshold : ℝ) (tax_paid : ℝ) (total_value : ℝ) : 
  tax_rate = 0.07 →
  tax_threshold = 1000 →
  tax_paid = 109.90 →
  tax_paid = tax_rate * (total_value - tax_threshold) →
  total_value = 2570 := by
  sorry

end import_tax_problem_l3762_376206


namespace two_forty_is_eighty_supplement_of_half_forty_is_onesixtey_half_forty_is_twenty_l3762_376232

-- Define the given angle
def given_angle : ℝ := 40

-- Theorem 1: Two 40° angles form an 80° angle
theorem two_forty_is_eighty : given_angle + given_angle = 80 := by sorry

-- Theorem 2: The supplement of half of a 40° angle is 160°
theorem supplement_of_half_forty_is_onesixtey : 180 - (given_angle / 2) = 160 := by sorry

-- Theorem 3: Half of a 40° angle is 20°
theorem half_forty_is_twenty : given_angle / 2 = 20 := by sorry

end two_forty_is_eighty_supplement_of_half_forty_is_onesixtey_half_forty_is_twenty_l3762_376232


namespace product_equals_square_l3762_376250

theorem product_equals_square : 50 * 24.96 * 2.496 * 500 = (1248 : ℝ)^2 := by sorry

end product_equals_square_l3762_376250


namespace tips_fraction_l3762_376278

/-- Given a worker who works for 7 months, with one month's tips being twice
    the average of the other 6 months, the fraction of total tips from that
    one month is 1/4. -/
theorem tips_fraction (total_months : ℕ) (special_month_tips : ℝ) 
    (other_months_tips : ℝ) : 
    total_months = 7 →
    special_month_tips = 2 * (other_months_tips / 6) →
    special_month_tips / (special_month_tips + other_months_tips) = 1/4 := by
  sorry

end tips_fraction_l3762_376278


namespace polynomial_simplification_l3762_376261

theorem polynomial_simplification (x : ℝ) : 
  (6 * x^10 + 8 * x^9 + 3 * x^7) + (2 * x^12 + 3 * x^10 + x^9 + 5 * x^7 + 4 * x^4 + 7 * x + 6) = 
  2 * x^12 + 9 * x^10 + 9 * x^9 + 8 * x^7 + 4 * x^4 + 7 * x + 6 := by
  sorry

end polynomial_simplification_l3762_376261


namespace smallest_w_l3762_376219

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_w : ∃ (w : ℕ), 
  w > 0 ∧
  is_factor (2^4) (1452 * w) ∧
  is_factor (3^3) (1452 * w) ∧
  is_factor (13^3) (1452 * w) ∧
  ∀ (x : ℕ), x > 0 ∧ 
    is_factor (2^4) (1452 * x) ∧
    is_factor (3^3) (1452 * x) ∧
    is_factor (13^3) (1452 * x) →
    w ≤ x :=
by
  -- Proof goes here
  sorry

end smallest_w_l3762_376219


namespace minimum_pieces_to_capture_all_l3762_376264

/-- Represents a rhombus-shaped game board -/
structure RhombusBoard where
  angle : ℝ
  side_divisions : ℕ

/-- Represents a piece on the game board -/
structure GamePiece where
  position : ℕ × ℕ

/-- Represents the set of cells captured by a piece -/
def captured_cells (board : RhombusBoard) (piece : GamePiece) : Set (ℕ × ℕ) :=
  sorry

/-- The total number of cells on the board -/
def total_cells (board : RhombusBoard) : ℕ :=
  sorry

/-- Checks if a set of pieces captures all cells on the board -/
def captures_all_cells (board : RhombusBoard) (pieces : List GamePiece) : Prop :=
  sorry

theorem minimum_pieces_to_capture_all (board : RhombusBoard)
  (h1 : board.angle = 60)
  (h2 : board.side_divisions = 9) :
  ∃ (pieces : List GamePiece),
    pieces.length = 6 ∧
    captures_all_cells board pieces ∧
    ∀ (other_pieces : List GamePiece),
      captures_all_cells board other_pieces →
      other_pieces.length ≥ 6 :=
  sorry

end minimum_pieces_to_capture_all_l3762_376264


namespace condition_neither_sufficient_nor_necessary_l3762_376212

theorem condition_neither_sufficient_nor_necessary 
  (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (ha₁ : a₁ ≠ 0) (hb₁ : b₁ ≠ 0) (hc₁ : c₁ ≠ 0)
  (ha₂ : a₂ ≠ 0) (hb₂ : b₂ ≠ 0) (hc₂ : c₂ ≠ 0)
  (M : Set ℝ) (hM : M = {x : ℝ | a₁ * x^2 + b₁ * x + c₁ > 0})
  (N : Set ℝ) (hN : N = {x : ℝ | a₂ * x^2 + b₂ * x + c₂ > 0}) :
  ¬(((a₁ / a₂ = b₁ / b₂) ∧ (b₁ / b₂ = c₁ / c₂)) → (M = N)) ∧ 
  ¬((M = N) → ((a₁ / a₂ = b₁ / b₂) ∧ (b₁ / b₂ = c₁ / c₂))) :=
sorry

end condition_neither_sufficient_nor_necessary_l3762_376212


namespace family_savings_theorem_l3762_376209

/-- Represents the monthly financial data of Ivan Tsarevich's family -/
structure FamilyFinances where
  ivan_salary : ℝ
  vasilisa_salary : ℝ
  mother_salary : ℝ
  father_salary : ℝ
  son_scholarship : ℝ
  monthly_expenses : ℝ
  tax_rate : ℝ

def calculate_net_income (gross_income : ℝ) (tax_rate : ℝ) : ℝ :=
  gross_income * (1 - tax_rate)

def calculate_total_net_income (f : FamilyFinances) : ℝ :=
  calculate_net_income f.ivan_salary f.tax_rate +
  calculate_net_income f.vasilisa_salary f.tax_rate +
  calculate_net_income f.mother_salary f.tax_rate +
  calculate_net_income f.father_salary f.tax_rate +
  f.son_scholarship

def calculate_monthly_savings (f : FamilyFinances) : ℝ :=
  calculate_total_net_income f - f.monthly_expenses

theorem family_savings_theorem (f : FamilyFinances) 
  (h1 : f.ivan_salary = 55000)
  (h2 : f.vasilisa_salary = 45000)
  (h3 : f.mother_salary = 18000)
  (h4 : f.father_salary = 20000)
  (h5 : f.son_scholarship = 3000)
  (h6 : f.monthly_expenses = 74000)
  (h7 : f.tax_rate = 0.13) :
  calculate_monthly_savings f = 49060 ∧
  calculate_monthly_savings { f with 
    mother_salary := 10000,
    son_scholarship := 3000 } = 43400 ∧
  calculate_monthly_savings { f with 
    mother_salary := 10000,
    son_scholarship := 16050 } = 56450 := by
  sorry

#check family_savings_theorem

end family_savings_theorem_l3762_376209


namespace cubic_equation_with_double_root_l3762_376290

theorem cubic_equation_with_double_root (k : ℝ) : 
  (∃ a b : ℝ, (3 * a^3 - 9 * a^2 - 81 * a + k = 0) ∧ 
               (3 * (2*a)^3 - 9 * (2*a)^2 - 81 * (2*a) + k = 0) ∧ 
               (3 * b^3 - 9 * b^2 - 81 * b + k = 0) ∧ 
               (a ≠ b) ∧ (k > 0)) →
  k = -6 * ((9 + Real.sqrt 837) / 14)^2 * (3 - 3 * ((9 + Real.sqrt 837) / 14)) :=
by sorry

end cubic_equation_with_double_root_l3762_376290


namespace elaines_rent_percentage_l3762_376243

/-- Proves that given the conditions in the problem, Elaine spent 20% of her annual earnings on rent last year. -/
theorem elaines_rent_percentage (E : ℝ) (P : ℝ) : 
  E > 0 → -- Elaine's earnings last year (assumed positive)
  0.30 * (1.35 * E) = 2.025 * (P / 100 * E) → -- Condition relating this year's and last year's rent
  P = 20 := by sorry

end elaines_rent_percentage_l3762_376243


namespace hyperbola_equation_l3762_376268

/-- A hyperbola with focal length 2√5 and asymptote x - 2y = 0 has equation x^2/4 - y^2 = 1 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →  -- Given hyperbola equation
  (a^2 + b^2 = 5) →                         -- Focal length condition
  (a = 2 * b) →                             -- Asymptote condition
  (∀ x y : ℝ, x^2 / 4 - y^2 = 1) :=         -- Conclusion: specific hyperbola equation
by sorry

end hyperbola_equation_l3762_376268


namespace inequality_solution_l3762_376230

theorem inequality_solution (x : ℝ) : 2 ≤ x / (2 * x - 5) ∧ x / (2 * x - 5) < 7 ↔ x ∈ Set.Ioo (35 / 13) (10 / 3) := by
  sorry

end inequality_solution_l3762_376230


namespace range_of_a_l3762_376291

/-- A monotonically decreasing function on [-2, 2] -/
def MonoDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ Set.Icc (-2) 2 → y ∈ Set.Icc (-2) 2 → x < y → f x > f y

theorem range_of_a (f : ℝ → ℝ) (h1 : MonoDecreasing f) 
    (h2 : ∀ a, f (a + 1) < f (2 * a)) :
    Set.Icc (-1) 1 \ {1} = {a | a + 1 ∈ Set.Icc (-2) 2 ∧ 2 * a ∈ Set.Icc (-2) 2 ∧ f (a + 1) < f (2 * a)} :=
  sorry

end range_of_a_l3762_376291


namespace intersection_subset_l3762_376299

def P : Set ℝ := {x : ℝ | 2 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {1, 2, 3}

theorem intersection_subset : P ∩ Q ⊆ Q := by
  sorry

end intersection_subset_l3762_376299


namespace geometric_sequence_common_ratio_l3762_376287

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a) 
  (h_incr : is_increasing_sequence a) 
  (h_a2 : a 2 = 2) 
  (h_diff : a 4 - a 3 = 4) : 
  ∃ q : ℝ, q = 2 ∧ ∀ n : ℕ, a (n + 1) = a n * q := by
  sorry

end geometric_sequence_common_ratio_l3762_376287


namespace smallest_marked_cells_for_unique_tiling_l3762_376279

/-- Represents a grid of size 2n × 2n -/
structure Grid (n : ℕ) where
  size : ℕ := 2 * n

/-- Represents a set of marked cells in the grid -/
def MarkedCells (n : ℕ) := Finset (Fin (2 * n) × Fin (2 * n))

/-- Represents a domino tiling of the grid -/
def Tiling (n : ℕ) := Finset (Fin (2 * n) × Fin (2 * n) × Bool)

/-- Checks if a tiling is valid for a given set of marked cells -/
def isValidTiling (n : ℕ) (marked : MarkedCells n) (tiling : Tiling n) : Prop :=
  sorry

/-- Checks if there exists a unique valid tiling for a given set of marked cells -/
def hasUniqueTiling (n : ℕ) (marked : MarkedCells n) : Prop :=
  sorry

/-- The main theorem: The smallest number of marked cells that ensures a unique tiling is 2n -/
theorem smallest_marked_cells_for_unique_tiling (n : ℕ) (h : 0 < n) :
  ∃ (marked : MarkedCells n),
    marked.card = 2 * n ∧
    hasUniqueTiling n marked ∧
    ∀ (marked' : MarkedCells n),
      marked'.card < 2 * n → ¬(hasUniqueTiling n marked') :=
  sorry

end smallest_marked_cells_for_unique_tiling_l3762_376279


namespace yellow_balls_percentage_l3762_376217

/-- The percentage of yellow balls in a collection of colored balls. -/
def percentage_yellow_balls (yellow brown blue green : ℕ) : ℚ :=
  (yellow : ℚ) / ((yellow + brown + blue + green : ℕ) : ℚ) * 100

/-- Theorem stating that the percentage of yellow balls is 25% given the specific numbers. -/
theorem yellow_balls_percentage :
  percentage_yellow_balls 75 120 45 60 = 25 := by
  sorry

end yellow_balls_percentage_l3762_376217


namespace tetrahedron_volume_and_height_l3762_376270

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Calculates the volume of a tetrahedron given its four vertices -/
def tetrahedronVolume (A₁ A₂ A₃ A₄ : Point3D) : ℝ :=
  sorry

/-- Calculates the height of a tetrahedron from a vertex to the opposite face -/
def tetrahedronHeight (A₁ A₂ A₃ A₄ : Point3D) : ℝ :=
  sorry

/-- Theorem stating the volume and height of a specific tetrahedron -/
theorem tetrahedron_volume_and_height :
  let A₁ : Point3D := ⟨2, 3, 1⟩
  let A₂ : Point3D := ⟨4, 1, -2⟩
  let A₃ : Point3D := ⟨6, 3, 7⟩
  let A₄ : Point3D := ⟨7, 5, -3⟩
  (tetrahedronVolume A₁ A₂ A₃ A₄ = 70/3) ∧
  (tetrahedronHeight A₁ A₂ A₃ A₄ = 5) :=
by
  sorry

end tetrahedron_volume_and_height_l3762_376270


namespace dot_product_condition_l3762_376211

/-- Given vectors a and b, if a · (2a - b) = 0, then k = 12 -/
theorem dot_product_condition (a b : ℝ × ℝ) (k : ℝ) 
  (h1 : a = (2, 1))
  (h2 : b = (-1, k))
  (h3 : a • (2 • a - b) = 0) :
  k = 12 := by sorry

end dot_product_condition_l3762_376211


namespace optimal_position_C_l3762_376267

/-- The optimal position of point C on segment AB to maximize the length of CD -/
theorem optimal_position_C (t : ℝ) : 
  (0 ≤ t) → (t < 1) → 
  (∀ s, (0 ≤ s ∧ s < 1) → (t * (1 - t^2) / 4 ≥ s * (1 - s^2) / 4)) → 
  t = 1 / Real.sqrt 3 := by
  sorry

#check optimal_position_C

end optimal_position_C_l3762_376267


namespace sum_zero_ratio_negative_half_l3762_376208

theorem sum_zero_ratio_negative_half 
  (w x y z : ℝ) 
  (hw : w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z) 
  (hsum : w + x + y + z = 0) :
  (w * y + x * z) / (w^2 + x^2 + y^2 + z^2) = -1/2 := by
  sorry

end sum_zero_ratio_negative_half_l3762_376208


namespace triangle_properties_l3762_376269

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : Real.sin t.C * Real.sin (t.A - t.B) = Real.sin t.B * Real.sin (t.C - t.A)) 
  (h2 : t.a = 5)
  (h3 : Real.cos t.A = 25 / 31) :
  (2 * t.a^2 = t.b^2 + t.c^2) ∧ 
  (t.a + t.b + t.c = 14) := by
  sorry


end triangle_properties_l3762_376269


namespace multiply_and_simplify_l3762_376245

theorem multiply_and_simplify (x y z : ℝ) :
  (3 * x^2 * z - 7 * y^3) * (9 * x^4 * z^2 + 21 * x^2 * y * z^3 + 49 * y^6) = 27 * x^6 * z^3 - 343 * y^9 := by
  sorry

end multiply_and_simplify_l3762_376245


namespace problem_solution_l3762_376265

noncomputable def f (a : ℝ) (θ : ℝ) (x : ℝ) : ℝ :=
  (a + 2 * (Real.cos x)^2) * Real.cos (2 * x + θ)

theorem problem_solution (a θ α : ℝ) :
  (∀ x, f a θ x = -f a θ (-x)) →  -- f is an odd function
  f a θ (π/4) = 0 →
  θ ∈ Set.Ioo 0 π →
  f a θ (α/4) = -2/5 →
  α ∈ Set.Ioo (π/2) π →
  (a = -1 ∧ θ = π/2 ∧ Real.sin (α + π/3) = (4 - 3 * Real.sqrt 3) / 10) := by
  sorry

end problem_solution_l3762_376265


namespace cylinder_base_area_l3762_376284

/-- Represents a container with a base area and height increase when a stone is submerged -/
structure Container where
  base_area : ℝ
  height_increase : ℝ

/-- Proves that the base area of the cylinder is 42 square centimeters -/
theorem cylinder_base_area
  (cylinder : Container)
  (prism : Container)
  (h1 : cylinder.height_increase = 8)
  (h2 : prism.height_increase = 6)
  (h3 : cylinder.base_area + prism.base_area = 98)
  : cylinder.base_area = 42 := by
  sorry

end cylinder_base_area_l3762_376284


namespace equation_solution_l3762_376223

theorem equation_solution :
  ∃ x : ℚ, (x + 2 ≠ 0 ∧ 3 - x ≠ 0) ∧
  ((3 * x - 5) / (x + 2) + (3 * x - 9) / (3 - x) = 2) ∧
  x = -15 / 2 :=
by sorry

end equation_solution_l3762_376223


namespace exam_students_count_l3762_376201

/-- The total number of students in an examination -/
def total_students : ℕ := 300

/-- The number of students who just passed -/
def students_just_passed : ℕ := 60

/-- The percentage of students who got first division -/
def first_division_percent : ℚ := 26 / 100

/-- The percentage of students who got second division -/
def second_division_percent : ℚ := 54 / 100

/-- Theorem stating that the total number of students is 300 -/
theorem exam_students_count :
  (students_just_passed : ℚ) / total_students = 1 - first_division_percent - second_division_percent :=
by sorry

end exam_students_count_l3762_376201


namespace largest_four_digit_sum_19_l3762_376226

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem largest_four_digit_sum_19 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 19 → n ≤ 9910 :=
by sorry

end largest_four_digit_sum_19_l3762_376226


namespace inheritance_tax_problem_l3762_376207

theorem inheritance_tax_problem (x : ℝ) : 
  (0.25 * x + 0.15 * (x - 0.25 * x) = 19500) → x = 53800 := by
  sorry

end inheritance_tax_problem_l3762_376207


namespace negation_of_existential_proposition_l3762_376252

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, abs x > 0) ↔ (∀ x : ℝ, ¬(abs x > 0)) :=
by sorry

end negation_of_existential_proposition_l3762_376252


namespace janabel_widget_sales_l3762_376277

def arithmeticSequenceSum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem janabel_widget_sales : 
  let a₁ : ℕ := 2  -- First day sales
  let d : ℕ := 3   -- Daily increase
  let n : ℕ := 15  -- Number of days
  let bonus : ℕ := 1  -- Bonus widget on last day
  arithmeticSequenceSum a₁ d n + bonus = 346 :=
by
  sorry

#check janabel_widget_sales

end janabel_widget_sales_l3762_376277


namespace reciprocal_of_negative_two_l3762_376273

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- State the theorem
theorem reciprocal_of_negative_two :
  reciprocal (-2) = -1/2 := by sorry

end reciprocal_of_negative_two_l3762_376273


namespace chrome_parts_total_l3762_376222

/-- Represents the number of machines of type A -/
def a : ℕ := sorry

/-- Represents the number of machines of type B -/
def b : ℕ := sorry

/-- The total number of machines -/
def total_machines : ℕ := 21

/-- The total number of steel parts -/
def total_steel : ℕ := 50

/-- The number of steel parts in a type A machine -/
def steel_parts_A : ℕ := 3

/-- The number of steel parts in a type B machine -/
def steel_parts_B : ℕ := 2

/-- The number of chrome parts in a type A machine -/
def chrome_parts_A : ℕ := 2

/-- The number of chrome parts in a type B machine -/
def chrome_parts_B : ℕ := 4

theorem chrome_parts_total : 
  a + b = total_machines ∧ 
  steel_parts_A * a + steel_parts_B * b = total_steel →
  chrome_parts_A * a + chrome_parts_B * b = 68 := by
  sorry

end chrome_parts_total_l3762_376222


namespace complement_of_P_intersection_P_M_range_of_a_l3762_376285

-- Define the sets P and M
def P : Set ℝ := {x | x * (x - 2) ≥ 0}
def M (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 3}

-- Theorem for the complement of P
theorem complement_of_P : 
  (Set.univ : Set ℝ) \ P = {x | 0 < x ∧ x < 2} := by sorry

-- Theorem for the intersection of P and M when a = 1
theorem intersection_P_M : 
  P ∩ M 1 = {x | 2 ≤ x ∧ x < 4} := by sorry

-- Theorem for the range of a when ∁ₗP ⊆ M
theorem range_of_a (a : ℝ) : 
  ((Set.univ : Set ℝ) \ P) ⊆ M a ↔ -1 ≤ a ∧ a ≤ 0 := by sorry

end complement_of_P_intersection_P_M_range_of_a_l3762_376285


namespace female_student_stats_l3762_376295

/-- Represents the class statistics -/
structure ClassStats where
  total_students : ℕ
  male_students : ℕ
  overall_avg_score : ℚ
  male_algebra_avg : ℚ
  male_geometry_avg : ℚ
  male_calculus_avg : ℚ
  female_algebra_avg : ℚ
  female_geometry_avg : ℚ
  female_calculus_avg : ℚ
  algebra_geometry_attendance : ℚ
  calculus_attendance_increase : ℚ

/-- Theorem stating the proportion and number of female students -/
theorem female_student_stats (stats : ClassStats)
  (h_total : stats.total_students = 30)
  (h_male : stats.male_students = 8)
  (h_overall_avg : stats.overall_avg_score = 90)
  (h_male_algebra : stats.male_algebra_avg = 87)
  (h_male_geometry : stats.male_geometry_avg = 95)
  (h_male_calculus : stats.male_calculus_avg = 89)
  (h_female_algebra : stats.female_algebra_avg = 92)
  (h_female_geometry : stats.female_geometry_avg = 94)
  (h_female_calculus : stats.female_calculus_avg = 91)
  (h_alg_geo_attendance : stats.algebra_geometry_attendance = 85)
  (h_calc_attendance : stats.calculus_attendance_increase = 4) :
  (stats.total_students - stats.male_students : ℚ) / stats.total_students = 11 / 15 ∧
  stats.total_students - stats.male_students = 22 := by
    sorry


end female_student_stats_l3762_376295


namespace geometric_sequence_iff_k_eq_neg_one_l3762_376247

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

def S (n : ℕ) (k : ℝ) : ℝ := 3^n + k

def a (n : ℕ) (k : ℝ) : ℝ :=
  if n = 1 then S 1 k else S n k - S (n-1) k

theorem geometric_sequence_iff_k_eq_neg_one (k : ℝ) :
  is_geometric_sequence (a · k) ↔ k = -1 :=
sorry

end geometric_sequence_iff_k_eq_neg_one_l3762_376247


namespace polygon_side_length_theorem_l3762_376255

/-- A convex polygon that can be divided into unit equilateral triangles and unit squares -/
structure ConvexPolygon where
  sides : List ℕ
  is_convex : Bool

/-- The number of ways to divide a ConvexPolygon into unit equilateral triangles and unit squares -/
def divisionWays (M : ConvexPolygon) : ℕ := sorry

theorem polygon_side_length_theorem (M : ConvexPolygon) (p : ℕ) (h_prime : Nat.Prime p) :
  divisionWays M = p → ∃ (side : ℕ), side ∈ M.sides ∧ side = p - 1 := by sorry

end polygon_side_length_theorem_l3762_376255


namespace sneakers_cost_l3762_376246

/-- The cost of sneakers calculated from lawn mowing charges -/
theorem sneakers_cost (charge_per_yard : ℝ) (yards_to_cut : ℕ) : 
  charge_per_yard * (yards_to_cut : ℝ) = 12.90 :=
by
  sorry

#check sneakers_cost 2.15 6

end sneakers_cost_l3762_376246


namespace gcd_problem_l3762_376228

theorem gcd_problem (b : ℤ) (h : ∃ k : ℤ, b = 2 * k * 431) :
  Int.gcd (8 * b^2 + 63 * b + 143) (4 * b + 17) = 1 := by
  sorry

end gcd_problem_l3762_376228


namespace chord_segment_ratio_l3762_376202

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a chord
structure Chord (c : Circle) where
  p1 : Point
  p2 : Point

-- Define the intersection of two chords
def chord_intersection (c : Circle) (ch1 ch2 : Chord c) : Point := sorry

-- Power of a Point theorem
axiom power_of_point (c : Circle) (ch1 ch2 : Chord c) (q : Point) :
  let x := chord_intersection c ch1 ch2
  (x.1 - q.1) * (ch1.p2.1 - q.1) = (x.1 - q.1) * (ch2.p2.1 - q.1)

-- Main theorem
theorem chord_segment_ratio (c : Circle) (ch1 ch2 : Chord c) :
  let q := chord_intersection c ch1 ch2
  let x := ch1.p1
  let y := ch1.p2
  let w := ch2.p1
  let z := ch2.p2
  (x.1 - q.1) = 5 →
  (w.1 - q.1) = 7 →
  (y.1 - q.1) / (z.1 - q.1) = 7 / 5 := by
  sorry

end chord_segment_ratio_l3762_376202


namespace coin_value_calculation_l3762_376281

theorem coin_value_calculation (num_quarters num_nickels : ℕ) 
  (quarter_value nickel_value : ℚ) : 
  num_quarters = 8 → 
  num_nickels = 13 → 
  quarter_value = 25 / 100 → 
  nickel_value = 5 / 100 → 
  num_quarters * quarter_value + num_nickels * nickel_value = 265 / 100 := by
  sorry

end coin_value_calculation_l3762_376281


namespace trigonometric_simplification_l3762_376225

theorem trigonometric_simplification (α : ℝ) : 
  (2 * Real.sin (π - α) + Real.sin (2 * α)) / (2 * (Real.cos (α / 2))^2) = 2 * Real.sin α := by
  sorry

end trigonometric_simplification_l3762_376225


namespace dinner_seating_arrangements_l3762_376203

theorem dinner_seating_arrangements (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 7) :
  (Nat.choose n k) * Nat.factorial (k - 1) = 25920 := by
  sorry

end dinner_seating_arrangements_l3762_376203


namespace cans_purchased_theorem_l3762_376236

/-- The number of cans that can be purchased given the conditions -/
def cans_purchased (N P T : ℚ) : ℚ :=
  5 * N * (T - 1) / P

/-- Theorem stating the number of cans that can be purchased under given conditions -/
theorem cans_purchased_theorem (N P T : ℚ) 
  (h_positive : N > 0 ∧ P > 0 ∧ T > 1) 
  (h_N_P : N / P > 0) -- N cans can be purchased for P quarters
  (h_dollar_worth : (1 : ℚ) = 5 / 4) -- 1 dollar is worth 5 quarters
  (h_fee : (1 : ℚ) > 0) -- There is a 1 dollar fee per transaction
  : cans_purchased N P T = 5 * N * (T - 1) / P :=
sorry

end cans_purchased_theorem_l3762_376236


namespace expression_simplification_l3762_376292

theorem expression_simplification (a b : ℝ) (h : a / b = 1 / 3) :
  1 - (a - b) / (a - 2 * b) / ((a^2 - b^2) / (a^2 - 4 * a * b + 4 * b^2)) = 3 / 4 := by
  sorry

end expression_simplification_l3762_376292


namespace max_value_theorem_min_value_theorem_l3762_376231

/-- Given a > b > 0 and 7a² + 8ab + 4b² = 24, the maximum value of 3a + 2b occurs when b = √2/2 -/
theorem max_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 7 * a^2 + 8 * a * b + 4 * b^2 = 24) :
  (∀ a' b', a' > b' ∧ b' > 0 ∧ 7 * a'^2 + 8 * a' * b' + 4 * b'^2 = 24 → 3 * a' + 2 * b' ≤ 3 * a + 2 * b) →
  b = Real.sqrt 2 / 2 :=
sorry

/-- Given a > b > 0 and 1/(a - b) + 1/b = 1, the minimum value of a + 3b is 9 -/
theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : 1 / (a - b) + 1 / b = 1) :
  a + 3 * b ≥ 9 :=
sorry

end max_value_theorem_min_value_theorem_l3762_376231


namespace overlap_length_l3762_376282

/-- Given a set of overlapping red segments, this theorem proves the length of each overlap. -/
theorem overlap_length (total_length : ℝ) (end_to_end : ℝ) (num_overlaps : ℕ) : 
  total_length = 98 →
  end_to_end = 83 →
  num_overlaps = 6 →
  (total_length - end_to_end) / num_overlaps = 2.5 := by
  sorry

end overlap_length_l3762_376282


namespace square_perimeter_l3762_376276

theorem square_perimeter (s : Real) : 
  s > 0 → 
  (5 * s / 2 = 32) → 
  (4 * s = 51.2) := by
  sorry

end square_perimeter_l3762_376276


namespace sufficient_not_necessary_l3762_376210

theorem sufficient_not_necessary (a : ℝ) :
  (a < -1 → ∃ x₀ : ℝ, a * Real.sin x₀ + 1 < 0) ∧
  (∃ a : ℝ, a ≥ -1 ∧ ∃ x₀ : ℝ, a * Real.sin x₀ + 1 < 0) :=
by sorry

end sufficient_not_necessary_l3762_376210


namespace lines_parallel_iff_same_slope_diff_intercept_l3762_376234

/-- Two lines in slope-intercept form are parallel if and only if they have the same slope and different y-intercepts -/
theorem lines_parallel_iff_same_slope_diff_intercept (k₁ k₂ l₁ l₂ : ℝ) :
  (∀ x y : ℝ, y = k₁ * x + l₁ ∨ y = k₂ * x + l₂) →
  (∀ x y : ℝ, y = k₁ * x + l₁ → y = k₂ * x + l₂ → False) ↔ k₁ = k₂ ∧ l₁ ≠ l₂ :=
by sorry

end lines_parallel_iff_same_slope_diff_intercept_l3762_376234


namespace touching_sphere_surface_area_l3762_376283

/-- A sphere touching a rectangle and additional segments -/
structure TouchingSphere where
  -- The rectangle ABCD
  ab : ℝ
  bc : ℝ
  -- The segment EF
  ef : ℝ
  -- EF is parallel to the plane of ABCD
  ef_parallel : True
  -- All sides of ABCD and segments AE, BE, CF, DF, EF touch the sphere
  all_touch : True
  -- Given conditions
  ef_length : ef = 3
  bc_length : bc = 5

/-- The surface area of the sphere is 180π/7 -/
theorem touching_sphere_surface_area (s : TouchingSphere) : 
  ∃ (r : ℝ), 4 * Real.pi * r^2 = (180 * Real.pi) / 7 :=
sorry

end touching_sphere_surface_area_l3762_376283


namespace sum_x_y_equals_ten_l3762_376242

theorem sum_x_y_equals_ten (x y : ℝ) 
  (h1 : |x| - x + y = 6)
  (h2 : x + |y| + y = 16) :
  x + y = 10 := by
sorry

end sum_x_y_equals_ten_l3762_376242


namespace sin_cos_value_l3762_376262

theorem sin_cos_value (θ : Real) (h : Real.tan θ = 2) : Real.sin θ * Real.cos θ = 2/5 := by
  sorry

end sin_cos_value_l3762_376262


namespace sarah_copies_360_pages_l3762_376241

/-- The number of copies Sarah needs to make for each person -/
def copies_per_person : ℕ := 2

/-- The number of people in the meeting -/
def number_of_people : ℕ := 9

/-- The number of pages in the contract -/
def pages_per_contract : ℕ := 20

/-- The total number of pages Sarah will copy -/
def total_pages : ℕ := copies_per_person * number_of_people * pages_per_contract

/-- Theorem stating that the total number of pages Sarah will copy is 360 -/
theorem sarah_copies_360_pages : total_pages = 360 := by
  sorry

end sarah_copies_360_pages_l3762_376241


namespace inequality_proof_l3762_376288

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) :
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
by sorry

end inequality_proof_l3762_376288


namespace tangent_circles_area_l3762_376235

/-- The area of the region outside a circle of radius 2 and inside two circles of radius 3
    that are internally tangent to the smaller circle at opposite ends of its diameter -/
theorem tangent_circles_area : Real :=
  let r₁ : Real := 2  -- radius of smaller circle
  let r₂ : Real := 3  -- radius of larger circles
  let total_area : Real := (5 * Real.pi) / 2 - 4 * Real.sqrt 5
  total_area

#check tangent_circles_area

end tangent_circles_area_l3762_376235


namespace collinear_vectors_magnitude_l3762_376240

/-- Given two planar vectors a and b that are collinear and have a negative dot product,
    prove that the magnitude of b is 2√2. -/
theorem collinear_vectors_magnitude (m : ℝ) :
  let a : ℝ × ℝ := (2 * m + 1, 3)
  let b : ℝ × ℝ := (2, m)
  (∃ (k : ℝ), a = k • b) →  -- collinearity condition
  (a.1 * b.1 + a.2 * b.2 < 0) →  -- dot product condition
  Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 2 := by
sorry

end collinear_vectors_magnitude_l3762_376240


namespace tasty_candy_identification_l3762_376259

/-- Represents a strategy for identifying tasty candies -/
structure TastyStrategy where
  query : (ℕ → Bool) → Finset ℕ → ℕ
  interpret : (Finset ℕ → ℕ) → Finset ℕ

/-- The total number of candies -/
def total_candies : ℕ := 28

/-- A function that determines if a candy is tasty -/
def is_tasty : ℕ → Bool := sorry

/-- The maximum number of queries allowed -/
def max_queries : ℕ := 20

theorem tasty_candy_identification :
  ∃ (s : TastyStrategy),
    (∀ (f : ℕ → Bool),
      let query_count := (Finset.range total_candies).card
      s.interpret (λ subset => s.query f subset) =
        {i | i ∈ Finset.range total_candies ∧ f i}) ∧
    (∀ (f : ℕ → Bool),
      (Finset.range total_candies).card ≤ max_queries) :=
sorry

end tasty_candy_identification_l3762_376259


namespace square_inequality_l3762_376213

theorem square_inequality (x y : ℝ) (h1 : x > y) (h2 : y > 2 / (x - y)) : x^2 > y^2 + 4 := by
  sorry

end square_inequality_l3762_376213


namespace y_percent_of_y_is_9_l3762_376275

theorem y_percent_of_y_is_9 (y : ℝ) (h1 : y > 0) (h2 : y / 100 * y = 9) : y = 30 := by
  sorry

end y_percent_of_y_is_9_l3762_376275


namespace total_pepper_pieces_l3762_376272

-- Define the number of bell peppers
def num_peppers : ℕ := 5

-- Define the number of large slices per pepper
def slices_per_pepper : ℕ := 20

-- Define the number of smaller pieces each half-slice is cut into
def smaller_pieces_per_slice : ℕ := 3

-- Theorem to prove
theorem total_pepper_pieces :
  let total_large_slices := num_peppers * slices_per_pepper
  let half_large_slices := total_large_slices / 2
  let smaller_pieces := half_large_slices * smaller_pieces_per_slice
  let remaining_large_slices := total_large_slices - half_large_slices
  remaining_large_slices + smaller_pieces = 200 := by
  sorry

end total_pepper_pieces_l3762_376272
