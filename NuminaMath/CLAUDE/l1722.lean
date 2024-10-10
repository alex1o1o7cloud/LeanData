import Mathlib

namespace benny_seashells_l1722_172218

/-- The number of seashells Benny has after giving some away -/
def remaining_seashells (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that Benny has 14 seashells after giving away 52 from his initial 66 -/
theorem benny_seashells : remaining_seashells 66 52 = 14 := by
  sorry

end benny_seashells_l1722_172218


namespace total_legs_is_43_l1722_172208

/-- Represents the number of legs for different types of passengers -/
structure LegCount where
  cat : Nat
  human : Nat
  oneLeggedCaptain : Nat

/-- Calculates the total number of legs given the number of heads and cats -/
def totalLegs (totalHeads : Nat) (catCount : Nat) (legCount : LegCount) : Nat :=
  let humanCount := totalHeads - catCount
  let regularHumanCount := humanCount - 1 -- Subtract the one-legged captain
  catCount * legCount.cat + regularHumanCount * legCount.human + legCount.oneLeggedCaptain

/-- Theorem stating that given the conditions, the total number of legs is 43 -/
theorem total_legs_is_43 (totalHeads : Nat) (catCount : Nat) (legCount : LegCount)
    (h1 : totalHeads = 15)
    (h2 : catCount = 7)
    (h3 : legCount.cat = 4)
    (h4 : legCount.human = 2)
    (h5 : legCount.oneLeggedCaptain = 1) :
    totalLegs totalHeads catCount legCount = 43 := by
  sorry


end total_legs_is_43_l1722_172208


namespace reflection_across_x_axis_l1722_172213

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The point to be reflected -/
def original_point : ℝ × ℝ := (-2, -3)

/-- The expected result after reflection -/
def expected_reflection : ℝ × ℝ := (-2, 3)

theorem reflection_across_x_axis :
  reflect_x original_point = expected_reflection := by sorry

end reflection_across_x_axis_l1722_172213


namespace not_or_implies_both_false_l1722_172252

theorem not_or_implies_both_false (p q : Prop) : 
  ¬(p ∨ q) → ¬p ∧ ¬q := by sorry

end not_or_implies_both_false_l1722_172252


namespace incorrect_log_values_l1722_172211

-- Define the logarithm function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the variables a, b, c
variable (a b c : ℝ)

-- Define the given correct logarithm values
axiom lg_2 : lg 2 = 1 - a - c
axiom lg_3 : lg 3 = 2*a - b
axiom lg_5 : lg 5 = a + c
axiom lg_9 : lg 9 = 4*a - 2*b

-- State the theorem
theorem incorrect_log_values :
  lg 1.5 ≠ 3*a - b + c ∧ lg 7 ≠ 2*(a + c) :=
sorry

end incorrect_log_values_l1722_172211


namespace systematic_sampling_l1722_172251

theorem systematic_sampling (total_students : Nat) (sample_size : Nat) (last_sample : Nat) 
  (h1 : total_students = 300)
  (h2 : sample_size = 60)
  (h3 : last_sample = 293) :
  ∃ (first_sample : Nat), first_sample = 3 ∧ 
  (first_sample + (sample_size - 1) * (total_students / sample_size) = last_sample) := by
  sorry

end systematic_sampling_l1722_172251


namespace complement_of_A_union_B_l1722_172231

def U : Set ℕ := {1, 2, 3, 4, 5}

def A : Set ℕ := {x ∈ U | x^2 - 3*x + 2 = 0}

def B : Set ℕ := {x ∈ U | ∃ a ∈ A, x = 2*a}

theorem complement_of_A_union_B (h : Set ℕ) : 
  h = U \ (A ∪ B) → h = {3, 5} := by
  sorry

end complement_of_A_union_B_l1722_172231


namespace distance_between_given_lines_l1722_172241

def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 6 = 0

def line2 (x y : ℝ) : Prop := 6 * x + 8 * y + 3 = 0

def are_parallel (l1 l2 : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (k : ℝ), k ≠ 0 ∧ ∀ (x y : ℝ), l1 x y ↔ l2 (k * x) (k * y)

def distance_between_lines (l1 l2 : (ℝ → ℝ → Prop)) : ℝ := sorry

theorem distance_between_given_lines :
  are_parallel line1 line2 →
  distance_between_lines line1 line2 = 1.5 := by sorry

end distance_between_given_lines_l1722_172241


namespace sum_zero_from_absolute_value_inequalities_l1722_172214

theorem sum_zero_from_absolute_value_inequalities (a b c : ℝ) 
  (h1 : |a| ≥ |b+c|) 
  (h2 : |b| ≥ |c+a|) 
  (h3 : |c| ≥ |a+b|) : 
  a + b + c = 0 := by 
sorry

end sum_zero_from_absolute_value_inequalities_l1722_172214


namespace quadratic_minimum_value_l1722_172290

/-- A quadratic function satisfying given conditions -/
def f (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

/-- The theorem stating the minimum value of the quadratic function -/
theorem quadratic_minimum_value
  (a b c : ℝ)
  (h1 : f a b c (-7) = -9)
  (h2 : f a b c (-5) = -4)
  (h3 : f a b c (-3) = -1)
  (h4 : f a b c (-1) = 0)
  (h5 : f a b c 1 = -1) :
  ∀ x ∈ Set.Icc (-7 : ℝ) 7, f a b c x ≥ -16 ∧ ∃ x₀ ∈ Set.Icc (-7 : ℝ) 7, f a b c x₀ = -16 :=
by sorry

end quadratic_minimum_value_l1722_172290


namespace symmetric_point_in_fourth_quadrant_l1722_172200

theorem symmetric_point_in_fourth_quadrant (a : ℝ) (P : ℝ × ℝ) :
  a < 0 →
  P = (-a^2 - 1, -a + 3) →
  (∃ P1 : ℝ × ℝ, P1 = (-P.1, -P.2) ∧ P1.1 > 0 ∧ P1.2 < 0) :=
by sorry

end symmetric_point_in_fourth_quadrant_l1722_172200


namespace change_calculation_l1722_172278

/-- Given an apple cost of $0.75 and a payment of $5, the change returned is $4.25. -/
theorem change_calculation (apple_cost payment : ℚ) (h1 : apple_cost = 0.75) (h2 : payment = 5) :
  payment - apple_cost = 4.25 := by
  sorry

end change_calculation_l1722_172278


namespace weight_of_doubled_cube_l1722_172227

/-- Given two cubes of the same material, if one cube has sides twice as long as the other,
    and the smaller cube weighs 4 pounds, then the larger cube weighs 32 pounds. -/
theorem weight_of_doubled_cube (s : ℝ) (weight : ℝ → ℝ) (volume : ℝ → ℝ) :
  (∀ x, weight x = (weight s / volume s) * volume x) →  -- weight is proportional to volume
  volume s = s^3 →  -- volume of a cube is side length cubed
  weight s = 4 →  -- weight of original cube is 4 pounds
  weight (2*s) = 32 :=  -- weight of new cube with doubled side length
by
  sorry


end weight_of_doubled_cube_l1722_172227


namespace least_subtraction_for_divisibility_l1722_172279

theorem least_subtraction_for_divisibility (n m : ℕ) (hn : n > 0) (hm : m > 0) : 
  ∃ k, k ≥ 0 ∧ k < m ∧ (n ^ 1000 - k) % m = 0 ∧ 
  ∀ j, 0 ≤ j ∧ j < k → (n ^ 1000 - j) % m ≠ 0 :=
by sorry

#check least_subtraction_for_divisibility 10 97

end least_subtraction_for_divisibility_l1722_172279


namespace beans_to_seitan_ratio_l1722_172288

/-- Represents the number of dishes with a specific protein combination -/
structure DishCount where
  total : ℕ
  beansAndLentils : ℕ
  beansAndSeitan : ℕ
  withLentils : ℕ
  onlyBeans : ℕ
  onlySeitan : ℕ

/-- The conditions of the problem -/
def restaurantMenu : DishCount where
  total := 10
  beansAndLentils := 2
  beansAndSeitan := 2
  withLentils := 4
  onlyBeans := 2
  onlySeitan := 2

/-- The theorem to prove -/
theorem beans_to_seitan_ratio (menu : DishCount) 
  (h1 : menu.total = 10)
  (h2 : menu.beansAndLentils = 2)
  (h3 : menu.beansAndSeitan = 2)
  (h4 : menu.withLentils = 4)
  (h5 : menu.onlyBeans + menu.onlySeitan = menu.total - menu.beansAndLentils - menu.beansAndSeitan - (menu.withLentils - menu.beansAndLentils))
  (h6 : menu.onlyBeans = menu.onlySeitan) :
  menu.onlyBeans = menu.onlySeitan := by
  sorry

end beans_to_seitan_ratio_l1722_172288


namespace color_tape_overlap_l1722_172238

theorem color_tape_overlap (total_length : ℝ) (tape_length : ℝ) (num_tapes : ℕ) 
  (h1 : total_length = 50.5)
  (h2 : tape_length = 18)
  (h3 : num_tapes = 3) :
  (num_tapes * tape_length - total_length) / 2 = 1.75 := by
  sorry

end color_tape_overlap_l1722_172238


namespace basketball_team_selection_l1722_172229

theorem basketball_team_selection (total_students : Nat) 
  (total_girls total_boys : Nat)
  (junior_girls senior_girls : Nat)
  (junior_boys senior_boys : Nat)
  (callback_junior_girls callback_senior_girls : Nat)
  (callback_junior_boys callback_senior_boys : Nat) :
  total_students = 56 →
  total_girls = 33 →
  total_boys = 23 →
  junior_girls = 15 →
  senior_girls = 18 →
  junior_boys = 12 →
  senior_boys = 11 →
  callback_junior_girls = 8 →
  callback_senior_girls = 9 →
  callback_junior_boys = 5 →
  callback_senior_boys = 6 →
  total_students - (callback_junior_girls + callback_senior_girls + callback_junior_boys + callback_senior_boys) = 28 := by
  sorry

#check basketball_team_selection

end basketball_team_selection_l1722_172229


namespace isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1722_172209

/-- An isosceles triangle with two sides of length 9 and one side of length 4 has a perimeter of 22 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∃ (a b : ℝ),
      a = 9 ∧ b = 4 ∧
      (a + a > b) ∧  -- Triangle inequality
      perimeter = a + a + b ∧
      perimeter = 22
      
#check isosceles_triangle_perimeter

/-- Proof of the theorem -/
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 22 := by
  sorry


end isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l1722_172209


namespace prime_composite_inequality_l1722_172262

theorem prime_composite_inequality (n : ℕ+) :
  (∀ (a : Fin n → ℕ+), (Function.Injective a) →
    ∃ (i j : Fin n), (a i + a j : ℚ) / (Nat.gcd (a i) (a j)) ≥ 2 * n - 1) ∨
  (∃ (a : Fin n → ℕ+), (Function.Injective a) ∧
    ∀ (i j : Fin n), (a i + a j : ℚ) / (Nat.gcd (a i) (a j)) < 2 * n - 1) :=
by sorry

end prime_composite_inequality_l1722_172262


namespace intercept_sum_l1722_172276

/-- The modulus of the congruence -/
def m : ℕ := 17

/-- The congruence relation -/
def congruence (x y : ℕ) : Prop :=
  (7 * x) % m = (3 * y + 2) % m

/-- The x-intercept of the congruence -/
def x_intercept : ℕ := 10

/-- The y-intercept of the congruence -/
def y_intercept : ℕ := 5

/-- Theorem stating that the sum of x and y intercepts is 15 -/
theorem intercept_sum :
  x_intercept + y_intercept = 15 ∧
  congruence x_intercept 0 ∧
  congruence 0 y_intercept ∧
  x_intercept < m ∧
  y_intercept < m :=
sorry

end intercept_sum_l1722_172276


namespace hannah_apple_pie_apples_l1722_172268

/-- Calculates the number of pounds of apples needed for Hannah's apple pie. -/
def apple_pie_apples (
  servings : ℕ)
  (cost_per_serving : ℚ)
  (apple_cost_per_pound : ℚ)
  (pie_crust_cost : ℚ)
  (lemon_cost : ℚ)
  (butter_cost : ℚ) : ℚ :=
  let total_cost := servings * cost_per_serving
  let apple_cost := total_cost - pie_crust_cost - lemon_cost - butter_cost
  apple_cost / apple_cost_per_pound

/-- Theorem stating that Hannah needs 2 pounds of apples for her pie. -/
theorem hannah_apple_pie_apples :
  apple_pie_apples 8 1 2 2 (1/2) (3/2) = 2 := by
  sorry

end hannah_apple_pie_apples_l1722_172268


namespace negation_equivalence_l1722_172277

theorem negation_equivalence :
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 - 5*x + 6 > 0)) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5*x + 6 ≤ 0) :=
by sorry

end negation_equivalence_l1722_172277


namespace geometric_sequence_property_l1722_172206

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence, if a₃ * a₉ = 4 * a₄, then a₈ = 4 -/
theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_cond : a 3 * a 9 = 4 * a 4) : 
  a 8 = 4 := by
sorry

end geometric_sequence_property_l1722_172206


namespace perfect_squares_l1722_172202

theorem perfect_squares (a b c : ℕ+) 
  (h_gcd : Nat.gcd a.val (Nat.gcd b.val c.val) = 1)
  (h_eq : a.val ^ 2 + b.val ^ 2 + c.val ^ 2 = 2 * (a.val * b.val + b.val * c.val + c.val * a.val)) :
  ∃ (x y z : ℕ), a.val = x ^ 2 ∧ b.val = y ^ 2 ∧ c.val = z ^ 2 := by
sorry

end perfect_squares_l1722_172202


namespace probability_at_least_one_girl_l1722_172235

theorem probability_at_least_one_girl (total : ℕ) (boys : ℕ) (girls : ℕ) (select : ℕ) :
  total = boys + girls →
  boys = 4 →
  girls = 2 →
  select = 3 →
  (1 - (Nat.choose boys select / Nat.choose total select : ℚ)) = 4/5 := by
sorry

end probability_at_least_one_girl_l1722_172235


namespace range_of_g_l1722_172246

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^2 - 2

-- Define the function g as the composition of f with itself
def g (x : ℝ) : ℝ := f (f x)

-- State the theorem about the range of g
theorem range_of_g :
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 1 ≤ g x ∧ g x ≤ 12 :=
sorry

end range_of_g_l1722_172246


namespace min_value_fraction_l1722_172281

theorem min_value_fraction (x : ℝ) (h : x > 7) :
  (x^2 + 49) / (x - 7) ≥ 7 + 14 * Real.sqrt 2 ∧
  ∃ y > 7, (y^2 + 49) / (y - 7) = 7 + 14 * Real.sqrt 2 :=
by sorry

end min_value_fraction_l1722_172281


namespace equation_equivalence_l1722_172269

theorem equation_equivalence : ∀ x : ℝ, 2 * (x + 1) = x + 7 ↔ x = 5 := by sorry

end equation_equivalence_l1722_172269


namespace sqrt_difference_equals_negative_two_tan_l1722_172259

theorem sqrt_difference_equals_negative_two_tan (α : Real) 
  (h : α ∈ Set.Ioo (-Real.pi) (-Real.pi/2)) : 
  Real.sqrt ((1 + Real.sin α) / (1 - Real.sin α)) - 
  Real.sqrt ((1 - Real.sin α) / (1 + Real.sin α)) = 
  -2 * Real.tan α :=
by sorry

end sqrt_difference_equals_negative_two_tan_l1722_172259


namespace meaningful_fraction_range_l1722_172296

theorem meaningful_fraction_range (x : ℝ) :
  (|x| - 6 ≠ 0) ↔ (x ≠ 6 ∧ x ≠ -6) := by
  sorry

end meaningful_fraction_range_l1722_172296


namespace binary_multiplication_l1722_172244

/-- Converts a list of bits to a natural number -/
def bitsToNat (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- The first binary number 11101₂ -/
def num1 : List Bool := [true, true, true, false, true]

/-- The second binary number 1101₂ -/
def num2 : List Bool := [true, true, false, true]

/-- The expected product 1001101101₂ -/
def expectedProduct : List Bool := [true, false, false, true, true, false, true, true, false, true]

theorem binary_multiplication :
  bitsToNat num1 * bitsToNat num2 = bitsToNat expectedProduct := by
  sorry

end binary_multiplication_l1722_172244


namespace isosceles_triangle_with_circles_perimeter_l1722_172204

/-- Represents a triangle with circles inside --/
structure TriangleWithCircles where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  circle_radius : ℝ

/-- Calculates the perimeter of a triangle with circles inside --/
def perimeter_with_circles (t : TriangleWithCircles) : ℝ :=
  t.side1 + t.side2 + t.base - 4 * t.circle_radius

/-- Theorem: The perimeter of the specified isosceles triangle with circles is 24 --/
theorem isosceles_triangle_with_circles_perimeter :
  let t : TriangleWithCircles := {
    side1 := 12,
    side2 := 12,
    base := 8,
    circle_radius := 2
  }
  perimeter_with_circles t = 24 := by
  sorry

end isosceles_triangle_with_circles_perimeter_l1722_172204


namespace no_positive_integer_solutions_l1722_172253

theorem no_positive_integer_solutions : 
  ¬ ∃ (x y : ℕ), x > 0 ∧ y > 0 ∧ x^4 * y^4 - 8 * x^2 * y^2 + 12 = 0 :=
sorry

end no_positive_integer_solutions_l1722_172253


namespace matrix_addition_a_l1722_172266

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; -1, 3]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![-1, 3; 1, -4]

theorem matrix_addition_a : A + B = !![1, 7; 0, -1] := by sorry

end matrix_addition_a_l1722_172266


namespace fraction_sum_times_two_l1722_172219

theorem fraction_sum_times_two : 
  (3 / 20 + 5 / 200 + 7 / 2000) * 2 = 0.357 := by sorry

end fraction_sum_times_two_l1722_172219


namespace amusement_park_problem_l1722_172254

/-- Proves that given the conditions of the amusement park problem, 
    the number of parents is 10 and the number of students is 5 -/
theorem amusement_park_problem 
  (total_people : ℕ)
  (adult_ticket_price : ℕ)
  (student_discount : ℚ)
  (total_spent : ℕ)
  (h1 : total_people = 15)
  (h2 : adult_ticket_price = 50)
  (h3 : student_discount = 0.6)
  (h4 : total_spent = 650) :
  ∃ (parents students : ℕ),
    parents + students = total_people ∧
    parents * adult_ticket_price + 
    students * (adult_ticket_price * (1 - student_discount)) = total_spent ∧
    parents = 10 ∧
    students = 5 :=
by sorry

end amusement_park_problem_l1722_172254


namespace unique_solution_condition_l1722_172286

/-- The logarithm function to base 10 -/
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

/-- The equation has exactly one solution iff a = 4 or a < 0 -/
theorem unique_solution_condition (a : ℝ) :
  (∃! x : ℝ, log10 (a * x) = 2 * log10 (x + 1) ∧ a * x > 0 ∧ x + 1 > 0) ↔ 
  (a = 4 ∨ a < 0) :=
sorry

end unique_solution_condition_l1722_172286


namespace simplify_expression_1_simplify_expression_2_calculate_expression_3_calculate_expression_4_l1722_172280

-- Part 1
theorem simplify_expression_1 (x : ℝ) : 2 * x^2 + 3 * x - 3 * x^2 + 4 * x = -x^2 + 7 * x := by sorry

-- Part 2
theorem simplify_expression_2 (a : ℝ) : 3 * a - 5 * (a + 1) + 4 * (2 + a) = 2 * a + 3 := by sorry

-- Part 3
theorem calculate_expression_3 : (-2/3 + 5/8 - 1/6) * (-24) = 5 := by sorry

-- Part 4
theorem calculate_expression_4 : -(1^4) + 16 / ((-2)^3) * |(-3) - 1| = -9 := by sorry

end simplify_expression_1_simplify_expression_2_calculate_expression_3_calculate_expression_4_l1722_172280


namespace remaining_laps_is_58_l1722_172201

/-- Represents the number of laps swum on each day --/
structure DailyLaps where
  friday : Nat
  saturday : Nat
  sundayMorning : Nat

/-- Calculates the remaining laps after Sunday morning --/
def remainingLaps (totalRequired : Nat) (daily : DailyLaps) : Nat :=
  totalRequired - (daily.friday + daily.saturday + daily.sundayMorning)

/-- Theorem stating that the remaining laps after Sunday morning is 58 --/
theorem remaining_laps_is_58 (totalRequired : Nat) (daily : DailyLaps) :
  totalRequired = 198 →
  daily.friday = 63 →
  daily.saturday = 62 →
  daily.sundayMorning = 15 →
  remainingLaps totalRequired daily = 58 := by
  sorry

#eval remainingLaps 198 { friday := 63, saturday := 62, sundayMorning := 15 }

end remaining_laps_is_58_l1722_172201


namespace cost_of_candies_in_dollars_l1722_172271

-- Define the cost of one piece of candy in cents
def cost_per_candy : ℕ := 2

-- Define the number of pieces of candy
def number_of_candies : ℕ := 500

-- Define the conversion rate from cents to dollars
def cents_per_dollar : ℕ := 100

-- Theorem to prove
theorem cost_of_candies_in_dollars :
  (number_of_candies * cost_per_candy) / cents_per_dollar = 10 := by
  sorry

end cost_of_candies_in_dollars_l1722_172271


namespace sum_of_squares_l1722_172263

theorem sum_of_squares (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 4)
  (h2 : a/x + b/y + c/z = 0) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 16 := by
sorry

end sum_of_squares_l1722_172263


namespace triangle_properties_l1722_172261

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  -- a and b are roots of x^2 - 2√3x + 2 = 0
  t.a^2 - 2 * Real.sqrt 3 * t.a + 2 = 0 ∧
  t.b^2 - 2 * Real.sqrt 3 * t.b + 2 = 0 ∧
  -- 2cos(A+B) = 1
  2 * Real.cos (t.A + t.B) = 1

-- State the theorem
theorem triangle_properties (t : Triangle) (h : is_valid_triangle t) :
  t.C = 2 * π / 3 ∧  -- 120° in radians
  t.c = Real.sqrt 10 ∧
  (1 / 2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2 := by
  sorry

end triangle_properties_l1722_172261


namespace max_parrots_in_zoo_l1722_172292

/-- Represents the number of parrots of each color in the zoo -/
structure ParrotCount where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- The conditions of the zoo problem -/
def ZooConditions (p : ParrotCount) : Prop :=
  p.red > 0 ∧ p.yellow > 0 ∧ p.green > 0 ∧
  ∀ (s : Finset ℕ), s.card = 10 → (∃ i ∈ s, i < p.red) ∧
  ∀ (s : Finset ℕ), s.card = 12 → (∃ i ∈ s, i < p.yellow)

/-- The theorem stating the maximum number of parrots in the zoo -/
theorem max_parrots_in_zoo :
  ∃ (max : ℕ), max = 19 ∧
  (∃ (p : ParrotCount), ZooConditions p ∧ p.red + p.yellow + p.green = max) ∧
  ∀ (p : ParrotCount), ZooConditions p → p.red + p.yellow + p.green ≤ max :=
sorry

end max_parrots_in_zoo_l1722_172292


namespace tangent_line_parallel_l1722_172275

/-- Given a function f(x) = x^3 - ax^2 + x, prove that if its tangent line at x=1 
    is parallel to y=2x, then a = 1. -/
theorem tangent_line_parallel (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 - a*x^2 + x
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 - 2*a*x + 1
  (f' 1 = 2) → a = 1 := by
sorry

end tangent_line_parallel_l1722_172275


namespace roberta_shopping_l1722_172285

def shopping_trip (initial_amount bag_price_difference : ℕ) : Prop :=
  let shoe_price := 45
  let bag_price := shoe_price - bag_price_difference
  let lunch_price := bag_price / 4
  let total_expenses := shoe_price + bag_price + lunch_price
  let money_left := initial_amount - total_expenses
  money_left = 78

theorem roberta_shopping :
  shopping_trip 158 17 := by
  sorry

end roberta_shopping_l1722_172285


namespace tan_3_degrees_decomposition_l1722_172223

theorem tan_3_degrees_decomposition :
  ∃ (p q r s : ℕ+),
    (Real.tan (3 * Real.pi / 180) = Real.sqrt p - Real.sqrt q + Real.sqrt r - s) ∧
    (p ≥ q) ∧ (q ≥ r) ∧ (r ≥ s) →
    p + q + r + s = 20 := by
  sorry

end tan_3_degrees_decomposition_l1722_172223


namespace second_purchase_profit_less_than_first_l1722_172260

/-- Represents a type of T-shirt -/
structure TShirt where
  purchasePrice : ℕ
  sellingPrice : ℕ

/-- Represents the store's inventory and sales -/
structure Store where
  typeA : TShirt
  typeB : TShirt
  firstPurchaseQuantityA : ℕ
  firstPurchaseQuantityB : ℕ
  secondPurchaseQuantityA : ℕ
  secondPurchaseQuantityB : ℕ

/-- Calculate the profit from the first purchase -/
def firstPurchaseProfit (s : Store) : ℕ :=
  (s.typeA.sellingPrice - s.typeA.purchasePrice) * s.firstPurchaseQuantityA +
  (s.typeB.sellingPrice - s.typeB.purchasePrice) * s.firstPurchaseQuantityB

/-- Calculate the maximum profit from the second purchase -/
def maxSecondPurchaseProfit (s : Store) : ℕ :=
  let newTypeA := TShirt.mk (s.typeA.purchasePrice + 5) s.typeA.sellingPrice
  let newTypeB := TShirt.mk (s.typeB.purchasePrice + 10) s.typeB.sellingPrice
  (newTypeA.sellingPrice - newTypeA.purchasePrice) * s.secondPurchaseQuantityA +
  (newTypeB.sellingPrice - newTypeB.purchasePrice) * s.secondPurchaseQuantityB

/-- The theorem to be proved -/
theorem second_purchase_profit_less_than_first (s : Store) :
  s.firstPurchaseQuantityA + s.firstPurchaseQuantityB = 120 →
  s.typeA.purchasePrice * s.firstPurchaseQuantityA + s.typeB.purchasePrice * s.firstPurchaseQuantityB = 6000 →
  s.secondPurchaseQuantityA + s.secondPurchaseQuantityB = 150 →
  s.secondPurchaseQuantityB ≤ 2 * s.secondPurchaseQuantityA →
  maxSecondPurchaseProfit s < firstPurchaseProfit s :=
by
  sorry

#check second_purchase_profit_less_than_first

end second_purchase_profit_less_than_first_l1722_172260


namespace class_overlap_difference_l1722_172239

theorem class_overlap_difference (total students_geometry students_biology : ℕ) 
  (h_total : total = 232)
  (h_geometry : students_geometry = 144)
  (h_biology : students_biology = 119) :
  let max_overlap := min students_geometry students_biology
  let min_overlap := students_geometry + students_biology - total
  max_overlap - min_overlap = 88 := by
sorry

end class_overlap_difference_l1722_172239


namespace union_when_k_neg_one_intersection_equality_iff_k_range_l1722_172289

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2 - k}

-- Theorem for part I
theorem union_when_k_neg_one :
  A ∪ B (-1) = {x : ℝ | -1 < x ∧ x < 3} := by sorry

-- Theorem for part II
theorem intersection_equality_iff_k_range :
  ∀ k : ℝ, A ∩ B k = B k ↔ k ∈ Set.Ici 0 := by sorry

end union_when_k_neg_one_intersection_equality_iff_k_range_l1722_172289


namespace largest_multiple_of_9_less_than_100_l1722_172216

theorem largest_multiple_of_9_less_than_100 : ∃ (n : ℕ), n = 99 ∧ 9 ∣ n ∧ n < 100 ∧ ∀ (m : ℕ), 9 ∣ m → m < 100 → m ≤ n :=
by sorry

end largest_multiple_of_9_less_than_100_l1722_172216


namespace three_W_seven_l1722_172274

-- Define the W operation
def W (a b : ℝ) : ℝ := b + 5 * a - 3 * a^2

-- Theorem to prove
theorem three_W_seven : W 3 7 = -5 := by
  sorry

end three_W_seven_l1722_172274


namespace dog_hare_speed_ratio_l1722_172220

/-- The ratio of dog's speed to hare's speed given their leap patterns -/
theorem dog_hare_speed_ratio :
  ∀ (dog_leaps hare_leaps : ℕ) (dog_distance hare_distance : ℝ),
  dog_leaps = 10 →
  hare_leaps = 2 →
  dog_distance = 2 * hare_distance →
  (dog_leaps * dog_distance) / (hare_leaps * hare_distance) = 10 :=
by
  sorry

end dog_hare_speed_ratio_l1722_172220


namespace quadratic_roots_property_l1722_172228

theorem quadratic_roots_property (d e : ℝ) : 
  (5 * d^2 - 4 * d - 1 = 0) → 
  (5 * e^2 - 4 * e - 1 = 0) → 
  (d - 2) * (e - 2) = 11/5 := by
sorry

end quadratic_roots_property_l1722_172228


namespace age_difference_l1722_172245

/-- Given four people A, B, C, and D with ages a, b, c, and d respectively,
    prove that C is 14 years younger than A under the given conditions. -/
theorem age_difference (a b c d : ℕ) : 
  (a + b = b + c + 14) →
  (b + d = c + a + 10) →
  (d = c + 6) →
  (a = c + 14) := by sorry

end age_difference_l1722_172245


namespace sqrt_equation_condition_l1722_172270

theorem sqrt_equation_condition (a : ℝ) : 
  Real.sqrt (a^2 - 4*a + 4) = 2 - a ↔ a ≤ 2 := by sorry

end sqrt_equation_condition_l1722_172270


namespace flower_shop_problem_l1722_172205

/-- Flower shop problem -/
theorem flower_shop_problem (roses_per_bouquet : ℕ) 
  (total_bouquets : ℕ) (rose_bouquets : ℕ) (daisy_bouquets : ℕ) 
  (total_flowers : ℕ) : 
  roses_per_bouquet = 12 →
  total_bouquets = 20 →
  rose_bouquets = 10 →
  daisy_bouquets = 10 →
  rose_bouquets + daisy_bouquets = total_bouquets →
  total_flowers = 190 →
  (total_flowers - roses_per_bouquet * rose_bouquets) / daisy_bouquets = 7 :=
by sorry

end flower_shop_problem_l1722_172205


namespace quadratic_roots_range_l1722_172225

theorem quadratic_roots_range (θ : Real) (α β : Complex) : 
  (∃ x : Complex, x^2 + 2*(Real.cos θ + 1)*x + (Real.cos θ)^2 = 0 ↔ x = α ∨ x = β) →
  Complex.abs (α - β) ≤ 2 * Real.sqrt 2 →
  ∃ k : ℤ, (θ ∈ Set.Icc (2*k*Real.pi + Real.pi/3) (2*k*Real.pi + 2*Real.pi/3)) ∨
           (θ ∈ Set.Icc (2*k*Real.pi + 4*Real.pi/3) (2*k*Real.pi + 5*Real.pi/3)) :=
by sorry

end quadratic_roots_range_l1722_172225


namespace min_cubes_for_box_l1722_172249

/-- The minimum number of cubes required to build a box -/
def min_cubes (length width height cube_volume : ℕ) : ℕ :=
  (length * width * height + cube_volume - 1) / cube_volume

/-- Theorem: The minimum number of 10 cubic cm cubes required to build a box
    with dimensions 8 cm x 15 cm x 5 cm is 60 -/
theorem min_cubes_for_box : min_cubes 8 15 5 10 = 60 := by
  sorry

end min_cubes_for_box_l1722_172249


namespace not_false_is_true_l1722_172224

theorem not_false_is_true (p q : Prop) (hp : p) (hq : ¬q) : ¬q := by
  sorry

end not_false_is_true_l1722_172224


namespace function_form_from_inequality_l1722_172283

/-- A function satisfying the given inequality property. -/
def SatisfiesInequality (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, |f (x + y) - f (x - y) - y| ≤ y^2

/-- The main theorem stating that a function satisfying the inequality
    must be of the form f(x) = x/2 + c for some constant c. -/
theorem function_form_from_inequality (f : ℝ → ℝ) 
    (h : SatisfiesInequality f) : 
    ∃ c : ℝ, ∀ x : ℝ, f x = x / 2 + c := by
  sorry

end function_form_from_inequality_l1722_172283


namespace concert_ticket_sales_l1722_172240

/-- Proves that the total number of tickets sold is 45 given the specified conditions --/
theorem concert_ticket_sales : 
  let ticket_price : ℕ := 20
  let first_group_size : ℕ := 10
  let second_group_size : ℕ := 20
  let first_group_discount : ℚ := 40 / 100
  let second_group_discount : ℚ := 15 / 100
  let total_revenue : ℕ := 760
  ∃ (full_price_tickets : ℕ),
    (first_group_size * (ticket_price * (1 - first_group_discount)).floor + 
     second_group_size * (ticket_price * (1 - second_group_discount)).floor + 
     full_price_tickets * ticket_price = total_revenue) ∧
    (first_group_size + second_group_size + full_price_tickets = 45) :=
by sorry

end concert_ticket_sales_l1722_172240


namespace derivative_sum_at_one_l1722_172247

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the theorem
theorem derivative_sum_at_one 
  (h1 : ∀ x, f x + x * g x = x^2 - 1) 
  (h2 : f 1 = 1) : 
  deriv f 1 + deriv g 1 = 3 := by
sorry

end derivative_sum_at_one_l1722_172247


namespace arithmetic_geometric_ratio_l1722_172297

/-- An arithmetic sequence with a non-zero common difference -/
def ArithmeticSequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  d ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n + d

/-- Three terms form a geometric sequence -/
def GeometricSequence (x y z : ℝ) : Prop :=
  y^2 = x * z

theorem arithmetic_geometric_ratio
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ArithmeticSequence a d)
  (h_geom : GeometricSequence (a 5) (a 9) (a 15)) :
  a 15 / a 9 = 3/2 := by
sorry

end arithmetic_geometric_ratio_l1722_172297


namespace extra_chairs_added_l1722_172287

/-- The number of extra chairs added to a wedding seating arrangement -/
theorem extra_chairs_added (rows : ℕ) (chairs_per_row : ℕ) (total_chairs : ℕ) : 
  rows = 7 → chairs_per_row = 12 → total_chairs = 95 → 
  total_chairs - (rows * chairs_per_row) = 11 := by
  sorry

end extra_chairs_added_l1722_172287


namespace milk_container_problem_l1722_172299

theorem milk_container_problem (capacity_A : ℝ) : 
  (capacity_A > 0) →
  (0.375 * capacity_A + 156 = 0.625 * capacity_A - 156) →
  capacity_A = 1248 := by
sorry

end milk_container_problem_l1722_172299


namespace gift_cost_l1722_172264

theorem gift_cost (dave_money : ℕ) (kyle_initial : ℕ) (kyle_spent : ℕ) (kyle_remaining : ℕ) (lisa_money : ℕ) (gift_cost : ℕ) : 
  dave_money = 46 →
  kyle_initial = 3 * dave_money - 12 →
  kyle_spent = kyle_initial / 3 →
  kyle_remaining = kyle_initial - kyle_spent →
  lisa_money = kyle_remaining + 20 →
  gift_cost = (kyle_remaining + lisa_money) / 2 →
  gift_cost = 94 := by
sorry

end gift_cost_l1722_172264


namespace abc_inequality_l1722_172215

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define a, b, and c
noncomputable def a : ℝ := log (3/4) (log 3 4)
noncomputable def b : ℝ := (3/4) ^ (1/2 : ℝ)
noncomputable def c : ℝ := (4/3) ^ (1/2 : ℝ)

-- Theorem statement
theorem abc_inequality : a < b ∧ b < c := by sorry

end abc_inequality_l1722_172215


namespace isabel_earnings_l1722_172248

/-- The number of bead necklaces sold -/
def bead_necklaces : ℕ := 3

/-- The number of gem stone necklaces sold -/
def gem_necklaces : ℕ := 3

/-- The cost of each necklace in dollars -/
def necklace_cost : ℕ := 6

/-- The total number of necklaces sold -/
def total_necklaces : ℕ := bead_necklaces + gem_necklaces

/-- The total money earned in dollars -/
def total_earned : ℕ := total_necklaces * necklace_cost

theorem isabel_earnings : total_earned = 36 := by
  sorry

end isabel_earnings_l1722_172248


namespace lcm_gcd_sum_problem_l1722_172265

theorem lcm_gcd_sum_problem (a b : ℕ) (ha : a = 12) (hb : b = 20) :
  (Nat.lcm a b * Nat.gcd a b) + (a + b) = 272 := by
  sorry

end lcm_gcd_sum_problem_l1722_172265


namespace rudy_running_time_l1722_172233

-- Define the running segments
def segment1_distance : ℝ := 5
def segment1_rate : ℝ := 10
def segment2_distance : ℝ := 4
def segment2_rate : ℝ := 9.5
def segment3_distance : ℝ := 3
def segment3_rate : ℝ := 8.5
def segment4_distance : ℝ := 2
def segment4_rate : ℝ := 12

-- Define the rest times
def rest1 : ℝ := 15
def rest2 : ℝ := 10
def rest3 : ℝ := 5

-- Define the total time function
def total_time : ℝ :=
  segment1_distance * segment1_rate +
  segment2_distance * segment2_rate +
  segment3_distance * segment3_rate +
  segment4_distance * segment4_rate +
  rest1 + rest2 + rest3

-- Theorem statement
theorem rudy_running_time : total_time = 167.5 := by
  sorry

end rudy_running_time_l1722_172233


namespace petyas_calculation_error_l1722_172255

theorem petyas_calculation_error :
  ¬∃ (a : ℕ), a > 3 ∧ 
  ∃ (n : ℕ), ((a - 3) * (a + 4) - a = n) ∧ 
  (∃ (digits : List ℕ), 
    digits.length = 6069 ∧
    digits.count 8 = 2023 ∧
    digits.count 0 = 2023 ∧
    digits.count 3 = 2023 ∧
    (∀ d, d ∈ digits → d ∈ [8, 0, 3]) ∧
    n = digits.foldl (λ acc d => acc * 10 + d) 0) :=
sorry

end petyas_calculation_error_l1722_172255


namespace four_jumps_reduction_l1722_172226

def jump_reduction (initial : ℕ) (jumps : ℕ) (reduction : ℕ) : ℕ :=
  initial - jumps * reduction

theorem four_jumps_reduction : jump_reduction 320 4 10 = 280 := by
  sorry

end four_jumps_reduction_l1722_172226


namespace yuna_has_greatest_sum_l1722_172203

/-- Yoojung's first number -/
def yoojung_num1 : ℕ := 5

/-- Yoojung's second number -/
def yoojung_num2 : ℕ := 8

/-- Yuna's first number -/
def yuna_num1 : ℕ := 7

/-- Yuna's second number -/
def yuna_num2 : ℕ := 9

/-- The sum of Yoojung's numbers -/
def yoojung_sum : ℕ := yoojung_num1 + yoojung_num2

/-- The sum of Yuna's numbers -/
def yuna_sum : ℕ := yuna_num1 + yuna_num2

theorem yuna_has_greatest_sum : yuna_sum > yoojung_sum := by
  sorry

end yuna_has_greatest_sum_l1722_172203


namespace west_asian_percentage_approx_46_percent_l1722_172217

/-- Represents the Asian population (in millions) for each region of the U.S. -/
structure AsianPopulation where
  ne : ℕ
  mw : ℕ
  south : ℕ
  west : ℕ

/-- Calculates the percentage of Asian population living in the West -/
def westAsianPercentage (pop : AsianPopulation) : ℚ :=
  (pop.west : ℚ) / (pop.ne + pop.mw + pop.south + pop.west)

/-- The given Asian population data for 1990 -/
def population1990 : AsianPopulation :=
  { ne := 2, mw := 3, south := 2, west := 6 }

theorem west_asian_percentage_approx_46_percent :
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 : ℚ) / 100 ∧ 
  |westAsianPercentage population1990 - (46 : ℚ) / 100| < ε :=
sorry

end west_asian_percentage_approx_46_percent_l1722_172217


namespace inequality_proof_l1722_172236

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_eq_two : a + b + c = 2) :
  (1 / (1 + a*b)) + (1 / (1 + b*c)) + (1 / (1 + c*a)) ≥ 27/13 ∧
  ((1 / (1 + a*b)) + (1 / (1 + b*c)) + (1 / (1 + c*a)) = 27/13 ↔ a = 2/3 ∧ b = 2/3 ∧ c = 2/3) :=
by sorry

end inequality_proof_l1722_172236


namespace revenue_decrease_l1722_172234

theorem revenue_decrease (T C : ℝ) (h1 : T > 0) (h2 : C > 0) :
  let new_tax := 0.8 * T
  let new_consumption := 1.1 * C
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  (original_revenue - new_revenue) / original_revenue = 0.12 :=
by sorry

end revenue_decrease_l1722_172234


namespace simplify_fraction_l1722_172284

theorem simplify_fraction : (120 : ℚ) / 180 = 2 / 3 := by
  sorry

end simplify_fraction_l1722_172284


namespace original_number_l1722_172221

theorem original_number (x : ℝ) : x * 1.4 = 1680 ↔ x = 1200 := by sorry

end original_number_l1722_172221


namespace sum_and_reciprocal_sum_l1722_172267

theorem sum_and_reciprocal_sum (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_sum : a + b = 6 * x) (h_reciprocal_sum : 1 / a + 1 / b = 6) : x = a * b :=
by sorry

end sum_and_reciprocal_sum_l1722_172267


namespace expand_product_l1722_172258

theorem expand_product (x : ℝ) : (x + 5) * (x - 4^2) = x^2 - 11*x - 80 := by
  sorry

end expand_product_l1722_172258


namespace min_dividend_with_quotient_and_remainder_six_l1722_172212

theorem min_dividend_with_quotient_and_remainder_six (dividend : ℕ) (divisor : ℕ) : 
  dividend ≥ 48 → 
  dividend / divisor = 6 → 
  dividend % divisor = 6 → 
  dividend ≥ 48 :=
by
  sorry

end min_dividend_with_quotient_and_remainder_six_l1722_172212


namespace square_difference_divided_by_eleven_l1722_172230

theorem square_difference_divided_by_eleven : (121^2 - 110^2) / 11 = 231 := by
  sorry

end square_difference_divided_by_eleven_l1722_172230


namespace max_robot_weight_is_270_l1722_172242

/-- Represents the weight constraints and components of a robot in the competition. -/
structure RobotWeightConstraints where
  standard_robot_weight : ℝ
  battery_weight : ℝ
  min_payload_weight : ℝ
  max_payload_weight : ℝ
  min_robot_weight_diff : ℝ

/-- Calculates the maximum weight of a robot in the competition. -/
def max_robot_weight (constraints : RobotWeightConstraints) : ℝ :=
  let min_robot_weight := constraints.standard_robot_weight + constraints.min_robot_weight_diff
  let min_total_weight := min_robot_weight + constraints.battery_weight + constraints.min_payload_weight
  2 * min_total_weight

/-- Theorem stating the maximum weight of a robot in the competition. -/
theorem max_robot_weight_is_270 (constraints : RobotWeightConstraints) 
    (h1 : constraints.standard_robot_weight = 100)
    (h2 : constraints.battery_weight = 20)
    (h3 : constraints.min_payload_weight = 10)
    (h4 : constraints.max_payload_weight = 25)
    (h5 : constraints.min_robot_weight_diff = 5) :
  max_robot_weight constraints = 270 := by
  sorry

#eval max_robot_weight { 
  standard_robot_weight := 100,
  battery_weight := 20,
  min_payload_weight := 10,
  max_payload_weight := 25,
  min_robot_weight_diff := 5
}

end max_robot_weight_is_270_l1722_172242


namespace ellipse_equation_l1722_172257

/-- Given an ellipse and a line passing through its upper vertex and right focus,
    prove that the equation of the ellipse is x^2/5 + y^2/4 = 1. -/
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ c : ℝ, c > 0 ∧ c < a ∧ a^2 = b^2 + c^2 ∧
   2*0 + b - 2 = 0 ∧ 2*c + 0 - 2 = 0) →
  ∀ x y : ℝ, x^2/a^2 + y^2/b^2 = 1 ↔ x^2/5 + y^2/4 = 1 :=
by sorry


end ellipse_equation_l1722_172257


namespace min_sum_m_n_l1722_172293

theorem min_sum_m_n (m n : ℕ+) (h1 : 45 * m = n ^ 3) (h2 : ∃ k : ℕ+, n = 5 * k) :
  (∀ m' n' : ℕ+, 45 * m' = n' ^ 3 → (∃ k' : ℕ+, n' = 5 * k') → m + n ≤ m' + n') →
  m + n = 90 := by
sorry

end min_sum_m_n_l1722_172293


namespace chess_tournament_games_l1722_172232

def tournament_games (n : ℕ) : ℕ := n * (n - 1)

theorem chess_tournament_games :
  tournament_games 17 * 2 = 544 := by
  sorry

end chess_tournament_games_l1722_172232


namespace min_value_x_plus_y_l1722_172222

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 1 → x + y ≤ a + b :=
by sorry

end min_value_x_plus_y_l1722_172222


namespace cubic_function_symmetry_l1722_172256

theorem cubic_function_symmetry (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 - b * x + 1
  f (-2) = 1 → f 2 = 1 := by
  sorry

end cubic_function_symmetry_l1722_172256


namespace hammer_wrench_ratio_l1722_172250

/-- Given that the weight of a wrench is twice the weight of a hammer,
    prove that the ratio of (weight of 2 hammers + 2 wrenches) to
    (weight of 8 hammers + 5 wrenches) is 1/3. -/
theorem hammer_wrench_ratio (h w : ℝ) (hw : w = 2 * h) :
  (2 * h + 2 * w) / (8 * h + 5 * w) = 1 / 3 := by
  sorry

end hammer_wrench_ratio_l1722_172250


namespace continuous_additive_function_is_linear_l1722_172295

theorem continuous_additive_function_is_linear 
  (f : ℝ → ℝ) 
  (hf_continuous : Continuous f) 
  (hf_additive : ∀ x y : ℝ, f (x + y) = f x + f y) : 
  ∃ a : ℝ, ∀ x : ℝ, f x = a * x :=
by sorry

end continuous_additive_function_is_linear_l1722_172295


namespace locus_and_tangent_l1722_172210

-- Define the points and lines
def A : ℝ × ℝ := (1, 0)
def B : ℝ → ℝ × ℝ := λ y ↦ (-1, y)
def l : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = -1}

-- Define the locus E
def E : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define point P
def P : ℝ × ℝ := (1, 2)

-- Define the tangent line
def tangent_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 - p.2 + 1 = 0}

theorem locus_and_tangent :
  (∀ y : ℝ, ∃ M : ℝ × ℝ, 
    (M.1 - A.1)^2 + (M.2 - A.2)^2 = (M.1 - (B y).1)^2 + (M.2 - (B y).2)^2 ∧
    M ∈ E) ∧
  (P ∈ E ∧ tangent_line ∩ E = {P}) := by sorry

end locus_and_tangent_l1722_172210


namespace allocation_theorem_l1722_172294

/-- The number of ways to allocate doctors and nurses to schools -/
def allocation_methods (num_doctors num_nurses num_schools : ℕ) : ℕ :=
  num_doctors * (num_nurses.choose (num_nurses / num_schools))

/-- Theorem: There are 12 ways to allocate 2 doctors and 4 nurses to 2 schools -/
theorem allocation_theorem :
  allocation_methods 2 4 2 = 12 := by
  sorry

#eval allocation_methods 2 4 2

end allocation_theorem_l1722_172294


namespace manufacturing_department_percentage_l1722_172272

theorem manufacturing_department_percentage (total_degrees : ℝ) (manufacturing_degrees : ℝ) 
  (h1 : total_degrees = 360) 
  (h2 : manufacturing_degrees = 162) : 
  (manufacturing_degrees / total_degrees) * 100 = 45 := by
  sorry

end manufacturing_department_percentage_l1722_172272


namespace bread_making_time_is_375_l1722_172282

/-- Represents the duration of each step in Mark's bread-making process -/
def bread_making_steps : List ℕ := [30, 120, 20, 120, 10, 30, 30, 15]

/-- The total time Mark spends making bread -/
def total_bread_making_time : ℕ := bread_making_steps.sum

/-- Theorem stating that the total time Mark spends making bread is 375 minutes -/
theorem bread_making_time_is_375 : total_bread_making_time = 375 := by
  sorry

#eval total_bread_making_time

end bread_making_time_is_375_l1722_172282


namespace common_tangent_implies_a_b_equal_three_l1722_172243

/-- Given two functions f and g with a common tangent at (1, c), prove a = b = 3 -/
theorem common_tangent_implies_a_b_equal_three
  (f : ℝ → ℝ)
  (g : ℝ → ℝ)
  (a b : ℝ)
  (h_f : ∀ x, f x = a * x^2 + 1)
  (h_a_pos : a > 0)
  (h_g : ∀ x, g x = x^3 + b * x)
  (h_intersection : f 1 = g 1)
  (h_common_tangent : (deriv f) 1 = (deriv g) 1) :
  a = 3 ∧ b = 3 := by
  sorry

end common_tangent_implies_a_b_equal_three_l1722_172243


namespace abs_diff_two_equiv_interval_l1722_172298

theorem abs_diff_two_equiv_interval (x : ℝ) : (1 < x ∧ x < 3) ↔ |x - 2| < 1 := by sorry

end abs_diff_two_equiv_interval_l1722_172298


namespace min_value_reciprocal_sum_l1722_172273

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 5) :
  (1 / (x + 2) + 1 / (y + 2)) ≥ 4 / 9 ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x + y = 5 ∧ 1 / (x + 2) + 1 / (y + 2) = 4 / 9) :=
by sorry

end min_value_reciprocal_sum_l1722_172273


namespace intersection_equiv_open_interval_l1722_172237

def set_A : Set ℝ := {x | x / (x - 1) ≤ 0}
def set_B : Set ℝ := {x | x^2 < 2*x}

theorem intersection_equiv_open_interval : 
  ∀ x : ℝ, x ∈ set_A ∩ set_B ↔ x ∈ Set.Ioo 0 1 := by sorry

end intersection_equiv_open_interval_l1722_172237


namespace total_distance_flown_l1722_172291

theorem total_distance_flown (trip_distance : ℝ) (num_trips : ℝ) 
  (h1 : trip_distance = 256.0) 
  (h2 : num_trips = 32.0) : 
  trip_distance * num_trips = 8192.0 := by
  sorry

end total_distance_flown_l1722_172291


namespace percentage_problem_l1722_172207

theorem percentage_problem (P : ℝ) : (P / 100) * 600 = (50 / 100) * 720 → P = 60 := by
  sorry

end percentage_problem_l1722_172207
