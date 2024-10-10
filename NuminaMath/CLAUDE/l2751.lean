import Mathlib

namespace f_uniquely_determined_l2751_275114

/-- A function from ℝ² to ℝ² defined as f(x, y) = (kx, y + b) -/
def f (k b : ℝ) : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (k * x, y + b)

/-- Theorem: If f(3, 1) = (6, 2), then k = 2 and b = 1 -/
theorem f_uniquely_determined (k b : ℝ) : 
  f k b (3, 1) = (6, 2) → k = 2 ∧ b = 1 := by
  sorry

end f_uniquely_determined_l2751_275114


namespace number_equation_solution_l2751_275191

theorem number_equation_solution : 
  ∃ n : ℚ, (3/4 : ℚ) * n - (8/5 : ℚ) * n + 63 = 12 ∧ n = 60 := by
  sorry

end number_equation_solution_l2751_275191


namespace zeros_not_adjacent_probability_probability_calculation_main_theorem_l2751_275115

/-- The probability of 2 zeros not being adjacent when 4 ones and 2 zeros are randomly arranged in a row -/
theorem zeros_not_adjacent_probability : ℚ :=
  2/3

/-- The total number of ways to arrange 4 ones and 2 zeros in a row -/
def total_arrangements : ℕ :=
  Nat.choose 6 2

/-- The number of arrangements where the 2 zeros are not adjacent -/
def non_adjacent_arrangements : ℕ :=
  Nat.choose 5 2

/-- The probability is the ratio of non-adjacent arrangements to total arrangements -/
theorem probability_calculation (h : zeros_not_adjacent_probability = non_adjacent_arrangements / total_arrangements) :
  zeros_not_adjacent_probability = 2/3 := by
  sorry

/-- The main theorem stating that the probability of 2 zeros not being adjacent is 2/3 -/
theorem main_theorem :
  zeros_not_adjacent_probability = 2/3 := by
  sorry

end zeros_not_adjacent_probability_probability_calculation_main_theorem_l2751_275115


namespace magic_box_solution_l2751_275122

def magic_box (a b : ℝ) : ℝ := a^2 + 2*b - 3

theorem magic_box_solution (m : ℝ) : 
  magic_box m (-3*m) = 4 ↔ m = 7 ∨ m = -1 := by sorry

end magic_box_solution_l2751_275122


namespace f_range_at_1_3_l2751_275147

def f (a b x y : ℝ) : ℝ := a * (x^3 + 3*x) + b * (y^2 + 2*y + 1)

theorem f_range_at_1_3 (a b : ℝ) (h1 : 1 ≤ f a b 1 2) (h2 : f a b 1 2 ≤ 2) 
  (h3 : 2 ≤ f a b 3 4) (h4 : f a b 3 4 ≤ 5) : 
  (3/2 : ℝ) ≤ f a b 1 3 ∧ f a b 1 3 ≤ 4 := by
sorry

end f_range_at_1_3_l2751_275147


namespace initial_amount_simple_interest_l2751_275142

/-- Simple interest calculation --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- Total amount after applying simple interest --/
def total_amount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + simple_interest principal rate time

/-- Theorem: Initial amount in a simple interest scenario --/
theorem initial_amount_simple_interest :
  ∃ (principal : ℝ),
    total_amount principal 0.10 5 = 1125 ∧
    principal = 750 := by
  sorry

end initial_amount_simple_interest_l2751_275142


namespace profit_calculation_l2751_275151

/-- Given a profit divided between two parties X and Y in the ratio 1/2 : 1/3,
    where the difference between their shares is 140, prove that the total profit is 700. -/
theorem profit_calculation (profit_x profit_y : ℚ) :
  profit_x / profit_y = 1/2 / (1/3 : ℚ) →
  profit_x - profit_y = 140 →
  profit_x + profit_y = 700 := by
  sorry

end profit_calculation_l2751_275151


namespace calculation_proof_l2751_275155

theorem calculation_proof : 8500 + 45 * 2 - 500 / 25 + 100 = 8670 := by
  sorry

end calculation_proof_l2751_275155


namespace sufficient_not_necessary_l2751_275120

theorem sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x > 1 ∧ y > 1 → x + y > 2) ∧
  (∃ x y : ℝ, x + y > 2 ∧ ¬(x > 1 ∧ y > 1)) :=
by sorry

end sufficient_not_necessary_l2751_275120


namespace jimmy_and_irene_payment_l2751_275176

/-- The amount paid by Jimmy and Irene for their clothing purchases with a senior citizen discount --/
def amountPaid (jimmyShorts : ℕ) (jimmyShortPrice : ℚ) (ireneShirts : ℕ) (ireneShirtPrice : ℚ) (discountPercentage : ℚ) : ℚ :=
  let totalCost := jimmyShorts * jimmyShortPrice + ireneShirts * ireneShirtPrice
  let discountAmount := totalCost * (discountPercentage / 100)
  totalCost - discountAmount

/-- Theorem stating that Jimmy and Irene pay $117 for their purchases --/
theorem jimmy_and_irene_payment :
  amountPaid 3 15 5 17 10 = 117 := by
  sorry

end jimmy_and_irene_payment_l2751_275176


namespace combined_standard_deviation_l2751_275198

/-- Given two groups of numbers with known means and variances, 
    calculate the standard deviation of the combined set. -/
theorem combined_standard_deviation 
  (n₁ n₂ : ℕ) 
  (mean₁ mean₂ : ℝ) 
  (var₁ var₂ : ℝ) :
  n₁ = 10 →
  n₂ = 10 →
  mean₁ = 50 →
  mean₂ = 40 →
  var₁ = 33 →
  var₂ = 45 →
  let n_total := n₁ + n₂
  let var_total := (n₁ * var₁ + n₂ * var₂) / n_total + 
                   (n₁ * n₂ : ℝ) / (n_total ^ 2 : ℝ) * (mean₁ - mean₂) ^ 2
  Real.sqrt var_total = 8 := by
  sorry

#check combined_standard_deviation

end combined_standard_deviation_l2751_275198


namespace compare_expressions_min_value_expression_l2751_275144

variable (m n : ℝ)

/-- Part 1: Compare m² + n and mn + m when m > n > 1 -/
theorem compare_expressions (hm : m > 0) (hn : n > 0) (hmn : m > n) (hn1 : n > 1) :
  m^2 + n > m*n + m := by sorry

/-- Part 2: Find the minimum value of 2/m + 1/n when m + 2n = 1 -/
theorem min_value_expression (hm : m > 0) (hn : n > 0) (hmn : m > n) (hsum : m + 2*n = 1) :
  ∃ (min_val : ℝ), min_val = 8 ∧ ∀ x, x = 2/m + 1/n → x ≥ min_val := by sorry

end compare_expressions_min_value_expression_l2751_275144


namespace diophantine_equation_solutions_l2751_275136

theorem diophantine_equation_solutions :
  ∀ m n : ℤ, 1 + 1996 * m + 1998 * n = m * n ↔
    (m = 1999 ∧ n = 1997^2 + 1996) ∨
    (m = 3995 ∧ n = 3993) ∨
    (m = 1997^2 + 1998 ∧ n = 1997) := by
  sorry

end diophantine_equation_solutions_l2751_275136


namespace factorization_theorem_l2751_275192

theorem factorization_theorem (x y a : ℝ) : 2*x*(a-2) - y*(2-a) = (a-2)*(2*x+y) := by
  sorry

end factorization_theorem_l2751_275192


namespace root_condition_for_k_l2751_275166

/-- The function f(x) = kx - 3 -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x - 3

/-- A function has a root in an interval if its values at the endpoints have different signs -/
def has_root_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  f a * f b ≤ 0

theorem root_condition_for_k (k : ℝ) :
  (k ≥ 3 → has_root_in_interval (f k) (-1) 1) ∧
  (∃ k', k' < 3 ∧ has_root_in_interval (f k') (-1) 1) :=
sorry

end root_condition_for_k_l2751_275166


namespace cook_selection_theorem_l2751_275149

/-- The number of ways to choose k items from n items. -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose 2 cooks from a group of 10 people,
    where one specific person must always be selected. -/
def cookSelectionWays : ℕ := choose 9 1

theorem cook_selection_theorem :
  cookSelectionWays = 9 := by sorry

end cook_selection_theorem_l2751_275149


namespace jills_salary_l2751_275154

theorem jills_salary (discretionary_income : ℝ) (net_salary : ℝ) : 
  discretionary_income = net_salary / 5 →
  0.30 * discretionary_income + 
  0.20 * discretionary_income + 
  0.35 * discretionary_income + 
  102 = discretionary_income →
  net_salary = 3400 := by
  sorry

end jills_salary_l2751_275154


namespace conference_handshakes_l2751_275196

/-- Represents a group of people at a conference -/
structure ConferenceGroup where
  total : ℕ
  group_a : ℕ
  group_b : ℕ
  h_total : total = group_a + group_b

/-- Calculates the number of handshakes in a conference group -/
def count_handshakes (g : ConferenceGroup) : ℕ :=
  (g.group_b * (g.group_b - 1)) / 2

/-- Theorem stating the number of handshakes in the specific conference scenario -/
theorem conference_handshakes :
  ∀ g : ConferenceGroup,
    g.total = 30 →
    g.group_a = 25 →
    g.group_b = 5 →
    count_handshakes g = 10 := by
  sorry


end conference_handshakes_l2751_275196


namespace combination_sum_equals_c_11_3_l2751_275171

theorem combination_sum_equals_c_11_3 :
  (Finset.range 9).sum (fun k => Nat.choose (k + 2) 2) = Nat.choose 11 3 := by
  sorry

end combination_sum_equals_c_11_3_l2751_275171


namespace town_distance_l2751_275110

/-- Three towns A, B, and C are equidistant from each other and are 3, 5, and 8 miles 
    respectively from a common railway station D. -/
structure TownConfiguration where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  equidistant : dist A B = dist B C ∧ dist B C = dist C A
  dist_AD : dist A D = 3
  dist_BD : dist B D = 5
  dist_CD : dist C D = 8

/-- The distance between any two towns is 7 miles. -/
theorem town_distance (config : TownConfiguration) : 
  dist config.A config.B = 7 ∧ dist config.B config.C = 7 ∧ dist config.C config.A = 7 := by
  sorry


end town_distance_l2751_275110


namespace octal_minus_quinary_in_decimal_l2751_275164

def base_to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * base ^ i) 0

def octal_54321 : List Nat := [1, 2, 3, 4, 5]
def quinary_4321 : List Nat := [1, 2, 3, 4]

theorem octal_minus_quinary_in_decimal : 
  base_to_decimal octal_54321 8 - base_to_decimal quinary_4321 5 = 22151 := by
  sorry

end octal_minus_quinary_in_decimal_l2751_275164


namespace equation_solution_l2751_275128

theorem equation_solution : 
  let f : ℝ → ℝ := λ x => (20 / (x^2 - 9)) - (3 / (x + 3)) - 2
  ∃ x₁ x₂ : ℝ, x₁ = (-3 + Real.sqrt 385) / 4 ∧ 
              x₂ = (-3 - Real.sqrt 385) / 4 ∧
              f x₁ = 0 ∧ f x₂ = 0 ∧
              ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end equation_solution_l2751_275128


namespace total_money_from_tshirts_l2751_275146

/-- The amount of money made from each t-shirt -/
def money_per_tshirt : ℕ := 215

/-- The number of t-shirts sold -/
def tshirts_sold : ℕ := 20

/-- The total money made from selling t-shirts -/
def total_money : ℕ := money_per_tshirt * tshirts_sold

theorem total_money_from_tshirts : total_money = 4300 := by
  sorry

end total_money_from_tshirts_l2751_275146


namespace mikes_work_days_l2751_275174

/-- Given that Mike worked 3 hours each day for a total of 15 hours,
    prove that he worked for 5 days. -/
theorem mikes_work_days (hours_per_day : ℕ) (total_hours : ℕ) (days : ℕ) : 
  hours_per_day = 3 → total_hours = 15 → days * hours_per_day = total_hours → days = 5 := by
  sorry

end mikes_work_days_l2751_275174


namespace minimum_transportation_cost_l2751_275141

/-- Represents the capacity of a truck -/
structure TruckCapacity where
  tents : ℕ
  food : ℕ

/-- Represents a truck arrangement -/
structure TruckArrangement where
  typeA : ℕ
  typeB : ℕ

/-- Calculate the total items an arrangement can carry -/
def totalCapacity (c : TruckCapacity × TruckCapacity) (a : TruckArrangement) : ℕ × ℕ :=
  (a.typeA * c.1.tents + a.typeB * c.2.tents, a.typeA * c.1.food + a.typeB * c.2.food)

/-- Calculate the cost of an arrangement -/
def arrangementCost (costs : ℕ × ℕ) (a : TruckArrangement) : ℕ :=
  a.typeA * costs.1 + a.typeB * costs.2

theorem minimum_transportation_cost :
  let totalItems : ℕ := 320
  let tentsDiff : ℕ := 80
  let totalTrucks : ℕ := 8
  let typeACapacity : TruckCapacity := ⟨40, 10⟩
  let typeBCapacity : TruckCapacity := ⟨20, 20⟩
  let costs : ℕ × ℕ := (4000, 3600)
  let tents : ℕ := (totalItems + tentsDiff) / 2
  let food : ℕ := (totalItems - tentsDiff) / 2
  ∃ (a : TruckArrangement),
    a.typeA + a.typeB = totalTrucks ∧
    totalCapacity (typeACapacity, typeBCapacity) a = (tents, food) ∧
    ∀ (b : TruckArrangement),
      b.typeA + b.typeB = totalTrucks →
      totalCapacity (typeACapacity, typeBCapacity) b = (tents, food) →
      arrangementCost costs a ≤ arrangementCost costs b ∧
      arrangementCost costs a = 29600 :=
by
  sorry

end minimum_transportation_cost_l2751_275141


namespace triangle_side_length_l2751_275186

/-- Given a triangle DEF with sides d, e, and f, where d = 7, e = 3, and cos(D - E) = 39/40,
    prove that the length of side f is equal to √(9937)/10. -/
theorem triangle_side_length (D E F : ℝ) (d e f : ℝ) : 
  d = 7 → 
  e = 3 → 
  Real.cos (D - E) = 39 / 40 → 
  f = Real.sqrt 9937 / 10 := by
  sorry

end triangle_side_length_l2751_275186


namespace max_group_size_problem_l2751_275165

/-- The maximum number of people in a group for two classes with given total students and leftovers -/
def max_group_size (class1_total : ℕ) (class2_total : ℕ) (class1_leftover : ℕ) (class2_leftover : ℕ) : ℕ :=
  Nat.gcd (class1_total - class1_leftover) (class2_total - class2_leftover)

/-- Theorem stating that the maximum group size for the given problem is 16 -/
theorem max_group_size_problem : max_group_size 69 86 5 6 = 16 := by
  sorry

end max_group_size_problem_l2751_275165


namespace max_reflections_is_largest_l2751_275190

/-- Represents the angle between lines AD and CD in degrees -/
def angle_CDA : ℝ := 5

/-- Represents the maximum allowed path length -/
def max_path_length : ℝ := 100

/-- Calculates the total angle after n reflections -/
def total_angle (n : ℕ) : ℝ := n * angle_CDA

/-- Represents the condition that the total angle must not exceed 90 degrees -/
def angle_condition (n : ℕ) : Prop := total_angle n ≤ 90

/-- Represents an approximation of the path length after n reflections -/
def approx_path_length (n : ℕ) : ℝ := 2 * n * 5

/-- Represents the condition that the path length must not exceed the maximum allowed length -/
def path_length_condition (n : ℕ) : Prop := approx_path_length n ≤ max_path_length

/-- Represents the maximum number of reflections that satisfies all conditions -/
def max_reflections : ℕ := 10

/-- Theorem stating that max_reflections is the largest value that satisfies all conditions -/
theorem max_reflections_is_largest :
  (angle_condition max_reflections) ∧
  (path_length_condition max_reflections) ∧
  (∀ m : ℕ, m > max_reflections → ¬(angle_condition m ∧ path_length_condition m)) :=
sorry

end max_reflections_is_largest_l2751_275190


namespace pizza_toppings_l2751_275156

/-- Given a pizza with the following properties:
  * It has 16 slices in total
  * Every slice has at least one topping
  * There are three toppings: cheese, chicken, and olives
  * 8 slices have cheese
  * 12 slices have chicken
  * 6 slices have olives
  This theorem proves that exactly 5 slices have all three toppings. -/
theorem pizza_toppings (total_slices : ℕ) (cheese_slices : ℕ) (chicken_slices : ℕ) (olive_slices : ℕ)
    (h_total : total_slices = 16)
    (h_cheese : cheese_slices = 8)
    (h_chicken : chicken_slices = 12)
    (h_olives : olive_slices = 6)
    (h_at_least_one : ∀ slice, slice ∈ Finset.range total_slices →
      (slice ∈ Finset.range cheese_slices ∨
       slice ∈ Finset.range chicken_slices ∨
       slice ∈ Finset.range olive_slices)) :
    ∃ all_toppings : ℕ, all_toppings = 5 ∧
      (∀ slice, slice ∈ Finset.range total_slices →
        (slice ∈ Finset.range cheese_slices ∧
         slice ∈ Finset.range chicken_slices ∧
         slice ∈ Finset.range olive_slices) ↔
        slice ∈ Finset.range all_toppings) := by
  sorry

end pizza_toppings_l2751_275156


namespace difference_of_squares_l2751_275197

theorem difference_of_squares (x y : ℝ) : x^2 - y^2 = (x + y) * (x - y) := by
  sorry

end difference_of_squares_l2751_275197


namespace games_last_year_l2751_275121

/-- The number of basketball games Fred attended this year -/
def games_this_year : ℕ := 25

/-- The difference in games attended between last year and this year -/
def games_difference : ℕ := 11

/-- Theorem stating the number of games Fred attended last year -/
theorem games_last_year : games_this_year + games_difference = 36 := by
  sorry

end games_last_year_l2751_275121


namespace quadratic_symmetry_axis_l2751_275168

/-- Given a quadratic function y = x^2 + 2mx + 2 with symmetry axis x = 2, prove that m = -2 -/
theorem quadratic_symmetry_axis (m : ℝ) : 
  (∀ x, x^2 + 2*m*x + 2 = (x-2)^2 + (2^2 + 2*m*2 + 2)) → m = -2 := by
  sorry

end quadratic_symmetry_axis_l2751_275168


namespace log_inequality_l2751_275101

theorem log_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  Real.log (1 + x + y) < x + y := by
  sorry

end log_inequality_l2751_275101


namespace curve_C_equation_sum_of_slopes_constant_l2751_275180

noncomputable section

def Circle (O : ℝ × ℝ) (r : ℝ) := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r^2}

def Curve_C := {N : ℝ × ℝ | N.1^2 / 6 + N.2^2 / 3 = 1}

def Point_on_circle (P : ℝ × ℝ) := P.1^2 + P.2^2 = 6

def Point_N (P N : ℝ × ℝ) := 
  ∃ (M : ℝ × ℝ), M.2 = 0 ∧ (P.1 - M.1)^2 + (P.2 - M.2)^2 = 2 * ((N.1 - M.1)^2 + (N.2 - M.2)^2)

def Line_through_B (k : ℝ) := {P : ℝ × ℝ | P.2 = k * (P.1 - 3)}

def Slope (A B : ℝ × ℝ) := (B.2 - A.2) / (B.1 - A.1)

theorem curve_C_equation :
  ∀ N : ℝ × ℝ, (∃ P : ℝ × ℝ, Point_on_circle P ∧ Point_N P N) → N ∈ Curve_C := by sorry

theorem sum_of_slopes_constant :
  ∀ k : ℝ, ∀ D E : ℝ × ℝ,
    D ∈ Curve_C ∧ E ∈ Curve_C ∧ D ∈ Line_through_B k ∧ E ∈ Line_through_B k ∧ D ≠ E →
    Slope (2, 1) D + Slope (2, 1) E = -2 := by sorry

end curve_C_equation_sum_of_slopes_constant_l2751_275180


namespace rectangular_prism_width_l2751_275173

theorem rectangular_prism_width (l h d : ℝ) (hl : l = 5) (hh : h = 15) (hd : d = 17) :
  ∃ w : ℝ, w > 0 ∧ w^2 = 39 ∧ d^2 = l^2 + w^2 + h^2 := by
  sorry

end rectangular_prism_width_l2751_275173


namespace f_property_l2751_275177

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1)^2 - a * Real.log x

theorem f_property (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ > 0 ∧ x₂ > 0 ∧
    ∀ y₁ y₂ : ℝ, y₁ ≠ y₂ → y₁ > 0 → y₂ > 0 →
      (f a (y₁ + 1) - f a (y₂ + 1)) / (y₁ - y₂) > 1) →
  a ≤ 3 :=
sorry

end f_property_l2751_275177


namespace max_one_truthful_dwarf_l2751_275158

/-- Represents the height claim of a dwarf -/
structure HeightClaim where
  position : Nat
  claimed_height : Nat

/-- The problem setup for the seven dwarfs -/
def dwarfs_problem : List HeightClaim :=
  [
    ⟨1, 60⟩,
    ⟨2, 61⟩,
    ⟨3, 62⟩,
    ⟨4, 63⟩,
    ⟨5, 64⟩,
    ⟨6, 65⟩,
    ⟨7, 66⟩
  ]

/-- A function to count the maximum number of truthful dwarfs -/
def max_truthful_dwarfs (claims : List HeightClaim) : Nat :=
  sorry

/-- The theorem stating that the maximum number of truthful dwarfs is 1 -/
theorem max_one_truthful_dwarf :
  max_truthful_dwarfs dwarfs_problem = 1 :=
sorry

end max_one_truthful_dwarf_l2751_275158


namespace product_of_terms_l2751_275148

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem product_of_terms (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1) ^ 2 - 8 * (a 1) + 1 = 0 →
  (a 13) ^ 2 - 8 * (a 13) + 1 = 0 →
  a 5 * a 7 * a 9 = 1 :=
by sorry

end product_of_terms_l2751_275148


namespace power_sum_equality_l2751_275113

theorem power_sum_equality : (-1)^53 + 3^(2^3 + 5^2 - 7^2) = -43046720 / 43046721 := by
  sorry

end power_sum_equality_l2751_275113


namespace representation_theorem_l2751_275162

theorem representation_theorem (a b : ℕ+) :
  (∃ (S : Finset ℕ), ∀ (n : ℕ), ∃ (x y : ℕ) (s : ℕ), s ∈ S ∧ n = x^(a:ℕ) + y^(b:ℕ) + s) ↔
  (a = 1 ∨ b = 1) :=
sorry

end representation_theorem_l2751_275162


namespace trigonometric_identities_l2751_275103

theorem trigonometric_identities (α : Real) 
  (h1 : (Real.tan α) / (Real.tan α - 1) = -1)
  (h2 : α ∈ Set.Icc (Real.pi) (3 * Real.pi / 2)) :
  (((Real.sin α - 3 * Real.cos α) / (Real.sin α + Real.cos α)) = -5/3) ∧
  ((Real.cos (-Real.pi + α) + Real.cos (Real.pi/2 + α)) = 3 * Real.sqrt 5 / 5) := by
  sorry

end trigonometric_identities_l2751_275103


namespace sequence_convergence_bound_l2751_275183

def x : ℕ → ℚ
  | 0 => 6
  | n + 1 => (x n ^ 2 + 6 * x n + 7) / (x n + 7)

theorem sequence_convergence_bound :
  ∃ m : ℕ, m ∈ Set.Icc 151 300 ∧
    x m ≤ 4 + 1 / (2^25) ∧
    ∀ k : ℕ, k < m → x k > 4 + 1 / (2^25) :=
by sorry

end sequence_convergence_bound_l2751_275183


namespace mark_second_play_time_l2751_275134

/-- Calculates the time Mark played in the second part of a soccer game. -/
def second_play_time (total_time initial_play sideline : ℕ) : ℕ :=
  total_time - initial_play - sideline

/-- Theorem: Mark played 35 minutes in the second part of the game. -/
theorem mark_second_play_time :
  let total_time : ℕ := 90
  let initial_play : ℕ := 20
  let sideline : ℕ := 35
  second_play_time total_time initial_play sideline = 35 := by
  sorry

end mark_second_play_time_l2751_275134


namespace tax_discount_commute_l2751_275137

theorem tax_discount_commute (price : ℝ) (tax_rate discount_rate : ℝ) 
  (h1 : 0 ≤ tax_rate) (h2 : 0 ≤ discount_rate) (h3 : tax_rate < 1) (h4 : discount_rate < 1) :
  price * (1 + tax_rate) * (1 - discount_rate) = price * (1 - discount_rate) * (1 + tax_rate) :=
by sorry

end tax_discount_commute_l2751_275137


namespace complex_equation_solution_l2751_275143

theorem complex_equation_solution (z : ℂ) :
  z / (1 - 2 * Complex.I) = Complex.I → z = 2 + Complex.I := by
  sorry

end complex_equation_solution_l2751_275143


namespace pentagon_perimeter_division_l2751_275117

/-- Given a regular pentagon with perimeter 125 and side length 25,
    prove that the perimeter divided by the side length equals 5. -/
theorem pentagon_perimeter_division (perimeter : ℝ) (side_length : ℝ) :
  perimeter = 125 →
  side_length = 25 →
  perimeter / side_length = 5 := by
  sorry

end pentagon_perimeter_division_l2751_275117


namespace set_equality_l2751_275175

-- Define the sets A, B, and C as subsets of ℝ
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 2 ≤ x ∧ x ≤ 4}
def C : Set ℝ := {x | 3 < x ∧ x ≤ 4}

-- State the theorem
theorem set_equality : C = (Set.univ \ A) ∩ B := by
  sorry

end set_equality_l2751_275175


namespace polygon_sides_when_interior_four_times_exterior_l2751_275132

theorem polygon_sides_when_interior_four_times_exterior : 
  ∀ n : ℕ, n > 2 →
  (n - 2) * 180 = 4 * 360 →
  n = 6 :=
by
  sorry

end polygon_sides_when_interior_four_times_exterior_l2751_275132


namespace train_carriages_l2751_275140

theorem train_carriages (num_trains : ℕ) (rows_per_carriage : ℕ) (wheels_per_row : ℕ) (total_wheels : ℕ) :
  num_trains = 4 → rows_per_carriage = 3 → wheels_per_row = 5 → total_wheels = 240 →
  (total_wheels / (rows_per_carriage * wheels_per_row)) / num_trains = 4 :=
by
  sorry

end train_carriages_l2751_275140


namespace time_to_distance_l2751_275150

/-- Theorem: Time to reach a certain distance for two people walking in opposite directions -/
theorem time_to_distance (mary_speed sharon_speed : ℝ) (initial_time initial_distance : ℝ) :
  mary_speed = 4 →
  sharon_speed = 6 →
  initial_time = 0.3 →
  initial_distance = 3 →
  ∀ d : ℝ, d > 0 → ∃ t : ℝ, t = d / (mary_speed + sharon_speed) ∧ d = (mary_speed + sharon_speed) * t :=
by sorry

end time_to_distance_l2751_275150


namespace function_always_positive_range_l2751_275153

theorem function_always_positive_range (a : ℝ) : 
  (∀ x : ℝ, (5 - a) * x^2 - 6 * x + a + 5 > 0) ↔ (-4 < a ∧ a < 4) := by
  sorry

end function_always_positive_range_l2751_275153


namespace range_of_a_l2751_275112

def f (a : ℝ) (x : ℝ) : ℝ := (x - 2)^2 * |x - a|

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 2 4, x * (deriv (f a) x) ≥ 0) ↔ a ∈ Set.Iic 2 ∪ Set.Ici 5 :=
sorry

end range_of_a_l2751_275112


namespace at_least_one_nonzero_l2751_275108

theorem at_least_one_nonzero (a b : ℝ) : (a ≠ 0 ∨ b ≠ 0) ↔ a^2 + b^2 > 0 := by
  sorry

end at_least_one_nonzero_l2751_275108


namespace perpendicular_line_equation_l2751_275181

/-- The equation of a line perpendicular to x-2y=3 and passing through (1,2) is y=-2x+4 -/
theorem perpendicular_line_equation :
  ∀ (x y : ℝ),
  (∃ (m b : ℝ), y = m*x + b ∧ 
                 (1, 2) ∈ {(x, y) | y = m*x + b} ∧
                 m * (1/2) = -1) →
  y = -2*x + 4 := by
sorry

end perpendicular_line_equation_l2751_275181


namespace largest_of_four_consecutive_odd_integers_l2751_275107

theorem largest_of_four_consecutive_odd_integers (a b c d : ℤ) : 
  (∀ n : ℤ, a = 2*n + 1 ∧ b = 2*n + 3 ∧ c = 2*n + 5 ∧ d = 2*n + 7) → 
  (a + b + c + d = 200) → 
  d = 53 := by
sorry

end largest_of_four_consecutive_odd_integers_l2751_275107


namespace difference_of_101st_terms_l2751_275167

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + d * (n - 1)

theorem difference_of_101st_terms : 
  let X := arithmetic_sequence 40 12
  let Y := arithmetic_sequence 40 (-8)
  |X 101 - Y 101| = 2000 := by
sorry

end difference_of_101st_terms_l2751_275167


namespace student_distribution_l2751_275111

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of permutations of n items -/
def factorial (n : ℕ) : ℕ := sorry

theorem student_distribution (total : ℕ) (male : ℕ) (schemes : ℕ) :
  total = 8 →
  choose male 2 * choose (total - male) 1 * factorial 3 = schemes →
  schemes = 90 →
  male = 3 ∧ total - male = 5 := by sorry

end student_distribution_l2751_275111


namespace perfect_square_from_48_numbers_l2751_275184

theorem perfect_square_from_48_numbers (S : Finset ℕ) 
  (h1 : S.card = 48)
  (h2 : (S.prod id).factors.toFinset.card = 10) :
  ∃ (a b c d : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
  ∃ (m : ℕ), a * b * c * d = m ^ 2 :=
sorry

end perfect_square_from_48_numbers_l2751_275184


namespace intersection_and_parallel_line_intersection_and_double_angle_line_l2751_275133

-- Define the two original lines
def l₁ (x y : ℝ) : Prop := x - 2*y + 3 = 0
def l₂ (x y : ℝ) : Prop := x + 2*y - 9 = 0

-- Define the intersection point A
def A : ℝ × ℝ := (3, 3)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 2*x + 3*y - 1 = 0

theorem intersection_and_parallel_line :
  ∀ x y : ℝ, l₁ x y ∧ l₂ x y → (2*x + 3*y - 15 = 0) := by sorry

theorem intersection_and_double_angle_line :
  ∀ x y : ℝ, l₁ x y ∧ l₂ x y → (4*x - 3*y - 3 = 0) := by sorry

end intersection_and_parallel_line_intersection_and_double_angle_line_l2751_275133


namespace time_to_go_up_mountain_l2751_275104

/-- Represents the hiking trip with given parameters. -/
structure HikingTrip where
  rate_up : ℝ
  rate_down : ℝ
  distance_down : ℝ
  time_up : ℝ
  time_down : ℝ

/-- The hiking trip satisfies the given conditions. -/
def satisfies_conditions (trip : HikingTrip) : Prop :=
  trip.time_up = trip.time_down ∧
  trip.rate_down = 1.5 * trip.rate_up ∧
  trip.rate_up = 5 ∧
  trip.distance_down = 15

/-- Theorem stating that for a trip satisfying the conditions, 
    the time to go up the mountain is 2 days. -/
theorem time_to_go_up_mountain (trip : HikingTrip) 
  (h : satisfies_conditions trip) : trip.time_up = 2 := by
  sorry


end time_to_go_up_mountain_l2751_275104


namespace work_done_by_force_l2751_275118

/-- Work done by a force on a particle -/
theorem work_done_by_force (F S : ℝ × ℝ) : 
  F = (-1, -2) → S = (3, 4) → F.1 * S.1 + F.2 * S.2 = -11 := by sorry

end work_done_by_force_l2751_275118


namespace arithmetic_computation_l2751_275102

theorem arithmetic_computation : 1325 + 572 / 52 - 225 + 2^3 = 1119 := by
  sorry

end arithmetic_computation_l2751_275102


namespace chocolate_count_l2751_275195

theorem chocolate_count : ∀ x : ℚ,
  let day1_remaining := (3 / 5 : ℚ) * x - 3
  let day2_remaining := (3 / 4 : ℚ) * day1_remaining - 5
  day2_remaining = 10 → x = 105 := by
  sorry

end chocolate_count_l2751_275195


namespace sqrt_12_sqrt_2_div_sqrt_3_minus_2sin45_equals_1_l2751_275125

theorem sqrt_12_sqrt_2_div_sqrt_3_minus_2sin45_equals_1 :
  let sqrt_12 := 2 * Real.sqrt 3
  let sin_45 := Real.sqrt 2 / 2
  (sqrt_12 * Real.sqrt 2) / Real.sqrt 3 - 2 * sin_45 = 1 := by
  sorry

end sqrt_12_sqrt_2_div_sqrt_3_minus_2sin45_equals_1_l2751_275125


namespace proposition_and_variations_l2751_275170

theorem proposition_and_variations (x : ℝ) :
  ((x = 3 ∨ x = 7) → (x - 3) * (x - 7) = 0) ∧
  ((x - 3) * (x - 7) = 0 → (x = 3 ∨ x = 7)) ∧
  ((x ≠ 3 ∧ x ≠ 7) → (x - 3) * (x - 7) ≠ 0) ∧
  ((x - 3) * (x - 7) ≠ 0 → (x ≠ 3 ∧ x ≠ 7)) :=
by sorry

end proposition_and_variations_l2751_275170


namespace food_court_combinations_l2751_275172

/-- Represents the number of options for each meal component -/
structure MealOptions where
  entrees : Nat
  drinks : Nat
  desserts : Nat

/-- Calculates the total number of meal combinations -/
def mealCombinations (options : MealOptions) : Nat :=
  options.entrees * options.drinks * options.desserts

/-- The given meal options in the food court -/
def foodCourtOptions : MealOptions :=
  { entrees := 4, drinks := 4, desserts := 2 }

/-- Theorem: The number of distinct meal combinations in the food court is 32 -/
theorem food_court_combinations :
  mealCombinations foodCourtOptions = 32 := by
  sorry

end food_court_combinations_l2751_275172


namespace greatest_n_for_perfect_square_T_l2751_275199

/-- The greatest power of 4 that divides an even positive integer -/
def h (x : ℕ+) : ℕ :=
  sorry

/-- Sum of h(4k) from k = 1 to 2^(n-1) -/
def T (n : ℕ+) : ℕ :=
  sorry

/-- Predicate for perfect squares -/
def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k^2

theorem greatest_n_for_perfect_square_T :
  ∀ n : ℕ+, n < 500 → is_perfect_square (T n) → n ≤ 143 ∧
  is_perfect_square (T 143) :=
sorry

end greatest_n_for_perfect_square_T_l2751_275199


namespace expression_simplification_and_evaluation_l2751_275129

theorem expression_simplification_and_evaluation :
  let x : ℝ := Real.sqrt 3 + 1
  (x / (x^2 - 1)) / (1 - 1 / (x + 1)) = Real.sqrt 3 / 3 := by
  sorry

end expression_simplification_and_evaluation_l2751_275129


namespace sum_equals_difference_l2751_275159

theorem sum_equals_difference (N : ℤ) : 
  995 + 997 + 999 + 1001 + 1003 = 5100 - N → N = 100 := by
  sorry

end sum_equals_difference_l2751_275159


namespace body_speeds_correct_l2751_275193

/-- The distance between points A and B in meters -/
def distance : ℝ := 270

/-- The time (in seconds) after which the second body starts moving -/
def delay : ℝ := 11

/-- The time (in seconds) of the first meeting after the second body starts moving -/
def first_meeting : ℝ := 10

/-- The time (in seconds) of the second meeting after the second body starts moving -/
def second_meeting : ℝ := 40

/-- The speed of the first body in meters per second -/
def v1 : ℝ := 16

/-- The speed of the second body in meters per second -/
def v2 : ℝ := 9.6

theorem body_speeds_correct : 
  (delay + first_meeting) * v1 + first_meeting * v2 = distance ∧
  (delay + second_meeting) * v1 - second_meeting * v2 = distance ∧
  v1 > v2 ∧ v2 > 0 := by sorry

end body_speeds_correct_l2751_275193


namespace wire_cutting_problem_l2751_275152

theorem wire_cutting_problem (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 14 →
  ratio = 2 / 5 →
  shorter_length + (shorter_length / ratio) = total_length →
  shorter_length = 4 := by
  sorry

end wire_cutting_problem_l2751_275152


namespace quadratic_equation_result_l2751_275163

theorem quadratic_equation_result (a : ℝ) (h : a^2 - 2*a + 1 = 0) : 
  4*a - 2*a^2 + 2 = 4 := by
  sorry

end quadratic_equation_result_l2751_275163


namespace initial_plums_count_l2751_275100

/-- The number of plums Melanie picked initially -/
def initial_plums : ℕ := sorry

/-- The number of plums Melanie gave to Sam -/
def plums_given : ℕ := 3

/-- The number of plums Melanie has left -/
def plums_left : ℕ := 4

/-- Theorem stating that the initial number of plums equals 7 -/
theorem initial_plums_count : initial_plums = 7 := by sorry

end initial_plums_count_l2751_275100


namespace quadratic_solution_sum_l2751_275131

theorem quadratic_solution_sum (p q : ℝ) : 
  (∀ x : ℂ, (2 * x^2 + 5 = 7 * x - 2) ↔ (x = p + q * I ∨ x = p - q * I)) →
  p + q^2 = 35/16 := by
  sorry

end quadratic_solution_sum_l2751_275131


namespace x_squared_plus_reciprocal_l2751_275178

theorem x_squared_plus_reciprocal (x : ℝ) (h : 47 = x^4 + 1/x^4) : x^2 + 1/x^2 = 7 := by
  sorry

end x_squared_plus_reciprocal_l2751_275178


namespace male_salmon_count_l2751_275124

def total_salmon : ℕ := 971639
def female_salmon : ℕ := 259378

theorem male_salmon_count : total_salmon - female_salmon = 712261 := by
  sorry

end male_salmon_count_l2751_275124


namespace double_reflection_result_l2751_275116

/-- Reflects a point about the line y = x -/
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, p.1)

/-- Reflects a point about the line y = -x -/
def reflect_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

/-- The initial point -/
def initial_point : ℝ × ℝ := (3, -8)

theorem double_reflection_result :
  (reflect_y_eq_neg_x ∘ reflect_y_eq_x) initial_point = (-3, 8) := by
sorry

end double_reflection_result_l2751_275116


namespace marcus_final_cards_l2751_275169

def marcus_initial_cards : ℕ := 2100
def carter_initial_cards : ℕ := 3040
def carter_gift_cards : ℕ := 750
def carter_gift_percentage : ℚ := 125 / 1000

theorem marcus_final_cards : 
  marcus_initial_cards + carter_gift_cards + 
  (carter_initial_cards * carter_gift_percentage).floor = 3230 :=
by sorry

end marcus_final_cards_l2751_275169


namespace equation_solutions_l2751_275185

-- Define the equation
def equation (r p : ℤ) : Prop := r^2 - r*(p + 6) + p^2 + 5*p + 6 = 0

-- Define the set of solution pairs
def solution_set : Set (ℤ × ℤ) := {(3,1), (4,1), (0,-2), (4,-2), (0,-3), (3,-3)}

-- Theorem statement
theorem equation_solutions :
  ∀ (r p : ℤ), equation r p ↔ (r, p) ∈ solution_set :=
sorry

end equation_solutions_l2751_275185


namespace absolute_value_comparison_l2751_275119

theorem absolute_value_comparison (a b : ℚ) : 
  |a| = 2/3 ∧ |b| = 3/5 → 
  ((a = 2/3 ∨ a = -2/3) ∧ 
   (b = 3/5 ∨ b = -3/5) ∧ 
   (a = 2/3 → a > b) ∧ 
   (a = -2/3 → a < b)) := by
  sorry

end absolute_value_comparison_l2751_275119


namespace ratio_fraction_equality_l2751_275105

theorem ratio_fraction_equality (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 6) :
  (5 * A + 3 * B) / (5 * C - 2 * A) = 7 / 8 := by
  sorry

end ratio_fraction_equality_l2751_275105


namespace coffee_cost_is_18_l2751_275139

/-- Represents the coffee consumption and cost parameters --/
structure CoffeeParams where
  cups_per_day : ℕ
  oz_per_cup : ℚ
  bag_cost : ℚ
  oz_per_bag : ℚ
  milk_gal_per_week : ℚ
  milk_cost_per_gal : ℚ

/-- Calculates the weekly cost of coffee given the parameters --/
def weekly_coffee_cost (params : CoffeeParams) : ℚ :=
  let beans_oz_per_week := params.cups_per_day * params.oz_per_cup * 7
  let bags_per_week := beans_oz_per_week / params.oz_per_bag
  let bean_cost := bags_per_week * params.bag_cost
  let milk_cost := params.milk_gal_per_week * params.milk_cost_per_gal
  bean_cost + milk_cost

/-- Theorem stating that the weekly coffee cost is $18 --/
theorem coffee_cost_is_18 :
  ∃ (params : CoffeeParams),
    params.cups_per_day = 2 ∧
    params.oz_per_cup = 3/2 ∧
    params.bag_cost = 8 ∧
    params.oz_per_bag = 21/2 ∧
    params.milk_gal_per_week = 1/2 ∧
    params.milk_cost_per_gal = 4 ∧
    weekly_coffee_cost params = 18 :=
  sorry

end coffee_cost_is_18_l2751_275139


namespace average_pages_per_day_l2751_275138

theorem average_pages_per_day 
  (total_pages : ℕ) 
  (pages_read : ℕ) 
  (remaining_days : ℕ) 
  (h1 : total_pages = 212) 
  (h2 : pages_read = 97) 
  (h3 : remaining_days = 5) :
  (total_pages - pages_read) / remaining_days = 23 := by
sorry

end average_pages_per_day_l2751_275138


namespace boys_to_girls_ratio_l2751_275188

/-- Given a class of students where half the number of girls equals one-fifth of the total number of students, prove that the ratio of boys to girls is 3:2. -/
theorem boys_to_girls_ratio (S : ℕ) (G : ℕ) (h : 2 * G = S) :
  (S - G) / G = 3 / 2 := by
  sorry

end boys_to_girls_ratio_l2751_275188


namespace monotonic_increase_interval_l2751_275109

theorem monotonic_increase_interval
  (f : ℝ → ℝ)
  (φ : ℝ)
  (h1 : ∀ x, f x = Real.sin (2 * x + φ))
  (h2 : ∀ x, f x ≤ |f (π / 6)|)
  (h3 : f (π / 2) > f π) :
  ∀ k : ℤ, StrictMonoOn f (Set.Icc (k * π + π / 6) (k * π + 2 * π / 3)) :=
by sorry

end monotonic_increase_interval_l2751_275109


namespace discount_percentage_l2751_275145

theorem discount_percentage (M : ℝ) (C : ℝ) (SP : ℝ) 
  (h1 : C = 0.64 * M) 
  (h2 : SP = C * 1.171875) : 
  (M - SP) / M * 100 = 25 := by
  sorry

end discount_percentage_l2751_275145


namespace complex_fraction_equality_l2751_275194

theorem complex_fraction_equality : Complex.I * 5 / (1 - Complex.I) = -5/2 + Complex.I * (5/2) := by
  sorry

end complex_fraction_equality_l2751_275194


namespace smartphone_price_difference_l2751_275179

/-- Calculate the final price after applying a discount --/
def final_price (initial_price : ℚ) (discount_percent : ℚ) : ℚ :=
  initial_price * (1 - discount_percent / 100)

/-- The problem statement --/
theorem smartphone_price_difference :
  let store_a_initial_price : ℚ := 125
  let store_a_discount : ℚ := 8
  let store_b_initial_price : ℚ := 130
  let store_b_discount : ℚ := 10
  
  final_price store_b_initial_price store_b_discount -
  final_price store_a_initial_price store_a_discount = 2 := by
  sorry

end smartphone_price_difference_l2751_275179


namespace maria_cookies_left_l2751_275187

/-- Calculates the number of cookies Maria has left after distributing them -/
def cookiesLeft (initialCookies : ℕ) : ℕ :=
  let afterFriend := initialCookies - (initialCookies * 20 / 100)
  let afterFamily := afterFriend - (afterFriend / 3)
  let afterEating := afterFamily - 4
  let toNeighbor := afterEating / 6
  afterEating - toNeighbor

/-- Theorem stating that Maria will have 24 cookies left -/
theorem maria_cookies_left : cookiesLeft 60 = 24 := by
  sorry

end maria_cookies_left_l2751_275187


namespace absolute_difference_of_integers_l2751_275189

theorem absolute_difference_of_integers (x y : ℤ) 
  (h1 : x ≠ y)
  (h2 : (x + y) / 2 = 15)
  (h3 : Real.sqrt (x * y) + 6 = 15) : 
  |x - y| = 24 := by
  sorry

end absolute_difference_of_integers_l2751_275189


namespace parade_average_l2751_275106

theorem parade_average (boys girls rows : ℕ) (h1 : boys = 24) (h2 : girls = 24) (h3 : rows = 6) :
  (boys + girls) / rows = 8 :=
sorry

end parade_average_l2751_275106


namespace geometric_sequence_quadratic_roots_l2751_275123

/-- Given real numbers 2, b, and a form a geometric sequence, 
    the equation ax^2 + bx + 1/3 = 0 has exactly 2 real roots -/
theorem geometric_sequence_quadratic_roots 
  (b a : ℝ) 
  (h_geometric : ∃ (q : ℝ), b = 2 * q ∧ a = 2 * q^2) :
  (∃! (x y : ℝ), x ≠ y ∧ 
    (∀ (z : ℝ), a * z^2 + b * z + 1/3 = 0 ↔ z = x ∨ z = y)) := by
  sorry

end geometric_sequence_quadratic_roots_l2751_275123


namespace seventh_grade_rooms_l2751_275161

/-- The number of rooms on the first floor where seventh-grade boys live -/
def num_rooms : ℕ := sorry

/-- The total number of students -/
def total_students : ℕ := sorry

theorem seventh_grade_rooms :
  (6 * (num_rooms - 1) = total_students) ∧
  (5 * num_rooms + 4 = total_students) →
  num_rooms = 10 := by
sorry

end seventh_grade_rooms_l2751_275161


namespace order_relation_abc_l2751_275127

/-- Prove that given a = (4 - ln 4) / e^2, b = ln 2 / 2, and c = 1/e, we have b < a < c -/
theorem order_relation_abc :
  let a : ℝ := (4 - Real.log 4) / Real.exp 2
  let b : ℝ := Real.log 2 / 2
  let c : ℝ := 1 / Real.exp 1
  b < a ∧ a < c := by
  sorry

end order_relation_abc_l2751_275127


namespace unique_modulo_representation_l2751_275126

theorem unique_modulo_representation :
  ∃! n : ℤ, 0 ≤ n ∧ n < 7 ∧ -2222 ≡ n [ZMOD 7] ∧ n = 3 := by
  sorry

end unique_modulo_representation_l2751_275126


namespace right_triangle_sin_c_l2751_275160

theorem right_triangle_sin_c (A B C : Real) (h1 : A + B + C = Real.pi) 
  (h2 : B = Real.pi / 2) (h3 : Real.tan A = 3 / 4) : Real.sin C = 4 / 5 := by
  sorry

end right_triangle_sin_c_l2751_275160


namespace f_sum_equals_e_minus_one_l2751_275157

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period_two (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

theorem f_sum_equals_e_minus_one 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_period : has_period_two f)
  (h_interval : ∀ x ∈ Set.Icc 0 1, f x = Real.exp x - 1) :
  f 2018 + f (-2019) = Real.exp 1 - 1 := by
  sorry

end f_sum_equals_e_minus_one_l2751_275157


namespace siding_total_cost_l2751_275130

def wall_width : ℝ := 10
def wall_height : ℝ := 7
def roof_width : ℝ := 10
def roof_height : ℝ := 6
def roof_sections : ℕ := 2
def siding_width : ℝ := 10
def siding_height : ℝ := 15
def siding_cost : ℝ := 35

theorem siding_total_cost :
  let total_area := wall_width * wall_height + roof_width * roof_height * roof_sections
  let siding_area := siding_width * siding_height
  let sections_needed := Int.ceil (total_area / siding_area)
  sections_needed * siding_cost = 70 := by sorry

end siding_total_cost_l2751_275130


namespace cookie_radius_l2751_275182

theorem cookie_radius (x y : ℝ) :
  (x^2 + y^2 - 8 = 2*x + 4*y) →
  ∃ (h k r : ℝ), (x - h)^2 + (y - k)^2 = r^2 ∧ r = Real.sqrt 13 :=
by sorry

end cookie_radius_l2751_275182


namespace discounted_three_books_cost_l2751_275135

/-- The cost of two identical books without discount -/
def two_books_cost : ℝ := 36

/-- The discount rate applied to each book -/
def discount_rate : ℝ := 0.1

/-- The number of books to purchase after discount -/
def num_books_after_discount : ℕ := 3

/-- Theorem stating the total cost of three books after applying a 10% discount -/
theorem discounted_three_books_cost :
  let original_price := two_books_cost / 2
  let discounted_price := original_price * (1 - discount_rate)
  discounted_price * num_books_after_discount = 48.60 := by
  sorry

end discounted_three_books_cost_l2751_275135
