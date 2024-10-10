import Mathlib

namespace three_lines_theorem_l2771_277130

/-- A line in 3D space -/
structure Line3D where
  -- Define a line using two points
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ
  ne_points : point1 ≠ point2

/-- Three lines in 3D space -/
structure ThreeLines where
  line1 : Line3D
  line2 : Line3D
  line3 : Line3D

/-- Predicate to check if three lines are coplanar -/
def are_coplanar (lines : ThreeLines) : Prop :=
  sorry

/-- Predicate to check if two lines are skew -/
def are_skew (l1 l2 : Line3D) : Prop :=
  sorry

/-- Predicate to check if three lines intersect at a single point -/
def intersect_at_point (lines : ThreeLines) : Prop :=
  sorry

/-- Predicate to check if three lines are parallel -/
def are_parallel (lines : ThreeLines) : Prop :=
  sorry

/-- Theorem stating that three non-coplanar lines with no two being skew
    either intersect at a single point or are parallel -/
theorem three_lines_theorem (lines : ThreeLines) 
  (h1 : ¬ are_coplanar lines)
  (h2 : ¬ are_skew lines.line1 lines.line2)
  (h3 : ¬ are_skew lines.line1 lines.line3)
  (h4 : ¬ are_skew lines.line2 lines.line3) :
  intersect_at_point lines ∨ are_parallel lines :=
sorry

end three_lines_theorem_l2771_277130


namespace probability_same_fruit_choices_l2771_277101

/-- The number of fruit types available -/
def num_fruits : ℕ := 4

/-- The number of fruit types each student must choose -/
def num_choices : ℕ := 2

/-- The probability that two students choose the same two types of fruits -/
def probability_same_choice : ℚ := 1 / 6

/-- Theorem stating the probability of two students choosing the same fruits -/
theorem probability_same_fruit_choices :
  (Nat.choose num_fruits num_choices : ℚ) / ((Nat.choose num_fruits num_choices : ℚ) ^ 2) = probability_same_choice :=
sorry

end probability_same_fruit_choices_l2771_277101


namespace term1_and_term2_are_like_terms_l2771_277186

-- Define a structure for terms
structure Term where
  coefficient : ℚ
  x_power : ℕ
  y_power : ℕ

-- Define what it means for two terms to be like terms
def are_like_terms (t1 t2 : Term) : Prop :=
  t1.x_power = t2.x_power ∧ t1.y_power = t2.y_power

-- Define the two terms we're comparing
def term1 : Term := { coefficient := 4, x_power := 2, y_power := 1 }
def term2 : Term := { coefficient := -1, x_power := 2, y_power := 1 }

-- Theorem stating that term1 and term2 are like terms
theorem term1_and_term2_are_like_terms : are_like_terms term1 term2 := by
  sorry


end term1_and_term2_are_like_terms_l2771_277186


namespace total_cows_l2771_277184

theorem total_cows (cows_per_herd : ℕ) (num_herds : ℕ) (h1 : cows_per_herd = 40) (h2 : num_herds = 8) :
  cows_per_herd * num_herds = 320 := by
  sorry

end total_cows_l2771_277184


namespace girls_in_first_year_l2771_277182

theorem girls_in_first_year 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (boys_in_sample : ℕ) 
  (h1 : total_students = 2400) 
  (h2 : sample_size = 80) 
  (h3 : boys_in_sample = 42) : 
  ℕ := by
  sorry

#check girls_in_first_year

end girls_in_first_year_l2771_277182


namespace xy_zero_necessary_not_sufficient_l2771_277162

theorem xy_zero_necessary_not_sufficient (x y : ℝ) :
  (x^2 + y^2 = 0 → x * y = 0) ∧
  ∃ x y : ℝ, x * y = 0 ∧ x^2 + y^2 ≠ 0 :=
by sorry

end xy_zero_necessary_not_sufficient_l2771_277162


namespace vector_at_negative_one_l2771_277149

/-- A parameterized line in 3D space -/
structure ParameterizedLine where
  point_at_zero : ℝ × ℝ × ℝ
  point_at_one : ℝ × ℝ × ℝ

/-- The vector on the line at a given parameter value -/
def vector_at_t (line : ParameterizedLine) (t : ℝ) : ℝ × ℝ × ℝ :=
  let (x₀, y₀, z₀) := line.point_at_zero
  let (x₁, y₁, z₁) := line.point_at_one
  (x₀ + t * (x₁ - x₀), y₀ + t * (y₁ - y₀), z₀ + t * (z₁ - z₀))

theorem vector_at_negative_one (line : ParameterizedLine) 
  (h₀ : line.point_at_zero = (2, 6, 16))
  (h₁ : line.point_at_one = (1, 1, 8)) :
  vector_at_t line (-1) = (3, 11, 24) := by
  sorry

end vector_at_negative_one_l2771_277149


namespace eliana_steps_theorem_l2771_277185

/-- The number of steps Eliana walked on the first day -/
def first_day_steps : ℕ := 200 + 300

/-- The number of steps Eliana walked on the second day -/
def second_day_steps : ℕ := 2 * first_day_steps

/-- The additional steps Eliana walked on the third day -/
def third_day_additional_steps : ℕ := 100

/-- The total number of steps Eliana walked during the three days -/
def total_steps : ℕ := first_day_steps + second_day_steps + third_day_additional_steps

theorem eliana_steps_theorem : total_steps = 1600 := by
  sorry

end eliana_steps_theorem_l2771_277185


namespace rhombus_area_theorem_l2771_277153

/-- A rhombus with perpendicular bisecting diagonals -/
structure Rhombus where
  side_length : ℝ
  diagonal_difference : ℝ
  perpendicular_bisectors : Bool

/-- Calculate the area of a rhombus given its properties -/
def rhombus_area (r : Rhombus) : ℝ :=
  sorry

/-- Theorem: The area of a rhombus with side length √117 and diagonals differing by 8 units is 101 -/
theorem rhombus_area_theorem (r : Rhombus) 
    (h1 : r.side_length = Real.sqrt 117)
    (h2 : r.diagonal_difference = 8)
    (h3 : r.perpendicular_bisectors = true) : 
  rhombus_area r = 101 := by
  sorry

end rhombus_area_theorem_l2771_277153


namespace min_production_time_proof_l2771_277155

/-- Represents the production capacity of a factory --/
structure FactoryCapacity where
  typeI : ℝ  -- Units of Type I produced per day
  typeII : ℝ -- Units of Type II produced per day

/-- Represents the total order quantity --/
structure OrderQuantity where
  typeI : ℝ
  typeII : ℝ

/-- Calculates the minimum production time given factory capacities and order quantity --/
def minProductionTime (factoryA factoryB : FactoryCapacity) (order : OrderQuantity) : ℝ :=
  sorry

/-- Theorem stating the minimum production time for the given problem --/
theorem min_production_time_proof 
  (factoryA : FactoryCapacity)
  (factoryB : FactoryCapacity)
  (order : OrderQuantity)
  (h1 : factoryA.typeI = 30 ∧ factoryA.typeII = 20)
  (h2 : factoryB.typeI = 50 ∧ factoryB.typeII = 40)
  (h3 : order.typeI = 1500 ∧ order.typeII = 800) :
  minProductionTime factoryA factoryB order = 31.25 :=
sorry

end min_production_time_proof_l2771_277155


namespace pete_age_triple_son_l2771_277197

/-- 
Given:
- Pete's current age is 35
- Pete's son's current age is 9

Prove that in 4 years, Pete will be exactly three times older than his son.
-/
theorem pete_age_triple_son (pete_age : ℕ) (son_age : ℕ) : 
  pete_age = 35 → son_age = 9 → 
  ∃ (years : ℕ), years = 4 ∧ pete_age + years = 3 * (son_age + years) :=
by sorry

end pete_age_triple_son_l2771_277197


namespace mrs_sheridan_cats_l2771_277122

theorem mrs_sheridan_cats (initial_cats : ℕ) : 
  initial_cats + 14 = 31 → initial_cats = 17 := by
  sorry

end mrs_sheridan_cats_l2771_277122


namespace polynomial_real_root_condition_l2771_277105

theorem polynomial_real_root_condition (a : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 - x^2 + a^2*x + 1 = 0) ↔ (a ≤ -1 ∨ a ≥ 1) :=
by sorry

end polynomial_real_root_condition_l2771_277105


namespace jenny_house_worth_l2771_277123

/-- The current worth of Jenny's house -/
def house_worth : ℝ := 500000

/-- Jenny's property tax rate -/
def tax_rate : ℝ := 0.02

/-- The increase in house value due to the high-speed rail project -/
def value_increase : ℝ := 0.25

/-- The maximum amount Jenny can spend on property tax per year -/
def max_tax : ℝ := 15000

/-- The value of improvements Jenny can make to her house -/
def improvements : ℝ := 250000

theorem jenny_house_worth :
  tax_rate * (house_worth * (1 + value_increase) + improvements) = max_tax := by
  sorry

#check jenny_house_worth

end jenny_house_worth_l2771_277123


namespace sandy_token_ratio_l2771_277146

theorem sandy_token_ratio : 
  ∀ (total_tokens : ℕ) (num_siblings : ℕ) (extra_tokens : ℕ),
    total_tokens = 1000000 →
    num_siblings = 4 →
    extra_tokens = 375000 →
    ∃ (tokens_per_sibling : ℕ),
      tokens_per_sibling * num_siblings + (tokens_per_sibling + extra_tokens) = total_tokens ∧
      (tokens_per_sibling + extra_tokens) * 2 = total_tokens :=
by sorry

end sandy_token_ratio_l2771_277146


namespace quadratic_point_on_graph_l2771_277137

theorem quadratic_point_on_graph (a m : ℝ) (ha : a > 0) (hm : m ≠ 0) :
  (3 = -a * m^2 + 2 * a * m + 3) → m = 2 := by
  sorry

end quadratic_point_on_graph_l2771_277137


namespace unique_solution_l2771_277152

/-- Two functions f and g from ℝ to ℝ satisfying the given functional equation -/
def SatisfyEquation (f g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x^2 - g y) = (g x)^2 - y

/-- The theorem stating that the only functions satisfying the equation are the identity function -/
theorem unique_solution {f g : ℝ → ℝ} (h : SatisfyEquation f g) :
    (∀ x : ℝ, f x = x) ∧ (∀ x : ℝ, g x = x) := by
  sorry

end unique_solution_l2771_277152


namespace max_profit_at_180_l2771_277103

/-- The total cost function for a certain product -/
def total_cost (x : ℝ) : ℝ := 0.1 * x^2 - 11 * x + 3000

/-- The selling price per unit in ten thousand yuan -/
def selling_price : ℝ := 25

/-- The profit function -/
def profit (x : ℝ) : ℝ := selling_price * x - total_cost x

/-- Theorem: The production volume that maximizes profit is 180 units -/
theorem max_profit_at_180 : 
  ∃ (max_x : ℝ), (∀ x : ℝ, profit x ≤ profit max_x) ∧ max_x = 180 :=
sorry

end max_profit_at_180_l2771_277103


namespace constant_c_value_l2771_277168

theorem constant_c_value (b c : ℝ) : 
  (∀ x : ℝ, (x + 3) * (x + b) = x^2 + c*x + 12) → c = 7 := by
  sorry

end constant_c_value_l2771_277168


namespace tangent_slope_is_e_l2771_277118

/-- The exponential function -/
noncomputable def f (x : ℝ) : ℝ := Real.exp x

/-- A line passing through the origin -/
def line_through_origin (k : ℝ) (x : ℝ) : ℝ := k * x

/-- Tangent condition: The line touches the curve at exactly one point -/
def is_tangent (k : ℝ) : Prop :=
  ∃ x₀ : ℝ, 
    f x₀ = line_through_origin k x₀ ∧
    ∀ x ≠ x₀, f x ≠ line_through_origin k x

theorem tangent_slope_is_e :
  ∃ k : ℝ, is_tangent k ∧ k = Real.exp 1 :=
sorry

end tangent_slope_is_e_l2771_277118


namespace partition_of_positive_integers_l2771_277114

def nth_prime (n : ℕ) : ℕ := sorry

def count_primes (n : ℕ) : ℕ := sorry

def set_A : Set ℕ := {m | ∃ n : ℕ, n > 0 ∧ m = n + nth_prime n - 1}

def set_B : Set ℕ := {m | ∃ n : ℕ, n > 0 ∧ m = n + count_primes n}

theorem partition_of_positive_integers : 
  ∀ m : ℕ, m > 0 → (m ∈ set_A ∧ m ∉ set_B) ∨ (m ∉ set_A ∧ m ∈ set_B) :=
sorry

end partition_of_positive_integers_l2771_277114


namespace oak_grove_library_books_l2771_277136

/-- The number of books in Oak Grove's public library -/
def public_library_books : ℕ := 1986

/-- The number of books in Oak Grove's school libraries -/
def school_libraries_books : ℕ := 5106

/-- The total number of books in Oak Grove libraries -/
def total_books : ℕ := public_library_books + school_libraries_books

theorem oak_grove_library_books : total_books = 7092 := by
  sorry

end oak_grove_library_books_l2771_277136


namespace tom_solo_time_is_four_l2771_277165

/-- The time it takes Avery to build the wall alone, in hours -/
def avery_time : ℝ := 2

/-- The time Avery and Tom work together, in hours -/
def together_time : ℝ := 1

/-- The time it takes Tom to finish the wall after Avery leaves, in hours -/
def tom_finish_time : ℝ := 1

/-- The time it takes Tom to build the wall alone, in hours -/
def tom_solo_time : ℝ := 4

/-- Theorem stating that Tom's solo time is 4 hours -/
theorem tom_solo_time_is_four :
  (1 / avery_time + 1 / tom_solo_time) * together_time + 
  (1 / tom_solo_time) * tom_finish_time = 1 →
  tom_solo_time = 4 := by
sorry

end tom_solo_time_is_four_l2771_277165


namespace divisor_problem_l2771_277193

theorem divisor_problem (x d : ℝ) (h1 : x = 33) (h2 : x / d + 9 = 15) : d = 5.5 := by
  sorry

end divisor_problem_l2771_277193


namespace pets_count_l2771_277121

/-- The total number of pets owned by Teddy, Ben, and Dave -/
def total_pets (x y z a b c d e f : ℕ) : ℕ := x + y + z + a + b + c + d + e + f

/-- Theorem stating the total number of pets is 118 -/
theorem pets_count (x y z a b c d e f : ℕ) 
  (eq1 : x = 9)
  (eq2 : y = 8)
  (eq3 : z = 10)
  (eq4 : a = 21)
  (eq5 : b = 2 * y)
  (eq6 : c = z)
  (eq7 : d = x - 4)
  (eq8 : e = y + 13)
  (eq9 : f = 18) :
  total_pets x y z a b c d e f = 118 := by
  sorry


end pets_count_l2771_277121


namespace scientific_notation_of_132000000_l2771_277125

theorem scientific_notation_of_132000000 :
  (132000000 : ℝ) = 1.32 * (10 : ℝ) ^ 8 := by sorry

end scientific_notation_of_132000000_l2771_277125


namespace power_sum_seven_l2771_277124

theorem power_sum_seven (α₁ α₂ α₃ : ℂ) 
  (h1 : α₁ + α₂ + α₃ = 2)
  (h2 : α₁^2 + α₂^2 + α₃^2 = 6)
  (h3 : α₁^3 + α₂^3 + α₃^3 = 14) :
  α₁^7 + α₂^7 + α₃^7 = 46 := by
  sorry

end power_sum_seven_l2771_277124


namespace intersection_points_sum_greater_than_two_l2771_277113

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * log x - a * x^2 + (2 * a - 1) * x

theorem intersection_points_sum_greater_than_two (a t : ℝ) (x₁ x₂ : ℝ) :
  a ≤ 0 →
  -1 < t →
  t < 0 →
  x₁ < x₂ →
  f a x₁ = t →
  f a x₂ = t →
  x₁ + x₂ > 2 := by
  sorry

end intersection_points_sum_greater_than_two_l2771_277113


namespace abs_z_equals_10_l2771_277108

def z : ℂ := (3 + Complex.I)^2 * Complex.I

theorem abs_z_equals_10 : Complex.abs z = 10 := by sorry

end abs_z_equals_10_l2771_277108


namespace least_positive_integer_congruence_l2771_277144

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 5683 : ℤ) ≡ 420 [ZMOD 17] ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 5683 : ℤ) ≡ 420 [ZMOD 17] → x ≤ y :=
by
  use 7
  sorry

end least_positive_integer_congruence_l2771_277144


namespace puzzle_completion_l2771_277191

theorem puzzle_completion (P : ℝ) : 
  (P ≥ 0) →
  (P ≤ 1) →
  ((1 - P) * 0.8 * 0.7 * 1000 = 504) →
  (P = 0.1) := by
sorry

end puzzle_completion_l2771_277191


namespace cookie_problem_l2771_277190

theorem cookie_problem :
  ∃! N : ℕ, 0 < N ∧ N < 150 ∧ N % 13 = 5 ∧ N % 8 = 3 :=
by sorry

end cookie_problem_l2771_277190


namespace marco_card_trade_ratio_l2771_277188

theorem marco_card_trade_ratio : 
  ∀ (total_cards duplicates_traded new_cards : ℕ),
    total_cards = 500 →
    duplicates_traded = new_cards →
    new_cards = 25 →
    (duplicates_traded : ℚ) / (total_cards / 4 : ℚ) = 1 / 5 := by
  sorry

end marco_card_trade_ratio_l2771_277188


namespace three_card_selection_l2771_277163

/-- The number of cards in the special deck -/
def deck_size : ℕ := 60

/-- The number of cards to be picked -/
def cards_to_pick : ℕ := 3

/-- The number of ways to choose and order 3 different cards from a 60-card deck -/
def ways_to_pick : ℕ := 205320

/-- Theorem stating that the number of ways to choose and order 3 different cards
    from a 60-card deck is equal to 205320 -/
theorem three_card_selection :
  (deck_size * (deck_size - 1) * (deck_size - 2)) = ways_to_pick :=
by sorry

end three_card_selection_l2771_277163


namespace sixteen_not_valid_l2771_277176

/-- Represents a set of lines in a plane -/
structure LineSet where
  numLines : ℕ
  intersectionCount : ℕ

/-- Checks if a LineSet is valid according to the problem conditions -/
def isValidLineSet (ls : LineSet) : Prop :=
  ls.intersectionCount = 10 ∧
  ∃ (n k : ℕ), n > 1 ∧ k > 0 ∧ ls.numLines = n * k ∧ (n - 1) * k = ls.intersectionCount

/-- Theorem stating that 16 cannot be a valid number of lines in the set -/
theorem sixteen_not_valid : ¬ (∃ (ls : LineSet), ls.numLines = 16 ∧ isValidLineSet ls) := by
  sorry


end sixteen_not_valid_l2771_277176


namespace time_difference_per_question_l2771_277198

def english_questions : ℕ := 30
def math_questions : ℕ := 15
def english_time_hours : ℚ := 1
def math_time_hours : ℚ := (3/2)

def english_time_minutes : ℚ := english_time_hours * 60
def math_time_minutes : ℚ := math_time_hours * 60

def english_time_per_question : ℚ := english_time_minutes / english_questions
def math_time_per_question : ℚ := math_time_minutes / math_questions

theorem time_difference_per_question :
  math_time_per_question - english_time_per_question = 4 := by
  sorry

end time_difference_per_question_l2771_277198


namespace geometric_sequence_product_l2771_277100

/-- Given a geometric sequence of 10 terms, prove that if the sum of these terms is 18
    and the sum of their reciprocals is 6, then the product of these terms is (1/6)^55 -/
theorem geometric_sequence_product (a r : ℝ) (h1 : a ≠ 0) (h2 : r ≠ 0) (h3 : r ≠ 1) :
  (a * r * (r^10 - 1) / (r - 1) = 18) →
  (1 / (a * r) * (1 - 1/r^10) / (1 - 1/r) = 6) →
  (a * r)^55 = (1/6)^55 := by
sorry

end geometric_sequence_product_l2771_277100


namespace pythagorean_theorem_for_triples_scaled_right_triangle_is_pythagorean_l2771_277150

/-- Pythagorean numbers are positive integers that can be the lengths of the sides of a right triangle. -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem pythagorean_theorem_for_triples :
  ∀ (a b c : ℕ), isPythagoreanTriple a b c → a * a + b * b = c * c :=
sorry

theorem scaled_right_triangle_is_pythagorean :
  ∀ (a b c : ℕ), isPythagoreanTriple a b c → isPythagoreanTriple (2*a) (2*b) (2*c) :=
sorry

end pythagorean_theorem_for_triples_scaled_right_triangle_is_pythagorean_l2771_277150


namespace bacteria_growth_l2771_277145

/-- The number of cells after a given number of days, where the initial population
    doubles every two days. -/
def cell_population (initial_cells : ℕ) (days : ℕ) : ℕ :=
  initial_cells * 2^(days / 2)

/-- Theorem stating that given an initial population of 4 cells that double
    every two days, the number of cells after 10 days is 64. -/
theorem bacteria_growth : cell_population 4 10 = 64 := by
  sorry

end bacteria_growth_l2771_277145


namespace cubic_fraction_factorization_l2771_277129

theorem cubic_fraction_factorization (a b c : ℝ) :
  ((a^3 - b^3)^3 + (b^3 - c^3)^3 + (c^3 - a^3)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) = 
  (a^2 + a*b + b^2) * (b^2 + b*c + c^2) * (c^2 + c*a + a^2) :=
by sorry

end cubic_fraction_factorization_l2771_277129


namespace rationalize_denominator_l2771_277151

theorem rationalize_denominator : (14 : ℝ) / Real.sqrt 14 = Real.sqrt 14 := by
  sorry

end rationalize_denominator_l2771_277151


namespace unique_triplet_divisibility_l2771_277156

theorem unique_triplet_divisibility :
  ∃! (a b c : ℕ), 
    (∀ n : ℕ, (∀ p < 2015, Nat.Prime p → ¬(p ∣ n)) → 
      (n + c ∣ a^n + b^n + n)) ∧
    a = 1 ∧ b = 1 ∧ c = 2 := by
  sorry

end unique_triplet_divisibility_l2771_277156


namespace max_abs_cexp_minus_two_l2771_277120

-- Define the complex exponential function
noncomputable def cexp (x : ℝ) : ℂ := Complex.exp (Complex.I * x)

-- State Euler's formula
axiom euler_formula (x : ℝ) : cexp x = Complex.cos x + Complex.I * Complex.sin x

-- State the theorem
theorem max_abs_cexp_minus_two :
  ∃ (M : ℝ), M = 3 ∧ ∀ (x : ℝ), Complex.abs (cexp x - 2) ≤ M :=
sorry

end max_abs_cexp_minus_two_l2771_277120


namespace age_ratio_theorem_l2771_277196

/-- Represents the ages of John and Mary -/
structure Ages where
  john : ℕ
  mary : ℕ

/-- The conditions of the problem -/
def problem_conditions (a : Ages) : Prop :=
  (a.john - 5 = 2 * (a.mary - 5)) ∧ 
  (a.john - 12 = 3 * (a.mary - 12))

/-- The ratio condition we're looking for -/
def ratio_condition (a : Ages) (years : ℕ) : Prop :=
  3 * (a.mary + years) = 2 * (a.john + years)

/-- The main theorem -/
theorem age_ratio_theorem (a : Ages) :
  problem_conditions a → ∃ years : ℕ, years = 9 ∧ ratio_condition a years := by
  sorry


end age_ratio_theorem_l2771_277196


namespace multiplication_puzzle_l2771_277171

theorem multiplication_puzzle :
  (142857 * 5 = 714285) ∧ (142857 * 3 = 428571) := by
  sorry

end multiplication_puzzle_l2771_277171


namespace finger_multiplication_rule_l2771_277167

theorem finger_multiplication_rule (n : ℕ) (h : 1 ≤ n ∧ n ≤ 9) : 9 * n = 10 * (n - 1) + (10 - n) := by
  sorry

end finger_multiplication_rule_l2771_277167


namespace shifted_parabola_vertex_l2771_277174

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := -(x + 1)^2 + 4

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ := original_parabola (x - 1) - 2

-- Theorem statement
theorem shifted_parabola_vertex :
  ∃ (vertex_x vertex_y : ℝ),
    vertex_x = 0 ∧
    vertex_y = 2 ∧
    ∀ (x : ℝ), shifted_parabola x ≤ shifted_parabola vertex_x :=
by
  sorry

end shifted_parabola_vertex_l2771_277174


namespace quadratic_inequality_l2771_277173

/-- Quadratic trinomial with integer coefficients -/
structure QuadraticTrinomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluate a quadratic trinomial at a given x -/
def evaluate (q : QuadraticTrinomial) (x : ℝ) : ℝ :=
  (q.a : ℝ) * x^2 + (q.b : ℝ) * x + (q.c : ℝ)

/-- A quadratic trinomial is positive for all real x -/
def IsAlwaysPositive (q : QuadraticTrinomial) : Prop :=
  ∀ x : ℝ, evaluate q x > 0

theorem quadratic_inequality {f g : QuadraticTrinomial} 
  (hf : IsAlwaysPositive f) 
  (hg : IsAlwaysPositive g)
  (h : ∀ x : ℝ, evaluate f x / evaluate g x ≥ Real.sqrt 2) :
  ∀ x : ℝ, evaluate f x / evaluate g x > Real.sqrt 2 := by
  sorry

end quadratic_inequality_l2771_277173


namespace smallest_divisible_by_18_and_35_l2771_277159

theorem smallest_divisible_by_18_and_35 : 
  ∀ n : ℕ, n > 0 → (18 ∣ n) → (35 ∣ n) → n ≥ 630 :=
by sorry

end smallest_divisible_by_18_and_35_l2771_277159


namespace sector_central_angle_l2771_277142

theorem sector_central_angle (arc_length : Real) (area : Real) :
  arc_length = π → area = 2 * π → ∃ (r : Real) (α : Real),
    r > 0 ∧ α > 0 ∧ area = 1/2 * r * arc_length ∧ arc_length = r * α ∧ α = π/4 :=
by sorry

end sector_central_angle_l2771_277142


namespace donovan_lap_time_donovan_lap_time_is_45_l2771_277138

/-- Given two runners on a circular track, this theorem proves the lap time of the slower runner. -/
theorem donovan_lap_time (michael_lap_time : ℕ) (laps_to_pass : ℕ) : ℕ :=
  let michael_total_time := michael_lap_time * laps_to_pass
  let donovan_laps := laps_to_pass - 1
  michael_total_time / donovan_laps

/-- Proves that Donovan's lap time is 45 seconds given the problem conditions. -/
theorem donovan_lap_time_is_45 :
  donovan_lap_time 40 9 = 45 := by
  sorry

end donovan_lap_time_donovan_lap_time_is_45_l2771_277138


namespace inequality_properties_l2771_277194

theorem inequality_properties (a b c d : ℝ) :
  (a > b ∧ c > d → a + c > b + d) ∧
  (a > b ∧ b > 0 ∧ c < 0 → c / a > c / b) :=
by sorry

end inequality_properties_l2771_277194


namespace euler_family_mean_age_l2771_277109

theorem euler_family_mean_age :
  let ages : List ℕ := [6, 6, 6, 6, 10, 10, 16]
  (List.sum ages) / (List.length ages) = 60 / 7 := by
  sorry

end euler_family_mean_age_l2771_277109


namespace geometric_series_problem_l2771_277133

theorem geometric_series_problem (n : ℝ) : 
  let a₁ : ℝ := 18
  let r₁ : ℝ := 6 / 18
  let S₁ : ℝ := a₁ / (1 - r₁)
  let r₂ : ℝ := (6 + n) / 18
  let S₂ : ℝ := a₁ / (1 - r₂)
  S₂ = 5 * S₁ → n = 9.6 := by
sorry

end geometric_series_problem_l2771_277133


namespace carpet_for_room_l2771_277166

/-- Calculates the minimum number of whole square yards of carpet needed for a rectangular room with overlap -/
def carpet_needed (length width overlap : ℕ) : ℕ :=
  let adjusted_length := length + 2 * overlap
  let adjusted_width := width + 2 * overlap
  let area := adjusted_length * adjusted_width
  (area + 8) / 9  -- Adding 8 before division by 9 to round up

theorem carpet_for_room : carpet_needed 15 9 1 = 21 := by
  sorry

end carpet_for_room_l2771_277166


namespace sequence_property_initial_condition_main_theorem_l2771_277178

def sequence_a (n : ℕ) : ℝ :=
  sorry

theorem sequence_property (n : ℕ) :
  (2 * n + 3 : ℝ) * sequence_a (n + 1) - (2 * n + 5 : ℝ) * sequence_a n =
  (2 * n + 3 : ℝ) * (2 * n + 5 : ℝ) * Real.log (1 + 1 / (n : ℝ)) :=
  sorry

theorem initial_condition : sequence_a 1 = 5 :=
  sorry

theorem main_theorem (n : ℕ) (hn : n > 0) :
  sequence_a n / (2 * n + 3 : ℝ) = 1 + Real.log n :=
  sorry

end sequence_property_initial_condition_main_theorem_l2771_277178


namespace right_triangle_345_l2771_277127

/-- A triangle with side lengths 3, 4, and 5 is a right triangle. -/
theorem right_triangle_345 :
  ∀ (a b c : ℝ), a = 3 ∧ b = 4 ∧ c = 5 →
  a^2 + b^2 = c^2 := by
  sorry

end right_triangle_345_l2771_277127


namespace symmetric_points_sum_l2771_277140

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = -y₂

/-- Given that point A(1, a) and point B(b, -2) are symmetric with respect to the origin O, prove that a + b = 1 -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_origin 1 a b (-2)) : a + b = 1 := by
  sorry

end symmetric_points_sum_l2771_277140


namespace total_watermelon_seeds_l2771_277106

/-- The number of watermelon seeds each person has -/
structure WatermelonSeeds where
  bom : ℕ
  gwi : ℕ
  yeon : ℕ
  eun : ℕ

/-- Given conditions about watermelon seeds -/
def watermelon_seed_conditions (w : WatermelonSeeds) : Prop :=
  w.yeon = 3 * w.gwi ∧
  w.gwi = w.bom + 40 ∧
  w.eun = 2 * w.gwi ∧
  w.bom = 300

/-- Theorem stating the total number of watermelon seeds -/
theorem total_watermelon_seeds (w : WatermelonSeeds) 
  (h : watermelon_seed_conditions w) : 
  w.bom + w.gwi + w.yeon + w.eun = 2340 := by
  sorry

end total_watermelon_seeds_l2771_277106


namespace picnic_theorem_l2771_277199

-- Define the propositions
variable (P : Prop) -- "The picnic on Sunday will be held"
variable (Q : Prop) -- "The weather is fair on Sunday"

-- State the given condition
axiom given_statement : (¬P → ¬Q)

-- State the theorem to be proved
theorem picnic_theorem : Q → P := by sorry

end picnic_theorem_l2771_277199


namespace existence_of_monotonic_tail_l2771_277115

def IsMonotonicSegment (a : ℕ → ℝ) (i m : ℕ) : Prop :=
  (∀ j ∈ Finset.range (m - 1), a (i + j) < a (i + j + 1)) ∨
  (∀ j ∈ Finset.range (m - 1), a (i + j) > a (i + j + 1))

theorem existence_of_monotonic_tail
  (a : ℕ → ℝ)
  (distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (monotonic_segment : ∀ k, ∃ i m, k ∈ Finset.range m ∧ IsMonotonicSegment a i (k + 1)) :
  ∃ N, (∀ i j, N ≤ i → i < j → a i < a j) ∨ (∀ i j, N ≤ i → i < j → a i > a j) :=
sorry

end existence_of_monotonic_tail_l2771_277115


namespace geometric_sequence_solution_l2771_277183

/-- A geometric sequence {a_n} satisfying given conditions -/
def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (q : ℚ), ∀ (k : ℕ), a (k + 1) = q * a k

theorem geometric_sequence_solution (a : ℕ → ℚ) :
  geometric_sequence a →
  a 3 + a 6 = 36 →
  a 4 + a 7 = 18 →
  (∃ n : ℕ, a n = 1/2) →
  ∃ n : ℕ, a n = 1/2 ∧ n = 9 :=
sorry

end geometric_sequence_solution_l2771_277183


namespace gcd_90_210_l2771_277139

theorem gcd_90_210 : Nat.gcd 90 210 = 30 := by
  sorry

end gcd_90_210_l2771_277139


namespace fractional_linear_transformation_cross_ratio_l2771_277132

theorem fractional_linear_transformation_cross_ratio 
  (a b c d : ℝ) (h : a * d - b * c ≠ 0)
  (x₁ x₂ x₃ x₄ : ℝ) :
  let y : ℝ → ℝ := λ x => (a * x + b) / (c * x + d)
  let y₁ := y x₁
  let y₂ := y x₂
  let y₃ := y x₃
  let y₄ := y x₄
  (y₃ - y₁) / (y₃ - y₂) / ((y₄ - y₁) / (y₄ - y₂)) = 
  (x₃ - x₁) / (x₃ - x₂) / ((x₄ - x₁) / (x₄ - x₂)) :=
by sorry

end fractional_linear_transformation_cross_ratio_l2771_277132


namespace vertical_line_condition_l2771_277170

/-- Given two points A and B, if the line AB has an angle of inclination of 90°, then a = 0 -/
theorem vertical_line_condition (a : ℝ) : 
  let A : ℝ × ℝ := (1 + a, 2 * a)
  let B : ℝ × ℝ := (1 - a, 3)
  (A.1 = B.1) →  -- This condition represents a vertical line (90° inclination)
  a = 0 := by
  sorry

end vertical_line_condition_l2771_277170


namespace larger_number_problem_l2771_277135

theorem larger_number_problem (x y : ℝ) : 
  y = x + 10 →  -- One number exceeds another by 10
  x = y / 2 →   -- The smaller number is half the larger number
  x + y = 34 →  -- Their sum is 34
  y = 20        -- The larger number is 20
:= by sorry

end larger_number_problem_l2771_277135


namespace chess_match_probability_l2771_277134

theorem chess_match_probability (p_win p_draw : ℝ) 
  (h1 : p_win = 0.4) 
  (h2 : p_draw = 0.2) : 
  p_win + p_draw = 0.6 := by
sorry

end chess_match_probability_l2771_277134


namespace problem_one_problem_two_l2771_277157

-- Problem 1
theorem problem_one : (Real.sqrt 3)^2 + |1 - Real.sqrt 3| + ((-27 : ℝ)^(1/3)) = Real.sqrt 3 - 1 := by
  sorry

-- Problem 2
theorem problem_two : (Real.sqrt 12 - Real.sqrt (1/3)) * Real.sqrt 6 = 5 * Real.sqrt 2 := by
  sorry

end problem_one_problem_two_l2771_277157


namespace problem_solution_l2771_277187

theorem problem_solution (x y : ℝ) (hx_pos : x > 0) :
  (2/3 : ℝ) * x = (144/216 : ℝ) * (1/x) ∧ y * (1/x) = Real.sqrt x → x = 1 ∧ y = 1 := by
  sorry

end problem_solution_l2771_277187


namespace four_square_product_l2771_277119

theorem four_square_product (p q r s p₁ q₁ r₁ s₁ : ℝ) :
  ∃ A B C D : ℝ, (p^2 + q^2 + r^2 + s^2) * (p₁^2 + q₁^2 + r₁^2 + s₁^2) = A^2 + B^2 + C^2 + D^2 := by
  sorry

end four_square_product_l2771_277119


namespace ball_bounce_distance_l2771_277147

/-- Calculates the total distance traveled by a bouncing ball -/
def totalDistance (initialHeight : ℝ) (bounceRatio : ℝ) (numBounces : ℕ) : ℝ :=
  let bounceSequence := (List.range numBounces).map (fun n => initialHeight * bounceRatio^n)
  let ascents := bounceSequence.sum
  let descents := initialHeight + (List.take (numBounces - 1) bounceSequence).sum
  ascents + descents

/-- The problem statement -/
theorem ball_bounce_distance :
  ∃ (d : ℝ), abs (totalDistance 20 (2/3) 4 - d) < 1 ∧ Int.floor d = 68 := by
  sorry


end ball_bounce_distance_l2771_277147


namespace unique_a_value_l2771_277107

theorem unique_a_value (a : ℝ) : 3 ∈ ({1, a, a - 2} : Set ℝ) → a = 5 := by
  sorry

end unique_a_value_l2771_277107


namespace sum_of_cubes_zero_l2771_277179

theorem sum_of_cubes_zero (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h_sum_squares : a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 0) :
  a / (b - c)^3 + b / (c - a)^3 + c / (a - b)^3 = 0 := by
  sorry

end sum_of_cubes_zero_l2771_277179


namespace f_equal_range_l2771_277128

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then (1/2) * x + (3/2) else Real.log x

theorem f_equal_range (m n : ℝ) (h1 : m < n) (h2 : f m = f n) :
  n - m ∈ Set.Icc (5 - 2 * Real.log 2) (Real.exp 2 - 1) :=
sorry

end f_equal_range_l2771_277128


namespace complement_union_M_N_l2771_277141

universe u

def U : Finset ℕ := {1,2,3,4,5}
def M : Finset ℕ := {1,2}
def N : Finset ℕ := {3,4}

theorem complement_union_M_N : (U \ (M ∪ N)) = {5} := by sorry

end complement_union_M_N_l2771_277141


namespace files_remaining_l2771_277177

theorem files_remaining (initial_files initial_apps final_apps files_deleted : ℕ) :
  initial_files = 24 →
  initial_apps = 13 →
  final_apps = 17 →
  files_deleted = 3 →
  initial_files - (final_apps - initial_apps) - files_deleted = 17 := by
  sorry

end files_remaining_l2771_277177


namespace total_leaves_calculation_l2771_277148

/-- Calculates the total number of leaves falling from cherry and maple trees -/
def total_leaves (initial_cherry : ℕ) (initial_maple : ℕ) 
                 (cherry_ratio : ℕ) (maple_ratio : ℕ) 
                 (cherry_leaves : ℕ) (maple_leaves : ℕ) : ℕ :=
  (initial_cherry * cherry_ratio * cherry_leaves) + 
  (initial_maple * maple_ratio * maple_leaves)

/-- Theorem stating that the total number of leaves is 3650 -/
theorem total_leaves_calculation : 
  total_leaves 7 5 2 3 100 150 = 3650 := by
  sorry

#eval total_leaves 7 5 2 3 100 150

end total_leaves_calculation_l2771_277148


namespace binomial_expectation_l2771_277112

/-- The number of trials -/
def n : ℕ := 3

/-- The probability of drawing a red ball -/
def p : ℚ := 3/5

/-- The expected value of a binomial distribution -/
def expected_value (n : ℕ) (p : ℚ) : ℚ := n * p

theorem binomial_expectation :
  expected_value n p = 9/5 := by sorry

end binomial_expectation_l2771_277112


namespace sum_and_product_of_averages_l2771_277195

def avg1 : ℚ := (0 + 100) / 2

def avg2 : ℚ := (0 + 50) / 2

def avg3 : ℚ := 560 / 8

theorem sum_and_product_of_averages :
  avg1 + avg2 + avg3 = 145 ∧ avg1 * avg2 * avg3 = 87500 := by sorry

end sum_and_product_of_averages_l2771_277195


namespace odd_numbers_sum_product_l2771_277164

theorem odd_numbers_sum_product (n : ℕ) (odds : Finset ℕ) (a b c : ℕ) : 
  n = 1997 →
  odds.card = n →
  (∀ x ∈ odds, Odd x) →
  (odds.sum id = odds.prod id) →
  a ∈ odds ∧ b ∈ odds ∧ c ∈ odds →
  a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 →
  a.Prime ∧ b.Prime ∧ c.Prime →
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (a = 5 ∧ b = 7 ∧ c = 59) ∨ (a = 5 ∧ b = 59 ∧ c = 7) ∨ 
  (a = 7 ∧ b = 5 ∧ c = 59) ∨ (a = 7 ∧ b = 59 ∧ c = 5) ∨ 
  (a = 59 ∧ b = 5 ∧ c = 7) ∨ (a = 59 ∧ b = 7 ∧ c = 5) :=
by sorry


end odd_numbers_sum_product_l2771_277164


namespace exponent_of_nine_in_nine_to_seven_l2771_277102

theorem exponent_of_nine_in_nine_to_seven (h : ∀ y : ℕ, y > 14 → ¬(3^y ∣ 9^7)) :
  ∃ n : ℕ, 9^7 = 9^n ∧ n = 7 :=
by sorry

end exponent_of_nine_in_nine_to_seven_l2771_277102


namespace y_coordinate_order_l2771_277172

/-- A quadratic function passing through three specific points -/
def quadratic_function (c : ℝ) (x : ℝ) : ℝ := x^2 - 6*x + c

/-- The y-coordinate of point A -/
def y₁ (c : ℝ) : ℝ := quadratic_function c (-1)

/-- The y-coordinate of point B -/
def y₂ (c : ℝ) : ℝ := quadratic_function c 2

/-- The y-coordinate of point C -/
def y₃ (c : ℝ) : ℝ := quadratic_function c 5

/-- Theorem stating the order of y-coordinates -/
theorem y_coordinate_order (c : ℝ) : y₁ c > y₃ c ∧ y₃ c > y₂ c := by
  sorry

end y_coordinate_order_l2771_277172


namespace right_triangle_third_side_l2771_277143

theorem right_triangle_third_side (a b c : ℝ) : 
  a = 4 ∧ b = 5 → 
  (a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2) → 
  c = 3 ∨ c = Real.sqrt 41 :=
by sorry

end right_triangle_third_side_l2771_277143


namespace rectangle_width_equals_eight_l2771_277189

theorem rectangle_width_equals_eight (square_side : ℝ) (rect_length : ℝ) (rect_width : ℝ)
  (h1 : square_side = 4)
  (h2 : rect_length = 2)
  (h3 : square_side * square_side = rect_length * rect_width) :
  rect_width = 8 := by
  sorry

end rectangle_width_equals_eight_l2771_277189


namespace distance_focus_to_line_l2771_277161

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point A (directrix intersection with x-axis)
def A : ℝ × ℝ := (-1, 0)

-- Define the focus F
def F : ℝ × ℝ := (1, 0)

-- Define the line l
def line_l (x y : ℝ) : Prop := y = Real.sqrt 3 * (x + 1)

-- State the theorem
theorem distance_focus_to_line :
  let d := Real.sqrt 3
  ∃ (x y : ℝ), parabola x y ∧ 
               line_l (A.1) (A.2) ∧
               (F.1 - x)^2 + (F.2 - y)^2 = d^2 :=
sorry

end distance_focus_to_line_l2771_277161


namespace games_within_division_is_16_l2771_277169

/-- Represents a baseball league with two divisions -/
structure BaseballLeague where
  /-- Number of games played against each team in the same division -/
  n : ℕ
  /-- Number of games played against each team in the other division -/
  m : ℕ
  /-- n is greater than 3m -/
  n_gt_3m : n > 3 * m
  /-- m is greater than 6 -/
  m_gt_6 : m > 6
  /-- Total number of games each team plays is 96 -/
  total_games : 4 * n + 5 * m = 96

/-- The number of games a team plays within its own division -/
def games_within_division (league : BaseballLeague) : ℕ := 4 * league.n

/-- Theorem stating that the number of games played within a team's division is 16 -/
theorem games_within_division_is_16 (league : BaseballLeague) :
  games_within_division league = 16 := by
  sorry

end games_within_division_is_16_l2771_277169


namespace shopping_expenses_total_l2771_277110

/-- Represents the shopping expenses of Lisa and Carly -/
def ShoppingExpenses (lisa_tshirt : ℝ) : Prop :=
  let lisa_jeans := lisa_tshirt / 2
  let lisa_coat := lisa_tshirt * 2
  let shoe_cost := lisa_jeans * 3
  let carly_tshirt := lisa_tshirt / 4
  let carly_jeans := lisa_jeans * 3
  let carly_coat := lisa_coat / 2
  let carly_dress := shoe_cost * 2
  let lisa_total := lisa_tshirt + lisa_jeans + lisa_coat + shoe_cost
  let carly_total := carly_tshirt + carly_jeans + carly_coat + shoe_cost + carly_dress
  lisa_tshirt = 40 ∧ lisa_total + carly_total = 490

/-- Theorem stating that the total amount spent by Lisa and Carly is $490 -/
theorem shopping_expenses_total : ShoppingExpenses 40 := by
  sorry

end shopping_expenses_total_l2771_277110


namespace abc_value_l2771_277154

theorem abc_value (a b c : ℝ) 
  (sum_condition : a + b + c = 1)
  (sum_squares : a^2 + b^2 + c^2 = 2)
  (sum_cubes : a^3 + b^3 + c^3 = 3) :
  a * b * c = 1/6 := by
  sorry

end abc_value_l2771_277154


namespace polynomial_simplification_l2771_277126

/-- Simplification of a polynomial expression -/
theorem polynomial_simplification (x : ℝ) :
  3 * x + 10 * x^2 + 5 * x^3 + 15 - (7 - 3 * x - 10 * x^2 - 5 * x^3) =
  10 * x^3 + 20 * x^2 + 6 * x + 8 := by
  sorry

end polynomial_simplification_l2771_277126


namespace bicycle_speed_problem_l2771_277117

/-- Proves that given a distance of 12 km, if person A's speed is 1.2 times person B's speed,
    and A arrives 1/6 hour earlier than B, then B's speed is 12 km/h. -/
theorem bicycle_speed_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) 
    (h1 : distance = 12)
    (h2 : speed_ratio = 1.2)
    (h3 : time_difference = 1/6) : 
  let speed_B := distance / (distance / (speed_ratio * (distance / time_difference)) + time_difference)
  speed_B = 12 := by
  sorry


end bicycle_speed_problem_l2771_277117


namespace point_position_on_line_l2771_277181

/-- Given points on a line, prove the position of a point P satisfying a ratio condition -/
theorem point_position_on_line (a b c d e : ℝ) :
  ∀ (O A B C D E P : ℝ),
    O < A ∧ A < B ∧ B < C ∧ C < D ∧  -- Points are ordered on the line
    A - O = 2 * a ∧                  -- OA = 2a
    B - O = b ∧                      -- OB = b
    C - O = 3 * c ∧                  -- OC = 3c
    D - O = d ∧                      -- OD = d
    E - O = e ∧                      -- OE = e
    B ≤ P ∧ P ≤ C ∧                  -- P is between B and C
    (A - P) * (P - E) = (B - P) * (P - C) →  -- AP:PE = BP:PC
  P - O = (b * e - 6 * a * c) / (2 * a + 3 * c - b - e) :=
by sorry

end point_position_on_line_l2771_277181


namespace slower_walking_speed_l2771_277160

/-- Proves that the slower walking speed is 10 km/hr given the conditions of the problem -/
theorem slower_walking_speed 
  (actual_distance : ℝ) 
  (faster_speed : ℝ) 
  (additional_distance : ℝ) 
  (h1 : actual_distance = 13.333333333333332)
  (h2 : faster_speed = 25)
  (h3 : additional_distance = 20)
  : ∃ (v : ℝ), 
    actual_distance / v = (actual_distance + additional_distance) / faster_speed ∧ 
    v = 10 := by
  sorry

end slower_walking_speed_l2771_277160


namespace line_direction_vector_l2771_277192

/-- The direction vector of a parameterized line -/
def direction_vector (line : ℝ → ℝ × ℝ) : ℝ × ℝ := sorry

theorem line_direction_vector :
  let line (t : ℝ) : ℝ × ℝ := (2, 0) + t • d
  let d : ℝ × ℝ := direction_vector line
  let y (x : ℝ) : ℝ := (5 * x - 7) / 6
  ∀ x ≥ 2, (x - 2) ^ 2 + (y x) ^ 2 = t ^ 2 →
  d = (6 / Real.sqrt 61, 5 / Real.sqrt 61) :=
by sorry

end line_direction_vector_l2771_277192


namespace substitution_result_l2771_277158

theorem substitution_result (x y : ℝ) :
  y = 2 * x + 1 ∧ 5 * x - 2 * y = 7 →
  5 * x - 4 * x - 2 = 7 :=
by sorry

end substitution_result_l2771_277158


namespace right_triangle_sin_z_l2771_277180

theorem right_triangle_sin_z (X Y Z : ℝ) : 
  X + Y + Z = π →
  X = π / 2 →
  Real.cos Y = 3 / 5 →
  Real.sin Z = 3 / 5 := by
  sorry

end right_triangle_sin_z_l2771_277180


namespace regular_octagon_interior_angle_measure_l2771_277116

/-- The measure of each interior angle of a regular octagon -/
def regular_octagon_interior_angle : ℝ := 135

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- Formula for the sum of interior angles of a polygon with n sides -/
def polygon_interior_angle_sum (n : ℕ) : ℝ := (n - 2) * 180

theorem regular_octagon_interior_angle_measure :
  regular_octagon_interior_angle = polygon_interior_angle_sum octagon_sides / octagon_sides := by
  sorry

end regular_octagon_interior_angle_measure_l2771_277116


namespace rectangular_field_shortcut_l2771_277111

theorem rectangular_field_shortcut (x y : ℝ) (hxy : 0 < x ∧ x < y) :
  x + y - Real.sqrt (x^2 + y^2) = (1/3) * y → x / y = 5/12 := by
  sorry

end rectangular_field_shortcut_l2771_277111


namespace dogs_groomed_l2771_277104

/-- Proves that the number of dogs groomed is 5, given the grooming times for dogs and cats,
    and the total time spent grooming dogs and 3 cats. -/
theorem dogs_groomed (dog_time : ℝ) (cat_time : ℝ) (total_time : ℝ) :
  dog_time = 2.5 →
  cat_time = 0.5 →
  total_time = 840 / 60 →
  (dog_time * ⌊(total_time - 3 * cat_time) / dog_time⌋ + 3 * cat_time) = total_time →
  ⌊(total_time - 3 * cat_time) / dog_time⌋ = 5 := by
  sorry

end dogs_groomed_l2771_277104


namespace seating_theorem_l2771_277131

/-- The number of seats in the row -/
def n : ℕ := 7

/-- The number of people to be seated -/
def k : ℕ := 2

/-- The number of different seating arrangements for two people in n seats
    with at least one empty seat between them -/
def seating_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  (n.factorial / ((n - k).factorial * k.factorial)) - ((n - 1).factorial / ((n - k - 1).factorial * k.factorial))

theorem seating_theorem : seating_arrangements n k = 30 := by
  sorry

end seating_theorem_l2771_277131


namespace modulus_of_z_l2771_277175

theorem modulus_of_z (i z : ℂ) (hi : i * i = -1) (hz : i * z = 3 + 4 * i) : Complex.abs z = 5 := by
  sorry

end modulus_of_z_l2771_277175
