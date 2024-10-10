import Mathlib

namespace function_inequality_solution_l154_15435

/-- Given a real number q with |q| < 1 and q ≠ 0, there exists a function f and a non-negative function g
    satisfying the given conditions. -/
theorem function_inequality_solution (q : ℝ) (hq1 : |q| < 1) (hq2 : q ≠ 0) :
  ∃ (f g : ℝ → ℝ) (a : ℕ → ℝ),
    (∀ x, g x ≥ 0) ∧
    (∀ x, f x = (1 - q * x) * f (q * x) + g x) ∧
    (∀ x, f x = ∑' i, a i * x^i) ∧
    (∀ k, k > 0 → a k = (a (k-1) * q^k - (1 / k.factorial) * (deriv^[k] g) 0) / (q^k - 1)) :=
by sorry

end function_inequality_solution_l154_15435


namespace total_cost_is_540_l154_15486

def cherry_price : ℝ := 5
def olive_price : ℝ := 7
def bag_count : ℕ := 50
def discount_rate : ℝ := 0.1

def discounted_price (original_price : ℝ) : ℝ :=
  original_price * (1 - discount_rate)

def total_cost : ℝ :=
  bag_count * (discounted_price cherry_price + discounted_price olive_price)

theorem total_cost_is_540 : total_cost = 540 := by
  sorry

end total_cost_is_540_l154_15486


namespace volume_ratio_l154_15455

/-- A right rectangular prism with edge lengths 2, 3, and 5 -/
structure Prism where
  length : ℝ := 2
  width : ℝ := 3
  height : ℝ := 5

/-- The set of points within distance r of the prism -/
def S (B : Prism) (r : ℝ) : Set (ℝ × ℝ × ℝ) := sorry

/-- The volume of S(r) -/
def volume_S (B : Prism) (r : ℝ) (a b c d : ℝ) : ℝ :=
  a * r^3 + b * r^2 + c * r + d

theorem volume_ratio (B : Prism) (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  b * c / (a * d) = 15.5 := by
  sorry

end volume_ratio_l154_15455


namespace car_average_speed_l154_15400

/-- Given a car's speed for two consecutive hours, calculate its average speed. -/
theorem car_average_speed (speed1 speed2 : ℝ) (h1 : speed1 = 100) (h2 : speed2 = 30) :
  (speed1 + speed2) / 2 = 65 := by
  sorry

end car_average_speed_l154_15400


namespace calculation_proof_l154_15489

theorem calculation_proof : (8.036 / 0.04) * (1.5 / 0.03) = 10045 := by
  sorry

end calculation_proof_l154_15489


namespace effervesces_arrangements_l154_15499

def word := "EFFERVESCES"

/-- Number of ways to arrange letters in a line with no adjacent E's -/
def linear_arrangements (w : String) : ℕ := sorry

/-- Number of ways to arrange letters in a circle with no adjacent E's -/
def circular_arrangements (w : String) : ℕ := sorry

/-- No two E's are adjacent in the arrangement -/
def no_adjacent_es (arrangement : List Char) : Prop := sorry

/-- The arrangement preserves the letter count of the original word -/
def preserves_letter_count (w : String) (arrangement : List Char) : Prop := sorry

theorem effervesces_arrangements :
  (linear_arrangements word = 88200) ∧
  (circular_arrangements word = 6300) :=
by sorry

end effervesces_arrangements_l154_15499


namespace max_a_value_l154_15409

theorem max_a_value (a : ℝ) : 
  (∀ x : ℝ, x ≥ 0 → Real.exp x + Real.sin x - 2*x ≥ a*x^2 + 1) → 
  a ≤ 1/2 :=
sorry

end max_a_value_l154_15409


namespace orange_calorie_distribution_l154_15454

theorem orange_calorie_distribution 
  (num_oranges : ℕ) 
  (pieces_per_orange : ℕ) 
  (num_people : ℕ) 
  (calories_per_orange : ℕ) 
  (h1 : num_oranges = 5)
  (h2 : pieces_per_orange = 8)
  (h3 : num_people = 4)
  (h4 : calories_per_orange = 80) :
  (num_oranges * calories_per_orange) / (num_people) = 100 := by
  sorry

end orange_calorie_distribution_l154_15454


namespace no_integer_solutions_l154_15445

theorem no_integer_solutions : 
  ¬∃ (x y z : ℤ), 
    (x^2 - 2*x*y + 3*y^2 - z^2 = 17) ∧ 
    (-x^2 + 4*y*z + z^2 = 28) ∧ 
    (x^2 + 2*x*y + 5*z^2 = 42) :=
by sorry

end no_integer_solutions_l154_15445


namespace min_value_implies_a_inequality_implies_m_range_l154_15424

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 2*a| + |x - 3*a|

-- Theorem 1
theorem min_value_implies_a (a : ℝ) :
  (∀ x, f x a ≥ 2) ∧ (∃ x, f x a = 2) → a = 2 ∨ a = -2 :=
sorry

-- Theorem 2
theorem inequality_implies_m_range (m : ℝ) :
  (∀ x, ∃ a ∈ Set.Icc (-2) 2, m^2 - |m| - f x a < 0) →
  -1 < m ∧ m < 2 :=
sorry

end min_value_implies_a_inequality_implies_m_range_l154_15424


namespace tangent_two_implications_l154_15460

open Real

theorem tangent_two_implications (α : ℝ) (h : tan α = 2) :
  (sin α + 2 * cos α) / (4 * cos α - sin α) = 2 ∧
  Real.sqrt 2 * sin (2 * α + π / 4) + 1 = 6 / 5 := by
  sorry

end tangent_two_implications_l154_15460


namespace carina_coffee_amount_l154_15417

/-- The number of 10-ounce packages of coffee -/
def num_10oz_packages : ℕ := 4

/-- The number of 5-ounce packages of coffee -/
def num_5oz_packages : ℕ := num_10oz_packages + 2

/-- The total amount of coffee in ounces -/
def total_coffee : ℕ := num_10oz_packages * 10 + num_5oz_packages * 5

theorem carina_coffee_amount : total_coffee = 70 := by
  sorry

end carina_coffee_amount_l154_15417


namespace cubic_equation_sum_l154_15440

theorem cubic_equation_sum (r s t : ℝ) : 
  r^3 - 4*r^2 + 4*r = 6 →
  s^3 - 4*s^2 + 4*s = 6 →
  t^3 - 4*t^2 + 4*t = 6 →
  r*s/t + s*t/r + t*r/s = -16/3 := by
  sorry

end cubic_equation_sum_l154_15440


namespace arithmetic_series_sum_l154_15474

/-- Sum of an arithmetic series -/
def arithmetic_sum (a₁ : ℚ) (aₙ : ℚ) (d : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- The sum of the arithmetic series with first term 22, last term 73, and common difference 3/7 is 5700 -/
theorem arithmetic_series_sum :
  arithmetic_sum 22 73 (3/7) = 5700 := by
  sorry

end arithmetic_series_sum_l154_15474


namespace target_hit_probability_l154_15423

theorem target_hit_probability 
  (prob_A prob_B prob_C : ℚ)
  (h_A : prob_A = 1/2)
  (h_B : prob_B = 1/3)
  (h_C : prob_C = 1/4) :
  1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C) = 3/4 :=
by sorry

end target_hit_probability_l154_15423


namespace equation_solutions_l154_15436

theorem equation_solutions :
  (∃ x : ℝ, 3 * x + 7 = 32 - 2 * x ∧ x = 5) ∧
  (∃ x : ℝ, (2 * x - 3) / 5 = (3 * x - 1) / 2 + 1 ∧ x = -1) := by
  sorry

end equation_solutions_l154_15436


namespace product_ABCD_eq_one_l154_15404

/-- Given A, B, C, and D as defined, prove that their product is 1 -/
theorem product_ABCD_eq_one (A B C D : ℝ) 
  (hA : A = Real.sqrt 2008 + Real.sqrt 2009)
  (hB : B = -(Real.sqrt 2008) - Real.sqrt 2009)
  (hC : C = Real.sqrt 2008 - Real.sqrt 2009)
  (hD : D = Real.sqrt 2009 - Real.sqrt 2008) : 
  A * B * C * D = 1 := by
  sorry


end product_ABCD_eq_one_l154_15404


namespace slope_values_theorem_l154_15458

def valid_slopes : List ℕ := [81, 192, 399, 501, 1008, 2019]

def intersects_parabola (k : ℝ) : Prop :=
  ∃ x₁ x₂ : ℤ, x₁ ≠ x₂ ∧ x₁^2 = k * x₁ + 2020 ∧ x₂^2 = k * x₂ + 2020

theorem slope_values_theorem :
  ∀ k : ℝ, k > 0 → intersects_parabola k → k ∈ valid_slopes.map (λ x => x : ℕ → ℝ) := by
  sorry

end slope_values_theorem_l154_15458


namespace sum_in_base5_l154_15485

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem sum_in_base5 : toBase5 (45 + 78) = [4, 4, 3] := by
  sorry

end sum_in_base5_l154_15485


namespace xiaoxi_has_largest_result_l154_15431

def start_number : ℕ := 8

def laura_result (n : ℕ) : ℕ := ((n - 2) * 3) + 3

def navin_result (n : ℕ) : ℕ := (n * 3 - 2) + 3

def xiaoxi_result (n : ℕ) : ℕ := ((n - 2) + 3) * 3

theorem xiaoxi_has_largest_result :
  xiaoxi_result start_number > laura_result start_number ∧
  xiaoxi_result start_number > navin_result start_number :=
by sorry

end xiaoxi_has_largest_result_l154_15431


namespace tan_105_degrees_l154_15410

theorem tan_105_degrees : Real.tan (105 * π / 180) = 2 + Real.sqrt 3 := by
  sorry

end tan_105_degrees_l154_15410


namespace occupation_combinations_eq_636_l154_15480

/-- The number of Earth-like planets -/
def earth_like_planets : ℕ := 5

/-- The number of Mars-like planets -/
def mars_like_planets : ℕ := 9

/-- The number of colonization units required for an Earth-like planet -/
def earth_units : ℕ := 2

/-- The number of colonization units required for a Mars-like planet -/
def mars_units : ℕ := 1

/-- The total number of available colonization units -/
def total_units : ℕ := 14

/-- Function to calculate the number of combinations of occupying planets -/
def occupation_combinations : ℕ := sorry

theorem occupation_combinations_eq_636 : occupation_combinations = 636 := by sorry

end occupation_combinations_eq_636_l154_15480


namespace correct_propositions_l154_15476

theorem correct_propositions : 
  -- Proposition 2
  (∀ a b : ℝ, a > 0 ∧ b > 0 → (a + b) / 2 ≥ Real.sqrt (a * b) ∧ Real.sqrt (a * b) ≥ (a * b) / (a + b)) ∧
  -- Proposition 3
  (∀ a b : ℝ, a < b ∧ b < 0 → a^2 > a * b ∧ a * b > b^2) ∧
  -- Proposition 4
  (Real.log 9 * Real.log 11 < 1) ∧
  -- Proposition 5
  (∀ a b : ℝ, a > b ∧ 1/a > 1/b → a > 0 ∧ b < 0) ∧
  -- Proposition 1 (incorrect)
  ¬(∀ a b : ℝ, a < b ∧ b < 0 → 1/a < 1/b) ∧
  -- Proposition 6 (incorrect)
  ¬(∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 1/y = 1 → (x + 2*y ≥ 6 ∧ ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 1/y₀ = 1 ∧ x₀ + 2*y₀ = 6)) :=
by sorry


end correct_propositions_l154_15476


namespace quadratic_inequality_solution_l154_15442

theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 2 > 0 ↔ -1/2 < x ∧ x < 1/3) → a - b = -10 :=
by sorry

end quadratic_inequality_solution_l154_15442


namespace raise_upper_bound_l154_15426

/-- Represents a percentage as a real number between 0 and 1 -/
def Percentage := { p : ℝ // 0 ≤ p ∧ p ≤ 1 }

/-- The lower bound of the raise -/
def lower_bound : Percentage := ⟨0.05, by sorry⟩

/-- A possible raise value within the range -/
def possible_raise : Percentage := ⟨0.08, by sorry⟩

/-- The upper bound of the raise -/
def upper_bound : Percentage := ⟨0.09, by sorry⟩

theorem raise_upper_bound :
  lower_bound.val < possible_raise.val ∧
  possible_raise.val < upper_bound.val ∧
  ∀ (p : Percentage), lower_bound.val < p.val → p.val < upper_bound.val →
    p.val ≤ possible_raise.val ∨ possible_raise.val < p.val :=
by sorry

end raise_upper_bound_l154_15426


namespace triangle_abc_properties_l154_15447

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) :
  b * Real.sin A = Real.sqrt 3 * a * Real.cos B →
  b = 2 →
  B = π / 3 ∧
  (∃ (S : ℝ), S = Real.sqrt 3 ∧ ∀ (S' : ℝ), S' ≤ S) :=
by sorry

end triangle_abc_properties_l154_15447


namespace smallest_k_multiple_of_144_l154_15461

def sum_of_squares (k : ℕ+) : ℕ := k.val * (k.val + 1) * (2 * k.val + 1) / 6

theorem smallest_k_multiple_of_144 :
  ∀ k : ℕ+, k.val < 26 → ¬(144 ∣ sum_of_squares k) ∧
  144 ∣ sum_of_squares 26 :=
sorry

end smallest_k_multiple_of_144_l154_15461


namespace typhoon_tree_problem_l154_15408

theorem typhoon_tree_problem :
  ∀ (total survived died : ℕ),
    total = 14 →
    died = survived + 4 →
    survived + died = total →
    died = 9 :=
by
  sorry

end typhoon_tree_problem_l154_15408


namespace inverse_variation_problem_l154_15438

/-- Two quantities vary inversely if their product is constant -/
def VaryInversely (r s : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, r x * s x = k

theorem inverse_variation_problem (r s : ℝ → ℝ) 
  (h1 : VaryInversely r s)
  (h2 : r 1 = 1500)
  (h3 : s 1 = 0.4)
  (h4 : r 2 = 3000) :
  s 2 = 0.2 := by
  sorry

end inverse_variation_problem_l154_15438


namespace car_rental_maximum_profit_l154_15430

/-- Represents the car rental company's profit function --/
def profit_function (n : ℝ) : ℝ :=
  -50 * (n^2 - 100*n + 630000)

/-- Represents the rental fee calculation --/
def rental_fee (n : ℝ) : ℝ :=
  3000 + 50 * n

theorem car_rental_maximum_profit :
  ∃ (n : ℝ),
    profit_function n = 307050 ∧
    rental_fee n = 4050 ∧
    ∀ (m : ℝ), profit_function m ≤ profit_function n :=
by
  sorry

end car_rental_maximum_profit_l154_15430


namespace max_sum_squares_l154_15401

theorem max_sum_squares (a b c d : ℝ) 
  (h1 : a + b = 18)
  (h2 : a * b + c + d = 91)
  (h3 : a * d + b * c = 195)
  (h4 : c * d = 120) :
  a^2 + b^2 + c^2 + d^2 ≤ 82 ∧ 
  ∃ (a' b' c' d' : ℝ), 
    a' + b' = 18 ∧
    a' * b' + c' + d' = 91 ∧
    a' * d' + b' * c' = 195 ∧
    c' * d' = 120 ∧
    a'^2 + b'^2 + c'^2 + d'^2 = 82 :=
by sorry

end max_sum_squares_l154_15401


namespace tenth_finger_is_two_l154_15421

-- Define the functions f and g
def f : ℕ → ℕ
| 4 => 3
| 1 => 8
| 7 => 2
| _ => 0  -- Default case

def g : ℕ → ℕ
| 3 => 1
| 8 => 7
| 2 => 1
| _ => 0  -- Default case

-- Define a function that applies f and g alternately n times
def applyAlternately (n : ℕ) (start : ℕ) : ℕ :=
  match n with
  | 0 => start
  | n + 1 => if n % 2 = 0 then g (applyAlternately n start) else f (applyAlternately n start)

-- Theorem statement
theorem tenth_finger_is_two : applyAlternately 9 4 = 2 := by
  sorry

end tenth_finger_is_two_l154_15421


namespace polygon_interior_angles_sum_l154_15415

theorem polygon_interior_angles_sum (n : ℕ) : 
  (n ≥ 3) → ((n - 2) * 180 = 900) → n = 7 := by
  sorry

end polygon_interior_angles_sum_l154_15415


namespace boat_drift_l154_15452

/-- Calculate the drift of a boat crossing a river -/
theorem boat_drift (river_width : ℝ) (boat_speed : ℝ) (crossing_time : ℝ) :
  river_width = 400 ∧ boat_speed = 10 ∧ crossing_time = 50 →
  boat_speed * crossing_time - river_width = 100 := by
  sorry

end boat_drift_l154_15452


namespace mans_speed_against_current_l154_15482

/-- Given a man's speed with the current and the speed of the current,
    calculates the man's speed against the current. -/
def speedAgainstCurrent (speedWithCurrent : ℝ) (currentSpeed : ℝ) : ℝ :=
  speedWithCurrent - 2 * currentSpeed

/-- Theorem stating that given the specific conditions of the problem,
    the man's speed against the current is 18 kmph. -/
theorem mans_speed_against_current :
  speedAgainstCurrent 20 1 = 18 := by
  sorry

#eval speedAgainstCurrent 20 1

end mans_speed_against_current_l154_15482


namespace rays_grocery_bill_l154_15483

/-- Calculates the total grocery bill for Ray's purchase with a store rewards discount --/
theorem rays_grocery_bill :
  let meat_price : ℚ := 5
  let crackers_price : ℚ := 3.5
  let vegetable_price : ℚ := 2
  let vegetable_quantity : ℕ := 4
  let cheese_price : ℚ := 3.5
  let discount_rate : ℚ := 0.1

  let subtotal : ℚ := meat_price + crackers_price + (vegetable_price * vegetable_quantity) + cheese_price
  let discount : ℚ := subtotal * discount_rate
  let total : ℚ := subtotal - discount

  total = 18 := by sorry

end rays_grocery_bill_l154_15483


namespace sally_forgot_seven_poems_l154_15443

/-- Represents the number of poems in different categories --/
structure PoemCounts where
  initial : ℕ
  correct : ℕ
  mixed : ℕ

/-- Calculates the number of completely forgotten poems --/
def forgotten_poems (counts : PoemCounts) : ℕ :=
  counts.initial - (counts.correct + counts.mixed)

/-- Theorem stating that Sally forgot 7 poems --/
theorem sally_forgot_seven_poems : 
  let sally_counts : PoemCounts := ⟨15, 5, 3⟩
  forgotten_poems sally_counts = 7 := by
  sorry

end sally_forgot_seven_poems_l154_15443


namespace team_a_more_uniform_heights_l154_15418

/-- Represents a team with its height statistics -/
structure Team where
  averageHeight : ℝ
  variance : ℝ

/-- Defines when a team has more uniform heights than another -/
def hasMoreUniformHeights (t1 t2 : Team) : Prop :=
  t1.variance < t2.variance

/-- Theorem stating that Team A has more uniform heights than Team B -/
theorem team_a_more_uniform_heights :
  let teamA : Team := { averageHeight := 1.82, variance := 0.56 }
  let teamB : Team := { averageHeight := 1.82, variance := 2.1 }
  hasMoreUniformHeights teamA teamB := by
  sorry

end team_a_more_uniform_heights_l154_15418


namespace g_difference_l154_15479

/-- The function g(x) = 2x^3 + 5x^2 - 2x - 1 -/
def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 - 2 * x - 1

/-- Theorem stating that g(x+h) - g(x) = h(6x^2 + 6xh + 2h^2 + 10x + 5h - 2) for all x and h -/
theorem g_difference (x h : ℝ) : g (x + h) - g x = h * (6 * x^2 + 6 * x * h + 2 * h^2 + 10 * x + 5 * h - 2) := by
  sorry

end g_difference_l154_15479


namespace reflection_across_x_axis_l154_15495

/-- A point in a 2D Cartesian coordinate system -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Reflection of a point across the x-axis -/
def reflect_x (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

theorem reflection_across_x_axis : 
  let P : Point2D := { x := -2, y := 5 }
  reflect_x P = { x := -2, y := -5 } := by
  sorry

end reflection_across_x_axis_l154_15495


namespace expected_girls_left_ten_boys_seven_girls_l154_15428

/-- The expected number of girls standing to the left of all boys in a random lineup -/
def expected_girls_left (num_boys : ℕ) (num_girls : ℕ) : ℚ :=
  num_girls / (num_boys + 1 : ℚ)

/-- Theorem: In a random lineup of 10 boys and 7 girls, the expected number of girls 
    standing to the left of all boys is 7/11 -/
theorem expected_girls_left_ten_boys_seven_girls :
  expected_girls_left 10 7 = 7 / 11 := by
  sorry

end expected_girls_left_ten_boys_seven_girls_l154_15428


namespace constant_term_of_binomial_expansion_l154_15478

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the constant term in the expansion of (x - 1/x)^8
def constantTerm : ℕ := binomial 8 4

-- Theorem statement
theorem constant_term_of_binomial_expansion :
  constantTerm = 70 := by sorry

end constant_term_of_binomial_expansion_l154_15478


namespace project_payment_l154_15450

/-- Represents the hourly wage of candidate q -/
def q_wage : ℝ := 14

/-- Represents the hourly wage of candidate p -/
def p_wage : ℝ := 21

/-- Represents the number of hours candidate p needs to complete the job -/
def p_hours : ℝ := 20

/-- Represents the number of hours candidate q needs to complete the job -/
def q_hours : ℝ := p_hours + 10

/-- The total payment for the project -/
def total_payment : ℝ := 420

theorem project_payment :
  (p_wage = q_wage * 1.5) ∧
  (p_wage = q_wage + 7) ∧
  (q_hours = p_hours + 10) ∧
  (p_wage * p_hours = q_wage * q_hours) →
  total_payment = 420 := by sorry

end project_payment_l154_15450


namespace unique_quadratic_solution_l154_15488

theorem unique_quadratic_solution (b : ℝ) (h1 : b ≠ 0) 
  (h2 : ∃! x, b * x^2 + 16 * x + 12 = 0) : 
  ∃ x, b * x^2 + 16 * x + 12 = 0 ∧ x = -3/2 := by
  sorry

end unique_quadratic_solution_l154_15488


namespace locus_of_symmetric_points_l154_15491

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The origin point (0, 0) -/
def origin : Point2D := ⟨0, 0⟩

/-- Check if a point is on the x-axis -/
def isOnXAxis (p : Point2D) : Prop := p.y = 0

/-- Check if a point is on the y-axis -/
def isOnYAxis (p : Point2D) : Prop := p.x = 0

/-- Check if three points form a right angle -/
def isRightAngle (p1 p2 p3 : Point2D) : Prop :=
  (p2.x - p1.x) * (p3.x - p1.x) + (p2.y - p1.y) * (p3.y - p1.y) = 0

/-- The line symmetric to a point with respect to the coordinate axes -/
def symmetricLine (m : Point2D) : Set Point2D :=
  {n : Point2D | n.x * m.y = n.y * m.x}

/-- The main theorem -/
theorem locus_of_symmetric_points (m : Point2D) 
  (h1 : m ≠ origin) 
  (h2 : ¬isOnXAxis m) 
  (h3 : ¬isOnYAxis m) :
  ∀ (p q : Point2D), 
    isOnXAxis p → isOnYAxis q → isRightAngle p m q →
    ∃ (n : Point2D), n ∈ symmetricLine m :=
by sorry

end locus_of_symmetric_points_l154_15491


namespace last_digit_98_base5_l154_15457

def last_digit_base5 (n : ℕ) : ℕ :=
  n % 5

theorem last_digit_98_base5 :
  last_digit_base5 98 = 3 := by
sorry

end last_digit_98_base5_l154_15457


namespace unique_solution_condition_l154_15475

theorem unique_solution_condition (k : ℝ) : 
  (∃! x : ℝ, (3*x + 8)*(x - 6) = -54 + k*x) ↔ 
  (k = 6*Real.sqrt 2 - 10 ∨ k = -6*Real.sqrt 2 - 10) := by sorry

end unique_solution_condition_l154_15475


namespace problem_solution_l154_15439

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem problem_solution (n : ℕ) (h : n * factorial n + 2 * factorial n = 5040) : n = 5 := by
  sorry

end problem_solution_l154_15439


namespace merchant_profit_l154_15496

theorem merchant_profit (C S : ℝ) (h : 22 * C = 16 * S) : 
  (S - C) / C * 100 = 37.5 := by sorry

end merchant_profit_l154_15496


namespace probability_is_correct_l154_15467

/-- Represents a unit cube within the larger cube -/
structure UnitCube where
  painted_faces : Nat
  deriving Repr

/-- Represents the larger 5x5x5 cube -/
def LargeCube : Type := Array UnitCube

/-- Creates a large cube with the specified painting configuration -/
def create_large_cube : LargeCube :=
  sorry

/-- Calculates the number of ways to choose 2 items from n items -/
def choose_two (n : Nat) : Nat :=
  n * (n - 1) / 2

/-- Calculates the probability of selecting one cube with two painted faces
    and one cube with no painted faces -/
def probability_two_and_none (cube : LargeCube) : Rat :=
  sorry

theorem probability_is_correct (cube : LargeCube) :
  probability_two_and_none (create_large_cube) = 187 / 3875 := by
  sorry

end probability_is_correct_l154_15467


namespace right_triangles_shared_hypotenuse_l154_15492

theorem right_triangles_shared_hypotenuse (b : ℝ) (h : b ≥ Real.sqrt 3) :
  let BC : ℝ := 1
  let AC : ℝ := b
  let AD : ℝ := 2
  let AB : ℝ := Real.sqrt (AC^2 + BC^2)
  let BD : ℝ := Real.sqrt (AB^2 - AD^2)
  BD = Real.sqrt (b^2 - 3) :=
by sorry

end right_triangles_shared_hypotenuse_l154_15492


namespace thirteen_points_guarantee_win_thirteen_smallest_guarantee_l154_15497

/-- Represents the points awarded for each position in a race -/
def race_points : Fin 3 → ℕ
  | 0 => 5  -- First place
  | 1 => 3  -- Second place
  | 2 => 1  -- Third place
  | _ => 0  -- This case should never occur due to Fin 3

/-- The total number of races -/
def num_races : ℕ := 3

/-- A function to calculate the maximum points possible for the second-place student -/
def max_second_place_points : ℕ := sorry

/-- Theorem stating that 13 points guarantees more points than any other student -/
theorem thirteen_points_guarantee_win :
  ∀ (student_points : ℕ),
    student_points ≥ 13 →
    student_points > max_second_place_points :=
  sorry

/-- Theorem stating that 13 is the smallest number of points that guarantees a win -/
theorem thirteen_smallest_guarantee :
  ∀ (n : ℕ),
    n < 13 →
    ∃ (other_points : ℕ),
      other_points ≥ n ∧
      other_points ≤ max_second_place_points :=
  sorry

end thirteen_points_guarantee_win_thirteen_smallest_guarantee_l154_15497


namespace grid_broken_lines_theorem_l154_15449

/-- Represents a grid of cells -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a broken line in the grid -/
structure BrokenLine :=
  (length : ℕ)

/-- Checks if it's possible to construct a set of broken lines in a grid -/
def canConstructBrokenLines (g : Grid) (lines : List BrokenLine) : Prop :=
  -- The actual implementation would be complex and is omitted
  sorry

theorem grid_broken_lines_theorem (g : Grid) :
  g.width = 11 ∧ g.height = 1 →
  (canConstructBrokenLines g (List.replicate 8 ⟨5⟩)) ∧
  ¬(canConstructBrokenLines g (List.replicate 5 ⟨8⟩)) :=
by sorry

end grid_broken_lines_theorem_l154_15449


namespace office_staff_composition_l154_15432

/-- Represents the number of non-officers in an office -/
def num_non_officers : ℕ := sorry

/-- Represents the number of officers in an office -/
def num_officers : ℕ := 15

/-- Represents the average salary of all employees in rupees -/
def avg_salary_all : ℚ := 120

/-- Represents the average salary of officers in rupees -/
def avg_salary_officers : ℚ := 440

/-- Represents the average salary of non-officers in rupees -/
def avg_salary_non_officers : ℚ := 110

theorem office_staff_composition :
  (num_officers * avg_salary_officers + num_non_officers * avg_salary_non_officers) / (num_officers + num_non_officers) = avg_salary_all ∧
  num_non_officers = 480 := by sorry

end office_staff_composition_l154_15432


namespace stating_men_meet_at_calculated_point_l154_15456

/-- Two men walk towards each other from points A and B, which are 90 miles apart. -/
def total_distance : ℝ := 90

/-- The speed of the man starting from point A in miles per hour. -/
def speed_a : ℝ := 5

/-- The initial speed of the man starting from point B in miles per hour. -/
def initial_speed_b : ℝ := 2

/-- The hourly increase in speed for the man starting from point B. -/
def speed_increase_b : ℝ := 1

/-- The number of hours the man from A waits before starting. -/
def wait_time : ℕ := 1

/-- The total time in hours until the men meet. -/
def total_time : ℕ := 10

/-- The distance from point B where the men meet. -/
def meeting_point : ℝ := 52.5

/-- 
Theorem stating that the men meet at the specified distance from B after the given time,
given their walking patterns.
-/
theorem men_meet_at_calculated_point :
  let distance_a := speed_a * (total_time - wait_time)
  let distance_b := (total_time / 2 : ℝ) * (initial_speed_b + initial_speed_b + speed_increase_b * (total_time - 1))
  distance_a + distance_b = total_distance ∧ distance_b = meeting_point := by sorry

end stating_men_meet_at_calculated_point_l154_15456


namespace certain_number_proof_l154_15490

theorem certain_number_proof : ∃ x : ℚ, x - (390 / 5) = (4 - (210 / 7)) + 114 ∧ x = 166 := by
  sorry

end certain_number_proof_l154_15490


namespace log_product_equality_l154_15481

theorem log_product_equality : (Real.log 3 / Real.log 4 + Real.log 3 / Real.log 8) *
  (Real.log 2 / Real.log 3 + Real.log 8 / Real.log 9) = 25 / 12 := by
  sorry

end log_product_equality_l154_15481


namespace circle_symmetry_minimum_l154_15414

/-- Given a circle x^2 + y^2 + 2x - 4y + 1 = 0 symmetric with respect to the line 2ax - by + 2 = 0,
    where a > 0 and b > 0, the minimum value of 4/a + 1/b is 9. -/
theorem circle_symmetry_minimum (a b : ℝ) : a > 0 → b > 0 →
  (∀ x y : ℝ, x^2 + y^2 + 2*x - 4*y + 1 = 0 →
    ∃ x' y' : ℝ, x'^2 + y'^2 + 2*x' - 4*y' + 1 = 0 ∧
      2*a*x - b*y + 2 = 2*a*x' - b*y' + 2) →
  (∀ t : ℝ, 4/a + 1/b ≥ t) →
  t = 9 :=
sorry

end circle_symmetry_minimum_l154_15414


namespace stock_price_uniqueness_l154_15412

theorem stock_price_uniqueness : ¬∃ (k l : ℕ), (117/100)^k * (83/100)^l = 1 := by
  sorry

end stock_price_uniqueness_l154_15412


namespace right_angled_triangle_unique_k_l154_15462

/-- A triangle with side lengths 13, 17, and k is right-angled if and only if k = 21 -/
theorem right_angled_triangle_unique_k : ∃! (k : ℕ), k > 0 ∧ 
  (13^2 + 17^2 = k^2 ∨ 13^2 + k^2 = 17^2 ∨ 17^2 + k^2 = 13^2) := by
  sorry

end right_angled_triangle_unique_k_l154_15462


namespace complex_number_in_second_quadrant_l154_15465

theorem complex_number_in_second_quadrant :
  let z : ℂ := (-2 + 3 * Complex.I) / (3 - 4 * Complex.I)
  (z.re < 0) ∧ (z.im > 0) := by sorry

end complex_number_in_second_quadrant_l154_15465


namespace integer_ratio_problem_l154_15466

theorem integer_ratio_problem (s l : ℕ) : 
  s = 32 →
  ∃ k : ℕ, l = k * s →
  s + l = 96 →
  l / s = 2 :=
by sorry

end integer_ratio_problem_l154_15466


namespace tens_digit_of_smallest_divisible_l154_15444

-- Define the smallest positive integer divisible by 20, 16, and 2016
def smallest_divisible : ℕ := 10080

-- Define a function to get the tens digit of a natural number
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

-- Theorem statement
theorem tens_digit_of_smallest_divisible :
  tens_digit smallest_divisible = 8 ∧
  ∀ m : ℕ, m > 0 ∧ 20 ∣ m ∧ 16 ∣ m ∧ 2016 ∣ m → m ≥ smallest_divisible :=
by sorry

end tens_digit_of_smallest_divisible_l154_15444


namespace min_a_value_l154_15411

theorem min_a_value (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x + Real.sqrt (x * y) ≤ a * (x + y)) → 
  a ≥ (Real.sqrt 2 + 1) / 2 :=
by sorry

end min_a_value_l154_15411


namespace power_sum_equals_eighteen_l154_15416

theorem power_sum_equals_eighteen :
  (-3)^4 + (-3)^2 + (-3)^1 + 3^1 - 3^4 + 3^2 = 18 := by
  sorry

end power_sum_equals_eighteen_l154_15416


namespace hybrid_car_trip_length_l154_15451

theorem hybrid_car_trip_length : 
  ∀ (trip_length : ℝ),
  (trip_length / (0.02 * (trip_length - 40)) = 55) →
  trip_length = 440 :=
by sorry

end hybrid_car_trip_length_l154_15451


namespace cindy_wins_prob_l154_15446

-- Define the probability of tossing a five
def prob_five : ℚ := 1/6

-- Define the probability of not tossing a five
def prob_not_five : ℚ := 1 - prob_five

-- Define the probability of Cindy winning in the first cycle
def prob_cindy_first_cycle : ℚ := prob_not_five * prob_five

-- Define the probability of the game continuing after one full cycle
def prob_continue : ℚ := prob_not_five^3

-- Theorem statement
theorem cindy_wins_prob : 
  let a : ℚ := prob_cindy_first_cycle
  let r : ℚ := prob_continue
  (a / (1 - r)) = 30/91 := by
  sorry

end cindy_wins_prob_l154_15446


namespace log_simplification_l154_15487

-- Define the variables as positive real numbers
variable (a b d e y z : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hd : 0 < d) (he : 0 < e) (hy : 0 < y) (hz : 0 < z)

-- State the theorem
theorem log_simplification :
  Real.log (a / b) + Real.log (b / e) + Real.log (e / d) - Real.log (a * z / (d * y)) = Real.log (d * y / z) :=
by sorry

end log_simplification_l154_15487


namespace paving_cost_example_l154_15472

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

theorem paving_cost_example :
  paving_cost 5.5 4 700 = 15400 := by
  sorry

end paving_cost_example_l154_15472


namespace line_2x_plus_1_not_in_fourth_quadrant_l154_15494

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  y_intercept : ℝ

/-- Defines the fourth quadrant of the 2D plane -/
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

/-- Checks if a given line passes through the fourth quadrant -/
def passes_through_fourth_quadrant (l : Line) : Prop :=
  ∃ x y : ℝ, y = l.slope * x + l.y_intercept ∧ fourth_quadrant x y

/-- The main theorem stating that the line y = 2x + 1 does not pass through the fourth quadrant -/
theorem line_2x_plus_1_not_in_fourth_quadrant :
  ¬ passes_through_fourth_quadrant (Line.mk 2 1) := by
  sorry


end line_2x_plus_1_not_in_fourth_quadrant_l154_15494


namespace opposite_direction_speed_l154_15406

/-- Given two people moving in opposite directions, this theorem proves
    the speed of one person given the conditions of the problem. -/
theorem opposite_direction_speed
  (time : ℝ)
  (total_distance : ℝ)
  (speed_person1 : ℝ)
  (h1 : time = 4)
  (h2 : total_distance = 28)
  (h3 : speed_person1 = 3)
  : ∃ speed_person2 : ℝ,
    speed_person2 = 4 ∧ 
    total_distance = time * (speed_person1 + speed_person2) :=
by
  sorry

#check opposite_direction_speed

end opposite_direction_speed_l154_15406


namespace sum_of_products_l154_15405

theorem sum_of_products : 5 * 7 + 6 * 12 + 15 * 4 + 4 * 9 = 203 := by
  sorry

end sum_of_products_l154_15405


namespace min_repetitions_divisible_by_15_l154_15420

def repeated_2002_plus_15 (n : ℕ) : ℕ :=
  2002 * (10^(4*n) - 1) / 9 * 10 + 15

theorem min_repetitions_divisible_by_15 :
  ∀ k : ℕ, k < 3 → ¬(repeated_2002_plus_15 k % 15 = 0) ∧
  repeated_2002_plus_15 3 % 15 = 0 :=
sorry

end min_repetitions_divisible_by_15_l154_15420


namespace tangent_points_concyclic_l154_15407

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the center of a circle
variable (center : Circle → Point)

-- Define a predicate for a point being outside a circle
variable (is_outside : Point → Circle → Prop)

-- Define a predicate for two circles being concentric
variable (concentric : Circle → Circle → Prop)

-- Define a predicate for a line being tangent to a circle at a point
variable (is_tangent : Point → Point → Circle → Prop)

-- Define a predicate for points being concyclic
variable (concyclic : List Point → Prop)

-- State the theorem
theorem tangent_points_concyclic 
  (O : Point) (c1 c2 : Circle) (M A B C D : Point) :
  center c1 = O →
  center c2 = O →
  concentric c1 c2 →
  is_outside M c1 →
  is_outside M c2 →
  is_tangent M A c1 →
  is_tangent M B c1 →
  is_tangent M C c2 →
  is_tangent M D c2 →
  concyclic [M, A, B, C, D] :=
sorry

end tangent_points_concyclic_l154_15407


namespace books_for_girls_l154_15477

theorem books_for_girls (num_girls num_boys total_books : ℕ) 
  (h_girls : num_girls = 15)
  (h_boys : num_boys = 10)
  (h_books : total_books = 375)
  (h_equal_division : ∃ (books_per_student : ℕ), 
    total_books = books_per_student * (num_girls + num_boys)) :
  ∃ (books_for_girls : ℕ), books_for_girls = 225 ∧ 
    books_for_girls = num_girls * (total_books / (num_girls + num_boys)) := by
  sorry

end books_for_girls_l154_15477


namespace janice_purchase_problem_l154_15470

theorem janice_purchase_problem :
  ∃ (a d b c : ℕ), 
    a + d + b + c = 50 ∧ 
    30 * a + 150 * d + 200 * b + 300 * c = 6000 :=
by sorry

end janice_purchase_problem_l154_15470


namespace milk_tea_price_proof_l154_15425

/-- The cost of a cup of milk tea in dollars -/
def milk_tea_cost : ℝ := 2.4

/-- The cost of a slice of cake in dollars -/
def cake_slice_cost : ℝ := 0.75 * milk_tea_cost

theorem milk_tea_price_proof :
  (cake_slice_cost = 0.75 * milk_tea_cost) ∧
  (2 * cake_slice_cost + milk_tea_cost = 6) →
  milk_tea_cost = 2.4 := by
  sorry

end milk_tea_price_proof_l154_15425


namespace binomial_5_choose_3_l154_15427

theorem binomial_5_choose_3 : Nat.choose 5 3 = 10 := by
  sorry

end binomial_5_choose_3_l154_15427


namespace log_850_between_consecutive_integers_l154_15484

theorem log_850_between_consecutive_integers :
  ∃ (a b : ℤ), a + 1 = b ∧ (a : ℝ) < Real.log 850 / Real.log 10 ∧ Real.log 850 / Real.log 10 < b ∧ a + b = 5 := by
  sorry

end log_850_between_consecutive_integers_l154_15484


namespace total_distance_is_164_l154_15419

-- Define the parameters
def flat_speed : ℝ := 20
def flat_time : ℝ := 4.5
def uphill_speed : ℝ := 12
def uphill_time : ℝ := 2.5
def downhill_speed : ℝ := 24
def downhill_time : ℝ := 1.5
def walking_distance : ℝ := 8

-- Define the function to calculate distance
def distance (speed time : ℝ) : ℝ := speed * time

-- Theorem statement
theorem total_distance_is_164 :
  distance flat_speed flat_time +
  distance uphill_speed uphill_time +
  distance downhill_speed downhill_time +
  walking_distance = 164 := by
  sorry

end total_distance_is_164_l154_15419


namespace only_blue_possible_all_blue_possible_l154_15403

/-- Represents the number of sheep of each color -/
structure SheepCounts where
  blue : ℕ
  red : ℕ
  green : ℕ

/-- Represents a valid transformation of sheep colors -/
inductive SheepTransform : SheepCounts → SheepCounts → Prop where
  | blue_red_to_green : ∀ b r g, SheepTransform ⟨b+1, r+1, g-2⟩ ⟨b, r, g⟩
  | blue_green_to_red : ∀ b r g, SheepTransform ⟨b+1, r-2, g+1⟩ ⟨b, r, g⟩
  | red_green_to_blue : ∀ b r g, SheepTransform ⟨b-2, r+1, g+1⟩ ⟨b, r, g⟩

/-- Represents a sequence of transformations -/
def TransformSequence : SheepCounts → SheepCounts → Prop :=
  Relation.ReflTransGen SheepTransform

/-- The theorem stating that only blue is possible as the final uniform color -/
theorem only_blue_possible (initial : SheepCounts) (final : SheepCounts) :
  initial = ⟨22, 18, 15⟩ →
  TransformSequence initial final →
  (final.blue = 0 ∧ final.red = 55) ∨ (final.blue = 0 ∧ final.green = 55) →
  False :=
sorry

/-- The theorem stating that all blue is possible -/
theorem all_blue_possible (initial : SheepCounts) (final : SheepCounts) :
  initial = ⟨22, 18, 15⟩ →
  ∃ final, TransformSequence initial final ∧ final = ⟨55, 0, 0⟩ :=
sorry

end only_blue_possible_all_blue_possible_l154_15403


namespace money_left_relation_l154_15473

/-- The relationship between money left and masks bought -/
theorem money_left_relation (initial_amount : ℝ) (mask_price : ℝ) (x : ℝ) (y : ℝ) :
  initial_amount = 60 →
  mask_price = 2 →
  y = initial_amount - mask_price * x →
  y = 60 - 2 * x :=
by sorry

end money_left_relation_l154_15473


namespace dark_light_difference_l154_15471

/-- Represents a square on the grid -/
inductive Square
| Dark
| Light

/-- Represents a row in the grid -/
def Row := Vector Square 9

/-- Represents the entire 9x9 grid -/
def Grid := Vector Row 9

/-- Creates an alternating row starting with the given square color -/
def alternatingRow (start : Square) : Row := sorry

/-- Creates the 9x9 grid with alternating pattern -/
def createGrid : Grid :=
  Vector.ofFn (λ i => alternatingRow (if i % 2 = 0 then Square.Dark else Square.Light))

/-- Counts the number of dark squares in the grid -/
def countDarkSquares (grid : Grid) : Nat := sorry

/-- Counts the number of light squares in the grid -/
def countLightSquares (grid : Grid) : Nat := sorry

/-- The main theorem stating the difference between dark and light squares -/
theorem dark_light_difference :
  let grid := createGrid
  countDarkSquares grid = countLightSquares grid + 1 := by sorry

end dark_light_difference_l154_15471


namespace probability_ratio_l154_15493

def total_slips : ℕ := 40
def distinct_numbers : ℕ := 8
def slips_per_number : ℕ := 5
def drawn_slips : ℕ := 4

def probability_same_number (n : ℕ) : ℚ :=
  (n * slips_per_number.choose drawn_slips) / total_slips.choose drawn_slips

def probability_two_pairs (n : ℕ) : ℚ :=
  (n.choose 2 * slips_per_number.choose 2 * slips_per_number.choose 2) / total_slips.choose drawn_slips

theorem probability_ratio :
  probability_two_pairs distinct_numbers / probability_same_number distinct_numbers = 70 := by
  sorry

end probability_ratio_l154_15493


namespace sine_equality_equivalence_l154_15402

theorem sine_equality_equivalence (α β : ℝ) : 
  (∃ k : ℤ, α = k * Real.pi + (-1)^k * β) ↔ Real.sin α = Real.sin β := by
  sorry

end sine_equality_equivalence_l154_15402


namespace arithmetic_expression_evaluation_l154_15498

theorem arithmetic_expression_evaluation : 3 * 4 + (2 * 5)^2 - 6 * 2 = 100 := by
  sorry

end arithmetic_expression_evaluation_l154_15498


namespace formula_always_zero_l154_15434

theorem formula_always_zero :
  ∃ (F : (ℝ → ℝ → ℝ) → (ℝ → ℝ → ℝ) → ℝ → ℝ), 
    ∀ (sub mul : ℝ → ℝ → ℝ) (a : ℝ), 
      (∀ x y, sub x y = x - y ∨ sub x y = x * y) →
      (∀ x y, mul x y = x * y ∨ mul x y = x - y) →
      F sub mul a = 0 :=
by sorry

end formula_always_zero_l154_15434


namespace problem_1_l154_15453

theorem problem_1 : |-2| + (1/3)⁻¹ - (Real.sqrt 3 - 2021)^0 - Real.sqrt 3 * Real.tan (π/3) = 1 := by
  sorry

end problem_1_l154_15453


namespace turtleneck_profit_l154_15468

/-- Represents the pricing strategy and profit calculation for turtleneck sweaters -/
theorem turtleneck_profit (C : ℝ) : 
  let initial_price := C * 1.20
  let new_year_price := initial_price * 1.25
  let february_price := new_year_price * 0.94
  let profit := february_price - C
  profit = C * 0.41 := by sorry

end turtleneck_profit_l154_15468


namespace rachel_plant_arrangement_l154_15437

def num_arrangements (n : ℕ) : ℕ :=
  let all_under_one := 2  -- All plants under one white lamp or one red lamp
  let all_same_color := 2 * (n.choose 2)  -- All plants under lamps of the same color
  let diff_colors := (n.choose 2) + 2 * (n.choose 1)  -- Plants under lamps of different colors
  all_under_one + all_same_color + diff_colors

theorem rachel_plant_arrangement :
  num_arrangements 4 = 28 :=
by sorry

end rachel_plant_arrangement_l154_15437


namespace sin_30_tan_45_calculation_l154_15413

theorem sin_30_tan_45_calculation : 2 * Real.sin (30 * π / 180) - Real.tan (45 * π / 180) = 0 := by
  sorry

end sin_30_tan_45_calculation_l154_15413


namespace decagon_perimeter_l154_15469

/-- The perimeter of a regular polygon with n sides of length s -/
def perimeter (n : ℕ) (s : ℝ) : ℝ := n * s

/-- The number of sides in a decagon -/
def decagon_sides : ℕ := 10

/-- The side length of our specific decagon -/
def side_length : ℝ := 3

theorem decagon_perimeter : perimeter decagon_sides side_length = 30 := by
  sorry

end decagon_perimeter_l154_15469


namespace sphere_volume_equals_surface_area_l154_15464

theorem sphere_volume_equals_surface_area (r : ℝ) : 
  (4 / 3 * Real.pi * r^3 = 4 * Real.pi * r^2) → r = 1 := by
  sorry

end sphere_volume_equals_surface_area_l154_15464


namespace complex_equation_solution_l154_15459

theorem complex_equation_solution (z : ℂ) (h : z * Complex.I + z = 2) : z = 1 - Complex.I := by
  sorry

end complex_equation_solution_l154_15459


namespace diorama_building_time_l154_15422

/-- Given the total time spent on a diorama and the relationship between building and planning time,
    prove the time spent building the diorama. -/
theorem diorama_building_time (total_time planning_time building_time : ℕ) 
    (h1 : total_time = 67)
    (h2 : building_time = 3 * planning_time - 5)
    (h3 : total_time = planning_time + building_time) : 
    building_time = 49 := by
  sorry

end diorama_building_time_l154_15422


namespace movie_children_count_prove_movie_children_count_l154_15448

theorem movie_children_count : ℕ → Prop :=
  fun num_children =>
    let total_cost : ℕ := 76
    let num_adults : ℕ := 5
    let adult_ticket_cost : ℕ := 10
    let child_ticket_cost : ℕ := 7
    let concession_cost : ℕ := 12
    
    (num_adults * adult_ticket_cost + num_children * child_ticket_cost + concession_cost = total_cost) →
    num_children = 2

theorem prove_movie_children_count : ∃ (n : ℕ), movie_children_count n :=
  sorry

end movie_children_count_prove_movie_children_count_l154_15448


namespace lizette_quiz_average_l154_15429

theorem lizette_quiz_average (q1 q2 : ℝ) : 
  (q1 + q2 + 92) / 3 = 94 → (q1 + q2) / 2 = 95 := by
  sorry

end lizette_quiz_average_l154_15429


namespace class_representatives_count_l154_15433

/-- The number of ways to select and arrange class representatives -/
def class_representatives (num_male num_female : ℕ) (male_to_select female_to_select : ℕ) : ℕ :=
  Nat.choose num_male male_to_select *
  Nat.choose num_female female_to_select *
  Nat.factorial (male_to_select + female_to_select)

/-- Theorem stating that the number of ways to select and arrange class representatives
    from 3 male and 3 female students, selecting 1 male and 2 females, is 54 -/
theorem class_representatives_count :
  class_representatives 3 3 1 2 = 54 := by
  sorry

end class_representatives_count_l154_15433


namespace regular_polygon_sides_l154_15463

theorem regular_polygon_sides (n : ℕ) (h : n > 2) : 
  (∀ θ : ℝ, θ = 160 ∧ θ = (n - 2 : ℝ) * 180 / n) → n = 18 :=
by sorry

end regular_polygon_sides_l154_15463


namespace right_triangle_hypotenuse_l154_15441

theorem right_triangle_hypotenuse (a b c : ℝ) :
  a = 36 → b = 48 → c^2 = a^2 + b^2 → c = 60 := by
  sorry

end right_triangle_hypotenuse_l154_15441
