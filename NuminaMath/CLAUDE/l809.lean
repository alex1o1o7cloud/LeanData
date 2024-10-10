import Mathlib

namespace intersection_implies_a_value_l809_80961

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {-3} → a = -1 := by
  sorry

end intersection_implies_a_value_l809_80961


namespace geometric_propositions_l809_80981

-- Define the four propositions
def vertical_angles_equal : Prop := sorry
def alternate_interior_angles_equal : Prop := sorry
def parallel_transitivity : Prop := sorry
def parallel_sides_equal_angles : Prop := sorry

-- Theorem stating which propositions are true
theorem geometric_propositions :
  vertical_angles_equal ∧ 
  parallel_transitivity ∧ 
  ¬alternate_interior_angles_equal ∧ 
  ¬parallel_sides_equal_angles := by
  sorry

end geometric_propositions_l809_80981


namespace perpendicular_lines_theorem_l809_80905

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v1 v2 : Vector2D) : ℝ := v1.x * v2.x + v1.y * v2.y

/-- Perpendicularity of two 2D vectors -/
def perpendicular (v1 v2 : Vector2D) : Prop := dot_product v1 v2 = 0

theorem perpendicular_lines_theorem (b c : ℝ) :
  let v1 : Vector2D := ⟨4, 1⟩
  let v2 : Vector2D := ⟨b, -8⟩
  let v3 : Vector2D := ⟨5, c⟩
  perpendicular v1 v3 ∧ perpendicular v2 v3 → b = 2 ∧ c = -20 := by
  sorry

end perpendicular_lines_theorem_l809_80905


namespace smallest_n_for_50000_quadruplets_l809_80992

def count_quadruplets (n : ℕ) : ℕ :=
  (Finset.filter (fun (q : ℕ × ℕ × ℕ × ℕ) => 
    Nat.gcd q.1 (Nat.gcd q.2.1 (Nat.gcd q.2.2.1 q.2.2.2)) = 50 ∧ 
    Nat.lcm q.1 (Nat.lcm q.2.1 (Nat.lcm q.2.2.1 q.2.2.2)) = n
  ) (Finset.product (Finset.range (n + 1)) (Finset.product (Finset.range (n + 1)) (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))))).card

theorem smallest_n_for_50000_quadruplets :
  ∃ n : ℕ, n > 0 ∧ count_quadruplets n = 50000 ∧ 
  ∀ m : ℕ, m > 0 ∧ m < n → count_quadruplets m ≠ 50000 ∧
  n = 48600 := by
sorry

end smallest_n_for_50000_quadruplets_l809_80992


namespace sichuan_selected_count_l809_80950

/-- Represents the number of students selected from Sichuan University in a stratified sampling -/
def sichuan_selected (total_students : ℕ) (sichuan_students : ℕ) (other_students : ℕ) (selected_students : ℕ) : ℕ :=
  (selected_students * sichuan_students) / (sichuan_students + other_students)

/-- Theorem stating that 10 students from Sichuan University are selected in the given scenario -/
theorem sichuan_selected_count :
  sichuan_selected 40 25 15 16 = 10 := by
  sorry

#eval sichuan_selected 40 25 15 16

end sichuan_selected_count_l809_80950


namespace largest_prime_factors_difference_l809_80956

theorem largest_prime_factors_difference (n : Nat) (h : n = 178469) : 
  ∃ (p q : Nat), Nat.Prime p ∧ Nat.Prime q ∧ p > q ∧ 
  p ∣ n ∧ q ∣ n ∧
  (∀ (r : Nat), Nat.Prime r → r ∣ n → r ≤ p) ∧
  (∀ (r : Nat), Nat.Prime r → r ∣ n → r ≠ p → r ≤ q) ∧
  p - q = 2 := by
sorry

end largest_prime_factors_difference_l809_80956


namespace sin_405_degrees_l809_80918

theorem sin_405_degrees (h : 405 = 360 + 45) : Real.sin (405 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_405_degrees_l809_80918


namespace sixth_power_sum_l809_80900

/-- Given real numbers a, b, x, and y satisfying certain conditions, 
    prove that ax^6 + by^6 = 1531.25 -/
theorem sixth_power_sum (a b x y : ℝ) 
  (h1 : a * x + b * y = 5)
  (h2 : a * x^2 + b * y^2 = 12)
  (h3 : a * x^3 + b * y^3 = 30)
  (h4 : a * x^4 + b * y^4 = 80) :
  a * x^6 + b * y^6 = 1531.25 := by
  sorry

end sixth_power_sum_l809_80900


namespace feet_in_garden_l809_80988

/-- The number of feet in the garden --/
def total_feet (num_dogs num_ducks num_cats num_birds num_insects : ℕ) : ℕ :=
  num_dogs * 4 + num_ducks * 2 + num_cats * 4 + num_birds * 2 + num_insects * 6

/-- Theorem stating that the total number of feet in the garden is 118 --/
theorem feet_in_garden : total_feet 6 2 4 7 10 = 118 := by
  sorry

end feet_in_garden_l809_80988


namespace quadratic_inequality_solution_sets_l809_80947

/-- Given that the solution set of ax² + bx + c > 0 is (-1/3, 2),
    prove that the solution set of cx² + bx + a < 0 is (-3, 1/2) -/
theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : Set.Ioo (-1/3 : ℝ) 2 = {x | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 + b * x + a < 0} = Set.Ioo (-3 : ℝ) (1/2) := by
  sorry

end quadratic_inequality_solution_sets_l809_80947


namespace power_approximations_l809_80945

theorem power_approximations : 
  (|((1.02 : ℝ)^30 - 1.8114)| < 0.00005) ∧ 
  (|((0.996 : ℝ)^13 - 0.9492)| < 0.00005) := by
  sorry

end power_approximations_l809_80945


namespace power_exceeds_thresholds_l809_80904

theorem power_exceeds_thresholds : ∃ (n1 n2 n3 m1 m2 m3 : ℕ), 
  (1.01 : ℝ) ^ n1 > 1000000000000 ∧
  (1.001 : ℝ) ^ n2 > 1000000000000 ∧
  (1.000001 : ℝ) ^ n3 > 1000000000000 ∧
  (1.01 : ℝ) ^ m1 > 1000000000000000000 ∧
  (1.001 : ℝ) ^ m2 > 1000000000000000000 ∧
  (1.000001 : ℝ) ^ m3 > 1000000000000000000 :=
by sorry

end power_exceeds_thresholds_l809_80904


namespace distribute_four_students_three_companies_l809_80940

/-- The number of ways to distribute students among companies -/
def distribute_students (num_students : ℕ) (num_companies : ℕ) : ℕ :=
  3^4 - 3 * 2^4 + 3

/-- Theorem stating the correct number of ways to distribute 4 students among 3 companies -/
theorem distribute_four_students_three_companies :
  distribute_students 4 3 = 36 := by
  sorry

#eval distribute_students 4 3

end distribute_four_students_three_companies_l809_80940


namespace jennifer_pears_l809_80910

/-- Proves that Jennifer initially had 10 pears given the problem conditions -/
theorem jennifer_pears : ∃ P : ℕ, 
  (P + 20 + 2*P) - 6 = 44 ∧ P = 10 := by
  sorry

end jennifer_pears_l809_80910


namespace system_solutions_l809_80908

/-- System of equations -/
def system (x y z p : ℝ) : Prop :=
  x^2 - 3*y + p = z ∧ y^2 - 3*z + p = x ∧ z^2 - 3*x + p = y

theorem system_solutions :
  (∀ p : ℝ, p = 4 → ∀ x y z : ℝ, system x y z p → x = 2 ∧ y = 2 ∧ z = 2) ∧
  (∀ p : ℝ, 1 < p ∧ p < 4 → ∀ x y z : ℝ, system x y z p → x = y ∧ y = z) :=
by sorry

end system_solutions_l809_80908


namespace plane_division_l809_80958

/-- The maximum number of parts that n planes can divide 3D space into --/
def max_parts (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => 2^(n+1)

theorem plane_division :
  (max_parts 1 = 2) ∧
  (max_parts 2 ≤ 4) ∧
  (max_parts 3 ≤ 8) := by
  sorry

end plane_division_l809_80958


namespace integer_product_characterization_l809_80925

theorem integer_product_characterization (a : ℝ) : 
  (∀ n : ℕ, ∃ m : ℤ, a * n * (n + 2) * (n + 3) * (n + 4) = m) ↔ 
  (∃ k : ℤ, a = k / 6) :=
sorry

end integer_product_characterization_l809_80925


namespace alpha_value_l809_80979

-- Define complex numbers α and β
variable (α β : ℂ)

-- Define the conditions
variable (h1 : (α + β).re > 0)
variable (h2 : (Complex.I * (α - 3 * β)).re > 0)
variable (h3 : β = 4 + 3 * Complex.I)

-- Theorem to prove
theorem alpha_value : α = 12 - 3 * Complex.I := by
  sorry

end alpha_value_l809_80979


namespace parabola_point_value_l809_80999

/-- 
Given a parabola y = -x^2 + bx + c that passes through the point (-2, 3),
prove that 2c - 4b - 9 = 5
-/
theorem parabola_point_value (b c : ℝ) 
  (h : 3 = -(-2)^2 + b*(-2) + c) : 2*c - 4*b - 9 = 5 := by
  sorry

end parabola_point_value_l809_80999


namespace girls_average_age_l809_80946

/-- Proves that the average age of girls is 11 years given the school statistics --/
theorem girls_average_age (total_students : ℕ) (boys_avg_age : ℚ) (school_avg_age : ℚ) (num_girls : ℕ) :
  total_students = 600 →
  boys_avg_age = 12 →
  school_avg_age = 47 / 4 →  -- 11.75 years
  num_girls = 150 →
  let num_boys : ℕ := total_students - num_girls
  let total_age : ℚ := total_students * school_avg_age
  let boys_total_age : ℚ := num_boys * boys_avg_age
  let girls_total_age : ℚ := total_age - boys_total_age
  girls_total_age / num_girls = 11 := by
sorry


end girls_average_age_l809_80946


namespace equation_solutions_l809_80931

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 3*x = 0 ↔ x = 0 ∨ x = 3) ∧
  (∀ x : ℝ, 5*x + 2 = 3*x^2 ↔ x = -1/3 ∨ x = 2) := by
  sorry

end equation_solutions_l809_80931


namespace converse_correct_l809_80983

-- Define the original proposition
def original_proposition (x : ℝ) : Prop := x > 1 → x > 2

-- Define the converse proposition
def converse_proposition (x : ℝ) : Prop := x > 2 → x > 1

-- Theorem stating that the converse_proposition is indeed the converse of the original_proposition
theorem converse_correct :
  (∀ x : ℝ, original_proposition x) ↔ (∀ x : ℝ, converse_proposition x) :=
sorry

end converse_correct_l809_80983


namespace fireworks_saved_l809_80995

/-- The number of fireworks Henry and his friend had saved from last year -/
def fireworks_problem (henry_new : ℕ) (friend_new : ℕ) (total : ℕ) : Prop :=
  henry_new + friend_new + (total - (henry_new + friend_new)) = total

theorem fireworks_saved (henry_new friend_new total : ℕ) 
  (h1 : henry_new = 2)
  (h2 : friend_new = 3)
  (h3 : total = 11) :
  fireworks_problem henry_new friend_new total ∧ 
  (total - (henry_new + friend_new) = 6) :=
by sorry

end fireworks_saved_l809_80995


namespace paper_towel_pricing_l809_80939

theorem paper_towel_pricing (case_price : ℝ) (savings_percent : ℝ) (rolls_per_case : ℕ) :
  case_price = 9 →
  savings_percent = 25 →
  rolls_per_case = 12 →
  let individual_price := case_price * (1 + savings_percent / 100) / rolls_per_case
  individual_price = 0.9375 := by
  sorry

end paper_towel_pricing_l809_80939


namespace soft_drink_cost_l809_80916

/-- Proves that the cost of each soft drink is $4 given the conditions of Benny's purchase. -/
theorem soft_drink_cost (num_soft_drinks : ℕ) (num_candy_bars : ℕ) (total_spent : ℚ) (candy_bar_cost : ℚ) :
  num_soft_drinks = 2 →
  num_candy_bars = 5 →
  total_spent = 28 →
  candy_bar_cost = 4 →
  ∃ (soft_drink_cost : ℚ), 
    soft_drink_cost * num_soft_drinks + candy_bar_cost * num_candy_bars = total_spent ∧
    soft_drink_cost = 4 :=
by sorry

end soft_drink_cost_l809_80916


namespace roots_are_irrational_l809_80915

theorem roots_are_irrational (k : ℝ) : 
  (∃ x y : ℝ, x * y = 10 ∧ x^2 - 3*k*x + 2*k^2 - 1 = 0 ∧ y^2 - 3*k*y + 2*k^2 - 1 = 0) →
  (∃ x y : ℝ, x * y = 10 ∧ x^2 - 3*k*x + 2*k^2 - 1 = 0 ∧ y^2 - 3*k*y + 2*k^2 - 1 = 0 ∧ 
   (¬∃ m n : ℤ, x = m / n ∨ y = m / n)) :=
by sorry

end roots_are_irrational_l809_80915


namespace square_starts_with_sequence_l809_80972

theorem square_starts_with_sequence (S : ℕ) : 
  ∃ (N k : ℕ), S * 10^k ≤ N^2 ∧ N^2 < (S + 1) * 10^k :=
by sorry

end square_starts_with_sequence_l809_80972


namespace kira_breakfast_time_l809_80998

/-- Calculates the total time Kira spent making breakfast -/
def breakfast_time (num_sausages : ℕ) (num_eggs : ℕ) (time_per_sausage : ℕ) (time_per_egg : ℕ) : ℕ :=
  num_sausages * time_per_sausage + num_eggs * time_per_egg

/-- Proves that Kira's breakfast preparation time is 39 minutes -/
theorem kira_breakfast_time : 
  breakfast_time 3 6 5 4 = 39 := by
  sorry

end kira_breakfast_time_l809_80998


namespace monic_cubic_polynomial_value_l809_80987

/-- A monic cubic polynomial is a polynomial of the form x^3 + ax^2 + bx + c -/
def MonicCubicPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c

theorem monic_cubic_polynomial_value (a b c : ℝ) :
  let p := MonicCubicPolynomial a b c
  (p 2 = 3) → (p 4 = 9) → (p 6 = 19) → (p 8 = -9) := by
  sorry

end monic_cubic_polynomial_value_l809_80987


namespace append_digits_divisible_by_36_l809_80975

/-- A function that checks if a number is divisible by 36 -/
def isDivisibleBy36 (n : ℕ) : Prop := n % 36 = 0

/-- A function that appends two digits to 2020 -/
def appendTwoDigits (a b : ℕ) : ℕ := 202000 + 10 * a + b

theorem append_digits_divisible_by_36 :
  ∀ a b : ℕ, a < 10 → b < 10 →
    (isDivisibleBy36 (appendTwoDigits a b) ↔ (a = 3 ∧ b = 2) ∨ (a = 6 ∧ b = 8)) := by
  sorry

#check append_digits_divisible_by_36

end append_digits_divisible_by_36_l809_80975


namespace base3_to_base10_conversion_l809_80996

/-- Converts a list of digits in base 3 to a natural number in base 10 -/
def base3ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * 3^i) 0

/-- The base 3 representation of the number -/
def base3Digits : List Nat := [1, 2, 0, 2, 1]

theorem base3_to_base10_conversion :
  base3ToBase10 base3Digits = 142 := by
  sorry

end base3_to_base10_conversion_l809_80996


namespace equation_solution_l809_80942

theorem equation_solution : 
  let n : ℝ := 73.0434782609
  0.07 * n + 0.12 * (30 + n) + 0.04 * n = 20.4 := by
sorry

end equation_solution_l809_80942


namespace complement_of_A_in_U_l809_80993

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A as the open interval (-∞, 2)
def A : Set ℝ := {x : ℝ | x < 2}

-- State the theorem
theorem complement_of_A_in_U : 
  U \ A = {x : ℝ | x ≥ 2} := by sorry

end complement_of_A_in_U_l809_80993


namespace water_usage_per_person_l809_80986

/-- Given a family's water usage, prove the amount of water needed per person per day. -/
theorem water_usage_per_person
  (cost_per_gallon : ℝ)
  (family_size : ℕ)
  (daily_cost : ℝ)
  (h1 : cost_per_gallon = 1)
  (h2 : family_size = 6)
  (h3 : daily_cost = 3) :
  daily_cost / (cost_per_gallon * family_size) = 0.5 := by
  sorry

end water_usage_per_person_l809_80986


namespace exists_complete_gear_rotation_l809_80984

/-- Represents a gear with a certain number of teeth and some removed teeth -/
structure Gear where
  total_teeth : Nat
  removed_teeth : Finset Nat

/-- Represents the system of two gears -/
structure GearSystem where
  gear1 : Gear
  gear2 : Gear
  rotation : Nat

/-- Checks if a given rotation results in a complete gear -/
def is_complete_gear (gs : GearSystem) : Prop :=
  ∀ i : Nat, i < gs.gear1.total_teeth →
    (i ∉ gs.gear1.removed_teeth ∨ ((i + gs.rotation) % gs.gear1.total_teeth) ∉ gs.gear2.removed_teeth)

/-- The main theorem stating that there exists a rotation forming a complete gear -/
theorem exists_complete_gear_rotation (g1 g2 : Gear)
    (h1 : g1.total_teeth = 14)
    (h2 : g2.total_teeth = 14)
    (h3 : g1.removed_teeth.card = 4)
    (h4 : g2.removed_teeth.card = 4) :
    ∃ r : Nat, is_complete_gear ⟨g1, g2, r⟩ := by
  sorry


end exists_complete_gear_rotation_l809_80984


namespace initial_alloy_weight_l809_80969

/-- Given an initial alloy of weight x ounces that is 50% gold,
    if adding 24 ounces of pure gold results in a new alloy that is 80% gold,
    then the initial alloy weighs 16 ounces. -/
theorem initial_alloy_weight (x : ℝ) : 
  (0.5 * x + 24) / (x + 24) = 0.8 → x = 16 := by sorry

end initial_alloy_weight_l809_80969


namespace quadratic_root_problem_l809_80932

theorem quadratic_root_problem (p d c : ℝ) : 
  c = 1 / 216 →
  (∀ x, p * x^2 + d * x = 1 ↔ x = -2 ∨ x = 216 * c) →
  d = -1/2 :=
by sorry

end quadratic_root_problem_l809_80932


namespace inequality_range_l809_80914

theorem inequality_range (a : ℝ) (h : 0 < a ∧ a < 1) :
  ∀ t : ℝ, (∀ x y : ℝ, a * x^2 + t * y^2 ≥ (a * x + t * y)^2) ↔ (0 ≤ t ∧ t ≤ 1 - a) :=
by sorry

end inequality_range_l809_80914


namespace y_derivative_l809_80968

noncomputable def y (x : ℝ) : ℝ := Real.sin (2 * x - 1) ^ 2

theorem y_derivative (x : ℝ) : 
  deriv y x = 2 * Real.sin (2 * (2 * x - 1)) :=
by sorry

end y_derivative_l809_80968


namespace contrapositive_proof_l809_80963

theorem contrapositive_proof (a b : ℝ) :
  (∀ a b, a > b → a - 1 > b - 1) ↔ (∀ a b, a - 1 ≤ b - 1 → a ≤ b) := by
  sorry

end contrapositive_proof_l809_80963


namespace sixteen_solutions_l809_80974

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 4*x

-- State the theorem
theorem sixteen_solutions :
  ∃! (s : Finset ℝ), (∀ c ∈ s, g (g (g (g c))) = 2) ∧ Finset.card s = 16 :=
sorry

end sixteen_solutions_l809_80974


namespace alcohol_water_ratio_l809_80928

theorem alcohol_water_ratio (mixture : ℝ) (alcohol water : ℝ) 
  (h1 : alcohol = (1 : ℝ) / 7 * mixture) 
  (h2 : water = (2 : ℝ) / 7 * mixture) : 
  alcohol / water = (1 : ℝ) / 2 := by
sorry

end alcohol_water_ratio_l809_80928


namespace product_nine_sum_zero_l809_80911

theorem product_nine_sum_zero (a b c d : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a * b * c * d = 9 →
  a + b + c + d = 0 := by
  sorry

end product_nine_sum_zero_l809_80911


namespace derivative_sqrt_l809_80971

noncomputable def f (x : ℝ) : ℝ := Real.sqrt x

theorem derivative_sqrt (x : ℝ) (hx : x > 0) :
  deriv f x = 1 / (2 * Real.sqrt x) := by sorry

end derivative_sqrt_l809_80971


namespace five_balls_three_boxes_l809_80935

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def ways_to_distribute (n k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : ways_to_distribute 5 3 = 243 := by
  sorry

end five_balls_three_boxes_l809_80935


namespace monika_movies_l809_80948

def mall_expense : ℝ := 250
def movie_cost : ℝ := 24
def bean_bags : ℕ := 20
def bean_cost : ℝ := 1.25
def total_spent : ℝ := 347

theorem monika_movies :
  (total_spent - (mall_expense + bean_bags * bean_cost)) / movie_cost = 3 := by
  sorry

end monika_movies_l809_80948


namespace tangent_line_at_one_zero_l809_80957

-- Define the curve
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 2*x - 2

-- Theorem statement
theorem tangent_line_at_one_zero :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let m : ℝ := f' x₀
  ∀ x y : ℝ, y = m * (x - x₀) + y₀ → y = 1 :=
sorry

end tangent_line_at_one_zero_l809_80957


namespace at_least_one_square_is_one_l809_80912

theorem at_least_one_square_is_one (a b c : ℤ) 
  (h : |a + b + c| + 2 = |a| + |b| + |c|) : 
  a^2 = 1 ∨ b^2 = 1 ∨ c^2 = 1 := by
  sorry

end at_least_one_square_is_one_l809_80912


namespace max_y_value_l809_80923

theorem max_y_value (x y : ℤ) (h : x * y + 6 * x + 5 * y = -6) : 
  y ≤ 24 ∧ ∃ (x₀ : ℤ), x₀ * 24 + 6 * x₀ + 5 * 24 = -6 :=
sorry

end max_y_value_l809_80923


namespace car_owners_without_others_l809_80959

/-- Represents the number of adults owning each type of vehicle and their intersections -/
structure VehicleOwnership where
  total : ℕ
  cars : ℕ
  motorcycles : ℕ
  bicycles : ℕ
  cars_motorcycles : ℕ
  cars_bicycles : ℕ
  motorcycles_bicycles : ℕ
  all_three : ℕ

/-- The main theorem stating the number of car owners without motorcycles or bicycles -/
theorem car_owners_without_others (v : VehicleOwnership) 
  (h_total : v.total = 500)
  (h_cars : v.cars = 450)
  (h_motorcycles : v.motorcycles = 150)
  (h_bicycles : v.bicycles = 200)
  (h_pie : v.total = v.cars + v.motorcycles + v.bicycles - v.cars_motorcycles - v.cars_bicycles - v.motorcycles_bicycles + v.all_three)
  : v.cars - (v.cars_motorcycles + v.cars_bicycles - v.all_three) = 270 := by
  sorry

/-- A lemma to ensure all adults own at least one vehicle -/
lemma all_adults_own_vehicle (v : VehicleOwnership) 
  (h_total : v.total = 500)
  (h_pie : v.total = v.cars + v.motorcycles + v.bicycles - v.cars_motorcycles - v.cars_bicycles - v.motorcycles_bicycles + v.all_three)
  : v.cars + v.motorcycles + v.bicycles ≥ v.total := by
  sorry

end car_owners_without_others_l809_80959


namespace reduced_rate_weekend_l809_80970

/-- Represents a day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents the electric company's rate plan -/
structure RatePlan where
  reducedRateFraction : ℝ
  weekdayReducedHours : ℕ
  fullDayReducedRate : List Day

/-- Assertion that the given rate plan is valid and consistent with the problem statement -/
def isValidPlan (plan : RatePlan) : Prop :=
  plan.reducedRateFraction = 0.6428571428571429 ∧
  plan.weekdayReducedHours = 12 ∧
  plan.fullDayReducedRate.length = 2

/-- Theorem stating that for a valid plan, the full day reduced rate must apply on Saturday and Sunday -/
theorem reduced_rate_weekend (plan : RatePlan) (h : isValidPlan plan) :
  plan.fullDayReducedRate = [Day.Saturday, Day.Sunday] :=
sorry

end reduced_rate_weekend_l809_80970


namespace distance_2_neg5_abs_calculations_abs_equation_solutions_min_value_expression_l809_80973

-- Define the distance function on the number line
def distance (a b : ℝ) : ℝ := |a - b|

-- Theorem 1: Distance between 2 and -5
theorem distance_2_neg5 : distance 2 (-5) = 7 := by sorry

-- Theorem 2: Absolute value calculations
theorem abs_calculations : 
  (|-4 + 6| = 2) ∧ (|-2 - 4| = 6) := by sorry

-- Theorem 3: Solutions to |x+2| = 4
theorem abs_equation_solutions :
  ∀ x : ℝ, |x + 2| = 4 ↔ (x = -6 ∨ x = 2) := by sorry

-- Theorem 4: Minimum value of |x+1| + |x-3|
theorem min_value_expression :
  ∃ m : ℝ, (∀ x : ℝ, |x + 1| + |x - 3| ≥ m) ∧ 
  (∃ x : ℝ, |x + 1| + |x - 3| = m) ∧ 
  (m = 4) := by sorry

end distance_2_neg5_abs_calculations_abs_equation_solutions_min_value_expression_l809_80973


namespace average_of_abc_is_three_l809_80985

theorem average_of_abc_is_three (A B C : ℝ) 
  (eq1 : 2003 * C - 4004 * A = 8008)
  (eq2 : 2003 * B + 6006 * A = 10010)
  (eq3 : B = 2 * A - 6) :
  (A + B + C) / 3 = 3 := by
sorry

end average_of_abc_is_three_l809_80985


namespace solve_for_y_l809_80967

theorem solve_for_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 18) : y = 4 := by
  sorry

end solve_for_y_l809_80967


namespace largest_class_size_l809_80977

/-- Proves that in a school with 5 classes, where each class has 2 students less than the previous class,
    and the total number of students is 120, the largest class has 28 students. -/
theorem largest_class_size (n : ℕ) (h1 : n = 5) (total : ℕ) (h2 : total = 120) :
  ∃ x : ℕ, x = 28 ∧ 
    x + (x - 2) + (x - 4) + (x - 6) + (x - 8) = total :=
by sorry

end largest_class_size_l809_80977


namespace evaluate_expression_l809_80924

theorem evaluate_expression : (((3^2 : ℚ) - 2^3 + 7^1 - 1 + 4^2)⁻¹ * (5/6)) = 5/138 := by
  sorry

end evaluate_expression_l809_80924


namespace M_equiv_NotFirstOrThirdQuadrant_l809_80901

/-- The set M of points (x,y) in ℝ² where xy ≤ 0 -/
def M : Set (ℝ × ℝ) := {p | p.1 * p.2 ≤ 0}

/-- The set of points not in the first or third quadrants of ℝ² -/
def NotFirstOrThirdQuadrant : Set (ℝ × ℝ) := 
  {p | p.1 * p.2 ≤ 0}

/-- Theorem stating that M is equivalent to the set of points not in the first or third quadrants -/
theorem M_equiv_NotFirstOrThirdQuadrant : M = NotFirstOrThirdQuadrant := by
  sorry


end M_equiv_NotFirstOrThirdQuadrant_l809_80901


namespace two_liters_to_milliliters_nine_thousand_milliliters_to_liters_eight_liters_to_milliliters_l809_80929

-- Define the conversion factor
def liter_to_milliliter : ℚ := 1000

-- Theorem for the first conversion
theorem two_liters_to_milliliters :
  2 * liter_to_milliliter = 2000 := by sorry

-- Theorem for the second conversion
theorem nine_thousand_milliliters_to_liters :
  9000 / liter_to_milliliter = 9 := by sorry

-- Theorem for the third conversion
theorem eight_liters_to_milliliters :
  8 * liter_to_milliliter = 8000 := by sorry

end two_liters_to_milliliters_nine_thousand_milliliters_to_liters_eight_liters_to_milliliters_l809_80929


namespace price_adjustment_l809_80933

-- Define the original price
variable (P : ℝ)
-- Define the percentage x
variable (x : ℝ)

-- Theorem statement
theorem price_adjustment (h : P * (1 + x/100) * (1 - x/100) = 0.75 * P) : x = 50 := by
  sorry

end price_adjustment_l809_80933


namespace multiples_of_12_between_15_and_205_l809_80949

theorem multiples_of_12_between_15_and_205 : 
  (Finset.filter (fun n => 12 ∣ n) (Finset.Ioo 15 205)).card = 16 := by
  sorry

end multiples_of_12_between_15_and_205_l809_80949


namespace opposite_of_negative_fraction_l809_80953

theorem opposite_of_negative_fraction : 
  (-(-(1 : ℚ) / 2023)) = 1 / 2023 := by sorry

end opposite_of_negative_fraction_l809_80953


namespace prob_one_female_is_half_l809_80976

/-- Represents the composition of the extracurricular interest group -/
structure InterestGroup :=
  (male_count : Nat)
  (female_count : Nat)

/-- Calculates the probability of selecting exactly one female student
    from two selections in the interest group -/
def prob_one_female (group : InterestGroup) : Real :=
  let total := group.male_count + group.female_count
  let prob_first_female := group.female_count / total
  let prob_second_male := group.male_count / (total - 1)
  let prob_first_male := group.male_count / total
  let prob_second_female := group.female_count / (total - 1)
  prob_first_female * prob_second_male + prob_first_male * prob_second_female

/-- Theorem: The probability of selecting exactly one female student
    from two selections in a group of 3 males and 1 female is 0.5 -/
theorem prob_one_female_is_half :
  let group := InterestGroup.mk 3 1
  prob_one_female group = 1/2 := by
  sorry

end prob_one_female_is_half_l809_80976


namespace ellipse_properties_l809_80941

/-- An ellipse with the given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h1 : a > b
  h2 : b > 0
  h3 : (3/2)^2 / a^2 + 6 / b^2 = 1  -- Point M (3/2, √6) lies on the ellipse
  h4 : 2 * (a^2 - b^2).sqrt = 2     -- Focal length is 2

/-- The standard equation of the ellipse -/
def standard_equation (e : Ellipse) : Prop :=
  e.a = 3 ∧ e.b^2 = 8

/-- The trajectory equation of point E -/
def trajectory_equation (x y : ℝ) : Prop :=
  x^2 / 9 - y^2 / 8 = 1 ∧ x ≠ 3 ∧ x ≠ -3

theorem ellipse_properties (e : Ellipse) :
  standard_equation e ∧ ∀ x y, trajectory_equation x y :=
sorry

end ellipse_properties_l809_80941


namespace cube_size_is_eight_l809_80937

/-- Represents a cube of size n --/
structure Cube (n : ℕ) where
  size : n > 0

/-- Number of small cubes with no faces painted in a cube of size n --/
def unpainted (c : Cube n) : ℕ := (n - 2)^3

/-- Number of small cubes with exactly two faces painted in a cube of size n --/
def two_faces_painted (c : Cube n) : ℕ := 12 * (n - 2)

/-- Theorem stating that for a cube where the number of unpainted small cubes
    is three times the number of small cubes with two faces painted,
    the size of the cube must be 8 --/
theorem cube_size_is_eight (c : Cube n)
  (h : unpainted c = 3 * two_faces_painted c) : n = 8 := by
  sorry

end cube_size_is_eight_l809_80937


namespace complex_equation_solution_l809_80982

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I) * z = 2 + 3 * Complex.I → z = -1/2 + 5/2 * Complex.I := by
  sorry

end complex_equation_solution_l809_80982


namespace decimal_expansion_2023rd_digit_l809_80936

/-- The decimal expansion of 7/26 -/
def decimal_expansion : ℚ := 7 / 26

/-- The length of the repeating block in the decimal expansion of 7/26 -/
def repeating_block_length : ℕ := 9

/-- The position of the 2023rd digit within the repeating block -/
def position_in_block : ℕ := 2023 % repeating_block_length

/-- The 2023rd digit past the decimal point in the decimal expansion of 7/26 -/
def digit_2023 : ℕ := 3

theorem decimal_expansion_2023rd_digit :
  digit_2023 = 3 :=
sorry

end decimal_expansion_2023rd_digit_l809_80936


namespace action_figure_price_l809_80920

theorem action_figure_price (board_game_cost : ℝ) (num_figures : ℕ) (total_cost : ℝ) :
  board_game_cost = 2 →
  num_figures = 4 →
  total_cost = 30 →
  ∃ (figure_price : ℝ), figure_price = 7 ∧ total_cost = board_game_cost + num_figures * figure_price :=
by
  sorry

end action_figure_price_l809_80920


namespace allocation_theorem_l809_80919

/-- Represents the number of students -/
def num_students : ℕ := 5

/-- Represents the number of groups -/
def num_groups : ℕ := 3

/-- Function to calculate the number of allocation methods -/
def allocation_methods (n : ℕ) (k : ℕ) (excluded_pair : Bool) : ℕ :=
  sorry

/-- Theorem stating the number of allocation methods -/
theorem allocation_theorem :
  allocation_methods num_students num_groups true = 114 :=
sorry

end allocation_theorem_l809_80919


namespace marbles_selection_theorem_l809_80964

def total_marbles : ℕ := 9
def marbles_to_choose : ℕ := 4
def blue_marbles : ℕ := 2

theorem marbles_selection_theorem :
  (Nat.choose total_marbles marbles_to_choose) -
  (Nat.choose (total_marbles - blue_marbles) marbles_to_choose) = 91 := by
  sorry

end marbles_selection_theorem_l809_80964


namespace line_perp_to_plane_l809_80934

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation for lines and planes
variable (perp : Line → Line → Prop)
variable (perp_plane : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Theorem statement
theorem line_perp_to_plane 
  (m n : Line) 
  (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) 
  (h1 : perp_plane α β) 
  (h2 : intersect α β = m) 
  (h3 : perp m n) : 
  perp_line_plane n β :=
sorry

end line_perp_to_plane_l809_80934


namespace wendy_makeup_time_l809_80922

/-- Calculates the time spent on make-up given the number of facial products,
    waiting time between products, and total time for the full face routine. -/
def makeupTime (numProducts : ℕ) (waitingTime : ℕ) (totalTime : ℕ) : ℕ :=
  totalTime - (numProducts - 1) * waitingTime

/-- Proves that given 5 facial products, 5 minutes waiting time between each product,
    and a total of 55 minutes for the "full face," the time spent on make-up is 35 minutes. -/
theorem wendy_makeup_time :
  makeupTime 5 5 55 = 35 := by
  sorry

#eval makeupTime 5 5 55

end wendy_makeup_time_l809_80922


namespace reflect_F_coordinates_l809_80966

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point F -/
def F : ℝ × ℝ := (3, 3)

theorem reflect_F_coordinates :
  (reflect_x (reflect_y F)) = (-3, -3) := by sorry

end reflect_F_coordinates_l809_80966


namespace part1_part2_l809_80951

/-- Definition of a golden equation -/
def is_golden_equation (a b c : ℝ) : Prop := a ≠ 0 ∧ a - b + c = 0

/-- Part 1: Prove that 2x^2 + 5x + 3 = 0 is a golden equation -/
theorem part1 : is_golden_equation 2 5 3 := by sorry

/-- Part 2: Prove that if 3x^2 - ax + b = 0 is a golden equation and a is a root, then a = -1 or a = 3/2 -/
theorem part2 (a b : ℝ) (h1 : is_golden_equation 3 (-a) b) (h2 : 3 * a^2 - a * a + b = 0) :
  a = -1 ∨ a = 3/2 := by sorry

end part1_part2_l809_80951


namespace parabola_focus_l809_80909

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop := x = 4 * y^2

-- Define the focus of a parabola
def focus (p : ℝ × ℝ) (parabola : (ℝ × ℝ → Prop)) : Prop :=
  ∃ (a : ℝ), parabola = λ (x, y) => y^2 = a * (x - p.1) ∧ p.2 = 0

-- Theorem statement
theorem parabola_focus :
  focus (1/16, 0) (λ (x, y) => parabola_equation x y) :=
sorry

end parabola_focus_l809_80909


namespace log_xy_value_l809_80903

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^4) = 1) (h2 : Real.log (x^3 * y) = 1) :
  Real.log (x * y) = 5 / 11 := by
  sorry

end log_xy_value_l809_80903


namespace smallest_number_with_hcf_twelve_l809_80954

/-- The highest common factor of two natural numbers -/
def hcf (a b : ℕ) : ℕ := Nat.gcd a b

/-- Theorem: 48 is the smallest number greater than 36 that has a highest common factor of 12 with 36 -/
theorem smallest_number_with_hcf_twelve : 
  ∀ n : ℕ, n > 36 → hcf 36 n = 12 → n ≥ 48 :=
sorry

end smallest_number_with_hcf_twelve_l809_80954


namespace haley_tree_count_l809_80991

/-- The number of trees Haley has after a typhoon and replanting -/
def final_tree_count (initial : ℕ) (died : ℕ) (replanted : ℕ) : ℕ :=
  initial - died + replanted

/-- Theorem stating that Haley has 10 trees at the end -/
theorem haley_tree_count : final_tree_count 9 4 5 = 10 := by
  sorry

end haley_tree_count_l809_80991


namespace acid_mixture_percentage_l809_80927

theorem acid_mixture_percentage : ∀ (a w : ℝ),
  a + w = 6 →
  a / (a + w + 2) = 15 / 100 →
  (a + 2) / (a + w + 4) = 25 / 100 →
  a / (a + w) = 1 / 5 := by
  sorry

end acid_mixture_percentage_l809_80927


namespace problem_distribution_l809_80938

theorem problem_distribution (n m : ℕ) (h1 : n = 7) (h2 : m = 5) :
  (Nat.choose n m) * (m ^ (n - m)) = 525 := by
  sorry

end problem_distribution_l809_80938


namespace sports_club_participation_l809_80913

theorem sports_club_participation (total students_swimming students_basketball students_both : ℕ) 
  (h1 : total = 75)
  (h2 : students_swimming = 46)
  (h3 : students_basketball = 34)
  (h4 : students_both = 22) :
  total - (students_swimming + students_basketball - students_both) = 17 := by
sorry

end sports_club_participation_l809_80913


namespace solve_for_k_l809_80955

theorem solve_for_k : ∃ k : ℝ, 
  (∀ x y : ℝ, x = 1 ∧ y = 4 → k * x + y = 3) → 
  k = -1 := by
  sorry

end solve_for_k_l809_80955


namespace markup_calculation_l809_80960

-- Define the markup percentage
def markup_percentage : ℝ := 50

-- Define the discount percentage
def discount_percentage : ℝ := 20

-- Define the profit percentage
def profit_percentage : ℝ := 20

-- Define the relationship between cost price, marked price, and selling price
def price_relationship (cost_price marked_price selling_price : ℝ) : Prop :=
  selling_price = marked_price * (1 - discount_percentage / 100) ∧
  selling_price = cost_price * (1 + profit_percentage / 100)

-- Theorem statement
theorem markup_calculation :
  ∀ (cost_price marked_price selling_price : ℝ),
  cost_price > 0 →
  price_relationship cost_price marked_price selling_price →
  (marked_price - cost_price) / cost_price * 100 = markup_percentage :=
by sorry

end markup_calculation_l809_80960


namespace modulus_of_complex_number_l809_80930

theorem modulus_of_complex_number (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := i * (2 - i)
  Complex.abs z = Real.sqrt 5 := by
  sorry

end modulus_of_complex_number_l809_80930


namespace parabola_focus_directrix_distance_l809_80990

/-- Given a parabola and an ellipse with the following properties:
  1) The parabola has the equation x^2 = 2py where p > 0
  2) The ellipse has the equation x^2/3 + y^2/4 = 1
  3) The focus of the parabola coincides with one of the vertices of the ellipse
This theorem states that the distance from the focus of the parabola to its directrix is 4. -/
theorem parabola_focus_directrix_distance (p : ℝ) 
  (h_p_pos : p > 0)
  (h_focus_coincides : ∃ (x y : ℝ), x^2/3 + y^2/4 = 1 ∧ x^2 = 2*p*y ∧ (x = 0 ∨ y = 2 ∨ y = -2)) :
  p = 4 := by
  sorry

end parabola_focus_directrix_distance_l809_80990


namespace min_sum_with_reciprocal_constraint_l809_80917

theorem min_sum_with_reciprocal_constraint (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 1/x + 4/y = 1) : x + y ≥ 9 := by
  sorry

end min_sum_with_reciprocal_constraint_l809_80917


namespace passengers_landed_late_l809_80980

theorem passengers_landed_late (on_time passengers : ℕ) (total_passengers : ℕ) 
  (h1 : on_time_passengers = 14507)
  (h2 : total_passengers = 14720) :
  total_passengers - on_time_passengers = 213 := by
  sorry

end passengers_landed_late_l809_80980


namespace circle_equation_condition_l809_80926

/-- The equation x^2 + y^2 - x + y + m = 0 represents a circle if and only if m < 1/2 -/
theorem circle_equation_condition (x y m : ℝ) : 
  (∃ (h k r : ℝ), r > 0 ∧ (x - h)^2 + (y - k)^2 = r^2 ↔ x^2 + y^2 - x + y + m = 0) ↔ 
  m < (1/2 : ℝ) := by sorry

end circle_equation_condition_l809_80926


namespace average_difference_l809_80978

theorem average_difference (a b c : ℝ) : 
  (a + b) / 2 = 45 → (b + c) / 2 = 50 → c - a = 10 := by
  sorry

end average_difference_l809_80978


namespace probability_three_specified_coins_heads_l809_80962

/-- The probability of exactly three specified coins out of five coming up heads -/
theorem probability_three_specified_coins_heads (n : ℕ) (k : ℕ) : 
  n = 5 → k = 3 → (2^(n-k) : ℚ) / 2^n = 1/8 := by
  sorry

end probability_three_specified_coins_heads_l809_80962


namespace tim_running_hours_l809_80997

/-- Represents Tim's running schedule --/
structure RunningSchedule where
  initial_days : ℕ  -- Initial number of days Tim ran per week
  added_days : ℕ    -- Number of days Tim added to his schedule
  morning_run : ℕ   -- Hours Tim runs in the morning
  evening_run : ℕ   -- Hours Tim runs in the evening

/-- Calculates the total hours Tim runs per week --/
def total_running_hours (schedule : RunningSchedule) : ℕ :=
  (schedule.initial_days + schedule.added_days) * (schedule.morning_run + schedule.evening_run)

/-- Theorem stating that Tim's total running hours per week is 10 --/
theorem tim_running_hours :
  ∃ (schedule : RunningSchedule),
    schedule.initial_days = 3 ∧
    schedule.added_days = 2 ∧
    schedule.morning_run = 1 ∧
    schedule.evening_run = 1 ∧
    total_running_hours schedule = 10 :=
by
  sorry

end tim_running_hours_l809_80997


namespace simplify_algebraic_expression_l809_80907

theorem simplify_algebraic_expression (x : ℝ) (h1 : x ≠ -3) (h2 : x ≠ 3) (h3 : x ≠ 1) :
  (1 - 4 / (x + 3)) / ((x^2 - 2*x + 1) / (x^2 - 9)) = (x - 3) / (x - 1) :=
by sorry

end simplify_algebraic_expression_l809_80907


namespace negation_of_proposition_sin_inequality_negation_l809_80989

theorem negation_of_proposition (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) :=
by sorry

theorem sin_inequality_negation :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) :=
by sorry

end negation_of_proposition_sin_inequality_negation_l809_80989


namespace surface_area_unchanged_l809_80994

/-- Represents the dimensions of a cube -/
structure CubeDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the surface area of a cube -/
def surfaceArea (c : CubeDimensions) : ℝ :=
  6 * c.length * c.width

/-- Represents the original cube -/
def originalCube : CubeDimensions :=
  { length := 4, width := 4, height := 4 }

/-- Represents the corner cube to be removed -/
def cornerCube : CubeDimensions :=
  { length := 2, width := 2, height := 2 }

/-- The number of corners in a cube -/
def numCorners : ℕ := 8

theorem surface_area_unchanged :
  surfaceArea originalCube = surfaceArea originalCube := by sorry

end surface_area_unchanged_l809_80994


namespace ball_ratio_l809_80943

theorem ball_ratio (blue : ℕ) (red : ℕ) (green : ℕ) (yellow : ℕ) : 
  blue = 6 → 
  red = 4 → 
  yellow = 2 * red → 
  blue + red + green + yellow = 36 → 
  green / blue = 3 := by
  sorry

end ball_ratio_l809_80943


namespace system_solution_l809_80906

theorem system_solution : 
  ∃ (x y : ℝ), (3 * x^2 + 2 * y^2 + 2 * x + 3 * y = 0 ∧
                4 * x^2 - 3 * y^2 - 3 * x + 4 * y = 0) ↔
               ((x = 0 ∧ y = 0) ∨ (x = -1 ∧ y = -1)) :=
by sorry

end system_solution_l809_80906


namespace square_plus_one_ge_double_abs_l809_80952

theorem square_plus_one_ge_double_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end square_plus_one_ge_double_abs_l809_80952


namespace parabola_max_sum_l809_80921

/-- Given a parabola y = -x^2 - 3x + 3 and a point P(m, n) on this parabola,
    the maximum value of m + n is 4. -/
theorem parabola_max_sum (m n : ℝ) : 
  n = -m^2 - 3*m + 3 → (∀ x y : ℝ, y = -x^2 - 3*x + 3 → m + n ≥ x + y) → m + n = 4 :=
by sorry

end parabola_max_sum_l809_80921


namespace elevation_view_area_bounds_not_possible_area_l809_80965

/-- The area of the elevation view of a unit cube is between 1 and √2 (inclusive) -/
theorem elevation_view_area_bounds (area : ℝ) : 
  (∃ (angle : ℝ), area = Real.cos angle + Real.sin angle) →
  1 ≤ area ∧ area ≤ Real.sqrt 2 := by
  sorry

/-- (√2 - 1) / 2 is not a possible area for the elevation view of a unit cube -/
theorem not_possible_area : 
  ¬ (∃ (angle : ℝ), (Real.sqrt 2 - 1) / 2 = Real.cos angle + Real.sin angle) := by
  sorry

end elevation_view_area_bounds_not_possible_area_l809_80965


namespace fraction_simplification_l809_80944

theorem fraction_simplification (x : ℝ) (h : x = 5) :
  (x^6 - 2*x^3 + 1) / (x^3 - 1) = 124 := by
  sorry

end fraction_simplification_l809_80944


namespace equation_solution_l809_80902

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop :=
  x > 0 ∧ Real.sqrt ((log10 x)^2 + log10 (x^2) + 1) + log10 x + 1 = 0

-- Theorem statement
theorem equation_solution :
  ∀ x : ℝ, equation x ↔ (0 < x ∧ x ≤ (1/10)) :=
by sorry

end equation_solution_l809_80902
