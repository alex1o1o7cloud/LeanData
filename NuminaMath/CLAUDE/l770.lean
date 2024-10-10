import Mathlib

namespace swimming_pool_capacity_l770_77087

theorem swimming_pool_capacity (initial_fraction : ℚ) (added_amount : ℚ) (final_fraction : ℚ) :
  initial_fraction = 1/3 →
  added_amount = 180 →
  final_fraction = 4/5 →
  ∃ (total_capacity : ℚ), 
    total_capacity * initial_fraction + added_amount = total_capacity * final_fraction ∧
    total_capacity = 2700/7 := by
  sorry

#eval (2700 : ℚ) / 7

end swimming_pool_capacity_l770_77087


namespace xyz_sum_l770_77067

theorem xyz_sum (x y z : ℕ+) 
  (h : (x + y + z : ℕ+)^3 - x^3 - y^3 - z^3 = 300) : 
  (x : ℕ) + y + z = 7 := by
sorry

end xyz_sum_l770_77067


namespace pastry_eating_time_l770_77071

/-- The time it takes for two people to eat a certain number of pastries together -/
def eating_time (quick_rate : ℚ) (slow_rate : ℚ) (total_pastries : ℚ) : ℚ :=
  total_pastries / (quick_rate + slow_rate)

/-- Theorem stating the time it takes Miss Quick and Miss Slow to eat 5 pastries together -/
theorem pastry_eating_time :
  let quick_rate : ℚ := 1 / 15
  let slow_rate : ℚ := 1 / 25
  let total_pastries : ℚ := 5
  eating_time quick_rate slow_rate total_pastries = 375 / 8 := by
sorry

end pastry_eating_time_l770_77071


namespace complement_union_A_B_l770_77064

def A : Set ℝ := {x : ℝ | x ≤ 0}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 1}

theorem complement_union_A_B : 
  (A ∪ B)ᶜ = {x : ℝ | x > 1} :=
sorry

end complement_union_A_B_l770_77064


namespace car_speed_problem_l770_77058

/-- Given two cars leaving town A at the same time in the same direction,
    prove that if one car travels at 55 mph and they are 45 miles apart after 3 hours,
    then the speed of the other car must be 70 mph. -/
theorem car_speed_problem (v : ℝ) : 
  v * 3 - 55 * 3 = 45 → v = 70 := by
  sorry

end car_speed_problem_l770_77058


namespace range_of_f_l770_77002

/-- The function f(x) = x^2 - 1 --/
def f (x : ℝ) : ℝ := x^2 - 1

/-- The range of f is [-1, +∞) --/
theorem range_of_f :
  Set.range f = Set.Ici (-1) := by sorry

end range_of_f_l770_77002


namespace necessary_but_not_sufficient_l770_77072

theorem necessary_but_not_sufficient :
  ∀ a : ℝ,
  (∀ x : ℝ, x^2 + 1 > a) →
  (∃ b : ℝ, b > 0 ∧ b ≠ 1 ∧ (∀ x y : ℝ, x < y → b^x > b^y)) ∧
  (∃ c : ℝ, (∀ x : ℝ, x^2 + 1 > c) ∧ 
   ¬(∀ x y : ℝ, x < y → c^x > c^y)) :=
by sorry

end necessary_but_not_sufficient_l770_77072


namespace total_income_scientific_notation_exponent_l770_77054

/-- Represents the average annual income from 1 acre of medicinal herbs in dollars -/
def average_income_per_acre : ℝ := 20000

/-- Represents the number of acres of medicinal herbs planted in the county -/
def acres_planted : ℝ := 8000

/-- Calculates the total annual income from medicinal herbs in the county -/
def total_income : ℝ := average_income_per_acre * acres_planted

/-- Represents the exponent in the scientific notation of the total income -/
def n : ℕ := 8

/-- Theorem stating that the exponent in the scientific notation of the total income is 8 -/
theorem total_income_scientific_notation_exponent : 
  ∃ (a : ℝ), a > 1 ∧ a < 10 ∧ total_income = a * (10 : ℝ) ^ n :=
sorry

end total_income_scientific_notation_exponent_l770_77054


namespace closest_multiple_of_18_to_2500_l770_77016

/-- The multiple of 18 closest to 2500 is 2502 -/
theorem closest_multiple_of_18_to_2500 :
  ∀ n : ℤ, 18 ∣ n → |n - 2500| ≥ |2502 - 2500| :=
by
  sorry

end closest_multiple_of_18_to_2500_l770_77016


namespace certain_number_proof_l770_77046

theorem certain_number_proof (x : ℝ) : 144 / x = 14.4 / 0.0144 → x = 0.144 := by
  sorry

end certain_number_proof_l770_77046


namespace train_length_l770_77035

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed : ℝ) (time : ℝ) (h1 : speed = 56.8) (h2 : time = 18) :
  speed * time = 1022.4 := by
  sorry

end train_length_l770_77035


namespace specific_trapezoid_area_l770_77088

/-- A right trapezoid with specific properties -/
structure RightTrapezoid where
  /-- The length of one lateral side -/
  side1 : ℝ
  /-- The length of the other lateral side -/
  side2 : ℝ
  /-- The diagonal bisects the acute angle -/
  diagonal_bisects_acute_angle : Bool

/-- The area of the right trapezoid -/
def area (t : RightTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific right trapezoid is 104 -/
theorem specific_trapezoid_area :
  ∀ (t : RightTrapezoid),
    t.side1 = 10 ∧
    t.side2 = 8 ∧
    t.diagonal_bisects_acute_angle = true →
    area t = 104 :=
  sorry

end specific_trapezoid_area_l770_77088


namespace collin_cans_at_home_l770_77009

/-- The number of cans Collin found at home -/
def cans_at_home : ℕ := sorry

/-- The amount earned per can in cents -/
def cents_per_can : ℕ := 25

/-- The number of cans from the neighbor -/
def cans_from_neighbor : ℕ := 46

/-- The number of cans from dad's office -/
def cans_from_office : ℕ := 250

/-- The amount Collin has to put into savings in cents -/
def savings_amount : ℕ := 4300

theorem collin_cans_at_home :
  cans_at_home = 12 ∧
  cents_per_can * (cans_at_home + 3 * cans_at_home + cans_from_neighbor + cans_from_office) = 2 * savings_amount :=
by sorry

end collin_cans_at_home_l770_77009


namespace constant_discount_increase_l770_77038

/-- Represents the discount percentage for a given number of pizzas -/
def discount (n : ℕ) : ℚ :=
  match n with
  | 1 => 0
  | 2 => 4/100
  | 3 => 8/100
  | _ => 0  -- Default case, not used in this problem

/-- The theorem states that the discount increase is constant -/
theorem constant_discount_increase :
  ∃ (r : ℚ), (discount 2 - discount 1 = r) ∧ (discount 3 - discount 2 = r) ∧ (r = 4/100) := by
  sorry

#check constant_discount_increase

end constant_discount_increase_l770_77038


namespace octopus_gloves_bracelets_arrangements_l770_77037

/-- The number of arms an octopus has -/
def num_arms : ℕ := 8

/-- The total number of items (gloves and bracelets) -/
def total_items : ℕ := 2 * num_arms

/-- The number of valid arrangements for putting on gloves and bracelets -/
def valid_arrangements : ℕ := Nat.factorial total_items / (2^num_arms)

/-- Theorem stating the correct number of valid arrangements -/
theorem octopus_gloves_bracelets_arrangements :
  valid_arrangements = Nat.factorial total_items / (2^num_arms) :=
by sorry

end octopus_gloves_bracelets_arrangements_l770_77037


namespace original_solution_concentration_l770_77089

/-- Represents a chemical solution with a certain concentration --/
structure ChemicalSolution :=
  (concentration : ℝ)

/-- Represents a mixture of two chemical solutions --/
def mix (s1 s2 : ChemicalSolution) (ratio : ℝ) : ChemicalSolution :=
  { concentration := ratio * s1.concentration + (1 - ratio) * s2.concentration }

/-- Theorem: If half of an original solution is replaced with a 60% solution,
    resulting in a 55% solution, then the original solution was 50% --/
theorem original_solution_concentration
  (original replacement result : ChemicalSolution)
  (h1 : replacement.concentration = 0.6)
  (h2 : result = mix original replacement 0.5)
  (h3 : result.concentration = 0.55) :
  original.concentration = 0.5 := by
  sorry

#check original_solution_concentration

end original_solution_concentration_l770_77089


namespace inequality_solution_l770_77008

theorem inequality_solution (x : ℝ) : (x^2 - 49) / (x + 7) < 0 ↔ -7 < x ∧ x < 7 := by
  sorry

end inequality_solution_l770_77008


namespace sin_three_pi_fourth_minus_alpha_l770_77040

theorem sin_three_pi_fourth_minus_alpha (α : ℝ) 
  (h : Real.sin (π / 4 + α) = Real.sqrt 3 / 2) : 
  Real.sin (3 * π / 4 - α) = Real.sqrt 3 / 2 := by
  sorry

end sin_three_pi_fourth_minus_alpha_l770_77040


namespace possible_values_of_a_l770_77010

/-- Given sets A and B, where A ⊆ B, prove that a can only be 0, 1, or 1/2 -/
theorem possible_values_of_a (a : ℝ) :
  let A := {x : ℝ | a * x - 1 = 0}
  let B := {x : ℝ | x^2 - 3*x + 2 = 0}
  A ⊆ B → (a = 0 ∨ a = 1 ∨ a = 1/2) := by
  sorry


end possible_values_of_a_l770_77010


namespace lineup_combinations_l770_77030

/-- The number of ways to choose a starting lineup for a football team -/
def choose_lineup (total_players : ℕ) (offensive_linemen : ℕ) : ℕ :=
  offensive_linemen * (total_players - 1) * (total_players - 2) * (total_players - 3)

/-- Theorem stating the number of ways to choose a starting lineup for the given team composition -/
theorem lineup_combinations : choose_lineup 12 4 = 3960 := by
  sorry

end lineup_combinations_l770_77030


namespace parallelepiped_volume_l770_77082

/-- A right parallelepiped with a parallelogram base -/
structure RightParallelepiped where
  base_angle : ℝ
  base_area : ℝ
  lateral_face_area1 : ℝ
  lateral_face_area2 : ℝ

/-- The volume of the right parallelepiped -/
def volume (p : RightParallelepiped) : ℝ := sorry

theorem parallelepiped_volume (p : RightParallelepiped) 
  (h1 : p.base_angle = π / 6)
  (h2 : p.base_area = 4)
  (h3 : p.lateral_face_area1 = 6)
  (h4 : p.lateral_face_area2 = 12) :
  volume p = 12 := by sorry

end parallelepiped_volume_l770_77082


namespace power_product_equals_l770_77021

theorem power_product_equals : (3 : ℕ)^4 * (6 : ℕ)^4 = 104976 := by
  sorry

end power_product_equals_l770_77021


namespace julio_salary_julio_salary_is_500_l770_77031

/-- Calculates Julio's salary for 3 weeks based on given conditions --/
theorem julio_salary (commission_per_customer : ℕ) (first_week_customers : ℕ) 
  (bonus : ℕ) (total_earnings : ℕ) : ℕ :=
  let second_week_customers := 2 * first_week_customers
  let third_week_customers := 3 * first_week_customers
  let total_customers := first_week_customers + second_week_customers + third_week_customers
  let total_commission := total_customers * commission_per_customer
  let salary := total_earnings - (total_commission + bonus)
  salary

/-- Proves that Julio's salary for 3 weeks is $500 --/
theorem julio_salary_is_500 : 
  julio_salary 1 35 50 760 = 500 := by
  sorry

end julio_salary_julio_salary_is_500_l770_77031


namespace binomial_55_3_l770_77007

theorem binomial_55_3 : Nat.choose 55 3 = 26235 := by
  sorry

end binomial_55_3_l770_77007


namespace square_sum_difference_l770_77097

theorem square_sum_difference (a b : ℝ) 
  (h1 : (a + b)^2 = 17) 
  (h2 : (a - b)^2 = 11) : 
  a^2 + b^2 = 14 := by sorry

end square_sum_difference_l770_77097


namespace max_third_place_books_l770_77063

structure BookDistribution where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ

def is_valid_distribution (d : BookDistribution) : Prop :=
  d.first > d.second ∧
  d.second > d.third ∧
  d.third > d.fourth ∧
  d.fourth > d.fifth ∧
  d.first % 100 = 0 ∧
  d.second % 100 = 0 ∧
  d.third % 100 = 0 ∧
  d.fourth % 100 = 0 ∧
  d.fifth % 100 = 0 ∧
  d.first = d.second + d.third ∧
  d.second = d.fourth + d.fifth ∧
  d.first + d.second + d.third + d.fourth + d.fifth ≤ 10000

theorem max_third_place_books :
  ∀ d : BookDistribution,
    is_valid_distribution d →
    d.third ≤ 1900 :=
by sorry

end max_third_place_books_l770_77063


namespace spring_ice_cream_percentage_spring_ice_cream_percentage_proof_l770_77014

theorem spring_ice_cream_percentage : ℝ → Prop :=
  fun spring_percentage =>
    (spring_percentage + 30 + 25 + 20 = 100) →
    spring_percentage = 25

-- The proof is omitted
theorem spring_ice_cream_percentage_proof : spring_ice_cream_percentage 25 := by
  sorry

end spring_ice_cream_percentage_spring_ice_cream_percentage_proof_l770_77014


namespace range_of_a_l770_77081

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 1 2 ∧ x^2 + 2*x + a ≥ 0) ↔ a ∈ Set.Ici (-8) :=
sorry

end range_of_a_l770_77081


namespace problem_solution_l770_77029

def f (n : ℤ) : ℤ := 3 * n^6 + 26 * n^4 + 33 * n^2 + 1

def valid_k (k : ℕ) : Prop :=
  k ≤ 100 ∧ ∃ n : ℤ, f n % k = 0

def solution_set : Finset ℕ :=
  {9, 21, 27, 39, 49, 57, 63, 81, 87, 91, 93}

theorem problem_solution :
  ∀ k : ℕ, valid_k k ↔ k ∈ solution_set :=
sorry

end problem_solution_l770_77029


namespace parallel_vectors_m_l770_77084

def vector_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m (a b : ℝ × ℝ) (h : vector_parallel a b) :
  a = (2, -1) → b = (-1, 1/2) := by sorry

end parallel_vectors_m_l770_77084


namespace c_share_of_profit_l770_77091

/-- Calculates the share of profit for a partner in a business partnership --/
def calculate_share_of_profit (investment : ℕ) (total_investment : ℕ) (total_profit : ℕ) : ℕ :=
  (investment * total_profit) / total_investment

theorem c_share_of_profit (investment_a investment_b investment_c total_profit : ℕ) 
  (h1 : investment_a = 27000)
  (h2 : investment_b = 72000)
  (h3 : investment_c = 81000)
  (h4 : total_profit = 80000) :
  calculate_share_of_profit investment_c (investment_a + investment_b + investment_c) total_profit = 36000 :=
by
  sorry

#eval calculate_share_of_profit 81000 (27000 + 72000 + 81000) 80000

end c_share_of_profit_l770_77091


namespace multiple_birth_statistics_l770_77070

theorem multiple_birth_statistics (total_babies : ℕ) 
  (h_total : total_babies = 1200) 
  (twins triplets quintuplets : ℕ) 
  (h_twins : twins = 3 * triplets) 
  (h_triplets : triplets = 2 * quintuplets) 
  (h_sum : 2 * twins + 3 * triplets + 5 * quintuplets = total_babies) : 
  5 * quintuplets = 260 := by
  sorry

end multiple_birth_statistics_l770_77070


namespace commission_per_car_l770_77076

/-- Proves that the commission per car is $200 given the specified conditions -/
theorem commission_per_car 
  (base_salary : ℕ) 
  (march_earnings : ℕ) 
  (cars_to_double : ℕ) 
  (h1 : base_salary = 1000)
  (h2 : march_earnings = 2000)
  (h3 : cars_to_double = 15) :
  (2 * march_earnings - base_salary) / cars_to_double = 200 := by
  sorry

end commission_per_car_l770_77076


namespace animal_path_distance_l770_77096

/-- The total distance traveled by an animal along a specific path between two concentric circles -/
theorem animal_path_distance (r₁ r₂ : ℝ) (h₁ : r₁ = 15) (h₂ : r₂ = 25) :
  let outer_arc := (1/4) * 2 * Real.pi * r₂
  let radial_line := r₂ - r₁
  let inner_circle := 2 * Real.pi * r₁
  outer_arc + radial_line + inner_circle + radial_line = 42.5 * Real.pi + 20 := by
  sorry

end animal_path_distance_l770_77096


namespace system_solution_range_l770_77043

theorem system_solution_range (a x y : ℝ) : 
  x - y = a + 3 →
  2 * x + y = 5 * a →
  x < y →
  a < -3 := by
sorry

end system_solution_range_l770_77043


namespace base_conversion_subtraction_l770_77099

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

theorem base_conversion_subtraction :
  let base_5_num := [1, 3, 4, 2, 5]
  let base_8_num := [2, 3, 4, 1]
  to_base_10 base_5_num 5 - to_base_10 base_8_num 8 = 2697 := by
  sorry

end base_conversion_subtraction_l770_77099


namespace forest_tree_ratio_l770_77028

/-- Proves the ratio of trees after Monday to initial trees is 3:1 --/
theorem forest_tree_ratio : 
  ∀ (initial_trees monday_trees : ℕ),
    initial_trees = 30 →
    monday_trees + (monday_trees / 3) = 80 →
    (initial_trees + monday_trees) / initial_trees = 3 := by
  sorry

end forest_tree_ratio_l770_77028


namespace quartic_roots_l770_77036

theorem quartic_roots (x : ℝ) :
  (7 * x^4 - 50 * x^3 + 94 * x^2 - 50 * x + 7 = 0) ↔
  (x + 1/x = (50 + Real.sqrt 260)/14 ∨ x + 1/x = (50 - Real.sqrt 260)/14) :=
by sorry

end quartic_roots_l770_77036


namespace smallest_seating_arrangement_three_satisfies_seating_arrangement_smallest_M_is_three_l770_77025

theorem smallest_seating_arrangement (M : ℕ+) : (∃ (x y : ℕ+), 8 * M = 12 * x ∧ 12 * M = 8 * y ∧ x = y) → M ≥ 3 :=
by sorry

theorem three_satisfies_seating_arrangement : ∃ (x y : ℕ+), 8 * 3 = 12 * x ∧ 12 * 3 = 8 * y ∧ x = y :=
by sorry

theorem smallest_M_is_three : (∀ M : ℕ+, M < 3 → ¬(∃ (x y : ℕ+), 8 * M = 12 * x ∧ 12 * M = 8 * y ∧ x = y)) ∧
                              (∃ (x y : ℕ+), 8 * 3 = 12 * x ∧ 12 * 3 = 8 * y ∧ x = y) :=
by sorry

end smallest_seating_arrangement_three_satisfies_seating_arrangement_smallest_M_is_three_l770_77025


namespace trapezoid_angles_l770_77034

-- Define a trapezoid
structure Trapezoid where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  sum_360 : a + b + c + d = 360
  sum_ab_180 : a + b = 180
  sum_cd_180 : c + d = 180

-- Theorem statement
theorem trapezoid_angles (t : Trapezoid) (h1 : t.a = 60) (h2 : t.b = 130) :
  t.c = 50 ∧ t.d = 120 := by
  sorry

end trapezoid_angles_l770_77034


namespace min_chord_length_l770_77057

/-- The minimum length of a chord passing through (1,1) in the circle (x-2)^2 + (y-3)^2 = 9 is 4 -/
theorem min_chord_length (x y : ℝ) : 
  let circle := fun (x y : ℝ) => (x - 2)^2 + (y - 3)^2 = 9
  let point := (1, 1)
  let chord_length := fun (a b c d : ℝ) => Real.sqrt ((a - c)^2 + (b - d)^2)
  ∃ (a b c d : ℝ), 
    circle a b ∧ circle c d ∧ 
    (1 - a) * (d - b) = (1 - c) * (b - 1) ∧ 
    (∀ (e f g h : ℝ), circle e f ∧ circle g h ∧ 
      (1 - e) * (h - f) = (1 - g) * (f - 1) → 
      chord_length a b c d ≤ chord_length e f g h) ∧
    chord_length a b c d = 4 :=
by sorry

end min_chord_length_l770_77057


namespace min_value_of_expression_l770_77039

theorem min_value_of_expression : 
  ∃ (min : ℝ), min = Real.sqrt 2 * Real.sqrt 5 ∧ 
  ∀ (x : ℝ), Real.sqrt (x^2 + (1 + 2*x)^2) + Real.sqrt ((x - 1)^2 + (x - 1)^2) ≥ min :=
by sorry

end min_value_of_expression_l770_77039


namespace range_of_a1_l770_77053

/-- A sequence satisfying the given recurrence relation -/
def RecurrenceSequence (a : ℕ+ → ℝ) : Prop :=
  ∀ n : ℕ+, a (n + 1) = 2 * (|a n| - 1)

/-- The sequence is bounded by some positive constant M -/
def BoundedSequence (a : ℕ+ → ℝ) : Prop :=
  ∃ M : ℝ, M > 0 ∧ ∀ n : ℕ+, |a n| ≤ M

/-- The main theorem stating the range of a₁ -/
theorem range_of_a1 (a : ℕ+ → ℝ) 
    (h1 : RecurrenceSequence a) 
    (h2 : BoundedSequence a) : 
    -2 ≤ a 1 ∧ a 1 ≤ 2 := by
  sorry


end range_of_a1_l770_77053


namespace simplify_expression_l770_77062

theorem simplify_expression (x y : ℝ) :
  5 * x^4 + 3 * x^2 * y - 4 - 3 * x^2 * y - 3 * x^4 - 1 = 2 * x^4 - 5 := by
  sorry

end simplify_expression_l770_77062


namespace plant_arrangement_count_l770_77092

/-- Represents the number of basil plants -/
def num_basil : ℕ := 4

/-- Represents the number of tomato plants -/
def num_tomato : ℕ := 4

/-- Calculates the factorial of a natural number -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

/-- Theorem stating the number of ways to arrange the plants -/
theorem plant_arrangement_count : 
  factorial (num_basil + 1) * factorial num_tomato = 2880 := by
  sorry

end plant_arrangement_count_l770_77092


namespace heather_blocks_shared_l770_77079

/-- The number of blocks Heather shared with Jose -/
def blocks_shared (initial final : ℕ) : ℕ := initial - final

/-- Theorem stating that the number of blocks shared is the difference between initial and final counts -/
theorem heather_blocks_shared : 
  blocks_shared 86 45 = 41 := by
  sorry

end heather_blocks_shared_l770_77079


namespace fraction_subtraction_division_l770_77045

theorem fraction_subtraction_division : 
  (10 : ℚ) / 5 - (10 : ℚ) / 2 / ((2 : ℚ) / 5) = -21 / 2 := by
  sorry

end fraction_subtraction_division_l770_77045


namespace polynomial_remainder_l770_77006

/-- Given a polynomial p(x) such that p(2) = 7 and p(5) = 11,
    prove that the remainder when p(x) is divided by (x-2)(x-5) is (4/3)x + (13/3) -/
theorem polynomial_remainder (p : ℝ → ℝ) (h1 : p 2 = 7) (h2 : p 5 = 11) :
  ∃ q : ℝ → ℝ, ∀ x, p x = q x * (x - 2) * (x - 5) + (4/3 * x + 13/3) :=
sorry

end polynomial_remainder_l770_77006


namespace necessary_but_not_sufficient_l770_77056

theorem necessary_but_not_sufficient (x : ℝ) :
  (∀ x, (abs x = -x → x^2 ≥ -x)) ∧
  (∃ x, x^2 ≥ -x ∧ abs x ≠ -x) :=
sorry

end necessary_but_not_sufficient_l770_77056


namespace oleg_can_win_l770_77041

/-- The nth prime number -/
def nthPrime (n : ℕ) : ℕ := sorry

/-- A list of n positive integers, all smaller than the nth prime -/
def validList (n : ℕ) (list : List ℕ) : Prop :=
  list.length = n ∧ 
  ∀ x ∈ list, 0 < x ∧ x < nthPrime n

/-- The operation of replacing one number with the product of two numbers -/
def replaceWithProduct (list : List ℕ) (i j k : ℕ) : List ℕ :=
  sorry

/-- Predicate to check if a list contains at least two equal elements -/
def hasEqualElements (list : List ℕ) : Prop :=
  ∃ i j, i ≠ j ∧ list.get! i = list.get! j

/-- The main theorem: Oleg can always win for n > 1 -/
theorem oleg_can_win (n : ℕ) (list : List ℕ) (h : n > 1) (hlist : validList n list) :
  ∃ (steps : List (ℕ × ℕ × ℕ)), 
    let finalList := steps.foldl (fun acc step => replaceWithProduct acc step.1 step.2.1 step.2.2) list
    hasEqualElements finalList :=
  sorry

end oleg_can_win_l770_77041


namespace sin_70_cos_20_plus_cos_70_sin_20_l770_77033

theorem sin_70_cos_20_plus_cos_70_sin_20 : 
  Real.sin (70 * π / 180) * Real.cos (20 * π / 180) + 
  Real.cos (70 * π / 180) * Real.sin (20 * π / 180) = 1 := by
  sorry

end sin_70_cos_20_plus_cos_70_sin_20_l770_77033


namespace polynomial_divisibility_l770_77051

def is_divisible_by_one_of (F : ℤ → ℤ) (divisors : List ℤ) : Prop :=
  ∀ n : ℤ, ∃ a ∈ divisors, (F n) % a = 0

theorem polynomial_divisibility
  (F : ℤ → ℤ)
  (divisors : List ℤ)
  (h_polynomial : ∀ x y : ℤ, (F x - F y) % (x - y) = 0)
  (h_divisible : is_divisible_by_one_of F divisors) :
  ∃ a ∈ divisors, ∀ n : ℤ, (F n) % a = 0 :=
sorry

end polynomial_divisibility_l770_77051


namespace intersection_line_ellipse_l770_77027

/-- Prove that if a line y = kx intersects the ellipse x^2/4 + y^2/3 = 1 at points A and B, 
    and the perpendiculars from A and B to the x-axis have their feet at ±1 
    (which are the foci of the ellipse), then k = ± 3/2. -/
theorem intersection_line_ellipse (k : ℝ) : 
  (∀ x y : ℝ, y = k * x → x^2 / 4 + y^2 / 3 = 1 → 
    (x = 1 ∨ x = -1) → k = 3/2 ∨ k = -3/2) := by
  sorry


end intersection_line_ellipse_l770_77027


namespace terry_spending_l770_77066

def weekly_spending (monday : ℚ) : ℚ :=
  let tuesday := 2 * monday
  let wednesday := 2 * (monday + tuesday)
  let thursday := (monday + tuesday + wednesday) / 3
  let friday := thursday - 4
  let saturday := friday + (friday / 2)
  let sunday := tuesday + saturday
  monday + tuesday + wednesday + thursday + friday + saturday + sunday

theorem terry_spending :
  weekly_spending 6 = 140 := by sorry

end terry_spending_l770_77066


namespace expression_zero_iff_x_neg_two_l770_77015

theorem expression_zero_iff_x_neg_two (x : ℝ) :
  (x^2 - 4) / (4*x - 8) = 0 ↔ x = -2 :=
by
  sorry

end expression_zero_iff_x_neg_two_l770_77015


namespace exam_maximum_marks_l770_77090

theorem exam_maximum_marks :
  ∀ (max_marks : ℕ) (student_marks : ℕ) (failing_margin : ℕ),
    student_marks = 92 →
    failing_margin = 40 →
    (student_marks + failing_margin : ℚ) = (33 / 100) * max_marks →
    max_marks = 400 :=
by
  sorry

end exam_maximum_marks_l770_77090


namespace ball_pit_count_l770_77085

theorem ball_pit_count : ∃ (total : ℕ), 
  let red := total / 4
  let non_red := total - red
  let blue := non_red / 5
  let neither_red_nor_blue := total - red - blue
  neither_red_nor_blue = 216 ∧ total = 360 := by
sorry

end ball_pit_count_l770_77085


namespace complex_square_minus_i_l770_77073

theorem complex_square_minus_i (z : ℂ) : z = 1 + I → z^2 - I = I := by
  sorry

end complex_square_minus_i_l770_77073


namespace alien_energy_conversion_l770_77077

/-- Converts a base 7 number to base 10 --/
def base7ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The base 7 representation of the alien's energy units --/
def alienEnergy : List Nat := [3, 6, 2]

theorem alien_energy_conversion :
  base7ToBase10 alienEnergy = 143 := by
  sorry

end alien_energy_conversion_l770_77077


namespace tangent_line_intersection_l770_77094

/-- Given two circles and a line tangent to both, proves the x-coordinate of the intersection point --/
theorem tangent_line_intersection (r1 r2 c2_x : ℝ) (h1 : r1 = 2) (h2 : r2 = 7) (h3 : c2_x = 15) :
  ∃ x : ℝ, x > 0 ∧ (r1 / x = r2 / (c2_x - x)) ∧ x = 10 / 3 := by
  sorry

end tangent_line_intersection_l770_77094


namespace abs_plus_one_positive_l770_77080

theorem abs_plus_one_positive (a : ℚ) : 0 < |a| + 1 := by
  sorry

end abs_plus_one_positive_l770_77080


namespace probability_is_31_over_473_l770_77061

/-- Represents a standard deck of cards --/
def StandardDeck : ℕ := 52

/-- Number of cards per rank in a standard deck --/
def CardsPerRank : ℕ := 4

/-- Number of pairs removed (two pairs of Aces and two pairs of Kings) --/
def PairsRemoved : ℕ := 2

/-- Number of ranks affected by pair removal --/
def RanksAffected : ℕ := 2

/-- Number of unaffected ranks (from Two to Queen) --/
def UnaffectedRanks : ℕ := 11

/-- Calculates the probability of selecting a pair from the modified deck --/
def probability_of_pair (deck : ℕ) (cards_per_rank : ℕ) (pairs_removed : ℕ) (ranks_affected : ℕ) (unaffected_ranks : ℕ) : ℚ :=
  let remaining_cards := deck - 2 * pairs_removed * cards_per_rank
  let total_combinations := remaining_cards.choose 2
  let affected_pairs := ranks_affected
  let unaffected_pairs := unaffected_ranks * (cards_per_rank.choose 2)
  let favorable_outcomes := affected_pairs + unaffected_pairs
  ↑favorable_outcomes / ↑total_combinations

theorem probability_is_31_over_473 :
  probability_of_pair StandardDeck CardsPerRank PairsRemoved RanksAffected UnaffectedRanks = 31 / 473 :=
sorry

end probability_is_31_over_473_l770_77061


namespace complement_of_A_in_U_l770_77060

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {3, 5}

theorem complement_of_A_in_U :
  U \ A = {1, 2, 4} := by sorry

end complement_of_A_in_U_l770_77060


namespace min_triangle_perimeter_l770_77044

theorem min_triangle_perimeter (a b x : ℕ) (ha : a = 24) (hb : b = 51) : 
  (a + b + x > a + b ∧ a + x > b ∧ b + x > a) → (∀ y : ℕ, (a + b + y > a + b ∧ a + y > b ∧ b + y > a) → x ≤ y) 
  → a + b + x = 103 :=
sorry

end min_triangle_perimeter_l770_77044


namespace rectangle_with_tangent_circle_l770_77093

theorem rectangle_with_tangent_circle 
  (r : ℝ) 
  (h1 : r = 6) 
  (A_circle : ℝ) 
  (h2 : A_circle = π * r^2) 
  (A_rectangle : ℝ) 
  (h3 : A_rectangle = 3 * A_circle) 
  (shorter_side : ℝ) 
  (h4 : shorter_side = 2 * r) 
  (longer_side : ℝ) 
  (h5 : A_rectangle = shorter_side * longer_side) : 
  longer_side = 9 * π := by
sorry

end rectangle_with_tangent_circle_l770_77093


namespace prove_trip_length_l770_77032

def trip_length : ℚ := 360 / 7

theorem prove_trip_length :
  let first_part : ℚ := 1 / 4
  let second_part : ℚ := 30
  let third_part : ℚ := 1 / 6
  (first_part + third_part + second_part / trip_length = 1) →
  trip_length = 360 / 7 := by
sorry

end prove_trip_length_l770_77032


namespace marys_unique_score_l770_77024

/-- Represents the score in a mathematics competition. -/
structure Score where
  total : ℕ
  correct : ℕ
  wrong : ℕ
  h_total : total = 35 + 5 * correct - 2 * wrong

/-- Determines if a score is uniquely determinable. -/
def isUniqueDeterminable (s : Score) : Prop :=
  ∀ s' : Score, s'.total = s.total → s'.correct = s.correct ∧ s'.wrong = s.wrong

/-- The theorem stating Mary's unique score. -/
theorem marys_unique_score :
  ∃! s : Score,
    s.total > 90 ∧
    isUniqueDeterminable s ∧
    ∀ s' : Score, s'.total > 90 ∧ s'.total < s.total → ¬isUniqueDeterminable s' :=
by sorry

end marys_unique_score_l770_77024


namespace jellybeans_in_larger_box_l770_77020

/-- Given a box with jellybeans and another box with tripled dimensions, 
    calculate the number of jellybeans in the larger box. -/
theorem jellybeans_in_larger_box 
  (small_box_jellybeans : ℕ) 
  (scale_factor : ℕ) 
  (h1 : small_box_jellybeans = 150) 
  (h2 : scale_factor = 3) : 
  (scale_factor ^ 3 : ℕ) * small_box_jellybeans = 4050 := by
  sorry

end jellybeans_in_larger_box_l770_77020


namespace simplify_and_evaluate_l770_77059

theorem simplify_and_evaluate (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end simplify_and_evaluate_l770_77059


namespace train_trip_time_difference_l770_77069

/-- The time difference between two trips with given distance and speeds -/
theorem train_trip_time_difference 
  (distance : ℝ) 
  (speed_outbound speed_return : ℝ) 
  (h1 : distance = 480) 
  (h2 : speed_outbound = 160) 
  (h3 : speed_return = 120) : 
  distance / speed_return - distance / speed_outbound = 1 := by
sorry

end train_trip_time_difference_l770_77069


namespace pure_imaginary_condition_l770_77055

theorem pure_imaginary_condition (θ : ℝ) : 
  let z : ℂ := (Complex.exp (Complex.I * -θ)) * (1 + Complex.I)
  θ = 3 * Real.pi / 4 → Complex.re z = 0 ∧ Complex.im z ≠ 0 :=
by sorry

end pure_imaginary_condition_l770_77055


namespace circle_op_eq_power_l770_77011

noncomputable def circle_op (a : ℚ) (n : ℕ) : ℚ :=
  if n = 0 then 1 else if n = 1 then a else a / (circle_op a (n - 1))

theorem circle_op_eq_power (a : ℚ) (n : ℕ) (h : a ≠ 0) :
  circle_op a n = (1 / a) ^ (n - 2) :=
sorry

end circle_op_eq_power_l770_77011


namespace playground_boys_count_l770_77017

theorem playground_boys_count (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  total = 63 → girls = 28 → boys = total - girls → boys = 35 := by
  sorry

end playground_boys_count_l770_77017


namespace road_trip_distance_l770_77078

theorem road_trip_distance (D : ℝ) 
  (h1 : D > 0)
  (h2 : D * (1 - 1/3) * (1 - 1/5) * (1 - 1/4) = 400) : D = 1000 := by
  sorry

end road_trip_distance_l770_77078


namespace pencil_box_puzzle_l770_77005

structure Box where
  blue : ℕ
  green : ℕ

def vasya_statement (box : Box) : Prop :=
  box.blue ≥ 4

def kolya_statement (box : Box) : Prop :=
  box.green ≥ 5

def petya_statement (box : Box) : Prop :=
  box.blue ≥ 3 ∧ box.green ≥ 4

def misha_statement (box : Box) : Prop :=
  box.blue ≥ 4 ∧ box.green ≥ 4

theorem pencil_box_puzzle (box : Box) :
  (vasya_statement box ∧ ¬kolya_statement box ∧ petya_statement box ∧ misha_statement box) ↔
  (box.blue ≥ 4 ∧ box.green = 4) :=
by sorry

end pencil_box_puzzle_l770_77005


namespace danny_bottle_caps_wrappers_l770_77074

theorem danny_bottle_caps_wrappers : 
  let bottle_caps_found : ℕ := 50
  let wrappers_found : ℕ := 46
  bottle_caps_found - wrappers_found = 4 := by
  sorry

end danny_bottle_caps_wrappers_l770_77074


namespace three_cakes_cooking_time_l770_77004

/-- Represents the cooking process for cakes -/
structure CookingProcess where
  pot_capacity : ℕ
  cooking_time_per_cake : ℕ
  num_cakes : ℕ

/-- The minimum time required to cook the given number of cakes -/
def min_cooking_time (process : CookingProcess) : ℕ :=
  sorry

/-- Theorem stating the minimum time to cook three cakes under given conditions -/
theorem three_cakes_cooking_time :
  ∀ (process : CookingProcess),
    process.pot_capacity = 2 →
    process.cooking_time_per_cake = 5 →
    process.num_cakes = 3 →
    min_cooking_time process = 15 :=
by sorry

end three_cakes_cooking_time_l770_77004


namespace fraction_simplification_l770_77013

theorem fraction_simplification (x y : ℝ) (hx : -x ≥ 0) (hy : -y ≥ 0) :
  (Real.sqrt (-x) - Real.sqrt (-3 * y)) / (x + 3 * y + 2 * Real.sqrt (3 * x * y)) =
  1 / (Real.sqrt (-3 * y) - Real.sqrt (-x)) :=
by sorry

end fraction_simplification_l770_77013


namespace no_real_b_for_single_solution_l770_77052

-- Define the quadratic function g(x) with parameter b
def g (b : ℝ) (x : ℝ) : ℝ := x^2 + 3*b*x + 4*b

-- Theorem stating that no real b exists such that g(x) has its vertex at y = 5
theorem no_real_b_for_single_solution :
  ¬ ∃ b : ℝ, ∃ x : ℝ, g b x = 5 ∧ ∀ y : ℝ, g b y ≥ 5 :=
sorry

end no_real_b_for_single_solution_l770_77052


namespace first_woman_work_time_l770_77048

/-- Represents the wall-building scenario with women joining at intervals -/
structure WallBuilding where
  /-- Total time to build the wall if all women worked together -/
  totalTime : ℝ
  /-- Number of women -/
  numWomen : ℕ
  /-- Time interval between each woman joining -/
  joinInterval : ℝ
  /-- Time all women work together -/
  allWorkTime : ℝ

/-- The first woman works 5 times as long as the last woman -/
def firstLastRatio (w : WallBuilding) : Prop :=
  w.joinInterval * (w.numWomen - 1) + w.allWorkTime = 5 * w.allWorkTime

/-- The total work done is equivalent to all women working for the total time -/
def totalWorkEquivalence (w : WallBuilding) : Prop :=
  (w.joinInterval * (w.numWomen - 1) / 2 + w.allWorkTime) * w.numWomen = w.totalTime * w.numWomen

/-- Main theorem: The first woman works for 75 hours -/
theorem first_woman_work_time (w : WallBuilding) 
    (h1 : w.totalTime = 45)
    (h2 : firstLastRatio w)
    (h3 : totalWorkEquivalence w) : 
  w.joinInterval * (w.numWomen - 1) + w.allWorkTime = 75 := by
  sorry

#check first_woman_work_time

end first_woman_work_time_l770_77048


namespace weight_difference_is_35_l770_77049

def labrador_start : ℝ := 40
def dachshund_start : ℝ := 12
def weight_gain_percentage : ℝ := 0.25

def weight_difference : ℝ :=
  (labrador_start + labrador_start * weight_gain_percentage) -
  (dachshund_start + dachshund_start * weight_gain_percentage)

theorem weight_difference_is_35 : weight_difference = 35 := by
  sorry

end weight_difference_is_35_l770_77049


namespace polynomial_division_remainder_l770_77098

theorem polynomial_division_remainder (x : ℂ) : 
  (x^6 - 1) * (x^3 - 1) % (x^2 + x + 1) = 0 := by
sorry

end polynomial_division_remainder_l770_77098


namespace marias_workday_ends_at_5pm_l770_77095

/-- Represents time in 24-hour format -/
structure Time where
  hour : Nat
  minute : Nat
  deriving Repr

/-- Represents a workday -/
structure Workday where
  start : Time
  totalWorkHours : Nat
  lunchBreakStart : Time
  lunchBreakDuration : Nat
  deriving Repr

def addHours (t : Time) (h : Nat) : Time :=
  { hour := (t.hour + h) % 24, minute := t.minute }

def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hour * 60 + t.minute + m
  { hour := totalMinutes / 60, minute := totalMinutes % 60 }

def mariasWorkday : Workday :=
  { start := { hour := 8, minute := 0 },
    totalWorkHours := 8,
    lunchBreakStart := { hour := 13, minute := 0 },
    lunchBreakDuration := 1 }

theorem marias_workday_ends_at_5pm :
  let endTime := addHours (addMinutes mariasWorkday.lunchBreakStart mariasWorkday.lunchBreakDuration)
                          (mariasWorkday.totalWorkHours - (mariasWorkday.lunchBreakStart.hour - mariasWorkday.start.hour))
  endTime = { hour := 17, minute := 0 } :=
by sorry

end marias_workday_ends_at_5pm_l770_77095


namespace greatest_integer_for_all_real_domain_greatest_integer_value_l770_77000

theorem greatest_integer_for_all_real_domain (b : ℤ) : 
  (∀ x : ℝ, (x^2 + b*x + 5 ≠ 0)) ↔ b^2 < 20 :=
by sorry

theorem greatest_integer_value : 
  ∃ b : ℤ, b = 4 ∧ (∀ x : ℝ, (x^2 + b*x + 5 ≠ 0)) ∧ 
  (∀ c : ℤ, c > b → ∃ x : ℝ, (x^2 + c*x + 5 = 0)) :=
by sorry

end greatest_integer_for_all_real_domain_greatest_integer_value_l770_77000


namespace rotation_of_A_to_B_l770_77018

def rotate90CCW (x y : ℝ) : ℝ × ℝ := (-y, x)

theorem rotation_of_A_to_B :
  let A : ℝ × ℝ := (Real.sqrt 3, 1)
  let B : ℝ × ℝ := rotate90CCW A.1 A.2
  B = (-1, Real.sqrt 3) := by sorry

end rotation_of_A_to_B_l770_77018


namespace class_size_proof_l770_77047

theorem class_size_proof (boys_ratio : Nat) (girls_ratio : Nat) (num_girls : Nat) :
  boys_ratio = 5 →
  girls_ratio = 8 →
  num_girls = 160 →
  (boys_ratio + girls_ratio : Rat) * (num_girls / girls_ratio : Rat) = 260 :=
by sorry

end class_size_proof_l770_77047


namespace probability_sum_seven_l770_77042

/-- Represents the faces of the first die -/
def die1 : Finset ℕ := {1, 3, 5}

/-- Represents the faces of the second die -/
def die2 : Finset ℕ := {2, 4, 6}

/-- The total number of possible outcomes when rolling both dice -/
def total_outcomes : ℕ := 36

/-- The number of favorable outcomes (sum of 7) -/
def favorable_outcomes : ℕ := 12

/-- Theorem stating that the probability of rolling a sum of 7 is 1/3 -/
theorem probability_sum_seven :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 := by
  sorry

end probability_sum_seven_l770_77042


namespace inverse_uniqueness_non_commutative_non_unique_inverse_l770_77022

/-- A binary operation on a type α -/
def BinaryOp (α : Type) := α → α → α

/-- The inverse of a binary operation -/
def InverseOp (α : Type) (op : BinaryOp α) := BinaryOp α

/-- Property of an inverse operation -/
def IsInverse {α : Type} (op : BinaryOp α) (inv : InverseOp α op) : Prop :=
  ∀ a b c : α, op a b = c → inv c b = a ∧ op (inv c b) b = c

/-- Uniqueness of inverse operation -/
theorem inverse_uniqueness {α : Type} (op : BinaryOp α) :
  ∃! inv : InverseOp α op, IsInverse op inv :=
sorry

/-- Non-uniqueness for non-commutative operations -/
theorem non_commutative_non_unique_inverse {α : Type} (op : BinaryOp α) :
  (∃ a b : α, op a b ≠ op b a) →
  ¬∃! inv : InverseOp α op, IsInverse op inv :=
sorry

end inverse_uniqueness_non_commutative_non_unique_inverse_l770_77022


namespace exponent_addition_l770_77019

theorem exponent_addition (x : ℝ) : x^3 * x^2 = x^5 := by
  sorry

end exponent_addition_l770_77019


namespace time_to_write_rearrangements_l770_77075

/-- The time required to write all rearrangements of a name -/
theorem time_to_write_rearrangements 
  (num_letters : ℕ) 
  (rearrangements_per_minute : ℕ) 
  (h1 : num_letters = 5) 
  (h2 : rearrangements_per_minute = 15) : 
  (Nat.factorial num_letters : ℚ) / (rearrangements_per_minute * 60 : ℚ) = 2 / 15 := by
  sorry

end time_to_write_rearrangements_l770_77075


namespace tangent_circle_circumference_is_36_l770_77050

/-- Represents a geometric setup with two circular arcs and a tangent circle -/
structure GeometricSetup where
  -- The length of arc BC
  arc_length : ℝ
  -- Predicate that the arcs subtend 90° angles
  subtend_right_angle : Prop
  -- Predicate that the circle is tangent to both arcs and line segment AB
  circle_tangent : Prop

/-- The circumference of the tangent circle in the given geometric setup -/
def tangent_circle_circumference (setup : GeometricSetup) : ℝ :=
  sorry

/-- Theorem stating that the circumference of the tangent circle is 36 -/
theorem tangent_circle_circumference_is_36 (setup : GeometricSetup) 
  (h1 : setup.arc_length = 18)
  (h2 : setup.subtend_right_angle)
  (h3 : setup.circle_tangent) :
  tangent_circle_circumference setup = 36 :=
sorry

end tangent_circle_circumference_is_36_l770_77050


namespace books_read_in_week_l770_77003

/-- The number of books Mrs. Hilt reads per day -/
def books_per_day : ℕ := 2

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Theorem: Mrs. Hilt reads 14 books in one week -/
theorem books_read_in_week : books_per_day * days_in_week = 14 := by
  sorry

end books_read_in_week_l770_77003


namespace clarence_spent_12_96_l770_77026

/-- The cost of Clarence's amusement park visit -/
def clarence_total_cost (cost_per_ride : ℚ) (water_slide_rides : ℕ) (roller_coaster_rides : ℕ) : ℚ :=
  cost_per_ride * (water_slide_rides + roller_coaster_rides)

/-- Theorem stating that Clarence's total cost at the amusement park was $12.96 -/
theorem clarence_spent_12_96 :
  clarence_total_cost 2.16 3 3 = 12.96 := by
  sorry

end clarence_spent_12_96_l770_77026


namespace symmetry_properties_l770_77023

-- Define the shapes
inductive Shape
  | Parallelogram
  | Rectangle
  | Square
  | Rhombus
  | IsoscelesTrapezoid

-- Define the symmetry properties
def isAxisymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => true
  | Shape.Square => true
  | Shape.Rhombus => true
  | Shape.IsoscelesTrapezoid => true
  | _ => false

def isCentrallySymmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => true
  | Shape.Rectangle => true
  | Shape.Square => true
  | Shape.Rhombus => true
  | _ => false

-- Theorem statement
theorem symmetry_properties :
  ∀ s : Shape,
    (isAxisymmetric s ∧ isCentrallySymmetric s) ↔
    (s = Shape.Rectangle ∨ s = Shape.Square ∨ s = Shape.Rhombus) :=
by sorry

end symmetry_properties_l770_77023


namespace outfit_count_l770_77065

/-- Represents the number of shirts available -/
def num_shirts : ℕ := 7

/-- Represents the number of pants available -/
def num_pants : ℕ := 5

/-- Represents the number of ties available -/
def num_ties : ℕ := 4

/-- Represents the total number of tie options (including the option of not wearing a tie) -/
def tie_options : ℕ := num_ties + 1

/-- Calculates the total number of possible outfits -/
def total_outfits : ℕ := num_shirts * num_pants * tie_options

/-- Theorem stating that the total number of possible outfits is 175 -/
theorem outfit_count : total_outfits = 175 := by sorry

end outfit_count_l770_77065


namespace tangent_length_from_point_to_circle_l770_77068

/-- The length of the tangent from a point to a circle -/
theorem tangent_length_from_point_to_circle 
  (P : ℝ × ℝ) -- Point P
  (center : ℝ × ℝ) -- Center of the circle
  (r : ℝ) -- Radius of the circle
  (h1 : P = (2, 3)) -- P coordinates
  (h2 : center = (0, 0)) -- Circle center
  (h3 : r = 1) -- Circle radius
  (h4 : (P.1 - center.1)^2 + (P.2 - center.2)^2 > r^2) -- P is outside the circle
  : Real.sqrt ((P.1 - center.1)^2 + (P.2 - center.2)^2 - r^2) = 2 * Real.sqrt 3 :=
by sorry

end tangent_length_from_point_to_circle_l770_77068


namespace rs_value_l770_77086

theorem rs_value (r s : ℝ) (hr : 0 < r) (hs : 0 < s) 
  (h1 : r^2 + s^2 = 2) (h2 : r^4 + s^4 = 9/8) : r * s = Real.sqrt 23 / 4 := by
  sorry

end rs_value_l770_77086


namespace simultaneous_equations_solution_l770_77012

theorem simultaneous_equations_solution (m : ℝ) : 
  ∃ (x y : ℝ), y = 3 * m * x + 5 ∧ y = (3 * m - 2) * x + 7 :=
by sorry

end simultaneous_equations_solution_l770_77012


namespace greatest_difference_units_digit_l770_77083

theorem greatest_difference_units_digit (x : ℕ) :
  x < 10 →
  (840 + x) % 3 = 0 →
  ∃ y, y < 10 ∧ (840 + y) % 3 = 0 ∧ 
  ∀ z, z < 10 → (840 + z) % 3 = 0 → (max x y - min x y) ≥ (max x z - min x z) :=
by sorry

end greatest_difference_units_digit_l770_77083


namespace regular_tire_usage_l770_77001

theorem regular_tire_usage
  (total_miles : ℕ)
  (spare_miles : ℕ)
  (regular_tires : ℕ)
  (h1 : total_miles = 50000)
  (h2 : spare_miles = 2000)
  (h3 : regular_tires = 4) :
  (total_miles - spare_miles) / regular_tires = 12000 :=
by sorry

end regular_tire_usage_l770_77001
