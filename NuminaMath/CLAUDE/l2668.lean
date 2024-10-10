import Mathlib

namespace element_14_is_si_l2668_266888

/-- Represents chemical elements -/
inductive Element : Type
| helium : Element
| lithium : Element
| silicon : Element
| argon : Element

/-- Returns the atomic number of an element -/
def atomic_number (e : Element) : ℕ :=
  match e with
  | Element.helium => 2
  | Element.lithium => 3
  | Element.silicon => 14
  | Element.argon => 18

/-- Returns the symbol of an element -/
def symbol (e : Element) : String :=
  match e with
  | Element.helium => "He"
  | Element.lithium => "Li"
  | Element.silicon => "Si"
  | Element.argon => "Ar"

/-- Theorem: The symbol for the element with atomic number 14 is Si -/
theorem element_14_is_si :
  ∃ (e : Element), atomic_number e = 14 ∧ symbol e = "Si" :=
by
  sorry

end element_14_is_si_l2668_266888


namespace smallest_product_of_factors_l2668_266858

theorem smallest_product_of_factors (a b : ℕ) : 
  a ≠ b → 
  a > 0 → 
  b > 0 → 
  (∃ k : ℕ, k * a = 48) → 
  (∃ l : ℕ, l * b = 48) → 
  ¬(∃ m : ℕ, m * (a * b) = 48) → 
  (∀ c d : ℕ, c ≠ d → c > 0 → d > 0 → 
    (∃ k : ℕ, k * c = 48) → 
    (∃ l : ℕ, l * d = 48) → 
    ¬(∃ m : ℕ, m * (c * d) = 48) → 
    a * b ≤ c * d) → 
  a * b = 32 := by
sorry

end smallest_product_of_factors_l2668_266858


namespace derivative_of_cube_root_l2668_266870

theorem derivative_of_cube_root (x : ℝ) (h : x > 0) : 
  deriv (λ x => Real.sqrt (x^3)) x = (3/2) * Real.sqrt x := by
  sorry

end derivative_of_cube_root_l2668_266870


namespace only_99th_statement_true_l2668_266882

/-- Represents a statement in the notebook -/
def Statement (n : ℕ) := "There are exactly n false statements in this notebook"

/-- The total number of statements in the notebook -/
def totalStatements : ℕ := 100

/-- A function that determines if a statement is true -/
def isTrue (n : ℕ) : Prop := 
  n ≤ totalStatements ∧ (totalStatements - n) = 1

theorem only_99th_statement_true : 
  ∃! n : ℕ, n ≤ totalStatements ∧ isTrue n ∧ n = 99 := by
  sorry

#check only_99th_statement_true

end only_99th_statement_true_l2668_266882


namespace remi_and_father_seedlings_l2668_266834

/-- The number of seedlings Remi's father planted -/
def fathers_seedlings (day1 : ℕ) (total : ℕ) : ℕ :=
  total - (day1 + 2 * day1)

theorem remi_and_father_seedlings :
  fathers_seedlings 200 1200 = 600 := by
  sorry

end remi_and_father_seedlings_l2668_266834


namespace total_addresses_l2668_266802

/-- The number of commencement addresses given by each governor -/
structure GovernorAddresses where
  sandoval : ℕ
  hawkins : ℕ
  sloan : ℕ
  davenport : ℕ
  adkins : ℕ

/-- The conditions of the problem -/
def problem_conditions (g : GovernorAddresses) : Prop :=
  g.sandoval = 12 ∧
  g.hawkins = g.sandoval / 2 ∧
  g.sloan = g.sandoval + 10 ∧
  g.davenport = (g.sandoval + g.sloan) / 2 - 3 ∧
  g.adkins = g.hawkins + g.davenport + 2

/-- The theorem to be proved -/
theorem total_addresses (g : GovernorAddresses) :
  problem_conditions g →
  g.sandoval + g.hawkins + g.sloan + g.davenport + g.adkins = 70 :=
by sorry

end total_addresses_l2668_266802


namespace cube_root_of_sum_l2668_266886

theorem cube_root_of_sum (a b : ℝ) : 
  Real.sqrt (a - 1) + Real.sqrt ((9 + b)^2) = 0 → (a + b)^(1/3 : ℝ) = -2 :=
by sorry

end cube_root_of_sum_l2668_266886


namespace sector_central_angle_l2668_266832

theorem sector_central_angle (area : Real) (radius : Real) (centralAngle : Real) :
  area = 3 * Real.pi / 8 →
  radius = 1 →
  centralAngle = area * 2 / (radius ^ 2) →
  centralAngle = 3 * Real.pi / 4 := by
  sorry

end sector_central_angle_l2668_266832


namespace equation_to_parabola_l2668_266829

/-- The equation y^4 - 16x^2 = 2y^2 - 64 can be transformed into a parabolic form -/
theorem equation_to_parabola :
  ∃ (a b c : ℝ), ∀ (x y : ℝ),
    y^4 - 16*x^2 = 2*y^2 - 64 →
    ∃ (t : ℝ), y^2 = a*x + b*t + c :=
by sorry

end equation_to_parabola_l2668_266829


namespace min_point_is_correct_l2668_266889

/-- The equation of the transformed graph -/
def f (x : ℝ) : ℝ := 2 * |x - 4| - 1

/-- The minimum point of the transformed graph -/
def min_point : ℝ × ℝ := (-4, -1)

/-- Theorem: The minimum point of the transformed graph is (-4, -1) -/
theorem min_point_is_correct :
  ∀ x : ℝ, f x ≥ f (min_point.1) ∧ f (min_point.1) = min_point.2 :=
by sorry

end min_point_is_correct_l2668_266889


namespace repeating_decimal_36_equals_4_11_l2668_266873

/-- The decimal expansion 0.363636... (infinitely repeating 36) is equal to 4/11 -/
theorem repeating_decimal_36_equals_4_11 : ∃ (x : ℚ), x = 4/11 ∧ x = ∑' n, 36 / (100 ^ (n + 1)) :=
by sorry

end repeating_decimal_36_equals_4_11_l2668_266873


namespace cubic_equation_three_distinct_roots_l2668_266860

theorem cubic_equation_three_distinct_roots (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    x^3 - 3*x^2 - a = 0 ∧
    y^3 - 3*y^2 - a = 0 ∧
    z^3 - 3*z^2 - a = 0) ↔
  -4 < a ∧ a < 0 := by
sorry

end cubic_equation_three_distinct_roots_l2668_266860


namespace prob_five_odd_in_six_rolls_l2668_266896

/-- The probability of rolling an odd number on a fair 6-sided die -/
def p_odd : ℚ := 1/2

/-- The number of rolls -/
def n : ℕ := 6

/-- The number of successful outcomes (rolls with odd numbers) -/
def k : ℕ := 5

/-- The probability of getting exactly k odd numbers in n rolls of a fair 6-sided die -/
def prob_k_odd_in_n_rolls (p : ℚ) (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

theorem prob_five_odd_in_six_rolls :
  prob_k_odd_in_n_rolls p_odd n k = 3/32 := by
  sorry

end prob_five_odd_in_six_rolls_l2668_266896


namespace sequence_properties_l2668_266826

-- Define the sequence a_n and its partial sum S_n
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := 2 * a n - 2^n

theorem sequence_properties (a : ℕ → ℝ) :
  (∀ n, S n a = 2 * a n - 2^n) →
  (∃ r : ℝ, ∀ n, a (n + 1) - 2 * a n = r * (a n - 2 * a (n - 1))) ∧
  (∀ n, a n = (n + 1) * 2^(n - 1)) :=
by sorry

end sequence_properties_l2668_266826


namespace two_digit_integer_count_l2668_266895

/-- A function that counts the number of three-digit integers less than 1000 with exactly two different digits. -/
def count_two_digit_integers : ℕ :=
  let case1 := 9  -- Numbers with one digit as zero
  let case2 := 9 * 9 * 3  -- Numbers with two non-zero digits
  case1 + case2

/-- Theorem stating that the count of three-digit integers less than 1000 with exactly two different digits is 252. -/
theorem two_digit_integer_count : count_two_digit_integers = 252 := by
  sorry

end two_digit_integer_count_l2668_266895


namespace raccoon_lock_ratio_l2668_266811

/-- Proves that the ratio of time both locks stall raccoons to time second lock alone stalls raccoons is 5 -/
theorem raccoon_lock_ratio : 
  let first_lock_time : ℕ := 5
  let second_lock_time : ℕ := 3 * first_lock_time - 3
  let both_locks_time : ℕ := 60
  both_locks_time / second_lock_time = 5 := by
  sorry

end raccoon_lock_ratio_l2668_266811


namespace one_and_one_third_problem_l2668_266810

theorem one_and_one_third_problem :
  ∃ x : ℚ, (4 / 3) * x = 45 ∧ x = 33.75 := by
  sorry

end one_and_one_third_problem_l2668_266810


namespace max_value_of_sine_plus_one_l2668_266815

theorem max_value_of_sine_plus_one :
  ∀ x : ℝ, 1 + Real.sin x ≤ 2 ∧ ∃ x : ℝ, 1 + Real.sin x = 2 := by
  sorry

end max_value_of_sine_plus_one_l2668_266815


namespace marble_sculpture_weight_l2668_266822

theorem marble_sculpture_weight (W : ℝ) : 
  W > 0 →
  (1 - 0.3) * (1 - 0.2) * (1 - 0.25) * W = 105 →
  W = 250 := by
sorry

end marble_sculpture_weight_l2668_266822


namespace problem_statement_l2668_266825

noncomputable def f (x : ℝ) : ℝ := x * Real.log x - x
noncomputable def g (x a : ℝ) : ℝ := Real.log x - a * x + 1

theorem problem_statement :
  (∀ x : ℝ, x > 0 → deriv f x = Real.log x) ∧
  (∀ a : ℝ, (∀ x : ℝ, x > 0 → g x a ≤ 0) → a ≥ 1) ∧
  (∀ m x n : ℝ, 0 < m → m < x → x < n →
    (f x - f m) / (x - m) < (f x - f n) / (x - n)) := by
  sorry

end problem_statement_l2668_266825


namespace median_of_special_list_l2668_266814

/-- Represents the special list where each number n from 1 to 100 appears n times -/
def special_list : List ℕ := sorry

/-- The length of the special list -/
def list_length : ℕ := (List.range 100).sum + 100

/-- The median of a list is the average of the middle two elements when the list has even length -/
def median (l : List ℕ) : ℚ := sorry

theorem median_of_special_list : median special_list = 71 := by sorry

end median_of_special_list_l2668_266814


namespace yanni_remaining_money_l2668_266878

/-- Calculates the remaining money in cents after Yanni's transactions --/
def remaining_money_in_cents (initial_money : ℚ) (mother_gave : ℚ) (found_money : ℚ) (toy_cost : ℚ) : ℕ :=
  let total_money := initial_money + mother_gave + found_money
  let remaining_money := total_money - toy_cost
  (remaining_money * 100).floor.toNat

/-- Proves that Yanni has 15 cents left after his transactions --/
theorem yanni_remaining_money :
  remaining_money_in_cents 0.85 0.40 0.50 1.60 = 15 := by
  sorry

end yanni_remaining_money_l2668_266878


namespace owls_on_fence_l2668_266843

/-- The number of owls on a fence after more owls join is the sum of the initial number and the number that joined. -/
theorem owls_on_fence (initial_owls joining_owls : ℕ) :
  let total_owls := initial_owls + joining_owls
  total_owls = initial_owls + joining_owls :=
by
  sorry

end owls_on_fence_l2668_266843


namespace power_division_rule_l2668_266821

theorem power_division_rule (a : ℝ) : a^3 / a^2 = a := by sorry

end power_division_rule_l2668_266821


namespace competition_result_l2668_266899

def math_competition (sammy_score : ℕ) (opponent_score : ℕ) : Prop :=
  let gab_score := 2 * sammy_score
  let cher_score := 2 * gab_score
  let total_score := sammy_score + gab_score + cher_score
  total_score - opponent_score = 55

theorem competition_result : math_competition 20 85 := by
  sorry

end competition_result_l2668_266899


namespace journey_time_proof_l2668_266877

/-- Proves that the time taken to complete a 224 km journey, where the first half is traveled at 21 km/hr and the second half at 24 km/hr, is equal to 10 hours. -/
theorem journey_time_proof (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_distance = 224 →
  speed1 = 21 →
  speed2 = 24 →
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 10 := by
  sorry

#check journey_time_proof

end journey_time_proof_l2668_266877


namespace shooting_performance_and_probability_l2668_266846

def shooter_A_scores : List ℕ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]
def shooter_B_scores : List ℕ := [9, 5, 7, 8, 7, 6, 8, 6, 7, 7]

def mean (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

def variance (scores : List ℕ) : ℚ :=
  let m := mean scores
  (scores.map (fun x => ((x : ℚ) - m)^2)).sum / scores.length

def is_excellent (score : ℕ) : Bool :=
  score ≥ 8

def excellent_probability (scores : List ℕ) : ℚ :=
  (scores.filter is_excellent).length / scores.length

theorem shooting_performance_and_probability :
  (variance shooter_B_scores < variance shooter_A_scores) ∧
  (excellent_probability shooter_A_scores + excellent_probability shooter_B_scores = 19/25) := by
  sorry

end shooting_performance_and_probability_l2668_266846


namespace student_mistake_difference_l2668_266852

theorem student_mistake_difference : (5/6 : ℚ) * 96 - (5/16 : ℚ) * 96 = 50 := by
  sorry

end student_mistake_difference_l2668_266852


namespace extremum_implies_a_b_values_l2668_266831

/-- The function f(x) = x^3 - ax^2 - bx + a^2 has an extremum value of 10 at x = 1 -/
def has_extremum (a b : ℝ) : Prop :=
  let f := fun x : ℝ => x^3 - a*x^2 - b*x + a^2
  (∃ ε > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < ε → f x ≤ f 1) ∧
  (∃ ε > 0, ∀ x, 0 < |x - 1| ∧ |x - 1| < ε → f x ≥ f 1) ∧
  f 1 = 10

/-- If f(x) = x^3 - ax^2 - bx + a^2 has an extremum value of 10 at x = 1, then a = -4 and b = 11 -/
theorem extremum_implies_a_b_values :
  ∀ a b : ℝ, has_extremum a b → a = -4 ∧ b = 11 := by
  sorry

end extremum_implies_a_b_values_l2668_266831


namespace product_equality_implies_composite_sums_l2668_266863

theorem product_equality_implies_composite_sums (a b c d : ℕ) (h : a * b = c * d) :
  (∃ x y : ℕ, x > 1 ∧ y > 1 ∧ a + b + c + d = x * y) ∧
  (∃ x y : ℕ, x > 1 ∧ y > 1 ∧ a^2 + b^2 + c^2 + d^2 = x * y) :=
by sorry

end product_equality_implies_composite_sums_l2668_266863


namespace product_of_tripled_numbers_with_reciprocals_l2668_266803

theorem product_of_tripled_numbers_with_reciprocals (x : ℝ) : 
  (x + 1/x = 3*x) → (∃ y : ℝ, (y + 1/y = 3*y) ∧ (x * y = -1/2)) :=
by sorry

end product_of_tripled_numbers_with_reciprocals_l2668_266803


namespace floor_abs_negative_l2668_266897

theorem floor_abs_negative : ⌊|(-45.7 : ℝ)|⌋ = 45 := by sorry

end floor_abs_negative_l2668_266897


namespace rice_mixture_cost_l2668_266841

/-- Given the cost of two varieties of rice and their mixing ratio, 
    calculate the cost of the second variety. -/
theorem rice_mixture_cost 
  (cost_first : ℝ) 
  (cost_mixture : ℝ) 
  (ratio : ℝ) 
  (h1 : cost_first = 5.5)
  (h2 : cost_mixture = 7.50)
  (h3 : ratio = 0.625) :
  ∃ (cost_second : ℝ), 
    cost_second = 10.7 ∧ 
    (cost_first - cost_mixture) / (cost_mixture - cost_second) = ratio / 1 :=
by sorry

end rice_mixture_cost_l2668_266841


namespace frog_flies_consumption_l2668_266884

/-- Proves that each frog needs to eat 30 flies per day in a swamp ecosystem -/
theorem frog_flies_consumption
  (fish_frog_consumption : ℕ) -- Number of frogs each fish eats per day
  (gharial_fish_consumption : ℕ) -- Number of fish each gharial eats per day
  (gharial_count : ℕ) -- Number of gharials in the swamp
  (total_flies_eaten : ℕ) -- Total number of flies eaten per day
  (h1 : fish_frog_consumption = 8)
  (h2 : gharial_fish_consumption = 15)
  (h3 : gharial_count = 9)
  (h4 : total_flies_eaten = 32400) :
  total_flies_eaten / (gharial_count * gharial_fish_consumption * fish_frog_consumption) = 30 := by
  sorry


end frog_flies_consumption_l2668_266884


namespace problem_1_solution_l2668_266816

theorem problem_1_solution (x : ℝ) : 
  (2 / (x - 3) = 1 / x) ↔ (x = -3) :=
sorry

end problem_1_solution_l2668_266816


namespace train_length_calculation_l2668_266853

theorem train_length_calculation (v1 v2 : ℝ) (t : ℝ) (h1 : v1 = 95) (h2 : v2 = 85) (h3 : t = 6) :
  let relative_speed := (v1 + v2) * (5/18)
  let total_length := relative_speed * t
  let train_length := total_length / 2
  train_length = 150 := by sorry

end train_length_calculation_l2668_266853


namespace license_combinations_l2668_266808

/-- Represents the number of choices for the letter in a license -/
def letter_choices : ℕ := 3

/-- Represents the number of choices for each digit in a license -/
def digit_choices : ℕ := 10

/-- Represents the number of digits in a license -/
def num_digits : ℕ := 4

/-- Calculates the total number of possible license combinations -/
def total_combinations : ℕ := letter_choices * digit_choices ^ num_digits

/-- Proves that the number of unique license combinations is 30000 -/
theorem license_combinations : total_combinations = 30000 := by
  sorry

end license_combinations_l2668_266808


namespace second_discount_percentage_l2668_266864

theorem second_discount_percentage
  (original_price : ℝ)
  (first_discount_percent : ℝ)
  (final_price : ℝ)
  (h1 : original_price = 175)
  (h2 : first_discount_percent = 20)
  (h3 : final_price = 133)
  : ∃ (second_discount_percent : ℝ),
    final_price = original_price * (1 - first_discount_percent / 100) * (1 - second_discount_percent / 100) ∧
    second_discount_percent = 5 :=
sorry

end second_discount_percentage_l2668_266864


namespace T_equals_one_l2668_266820

theorem T_equals_one (S : ℝ) : 
  let T := Real.sin (50 * π / 180) * (S + Real.sqrt 3 * Real.tan (10 * π / 180))
  T = 1 :=
by sorry

end T_equals_one_l2668_266820


namespace number_of_divisors_of_60_l2668_266837

theorem number_of_divisors_of_60 : ∃ (s : Finset Nat), ∀ d : Nat, d ∈ s ↔ d ∣ 60 ∧ d > 0 ∧ Finset.card s = 12 := by
  sorry

end number_of_divisors_of_60_l2668_266837


namespace inequality_proof_l2668_266839

theorem inequality_proof (a b : ℝ) (h : |a + b| ≤ 2) :
  |a^2 + 2*a - b^2 + 2*b| ≤ 4*(|a| + 2) := by
sorry

end inequality_proof_l2668_266839


namespace complement_intersection_theorem_l2668_266883

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def A : Set Nat := {2, 4, 5, 7}
def B : Set Nat := {3, 4, 5}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {1, 6} := by sorry

end complement_intersection_theorem_l2668_266883


namespace usual_time_calculation_l2668_266804

theorem usual_time_calculation (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_time > 0) (h2 : usual_speed > 0) : 
  (usual_speed * usual_time = (usual_speed / 2) * (usual_time + 24)) → 
  usual_time = 24 := by
  sorry

end usual_time_calculation_l2668_266804


namespace min_value_quadratic_l2668_266844

theorem min_value_quadratic (x y : ℝ) : 
  x^2 + y^2 + 10*x - 8*y + 34 ≥ -7 ∧ 
  ∃ (a b : ℝ), a^2 + b^2 + 10*a - 8*b + 34 = -7 := by
  sorry

end min_value_quadratic_l2668_266844


namespace john_hats_cost_l2668_266859

def weeks : ℕ := 20
def days_per_week : ℕ := 7
def odd_day_price : ℕ := 45
def even_day_price : ℕ := 60
def discount_threshold : ℕ := 50
def discount_rate : ℚ := 1 / 10

def total_hats : ℕ := weeks * days_per_week
def odd_days : ℕ := total_hats / 2
def even_days : ℕ := total_hats / 2

def total_cost : ℕ := odd_days * odd_day_price + even_days * even_day_price
def discounted_cost : ℚ := total_cost * (1 - discount_rate)

theorem john_hats_cost : 
  total_hats ≥ discount_threshold → discounted_cost = 6615 := by
  sorry

end john_hats_cost_l2668_266859


namespace rectangle_dimensions_l2668_266842

theorem rectangle_dimensions (area perimeter : ℝ) (h1 : area = 12) (h2 : perimeter = 26) :
  ∃ (length width : ℝ),
    length * width = area ∧
    2 * (length + width) = perimeter ∧
    ((length = 1 ∧ width = 12) ∨ (length = 12 ∧ width = 1)) :=
by sorry

end rectangle_dimensions_l2668_266842


namespace right_triangle_shorter_side_l2668_266885

/-- A right triangle with perimeter 40 and area 30 has a shorter side of length 5.25 -/
theorem right_triangle_shorter_side : ∃ (a b c : ℝ),
  a > 0 ∧ b > 0 ∧ c > 0 ∧  -- positive side lengths
  a^2 + b^2 = c^2 ∧        -- right triangle (Pythagorean theorem)
  a + b + c = 40 ∧         -- perimeter is 40
  (1/2) * a * b = 30 ∧     -- area is 30
  (a = 5.25 ∨ b = 5.25) :=  -- one shorter side is 5.25
by sorry

end right_triangle_shorter_side_l2668_266885


namespace isosceles_triangle_base_l2668_266890

/-- An isosceles triangle with perimeter 11 and one side length 3 -/
structure IsoscelesTriangle where
  /-- The length of two equal sides -/
  side : ℝ
  /-- The length of the base -/
  base : ℝ
  /-- The triangle is isosceles -/
  isIsosceles : side ≥ 0 ∧ base ≥ 0
  /-- The perimeter is 11 -/
  perimeterIs11 : 2 * side + base = 11
  /-- One side length is 3 -/
  oneSideIs3 : side = 3 ∨ base = 3

/-- The base of an isosceles triangle with perimeter 11 and one side length 3 can only be 3 or 5 -/
theorem isosceles_triangle_base (t : IsoscelesTriangle) : t.base = 3 ∨ t.base = 5 := by
  sorry

end isosceles_triangle_base_l2668_266890


namespace thousandth_digit_is_three_l2668_266805

/-- The sequence of digits obtained by concatenating integers from 1 to 499 -/
def digit_sequence : ℕ → ℕ
| 0 => 1
| n + 1 => if n + 1 < 499 then digit_sequence n * 10 + (n + 2) else digit_sequence n

/-- The nth digit in the sequence -/
def nth_digit (n : ℕ) : ℕ :=
  (digit_sequence (n / 9) / (10 ^ (n % 9))) % 10

/-- Theorem stating that the 1000th digit is 3 -/
theorem thousandth_digit_is_three : nth_digit 999 = 3 := by
  sorry

end thousandth_digit_is_three_l2668_266805


namespace arithmetic_sequence_length_3_199_4_l2668_266868

/-- Calculates the number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (first last commonDiff : ℕ) : ℕ :=
  (last - first) / commonDiff + 1

/-- Theorem: The arithmetic sequence starting with 3, ending with 199,
    and having a common difference of 4 contains exactly 50 terms -/
theorem arithmetic_sequence_length_3_199_4 :
  arithmeticSequenceLength 3 199 4 = 50 := by
  sorry

end arithmetic_sequence_length_3_199_4_l2668_266868


namespace cube_structure_extension_l2668_266807

/-- Represents a cube structure with a central cube and attached cubes -/
structure CubeStructure :=
  (central : ℕ)
  (attached : ℕ)

/-- The number of cubes in the initial structure -/
def initial_cubes (s : CubeStructure) : ℕ := s.central + s.attached

/-- The number of exposed faces in the initial structure -/
def exposed_faces (s : CubeStructure) : ℕ := s.attached * 5

/-- The number of extra cubes needed for the extended structure -/
def extra_cubes_needed (s : CubeStructure) : ℕ := 12 + 6

theorem cube_structure_extension (s : CubeStructure) 
  (h1 : s.central = 1) 
  (h2 : s.attached = 6) : 
  extra_cubes_needed s = 18 := by sorry

end cube_structure_extension_l2668_266807


namespace pool_capacity_l2668_266840

theorem pool_capacity (C : ℝ) (h1 : C > 0) : 
  (0.4 * C + 300 = 0.8 * C) → C = 750 := by
  sorry

end pool_capacity_l2668_266840


namespace sqrt_equality_implies_unique_pair_l2668_266869

theorem sqrt_equality_implies_unique_pair :
  ∀ a b : ℕ,
  0 < a → 0 < b → a < b →
  (Real.sqrt (4 + Real.sqrt (36 + 24 * Real.sqrt 2)) = Real.sqrt a + Real.sqrt b) →
  a = 1 ∧ b = 7 := by
  sorry

end sqrt_equality_implies_unique_pair_l2668_266869


namespace sum_of_distinct_prime_factors_1320_l2668_266809

theorem sum_of_distinct_prime_factors_1320 :
  (Finset.sum (Finset.filter Nat.Prime (Finset.range (1320 + 1)))
    (fun p => if p ∣ 1320 then p else 0)) = 21 := by sorry

end sum_of_distinct_prime_factors_1320_l2668_266809


namespace unique_arrangement_l2668_266801

-- Define the containers and liquids as enumerated types
inductive Container : Type
| Cup : Container
| Glass : Container
| Jug : Container
| Jar : Container

inductive Liquid : Type
| Milk : Liquid
| Lemonade : Liquid
| Kvass : Liquid
| Water : Liquid

-- Define the arrangement as a function from Container to Liquid
def Arrangement := Container → Liquid

-- Define the conditions
def satisfiesConditions (arr : Arrangement) : Prop :=
  (arr Container.Cup ≠ Liquid.Water ∧ arr Container.Cup ≠ Liquid.Milk) ∧
  (∃ c, (c = Container.Jug ∨ c = Container.Jar) ∧
        arr c = Liquid.Kvass ∧
        (arr Container.Cup = Liquid.Lemonade ∨
         arr Container.Glass = Liquid.Lemonade)) ∧
  (arr Container.Jar ≠ Liquid.Lemonade ∧ arr Container.Jar ≠ Liquid.Water) ∧
  ((arr Container.Glass = Liquid.Milk ∧ arr Container.Jug = Liquid.Milk) ∨
   (arr Container.Glass = Liquid.Milk ∧ arr Container.Jar = Liquid.Milk) ∨
   (arr Container.Jug = Liquid.Milk ∧ arr Container.Jar = Liquid.Milk))

-- Define the correct arrangement
def correctArrangement : Arrangement
| Container.Cup => Liquid.Lemonade
| Container.Glass => Liquid.Water
| Container.Jug => Liquid.Milk
| Container.Jar => Liquid.Kvass

-- Theorem statement
theorem unique_arrangement :
  ∀ (arr : Arrangement), satisfiesConditions arr → arr = correctArrangement :=
by sorry

end unique_arrangement_l2668_266801


namespace shiela_neighbors_l2668_266813

theorem shiela_neighbors (total_drawings : ℕ) (drawings_per_neighbor : ℕ) 
  (h1 : total_drawings = 54)
  (h2 : drawings_per_neighbor = 9)
  (h3 : total_drawings % drawings_per_neighbor = 0) :
  total_drawings / drawings_per_neighbor = 6 := by
  sorry

end shiela_neighbors_l2668_266813


namespace dice_throw_probability_l2668_266892

theorem dice_throw_probability (n : ℕ) : 
  (1 / 2 : ℚ) ^ n = 1 / 4 → n = 2 := by
  sorry

end dice_throw_probability_l2668_266892


namespace systematic_sample_fifth_seat_l2668_266851

/-- Represents a systematic sample from a class -/
structure SystematicSample where
  class_size : ℕ
  sample_size : ℕ
  known_seats : Fin 4 → ℕ
  (class_size_pos : class_size > 0)
  (sample_size_pos : sample_size > 0)
  (sample_size_le_class : sample_size ≤ class_size)
  (known_seats_valid : ∀ i, known_seats i ≤ class_size)
  (known_seats_ordered : ∀ i j, i < j → known_seats i < known_seats j)

/-- The theorem to be proved -/
theorem systematic_sample_fifth_seat
  (s : SystematicSample)
  (h1 : s.class_size = 60)
  (h2 : s.sample_size = 5)
  (h3 : s.known_seats 0 = 3)
  (h4 : s.known_seats 1 = 15)
  (h5 : s.known_seats 2 = 39)
  (h6 : s.known_seats 3 = 51) :
  ∃ (fifth_seat : ℕ), fifth_seat = 27 ∧
    (∀ i j, i ≠ j → s.known_seats i ≠ fifth_seat) ∧
    fifth_seat ≤ s.class_size :=
sorry

end systematic_sample_fifth_seat_l2668_266851


namespace solution_product_l2668_266828

theorem solution_product (r s : ℝ) : 
  (r - 3) * (3 * r + 8) = r^2 - 20 * r + 75 →
  (s - 3) * (3 * s + 8) = s^2 - 20 * s + 75 →
  r ≠ s →
  (r + 4) * (s + 4) = -119/2 := by
sorry

end solution_product_l2668_266828


namespace expression_between_two_and_three_l2668_266861

theorem expression_between_two_and_three (a b : ℝ) (h : 3 * a = 5 * b) :
  2 < |a + b| / b ∧ |a + b| / b < 3 := by sorry

end expression_between_two_and_three_l2668_266861


namespace cos_equality_problem_l2668_266866

theorem cos_equality_problem (n : ℤ) : 
  0 ≤ n ∧ n ≤ 270 → 
  Real.cos (n * π / 180) = Real.cos (962 * π / 180) →
  n = 118 := by sorry

end cos_equality_problem_l2668_266866


namespace irrational_pi_among_options_l2668_266827

theorem irrational_pi_among_options : 
  (∃ (a b : ℤ), (3.142 : ℝ) = a / b) ∧ 
  (∃ (a b : ℤ), (Real.sqrt 4 : ℝ) = a / b) ∧ 
  (∃ (a b : ℤ), (22 / 7 : ℝ) = a / b) ∧ 
  (¬ ∃ (a b : ℤ), (Real.pi : ℝ) = a / b) :=
by sorry

end irrational_pi_among_options_l2668_266827


namespace resistance_value_l2668_266817

/-- Given two identical resistors connected in series to a DC voltage source,
    prove that the resistance of each resistor is 2 Ω based on voltmeter and ammeter readings. -/
theorem resistance_value (R U Uv IA : ℝ) : 
  Uv = 10 →  -- Voltmeter reading
  IA = 10 →  -- Ammeter reading
  U = 2 * Uv →  -- Total voltage
  U = R * IA →  -- Ohm's law for the circuit with ammeter
  R = 2 := by
  sorry

end resistance_value_l2668_266817


namespace f_max_min_implies_m_range_l2668_266875

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 4*x + 5

-- State the theorem
theorem f_max_min_implies_m_range (m : ℝ) :
  (∀ x ∈ Set.Icc 0 m, f x ≤ 5) ∧  -- Maximum value is 5
  (∃ x ∈ Set.Icc 0 m, f x = 5) ∧  -- Maximum value is attained
  (∀ x ∈ Set.Icc 0 m, f x ≥ 1) ∧  -- Minimum value is 1
  (∃ x ∈ Set.Icc 0 m, f x = 1) →  -- Minimum value is attained
  m ∈ Set.Icc 2 4 :=
by sorry

end f_max_min_implies_m_range_l2668_266875


namespace expression_simplification_l2668_266856

theorem expression_simplification (x y : ℝ) 
  (h : (x - 2)^2 + |1 + y| = 0) : 
  ((x - y) * (x + 2*y) - (x + y)^2) / y = 1 := by
  sorry

end expression_simplification_l2668_266856


namespace inserted_square_side_length_l2668_266818

/-- An isosceles triangle with a square inserted -/
structure TriangleWithSquare where
  /-- Length of the lateral sides of the isosceles triangle -/
  lateral_side : ℝ
  /-- Length of the base of the isosceles triangle -/
  base : ℝ
  /-- Side length of the inserted square -/
  square_side : ℝ

/-- Theorem: In an isosceles triangle with lateral sides of 10 and base of 12, 
    the side length of an inserted square is 24/5 -/
theorem inserted_square_side_length 
  (t : TriangleWithSquare) 
  (h1 : t.lateral_side = 10) 
  (h2 : t.base = 12) : 
  t.square_side = 24/5 := by
  sorry

end inserted_square_side_length_l2668_266818


namespace sequence_increasing_l2668_266833

theorem sequence_increasing (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) (h_rel : ∀ n, a (n + 1) = 2 * a n) :
  ∀ n, a (n + 1) > a n :=
sorry

end sequence_increasing_l2668_266833


namespace cannot_determine_start_month_l2668_266898

/-- Represents a month of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Represents Nolan's GRE preparation period -/
structure PreparationPeriod where
  start_month : Month
  end_month : Month
  end_day : Nat

/-- The given information about Nolan's GRE preparation -/
def nolans_preparation : PreparationPeriod :=
  { end_month := Month.August,
    end_day := 3,
    start_month := sorry }  -- We don't know the start month

/-- Theorem stating that we cannot determine Nolan's start month -/
theorem cannot_determine_start_month :
  ∀ m : Month, ∃ p : PreparationPeriod,
    p.end_month = nolans_preparation.end_month ∧
    p.end_day = nolans_preparation.end_day ∧
    p.start_month = m :=
sorry

end cannot_determine_start_month_l2668_266898


namespace diophantine_equation_solutions_l2668_266872

theorem diophantine_equation_solutions :
  ∀ a b : ℕ, 3 * 2^a + 1 = b^2 ↔ (a = 0 ∧ b = 2) ∨ (a = 3 ∧ b = 5) ∨ (a = 4 ∧ b = 7) :=
by sorry

end diophantine_equation_solutions_l2668_266872


namespace expected_waiting_time_for_last_suitcase_l2668_266838

theorem expected_waiting_time_for_last_suitcase 
  (total_suitcases : ℕ) 
  (business_suitcases : ℕ) 
  (placement_interval : ℕ) 
  (h1 : total_suitcases = 200) 
  (h2 : business_suitcases = 10) 
  (h3 : placement_interval = 2) :
  (((total_suitcases + 1) * placement_interval * business_suitcases) / (business_suitcases + 1) : ℚ) = 4020 / 11 := by
  sorry

#check expected_waiting_time_for_last_suitcase

end expected_waiting_time_for_last_suitcase_l2668_266838


namespace transformed_sine_value_l2668_266819

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem transformed_sine_value 
  (ω φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : -π/2 ≤ φ ∧ φ < π/2) 
  (h_transform : ∀ x, Real.sin x = Real.sin (2 * ω * (x - π/6) + φ)) :
  f ω φ (π/6) = Real.sqrt 2 / 2 := by
  sorry

end transformed_sine_value_l2668_266819


namespace prob_score_exceeds_14_is_0_3_expected_value_two_triple_jumps_is_13_6_l2668_266848

-- Define the success rates and scores
def triple_jump_success_rate : ℝ := 0.7
def quadruple_jump_success_rate : ℝ := 0.3
def successful_triple_jump_score : ℕ := 8
def failed_triple_jump_score : ℕ := 4
def successful_quadruple_jump_score : ℕ := 15
def failed_quadruple_jump_score : ℕ := 6

-- Define the probability of score exceeding 14 points for triple jump followed by quadruple jump
def prob_score_exceeds_14 : ℝ := 
  triple_jump_success_rate * quadruple_jump_success_rate + 
  (1 - triple_jump_success_rate) * quadruple_jump_success_rate

-- Define the expected value of score for two consecutive triple jumps
def expected_value_two_triple_jumps : ℝ := 
  (1 - triple_jump_success_rate)^2 * (2 * failed_triple_jump_score) +
  2 * triple_jump_success_rate * (1 - triple_jump_success_rate) * (successful_triple_jump_score + failed_triple_jump_score) +
  triple_jump_success_rate^2 * (2 * successful_triple_jump_score)

-- Theorem statements
theorem prob_score_exceeds_14_is_0_3 : 
  prob_score_exceeds_14 = 0.3 := by sorry

theorem expected_value_two_triple_jumps_is_13_6 : 
  expected_value_two_triple_jumps = 13.6 := by sorry

end prob_score_exceeds_14_is_0_3_expected_value_two_triple_jumps_is_13_6_l2668_266848


namespace power_mean_inequality_l2668_266871

theorem power_mean_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≥ b) :
  (a^4 + b^4) / 2 ≥ ((a + b) / 2)^4 := by
  sorry

end power_mean_inequality_l2668_266871


namespace percentage_of_unsold_books_l2668_266867

def initial_stock : ℕ := 1400
def monday_sales : ℕ := 62
def tuesday_sales : ℕ := 62
def wednesday_sales : ℕ := 60
def thursday_sales : ℕ := 48
def friday_sales : ℕ := 40

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales

def books_not_sold : ℕ := initial_stock - total_sales

def percentage_not_sold : ℚ := (books_not_sold : ℚ) / (initial_stock : ℚ) * 100

theorem percentage_of_unsold_books :
  abs (percentage_not_sold - 80.57) < 0.01 := by sorry

end percentage_of_unsold_books_l2668_266867


namespace cherries_eaten_l2668_266894

theorem cherries_eaten (initial : ℕ) (remaining : ℕ) (h1 : initial = 67) (h2 : remaining = 42) :
  initial - remaining = 25 := by
  sorry

end cherries_eaten_l2668_266894


namespace average_of_tenths_and_thousandths_l2668_266845

theorem average_of_tenths_and_thousandths :
  let a : ℚ := 4/10
  let b : ℚ := 5/1000
  (a + b) / 2 = 2025/10000 := by
  sorry

end average_of_tenths_and_thousandths_l2668_266845


namespace second_next_perfect_square_l2668_266865

theorem second_next_perfect_square (x : ℕ) (h : ∃ k : ℕ, x = k^2) :
  ∃ n : ℕ, n^2 = x + 4 * (x : ℝ).sqrt + 4 :=
sorry

end second_next_perfect_square_l2668_266865


namespace product_of_binary_and_ternary_l2668_266881

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.foldr (fun b n => 2 * n + if b then 1 else 0) 0

/-- Converts a ternary number represented as a list of trits to its decimal equivalent -/
def ternary_to_decimal (trits : List ℕ) : ℕ :=
  trits.foldr (fun t n => 3 * n + t) 0

theorem product_of_binary_and_ternary :
  let binary_num := [true, false, true, true]  -- 1011 in binary
  let ternary_num := [1, 1, 1]  -- 111 in ternary
  (binary_to_decimal binary_num) * (ternary_to_decimal ternary_num) = 143 := by
  sorry

end product_of_binary_and_ternary_l2668_266881


namespace josh_marbles_remaining_l2668_266876

/-- The number of marbles Josh has remaining after losing some. -/
def remaining_marbles (initial : ℝ) (lost : ℝ) : ℝ :=
  initial - lost

/-- Theorem stating that Josh has 7.75 marbles remaining. -/
theorem josh_marbles_remaining :
  remaining_marbles 19.5 11.75 = 7.75 := by
  sorry

end josh_marbles_remaining_l2668_266876


namespace area_of_integral_triangle_with_perimeter_12_l2668_266850

/-- Represents a triangle with integral sides --/
structure IntegralTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  sum_eq_12 : a + b + c = 12
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The area of an integral triangle with perimeter 12 is 2√6 --/
theorem area_of_integral_triangle_with_perimeter_12 (t : IntegralTriangle) : 
  Real.sqrt (6 * (6 - t.a) * (6 - t.b) * (6 - t.c)) = 2 * Real.sqrt 6 :=
sorry

end area_of_integral_triangle_with_perimeter_12_l2668_266850


namespace imaginary_part_of_complex_division_l2668_266862

theorem imaginary_part_of_complex_division : 
  let z : ℂ := -3 + 4*I
  let w : ℂ := 1 + I
  (z / w).im = -1/2 := by sorry

end imaginary_part_of_complex_division_l2668_266862


namespace intersection_of_A_and_B_l2668_266887

-- Define set A
def A : Set ℝ := {y | ∃ x, y = x^2 - 1}

-- Define set B
def B : Set ℝ := {x | |x^2 - 1| ≤ 3}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = Set.Icc (-1) 2 := by
  sorry

end intersection_of_A_and_B_l2668_266887


namespace binary_11010_is_26_l2668_266880

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_11010_is_26 :
  binary_to_decimal [false, true, false, true, true] = 26 := by
  sorry

end binary_11010_is_26_l2668_266880


namespace complement_of_union_equals_set_l2668_266849

/-- The universal set U -/
def U : Set Int := {-2, -1, 0, 1, 2, 3}

/-- Set A -/
def A : Set Int := {-1, 2}

/-- Set B -/
def B : Set Int := {x : Int | x^2 - 4*x + 3 = 0}

/-- The main theorem -/
theorem complement_of_union_equals_set (h : U = {-2, -1, 0, 1, 2, 3}) :
  (U \ (A ∪ B)) = {-2, 0} := by
  sorry

end complement_of_union_equals_set_l2668_266849


namespace group_5_frequency_l2668_266893

theorem group_5_frequency (total : ℕ) (group1 group2 group3 group4 : ℕ) 
  (h_total : total = 50)
  (h_group1 : group1 = 2)
  (h_group2 : group2 = 8)
  (h_group3 : group3 = 15)
  (h_group4 : group4 = 5) :
  (total - group1 - group2 - group3 - group4 : ℚ) / total = 0.4 := by
  sorry

end group_5_frequency_l2668_266893


namespace derivative_at_zero_l2668_266806

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then Real.arctan ((3 * x / 2) - x^2 * Real.sin (1 / x))
  else 0

-- State the theorem
theorem derivative_at_zero (h : HasDerivAt f (3/2) 0) : 
  deriv f 0 = 3/2 := by sorry

end derivative_at_zero_l2668_266806


namespace yard_area_l2668_266824

/-- The area of a rectangular yard with a square cut out -/
theorem yard_area (length width cut_side : ℝ) 
  (h1 : length = 15)
  (h2 : width = 12)
  (h3 : cut_side = 3) : 
  length * width - cut_side^2 = 171 := by
  sorry

end yard_area_l2668_266824


namespace profit_equation_l2668_266812

/-- Represents the profit equation for a product with given cost and selling prices,
    initial quantity sold, and price reduction effects. -/
theorem profit_equation
  (cost_price : ℝ)
  (initial_selling_price : ℝ)
  (initial_quantity : ℝ)
  (additional_units_per_reduction : ℝ)
  (target_profit : ℝ)
  (h1 : cost_price = 40)
  (h2 : initial_selling_price = 60)
  (h3 : initial_quantity = 200)
  (h4 : additional_units_per_reduction = 8)
  (h5 : target_profit = 8450)
  (x : ℝ) :
  (initial_selling_price - cost_price - x) * (initial_quantity + additional_units_per_reduction * x) = target_profit :=
by sorry

end profit_equation_l2668_266812


namespace negation_of_forall_greater_than_two_l2668_266836

theorem negation_of_forall_greater_than_two :
  (¬ (∀ x : ℝ, x > 2)) ↔ (∃ x : ℝ, x ≤ 2) := by
  sorry

end negation_of_forall_greater_than_two_l2668_266836


namespace complex_modulus_power_eight_l2668_266874

theorem complex_modulus_power_eight : 
  Complex.abs ((1/2 : ℂ) + (Complex.I * (Real.sqrt 3 / 2)))^8 = 1 := by sorry

end complex_modulus_power_eight_l2668_266874


namespace inverse_proportional_point_l2668_266854

theorem inverse_proportional_point :
  let f : ℝ → ℝ := λ x => 6 / x
  f (-2) = -3 :=
by
  sorry

end inverse_proportional_point_l2668_266854


namespace antonov_candy_packs_l2668_266830

/-- Given a total number of candies and packs, calculate the number of candies per pack -/
def candies_per_pack (total_candies : ℕ) (total_packs : ℕ) : ℕ :=
  total_candies / total_packs

/-- Theorem: The number of candies per pack is 20 -/
theorem antonov_candy_packs : candies_per_pack 60 3 = 20 := by
  sorry

end antonov_candy_packs_l2668_266830


namespace min_value_of_expression_l2668_266855

theorem min_value_of_expression (x y : ℝ) : (x * y + 2)^2 + (x - y)^2 ≥ 4 := by
  sorry

end min_value_of_expression_l2668_266855


namespace current_babysitter_rate_is_16_l2668_266800

/-- Represents the babysitting scenario with given conditions -/
structure BabysittingScenario where
  new_hourly_rate : ℕ
  scream_charge : ℕ
  hours : ℕ
  scream_count : ℕ
  cost_difference : ℕ

/-- Calculates the hourly rate of the current babysitter -/
def current_babysitter_rate (scenario : BabysittingScenario) : ℕ :=
  ((scenario.new_hourly_rate * scenario.hours + scenario.scream_charge * scenario.scream_count) + scenario.cost_difference) / scenario.hours

/-- Theorem stating that given the conditions, the current babysitter's hourly rate is $16 -/
theorem current_babysitter_rate_is_16 (scenario : BabysittingScenario) 
    (h1 : scenario.new_hourly_rate = 12)
    (h2 : scenario.scream_charge = 3)
    (h3 : scenario.hours = 6)
    (h4 : scenario.scream_count = 2)
    (h5 : scenario.cost_difference = 18) :
  current_babysitter_rate scenario = 16 := by
  sorry

#eval current_babysitter_rate { new_hourly_rate := 12, scream_charge := 3, hours := 6, scream_count := 2, cost_difference := 18 }

end current_babysitter_rate_is_16_l2668_266800


namespace not_square_and_floor_sqrt_cube_divides_square_l2668_266835

theorem not_square_and_floor_sqrt_cube_divides_square (n : ℕ) :
  (∀ k : ℕ, n ≠ k^2) →
  (Nat.floor (Real.sqrt n))^3 ∣ n^2 →
  n = 2 ∨ n = 3 ∨ n = 8 ∨ n = 24 := by
  sorry

end not_square_and_floor_sqrt_cube_divides_square_l2668_266835


namespace trader_markup_percentage_l2668_266857

theorem trader_markup_percentage (discount : ℝ) (loss : ℝ) : 
  discount = 7.857142857142857 / 100 →
  loss = 1 / 100 →
  ∃ (markup : ℝ), 
    (1 + markup) * (1 - discount) = 1 - loss ∧ 
    abs (markup - 7.4285714285714 / 100) < 1e-10 := by
  sorry

end trader_markup_percentage_l2668_266857


namespace people_owning_cats_and_dogs_l2668_266823

theorem people_owning_cats_and_dogs (
  total_pet_owners : ℕ)
  (only_dog_owners : ℕ)
  (only_cat_owners : ℕ)
  (cat_dog_snake_owners : ℕ)
  (h1 : total_pet_owners = 69)
  (h2 : only_dog_owners = 15)
  (h3 : only_cat_owners = 10)
  (h4 : cat_dog_snake_owners = 3) :
  total_pet_owners = only_dog_owners + only_cat_owners + 41 + cat_dog_snake_owners :=
by
  sorry

end people_owning_cats_and_dogs_l2668_266823


namespace removal_ways_count_l2668_266847

/-- Represents a block in the stack -/
structure Block where
  layer : Nat
  exposed : Bool

/-- Represents the stack of blocks -/
def Stack : Type := List Block

/-- The initial stack configuration -/
def initialStack : Stack := sorry

/-- Function to check if a block can be removed -/
def canRemove (b : Block) (s : Stack) : Bool := sorry

/-- Function to remove a block and update the stack -/
def removeBlock (b : Block) (s : Stack) : Stack := sorry

/-- Function to count the number of ways to remove 5 blocks -/
def countRemovalWays (s : Stack) : Nat := sorry

/-- The main theorem stating the number of ways to remove 5 blocks -/
theorem removal_ways_count : 
  countRemovalWays initialStack = 3384 := by sorry

end removal_ways_count_l2668_266847


namespace square_garden_area_perimeter_relation_l2668_266879

theorem square_garden_area_perimeter_relation :
  ∀ (s : ℝ), 
    s > 0 →
    4 * s = 40 →
    s^2 - 2 * (4 * s) = 20 :=
by
  sorry

end square_garden_area_perimeter_relation_l2668_266879


namespace cassidy_grounding_period_l2668_266891

/-- Calculates the total grounding period for Cassidy based on her grades and volunteering. -/
def calculate_grounding_period (
  initial_grounding : ℕ
  ) (extra_days_per_grade : ℕ
  ) (grades_below_b : ℕ
  ) (extracurricular_below_b : ℕ
  ) (volunteering_reduction : ℕ
  ) : ℕ :=
  let subject_penalty := grades_below_b * extra_days_per_grade
  let extracurricular_penalty := extracurricular_below_b * (extra_days_per_grade / 2)
  let total_before_volunteering := initial_grounding + subject_penalty + extracurricular_penalty
  total_before_volunteering - volunteering_reduction

/-- Theorem stating that Cassidy's total grounding period is 27 days. -/
theorem cassidy_grounding_period :
  calculate_grounding_period 14 3 4 2 2 = 27 := by
  sorry

end cassidy_grounding_period_l2668_266891
