import Mathlib

namespace supplement_of_complement_of_65_degrees_l3205_320560

theorem supplement_of_complement_of_65_degrees : 
  let α : ℝ := 65
  let complement_of_α : ℝ := 90 - α
  let supplement_of_complement : ℝ := 180 - complement_of_α
  supplement_of_complement = 155 := by
  sorry

end supplement_of_complement_of_65_degrees_l3205_320560


namespace max_profit_theorem_l3205_320562

-- Define the cost and profit for each pen type
def cost_A : ℝ := 5
def cost_B : ℝ := 10
def profit_A : ℝ := 2
def profit_B : ℝ := 3

-- Define the total number of pens and the constraint
def total_pens : ℕ := 300
def constraint (x : ℕ) : Prop := x ≥ 4 * (total_pens - x)

-- Define the profit function
def profit (x : ℕ) : ℝ := profit_A * x + profit_B * (total_pens - x)

theorem max_profit_theorem :
  ∃ x : ℕ, x ≤ total_pens ∧ constraint x ∧
  profit x = 660 ∧
  ∀ y : ℕ, y ≤ total_pens → constraint y → profit y ≤ profit x :=
by sorry

end max_profit_theorem_l3205_320562


namespace imaginary_part_of_z_l3205_320555

theorem imaginary_part_of_z (z : ℂ) (h : (3 + 4*I)*z = Complex.abs (4 - 3*I)) : 
  z.im = -4/5 := by
  sorry

end imaginary_part_of_z_l3205_320555


namespace race_difference_l3205_320570

/-- Represents a racer in the competition -/
structure Racer where
  time : ℝ  -- Time taken to complete the race in seconds
  speed : ℝ  -- Speed of the racer in meters per second

/-- Calculates the distance covered by a racer in a given time -/
def distance_covered (r : Racer) (t : ℝ) : ℝ := r.speed * t

theorem race_difference (race_distance : ℝ) (a b : Racer) 
  (h1 : race_distance = 80)
  (h2 : a.time = 20)
  (h3 : b.time = 25)
  (h4 : a.speed = race_distance / a.time)
  (h5 : b.speed = race_distance / b.time) :
  race_distance - distance_covered b a.time = 16 := by
  sorry

end race_difference_l3205_320570


namespace inequality_proof_l3205_320595

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 1/b = 1) :
  ∀ n : ℕ, (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) := by
  sorry

end inequality_proof_l3205_320595


namespace area_ratio_of_squares_l3205_320506

/-- Given three square regions A, B, and C, where the perimeter of A is 16 units,
    the perimeter of B is 32 units, and the side length of each subsequent region doubles,
    prove that the ratio of the area of region B to the area of region C is 1/4. -/
theorem area_ratio_of_squares (side_A side_B side_C : ℝ) : 
  side_A * 4 = 16 →
  side_B * 4 = 32 →
  side_C = 2 * side_B →
  (side_B ^ 2) / (side_C ^ 2) = 1 / 4 :=
by sorry

end area_ratio_of_squares_l3205_320506


namespace inequality_solution_set_l3205_320552

theorem inequality_solution_set (x : ℝ) : 
  1 / (x + 2) + 8 / (x + 6) ≥ 1 ↔ -6 < x ∧ x ≤ 5 := by
  sorry

end inequality_solution_set_l3205_320552


namespace set_union_problem_l3205_320529

theorem set_union_problem (a b : ℝ) : 
  let A : Set ℝ := {3, 2^a}
  let B : Set ℝ := {a, b}
  (A ∩ B = {2}) → (A ∪ B = {1, 2, 3}) := by
sorry

end set_union_problem_l3205_320529


namespace bug_return_probability_l3205_320545

/-- Probability of the bug being at the starting vertex after n moves -/
def P : ℕ → ℚ
  | 0 => 1
  | n + 1 => (1 / 3) * (1 - P n)

/-- The probability of the bug returning to its starting vertex after 12 moves -/
theorem bug_return_probability : P 12 = 14762 / 59049 := by
  sorry

end bug_return_probability_l3205_320545


namespace sum_of_angles_in_four_intersecting_lines_l3205_320527

-- Define the angles as real numbers
variable (p q r s : ℝ)

-- Define the property of four intersecting lines
def four_intersecting_lines (p q r s : ℝ) : Prop :=
  -- Add any additional properties that define four intersecting lines
  True

-- Theorem statement
theorem sum_of_angles_in_four_intersecting_lines 
  (h : four_intersecting_lines p q r s) : 
  p + q + r + s = 540 := by
  sorry

end sum_of_angles_in_four_intersecting_lines_l3205_320527


namespace inequality_solution_set_l3205_320522

theorem inequality_solution_set (x : ℝ) : -x^2 - 2*x + 3 > 0 ↔ -3 < x ∧ x < 1 := by
  sorry

end inequality_solution_set_l3205_320522


namespace captain_age_l3205_320551

/-- Represents a cricket team with given properties -/
structure CricketTeam where
  size : ℕ
  captainAge : ℕ
  wicketKeeperAge : ℕ
  teamAverageAge : ℝ
  remainingPlayersAverageAge : ℝ

/-- Theorem stating the captain's age in the given cricket team scenario -/
theorem captain_age (team : CricketTeam) 
  (h1 : team.size = 11)
  (h2 : team.wicketKeeperAge = team.captainAge + 5)
  (h3 : team.teamAverageAge = 23)
  (h4 : team.remainingPlayersAverageAge = team.teamAverageAge - 1)
  : team.captainAge = 25 := by
  sorry

end captain_age_l3205_320551


namespace bridge_length_calculation_bridge_length_proof_l3205_320525

/-- Calculates the length of a bridge given train parameters --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_pass : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  let total_distance := train_speed_ms * time_to_pass
  let bridge_length := total_distance - train_length
  bridge_length

/-- Proves that the bridge length is approximately 140 meters --/
theorem bridge_length_proof :
  let train_length : ℝ := 360
  let train_speed_kmh : ℝ := 56
  let time_to_pass : ℝ := 32.142857142857146
  let calculated_bridge_length := bridge_length_calculation train_length train_speed_kmh time_to_pass
  ∃ ε > 0, |calculated_bridge_length - 140| < ε :=
by
  sorry

end bridge_length_calculation_bridge_length_proof_l3205_320525


namespace simplify_expression_l3205_320507

theorem simplify_expression (x : ℝ) : 2 * x^8 / x^4 = 2 * x^4 := by sorry

end simplify_expression_l3205_320507


namespace R_zero_value_l3205_320586

-- Define the polynomial P
def P (x : ℝ) : ℝ := x^2 - 3*x - 7

-- Define the properties for Q and R
def is_valid_polynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b, ∀ x, f x = x^2 + a*x + b

-- Define the condition that P + Q, P + R, and Q + R each have a common root
def have_common_roots (P Q R : ℝ → ℝ) : Prop :=
  ∃ p q r, p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    (P p + Q p = 0 ∧ P p + R p = 0) ∧
    (P q + Q q = 0 ∧ Q q + R q = 0) ∧
    (P r + R r = 0 ∧ Q r + R r = 0)

-- Main theorem
theorem R_zero_value (Q R : ℝ → ℝ) 
  (hQ : is_valid_polynomial Q)
  (hR : is_valid_polynomial R)
  (hQR : have_common_roots P Q R)
  (hQ0 : Q 0 = 2) :
  R 0 = 52 / 19 :=
sorry

end R_zero_value_l3205_320586


namespace unique_solution_floor_equation_l3205_320549

theorem unique_solution_floor_equation :
  ∃! n : ℤ, ⌊(n^2 : ℚ) / 5⌋ - ⌊(n : ℚ) / 2⌋^2 = 3 :=
by sorry

end unique_solution_floor_equation_l3205_320549


namespace perpendicular_lines_line_through_P_l3205_320569

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x + m * y - 6 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := m * x + y - 3 = 0

-- Define perpendicularity of lines
def perpendicular (m : ℝ) : Prop := 
  (m = 0) ∨ (m = -3)

-- Define point P on l₂
def P_on_l₂ (m : ℝ) : Prop := l₂ m 1 (2 * m)

-- Define line l passing through P with opposite intercepts
def line_l (x y : ℝ) : Prop := 
  (2 * x - y = 0) ∨ (x - y + 1 = 0)

-- Theorem statements
theorem perpendicular_lines (m : ℝ) : 
  (∀ x y, l₁ m x y ∧ l₂ m x y → perpendicular m) := sorry

theorem line_through_P (m : ℝ) : 
  P_on_l₂ m → (∀ x y, line_l x y) := sorry

end perpendicular_lines_line_through_P_l3205_320569


namespace fourth_row_from_bottom_sum_l3205_320579

/-- Represents a position in the grid -/
structure Position :=
  (row : ℕ)
  (col : ℕ)

/-- Represents the grid and its properties -/
structure Grid :=
  (size : ℕ)
  (start : Position)
  (max_num : ℕ)

/-- Represents the spiral filling of the grid -/
def spiral_fill (g : Grid) : Position → ℕ := sorry

/-- The sum of the greatest and least number in a given row -/
def row_sum (g : Grid) (row : ℕ) : ℕ := sorry

/-- The main theorem to prove -/
theorem fourth_row_from_bottom_sum :
  let g : Grid := {
    size := 16,
    start := { row := 8, col := 8 },
    max_num := 256
  }
  row_sum g 4 = 497 := by sorry

end fourth_row_from_bottom_sum_l3205_320579


namespace sqrt_seven_to_sixth_l3205_320546

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by sorry

end sqrt_seven_to_sixth_l3205_320546


namespace solution_set_part_i_range_of_a_part_ii_l3205_320539

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Part I
theorem solution_set_part_i :
  {x : ℝ | f 1 x ≥ |x + 1| + 1} = {x : ℝ | x > 0.5} := by sorry

-- Part II
theorem range_of_a_part_ii :
  {a : ℝ | {x : ℝ | x ≤ -1} ⊆ {x : ℝ | f a x + 3*x ≤ 0}} = {a : ℝ | -4 ≤ a ∧ a ≤ 2} := by sorry

end solution_set_part_i_range_of_a_part_ii_l3205_320539


namespace molecule_count_l3205_320503

-- Define Avogadro's constant
def avogadro_constant : ℝ := 6.022e23

-- Define the number of molecules
def number_of_molecules : ℝ := 3e26

-- Theorem to prove
theorem molecule_count : number_of_molecules = 3e26 := by
  sorry

end molecule_count_l3205_320503


namespace system_solution_l3205_320568

theorem system_solution (n k m : ℕ+) 
  (eq1 : n + k = (Nat.gcd n k)^2)
  (eq2 : k + m = (Nat.gcd k m)^2) :
  n = 2 ∧ k = 2 ∧ m = 2 := by
  sorry

end system_solution_l3205_320568


namespace distribute_five_balls_four_boxes_l3205_320548

/-- The number of ways to distribute n distinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: The number of ways to distribute 5 distinguishable balls into 4 distinguishable boxes is 4^5 -/
theorem distribute_five_balls_four_boxes : distribute_balls 5 4 = 4^5 := by
  sorry

end distribute_five_balls_four_boxes_l3205_320548


namespace initial_ribbon_amount_l3205_320501

/-- The number of gifts Josh is preparing -/
def num_gifts : ℕ := 6

/-- The amount of ribbon used for each gift in yards -/
def ribbon_per_gift : ℕ := 2

/-- The amount of ribbon left after preparing the gifts in yards -/
def leftover_ribbon : ℕ := 6

/-- Theorem: Josh initially has 18 yards of ribbon -/
theorem initial_ribbon_amount :
  num_gifts * ribbon_per_gift + leftover_ribbon = 18 := by
  sorry

end initial_ribbon_amount_l3205_320501


namespace domain_condition_implies_m_range_l3205_320500

theorem domain_condition_implies_m_range (m : ℝ) :
  (∀ x : ℝ, mx^2 + mx + 1 ≥ 0) ↔ 0 ≤ m ∧ m ≤ 4 :=
sorry

end domain_condition_implies_m_range_l3205_320500


namespace area_scientific_notation_l3205_320557

/-- Represents the area in square meters -/
def area : ℝ := 216000

/-- Represents the coefficient in scientific notation -/
def coefficient : ℝ := 2.16

/-- Represents the exponent in scientific notation -/
def exponent : ℤ := 5

/-- Theorem stating that the area is equal to its scientific notation representation -/
theorem area_scientific_notation : area = coefficient * (10 : ℝ) ^ exponent := by sorry

end area_scientific_notation_l3205_320557


namespace simplified_expression_value_l3205_320533

theorem simplified_expression_value (a b : ℤ) (ha : a = 2) (hb : b = -3) :
  10 * a^2 * b - (2 * a * b^2 - 2 * (a * b - 5 * a^2 * b)) = -48 := by
  sorry

end simplified_expression_value_l3205_320533


namespace difference_of_differences_l3205_320524

theorem difference_of_differences (a b c : ℤ) 
  (hab : a - b = 2) (hbc : b - c = -3) : a - c = -1 := by
  sorry

end difference_of_differences_l3205_320524


namespace johnny_guitar_practice_l3205_320518

/-- Represents the number of days Johnny has been practicing guitar -/
def current_practice : ℕ := 40

/-- Represents the daily practice amount -/
def daily_practice : ℕ := 2

theorem johnny_guitar_practice :
  let days_to_triple := (3 * current_practice - current_practice) / daily_practice
  (2 * (current_practice - 20 * daily_practice) = current_practice) →
  days_to_triple = 80 := by
  sorry

end johnny_guitar_practice_l3205_320518


namespace complex_number_in_quadrant_iv_l3205_320536

/-- The complex number (2-i)/(1+i) corresponds to a point in Quadrant IV of the complex plane -/
theorem complex_number_in_quadrant_iv : 
  let z : ℂ := (2 - Complex.I) / (1 + Complex.I)
  (z.re > 0) ∧ (z.im < 0) := by sorry

end complex_number_in_quadrant_iv_l3205_320536


namespace pizza_toppings_l3205_320561

theorem pizza_toppings (total_slices : ℕ) (pepperoni_slices : ℕ) (mushroom_slices : ℕ) 
  (h1 : total_slices = 16)
  (h2 : pepperoni_slices = 9)
  (h3 : mushroom_slices = 12)
  (h4 : ∀ slice, slice ∈ Finset.range total_slices → 
    (slice ∈ Finset.range pepperoni_slices ∨ slice ∈ Finset.range mushroom_slices)) :
  mushroom_slices - (pepperoni_slices + mushroom_slices - total_slices) = 7 := by
  sorry

end pizza_toppings_l3205_320561


namespace periodic_odd_function_value_l3205_320592

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem periodic_odd_function_value
  (f : ℝ → ℝ)
  (h_periodic : is_periodic f 6)
  (h_odd : is_odd f)
  (h_value : f (-1) = -1) :
  f 5 = -1 := by
  sorry

end periodic_odd_function_value_l3205_320592


namespace inequality_proof_l3205_320588

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 8*y + 2*z) * (x + 2*y + z) * (x + 4*y + 4*z) ≥ 256*x*y*z := by
  sorry

end inequality_proof_l3205_320588


namespace power_six_equivalence_l3205_320583

theorem power_six_equivalence (m : ℝ) : m^2 * m^4 = m^6 := by
  sorry

end power_six_equivalence_l3205_320583


namespace product_of_sum_and_cube_sum_l3205_320587

theorem product_of_sum_and_cube_sum (c d : ℝ) 
  (h1 : c + d = 10) 
  (h2 : c^3 + d^3 = 370) : 
  c * d = 21 := by
sorry

end product_of_sum_and_cube_sum_l3205_320587


namespace x_range_when_ln_x_less_than_neg_one_l3205_320531

theorem x_range_when_ln_x_less_than_neg_one (x : ℝ) (h : Real.log x < -1) : 0 < x ∧ x < Real.exp (-1) :=
sorry

end x_range_when_ln_x_less_than_neg_one_l3205_320531


namespace difference_of_cubes_divisible_by_27_l3205_320520

theorem difference_of_cubes_divisible_by_27 (a b : ℤ) :
  ∃ k : ℤ, (3 * a + 2)^3 - (3 * b + 2)^3 = 27 * k := by sorry

end difference_of_cubes_divisible_by_27_l3205_320520


namespace correct_sums_l3205_320532

theorem correct_sums (R W : ℕ) : W = 5 * R → R + W = 180 → R = 30 := by
  sorry

end correct_sums_l3205_320532


namespace odd_sum_probability_l3205_320564

def cards : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}

def is_odd_sum (pair : ℕ × ℕ) : Bool :=
  (pair.1 + pair.2) % 2 = 1

def odd_sum_pairs : Finset (ℕ × ℕ) :=
  (cards.product cards).filter (λ pair => pair.1 < pair.2 ∧ is_odd_sum pair)

theorem odd_sum_probability :
  (odd_sum_pairs.card : ℚ) / (cards.card.choose 2) = 5 / 9 := by
  sorry

end odd_sum_probability_l3205_320564


namespace quadratic_minimum_quadratic_minimum_unique_l3205_320511

theorem quadratic_minimum (x : ℝ) : 
  2 * x^2 - 8 * x + 1 ≥ 2 * 2^2 - 8 * 2 + 1 := by
  sorry

theorem quadratic_minimum_unique (x : ℝ) : 
  (2 * x^2 - 8 * x + 1 = 2 * 2^2 - 8 * 2 + 1) → (x = 2) := by
  sorry

end quadratic_minimum_quadratic_minimum_unique_l3205_320511


namespace divisible_by_nine_sequence_l3205_320591

theorem divisible_by_nine_sequence (n : ℕ) : 
  (n % 9 = 0) ∧ 
  (n + 54 ≤ 97) ∧ 
  (∀ k : ℕ, k < 7 → (n + 9 * k) % 9 = 0) →
  n = 36 :=
sorry

end divisible_by_nine_sequence_l3205_320591


namespace molecular_weight_NaOCl_approx_l3205_320537

/-- The atomic weight of Sodium in g/mol -/
def atomic_weight_Na : ℝ := 22.99

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of Chlorine in g/mol -/
def atomic_weight_Cl : ℝ := 35.45

/-- The molecular weight of NaOCl in g/mol -/
def molecular_weight_NaOCl : ℝ := atomic_weight_Na + atomic_weight_O + atomic_weight_Cl

/-- Theorem stating that the molecular weight of NaOCl is approximately 74.44 g/mol -/
theorem molecular_weight_NaOCl_approx :
  ∀ ε > 0, |molecular_weight_NaOCl - 74.44| < ε :=
sorry

end molecular_weight_NaOCl_approx_l3205_320537


namespace matrix_commute_special_case_l3205_320574

open Matrix

theorem matrix_commute_special_case 
  (C D : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : C + D = C * D) 
  (h2 : C * D = !![10, 3; -2, 5]) : 
  D * C = C * D := by
sorry

end matrix_commute_special_case_l3205_320574


namespace complex_quadrant_l3205_320504

theorem complex_quadrant (z : ℂ) (h : -2 * I * z = 1 - I) : 
  0 < z.re ∧ 0 < z.im :=
sorry

end complex_quadrant_l3205_320504


namespace arrangement_count_l3205_320593

/-- The number of volunteers -/
def num_volunteers : ℕ := 5

/-- The number of elderly people -/
def num_elderly : ℕ := 2

/-- The number of positions where the elderly pair can be placed -/
def elderly_pair_positions : ℕ := num_volunteers - 1

/-- The number of arrangements of volunteers -/
def volunteer_arrangements : ℕ := Nat.factorial num_volunteers

/-- The number of arrangements of elderly people -/
def elderly_arrangements : ℕ := Nat.factorial num_elderly

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := elderly_pair_positions * volunteer_arrangements * elderly_arrangements

theorem arrangement_count : total_arrangements = 960 := by
  sorry

end arrangement_count_l3205_320593


namespace warehouse_shoes_l3205_320598

/-- The number of pairs of shoes in a warehouse -/
def total_shoes (blue green purple : ℕ) : ℕ := blue + green + purple

/-- Theorem: The total number of shoes in the warehouse is 1250 -/
theorem warehouse_shoes : ∃ (green : ℕ), 
  let blue := 540
  let purple := 355
  (green = purple) ∧ (total_shoes blue green purple = 1250) := by
  sorry

end warehouse_shoes_l3205_320598


namespace part_not_scrap_l3205_320542

/-- The probability of producing scrap in the first process -/
def p1 : ℝ := 0.01

/-- The probability of producing scrap in the second process -/
def p2 : ℝ := 0.02

/-- The probability that a part is not scrap after two independent processes -/
def prob_not_scrap : ℝ := (1 - p1) * (1 - p2)

theorem part_not_scrap : prob_not_scrap = 0.9702 := by sorry

end part_not_scrap_l3205_320542


namespace cube_volume_surface_area_l3205_320514

theorem cube_volume_surface_area (x : ℝ) : 
  (∃ (s : ℝ), s^3 = 8*x ∧ 6*s^2 = 2*x) → x = 1728 := by
  sorry

end cube_volume_surface_area_l3205_320514


namespace two_digit_number_proof_l3205_320594

theorem two_digit_number_proof : 
  ∀ n : ℕ, 
  (10 ≤ n ∧ n < 100) → -- two-digit number
  (∃ x y : ℕ, n = 10 * x + y ∧ y = x + 3 ∧ n = y * y) → -- conditions
  (n = 25 ∨ n = 36) := by
sorry

end two_digit_number_proof_l3205_320594


namespace linear_equation_solution_l3205_320530

theorem linear_equation_solution : ∃ (x y : ℤ), x + y = 5 := by
  sorry

end linear_equation_solution_l3205_320530


namespace special_divisors_count_l3205_320516

/-- The number of positive integer divisors of 2022^2022 that are divisible by exactly 2022 positive integers -/
def num_special_divisors : ℕ := 6

/-- 2022 factorized as 2 * 3 * 337 -/
def factorization_2022 : ℕ × ℕ × ℕ := (2, 3, 337)

theorem special_divisors_count :
  (factorization_2022.1 * factorization_2022.2.1 * factorization_2022.2.2 = 2022) →
  (∃ (a b c : ℕ), (a + 1) * (b + 1) * (c + 1) = 2022 ∧
    num_special_divisors = (List.length [
      (factorization_2022.1, factorization_2022.2.1, factorization_2022.2.2),
      (factorization_2022.1, factorization_2022.2.2, factorization_2022.2.1),
      (factorization_2022.2.1, factorization_2022.1, factorization_2022.2.2),
      (factorization_2022.2.1, factorization_2022.2.2, factorization_2022.1),
      (factorization_2022.2.2, factorization_2022.1, factorization_2022.2.1),
      (factorization_2022.2.2, factorization_2022.2.1, factorization_2022.1)
    ])) :=
by sorry

end special_divisors_count_l3205_320516


namespace vegetables_for_movie_day_l3205_320541

theorem vegetables_for_movie_day 
  (points_needed : ℕ) 
  (points_per_vegetable : ℕ) 
  (num_students : ℕ) 
  (num_days : ℕ) 
  (h1 : points_needed = 200) 
  (h2 : points_per_vegetable = 2) 
  (h3 : num_students = 25) 
  (h4 : num_days = 10) : 
  (points_needed / (points_per_vegetable * num_students * (num_days / 2))) = 2 := by
  sorry

end vegetables_for_movie_day_l3205_320541


namespace melanie_dimes_count_l3205_320528

/-- The total number of dimes Melanie has after receiving gifts from her parents -/
def total_dimes (initial : ℕ) (from_dad : ℕ) (from_mom : ℕ) : ℕ :=
  initial + from_dad + from_mom

/-- Theorem: Given Melanie's initial dimes and the dimes given by her parents, 
    the total number of dimes Melanie has now is 83. -/
theorem melanie_dimes_count : total_dimes 19 39 25 = 83 := by
  sorry

end melanie_dimes_count_l3205_320528


namespace simplify_fraction_l3205_320596

theorem simplify_fraction (x : ℝ) (h : x > 0) :
  (Real.sqrt x * 3 * x^2) / (x * 6 * x) = 1 := by
  sorry

end simplify_fraction_l3205_320596


namespace sally_napkins_l3205_320573

def tablecloth_length : ℕ := 102
def tablecloth_width : ℕ := 54
def napkin_length : ℕ := 6
def napkin_width : ℕ := 7
def total_material : ℕ := 5844

theorem sally_napkins :
  let tablecloth_area := tablecloth_length * tablecloth_width
  let napkin_area := napkin_length * napkin_width
  let remaining_material := total_material - tablecloth_area
  remaining_material / napkin_area = 8 := by sorry

end sally_napkins_l3205_320573


namespace hyperbola_eccentricity_l3205_320571

theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) :
  let hyperbola := fun (x y : ℝ) => x^2 / a^2 - y^2 = 1
  let c := Real.sqrt (a^2 + 1)
  let e := c / a
  let asymptote_slope := 1 / a
  let perpendicular_slope := -a
  let y_coordinate_P := a * c / (1 + a^2)
  y_coordinate_P = 2 * Real.sqrt 5 / 5 →
  e = Real.sqrt 5 / 2 := by
  sorry

end hyperbola_eccentricity_l3205_320571


namespace min_value_theorem_l3205_320577

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : (x - 1/y)^2 = 16*y/x) :
  (∀ x' y', x' > 0 → y' > 0 → (x' - 1/y')^2 = 16*y'/x' → x + 1/y ≤ x' + 1/y') →
  x^2 + 1/y^2 = 12 := by
sorry

end min_value_theorem_l3205_320577


namespace hundredth_odd_and_plus_ten_l3205_320540

/-- The nth odd positive integer -/
def nthOddPositive (n : ℕ) : ℕ := 2 * n - 1

theorem hundredth_odd_and_plus_ten :
  (nthOddPositive 100 = 199) ∧ (nthOddPositive 100 + 10 = 209) := by
  sorry

end hundredth_odd_and_plus_ten_l3205_320540


namespace complex_problem_l3205_320566

def complex_operation (a b c d : ℂ) : ℂ := a * d - b * c

theorem complex_problem (x y : ℂ) : 
  x = (1 - I) / (1 + I) →
  y = complex_operation (4 * I) (1 + I) (3 - x * I) (x + I) →
  y = -2 - 2 * I :=
by sorry

end complex_problem_l3205_320566


namespace fraction_product_sum_l3205_320563

theorem fraction_product_sum : (1/3 : ℚ) * (17/6 : ℚ) * (3/7 : ℚ) + (1/4 : ℚ) * (1/8 : ℚ) = 101/672 := by
  sorry

end fraction_product_sum_l3205_320563


namespace equal_chance_in_all_methods_l3205_320519

/-- Represents a sampling method -/
structure SamplingMethod where
  name : String
  equal_chance : Bool

/-- Simple random sampling -/
def simple_random_sampling : SamplingMethod :=
  { name := "Simple Random Sampling", equal_chance := true }

/-- Systematic sampling -/
def systematic_sampling : SamplingMethod :=
  { name := "Systematic Sampling", equal_chance := true }

/-- Stratified sampling -/
def stratified_sampling : SamplingMethod :=
  { name := "Stratified Sampling", equal_chance := true }

/-- Theorem: All three sampling methods have equal chance of selection for each individual -/
theorem equal_chance_in_all_methods :
  simple_random_sampling.equal_chance ∧
  systematic_sampling.equal_chance ∧
  stratified_sampling.equal_chance :=
by sorry

end equal_chance_in_all_methods_l3205_320519


namespace book_arrangement_count_l3205_320550

theorem book_arrangement_count : 
  let total_books : ℕ := 12
  let arabic_books : ℕ := 3
  let german_books : ℕ := 4
  let spanish_books : ℕ := 3
  let french_books : ℕ := 2
  let grouped_units : ℕ := 3  -- Arabic, Spanish, French groups
  let total_arrangements : ℕ := 
    (Nat.factorial (grouped_units + german_books)) * 
    (Nat.factorial arabic_books) * 
    (Nat.factorial spanish_books) * 
    (Nat.factorial french_books)
  total_books = arabic_books + german_books + spanish_books + french_books →
  total_arrangements = 362880 := by
sorry

end book_arrangement_count_l3205_320550


namespace cori_age_relation_cori_current_age_l3205_320502

/-- Cori's current age -/
def cori_age : ℕ := sorry

/-- Cori's aunt's current age -/
def aunt_age : ℕ := 19

/-- In 5 years, Cori will be one-third the age of her aunt -/
theorem cori_age_relation : cori_age + 5 = (aunt_age + 5) / 3 := sorry

theorem cori_current_age : cori_age = 3 := by sorry

end cori_age_relation_cori_current_age_l3205_320502


namespace marble_203_is_blue_l3205_320512

/-- Represents the color of a marble -/
inductive Color
| Red
| Blue
| Green

/-- The length of one complete cycle of marbles -/
def cycleLength : Nat := 6 + 5 + 4

/-- The position of a marble within its cycle -/
def positionInCycle (n : Nat) : Nat :=
  n % cycleLength

/-- The color of a marble at a given position within a cycle -/
def colorInCycle (pos : Nat) : Color :=
  if pos ≤ 6 then Color.Red
  else if pos ≤ 11 then Color.Blue
  else Color.Green

/-- The color of the nth marble in the sequence -/
def marbleColor (n : Nat) : Color :=
  colorInCycle (positionInCycle n)

/-- Theorem: The 203rd marble is blue -/
theorem marble_203_is_blue : marbleColor 203 = Color.Blue := by
  sorry

end marble_203_is_blue_l3205_320512


namespace arrangement_probability_l3205_320556

/-- The probability of arranging n(n + 1)/2 distinct numbers into n rows,
    where the i-th row has i numbers, such that the largest number in each row
    is smaller than the largest number in all rows with more numbers. -/
def probability (n : ℕ) : ℚ :=
  (2 ^ n : ℚ) / (Nat.factorial (n + 1) : ℚ)

/-- Theorem stating that the probability of the described arrangement
    is equal to 2^n / (n+1)! -/
theorem arrangement_probability (n : ℕ) :
  probability n = (2 ^ n : ℚ) / (Nat.factorial (n + 1) : ℚ) := by
  sorry

end arrangement_probability_l3205_320556


namespace free_throw_difference_l3205_320510

/-- The number of free-throws made by each player in one minute -/
structure FreeThrows where
  deshawn : ℕ
  kayla : ℕ
  annieka : ℕ

/-- The conditions of the basketball free-throw practice -/
def free_throw_practice (ft : FreeThrows) : Prop :=
  ft.deshawn = 12 ∧
  ft.kayla = ft.deshawn + ft.deshawn / 2 ∧
  ft.annieka = 14 ∧
  ft.annieka < ft.kayla

/-- The theorem stating the difference between Kayla's and Annieka's free-throws -/
theorem free_throw_difference (ft : FreeThrows) 
  (h : free_throw_practice ft) : ft.kayla - ft.annieka = 4 := by
  sorry

#check free_throw_difference

end free_throw_difference_l3205_320510


namespace monica_books_l3205_320553

/-- The number of books Monica read last year -/
def books_last_year : ℕ := sorry

/-- The number of books Monica read this year -/
def books_this_year : ℕ := 2 * books_last_year

/-- The number of books Monica will read next year -/
def books_next_year : ℕ := 2 * books_this_year + 5

theorem monica_books : books_last_year = 16 ∧ books_next_year = 69 := by
  sorry

end monica_books_l3205_320553


namespace trig_identity_l3205_320576

theorem trig_identity (α : ℝ) :
  3.410 * (Real.sin (2 * α))^3 * Real.cos (6 * α) + 
  (Real.cos (2 * α))^3 * Real.sin (6 * α) = 
  3/4 * Real.sin (8 * α) := by
  sorry

end trig_identity_l3205_320576


namespace change_calculation_l3205_320575

/-- Calculates the change in USD given the cost per cup in Euros, payment in Euros, and USD/Euro conversion rate -/
def calculate_change_usd (cost_per_cup_eur : ℝ) (payment_eur : ℝ) (usd_per_eur : ℝ) : ℝ :=
  (payment_eur - cost_per_cup_eur) * usd_per_eur

/-- Proves that the change received is 0.4956 USD given the specified conditions -/
theorem change_calculation :
  let cost_per_cup_eur : ℝ := 0.58
  let payment_eur : ℝ := 1
  let usd_per_eur : ℝ := 1.18
  calculate_change_usd cost_per_cup_eur payment_eur usd_per_eur = 0.4956 := by
  sorry

end change_calculation_l3205_320575


namespace function_equation_implies_identity_l3205_320597

/-- A function satisfying the given functional equation is the identity function. -/
theorem function_equation_implies_identity (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (f y + x^2 + 1) + 2*x = y + (f (x + 1))^2) : 
  ∀ x : ℝ, f x = x := by
sorry

end function_equation_implies_identity_l3205_320597


namespace x_power_6_minus_6x_equals_711_l3205_320582

theorem x_power_6_minus_6x_equals_711 (x : ℝ) (h : x = 3) : x^6 - 6*x = 711 := by
  sorry

end x_power_6_minus_6x_equals_711_l3205_320582


namespace range_of_exponential_function_l3205_320572

theorem range_of_exponential_function :
  ∀ y : ℝ, y > 0 → ∃ x : ℝ, 3^x = y := by
  sorry

end range_of_exponential_function_l3205_320572


namespace max_students_distribution_max_students_is_184_l3205_320515

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem max_students_distribution (pens pencils markers : ℕ) : Prop :=
  pens = 1080 →
  pencils = 920 →
  markers = 680 →
  ∃ (students : ℕ) (pens_per_student pencils_per_student markers_per_student : ℕ),
    students > 0 ∧
    students * pens_per_student = pens ∧
    students * pencils_per_student = pencils ∧
    students * markers_per_student = markers ∧
    pens_per_student > 0 ∧
    pencils_per_student > 0 ∧
    markers_per_student > 0 ∧
    is_prime pencils_per_student ∧
    ∀ (n : ℕ), n > students →
      ¬(∃ (p q r : ℕ),
        p > 0 ∧ q > 0 ∧ r > 0 ∧
        is_prime q ∧
        n * p = pens ∧
        n * q = pencils ∧
        n * r = markers)

theorem max_students_is_184 : max_students_distribution 1080 920 680 → 
  ∃ (pens_per_student pencils_per_student markers_per_student : ℕ),
    184 * pens_per_student = 1080 ∧
    184 * pencils_per_student = 920 ∧
    184 * markers_per_student = 680 ∧
    pens_per_student > 0 ∧
    pencils_per_student > 0 ∧
    markers_per_student > 0 ∧
    is_prime pencils_per_student :=
by
  sorry

end max_students_distribution_max_students_is_184_l3205_320515


namespace rotation_180_complex_l3205_320584

def rotate_180_degrees (z : ℂ) : ℂ := -z

theorem rotation_180_complex :
  rotate_180_degrees (3 - 4*I) = -3 + 4*I :=
by sorry

end rotation_180_complex_l3205_320584


namespace not_perfect_square_sum_l3205_320508

theorem not_perfect_square_sum (x y : ℤ) : 
  ∃ (n : ℤ), (x^2 + x + 1)^2 + (y^2 + y + 1)^2 ≠ n^2 := by
  sorry

end not_perfect_square_sum_l3205_320508


namespace sum_of_five_terms_positive_l3205_320599

def isOddFunction (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def isMonotonicallyDecreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  a ≤ b → f b ≤ f a

def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_five_terms_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (hf_odd : isOddFunction f)
  (hf_mono : ∀ x y, 0 ≤ x → 0 ≤ y → isMonotonicallyDecreasing f x y)
  (ha_arith : isArithmeticSequence a)
  (ha3_neg : a 3 < 0) :
  f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) > 0 :=
sorry

end sum_of_five_terms_positive_l3205_320599


namespace inverse_proportion_order_l3205_320505

theorem inverse_proportion_order (k : ℝ) :
  let f (x : ℝ) := (k^2 + 1) / x
  let y₁ := f (-1)
  let y₂ := f 1
  let y₃ := f 2
  y₁ < y₃ ∧ y₃ < y₂ := by sorry

end inverse_proportion_order_l3205_320505


namespace curve_symmetry_l3205_320538

-- Define the original curve E
def E (x y : ℝ) : Prop := 2 * x^2 + 4 * x * y + 5 * y^2 - 22 = 0

-- Define the line of symmetry l
def l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the symmetric curve E'
def E' (x y : ℝ) : Prop := 5 * x^2 + 4 * x * y + 2 * y^2 + 6 * x - 19 = 0

-- Theorem statement
theorem curve_symmetry :
  ∀ (x y : ℝ), E x y ↔ ∃ (x' y' : ℝ), l ((x + x') / 2) ((y + y') / 2) ∧ E' x' y' :=
sorry

end curve_symmetry_l3205_320538


namespace yellow_parrots_count_l3205_320526

theorem yellow_parrots_count (total : ℕ) (red_fraction : ℚ) (green_fraction : ℚ) :
  total = 180 →
  red_fraction = 2/3 →
  green_fraction = 1/6 →
  (total : ℚ) * (1 - (red_fraction + green_fraction)) = 30 :=
by
  sorry

end yellow_parrots_count_l3205_320526


namespace equations_consistency_l3205_320581

/-- Given a system of equations, prove its consistency -/
theorem equations_consistency 
  (r₁ r₂ r₃ s a b c : ℝ) 
  (eq1 : r₁ * r₂ + r₂ * r₃ + r₃ * r₁ = s^2)
  (eq2 : (s - b) * (s - c) * r₁ + (s - c) * (s - a) * r₂ + (s - a) * (s - b) * r₃ = r₁ * r₂ * r₃) :
  ∃ (r₁' r₂' r₃' s' a' b' c' : ℝ),
    r₁' * r₂' + r₂' * r₃' + r₃' * r₁' = s'^2 ∧
    (s' - b') * (s' - c') * r₁' + (s' - c') * (s' - a') * r₂' + (s' - a') * (s' - b') * r₃' = r₁' * r₂' * r₃' :=
by
  sorry


end equations_consistency_l3205_320581


namespace complement_of_A_l3205_320513

def U : Set Nat := {1, 3, 5, 7, 9}
def A : Set Nat := {1, 5, 7}

theorem complement_of_A :
  (U \ A) = {3, 9} := by sorry

end complement_of_A_l3205_320513


namespace desk_height_in_cm_mm_l3205_320578

/-- The height of a chair in millimeters -/
def chair_height : ℕ := 537

/-- Dong-min's height when standing on the chair, in millimeters -/
def height_on_chair : ℕ := 1900

/-- Dong-min's height when standing on the desk, in millimeters -/
def height_on_desk : ℕ := 2325

/-- The height of the desk in millimeters -/
def desk_height : ℕ := height_on_desk - (height_on_chair - chair_height)

theorem desk_height_in_cm_mm : 
  desk_height = 96 * 10 + 2 := by sorry

end desk_height_in_cm_mm_l3205_320578


namespace solve_equation_l3205_320509

theorem solve_equation : 
  ∃ x : ℝ, (2 * x + 10 = (1/2) * (5 * x + 30)) ∧ (x = -10) := by sorry

end solve_equation_l3205_320509


namespace isosceles_triangle_l3205_320589

theorem isosceles_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  a + b > c ∧ b + c > a ∧ c + a > b →
  c / b = Real.cos C / Real.cos B →
  C = B :=
sorry

end isosceles_triangle_l3205_320589


namespace right_triangle_angle_calculation_l3205_320543

/-- In a triangle ABC with a right angle at A, prove that x = 10/3 degrees -/
theorem right_triangle_angle_calculation (x y : ℝ) : 
  x + y = 40 →
  3 * x + 2 * y = 90 →
  x = 10 / 3 := by
  sorry

end right_triangle_angle_calculation_l3205_320543


namespace triangle_angle_calculation_l3205_320565

theorem triangle_angle_calculation (a b : ℝ) (B : ℝ) (hA : a = Real.sqrt 2) (hB : B = 45 * π / 180) (hb : b = 2) :
  ∃ (A : ℝ), A = 30 * π / 180 ∧ a / Real.sin A = b / Real.sin B := by
  sorry

end triangle_angle_calculation_l3205_320565


namespace average_age_of_six_students_l3205_320523

theorem average_age_of_six_students
  (total_students : Nat)
  (average_age_all : ℝ)
  (num_group1 : Nat)
  (average_age_group1 : ℝ)
  (age_last_student : ℝ)
  (h1 : total_students = 15)
  (h2 : average_age_all = 15)
  (h3 : num_group1 = 8)
  (h4 : average_age_group1 = 14)
  (h5 : age_last_student = 17)
  : ∃ (num_group2 : Nat) (average_age_group2 : ℝ),
    num_group2 = total_students - num_group1 - 1 ∧
    average_age_group2 = 16 :=
by
  sorry

#check average_age_of_six_students

end average_age_of_six_students_l3205_320523


namespace lives_lost_l3205_320534

/-- Represents the number of lives Kaleb had initially -/
def initial_lives : ℕ := 98

/-- Represents the number of lives Kaleb had remaining -/
def remaining_lives : ℕ := 73

/-- Theorem stating that the number of lives Kaleb lost is 25 -/
theorem lives_lost : initial_lives - remaining_lives = 25 := by
  sorry

end lives_lost_l3205_320534


namespace not_cube_sum_l3205_320590

theorem not_cube_sum (a b : ℕ) : ¬ ∃ c : ℤ, (a : ℤ)^3 + (b : ℤ)^3 + 4 = c^3 := by
  sorry

end not_cube_sum_l3205_320590


namespace second_derivative_at_x₀_l3205_320585

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := sorry

-- Define the point x₀
def x₀ : ℝ := sorry

-- Define constants a and b
def a : ℝ := sorry
def b : ℝ := sorry

-- State the theorem
theorem second_derivative_at_x₀ (h : ∀ Δx, f (x₀ + Δx) - f x₀ = a * Δx + b * Δx^2) :
  deriv (deriv f) x₀ = 2 * b := by sorry

end second_derivative_at_x₀_l3205_320585


namespace shortest_tangent_theorem_l3205_320554

noncomputable def circle_C3 (x y : ℝ) : Prop := (x - 8) ^ 2 + (y - 3) ^ 2 = 49

noncomputable def circle_C4 (x y : ℝ) : Prop := (x + 12) ^ 2 + (y + 4) ^ 2 = 16

noncomputable def shortest_tangent_length : ℝ := (Real.sqrt 7840 + Real.sqrt 24181) / 11 - 11

theorem shortest_tangent_theorem :
  ∃ (R S : ℝ × ℝ),
    circle_C3 R.1 R.2 ∧
    circle_C4 S.1 S.2 ∧
    (∀ (P Q : ℝ × ℝ),
      circle_C3 P.1 P.2 →
      circle_C4 Q.1 Q.2 →
      Real.sqrt ((R.1 - S.1) ^ 2 + (R.2 - S.2) ^ 2) ≤ Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)) ∧
    Real.sqrt ((R.1 - S.1) ^ 2 + (R.2 - S.2) ^ 2) = shortest_tangent_length :=
by
  sorry

end shortest_tangent_theorem_l3205_320554


namespace polar_to_rectangular_transformation_l3205_320535

theorem polar_to_rectangular_transformation (x y : ℝ) (r θ : ℝ) 
  (h1 : x = 12 ∧ y = 5)
  (h2 : r = (x^2 + y^2).sqrt)
  (h3 : θ = Real.arctan (y / x)) :
  let new_r := 2 * r^2
  let new_θ := 3 * θ
  (new_r * Real.cos new_θ = 338 * 828 / 2197) ∧
  (new_r * Real.sin new_θ = 338 * 2035 / 2197) := by
sorry

end polar_to_rectangular_transformation_l3205_320535


namespace cross_in_square_l3205_320517

theorem cross_in_square (a : ℝ) (h : a > 0) : 
  (2 * (a/2)^2 + 2 * (a/4)^2 = 810) → a = 36 := by
  sorry

end cross_in_square_l3205_320517


namespace triangle_inequality_theorem_l3205_320559

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  positive_a : 0 < a
  positive_b : 0 < b
  positive_c : 0 < c
  triangle_inequality_ab : a + b > c
  triangle_inequality_bc : b + c > a
  triangle_inequality_ca : c + a > b

-- State the theorem
theorem triangle_inequality_theorem (t : Triangle) :
  t.a^2 * t.c * (t.a - t.b) + t.b^2 * t.a * (t.b - t.c) + t.c^2 * t.b * (t.c - t.a) ≥ 0 := by
  sorry

end triangle_inequality_theorem_l3205_320559


namespace bus_problem_l3205_320580

theorem bus_problem (initial : ℕ) (got_on : ℕ) (final : ℕ) : 
  initial = 28 → got_on = 82 → final = 30 → 
  ∃ (got_off : ℕ), got_on - got_off = 2 ∧ initial + got_on - got_off = final :=
by sorry

end bus_problem_l3205_320580


namespace house_sale_buyback_loss_l3205_320567

/-- Represents the financial outcome of a house sale and buyback transaction -/
def houseSaleBuybackOutcome (initialValue : ℝ) (profitPercentage : ℝ) (lossPercentage : ℝ) : ℝ :=
  let salePrice := initialValue * (1 + profitPercentage)
  let buybackPrice := salePrice * (1 - lossPercentage)
  buybackPrice - initialValue

/-- Theorem stating that the financial outcome for the given scenario results in a $240 loss -/
theorem house_sale_buyback_loss :
  houseSaleBuybackOutcome 12000 0.2 0.15 = -240 := by
  sorry

end house_sale_buyback_loss_l3205_320567


namespace vanessa_savings_time_l3205_320547

/-- Calculates the number of weeks needed to save for a dress -/
def weeks_to_save (dress_cost : ℕ) (initial_savings : ℕ) (weekly_allowance : ℕ) (weekly_spending : ℕ) : ℕ :=
  let additional_needed := dress_cost - initial_savings
  let weekly_savings := weekly_allowance - weekly_spending
  (additional_needed + weekly_savings - 1) / weekly_savings

/-- Proof that Vanessa needs exactly 3 weeks to save for the dress -/
theorem vanessa_savings_time : 
  weeks_to_save 80 20 30 10 = 3 := by
  sorry

end vanessa_savings_time_l3205_320547


namespace sparklers_to_crackers_theorem_value_comparison_theorem_l3205_320544

/-- Represents the exchange rates between different holiday items -/
structure ExchangeRates where
  ornament_to_cracker : ℚ
  sparkler_to_garland : ℚ
  ornament_to_garland : ℚ

/-- Converts sparklers to crackers based on the given exchange rates -/
def sparklers_to_crackers (rates : ExchangeRates) (sparklers : ℚ) : ℚ :=
  let garlands := (sparklers / 5) * 2
  let ornaments := garlands * 4
  ornaments * rates.ornament_to_cracker

/-- Compares the value of ornaments and crackers to sparklers -/
def compare_values (rates : ExchangeRates) (ornaments crackers sparklers : ℚ) : Prop :=
  ornaments * rates.ornament_to_cracker + crackers > 
  (sparklers / 5) * 2 * 4 * rates.ornament_to_cracker

/-- Theorem stating the equivalence of 10 sparklers to 32 crackers -/
theorem sparklers_to_crackers_theorem (rates : ExchangeRates) : 
  sparklers_to_crackers rates 10 = 32 :=
by sorry

/-- Theorem comparing the value of 5 ornaments and 1 cracker to 2 sparklers -/
theorem value_comparison_theorem (rates : ExchangeRates) : 
  compare_values rates 5 1 2 :=
by sorry

end sparklers_to_crackers_theorem_value_comparison_theorem_l3205_320544


namespace second_digit_prime_in_powers_l3205_320521

-- Define four-digit powers of 2 and 5
def fourDigitPowersOf2 : Set ℕ := {n : ℕ | ∃ m : ℕ, n = 2^m ∧ 1000 ≤ n ∧ n < 10000}
def fourDigitPowersOf5 : Set ℕ := {n : ℕ | ∃ m : ℕ, n = 5^m ∧ 1000 ≤ n ∧ n < 10000}

-- Function to get the second digit of a number
def secondDigit (n : ℕ) : ℕ := (n / 100) % 10

-- The theorem to prove
theorem second_digit_prime_in_powers :
  ∃! p : ℕ, 
    Nat.Prime p ∧ 
    (∃ n ∈ fourDigitPowersOf2, secondDigit n = p) ∧
    (∃ n ∈ fourDigitPowersOf5, secondDigit n = p) :=
  sorry

end second_digit_prime_in_powers_l3205_320521


namespace dinner_attendees_l3205_320558

theorem dinner_attendees (total_clinks : ℕ) : total_clinks = 45 → ∃ x : ℕ, x = 10 ∧ x * (x - 1) / 2 = total_clinks := by
  sorry

end dinner_attendees_l3205_320558
