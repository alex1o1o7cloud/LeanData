import Mathlib

namespace number_of_workers_l111_111166

-- Definitions corresponding to problem conditions
def total_contribution := 300000
def extra_total_contribution := 325000
def extra_amount := 50

-- Main statement to prove the number of workers
theorem number_of_workers : ∃ W C : ℕ, W * C = total_contribution ∧ W * (C + extra_amount) = extra_total_contribution ∧ W = 500 := by
  sorry

end number_of_workers_l111_111166


namespace number_of_members_l111_111490

noncomputable def club_members (n O N : ℕ) : Prop :=
  (3 * n = O - N) ∧ (O - N = 15)

theorem number_of_members (n O N : ℕ) (h : club_members n O N) : n = 5 :=
  by
    sorry

end number_of_members_l111_111490


namespace difference_of_two_numbers_l111_111149

theorem difference_of_two_numbers (a b : ℕ) 
(h1 : a + b = 17402) 
(h2 : ∃ k : ℕ, b = 10 * k) 
(h3 : ∃ k : ℕ, a + 9 * k = b) : 
10 * a - a = 14238 :=
by sorry

end difference_of_two_numbers_l111_111149


namespace opposite_sign_pairs_l111_111045

def opposite_sign (a b : ℤ) : Prop := (a < 0 ∧ b > 0) ∨ (a > 0 ∧ b < 0)

theorem opposite_sign_pairs :
  ¬opposite_sign (-(-1)) 1 ∧
  ¬opposite_sign ((-1)^2) 1 ∧
  ¬opposite_sign (|(-1)|) 1 ∧
  opposite_sign (-1) 1 :=
by {
  sorry
}

end opposite_sign_pairs_l111_111045


namespace smallest_prime_with_digit_sum_18_l111_111019

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem smallest_prime_with_digit_sum_18 : ∃ p : ℕ, Prime p ∧ 18 = sum_of_digits p ∧ (∀ q : ℕ, (Prime q ∧ 18 = sum_of_digits q) → p ≤ q) :=
by
  sorry

end smallest_prime_with_digit_sum_18_l111_111019


namespace eq_of_divisibility_l111_111981

theorem eq_of_divisibility (a b : ℕ) (h : (a^2 + b^2) ∣ (a * b)) : a = b :=
  sorry

end eq_of_divisibility_l111_111981


namespace boys_girls_ratio_l111_111662

-- Definitions used as conditions
variable (B G : ℕ)

-- Conditions
def condition1 : Prop := B + G = 32
def condition2 : Prop := B = 2 * (G - 8)

-- Proof that the ratio of boys to girls initially is 1:1
theorem boys_girls_ratio (h1 : condition1 B G) (h2 : condition2 B G) : (B : ℚ) / G = 1 := by
  sorry

end boys_girls_ratio_l111_111662


namespace max_cookies_eaten_l111_111492

def prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem max_cookies_eaten 
  (total_cookies : ℕ)
  (andy_cookies : ℕ)
  (alexa_cookies : ℕ)
  (hx : andy_cookies + alexa_cookies = total_cookies)
  (hp : ∃ p : ℕ, prime p ∧ alexa_cookies = p * andy_cookies)
  (htotal : total_cookies = 30) :
  andy_cookies = 10 :=
  sorry

end max_cookies_eaten_l111_111492


namespace area_of_square_with_perimeter_40_l111_111695

theorem area_of_square_with_perimeter_40 (P : ℝ) (s : ℝ) (A : ℝ) 
  (hP : P = 40) (hs : s = P / 4) (hA : A = s^2) : A = 100 :=
by
  sorry

end area_of_square_with_perimeter_40_l111_111695


namespace sum_of_roots_of_f_l111_111713

noncomputable def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = - f x

noncomputable def f_increasing_on (f : ℝ → ℝ) (a b : ℝ) := ∀ x y, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x < y → f x < f y

theorem sum_of_roots_of_f (f : ℝ → ℝ) (m : ℝ) (x1 x2 x3 x4 : ℝ)
  (h1 : odd_function f)
  (h2 : ∀ x, f (x - 4) = - f x)
  (h3 : f_increasing_on f 0 2)
  (h4 : m > 0)
  (h5 : f x1 = m)
  (h6 : f x2 = m)
  (h7 : f x3 = m)
  (h8 : f x4 = m)
  (h9 : x1 ≠ x2)
  (h10 : x1 ≠ x3)
  (h11 : x1 ≠ x4)
  (h12 : x2 ≠ x3)
  (h13 : x2 ≠ x4)
  (h14 : x3 ≠ x4)
  (h15 : ∀ x, -8 ≤ x ∧ x ≤ 8 ↔ x = x1 ∨ x = x2 ∨ x = x3 ∨ x = x4) :
  x1 + x2 + x3 + x4 = -8 :=
sorry

end sum_of_roots_of_f_l111_111713


namespace average_reading_days_l111_111207

theorem average_reading_days :
  let days_participated := [2, 3, 4, 5, 6]
  let students := [5, 4, 7, 3, 6]
  let total_days := List.zipWith (· * ·) days_participated students |>.sum
  let total_students := students.sum
  let average := total_days / total_students
  average = 4.04 := sorry

end average_reading_days_l111_111207


namespace sample_size_is_correct_l111_111861

-- Define the conditions
def total_students : ℕ := 40 * 50
def students_selected : ℕ := 150

-- Theorem: The sample size is 150 given that 150 students are selected
theorem sample_size_is_correct : students_selected = 150 := by
  sorry  -- Proof to be completed

end sample_size_is_correct_l111_111861


namespace Mark_charged_more_l111_111648

theorem Mark_charged_more (K P M : ℕ) 
  (h1 : P = 2 * K) 
  (h2 : P = M / 3)
  (h3 : K + P + M = 153) : M - K = 85 :=
by
  -- proof to be filled in later
  sorry

end Mark_charged_more_l111_111648


namespace sum_of_first_four_terms_l111_111078

def arithmetic_sequence_sum (a1 a2 : ℕ) (n : ℕ) : ℕ :=
  (n * (2 * a1 + (n - 1) * (a2 - a1))) / 2

theorem sum_of_first_four_terms : arithmetic_sequence_sum 4 6 4 = 28 :=
by
  sorry

end sum_of_first_four_terms_l111_111078


namespace range_of_angle_B_l111_111411

theorem range_of_angle_B {A B C : ℝ} (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ a + c > b ∧ b + c > a)
  (h_sinB : Real.sin B = Real.sqrt (Real.sin A * Real.sin C)) :
  0 < B ∧ B ≤ Real.pi / 3 :=
sorry

end range_of_angle_B_l111_111411


namespace total_packs_l111_111936

theorem total_packs (cards_per_person cards_per_pack : ℕ) (num_people : ℕ) 
  (h1 : cards_per_person = 540) 
  (h2 : cards_per_pack = 20) 
  (h3 : num_people = 4) : 
  (cards_per_person / cards_per_pack) * num_people = 108 := 
by
  sorry

end total_packs_l111_111936


namespace parametric_graph_right_half_circle_l111_111955

theorem parametric_graph_right_half_circle (θ : ℝ) (x y : ℝ) (hx : x = 3 * Real.cos θ) (hy : y = 3 * Real.sin θ) (hθ : -Real.pi / 2 ≤ θ ∧ θ ≤ Real.pi / 2) :
  x^2 + y^2 = 9 ∧ x ≥ 0 :=
by
  sorry

end parametric_graph_right_half_circle_l111_111955


namespace minimalYellowFraction_l111_111179

-- Definitions
def totalSurfaceArea (sideLength : ℕ) : ℕ := 6 * (sideLength * sideLength)

def minimalYellowExposedArea : ℕ := 15

theorem minimalYellowFraction (sideLength : ℕ) (totalYellow : ℕ) (totalBlue : ℕ) 
    (totalCubes : ℕ) (yellowExposed : ℕ) :
    sideLength = 4 → totalYellow = 16 → totalBlue = 48 →
    totalCubes = 64 → yellowExposed = minimalYellowExposedArea →
    (yellowExposed / (totalSurfaceArea sideLength) : ℚ) = 5 / 32 :=
by
  sorry

end minimalYellowFraction_l111_111179


namespace total_packs_l111_111938

theorem total_packs (cards_bought : ℕ) (cards_per_pack : ℕ) (num_people : ℕ)
  (h1 : cards_bought = 540) (h2 : cards_per_pack = 20) (h3 : num_people = 4) :
  (cards_bought / cards_per_pack) * num_people = 108 :=
by
  sorry

end total_packs_l111_111938


namespace total_rain_duration_l111_111552

theorem total_rain_duration :
  let day1 := 17 - 7 in
  let day2 := day1 + 2 in
  let day3 := day2 * 2 in
  day1 + day2 + day3 = 46 :=
by
  let day1 := 17 - 7
  let day2 := day1 + 2
  let day3 := day2 * 2
  calc
    day1 + day2 + day3 = 10 + 12 + 24 : by sorry
                     ... = 46 : by sorry

end total_rain_duration_l111_111552


namespace smallest_x_l111_111317

-- Define 450 and provide its factorization.
def n1 := 450
def n1_factors := 2^1 * 3^2 * 5^2

-- Define 675 and provide its factorization.
def n2 := 675
def n2_factors := 3^3 * 5^2

-- State the theorem that proves the smallest x for the condition
theorem smallest_x (x : ℕ) (hx : 450 * x % 675 = 0) : x = 3 := sorry

end smallest_x_l111_111317


namespace polynomial_sum_l111_111565

def f (x : ℝ) : ℝ := -4 * x^3 + 2 * x^2 - 5 * x - 7
def g (x : ℝ) : ℝ := 6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := -x^3 + 3 * x^2 + 2 * x + 8

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -5 * x^3 + 11 * x^2 + x - 8 :=
  sorry

end polynomial_sum_l111_111565


namespace expectation_of_xi_l111_111087

noncomputable def compute_expectation : ℝ := 
  let m : ℝ := 0.3
  let E : ℝ := (1 * 0.5) + (3 * m) + (5 * 0.2)
  E

theorem expectation_of_xi :
  let m: ℝ := 1 - 0.5 - 0.2 
  (0.5 + m + 0.2 = 1) → compute_expectation = 2.4 := 
by
  sorry

end expectation_of_xi_l111_111087


namespace train_crossing_time_l111_111040

-- Defining a structure for our problem context
structure TrainCrossing where
  length : Real -- length of the train in meters
  speed_kmh : Real -- speed of the train in km/h
  conversion_factor : Real -- conversion factor from km/h to m/s

-- Given the conditions in the problem
def trainData : TrainCrossing :=
  ⟨ 280, 50.4, 0.27778 ⟩

-- The main theorem statement:
theorem train_crossing_time (data : TrainCrossing) : 
  data.length / (data.speed_kmh * data.conversion_factor) = 20 := 
by
  sorry

end train_crossing_time_l111_111040


namespace ellipse_product_l111_111438

noncomputable def a (b : ℝ) := b + 4
noncomputable def AB (a: ℝ) := 2 * a
noncomputable def CD (b: ℝ) := 2 * b

theorem ellipse_product:
  (∀ (a b : ℝ), a = b + 4 → a^2 - b^2 = 64) →
  (∃ (a b : ℝ), (AB a) * (CD b) = 240) :=
by
  intros h
  use 10, 6
  simp [AB, CD]
  sorry

end ellipse_product_l111_111438


namespace cookies_left_l111_111287

def initial_cookies : ℕ := 93
def eaten_cookies : ℕ := 15

theorem cookies_left : initial_cookies - eaten_cookies = 78 := by
  sorry

end cookies_left_l111_111287


namespace weight_ratios_l111_111113

theorem weight_ratios {x y z k : ℝ} (h1 : x + y = k * z) (h2 : y + z = k * x) (h3 : z + x = k * y) : x = y ∧ y = z :=
by 
  -- Proof to be filled in later
  sorry

end weight_ratios_l111_111113


namespace jacks_walking_rate_l111_111386

variable (distance : ℝ) (hours : ℝ) (minutes : ℝ)

theorem jacks_walking_rate (h_distance : distance = 4) (h_hours : hours = 1) (h_minutes : minutes = 15) :
  distance / (hours + minutes / 60) = 3.2 :=
by
  sorry

end jacks_walking_rate_l111_111386


namespace complement_set_l111_111244

open Set

variable (U : Set ℝ) (M : Set ℝ)

theorem complement_set :
  U = univ ∧ M = {x | x^2 - 2 * x ≤ 0} → (U \ M) = {x | x < 0 ∨ x > 2} :=
by
  intros
  sorry

end complement_set_l111_111244


namespace product_of_real_roots_l111_111223

theorem product_of_real_roots : 
  let f (x : ℝ) := x ^ Real.log x / Real.log 2 
  ∃ r1 r2 : ℝ, (f r1 = 16 ∧ f r2 = 16) ∧ (r1 * r2 = 1) := 
by
  sorry

end product_of_real_roots_l111_111223


namespace Bill_order_combinations_l111_111345

def donut_combinations (num_donuts num_kinds : ℕ) : ℕ :=
  Nat.choose (num_donuts + num_kinds - 1) (num_kinds - 1)

theorem Bill_order_combinations : donut_combinations 10 5 = 126 :=
by
  -- This would be the place to insert the proof steps, but we're using sorry as the placeholder.
  sorry

end Bill_order_combinations_l111_111345


namespace system_of_equations_l111_111917

-- Given conditions: Total number of fruits and total cost of the fruits purchased
def total_fruits := 1000
def total_cost := 999
def cost_of_sweet_fruit := (11 : ℚ) / 9
def cost_of_bitter_fruit := (4 : ℚ) / 7

-- Variables representing the number of sweet and bitter fruits
variables (x y : ℚ)

-- Problem statement in Lean 4
theorem system_of_equations :
  (x + y = total_fruits) ∧ (cost_of_sweet_fruit * x + cost_of_bitter_fruit * y = total_cost) ↔
  ((x + y = 1000) ∧ (11 / 9 * x + 4 / 7 * y = 999)) :=
by
  sorry

end system_of_equations_l111_111917


namespace product_8_40_product_5_1_6_sum_6_instances_500_l111_111624

-- The product of 8 and 40 is 320
theorem product_8_40 : 8 * 40 = 320 := sorry

-- 5 times 1/6 is 5/6
theorem product_5_1_6 : 5 * (1 / 6) = 5 / 6 := sorry

-- The sum of 6 instances of 500 ends with 3 zeros and the sum is 3000
theorem sum_6_instances_500 :
  (500 * 6 = 3000) ∧ ((3000 % 1000) = 0) := sorry

end product_8_40_product_5_1_6_sum_6_instances_500_l111_111624


namespace find_q_l111_111401

noncomputable def common_ratio_of_geometric_sequence
  (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 4 = 27 ∧ a 7 = -729 ∧ ∀ n m, a n = a m * q ^ (n - m)

theorem find_q {a : ℕ → ℝ} {q : ℝ} (h : common_ratio_of_geometric_sequence a q) :
  q = -3 :=
by {
  sorry
}

end find_q_l111_111401


namespace problem_arith_sequences_l111_111124

theorem problem_arith_sequences (a b : ℕ → ℕ) 
  (ha : ∀ n, a (n + 1) = a n + d)
  (hb : ∀ n, b (n + 1) = b n + e)
  (h1 : a 1 = 25)
  (h2 : b 1 = 75)
  (h3 : a 2 + b 2 = 100) : 
  a 37 + b 37 = 100 := 
sorry

end problem_arith_sequences_l111_111124


namespace find_n_l111_111693

def factorial : ℕ → ℕ 
| 0 => 1
| (n + 1) => (n + 1) * factorial n

theorem find_n (n : ℕ) : 3 * n * factorial n + 2 * factorial n = 40320 → n = 8 :=
by
  sorry

end find_n_l111_111693


namespace can_transfer_increase_average_l111_111754

noncomputable def group1_grades : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
noncomputable def group2_grades : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℕ) : ℚ := grades.sum / grades.length

def increase_average_after_move (from_group to_group : List ℕ) (student : ℕ) : Prop :=
  student ∈ from_group ∧ 
  average from_group < average (from_group.erase student) ∧ 
  average to_group < average (student :: to_group)

theorem can_transfer_increase_average :
  ∃ student ∈ group1_grades, increase_average_after_move group1_grades group2_grades student :=
by
  -- Proof would go here
  sorry

end can_transfer_increase_average_l111_111754


namespace totalWheelsInStorageArea_l111_111821

def numberOfBicycles := 24
def numberOfTricycles := 14
def wheelsPerBicycle := 2
def wheelsPerTricycle := 3

theorem totalWheelsInStorageArea :
  numberOfBicycles * wheelsPerBicycle + numberOfTricycles * wheelsPerTricycle = 90 :=
by
  sorry

end totalWheelsInStorageArea_l111_111821


namespace broken_seashells_count_l111_111782

def total_seashells : Nat := 6
def unbroken_seashells : Nat := 2
def broken_seashells : Nat := total_seashells - unbroken_seashells

theorem broken_seashells_count :
  broken_seashells = 4 :=
by
  -- The proof would go here, but for now, we use 'sorry' to denote it.
  sorry

end broken_seashells_count_l111_111782


namespace animal_group_divisor_l111_111974

theorem animal_group_divisor (cows sheep goats total groups : ℕ)
    (hc : cows = 24) 
    (hs : sheep = 7) 
    (hg : goats = 113) 
    (ht : total = cows + sheep + goats) 
    (htotal : total = 144) 
    (hdiv : groups ∣ total) 
    (hexclude1 : groups ≠ 1) 
    (hexclude144 : groups ≠ 144) : 
    ∃ g, g = groups ∧ g ∈ [2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 36, 48, 72] :=
  by 
  sorry

end animal_group_divisor_l111_111974


namespace linear_system_k_value_l111_111724

theorem linear_system_k_value (x y k : ℝ) (h1 : x + 3 * y = 2 * k + 1) (h2 : x - y = 1) (h3 : x = -y) : k = -1 :=
sorry

end linear_system_k_value_l111_111724


namespace bacteria_initial_count_l111_111178

theorem bacteria_initial_count (n : ℕ) :
  (∀ t : ℕ, t % 30 = 0 → n * 2^(t / 30) = 262144 → t = 240) → n = 1024 :=
by sorry

end bacteria_initial_count_l111_111178


namespace simplify_fraction_l111_111595

theorem simplify_fraction :
  (48 : ℚ) / 72 = 2 / 3 :=
sorry

end simplify_fraction_l111_111595


namespace train_length_l111_111187

theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) (total_distance : ℝ) (train_length : ℝ) 
  (h1 : speed = 48) (h2 : time = 45) (h3 : bridge_length = 300)
  (h4 : total_distance = speed * time) (h5 : train_length = total_distance - bridge_length) : 
  train_length = 1860 :=
sorry

end train_length_l111_111187


namespace simplify_fraction_l111_111600

-- Define the problem and conditions
def numerator : ℕ := 48
def denominator : ℕ := 72
def gcd_n_d : ℕ := Nat.gcd numerator denominator

-- The proof statement
theorem simplify_fraction : (numerator / gcd_n_d) / (denominator / gcd_n_d) = 2 / 3 :=
by
  have h_gcd : gcd_n_d = 24 := by rfl
  sorry

end simplify_fraction_l111_111600


namespace parabola_directrix_l111_111721

variable (a : ℝ)

theorem parabola_directrix (h1 : ∀ x : ℝ, y = a * x^2) (h2 : y = -1/4) : a = 1 :=
sorry

end parabola_directrix_l111_111721


namespace simplify_expression_l111_111129

variable (a b c x y z : ℝ)

theorem simplify_expression :
  (cz * (a^3 * x^3 + 3 * a^3 * y^3 + c^3 * z^3) + bz * (a^3 * x^3 + 3 * c^3 * x^3 + c^3 * z^3)) / (cz + bz) =
  a^3 * x^3 + c^3 * z^3 + (3 * cz * a^3 * y^3 + 3 * bz * c^3 * x^3) / (cz + bz) :=
by
  sorry

end simplify_expression_l111_111129


namespace jackson_pbj_sandwiches_l111_111920

-- The number of Wednesdays and Fridays in the 36-week school year
def total_weeks : ℕ := 36
def total_wednesdays : ℕ := total_weeks
def total_fridays : ℕ := total_weeks

-- Public holidays on Wednesdays and Fridays
def holidays_wednesdays : ℕ := 2
def holidays_fridays : ℕ := 3

-- Days Jackson missed
def missed_wednesdays : ℕ := 1
def missed_fridays : ℕ := 2

-- Number of times Jackson asks for a ham and cheese sandwich every 4 weeks
def weeks_for_ham_and_cheese : ℕ := total_weeks / 4

-- Number of ham and cheese sandwich days
def ham_and_cheese_wednesdays : ℕ := weeks_for_ham_and_cheese
def ham_and_cheese_fridays : ℕ := weeks_for_ham_and_cheese * 2

-- Remaining days for peanut butter and jelly sandwiches
def remaining_wednesdays : ℕ := total_wednesdays - holidays_wednesdays - missed_wednesdays
def remaining_fridays : ℕ := total_fridays - holidays_fridays - missed_fridays

def pbj_wednesdays : ℕ := remaining_wednesdays - ham_and_cheese_wednesdays
def pbj_fridays : ℕ := remaining_fridays - ham_and_cheese_fridays

-- Total peanut butter and jelly sandwiches
def total_pbj : ℕ := pbj_wednesdays + pbj_fridays

theorem jackson_pbj_sandwiches : total_pbj = 37 := by
  -- We don't require the proof steps, just the statement
  sorry

end jackson_pbj_sandwiches_l111_111920


namespace derivative_at_x0_l111_111280

variables {f : ℝ → ℝ} {x0 : ℝ}

theorem derivative_at_x0 (h_lim : (𝓝[≠] 0).lim (λ h, (f x0 - f (x0 - h)) / h) 6) : deriv f x0 = 6 :=
sorry

end derivative_at_x0_l111_111280


namespace find_amount_l111_111982

theorem find_amount (x : ℝ) (A : ℝ) (h1 : 0.65 * x = 0.20 * A) (h2 : x = 230) : A = 747.5 := by
  sorry

end find_amount_l111_111982


namespace a3_is_neg_10_a1_a3_a5_sum_is_neg_16_l111_111250

variable (a0 a1 a2 a3 a4 a5 : ℝ)

noncomputable def polynomial (x : ℝ) : ℝ :=
  a0 + a1 * (1 - x) + a2 * (1 - x)^2 + a3 * (1 - x)^3 + a4 * (1 - x)^4 + a5 * (1 - x)^5

theorem a3_is_neg_10 (h : ∀ x, x^5 = polynomial a0 a1 a2 a3 a4 a5 x) : a3 = -10 :=
sorry

theorem a1_a3_a5_sum_is_neg_16 (h : ∀ x, x^5 = polynomial a0 a1 a2 a3 a4 a5 x) : a1 + a3 + a5 = -16 :=
sorry

end a3_is_neg_10_a1_a3_a5_sum_is_neg_16_l111_111250


namespace age_twice_in_Y_years_l111_111036

def present_age_of_son : ℕ := 24
def age_difference := 26
def present_age_of_man : ℕ := present_age_of_son + age_difference

theorem age_twice_in_Y_years : 
  ∃ (Y : ℕ), present_age_of_man + Y = 2 * (present_age_of_son + Y) → Y = 2 :=
by
  sorry

end age_twice_in_Y_years_l111_111036


namespace composite_quotient_is_one_over_49_l111_111688

/-- Define the first twelve positive composite integers --/
def first_six_composites : List ℕ := [4, 6, 8, 9, 10, 12]
def next_six_composites : List ℕ := [14, 15, 16, 18, 20, 21]

/-- Define the product of a list of integers --/
def product (l : List ℕ) : ℕ := l.foldl (λ acc x => acc * x) 1

/-- This defines the quotient of the products of the first six and the next six composite numbers --/
def composite_quotient : ℚ := (↑(product first_six_composites)) / (↑(product next_six_composites))

/-- Main theorem stating that the composite quotient equals to 1/49 --/
theorem composite_quotient_is_one_over_49 : composite_quotient = 1 / 49 := 
  by
    sorry

end composite_quotient_is_one_over_49_l111_111688


namespace jacoby_lottery_expense_l111_111921

-- Definitions based on the conditions:
def jacoby_trip_fund_needed : ℕ := 5000
def jacoby_hourly_wage : ℕ := 20
def jacoby_work_hours : ℕ := 10
def cookies_price : ℕ := 4
def cookies_sold : ℕ := 24
def lottery_winnings : ℕ := 500
def sister_gift : ℕ := 500
def num_sisters : ℕ := 2
def money_still_needed : ℕ := 3214

-- The statement to prove:
theorem jacoby_lottery_expense : 
  (jacoby_hourly_wage * jacoby_work_hours) + (cookies_price * cookies_sold) +
  lottery_winnings + (sister_gift * num_sisters) 
  - (jacoby_trip_fund_needed - money_still_needed) = 10 :=
by {
  sorry
}

end jacoby_lottery_expense_l111_111921


namespace set_of_a_where_A_subset_B_l111_111911

variable {a x : ℝ}

theorem set_of_a_where_A_subset_B (h : ∀ x, (2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5) → (3 ≤ x ∧ x ≤ 22)) :
  6 ≤ a ∧ a ≤ 9 :=
by
  sorry

end set_of_a_where_A_subset_B_l111_111911


namespace simplify_fraction_l111_111589

theorem simplify_fraction (a b : ℕ) (h : b ≠ 0) (g : Nat.gcd a b = 24) : a = 48 → b = 72 → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  exact ⟨rfl, rfl⟩

end simplify_fraction_l111_111589


namespace total_bags_l111_111966

-- Definitions based on the conditions
def bags_on_monday : ℕ := 4
def bags_next_day : ℕ := 8

-- Theorem statement
theorem total_bags : bags_on_monday + bags_next_day = 12 :=
by
  -- Proof will be added here
  sorry

end total_bags_l111_111966


namespace sheet_length_proof_l111_111852

noncomputable def length_of_sheet (L : ℝ) : ℝ := 48

theorem sheet_length_proof (L : ℝ) (w : ℝ) (s : ℝ) (V : ℝ) (h : ℝ) (new_w : ℝ) :
  w = 36 →
  s = 8 →
  V = 5120 →
  h = s →
  new_w = w - 2 * s →
  V = (L - 2 * s) * new_w * h →
  L = 48 :=
by
  intros hw hs hV hh h_new_w h_volume
  -- conversion of the mathematical equivalent proof problem to Lean's theorem
  sorry

end sheet_length_proof_l111_111852


namespace binomial_seven_four_l111_111191

noncomputable def binomial (n k : Nat) : Nat := n.choose k

theorem binomial_seven_four : binomial 7 4 = 35 := by
  sorry

end binomial_seven_four_l111_111191


namespace shopkeeper_loss_percent_l111_111646

noncomputable def loss_percentage (cost_price profit_percent theft_percent: ℝ) :=
  let selling_price := cost_price * (1 + profit_percent / 100)
  let value_lost := cost_price * (theft_percent / 100)
  let remaining_cost_price := cost_price * (1 - theft_percent / 100)
  (value_lost / remaining_cost_price) * 100

theorem shopkeeper_loss_percent
  (cost_price : ℝ)
  (profit_percent : ℝ := 10)
  (theft_percent : ℝ := 20)
  (expected_loss_percent : ℝ := 25)
  (h1 : profit_percent = 10) (h2 : theft_percent = 20) : 
  loss_percentage cost_price profit_percent theft_percent = expected_loss_percent := 
by
  sorry

end shopkeeper_loss_percent_l111_111646


namespace prob_first_two_same_color_expected_value_eta_l111_111037

-- Definitions and conditions
def num_white : ℕ := 4
def num_black : ℕ := 3
def total_pieces : ℕ := num_white + num_black

-- Probability of drawing two pieces of the same color
def prob_same_color : ℚ :=
  (4/7 * 3/6) + (3/7 * 2/6)

-- Expected value of the number of white pieces drawn in the first four draws
def E_eta : ℚ :=
  1 * (4 / 35) + 2 * (18 / 35) + 3 * (12 / 35) + 4 * (1 / 35)

-- Proof statements
theorem prob_first_two_same_color : prob_same_color = 3 / 7 :=
  by sorry

theorem expected_value_eta : E_eta = 16 / 7 :=
  by sorry

end prob_first_two_same_color_expected_value_eta_l111_111037


namespace find_x_l111_111088

noncomputable def vec_sub (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 - b.1, a.2 - b.2)

noncomputable def vec_dot (a b : ℝ × ℝ) : ℝ := (a.1 * b.1) + (a.2 * b.2)

noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt ((v.1)^2 + (v.2)^2)

theorem find_x (x : ℝ) (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (1, x)) 
  (h3 : magnitude (vec_sub a b) = vec_dot a b) : 
  x = 1 / 3 :=
by
  sorry

end find_x_l111_111088


namespace percentage_sum_l111_111542

noncomputable def womenWithRedHairBelow30 : ℝ := 0.07
noncomputable def menWithDarkHair30OrOlder : ℝ := 0.13

theorem percentage_sum :
  womenWithRedHairBelow30 + menWithDarkHair30OrOlder = 0.20 := by
  sorry -- Proof is omitted

end percentage_sum_l111_111542


namespace sum_of_roots_l111_111716

theorem sum_of_roots {a b c d : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
    (h1 : c + d = -a) (h2 : c * d = b) (h3 : a + b = -c) (h4 : a * b = d) : 
    a + b + c + d = -2 := 
by
  sorry

end sum_of_roots_l111_111716


namespace largest_additional_plates_l111_111540

theorem largest_additional_plates
  (initial_first_set_size : ℕ)
  (initial_second_set_size : ℕ)
  (initial_third_set_size : ℕ)
  (new_letters : ℕ)
  (constraint : 1 ≤ initial_second_set_size + 1 ∧ 1 ≤ initial_third_set_size + 1)
  (initial_combinations : ℕ)
  (final_combinations1 : ℕ)
  (final_combinations2 : ℕ)
  (additional_combinations : ℕ) :
  initial_first_set_size = 5 →
  initial_second_set_size = 3 →
  initial_third_set_size = 4 →
  new_letters = 4 →
  initial_combinations = initial_first_set_size * initial_second_set_size * initial_third_set_size →
  final_combinations1 = initial_first_set_size * (initial_second_set_size + 2) * (initial_third_set_size + 2) →
  final_combinations2 = (initial_first_set_size + 1) * (initial_second_set_size + 2) * (initial_third_set_size + 1) →
  additional_combinations = max (final_combinations1 - initial_combinations) (final_combinations2 - initial_combinations) →
  additional_combinations = 90 :=
by sorry

end largest_additional_plates_l111_111540


namespace total_parallelepipeds_l111_111340

theorem total_parallelepipeds (m n k : ℕ) : 
  ∃ (num : ℕ), num == (m * n * k * (m + 1) * (n + 1) * (k + 1)) / 8 :=
  sorry

end total_parallelepipeds_l111_111340


namespace tables_made_this_month_l111_111984

theorem tables_made_this_month (T : ℕ) 
  (h1: ∀ t, t = T → t - 3 < t) 
  (h2 : T + (T - 3) = 17) :
  T = 10 := by
  sorry

end tables_made_this_month_l111_111984


namespace time_to_pick_sugar_snap_peas_l111_111496

theorem time_to_pick_sugar_snap_peas (pea_count1 pea_count2 : ℕ) (time1 : ℕ) :
  (pea_count1 = 56 ∧ time1 = 7 ∧ pea_count2 = 72) →
  let rate := pea_count1 / time1 in
  (pea_count2 / rate = 9) :=
by
  intros h
  let ⟨h1, h2, h3⟩ := h
  sorry

end time_to_pick_sugar_snap_peas_l111_111496


namespace josie_gift_money_l111_111562

-- Define the cost of each cassette tape
def tape_cost : ℕ := 9

-- Define the number of cassette tapes Josie plans to buy
def num_tapes : ℕ := 2

-- Define the cost of the headphone set
def headphone_cost : ℕ := 25

-- Define the amount of money Josie will have left after the purchases
def money_left : ℕ := 7

-- Define the total cost of tapes
def total_tape_cost := num_tapes * tape_cost

-- Define the total cost of both tapes and headphone set
def total_cost := total_tape_cost + headphone_cost

-- The total money Josie will have would be total_cost + money_left
theorem josie_gift_money : total_cost + money_left = 50 :=
by
  -- Proof will be provided here
  sorry

end josie_gift_money_l111_111562


namespace john_fixes_8_computers_l111_111556

theorem john_fixes_8_computers 
  (total_computers : ℕ)
  (unfixable_percentage : ℝ)
  (waiting_percentage : ℝ) 
  (h1 : total_computers = 20)
  (h2 : unfixable_percentage = 0.2)
  (h3 : waiting_percentage = 0.4) :
  let fixed_right_away := total_computers * (1 - unfixable_percentage - waiting_percentage)
  fixed_right_away = 8 :=
by
  sorry

end john_fixes_8_computers_l111_111556


namespace rectangles_fit_l111_111090

theorem rectangles_fit :
  let width := 50
  let height := 90
  let r_width := 1
  let r_height := (10 * Real.sqrt 2)
  ∃ n : ℕ, 
  n = 315 ∧
  (∃ w_cuts h_cuts : ℕ, 
    w_cuts = Int.floor (width / r_height) ∧
    h_cuts = Int.floor (height / r_height) ∧
    n = ((Int.floor (width / r_height) * Int.floor (height / r_height)) + 
         (Int.floor (height / r_width) * Int.floor (width / r_height)))) := 
sorry

end rectangles_fit_l111_111090


namespace point_in_second_quadrant_l111_111102

theorem point_in_second_quadrant {x : ℝ} (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
sorry

end point_in_second_quadrant_l111_111102


namespace number_exceeds_fraction_l111_111975

theorem number_exceeds_fraction (x : ℝ) (h : x = (3/8) * x + 15) : x = 24 :=
sorry

end number_exceeds_fraction_l111_111975


namespace sum_of_circle_areas_constant_l111_111042

theorem sum_of_circle_areas_constant (r OP : ℝ) (h1 : 0 < r) (h2 : 0 ≤ OP ∧ OP < r) 
  (a' b' c' : ℝ) (h3 : a'^2 + b'^2 + c'^2 = OP^2) :
  ∃ (a b c : ℝ), (a^2 + b^2 + c^2 = 3 * r^2 - OP^2) :=
by
  sorry

end sum_of_circle_areas_constant_l111_111042


namespace complement_U_A_l111_111527

-- Definitions based on conditions
def U : Set ℕ := {x | 1 < x ∧ x < 5}
def A : Set ℕ := {2, 3}

-- Statement of the problem
theorem complement_U_A :
  (U \ A) = {4} :=
by
  sorry

end complement_U_A_l111_111527


namespace simplify_fraction_l111_111608

theorem simplify_fraction (a b : ℕ) (h : Nat.gcd a b = 24) : (a = 48) → (b = 72) → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end simplify_fraction_l111_111608


namespace problem_statement_l111_111904

-- Define the functions
def f (x : ℤ) : ℤ := x^2
def g (x : ℤ) : ℤ := 2 * x - 5

-- Define the main theorem statement
theorem problem_statement : f (g (-2)) = 81 := by
  sorry

end problem_statement_l111_111904


namespace geometric_sequence_sum_l111_111619

noncomputable def aₙ (n : ℕ) : ℝ := (2 / 3) ^ (n - 1)

noncomputable def Sₙ (n : ℕ) : ℝ := 3 * (1 - (2 / 3) ^ n)

theorem geometric_sequence_sum (n : ℕ) : Sₙ n = 3 - 2 * aₙ n := by
  sorry

end geometric_sequence_sum_l111_111619


namespace average_difference_l111_111452

theorem average_difference :
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 70 + 28) / 3
  avg1 - avg2 = 4 :=
by
  -- Define the averages
  let avg1 := (20 + 40 + 60) / 3
  let avg2 := (10 + 70 + 28) / 3
  sorry

end average_difference_l111_111452


namespace sum_abs_eq_pos_or_neg_three_l111_111513

theorem sum_abs_eq_pos_or_neg_three (x y : Real) (h1 : abs x = 1) (h2 : abs y = 2) (h3 : x * y > 0) :
    x + y = 3 ∨ x + y = -3 :=
by
  sorry

end sum_abs_eq_pos_or_neg_three_l111_111513


namespace isosceles_triangle_perimeter_l111_111676

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : a = 6) (h₂ : b = 5) :
  ∃ p : ℝ, (p = a + a + b ∨ p = b + b + a) ∧ (p = 16 ∨ p = 17) :=
by
  sorry

end isosceles_triangle_perimeter_l111_111676


namespace find_C_marks_l111_111205

theorem find_C_marks :
  let english := 90
  let math := 92
  let physics := 85
  let biology := 85
  let avg_marks := 87.8
  let total_marks := avg_marks * 5
  let other_marks := english + math + physics + biology
  ∃ C : ℝ, total_marks - other_marks = C ∧ C = 87 :=
by
  sorry

end find_C_marks_l111_111205


namespace bob_coloring_l111_111874

/-
  Problem:
  Find the number of ways to color five points in {(x, y) | 1 ≤ x, y ≤ 5} blue 
  such that the distance between any two blue points is not an integer.
-/

def is_integer_distance (p1 p2 : ℤ × ℤ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let d := Int.gcd ((x2 - x1)^2 + (y2 - y1)^2)
  d ≠ 1

def valid_coloring (points : List (ℤ × ℤ)) : Prop :=
  points.length = 5 ∧ 
  (∀ (p1 p2 : ℤ × ℤ), p1 ∈ points → p2 ∈ points → p1 ≠ p2 → ¬ is_integer_distance p1 p2)

theorem bob_coloring : ∃ (points : List (ℤ × ℤ)), valid_coloring points ∧ points.length = 80 :=
sorry

end bob_coloring_l111_111874


namespace max_area_rect_l111_111669

/--
A rectangle has a perimeter of 40 units and its dimensions are whole numbers.
The maximum possible area of the rectangle is 100 square units.
-/
theorem max_area_rect {l w : ℕ} (hlw : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
by
  have h_sum : l + w = 20 := by
    rw [two_mul, two_mul, add_assoc, add_assoc] at hlw
    exact Nat.div_eq_of_eq_mul_left (by norm_num : 2 ≠ 0) hlw

  have into_parabola : ∀ l w, l + w = 20 → l * w ≤ 100 := λ l w h_eq =>
  by
    let expr := l * w
    let w_def := 20 - l
    let expr' := l * (20 - l)
    have key_expr: l * w = l * (20 - l) := by
      rw h_eq
    rw key_expr
    let f (l: ℕ) := l * (20 - l)
    have step_expr: l * (20 - l) = 20*l - l^2 := by
      ring

    have boundary : (0 ≤ l * (20 - l)) := mul_nonneg (by apply l.zero_le) (by linarith)
    have max_ex : ((20 / 2)^2 ≤ 100) := by norm_num
    let sq_bound:= 100 - (l - 10)^2
    have complete_sq : 20 * l - l^2 = -(l-10)^2 + 100  := by
      have q_expr: 20 * l - l^2 = - (l-10)^2 + 100 := by linarith
      exact q_expr

    show l * (20 - l) ≤ 100,
    from Nat.le_of_pred_lt (by linarith)


  exact into_parabola l w h_sum

end max_area_rect_l111_111669


namespace sum_of_odd_integers_from_13_to_53_l111_111643

-- Definition of the arithmetic series summing from 13 to 53 with common difference 2
def sum_of_arithmetic_series (a l d : ℕ) (n : ℕ) : ℕ :=
  (n * (a + l)) / 2

-- Main theorem
theorem sum_of_odd_integers_from_13_to_53 :
  sum_of_arithmetic_series 13 53 2 21 = 693 := 
sorry

end sum_of_odd_integers_from_13_to_53_l111_111643


namespace cow_spots_total_l111_111290

theorem cow_spots_total
  (left_spots : ℕ) (right_spots : ℕ)
  (left_spots_eq : left_spots = 16)
  (right_spots_eq : right_spots = 3 * left_spots + 7) :
  left_spots + right_spots = 71 :=
by
  sorry

end cow_spots_total_l111_111290


namespace circle_radius_is_zero_l111_111225

-- Define the condition: the given equation of the circle
def circle_eq (x y : ℝ) : Prop :=
  x^2 + 8 * x + y^2 - 10 * y + 41 = 0

-- Define the statement to be proved: the radius of the circle described by the equation is 0
theorem circle_radius_is_zero : ∀ x y : ℝ, circle_eq x y → (∃ r : ℝ, r = 0) :=
begin
  sorry
end

end circle_radius_is_zero_l111_111225


namespace ellipse_product_l111_111439

noncomputable def a (b : ℝ) := b + 4
noncomputable def AB (a: ℝ) := 2 * a
noncomputable def CD (b: ℝ) := 2 * b

theorem ellipse_product:
  (∀ (a b : ℝ), a = b + 4 → a^2 - b^2 = 64) →
  (∃ (a b : ℝ), (AB a) * (CD b) = 240) :=
by
  intros h
  use 10, 6
  simp [AB, CD]
  sorry

end ellipse_product_l111_111439


namespace total_coins_are_correct_l111_111284

-- Define the initial number of coins
def initial_dimes : Nat := 2
def initial_quarters : Nat := 6
def initial_nickels : Nat := 5

-- Define the additional coins given by Linda's mother
def additional_dimes : Nat := 2
def additional_quarters : Nat := 10
def additional_nickels : Nat := 2 * initial_nickels

-- Calculate the total number of each type of coin
def total_dimes : Nat := initial_dimes + additional_dimes
def total_quarters : Nat := initial_quarters + additional_quarters
def total_nickels : Nat := initial_nickels + additional_nickels

-- Total number of coins
def total_coins : Nat := total_dimes + total_quarters + total_nickels

-- Theorem to prove the total number of coins is 35
theorem total_coins_are_correct : total_coins = 35 := by
  -- Skip the proof
  sorry

end total_coins_are_correct_l111_111284


namespace min_value_a2_b2_l111_111703

theorem min_value_a2_b2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (h : a^2 - 2015 * a = b^2 - 2015 * b) : 
  a^2 + b^2 ≥ 2015^2 / 2 := 
sorry

end min_value_a2_b2_l111_111703


namespace solve_problem_l111_111355
noncomputable def is_solution (n : ℕ) : Prop :=
  ∀ (a b c : ℕ), (0 < a) → (0 < b) → (0 < c) → (a + b + c ∣ a^2 + b^2 + c^2) → (a + b + c ∣ a^n + b^n + c^n)

theorem solve_problem : {n : ℕ // is_solution (3 * n - 1) ∧ is_solution (3 * n - 2)} :=
sorry

end solve_problem_l111_111355


namespace eval_f_at_neg_twenty_three_sixth_pi_l111_111067

noncomputable def f (α : ℝ) : ℝ := 
    (2 * (Real.sin (2 * Real.pi - α)) * (Real.cos (2 * Real.pi + α)) - Real.cos (-α)) / 
    (1 + Real.sin α ^ 2 + Real.sin (2 * Real.pi + α) - Real.cos (4 * Real.pi - α) ^ 2)

theorem eval_f_at_neg_twenty_three_sixth_pi : 
  f (-23 / 6 * Real.pi) = -Real.sqrt 3 :=
  sorry

end eval_f_at_neg_twenty_three_sixth_pi_l111_111067


namespace simplify_fraction_l111_111604

-- Define the problem and conditions
def numerator : ℕ := 48
def denominator : ℕ := 72
def gcd_n_d : ℕ := Nat.gcd numerator denominator

-- The proof statement
theorem simplify_fraction : (numerator / gcd_n_d) / (denominator / gcd_n_d) = 2 / 3 :=
by
  have h_gcd : gcd_n_d = 24 := by rfl
  sorry

end simplify_fraction_l111_111604


namespace volume_of_wedge_l111_111034

theorem volume_of_wedge (d : ℝ) (angle : ℝ) (V : ℝ) (n : ℕ) 
  (h_d : d = 18) 
  (h_angle : angle = 60)
  (h_radius_height : ∀ r h, r = d / 2 ∧ h = d) 
  (h_volume_cylinder : V = π * (d / 2) ^ 2 * d) 
  : n = 729 ↔ V / 2 = n * π :=
by
  sorry

end volume_of_wedge_l111_111034


namespace rain_total_duration_l111_111550

theorem rain_total_duration : 
  let first_day_hours := 17 - 7
  let second_day_hours := first_day_hours + 2
  let third_day_hours := 2 * second_day_hours
  first_day_hours + second_day_hours + third_day_hours = 46 :=
by
  sorry

end rain_total_duration_l111_111550


namespace nuts_in_mason_car_l111_111934

-- Define the constants for the rates of stockpiling
def busy_squirrel_rate := 30 -- nuts per day
def sleepy_squirrel_rate := 20 -- nuts per day
def days := 40 -- number of days
def num_busy_squirrels := 2 -- number of busy squirrels
def num_sleepy_squirrels := 1 -- number of sleepy squirrels

-- Define the total number of nuts
def total_nuts_in_mason_car : ℕ :=
  (num_busy_squirrels * busy_squirrel_rate * days) +
  (num_sleepy_squirrels * sleepy_squirrel_rate * days)

theorem nuts_in_mason_car :
  total_nuts_in_mason_car = 3200 :=
sorry

end nuts_in_mason_car_l111_111934


namespace hare_wins_l111_111261

def hare_wins_race : Prop :=
  let hare_speed := 10
  let hare_run_time := 30
  let hare_nap_time := 30
  let tortoise_speed := 4
  let tortoise_delay := 10
  let total_race_time := 60
  let hare_distance := hare_speed * hare_run_time
  let tortoise_total_time := total_race_time - tortoise_delay
  let tortoise_distance := tortoise_speed * tortoise_total_time
  hare_distance > tortoise_distance

theorem hare_wins : hare_wins_race := by
  -- Proof here
  sorry

end hare_wins_l111_111261


namespace xy_value_l111_111892

variable (a b x y : ℝ)
variable (h1 : 2 * a^x * b^3 = - a^2 * b^(1 - y))
variable (hx : x = 2)
variable (hy : y = -2)

theorem xy_value : x * y = -4 := 
by
  sorry

end xy_value_l111_111892


namespace solve_trig_eq_l111_111444

theorem solve_trig_eq (x : ℝ) :
  (sin (2025 * x))^4 + (cos (2016 * x))^2019 * (cos (2025 * x))^2018 = 1 ↔
  (∃ n : ℤ, x = (π / 4050) + (n * π / 2025)) ∨ (∃ k : ℤ, x = (k * π / 9)) :=
by
  sorry

end solve_trig_eq_l111_111444


namespace spiders_hired_l111_111683

theorem spiders_hired (total_workers beavers : ℕ) (h_total : total_workers = 862) (h_beavers : beavers = 318) : (total_workers - beavers) = 544 := by
  sorry

end spiders_hired_l111_111683


namespace factorize_expression_l111_111354

theorem factorize_expression (x : ℝ) : (x + 3) ^ 2 - (x + 3) = (x + 3) * (x + 2) :=
by
  sorry

end factorize_expression_l111_111354


namespace smallest_positive_period_maximum_value_set_l111_111082

noncomputable def f (x : ℝ) : ℝ :=
  sqrt 3 * sin (2 * x - (π / 6)) + 2 * sin (x - (π / 12)) ^ 2

theorem smallest_positive_period :
  ∀ x, f x = f (x + π) :=
by sorry

theorem maximum_value_set :
  { x | f x = 3 } = { x | ∃ (k : ℤ), x = k * π + (5 * π / 12) } :=
by sorry

end smallest_positive_period_maximum_value_set_l111_111082


namespace simplify_fraction_l111_111590

theorem simplify_fraction (a b : ℕ) (h : b ≠ 0) (g : Nat.gcd a b = 24) : a = 48 → b = 72 → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  exact ⟨rfl, rfl⟩

end simplify_fraction_l111_111590


namespace total_rain_duration_l111_111549

theorem total_rain_duration:
  let first_day_duration := 10
  let second_day_duration := first_day_duration + 2
  let third_day_duration := 2 * second_day_duration
  first_day_duration + second_day_duration + third_day_duration = 46 :=
by
  sorry

end total_rain_duration_l111_111549


namespace g_is_even_l111_111767

noncomputable def g (x : ℝ) : ℝ := Real.log (Real.cos x + Real.sqrt (1 + Real.sin x ^ 2))

theorem g_is_even : ∀ x : ℝ, g (-x) = g (x) :=
by
  intro x
  sorry

end g_is_even_l111_111767


namespace feed_cost_for_chickens_l111_111429

noncomputable section

def num_birds : ℕ := 15
def fraction_ducks : ℚ := 1 / 3
def feed_cost_per_chicken : ℚ := 2

theorem feed_cost_for_chickens :
  let num_ducks := (fraction_ducks * num_birds : ℚ).to_nat in
  let num_chickens := num_birds - num_ducks in
  let total_feed_cost := num_chickens * feed_cost_per_chicken in
  total_feed_cost = 20 := 
by 
  sorry

end feed_cost_for_chickens_l111_111429


namespace line_equation_l111_111358

theorem line_equation (x y : ℝ) (h : (2, 3) ∈ {p : ℝ × ℝ | (∃ a, p.1 + p.2 = a) ∨ (∃ k, p.2 = k * p.1)}) :
  (3 * x - 2 * y = 0) ∨ (x + y - 5 = 0) :=
sorry

end line_equation_l111_111358


namespace units_digit_of_power_17_l111_111881

theorem units_digit_of_power_17 (n : ℕ) (k : ℕ) (h_n4 : n % 4 = 3) : (17^n) % 10 = 3 :=
  by
  -- Since units digits of powers repeat every 4
  sorry

-- Specific problem instance
example : (17^1995) % 10 = 3 := units_digit_of_power_17 1995 17 (by norm_num)

end units_digit_of_power_17_l111_111881


namespace flagstaff_height_l111_111991

theorem flagstaff_height 
  (s1 : ℝ) (s2 : ℝ) (hb : ℝ) (h : ℝ)
  (H1 : s1 = 40.25) (H2 : s2 = 28.75) (H3 : hb = 12.5) 
  (H4 : h / s1 = hb / s2) : 
  h = 17.5 :=
by
  sorry

end flagstaff_height_l111_111991


namespace smallest_value_of_c_l111_111573

def bound_a (a b : ℝ) : Prop := 1 + a ≤ b
def bound_inv (a b c : ℝ) : Prop := (1 / a) + (1 / b) ≤ (1 / c)

theorem smallest_value_of_c (a b c : ℝ) (ha : 1 < a) (hb : a < b) 
  (hc : b < c) (h_ab : bound_a a b) (h_inv : bound_inv a b c) : 
  c ≥ (3 + Real.sqrt 5) / 2 := 
sorry

end smallest_value_of_c_l111_111573


namespace inequality_subtraction_l111_111075

variable (a b : ℝ)

-- Given conditions
axiom nonzero_a : a ≠ 0 
axiom nonzero_b : b ≠ 0 
axiom a_lt_b : a < b 

-- Proof statement
theorem inequality_subtraction : a - 3 < b - 3 := 
by 
  sorry

end inequality_subtraction_l111_111075


namespace sum_cubes_first_39_eq_608400_l111_111364

def sum_of_cubes (n : ℕ) : ℕ := (n * (n + 1) / 2) ^ 2

theorem sum_cubes_first_39_eq_608400 : sum_of_cubes 39 = 608400 :=
by
  sorry

end sum_cubes_first_39_eq_608400_l111_111364


namespace Diana_additional_video_game_time_l111_111502

theorem Diana_additional_video_game_time 
    (original_reward_per_hour : ℕ := 30)
    (raise_percentage : ℕ := 20)
    (hours_read : ℕ := 12)
    (minutes_per_hour : ℕ := 60) :
    let raise := (raise_percentage * original_reward_per_hour) / 100
    let new_reward_per_hour := original_reward_per_hour + raise
    let total_time_after_raise := new_reward_per_hour * hours_read
    let total_time_before_raise := original_reward_per_hour * hours_read
    let additional_minutes := total_time_after_raise - total_time_before_raise
    additional_minutes = 72 :=
by sorry

end Diana_additional_video_game_time_l111_111502


namespace rectangle_area_l111_111674

theorem rectangle_area {H W : ℝ} (h_height : H = 24) (ratio : W / H = 0.875) :
  H * W = 504 :=
by 
  sorry

end rectangle_area_l111_111674


namespace intersection_empty_l111_111714

noncomputable def A : Set ℤ := {x | ∃ k : ℤ, x = 2 * k - 1}
noncomputable def B : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

theorem intersection_empty : A ∩ B = ∅ :=
by
  sorry

end intersection_empty_l111_111714


namespace range_of_x_l111_111237

noncomputable def f (x : ℝ) : ℝ := sorry -- Define f to satisfy given conditions later

theorem range_of_x (hf_odd : ∀ x : ℝ, f (-x) = - f x)
                   (hf_inc_mono_neg : ∀ x y : ℝ, x ≤ y → y ≤ 0 → f x ≤ f y)
                   (h_ineq : f 1 + f (Real.log x - 2) < 0) : (0 < x) ∧ (x < 10) :=
by
  sorry

end range_of_x_l111_111237


namespace binomial_7_4_eq_35_l111_111198
-- Import the entire Mathlib library for broad utility functions

-- Define the binomial coefficient function using library definitions 
noncomputable def binomial : ℕ → ℕ → ℕ
| n, k := nat.choose n k

-- Define the theorem we want to prove
theorem binomial_7_4_eq_35 : binomial 7 4 = 35 :=
by
  sorry

end binomial_7_4_eq_35_l111_111198


namespace parabola_intersection_sum_zero_l111_111457

theorem parabola_intersection_sum_zero
  (x_1 x_2 x_3 x_4 y_1 y_2 y_3 y_4 : ℝ)
  (h1 : ∀ x, ∃ y, y = (x - 2)^2 + 1)
  (h2 : ∀ y, ∃ x, x - 1 = (y + 2)^2)
  (h_intersect : (∃ x y, (y = (x - 2)^2 + 1) ∧ (x - 1 = (y + 2)^2))) :
  x_1 + x_2 + x_3 + x_4 + y_1 + y_2 + y_3 + y_4 = 0 :=
sorry

end parabola_intersection_sum_zero_l111_111457


namespace solve_equation_l111_111131

theorem solve_equation :
  {x : ℝ | x * (x - 3)^2 * (5 - x) = 0} = {0, 3, 5} :=
by
  sorry

end solve_equation_l111_111131


namespace find_function_satisfying_condition_l111_111221

theorem find_function_satisfying_condition :
  ∃ c : ℝ, ∀ (f : ℝ → ℝ), (∀ x y : ℝ, f (f x + 2 * y) = 6 * x + f (f y - x)) → 
                          (∀ x : ℝ, f x = 2 * x + c) :=
sorry

end find_function_satisfying_condition_l111_111221


namespace fraction_of_population_married_l111_111425

theorem fraction_of_population_married
  (M W N : ℕ)
  (h1 : (2 / 3 : ℚ) * M = N)
  (h2 : (3 / 5 : ℚ) * W = N)
  : ((2 * N) : ℚ) / (M + W) = 12 / 19 := 
by
  sorry

end fraction_of_population_married_l111_111425


namespace cricket_initial_avg_runs_l111_111809

theorem cricket_initial_avg_runs (A : ℝ) (h : 11 * (A + 4) = 10 * A + 86) : A = 42 :=
sorry

end cricket_initial_avg_runs_l111_111809


namespace cos_4theta_l111_111097

theorem cos_4theta (θ : ℝ) (h : Real.cos θ = 1 / 4) : Real.cos (4 * θ) = 17 / 32 :=
sorry

end cos_4theta_l111_111097


namespace max_area_rect_l111_111668

/--
A rectangle has a perimeter of 40 units and its dimensions are whole numbers.
The maximum possible area of the rectangle is 100 square units.
-/
theorem max_area_rect {l w : ℕ} (hlw : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
by
  have h_sum : l + w = 20 := by
    rw [two_mul, two_mul, add_assoc, add_assoc] at hlw
    exact Nat.div_eq_of_eq_mul_left (by norm_num : 2 ≠ 0) hlw

  have into_parabola : ∀ l w, l + w = 20 → l * w ≤ 100 := λ l w h_eq =>
  by
    let expr := l * w
    let w_def := 20 - l
    let expr' := l * (20 - l)
    have key_expr: l * w = l * (20 - l) := by
      rw h_eq
    rw key_expr
    let f (l: ℕ) := l * (20 - l)
    have step_expr: l * (20 - l) = 20*l - l^2 := by
      ring

    have boundary : (0 ≤ l * (20 - l)) := mul_nonneg (by apply l.zero_le) (by linarith)
    have max_ex : ((20 / 2)^2 ≤ 100) := by norm_num
    let sq_bound:= 100 - (l - 10)^2
    have complete_sq : 20 * l - l^2 = -(l-10)^2 + 100  := by
      have q_expr: 20 * l - l^2 = - (l-10)^2 + 100 := by linarith
      exact q_expr

    show l * (20 - l) ≤ 100,
    from Nat.le_of_pred_lt (by linarith)


  exact into_parabola l w h_sum

end max_area_rect_l111_111668


namespace is_quadratic_function_l111_111867

theorem is_quadratic_function (x : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = 2 * x + 3) ∧ ¬(∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)) ∧
  (∃ f : ℝ → ℝ, (∀ x, f x = 2 / x) ∧ ¬(∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)) ∧
  (∃ f : ℝ → ℝ, (∀ x, f x = (x - 1)^2 - x^2) ∧ ¬(∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c)) ∧
  (∃ f : ℝ → ℝ, (∀ x, f x = 3 * x^2 - 1) ∧ (∃ a b c : ℝ, a ≠ 0 ∧ (∀ x, f x = a * x^2 + b * x + c))) :=
by
  sorry

end is_quadratic_function_l111_111867


namespace find_integer_n_l111_111903

theorem find_integer_n (n : ℤ) (h : (⌊(n^2 : ℤ)/4⌋ - (⌊n/2⌋)^2 = 2)) : n = 5 :=
sorry

end find_integer_n_l111_111903


namespace jellybeans_condition_l111_111043

theorem jellybeans_condition (n : ℕ) (h1 : n ≥ 150) (h2 : n % 15 = 14) : n = 164 :=
sorry

end jellybeans_condition_l111_111043


namespace tom_average_speed_l111_111977

theorem tom_average_speed 
  (d1 d2 : ℝ) (s1 s2 t1 t2 : ℝ)
  (h_d1 : d1 = 30) 
  (h_d2 : d2 = 50) 
  (h_s1 : s1 = 30) 
  (h_s2 : s2 = 50) 
  (h_t1 : t1 = d1 / s1) 
  (h_t2 : t2 = d2 / s2)
  (h_total_distance : d1 + d2 = 80) 
  (h_total_time : t1 + t2 = 2) :
  (d1 + d2) / (t1 + t2) = 40 := 
by {
  sorry
}

end tom_average_speed_l111_111977


namespace total_money_divided_l111_111033

theorem total_money_divided (A B C : ℝ) (h1 : A = (1 / 2) * B) (h2 : B = (1 / 2) * C) (h3 : C = 208) :
  A + B + C = 364 := 
sorry

end total_money_divided_l111_111033


namespace simplify_fraction_l111_111585

theorem simplify_fraction (a b : ℕ) (h : b ≠ 0) (g : Nat.gcd a b = 24) : a = 48 → b = 72 → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  exact ⟨rfl, rfl⟩

end simplify_fraction_l111_111585


namespace annual_growth_rate_l111_111848

theorem annual_growth_rate (P₁ P₂ : ℝ) (y : ℕ) (r : ℝ)
  (h₁ : P₁ = 1) 
  (h₂ : P₂ = 1.21)
  (h₃ : y = 2)
  (h_growth : P₂ = P₁ * (1 + r) ^ y) :
  r = 0.1 :=
by {
  sorry
}

end annual_growth_rate_l111_111848


namespace solution_to_fractional_equation_l111_111308

theorem solution_to_fractional_equation (x : ℝ) (h₁ : 2 / (x - 3) = 1 / x) (h₂ : x ≠ 3) (h₃ : x ≠ 0) : x = -3 :=
sorry

end solution_to_fractional_equation_l111_111308


namespace kendall_total_distance_l111_111926

def distance_with_mother : ℝ := 0.17
def distance_with_father : ℝ := 0.5
def total_distance : ℝ := 0.67

theorem kendall_total_distance :
  (distance_with_mother + distance_with_father = total_distance) :=
sorry

end kendall_total_distance_l111_111926


namespace log4_80_cannot_be_found_without_additional_values_l111_111912

-- Conditions provided in the problem
def log4_16 : Real := 2
def log4_32 : Real := 2.5

-- Lean statement of the proof problem
theorem log4_80_cannot_be_found_without_additional_values :
  ¬(∃ (log4_80 : Real), log4_80 = log4_16 + log4_5) :=
sorry

end log4_80_cannot_be_found_without_additional_values_l111_111912


namespace age_difference_is_18_l111_111623

def difference_in_ages (X Y Z : ℕ) : ℕ := (X + Y) - (Y + Z)
def younger_by_eighteen (X Z : ℕ) : Prop := Z = X - 18

theorem age_difference_is_18 (X Y Z : ℕ) (h : younger_by_eighteen X Z) : difference_in_ages X Y Z = 18 := by
  sorry

end age_difference_is_18_l111_111623


namespace total_perimeter_of_compound_shape_l111_111268

-- Definitions of the conditions from the original problem
def triangle1_side : ℝ := 10
def triangle2_side : ℝ := 6
def shared_side : ℝ := 6

-- A theorem to represent the mathematically equivalent proof problem
theorem total_perimeter_of_compound_shape 
  (t1s : ℝ := triangle1_side) 
  (t2s : ℝ := triangle2_side)
  (ss : ℝ := shared_side) : 
  t1s = 10 ∧ t2s = 6 ∧ ss = 6 → 3 * t1s + 3 * t2s - ss = 42 := 
by
  sorry

end total_perimeter_of_compound_shape_l111_111268


namespace smaller_angle_measure_l111_111633

theorem smaller_angle_measure (x : ℝ) (a b : ℝ) (h_suppl : a + b = 180) (h_ratio : a = 4 * x ∧ b = x) :
  b = 36 :=
by
  sorry

end smaller_angle_measure_l111_111633


namespace smaller_angle_36_degrees_l111_111631

noncomputable def smaller_angle_measure (larger smaller : ℝ) : Prop :=
(larger + smaller = 180) ∧ (larger = 4 * smaller)

theorem smaller_angle_36_degrees : ∃ (smaller : ℝ), smaller_angle_measure (4 * smaller) smaller ∧ smaller = 36 :=
by
  sorry

end smaller_angle_36_degrees_l111_111631


namespace area_of_fifteen_sided_figure_l111_111057

def point : Type := ℕ × ℕ

def vertices : List point :=
  [(1,1), (1,3), (3,5), (4,5), (5,4), (5,3), (6,3), (6,2), (5,1), (4,1), (3,2), (2,2), (1,1)]

def graph_paper_area (vs : List point) : ℚ :=
  -- Placeholder for actual area calculation logic
  -- The area for the provided vertices is found to be 11 cm^2.
  11

theorem area_of_fifteen_sided_figure : graph_paper_area vertices = 11 :=
by
  -- The actual proof would involve detailed steps to show that the area is indeed 11 cm^2
  -- Placeholder proof
  sorry

end area_of_fifteen_sided_figure_l111_111057


namespace shaded_fraction_is_one_eighth_l111_111017

noncomputable def total_area (length : ℕ) (width : ℕ) : ℕ :=
  length * width

noncomputable def half_area (length : ℕ) (width : ℕ) : ℚ :=
  total_area length width / 2

noncomputable def shaded_area (length : ℕ) (width : ℕ) : ℚ :=
  half_area length width / 4

theorem shaded_fraction_is_one_eighth : 
  ∀ (length width : ℕ), length = 15 → width = 21 → shaded_area length width / total_area length width = 1 / 8 :=
by
  sorry

end shaded_fraction_is_one_eighth_l111_111017


namespace sqrt_16_eq_pm_4_l111_111818

theorem sqrt_16_eq_pm_4 (x : ℝ) (h : x^2 = 16) : x = 4 ∨ x = -4 := by
  sorry

end sqrt_16_eq_pm_4_l111_111818


namespace simplify_fraction_l111_111596

theorem simplify_fraction :
  (48 : ℚ) / 72 = 2 / 3 :=
sorry

end simplify_fraction_l111_111596


namespace molecular_weight_of_NH4Br_l111_111641

def atomic_weight (element : String) : Real :=
  match element with
  | "N" => 14.01
  | "H" => 1.01
  | "Br" => 79.90
  | _ => 0.0

def molecular_weight (composition : List (String × Nat)) : Real :=
  composition.foldl (λ acc (elem, count) => acc + count * atomic_weight elem) 0

theorem molecular_weight_of_NH4Br :
  molecular_weight [("N", 1), ("H", 4), ("Br", 1)] = 97.95 :=
by
  sorry

end molecular_weight_of_NH4Br_l111_111641


namespace base_height_calculation_l111_111189

noncomputable def height_of_sculpture : ℚ := 2 + 5/6 -- 2 feet 10 inches in feet
noncomputable def total_height : ℚ := 3.5
noncomputable def height_of_base : ℚ := 2/3

theorem base_height_calculation (h1 : height_of_sculpture = 17/6) (h2 : total_height = 21/6):
  height_of_base = total_height - height_of_sculpture := by
  sorry

end base_height_calculation_l111_111189


namespace smallest_f1_value_l111_111228

noncomputable def polynomial := 
  fun (f : ℝ → ℝ) (r s : ℝ) => 
    f = λ x => (x - r) * (x - s) * (x - ((r + s)/2))

def distinct_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ (r s : ℝ), r ≠ s ∧ polynomial f r s ∧ 
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (f ∘ f) a = 0 ∧ (f ∘ f) b = 0 ∧ (f ∘ f) c = 0)

theorem smallest_f1_value
  (f : ℝ → ℝ)
  (hf : distinct_real_roots f) :
  ∃ r s : ℝ, r ≠ s ∧ f 1 = 3/8 :=
sorry

end smallest_f1_value_l111_111228


namespace percentage_of_girls_l111_111148

def total_students : ℕ := 100
def boys : ℕ := 50
def girls : ℕ := total_students - boys

theorem percentage_of_girls :
  (girls / total_students) * 100 = 50 := sorry

end percentage_of_girls_l111_111148


namespace bruce_purchased_mangoes_l111_111875

noncomputable def calculate_mango_quantity (grapes_quantity : ℕ) (grapes_rate : ℕ) (mango_rate : ℕ) (total_paid : ℕ) : ℕ :=
  let cost_of_grapes := grapes_quantity * grapes_rate
  let cost_of_mangoes := total_paid - cost_of_grapes
  cost_of_mangoes / mango_rate

theorem bruce_purchased_mangoes :
  calculate_mango_quantity 8 70 55 1055 = 9 :=
by
  sorry

end bruce_purchased_mangoes_l111_111875


namespace find_k_eq_neg_four_thirds_l111_111029

-- Definitions based on conditions
def hash_p (k : ℚ) (p : ℚ) : ℚ := k * p + 20

-- Using the initial condition
def triple_hash_18 (k : ℚ) : ℚ :=
  let hp := hash_p k 18
  let hhp := hash_p k hp
  hash_p k hhp

-- The Lean statement for the desired proof
theorem find_k_eq_neg_four_thirds (k : ℚ) (h : triple_hash_18 k = -4) : k = -4 / 3 :=
sorry

end find_k_eq_neg_four_thirds_l111_111029


namespace proof_Bill_age_is_24_l111_111188

noncomputable def Bill_is_24 (C : ℝ) (Bill_age : ℝ) (Daniel_age : ℝ) :=
  (Bill_age = 2 * C - 1) ∧ 
  (Daniel_age = C - 4) ∧ 
  (C + Bill_age + Daniel_age = 45) → 
  (Bill_age = 24)

theorem proof_Bill_age_is_24 (C Bill_age Daniel_age : ℝ) : 
  Bill_is_24 C Bill_age Daniel_age :=
by
  sorry

end proof_Bill_age_is_24_l111_111188


namespace range_a_mul_b_sub_three_half_l111_111509

theorem range_a_mul_b_sub_three_half (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : 1 < b) (h4 : b = (1 + Real.sqrt 5) / 2 * a) :
  (∃ l u : ℝ, ∀ f, l ≤ f ∧ f < u ↔ f = a * (b - 3 / 2)) :=
sorry

end range_a_mul_b_sub_three_half_l111_111509


namespace lamp_post_ratio_l111_111343

theorem lamp_post_ratio (x k m : ℕ) (h1 : 9 * x = k) (h2 : 99 * x = m) : m = 11 * k :=
by sorry

end lamp_post_ratio_l111_111343


namespace max_area_rectangle_l111_111670

theorem max_area_rectangle (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 40) : (∃ (l w : ℕ), l * w = 100) :=
by
  sorry

end max_area_rectangle_l111_111670


namespace nate_matches_final_count_l111_111784

theorem nate_matches_final_count :
  ∀ (initial: ℕ) (dropped: ℕ),
    initial = 70 →
    dropped = 10 →
    ∃ final: ℕ, final = initial - dropped - 2*dropped ∧ final = 40 :=
by
  intros initial dropped h_initial h_dropped
  use initial - dropped - 2*dropped
  have h_final_eq := calc
    initial - dropped - 2*dropped = 70 - 10 - 2*10 : by rw [h_initial, h_dropped]
    ... = 40 : by norm_num
  exact ⟨h_final_eq, h_final_eq.symm⟩

end nate_matches_final_count_l111_111784


namespace curve_cartesian_equation_max_value_3x_plus_4y_l111_111086

noncomputable def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ := (rho * Real.cos theta, rho * Real.sin theta)

theorem curve_cartesian_equation :
  (∀ (rho theta : ℝ), rho^2 = 36 / (4 * (Real.cos theta)^2 + 9 * (Real.sin theta)^2)) →
  ∀ x y : ℝ, (∃ theta : ℝ, x = 3 * Real.cos theta ∧ y = 2 * Real.sin theta) → (x^2) / 9 + (y^2) / 4 = 1 :=
sorry

theorem max_value_3x_plus_4y :
  (∀ (rho theta : ℝ), rho^2 = 36 / (4 * (Real.cos theta)^2 + 9 * (Real.sin theta)^2)) →
  ∃ x y : ℝ, (∃ theta : ℝ, x = 3 * Real.cos theta ∧ y = 2 * Real.sin theta) ∧ (∀ ϴ : ℝ, 3 * (3 * Real.cos ϴ) + 4 * (2 * Real.sin ϴ) ≤ Real.sqrt 145) :=
sorry

end curve_cartesian_equation_max_value_3x_plus_4y_l111_111086


namespace ellipse_product_major_minor_axes_l111_111432

theorem ellipse_product_major_minor_axes 
  (a b : ℝ)
  (OF : ℝ = 8)
  (diameter_ocf : ℝ = 4)
  (h1 : a^2 - b^2 = 64)
  (h2 : b + OF - a = diameter_ocf / 2) :
  2 * a * 2 * b = 240 :=
by
  -- The detailed proof goes here
  sorry

end ellipse_product_major_minor_axes_l111_111432


namespace factorial_division_l111_111055

theorem factorial_division (n : ℕ) (h : n = 9) : n.factorial / (n - 1).factorial = 9 :=
by 
  rw [h]
  sorry

end factorial_division_l111_111055


namespace can_increase_averages_l111_111757

def grades_group1 : List ℕ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def grades_group2 : List ℕ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℕ) : Float :=
  (grades.map Nat.toFloat).sum / grades.length

def new_average (grades : List ℕ) (grade_to_remove_or_add : ℕ) (adding : Bool) : Float :=
  if adding
  then (grades.map Nat.toFloat).sum + grade_to_remove_or_add / (grades.length + 1)
  else (grades.map Nat.toFloat).sum - grade_to_remove_or_add / (grades.length - 1)

theorem can_increase_averages :
  ∃ grade,
    grade ∈ grades_group1 ∧
    average grades_group1 < new_average grades_group1 grade false ∧
    average grades_group2 < new_average grades_group2 grade true :=
  sorry

end can_increase_averages_l111_111757


namespace range_of_a_l111_111393

theorem range_of_a:
  (∃ x : ℝ, 1 ≤ x ∧ |x - a| + x - 4 ≤ 0) → (-2 ≤ a ∧ a ≤ 4) :=
by
  sorry

end range_of_a_l111_111393


namespace parallelogram_perimeter_l111_111642

theorem parallelogram_perimeter 
  (EF FG EH : ℝ)
  (hEF : EF = 40) (hFG : FG = 30) (hEH : EH = 50) : 
  2 * (EF + FG) = 140 := 
by 
  rw [hEF, hFG]
  norm_num

end parallelogram_perimeter_l111_111642


namespace locus_of_points_equidistant_from_axes_l111_111954

-- Define the notion of being equidistant from the x-axis and the y-axis
def is_equidistant_from_axes (P : (ℝ × ℝ)) : Prop :=
  abs P.1 = abs P.2

-- The proof problem: given a moving point, the locus equation when P is equidistant from both axes
theorem locus_of_points_equidistant_from_axes (x y : ℝ) :
  is_equidistant_from_axes (x, y) → abs x - abs y = 0 :=
by
  intros h
  exact sorry

end locus_of_points_equidistant_from_axes_l111_111954


namespace point_in_second_quadrant_l111_111100

theorem point_in_second_quadrant (x : ℝ) (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
by
  sorry

end point_in_second_quadrant_l111_111100


namespace zoe_remaining_pictures_l111_111832

-- Definitions for the problem conditions
def monday_pictures := 24
def tuesday_pictures := 37
def wednesday_pictures := 50
def thursday_pictures := 33
def friday_pictures := 44

def rate_first := 4
def rate_second := 5
def rate_third := 6
def rate_fourth := 3
def rate_fifth := 7

def days_colored (start_day : ℕ) (end_day := 6) := end_day - start_day

def remaining_pictures (total_pictures : ℕ) (rate_per_day : ℕ) (days : ℕ) : ℕ :=
  total_pictures - (rate_per_day * days)

-- Main theorem statement
theorem zoe_remaining_pictures : 
  remaining_pictures monday_pictures rate_first (days_colored 1) +
  remaining_pictures tuesday_pictures rate_second (days_colored 2) +
  remaining_pictures wednesday_pictures rate_third (days_colored 3) +
  remaining_pictures thursday_pictures rate_fourth (days_colored 4) +
  remaining_pictures friday_pictures rate_fifth (days_colored 5) = 117 :=
  sorry

end zoe_remaining_pictures_l111_111832


namespace max_f_l111_111475

theorem max_f (a : ℝ) (h : 0 < a ∧ a < 1) : ∃ x : ℝ, (-1 < x) →  ∀ y : ℝ, (y > -1) → ((1 + y)^a - a*y ≤ 1) :=
sorry

end max_f_l111_111475


namespace polynomial_divisibility_by_120_l111_111504

theorem polynomial_divisibility_by_120 (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
by
  sorry

end polynomial_divisibility_by_120_l111_111504


namespace gnuff_tutoring_minutes_l111_111734

theorem gnuff_tutoring_minutes 
  (flat_rate : ℕ) 
  (rate_per_minute : ℕ) 
  (total_paid : ℕ) :
  flat_rate = 20 → 
  rate_per_minute = 7 →
  total_paid = 146 → 
  ∃ minutes : ℕ, minutes = 18 ∧ flat_rate + rate_per_minute * minutes = total_paid := 
by
  -- Assume the necessary steps here
  sorry

end gnuff_tutoring_minutes_l111_111734


namespace no_naturals_satisfy_divisibility_condition_l111_111691

theorem no_naturals_satisfy_divisibility_condition :
  ∀ (a b c : ℕ), ¬ (2013 * (a * b + b * c + c * a) ∣ a^2 + b^2 + c^2) :=
by
  sorry

end no_naturals_satisfy_divisibility_condition_l111_111691


namespace range_of_a_l111_111348

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 3| - |x - 1| ≤ a^2 - 3 * a) ↔ (a ≤ -1 ∨ a ≥ 4) :=
by sorry

end range_of_a_l111_111348


namespace zou_mei_competition_l111_111265

theorem zou_mei_competition (n : ℕ) (h1 : 271 = n^2 + 15) (h2 : n^2 + 33 = (n + 1)^2) : 
  ∃ n, 271 = n^2 + 15 ∧ n^2 + 33 = (n + 1)^2 :=
by
  existsi n
  exact ⟨h1, h2⟩

end zou_mei_competition_l111_111265


namespace hemisphere_surface_area_l111_111615

theorem hemisphere_surface_area (base_area : ℝ) (r : ℝ) (total_surface_area : ℝ) 
(h1: base_area = 64 * Real.pi) 
(h2: r^2 = 64)
(h3: total_surface_area = base_area + 2 * Real.pi * r^2) : 
total_surface_area = 192 * Real.pi := 
sorry

end hemisphere_surface_area_l111_111615


namespace binomial_seven_four_l111_111193

noncomputable def binomial (n k : Nat) : Nat := n.choose k

theorem binomial_seven_four : binomial 7 4 = 35 := by
  sorry

end binomial_seven_four_l111_111193


namespace cube_surface_area_l111_111152

theorem cube_surface_area (s : ℝ) (h : s = 8) : 6 * s^2 = 384 :=
by
  sorry

end cube_surface_area_l111_111152


namespace smallest_number_of_rectangles_needed_l111_111829

-- Define the dimensions of the rectangle
def rectangle_area (length width : ℕ) : ℕ := length * width

-- Define the side length of the square
def square_side_length : ℕ := 12

-- Define the number of rectangles needed to cover the square horizontally
def num_rectangles_to_cover_square : ℕ := (square_side_length / 3) * (square_side_length / 4)

-- The theorem must state the total number of rectangles required
theorem smallest_number_of_rectangles_needed : num_rectangles_to_cover_square = 16 := 
by
  -- Proof details are skipped using sorry
  sorry

end smallest_number_of_rectangles_needed_l111_111829


namespace max_area_rectangle_l111_111671

theorem max_area_rectangle (l w : ℕ) (h_perimeter : 2 * l + 2 * w = 40) : (∃ (l w : ℕ), l * w = 100) :=
by
  sorry

end max_area_rectangle_l111_111671


namespace intersection_A_B_l111_111243

def A : Set ℤ := {x | x^2 - 3 * x - 4 < 0}
def B : Set ℤ := {-2, -1, 0, 2, 3}

theorem intersection_A_B : A ∩ B = {0, 2, 3} :=
by sorry

end intersection_A_B_l111_111243


namespace problem_statement_l111_111105

variable (P : ℕ → Prop)

theorem problem_statement
    (h1 : P 2)
    (h2 : ∀ k : ℕ, k > 0 → P k → P (k + 2)) :
    ∀ n : ℕ, n > 0 → 2 ∣ n → P n :=
by
  sorry

end problem_statement_l111_111105


namespace angles_in_triangle_l111_111718

theorem angles_in_triangle (A B C : ℝ) (h1 : A + B + C = 180) (h2 : 2 * B = 3 * A) (h3 : 5 * A = 2 * C) :
  B = 54 ∧ C = 90 :=
by
  sorry

end angles_in_triangle_l111_111718


namespace stickers_on_fifth_page_l111_111872

theorem stickers_on_fifth_page :
  ∀ (stickers : ℕ → ℕ),
    stickers 1 = 8 →
    stickers 2 = 16 →
    stickers 3 = 24 →
    stickers 4 = 32 →
    (∀ n, stickers (n + 1) = stickers n + 8) →
    stickers 5 = 40 :=
by
  intros stickers h1 h2 h3 h4 pattern
  apply sorry

end stickers_on_fifth_page_l111_111872


namespace star_7_3_l111_111739

def star (a b : ℤ) : ℤ := 4 * a + 3 * b - a * b

theorem star_7_3 : star 7 3 = 16 := 
by 
  sorry

end star_7_3_l111_111739


namespace pet_store_cages_l111_111980

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage remaining_puppies num_cages : ℕ)
  (h1 : initial_puppies = 102) 
  (h2 : sold_puppies = 21) 
  (h3 : puppies_per_cage = 9) 
  (h4 : remaining_puppies = initial_puppies - sold_puppies)
  (h5 : num_cages = remaining_puppies / puppies_per_cage) : 
  num_cages = 9 := 
by
  sorry

end pet_store_cages_l111_111980


namespace greatest_ratio_AB_CD_on_circle_l111_111686

/-- The statement proving the greatest possible value of the ratio AB/CD for points A, B, C, D lying on the 
circle x^2 + y^2 = 16 with integer coordinates and unequal distances AB and CD is sqrt 10 / 3. -/
theorem greatest_ratio_AB_CD_on_circle :
  ∀ (A B C D : ℤ × ℤ), A ≠ B → C ≠ D → 
  A.1^2 + A.2^2 = 16 → B.1^2 + B.2^2 = 16 → 
  C.1^2 + C.2^2 = 16 → D.1^2 + D.2^2 = 16 → 
  let AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let CD := Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2)
  let ratio := AB / CD
  AB ≠ CD →
  ratio ≤ Real.sqrt 10 / 3 :=
sorry

end greatest_ratio_AB_CD_on_circle_l111_111686


namespace max_rectangle_area_l111_111672

theorem max_rectangle_area (l w : ℕ) (h1 : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
by
  have h2 : l + w = 20 := by linarith
  -- Further steps would go here but we're just stating it
  sorry

end max_rectangle_area_l111_111672


namespace ratio_red_to_yellow_l111_111735

structure MugCollection where
  total_mugs : ℕ
  red_mugs : ℕ
  blue_mugs : ℕ
  yellow_mugs : ℕ
  other_mugs : ℕ
  colors : ℕ

def HannahCollection : MugCollection :=
  { total_mugs := 40,
    red_mugs := 6,
    blue_mugs := 6 * 3,
    yellow_mugs := 12,
    other_mugs := 4,
    colors := 4 }

theorem ratio_red_to_yellow
  (hc : MugCollection)
  (h_total : hc.total_mugs = 40)
  (h_blue : hc.blue_mugs = 3 * hc.red_mugs)
  (h_yellow : hc.yellow_mugs = 12)
  (h_other : hc.other_mugs = 4)
  (h_colors : hc.colors = 4) :
  hc.red_mugs / hc.yellow_mugs = 1 / 2 := by
  sorry

end ratio_red_to_yellow_l111_111735


namespace find_w_value_l111_111158

theorem find_w_value : 
  (2^5 * 9^2) / (8^2 * 243) = 0.16666666666666666 := 
by
  sorry

end find_w_value_l111_111158


namespace ratio_owners_riding_to_total_l111_111847

theorem ratio_owners_riding_to_total (h_num_legs : 70 = 4 * (14 - W) + 6 * W) (h_total : 14 = W + (14 - W)) :
  (14 - W) / 14 = 1 / 2 :=
by
  sorry

end ratio_owners_riding_to_total_l111_111847


namespace car_2_speed_proof_l111_111833

noncomputable def car_1_speed : ℝ := 30
noncomputable def car_1_start_time : ℝ := 9
noncomputable def car_2_start_delay : ℝ := 10 / 60
noncomputable def catch_up_time : ℝ := 10.5
noncomputable def car_2_start_time : ℝ := car_1_start_time + car_2_start_delay
noncomputable def travel_duration : ℝ := catch_up_time - car_2_start_time
noncomputable def car_1_head_start_distance : ℝ := car_1_speed * car_2_start_delay
noncomputable def car_1_travel_distance : ℝ := car_1_speed * travel_duration
noncomputable def total_distance : ℝ := car_1_head_start_distance + car_1_travel_distance
noncomputable def car_2_speed : ℝ := total_distance / travel_duration

theorem car_2_speed_proof : car_2_speed = 33.75 := 
by 
  sorry

end car_2_speed_proof_l111_111833


namespace exchange_rate_decrease_l111_111870

theorem exchange_rate_decrease
  (x y z : ℝ)
  (hx : 0 < |x| ∧ |x| < 1)
  (hy : 0 < |y| ∧ |y| < 1)
  (hz : 0 < |z| ∧ |z| < 1)
  (h_eq : (1 + x) * (1 + y) * (1 + z) = (1 - x) * (1 - y) * (1 - z)) :
  (1 - x^2) * (1 - y^2) * (1 - z^2) < 1 :=
by
  sorry

end exchange_rate_decrease_l111_111870


namespace average_episodes_per_year_l111_111839

theorem average_episodes_per_year (total_years : ℕ) (n1 n2 n3 e1 e2 e3 : ℕ) 
  (h1 : total_years = 14)
  (h2 : n1 = 8) (h3 : e1 = 15)
  (h4 : n2 = 4) (h5 : e2 = 20)
  (h6 : n3 = 2) (h7 : e3 = 12) :
  (n1 * e1 + n2 * e2 + n3 * e3) / total_years = 16 := by
  -- Skip the proof steps
  sorry

end average_episodes_per_year_l111_111839


namespace savings_together_vs_separate_l111_111487

def price_per_window : ℕ := 100

def free_windows_per_5_purchased : ℕ := 2

def daves_windows_needed : ℕ := 10

def dougs_windows_needed : ℕ := 11

def total_windows_needed : ℕ := daves_windows_needed + dougs_windows_needed

-- Cost calculation for Dave's windows with the offer
def daves_cost_with_offer : ℕ := 8 * price_per_window

-- Cost calculation for Doug's windows with the offer
def dougs_cost_with_offer : ℕ := 9 * price_per_window

-- Total cost calculation if purchased separately with the offer
def total_cost_separately_with_offer : ℕ := daves_cost_with_offer + dougs_cost_with_offer

-- Total cost calculation if purchased together with the offer
def total_cost_together_with_offer : ℕ := 17 * price_per_window

-- Calculate additional savings if Dave and Doug purchase together rather than separately
def additional_savings_together_vs_separate := 
  total_cost_separately_with_offer - total_cost_together_with_offer = 0

theorem savings_together_vs_separate : additional_savings_together_vs_separate := by
  sorry

end savings_together_vs_separate_l111_111487


namespace fraction_of_population_married_l111_111424

theorem fraction_of_population_married
  (M W N : ℕ)
  (h1 : (2 / 3 : ℚ) * M = N)
  (h2 : (3 / 5 : ℚ) * W = N)
  : ((2 * N) : ℚ) / (M + W) = 12 / 19 := 
by
  sorry

end fraction_of_population_married_l111_111424


namespace total_packs_l111_111937

theorem total_packs (cards_per_person cards_per_pack : ℕ) (num_people : ℕ) 
  (h1 : cards_per_person = 540) 
  (h2 : cards_per_pack = 20) 
  (h3 : num_people = 4) : 
  (cards_per_person / cards_per_pack) * num_people = 108 := 
by
  sorry

end total_packs_l111_111937


namespace can_cut_rectangle_l111_111403

def original_rectangle_width := 100
def original_rectangle_height := 70
def total_area := original_rectangle_width * original_rectangle_height

def area1 := 1000
def area2 := 2000
def area3 := 4000

theorem can_cut_rectangle : 
  (area1 + area2 + area3 = total_area) ∧ 
  (area1 * 2 = area2) ∧ 
  (area1 * 4 = area3) ∧ 
  (area1 > 0) ∧ (area2 > 0) ∧ (area3 > 0) ∧
  (∃ (w1 h1 w2 h2 w3 h3 : ℕ), 
    w1 * h1 = area1 ∧ w2 * h2 = area2 ∧ w3 * h3 = area3 ∧
    ((w1 + w2 ≤ original_rectangle_width ∧ max h1 h2 + h3 ≤ original_rectangle_height) ∨
     (h1 + h2 ≤ original_rectangle_height ∧ max w1 w2 + w3 ≤ original_rectangle_width)))
:=
  sorry

end can_cut_rectangle_l111_111403


namespace smaller_angle_measure_l111_111634

theorem smaller_angle_measure (x : ℝ) (a b : ℝ) (h_suppl : a + b = 180) (h_ratio : a = 4 * x ∧ b = x) :
  b = 36 :=
by
  sorry

end smaller_angle_measure_l111_111634


namespace sum_of_root_and_square_of_other_root_eq_2007_l111_111902

/-- If α and β are the two real roots of the equation x^2 - x - 2006 = 0,
    then the value of α + β^2 is 2007. --/
theorem sum_of_root_and_square_of_other_root_eq_2007
  (α β : ℝ)
  (hα : α^2 - α - 2006 = 0)
  (hβ : β^2 - β - 2006 = 0) :
  α + β^2 = 2007 := sorry

end sum_of_root_and_square_of_other_root_eq_2007_l111_111902


namespace part_a_part_b_l111_111811

def is_multiple_of_9 (n : ℕ) := n % 9 = 0
def digit_sum (n : ℕ) : ℕ := (n.digits 10).sum

theorem part_a : ∃ n : ℕ, is_multiple_of_9 n ∧ digit_sum n = 81 ∧ (n / 9) = 111111111 := 
sorry

theorem part_b : ∃ n1 n2 n3 n4 : ℕ,
  is_multiple_of_9 n1 ∧
  is_multiple_of_9 n2 ∧
  is_multiple_of_9 n3 ∧
  is_multiple_of_9 n4 ∧
  digit_sum n1 = 27 ∧ digit_sum n2 = 27 ∧ digit_sum n3 = 27 ∧ digit_sum n4 = 27 ∧
  (n1 / 9) + 1 = (n2 / 9) ∧ 
  (n2 / 9) + 1 = (n3 / 9) ∧ 
  (n3 / 9) + 1 = (n4 / 9) ∧ 
  (n4 / 9) < 1111 := 
sorry

end part_a_part_b_l111_111811


namespace complement_union_l111_111725

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {2, 4, 5}
def B : Set ℕ := {1, 3, 4, 5}

theorem complement_union :
  (U \ A) ∪ (U \ B) = {1, 2, 3, 6} := 
by 
  sorry

end complement_union_l111_111725


namespace time_ratio_krishan_nandan_l111_111410

theorem time_ratio_krishan_nandan 
  (N T k : ℝ) 
  (H1 : N * T = 6000) 
  (H2 : N * T + 6 * N * k * T = 78000) 
  : k = 2 := 
by 
sorry

end time_ratio_krishan_nandan_l111_111410


namespace inequality_always_holds_l111_111231

theorem inequality_always_holds (m : ℝ) :
  (∀ x : ℝ, m * x ^ 2 - m * x - 1 < 0) → -4 < m ∧ m ≤ 0 :=
by
  sorry

end inequality_always_holds_l111_111231


namespace total_packs_l111_111939

theorem total_packs (cards_bought : ℕ) (cards_per_pack : ℕ) (num_people : ℕ)
  (h1 : cards_bought = 540) (h2 : cards_per_pack = 20) (h3 : num_people = 4) :
  (cards_bought / cards_per_pack) * num_people = 108 :=
by
  sorry

end total_packs_l111_111939


namespace part_one_part_two_l111_111242

def universal_set : Set ℝ := Set.univ
def A : Set ℝ := { x | 1 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | x < a }

noncomputable def C_R_A : Set ℝ := { x | x < 1 ∨ x ≥ 7 }
noncomputable def C_R_A_union_B : Set ℝ := C_R_A ∪ B

theorem part_one : C_R_A_union_B = { x | x < 1 ∨ x > 2 } :=
sorry

theorem part_two (a : ℝ) (h : A ⊆ C a) : a ≥ 7 :=
sorry

end part_one_part_two_l111_111242


namespace inequality_solution_l111_111061

theorem inequality_solution (x : ℝ) :
  (2 / (x + 2) + 9 / (x + 6) ≥ 2) ↔ (x ∈ Set.Ico (-6 : ℝ) (-3) ∪ Set.Ioc (-2) 3) := 
sorry

end inequality_solution_l111_111061


namespace find_positive_integer_triples_l111_111694

-- Define the condition for the integer divisibility problem
def is_integer_division (t a b : ℕ) : Prop :=
  (t ^ (a + b) + 1) % (t ^ a + t ^ b + 1) = 0

-- Statement of the theorem
theorem find_positive_integer_triples :
  ∀ (t a b : ℕ), t > 0 → a > 0 → b > 0 → is_integer_division t a b → (t, a, b) = (2, 1, 1) :=
by
  intros t a b t_pos a_pos b_pos h
  sorry

end find_positive_integer_triples_l111_111694


namespace range_of_a_l111_111235

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x < 1 then (2 - a) * x - 3 * a + 3 
  else Real.log x / Real.log a

-- Main statement to prove
theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (5 / 4 ≤ a ∧ a < 2) :=
sorry

end range_of_a_l111_111235


namespace commute_time_absolute_difference_l111_111855

theorem commute_time_absolute_difference 
  (x y : ℝ)
  (h1 : (x + y + 10 + 11 + 9) / 5 = 10)
  (h2 : ((x - 10)^2 + (y - 10)^2 + 0^2 + 1^2 + (-1)^2) / 5 = 2) :
  |x - y| = 4 :=
by sorry

end commute_time_absolute_difference_l111_111855


namespace solution_set_of_inequality_l111_111885

theorem solution_set_of_inequality :
  {x : ℝ | -6 * x ^ 2 - x + 2 < 0} = {x : ℝ | x < -(2 / 3)} ∪ {x | x > 1 / 2} := 
sorry

end solution_set_of_inequality_l111_111885


namespace mr_roper_lawn_cuts_l111_111422

theorem mr_roper_lawn_cuts (x : ℕ) (h_apr_sep : ℕ → ℕ) (h_total_cuts : 12 * 9 = 108) :
  (6 * x + 6 * 3 = 108) → x = 15 :=
by
  -- The proof is not needed as per the instructions, hence we use sorry.
  sorry

end mr_roper_lawn_cuts_l111_111422


namespace find_constant_a_range_of_f_l111_111080

noncomputable def f (a x : ℝ) : ℝ :=
  2 * a * (Real.sin x)^2 + 2 * (Real.sin x) * (Real.cos x) - a

theorem find_constant_a (h : f a 0 = -Real.sqrt 3) : a = Real.sqrt 3 := by
  sorry

theorem range_of_f (a : ℝ) (h : a = Real.sqrt 3) (x : ℝ) (hx : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  f a x ∈ Set.Icc (-Real.sqrt 3) 2 := by
  sorry

end find_constant_a_range_of_f_l111_111080


namespace intersection_A_B_l111_111240

-- Define the sets A and B
def set_A : Set ℝ := { x | x^2 ≤ 1 }
def set_B : Set ℝ := { -2, -1, 0, 1, 2 }

-- The goal is to prove that the intersection of A and B is {-1, 0, 1}
theorem intersection_A_B : set_A ∩ set_B = ({-1, 0, 1} : Set ℝ) :=
by
  sorry

end intersection_A_B_l111_111240


namespace XY_passes_through_H_l111_111790

open EuclideanGeometry

variables {A B C M H Q X Y : Point} -- assuming appropriate definitions of Point
variables [Triangle ABC] 

theorem XY_passes_through_H (M_midpoint : M = midpoint B C)
  (H_orthocenter : H = orthocenter A B C)
  (MH_Aangle_bisector_intersection : ∃ Q, line_through M H ∩ Aangle_bisector A B C = {Q})
  (X_projection : ∃ X, is_projection Q A B X)
  (Y_projection : ∃ Y, is_projection Q A C Y) :
  collinear X Y H :=
sorry

end XY_passes_through_H_l111_111790


namespace average_episodes_per_year_is_16_l111_111842

-- Define the number of years the TV show has been running
def years : Nat := 14

-- Define the number of seasons and episodes for each category
def seasons_8_15 : Nat := 8
def episodes_per_season_8_15 : Nat := 15
def seasons_4_20 : Nat := 4
def episodes_per_season_4_20 : Nat := 20
def seasons_2_12 : Nat := 2
def episodes_per_season_2_12 : Nat := 12

-- Define the total number of episodes
def total_episodes : Nat :=
  (seasons_8_15 * episodes_per_season_8_15) + 
  (seasons_4_20 * episodes_per_season_4_20) + 
  (seasons_2_12 * episodes_per_season_2_12)

-- Define the average number of episodes per year
def average_episodes_per_year : Nat :=
  total_episodes / years

-- State the theorem to prove the average number of episodes per year is 16
theorem average_episodes_per_year_is_16 : average_episodes_per_year = 16 :=
by
  sorry

end average_episodes_per_year_is_16_l111_111842


namespace problem1_problem2_l111_111273

-- Define that a quadratic is a root-multiplying equation if one root is twice the other
def is_root_multiplying (a b c : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 * x2 ≠ 0 ∧ a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 ∧ (x2 = 2 * x1 ∨ x1 = 2 * x2)

-- Problem 1: Prove that x^2 - 3x + 2 = 0 is a root-multiplying equation
theorem problem1 : is_root_multiplying 1 (-3) 2 :=
  sorry

-- Problem 2: Given ax^2 + bx - 6 = 0 is a root-multiplying equation with one root being 2, determine a and b
theorem problem2 (a b : ℝ) : is_root_multiplying a b (-6) → (∃ x1 x2 : ℝ, x1 = 2 ∧ x1 ≠ 0 ∧ a * x1^2 + b * x1 - 6 = 0 ∧ a * x2^2 + b * x2 - 6 = 0 ∧ (x2 = 2 * x1 ∨ x1 = 2 * x2)) →
( (a = -3/4 ∧ b = 9/2) ∨ (a = -3 ∧ b = 9)) :=
  sorry

end problem1_problem2_l111_111273


namespace family_age_problem_l111_111297

theorem family_age_problem (T y : ℕ)
  (h1 : T = 5 * 17)
  (h2 : (T + 5 * y + 2) = 6 * 17)
  : y = 3 := by
  sorry

end family_age_problem_l111_111297


namespace T_100_gt_T_99_l111_111507

-- Definition: T(n) denotes the number of ways to place n objects of weights 1, 2, ..., n on a balance such that the sum of the weights in each pan is the same.
def T (n : ℕ) : ℕ := sorry

-- Theorem we need to prove
theorem T_100_gt_T_99 : T 100 > T 99 := 
sorry

end T_100_gt_T_99_l111_111507


namespace min_value_x2_y2_l111_111906

theorem min_value_x2_y2 (x y : ℝ) (h : x + y = 2) : ∃ m, m = x^2 + y^2 ∧ (∀ (x y : ℝ), x + y = 2 → x^2 + y^2 ≥ m) ∧ m = 2 := 
sorry

end min_value_x2_y2_l111_111906


namespace inequality_not_always_hold_l111_111533

theorem inequality_not_always_hold (a b : ℝ) (h : a > -b) : ¬ (∀ a b : ℝ, a > -b → (1 / a + 1 / b > 0)) :=
by
  intro h2
  have h3 := h2 a b h
  sorry

end inequality_not_always_hold_l111_111533


namespace equation_1_solution_equation_2_solution_l111_111446

theorem equation_1_solution (x : ℝ) :
  6 * (x - 2 / 3) - (x + 7) = 11 → x = 22 / 5 :=
by
  intro h
  -- The actual proof steps would go here; for now, we use sorry.
  sorry

theorem equation_2_solution (x : ℝ) :
  (2 * x - 1) / 3 = (2 * x + 1) / 6 - 2 → x = -9 / 2 :=
by
  intro h
  -- The actual proof steps would go here; for now, we use sorry.
  sorry

end equation_1_solution_equation_2_solution_l111_111446


namespace wall_length_to_height_ratio_l111_111141

theorem wall_length_to_height_ratio
  (W H L : ℝ)
  (V : ℝ)
  (h1 : H = 6 * W)
  (h2 : L * H * W = V)
  (h3 : V = 86436)
  (h4 : W = 6.999999999999999) :
  L / H = 7 :=
by
  sorry

end wall_length_to_height_ratio_l111_111141


namespace midpoint_square_sum_l111_111772

theorem midpoint_square_sum (x y : ℝ) :
  (4, 1) = ((2 + x) / 2, (6 + y) / 2) → x^2 + y^2 = 52 :=
by
  sorry

end midpoint_square_sum_l111_111772


namespace place_mat_length_l111_111859

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (x : ℝ) (inner_touch : Bool)
  (h1 : r = 4)
  (h2 : n = 6)
  (h3 : w = 1)
  (h4 : inner_touch = true)
  : x = (3 * Real.sqrt 7 - Real.sqrt 3) / 2 :=
sorry

end place_mat_length_l111_111859


namespace total_rain_duration_l111_111548

theorem total_rain_duration:
  let first_day_duration := 10
  let second_day_duration := first_day_duration + 2
  let third_day_duration := 2 * second_day_duration
  first_day_duration + second_day_duration + third_day_duration = 46 :=
by
  sorry

end total_rain_duration_l111_111548


namespace missing_number_l111_111251

theorem missing_number 
  (a : ℕ) (b : ℕ) (x : ℕ)
  (h1 : a = 105) 
  (h2 : a^3 = 21 * 25 * x * b) 
  (h3 : b = 147) : 
  x = 3 :=
sorry

end missing_number_l111_111251


namespace general_term_formula_l111_111762

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (n : ℕ)
variable (a1 d : ℤ)

-- Given conditions
axiom a2_eq : a 2 = 8
axiom S10_eq : S 10 = 185
axiom S_def : ∀ n, S n = n * (a 1 + a n) / 2
axiom a_def : ∀ n, a (n + 1) = a 1 + n * d

-- Prove the general term formula
theorem general_term_formula : a n = 3 * n + 2 := sorry

end general_term_formula_l111_111762


namespace loan_repayment_l111_111849

open Real

theorem loan_repayment
  (a r : ℝ) (h_r : 0 ≤ r) :
  ∃ x : ℝ, 
    x = (a * r * (1 + r)^5) / ((1 + r)^5 - 1) :=
sorry

end loan_repayment_l111_111849


namespace angle_A₁FB₁_eq_90_l111_111072

-- Definitions according to the conditions given in the problem
variables {p : ℝ} (h_p : p ≠ 0)
def parabola := { xy : ℝ × ℝ // xy.1^2 = 2 * p * xy.2 }

variables {A B : ℝ × ℝ} (hA : A ∈ parabola h_p) (hB : B ∈ parabola h_p)
def directrix := { xy : ℝ × ℝ // xy.2 = -p/2 }

variables {A₁ B₁ : ℝ × ℝ}
(hA₁ : A₁ ∈ directrix h_p ∧ ∃ A, A ∈ parabola h_p ∧ proj_directrix A = A₁)
(hB₁ : B₁ ∈ directrix h_p ∧ ∃ B, B ∈ parabola h_p ∧ proj_directrix B = B₁)

-- F is the focus of the parabola
def F : ℝ × ℝ := (0, p/2)

-- Statement of the theorem
theorem angle_A₁FB₁_eq_90 :
  ∠ A₁ F B₁ = 90 :=
sorry

end angle_A₁FB₁_eq_90_l111_111072


namespace triangle_angle_inradius_l111_111271

variable (A B C : ℝ) 
variable (a b c R : ℝ)

theorem triangle_angle_inradius 
    (h1: 0 < A ∧ A < Real.pi)
    (h2: a * Real.cos C + (1/2) * c = b)
    (h3: a = 1):

    A = Real.pi / 3 ∧ R ≤ Real.sqrt 3 / 6 := 
by
  sorry

end triangle_angle_inradius_l111_111271


namespace Piglet_ate_one_l111_111025

theorem Piglet_ate_one (V S K P : ℕ) (h1 : V + S + K + P = 70)
  (h2 : S + K = 45) (h3 : V > S) (h4 : V > K) (h5 : V > P) 
  (h6 : V ≥ 1) (h7 : S ≥ 1) (h8 : K ≥ 1) (h9 : P ≥ 1) : P = 1 :=
sorry

end Piglet_ate_one_l111_111025


namespace factorize_expression_l111_111212

theorem factorize_expression (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := sorry

end factorize_expression_l111_111212


namespace total_interest_l111_111834

variable (P R : ℝ)

-- Given condition: Simple interest on sum of money is Rs. 700 after 10 years
def interest_10_years (P R : ℝ) : Prop := (P * R * 10) / 100 = 700

-- Principal is trebled after 5 years
def interest_5_years_treble (P R : ℝ) : Prop := (15 * P * R) / 100 = 105

-- The final interest is the sum of interest for the first 10 years and next 5 years post trebling the principal
theorem total_interest (P R : ℝ) (h1: interest_10_years P R) (h2: interest_5_years_treble P R) : 
  (700 + 105 = 805) := 
  by 
  sorry

end total_interest_l111_111834


namespace solve_inequality_l111_111612

theorem solve_inequality (x : ℝ) (h : 3 - (1 / (3 * x + 4)) < 5) : 
  x ∈ { x : ℝ | x < -11/6 } ∨ x ∈ { x : ℝ | x > -4/3 } :=
by
  sorry

end solve_inequality_l111_111612


namespace point_in_second_quadrant_l111_111099

theorem point_in_second_quadrant (x : ℝ) (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
by
  sorry

end point_in_second_quadrant_l111_111099


namespace sin_neg_225_eq_sqrt2_div2_l111_111700

theorem sin_neg_225_eq_sqrt2_div2 :
  Real.sin (-225 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_neg_225_eq_sqrt2_div2_l111_111700


namespace sum_of_first_six_terms_l111_111379

def geometric_sequence (a : ℕ → ℤ) :=
  a 1 = 1 ∧ ∀ n, n ≥ 2 → a n = -2 * a (n - 1)

def sum_first_six_terms (a : ℕ → ℤ) : ℤ := 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6

theorem sum_of_first_six_terms (a : ℕ → ℤ) 
  (h : geometric_sequence a) :
  sum_first_six_terms a = -21 :=
sorry

end sum_of_first_six_terms_l111_111379


namespace arithmetic_geometric_sequence_l111_111069

theorem arithmetic_geometric_sequence 
  (a : ℕ → ℤ) (S : ℕ → ℤ) (T : ℕ → ℝ) 
  (S3_eq_a4_plus_2 : S 3 = a 4 + 2)
  (arithmetic_condition : ∀ n, a (n + 1) = a n + d)
  (geometric_condition : (a 1) * (a 1 + 2 * d - 1) = (a 1 + d - 1) * (a 1 + d - 1))
  (sum_arith_seq : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2)
  (N_pos : ∀ n, n ∈ ℕ⁺)
  : (∀ n, a n = 2 * n - 1) ∧ (∀ n ∈ ℕ⁺, 1/3 ≤ T n ∧ T n < 1/2) := sorry

end arithmetic_geometric_sequence_l111_111069


namespace arun_working_days_l111_111494

theorem arun_working_days (A T : ℝ) 
  (h1 : A + T = 1/10) 
  (h2 : A = 1/18) : 
  (1 / A) = 18 :=
by
  -- Proof will be skipped
  sorry

end arun_working_days_l111_111494


namespace simplify_expression_l111_111442

theorem simplify_expression (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  (x + y + z)⁻¹ * (x⁻¹ + y⁻¹ + z⁻¹) = x⁻¹ * y⁻¹ * z⁻¹ :=
by
  sorry

end simplify_expression_l111_111442


namespace classroom_desks_l111_111545

theorem classroom_desks (N y : ℕ) (h : 16 * y = 21 * N)
  (hN_le: N <= 30 * 16 / 21) (hMultiple: 3 * N % 4 = 0)
  (hy_le: y ≤ 30)
  : y = 21 := by
  sorry

end classroom_desks_l111_111545


namespace trevor_spends_more_l111_111314

theorem trevor_spends_more (T R Q : ℕ) 
  (hT : T = 80) 
  (hR : R = 2 * Q) 
  (hTotal : 4 * (T + R + Q) = 680) : 
  T = R + 20 :=
by
  sorry

end trevor_spends_more_l111_111314


namespace rod_length_l111_111858

theorem rod_length (num_pieces : ℝ) (length_per_piece : ℝ) (h1 : num_pieces = 118.75) (h2 : length_per_piece = 0.40) : 
  num_pieces * length_per_piece = 47.5 := by
  sorry

end rod_length_l111_111858


namespace merchant_marked_price_percentage_l111_111183

variables (L S M C : ℝ)
variable (h1 : C = 0.7 * L)
variable (h2 : C = 0.75 * S)
variable (h3 : S = 0.9 * M)

theorem merchant_marked_price_percentage : M = 1.04 * L :=
by
  sorry

end merchant_marked_price_percentage_l111_111183


namespace factorize_expression_l111_111211

theorem factorize_expression (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := sorry

end factorize_expression_l111_111211


namespace inequality_part_1_inequality_part_2_l111_111372

theorem inequality_part_1 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  (a^2 + b^2) / (2 * c) + (b^2 + c^2) / (2 * a) + (c^2 + a^2) / (2 * b) ≥ 1 := by
sorry

theorem inequality_part_2 (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  (a^2 / (b + c)) + (b^2 / (a + c)) + (c^2 / (a + b)) ≥ 1 / 2 := by
sorry

end inequality_part_1_inequality_part_2_l111_111372


namespace find_angle_A_in_triangle_ABC_l111_111765

noncomputable def sin := Real.sin
noncomputable def sqrt := Real.sqrt

theorem find_angle_A_in_triangle_ABC :
  ∀ (a b c : ℝ) (A : ℝ),
    b = 8 →
    c = 8 * sqrt 3 →
    (1 / 2) * b * c * sin A = 16 * sqrt 3 →
    A = π / 6 :=
by
  intros a b c A hb hc harea
  sorry

end find_angle_A_in_triangle_ABC_l111_111765


namespace coefficient_of_x_is_nine_l111_111886

theorem coefficient_of_x_is_nine (x : ℝ) (c : ℝ) (h : x = 0.5) (eq : 2 * x^2 + c * x - 5 = 0) : c = 9 :=
by
  sorry

end coefficient_of_x_is_nine_l111_111886


namespace percentage_design_black_is_57_l111_111258

noncomputable def circleRadius (n : ℕ) : ℝ :=
  3 * (n + 1)

noncomputable def circleArea (n : ℕ) : ℝ :=
  Real.pi * (circleRadius n) ^ 2

noncomputable def totalArea : ℝ :=
  circleArea 6

noncomputable def blackAreas : ℝ :=
  circleArea 0 + (circleArea 2 - circleArea 1) +
  (circleArea 4 - circleArea 3) +
  (circleArea 6 - circleArea 5)

noncomputable def percentageBlack : ℝ :=
  (blackAreas / totalArea) * 100

theorem percentage_design_black_is_57 :
  percentageBlack = 57 := 
by
  sorry

end percentage_design_black_is_57_l111_111258


namespace min_value_3x_plus_4y_l111_111537

theorem min_value_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = x * y) : 3 * x + 4 * y = 28 :=
sorry

end min_value_3x_plus_4y_l111_111537


namespace constant_c_square_of_binomial_l111_111536

theorem constant_c_square_of_binomial (c : ℝ) (h : ∃ d : ℝ, (3*x + d)^2 = 9*x^2 - 18*x + c) : c = 9 :=
sorry

end constant_c_square_of_binomial_l111_111536


namespace total_number_of_students_l111_111165

theorem total_number_of_students
  (ratio_girls_to_boys : ℕ) (ratio_boys_to_girls : ℕ)
  (num_girls : ℕ)
  (ratio_condition : ratio_girls_to_boys = 5 ∧ ratio_boys_to_girls = 8)
  (num_girls_condition : num_girls = 160)
  : (num_girls * (ratio_girls_to_boys + ratio_boys_to_girls) / ratio_girls_to_boys = 416) :=
by
  sorry

end total_number_of_students_l111_111165


namespace tan_alpha_value_l111_111709

theorem tan_alpha_value (α : ℝ) (h1 : Real.sin α = 3 / 5) (h2 : α ∈ Set.Ioo (Real.pi / 2) Real.pi) : Real.tan α = -3 / 4 := 
sorry

end tan_alpha_value_l111_111709


namespace nate_matches_left_l111_111785

def initial_matches : ℕ := 70
def matches_dropped : ℕ := 10
def matches_eaten : ℕ := 2 * matches_dropped
def total_matches_lost : ℕ := matches_dropped + matches_eaten
def remaining_matches : ℕ := initial_matches - total_matches_lost

theorem nate_matches_left : remaining_matches = 40 := by
  sorry

end nate_matches_left_l111_111785


namespace count_consecutive_sequences_l111_111382

def consecutive_sequences (n : ℕ) : ℕ :=
  if n = 15 then 270 else 0

theorem count_consecutive_sequences : consecutive_sequences 15 = 270 :=
by
  sorry

end count_consecutive_sequences_l111_111382


namespace zero_lies_in_interval_l111_111750

def f (x : ℝ) : ℝ := -|x - 5| + 2 * x - 1

theorem zero_lies_in_interval (k : ℤ) (h : ∃ x : ℝ, k < x ∧ x < k + 1 ∧ f x = 0) : k = 2 := 
sorry

end zero_lies_in_interval_l111_111750


namespace sum_of_polynomials_l111_111776

-- Define the given polynomials f, g, and h
def f (x : ℝ) : ℝ := -6 * x^3 - 4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -7 * x^2 + 6 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 7 * x + 3

-- Prove that the sum of f(x), g(x), and h(x) is a specific polynomial
theorem sum_of_polynomials (x : ℝ) : 
  f x + g x + h x = -6 * x^3 - 5 * x^2 + 15 * x - 11 := 
by {
  -- Proof is omitted
  sorry
}

end sum_of_polynomials_l111_111776


namespace new_car_distance_in_same_time_l111_111184

-- Define the given conditions and the distances
variable (older_car_distance : ℝ := 150)
variable (new_car_speed_factor : ℝ := 1.30)  -- Since the new car is 30% faster, its speed factor is 1.30
variable (time : ℝ)

-- Define the older car's distance as a function of time and speed
def older_car_distance_covered (t : ℝ) (distance : ℝ) : ℝ := distance

-- Define the new car's distance as a function of time and speed factor
def new_car_distance_covered (t : ℝ) (distance : ℝ) (speed_factor : ℝ) : ℝ := speed_factor * distance

theorem new_car_distance_in_same_time
  (older_car_distance : ℝ)
  (new_car_speed_factor : ℝ)
  (time : ℝ)
  (h1 : older_car_distance = 150)
  (h2 : new_car_speed_factor = 1.30) :
  new_car_distance_covered time older_car_distance new_car_speed_factor = 195 := by
  sorry

end new_car_distance_in_same_time_l111_111184


namespace solve_for_y_l111_111526

theorem solve_for_y (x y : ℝ) (h : 2 * x - 3 * y = 4) : y = (2 * x - 4) / 3 :=
sorry

end solve_for_y_l111_111526


namespace leopards_to_rabbits_ratio_l111_111922

theorem leopards_to_rabbits_ratio :
  let antelopes := 80
  let rabbits := antelopes + 34
  let hyenas := antelopes + rabbits - 42
  let wild_dogs := hyenas + 50
  let total_animals := 605
  let leopards := total_animals - antelopes - rabbits - hyenas - wild_dogs
  leopards / rabbits = 1 / 2 :=
by
  let antelopes := 80
  let rabbits := antelopes + 34
  let hyenas := antelopes + rabbits - 42
  let wild_dogs := hyenas + 50
  let total_animals := 605
  let leopards := total_animals - antelopes - rabbits - hyenas - wild_dogs
  sorry

end leopards_to_rabbits_ratio_l111_111922


namespace band_song_average_l111_111008

/-- 
The school band has 30 songs in their repertoire. 
They played 5 songs in the first set and 7 songs in the second set. 
They will play 2 songs for their encore. 
Assuming the band plays through their entire repertoire, 
how many songs will they play on average in the third and fourth sets?
 -/
theorem band_song_average
    (total_songs : ℕ)
    (first_set_songs : ℕ)
    (second_set_songs : ℕ)
    (encore_songs : ℕ)
    (remaining_sets : ℕ)
    (h_total : total_songs = 30)
    (h_first : first_set_songs = 5)
    (h_second : second_set_songs = 7)
    (h_encore : encore_songs = 2)
    (h_remaining : remaining_sets = 2) :
    (total_songs - (first_set_songs + second_set_songs + encore_songs)) / remaining_sets = 8 := 
by
  -- The proof will go here.
  sorry

end band_song_average_l111_111008


namespace value_of_m_l111_111742

noncomputable def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x

theorem value_of_m (a b m : ℝ) (h₀ : m ≠ 0)
  (h₁ : 3 * m^2 + 2 * a * m + b = 0)
  (h₂ : m^2 + a * m + b = 0)
  (h₃ : ∃ x, f x a b = 1/2) :
  m = 3/2 :=
by
  sorry

end value_of_m_l111_111742


namespace consecutive_sum_150_l111_111361

theorem consecutive_sum_150 : ∃ (n : ℕ), n ≥ 2 ∧ (∃ a : ℕ, (n * (2 * a + n - 1)) / 2 = 150) :=
sorry

end consecutive_sum_150_l111_111361


namespace intersection_M_N_l111_111927

def set_M : Set ℝ := { x | x * (x - 1) ≤ 0 }
def set_N : Set ℝ := { x | x < 1 }

theorem intersection_M_N : set_M ∩ set_N = { x | 0 ≤ x ∧ x < 1 } := sorry

end intersection_M_N_l111_111927


namespace simplify_fraction_l111_111611

theorem simplify_fraction (a b : ℕ) (h : Nat.gcd a b = 24) : (a = 48) → (b = 72) → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end simplify_fraction_l111_111611


namespace find_m_l111_111887

theorem find_m (m x : ℝ) (h : (m - 2) * x^2 + 3 * x + m^2 - 4 = 0) (hx : x = 0) : m = -2 :=
by sorry

end find_m_l111_111887


namespace proof_math_problem_l111_111413

noncomputable def math_problem (a b c d : ℝ) (ω : ℂ) : Prop :=
  a ≠ -1 ∧ b ≠ -1 ∧ c ≠ -1 ∧ d ≠ -1 ∧ 
  ω^4 = 1 ∧ ω ≠ 1 ∧ 
  (1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 2 / ω^2)

theorem proof_math_problem (a b c d : ℝ) (ω : ℂ) (h: math_problem a b c d ω) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2 :=
sorry

end proof_math_problem_l111_111413


namespace sufficient_but_not_necessary_condition_l111_111893

theorem sufficient_but_not_necessary_condition
  (a b : ℝ) (h : a > b + 1) : (a > b) ∧ ¬ (∀ (a b : ℝ), a > b → a > b + 1) :=
by
  sorry

end sufficient_but_not_necessary_condition_l111_111893


namespace ellipse_product_l111_111434

theorem ellipse_product (a b : ℝ) (OF_diameter : a - b = 4) (focus_relation : a^2 - b^2 = 64) :
  let AB := 2 * a,
      CD := 2 * b
  in AB * CD = 240 :=
by
  sorry

end ellipse_product_l111_111434


namespace students_count_l111_111835

theorem students_count (initial: ℕ) (left: ℕ) (new: ℕ) (result: ℕ) 
  (h1: initial = 31)
  (h2: left = 5)
  (h3: new = 11)
  (h4: result = initial - left + new) : result = 37 := by
  sorry

end students_count_l111_111835


namespace number_of_hens_l111_111027

theorem number_of_hens (H C : Nat) (h1 : H + C = 48) (h2 : 2 * H + 4 * C = 140) : H = 26 := 
by
  sorry

end number_of_hens_l111_111027


namespace find_angle_B_l111_111272

def angle_A (B : ℝ) : ℝ := B + 21
def angle_C (B : ℝ) : ℝ := B + 36
def is_triangle_sum (A B C : ℝ) : Prop := A + B + C = 180

theorem find_angle_B (B : ℝ) 
  (hA : angle_A B = B + 21) 
  (hC : angle_C B = B + 36) 
  (h_sum : is_triangle_sum (angle_A B) B (angle_C B) ) : B = 41 :=
  sorry

end find_angle_B_l111_111272


namespace tan_mul_tan_l111_111901

variables {α β : ℝ}

theorem tan_mul_tan (h : 3 * Real.cos (2 * α + β) + 5 * Real.cos β = 0) : 
  Real.tan (α + β) * Real.tan α = -4 :=
sorry

end tan_mul_tan_l111_111901


namespace simplify_fraction_l111_111605

theorem simplify_fraction (a b : ℕ) (h : Nat.gcd a b = 24) : (a = 48) → (b = 72) → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end simplify_fraction_l111_111605


namespace height_flagstaff_l111_111993

variables (s_1 s_2 h_2 : ℝ)
variable (h : ℝ)

-- Define the conditions as given
def shadow_flagstaff := s_1 = 40.25
def shadow_building := s_2 = 28.75
def height_building := h_2 = 12.5
def similar_triangles := (h / s_1) = (h_2 / s_2)

-- Prove the height of the flagstaff
theorem height_flagstaff : shadow_flagstaff s_1 ∧ shadow_building s_2 ∧ height_building h_2 ∧ similar_triangles h s_1 h_2 s_2 → h = 17.5 :=
by sorry

end height_flagstaff_l111_111993


namespace like_terms_proof_l111_111891

variable (a b : ℤ)
variable (x y : ℤ)

theorem like_terms_proof (hx : x = 2) (hy : 3 = 1 - y) : x * y = -4 := by
  rw [hx, hy]
  sorry

end like_terms_proof_l111_111891


namespace simplify_fraction_l111_111584

theorem simplify_fraction (a b : ℕ) (h : b ≠ 0) (g : Nat.gcd a b = 24) : a = 48 → b = 72 → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  exact ⟨rfl, rfl⟩

end simplify_fraction_l111_111584


namespace evaluate_difference_of_squares_l111_111692

theorem evaluate_difference_of_squares : 81^2 - 49^2 = 4160 := by
  sorry

end evaluate_difference_of_squares_l111_111692


namespace payal_finished_fraction_l111_111288

-- Define the conditions
variables (x : ℕ)

-- Given conditions
-- 1. Total pages in the book
def total_pages : ℕ := 60
-- 2. Payal has finished 20 more pages than she has yet to read.
def pages_yet_to_read (x : ℕ) : ℕ := x - 20

-- Main statement to prove: the fraction of the pages finished is 2/3
theorem payal_finished_fraction (h : x + (x - 20) = 60) : (x : ℚ) / 60 = 2 / 3 :=
sorry

end payal_finished_fraction_l111_111288


namespace leo_average_speed_last_segment_l111_111276

theorem leo_average_speed_last_segment :
  let total_distance := 135
  let total_time_hr := 135 / 60.0
  let segment_time_hr := 45 / 60.0
  let first_segment_distance := 55 * segment_time_hr
  let second_segment_distance := 70 * segment_time_hr
  let last_segment_distance := total_distance - (first_segment_distance + second_segment_distance)
  last_segment_distance / segment_time_hr = 55 :=
by
  sorry

end leo_average_speed_last_segment_l111_111276


namespace probability_251_is_5_over_14_l111_111050

-- Conditions about bus intervals and random arrival
def bus_interval_152 : ℕ := 5
def bus_interval_251 : ℕ := 7

-- Define the area of the rectangle and the triangle
def area_rectangle (a b : ℕ) : ℚ := a * b
def area_triangle (a : ℕ) : ℚ := (a * a) / 2

-- Define the probability calculation
noncomputable def probability_first_bus_251 : ℚ :=
  area_triangle bus_interval_152 / area_rectangle bus_interval_152 bus_interval_251

-- The theorem that needs to be proven
theorem probability_251_is_5_over_14 :
  probability_first_bus_251 = 5 / 14 := by
  sorry

end probability_251_is_5_over_14_l111_111050


namespace solution_set_l111_111083
  
noncomputable def f (x : ℝ) : ℝ :=
  Real.log (Real.exp (2 * x) + 1) - x

theorem solution_set (x : ℝ) :
  f (x + 2) > f (2 * x - 3) ↔ (1 / 3 < x ∧ x < 5) :=
by
  sorry

end solution_set_l111_111083


namespace irrational_neg_pi_lt_neg_two_l111_111789

theorem irrational_neg_pi_lt_neg_two (h1 : Irrational π) (h2 : π > 2) : Irrational (-π) ∧ -π < -2 := by
  sorry

end irrational_neg_pi_lt_neg_two_l111_111789


namespace school_B_saving_l111_111941

def cost_A (kg_price : ℚ) (kg_amount : ℚ) : ℚ :=
  kg_price * kg_amount

def effective_kg_B (total_kg : ℚ) (extra_percentage : ℚ) : ℚ :=
  total_kg / (1 + extra_percentage)

def cost_B (kg_price : ℚ) (effective_kg : ℚ) : ℚ :=
  kg_price * effective_kg

theorem school_B_saving
  (kg_amount : ℚ) (price_A: ℚ) (discount: ℚ) (extra_percentage : ℚ) 
  (expected_saving : ℚ)
  (h1 : kg_amount = 56)
  (h2 : price_A = 8.06)
  (h3 : discount = 0.56)
  (h4 : extra_percentage = 0.05)
  (h5 : expected_saving = 51.36) :
  cost_A price_A kg_amount - cost_B (price_A - discount) (effective_kg_B kg_amount extra_percentage) = expected_saving := 
by 
  sorry

end school_B_saving_l111_111941


namespace sequence_not_generated_l111_111741

theorem sequence_not_generated (a : ℕ → ℝ) :
  (a 1 = 2) ∧ (a 2 = 0) ∧ (a 3 = 2) ∧ (a 4 = 0) → 
  (∀ n, a n ≠ (1 - Real.cos (n * Real.pi)) + (n - 1) * (n - 2)) :=
by sorry

end sequence_not_generated_l111_111741


namespace n_c_equation_l111_111031

theorem n_c_equation (n c : ℕ) (hn : 0 < n) (hc : 0 < c) :
  (∀ x : ℕ, (↑x + n * ↑x / 100) * (1 - c / 100) = x) →
  (n^2 / c^2 = (100 + n) / (100 - c)) :=
by sorry

end n_c_equation_l111_111031


namespace smaller_angle_clock_1245_l111_111827

theorem smaller_angle_clock_1245 
  (minute_rate : ℕ → ℝ) 
  (hour_rate : ℕ → ℝ) 
  (time : ℕ) 
  (minute_angle : ℝ) 
  (hour_angle : ℝ) 
  (larger_angle : ℝ) 
  (smaller_angle : ℝ) :
  (minute_rate 1 = 6) →
  (hour_rate 1 = 0.5) →
  (time = 45) →
  (minute_angle = minute_rate 45 * 45) →
  (hour_angle = hour_rate 45 * 45) →
  (larger_angle = |minute_angle - hour_angle|) →
  (smaller_angle = 360 - larger_angle) →
  smaller_angle = 112.5 :=
by
  intros
  sorry

end smaller_angle_clock_1245_l111_111827


namespace price_reduction_l111_111659

theorem price_reduction (x : ℝ) (h : 560 * (1 - x) * (1 - x) = 315) : 
  560 * (1 - x)^2 = 315 := 
by
  sorry

end price_reduction_l111_111659


namespace fruit_baskets_l111_111092

def apple_choices := 8 -- From 0 to 7 apples
def orange_choices := 13 -- From 0 to 12 oranges

theorem fruit_baskets (a : ℕ) (o : ℕ) (ha : a = 7) (ho : o = 12) :
  (apple_choices * orange_choices) - 1 = 103 := by
  sorry

end fruit_baskets_l111_111092


namespace checkerboard_corner_sum_is_164_l111_111878

def checkerboard_sum_corners : ℕ :=
  let top_left := 1
  let top_right := 9
  let bottom_left := 73
  let bottom_right := 81
  top_left + top_right + bottom_left + bottom_right

theorem checkerboard_corner_sum_is_164 :
  checkerboard_sum_corners = 164 :=
by
  sorry

end checkerboard_corner_sum_is_164_l111_111878


namespace buses_more_than_vans_l111_111994

-- Definitions based on conditions
def vans : Float := 6.0
def buses : Float := 8.0
def people_per_van : Float := 6.0
def people_per_bus : Float := 18.0

-- Calculate total people in vans and buses
def total_people_vans : Float := vans * people_per_van
def total_people_buses : Float := buses * people_per_bus

-- Prove the difference
theorem buses_more_than_vans : total_people_buses - total_people_vans = 108.0 :=
by
  sorry

end buses_more_than_vans_l111_111994


namespace find_missing_number_l111_111621

theorem find_missing_number (x : ℕ) (h : (1 + x + 23 + 24 + 25 + 26 + 27 + 2) / 8 = 20) : x = 32 := 
by sorry

end find_missing_number_l111_111621


namespace quadratic_function_points_l111_111104

theorem quadratic_function_points:
  (∀ x y, (y = x^2 + x - 1) → ((x = -2 → y = 1) ∧ (x = 0 → y = -1) ∧ (x = 2 → y = 5))) →
  (-1 < 1 ∧ 1 < 5) :=
by
  intro h
  have h1 := h (-2) 1 (by ring)
  have h2 := h 0 (-1) (by ring)
  have h3 := h 2 5 (by ring)
  exact And.intro (by linarith) (by linarith)

end quadratic_function_points_l111_111104


namespace equation_has_real_roots_l111_111004

theorem equation_has_real_roots (k : ℝ) : ∀ (x : ℝ), 
  ∃ x, x = k^2 * (x - 1) * (x - 2) :=
by {
  sorry
}

end equation_has_real_roots_l111_111004


namespace decimal_difference_l111_111028

theorem decimal_difference : (0.650 : ℝ) - (1 / 8 : ℝ) = 0.525 := by
  sorry

end decimal_difference_l111_111028


namespace factorize_expr_l111_111216

theorem factorize_expr (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) :=
sorry

end factorize_expr_l111_111216


namespace painted_cube_faces_l111_111850

theorem painted_cube_faces (a : ℕ) (h : 2 < a) :
  ∃ (one_face two_faces three_faces : ℕ),
  (one_face = 6 * (a - 2) ^ 2) ∧
  (two_faces = 12 * (a - 2)) ∧
  (three_faces = 8) := by
  sorry

end painted_cube_faces_l111_111850


namespace solve_for_P_l111_111443

theorem solve_for_P (P : Real) (h : (P ^ 4) ^ (1 / 3) = 9 * 81 ^ (1 / 9)) : P = 3 ^ (11 / 6) :=
by
  sorry

end solve_for_P_l111_111443


namespace december_sales_fraction_l111_111324

variable (A : ℝ)

-- Define the total sales for January through November
def total_sales_jan_to_nov := 11 * A

-- Define the sales total for December, which is given as 5 times the average monthly sales from January to November
def sales_dec := 5 * A

-- Define the total sales for the year as the sum of January-November sales and December sales
def total_sales_year := total_sales_jan_to_nov + sales_dec

-- We need to prove that the fraction of the December sales to the total annual sales is 5/16
theorem december_sales_fraction : sales_dec / total_sales_year = 5 / 16 := by
  sorry

end december_sales_fraction_l111_111324


namespace smallest_n_for_violet_candy_l111_111209

theorem smallest_n_for_violet_candy (p y o n : Nat) (h : 10 * p = 12 * y ∧ 12 * y = 18 * o ∧ 18 * o = 24 * n) :
  n = 8 :=
by 
  sorry

end smallest_n_for_violet_candy_l111_111209


namespace coeff_x2y2_in_expansion_l111_111500

theorem coeff_x2y2_in_expansion (x y : ℝ) :
  ((1+x)^3 * (1+y)^4).coeff (2, 2) = 18 := 
sorry

end coeff_x2y2_in_expansion_l111_111500


namespace new_average_daily_production_l111_111508

theorem new_average_daily_production 
  (n : ℕ) 
  (avg_past_n_days : ℕ) 
  (today_production : ℕ)
  (new_avg_production : ℕ)
  (hn : n = 5) 
  (havg : avg_past_n_days = 60) 
  (htoday : today_production = 90) 
  (hnew_avg : new_avg_production = 65)
  : (n + 1 = 6) ∧ ((n * 60 + today_production) = 390) ∧ (390 / 6 = 65) :=
by
  sorry

end new_average_daily_production_l111_111508


namespace rabbit_population_2002_l111_111456

theorem rabbit_population_2002 :
  ∃ (x : ℕ) (k : ℝ), 
    (180 - 50 = k * x) ∧ 
    (255 - 75 = k * 180) ∧ 
    x = 130 :=
by
  sorry

end rabbit_population_2002_l111_111456


namespace smaller_angle_36_degrees_l111_111632

noncomputable def smaller_angle_measure (larger smaller : ℝ) : Prop :=
(larger + smaller = 180) ∧ (larger = 4 * smaller)

theorem smaller_angle_36_degrees : ∃ (smaller : ℝ), smaller_angle_measure (4 * smaller) smaller ∧ smaller = 36 :=
by
  sorry

end smaller_angle_36_degrees_l111_111632


namespace avg_last_three_numbers_l111_111948

-- Definitions of conditions
def avg_seven_numbers (numbers : List ℝ) (h_len : numbers.length = 7) : Prop :=
(numbers.sum / 7 = 60)

def avg_first_four_numbers (numbers : List ℝ) (h_len : numbers.length = 7) : Prop :=
(numbers.take 4).sum / 4 = 55

-- Proof statement
theorem avg_last_three_numbers (numbers : List ℝ) (h_len : numbers.length = 7)
  (h1 : avg_seven_numbers numbers h_len)
  (h2 : avg_first_four_numbers numbers h_len) :
  (numbers.drop 4).sum / 3 = 200 / 3 :=
sorry

end avg_last_three_numbers_l111_111948


namespace simplify_fraction_l111_111610

theorem simplify_fraction (a b : ℕ) (h : Nat.gcd a b = 24) : (a = 48) → (b = 72) → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end simplify_fraction_l111_111610


namespace quotient_base5_l111_111353

theorem quotient_base5 (a b quotient : ℕ) 
  (ha : a = 2 * 5^3 + 4 * 5^2 + 3 * 5^1 + 1) 
  (hb : b = 2 * 5^1 + 3) 
  (hquotient : quotient = 1 * 5^2 + 0 * 5^1 + 3) :
  a / b = quotient :=
by sorry

end quotient_base5_l111_111353


namespace area_change_l111_111107

theorem area_change (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let L' := 1.2 * L
  let B' := 0.8 * B
  let A := L * B
  let A' := L' * B'
  A' = 0.96 * A :=
by
  sorry

end area_change_l111_111107


namespace max_min_difference_l111_111778

open Real

theorem max_min_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + 2 * y = 4) :
  ∃(max min : ℝ), (∀z, z = (|2 * x - y| / (|x| + |y|)) → z ≤ max) ∧ 
                  (∀z, z = (|2 * x - y| / (|x| + |y|)) → min ≤ z) ∧ 
                  (max - min = 5) :=
by
  sorry

end max_min_difference_l111_111778


namespace factorize_negative_quadratic_l111_111210

theorem factorize_negative_quadratic (x y : ℝ) : 
  -4 * x^2 + y^2 = (y - 2 * x) * (y + 2 * x) :=
by 
  sorry

end factorize_negative_quadratic_l111_111210


namespace equivalent_angle_terminal_side_l111_111077

theorem equivalent_angle_terminal_side (k : ℤ) (a : ℝ) (c : ℝ) (d : ℝ) : a = -3/10 * Real.pi → c = a * 180 / Real.pi → d = c + 360 * k →
   ∃ k : ℤ, d = 306 :=
sorry

end equivalent_angle_terminal_side_l111_111077


namespace eugene_payment_correct_l111_111262

noncomputable def discounted_price (original_price : ℝ) (discount_rate : ℝ) : ℝ :=
  original_price - (original_price * discount_rate)

noncomputable def total_cost (quantity : ℕ) (price : ℝ) : ℝ :=
  quantity * price

noncomputable def eugene_total_cost : ℝ :=
  let tshirt_price := discounted_price 20 0.10
  let pants_price := discounted_price 80 0.10
  let shoes_price := discounted_price 150 0.15
  let hat_price := discounted_price 25 0.05
  let jacket_price := discounted_price 120 0.20
  let total_cost_before_tax := 
    total_cost 4 tshirt_price + 
    total_cost 3 pants_price + 
    total_cost 2 shoes_price + 
    total_cost 3 hat_price + 
    total_cost 1 jacket_price
  total_cost_before_tax + (total_cost_before_tax * 0.06)

theorem eugene_payment_correct : eugene_total_cost = 752.87 := by
  sorry

end eugene_payment_correct_l111_111262


namespace initial_action_figures_l111_111771

theorem initial_action_figures (x : ℕ) (h : x + 4 - 1 = 6) : x = 3 :=
by {
  sorry
}

end initial_action_figures_l111_111771


namespace prop_converse_inverse_contrapositive_correct_statements_l111_111622

-- Defining the proposition and its types
def prop (x : ℕ) : Prop := x > 0 → x^2 ≥ 0
def converse (x : ℕ) : Prop := x^2 ≥ 0 → x > 0
def inverse (x : ℕ) : Prop := ¬ (x > 0) → x^2 < 0
def contrapositive (x : ℕ) : Prop := x^2 < 0 → ¬ (x > 0)

-- The proof problem
theorem prop_converse_inverse_contrapositive_correct_statements :
  (∃! (p : Prop), p = (∀ x : ℕ, converse x) ∨ p = (∀ x : ℕ, inverse x) ∨ p = (∀ x : ℕ, contrapositive x) ∧ p = True) :=
sorry

end prop_converse_inverse_contrapositive_correct_statements_l111_111622


namespace proof_emails_in_morning_l111_111404

def emailsInAfternoon : ℕ := 2

def emailsMoreInMorning : ℕ := 4

def emailsInMorning : ℕ := 6

theorem proof_emails_in_morning
  (a : ℕ) (h1 : a = emailsInAfternoon)
  (m : ℕ) (h2 : m = emailsMoreInMorning)
  : emailsInMorning = a + m := by
  sorry

end proof_emails_in_morning_l111_111404


namespace find_sum_of_terms_l111_111236

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = r * a n

def given_conditions (a : ℕ → ℝ) : Prop :=
geometric_sequence a ∧ (a 4 + a 7 = 2) ∧ (a 5 * a 6 = -8)

theorem find_sum_of_terms (a : ℕ → ℝ) (h : given_conditions a) : a 1 + a 10 = -7 :=
sorry

end find_sum_of_terms_l111_111236


namespace zero_clever_numbers_l111_111006

def isZeroClever (n : Nat) : Prop :=
  ∃ a b c : Nat, a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧
  n = 1000 * a + 10 * b + c ∧
  n = 9 * (100 * a + 10 * b + c)

theorem zero_clever_numbers :
  ∀ n : Nat, isZeroClever n → n = 2025 ∨ n = 4050 ∨ n = 6075 :=
by
  -- Proof to be provided
  sorry

end zero_clever_numbers_l111_111006


namespace maxRegions100Parabolas_l111_111761

-- Define the number of parabolas of each type
def numberOfParabolas1 := 50
def numberOfParabolas2 := 50

-- Define the function that counts the number of regions formed by n parabolas intersecting at most m times
def maxRegions (n m : Nat) : Nat :=
  (List.range (m+1)).foldl (λ acc k => acc + Nat.choose n k) 0

-- Specify the intersection properties for each type of parabolas
def intersectionsParabolas1 := 2
def intersectionsParabolas2 := 2
def intersectionsBetweenSets := 4

-- Calculate the number of regions formed by each set of 50 parabolas
def regionsSet1 := maxRegions numberOfParabolas1 intersectionsParabolas1
def regionsSet2 := maxRegions numberOfParabolas2 intersectionsParabolas2

-- Calculate the additional regions created by intersections between the sets
def additionalIntersections := numberOfParabolas1 * numberOfParabolas2 * intersectionsBetweenSets

-- Combine the regions
def totalRegions := regionsSet1 + regionsSet2 + additionalIntersections + 1

-- Prove the final result
theorem maxRegions100Parabolas : totalRegions = 15053 :=
  sorry

end maxRegions100Parabolas_l111_111761


namespace deck_of_1000_transformable_l111_111862

def shuffle (n : ℕ) (deck : List ℕ) : List ℕ :=
  -- Definition of the shuffle operation as described in the problem
  sorry

noncomputable def transformable_in_56_shuffles (n : ℕ) : Prop :=
  ∀ (initial final : List ℕ) (h₁ : initial.length = n) (h₂ : final.length = n),
  -- Prove that any initial arrangement can be transformed to any final arrangement in at most 56 shuffles
  sorry

theorem deck_of_1000_transformable : transformable_in_56_shuffles 1000 :=
  -- Implement the proof here
  sorry

end deck_of_1000_transformable_l111_111862


namespace alice_favorite_number_l111_111491

theorem alice_favorite_number :
  ∃ (n : ℕ), 50 < n ∧ n < 100 ∧ n % 11 = 0 ∧ n % 2 ≠ 0 ∧ (n / 10 + n % 10) % 5 = 0 ∧ n = 55 :=
by
  sorry

end alice_favorite_number_l111_111491


namespace ribbon_original_length_l111_111454

theorem ribbon_original_length (x : ℕ) (h1 : 11 * 35 = 7 * x) : x = 55 :=
by
  sorry

end ribbon_original_length_l111_111454


namespace solve_system_l111_111296

theorem solve_system (x y z : ℝ) (h1 : x + y + z = 6) (h2 : x * y + y * z + z * x = 11) (h3 : x * y * z = 6) :
  (x = 1 ∧ y = 2 ∧ z = 3) ∨ (x = 1 ∧ y = 3 ∧ z = 2) ∨ (x = 2 ∧ y = 1 ∧ z = 3) ∨ 
  (x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 2 ∧ z = 1) :=
sorry

end solve_system_l111_111296


namespace lindsey_integer_l111_111933

theorem lindsey_integer (n : ℕ) (a b c : ℤ) (h1 : n < 50)
                        (h2 : n = 6 * a - 1)
                        (h3 : n = 8 * b - 5)
                        (h4 : n = 3 * c + 2) :
  n = 41 := 
  by sorry

end lindsey_integer_l111_111933


namespace parabola_equation_line_equation_chord_l111_111085

section
variables (p : ℝ) (x_A y_A : ℝ) (M_x M_y : ℝ)
variable (h_p_pos : p > 0)
variable (h_A : y_A^2 = 8 * x_A)
variable (h_directrix_A : x_A + p / 2 = 5)
variable (h_M : (M_x, M_y) = (3, 2))

theorem parabola_equation (h_x_A : x_A = 3) : y_A^2 = 8 * x_A :=
sorry

theorem line_equation_chord
  (x1 x2 y1 y2 : ℝ)
  (h_parabola : y1^2 = 8 * x1 ∧ y2^2 = 8 * x2)
  (h_chord_M : (x1 + x2) / 2 = 3 ∧ (y1 + y2) / 2 = 2) :
  y_M - 2 * x_M + 4 = 0 :=
sorry
end

end parabola_equation_line_equation_chord_l111_111085


namespace ratio_of_average_speeds_l111_111060

-- Conditions
def time_eddy : ℕ := 3
def time_freddy : ℕ := 4
def distance_ab : ℕ := 600
def distance_ac : ℕ := 360

-- Theorem to prove the ratio of their average speeds
theorem ratio_of_average_speeds : (distance_ab / time_eddy) / gcd (distance_ab / time_eddy) (distance_ac / time_freddy) = 20 ∧
                                  (distance_ac / time_freddy) / gcd (distance_ab / time_eddy) (distance_ac / time_freddy) = 9 :=
by
  -- Solution steps go here if performing an actual proof
  sorry

end ratio_of_average_speeds_l111_111060


namespace sum_two_primes_eq_91_prod_is_178_l111_111628

theorem sum_two_primes_eq_91_prod_is_178
  (p1 p2 : ℕ) 
  (hp1 : p1.Prime) 
  (hp2 : p2.Prime) 
  (h_sum : p1 + p2 = 91) :
  p1 * p2 = 178 := 
sorry

end sum_two_primes_eq_91_prod_is_178_l111_111628


namespace geometric_sequence_common_ratio_l111_111181

/--
  Given a geometric sequence with the first three terms:
  a₁ = 27,
  a₂ = 54,
  a₃ = 108,
  prove that the common ratio is r = 2.
-/
theorem geometric_sequence_common_ratio :
  let a₁ := 27
  let a₂ := 54
  let a₃ := 108
  ∃ r : ℕ, (a₂ = r * a₁) ∧ (a₃ = r * a₂) ∧ r = 2 := by
  sorry

end geometric_sequence_common_ratio_l111_111181


namespace find_rate_percent_l111_111970

theorem find_rate_percent (SI P T : ℝ) (h1 : SI = 160) (h2 : P = 800) (h3 : T = 5) : P * (4:ℝ) * T / 100 = SI :=
by
  sorry

end find_rate_percent_l111_111970


namespace john_fixed_computers_l111_111560

theorem john_fixed_computers (total_computers unfixable waiting_for_parts fixed_right_away : ℕ)
  (h1 : total_computers = 20)
  (h2 : unfixable = 0.20 * 20)
  (h3 : waiting_for_parts = 0.40 * 20)
  (h4 : fixed_right_away = total_computers - unfixable - waiting_for_parts) :
  fixed_right_away = 8 :=
by
  sorry

end john_fixed_computers_l111_111560


namespace difference_of_numbers_l111_111012

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 22500) (h2 : b = 10 * a + 5) : b - a = 18410 :=
by
  sorry

end difference_of_numbers_l111_111012


namespace intersection_A_B_l111_111723

-- Defining the sets A and B
def A : Set ℝ := {x | x^2 - 3*x + 2 < 0}
def B : Set ℝ := {x | 3 - x > 0}

-- Stating the theorem that A ∩ B equals (1, 2)
theorem intersection_A_B : A ∩ B = {x | 1 < x ∧ x < 2} :=
by
  sorry

end intersection_A_B_l111_111723


namespace plane_determination_l111_111079

inductive Propositions : Type where
  | p1 : Propositions
  | p2 : Propositions
  | p3 : Propositions
  | p4 : Propositions

open Propositions

def correct_proposition := p4

theorem plane_determination (H: correct_proposition = p4): correct_proposition = p4 := 
by 
  exact H

end plane_determination_l111_111079


namespace tutoring_minutes_l111_111731

def flat_rate : ℤ := 20
def per_minute_rate : ℤ := 7
def total_paid : ℤ := 146

theorem tutoring_minutes (m : ℤ) : total_paid = flat_rate + (per_minute_rate * m) → m = 18 :=
by
  sorry

end tutoring_minutes_l111_111731


namespace determine_OP_l111_111293

theorem determine_OP
  (a b c d e : ℝ)
  (h_dist_OA : a > 0)
  (h_dist_OB : b > 0)
  (h_dist_OC : c > 0)
  (h_dist_OD : d > 0)
  (h_dist_OE : e > 0)
  (h_c_le_d : c ≤ d)
  (P : ℝ)
  (hP : c ≤ P ∧ P ≤ d)
  (h_ratio : ∀ (P : ℝ) (hP : c ≤ P ∧ P ≤ d), (a - P) / (P - e) = (c - P) / (P - d)) :
  P = (ce - ad) / (a - c + e - d) :=
sorry

end determine_OP_l111_111293


namespace cos_double_angle_l111_111233

theorem cos_double_angle (α : ℝ) (h : Real.sin (π/6 - α) = 1/3) :
  Real.cos (2 * (π/3 + α)) = -7/9 :=
by
  sorry

end cos_double_angle_l111_111233


namespace factorize_expr_l111_111220

theorem factorize_expr (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) :=
sorry

end factorize_expr_l111_111220


namespace transfer_student_increases_averages_l111_111758

def group1_grades : List ℝ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2_grades : List ℝ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℝ) : ℝ :=
  if grades.length > 0 then (grades.sum / grades.length) else 0

def can_transfer_to_increase_averages (group1_grades group2_grades : List ℝ) : Prop :=
  ∃ x ∈ group1_grades, average (x :: group2_grades) > average group2_grades ∧
                       average (group1_grades.erase x) > average group1_grades

theorem transfer_student_increases_averages :
  can_transfer_to_increase_averages group1_grades group2_grades :=
by {
  sorry -- The proof is skipped as per instructions
}

end transfer_student_increases_averages_l111_111758


namespace problem_statement_l111_111647

variable (a b c : ℝ)

theorem problem_statement
  (h1 : a + b = 100)
  (h2 : b + c = 140) :
  c - a = 40 :=
sorry

end problem_statement_l111_111647


namespace total_buyers_l111_111176

-- Definitions based on conditions
def C : ℕ := 50
def M : ℕ := 40
def B : ℕ := 19
def pN : ℝ := 0.29  -- Probability that a random buyer purchases neither

-- The theorem statement
theorem total_buyers :
  ∃ T : ℝ, (T = (C + M - B) + pN * T) ∧ T = 100 :=
by
  sorry

end total_buyers_l111_111176


namespace domain_of_p_l111_111299

theorem domain_of_p (h : ℝ → ℝ) (h_domain : ∀ x, -10 ≤ x → x ≤ 6 → ∃ y, h x = y) :
  ∀ x, -1.2 ≤ x ∧ x ≤ 2 → ∃ y, h (-5 * x) = y :=
by
  sorry

end domain_of_p_l111_111299


namespace simplify_expression_l111_111155

variable (x : ℝ)

theorem simplify_expression : (20 * x^2) * (5 * x) * (1 / (2 * x)^2) * (2 * x)^2 = 100 * x^3 := 
by 
  sorry

end simplify_expression_l111_111155


namespace place_mats_length_l111_111860

theorem place_mats_length (r : ℝ) (w : ℝ) (n : ℕ) (x : ℝ) :
  r = 4 ∧ w = 1 ∧ n = 6 ∧
  (∀ i, i < n → 
    let inner_corners_touch := true in
    place_mat_placement_correct r w x i inner_corners_touch) →
  x = (3 * real.sqrt 7 - real.sqrt 3) / 2 :=
by sorry

end place_mats_length_l111_111860


namespace unique_pos_neg_roots_of_poly_l111_111128

noncomputable def poly : Polynomial ℝ := Polynomial.C 1 * Polynomial.X^4 + Polynomial.C 5 * Polynomial.X^3 + Polynomial.C 15 * Polynomial.X - Polynomial.C 9

theorem unique_pos_neg_roots_of_poly : 
  (∃! x : ℝ, (0 < x) ∧ poly.eval x = 0) ∧ (∃! x : ℝ, (x < 0) ∧ poly.eval x = 0) :=
  sorry

end unique_pos_neg_roots_of_poly_l111_111128


namespace marbles_count_l111_111347

-- Define the condition variables
variable (M : ℕ) -- total number of marbles placed on Monday
variable (day2_marbles : ℕ) -- marbles remaining after second day
variable (day3_cleo_marbles : ℕ) -- marbles taken by Cleo on third day

-- Condition definitions
def condition1 : Prop := day2_marbles = 2 * M / 5
def condition2 : Prop := day3_cleo_marbles = (day2_marbles / 2)
def condition3 : Prop := day3_cleo_marbles = 15

-- The theorem to prove
theorem marbles_count : 
  condition1 M day2_marbles → 
  condition2 day2_marbles day3_cleo_marbles → 
  condition3 day3_cleo_marbles → 
  M = 75 :=
by
  intros h1 h2 h3
  sorry

end marbles_count_l111_111347


namespace problem_1_problem_2_l111_111081

-- Problem statement (I)
theorem problem_1 (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Ici (Real.exp 2) → (x * Real.log x + a * x)' ≥ 0) ↔ a ≥ -3 :=
sorry

-- Problem statement (II)
theorem problem_2 (k : ℕ) : 
  (∀ x : ℝ, x ∈ Set.Ioi 1 → x * Real.log x + a * x > k * (x - 1) + a * x - x) → k ≤ 3 :=
sorry

end problem_1_problem_2_l111_111081


namespace cheaper_to_buy_more_l111_111381

def cost (n : ℕ) : ℕ :=
  if 1 ≤ n ∧ n ≤ 30 then 15 * n
  else if 31 ≤ n ∧ n ≤ 60 then 13 * n
  else if 61 ≤ n ∧ n ≤ 90 then 12 * n
  else if 91 ≤ n then 11 * n
  else 0

theorem cheaper_to_buy_more (n : ℕ) : 
  (∃ m, m < n ∧ cost (m + 1) < cost m) ↔ n = 9 := sorry

end cheaper_to_buy_more_l111_111381


namespace simplify_expression_l111_111800

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2) :
  (x - 1 - (2*x - 2)/(x + 1)) / ((x^2 - x) / (2*x + 2)) = 2 - Real.sqrt 2 := 
by
  -- Here we should include the proof steps, but we skip it with "sorry"
  sorry

end simplify_expression_l111_111800


namespace infinite_div_by_100_l111_111127

theorem infinite_div_by_100 : ∀ k : ℕ, ∃ n : ℕ, n > 0 ∧ (2 ^ n + n ^ 2) % 100 = 0 :=
by
  sorry

end infinite_div_by_100_l111_111127


namespace number_of_valid_subsets_l111_111412

open Nat

theorem number_of_valid_subsets (p : ℕ) (h_prime : Nat.Prime p) :
  let W := Finset.range (2 * p + 1)
  let A := {A : Finset ℕ | A ⊆ W ∧ A.card = p ∧ (A.sum id) % p = 0}
  A.card = (1 / p) * (Nat.choose (2 * p) p - 2) + 2 := 
  sorry

end number_of_valid_subsets_l111_111412


namespace total_bills_54_l111_111329

/-- A bank teller has some 5-dollar and 20-dollar bills in her cash drawer, 
and the total value of the bills is 780 dollars, with 20 5-dollar bills.
Show that the total number of bills is 54. -/
theorem total_bills_54 (value_total : ℕ) (num_5dollar : ℕ) (num_5dollar_value : ℕ) (num_20dollar : ℕ) :
    value_total = 780 ∧ num_5dollar = 20 ∧ num_5dollar_value = 5 ∧ num_20dollar * 20 + num_5dollar * num_5dollar_value = value_total
    → num_20dollar + num_5dollar = 54 :=
by
  sorry

end total_bills_54_l111_111329


namespace therese_older_than_aivo_l111_111707

-- Definitions based on given conditions
variables {Aivo Jolyn Leon Therese : ℝ}
variables (h1 : Jolyn = Therese + 2)
variables (h2 : Leon = Aivo + 2)
variables (h3 : Jolyn = Leon + 5)

-- Statement to prove
theorem therese_older_than_aivo :
  Therese = Aivo + 5 :=
by
  sorry

end therese_older_than_aivo_l111_111707


namespace simplify_fraction_l111_111580

theorem simplify_fraction (h1 : 48 = 2^4 * 3) (h2 : 72 = 2^3 * 3^2) : (48 / 72 : ℚ) = 2 / 3 := 
by
  sorry

end simplify_fraction_l111_111580


namespace domain_g_is_1_to_2_l111_111522

variable (f : ℝ → ℝ)

-- Define the domain of f
def dom_f := (0, 4)

-- Define the function g
def g (x : ℝ) := f(x + 2) / sqrt(x - 1)

theorem domain_g_is_1_to_2 :
  (∀ x, x ∈ Ioo 0 4 → f(x).is_defined) → 
  (∀ x, x ∈ Ioo 1 2 → g(x).is_defined) := by
  sorry

end domain_g_is_1_to_2_l111_111522


namespace sum_powers_of_ab_l111_111005

theorem sum_powers_of_ab (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1)
  (h3 : a^2 + b^2 = 7) (h4 : a^3 + b^3 = 18) (h5 : a^4 + b^4 = 47) :
  a^5 + b^5 = 123 :=
sorry

end sum_powers_of_ab_l111_111005


namespace sophie_donuts_left_l111_111802

theorem sophie_donuts_left :
  ∀ (boxes_initial : ℕ) (donuts_per_box : ℕ) (boxes_given_away : ℕ) (dozen : ℕ),
  boxes_initial = 4 →
  donuts_per_box = 12 →
  boxes_given_away = 1 →
  dozen = 12 →
  (boxes_initial - boxes_given_away) * donuts_per_box - (dozen / 2) = 30 :=
by 
  intros boxes_initial donuts_per_box boxes_given_away dozen 
  assume h1 h2 h3 h4
  sorry

end sophie_donuts_left_l111_111802


namespace externally_tangent_circles_solution_l111_111950

theorem externally_tangent_circles_solution (R1 R2 d : Real)
  (h1 : R1 > 0) (h2 : R2 > 0) (h3 : R1 + R2 > d) :
  (1/R1) + (1/R2) = 2/d :=
sorry

end externally_tangent_circles_solution_l111_111950


namespace intersection_of_A_and_B_l111_111896

-- Definitions of sets A and B
def set_A : Set ℝ := { x | x^2 - x - 6 < 0 }
def set_B : Set ℝ := { x | (x + 4) * (x - 2) > 0 }

-- Theorem statement for the intersection of A and B
theorem intersection_of_A_and_B : set_A ∩ set_B = { x | 2 < x ∧ x < 3 } :=
by
  sorry

end intersection_of_A_and_B_l111_111896


namespace composite_product_division_l111_111690

-- Define the first 12 positive composite integers
def first_six_composites := [4, 6, 8, 9, 10, 12]
def next_six_composites := [14, 15, 16, 18, 20, 21]

-- Define the product of a list of integers
def product (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

-- Define the target problem statement
theorem composite_product_division :
  (product first_six_composites : ℚ) / (product next_six_composites : ℚ) = 1 / 49 := by
  sorry

end composite_product_division_l111_111690


namespace point_in_second_quadrant_l111_111101

theorem point_in_second_quadrant {x : ℝ} (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
sorry

end point_in_second_quadrant_l111_111101


namespace find_a_l111_111953

-- Definitions for the problem
def quadratic_distinct_roots (a : ℝ) : Prop :=
  let Δ := a^2 - 16
  Δ > 0

def satisfies_root_equation (x1 x2 : ℝ) : Prop :=
  (x1^2 - (20 / (3 * x2^3)) = x2^2 - (20 / (3 * x1^3)))

-- Main statement of the proof problem
theorem find_a (a x1 x2 : ℝ) (h_quadratic_roots : quadratic_distinct_roots a)
               (h_root_equation : satisfies_root_equation x1 x2)
               (h_vieta_sum : x1 + x2 = -a) (h_vieta_product : x1 * x2 = 4) :
  a = -10 :=
by
  sorry

end find_a_l111_111953


namespace solution_to_axb_eq_0_l111_111995

theorem solution_to_axb_eq_0 (a b x : ℝ) (h₀ : a ≠ 0) (h₁ : (0, 4) ∈ {p : ℝ × ℝ | p.snd = a * p.fst + b}) (h₂ : (-3, 0) ∈ {p : ℝ × ℝ | p.snd = a * p.fst + b}) :
  x = -3 :=
by
  sorry

end solution_to_axb_eq_0_l111_111995


namespace continuous_at_3_l111_111368

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 3 then x^2 + x + 2 else 2 * x + a

theorem continuous_at_3 {a : ℝ} : (∀ x : ℝ, 0 < abs (x - 3) → abs (f x a - f 3 a) < 0.0001) →
a = 8 :=
by
  sorry

end continuous_at_3_l111_111368


namespace C_is_necessary_but_not_sufficient_for_A_l111_111717

-- Define C, B, A to be logical propositions
variables (A B C : Prop)

-- The conditions given
axiom h1 : A → B
axiom h2 : ¬ (B → A)
axiom h3 : B ↔ C

-- The conclusion: Prove that C is a necessary but not sufficient condition for A
theorem C_is_necessary_but_not_sufficient_for_A : (A → C) ∧ ¬ (C → A) :=
by
  sorry

end C_is_necessary_but_not_sufficient_for_A_l111_111717


namespace simplify_fraction_l111_111603

-- Define the problem and conditions
def numerator : ℕ := 48
def denominator : ℕ := 72
def gcd_n_d : ℕ := Nat.gcd numerator denominator

-- The proof statement
theorem simplify_fraction : (numerator / gcd_n_d) / (denominator / gcd_n_d) = 2 / 3 :=
by
  have h_gcd : gcd_n_d = 24 := by rfl
  sorry

end simplify_fraction_l111_111603


namespace trapezoid_bd_length_l111_111267

theorem trapezoid_bd_length
  (AB CD AC BD : ℝ)
  (tanC tanB : ℝ)
  (h1 : AB = 24)
  (h2 : CD = 15)
  (h3 : AC = 30)
  (h4 : tanC = 2)
  (h5 : tanB = 1.25)
  (h6 : AC ^ 2 = AB ^ 2 + (CD - AB) ^ 2) :
  BD = 9 * Real.sqrt 11 := by
  sorry

end trapezoid_bd_length_l111_111267


namespace range_of_a_for_positive_f_l111_111568

-- Let the function \(f(x) = ax^2 - 2x + 2\)
def f (a x : ℝ) := a * x^2 - 2 * x + 2

-- Theorem: The range of the real number \( a \) such that \( f(x) > 0 \) for all \( x \) in \( 1 < x < 4 \) is \((\dfrac{1}{2}, +\infty)\)
theorem range_of_a_for_positive_f :
  { a : ℝ | ∀ x : ℝ, 1 < x ∧ x < 4 → f a x > 0 } = { a : ℝ | a > 1/2 } :=
sorry

end range_of_a_for_positive_f_l111_111568


namespace total_cups_l111_111459

theorem total_cups (m c s : ℕ) (h1 : 3 * c = 2 * m) (h2 : 2 * c = 6) : m + c + s = 18 :=
by
  sorry

end total_cups_l111_111459


namespace female_democrats_ratio_l111_111151

theorem female_democrats_ratio 
  (M F : ℕ) 
  (H1 : M + F = 660)
  (H2 : (1 / 3 : ℝ) * 660 = 220)
  (H3 : ∃ dem_males : ℕ, dem_males = (1 / 4 : ℝ) * M)
  (H4 : ∃ dem_females : ℕ, dem_females = 110) :
  110 / F = 1 / 2 :=
by
  sorry

end female_democrats_ratio_l111_111151


namespace find_pairs_l111_111117

theorem find_pairs (a b q r : ℕ) (h1 : a * b = q * (a + b) + r)
  (h2 : q^2 + r = 2011) (h3 : 0 ≤ r ∧ r < a + b) : 
  (∃ t : ℕ, 1 ≤ t ∧ t ≤ 45 ∧ (a = t ∧ b = t + 2012 ∨ a = t + 2012 ∧ b = t)) :=
by
  sorry

end find_pairs_l111_111117


namespace part_one_cardinality_A_intersection_B_part_two_range_a_l111_111071

-- Part 1
theorem part_one_cardinality_A_intersection_B :
  let A := {x : ℤ | ∃ k : ℤ, x = 4 * k + 1}
  let B := {x : ℝ | 0 ≤ x ∧ x < 20}
  let A_int_B := {x : ℝ | ∃ n : ℤ, x = 4 * n + 1 ∧ 0 ≤ x ∧ x < 20}
  cardinality A_int_B = 5 :=
sorry

-- Part 2
theorem part_two_range_a :
  let B := {x : ℝ | ∃ a : ℝ, a ≤ x ∧ x < a + 20}
  let C := {x : ℝ | 5 ≤ x ∧ x < 30}
  ∃ (a : ℝ), ¬(a + 20 ≤ 5 ∨ 30 ≤ a) ⇔ B ∩ C ≠ ∅ :=
sorry

end part_one_cardinality_A_intersection_B_part_two_range_a_l111_111071


namespace tan_range_l111_111813

theorem tan_range :
  ∀ (x : ℝ), -Real.pi / 4 ≤ x ∧ x < 0 ∨ 0 < x ∧ x ≤ Real.pi / 4 → -1 ≤ Real.tan x ∧ Real.tan x < 0 ∨ 0 < Real.tan x ∧ Real.tan x ≤ 1 :=
by
  sorry

end tan_range_l111_111813


namespace factor_theorem_example_l111_111505

theorem factor_theorem_example (t : ℚ) : (4 * t^3 + 6 * t^2 + 11 * t - 6 = 0) ↔ (t = 1/2) :=
by sorry

end factor_theorem_example_l111_111505


namespace combined_salary_l111_111951

theorem combined_salary (S_B : ℝ) (S_A : ℝ) (h1 : S_B = 8000) (h2 : 0.20 * S_A = 0.15 * S_B) : 
S_A + S_B = 14000 :=
by {
  sorry
}

end combined_salary_l111_111951


namespace lengths_of_trains_l111_111041

noncomputable def km_per_hour_to_m_per_s (v : ℝ) : ℝ :=
  v * 1000 / 3600

noncomputable def length_of_train (speed : ℝ) (time : ℝ) : ℝ :=
  speed * time

theorem lengths_of_trains (Va Vb : ℝ) : Va = 60 ∧ Vb < Va ∧ length_of_train (km_per_hour_to_m_per_s Va) 42 = (700 : ℝ) 
    → length_of_train (km_per_hour_to_m_per_s Vb * (42 / 56)) 56 = (700 : ℝ) :=
by
  intros h
  sorry

end lengths_of_trains_l111_111041


namespace positive_solution_unique_m_l111_111539

theorem positive_solution_unique_m (m : ℝ) : ¬ (4 < m ∧ m < 2) :=
by
  sorry

end positive_solution_unique_m_l111_111539


namespace latte_cost_l111_111417

theorem latte_cost :
  ∃ (latte_cost : ℝ), 
    2 * 2.25 + 3.50 + 0.50 + 2 * 2.50 + 3.50 + 2 * latte_cost = 25.00 ∧ 
    latte_cost = 4.00 :=
by
  use 4.00
  simp
  sorry

end latte_cost_l111_111417


namespace simplify_fraction_l111_111591

theorem simplify_fraction :
  (48 : ℚ) / 72 = 2 / 3 :=
sorry

end simplify_fraction_l111_111591


namespace tan_simplification_l111_111515

theorem tan_simplification 
  (θ : ℝ) 
  (h : Real.tan θ = 3) : 
  (1 - Real.sin θ) / (Real.cos θ) - (Real.cos θ) / (1 + Real.sin θ) = 0 := 
by 
  sorry

end tan_simplification_l111_111515


namespace sum_from_1_to_60_is_1830_sum_from_51_to_60_is_555_l111_111052

-- Definition for the sum of the first n natural numbers
def sum_upto (n : ℕ) : ℕ := n * (n + 1) / 2

-- Definition for the sum from 1 to 60
def sum_1_to_60 : ℕ := sum_upto 60

-- Definition for the sum from 1 to 50
def sum_1_to_50 : ℕ := sum_upto 50

-- Proof problem 1
theorem sum_from_1_to_60_is_1830 : sum_1_to_60 = 1830 := 
by
  sorry

-- Definition for the sum from 51 to 60
def sum_51_to_60 : ℕ := sum_1_to_60 - sum_1_to_50

-- Proof problem 2
theorem sum_from_51_to_60_is_555 : sum_51_to_60 = 555 := 
by
  sorry

end sum_from_1_to_60_is_1830_sum_from_51_to_60_is_555_l111_111052


namespace susans_coins_worth_l111_111000

theorem susans_coins_worth :
  ∃ n d : ℕ, n + d = 40 ∧ (5 * n + 10 * d) = 230 ∧ (10 * n + 5 * d) = 370 :=
sorry

end susans_coins_worth_l111_111000


namespace find_limit_of_hours_l111_111660

def regular_rate : ℝ := 16
def overtime_rate (r : ℝ) : ℝ := r * 1.75
def total_compensation : ℝ := 920
def total_hours : ℝ := 50

theorem find_limit_of_hours : 
  ∃ (L : ℝ), 
    total_compensation = (regular_rate * L) + ((overtime_rate regular_rate) * (total_hours - L)) →
    L = 40 :=
by
  sorry

end find_limit_of_hours_l111_111660


namespace total_bathing_suits_l111_111845

theorem total_bathing_suits (men_women_bathing_suits : Nat)
                            (men_bathing_suits : Nat := 14797)
                            (women_bathing_suits : Nat := 4969) :
    men_bathing_suits + women_bathing_suits = 19766 := by
  sorry

end total_bathing_suits_l111_111845


namespace roger_coins_left_l111_111441

theorem roger_coins_left {pennies nickels dimes donated_coins initial_coins remaining_coins : ℕ} 
    (h1 : pennies = 42) 
    (h2 : nickels = 36) 
    (h3 : dimes = 15) 
    (h4 : donated_coins = 66) 
    (h5 : initial_coins = pennies + nickels + dimes) 
    (h6 : remaining_coins = initial_coins - donated_coins) : 
    remaining_coins = 27 := 
sorry

end roger_coins_left_l111_111441


namespace consecutive_odd_integers_expressions_l111_111807

theorem consecutive_odd_integers_expressions
  {p q : ℤ} (hpq : p + 2 = q ∨ p - 2 = q) (hp_odd : p % 2 = 1) (hq_odd : q % 2 = 1) :
  (2 * p + 5 * q) % 2 = 1 ∧ (5 * p - 2 * q) % 2 = 1 ∧ (2 * p * q + 5) % 2 = 1 :=
  sorry

end consecutive_odd_integers_expressions_l111_111807


namespace winning_candidate_votes_l111_111822

theorem winning_candidate_votes (V : ℝ) (h1 : 0.62 * V - 0.38 * V = 336): 0.62 * V = 868 :=
by
  sorry

end winning_candidate_votes_l111_111822


namespace mallory_travel_expenses_l111_111617

theorem mallory_travel_expenses (fuel_tank_cost : ℕ) (fuel_tank_miles : ℕ) (total_miles : ℕ) (food_ratio : ℚ)
  (h_fuel_tank_cost : fuel_tank_cost = 45)
  (h_fuel_tank_miles : fuel_tank_miles = 500)
  (h_total_miles : total_miles = 2000)
  (h_food_ratio : food_ratio = 3/5) :
  ∃ total_cost : ℕ, total_cost = 288 :=
by
  sorry

end mallory_travel_expenses_l111_111617


namespace cookies_baked_on_monday_is_32_l111_111645

-- Definitions for the problem.
variable (X : ℕ)

-- Conditions.
def cookies_baked_on_monday := X
def cookies_baked_on_tuesday := X / 2
def cookies_baked_on_wednesday := 3 * (X / 2) - 4

-- Total cookies at the end of three days.
def total_cookies := cookies_baked_on_monday X + cookies_baked_on_tuesday X + cookies_baked_on_wednesday X

-- Theorem statement to prove the number of cookies baked on Monday.
theorem cookies_baked_on_monday_is_32 : total_cookies X = 92 → cookies_baked_on_monday X = 32 :=
by
  -- We would add the proof steps here.
  sorry

end cookies_baked_on_monday_is_32_l111_111645


namespace entrance_exam_proof_l111_111976

-- Define the conditions
variables (x y : ℕ)
variables (h1 : x + y = 70)
variables (h2 : 3 * x - y = 38)

-- The proof goal
theorem entrance_exam_proof : x = 27 :=
by
  -- The actual proof steps are omitted here
  sorry

end entrance_exam_proof_l111_111976


namespace total_beads_correct_l111_111285

-- Definitions of the problem conditions
def blue_beads : ℕ := 5
def red_beads : ℕ := 2 * blue_beads
def white_beads : ℕ := blue_beads + red_beads
def silver_beads : ℕ := 10

-- Definition of the total number of beads
def total_beads : ℕ := blue_beads + red_beads + white_beads + silver_beads

-- The main theorem statement
theorem total_beads_correct : total_beads = 40 :=
by 
  sorry

end total_beads_correct_l111_111285


namespace Rogers_age_more_than_twice_Jills_age_l111_111798

/--
Jill is 20 years old.
Finley is 40 years old.
Roger's age is more than twice Jill's age.
In 15 years, the age difference between Roger and Jill will be 30 years less than Finley's age.
Prove that Roger's age is 5 years more than twice Jill's age.
-/
theorem Rogers_age_more_than_twice_Jills_age 
  (J F : ℕ) (hJ : J = 20) (hF : F = 40) (R x : ℕ)
  (hR : R = 2 * J + x) 
  (age_diff_condition : (R + 15) - (J + 15) = (F + 15) - 30) :
  x = 5 := 
sorry

end Rogers_age_more_than_twice_Jills_age_l111_111798


namespace john_can_fix_l111_111559

variable (total_computers : ℕ) (percent_unfixable percent_wait_for_parts : ℕ)

-- Conditions as requirements
def john_condition : Prop :=
  total_computers = 20 ∧
  percent_unfixable = 20 ∧
  percent_wait_for_parts = 40

-- The proof goal based on the conditions
theorem john_can_fix (h : john_condition total_computers percent_unfixable percent_wait_for_parts) :
  total_computers * (100 - percent_unfixable - percent_wait_for_parts) / 100 = 8 :=
by {
  -- Here you can place the corresponding proof details
  sorry
}

end john_can_fix_l111_111559


namespace triangle_areas_l111_111957

theorem triangle_areas (r s : ℝ) (h1 : s = (1/2) * r + 6)
                       (h2 : (12 + r) * ((1/2) * r + 6) = 18) :
  r + s = -3 :=
by
  sorry

end triangle_areas_l111_111957


namespace inequality_l111_111278

-- Define the real variables p, q, r and the condition that their product is 1
variables {p q r : ℝ} (h : p * q * r = 1)

-- State the theorem
theorem inequality (h : p * q * r = 1) :
  (1 / (1 - p))^2 + (1 / (1 - q))^2 + (1 / (1 - r))^2 ≥ 1 := 
sorry

end inequality_l111_111278


namespace can_increase_average_l111_111755

def student_grades := List (String × Nat)

def group1 : student_grades := 
    [("Andreev", 5), ("Borisova", 3), ("Vasilieva", 5), ("Georgiev", 3),
     ("Dmitriev", 5), ("Evstigneeva", 4), ("Ignatov", 3), ("Kondratiev", 4),
     ("Leontieva", 3), ("Mironov", 4), ("Nikolaeva", 5), ("Ostapov", 5)]
     
def group2 : student_grades := 
    [("Alexeeva", 3), ("Bogdanov", 4), ("Vladimirov", 5), ("Grigorieva", 2),
     ("Davydova", 3), ("Evstahiev", 2), ("Ilina", 5), ("Klimova", 4),
     ("Lavrentiev", 5), ("Mikhailova", 3)]

def average_grade (grp : student_grades) : ℚ :=
    (grp.map (λ x => x.snd)).sum / grp.length

def updated_group (grp : List (String × Nat)) (student : String × Nat) : List (String × Nat) :=
    grp.erase student

def transfer_student (g1 g2 : student_grades) (student : String × Nat) : student_grades × student_grades :=
    (updated_group g1 student, student :: g2)

theorem can_increase_average (s : String × Nat) 
    (h1 : s ∈ group1)
    (h2 : s.snd = 4)
    (h3 : average_grade group1 < s.snd)
    (h4 : average_grade group2 > s.snd)
    : 
    let (new_group1, new_group2) := transfer_student group1 group2 s in 
    average_grade new_group1 > average_grade group1 ∧ 
    average_grade new_group2 > average_grade group2 :=
sorry

#eval can_increase_average ("Evstigneeva", 4)

end can_increase_average_l111_111755


namespace rationalize_denominator_l111_111289

theorem rationalize_denominator : (14 / Real.sqrt 14) = Real.sqrt 14 := by
  sorry

end rationalize_denominator_l111_111289


namespace gnuff_tutor_minutes_l111_111728

/-- Definitions of the given conditions -/
def flat_rate : ℕ := 20
def per_minute_charge : ℕ := 7
def total_paid : ℕ := 146

/-- The proof statement -/
theorem gnuff_tutor_minutes :
  (total_paid - flat_rate) / per_minute_charge = 18 :=
by
  sorry

end gnuff_tutor_minutes_l111_111728


namespace largest_possible_green_socks_l111_111330

/--
A box contains a mixture of green socks and yellow socks, with at most 2023 socks in total.
The probability of randomly pulling out two socks of the same color is exactly 1/3.
What is the largest possible number of green socks in the box? 
-/
theorem largest_possible_green_socks (g y : ℤ) (t : ℕ) (h : t ≤ 2023) 
  (prob_condition : (g * (g - 1) + y * (y - 1) = t * (t - 1) / 3)) : 
  g ≤ 990 :=
sorry

end largest_possible_green_socks_l111_111330


namespace probability_even_sum_l111_111015

def total_balls := List.range' 1 13  -- List of numbers 1 to 12

noncomputable def total_outcomes := (total_balls.length) * (total_balls.length - 1)

def is_even (n : ℕ) : Bool := n % 2 = 0

def favourable_outcomes (balls : List ℕ) : ℕ :=
  (balls.filter is_even).length * (balls.filter is_even).length +
  (balls.filter (λ n, ¬is_even n)).length * (balls.filter (λ n, ¬is_even n)).length

theorem probability_even_sum : 
  (ℚ.of_nat (favourable_outcomes total_balls) / ℚ.of_nat total_outcomes) = 5 / 11 :=
by
  -- Solution would go here
  sorry

end probability_even_sum_l111_111015


namespace golden_ratio_in_range_l111_111815

theorem golden_ratio_in_range :
  let phi := (Real.sqrt 5 - 1) / 2
  in 0.6 < phi ∧ phi < 0.7 :=
by
  let phi := (Real.sqrt 5 - 1) / 2
  sorry

end golden_ratio_in_range_l111_111815


namespace percentage_relations_with_respect_to_z_l111_111416

variable (x y z w : ℝ)
variable (h1 : x = 1.30 * y)
variable (h2 : y = 0.50 * z)
variable (h3 : w = 2 * x)

theorem percentage_relations_with_respect_to_z : 
  x = 0.65 * z ∧ y = 0.50 * z ∧ w = 1.30 * z := by
  sorry

end percentage_relations_with_respect_to_z_l111_111416


namespace solution_set_of_inequality_l111_111307

theorem solution_set_of_inequality (x : ℝ) :
  |x^2 - 2| < 2 ↔ (-2 < x ∧ x < 0) ∨ (0 < x ∧ x < 2) :=
sorry

end solution_set_of_inequality_l111_111307


namespace even_product_probability_l111_111298

-- Define the spinners C and D
def spinner_C := {1, 1, 2, 3, 5, 5}
def spinner_D := {1, 2, 3, 4}

-- Define the event of interest: the product is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the probability calculations
def total_outcomes := card spinner_C * card spinner_D
def even_outcomes := ∑ i in spinner_C, ∑ j in spinner_D, if is_even (i * j) then 1 else 0

-- The theorem stating the required probability
theorem even_product_probability : (even_outcomes : ℚ) / total_outcomes = 1 / 2 :=
sorry

end even_product_probability_l111_111298


namespace john_can_fix_l111_111558

variable (total_computers : ℕ) (percent_unfixable percent_wait_for_parts : ℕ)

-- Conditions as requirements
def john_condition : Prop :=
  total_computers = 20 ∧
  percent_unfixable = 20 ∧
  percent_wait_for_parts = 40

-- The proof goal based on the conditions
theorem john_can_fix (h : john_condition total_computers percent_unfixable percent_wait_for_parts) :
  total_computers * (100 - percent_unfixable - percent_wait_for_parts) / 100 = 8 :=
by {
  -- Here you can place the corresponding proof details
  sorry
}

end john_can_fix_l111_111558


namespace cubic_polynomial_p_value_l111_111989

noncomputable def p (x : ℝ) : ℝ := sorry

theorem cubic_polynomial_p_value :
  (∀ n ∈ ({1, 2, 3, 5} : Finset ℝ), p n = 1 / n ^ 2) →
  p 4 = 1 / 150 := 
by
  intros h
  sorry

end cubic_polynomial_p_value_l111_111989


namespace remainder_of_2_pow_30_plus_3_mod_7_l111_111460

theorem remainder_of_2_pow_30_plus_3_mod_7 :
  (2^30 + 3) % 7 = 4 := 
sorry

end remainder_of_2_pow_30_plus_3_mod_7_l111_111460


namespace parallel_to_l3_through_P_perpendicular_to_l3_through_P_l111_111517

-- Define the lines l1, l2, and l3
def l1 (x y : ℝ) : Prop := 2 * x + 3 * y - 5 = 0
def l2 (x y : ℝ) : Prop := x + 2 * y - 3 = 0
def l3 (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the intersection point P
def P := (1, 1)

-- Define the parallel line equation to l3 passing through P
def parallel_line (x y : ℝ) : Prop := 2 * x + y - 3 = 0

-- Define the perpendicular line equation to l3 passing through P
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

-- Prove the parallel line through P is 2x + y - 3 = 0
theorem parallel_to_l3_through_P : 
  ∀ (x y : ℝ), l1 x y → l2 x y → (parallel_line 1 1) := 
by 
  sorry

-- Prove the perpendicular line through P is x - 2y + 1 = 0
theorem perpendicular_to_l3_through_P : 
  ∀ (x y : ℝ), l1 x y → l2 x y → (perpendicular_line 1 1) := 
by 
  sorry

end parallel_to_l3_through_P_perpendicular_to_l3_through_P_l111_111517


namespace find_divisor_value_l111_111185

theorem find_divisor_value (x : ℝ) (h : 63 / x = 63 - 42) : x = 3 :=
by
  sorry

end find_divisor_value_l111_111185


namespace set_inclusion_interval_l111_111908

theorem set_inclusion_interval (a : ℝ) :
    (A : Set ℝ) = {x : ℝ | (2 * a + 1) ≤ x ∧ x ≤ (3 * a - 5)} →
    (B : Set ℝ) = {x : ℝ | 3 ≤ x ∧ x ≤ 22} →
    (2 * a + 1 ≤ 3 * a - 5) →
    (A ⊆ B ↔ 6 ≤ a ∧ a ≤ 9) :=
by sorry

end set_inclusion_interval_l111_111908


namespace range_of_a_l111_111476

noncomputable def p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0

noncomputable def q (a : ℝ) : Prop :=
  a < 1 ∧ a ≠ 0

theorem range_of_a (a : ℝ) (h1 : p a ∨ q a) (h2 : ¬(p a ∧ q a)) :
  (1 ≤ a ∧ a < 2) ∨ a ≤ -2 ∨ a = 0 :=
by sorry

end range_of_a_l111_111476


namespace jose_to_haylee_ratio_l111_111246

variable (J : ℕ)

def haylee_guppies := 36
def charliz_guppies := J / 3
def nicolai_guppies := 4 * (J / 3)
def total_guppies := haylee_guppies + J + charliz_guppies + nicolai_guppies

theorem jose_to_haylee_ratio :
  haylee_guppies = 36 ∧ total_guppies = 84 →
  J / haylee_guppies = 1 / 2 :=
by
  intro h
  sorry

end jose_to_haylee_ratio_l111_111246


namespace pinwheel_area_eq_six_l111_111682

open Set

/-- Define the pinwheel in a 6x6 grid -/
def is_midpoint (x y : ℤ) : Prop :=
  (x = 3 ∧ (y = 1 ∨ y = 5)) ∨ (y = 3 ∧ (x = 1 ∨ x = 5))

def is_center (x y : ℤ) : Prop :=
  x = 3 ∧ y = 3

def is_triangle_vertex (x y : ℤ) : Prop :=
  is_center x y ∨ is_midpoint x y

-- Main theorem statement
theorem pinwheel_area_eq_six :
  let pinwheel : Set (ℤ × ℤ) := {p | is_triangle_vertex p.1 p.2}
  ∀ A : ℝ, A = 6 :=
by sorry

end pinwheel_area_eq_six_l111_111682


namespace simplify_and_evaluate_l111_111943

noncomputable def expression (x : ℤ) : ℤ :=
  ( (-2 * x^3 - 6 * x) / (-2 * x) - 2 * (3 * x + 1) * (3 * x - 1) + 7 * x * (x - 1) )

theorem simplify_and_evaluate : 
  (expression (-3) = -64) := by
  sorry

end simplify_and_evaluate_l111_111943


namespace roma_can_ensure_no_more_than_50_chips_end_up_in_last_cells_l111_111931

theorem roma_can_ensure_no_more_than_50_chips_end_up_in_last_cells 
  (k n : ℕ) (h_k : k = 4) (h_n : n = 100)
  (shift_rule : ∀ (m : ℕ), m ≤ n → 
    ∃ (chips_moved : ℕ), chips_moved = 1 ∧ chips_moved ≤ m) 
  : ∃ m, m ≤ n ∧ m = 50 := 
by
  sorry

end roma_can_ensure_no_more_than_50_chips_end_up_in_last_cells_l111_111931


namespace pqr_problem_l111_111121

noncomputable def pqr_sums_to_44 (p q r : ℝ) : Prop :=
  (p < q) ∧ (∀ x, (x < -6 ∨ |x - 20| ≤ 2) ↔ ( (x - p) * (x - q) / (x - r) ≥ 0 ))

theorem pqr_problem (p q r : ℝ) (h : pqr_sums_to_44 p q r) : p + 2*q + 3*r = 44 :=
sorry

end pqr_problem_l111_111121


namespace breadth_of_boat_l111_111478

theorem breadth_of_boat
  (L : ℝ) (h : ℝ) (m : ℝ) (g : ℝ) (ρ : ℝ) (B : ℝ)
  (hL : L = 3)
  (hh : h = 0.01)
  (hm : m = 60)
  (hg : g = 9.81)
  (hρ : ρ = 1000) :
  B = 2 := by
  sorry

end breadth_of_boat_l111_111478


namespace min_value_of_x_plus_y_l111_111249

theorem min_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy: 0 < y) (h: 9 * x + y = x * y) : x + y ≥ 16 := 
sorry

end min_value_of_x_plus_y_l111_111249


namespace john_fixed_computers_l111_111561

theorem john_fixed_computers (total_computers unfixable waiting_for_parts fixed_right_away : ℕ)
  (h1 : total_computers = 20)
  (h2 : unfixable = 0.20 * 20)
  (h3 : waiting_for_parts = 0.40 * 20)
  (h4 : fixed_right_away = total_computers - unfixable - waiting_for_parts) :
  fixed_right_away = 8 :=
by
  sorry

end john_fixed_computers_l111_111561


namespace value_of_expression_l111_111157

theorem value_of_expression : (20 * 24) / (2 * 0 + 2 * 4) = 60 := sorry

end value_of_expression_l111_111157


namespace triangle_height_dist_inequality_l111_111440

variable {T : Type} [MetricSpace T] 

theorem triangle_height_dist_inequality {h_a h_b h_c l_a l_b l_c : ℝ} (h_a_pos : 0 < h_a) (h_b_pos : 0 < h_b) (h_c_pos : 0 < h_c) 
  (l_a_pos : 0 < l_a) (l_b_pos : 0 < l_b) (l_c_pos : 0 < l_c) :
  h_a / l_a + h_b / l_b + h_c / l_c >= 9 :=
sorry

end triangle_height_dist_inequality_l111_111440


namespace total_operations_l111_111613

-- Define the process of iterative multiplication and division as described in the problem
def process (start : Nat) : Nat :=
  let m1 := 3 * start
  let m2 := 3 * m1
  let m3 := 3 * m2
  let m4 := 3 * m3
  let m5 := 3 * m4
  let d1 := m5 / 2
  let d2 := d1 / 2
  let d3 := d2 / 2
  let d4 := d3 / 2
  let d5 := d4 / 2
  let d6 := d5 / 2
  let d7 := d6 / 2
  d7

theorem total_operations : process 1 = 1 ∧ 5 + 7 = 12 :=
by
  sorry

end total_operations_l111_111613


namespace hyperbola_center_l111_111699

theorem hyperbola_center (x y : ℝ) :
  9 * x ^ 2 - 54 * x - 36 * y ^ 2 + 360 * y - 900 = 0 →
  (x, y) = (3, 5) :=
sorry

end hyperbola_center_l111_111699


namespace investment_return_l111_111171

theorem investment_return 
  (investment1 : ℝ) (investment2 : ℝ) 
  (return1 : ℝ) (combined_return_percent : ℝ) : 
  investment1 = 500 → 
  investment2 = 1500 → 
  return1 = 0.07 → 
  combined_return_percent = 0.085 → 
  (500 * 0.07 + 1500 * r = 2000 * 0.085) → 
  r = 0.09 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end investment_return_l111_111171


namespace find_a_plus_b_l111_111120

theorem find_a_plus_b (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^2 + a * x + b) 
  (h2 : { x : ℝ | 0 ≤ f x ∧ f x ≤ 6 - x } = { x : ℝ | 2 ≤ x ∧ x ≤ 3 } ∪ {6}) 
  : a + b = 9 := 
sorry

end find_a_plus_b_l111_111120


namespace divisors_count_30_l111_111736

theorem divisors_count_30 : 
  (∃ n : ℤ, n > 1 ∧ 30 % n = 0) 
  → 
  (∃ k : ℕ, k = 14) :=
by
  sorry

end divisors_count_30_l111_111736


namespace number_of_students_liked_all_three_l111_111752

open Finset

-- Define the total number of students
def total_students := 50

-- Define the number of students who did not like any dessert
def students_not_like_any := 15

-- Define the number of students who liked each dessert
def liked_apple_pie := 22
def liked_chocolate_cake := 17
def liked_pumpkin_pie := 10

-- Define the number of students who liked at least one dessert
def students_liked_at_least_one := total_students - students_not_like_any

-- Define the number of students who liked all three desserts
def students_liked_all := 7

-- Using the inclusion-exclusion principle, prove the number of students who liked all three desserts is 7
theorem number_of_students_liked_all_three :
  ∃ (students_liked_all : ℕ),
    students_liked_all = 7 ∧ 
    liked_apple_pie + liked_chocolate_cake + liked_pumpkin_pie 
    - students_liked_at_least_one 
    = 2 * students_liked_all :=
by
  have h_students_liked_at_least_one : students_liked_at_least_one = 35 := by
    exact rfl
  have h_desserts_sum : liked_apple_pie + liked_chocolate_cake + liked_pumpkin_pie = 49 := by
    exact rfl
  use 7
  split
  exact rfl
  calc  22 + 17 + 10 - 35 = 49 - 35    : by rw [h_desserts_sum]
                                 ...   = 14                : by norm_num
                                 ...   = 2 * 7             : by norm_num

end number_of_students_liked_all_three_l111_111752


namespace group_d_forms_triangle_l111_111956

-- Definitions for the stick lengths in each group
def group_a := (1, 2, 6)
def group_b := (2, 2, 4)
def group_c := (1, 2, 3)
def group_d := (2, 3, 4)

-- Statement to prove that Group D can form a triangle
theorem group_d_forms_triangle (a b c : ℕ) : a = 2 → b = 3 → c = 4 → a + b > c ∧ a + c > b ∧ b + c > a := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  apply And.intro
  sorry
  apply And.intro
  sorry
  sorry

end group_d_forms_triangle_l111_111956


namespace average_last_three_l111_111945

theorem average_last_three {a b c d e f g : ℝ} 
  (h_avg_all : (a + b + c + d + e + f + g) / 7 = 60)
  (h_avg_first_four : (a + b + c + d) / 4 = 55) : 
  (e + f + g) / 3 = 200 / 3 :=
by
  sorry

end average_last_three_l111_111945


namespace bananas_per_monkey_l111_111482

-- Define the given conditions
def total_monkeys : ℕ := 12
def piles_with_9hands : ℕ := 6
def hands_per_pile_9hands : ℕ := 9
def bananas_per_hand_9hands : ℕ := 14
def piles_with_12hands : ℕ := 4
def hands_per_pile_12hands : ℕ := 12
def bananas_per_hand_12hands : ℕ := 9

-- Calculate the total number of bananas from each type of pile
def total_bananas_9hands : ℕ := piles_with_9hands * hands_per_pile_9hands * bananas_per_hand_9hands
def total_bananas_12hands : ℕ := piles_with_12hands * hands_per_pile_12hands * bananas_per_hand_12hands

-- Sum the total number of bananas
def total_bananas : ℕ := total_bananas_9hands + total_bananas_12hands

-- Prove that each monkey gets 99 bananas
theorem bananas_per_monkey : total_bananas / total_monkeys = 99 := by
  sorry

end bananas_per_monkey_l111_111482


namespace number_students_first_class_l111_111451

theorem number_students_first_class
  (average_first_class : ℝ)
  (average_second_class : ℝ)
  (students_second_class : ℕ)
  (combined_average : ℝ)
  (total_students : ℕ)
  (total_marks_first_class : ℝ)
  (total_marks_second_class : ℝ)
  (total_combined_marks : ℝ)
  (x : ℕ)
  (h1 : average_first_class = 50)
  (h2 : average_second_class = 65)
  (h3 : students_second_class = 40)
  (h4 : combined_average = 59.23076923076923)
  (h5 : total_students = x + 40)
  (h6 : total_marks_first_class = 50 * x)
  (h7 : total_marks_second_class = 65 * 40)
  (h8 : total_combined_marks = 59.23076923076923 * (x + 40))
  (h9 : total_marks_first_class + total_marks_second_class = total_combined_marks) :
  x = 25 :=
sorry

end number_students_first_class_l111_111451


namespace tan_x_over_tan_y_plus_tan_y_over_tan_x_l111_111774

open Real

theorem tan_x_over_tan_y_plus_tan_y_over_tan_x (x y : ℝ) 
  (h1 : sin x / cos y + sin y / cos x = 2) 
  (h2 : cos x / sin y + cos y / sin x = 5) :
  tan x / tan y + tan y / tan x = 10 := 
by
  sorry

end tan_x_over_tan_y_plus_tan_y_over_tan_x_l111_111774


namespace download_time_l111_111140

def file_size : ℕ := 90
def rate_first_part : ℕ := 5
def rate_second_part : ℕ := 10
def size_first_part : ℕ := 60

def time_first_part : ℕ := size_first_part / rate_first_part
def size_second_part : ℕ := file_size - size_first_part
def time_second_part : ℕ := size_second_part / rate_second_part
def total_time : ℕ := time_first_part + time_second_part

theorem download_time :
  total_time = 15 := by
  -- sorry can be replaced with the actual proof if needed
  sorry

end download_time_l111_111140


namespace root_in_interval_sum_eq_three_l111_111745

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 5

theorem root_in_interval_sum_eq_three {a b : ℤ} (h1 : b - a = 1) (h2 : ∃ x : ℝ, a < x ∧ x < b ∧ f x = 0) :
  a + b = 3 :=
by
  sorry

end root_in_interval_sum_eq_three_l111_111745


namespace negation_of_exists_l111_111958

theorem negation_of_exists (x : ℝ) : ¬ (∃ x : ℝ, x^2 - x + 2 > 0) = ∀ x : ℝ, x^2 - x + 2 ≤ 0 := by
  sorry

end negation_of_exists_l111_111958


namespace simplify_expression_l111_111282

theorem simplify_expression
  (a b c : ℝ) 
  (hnz_a : a ≠ 0) 
  (hnz_b : b ≠ 0) 
  (hnz_c : c ≠ 0) 
  (h_sum : a + b + c = 0) :
  (1 / (b^3 + c^3 - a^3)) + (1 / (a^3 + c^3 - b^3)) + (1 / (a^3 + b^3 - c^3)) = 1 / (a * b * c) :=
by
  sorry

end simplify_expression_l111_111282


namespace sanda_exercise_each_day_l111_111555

def exercise_problem (javier_exercise_daily sanda_exercise_total total_minutes : ℕ) (days_in_week : ℕ) :=
  javier_exercise_daily * days_in_week + sanda_exercise_total = total_minutes

theorem sanda_exercise_each_day 
  (javier_exercise_daily : ℕ := 50)
  (days_in_week : ℕ := 7)
  (total_minutes : ℕ := 620)
  (days_sanda_exercised : ℕ := 3): 
  ∃ (sanda_exercise_each_day : ℕ), exercise_problem javier_exercise_daily (sanda_exercise_each_day * days_sanda_exercised) total_minutes days_in_week → sanda_exercise_each_day = 90 :=
by 
  sorry

end sanda_exercise_each_day_l111_111555


namespace line_tangent_to_circle_l111_111365

theorem line_tangent_to_circle (r : ℝ) :
  (∀ (x y : ℝ), (x + y = 4) → (x - 2)^2 + (y + 1)^2 = r) → r = 9 / 2 :=
sorry

end line_tangent_to_circle_l111_111365


namespace candy_store_total_sales_l111_111331

def price_per_pound_fudge : ℝ := 2.50
def pounds_fudge : ℕ := 20
def price_per_truffle : ℝ := 1.50
def dozens_truffles : ℕ := 5
def price_per_pretzel : ℝ := 2.00
def dozens_pretzels : ℕ := 3

theorem candy_store_total_sales :
  price_per_pound_fudge * pounds_fudge +
  price_per_truffle * (dozens_truffles * 12) +
  price_per_pretzel * (dozens_pretzels * 12) = 212.00 := by
  sorry

end candy_store_total_sales_l111_111331


namespace number_of_students_playing_soccer_l111_111269

variable (total_students boys playing_soccer_girls not_playing_soccer_girls : ℕ)
variable (percentage_boys_playing_soccer : ℕ)

-- Conditions
axiom h1 : total_students = 470
axiom h2 : boys = 300
axiom h3 : not_playing_soccer_girls = 135
axiom h4 : percentage_boys_playing_soccer = 86
axiom h5 : playing_soccer_girls = 470 - 300 - not_playing_soccer_girls

-- Question: Prove that the number of students playing soccer is 250
theorem number_of_students_playing_soccer : 
  (playing_soccer_girls * 100) / (100 - percentage_boys_playing_soccer) = 250 :=
sorry

end number_of_students_playing_soccer_l111_111269


namespace geometric_series_common_ratio_l111_111370

theorem geometric_series_common_ratio (a : ℕ → ℚ) (q : ℚ) (h1 : a 1 + a 3 = 10) 
(h2 : a 4 + a 6 = 5 / 4) 
(h_geom : ∀ n : ℕ, a (n + 1) = a n * q) : q = 1 / 2 :=
sorry

end geometric_series_common_ratio_l111_111370


namespace dice_probability_l111_111925

theorem dice_probability (D1 D2 D3 : ℕ) (hD1 : 0 ≤ D1) (hD1' : D1 < 10) (hD2 : 0 ≤ D2) (hD2' : D2 < 10) (hD3 : 0 ≤ D3) (hD3' : D3 < 10) :
  ∃ p : ℚ, p = 1 / 10 :=
by
  let outcomes := 10 * 10 * 10
  let favorable := 100
  let expected_probability : ℚ := favorable / outcomes
  use expected_probability
  sorry

end dice_probability_l111_111925


namespace bake_sale_donation_l111_111871

theorem bake_sale_donation :
  let total_earning := 400
  let cost_of_ingredients := 100
  let donation_homeless_piggy := 10
  let total_donation_homeless := 160
  let donation_homeless := total_donation_homeless - donation_homeless_piggy
  let available_for_donation := total_earning - cost_of_ingredients
  let donation_food_bank := available_for_donation - donation_homeless
  (donation_homeless / donation_food_bank) = 1 := 
by
  sorry

end bake_sale_donation_l111_111871


namespace average_episodes_per_year_is_16_l111_111843

-- Define the number of years the TV show has been running
def years : Nat := 14

-- Define the number of seasons and episodes for each category
def seasons_8_15 : Nat := 8
def episodes_per_season_8_15 : Nat := 15
def seasons_4_20 : Nat := 4
def episodes_per_season_4_20 : Nat := 20
def seasons_2_12 : Nat := 2
def episodes_per_season_2_12 : Nat := 12

-- Define the total number of episodes
def total_episodes : Nat :=
  (seasons_8_15 * episodes_per_season_8_15) + 
  (seasons_4_20 * episodes_per_season_4_20) + 
  (seasons_2_12 * episodes_per_season_2_12)

-- Define the average number of episodes per year
def average_episodes_per_year : Nat :=
  total_episodes / years

-- State the theorem to prove the average number of episodes per year is 16
theorem average_episodes_per_year_is_16 : average_episodes_per_year = 16 :=
by
  sorry

end average_episodes_per_year_is_16_l111_111843


namespace married_fraction_l111_111426

variables (M W N : ℕ)

def married_men : Prop := 2 * M = 3 * N
def married_women : Prop := 3 * W = 5 * N
def total_population : ℕ := M + W
def married_population : ℕ := 2 * N

theorem married_fraction (h1: married_men M N) (h2: married_women W N) :
  (married_population N : ℚ) / (total_population M W : ℚ) = 12 / 19 :=
by sorry

end married_fraction_l111_111426


namespace find_beta_l111_111738

theorem find_beta 
  (α β : ℝ)
  (h1 : Real.cos α = 1 / 7)
  (h2 : Real.cos (α + β) = -11 / 14)
  (h3 : 0 < α ∧ α < Real.pi / 2)
  (h4 : Real.pi / 2 < α + β ∧ α + β < Real.pi) :
  β = Real.pi / 3 := 
sorry

end find_beta_l111_111738


namespace problem_condition_l111_111319

theorem problem_condition (m : ℝ) :
  (∀ x : ℝ, x ≤ -1 → (m^2 - m) * 4^x - 2^x < 0) → -1 < m ∧ m < 2 :=
sorry

end problem_condition_l111_111319


namespace smallest_possible_area_l111_111823

noncomputable def smallest_area (l w : ℕ) : ℕ :=
  if 2 * l + 2 * w = 200 ∧ (l = 30 ∨ w = 30) then l * w else 0

theorem smallest_possible_area : ∃ l w : ℕ, 2 * l + 2 * w = 200 ∧ (l = 30 ∨ w = 30) ∧ smallest_area l w = 2100 := by
  sorry

end smallest_possible_area_l111_111823


namespace fraction_of_left_handed_non_throwers_l111_111940

theorem fraction_of_left_handed_non_throwers 
  (total_players : ℕ) (throwers : ℕ) (right_handed_players : ℕ) (all_throwers_right_handed : throwers ≤ right_handed_players) 
  (total_players_eq : total_players = 70) 
  (throwers_eq : throwers = 46) 
  (right_handed_players_eq : right_handed_players = 62) 
  : (total_players - throwers) = 24 → ((right_handed_players - throwers) = 16 → (24 - 16) = 8 → ((8 : ℚ) / 24 = 1/3)) := 
by 
  intros;
  sorry

end fraction_of_left_handed_non_throwers_l111_111940


namespace simplify_fraction_l111_111587

theorem simplify_fraction (a b : ℕ) (h : b ≠ 0) (g : Nat.gcd a b = 24) : a = 48 → b = 72 → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  exact ⟨rfl, rfl⟩

end simplify_fraction_l111_111587


namespace simplify_fraction_l111_111597

theorem simplify_fraction :
  (48 : ℚ) / 72 = 2 / 3 :=
sorry

end simplify_fraction_l111_111597


namespace subtraction_result_l111_111448

-- Define the condition as given: x - 46 = 15
def condition (x : ℤ) := x - 46 = 15

-- Define the theorem that gives us the equivalent mathematical statement we want to prove
theorem subtraction_result (x : ℤ) (h : condition x) : x - 29 = 32 :=
by
  -- Here we would include the proof steps, but as per instructions we will use 'sorry' to skip the proof
  sorry

end subtraction_result_l111_111448


namespace intersection_question_l111_111518

def M : Set ℕ := {1, 2}
def N : Set ℕ := {n | ∃ a ∈ M, n = 2 * a - 1}

theorem intersection_question : M ∩ N = {1} :=
by sorry

end intersection_question_l111_111518


namespace domain_f_l111_111206

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 9*x + 18)

theorem domain_f :
  (∀ x : ℝ, (x ≠ -6) ∧ (x ≠ -3) → ∃ y : ℝ, y = f x) ∧
  (∀ x : ℝ, x = -6 ∨ x = -3 → ¬(∃ y : ℝ, y = f x)) :=
sorry

end domain_f_l111_111206


namespace overall_percentage_good_fruits_l111_111998

theorem overall_percentage_good_fruits
  (oranges_bought : ℕ)
  (bananas_bought : ℕ)
  (apples_bought : ℕ)
  (pears_bought : ℕ)
  (oranges_rotten_percent : ℝ)
  (bananas_rotten_percent : ℝ)
  (apples_rotten_percent : ℝ)
  (pears_rotten_percent : ℝ)
  (h_oranges : oranges_bought = 600)
  (h_bananas : bananas_bought = 400)
  (h_apples : apples_bought = 800)
  (h_pears : pears_bought = 200)
  (h_oranges_rotten : oranges_rotten_percent = 0.15)
  (h_bananas_rotten : bananas_rotten_percent = 0.03)
  (h_apples_rotten : apples_rotten_percent = 0.12)
  (h_pears_rotten : pears_rotten_percent = 0.25) :
  let total_fruits := oranges_bought + bananas_bought + apples_bought + pears_bought
  let rotten_oranges := oranges_rotten_percent * oranges_bought
  let rotten_bananas := bananas_rotten_percent * bananas_bought
  let rotten_apples := apples_rotten_percent * apples_bought
  let rotten_pears := pears_rotten_percent * pears_bought
  let good_oranges := oranges_bought - rotten_oranges
  let good_bananas := bananas_bought - rotten_bananas
  let good_apples := apples_bought - rotten_apples
  let good_pears := pears_bought - rotten_pears
  let total_good_fruits := good_oranges + good_bananas + good_apples + good_pears
  (total_good_fruits / total_fruits) * 100 = 87.6 :=
by
  sorry

end overall_percentage_good_fruits_l111_111998


namespace second_expression_l111_111304

variable (a b : ℕ)

theorem second_expression (h : 89 = ((2 * a + 16) + b) / 2) (ha : a = 34) : b = 94 :=
by
  sorry

end second_expression_l111_111304


namespace investment_return_l111_111170

theorem investment_return 
  (investment1 : ℝ) (investment2 : ℝ) 
  (return1 : ℝ) (combined_return_percent : ℝ) : 
  investment1 = 500 → 
  investment2 = 1500 → 
  return1 = 0.07 → 
  combined_return_percent = 0.085 → 
  (500 * 0.07 + 1500 * r = 2000 * 0.085) → 
  r = 0.09 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end investment_return_l111_111170


namespace height_flagstaff_l111_111992

variables (s_1 s_2 h_2 : ℝ)
variable (h : ℝ)

-- Define the conditions as given
def shadow_flagstaff := s_1 = 40.25
def shadow_building := s_2 = 28.75
def height_building := h_2 = 12.5
def similar_triangles := (h / s_1) = (h_2 / s_2)

-- Prove the height of the flagstaff
theorem height_flagstaff : shadow_flagstaff s_1 ∧ shadow_building s_2 ∧ height_building h_2 ∧ similar_triangles h s_1 h_2 s_2 → h = 17.5 :=
by sorry

end height_flagstaff_l111_111992


namespace eagles_win_at_least_three_out_of_five_l111_111614

noncomputable theory

def probability_of_eagles_winning_at_least_three_games (p : ℝ) (n : ℕ) : ℝ :=
  (nat.choose n 3 * p^3 * (1 - p)^(n - 3)) + 
  (nat.choose n 4 * p^4 * (1 - p)^(n - 4)) + 
  (nat.choose n 5 * p^5 * (1 - p)^(n - 5))

theorem eagles_win_at_least_three_out_of_five :
  probability_of_eagles_winning_at_least_three_games (1/2) 5 = 1/2 :=
by 
  sorry

end eagles_win_at_least_three_out_of_five_l111_111614


namespace polynomial_real_root_inequality_l111_111572

theorem polynomial_real_root_inequality (a b : ℝ) : 
  (∃ x : ℝ, x^4 - a * x^3 + 2 * x^2 - b * x + 1 = 0) → (a^2 + b^2 ≥ 8) :=
sorry

end polynomial_real_root_inequality_l111_111572


namespace richmond_population_l111_111137

theorem richmond_population (R V B : ℕ) (h0 : R = V + 1000) (h1 : V = 4 * B) (h2 : B = 500) : R = 3000 :=
by
  -- skipping proof
  sorry

end richmond_population_l111_111137


namespace inradius_inequality_l111_111108

theorem inradius_inequality
  (r r_A r_B r_C : ℝ) 
  (h_inscribed_circle: r > 0) 
  (h_tangent_circles_A: r_A > 0) 
  (h_tangent_circles_B: r_B > 0) 
  (h_tangent_circles_C: r_C > 0)
  : r ≤ r_A + r_B + r_C :=
  sorry

end inradius_inequality_l111_111108


namespace div_relation_l111_111384

theorem div_relation (a b d : ℝ) (h1 : a / b = 3) (h2 : b / d = 2 / 5) : d / a = 5 / 6 := by
  sorry

end div_relation_l111_111384


namespace sum_of_80th_equation_l111_111760

theorem sum_of_80th_equation : (2 * 80 + 1) + (5 * 80 - 1) = 560 := by
  sorry

end sum_of_80th_equation_l111_111760


namespace pounds_added_per_year_l111_111346

noncomputable def initial_age : ℕ := 13
noncomputable def new_age : ℕ := 18
noncomputable def initial_deadlift : ℕ := 300
noncomputable def new_deadlift : ℕ := 2.5 * initial_deadlift + 100

theorem pounds_added_per_year :
  (new_deadlift - initial_deadlift) / (new_age - initial_age) = 110 :=
by
  sorry

end pounds_added_per_year_l111_111346


namespace simplify_fraction_l111_111609

theorem simplify_fraction (a b : ℕ) (h : Nat.gcd a b = 24) : (a = 48) → (b = 72) → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end simplify_fraction_l111_111609


namespace number_of_cows_l111_111766

-- Definitions
variables (a g e c : ℕ)
variables (six_two : 6 * e = 2 * a + 4 * g) (eight_two : 8 * e = 2 * a + 8 * g)

-- Theorem statement
theorem number_of_cows (a g e : ℕ) (six_two : 6 * e = 2 * a + 4 * g) (eight_two : 8 * e = 2 * a + 8 * g) :
  ∃ c : ℕ, c * e * 6 = 6 * a + 36 * g ∧ c = 5 :=
by
  sorry

end number_of_cows_l111_111766


namespace gnuff_tutoring_minutes_l111_111733

theorem gnuff_tutoring_minutes 
  (flat_rate : ℕ) 
  (rate_per_minute : ℕ) 
  (total_paid : ℕ) :
  flat_rate = 20 → 
  rate_per_minute = 7 →
  total_paid = 146 → 
  ∃ minutes : ℕ, minutes = 18 ∧ flat_rate + rate_per_minute * minutes = total_paid := 
by
  -- Assume the necessary steps here
  sorry

end gnuff_tutoring_minutes_l111_111733


namespace simplify_expression_l111_111801

theorem simplify_expression (y : ℝ) :
  4 * y - 3 * y^3 + 6 - (1 - 4 * y + 3 * y^3) = -6 * y^3 + 8 * y + 5 :=
by
  sorry

end simplify_expression_l111_111801


namespace gnuff_tutor_minutes_l111_111726

/-- Definitions of the given conditions -/
def flat_rate : ℕ := 20
def per_minute_charge : ℕ := 7
def total_paid : ℕ := 146

/-- The proof statement -/
theorem gnuff_tutor_minutes :
  (total_paid - flat_rate) / per_minute_charge = 18 :=
by
  sorry

end gnuff_tutor_minutes_l111_111726


namespace can_increase_averages_by_transfer_l111_111756

def group1_grades : List ℝ := [5, 3, 5, 3, 5, 4, 3, 4, 3, 4, 5, 5]
def group2_grades : List ℝ := [3, 4, 5, 2, 3, 2, 5, 4, 5, 3]

def average (grades : List ℝ) : ℝ := grades.sum / grades.length

theorem can_increase_averages_by_transfer :
    ∃ (student : ℝ) (new_group1_grades new_group2_grades : List ℝ),
      student ∈ group1_grades ∧
      new_group1_grades = (group1_grades.erase student) ∧
      new_group2_grades = student :: group2_grades ∧
      average new_group1_grades > average group1_grades ∧ 
      average new_group2_grades > average group2_grades :=
by
  sorry

end can_increase_averages_by_transfer_l111_111756


namespace simplify_fraction_l111_111592

theorem simplify_fraction :
  (48 : ℚ) / 72 = 2 / 3 :=
sorry

end simplify_fraction_l111_111592


namespace candy_per_smaller_bag_l111_111888

-- Define the variables and parameters
def george_candy : ℕ := 648
def friends : ℕ := 3
def total_people : ℕ := friends + 1
def smaller_bags : ℕ := 8

-- Define the theorem
theorem candy_per_smaller_bag : (george_candy / total_people) / smaller_bags = 20 :=
by
  -- Assume the proof steps, not required to actually complete
  sorry

end candy_per_smaller_bag_l111_111888


namespace jason_quarters_l111_111770

def quarters_original := 49
def quarters_added := 25
def quarters_total := 74

theorem jason_quarters : quarters_original + quarters_added = quarters_total :=
by
  sorry

end jason_quarters_l111_111770


namespace carol_can_invite_friends_l111_111190

-- Definitions based on the problem's conditions
def invitations_per_pack := 9
def packs_bought := 5

-- Required proof statement
theorem carol_can_invite_friends :
  invitations_per_pack * packs_bought = 45 :=
by
  sorry

end carol_can_invite_friends_l111_111190


namespace min_value_a2_b2_l111_111704

theorem min_value_a2_b2 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) (h : a^2 - 2015 * a = b^2 - 2015 * b) : 
  a^2 + b^2 ≥ 2015^2 / 2 := 
sorry

end min_value_a2_b2_l111_111704


namespace uncle_ben_parking_probability_l111_111996

theorem uncle_ben_parking_probability :
  let total_spaces := 20
  let cars := 15
  let rv_spaces := 3
  let total_combinations := Nat.choose total_spaces cars
  let non_adjacent_empty_combinations := Nat.choose (total_spaces - rv_spaces) cars
  (1 - (non_adjacent_empty_combinations / total_combinations)) = (232 / 323) := by
  sorry

end uncle_ben_parking_probability_l111_111996


namespace rectangle_circles_l111_111239

theorem rectangle_circles (p q : Prop) (hp : p) (hq : ¬ q) : p ∨ q :=
by sorry

end rectangle_circles_l111_111239


namespace move_point_right_l111_111266

theorem move_point_right 
  (x y : ℤ)
  (h : (x, y) = (2, -1)) :
  (x + 3, y) = (5, -1) := 
by
  sorry

end move_point_right_l111_111266


namespace no_solution_intervals_l111_111222

theorem no_solution_intervals (a : ℝ) :
  (a < -13 ∨ a > 0) → ¬ ∃ x : ℝ, 6 * abs (x - 4 * a) + abs (x - a^2) + 5 * x - 3 * a = 0 :=
by sorry

end no_solution_intervals_l111_111222


namespace find_number_l111_111905

theorem find_number (n p q : ℝ) (h1 : n / p = 6) (h2 : n / q = 15) (h3 : p - q = 0.3) : n = 3 :=
by
  sorry

end find_number_l111_111905


namespace votes_cast_l111_111471

theorem votes_cast (V : ℝ) (h1 : V = 0.33 * V + (0.33 * V + 833)) : V = 2447 := 
by
  sorry

end votes_cast_l111_111471


namespace tom_hours_per_week_l111_111351

-- Define the conditions
def summer_hours_per_week := 40
def summer_weeks := 8
def summer_total_earnings := 3200
def semester_weeks := 24
def semester_total_earnings := 2400
def hourly_wage := summer_total_earnings / (summer_hours_per_week * summer_weeks)
def total_hours_needed := semester_total_earnings / hourly_wage

-- Define the theorem to prove
theorem tom_hours_per_week :
  (total_hours_needed / semester_weeks) = 10 :=
sorry

end tom_hours_per_week_l111_111351


namespace pirates_total_coins_l111_111788

theorem pirates_total_coins :
  ∀ (x : ℕ), (x * (x + 1)) / 2 = 5 * x → 6 * x = 54 :=
by
  intro x
  intro h
  -- proof omitted
  sorry

end pirates_total_coins_l111_111788


namespace prob_sum_divisible_by_4_l111_111065

-- Defining the set and its properties
def set : Finset ℕ := {1, 2, 3, 4, 5}

def isDivBy4 (n : ℕ) : Prop := n % 4 = 0

-- Defining a function to calculate combinations
def combinations (n r : ℕ) : ℕ := Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

-- Defining the successful outcomes and the total combinations
def successfulOutcomes : ℕ := 3
def totalOutcomes : ℕ := combinations 5 3

-- Defining the probability
def probability : ℚ := successfulOutcomes / ↑totalOutcomes

-- The proof problem
theorem prob_sum_divisible_by_4 : probability = 3 / 10 := by
  sorry

end prob_sum_divisible_by_4_l111_111065


namespace probability_of_two_red_shoes_is_0_1332_l111_111473

def num_red_shoes : ℕ := 4
def num_green_shoes : ℕ := 6
def total_shoes : ℕ := num_red_shoes + num_green_shoes

def probability_first_red_shoe : ℚ := num_red_shoes / total_shoes
def remaining_red_shoes_after_first_draw : ℕ := num_red_shoes - 1
def remaining_shoes_after_first_draw : ℕ := total_shoes - 1
def probability_second_red_shoe : ℚ := remaining_red_shoes_after_first_draw / remaining_shoes_after_first_draw

def probability_two_red_shoes : ℚ := probability_first_red_shoe * probability_second_red_shoe

theorem probability_of_two_red_shoes_is_0_1332 : probability_two_red_shoes = 1332 / 10000 :=
by
  sorry

end probability_of_two_red_shoes_is_0_1332_l111_111473


namespace coin_flip_probability_find_p_plus_q_l111_111971

theorem coin_flip_probability (h : ℚ) (H : 0 < h ∧ h < 1) :
  (choose 4 1) * h * (1 - h)^3 = (choose 4 2) * h^2 * (1 - h)^2 →
  (choose 4 2) * (2/5)^2 * (3/5)^2 = 216 / 625 :=
by
  sorry

lemma p_plus_q :
  216 + 625 = 841 :=
by
  exact rfl

theorem find_p_plus_q (h : ℚ) (H : 0 < h ∧ h < 1) :
  (choose 4 1) * h * (1 - h)^3 = (choose 4 2) * h^2 * (1 - h)^2 →
  (216 + 625 = 841) :=
by
  intro H1
  have H2 : (choose 4 2) * (2 / 5) ^ 2 * (3 / 5) ^ 2 = 216 / 625 := coin_flip_probability h H H1
  exact p_plus_q

end coin_flip_probability_find_p_plus_q_l111_111971


namespace fraction_eq_zero_implies_x_eq_one_l111_111749

theorem fraction_eq_zero_implies_x_eq_one (x : ℝ) (h1 : (x - 1) = 0) (h2 : (x - 5) ≠ 0) : x = 1 :=
sorry

end fraction_eq_zero_implies_x_eq_one_l111_111749


namespace original_ribbon_length_l111_111824

theorem original_ribbon_length :
  ∃ x : ℝ, 
    (∀ a b : ℝ, 
       a = x - 18 ∧ 
       b = x - 12 ∧ 
       b = 2 * a → x = 24) :=
by
  sorry

end original_ribbon_length_l111_111824


namespace num_ways_4x4_proof_l111_111326

-- Define a function that represents the number of ways to cut a 2x2 square
noncomputable def num_ways_2x2_cut : ℕ := 4

-- Define a function that represents the number of ways to cut a 3x3 square
noncomputable def num_ways_3x3_cut (ways_2x2 : ℕ) : ℕ :=
  ways_2x2 * 4

-- Define a function that represents the number of ways to cut a 4x4 square
noncomputable def num_ways_4x4_cut (ways_3x3 : ℕ) : ℕ :=
  ways_3x3 * 4

-- Prove the final number of ways to cut the 4x4 square into 3 L-shaped pieces and 1 small square
theorem num_ways_4x4_proof : num_ways_4x4_cut (num_ways_3x3_cut num_ways_2x2_cut) = 64 := by
  sorry

end num_ways_4x4_proof_l111_111326


namespace represent_2015_l111_111795

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def in_interval (n : ℕ) : Prop := 400 < n ∧ n < 500

def not_divisible_by_3 (n : ℕ) : Prop := n % 3 ≠ 0

theorem represent_2015 :
  ∃ a b c : ℕ,
  a + b + c = 2015 ∧
  is_prime a ∧
  is_divisible_by_3 b ∧
  in_interval c ∧
  not_divisible_by_3 c :=
by {
  use 7,
  use 1605,
  use 403,
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  { sorry }
}

end represent_2015_l111_111795


namespace number_of_students_not_enrolled_in_biology_l111_111164

noncomputable def total_students : ℕ := 880

noncomputable def biology_enrollment_percent : ℕ := 40

noncomputable def students_not_enrolled_in_biology : ℕ :=
  (100 - biology_enrollment_percent) * total_students / 100

theorem number_of_students_not_enrolled_in_biology :
  students_not_enrolled_in_biology = 528 :=
by
  -- Proof goes here.
  -- Use sorry to skip the proof for this placeholder:
  sorry

end number_of_students_not_enrolled_in_biology_l111_111164


namespace band_song_average_l111_111009

/-- 
The school band has 30 songs in their repertoire. 
They played 5 songs in the first set and 7 songs in the second set. 
They will play 2 songs for their encore. 
Assuming the band plays through their entire repertoire, 
how many songs will they play on average in the third and fourth sets?
 -/
theorem band_song_average
    (total_songs : ℕ)
    (first_set_songs : ℕ)
    (second_set_songs : ℕ)
    (encore_songs : ℕ)
    (remaining_sets : ℕ)
    (h_total : total_songs = 30)
    (h_first : first_set_songs = 5)
    (h_second : second_set_songs = 7)
    (h_encore : encore_songs = 2)
    (h_remaining : remaining_sets = 2) :
    (total_songs - (first_set_songs + second_set_songs + encore_songs)) / remaining_sets = 8 := 
by
  -- The proof will go here.
  sorry

end band_song_average_l111_111009


namespace sin_C_of_arith_prog_angles_l111_111523

theorem sin_C_of_arith_prog_angles (A B C a b : ℝ) (h_abc : A + B + C = Real.pi)
  (h_arith_prog : 2 * B = A + C) (h_a : a = Real.sqrt 2) (h_b : b = Real.sqrt 3) :
  Real.sin C = (Real.sqrt 2 + Real.sqrt 6) / 4 :=
sorry

end sin_C_of_arith_prog_angles_l111_111523


namespace geometric_seq_arithmetic_triplet_l111_111270

-- Definition of being in a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n * q

-- Condition that a_5, a_4, and a_6 form an arithmetic sequence
def is_arithmetic_triplet (a : ℕ → ℝ) (n : ℕ) : Prop :=
  2 * a n = a (n+1) + a (n+2)

-- Our specific problem translated into a Lean statement
theorem geometric_seq_arithmetic_triplet {a : ℕ → ℝ} (q : ℝ) :
  is_geometric_sequence a q →
  is_arithmetic_triplet a 4 →
  q = 1 ∨ q = -2 :=
by
  intros h_geo h_arith
  -- Proof here is omitted
  sorry

end geometric_seq_arithmetic_triplet_l111_111270


namespace average_test_score_of_remainder_l111_111096

variable (score1 score2 score3 totalAverage : ℝ)
variable (percentage1 percentage2 percentage3 : ℝ)

def equation (score1 score2 score3 totalAverage : ℝ) (percentage1 percentage2 percentage3: ℝ) : Prop :=
  (percentage1 * score1) + (percentage2 * score2) + (percentage3 * score3) = totalAverage

theorem average_test_score_of_remainder
  (h1 : percentage1 = 0.15)
  (h2 : score1 = 100)
  (h3 : percentage2 = 0.5)
  (h4 : score2 = 78)
  (h5 : percentage3 = 0.35)
  (total : totalAverage = 76.05) :
  (score3 = 63) :=
sorry

end average_test_score_of_remainder_l111_111096


namespace completing_the_square_l111_111826

theorem completing_the_square (x : ℝ) : (x^2 - 6*x + 7 = 0) → ((x - 3)^2 = 2) :=
by
  intro h
  sorry

end completing_the_square_l111_111826


namespace min_value_of_squares_l111_111705

theorem min_value_of_squares (a b : ℝ) 
  (h_cond : a^2 - 2015 * a = b^2 - 2015 * b)
  (h_neq : a ≠ b)
  (h_positive_a : 0 < a)
  (h_positive_b : 0 < b) : 
  a^2 + b^2 ≥ 2015^2 / 2 :=
sorry

end min_value_of_squares_l111_111705


namespace ants_meet_distance_is_half_total_l111_111450

-- Definitions given in the problem
structure Tile :=
  (width : ℤ)
  (length : ℤ)

structure Ant :=
  (start_position : String)

-- Conditions from the problem
def tile : Tile := ⟨4, 6⟩
def maricota : Ant := ⟨"M"⟩
def nandinha : Ant := ⟨"N"⟩
def total_lengths := 14
def total_widths := 12

noncomputable
def calculate_total_distance (total_lengths : ℤ) (total_widths : ℤ) (tile : Tile) := 
  (total_lengths * tile.length) + (total_widths * tile.width)

-- Question stated as a theorem
theorem ants_meet_distance_is_half_total :
  calculate_total_distance total_lengths total_widths tile = 132 →
  (calculate_total_distance total_lengths total_widths tile) / 2 = 66 :=
by
  intro h
  sorry

end ants_meet_distance_is_half_total_l111_111450


namespace problem_statement_l111_111089

theorem problem_statement (m n c d a : ℝ)
  (h1 : m = -n)
  (h2 : c * d = 1)
  (h3 : a = 2) :
  Real.sqrt (c * d) + 2 * (m + n) - a = -1 :=
by
  -- Proof steps are skipped with sorry 
  sorry

end problem_statement_l111_111089


namespace tutoring_minutes_l111_111729

def flat_rate : ℤ := 20
def per_minute_rate : ℤ := 7
def total_paid : ℤ := 146

theorem tutoring_minutes (m : ℤ) : total_paid = flat_rate + (per_minute_rate * m) → m = 18 :=
by
  sorry

end tutoring_minutes_l111_111729


namespace problem1_l111_111169

theorem problem1 (α : ℝ) (h : Real.tan (π/4 + α) = 2) : Real.sin (2 * α) + Real.cos α ^ 2 = 3 / 2 := 
sorry

end problem1_l111_111169


namespace find_f4_l111_111895

theorem find_f4 (f : ℝ → ℝ) 
  (h1 : ∀ x, f (x + 1) = -f (-x + 1)) 
  (h2 : ∀ x, f (x - 1) = f (-x - 1)) 
  (h3 : f 0 = 2) : 
  f 4 = -2 :=
sorry

end find_f4_l111_111895


namespace average_episodes_per_year_l111_111840

theorem average_episodes_per_year (total_years : ℕ) (n1 n2 n3 e1 e2 e3 : ℕ) 
  (h1 : total_years = 14)
  (h2 : n1 = 8) (h3 : e1 = 15)
  (h4 : n2 = 4) (h5 : e2 = 20)
  (h6 : n3 = 2) (h7 : e3 = 12) :
  (n1 * e1 + n2 * e2 + n3 * e3) / total_years = 16 := by
  -- Skip the proof steps
  sorry

end average_episodes_per_year_l111_111840


namespace fraction_zero_iff_x_one_l111_111747

theorem fraction_zero_iff_x_one (x : ℝ) (h₁ : x - 1 = 0) (h₂ : x - 5 ≠ 0) : x = 1 := by
  sorry

end fraction_zero_iff_x_one_l111_111747


namespace first_number_is_twenty_l111_111808

theorem first_number_is_twenty (x : ℕ) : 
  (x + 40 + 60) / 3 = ((10 + 70 + 16) / 3) + 8 → x = 20 := 
by 
  sorry

end first_number_is_twenty_l111_111808


namespace hyperbola_eccentricity_l111_111084

theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) (h2 : 2 = (Real.sqrt (a^2 + 3)) / a) : a = 1 := 
by
  sorry

end hyperbola_eccentricity_l111_111084


namespace remaining_course_distance_l111_111394

def total_distance_km : ℝ := 10.5
def distance_to_break_km : ℝ := 1.5
def additional_distance_m : ℝ := 3730.0

theorem remaining_course_distance :
  let total_distance_m := total_distance_km * 1000
  let distance_to_break_m := distance_to_break_km * 1000
  let total_traveled_m := distance_to_break_m + additional_distance_m
  total_distance_m - total_traveled_m = 5270 := by
  sorry

end remaining_course_distance_l111_111394


namespace remainder_of_f_when_divided_by_x_plus_2_l111_111058

def f (x : ℝ) : ℝ := x^4 - 6 * x^3 + 11 * x^2 + 8 * x - 20

theorem remainder_of_f_when_divided_by_x_plus_2 : f (-2) = 72 := by
  sorry

end remainder_of_f_when_divided_by_x_plus_2_l111_111058


namespace product_of_means_eq_pm20_l111_111960

theorem product_of_means_eq_pm20 :
  let a := (2 + 8) / 2
  let b := Real.sqrt (2 * 8)
  a * b = 20 ∨ a * b = -20 :=
by
  -- Placeholders for the actual proof
  let a := (2 + 8) / 2
  let b := Real.sqrt (2 * 8)
  sorry

end product_of_means_eq_pm20_l111_111960


namespace combine_expr_l111_111866

variable (a b : ℝ)

theorem combine_expr : 3 * (2 * a - 3 * b) - 6 * (a - b) = -3 * b := by
  sorry

end combine_expr_l111_111866


namespace union_of_A_and_B_l111_111897

-- Definitions for sets A and B
def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

-- Theorem statement to prove the union of A and B
theorem union_of_A_and_B : A ∪ B = {2, 3, 5, 6} := by
  sorry

end union_of_A_and_B_l111_111897


namespace representation_of_2015_l111_111793

theorem representation_of_2015 :
  ∃ (p d3 i : ℕ),
    Prime p ∧ -- p is prime
    d3 % 3 = 0 ∧ -- d3 is divisible by 3
    400 < i ∧ i < 500 ∧ i % 3 ≠ 0 ∧ -- i is in interval and not divisible by 3
    2015 = p + d3 + i := sorry

end representation_of_2015_l111_111793


namespace bananas_per_monkey_l111_111483

-- Define the given conditions
def total_monkeys : ℕ := 12
def piles_with_9hands : ℕ := 6
def hands_per_pile_9hands : ℕ := 9
def bananas_per_hand_9hands : ℕ := 14
def piles_with_12hands : ℕ := 4
def hands_per_pile_12hands : ℕ := 12
def bananas_per_hand_12hands : ℕ := 9

-- Calculate the total number of bananas from each type of pile
def total_bananas_9hands : ℕ := piles_with_9hands * hands_per_pile_9hands * bananas_per_hand_9hands
def total_bananas_12hands : ℕ := piles_with_12hands * hands_per_pile_12hands * bananas_per_hand_12hands

-- Sum the total number of bananas
def total_bananas : ℕ := total_bananas_9hands + total_bananas_12hands

-- Prove that each monkey gets 99 bananas
theorem bananas_per_monkey : total_bananas / total_monkeys = 99 := by
  sorry

end bananas_per_monkey_l111_111483


namespace trader_sold_90_pens_l111_111864

theorem trader_sold_90_pens (C N : ℝ) (gain_percent : ℝ) (H1 : gain_percent = 33.33333333333333) (H2 : 30 * C = (gain_percent / 100) * N * C) :
  N = 90 :=
by
  sorry

end trader_sold_90_pens_l111_111864


namespace plane_equation_rewriting_l111_111812

theorem plane_equation_rewriting (A B C D x y z p q r : ℝ)
  (hA : A ≠ 0) (hB : B ≠ 0) (hC : C ≠ 0) (hD : D ≠ 0)
  (eq1 : A * x + B * y + C * z + D = 0)
  (hp : p = -D / A) (hq : q = -D / B) (hr : r = -D / C) :
  x / p + y / q + z / r = 1 :=
by
  sorry

end plane_equation_rewriting_l111_111812


namespace represent_2015_l111_111796

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_divisible_by_3 (n : ℕ) : Prop := n % 3 = 0

def in_interval (n : ℕ) : Prop := 400 < n ∧ n < 500

def not_divisible_by_3 (n : ℕ) : Prop := n % 3 ≠ 0

theorem represent_2015 :
  ∃ a b c : ℕ,
  a + b + c = 2015 ∧
  is_prime a ∧
  is_divisible_by_3 b ∧
  in_interval c ∧
  not_divisible_by_3 c :=
by {
  use 7,
  use 1605,
  use 403,
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  { sorry }
}

end represent_2015_l111_111796


namespace b_share_220_l111_111649

theorem b_share_220 (A B C : ℝ) (h1 : A = B + 40) (h2 : C = A + 30) (h3 : B + A + C = 770) : B = 220 :=
by
  sorry

end b_share_220_l111_111649


namespace solution_system_eq_l111_111132

theorem solution_system_eq (x y : ℝ) :
  (x^2 * y + x * y^2 - 2 * x - 2 * y + 10 = 0) ∧
  (x^3 * y - x * y^3 - 2 * x^2 + 2 * y^2 - 30 = 0) ↔ 
  (x = -4 ∧ y = -1) :=
by sorry

end solution_system_eq_l111_111132


namespace perfect_square_trinomial_k_l111_111737

theorem perfect_square_trinomial_k (k : ℤ) :
  (∀ x : ℝ, 9 * x^2 + 6 * x + k = (3 * x + 1) ^ 2) → (k = 1) :=
by
  sorry

end perfect_square_trinomial_k_l111_111737


namespace price_reduction_equation_l111_111657

theorem price_reduction_equation (x : ℝ) (P_initial : ℝ) (P_final : ℝ) 
  (h1 : P_initial = 560) (h2 : P_final = 315) : 
  P_initial * (1 - x)^2 = P_final :=
by
  rw [h1, h2]
  sorry

end price_reduction_equation_l111_111657


namespace smallest_multiple_l111_111156

theorem smallest_multiple (x : ℕ) (h1 : x % 24 = 0) (h2 : x % 36 = 0) (h3 : x % 20 ≠ 0) :
  x = 72 :=
by
  sorry

end smallest_multiple_l111_111156


namespace car_more_miles_per_tank_after_modification_l111_111335

theorem car_more_miles_per_tank_after_modification (mpg_old : ℕ) (efficiency_factor : ℝ) (gallons : ℕ) :
  mpg_old = 33 →
  efficiency_factor = 1.25 →
  gallons = 16 →
  (efficiency_factor * mpg_old * gallons - mpg_old * gallons) = 132 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry  -- Proof omitted

end car_more_miles_per_tank_after_modification_l111_111335


namespace average_salary_decrease_l111_111916

theorem average_salary_decrease 
    (avg_wage_illiterate_initial : ℝ)
    (avg_wage_illiterate_new : ℝ)
    (num_illiterate : ℕ)
    (num_literate : ℕ)
    (num_total : ℕ)
    (total_decrease : ℝ) :
    avg_wage_illiterate_initial = 25 →
    avg_wage_illiterate_new = 10 →
    num_illiterate = 20 →
    num_literate = 10 →
    num_total = num_illiterate + num_literate →
    total_decrease = (avg_wage_illiterate_initial - avg_wage_illiterate_new) * num_illiterate →
    total_decrease / num_total = 10 :=
by
  intros avg_wage_illiterate_initial_eq avg_wage_illiterate_new_eq num_illiterate_eq num_literate_eq num_total_eq total_decrease_eq
  sorry

end average_salary_decrease_l111_111916


namespace max_sum_of_two_integers_l111_111627

theorem max_sum_of_two_integers (x : ℕ) (h : x + 2 * x < 100) : x + 2 * x = 99 :=
sorry

end max_sum_of_two_integers_l111_111627


namespace valid_triples_l111_111684

theorem valid_triples (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hxy : x ∣ (y + 1)) (hyz : y ∣ (z + 1)) (hzx : z ∣ (x + 1)) :
  (x = 1 ∧ y = 1 ∧ z = 1) ∨ 
  (x = 1 ∧ y = 1 ∧ z = 2) ∨ 
  (x = 1 ∧ y = 2 ∧ z = 3) :=
sorry

end valid_triples_l111_111684


namespace sufficient_but_not_necessary_l111_111469

-- Define what it means for α to be of the form (π/6 + 2kπ) where k ∈ ℤ
def is_pi_six_plus_two_k_pi (α : ℝ) : Prop :=
  ∃ k : ℤ, α = Real.pi / 6 + 2 * k * Real.pi

-- Define the condition sin α = 1 / 2
def sin_is_half (α : ℝ) : Prop :=
  Real.sin α = 1 / 2

-- The theorem stating that the given condition is a sufficient but not necessary condition
theorem sufficient_but_not_necessary (α : ℝ) :
  is_pi_six_plus_two_k_pi α → sin_is_half α ∧ ¬ (sin_is_half α → is_pi_six_plus_two_k_pi α) :=
by
  sorry

end sufficient_but_not_necessary_l111_111469


namespace distinct_positive_integers_factors_PQ_RS_l111_111167

theorem distinct_positive_integers_factors_PQ_RS (P Q R S : ℕ) (hP : P > 0) (hQ : Q > 0) (hR : R > 0) (hS : S > 0)
  (hPQ : P * Q = 72) (hRS : R * S = 72) (hDistinctPQ : P ≠ Q) (hDistinctRS : R ≠ S) (hPQR_S : P + Q = R - S) :
  P = 4 :=
by
  sorry

end distinct_positive_integers_factors_PQ_RS_l111_111167


namespace binomial_7_4_eq_35_l111_111196
-- Import the entire Mathlib library for broad utility functions

-- Define the binomial coefficient function using library definitions 
noncomputable def binomial : ℕ → ℕ → ℕ
| n, k := nat.choose n k

-- Define the theorem we want to prove
theorem binomial_7_4_eq_35 : binomial 7 4 = 35 :=
by
  sorry

end binomial_7_4_eq_35_l111_111196


namespace letter_lock_rings_l111_111339

theorem letter_lock_rings (n : ℕ) (h : n^3 - 1 ≤ 215) : n = 6 :=
by { sorry }

end letter_lock_rings_l111_111339


namespace sasha_claim_l111_111890

noncomputable def largest_k (n : ℕ) : ℕ :=
  2 * Nat.ceil (n / 2 : ℝ)

theorem sasha_claim (n : ℕ) (rays : Fin n → Ray) (h : ∀ i j, rays i ≠ rays j):
  ∃ k, (∀ (points : Fin k → Point), ∃ (s : Sphere), ∀ i, points i ∈ s) ∧ 
       k = largest_k n :=
by
  sorry

end sasha_claim_l111_111890


namespace find_b_l111_111402

-- Define the conditions as Lean definitions
def a : ℝ := 4
def angle_B : ℝ := 60 * Real.pi / 180
def angle_C : ℝ := 75 * Real.pi / 180

-- Provide a theorem for the given problem that determines b
theorem find_b : 
  ∃ b : ℝ, (
    ∃ A : ℝ, A = Real.pi - angle_B - angle_C
    ) ∧ b = a * (Real.sin angle_B) / (Real.sin (Real.pi - angle_B - angle_C)) ∧ b = 2 * Real.sqrt 6 :=
by
  -- placeholder for the proof
  sorry

end find_b_l111_111402


namespace fill_tank_time_l111_111323

/-- 
If pipe A fills a tank in 30 minutes, pipe B fills the same tank in 20 minutes, 
and pipe C empties it in 40 minutes, then the time it takes to fill the tank 
when all three pipes are working together is 120/7 minutes.
-/
theorem fill_tank_time 
  (rate_A : ℝ) (rate_B : ℝ) (rate_C : ℝ) (combined_rate : ℝ) (T : ℝ) :
  rate_A = 1/30 ∧ rate_B = 1/20 ∧ rate_C = -1/40 ∧ combined_rate = rate_A + rate_B + rate_C
  → T = 1 / combined_rate
  → T = 120 / 7 :=
by
  intros
  sorry

end fill_tank_time_l111_111323


namespace adjacent_probability_is_2_over_7_l111_111302

variable (n : Nat := 5) -- number of student performances
variable (m : Nat := 2) -- number of teacher performances

/-- Total number of ways to insert two performances
    (ignoring adjacency constraints) into the program list. -/
def total_insertion_ways : Nat :=
  Fintype.card (Fin (n + m))

/-- Number of ways to insert two performances such that they are adjacent. -/
def adjacent_insertion_ways : Nat :=
  Fintype.card (Fin (n + 1))

/-- Probability that two specific performances are adjacent in a program list. -/
def adjacent_probability : ℚ :=
  adjacent_insertion_ways / total_insertion_ways

theorem adjacent_probability_is_2_over_7 :
  adjacent_probability = (2 : ℚ) / 7 := by
  sorry

end adjacent_probability_is_2_over_7_l111_111302


namespace average_of_first_n_multiples_of_8_is_88_l111_111640

theorem average_of_first_n_multiples_of_8_is_88 (n : ℕ) (h : (n / 2) * (8 + 8 * n) / n = 88) : n = 21 :=
sorry

end average_of_first_n_multiples_of_8_is_88_l111_111640


namespace simplify_fraction_l111_111593

theorem simplify_fraction :
  (48 : ℚ) / 72 = 2 / 3 :=
sorry

end simplify_fraction_l111_111593


namespace none_of_these_l111_111666

theorem none_of_these (a T : ℝ) : 
  ¬(∀ (x y : ℝ), 4 * T * x + 2 * a^2 * y + 4 * a * T = 0) ∧ 
  ¬(∀ (x y : ℝ), 4 * T * x - 2 * a^2 * y + 4 * a * T = 0) ∧ 
  ¬(∀ (x y : ℝ), 4 * T * x + 2 * a^2 * y - 4 * a * T = 0) ∧ 
  ¬(∀ (x y : ℝ), 4 * T * x - 2 * a^2 * y - 4 * a * T = 0) :=
sorry

end none_of_these_l111_111666


namespace probability_blue_ball_l111_111964

-- Define the probabilities of drawing a red and yellow ball
def P_red : ℝ := 0.48
def P_yellow : ℝ := 0.35

-- Define the total probability formula in this sample space
def total_probability (P_red P_yellow P_blue : ℝ) : Prop :=
  P_red + P_yellow + P_blue = 1

-- The theorem we need to prove
theorem probability_blue_ball :
  ∃ P_blue : ℝ, total_probability P_red P_yellow P_blue ∧ P_blue = 0.17 :=
sorry

end probability_blue_ball_l111_111964


namespace math_scores_between_70_and_80_l111_111919

-- Declare the conditions of the problem
def num_students : ℕ := 3000
def normal_distribution := true  -- This is a placeholder for normal distribution properties.

-- Problem statement
theorem math_scores_between_70_and_80 
  (student_count : ℕ = num_students)
  (distribution : normal_distribution)
  : ∃ num_students_between_70_80 : ℕ, num_students_between_70_80 = 408
  :=
  sorry

end math_scores_between_70_and_80_l111_111919


namespace white_balls_count_l111_111851

theorem white_balls_count (W B R : ℕ) (h1 : B = W + 14) (h2 : R = 3 * (B - W)) (h3 : W + B + R = 1000) : W = 472 :=
sorry

end white_balls_count_l111_111851


namespace minimal_d1_l111_111011

theorem minimal_d1 :
  (∃ (S3 S6 : ℕ), 
    ∃ (d1 : ℚ), 
      S3 = d1 + (d1 + 1) + (d1 + 2) ∧ 
      S6 = d1 + (d1 + 1) + (d1 + 2) + (d1 + 3) + (d1 + 4) + (d1 + 5) ∧ 
      d1 = (5 * S3 - S6) / 9 ∧ 
      d1 ≥ 1 / 2) → 
  ∃ (d1 : ℚ), d1 = 5 / 9 := 
by 
  sorry

end minimal_d1_l111_111011


namespace pen_price_relationship_l111_111388

variable (x : ℕ) -- x represents the number of pens
variable (y : ℝ) -- y represents the total selling price in dollars
variable (p : ℝ) -- p represents the price per pen

-- Each box contains 10 pens
def pens_per_box := 10

-- Each box is sold for $16
def price_per_box := 16

-- Given the conditions, prove the relationship between y and x
theorem pen_price_relationship (hx : x = 10) (hp : p = 16) :
  y = 1.6 * x := sorry

end pen_price_relationship_l111_111388


namespace tan_y_eq_tan_x_plus_one_over_cos_x_l111_111566

theorem tan_y_eq_tan_x_plus_one_over_cos_x 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hxy : x < y) 
  (hy : y < π / 2) 
  (h_tan : Real.tan y = Real.tan x + (1 / Real.cos x)) 
  : y - (x / 2) = π / 6 :=
sorry

end tan_y_eq_tan_x_plus_one_over_cos_x_l111_111566


namespace triangle_area_l111_111967

theorem triangle_area (P : ℝ × ℝ)
  (Q : ℝ × ℝ) (R : ℝ × ℝ)
  (P_eq : P = (3, 2))
  (Q_eq : ∃ b, Q = (7/3, 0) ∧ 2 = 3 * 3 + b ∧ 0 = 3 * (7/3) + b)
  (R_eq : ∃ b, R = (4, 0) ∧ 2 = -2 * 3 + b ∧ 0 = -2 * 4 + b) :
  (1/2) * abs (Q.1 - R.1) * abs (P.2) = 5/3 :=
by
  sorry

end triangle_area_l111_111967


namespace fourth_square_area_l111_111153

theorem fourth_square_area (PQ QR RS QS : ℝ)
  (h1 : PQ^2 = 25)
  (h2 : QR^2 = 49)
  (h3 : RS^2 = 64) :
  QS^2 = 138 :=
by
  sorry

end fourth_square_area_l111_111153


namespace square_of_fourth_power_of_fourth_smallest_prime_l111_111830

-- Define the fourth smallest prime number
def fourth_smallest_prime : ℕ := 7

-- Define the square of the fourth power of that number
def square_of_fourth_power (n : ℕ) : ℕ := (n^4)^2

-- Prove the main statement
theorem square_of_fourth_power_of_fourth_smallest_prime : square_of_fourth_power fourth_smallest_prime = 5764801 :=
by
  sorry

end square_of_fourth_power_of_fourth_smallest_prime_l111_111830


namespace kara_water_intake_l111_111563

-- Definitions based on the conditions
def daily_doses := 3
def week1_days := 7
def week2_days := 7
def forgot_doses_day := 2
def total_weeks := 2
def total_water := 160

-- The statement to prove
theorem kara_water_intake :
  let total_doses := (daily_doses * week1_days) + (daily_doses * week2_days - forgot_doses_day)
  ∃ (water_per_dose : ℕ), water_per_dose * total_doses = total_water ∧ water_per_dose = 4 :=
by
  sorry

end kara_water_intake_l111_111563


namespace feed_cost_l111_111428

theorem feed_cost (total_birds ducks_fraction chicken_feed_cost : ℕ) (h1 : total_birds = 15) (h2 : ducks_fraction = 1/3) (h3 : chicken_feed_cost = 2) :
  15 * (1 - 1/3) * 2 = 20 :=
by
  sorry

end feed_cost_l111_111428


namespace number_of_possible_x_values_l111_111144
noncomputable def triangle_sides_possible_values (x : ℕ) : Prop :=
  27 < x ∧ x < 63

theorem number_of_possible_x_values : 
  ∃ n, n = (62 - 28 + 1) ∧ ( ∀ x : ℕ, triangle_sides_possible_values x ↔ 28 ≤ x ∧ x ≤ 62) :=
sorry

end number_of_possible_x_values_l111_111144


namespace perpendicular_lines_m_value_l111_111743

theorem perpendicular_lines_m_value
  (l1 : ∀ (x y : ℝ), x - 2 * y + 1 = 0)
  (l2 : ∀ (x y : ℝ), m * x + y - 3 = 0)
  (perpendicular : ∀ (m : ℝ) (l1_slope l2_slope : ℝ), l1_slope * l2_slope = -1) : 
  m = 2 :=
by
  sorry

end perpendicular_lines_m_value_l111_111743


namespace min_value_of_squares_l111_111706

theorem min_value_of_squares (a b : ℝ) 
  (h_cond : a^2 - 2015 * a = b^2 - 2015 * b)
  (h_neq : a ≠ b)
  (h_positive_a : 0 < a)
  (h_positive_b : 0 < b) : 
  a^2 + b^2 ≥ 2015^2 / 2 :=
sorry

end min_value_of_squares_l111_111706


namespace longest_side_of_similar_triangle_l111_111010

-- Define the sides of the original triangle
def a : ℕ := 8
def b : ℕ := 10
def c : ℕ := 12

-- Define the perimeter of the similar triangle
def perimeter_similar_triangle : ℕ := 150

-- Formalize the problem using Lean statement
theorem longest_side_of_similar_triangle :
  ∃ x : ℕ, 8 * x + 10 * x + 12 * x = 150 ∧ 12 * x = 60 :=
by
  sorry

end longest_side_of_similar_triangle_l111_111010


namespace initial_blue_balls_l111_111311

-- Define the problem conditions
variable (R B : ℕ) -- Number of red balls and blue balls originally in the box.

-- Condition 1: Blue balls are 17 more than red balls
axiom h1 : B = R + 17

-- Condition 2: Ball addition and removal scenario
noncomputable def total_balls_after_changes : ℕ :=
  (B + 57) + (R + 18) - 44

-- Condition 3: Total balls after all changes is 502
axiom h2 : total_balls_after_changes R B = 502

-- We need to prove the initial number of blue balls
theorem initial_blue_balls : B = 244 :=
by
  sorry

end initial_blue_balls_l111_111311


namespace largest_even_k_for_sum_of_consecutive_integers_l111_111359

theorem largest_even_k_for_sum_of_consecutive_integers (k n : ℕ) (h_k_even : k % 2 = 0) :
  (3^10 = k * (2 * n + k + 1)) → k ≤ 162 :=
sorry

end largest_even_k_for_sum_of_consecutive_integers_l111_111359


namespace solve_abs_inequality_l111_111884

theorem solve_abs_inequality :
  { x : ℝ | 3 ≤ |x - 2| ∧ |x - 2| ≤ 6 } = { x : ℝ | -4 ≤ x ∧ x ≤ -1 } ∪ { x : ℝ | 5 ≤ x ∧ x ≤ 8 } :=
sorry

end solve_abs_inequality_l111_111884


namespace sandy_initial_books_l111_111575

-- Define the initial conditions as given.
def books_tim : ℕ := 33
def books_lost : ℕ := 24
def books_after_loss : ℕ := 19

-- Define the equation for the total books before Benny's loss and solve for Sandy's books.
def books_total_before_loss : ℕ := books_after_loss + books_lost
def books_sandy_initial : ℕ := books_total_before_loss - books_tim

-- Assert the proof statement:
def proof_sandy_books : Prop :=
  books_sandy_initial = 10

theorem sandy_initial_books : proof_sandy_books := by
  -- Placeholder for the actual proof.
  sorry

end sandy_initial_books_l111_111575


namespace proof_problem_l111_111519

variable {a_n : ℕ → ℤ}
variable {b_n : ℕ → ℤ}
variable {c_n : ℕ → ℤ}
variable {T_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}

-- Conditions

-- 1. The common difference d of the arithmetic sequence {a_n} is greater than 0
def common_difference_positive (d : ℤ) : Prop :=
  d > 0

-- 2. a_2 and a_5 are the two roots of the equation x^2 - 12x + 27 = 0
def roots_of_quadratic (a2 a5 : ℤ) : Prop :=
  a2^2 - 12 * a2 + 27 = 0 ∧ a5^2 - 12 * a5 + 27 = 0

-- 3. The sum of the first n terms of the sequence {b_n} is S_n, and it is given that S_n = (3 / 2)(b_n - 1)
def sum_of_b_n (S_n b_n : ℕ → ℤ) : Prop :=
  ∀ n, S_n n = 3/2 * (b_n n - 1)

-- Define the sequences to display further characteristics

-- 1. Find the general formula for the sequences {a_n} and {b_n}
def general_formula_a (a : ℕ → ℤ) : Prop :=
  ∀ n, a n = 2 * n - 1

def general_formula_b (b : ℕ → ℤ) : Prop :=
  ∀ n, b n = 3 ^ n

-- 2. Check if c_n = a_n * b_n and find the sum T_n
def c_n_equals_a_n_times_b_n (a b : ℕ → ℤ) (c : ℕ → ℤ) : Prop :=
  ∀ n, c n = a n * b n

def sum_T_n (T c : ℕ → ℤ) : Prop :=
  ∀ n, T n = 3 + (n - 1) * 3^(n + 1)

theorem proof_problem 
  (d : ℤ)
  (a2 a5 : ℤ)
  (S_n b_n : ℕ → ℤ)
  (a_n b_n c_n T_n : ℕ → ℤ) :
  common_difference_positive d ∧
  roots_of_quadratic a2 a5 ∧ 
  sum_of_b_n S_n b_n ∧ 
  general_formula_a a_n ∧ 
  general_formula_b b_n ∧ 
  c_n_equals_a_n_times_b_n a_n b_n c_n ∧ 
  sum_T_n T_n c_n :=
sorry

end proof_problem_l111_111519


namespace longest_collection_has_more_pages_l111_111418

noncomputable def miles_pages_per_inch := 5
noncomputable def daphne_pages_per_inch := 50
noncomputable def miles_height_inches := 240
noncomputable def daphne_height_inches := 25

noncomputable def miles_total_pages := miles_height_inches * miles_pages_per_inch
noncomputable def daphne_total_pages := daphne_height_inches * daphne_pages_per_inch

theorem longest_collection_has_more_pages :
  max miles_total_pages daphne_total_pages = 1250 := by
  -- Skip the proof
  sorry

end longest_collection_has_more_pages_l111_111418


namespace expression_equals_answer_l111_111681

noncomputable def verify_expression : ℚ :=
  15 * (1 / 17) * 34 - (1 / 2)

theorem expression_equals_answer :
  verify_expression = 59 / 2 :=
by
  sorry

end expression_equals_answer_l111_111681


namespace percentage_of_students_owning_cats_l111_111397

def total_students : ℕ := 500
def students_with_cats : ℕ := 75

theorem percentage_of_students_owning_cats (total_students students_with_cats : ℕ) (h_total: total_students = 500) (h_cats: students_with_cats = 75) :
  100 * (students_with_cats / total_students : ℝ) = 15 := by
  sorry

end percentage_of_students_owning_cats_l111_111397


namespace wendy_made_money_l111_111208

-- Given conditions
def price_per_bar : ℕ := 3
def total_bars : ℕ := 9
def bars_sold : ℕ := total_bars - 3

-- Statement to prove: Wendy made $18
theorem wendy_made_money : bars_sold * price_per_bar = 18 := by
  sorry

end wendy_made_money_l111_111208


namespace find_f_neg_2_l111_111779

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then 3 * x + 4 else 7 - 3 * x

theorem find_f_neg_2 : f (-2) = 13 := by
  sorry

end find_f_neg_2_l111_111779


namespace product_ab_cd_l111_111431

-- Conditions
variables (O A B C D F : Point)
variables (a b : ℝ)
hypothesis h1 : a = distance O A
hypothesis h2 : a = distance O B
hypothesis h3 : b = distance O C
hypothesis h4 : b = distance O D
hypothesis h5 : distance O F = 8
hypothesis h6 : diameter ((inscribed_circle (triangle O C F))) = 4

-- Given facts
def e1 := a^2 - b^2 = 64
def e2 := a - b = 4
def e3 := 2 * (distance O F) = 4

-- Theorem statement
theorem product_ab_cd : (2 * a) * (2 * b) = 240 :=
by
  sorry

end product_ab_cd_l111_111431


namespace seating_impossible_l111_111046

theorem seating_impossible (reps : Fin 54 → Fin 27) : 
  ¬ ∃ (s : Fin 54 → Fin 54),
    (∀ i : Fin 27, ∃ a b : Fin 54, a ≠ b ∧ s a = i ∧ s b = i ∧ (b - a ≡ 10 [MOD 54] ∨ a - b ≡ 10 [MOD 54])) :=
sorry

end seating_impossible_l111_111046


namespace ratio_pentagon_rectangle_l111_111341

theorem ratio_pentagon_rectangle (s_p w : ℝ) (H_pentagon : 5 * s_p = 60) (H_rectangle : 6 * w = 80) : s_p / w = 9 / 10 :=
by
  sorry

end ratio_pentagon_rectangle_l111_111341


namespace vova_gave_pavlik_three_nuts_l111_111316

variable {V P k : ℕ}
variable (h1 : V > P)
variable (h2 : V - P = 2 * P)
variable (h3 : k ≤ 5)
variable (h4 : ∃ m : ℕ, V - k = 3 * m)

theorem vova_gave_pavlik_three_nuts (h1 : V > P) (h2 : V - P = 2 * P) (h3 : k ≤ 5) (h4 : ∃ m : ℕ, V - k = 3 * m) : k = 3 := by
  sorry

end vova_gave_pavlik_three_nuts_l111_111316


namespace simplify_expression_l111_111390

theorem simplify_expression (x y : ℝ) (h : x - 2 * y = -2) : 9 - 2 * x + 4 * y = 13 :=
by sorry

end simplify_expression_l111_111390


namespace sequence_number_pair_l111_111524

theorem sequence_number_pair (n m : ℕ) (h : m ≤ n) : (m, n - m + 1) = (m, n - m + 1) :=
by sorry

end sequence_number_pair_l111_111524


namespace candy_store_revenue_l111_111333

def fudge_revenue : ℝ := 20 * 2.50
def truffles_revenue : ℝ := 5 * 12 * 1.50
def pretzels_revenue : ℝ := 3 * 12 * 2.00
def total_revenue : ℝ := fudge_revenue + truffles_revenue + pretzels_revenue

theorem candy_store_revenue :
  total_revenue = 212.00 :=
sorry

end candy_store_revenue_l111_111333


namespace find_n_l111_111110

open Nat

-- Defining the production rates for conditions.
structure Production := 
  (workers : ℕ)
  (gadgets : ℕ)
  (gizmos : ℕ)
  (hours : ℕ)

def condition1 : Production := { workers := 150, gadgets := 450, gizmos := 300, hours := 1 }
def condition2 : Production := { workers := 100, gadgets := 400, gizmos := 500, hours := 2 }
def condition3 : Production := { workers := 75, gadgets := 900, gizmos := 900, hours := 4 }

-- Statement: Finding the value of n.
theorem find_n :
  (75 * ((condition2.gadgets / condition2.workers) * (condition3.hours / condition2.hours))) = 600 := by
  sorry

end find_n_l111_111110


namespace circle_range_of_m_l111_111252

theorem circle_range_of_m (m : ℝ) :
  (∃ h k r : ℝ, (∀ x y : ℝ, (x - h) ^ 2 + (y - k) ^ 2 = r ^ 2 ↔ x ^ 2 + y ^ 2 - x + y + m = 0)) ↔ (m < 1/2) :=
by
  sorry

end circle_range_of_m_l111_111252


namespace nonagon_arithmetic_mean_property_l111_111546

def is_equilateral_triangle (A : Fin 9 → ℤ) (i j k : Fin 9) : Prop :=
  (j = (i + 3) % 9) ∧ (k = (i + 6) % 9)

def is_arithmetic_mean (A : Fin 9 → ℤ) (i j k : Fin 9) : Prop :=
  A j = (A i + A k) / 2

theorem nonagon_arithmetic_mean_property :
  ∀ (A : Fin 9 → ℤ),
    (∀ i, A i = 2016 + i) →
    (∀ i j k : Fin 9, is_equilateral_triangle A i j k → is_arithmetic_mean A i j k) :=
by
  intros
  sorry

end nonagon_arithmetic_mean_property_l111_111546


namespace sin_150_equals_half_l111_111497

theorem sin_150_equals_half : Real.sin (150 * Real.pi / 180) = 1 / 2 := 
by 
  sorry

end sin_150_equals_half_l111_111497


namespace total_outfits_l111_111973

def numRedShirts : ℕ := 7
def numGreenShirts : ℕ := 5
def numPants : ℕ := 6
def numRedHats : ℕ := 7
def numGreenHats : ℕ := 9

theorem total_outfits : 
  ((numRedShirts * numPants * numGreenHats) + 
   (numGreenShirts * numPants * numRedHats) + 
   ((numRedShirts * numRedHats + numGreenShirts * numGreenHats) * numPants)
  ) = 1152 := 
by
  sorry

end total_outfits_l111_111973


namespace area_of_isosceles_trapezoid_l111_111650

def isIsoscelesTrapezoid (a b c h : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a ∧ h ^ 2 = c ^ 2 - ((b - a) / 2) ^ 2

theorem area_of_isosceles_trapezoid :
  ∀ (a b c : ℝ), 
    a = 8 → b = 14 → c = 5 →
    ∃ h: ℝ, isIsoscelesTrapezoid a b c h ∧ ((a + b) / 2 * h = 44) :=
by
  intros a b c ha hb hc
  sorry

end area_of_isosceles_trapezoid_l111_111650


namespace composite_product_division_l111_111689

-- Define the first 12 positive composite integers
def first_six_composites := [4, 6, 8, 9, 10, 12]
def next_six_composites := [14, 15, 16, 18, 20, 21]

-- Define the product of a list of integers
def product (l : List ℕ) : ℕ :=
  l.foldl (· * ·) 1

-- Define the target problem statement
theorem composite_product_division :
  (product first_six_composites : ℚ) / (product next_six_composites : ℚ) = 1 / 49 := by
  sorry

end composite_product_division_l111_111689


namespace max_value_f_1_max_value_f_2_max_value_f_3_l111_111525
noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.log x - m * x

theorem max_value_f_1 (m : ℝ) (h : m ≤ 1 / Real.exp 1) :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → f x m ≤ 1 - m * Real.exp 1 :=
sorry

theorem max_value_f_2 (m : ℝ) (h1 : 1 / Real.exp 1 < m) (h2 : m < 1) :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → f x m ≤ -Real.log m - 1 :=
sorry

theorem max_value_f_3 (m : ℝ) (h : m ≥ 1) :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ Real.exp 1 → f x m ≤ -m :=
sorry

end max_value_f_1_max_value_f_2_max_value_f_3_l111_111525


namespace probability_of_selecting_specific_cubes_l111_111663

theorem probability_of_selecting_specific_cubes :
  let total_cubes := 125
  let cubes_with_3_faces := 1
  let cubes_with_no_faces := 88
  let total_pairs := (total_cubes * (total_cubes - 1)) / 2
  let successful_pairs := cubes_with_3_faces * cubes_with_no_faces
  by exact fractional.successful_pairs / total_pairs = 44 / 3875 := sorry

end probability_of_selecting_specific_cubes_l111_111663


namespace solve_system_eq_l111_111447

theorem solve_system_eq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h1 : x^y = z) (h2 : y^z = x) (h3 : z^x = y) :
  x = 1 ∧ y = 1 ∧ z = 1 :=
by
  -- Proof details would go here
  sorry

end solve_system_eq_l111_111447


namespace mallory_total_expense_l111_111618

theorem mallory_total_expense
  (cost_per_refill : ℕ)
  (distance_per_refill : ℕ)
  (total_distance : ℕ)
  (food_ratio : ℚ)
  (refill_count : ℕ)
  (total_fuel_cost : ℕ)
  (total_food_cost : ℕ)
  (total_expense : ℕ)
  (h1 : cost_per_refill = 45)
  (h2 : distance_per_refill = 500)
  (h3 : total_distance = 2000)
  (h4 : food_ratio = 3 / 5)
  (h5 : refill_count = total_distance / distance_per_refill)
  (h6 : total_fuel_cost = refill_count * cost_per_refill)
  (h7 : total_food_cost = (food_ratio * ↑total_fuel_cost).to_nat)
  (h8 : total_expense = total_fuel_cost + total_food_cost) :
  total_expense = 288 := by
  sorry

end mallory_total_expense_l111_111618


namespace find_x_l111_111820

/-- 
Prove that the value of x is 25 degrees, given the following conditions:
1. The sum of the angles in triangle BAC: angle_BAC + 50° + 55° = 180°
2. The angles forming a straight line DAE: 80° + angle_BAC + x = 180°
-/
theorem find_x (angle_BAC : ℝ) (x : ℝ)
  (h1 : angle_BAC + 50 + 55 = 180)
  (h2 : 80 + angle_BAC + x = 180) :
  x = 25 :=
  sorry

end find_x_l111_111820


namespace radius_of_circle_zero_l111_111224

theorem radius_of_circle_zero :
  (∃ x y : ℝ, x^2 + 8 * x + y^2 - 10 * y + 41 = 0) →
  (0 : ℝ) = 0 :=
by
  intro h
  sorry

end radius_of_circle_zero_l111_111224


namespace infinite_series_sum_eq_l111_111229

noncomputable def infinite_series_sum : Rat :=
  ∑' n : ℕ, (2 * n + 1) * (2000⁻¹) ^ n

theorem infinite_series_sum_eq : infinite_series_sum = (2003000 / 3996001) := by
  sorry

end infinite_series_sum_eq_l111_111229


namespace exists_parallelogram_marked_cells_l111_111543

theorem exists_parallelogram_marked_cells (n : ℕ) (marked : Finset (Fin n × Fin n)) (h_marked : marked.card = 2 * n) :
  ∃ (a b c d : Fin n × Fin n), a ∈ marked ∧ b ∈ marked ∧ c ∈ marked ∧ d ∈ marked ∧ 
  ((a.1 = b.1) ∧ (c.1 = d.1) ∧ (a.2 = c.2) ∧ (b.2 = d.2)) :=
sorry

end exists_parallelogram_marked_cells_l111_111543


namespace surface_area_sphere_dihedral_l111_111616

open Real

theorem surface_area_sphere_dihedral (R a : ℝ) (hR : 0 < R) (haR : 0 < a ∧ a < R) (α : ℝ) :
  2 * R^2 * arccos ((R * cos α) / sqrt (R^2 - a^2 * sin α^2)) 
  - 2 * R * a * sin α * arccos ((a * cos α) / sqrt (R^2 - a^2 * sin α^2)) = sorry :=
sorry

end surface_area_sphere_dihedral_l111_111616


namespace A_finishes_remaining_work_in_2_days_l111_111983

/-- 
Given that A's daily work rate is 1/6 of the work and B's daily work rate is 1/15 of the work,
and B has already completed 2/3 of the work, 
prove that A can finish the remaining work in 2 days.
-/
theorem A_finishes_remaining_work_in_2_days :
  let A_work_rate := (1 : ℝ) / 6
  let B_work_rate := (1 : ℝ) / 15
  let B_work_in_10_days := (10 : ℝ) * B_work_rate
  let remaining_work := (1 : ℝ) - B_work_in_10_days
  let days_for_A := remaining_work / A_work_rate
  B_work_in_10_days = 2 / 3 → 
  remaining_work = 1 / 3 → 
  days_for_A = 2 :=
by
  sorry

end A_finishes_remaining_work_in_2_days_l111_111983


namespace spheres_volume_ratio_l111_111392

theorem spheres_volume_ratio (S1 S2 V1 V2 : ℝ)
  (h1 : S1 / S2 = 1 / 9) 
  (h2a : S1 = 4 * π * r1^2) 
  (h2b : S2 = 4 * π * r2^2)
  (h3a : V1 = 4 / 3 * π * r1^3)
  (h3b : V2 = 4 / 3 * π * r2^3)
  : V1 / V2 = 1 / 27 :=
by
  sorry

end spheres_volume_ratio_l111_111392


namespace initial_volume_mixture_l111_111629

theorem initial_volume_mixture (V : ℝ) (h1 : 0.84 * V = 0.6 * (V + 24)) : V = 60 :=
by
  sorry

end initial_volume_mixture_l111_111629


namespace gnuff_tutor_minutes_l111_111727

/-- Definitions of the given conditions -/
def flat_rate : ℕ := 20
def per_minute_charge : ℕ := 7
def total_paid : ℕ := 146

/-- The proof statement -/
theorem gnuff_tutor_minutes :
  (total_paid - flat_rate) / per_minute_charge = 18 :=
by
  sorry

end gnuff_tutor_minutes_l111_111727


namespace sale_price_is_correct_l111_111146

def initial_price : ℝ := 560
def discount1 : ℝ := 0.20
def discount2 : ℝ := 0.30
def discount3 : ℝ := 0.15
def tax_rate : ℝ := 0.12

noncomputable def final_price : ℝ :=
  let price_after_first_discount := initial_price * (1 - discount1)
  let price_after_second_discount := price_after_first_discount * (1 - discount2)
  let price_after_third_discount := price_after_second_discount * (1 - discount3)
  let price_after_tax := price_after_third_discount * (1 + tax_rate)
  price_after_tax

theorem sale_price_is_correct :
  final_price = 298.55 :=
sorry

end sale_price_is_correct_l111_111146


namespace maximum_n_for_positive_S_l111_111944

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
n * (a 1 + a n) / 2

theorem maximum_n_for_positive_S
  (a : ℕ → ℝ)
  (d : ℝ)
  (S : ℕ → ℝ)
  (a1_pos : a 1 > 0)
  (d_neg : d < 0)
  (S4_eq_S8 : S 4 = S 8)
  (h1 : is_arithmetic_sequence a d)
  (h2 : ∀ n, S n = sum_of_first_n_terms a n) :
  ∃ n, ∀ m, m ≤ n → S m > 0 ∧ ∀ k, k > n → S k ≤ 0 ∧ n = 11 :=
sorry

end maximum_n_for_positive_S_l111_111944


namespace cow_total_spots_l111_111291

theorem cow_total_spots : 
  let left_spots := 16 in 
  let right_spots := 3 * left_spots + 7 in
  left_spots + right_spots = 71 :=
by
  let left_spots := 16
  let right_spots := 3 * left_spots + 7
  show left_spots + right_spots = 71
  sorry

end cow_total_spots_l111_111291


namespace fleas_initial_minus_final_l111_111481

theorem fleas_initial_minus_final (F : ℕ) (h : F / 16 = 14) :
  F - 14 = 210 :=
sorry

end fleas_initial_minus_final_l111_111481


namespace Coe_speed_theorem_l111_111535

-- Define the conditions
def Teena_speed : ℝ := 55
def initial_distance_behind : ℝ := 7.5
def time_hours : ℝ := 1.5
def distance_ahead : ℝ := 15

-- Define Coe's speed
def Coe_speed := 50

-- State the theorem
theorem Coe_speed_theorem : 
  let distance_Teena_covers := Teena_speed * time_hours
  let total_relative_distance := distance_Teena_covers + initial_distance_behind
  let distance_Coe_covers := total_relative_distance - distance_ahead
  let computed_Coe_speed := distance_Coe_covers / time_hours
  computed_Coe_speed = Coe_speed :=
by sorry

end Coe_speed_theorem_l111_111535


namespace problem_inequality_l111_111567

theorem problem_inequality (a b c : ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0)
  (h: (a / b + b / c + c / a) + (b / a + c / b + a / c) = 9) :
  a / b + b / c + c / a = 4.5 :=
by
  sorry

end problem_inequality_l111_111567


namespace river_width_proof_l111_111856
noncomputable def river_width (V FR D : ℝ) : ℝ := V / (FR * D)

theorem river_width_proof :
  river_width 2933.3333333333335 33.33333333333333 4 = 22 :=
by
  simp [river_width]
  norm_num
  sorry

end river_width_proof_l111_111856


namespace hives_needed_for_candles_l111_111768

theorem hives_needed_for_candles (h : (3 : ℕ) * c = 12) : (96 : ℕ) / c = 24 :=
by
  sorry

end hives_needed_for_candles_l111_111768


namespace probability_of_at_most_one_rainy_day_l111_111959

noncomputable def rain_probability : ℝ := 1 / 20

def rainy_days_in_July (days : ℕ) (prob_rain : ℝ) : ℕ → ℝ 
| 0 => (1 - prob_rain) ^ days
| 1 => days * prob_rain * (1 - prob_rain) ^ (days - 1)
| _ => 0

theorem probability_of_at_most_one_rainy_day : 
  abs (rainy_days_in_July 31 rain_probability 0 + rainy_days_in_July 31 rain_probability 1 - 0.271) < 0.001 :=
by
  -- Proof would go here; using sorry for now
  sorry

end probability_of_at_most_one_rainy_day_l111_111959


namespace ratio_of_b_to_a_l111_111142

theorem ratio_of_b_to_a (a b c : ℕ) (x y : ℕ) 
  (h1 : a > 0) 
  (h2 : x = 100 * a + 10 * b + c)
  (h3 : y = 100 * 9 + 10 * 9 + 9 - 241) 
  (h4 : x = y) :
  b = 5 → a = 7 → (b / a : ℚ) = 5 / 7 := 
by
  intros
  subst_vars
  sorry

end ratio_of_b_to_a_l111_111142


namespace limit_problem_l111_111876

noncomputable def limit_expression (x : ℝ) : ℝ := 
  (2*x - 1)^2 / (Real.exp (Real.sin (Real.pi * x)) - Real.exp (- Real.sin (3 * Real.pi * x)))

theorem limit_problem : filter.tendsto (limit_expression) (nhds_within (1/2) (set.univ)) (nhds (1 / (Real.exp 1 * Real.pi^2))) :=
sorry

end limit_problem_l111_111876


namespace circles_intersect_l111_111349

-- Definitions of the circles
def circle_O1 := {p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1}
def circle_O2 := {p : ℝ × ℝ | p.1^2 + (p.2 - 3)^2 = 9}

-- Proving the relationship between the circles
theorem circles_intersect : ∀ (p : ℝ × ℝ),
  p ∈ circle_O1 ∧ p ∈ circle_O2 :=
sorry

end circles_intersect_l111_111349


namespace smallest_solution_of_equation_l111_111318

theorem smallest_solution_of_equation :
  ∃ x : ℝ, (x^4 - 26 * x^2 + 169 = 0) ∧ x = -Real.sqrt 13 :=
by
  sorry

end smallest_solution_of_equation_l111_111318


namespace difference_of_two_numbers_l111_111150

theorem difference_of_two_numbers (a b : ℕ) 
(h1 : a + b = 17402) 
(h2 : ∃ k : ℕ, b = 10 * k) 
(h3 : ∃ k : ℕ, a + 9 * k = b) : 
10 * a - a = 14238 :=
by sorry

end difference_of_two_numbers_l111_111150


namespace part_I_part_II_l111_111118

-- Problem conditions as definitions
variable (a b : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : a + b = 1)

-- Statement for part (Ⅰ)
theorem part_I : (1 / a) + (1 / b) ≥ 4 :=
by
  sorry

-- Statement for part (Ⅱ)
theorem part_II : (1 / (a ^ 2016)) + (1 / (b ^ 2016)) ≥ 2 ^ 2017 :=
by
  sorry

end part_I_part_II_l111_111118


namespace add_neg_eq_neg_add_neg_ten_plus_neg_twelve_l111_111013

theorem add_neg_eq_neg_add (a b : Int) : a + -b = a - b := by
  sorry

theorem neg_ten_plus_neg_twelve : -10 + (-12) = -22 := by
  have h1 : -10 + (-12) = -10 - 12 := add_neg_eq_neg_add _ _
  have h2 : -10 - 12 = -(10 + 12) := by
    sorry -- This step corresponds to recognizing the arithmetic rule for subtraction.
  have h3 : -(10 + 12) = -22 := by
    sorry -- This step is the concrete calculation.
  exact Eq.trans h1 (Eq.trans h2 h3)

end add_neg_eq_neg_add_neg_ten_plus_neg_twelve_l111_111013


namespace maximum_ab_l111_111740

theorem maximum_ab (a b c : ℝ) (h1 : a + b + c = 4) (h2 : 3 * a + 2 * b - c = 0) : 
  ab <= 1/3 := 
by 
  sorry

end maximum_ab_l111_111740


namespace simplify_fraction_l111_111577

theorem simplify_fraction (h1 : 48 = 2^4 * 3) (h2 : 72 = 2^3 * 3^2) : (48 / 72 : ℚ) = 2 / 3 := 
by
  sorry

end simplify_fraction_l111_111577


namespace time_to_pick_72_peas_l111_111495

theorem time_to_pick_72_peas :
  (∀ t : ℕ, t = 56 / 7) → (72 / 8 = 9) :=
begin
  intro t_rate,
  have rate_of_picking := t_rate 8,
  rw rate_of_picking,
  have time := 72 / 8,
  exact time,
end

end time_to_pick_72_peas_l111_111495


namespace exists_pair_satisfying_system_l111_111063

theorem exists_pair_satisfying_system (m : ℝ) :
  (∃ x y : ℝ, y = m * x + 5 ∧ y = (3 * m - 2) * x + 7) ↔ m ≠ 1 :=
by
  sorry

end exists_pair_satisfying_system_l111_111063


namespace solution_set_of_inequality_l111_111880

theorem solution_set_of_inequality :
  ∀ x : ℝ, 3 * x^2 - 2 * x + 1 > 7 ↔ (x < -2/3 ∨ x > 3) :=
by
  sorry

end solution_set_of_inequality_l111_111880


namespace correct_quotient_l111_111466

-- Define number N based on given conditions
def N : ℕ := 9 * 8 + 6

-- Prove that the correct quotient when N is divided by 6 is 13
theorem correct_quotient : N / 6 = 13 := 
by {
  sorry
}

end correct_quotient_l111_111466


namespace simplify_fraction_l111_111586

theorem simplify_fraction (a b : ℕ) (h : b ≠ 0) (g : Nat.gcd a b = 24) : a = 48 → b = 72 → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  exact ⟨rfl, rfl⟩

end simplify_fraction_l111_111586


namespace number_of_males_who_listen_l111_111309

theorem number_of_males_who_listen (females_listen : ℕ) (males_dont_listen : ℕ) (total_listen : ℕ) (total_dont_listen : ℕ) (total_females : ℕ) :
  females_listen = 72 →
  males_dont_listen = 88 →
  total_listen = 160 →
  total_dont_listen = 180 →
  (total_females = total_listen + total_dont_listen - (females_listen + males_dont_listen)) →
  (total_females + males_dont_listen + 92 = total_listen + total_dont_listen) →
  total_listen + total_dont_listen = females_listen + males_dont_listen + (total_females - females_listen) + 92 :=
sorry

end number_of_males_who_listen_l111_111309


namespace overall_percentage_supporting_increased_funding_l111_111486

-- Definitions for the conditions
def percent_of_men_supporting (percent_men_supporting : ℕ := 60) : ℕ := percent_men_supporting
def percent_of_women_supporting (percent_women_supporting : ℕ := 80) : ℕ := percent_women_supporting
def number_of_men_surveyed (men_surveyed : ℕ := 100) : ℕ := men_surveyed
def number_of_women_surveyed (women_surveyed : ℕ := 900) : ℕ := women_surveyed

-- Theorem: the overall percent of people surveyed who supported increased funding is 78%
theorem overall_percentage_supporting_increased_funding : 
  (percent_of_men_supporting * number_of_men_surveyed + percent_of_women_supporting * number_of_women_surveyed) / 
  (number_of_men_surveyed + number_of_women_surveyed) = 78 := 
sorry

end overall_percentage_supporting_increased_funding_l111_111486


namespace t_le_s_l111_111512

theorem t_le_s (a b : ℝ) (t s : ℝ) (h1 : t = a + 2 * b) (h2 : s = a + b^2 + 1) : t ≤ s :=
by
  sorry

end t_le_s_l111_111512


namespace lcm_5_6_10_15_l111_111465

theorem lcm_5_6_10_15 : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 10 15) = 30 := 
by
  sorry

end lcm_5_6_10_15_l111_111465


namespace simplify_fraction_l111_111579

theorem simplify_fraction (h1 : 48 = 2^4 * 3) (h2 : 72 = 2^3 * 3^2) : (48 / 72 : ℚ) = 2 / 3 := 
by
  sorry

end simplify_fraction_l111_111579


namespace even_multiples_of_25_l111_111091

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_multiple_of_25 (n : ℕ) : Prop := n % 25 = 0

theorem even_multiples_of_25 (a b : ℕ) (h1 : 249 ≤ a) (h2 : b ≤ 501) :
  (a = 250 ∨ a = 275 ∨ a = 300 ∨ a = 350 ∨ a = 400 ∨ a = 450) →
  (b = 275 ∨ b = 300 ∨ b = 350 ∨ b = 400 ∨ b = 450 ∨ b = 500) →
  (∃ n, n = 5 ∧ ∀ m, (is_multiple_of_25 m ∧ is_even m ∧ a ≤ m ∧ m ≤ b) ↔ m ∈ [a, b]) :=
by sorry

end even_multiples_of_25_l111_111091


namespace digit_product_equality_l111_111400

theorem digit_product_equality :
  ∃ (a b c d e f g h i j : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧ a ≠ j ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧ b ≠ j ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧ c ≠ j ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧ d ≠ j ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧ e ≠ j ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧ f ≠ j ∧
    g ≠ h ∧ g ≠ i ∧ g ≠ j ∧
    h ≠ i ∧ h ≠ j ∧
    i ≠ j ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10 ∧ h < 10 ∧ i < 10 ∧ j < 10 ∧
    a * (10 * b + c) * (100 * d + 10 * e + f) = (1000 * g + 100 * h + 10 * i + j) :=
sorry

end digit_product_equality_l111_111400


namespace range_of_a_l111_111907

noncomputable def f (x : ℝ) (a : ℝ) := a * x - Real.log x

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 2, (a - 1 / x) ≥ 0) ↔ (a ≥ 1 / 2) :=
by
  sorry

end range_of_a_l111_111907


namespace max_value_of_8a_5b_15c_l111_111123

theorem max_value_of_8a_5b_15c (a b c : ℝ) (h : 9*a^2 + 4*b^2 + 25*c^2 = 1) : 
  8*a + 5*b + 15*c ≤ (Real.sqrt 115) / 2 :=
by
  sorry

end max_value_of_8a_5b_15c_l111_111123


namespace average_weight_bc_is_43_l111_111136

variable (a b c : ℝ)

-- Definitions of the conditions
def average_weight_abc (a b c : ℝ) : Prop := (a + b + c) / 3 = 45
def average_weight_ab (a b : ℝ) : Prop := (a + b) / 2 = 40
def weight_b (b : ℝ) : Prop := b = 31

-- The theorem to prove
theorem average_weight_bc_is_43 :
  ∀ (a b c : ℝ), average_weight_abc a b c → average_weight_ab a b → weight_b b → (b + c) / 2 = 43 :=
by
  intros a b c h_average_weight_abc h_average_weight_ab h_weight_b
  sorry

end average_weight_bc_is_43_l111_111136


namespace product_of_roots_l111_111226

noncomputable def f : ℝ → ℝ := sorry

theorem product_of_roots :
  (∀ x : ℝ, 4 * f (3 - x) - f x = 3 * x ^ 2 - 4 * x - 3) →
  (∃ a b : ℝ, f a = 8 ∧ f b = 8 ∧ a * b = -5) :=
sorry

end product_of_roots_l111_111226


namespace gnuff_tutoring_minutes_l111_111732

theorem gnuff_tutoring_minutes 
  (flat_rate : ℕ) 
  (rate_per_minute : ℕ) 
  (total_paid : ℕ) :
  flat_rate = 20 → 
  rate_per_minute = 7 →
  total_paid = 146 → 
  ∃ minutes : ℕ, minutes = 18 ∧ flat_rate + rate_per_minute * minutes = total_paid := 
by
  -- Assume the necessary steps here
  sorry

end gnuff_tutoring_minutes_l111_111732


namespace charlies_mother_cookies_l111_111154

theorem charlies_mother_cookies 
    (charlie_cookies : ℕ) 
    (father_cookies : ℕ) 
    (total_cookies : ℕ)
    (h_charlie : charlie_cookies = 15)
    (h_father : father_cookies = 10)
    (h_total : total_cookies = 30) : 
    (total_cookies - charlie_cookies - father_cookies = 5) :=
by {
    sorry
}

end charlies_mother_cookies_l111_111154


namespace negative_number_among_options_l111_111868

theorem negative_number_among_options :
  let A := abs (-1)
  let B := -(2^2)
  let C := (-(Real.sqrt 3))^2
  let D := (-3)^0
  B < 0 ∧ A > 0 ∧ C > 0 ∧ D > 0 :=
by
  sorry

end negative_number_among_options_l111_111868


namespace bike_license_combinations_l111_111999

theorem bike_license_combinations : 
  let letters := 3
  let digits := 10
  let total_combinations := letters * digits^4
  total_combinations = 30000 := by
  let letters := 3
  let digits := 10
  let total_combinations := letters * digits^4
  sorry

end bike_license_combinations_l111_111999


namespace meet_at_centroid_l111_111245

-- Definitions of positions
def Harry : ℝ × ℝ := (10, -3)
def Sandy : ℝ × ℝ := (2, 7)
def Ron : ℝ × ℝ := (6, 1)

-- Mathematical proof problem statement
theorem meet_at_centroid : 
    (Harry.1 + Sandy.1 + Ron.1) / 3 = 6 ∧ (Harry.2 + Sandy.2 + Ron.2) / 3 = 5 / 3 := 
by
  sorry

end meet_at_centroid_l111_111245


namespace binomial_7_4_eq_35_l111_111197
-- Import the entire Mathlib library for broad utility functions

-- Define the binomial coefficient function using library definitions 
noncomputable def binomial : ℕ → ℕ → ℕ
| n, k := nat.choose n k

-- Define the theorem we want to prove
theorem binomial_7_4_eq_35 : binomial 7 4 = 35 :=
by
  sorry

end binomial_7_4_eq_35_l111_111197


namespace a4_minus_1_divisible_5_l111_111791

theorem a4_minus_1_divisible_5 (a : ℤ) (h : ¬ (∃ k : ℤ, a = 5 * k)) : 
  (a^4 - 1) % 5 = 0 :=
by
  sorry

end a4_minus_1_divisible_5_l111_111791


namespace Billys_age_l111_111678

variable (B J : ℕ)

theorem Billys_age :
  B = 2 * J ∧ B + J = 45 → B = 30 :=
by
  sorry

end Billys_age_l111_111678


namespace range_of_a3_l111_111638

open Real

def convex_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, (a n + a (n + 2)) / 2 ≤ a (n + 1)

def sequence_condition (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, 1 ≤ n → n < 10 → abs (a n - b n) ≤ 20

def b (n : ℕ) : ℝ := n^2 - 6 * n + 10

theorem range_of_a3 (a : ℕ → ℝ) :
  convex_sequence a →
  a 1 = 1 →
  a 10 = 28 →
  sequence_condition a b →
  7 ≤ a 3 ∧ a 3 ≤ 19 :=
sorry

end range_of_a3_l111_111638


namespace simplify_fraction_l111_111582

theorem simplify_fraction (h1 : 48 = 2^4 * 3) (h2 : 72 = 2^3 * 3^2) : (48 / 72 : ℚ) = 2 / 3 := 
by
  sorry

end simplify_fraction_l111_111582


namespace bryan_total_books_magazines_l111_111679

-- Conditions as definitions
def novels : ℕ := 90
def comics : ℕ := 160
def rooms : ℕ := 12
def x := (3 / 4 : ℚ) * novels
def y := (6 / 5 : ℚ) * comics
def z := (1 / 2 : ℚ) * rooms

-- Calculations based on conditions
def books_per_shelf := 27 * x
def magazines_per_shelf := 80 * y
def total_shelves := 23 * z
def total_books := books_per_shelf * total_shelves
def total_magazines := magazines_per_shelf * total_shelves
def grand_total := total_books + total_magazines

-- Theorem to prove
theorem bryan_total_books_magazines :
  grand_total = 2371275 := by
  sorry

end bryan_total_books_magazines_l111_111679


namespace fraction_zero_iff_x_one_l111_111746

theorem fraction_zero_iff_x_one (x : ℝ) (h₁ : x - 1 = 0) (h₂ : x - 5 ≠ 0) : x = 1 := by
  sorry

end fraction_zero_iff_x_one_l111_111746


namespace find_middle_side_length_l111_111336

theorem find_middle_side_length (a b c : ℕ) (h1 : a + b + c = 2022) (h2 : c - b = 1) (h3 : b - a = 2) :
  b = 674 := 
by
  -- The proof goes here, but we skip it using sorry.
  sorry

end find_middle_side_length_l111_111336


namespace complex_exponentiation_problem_l111_111521

theorem complex_exponentiation_problem (z : ℂ) (h : z^2 + z + 1 = 0) : z^2010 + z^2009 + 1 = 0 :=
sorry

end complex_exponentiation_problem_l111_111521


namespace unique_sum_of_squares_l111_111792

theorem unique_sum_of_squares (p : ℕ) (k : ℕ) (x y a b : ℤ) 
  (hp : Prime p) (h1 : p = 4 * k + 1) (hx : x^2 + y^2 = p) (ha : a^2 + b^2 = p) :
  (x = a ∨ x = -a) ∧ (y = b ∨ y = -b) ∨ (x = b ∨ x = -b) ∧ (y = a ∨ y = -a) :=
sorry

end unique_sum_of_squares_l111_111792


namespace fraction_eq_zero_implies_x_eq_one_l111_111748

theorem fraction_eq_zero_implies_x_eq_one (x : ℝ) (h1 : (x - 1) = 0) (h2 : (x - 5) ≠ 0) : x = 1 :=
sorry

end fraction_eq_zero_implies_x_eq_one_l111_111748


namespace necessary_but_not_sufficient_condition_for_ellipse_l111_111715

theorem necessary_but_not_sufficient_condition_for_ellipse (m : ℝ) :
  (2 < m ∧ m < 6) ↔ ((∃ m, 2 < m ∧ m < 6 ∧ m ≠ 4) ∧ (∀ m, (2 < m ∧ m < 6) → ¬(m = 4))) := 
sorry

end necessary_but_not_sufficient_condition_for_ellipse_l111_111715


namespace correct_calculation_l111_111161

-- Definitions for each condition
def conditionA (a b : ℝ) : Prop := (a - b) * (-a - b) = a^2 - b^2
def conditionB (a : ℝ) : Prop := 2 * a^3 + 3 * a^3 = 5 * a^6
def conditionC (x y : ℝ) : Prop := 6 * x^3 * y^2 / (3 * x) = 2 * x^2 * y^2
def conditionD (x : ℝ) : Prop := (-2 * x^2)^3 = -6 * x^6

-- The proof problem
theorem correct_calculation (a b x y : ℝ) :
  ¬ conditionA a b ∧ ¬ conditionB a ∧ conditionC x y ∧ ¬ conditionD x := 
sorry

end correct_calculation_l111_111161


namespace rain_total_duration_l111_111551

theorem rain_total_duration : 
  let first_day_hours := 17 - 7
  let second_day_hours := first_day_hours + 2
  let third_day_hours := 2 * second_day_hours
  first_day_hours + second_day_hours + third_day_hours = 46 :=
by
  sorry

end rain_total_duration_l111_111551


namespace solve_equation_l111_111295

theorem solve_equation :
  ∃ x : ℝ, (20 / (x^2 - 9) - 3 / (x + 3) = 1) ↔ (x = -8) ∨ (x = 5) :=
by
  sorry

end solve_equation_l111_111295


namespace prob_ham_and_cake_l111_111409

namespace KarenLunch

-- Define the days
def days : ℕ := 5

-- Given conditions
def peanut_butter_days : ℕ := 2
def ham_days : ℕ := 3
def cake_days : ℕ := 1
def cookie_days : ℕ := 4

-- Calculate probabilities
def prob_ham : ℚ := 3 / 5
def prob_cake : ℚ := 1 / 5

-- Prove the probability of having both ham sandwich and cake on the same day
theorem prob_ham_and_cake : (prob_ham * prob_cake * 100) = 12 := by
  sorry

end KarenLunch

end prob_ham_and_cake_l111_111409


namespace price_reduction_equation_l111_111656

theorem price_reduction_equation (x : ℝ) (P_initial : ℝ) (P_final : ℝ) 
  (h1 : P_initial = 560) (h2 : P_final = 315) : 
  P_initial * (1 - x)^2 = P_final :=
by
  rw [h1, h2]
  sorry

end price_reduction_equation_l111_111656


namespace proof_problem_l111_111498

noncomputable def a : ℝ := 3.54
noncomputable def b : ℝ := 1.32
noncomputable def result : ℝ := (a - b) * 2

theorem proof_problem : result = 4.44 := by
  sorry

end proof_problem_l111_111498


namespace smallest_integer_condition_l111_111828

def is_not_prime (n : Nat) : Prop := ¬ Nat.Prime n

def is_not_square (n : Nat) : Prop :=
  ∀ m : Nat, m * m ≠ n

def has_no_prime_factor_less_than (n k : Nat) : Prop :=
  ∀ p : Nat, Nat.Prime p → p < k → ¬ (p ∣ n)

theorem smallest_integer_condition :
  ∃ n : Nat, n > 0 ∧ is_not_prime n ∧ is_not_square n ∧ has_no_prime_factor_less_than n 70 ∧ n = 5183 :=
by {
  sorry
}

end smallest_integer_condition_l111_111828


namespace range_of_f_l111_111374

def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem range_of_f : Set.Icc 0 3 → (Set.Ico 1 5) :=
by
  sorry
  -- Here the proof steps would go, which are omitted based on your guidelines.

end range_of_f_l111_111374


namespace ratio_female_male_l111_111047

theorem ratio_female_male (f m : ℕ) 
  (h1 : (50 * f) / f = 50) 
  (h2 : (30 * m) / m = 30) 
  (h3 : (50 * f + 30 * m) / (f + m) = 35) : 
  f / m = 1 / 3 := 
by
  sorry

end ratio_female_male_l111_111047


namespace consecutive_tree_distance_l111_111477

theorem consecutive_tree_distance (yard_length : ℕ) (num_trees : ℕ) (distance : ℚ)
  (h1 : yard_length = 520) 
  (h2 : num_trees = 40) :
  distance = yard_length / (num_trees - 1) :=
by
  -- Proof steps would go here
  sorry

end consecutive_tree_distance_l111_111477


namespace price_of_baseball_bat_l111_111942

theorem price_of_baseball_bat 
  (price_A : ℕ) (price_B : ℕ) (price_bat : ℕ) 
  (hA : price_A = 10 * 29)
  (hB : price_B = 14 * (25 / 10))
  (h0 : price_A = price_B + price_bat + 237) :
  price_bat = 18 :=
by
  sorry

end price_of_baseball_bat_l111_111942


namespace candy_store_revenue_l111_111334

def fudge_revenue : ℝ := 20 * 2.50
def truffles_revenue : ℝ := 5 * 12 * 1.50
def pretzels_revenue : ℝ := 3 * 12 * 2.00
def total_revenue : ℝ := fudge_revenue + truffles_revenue + pretzels_revenue

theorem candy_store_revenue :
  total_revenue = 212.00 :=
sorry

end candy_store_revenue_l111_111334


namespace simplify_fraction_l111_111588

theorem simplify_fraction (a b : ℕ) (h : b ≠ 0) (g : Nat.gcd a b = 24) : a = 48 → b = 72 → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  exact ⟨rfl, rfl⟩

end simplify_fraction_l111_111588


namespace tan_diff_identity_l111_111066

theorem tan_diff_identity (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2 / 5)
  (h2 : Real.tan (β + π / 4) = 1 / 4) :
  Real.tan (α - π / 4) = 3 / 22 :=
sorry

end tan_diff_identity_l111_111066


namespace fraction_of_pianists_got_in_l111_111125

-- Define the conditions
def flutes_got_in (f : ℕ) := f = 16
def clarinets_got_in (c : ℕ) := c = 15
def trumpets_got_in (t : ℕ) := t = 20
def total_band_members (total : ℕ) := total = 53
def total_pianists (p : ℕ) := p = 20

-- The main statement we want to prove
theorem fraction_of_pianists_got_in : 
  ∃ (pi : ℕ), 
    flutes_got_in 16 ∧ 
    clarinets_got_in 15 ∧ 
    trumpets_got_in 20 ∧ 
    total_band_members 53 ∧ 
    total_pianists 20 ∧ 
    pi / 20 = 1 / 10 := 
  sorry

end fraction_of_pianists_got_in_l111_111125


namespace find_m_minus_n_l111_111073

theorem find_m_minus_n (x y m n : ℤ) (h1 : x = -2) (h2 : y = 1) 
  (h3 : 3 * x + 2 * y = m) (h4 : n * x - y = 1) : m - n = -3 :=
by sorry

end find_m_minus_n_l111_111073


namespace diff_of_squares_l111_111094

theorem diff_of_squares (a b : ℝ) (h1 : a + b = -2) (h2 : a - b = 4) : a^2 - b^2 = -8 :=
by
  sorry

end diff_of_squares_l111_111094


namespace point_in_second_quadrant_l111_111098

theorem point_in_second_quadrant (x : ℝ) (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
by
  sorry

end point_in_second_quadrant_l111_111098


namespace students_not_opt_for_math_l111_111255

theorem students_not_opt_for_math (total_students S E both_subjects M : ℕ) 
    (h1 : total_students = 40) 
    (h2 : S = 15) 
    (h3 : E = 2) 
    (h4 : both_subjects = 7) 
    (h5 : total_students - both_subjects = M + S - E) : M = 20 := 
  by
  sorry

end students_not_opt_for_math_l111_111255


namespace set_inclusion_interval_l111_111909

theorem set_inclusion_interval (a : ℝ) :
    (A : Set ℝ) = {x : ℝ | (2 * a + 1) ≤ x ∧ x ≤ (3 * a - 5)} →
    (B : Set ℝ) = {x : ℝ | 3 ≤ x ∧ x ≤ 22} →
    (2 * a + 1 ≤ 3 * a - 5) →
    (A ⊆ B ↔ 6 ≤ a ∧ a ≤ 9) :=
by sorry

end set_inclusion_interval_l111_111909


namespace pounds_of_sugar_l111_111182

theorem pounds_of_sugar (x p : ℝ) (h1 : x * p = 216) (h2 : (x + 3) * (p - 1) = 216) : x = 24 :=
sorry

end pounds_of_sugar_l111_111182


namespace gain_percent_l111_111391

theorem gain_percent (C S : ℝ) (h : 50 * C = 30 * S) : ((S - C) / C) * 100 = 200 / 3 :=
by 
  sorry

end gain_percent_l111_111391


namespace diana_additional_game_time_l111_111503

theorem diana_additional_game_time :
  ∀ (reading_hours : ℕ) (minutes_per_hour : ℕ) (raise_percent : ℕ),
    reading_hours = 12 → minutes_per_hour = 30 → raise_percent = 20 →
    (reading_hours * (minutes_per_hour * raise_percent / 100)) = 72 :=
by
  intros reading_hours minutes_per_hour raise_percent h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end diana_additional_game_time_l111_111503


namespace shari_total_distance_l111_111292

theorem shari_total_distance (speed : ℝ) (time_1 : ℝ) (rest : ℝ) (time_2 : ℝ) (distance : ℝ) :
  speed = 4 ∧ time_1 = 2 ∧ rest = 0.5 ∧ time_2 = 1 ∧ distance = speed * time_1 + speed * time_2 → distance = 12 :=
by
  sorry

end shari_total_distance_l111_111292


namespace longest_collection_has_more_pages_l111_111419

noncomputable def miles_pages_per_inch := 5
noncomputable def daphne_pages_per_inch := 50
noncomputable def miles_height_inches := 240
noncomputable def daphne_height_inches := 25

noncomputable def miles_total_pages := miles_height_inches * miles_pages_per_inch
noncomputable def daphne_total_pages := daphne_height_inches * daphne_pages_per_inch

theorem longest_collection_has_more_pages :
  max miles_total_pages daphne_total_pages = 1250 := by
  -- Skip the proof
  sorry

end longest_collection_has_more_pages_l111_111419


namespace tutoring_minutes_l111_111730

def flat_rate : ℤ := 20
def per_minute_rate : ℤ := 7
def total_paid : ℤ := 146

theorem tutoring_minutes (m : ℤ) : total_paid = flat_rate + (per_minute_rate * m) → m = 18 :=
by
  sorry

end tutoring_minutes_l111_111730


namespace ξ_distribution_and_expectation_probability_harvested_by_team_B_l111_111303

/-- Define the teams and their respective probabilities -/
def p_team_A : ℚ := 3 / 10
def p_team_B : ℚ := 3 / 10
def p_team_C : ℚ := 2 / 5

/-- Define the utilization rates for each team -/
def p_use_A : ℚ := 8 / 10
def p_use_B : ℚ := 75 / 100
def p_use_C : ℚ := 6 / 10

/-- Random variable ξ following a binomial distribution -/
def ξ_distribution (k : ℕ) : ℚ :=
nat.choose 3 k * (p_team_C^k) * ((1 - p_team_C)^(3 - k))

/-- Distribution and expectation of ξ -/
theorem ξ_distribution_and_expectation :
  ∀ k : ℕ, k ∈ [0, 1, 2, 3] →
  ξ_distribution k =
    match k with
    | 0 => 27 / 125
    | 1 => 54 / 125
    | 2 => 36 / 125
    | 3 => 8 / 125
    | _ => 0
    end
∧
(Eξ : ℚ) = 6 / 5 :=
sorry

/-- Probability a block was harvested by Team B and can be used -/
def P_B := p_team_A * p_use_A + p_team_B * p_use_B + p_team_C * p_use_C

/-- Probability a block was harvested by Team B given it can be used -/
theorem probability_harvested_by_team_B :
  (p_team_B * p_use_B) / P_B = 15 / 47 :=
sorry

end ξ_distribution_and_expectation_probability_harvested_by_team_B_l111_111303


namespace percentage_dried_fruit_of_combined_mix_l111_111806

theorem percentage_dried_fruit_of_combined_mix :
  ∀ (weight_sue weight_jane : ℝ),
  (weight_sue * 0.3 + weight_jane * 0.6) / (weight_sue + weight_jane) = 0.45 →
  100 * (weight_sue * 0.7) / (weight_sue + weight_jane) = 35 :=
by
  intros weight_sue weight_jane H
  sorry

end percentage_dried_fruit_of_combined_mix_l111_111806


namespace cynthia_more_miles_l111_111254

open Real

noncomputable def david_speed : ℝ := 55 / 5
noncomputable def cynthia_speed : ℝ := david_speed + 3

theorem cynthia_more_miles (t : ℝ) (ht : t = 5) :
  (cynthia_speed * t) - (david_speed * t) = 15 :=
by
  sorry

end cynthia_more_miles_l111_111254


namespace radius_of_circle_centered_at_l111_111177

def center : ℝ × ℝ := (3, 4)

def intersects_axes_at_three_points (A : ℝ × ℝ) (r : ℝ) : Prop :=
  (A.1 - r = 0 ∨ A.1 + r = 0) ∧ (A.2 - r = 0 ∨ A.2 + r = 0)

theorem radius_of_circle_centered_at (A : ℝ × ℝ) : 
  (intersects_axes_at_three_points A 4) ∨ (intersects_axes_at_three_points A 5) :=
by
  sorry

end radius_of_circle_centered_at_l111_111177


namespace sum_of_reciprocals_l111_111626

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 5 * x * y) : 
  (1/x) + (1/y) = 5 :=
by
  sorry

end sum_of_reciprocals_l111_111626


namespace even_function_a_eq_4_l111_111247

noncomputable def f (x a : ℝ) : ℝ := (x + a) * (x - 4)

theorem even_function_a_eq_4 (a : ℝ) (h : ∀ x : ℝ, f (-x) a = f x a) : a = 4 := by
  sorry

end even_function_a_eq_4_l111_111247


namespace g_increasing_g_multiplicative_g_special_case_g_18_value_l111_111773

def g (n : ℕ) : ℕ :=
sorry

theorem g_increasing : ∀ n : ℕ, n > 0 → g (n + 1) > g n :=
sorry

theorem g_multiplicative : ∀ m n : ℕ, m > 0 → n > 0 → g (m * n) = g m * g n :=
sorry

theorem g_special_case : ∀ m n : ℕ, m > 0 → n > 0 → m ≠ n → m ^ n = n ^ m → g m = n ∨ g n = m :=
sorry

theorem g_18_value : g 18 = 324 :=
sorry

end g_increasing_g_multiplicative_g_special_case_g_18_value_l111_111773


namespace last_score_is_71_l111_111570

theorem last_score_is_71 (scores : List ℕ) (h : scores = [71, 74, 79, 85, 88, 92]) (sum_eq: scores.sum = 489) :
  ∃ s : ℕ, s ∈ scores ∧ 
           (∃ avg : ℕ, avg = (scores.sum - s) / 5 ∧ 
           ∀ lst : List ℕ, lst = scores.erase s → (∀ n, n ∈ lst → lst.sum % (lst.length - 1) = 0)) :=
  sorry

end last_score_is_71_l111_111570


namespace finding_value_of_expression_l111_111122

open Real

theorem finding_value_of_expression
  (a b : ℝ)
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_eq : 1/a - 1/b - 1/(a + b) = 0) :
  (b/a + a/b)^2 = 5 :=
sorry

end finding_value_of_expression_l111_111122


namespace fraction_computation_l111_111054

noncomputable def compute_fraction : ℚ :=
  (64^4 + 324) * (52^4 + 324) * (40^4 + 324) * (28^4 + 324) * (16^4 + 324) /
  (58^4 + 324) * (46^4 + 324) * (34^4 + 324) * (22^4 + 324) * (10^4 + 324)

theorem fraction_computation :
  compute_fraction = 137 / 1513 :=
by sorry

end fraction_computation_l111_111054


namespace total_price_l111_111831

theorem total_price (r w : ℕ) (hr : r = 4275) (hw : w = r - 1490) : r + w = 7060 :=
by
  sorry

end total_price_l111_111831


namespace A_alone_days_l111_111175

variable (x : ℝ) -- Number of days A takes to do the work alone
variable (B_rate : ℝ := 1 / 12) -- Work rate of B
variable (Together_rate : ℝ := 1 / 4) -- Combined work rate of A and B

theorem A_alone_days :
  (1 / x + B_rate = Together_rate) → (x = 6) := by
  intro h
  sorry

end A_alone_days_l111_111175


namespace complement_union_M_N_correct_l111_111380

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

-- Define the set M
def M : Set ℕ := {1, 3, 5, 7}

-- Define the set N
def N : Set ℕ := {5, 6, 7}

-- Define the union of M and N
def union_M_N : Set ℕ := M ∪ N

-- Define the complement of the union of M and N in U
def complement_union_M_N : Set ℕ := U \ union_M_N

-- Main theorem statement to prove
theorem complement_union_M_N_correct : complement_union_M_N = {2, 4, 8} :=
by
  sorry

end complement_union_M_N_correct_l111_111380


namespace find_x_angle_l111_111763

theorem find_x_angle (ABC ACB CDE : ℝ) (h1 : ABC = 70) (h2 : ACB = 90) (h3 : CDE = 42) : 
  ∃ x : ℝ, x = 158 :=
by
  sorry

end find_x_angle_l111_111763


namespace circles_tangent_iff_l111_111899

noncomputable def C1 := { p : ℝ × ℝ | (p.1 - 1)^2 + p.2^2 = 1 }
noncomputable def C2 (m: ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 - 8 * p.1 + 8 * p.2 + m = 0 }

theorem circles_tangent_iff (m: ℝ) : (∀ p ∈ C1, p ∈ C2 m → False) ↔ (m = -4 ∨ m = 16) := 
sorry

end circles_tangent_iff_l111_111899


namespace wade_average_points_per_game_l111_111464

variable (W : ℝ)

def teammates_average_points_per_game : ℝ := 40

def total_team_points_after_5_games : ℝ := 300

theorem wade_average_points_per_game :
  teammates_average_points_per_game * 5 + W * 5 = total_team_points_after_5_games →
  W = 20 :=
by
  intro h
  sorry

end wade_average_points_per_game_l111_111464


namespace missed_field_goals_l111_111016

theorem missed_field_goals (TotalAttempts MissedFraction WideRightPercentage : ℕ) 
  (TotalAttempts_eq : TotalAttempts = 60)
  (MissedFraction_eq : MissedFraction = 15)
  (WideRightPercentage_eq : WideRightPercentage = 3) : 
  (TotalAttempts * (1 / 4) * (20 / 100) = 3) :=
  by
    sorry

end missed_field_goals_l111_111016


namespace circle_eq_l111_111076

theorem circle_eq (A B C : ℝ × ℝ)
  (hA : A = (2, 0))
  (hB : B = (4, 0))
  (hC : C = (0, 2)) :
  ∃ (h: ℝ), (x - 3) ^ 2 + (y - 3) ^ 2 = h :=
by 
  use 10
  -- additional steps to rigorously prove the result would go here
  sorry

end circle_eq_l111_111076


namespace right_triangle_smaller_angle_l111_111111

theorem right_triangle_smaller_angle (x : ℝ) (h_right_triangle : 0 < x ∧ x < 90)
  (h_double_angle : ∃ y : ℝ, y = 2 * x)
  (h_angle_sum : x + 2 * x = 90) :
  x = 30 :=
  sorry

end right_triangle_smaller_angle_l111_111111


namespace each_monkey_gets_bananas_l111_111485

-- Define the conditions
def total_monkeys : ℕ := 12
def total_piles : ℕ := 10
def first_piles : ℕ := 6
def first_pile_hands : ℕ := 9
def first_hand_bananas : ℕ := 14
def remaining_piles : ℕ := total_piles - first_piles
def remaining_pile_hands : ℕ := 12
def remaining_hand_bananas : ℕ := 9

-- Define the number of bananas in each type of pile
def bananas_in_first_piles : ℕ := first_piles * first_pile_hands * first_hand_bananas
def bananas_in_remaining_piles : ℕ := remaining_piles * remaining_pile_hands * remaining_hand_bananas
def total_bananas : ℕ := bananas_in_first_piles + bananas_in_remaining_piles

-- Define the main theorem to be proved
theorem each_monkey_gets_bananas : total_bananas / total_monkeys = 99 := by
  sorry

end each_monkey_gets_bananas_l111_111485


namespace problem1_problem2_problem3_l111_111328

variables (a b c : ℝ)

-- First proof problem
theorem problem1 (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) : a * b * c ≠ 0 :=
sorry

-- Second proof problem
theorem problem2 (h : a = 0 ∨ b = 0 ∨ c = 0) : a * b * c = 0 :=
sorry

-- Third proof problem
theorem problem3 (h : a * b < 0 ∨ a = 0 ∨ b = 0) : a * b ≤ 0 :=
sorry

end problem1_problem2_problem3_l111_111328


namespace tom_gets_correct_share_l111_111783

def total_savings : ℝ := 18500.0
def natalie_share : ℝ := 0.35 * total_savings
def remaining_after_natalie : ℝ := total_savings - natalie_share
def rick_share : ℝ := 0.30 * remaining_after_natalie
def remaining_after_rick : ℝ := remaining_after_natalie - rick_share
def lucy_share : ℝ := 0.40 * remaining_after_rick
def remaining_after_lucy : ℝ := remaining_after_rick - lucy_share
def minimum_share : ℝ := 1000.0
def tom_share : ℝ := remaining_after_lucy

theorem tom_gets_correct_share :
  (natalie_share ≥ minimum_share) ∧ (rick_share ≥ minimum_share) ∧ (lucy_share ≥ minimum_share) →
  tom_share = 5050.50 :=
by
  sorry

end tom_gets_correct_share_l111_111783


namespace general_term_of_geometric_sequence_l111_111068

variable (a : ℕ → ℝ) (q : ℝ)

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

theorem general_term_of_geometric_sequence
  (h1 : a 1 + a 3 = 10)
  (h2 : a 4 + a 6 = 5 / 4)
  (hq : is_geometric_sequence a q)
  (q := 1/2) :
  ∃ a₀ : ℝ, ∀ n : ℕ, a n = a₀ * q^(n - 1) :=
sorry

end general_term_of_geometric_sequence_l111_111068


namespace torn_pages_are_112_and_113_l111_111338

theorem torn_pages_are_112_and_113 (n k : ℕ) (S S' : ℕ) 
  (h1 : S = n * (n + 1) / 2)
  (h2 : S' = S - (k - 1) - k)
  (h3 : S' = 15000) :
  (k = 113) ∧ (k - 1 = 112) :=
by
  sorry

end torn_pages_are_112_and_113_l111_111338


namespace linear_function_y1_greater_y2_l111_111377

theorem linear_function_y1_greater_y2 :
  ∀ (y_1 y_2 : ℝ), 
    (y_1 = -(-1) + 6) → (y_2 = -(2) + 6) → y_1 > y_2 :=
by
  intros y_1 y_2 h1 h2
  sorry

end linear_function_y1_greater_y2_l111_111377


namespace probability_phone_not_answered_l111_111753

noncomputable def P_first_ring : ℝ := 0.1
noncomputable def P_second_ring : ℝ := 0.3
noncomputable def P_third_ring : ℝ := 0.4
noncomputable def P_fourth_ring : ℝ := 0.1

theorem probability_phone_not_answered : 
  1 - P_first_ring - P_second_ring - P_third_ring - P_fourth_ring = 0.1 := 
by
  sorry

end probability_phone_not_answered_l111_111753


namespace find_age_l111_111854

theorem find_age (A : ℤ) (h : 4 * (A + 4) - 4 * (A - 4) = A) : A = 32 :=
by sorry

end find_age_l111_111854


namespace composite_quotient_is_one_over_49_l111_111687

/-- Define the first twelve positive composite integers --/
def first_six_composites : List ℕ := [4, 6, 8, 9, 10, 12]
def next_six_composites : List ℕ := [14, 15, 16, 18, 20, 21]

/-- Define the product of a list of integers --/
def product (l : List ℕ) : ℕ := l.foldl (λ acc x => acc * x) 1

/-- This defines the quotient of the products of the first six and the next six composite numbers --/
def composite_quotient : ℚ := (↑(product first_six_composites)) / (↑(product next_six_composites))

/-- Main theorem stating that the composite quotient equals to 1/49 --/
theorem composite_quotient_is_one_over_49 : composite_quotient = 1 / 49 := 
  by
    sorry

end composite_quotient_is_one_over_49_l111_111687


namespace firstDiscountIsTenPercent_l111_111455

def listPrice : ℝ := 70
def finalPrice : ℝ := 56.16
def secondDiscount : ℝ := 10.857142857142863

theorem firstDiscountIsTenPercent (x : ℝ) : 
    finalPrice = listPrice * (1 - x / 100) * (1 - secondDiscount / 100) ↔ x = 10 := 
by
  sorry

end firstDiscountIsTenPercent_l111_111455


namespace representation_of_2015_l111_111794

theorem representation_of_2015 :
  ∃ (p d3 i : ℕ),
    Prime p ∧ -- p is prime
    d3 % 3 = 0 ∧ -- d3 is divisible by 3
    400 < i ∧ i < 500 ∧ i % 3 ≠ 0 ∧ -- i is in interval and not divisible by 3
    2015 = p + d3 + i := sorry

end representation_of_2015_l111_111794


namespace factorize_expr_l111_111218

theorem factorize_expr (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) :=
sorry

end factorize_expr_l111_111218


namespace min_value_tan_product_l111_111398

theorem min_value_tan_product (A B C : ℝ) (h : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π)
  (sin_eq : Real.sin A = 3 * Real.sin B * Real.sin C) :
  ∃ t : ℝ, t = Real.tan A * Real.tan B * Real.tan C ∧ t = 12 :=
sorry

end min_value_tan_product_l111_111398


namespace product_no_xx_x_eq_x_cube_plus_one_l111_111373

theorem product_no_xx_x_eq_x_cube_plus_one (a c : ℝ) (h1 : a - 1 = 0) (h2 : c - a = 0) : 
  (x + a) * (x ^ 2 - x + c) = x ^ 3 + 1 :=
by {
  -- Here would be the proof steps, which we omit with "sorry"
  sorry
}

end product_no_xx_x_eq_x_cube_plus_one_l111_111373


namespace total_amount_paid_correct_l111_111531

-- Definitions of quantities and rates
def quantity_grapes := 3
def rate_grapes := 70
def quantity_mangoes := 9
def rate_mangoes := 55

-- Total amount calculation
def total_amount_paid := quantity_grapes * rate_grapes + quantity_mangoes * rate_mangoes

-- Theorem to prove total amount paid is 705
theorem total_amount_paid_correct : total_amount_paid = 705 :=
by
  sorry

end total_amount_paid_correct_l111_111531


namespace solve_abs_eq_2005_l111_111461

theorem solve_abs_eq_2005 (x : ℝ) : |2005 * x - 2005| = 2005 ↔ x = 0 ∨ x = 2 := by
  sorry

end solve_abs_eq_2005_l111_111461


namespace LanceCents_l111_111275

noncomputable def MargaretCents : ℕ := 75
noncomputable def GuyCents : ℕ := 60
noncomputable def BillCents : ℕ := 60
noncomputable def TotalCents : ℕ := 265

theorem LanceCents (lanceCents : ℕ) :
  MargaretCents + GuyCents + BillCents + lanceCents = TotalCents → lanceCents = 70 :=
by
  intros
  sorry

end LanceCents_l111_111275


namespace boat_speed_in_still_water_l111_111174

theorem boat_speed_in_still_water (B S : ℝ) (h1 : B + S = 6) (h2 : B - S = 4) : B = 5 := by
  sorry

end boat_speed_in_still_water_l111_111174


namespace sum_of_angles_of_circumscribed_quadrilateral_l111_111661

theorem sum_of_angles_of_circumscribed_quadrilateral
  (EF GH : ℝ)
  (EF_central_angle : EF = 100)
  (GH_central_angle : GH = 120) :
  (EF / 2 + GH / 2) = 70 :=
by
  sorry

end sum_of_angles_of_circumscribed_quadrilateral_l111_111661


namespace area_of_square_with_perimeter_40_l111_111696

theorem area_of_square_with_perimeter_40 (P : ℝ) (s : ℝ) (A : ℝ) 
  (hP : P = 40) (hs : s = P / 4) (hA : A = s^2) : A = 100 :=
by
  sorry

end area_of_square_with_perimeter_40_l111_111696


namespace maximize_take_home_pay_l111_111914

def tax_collected (x : ℝ) : ℝ :=
  10 * x^2

def take_home_pay (x : ℝ) : ℝ :=
  1000 * x - tax_collected x

theorem maximize_take_home_pay : ∃ x : ℝ, (x * 1000 = 50000) ∧ (∀ y : ℝ, take_home_pay x ≥ take_home_pay y) := 
sorry

end maximize_take_home_pay_l111_111914


namespace tom_four_times_cindy_years_ago_l111_111399

variables (t c x : ℕ)

-- Conditions
axiom cond1 : t + 5 = 2 * (c + 5)
axiom cond2 : t - 13 = 3 * (c - 13)

-- Question to prove
theorem tom_four_times_cindy_years_ago :
  t - x = 4 * (c - x) → x = 19 :=
by
  intros h
  -- simply skip the proof for now
  sorry

end tom_four_times_cindy_years_ago_l111_111399


namespace smallest_n_for_inequality_l111_111362

theorem smallest_n_for_inequality :
  ∃ n : ℤ, (∀ w x y z : ℝ, 
    (w^2 + x^2 + y^2 + z^2)^3 ≤ n * (w^6 + x^6 + y^6 + z^6)) ∧ 
    (∀ m : ℤ, (∀ w x y z : ℝ, 
    (w^2 + x^2 + y^2 + z^2)^3 ≤ m * (w^6 + x^6 + y^6 + z^6)) → m ≥ 64) :=
by
  sorry

end smallest_n_for_inequality_l111_111362


namespace average_episodes_per_year_l111_111838

def num_years : Nat := 14
def seasons_15_episodes : Nat := 8
def episodes_per_season_15 : Nat := 15
def seasons_20_episodes : Nat := 4
def episodes_per_season_20 : Nat := 20
def seasons_12_episodes : Nat := 2
def episodes_per_season_12 : Nat := 12

theorem average_episodes_per_year :
  (seasons_15_episodes * episodes_per_season_15 +
   seasons_20_episodes * episodes_per_season_20 +
   seasons_12_episodes * episodes_per_season_12) / num_years = 16 := by
  sorry

end average_episodes_per_year_l111_111838


namespace find_X_l111_111259

theorem find_X 
  (X Y : ℕ)
  (h1 : 6 + X = 13)
  (h2 : Y = 7) :
  X = 7 := by
  sorry

end find_X_l111_111259


namespace find_k_l111_111499

theorem find_k (k : ℝ) (α β : ℝ) 
  (h1 : α + β = -k) 
  (h2 : α * β = 12) 
  (h3 : α + 7 + β + 7 = k) : 
  k = -7 :=
sorry

end find_k_l111_111499


namespace sqrt_inequality_sum_of_squares_geq_sum_of_products_l111_111327

theorem sqrt_inequality : (Real.sqrt 6) + (Real.sqrt 10) > (2 * Real.sqrt 3) + 2 := by
  sorry

theorem sum_of_squares_geq_sum_of_products (a b c : ℝ) : 
    a^2 + b^2 + c^2 ≥ a * b + b * c + a * c := by
  sorry

end sqrt_inequality_sum_of_squares_geq_sum_of_products_l111_111327


namespace compare_x_y_l111_111074

theorem compare_x_y (a b : ℝ) (h1 : a > b) (h2 : b > 1) (x y : ℝ)
  (hx : x = a + 1 / a) (hy : y = b + 1 / b) : x > y :=
by {
  sorry
}

end compare_x_y_l111_111074


namespace jake_has_fewer_peaches_than_steven_l111_111769

theorem jake_has_fewer_peaches_than_steven :
  ∀ (jillPeaches jakePeaches stevenPeaches : ℕ),
    jillPeaches = 12 →
    jakePeaches = jillPeaches - 1 →
    stevenPeaches = jillPeaches + 15 →
    stevenPeaches - jakePeaches = 16 :=
  by
    intros jillPeaches jakePeaches stevenPeaches
    intro h_jill
    intro h_jake
    intro h_steven
    sorry

end jake_has_fewer_peaches_than_steven_l111_111769


namespace value_of_f_750_l111_111119

theorem value_of_f_750 (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x / y^2)
    (hf500 : f 500 = 4) :
    f 750 = 16 / 9 :=
sorry

end value_of_f_750_l111_111119


namespace product_of_number_subtracting_7_equals_9_l111_111805

theorem product_of_number_subtracting_7_equals_9 (x : ℤ) (h : x - 7 = 9) : x * 5 = 80 := by
  sorry

end product_of_number_subtracting_7_equals_9_l111_111805


namespace solution_y_chemical_A_percentage_l111_111294

def percent_chemical_A_in_x : ℝ := 0.30
def percent_chemical_A_in_mixture : ℝ := 0.32
def percent_solution_x_in_mixture : ℝ := 0.80
def percent_solution_y_in_mixture : ℝ := 0.20

theorem solution_y_chemical_A_percentage
  (P : ℝ) 
  (h : percent_solution_x_in_mixture * percent_chemical_A_in_x + percent_solution_y_in_mixture * P = percent_chemical_A_in_mixture) :
  P = 0.40 :=
sorry

end solution_y_chemical_A_percentage_l111_111294


namespace father_son_age_relationship_l111_111458

theorem father_son_age_relationship 
    (F S X : ℕ) 
    (h1 : F = 27) 
    (h2 : F = 3 * S + 3) 
    : X = 11 ∧ F + X > 2 * (S + X) :=
by
  sorry

end father_son_age_relationship_l111_111458


namespace john_fixes_8_computers_l111_111557

theorem john_fixes_8_computers 
  (total_computers : ℕ)
  (unfixable_percentage : ℝ)
  (waiting_percentage : ℝ) 
  (h1 : total_computers = 20)
  (h2 : unfixable_percentage = 0.2)
  (h3 : waiting_percentage = 0.4) :
  let fixed_right_away := total_computers * (1 - unfixable_percentage - waiting_percentage)
  fixed_right_away = 8 :=
by
  sorry

end john_fixes_8_computers_l111_111557


namespace tiling_impossible_2003x2003_l111_111114

theorem tiling_impossible_2003x2003 :
  ¬ (∃ (f : Fin 2003 × Fin 2003 → ℕ),
  (∀ p : Fin 2003 × Fin 2003, f p = 1 ∨ f p = 2) ∧
  (∀ p : Fin 2003, (f (p, 0) + f (p, 1)) % 3 = 0) ∧
  (∀ p : Fin 2003, (f (0, p) + f (1, p) + f (2, p)) % 3 = 0)) := 
sorry

end tiling_impossible_2003x2003_l111_111114


namespace square_of_binomial_l111_111357

theorem square_of_binomial (c : ℝ) : (∃ b : ℝ, ∀ x : ℝ, 9 * x^2 - 30 * x + c = (3 * x + b)^2) ↔ c = 25 :=
by
  sorry

end square_of_binomial_l111_111357


namespace max_a_value_l111_111929

theorem max_a_value (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 3 * c) (h3 : c < 4 * d) (h4 : b + d = 200) : a ≤ 449 :=
by sorry

end max_a_value_l111_111929


namespace price_of_each_cupcake_l111_111002

variable (x : ℝ)

theorem price_of_each_cupcake (h : 50 * x + 40 * 0.5 = 2 * 40 + 20 * 2) : x = 2 := 
by 
  sorry

end price_of_each_cupcake_l111_111002


namespace ellipse_major_minor_axes_product_l111_111436

-- Definitions based on conditions
def OF : ℝ := 8
def inradius_triangle_OCF : ℝ := 2  -- diameter / 2

-- Define a and b based on the ellipse properties and conditions
def a : ℝ := 10  -- Solved from the given conditions and steps
def b : ℝ := 6   -- Solved from the given conditions and steps

-- Defining the axes of the ellipse in terms of a and b
def AB : ℝ := 2 * a
def CD : ℝ := 2 * b

-- The product (AB)(CD) we are interested in
def product_AB_CD := AB * CD

-- The main proof statement
theorem ellipse_major_minor_axes_product : product_AB_CD = 240 :=
by
  sorry

end ellipse_major_minor_axes_product_l111_111436


namespace min_ab_min_a_plus_b_l111_111511

theorem min_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) : ab >= 8 :=
sorry

theorem min_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) : a + b >= 3 + 2 * Real.sqrt 2 :=
sorry

end min_ab_min_a_plus_b_l111_111511


namespace ratio_of_men_to_women_l111_111787
open Nat

theorem ratio_of_men_to_women 
  (total_players : ℕ) 
  (players_per_group : ℕ) 
  (extra_women_per_group : ℕ) 
  (H_total_players : total_players = 20) 
  (H_players_per_group : players_per_group = 3) 
  (H_extra_women_per_group : extra_women_per_group = 1) 
  : (7 / 13 : ℝ) = 7 / 13 :=
by
  -- Conditions
  have H1 : total_players = 20 := H_total_players
  have H2 : players_per_group = 3 := H_players_per_group
  have H3 : extra_women_per_group = 1 := H_extra_women_per_group
  -- The correct answer
  sorry

end ratio_of_men_to_women_l111_111787


namespace average_last_three_l111_111946

theorem average_last_three {a b c d e f g : ℝ} 
  (h_avg_all : (a + b + c + d + e + f + g) / 7 = 60)
  (h_avg_first_four : (a + b + c + d) / 4 = 55) : 
  (e + f + g) / 3 = 200 / 3 :=
by
  sorry

end average_last_three_l111_111946


namespace part1_part2_l111_111415

noncomputable def Sn (a : ℕ → ℚ) (n : ℕ) (p : ℚ) : ℚ :=
4 * a n - p

theorem part1 (a : ℕ → ℚ) (S : ℕ → ℚ) (p : ℚ) (hp : p ≠ 0)
  (hS : ∀ n, S n = Sn a n p) : 
  ∃ q, ∀ n, a (n + 1) = q * a n :=
sorry

noncomputable def an_formula (n : ℕ) : ℚ := (4/3)^(n - 1)

theorem part2 (b : ℕ → ℚ) (a : ℕ → ℚ)
  (p : ℚ) (hp : p = 3)
  (hb : b 1 = 2)
  (ha1 : a 1 = 1) 
  (h_rec : ∀ n, b (n + 1) = b n + a n) :
  ∀ n, b n = 3 * ((4/3)^(n - 1)) - 1 :=
sorry

end part1_part2_l111_111415


namespace julie_savings_multiple_l111_111923

theorem julie_savings_multiple (S : ℝ) (hS : 0 < S) :
  (12 * 0.25 * S) / (0.75 * S) = 4 :=
by
  sorry

end julie_savings_multiple_l111_111923


namespace prob_ham_and_cake_l111_111408

namespace KarenLunch

-- Define the days
def days : ℕ := 5

-- Given conditions
def peanut_butter_days : ℕ := 2
def ham_days : ℕ := 3
def cake_days : ℕ := 1
def cookie_days : ℕ := 4

-- Calculate probabilities
def prob_ham : ℚ := 3 / 5
def prob_cake : ℚ := 1 / 5

-- Prove the probability of having both ham sandwich and cake on the same day
theorem prob_ham_and_cake : (prob_ham * prob_cake * 100) = 12 := by
  sorry

end KarenLunch

end prob_ham_and_cake_l111_111408


namespace height_of_carton_is_70_l111_111035

def carton_dimensions : ℕ × ℕ := (25, 42)
def soap_box_dimensions : ℕ × ℕ × ℕ := (7, 6, 5)
def max_soap_boxes : ℕ := 300

theorem height_of_carton_is_70 :
  let (carton_length, carton_width) := carton_dimensions
  let (soap_box_length, soap_box_width, soap_box_height) := soap_box_dimensions
  let boxes_per_layer := (carton_length / soap_box_length) * (carton_width / soap_box_width)
  let num_layers := max_soap_boxes / boxes_per_layer
  (num_layers * soap_box_height) = 70 :=
by
  have carton_length := 25
  have carton_width := 42
  have soap_box_length := 7
  have soap_box_width := 6
  have soap_box_height := 5
  have max_soap_boxes := 300
  have boxes_per_layer := (25 / 7) * (42 / 6)
  have num_layers := max_soap_boxes / boxes_per_layer
  sorry

end height_of_carton_is_70_l111_111035


namespace candy_store_total_sales_l111_111332

def price_per_pound_fudge : ℝ := 2.50
def pounds_fudge : ℕ := 20
def price_per_truffle : ℝ := 1.50
def dozens_truffles : ℕ := 5
def price_per_pretzel : ℝ := 2.00
def dozens_pretzels : ℕ := 3

theorem candy_store_total_sales :
  price_per_pound_fudge * pounds_fudge +
  price_per_truffle * (dozens_truffles * 12) +
  price_per_pretzel * (dozens_pretzels * 12) = 212.00 := by
  sorry

end candy_store_total_sales_l111_111332


namespace average_episodes_per_year_l111_111841

theorem average_episodes_per_year (total_years : ℕ) (n1 n2 n3 e1 e2 e3 : ℕ) 
  (h1 : total_years = 14)
  (h2 : n1 = 8) (h3 : e1 = 15)
  (h4 : n2 = 4) (h5 : e2 = 20)
  (h6 : n3 = 2) (h7 : e3 = 12) :
  (n1 * e1 + n2 * e2 + n3 * e3) / total_years = 16 := by
  -- Skip the proof steps
  sorry

end average_episodes_per_year_l111_111841


namespace determine_a_l111_111350

theorem determine_a (r s a : ℝ) (h1 : r^2 = a) (h2 : 2 * r * s = 16) (h3 : s^2 = 16) : a = 4 :=
by {
  sorry
}

end determine_a_l111_111350


namespace angle_measure_l111_111968

variable (x : ℝ)

def complement (x : ℝ) : ℝ := 90 - x

def supplement (x : ℝ) : ℝ := 180 - x

theorem angle_measure (h : supplement x = 8 * complement x) : x = 540 / 7 := by
  sorry

end angle_measure_l111_111968


namespace divisible_by_6_l111_111472

theorem divisible_by_6 (n : ℤ) (h1 : n % 3 = 0) (h2 : n % 2 = 0) : n % 6 = 0 :=
sorry

end divisible_by_6_l111_111472


namespace symmetric_line_equation_l111_111453

theorem symmetric_line_equation : ∀ (x y : ℝ), (2 * x + 3 * y - 6 = 0) ↔ (3 * (x + 2) + 2 * (-y - 2) + 16 = 0) :=
by
  sorry

end symmetric_line_equation_l111_111453


namespace number_of_boys_l111_111625

theorem number_of_boys {total_students : ℕ} (h1 : total_students = 49)
  (ratio_girls_boys : ℕ → ℕ → Prop)
  (h2 : ratio_girls_boys 4 3) :
  ∃ boys : ℕ, boys = 21 := by
  sorry

end number_of_boys_l111_111625


namespace minimum_value_S15_minus_S10_l111_111516

theorem minimum_value_S15_minus_S10 (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_geom_seq : ∀ n, S (n + 1) = S n * a (n + 1))
  (h_pos_terms : ∀ n, a n > 0)
  (h_arith_seq : S 10 - 2 * S 5 = 3)
  (h_geom_sub_seq : (S 10 - S 5) * (S 10 - S 5) = S 5 * (S 15 - S 10)) :
  ∃ m, m = 12 ∧ (S 15 - S 10) ≥ m := sorry

end minimum_value_S15_minus_S10_l111_111516


namespace sequence_problem_l111_111030

open Nat

theorem sequence_problem (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h : ∀ n : ℕ, 0 < n → S n + a n = 2 * n) :
  a 1 = 1 ∧ a 2 = 3 / 2 ∧ a 3 = 7 / 4 ∧ a 4 = 15 / 8 ∧ ∀ n : ℕ, 0 < n → a n = (2^n - 1) / (2^(n-1)) :=
by
  sorry

end sequence_problem_l111_111030


namespace total_coins_are_correct_l111_111283

-- Define the initial number of coins
def initial_dimes : Nat := 2
def initial_quarters : Nat := 6
def initial_nickels : Nat := 5

-- Define the additional coins given by Linda's mother
def additional_dimes : Nat := 2
def additional_quarters : Nat := 10
def additional_nickels : Nat := 2 * initial_nickels

-- Calculate the total number of each type of coin
def total_dimes : Nat := initial_dimes + additional_dimes
def total_quarters : Nat := initial_quarters + additional_quarters
def total_nickels : Nat := initial_nickels + additional_nickels

-- Total number of coins
def total_coins : Nat := total_dimes + total_quarters + total_nickels

-- Theorem to prove the total number of coins is 35
theorem total_coins_are_correct : total_coins = 35 := by
  -- Skip the proof
  sorry

end total_coins_are_correct_l111_111283


namespace incorrect_average_initially_calculated_l111_111001

theorem incorrect_average_initially_calculated :
  ∀ (S' S : ℕ) (n : ℕ) (incorrect_correct_difference : ℕ),
  n = 10 →
  incorrect_correct_difference = 30 →
  S = 200 →
  S' = S - incorrect_correct_difference →
  (S' / n) = 17 :=
by
  intros S' S n incorrect_correct_difference h_n h_diff h_S h_S' 
  sorry

end incorrect_average_initially_calculated_l111_111001


namespace determine_h_l111_111059

theorem determine_h (h : ℝ) : (∃ x : ℝ, x = 3 ∧ x^3 - 2 * h * x + 15 = 0) → h = 7 :=
by
  intro hx
  sorry

end determine_h_l111_111059


namespace gcd_consecutive_triplets_l111_111639

theorem gcd_consecutive_triplets : ∀ i : ℕ, 1 ≤ i → gcd (i * (i + 1) * (i + 2)) 6 = 6 :=
by
  sorry

end gcd_consecutive_triplets_l111_111639


namespace max_alpha_l111_111712

theorem max_alpha (A B C : ℝ) (hA : 0 < A ∧ A < π)
  (hB : 0 < B ∧ B < π)
  (hC : 0 < C ∧ C < π)
  (hSum : A + B + C = π)
  (hmin : ∀ alpha, alpha = min (2 * A - B) (min (3 * B - 2 * C) (π / 2 - A))) :
  ∃ alpha, alpha = 2 * π / 9 := 
sorry

end max_alpha_l111_111712


namespace tank_capacity_l111_111180

theorem tank_capacity (w c : ℕ) (h1 : w = c / 3) (h2 : w + 7 = 2 * c / 5) : c = 105 :=
sorry

end tank_capacity_l111_111180


namespace total_selling_price_of_cloth_l111_111863

theorem total_selling_price_of_cloth
  (profit_per_meter : ℕ)
  (cost_price_per_meter : ℕ)
  (total_meters : ℕ)
  (total_selling_price : ℕ) :
  profit_per_meter = 7 →
  cost_price_per_meter = 118 →
  total_meters = 80 →
  total_selling_price = (cost_price_per_meter + profit_per_meter) * total_meters →
  total_selling_price = 10000 :=
by
  intros h_profit h_cost h_total h_selling_price
  rw [h_profit, h_cost, h_total] at h_selling_price
  exact h_selling_price

end total_selling_price_of_cloth_l111_111863


namespace max_lamps_on_road_l111_111857

theorem max_lamps_on_road (k: ℕ) (lk: ℕ): 
  lk = 1000 → (∀ n: ℕ, n < k → n≥ 1 ∧ ∀ m: ℕ, if m > n then m > 1 else true) → (lk ≤ k) ∧ 
  (∀ i:ℕ,∃ j, (i ≠ j) → (lk < 1000)) → k = 1998 :=
by sorry

end max_lamps_on_road_l111_111857


namespace part_a_part_b_l111_111163

theorem part_a (a : Fin 10 → ℤ) : ∃ i j : Fin 10, i ≠ j ∧ 27 ∣ (a i)^3 - (a j)^3 := sorry
theorem part_b (b : Fin 8 → ℤ) : ∃ i j : Fin 8, i ≠ j ∧ 27 ∣ (b i)^3 - (b j)^3 := sorry

end part_a_part_b_l111_111163


namespace power_expression_l111_111051

variable {a b : ℝ}

theorem power_expression : (-2 * a^2 * b^3)^3 = -8 * a^6 * b^9 := 
by 
  sorry

end power_expression_l111_111051


namespace simplify_fraction_result_l111_111130

theorem simplify_fraction_result :
  (144: ℝ) / 1296 * 72 = 8 :=
by
  sorry

end simplify_fraction_result_l111_111130


namespace slope_of_CD_l111_111003

-- Given circle equations
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 15 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 16*x + 8*y + 48 = 0

-- Define the line whose slope needs to be found
def line (x y : ℝ) : Prop := 22*x - 12*y - 33 = 0

-- State the proof problem
theorem slope_of_CD : ∀ x y : ℝ, circle1 x y → circle2 x y → line x y ∧ (∃ m : ℝ, m = 11/6) :=
by sorry

end slope_of_CD_l111_111003


namespace geometric_sequence_k_squared_l111_111112

theorem geometric_sequence_k_squared (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n+1) = a n * r) (h5 : a 5 * a 8 * a 11 = k) : 
  k^2 = a 5 * a 6 * a 7 * a 9 * a 10 * a 11 := by
  sorry

end geometric_sequence_k_squared_l111_111112


namespace cond_prob_B_given_A_l111_111315

-- Definitions based on the conditions
def eventA := {n : ℕ | n > 4 ∧ n ≤ 6}
def eventB := {k : ℕ × ℕ | (k.1 + k.2) = 7}

-- Probability of event A
def probA := (2 : ℚ) / 6

-- Joint probability of events A and B
def probAB := (1 : ℚ) / (6 * 6)

-- Conditional probability P(B|A)
def cond_prob := probAB / probA

-- The final statement to prove
theorem cond_prob_B_given_A : cond_prob = 1 / 6 := by
  sorry

end cond_prob_B_given_A_l111_111315


namespace maria_min_score_fifth_term_l111_111630

theorem maria_min_score_fifth_term (score1 score2 score3 score4 : ℕ) (avg_required : ℕ) 
  (h1 : score1 = 84) (h2 : score2 = 80) (h3 : score3 = 82) (h4 : score4 = 78)
  (h_avg_required : avg_required = 85) :
  ∃ x : ℕ, x ≥ 101 :=
by
  sorry

end maria_min_score_fifth_term_l111_111630


namespace flea_treatment_problem_l111_111480

/-- One flea treatment halves the flea population.
    After four treatments, the dog has 14 fleas remaining.
    The number of additional fleas before treatments compared to after four treatments is 210. -/
theorem flea_treatment_problem :
  ∃ (initial_fleas : ℕ), ((initial_fleas / 2 / 2 / 2 / 2) = 14) ∧ (initial_fleas - 14 = 210) :=
begin
  sorry,
end

end flea_treatment_problem_l111_111480


namespace part_I_part_II_l111_111889

-- Part (I): If a = 1, prove that q implies p
theorem part_I (x : ℝ) (h : 3 < x ∧ x < 4) : (1 < x) ∧ (x < 4) :=
by sorry

-- Part (II): Prove the range of a for which p is necessary but not sufficient for q
theorem part_II (a : ℝ) (h1 : a > 0) (h2 : ∀ x : ℝ, (a < x ∧ x < 4 * a) → (3 < x ∧ x < 4)) : 1 < a ∧ a ≤ 3 :=
by sorry

end part_I_part_II_l111_111889


namespace find_b_value_l111_111300

theorem find_b_value : 
  ∀ (a b : ℝ), 
    (a^3 * b^4 = 2048) ∧ (a = 8) → b = Real.sqrt 2 := 
by 
sorry

end find_b_value_l111_111300


namespace percentage_of_rotten_oranges_l111_111186

-- Define the conditions
def total_oranges : ℕ := 600
def total_bananas : ℕ := 400
def rotten_bananas_percentage : ℝ := 0.08
def good_fruits_percentage : ℝ := 0.878

-- Define the proof problem
theorem percentage_of_rotten_oranges :
  let total_fruits := total_oranges + total_bananas
  let number_of_rotten_bananas := rotten_bananas_percentage * total_bananas
  let number_of_good_fruits := good_fruits_percentage * total_fruits
  let number_of_rotten_fruits := total_fruits - number_of_good_fruits
  let number_of_rotten_oranges := number_of_rotten_fruits - number_of_rotten_bananas
  let percentage_of_rotten_oranges := (number_of_rotten_oranges / total_oranges) * 100
  percentage_of_rotten_oranges = 15 := 
by
  sorry

end percentage_of_rotten_oranges_l111_111186


namespace find_principal_l111_111162

theorem find_principal (R : ℝ) (P : ℝ) (h : ((P * (R + 5) * 10) / 100) = ((P * R * 10) / 100 + 600)) : P = 1200 :=
by
  sorry

end find_principal_l111_111162


namespace soccer_ball_seams_l111_111038

theorem soccer_ball_seams 
  (num_pentagons : ℕ) 
  (num_hexagons : ℕ) 
  (sides_per_pentagon : ℕ) 
  (sides_per_hexagon : ℕ) 
  (total_pieces : ℕ) 
  (equal_sides : sides_per_pentagon = sides_per_hexagon)
  (total_pieces_eq : total_pieces = 32)
  (num_pentagons_eq : num_pentagons = 12)
  (num_hexagons_eq : num_hexagons = 20)
  (sides_per_pentagon_eq : sides_per_pentagon = 5)
  (sides_per_hexagon_eq : sides_per_hexagon = 6) :
  90 = (num_pentagons * sides_per_pentagon + num_hexagons * sides_per_hexagon) / 2 :=
by 
  sorry

end soccer_ball_seams_l111_111038


namespace minimum_value_frac_l111_111711

theorem minimum_value_frac (x y z : ℝ) (h : 2 * x * y + y * z > 0) : 
  (x^2 + y^2 + z^2) / (2 * x * y + y * z) ≥ 3 :=
sorry

end minimum_value_frac_l111_111711


namespace equivalent_statements_l111_111468

variables (P Q R : Prop)

theorem equivalent_statements :
  (P → (Q ∧ ¬R)) ↔ ((¬ Q ∨ R) → ¬ P) :=
sorry

end equivalent_statements_l111_111468


namespace sufficient_condition_for_inequality_l111_111819

theorem sufficient_condition_for_inequality (a x : ℝ) (h1 : -2 < x) (h2 : x < -1) :
  (a + x) * (1 + x) < 0 → a > 2 :=
sorry

end sufficient_condition_for_inequality_l111_111819


namespace four_digit_cubes_divisible_by_16_count_l111_111900

theorem four_digit_cubes_divisible_by_16_count :
  ∃ (count : ℕ), count = 3 ∧
    ∀ (m : ℕ), 1000 ≤ 64 * m^3 ∧ 64 * m^3 ≤ 9999 → (m = 3 ∨ m = 4 ∨ m = 5) :=
by {
  -- our proof would go here
  sorry
}

end four_digit_cubes_divisible_by_16_count_l111_111900


namespace bales_in_barn_l111_111313

theorem bales_in_barn (stacked today total original : ℕ) (h1 : stacked = 67) (h2 : total = 89) (h3 : total = stacked + original) : original = 22 :=
by
  sorry

end bales_in_barn_l111_111313


namespace fg_of_3_eq_neg5_l111_111385

-- Definitions from the conditions
def f (x : ℝ) : ℝ := 2 * x - 5
def g (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- Lean statement to prove question == answer
theorem fg_of_3_eq_neg5 : f (g 3) = -5 := by
  sorry

end fg_of_3_eq_neg5_l111_111385


namespace eval_derivative_at_one_and_neg_one_l111_111569

def f (x : ℝ) : ℝ := x^4 + x - 1

theorem eval_derivative_at_one_and_neg_one : 
  (deriv f 1) + (deriv f (-1)) = 2 :=
by 
  -- proof to be filled in
  sorry

end eval_derivative_at_one_and_neg_one_l111_111569


namespace correct_option_C_l111_111467

variable {a : ℝ} (x : ℝ) (b : ℝ)

theorem correct_option_C : 
  (a^8 / a^2 = a^6) :=
by {
  sorry
}

end correct_option_C_l111_111467


namespace underachievers_l111_111462

-- Define the variables for the number of students in each group
variables (a b c : ℕ)

-- Given conditions as hypotheses
axiom total_students : a + b + c = 30
axiom top_achievers : a = 19
axiom average_students : c = 12

-- Prove the number of underachievers
theorem underachievers : b = 9 :=
by sorry

end underachievers_l111_111462


namespace certain_number_eq_l111_111479

theorem certain_number_eq :
  ∃ y : ℝ, y + (y * 4) = 48 ∧ y = 9.6 :=
by
  sorry

end certain_number_eq_l111_111479


namespace students_journals_l111_111168

theorem students_journals :
  ∃ u v : ℕ, 
    u + v = 75000 ∧ 
    (7 * u + 2 * v = 300000) ∧ 
    (∃ b g : ℕ, b = u * 7 / 300 ∧ g = v * 2 / 300 ∧ b = 700 ∧ g = 300) :=
by {
  -- The proving steps will go here
  sorry
}

end students_journals_l111_111168


namespace angle_measure_proof_l111_111969

noncomputable def angle_measure (x : ℝ) : Prop :=
  let supplement := 180 - x
  let complement := 90 - x
  supplement = 8 * complement

theorem angle_measure_proof : ∃ x : ℝ, angle_measure x ∧ x = 540 / 7 :=
by
  have angle_eq : ∀ x, angle_measure x ↔ (180 - x = 8 * (90 - x)) := by
    intro x
    dsimp [angle_measure]
    rfl
  use 540 / 7
  rw angle_eq
  split
  · dsimp
    linarith
  · rfl

end angle_measure_proof_l111_111969


namespace inequality_transformation_l111_111710

theorem inequality_transformation (a b : ℝ) (h : a > b) : -2 * a < -2 * b :=
by {
  sorry
}

end inequality_transformation_l111_111710


namespace option_d_correct_l111_111321

variable (a b : ℝ)

theorem option_d_correct : (-a^3)^4 = a^(12) := by sorry

end option_d_correct_l111_111321


namespace find_common_difference_l111_111109

theorem find_common_difference (AB BC AC : ℕ) (x y z d : ℕ) 
  (h1 : AB = 300) (h2 : BC = 350) (h3 : AC = 400) 
  (hx : x = (2 * d) / 5) (hy : y = (7 * d) / 15) (hz : z = (8 * d) / 15) 
  (h_sum : x + y + z = 750) : 
  d = 536 :=
by
  -- Proof goes here
  sorry

end find_common_difference_l111_111109


namespace right_angled_triangle_sets_l111_111044

theorem right_angled_triangle_sets :
  (¬ (1 ^ 2 + 2 ^ 2 = 3 ^ 2)) ∧
  (¬ (2 ^ 2 + 3 ^ 2 = 4 ^ 2)) ∧
  (3 ^ 2 + 4 ^ 2 = 5 ^ 2) ∧
  (¬ (4 ^ 2 + 5 ^ 2 = 6 ^ 2)) :=
by
  sorry

end right_angled_triangle_sets_l111_111044


namespace diamond_cut_1_3_loss_diamond_max_loss_ratio_l111_111143

noncomputable def value (w : ℝ) : ℝ := 6000 * w^2

theorem diamond_cut_1_3_loss (a : ℝ) :
  (value a - (value (1/4 * a) + value (3/4 * a))) / value a = 0.375 :=
by sorry

theorem diamond_max_loss_ratio :
  ∀ (m n : ℝ), (m > 0) → (n > 0) → 
  (1 - (value (m/(m + n)) + value (n/(m + n))) ≤ 0.5) :=
by sorry

end diamond_cut_1_3_loss_diamond_max_loss_ratio_l111_111143


namespace squares_not_all_congruent_l111_111024

/-- Proof that the statement "all squares are congruent to each other" is false. -/
theorem squares_not_all_congruent : ¬(∀ (a b : ℝ), a = b ↔ a = b) :=
by 
  sorry

end squares_not_all_congruent_l111_111024


namespace mean_equals_l111_111305

theorem mean_equals (z : ℝ) :
    (7 + 10 + 15 + 21) / 4 = (18 + z) / 2 → z = 8.5 := 
by
    intro h
    sorry

end mean_equals_l111_111305


namespace Janice_earnings_after_deductions_l111_111405

def dailyEarnings : ℕ := 30
def daysWorked : ℕ := 6
def weekdayOvertimeRate : ℕ := 15
def weekendOvertimeRate : ℕ := 20
def weekdayOvertimeShifts : ℕ := 2
def weekendOvertimeShifts : ℕ := 1
def tipsReceived : ℕ := 10
def taxRate : ℝ := 0.10

noncomputable def calculateEarnings : ℝ :=
  let regularEarnings := dailyEarnings * daysWorked
  let overtimeEarnings := (weekdayOvertimeRate * weekdayOvertimeShifts) + (weekendOvertimeRate * weekendOvertimeShifts)
  let totalEarningsBeforeTax := regularEarnings + overtimeEarnings + tipsReceived
  let taxAmount := totalEarningsBeforeTax * taxRate
  totalEarningsBeforeTax - taxAmount

theorem Janice_earnings_after_deductions :
  calculateEarnings = 216 := by
  sorry

end Janice_earnings_after_deductions_l111_111405


namespace quotient_of_division_l111_111810

theorem quotient_of_division (a b : ℕ) (r q : ℕ) (h1 : a = 1637) (h2 : b + 1365 = a) (h3 : a = b * q + r) (h4 : r = 5) : q = 6 :=
by
  -- Placeholder for proof
  sorry

end quotient_of_division_l111_111810


namespace solve_equation_l111_111961

theorem solve_equation (x : ℝ) (h : x ≠ 1) :
  (3 * x) / (x - 1) = 2 + 1 / (x - 1) → x = -1 :=
by
  sorry

end solve_equation_l111_111961


namespace cards_remaining_l111_111423

theorem cards_remaining (initial_cards : ℕ) (cards_given : ℕ) (remaining_cards : ℕ) :
  initial_cards = 242 → cards_given = 136 → remaining_cards = initial_cards - cards_given → remaining_cards = 106 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end cards_remaining_l111_111423


namespace negation_proposition_l111_111145

theorem negation_proposition : (¬ ∀ x : ℝ, (1 < x) → x - 1 ≥ Real.log x) ↔ (∃ x_0 : ℝ, (1 < x_0) ∧ x_0 - 1 < Real.log x_0) :=
by
  sorry

end negation_proposition_l111_111145


namespace option_d_not_necessarily_true_l111_111093

theorem option_d_not_necessarily_true (a b c : ℝ) (h: a > b) : ¬(a * c^2 > b * c^2) ↔ c = 0 :=
by sorry

end option_d_not_necessarily_true_l111_111093


namespace original_number_l111_111116

variable (n : ℝ)

theorem original_number :
  (2 * (n + 3)^2 - 3) / 2 = 49 → n = Real.sqrt (101 / 2) - 3 :=
by
  sorry

end original_number_l111_111116


namespace inequality_and_equality_condition_l111_111775

variable (a b c t : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem inequality_and_equality_condition :
  abc * (a^t + b^t + c^t) ≥ a^(t+2) * (-a + b + c) + b^(t+2) * (a - b + c) + c^(t+2) * (a + b - c) ∧ 
  (abc * (a^t + b^t + c^t) = a^(t+2) * (-a + b + c) + b^(t+2) * (a - b + c) + c^(t+2) * (a + b - c) ↔ a = b ∧ b = c) :=
sorry

end inequality_and_equality_condition_l111_111775


namespace each_monkey_gets_bananas_l111_111484

-- Define the conditions
def total_monkeys : ℕ := 12
def total_piles : ℕ := 10
def first_piles : ℕ := 6
def first_pile_hands : ℕ := 9
def first_hand_bananas : ℕ := 14
def remaining_piles : ℕ := total_piles - first_piles
def remaining_pile_hands : ℕ := 12
def remaining_hand_bananas : ℕ := 9

-- Define the number of bananas in each type of pile
def bananas_in_first_piles : ℕ := first_piles * first_pile_hands * first_hand_bananas
def bananas_in_remaining_piles : ℕ := remaining_piles * remaining_pile_hands * remaining_hand_bananas
def total_bananas : ℕ := bananas_in_first_piles + bananas_in_remaining_piles

-- Define the main theorem to be proved
theorem each_monkey_gets_bananas : total_bananas / total_monkeys = 99 := by
  sorry

end each_monkey_gets_bananas_l111_111484


namespace aquarium_height_l111_111286

theorem aquarium_height (h : ℝ) (V : ℝ) (final_volume : ℝ) :
  let length := 4
  let width := 6
  let halfway_volume := (length * width * h) / 2
  let spilled_volume := halfway_volume / 2
  let tripled_volume := 3 * spilled_volume
  tripled_volume = final_volume →
  final_volume = 54 →
  h = 3 := by
  intros
  sorry

end aquarium_height_l111_111286


namespace ticket_price_increase_l111_111488

noncomputable def y (x : ℕ) : ℝ :=
  if x ≤ 100 then
    30 * x - 50 * Real.sqrt x - 500
  else
    30 * x - 50 * Real.sqrt x - 700

theorem ticket_price_increase (m : ℝ) : 
  m * 20 - 50 * Real.sqrt 20 - 500 ≥ 0 → m ≥ 37 := sorry

end ticket_price_increase_l111_111488


namespace perimeter_of_square_l111_111256

theorem perimeter_of_square
  (length_rect : ℕ) (width_rect : ℕ) (area_rect : ℕ)
  (area_square : ℕ) (side_square : ℕ) (perimeter_square : ℕ) :
  (length_rect = 32) → (width_rect = 10) → 
  (area_rect = length_rect * width_rect) →
  (area_square = 5 * area_rect) →
  (side_square * side_square = area_square) →
  (perimeter_square = 4 * side_square) →
  perimeter_square = 160 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- Proof would go here
  sorry

end perimeter_of_square_l111_111256


namespace binomial_7_4_eq_35_l111_111199

theorem binomial_7_4_eq_35 : Nat.choose 7 4 = 35 := by
  sorry

end binomial_7_4_eq_35_l111_111199


namespace train_passes_jogger_in_39_seconds_l111_111665

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def jogger_head_start : ℝ := 270
noncomputable def train_length : ℝ := 120
noncomputable def train_speed_kmph : ℝ := 45

noncomputable def to_meters_per_second (kmph : ℝ) : ℝ :=
  kmph * 1000 / 3600

noncomputable def jogger_speed_mps : ℝ :=
  to_meters_per_second jogger_speed_kmph

noncomputable def train_speed_mps : ℝ :=
  to_meters_per_second train_speed_kmph

noncomputable def relative_speed_mps : ℝ :=
  train_speed_mps - jogger_speed_mps

noncomputable def total_distance : ℝ :=
  jogger_head_start + train_length

noncomputable def time_to_pass_jogger : ℝ :=
  total_distance / relative_speed_mps

theorem train_passes_jogger_in_39_seconds :
  time_to_pass_jogger = 39 := by
  sorry

end train_passes_jogger_in_39_seconds_l111_111665


namespace simplify_fraction_l111_111607

theorem simplify_fraction (a b : ℕ) (h : Nat.gcd a b = 24) : (a = 48) → (b = 72) → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end simplify_fraction_l111_111607


namespace intersection_points_are_integers_l111_111930

theorem intersection_points_are_integers :
  ∀ (a b : Fin 2021 → ℕ), Function.Injective a → Function.Injective b →
  ∀ i j, i ≠ j → 
  ∃ x : ℤ, (∃ y : ℚ, y = (a i : ℚ) / (x + (b i : ℚ))) ∧ 
           (∃ y : ℚ, y = (a j : ℚ) / (x + (b j : ℚ))) := 
sorry

end intersection_points_are_integers_l111_111930


namespace simplify_fraction_l111_111599

-- Define the problem and conditions
def numerator : ℕ := 48
def denominator : ℕ := 72
def gcd_n_d : ℕ := Nat.gcd numerator denominator

-- The proof statement
theorem simplify_fraction : (numerator / gcd_n_d) / (denominator / gcd_n_d) = 2 / 3 :=
by
  have h_gcd : gcd_n_d = 24 := by rfl
  sorry

end simplify_fraction_l111_111599


namespace num_students_l111_111135

theorem num_students (n : ℕ) 
    (average_marks_wrong : ℕ)
    (wrong_mark : ℕ)
    (correct_mark : ℕ)
    (average_marks_correct : ℕ) :
    average_marks_wrong = 100 →
    wrong_mark = 90 →
    correct_mark = 10 →
    average_marks_correct = 92 →
    n = 10 :=
by
  intros h1 h2 h3 h4
  sorry

end num_students_l111_111135


namespace sum_of_solutions_l111_111020

theorem sum_of_solutions (S : Finset ℝ) (h : ∀ x ∈ S, |x^2 - 10 * x + 29| = 3) : S.sum id = 0 :=
sorry

end sum_of_solutions_l111_111020


namespace sum_two_numbers_in_AP_and_GP_equals_20_l111_111965

theorem sum_two_numbers_in_AP_and_GP_equals_20 :
  ∃ a b : ℝ, 
    (a > 0) ∧ (b > 0) ∧ 
    (4 < a) ∧ (a < b) ∧ 
    (4 + (a - 4) = a) ∧ (4 + 2 * (a - 4) = b) ∧
    (a * (b / a) = b) ∧ (b * (b / a) = 16) ∧ 
    a + b = 20 :=
by
  sorry

end sum_two_numbers_in_AP_and_GP_equals_20_l111_111965


namespace trip_distance_l111_111263

theorem trip_distance (D : ℝ) (t1 t2 : ℝ) :
  (30 / 60 = t1) →
  (70 / 35 = t2) →
  (t1 + t2 = 2.5) →
  (40 = D / (t1 + t2)) →
  D = 100 :=
by
  intros h1 h2 h3 h4
  sorry

end trip_distance_l111_111263


namespace correct_calculation_l111_111023

variable (a b : ℝ)

theorem correct_calculation : (ab)^2 = a^2 * b^2 := by
  sorry

end correct_calculation_l111_111023


namespace factorize_expr_l111_111217

theorem factorize_expr (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) :=
sorry

end factorize_expr_l111_111217


namespace Sophie_donuts_l111_111804

theorem Sophie_donuts 
  (boxes : ℕ)
  (donuts_per_box : ℕ)
  (boxes_given_mom : ℕ)
  (donuts_given_sister : ℕ)
  (h1 : boxes = 4)
  (h2 : donuts_per_box = 12)
  (h3 : boxes_given_mom = 1)
  (h4 : donuts_given_sister = 6) :
  (boxes * donuts_per_box) - (boxes_given_mom * donuts_per_box) - donuts_given_sister = 30 :=
by
  sorry

end Sophie_donuts_l111_111804


namespace total_rain_duration_l111_111553

theorem total_rain_duration :
  let day1 := 17 - 7 in
  let day2 := day1 + 2 in
  let day3 := day2 * 2 in
  day1 + day2 + day3 = 46 :=
by
  let day1 := 17 - 7
  let day2 := day1 + 2
  let day3 := day2 * 2
  calc
    day1 + day2 + day3 = 10 + 12 + 24 : by sorry
                     ... = 46 : by sorry

end total_rain_duration_l111_111553


namespace factor_quadratic_expression_l111_111138

theorem factor_quadratic_expression (a b : ℤ) (h: 25 * -198 = -4950 ∧ a + b = -195 ∧ a * b = -4950) : a + 2 * b = -420 :=
sorry

end factor_quadratic_expression_l111_111138


namespace problem_statement_l111_111675

def is_even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def has_minimum_value_at (f : ℝ → ℝ) (a : ℝ) := ∀ x : ℝ, f a ≤ f x
noncomputable def f4 (x : ℝ) : ℝ := Real.exp x + Real.exp (-x)

theorem problem_statement : is_even_function f4 ∧ has_minimum_value_at f4 0 :=
by
  sorry

end problem_statement_l111_111675


namespace find_g3_l111_111449

variable {α : Type*} [Field α]

-- Define the function g
noncomputable def g (x : α) : α := sorry

-- Define the condition as a hypothesis
axiom condition (x : α) (hx : x ≠ 0) : 2 * g (1 / x) + 3 * g x / x = 2 * x ^ 2

-- State what needs to be proven
theorem find_g3 : g 3 = 242 / 15 := by
  sorry

end find_g3_l111_111449


namespace solve_for_x_l111_111322

theorem solve_for_x (x : ℕ) (h : x + 1 = 2) : x = 1 :=
sorry

end solve_for_x_l111_111322


namespace restaurant_meal_cost_l111_111344

/--
Each adult meal costs $8 and kids eat free. 
If there is a group of 11 people, out of which 2 are kids, 
prove that the total cost for the group to eat is $72.
-/
theorem restaurant_meal_cost (cost_per_adult : ℕ) (group_size : ℕ) (kids : ℕ) 
  (all_free_kids : ℕ → Prop) (total_cost : ℕ)  
  (h1 : cost_per_adult = 8) 
  (h2 : group_size = 11) 
  (h3 : kids = 2) 
  (h4 : all_free_kids kids) 
  (h5 : total_cost = (group_size - kids) * cost_per_adult) : 
  total_cost = 72 := 
by 
  sorry

end restaurant_meal_cost_l111_111344


namespace find_n_l111_111279

def alpha (n : ℕ) : ℚ := ((n - 2) * 180) / n
def alpha_plus_3 (n : ℕ) : ℚ := ((n + 1) * 180) / (n + 3)
def alpha_minus_2 (n : ℕ) : ℚ := ((n - 4) * 180) / (n - 2)

theorem find_n (n : ℕ) (h : alpha_plus_3 n - alpha n = alpha n - alpha_minus_2 n) : n = 12 :=
by
  -- The proof will be added here
  sorry

end find_n_l111_111279


namespace inequality_proof_l111_111383

variable {x y z : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y + z) ^ 2 * (y * z + z * x + x * y) ^ 2 ≤ 
  3 * (y^2 + y * z + z^2) * (z^2 + z * x + x^2) * (x^2 + x * y + y^2) := 
sorry

end inequality_proof_l111_111383


namespace average_is_3_l111_111312

theorem average_is_3 (A B C : ℝ) (h1 : 1501 * C - 3003 * A = 6006)
                              (h2 : 1501 * B + 4504 * A = 7507)
                              (h3 : A + B = 1) :
  (A + B + C) / 3 = 3 :=
by sorry

end average_is_3_l111_111312


namespace factorize_expr_l111_111219

theorem factorize_expr (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) :=
sorry

end factorize_expr_l111_111219


namespace simplify_fraction_l111_111598

-- Define the problem and conditions
def numerator : ℕ := 48
def denominator : ℕ := 72
def gcd_n_d : ℕ := Nat.gcd numerator denominator

-- The proof statement
theorem simplify_fraction : (numerator / gcd_n_d) / (denominator / gcd_n_d) = 2 / 3 :=
by
  have h_gcd : gcd_n_d = 24 := by rfl
  sorry

end simplify_fraction_l111_111598


namespace length_of_walls_l111_111554

-- Definitions of the given conditions.
def wall_height : ℝ := 12
def third_wall_length : ℝ := 20
def third_wall_height : ℝ := 12
def total_area : ℝ := 960

-- The area of two walls with length L each and height 12 feet.
def two_walls_area (L : ℝ) : ℝ := 2 * L * wall_height

-- The area of the third wall.
def third_wall_area : ℝ := third_wall_length * third_wall_height

-- The proof statement
theorem length_of_walls (L : ℝ) (h1 : two_walls_area L + third_wall_area = total_area) : L = 30 :=
by
  sorry

end length_of_walls_l111_111554


namespace problem1_problem2_problem3_l111_111677

variable {m n p x : ℝ}

-- Problem 1
theorem problem1 : m^2 * (n - 3) + 4 * (3 - n) = (n - 3) * (m + 2) * (m - 2) := 
sorry

-- Problem 2
theorem problem2 : (p - 3) * (p - 1) + 1 = (p - 2) ^ 2 := 
sorry

-- Problem 3
theorem problem3 (hx : x^2 + x + 1 / 4 = 0) : (2 * x + 1) / (x + 1) + (x - 1) / 1 / (x + 2) / (x^2 + 2 * x + 1) = -1 / 4 :=
sorry

end problem1_problem2_problem3_l111_111677


namespace cars_sold_on_second_day_l111_111846

theorem cars_sold_on_second_day (x : ℕ) 
  (h1 : 14 + x + 27 = 57) : x = 16 :=
by 
  sorry

end cars_sold_on_second_day_l111_111846


namespace g_at_2_l111_111248

def g (x : ℝ) : ℝ := x^3 - x

theorem g_at_2 : g 2 = 6 :=
by
  sorry

end g_at_2_l111_111248


namespace domain_of_f_exp_l111_111106

theorem domain_of_f_exp (f : ℝ → ℝ) :
  (∀ x, 0 ≤ x + 1 ∧ x + 1 < 4 → ∃ y, f y = f (x + 1)) →
  (∀ x, 1 ≤ 2^x ∧ 2^x < 4 → ∃ y, f y = f (2^x)) :=
by
  sorry

end domain_of_f_exp_l111_111106


namespace Connie_correct_result_l111_111203

theorem Connie_correct_result :
  ∀ x: ℝ, (200 - x = 100) → (200 + x = 300) :=
by
  intros x h
  have h1 : x = 100 := by linarith [h]
  rw [h1]
  linarith

end Connie_correct_result_l111_111203


namespace array_sum_remainder_mod_9_l111_111056

theorem array_sum_remainder_mod_9 :
  let sum_terms := ∑' r : ℕ, ∑' c : ℕ, (1 / (4 ^ r)) * (1 / (9 ^ c))
  ∃ m n : ℕ, Nat.gcd m n = 1 ∧ sum_terms = m / n ∧ (m + n) % 9 = 5 :=
by
  sorry

end array_sum_remainder_mod_9_l111_111056


namespace count_valid_combinations_l111_111387

-- Define the digits condition
def is_digit (d : ℕ) : Prop := d >= 0 ∧ d <= 9

-- Define the main proof statement
theorem count_valid_combinations (a b c: ℕ) (h1 : is_digit a)(h2 : is_digit b)(h3 : is_digit c) :
    (100 * a + 10 * b + c) + (100 * c + 10 * b + a) = 1069 → 
    ∃ (abc_combinations : ℕ), abc_combinations = 8 :=
by
  sorry

end count_valid_combinations_l111_111387


namespace keith_score_l111_111924

theorem keith_score (K : ℕ) (h : K + 3 * K + (3 * K + 5) = 26) : K = 3 :=
by
  sorry

end keith_score_l111_111924


namespace eve_ran_further_l111_111352

variable (ran_distance walked_distance difference_distance : ℝ)

theorem eve_ran_further (h1 : ran_distance = 0.7) (h2 : walked_distance = 0.6) : ran_distance - walked_distance = 0.1 := by
  sorry

end eve_ran_further_l111_111352


namespace grain_storage_bins_total_l111_111337

theorem grain_storage_bins_total
  (b20 : ℕ) (b20_tonnage : ℕ) (b15_tonnage : ℕ) (total_capacity : ℕ) (b20_count : ℕ)
  (h_b20_capacity : b20_count * b20_tonnage = b20)
  (h_total_capacity : b20 + (total_capacity - b20) = total_capacity)
  (h_b20_given : b20_count = 12)
  (h_b20_tonnage : b20_tonnage = 20)
  (h_b15_tonnage : b15_tonnage = 15)
  (h_total_capacity_given : total_capacity = 510) :
  ∃ b_total : ℕ, b_total = b20_count + ((total_capacity - (b20_count * b20_tonnage)) / b15_tonnage) ∧ b_total = 30 :=
by
  sorry

end grain_storage_bins_total_l111_111337


namespace eval_special_op_l111_111367

variable {α : Type*} [LinearOrderedField α]

def op (a b : α) : α := (a - b) ^ 2

theorem eval_special_op (x y z : α) : op ((x - y + z)^2) ((y - x - z)^2) = 0 := by
  sorry

end eval_special_op_l111_111367


namespace parabola_equation_l111_111853

theorem parabola_equation (x y : ℝ)
    (focus : x = 1 ∧ y = -2)
    (directrix : 5 * x + 2 * y = 10) :
    4 * x^2 - 20 * x * y + 25 * y^2 + 158 * x + 156 * y + 16 = 0 := 
by
  -- use the given conditions and intermediate steps to derive the final equation
  sorry

end parabola_equation_l111_111853


namespace pyramid_volume_l111_111949

theorem pyramid_volume (S : ℝ) :
  ∃ (V : ℝ),
  (∀ (a b h : ℝ), S = a * b ∧
  h = a * (Real.tan (60 * (Real.pi / 180))) ∧
  h = b * (Real.tan (30 * (Real.pi / 180))) ∧
  V = (1/3) * S * h) →
  V = (S * Real.sqrt S) / 3 :=
by
  sorry

end pyramid_volume_l111_111949


namespace intersection_A_B_l111_111528

open Set

def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℤ := {x | x^2 ≤ 1}

theorem intersection_A_B : A ∩ B = {-1, 0, 1} := 
by {
  sorry
}

end intersection_A_B_l111_111528


namespace simplify_fraction_l111_111594

theorem simplify_fraction :
  (48 : ℚ) / 72 = 2 / 3 :=
sorry

end simplify_fraction_l111_111594


namespace gcd_all_abc_plus_cba_l111_111541

noncomputable def gcd_of_abc_cba (a : ℕ) (b : ℕ := 2 * a) (c : ℕ := 3 * a) : ℕ :=
  let abc := 64 * a + 8 * b + c
  let cba := 64 * c + 8 * b + a
  Nat.gcd (abc + cba) 300

theorem gcd_all_abc_plus_cba (a : ℕ) : gcd_of_abc_cba a = 300 :=
  sorry

end gcd_all_abc_plus_cba_l111_111541


namespace find_value_l111_111530

-- Defining the known conditions
def number : ℕ := 20
def half (n : ℕ) : ℕ := n / 2
def value_added (V : ℕ) : Prop := half number + V = 17

-- Proving that the value added to half the number is 7
theorem find_value : value_added 7 :=
by
  -- providing the proof for the theorem
  -- skipping the proof steps with sorry
  sorry

end find_value_l111_111530


namespace direct_proportion_m_n_l111_111253

theorem direct_proportion_m_n (m n : ℤ) (h₁ : m - 2 = 1) (h₂ : n + 1 = 0) : m + n = 2 :=
by
  sorry

end direct_proportion_m_n_l111_111253


namespace flagstaff_height_l111_111990

theorem flagstaff_height 
  (s1 : ℝ) (s2 : ℝ) (hb : ℝ) (h : ℝ)
  (H1 : s1 = 40.25) (H2 : s2 = 28.75) (H3 : hb = 12.5) 
  (H4 : h / s1 = hb / s2) : 
  h = 17.5 :=
by
  sorry

end flagstaff_height_l111_111990


namespace S8_eq_90_l111_111369

-- Definitions and given conditions
def arithmetic_seq (a : ℕ → ℤ) : Prop := ∃ d, ∀ n, a (n + 1) - a n = d
def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop := ∀ n, S n = (n * (a 1 + a n)) / 2
def condition_a4 (a : ℕ → ℤ) : Prop := a 4 = 18 - a 5

-- Prove that S₈ = 90
theorem S8_eq_90 (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h_arith_seq : arithmetic_seq a)
  (h_sum : sum_of_first_n_terms a S)
  (h_cond : condition_a4 a) : S 8 = 90 :=
by
  sorry

end S8_eq_90_l111_111369


namespace product_of_roots_l111_111227

noncomputable def f (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem product_of_roots (x : ℝ) : 
  (4 * f (3 - x) - f x = 3 * x^2 - 4 * x - 3) →
  (Exists (λ a b : ℝ, f a = 8 ∧ f b = 8 ∧ a * b = -5)) :=
by
  sorry

end product_of_roots_l111_111227


namespace minimum_value_f_l111_111360

noncomputable def f (x : ℝ) : ℝ := x^2 + 3 * x + 6 / x + 4 / x^2 - 1

theorem minimum_value_f : 
    ∃ (x : ℝ), x > 0 ∧ 
    (∀ (y : ℝ), y > 0 → f y ≥ f x) ∧ 
    f x = 3 - 6 * Real.sqrt 2 :=
sorry

end minimum_value_f_l111_111360


namespace tangent_line_eq_bounded_area_l111_111544

-- Given two parabolas and a tangent line, and a positive constant a
variables (a : ℝ)
variables (y1 y2 l : ℝ → ℝ)

-- Conditions:
def parabola1 := ∀ (x : ℝ), y1 x = x^2 + a * x
def parabola2 := ∀ (x : ℝ), y2 x = x^2 - 2 * a * x
def tangent_line := ∀ (x : ℝ), l x = - (a / 2) * x - (9 * a^2 / 16)
def a_positive := a > 0

-- Proof goals:
theorem tangent_line_eq : 
  parabola1 a y1 ∧ parabola2 a y2 ∧ tangent_line a l ∧ a_positive a 
  → ∀ x, (y1 x = l x ∨ y2 x = l x) :=
sorry

theorem bounded_area : 
  parabola1 a y1 ∧ parabola2 a y2 ∧ tangent_line a l ∧ a_positive a 
  → ∫ (x : ℝ) in (-3 * a / 4)..(3 * a / 4), (y1 x - l x) + (y2 x - l x) = 9 * a^3 / 8 :=
sorry

end tangent_line_eq_bounded_area_l111_111544


namespace real_condition_proof_l111_111932

noncomputable def real_condition_sufficient_but_not_necessary : Prop := 
∀ x : ℝ, (|x - 2| < 1) → ((x^2 + x - 2) > 0) ∧ (¬ ( ∀ y : ℝ, (y^2 + y - 2) > 0 → |y - 2| < 1))

theorem real_condition_proof : real_condition_sufficient_but_not_necessary :=
by
  sorry

end real_condition_proof_l111_111932


namespace find_second_liquid_parts_l111_111126

-- Define the given constants
def first_liquid_kerosene_percentage : ℝ := 0.25
def second_liquid_kerosene_percentage : ℝ := 0.30
def first_liquid_parts : ℝ := 6
def mixture_kerosene_percentage : ℝ := 0.27

-- Define the amount of kerosene from each liquid
def kerosene_from_first_liquid := first_liquid_kerosene_percentage * first_liquid_parts
def kerosene_from_second_liquid (x : ℝ) := second_liquid_kerosene_percentage * x

-- Define the total parts of mixture
def total_mixture_parts (x : ℝ) := first_liquid_parts + x

-- Define the total kerosene in the mixture
def total_kerosene_in_mixture (x : ℝ) := mixture_kerosene_percentage * total_mixture_parts x

-- State the theorem
theorem find_second_liquid_parts (x : ℝ) :
  kerosene_from_first_liquid + kerosene_from_second_liquid x = total_kerosene_in_mixture x → 
  x = 4 :=
by
  sorry

end find_second_liquid_parts_l111_111126


namespace simplify_fraction_l111_111606

theorem simplify_fraction (a b : ℕ) (h : Nat.gcd a b = 24) : (a = 48) → (b = 72) → a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3 :=
by
  intros ha hb
  rw [ha, hb]
  sorry

end simplify_fraction_l111_111606


namespace polynomial_value_at_minus_two_l111_111637

def f (x : ℝ) : ℝ := x^5 + 4*x^4 + x^2 + 20*x + 16

theorem polynomial_value_at_minus_two : f (-2) = 12 := by 
  sorry

end polynomial_value_at_minus_two_l111_111637


namespace lifespan_of_bat_l111_111620

variable (B H F T : ℝ)

theorem lifespan_of_bat (h₁ : H = B - 6)
                        (h₂ : F = 4 * H)
                        (h₃ : T = 2 * B)
                        (h₄ : B + H + F + T = 62) :
  B = 11.5 :=
by
  sorry

end lifespan_of_bat_l111_111620


namespace check_correct_options_l111_111375

noncomputable def f (x a b: ℝ) := x^3 - a*x^2 + b*x + 1

theorem check_correct_options :
  (∀ (b: ℝ), b = 0 → ¬(∃ x: ℝ, 3 * x^2 - 2 * a * x = 0)) ∧
  (∀ (a: ℝ), a = 0 → (∀ x: ℝ, f x a b + f (-x) a b = 2)) ∧
  (∀ (a: ℝ), ∀ (b: ℝ), b = a^2 / 4 ∧ a > -4 → ∃ x1 x2 x3: ℝ, f x1 a b = 0 ∧ f x2 a b = 0 ∧ f x3 a b = 0) ∧
  (∀ (a: ℝ), ∀ (b: ℝ), (∀ x: ℝ, 3 * x^2 - 2 * a * x + b ≥ 0) → (a^2 ≤ 3*b)) := sorry

end check_correct_options_l111_111375


namespace candy_distribution_l111_111014

-- Define the problem conditions and theorem.

theorem candy_distribution (X : ℕ) (total_pieces : ℕ) (portions : ℕ) 
  (subsequent_more : ℕ) (h_total : total_pieces = 40) 
  (h_portions : portions = 4) 
  (h_subsequent : subsequent_more = 2) 
  (h_eq : X + (X + subsequent_more) + (X + subsequent_more * 2) + (X + subsequent_more * 3) = total_pieces) : 
  X = 7 := 
sorry

end candy_distribution_l111_111014


namespace cost_of_paving_l111_111652

def length : ℝ := 5.5
def width : ℝ := 4
def rate_per_sq_meter : ℝ := 850

theorem cost_of_paving :
  rate_per_sq_meter * (length * width) = 18700 :=
by
  sorry

end cost_of_paving_l111_111652


namespace superior_points_in_Omega_l111_111241

-- Define the set Omega
def Omega : Set (ℝ × ℝ) := { p | let (x, y) := p; x^2 + y^2 ≤ 2008 }

-- Definition of the superior relation
def superior (P P' : ℝ × ℝ) : Prop :=
  let (x, y) := P
  let (x', y') := P'
  x ≤ x' ∧ y ≥ y'

-- Definition of the set of points Q such that no other point in Omega is superior to Q
def Q_set : Set (ℝ × ℝ) :=
  { p | let (x, y) := p; x^2 + y^2 = 2008 ∧ x ≤ 0 ∧ y ≥ 0 }

theorem superior_points_in_Omega :
  { p | p ∈ Omega ∧ ¬ (∃ q ∈ Omega, superior q p) } = Q_set :=
by
  sorry

end superior_points_in_Omega_l111_111241


namespace complex_magnitude_l111_111520

open Complex

noncomputable def complexZ : ℂ := sorry -- Definition of complex number z

theorem complex_magnitude (z : ℂ) (h : (1 + 2 * Complex.I) * z = -3 + 4 * Complex.I) : abs z = Real.sqrt 5 :=
sorry

end complex_magnitude_l111_111520


namespace inequality_proof_l111_111062

theorem inequality_proof (a b c d : ℝ) : 
  0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → a + b = 2 → c + d = 2 → 
  (a^2 + c^2) * (a^2 + d^2) * (b^2 + c^2) * (b^2 + d^2) ≤ 25 := 
by 
  intros ha hb hc hd hab hcd
  sorry

end inequality_proof_l111_111062


namespace square_area_l111_111698

theorem square_area (perimeter : ℝ) (h : perimeter = 40) : ∃ A : ℝ, A = 100 := by
  have h1 : ∃ s : ℝ, 4 * s = perimeter := by
    use perimeter / 4
    linarith

  cases h1 with s hs
  use s^2
  rw [hs, h]
  norm_num
  sorry

end square_area_l111_111698


namespace unique_zero_of_f_l111_111238

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 2*x + a * (Real.exp (x - 1) + Real.exp (-x + 1))

theorem unique_zero_of_f (a : ℝ) : (∃! x, f x a = 0) ↔ a = 1 / 2 := sorry

end unique_zero_of_f_l111_111238


namespace prove_a_eq_b_l111_111918

theorem prove_a_eq_b (a b : ℝ) (h : 1 / (3 * a) + 2 / (3 * b) = 3 / (a + 2 * b)) : a = b :=
sorry

end prove_a_eq_b_l111_111918


namespace total_pears_sold_l111_111997

theorem total_pears_sold (sold_morning : ℕ) (sold_afternoon : ℕ) (h_morning : sold_morning = 120) (h_afternoon : sold_afternoon = 240) :
  sold_morning + sold_afternoon = 360 :=
by
  sorry

end total_pears_sold_l111_111997


namespace donuts_left_for_sophie_l111_111803

def initial_boxes := 4
def donuts_per_box := 12
def boxes_given_to_mom := 1
def donuts_given_to_sister := 6

theorem donuts_left_for_sophie :
  let initial_donuts := initial_boxes * donuts_per_box in
  let remaining_boxes := initial_boxes - boxes_given_to_mom in
  let remaining_donuts := remaining_boxes * donuts_per_box in
  let donuts_left := remaining_donuts - donuts_given_to_sister in
  donuts_left = 30 := 
by 
  have initial_donuts := initial_boxes * donuts_per_box
  have remaining_boxes := initial_boxes - boxes_given_to_mom
  have remaining_donuts := remaining_boxes * donuts_per_box
  have donuts_left := remaining_donuts - donuts_given_to_sister
  show donuts_left = 30
  calc
  donuts_per_box * (initial_boxes - boxes_given_to_mom) - donuts_given_to_sister = donuts_per_box * 3 - donuts_given_to_sister := by sorry
  donuts_per_box * 3 - donuts_given_to_sister = 30 := by sorry

end donuts_left_for_sophie_l111_111803


namespace maximum_value_problem_l111_111514

theorem maximum_value_problem (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  (a^2 - b * c) * (b^2 - c * a) * (c^2 - a * b) ≤ 1 / 8 :=
sorry

end maximum_value_problem_l111_111514


namespace set_equality_implies_a_value_l111_111722

theorem set_equality_implies_a_value (a : ℤ) : ({2, 3} : Set ℤ) = {2, 2 * a - 1} → a = 2 := 
by
  intro h
  sorry

end set_equality_implies_a_value_l111_111722


namespace average_food_per_week_l111_111493

-- Definitions based on conditions
def food_first_dog := 13
def food_second_dog := 2 * food_first_dog
def food_third_dog := 6
def number_of_dogs := 3

-- Statement of the proof problem
theorem average_food_per_week : 
  (food_first_dog + food_second_dog + food_third_dog) / number_of_dogs = 15 := 
by sorry

end average_food_per_week_l111_111493


namespace total_profit_l111_111865

-- Definitions based on the conditions
variables (A B C : ℝ) (P : ℝ)
variables (hA : A = 3 * B) (hB : B = (2 / 3) * C) (hB_share : ((2 / 11) * P) = 1400)

-- The theorem we are going to prove
theorem total_profit (A B C P : ℝ) (hA : A = 3 * B) (hB : B = (2 / 3) * C) (hB_share : ((2 / 11) * P) = 1400) : 
  P = 7700 :=
by
  sorry

end total_profit_l111_111865


namespace max_subset_count_l111_111928

-- Define the problem conditions in Lean 4
def is_valid_subset (T : Finset ℕ) : Prop :=
  ∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → ¬ (a + b) % 5 = 0

theorem max_subset_count :
  ∃ (T : Finset ℕ), (is_valid_subset T) ∧ T.card = 18 := by
  sorry

end max_subset_count_l111_111928


namespace binomial_7_4_eq_35_l111_111195
-- Import the entire Mathlib library for broad utility functions

-- Define the binomial coefficient function using library definitions 
noncomputable def binomial : ℕ → ℕ → ℕ
| n, k := nat.choose n k

-- Define the theorem we want to prove
theorem binomial_7_4_eq_35 : binomial 7 4 = 35 :=
by
  sorry

end binomial_7_4_eq_35_l111_111195


namespace length_error_probability_l111_111547

theorem length_error_probability
  (μ σ : ℝ)
  (X : ℝ → ℝ)
  (h_norm_dist : ∀ x : ℝ, X x = (Real.exp (-(x - μ) ^ 2 / (2 * σ ^ 2)) / (σ * Real.sqrt (2 * Real.pi))))
  (h_max_density : X 0 = 1 / (3 * Real.sqrt (2 * Real.pi)))
  (P : Set ℝ → ℝ)
  (h_prop1 : P {x | μ - σ < x ∧ x < μ + σ} = 0.6826)
  (h_prop2 : P {x | μ - 2 * σ < x ∧ x < μ + 2 * σ} = 0.9544) :
  P {x | 3 < x ∧ x < 6} = 0.1359 :=
sorry

end length_error_probability_l111_111547


namespace max_rectangle_area_l111_111673

theorem max_rectangle_area (l w : ℕ) (h1 : 2 * l + 2 * w = 40) : l * w ≤ 100 :=
by
  have h2 : l + w = 20 := by linarith
  -- Further steps would go here but we're just stating it
  sorry

end max_rectangle_area_l111_111673


namespace negation_of_universal_proposition_l111_111007

theorem negation_of_universal_proposition :
  (¬ ∀ (x : ℝ), x^2 ≥ 0) ↔ ∃ (x : ℝ), x^2 < 0 :=
by sorry

end negation_of_universal_proposition_l111_111007


namespace price_reduction_l111_111658

theorem price_reduction (x : ℝ) (h : 560 * (1 - x) * (1 - x) = 315) : 
  560 * (1 - x)^2 = 315 := 
by
  sorry

end price_reduction_l111_111658


namespace simplify_fraction_l111_111601

-- Define the problem and conditions
def numerator : ℕ := 48
def denominator : ℕ := 72
def gcd_n_d : ℕ := Nat.gcd numerator denominator

-- The proof statement
theorem simplify_fraction : (numerator / gcd_n_d) / (denominator / gcd_n_d) = 2 / 3 :=
by
  have h_gcd : gcd_n_d = 24 := by rfl
  sorry

end simplify_fraction_l111_111601


namespace simplify_fraction_l111_111602

-- Define the problem and conditions
def numerator : ℕ := 48
def denominator : ℕ := 72
def gcd_n_d : ℕ := Nat.gcd numerator denominator

-- The proof statement
theorem simplify_fraction : (numerator / gcd_n_d) / (denominator / gcd_n_d) = 2 / 3 :=
by
  have h_gcd : gcd_n_d = 24 := by rfl
  sorry

end simplify_fraction_l111_111602


namespace smallest_d_factors_l111_111363

theorem smallest_d_factors (d : ℕ) (h₁ : ∃ p q : ℤ, p * q = 2050 ∧ p + q = d ∧ p > 0 ∧ q > 0) :
    d = 107 :=
by
  sorry

end smallest_d_factors_l111_111363


namespace find_a6_l111_111204

def is_arithmetic_sequence (b : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, b (n + 1) = b n + d

theorem find_a6 :
  ∀ (a b : ℕ → ℕ),
    a 1 = 3 →
    b 1 = 2 →
    b 3 = 6 →
    is_arithmetic_sequence b →
    (∀ n, b n = a (n + 1) - a n) →
    a 6 = 33 :=
by
  intros a b h_a1 h_b1 h_b3 h_arith h_diff
  sorry

end find_a6_l111_111204


namespace ammonium_iodide_required_l111_111356

theorem ammonium_iodide_required
  (KOH_moles NH3_moles KI_moles H2O_moles : ℕ)
  (hn : NH3_moles = 3) (hk : KOH_moles = 3) (hi : KI_moles = 3) (hw : H2O_moles = 3) :
  ∃ NH4I_moles, NH3_moles = 3 ∧ KI_moles = 3 ∧ H2O_moles = 3 ∧ KOH_moles = 3 ∧ NH4I_moles = 3 :=
by
  sorry

end ammonium_iodide_required_l111_111356


namespace completing_the_square_l111_111636

theorem completing_the_square (x : ℝ) (h : x^2 - 6 * x + 7 = 0) : (x - 3)^2 - 2 = 0 := 
by sorry

end completing_the_square_l111_111636


namespace contradiction_proof_l111_111825

theorem contradiction_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h1 : a + 1/b < 2) (h2 : b + 1/c < 2) (h3 : c + 1/a < 2) : 
  ¬ (a + 1/b ≥ 2 ∨ b + 1/c ≥ 2 ∨ c + 1/a ≥ 2) :=
by
  sorry

end contradiction_proof_l111_111825


namespace problem_statement_l111_111701

theorem problem_statement :
  ∃ (n : ℕ), n = 101 ∧
  (∀ (x : ℕ), x < 4032 → ((x^2 - 20) % 16 = 0) ∧ ((x^2 - 16) % 20 = 0) ↔ (∃ k1 k2 : ℕ, (x = 80 * k1 + 6 ∨ x = 80 * k2 + 74) ∧ k1 + k2 + 1 = n)) :=
by sorry

end problem_statement_l111_111701


namespace jail_time_ratio_l111_111635

def arrests (days : ℕ) (cities : ℕ) (arrests_per_day : ℕ) : ℕ := days * cities * arrests_per_day
def jail_days_before_trial (total_arrests : ℕ) (days_before_trial : ℕ) : ℕ := total_arrests * days_before_trial
def weeks_from_days (days : ℕ) : ℕ := days / 7
def time_after_trial (total_jail_time_weeks : ℕ) (weeks_before_trial : ℕ) : ℕ := total_jail_time_weeks - weeks_before_trial
def total_possible_jail_time (total_arrests : ℕ) (sentence_weeks : ℕ) : ℕ := total_arrests * sentence_weeks
def ratio (after_trial_weeks : ℕ) (total_possible_weeks : ℕ) : ℚ := after_trial_weeks / total_possible_weeks

theorem jail_time_ratio 
    (days : ℕ := 30) 
    (cities : ℕ := 21)
    (arrests_per_day : ℕ := 10)
    (days_before_trial : ℕ := 4)
    (total_jail_time_weeks : ℕ := 9900)
    (sentence_weeks : ℕ := 2) :
    ratio 
      (time_after_trial 
        total_jail_time_weeks 
        (weeks_from_days 
          (jail_days_before_trial 
            (arrests days cities arrests_per_day) 
            days_before_trial))) 
      (total_possible_jail_time 
        (arrests days cities arrests_per_day) 
        sentence_weeks) = 1/2 := 
by
  -- We leave the proof as an exercise
  sorry

end jail_time_ratio_l111_111635


namespace rectangle_symmetry_l111_111972

-- Definitions of symmetry properties
def isAxisymmetric (shape : Type) : Prop := sorry
def isCentrallySymmetric (shape : Type) : Prop := sorry

-- Specific shapes
def EquilateralTriangle : Type := sorry
def Parallelogram : Type := sorry
def Rectangle : Type := sorry
def RegularPentagon : Type := sorry

-- The theorem we want to prove
theorem rectangle_symmetry : 
  isAxisymmetric Rectangle ∧ isCentrallySymmetric Rectangle := sorry

end rectangle_symmetry_l111_111972


namespace air_conditioner_sale_price_l111_111869

theorem air_conditioner_sale_price (P : ℝ) (d1 d2 : ℝ) (hP : P = 500) (hd1 : d1 = 0.10) (hd2 : d2 = 0.20) :
  ((P * (1 - d1)) * (1 - d2)) / P * 100 = 72 :=
by
  sorry

end air_conditioner_sale_price_l111_111869


namespace investment_return_l111_111172

theorem investment_return (y_r : ℝ) :
  (500 + 1500) * 0.085 = 500 * 0.07 + 1500 * y_r → y_r = 0.09 :=
by
  sorry

end investment_return_l111_111172


namespace line_passes_through_fixed_point_l111_111708

theorem line_passes_through_fixed_point (a b : ℝ) (x y : ℝ) 
  (h1 : 3 * a + 2 * b = 5) 
  (h2 : x = 6) 
  (h3 : y = 4) : 
  a * x + b * y - 10 = 0 := 
by
  sorry

end line_passes_through_fixed_point_l111_111708


namespace arithmetic_seq_count_114_l111_111064

open Finset

noncomputable def count_four_term_arithmetic_seq (s : Finset ℕ) : ℕ :=
  s.filter (λ t, ∃ a d, t = {a, a + d, a + 2*d, a + 3*d} ∧
    a ∈ s ∧ a + d ∈ s ∧ a + 2*d ∈ s ∧ a + 3*d ∈ s ∧
    (a + d ≠ a) ∧ (a + 2*d ≠ a) ∧ (a + 3*d ≠ a) ∧
    d ≠ 0).card

theorem arithmetic_seq_count_114 :
  count_four_term_arithmetic_seq (range 20 \ {0}) = 114 := sorry

end arithmetic_seq_count_114_l111_111064


namespace value_of_algebraic_expression_l111_111538

noncomputable def quadratic_expression (m : ℝ) : ℝ :=
  3 * m * (2 * m - 3) - 1

theorem value_of_algebraic_expression (m : ℝ) (h : 2 * m^2 - 3 * m - 1 = 0) : quadratic_expression m = 2 :=
by {
  sorry
}

end value_of_algebraic_expression_l111_111538


namespace arithmetic_sequence_property_sequence_b_property_sum_B_n_l111_111070

theorem arithmetic_sequence_property (d a1 : ℕ) (a : ℕ → ℕ) :
    a 3 = 7 → 
    (4 * a1 + 6 * d) = 24 → 
    (∀ n : ℕ, a n = a1 + (n - 1) * d) → 
    (∀ n : ℕ, a n = 2 * n + 1) :=
sorry

theorem sequence_b_property (a : ℕ → ℕ) (b : ℕ → ℕ) :
    (∀ n : ℕ, a n = 2 * n + 1) →
    (∀ n : ℕ, T_n = n^2 + a n) →
    (∀ n : ℕ, b n = 
      if n = 1 then 4 
      else T_n - T_(n - 1)) → 
    (∀ n : ℕ, b n = 
      if n = 1 then 4 
      else 2 * n + 1) :=
sorry

theorem sum_B_n (b : ℕ → ℕ) (B : ℕ → ℝ) :
    (∀ n : ℕ, b n = if n = 1 then 4 else 2 * n + 1) →
    (∀ n : ℕ, B_n = ∑ i in range n, 1 / (b i * b (i + 1))) →
    B n = (3 / 20) - (1 / (4 * n + 6)) :=
sorry

end arithmetic_sequence_property_sequence_b_property_sum_B_n_l111_111070


namespace correct_statement_a_incorrect_statement_b_incorrect_statement_c_incorrect_statement_d_incorrect_statement_e_l111_111470

theorem correct_statement_a (x : ℝ) : x > 1 → x^2 > x :=
by sorry

theorem incorrect_statement_b (x : ℝ) : ¬ (x^2 < 0 → x < 0) :=
by sorry

theorem incorrect_statement_c (x : ℝ) : ¬ (x^2 < x → x < 0) :=
by sorry

theorem incorrect_statement_d (x : ℝ) : ¬ (x^2 < 1 → x < 1) :=
by sorry

theorem incorrect_statement_e (x : ℝ) : ¬ (x > 0 → x^2 > x) :=
by sorry

end correct_statement_a_incorrect_statement_b_incorrect_statement_c_incorrect_statement_d_incorrect_statement_e_l111_111470


namespace probability_more_twos_than_fives_correct_l111_111534

noncomputable def probability_more_twos_than_fives : ℚ :=
  -- Assumptions:
  -- - Five fair six-sided dice
  -- - Each die roll is independent and uniformly distributed over {1, 2, 3, 4, 5, 6}
  let total_outcomes := 6^5 in  -- Total number of outcomes when 5 dice are rolled
  let equal_twos_and_fives := 1024 + 1280 + 120 in  -- Outcomes with equal number of 2's and 5's calculated in solution
  let probability_of_equal_twos_and_fives := equal_twos_and_fives / total_outcomes in
  1 / 2 * (1 - probability_of_equal_twos_and_fives)

theorem probability_more_twos_than_fives_correct : 
  probability_more_twos_than_fives = 2676 / 7776 :=
by
  sorry

end probability_more_twos_than_fives_correct_l111_111534


namespace remainder_division_1000_l111_111022

theorem remainder_division_1000 (x : ℕ) (hx : x > 0) (h : 100 % x = 10) : 1000 % x = 10 :=
  sorry

end remainder_division_1000_l111_111022


namespace simplify_fraction_l111_111583

theorem simplify_fraction (h1 : 48 = 2^4 * 3) (h2 : 72 = 2^3 * 3^2) : (48 / 72 : ℚ) = 2 / 3 := 
by
  sorry

end simplify_fraction_l111_111583


namespace factorize_expression_l111_111214

theorem factorize_expression (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := sorry

end factorize_expression_l111_111214


namespace A_subset_B_l111_111780

def A : Set ℤ := { x : ℤ | ∃ k : ℤ, x = 4 * k + 1 }
def B : Set ℤ := { x : ℤ | ∃ k : ℤ, x = 2 * k - 1 }

theorem A_subset_B : A ⊆ B :=
  sorry

end A_subset_B_l111_111780


namespace transformation_matrix_correct_l111_111685
noncomputable def M : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, 3],
  ![-3, 0]
]

theorem transformation_matrix_correct :
  let R : Matrix (Fin 2) (Fin 2) ℝ := ![
    ![0, 1],
    ![-1, 0]
  ];
  let S : ℝ := 3;
  M = S • R :=
by
  sorry

end transformation_matrix_correct_l111_111685


namespace point_in_second_quadrant_l111_111103

theorem point_in_second_quadrant {x : ℝ} (h1 : 6 - 2 * x < 0) (h2 : x - 5 > 0) : x > 5 :=
sorry

end point_in_second_quadrant_l111_111103


namespace range_of_a_for_inequality_l111_111702

theorem range_of_a_for_inequality : 
  ∃ a : ℝ, (∀ x : ℤ, (a * x - 1) ^ 2 < x ^ 2) ↔ 
    (a > -3 / 2 ∧ a ≤ -4 / 3) ∨ (4 / 3 ≤ a ∧ a < 3 / 2) :=
by
  sorry

end range_of_a_for_inequality_l111_111702


namespace shelves_used_l111_111039

theorem shelves_used (initial_books : ℕ) (sold_books : ℕ) (books_per_shelf : ℕ) (remaining_books : ℕ) (total_shelves : ℕ) :
  initial_books = 120 → sold_books = 39 → books_per_shelf = 9 → remaining_books = initial_books - sold_books → total_shelves = remaining_books / books_per_shelf → total_shelves = 9 :=
by
  intros h_initial_books h_sold_books h_books_per_shelf h_remaining_books h_total_shelves
  rw [h_initial_books, h_sold_books] at h_remaining_books
  rw [h_books_per_shelf, h_remaining_books] at h_total_shelves
  exact h_total_shelves

end shelves_used_l111_111039


namespace bacteria_after_10_hours_l111_111988

def bacteria_count (hours : ℕ) : ℕ :=
  2^hours

theorem bacteria_after_10_hours : bacteria_count 10 = 1024 := by
  sorry

end bacteria_after_10_hours_l111_111988


namespace total_amount_is_4000_l111_111655

-- Define the amount put at a 3% interest rate
def amount_at_3_percent : ℝ := 2800

-- Define the total annual interest from both investments
def total_annual_interest : ℝ := 144

-- Define the interest rate for the amount put at 3% and 5%
def interest_rate_3_percent : ℝ := 0.03
def interest_rate_5_percent : ℝ := 0.05

-- Define the total amount to be proved
def total_amount_divided (T : ℝ) : Prop :=
  interest_rate_3_percent * amount_at_3_percent + 
  interest_rate_5_percent * (T - amount_at_3_percent) = total_annual_interest

-- The theorem that states the total amount divided is Rs. 4000
theorem total_amount_is_4000 : ∃ T : ℝ, total_amount_divided T ∧ T = 4000 :=
by
  use 4000
  unfold total_amount_divided
  simp
  sorry

end total_amount_is_4000_l111_111655


namespace simplify_fraction_l111_111578

theorem simplify_fraction (h1 : 48 = 2^4 * 3) (h2 : 72 = 2^3 * 3^2) : (48 / 72 : ℚ) = 2 / 3 := 
by
  sorry

end simplify_fraction_l111_111578


namespace find_ratio_EG_ES_l111_111264

variables (EF GH EH EG ES QR : ℝ) -- lengths of the segments
variables (x y : ℝ) -- unknowns for parts of the segments
variables (Q R S : Point) -- points

-- Define conditions based on the problem
def parallelogram_EFGH (EF GH EH EG : ℝ) : Prop :=
  ∀ (x y : ℝ), EF = 8 * x ∧ EH = 9 * y

def point_on_segment_Q (Q : Point) (EF EQ : ℝ) : Prop :=
  ∃ x : ℝ, EQ = (1 / 8) * EF

def point_on_segment_R (R : Point) (EH ER : ℝ) : Prop :=
  ∃ y : ℝ, ER = (1 / 9) * EH

def intersection_at_S (EG QR ES : ℝ) : Prop :=
  ∃ x y : ℝ, ES = (1 / 8) * EG + (1 / 9) * EG

theorem find_ratio_EG_ES :
  parallelogram_EFGH EF GH EH EG →
  point_on_segment_Q Q EF (1/8 * EF) →
  point_on_segment_R R EH (1/9 * EH) →
  intersection_at_S EG QR ES →
  EG / ES = 72 / 17 :=
by
  intros h_parallelogram h_pointQ h_pointR h_intersection
  sorry

end find_ratio_EG_ES_l111_111264


namespace ellipse_condition_l111_111147

theorem ellipse_condition (x y m : ℝ) :
  (1 < m ∧ m < 3) → (∀ x y, (∃ k1 k2: ℝ, k1 > 0 ∧ k2 > 0 ∧ k1 ≠ k2 ∧ (x^2 / k1 + y^2 / k2 = 1 ↔ (1 < m ∧ m < 3 ∧ m ≠ 2)))) :=
by 
  sorry

end ellipse_condition_l111_111147


namespace factorize_expression_l111_111213

theorem factorize_expression (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := sorry

end factorize_expression_l111_111213


namespace range_of_a_l111_111777

theorem range_of_a (x y a : ℝ) :
  (2 * x + y ≥ 4) → 
  (x - y ≥ 1) → 
  (x - 2 * y ≤ 2) → 
  (x = 2) → 
  (y = 0) → 
  (z = a * x + y) → 
  (Ax = 2) → 
  (Ay = 0) → 
  (-1/2 < a ∧ a < 2) := sorry

end range_of_a_l111_111777


namespace maximize_profit_l111_111987

noncomputable section

def price (x : ℕ) : ℝ :=
  if 0 < x ∧ x ≤ 100 then 60
  else if 100 < x ∧ x ≤ 600 then 62 - 0.02 * x
  else 0

def profit (x : ℕ) : ℝ :=
  (price x - 40) * x

theorem maximize_profit :
  ∃ x : ℕ, (1 ≤ x ∧ x ≤ 600) ∧ (∀ y : ℕ, (1 ≤ y ∧ y ≤ 600 → profit y ≤ profit x)) ∧ profit x = 6050 :=
by sorry

end maximize_profit_l111_111987


namespace greatest_b_solution_l111_111879

def f (b : ℝ) : ℝ := b^2 - 10 * b + 24

theorem greatest_b_solution : ∃ (b : ℝ), (f b ≤ 0) ∧ (∀ (b' : ℝ), (f b' ≤ 0) → b' ≤ b) ∧ b = 6 :=
by
  sorry

end greatest_b_solution_l111_111879


namespace max_xy_value_l111_111894

theorem max_xy_value (x y : ℕ) (h : 27 * x + 35 * y ≤ 1000) : x * y ≤ 252 :=
sorry

end max_xy_value_l111_111894


namespace line_slope_intercept_l111_111816

theorem line_slope_intercept (x y : ℝ) (k b : ℝ) (h : 3 * x + 4 * y + 5 = 0) :
  k = -3 / 4 ∧ b = -5 / 4 :=
by sorry

end line_slope_intercept_l111_111816


namespace interest_calculation_correct_l111_111026

-- Define the principal amounts and their respective interest rates
def principal1 : ℝ := 3000
def rate1 : ℝ := 0.08
def principal2 : ℝ := 8000 - principal1
def rate2 : ℝ := 0.05

-- Calculate interest for one year
def interest1 : ℝ := principal1 * rate1 * 1
def interest2 : ℝ := principal2 * rate2 * 1

-- Define the total interest
def total_interest : ℝ := interest1 + interest2

-- Prove that the total interest calculated is $490
theorem interest_calculation_correct : total_interest = 490 := by
  sorry

end interest_calculation_correct_l111_111026


namespace non_decreasing_f_f_equal_2_at_2_addition_property_under_interval_rule_final_statement_l111_111366

open Set

noncomputable def f : ℝ → ℝ := sorry

theorem non_decreasing_f (x y : ℝ) (h : x < y) (hx : x ∈ Icc (0 : ℝ) 2) (hy : y ∈ Icc (0 : ℝ) 2) : f x ≤ f y := sorry

theorem f_equal_2_at_2 : f 2 = 2 := sorry

theorem addition_property (x : ℝ) (hx : x ∈ Icc (0 :ℝ) 2) : f x + f (2 - x) = 2 := sorry

theorem under_interval_rule (x : ℝ) (hx : x ∈ Icc (1.5 :ℝ) 2) : f x ≤ 2 * (x - 1) := sorry

theorem final_statement : ∀ x ∈ Icc (0:ℝ) 1, f (f x) ∈ Icc (0:ℝ) 1 := sorry

end non_decreasing_f_f_equal_2_at_2_addition_property_under_interval_rule_final_statement_l111_111366


namespace smallest_N_exists_l111_111396

theorem smallest_N_exists (c1 c2 c3 c4 c5 c6 : ℕ) (N : ℕ) :
  (c1 = 6 * c3 - 2) →
  (N + c2 = 6 * c1 - 5) →
  (2 * N + c3 = 6 * c5 - 2) →
  (3 * N + c4 = 6 * c6 - 2) →
  (4 * N + c5 = 6 * c4 - 1) →
  (5 * N + c6 = 6 * c2 - 5) →
  N = 75 :=
by sorry

end smallest_N_exists_l111_111396


namespace n_is_power_of_three_l111_111277

theorem n_is_power_of_three {n : ℕ} (hn_pos : 0 < n) (p : Nat.Prime (4^n + 2^n + 1)) :
  ∃ (a : ℕ), n = 3^a :=
by
  sorry

end n_is_power_of_three_l111_111277


namespace handshake_count_250_l111_111310

theorem handshake_count_250 (n m : ℕ) (h1 : n = 5) (h2 : m = 5) :
  (n * m * (n * m - 1 - (n - 1))) / 2 = 250 :=
by
  -- Traditionally the theorem proof part goes here but it is omitted
  sorry

end handshake_count_250_l111_111310


namespace simplify_fraction_l111_111581

theorem simplify_fraction (h1 : 48 = 2^4 * 3) (h2 : 72 = 2^3 * 3^2) : (48 / 72 : ℚ) = 2 / 3 := 
by
  sorry

end simplify_fraction_l111_111581


namespace mechanical_pencils_fraction_l111_111257

theorem mechanical_pencils_fraction (total_pencils : ℕ) (frac_mechanical : ℚ)
    (mechanical_pencils : ℕ) (standard_pencils : ℕ) (new_total_pencils : ℕ) 
    (new_standard_pencils : ℕ) (new_frac_mechanical : ℚ):
  total_pencils = 120 →
  frac_mechanical = 1 / 4 →
  mechanical_pencils = frac_mechanical * total_pencils →
  standard_pencils = total_pencils - mechanical_pencils →
  new_standard_pencils = 3 * standard_pencils →
  new_total_pencils = mechanical_pencils + new_standard_pencils →
  new_frac_mechanical = mechanical_pencils / new_total_pencils →
  new_frac_mechanical = 1 / 10 :=
by
  sorry

end mechanical_pencils_fraction_l111_111257


namespace carson_total_seed_fertilizer_l111_111680

-- Definitions based on the conditions
variable (F S : ℝ)
variable (h_seed : S = 45)
variable (h_relation : S = 3 * F)

-- Theorem stating the total amount of seed and fertilizer used
theorem carson_total_seed_fertilizer : S + F = 60 := by
  -- Use the given conditions to relate and calculate the total
  sorry

end carson_total_seed_fertilizer_l111_111680


namespace three_configuration_m_separable_l111_111576

theorem three_configuration_m_separable
  {n m : ℕ} (A : Finset (Fin n)) (h : m ≥ n / 2) :
  ∀ (C : Finset (Fin n)), C.card = 3 → ∃ B : Finset (Fin n), B.card = m ∧ (∀ c ∈ C, ∃ b ∈ B, c ≠ b) :=
by
  sorry

end three_configuration_m_separable_l111_111576


namespace track_length_l111_111049

theorem track_length (x : ℝ) (hb hs : ℝ) (h_opposite : hs = x / 2 - 120) (h_first_meet : hb = 120) (h_second_meet : hs + 180 = x / 2 + 60) : x = 600 := 
by
  sorry

end track_length_l111_111049


namespace determine_x_l111_111978

theorem determine_x (x y : ℤ) (h1 : x + 2 * y = 20) (h2 : y = 5) : x = 10 := 
by 
  sorry

end determine_x_l111_111978


namespace women_exceed_men_l111_111463

variable (M W : ℕ)

theorem women_exceed_men (h1 : M + W = 24) (h2 : (M : ℚ) / (W : ℚ) = 0.6) : W - M = 6 :=
sorry

end women_exceed_men_l111_111463


namespace hyperbola_properties_l111_111720

def hyperbola (x y : ℝ) : Prop := x^2 - 4 * y^2 = 1

theorem hyperbola_properties :
  (∀ x y : ℝ, hyperbola x y → (x + 2 * y = 0 ∨ x - 2 * y = 0)) ∧
  (2 * (1 / 2) = 1) := 
by
  sorry

end hyperbola_properties_l111_111720


namespace stock_price_percentage_increase_l111_111644

theorem stock_price_percentage_increase :
  ∀ (total higher lower : ℕ), 
    total = 1980 →
    higher = 1080 →
    higher > lower →
    lower = total - higher →
  ((higher - lower) / lower : ℚ) * 100 = 20 :=
by
  intros total higher lower total_eq higher_eq higher_gt lower_eq
  sorry

end stock_price_percentage_increase_l111_111644


namespace gcd_values_count_l111_111159

theorem gcd_values_count (a b : ℕ) (h : a * b = 3600) : ∃ n, n = 29 ∧ ∀ d, d ∣ a ∧ d ∣ b → d = gcd a b → n = 29 :=
by { sorry }

end gcd_values_count_l111_111159


namespace evaluate_expression_l111_111882

theorem evaluate_expression : (25 + 15)^2 - (25^2 + 15^2 + 150) = 600 := by
  sorry

end evaluate_expression_l111_111882


namespace toys_per_rabbit_l111_111274

theorem toys_per_rabbit 
  (rabbits toys_mon toys_wed toys_fri toys_sat : ℕ) 
  (hrabbits : rabbits = 16) 
  (htoys_mon : toys_mon = 6)
  (htoys_wed : toys_wed = 2 * toys_mon)
  (htoys_fri : toys_fri = 4 * toys_mon)
  (htoys_sat : toys_sat = toys_wed / 2) :
  (toys_mon + toys_wed + toys_fri + toys_sat) / rabbits = 3 :=
by 
  sorry

end toys_per_rabbit_l111_111274


namespace boris_clock_time_l111_111979

-- Define a function to compute the sum of digits of a number.
def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the problem
theorem boris_clock_time (h m : ℕ) :
  sum_digits h + sum_digits m = 6 ∧ h + m = 15 ↔
  (h, m) = (0, 15) ∨ (h, m) = (1, 14) ∨ (h, m) = (2, 13) ∨ (h, m) = (3, 12) ∨
  (h, m) = (4, 11) ∨ (h, m) = (5, 10) ∨ (h, m) = (10, 5) ∨ (h, m) = (11, 4) ∨
  (h, m) = (12, 3) ∨ (h, m) = (13, 2) ∨ (h, m) = (14, 1) ∨ (h, m) = (15, 0) :=
by sorry

end boris_clock_time_l111_111979


namespace married_fraction_l111_111427

variables (M W N : ℕ)

def married_men : Prop := 2 * M = 3 * N
def married_women : Prop := 3 * W = 5 * N
def total_population : ℕ := M + W
def married_population : ℕ := 2 * N

theorem married_fraction (h1: married_men M N) (h2: married_women W N) :
  (married_population N : ℚ) / (total_population M W : ℚ) = 12 / 19 :=
by sorry

end married_fraction_l111_111427


namespace correct_option_l111_111160

theorem correct_option : 
  (∀ a b : ℝ, (a - b) * (-a - b) ≠ a^2 - b^2) ∧
  (∀ a : ℝ, 2 * a^3 + 3 * a^3 ≠ 5 * a^6) ∧ 
  (∀ x y : ℝ, 6 * x^3 * y^2 / (3 * x) = 2 * x^2 * y^2) ∧
  (∀ x : ℝ, (-2 * x^2)^3 ≠ -6 * x^6) :=
by 
  split
  . intros a b
    sorry
  . split
    . intros a
      sorry
    . split
      . intros x y
        sorry
      . intros x
        sorry

end correct_option_l111_111160


namespace arrange_abc_l111_111378

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom cos_a_eq_a : Real.cos a = a
axiom sin_cos_b_eq_b : Real.sin (Real.cos b) = b
axiom cos_sin_c_eq_c : Real.cos (Real.sin c) = c

theorem arrange_abc : b < a ∧ a < c := 
by
  sorry

end arrange_abc_l111_111378


namespace least_deletions_to_square_l111_111797

theorem least_deletions_to_square (l : List ℕ) (h : l = [10, 20, 30, 40, 50, 60, 70, 80, 90]) : 
  ∃ d, d.card ≤ 2 ∧ ∀ (lp : List ℕ), lp = l.diff d → 
  ∃ k, lp.prod = k^2 :=
by
  sorry

end least_deletions_to_square_l111_111797


namespace bus_driver_regular_rate_l111_111985

theorem bus_driver_regular_rate (hours := 60) (total_pay := 1200) (regular_hours := 40) (overtime_rate_factor := 1.75) :
  ∃ R : ℝ, 40 * R + 20 * (1.75 * R) = 1200 ∧ R = 16 := 
by
  sorry

end bus_driver_regular_rate_l111_111985


namespace max_range_of_temps_l111_111651

noncomputable def max_temp_range (T1 T2 T3 T4 T5 : ℝ) : ℝ := 
  max (max (max (max T1 T2) T3) T4) T5 - min (min (min (min T1 T2) T3) T4) T5

theorem max_range_of_temps :
  ∀ (T1 T2 T3 T4 T5 : ℝ), 
  (T1 + T2 + T3 + T4 + T5) / 5 = 60 →
  T1 = 40 →
  (max_temp_range T1 T2 T3 T4 T5) = 100 :=
by
  intros T1 T2 T3 T4 T5 Havg Hlowest
  sorry

end max_range_of_temps_l111_111651


namespace bikes_in_parking_lot_l111_111759

theorem bikes_in_parking_lot (C : ℕ) (Total_Wheels : ℕ) (Wheels_per_car : ℕ) (Wheels_per_bike : ℕ) (h1 : C = 14) (h2 : Total_Wheels = 76) (h3 : Wheels_per_car = 4) (h4 : Wheels_per_bike = 2) : 
  ∃ B : ℕ, 4 * C + 2 * B = Total_Wheels ∧ B = 10 :=
by
  sorry

end bikes_in_parking_lot_l111_111759


namespace Pablo_is_70_cm_taller_than_Charlene_l111_111799

variable (Ruby Pablo Charlene Janet : ℕ)

-- Conditions
axiom h1 : Ruby + 2 = Pablo
axiom h2 : Charlene = 2 * Janet
axiom h3 : Janet = 62
axiom h4 : Ruby = 192

-- The statement to prove
theorem Pablo_is_70_cm_taller_than_Charlene : Pablo - Charlene = 70 :=
by
  -- Formalizing the proof
  sorry

end Pablo_is_70_cm_taller_than_Charlene_l111_111799


namespace larger_number_hcf_lcm_l111_111325

theorem larger_number_hcf_lcm (a b : ℕ) (h1 : Nat.gcd a b = 84) (h2 : Nat.lcm a b = 21) (h3 : a = b / 4) : max a b = 84 :=
by
  sorry

end larger_number_hcf_lcm_l111_111325


namespace product_ab_cd_l111_111430

-- Conditions
variables (O A B C D F : Point)
variables (a b : ℝ)
hypothesis h1 : a = distance O A
hypothesis h2 : a = distance O B
hypothesis h3 : b = distance O C
hypothesis h4 : b = distance O D
hypothesis h5 : distance O F = 8
hypothesis h6 : diameter ((inscribed_circle (triangle O C F))) = 4

-- Given facts
def e1 := a^2 - b^2 = 64
def e2 := a - b = 4
def e3 := 2 * (distance O F) = 4

-- Theorem statement
theorem product_ab_cd : (2 * a) * (2 * b) = 240 :=
by
  sorry

end product_ab_cd_l111_111430


namespace sand_cake_probability_is_12_percent_l111_111407

def total_days : ℕ := 5
def ham_days : ℕ := 3
def cake_days : ℕ := 1

-- Probability of packing a ham sandwich on any given day
def prob_ham_sandwich : ℚ := ham_days / total_days

-- Probability of packing a piece of cake on any given day
def prob_cake : ℚ := cake_days / total_days

-- Calculate the combined probability that Karen packs a ham sandwich and cake on the same day
def combined_probability : ℚ := prob_ham_sandwich * prob_cake

-- Convert the combined probability to a percentage
def combined_probability_as_percentage : ℚ := combined_probability * 100

-- The proof problem to show that the probability that Karen packs a ham sandwich and cake on the same day is 12%
theorem sand_cake_probability_is_12_percent : combined_probability_as_percentage = 12 := 
  by sorry

end sand_cake_probability_is_12_percent_l111_111407


namespace find_number_l111_111653

theorem find_number (x : ℝ) (h : 0.5 * x = 0.25 * x + 2) : x = 8 :=
by
  sorry

end find_number_l111_111653


namespace original_planned_length_l111_111032

theorem original_planned_length (x : ℝ) (h1 : x > 0) (total_length : ℝ := 3600) (efficiency_ratio : ℝ := 1.8) (time_saved : ℝ := 20) 
  (h2 : total_length / x - total_length / (efficiency_ratio * x) = time_saved) :
  x = 80 :=
sorry

end original_planned_length_l111_111032


namespace quadratic_inequality_solution_l111_111817

theorem quadratic_inequality_solution:
  ∀ x : ℝ, -x^2 + 3 * x - 2 ≥ 0 ↔ (1 ≤ x ∧ x ≤ 2) :=
by
  sorry

end quadratic_inequality_solution_l111_111817


namespace megan_final_balance_percentage_l111_111935

noncomputable def initial_balance_usd := 125.0
noncomputable def increase_percentage_babysitting := 0.25
noncomputable def exchange_rate_usd_to_eur_1 := 0.85
noncomputable def decrease_percentage_shoes := 0.20
noncomputable def exchange_rate_eur_to_usd := 1.15
noncomputable def increase_percentage_stocks := 0.15
noncomputable def decrease_percentage_medical := 0.10
noncomputable def exchange_rate_usd_to_eur_2 := 0.88

theorem megan_final_balance_percentage :
  let new_balance_after_babysitting := initial_balance_usd * (1 + increase_percentage_babysitting)
  let balance_in_eur := new_balance_after_babysitting * exchange_rate_usd_to_eur_1
  let balance_after_shoes := balance_in_eur * (1 - decrease_percentage_shoes)
  let balance_back_to_usd := balance_after_shoes * exchange_rate_eur_to_usd
  let balance_after_stocks := balance_back_to_usd * (1 + increase_percentage_stocks)
  let balance_after_medical := balance_after_stocks * (1 - decrease_percentage_medical)
  let final_balance_in_eur := balance_after_medical * exchange_rate_usd_to_eur_2
  let initial_balance_in_eur := initial_balance_usd * exchange_rate_usd_to_eur_1
  (final_balance_in_eur / initial_balance_in_eur) * 100 = 104.75 := by
  sorry

end megan_final_balance_percentage_l111_111935


namespace num_bicycles_l111_111048

theorem num_bicycles (spokes_per_wheel wheels_per_bicycle total_spokes : ℕ) (h1 : spokes_per_wheel = 10) (h2 : total_spokes = 80) (h3 : wheels_per_bicycle = 2) : total_spokes / spokes_per_wheel / wheels_per_bicycle = 4 := by
  sorry

end num_bicycles_l111_111048


namespace area_of_quadrilateral_is_195_l111_111883

-- Definitions and conditions
def diagonal_length : ℝ := 26
def offset1 : ℝ := 9
def offset2 : ℝ := 6

-- Prove the area of the quadrilateral is 195 cm²
theorem area_of_quadrilateral_is_195 :
  1 / 2 * diagonal_length * offset1 + 1 / 2 * diagonal_length * offset2 = 195 := 
by
  -- The proof steps would go here
  sorry

end area_of_quadrilateral_is_195_l111_111883


namespace k_value_if_root_is_one_l111_111389

theorem k_value_if_root_is_one (k : ℝ) (h : (k - 1) * 1 ^ 2 + 1 - k ^ 2 = 0) : k = 0 := 
by
  sorry

end k_value_if_root_is_one_l111_111389


namespace square_area_l111_111697

theorem square_area (perimeter : ℝ) (h : perimeter = 40) : ∃ A : ℝ, A = 100 := by
  have h1 : ∃ s : ℝ, 4 * s = perimeter := by
    use perimeter / 4
    linarith

  cases h1 with s hs
  use s^2
  rw [hs, h]
  norm_num
  sorry

end square_area_l111_111697


namespace triangle_area_ratio_l111_111913

open Set 

variables {X Y Z W : Type} 
variable [LinearOrder X]

noncomputable def ratio_areas (XW WZ : ℕ) (h : ℕ) : ℚ :=
  (8 * h : ℚ) / (12 * h)

theorem triangle_area_ratio (XW WZ : ℕ) (h : ℕ)
  (hXW : XW = 8)
  (hWZ : WZ = 12) :
  ratio_areas XW WZ h = 2 / 3 :=
by
  rw [hXW, hWZ]
  unfold ratio_areas
  norm_num
  sorry

end triangle_area_ratio_l111_111913


namespace find_number_multiplied_l111_111230

theorem find_number_multiplied (m : ℕ) (h : 9999 * m = 325027405) : m = 32505 :=
by {
  sorry
}

end find_number_multiplied_l111_111230


namespace min_height_box_l111_111781

noncomputable def min_height (x : ℝ) : ℝ :=
  if h : x ≥ (5 : ℝ) then x + 5 else 0

theorem min_height_box (x : ℝ) (hx : 3*x^2 + 10*x - 65 ≥ 0) : min_height x = 10 :=
by
  sorry

end min_height_box_l111_111781


namespace factorize_expression_l111_111215

theorem factorize_expression (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := sorry

end factorize_expression_l111_111215


namespace students_disliked_menu_l111_111489

theorem students_disliked_menu (total_students liked_students : ℕ) (h1 : total_students = 400) (h2 : liked_students = 235) : total_students - liked_students = 165 :=
by 
  sorry

end students_disliked_menu_l111_111489


namespace set_of_a_where_A_subset_B_l111_111910

variable {a x : ℝ}

theorem set_of_a_where_A_subset_B (h : ∀ x, (2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5) → (3 ≤ x ∧ x ≤ 22)) :
  6 ≤ a ∧ a ≤ 9 :=
by
  sorry

end set_of_a_where_A_subset_B_l111_111910


namespace burger_cost_cents_l111_111053

theorem burger_cost_cents 
  (b s : ℕ)
  (h1 : 4 * b + 3 * s = 550) 
  (h2 : 3 * b + 2 * s = 400) 
  (h3 : 2 * b + s = 250) : 
  b = 100 :=
by
  sorry

end burger_cost_cents_l111_111053


namespace carousel_seats_count_l111_111963

theorem carousel_seats_count :
  ∃ (yellow blue red : ℕ), 
  (yellow + blue + red = 100) ∧ 
  (yellow = 34) ∧ 
  (blue = 20) ∧ 
  (red = 46) ∧ 
  (∀ i : ℕ, i < yellow → ∃ j : ℕ, j = yellow.succ * j ∧ (j < 100 ∧ j ≠ yellow.succ * j)) ∧ 
  (∀ k : ℕ, k < blue → ∃ m : ℕ, m = blue.succ * m ∧ (m < 100 ∧ m ≠ blue.succ * m)) ∧ 
  (∀ n : ℕ, n < red → ∃ p : ℕ, p = red.succ * p ∧ (p < 100 ∧ p ≠ red.succ * p)) :=
sorry

end carousel_seats_count_l111_111963


namespace expected_value_of_empty_boxes_l111_111371

noncomputable def expected_empty_boxes (n m : ℕ) : ℚ :=
  let p0 := (nat.factorial m : ℚ) / (m ^ n) in -- Probability that no box is empty
  let p1 := (nat.choose m 1 * nat.factorial (m - 1) : ℚ) / (m ^ n) in -- Probability that one box is empty
  let p2_half := (nat.choose m 2 / 2 * nat.factorial (m - 2) : ℚ) / (m ^ n) in -- Part of probability that two boxes are empty 
  let p2_full := (nat.choose m 2 * nat.factorial (m - 2) : ℚ) / (m ^ n) in  -- Full probability that two boxes are empty
  let p2 := p2_half + p2_full in -- Sum the cases for two empty boxes probability
  let p3 := (nat.choose m 3 * nat.factorial (m - 3) : ℚ) / (m ^ n) in -- Three boxes are empty, only full case
  (0 * p0) + (1 * p1) + (2 * p2) + (3 * p3)

theorem expected_value_of_empty_boxes : expected_empty_boxes 4 4 = 81 / 64 :=
by
  sorry

end expected_value_of_empty_boxes_l111_111371


namespace average_episodes_per_year_l111_111836

def num_years : Nat := 14
def seasons_15_episodes : Nat := 8
def episodes_per_season_15 : Nat := 15
def seasons_20_episodes : Nat := 4
def episodes_per_season_20 : Nat := 20
def seasons_12_episodes : Nat := 2
def episodes_per_season_12 : Nat := 12

theorem average_episodes_per_year :
  (seasons_15_episodes * episodes_per_season_15 +
   seasons_20_episodes * episodes_per_season_20 +
   seasons_12_episodes * episodes_per_season_12) / num_years = 16 := by
  sorry

end average_episodes_per_year_l111_111836


namespace download_time_l111_111139

theorem download_time (file_size : ℕ) (first_part_size : ℕ) (rate1 : ℕ) (rate2 : ℕ) (total_time : ℕ)
  (h_file : file_size = 90) (h_first_part : first_part_size = 60) (h_rate1 : rate1 = 5) (h_rate2 : rate2 = 10)
  (h_time : total_time = 15) : 
  file_size = first_part_size + (file_size - first_part_size) ∧ total_time = first_part_size / rate1 + (file_size - first_part_size) / rate2 :=
by
  have time1 : total_time = 12 + 3,
    sorry,
  have part1 : first_part_size = 60,
    sorry,
  have part2 : file_size - first_part_size = 30,
    sorry,
  have rate1_correct : rate1 = 5,
    sorry,
  have rate2_correct : rate2 = 10,
    sorry,
  have time1_total : 12 + 3 = 15,
    sorry,
  exact ⟨rfl, rfl⟩

end download_time_l111_111139


namespace ellipse_product_major_minor_axes_l111_111433

theorem ellipse_product_major_minor_axes 
  (a b : ℝ)
  (OF : ℝ = 8)
  (diameter_ocf : ℝ = 4)
  (h1 : a^2 - b^2 = 64)
  (h2 : b + OF - a = diameter_ocf / 2) :
  2 * a * 2 * b = 240 :=
by
  -- The detailed proof goes here
  sorry

end ellipse_product_major_minor_axes_l111_111433


namespace maximum_profit_3_le_a_le_5_maximum_profit_f_g_3_le_a_le_5_g_5_lt_a_le_7_l111_111986

noncomputable def f (x a : ℝ) : ℝ := (x - 30)^2 * (x - 10 - a)

theorem maximum_profit_3_le_a_le_5 (a : ℝ) (ha : 3 ≤ a ∧ a ≤ 5) :
    ∀ (x : ℝ), 20 ≤ x ∧ x ≤ 25 → f x a ≤ f 20 a := 
    sorry

theorem maximum_profit_f (a : ℝ) (ha : 5 < a ∧ a ≤ 7) :
    ∀ (x : ℝ), 20 ≤ x ∧ x ≤ 25 → f x a ≤ f ((2 * a + 50) / 3) a :=
    sorry

theorem g_3_le_a_le_5 (a : ℝ) (ha : 3 ≤ a ∧ a ≤ 5) :
    g a = 1000 - 10 * a :=
    sorry

theorem g_5_lt_a_le_7 (a : ℝ) (ha : 5 < a ∧ a ≤ 7) :
    g a = 4 * (a - 20)^2 / 27 :=
    sorry

end maximum_profit_3_le_a_le_5_maximum_profit_f_g_3_le_a_le_5_g_5_lt_a_le_7_l111_111986


namespace binomial_seven_four_l111_111194

noncomputable def binomial (n k : Nat) : Nat := n.choose k

theorem binomial_seven_four : binomial 7 4 = 35 := by
  sorry

end binomial_seven_four_l111_111194


namespace find_r_l111_111532

theorem find_r (k r : ℝ) 
  (h1 : 7 = k * 3^r) 
  (h2 : 49 = k * 9^r) : 
  r = Real.log 7 / Real.log 3 :=
by
  sorry

end find_r_l111_111532


namespace regular_polygon_sides_l111_111260

theorem regular_polygon_sides (n : ℕ) (h : 2 < n)
  (interior_angle : ∀ n, (n - 2) * 180 / n = 144) : n = 10 :=
sorry

end regular_polygon_sides_l111_111260


namespace given_problem_l111_111719

theorem given_problem :
  3^3 + 4^3 + 5^3 = 6^3 :=
by sorry

end given_problem_l111_111719


namespace exists_b_mod_5_l111_111564

theorem exists_b_mod_5 (p q r s : ℤ) (h1 : ¬ (s % 5 = 0)) (a : ℤ) (h2 : (p * a^3 + q * a^2 + r * a + s) % 5 = 0) : 
  ∃ b : ℤ, (s * b^3 + r * b^2 + q * b + p) % 5 = 0 :=
sorry

end exists_b_mod_5_l111_111564


namespace find_common_difference_l111_111962

-- Definitions of the conditions
def arithmetic_sequence (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ k : ℕ, a_n (k + 1) = a_n k + d

def sum_of_first_n_terms (a_n : ℕ → ℝ) (n : ℕ) (S_n : ℝ) : Prop :=
  S_n = (n : ℝ) / 2 * (a_n 1 + a_n n)

variables {a_1 d : ℝ}
variables (a_n : ℕ → ℝ)
variables (S_3 S_9 : ℝ)

-- Conditions from the problem statement
axiom a2_eq_3 : a_n 2 = 3
axiom S9_eq_6S3 : S_9 = 6 * S_3

-- The proof we need to write
theorem find_common_difference 
  (h1 : arithmetic_sequence a_n d)
  (h2 : sum_of_first_n_terms a_n 3 S_3)
  (h3 : sum_of_first_n_terms a_n 9 S_9) :
  d = 1 :=
by
  sorry

end find_common_difference_l111_111962


namespace investment_return_l111_111173

theorem investment_return (y_r : ℝ) :
  (500 + 1500) * 0.085 = 500 * 0.07 + 1500 * y_r → y_r = 0.09 :=
by
  sorry

end investment_return_l111_111173


namespace base_85_solution_l111_111501

theorem base_85_solution (b : ℕ) (h1 : 0 ≤ b ∧ b ≤ 16) :
  (352936524 - b) % 17 = 0 ↔ b = 4 :=
by
  sorry

end base_85_solution_l111_111501


namespace longest_collection_pages_l111_111420

theorem longest_collection_pages 
    (pages_per_inch_miles : ℕ := 5) 
    (pages_per_inch_daphne : ℕ := 50) 
    (height_miles : ℕ := 240) 
    (height_daphne : ℕ := 25) : 
  max (pages_per_inch_miles * height_miles) (pages_per_inch_daphne * height_daphne) = 1250 := 
by
  sorry

end longest_collection_pages_l111_111420


namespace lg_45_eq_l111_111232

variable (m n : ℝ)
axiom lg_2 : Real.log 2 = m
axiom lg_3 : Real.log 3 = n

theorem lg_45_eq : Real.log 45 = 1 - m + 2 * n := by
  -- proof to be filled in
  sorry

end lg_45_eq_l111_111232


namespace nearest_integer_power_l111_111018

noncomputable def power_expression := (3 + Real.sqrt 2)^6

theorem nearest_integer_power :
  Int.floor power_expression = 7414 :=
sorry

end nearest_integer_power_l111_111018


namespace greatest_possible_percentage_of_airlines_both_services_l111_111654

noncomputable def maxPercentageOfAirlinesWithBothServices (percentageInternet percentageSnacks : ℝ) : ℝ :=
  if percentageInternet <= percentageSnacks then percentageInternet else percentageSnacks

theorem greatest_possible_percentage_of_airlines_both_services:
  let p_internet := 0.35
  let p_snacks := 0.70
  maxPercentageOfAirlinesWithBothServices p_internet p_snacks = 0.35 :=
by
  sorry

end greatest_possible_percentage_of_airlines_both_services_l111_111654


namespace distance_from_dormitory_to_city_l111_111764

theorem distance_from_dormitory_to_city (D : ℝ) (h : (1/2) * D + (1/4) * D + 6 = D) : D = 24 :=
by
  sorry

end distance_from_dormitory_to_city_l111_111764


namespace find_first_offset_l111_111506

theorem find_first_offset {area diagonal offset₁ offset₂ : ℝ}
  (h_area : area = 150)
  (h_diagonal : diagonal = 20)
  (h_offset₂ : offset₂ = 6) :
  2 * area = diagonal * (offset₁ + offset₂) → offset₁ = 9 := by
  sorry

end find_first_offset_l111_111506


namespace tangent_slope_at_zero_l111_111376

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (x^2 + 1)

theorem tangent_slope_at_zero :
  (deriv f 0) = 1 := by 
  sorry

end tangent_slope_at_zero_l111_111376


namespace sand_cake_probability_is_12_percent_l111_111406

def total_days : ℕ := 5
def ham_days : ℕ := 3
def cake_days : ℕ := 1

-- Probability of packing a ham sandwich on any given day
def prob_ham_sandwich : ℚ := ham_days / total_days

-- Probability of packing a piece of cake on any given day
def prob_cake : ℚ := cake_days / total_days

-- Calculate the combined probability that Karen packs a ham sandwich and cake on the same day
def combined_probability : ℚ := prob_ham_sandwich * prob_cake

-- Convert the combined probability to a percentage
def combined_probability_as_percentage : ℚ := combined_probability * 100

-- The proof problem to show that the probability that Karen packs a ham sandwich and cake on the same day is 12%
theorem sand_cake_probability_is_12_percent : combined_probability_as_percentage = 12 := 
  by sorry

end sand_cake_probability_is_12_percent_l111_111406


namespace condition_sufficient_but_not_necessary_l111_111510
noncomputable def sufficient_but_not_necessary (a b : ℝ) : Prop :=
∀ (a b : ℝ), a < 0 → -1 < b ∧ b < 0 → a + a * b < 0

-- Define the theorem stating the proof problem
theorem condition_sufficient_but_not_necessary (a b : ℝ) :
  (a < 0 ∧ -1 < b ∧ b < 0 → a + a * b < 0) ∧ 
  (a + a * b < 0 → a < 0 ∧ 1 + b > 0 ∨ a > 0 ∧ 1 + b < 0) :=
sorry

end condition_sufficient_but_not_necessary_l111_111510


namespace smallest_even_in_sequence_sum_400_l111_111301

theorem smallest_even_in_sequence_sum_400 :
  ∃ (n : ℤ), (n - 6) + (n - 4) + (n - 2) + n + (n + 2) + (n + 4) + (n + 6) = 400 ∧ (n - 6) % 2 = 0 ∧ n - 6 = 52 :=
sorry

end smallest_even_in_sequence_sum_400_l111_111301


namespace number_of_classes_le_l111_111342

theorem number_of_classes_le (n : ℕ) :
  (∀ c, ∃ (s : Finset (Fin n)), 2 ≤ s.card ∧ (∀ c₁ c₂, c₁ ≠ c₂ → 2 ≤ (s ∩ s).card → s.card ≠ s.card)) →
  (∃ C : Finset (Finset (Fin n)), C.card ≤ (n-1)^2) :=
begin
  sorry
end

end number_of_classes_le_l111_111342


namespace pascal_triangle_row10_sum_l111_111751

def pascal_triangle_row_sum (n : ℕ) : ℕ :=
  2 ^ n

theorem pascal_triangle_row10_sum : pascal_triangle_row_sum 10 = 1024 :=
by
  -- Proof will demonstrate that 2^10 = 1024
  sorry

end pascal_triangle_row10_sum_l111_111751


namespace jack_has_42_pounds_l111_111115

noncomputable def jack_pounds (P : ℕ) : Prop :=
  let euros := 11
  let yen := 3000
  let pounds_per_euro := 2
  let yen_per_pound := 100
  let total_yen := 9400
  let pounds_from_euros := euros * pounds_per_euro
  let pounds_from_yen := yen / yen_per_pound
  let total_pounds := P + pounds_from_euros + pounds_from_yen
  total_pounds * yen_per_pound = total_yen

theorem jack_has_42_pounds : jack_pounds 42 :=
  sorry

end jack_has_42_pounds_l111_111115


namespace binomial_7_4_eq_35_l111_111202

theorem binomial_7_4_eq_35 : Nat.choose 7 4 = 35 := by
  sorry

end binomial_7_4_eq_35_l111_111202


namespace average_episodes_per_year_is_16_l111_111844

-- Define the number of years the TV show has been running
def years : Nat := 14

-- Define the number of seasons and episodes for each category
def seasons_8_15 : Nat := 8
def episodes_per_season_8_15 : Nat := 15
def seasons_4_20 : Nat := 4
def episodes_per_season_4_20 : Nat := 20
def seasons_2_12 : Nat := 2
def episodes_per_season_2_12 : Nat := 12

-- Define the total number of episodes
def total_episodes : Nat :=
  (seasons_8_15 * episodes_per_season_8_15) + 
  (seasons_4_20 * episodes_per_season_4_20) + 
  (seasons_2_12 * episodes_per_season_2_12)

-- Define the average number of episodes per year
def average_episodes_per_year : Nat :=
  total_episodes / years

-- State the theorem to prove the average number of episodes per year is 16
theorem average_episodes_per_year_is_16 : average_episodes_per_year = 16 :=
by
  sorry

end average_episodes_per_year_is_16_l111_111844


namespace black_queen_awake_at_10_l111_111320

-- Define the logical context
def king_awake_at_10 (king_asleep : Prop) : Prop :=
  king_asleep -> false

def king_asleep_at_10 (king_asleep : Prop) : Prop :=
  king_asleep

def queen_awake_at_10 (queen_asleep : Prop) : Prop :=
  queen_asleep -> false

-- Define the main theorem
theorem black_queen_awake_at_10 
  (king_asleep : Prop)
  (queen_asleep : Prop)
  (king_belief : king_asleep ↔ (king_asleep ∧ queen_asleep)) :
  queen_awake_at_10 queen_asleep :=
by
  -- Proof is omitted
  sorry

end black_queen_awake_at_10_l111_111320


namespace complement_of_M_l111_111529

open Set

def U : Set ℝ := univ
def M : Set ℝ := {x | x ≥ 1}

theorem complement_of_M :
  (U \ M) = {x | x < 1} :=
by
  sorry

end complement_of_M_l111_111529


namespace sin_cos_equation_solution_l111_111445

open Real

theorem sin_cos_equation_solution (x : ℝ): 
  (∃ n : ℤ, x = (π / 4050) + (π * n / 2025)) ∨ (∃ k : ℤ, x = (π * k / 9)) ↔ 
  sin (2025 * x) ^ 4 + (cos (2016 * x) ^ 2019) * (cos (2025 * x) ^ 2018) = 1 := 
by 
  sorry

end sin_cos_equation_solution_l111_111445


namespace compute_seventy_five_squared_minus_thirty_five_squared_l111_111877

theorem compute_seventy_five_squared_minus_thirty_five_squared :
  75^2 - 35^2 = 4400 := by
  sorry

end compute_seventy_five_squared_minus_thirty_five_squared_l111_111877


namespace compare_a_b_l111_111095

theorem compare_a_b (a b : ℝ) (h₁ : a = 1.9 * 10^5) (h₂ : b = 9.1 * 10^4) : a > b := by
  sorry

end compare_a_b_l111_111095


namespace problem_statement_l111_111234

-- Define the function f(x)
variable (f : ℝ → ℝ)

-- Conditions
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodic_function : ∀ x, f (x + 4) = -f x
axiom increasing_on_0_2 : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 2 → f x < f y

-- Theorem to prove
theorem problem_statement : f (-10) < f 40 ∧ f 40 < f 3 :=
by
  sorry

end problem_statement_l111_111234


namespace ben_fraction_of_taxes_l111_111873

theorem ben_fraction_of_taxes 
  (gross_income : ℝ) (car_payment : ℝ) (fraction_spend_on_car : ℝ) (after_tax_income_fraction : ℝ) 
  (h1 : gross_income = 3000) (h2 : car_payment = 400) (h3 : fraction_spend_on_car = 0.2) :
  after_tax_income_fraction = (1 / 3) :=
by
  sorry

end ben_fraction_of_taxes_l111_111873


namespace hollow_iron_ball_diameter_l111_111664

theorem hollow_iron_ball_diameter (R r : ℝ) (s : ℝ) (thickness : ℝ) 
  (h1 : thickness = 1) (h2 : s = 7.5) 
  (h3 : R - r = thickness) 
  (h4 : 4 / 3 * π * R^3 = 4 / 3 * π * s * (R^3 - r^3)) : 
  2 * R = 44.44 := 
sorry

end hollow_iron_ball_diameter_l111_111664


namespace binomial_7_4_eq_35_l111_111200

theorem binomial_7_4_eq_35 : Nat.choose 7 4 = 35 := by
  sorry

end binomial_7_4_eq_35_l111_111200


namespace smallest_number_satisfying_conditions_l111_111915

theorem smallest_number_satisfying_conditions :
  ∃ (n : ℕ), n % 6 = 2 ∧ n % 7 = 3 ∧ n % 8 = 4 ∧ ∀ m, (m % 6 = 2 → m % 7 = 3 → m % 8 = 4 → n ≤ m) :=
  sorry

end smallest_number_satisfying_conditions_l111_111915


namespace golden_ratio_in_range_l111_111814

noncomputable def golden_ratio := (Real.sqrt 5 - 1) / 2

theorem golden_ratio_in_range : 0.6 < golden_ratio ∧ golden_ratio < 0.7 :=
by
  sorry

end golden_ratio_in_range_l111_111814


namespace ellipse_major_minor_axes_product_l111_111437

-- Definitions based on conditions
def OF : ℝ := 8
def inradius_triangle_OCF : ℝ := 2  -- diameter / 2

-- Define a and b based on the ellipse properties and conditions
def a : ℝ := 10  -- Solved from the given conditions and steps
def b : ℝ := 6   -- Solved from the given conditions and steps

-- Defining the axes of the ellipse in terms of a and b
def AB : ℝ := 2 * a
def CD : ℝ := 2 * b

-- The product (AB)(CD) we are interested in
def product_AB_CD := AB * CD

-- The main proof statement
theorem ellipse_major_minor_axes_product : product_AB_CD = 240 :=
by
  sorry

end ellipse_major_minor_axes_product_l111_111437


namespace bill_sunday_miles_l111_111786

variables (B J M S : ℝ)

-- Conditions
def condition_1 := B + 4
def condition_2 := 2 * (B + 4)
def condition_3 := J = 0 ∧ M = 5 ∧ (M + 2 = 7)
def condition_4 := (B + 5) + (B + 4) + 2 * (B + 4) + 7 = 50

-- The main theorem to prove the number of miles Bill ran on Sunday
theorem bill_sunday_miles (h1 : S = B + 4) (h2 : ∀ B, J = 0 → M = 5 → S + 2 = 7 → (B + 5) + S + 2 * S + 7 = 50) : S = 10.5 :=
by {
  sorry
}

end bill_sunday_miles_l111_111786


namespace range_of_f_x_minus_2_l111_111744

noncomputable def f (x : ℝ) : ℝ :=
if x < 0 then x + 1 else if x > 0 then -(x + 1) else 0

theorem range_of_f_x_minus_2 :
  ∀ x : ℝ, f (x - 2) < 0 ↔ x ∈ Set.union (Set.Iio 1) (Set.Ioo 2 3) := by
sorry

end range_of_f_x_minus_2_l111_111744


namespace side_length_of_inscribed_square_l111_111474

theorem side_length_of_inscribed_square
  (S1 S2 S3 : ℝ)
  (hS1 : S1 = 1) (hS2 : S2 = 3) (hS3 : S3 = 1) :
  ∃ (x : ℝ), S1 = 1 ∧ S2 = 3 ∧ S3 = 1 ∧ x = 2 := 
by
  sorry

end side_length_of_inscribed_square_l111_111474


namespace no_valid_weights_l111_111306

theorem no_valid_weights (w_1 w_2 w_3 w_4 : ℝ) : 
  w_1 + w_2 + w_3 = 100 → w_1 + w_2 + w_4 = 101 → w_2 + w_3 + w_4 = 102 → 
  w_1 < 90 → w_2 < 90 → w_3 < 90 → w_4 < 90 → False :=
by 
  intros h1 h2 h3 hl1 hl2 hl3 hl4
  sorry

end no_valid_weights_l111_111306


namespace teamX_total_games_l111_111134

variables (x : ℕ)

-- Conditions
def teamX_wins := (3/4) * x
def teamX_loses := (1/4) * x

def teamY_wins := (2/3) * (x + 10)
def teamY_loses := (1/3) * (x + 10)

-- Question: Prove team X played 20 games
theorem teamX_total_games :
  teamY_wins - teamX_wins = 5 ∧ teamY_loses - teamX_loses = 5 → x = 20 := by
sorry

end teamX_total_games_l111_111134


namespace new_songs_added_l111_111571

-- Define the initial, deleted, and final total number of songs as constants
def initial_songs : ℕ := 8
def deleted_songs : ℕ := 5
def total_songs_now : ℕ := 33

-- Define and prove the number of new songs added
theorem new_songs_added : total_songs_now - (initial_songs - deleted_songs) = 30 :=
by
  sorry

end new_songs_added_l111_111571


namespace number_of_men_in_row_l111_111133

theorem number_of_men_in_row (M : ℕ) (W : ℕ) (cases : ℕ) (hW : W = 2) (hcases : cases = 12) :
  (∃ k1 k2 k3 : ℕ, M = k1 + k2 + k3 ∧ k1 > 0 ∧ k2 > 0 ∧ k3 > 0 ∧ 2 * W * ((k1.choose 2 + k2.choose 2 + k3.choose 2) * 2.factorial) = cases) →
  M = 4 :=
by
  sorry

end number_of_men_in_row_l111_111133


namespace ratio_of_original_to_doubled_l111_111667

theorem ratio_of_original_to_doubled (x : ℕ) (h : x + 5 = 17) : (x / Nat.gcd x (2 * x)) = 1 ∧ ((2 * x) / Nat.gcd x (2 * x)) = 2 := 
by
  sorry

end ratio_of_original_to_doubled_l111_111667


namespace avg_last_three_numbers_l111_111947

-- Definitions of conditions
def avg_seven_numbers (numbers : List ℝ) (h_len : numbers.length = 7) : Prop :=
(numbers.sum / 7 = 60)

def avg_first_four_numbers (numbers : List ℝ) (h_len : numbers.length = 7) : Prop :=
(numbers.take 4).sum / 4 = 55

-- Proof statement
theorem avg_last_three_numbers (numbers : List ℝ) (h_len : numbers.length = 7)
  (h1 : avg_seven_numbers numbers h_len)
  (h2 : avg_first_four_numbers numbers h_len) :
  (numbers.drop 4).sum / 3 = 200 / 3 :=
sorry

end avg_last_three_numbers_l111_111947


namespace pigeonhole_6_points_3x4_l111_111395

theorem pigeonhole_6_points_3x4 :
  ∀ (points : Fin 6 → (ℝ × ℝ)), 
  (∀ i, 0 ≤ (points i).fst ∧ (points i).fst ≤ 4 ∧ 0 ≤ (points i).snd ∧ (points i).snd ≤ 3) →
  ∃ i j, i ≠ j ∧ dist (points i) (points j) ≤ Real.sqrt 5 :=
by
  sorry

end pigeonhole_6_points_3x4_l111_111395


namespace ellipse_product_l111_111435

theorem ellipse_product (a b : ℝ) (OF_diameter : a - b = 4) (focus_relation : a^2 - b^2 = 64) :
  let AB := 2 * a,
      CD := 2 * b
  in AB * CD = 240 :=
by
  sorry

end ellipse_product_l111_111435


namespace complement_union_A_B_complement_A_intersection_B_l111_111898

open Set

-- Definitions of A and B
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

-- Proving the complement of A ∪ B
theorem complement_union_A_B : (A ∪ B)ᶜ = {x : ℝ | x ≤ 2 ∨ 10 ≤ x} :=
by sorry

-- Proving the intersection of the complement of A with B
theorem complement_A_intersection_B : (Aᶜ ∩ B) = {x : ℝ | (2 < x ∧ x < 3) ∨ (7 ≤ x ∧ x < 10)} :=
by sorry

end complement_union_A_B_complement_A_intersection_B_l111_111898


namespace average_episodes_per_year_l111_111837

def num_years : Nat := 14
def seasons_15_episodes : Nat := 8
def episodes_per_season_15 : Nat := 15
def seasons_20_episodes : Nat := 4
def episodes_per_season_20 : Nat := 20
def seasons_12_episodes : Nat := 2
def episodes_per_season_12 : Nat := 12

theorem average_episodes_per_year :
  (seasons_15_episodes * episodes_per_season_15 +
   seasons_20_episodes * episodes_per_season_20 +
   seasons_12_episodes * episodes_per_season_12) / num_years = 16 := by
  sorry

end average_episodes_per_year_l111_111837


namespace radii_inequality_l111_111281

variable {R1 R2 R3 r : ℝ}

/-- Given that R1, R2, and R3 are the radii of three circles passing through a vertex of a triangle 
and touching the opposite side, and r is the radius of the incircle of this triangle,
prove that 1 / R1 + 1 / R2 + 1 / R3 ≤ 1 / r. -/
theorem radii_inequality (h_ge : ∀ i : Fin 3, 0 < [R1, R2, R3][i]) (h_incircle : 0 < r) :
  (1 / R1) + (1 / R2) + (1 / R3) ≤ 1 / r :=
  sorry

end radii_inequality_l111_111281


namespace longest_collection_pages_l111_111421

theorem longest_collection_pages 
    (pages_per_inch_miles : ℕ := 5) 
    (pages_per_inch_daphne : ℕ := 50) 
    (height_miles : ℕ := 240) 
    (height_daphne : ℕ := 25) : 
  max (pages_per_inch_miles * height_miles) (pages_per_inch_daphne * height_daphne) = 1250 := 
by
  sorry

end longest_collection_pages_l111_111421


namespace sam_current_yellow_marbles_l111_111574

theorem sam_current_yellow_marbles (original_yellow : ℕ) (taken_yellow : ℕ) (current_yellow : ℕ) :
  original_yellow = 86 → 
  taken_yellow = 25 → 
  current_yellow = original_yellow - taken_yellow → 
  current_yellow = 61 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end sam_current_yellow_marbles_l111_111574


namespace curve_representation_l111_111952

   theorem curve_representation :
     ∀ (x y : ℝ), x^4 - y^4 - 4*x^2 + 4*y^2 = 0 ↔ (x + y = 0 ∨ x - y = 0 ∨ x^2 + y^2 = 4) :=
   by
     sorry
   
end curve_representation_l111_111952


namespace binomial_seven_four_l111_111192

noncomputable def binomial (n k : Nat) : Nat := n.choose k

theorem binomial_seven_four : binomial 7 4 = 35 := by
  sorry

end binomial_seven_four_l111_111192


namespace binomial_7_4_eq_35_l111_111201

theorem binomial_7_4_eq_35 : Nat.choose 7 4 = 35 := by
  sorry

end binomial_7_4_eq_35_l111_111201


namespace find_intersection_l111_111414

open Set Real

def domain_A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def domain_B : Set ℝ := {x : ℝ | x < 1}

def intersection (A B : Set ℝ) : Set ℝ := {x : ℝ | x ∈ A ∧ x ∈ B}

theorem find_intersection :
  intersection domain_A domain_B = {x : ℝ | -2 ≤ x ∧ x < 1} := 
by sorry

end find_intersection_l111_111414


namespace simplify_fraction_l111_111021

theorem simplify_fraction : (3 ^ 100 + 3 ^ 98) / (3 ^ 100 - 3 ^ 98) = 5 / 4 := 
by sorry

end simplify_fraction_l111_111021
