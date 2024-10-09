import Mathlib

namespace terminating_decimal_l725_72577

theorem terminating_decimal : (45 / (2^2 * 5^3) : ℚ) = 0.090 :=
by
  sorry

end terminating_decimal_l725_72577


namespace selection_at_most_one_l725_72533

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_at_most_one (A B : ℕ) :
  (combination 5 3) - (combination 3 1) = 7 :=
by
  sorry

end selection_at_most_one_l725_72533


namespace probability_two_red_balls_l725_72579

def total_balls : ℕ := 15
def red_balls_initial : ℕ := 7
def blue_balls_initial : ℕ := 8
def red_balls_after_first_draw : ℕ := 6
def remaining_balls_after_first_draw : ℕ := 14

theorem probability_two_red_balls :
  (red_balls_initial / total_balls) *
  (red_balls_after_first_draw / remaining_balls_after_first_draw) = 1 / 5 :=
by sorry

end probability_two_red_balls_l725_72579


namespace train_length_is_correct_l725_72520

-- Definitions
def speed_kmh := 48.0 -- in km/hr
def time_sec := 9.0 -- in seconds

-- Conversion function
def convert_speed (s_kmh : Float) : Float :=
  s_kmh * 1000 / 3600

-- Function to calculate length of train
def length_of_train (speed_kmh : Float) (time_sec : Float) : Float :=
  let speed_ms := convert_speed speed_kmh
  speed_ms * time_sec

-- Proof problem: Given the speed of the train and the time it takes to cross a pole, prove the length of the train
theorem train_length_is_correct : length_of_train speed_kmh time_sec = 119.97 :=
by
  sorry

end train_length_is_correct_l725_72520


namespace symmetric_line_eq_x_axis_l725_72571

theorem symmetric_line_eq_x_axis (x y : ℝ) :
  (3 * x - 4 * y + 5 = 0) → (3 * x + 4 * y + 5 = 0) :=
sorry

end symmetric_line_eq_x_axis_l725_72571


namespace calculate_volume_and_diagonal_calculate_volume_and_surface_rotation_calculate_radius_given_volume_l725_72511

noncomputable def volume_of_parallelepiped (R : ℝ) : ℝ := R^3 * Real.sqrt 6

noncomputable def diagonal_A_C_prime (R: ℝ) : ℝ := R * Real.sqrt 6

noncomputable def volume_of_rotation (R: ℝ) : ℝ := R^3 * Real.sqrt 12

theorem calculate_volume_and_diagonal (R : ℝ) : 
  volume_of_parallelepiped R = R^3 * Real.sqrt 6 ∧ 
  diagonal_A_C_prime R = R * Real.sqrt 6 :=
by sorry

theorem calculate_volume_and_surface_rotation (R : ℝ) :
  volume_of_rotation R = R^3 * Real.sqrt 12 :=
by sorry

theorem calculate_radius_given_volume (V : ℝ) (h : V = 0.034786) : 
  ∃ R : ℝ, V = volume_of_parallelepiped R :=
by sorry

end calculate_volume_and_diagonal_calculate_volume_and_surface_rotation_calculate_radius_given_volume_l725_72511


namespace used_car_percentage_l725_72573

-- Define the variables and conditions
variables (used_car_price original_car_price : ℕ) (h_used_car_price : used_car_price = 15000) (h_original_price : original_car_price = 37500)

-- Define the statement to prove the percentage
theorem used_car_percentage (h : used_car_price / original_car_price * 100 = 40) : true :=
sorry

end used_car_percentage_l725_72573


namespace given_roots_find_coefficients_l725_72563

theorem given_roots_find_coefficients {a b c : ℝ} :
  (1:ℝ)^5 + 2*(1)^4 + a * (1:ℝ)^2 + b * (1:ℝ) = c →
  (-1:ℝ)^5 + 2*(-1:ℝ)^4 + a * (-1:ℝ)^2 + b * (-1:ℝ) = c →
  a = -6 ∧ b = -1 ∧ c = -4 :=
by
  intros h1 h2
  sorry

end given_roots_find_coefficients_l725_72563


namespace tan_alpha_eq_7_over_5_l725_72530

theorem tan_alpha_eq_7_over_5
  (α : ℝ)
  (h : Real.tan (α - π / 4) = 1 / 6) :
  Real.tan α = 7 / 5 :=
by
  sorry

end tan_alpha_eq_7_over_5_l725_72530


namespace sum_of_interior_angles_of_polygon_with_14_diagonals_is_900_degrees_l725_72584

theorem sum_of_interior_angles_of_polygon_with_14_diagonals_is_900_degrees :
  ∃ (n : ℕ), (n * (n - 3) / 2 = 14) → ((n - 2) * 180 = 900) :=
by
  sorry

end sum_of_interior_angles_of_polygon_with_14_diagonals_is_900_degrees_l725_72584


namespace shortest_fence_length_l725_72588

open Real

noncomputable def area_of_garden (length width : ℝ) : ℝ := length * width

theorem shortest_fence_length (length width : ℝ) (h : area_of_garden length width = 64) :
  4 * sqrt 64 = 32 :=
by
  -- The statement sets up the condition that the area is 64 and asks to prove minimum perimeter (fence length = perimeter).
  sorry

end shortest_fence_length_l725_72588


namespace find_a_l725_72566

theorem find_a (x a : ℝ) (h₁ : x^2 + x - 6 = 0) :
  (ax + 1 = 0 → (a = -1/2 ∨ a = -1/3) ∧ ax + 1 ≠ 0 ↔ false) := 
by
  sorry

end find_a_l725_72566


namespace sam_dimes_l725_72569

theorem sam_dimes (dimes_original dimes_given : ℕ) :
  dimes_original = 9 → dimes_given = 7 → dimes_original + dimes_given = 16 :=
by
  intros h1 h2
  sorry

end sam_dimes_l725_72569


namespace original_rectangle_area_l725_72516

-- Define the original rectangle sides, square side, and perimeters of rectangles adjacent to the square
variables {a b x : ℝ}
variable (h1 : a + x = 10)
variable (h2 : b + x = 8)

-- Define the area calculation
def area (a b : ℝ) := a * b

-- The area of the original rectangle should be 80 cm²
theorem original_rectangle_area : area (10 - x) (8 - x) = 80 := by
  sorry

end original_rectangle_area_l725_72516


namespace union_sets_l725_72581

def set_A : Set ℝ := {x | x^3 - 3 * x^2 - x + 3 < 0}
def set_B : Set ℝ := {x | |x + 1 / 2| ≥ 1}

theorem union_sets :
  set_A ∪ set_B = ( {x : ℝ | x < -1} ∪ {x : ℝ | x ≥ 1 / 2} ) :=
by
  sorry

end union_sets_l725_72581


namespace range_of_a_l725_72512

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 - a*x + 2*a > 0) : 0 < a ∧ a < 8 :=
by
  sorry

end range_of_a_l725_72512


namespace denis_neighbors_l725_72527

theorem denis_neighbors :
  ∃ (positions : ℕ → String), 
  (positions 1 = "Borya") ∧ 
  (positions 2 ≠ "Gena") ∧ (positions 2 = "Vera" → positions 3 = "Anya" ∨ positions 3 = "Gena") ∧ 
  (positions 3 ≠ "Borya") ∧ (positions 3 ≠ "Gena") ∧ 
  (positions 5 ≠ "Borya") ∧ (positions 5 ≠ "Anya") → 
  (positions 4 = "Denis" → 
    (positions 3 = "Anya" ∨ positions 5 = "Gena") ∧ 
    (positions 3 ≠ "Gena" ∨ positions 5 = "Anya")) :=
by
  sorry

end denis_neighbors_l725_72527


namespace problem1_l725_72578

theorem problem1 (a : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) → (-2 * Real.sqrt 2 ≤ a ∧ a ≤ 2 * Real.sqrt 2) := by 
  sorry

end problem1_l725_72578


namespace mixture_weight_l725_72586

theorem mixture_weight (a b : ℝ) (h1 : a = 26.1) (h2 : a / (a + b) = 9 / 20) : a + b = 58 :=
sorry

end mixture_weight_l725_72586


namespace find_a_b_solution_set_l725_72518

-- Given function
def f (x : ℝ) (a b : ℝ) := x^2 - (a + b) * x + 3 * a

-- Part 1: Prove the values of a and b given the solution set of the inequality
theorem find_a_b (a b : ℝ) 
  (h1 : 1^2 - (a + b) * 1 + 3 * 1 = 0)
  (h2 : 3^2 - (a + b) * 3 + 3 * 1 = 0) :
  a = 1 ∧ b = 3 :=
sorry

-- Part 2: Find the solution set of the inequality f(x) > 0 given b = 3
theorem solution_set (a : ℝ)
  (h : b = 3) :
  (a > 3 → (∀ x, f x a 3 > 0 ↔ x < 3 ∨ x > a)) ∧
  (a < 3 → (∀ x, f x a 3 > 0 ↔ x < a ∨ x > 3)) ∧
  (a = 3 → (∀ x, f x a 3 > 0 ↔ x ≠ 3)) :=
sorry

end find_a_b_solution_set_l725_72518


namespace arithmetic_sequence_common_difference_l725_72599

theorem arithmetic_sequence_common_difference (a : Nat → Int)
  (h1 : a 1 = 2) 
  (h3 : a 3 = 8)
  (h_arith : ∀ n, a (n + 1) = a n + (a 2 - a 1))  -- General form for an arithmetic sequence given two terms
  : a 2 - a 1 = 3 :=
by
  -- The main steps of the proof will follow from the arithmetic progression properties
  sorry

end arithmetic_sequence_common_difference_l725_72599


namespace handshake_count_l725_72501

-- Defining the conditions
def number_of_companies : ℕ := 5
def representatives_per_company : ℕ := 5
def total_participants : ℕ := number_of_companies * representatives_per_company

-- Defining the number of handshakes each person makes
def handshakes_per_person : ℕ := total_participants - 1 - (representatives_per_company - 1)

-- Defining the total number of handshakes
def total_handshakes : ℕ := (total_participants * handshakes_per_person) / 2

theorem handshake_count :
  total_handshakes = 250 :=
by
  sorry

end handshake_count_l725_72501


namespace inequality_necessary_not_sufficient_l725_72515

theorem inequality_necessary_not_sufficient (m : ℝ) : 
  (-3 < m ∧ m < 5) → (5 - m > 0 ∧ m + 3 > 0 ∧ 5 - m ≠ m + 3) :=
by
  intro h
  sorry

end inequality_necessary_not_sufficient_l725_72515


namespace binary_multiplication_l725_72574

theorem binary_multiplication :
  let a := 0b1101101
  let b := 0b1011
  let product := 0b10001001111
  a * b = product :=
sorry

end binary_multiplication_l725_72574


namespace sum_of_largest_100_l725_72548

theorem sum_of_largest_100 (a : Fin 123 → ℝ) (h1 : (Finset.univ.sum a) = 3813) 
  (h2 : ∀ i j : Fin 123, i ≤ j → a i ≤ a j) : 
  ∃ s : Finset (Fin 123), s.card = 100 ∧ (s.sum a) ≥ 3100 :=
by
  sorry

end sum_of_largest_100_l725_72548


namespace triangle_angle_ratio_l725_72562

theorem triangle_angle_ratio (A B C D : Type*) 
  (α β γ δ : ℝ) -- α = ∠BAC, β = ∠ABC, γ = ∠BCA, δ = external angles
  (h1 : α + β + γ = 180)
  (h2 : δ = α + γ)
  (h3 : δ = β + γ) : (2 * 180 - (α + β)) / (α + β) = 2 :=
by
  sorry

end triangle_angle_ratio_l725_72562


namespace average_age_of_choir_l725_72523

theorem average_age_of_choir 
  (num_females : ℕ) (avg_age_females : ℝ)
  (num_males : ℕ) (avg_age_males : ℝ)
  (total_people : ℕ) (total_people_eq : total_people = num_females + num_males) :
  num_females = 12 → avg_age_females = 28 → num_males = 18 → avg_age_males = 38 → total_people = 30 →
  (num_females * avg_age_females + num_males * avg_age_males) / total_people = 34 := by
  intros
  sorry

end average_age_of_choir_l725_72523


namespace negation_of_existential_proposition_l725_72559

theorem negation_of_existential_proposition :
  (¬ (∃ x : ℝ, x > Real.sin x)) ↔ (∀ x : ℝ, x ≤ Real.sin x) :=
by 
  sorry

end negation_of_existential_proposition_l725_72559


namespace prove_m_add_n_l725_72590

-- Definitions from conditions
variables (m n : ℕ)

def condition1 : Prop := m + 1 = 3
def condition2 : Prop := m = n - 1

-- Statement to prove
theorem prove_m_add_n (h1 : condition1 m) (h2 : condition2 m n) : m + n = 5 := 
sorry

end prove_m_add_n_l725_72590


namespace sticks_difference_l725_72508

-- Definitions of the conditions
def d := 14  -- number of sticks Dave picked up
def a := 9   -- number of sticks Amy picked up
def total := 50  -- initial total number of sticks in the yard

-- The proof problem statement
theorem sticks_difference : (d + a) - (total - (d + a)) = 4 :=
by
  sorry

end sticks_difference_l725_72508


namespace algebraic_expression_l725_72546

-- Given conditions in the problem.
variables (x y : ℝ)

-- The statement to be proved: If 2x - 3y = 1, then 6y - 4x + 8 = 6.
theorem algebraic_expression (h : 2 * x - 3 * y = 1) : 6 * y - 4 * x + 8 = 6 :=
by 
  sorry

end algebraic_expression_l725_72546


namespace polynomial_coeff_properties_l725_72502

theorem polynomial_coeff_properties :
  (∃ a0 a1 a2 a3 a4 a5 a6 a7 : ℤ,
  (∀ x : ℤ, (1 - 2 * x)^7 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7) ∧
  a0 = 1 ∧
  (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 = -1) ∧
  (|a0| + |a1| + |a2| + |a3| + |a4| + |a5| + |a6| + |a7| = 3^7)) :=
sorry

end polynomial_coeff_properties_l725_72502


namespace parallel_lines_condition_l725_72542

theorem parallel_lines_condition (k1 k2 b : ℝ) (l1 l2 : ℝ → ℝ) (H1 : ∀ x, l1 x = k1 * x + 1)
  (H2 : ∀ x, l2 x = k2 * x + b) : (∀ x, l1 x = l2 x ↔ k1 = k2 ∧ b = 1) → (k1 = k2) ↔ (∀ x, l1 x ≠ l2 x ∧ l1 x - l2 x = 1 - b) := 
by
  sorry

end parallel_lines_condition_l725_72542


namespace find_b_l725_72538

theorem find_b (x : ℝ) (b : ℝ) :
  (3 * x + 9 = 0) → (2 * b * x - 15 = -5) → b = -5 / 3 :=
by
  intros h1 h2
  sorry

end find_b_l725_72538


namespace grandma_mushrooms_l725_72544

theorem grandma_mushrooms (M : ℕ) (h₁ : ∀ t : ℕ, t = 2 * M)
                         (h₂ : ∀ p : ℕ, p = 4 * t)
                         (h₃ : ∀ b : ℕ, b = 4 * p)
                         (h₄ : ∀ r : ℕ, r = b / 3)
                         (h₅ : r = 32) :
  M = 3 :=
by
  -- We are expected to fill the steps here to provide the proof if required
  sorry

end grandma_mushrooms_l725_72544


namespace negation_of_proposition_l725_72504

theorem negation_of_proposition : 
  ¬(∀ x : ℝ, x^2 + x + 1 > 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
by
  sorry

end negation_of_proposition_l725_72504


namespace number_of_employees_l725_72539

-- Definitions
def emily_original_salary : ℕ := 1000000
def emily_new_salary : ℕ := 850000
def employee_original_salary : ℕ := 20000
def employee_new_salary : ℕ := 35000
def salary_difference : ℕ := emily_original_salary - emily_new_salary
def salary_increase_per_employee : ℕ := employee_new_salary - employee_original_salary

-- Theorem: Prove Emily has n employees where n = 10
theorem number_of_employees : salary_difference / salary_increase_per_employee = 10 :=
by sorry

end number_of_employees_l725_72539


namespace lcm_of_48_and_14_is_56_l725_72521

theorem lcm_of_48_and_14_is_56 :
  ∀ n : ℕ, (n = 48 ∧ Nat.gcd n 14 = 12) → Nat.lcm n 14 = 56 :=
by
  intro n h
  sorry

end lcm_of_48_and_14_is_56_l725_72521


namespace john_eggs_per_week_l725_72560

theorem john_eggs_per_week
  (pens : ℕ)
  (emus_per_pen : ℕ)
  (female_ratio : ℚ)
  (eggs_per_female_per_day : ℕ)
  (days_in_week : ℕ) :
  pens = 4 →
  emus_per_pen = 6 →
  female_ratio = 1/2 →
  eggs_per_female_per_day = 1 →
  days_in_week = 7 →
  (pens * emus_per_pen * female_ratio * eggs_per_female_per_day * days_in_week = 84) :=
by
  intros h_pens h_emus h_ratio h_eggs h_days
  rw [h_pens, h_emus, h_ratio, h_eggs, h_days]
  norm_num

end john_eggs_per_week_l725_72560


namespace steve_final_height_l725_72500

-- Define the initial height of Steve in inches.
def initial_height : ℕ := 5 * 12 + 6

-- Define how many inches Steve grew.
def growth : ℕ := 6

-- Define Steve's final height after growing.
def final_height : ℕ := initial_height + growth

-- The final height should be 72 inches.
theorem steve_final_height : final_height = 72 := by
  -- we don't provide the proof here
  sorry

end steve_final_height_l725_72500


namespace average_weight_of_whole_class_l725_72556

/-- Section A has 30 students -/
def num_students_A : ℕ := 30

/-- Section B has 20 students -/
def num_students_B : ℕ := 20

/-- The average weight of Section A is 40 kg -/
def avg_weight_A : ℕ := 40

/-- The average weight of Section B is 35 kg -/
def avg_weight_B : ℕ := 35

/-- The average weight of the whole class is 38 kg -/
def avg_weight_whole_class : ℕ := 38

-- Proof that the average weight of the whole class is equal to 38 kg

theorem average_weight_of_whole_class : 
  ((num_students_A * avg_weight_A) + (num_students_B * avg_weight_B)) / (num_students_A + num_students_B) = avg_weight_whole_class :=
by
  -- Sorry indicates that the proof is omitted.
  sorry

end average_weight_of_whole_class_l725_72556


namespace modulo_remainder_l725_72509

theorem modulo_remainder : (7^2023) % 17 = 15 := 
by 
  sorry

end modulo_remainder_l725_72509


namespace polynomial_sat_condition_l725_72595

theorem polynomial_sat_condition (P : Polynomial ℝ) (k : ℕ) (hk : 0 < k) :
  (P.comp P = P ^ k) →
  (P = 0 ∨ P = 1 ∨ (k % 2 = 1 ∧ P = -1) ∨ P = Polynomial.X ^ k) :=
sorry

end polynomial_sat_condition_l725_72595


namespace slower_train_speed_l725_72537

theorem slower_train_speed (faster_speed : ℝ) (time_passed : ℝ) (train_length : ℝ) (slower_speed: ℝ) :
  faster_speed = 50 ∧ time_passed = 15 ∧ train_length = 75 →
  slower_speed = 32 :=
by
  intro h
  sorry

end slower_train_speed_l725_72537


namespace inequality_solution_l725_72576

theorem inequality_solution :
  {x : ℝ | |2 * x - 3| + |x + 1| < 7 ∧ x ≤ 4} = {x : ℝ | -5 / 3 < x ∧ x < 3} :=
by
  sorry

end inequality_solution_l725_72576


namespace infinitely_many_n_divide_b_pow_n_plus_1_l725_72524

theorem infinitely_many_n_divide_b_pow_n_plus_1 (b : ℕ) (h1 : b > 2) :
  (∃ᶠ n in at_top, n^2 ∣ b^n + 1) ↔ ¬ ∃ k : ℕ, b + 1 = 2^k :=
sorry

end infinitely_many_n_divide_b_pow_n_plus_1_l725_72524


namespace part2_inequality_l725_72551

-- Define the function f and its conditions
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- The main theorem we want to prove
theorem part2_inequality (a b c : ℝ) (h : a^2 + 2 * b^2 + 3 * c^2 = 6) : 
  |a + 2 * b + 3 * c| ≤ 6 :=
by {
-- Proof goes here
sorry
}

end part2_inequality_l725_72551


namespace fraction_is_perfect_square_l725_72522

theorem fraction_is_perfect_square (a b : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hdiv : (ab + 1) ∣ (a^2 + b^2)) : 
  ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end fraction_is_perfect_square_l725_72522


namespace arithmetic_sequence_length_l725_72555

theorem arithmetic_sequence_length :
  ∀ (a d a_n : ℕ), a = 6 → d = 4 → a_n = 154 → ∃ n: ℕ, a_n = a + (n-1) * d ∧ n = 38 :=
by
  intro a d a_n ha hd ha_n
  use 38
  rw [ha, hd, ha_n]
  -- Leaving the proof as an exercise
  sorry

end arithmetic_sequence_length_l725_72555


namespace flatville_additional_plates_max_count_l725_72593

noncomputable def flatville_initial_plate_count : Nat :=
  6 * 4 * 5

noncomputable def flatville_max_plate_count : Nat :=
  6 * 6 * 6

theorem flatville_additional_plates_max_count : flatville_max_plate_count - flatville_initial_plate_count = 96 :=
by
  sorry

end flatville_additional_plates_max_count_l725_72593


namespace geometric_sequence_common_ratio_l725_72567

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h : ∀ n, a (n + 1) = a n * q) 
  (h_inc : ∀ n, a (n + 1) > a n) (h2 : a 2 = 2) (h3 : a 4 - a 3 = 4) : q = 2 :=
by
  sorry

end geometric_sequence_common_ratio_l725_72567


namespace determine_gx_l725_72550

/-
  Given two polynomials f(x) and h(x), we need to show that g(x) is a certain polynomial
  when f(x) + g(x) = h(x).
-/

def f (x : ℝ) : ℝ := 4 * x^5 + 3 * x^3 + x - 2
def h (x : ℝ) : ℝ := 7 * x^3 - 5 * x + 4
def g (x : ℝ) : ℝ := -4 * x^5 + 4 * x^3 - 4 * x + 6

theorem determine_gx (x : ℝ) : f x + g x = h x :=
by
  -- proof will go here
  sorry

end determine_gx_l725_72550


namespace value_of_2x_plus_3y_l725_72505

theorem value_of_2x_plus_3y {x y : ℝ} (h1 : 2 * x - 1 = 5) (h2 : 3 * y + 2 = 17) : 2 * x + 3 * y = 21 :=
by
  sorry

end value_of_2x_plus_3y_l725_72505


namespace nitin_rank_last_l725_72589

theorem nitin_rank_last (total_students : ℕ) (rank_start : ℕ) (rank_last : ℕ) 
  (h1 : total_students = 58) 
  (h2 : rank_start = 24) 
  (h3 : rank_last = total_students - rank_start + 1) : 
  rank_last = 35 := 
by 
  -- proof can be filled in here
  sorry

end nitin_rank_last_l725_72589


namespace dot_product_AB_BC_l725_72529

theorem dot_product_AB_BC (AB BC : ℝ) (B : ℝ) 
  (h1 : AB = 3) (h2 : BC = 4) (h3 : B = π/6) :
  (AB * BC * Real.cos (π - B) = -6 * Real.sqrt 3) :=
by
  rw [h1, h2, h3]
  sorry

end dot_product_AB_BC_l725_72529


namespace polynomial_remainder_l725_72582

theorem polynomial_remainder (a : ℝ) (h : ∀ x : ℝ, x^3 + a * x^2 + 1 = (x^2 - 1) * (x + 2) + (x + 3)) : a = 2 :=
sorry

end polynomial_remainder_l725_72582


namespace faye_coloring_books_l725_72517

theorem faye_coloring_books (x : ℕ) : 34 - x + 48 = 79 → x = 3 :=
by
  sorry

end faye_coloring_books_l725_72517


namespace karsyn_total_payment_l725_72597

-- Define the initial price of the phone
def initial_price : ℝ := 600

-- Define the discounted rate for the phone
def discount_rate_phone : ℝ := 0.20

-- Define the prices for additional items
def phone_case_price : ℝ := 25
def screen_protector_price : ℝ := 15

-- Define the discount rates
def discount_rate_125 : ℝ := 0.05
def discount_rate_150 : ℝ := 0.10
def final_discount_rate : ℝ := 0.03

-- Define the tax rate and fee
def exchange_rate_fee : ℝ := 0.02

noncomputable def total_payment (initial_price : ℝ) (discount_rate_phone : ℝ) 
  (phone_case_price : ℝ) (screen_protector_price : ℝ) (discount_rate_125 : ℝ) 
  (discount_rate_150 : ℝ) (final_discount_rate : ℝ) (exchange_rate_fee : ℝ) : ℝ :=
  let discounted_phone_price := initial_price * discount_rate_phone
  let additional_items_price := phone_case_price + screen_protector_price
  let total_before_discounts := discounted_phone_price + additional_items_price
  let total_after_first_discount := total_before_discounts * (1 - discount_rate_125)
  let total_after_second_discount := total_after_first_discount * (1 - discount_rate_150)
  let total_after_all_discounts := total_after_second_discount * (1 - final_discount_rate)
  let total_with_exchange_fee := total_after_all_discounts * (1 + exchange_rate_fee)
  total_with_exchange_fee

theorem karsyn_total_payment :
  total_payment initial_price discount_rate_phone phone_case_price screen_protector_price 
    discount_rate_125 discount_rate_150 final_discount_rate exchange_rate_fee = 135.35 := 
  by 
  -- Specify proof steps here
  sorry

end karsyn_total_payment_l725_72597


namespace problem1_problem2_l725_72545

-- The first problem
theorem problem1 (x : ℝ) (h : Real.tan x = 3) :
  (2 * Real.sin (Real.pi - x) + 3 * Real.cos (-x)) /
  (Real.sin (x + Real.pi / 2) - Real.sin (x + Real.pi)) = 9 / 4 :=
by
  sorry

-- The second problem
theorem problem2 (x : ℝ) (h : Real.tan x = 3) :
  2 * Real.sin x ^ 2 - Real.sin (2 * x) + Real.cos x ^ 2 = 13 / 10 :=
by
  sorry

end problem1_problem2_l725_72545


namespace number_of_outfits_l725_72553

-- Definitions based on conditions
def trousers : ℕ := 4
def shirts : ℕ := 8
def jackets : ℕ := 3
def belts : ℕ := 2

-- The statement to prove
theorem number_of_outfits : trousers * shirts * jackets * belts = 192 := by
  sorry

end number_of_outfits_l725_72553


namespace annual_percentage_increase_l725_72561

theorem annual_percentage_increase (present_value future_value : ℝ) (years: ℝ) (r : ℝ) 
  (h1 : present_value = 20000)
  (h2 : future_value = 24200)
  (h3 : years = 2) : 
  future_value = present_value * (1 + r)^years → r = 0.1 :=
sorry

end annual_percentage_increase_l725_72561


namespace value_of_p_l725_72596

theorem value_of_p (p q r : ℕ) (h1 : p + q + r = 70) (h2 : p = 2*q) (h3 : q = 3*r) : p = 42 := 
by 
  sorry

end value_of_p_l725_72596


namespace no_positive_integer_solution_l725_72564

theorem no_positive_integer_solution (m n : ℕ) (h : 0 < m) (h1 : 0 < n) : ¬ (5 * m^2 - 6 * m * n + 7 * n^2 = 2006) :=
sorry

end no_positive_integer_solution_l725_72564


namespace increasing_condition_sufficient_not_necessary_l725_72519

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x

theorem increasing_condition_sufficient_not_necessary (a : ℝ) :
  (∀ x : ℝ, x > 0 → (3 * x^2 + a) ≥ 0) → (a ≥ 0) ∧ ¬ (a > 0 ↔ (∀ x : ℝ, x > 0 → (3 * x^2 + a) ≥ 0)) :=
by
  sorry

end increasing_condition_sufficient_not_necessary_l725_72519


namespace problem_solution_l725_72585

noncomputable def a : ℝ := Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 10
noncomputable def b : ℝ := -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 10
noncomputable def c : ℝ := Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 10
noncomputable def d : ℝ := -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 10

theorem problem_solution : ((1 / a) + (1 / b) + (1 / c) + (1 / d))^2 = 0 :=
by
  sorry

end problem_solution_l725_72585


namespace tv_show_duration_l725_72531

theorem tv_show_duration (total_air_time : ℝ) (num_commercials : ℕ) (commercial_duration_min : ℝ) :
  total_air_time = 1.5 ∧ num_commercials = 3 ∧ commercial_duration_min = 10 →
  (total_air_time - (num_commercials * commercial_duration_min / 60)) = 1 :=
by
  sorry

end tv_show_duration_l725_72531


namespace find_hourly_charge_computer_B_l725_72594

noncomputable def hourly_charge_computer_B (B : ℝ) :=
  ∃ (A h : ℝ),
    A = 1.4 * B ∧
    B * (h + 20) = 550 ∧
    A * h = 550 ∧
    B = 7.86

theorem find_hourly_charge_computer_B : ∃ B : ℝ, hourly_charge_computer_B B :=
  sorry

end find_hourly_charge_computer_B_l725_72594


namespace multiply_preserve_equiv_l725_72568

noncomputable def conditions_equiv_eqn (N D F : Polynomial ℝ) : Prop :=
  (D = F * (D / F)) ∧ (N.degree ≥ F.degree) ∧ (D ≠ 0)

theorem multiply_preserve_equiv (N D F : Polynomial ℝ) :
  conditions_equiv_eqn N D F →
  (N / D = 0 ↔ (N * F) / (D * F) = 0) :=
by
  sorry

end multiply_preserve_equiv_l725_72568


namespace book_arrangement_count_l725_72575

theorem book_arrangement_count :
  let total_books := 7
  let identical_math_books := 3
  let identical_physics_books := 2
  (Nat.factorial 7) / ((Nat.factorial 3) * (Nat.factorial 2)) = 420 := 
by
  sorry

end book_arrangement_count_l725_72575


namespace inequality_proof_l725_72507

variables {x y z : ℝ}

theorem inequality_proof 
  (h1 : y ≥ 2 * z) 
  (h2 : 2 * z ≥ 4 * x) 
  (h3 : 2 * (x^3 + y^3 + z^3) + 15 * (x * y^2 + y * z^2 + z * x^2) ≥ 16 * (x^2 * y + y^2 * z + z^2 * x) + 2 * x * y * z) : 
  4 * x + y ≥ 4 * z :=
sorry

end inequality_proof_l725_72507


namespace maximum_value_of_f_l725_72598

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.sqrt 3 * Real.cos x - 2 * Real.sin (3 * x))

theorem maximum_value_of_f :
  ∃ x : ℝ, f x = (16 * Real.sqrt 3) / 9 :=
sorry

end maximum_value_of_f_l725_72598


namespace no_value_of_b_l725_72592

theorem no_value_of_b (b : ℤ) : ¬ ∃ (n : ℤ), 2 * b^2 + 3 * b + 2 = n^2 := 
sorry

end no_value_of_b_l725_72592


namespace goats_at_farm_l725_72513

theorem goats_at_farm (G C D P : ℕ) 
  (h1: C = 2 * G)
  (h2: D = (G + C) / 2)
  (h3: P = D / 3)
  (h4: G = P + 33) :
  G = 66 :=
by
  sorry

end goats_at_farm_l725_72513


namespace four_digit_square_number_divisible_by_11_with_unit_1_l725_72525

theorem four_digit_square_number_divisible_by_11_with_unit_1 
  : ∃ y : ℕ, y >= 1000 ∧ y <= 9999 ∧ (∃ n : ℤ, y = n^2) ∧ y % 11 = 0 ∧ y % 10 = 1 ∧ y = 9801 := 
by {
  -- sorry statement to skip the proof.
  sorry 
}

end four_digit_square_number_divisible_by_11_with_unit_1_l725_72525


namespace simplify_expression_l725_72514

theorem simplify_expression : ( (144^2 - 12^2) / (120^2 - 18^2) * ((120 - 18) * (120 + 18)) / ((144 - 12) * (144 + 12)) ) = 1 :=
by
  sorry

end simplify_expression_l725_72514


namespace proof_problem_l725_72591

open Real

noncomputable def problem (c d : ℝ) : ℝ :=
  5^(c / d) + 2^(d / c)

theorem proof_problem :
  let c := log 8
  let d := log 25
  problem c d = 2 * sqrt 2 + 5^(2 / 3) :=
by
  intro c d
  have c_def : c = log 8 := rfl
  have d_def : d = log 25 := rfl
  rw [c_def, d_def]
  sorry

end proof_problem_l725_72591


namespace no_valid_n_for_conditions_l725_72534

theorem no_valid_n_for_conditions :
  ∀ (n : ℕ), (100 ≤ n / 5 ∧ n / 5 ≤ 999) ∧ (100 ≤ 5 * n ∧ 5 * n ≤ 999) → false :=
by
  sorry

end no_valid_n_for_conditions_l725_72534


namespace tailor_trimming_l725_72540

theorem tailor_trimming (x : ℝ) (A B : ℝ)
  (h1 : ∃ (L : ℝ), L = 22) -- Original length of a side of the cloth is 22 feet
  (h2 : 6 = 6) -- Feet trimmed from two opposite edges
  (h3 : ∃ (remaining_area : ℝ), remaining_area = 120) -- 120 square feet of cloth remain after trimming
  (h4 : A = 22 - 2 * 6) -- New length of the side after trimming 6 feet from opposite edges
  (h5 : B = 22 - x) -- New length of the side after trimming x feet from the other two edges
  (h6 : remaining_area = A * B) -- Relationship of the remaining area
: x = 10 :=
by
  sorry

end tailor_trimming_l725_72540


namespace simplify_expression_l725_72587

theorem simplify_expression (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a ≠ b) :
  (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - a^2) = a / b :=
by sorry

end simplify_expression_l725_72587


namespace find_k_for_xy_solution_l725_72572

theorem find_k_for_xy_solution :
  ∀ (k : ℕ), (∃ (x y : ℕ), x * (x + k) = y * (y + 1))
  → k = 1 ∨ k ≥ 4 :=
by
  intros k h
  sorry -- proof goes here

end find_k_for_xy_solution_l725_72572


namespace dessert_menu_count_is_192_l725_72526

-- Defining the set of desserts
inductive Dessert
| cake | pie | ice_cream

-- Function to count valid dessert menus (not repeating on consecutive days) with cake on Friday
def countDessertMenus : Nat :=
  -- Let's denote Sunday as day 1 and Saturday as day 7
  let sunday_choices := 3
  let weekday_choices := 2 -- for Monday to Thursday (no repeats consecutive)
  let weekend_choices := 2 -- for Saturday and Sunday after
  sunday_choices * weekday_choices^4 * 1 * weekend_choices^2

-- Theorem stating the number of valid dessert menus for the week
theorem dessert_menu_count_is_192 : countDessertMenus = 192 :=
  by
    -- Actual proof is omitted
    sorry

end dessert_menu_count_is_192_l725_72526


namespace production_value_equation_l725_72549

theorem production_value_equation (x : ℝ) :
  (2000000 * (1 + x)^2) - (2000000 * (1 + x)) = 220000 := 
sorry

end production_value_equation_l725_72549


namespace journey_speed_l725_72547

theorem journey_speed
  (v : ℝ) -- Speed during the first four hours
  (total_distance : ℝ) (total_time : ℝ) -- Total distance and time of the journey
  (distance_part1 : ℝ) (time_part1 : ℝ) -- Distance and time for the first part of journey
  (distance_part2 : ℝ) (time_part2 : ℝ) -- Distance and time for the second part of journey
  (speed_part2 : ℝ) : -- Speed during the second part of journey
  total_distance = 24 ∧ total_time = 8 ∧ speed_part2 = 2 ∧ 
  time_part1 = 4 ∧ time_part2 = 4 ∧ 
  distance_part1 = v * time_part1 ∧ distance_part2 = speed_part2 * time_part2 →
  v = 4 := 
by
  sorry

end journey_speed_l725_72547


namespace nina_jerome_age_ratio_l725_72565

variable (N J L : ℕ)

theorem nina_jerome_age_ratio (h1 : L = N - 4) (h2 : L + N + J = 36) (h3 : L = 6) : N / J = 1 / 2 := by
  sorry

end nina_jerome_age_ratio_l725_72565


namespace angela_deliveries_l725_72510

theorem angela_deliveries
  (n_meals : ℕ)
  (h_meals : n_meals = 3)
  (n_packages : ℕ)
  (h_packages : n_packages = 8 * n_meals) :
  n_meals + n_packages = 27 := by
  sorry

end angela_deliveries_l725_72510


namespace horner_method_value_at_neg1_l725_72552

theorem horner_method_value_at_neg1 : 
  let f (x : ℤ) := 4 * x ^ 4 + 3 * x ^ 3 - 6 * x ^ 2 + x - 1
  let x := -1
  let v0 := 4
  let v1 := v0 * x + 3
  let v2 := v1 * x - 6
  v2 = -5 := by
  sorry

end horner_method_value_at_neg1_l725_72552


namespace remainder_6_pow_23_mod_5_l725_72558

theorem remainder_6_pow_23_mod_5 : (6 ^ 23) % 5 = 1 := 
by {
  sorry
}

end remainder_6_pow_23_mod_5_l725_72558


namespace avg_weights_N_square_of_integer_l725_72580

theorem avg_weights_N_square_of_integer (N : ℕ) :
  (∃ S : ℕ, S > 0 ∧ ∃ k : ℕ, k * k = N + 1 ∧ S = (N * (N + 1)) / 2 / (N - k + 1) ∧ (N * (N + 1)) / 2 - S = (N - k) * S) ↔ (∃ k : ℕ, k * k = N + 1) := by
  sorry

end avg_weights_N_square_of_integer_l725_72580


namespace diff_one_tenth_and_one_tenth_percent_of_6000_l725_72541

def one_tenth_of_6000 := 6000 / 10
def one_tenth_percent_of_6000 := (1 / 1000) * 6000

theorem diff_one_tenth_and_one_tenth_percent_of_6000 : 
  (one_tenth_of_6000 - one_tenth_percent_of_6000) = 594 :=
by
  sorry

end diff_one_tenth_and_one_tenth_percent_of_6000_l725_72541


namespace exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l725_72583

-- Lean 4 statement for part (a)
theorem exists_palindromic_number_divisible_by_5 : 
  ∃ (n : ℕ), (n = 51715) ∧ (n % 5 = 0) := sorry

-- Lean 4 statement for part (b)
theorem count_palindromic_numbers_divisible_by_5 : 
  (∃ (count : ℕ), count = 100) := sorry

end exists_palindromic_number_divisible_by_5_count_palindromic_numbers_divisible_by_5_l725_72583


namespace find_function_l725_72535

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_function (h : ∀ x y : ℝ, (f x * f y - f (x * y)) / 2 = x + y + 1) : 
  ∀ x : ℝ, f x = x + 2 := sorry

end find_function_l725_72535


namespace alberto_bjorn_distance_difference_l725_72536

-- Definitions based on given conditions
def alberto_speed : ℕ := 12  -- miles per hour
def bjorn_speed : ℕ := 10    -- miles per hour
def total_time : ℕ := 6      -- hours
def bjorn_rest_time : ℕ := 1 -- hours

def alberto_distance : ℕ := alberto_speed * total_time
def bjorn_distance : ℕ := bjorn_speed * (total_time - bjorn_rest_time)

-- The statement to prove
theorem alberto_bjorn_distance_difference :
  (alberto_distance - bjorn_distance) = 22 :=
by
  sorry

end alberto_bjorn_distance_difference_l725_72536


namespace rotation_problem_l725_72557

theorem rotation_problem (y : ℝ) (hy : y < 360) :
  (450 % 360 == 90) ∧ (y == 360 - 90) ∧ (90 + (360 - y) % 360 == 0) → y == 270 :=
by {
  -- Proof steps go here
  sorry
}

end rotation_problem_l725_72557


namespace matrix_determinant_zero_l725_72528

noncomputable def matrix_example : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    ![Real.sin 1, Real.sin 2, Real.sin 3],
    ![Real.sin 4, Real.sin 5, Real.sin 6],
    ![Real.sin 7, Real.sin 8, Real.sin 9]
  ]

theorem matrix_determinant_zero : matrix_example.det = 0 := 
by 
  sorry

end matrix_determinant_zero_l725_72528


namespace race_result_l725_72570

-- Definitions based on conditions
variable (hare_won : Bool)
variable (fox_second : Bool)
variable (hare_second : Bool)
variable (moose_first : Bool)

-- Condition that each squirrel had one error.
axiom owl_statement : xor hare_won fox_second ∧ xor hare_second moose_first

-- The final proof problem
theorem race_result : moose_first = true ∧ fox_second = true :=
by {
  -- Proving based on the owl's statement that each squirrel had one error
  sorry
}

end race_result_l725_72570


namespace probability_of_continuous_stripe_loop_l725_72543

-- Definitions corresponding to identified conditions:
def cube_faces : ℕ := 6

def diagonal_orientations_per_face : ℕ := 2

def total_stripe_combinations (faces : ℕ) (orientations : ℕ) : ℕ :=
  orientations ^ faces

def satisfying_stripe_combinations : ℕ := 2

-- Proof statement:
theorem probability_of_continuous_stripe_loop :
  (satisfying_stripe_combinations : ℚ) / (total_stripe_combinations cube_faces diagonal_orientations_per_face : ℚ) = 1 / 32 :=
by
  -- Proof goes here
  sorry

end probability_of_continuous_stripe_loop_l725_72543


namespace intersection_A_B_l725_72532

def A := {x : ℝ | x < -1 ∨ x > 1}
def B := {x : ℝ | Real.log x / Real.log 2 > 0}

theorem intersection_A_B:
  A ∩ B = {x : ℝ | x > 1} :=
by
  sorry

end intersection_A_B_l725_72532


namespace remainder_3_pow_19_mod_10_l725_72506

theorem remainder_3_pow_19_mod_10 : (3 ^ 19) % 10 = 7 := by
  sorry

end remainder_3_pow_19_mod_10_l725_72506


namespace oranges_to_put_back_l725_72554

theorem oranges_to_put_back
  (p_A p_O : ℕ)
  (A O : ℕ)
  (total_fruits : ℕ)
  (initial_avg_price new_avg_price : ℕ)
  (x : ℕ)
  (h1 : p_A = 40)
  (h2 : p_O = 60)
  (h3 : total_fruits = 15)
  (h4 : initial_avg_price = 48)
  (h5 : new_avg_price = 45)
  (h6 : A + O = total_fruits)
  (h7 : (p_A * A + p_O * O) / total_fruits = initial_avg_price)
  (h8 : (720 - 60 * x) / (15 - x) = 45) :
  x = 3 :=
by
  sorry

end oranges_to_put_back_l725_72554


namespace inequality_proof_l725_72503

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l725_72503
