import Mathlib

namespace decimal_to_binary_25_l1589_158919

theorem decimal_to_binary_25 : (25 : Nat) = 1 * 2^4 + 1 * 2^3 + 0 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by
  sorry

end decimal_to_binary_25_l1589_158919


namespace dante_age_l1589_158957

def combined_age (D : ℕ) : ℕ := D + D / 2 + (D + 1)

theorem dante_age :
  ∃ D : ℕ, combined_age D = 31 ∧ D = 12 :=
by
  sorry

end dante_age_l1589_158957


namespace relationship_A_B_l1589_158924

variable (x y : ℝ)

noncomputable def A : ℝ := (x + y) / (1 + x + y)

noncomputable def B : ℝ := (x / (1 + x)) + (y / (1 + y))

theorem relationship_A_B (hx : 0 < x) (hy : 0 < y) : A x y < B x y := sorry

end relationship_A_B_l1589_158924


namespace distinct_configurations_l1589_158985

/-- 
Define m, n, and the binomial coefficient function.
conditions:
  - integer grid dimensions m and n with m >= 1, n >= 1.
  - initially (m-1)(n-1) coins in the subgrid of size (m-1) x (n-1).
  - legal move conditions for coins.
question:
  - Prove the number of distinct configurations of coins equals the binomial coefficient.
-/
def number_of_distinct_configurations (m n : ℕ) : ℕ :=
  Nat.choose (m + n - 2) (m - 1)

theorem distinct_configurations (m n : ℕ) (h_m : 1 ≤ m) (h_n : 1 ≤ n) :
  number_of_distinct_configurations m n = Nat.choose (m + n - 2) (m - 1) :=
sorry

end distinct_configurations_l1589_158985


namespace solve_for_x_l1589_158914

noncomputable def is_satisfied (x : ℝ) : Prop :=
  (Real.log x / Real.log 2) * (Real.log 7 / Real.log x) = Real.log 7 / Real.log 2

theorem solve_for_x :
  ∀ x : ℝ, 0 < x → x ≠ 1 ↔ is_satisfied x := by
  sorry

end solve_for_x_l1589_158914


namespace length_of_QR_of_triangle_l1589_158972

def length_of_QR (PQ PR PM : ℝ) : ℝ := sorry

theorem length_of_QR_of_triangle (PQ PR : ℝ) (PM : ℝ) (hPQ : PQ = 4) (hPR : PR = 7) (hPM : PM = 7 / 2) : length_of_QR PQ PR PM = 9 := by
  sorry

end length_of_QR_of_triangle_l1589_158972


namespace maximum_value_of_vectors_l1589_158954

open Real EuclideanGeometry

variables (a b c : EuclideanSpace ℝ (Fin 3))

def unit_vector (v : EuclideanSpace ℝ (Fin 3)) : Prop := ‖v‖ = 1

def given_conditions (a b c : EuclideanSpace ℝ (Fin 3)) : Prop :=
  unit_vector a ∧ unit_vector b ∧ ‖3 • a + 4 • b‖ = ‖4 • a - 3 • b‖ ∧ ‖c‖ = 2

theorem maximum_value_of_vectors
  (ha : unit_vector a)
  (hb : unit_vector b)
  (hab : ‖3 • a + 4 • b‖ = ‖4 • a - 3 • b‖)
  (hc : ‖c‖ = 2) :
  ‖a + b - c‖ ≤ sqrt 2 + 2 := 
by
  sorry

end maximum_value_of_vectors_l1589_158954


namespace max_value_of_expression_l1589_158905

theorem max_value_of_expression (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 3) :
  (∃ (x : ℝ), x = (ab/(a + b)) + (ac/(a + c)) + (bc/(b + c)) ∧ x = 9/4) :=
  sorry

end max_value_of_expression_l1589_158905


namespace find_n_l1589_158922

-- Define the conditions as hypothesis
variables (A B n : ℕ)

-- Hypothesis 1: This year, Ana's age is the square of Bonita's age.
-- A = B^2
#check (A = B^2) 

-- Hypothesis 2: Last year Ana was 5 times as old as Bonita.
-- A - 1 = 5 * (B - 1)
#check (A - 1 = 5 * (B - 1))

-- Hypothesis 3: Ana and Bonita were born n years apart.
-- A = B + n
#check (A = B + n)

-- Goal: The difference in their ages, n, should be 12.
theorem find_n (A B n : ℕ) (h1 : A = B^2) (h2 : A - 1 = 5 * (B - 1)) (h3 : A = B + n) : n = 12 :=
sorry

end find_n_l1589_158922


namespace tangent_line_at_P0_is_parallel_l1589_158971

noncomputable def curve (x : ℝ) : ℝ := x^3 + x - 2

def tangent_slope (x : ℝ) : ℝ := 3 * x^2 + 1

theorem tangent_line_at_P0_is_parallel (x y : ℝ) (h_curve : y = curve x) (h_slope : tangent_slope x = 4) :
  (x, y) = (-1, -4) :=
sorry

end tangent_line_at_P0_is_parallel_l1589_158971


namespace least_possible_number_l1589_158944

theorem least_possible_number :
  ∃ x : ℕ, (∃ q r : ℕ, x = 34 * q + r ∧ 0 ≤ r ∧ r < 34) ∧
            (∃ q' : ℕ, x = 5 * q' ∧ q' = r + 8) ∧
            x = 75 :=
by
  sorry

end least_possible_number_l1589_158944


namespace cupcakes_leftover_l1589_158915

-- Definitions based on the conditions
def total_cupcakes : ℕ := 17
def num_children : ℕ := 3

-- Theorem proving the correct answer
theorem cupcakes_leftover : total_cupcakes % num_children = 2 := by
  sorry

end cupcakes_leftover_l1589_158915


namespace area_ratio_trapezoid_l1589_158910

/--
In trapezoid PQRS, the lengths of the bases PQ and RS are 10 and 21 respectively.
The legs of the trapezoid are extended beyond P and Q to meet at point T.
Prove that the ratio of the area of triangle TPQ to the area of trapezoid PQRS is 100/341.
-/
theorem area_ratio_trapezoid (PQ RS TPQ PQRS : ℝ) (hPQ : PQ = 10) (hRS : RS = 21) :
  let area_TPQ := TPQ
  let area_PQRS := PQRS
  area_TPQ / area_PQRS = 100 / 341 :=
by
  sorry

end area_ratio_trapezoid_l1589_158910


namespace multiplication_problem_l1589_158990

-- Definitions for different digits A, B, C, D
def is_digit (n : ℕ) := n < 10

theorem multiplication_problem 
  (A B C D : ℕ) 
  (hA : is_digit A) 
  (hB : is_digit B) 
  (hC : is_digit C) 
  (hD : is_digit D) 
  (h_diff : ∀ x y : ℕ, x ≠ y → is_digit x → is_digit y → x ≠ A → y ≠ B → x ≠ C → y ≠ D)
  (hD1 : D = 1)
  (h_mult : A * D = A) 
  (hC_eq : C = A + B) :
  A + C = 5 := sorry

end multiplication_problem_l1589_158990


namespace mike_total_money_l1589_158936

theorem mike_total_money (num_bills : ℕ) (value_per_bill : ℕ) (h1 : num_bills = 9) (h2 : value_per_bill = 5) :
  (num_bills * value_per_bill) = 45 :=
by
  sorry

end mike_total_money_l1589_158936


namespace tangent_line_eq_monotonic_intervals_l1589_158933

noncomputable def f (x : ℝ) (a : ℝ) := x - a * Real.log x
noncomputable def f' (x : ℝ) (a : ℝ) := 1 - (a / x)

theorem tangent_line_eq (x y : ℝ) (h : x = 1 ∧ a = 2) :
  y = f 1 2 → (x - 1) + (y - 1) - 2 * ((x - 1) + (y - 1)) = 0 := by sorry

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → ∀ x > 0, f' x a > 0) ∧
  (a > 0 → ∀ x > 0, (x < a → f' x a < 0) ∧ (x > a → f' x a > 0)) := by sorry

end tangent_line_eq_monotonic_intervals_l1589_158933


namespace find_trousers_l1589_158902

variables (S T Ti : ℝ) -- Prices of shirt, trousers, and tie respectively
variables (x : ℝ)      -- The number of trousers in the first scenario

-- Conditions given in the problem
def condition1 : Prop := 6 * S + x * T + 2 * Ti = 80
def condition2 : Prop := 4 * S + 2 * T + 2 * Ti = 140
def condition3 : Prop := 5 * S + 3 * T + 2 * Ti = 110

-- Theorem to prove
theorem find_trousers : condition1 S T Ti x ∧ condition2 S T Ti ∧ condition3 S T Ti → x = 4 :=
by
  sorry

end find_trousers_l1589_158902


namespace relationship_between_a_b_l1589_158994

theorem relationship_between_a_b (a b c : ℝ) (x y : ℝ) (h1 : x = -3) (h2 : y = -2)
  (h3 : a * x + c * y = 1) (h4 : c * x - b * y = 2) : 9 * a + 4 * b = 1 :=
sorry

end relationship_between_a_b_l1589_158994


namespace cubic_transform_l1589_158986

theorem cubic_transform (A B C x z β : ℝ) (h₁ : z = x + β) (h₂ : 3 * β + A = 0) :
  z^3 + A * z^2 + B * z + C = 0 ↔ x^3 + (B - (A^2 / 3)) * x + (C - A * B / 3 + 2 * A^3 / 27) = 0 :=
sorry

end cubic_transform_l1589_158986


namespace multiply_abs_value_l1589_158974

theorem multiply_abs_value : -2 * |(-3 : ℤ)| = -6 := by
  sorry

end multiply_abs_value_l1589_158974


namespace find_y_positive_monotone_l1589_158961

noncomputable def y (y : ℝ) : Prop :=
  0 < y ∧ y * (⌊y⌋₊ : ℝ) = 132 ∧ y = 12

theorem find_y_positive_monotone : ∃ y : ℝ, 0 < y ∧ y * (⌊y⌋₊ : ℝ) = 132 := by
  sorry

end find_y_positive_monotone_l1589_158961


namespace square_area_is_256_l1589_158976

-- Definitions of the conditions
def rect_width : ℝ := 4
def rect_length : ℝ := 3 * rect_width
def side_of_square : ℝ := rect_length + rect_width

-- Proposition
theorem square_area_is_256 (rect_width : ℝ) (h1 : rect_width = 4) 
                           (rect_length : ℝ) (h2 : rect_length = 3 * rect_width) :
  side_of_square ^ 2 = 256 :=
by 
  sorry

end square_area_is_256_l1589_158976


namespace original_price_l1589_158928

theorem original_price (total_payment : ℝ) (num_units : ℕ) (discount_rate : ℝ) 
(h1 : total_payment = 500) (h2 : num_units = 18) (h3 : discount_rate = 0.20) : 
  (total_payment / (1 - discount_rate) * num_units) = 625.05 :=
by
  sorry

end original_price_l1589_158928


namespace unique_sum_of_squares_l1589_158966

theorem unique_sum_of_squares (p : ℕ) (k : ℕ) (x y a b : ℤ) 
  (hp : Prime p) (h1 : p = 4 * k + 1) (hx : x^2 + y^2 = p) (ha : a^2 + b^2 = p) :
  (x = a ∨ x = -a) ∧ (y = b ∨ y = -b) ∨ (x = b ∨ x = -b) ∧ (y = a ∨ y = -a) :=
sorry

end unique_sum_of_squares_l1589_158966


namespace number_of_ways_to_form_committee_with_president_l1589_158906

open Nat

def number_of_ways_to_choose_members (total_members : ℕ) (committee_size : ℕ) (president_required : Bool) : ℕ :=
  if president_required then choose (total_members - 1) (committee_size - 1) else choose total_members committee_size

theorem number_of_ways_to_form_committee_with_president :
  number_of_ways_to_choose_members 30 5 true = 23741 :=
by
  -- Given that total_members = 30, committee_size = 5, and president_required = true,
  -- we need to show that the number of ways to choose the remaining members is 23741.
  sorry

end number_of_ways_to_form_committee_with_president_l1589_158906


namespace tan_135_eq_neg_one_l1589_158940

theorem tan_135_eq_neg_one : Real.tan (135 * Real.pi / 180) = -1 := by
  sorry

end tan_135_eq_neg_one_l1589_158940


namespace sum_after_operations_l1589_158999

theorem sum_after_operations (a b S : ℝ) (h : a + b = S) : 
  3 * (a + 5) + 3 * (b + 5) = 3 * S + 30 := 
by 
  sorry

end sum_after_operations_l1589_158999


namespace students_per_class_l1589_158927

theorem students_per_class (total_cupcakes : ℕ) (num_classes : ℕ) (pe_students : ℕ) 
  (h1 : total_cupcakes = 140) (h2 : num_classes = 3) (h3 : pe_students = 50) : 
  (total_cupcakes - pe_students) / num_classes = 30 :=
by
  sorry

end students_per_class_l1589_158927


namespace total_video_hours_in_june_l1589_158912

-- Definitions for conditions
def upload_rate_first_half : ℕ := 10 -- one-hour videos per day
def upload_rate_second_half : ℕ := 20 -- doubled one-hour videos per day
def days_in_half_month : ℕ := 15
def total_days_in_june : ℕ := 30

-- Number of video hours uploaded in the first half of the month
def video_hours_first_half : ℕ := upload_rate_first_half * days_in_half_month

-- Number of video hours uploaded in the second half of the month
def video_hours_second_half : ℕ := upload_rate_second_half * days_in_half_month

-- Total number of video hours in June
theorem total_video_hours_in_june : video_hours_first_half + video_hours_second_half = 450 :=
by {
  sorry
}

end total_video_hours_in_june_l1589_158912


namespace problem1_problem2_problem3_l1589_158907

def is_real (m : ℝ) : Prop := (m^2 - 3 * m) = 0
def is_complex (m : ℝ) : Prop := (m^2 - 3 * m) ≠ 0
def is_pure_imaginary (m : ℝ) : Prop := (m^2 - 5 * m + 6) = 0 ∧ (m^2 - 3 * m) ≠ 0

theorem problem1 (m : ℝ) : is_real m ↔ (m = 0 ∨ m = 3) :=
sorry

theorem problem2 (m : ℝ) : is_complex m ↔ (m ≠ 0 ∧ m ≠ 3) :=
sorry

theorem problem3 (m : ℝ) : is_pure_imaginary m ↔ (m = 2) :=
sorry

end problem1_problem2_problem3_l1589_158907


namespace purchasing_plans_count_l1589_158904

theorem purchasing_plans_count :
  ∃ (x y : ℕ), (4 * y + 6 * x = 40)  ∧ (y ≥ 0) ∧ (x ≥ 0) ∧ (∃! (x y : ℕ), (4 * y + 6 * x = 40)  ∧ (y ≥ 0) ∧ (x ≥ 0)) := sorry

end purchasing_plans_count_l1589_158904


namespace rectangular_plot_width_l1589_158952

theorem rectangular_plot_width :
  ∀ (length width : ℕ), 
    length = 60 → 
    ∀ (poles spacing : ℕ), 
      poles = 44 → 
      spacing = 5 → 
      2 * length + 2 * width = poles * spacing →
      width = 50 :=
by
  intros length width h_length poles spacing h_poles h_spacing h_perimeter
  rw [h_length, h_poles, h_spacing] at h_perimeter
  linarith

end rectangular_plot_width_l1589_158952


namespace heartsuit_ratio_l1589_158995

-- Define the operation ⧡
def heartsuit (n m : ℕ) := n^(3+m) * m^(2+n)

-- The problem statement to prove
theorem heartsuit_ratio : heartsuit 2 4 / heartsuit 4 2 = 1 / 2 := by
  sorry

end heartsuit_ratio_l1589_158995


namespace cylindrical_to_rectangular_l1589_158997

noncomputable def convertToRectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem cylindrical_to_rectangular :
  let r := 10
  let θ := Real.pi / 3
  let z := 2
  let r' := 2 * r
  let z' := z + 1
  convertToRectangular r' θ z' = (10, 10 * Real.sqrt 3, 3) :=
by
  sorry

end cylindrical_to_rectangular_l1589_158997


namespace total_odd_green_red_marbles_l1589_158900

def Sara_green : ℕ := 3
def Sara_red : ℕ := 5
def Tom_green : ℕ := 4
def Tom_red : ℕ := 7
def Lisa_green : ℕ := 5
def Lisa_red : ℕ := 3

theorem total_odd_green_red_marbles : 
  (if Sara_green % 2 = 1 then Sara_green else 0) +
  (if Sara_red % 2 = 1 then Sara_red else 0) +
  (if Tom_green % 2 = 1 then Tom_green else 0) +
  (if Tom_red % 2 = 1 then Tom_red else 0) +
  (if Lisa_green % 2 = 1 then Lisa_green else 0) +
  (if Lisa_red % 2 = 1 then Lisa_red else 0) = 23 := by
  sorry

end total_odd_green_red_marbles_l1589_158900


namespace sectorChordLength_correct_l1589_158964

open Real

noncomputable def sectorChordLength (r α : ℝ) : ℝ :=
  2 * r * sin (α / 2)

theorem sectorChordLength_correct :
  ∃ (r α : ℝ), (1/2) * α * r^2 = 1 ∧ 2 * r + α * r = 4 ∧ sectorChordLength r α = 2 * sin 1 :=
by {
  sorry
}

end sectorChordLength_correct_l1589_158964


namespace intersection_of_A_and_B_l1589_158943

def A : Set ℝ := {x | 1 < x ∧ x < 7}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 5}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x ≤ 5} := by
  sorry

end intersection_of_A_and_B_l1589_158943


namespace find_ratio_l1589_158984

variable {d : ℕ}
variable {a : ℕ → ℝ}

-- Conditions: arithmetic sequence with non-zero common difference, and geometric sequence terms
axiom arithmetic_sequence (n : ℕ) : a n = a 1 + (n - 1) * d
axiom non_zero_d : d ≠ 0
axiom geometric_sequence : (a 1 + 2*d)^2 = a 1 * (a 1 + 8*d)

-- Theorem to prove the desired ratio
theorem find_ratio : (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 :=
sorry

end find_ratio_l1589_158984


namespace direct_variation_exponent_l1589_158948

variable {X Y Z : Type}

theorem direct_variation_exponent (k j : ℝ) (x y z : ℝ) 
  (h1 : x = k * y^4) 
  (h2 : y = j * z^3) : 
  ∃ m : ℝ, x = m * z^12 :=
by
  sorry

end direct_variation_exponent_l1589_158948


namespace ten_percent_of_fifty_percent_of_five_hundred_l1589_158908

theorem ten_percent_of_fifty_percent_of_five_hundred :
  0.10 * (0.50 * 500) = 25 :=
by
  sorry

end ten_percent_of_fifty_percent_of_five_hundred_l1589_158908


namespace determine_number_l1589_158932

def is_divisible_by_9 (n : ℕ) : Prop :=
  (n.digits 10).sum % 9 = 0

def is_divisible_by_5 (n : ℕ) : Prop :=
  n % 10 = 0 ∨ n % 10 = 5

def ten_power (n p : ℕ) : ℕ :=
  n * 10 ^ p

theorem determine_number (a b : ℕ) (h₁ : b = 0 ∨ b = 5)
  (h₂ : is_divisible_by_9 (7 + 2 + a + 3 + b))
  (h₃ : is_divisible_by_5 (7 * 10000 + 2 * 1000 + a * 100 + 3 * 10 + b)) :
  (7 * 10000 + 2 * 1000 + a * 100 + 3 * 10 + b = 72630 ∨ 
   7 * 10000 + 2 * 1000 + a * 100 + 3 * 10 + b = 72135) :=
by sorry

end determine_number_l1589_158932


namespace math_problem_l1589_158967

variables (a b c : ℤ)

theorem math_problem (h1 : a - (b - 2 * c) = 19) (h2 : a - b - 2 * c = 7) : a - b = 13 := by
  sorry

end math_problem_l1589_158967


namespace least_positive_integer_with_12_factors_l1589_158920

def has_exactly_12_factors (n : ℕ) : Prop :=
  n.factors.length = 12

theorem least_positive_integer_with_12_factors : ∃ k : ℕ, has_exactly_12_factors k ∧ (∀ n : ℕ, has_exactly_12_factors n → n ≥ k) ∧ k = 72 :=
by
  sorry

end least_positive_integer_with_12_factors_l1589_158920


namespace difference_of_cubes_divisible_by_8_l1589_158970

theorem difference_of_cubes_divisible_by_8 (a b : ℤ) : 
  8 ∣ ((2 * a - 1) ^ 3 - (2 * b - 1) ^ 3) := 
by
  sorry

end difference_of_cubes_divisible_by_8_l1589_158970


namespace sequence_general_formula_l1589_158965

theorem sequence_general_formula (n : ℕ) (hn : n > 0) 
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (hS : ∀ n, S n = 1 - n * a n) 
  (hpos : ∀ n, a n > 0) : 
  (a n = 1 / (n * (n + 1))) :=
sorry

end sequence_general_formula_l1589_158965


namespace relation_1_relation_2_relation_3_general_relationship_l1589_158981

theorem relation_1 (a b : ℝ) (h1: a = 3) (h2: b = 3) : a^2 + b^2 = 2 * a * b :=
by 
  have h : a = 3 := h1
  have h' : b = 3 := h2
  sorry

theorem relation_2 (a b : ℝ) (h1: a = 2) (h2: b = 1/2) : a^2 + b^2 > 2 * a * b :=
by 
  have h : a = 2 := h1
  have h' : b = 1/2 := h2
  sorry

theorem relation_3 (a b : ℝ) (h1: a = -2) (h2: b = 3) : a^2 + b^2 > 2 * a * b :=
by 
  have h : a = -2 := h1
  have h' : b = 3 := h2
  sorry

theorem general_relationship (a b : ℝ) : a^2 + b^2 ≥ 2 * a * b :=
by
  sorry

end relation_1_relation_2_relation_3_general_relationship_l1589_158981


namespace triangle_sine_value_l1589_158956

-- Define the triangle sides and angles
variables {a b c A B C : ℝ}

-- Main theorem stating the proof problem
theorem triangle_sine_value (h : a^2 = b^2 + c^2 - bc) :
  (a * Real.sin B) / b = Real.sqrt 3 / 2 := sorry

end triangle_sine_value_l1589_158956


namespace pencil_cost_l1589_158903

-- Definitions of given conditions
def has_amount : ℝ := 5.00  -- Elizabeth has 5 dollars
def borrowed_amount : ℝ := 0.53  -- She borrowed 53 cents
def needed_amount : ℝ := 0.47  -- She needs 47 cents more

-- Theorem to prove the cost of the pencil
theorem pencil_cost : has_amount + borrowed_amount + needed_amount = 6.00 := by 
  sorry

end pencil_cost_l1589_158903


namespace find_m_of_parallel_lines_l1589_158935

theorem find_m_of_parallel_lines
  (m : ℝ) 
  (parallel : ∀ x y, (x - 2 * y + 5 = 0 → 2 * x + m * y - 5 = 0)) :
  m = -4 :=
sorry

end find_m_of_parallel_lines_l1589_158935


namespace shuttle_speed_in_km_per_sec_l1589_158923

variable (speed_mph : ℝ) (miles_to_km : ℝ) (hour_to_sec : ℝ)

theorem shuttle_speed_in_km_per_sec
  (h_speed_mph : speed_mph = 18000)
  (h_miles_to_km : miles_to_km = 1.60934)
  (h_hour_to_sec : hour_to_sec = 3600) :
  (speed_mph * miles_to_km) / hour_to_sec = 8.046 := by
sorry

end shuttle_speed_in_km_per_sec_l1589_158923


namespace sum_of_areas_is_858_l1589_158911

def first_six_odd_squares : List ℕ := [1^2, 3^2, 5^2, 7^2, 9^2, 11^2]

def rectangle_area (width length : ℕ) : ℕ := width * length

def sum_of_areas : ℕ := (first_six_odd_squares.map (rectangle_area 3)).sum

theorem sum_of_areas_is_858 : sum_of_areas = 858 := 
by
  -- Our aim is to show that sum_of_areas is 858
  -- The proof will be developed here
  sorry

end sum_of_areas_is_858_l1589_158911


namespace expenditure_on_digging_l1589_158938

noncomputable def volume_of_cylinder (r h : ℝ) := 
  Real.pi * r^2 * h

noncomputable def rate_per_cubic_meter (cost : ℝ) (r h : ℝ) : ℝ := 
  cost / (volume_of_cylinder r h)

theorem expenditure_on_digging (d h : ℝ) (cost : ℝ) (r : ℝ) (π : ℝ) (rate : ℝ)
  (h₀ : d = 3) (h₁ : h = 14) (h₂ : cost = 1682.32) (h₃ : r = d / 2) (h₄ : π = Real.pi) 
  : rate_per_cubic_meter cost r h = 17 := sorry

end expenditure_on_digging_l1589_158938


namespace simplify_expression_l1589_158913

theorem simplify_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
  (a^3 - b^3) / (a * b^2) - (ab^2 - b^3) / (ab^2 - a^3) = (a^3 - ab^2 + b^4) / (a * b^2) :=
sorry

end simplify_expression_l1589_158913


namespace algebraic_expression_value_l1589_158982

theorem algebraic_expression_value (a : ℝ) (h : a^2 - 2*a - 1 = 0) : 2*a^2 - 4*a + 2023 = 2025 :=
sorry

end algebraic_expression_value_l1589_158982


namespace partial_fractions_sum_zero_l1589_158960

theorem partial_fractions_sum_zero (A B C D E : ℚ) :
  (∀ x : ℚ, 
     x ≠ 0 → x ≠ -1 → x ≠ -2 → x ≠ -3 → x ≠ -4 →
     1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4)) = 
     A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4)) →
  A + B + C + D + E = 0 :=
by
  intros h
  sorry

end partial_fractions_sum_zero_l1589_158960


namespace cumulative_revenue_eq_l1589_158963

-- Define the initial box office revenue and growth rate
def initial_revenue : ℝ := 3
def growth_rate (x : ℝ) : ℝ := x

-- Define the cumulative revenue equation after 3 days
def cumulative_revenue (x : ℝ) : ℝ :=
  initial_revenue + initial_revenue * (1 + growth_rate x) + initial_revenue * (1 + growth_rate x) ^ 2

-- State the theorem that proves the equation
theorem cumulative_revenue_eq (x : ℝ) :
  cumulative_revenue x = 10 :=
sorry

end cumulative_revenue_eq_l1589_158963


namespace sales_tax_amount_l1589_158958

variable (T : ℝ := 25) -- Total amount spent
variable (y : ℝ := 19.7) -- Cost of tax-free items
variable (r : ℝ := 0.06) -- Tax rate

theorem sales_tax_amount : 
  ∃ t : ℝ, t = 0.3 ∧ (T - y) * r = t :=
by 
  sorry

end sales_tax_amount_l1589_158958


namespace correct_solution_l1589_158929

variable (x y : ℤ) (a b : ℤ) (h1 : 2 * x + a * y = 6) (h2 : b * x - 7 * y = 16)

theorem correct_solution : 
  (∃ x y : ℤ, 2 * x - 3 * y = 6 ∧ 5 * x - 7 * y = 16 ∧ x = 6 ∧ y = 2) :=
by
  use 6, 2
  constructor
  · exact sorry -- 2 * 6 - 3 * 2 = 6
  constructor
  · exact sorry -- 5 * 6 - 7 * 2 = 16
  constructor
  · exact rfl
  · exact rfl

end correct_solution_l1589_158929


namespace trajectory_is_parabola_l1589_158926

def distance_to_line (p : ℝ × ℝ) (a : ℝ) : ℝ :=
|p.1 - a|

noncomputable def distance_to_point (p q : ℝ × ℝ) : ℝ :=
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

def parabola_condition (P : ℝ × ℝ) : Prop :=
distance_to_line P (-1) + 1 = distance_to_point P (2, 0)

theorem trajectory_is_parabola : ∀ (P : ℝ × ℝ), parabola_condition P ↔
(P.1 + 1)^2 = (Real.sqrt ((P.1 - 2)^2 + P.2^2))^2 := 
by 
  sorry

end trajectory_is_parabola_l1589_158926


namespace tail_to_body_ratio_l1589_158975

variables (B : ℝ) (tail : ℝ := 9) (total_length : ℝ := 30)
variables (head_ratio : ℝ := 1/6)

-- Condition: The overall length is 30 inches
def overall_length_eq : Prop := B + B * head_ratio + tail = total_length

-- Theorem: Ratio of tail length to body length is 1:2
theorem tail_to_body_ratio (h : overall_length_eq B) : tail / B = 1 / 2 :=
sorry

end tail_to_body_ratio_l1589_158975


namespace polynomial_value_l1589_158977

theorem polynomial_value
  (x : ℝ)
  (h : x^2 + 2 * x - 2 = 0) :
  4 - 2 * x - x^2 = 2 :=
by
  sorry

end polynomial_value_l1589_158977


namespace solution_set_of_inequality_l1589_158917

theorem solution_set_of_inequality (x : ℝ) : 
  abs ((x + 2) / x) < 1 ↔ x < -1 :=
by
  sorry

end solution_set_of_inequality_l1589_158917


namespace sequence_is_increasing_l1589_158947

theorem sequence_is_increasing :
  ∀ n m : ℕ, n < m → (1 - 2 / (n + 1) : ℝ) < (1 - 2 / (m + 1) : ℝ) :=
by
  intro n m hnm
  have : (2 : ℝ) / (n + 1) > 2 / (m + 1) :=
    sorry
  linarith [this]

end sequence_is_increasing_l1589_158947


namespace min_even_integers_least_one_l1589_158978

theorem min_even_integers_least_one (x y a b m n o : ℤ) 
  (h1 : x + y = 29)
  (h2 : x + y + a + b = 47)
  (h3 : x + y + a + b + m + n + o = 66) :
  ∃ e : ℕ, (e = 1) := by
sorry

end min_even_integers_least_one_l1589_158978


namespace option_C_correct_l1589_158973

theorem option_C_correct : ∀ x : ℝ, x^2 + 1 ≥ 2 * |x| :=
by
  intro x
  sorry

end option_C_correct_l1589_158973


namespace projectile_first_reaches_28_l1589_158934

theorem projectile_first_reaches_28 (t : ℝ) (h_eq : ∀ t, -4.9 * t^2 + 23.8 * t = 28) : 
    t = 2 :=
sorry

end projectile_first_reaches_28_l1589_158934


namespace abby_potatoes_peeled_l1589_158993

theorem abby_potatoes_peeled (total_potatoes : ℕ) (homers_rate : ℕ) (abbys_rate : ℕ) (time_alone : ℕ) (potatoes_peeled : ℕ) :
  (total_potatoes = 60) →
  (homers_rate = 4) →
  (abbys_rate = 6) →
  (time_alone = 6) →
  (potatoes_peeled = 22) :=
  sorry

end abby_potatoes_peeled_l1589_158993


namespace distance_between_centers_l1589_158980

variable (P R r : ℝ)
variable (h_tangent : P = R - r)
variable (h_radius1 : R = 6)
variable (h_radius2 : r = 3)

theorem distance_between_centers : P = 3 := by
  sorry

end distance_between_centers_l1589_158980


namespace simple_interest_years_l1589_158949

theorem simple_interest_years (r1 r2 t2 P1 P2 S : ℝ) (hP1: P1 = 3225) (hP2: P2 = 8000) (hr1: r1 = 0.08) (hr2: r2 = 0.15) (ht2: t2 = 2) (hCI : S = 2580) :
    S / 2 = (P1 * r1 * t) / 100 → t = 5 :=
by
  sorry

end simple_interest_years_l1589_158949


namespace base_form_exists_l1589_158955

-- Definitions for three-digit number and its reverse in base g
def N (a b c g : ℕ) : ℕ := a * g^2 + b * g + c
def N_reverse (a b c g : ℕ) : ℕ := c * g^2 + b * g + a

-- The problem statement in Lean
theorem base_form_exists (a b c g : ℕ) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) (h₄ : 0 < g)
    (h₅ : N a b c g = 2 * N_reverse a b c g) : ∃ k : ℕ, g = 3 * k + 2 ∧ k > 0 :=
by
  sorry

end base_form_exists_l1589_158955


namespace sum_m_n_l1589_158925

-- We define the conditions and problem
variables (m n : ℕ)

-- Conditions
def conditions := m > 50 ∧ n > 50 ∧ Nat.lcm m n = 480 ∧ Nat.gcd m n = 12

-- Statement to prove
theorem sum_m_n : conditions m n → m + n = 156 := by sorry

end sum_m_n_l1589_158925


namespace base4_division_l1589_158983

/-- Given in base 4:
2023_4 div 13_4 = 155_4
We need to prove the quotient is equal to 155_4.
-/
theorem base4_division (n m q r : ℕ) (h1 : n = 2 * 4^3 + 0 * 4^2 + 2 * 4^1 + 3 * 4^0)
    (h2 : m = 1 * 4^1 + 3 * 4^0)
    (h3 : q = 1 * 4^2 + 5 * 4^1 + 5 * 4^0)
    (h4 : n = m * q + r)
    (h5 : 0 ≤ r ∧ r < m):
  q = 1 * 4^2 + 5 * 4^1 + 5 * 4^0 := 
by
  sorry

end base4_division_l1589_158983


namespace smallest_square_condition_l1589_158992

-- Definition of the conditions
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_last_digit_not_zero (n : ℕ) : Prop := n % 10 ≠ 0

def remove_last_two_digits (n : ℕ) : ℕ :=
  n / 100

-- The statement of the theorem we need to prove
theorem smallest_square_condition : 
  ∃ n : ℕ, is_square n ∧ has_last_digit_not_zero n ∧ is_square (remove_last_two_digits n) ∧ 121 ≤ n :=
sorry

end smallest_square_condition_l1589_158992


namespace solve_nested_function_l1589_158968

def f (x : ℝ) : ℝ := x^2 + 12 * x + 30

theorem solve_nested_function :
  ∃ x : ℝ, f (f (f (f (f x)))) = 0 ↔ (x = -6 + 6^(1/32) ∨ x = -6 - 6^(1/32)) :=
by sorry

end solve_nested_function_l1589_158968


namespace find_coordinates_of_P_l1589_158979

-- Define points N and M with given symmetries.
structure Point where
  x : ℝ
  y : ℝ

def symmetric_about_x (P1 P2 : Point) : Prop :=
  P1.x = P2.x ∧ P1.y = -P2.y

def symmetric_about_y (P1 P2 : Point) : Prop :=
  P1.x = -P2.x ∧ P1.y = P2.y

-- Given conditions
def N : Point := ⟨1, 2⟩
def M : Point := ⟨-1, 2⟩ -- derived from symmetry about y-axis with N
def P : Point := ⟨-1, -2⟩ -- derived from symmetry about x-axis with M

theorem find_coordinates_of_P :
  symmetric_about_x M P ∧ symmetric_about_y N M → P = ⟨-1, -2⟩ :=
by
  sorry

end find_coordinates_of_P_l1589_158979


namespace interest_difference_l1589_158969

theorem interest_difference
  (principal : ℕ) (rate : ℚ) (time : ℕ) (interest : ℚ) (difference : ℚ)
  (h1 : principal = 600)
  (h2 : rate = 0.05)
  (h3 : time = 8)
  (h4 : interest = principal * (rate * time))
  (h5 : difference = principal - interest) :
  difference = 360 :=
by sorry

end interest_difference_l1589_158969


namespace gina_total_cost_l1589_158946

-- Define the constants based on the conditions
def total_credits : ℕ := 18
def reg_credits : ℕ := 12
def reg_cost_per_credit : ℕ := 450
def lab_credits : ℕ := 6
def lab_cost_per_credit : ℕ := 550
def num_textbooks : ℕ := 3
def textbook_cost : ℕ := 150
def num_online_resources : ℕ := 4
def online_resource_cost : ℕ := 95
def facilities_fee : ℕ := 200
def lab_fee_per_credit : ℕ := 75

-- Calculating the total cost
noncomputable def total_cost : ℕ :=
  (reg_credits * reg_cost_per_credit) +
  (lab_credits * lab_cost_per_credit) +
  (num_textbooks * textbook_cost) +
  (num_online_resources * online_resource_cost) +
  facilities_fee +
  (lab_credits * lab_fee_per_credit)

-- The proof problem to show that the total cost is 10180
theorem gina_total_cost : total_cost = 10180 := by
  sorry

end gina_total_cost_l1589_158946


namespace distance_from_A_to_B_l1589_158942

theorem distance_from_A_to_B (d C1A C1B C2A C2B : ℝ) (h1 : C1A + C1B = d)
  (h2 : C2A + C2B = d) (h3 : (C1A = 2 * C1B) ∨ (C1B = 2 * C1A)) 
  (h4 : (C2A = 3 * C2B) ∨ (C2B = 3 * C2A))
  (h5 : |C2A - C1A| = 10) : d = 120 ∨ d = 24 :=
sorry

end distance_from_A_to_B_l1589_158942


namespace ice_cream_remaining_l1589_158916

def total_initial_scoops : ℕ := 3 * 10
def ethan_scoops : ℕ := 1 + 1
def lucas_danny_connor_scoops : ℕ := 2 * 3
def olivia_scoops : ℕ := 1 + 1
def shannon_scoops : ℕ := 2 * olivia_scoops
def total_consumed_scoops : ℕ := ethan_scoops + lucas_danny_connor_scoops + olivia_scoops + shannon_scoops
def remaining_scoops : ℕ := total_initial_scoops - total_consumed_scoops

theorem ice_cream_remaining : remaining_scoops = 16 := by
  sorry

end ice_cream_remaining_l1589_158916


namespace polynomial_has_real_root_l1589_158901

noncomputable def P : Polynomial ℝ := sorry

variables (a1 a2 a3 b1 b2 b3 : ℝ) (h_nonzero : a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0)
variables (h_eq : ∀ x : ℝ, P.eval (a1 * x + b1) + P.eval (a2 * x + b2) = P.eval (a3 * x + b3))

theorem polynomial_has_real_root : ∃ x : ℝ, P.eval x = 0 :=
sorry

end polynomial_has_real_root_l1589_158901


namespace A_leaves_after_2_days_l1589_158953

noncomputable def A_work_rate : ℚ := 1 / 20
noncomputable def B_work_rate : ℚ := 1 / 30
noncomputable def C_work_rate : ℚ := 1 / 10
noncomputable def C_days_work : ℚ := 4
noncomputable def total_days_work : ℚ := 15

theorem A_leaves_after_2_days (x : ℚ) : 
  2 / 5 + x / 12 + (15 - x) / 30 = 1 → x = 2 :=
by
  intro h
  sorry

end A_leaves_after_2_days_l1589_158953


namespace number_of_digits_in_x_l1589_158962

open Real

theorem number_of_digits_in_x
  (x y : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (hxy_inequality : x > y)
  (hxy_prod : x * y = 490)
  (hlog_cond : (log x - log 7) * (log y - log 7) = -143/4) :
  ∃ n : ℕ, n = 8 ∧ (10^(n - 1) ≤ x ∧ x < 10^n) :=
by
  sorry

end number_of_digits_in_x_l1589_158962


namespace correct_polynomial_multiplication_l1589_158959

theorem correct_polynomial_multiplication (a b : ℤ) (x : ℝ)
  (h1 : 2 * b - 3 * a = 11)
  (h2 : 2 * b + a = -9) :
  (2 * x + a) * (3 * x + b) = 6 * x^2 - 19 * x + 10 := by
  sorry

end correct_polynomial_multiplication_l1589_158959


namespace balloons_difference_l1589_158941

-- Define the balloons each person brought
def Allan_red := 150
def Allan_blue_total := 75
def Allan_forgotten_blue := 25
def Allan_green := 30

def Jake_red := 100
def Jake_blue := 50
def Jake_green := 45

-- Calculate the actual balloons Allan brought to the park
def Allan_blue := Allan_blue_total - Allan_forgotten_blue
def Allan_total := Allan_red + Allan_blue + Allan_green

-- Calculate the total number of balloons Jake brought
def Jake_total := Jake_red + Jake_blue + Jake_green

-- State the problem: Prove Allan distributed 35 more balloons than Jake
theorem balloons_difference : Allan_total - Jake_total = 35 := 
by
  sorry

end balloons_difference_l1589_158941


namespace expected_winnings_l1589_158996

def probability_heads : ℚ := 1 / 3
def probability_tails : ℚ := 1 / 2
def probability_edge : ℚ := 1 / 6

def winning_heads : ℚ := 2
def winning_tails : ℚ := 2
def losing_edge : ℚ := -4

def expected_value : ℚ := probability_heads * winning_heads + probability_tails * winning_tails + probability_edge * losing_edge

theorem expected_winnings : expected_value = 1 := by
  sorry

end expected_winnings_l1589_158996


namespace minimum_perimeter_is_12_l1589_158921

noncomputable def minimum_perimeter_upper_base_frustum
  (a b : ℝ) (h : ℝ) (V : ℝ) : ℝ :=
if h = 3 ∧ V = 63 ∧ (a * b = 9) then
  2 * (a + b)
else
  0 -- this case will never be used

theorem minimum_perimeter_is_12 :
  ∃ a b : ℝ, a * b = 9 ∧ 2 * (a + b) = 12 :=
by
  existsi 3
  existsi 3
  sorry

end minimum_perimeter_is_12_l1589_158921


namespace find_number_l1589_158991

theorem find_number (x : ℝ) (h : 0.35 * x = 0.50 * x - 24) : x = 160 :=
by
  sorry

end find_number_l1589_158991


namespace median_eq_altitude_eq_perp_bisector_eq_l1589_158931

open Real

def point := ℝ × ℝ

def A : point := (1, 3)
def B : point := (3, 1)
def C : point := (-1, 0)

-- Median on BC
theorem median_eq : ∀ (x y : ℝ), (x, y) = A ∨ (x, y) = ((1 + (-1))/2, (1 + 0)/2) → x = 1 :=
by
  intros x y h
  sorry

-- Altitude on BC
theorem altitude_eq : ∀ (x y : ℝ), (x, y) = A ∨ (x - 1) / (y - 3) = -4 → 4*x + y - 7 = 0 :=
by
  intros x y h
  sorry

-- Perpendicular bisector of BC
theorem perp_bisector_eq : ∀ (x y : ℝ), (x = 1 ∧ y = 1/2) ∨ (x - 1) / (y - 1/2) = -4 
                          → 8*x + 2*y - 9 = 0 :=
by
  intros x y h
  sorry

end median_eq_altitude_eq_perp_bisector_eq_l1589_158931


namespace points_lie_on_hyperbola_l1589_158909

noncomputable def point_on_hyperbola (t : ℝ) : Prop :=
  let x := 2 * (Real.exp t + Real.exp (-t))
  let y := 4 * (Real.exp t - Real.exp (-t))
  (x^2 / 16) - (y^2 / 64) = 1

theorem points_lie_on_hyperbola (t : ℝ) : point_on_hyperbola t := 
by
  sorry

end points_lie_on_hyperbola_l1589_158909


namespace difference_of_squares_l1589_158950

theorem difference_of_squares (a b : ℕ) (h1: a = 630) (h2: b = 570) : a^2 - b^2 = 72000 :=
by
  sorry

end difference_of_squares_l1589_158950


namespace part1_part2_l1589_158998

-- Definitions for part 1
def total_souvenirs := 60
def price_a := 100
def price_b := 60
def total_cost_1 := 4600

-- Definitions for part 2
def max_total_cost := 4500
def twice (m : ℕ) := 2 * m

theorem part1 (x y : ℕ) (hx : x + y = total_souvenirs) (hc : price_a * x + price_b * y = total_cost_1) :
  x = 25 ∧ y = 35 :=
by
  -- You can provide the detailed proof here
  sorry

theorem part2 (m : ℕ) (hm1 : 20 ≤ m) (hm2 : m ≤ 22) (hc2 : price_a * m + price_b * (total_souvenirs - m) ≤ max_total_cost) :
  (m = 20 ∨ m = 21 ∨ m = 22) ∧ 
  ∃ W, W = min (40 * 20 + 3600) (min (40 * 21 + 3600) (40 * 22 + 3600)) ∧ W = 4400 :=
by
  -- You can provide the detailed proof here
  sorry

end part1_part2_l1589_158998


namespace repayment_is_correct_l1589_158989

noncomputable def repayment_amount (a r : ℝ) : ℝ := a * r * (1 + r) ^ 5 / ((1 + r) ^ 5 - 1)

theorem repayment_is_correct (a r : ℝ) (h_a : a > 0) (h_r : r > 0) :
  repayment_amount a r = a * r * (1 + r) ^ 5 / ((1 + r) ^ 5 - 1) :=
by
  sorry

end repayment_is_correct_l1589_158989


namespace length_of_second_platform_l1589_158918

theorem length_of_second_platform (train_length first_platform_length : ℕ) (time_to_cross_first_platform time_to_cross_second_platform : ℕ) 
  (H1 : train_length = 110) (H2 : first_platform_length = 160) (H3 : time_to_cross_first_platform = 15) 
  (H4 : time_to_cross_second_platform = 20) : ∃ second_platform_length, second_platform_length = 250 := 
by
  sorry

end length_of_second_platform_l1589_158918


namespace number_of_adult_female_alligators_l1589_158945

-- Define the conditions
def total_alligators (females males: ℕ) : ℕ := females + males

def male_alligators : ℕ := 25
def female_alligators : ℕ := 25
def juvenile_percentage : ℕ := 40

-- Calculate the number of juveniles
def juvenile_count : ℕ := (juvenile_percentage * female_alligators) / 100

-- Calculate the number of adults
def adult_female_alligators : ℕ := female_alligators - juvenile_count

-- The main theorem statement
theorem number_of_adult_female_alligators : adult_female_alligators = 15 :=
by
    sorry

end number_of_adult_female_alligators_l1589_158945


namespace find_X_l1589_158951

theorem find_X : 
  let M := 3012 / 4
  let N := M / 4
  let X := M - N
  X = 564.75 :=
by
  sorry

end find_X_l1589_158951


namespace average_score_is_7_stddev_is_2_l1589_158939

-- Define the scores list
def scores : List ℝ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]

-- Proof statement for average score
theorem average_score_is_7 : (scores.sum / scores.length) = 7 :=
by
  simp [scores]
  sorry

-- Proof statement for standard deviation
theorem stddev_is_2 : Real.sqrt ((scores.map (λ x => (x - (scores.sum / scores.length))^2)).sum / scores.length) = 2 :=
by
  simp [scores]
  sorry

end average_score_is_7_stddev_is_2_l1589_158939


namespace problem1_problem2_l1589_158930

noncomputable def cos_alpha (α : ℝ) : ℝ := (Real.sqrt 2 + 4) / 6
noncomputable def cos_alpha_plus_half_beta (α β : ℝ) : ℝ := 5 * Real.sqrt 3 / 9

theorem problem1 {α : ℝ} (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
                 (h1 : Real.cos (Real.pi / 4 + α) = 1 / 3) :
  Real.cos α = cos_alpha α :=
sorry

theorem problem2 {α β : ℝ} (hα1 : 0 < α) (hα2 : α < Real.pi / 2) 
                 (hβ1 : -Real.pi / 2 < β) (hβ2 : β < 0) 
                 (h1 : Real.cos (Real.pi / 4 + α) = 1 / 3) 
                 (h2 : Real.cos (Real.pi / 4 - β / 2) = Real.sqrt 3 / 3) :
  Real.cos (α + β / 2) = cos_alpha_plus_half_beta α β :=
sorry

end problem1_problem2_l1589_158930


namespace cubic_polynomial_sum_l1589_158987

noncomputable def Q (a b c m x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2 * m

theorem cubic_polynomial_sum (a b c m : ℝ) :
  Q a b c m 0 = 2 * m ∧ Q a b c m 1 = 3 * m ∧ Q a b c m (-1) = 5 * m →
  Q a b c m 2 + Q a b c m (-2) = 20 * m :=
by
  intro h
  sorry

end cubic_polynomial_sum_l1589_158987


namespace num_ordered_pairs_l1589_158988

theorem num_ordered_pairs :
  ∃ (m n : ℤ), (m * n ≥ 0) ∧ (m^3 + n^3 + 99 * m * n = 33^3) ∧ (35 = 35) :=
by
  sorry

end num_ordered_pairs_l1589_158988


namespace range_of_a_for_inequality_l1589_158937

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1) ↔ a ≥ 2 :=
by {
  sorry
}

end range_of_a_for_inequality_l1589_158937
