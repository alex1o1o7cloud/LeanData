import Mathlib

namespace min_value_two_x_plus_y_l41_4181

theorem min_value_two_x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y + 2 * x * y = 5 / 4) : 2 * x + y ≥ 1 :=
by
  sorry

end min_value_two_x_plus_y_l41_4181


namespace marks_lost_per_wrong_answer_l41_4120

theorem marks_lost_per_wrong_answer (score_per_correct : ℕ) (total_questions : ℕ) 
(total_score : ℕ) (correct_attempts : ℕ) (wrong_attempts : ℕ) (marks_lost_total : ℕ)
(H1 : score_per_correct = 4)
(H2 : total_questions = 75)
(H3 : total_score = 125)
(H4 : correct_attempts = 40)
(H5 : wrong_attempts = total_questions - correct_attempts)
(H6 : marks_lost_total = (correct_attempts * score_per_correct) - total_score)
: (marks_lost_total / wrong_attempts) = 1 := by
  sorry

end marks_lost_per_wrong_answer_l41_4120


namespace dot_product_solution_1_l41_4135

variable (a b : ℝ × ℝ)
variable (k : ℝ)

def two_a_add_b (a b : ℝ × ℝ) : ℝ × ℝ :=
(2 * a.1 + b.1, 2 * a.2 + b.2)

def dot_product (x y : ℝ × ℝ) : ℝ :=
x.1 * y.1 + x.2 * y.2

theorem dot_product_solution_1 :
  let a := (1, -1)
  let b := (-1, 2)
  dot_product (two_a_add_b a b) a = 1 := by
sorry

end dot_product_solution_1_l41_4135


namespace perimeter_of_triangle_l41_4124

-- Define the side lengths of the triangle
def side1 : ℕ := 2
def side2 : ℕ := 7

-- Define the third side of the triangle, which is an even number and satisfies the triangle inequality conditions
def side3 : ℕ := 6

-- Define the theorem to prove the perimeter of the triangle
theorem perimeter_of_triangle : side1 + side2 + side3 = 15 := by
  -- The proof is omitted for brevity
  sorry

end perimeter_of_triangle_l41_4124


namespace coordinates_of_N_l41_4191

theorem coordinates_of_N
  (M : ℝ × ℝ)
  (a : ℝ × ℝ)
  (x y : ℝ)
  (hM : M = (5, -6))
  (ha : a = (1, -2))
  (hMN : (x - M.1, y - M.2) = (-3 * a.1, -3 * a.2)) :
  (x, y) = (2, 0) :=
by
  sorry

end coordinates_of_N_l41_4191


namespace mr_bird_on_time_58_mph_l41_4138

def mr_bird_travel_speed_exactly_on_time (d t: ℝ) (h₁ : d = 50 * (t + 1 / 15)) (h₂ : d = 70 * (t - 1 / 15)) : ℝ :=
  58

theorem mr_bird_on_time_58_mph (d t: ℝ) (h₁ : d = 50 * (t + 1 / 15)) (h₂ : d = 70 * (t - 1 / 15)) :
  mr_bird_travel_speed_exactly_on_time d t h₁ h₂ = 58 := 
  by
  sorry

end mr_bird_on_time_58_mph_l41_4138


namespace find_unknown_number_l41_4169

theorem find_unknown_number (a n : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * n * 45 * 49) : n = 125 :=
sorry

end find_unknown_number_l41_4169


namespace distance_eq_3_implies_points_l41_4115

-- Definition of the distance of point A to the origin
def distance_to_origin (x : ℝ) : ℝ := |x|

-- Theorem statement translating the problem
theorem distance_eq_3_implies_points (x : ℝ) (h : distance_to_origin x = 3) :
  x = 3 ∨ x = -3 :=
sorry

end distance_eq_3_implies_points_l41_4115


namespace tank_empty_time_l41_4159

noncomputable def capacity : ℝ := 5760
noncomputable def leak_rate_time : ℝ := 6
noncomputable def inlet_rate_per_minute : ℝ := 4

-- leak rate calculation
noncomputable def leak_rate : ℝ := capacity / leak_rate_time

-- inlet rate calculation in litres per hour
noncomputable def inlet_rate : ℝ := inlet_rate_per_minute * 60

-- net emptying rate calculation
noncomputable def net_empty_rate : ℝ := leak_rate - inlet_rate

-- time to empty the tank calculation
noncomputable def time_to_empty : ℝ := capacity / net_empty_rate

-- The statement to prove
theorem tank_empty_time : time_to_empty = 8 :=
by
  -- Definition step
  have h1 : leak_rate = capacity / leak_rate_time := rfl
  have h2 : inlet_rate = inlet_rate_per_minute * 60 := rfl
  have h3 : net_empty_rate = leak_rate - inlet_rate := rfl
  have h4 : time_to_empty = capacity / net_empty_rate := rfl

  -- Final proof (skipped with sorry)
  sorry

end tank_empty_time_l41_4159


namespace equation1_equation2_equation3_equation4_l41_4175

-- 1. Solve: 2(2x-1)^2 = 8
theorem equation1 (x : ℝ) : 2 * (2 * x - 1)^2 = 8 ↔ (x = 3/2) ∨ (x = -1/2) :=
sorry

-- 2. Solve: 2x^2 + 3x - 2 = 0
theorem equation2 (x : ℝ) : 2 * x^2 + 3 * x - 2 = 0 ↔ (x = 1/2) ∨ (x = -2) :=
sorry

-- 3. Solve: x(2x-7) = 3(2x-7)
theorem equation3 (x : ℝ) : x * (2 * x - 7) = 3 * (2 * x - 7) ↔ (x = 7/2) ∨ (x = 3) :=
sorry

-- 4. Solve: 2y^2 + 8y - 1 = 0
theorem equation4 (y : ℝ) : 2 * y^2 + 8 * y - 1 = 0 ↔ (y = (-4 + 3 * Real.sqrt 2) / 2) ∨ (y = (-4 - 3 * Real.sqrt 2) / 2) :=
sorry

end equation1_equation2_equation3_equation4_l41_4175


namespace fraction_red_knights_magical_l41_4125

theorem fraction_red_knights_magical (total_knights : ℕ) (fraction_red fraction_magical : ℚ)
  (fraction_red_twice_fraction_blue : ℚ) 
  (h_total_knights : total_knights > 0)
  (h_fraction_red : fraction_red = 2 / 7)
  (h_fraction_magical : fraction_magical = 1 / 6)
  (h_relation : fraction_red_twice_fraction_blue = 2)
  (h_magic_eq : (total_knights : ℚ) * fraction_magical = 
    total_knights * fraction_red * fraction_red_twice_fraction_blue * fraction_magical / 2 + 
    total_knights * (1 - fraction_red) * fraction_magical / 2) :
  total_knights * (fraction_red * fraction_red_twice_fraction_blue / (fraction_red * fraction_red_twice_fraction_blue + (1 - fraction_red) / 2)) = 
  total_knights * 7 / 27 := 
sorry

end fraction_red_knights_magical_l41_4125


namespace total_pages_read_l41_4158

-- Definitions of the conditions
def pages_read_by_jairus : ℕ := 20

def pages_read_by_arniel : ℕ := 2 + 2 * pages_read_by_jairus

-- The statement to prove the total number of pages read by both is 62
theorem total_pages_read : pages_read_by_jairus + pages_read_by_arniel = 62 := by
  sorry

end total_pages_read_l41_4158


namespace smallest_possible_floor_sum_l41_4165

theorem smallest_possible_floor_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  ∃ (a b c : ℝ), ⌊(x + y) / z⌋ + ⌊(y + z) / x⌋ + ⌊(z + x) / y⌋ = 4 :=
sorry

end smallest_possible_floor_sum_l41_4165


namespace binom_15_3_eq_455_l41_4128

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Problem statement: Prove that binom 15 3 = 455
theorem binom_15_3_eq_455 : binom 15 3 = 455 := sorry

end binom_15_3_eq_455_l41_4128


namespace cos_of_theta_l41_4105

theorem cos_of_theta
  (A : ℝ) (a : ℝ) (m : ℝ) (θ : ℝ) 
  (hA : A = 40) 
  (ha : a = 12) 
  (hm : m = 10) 
  (h_area: A = (1/2) * a * m * Real.sin θ) 
  : Real.cos θ = (Real.sqrt 5) / 3 :=
by
  sorry

end cos_of_theta_l41_4105


namespace recorded_expenditure_l41_4133

-- Define what it means to record an income and an expenditure
def record_income (y : ℝ) : ℝ := y
def record_expenditure (y : ℝ) : ℝ := -y

-- Define specific instances for the problem
def income_recorded_as : ℝ := 20
def expenditure_value : ℝ := 75

-- Given condition
axiom income_condition : record_income income_recorded_as = 20

-- Theorem to prove the recorded expenditure
theorem recorded_expenditure : record_expenditure expenditure_value = -75 := by
  sorry

end recorded_expenditure_l41_4133


namespace range_of_m_l41_4111

theorem range_of_m 
    (m : ℝ) (x : ℝ)
    (p : x^2 - 8 * x - 20 > 0)
    (q : (x - (1 - m)) * (x - (1 + m)) > 0)
    (h : ∀ x, (x < -2 ∨ x > 10) → (x < 1 - m ∨ x > 1 + m)) :
    0 < m ∧ m ≤ 3 := by
  sorry

end range_of_m_l41_4111


namespace least_number_of_equal_cubes_l41_4114

def cuboid_dimensions := (18, 27, 36)
def ratio := (1, 2, 3)

theorem least_number_of_equal_cubes :
  ∃ n, n = 648 ∧
  ∃ a b c : ℕ,
    (a, b, c) = (3, 6, 9) ∧
    (18 % a = 0 ∧ 27 % b = 0 ∧ 36 % c = 0) ∧
    18 * 27 * 36 = n * (a * b * c) :=
sorry

end least_number_of_equal_cubes_l41_4114


namespace find_number_l41_4130

-- Let's define the condition
def condition (x : ℝ) : Prop := x * 99999 = 58293485180

-- Statement to be proved
theorem find_number : ∃ x : ℝ, condition x ∧ x = 582.935 := 
by
  sorry

end find_number_l41_4130


namespace calc_expr_eq_l41_4149

theorem calc_expr_eq : 2 + 3 / (4 + 5 / 6) = 76 / 29 := 
by 
  sorry

end calc_expr_eq_l41_4149


namespace abc_le_sqrt2_div_4_l41_4104

variable {a b c : ℝ}
variable (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
variable (h : (a^2 / (1 + a^2)) + (b^2 / (1 + b^2)) + (c^2 / (1 + c^2)) = 1)

theorem abc_le_sqrt2_div_4 (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
  (h : (a^2 / (1 + a^2)) + (b^2 / (1 + b^2)) + (c^2 / (1 + c^2)) = 1) :
  a * b * c ≤ (Real.sqrt 2) / 4 := 
sorry

end abc_le_sqrt2_div_4_l41_4104


namespace find_13th_result_l41_4137

theorem find_13th_result 
  (average_25 : ℕ → ℝ) (h1 : average_25 25 = 19)
  (average_first_12 : ℕ → ℝ) (h2 : average_first_12 12 = 14)
  (average_last_12 : ℕ → ℝ) (h3 : average_last_12 12 = 17) :
    let totalSum_25 := 25 * average_25 25
    let totalSum_first_12 := 12 * average_first_12 12
    let totalSum_last_12 := 12 * average_last_12 12
    let result_13 := totalSum_25 - totalSum_first_12 - totalSum_last_12
    result_13 = 103 :=
  by sorry

end find_13th_result_l41_4137


namespace oblique_line_plane_angle_range_l41_4109

/-- 
An oblique line intersects the plane at an angle other than a right angle. 
The angle cannot be $0$ radians or $\frac{\pi}{2}$ radians.
-/
theorem oblique_line_plane_angle_range (θ : ℝ) (h₀ : 0 < θ) (h₁ : θ < π / 2) : 
  0 < θ ∧ θ < π / 2 :=
by {
  exact ⟨h₀, h₁⟩
}

end oblique_line_plane_angle_range_l41_4109


namespace find_positive_number_l41_4162

theorem find_positive_number (m : ℝ) 
  (h : (m - 1)^2 = (3 * m - 5)^2) : 
  (m - 1)^2 = 1 ∨ (m - 1)^2 = 1 / 4 :=
by sorry

end find_positive_number_l41_4162


namespace faye_age_l41_4150

theorem faye_age (D E C F : ℤ)
  (h1 : D = E - 4)
  (h2 : E = C + 5)
  (h3 : F = C + 4)
  (hD : D = 18) :
  F = 21 :=
by
  sorry

end faye_age_l41_4150


namespace chloe_boxes_l41_4195

/-- Chloe was unboxing some of her old winter clothes. She found some boxes of clothing and
inside each box, there were 2 scarves and 6 mittens. Chloe had a total of 32 pieces of
winter clothing. How many boxes of clothing did Chloe find? -/
theorem chloe_boxes (boxes : ℕ) (total_clothing : ℕ) (pieces_per_box : ℕ) :
  pieces_per_box = 8 -> total_clothing = 32 -> total_clothing / pieces_per_box = boxes -> boxes = 4 :=
by
  intros
  sorry

end chloe_boxes_l41_4195


namespace problem1_problem2_problem3_l41_4174

-- Problem 1
theorem problem1 (x : ℝ) (h : x^2 + x - 2 = 0) : x^2 + x + 2023 = 2025 := 
  sorry

-- Problem 2
theorem problem2 (a b : ℝ) (h : a + b = 5) : 2 * (a + b) - 4 * a - 4 * b + 21 = 11 := 
  sorry

-- Problem 3
theorem problem3 (a b : ℝ) (h1 : a^2 + 3 * a * b = 20) (h2 : b^2 + 5 * a * b = 8) : 2 * a^2 - b^2 + a * b = 32 := 
  sorry

end problem1_problem2_problem3_l41_4174


namespace bill_and_harry_nuts_l41_4141

theorem bill_and_harry_nuts {Bill Harry Sue : ℕ} 
    (h1 : Bill = 6 * Harry) 
    (h2 : Harry = 2 * Sue) 
    (h3 : Sue = 48) : 
    Bill + Harry = 672 := 
by
  sorry

end bill_and_harry_nuts_l41_4141


namespace rows_seat_7_students_are_5_l41_4179

-- Definitions based on provided conditions
def total_students : Nat := 53
def total_rows (six_seat_rows seven_seat_rows : Nat) : Prop := 
  total_students = 6 * six_seat_rows + 7 * seven_seat_rows

-- To prove the number of rows seating exactly 7 students is 5
def number_of_7_seat_rows (six_seat_rows seven_seat_rows : Nat) : Prop := 
  total_rows six_seat_rows seven_seat_rows ∧ seven_seat_rows = 5

-- Statement to be proved
theorem rows_seat_7_students_are_5 : ∃ (six_seat_rows seven_seat_rows : Nat), number_of_7_seat_rows six_seat_rows seven_seat_rows := 
by
  -- Skipping the proof
  sorry

end rows_seat_7_students_are_5_l41_4179


namespace problem_I_problem_II_l41_4171

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 4 * a * x + 1
noncomputable def g (a b : ℝ) (x : ℝ) : ℝ := 6 * a^2 * Real.log x + 2 * b + 1
noncomputable def h (a b : ℝ) (x : ℝ) : ℝ := f a x + g a b x

theorem problem_I (a : ℝ) (ha : a > 0) :
  ∃ b, b = 5 / 2 * a^2 - 3 * a^2 * Real.log a ∧ ∀ b', b' ≤ 3 / 2 * Real.exp (2 / 3) :=
sorry

theorem problem_II (a x₁ x₂ : ℝ) (ha : a ≥ Real.sqrt 3 - 1) (hx : 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂) :
  (h a b x₂ - h a b x₁) / (x₂ - x₁) > 8 :=
sorry

end problem_I_problem_II_l41_4171


namespace possible_values_of_a_plus_b_l41_4199

variable (a b : ℤ)

theorem possible_values_of_a_plus_b (h1 : |a| = 2) (h2 : |b| = a) :
  (a + b = 0 ∨ a + b = 4 ∨ a + b = -4) :=
sorry

end possible_values_of_a_plus_b_l41_4199


namespace train_speed_conversion_l41_4188

-- Define the speed of the train in meters per second.
def speed_mps : ℝ := 37.503

-- Definition of the conversion factor between m/s and km/h.
def conversion_factor : ℝ := 3.6

-- Define the expected speed of the train in kilometers per hour.
def expected_speed_kmph : ℝ := 135.0108

-- Prove that the speed in km/h is the expected value.
theorem train_speed_conversion :
  (speed_mps * conversion_factor = expected_speed_kmph) :=
by
  sorry

end train_speed_conversion_l41_4188


namespace fraction_irreducible_l41_4190

theorem fraction_irreducible (n : ℤ) : Int.gcd (39 * n + 4) (26 * n + 3) = 1 := 
by 
  sorry

end fraction_irreducible_l41_4190


namespace time_spent_on_type_a_l41_4127

theorem time_spent_on_type_a (num_questions : ℕ) 
                             (exam_duration : ℕ)
                             (type_a_count : ℕ)
                             (time_ratio : ℕ)
                             (type_b_count : ℕ)
                             (x : ℕ)
                             (total_time : ℕ) :
  num_questions = 200 ∧
  exam_duration = 180 ∧
  type_a_count = 20 ∧
  time_ratio = 2 ∧
  type_b_count = 180 ∧
  total_time = 36 →
  time_ratio * x * type_a_count + x * type_b_count = exam_duration →
  total_time = 36 :=
by
  sorry

end time_spent_on_type_a_l41_4127


namespace true_discount_correct_l41_4147

noncomputable def true_discount (FV BD : ℝ) : ℝ :=
  BD / (1 + (BD / FV))

theorem true_discount_correct
  (FV BD : ℝ)
  (hFV : FV = 2260)
  (hBD : BD = 428.21) :
  true_discount FV BD = 360.00 :=
by
  sorry

end true_discount_correct_l41_4147


namespace max_inscribed_triangle_area_l41_4167

theorem max_inscribed_triangle_area (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ A, A = (3 * Real.sqrt 3 / 4) * a * b := 
sorry

end max_inscribed_triangle_area_l41_4167


namespace frank_cookies_l41_4106

theorem frank_cookies :
  ∀ (F M M_i L : ℕ),
    (F = M / 2 - 3) →
    (M = 3 * M_i) →
    (M_i = 2 * L) →
    (L = 5) →
    F = 12 :=
by
  intros F M M_i L h1 h2 h3 h4
  rw [h4] at h3
  rw [h3] at h2
  rw [h2] at h1
  sorry

end frank_cookies_l41_4106


namespace inequality_always_true_l41_4132

theorem inequality_always_true (a : ℝ) (x : ℝ) (h : -1 ≤ a ∧ a ≤ 1) :
  x^2 + (a - 4) * x + 4 - 2 * a > 0 → (x < 1 ∨ x > 3) :=
by {
  sorry
}

end inequality_always_true_l41_4132


namespace part1_part2_l41_4153

open Real

-- Define the function f
def f (x m : ℝ) : ℝ := |x - m| - 1

-- Define the function g for the second part
def g (x : ℝ) : ℝ := |x - 2| + |x + 3|

theorem part1 (m : ℝ) : (∀ x, f x m ≤ 2 ↔ -1 ≤ x ∧ x ≤ 5) → m = 2 :=
  by sorry

theorem part2 (t x: ℝ) (h: ∀ x: ℝ, f x 2 + f (x + 5) 2 ≥ t - 2) : t ≤ 5 :=
  by sorry

end part1_part2_l41_4153


namespace barbara_candies_l41_4184

theorem barbara_candies : (9 + 18) = 27 :=
by
  sorry

end barbara_candies_l41_4184


namespace sqrt_product_eq_l41_4144

theorem sqrt_product_eq :
  (16 ^ (1 / 4) : ℝ) * (64 ^ (1 / 2)) = 16 := by
  sorry

end sqrt_product_eq_l41_4144


namespace B_representation_l41_4166

def A : Set ℤ := {-1, 2, 3, 4}

def f (x : ℤ) : ℤ := x^2 - 2*x + 2

def B : Set ℤ := {y | ∃ x ∈ A, y = f x}

theorem B_representation : B = {2, 5, 10} :=
by {
  -- Proof to be provided
  sorry
}

end B_representation_l41_4166


namespace negation_equivalence_l41_4173

theorem negation_equivalence (x : ℝ) :
  (¬ (x ≥ 1 → x^2 - 4*x + 2 ≥ -1)) ↔ (x < 1 → x^2 - 4*x + 2 < -1) :=
by
  sorry

end negation_equivalence_l41_4173


namespace factorize_1_factorize_2_l41_4196

theorem factorize_1 {x : ℝ} : 2*x^2 - 4*x = 2*x*(x - 2) := 
by sorry

theorem factorize_2 {a b x y : ℝ} : a^2*(x - y) + b^2*(y - x) = (x - y) * (a + b) * (a - b) := 
by sorry

end factorize_1_factorize_2_l41_4196


namespace trees_left_after_typhoon_l41_4145

theorem trees_left_after_typhoon (trees_grown : ℕ) (trees_died : ℕ) (h1 : trees_grown = 17) (h2 : trees_died = 5) : (trees_grown - trees_died = 12) :=
by
  -- The proof would go here
  sorry

end trees_left_after_typhoon_l41_4145


namespace maximum_n_l41_4131

def arithmetic_sequence_max_n (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) : Prop :=
  ∃ d : ℤ, ∀ m : ℕ, a (m + 1) = a m + d

def is_positive_first_term (a : ℕ → ℤ) : Prop :=
  a 0 > 0

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 0 + a (n-1))) / 2

def roots_of_equation (a1006 a1007 : ℤ) : Prop :=
  a1006 * a1007 = -2011 ∧ a1006 + a1007 = 2012

theorem maximum_n (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : arithmetic_sequence_max_n a S 1007)
  (h2 : is_positive_first_term a)
  (h3 : sum_of_first_n_terms a S)
  (h4 : ∃ a1006 a1007, roots_of_equation a1006 a1007 ∧ a 1006 = a1006 ∧ a 1007 = a1007) :
  ∃ n, S n > 0 → n ≤ 1007 := 
sorry

end maximum_n_l41_4131


namespace jelly_bean_count_l41_4101

variable (b c : ℕ)
variable (h1 : b = 3 * c)
variable (h2 : b - 5 = 5 * (c - 15))

theorem jelly_bean_count : b = 105 := by
  sorry

end jelly_bean_count_l41_4101


namespace average_minutes_per_player_is_2_l41_4116

def total_player_footage := 130 + 145 + 85 + 60 + 180
def total_additional_content := 120 + 90 + 30
def pause_transition_time := 15 * (5 + 3) -- 5 players + game footage + interviews + opening/closing scenes - 1
def total_film_time := total_player_footage + total_additional_content + pause_transition_time
def number_of_players := 5
def average_seconds_per_player := total_player_footage / number_of_players
def average_minutes_per_player := average_seconds_per_player / 60

theorem average_minutes_per_player_is_2 :
  average_minutes_per_player = 2 := by
  -- Proof goes here.
  sorry

end average_minutes_per_player_is_2_l41_4116


namespace table_to_chair_ratio_l41_4126

noncomputable def price_chair : ℤ := 20
noncomputable def price_table : ℤ := 60
noncomputable def price_couch : ℤ := 300

theorem table_to_chair_ratio 
  (h1 : price_couch = 300)
  (h2 : price_couch = 5 * price_table)
  (h3 : price_chair + price_table + price_couch = 380)
  : price_table / price_chair = 3 := 
by 
  sorry

end table_to_chair_ratio_l41_4126


namespace sum_of_coordinates_x_l41_4108

-- Given points Y and Z
def Y : ℝ × ℝ := (2, 8)
def Z : ℝ × ℝ := (0, -4)

-- Given ratio conditions
def ratio_condition (X Y Z : ℝ × ℝ) : Prop :=
  dist X Z / dist X Y = 1/3 ∧ dist Z Y / dist X Y = 1/3

-- Define X, ensuring Z is the midpoint of XY
def X : ℝ × ℝ := (4, 20)

-- Prove that sum of coordinates of X is 10
theorem sum_of_coordinates_x (h : ratio_condition X Y Z) : (X.1 + X.2) = 10 := 
  sorry

end sum_of_coordinates_x_l41_4108


namespace quadratic_solution_l41_4177

theorem quadratic_solution (a : ℝ) (h : 2^2 - 3 * 2 + a = 0) : 2 * a - 1 = 3 :=
by {
  sorry
}

end quadratic_solution_l41_4177


namespace wrongly_recorded_height_l41_4185

theorem wrongly_recorded_height 
  (avg_incorrect : ℕ → ℕ → ℕ)
  (avg_correct : ℕ → ℕ → ℕ)
  (boy_count : ℕ)
  (incorrect_avg_height : ℕ) 
  (correct_avg_height : ℕ) 
  (actual_height : ℕ) 
  (correct_total_height : ℕ) 
  (incorrect_total_height: ℕ)
  (x : ℕ) :
  avg_incorrect boy_count incorrect_avg_height = incorrect_total_height →
  avg_correct boy_count correct_avg_height = correct_total_height →
  incorrect_total_height - x + actual_height = correct_total_height →
  x = 176 := 
by 
  intros h1 h2 h3
  sorry

end wrongly_recorded_height_l41_4185


namespace solution_for_equation_l41_4155

theorem solution_for_equation (m n : ℕ) (h : 0 < m ∧ 0 < n ∧ 2 * m^2 = 3 * n^3) :
  ∃ k : ℕ, 0 < k ∧ m = 18 * k^3 ∧ n = 6 * k^2 :=
by sorry

end solution_for_equation_l41_4155


namespace susan_age_l41_4157

theorem susan_age (S J B : ℝ) 
  (h1 : S = 2 * J)
  (h2 : S + J + B = 60) 
  (h3 : B = J + 10) : 
  S = 25 := sorry

end susan_age_l41_4157


namespace binomial_expansion_problem_l41_4134

theorem binomial_expansion_problem :
  ∀ (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 a_11 : ℝ),
    (1 + 2 * x) ^ 11 =
      a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 +
      a_5 * x^5 + a_6 * x^6 + a_7 * x^7 + a_8 * x^8 +
      a_9 * x^9 + a_10 * x^10 + a_11 * x^11 →
    a_1 - 2 * a_2 + 3 * a_3 - 4 * a_4 + 5 * a_5 - 6 * a_6 +
    7 * a_7 - 8 * a_8 + 9 * a_9 - 10 * a_10 + 11 * a_11 = 22 :=
by
  -- The proof is omitted for this exercise
  sorry

end binomial_expansion_problem_l41_4134


namespace average_pages_per_day_is_correct_l41_4156

-- Definitions based on the given conditions
def first_book_pages := 249
def first_book_days := 3

def second_book_pages := 379
def second_book_days := 5

def third_book_pages := 480
def third_book_days := 6

-- Definition of total pages read
def total_pages := first_book_pages + second_book_pages + third_book_pages

-- Definition of total days spent reading
def total_days := first_book_days + second_book_days + third_book_days

-- Definition of expected average pages per day
def expected_average_pages_per_day := 79.14

-- The theorem to prove
theorem average_pages_per_day_is_correct : (total_pages.toFloat / total_days.toFloat) = expected_average_pages_per_day :=
by
  sorry

end average_pages_per_day_is_correct_l41_4156


namespace material_left_eq_l41_4103

theorem material_left_eq :
  let a := (4 / 17 : ℚ)
  let b := (3 / 10 : ℚ)
  let total_bought := a + b
  let used := (0.23529411764705882 : ℚ)
  total_bought - used = (51 / 170 : ℚ) :=
by
  let a := (4 / 17 : ℚ)
  let b := (3 / 10 : ℚ)
  let total_bought := a + b
  let used := (0.23529411764705882 : ℚ)
  show total_bought - used = (51 / 170)
  sorry

end material_left_eq_l41_4103


namespace distinct_values_least_count_l41_4187

theorem distinct_values_least_count (total_integers : ℕ) (mode_count : ℕ) (unique_mode : Prop) 
  (h1 : total_integers = 3200)
  (h2 : mode_count = 17)
  (h3 : unique_mode):
  ∃ (least_count : ℕ), least_count = 200 := by
  sorry

end distinct_values_least_count_l41_4187


namespace opposite_of_negative_2023_l41_4161

-- Define the opposite condition
def is_opposite (y x : Int) : Prop := y + x = 0

theorem opposite_of_negative_2023 : ∃ x : Int, is_opposite (-2023) x ∧ x = 2023 :=
by 
  use 2023
  sorry

end opposite_of_negative_2023_l41_4161


namespace laptop_cost_l41_4189

theorem laptop_cost (L : ℝ) (smartphone_cost : ℝ) (total_cost : ℝ) (change : ℝ) (n_laptops n_smartphones : ℕ) 
  (hl_smartphone : smartphone_cost = 400) 
  (hl_laptops : n_laptops = 2) 
  (hl_smartphones : n_smartphones = 4) 
  (hl_total : total_cost = 3000)
  (hl_change : change = 200) 
  (hl_total_spent : total_cost - change = 2 * L + 4 * smartphone_cost) : 
  L = 600 :=
by 
  sorry

end laptop_cost_l41_4189


namespace smallest_lucky_number_exists_l41_4198

theorem smallest_lucky_number_exists :
  ∃ (a b c d N: ℕ), 
  N = a^2 + b^2 ∧ 
  N = c^2 + d^2 ∧ 
  a - c = 7 ∧ 
  d - b = 13 ∧ 
  N = 545 := 
by {
  sorry
}

end smallest_lucky_number_exists_l41_4198


namespace neither_sufficient_nor_necessary_l41_4151

noncomputable def a_b_conditions (a b: ℝ) : Prop :=
∃ (a b: ℝ), ¬((a - b > 0) → (a^2 - b^2 > 0)) ∧ ¬((a^2 - b^2 > 0) → (a - b > 0))

theorem neither_sufficient_nor_necessary (a b: ℝ) : a_b_conditions a b :=
sorry

end neither_sufficient_nor_necessary_l41_4151


namespace Grandfather_age_correct_l41_4183

-- Definitions based on the conditions
def Yuna_age : Nat := 9
def Father_age (Yuna_age : Nat) : Nat := Yuna_age + 27
def Grandfather_age (Father_age : Nat) : Nat := Father_age + 23

-- The theorem stating the problem to prove
theorem Grandfather_age_correct : Grandfather_age (Father_age Yuna_age) = 59 := by
  sorry

end Grandfather_age_correct_l41_4183


namespace mike_remaining_cards_l41_4102

def initial_cards (mike_cards : ℕ) : ℕ := 87
def sam_cards (sam_bought : ℕ) : ℕ := 13
def alex_cards (alex_bought : ℕ) : ℕ := 15

theorem mike_remaining_cards (mike_cards sam_bought alex_bought : ℕ) :
  mike_cards - (sam_bought + alex_bought) = 59 :=
by
  let mike_cards := initial_cards 87
  let sam_cards := sam_bought
  let alex_cards := alex_bought
  sorry

end mike_remaining_cards_l41_4102


namespace find_value_of_expression_l41_4160

theorem find_value_of_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) : x^2 + 2*x + 3 = 4 := by
  sorry

end find_value_of_expression_l41_4160


namespace largest_real_number_l41_4121

theorem largest_real_number (x : ℝ) (h : (⌊x⌋ / x) = (8 / 9)) : x ≤ 63 / 8 :=
by
  sorry

end largest_real_number_l41_4121


namespace value_subtracted_l41_4129

theorem value_subtracted (n v : ℝ) (h1 : 2 * n - v = -12) (h2 : n = -10.0) : v = -8 :=
by
  sorry

end value_subtracted_l41_4129


namespace final_solution_sugar_percentage_l41_4118

-- Define the conditions of the problem
def initial_solution_sugar_percentage : ℝ := 0.10
def replacement_fraction : ℝ := 0.25
def second_solution_sugar_percentage : ℝ := 0.26

-- Define the Lean statement that proves the final sugar percentage
theorem final_solution_sugar_percentage:
  (0.10 * (1 - 0.25) + 0.26 * 0.25) * 100 = 14 :=
by
  sorry

end final_solution_sugar_percentage_l41_4118


namespace part1_part2_l41_4180

open Set Real

-- Definitions of sets A, B, and C
def setA : Set ℝ := { x | 2 ≤ x ∧ x < 5 }
def setB : Set ℝ := { x | 1 < x ∧ x < 8 }
def setC (a : ℝ) : Set ℝ := { x | x < a - 1 ∨ x > a }

-- Conditions:
-- - Complement of A
def complementA : Set ℝ := { x | x < 2 ∨ x ≥ 5 }

-- Question parts:
-- (1) Finding intersection of complementA and B
theorem part1 : (complementA ∩ setB) = { x | (1 < x ∧ x < 2) ∨ (5 ≤ x ∧ x < 8) } := sorry

-- (2) Finding range of a for specific condition on C
theorem part2 (a : ℝ) : (setA ∪ setC a = univ) → (a ≤ 2 ∨ a > 6) := sorry

end part1_part2_l41_4180


namespace correct_system_of_equations_l41_4152

theorem correct_system_of_equations : 
  ∃ (x y : ℕ), x + y = 12 ∧ 4 * x + 3 * y = 40 := by
  -- we are stating the existence of x and y that satisfy both equations given as conditions.
  sorry

end correct_system_of_equations_l41_4152


namespace zero_of_function_l41_4123

theorem zero_of_function : ∃ x : Real, 4 * x - 2 = 0 ∧ x = 1 / 2 :=
by
  sorry

end zero_of_function_l41_4123


namespace milk_production_per_cow_l41_4192

theorem milk_production_per_cow :
  ∀ (total_cows : ℕ) (milk_price_per_gallon butter_price_per_stick total_earnings : ℝ)
    (customers customer_milk_demand gallons_per_butter : ℕ),
  total_cows = 12 →
  milk_price_per_gallon = 3 →
  butter_price_per_stick = 1.5 →
  total_earnings = 144 →
  customers = 6 →
  customer_milk_demand = 6 →
  gallons_per_butter = 2 →
  (∀ (total_milk_sold_to_customers produced_milk used_for_butter : ℕ),
    total_milk_sold_to_customers = customers * customer_milk_demand →
    produced_milk = total_milk_sold_to_customers + used_for_butter →
    used_for_butter = (total_earnings - (total_milk_sold_to_customers * milk_price_per_gallon)) / butter_price_per_stick / gallons_per_butter →
    produced_milk / total_cows = 4)
:= by sorry

end milk_production_per_cow_l41_4192


namespace polynomial_evaluation_l41_4170

theorem polynomial_evaluation (x y : ℝ) (h : 2 * x^2 + 3 * y + 3 = 8) : 6 * x^2 + 9 * y + 8 = 23 :=
sorry

end polynomial_evaluation_l41_4170


namespace cone_lateral_surface_area_l41_4194

theorem cone_lateral_surface_area (r h : ℝ) (hr : r = 3) (hh : h = 4) : 15 * Real.pi = Real.pi * r * (Real.sqrt (r^2 + h^2)) :=
by
  -- Prove that 15π = π * r * sqrt(r^2 + h^2) for r = 3 and h = 4
  sorry

end cone_lateral_surface_area_l41_4194


namespace james_total_cost_is_100_l41_4176

def cost_of_shirts (number_of_shirts : Nat) (cost_per_shirt : Nat) : Nat :=
  number_of_shirts * cost_per_shirt

def cost_of_pants (number_of_pants : Nat) (cost_per_pants : Nat) : Nat :=
  number_of_pants * cost_per_pants

def total_cost (number_of_shirts : Nat) (number_of_pants : Nat) (cost_per_shirt : Nat) (cost_per_pants : Nat) : Nat :=
  cost_of_shirts number_of_shirts cost_per_shirt + cost_of_pants number_of_pants cost_per_pants

theorem james_total_cost_is_100 : 
  total_cost 10 (10 / 2) 6 8 = 100 :=
by
  sorry

end james_total_cost_is_100_l41_4176


namespace unique_rs_exists_l41_4117

theorem unique_rs_exists (a b : ℕ) (ha : a > 1) (hb : b > 1) (gcd_ab : Nat.gcd a b = 1) :
  ∃! (r s : ℤ), (0 < r ∧ r < b) ∧ (0 < s ∧ s < a) ∧ (a * r - b * s = 1) :=
  sorry

end unique_rs_exists_l41_4117


namespace number_of_friends_l41_4100

def has14_pokemon_cards (x : String) : Prop :=
  x = "Sam" ∨ x = "Dan" ∨ x = "Tom" ∨ x = "Keith"

theorem number_of_friends :
  ∃ n, n = 4 ∧
        ∀ x, has14_pokemon_cards x ↔ x = "Sam" ∨ x = "Dan" ∨ x = "Tom" ∨ x = "Keith" :=
by
  sorry

end number_of_friends_l41_4100


namespace hypotenuse_length_l41_4143

theorem hypotenuse_length
    (a b c : ℝ)
    (h1: a^2 + b^2 + c^2 = 2450)
    (h2: b = a + 7)
    (h3: c^2 = a^2 + b^2) :
    c = 35 := sorry

end hypotenuse_length_l41_4143


namespace johns_overall_profit_l41_4163

def cost_price_grinder : ℕ := 15000
def cost_price_mobile : ℕ := 8000
def loss_percent_grinder : ℝ := 0.05
def profit_percent_mobile : ℝ := 0.10

noncomputable def loss_amount_grinder : ℝ := loss_percent_grinder * cost_price_grinder
noncomputable def selling_price_grinder : ℝ := cost_price_grinder - loss_amount_grinder

noncomputable def profit_amount_mobile : ℝ := profit_percent_mobile * cost_price_mobile
noncomputable def selling_price_mobile : ℝ := cost_price_mobile + profit_amount_mobile

noncomputable def total_cost_price : ℝ := cost_price_grinder + cost_price_mobile
noncomputable def total_selling_price : ℝ := selling_price_grinder + selling_price_mobile
noncomputable def overall_profit : ℝ := total_selling_price - total_cost_price

theorem johns_overall_profit :
  overall_profit = 50 := 
by
  sorry

end johns_overall_profit_l41_4163


namespace quadratic_root_value_l41_4136
-- Import the entirety of the necessary library

-- Define the quadratic equation with one root being -1
theorem quadratic_root_value 
    (m : ℝ)
    (h1 : ∀ x : ℝ, x^2 + m * x + 3 = 0)
    (root1 : -1 ∈ {x : ℝ | x^2 + m * x + 3 = 0}) :
    m = 4 ∧ ∃ root2 : ℝ, root2 = -3 ∧ root2 ∈ {x : ℝ | x^2 + m * x + 3 = 0} :=
by
  sorry

end quadratic_root_value_l41_4136


namespace no_real_roots_range_l41_4193

theorem no_real_roots_range (k : ℝ) :
  (∀ x : ℝ, k * x^2 - 2 * x - 1 ≠ 0) ↔ k < -1 :=
by
  sorry

end no_real_roots_range_l41_4193


namespace part1_part2_l41_4142

variable (a b : ℝ)

-- Part (1)
theorem part1 (hA : a^2 - 2 * a * b + b^2 = A) (hB: a^2 + 2 * a * b + b^2 = B) (h : a ≠ b) :
  A + B > 0 := sorry

-- Part (2)
theorem part2 (hA : a^2 - 2 * a * b + b^2 = A) (hB: a^2 + 2 * a * b + b^2 = B) (h: a * b = 1) : 
  A - B = -4 := sorry

end part1_part2_l41_4142


namespace correct_product_l41_4139

-- We define the conditions
def number1 : ℝ := 0.85
def number2 : ℝ := 3.25
def without_decimal_points_prod : ℕ := 27625

-- We state the problem
theorem correct_product (h1 : (85 : ℕ) * (325 : ℕ) = without_decimal_points_prod)
                        (h2 : number1 * number2 * 10000 = (without_decimal_points_prod : ℝ)) :
  number1 * number2 = 2.7625 :=
by sorry

end correct_product_l41_4139


namespace length_of_chord_l41_4168

theorem length_of_chord (r AB : ℝ) (h1 : r = 6) (h2 : 0 < AB) (h3 : AB <= 2 * r) : AB ≠ 14 :=
by
  sorry

end length_of_chord_l41_4168


namespace no_integer_solution_l41_4178

theorem no_integer_solution (x y z : ℤ) (n : ℕ) (h1 : Prime (x + y)) (h2 : Odd n) : ¬ (x^n + y^n = z^n) :=
sorry

end no_integer_solution_l41_4178


namespace price_second_day_is_81_percent_l41_4186

-- Define the original price P (for the sake of clarity in the proof statement)
variable (P : ℝ)

-- Define the reductions
def first_reduction (P : ℝ) : ℝ := P - 0.1 * P
def second_reduction (P : ℝ) : ℝ := first_reduction P - 0.1 * first_reduction P

-- Question translated to Lean statement
theorem price_second_day_is_81_percent (P : ℝ) : 
  (second_reduction P / P) * 100 = 81 := by
  sorry

end price_second_day_is_81_percent_l41_4186


namespace probability_winning_probability_not_winning_l41_4110

section Lottery

variable (p1 p2 p3 : ℝ)
variable (h1 : p1 = 0.1)
variable (h2 : p2 = 0.2)
variable (h3 : p3 = 0.4)

theorem probability_winning (h1 : p1 = 0.1) (h2 : p2 = 0.2) (h3 : p3 = 0.4) :
  p1 + p2 + p3 = 0.7 :=
by
  rw [h1, h2, h3]
  norm_num
  done

theorem probability_not_winning (h1 : p1 = 0.1) (h2 : p2 = 0.2) (h3 : p3 = 0.4) :
  1 - (p1 + p2 + p3) = 0.3 :=
by
  rw [h1, h2, h3]
  norm_num
  done

end Lottery

end probability_winning_probability_not_winning_l41_4110


namespace smallest_n_7n_eq_n7_mod_3_l41_4146

theorem smallest_n_7n_eq_n7_mod_3 : ∃ n : ℕ, n > 0 ∧ (7^n ≡ n^7 [MOD 3]) ∧ ∀ m : ℕ, m > 0 → (7^m ≡ m^7 [MOD 3] → m ≥ n) :=
by
  sorry

end smallest_n_7n_eq_n7_mod_3_l41_4146


namespace segment_length_294_l41_4172

theorem segment_length_294
  (A B P Q : ℝ)   -- Define points A, B, P, Q on the real line
  (h1 : P = A + (3 / 8) * (B - A))   -- P divides AB in the ratio 3:5
  (h2 : Q = A + (4 / 11) * (B - A))  -- Q divides AB in the ratio 4:7
  (h3 : Q - P = 3)                   -- The length of PQ is 3
  : B - A = 294 := 
sorry

end segment_length_294_l41_4172


namespace multiplication_subtraction_difference_l41_4154

theorem multiplication_subtraction_difference (x n : ℕ) (h₁ : x = 5) (h₂ : 3 * x = (16 - x) + n) : n = 4 :=
by
  -- Proof will go here
  sorry

end multiplication_subtraction_difference_l41_4154


namespace separate_curves_l41_4107

variable {A : Type} [CommRing A]

def crossing_characteristic (ε : A → ℤ) (A1 A2 A3 A4 : A) : Prop :=
  ε A1 + ε A2 + ε A3 + ε A4 = 0

theorem separate_curves {A : Type} [CommRing A]
  {ε : A → ℤ} {A1 A2 A3 A4 : A} 
  (h : ε A1 + ε A2 + ε A3 + ε A4 = 0)
  (h1 : ε A1 = 1 ∨ ε A1 = -1)
  (h2 : ε A2 = 1 ∨ ε A2 = -1)
  (h3 : ε A3 = 1 ∨ ε A3 = -1)
  (h4 : ε A4 = 1 ∨ ε A4 = -1) :
  (∃ B1 B2 : A, B1 ≠ B2 ∧  ∀ (A : A), ((ε A = 1) → (A = B1)) ∨ ((ε A = -1) → (A = B2))) :=
  sorry

end separate_curves_l41_4107


namespace exponential_function_fixed_point_l41_4122

theorem exponential_function_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : (1, 1) ∈ {p : ℝ × ℝ | ∃ x, p = (x, a^(x-1))} :=
by
  sorry

end exponential_function_fixed_point_l41_4122


namespace compute_expression_l41_4112

theorem compute_expression : 7^2 - 2 * 6 + (3^2 - 1) = 45 :=
by
  sorry

end compute_expression_l41_4112


namespace hexagon_shaded_area_l41_4197

-- Given conditions
variable (A B C D T : ℝ)
variable (h₁ : A = 2)
variable (h₂ : B = 3)
variable (h₃ : C = 4)
variable (h₄ : T = 20)
variable (h₅ : A + B + C + D = T)

-- The goal is to prove that the area of the shaded region (D) is 11 cm².
theorem hexagon_shaded_area : D = 11 := by
  sorry

end hexagon_shaded_area_l41_4197


namespace not_all_zero_iff_at_least_one_nonzero_l41_4148

theorem not_all_zero_iff_at_least_one_nonzero (a b c : ℝ) :
  ¬ (a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) :=
by 
  sorry

end not_all_zero_iff_at_least_one_nonzero_l41_4148


namespace solve_abs_eq_2005_l41_4164

theorem solve_abs_eq_2005 (x : ℝ) : |2005 * x - 2005| = 2005 ↔ x = 0 ∨ x = 2 := by
  sorry

end solve_abs_eq_2005_l41_4164


namespace range_of_m_for_ellipse_l41_4140

-- Define the equation of the ellipse
def ellipse_equation (m : ℝ) (x y : ℝ) : Prop :=
  m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

-- The theorem to prove
theorem range_of_m_for_ellipse (m : ℝ) :
  (∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) →
  5 < m :=
sorry

end range_of_m_for_ellipse_l41_4140


namespace no_arith_geo_progression_S1_S2_S3_l41_4119

noncomputable def S_1 (A B C : Point) : ℝ := sorry -- area of triangle ABC
noncomputable def S_2 (A B E : Point) : ℝ := sorry -- area of triangle ABE
noncomputable def S_3 (A B D : Point) : ℝ := sorry -- area of triangle ABD

def bisecting_plane (A B D C E : Point) : Prop := sorry -- plane bisects dihedral angle at AB

theorem no_arith_geo_progression_S1_S2_S3 (A B C D E : Point) 
(h_bisect : bisecting_plane A B D C E) :
¬ (∃ (S1 S2 S3 : ℝ), S1 = S_1 A B C ∧ S2 = S_2 A B E ∧ S3 = S_3 A B D ∧ 
  (S2 = (S1 + S3) / 2 ∨ S2^2 = S1 * S3 )) :=
sorry

end no_arith_geo_progression_S1_S2_S3_l41_4119


namespace odd_function_f_2_eq_2_l41_4113

noncomputable def f (x : ℝ) : ℝ := 
if x < 0 then x^2 + 3 * x else -(if -x < 0 then (-x)^2 + 3 * (-x) else x^2 + 3 * x)

theorem odd_function_f_2_eq_2 : f 2 = 2 :=
by
  -- sorry will be used to skip the actual proof
  sorry

end odd_function_f_2_eq_2_l41_4113


namespace total_games_l41_4182

variable (L : ℕ) -- Number of games the team lost

-- Define the number of wins
def Wins := 3 * L + 14

theorem total_games (h_wins : Wins = 101) : (Wins + L = 130) :=
by
  sorry

end total_games_l41_4182
