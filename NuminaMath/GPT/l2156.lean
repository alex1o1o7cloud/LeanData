import Mathlib

namespace calculate_y_when_x_is_neg2_l2156_215692

def conditional_program (x : ℤ) : ℤ :=
  if x < 0 then
    2 * x + 3
  else if x > 0 then
    -2 * x + 5
  else
    0

theorem calculate_y_when_x_is_neg2 : conditional_program (-2) = -1 :=
by
  sorry

end calculate_y_when_x_is_neg2_l2156_215692


namespace common_difference_is_1_l2156_215600

variable (a_2 a_5 : ℕ) (d : ℤ)

def arithmetic_sequence (n a_1 : ℤ) (d : ℤ) : ℤ := a_1 + (n - 1) * d

theorem common_difference_is_1 
  (h1 : arithmetic_sequence 2 a_1 d = 3) 
  (h2 : arithmetic_sequence 5 a_1 d = 6) : 
  d = 1 := 
sorry

end common_difference_is_1_l2156_215600


namespace probability_eq_l2156_215605

noncomputable def probability_exactly_two_one_digit_and_three_two_digit : ℚ := 
  let n := 5
  let p_one_digit := 9 / 20
  let p_two_digit := 11 / 20
  let binomial_coeff := Nat.choose 5 2
  (binomial_coeff * p_one_digit^2 * p_two_digit^3)

theorem probability_eq : probability_exactly_two_one_digit_and_three_two_digit = 539055 / 1600000 := 
  sorry

end probability_eq_l2156_215605


namespace money_r_gets_l2156_215665

def total_amount : ℕ := 1210
def p_to_q := 5 / 4
def q_to_r := 9 / 10

theorem money_r_gets :
  let P := (total_amount * 45) / 121
  let Q := (total_amount * 36) / 121
  let R := (total_amount * 40) / 121
  R = 400 := by
  sorry

end money_r_gets_l2156_215665


namespace sum_of_smallest_x_and_y_l2156_215695

theorem sum_of_smallest_x_and_y (x y : ℕ) (h1 : 0 < x) (h2 : 0 < y)
  (hx : ∃ k : ℕ, (480 * x) = k * k ∧ ∀ z : ℕ, 0 < z → (480 * z) = k * k → x ≤ z)
  (hy : ∃ n : ℕ, (480 * y) = n * n * n ∧ ∀ z : ℕ, 0 < z → (480 * z) = n * n * n → y ≤ z) :
  x + y = 480 := sorry

end sum_of_smallest_x_and_y_l2156_215695


namespace max_ratio_a_c_over_b_d_l2156_215601

-- Given conditions as Lean definitions
variables {a b c d : ℝ}
variable (h1 : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ 0)
variable (h2 : (a^2 + b^2 + c^2 + d^2) / (a + b + c + d)^2 = 3 / 8)

-- The statement to prove the maximum value of the given expression
theorem max_ratio_a_c_over_b_d : ∃ t : ℝ, t = (a + c) / (b + d) ∧ t ≤ 3 :=
by {
  -- The proof of this theorem is omitted.
  sorry
}

end max_ratio_a_c_over_b_d_l2156_215601


namespace translate_line_up_l2156_215636

theorem translate_line_up (x y : ℝ) (h : y = 2 * x - 3) : y + 6 = 2 * x + 3 :=
by sorry

end translate_line_up_l2156_215636


namespace minimum_distance_proof_l2156_215609

noncomputable def minimum_distance_AB : ℝ :=
  let f (x : ℝ) := x^2 - Real.log x
  let x_min := Real.sqrt 2 / 2
  let min_dist := (5 + Real.log 2) / 4
  min_dist

theorem minimum_distance_proof :
  ∃ a : ℝ, a = minimum_distance_AB :=
by
  use (5 + Real.log 2) / 4
  sorry

end minimum_distance_proof_l2156_215609


namespace icosahedron_minimal_rotation_l2156_215666

structure Icosahedron :=
  (faces : ℕ)
  (is_regular : Prop)
  (face_shape : Prop)

def icosahedron := Icosahedron.mk 20 (by sorry) (by sorry)

def theta (θ : ℝ) : Prop :=
  ∃ θ > 0, ∀ h : Icosahedron, 
  h.faces = 20 ∧ h.is_regular ∧ h.face_shape → θ = 72

theorem icosahedron_minimal_rotation :
  ∃ θ > 0, ∀ h : Icosahedron,
  h.faces = 20 ∧ h.is_regular ∧ h.face_shape → θ = 72 :=
by sorry

end icosahedron_minimal_rotation_l2156_215666


namespace geometric_sequence_min_value_l2156_215681

theorem geometric_sequence_min_value 
  (a b c : ℝ)
  (h1 : b^2 = ac)
  (h2 : b = -Real.exp 1) :
  ac = Real.exp 2 := 
by
  sorry

end geometric_sequence_min_value_l2156_215681


namespace expression_value_l2156_215606

def a : ℕ := 1000
def b1 : ℕ := 15
def b2 : ℕ := 314
def c1 : ℕ := 201
def c2 : ℕ := 360
def c3 : ℕ := 110
def d1 : ℕ := 201
def d2 : ℕ := 360
def d3 : ℕ := 110
def e1 : ℕ := 15
def e2 : ℕ := 314

theorem expression_value :
  (a + b1 + b2) * (c1 + c2 + c3) + (a - d1 - d2 - d3) * (e1 + e2) = 1000000 :=
by
  sorry

end expression_value_l2156_215606


namespace triangle_problem_l2156_215646

-- Define a triangle with given parameters and properties
variables {A B C : ℝ}
variables {a b c : ℝ} (h_tri : b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C) 
variables (h_b2a : b = 2 * a)
variables (h_area : 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3)

-- Prove the required angles and side length
theorem triangle_problem 
    (h_tri : b * Real.cos A + a * Real.cos B = -2 * c * Real.cos C)
    (h_b2a : b = 2 * a)
    (h_area : 1/2 * a * b * Real.sin C = 2 * Real.sqrt 3) :

    Real.cos C = -1/2 ∧ C = 2 * Real.pi / 3 ∧ c = 2 * Real.sqrt 7 := 
by 
  sorry

end triangle_problem_l2156_215646


namespace katya_age_l2156_215674

theorem katya_age (A K V : ℕ) (h1 : A + K = 19) (h2 : A + V = 14) (h3 : K + V = 7) : K = 6 := by
  sorry

end katya_age_l2156_215674


namespace range_of_a_l2156_215624

theorem range_of_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1^2 * Real.exp (-x1) = a) 
    ∧ (x2^2 * Real.exp (-x2) = a) ∧ (x3^2 * Real.exp (-x3) = a)) ↔ (0 < a ∧ a < 4 * Real.exp (-2)) :=
sorry

end range_of_a_l2156_215624


namespace find_coefficient_b_l2156_215644

noncomputable def polynomial_f (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem find_coefficient_b 
  (a b c d : ℝ)
  (h1 : polynomial_f a b c d (-2) = 0)
  (h2 : polynomial_f a b c d 0 = 0)
  (h3 : polynomial_f a b c d 2 = 0)
  (h4 : polynomial_f a b c d (-1) = 3) :
  b = 0 :=
sorry

end find_coefficient_b_l2156_215644


namespace molecular_weight_AlOH3_l2156_215680

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

theorem molecular_weight_AlOH3 :
  (atomic_weight_Al + 3 * atomic_weight_O + 3 * atomic_weight_H) = 78.01 :=
by
  sorry

end molecular_weight_AlOH3_l2156_215680


namespace field_day_difference_l2156_215698

theorem field_day_difference :
  let girls_class_4_1 := 12
  let boys_class_4_1 := 13
  let girls_class_4_2 := 15
  let boys_class_4_2 := 11
  let girls_class_5_1 := 9
  let boys_class_5_1 := 13
  let girls_class_5_2 := 10
  let boys_class_5_2 := 11
  let total_girls := girls_class_4_1 + girls_class_4_2 + girls_class_5_1 + girls_class_5_2
  let total_boys := boys_class_4_1 + boys_class_4_2 + boys_class_5_1 + boys_class_5_2
  total_boys - total_girls = 2 := by
  sorry

end field_day_difference_l2156_215698


namespace find_g2_l2156_215650

variable (g : ℝ → ℝ)

def condition (x : ℝ) : Prop :=
  g x - 2 * g (1 / x) = 3^x

theorem find_g2 (h : ∀ x ≠ 0, condition g x) : g 2 = -3 - (4 * Real.sqrt 3) / 9 :=
  sorry

end find_g2_l2156_215650


namespace train_speed_late_l2156_215602

theorem train_speed_late (v : ℝ) 
  (h1 : ∀ (d : ℝ) (s : ℝ), d = 15 ∧ s = 100 → d / s = 0.15) 
  (h2 : ∀ (t1 t2 : ℝ), t1 = 0.15 ∧ t2 = 0.4 → t2 = t1 + 0.25)
  (h3 : ∀ (d : ℝ) (t : ℝ), d = 15 ∧ t = 0.4 → v = d / t) : 
  v = 37.5 := sorry

end train_speed_late_l2156_215602


namespace exists_divisor_c_of_f_l2156_215661

theorem exists_divisor_c_of_f (f : ℕ → ℕ) 
  (h₁ : ∀ n, f n ≥ 2)
  (h₂ : ∀ m n, f (m + n) ∣ (f m + f n)) :
  ∃ c > 1, ∀ n, c ∣ f n :=
sorry

end exists_divisor_c_of_f_l2156_215661


namespace rohit_distance_from_start_l2156_215659

noncomputable def rohit_final_position : ℕ × ℕ :=
  let start := (0, 0)
  let p1 := (start.1, start.2 - 25)       -- Moves 25 meters south.
  let p2 := (p1.1 + 20, p1.2)           -- Turns left (east) and moves 20 meters.
  let p3 := (p2.1, p2.2 + 25)           -- Turns left (north) and moves 25 meters.
  let result := (p3.1 + 15, p3.2)       -- Turns right (east) and moves 15 meters.
  result

theorem rohit_distance_from_start :
  rohit_final_position = (35, 0) :=
sorry

end rohit_distance_from_start_l2156_215659


namespace find_parameters_l2156_215622

noncomputable def cubic_function (a b : ℝ) (x : ℝ) : ℝ :=
  x^3 + a * x^2 + b * x + 27

def deriv_cubic_function (a b : ℝ) (x : ℝ) : ℝ :=
  3 * x^2 + 2 * a * x + b

theorem find_parameters
  (a b : ℝ)
  (h1 : deriv_cubic_function a b (-1) = 0)
  (h2 : deriv_cubic_function a b 3 = 0) :
  a = -3 ∧ b = -9 :=
by
  -- leaving proof as sorry since the task doesn't require proving
  sorry

end find_parameters_l2156_215622


namespace smallest_k_for_perfect_cube_l2156_215634

noncomputable def isPerfectCube (m : ℕ) : Prop :=
  ∃ n : ℤ, n^3 = m

theorem smallest_k_for_perfect_cube :
  ∃ k : ℕ, k > 0 ∧ (∀ m : ℕ, ((2^4) * (3^2) * (5^5) * k = m) → isPerfectCube m) ∧ k = 60 :=
sorry

end smallest_k_for_perfect_cube_l2156_215634


namespace usual_time_to_reach_school_l2156_215687

theorem usual_time_to_reach_school
  (R T : ℝ)
  (h1 : (7 / 6) * R = R / (T - 3) * T) : T = 21 :=
sorry

end usual_time_to_reach_school_l2156_215687


namespace fifth_equation_in_pattern_l2156_215683

theorem fifth_equation_in_pattern :
  (1 - 4 + 9 - 16 + 25) = (1 + 2 + 3 + 4 + 5) :=
sorry

end fifth_equation_in_pattern_l2156_215683


namespace total_spent_l2156_215671

/-- Define the prices of the rides in the morning and the afternoon --/
def morning_price (ride : String) (age : Nat) : Nat :=
  match ride, age with
  | "bumper_car", n => if n < 18 then 2 else 3
  | "space_shuttle", n => if n < 18 then 4 else 5
  | "ferris_wheel", n => if n < 18 then 5 else 6
  | _, _ => 0

def afternoon_price (ride : String) (age : Nat) : Nat :=
  (morning_price ride age) + 1

/-- Define the number of rides taken by Mara and Riley --/
def rides_morning (person : String) (ride : String) : Nat :=
  match person, ride with
  | "Mara", "bumper_car" => 1
  | "Mara", "ferris_wheel" => 2
  | "Riley", "space_shuttle" => 2
  | "Riley", "ferris_wheel" => 2
  | _, _ => 0

def rides_afternoon (person : String) (ride : String) : Nat :=
  match person, ride with
  | "Mara", "bumper_car" => 1
  | "Mara", "ferris_wheel" => 1
  | "Riley", "space_shuttle" => 2
  | "Riley", "ferris_wheel" => 1
  | _, _ => 0

/-- Define the ages of Mara and Riley --/
def age (person : String) : Nat :=
  match person with
  | "Mara" => 17
  | "Riley" => 19
  | _ => 0

/-- Calculate the total expenditure --/
def total_cost (person : String) : Nat :=
  List.sum ([
    (rides_morning person "bumper_car") * (morning_price "bumper_car" (age person)),
    (rides_afternoon person "bumper_car") * (afternoon_price "bumper_car" (age person)),
    (rides_morning person "space_shuttle") * (morning_price "space_shuttle" (age person)),
    (rides_afternoon person "space_shuttle") * (afternoon_price "space_shuttle" (age person)),
    (rides_morning person "ferris_wheel") * (morning_price "ferris_wheel" (age person)),
    (rides_afternoon person "ferris_wheel") * (afternoon_price "ferris_wheel" (age person))
  ])

/-- Prove the total cost for Mara and Riley is $62 --/
theorem total_spent : total_cost "Mara" + total_cost "Riley" = 62 :=
by
  sorry

end total_spent_l2156_215671


namespace correct_subtraction_l2156_215617

theorem correct_subtraction (x : ℕ) (h : x - 63 = 8) : x - 36 = 35 :=
by sorry

end correct_subtraction_l2156_215617


namespace fraction_sent_afternoon_l2156_215688

theorem fraction_sent_afternoon :
  ∀ (total_fliers morning_fraction fliers_left_next_day : ℕ),
  total_fliers = 3000 →
  morning_fraction = 1/5 →
  fliers_left_next_day = 1800 →
  ((total_fliers - total_fliers * morning_fraction) - fliers_left_next_day) / (total_fliers - total_fliers * morning_fraction) = 1/4 :=
by
  intros total_fliers morning_fraction fliers_left_next_day h1 h2 h3
  sorry

end fraction_sent_afternoon_l2156_215688


namespace ellipse_equation_max_area_abcd_l2156_215638

open Real

theorem ellipse_equation (x y : ℝ) (a b c : ℝ) (h₀ : a > b) (h₁ : b > 0)
  (h₂ : a^2 = b^2 + c^2) (h₃ : b * c = 1) (h₄ : b = c) :
  (x^2 / 2 + y^2 = 1) ↔ (x^2 / a^2 + y^2 / b^2 = 1) := by
  sorry

theorem max_area_abcd (a b c t : ℝ) (h₀ : a > b) (h₁ : b > 0)
  (h₂ : a^2 = b^2 + c^2) (h₃ : b * c = 1) (h₄ : b = c) :
  (∀ (t : ℝ), 4 * sqrt 2 * sqrt (1 + t^2) / (t^2 + 2) ≤ 2 * sqrt 2) := by
  sorry

end ellipse_equation_max_area_abcd_l2156_215638


namespace gcd_of_B_is_2_l2156_215618

-- Condition: B is the set of all numbers which can be represented as the sum of four consecutive positive integers
def B := { n : ℕ | ∃ y : ℕ, n = (y - 1) + y + (y + 1) + (y + 2) }

-- Question: What is the greatest common divisor of all numbers in \( B \)
-- Mathematical equivalent proof problem: Prove gcd of all elements in set \( B \) is 2

theorem gcd_of_B_is_2 : ∀ n ∈ B, ∃ y : ℕ, n = 2 * (2 * y + 1) → ∀ m ∈ B, n.gcd m = 2 :=
by
  sorry

end gcd_of_B_is_2_l2156_215618


namespace inscribed_square_ratio_l2156_215697

-- Define the problem context:
variables {x y : ℝ}

-- Conditions on the triangles and squares:
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ a > 0 ∧ b > 0 ∧ c > 0

def inscribed_square_first_triangle (a b c x : ℝ) : Prop :=
  is_right_triangle a b c ∧ a = 5 ∧ b = 12 ∧ c = 13 ∧
  x = 60 / 17

def inscribed_square_second_triangle (d e f y : ℝ) : Prop :=
  is_right_triangle d e f ∧ d = 6 ∧ e = 8 ∧ f = 10 ∧
  y = 25 / 8

-- Lean theorem to be proven with given conditions:
theorem inscribed_square_ratio :
  inscribed_square_first_triangle 5 12 13 x →
  inscribed_square_second_triangle 6 8 10 y →
  x / y = 96 / 85 := by
  sorry

end inscribed_square_ratio_l2156_215697


namespace cars_no_air_conditioning_l2156_215616

variables {A R AR : Nat}

/-- Given a total of 100 cars, of which at least 51 have racing stripes,
and the greatest number of cars that could have air conditioning but not racing stripes is 49,
prove that the number of cars that do not have air conditioning is 49. -/
theorem cars_no_air_conditioning :
  ∀ (A R AR : ℕ), 
  (A = AR + 49) → 
  (R ≥ 51) → 
  (AR ≤ R) → 
  (AR ≤ 51) → 
  (100 - A = 49) :=
by
  intros A R AR h1 h2 h3 h4
  exact sorry

end cars_no_air_conditioning_l2156_215616


namespace base3_sum_example_l2156_215662

noncomputable def base3_add (a b : ℕ) : ℕ := sorry  -- Function to perform base-3 addition

theorem base3_sum_example : 
  base3_add (base3_add (base3_add (base3_add 2 120) 221) 1112) 1022 = 21201 := sorry

end base3_sum_example_l2156_215662


namespace ratio_of_votes_l2156_215629

theorem ratio_of_votes (votes_A votes_B total_votes : ℕ) (hA : votes_A = 14) (hTotal : votes_A + votes_B = 21) : votes_A / Nat.gcd votes_A votes_B = 2 ∧ votes_B / Nat.gcd votes_A votes_B = 1 := 
by
  sorry

end ratio_of_votes_l2156_215629


namespace farmer_kent_income_l2156_215603

-- Define the constants and conditions
def watermelon_weight : ℕ := 23
def price_per_pound : ℕ := 2
def number_of_watermelons : ℕ := 18

-- Construct the proof statement
theorem farmer_kent_income : 
  price_per_pound * watermelon_weight * number_of_watermelons = 828 := 
by
  -- Skipping the proof here, just stating the theorem.
  sorry

end farmer_kent_income_l2156_215603


namespace sqrt_meaningful_iff_l2156_215669

theorem sqrt_meaningful_iff (x: ℝ) : (6 - 2 * x ≥ 0) ↔ (x ≤ 3) :=
by
  sorry

end sqrt_meaningful_iff_l2156_215669


namespace unit_digit_of_power_of_two_l2156_215633

theorem unit_digit_of_power_of_two (n : ℕ) :
  (2 ^ 2023) % 10 = 8 := 
by
  sorry

end unit_digit_of_power_of_two_l2156_215633


namespace factorize_expression_simplify_fraction_expr_l2156_215686

-- (1) Prove the factorization of m^3 - 4m^2 + 4m
theorem factorize_expression (m : ℝ) : 
  m^3 - 4 * m^2 + 4 * m = m * (m - 2)^2 :=
by
  sorry

-- (2) Simplify the fraction operation correctly
theorem simplify_fraction_expr (x : ℝ) (h : x ≠ 1) : 
  2 / (x^2 - 1) - 1 / (x - 1) = -1 / (x + 1) :=
by
  sorry

end factorize_expression_simplify_fraction_expr_l2156_215686


namespace nonnegative_solution_positive_solution_l2156_215648

/-- For k > 7, there exist non-negative integers x and y such that 5*x + 3*y = k. -/
theorem nonnegative_solution (k : ℤ) (hk : k > 7) : ∃ x y : ℕ, 5 * x + 3 * y = k :=
sorry

/-- For k > 15, there exist positive integers x and y such that 5*x + 3*y = k. -/
theorem positive_solution (k : ℤ) (hk : k > 15) : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ 5 * x + 3 * y = k :=
sorry

end nonnegative_solution_positive_solution_l2156_215648


namespace exists_monotonicity_b_range_l2156_215655

noncomputable def f (a x : ℝ) : ℝ := (1 / 2) * a * x ^ 2 - 2 * a * x + Real.log x

theorem exists_monotonicity_b_range :
  ∀ (a : ℝ) (b : ℝ), 1 < a ∧ a < 2 →
  (∀ (x0 : ℝ), x0 ∈ Set.Icc (1 + Real.sqrt 2 / 2) 2 →
   f a x0 + Real.log (a + 1) > b * (a^2 - 1) - (a + 1) + 2 * Real.log 2) →
   b ∈ Set.Iic (-1/4) :=
sorry

end exists_monotonicity_b_range_l2156_215655


namespace find_t_over_q_l2156_215689

theorem find_t_over_q
  (q r s v t : ℝ)
  (h1 : q / r = 12)
  (h2 : s / r = 8)
  (h3 : v / t = 4)
  (h4 : s / v = 1 / 3) :
  t / q = 1 / 2 := 
sorry

end find_t_over_q_l2156_215689


namespace inequality_holds_for_k_2_l2156_215640

theorem inequality_holds_for_k_2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  a * b + b * c + c * a + 2 * (1 / a + 1 / b + 1 / c) ≥ 9 := 
by 
  sorry

end inequality_holds_for_k_2_l2156_215640


namespace rosy_current_age_l2156_215608

theorem rosy_current_age 
  (R : ℕ) 
  (h1 : ∀ (david_age rosy_age : ℕ), david_age = rosy_age + 12) 
  (h2 : ∀ (david_age_plus_4 rosy_age_plus_4 : ℕ), david_age_plus_4 = 2 * rosy_age_plus_4) : 
  R = 8 := 
sorry

end rosy_current_age_l2156_215608


namespace segment_length_calc_l2156_215635

noncomputable def segment_length_parallel_to_side
  (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) : ℝ :=
  a * (b + c) / (a + b + c)

theorem segment_length_calc
  (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) :
  segment_length_parallel_to_side a b c a_pos b_pos c_pos = a * (b + c) / (a + b + c) :=
sorry

end segment_length_calc_l2156_215635


namespace sum_of_roots_of_cubic_l2156_215675

noncomputable def P (a b c d x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem sum_of_roots_of_cubic (a b c d : ℝ) (h : ∀ x : ℝ, P a b c d (x^2 + x) ≥ P a b c d (x + 1)) :
  (-b / a) = (P a b c d 0) :=
sorry

end sum_of_roots_of_cubic_l2156_215675


namespace jasmine_first_exceed_500_l2156_215658

theorem jasmine_first_exceed_500 {k : ℕ} (initial : ℕ) (factor : ℕ) :
  initial = 5 → factor = 4 → (5 * 4^k > 500) → k = 4 :=
by
  sorry

end jasmine_first_exceed_500_l2156_215658


namespace monomial_exponent_match_l2156_215607

theorem monomial_exponent_match (m : ℤ) (x y : ℂ) : (-x^(2*m) * y^3 = 2 * x^6 * y^3) → m = 3 := 
by 
  sorry

end monomial_exponent_match_l2156_215607


namespace max_numbers_with_240_product_square_l2156_215673

theorem max_numbers_with_240_product_square :
  ∃ (S : Finset ℕ), S.card = 11 ∧ ∀ k ∈ S, 1 ≤ k ∧ k ≤ 2015 ∧ ∃ n m, 240 * k = (n * m) ^ 2 :=
sorry

end max_numbers_with_240_product_square_l2156_215673


namespace MrsHiltCanTakeFriendsToMovies_l2156_215654

def TotalFriends : ℕ := 15
def FriendsCantGo : ℕ := 7
def FriendsCanGo : ℕ := 8

theorem MrsHiltCanTakeFriendsToMovies : TotalFriends - FriendsCantGo = FriendsCanGo := by
  -- The proof will show that 15 - 7 = 8.
  sorry

end MrsHiltCanTakeFriendsToMovies_l2156_215654


namespace problem_l2156_215649

def f (x : ℝ) (a b c d : ℝ) : ℝ := a * x^7 + b * x^5 - c * x^3 + d * x + 3

theorem problem (a b c d : ℝ) (h : f 92 a b c d = 2) : f 92 a b c d + f (-92) a b c d = 6 :=
by
  sorry

end problem_l2156_215649


namespace product_area_perimeter_eq_104sqrt26_l2156_215611

noncomputable def distance (a b : ℝ × ℝ) : ℝ :=
  ((b.1 - a.1) ^ 2 + (b.2 - a.2) ^ 2).sqrt

noncomputable def side_length := distance (5, 5) (0, 4)

noncomputable def area_of_square := side_length ^ 2

noncomputable def perimeter_of_square := 4 * side_length

noncomputable def product_area_perimeter := area_of_square * perimeter_of_square

theorem product_area_perimeter_eq_104sqrt26 :
  product_area_perimeter = 104 * Real.sqrt 26 :=
by 
  -- placeholder for the proof
  sorry

end product_area_perimeter_eq_104sqrt26_l2156_215611


namespace find_David_marks_in_Physics_l2156_215637

theorem find_David_marks_in_Physics
  (english_marks : ℕ) (math_marks : ℕ) (chem_marks : ℕ) (biology_marks : ℕ)
  (avg_marks : ℕ) (num_subjects : ℕ)
  (h_english : english_marks = 76)
  (h_math : math_marks = 65)
  (h_chem : chem_marks = 67)
  (h_bio : biology_marks = 85)
  (h_avg : avg_marks = 75) 
  (h_num_subjects : num_subjects = 5) :
  english_marks + math_marks + chem_marks + biology_marks + physics_marks = avg_marks * num_subjects → physics_marks = 82 := 
  sorry

end find_David_marks_in_Physics_l2156_215637


namespace age_relation_l2156_215668

theorem age_relation (S M D Y : ℝ)
  (h1 : M = S + 37)
  (h2 : M + 2 = 2 * (S + 2))
  (h3 : D = S - 4)
  (h4 : M + Y = 3 * (D + Y))
  : Y = -10.5 :=
by
  sorry

end age_relation_l2156_215668


namespace esteban_exercise_days_l2156_215639

theorem esteban_exercise_days
  (natasha_exercise_per_day : ℕ)
  (natasha_days : ℕ)
  (esteban_exercise_per_day : ℕ)
  (total_exercise_hours : ℕ)
  (hours_to_minutes : ℕ)
  (natasha_exercise_total : ℕ)
  (total_exercise_minutes : ℕ)
  (esteban_exercise_total : ℕ)
  (esteban_days : ℕ) :
  natasha_exercise_per_day = 30 →
  natasha_days = 7 →
  esteban_exercise_per_day = 10 →
  total_exercise_hours = 5 →
  hours_to_minutes = 60 →
  natasha_exercise_total = natasha_exercise_per_day * natasha_days →
  total_exercise_minutes = total_exercise_hours * hours_to_minutes →
  esteban_exercise_total = total_exercise_minutes - natasha_exercise_total →
  esteban_days = esteban_exercise_total / esteban_exercise_per_day →
  esteban_days = 9 :=
by
  sorry

end esteban_exercise_days_l2156_215639


namespace hyperbola_asymptotes_l2156_215645

-- Define the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := y^2 - (x^2 / 4) = 1

-- The statement to prove: The equation of the asymptotes of the hyperbola is as follows
theorem hyperbola_asymptotes :
  (∀ x y : ℝ, hyperbola_eq x y → (y = (1/2) * x ∨ y = -(1/2) * x)) :=
sorry

end hyperbola_asymptotes_l2156_215645


namespace second_bag_roger_is_3_l2156_215670

def total_candy_sandra := 2 * 6
def total_candy_roger := total_candy_sandra + 2
def first_bag_roger := 11
def second_bag_roger := total_candy_roger - first_bag_roger

theorem second_bag_roger_is_3 : second_bag_roger = 3 :=
by
  sorry

end second_bag_roger_is_3_l2156_215670


namespace price_of_other_frisbees_l2156_215663

theorem price_of_other_frisbees 
  (P : ℝ) 
  (x : ℝ)
  (h1 : x + (64 - x) = 64)
  (h2 : P * x + 4 * (64 - x) = 196)
  (h3 : 64 - x ≥ 4) 
  : P = 3 :=
sorry

end price_of_other_frisbees_l2156_215663


namespace range_of_m_l2156_215672

-- Defining the point P and the required conditions for it to lie in the fourth quadrant
def point_in_fourth_quadrant (m : ℝ) : Prop :=
  let P := (m + 3, m - 1)
  P.1 > 0 ∧ P.2 < 0

-- Defining the range of m for which the point lies in the fourth quadrant
theorem range_of_m (m : ℝ) : point_in_fourth_quadrant m ↔ (-3 < m ∧ m < 1) :=
by
  sorry

end range_of_m_l2156_215672


namespace base_conversion_subtraction_l2156_215619

def base8_to_base10 : Nat := 5 * 8^5 + 4 * 8^4 + 3 * 8^3 + 2 * 8^2 + 1 * 8^1 + 0 * 8^0
def base9_to_base10 : Nat := 6 * 9^4 + 5 * 9^3 + 4 * 9^2 + 3 * 9^1 + 2 * 9^0

theorem base_conversion_subtraction :
  base8_to_base10 - base9_to_base10 = 136532 :=
by
  -- Proof steps go here
  sorry

end base_conversion_subtraction_l2156_215619


namespace smallest_bob_number_l2156_215653

theorem smallest_bob_number (b : ℕ) (h : ∀ p : ℕ, Prime p → p ∣ 30 → p ∣ b) : 30 ≤ b :=
by {
  sorry
}

end smallest_bob_number_l2156_215653


namespace mail_in_six_months_l2156_215613

/-- The post office receives 60 letters and 20 packages per day. Each month has 30 days. -/
def daily_letters := 60
def daily_packages := 20
def days_per_month := 30
def months := 6

/-- Prove that the post office handles 14400 pieces of mail in six months. -/
theorem mail_in_six_months : (daily_letters + daily_packages) * days_per_month * months = 14400 :=
by
  sorry

end mail_in_six_months_l2156_215613


namespace value_of_expression_l2156_215643

theorem value_of_expression (n : ℝ) (h : n + 1/n = 10) : n^2 + (1/n^2) + 6 = 104 :=
by
  sorry

end value_of_expression_l2156_215643


namespace find_x_angle_l2156_215642

theorem find_x_angle (ABC ACB CDE : ℝ) (h1 : ABC = 70) (h2 : ACB = 90) (h3 : CDE = 42) : 
  ∃ x : ℝ, x = 158 :=
by
  sorry

end find_x_angle_l2156_215642


namespace rectangle_decomposition_l2156_215693

theorem rectangle_decomposition (m n k : ℕ) : ((k ∣ m) ∨ (k ∣ n)) ↔ (∃ P : ℕ, m * n = P * k) :=
by
  sorry

end rectangle_decomposition_l2156_215693


namespace wrongly_read_number_l2156_215627

theorem wrongly_read_number 
  (S_initial : ℕ) (S_correct : ℕ) (correct_num : ℕ) (num_count : ℕ) 
  (h_initial : S_initial = num_count * 18) 
  (h_correct : S_correct = num_count * 19) 
  (h_correct_num : correct_num = 36) 
  (h_diff : S_correct - S_initial = correct_num - wrong_num) 
  (h_num_count : num_count = 10) 
  : wrong_num = 26 :=
sorry

end wrongly_read_number_l2156_215627


namespace Bill_trips_l2156_215631

theorem Bill_trips (total_trips : ℕ) (Jean_trips : ℕ) (Bill_trips : ℕ) 
  (h1 : total_trips = 40) 
  (h2 : Jean_trips = 23) 
  (h3 : Bill_trips + Jean_trips = total_trips) : 
  Bill_trips = 17 := 
by
  sorry

end Bill_trips_l2156_215631


namespace calculate_star_difference_l2156_215685

def star (a b : ℕ) : ℕ := a^2 + 2 * a * b + b^2

theorem calculate_star_difference : (star 3 5) - (star 2 4) = 28 := by
  sorry

end calculate_star_difference_l2156_215685


namespace no_solution_eqn_l2156_215612

theorem no_solution_eqn (m : ℝ) : (∀ x : ℝ, (m * (x + 1) - 5) / (2 * x + 1) ≠ m - 3) ↔ m = 6 := 
by
  sorry

end no_solution_eqn_l2156_215612


namespace paco_salty_cookies_left_l2156_215684

theorem paco_salty_cookies_left (S₁ S₂ : ℕ) (h₁ : S₁ = 6) (e1_eaten : ℕ) (a₁ : e1_eaten = 3)
(h₂ : S₂ = 24) (r1_ratio : ℚ) (a_ratio : r1_ratio = (2/3)) :
  S₁ - e1_eaten + r1_ratio * S₂ = 19 :=
by
  sorry

end paco_salty_cookies_left_l2156_215684


namespace greatest_of_3_consecutive_integers_l2156_215696

theorem greatest_of_3_consecutive_integers (x : ℤ) (h : x + (x + 1) + (x + 2) = 24) : (x + 2) = 9 :=
by
-- Proof would go here.
sorry

end greatest_of_3_consecutive_integers_l2156_215696


namespace percent_increase_visual_range_l2156_215691

theorem percent_increase_visual_range (original new : ℝ) (h_original : original = 60) (h_new : new = 150) : 
  ((new - original) / original) * 100 = 150 :=
by
  sorry

end percent_increase_visual_range_l2156_215691


namespace min_value_of_x2_plus_y2_l2156_215621

-- Define the problem statement
theorem min_value_of_x2_plus_y2 (x y : ℝ) (h : 3 * x + y = 10) : x^2 + y^2 ≥ 10 :=
sorry

end min_value_of_x2_plus_y2_l2156_215621


namespace balance_difference_l2156_215677

def compound_interest (P r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r)^n

def simple_interest (P r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r * t)

theorem balance_difference :
  let angela_balance := compound_interest 12000 0.05 15
  let bob_balance := simple_interest 15000 0.06 15
  let diff := abs (bob_balance - angela_balance)
  (round diff = 3553) :=
by 
  let angela_balance := compound_interest 12000 0.05 15
  let bob_balance := simple_interest 15000 0.06 15
  let diff := abs (bob_balance - angela_balance)
  have h : round diff = 3553 := sorry
  assumption

end balance_difference_l2156_215677


namespace find_k_l2156_215630

theorem find_k (k : ℝ) : 4 + ∑' (n : ℕ), (4 + n * k) / 5^n = 10 → k = 16 := by
  sorry

end find_k_l2156_215630


namespace value_of_f_at_2_l2156_215678

-- Given the conditions
variable (f : ℝ → ℝ)
variable (h_mono : Monotone f)
variable (h_cond : ∀ x : ℝ, f (f x - 3^x) = 4)

-- Define the proof goal
theorem value_of_f_at_2 : f 2 = 10 := 
sorry

end value_of_f_at_2_l2156_215678


namespace max_n_sum_pos_largest_term_seq_l2156_215628

-- Define the arithmetic sequence {a_n} and sum of first n terms S_n along with given conditions
def arithmetic_seq (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ := a_1 + (n - 1) * d
def sum_arith_seq (a_1 : ℤ) (d : ℤ) (n : ℤ) : ℤ := n * (2 * a_1 + (n - 1) * d) / 2

variable (a_1 d : ℤ)
-- Conditions from problem
axiom a8_pos : arithmetic_seq a_1 d 8 > 0
axiom a8_a9_neg : arithmetic_seq a_1 d 8 + arithmetic_seq a_1 d 9 < 0

-- Prove the maximum n for which Sum S_n > 0 is 15
theorem max_n_sum_pos : ∃ n_max : ℤ, sum_arith_seq a_1 d n_max > 0 ∧ 
  ∀ n : ℤ, n > n_max → sum_arith_seq a_1 d n ≤ 0 := by
    exact ⟨15, sorry⟩  -- Substitute 'sorry' for the proof part

-- Determine the largest term in the sequence {S_n / a_n} for 1 ≤ n ≤ 15
theorem largest_term_seq : ∃ n_largest : ℤ, ∀ n : ℤ, 1 ≤ n → n ≤ 15 → 
  (sum_arith_seq a_1 d n / arithmetic_seq a_1 d n) ≤ (sum_arith_seq a_1 d n_largest / arithmetic_seq a_1 d n_largest) := by
    exact ⟨8, sorry⟩  -- Substitute 'sorry' for the proof part

end max_n_sum_pos_largest_term_seq_l2156_215628


namespace girls_at_start_l2156_215699

theorem girls_at_start (B G : ℕ) (h1 : B + G = 600) (h2 : 6 * B + 7 * G = 3840) : G = 240 :=
by
  -- actual proof is omitted
  sorry

end girls_at_start_l2156_215699


namespace ivan_chess_false_l2156_215676

theorem ivan_chess_false (n : ℕ) :
  ∃ n, n + 3 * n + 6 * n = 64 → False :=
by
  use 6
  sorry

end ivan_chess_false_l2156_215676


namespace xy_diff_square_l2156_215623

theorem xy_diff_square (x y : ℝ) (h1 : x + y = -5) (h2 : x * y = 6) : (x - y)^2 = 1 :=
by
  sorry

end xy_diff_square_l2156_215623


namespace hypotenuse_length_l2156_215604

-- Definitions and conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Hypotheses
def leg1 := 8
def leg2 := 15

-- The theorem to be proven
theorem hypotenuse_length : ∃ c : ℕ, is_right_triangle leg1 leg2 c ∧ c = 17 :=
by { sorry }

end hypotenuse_length_l2156_215604


namespace solve_fraction_equation_l2156_215656

theorem solve_fraction_equation :
  {x : ℝ | (1 / (x^2 + 13 * x - 12) + 1 / (x^2 + 4 * x - 12) + 1 / (x^2 - 15 * x - 12) = 0)} =
  {1, -12, 12, -1} :=
by
  sorry

end solve_fraction_equation_l2156_215656


namespace Jeremy_strolled_20_kilometers_l2156_215651

def speed : ℕ := 2 -- Jeremy's speed in kilometers per hour
def time : ℕ := 10 -- Time Jeremy strolled in hours

noncomputable def distance : ℕ := speed * time -- The computed distance

theorem Jeremy_strolled_20_kilometers : distance = 20 := by
  sorry

end Jeremy_strolled_20_kilometers_l2156_215651


namespace tree_boy_growth_ratio_l2156_215625

theorem tree_boy_growth_ratio 
    (initial_tree_height final_tree_height initial_boy_height final_boy_height : ℕ) 
    (h₀ : initial_tree_height = 16) 
    (h₁ : final_tree_height = 40) 
    (h₂ : initial_boy_height = 24) 
    (h₃ : final_boy_height = 36) 
:
  (final_tree_height - initial_tree_height) / (final_boy_height - initial_boy_height) = 2 := 
by {
    -- Definitions and given conditions used in the statement part of the proof
    sorry
}

end tree_boy_growth_ratio_l2156_215625


namespace initial_deadline_is_75_days_l2156_215660

-- Define constants for the problem
def initial_men : ℕ := 100
def initial_hours_per_day : ℕ := 8
def days_worked_initial : ℕ := 25
def fraction_work_completed : ℚ := 1 / 3
def additional_men : ℕ := 60
def new_hours_per_day : ℕ := 10
def total_man_hours : ℕ := 60000

-- Prove that the initial deadline for the project is 75 days
theorem initial_deadline_is_75_days : 
  ∃ (D : ℕ), (D * initial_men * initial_hours_per_day = total_man_hours) ∧ D = 75 := 
by {
  sorry
}

end initial_deadline_is_75_days_l2156_215660


namespace compound_interest_difference_l2156_215664

variable (P r : ℝ)

theorem compound_interest_difference :
  (P * 9 * r^2 = 360) → (P * r^2 = 40) :=
by
  sorry

end compound_interest_difference_l2156_215664


namespace domain_of_k_l2156_215615

noncomputable def k (x : ℝ) : ℝ := (1 / (x + 6)) + (1 / (x^2 + 2*x + 9)) + (1 / (x^3 - 27))

theorem domain_of_k : {x : ℝ | k x ≠ 0} = {x : ℝ | x ≠ -6 ∧ x ≠ 3} :=
by
  sorry

end domain_of_k_l2156_215615


namespace tan_theta_3_l2156_215667

noncomputable def tan_triple_angle (θ : ℝ) : ℝ := (3 * (Real.tan θ) - ((Real.tan θ) ^ 3)) / (1 - 3 * (Real.tan θ)^2)

theorem tan_theta_3 (θ : ℝ) (h : Real.tan θ = 3) : tan_triple_angle θ = 9 / 13 :=
by
  sorry

end tan_theta_3_l2156_215667


namespace f_at_3_l2156_215647

-- Define the function f and its conditions
variable (f : ℝ → ℝ)

-- The domain of the function f is ℝ, hence f : ℝ → ℝ
-- Also given:
axiom f_symm : ∀ x : ℝ, f (1 - x) = f (1 + x)
axiom f_add : f (-1) + f (3) = 12

-- Final proof statement
theorem f_at_3 : f 3 = 6 :=
by
  sorry

end f_at_3_l2156_215647


namespace surface_area_of_cube_l2156_215682

-- Definition of the problem in Lean 4
theorem surface_area_of_cube (a : ℝ) (s : ℝ) (h : s * Real.sqrt 3 = a) : 6 * (s^2) = 2 * a^2 :=
by
  sorry

end surface_area_of_cube_l2156_215682


namespace junior_score_proof_l2156_215694

noncomputable def class_total_score (total_students : ℕ) (average_class_score : ℕ) : ℕ :=
total_students * average_class_score

noncomputable def number_of_juniors (total_students : ℕ) (percent_juniors : ℕ) : ℕ :=
percent_juniors * total_students / 100

noncomputable def number_of_seniors (total_students juniors : ℕ) : ℕ :=
total_students - juniors

noncomputable def total_senior_score (seniors average_senior_score : ℕ) : ℕ :=
seniors * average_senior_score

noncomputable def total_junior_score (total_score senior_score : ℕ) : ℕ :=
total_score - senior_score

noncomputable def junior_score (junior_total_score juniors : ℕ) : ℕ :=
junior_total_score / juniors

theorem junior_score_proof :
  ∀ (total_students: ℕ) (percent_juniors average_class_score average_senior_score : ℕ),
  total_students = 20 →
  percent_juniors = 15 →
  average_class_score = 85 →
  average_senior_score = 84 →
  (junior_score (total_junior_score (class_total_score total_students average_class_score)
                                    (total_senior_score (number_of_seniors total_students (number_of_juniors total_students percent_juniors))
                                                        average_senior_score))
                (number_of_juniors total_students percent_juniors)) = 91 :=
by
  intros
  sorry

end junior_score_proof_l2156_215694


namespace log_comparison_l2156_215614

/-- Assuming a = log base 3 of 2, b = natural log of 3, and c = log base 2 of 3,
    prove that c > b > a. -/
theorem log_comparison (a b c : ℝ) (h1 : a = Real.log 2 / Real.log 3)
                                (h2 : b = Real.log 3)
                                (h3 : c = Real.log 3 / Real.log 2) :
  c > b ∧ b > a :=
by {
  sorry
}

end log_comparison_l2156_215614


namespace problem_statement_l2156_215641

theorem problem_statement (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a + b) ^ 2002 + a ^ 2001 = 2 := 
by 
  sorry

end problem_statement_l2156_215641


namespace choir_blonde_black_ratio_l2156_215632

theorem choir_blonde_black_ratio 
  (b x : ℕ) 
  (h1 : ∀ (b x : ℕ), b / ((5 / 3 : ℚ) * b) = (3 / 5 : ℚ)) 
  (h2 : ∀ (b x : ℕ), (b + x) / ((5 / 3 : ℚ) * b) = (3 / 2 : ℚ)) :
  x = (3 / 2 : ℚ) * b ∧ 
  ∃ k : ℚ, k = (5 / 3 : ℚ) * b :=
by {
  sorry
}

end choir_blonde_black_ratio_l2156_215632


namespace constant_c_square_of_binomial_l2156_215657

theorem constant_c_square_of_binomial (c : ℝ) (h : ∃ d : ℝ, (3*x + d)^2 = 9*x^2 - 18*x + c) : c = 9 :=
sorry

end constant_c_square_of_binomial_l2156_215657


namespace calculation_correct_l2156_215652

theorem calculation_correct : 200 * 19.9 * 1.99 * 100 = 791620 := by
  sorry

end calculation_correct_l2156_215652


namespace ratio_of_A_to_B_l2156_215690

theorem ratio_of_A_to_B (A B C : ℝ) (h1 : A + B + C = 544) (h2 : B = (1/4) * C) (hA : A = 64) (hB : B = 96) (hC : C = 384) : A / B = 2 / 3 :=
by 
  sorry

end ratio_of_A_to_B_l2156_215690


namespace triangle_base_is_8_l2156_215610

/- Problem Statement:
We have a square with a perimeter of 48 and a triangle with a height of 36.
We need to prove that if both the square and the triangle have the same area, then the base of the triangle (x) is 8.
-/

theorem triangle_base_is_8
  (square_perimeter : ℝ)
  (triangle_height : ℝ)
  (same_area : ℝ) :
  square_perimeter = 48 →
  triangle_height = 36 →
  same_area = (square_perimeter / 4) ^ 2 →
  same_area = (1 / 2) * x * triangle_height →
  x = 8 :=
by
  sorry

end triangle_base_is_8_l2156_215610


namespace derive_units_equivalent_to_velocity_l2156_215626

-- Define the unit simplifications
def watt := 1 * (1 * (1 * (1 / 1)))
def newton := 1 * (1 * (1 / (1 * 1)))

-- Define the options
def option_A := watt / newton
def option_B := newton / watt
def option_C := watt / (newton * newton)
def option_D := (watt * watt) / newton
def option_E := (newton * newton) / (watt * watt)

-- Define what it means for a unit to be equivalent to velocity
def is_velocity (unit : ℚ) : Prop := unit = (1 * (1 / 1))

theorem derive_units_equivalent_to_velocity :
  is_velocity option_A ∧ 
  ¬ is_velocity option_B ∧ 
  ¬ is_velocity option_C ∧ 
  ¬ is_velocity option_D ∧ 
  ¬ is_velocity option_E := 
by sorry

end derive_units_equivalent_to_velocity_l2156_215626


namespace two_digit_numbers_of_form_3_pow_n_l2156_215620

theorem two_digit_numbers_of_form_3_pow_n :
  ∃ (n1 n2 : ℕ), (10 ≤ 3^n1 ∧ 3^n1 ≤ 99) ∧ (10 ≤ 3^n2 ∧ 3^n2 ≤ 99) ∧ n2 - n1 + 1 = 2 :=
by
  sorry

end two_digit_numbers_of_form_3_pow_n_l2156_215620


namespace sock_ratio_l2156_215679

theorem sock_ratio (b : ℕ) (x : ℕ) (hx_pos : 0 < x)
  (h1 : 5 * x + 3 * b * x = k) -- Original cost is 5x + 3bx
  (h2 : b * x + 15 * x = 2 * k) -- Interchanged cost is doubled
  : b = 1 :=
by sorry

end sock_ratio_l2156_215679
