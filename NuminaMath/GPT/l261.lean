import Mathlib

namespace NUMINAMATH_GPT_identify_smart_person_l261_26194

theorem identify_smart_person (F S : ℕ) (h_total : F + S = 30) (h_max_fools : F ≤ 8) : S ≥ 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_identify_smart_person_l261_26194


namespace NUMINAMATH_GPT_num_chords_num_triangles_l261_26139

noncomputable def num_points : ℕ := 10

theorem num_chords (n : ℕ) (h : n = num_points) : (n.choose 2) = 45 := by
  sorry

theorem num_triangles (n : ℕ) (h : n = num_points) : (n.choose 3) = 120 := by
  sorry

end NUMINAMATH_GPT_num_chords_num_triangles_l261_26139


namespace NUMINAMATH_GPT_koschei_never_escapes_l261_26109

-- Define a structure for the initial setup
structure Setup where
  koschei_initial_room : Nat -- Initial room of Koschei
  guard_positions : List (Bool) -- Guards' positions, True for West, False for East

-- Example of the required setup:
def initial_setup : Setup :=
  { koschei_initial_room := 1, guard_positions := [true, false, true] }

-- Function to simulate the movement of guards
def move_guards (guards : List Bool) (room : Nat) : List Bool :=
  guards.map (λ g => not g)

-- Function to check if all guards are on the same wall
def all_guards_same_wall (guards : List Bool) : Bool :=
  List.all guards id ∨ List.all guards (λ g => ¬g)

-- Main statement: 
theorem koschei_never_escapes (setup : Setup) :
  ∀ room : Nat, ¬(all_guards_same_wall (move_guards setup.guard_positions room)) :=
  sorry

end NUMINAMATH_GPT_koschei_never_escapes_l261_26109


namespace NUMINAMATH_GPT_fraction_of_usual_speed_l261_26195

-- Definitions based on conditions
variable (S R : ℝ)
variable (h1 : S * 60 = R * 72)

-- Goal statement
theorem fraction_of_usual_speed (h1 : S * 60 = R * 72) : R / S = 5 / 6 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_usual_speed_l261_26195


namespace NUMINAMATH_GPT_average_speed_calculation_l261_26192

def average_speed (s1 s2 t1 t2 : ℕ) : ℕ :=
  (s1 * t1 + s2 * t2) / (t1 + t2)

theorem average_speed_calculation :
  average_speed 40 60 1 3 = 55 :=
by
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_average_speed_calculation_l261_26192


namespace NUMINAMATH_GPT_females_advanced_degrees_under_40_l261_26100

-- Definitions derived from conditions
def total_employees : ℕ := 280
def female_employees : ℕ := 160
def male_employees : ℕ := 120
def advanced_degree_holders : ℕ := 120
def college_degree_holders : ℕ := 100
def high_school_diploma_holders : ℕ := 60
def male_advanced_degree_holders : ℕ := 50
def male_college_degree_holders : ℕ := 35
def male_high_school_diploma_holders : ℕ := 35
def percentage_females_under_40 : ℝ := 0.75

-- The mathematically equivalent proof problem
theorem females_advanced_degrees_under_40 : 
  (advanced_degree_holders - male_advanced_degree_holders) * percentage_females_under_40 = 52 :=
by
  sorry -- Proof to be provided

end NUMINAMATH_GPT_females_advanced_degrees_under_40_l261_26100


namespace NUMINAMATH_GPT_combined_weight_l261_26113

theorem combined_weight (mary_weight : ℝ) (jamison_weight : ℝ) (john_weight : ℝ) :
  mary_weight = 160 ∧ jamison_weight = mary_weight + 20 ∧ john_weight = mary_weight + (0.25 * mary_weight) →
  john_weight + mary_weight + jamison_weight = 540 :=
by
  intros h
  obtain ⟨hm, hj, hj'⟩ := h
  rw [hm, hj, hj']
  norm_num
  sorry

end NUMINAMATH_GPT_combined_weight_l261_26113


namespace NUMINAMATH_GPT_banana_equivalence_l261_26129

theorem banana_equivalence :
  (3 / 4 : ℚ) * 12 = 9 → (1 / 3 : ℚ) * 6 = 2 :=
by
  intro h1
  linarith

end NUMINAMATH_GPT_banana_equivalence_l261_26129


namespace NUMINAMATH_GPT_total_slices_l261_26137

theorem total_slices {slices_per_pizza pizzas : ℕ} (h1 : slices_per_pizza = 2) (h2 : pizzas = 14) : 
  slices_per_pizza * pizzas = 28 :=
by
  -- This is where the proof would go, but we are omitting it as instructed.
  sorry

end NUMINAMATH_GPT_total_slices_l261_26137


namespace NUMINAMATH_GPT_circle_x_intercept_of_given_diameter_l261_26128

theorem circle_x_intercept_of_given_diameter (A B : ℝ × ℝ) (hA : A = (2, 2)) (hB : B = (10, 8)) : ∃ x : ℝ, ((A.1 + B.1) / 2, (A.2 + B.2) / 2).1 - 6 = 0 :=
by
  -- Sorry to skip the proof
  sorry

end NUMINAMATH_GPT_circle_x_intercept_of_given_diameter_l261_26128


namespace NUMINAMATH_GPT_part1_part2_l261_26151

variable (x : ℝ)
def A : ℝ := 2 * x^2 - 3 * x + 2
def B : ℝ := x^2 - 3 * x - 2

theorem part1 : A x - B x = x^2 + 4 := sorry

theorem part2 (h : x = -2) : A x - B x = 8 := sorry

end NUMINAMATH_GPT_part1_part2_l261_26151


namespace NUMINAMATH_GPT_find_x_values_l261_26140

noncomputable def condition (x : ℝ) : Prop :=
  1 / (x * (x + 2)) - 1 / ((x + 1) * (x + 3)) < 1 / 4

theorem find_x_values : 
  {x : ℝ | condition  x} = {x : ℝ | x < -3} ∪ {x : ℝ | -1 < x ∧ x < 0} :=
by sorry

end NUMINAMATH_GPT_find_x_values_l261_26140


namespace NUMINAMATH_GPT_complement_of_angle_l261_26191

theorem complement_of_angle (x : ℝ) (h1 : 3 * x + 10 = 90 - x) : 3 * x + 10 = 70 :=
by
  sorry

end NUMINAMATH_GPT_complement_of_angle_l261_26191


namespace NUMINAMATH_GPT_rate_of_work_l261_26176

theorem rate_of_work (A : ℝ) (h1: 0 < A) (h_eq : 1 / A + 1 / 6 = 1 / 2) : A = 3 := sorry

end NUMINAMATH_GPT_rate_of_work_l261_26176


namespace NUMINAMATH_GPT_points_on_line_relation_l261_26175

theorem points_on_line_relation (b y1 y2 y3 : ℝ) 
  (h1 : y1 = -3 * (-2) + b) 
  (h2 : y2 = -3 * (-1) + b) 
  (h3 : y3 = -3 * 1 + b) : 
  y1 > y2 ∧ y2 > y3 :=
sorry

end NUMINAMATH_GPT_points_on_line_relation_l261_26175


namespace NUMINAMATH_GPT_find_V_D_l261_26116

noncomputable def V_A : ℚ := sorry
noncomputable def V_B : ℚ := sorry
noncomputable def V_C : ℚ := sorry
noncomputable def V_D : ℚ := sorry
noncomputable def V_E : ℚ := sorry

axiom condition1 : V_A + V_B + V_C + V_D + V_E = 1 / 7.5
axiom condition2 : V_A + V_C + V_E = 1 / 5
axiom condition3 : V_A + V_C + V_D = 1 / 6
axiom condition4 : V_B + V_D + V_E = 1 / 4

theorem find_V_D : V_D = 1 / 12 := 
  by
    sorry

end NUMINAMATH_GPT_find_V_D_l261_26116


namespace NUMINAMATH_GPT_final_price_is_correct_l261_26122

def initial_price : ℝ := 15
def first_discount_rate : ℝ := 0.2
def second_discount_rate : ℝ := 0.25

def first_discount : ℝ := initial_price * first_discount_rate
def price_after_first_discount : ℝ := initial_price - first_discount

def second_discount : ℝ := price_after_first_discount * second_discount_rate
def final_price : ℝ := price_after_first_discount - second_discount

theorem final_price_is_correct :
  final_price = 9 :=
by
  -- The actual proof steps will go here.
  sorry

end NUMINAMATH_GPT_final_price_is_correct_l261_26122


namespace NUMINAMATH_GPT_trig_identity_example_l261_26187

theorem trig_identity_example :
  (2 * (Real.sin (Real.pi / 6)) - Real.tan (Real.pi / 4)) = 0 :=
by
  -- Definitions from conditions
  have h1 : Real.sin (Real.pi / 6) = 1/2 := Real.sin_pi_div_six
  have h2 : Real.tan (Real.pi / 4) = 1 := Real.tan_pi_div_four
  rw [h1, h2]
  sorry -- The proof is omitted as per instructions

end NUMINAMATH_GPT_trig_identity_example_l261_26187


namespace NUMINAMATH_GPT_value_of_expression_l261_26143

theorem value_of_expression (a b : ℤ) (h : 1^2 + a * 1 + 2 * b = 0) : 2023 - a - 2 * b = 2024 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l261_26143


namespace NUMINAMATH_GPT_number_of_valid_m_l261_26145

def is_right_triangle (P Q R : ℝ × ℝ) : Prop :=
  let (Px, Py) := P;
  let (Qx, Qy) := Q;
  let (Rx, Ry) := R;
  (Qx - Px) * (Qx - Px) + (Qy - Py) * (Qy - Py) + (Rx - Qx) * (Rx - Qx) + (Ry - Qy) * (Ry - Qy) ==
  (Px - Rx) * (Px - Rx) + (Py - Ry) * (Py - Ry) + 2 * ((Qx - Px) * (Rx - Qx) + (Qy - Py) * (Ry - Qy))

def legs_parallel_to_axes (P Q R : ℝ × ℝ) : Prop :=
  let (Px, Py) := P;
  let (Qx, Qy) := Q;
  let (Rx, Ry) := R;
  Px = Qx ∨ Px = Rx ∨ Qx = Rx ∧ Py = Qy ∨ Py = Ry ∨ Qy = Ry

def medians_condition (P Q R : ℝ × ℝ) : Prop :=
  let (Px, Py) := P;
  let (Qx, Qy) := Q;
  let (Rx, Ry) := R;
  let M_PQ := ((Px + Qx) / 2, (Py + Qy) / 2);
  let M_PR := ((Px + Rx) / 2, (Py + Ry) / 2);
  (M_PQ.2 = 3 * M_PQ.1 + 1) ∧ (M_PR.2 = 2)

theorem number_of_valid_m (a b c d : ℝ) (h1 : c > 0) (h2 : d > 0)
  (P := (a, b)) (Q := (a, b+2*c)) (R := (a-2*d, b)) :
  is_right_triangle P Q R →
  legs_parallel_to_axes P Q R →
  medians_condition P Q R →
  ∃ m, m = 1 :=
sorry

end NUMINAMATH_GPT_number_of_valid_m_l261_26145


namespace NUMINAMATH_GPT_quadratic_equation_solution_l261_26130

theorem quadratic_equation_solution (m : ℝ) :
  (m - 3) * x ^ (m^2 - 7) - x + 3 = 0 → m^2 - 7 = 2 → m ≠ 3 → m = -3 :=
by
  intros h_eq h_power h_nonzero
  sorry

end NUMINAMATH_GPT_quadratic_equation_solution_l261_26130


namespace NUMINAMATH_GPT_mean_of_two_means_eq_l261_26183

theorem mean_of_two_means_eq (z : ℚ) (h : (5 + 10 + 20) / 3 = (15 + z) / 2) : z = 25 / 3 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_two_means_eq_l261_26183


namespace NUMINAMATH_GPT_inverse_proportion_inequality_l261_26117

variable (x1 x2 k : ℝ)

theorem inverse_proportion_inequality (hA : 2 = k / x1) (hB : 4 = k / x2) (hk : 0 < k) : 
  x1 > x2 ∧ x1 > 0 ∧ x2 > 0 :=
sorry

end NUMINAMATH_GPT_inverse_proportion_inequality_l261_26117


namespace NUMINAMATH_GPT_willie_cream_l261_26125

theorem willie_cream : ∀ (total_cream needed_cream: ℕ), total_cream = 300 → needed_cream = 149 → (total_cream - needed_cream) = 151 :=
by
  intros total_cream needed_cream h1 h2
  sorry

end NUMINAMATH_GPT_willie_cream_l261_26125


namespace NUMINAMATH_GPT_interest_rate_of_A_to_B_l261_26124

theorem interest_rate_of_A_to_B :
  ∀ (principal gain interest_B_to_C : ℝ), 
  principal = 3500 →
  gain = 525 →
  interest_B_to_C = 0.15 →
  (principal * interest_B_to_C * 3 - gain) = principal * (10 / 100) * 3 :=
by
  intros principal gain interest_B_to_C h_principal h_gain h_interest_B_to_C
  sorry

end NUMINAMATH_GPT_interest_rate_of_A_to_B_l261_26124


namespace NUMINAMATH_GPT_printing_time_l261_26190

-- Definitions based on the problem conditions
def printer_rate : ℕ := 25 -- Pages per minute
def total_pages : ℕ := 325 -- Total number of pages to be printed

-- Statement of the problem rewritten as a Lean 4 statement
theorem printing_time : total_pages / printer_rate = 13 := by
  sorry

end NUMINAMATH_GPT_printing_time_l261_26190


namespace NUMINAMATH_GPT_symmetric_function_value_l261_26108

noncomputable def f (x a : ℝ) := (|x - 2| + a) / (Real.sqrt (4 - x^2))

theorem symmetric_function_value :
  ∃ a : ℝ, (∀ x : ℝ, f x a = (|x - 2| + a) / (Real.sqrt (4 - x^2)) ∧ f x a = -f (-x) a) →
  f (a / 2) a = (Real.sqrt 3) / 3 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_function_value_l261_26108


namespace NUMINAMATH_GPT_smallest_repeating_block_7_over_13_l261_26155

theorem smallest_repeating_block_7_over_13 : 
  ∃ n : ℕ, (∀ d : ℕ, d < n → 
  (∃ (q r : ℕ), r < 13 ∧ 10 ^ (d + 1) * 7 % 13 = q * 10 ^ n + r)) ∧ n = 6 := sorry

end NUMINAMATH_GPT_smallest_repeating_block_7_over_13_l261_26155


namespace NUMINAMATH_GPT_xyz_value_l261_26147

-- Define real numbers x, y, z
variables {x y z : ℝ}

-- Define the theorem with the given conditions and conclusion
theorem xyz_value 
  (h1 : (x + y + z) * (xy + xz + yz) = 36)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12)
  (h3 : (x + y + z)^2 = x^2 + y^2 + z^2 + 12) :
  x * y * z = 8 := 
sorry

end NUMINAMATH_GPT_xyz_value_l261_26147


namespace NUMINAMATH_GPT_reciprocal_of_neg_five_l261_26135

theorem reciprocal_of_neg_five : ∃ b : ℚ, (-5) * b = 1 ∧ b = -1/5 :=
by
  sorry

end NUMINAMATH_GPT_reciprocal_of_neg_five_l261_26135


namespace NUMINAMATH_GPT_graph_of_function_does_not_pass_through_first_quadrant_l261_26127

theorem graph_of_function_does_not_pass_through_first_quadrant (k : ℝ) (h : k < 0) : 
  ¬(∃ x y : ℝ, y = k * (x - k) ∧ x > 0 ∧ y > 0) :=
sorry

end NUMINAMATH_GPT_graph_of_function_does_not_pass_through_first_quadrant_l261_26127


namespace NUMINAMATH_GPT_flowers_total_l261_26163

theorem flowers_total (yoojung_flowers : ℕ) (namjoon_flowers : ℕ)
 (h1 : yoojung_flowers = 32)
 (h2 : yoojung_flowers = 4 * namjoon_flowers) :
  yoojung_flowers + namjoon_flowers = 40 := by
  sorry

end NUMINAMATH_GPT_flowers_total_l261_26163


namespace NUMINAMATH_GPT_solve_for_q_l261_26150

theorem solve_for_q (x y q : ℚ) 
  (h1 : 7 / 8 = x / 96) 
  (h2 : 7 / 8 = (x + y) / 104) 
  (h3 : 7 / 8 = (q - y) / 144) : 
  q = 133 := 
sorry

end NUMINAMATH_GPT_solve_for_q_l261_26150


namespace NUMINAMATH_GPT_room_length_difference_l261_26104

def width := 19
def length := 20
def difference := length - width

theorem room_length_difference : difference = 1 := by
  sorry

end NUMINAMATH_GPT_room_length_difference_l261_26104


namespace NUMINAMATH_GPT_largest_value_WY_cyclic_quadrilateral_l261_26185

theorem largest_value_WY_cyclic_quadrilateral :
  ∃ WZ ZX ZY YW : ℕ, 
    WZ ≠ ZX ∧ WZ ≠ ZY ∧ WZ ≠ YW ∧ ZX ≠ ZY ∧ ZX ≠ YW ∧ ZY ≠ YW ∧ 
    WZ < 20 ∧ ZX < 20 ∧ ZY < 20 ∧ YW < 20 ∧ 
    WZ * ZY = ZX * YW ∧
    (∀ WY', (∃ WY : ℕ, WY' < WY → WY <= 19 )) :=
sorry

end NUMINAMATH_GPT_largest_value_WY_cyclic_quadrilateral_l261_26185


namespace NUMINAMATH_GPT_cost_price_of_cupboard_l261_26180

theorem cost_price_of_cupboard (C S S_profit : ℝ) (h1 : S = 0.88 * C) (h2 : S_profit = 1.12 * C) (h3 : S_profit - S = 1650) :
  C = 6875 := by
  sorry

end NUMINAMATH_GPT_cost_price_of_cupboard_l261_26180


namespace NUMINAMATH_GPT_find_n_l261_26188

theorem find_n (n : ℕ) (h1 : 0 < n) (h2 : n < 11) (h3 : (18888 - n) % 11 = 0) : n = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l261_26188


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l261_26168

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + 1 > 0)) ↔ ∃ x : ℝ, x^2 + 1 ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l261_26168


namespace NUMINAMATH_GPT_analytic_expression_and_symmetry_l261_26158

noncomputable def f (A : ℝ) (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ :=
  A * Real.sin (ω * x + φ)

theorem analytic_expression_and_symmetry {A ω φ : ℝ}
  (hA : A > 0) 
  (hω : ω > 0)
  (h_period : ∀ x, f A ω φ (x + 2) = f A ω φ x)
  (h_max : f A ω φ (1 / 3) = 2) :
  (f 2 π (π / 6) = fun x => 2 * Real.sin (π * x + π / 6)) ∧
  (∃ k : ℤ, k = 5 ∧ (1 / 3 + k = 16 / 3) ∧ (21 / 4 ≤ 1 / 3 + ↑k) ∧ (1 / 3 + ↑k ≤ 23 / 4)) :=
  sorry

end NUMINAMATH_GPT_analytic_expression_and_symmetry_l261_26158


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l261_26106

theorem boat_speed_in_still_water
  (speed_of_stream : ℝ)
  (time_downstream : ℝ)
  (distance_downstream : ℝ)
  (effective_speed : ℝ)
  (boat_speed : ℝ)
  (h1: speed_of_stream = 5)
  (h2: time_downstream = 2)
  (h3: distance_downstream = 54)
  (h4: effective_speed = boat_speed + speed_of_stream)
  (h5: distance_downstream = effective_speed * time_downstream) :
  boat_speed = 22 := by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l261_26106


namespace NUMINAMATH_GPT_gcd_135_81_l261_26197

-- Define the numbers
def a : ℕ := 135
def b : ℕ := 81

-- State the goal: greatest common divisor of a and b is 27
theorem gcd_135_81 : Nat.gcd a b = 27 := by
  sorry

end NUMINAMATH_GPT_gcd_135_81_l261_26197


namespace NUMINAMATH_GPT_train_length_correct_l261_26179

noncomputable def train_length (time : ℝ) (speed_kmph : ℝ) : ℝ :=
  let speed_mps := speed_kmph * 1000 / 3600
  speed_mps * time

theorem train_length_correct :
  train_length 17.998560115190784 36 = 179.98560115190784 :=
by
  sorry

end NUMINAMATH_GPT_train_length_correct_l261_26179


namespace NUMINAMATH_GPT_max_distance_bicycle_l261_26149

theorem max_distance_bicycle (front_tire_last : ℕ) (rear_tire_last : ℕ) :
  front_tire_last = 5000 ∧ rear_tire_last = 3000 →
  ∃ (max_distance : ℕ), max_distance = 3750 :=
by
  sorry

end NUMINAMATH_GPT_max_distance_bicycle_l261_26149


namespace NUMINAMATH_GPT_find_a_l261_26164

theorem find_a (x y a : ℕ) (h1 : ((10 : ℕ) ^ ((32 : ℕ) / y)) ^ a - (64 : ℕ) = (279 : ℕ))
                 (h2 : a > 0)
                 (h3 : x * y = 32) :
  a = 1 :=
sorry

end NUMINAMATH_GPT_find_a_l261_26164


namespace NUMINAMATH_GPT_c_share_of_profit_l261_26153

theorem c_share_of_profit (a b c total_profit : ℕ) 
  (h₁ : a = 5000) (h₂ : b = 8000) (h₃ : c = 9000) (h₄ : total_profit = 88000) :
  c * total_profit / (a + b + c) = 36000 :=
by
  sorry

end NUMINAMATH_GPT_c_share_of_profit_l261_26153


namespace NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l261_26115

def f (x : ℝ) : ℝ := x^5 - 4 * x^4 + 6 * x^3 + 25 * x^2 - 20 * x - 24

theorem remainder_when_divided_by_x_minus_2 : f 2 = 52 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_x_minus_2_l261_26115


namespace NUMINAMATH_GPT_total_questions_attempted_l261_26174

/-- 
In an examination, a student scores 3 marks for every correct answer and loses 1 mark for
every wrong answer. He attempts some questions and secures 180 marks. The number of questions
he attempts correctly is 75. Prove that the total number of questions he attempts is 120. 
-/
theorem total_questions_attempted
  (marks_per_correct : ℕ := 3)
  (marks_lost_per_wrong : ℕ := 1)
  (total_marks : ℕ := 180)
  (correct_answers : ℕ := 75) :
  ∃ (wrong_answers total_questions : ℕ), 
    total_marks = (marks_per_correct * correct_answers) - (marks_lost_per_wrong * wrong_answers) ∧
    total_questions = correct_answers + wrong_answers ∧
    total_questions = 120 := 
by {
  sorry -- proof omitted
}

end NUMINAMATH_GPT_total_questions_attempted_l261_26174


namespace NUMINAMATH_GPT_total_tickets_sold_l261_26154

-- Define the conditions
variables (V G : ℕ)

-- Condition 1: Total revenue from VIP and general admission
def total_revenue_eq : Prop := 40 * V + 15 * G = 7500

-- Condition 2: There are 212 fewer VIP tickets than general admission
def vip_tickets_eq : Prop := V = G - 212

-- Main statement to prove: the total number of tickets sold
theorem total_tickets_sold (h1 : total_revenue_eq V G) (h2 : vip_tickets_eq V G) : V + G = 370 :=
sorry

end NUMINAMATH_GPT_total_tickets_sold_l261_26154


namespace NUMINAMATH_GPT_isosceles_triangle_circles_distance_l261_26166

theorem isosceles_triangle_circles_distance (h α : ℝ) (hα : α ≤ π / 6) :
    let R := h / (2 * (Real.cos α)^2)
    let r := h * (Real.tan α) * (Real.tan (π / 4 - α / 2))
    let OO1 := h * (1 - 1 / (2 * (Real.cos α)^2) - (Real.tan α) * (Real.tan (π / 4 - α / 2)))
    OO1 = (2 * h * Real.sin (π / 12 - α / 2) * Real.cos (π / 12 + α / 2)) / (Real.cos α)^2 :=
    sorry

end NUMINAMATH_GPT_isosceles_triangle_circles_distance_l261_26166


namespace NUMINAMATH_GPT_find_day_for_balance_l261_26161

-- Define the initial conditions and variables
def initialEarnings : ℤ := 20
def secondDaySpending : ℤ := 15
variables (X Y : ℤ)

-- Define the function for net balance on day D
def netBalance (D : ℤ) : ℤ :=
  initialEarnings + (D - 1) * X - (secondDaySpending + (D - 2) * Y)

-- The main theorem proving the day D for net balance of Rs. 60
theorem find_day_for_balance (X Y : ℤ) : ∃ D : ℤ, netBalance X Y D = 60 → 55 = (D + 1) * (X - Y) :=
by
  sorry

end NUMINAMATH_GPT_find_day_for_balance_l261_26161


namespace NUMINAMATH_GPT_stratified_sampling_number_l261_26169

noncomputable def students_in_grade_10 : ℕ := 150
noncomputable def students_in_grade_11 : ℕ := 180
noncomputable def students_in_grade_12 : ℕ := 210
noncomputable def total_students : ℕ := students_in_grade_10 + students_in_grade_11 + students_in_grade_12
noncomputable def sample_size : ℕ := 72
noncomputable def selection_probability : ℚ := sample_size / total_students
noncomputable def combined_students_grade_10_11 : ℕ := students_in_grade_10 + students_in_grade_11

theorem stratified_sampling_number :
  combined_students_grade_10_11 * selection_probability = 44 := 
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_number_l261_26169


namespace NUMINAMATH_GPT_triangle_angle_sum_l261_26101

-- Definitions of the given angles and relationships
def angle_BAC := 95
def angle_ABC := 55
def angle_ABD := 125

-- We need to express the configuration of points and the measure of angle ACB
noncomputable def angle_ACB (angle_BAC angle_ABC angle_ABD : ℝ) : ℝ :=
  180 - angle_BAC - angle_ABC

-- The formalization of the problem statement in Lean 4
theorem triangle_angle_sum (angle_BAC angle_ABC angle_ABD : ℝ) :
  angle_BAC = 95 → angle_ABC = 55 → angle_ABD = 125 → angle_ACB angle_BAC angle_ABC angle_ABD = 30 :=
by
  intros h_BAC h_ABC h_ABD
  rw [h_BAC, h_ABC, h_ABD]
  sorry

end NUMINAMATH_GPT_triangle_angle_sum_l261_26101


namespace NUMINAMATH_GPT_colin_speed_l261_26142

variable (B T Bn C : ℝ)
variable (m : ℝ)

-- Given conditions
def condition1 := B = 1
def condition2 := T = m * B
def condition3 := Bn = T / 3
def condition4 := C = 6 * Bn
def condition5 := C = 4

-- We need to prove C = 4 given the conditions
theorem colin_speed :
  (B = 1) →
  (T = m * B) →
  (Bn = T / 3) →
  (C = 6 * Bn) →
  C = 4 :=
by
  intros _ _ _ _
  sorry

end NUMINAMATH_GPT_colin_speed_l261_26142


namespace NUMINAMATH_GPT_problem1_problem2_l261_26119

-- Define the first problem
theorem problem1 (x : ℝ) : (x - 2) ^ 2 = 2 * x - 4 ↔ (x = 2 ∨ x = 4) := 
by 
  sorry

-- Define the second problem using completing the square method
theorem problem2 (x : ℝ) : x ^ 2 - 4 * x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) := 
by 
  sorry

end NUMINAMATH_GPT_problem1_problem2_l261_26119


namespace NUMINAMATH_GPT_sum_of_n_values_l261_26171

theorem sum_of_n_values : ∃ n1 n2 : ℚ, (abs (3 * n1 - 4) = 5) ∧ (abs (3 * n2 - 4) = 5) ∧ n1 + n2 = 8 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_n_values_l261_26171


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l261_26193

def setA : Set ℕ := {1, 2, 3}
def setB : Set ℕ := {2, 4, 6}

theorem intersection_of_A_and_B : setA ∩ setB = {2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l261_26193


namespace NUMINAMATH_GPT_inequality_proof_l261_26186

theorem inequality_proof (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b + c = 3) : 
  a^b + b^c + c^a ≤ a^2 + b^2 + c^2 := 
by sorry

end NUMINAMATH_GPT_inequality_proof_l261_26186


namespace NUMINAMATH_GPT_james_writing_time_l261_26138

theorem james_writing_time (pages_per_hour : ℕ) (pages_per_person_per_day : ℕ) (num_people : ℕ) (days_per_week : ℕ):
  pages_per_hour = 10 →
  pages_per_person_per_day = 5 →
  num_people = 2 →
  days_per_week = 7 →
  (5 * 2 * 7) / 10 = 7 :=
by
  intros
  sorry

end NUMINAMATH_GPT_james_writing_time_l261_26138


namespace NUMINAMATH_GPT_quadratic_polynomials_exist_l261_26123

-- Definitions of the polynomials
def p1 (x : ℝ) := (x - 10)^2 - 1
def p2 (x : ℝ) := x^2 - 1
def p3 (x : ℝ) := (x + 10)^2 - 1

-- The theorem to prove
theorem quadratic_polynomials_exist :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ p1 x1 = 0 ∧ p1 x2 = 0) ∧
  (∃ y1 y2 : ℝ, y1 ≠ y2 ∧ p2 y1 = 0 ∧ p2 y2 = 0) ∧
  (∃ z1 z2 : ℝ, z1 ≠ z2 ∧ p3 z1 = 0 ∧ p3 z2 = 0) ∧
  (∀ x : ℝ, p1 x + p2 x ≠ 0 ∧ p1 x + p3 x ≠ 0 ∧ p2 x + p3 x ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_polynomials_exist_l261_26123


namespace NUMINAMATH_GPT_decode_division_problem_l261_26152

theorem decode_division_problem :
  let dividend := 1089708
  let divisor := 12
  let quotient := 90809
  dividend / divisor = quotient :=
by {
  -- Definitions of given and derived values
  let dividend := 1089708
  let divisor := 12
  let quotient := 90809
  -- The statement to prove
  sorry
}

end NUMINAMATH_GPT_decode_division_problem_l261_26152


namespace NUMINAMATH_GPT_square_roots_equal_49_l261_26118

theorem square_roots_equal_49 (x a : ℝ) (hx1 : (2 * x - 3)^2 = a) (hx2 : (5 - x)^2 = a) (ha_pos: a > 0) : a = 49 := 
by 
  sorry

end NUMINAMATH_GPT_square_roots_equal_49_l261_26118


namespace NUMINAMATH_GPT_fraction_of_remaining_birds_left_l261_26162

theorem fraction_of_remaining_birds_left (B : ℕ) (F : ℚ) (hB : B = 60)
  (H : (1/3) * (2/3 : ℚ) * B * (1 - F) = 8) :
  F = 4/5 := 
sorry

end NUMINAMATH_GPT_fraction_of_remaining_birds_left_l261_26162


namespace NUMINAMATH_GPT_area_ratio_of_circles_l261_26102

theorem area_ratio_of_circles 
  (CX : ℝ)
  (CY : ℝ)
  (RX RY : ℝ)
  (hX : CX = 2 * π * RX)
  (hY : CY = 2 * π * RY)
  (arc_length_equality : (90 / 360) * CX = (60 / 360) * CY) :
  (π * RX^2) / (π * RY^2) = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_area_ratio_of_circles_l261_26102


namespace NUMINAMATH_GPT_vinegar_ratio_to_total_capacity_l261_26159

theorem vinegar_ratio_to_total_capacity (bowl_capacity : ℝ) (oil_fraction : ℝ) 
  (oil_density : ℝ) (vinegar_density : ℝ) (total_weight : ℝ) :
  bowl_capacity = 150 ∧ oil_fraction = 2/3 ∧ oil_density = 5 ∧ vinegar_density = 4 ∧ total_weight = 700 →
  (total_weight - (bowl_capacity * oil_fraction * oil_density)) / vinegar_density / bowl_capacity = 1/3 :=
by
  sorry

end NUMINAMATH_GPT_vinegar_ratio_to_total_capacity_l261_26159


namespace NUMINAMATH_GPT_reinforcement_1600_l261_26105

/-- A garrison of 2000 men has provisions for 54 days. After 18 days, a reinforcement arrives, and it is now found that the provisions will last only for 20 days more. We define the initial total provisions, remaining provisions after 18 days, and form equations to solve for the unknown reinforcement R.
We need to prove that R = 1600 given these conditions.
-/
theorem reinforcement_1600 (P : ℕ) (M1 M2 : ℕ) (D1 D2 : ℕ) (R : ℕ) :
  M1 = 2000 →
  D1 = 54 →
  D2 = 20 →
  M2 = 2000 + R →
  P = M1 * D1 →
  (M1 * (D1 - 18) = M2 * D2) →
  R = 1600 :=
by
  intros hM1 hD1 hD2 hM2 hP hEquiv
  sorry

end NUMINAMATH_GPT_reinforcement_1600_l261_26105


namespace NUMINAMATH_GPT_graph_intersection_l261_26167

noncomputable def log : ℝ → ℝ := sorry

lemma log_properties (a b : ℝ) (ha : 0 < a) (hb : 0 < b): log (a * b) = log a + log b := sorry

theorem graph_intersection :
  ∃! x : ℝ, 2 * log x = log (2 * x) :=
by
  sorry

end NUMINAMATH_GPT_graph_intersection_l261_26167


namespace NUMINAMATH_GPT_ratio_first_term_common_diff_l261_26134

theorem ratio_first_term_common_diff {a d : ℤ} 
  (S_20 : ℤ) (S_10 : ℤ)
  (h1 : S_20 = 10 * (2 * a + 19 * d))
  (h2 : S_10 = 5 * (2 * a + 9 * d))
  (h3 : S_20 = 6 * S_10) :
  a / d = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_first_term_common_diff_l261_26134


namespace NUMINAMATH_GPT_minimum_value_of_reciprocals_l261_26148

theorem minimum_value_of_reciprocals (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hline : a - b = 1) :
  (1 / a) + (1 / b) ≥ 4 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_reciprocals_l261_26148


namespace NUMINAMATH_GPT_union_of_P_and_Q_l261_26173

noncomputable def P : Set ℝ := {x | -1 < x ∧ x < 1}
noncomputable def Q : Set ℝ := {x | 0 < x ∧ x < 2}

theorem union_of_P_and_Q :
  P ∪ Q = {x | -1 < x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_union_of_P_and_Q_l261_26173


namespace NUMINAMATH_GPT_element_in_set_l261_26132

open Set

theorem element_in_set : -7 ∈ ({1, -7} : Set ℤ) := by
  sorry

end NUMINAMATH_GPT_element_in_set_l261_26132


namespace NUMINAMATH_GPT_increasing_interval_l261_26196

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * x^2

theorem increasing_interval :
  ∃ a b : ℝ, (0 < a) ∧ (a < b) ∧ (b = 1/2) ∧ (∀ x : ℝ, a < x ∧ x < b → (deriv f x > 0)) :=
by
  sorry

end NUMINAMATH_GPT_increasing_interval_l261_26196


namespace NUMINAMATH_GPT_sum_tens_units_digit_9_pow_1001_l261_26103

-- Define a function to extract the last two digits of a number
def last_two_digits (n : ℕ) : ℕ := n % 100

-- Define a function to extract the tens digit
def tens_digit (n : ℕ) : ℕ := (last_two_digits n) / 10

-- Define a function to extract the units digit
def units_digit (n : ℕ) : ℕ := (last_two_digits n) % 10

-- The main theorem
theorem sum_tens_units_digit_9_pow_1001 :
  tens_digit (9 ^ 1001) + units_digit (9 ^ 1001) = 9 :=
by
  sorry

end NUMINAMATH_GPT_sum_tens_units_digit_9_pow_1001_l261_26103


namespace NUMINAMATH_GPT_sum_of_roots_of_quadratic_l261_26198

theorem sum_of_roots_of_quadratic (a b c : ℝ) (h_eq : a = 3 ∧ b = 6 ∧ c = -9) :
  (-b / a) = -2 :=
by
  rcases h_eq with ⟨ha, hb, hc⟩
  -- Proof goes here, but we can use sorry to skip it
  sorry

end NUMINAMATH_GPT_sum_of_roots_of_quadratic_l261_26198


namespace NUMINAMATH_GPT_cost_of_pencils_l261_26160

def cost_of_notebooks : ℝ := 3 * 1.2
def cost_of_pens : ℝ := 1.7
def total_spent : ℝ := 6.8

theorem cost_of_pencils :
  total_spent - (cost_of_notebooks + cost_of_pens) = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_pencils_l261_26160


namespace NUMINAMATH_GPT_lines_intersection_l261_26177

theorem lines_intersection :
  ∃ (t u : ℚ), 
    (∃ (x y : ℚ),
    (x = 2 - t ∧ y = 3 + 4 * t) ∧ 
    (x = -1 + 3 * u ∧ y = 6 + 5 * u) ∧ 
    (x = 28 / 17 ∧ y = 75 / 17)) := sorry

end NUMINAMATH_GPT_lines_intersection_l261_26177


namespace NUMINAMATH_GPT_range_of_m_l261_26133

-- Define the sets A and B
def setA := {x : ℝ | abs (x - 1) < 2}
def setB (m : ℝ) := {x : ℝ | x >= m}

-- State the theorem
theorem range_of_m : ∀ (m : ℝ), (setA ∩ setB m = setA) → m <= -1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l261_26133


namespace NUMINAMATH_GPT_range_of_a_l261_26126

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l261_26126


namespace NUMINAMATH_GPT_polynomial_remainder_l261_26114

noncomputable def h (x : ℕ) := x^5 + x^4 + x^3 + x^2 + x + 1

theorem polynomial_remainder (x : ℕ) : (h (x^10)) % (h x) = 5 :=
sorry

end NUMINAMATH_GPT_polynomial_remainder_l261_26114


namespace NUMINAMATH_GPT_inequality_and_equality_condition_l261_26111

variable (a b c t : ℝ)
variable (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem inequality_and_equality_condition :
  abc * (a^t + b^t + c^t) ≥ a^(t+2) * (-a + b + c) + b^(t+2) * (a - b + c) + c^(t+2) * (a + b - c) ∧ 
  (abc * (a^t + b^t + c^t) = a^(t+2) * (-a + b + c) + b^(t+2) * (a - b + c) + c^(t+2) * (a + b - c) ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_GPT_inequality_and_equality_condition_l261_26111


namespace NUMINAMATH_GPT_negative_values_count_l261_26120

theorem negative_values_count (n : ℕ) : (n < 13) → (n^2 < 150) → ∃ (k : ℕ), k = 12 :=
by
  sorry

end NUMINAMATH_GPT_negative_values_count_l261_26120


namespace NUMINAMATH_GPT_line_CD_area_triangle_equality_line_CD_midpoint_l261_26184

theorem line_CD_area_triangle_equality :
  ∃ k : ℝ, 4 * k - 1 = 1 - k := sorry

theorem line_CD_midpoint :
  ∃ k : ℝ, 9 * k - 2 = 1 := sorry

end NUMINAMATH_GPT_line_CD_area_triangle_equality_line_CD_midpoint_l261_26184


namespace NUMINAMATH_GPT_watch_cost_l261_26156

-- Definitions based on conditions
def initial_money : ℤ := 1
def money_from_david : ℤ := 12
def money_needed : ℤ := 7

-- Indicating the total money Evan has after receiving money from David
def total_money := initial_money + money_from_david

-- The cost of the watch based on total money Evan has and additional money needed
def cost_of_watch := total_money + money_needed

-- Proving the cost of the watch
theorem watch_cost : cost_of_watch = 20 := by
  -- We are skipping the proof steps here
  sorry

end NUMINAMATH_GPT_watch_cost_l261_26156


namespace NUMINAMATH_GPT_cannot_form_shape_B_l261_26182

-- Define the given pieces
def pieces : List (List (Nat × Nat)) :=
  [ [(1, 1)],
    [(1, 2)],
    [(1, 1), (1, 1), (1, 1)],
    [(1, 1), (1, 1), (1, 1)],
    [(1, 3)],
    [(1, 3)] ]

-- Define shape B requirement
def shapeB : List (Nat × Nat) := [(1, 6)]

theorem cannot_form_shape_B :
  ¬ (∃ (combinations : List (List (Nat × Nat))), combinations ⊆ pieces ∧ 
     (List.foldr (λ x acc => acc + x) 0 (combinations.map (List.foldr (λ y acc => acc + (y.1 * y.2)) 0)) = 6)) :=
sorry

end NUMINAMATH_GPT_cannot_form_shape_B_l261_26182


namespace NUMINAMATH_GPT_tan_degree_identity_l261_26107

theorem tan_degree_identity (k : ℝ) (hk : Real.cos (Real.pi * -80 / 180) = k) : 
  Real.tan (Real.pi * 100 / 180) = - (Real.sqrt (1 - k^2) / k) := 
by 
  sorry

end NUMINAMATH_GPT_tan_degree_identity_l261_26107


namespace NUMINAMATH_GPT_percentage_needed_to_pass_l261_26172

-- Definitions for conditions
def obtained_marks : ℕ := 125
def failed_by : ℕ := 40
def total_marks : ℕ := 500
def passing_marks := obtained_marks + failed_by

-- Assertion to prove
theorem percentage_needed_to_pass : (passing_marks : ℕ) * 100 / total_marks = 33 := by
  sorry

end NUMINAMATH_GPT_percentage_needed_to_pass_l261_26172


namespace NUMINAMATH_GPT_ufo_convention_attendees_l261_26121

theorem ufo_convention_attendees (f m total : ℕ) 
  (h1 : m = 62) 
  (h2 : m = f + 4) : 
  total = 120 :=
by
  sorry

end NUMINAMATH_GPT_ufo_convention_attendees_l261_26121


namespace NUMINAMATH_GPT_imaginary_part_of_conjugate_l261_26165

def z : Complex := Complex.mk 1 2

def z_conj : Complex := Complex.mk 1 (-2)

theorem imaginary_part_of_conjugate :
  z_conj.im = -2 := by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_conjugate_l261_26165


namespace NUMINAMATH_GPT_closest_fraction_to_team_alpha_medals_l261_26181

theorem closest_fraction_to_team_alpha_medals :
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 5) ∧ 
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 6) ∧ 
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 7) ∧ 
  abs ((25 : ℚ) / 160 - 1 / 8) < abs ((25 : ℚ) / 160 - 1 / 9) := 
by
  sorry

end NUMINAMATH_GPT_closest_fraction_to_team_alpha_medals_l261_26181


namespace NUMINAMATH_GPT_max_pies_without_ingredients_l261_26136

theorem max_pies_without_ingredients (total_pies half_chocolate two_thirds_marshmallows three_fifths_cayenne one_eighth_peanuts : ℕ) 
  (h1 : total_pies = 48) 
  (h2 : half_chocolate = total_pies / 2)
  (h3 : two_thirds_marshmallows = 2 * total_pies / 3) 
  (h4 : three_fifths_cayenne = 3 * total_pies / 5)
  (h5 : one_eighth_peanuts = total_pies / 8) : 
  ∃ pies_without_any_ingredients, pies_without_any_ingredients = 16 :=
  by 
    sorry

end NUMINAMATH_GPT_max_pies_without_ingredients_l261_26136


namespace NUMINAMATH_GPT_average_annual_growth_rate_equation_l261_26141

variable (x : ℝ)
axiom seventh_to_ninth_reading_increase : (1 : ℝ) * (1 + x) * (1 + x) = 1.21

theorem average_annual_growth_rate_equation :
  100 * (1 + x) ^ 2 = 121 :=
by
  have h : (1 : ℝ) * (1 + x) * (1 + x) = 1.21 := seventh_to_ninth_reading_increase x
  sorry

end NUMINAMATH_GPT_average_annual_growth_rate_equation_l261_26141


namespace NUMINAMATH_GPT_base_eight_seventeen_five_is_one_two_five_l261_26112

def base_eight_to_base_ten (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (n / 100) * 8^2 + ((n % 100) / 10) * 8^1 + (n % 10) * 8^0

theorem base_eight_seventeen_five_is_one_two_five :
  base_eight_to_base_ten 175 = 125 :=
by
  sorry

end NUMINAMATH_GPT_base_eight_seventeen_five_is_one_two_five_l261_26112


namespace NUMINAMATH_GPT_best_fitting_model_l261_26146

theorem best_fitting_model (R2_1 R2_2 R2_3 R2_4 : ℝ)
  (h1 : R2_1 = 0.25) 
  (h2 : R2_2 = 0.50) 
  (h3 : R2_3 = 0.80) 
  (h4 : R2_4 = 0.98) : 
  (R2_4 = max (max R2_1 (max R2_2 R2_3)) R2_4) :=
by
  sorry

end NUMINAMATH_GPT_best_fitting_model_l261_26146


namespace NUMINAMATH_GPT_isosceles_triangle_base_length_l261_26157

theorem isosceles_triangle_base_length
  (perimeter_eq_triangle : ℕ)
  (perimeter_isosceles_triangle : ℕ)
  (side_eq_triangle_isosceles : ℕ)
  (side_eq : side_eq_triangle_isosceles = perimeter_eq_triangle / 3)
  (perimeter_eq : perimeter_isosceles_triangle = 2 * side_eq_triangle_isosceles + 15) :
  15 = perimeter_isosceles_triangle - 2 * side_eq_triangle_isosceles :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_base_length_l261_26157


namespace NUMINAMATH_GPT_test_score_range_l261_26131

theorem test_score_range
  (mark_score : ℕ) (least_score : ℕ) (highest_score : ℕ)
  (twice_least_score : mark_score = 2 * least_score)
  (mark_fixed : mark_score = 46)
  (highest_fixed : highest_score = 98) :
  (highest_score - least_score) = 75 :=
by
  sorry

end NUMINAMATH_GPT_test_score_range_l261_26131


namespace NUMINAMATH_GPT_erased_number_is_30_l261_26170

-- Definitions based on conditions
def consecutiveNumbers (start n : ℕ) : List ℕ :=
  List.range' start n

def erase (l : List ℕ) (x : ℕ) : List ℕ :=
  List.filter (λ y => y ≠ x) l

def average (l : List ℕ) : ℚ :=
  l.sum / l.length

-- Statement to prove
theorem erased_number_is_30 :
  ∃ n x, average (erase (consecutiveNumbers 11 n) x) = 23 ∧ x = 30 := by
  sorry

end NUMINAMATH_GPT_erased_number_is_30_l261_26170


namespace NUMINAMATH_GPT_quadratic_has_integer_solutions_l261_26189

theorem quadratic_has_integer_solutions : 
  ∃ (s : Finset ℕ), ∀ a : ℕ, a ∈ s ↔ (1 ≤ a ∧ a ≤ 50 ∧ ((∃ n : ℕ, 4 * a + 1 = n^2))) ∧ s.card = 6 := 
  sorry

end NUMINAMATH_GPT_quadratic_has_integer_solutions_l261_26189


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_l261_26144

-- Define the condition that represents the family of lines
def family_of_lines (k : ℝ) (x y : ℝ) : Prop := k * x + y + 2 * k + 1 = 0

-- Formulate the theorem stating that (-2, -1) always lies on the line
theorem line_passes_through_fixed_point (k : ℝ) : family_of_lines k (-2) (-1) :=
by
  -- Proof skipped with sorry.
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_l261_26144


namespace NUMINAMATH_GPT_possible_values_a_possible_values_m_l261_26199

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - 1) * (x - a + 1) = 0}
noncomputable def C (m : ℝ) : Set ℝ := {x | x^2 - m * x + 2 = 0}

theorem possible_values_a (a : ℝ) : 
  (A ∪ B a = A) → a = 2 ∨ a = 3 := sorry

theorem possible_values_m (m : ℝ) : 
  (A ∩ C m = C m) → (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) := sorry

end NUMINAMATH_GPT_possible_values_a_possible_values_m_l261_26199


namespace NUMINAMATH_GPT_sqrt_sqrt_16_eq_pm2_l261_26110

theorem sqrt_sqrt_16_eq_pm2 (h : Real.sqrt 16 = 4) : Real.sqrt 4 = 2 ∨ Real.sqrt 4 = -2 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_sqrt_16_eq_pm2_l261_26110


namespace NUMINAMATH_GPT_find_first_set_length_l261_26178

def length_of_second_set : ℤ := 20
def ratio := 5

theorem find_first_set_length (x : ℤ) (h1 : length_of_second_set = ratio * x) : x = 4 := 
sorry

end NUMINAMATH_GPT_find_first_set_length_l261_26178
