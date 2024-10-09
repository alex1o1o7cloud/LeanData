import Mathlib

namespace Rachel_age_when_father_is_60_l1885_188562

-- Given conditions
def Rachel_age : ℕ := 12
def Grandfather_age : ℕ := 7 * Rachel_age
def Mother_age : ℕ := Grandfather_age / 2
def Father_age : ℕ := Mother_age + 5

-- Proof problem statement
theorem Rachel_age_when_father_is_60 : Rachel_age + (60 - Father_age) = 25 :=
by sorry

end Rachel_age_when_father_is_60_l1885_188562


namespace circle_diameter_l1885_188517

theorem circle_diameter (A : ℝ) (h : A = 64 * Real.pi) : ∃ d : ℝ, d = 16 :=
by
  sorry

end circle_diameter_l1885_188517


namespace circumcircle_eqn_l1885_188538

def point := ℝ × ℝ

def A : point := (-1, 5)
def B : point := (5, 5)
def C : point := (6, -2)

def circ_eq (D E F : ℝ) (x y : ℝ) : Prop := x^2 + y^2 + D * x + E * y + F = 0

theorem circumcircle_eqn :
  ∃ D E F : ℝ, (∀ (p : point), p ∈ [A, B, C] → circ_eq D E F p.1 p.2) ∧
              circ_eq (-4) (-2) (-20) = circ_eq D E F := by
  sorry

end circumcircle_eqn_l1885_188538


namespace part_one_part_two_l1885_188564

noncomputable def f (x a : ℝ) : ℝ :=
  x^2 - (a + 1/a) * x + 1

theorem part_one (x : ℝ) : f x (1/2) ≤ 0 ↔ (1/2 ≤ x ∧ x ≤ 2) :=
by
  sorry

theorem part_two (x a : ℝ) (h : a > 0) : 
  ((a < 1) → (f x a ≤ 0 ↔ (a ≤ x ∧ x ≤ 1/a))) ∧
  ((a > 1) → (f x a ≤ 0 ↔ (1/a ≤ x ∧ x ≤ a))) ∧
  ((a = 1) → (f x a ≤ 0 ↔ (x = 1))) :=
by
  sorry

end part_one_part_two_l1885_188564


namespace empty_to_occupied_ratio_of_spheres_in_cylinder_package_l1885_188546

theorem empty_to_occupied_ratio_of_spheres_in_cylinder_package
  (R : ℝ) 
  (volume_sphere : ℝ)
  (volume_cylinder : ℝ)
  (sphere_occupies_fraction : ∀ R : ℝ, volume_sphere = (2 / 3) * volume_cylinder) 
  (num_spheres : ℕ) 
  (h_num_spheres : num_spheres = 5) :
  (num_spheres : ℝ) * volume_sphere = (5 * (2 / 3) * π * R^3) → 
  volume_sphere = (4 / 3) * π * R^3 → 
  volume_cylinder = 2 * π * R^3 → 
  (volume_cylinder - volume_sphere) / volume_sphere = 1 / 2 := by 
  sorry

end empty_to_occupied_ratio_of_spheres_in_cylinder_package_l1885_188546


namespace roland_thread_length_l1885_188556

noncomputable def length_initial : ℝ := 12
noncomputable def length_two_thirds : ℝ := (2 / 3) * length_initial
noncomputable def length_increased : ℝ := length_initial + length_two_thirds
noncomputable def length_half_increased : ℝ := (1 / 2) * length_increased
noncomputable def length_total : ℝ := length_increased + length_half_increased
noncomputable def length_inches : ℝ := length_total / 2.54

theorem roland_thread_length : length_inches = 11.811 :=
by sorry

end roland_thread_length_l1885_188556


namespace solve_for_y_l1885_188577

theorem solve_for_y : 
  ∀ (y : ℚ), y = 45 / (8 - 3 / 7) → y = 315 / 53 :=
by
  intro y
  intro h
  -- proof steps would be placed here
  sorry

end solve_for_y_l1885_188577


namespace f_neg1_plus_f_2_l1885_188510

def f (x : Int) : Int :=
  if x = -3 then -1
  else if x = -2 then -5
  else if x = -1 then -2
  else if x = 0 then 0
  else if x = 1 then 2
  else if x = 2 then 1
  else if x = 3 then 4
  else 0  -- This handles x values not explicitly in the table, although technically unnecessary.

theorem f_neg1_plus_f_2 : f (-1) + f (2) = -1 := by
  sorry

end f_neg1_plus_f_2_l1885_188510


namespace lines_intersect_l1885_188503

structure Point where
  x : ℝ
  y : ℝ

def line1 (t : ℝ) : Point :=
  ⟨1 + 2 * t, 4 - 3 * t⟩

def line2 (u : ℝ) : Point :=
  ⟨5 + 4 * u, -2 - 5 * u⟩

theorem lines_intersect (x y t u : ℝ) 
  (h1 : x = 1 + 2 * t)
  (h2 : y = 4 - 3 * t)
  (h3 : x = 5 + 4 * u)
  (h4 : y = -2 - 5 * u) :
  x = 5 ∧ y = -2 := 
sorry

end lines_intersect_l1885_188503


namespace inequality_solution_set_l1885_188535

theorem inequality_solution_set :
  {x : ℝ | 2 * x^2 - x > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 1 / 2} :=
by
  sorry

end inequality_solution_set_l1885_188535


namespace geom_progression_vertex_ad_l1885_188569

theorem geom_progression_vertex_ad
  (a b c d : ℝ)
  (geom_prog : a * c = b * b ∧ b * d = c * c)
  (vertex : (b, c) = (1, 3)) :
  a * d = 3 :=
sorry

end geom_progression_vertex_ad_l1885_188569


namespace min_PA_squared_plus_PB_squared_l1885_188547

-- Let points A, B, and the circle be defined as given in the problem.
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨-2, 0⟩
def B : Point := ⟨2, 0⟩

def on_circle (P : Point) : Prop :=
  (P.x - 3)^2 + (P.y - 4)^2 = 4

def PA_squared (P : Point) : ℝ :=
  (P.x - A.x)^2 + (P.y - A.y)^2

def PB_squared (P : Point) : ℝ :=
  (P.x - B.x)^2 + (P.y - B.y)^2

def F (P : Point) : ℝ := PA_squared P + PB_squared P

theorem min_PA_squared_plus_PB_squared : ∃ P : Point, on_circle P ∧ F P = 26 := sorry

end min_PA_squared_plus_PB_squared_l1885_188547


namespace probability_A_seven_rolls_l1885_188543

noncomputable def probability_A_after_n_rolls (n : ℕ) : ℚ :=
  if n = 0 then 1 else 1/3 * (1 - (-1/2)^(n-1))

theorem probability_A_seven_rolls : probability_A_after_n_rolls 7 = 21 / 64 :=
by sorry

end probability_A_seven_rolls_l1885_188543


namespace sum_a_eq_9_l1885_188589

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sum_a_eq_9 (a2 a3 a4 a5 a6 a7 : ℤ) 
  (h1 : 0 ≤ a2 ∧ a2 < 2) (h2 : 0 ≤ a3 ∧ a3 < 3) (h3 : 0 ≤ a4 ∧ a4 < 4)
  (h4 : 0 ≤ a5 ∧ a5 < 5) (h5 : 0 ≤ a6 ∧ a6 < 6) (h6 : 0 ≤ a7 ∧ a7 < 7)
  (h_eq : (5 : ℚ) / 7 = (a2 : ℚ) / factorial 2 + (a3 : ℚ) / factorial 3 + (a4 : ℚ) / factorial 4 + 
                         (a5 : ℚ) / factorial 5 + (a6 : ℚ) / factorial 6 + (a7 : ℚ) / factorial 7) :
  a2 + a3 + a4 + a5 + a6 + a7 = 9 := 
sorry

end sum_a_eq_9_l1885_188589


namespace right_triangle_median_square_l1885_188560

theorem right_triangle_median_square (a b c k_a k_b : ℝ) :
  c = Real.sqrt (a^2 + b^2) → -- c is the hypotenuse
  k_a = Real.sqrt ((2 * b^2 + 2 * (a^2 + b^2) - a^2) / 4) → -- k_a is the median to side a
  k_b = Real.sqrt ((2 * a^2 + 2 * (a^2 + b^2) - b^2) / 4) → -- k_b is the median to side b
  c^2 = (4 / 5) * (k_a^2 + k_b^2) :=
by
  intros h_c h_ka h_kb
  sorry

end right_triangle_median_square_l1885_188560


namespace simple_interest_rate_l1885_188561

theorem simple_interest_rate (P A T : ℝ) (H1 : P = 1750) (H2 : A = 2000) (H3 : T = 4) :
  ∃ R : ℝ, R = 3.57 ∧ A = P * (1 + (R * T) / 100) :=
by
  sorry

end simple_interest_rate_l1885_188561


namespace special_blend_probability_l1885_188597

/-- Define the probability variables and conditions -/
def visit_count : ℕ := 6
def special_blend_prob : ℚ := 3 / 4
def non_special_blend_prob : ℚ := 1 / 4

/-- The binomial coefficient for choosing 5 days out of 6 -/
def choose_6_5 : ℕ := Nat.choose 6 5

/-- The probability of serving the special blend exactly 5 times out of 6 -/
def prob_special_blend_5 : ℚ := (choose_6_5 : ℚ) * (special_blend_prob ^ 5) * (non_special_blend_prob ^ 1)

/-- Statement to prove the desired probability -/
theorem special_blend_probability :
  prob_special_blend_5 = 1458 / 4096 :=
by
  sorry

end special_blend_probability_l1885_188597


namespace sum_of_abs_first_10_terms_l1885_188586

noncomputable def sum_of_first_n_terms (n : ℕ) : ℤ := n^2 - 5 * n + 2

theorem sum_of_abs_first_10_terms : 
  let S := sum_of_first_n_terms 10
  let S3 := sum_of_first_n_terms 3
  (S - 2 * S3) = 60 := 
by
  sorry

end sum_of_abs_first_10_terms_l1885_188586


namespace distinct_ordered_pairs_l1885_188530

theorem distinct_ordered_pairs (a b : ℕ) (h : a + b = 40) (ha : a > 0) (hb : b > 0) :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 39 ∧ ∀ p ∈ pairs, p.1 + p.2 = 40 := 
sorry

end distinct_ordered_pairs_l1885_188530


namespace miranda_pillows_l1885_188511

theorem miranda_pillows (feathers_per_pound : ℕ) (total_feathers : ℕ) (pillows : ℕ)
  (h1 : feathers_per_pound = 300) (h2 : total_feathers = 3600) (h3 : pillows = 6) :
  (total_feathers / feathers_per_pound) / pillows = 2 := by
  sorry

end miranda_pillows_l1885_188511


namespace solution_to_system_l1885_188545

theorem solution_to_system (x y z : ℝ) (h1 : x^2 + y^2 = 6 * z) (h2 : y^2 + z^2 = 6 * x) (h3 : z^2 + x^2 = 6 * y) :
  (x = 3) ∧ (y = 3) ∧ (z = 3) :=
sorry

end solution_to_system_l1885_188545


namespace opposite_of_negative_a_is_a_l1885_188567

-- Define the problem:
theorem opposite_of_negative_a_is_a (a : ℝ) : -(-a) = a :=
by 
  sorry

end opposite_of_negative_a_is_a_l1885_188567


namespace find_a_integer_condition_l1885_188551

theorem find_a_integer_condition (a : ℚ) :
  (∀ n : ℕ, (a * (n * (n+2) * (n+3) * (n+4)) : ℚ).den = 1) ↔ ∃ k : ℤ, a = k / 6 := 
sorry

end find_a_integer_condition_l1885_188551


namespace population_scientific_notation_l1885_188578

theorem population_scientific_notation : 
  (1.41: ℝ) * (10 ^ 9) = 1.41 * 10 ^ 9 := 
by
  sorry

end population_scientific_notation_l1885_188578


namespace find_a_l1885_188520

-- Define the real numbers x, y, and a
variables (x y a : ℝ)

-- Define the conditions as premises
axiom cond1 : x + 3 * y + 5 ≥ 0
axiom cond2 : x + y - 1 ≤ 0
axiom cond3 : x + a ≥ 0

-- Define z as x + 2y and state its minimum value is -4
def z : ℝ := x + 2 * y
axiom min_z : z = -4

-- The theorem to prove the value of a given the above conditions
theorem find_a : a = 2 :=
sorry

end find_a_l1885_188520


namespace inequality_proof_l1885_188533

theorem inequality_proof (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : x + y ≤ 1) : 
  8 * x * y ≤ 5 * x * (1 - x) + 5 * y * (1 - y) :=
sorry

end inequality_proof_l1885_188533


namespace distinct_real_solutions_l1885_188512

theorem distinct_real_solutions
  (a b c d e : ℝ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) :
  ∃ x₁ x₂ x₃ x₄ : ℝ,
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) ∧
    (x₁ - a) * (x₁ - b) * (x₁ - c) * (x₁ - d) +
    (x₁ - a) * (x₁ - b) * (x₁ - c) * (x₁ - e) +
    (x₁ - a) * (x₁ - b) * (x₁ - d) * (x₁ - e) +
    (x₁ - a) * (x₁ - c) * (x₁ - d) * (x₁ - e) +
    (x₁ - b) * (x₁ - c) * (x₁ - d) * (x₁ - e) = 0 ∧
    (x₂ - a) * (x₂ - b) * (x₂ - c) * (x₂ - d) +
    (x₂ - a) * (x₂ - b) * (x₂ - c) * (x₂ - e) +
    (x₂ - a) * (x₂ - b) * (x₂ - d) * (x₂ - e) +
    (x₂ - a) * (x₂ - c) * (x₂ - d) * (x₂ - e) +
    (x₂ - b) * (x₂ - c) * (x₂ - d) * (x₂ - e) = 0 ∧
    (x₃ - a) * (x₃ - b) * (x₃ - c) * (x₃ - d) +
    (x₃ - a) * (x₃ - b) * (x₃ - c) * (x₃ - e) +
    (x₃ - a) * (x₃ - b) * (x₃ - d) * (x₃ - e) +
    (x₃ - a) * (x₃ - c) * (x₃ - d) * (x₃ - e) +
    (x₃ - b) * (x₃ - c) * (x₃ - d) * (x₃ - e) = 0 ∧
    (x₄ - a) * (x₄ - b) * (x₄ - c) * (x₄ - d) +
    (x₄ - a) * (x₄ - b) * (x₄ - c) * (x₄ - e) +
    (x₄ - a) * (x₄ - b) * (x₄ - d) * (x₄ - e) +
    (x₄ - a) * (x₄ - c) * (x₄ - d) * (x₄ - e) +
    (x₄ - b) * (x₄ - c) * (x₄ - d) * (x₄ - e) = 0 :=
  sorry

end distinct_real_solutions_l1885_188512


namespace angle_between_sum_is_pi_over_6_l1885_188536

open Real EuclideanSpace

noncomputable def angle_between_vectors (u v : ℝ × ℝ) : ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_u := sqrt (u.1^2 + u.2^2)
  let norm_v := sqrt (v.1^2 + v.2^2)
  arccos (dot_product / (norm_u * norm_v))

noncomputable def a : ℝ × ℝ := (1, 0)
noncomputable def b : ℝ × ℝ := (1/2 * cos (π / 3), 1/2 * sin (π / 3))

theorem angle_between_sum_is_pi_over_6 :
  angle_between_vectors (a.1 + 2 * b.1, a.2 + 2 * b.2) b = π / 6 :=
by
  sorry

end angle_between_sum_is_pi_over_6_l1885_188536


namespace find_a_values_l1885_188513

noncomputable def function_a_max_value (a : ℝ) : ℝ :=
  a^2 + 2 * a - 9

theorem find_a_values (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : function_a_max_value a = 6) : 
    a = 3 ∨ a = 1/3 :=
  sorry

end find_a_values_l1885_188513


namespace ratio_of_patients_l1885_188570

def one_in_four_zx (current_patients : ℕ) : ℕ :=
  current_patients / 4

def previous_patients : ℕ :=
  26

def diagnosed_patients : ℕ :=
  13

def current_patients : ℕ :=
  diagnosed_patients * 4

theorem ratio_of_patients : 
  one_in_four_zx current_patients = diagnosed_patients → 
  (current_patients / previous_patients) = 2 := 
by 
  sorry

end ratio_of_patients_l1885_188570


namespace min_value_of_linear_combination_of_variables_l1885_188559

-- Define the conditions that x and y are positive numbers and satisfy the equation x + 3y = 5xy
def conditions (x y : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ x + 3 * y = 5 * x * y

-- State the theorem that the minimum value of 3x + 4y given the conditions is 5
theorem min_value_of_linear_combination_of_variables (x y : ℝ) (h: conditions x y) : 3 * x + 4 * y ≥ 5 :=
by 
  sorry

end min_value_of_linear_combination_of_variables_l1885_188559


namespace find_a_l1885_188576

-- Define the necessary variables
variables (a b : ℝ) (t : ℝ)

-- Given conditions
def b_val : ℝ := 2120
def t_val : ℝ := 0.5

-- The statement we need to prove
theorem find_a (h: b = b_val) (h2: t = t_val) (h3: t = a / b) : a = 1060 := by
  -- Placeholder for proof
  sorry

end find_a_l1885_188576


namespace non_neg_sum_sq_inequality_l1885_188595

theorem non_neg_sum_sq_inequality (a b c : ℝ) (h₀ : a ≥ 0) (h₁ : b ≥ 0) (h₂ : c ≥ 0) (h₃ : a + b + c = 1) :
  (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 :=
sorry

end non_neg_sum_sq_inequality_l1885_188595


namespace value_at_7_6_l1885_188541

noncomputable def f : ℝ → ℝ := sorry

lemma periodic_f (x : ℝ) : f (x + 4) = f x := sorry

lemma f_on_interval (x : ℝ) (hx : -2 ≤ x ∧ x ≤ 2) : f x = x := sorry

theorem value_at_7_6 : f 7.6 = -0.4 :=
by
  have p := periodic_f 7.6
  have q := periodic_f 3.6
  have r := f_on_interval (-0.4)
  sorry

end value_at_7_6_l1885_188541


namespace total_handshakes_l1885_188539

-- Definitions and conditions
def num_dwarves := 25
def num_elves := 18

def handshakes_among_dwarves : ℕ := num_dwarves * (num_dwarves - 1) / 2
def handshakes_between_dwarves_and_elves : ℕ := num_elves * num_dwarves

-- Total number of handshakes
theorem total_handshakes : handshakes_among_dwarves + handshakes_between_dwarves_and_elves = 750 := by 
  sorry

end total_handshakes_l1885_188539


namespace percent_change_area_decrease_l1885_188502

theorem percent_change_area_decrease (L W : ℝ) (hL : L > 0) (hW : W > 0) :
    let A_initial := L * W
    let L_new := 1.60 * L
    let W_new := 0.40 * W
    let A_new := L_new * W_new
    let percent_change := (A_new - A_initial) / A_initial * 100
    percent_change = -36 :=
by
  sorry

end percent_change_area_decrease_l1885_188502


namespace trajectory_of_Q_l1885_188550

/-- Let P(m, n) be a point moving on the circle x^2 + y^2 = 2.
     The trajectory of the point Q(m+n, 2mn) is y = x^2 - 2. -/
theorem trajectory_of_Q (m n : ℝ) (hyp : m^2 + n^2 = 2) : 
  ∃ x y : ℝ, x = m + n ∧ y = 2 * m * n ∧ y = x^2 - 2 :=
by
  sorry

end trajectory_of_Q_l1885_188550


namespace opposite_sides_line_l1885_188572

theorem opposite_sides_line (m : ℝ) :
  ( (3 * 3 - 2 * 1 + m) * (3 * (-4) - 2 * 6 + m) < 0 ) → (-7 < m ∧ m < 24) :=
by sorry

end opposite_sides_line_l1885_188572


namespace new_person_weight_l1885_188523

theorem new_person_weight 
    (W : ℝ) -- total weight of original 8 people
    (x : ℝ) -- weight of the new person
    (increase_by : ℝ) -- average weight increases by 2.5 kg
    (replaced_weight : ℝ) -- weight of the replaced person (55 kg)
    (h1 : increase_by = 2.5)
    (h2 : replaced_weight = 55)
    (h3 : x = replaced_weight + (8 * increase_by)) : x = 75 := 
by
  sorry

end new_person_weight_l1885_188523


namespace find_f_2011_l1885_188590

open Function

variable {R : Type} [Field R]

def functional_equation (f : R → R) : Prop :=
  ∀ a b : R, f (a * f b) = a * b

theorem find_f_2011 (f : ℝ → ℝ) (h : functional_equation f) : f 2011 = 2011 :=
sorry

end find_f_2011_l1885_188590


namespace checkerboard_sums_l1885_188596

-- Define the dimensions and the arrangement of the checkerboard
def n : ℕ := 10
def board (i j : ℕ) : ℕ := i * n + j + 1

-- Define corner positions
def top_left_corner : ℕ := board 0 0
def top_right_corner : ℕ := board 0 (n - 1)
def bottom_left_corner : ℕ := board (n - 1) 0
def bottom_right_corner : ℕ := board (n - 1) (n - 1)

-- Sum of the corners
def corner_sum : ℕ := top_left_corner + top_right_corner + bottom_left_corner + bottom_right_corner

-- Define the positions of the main diagonals
def main_diagonal (i : ℕ) : ℕ := board i i
def anti_diagonal (i : ℕ) : ℕ := board i (n - 1 - i)

-- Sum of the main diagonals
def diagonal_sum : ℕ := (Finset.range n).sum main_diagonal + (Finset.range n).sum anti_diagonal - (main_diagonal 0 + main_diagonal (n - 1))

-- Statement to prove
theorem checkerboard_sums : corner_sum = 202 ∧ diagonal_sum = 101 :=
by
-- Proof is not required as per the instructions
sorry

end checkerboard_sums_l1885_188596


namespace initial_amount_is_800_l1885_188529

variables (P R : ℝ)

theorem initial_amount_is_800
  (h1 : 956 = P * (1 + 3 * R / 100))
  (h2 : 1052 = P * (1 + 3 * (R + 4) / 100)) :
  P = 800 :=
sorry

end initial_amount_is_800_l1885_188529


namespace remainder_4032_125_l1885_188594

theorem remainder_4032_125 : 4032 % 125 = 32 := by
  sorry

end remainder_4032_125_l1885_188594


namespace abs_inequality_range_l1885_188519

theorem abs_inequality_range (a : ℝ) :
  (∀ x : ℝ, |x + 1| + |x + 6| > a) ↔ a < 5 :=
by
  sorry

end abs_inequality_range_l1885_188519


namespace hundredths_digit_of_power_l1885_188525

theorem hundredths_digit_of_power (n : ℕ) (h : n % 20 = 14) : 
  (8 ^ n % 1000) / 100 = 1 :=
by sorry

lemma test_power_hundredths_digit : (8 ^ 1234 % 1000) / 100 = 1 :=
hundredths_digit_of_power 1234 (by norm_num)

end hundredths_digit_of_power_l1885_188525


namespace distance_between_lines_correct_l1885_188521

noncomputable def distance_between_parallel_lines 
  (a b c₁ c₂ : ℝ) : ℝ :=
  |c₁ - c₂| / Real.sqrt (a^2 + b^2)

theorem distance_between_lines_correct :
  distance_between_parallel_lines 4 2 (-2) 1 = 3 * Real.sqrt 5 / 10 :=
by
  -- Proof steps would go here
  sorry

end distance_between_lines_correct_l1885_188521


namespace geometric_sequence_a3_l1885_188574

theorem geometric_sequence_a3 (
  a : ℕ → ℝ
) 
(h1 : a 1 = 1)
(h5 : a 5 = 16)
(h_geometric : ∀ (n : ℕ), a (n + 1) / a n = a 2 / a 1) :
a 3 = 4 := by
  sorry

end geometric_sequence_a3_l1885_188574


namespace sufficient_condition_inequalities_l1885_188592

theorem sufficient_condition_inequalities (x a : ℝ) :
  (¬ (a-4 < x ∧ x < a+4) → ¬ (1 < x ∧ x < 2)) ↔ -2 ≤ a ∧ a ≤ 5 :=
by
  sorry

end sufficient_condition_inequalities_l1885_188592


namespace average_weight_increase_l1885_188571

variable {W : ℝ} -- Total weight before replacement
variable {n : ℝ} -- Number of men in the group

theorem average_weight_increase
  (h1 : (W - 58 + 83) / n - W / n = 2.5) : n = 10 :=
by
  sorry

end average_weight_increase_l1885_188571


namespace interval_contains_solution_l1885_188598

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 2

theorem interval_contains_solution :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  sorry

end interval_contains_solution_l1885_188598


namespace calc_a_minus_3b_l1885_188514

noncomputable def a : ℂ := 5 - 3 * Complex.I
noncomputable def b : ℂ := 2 + 3 * Complex.I

theorem calc_a_minus_3b : a - 3 * b = -1 - 12 * Complex.I := by
  sorry

end calc_a_minus_3b_l1885_188514


namespace rectangle_area_l1885_188573

-- Define the vertices of the rectangle
def V1 : ℝ × ℝ := (-7, 1)
def V2 : ℝ × ℝ := (1, 1)
def V3 : ℝ × ℝ := (1, -6)
def V4 : ℝ × ℝ := (-7, -6)

-- Define the function to compute the area of the rectangle given the vertices
noncomputable def area_of_rectangle (A B C D : ℝ × ℝ) : ℝ :=
  let length := abs (B.1 - A.1)
  let width := abs (A.2 - D.2)
  length * width

-- The statement to prove
theorem rectangle_area : area_of_rectangle V1 V2 V3 V4 = 56 := by
  sorry

end rectangle_area_l1885_188573


namespace scientific_notation_of_935million_l1885_188504

theorem scientific_notation_of_935million :
  935000000 = 9.35 * 10 ^ 8 :=
  sorry

end scientific_notation_of_935million_l1885_188504


namespace find_initial_amount_l1885_188526

noncomputable def initial_amount (diff : ℝ) : ℝ :=
  diff / (1.4641 - 1.44)

theorem find_initial_amount
  (diff : ℝ)
  (h : diff = 964.0000000000146) :
  initial_amount diff = 40000 :=
by
  -- the steps to prove this can be added here later
  sorry

end find_initial_amount_l1885_188526


namespace exponent_equality_l1885_188537

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
sorry

end exponent_equality_l1885_188537


namespace three_digit_identical_divisible_by_37_l1885_188548

theorem three_digit_identical_divisible_by_37 (A : ℕ) (h : A ≤ 9) : 37 ∣ (111 * A) :=
sorry

end three_digit_identical_divisible_by_37_l1885_188548


namespace arithmetic_sequence_property_l1885_188585

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_property (h1 : is_arithmetic_sequence a)
                                     (h2 : a 3 + a 11 = 40) :
  a 6 - a 7 + a 8 = 20 :=
by
  sorry

end arithmetic_sequence_property_l1885_188585


namespace sum_of_remainders_mod_l1885_188593

theorem sum_of_remainders_mod (a b c : ℕ) (h1 : a % 53 = 31) (h2 : b % 53 = 22) (h3 : c % 53 = 7) :
  (a + b + c) % 53 = 7 :=
by
  sorry

end sum_of_remainders_mod_l1885_188593


namespace nonnegative_integer_solutions_l1885_188587

theorem nonnegative_integer_solutions (x : ℕ) :
  2 * x - 1 < 5 ↔ x = 0 ∨ x = 1 ∨ x = 2 := by
sorry

end nonnegative_integer_solutions_l1885_188587


namespace f_7_5_l1885_188528

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (x + 2) = -f x
axiom f_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem f_7_5 : f 7.5 = -0.5 := by
  sorry

end f_7_5_l1885_188528


namespace complex_division_result_l1885_188531

theorem complex_division_result :
  let z := (⟨0, 1⟩ - ⟨2, 0⟩) / (⟨1, 0⟩ + ⟨0, 1⟩ : ℂ)
  let a := z.re
  let b := z.im
  a + b = 1 :=
by
  sorry

end complex_division_result_l1885_188531


namespace problem_solution_l1885_188540

theorem problem_solution
  (a b : ℝ)
  (h_eqn : ∃ (a b : ℝ), 3 * a * a + 9 * a - 21 = 0 ∧ 3 * b * b + 9 * b - 21 = 0 )
  (h_vieta_sum : a + b = -3)
  (h_vieta_prod : a * b = -7) :
  (2 * a - 5) * (3 * b - 4) = 47 := 
by
  sorry

end problem_solution_l1885_188540


namespace stratified_sampling_students_l1885_188588

theorem stratified_sampling_students :
  let F := 1600
  let S := 1200
  let Sr := 800
  let sr := 20
  let f := (F * sr) / Sr
  let s := (S * sr) / Sr
  f + s = 70 :=
by
  let F := 1600
  let S := 1200
  let Sr := 800
  let sr := 20
  let f := (F * sr) / Sr
  let s := (S * sr) / Sr
  sorry

end stratified_sampling_students_l1885_188588


namespace tan_eq_243_deg_l1885_188558

theorem tan_eq_243_deg (n : ℤ) : -90 < n ∧ n < 90 ∧ Real.tan (n * Real.pi / 180) = Real.tan (243 * Real.pi / 180) ↔ n = 63 :=
by sorry

end tan_eq_243_deg_l1885_188558


namespace total_value_of_horse_and_saddle_l1885_188555

def saddle_value : ℝ := 12.5
def horse_value : ℝ := 7 * saddle_value

theorem total_value_of_horse_and_saddle : horse_value + saddle_value = 100 := by
  sorry

end total_value_of_horse_and_saddle_l1885_188555


namespace final_elephants_count_l1885_188565

def E_0 : Int := 30000
def R_exodus : Int := 2880
def H_exodus : Int := 4
def R_entry : Int := 1500
def H_entry : Int := 7
def E_final : Int := E_0 - (R_exodus * H_exodus) + (R_entry * H_entry)

theorem final_elephants_count : E_final = 28980 := by
  sorry

end final_elephants_count_l1885_188565


namespace center_of_circle_is_at_10_3_neg5_l1885_188524

noncomputable def center_of_tangent_circle (x y : ℝ) : Prop :=
  (6 * x - 5 * y = 50 ∨ 6 * x - 5 * y = -20) ∧ (3 * x + 2 * y = 0)

theorem center_of_circle_is_at_10_3_neg5 :
  ∃ x y : ℝ, center_of_tangent_circle x y ∧ x = 10 / 3 ∧ y = -5 :=
by
  sorry

end center_of_circle_is_at_10_3_neg5_l1885_188524


namespace running_race_l1885_188554

-- Define participants
inductive Participant : Type
| Anna
| Bella
| Csilla
| Dora

open Participant

-- Define positions
@[ext] structure Position :=
(first : Participant)
(last : Participant)

-- Conditions:
def conditions (p : Participant) (q : Participant) (r : Participant) (s : Participant)
  (pa : Position) : Prop :=
  (pa.first = r) ∧ -- Csilla was first
  (pa.first ≠ q) ∧ -- Bella was not first
  (pa.first ≠ p) ∧ (pa.last ≠ p) ∧ -- Anna was not first or last
  (pa.last = s) -- Dóra's statement about being last

-- Definition of the liar
def liar (p : Participant) : Prop :=
  p = Dora

-- Proof problem
theorem running_race : ∃ (pa : Position), liar Dora ∧ (pa.first = Csilla) :=
  sorry

end running_race_l1885_188554


namespace arcsin_sqrt_three_over_two_l1885_188591

theorem arcsin_sqrt_three_over_two : 
  ∃ θ, θ = Real.arcsin (Real.sqrt 3 / 2) ∧ θ = Real.pi / 3 :=
by
  sorry

end arcsin_sqrt_three_over_two_l1885_188591


namespace mod_exp_equivalence_l1885_188552

theorem mod_exp_equivalence :
  (81^1814 - 25^1814) % 7 = 0 := by
  sorry

end mod_exp_equivalence_l1885_188552


namespace g_at_0_eq_1_l1885_188505

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g (x + y) = g x * g y
axiom g_deriv_at_0 : deriv g 0 = 2

theorem g_at_0_eq_1 : g 0 = 1 :=
by
  sorry

end g_at_0_eq_1_l1885_188505


namespace paint_coverage_is_10_l1885_188508

noncomputable def paintCoverage (cost_per_quart : ℝ) (cube_edge_length : ℝ) (total_cost : ℝ) : ℝ :=
  let total_surface_area := 6 * (cube_edge_length ^ 2)
  let number_of_quarts := total_cost / cost_per_quart
  total_surface_area / number_of_quarts

theorem paint_coverage_is_10 :
  paintCoverage 3.2 10 192 = 10 :=
by
  sorry

end paint_coverage_is_10_l1885_188508


namespace isosceles_base_length_l1885_188575

theorem isosceles_base_length (x b : ℕ) (h1 : 2 * x + b = 40) (h2 : x = 15) : b = 10 :=
by
  sorry

end isosceles_base_length_l1885_188575


namespace device_identification_l1885_188516

def sum_of_device_numbers (numbers : List ℕ) : ℕ :=
  numbers.foldr (· + ·) 0

def is_standard_device (d : List ℕ) : Prop :=
  (d = [1, 2, 3, 4, 5, 6, 7, 8, 9]) ∧ (sum_of_device_numbers d = 45)

theorem device_identification (d : List ℕ) : 
  (sum_of_device_numbers d = 45) → is_standard_device d :=
by
  sorry

end device_identification_l1885_188516


namespace correct_calculation_l1885_188500

theorem correct_calculation (a b : ℝ) :
  2 * a^2 * b - 3 * a^2 * b = -a^2 * b ∧
  ¬ (a^3 * a^4 = a^12) ∧
  ¬ ((-2 * a^2 * b)^3 = -6 * a^6 * b^3) ∧
  ¬ ((a + b)^2 = a^2 + b^2) :=
by
  sorry

end correct_calculation_l1885_188500


namespace sufficient_but_not_necessary_condition_l1885_188582

def condition_p (x : ℝ) : Prop := x^2 - 3*x + 2 < 0
def condition_q (x : ℝ) : Prop := |x - 2| < 1

theorem sufficient_but_not_necessary_condition : 
  (∀ x : ℝ, condition_p x → condition_q x) ∧ ¬(∀ x : ℝ, condition_q x → condition_p x) :=
by 
  sorry

end sufficient_but_not_necessary_condition_l1885_188582


namespace line_equation_from_point_normal_l1885_188566

theorem line_equation_from_point_normal :
  let M1 : ℝ × ℝ := (7, -8)
  let n : ℝ × ℝ := (-2, 3)
  ∃ C : ℝ, ∀ x y : ℝ, 2 * x - 3 * y + C = 0 ↔ (C = -38) := 
by
  sorry

end line_equation_from_point_normal_l1885_188566


namespace range_of_k_condition_l1885_188581

noncomputable def inverse_proportion_function (k x : ℝ) : ℝ := (4 - k) / x

theorem range_of_k_condition (k x1 x2 y1 y2 : ℝ) 
    (h1 : x1 < 0) (h2 : 0 < x2) (h3 : y1 < y2) 
    (hA : inverse_proportion_function k x1 = y1) 
    (hB : inverse_proportion_function k x2 = y2) : 
    k < 4 :=
sorry

end range_of_k_condition_l1885_188581


namespace solve_equation_l1885_188557

def f (x : ℝ) := |3 * x - 2|

theorem solve_equation 
  (x : ℝ) 
  (a : ℝ)
  (hx1 : x ≠ 3)
  (hx2 : x ≠ 0) :
  (3 * x - 2) ^ 2 = (x + a) ^ 2 ↔
  (a = -4 * x + 2) ∨ (a = 2 * x - 2) := by
  sorry

end solve_equation_l1885_188557


namespace profit_percentage_each_portion_l1885_188518

theorem profit_percentage_each_portion (P : ℝ) (total_apples : ℝ) 
  (portion1_percentage : ℝ) (portion2_percentage : ℝ) (total_profit_percentage : ℝ) :
  total_apples = 280 →
  portion1_percentage = 0.4 →
  portion2_percentage = 0.6 →
  total_profit_percentage = 0.3 →
  portion1_percentage * P + portion2_percentage * P = total_profit_percentage →
  P = 0.3 :=
by
  intros
  sorry

end profit_percentage_each_portion_l1885_188518


namespace train_length_proof_l1885_188506

-- Define the conditions
def time_to_cross := 12 -- Time in seconds
def speed_km_per_h := 75 -- Speed in km/h

-- Convert the speed to m/s
def speed_m_per_s := speed_km_per_h * (5 / 18 : ℚ)

-- The length of the train using the formula: length = speed * time
def length_of_train := speed_m_per_s * (time_to_cross : ℚ)

-- The theorem to prove
theorem train_length_proof : length_of_train = 250 := by
  sorry

end train_length_proof_l1885_188506


namespace c_minus_a_value_l1885_188542

theorem c_minus_a_value (a b c : ℝ) 
  (h1 : (a + b) / 2 = 50)
  (h2 : (b + c) / 2 = 70) : 
  c - a = 40 :=
by 
  sorry

end c_minus_a_value_l1885_188542


namespace hexagon_angle_Q_l1885_188563

theorem hexagon_angle_Q
  (a1 a2 a3 a4 a5 : ℝ)
  (h1 : a1 = 134) 
  (h2 : a2 = 98) 
  (h3 : a3 = 120) 
  (h4 : a4 = 110) 
  (h5 : a5 = 96) 
  (sum_hexagon_angles : a1 + a2 + a3 + a4 + a5 + Q = 720) : 
  Q = 162 := by {
  sorry
}

end hexagon_angle_Q_l1885_188563


namespace common_root_values_max_n_and_a_range_l1885_188549

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - (a+1) * x - 4 * (a+5)
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x^2 - x + 5

-- Part 1
theorem common_root_values (a : ℝ) :
  (∃ x : ℝ, f a x = 0 ∧ g a x = 0) → a = -9/16 ∨ a = -6 ∨ a = -4 ∨ a = 0 :=
sorry

-- Part 2
theorem max_n_and_a_range (a : ℝ) (m n : ℕ) (x0 : ℝ) :
  (m < n ∧ (m : ℝ) < x0 ∧ x0 < (n : ℝ) ∧ f a x0 < 0 ∧ g a x0 < 0) →
  n = 4 ∧ -1 ≤ a ∧ a ≤ -2/9 :=
sorry

end common_root_values_max_n_and_a_range_l1885_188549


namespace middle_group_frequency_l1885_188599

theorem middle_group_frequency (capacity : ℕ) (n_rectangles : ℕ) (A_mid A_other : ℝ) 
  (h_capacity : capacity = 300)
  (h_rectangles : n_rectangles = 9)
  (h_areas : A_mid = 1 / 5 * A_other)
  (h_total_area : A_mid + A_other = 1) : 
  capacity * A_mid = 50 := by
  sorry

end middle_group_frequency_l1885_188599


namespace sum_of_two_numbers_l1885_188527

theorem sum_of_two_numbers (x y : ℕ) (h1 : y = x + 4) (h2 : y = 30) : x + y = 56 :=
by
  -- Asserts the conditions and goal statement
  sorry

end sum_of_two_numbers_l1885_188527


namespace inequality_abc_l1885_188532

theorem inequality_abc (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_abc : a * b * c = 1) :
  1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 := 
by {
  sorry
}

end inequality_abc_l1885_188532


namespace tracy_initial_candies_l1885_188568

noncomputable def initial_candies : Nat := 80

theorem tracy_initial_candies
  (x : Nat)
  (hx1 : ∃ y : Nat, (1 ≤ y ∧ y ≤ 6) ∧ x = (5 * (44 + y)) / 3)
  (hx2 : x % 20 = 0) : x = initial_candies := by
  sorry

end tracy_initial_candies_l1885_188568


namespace percentage_both_colors_l1885_188583

theorem percentage_both_colors
  (total_flags : ℕ)
  (even_flags : total_flags % 2 = 0)
  (C : ℕ)
  (total_flags_eq : total_flags = 2 * C)
  (blue_percent : ℕ)
  (blue_percent_eq : blue_percent = 60)
  (red_percent : ℕ)
  (red_percent_eq : red_percent = 65) :
  ∃ both_colors_percent : ℕ, both_colors_percent = 25 :=
by
  sorry

end percentage_both_colors_l1885_188583


namespace muffins_per_person_l1885_188501

-- Definitions based on conditions
def total_friends : ℕ := 4
def total_people : ℕ := 1 + total_friends
def total_muffins : ℕ := 20

-- Theorem statement for the proof
theorem muffins_per_person : total_muffins / total_people = 4 := by
  sorry

end muffins_per_person_l1885_188501


namespace find_functions_l1885_188553

-- Define the function f and its properties.
variable {f : ℝ → ℝ}

-- Define the condition given in the problem as a hypothesis.
def condition (f : ℝ → ℝ) :=
  ∀ x y : ℝ, f (x * f x + f y) = y + f x ^ 2

-- State the theorem we want to prove.
theorem find_functions (hf : condition f) : (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
  sorry

end find_functions_l1885_188553


namespace B2F_base16_to_base10_l1885_188522

theorem B2F_base16_to_base10 :
  let d2 := 11
  let d1 := 2
  let d0 := 15
  d2 * 16^2 + d1 * 16^1 + d0 * 16^0 = 2863 :=
by
  let d2 := 11
  let d1 := 2
  let d0 := 15
  sorry

end B2F_base16_to_base10_l1885_188522


namespace fraction_difference_l1885_188584

theorem fraction_difference : 
  (2 + 4 + 6 + 8) / (1 + 3 + 5 + 7) - (1 + 3 + 5 + 7) / (2 + 4 + 6 + 8) = 9 / 20 := 
  sorry

end fraction_difference_l1885_188584


namespace inequality_solution_range_l1885_188507

theorem inequality_solution_range (a : ℝ) :
  (∃ (x : ℝ), |x + 1| - |x - 2| < a^2 - 4 * a) → (a > 3 ∨ a < 1) :=
by
  sorry

end inequality_solution_range_l1885_188507


namespace ratio_sprite_to_coke_l1885_188544

theorem ratio_sprite_to_coke (total_drink : ℕ) (coke_ounces : ℕ) (mountain_dew_parts : ℕ)
  (parts_coke : ℕ) (parts_mountain_dew : ℕ) (total_parts : ℕ) :
  total_drink = 18 →
  coke_ounces = 6 →
  parts_coke = 2 →
  parts_mountain_dew = 3 →
  total_parts = parts_coke + parts_mountain_dew + ((total_drink - coke_ounces - (parts_mountain_dew * (coke_ounces / parts_coke))) / (coke_ounces / parts_coke)) →
  (total_drink - coke_ounces - (parts_mountain_dew * (coke_ounces / parts_coke))) / coke_ounces = 1 / 2 :=
by sorry

end ratio_sprite_to_coke_l1885_188544


namespace vector_perpendicular_sets_l1885_188579

-- Define the problem in Lean
theorem vector_perpendicular_sets (x : ℝ) : 
  let a := (Real.sin x, Real.cos x)
  let b := (Real.sin x + Real.cos x, Real.sin x - Real.cos x)
  a.1 * b.1 + a.2 * b.2 = 0 ↔ ∃ (k : ℤ), x = k * (π / 2) + (π / 8) :=
sorry

end vector_perpendicular_sets_l1885_188579


namespace ratio_proof_l1885_188534

theorem ratio_proof (x y z s : ℝ) (h1 : x < y) (h2 : y < z)
    (h3 : (x : ℝ) / y = y / z) (h4 : x + y + z = s) (h5 : x + y = z) :
    (x / y = (-1 + Real.sqrt 5) / 2) :=
by
  sorry

end ratio_proof_l1885_188534


namespace math_problem_l1885_188515
open Real

noncomputable def problem_statement : Prop :=
  let a := 99
  let b := 3
  let c := 20
  let area := (99 * sqrt 3) / 20
  a + b + c = 122 ∧ 
  ∃ (AB: ℝ) (QR: ℝ), AB = 14 ∧ QR = 3 * sqrt 3 ∧ area = (1 / 2) * QR * (QR / (2 * (sqrt 3 / 2))) * (sqrt 3 / 2)

theorem math_problem : problem_statement := by
  sorry

end math_problem_l1885_188515


namespace unique_intersection_l1885_188509

def line1 (x y : ℝ) : Prop := 3 * x - 2 * y - 9 = 0
def line2 (x y : ℝ) : Prop := 6 * x + 4 * y - 12 = 0
def line3 (x : ℝ) : Prop := x = 3
def line4 (y : ℝ) : Prop := y = -1

theorem unique_intersection : ∃! p : ℝ × ℝ, 
                             (line1 p.1 p.2) ∧ 
                             (line2 p.1 p.2) ∧ 
                             (line3 p.1) ∧ 
                             (line4 p.2) ∧ 
                             p = (3, -1) :=
by
  sorry

end unique_intersection_l1885_188509


namespace carlson_handkerchief_usage_l1885_188580

def problem_statement : Prop :=
  let handkerchief_area := 25 * 25 -- Area in cm²
  let total_fabric_area := 3 * 10000 -- Total fabric area in cm²
  let days := 8
  let total_handkerchiefs := total_fabric_area / handkerchief_area
  let handkerchiefs_per_day := total_handkerchiefs / days
  handkerchiefs_per_day = 6

theorem carlson_handkerchief_usage : problem_statement := by
  sorry

end carlson_handkerchief_usage_l1885_188580
