import Mathlib

namespace solve_for_a_l1783_178300

theorem solve_for_a (a b : ℝ) (h1 : b = 4 * a) (h2 : b = 16 - 6 * a + a ^ 2) : 
  a = -5 + Real.sqrt 41 ∨ a = -5 - Real.sqrt 41 := by
  sorry

end solve_for_a_l1783_178300


namespace locus_of_point_P_l1783_178379

/-- Given three points in the coordinate plane A(0,3), B(-√3, 0), and C(√3, 0), 
    and a point P on the coordinate plane such that PA = PB + PC, 
    determine the equation of the locus of point P. -/
noncomputable def locus_equation : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | (P.1^2 + (P.2 - 1)^2 = 4) ∧ (P.2 ≤ 0)}

theorem locus_of_point_P :
  ∀ (P : ℝ × ℝ),
  (∃ A B C : ℝ × ℝ, A = (0, 3) ∧ B = (-Real.sqrt 3, 0) ∧ C = (Real.sqrt 3, 0) ∧ 
     dist P A = dist P B + dist P C) →
  P ∈ locus_equation :=
by
  intros P hp
  sorry

end locus_of_point_P_l1783_178379


namespace eq_correct_l1783_178374

variable (x : ℝ)

def width (x : ℝ) : ℝ := x - 6

def area_eq (x : ℝ) : Prop := x * width x = 720

theorem eq_correct (h : area_eq x) : x * (x - 6) = 720 :=
by exact h

end eq_correct_l1783_178374


namespace line_through_center_parallel_to_given_line_l1783_178310

def point_in_line (p : ℝ × ℝ) (a b c : ℝ) : Prop :=
  a * p.1 + b * p.2 + c = 0

noncomputable def slope_of_line (a b c : ℝ) : ℝ :=
  -a / b

theorem line_through_center_parallel_to_given_line :
  ∃ a b c : ℝ, a = 2 ∧ b = -1 ∧ c = -4 ∧
    point_in_line (2, 0) a b c ∧
    slope_of_line a b c = slope_of_line 2 (-1) 1 :=
by
  sorry

end line_through_center_parallel_to_given_line_l1783_178310


namespace fourth_angle_of_quadrilateral_l1783_178320

theorem fourth_angle_of_quadrilateral (A : ℝ) : 
  (120 + 85 + 90 + A = 360) ↔ A = 65 := 
by
  sorry

end fourth_angle_of_quadrilateral_l1783_178320


namespace lassis_from_mangoes_l1783_178376

theorem lassis_from_mangoes (mangoes lassis mangoes' lassis' : ℕ) 
  (h1 : lassis = (8 * mangoes) / 3)
  (h2 : mangoes = 15) :
  lassis = 40 :=
by
  sorry

end lassis_from_mangoes_l1783_178376


namespace lost_card_number_l1783_178396

theorem lost_card_number (n : ℕ) (x : ℕ) (h₁ : (n * (n + 1)) / 2 - x = 101) : x = 4 :=
sorry

end lost_card_number_l1783_178396


namespace graph_not_passing_through_origin_l1783_178341

theorem graph_not_passing_through_origin (m : ℝ) (h : 3 * m^2 - 2 * m ≠ 0) : m = -(1 / 3) :=
sorry

end graph_not_passing_through_origin_l1783_178341


namespace flowchart_output_proof_l1783_178352

def flowchart_output (x : ℕ) : ℕ :=
  let x := x + 2
  let x := x + 2
  let x := x + 2
  x

theorem flowchart_output_proof :
  flowchart_output 10 = 16 := by
  -- Assume initial value of x is 10
  let x0 := 10
  -- First iteration
  let x1 := x0 + 2
  -- Second iteration
  let x2 := x1 + 2
  -- Third iteration
  let x3 := x2 + 2
  -- Final value of x
  have hx_final : x3 = 16 := by rfl
  -- The result should be 16
  have h_result : flowchart_output 10 = x3 := by rfl
  rw [hx_final] at h_result
  exact h_result

end flowchart_output_proof_l1783_178352


namespace geometric_sequence_q_and_an_l1783_178395

theorem geometric_sequence_q_and_an
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_pos : q > 0)
  (h2_eq : a 2 = 1)
  (h2_h6_eq_9h4 : a 2 * a 6 = 9 * a 4) :
  q = 3 ∧ ∀ n, a n = 3^(n - 2) := by
sorry

end geometric_sequence_q_and_an_l1783_178395


namespace evaluate_expression_l1783_178385

theorem evaluate_expression : 68 + (126 / 18) + (35 * 13) - 300 - (420 / 7) = 170 := by
  sorry

end evaluate_expression_l1783_178385


namespace roots_quadratic_identity_l1783_178356

theorem roots_quadratic_identity (p q : ℝ) (r s : ℝ) (h1 : r + s = 3 * p) (h2 : r * s = 2 * q) :
  r^2 + s^2 = 9 * p^2 - 4 * q := 
by 
  sorry

end roots_quadratic_identity_l1783_178356


namespace ellipse_equation_and_line_intersection_unique_l1783_178388

-- Definitions from conditions
def ellipse (x y : ℝ) : Prop := (x^2)/4 + (y^2)/3 = 1
def line (x0 y0 x y : ℝ) : Prop := 3*x0*x + 4*y0*y - 12 = 0
def on_ellipse (x0 y0 : ℝ) : Prop := ellipse x0 y0

theorem ellipse_equation_and_line_intersection_unique :
  ∀ (x0 y0 : ℝ), on_ellipse x0 y0 → ∀ (x y : ℝ), line x0 y0 x y → ellipse x y → x = x0 ∧ y = y0 :=
by
  sorry

end ellipse_equation_and_line_intersection_unique_l1783_178388


namespace find_x_plus_inv_x_l1783_178316

theorem find_x_plus_inv_x (x : ℝ) (hx_pos : 0 < x) (h : x^10 + x^5 + 1/x^5 + 1/x^10 = 15250) :
  x + 1/x = 3 :=
by
  sorry

end find_x_plus_inv_x_l1783_178316


namespace hannahs_peppers_total_weight_l1783_178349

theorem hannahs_peppers_total_weight:
  let green := 0.3333333333333333
  let red := 0.3333333333333333
  let yellow := 0.25
  let orange := 0.5
  green + red + yellow + orange = 1.4166666666666665 :=
by
  repeat { sorry } -- Placeholder for the actual proof

end hannahs_peppers_total_weight_l1783_178349


namespace tim_morning_running_hours_l1783_178308

theorem tim_morning_running_hours 
  (runs_per_week : ℕ) 
  (total_hours_per_week : ℕ) 
  (runs_per_day : ℕ → ℕ) 
  (hrs_per_day_morning_evening_equal : ∀ (d : ℕ), runs_per_day d = runs_per_week * total_hours_per_week / runs_per_week) 
  (hrs_per_day : ℕ) 
  (hrs_per_morning : ℕ) 
  (hrs_per_evening : ℕ) 
  : hrs_per_morning = 1 :=
by 
  -- Given conditions
  have hrs_per_day := total_hours_per_week / runs_per_week
  have hrs_per_morning_evening := hrs_per_day / 2
  -- Conclusion
  sorry

end tim_morning_running_hours_l1783_178308


namespace shark_ratio_l1783_178321

theorem shark_ratio (N D : ℕ) (h1 : N = 22) (h2 : D + N = 110) (h3 : ∃ x : ℕ, D = x * N) : 
  (D / N) = 4 :=
by
  -- conditions use only definitions given in the problem.
  sorry

end shark_ratio_l1783_178321


namespace ending_number_of_range_l1783_178381

/-- The sum of the first n consecutive odd integers is n^2. -/
def sum_first_n_odd : ℕ → ℕ 
| 0       => 0
| (n + 1) => (2 * n + 1) + sum_first_n_odd n

/-- The sum of all odd integers between 11 and the ending number is 416. -/
def sum_odd_integers (a b : ℕ) : ℕ :=
  let s := (1 + b) / 2 - (1 + a) / 2 + 1
  sum_first_n_odd s

theorem ending_number_of_range (n : ℕ) (h1 : sum_first_n_odd n = n^2) 
  (h2 : sum_odd_integers 11 n = 416) : 
  n = 67 :=
sorry

end ending_number_of_range_l1783_178381


namespace polynomial_divisibility_l1783_178302

theorem polynomial_divisibility (C D : ℝ) (h : ∀ (ω : ℂ), ω^2 + ω + 1 = 0 → (ω^106 + C * ω + D = 0)) : C + D = -1 :=
by
  -- Add proof here
  sorry

end polynomial_divisibility_l1783_178302


namespace exists_m_n_for_any_d_l1783_178366

theorem exists_m_n_for_any_d (d : ℤ) : ∃ m n : ℤ, d = (n - 2 * m + 1) / (m^2 - n) :=
by
  sorry

end exists_m_n_for_any_d_l1783_178366


namespace smallest_n_l1783_178307

theorem smallest_n (n : ℕ) (hn : 0 < n) (h : 253 * n % 15 = 989 * n % 15) : n = 15 := by
  sorry

end smallest_n_l1783_178307


namespace smallest_d_for_inverse_l1783_178351

noncomputable def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 7

theorem smallest_d_for_inverse : ∃ d : ℝ, (∀ x1 x2, x1 ≥ d → x2 ≥ d → g x1 = g x2 → x1 = x2) ∧ d = 3 := 
sorry

end smallest_d_for_inverse_l1783_178351


namespace probability_correct_digit_in_two_attempts_l1783_178339

theorem probability_correct_digit_in_two_attempts :
  let total_digits := 10
  let probability_first_correct := 1 / total_digits
  let probability_first_incorrect := 9 / total_digits
  let probability_second_correct_if_first_incorrect := 1 / (total_digits - 1)
  (probability_first_correct + probability_first_incorrect * probability_second_correct_if_first_incorrect) = 1 / 5 := 
sorry

end probability_correct_digit_in_two_attempts_l1783_178339


namespace cube_ratio_sum_l1783_178340

theorem cube_ratio_sum (a b : ℝ) (h1 : |a| ≠ |b|) (h2 : (a + b) / (a - b) + (a - b) / (a + b) = 6) :
  (a^3 + b^3) / (a^3 - b^3) + (a^3 - b^3) / (a^3 + b^3) = 18 / 7 :=
by
  sorry

end cube_ratio_sum_l1783_178340


namespace subcommittee_count_l1783_178311

theorem subcommittee_count :
  let total_members := 12
  let total_teachers := 5
  let subcommittee_size := 5
  let total_subcommittees := Nat.choose total_members subcommittee_size
  let non_teacher_subcommittees_with_0_teachers := Nat.choose (total_members - total_teachers) subcommittee_size
  let non_teacher_subcommittees_with_1_teacher :=
    Nat.choose total_teachers 1 * Nat.choose (total_members - total_teachers) (subcommittee_size - 1)
  (total_subcommittees
   - (non_teacher_subcommittees_with_0_teachers + non_teacher_subcommittees_with_1_teacher)) = 596 := 
by
  sorry

end subcommittee_count_l1783_178311


namespace cost_of_paving_floor_l1783_178384

-- Define the constants given in the problem
def length1 : ℝ := 5.5
def width1 : ℝ := 3.75
def length2 : ℝ := 4
def width2 : ℝ := 3
def cost_per_sq_meter : ℝ := 800

-- Define the areas of the two rectangles
def area1 : ℝ := length1 * width1
def area2 : ℝ := length2 * width2

-- Define the total area of the floor
def total_area : ℝ := area1 + area2

-- Define the total cost of paving the floor
def total_cost : ℝ := total_area * cost_per_sq_meter

-- The statement to prove: the total cost equals 26100 Rs
theorem cost_of_paving_floor : total_cost = 26100 := by
  -- Proof skipped
  sorry

end cost_of_paving_floor_l1783_178384


namespace geometric_sequence_sum_l1783_178380

theorem geometric_sequence_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (t : ℝ) (n : ℕ) (hS : ∀ n, S n = t - 3 * 2^n) (h_geom : ∀ n, a (n + 1) = a n * r) :
  t = 3 :=
by
  sorry

end geometric_sequence_sum_l1783_178380


namespace solution_exists_l1783_178309

theorem solution_exists (x : ℝ) : (x - 1)^2 = 4 → (x = 3 ∨ x = -1) :=
by
  sorry

end solution_exists_l1783_178309


namespace doctor_lawyer_ratio_l1783_178367

variables {d l : ℕ} -- Number of doctors and lawyers

-- Conditions
def avg_age_group (d l : ℕ) : Prop := (40 * d + 55 * l) / (d + l) = 45

-- Theorem: Given the conditions, the ratio of doctors to lawyers is 2:1.
theorem doctor_lawyer_ratio (hdl : avg_age_group d l) : d / l = 2 :=
sorry

end doctor_lawyer_ratio_l1783_178367


namespace simplify_fraction_l1783_178347

variable (c : ℝ)

theorem simplify_fraction :
  (6 + 2 * c) / 7 + 3 = (27 + 2 * c) / 7 := 
by 
  sorry

end simplify_fraction_l1783_178347


namespace slab_length_l1783_178325

noncomputable def area_of_one_slab (total_area: ℝ) (num_slabs: ℕ) : ℝ :=
  total_area / num_slabs

noncomputable def length_of_one_slab (slab_area : ℝ) : ℝ :=
  Real.sqrt slab_area

theorem slab_length (total_area : ℝ) (num_slabs : ℕ)
  (h_total_area : total_area = 98)
  (h_num_slabs : num_slabs = 50) :
  length_of_one_slab (area_of_one_slab total_area num_slabs) = 1.4 :=
by
  sorry

end slab_length_l1783_178325


namespace minimum_prime_product_l1783_178327

noncomputable def is_prime : ℕ → Prop := sorry -- Assume the definition of prime

theorem minimum_prime_product (m n p : ℕ) 
  (hm : is_prime m) 
  (hn : is_prime n) 
  (hp : is_prime p) 
  (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p)
  (h_sum : m + n = p) : 
  m * n * p = 30 :=
sorry

end minimum_prime_product_l1783_178327


namespace set_intersection_is_result_l1783_178360

def set_A := {x : ℝ | 1 < x^2 ∧ x^2 < 4 }
def set_B := {x : ℝ | x ≥ 1}
def result_set := {x : ℝ | 1 < x ∧ x < 2}

theorem set_intersection_is_result : (set_A ∩ set_B) = result_set :=
by sorry

end set_intersection_is_result_l1783_178360


namespace f_m_plus_1_positive_l1783_178364

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + x + a

theorem f_m_plus_1_positive {m a : ℝ} (h_a_pos : a > 0) (h_f_m_neg : f m a < 0) : f (m + 1) a > 0 := by
  sorry

end f_m_plus_1_positive_l1783_178364


namespace largest_inscribed_rectangle_area_l1783_178326

theorem largest_inscribed_rectangle_area : 
  ∀ (width length : ℝ) (a b : ℝ), 
  width = 8 → length = 12 → 
  (a = (8 / Real.sqrt 3) ∧ b = 2 * a) → 
  (area : ℝ) = (12 * (8 - a)) → 
  area = (96 - 32 * Real.sqrt 3) :=
by
  intros width length a b hw hl htr harea
  sorry

end largest_inscribed_rectangle_area_l1783_178326


namespace smallest_perfect_square_divisible_by_4_and_5_l1783_178314

theorem smallest_perfect_square_divisible_by_4_and_5 : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (m : ℕ), n = m * m) ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ (n = 400) := 
by
  sorry

end smallest_perfect_square_divisible_by_4_and_5_l1783_178314


namespace minibus_children_count_l1783_178334

theorem minibus_children_count
  (total_seats : ℕ)
  (seats_with_3_children : ℕ)
  (seats_with_2_children : ℕ)
  (children_per_seat_3 : ℕ)
  (children_per_seat_2 : ℕ)
  (h_seats_count : total_seats = 7)
  (h_seats_distribution : seats_with_3_children = 5 ∧ seats_with_2_children = 2)
  (h_children_per_seat : children_per_seat_3 = 3 ∧ children_per_seat_2 = 2) :
  seats_with_3_children * children_per_seat_3 + seats_with_2_children * children_per_seat_2 = 19 :=
by
  sorry

end minibus_children_count_l1783_178334


namespace ratio_of_larger_to_smaller_is_sqrt_six_l1783_178305

def sum_of_squares_eq_seven_times_difference (a b : ℝ) : Prop := 
  a^2 + b^2 = 7 * (a - b)

theorem ratio_of_larger_to_smaller_is_sqrt_six {a b : ℝ} (h : sum_of_squares_eq_seven_times_difference a b) (h1 : a > b) : 
  a / b = Real.sqrt 6 :=
sorry

end ratio_of_larger_to_smaller_is_sqrt_six_l1783_178305


namespace average_class_is_45_6_l1783_178331

noncomputable def average_class_score (total_students : ℕ) (top_scorers : ℕ) (top_score : ℕ) 
  (zero_scorers : ℕ) (remaining_students_avg : ℕ) : ℚ :=
  let total_top_score := top_scorers * top_score
  let total_zero_score := zero_scorers * 0
  let remaining_students := total_students - top_scorers - zero_scorers
  let total_remaining_score := remaining_students * remaining_students_avg
  let total_score := total_top_score + total_zero_score + total_remaining_score
  total_score / total_students

theorem average_class_is_45_6 : average_class_score 25 3 95 3 45 = 45.6 := 
by
  -- sorry is used here to skip the proof. Lean will expect a proof here.
  sorry

end average_class_is_45_6_l1783_178331


namespace population_initial_count_l1783_178391

theorem population_initial_count
  (P : ℕ)
  (birth_rate : ℕ := 52)
  (death_rate : ℕ := 16)
  (net_growth_rate : ℝ := 1.2) :
  36 = (net_growth_rate / 100) * P ↔ P = 3000 :=
by sorry

end population_initial_count_l1783_178391


namespace man_older_than_son_l1783_178369

variables (S M : ℕ)

theorem man_older_than_son (h1 : S = 32) (h2 : M + 2 = 2 * (S + 2)) : M - S = 34 :=
by
  sorry

end man_older_than_son_l1783_178369


namespace sum_of_first_90_terms_l1783_178345

def arithmetic_progression_sum (n : ℕ) (a d : ℚ) : ℚ :=
  (n : ℚ) / 2 * (2 * a + (n - 1) * d)

theorem sum_of_first_90_terms (a d : ℚ) :
  (arithmetic_progression_sum 15 a d = 150) →
  (arithmetic_progression_sum 75 a d = 75) →
  (arithmetic_progression_sum 90 a d = -112.5) :=
by
  sorry

end sum_of_first_90_terms_l1783_178345


namespace find_c_l1783_178332

-- Define c and the floor function
def c : ℝ := 13.1

theorem find_c (h : c + ⌊c⌋ = 25.6) : c = 13.1 :=
sorry

end find_c_l1783_178332


namespace cone_height_ratio_l1783_178370

theorem cone_height_ratio (C : ℝ) (h₁ : ℝ) (V₂ : ℝ) (r : ℝ) (h₂ : ℝ) :
  C = 20 * Real.pi → 
  h₁ = 40 →
  V₂ = 400 * Real.pi →
  2 * Real.pi * r = 20 * Real.pi →
  V₂ = (1 / 3) * Real.pi * r^2 * h₂ →
  h₂ / h₁ = (3 / 10) := by
sorry

end cone_height_ratio_l1783_178370


namespace well_diameter_l1783_178342

theorem well_diameter (V h : ℝ) (pi : ℝ) (r : ℝ) :
  h = 8 ∧ V = 25.132741228718345 ∧ pi = 3.141592653589793 ∧ V = pi * r^2 * h → 2 * r = 2 :=
by
  sorry

end well_diameter_l1783_178342


namespace carla_glasses_lemonade_l1783_178323

theorem carla_glasses_lemonade (time_total : ℕ) (rate : ℕ) (glasses : ℕ) 
  (h1 : time_total = 3 * 60 + 40) 
  (h2 : rate = 20) 
  (h3 : glasses = time_total / rate) : 
  glasses = 11 := 
by 
  -- We'll fill in the proof here in a real scenario
  sorry

end carla_glasses_lemonade_l1783_178323


namespace sin_45_eq_sqrt2_div_2_l1783_178319

theorem sin_45_eq_sqrt2_div_2 :
  Real.sin (45 * Real.pi / 180) = Real.sqrt 2 / 2 :=
by
  sorry

end sin_45_eq_sqrt2_div_2_l1783_178319


namespace zach_needs_more_money_l1783_178365

noncomputable def cost_of_bike : ℕ := 100
noncomputable def weekly_allowance : ℕ := 5
noncomputable def mowing_income : ℕ := 10
noncomputable def babysitting_rate_per_hour : ℕ := 7
noncomputable def initial_savings : ℕ := 65
noncomputable def hours_babysitting : ℕ := 2

theorem zach_needs_more_money : 
  cost_of_bike - (initial_savings + weekly_allowance + mowing_income + (babysitting_rate_per_hour * hours_babysitting)) = 6 :=
by
  sorry

end zach_needs_more_money_l1783_178365


namespace matrix_power_eq_l1783_178335

def MatrixC : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![3, 4], ![-8, -10]]

def MatrixA : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![201, 200], ![-400, -449]]

theorem matrix_power_eq :
  MatrixC ^ 50 = MatrixA := 
  sorry

end matrix_power_eq_l1783_178335


namespace find_number_l1783_178389

theorem find_number (n : ℕ) (some_number : ℕ) 
  (h : (1/5 : ℝ)^n * (1/4 : ℝ)^(18 : ℕ) = 1 / (2 * (some_number : ℝ)^n))
  (hn : n = 35) : some_number = 10 := 
by 
  sorry

end find_number_l1783_178389


namespace coaching_fee_correct_l1783_178358

noncomputable def total_coaching_fee : ℝ :=
  let daily_fee : ℝ := 39
  let discount_threshold : ℝ := 50
  let discount_rate : ℝ := 0.10
  let total_days : ℝ := 31 + 28 + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 3 -- non-leap year days count up to Nov 3
  let discount_days : ℝ := total_days - discount_threshold
  let discounted_fee : ℝ := daily_fee * (1 - discount_rate)
  let fee_before_discount : ℝ := discount_threshold * daily_fee
  let fee_after_discount : ℝ := discount_days * discounted_fee
  fee_before_discount + fee_after_discount

theorem coaching_fee_correct :
  total_coaching_fee = 10967.7 := by
  sorry

end coaching_fee_correct_l1783_178358


namespace seats_in_row_l1783_178348

theorem seats_in_row (y : ℕ → ℕ) (k b : ℕ) :
  (∀ x, y x = k * x + b) →
  y 1 = 20 →
  y 19 = 56 →
  y 26 = 70 :=
by
  intro h1 h2 h3
  -- Additional constraints to prove the given requirements
  sorry

end seats_in_row_l1783_178348


namespace profit_relationship_max_profit_l1783_178377

noncomputable def W (x : ℝ) : ℝ :=
if h : 0 ≤ x ∧ x ≤ 2 then 5 * (x^2 + 3)
else if h : 2 < x ∧ x ≤ 5 then 50 * x / (1 + x)
else 0

noncomputable def f (x : ℝ) : ℝ :=
15 * W x - 10 * x - 20 * x

theorem profit_relationship:
  (∀ x, 0 ≤ x ∧ x ≤ 2 → f x = 75 * x^2 - 30 * x + 225) ∧
  (∀ x, 2 < x ∧ x ≤ 5 → f x = (750 * x)/(1 + x) - 30 * x) :=
by
  -- to be proven
  sorry

theorem max_profit:
  ∃ x, 0 ≤ x ∧ x ≤ 5 ∧ f x = 480 ∧ 10 * x = 40 :=
by
  -- to be proven
  sorry

end profit_relationship_max_profit_l1783_178377


namespace fourth_number_is_8_l1783_178363

theorem fourth_number_is_8 (a b c : ℕ) (mean : ℕ) (h_mean : mean = 20) (h_a : a = 12) (h_b : b = 24) (h_c : c = 36) :
  ∃ d : ℕ, mean * 4 = a + b + c + d ∧ (∃ x : ℕ, d = x^2) ∧ d = 8 := by
sorry

end fourth_number_is_8_l1783_178363


namespace remainder_when_13_plus_y_divided_by_31_l1783_178394

theorem remainder_when_13_plus_y_divided_by_31
  (y : ℕ)
  (hy : 7 * y % 31 = 1) :
  (13 + y) % 31 = 22 :=
sorry

end remainder_when_13_plus_y_divided_by_31_l1783_178394


namespace complementary_event_A_l1783_178359

def EventA (n : ℕ) := n ≥ 2

def ComplementaryEventA (n : ℕ) := n ≤ 1

theorem complementary_event_A (n : ℕ) : ComplementaryEventA n ↔ ¬ EventA n := by
  sorry

end complementary_event_A_l1783_178359


namespace range_of_x_when_a_is_1_range_of_a_for_necessity_l1783_178343

-- Define the statements p and q based on the conditions
def p (x a : ℝ) := (x - a) * (x - 3 * a) < 0
def q (x : ℝ) := (x - 3) / (x - 2) ≤ 0

-- (1) Prove the range of x when a = 1 and p ∧ q is true
theorem range_of_x_when_a_is_1 {x : ℝ} (h1 : ∀ x, p x 1) (h2 : q x) : 2 < x ∧ x < 3 :=
  sorry

-- (2) Prove the range of a for p to be necessary but not sufficient for q
theorem range_of_a_for_necessity : ∀ a, (∀ x, p x a → q x) → (1 ≤ a ∧ a ≤ 2) :=
  sorry

end range_of_x_when_a_is_1_range_of_a_for_necessity_l1783_178343


namespace total_receipts_correct_l1783_178398

def cost_adult_ticket : ℝ := 5.50
def cost_children_ticket : ℝ := 2.50
def number_of_adults : ℕ := 152
def number_of_children : ℕ := number_of_adults / 2

def receipts_from_adults : ℝ := number_of_adults * cost_adult_ticket
def receipts_from_children : ℝ := number_of_children * cost_children_ticket
def total_receipts : ℝ := receipts_from_adults + receipts_from_children

theorem total_receipts_correct : total_receipts = 1026 := 
by
  -- Proof omitted, proof needed to validate theorem statement.
  sorry

end total_receipts_correct_l1783_178398


namespace mr_mcpherson_needs_to_raise_840_l1783_178386

def total_rent : ℝ := 1200
def mrs_mcpherson_contribution : ℝ := 0.30 * total_rent
def mr_mcpherson_contribution : ℝ := total_rent - mrs_mcpherson_contribution

theorem mr_mcpherson_needs_to_raise_840 :
  mr_mcpherson_contribution = 840 := 
by
  sorry

end mr_mcpherson_needs_to_raise_840_l1783_178386


namespace marina_drive_l1783_178313

theorem marina_drive (a b c : ℕ) (x : ℕ) 
  (h1 : 1 ≤ a) 
  (h2 : a + b + c ≤ 9)
  (h3 : 90 * (b - a) = 60 * x)
  (h4 : x = 3 * (b - a) / 2) :
  a = 1 ∧ b = 3 ∧ c = 5 ∧ a^2 + b^2 + c^2 = 35 :=
by {
  sorry
}

end marina_drive_l1783_178313


namespace percentage_is_60_l1783_178324

-- Definitions based on the conditions
def fraction_value (x : ℕ) : ℕ := x / 3
def percentage_less_value (x p : ℕ) : ℕ := x - (p * x) / 100

-- Lean statement based on the mathematically equivalent proof problem
theorem percentage_is_60 : ∀ (x p : ℕ), x = 180 → fraction_value x = 60 → percentage_less_value 60 p = 24 → p = 60 :=
by
  intros x p H1 H2 H3
  -- Proof is not required, so we use sorry
  sorry

end percentage_is_60_l1783_178324


namespace probability_forming_more_from_remont_probability_forming_papa_from_papaha_l1783_178387

-- Definition for part (a)
theorem probability_forming_more_from_remont : 
  (6 * 5 * 4 * 3 = 360) ∧ (1 / 360 = 0.00278) :=
by
  sorry

-- Definition for part (b)
theorem probability_forming_papa_from_papaha : 
  (6 * 5 * 4 * 3 = 360) ∧ (12 / 360 = 0.03333) :=
by
  sorry

end probability_forming_more_from_remont_probability_forming_papa_from_papaha_l1783_178387


namespace lines_perpendicular_to_same_plane_are_parallel_l1783_178337

variables {Point Line Plane : Type}
variables (a b c : Line) (α β γ : Plane)
variables (perp_line_to_plane : Line → Plane → Prop) (parallel_lines : Line → Line → Prop)
variables (subset_line_in_plane : Line → Plane → Prop)

-- The conditions
axiom a_perp_alpha : perp_line_to_plane a α
axiom b_perp_alpha : perp_line_to_plane b α

-- The statement to prove
theorem lines_perpendicular_to_same_plane_are_parallel :
  parallel_lines a b :=
by sorry

end lines_perpendicular_to_same_plane_are_parallel_l1783_178337


namespace integer_triangle_600_integer_triangle_144_l1783_178375

-- Problem Part I
theorem integer_triangle_600 :
  ∃ (a b c : ℕ), a * b * c = 600 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a + b + c = 26 :=
by {
  sorry
}

-- Problem Part II
theorem integer_triangle_144 :
  ∃ (a b c : ℕ), a * b * c = 144 ∧ a + b > c ∧ b + c > a ∧ c + a > b ∧ a + b + c = 16 :=
by {
  sorry
}

end integer_triangle_600_integer_triangle_144_l1783_178375


namespace probability_blue_face_facing_up_l1783_178317

-- Define the context
def octahedron_faces : ℕ := 8
def blue_faces : ℕ := 5
def red_faces : ℕ := 3
def total_faces : ℕ := blue_faces + red_faces

-- The probability calculation theorem
theorem probability_blue_face_facing_up (h : total_faces = octahedron_faces) :
  (blue_faces : ℝ) / (octahedron_faces : ℝ) = 5 / 8 :=
by
  -- Placeholder for proof
  sorry

end probability_blue_face_facing_up_l1783_178317


namespace parallel_vectors_sum_coords_l1783_178315

theorem parallel_vectors_sum_coords
  (x y : ℝ)
  (a b : ℝ × ℝ × ℝ)
  (h_a : a = (2, x, 3))
  (h_b : b = (-4, 2, y))
  (h_parallel : ∃ k : ℝ, a = k • b) :
  x + y = -7 :=
sorry

end parallel_vectors_sum_coords_l1783_178315


namespace selection_count_l1783_178355

def choose (n k : ℕ) : ℕ := -- Binomial coefficient definition
  if h : 0 ≤ k ∧ k ≤ n then
    Nat.choose n k
  else
    0

theorem selection_count : choose 9 5 - choose 6 5 = 120 := by
  sorry

end selection_count_l1783_178355


namespace number_of_cherries_l1783_178393

-- Definitions for the problem conditions
def total_fruits : ℕ := 580
def raspberries (b : ℕ) : ℕ := 2 * b
def grapes (c : ℕ) : ℕ := 3 * c
def cherries (r : ℕ) : ℕ := 3 * r

-- Theorem to prove the number of cherries
theorem number_of_cherries (b r g c : ℕ) 
  (H1 : b + r + g + c = total_fruits)
  (H2 : r = raspberries b)
  (H3 : g = grapes c)
  (H4 : c = cherries r) :
  c = 129 :=
by sorry

end number_of_cherries_l1783_178393


namespace tan_triple_angle_l1783_178378

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 1/3) : Real.tan (3 * θ) = 13/9 :=
by
  sorry

end tan_triple_angle_l1783_178378


namespace course_selection_plans_l1783_178373

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem course_selection_plans :
  let A_courses := C 4 2
  let B_courses := C 4 3
  let C_courses := C 4 3
  A_courses * B_courses * C_courses = 96 :=
by
  sorry

end course_selection_plans_l1783_178373


namespace polynomial_system_solution_l1783_178336

variable {x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ}

theorem polynomial_system_solution (
  h1 : x₁ + 3 * x₂ + 5 * x₃ + 7 * x₄ + 9 * x₅ + 11 * x₆ + 13 * x₇ = 3)
  (h2 : 3 * x₁ + 5 * x₂ + 7 * x₃ + 9 * x₄ + 11 * x₅ + 13 * x₆ + 15 * x₇ = 15)
  (h3 : 5 * x₁ + 7 * x₂ + 9 * x₃ + 11 * x₄ + 13 * x₅ + 15 * x₆ + 17 * x₇ = 85) :
  7 * x₁ + 9 * x₂ + 11 * x₃ + 13 * x₄ + 15 * x₅ + 17 * x₆ + 19 * x₇ = 213 :=
sorry

end polynomial_system_solution_l1783_178336


namespace incorrect_statement_C_l1783_178362

theorem incorrect_statement_C 
  (x y : ℝ)
  (n : ℕ)
  (data : Fin n → (ℝ × ℝ))
  (h : ∀ (i : Fin n), (x, y) = data i)
  (reg_eq : ∀ (x : ℝ), 0.85 * x - 85.71 = y) :
  ¬ (forall (x : ℝ), x = 160 → ∀ (y : ℝ), y = 50.29) := 
sorry

end incorrect_statement_C_l1783_178362


namespace eggs_in_second_tree_l1783_178382

theorem eggs_in_second_tree
  (nests_in_first_tree : ℕ)
  (eggs_per_nest : ℕ)
  (eggs_in_front_yard : ℕ)
  (total_eggs : ℕ)
  (eggs_in_second_tree : ℕ)
  (h1 : nests_in_first_tree = 2)
  (h2 : eggs_per_nest = 5)
  (h3 : eggs_in_front_yard = 4)
  (h4 : total_eggs = 17)
  (h5 : nests_in_first_tree * eggs_per_nest + eggs_in_front_yard + eggs_in_second_tree = total_eggs) :
  eggs_in_second_tree = 3 :=
sorry

end eggs_in_second_tree_l1783_178382


namespace mushrooms_weight_change_l1783_178301

-- Conditions
variables (x W : ℝ)
variable (initial_weight : ℝ := 100 * x)
variable (dry_weight : ℝ := x)
variable (final_weight_dry : ℝ := 2 * W / 100)

-- Given fresh mushrooms have moisture content of 99%
-- and dried mushrooms have moisture content of 98%
theorem mushrooms_weight_change 
  (h1 : dry_weight = x) 
  (h2 : final_weight_dry = x / 0.02) 
  (h3 : W = x / 0.02) 
  (initial_weight : ℝ := 100 * x) : 
  2 * W = initial_weight / 2 :=
by
  -- This is a placeholder for the proof steps which we skip
  sorry

end mushrooms_weight_change_l1783_178301


namespace rainfall_on_tuesday_l1783_178304

theorem rainfall_on_tuesday 
  (r_Mon r_Wed r_Total r_Tue : ℝ)
  (h_Mon : r_Mon = 0.16666666666666666)
  (h_Wed : r_Wed = 0.08333333333333333)
  (h_Total : r_Total = 0.6666666666666666)
  (h_Tue : r_Tue = r_Total - (r_Mon + r_Wed)) :
  r_Tue = 0.41666666666666663 := 
sorry

end rainfall_on_tuesday_l1783_178304


namespace large_circle_radius_l1783_178329

noncomputable def radius_of_large_circle (R : ℝ) : Prop :=
  ∃ r : ℝ, (r = 2) ∧
           (R = r + r) ∧
           (r = 2) ∧
           (R - r = 2) ∧
           (R = 4)

theorem large_circle_radius :
  radius_of_large_circle 4 :=
by
  sorry

end large_circle_radius_l1783_178329


namespace friends_recycled_pounds_l1783_178330

-- Definitions of given conditions
def points_earned : ℕ := 6
def pounds_per_point : ℕ := 8
def zoe_pounds : ℕ := 25

-- Calculation based on given conditions
def total_pounds := points_earned * pounds_per_point
def friends_pounds := total_pounds - zoe_pounds

-- Statement of the proof problem
theorem friends_recycled_pounds : friends_pounds = 23 := by
  sorry

end friends_recycled_pounds_l1783_178330


namespace factorization_solution_l1783_178390

def factorization_problem : Prop :=
  ∃ (a b c : ℤ), (∀ (x : ℤ), x^2 + 17 * x + 70 = (x + a) * (x + b)) ∧ 
                 (∀ (x : ℤ), x^2 - 18 * x + 80 = (x - b) * (x - c)) ∧ 
                 (a + b + c = 28)

theorem factorization_solution : factorization_problem :=
sorry

end factorization_solution_l1783_178390


namespace proof_problem_l1783_178397

variable (x y : ℝ)

theorem proof_problem 
  (h1 : 0.30 * x = 0.40 * 150 + 90)
  (h2 : 0.20 * x = 0.50 * 180 - 60)
  (h3 : y = 0.75 * x)
  (h4 : y^2 > x + 100) :
  x = 150 ∧ y = 112.5 :=
by
  sorry

end proof_problem_l1783_178397


namespace tyler_brother_age_difference_l1783_178322

-- Definitions of Tyler's age and the sum of their ages:
def tyler_age : ℕ := 7
def sum_of_ages (brother_age : ℕ) : Prop := tyler_age + brother_age = 11

-- Proof problem: Prove that Tyler's brother's age minus Tyler's age equals 4 years.
theorem tyler_brother_age_difference (B : ℕ) (h : sum_of_ages B) : B - tyler_age = 4 :=
by
  sorry

end tyler_brother_age_difference_l1783_178322


namespace sum_of_distinct_elements_not_square_l1783_178372

open Set

noncomputable def setS : Set ℕ := { n | ∃ k : ℕ, n = 2^(2*k+1) }

theorem sum_of_distinct_elements_not_square (s : Finset ℕ) (hs: ∀ x ∈ s, x ∈ setS) :
  ¬∃ k : ℕ, s.sum id = k^2 :=
sorry

end sum_of_distinct_elements_not_square_l1783_178372


namespace number_of_rectangles_l1783_178371

-- Definition of the problem: We have 12 equally spaced points on a circle.
def points_on_circle : ℕ := 12

-- The number of diameters is half the number of points, as each diameter involves two points.
def diameters (n : ℕ) : ℕ := n / 2

-- The number of ways to choose 2 diameters out of n/2 is given by the binomial coefficient.
noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Prove the number of rectangles that can be formed is 15.
theorem number_of_rectangles :
  binomial_coefficient (diameters points_on_circle) 2 = 15 := by
  sorry

end number_of_rectangles_l1783_178371


namespace roots_product_l1783_178346

theorem roots_product {a b : ℝ} (h1 : a^2 - a - 2 = 0) (h2 : b^2 - b - 2 = 0) 
(roots : a ≠ b ∧ ∀ x, x^2 - x - 2 = 0 ↔ (x = a ∨ x = b)) : (a - 1) * (b - 1) = -2 := by
  -- proof
  sorry

end roots_product_l1783_178346


namespace log_base_2_of_7_l1783_178354

variable (m n : ℝ)

theorem log_base_2_of_7 (h1 : Real.log 5 = m) (h2 : Real.log 7 = n) : Real.logb 2 7 = n / (1 - m) :=
by
  sorry

end log_base_2_of_7_l1783_178354


namespace intersecting_circles_l1783_178333

theorem intersecting_circles (m c : ℝ)
  (h1 : ∃ (x1 y1 x2 y2 : ℝ), x1 = 1 ∧ y1 = 3 ∧ x2 = m ∧ y2 = 1 ∧ x1 ≠ x2 ∧ y1 ≠ y2)
  (h2 : ∀ (x y : ℝ), (x - y + (c / 2) = 0) → (x = 1 ∨ y = 3)) :
  m + c = 3 :=
sorry

end intersecting_circles_l1783_178333


namespace negation_of_p_l1783_178399

def p : Prop := ∀ x : ℝ, x > Real.sin x

theorem negation_of_p : ¬p ↔ ∃ x : ℝ, x ≤ Real.sin x :=
by sorry

end negation_of_p_l1783_178399


namespace problem_1_problem_2_problem_3_l1783_178383

open Set

def U : Set ℝ := univ
def A : Set ℝ := {x | -4 ≤ x ∧ x < 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 3}
def P : Set ℝ := {x | x ≤ 0 ∨ x ≥ 5 / 2}

theorem problem_1 : A ∩ B = {x | -1 < x ∧ x < 2} := sorry

theorem problem_2 : compl B ∪ P = {x | x ≤ 0 ∨ x ≥ 5 / 2} := sorry

theorem problem_3 : (A ∩ B) ∩ compl P = {x | 0 < x ∧ x < 2} := sorry

end problem_1_problem_2_problem_3_l1783_178383


namespace min_bottles_required_l1783_178357

theorem min_bottles_required (bottle_ounces : ℕ) (total_ounces : ℕ) (h : bottle_ounces = 15) (ht : total_ounces = 150) :
  ∃ (n : ℕ), n * bottle_ounces >= total_ounces ∧ n = 10 :=
by
  sorry

end min_bottles_required_l1783_178357


namespace solution_exists_l1783_178328

def valid_grid (grid : List (List Nat)) : Prop :=
  grid = [[2, 3, 6], [6, 3, 2]] ∨
  grid = [[2, 4, 8], [8, 4, 2]]

theorem solution_exists :
  ∃ (grid : List (List Nat)), valid_grid grid := by
  sorry

end solution_exists_l1783_178328


namespace find_range_of_x_l1783_178338

variable (f : ℝ → ℝ) (x : ℝ)

-- Assume f is an increasing function on [-1, 1]
def is_increasing_on_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x ≤ y → f x ≤ f y

-- Main theorem statement based on the problem
theorem find_range_of_x (h_increasing : is_increasing_on_interval f (-1) 1)
                        (h_condition : f (x - 1) < f (1 - 3 * x)) :
  0 ≤ x ∧ x < (1 / 2) :=
sorry

end find_range_of_x_l1783_178338


namespace max_area_with_22_matches_l1783_178318

-- Definitions based on the conditions
def perimeter := 22

def is_valid_length_width (l w : ℕ) : Prop := l + w = 11

def area (l w : ℕ) : ℕ := l * w

-- Statement of the proof problem
theorem max_area_with_22_matches : 
  ∃ (l w : ℕ), is_valid_length_width l w ∧ (∀ l' w', is_valid_length_width l' w' → area l w ≥ area l' w') ∧ area l w = 30 :=
  sorry

end max_area_with_22_matches_l1783_178318


namespace possible_values_of_x_and_factors_l1783_178353

theorem possible_values_of_x_and_factors (p : ℕ) (h_prime : Nat.Prime p) :
  ∃ (x : ℕ), x = p^5 ∧ (∀ (d : ℕ), d ∣ x → d = p^0 ∨ d = p^1 ∨ d = p^2 ∨ d = p^3 ∨ d = p^4 ∨ d = p^5) ∧ Nat.divisors x ≠ ∅ ∧ (Nat.divisors x).card = 6 := 
  by 
    sorry

end possible_values_of_x_and_factors_l1783_178353


namespace basketball_team_initial_games_l1783_178350

theorem basketball_team_initial_games (G W : ℝ) 
  (h1 : W = 0.70 * G) 
  (h2 : W + 2 = 0.60 * (G + 10)) : 
  G = 40 :=
by
  sorry

end basketball_team_initial_games_l1783_178350


namespace parallel_heater_time_l1783_178344

theorem parallel_heater_time (t1 t2 : ℕ) (R1 R2 : ℝ) (t : ℕ) (I : ℝ) (Q : ℝ) (h₁ : t1 = 3) 
  (h₂ : t2 = 6) (hq1 : Q = I^2 * R1 * t1) (hq2 : Q = I^2 * R2 * t2) :
  t = (t1 * t2) / (t1 + t2) := by
  sorry

end parallel_heater_time_l1783_178344


namespace trigonometric_identity_l1783_178361

theorem trigonometric_identity 
  (α : ℝ)
  (h : Real.tan (α + Real.pi / 4) = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l1783_178361


namespace regular_pay_calculation_l1783_178306

theorem regular_pay_calculation
  (R : ℝ)  -- defining the regular pay per hour
  (H1 : 40 * R + 20 * R = 180):  -- condition given based on the total actual pay calculation.
  R = 3 := 
by
  -- Skipping the proof
  sorry

end regular_pay_calculation_l1783_178306


namespace positive_integers_satisfying_condition_l1783_178303

theorem positive_integers_satisfying_condition :
  ∃! n : ℕ, 0 < n ∧ 24 - 6 * n > 12 :=
by
  sorry

end positive_integers_satisfying_condition_l1783_178303


namespace problem_solution_l1783_178368

theorem problem_solution (a b : ℤ) (h1 : 6 * b + 4 * a = -50) (h2 : a * b = -84) : a + 2 * b = -17 := 
  sorry

end problem_solution_l1783_178368


namespace inverse_square_variation_l1783_178312

variable (x y : ℝ)

theorem inverse_square_variation (h1 : x = 1) (h2 : y = 3) (h3 : y = 2) : x = 2.25 :=
by
  sorry

end inverse_square_variation_l1783_178312


namespace cristina_catches_nicky_l1783_178392

-- Definitions from the conditions
def cristina_speed : ℝ := 4 -- meters per second
def nicky_speed : ℝ := 3 -- meters per second
def nicky_head_start : ℝ := 36 -- meters

-- The proof to find the time 't'
theorem cristina_catches_nicky (t : ℝ) : cristina_speed * t = nicky_head_start + nicky_speed * t -> t = 36 := by
  intros h
  sorry

end cristina_catches_nicky_l1783_178392
