import Mathlib

namespace NUMINAMATH_GPT_angle_T_in_pentagon_l1603_160395

theorem angle_T_in_pentagon (P Q R S T : ℝ) 
  (h1 : P = R) (h2 : P = T) (h3 : Q + S = 180) 
  (h4 : P + Q + R + S + T = 540) : T = 120 :=
by
  sorry

end NUMINAMATH_GPT_angle_T_in_pentagon_l1603_160395


namespace NUMINAMATH_GPT_problem_statement_l1603_160367

theorem problem_statement :
  (¬ (∀ x : ℝ, 2 * x < 3 * x)) ∧ (∃ x : ℝ, x^3 = 1 - x^2) := 
sorry

end NUMINAMATH_GPT_problem_statement_l1603_160367


namespace NUMINAMATH_GPT_sophomores_sampled_correct_l1603_160327

def stratified_sampling_sophomores (total_students num_sophomores sample_size : ℕ) : ℕ :=
  (num_sophomores * sample_size) / total_students

theorem sophomores_sampled_correct :
  stratified_sampling_sophomores 4500 1500 600 = 200 :=
by
  sorry

end NUMINAMATH_GPT_sophomores_sampled_correct_l1603_160327


namespace NUMINAMATH_GPT_hyperbola_other_asymptote_l1603_160326

-- Define the problem conditions
def one_asymptote (x y : ℝ) : Prop := y = 2 * x
def foci_x_coordinate : ℝ := -4

-- Define the equation of the other asymptote
def other_asymptote (x y : ℝ) : Prop := y = -2 * x - 16

-- The statement to be proved
theorem hyperbola_other_asymptote : 
  (∀ x y, one_asymptote x y) → (∀ x, x = -4 → ∃ y, ∃ C, other_asymptote x y ∧ y = C + -2 * x - 8) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_other_asymptote_l1603_160326


namespace NUMINAMATH_GPT_reduced_admission_price_is_less_l1603_160357

-- Defining the conditions
def regular_admission_cost : ℕ := 8
def total_people : ℕ := 2 + 3 + 1
def total_cost_before_6pm : ℕ := 30
def cost_per_person_before_6pm : ℕ := total_cost_before_6pm / total_people

-- Stating the theorem
theorem reduced_admission_price_is_less :
  (regular_admission_cost - cost_per_person_before_6pm) = 3 :=
by
  sorry -- Proof to be filled

end NUMINAMATH_GPT_reduced_admission_price_is_less_l1603_160357


namespace NUMINAMATH_GPT_partial_fraction_product_l1603_160322

theorem partial_fraction_product (A B C : ℝ) :
  (∀ x : ℝ, x^3 - 3 * x^2 - 10 * x + 24 ≠ 0 →
            (x^2 - 25) / (x^3 - 3 * x^2 - 10 * x + 24) = A / (x - 2) + B / (x + 3) + C / (x - 4)) →
  A = 1 ∧ B = 1 ∧ C = 1 →
  A * B * C = 1 := by
  sorry

end NUMINAMATH_GPT_partial_fraction_product_l1603_160322


namespace NUMINAMATH_GPT_avg_speed_correct_l1603_160331

noncomputable def avg_speed_round_trip
  (flight_up_speed : ℝ)
  (tailwind_speed : ℝ)
  (tailwind_angle : ℝ)
  (flight_home_speed : ℝ)
  (headwind_speed : ℝ)
  (headwind_angle : ℝ) : ℝ :=
  let effective_tailwind_speed := tailwind_speed * Real.cos (tailwind_angle * Real.pi / 180)
  let ground_speed_to_mother := flight_up_speed + effective_tailwind_speed
  let effective_headwind_speed := headwind_speed * Real.cos (headwind_angle * Real.pi / 180)
  let ground_speed_back_home := flight_home_speed - effective_headwind_speed
  (ground_speed_to_mother + ground_speed_back_home) / 2

theorem avg_speed_correct :
  avg_speed_round_trip 96 12 30 88 15 60 = 93.446 :=
by
  sorry

end NUMINAMATH_GPT_avg_speed_correct_l1603_160331


namespace NUMINAMATH_GPT_find_g3_l1603_160317

variable {g : ℝ → ℝ}

-- Defining the condition from the problem
def g_condition (x : ℝ) (h : x ≠ 0) : g x - 3 * g (1 / x) = 3^x + x^2 := sorry

-- The main statement to prove
theorem find_g3 : g 3 = - (3 * 3^(1/3) + 1/3 + 36) / 8 := sorry

end NUMINAMATH_GPT_find_g3_l1603_160317


namespace NUMINAMATH_GPT_max_value_output_l1603_160329

theorem max_value_output (a b c : ℝ) (h_a : a = 3) (h_b : b = 7) (h_c : c = 2) : max (max a b) c = 7 := 
by
  sorry

end NUMINAMATH_GPT_max_value_output_l1603_160329


namespace NUMINAMATH_GPT_find_XY_in_triangle_l1603_160387

-- Definitions
def Triangle := Type
def angle_measures (T : Triangle) : (ℕ × ℕ × ℕ) := sorry
def side_lengths (T : Triangle) : (ℕ × ℕ × ℕ) := sorry
def is_30_60_90_triangle (T : Triangle) : Prop := (angle_measures T = (30, 60, 90))

-- Given conditions and statement we want to prove
def triangle_XYZ : Triangle := sorry
def XY : ℕ := 6

-- Proof statement
theorem find_XY_in_triangle :
  is_30_60_90_triangle triangle_XYZ ∧ (side_lengths triangle_XYZ).1 = XY →
  XY = 6 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_XY_in_triangle_l1603_160387


namespace NUMINAMATH_GPT_intersection_point_l1603_160368

def line_eq (x y z : ℝ) : Prop :=
  (x - 1) / 1 = (y + 1) / 0 ∧ (y + 1) / 0 = (z - 1) / -1

def plane_eq (x y z : ℝ) : Prop :=
  3 * x - 2 * y - 4 * z - 8 = 0

theorem intersection_point : 
  ∃ (x y z : ℝ), line_eq x y z ∧ plane_eq x y z ∧ x = -6 ∧ y = -1 ∧ z = 8 :=
by 
  sorry

end NUMINAMATH_GPT_intersection_point_l1603_160368


namespace NUMINAMATH_GPT_original_price_of_shoes_l1603_160303

theorem original_price_of_shoes (
  initial_amount : ℝ := 74
) (sweater_cost : ℝ := 9) (tshirt_cost : ℝ := 11) 
  (final_amount_after_refund : ℝ := 51)
  (refund_percentage : ℝ := 0.90)
  (S : ℝ) :
  (initial_amount - sweater_cost - tshirt_cost - S + refund_percentage * S = final_amount_after_refund) -> 
  S = 30 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_original_price_of_shoes_l1603_160303


namespace NUMINAMATH_GPT_min_value_fraction_l1603_160358

theorem min_value_fraction (x y : ℝ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hx_gt_hy : x > y) (h_eq : x + 2 * y = 3) : 
  (∃ t, t = (1 / (x - y) + 9 / (x + 5 * y)) ∧ t = 8 / 3) :=
by 
  sorry

end NUMINAMATH_GPT_min_value_fraction_l1603_160358


namespace NUMINAMATH_GPT_mul_exponent_result_l1603_160328

theorem mul_exponent_result : 112 * (5^4) = 70000 := 
by 
  sorry

end NUMINAMATH_GPT_mul_exponent_result_l1603_160328


namespace NUMINAMATH_GPT_seeds_total_l1603_160360

def seedsPerWatermelon : Nat := 345
def numberOfWatermelons : Nat := 27
def totalSeeds : Nat := seedsPerWatermelon * numberOfWatermelons

theorem seeds_total :
  totalSeeds = 9315 :=
by
  sorry

end NUMINAMATH_GPT_seeds_total_l1603_160360


namespace NUMINAMATH_GPT_alpha_range_theorem_l1603_160332

noncomputable def alpha_range (k : ℤ) (α : ℝ) : Prop :=
  2 * k * Real.pi - Real.pi ≤ α ∧ α ≤ 2 * k * Real.pi

theorem alpha_range_theorem (α : ℝ) (k : ℤ) (h : |Real.sin (4 * Real.pi - α)| = Real.sin (Real.pi + α)) :
  alpha_range k α :=
by
  sorry

end NUMINAMATH_GPT_alpha_range_theorem_l1603_160332


namespace NUMINAMATH_GPT_trapezoid_area_l1603_160394

theorem trapezoid_area (a b d1 d2 : ℝ) (ha : 0 < a) (hb : 0 < b) (hd1 : 0 < d1) (hd2 : 0 < d2)
  (hbase : a = 11) (hbase2 : b = 4) (hdiagonal1 : d1 = 9) (hdiagonal2 : d2 = 12) :
  (∃ area : ℝ, area = 54) :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1603_160394


namespace NUMINAMATH_GPT_geometric_sum_eight_terms_l1603_160321

theorem geometric_sum_eight_terms (a_1 : ℕ) (S_4 : ℕ) (r : ℕ) (S_8 : ℕ) 
    (h1 : r = 2) (h2 : S_4 = a_1 * (1 + r + r^2 + r^3)) (h3 : S_4 = 30) :
    S_8 = a_1 * (1 + r + r^2 + r^3 + r^4 + r^5 + r^6 + r^7) → S_8 = 510 := 
by sorry

end NUMINAMATH_GPT_geometric_sum_eight_terms_l1603_160321


namespace NUMINAMATH_GPT_mayoral_election_votes_l1603_160391

theorem mayoral_election_votes (Y Z : ℕ) 
  (h1 : 22500 = Y + Y / 2) 
  (h2 : 15000 = Z - Z / 5 * 2)
  : Z = 25000 := 
  sorry

end NUMINAMATH_GPT_mayoral_election_votes_l1603_160391


namespace NUMINAMATH_GPT_equation_graph_is_ellipse_l1603_160390

theorem equation_graph_is_ellipse :
  ∃ a b c d : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ b * (x^2 - 72 * y^2) + a * x + d = a * c * (y - 6)^2 :=
sorry

end NUMINAMATH_GPT_equation_graph_is_ellipse_l1603_160390


namespace NUMINAMATH_GPT_ball_of_yarn_costs_6_l1603_160345

-- Define the conditions as variables and hypotheses
variable (num_sweaters : ℕ := 28)
variable (balls_per_sweater : ℕ := 4)
variable (price_per_sweater : ℕ := 35)
variable (gain_from_sales : ℕ := 308)

-- Define derived values
def total_revenue : ℕ := num_sweaters * price_per_sweater
def total_cost_of_yarn : ℕ := total_revenue - gain_from_sales
def total_balls_of_yarn : ℕ := num_sweaters * balls_per_sweater
def cost_per_ball_of_yarn : ℕ := total_cost_of_yarn / total_balls_of_yarn

-- The theorem to be proven
theorem ball_of_yarn_costs_6 :
  cost_per_ball_of_yarn = 6 :=
by sorry

end NUMINAMATH_GPT_ball_of_yarn_costs_6_l1603_160345


namespace NUMINAMATH_GPT_never_2003_pieces_l1603_160313

theorem never_2003_pieces :
  ¬∃ n : ℕ, (n = 5 + 4 * k) ∧ (n = 2003) :=
by
  sorry

end NUMINAMATH_GPT_never_2003_pieces_l1603_160313


namespace NUMINAMATH_GPT_pen_more_expensive_than_two_notebooks_l1603_160369

variable (T R C : ℝ)

-- Conditions
axiom cond1 : T + R + C = 120
axiom cond2 : 5 * T + 2 * R + 3 * C = 350

-- Theorem statement
theorem pen_more_expensive_than_two_notebooks :
  R > 2 * T :=
by
  -- omit the actual proof, but check statement correctness
  sorry

end NUMINAMATH_GPT_pen_more_expensive_than_two_notebooks_l1603_160369


namespace NUMINAMATH_GPT_num_four_digit_with_5_or_7_l1603_160312

def num_four_digit_without_5_or_7 : Nat := 7 * 8 * 8 * 8

def num_four_digit_total : Nat := 9999 - 1000 + 1

theorem num_four_digit_with_5_or_7 : 
  num_four_digit_total - num_four_digit_without_5_or_7 = 5416 :=
by
  sorry

end NUMINAMATH_GPT_num_four_digit_with_5_or_7_l1603_160312


namespace NUMINAMATH_GPT_calculation_correct_l1603_160386

theorem calculation_correct : (18 / (3 + 9 - 6)) * 4 = 12 :=
by
  sorry

end NUMINAMATH_GPT_calculation_correct_l1603_160386


namespace NUMINAMATH_GPT_find_y_from_triangle_properties_l1603_160316

-- Define angle measures according to the given conditions
def angle_BAC := 45
def angle_CDE := 72

-- Define the proof problem
theorem find_y_from_triangle_properties
: ∀ (y : ℝ), (∃ (BAC ACB ABC ADC ADE AED DEB : ℝ),
    angle_BAC = 45 ∧
    angle_CDE = 72 ∧
    BAC + ACB + ABC = 180 ∧
    ADC = 180 ∧
    ADE = 180 - angle_CDE ∧
    EAD = angle_BAC ∧
    AED + ADE + EAD = 180 ∧
    DEB = 180 - AED ∧
    y = DEB) →
    y = 153 :=
by sorry

end NUMINAMATH_GPT_find_y_from_triangle_properties_l1603_160316


namespace NUMINAMATH_GPT_part1_part2_l1603_160333

theorem part1 (a x y : ℝ) (h1 : 3 * x - y = 2 * a - 5) (h2 : x + 2 * y = 3 * a + 3)
  (hx : x > 0) (hy : y > 0) : a > 1 :=
sorry

theorem part2 (a b : ℝ) (ha : a > 1) (h3 : a - b = 4) (hb : b < 2) : 
  -2 < a + b ∧ a + b < 8 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1603_160333


namespace NUMINAMATH_GPT_stairs_left_to_climb_l1603_160306

def total_stairs : ℕ := 96
def climbed_stairs : ℕ := 74

theorem stairs_left_to_climb : total_stairs - climbed_stairs = 22 := by
  sorry

end NUMINAMATH_GPT_stairs_left_to_climb_l1603_160306


namespace NUMINAMATH_GPT_total_cost_of_items_l1603_160351

-- Definitions based on conditions in a)
def price_of_caramel : ℕ := 3
def price_of_candy_bar : ℕ := 2 * price_of_caramel
def price_of_cotton_candy : ℕ := (4 * price_of_candy_bar) / 2
def cost_of_6_candy_bars : ℕ := 6 * price_of_candy_bar
def cost_of_3_caramels : ℕ := 3 * price_of_caramel

-- Problem statement to be proved
theorem total_cost_of_items : cost_of_6_candy_bars + cost_of_3_caramels + price_of_cotton_candy = 57 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_of_items_l1603_160351


namespace NUMINAMATH_GPT_compare_fractions_neg_l1603_160399

theorem compare_fractions_neg : (- (2 / 3) > - (3 / 4)) :=
  sorry

end NUMINAMATH_GPT_compare_fractions_neg_l1603_160399


namespace NUMINAMATH_GPT_total_employees_l1603_160389

def part_time_employees : ℕ := 2047
def full_time_employees : ℕ := 63109
def contractors : ℕ := 1500
def interns : ℕ := 333
def consultants : ℕ := 918

theorem total_employees : 
  part_time_employees + full_time_employees + contractors + interns + consultants = 66907 := 
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_total_employees_l1603_160389


namespace NUMINAMATH_GPT_weaving_sum_first_seven_days_l1603_160348

noncomputable def arithmetic_sequence (a_1 d : ℕ) (n : ℕ) : ℕ := a_1 + (n - 1) * d

theorem weaving_sum_first_seven_days
  (a_1 d : ℕ) :
  (arithmetic_sequence a_1 d 1) + (arithmetic_sequence a_1 d 2) + (arithmetic_sequence a_1 d 3) = 9 →
  (arithmetic_sequence a_1 d 2) + (arithmetic_sequence a_1 d 4) + (arithmetic_sequence a_1 d 6) = 15 →
  (arithmetic_sequence a_1 d 1) + (arithmetic_sequence a_1 d 2) + (arithmetic_sequence a_1 d 3) +
  (arithmetic_sequence a_1 d 4) + (arithmetic_sequence a_1 d 5) +
  (arithmetic_sequence a_1 d 6) + (arithmetic_sequence a_1 d 7) = 35 := by
  sorry

end NUMINAMATH_GPT_weaving_sum_first_seven_days_l1603_160348


namespace NUMINAMATH_GPT_smallest_N_constant_l1603_160337

-- Define the property to be proven
theorem smallest_N_constant (a b c : ℝ) 
  (h₁ : a + b > c) (h₂ : a + c > b) (h₃ : b + c > a) (h₄ : k = 0):
  (a^2 + b^2 + k) / c^2 > 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_N_constant_l1603_160337


namespace NUMINAMATH_GPT_samuel_initial_speed_l1603_160359

/-
Samuel is driving to San Francisco’s Comic-Con in his car and he needs to travel 600 miles to the hotel where he made a reservation. 
He drives at a certain speed for 3 hours straight, then he speeds up to 80 miles/hour for 4 hours. 
Now, he is 130 miles away from the hotel. What was his initial speed?
-/

theorem samuel_initial_speed : 
  ∃ v : ℝ, (3 * v + 320 = 470) ↔ (v = 50) :=
by
  use 50
  /- detailed proof goes here -/
  sorry

end NUMINAMATH_GPT_samuel_initial_speed_l1603_160359


namespace NUMINAMATH_GPT_max_value_of_expression_l1603_160373

theorem max_value_of_expression (x y : ℝ) 
  (h : (x - 4)^2 / 4 + y^2 / 9 = 1) : 
  (x^2 / 4 + y^2 / 9 ≤ 9) ∧ ∃ x y, (x - 4)^2 / 4 + y^2 / 9 = 1 ∧ x^2 / 4 + y^2 / 9 = 9 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l1603_160373


namespace NUMINAMATH_GPT_cost_of_dozen_pens_l1603_160349

-- Define the costs and conditions as given in the problem.
def cost_of_pen (x : ℝ) : ℝ := 5 * x
def cost_of_pencil (x : ℝ) : ℝ := x

-- The given conditions transformed into Lean definitions.
def condition1 (x : ℝ) : Prop := 3 * cost_of_pen x + 5 * cost_of_pencil x = 100
def condition2 (x : ℝ) : Prop := cost_of_pen x / cost_of_pencil x = 5

-- Prove that the cost of one dozen pens is Rs. 300.
theorem cost_of_dozen_pens : ∃ x : ℝ, condition1 x ∧ condition2 x ∧ 12 * cost_of_pen x = 300 := by
  sorry

end NUMINAMATH_GPT_cost_of_dozen_pens_l1603_160349


namespace NUMINAMATH_GPT_existential_inequality_false_iff_l1603_160320

theorem existential_inequality_false_iff {a : ℝ} :
  (∀ x : ℝ, x^2 + a * x - 2 * a ≥ 0) ↔ -8 ≤ a ∧ a ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_existential_inequality_false_iff_l1603_160320


namespace NUMINAMATH_GPT_equation_of_line_m_l1603_160363

-- Given conditions
def point (α : Type*) := α × α

def l_eq (p : point ℝ) : Prop := p.1 + 3 * p.2 = 7 -- Equation of line l
def m_intercept : point ℝ := (1, 2) -- Intersection point of l and m
def q : point ℝ := (2, 5) -- Point Q
def q'' : point ℝ := (5, 0) -- Point Q''

-- Proving the equation of line m
theorem equation_of_line_m (m_eq : point ℝ → Prop) :
  (∀ P : point ℝ, m_eq P ↔ P.2 = 2 * P.1 - 2) ↔
  (∃ P : point ℝ, m_eq P ∧ P = (5, 0)) :=
sorry

end NUMINAMATH_GPT_equation_of_line_m_l1603_160363


namespace NUMINAMATH_GPT_f_sub_f_inv_eq_2022_l1603_160398

def f (n : ℕ) : ℕ := 2 * n
def f_inv (n : ℕ) : ℕ := n

theorem f_sub_f_inv_eq_2022 : f 2022 - f_inv 2022 = 2022 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_f_sub_f_inv_eq_2022_l1603_160398


namespace NUMINAMATH_GPT_sum_n_terms_max_sum_n_l1603_160302

variable {a : ℕ → ℚ} (S : ℕ → ℚ)
variable (d a_1 : ℚ)

-- Conditions given in the problem
axiom sum_first_10 : S 10 = 125 / 7
axiom sum_first_20 : S 20 = -250 / 7
axiom sum_arithmetic_seq : ∀ n, S n = n * (a 1 + a n) / 2

-- Define the first term and common difference for the arithmetic sequence
axiom common_difference : ∀ n, a n = a_1 + (n - 1) * d

-- Theorem 1: Sum of the first n terms
theorem sum_n_terms (n : ℕ) : S n = (75 * n - 5 * n^2) / 14 := 
  sorry

-- Theorem 2: Value of n that maximizes S_n
theorem max_sum_n : n = 7 ∨ n = 8 ↔ (∀ m, S m ≤ S 7 ∨ S m ≤ S 8) := 
  sorry

end NUMINAMATH_GPT_sum_n_terms_max_sum_n_l1603_160302


namespace NUMINAMATH_GPT_quadratic_roots_evaluation_l1603_160388

theorem quadratic_roots_evaluation (x1 x2 : ℝ) (h1 : x1 + x2 = 1) (h2 : x1 * x2 = -2) :
  (1 + x1) + x2 * (1 - x1) = 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_evaluation_l1603_160388


namespace NUMINAMATH_GPT_shaded_region_area_proof_l1603_160308

/-- Define the geometric properties of the problem -/
structure Rectangle :=
  (width : ℝ)
  (height : ℝ)

structure Circle :=
  (radius : ℝ)
  (center : ℝ × ℝ)

noncomputable def shaded_region_area (rect : Rectangle) (circle1 circle2 : Circle) : ℝ :=
  let rect_area := rect.width * rect.height
  let circle_area := (Real.pi * circle1.radius ^ 2) + (Real.pi * circle2.radius ^ 2)
  rect_area - circle_area

theorem shaded_region_area_proof : shaded_region_area 
  {width := 10, height := 12} 
  {radius := 3, center := (0, 0)} 
  {radius := 3, center := (12, 10)} = 120 - 18 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_shaded_region_area_proof_l1603_160308


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l1603_160338

def f (x : ℕ) : ℕ := x*x + x

theorem no_positive_integer_solutions :
  ∀ (a b : ℕ), a > 0 → b > 0 → 4 * (f a) ≠ (f b) :=
by
  intro a b a_pos b_pos
  sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_l1603_160338


namespace NUMINAMATH_GPT_problem_statement_l1603_160366

variable (P : ℕ → Prop)

theorem problem_statement
    (h1 : P 2)
    (h2 : ∀ k : ℕ, k > 0 → P k → P (k + 2)) :
    ∀ n : ℕ, n > 0 → 2 ∣ n → P n :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1603_160366


namespace NUMINAMATH_GPT_chess_group_players_l1603_160377

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 36) : n = 9 :=
by
  sorry

end NUMINAMATH_GPT_chess_group_players_l1603_160377


namespace NUMINAMATH_GPT_max_chord_length_line_eq_orthogonal_vectors_line_eq_l1603_160397

-- Definitions
def circle_eq (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 4
def point_P (x y : ℝ) : Prop := x = 2 ∧ y = 1
def line_eq (slope intercept x y : ℝ) : Prop := y = slope * x + intercept

-- Problem 1: Prove the equation of line l that maximizes the length of chord AB
theorem max_chord_length_line_eq : 
  (∀ x y : ℝ, point_P x y → line_eq 1 (-1) x y)
  ∧ (∀ x y : ℝ, circle_eq x y → point_P x y) 
  → (∀ x y : ℝ, line_eq 1 (-1) x y) :=
by sorry

-- Problem 2: Prove the equation of line l given orthogonality condition of vectors
theorem orthogonal_vectors_line_eq : 
  (∀ x y : ℝ, point_P x y → line_eq (-1) 3 x y)
  ∧ (∀ x y : ℝ, circle_eq x y → point_P x y) 
  → (∀ x y : ℝ, line_eq (-1) 3 x y) :=
by sorry

end NUMINAMATH_GPT_max_chord_length_line_eq_orthogonal_vectors_line_eq_l1603_160397


namespace NUMINAMATH_GPT_quadratic_value_l1603_160353

theorem quadratic_value (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : 4 * a + 2 * b + c = 3) :
  a + 2 * b + 3 * c = 7 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_value_l1603_160353


namespace NUMINAMATH_GPT_pq_square_eq_169_div_4_l1603_160314

-- Defining the quadratic equation and the condition on solutions p and q.
def quadratic_eq (x : ℚ) : Prop := 2 * x^2 + 7 * x - 15 = 0

-- Defining the specific solutions p and q.
def p : ℚ := 3 / 2
def q : ℚ := -5

-- The main theorem stating that (p - q)^2 = 169 / 4 given the conditions.
theorem pq_square_eq_169_div_4 (hp : quadratic_eq p) (hq : quadratic_eq q) : (p - q) ^ 2 = 169 / 4 :=
by
  -- Proof is omitted using sorry
  sorry

end NUMINAMATH_GPT_pq_square_eq_169_div_4_l1603_160314


namespace NUMINAMATH_GPT_apples_in_bowl_l1603_160384

variable {A : ℕ}

theorem apples_in_bowl
  (initial_oranges : ℕ)
  (removed_oranges : ℕ)
  (final_oranges : ℕ)
  (total_fruit : ℕ)
  (fraction_apples : ℚ) :
  initial_oranges = 25 →
  removed_oranges = 19 →
  final_oranges = initial_oranges - removed_oranges →
  fraction_apples = (70 : ℚ) / (100 : ℚ) →
  final_oranges = total_fruit * (30 : ℚ) / (100 : ℚ) →
  A = total_fruit * fraction_apples →
  A = 14 :=
by
  sorry

end NUMINAMATH_GPT_apples_in_bowl_l1603_160384


namespace NUMINAMATH_GPT_min_value_pt_qu_rv_sw_l1603_160372

theorem min_value_pt_qu_rv_sw (p q r s t u v w : ℝ) (h1 : p * q * r * s = 8) (h2 : t * u * v * w = 27) :
  (p * t) ^ 2 + (q * u) ^ 2 + (r * v) ^ 2 + (s * w) ^ 2 ≥ 96 :=
by
  sorry

end NUMINAMATH_GPT_min_value_pt_qu_rv_sw_l1603_160372


namespace NUMINAMATH_GPT_seq_bound_gt_pow_two_l1603_160323

theorem seq_bound_gt_pow_two (a : Fin 101 → ℕ) 
  (h1 : a 1 > a 0) 
  (h2 : ∀ n : Fin 99, a (n + 2) = 3 * a (n + 1) - 2 * a n) :
  a 100 > 2 ^ 99 :=
sorry

end NUMINAMATH_GPT_seq_bound_gt_pow_two_l1603_160323


namespace NUMINAMATH_GPT_ellipse_transform_circle_l1603_160315

theorem ellipse_transform_circle (a b x y : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b)
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1)
  (y' : ℝ)
  (h_transform : y' = (a / b) * y) :
  x^2 + y'^2 = a^2 :=
by
  sorry

end NUMINAMATH_GPT_ellipse_transform_circle_l1603_160315


namespace NUMINAMATH_GPT_distance_between_midpoints_l1603_160324

-- Conditions
def AA' := 68 -- in centimeters
def BB' := 75 -- in centimeters
def CC' := 112 -- in centimeters
def DD' := 133 -- in centimeters

-- Question: Prove the distance between the midpoints of A'C' and B'D' is 14 centimeters
theorem distance_between_midpoints :
  let midpoint_A'C' := (AA' + CC') / 2
  let midpoint_B'D' := (BB' + DD') / 2
  (midpoint_B'D' - midpoint_A'C' = 14) :=
by
  sorry

end NUMINAMATH_GPT_distance_between_midpoints_l1603_160324


namespace NUMINAMATH_GPT_ab_is_zero_l1603_160370

-- Define that a function is odd
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Define the given function f
def f (a b : ℝ) (x : ℝ) : ℝ := 2 * x^3 + a * x^2 + b - 2

-- The main theorem to prove
theorem ab_is_zero (a b : ℝ) (h_odd : is_odd (f a b)) : a * b = 0 := 
sorry

end NUMINAMATH_GPT_ab_is_zero_l1603_160370


namespace NUMINAMATH_GPT_weight_difference_l1603_160325

variable (W_A W_D : Nat)

theorem weight_difference : W_A - W_D = 15 :=
by
  -- Given conditions
  have h1 : W_A = 67 := sorry
  have h2 : W_D = 52 := sorry
  -- Proof
  sorry

end NUMINAMATH_GPT_weight_difference_l1603_160325


namespace NUMINAMATH_GPT_a_beats_b_time_difference_l1603_160374

theorem a_beats_b_time_difference
  (d : ℝ) (d_A : ℝ) (d_B : ℝ)
  (t_A : ℝ)
  (h1 : d = 1000)
  (h2 : d_A = d)
  (h3 : d_B = d - 60)
  (h4 : t_A = 235) :
  (t_A - (d_B * t_A / d_A)) = 14.1 :=
by sorry

end NUMINAMATH_GPT_a_beats_b_time_difference_l1603_160374


namespace NUMINAMATH_GPT_red_blood_cells_surface_area_l1603_160301

-- Define the body surface area of an adult
def body_surface_area : ℝ := 1800

-- Define the multiplying factor for the surface areas of red blood cells
def multiplier : ℝ := 2000

-- Define the sum of the surface areas of all red blood cells
def sum_surface_area : ℝ := multiplier * body_surface_area

-- Define the expected sum in scientific notation
def expected_sum : ℝ := 3.6 * 10^6

-- The theorem that needs to be proved
theorem red_blood_cells_surface_area :
  sum_surface_area = expected_sum :=
by
  sorry

end NUMINAMATH_GPT_red_blood_cells_surface_area_l1603_160301


namespace NUMINAMATH_GPT_sum_of_abcd_l1603_160334

theorem sum_of_abcd (a b c d: ℝ) (h₁: a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h₂: c + d = 10 * a) (h₃: c * d = -11 * b) (h₄: a + b = 10 * c) (h₅: a * b = -11 * d)
  : a + b + c + d = 1210 := by
  sorry

end NUMINAMATH_GPT_sum_of_abcd_l1603_160334


namespace NUMINAMATH_GPT_cost_prices_max_profit_l1603_160376

theorem cost_prices (a b : ℝ) (x : ℝ) (y : ℝ)
    (h1 : a - b = 500)
    (h2 : 40000 / a = 30000 / b)
    (h3 : 0 ≤ x ∧ x ≤ 20)
    (h4 : 2000 * x + 1500 * (20 - x) ≤ 36000) :
    a = 2000 ∧ b = 1500 := sorry

theorem max_profit (x : ℝ) (y : ℝ)
    (h1 : 0 ≤ x ∧ x ≤ 12) :
    y = 200 * x + 6000 ∧ y ≤ 8400 := sorry

end NUMINAMATH_GPT_cost_prices_max_profit_l1603_160376


namespace NUMINAMATH_GPT_no_valid_m_n_l1603_160340

theorem no_valid_m_n (m n : ℕ) (hm : m ≠ 0) (hn : n ≠ 0) : ¬ (m * n ∣ 3^m + 1 ∧ m * n ∣ 3^n + 1) :=
by
  sorry

end NUMINAMATH_GPT_no_valid_m_n_l1603_160340


namespace NUMINAMATH_GPT_ratio_of_radii_l1603_160305

variable (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2)

theorem ratio_of_radii (ha : a > 0) (hb : b > 0) : (a / b = 1 / Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_radii_l1603_160305


namespace NUMINAMATH_GPT_B_work_time_alone_l1603_160352

theorem B_work_time_alone
  (A_rate : ℝ := 1 / 8)
  (together_rate : ℝ := 3 / 16) :
  ∃ (B_days : ℝ), B_days = 16 :=
by
  sorry

end NUMINAMATH_GPT_B_work_time_alone_l1603_160352


namespace NUMINAMATH_GPT_problem_solution_l1603_160347

noncomputable def problem (p q : ℝ) (α β γ δ : ℝ) : Prop :=
  (α^2 + p * α - 1 = 0) ∧
  (β^2 + p * β - 1 = 0) ∧
  (γ^2 + q * γ + 1 = 0) ∧
  (δ^2 + q * δ + 1 = 0) →
  (α - γ) * (β - γ) * (α - δ) * (β - δ) = p^2 - q^2

theorem problem_solution (p q α β γ δ : ℝ) : 
  problem p q α β γ δ := 
by sorry

end NUMINAMATH_GPT_problem_solution_l1603_160347


namespace NUMINAMATH_GPT_min_sum_of_factors_l1603_160339

theorem min_sum_of_factors 
  (a b c: ℕ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 0)
  (h4: a * b * c = 2450) :
  a + b + c ≥ 76 :=
sorry

end NUMINAMATH_GPT_min_sum_of_factors_l1603_160339


namespace NUMINAMATH_GPT_polygon_sides_l1603_160396

theorem polygon_sides (sum_of_interior_angles : ℕ) (h : sum_of_interior_angles = 1260) : ∃ n : ℕ, (n-2) * 180 = sum_of_interior_angles ∧ n = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_polygon_sides_l1603_160396


namespace NUMINAMATH_GPT_condition_necessity_not_sufficiency_l1603_160307

theorem condition_necessity_not_sufficiency (a : ℝ) : 
  (2 / a < 1 → a^2 > 4) ∧ ¬(2 / a < 1 ↔ a^2 > 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_condition_necessity_not_sufficiency_l1603_160307


namespace NUMINAMATH_GPT_number_of_possible_orders_l1603_160330

-- Define the total number of bowlers participating in the playoff
def num_bowlers : ℕ := 6

-- Define the number of games
def num_games : ℕ := 5

-- Define the number of possible outcomes per game
def outcomes_per_game : ℕ := 2

-- Prove the total number of possible orders for bowlers to receive prizes
theorem number_of_possible_orders : (outcomes_per_game ^ num_games) = 32 :=
by sorry

end NUMINAMATH_GPT_number_of_possible_orders_l1603_160330


namespace NUMINAMATH_GPT_total_cost_4kg_mangos_3kg_rice_5kg_flour_l1603_160355

def cost_per_kg_mangos (M : ℝ) (R : ℝ) := (10 * M = 24 * R)
def cost_per_kg_flour_equals_rice (F : ℝ) (R : ℝ) := (6 * F = 2 * R)
def cost_of_flour (F : ℝ) := (F = 24)

theorem total_cost_4kg_mangos_3kg_rice_5kg_flour 
  (M R F : ℝ) 
  (h1 : cost_per_kg_mangos M R) 
  (h2 : cost_per_kg_flour_equals_rice F R) 
  (h3 : cost_of_flour F) : 
  4 * M + 3 * R + 5 * F = 1027.2 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_cost_4kg_mangos_3kg_rice_5kg_flour_l1603_160355


namespace NUMINAMATH_GPT_only_real_solution_x_eq_6_l1603_160381

theorem only_real_solution_x_eq_6 : ∀ x : ℝ, (x^3 + (x+2)^3 + (x+4)^3 = (x+6)^3) → x = 6 :=
by
  sorry

end NUMINAMATH_GPT_only_real_solution_x_eq_6_l1603_160381


namespace NUMINAMATH_GPT_min_value_of_u_l1603_160379

theorem min_value_of_u : ∀ (x y : ℝ), x ∈ Set.Ioo (-2) 2 → y ∈ Set.Ioo (-2) 2 → x * y = -1 → 
  (∀ u, u = (4 / (4 - x^2)) + (9 / (9 - y^2)) → u ≥ 12 / 5) :=
by
  intros x y hx hy hxy u hu
  sorry

end NUMINAMATH_GPT_min_value_of_u_l1603_160379


namespace NUMINAMATH_GPT_fraction_of_plot_occupied_by_beds_l1603_160319

-- Define the conditions based on plot area and number of beds
def plot_area : ℕ := 64
def total_beds : ℕ := 13
def outer_beds : ℕ := 12
def central_bed_area : ℕ := 4 * 4

-- The proof statement showing that fraction of the plot occupied by the beds is 15/32
theorem fraction_of_plot_occupied_by_beds : 
  (central_bed_area + (plot_area - central_bed_area)) / plot_area = 15 / 32 := 
sorry

end NUMINAMATH_GPT_fraction_of_plot_occupied_by_beds_l1603_160319


namespace NUMINAMATH_GPT_common_difference_is_4_l1603_160375

variable {a_n : ℕ → ℝ}
variable {S_n : ℕ → ℝ}
variable {a_4 a_5 S_6 : ℝ}
variable {d : ℝ}

-- Definitions of conditions given in the problem
def a4_cond : a_4 = a_n 4 := sorry
def a5_cond : a_5 = a_n 5 := sorry
def sum_six : S_6 = (6/2) * (2 * a_n 1 + 5 * d) := sorry
def term_sum : a_4 + a_5 = 24 := sorry

-- Proof statement
theorem common_difference_is_4 : d = 4 :=
by
  sorry

end NUMINAMATH_GPT_common_difference_is_4_l1603_160375


namespace NUMINAMATH_GPT_Ramu_spent_on_repairs_l1603_160344

theorem Ramu_spent_on_repairs (purchase_price : ℝ) (selling_price : ℝ) (profit_percent : ℝ) (R : ℝ) 
  (h1 : purchase_price = 42000) 
  (h2 : selling_price = 61900) 
  (h3 : profit_percent = 12.545454545454545) 
  (h4 : selling_price = purchase_price + R + (profit_percent / 100) * (purchase_price + R)) : 
  R = 13000 :=
by
  sorry

end NUMINAMATH_GPT_Ramu_spent_on_repairs_l1603_160344


namespace NUMINAMATH_GPT_fraction_of_speedsters_l1603_160336

/-- Let S denote the total number of Speedsters and T denote the total inventory. 
    Given the following conditions:
    1. 54 Speedster convertibles constitute 3/5 of all Speedsters (S).
    2. There are 30 vehicles that are not Speedsters.

    Prove that the fraction of the current inventory that is Speedsters is 3/4.
-/
theorem fraction_of_speedsters (S T : ℕ)
  (h1 : 3 / 5 * S = 54)
  (h2 : T = S + 30) :
  (S : ℚ) / T = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_speedsters_l1603_160336


namespace NUMINAMATH_GPT_pieces_per_serving_l1603_160335

-- Definitions based on conditions
def jaredPopcorn : Nat := 90
def friendPopcorn : Nat := 60
def numberOfFriends : Nat := 3
def totalServings : Nat := 9

-- Statement to verify
theorem pieces_per_serving : 
  ((jaredPopcorn + numberOfFriends * friendPopcorn) / totalServings) = 30 :=
by
  sorry

end NUMINAMATH_GPT_pieces_per_serving_l1603_160335


namespace NUMINAMATH_GPT_negation_of_cube_of_every_odd_number_is_odd_l1603_160341

theorem negation_of_cube_of_every_odd_number_is_odd:
  ¬ (∀ n : ℤ, (n % 2 = 1 → (n^3 % 2 = 1))) ↔ ∃ n : ℤ, (n % 2 = 1 ∧ ¬ (n^3 % 2 = 1)) := 
by
  sorry

end NUMINAMATH_GPT_negation_of_cube_of_every_odd_number_is_odd_l1603_160341


namespace NUMINAMATH_GPT_negation_of_existence_statement_l1603_160311

theorem negation_of_existence_statement :
  (¬ (∃ x : ℝ, x^2 + x + 1 < 0)) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_statement_l1603_160311


namespace NUMINAMATH_GPT_option_C_correct_l1603_160364

theorem option_C_correct (x : ℝ) : x^3 * x^2 = x^5 := sorry

end NUMINAMATH_GPT_option_C_correct_l1603_160364


namespace NUMINAMATH_GPT_train_passes_jogger_in_46_seconds_l1603_160371

-- Definitions directly from conditions
def jogger_speed_kmh : ℕ := 10
def train_speed_kmh : ℕ := 46
def initial_distance_m : ℕ := 340
def train_length_m : ℕ := 120

-- Additional computed definitions based on conditions
def relative_speed_ms : ℕ := (train_speed_kmh - jogger_speed_kmh) * 1000 / 3600
def total_distance_m : ℕ := initial_distance_m + train_length_m

-- Prove that the time it takes for the train to pass the jogger is 46 seconds
theorem train_passes_jogger_in_46_seconds : total_distance_m / relative_speed_ms = 46 := by
  sorry

end NUMINAMATH_GPT_train_passes_jogger_in_46_seconds_l1603_160371


namespace NUMINAMATH_GPT_apple_distribution_l1603_160343

theorem apple_distribution : 
  (∀ (a b c d : ℕ), (a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) → (a + b + c + d = 30) → 
  ∃ k : ℕ, k = (Nat.choose 29 3) ∧ k = 3276) :=
by
  intros a b c d h_pos h_sum
  use Nat.choose 29 3
  have h_eq : Nat.choose 29 3 = 3276 := by sorry
  exact ⟨rfl, h_eq⟩

end NUMINAMATH_GPT_apple_distribution_l1603_160343


namespace NUMINAMATH_GPT_area_difference_of_circles_l1603_160318

theorem area_difference_of_circles (circumference_large: ℝ) (half_radius_relation: ℝ → ℝ) (hl: circumference_large = 36) (hr: ∀ R, half_radius_relation R = R / 2) :
  ∃ R r, R = 18 / π ∧ r = 9 / π ∧ (π * R ^ 2 - π * r ^ 2) = 243 / π :=
by 
  sorry

end NUMINAMATH_GPT_area_difference_of_circles_l1603_160318


namespace NUMINAMATH_GPT_find_a_for_inequality_l1603_160300

theorem find_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, (x < -2 ∨ x > 3) → -2 * x^2 + a * x + 6 > 0) → a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_for_inequality_l1603_160300


namespace NUMINAMATH_GPT_q_value_l1603_160393

theorem q_value (p q : ℝ) (hpq1 : 1 < p) (hpql : p < q) (hq_condition : (1 / p) + (1 / q) = 1) (hpq2 : p * q = 8) : q = 4 + 2 * Real.sqrt 2 :=
  sorry

end NUMINAMATH_GPT_q_value_l1603_160393


namespace NUMINAMATH_GPT_arithmetic_mean_squares_l1603_160346

theorem arithmetic_mean_squares (n : ℕ) (h : 0 < n) :
  let S_n2 := (n * (n + 1) * (2 * n + 1)) / 6 
  let A_n2 := S_n2 / n
  A_n2 = ((n + 1) * (2 * n + 1)) / 6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_squares_l1603_160346


namespace NUMINAMATH_GPT_find_ratio_l1603_160310

-- Definition of the system of equations with k = 5
def system_of_equations (x y z : ℝ) :=
  x + 10 * y + 5 * z = 0 ∧
  2 * x + 5 * y + 4 * z = 0 ∧
  3 * x + 6 * y + 5 * z = 0

-- Proof that if (x, y, z) solves the system, then yz / x^2 = -3 / 49
theorem find_ratio (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : system_of_equations x y z) :
  (y * z) / (x ^ 2) = -3 / 49 :=
by
  -- Substitute the system of equations and solve for the ratio.
  sorry

end NUMINAMATH_GPT_find_ratio_l1603_160310


namespace NUMINAMATH_GPT_area_of_region_l1603_160382

-- Definitions drawn from conditions
def circle_radius := 36
def num_small_circles := 8

-- Main statement to be proven
theorem area_of_region :
  ∃ K : ℝ, 
    K = π * (circle_radius ^ 2) - num_small_circles * π * ((circle_radius * (Real.sqrt 2 - 1)) ^ 2) ∧
    ⌊ K ⌋ = ⌊ π * (circle_radius ^ 2) - num_small_circles * π * ((circle_radius * (Real.sqrt 2 - 1)) ^ 2) ⌋ :=
  sorry

end NUMINAMATH_GPT_area_of_region_l1603_160382


namespace NUMINAMATH_GPT_sufficient_not_necessary_l1603_160380

theorem sufficient_not_necessary (x : ℝ) : abs x < 2 → (x^2 - x - 6 < 0) ∧ (¬(x^2 - x - 6 < 0) → abs x ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l1603_160380


namespace NUMINAMATH_GPT_legacy_earnings_per_hour_l1603_160342

-- Define the conditions
def totalFloors : ℕ := 4
def roomsPerFloor : ℕ := 10
def hoursPerRoom : ℕ := 6
def totalEarnings : ℝ := 3600

-- The statement to prove
theorem legacy_earnings_per_hour :
  (totalFloors * roomsPerFloor * hoursPerRoom) = 240 → 
  (totalEarnings / (totalFloors * roomsPerFloor * hoursPerRoom)) = 15 := by
  intros h
  sorry

end NUMINAMATH_GPT_legacy_earnings_per_hour_l1603_160342


namespace NUMINAMATH_GPT_combine_square_roots_l1603_160378

def can_be_combined (x y: ℝ) : Prop :=
  ∃ k: ℝ, y = k * x

theorem combine_square_roots :
  let sqrt12 := 2 * Real.sqrt 3
  let sqrt1_3 := Real.sqrt 1 / Real.sqrt 3
  let sqrt18 := 3 * Real.sqrt 2
  let sqrt27 := 6 * Real.sqrt 3
  can_be_combined (Real.sqrt 3) sqrt12 ∧
  can_be_combined (Real.sqrt 3) sqrt1_3 ∧
  ¬ can_be_combined (Real.sqrt 3) sqrt18 ∧
  can_be_combined (Real.sqrt 3) sqrt27 :=
by
  sorry

end NUMINAMATH_GPT_combine_square_roots_l1603_160378


namespace NUMINAMATH_GPT_face_card_then_number_card_prob_l1603_160350

-- Definitions from conditions
def num_cards := 52
def num_face_cards := 12
def num_number_cards := 40
def total_ways_to_pick_two_cards := 52 * 51

-- Theorem statement
theorem face_card_then_number_card_prob : 
  (num_face_cards * num_number_cards) / total_ways_to_pick_two_cards = (40 : ℚ) / 221 :=
by
  sorry

end NUMINAMATH_GPT_face_card_then_number_card_prob_l1603_160350


namespace NUMINAMATH_GPT_base7_first_digit_l1603_160361

noncomputable def first_base7_digit : ℕ := 625

theorem base7_first_digit (n : ℕ) (h : n = 625) : ∃ (d : ℕ), d = 12 ∧ (d * 49 ≤ n) ∧ (n < (d + 1) * 49) :=
by
  sorry

end NUMINAMATH_GPT_base7_first_digit_l1603_160361


namespace NUMINAMATH_GPT_average_weight_l1603_160362

variable (A B C : ℕ)

theorem average_weight (h1 : A + B = 140) (h2 : B + C = 100) (h3 : B = 60) :
  (A + B + C) / 3 = 60 := 
sorry

end NUMINAMATH_GPT_average_weight_l1603_160362


namespace NUMINAMATH_GPT_prove_inequalities_l1603_160365

variable {a b c R r_a r_b r_c : ℝ}

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def has_circumradius (a b c R : ℝ) : Prop :=
  ∃ S : ℝ, S = a * b * c / (4 * R)

def has_exradii (a b c r_a r_b r_c : ℝ) : Prop :=
  ∃ S : ℝ, 
    r_a = 2 * S / (b + c - a) ∧
    r_b = 2 * S / (a + c - b) ∧
    r_c = 2 * S / (a + b - c)

theorem prove_inequalities
  (h_triangle : is_triangle a b c)
  (h_circumradius : has_circumradius a b c R)
  (h_exradii : has_exradii a b c r_a r_b r_c)
  (h_two_R_le_r_a : 2 * R ≤ r_a) :
  a > b ∧ a > c ∧ 2 * R > r_b ∧ 2 * R > r_c := 
sorry

end NUMINAMATH_GPT_prove_inequalities_l1603_160365


namespace NUMINAMATH_GPT_average_income_proof_l1603_160392

theorem average_income_proof:
  ∀ (A B C : ℝ),
    (A + B) / 2 = 5050 →
    (B + C) / 2 = 6250 →
    A = 4000 →
    (A + C) / 2 = 5200 := by
  sorry

end NUMINAMATH_GPT_average_income_proof_l1603_160392


namespace NUMINAMATH_GPT_translated_B_is_B_l1603_160304

def point : Type := ℤ × ℤ

def A : point := (-4, -1)
def A' : point := (-2, 2)
def B : point := (1, 1)
def B' : point := (3, 4)

def translation_vector (p1 p2 : point) : point :=
  (p2.1 - p1.1, p2.2 - p1.2)

def translate_point (p : point) (v : point) : point :=
  (p.1 + v.1, p.2 + v.2)

theorem translated_B_is_B' : translate_point B (translation_vector A A') = B' :=
by
  sorry

end NUMINAMATH_GPT_translated_B_is_B_l1603_160304


namespace NUMINAMATH_GPT_tax_paid_at_fifth_checkpoint_l1603_160385

variable {x : ℚ}

theorem tax_paid_at_fifth_checkpoint (x : ℚ) (h : (x / 2) + (x / 2 * 1 / 3) + (x / 3 * 1 / 4) + (x / 4 * 1 / 5) + (x / 5 * 1 / 6) = 1) :
  (x / 5 * 1 / 6) = 1 / 25 :=
sorry

end NUMINAMATH_GPT_tax_paid_at_fifth_checkpoint_l1603_160385


namespace NUMINAMATH_GPT_pentagon_area_l1603_160356

-- Define the lengths of the sides of the pentagon
def side1 := 18
def side2 := 25
def side3 := 30
def side4 := 28
def side5 := 25

-- Define the sides of the rectangle and triangle
def rectangle_length := side4
def rectangle_width := side2
def triangle_base := side1
def triangle_height := rectangle_width

-- Define areas of rectangle and right triangle
def area_rectangle := rectangle_length * rectangle_width
def area_triangle := (triangle_base * triangle_height) / 2

-- Define the total area of the pentagon
def total_area_pentagon := area_rectangle + area_triangle

theorem pentagon_area : total_area_pentagon = 925 := by
  sorry

end NUMINAMATH_GPT_pentagon_area_l1603_160356


namespace NUMINAMATH_GPT_filtration_minimum_l1603_160309

noncomputable def lg : ℝ → ℝ := sorry

theorem filtration_minimum (x : ℕ) (lg2 : ℝ) (lg3 : ℝ) (h1 : lg2 = 0.3010) (h2 : lg3 = 0.4771) :
  (2 / 3 : ℝ) ^ x ≤ 1 / 20 → x ≥ 8 :=
sorry

end NUMINAMATH_GPT_filtration_minimum_l1603_160309


namespace NUMINAMATH_GPT_inequality_solution_l1603_160354

theorem inequality_solution (x : ℝ) : (3 * x^2 - 4 * x - 4 < 0) ↔ (-2/3 < x ∧ x < 2) :=
sorry

end NUMINAMATH_GPT_inequality_solution_l1603_160354


namespace NUMINAMATH_GPT_intersecting_chords_l1603_160383

noncomputable def length_of_other_chord (x : ℝ) : ℝ :=
  3 * x + 8 * x

theorem intersecting_chords
  (a b : ℝ) (h1 : a = 12) (h2 : b = 18) (r1 r2 : ℝ) (h3 : r1/r2 = 3/8) :
  length_of_other_chord 3 = 33 := by
  sorry

end NUMINAMATH_GPT_intersecting_chords_l1603_160383
