import Mathlib

namespace NUMINAMATH_GPT_solution_is_consecutive_even_integers_l726_72602

def consecutive_even_integers_solution_exists : Prop :=
  ∃ (x y z w : ℕ), (x + y + z + w = 68) ∧ 
                   (y = x + 2) ∧ (z = x + 4) ∧ (w = x + 6) ∧
                   (x % 2 = 0) ∧ (y % 2 = 0) ∧ (z % 2 = 0) ∧ (w % 2 = 0)

theorem solution_is_consecutive_even_integers : consecutive_even_integers_solution_exists :=
sorry

end NUMINAMATH_GPT_solution_is_consecutive_even_integers_l726_72602


namespace NUMINAMATH_GPT_tan_sub_eq_one_third_l726_72614

theorem tan_sub_eq_one_third (α β : Real) (hα : Real.tan α = 3) (hβ : Real.tan β = 4/3) : 
  Real.tan (α - β) = 1/3 := by
  sorry

end NUMINAMATH_GPT_tan_sub_eq_one_third_l726_72614


namespace NUMINAMATH_GPT_mixture_replacement_l726_72693

theorem mixture_replacement
  (A B : ℕ)
  (hA : A = 48)
  (h_ratio1 : A / B = 4)
  (x : ℕ)
  (h_ratio2 : A / (B + x) = 2 / 3) :
  x = 60 :=
by
  sorry

end NUMINAMATH_GPT_mixture_replacement_l726_72693


namespace NUMINAMATH_GPT_minimum_positive_period_of_f_l726_72629

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem minimum_positive_period_of_f : 
  (∃ T : ℝ, T > 0 ∧ ∀ x : ℝ, f (x + T) = f x) ∧ 
  ∀ T' : ℝ, (T' > 0 ∧ ∀ x : ℝ, f (x + T') = f x) → T' ≥ Real.pi := 
sorry

end NUMINAMATH_GPT_minimum_positive_period_of_f_l726_72629


namespace NUMINAMATH_GPT_general_term_b_l726_72607

noncomputable def S (n : ℕ) : ℚ := sorry -- Define the sum of the first n terms sequence S_n
noncomputable def a (n : ℕ) : ℚ := sorry -- Define the sequence a_n
noncomputable def b (n : ℕ) : ℤ := Int.log 3 (|a n|) -- Define the sequence b_n using log base 3

-- Theorem stating the general formula for the sequence b_n
theorem general_term_b (n : ℕ) (h : 0 < n) :
  b n = -n :=
sorry -- We skip the proof, focusing on statement declaration

end NUMINAMATH_GPT_general_term_b_l726_72607


namespace NUMINAMATH_GPT_molecular_weight_neutralization_l726_72696

def molecular_weight_acetic_acid : ℝ := 
  (12.01 * 2) + (1.008 * 4) + (16.00 * 2)

def molecular_weight_sodium_hydroxide : ℝ := 
  22.99 + 16.00 + 1.008

def total_weight_acetic_acid (moles : ℝ) : ℝ := 
  molecular_weight_acetic_acid * moles

def total_weight_sodium_hydroxide (moles : ℝ) : ℝ := 
  molecular_weight_sodium_hydroxide * moles

def total_molecular_weight (moles_ac: ℝ) (moles_naoh : ℝ) : ℝ :=
  total_weight_acetic_acid moles_ac + 
  total_weight_sodium_hydroxide moles_naoh

theorem molecular_weight_neutralization :
  total_molecular_weight 7 10 = 820.344 :=
by
  sorry

end NUMINAMATH_GPT_molecular_weight_neutralization_l726_72696


namespace NUMINAMATH_GPT_chairs_carried_per_trip_l726_72670

theorem chairs_carried_per_trip (x : ℕ) (friends : ℕ) (trips : ℕ) (total_chairs : ℕ) 
  (h1 : friends = 4) (h2 : trips = 10) (h3 : total_chairs = 250) 
  (h4 : 5 * (trips * x) = total_chairs) : x = 5 :=
by sorry

end NUMINAMATH_GPT_chairs_carried_per_trip_l726_72670


namespace NUMINAMATH_GPT_system_solution_b_l726_72682

theorem system_solution_b (x y b : ℚ) 
  (h1 : 4 * x + 2 * y = b) 
  (h2 : 3 * x + 7 * y = 3 * b) 
  (hy : y = 3) : 
  b = 22 / 3 := 
by
  sorry

end NUMINAMATH_GPT_system_solution_b_l726_72682


namespace NUMINAMATH_GPT_Marie_finish_time_l726_72647

def Time := Nat × Nat -- Represents time as (hours, minutes)

def start_time : Time := (9, 0)
def finish_two_tasks_time : Time := (11, 20)
def total_tasks : Nat := 4

def minutes_since_start (t : Time) : Nat :=
  let (h, m) := t
  (h - 9) * 60 + m

def calculate_finish_time (start: Time) (two_tasks_finish: Time) (total_tasks: Nat) : Time :=
  let duration_two_tasks := minutes_since_start two_tasks_finish
  let duration_each_task := duration_two_tasks / 2
  let total_time := duration_each_task * total_tasks
  let total_minutes_after_start := total_time + minutes_since_start start
  let finish_hour := 9 + total_minutes_after_start / 60
  let finish_minute := total_minutes_after_start % 60
  (finish_hour, finish_minute)

theorem Marie_finish_time :
  calculate_finish_time start_time finish_two_tasks_time total_tasks = (13, 40) :=
by
  sorry

end NUMINAMATH_GPT_Marie_finish_time_l726_72647


namespace NUMINAMATH_GPT_overall_gain_is_10_percent_l726_72644

noncomputable def total_cost_price : ℝ := 700 + 500 + 300
noncomputable def total_gain : ℝ := 70 + 50 + 30
noncomputable def overall_gain_percentage : ℝ := (total_gain / total_cost_price) * 100

theorem overall_gain_is_10_percent :
  overall_gain_percentage = 10 :=
by
  sorry

end NUMINAMATH_GPT_overall_gain_is_10_percent_l726_72644


namespace NUMINAMATH_GPT_bobby_candy_left_l726_72636

theorem bobby_candy_left (initial_candies := 21) (first_eaten := 5) (second_eaten := 9) : 
  initial_candies - first_eaten - second_eaten = 7 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_bobby_candy_left_l726_72636


namespace NUMINAMATH_GPT_Travis_spends_on_cereal_l726_72683

theorem Travis_spends_on_cereal (boxes_per_week : ℕ) (cost_per_box : ℝ) (weeks_per_year : ℕ) 
  (h1 : boxes_per_week = 2) 
  (h2 : cost_per_box = 3.00) 
  (h3 : weeks_per_year = 52) 
: boxes_per_week * weeks_per_year * cost_per_box = 312.00 := 
by
  sorry

end NUMINAMATH_GPT_Travis_spends_on_cereal_l726_72683


namespace NUMINAMATH_GPT_exists_a_b_not_multiple_p_l726_72622

theorem exists_a_b_not_multiple_p (p : ℕ) (hp : Nat.Prime p) :
  ∃ a b : ℤ, ∀ m : ℤ, ¬ (m^3 + 2017 * a * m + b) ∣ (p : ℤ) :=
sorry

end NUMINAMATH_GPT_exists_a_b_not_multiple_p_l726_72622


namespace NUMINAMATH_GPT_max_c_for_range_l726_72659

theorem max_c_for_range (c : ℝ) :
  (∃ x : ℝ, (x^2 - 7*x + c = 2)) → c ≤ 57 / 4 :=
by
  sorry

end NUMINAMATH_GPT_max_c_for_range_l726_72659


namespace NUMINAMATH_GPT_line_slope_l726_72609

theorem line_slope : 
  (∀ (x y : ℝ), (x / 4 - y / 3 = -2) → (y = -3/4 * x - 6)) ∧ (∀ (x : ℝ), ∃ y : ℝ, (x / 4 - y / 3 = -2)) :=
by
  sorry

end NUMINAMATH_GPT_line_slope_l726_72609


namespace NUMINAMATH_GPT_inequality_proof_equality_condition_l726_72626

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / c) + (c / b) ≥ (4 * a) / (a + b) := 
by
  sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / c) + (c / b) = (4 * a) / (a + b) ↔ a = b ∧ b = c :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_equality_condition_l726_72626


namespace NUMINAMATH_GPT_system_inequalities_1_system_inequalities_2_l726_72657

theorem system_inequalities_1 (x: ℝ):
  (4 * (x + 1) ≤ 7 * x + 10) → (x - 5 < (x - 8)/3) → (-2 ≤ x ∧ x < 7 / 2) :=
by
  intros h1 h2
  sorry

theorem system_inequalities_2 (x: ℝ):
  (x - 3 * (x - 2) ≥ 4) → ((2 * x - 1) / 5 ≥ (x + 1) / 2) → (x ≤ -7) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_system_inequalities_1_system_inequalities_2_l726_72657


namespace NUMINAMATH_GPT_factor_expression_l726_72610

theorem factor_expression (x : ℝ) : 45 * x^3 + 135 * x^2 = 45 * x^2 * (x + 3) :=
  by
    sorry

end NUMINAMATH_GPT_factor_expression_l726_72610


namespace NUMINAMATH_GPT_eq_neg2_multi_l726_72666

theorem eq_neg2_multi {m n : ℝ} (h : m = n) : -2 * m = -2 * n :=
by sorry

end NUMINAMATH_GPT_eq_neg2_multi_l726_72666


namespace NUMINAMATH_GPT_evaluate_fraction_l726_72651

theorem evaluate_fraction : 
  (7/3) / (8/15) = 35/8 :=
by
  -- we don't need to provide the proof as per instructions
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l726_72651


namespace NUMINAMATH_GPT_negation_example_l726_72625

theorem negation_example :
  (¬ (∀ a : ℕ, a > 0 → 2^a ≥ a^2)) ↔ (∃ a : ℕ, a > 0 ∧ 2^a < a^2) :=
by sorry

end NUMINAMATH_GPT_negation_example_l726_72625


namespace NUMINAMATH_GPT_remainder_sum_mod9_l726_72662

def a1 := 8243
def a2 := 8244
def a3 := 8245
def a4 := 8246

theorem remainder_sum_mod9 : ((a1 + a2 + a3 + a4) % 9) = 7 :=
by
  sorry

end NUMINAMATH_GPT_remainder_sum_mod9_l726_72662


namespace NUMINAMATH_GPT_log2_of_fraction_l726_72639

theorem log2_of_fraction : Real.logb 2 0.03125 = -5 := by
  sorry

end NUMINAMATH_GPT_log2_of_fraction_l726_72639


namespace NUMINAMATH_GPT_sum_of_three_numbers_l726_72640

theorem sum_of_three_numbers :
  ∃ (S1 S2 S3 : ℕ), 
    S2 = 72 ∧
    S1 = 2 * S2 ∧
    S3 = S1 / 3 ∧
    S1 + S2 + S3 = 264 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l726_72640


namespace NUMINAMATH_GPT_complement_union_l726_72650

def universal_set : Set ℝ := { x : ℝ | true }
def M : Set ℝ := { x : ℝ | x ≤ 0 }
def N : Set ℝ := { x : ℝ | x > 2 }

theorem complement_union (x : ℝ) :
  x ∈ compl (M ∪ N) ↔ (0 < x ∧ x ≤ 2) := by
  sorry

end NUMINAMATH_GPT_complement_union_l726_72650


namespace NUMINAMATH_GPT_worker_idle_days_l726_72679

variable (x y : ℤ)

theorem worker_idle_days :
  (30 * x - 5 * y = 500) ∧ (x + y = 60) → y = 38 :=
by
  intros h
  have h1 : 30 * x - 5 * y = 500 := h.left
  have h2 : x + y = 60 := h.right
  sorry

end NUMINAMATH_GPT_worker_idle_days_l726_72679


namespace NUMINAMATH_GPT_min_value_of_reciprocal_squares_l726_72641

variable (a b : ℝ)

-- Define the two circle equations
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * a * x + a^2 - 4 = 0

def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 - 4 * b * y - 1 + 4 * b^2 = 0

-- The condition that the two circles are externally tangent and have three common tangents
def externallyTangent (a b : ℝ) : Prop :=
  -- From the derivation in the solution, we must have:
  (a^2 + 4 * b^2 = 9)

-- Ensure a and b are non-zero
def nonzero (a b : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0

-- State the main theorem to prove
theorem min_value_of_reciprocal_squares (h1 : externallyTangent a b) (h2 : nonzero a b) :
  (1 / a^2) + (1 / b^2) = 1 := 
sorry

end NUMINAMATH_GPT_min_value_of_reciprocal_squares_l726_72641


namespace NUMINAMATH_GPT_find_t_l726_72690

noncomputable def a_n (n : ℕ) : ℝ := 1 * (2 : ℝ)^(n-1)

noncomputable def S_3n (n : ℕ) : ℝ := (1 - (2 : ℝ)^(3 * n)) / (1 - 2)

noncomputable def a_n_cubed (n : ℕ) : ℝ := (a_n n)^3

noncomputable def T_n (n : ℕ) : ℝ := (1 - (a_n_cubed 2)^n) / (1 - (a_n_cubed 2))

theorem find_t (n : ℕ) : S_3n n = 7 * T_n n :=
by
  sorry

end NUMINAMATH_GPT_find_t_l726_72690


namespace NUMINAMATH_GPT_trader_loss_percentage_l726_72656

def profit_loss_percentage (SP1 SP2 CP1 CP2 : ℚ) : ℚ :=
  ((SP1 + SP2) - (CP1 + CP2)) / (CP1 + CP2) * 100

theorem trader_loss_percentage :
  let SP1 := 325475
  let SP2 := 325475
  let CP1 := SP1 / (1 + 0.10)
  let CP2 := SP2 / (1 - 0.10)
  profit_loss_percentage SP1 SP2 CP1 CP2 = -1 := by
  sorry

end NUMINAMATH_GPT_trader_loss_percentage_l726_72656


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l726_72619

theorem sufficient_but_not_necessary_condition :
  (∀ x : ℝ, 0 < x → x < 4 → x^2 - 4 * x < 0) ∧ ¬ (∀ x : ℝ, x^2 - 4 * x < 0 → 0 < x ∧ x < 5) :=
sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l726_72619


namespace NUMINAMATH_GPT_spokes_ratio_l726_72665

theorem spokes_ratio (B : ℕ) (front_spokes : ℕ) (total_spokes : ℕ) 
  (h1 : front_spokes = 20) 
  (h2 : total_spokes = 60) 
  (h3 : front_spokes + B = total_spokes) : 
  B / front_spokes = 2 :=
by 
  sorry

end NUMINAMATH_GPT_spokes_ratio_l726_72665


namespace NUMINAMATH_GPT_cost_per_ball_correct_l726_72646

-- Define the values given in the conditions
def total_amount_paid : ℝ := 4.62
def number_of_balls : ℝ := 3.0

-- Define the expected cost per ball according to the problem statement
def expected_cost_per_ball : ℝ := 1.54

-- Statement to prove that the cost per ball is as expected
theorem cost_per_ball_correct : (total_amount_paid / number_of_balls) = expected_cost_per_ball := 
sorry

end NUMINAMATH_GPT_cost_per_ball_correct_l726_72646


namespace NUMINAMATH_GPT_river_depth_difference_l726_72642

theorem river_depth_difference
  (mid_may_depth : ℕ)
  (mid_july_depth : ℕ)
  (mid_june_depth : ℕ)
  (H1 : mid_july_depth = 45)
  (H2 : mid_may_depth = 5)
  (H3 : 3 * mid_june_depth = mid_july_depth) :
  mid_june_depth - mid_may_depth = 10 := 
sorry

end NUMINAMATH_GPT_river_depth_difference_l726_72642


namespace NUMINAMATH_GPT_fourth_metal_mass_approx_l726_72671

noncomputable def mass_of_fourth_metal 
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 = 1.5 * x2)
  (h2 : x3 = 4 / 3 * x2)
  (h3 : x4 = 6 / 5 * x3)
  (h4 : x1 + x2 + x3 + x4 = 25) : ℝ :=
  x4

theorem fourth_metal_mass_approx 
  (x1 x2 x3 x4 : ℝ)
  (h1 : x1 = 1.5 * x2)
  (h2 : x3 = 4 / 3 * x2)
  (h3 : x4 = 6 / 5 * x3)
  (h4 : x1 + x2 + x3 + x4 = 25) : 
  abs (mass_of_fourth_metal x1 x2 x3 x4 h1 h2 h3 h4 - 7.36) < 0.01 :=
by
  sorry

end NUMINAMATH_GPT_fourth_metal_mass_approx_l726_72671


namespace NUMINAMATH_GPT_find_the_number_l726_72689

variable (x : ℕ)

theorem find_the_number (h : 43 + 3 * x = 58) : x = 5 :=
by 
  sorry

end NUMINAMATH_GPT_find_the_number_l726_72689


namespace NUMINAMATH_GPT_a_seq_formula_T_seq_sum_l726_72600

-- Definition of the sequence \( \{a_n\} \)
def a_seq (n : ℕ) (p : ℤ) : ℤ := 2 * n + 5

-- Condition: Sum of the first n terms \( s_n = n^2 + pn \)
def s_seq (n : ℕ) (p : ℤ) : ℤ := n^2 + p * n

-- Condition: \( \{a_2, a_5, a_{10}\} \) form a geometric sequence
def is_geometric (a2 a5 a10 : ℤ) : Prop :=
  a2 * a10 = a5 * a5

-- Definition of the sequence \( \{b_n\} \)
def b_seq (n : ℕ) (p : ℤ) : ℚ := 1 + 5 / (a_seq n p * a_seq (n + 1) p)

-- Function to find the sum of first n terms of \( \{b_n\} \)
def T_seq (n : ℕ) (p : ℤ) : ℚ :=
  n + 5 * (1 / (7 : ℚ) - 1 / (2 * n + 7 : ℚ)) + n / (14 * n + 49 : ℚ)

theorem a_seq_formula (p : ℤ) : ∀ n, a_seq n p = 2 * n + 5 :=
by
  sorry

theorem T_seq_sum (p : ℤ) : ∀ n, T_seq n p = (14 * n^2 + 54 * n) / (14 * n + 49) :=
by
  sorry

end NUMINAMATH_GPT_a_seq_formula_T_seq_sum_l726_72600


namespace NUMINAMATH_GPT_thirty_divides_p_squared_minus_one_iff_p_eq_five_l726_72675

theorem thirty_divides_p_squared_minus_one_iff_p_eq_five (p : ℕ) (hp : Nat.Prime p) (h_ge : p ≥ 5) : 30 ∣ (p^2 - 1) ↔ p = 5 :=
by
  sorry

end NUMINAMATH_GPT_thirty_divides_p_squared_minus_one_iff_p_eq_five_l726_72675


namespace NUMINAMATH_GPT_square_area_l726_72616

def edge1 (x : ℝ) := 5 * x - 18
def edge2 (x : ℝ) := 27 - 4 * x
def x_val : ℝ := 5

theorem square_area : edge1 x_val = edge2 x_val → (edge1 x_val) ^ 2 = 49 :=
by
  intro h
  -- Proof required here
  sorry

end NUMINAMATH_GPT_square_area_l726_72616


namespace NUMINAMATH_GPT_no_real_pairs_arithmetic_prog_l726_72663

theorem no_real_pairs_arithmetic_prog :
  ¬ ∃ a b : ℝ, (a = (1 / 2) * (8 + b)) ∧ (a + a * b = 2 * b) := by
sorry

end NUMINAMATH_GPT_no_real_pairs_arithmetic_prog_l726_72663


namespace NUMINAMATH_GPT_daniel_biked_more_l726_72678

def miles_biked_after_4_hours_more (speed_plain_daniel : ℕ) (speed_plain_elsa : ℕ) (time_plain : ℕ) 
(speed_hilly_daniel : ℕ) (speed_hilly_elsa : ℕ) (time_hilly : ℕ) : ℕ :=
(speed_plain_daniel * time_plain + speed_hilly_daniel * time_hilly) - 
(speed_plain_elsa * time_plain + speed_hilly_elsa * time_hilly)

theorem daniel_biked_more : miles_biked_after_4_hours_more 20 18 3 16 15 1 = 7 :=
by
  sorry

end NUMINAMATH_GPT_daniel_biked_more_l726_72678


namespace NUMINAMATH_GPT_percentage_of_boys_l726_72612

theorem percentage_of_boys (total_students : ℕ) (ratio_boys_to_girls : ℕ) (ratio_girls_to_boys : ℕ) 
  (h_ratio : ratio_boys_to_girls = 3 ∧ ratio_girls_to_boys = 4 ∧ total_students = 42) : 
  (18 / 42) * 100 = 42.857 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_of_boys_l726_72612


namespace NUMINAMATH_GPT_principal_equivalence_l726_72654

-- Define the conditions
def SI : ℝ := 4020.75
def R : ℝ := 9
def T : ℝ := 5

-- Define the principal calculation
noncomputable def P := SI / (R * T / 100)

-- Prove that the principal P equals 8935
theorem principal_equivalence : P = 8935 := by
  sorry

end NUMINAMATH_GPT_principal_equivalence_l726_72654


namespace NUMINAMATH_GPT_proof1_proof2_l726_72628

variable (a : ℝ) (m n : ℝ)
axiom am_eq_two : a^m = 2
axiom an_eq_three : a^n = 3

theorem proof1 : a^(4 * m + 3 * n) = 432 := by
  sorry

theorem proof2 : a^(5 * m - 2 * n) = 32 / 9 := by
  sorry

end NUMINAMATH_GPT_proof1_proof2_l726_72628


namespace NUMINAMATH_GPT_quilt_percentage_shaded_l726_72615

theorem quilt_percentage_shaded :
  ∀ (total_squares full_shaded half_shaded quarter_shaded : ℕ),
    total_squares = 25 →
    full_shaded = 4 →
    half_shaded = 8 →
    quarter_shaded = 4 →
    ((full_shaded + half_shaded * 1 / 2 + quarter_shaded * 1 / 2) / total_squares * 100 = 40) :=
by
  intros
  sorry

end NUMINAMATH_GPT_quilt_percentage_shaded_l726_72615


namespace NUMINAMATH_GPT_find_m_l726_72681

-- Definitions based on conditions
def is_eccentricity (a b c : ℝ) (e : ℝ) : Prop :=
  e = c / a

def ellipse_relation (a b m : ℝ) : Prop :=
  a ^ 2 = 3 ∧ b ^ 2 = m

def eccentricity_square_relation (c a : ℝ) : Prop :=
  (c / a) ^ 2 = 1 / 4

-- Main theorem statement
theorem find_m (m : ℝ) :
  (∀ (a b c : ℝ), ellipse_relation a b m → is_eccentricity a b c (1 / 2) → eccentricity_square_relation c a)
  → (m = 9 / 4 ∨ m = 4) := sorry

end NUMINAMATH_GPT_find_m_l726_72681


namespace NUMINAMATH_GPT_rectangular_plot_width_l726_72634

/-- Theorem: The width of a rectangular plot where the length is thrice its width and the area is 432 sq meters is 12 meters. -/
theorem rectangular_plot_width (w l : ℝ) (h₁ : l = 3 * w) (h₂ : l * w = 432) : w = 12 :=
by
  sorry

end NUMINAMATH_GPT_rectangular_plot_width_l726_72634


namespace NUMINAMATH_GPT_tim_surprises_combinations_l726_72688

theorem tim_surprises_combinations :
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 6
  let thursday_choices := 5
  let friday_choices := 2
  monday_choices * tuesday_choices * wednesday_choices * thursday_choices * friday_choices = 120 :=
by
  let monday_choices := 1
  let tuesday_choices := 2
  let wednesday_choices := 6
  let thursday_choices := 5
  let friday_choices := 2
  sorry

end NUMINAMATH_GPT_tim_surprises_combinations_l726_72688


namespace NUMINAMATH_GPT_two_times_x_equals_two_l726_72618

theorem two_times_x_equals_two (x : ℝ) (h : x = 1) : 2 * x = 2 := by
  sorry

end NUMINAMATH_GPT_two_times_x_equals_two_l726_72618


namespace NUMINAMATH_GPT_a_plus_c_eq_neg800_l726_72605

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b
noncomputable def g (x : ℝ) (c d : ℝ) : ℝ := x^2 + c * x + d

theorem a_plus_c_eq_neg800 (a b c d : ℝ) (h1 : g (-a / 2) c d = 0)
  (h2 : f (-c / 2) a b = 0) (h3 : ∀ x, f x a b ≥ f (-a / 2) a b)
  (h4 : ∀ x, g x c d ≥ g (-c / 2) c d) (h5 : f (-a / 2) a b = g (-c / 2) c d)
  (h6 : f 200 a b = -200) (h7 : g 200 c d = -200) :
  a + c = -800 := sorry

end NUMINAMATH_GPT_a_plus_c_eq_neg800_l726_72605


namespace NUMINAMATH_GPT_outer_squares_equal_three_times_inner_squares_l726_72613

theorem outer_squares_equal_three_times_inner_squares
  (a b c m_a m_b m_c : ℝ) 
  (h : m_a^2 + m_b^2 + m_c^2 = 3 / 4 * (a^2 + b^2 + c^2)) :
  a^2 + b^2 + c^2 = 3 * (m_a^2 + m_b^2 + m_c^2) := 
by 
  sorry

end NUMINAMATH_GPT_outer_squares_equal_three_times_inner_squares_l726_72613


namespace NUMINAMATH_GPT_smallest_possible_value_of_n_l726_72608

theorem smallest_possible_value_of_n (n : ℕ) (h : lcm 60 n / gcd 60 n = 45) : n = 1080 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_value_of_n_l726_72608


namespace NUMINAMATH_GPT_chef_cherries_l726_72655

theorem chef_cherries :
  ∀ (total_cherries used_cherries remaining_cherries : ℕ),
    total_cherries = 77 →
    used_cherries = 60 →
    remaining_cherries = total_cherries - used_cherries →
    remaining_cherries = 17 :=
by
  sorry

end NUMINAMATH_GPT_chef_cherries_l726_72655


namespace NUMINAMATH_GPT_intersection_points_3_l726_72660

def eq1 (x y : ℝ) : Prop := (x - y + 3) * (2 * x + 3 * y - 9) = 0
def eq2 (x y : ℝ) : Prop := (2 * x - y + 2) * (x + 3 * y - 6) = 0

theorem intersection_points_3 :
  (∃ x y : ℝ, eq1 x y ∧ eq2 x y) ∧
  (∃ x1 y1 x2 y2 x3 y3 : ℝ, 
    eq1 x1 y1 ∧ eq2 x1 y1 ∧ 
    eq1 x2 y2 ∧ eq2 x2 y2 ∧ 
    eq1 x3 y3 ∧ eq2 x3 y3 ∧
    (x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x2, y2) ≠ (x3, y3)) :=
sorry

end NUMINAMATH_GPT_intersection_points_3_l726_72660


namespace NUMINAMATH_GPT_shorter_leg_in_right_triangle_l726_72692

theorem shorter_leg_in_right_triangle (a b c : ℕ) (h : a^2 + b^2 = c^2) (hc : c = 65) : a = 16 ∨ b = 16 :=
by
  sorry

end NUMINAMATH_GPT_shorter_leg_in_right_triangle_l726_72692


namespace NUMINAMATH_GPT_minimum_value_of_linear_expression_l726_72630

theorem minimum_value_of_linear_expression :
  ∀ (x y : ℝ), |y| ≤ 2 - x ∧ x ≥ -1 → 2 * x + y ≥ -5 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_linear_expression_l726_72630


namespace NUMINAMATH_GPT_line_segment_no_intersection_l726_72648

theorem line_segment_no_intersection (a : ℝ) :
  (¬ ∃ t : ℝ, (0 ≤ t ∧ t ≤ 1 ∧ (1 - t) * (3 : ℝ) + t * (1 : ℝ) = 2 ∧ (1 - t) * (1 : ℝ) + t * (2 : ℝ) = (2 - (1 - t) * (3 : ℝ)) / a)) ->
  (a < -1 ∨ a > 0.5) :=
by
  sorry

end NUMINAMATH_GPT_line_segment_no_intersection_l726_72648


namespace NUMINAMATH_GPT_mouse_away_from_cheese_l726_72699

theorem mouse_away_from_cheese:
  ∃ a b : ℝ, a = 3 ∧ b = 3 ∧ (a + b = 6) ∧
  ∀ x y : ℝ, (y = -3 * x + 12) → 
  ∀ (a y₀ : ℝ), y₀ = (1/3) * a + 11 →
  (a, b) = (3, 3) :=
by
  sorry

end NUMINAMATH_GPT_mouse_away_from_cheese_l726_72699


namespace NUMINAMATH_GPT_spadesuit_eval_l726_72606

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_eval : spadesuit 2 (spadesuit 3 (spadesuit 1 2)) = 4 := 
by
  sorry

end NUMINAMATH_GPT_spadesuit_eval_l726_72606


namespace NUMINAMATH_GPT_cost_price_of_article_l726_72687

theorem cost_price_of_article :
  ∃ (C : ℝ), 
  (∃ (G : ℝ), C + G = 500 ∧ C + 1.15 * G = 570) ∧ 
  C = (100 / 3) :=
by sorry

end NUMINAMATH_GPT_cost_price_of_article_l726_72687


namespace NUMINAMATH_GPT_smallest_area_of_ellipse_l726_72668

theorem smallest_area_of_ellipse 
    (a b : ℝ)
    (h1 : ∀ x y, (x - 2)^2 + y^2 < 4 → (x / a)^2 + (y / b)^2 < 1)
    (h2 : ∀ x y, (x + 2)^2 + y^2 < 4 → (x / a)^2 + (y / b)^2 < 1) :
    π * a * b = π :=
sorry

end NUMINAMATH_GPT_smallest_area_of_ellipse_l726_72668


namespace NUMINAMATH_GPT_distance_relationship_l726_72674

noncomputable def plane_parallel (α β : Type) : Prop := sorry
noncomputable def line_in_plane (m : Type) (α : Type) : Prop := sorry
noncomputable def point_on_line (A : Type) (m : Type) : Prop := sorry
noncomputable def distance (A B : Type) : ℝ := sorry
noncomputable def distance_point_to_line (A : Type) (n : Type) : ℝ := sorry
noncomputable def distance_between_lines (m n : Type) : ℝ := sorry

variables (α β m n A B : Type)
variables (a b c : ℝ)

axiom plane_parallel_condition : plane_parallel α β
axiom line_m_in_alpha : line_in_plane m α
axiom line_n_in_beta : line_in_plane n β
axiom point_A_on_m : point_on_line A m
axiom point_B_on_n : point_on_line B n
axiom distance_a : a = distance A B
axiom distance_b : b = distance_point_to_line A n
axiom distance_c : c = distance_between_lines m n

theorem distance_relationship : c ≤ b ∧ b ≤ a := by
  sorry

end NUMINAMATH_GPT_distance_relationship_l726_72674


namespace NUMINAMATH_GPT_largest_multiple_of_seven_smaller_than_neg_85_l726_72617

theorem largest_multiple_of_seven_smaller_than_neg_85 
  : ∃ k : ℤ, (k * 7 < -85) ∧ (∀ m : ℤ, (m * 7 < -85) → (m * 7 ≤ k * 7)) ∧ (k = -13) 
  := sorry

end NUMINAMATH_GPT_largest_multiple_of_seven_smaller_than_neg_85_l726_72617


namespace NUMINAMATH_GPT_lemons_left_l726_72624

/--
Prove that Cristine has 9 lemons left, given that she initially bought 12 lemons and gave away 1/4 of them.
-/
theorem lemons_left {initial_lemons : ℕ} (h1 : initial_lemons = 12) (fraction_given : ℚ) (h2 : fraction_given = 1 / 4) : initial_lemons - initial_lemons * fraction_given = 9 := by
  sorry

end NUMINAMATH_GPT_lemons_left_l726_72624


namespace NUMINAMATH_GPT_trigonometric_identities_l726_72658

noncomputable def tan (θ : ℝ) : ℝ := Real.tan θ
noncomputable def sin (θ : ℝ) : ℝ := Real.sin θ
noncomputable def cos (θ : ℝ) : ℝ := Real.cos θ

theorem trigonometric_identities (θ : ℝ) (h_tan : tan θ = 2) (h_identity : sin θ ^ 2 + cos θ ^ 2 = 1) :
    ((sin θ = 2 * Real.sqrt 5 / 5 ∧ cos θ = Real.sqrt 5 / 5) ∨ (sin θ = -2 * Real.sqrt 5 / 5 ∧ cos θ = -Real.sqrt 5 / 5)) ∧
    ((4 * sin θ - 3 * cos θ) / (6 * cos θ + 2 * sin θ) = 1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_trigonometric_identities_l726_72658


namespace NUMINAMATH_GPT_regular_hexagon_area_decrease_l726_72697

noncomputable def area_decrease (original_area : ℝ) (side_decrease : ℝ) : ℝ :=
  let s := (2 * original_area) / (3 * Real.sqrt 3)
  let new_side := s - side_decrease
  let new_area := (3 * Real.sqrt 3 / 2) * new_side ^ 2
  original_area - new_area

theorem regular_hexagon_area_decrease :
  area_decrease (150 * Real.sqrt 3) 3 = 76.5 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_regular_hexagon_area_decrease_l726_72697


namespace NUMINAMATH_GPT_rented_apartment_years_l726_72621

-- Given conditions
def months_in_year := 12
def payment_first_3_years_per_month := 300
def payment_remaining_years_per_month := 350
def total_paid := 19200
def first_period_years := 3

-- Define the total payment calculation
def total_payment (additional_years: ℕ): ℕ :=
  (first_period_years * months_in_year * payment_first_3_years_per_month) + 
  (additional_years * months_in_year * payment_remaining_years_per_month)

-- Main theorem statement
theorem rented_apartment_years (additional_years: ℕ) :
  total_payment additional_years = total_paid → (first_period_years + additional_years) = 5 :=
by
  intros h
  -- This skips the proof
  sorry

end NUMINAMATH_GPT_rented_apartment_years_l726_72621


namespace NUMINAMATH_GPT_Simplify_division_l726_72627

theorem Simplify_division :
  (5 * 10^9) / (2 * 10^5 * 5) = 5000 := sorry

end NUMINAMATH_GPT_Simplify_division_l726_72627


namespace NUMINAMATH_GPT_power_mod_equiv_l726_72633

theorem power_mod_equiv : 7^150 % 12 = 1 := 
  by
  sorry

end NUMINAMATH_GPT_power_mod_equiv_l726_72633


namespace NUMINAMATH_GPT_household_count_correct_l726_72643

def num_buildings : ℕ := 4
def floors_per_building : ℕ := 6
def households_first_floor : ℕ := 2
def households_other_floors : ℕ := 3
def total_households : ℕ := 68

theorem household_count_correct :
  num_buildings * (households_first_floor + (floors_per_building - 1) * households_other_floors) = total_households :=
by
  sorry

end NUMINAMATH_GPT_household_count_correct_l726_72643


namespace NUMINAMATH_GPT_cost_price_percentage_of_marked_price_l726_72652

theorem cost_price_percentage_of_marked_price (MP CP : ℝ) (discount gain_percent : ℝ) 
  (h_discount : discount = 0.12) (h_gain_percent : gain_percent = 0.375) 
  (h_SP_def : SP = MP * (1 - discount))
  (h_SP_gain : SP = CP * (1 + gain_percent)) :
  CP / MP = 0.64 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_percentage_of_marked_price_l726_72652


namespace NUMINAMATH_GPT_statement_A_correct_statement_C_correct_l726_72645

open Nat

def combinations (n r : ℕ) : ℕ := n.choose r

theorem statement_A_correct : combinations 5 3 = combinations 5 2 := sorry

theorem statement_C_correct : combinations 6 3 - combinations 4 1 = combinations 6 3 - 4 := sorry

end NUMINAMATH_GPT_statement_A_correct_statement_C_correct_l726_72645


namespace NUMINAMATH_GPT_sin_seventeen_pi_over_four_l726_72631

theorem sin_seventeen_pi_over_four : Real.sin (17 * Real.pi / 4) = Real.sqrt 2 / 2 := sorry

end NUMINAMATH_GPT_sin_seventeen_pi_over_four_l726_72631


namespace NUMINAMATH_GPT_min_n_of_inequality_l726_72667

theorem min_n_of_inequality : 
  ∀ (n : ℕ), (1 ≤ n) → (1 / n - 1 / (n + 1) < 1 / 10) → (n = 3 ∨ ∃ (k : ℕ), k ≥ 3 ∧ n = k) :=
by
  sorry

end NUMINAMATH_GPT_min_n_of_inequality_l726_72667


namespace NUMINAMATH_GPT_max_det_bound_l726_72649

noncomputable def max_det_estimate : ℕ := 327680 * 2^16

theorem max_det_bound (M : Matrix (Fin 17) (Fin 17) ℤ)
  (h : ∀ i j, M i j = 1 ∨ M i j = -1) :
  abs (Matrix.det M) ≤ max_det_estimate :=
sorry

end NUMINAMATH_GPT_max_det_bound_l726_72649


namespace NUMINAMATH_GPT_remainder_9876543210_mod_101_l726_72698

theorem remainder_9876543210_mod_101 : 
  let a := 9876543210
  let b := 101
  let c := 31
  a % b = c :=
by
  sorry

end NUMINAMATH_GPT_remainder_9876543210_mod_101_l726_72698


namespace NUMINAMATH_GPT_work_rate_problem_l726_72691

theorem work_rate_problem 
  (W : ℝ)
  (rate_ab : ℝ)
  (rate_c : ℝ)
  (rate_abc : ℝ)
  (cond1 : rate_c = W / 2)
  (cond2 : rate_abc = W / 1)
  (cond3 : rate_ab = (W / 1) - rate_c) :
  rate_ab = W / 2 :=
by 
  -- We can add the solution steps here, but we skip that part following the guidelines
  sorry

end NUMINAMATH_GPT_work_rate_problem_l726_72691


namespace NUMINAMATH_GPT_total_dolls_count_l726_72661

-- Define the conditions
def big_box_dolls : Nat := 7
def small_box_dolls : Nat := 4
def num_big_boxes : Nat := 5
def num_small_boxes : Nat := 9

-- State the theorem that needs to be proved
theorem total_dolls_count : 
  big_box_dolls * num_big_boxes + small_box_dolls * num_small_boxes = 71 := 
by
  sorry

end NUMINAMATH_GPT_total_dolls_count_l726_72661


namespace NUMINAMATH_GPT_temperature_of_Huangshan_at_night_l726_72677

theorem temperature_of_Huangshan_at_night 
  (T_morning : ℤ) (Rise_noon : ℤ) (Drop_night : ℤ)
  (h1 : T_morning = -12) (h2 : Rise_noon = 8) (h3 : Drop_night = 10) :
  T_morning + Rise_noon - Drop_night = -14 :=
by
  sorry

end NUMINAMATH_GPT_temperature_of_Huangshan_at_night_l726_72677


namespace NUMINAMATH_GPT_gcf_7fact_8fact_l726_72611

-- Definitions based on the conditions
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

noncomputable def greatest_common_divisor (a b : ℕ) : ℕ :=
  Nat.gcd a b

-- Theorem statement
theorem gcf_7fact_8fact : greatest_common_divisor (factorial 7) (factorial 8) = 5040 := by
  sorry

end NUMINAMATH_GPT_gcf_7fact_8fact_l726_72611


namespace NUMINAMATH_GPT_equal_playing_time_l726_72664

-- Given conditions
def total_minutes : Nat := 120
def number_of_children : Nat := 6
def children_playing_at_a_time : Nat := 2

-- Proof problem statement
theorem equal_playing_time :
  (children_playing_at_a_time * total_minutes) / number_of_children = 40 :=
by
  sorry

end NUMINAMATH_GPT_equal_playing_time_l726_72664


namespace NUMINAMATH_GPT_golf_tournament_percentage_increase_l726_72623

theorem golf_tournament_percentage_increase:
  let electricity_bill := 800
  let cell_phone_expenses := electricity_bill + 400
  let golf_tournament_cost := 1440
  (golf_tournament_cost - cell_phone_expenses) / cell_phone_expenses * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_golf_tournament_percentage_increase_l726_72623


namespace NUMINAMATH_GPT_min_value_proof_l726_72686

noncomputable def min_value_expr (x y : ℝ) : ℝ :=
  4 / (x + 3 * y) + 1 / (x - y)

theorem min_value_proof (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 2) : 
  min_value_expr x y = 9 / 4 := 
sorry

end NUMINAMATH_GPT_min_value_proof_l726_72686


namespace NUMINAMATH_GPT_parallel_lines_l726_72632

theorem parallel_lines (m : ℝ) :
    (∀ x y : ℝ, x + (m+1) * y - 1 = 0 → mx + 2 * y - 1 = 0 → (m = 1 → False)) → m = -2 :=
by
  sorry

end NUMINAMATH_GPT_parallel_lines_l726_72632


namespace NUMINAMATH_GPT_total_rubber_bands_l726_72638

theorem total_rubber_bands (harper_bands : ℕ) (brother_bands: ℕ):
  harper_bands = 15 →
  brother_bands = harper_bands - 6 →
  harper_bands + brother_bands = 24 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_total_rubber_bands_l726_72638


namespace NUMINAMATH_GPT_age_ratio_holds_l726_72695

variables (e s : ℕ)

-- Conditions based on the problem statement
def condition_1 : Prop := e - 3 = 2 * (s - 3)
def condition_2 : Prop := e - 5 = 3 * (s - 5)

-- Proposition to prove that in 1 year, the age ratio will be 3:2
def age_ratio_in_one_year : Prop := (e + 1) * 2 = (s + 1) * 3

theorem age_ratio_holds (h1 : condition_1 e s) (h2 : condition_2 e s) : age_ratio_in_one_year e s :=
by {
  sorry
}

end NUMINAMATH_GPT_age_ratio_holds_l726_72695


namespace NUMINAMATH_GPT_tournament_min_cost_l726_72680

variables (k : ℕ) (m : ℕ) (S E : ℕ → ℕ)

noncomputable def min_cost (k : ℕ) : ℕ :=
  k * (4 * k^2 + k - 1) / 2

theorem tournament_min_cost (k_pos : 0 < k) (players : m = 2 * k)
  (each_plays_once 
      : ∀ i j, i ≠ j → ∃ d, S d = i ∧ E d = j) -- every two players play once, matches have days
  (one_match_per_day : ∀ d, ∃! i j, i ≠ j ∧ S d = i ∧ E d = j) -- exactly one match per day
  : min_cost k = k * (4 * k^2 + k - 1) / 2 := 
sorry

end NUMINAMATH_GPT_tournament_min_cost_l726_72680


namespace NUMINAMATH_GPT_tennis_handshakes_l726_72685

theorem tennis_handshakes :
  let num_teams := 4
  let women_per_team := 2
  let total_women := num_teams * women_per_team
  let handshakes_per_woman := total_women - 2
  let total_handshakes_before_division := total_women * handshakes_per_woman
  let actual_handshakes := total_handshakes_before_division / 2
  actual_handshakes = 24 :=
by sorry

end NUMINAMATH_GPT_tennis_handshakes_l726_72685


namespace NUMINAMATH_GPT_smaller_number_is_17_l726_72604

theorem smaller_number_is_17 (x y : ℕ) (h1 : x * y = 323) (h2 : x - y = 2) : y = 17 :=
sorry

end NUMINAMATH_GPT_smaller_number_is_17_l726_72604


namespace NUMINAMATH_GPT_area_of_CDE_in_isosceles_triangle_l726_72694

noncomputable def isosceles_triangle_area (b : ℝ) (s : ℝ) (area : ℝ) : Prop :=
  area = (1 / 2) * b * s

noncomputable def cot (α : ℝ) : ℝ := 1 / Real.tan α

noncomputable def isosceles_triangle_vertex_angle (b : ℝ) (area : ℝ) (θ : ℝ) : Prop :=
  area = (b^2 / 4) * cot (θ / 2)

theorem area_of_CDE_in_isosceles_triangle (b θ area : ℝ) (hb : b = 3 * (2 * b / 3)) (hθ : θ = 100) (ha : area = 30) :
  ∃ CDE_area, CDE_area = area / 9 ∧ CDE_area = 10 / 3 :=
by
  sorry

end NUMINAMATH_GPT_area_of_CDE_in_isosceles_triangle_l726_72694


namespace NUMINAMATH_GPT_polynomial_simplification_l726_72653

theorem polynomial_simplification (w : ℝ) : 
  3 * w + 4 - 6 * w - 5 + 7 * w + 8 - 9 * w - 10 + 2 * w ^ 2 = 2 * w ^ 2 - 5 * w - 3 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_simplification_l726_72653


namespace NUMINAMATH_GPT_quadratic_inequality_l726_72676

theorem quadratic_inequality (a : ℝ) :
  (∃ x₀ : ℝ, x₀^2 + (a - 1) * x₀ + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by sorry

end NUMINAMATH_GPT_quadratic_inequality_l726_72676


namespace NUMINAMATH_GPT_exists_close_pair_in_interval_l726_72603

theorem exists_close_pair_in_interval (x1 x2 x3 : ℝ) (h1 : 0 ≤ x1 ∧ x1 < 1) (h2 : 0 ≤ x2 ∧ x2 < 1) (h3 : 0 ≤ x3 ∧ x3 < 1) :
  ∃ a b, (a = x1 ∨ a = x2 ∨ a = x3) ∧ (b = x1 ∨ b = x2 ∨ b = x3) ∧ a ≠ b ∧ |b - a| < 1 / 2 :=
sorry

end NUMINAMATH_GPT_exists_close_pair_in_interval_l726_72603


namespace NUMINAMATH_GPT_gcf_factor_l726_72669

theorem gcf_factor (x y : ℕ) : gcd (6 * x ^ 3 * y ^ 2) (3 * x ^ 2 * y ^ 3) = 3 * x ^ 2 * y ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_gcf_factor_l726_72669


namespace NUMINAMATH_GPT_tom_read_in_five_months_l726_72684

def books_in_may : ℕ := 2
def books_in_june : ℕ := 6
def books_in_july : ℕ := 12
def books_in_august : ℕ := 20
def books_in_september : ℕ := 30

theorem tom_read_in_five_months : 
  books_in_may + books_in_june + books_in_july + books_in_august + books_in_september = 70 := by
  sorry

end NUMINAMATH_GPT_tom_read_in_five_months_l726_72684


namespace NUMINAMATH_GPT_focus_of_parabola_l726_72673

theorem focus_of_parabola (f d : ℝ) :
  (∀ x : ℝ, x^2 + (4*x^2 - f)^2 = (4*x^2 - d)^2) → 8*f + 8*d = 1 → f^2 = d^2 → f = 1/16 :=
by
  intro hEq hCoeff hSq
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l726_72673


namespace NUMINAMATH_GPT_problem1_l726_72620

theorem problem1 (x : ℝ) : abs (2 * x - 3) < 1 ↔ 1 < x ∧ x < 2 := sorry

end NUMINAMATH_GPT_problem1_l726_72620


namespace NUMINAMATH_GPT_stratified_sampling_group_l726_72672

-- Definitions of conditions
def female_students : ℕ := 24
def male_students : ℕ := 36
def selected_females : ℕ := 8
def selected_males : ℕ := 12

-- Total number of ways to select the group
def total_combinations : ℕ := Nat.choose female_students selected_females * Nat.choose male_students selected_males

-- Proof of the problem
theorem stratified_sampling_group :
  (total_combinations = Nat.choose 24 8 * Nat.choose 36 12) :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_group_l726_72672


namespace NUMINAMATH_GPT_correct_reasoning_methods_l726_72637

-- Definitions based on conditions
def reasoning_1 : String := "Inductive reasoning"
def reasoning_2 : String := "Deductive reasoning"
def reasoning_3 : String := "Analogical reasoning"

-- Proposition stating that the correct answer is D
theorem correct_reasoning_methods :
  (reasoning_1 = "Inductive reasoning") ∧
  (reasoning_2 = "Deductive reasoning") ∧
  (reasoning_3 = "Analogical reasoning") ↔
  (choice = "D") :=
by sorry

end NUMINAMATH_GPT_correct_reasoning_methods_l726_72637


namespace NUMINAMATH_GPT_seats_in_16th_row_l726_72601

def arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d

theorem seats_in_16th_row : arithmetic_sequence 5 2 16 = 35 := by
  sorry

end NUMINAMATH_GPT_seats_in_16th_row_l726_72601


namespace NUMINAMATH_GPT_hyperbola_properties_l726_72635

-- Define the conditions and the final statements we need to prove
theorem hyperbola_properties (a : ℝ) (ha : a > 2) (E : ℝ → ℝ → Prop)
  (hE : ∀ x y, E x y ↔ (x^2 / a^2 - y^2 / (a^2 - 4) = 1))
  (e : ℝ) (he : e = (Real.sqrt (a^2 + (a^2 - 4))) / a) :
  (∃ E' : ℝ → ℝ → Prop,
   ∀ x y, E' x y ↔ (x^2 / 9 - y^2 / 5 = 1)) ∧
  (∃ foci line: ℝ → ℝ → Prop,
   (∀ P : ℝ × ℝ, (E P.1 P.2) →
    (∃ Q : ℝ × ℝ, (P.1 - Q.1) * (P.1 + (Real.sqrt (2*a^2-4))) = 0 ∧ Q.2=0 ∧ 
     line (P.1) (P.2) ↔ P.1 - P.2 = 2))) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_properties_l726_72635
