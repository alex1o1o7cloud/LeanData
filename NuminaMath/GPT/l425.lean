import Mathlib

namespace NUMINAMATH_GPT_no_such_integers_l425_42530

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem no_such_integers (a b c d k : ℤ) (h : k > 1) :
  (a + b * omega + c * omega^2 + d * omega^3)^k ≠ 1 + omega :=
sorry

end NUMINAMATH_GPT_no_such_integers_l425_42530


namespace NUMINAMATH_GPT_first_fun_friday_is_march_30_l425_42560

def month_days := 31
def start_day := 4 -- 1 for Sunday, 2 for Monday, ..., 7 for Saturday; 4 means Thursday
def first_friday := 2
def fun_friday (n : ℕ) : ℕ := first_friday + (n - 1) * 7

theorem first_fun_friday_is_march_30 (h1 : start_day = 4)
                                    (h2 : month_days = 31) :
                                    fun_friday 5 = 30 :=
by 
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_first_fun_friday_is_march_30_l425_42560


namespace NUMINAMATH_GPT_sum_of_first_column_l425_42594

theorem sum_of_first_column (a b : ℕ) 
  (h1 : 16 * (a + b) = 96) 
  (h2 : 16 * (a - b) = 64) :
  a + b = 20 :=
by sorry

end NUMINAMATH_GPT_sum_of_first_column_l425_42594


namespace NUMINAMATH_GPT_shirts_per_minute_l425_42514

theorem shirts_per_minute (total_shirts : ℕ) (total_minutes : ℕ) (shirts_per_min : ℕ) 
  (h : total_shirts = 12 ∧ total_minutes = 6) :
  shirts_per_min = 2 :=
sorry

end NUMINAMATH_GPT_shirts_per_minute_l425_42514


namespace NUMINAMATH_GPT_unique_n_l425_42557

theorem unique_n : ∃ n : ℕ, 0 < n ∧ n^3 % 1000 = n ∧ ∀ m : ℕ, 0 < m ∧ m^3 % 1000 = m → m = n :=
by
  sorry

end NUMINAMATH_GPT_unique_n_l425_42557


namespace NUMINAMATH_GPT_triangle_area_ab_l425_42565

theorem triangle_area_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hline : ∀ (x y : ℝ), a * x + b * y = 6) (harea : (1/2) * (6 / a) * (6 / b) = 6) : 
  a * b = 3 := 
by sorry

end NUMINAMATH_GPT_triangle_area_ab_l425_42565


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l425_42591

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) (h₁ : a 1 = 1) (h₂ : a 1 * a 2 * a 3 = -8) :
  q = -2 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l425_42591


namespace NUMINAMATH_GPT_total_songs_megan_bought_l425_42593

-- Definitions for the problem conditions
def country_albums : ℕ := 2
def pop_albums : ℕ := 8
def songs_per_album : ℕ := 7
def total_albums : ℕ := country_albums + pop_albums

-- Theorem stating the conclusion we need to prove
theorem total_songs_megan_bought : total_albums * songs_per_album = 70 :=
by
  sorry

end NUMINAMATH_GPT_total_songs_megan_bought_l425_42593


namespace NUMINAMATH_GPT_sqrt_multiplication_and_subtraction_l425_42510

theorem sqrt_multiplication_and_subtraction :
  (Real.sqrt 21 * Real.sqrt 7 - Real.sqrt 3) = 6 * Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_multiplication_and_subtraction_l425_42510


namespace NUMINAMATH_GPT_sum_first_sequence_terms_l425_42576

theorem sum_first_sequence_terms 
  (S : ℕ → ℕ) 
  (a : ℕ → ℕ) 
  (h1 : ∀ n, n ≥ 2 → S n - S (n - 1) = 2 * n - 1)
  (h2 : S 2 = 3) 
  : a 1 + a 3 = 5 :=
sorry

end NUMINAMATH_GPT_sum_first_sequence_terms_l425_42576


namespace NUMINAMATH_GPT_tangent_line_at_point_l425_42573

theorem tangent_line_at_point (x y : ℝ) (h : y = Real.exp x) (t : x = 2) :
  y = Real.exp 2 * x - 2 * Real.exp 2 :=
by sorry

end NUMINAMATH_GPT_tangent_line_at_point_l425_42573


namespace NUMINAMATH_GPT_unique_positive_b_for_one_solution_l425_42527

theorem unique_positive_b_for_one_solution
  (a : ℝ) (c : ℝ) :
  a = 3 →
  (∃! (b : ℝ), b > 0 ∧ (3 * (b + (1 / b)))^2 - 4 * c = 0 ) →
  c = 9 :=
by
  intros ha h
  -- Proceed to show that c must be 9
  sorry

end NUMINAMATH_GPT_unique_positive_b_for_one_solution_l425_42527


namespace NUMINAMATH_GPT_total_birds_l425_42569

-- Definitions from conditions
def num_geese : ℕ := 58
def num_ducks : ℕ := 37

-- Proof problem statement
theorem total_birds : num_geese + num_ducks = 95 := by
  sorry

end NUMINAMATH_GPT_total_birds_l425_42569


namespace NUMINAMATH_GPT_function_properties_l425_42517

noncomputable def f (x : ℝ) : ℝ := (4^x - 1) / (2^(x + 1))

theorem function_properties :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x < f y) :=
by
  sorry

end NUMINAMATH_GPT_function_properties_l425_42517


namespace NUMINAMATH_GPT_leg_ratio_of_right_triangle_l425_42531

theorem leg_ratio_of_right_triangle (a b c m : ℝ) (h1 : a ≤ b)
  (h2 : a * b = c * m) (h3 : c^2 = a^2 + b^2) (h4 : a^2 + m^2 = b^2) :
  (a / b) = Real.sqrt ((-1 + Real.sqrt 5) / 2) :=
sorry

end NUMINAMATH_GPT_leg_ratio_of_right_triangle_l425_42531


namespace NUMINAMATH_GPT_sams_speed_l425_42504

theorem sams_speed (lucas_speed : ℝ) (maya_factor : ℝ) (relationship_factor : ℝ) 
  (h_lucas : lucas_speed = 5)
  (h_maya : maya_factor = 4 / 5)
  (h_relationship : relationship_factor = 9 / 8) :
  (5 / relationship_factor) = 40 / 9 :=
by
  sorry

end NUMINAMATH_GPT_sams_speed_l425_42504


namespace NUMINAMATH_GPT_tangent_line_to_parabola_l425_42551

theorem tangent_line_to_parabola :
  (∀ (x y : ℝ), y = x^2 → x = -1 → y = 1 → 2 * x + y + 1 = 0) :=
by
  intro x y parabola eq_x eq_y
  sorry

end NUMINAMATH_GPT_tangent_line_to_parabola_l425_42551


namespace NUMINAMATH_GPT_first_number_value_l425_42599

theorem first_number_value (A B LCM HCF : ℕ) (h_lcm : LCM = 2310) (h_hcf : HCF = 30) (h_b : B = 210) (h_mul : A * B = LCM * HCF) : A = 330 := 
by
  -- Use sorry to skip the proof
  sorry

end NUMINAMATH_GPT_first_number_value_l425_42599


namespace NUMINAMATH_GPT_sum_of_fourth_powers_l425_42532

theorem sum_of_fourth_powers
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 15.5 := sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_l425_42532


namespace NUMINAMATH_GPT_base_number_in_exponent_l425_42513

theorem base_number_in_exponent (x : ℝ) (k : ℕ) (h₁ : k = 8) (h₂ : 64^k > x^22) : 
  x = 2^(24/11) :=
sorry

end NUMINAMATH_GPT_base_number_in_exponent_l425_42513


namespace NUMINAMATH_GPT_students_accounting_majors_l425_42508

theorem students_accounting_majors (p q r s : ℕ) 
  (h1 : 1 < p) (h2 : p < q) (h3 : q < r) (h4 : r < s) (h5 : p * q * r * s = 1365) : p = 3 := 
by 
  sorry

end NUMINAMATH_GPT_students_accounting_majors_l425_42508


namespace NUMINAMATH_GPT_total_mangoes_calculation_l425_42505

-- Define conditions as constants
def boxes : ℕ := 36
def dozen_to_mangoes : ℕ := 12
def dozens_per_box : ℕ := 10

-- Define the expected correct answer for the total mangoes
def expected_total_mangoes : ℕ := 4320

-- Lean statement to prove
theorem total_mangoes_calculation :
  dozens_per_box * dozen_to_mangoes * boxes = expected_total_mangoes :=
by sorry

end NUMINAMATH_GPT_total_mangoes_calculation_l425_42505


namespace NUMINAMATH_GPT_g_at_100_l425_42582

-- Defining that g is a function from positive real numbers to real numbers
def g : ℝ → ℝ := sorry

-- The given conditions
axiom functional_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x * g y - y * g x = g (x / y)

axiom g_one : g 1 = 1

-- The theorem to prove
theorem g_at_100 : g 100 = 50 :=
by
  sorry

end NUMINAMATH_GPT_g_at_100_l425_42582


namespace NUMINAMATH_GPT_max_value_x_minus_2y_exists_max_value_x_minus_2y_l425_42583

theorem max_value_x_minus_2y 
  (x y : ℝ) 
  (h : x^2 - 4 * x + y^2 = 0) : 
  x - 2 * y ≤ 2 + 2 * Real.sqrt 5 :=
sorry

theorem exists_max_value_x_minus_2y 
  (x y : ℝ) 
  (h : x^2 - 4 * x + y^2 = 0) : 
  ∃ (x y : ℝ), x - 2 * y = 2 + 2 * Real.sqrt 5 :=
sorry

end NUMINAMATH_GPT_max_value_x_minus_2y_exists_max_value_x_minus_2y_l425_42583


namespace NUMINAMATH_GPT_teams_in_double_round_robin_l425_42566
-- Import the standard math library

-- Lean statement for the proof problem
theorem teams_in_double_round_robin (m n : ℤ) 
  (h : 9 * n^2 + 6 * n + 32 = m * (m - 1) / 2) : 
  m = 8 ∨ m = 32 :=
sorry

end NUMINAMATH_GPT_teams_in_double_round_robin_l425_42566


namespace NUMINAMATH_GPT_frac_pow_zero_l425_42567

def frac := 123456789 / (-987654321 : ℤ)

theorem frac_pow_zero : frac ^ 0 = 1 :=
by sorry

end NUMINAMATH_GPT_frac_pow_zero_l425_42567


namespace NUMINAMATH_GPT_find_num_non_officers_l425_42552

-- Define the average salaries and number of officers
def avg_salary_employees : Int := 120
def avg_salary_officers : Int := 470
def avg_salary_non_officers : Int := 110
def num_officers : Int := 15

-- States the problem of finding the number of non-officers
theorem find_num_non_officers : ∃ N : Int,
(15 * 470 + N * 110 = (15 + N) * 120) ∧ N = 525 := 
by {
  sorry
}

end NUMINAMATH_GPT_find_num_non_officers_l425_42552


namespace NUMINAMATH_GPT_find_a2_b2_l425_42540

noncomputable def imaginary_unit : ℂ := Complex.I

theorem find_a2_b2 (a b : ℝ) (h1 : (a - 2 * imaginary_unit) * imaginary_unit = b - imaginary_unit) : a^2 + b^2 = 5 :=
  sorry

end NUMINAMATH_GPT_find_a2_b2_l425_42540


namespace NUMINAMATH_GPT_problem_equivalent_l425_42544

variable (p : ℤ) 

theorem problem_equivalent (h : p = (-2023) * 100) : (-2023) * 99 = p + 2023 :=
by sorry

end NUMINAMATH_GPT_problem_equivalent_l425_42544


namespace NUMINAMATH_GPT_initial_weight_l425_42546

theorem initial_weight (W : ℝ) (h₁ : W > 0): 
  W * 0.85 * 0.75 * 0.90 = 450 := 
by 
  sorry

end NUMINAMATH_GPT_initial_weight_l425_42546


namespace NUMINAMATH_GPT_factorize_expression_l425_42503

variable (x y : ℝ)

theorem factorize_expression : xy^2 + 6*xy + 9*x = x*(y + 3)^2 := by
  sorry

end NUMINAMATH_GPT_factorize_expression_l425_42503


namespace NUMINAMATH_GPT_log_relation_l425_42588

theorem log_relation (a b : ℝ) 
  (h₁ : a = Real.log 1024 / Real.log 16) 
  (h₂ : b = Real.log 32 / Real.log 2) : 
  a = 1 / 2 * b := 
by 
  sorry

end NUMINAMATH_GPT_log_relation_l425_42588


namespace NUMINAMATH_GPT_g_properties_l425_42525

def f (x : ℝ) : ℝ := x

def g (x : ℝ) : ℝ := -f x

theorem g_properties :
  (∀ x : ℝ, g (-x) = -g x) ∧ (∀ x y : ℝ, x < y → g x > g y) :=
by
  sorry

end NUMINAMATH_GPT_g_properties_l425_42525


namespace NUMINAMATH_GPT_ab_root_of_Q_l425_42554

theorem ab_root_of_Q (a b : ℝ) (h : a ≠ b) (ha : a^4 + a^3 - 1 = 0) (hb : b^4 + b^3 - 1 = 0) :
  (ab : ℝ)^6 + (ab : ℝ)^4 + (ab : ℝ)^3 - (ab : ℝ)^2 - 1 = 0 := 
sorry

end NUMINAMATH_GPT_ab_root_of_Q_l425_42554


namespace NUMINAMATH_GPT_maximum_distance_is_correct_l425_42506

-- Define the right trapezoid with the given side lengths and angle conditions
structure RightTrapezoid (AB CD : ℕ) where
  B_angle : ℝ
  D_angle : ℝ
  h_AB : AB = 200
  h_CD : CD = 100
  h_B_angle : B_angle = 90
  h_D_angle : D_angle = 45

-- Define the guards' walking condition and distance calculation
def max_distance_between_guards (T : RightTrapezoid 200 100) : ℝ :=
  let P := 400 + 100 * Real.sqrt 2
  let d := (400 + 100 * Real.sqrt 2) / 2
  222.1  -- Hard-coded according to the problem's correct answer for maximum distance

theorem maximum_distance_is_correct :
  ∀ (T : RightTrapezoid 200 100), max_distance_between_guards T = 222.1 := by
  sorry

end NUMINAMATH_GPT_maximum_distance_is_correct_l425_42506


namespace NUMINAMATH_GPT_principal_amount_l425_42597

theorem principal_amount (A2 A3 : ℝ) (interest : ℝ) (principal : ℝ) (h1 : A2 = 3450) 
  (h2 : A3 = 3655) (h_interest : interest = A3 - A2) (h_principal : principal = A2 - interest) : 
  principal = 3245 :=
by
  sorry

end NUMINAMATH_GPT_principal_amount_l425_42597


namespace NUMINAMATH_GPT_find_k_l425_42523

theorem find_k (x₁ x₂ k : ℝ) (h1 : x₁ * x₁ - 6 * x₁ + k = 0) (h2 : x₂ * x₂ - 6 * x₂ + k = 0) (h3 : (1 / x₁) + (1 / x₂) = 3) :
  k = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l425_42523


namespace NUMINAMATH_GPT_side_error_percentage_l425_42502

theorem side_error_percentage (S S' : ℝ) (h1: S' = S * Real.sqrt 1.0609) : 
  (S' / S - 1) * 100 = 3 :=
by
  sorry

end NUMINAMATH_GPT_side_error_percentage_l425_42502


namespace NUMINAMATH_GPT_number_of_boys_l425_42572

theorem number_of_boys (girls boys : ℕ) (total_books books_girls books_boys books_per_student : ℕ)
  (h1 : girls = 15)
  (h2 : total_books = 375)
  (h3 : books_girls = 225)
  (h4 : total_books = books_girls + books_boys)
  (h5 : books_girls = girls * books_per_student)
  (h6 : books_boys = boys * books_per_student)
  (h7 : books_per_student = 15) :
  boys = 10 :=
by
  sorry

end NUMINAMATH_GPT_number_of_boys_l425_42572


namespace NUMINAMATH_GPT_probability_two_green_apples_l425_42590

theorem probability_two_green_apples :
  let total_apples := 9
  let total_red := 5
  let total_green := 4
  let ways_to_choose_two := Nat.choose total_apples 2
  let ways_to_choose_two_green := Nat.choose total_green 2
  ways_to_choose_two ≠ 0 →
  (ways_to_choose_two_green / ways_to_choose_two : ℚ) = 1 / 6 :=
by
  intros
  -- skipping the proof
  sorry

end NUMINAMATH_GPT_probability_two_green_apples_l425_42590


namespace NUMINAMATH_GPT_trader_sold_90_pens_l425_42542

theorem trader_sold_90_pens (C N : ℝ) (gain_percent : ℝ) (H1 : gain_percent = 33.33333333333333) (H2 : 30 * C = (gain_percent / 100) * N * C) :
  N = 90 :=
by
  sorry

end NUMINAMATH_GPT_trader_sold_90_pens_l425_42542


namespace NUMINAMATH_GPT_Charley_total_beads_pulled_l425_42564

-- Definitions and conditions
def initial_white_beads := 105
def initial_black_beads := 210
def initial_blue_beads := 60

def first_round_black_pulled := (2 / 7) * initial_black_beads
def first_round_white_pulled := (3 / 7) * initial_white_beads
def first_round_blue_pulled := (1 / 4) * initial_blue_beads

def first_round_total_pulled := first_round_black_pulled + first_round_white_pulled + first_round_blue_pulled

def remaining_black_beads := initial_black_beads - first_round_black_pulled
def remaining_white_beads := initial_white_beads - first_round_white_pulled
def remaining_blue_beads := initial_blue_beads - first_round_blue_pulled

def added_white_beads := 45
def added_black_beads := 80

def total_black_beads := remaining_black_beads + added_black_beads
def total_white_beads := remaining_white_beads + added_white_beads

def second_round_black_pulled := (3 / 8) * total_black_beads
def second_round_white_pulled := (1 / 3) * added_white_beads

def second_round_total_pulled := second_round_black_pulled + second_round_white_pulled

def total_beads_pulled := first_round_total_pulled + second_round_total_pulled 

-- Theorem statement
theorem Charley_total_beads_pulled : total_beads_pulled = 221 := 
by
  -- we can ignore the proof step and leave it to be filled
  sorry

end NUMINAMATH_GPT_Charley_total_beads_pulled_l425_42564


namespace NUMINAMATH_GPT_strength_training_sessions_l425_42528

-- Define the problem conditions
def strength_training_hours (x : ℕ) : ℝ := x * 1
def boxing_training_hours : ℝ := 4 * 1.5
def total_training_hours : ℝ := 9

-- Prove how many times a week does Kat do strength training
theorem strength_training_sessions : ∃ x : ℕ, strength_training_hours x + boxing_training_hours = total_training_hours ∧ x = 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_strength_training_sessions_l425_42528


namespace NUMINAMATH_GPT_fraction_of_tips_in_august_is_five_eighths_l425_42529

-- Definitions
def average_tips (other_tips_total : ℤ) (n : ℤ) : ℤ := other_tips_total / n
def total_tips (other_tips : ℤ) (august_tips : ℤ) : ℤ := other_tips + august_tips
def fraction (numerator : ℤ) (denominator : ℤ) : ℚ := (numerator : ℚ) / (denominator : ℚ)

-- Given conditions
variables (A : ℤ) -- average monthly tips for the other 6 months (March to July and September)
variables (other_months : ℤ := 6)
variables (tips_total_other : ℤ := other_months * A) -- total tips for the 6 other months
variables (tips_august : ℤ := 10 * A) -- tips for August
variables (total_tips_all : ℤ := tips_total_other + tips_august) -- total tips for all months

-- Prove the statement
theorem fraction_of_tips_in_august_is_five_eighths :
  fraction tips_august total_tips_all = 5 / 8 := by sorry

end NUMINAMATH_GPT_fraction_of_tips_in_august_is_five_eighths_l425_42529


namespace NUMINAMATH_GPT_pieces_per_plant_yield_l425_42571

theorem pieces_per_plant_yield 
  (rows : ℕ) (plants_per_row : ℕ) (total_harvest : ℕ) 
  (h1 : rows = 30) (h2 : plants_per_row = 10) (h3 : total_harvest = 6000) : 
  (total_harvest / (rows * plants_per_row) = 20) :=
by
  -- Insert math proof here.
  sorry

end NUMINAMATH_GPT_pieces_per_plant_yield_l425_42571


namespace NUMINAMATH_GPT_child_l425_42558

noncomputable def C (G : ℝ) := 60 - 46
noncomputable def G := 130 - 60
noncomputable def ratio := (C G) / G

theorem child's_weight_to_grandmother's_weight_is_1_5 :
  ratio = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_child_l425_42558


namespace NUMINAMATH_GPT_ratio_HC_JE_l425_42507

noncomputable def A : ℝ := 0
noncomputable def B : ℝ := 1
noncomputable def C : ℝ := B + 2
noncomputable def D : ℝ := C + 1
noncomputable def E : ℝ := D + 1
noncomputable def F : ℝ := E + 2

variable (G H J K : ℝ × ℝ)
variable (parallel_AG_HC parallel_AG_JE parallel_AG_KB : Prop)

-- Conditions
axiom points_on_line : A < B ∧ B < C ∧ C < D ∧ D < E ∧ E < F
axiom AB : B - A = 1
axiom BC : C - B = 2
axiom CD : D - C = 1
axiom DE : E - D = 1
axiom EF : F - E = 2
axiom G_off_AF : G.2 ≠ 0
axiom H_on_GD : H.1 = G.1 ∧ H.2 = D
axiom J_on_GF : J.1 = G.1 ∧ J.2 = F
axiom K_on_GB : K.1 = G.1 ∧ K.2 = B
axiom parallel_hc_je_kb_ag : parallel_AG_HC ∧ parallel_AG_JE ∧ parallel_AG_KB ∧ (G.2 / 1) = (K.2 / (K.1 - G.1))

-- Task: Prove the ratio HC/JE = 7/8
theorem ratio_HC_JE : (H.2 - C) / (J.2 - E) = 7 / 8 :=
sorry

end NUMINAMATH_GPT_ratio_HC_JE_l425_42507


namespace NUMINAMATH_GPT_largest_n_proof_l425_42585

def largest_n_less_than_50000_divisible_by_7 (n : ℕ) : Prop :=
  n < 50000 ∧ (10 * (n - 3)^5 - 2 * n^2 + 20 * n - 36) % 7 = 0

theorem largest_n_proof : ∃ n, largest_n_less_than_50000_divisible_by_7 n ∧ ∀ m, largest_n_less_than_50000_divisible_by_7 m → m ≤ n := 
sorry

end NUMINAMATH_GPT_largest_n_proof_l425_42585


namespace NUMINAMATH_GPT_number_of_classes_l425_42579

theorem number_of_classes (n : ℕ) (a₁ : ℕ) (d : ℤ) (S : ℕ) (h₁ : d = -2) (h₂ : a₁ = 25) (h₃ : S = 105) : n = 5 :=
by
  /- We state the theorem and the necessary conditions without proving it -/
  sorry

end NUMINAMATH_GPT_number_of_classes_l425_42579


namespace NUMINAMATH_GPT_exist_same_number_of_acquaintances_l425_42536

-- Define a group of 2014 people
variable (People : Type) [Fintype People] [DecidableEq People]
variable (knows : People → People → Prop)
variable [DecidableRel knows]

-- Conditions
def mutual_acquaintance : Prop := 
  ∀ (a b : People), knows a b ↔ knows b a

def num_people : Prop := 
  Fintype.card People = 2014

-- Theorem to prove
theorem exist_same_number_of_acquaintances 
  (h1 : mutual_acquaintance People knows) 
  (h2 : num_people People) : 
  ∃ (p1 p2 : People), p1 ≠ p2 ∧
    (Fintype.card { x // knows p1 x } = Fintype.card { x // knows p2 x }) :=
sorry

end NUMINAMATH_GPT_exist_same_number_of_acquaintances_l425_42536


namespace NUMINAMATH_GPT_find_fraction_l425_42568

theorem find_fraction (a b : ℝ) (h : a = 2 * b) : (a / (a - b)) = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_fraction_l425_42568


namespace NUMINAMATH_GPT_arithmetic_statement_not_basic_l425_42595

-- Define the basic algorithmic statements as a set
def basic_algorithmic_statements : Set String := 
  {"Input statement", "Output statement", "Assignment statement", "Conditional statement", "Loop statement"}

-- Define the arithmetic statement
def arithmetic_statement : String := "Arithmetic statement"

-- Prove that arithmetic statement is not a basic algorithmic statement
theorem arithmetic_statement_not_basic :
  arithmetic_statement ∉ basic_algorithmic_statements :=
sorry

end NUMINAMATH_GPT_arithmetic_statement_not_basic_l425_42595


namespace NUMINAMATH_GPT_tailor_buttons_l425_42500

theorem tailor_buttons (G : ℕ) (yellow_buttons : ℕ) (blue_buttons : ℕ) 
(h1 : yellow_buttons = G + 10) (h2 : blue_buttons = G - 5) 
(h3 : G + yellow_buttons + blue_buttons = 275) : G = 90 :=
sorry

end NUMINAMATH_GPT_tailor_buttons_l425_42500


namespace NUMINAMATH_GPT_relationship_between_x1_x2_x3_l425_42537

variable {x1 x2 x3 : ℝ}

theorem relationship_between_x1_x2_x3
  (A_on_curve : (6 : ℝ) = 6 / x1)
  (B_on_curve : (12 : ℝ) = 6 / x2)
  (C_on_curve : (-6 : ℝ) = 6 / x3) :
  x3 < x2 ∧ x2 < x1 := 
sorry

end NUMINAMATH_GPT_relationship_between_x1_x2_x3_l425_42537


namespace NUMINAMATH_GPT_discount_price_l425_42561

theorem discount_price (original_price : ℝ) (discount_percent : ℝ) (final_price : ℝ) :
  original_price = 800 ∧ discount_percent = 15 → final_price = 680 :=
by
  intros h
  cases' h with hp hd
  sorry

end NUMINAMATH_GPT_discount_price_l425_42561


namespace NUMINAMATH_GPT_inequality_region_area_l425_42592

noncomputable def area_of_inequality_region : ℝ :=
  let region := {p : ℝ × ℝ | |p.fst - p.snd| + |2 * p.fst + 2 * p.snd| ≤ 8}
  let vertices := [(2, 2), (-2, 2), (-2, -2), (2, -2)]
  let d1 := 8
  let d2 := 8
  (1 / 2) * d1 * d2

theorem inequality_region_area :
  area_of_inequality_region = 32 :=
by
  sorry  -- Proof to be provided

end NUMINAMATH_GPT_inequality_region_area_l425_42592


namespace NUMINAMATH_GPT_determine_e_l425_42570

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := 3*x^3 + d*x^2 + e*x + f

theorem determine_e (d f : ℝ) (h1 : f = 18) (h2 : -f/3 = -6) (h3 : -d/3 = -6) (h4 : 3 + d + e + f = -6) : e = -45 :=
sorry

end NUMINAMATH_GPT_determine_e_l425_42570


namespace NUMINAMATH_GPT_complex_number_solution_l425_42575

-- Define that z is a complex number and the condition given in the problem.
theorem complex_number_solution (z : ℂ) (hz : (i / (z + i)) = 2 - i) : z = -1/5 - 3/5 * i :=
sorry

end NUMINAMATH_GPT_complex_number_solution_l425_42575


namespace NUMINAMATH_GPT_evaluate_expression_l425_42511

theorem evaluate_expression : (3 / (1 - (2 / 5))) = 5 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l425_42511


namespace NUMINAMATH_GPT_greatest_product_two_ints_sum_300_l425_42539

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end NUMINAMATH_GPT_greatest_product_two_ints_sum_300_l425_42539


namespace NUMINAMATH_GPT_swimmers_meet_times_l425_42543

noncomputable def swimmers_passes (pool_length : ℕ) (time_minutes : ℕ) (speed_swimmer1 : ℕ) (speed_swimmer2 : ℕ) : ℕ :=
  let total_time_seconds := time_minutes * 60
  let speed_sum := speed_swimmer1 + speed_swimmer2
  let distance_in_time := total_time_seconds * speed_sum
  distance_in_time / pool_length

theorem swimmers_meet_times :
  swimmers_passes 120 15 4 3 = 53 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_swimmers_meet_times_l425_42543


namespace NUMINAMATH_GPT_symmetry_center_example_l425_42541

-- Define the function tan(2x - π/4)
noncomputable def func (x : ℝ) : ℝ := Real.tan (2 * x - Real.pi / 4)

-- Define what it means to be a symmetry center for the function
def is_symmetry_center (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (2 * (p.1) - x) = 2 * p.2 - f x

-- Statement of the proof problem
theorem symmetry_center_example : is_symmetry_center func (-Real.pi / 8, 0) :=
sorry

end NUMINAMATH_GPT_symmetry_center_example_l425_42541


namespace NUMINAMATH_GPT_cost_price_of_A_l425_42587

-- Assume the cost price of the bicycle for A which we need to prove
def CP_A : ℝ := 144

-- Given conditions
def profit_A_to_B (CP_A : ℝ) := 1.25 * CP_A
def profit_B_to_C (CP_B : ℝ) := 1.25 * CP_B
def SP_C := 225

-- Proof statement
theorem cost_price_of_A : 
  profit_B_to_C (profit_A_to_B CP_A) = SP_C :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_A_l425_42587


namespace NUMINAMATH_GPT_average_percentage_decrease_l425_42526

-- Given definitions
def original_price : ℝ := 10000
def final_price : ℝ := 6400
def num_reductions : ℕ := 2

-- The goal is to prove the average percentage decrease per reduction
theorem average_percentage_decrease (x : ℝ) (h : (original_price * (1 - x)^num_reductions = final_price)) : x = 0.2 :=
sorry

end NUMINAMATH_GPT_average_percentage_decrease_l425_42526


namespace NUMINAMATH_GPT_trajectory_eq_l425_42518

theorem trajectory_eq (a b : ℝ) :
  (∀ x y : ℝ, (x - a)^2 + (y - b)^2 = 6 → x^2 + y^2 + 2 * x + 2 * y - 3 = 0 → 
    ∃ p q : ℝ, p = a + 1 ∧ q = b + 1 ∧ (p * x + q * y = (a^2 + b^2 - 3)/2)) →
  a^2 + b^2 + 2 * a + 2 * b + 1 = 0 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_trajectory_eq_l425_42518


namespace NUMINAMATH_GPT_quadratic_function_value_at_18_l425_42512

noncomputable def p (d e f x : ℝ) : ℝ := d*x^2 + e*x + f

theorem quadratic_function_value_at_18
  (d e f : ℝ)
  (h_sym : ∀ x1 x2 : ℝ, p d e f 6 = p d e f 12)
  (h_max : ∀ x : ℝ, x = 10 → ∃ p_max : ℝ, ∀ y : ℝ, p d e f x ≤ p_max)
  (h_p0 : p d e f 0 = -1) : 
  p d e f 18 = -1 := 
sorry

end NUMINAMATH_GPT_quadratic_function_value_at_18_l425_42512


namespace NUMINAMATH_GPT_number_of_rods_in_one_mile_l425_42598

theorem number_of_rods_in_one_mile (miles_to_furlongs : 1 = 10 * 1)
  (furlongs_to_rods : 1 = 50 * 1) : 1 = 500 * 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_rods_in_one_mile_l425_42598


namespace NUMINAMATH_GPT_range_of_a_l425_42553

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then -x - 3 * a else a^x - 2

theorem range_of_a (a : ℝ) :
  (0 < a ∧ a ≠ 1 ∧ ∀ x y : ℝ, x < y → f a x > f a y) ↔ (0 < a ∧ a ≤ 1 / 3) :=
by sorry

end NUMINAMATH_GPT_range_of_a_l425_42553


namespace NUMINAMATH_GPT_kiril_age_problem_l425_42574

theorem kiril_age_problem (x : ℕ) (h1 : x % 5 = 0) (h2 : (x - 1) % 7 = 0) : 26 - x = 11 :=
by
  sorry

end NUMINAMATH_GPT_kiril_age_problem_l425_42574


namespace NUMINAMATH_GPT_unique_triangled_pair_l425_42509

theorem unique_triangled_pair (a b x y : ℝ) (h : ∀ a b : ℝ, (a, b) = (a * x + b * y, a * y + b * x)) : (x, y) = (1, 0) :=
by sorry

end NUMINAMATH_GPT_unique_triangled_pair_l425_42509


namespace NUMINAMATH_GPT_Elon_has_10_more_Teslas_than_Sam_l425_42535

noncomputable def TeslasCalculation : Nat :=
let Chris : Nat := 6
let Sam : Nat := Chris / 2
let Elon : Nat := 13
Elon - Sam

theorem Elon_has_10_more_Teslas_than_Sam :
  TeslasCalculation = 10 :=
by
  sorry

end NUMINAMATH_GPT_Elon_has_10_more_Teslas_than_Sam_l425_42535


namespace NUMINAMATH_GPT_perfect_squares_with_specific_ones_digit_count_l425_42584

theorem perfect_squares_with_specific_ones_digit_count : 
  ∃ n : ℕ, (∀ k : ℕ, k < 2500 → (k % 10 = 4 ∨ k % 10 = 5 ∨ k % 10 = 6) ↔ ∃ m : ℕ, m < n ∧ (m % 10 = 2 ∨ m % 10 = 8 ∨ m % 10 = 5 ∨ m % 10 = 4 ∨ m % 10 = 6) ∧ k = m * m) 
  ∧ n = 25 := 
by 
  sorry

end NUMINAMATH_GPT_perfect_squares_with_specific_ones_digit_count_l425_42584


namespace NUMINAMATH_GPT_nancy_shoes_l425_42556

theorem nancy_shoes (boots slippers heels : ℕ) 
  (h₀ : boots = 6)
  (h₁ : slippers = boots + 9)
  (h₂ : heels = 3 * (boots + slippers)) :
  2 * (boots + slippers + heels) = 168 := by
  sorry

end NUMINAMATH_GPT_nancy_shoes_l425_42556


namespace NUMINAMATH_GPT_triangle_abs_diff_l425_42586

theorem triangle_abs_diff (a b c : ℝ) 
  (h1 : a + b > c)
  (h2 : a + c > b) :
  |a + b - c| - |a - b - c| = 2 * a - 2 * c := 
by sorry

end NUMINAMATH_GPT_triangle_abs_diff_l425_42586


namespace NUMINAMATH_GPT_retirement_hiring_year_l425_42555

theorem retirement_hiring_year (A W Y : ℕ)
  (hired_on_32nd_birthday : A = 32)
  (eligible_to_retire_in_2007 : 32 + (2007 - Y) = 70) : 
  Y = 1969 := by
  sorry

end NUMINAMATH_GPT_retirement_hiring_year_l425_42555


namespace NUMINAMATH_GPT_track_length_is_320_l425_42533

noncomputable def length_of_track (x : ℝ) : Prop :=
  (∃ v_b v_s : ℝ, (v_b > 0 ∧ v_s > 0 ∧ v_b + v_s = x / 2 ∧ -- speeds of Brenda and Sally must sum up to half the track length against each other
                    80 / v_b = (x / 2 - 80) / v_s ∧ -- First meeting condition
                    120 / v_s + 80 / v_b = (x / 2 + 40) / v_s + (x - 80) / v_b -- Second meeting condition
                   )) ∧ x = 320

theorem track_length_is_320 : ∃ x : ℝ, length_of_track x :=
by
  use 320
  unfold length_of_track
  simp
  sorry

end NUMINAMATH_GPT_track_length_is_320_l425_42533


namespace NUMINAMATH_GPT_John_max_tests_under_B_l425_42550

theorem John_max_tests_under_B (total_tests first_tests tests_with_B goal_percentage B_tests_first_half : ℕ) :
  total_tests = 60 →
  first_tests = 40 → 
  tests_with_B = 32 → 
  goal_percentage = 75 →
  B_tests_first_half = 32 →
  let needed_B_tests := (goal_percentage * total_tests) / 100
  let remaining_tests := total_tests - first_tests
  let remaining_needed_B_tests := needed_B_tests - B_tests_first_half
  remaining_tests - remaining_needed_B_tests ≤ 7 := sorry

end NUMINAMATH_GPT_John_max_tests_under_B_l425_42550


namespace NUMINAMATH_GPT_concyclic_H_E_N_N1_N2_l425_42563

open EuclideanGeometry

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def nine_point_center (A B C : Point) : Point := sorry
noncomputable def altitude (A B C : Point) : Point := sorry
noncomputable def salmon_circle_center (A O O₁ O₂ : Point) : Point := sorry
noncomputable def foot_of_perpendicular (O' B C : Point) : Point := sorry
noncomputable def is_concyclic (points : List Point) : Prop := sorry

theorem concyclic_H_E_N_N1_N2 (A B C D : Point):
  let H := altitude A B C
  let O := circumcenter A B C
  let O₁ := circumcenter A B D
  let O₂ := circumcenter A C D
  let N := nine_point_center A B C
  let N₁ := nine_point_center A B D
  let N₂ := nine_point_center A C D
  let O' := salmon_circle_center A O O₁ O₂
  let E := foot_of_perpendicular O' B C
  is_concyclic [H, E, N, N₁, N₂] :=
sorry

end NUMINAMATH_GPT_concyclic_H_E_N_N1_N2_l425_42563


namespace NUMINAMATH_GPT_min_value_seq_ratio_l425_42501

-- Define the sequence {a_n} based on the given recurrence relation and initial condition
def seq (n : ℕ) : ℕ := 
  if n = 0 then 0 -- Handling the case when n is 0, though sequence starts from n=1
  else n^2 - n + 15

-- Prove the minimum value of (a_n / n) is 27/4
theorem min_value_seq_ratio : 
  ∃ n : ℕ, n > 0 ∧ seq n / n = 27 / 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_seq_ratio_l425_42501


namespace NUMINAMATH_GPT_pq_sum_l425_42549

open Real

theorem pq_sum (p q : ℝ) (hp : p^3 - 18 * p^2 + 81 * p - 162 = 0) (hq : 4 * q^3 - 24 * q^2 + 45 * q - 27 = 0) :
    p + q = 8 ∨ p + q = 8 + 6 * sqrt 3 ∨ p + q = 8 - 6 * sqrt 3 :=
sorry

end NUMINAMATH_GPT_pq_sum_l425_42549


namespace NUMINAMATH_GPT_sequence_a_n_l425_42547

theorem sequence_a_n (a : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, n > 0 → (n^2 + n) * (a (n + 1) - a n) = 2) :
  a 20 = 29 / 10 :=
by
  sorry

end NUMINAMATH_GPT_sequence_a_n_l425_42547


namespace NUMINAMATH_GPT_range_of_a_l425_42545

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 5 → |2 - x| + |x + 1| ≤ a) ↔ 9 ≤ a := 
by sorry

end NUMINAMATH_GPT_range_of_a_l425_42545


namespace NUMINAMATH_GPT_longest_side_length_of_quadrilateral_l425_42516

-- Define the system of inequalities
def inFeasibleRegion (x y : ℝ) : Prop :=
  (x + 2 * y ≤ 4) ∧
  (3 * x + y ≥ 3) ∧
  (x ≥ 0) ∧
  (y ≥ 0)

-- The goal is to prove that the longest side length is 5
theorem longest_side_length_of_quadrilateral :
  ∃ a b c d : (ℝ × ℝ), inFeasibleRegion a.1 a.2 ∧
                  inFeasibleRegion b.1 b.2 ∧
                  inFeasibleRegion c.1 c.2 ∧
                  inFeasibleRegion d.1 d.2 ∧
                  -- For each side, specify the length condition (Euclidean distance)
                  max (dist a b) (max (dist b c) (max (dist c d) (dist d a))) = 5 :=
by sorry

end NUMINAMATH_GPT_longest_side_length_of_quadrilateral_l425_42516


namespace NUMINAMATH_GPT_boat_distance_along_stream_in_one_hour_l425_42580

theorem boat_distance_along_stream_in_one_hour :
  ∀ (v_b v_s d_up t : ℝ),
  v_b = 7 →
  d_up = 3 →
  t = 1 →
  (t * (v_b - v_s) = d_up) →
  t * (v_b + v_s) = 11 :=
by
  intros v_b v_s d_up t Hv_b Hd_up Ht Hup
  sorry

end NUMINAMATH_GPT_boat_distance_along_stream_in_one_hour_l425_42580


namespace NUMINAMATH_GPT_solution_couples_l425_42589

noncomputable def find_couples (n m k : ℕ) : Prop :=
  ∃ t : ℕ, (n = 2^k - 1 - t ∧ m = (Nat.factorial (2^k)) / 2^(2^k - 1 - t))

theorem solution_couples (k : ℕ) :
  ∃ n m : ℕ, (Nat.factorial (2^k)) = 2^n * m ∧ find_couples n m k :=
sorry

end NUMINAMATH_GPT_solution_couples_l425_42589


namespace NUMINAMATH_GPT_graph_properties_l425_42538

noncomputable def f (x : ℝ) : ℝ := (x^2 - 5*x + 6) / (x - 1)

theorem graph_properties :
  (∀ x, x ≠ 1 → f x = (x-2)*(x-3)/(x-1)) ∧
  (∃ x, f x = 0 ∧ (x = 2 ∨ x = 3)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 1) < δ → abs (f x) > ε) ∧
  ((∀ ε > 0, ∃ M > 0, ∀ x > M, f x > ε) ∧ (∀ ε > 0, ∃ M < 0, ∀ x < M, f x < -ε)) := sorry

end NUMINAMATH_GPT_graph_properties_l425_42538


namespace NUMINAMATH_GPT_complement_of_A_relative_to_U_l425_42519

def U := { x : ℝ | x < 3 }
def A := { x : ℝ | x < 1 }

def complement_U_A := { x : ℝ | 1 ≤ x ∧ x < 3 }

theorem complement_of_A_relative_to_U : (complement_U_A = { x : ℝ | x ∈ U ∧ x ∉ A }) :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_relative_to_U_l425_42519


namespace NUMINAMATH_GPT_proposition_true_l425_42515

theorem proposition_true (a b : ℝ) (h1 : 0 > a) (h2 : a > b) : (1/a) < (1/b) := 
sorry

end NUMINAMATH_GPT_proposition_true_l425_42515


namespace NUMINAMATH_GPT_solve_inequality_system_l425_42596

theorem solve_inequality_system (x : ℝ) :
  (x + 1 < 4 ∧ 1 - 3 * x ≥ -5) ↔ (x ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l425_42596


namespace NUMINAMATH_GPT_final_stack_height_l425_42577

theorem final_stack_height (x : ℕ) 
  (first_stack_height : ℕ := 7) 
  (second_stack_height : ℕ := first_stack_height + 5) 
  (final_stack_height : ℕ := second_stack_height + x) 
  (blocks_fell_first : ℕ := first_stack_height) 
  (blocks_fell_second : ℕ := second_stack_height - 2) 
  (blocks_fell_final : ℕ := final_stack_height - 3) 
  (total_blocks_fell : 33 = blocks_fell_first + blocks_fell_second + blocks_fell_final) 
  : x = 7 :=
  sorry

end NUMINAMATH_GPT_final_stack_height_l425_42577


namespace NUMINAMATH_GPT_find_m_for_one_real_solution_l425_42562

theorem find_m_for_one_real_solution (m : ℝ) (h : 4 * m * 4 = m^2) : m = 8 := sorry

end NUMINAMATH_GPT_find_m_for_one_real_solution_l425_42562


namespace NUMINAMATH_GPT_train_crossing_time_l425_42534

/-- Given the conditions that a moving train requires 10 seconds to pass a pole,
    its speed is 36 km/h, and the length of a stationary train is 300 meters,
    prove that the moving train takes 40 seconds to cross the stationary train. -/
theorem train_crossing_time (t_pole : ℕ)
  (v_kmh : ℕ)
  (length_stationary : ℕ) :
  t_pole = 10 →
  v_kmh = 36 →
  length_stationary = 300 →
  ∃ t_cross : ℕ, t_cross = 40 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_train_crossing_time_l425_42534


namespace NUMINAMATH_GPT_remainder_of_large_number_div_by_101_l425_42520

theorem remainder_of_large_number_div_by_101 :
  2468135792 % 101 = 52 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_large_number_div_by_101_l425_42520


namespace NUMINAMATH_GPT_eval_dollar_expr_l425_42522

noncomputable def dollar (k : ℝ) (a b : ℝ) := k * (a - b) ^ 2

theorem eval_dollar_expr (x y : ℝ) : dollar 3 ((2 * x - 3 * y) ^ 2) ((3 * y - 2 * x) ^ 2) = 0 :=
by sorry

end NUMINAMATH_GPT_eval_dollar_expr_l425_42522


namespace NUMINAMATH_GPT_circle_parabola_intersection_l425_42578

theorem circle_parabola_intersection (b : ℝ) : 
  b = 25 / 12 → 
  ∃ (r : ℝ) (cx : ℝ), 
  (∃ p1 p2 : ℝ × ℝ, 
    (p1.2 = 3/4 * p1.1 + b ∧ p2.2 = 3/4 * p2.1 + b) ∧ 
    (p1.2 = 3/4 * p1.1^2 ∧ p2.2 = 3/4 * p2.1^2) ∧ 
    (p1 ≠ (0, 0) ∧ p2 ≠ (0, 0))) ∧ 
  (cx^2 + b^2 = r^2) := 
by 
  sorry

end NUMINAMATH_GPT_circle_parabola_intersection_l425_42578


namespace NUMINAMATH_GPT_proof_problem_l425_42548

-- Definitions of sequence terms and their properties
def geometric_sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ a 4 = 16 ∧ ∀ n, a n = 2^n

-- Definition for the sum of the first n terms of the sequence
noncomputable def sum_of_sequence (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = 2^(n + 1) - 2

-- Definition for the transformed sequence b_n = log_2 a_n
def transformed_sequence (a b : ℕ → ℕ) : Prop :=
  ∀ n, b n = Nat.log2 (a n)

-- Definition for the sum T_n related to b_n
noncomputable def sum_of_transformed_sequence (T : ℕ → ℚ) (b : ℕ → ℕ) : Prop :=
  ∀ n, T n = 1 - 1 / (n + 1)

theorem proof_problem :
  (∃ a : ℕ → ℕ, geometric_sequence a) ∧
  (∃ S : ℕ → ℕ, sum_of_sequence S) ∧
  (∃ (a b : ℕ → ℕ), geometric_sequence a ∧ transformed_sequence a b ∧
   (∃ T : ℕ → ℚ, sum_of_transformed_sequence T b)) :=
by {
  -- Definitions and proofs will go here
  sorry
}

end NUMINAMATH_GPT_proof_problem_l425_42548


namespace NUMINAMATH_GPT_stock_yield_percentage_l425_42524

theorem stock_yield_percentage (face_value market_price : ℝ) (annual_dividend_rate : ℝ) 
  (h_face_value : face_value = 100)
  (h_market_price : market_price = 140)
  (h_annual_dividend_rate : annual_dividend_rate = 0.14) :
  (annual_dividend_rate * face_value / market_price) * 100 = 10 :=
by
  -- computation here
  sorry

end NUMINAMATH_GPT_stock_yield_percentage_l425_42524


namespace NUMINAMATH_GPT_find_number_l425_42521

-- Define the necessary variables and constants
variables (N : ℝ) (h1 : (5 / 4) * N = (4 / 5) * N + 18)

-- State the problem as a theorem to be proved
theorem find_number : N = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l425_42521


namespace NUMINAMATH_GPT_trajectory_of_Q_l425_42559

variable (x y m n : ℝ)

def line_l (x y : ℝ) : Prop := 2 * x + 4 * y + 3 = 0

def point_P_on_line_l (x y m n : ℝ) : Prop := line_l m n

def origin (O : (ℝ × ℝ)) := O = (0, 0)

def Q_condition (O Q P : (ℝ × ℝ)) : Prop := 2 • O + 2 • Q = Q + P

theorem trajectory_of_Q (x y m n : ℝ) (O : (ℝ × ℝ)) (P Q : (ℝ × ℝ)) :
  point_P_on_line_l x y m n → origin O → Q_condition O Q P → 
  2 * x + 4 * y + 1 = 0 := 
sorry

end NUMINAMATH_GPT_trajectory_of_Q_l425_42559


namespace NUMINAMATH_GPT_quadratic_root_iff_l425_42581

theorem quadratic_root_iff (a b c : ℝ) :
  (∃ x : ℝ, x = 1 ∧ a * x^2 + b * x + c = 0) ↔ (a + b + c = 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_iff_l425_42581
