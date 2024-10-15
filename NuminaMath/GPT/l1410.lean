import Mathlib

namespace NUMINAMATH_GPT_max_quarters_in_wallet_l1410_141012

theorem max_quarters_in_wallet:
  ∃ (q n : ℕ), 
    (30 * n) + 50 = 31 * (n + 1) ∧ 
    q = 22 :=
by
  sorry

end NUMINAMATH_GPT_max_quarters_in_wallet_l1410_141012


namespace NUMINAMATH_GPT_travel_agency_choice_l1410_141034

noncomputable def cost_A (x : ℕ) : ℝ :=
  350 * x + 1000

noncomputable def cost_B (x : ℕ) : ℝ :=
  400 * x + 800

theorem travel_agency_choice (x : ℕ) :
  if x < 4 then cost_A x > cost_B x
  else if x = 4 then cost_A x = cost_B x
  else cost_A x < cost_B x :=
by sorry

end NUMINAMATH_GPT_travel_agency_choice_l1410_141034


namespace NUMINAMATH_GPT_ratio_fourth_to_third_l1410_141086

theorem ratio_fourth_to_third (third_graders fifth_graders fourth_graders : ℕ) (H1 : third_graders = 20) (H2 : fifth_graders = third_graders / 2) (H3 : third_graders + fifth_graders + fourth_graders = 70) : fourth_graders / third_graders = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_fourth_to_third_l1410_141086


namespace NUMINAMATH_GPT_gcd_84_126_l1410_141035

-- Conditions
def a : ℕ := 84
def b : ℕ := 126

-- Theorem to prove gcd(a, b) = 42
theorem gcd_84_126 : Nat.gcd a b = 42 := by
  sorry

end NUMINAMATH_GPT_gcd_84_126_l1410_141035


namespace NUMINAMATH_GPT_incorrect_statement_d_l1410_141088

noncomputable def x := Complex.mk (-1/2) (Real.sqrt 3 / 2)
noncomputable def y := Complex.mk (-1/2) (-Real.sqrt 3 / 2)

theorem incorrect_statement_d : (x^12 + y^12) ≠ 1 := by
  sorry

end NUMINAMATH_GPT_incorrect_statement_d_l1410_141088


namespace NUMINAMATH_GPT_correct_incorrect_difference_l1410_141004

variable (x : ℝ)

theorem correct_incorrect_difference : (x - 2152) - (x - 1264) = 888 := by
  sorry

end NUMINAMATH_GPT_correct_incorrect_difference_l1410_141004


namespace NUMINAMATH_GPT_mn_value_l1410_141026

theorem mn_value (m n : ℤ) (h1 : m = n + 2) (h2 : 2 * m + n = 4) : m * n = 0 := by
  sorry

end NUMINAMATH_GPT_mn_value_l1410_141026


namespace NUMINAMATH_GPT_spend_on_video_games_l1410_141033

/-- Given the total allowance and the fractions of spending on various categories,
prove the amount spent on video games. -/
theorem spend_on_video_games (total_allowance : ℝ)
  (fraction_books fraction_snacks fraction_crafts : ℝ)
  (h_total : total_allowance = 50)
  (h_fraction_books : fraction_books = 1 / 4)
  (h_fraction_snacks : fraction_snacks = 1 / 5)
  (h_fraction_crafts : fraction_crafts = 3 / 10) :
  total_allowance - (fraction_books * total_allowance + fraction_snacks * total_allowance + fraction_crafts * total_allowance) = 12.5 :=
by
  sorry

end NUMINAMATH_GPT_spend_on_video_games_l1410_141033


namespace NUMINAMATH_GPT_fraction_unchanged_when_increased_by_ten_l1410_141056

variable {x y : ℝ}

theorem fraction_unchanged_when_increased_by_ten (x y : ℝ) :
  (5 * (10 * x)) / (10 * x + 10 * y) = 5 * x / (x + y) :=
by
  sorry

end NUMINAMATH_GPT_fraction_unchanged_when_increased_by_ten_l1410_141056


namespace NUMINAMATH_GPT_common_measure_largest_l1410_141044

theorem common_measure_largest {a b : ℕ} (h_a : a = 15) (h_b : b = 12): 
  (∀ c : ℕ, c ∣ a ∧ c ∣ b → c ≤ Nat.gcd a b) ∧ Nat.gcd a b = 3 := 
by
  sorry

end NUMINAMATH_GPT_common_measure_largest_l1410_141044


namespace NUMINAMATH_GPT_f_odd_f_inequality_solution_l1410_141000

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 ((1 + x) / (1 - x))

theorem f_odd: 
  ∀ x : ℝ, -1 < x ∧ x < 1 → f (-x) = - f x := 
by
  sorry

theorem f_inequality_solution:
  { x : ℝ // -1 < x ∧ x < 1 ∧ f x < -1 } = { x : ℝ // -1 < x ∧ x < -1/3 } := 
by 
  sorry

end NUMINAMATH_GPT_f_odd_f_inequality_solution_l1410_141000


namespace NUMINAMATH_GPT_MrKishoreSavings_l1410_141007

noncomputable def TotalExpenses : ℕ :=
  5000 + 1500 + 4500 + 2500 + 2000 + 5200

noncomputable def MonthlySalary : ℕ :=
  (TotalExpenses * 10) / 9

noncomputable def Savings : ℕ :=
  (MonthlySalary * 1) / 10

theorem MrKishoreSavings :
  Savings = 2300 :=
by
  sorry

end NUMINAMATH_GPT_MrKishoreSavings_l1410_141007


namespace NUMINAMATH_GPT_matrix_projection_ratios_l1410_141024

theorem matrix_projection_ratios (x y z : ℚ) (h : 
  (1 / 14 : ℚ) * x - (5 / 14 : ℚ) * y = x ∧
  - (5 / 14 : ℚ) * x + (24 / 14 : ℚ) * y = y ∧
  0 * x + 0 * y + 1 * z = z)
  : y / x = 13 / 5 ∧ z / x = 1 := 
by 
  sorry

end NUMINAMATH_GPT_matrix_projection_ratios_l1410_141024


namespace NUMINAMATH_GPT_main_l1410_141079

def prop_p (x0 : ℝ) : Prop := x0 > -2 ∧ 6 + abs x0 = 5
def p : Prop := ∃ x : ℝ, prop_p x

def q : Prop := ∀ x : ℝ, x < 0 → x^2 + 4 / x^2 ≥ 4

def r : Prop := ∀ x y : ℝ, abs x + abs y ≤ 1 → abs y / (abs x + 2) ≤ 1 / 2
def not_r : Prop := ∃ x y : ℝ, abs x + abs y > 1 ∧ abs y / (abs x + 2) > 1 / 2

theorem main : ¬ p ∧ ¬ p ∨ r ∧ (p ∧ q) := by
  sorry

end NUMINAMATH_GPT_main_l1410_141079


namespace NUMINAMATH_GPT_maciek_total_cost_l1410_141092

theorem maciek_total_cost :
  let p := 4
  let cost_of_chips := 1.75 * p
  let pretzels_cost := 2 * p
  let chips_cost := 2 * cost_of_chips
  let t := pretzels_cost + chips_cost
  t = 22 :=
by
  sorry

end NUMINAMATH_GPT_maciek_total_cost_l1410_141092


namespace NUMINAMATH_GPT_J_3_15_10_eq_68_over_15_l1410_141016

def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

theorem J_3_15_10_eq_68_over_15 : J 3 15 10 = 68 / 15 := by
  sorry

end NUMINAMATH_GPT_J_3_15_10_eq_68_over_15_l1410_141016


namespace NUMINAMATH_GPT_fifth_coordinate_is_14_l1410_141062

theorem fifth_coordinate_is_14
  (a : Fin 16 → ℝ)
  (h_1 : a 0 = 2)
  (h_16 : a 15 = 47)
  (h_avg : ∀ i : Fin 14, a (i + 1) = (a i + a (i + 2)) / 2) :
  a 4 = 14 :=
by
  sorry

end NUMINAMATH_GPT_fifth_coordinate_is_14_l1410_141062


namespace NUMINAMATH_GPT_max_area_rectangle_l1410_141054

theorem max_area_rectangle :
  ∃ (l w : ℕ), (2 * (l + w) = 40) ∧ (l ≥ w + 3) ∧ (l * w = 91) :=
by
  sorry

end NUMINAMATH_GPT_max_area_rectangle_l1410_141054


namespace NUMINAMATH_GPT_exists_acute_triangle_l1410_141084

-- Define the segments as a list of positive real numbers
variables (a b c d e : ℝ) (h0 : a > 0) (h1 : b > 0) (h2 : c > 0) (h3 : d > 0) (h4 : e > 0)
(h_order : a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e)

-- Conditions: Any three segments can form a triangle
variables (h_triangle_1 : a + b > c ∧ a + c > b ∧ b + c > a)
variables (h_triangle_2 : a + b > d ∧ a + d > b ∧ b + d > a)
variables (h_triangle_3 : a + b > e ∧ a + e > b ∧ b + e > a)
variables (h_triangle_4 : a + c > d ∧ a + d > c ∧ c + d > a)
variables (h_triangle_5 : a + c > e ∧ a + e > c ∧ c + e > a)
variables (h_triangle_6 : a + d > e ∧ a + e > d ∧ d + e > a)
variables (h_triangle_7 : b + c > d ∧ b + d > c ∧ c + d > b)
variables (h_triangle_8 : b + c > e ∧ b + e > c ∧ c + e > b)
variables (h_triangle_9 : b + d > e ∧ b + e > d ∧ d + e > b)
variables (h_triangle_10 : c + d > e ∧ c + e > d ∧ d + e > c)

-- Prove that there exists an acute-angled triangle 
theorem exists_acute_triangle : ∃ (x y z : ℝ), (x = a ∨ x = b ∨ x = c ∨ x = d ∨ x = e) ∧
                                        (y = a ∨ y = b ∨ y = c ∨ y = d ∨ y = e) ∧
                                        (z = a ∨ z = b ∨ z = c ∨ z = d ∨ z = e) ∧ 
                                        x + y > z ∧ x + z > y ∧ y + z > x ∧ 
                                        x^2 < y^2 + z^2 := 
sorry

end NUMINAMATH_GPT_exists_acute_triangle_l1410_141084


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1410_141064

def U := Set ℝ
def A := {x : ℝ | -2 ≤ x ∧ x ≤ 3 }
def B := {x : ℝ | x < -1}
def C := {x : ℝ | -2 ≤ x ∧ x < -1}

theorem intersection_of_A_and_B : A ∩ B = C :=
by sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1410_141064


namespace NUMINAMATH_GPT_wrench_force_l1410_141097

theorem wrench_force (F L k: ℝ) (h_inv: ∀ F L, F * L = k) (h_given: F * 12 = 240 * 12) : 
  (∀ L, (L = 16) → (F = 180)) ∧ (∀ L, (L = 8) → (F = 360)) := by 
sorry

end NUMINAMATH_GPT_wrench_force_l1410_141097


namespace NUMINAMATH_GPT_xiao_ming_correct_answers_l1410_141075

theorem xiao_ming_correct_answers :
  ∃ (m n : ℕ), m + n = 20 ∧ 5 * m - n = 76 ∧ m = 16 := 
by
  -- Definitions of points for correct and wrong answers
  let points_per_correct := 5 
  let points_deducted_per_wrong := 1

  -- Contestant's Scores and Conditions
  have contestant_a : 20 * points_per_correct - 0 * points_deducted_per_wrong = 100 := by sorry
  have contestant_b : 19 * points_per_correct - 1 * points_deducted_per_wrong = 94 := by sorry
  have contestant_c : 18 * points_per_correct - 2 * points_deducted_per_wrong = 88 := by sorry
  have contestant_d : 14 * points_per_correct - 6 * points_deducted_per_wrong = 64 := by sorry
  have contestant_e : 10 * points_per_correct - 10 * points_deducted_per_wrong = 40 := by sorry

  -- Xiao Ming's conditions translated to variables m and n
  have xiao_ming_conditions : (∃ m n : ℕ, m + n = 20 ∧ 5 * m - n = 76) := by sorry

  exact ⟨16, 4, rfl, rfl, rfl⟩

end NUMINAMATH_GPT_xiao_ming_correct_answers_l1410_141075


namespace NUMINAMATH_GPT_sum_of_interior_edges_l1410_141073

-- Define the problem parameters
def width_of_frame : ℝ := 2 -- width of the frame pieces in inches
def exposed_area : ℝ := 30 -- exposed area of the frame in square inches
def outer_edge_length : ℝ := 6 -- one of the outer edge length in inches

-- Define the statement to prove
theorem sum_of_interior_edges :
  ∃ (y : ℝ), (6 * y - 2 * (y - width_of_frame * 2) = exposed_area) ∧
  (2 * (6 - width_of_frame * 2) + 2 * (y - width_of_frame * 2) = 7) :=
sorry

end NUMINAMATH_GPT_sum_of_interior_edges_l1410_141073


namespace NUMINAMATH_GPT_lioness_age_l1410_141028

theorem lioness_age (H L : ℕ) 
  (h1 : L = 2 * H) 
  (h2 : (H / 2 + 5) + (L / 2 + 5) = 19) : 
  L = 12 :=
sorry

end NUMINAMATH_GPT_lioness_age_l1410_141028


namespace NUMINAMATH_GPT_sandbox_area_l1410_141071

def sandbox_length : ℕ := 312
def sandbox_width : ℕ := 146

theorem sandbox_area : sandbox_length * sandbox_width = 45552 := by
  sorry

end NUMINAMATH_GPT_sandbox_area_l1410_141071


namespace NUMINAMATH_GPT_set_inter_complement_l1410_141082

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem set_inter_complement :
  U = {1, 2, 3, 4, 5, 6, 7} ∧ A = {1, 2, 3, 4} ∧ B = {3, 5, 6} →
  A ∩ (U \ B) = {1, 2, 4} :=
by
  sorry

end NUMINAMATH_GPT_set_inter_complement_l1410_141082


namespace NUMINAMATH_GPT_sum_of_a_and_b_l1410_141059

theorem sum_of_a_and_b (a b : ℕ) (h1 : b > 1) (h2 : a^b < 500) (h3 : ∀ c d : ℕ, d > 1 → c^d < 500 → c^d ≤ a^b) : a + b = 24 :=
sorry

end NUMINAMATH_GPT_sum_of_a_and_b_l1410_141059


namespace NUMINAMATH_GPT_equation_of_line_l1410_141047

theorem equation_of_line (l : ℝ → ℝ) :
  (∀ (P : ℝ × ℝ), P = (4, 2) → 
    ∃ (a b : ℝ), ((P = ( (4 - a), (2 - b)) ∨ P = ( (4 + a), (2 + b))) ∧ 
    ((4 - a)^2 / 36 + (2 - b)^2 / 9 = 1) ∧ ((4 + a)^2 / 36 + (2 + b)^2 / 9 = 1)) ∧
    (P.2 = l P.1)) →
  (∀ (x y : ℝ), y = l x ↔ 2 * x + 3 * y - 16 = 0) :=
by
  intros h P hp
  sorry -- Placeholder for the proof

end NUMINAMATH_GPT_equation_of_line_l1410_141047


namespace NUMINAMATH_GPT_ratio_a3_b3_l1410_141070

theorem ratio_a3_b3 (a : ℝ) (ha : a ≠ 0)
  (h1 : a = b₁)
  (h2 : a * q * b = 2)
  (h3 : b₄ = 8 * a * q^3) :
  (∃ r : ℝ, r = -5 ∨ r = -3.2) :=
by
  sorry

end NUMINAMATH_GPT_ratio_a3_b3_l1410_141070


namespace NUMINAMATH_GPT_exists_nat_number_reduce_by_57_l1410_141051

theorem exists_nat_number_reduce_by_57 :
  ∃ (N : ℕ), ∃ (k : ℕ) (a x : ℕ),
    N = 10^k * a + x ∧
    10^k * a + x = 57 * x ∧
    N = 7125 :=
sorry

end NUMINAMATH_GPT_exists_nat_number_reduce_by_57_l1410_141051


namespace NUMINAMATH_GPT_weekly_crab_meat_cost_l1410_141078

-- Declare conditions as definitions
def dishes_per_day : ℕ := 40
def pounds_per_dish : ℝ := 1.5
def cost_per_pound : ℝ := 8
def closed_days_per_week : ℕ := 3
def days_per_week : ℕ := 7

-- Define the Lean statement to prove the weekly cost
theorem weekly_crab_meat_cost :
  let days_open_per_week := days_per_week - closed_days_per_week
  let pounds_per_day := dishes_per_day * pounds_per_dish
  let daily_cost := pounds_per_day * cost_per_pound
  let weekly_cost := daily_cost * (days_open_per_week : ℝ)
  weekly_cost = 1920 :=
by
  sorry

end NUMINAMATH_GPT_weekly_crab_meat_cost_l1410_141078


namespace NUMINAMATH_GPT_increasing_interval_m_range_l1410_141023

def y (x m : ℝ) : ℝ := x^2 + 2 * m * x + 10

theorem increasing_interval_m_range (m : ℝ) : (∀ x, 2 ≤ x → ∀ x', x' ≥ x → y x m ≤ y x' m) → (-2 : ℝ) ≤ m :=
sorry

end NUMINAMATH_GPT_increasing_interval_m_range_l1410_141023


namespace NUMINAMATH_GPT_price_increase_after_reduction_l1410_141060

theorem price_increase_after_reduction (P : ℝ) (h : P > 0) : 
  let reduced_price := P * 0.85
  let increase_factor := 1 / 0.85
  let percentage_increase := (increase_factor - 1) * 100
  percentage_increase = 17.65 := by
  sorry

end NUMINAMATH_GPT_price_increase_after_reduction_l1410_141060


namespace NUMINAMATH_GPT_markus_more_marbles_l1410_141096

theorem markus_more_marbles :
  let mara_bags := 12
  let marbles_per_mara_bag := 2
  let markus_bags := 2
  let marbles_per_markus_bag := 13
  let mara_marbles := mara_bags * marbles_per_mara_bag
  let markus_marbles := markus_bags * marbles_per_markus_bag
  mara_marbles + 2 = markus_marbles := 
by
  sorry

end NUMINAMATH_GPT_markus_more_marbles_l1410_141096


namespace NUMINAMATH_GPT_shifted_parabola_is_correct_l1410_141087

-- Define the initial parabola
def initial_parabola (x : ℝ) : ℝ :=
  -((x - 1) ^ 2) + 2

-- Define the shifted parabola
def shifted_parabola (x : ℝ) : ℝ :=
  -((x + 1 - 1) ^ 2) + 4

-- State the theorem
theorem shifted_parabola_is_correct :
  ∀ x : ℝ, shifted_parabola x = -x^2 + 4 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_shifted_parabola_is_correct_l1410_141087


namespace NUMINAMATH_GPT_solution_to_problem_l1410_141027

def f (x : ℝ) : ℝ := sorry

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x * y) = y^2 * f x + x^2 * f y

theorem solution_to_problem
  (f : ℝ → ℝ)
  (h : functional_equation f) :
  f 0 = 0 ∧ f 1 = 0 ∧ ∀ x : ℝ, f (-x) = f x :=
by
  sorry

end NUMINAMATH_GPT_solution_to_problem_l1410_141027


namespace NUMINAMATH_GPT_taxi_fare_distance_l1410_141010

theorem taxi_fare_distance (initial_fare : ℝ) (subsequent_fare : ℝ) (initial_distance : ℝ) (total_fare : ℝ) : 
  initial_fare = 2.0 →
  subsequent_fare = 0.60 →
  initial_distance = 1 / 5 →
  total_fare = 25.4 →
  ∃ d : ℝ, d = 8 :=
by 
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_taxi_fare_distance_l1410_141010


namespace NUMINAMATH_GPT_negation_of_proposition_l1410_141061

-- Definitions and conditions from the problem
def original_proposition (x : ℝ) : Prop := x^3 - x^2 + 1 > 0

-- The proof problem: Prove the negation
theorem negation_of_proposition : (¬ ∀ x : ℝ, original_proposition x) ↔ ∃ x : ℝ, ¬original_proposition x := 
by
  -- here we insert our proof later
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1410_141061


namespace NUMINAMATH_GPT_symmetric_line_equation_l1410_141030

theorem symmetric_line_equation (x y : ℝ) (h₁ : x + y + 1 = 0) : (2 - x) + (4 - y) - 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_line_equation_l1410_141030


namespace NUMINAMATH_GPT_f_even_l1410_141052

noncomputable def f : ℝ → ℝ := sorry

axiom f_not_identically_zero : ∃ x : ℝ, f x ≠ 0

axiom f_functional_eqn : ∀ a b : ℝ, 
  f (a + b) + f (a - b) = 2 * f a + 2 * f b

theorem f_even (x : ℝ) : f (-x) = f x :=
  sorry

end NUMINAMATH_GPT_f_even_l1410_141052


namespace NUMINAMATH_GPT_intersection_M_N_l1410_141041

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 ≠ x}

theorem intersection_M_N:
  M ∩ N = {-1} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1410_141041


namespace NUMINAMATH_GPT_maria_earnings_l1410_141098

def cost_of_brushes : ℕ := 20
def cost_of_canvas : ℕ := 3 * cost_of_brushes
def cost_per_liter_of_paint : ℕ := 8
def liters_of_paint : ℕ := 5
def cost_of_paint : ℕ := liters_of_paint * cost_per_liter_of_paint
def total_cost : ℕ := cost_of_brushes + cost_of_canvas + cost_of_paint
def selling_price : ℕ := 200

theorem maria_earnings : (selling_price - total_cost) = 80 := by
  sorry

end NUMINAMATH_GPT_maria_earnings_l1410_141098


namespace NUMINAMATH_GPT_expand_polynomials_l1410_141067

-- Define the given polynomials
def poly1 (x : ℝ) : ℝ := 12 * x^2 + 5 * x - 3
def poly2 (x : ℝ) : ℝ := 3 * x^3 + 2

-- Define the expected result of the polynomial multiplication
def expected (x : ℝ) : ℝ := 36 * x^5 + 15 * x^4 - 9 * x^3 + 24 * x^2 + 10 * x - 6

-- State the theorem
theorem expand_polynomials (x : ℝ) :
  (poly1 x) * (poly2 x) = expected x :=
by
  sorry

end NUMINAMATH_GPT_expand_polynomials_l1410_141067


namespace NUMINAMATH_GPT_find_x_l1410_141050

theorem find_x (x : ℤ) :
  3 < x ∧ x < 10 →
  5 < x ∧ x < 18 →
  -2 < x ∧ x < 9 →
  0 < x ∧ x < 8 →
  x + 1 < 9 →
  x = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_find_x_l1410_141050


namespace NUMINAMATH_GPT_quadratic_function_range_l1410_141045

theorem quadratic_function_range (x : ℝ) (y : ℝ) (h1 : y = x^2 - 2*x - 3) (h2 : -2 ≤ x ∧ x ≤ 2) :
  -4 ≤ y ∧ y ≤ 5 :=
sorry

end NUMINAMATH_GPT_quadratic_function_range_l1410_141045


namespace NUMINAMATH_GPT_tangent_line_through_origin_to_circle_in_third_quadrant_l1410_141042

theorem tangent_line_through_origin_to_circle_in_third_quadrant :
  ∃ m : ℝ, (∀ x y : ℝ, y = m * x) ∧ (∀ x y : ℝ, x^2 + y^2 + 4 * x + 3 = 0) ∧ (x < 0 ∧ y < 0) ∧ y = -3 * x :=
sorry

end NUMINAMATH_GPT_tangent_line_through_origin_to_circle_in_third_quadrant_l1410_141042


namespace NUMINAMATH_GPT_least_subtracted_divisible_by_5_l1410_141019

theorem least_subtracted_divisible_by_5 :
  ∃ n : ℕ, (568219 - n) % 5 = 0 ∧ n ≤ 4 ∧ (∀ m : ℕ, m < 4 → (568219 - m) % 5 ≠ 0) :=
sorry

end NUMINAMATH_GPT_least_subtracted_divisible_by_5_l1410_141019


namespace NUMINAMATH_GPT_new_salary_after_increase_l1410_141066

theorem new_salary_after_increase : 
  ∀ (previous_salary : ℝ) (percentage_increase : ℝ), 
    previous_salary = 2000 → percentage_increase = 0.05 → 
    previous_salary + (previous_salary * percentage_increase) = 2100 :=
by
  intros previous_salary percentage_increase h1 h2
  sorry

end NUMINAMATH_GPT_new_salary_after_increase_l1410_141066


namespace NUMINAMATH_GPT_fill_tank_with_two_pipes_l1410_141089

def Pipe (Rate : Type) := Rate

theorem fill_tank_with_two_pipes
  (capacity : ℝ)
  (three_pipes_fill_time : ℝ)
  (h1 : three_pipes_fill_time = 12)
  (pipe_rate : ℝ)
  (h2 : pipe_rate = capacity / 36) :
  2 * pipe_rate * 18 = capacity := 
by 
  sorry

end NUMINAMATH_GPT_fill_tank_with_two_pipes_l1410_141089


namespace NUMINAMATH_GPT_smallest_n_divisible_by_2009_l1410_141046

theorem smallest_n_divisible_by_2009 : ∃ n : ℕ, n > 1 ∧ (n^2 * (n - 1)) % 2009 = 0 ∧ (∀ m : ℕ, m > 1 → (m^2 * (m - 1)) % 2009 = 0 → m ≥ n) :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_divisible_by_2009_l1410_141046


namespace NUMINAMATH_GPT_cost_to_paint_cube_l1410_141048

theorem cost_to_paint_cube (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (side_length : ℝ) 
  (h1 : cost_per_kg = 40) 
  (h2 : coverage_per_kg = 20) 
  (h3 : side_length = 10) 
  : (6 * side_length^2 / coverage_per_kg) * cost_per_kg = 1200 :=
by
  sorry

end NUMINAMATH_GPT_cost_to_paint_cube_l1410_141048


namespace NUMINAMATH_GPT_fan_airflow_weekly_l1410_141029

def fan_airflow_per_second : ℕ := 10
def fan_work_minutes_per_day : ℕ := 10
def minutes_to_seconds (m : ℕ) : ℕ := m * 60
def days_per_week : ℕ := 7

theorem fan_airflow_weekly : 
  (fan_airflow_per_second * (minutes_to_seconds fan_work_minutes_per_day) * days_per_week) = 42000 := 
by
  sorry

end NUMINAMATH_GPT_fan_airflow_weekly_l1410_141029


namespace NUMINAMATH_GPT_valid_pairs_for_area_18_l1410_141037

theorem valid_pairs_for_area_18 (w l : ℕ) (hw : 0 < w) (hl : 0 < l) (h_area : w * l = 18) (h_lt : w < l) :
  (w, l) = (1, 18) ∨ (w, l) = (2, 9) ∨ (w, l) = (3, 6) :=
sorry

end NUMINAMATH_GPT_valid_pairs_for_area_18_l1410_141037


namespace NUMINAMATH_GPT_c_zero_roots_arithmetic_seq_range_f1_l1410_141090

section problem

variable (b : ℝ)
def f (x : ℝ) := x^3 + 3 * b * x^2 + 0 * x + (-2 * b^3)
def f' (x : ℝ) := 3 * x^2 + 6 * b * x + 0

-- Proving c = 0 if f(x) is increasing on (-∞, 0) and decreasing on (0, 2)
theorem c_zero (h_inc : ∀ x < 0, f' b x > 0) (h_dec : ∀ x > 0, f' b x < 0) : 0 = 0 := sorry

-- Proving f(x) = 0 has two other distinct real roots x1 and x2 different from -b, forming an arithmetic sequence
theorem roots_arithmetic_seq (hb : ∀ x : ℝ, f b x = 0 → (x = -b ∨ -b ≠ x)) : 
    ∃ (x1 x2 : ℝ), x1 ≠ -b ∧ x2 ≠ -b ∧ x1 + x2 = -2 * b := sorry

-- Proving the range of values for f(1) when the maximum value of f(x) is less than 16
theorem range_f1 (h_max : ∀ x : ℝ, f b x < 16 ) : 0 ≤ f b 1 ∧ f b 1 < 11 := sorry

end problem

end NUMINAMATH_GPT_c_zero_roots_arithmetic_seq_range_f1_l1410_141090


namespace NUMINAMATH_GPT_gcd_of_11121_and_12012_l1410_141095

def gcd_problem : Prop :=
  gcd 11121 12012 = 1

theorem gcd_of_11121_and_12012 : gcd_problem :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_gcd_of_11121_and_12012_l1410_141095


namespace NUMINAMATH_GPT_remainder_8547_div_9_l1410_141008

theorem remainder_8547_div_9 : 8547 % 9 = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_8547_div_9_l1410_141008


namespace NUMINAMATH_GPT_speed_conversion_l1410_141057

theorem speed_conversion (speed_kmh : ℝ) (conversion_factor : ℝ) :
  speed_kmh = 1.3 → conversion_factor = (1000 / 3600) → speed_kmh * conversion_factor = 0.3611 :=
by
  intros h_speed h_factor
  rw [h_speed, h_factor]
  norm_num
  sorry

end NUMINAMATH_GPT_speed_conversion_l1410_141057


namespace NUMINAMATH_GPT_jim_gold_per_hour_l1410_141055

theorem jim_gold_per_hour :
  ∀ (hours: ℕ) (treasure_chest: ℕ) (num_small_bags: ℕ)
    (each_small_bag_has: ℕ),
    hours = 8 →
    treasure_chest = 100 →
    num_small_bags = 2 →
    each_small_bag_has = (treasure_chest / 2) →
    (treasure_chest + num_small_bags * each_small_bag_has) / hours = 25 :=
by
  intros hours treasure_chest num_small_bags each_small_bag_has
  intros hours_eq treasure_chest_eq num_small_bags_eq small_bag_eq
  have total_gold : ℕ := treasure_chest + num_small_bags * each_small_bag_has
  have per_hour : ℕ := total_gold / hours
  sorry

end NUMINAMATH_GPT_jim_gold_per_hour_l1410_141055


namespace NUMINAMATH_GPT_product_4_6_7_14_l1410_141014

theorem product_4_6_7_14 : 4 * 6 * 7 * 14 = 2352 := by
  sorry

end NUMINAMATH_GPT_product_4_6_7_14_l1410_141014


namespace NUMINAMATH_GPT_smallest_positive_omega_l1410_141009

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * x - Real.pi / 6)

noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.sin (2 * ω * (x + Real.pi / 4) - Real.pi / 6)

theorem smallest_positive_omega (ω : ℝ) :
  (∀ x : ℝ, g (ω) x = g (ω) (-x)) → (ω = 4 / 3) := sorry

end NUMINAMATH_GPT_smallest_positive_omega_l1410_141009


namespace NUMINAMATH_GPT_coefficient_sum_of_squares_is_23456_l1410_141003

theorem coefficient_sum_of_squares_is_23456 
  (p q r s t u : ℤ)
  (h : ∀ x : ℤ, 1728 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) :
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 23456 := 
by
  sorry

end NUMINAMATH_GPT_coefficient_sum_of_squares_is_23456_l1410_141003


namespace NUMINAMATH_GPT_find_m_l1410_141006

def f (x m : ℝ) : ℝ := x^2 - 3 * x + m
def g (x m : ℝ) : ℝ := x^2 - 3 * x + 5 * m

theorem find_m :
  let m := 10 / 7
  3 * f 5 m = 2 * g 5 m :=
by
  sorry

end NUMINAMATH_GPT_find_m_l1410_141006


namespace NUMINAMATH_GPT_patients_per_doctor_l1410_141005

theorem patients_per_doctor (total_patients : ℕ) (total_doctors : ℕ) (h_patients : total_patients = 400) (h_doctors : total_doctors = 16) : 
  (total_patients / total_doctors) = 25 :=
by
  sorry

end NUMINAMATH_GPT_patients_per_doctor_l1410_141005


namespace NUMINAMATH_GPT_liquid_level_ratio_l1410_141018

theorem liquid_level_ratio (h1 h2 : ℝ) (r1 r2 : ℝ) (V_m : ℝ) 
  (h1_eq4h2 : h1 = 4 * h2) (r1_eq3 : r1 = 3) (r2_eq6 : r2 = 6) 
  (Vm_eq_four_over_three_Pi : V_m = (4/3) * Real.pi * 1^3) :
  ((4/9) : ℝ) / ((1/9) : ℝ) = (4 : ℝ) := 
by
  -- The proof details will be provided here.
  sorry

end NUMINAMATH_GPT_liquid_level_ratio_l1410_141018


namespace NUMINAMATH_GPT_problem_statement_l1410_141021

open Real

theorem problem_statement (x y z : ℝ)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (a : ℝ := x + x⁻¹) (b : ℝ := y + y⁻¹) (c : ℝ := z + z⁻¹) :
  a > 2 ∧ b > 2 ∧ c > 2 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1410_141021


namespace NUMINAMATH_GPT_cos_double_angle_l1410_141038

open Real

theorem cos_double_angle (α : ℝ) (h0 : 0 < α ∧ α < π) (h1 : sin α + cos α = 1 / 2) : cos (2 * α) = -sqrt 7 / 4 :=
by
  sorry

end NUMINAMATH_GPT_cos_double_angle_l1410_141038


namespace NUMINAMATH_GPT_yuna_has_biggest_number_l1410_141081

-- Define the collections
def yoongi_collected : ℕ := 4
def jungkook_collected : ℕ := 6 - 3
def yuna_collected : ℕ := 5

-- State the theorem
theorem yuna_has_biggest_number :
  yuna_collected > yoongi_collected ∧ yuna_collected > jungkook_collected :=
by
  sorry

end NUMINAMATH_GPT_yuna_has_biggest_number_l1410_141081


namespace NUMINAMATH_GPT_find_q_l1410_141043

def f (q : ℝ) : ℝ := 3 * q - 3

theorem find_q (q : ℝ) : f (f q) = 210 → q = 74 / 3 := by
  sorry

end NUMINAMATH_GPT_find_q_l1410_141043


namespace NUMINAMATH_GPT_jenny_run_distance_l1410_141053

theorem jenny_run_distance (walk_distance : ℝ) (ran_walk_diff : ℝ) (h_walk : walk_distance = 0.4) (h_diff : ran_walk_diff = 0.2) :
  (walk_distance + ran_walk_diff) = 0.6 :=
sorry

end NUMINAMATH_GPT_jenny_run_distance_l1410_141053


namespace NUMINAMATH_GPT_simplify_expression_l1410_141040

variable (a : ℝ)

theorem simplify_expression (h1 : 0 < a ∨ a < 0) : a * Real.sqrt (-(1 / a)) = -Real.sqrt (-a) :=
sorry

end NUMINAMATH_GPT_simplify_expression_l1410_141040


namespace NUMINAMATH_GPT_set_intersection_complement_l1410_141063

variable (U : Set ℕ)
variable (P Q : Set ℕ)

theorem set_intersection_complement {U : Set ℕ} {P Q : Set ℕ} 
  (hU : U = {1, 2, 3, 4, 5, 6}) 
  (hP : P = {1, 2, 3, 4}) 
  (hQ : Q = {3, 4, 5, 6}) : 
  P ∩ (U \ Q) = {1, 2} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_complement_l1410_141063


namespace NUMINAMATH_GPT_inequality_solution_set_l1410_141022

theorem inequality_solution_set :
  ∀ x : ℝ, 8 * x^3 + 9 * x^2 + 7 * x - 6 < 0 ↔ (( -6 < x ∧ x < -1/8) ∨ (-1/8 < x ∧ x < 1)) :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l1410_141022


namespace NUMINAMATH_GPT_green_flowers_count_l1410_141083

theorem green_flowers_count :
  ∀ (G R B Y T : ℕ),
    T = 96 →
    R = 3 * G →
    B = 48 →
    Y = 12 →
    G + R + B + Y = T →
    G = 9 :=
by
  intros G R B Y T
  intro hT
  intro hR
  intro hB
  intro hY
  intro hSum
  sorry

end NUMINAMATH_GPT_green_flowers_count_l1410_141083


namespace NUMINAMATH_GPT_find_natural_triples_l1410_141069

open Nat

noncomputable def satisfies_conditions (a b c : ℕ) : Prop :=
  (a + b) % c = 0 ∧ (b + c) % a = 0 ∧ (c + a) % b = 0

theorem find_natural_triples :
  ∀ (a b c : ℕ), satisfies_conditions a b c ↔
    (∃ a, (a = b ∧ b = c) ∨ 
          (a = b ∧ c = 2 * a) ∨ 
          (b = 2 * a ∧ c = 3 * a) ∨ 
          (b = 3 * a ∧ c = 2 * a) ∨ 
          (a = 2 * b ∧ c = 3 * b) ∨ 
          (a = 3 * b ∧ c = 2 * b)) :=
sorry

end NUMINAMATH_GPT_find_natural_triples_l1410_141069


namespace NUMINAMATH_GPT_sequence_non_positive_l1410_141011

theorem sequence_non_positive
  (a : ℕ → ℝ) (n : ℕ)
  (h0 : a 0 = 0)
  (hn : a n = 0)
  (h : ∀ k, 1 ≤ k → k ≤ n - 1 → a (k - 1) - 2 * a k + a (k + 1) ≥ 0) :
  ∀ k, k ≤ n → a k ≤ 0 := 
sorry

end NUMINAMATH_GPT_sequence_non_positive_l1410_141011


namespace NUMINAMATH_GPT_evaluate_expression_l1410_141015

theorem evaluate_expression : (3 + Real.sqrt 3 + 1 / (3 + Real.sqrt 3) + 1 / (Real.sqrt 3 - 3)) = 3 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1410_141015


namespace NUMINAMATH_GPT_y_coord_of_third_vertex_of_equilateral_l1410_141013

/-- Given two vertices of an equilateral triangle at (0, 6) and (10, 6), and the third vertex in the first quadrant,
    prove that the y-coordinate of the third vertex is 6 + 5 * sqrt 3. -/
theorem y_coord_of_third_vertex_of_equilateral (A B C : ℝ × ℝ)
  (hA : A = (0, 6)) (hB : B = (10, 6)) (hAB : dist A B = 10) (hC : C.2 > 6):
  C.2 = 6 + 5 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_y_coord_of_third_vertex_of_equilateral_l1410_141013


namespace NUMINAMATH_GPT_contrapositive_proof_l1410_141049

theorem contrapositive_proof (x : ℝ) : (x^2 < 1 → -1 < x ∧ x < 1) → (x ≥ 1 ∨ x ≤ -1 → x^2 ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_proof_l1410_141049


namespace NUMINAMATH_GPT_snack_eaters_left_l1410_141031

theorem snack_eaters_left (initial_participants : ℕ)
    (snack_initial : ℕ)
    (new_outsiders1 : ℕ)
    (half_left1 : ℕ)
    (new_outsiders2 : ℕ)
    (left2 : ℕ)
    (half_left2 : ℕ)
    (h1 : initial_participants = 200)
    (h2 : snack_initial = 100)
    (h3 : new_outsiders1 = 20)
    (h4 : half_left1 = (snack_initial + new_outsiders1) / 2)
    (h5 : new_outsiders2 = 10)
    (h6 : left2 = 30)
    (h7 : half_left2 = (half_left1 + new_outsiders2 - left2) / 2) :
    half_left2 = 20 := 
  sorry

end NUMINAMATH_GPT_snack_eaters_left_l1410_141031


namespace NUMINAMATH_GPT_evaluate_Y_l1410_141065

def Y (a b : ℤ) : ℤ := a^2 - 3 * a * b + b^2 + 3

theorem evaluate_Y : Y 2 5 = 2 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_Y_l1410_141065


namespace NUMINAMATH_GPT_shells_needed_l1410_141032

theorem shells_needed (current_shells : ℕ) (total_shells : ℕ) (difference : ℕ) :
  current_shells = 5 → total_shells = 17 → difference = total_shells - current_shells → difference = 12 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_shells_needed_l1410_141032


namespace NUMINAMATH_GPT_popsicles_eaten_l1410_141017

theorem popsicles_eaten (total_time : ℕ) (interval : ℕ) (p : ℕ)
  (h_total_time : total_time = 6 * 60)
  (h_interval : interval = 20) :
  p = total_time / interval :=
sorry

end NUMINAMATH_GPT_popsicles_eaten_l1410_141017


namespace NUMINAMATH_GPT_sqrt_meaningful_real_domain_l1410_141074

theorem sqrt_meaningful_real_domain (x : ℝ) (h : 6 - 4 * x ≥ 0) : x ≤ 3 / 2 :=
by sorry

end NUMINAMATH_GPT_sqrt_meaningful_real_domain_l1410_141074


namespace NUMINAMATH_GPT_twelve_times_reciprocal_sum_l1410_141094

theorem twelve_times_reciprocal_sum (a b c : ℚ) (h₁ : a = 1/3) (h₂ : b = 1/4) (h₃ : c = 1/6) :
  12 * (a + b + c)⁻¹ = 16 := 
by
  sorry

end NUMINAMATH_GPT_twelve_times_reciprocal_sum_l1410_141094


namespace NUMINAMATH_GPT_jet_bar_sales_difference_l1410_141001

variable (monday_sales : ℕ) (total_target : ℕ) (remaining_target : ℕ)
variable (sales_so_far : ℕ) (tuesday_sales : ℕ)
def JetBarsDifference : Prop :=
  monday_sales = 45 ∧ total_target = 90 ∧ remaining_target = 16 ∧
  sales_so_far = total_target - remaining_target ∧
  tuesday_sales = sales_so_far - monday_sales ∧
  (monday_sales - tuesday_sales = 16)

theorem jet_bar_sales_difference :
  JetBarsDifference 45 90 16 (90 - 16) (90 - 16 - 45) :=
by
  sorry

end NUMINAMATH_GPT_jet_bar_sales_difference_l1410_141001


namespace NUMINAMATH_GPT_log_inequality_l1410_141099

theorem log_inequality (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≠ 1) (h4 : y ≠ 1) :
    (Real.log y / Real.log x + Real.log x / Real.log y > 2) →
    (x ≠ y ∧ ((x > 1 ∧ y > 1) ∨ (x < 1 ∧ y < 1))) :=
by
    sorry

end NUMINAMATH_GPT_log_inequality_l1410_141099


namespace NUMINAMATH_GPT_negation_of_proposition_l1410_141085

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, 1 < x → (Real.log x / Real.log 2) + 4 * (Real.log 2 / Real.log x) > 4)) ↔
  (∃ x : ℝ, 1 < x ∧ (Real.log x / Real.log 2) + 4 * (Real.log 2 / Real.log x) ≤ 4) :=
sorry

end NUMINAMATH_GPT_negation_of_proposition_l1410_141085


namespace NUMINAMATH_GPT_shell_placements_l1410_141039

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem shell_placements : factorial 14 / 7 = 10480142147302400 := by
  sorry

end NUMINAMATH_GPT_shell_placements_l1410_141039


namespace NUMINAMATH_GPT_isosceles_triangle_construction_l1410_141072

noncomputable def isosceles_triangle_construction_impossible 
  (hb lb : ℝ) : Prop :=
  ∀ (α β : ℝ), 
  3 * β ≠ α

theorem isosceles_triangle_construction : 
  ∃ (hb lb : ℝ), isosceles_triangle_construction_impossible hb lb :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_construction_l1410_141072


namespace NUMINAMATH_GPT_opponent_final_score_l1410_141080

theorem opponent_final_score (x : ℕ) (h : x + 29 = 39) : x = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_opponent_final_score_l1410_141080


namespace NUMINAMATH_GPT_spinner_probability_l1410_141076

-- Define the game board conditions
def total_regions : ℕ := 12  -- The triangle is divided into 12 smaller regions
def shaded_regions : ℕ := 3  -- Three regions are shaded

-- Define the probability calculation
def probability (total : ℕ) (shaded : ℕ): ℚ := shaded / total

-- State the proof problem
theorem spinner_probability :
  probability total_regions shaded_regions = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_spinner_probability_l1410_141076


namespace NUMINAMATH_GPT_carl_profit_l1410_141020

-- Define the conditions
def price_per_watermelon : ℕ := 3
def watermelons_start : ℕ := 53
def watermelons_end : ℕ := 18

-- Define the number of watermelons sold
def watermelons_sold : ℕ := watermelons_start - watermelons_end

-- Define the profit
def profit : ℕ := watermelons_sold * price_per_watermelon

-- State the theorem about Carl's profit
theorem carl_profit : profit = 105 :=
by
  -- Proof can be filled in later
  sorry

end NUMINAMATH_GPT_carl_profit_l1410_141020


namespace NUMINAMATH_GPT_t_mobile_first_two_lines_cost_l1410_141002

theorem t_mobile_first_two_lines_cost :
  ∃ T : ℝ,
  (T + 16 * 3) = (45 + 14 * 3 + 11) → T = 50 :=
by
  sorry

end NUMINAMATH_GPT_t_mobile_first_two_lines_cost_l1410_141002


namespace NUMINAMATH_GPT_find_k_of_collinear_points_l1410_141077

theorem find_k_of_collinear_points :
  ∃ k : ℚ, ∀ (x1 y1 x2 y2 x3 y3 : ℚ), (x1, y1) = (4, 10) → (x2, y2) = (-3, k) → (x3, y3) = (-8, 5) → 
  ((y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)) → k = 85 / 12 :=
by
  sorry

end NUMINAMATH_GPT_find_k_of_collinear_points_l1410_141077


namespace NUMINAMATH_GPT_parabola_vertex_l1410_141058

theorem parabola_vertex :
  ∃ a k : ℝ, (∀ x y : ℝ, y^2 - 4*y + 2*x + 7 = 0 ↔ y = k ∧ x = a - (1/2)*(y - k)^2) ∧ a = -3/2 ∧ k = 2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_l1410_141058


namespace NUMINAMATH_GPT_smallest_z_l1410_141091

-- Given conditions
def distinct_consecutive_even_positive_perfect_cubes (w x y z : ℕ) : Prop :=
  w^3 + x^3 + y^3 = z^3 ∧
  ∃ a b c d : ℕ, 
    a < b ∧ b < c ∧ c < d ∧
    2 * a = w ∧ 2 * b = x ∧ 2 * c = y ∧ 2 * d = z

-- The smallest value of z proving the equation holds
theorem smallest_z (w x y z : ℕ) (h : distinct_consecutive_even_positive_perfect_cubes w x y z) : z = 12 :=
  sorry

end NUMINAMATH_GPT_smallest_z_l1410_141091


namespace NUMINAMATH_GPT_erased_length_l1410_141036

def original_length := 100 -- in cm
def final_length := 76 -- in cm

theorem erased_length : original_length - final_length = 24 :=
by
    sorry

end NUMINAMATH_GPT_erased_length_l1410_141036


namespace NUMINAMATH_GPT_find_x_l1410_141025

-- Definitions of binomial coefficients as conditions
def binomial (n k : ℕ) : ℕ := n.choose k

-- The specific conditions given
def C65_eq_6 : Prop := binomial 6 5 = 6
def C64_eq_15 : Prop := binomial 6 4 = 15

-- The theorem we need to prove: ∃ x, binomial 7 x = 21
theorem find_x (h1 : C65_eq_6) (h2 : C64_eq_15) : ∃ x, binomial 7 x = 21 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_find_x_l1410_141025


namespace NUMINAMATH_GPT_ny_mets_fans_count_l1410_141093

-- Define the known ratios and total fans
def ratio_Y_to_M (Y M : ℕ) : Prop := 3 * M = 2 * Y
def ratio_M_to_R (M R : ℕ) : Prop := 4 * R = 5 * M
def total_fans (Y M R : ℕ) : Prop := Y + M + R = 330

-- Define what we want to prove
theorem ny_mets_fans_count (Y M R : ℕ) (h1 : ratio_Y_to_M Y M) (h2 : ratio_M_to_R M R) (h3 : total_fans Y M R) : M = 88 :=
sorry

end NUMINAMATH_GPT_ny_mets_fans_count_l1410_141093


namespace NUMINAMATH_GPT_find_a_l1410_141068

theorem find_a (a k : ℝ) (h1 : ∀ x, a * x^2 + 3 * x - k = 0 → x = 7) (h2 : k = 119) : a = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1410_141068
