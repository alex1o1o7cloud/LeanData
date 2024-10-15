import Mathlib

namespace NUMINAMATH_GPT_visitors_not_ill_l1985_198522

theorem visitors_not_ill (total_visitors : ℕ) (percent_ill : ℕ) (H1 : total_visitors = 500) (H2 : percent_ill = 40) : 
  total_visitors * (100 - percent_ill) / 100 = 300 := 
by 
  sorry

end NUMINAMATH_GPT_visitors_not_ill_l1985_198522


namespace NUMINAMATH_GPT_chocolate_bar_cost_l1985_198575

-- Definitions based on the conditions given in the problem.
def total_bars : ℕ := 7
def remaining_bars : ℕ := 4
def total_money : ℚ := 9
def bars_sold : ℕ := total_bars - remaining_bars
def cost_per_bar := total_money / bars_sold

-- The theorem that needs to be proven.
theorem chocolate_bar_cost : cost_per_bar = 3 := by
  -- proof placeholder
  sorry

end NUMINAMATH_GPT_chocolate_bar_cost_l1985_198575


namespace NUMINAMATH_GPT__l1985_198573

-- Here we define our conditions

def parabola (x y : ℝ) := y^2 = 8 * x

def focus : ℝ × ℝ := (2, 0)

def point_on_parabola (y : ℝ) : ℝ × ℝ := (4, y)

example (y_P : ℝ) (hP : parabola 4 y_P) :
  dist (point_on_parabola y_P) focus = 6 := by
  -- Since we only need the theorem statement, we finish with sorry
  sorry

end NUMINAMATH_GPT__l1985_198573


namespace NUMINAMATH_GPT_zero_in_interval_l1985_198593

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^3 - 9

theorem zero_in_interval :
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) → -- f(x) is increasing on (0, +∞)
  f 2 < 0 → -- f(2) < 0
  f 3 > 0 → -- f(3) > 0
  ∃ c : ℝ, 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  intros h_increasing h_f2_lt_0 h_f3_gt_0
  sorry

end NUMINAMATH_GPT_zero_in_interval_l1985_198593


namespace NUMINAMATH_GPT_cubic_root_expression_l1985_198574

theorem cubic_root_expression (p q r : ℝ)
  (h₁ : p + q + r = 8)
  (h₂ : p * q + p * r + q * r = 11)
  (h₃ : p * q * r = 3) :
  p / (q * r + 1) + q / (p * r + 1) + r / (p * q + 1) = 32 / 15 :=
by 
  sorry

end NUMINAMATH_GPT_cubic_root_expression_l1985_198574


namespace NUMINAMATH_GPT_triangular_square_l1985_198530

def triangular (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem triangular_square (m n : ℕ) (h1 : 1 ≤ m) (h2 : 1 ≤ n) (h3 : 2 * triangular m = triangular n) :
  ∃ k : ℕ, triangular (2 * m - n) = k * k :=
by
  sorry

end NUMINAMATH_GPT_triangular_square_l1985_198530


namespace NUMINAMATH_GPT_simplify_expression_l1985_198528

theorem simplify_expression (x : ℝ) :
  ( ( ((x + 1) ^ 3 * (x ^ 2 - x + 1) ^ 3) / (x ^ 3 + 1) ^ 3 ) ^ 2 *
    ( ((x - 1) ^ 3 * (x ^ 2 + x + 1) ^ 3) / (x ^ 3 - 1) ^ 3 ) ^ 2 ) = 1 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1985_198528


namespace NUMINAMATH_GPT_resistor_problem_l1985_198584

theorem resistor_problem (R : ℝ)
  (initial_resistance : ℝ := 3 * R)
  (parallel_resistance : ℝ := R / 3)
  (resistance_change : ℝ := initial_resistance - parallel_resistance)
  (condition : resistance_change = 10) : 
  R = 3.75 := by
  sorry

end NUMINAMATH_GPT_resistor_problem_l1985_198584


namespace NUMINAMATH_GPT_graph_is_finite_distinct_points_l1985_198513

def cost (n : ℕ) : ℕ := 18 * n + 3

theorem graph_is_finite_distinct_points : 
  ∀ (n : ℕ), 1 ≤ n ∧ n ≤ 20 → 
  ∀ (m : ℕ), 1 ≤ m ∧ m ≤ 20 → 
  (cost n = cost m → n = m) ∧
  ∀ x : ℕ, ∃ n : ℕ, 1 ≤ n ∧ n ≤ 20 ∧ cost n = x :=
by
  sorry

end NUMINAMATH_GPT_graph_is_finite_distinct_points_l1985_198513


namespace NUMINAMATH_GPT_diamonds_in_G_20_equals_840_l1985_198519

def diamonds_in_G (n : ℕ) : ℕ :=
  if n < 3 then 1 else 2 * n * (n + 1)

theorem diamonds_in_G_20_equals_840 : diamonds_in_G 20 = 840 :=
by
  sorry

end NUMINAMATH_GPT_diamonds_in_G_20_equals_840_l1985_198519


namespace NUMINAMATH_GPT_find_age_of_D_l1985_198567

theorem find_age_of_D
(Eq1 : a + b + c + d = 108)
(Eq2 : a - b = 12)
(Eq3 : c - (a - 34) = 3 * (d - (a - 34)))
: d = 13 := 
sorry

end NUMINAMATH_GPT_find_age_of_D_l1985_198567


namespace NUMINAMATH_GPT_solve_cos_theta_l1985_198543

def cos_theta_proof (v1 v2 : ℝ × ℝ) (θ : ℝ) : Prop :=
  let dot_product := (v1.1 * v2.1 + v1.2 * v2.2)
  let norm_v1 := Real.sqrt (v1.1 ^ 2 + v1.2 ^ 2)
  let norm_v2 := Real.sqrt (v2.1 ^ 2 + v2.2 ^ 2)
  let cos_theta := dot_product / (norm_v1 * norm_v2)
  cos_theta = 43 / Real.sqrt 2173

theorem solve_cos_theta :
  cos_theta_proof (4, 5) (2, 7) (43 / Real.sqrt 2173) :=
by
  sorry

end NUMINAMATH_GPT_solve_cos_theta_l1985_198543


namespace NUMINAMATH_GPT_nissa_grooming_time_correct_l1985_198547

def clipping_time_per_claw : ℕ := 10
def cleaning_time_per_ear : ℕ := 90
def shampooing_time_minutes : ℕ := 5

def claws_per_foot : ℕ := 4
def feet_count : ℕ := 4
def ear_count : ℕ := 2

noncomputable def total_grooming_time_in_seconds : ℕ := 
  (clipping_time_per_claw * claws_per_foot * feet_count) + 
  (cleaning_time_per_ear * ear_count) + 
  (shampooing_time_minutes * 60) -- converting minutes to seconds

theorem nissa_grooming_time_correct :
  total_grooming_time_in_seconds = 640 := by
  sorry

end NUMINAMATH_GPT_nissa_grooming_time_correct_l1985_198547


namespace NUMINAMATH_GPT_domain_of_function_l1985_198596

def domain_conditions (x : ℝ) : Prop :=
  (1 - x ≥ 0) ∧ (x + 2 > 0)

theorem domain_of_function :
  {x : ℝ | domain_conditions x} = {x : ℝ | -2 < x ∧ x ≤ 1} :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l1985_198596


namespace NUMINAMATH_GPT_smallest_m_n_sum_l1985_198508

noncomputable def f (m n : ℕ) (x : ℝ) : ℝ := Real.arcsin (Real.log (n * x) / Real.log m)

theorem smallest_m_n_sum 
  (m n : ℕ) 
  (h_m1 : 1 < m) 
  (h_mn_closure : ∀ x, -1 ≤ Real.log (n * x) / Real.log m ∧ Real.log (n * x) / Real.log m ≤ 1) 
  (h_length : (m ^ 2 - 1) / (m * n) = 1 / 2021) : 
  m + n = 86259 := by
sorry

end NUMINAMATH_GPT_smallest_m_n_sum_l1985_198508


namespace NUMINAMATH_GPT_ralph_tv_hours_l1985_198551

theorem ralph_tv_hours :
  (4 * 5 + 6 * 2) = 32 :=
by
  sorry

end NUMINAMATH_GPT_ralph_tv_hours_l1985_198551


namespace NUMINAMATH_GPT_treasure_chest_total_value_l1985_198541

def base7_to_base10 (n : Nat) : Nat :=
  let rec convert (n acc base : Nat) : Nat :=
    if n = 0 then acc
    else convert (n / 10) (acc + (n % 10) * base) (base * 7)
  convert n 0 1

theorem treasure_chest_total_value :
  base7_to_base10 5346 + base7_to_base10 6521 + base7_to_base10 320 = 4305 :=
by
  sorry

end NUMINAMATH_GPT_treasure_chest_total_value_l1985_198541


namespace NUMINAMATH_GPT_eq_1_solution_eq_2_solution_eq_3_solution_eq_4_solution_l1985_198537

-- Equation (1): 2x^2 + 2x - 1 = 0
theorem eq_1_solution (x : ℝ) :
  2 * x^2 + 2 * x - 1 = 0 ↔ (x = (-1 + Real.sqrt 3) / 2 ∨ x = (-1 - Real.sqrt 3) / 2) := by
  sorry

-- Equation (2): x(x-1) = 2(x-1)
theorem eq_2_solution (x : ℝ) :
  x * (x - 1) = 2 * (x - 1) ↔ (x = 1 ∨ x = 2) := by
  sorry

-- Equation (3): 4(x-2)^2 = 9(2x+1)^2
theorem eq_3_solution (x : ℝ) :
  4 * (x - 2)^2 = 9 * (2 * x + 1)^2 ↔ (x = -7 / 4 ∨ x = 1 / 8) := by
  sorry

-- Equation (4): (2x-1)^2 - 3(2x-1) = 4
theorem eq_4_solution (x : ℝ) :
  (2 * x - 1)^2 - 3 * (2 * x - 1) = 4 ↔ (x = 5 / 2 ∨ x = 0) := by
  sorry

end NUMINAMATH_GPT_eq_1_solution_eq_2_solution_eq_3_solution_eq_4_solution_l1985_198537


namespace NUMINAMATH_GPT_helium_balloon_height_l1985_198586

theorem helium_balloon_height :
    let total_budget := 200
    let cost_sheet := 42
    let cost_rope := 18
    let cost_propane := 14
    let cost_per_ounce_helium := 1.5
    let height_per_ounce := 113
    let amount_spent := cost_sheet + cost_rope + cost_propane
    let money_left_for_helium := total_budget - amount_spent
    let ounces_helium := money_left_for_helium / cost_per_ounce_helium
    let total_height := height_per_ounce * ounces_helium
    total_height = 9492 := sorry

end NUMINAMATH_GPT_helium_balloon_height_l1985_198586


namespace NUMINAMATH_GPT_fraction_of_married_men_l1985_198506

theorem fraction_of_married_men (total_women married_women : ℕ) 
    (h1 : total_women = 7)
    (h2 : married_women = 4)
    (single_women_probability : ℚ)
    (h3 : single_women_probability = 3 / 7) : 
    (4 / 11 : ℚ) = (married_women / (total_women + married_women)) := 
sorry

end NUMINAMATH_GPT_fraction_of_married_men_l1985_198506


namespace NUMINAMATH_GPT_power_identity_l1985_198576

theorem power_identity (x a b : ℝ) (h1 : x^a = 2) (h2 : x^b = 3) : x^(3 * a + 2 * b) = 72 := 
  sorry

end NUMINAMATH_GPT_power_identity_l1985_198576


namespace NUMINAMATH_GPT_moses_income_l1985_198523

theorem moses_income (investment : ℝ) (percentage : ℝ) (dividend_rate : ℝ) (income : ℝ)
  (h1 : investment = 3000) (h2 : percentage = 0.72) (h3 : dividend_rate = 0.0504) :
  income = 210 :=
sorry

end NUMINAMATH_GPT_moses_income_l1985_198523


namespace NUMINAMATH_GPT_Clarissa_photos_needed_l1985_198559

theorem Clarissa_photos_needed :
  (7 + 10 + 9 <= 40) → 40 - (7 + 10 + 9) = 14 :=
by
  sorry

end NUMINAMATH_GPT_Clarissa_photos_needed_l1985_198559


namespace NUMINAMATH_GPT_ratio_books_purchased_l1985_198569

-- Definitions based on the conditions
def books_last_year : ℕ := 50
def books_before_purchase : ℕ := 100
def books_now : ℕ := 300

-- Let x be the multiple of the books purchased this year
def multiple_books_purchased_this_year (x : ℕ) : Prop :=
  books_now = books_before_purchase + books_last_year + books_last_year * x

-- Prove the ratio is 3:1
theorem ratio_books_purchased (x : ℕ) (h : multiple_books_purchased_this_year x) : x = 3 :=
  by sorry

end NUMINAMATH_GPT_ratio_books_purchased_l1985_198569


namespace NUMINAMATH_GPT_product_of_remaining_numbers_is_12_l1985_198515

noncomputable def final_numbers_product : ℕ := 
  12

theorem product_of_remaining_numbers_is_12 :
  ∀ (initial_ones initial_twos initial_threes initial_fours : ℕ)
  (erase_add_op : Π (a b c : ℕ), Prop),
  initial_ones = 11 ∧ initial_twos = 22 ∧ initial_threes = 33 ∧ initial_fours = 44 ∧
  (∀ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c → erase_add_op a b c) →
  (∃ (final1 final2 final3 : ℕ), erase_add_op 11 22 33 → final1 * final2 * final3 = final_numbers_product) :=
sorry

end NUMINAMATH_GPT_product_of_remaining_numbers_is_12_l1985_198515


namespace NUMINAMATH_GPT_distance_point_to_vertical_line_l1985_198560

/-- The distance from a point to a vertical line equals the absolute difference in the x-coordinates. -/
theorem distance_point_to_vertical_line (x1 y1 x2 : ℝ) (h_line : x2 = -2) (h_point : (x1, y1) = (1, 2)) :
  abs (x1 - x2) = 3 :=
by
  -- Place proof here
  sorry

end NUMINAMATH_GPT_distance_point_to_vertical_line_l1985_198560


namespace NUMINAMATH_GPT_jaime_can_buy_five_apples_l1985_198509

theorem jaime_can_buy_five_apples :
  ∀ (L M : ℝ),
  (L = M / 2 + 1 / 2) →
  (M / 3 = L / 4 + 1 / 2) →
  (15 / M = 5) :=
by
  intros L M h1 h2
  sorry

end NUMINAMATH_GPT_jaime_can_buy_five_apples_l1985_198509


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1985_198520

variable {a : ℕ → ℕ} -- Assuming a_n is a function from natural numbers to natural numbers

theorem arithmetic_sequence_problem (h1 : a 1 + a 2 = 10) (h2 : a 4 = a 3 + 2) :
  a 3 + a 4 = 18 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1985_198520


namespace NUMINAMATH_GPT_initial_men_checking_exam_papers_l1985_198562

theorem initial_men_checking_exam_papers :
  ∀ (M : ℕ),
  (M * 8 * 5 = (1/2 : ℝ) * (2 * 20 * 8)) → M = 4 :=
by
  sorry

end NUMINAMATH_GPT_initial_men_checking_exam_papers_l1985_198562


namespace NUMINAMATH_GPT_square_of_99_l1985_198591

theorem square_of_99 : 99 * 99 = 9801 :=
by sorry

end NUMINAMATH_GPT_square_of_99_l1985_198591


namespace NUMINAMATH_GPT_solution_to_fractional_equation_l1985_198501

theorem solution_to_fractional_equation (x : ℝ) (h₁ : 2 / (x - 3) = 1 / x) (h₂ : x ≠ 3) (h₃ : x ≠ 0) : x = -3 :=
sorry

end NUMINAMATH_GPT_solution_to_fractional_equation_l1985_198501


namespace NUMINAMATH_GPT_distance_between_foci_correct_l1985_198502

/-- Define the given conditions for the ellipse -/
def ellipse_center : ℝ × ℝ := (3, -2)
def semi_major_axis : ℝ := 7
def semi_minor_axis : ℝ := 3

/-- Define the distance between the foci of the ellipse -/
noncomputable def distance_between_foci : ℝ :=
  2 * Real.sqrt (semi_major_axis ^ 2 - semi_minor_axis ^ 2)

theorem distance_between_foci_correct :
  distance_between_foci = 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_GPT_distance_between_foci_correct_l1985_198502


namespace NUMINAMATH_GPT_Sara_Jim_equal_savings_l1985_198570

theorem Sara_Jim_equal_savings:
  ∃ (w : ℕ), (∃ (sara_saved jim_saved : ℕ),
  sara_saved = 4100 + 10 * w ∧
  jim_saved = 15 * w ∧
  sara_saved = jim_saved) → w = 820 :=
by
  sorry

end NUMINAMATH_GPT_Sara_Jim_equal_savings_l1985_198570


namespace NUMINAMATH_GPT_unripe_oranges_zero_l1985_198535

def oranges_per_day (harvest_duration : ℕ) (ripe_oranges_per_day : ℕ) : ℕ :=
  harvest_duration * ripe_oranges_per_day

theorem unripe_oranges_zero
  (harvest_duration : ℕ)
  (ripe_oranges_per_day : ℕ)
  (total_ripe_oranges : ℕ)
  (h1 : harvest_duration = 25)
  (h2 : ripe_oranges_per_day = 82)
  (h3 : total_ripe_oranges = 2050)
  (h4 : oranges_per_day harvest_duration ripe_oranges_per_day = total_ripe_oranges) :
  ∀ unripe_oranges_per_day, unripe_oranges_per_day = 0 :=
by
  sorry

end NUMINAMATH_GPT_unripe_oranges_zero_l1985_198535


namespace NUMINAMATH_GPT_negation_example_l1985_198510

theorem negation_example : ¬(∀ x : ℝ, x^2 + |x| ≥ 0) ↔ ∃ x : ℝ, x^2 + |x| < 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_example_l1985_198510


namespace NUMINAMATH_GPT_total_disks_in_bag_l1985_198585

/-- Given that the number of blue disks b, yellow disks y, and green disks g are in the ratio 3:7:8,
    and there are 30 more green disks than blue disks (g = b + 30),
    prove that the total number of disks is 108. -/
theorem total_disks_in_bag (b y g : ℕ) (h1 : 3 * y = 7 * b) (h2 : 8 * y = 7 * g) (h3 : g = b + 30) :
  b + y + g = 108 := by
  sorry

end NUMINAMATH_GPT_total_disks_in_bag_l1985_198585


namespace NUMINAMATH_GPT_solution_set_of_inequality_l1985_198525

theorem solution_set_of_inequality :
  { x : ℝ | x^2 - 5 * x + 6 ≤ 0 } = { x : ℝ | 2 ≤ x ∧ x ≤ 3 } :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l1985_198525


namespace NUMINAMATH_GPT_father_catches_up_l1985_198590

noncomputable def min_steps_to_catch_up : Prop :=
  let x := 30
  let father_steps := 5
  let xiaoming_steps := 8
  let distance_ratio := 2 / 5
  let xiaoming_headstart := 27
  ((xiaoming_headstart + (xiaoming_steps / father_steps) * x) / distance_ratio) = x

theorem father_catches_up : min_steps_to_catch_up :=
  by
  sorry

end NUMINAMATH_GPT_father_catches_up_l1985_198590


namespace NUMINAMATH_GPT_quadratic_expression_positive_l1985_198516

theorem quadratic_expression_positive
  (a b c : ℝ) (x : ℝ)
  (h1 : a + b > c)
  (h2 : a + c > b)
  (h3 : b + c > a) :
  b^2 * x^2 + (b^2 + c^2 - a^2) * x + c^2 > 0 :=
sorry

end NUMINAMATH_GPT_quadratic_expression_positive_l1985_198516


namespace NUMINAMATH_GPT_normal_price_of_article_l1985_198589

theorem normal_price_of_article 
  (final_price : ℝ) 
  (d1 d2 d3 : ℝ) 
  (P : ℝ) 
  (h_final_price : final_price = 36) 
  (h_d1 : d1 = 0.15) 
  (h_d2 : d2 = 0.25) 
  (h_d3 : d3 = 0.20) 
  (h_eq : final_price = P * (1 - d1) * (1 - d2) * (1 - d3)) : 
  P = 70.59 := sorry

end NUMINAMATH_GPT_normal_price_of_article_l1985_198589


namespace NUMINAMATH_GPT_butterfly_development_time_l1985_198518

theorem butterfly_development_time :
  ∀ (larva_time cocoon_time : ℕ), 
  (larva_time = 3 * cocoon_time) → 
  (cocoon_time = 30) → 
  (larva_time + cocoon_time = 120) :=
by 
  intros larva_time cocoon_time h1 h2
  sorry

end NUMINAMATH_GPT_butterfly_development_time_l1985_198518


namespace NUMINAMATH_GPT_train_length_eq_l1985_198533

theorem train_length_eq (L : ℝ) (time_tree time_platform length_platform : ℝ)
  (h_tree : time_tree = 60) (h_platform : time_platform = 105) (h_length_platform : length_platform = 450)
  (h_speed_eq : L / time_tree = (L + length_platform) / time_platform) :
  L = 600 :=
by
  sorry

end NUMINAMATH_GPT_train_length_eq_l1985_198533


namespace NUMINAMATH_GPT_sum_first_n_terms_arithmetic_sequence_l1985_198521

def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n m : ℕ, a (m + 1) - a m = d

theorem sum_first_n_terms_arithmetic_sequence (a : ℕ → ℤ) (S : ℕ → ℤ) (a12 : a 12 = -8) (S9 : S 9 = -9) (h_arith : is_arithmetic_sequence a) :
  S 16 = -72 :=
sorry

end NUMINAMATH_GPT_sum_first_n_terms_arithmetic_sequence_l1985_198521


namespace NUMINAMATH_GPT_area_of_BEIH_l1985_198505

structure Point where
  x : ℚ
  y : ℚ

def B : Point := ⟨0, 0⟩
def A : Point := ⟨0, 2⟩
def D : Point := ⟨2, 2⟩
def C : Point := ⟨2, 0⟩
def E : Point := ⟨0, 1⟩
def F : Point := ⟨1, 0⟩
def I : Point := ⟨2/5, 6/5⟩
def H : Point := ⟨2/3, 2/3⟩

def quadrilateral_area (p1 p2 p3 p4 : Point) : ℚ :=
  (1/2 : ℚ) * 
  ((p1.x * p2.y + p2.x * p3.y + p3.x * p4.y + p4.x * p1.y) - 
   (p1.y * p2.x + p2.y * p3.x + p3.y * p4.x + p4.y * p1.x))

theorem area_of_BEIH : quadrilateral_area B E I H = 7 / 15 := sorry

end NUMINAMATH_GPT_area_of_BEIH_l1985_198505


namespace NUMINAMATH_GPT_linear_equation_value_l1985_198554

-- Define the conditions of the equation
def equation_is_linear (m : ℝ) : Prop :=
  |m| = 1 ∧ m - 1 ≠ 0

-- Prove the equivalence statement
theorem linear_equation_value (m : ℝ) (h : equation_is_linear m) : m = -1 := 
sorry

end NUMINAMATH_GPT_linear_equation_value_l1985_198554


namespace NUMINAMATH_GPT_dodecagon_diagonals_l1985_198557

def numDiagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem dodecagon_diagonals : numDiagonals 12 = 54 := by
  sorry

end NUMINAMATH_GPT_dodecagon_diagonals_l1985_198557


namespace NUMINAMATH_GPT_arithmetic_sequence_is_a_l1985_198512

theorem arithmetic_sequence_is_a
  (a : ℚ) (d : ℚ)
  (h1 : 140 + d = a)
  (h2 : a + d = 45 / 28)
  (h3 : a > 0) :
  a = 3965 / 56 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_is_a_l1985_198512


namespace NUMINAMATH_GPT_sales_revenue_nonnegative_l1985_198597

def revenue (x : ℝ) : ℝ := -10 * x^2 + 200 * x + 15000

theorem sales_revenue_nonnegative (x : ℝ) (hx : x = 9 ∨ x = 11) : revenue x ≥ 15950 :=
by
  cases hx
  case inl h₁ =>
    sorry -- calculation for x = 9
  case inr h₂ =>
    sorry -- calculation for x = 11

end NUMINAMATH_GPT_sales_revenue_nonnegative_l1985_198597


namespace NUMINAMATH_GPT_tan_alpha_implies_fraction_l1985_198556

theorem tan_alpha_implies_fraction (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_tan_alpha_implies_fraction_l1985_198556


namespace NUMINAMATH_GPT_smallest_y_square_factor_l1985_198564

theorem smallest_y_square_factor (y n : ℕ) (h₀ : y = 10) 
  (h₁ : ∀ m : ℕ, ∃ k : ℕ, k * k = m * y)
  (h₂ : ∀ (y' : ℕ), (∀ m : ℕ, ∃ k : ℕ, k * k = m * y') → y ≤ y') : 
  n = 10 :=
by sorry

end NUMINAMATH_GPT_smallest_y_square_factor_l1985_198564


namespace NUMINAMATH_GPT_max_value_of_expression_l1985_198587

noncomputable def maxExpression (x y : ℝ) :=
  x^5 * y + x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4 + x * y^5

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  maxExpression x y ≤ (656^2 / 18) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_expression_l1985_198587


namespace NUMINAMATH_GPT_lcm_18_60_is_180_l1985_198511

theorem lcm_18_60_is_180 : Nat.lcm 18 60 = 180 := 
  sorry

end NUMINAMATH_GPT_lcm_18_60_is_180_l1985_198511


namespace NUMINAMATH_GPT_circle_problem_l1985_198546

theorem circle_problem 
  (x y : ℝ)
  (h : x^2 + 8*x - 10*y = 10 - y^2 + 6*x) :
  let a := -1
  let b := 5
  let r := 6
  a + b + r = 10 :=
by sorry

end NUMINAMATH_GPT_circle_problem_l1985_198546


namespace NUMINAMATH_GPT_find_y_l1985_198503

theorem find_y (x : ℝ) (h1 : x = 1.3333333333333333) (h2 : (x * y) / 3 = x^2) : y = 4 :=
by 
  sorry

end NUMINAMATH_GPT_find_y_l1985_198503


namespace NUMINAMATH_GPT_order_magnitudes_ln_subtraction_l1985_198583

noncomputable def ln (x : ℝ) : ℝ := Real.log x -- Assuming the natural logarithm definition for real numbers

theorem order_magnitudes_ln_subtraction :
  (ln (3/2) - (3/2)) > (ln 3 - 3) ∧ 
  (ln 3 - 3) > (ln π - π) :=
sorry

end NUMINAMATH_GPT_order_magnitudes_ln_subtraction_l1985_198583


namespace NUMINAMATH_GPT_snowdrift_ratio_l1985_198552

theorem snowdrift_ratio
  (depth_first_day : ℕ := 20)
  (depth_second_day : ℕ)
  (h1 : depth_second_day + 24 = 34)
  (h2 : depth_second_day = 10) :
  depth_second_day / depth_first_day = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_snowdrift_ratio_l1985_198552


namespace NUMINAMATH_GPT_range_of_a_l1985_198529

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 ≤ x → x^2 + 2 * a * x + 1 ≥ 0) ↔ a ≥ -1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1985_198529


namespace NUMINAMATH_GPT_smallest_n_l1985_198580

theorem smallest_n (n : ℕ) (k : ℕ) (a m : ℕ) 
  (h1 : 0 ≤ k)
  (h2 : k < n)
  (h3 : a ≡ k [MOD n])
  (h4 : m > 0) :
  (∀ a m, (∃ k, a = n * k + 5) -> (a^2 - 3*a + 1) ∣ (a^m + 3^m) → false) 
  → n = 11 := sorry

end NUMINAMATH_GPT_smallest_n_l1985_198580


namespace NUMINAMATH_GPT_general_formula_sum_first_n_terms_l1985_198536

variable (a : ℕ → ℕ)
variable (S : ℕ → ℕ)

axiom a_initial : a 1 = 1
axiom a_recurrence : ∀ n : ℕ, n > 0 → a (n + 1) = 3 * a n * (1 + 1 / n)

theorem general_formula : ∀ n : ℕ, n > 0 → a n = n * 3^(n - 1) :=
by
  sorry

theorem sum_first_n_terms : ∀ n : ℕ, S n = (2 * n - 1) * 3^n + 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_general_formula_sum_first_n_terms_l1985_198536


namespace NUMINAMATH_GPT_how_many_bottles_did_maria_drink_l1985_198542

-- Define the conditions as variables and constants.
variable (x : ℕ)
def initial_bottles : ℕ := 14
def bought_bottles : ℕ := 45
def total_bottles_after_drinking_and_buying : ℕ := 51

-- The goal is to prove that Maria drank 8 bottles of water.
theorem how_many_bottles_did_maria_drink (h : initial_bottles - x + bought_bottles = total_bottles_after_drinking_and_buying) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_how_many_bottles_did_maria_drink_l1985_198542


namespace NUMINAMATH_GPT_total_chips_is_90_l1985_198571

theorem total_chips_is_90
  (viv_vanilla : ℕ)
  (sus_choco : ℕ)
  (viv_choco_more : ℕ)
  (sus_vanilla_ratio : ℚ)
  (viv_choco : ℕ)
  (sus_vanilla : ℕ)
  (total_choco : ℕ)
  (total_vanilla : ℕ)
  (total_chips : ℕ) :
  viv_vanilla = 20 →
  sus_choco = 25 →
  viv_choco_more = 5 →
  sus_vanilla_ratio = 3 / 4 →
  viv_choco = sus_choco + viv_choco_more →
  sus_vanilla = (sus_vanilla_ratio * viv_vanilla) →
  total_choco = viv_choco + sus_choco →
  total_vanilla = viv_vanilla + sus_vanilla →
  total_chips = total_choco + total_vanilla →
  total_chips = 90 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_chips_is_90_l1985_198571


namespace NUMINAMATH_GPT_grasshopper_jump_distance_l1985_198578

variable (F G M : ℕ) -- F for frog's jump, G for grasshopper's jump, M for mouse's jump

theorem grasshopper_jump_distance (h1 : F = G + 39) (h2 : M = F - 94) (h3 : F = 58) : G = 19 := 
by
  sorry

end NUMINAMATH_GPT_grasshopper_jump_distance_l1985_198578


namespace NUMINAMATH_GPT_problem_statement_l1985_198582

noncomputable def alpha := 3 + Real.sqrt 8
noncomputable def beta := 3 - Real.sqrt 8
noncomputable def x := alpha ^ 1000
noncomputable def n := Int.floor x
noncomputable def f := x - n

theorem problem_statement : x * (1 - f) = 1 :=
by sorry

end NUMINAMATH_GPT_problem_statement_l1985_198582


namespace NUMINAMATH_GPT_log_base_2_y_l1985_198561

theorem log_base_2_y (y : ℝ) (h : y = (Real.log 3 / Real.log 9) ^ Real.log 27 / Real.log 3) : 
  Real.log y = -3 :=
by
  sorry

end NUMINAMATH_GPT_log_base_2_y_l1985_198561


namespace NUMINAMATH_GPT_angle_B_value_triangle_perimeter_l1985_198594

open Real

variables {A B C a b c : ℝ}

-- Statement 1
theorem angle_B_value (h1 : a = b * sin A + sqrt 3 * a * cos B) : B = π / 2 := by
  sorry

-- Statement 2
theorem triangle_perimeter 
  (h1 : B = π / 2)
  (h2 : b = 4)
  (h3 : (1 / 2) * a * c = 4) : 
  a + b + c = 4 + 4 * sqrt 2 := by
  sorry


end NUMINAMATH_GPT_angle_B_value_triangle_perimeter_l1985_198594


namespace NUMINAMATH_GPT_angle_in_third_quadrant_l1985_198534

theorem angle_in_third_quadrant (θ : ℝ) (hθ : θ = 2014) : 180 < θ % 360 ∧ θ % 360 < 270 :=
by
  sorry

end NUMINAMATH_GPT_angle_in_third_quadrant_l1985_198534


namespace NUMINAMATH_GPT_mod_sum_correct_l1985_198531

theorem mod_sum_correct (a b c : ℕ) (ha : a < 7) (hb : b < 7) (hc : c < 7)
    (h1 : a * b * c ≡ 1 [MOD 7])
    (h2 : 5 * c ≡ 2 [MOD 7])
    (h3 : 6 * b ≡ 3 + b [MOD 7]) :
    (a + b + c) % 7 = 4 := sorry

end NUMINAMATH_GPT_mod_sum_correct_l1985_198531


namespace NUMINAMATH_GPT_professor_oscar_review_questions_l1985_198514

-- Define the problem conditions.
def students_per_class : ℕ := 35
def questions_per_exam : ℕ := 10
def number_of_classes : ℕ := 5

-- Define the number of questions that must be reviewed.
def total_questions_to_review : ℕ := 1750

-- The theorem to be proved.
theorem professor_oscar_review_questions :
  students_per_class * questions_per_exam * number_of_classes = total_questions_to_review :=
by
  -- Here we write 'sorry' since we are not providing the full proof.
  sorry

end NUMINAMATH_GPT_professor_oscar_review_questions_l1985_198514


namespace NUMINAMATH_GPT_investment_rate_l1985_198558

theorem investment_rate (total : ℝ) (invested_at_3_percent : ℝ) (rate_3_percent : ℝ) 
                        (invested_at_5_percent : ℝ) (rate_5_percent : ℝ) 
                        (desired_income : ℝ) (remaining : ℝ) (additional_income : ℝ) (r : ℝ) : 
  total = 12000 ∧ 
  invested_at_3_percent = 5000 ∧ 
  rate_3_percent = 0.03 ∧ 
  invested_at_5_percent = 4000 ∧ 
  rate_5_percent = 0.05 ∧ 
  desired_income = 600 ∧ 
  remaining = total - invested_at_3_percent - invested_at_5_percent ∧ 
  additional_income = desired_income - (invested_at_3_percent * rate_3_percent + invested_at_5_percent * rate_5_percent) ∧ 
  r = (additional_income / remaining) * 100 → 
  r = 8.33 := 
by
  sorry

end NUMINAMATH_GPT_investment_rate_l1985_198558


namespace NUMINAMATH_GPT_Mike_siblings_l1985_198572

-- Define the types for EyeColor, HairColor and Sport
inductive EyeColor
| Blue
| Green

inductive HairColor
| Black
| Blonde

inductive Sport
| Soccer
| Basketball

-- Define the Child structure
structure Child where
  name : String
  eyeColor : EyeColor
  hairColor : HairColor
  favoriteSport : Sport

-- Define all the children based on the given conditions
def Lily : Child := { name := "Lily", eyeColor := EyeColor.Green, hairColor := HairColor.Black, favoriteSport := Sport.Soccer }
def Mike : Child := { name := "Mike", eyeColor := EyeColor.Blue, hairColor := HairColor.Blonde, favoriteSport := Sport.Basketball }
def Oliver : Child := { name := "Oliver", eyeColor := EyeColor.Blue, hairColor := HairColor.Black, favoriteSport := Sport.Soccer }
def Emma : Child := { name := "Emma", eyeColor := EyeColor.Green, hairColor := HairColor.Blonde, favoriteSport := Sport.Basketball }
def Jacob : Child := { name := "Jacob", eyeColor := EyeColor.Blue, hairColor := HairColor.Blonde, favoriteSport := Sport.Soccer }
def Sophia : Child := { name := "Sophia", eyeColor := EyeColor.Green, hairColor := HairColor.Blonde, favoriteSport := Sport.Soccer }

-- Siblings relation
def areSiblings (c1 c2 c3 : Child) : Prop :=
  (c1.eyeColor = c2.eyeColor ∨ c1.hairColor = c2.hairColor ∨ c1.favoriteSport = c2.favoriteSport) ∧
  (c1.eyeColor = c3.eyeColor ∨ c1.hairColor = c3.hairColor ∨ c1.favoriteSport = c3.favoriteSport) ∧
  (c2.eyeColor = c3.eyeColor ∨ c2.hairColor = c3.hairColor ∨ c2.favoriteSport = c3.favoriteSport)

-- The proof statement
theorem Mike_siblings : areSiblings Mike Emma Jacob := by
  -- Proof must be provided here
  sorry

end NUMINAMATH_GPT_Mike_siblings_l1985_198572


namespace NUMINAMATH_GPT_eggs_distribution_l1985_198568

theorem eggs_distribution
  (total_eggs : ℕ)
  (eggs_per_adult : ℕ)
  (num_adults : ℕ)
  (num_girls : ℕ)
  (num_boys : ℕ)
  (eggs_per_girl : ℕ)
  (total_eggs_def : total_eggs = 3 * 12)
  (eggs_per_adult_def : eggs_per_adult = 3)
  (num_adults_def : num_adults = 3)
  (num_girls_def : num_girls = 7)
  (num_boys_def : num_boys = 10)
  (eggs_per_girl_def : eggs_per_girl = 1) :
  ∃ eggs_per_boy : ℕ, eggs_per_boy - eggs_per_girl = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_eggs_distribution_l1985_198568


namespace NUMINAMATH_GPT_range_of_a_l1985_198592

theorem range_of_a (a : ℝ) : 
  (¬ (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0) ∧ (∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0)) → a > 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1985_198592


namespace NUMINAMATH_GPT_flight_time_sum_l1985_198524

theorem flight_time_sum (h m : ℕ)
  (Hdep : true)   -- Placeholder condition for the departure time being 3:45 PM
  (Hlay : 25 = 25)   -- Placeholder condition for the layover being 25 minutes
  (Harr : true)   -- Placeholder condition for the arrival time being 8:02 PM
  (HsameTZ : true)   -- Placeholder condition for the same time zone
  (H0m : 0 < m) 
  (Hm60 : m < 60)
  (Hfinal_time : (h, m) = (3, 52)) : 
  h + m = 55 := 
by {
  sorry
}

end NUMINAMATH_GPT_flight_time_sum_l1985_198524


namespace NUMINAMATH_GPT_sum_of_interior_edges_l1985_198566

def frame_width : ℝ := 1
def outer_length : ℝ := 5
def frame_area : ℝ := 18
def inner_length1 : ℝ := outer_length - 2 * frame_width

/-- Given conditions and required to prove:
1. The frame is made of one-inch-wide pieces of wood.
2. The area of just the frame is 18 square inches.
3. One of the outer edges of the frame is 5 inches long.
Prove: The sum of the lengths of the four interior edges is 14 inches.
-/
theorem sum_of_interior_edges (inner_length2 : ℝ) 
  (h1 : (outer_length * (inner_length2 + 2) - inner_length1 * inner_length2) = frame_area)
  (h2 : (inner_length2 - 2) / 2 = 1) : 
  inner_length1 + inner_length1 + inner_length2 + inner_length2 = 14 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_edges_l1985_198566


namespace NUMINAMATH_GPT_inequality_solution_inequality_solution_b_monotonic_increasing_monotonic_decreasing_l1985_198579

variable (a : ℝ) (x : ℝ)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.sqrt (x^2 + 1) - a * x

-- Part (1)
theorem inequality_solution (a : ℝ) (h1 : 0 < a ∧ a < 1) : (0 ≤ x ∧ x ≤ 2*a / (1 - a^2)) → (f x a ≤ 1) :=
sorry

theorem inequality_solution_b (a : ℝ) (h2 : a ≥ 1) : (0 ≤ x) → (f x a ≤ 1) :=
sorry

-- Part (2)
theorem monotonic_increasing (a : ℝ) (h3 : a ≤ 0) (x1 x2 : ℝ) (hx : 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 < x2) : f x1 a ≤ f x2 a :=
sorry

theorem monotonic_decreasing (a : ℝ) (h4 : a ≥ 1) (x1 x2 : ℝ) (hx : 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 < x2) : f x1 a ≥ f x2 a :=
sorry

end NUMINAMATH_GPT_inequality_solution_inequality_solution_b_monotonic_increasing_monotonic_decreasing_l1985_198579


namespace NUMINAMATH_GPT_tan_double_angle_l1985_198504

open Real

-- Given condition
def condition (x : ℝ) : Prop := tan x - 1 / tan x = 3 / 2

-- Main theorem to prove
theorem tan_double_angle (x : ℝ) (h : condition x) : tan (2 * x) = -4 / 3 := by
  sorry

end NUMINAMATH_GPT_tan_double_angle_l1985_198504


namespace NUMINAMATH_GPT_highest_growth_rate_at_K_div_2_l1985_198588

variable {K : ℝ}

-- Define the population growth rate as a function of the population size.
def population_growth_rate (N : ℝ) : ℝ := sorry

-- Define the S-shaped curve condition of population growth.
axiom s_shaped_curve : ∃ N : ℝ, population_growth_rate N = 0 ∧ population_growth_rate (N/2) > population_growth_rate N

theorem highest_growth_rate_at_K_div_2 (N : ℝ) (hN : N = K/2) :
  population_growth_rate N > population_growth_rate K :=
by
  sorry

end NUMINAMATH_GPT_highest_growth_rate_at_K_div_2_l1985_198588


namespace NUMINAMATH_GPT_number_of_M_subsets_l1985_198555

def P : Set ℤ := {0, 1, 2}
def Q : Set ℤ := {0, 2, 4}

theorem number_of_M_subsets (M : Set ℤ) (hP : M ⊆ P) (hQ : M ⊆ Q) : 
  ∃ n : ℕ, n = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_M_subsets_l1985_198555


namespace NUMINAMATH_GPT_union_area_of_reflected_triangles_l1985_198539

open Real

noncomputable def pointReflected (P : ℝ × ℝ) (line_y : ℝ) : ℝ × ℝ :=
  (P.1, 2 * line_y - P.2)

def areaOfTriangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem union_area_of_reflected_triangles :
  let A := (2, 6)
  let B := (5, -2)
  let C := (7, 3)
  let line_y := 2
  let A' := pointReflected A line_y
  let B' := pointReflected B line_y
  let C' := pointReflected C line_y
  areaOfTriangle A B C + areaOfTriangle A' B' C' = 29 := sorry

end NUMINAMATH_GPT_union_area_of_reflected_triangles_l1985_198539


namespace NUMINAMATH_GPT_lattice_points_on_sphere_at_distance_5_with_x_1_l1985_198500

theorem lattice_points_on_sphere_at_distance_5_with_x_1 :
  let points := [(1, 0, 4), (1, 0, -4), (1, 4, 0), (1, -4, 0),
                 (1, 2, 4), (1, 2, -4), (1, -2, 4), (1, -2, -4),
                 (1, 4, 2), (1, 4, -2), (1, -4, 2), (1, -4, -2),
                 (1, 2, 2), (1, 2, -2), (1, -2, 2), (1, -2, -2)]
  (hs : ∀ y z, (1, y, z) ∈ points → 1^2 + y^2 + z^2 = 25) →
  24 = points.length :=
sorry

end NUMINAMATH_GPT_lattice_points_on_sphere_at_distance_5_with_x_1_l1985_198500


namespace NUMINAMATH_GPT_combined_height_after_1_year_l1985_198563

def initial_heights : ℕ := 200 + 150 + 250
def spring_and_summer_growth_A : ℕ := (6 * 4 / 2) * 50
def spring_and_summer_growth_B : ℕ := (6 * 4 / 3) * 70
def spring_and_summer_growth_C : ℕ := (6 * 4 / 4) * 90
def autumn_and_winter_growth_A : ℕ := (6 * 4 / 2) * 25
def autumn_and_winter_growth_B : ℕ := (6 * 4 / 3) * 35
def autumn_and_winter_growth_C : ℕ := (6 * 4 / 4) * 45

def total_growth_A : ℕ := spring_and_summer_growth_A + autumn_and_winter_growth_A
def total_growth_B : ℕ := spring_and_summer_growth_B + autumn_and_winter_growth_B
def total_growth_C : ℕ := spring_and_summer_growth_C + autumn_and_winter_growth_C

def total_growth : ℕ := total_growth_A + total_growth_B + total_growth_C

def combined_height : ℕ := initial_heights + total_growth

theorem combined_height_after_1_year : combined_height = 3150 := by
  sorry

end NUMINAMATH_GPT_combined_height_after_1_year_l1985_198563


namespace NUMINAMATH_GPT_total_balls_without_holes_l1985_198548

theorem total_balls_without_holes 
  (soccer_balls : ℕ) (soccer_balls_with_hole : ℕ)
  (basketballs : ℕ) (basketballs_with_hole : ℕ)
  (h1 : soccer_balls = 40)
  (h2 : soccer_balls_with_hole = 30)
  (h3 : basketballs = 15)
  (h4 : basketballs_with_hole = 7) :
  soccer_balls - soccer_balls_with_hole + (basketballs - basketballs_with_hole) = 18 := by
  sorry

end NUMINAMATH_GPT_total_balls_without_holes_l1985_198548


namespace NUMINAMATH_GPT_math_problem_l1985_198553

theorem math_problem
  (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a)
  (h : a / (b - c) + b / (c - a) + c / (a - b) = 1) :
  a / (b - c)^2 + b / (c - a)^2 + c / (a - b)^2 = 1 / (b - c) + 1 / (c - a) + 1 / (a - b) :=
  sorry

end NUMINAMATH_GPT_math_problem_l1985_198553


namespace NUMINAMATH_GPT_shifted_linear_function_correct_l1985_198527

def original_function (x : ℝ) : ℝ := 5 * x - 8
def shifted_function (x : ℝ) : ℝ := original_function x + 4

theorem shifted_linear_function_correct (x : ℝ) :
  shifted_function x = 5 * x - 4 :=
by
  sorry

end NUMINAMATH_GPT_shifted_linear_function_correct_l1985_198527


namespace NUMINAMATH_GPT_binomial_10_2_equals_45_l1985_198550

open Nat

theorem binomial_10_2_equals_45 : Nat.choose 10 2 = 45 := 
by
  sorry

end NUMINAMATH_GPT_binomial_10_2_equals_45_l1985_198550


namespace NUMINAMATH_GPT_sum_of_geometric_ratios_l1985_198538

theorem sum_of_geometric_ratios (k p r : ℝ) (a2 a3 b2 b3 : ℝ)
  (hk : k ≠ 0) (hp : p ≠ r)
  (ha2 : a2 = k * p) (ha3 : a3 = k * p * p)
  (hb2 : b2 = k * r) (hb3 : b3 = k * r * r)
  (h : a3 - b3 = 3 * (a2 - b2)) :
  p + r = 3 :=
by sorry

end NUMINAMATH_GPT_sum_of_geometric_ratios_l1985_198538


namespace NUMINAMATH_GPT_range_of_x_l1985_198581

theorem range_of_x (a : ℕ → ℝ) (x : ℝ)
  (h_pos : ∀ n, a n > 0)
  (h_a1 : a 1 = 1)
  (h_condition : ∀ n, a (n + 1)^2 + a n^2 < (5 / 2) * a (n + 1) * a n)
  (h_a2 : a 2 = 3 / 2)
  (h_a3 : a 3 = x)
  (h_a4 : a 4 = 4) : 2 < x ∧ x < 3 := by
  sorry

end NUMINAMATH_GPT_range_of_x_l1985_198581


namespace NUMINAMATH_GPT_find_width_of_river_l1985_198507

theorem find_width_of_river
    (total_distance : ℕ)
    (river_width : ℕ)
    (prob_find_item : ℚ)
    (h1 : total_distance = 500)
    (h2 : prob_find_item = 4/5)
    : river_width = 100 :=
by
    sorry

end NUMINAMATH_GPT_find_width_of_river_l1985_198507


namespace NUMINAMATH_GPT_find_cupcakes_l1985_198565

def total_students : ℕ := 20
def treats_per_student : ℕ := 4
def cookies : ℕ := 20
def brownies : ℕ := 35
def total_treats : ℕ := total_students * treats_per_student
def cupcakes : ℕ := total_treats - (cookies + brownies)

theorem find_cupcakes : cupcakes = 25 := by
  sorry

end NUMINAMATH_GPT_find_cupcakes_l1985_198565


namespace NUMINAMATH_GPT_guests_accommodation_l1985_198540

open Nat

theorem guests_accommodation :
  let guests := 15
  let rooms := 4
  (4 ^ 15 - 4 * 3 ^ 15 + 6 * 2 ^ 15 - 4 = 4 ^ 15 - 4 * 3 ^ 15 + 6 * 2 ^ 15 - 4) :=
by
  sorry

end NUMINAMATH_GPT_guests_accommodation_l1985_198540


namespace NUMINAMATH_GPT_decreasing_interval_l1985_198599

def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the derivative function
def f_prime (x : ℝ) : ℝ := 3*x^2 - 3

theorem decreasing_interval : ∀ x : ℝ, -1 < x ∧ x < 1 → f_prime x < 0 :=
by
  intro x h
  have h1: x^2 < 1 := by
    sorry
  have h2: 3*x^2 < 3 := by
    sorry
  have h3: 3*x^2 - 3 < 0 := by
    sorry
  exact h3

end NUMINAMATH_GPT_decreasing_interval_l1985_198599


namespace NUMINAMATH_GPT_find_d_minus_c_l1985_198577

theorem find_d_minus_c (c d x : ℝ) (h : c ≤ 3 * x - 2 ∧ 3 * x - 2 ≤ d) : (d - c = 45) :=
  sorry

end NUMINAMATH_GPT_find_d_minus_c_l1985_198577


namespace NUMINAMATH_GPT_remaining_hours_needed_l1985_198544

noncomputable
def hours_needed_to_finish (x : ℚ) : Prop :=
  (1/5 : ℚ) * (2 + x) + (1/8 : ℚ) * x = 1

theorem remaining_hours_needed :
  ∃ x : ℚ, hours_needed_to_finish x ∧ x = 24/13 :=
by
  use 24/13
  sorry

end NUMINAMATH_GPT_remaining_hours_needed_l1985_198544


namespace NUMINAMATH_GPT_vanessa_made_16_l1985_198595

/-
Each chocolate bar in a box costs $4.
There are 11 bars in total in the box.
Vanessa sold all but 7 bars.
Prove that Vanessa made $16.
-/

def cost_per_bar : ℕ := 4
def total_bars : ℕ := 11
def bars_unsold : ℕ := 7
def bars_sold : ℕ := total_bars - bars_unsold
def money_made : ℕ := bars_sold * cost_per_bar

theorem vanessa_made_16 : money_made = 16 :=
by
  sorry

end NUMINAMATH_GPT_vanessa_made_16_l1985_198595


namespace NUMINAMATH_GPT_min_power_for_84_to_divide_336_l1985_198532

theorem min_power_for_84_to_divide_336 : 
  ∃ n : ℕ, (∀ m : ℕ, 84^m % 336 = 0 → m ≥ n) ∧ n = 2 := 
sorry

end NUMINAMATH_GPT_min_power_for_84_to_divide_336_l1985_198532


namespace NUMINAMATH_GPT_student_B_most_stable_l1985_198526

variable (S_A S_B S_C : ℝ)
variables (hA : S_A^2 = 2.6) (hB : S_B^2 = 1.7) (hC : S_C^2 = 3.5)

/-- Student B has the most stable performance among students A, B, and C based on their variances.
    Given the conditions:
    - S_A^2 = 2.6
    - S_B^2 = 1.7
    - S_C^2 = 3.5
    we prove that student B has the most stable performance.
-/
theorem student_B_most_stable : S_B^2 < S_A^2 ∧ S_B^2 < S_C^2 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_student_B_most_stable_l1985_198526


namespace NUMINAMATH_GPT_triangle_side_split_l1985_198549

theorem triangle_side_split
  (PQ QR PR : ℝ)  -- Triangle sides
  (PS SR : ℝ)     -- Segments of PR divided by angle bisector
  (h_ratio : PQ / QR = 3 / 4)
  (h_sum : PR = 15)
  (h_PS_SR : PS / SR = 3 / 4)
  (h_PR_split : PS + SR = PR) :
  SR = 60 / 7 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_split_l1985_198549


namespace NUMINAMATH_GPT_find_f_4500_l1985_198598

noncomputable def f : ℕ → ℕ
| 0 => 1
| (n + 3) => f n + 2 * n + 3
| n => sorry  -- This handles all other cases, but should not be called.

theorem find_f_4500 : f 4500 = 6750001 :=
by
  sorry

end NUMINAMATH_GPT_find_f_4500_l1985_198598


namespace NUMINAMATH_GPT_donation_ratio_l1985_198545

theorem donation_ratio (D1 : ℝ) (D1_value : D1 = 10)
  (total_donation : D1 + D1 * 2 + D1 * 4 + D1 * 8 + D1 * 16 = 310) : 
  2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_donation_ratio_l1985_198545


namespace NUMINAMATH_GPT_minimum_value_of_16b_over_ac_l1985_198517

noncomputable def minimum_16b_over_ac (a b c : ℝ) (A B C : ℝ) : ℝ :=
  if (0 < B) ∧ (B < Real.pi / 2) ∧
     (Real.cos B ^ 2 + (1 / 2) * Real.sin (2 * B) = 1) ∧
     ((Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B) = 3)) then
    16 * b / (a * c)
  else 0

theorem minimum_value_of_16b_over_ac (a b c : ℝ) (A B C : ℝ)
  (h1 : 0 < B)
  (h2 : B < Real.pi / 2)
  (h3 : Real.cos B ^ 2 + (1 / 2) * Real.sin (2 * B) = 1)
  (h4 : Real.sqrt (a^2 + c^2 - 2 * a * c * Real.cos B) = 3) :
  minimum_16b_over_ac a b c A B C = 16 * (2 - Real.sqrt 2) / 3 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_16b_over_ac_l1985_198517
