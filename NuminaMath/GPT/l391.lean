import Mathlib

namespace NUMINAMATH_GPT_percentage_range_l391_39188

noncomputable def minimum_maximum_percentage (x y z n m : ℝ) (hx1 : 0 < x) (hx2 : 0 < y) (hx3 : 0 < z) (hx4 : 0 < n) (hx5 : 0 < m)
    (h1 : 4 * x * n = y * m) 
    (h2 : x * n + y * m = z * (m + n)) 
    (h3 : 16 ≤ y - x ∧ y - x ≤ 20) 
    (h4 : 42 ≤ z ∧ z ≤ 60) : ℝ × ℝ := sorry

theorem percentage_range (x y z n m : ℝ) (hx1 : 0 < x) (hx2 : 0 < y) (hx3 : 0 < z) (hx4 : 0 < n) (hx5 : 0 < m)
    (h1 : 4 * x * n = y * m) 
    (h2 : x * n + y * m = z * (m + n)) 
    (h3 : 16 ≤ y - x ∧ y - x ≤ 20) 
    (h4 : 42 ≤ z ∧ z ≤ 60) : 
    minimum_maximum_percentage x y z n m hx1 hx2 hx3 hx4 hx5 h1 h2 h3 h4 = (12.5, 15) :=
sorry

end NUMINAMATH_GPT_percentage_range_l391_39188


namespace NUMINAMATH_GPT_angle_B_l391_39103

theorem angle_B (A B C a b c : ℝ) (h : 2 * b * (Real.cos A) = 2 * c - Real.sqrt 3 * a) :
  B = Real.pi / 6 :=
sorry

end NUMINAMATH_GPT_angle_B_l391_39103


namespace NUMINAMATH_GPT_total_pizza_eaten_l391_39177

def don_pizzas : ℝ := 80
def daria_pizzas : ℝ := 2.5 * don_pizzas
def total_pizzas : ℝ := don_pizzas + daria_pizzas

theorem total_pizza_eaten : total_pizzas = 280 := by
  sorry

end NUMINAMATH_GPT_total_pizza_eaten_l391_39177


namespace NUMINAMATH_GPT_number_of_positive_solutions_l391_39122

theorem number_of_positive_solutions (x y z : ℕ) (h_cond : x > 0 ∧ y > 0 ∧ z > 0 ∧ x + y + z = 12) :
    ∃ (n : ℕ), n = 55 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_positive_solutions_l391_39122


namespace NUMINAMATH_GPT_candy_cost_l391_39118

theorem candy_cost (J H C : ℕ) (h1 : J + 7 = C) (h2 : H + 1 = C) (h3 : J + H < C) : C = 7 :=
by
  sorry

end NUMINAMATH_GPT_candy_cost_l391_39118


namespace NUMINAMATH_GPT_boats_meet_time_l391_39161

theorem boats_meet_time (v_A v_C current distance : ℝ) : 
  v_A = 7 → 
  v_C = 3 → 
  current = 2 → 
  distance = 20 → 
  (distance / (v_A + current + v_C - current) = 2 ∨
   distance / (v_A + current - (v_C + current)) = 5) := 
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  -- Apply simplifications or calculations as necessary
  sorry

end NUMINAMATH_GPT_boats_meet_time_l391_39161


namespace NUMINAMATH_GPT_helen_cookies_till_last_night_l391_39110

theorem helen_cookies_till_last_night 
  (cookies_yesterday : Nat := 31) 
  (cookies_day_before_yesterday : Nat := 419) : 
  cookies_yesterday + cookies_day_before_yesterday = 450 := 
by
  sorry

end NUMINAMATH_GPT_helen_cookies_till_last_night_l391_39110


namespace NUMINAMATH_GPT_loop_condition_l391_39182

theorem loop_condition (b : ℕ) : (b = 10 ∧ ∀ n, b = 10 + 3 * n ∧ b < 16 → n + 1 = 16) → ∀ (condition : ℕ → Prop), condition b → b = 16 :=
by sorry

end NUMINAMATH_GPT_loop_condition_l391_39182


namespace NUMINAMATH_GPT_find_m_l391_39199

-- Definition and conditions
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

noncomputable def vertex_property (a b c : ℝ) : Prop := 
  (∀ x, quadratic a b c x ≤ quadratic a b c 2) ∧ quadratic a b c 2 = 4

noncomputable def passes_through_origin (a b c : ℝ) : Prop :=
  quadratic a b c 0 = -7

-- Main theorem statement
theorem find_m (a b c m : ℝ) 
  (h1 : vertex_property a b c) 
  (h2 : passes_through_origin a b c) 
  (h3 : quadratic a b c 5 = m) :
  m = -83/4 :=
sorry

end NUMINAMATH_GPT_find_m_l391_39199


namespace NUMINAMATH_GPT_determine_d_l391_39128

variables (u v : ℝ × ℝ × ℝ) -- defining u and v as 3D vectors

noncomputable def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.2 * b.2.1 - a.2.1 * b.2.2, a.1 * b.2.2 - a.2.2 * b.1 , a.2.1 * b.1 - a.1 * b.2.1)

noncomputable def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2

noncomputable def i : ℝ × ℝ × ℝ := (1, 0, 0)
noncomputable def j : ℝ × ℝ × ℝ := (0, 1, 0)
noncomputable def k : ℝ × ℝ × ℝ := (0, 0, 1)

theorem determine_d (u : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) :
  cross_product i (cross_product (u + v) i) +
  cross_product j (cross_product (u + v) j) +
  cross_product k (cross_product (u + v) k) =
  2 * (u + v) :=
sorry

end NUMINAMATH_GPT_determine_d_l391_39128


namespace NUMINAMATH_GPT_triangle_area_solution_l391_39170

noncomputable def solve_for_x (x : ℝ) : Prop :=
  x > 0 ∧ (1 / 2 * x * 3 * x = 96) → x = 8

theorem triangle_area_solution : solve_for_x 8 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_solution_l391_39170


namespace NUMINAMATH_GPT_unfolded_paper_has_eight_holes_l391_39152

theorem unfolded_paper_has_eight_holes
  (T : Type)
  (equilateral_triangle : T)
  (midpoint : T → T → T)
  (vertex_fold : T → T → T)
  (holes_punched : T → ℕ)
  (first_fold_vertex midpoint_1 : T)
  (second_fold_vertex midpoint_2 : T)
  (holes_near_first_fold holes_near_second_fold : ℕ) :
  holes_punched (vertex_fold second_fold_vertex midpoint_2)
    = 8 := 
by sorry

end NUMINAMATH_GPT_unfolded_paper_has_eight_holes_l391_39152


namespace NUMINAMATH_GPT_sin_inverse_equation_l391_39134

noncomputable def a := Real.arcsin (4/5)
noncomputable def b := Real.arctan 1
noncomputable def c := Real.arccos (1/3)
noncomputable def sin_a_plus_b_minus_c := Real.sin (a + b - c)

theorem sin_inverse_equation : sin_a_plus_b_minus_c = 11 / 15 := sorry

end NUMINAMATH_GPT_sin_inverse_equation_l391_39134


namespace NUMINAMATH_GPT_range_of_a_l391_39119

theorem range_of_a (p q : Prop)
  (hp : ∀ a : ℝ, (1 < a ↔ p))
  (hq : ∀ a : ℝ, (2 ≤ a ∨ a ≤ -2 ↔ q))
  (hpq : ∀ a : ℝ, ∀ (p : Prop), ∀ (q : Prop), (p ∧ q) → p ∧ q) :
    ∀ a : ℝ, p ∧ q → 2 ≤ a :=
sorry

end NUMINAMATH_GPT_range_of_a_l391_39119


namespace NUMINAMATH_GPT_max_x4_y6_l391_39107

noncomputable def maximum_product (x y : ℝ) := x^4 * y^6

theorem max_x4_y6 (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 100) :
  maximum_product x y ≤ maximum_product 40 60 := sorry

end NUMINAMATH_GPT_max_x4_y6_l391_39107


namespace NUMINAMATH_GPT_total_number_of_games_l391_39108

theorem total_number_of_games (n : ℕ) (k : ℕ) (teams : Finset ℕ)
  (h_n : n = 8) (h_k : k = 2) (h_teams : teams.card = n) :
  (teams.card.choose k) = 28 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_games_l391_39108


namespace NUMINAMATH_GPT_delta_max_success_ratio_l391_39149

/-- In a two-day math challenge, Gamma and Delta both attempted questions totalling 600 points. 
    Gamma scored 180 points out of 300 points attempted each day.
    Delta attempted a different number of points each day and their daily success ratios were less by both days than Gamma's, 
    whose overall success ratio was 3/5. Prove that the maximum possible two-day success ratio that Delta could have achieved was 359/600. -/
theorem delta_max_success_ratio :
  ∀ (x y z w : ℕ), (0 < x) ∧ (0 < y) ∧ (0 < z) ∧ (0 < w) ∧ (x ≤ (3 * y) / 5) ∧ (z ≤ (3 * w) / 5) ∧ (y + w = 600) ∧ (x + z < 360)
  → (x + z) / 600 ≤ 359 / 600 :=
by
  sorry

end NUMINAMATH_GPT_delta_max_success_ratio_l391_39149


namespace NUMINAMATH_GPT_eighth_box_contains_65_books_l391_39121

theorem eighth_box_contains_65_books (total_books boxes first_seven_books per_box eighth_box : ℕ) :
  total_books = 800 →
  boxes = 8 →
  first_seven_books = 7 →
  per_box = 105 →
  eighth_box = total_books - (first_seven_books * per_box) →
  eighth_box = 65 := by
  sorry

end NUMINAMATH_GPT_eighth_box_contains_65_books_l391_39121


namespace NUMINAMATH_GPT_derek_february_savings_l391_39180

theorem derek_february_savings :
  ∀ (savings : ℕ → ℕ),
  (savings 1 = 2) ∧
  (∀ n : ℕ, 1 ≤ n ∧ n < 12 → savings (n + 1) = 2 * savings n) ∧
  (savings 12 = 4096) →
  savings 2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_derek_february_savings_l391_39180


namespace NUMINAMATH_GPT_total_garbage_collected_l391_39192

def Daliah := 17.5
def Dewei := Daliah - 2
def Zane := 4 * Dewei
def Bela := Zane + 3.75

theorem total_garbage_collected :
  Daliah + Dewei + Zane + Bela = 160.75 :=
by
  sorry

end NUMINAMATH_GPT_total_garbage_collected_l391_39192


namespace NUMINAMATH_GPT_part1_q1_l391_39196

open Set Real

def A (m : ℝ) : Set ℝ := {x | 2 * m - 1 ≤ x ∧ x ≤ m + 2}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}
def U : Set ℝ := univ

theorem part1_q1 (m : ℝ) (h : m = -1) : 
  A m ∪ B = {x | -3 ≤ x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_part1_q1_l391_39196


namespace NUMINAMATH_GPT_symmetric_point_yOz_l391_39114

-- Given point A in 3D Cartesian system
def A : ℝ × ℝ × ℝ := (1, -3, 5)

-- Plane yOz where x = 0
def symmetric_yOz (point : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := point
  (-x, y, z)

-- Proof statement (without the actual proof)
theorem symmetric_point_yOz : symmetric_yOz A = (-1, -3, 5) :=
by sorry

end NUMINAMATH_GPT_symmetric_point_yOz_l391_39114


namespace NUMINAMATH_GPT_range_of_function_l391_39139

theorem range_of_function :
  ∀ y : ℝ, ∃ x : ℝ, (x ≤ 1/2) ∧ (y = 2 * x - Real.sqrt (1 - 2 * x)) ↔ y ∈ Set.Iic 1 := 
by
  sorry

end NUMINAMATH_GPT_range_of_function_l391_39139


namespace NUMINAMATH_GPT_multiple_of_fair_tickets_l391_39115

theorem multiple_of_fair_tickets (fair_tickets_sold : ℕ) (game_tickets_sold : ℕ) (h : fair_tickets_sold = game_tickets_sold * x + 6) :
  25 = 56 * x + 6 → x = 19 / 56 := by
  sorry

end NUMINAMATH_GPT_multiple_of_fair_tickets_l391_39115


namespace NUMINAMATH_GPT_average_speed_of_train_l391_39190

-- Definitions based on the conditions
def distance1 : ℝ := 325
def distance2 : ℝ := 470
def time1 : ℝ := 3.5
def time2 : ℝ := 4

-- Proof statement
theorem average_speed_of_train :
  (distance1 + distance2) / (time1 + time2) = 106 := 
by 
  sorry

end NUMINAMATH_GPT_average_speed_of_train_l391_39190


namespace NUMINAMATH_GPT_billy_restaurant_total_payment_l391_39124

noncomputable def cost_of_meal
  (adult_count child_count : ℕ)
  (adult_cost child_cost : ℕ) : ℕ :=
  adult_count * adult_cost + child_count * child_cost

noncomputable def cost_of_dessert
  (total_people : ℕ)
  (dessert_cost : ℕ) : ℕ :=
  total_people * dessert_cost

noncomputable def total_cost_before_discount
  (adult_count child_count : ℕ)
  (adult_cost child_cost dessert_cost : ℕ) : ℕ :=
  (cost_of_meal adult_count child_count adult_cost child_cost) +
  (cost_of_dessert (adult_count + child_count) dessert_cost)

noncomputable def discount_amount
  (total : ℕ)
  (discount_rate : ℝ) : ℝ :=
  total * discount_rate

noncomputable def total_amount_to_pay
  (total : ℕ)
  (discount : ℝ) : ℝ :=
  total - discount

theorem billy_restaurant_total_payment :
  total_amount_to_pay
  (total_cost_before_discount 2 5 7 3 2)
  (discount_amount (total_cost_before_discount 2 5 7 3 2) 0.15) = 36.55 := by
  sorry

end NUMINAMATH_GPT_billy_restaurant_total_payment_l391_39124


namespace NUMINAMATH_GPT_intersection_of_sets_l391_39197

def setA (x : ℝ) : Prop := 2 * x + 1 > 0
def setB (x : ℝ) : Prop := abs (x - 1) < 2

theorem intersection_of_sets :
  {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -1/2 < x ∧ x < 3} :=
by 
  sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_intersection_of_sets_l391_39197


namespace NUMINAMATH_GPT_transformation_result_l391_39144

def f (x y : ℝ) : ℝ × ℝ := (y, x)
def g (x y : ℝ) : ℝ × ℝ := (-x, -y)

theorem transformation_result : g (f (-6) (7)).1 (f (-6) (7)).2 = (-7, 6) :=
by
  sorry

end NUMINAMATH_GPT_transformation_result_l391_39144


namespace NUMINAMATH_GPT_polynomial_strictly_monotonic_l391_39185

variable {P : ℝ → ℝ}

/-- The polynomial P(x) is such that the polynomials P(P(x)) and P(P(P(x))) are strictly monotonic 
on the entire real axis. Prove that P(x) is also strictly monotonic on the entire real axis. -/
theorem polynomial_strictly_monotonic
  (h1 : StrictMono (P ∘ P))
  (h2 : StrictMono (P ∘ P ∘ P)) :
  StrictMono P :=
sorry

end NUMINAMATH_GPT_polynomial_strictly_monotonic_l391_39185


namespace NUMINAMATH_GPT_euler_totient_divisibility_l391_39142

theorem euler_totient_divisibility (a n: ℕ) (h1 : a ≥ 2) : (n ∣ Nat.totient (a^n - 1)) :=
sorry

end NUMINAMATH_GPT_euler_totient_divisibility_l391_39142


namespace NUMINAMATH_GPT_arrow_in_48th_position_l391_39158

def arrow_sequence := ["→", "↔", "↓", "→", "↕"]

theorem arrow_in_48th_position :
  arrow_sequence[48 % arrow_sequence.length] = "↓" :=
by
  sorry

end NUMINAMATH_GPT_arrow_in_48th_position_l391_39158


namespace NUMINAMATH_GPT_quadratic_roots_m_value_l391_39155

noncomputable def quadratic_roots_condition (m : ℝ) (x1 x2 : ℝ) : Prop :=
  (∀ a b c : ℝ, a = 1 ∧ b = 2 * (m + 1) ∧ c = m^2 - 1 → x1^2 + b * x1 + c = 0 ∧ x2^2 + b * x2 + c = 0) ∧ 
  (x1 - x2)^2 = 16 - x1 * x2

theorem quadratic_roots_m_value (m : ℝ) (x1 x2 : ℝ) (h : quadratic_roots_condition m x1 x2) : m = 1 :=
sorry

end NUMINAMATH_GPT_quadratic_roots_m_value_l391_39155


namespace NUMINAMATH_GPT_solve_for_y_l391_39140

variable {y : ℚ}
def algebraic_expression_1 (y : ℚ) : ℚ := 4 * y + 8
def algebraic_expression_2 (y : ℚ) : ℚ := 8 * y - 7

theorem solve_for_y (h : algebraic_expression_1 y = - algebraic_expression_2 y) : y = -1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l391_39140


namespace NUMINAMATH_GPT_total_cost_after_discounts_and_cashback_l391_39130

def iPhone_original_price : ℝ := 800
def iWatch_original_price : ℝ := 300
def iPhone_discount_rate : ℝ := 0.15
def iWatch_discount_rate : ℝ := 0.10
def cashback_rate : ℝ := 0.02

theorem total_cost_after_discounts_and_cashback :
  (iPhone_original_price * (1 - iPhone_discount_rate) + iWatch_original_price * (1 - iWatch_discount_rate)) * (1 - cashback_rate) = 931 :=
by sorry

end NUMINAMATH_GPT_total_cost_after_discounts_and_cashback_l391_39130


namespace NUMINAMATH_GPT_total_amount_contribution_l391_39193

theorem total_amount_contribution : 
  let r := 285
  let s := 35
  let a := 30
  let d := a / 2
  let c := 35
  r + s + a + d + c = 400 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_contribution_l391_39193


namespace NUMINAMATH_GPT_q_work_alone_in_10_days_l391_39151

theorem q_work_alone_in_10_days (p_rate : ℝ) (q_rate : ℝ) (d : ℕ) (h1 : p_rate = 1 / 20)
                                    (h2 : q_rate = 1 / d) (h3 : 2 * (p_rate + q_rate) = 0.3) :
                                    d = 10 :=
by sorry

end NUMINAMATH_GPT_q_work_alone_in_10_days_l391_39151


namespace NUMINAMATH_GPT_max_observing_relations_lemma_l391_39159

/-- There are 24 robots on a plane, each with a 70-degree field of view. -/
def robots : ℕ := 24

/-- Definition of field of view for each robot. -/
def field_of_view : ℝ := 70

/-- Maximum number of observing relations. Observing is a one-sided relation. -/
def max_observing_relations := 468

/-- Theorem: The maximum number of observing relations among 24 robots,
each with a 70-degree field of view, is 468. -/
theorem max_observing_relations_lemma : max_observing_relations = 468 :=
by
  sorry

end NUMINAMATH_GPT_max_observing_relations_lemma_l391_39159


namespace NUMINAMATH_GPT_total_students_l391_39137

noncomputable def total_students_in_gym (F : ℕ) (T : ℕ) : Prop :=
  T = 26

theorem total_students (F T : ℕ) (h1 : 4 = T - F) (h2 : F / (F + 4) = 11 / 13) : total_students_in_gym F T :=
by sorry

end NUMINAMATH_GPT_total_students_l391_39137


namespace NUMINAMATH_GPT_bruce_purchased_mangoes_l391_39195

-- Condition definitions
def cost_of_grapes (k_gra kg_cost_gra : ℕ) : ℕ := k_gra * kg_cost_gra
def amount_spent_on_mangoes (total_paid cost_gra : ℕ) : ℕ := total_paid - cost_gra
def quantity_of_mangoes (total_amt_mangoes rate_per_kg_mangoes : ℕ) : ℕ := total_amt_mangoes / rate_per_kg_mangoes

-- Parameters
variable (k_gra rate_per_kg_gra rate_per_kg_mangoes total_paid : ℕ)
variable (kg_gra_total_amt spent_amt_mangoes_qty : ℕ)

-- Given values
axiom A1 : k_gra = 7
axiom A2 : rate_per_kg_gra = 70
axiom A3 : rate_per_kg_mangoes = 55
axiom A4 : total_paid = 985

-- Calculations based on conditions
axiom H1 : cost_of_grapes k_gra rate_per_kg_gra = kg_gra_total_amt
axiom H2 : amount_spent_on_mangoes total_paid kg_gra_total_amt = spent_amt_mangoes_qty
axiom H3 : quantity_of_mangoes spent_amt_mangoes_qty rate_per_kg_mangoes = 9

-- Proof statement to be proven
theorem bruce_purchased_mangoes : quantity_of_mangoes spent_amt_mangoes_qty rate_per_kg_mangoes = 9 := sorry

end NUMINAMATH_GPT_bruce_purchased_mangoes_l391_39195


namespace NUMINAMATH_GPT_sunday_to_saturday_ratio_l391_39127

theorem sunday_to_saturday_ratio : 
  ∀ (sold_friday sold_saturday sold_sunday total_sold : ℕ),
  sold_friday = 40 →
  sold_saturday = (2 * sold_friday - 10) →
  total_sold = 145 →
  total_sold = sold_friday + sold_saturday + sold_sunday →
  (sold_sunday : ℚ) / (sold_saturday : ℚ) = 1 / 2 :=
by
  intro sold_friday sold_saturday sold_sunday total_sold
  intros h_friday h_saturday h_total h_sum
  sorry

end NUMINAMATH_GPT_sunday_to_saturday_ratio_l391_39127


namespace NUMINAMATH_GPT_problem_proof_l391_39113

variables (a b : ℝ) (n : ℕ)

theorem problem_proof (h1: a > 0) (h2: b > 0) (h3: a + b = 1) (h4: n >= 2) :
  3/2 < 1/(a^n + 1) + 1/(b^n + 1) ∧ 1/(a^n + 1) + 1/(b^n + 1) ≤ (2^(n+1))/(2^n + 1) := sorry

end NUMINAMATH_GPT_problem_proof_l391_39113


namespace NUMINAMATH_GPT_expand_polynomial_l391_39150

theorem expand_polynomial (t : ℝ) :
  (3 * t^3 - 4 * t^2 + 5 * t - 3) * (4 * t^2 - 2 * t + 1) = 12 * t^5 - 22 * t^4 + 31 * t^3 - 26 * t^2 + 11 * t - 3 := by
  sorry

end NUMINAMATH_GPT_expand_polynomial_l391_39150


namespace NUMINAMATH_GPT_chess_group_players_count_l391_39120

theorem chess_group_players_count (n : ℕ)
  (h1 : ∀ (x y : ℕ), x ≠ y → ∃ k, k = 2)
  (h2 : n * (n - 1) / 2 = 45) :
  n = 10 := sorry

end NUMINAMATH_GPT_chess_group_players_count_l391_39120


namespace NUMINAMATH_GPT_interval_probability_l391_39175

theorem interval_probability (x y z : ℝ) (h0x : 0 < x) (hy : y > 0) (hz : z > 0) (hx : x < y) (hy_mul_z_eq_half : y * z = 1 / 2) (hx_eq_third_y : x = 1 / 3 * y) :
  (x / y) = (2 / 3) :=
by
  -- Here you should provide the proof
  sorry

end NUMINAMATH_GPT_interval_probability_l391_39175


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_between_ln20_l391_39136

theorem sum_of_consecutive_integers_between_ln20 : ∃ a b : ℤ, a < b ∧ b = a + 1 ∧ 1 ≤ a ∧ a + 1 ≤ 3 ∧ (a + b = 4) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_between_ln20_l391_39136


namespace NUMINAMATH_GPT_xy_value_l391_39174

theorem xy_value (x y : ℝ) (h : (x - 3)^2 + |y + 2| = 0) : x * y = -6 :=
by {
  sorry
}

end NUMINAMATH_GPT_xy_value_l391_39174


namespace NUMINAMATH_GPT_average_length_is_21_08_l391_39117

def lengths : List ℕ := [20, 21, 22]
def quantities : List ℕ := [23, 64, 32]

def total_length := List.sum (List.zipWith (· * ·) lengths quantities)
def total_quantity := List.sum quantities

def average_length := total_length / total_quantity

theorem average_length_is_21_08 :
  average_length = 2508 / 119 := by
  sorry

end NUMINAMATH_GPT_average_length_is_21_08_l391_39117


namespace NUMINAMATH_GPT_find_number_l391_39145

variables (n : ℝ)

-- Condition: a certain number divided by 14.5 equals 173.
def condition_1 (n : ℝ) : Prop := n / 14.5 = 173

-- Condition: 29.94 ÷ 1.45 = 17.3.
def condition_2 : Prop := 29.94 / 1.45 = 17.3

-- Theorem: Prove that the number is 2508.5 given the conditions.
theorem find_number (h1 : condition_1 n) (h2 : condition_2) : n = 2508.5 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_l391_39145


namespace NUMINAMATH_GPT_election_votes_l391_39164

theorem election_votes (T V : ℕ) 
    (hT : 8 * T = 11 * 20000) 
    (h_total_votes : T = 2500 + V + 20000) :
    V = 5000 :=
by
    sorry

end NUMINAMATH_GPT_election_votes_l391_39164


namespace NUMINAMATH_GPT_range_of_m_l391_39131

-- Define the function and its properties
variable {f : ℝ → ℝ}
variable (increasing : ∀ x1 x2 : ℝ, x1 > x2 → f x1 > f x2)

theorem range_of_m (h: ∀ m : ℝ, f (2 * m) > f (-m + 9)) : 
  ∀ m : ℝ, m > 3 ↔ f (2 * m) > f (-m + 9) :=
by
  intros
  sorry

end NUMINAMATH_GPT_range_of_m_l391_39131


namespace NUMINAMATH_GPT_move_right_by_three_units_l391_39135

theorem move_right_by_three_units :
  (-1 + 3 = 2) :=
  by { sorry }

end NUMINAMATH_GPT_move_right_by_three_units_l391_39135


namespace NUMINAMATH_GPT_toothpaste_amount_in_tube_l391_39141

def dad_usage_per_brush : ℕ := 3
def mom_usage_per_brush : ℕ := 2
def kid_usage_per_brush : ℕ := 1
def brushes_per_day : ℕ := 3
def days : ℕ := 5

theorem toothpaste_amount_in_tube (dad_usage_per_brush mom_usage_per_brush kid_usage_per_brush brushes_per_day days : ℕ) : 
  dad_usage_per_brush * brushes_per_day * days + 
  mom_usage_per_brush * brushes_per_day * days + 
  (kid_usage_per_brush * brushes_per_day * days * 2) = 105 := 
  by sorry

end NUMINAMATH_GPT_toothpaste_amount_in_tube_l391_39141


namespace NUMINAMATH_GPT_perpendicular_vectors_x_value_l391_39165

theorem perpendicular_vectors_x_value :
  let a := (4, 2)
  let b := (x, 3)
  a.1 * b.1 + a.2 * b.2 = 0 -> x = -3/2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_x_value_l391_39165


namespace NUMINAMATH_GPT_total_texts_sent_l391_39173

theorem total_texts_sent (grocery_texts : ℕ) (response_texts_ratio : ℕ) (police_texts_percentage : ℚ) :
  grocery_texts = 5 →
  response_texts_ratio = 5 →
  police_texts_percentage = 0.10 →
  let response_texts := grocery_texts * response_texts_ratio
  let previous_texts := response_texts + grocery_texts
  let police_texts := previous_texts * police_texts_percentage
  response_texts + grocery_texts + police_texts = 33 :=
by
  sorry

end NUMINAMATH_GPT_total_texts_sent_l391_39173


namespace NUMINAMATH_GPT_percent_of_employed_females_l391_39143

theorem percent_of_employed_females (p e m f : ℝ) (h1 : e = 0.60 * p) (h2 : m = 0.15 * p) (h3 : f = e - m):
  (f / e) * 100 = 75 :=
by
  -- We place the proof here
  sorry

end NUMINAMATH_GPT_percent_of_employed_females_l391_39143


namespace NUMINAMATH_GPT_part_I_part_II_l391_39147

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + ((2 * a^2) / x) + x

theorem part_I (a : ℝ) (h : a > 0) : 
  (∃ x : ℝ, x = 1 ∧ deriv (f a) x = -2) → a = 3 / 2 :=
sorry

theorem part_II (a : ℝ) (h : a = 3 / 2) : 
  (∀ x : ℝ, 0 < x ∧ x < 3 / 2 → deriv (f a) x < 0) ∧ 
  (∀ x : ℝ, x > 3 / 2 → deriv (f a) x > 0) :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l391_39147


namespace NUMINAMATH_GPT_max_profit_at_l391_39179

variables (k x : ℝ) (hk : k > 0)

-- Define the quantities based on problem conditions
def profit (k x : ℝ) : ℝ :=
  0.072 * k * x ^ 2 - k * x ^ 3

-- State the theorem
theorem max_profit_at (k : ℝ) (hk : k > 0) : 
  ∃ x, profit k x = 0.072 * k * x ^ 2 - k * x ^ 3 ∧ x = 0.048 :=
sorry

end NUMINAMATH_GPT_max_profit_at_l391_39179


namespace NUMINAMATH_GPT_trapezoid_height_l391_39178

theorem trapezoid_height (a b : ℝ) (A : ℝ) (h : ℝ) : a = 5 → b = 9 → A = 56 → A = (1 / 2) * (a + b) * h → h = 8 :=
by 
  intros ha hb hA eqn
  sorry

end NUMINAMATH_GPT_trapezoid_height_l391_39178


namespace NUMINAMATH_GPT_LindaCandiesLeft_l391_39116

variable (initialCandies : ℝ)
variable (candiesGiven : ℝ)

theorem LindaCandiesLeft (h1 : initialCandies = 34.0) (h2 : candiesGiven = 28.0) : initialCandies - candiesGiven = 6.0 := by
  sorry

end NUMINAMATH_GPT_LindaCandiesLeft_l391_39116


namespace NUMINAMATH_GPT_lauri_ate_days_l391_39102

theorem lauri_ate_days
    (simone_rate : ℚ)
    (simone_days : ℕ)
    (lauri_rate : ℚ)
    (total_apples : ℚ)
    (simone_apples : ℚ)
    (lauri_apples : ℚ)
    (lauri_days : ℚ) :
  simone_rate = 1/2 → 
  simone_days = 16 →
  lauri_rate = 1/3 →
  total_apples = 13 →
  simone_apples = simone_rate * simone_days →
  lauri_apples = total_apples - simone_apples →
  lauri_days = lauri_apples / lauri_rate →
  lauri_days = 15 :=
by
  intros
  sorry

end NUMINAMATH_GPT_lauri_ate_days_l391_39102


namespace NUMINAMATH_GPT_part_a_part_b_l391_39157

variable (f : ℝ → ℝ)

-- Part (a)
theorem part_a (h : ∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) :
  ∀ x : ℝ, f (f x) ≤ 0 :=
sorry

-- Part (b)
theorem part_b (h : ∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) (h₀ : f 0 ≥ 0) :
  ∀ x : ℝ, f x = 0 :=
sorry

end NUMINAMATH_GPT_part_a_part_b_l391_39157


namespace NUMINAMATH_GPT_rabbit_stashed_nuts_l391_39181

theorem rabbit_stashed_nuts :
  ∃ r: ℕ, 
  ∃ f: ℕ, 
  4 * r = 6 * f ∧ f = r - 5 ∧ 4 * r = 60 :=
by {
  sorry
}

end NUMINAMATH_GPT_rabbit_stashed_nuts_l391_39181


namespace NUMINAMATH_GPT_infinite_solutions_iff_a_eq_neg12_l391_39169

theorem infinite_solutions_iff_a_eq_neg12 {a : ℝ} : 
  (∀ x : ℝ, 4 * (3 * x - a) = 3 * (4 * x + 16)) ↔ a = -12 :=
by 
  sorry

end NUMINAMATH_GPT_infinite_solutions_iff_a_eq_neg12_l391_39169


namespace NUMINAMATH_GPT_bead_necklaces_count_l391_39184

-- Define the conditions
def cost_per_necklace : ℕ := 9
def gemstone_necklaces_sold : ℕ := 3
def total_earnings : ℕ := 90

-- Define the total earnings from gemstone necklaces
def earnings_from_gemstone_necklaces : ℕ := gemstone_necklaces_sold * cost_per_necklace

-- Define the total earnings from bead necklaces
def earnings_from_bead_necklaces : ℕ := total_earnings - earnings_from_gemstone_necklaces

-- Define the number of bead necklaces sold
def bead_necklaces_sold : ℕ := earnings_from_bead_necklaces / cost_per_necklace

-- The statement to be proved
theorem bead_necklaces_count : bead_necklaces_sold = 7 := by
  sorry

end NUMINAMATH_GPT_bead_necklaces_count_l391_39184


namespace NUMINAMATH_GPT_math_class_problem_l391_39187

theorem math_class_problem
  (x a : ℝ)
  (h_mistaken : (2 * (2 * 4 - 1) + 1 = 5 * (4 + a)))
  (h_original : (2 * x - 1) / 5 + 1 = (x + a) / 2)
  : a = -1 ∧ x = 13 := by
  sorry

end NUMINAMATH_GPT_math_class_problem_l391_39187


namespace NUMINAMATH_GPT_proof_problem_l391_39153

def U : Set ℤ := {x | x^2 - x - 12 ≤ 0}
def A : Set ℤ := {-2, -1, 3}
def B : Set ℤ := {0, 1, 3, 4}

theorem proof_problem : (U \ A) ∩ B = {0, 1, 4} := 
by sorry

end NUMINAMATH_GPT_proof_problem_l391_39153


namespace NUMINAMATH_GPT_smallest_AAB_value_l391_39109

theorem smallest_AAB_value : ∃ (A B : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9 ∧ 110 * A + B = 8 * (10 * A + B) ∧ ¬ (A = B) ∧ 110 * A + B = 773 :=
by sorry

end NUMINAMATH_GPT_smallest_AAB_value_l391_39109


namespace NUMINAMATH_GPT_percentage_employees_at_picnic_l391_39172

theorem percentage_employees_at_picnic (total_employees men_attend men_percentage women_attend women_percentage : ℝ)
  (h1 : men_attend = 0.20 * (men_percentage * total_employees))
  (h2 : women_attend = 0.40 * ((1 - men_percentage) * total_employees))
  (h3 : men_percentage = 0.30)
  : ((men_attend + women_attend) / total_employees) * 100 = 34 := by
sorry

end NUMINAMATH_GPT_percentage_employees_at_picnic_l391_39172


namespace NUMINAMATH_GPT_range_of_m_l391_39106

theorem range_of_m (x y m : ℝ) : (∃ (x y : ℝ), x + y^2 - x + y + m = 0) → m < 1/2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l391_39106


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l391_39132

variables (x y m : ℤ)

-- Given the system of equations
def system_of_equations (x y m : ℤ) : Prop :=
  (2 * x - y = m) ∧ (3 * x + 2 * y = m + 7)

-- Part (1) m = 0, find x = 1, y = 2
theorem part1_solution : system_of_equations x y 0 → x = 1 ∧ y = 2 :=
sorry

-- Part (2) point A(-2,3) in the second quadrant with distances 3 and 2, find m = -7
def is_in_second_quadrant (x y : ℤ) : Prop :=
  x < 0 ∧ y > 0

def distance_to_axes (x y dx dy : ℤ) : Prop :=
  y = dy ∧ x = -dx

theorem part2_solution : is_in_second_quadrant x y →
  distance_to_axes x y 2 3 →
  system_of_equations x y m →
  m = -7 :=
sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l391_39132


namespace NUMINAMATH_GPT_sum_of_roots_eq_4140_l391_39167

open Complex

noncomputable def sum_of_roots : ℝ :=
  let θ0 := 270 / 5;
  let θ1 := (270 + 360) / 5;
  let θ2 := (270 + 2 * 360) / 5;
  let θ3 := (270 + 3 * 360) / 5;
  let θ4 := (270 + 4 * 360) / 5;
  θ0 + θ1 + θ2 + θ3 + θ4

theorem sum_of_roots_eq_4140 : sum_of_roots = 4140 := by
  sorry

end NUMINAMATH_GPT_sum_of_roots_eq_4140_l391_39167


namespace NUMINAMATH_GPT_proof_emails_in_morning_l391_39189

def emailsInAfternoon : ℕ := 2

def emailsMoreInMorning : ℕ := 4

def emailsInMorning : ℕ := 6

theorem proof_emails_in_morning
  (a : ℕ) (h1 : a = emailsInAfternoon)
  (m : ℕ) (h2 : m = emailsMoreInMorning)
  : emailsInMorning = a + m := by
  sorry

end NUMINAMATH_GPT_proof_emails_in_morning_l391_39189


namespace NUMINAMATH_GPT_determine_ABCC_l391_39194

theorem determine_ABCC :
  ∃ (A B C D E : ℕ), 
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ 
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ 
    C ≠ D ∧ C ≠ E ∧ 
    D ≠ E ∧ 
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧
    1000 * A + 100 * B + 11 * C = (11 * D - E) * 100 + 11 * D * E ∧ 
    1000 * A + 100 * B + 11 * C = 1966 :=
sorry

end NUMINAMATH_GPT_determine_ABCC_l391_39194


namespace NUMINAMATH_GPT_train_speed_l391_39133

theorem train_speed (length_m : ℕ) (time_s : ℕ) (length_km : ℝ) (time_hr : ℝ) 
(length_conversion : length_km = (length_m : ℝ) / 1000)
(time_conversion : time_hr = (time_s : ℝ) / 3600)
(speed : ℝ) (speed_formula : speed = length_km / time_hr) :
  length_m = 300 → time_s = 18 → speed = 60 :=
by
  intros h1 h2
  rw [h1, h2] at *
  simp [length_conversion, time_conversion, speed_formula]
  norm_num
  sorry

end NUMINAMATH_GPT_train_speed_l391_39133


namespace NUMINAMATH_GPT_contrapositive_squared_l391_39156

theorem contrapositive_squared (a : ℝ) : (a ≤ 0 → a^2 ≤ 0) ↔ (a > 0 → a^2 > 0) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_squared_l391_39156


namespace NUMINAMATH_GPT_tournament_participants_l391_39104

theorem tournament_participants (n : ℕ) (h : (n * (n - 1)) / 2 = 171) : n = 19 :=
by
  sorry

end NUMINAMATH_GPT_tournament_participants_l391_39104


namespace NUMINAMATH_GPT_green_paint_amount_l391_39148

theorem green_paint_amount (T W B : ℕ) (hT : T = 69) (hW : W = 20) (hB : B = 34) : 
  T - (W + B) = 15 := 
by
  sorry

end NUMINAMATH_GPT_green_paint_amount_l391_39148


namespace NUMINAMATH_GPT_parameter_range_exists_solution_l391_39160

theorem parameter_range_exists_solution :
  {a : ℝ | ∃ b : ℝ, ∃ x y : ℝ,
    x^2 + y^2 + 2 * a * (a + y - x) = 49 ∧
    y = 15 * Real.cos (x - b) - 8 * Real.sin (x - b)
  } = {a : ℝ | -24 ≤ a ∧ a ≤ 24} :=
sorry

end NUMINAMATH_GPT_parameter_range_exists_solution_l391_39160


namespace NUMINAMATH_GPT_length_of_each_stone_l391_39100

theorem length_of_each_stone {L : ℝ} (hall_length hall_breadth : ℝ) (stone_breadth : ℝ) (num_stones : ℕ) (area_hall : ℝ) (area_stone : ℝ) :
  hall_length = 36 * 10 ∧ hall_breadth = 15 * 10 ∧ stone_breadth = 5 ∧ num_stones = 3600 ∧
  area_hall = hall_length * hall_breadth ∧ area_stone = L * stone_breadth ∧
  area_stone * num_stones = area_hall →
  L = 3 :=
by
  sorry

end NUMINAMATH_GPT_length_of_each_stone_l391_39100


namespace NUMINAMATH_GPT_greatest_perimeter_of_triangle_l391_39166

theorem greatest_perimeter_of_triangle :
  ∃ (x : ℕ), (3 * x) + 15 = 57 ∧ 
  (x > 5 ∧ x < 15) ∧ 
  2 * x + x > 15 ∧ 
  x + 15 > 2 * x ∧ 
  2 * x + 15 > x := 
sorry

end NUMINAMATH_GPT_greatest_perimeter_of_triangle_l391_39166


namespace NUMINAMATH_GPT_fixed_point_of_function_l391_39105

-- Definition: The function passes through a fixed point (a, b) for all real numbers k.
def passes_through_fixed_point (f : ℝ → ℝ) (a b : ℝ) := ∀ k : ℝ, f a = b

-- Given the function y = 9x^2 + 3kx - 6k, we aim to prove the fixed point is (2, 36).
theorem fixed_point_of_function : passes_through_fixed_point (fun x => 9 * x^2 + 3 * k * x - 6 * k) 2 36 := by
  sorry

end NUMINAMATH_GPT_fixed_point_of_function_l391_39105


namespace NUMINAMATH_GPT_unique_x_inequality_l391_39146

theorem unique_x_inequality (a : ℝ) : (∀ x : ℝ, |x^2 + 2*a*x + 3*a| ≤ 2 → (a = 1 ∨ a = 2)) :=
by
  sorry

end NUMINAMATH_GPT_unique_x_inequality_l391_39146


namespace NUMINAMATH_GPT_no_integer_solutions_x_x_plus_1_eq_13y_plus_1_l391_39125

theorem no_integer_solutions_x_x_plus_1_eq_13y_plus_1 :
  ¬ ∃ x y : ℤ, x * (x + 1) = 13 * y + 1 :=
by sorry

end NUMINAMATH_GPT_no_integer_solutions_x_x_plus_1_eq_13y_plus_1_l391_39125


namespace NUMINAMATH_GPT_eccentricity_of_ellipse_equation_of_ellipse_l391_39123

variable {a b : ℝ}
variable {x y : ℝ}

/-- Problem 1: Eccentricity of the given ellipse --/
theorem eccentricity_of_ellipse (ha : a = 2 * b) (hb0 : 0 < b) :
  ∃ e : ℝ, e = Real.sqrt (1 - (b / a) ^ 2) ∧ e = Real.sqrt 3 / 2 := by
  sorry

/-- Problem 2: Equation of the ellipse with respect to maximizing the area of triangle OMN --/
theorem equation_of_ellipse (ha : a = 2 * b) (hb0 : 0 < b) :
  ∃ l : ℝ → ℝ, (∃ k : ℝ, ∀ x, l x = k * x + 2) →
  ∀ x y : ℝ, (x^2 / (a^2) + y^2 / (b^2) = 1) →
  (∀ x' y' : ℝ, (x'^2 + 4 * y'^2 = 4 * b^2) ∧ y' = k * x' + 2) →
  (∃ a b : ℝ, a = 8 ∧ b = 2 ∧ x^2 / a + y^2 / b = 1) := by
  sorry

end NUMINAMATH_GPT_eccentricity_of_ellipse_equation_of_ellipse_l391_39123


namespace NUMINAMATH_GPT_square_area_ratio_l391_39183

theorem square_area_ratio (a b : ℕ) (h : 4 * a = 4 * (4 * b)) : (a^2) = 16 * (b^2) := 
by sorry

end NUMINAMATH_GPT_square_area_ratio_l391_39183


namespace NUMINAMATH_GPT_four_digit_numbers_div_by_5_with_34_end_l391_39168

theorem four_digit_numbers_div_by_5_with_34_end : 
  ∃ (count : ℕ), count = 90 ∧
  ∀ (n : ℕ), (1000 ≤ n ∧ n < 10000) →
  (n % 100 = 34) →
  ((10 ∣ n) ∨ (5 ∣ n)) →
  (count = 90) :=
sorry

end NUMINAMATH_GPT_four_digit_numbers_div_by_5_with_34_end_l391_39168


namespace NUMINAMATH_GPT_absolute_value_sum_l391_39112

theorem absolute_value_sum (a b : ℤ) (h_a : |a| = 5) (h_b : |b| = 3) : 
  (a + b = 8) ∨ (a + b = 2) ∨ (a + b = -2) ∨ (a + b = -8) :=
by
  sorry

end NUMINAMATH_GPT_absolute_value_sum_l391_39112


namespace NUMINAMATH_GPT_difference_of_squares_65_35_l391_39171

theorem difference_of_squares_65_35 :
  let a := 65
  let b := 35
  a^2 - b^2 = (a + b) * (a - b) :=
by
  sorry

end NUMINAMATH_GPT_difference_of_squares_65_35_l391_39171


namespace NUMINAMATH_GPT_gain_in_transaction_per_year_l391_39162

noncomputable def borrowing_interest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal * rate * time

noncomputable def lending_interest (principal : ℕ) (rate : ℚ) (time : ℕ) : ℚ :=
  principal * rate * time

noncomputable def gain_per_year (borrow_principal : ℕ) (borrow_rate : ℚ) 
  (borrow_time : ℕ) (lend_principal : ℕ) (lend_rate : ℚ) (lend_time : ℕ) : ℚ :=
  (lending_interest lend_principal lend_rate lend_time - borrowing_interest borrow_principal borrow_rate borrow_time) / borrow_time

theorem gain_in_transaction_per_year :
  gain_per_year 4000 (4 / 100) 2 4000 (6 / 100) 2 = 80 := 
sorry

end NUMINAMATH_GPT_gain_in_transaction_per_year_l391_39162


namespace NUMINAMATH_GPT_sequence_sum_l391_39111

-- Defining the sequence terms
variables (J K L M N O P Q R S : ℤ)
-- Condition N = 7
def N_value : Prop := N = 7
-- Condition sum of any four consecutive terms is 40
def sum_of_consecutive : Prop := 
  J + K + L + M = 40 ∧
  K + L + M + N = 40 ∧
  L + M + N + O = 40 ∧
  M + N + O + P = 40 ∧
  N + O + P + Q = 40 ∧
  O + P + Q + R = 40 ∧
  P + Q + R + S = 40

-- The main theorem stating J + S = 40 given the conditions
theorem sequence_sum (N_value : N = 7) (sum_of_consecutive : 
  J + K + L + M = 40 ∧
  K + L + M + N = 40 ∧
  L + M + N + O = 40 ∧
  M + N + O + P = 40 ∧
  N + O + P + Q = 40 ∧
  O + P + Q + R = 40 ∧
  P + Q + R + S = 40) : 
  J + S = 40 := sorry

end NUMINAMATH_GPT_sequence_sum_l391_39111


namespace NUMINAMATH_GPT_relationship_between_a_b_c_l391_39154

noncomputable def a : ℝ := (1 / 2)^(1 / 3)
noncomputable def b : ℝ := Real.log 2 / Real.log (1 / 3)
noncomputable def c : ℝ := Real.log 3 / Real.log (1 / 2)

theorem relationship_between_a_b_c : c < b ∧ b < a :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_a_b_c_l391_39154


namespace NUMINAMATH_GPT_complement_of_P_in_U_l391_39126

def U : Set ℤ := {-1, 0, 1, 2}
def P : Set ℤ := {x | -Real.sqrt 2 < x ∧ x < Real.sqrt 2}
def compl_U (P : Set ℤ) : Set ℤ := {x ∈ U | x ∉ P}

theorem complement_of_P_in_U : compl_U P = {2} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_P_in_U_l391_39126


namespace NUMINAMATH_GPT_markup_percentage_l391_39191

-- Define the wholesale cost
def wholesale_cost : ℝ := sorry

-- Define the retail cost
def retail_cost : ℝ := sorry

-- Condition given in the problem: selling at 60% discount nets a 20% profit
def discount_condition (W R : ℝ) : Prop :=
  0.40 * R = 1.20 * W

-- We need to prove the markup percentage is 200%
theorem markup_percentage (W R : ℝ) (h : discount_condition W R) : 
  ((R - W) / W) * 100 = 200 :=
by sorry

end NUMINAMATH_GPT_markup_percentage_l391_39191


namespace NUMINAMATH_GPT_sum_of_values_of_M_l391_39176

theorem sum_of_values_of_M (M : ℝ) (h : M * (M - 8) = 12) :
  (∃ M1 M2 : ℝ, M^2 - 8 * M - 12 = 0 ∧ M1 + M2 = 8) :=
sorry

end NUMINAMATH_GPT_sum_of_values_of_M_l391_39176


namespace NUMINAMATH_GPT_ratio_of_original_to_doubled_l391_39163

theorem ratio_of_original_to_doubled (x : ℕ) (h : x + 5 = 17) : (x / Nat.gcd x (2 * x)) = 1 ∧ ((2 * x) / Nat.gcd x (2 * x)) = 2 := 
by
  sorry

end NUMINAMATH_GPT_ratio_of_original_to_doubled_l391_39163


namespace NUMINAMATH_GPT_fraction_equation_correct_l391_39101

theorem fraction_equation_correct : (1 / 2 - 1 / 6) / (1 / 6009) = 2003 := by
  sorry

end NUMINAMATH_GPT_fraction_equation_correct_l391_39101


namespace NUMINAMATH_GPT_correct_total_cost_l391_39198

-- Number of sandwiches and their cost
def num_sandwiches : ℕ := 7
def sandwich_cost : ℕ := 4

-- Number of sodas and their cost
def num_sodas : ℕ := 9
def soda_cost : ℕ := 3

-- Total cost calculation
def total_cost : ℕ := num_sandwiches * sandwich_cost + num_sodas * soda_cost

theorem correct_total_cost : total_cost = 55 := by
  -- skip the proof details
  sorry

end NUMINAMATH_GPT_correct_total_cost_l391_39198


namespace NUMINAMATH_GPT_pool_filling_time_l391_39186

theorem pool_filling_time :
  (∀ t : ℕ, t >= 6 → ∃ v : ℝ, v = (2^(t-6)) * 0.25) →
  ∃ t : ℕ, t = 8 :=
by
  intros h
  existsi 8
  sorry

end NUMINAMATH_GPT_pool_filling_time_l391_39186


namespace NUMINAMATH_GPT_solution_set_condition_l391_39129

-- The assumptions based on the given conditions
variables (a b : ℝ)

noncomputable def inequality_system_solution_set (x : ℝ) : Prop :=
  (x + 2 * a > 4) ∧ (2 * x - b < 5)

theorem solution_set_condition (a b : ℝ) :
  (∀ x : ℝ, inequality_system_solution_set a b x ↔ 0 < x ∧ x < 2) →
  (a + b) ^ 2023 = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_solution_set_condition_l391_39129


namespace NUMINAMATH_GPT_unique_B_squared_l391_39138

variable (B : Matrix (Fin 2) (Fin 2) ℝ)

theorem unique_B_squared (h : B ^ 4 = 0) :
  ∃! B2 : Matrix (Fin 2) (Fin 2) ℝ, B ^ 2 = B2 :=
by sorry

end NUMINAMATH_GPT_unique_B_squared_l391_39138
