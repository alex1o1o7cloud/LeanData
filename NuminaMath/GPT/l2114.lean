import Mathlib

namespace solve_system_of_equations_l2114_211441

theorem solve_system_of_equations (x y : ℝ) (h1 : 3 * x - 2 * y = 1) (h2 : x + y = 2) : x^2 - 2 * y^2 = -1 :=
by
  sorry

end solve_system_of_equations_l2114_211441


namespace find_number_l2114_211439

theorem find_number (x : ℤ) (h : 3 * x - 6 = 2 * x) : x = 6 :=
by
  sorry

end find_number_l2114_211439


namespace lcm_is_2310_l2114_211428

def a : ℕ := 210
def b : ℕ := 605
def hcf : ℕ := 55

theorem lcm_is_2310 (lcm : ℕ) : Nat.lcm a b = 2310 :=
by 
  have h : a * b = lcm * hcf := by sorry
  sorry

end lcm_is_2310_l2114_211428


namespace num_persons_initially_l2114_211432

theorem num_persons_initially (N : ℕ) (avg_weight : ℝ) 
  (h_increase_avg : avg_weight + 5 = avg_weight + 40 / N) :
  N = 8 := by
    sorry

end num_persons_initially_l2114_211432


namespace initial_numbers_count_l2114_211466

theorem initial_numbers_count (n : ℕ) (S : ℝ)
  (h1 : S / n = 56)
  (h2 : (S - 100) / (n - 2) = 56.25) :
  n = 50 :=
sorry

end initial_numbers_count_l2114_211466


namespace expense_of_three_yuan_l2114_211454

def isIncome (x : Int) : Prop := x > 0
def isExpense (x : Int) : Prop := x < 0
def incomeOfTwoYuan : Int := 2

theorem expense_of_three_yuan : isExpense (-3) :=
by
  -- Assuming the conditions:
  -- Income is positive: isIncome incomeOfTwoYuan (which is 2)
  -- Expenses are negative
  -- Expenses of 3 yuan should be denoted as -3 yuan
  sorry

end expense_of_three_yuan_l2114_211454


namespace ratio_of_adults_to_children_l2114_211468

-- Definitions based on conditions
def adult_ticket_price : ℝ := 5.50
def child_ticket_price : ℝ := 2.50
def total_receipts : ℝ := 1026
def number_of_adults : ℝ := 152

-- Main theorem to prove ratio of adults to children is 2:1
theorem ratio_of_adults_to_children : 
  ∃ (number_of_children : ℝ), adult_ticket_price * number_of_adults + child_ticket_price * number_of_children = total_receipts ∧ 
  number_of_adults / number_of_children = 2 :=
by
  sorry

end ratio_of_adults_to_children_l2114_211468


namespace real_solutions_count_l2114_211440

theorem real_solutions_count : 
  ∃ (n : ℕ), n = 2 ∧ ∀ (x : ℝ), (2 : ℝ) ^ (3 * x ^ 2 - 8 * x + 4) = 1 → x = 2 ∨ x = 2 / 3 :=
by
  sorry

end real_solutions_count_l2114_211440


namespace find_a3_l2114_211416

-- Given conditions
def sequence_sum (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = n^2 + n

-- Define the sequence term calculation from the sum function.
def seq_term (S : ℕ → ℕ) (n : ℕ) : ℕ :=
  S n - S (n - 1)

theorem find_a3 (S : ℕ → ℕ) (h : sequence_sum S) :
  seq_term S 3 = 6 :=
by
  sorry

end find_a3_l2114_211416


namespace right_triangle_area_l2114_211495

variable {AB BC AC : ℕ}

theorem right_triangle_area : ∀ (AB BC AC : ℕ), (AC = 50) → (AB + BC = 70) → (AB^2 + BC^2 = AC^2) → (1 / 2) * AB * BC = 300 :=
by
  intros AB BC AC h1 h2 h3
  -- Proof steps will be added here
  sorry

end right_triangle_area_l2114_211495


namespace total_amount_l2114_211452

def shares (a b c : ℕ) : Prop :=
  b = 1800 ∧ 2 * b = 3 * a ∧ 3 * c = 4 * b

theorem total_amount (a b c : ℕ) (h : shares a b c) : a + b + c = 5400 :=
by
  have h₁ : 2 * b = 3 * a := h.2.1
  have h₂ : 3 * c = 4 * b := h.2.2
  have hb : b = 1800 := h.1
  sorry

end total_amount_l2114_211452


namespace quadratic_expression_value_l2114_211417

theorem quadratic_expression_value (a : ℝ)
  (h1 : ∃ x₁ x₂ : ℝ, x₁^2 + 2 * (a - 1) * x₁ + a^2 - 7 * a - 4 = 0 ∧ x₂^2 + 2 * (a - 1) * x₂ + a^2 - 7 * a - 4 = 0)
  (h2 : ∀ x₁ x₂ : ℝ, x₁ * x₂ - 3 * x₁ - 3 * x₂ - 2 = 0) :
  (1 + 4 / (a^2 - 4)) * (a + 2) / a = 2 := 
sorry

end quadratic_expression_value_l2114_211417


namespace half_radius_of_circle_y_l2114_211408

theorem half_radius_of_circle_y 
  (r_x r_y : ℝ) 
  (h₁ : π * r_x^2 = π * r_y^2) 
  (h₂ : 2 * π * r_x = 14 * π) :
  r_y / 2 = 3.5 :=
by {
  sorry
}

end half_radius_of_circle_y_l2114_211408


namespace inscribed_circle_radius_inequality_l2114_211471

open Real

variables (ABC ABD BDC : Type) -- Representing the triangles

noncomputable def r (ABC : Type) : ℝ := sorry -- radius of the inscribed circle in ABC
noncomputable def r1 (ABD : Type) : ℝ := sorry -- radius of the inscribed circle in ABD
noncomputable def r2 (BDC : Type) : ℝ := sorry -- radius of the inscribed circle in BDC

noncomputable def p (ABC : Type) : ℝ := sorry -- semiperimeter of ABC
noncomputable def p1 (ABD : Type) : ℝ := sorry -- semiperimeter of ABD
noncomputable def p2 (BDC : Type) : ℝ := sorry -- semiperimeter of BDC

noncomputable def S (ABC : Type) : ℝ := sorry -- area of ABC
noncomputable def S1 (ABD : Type) : ℝ := sorry -- area of ABD
noncomputable def S2 (BDC : Type) : ℝ := sorry -- area of BDC

lemma triangle_area_sum (ABC ABD BDC : Type) :
  S ABC = S1 ABD + S2 BDC := sorry

lemma semiperimeter_area_relation (ABC ABD BDC : Type) :
  S ABC = p ABC * r ABC ∧
  S1 ABD = p1 ABD * r1 ABD ∧
  S2 BDC = p2 BDC * r2 BDC := sorry

theorem inscribed_circle_radius_inequality (ABC ABD BDC : Type) :
  r1 ABD + r2 BDC > r ABC := sorry

end inscribed_circle_radius_inequality_l2114_211471


namespace range_of_a_l2114_211467

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) ↔ (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_l2114_211467


namespace velocity_at_t1_l2114_211451

-- Define the motion equation
def s (t : ℝ) : ℝ := -t^2 + 2 * t

-- Define the velocity function as the derivative of s
def velocity (t : ℝ) : ℝ := -2 * t + 2

-- Prove that the velocity at t = 1 is 0
theorem velocity_at_t1 : velocity 1 = 0 :=
by
  -- Apply the definition of velocity
    sorry

end velocity_at_t1_l2114_211451


namespace f1_neither_even_nor_odd_f2_min_value_l2114_211401

noncomputable def f1 (x : ℝ) : ℝ :=
  x^2 + abs (x - 2) - 1

theorem f1_neither_even_nor_odd : ¬(∀ x : ℝ, f1 x = f1 (-x)) ∧ ¬(∀ x : ℝ, f1 x = -f1 (-x)) :=
sorry

noncomputable def f2 (x a : ℝ) : ℝ :=
  x^2 + abs (x - a) + 1

theorem f2_min_value (a : ℝ) :
  (if a < -1/2 then (∃ x, f2 x a = 3/4 - a)
  else if -1/2 ≤ a ∧ a ≤ 1/2 then (∃ x, f2 x a = a^2 + 1)
  else (∃ x, f2 x a = 3/4 + a)) :=
sorry

end f1_neither_even_nor_odd_f2_min_value_l2114_211401


namespace ratio_of_shares_l2114_211464

-- Definitions
variable (A B C : ℝ)   -- Representing the shares of a, b, and c
variable (x : ℝ)       -- Fraction

-- Conditions
axiom h1 : A = 80
axiom h2 : A + B + C = 200
axiom h3 : A = x * (B + C)
axiom h4 : B = (6 / 9) * (A + C)

-- Statement to prove
theorem ratio_of_shares : A / (B + C) = 2 / 3 :=
by sorry

end ratio_of_shares_l2114_211464


namespace factor_polynomial_l2114_211460

-- Statement of the proof problem
theorem factor_polynomial (x y z : ℝ) :
    x * (y - z)^4 + y * (z - x)^4 + z * (x - y)^4 =
    (x - y) * (y - z) * (z - x) * (-(x - y)^2 - (y - z)^2 - (z - x)^2) :=
by
  sorry

end factor_polynomial_l2114_211460


namespace men_entered_count_l2114_211413

variable (M W x : ℕ)

noncomputable def initial_ratio : Prop := M = 4 * W / 5
noncomputable def men_entered : Prop := M + x = 14
noncomputable def women_double : Prop := 2 * (W - 3) = 14

theorem men_entered_count (M W x : ℕ) (h1 : initial_ratio M W) (h2 : men_entered M x) (h3 : women_double W) : x = 6 := by
  sorry

end men_entered_count_l2114_211413


namespace mother_hubbard_children_l2114_211400

theorem mother_hubbard_children :
  (∃ c : ℕ, (2 / 3 : ℚ) = c * (1 / 12 : ℚ)) → c = 8 :=
by
  sorry

end mother_hubbard_children_l2114_211400


namespace statement_C_correct_l2114_211430

theorem statement_C_correct (a b c d : ℝ) (h_ab : a > b) (h_cd : c > d) : a + c > b + d :=
by
  sorry

end statement_C_correct_l2114_211430


namespace first_pipe_fill_time_l2114_211463

theorem first_pipe_fill_time 
  (T : ℝ)
  (h1 : 48 * (1 / T - 1 / 24) + 18 * (1 / T) = 1) :
  T = 22 :=
by
  sorry

end first_pipe_fill_time_l2114_211463


namespace polynomial_transformation_l2114_211494

noncomputable def p : ℝ → ℝ := sorry

variable (k : ℕ)

axiom ax1 (x : ℝ) : p (2 * x) = 2^(k - 1) * (p x + p (x + 1/2))

theorem polynomial_transformation (k : ℕ) (p : ℝ → ℝ)
  (h_p : ∀ x : ℝ, p (2 * x) = 2^(k - 1) * (p x + p (x + 1/2))) :
  ∀ x : ℝ, p (3 * x) = 3^(k - 1) * (p x + p (x + 1/3) + p (x + 2/3)) := sorry

end polynomial_transformation_l2114_211494


namespace combined_profit_percentage_correct_l2114_211436

-- Definitions based on the conditions
noncomputable def profit_percentage_A := 30
noncomputable def discount_percentage_A := 10
noncomputable def profit_percentage_B := 24
noncomputable def discount_percentage_B := 15
noncomputable def profit_percentage_C := 40
noncomputable def discount_percentage_C := 20

-- Function to calculate selling price without discount
noncomputable def selling_price_without_discount (cost_price profit_percentage : ℝ) : ℝ :=
  cost_price * (1 + profit_percentage / 100)

-- Assume cost price for simplicity
noncomputable def cost_price : ℝ := 100

-- Calculations based on the conditions
noncomputable def selling_price_A := selling_price_without_discount cost_price profit_percentage_A
noncomputable def selling_price_B := selling_price_without_discount cost_price profit_percentage_B
noncomputable def selling_price_C := selling_price_without_discount cost_price profit_percentage_C

-- Calculate total cost price and the total selling price without any discount
noncomputable def total_cost_price := 3 * cost_price
noncomputable def total_selling_price_without_discount := selling_price_A + selling_price_B + selling_price_C

-- Combined profit
noncomputable def combined_profit := total_selling_price_without_discount - total_cost_price

-- Combined profit percentage
noncomputable def combined_profit_percentage := (combined_profit / total_cost_price) * 100

theorem combined_profit_percentage_correct :
  combined_profit_percentage = 31.33 :=
by
  sorry

end combined_profit_percentage_correct_l2114_211436


namespace sam_age_l2114_211411

theorem sam_age (drew_current_age : ℕ) (drew_future_age : ℕ) (sam_future_age : ℕ) : 
  (drew_current_age = 12) → 
  (drew_future_age = drew_current_age + 5) → 
  (sam_future_age = 3 * drew_future_age) → 
  (sam_future_age - 5 = 46) := 
by sorry

end sam_age_l2114_211411


namespace borrowed_amount_l2114_211487

theorem borrowed_amount (P : ℝ) 
    (borrow_rate : ℝ := 4) 
    (lend_rate : ℝ := 6) 
    (borrow_time : ℝ := 2) 
    (lend_time : ℝ := 2) 
    (gain_per_year : ℝ := 140) 
    (h₁ : ∀ (P : ℝ), P / 8.333 - P / 12.5 = 280) 
    : P = 7000 := 
sorry

end borrowed_amount_l2114_211487


namespace equal_lengths_imply_equal_segments_l2114_211457

theorem equal_lengths_imply_equal_segments 
  (a₁ a₂ b₁ b₂ x y : ℝ) 
  (h₁ : a₁ = a₂) 
  (h₂ : b₁ = b₂) : 
  x = y := 
sorry

end equal_lengths_imply_equal_segments_l2114_211457


namespace total_deposit_amount_l2114_211434

def markDeposit : ℕ := 88
def bryanDeposit (markAmount : ℕ) : ℕ := 5 * markAmount - 40
def totalDeposit (markAmount bryanAmount : ℕ) : ℕ := markAmount + bryanAmount

theorem total_deposit_amount : totalDeposit markDeposit (bryanDeposit markDeposit) = 488 := 
by sorry

end total_deposit_amount_l2114_211434


namespace number_4_div_p_equals_l2114_211438

-- Assume the necessary conditions
variables (p q : ℝ)
variables (h1 : 4 / q = 18) (h2 : p - q = 0.2777777777777778)

-- Define the proof problem
theorem number_4_div_p_equals (N : ℝ) (hN : 4 / p = N) : N = 8 :=
by 
  sorry

end number_4_div_p_equals_l2114_211438


namespace transform_eq_l2114_211469

theorem transform_eq (m n x y : ℕ) (h1 : m + x = n + y) (h2 : x = y) : m = n :=
sorry

end transform_eq_l2114_211469


namespace distinct_roots_quadratic_l2114_211461

theorem distinct_roots_quadratic (a x₁ x₂ : ℝ) (h₁ : x^2 + a*x + 8 = 0) 
  (h₂ : x₁ ≠ x₂) (h₃ : x₁ - 64 / (17 * x₂^3) = x₂ - 64 / (17 * x₁^3)) : 
  a = 12 ∨ a = -12 := 
sorry

end distinct_roots_quadratic_l2114_211461


namespace factorization_of_a_square_minus_one_l2114_211458

theorem factorization_of_a_square_minus_one (a : ℤ) : a^2 - 1 = (a + 1) * (a - 1) := 
  by sorry

end factorization_of_a_square_minus_one_l2114_211458


namespace christina_total_payment_l2114_211465

def item1_ticket_price : ℝ := 200
def item1_discount1 : ℝ := 0.25
def item1_discount2 : ℝ := 0.15
def item1_tax_rate : ℝ := 0.07

def item2_ticket_price : ℝ := 150
def item2_discount : ℝ := 0.30
def item2_tax_rate : ℝ := 0.10

def item3_ticket_price : ℝ := 100
def item3_discount : ℝ := 0.20
def item3_tax_rate : ℝ := 0.05

def expected_total : ℝ := 335.93

theorem christina_total_payment :
  let item1_final_price :=
    (item1_ticket_price * (1 - item1_discount1) * (1 - item1_discount2)) * (1 + item1_tax_rate)
  let item2_final_price :=
    (item2_ticket_price * (1 - item2_discount)) * (1 + item2_tax_rate)
  let item3_final_price :=
    (item3_ticket_price * (1 - item3_discount)) * (1 + item3_tax_rate)
  item1_final_price + item2_final_price + item3_final_price = expected_total :=
by
  sorry

end christina_total_payment_l2114_211465


namespace inequality_solution_l2114_211404

theorem inequality_solution (x : ℝ) (h : x ≠ 2 ∧ x ≠ -2) :
    (x^2 - 9) / (x^2 - 4) > 0 ↔ (x < -3 ∨ x > 3) := by
  sorry

end inequality_solution_l2114_211404


namespace employed_females_percentage_l2114_211445

theorem employed_females_percentage (E M : ℝ) (hE : E = 60) (hM : M = 42) : ((E - M) / E) * 100 = 30 := by
  sorry

end employed_females_percentage_l2114_211445


namespace inequality_always_holds_l2114_211435

theorem inequality_always_holds (a b : ℝ) (h : a * b > 0) : (b / a + a / b) ≥ 2 :=
sorry

end inequality_always_holds_l2114_211435


namespace find_P_l2114_211402

theorem find_P (P : ℕ) (h : P^2 + P = 30) : P = 5 :=
sorry

end find_P_l2114_211402


namespace intersection_A_B_eq_l2114_211492

def A : Set ℝ := { x | (x / (x - 1)) ≥ 0 }

def B : Set ℝ := { y | ∃ x : ℝ, y = 3 * x^2 + 1 }

theorem intersection_A_B_eq :
  (A ∩ B) = { y : ℝ | 1 < y } :=
sorry

end intersection_A_B_eq_l2114_211492


namespace original_cost_price_of_car_l2114_211418

theorem original_cost_price_of_car
    (S_m S_f C : ℝ)
    (h1 : S_m = 0.86 * C)
    (h2 : S_f = 54000)
    (h3 : S_f = 1.20 * S_m) :
    C = 52325.58 :=
by
    sorry

end original_cost_price_of_car_l2114_211418


namespace triangle_construction_possible_l2114_211423

-- Define the entities involved
variables {α β : ℝ} {a c : ℝ}

-- State the theorem
theorem triangle_construction_possible (a c : ℝ) (h : α = 2 * β) : a > (2 / 3) * c :=
sorry

end triangle_construction_possible_l2114_211423


namespace necklace_sum_l2114_211478

theorem necklace_sum (H J x S : ℕ) (hH : H = 25) (h1 : H = J + 5) (h2 : x = J / 2) (h3 : S = 2 * H) : H + J + x + S = 105 :=
by 
  sorry

end necklace_sum_l2114_211478


namespace count_primes_1021_eq_one_l2114_211405

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem count_primes_1021_eq_one :
  (∃ n : ℕ, 3 ≤ n ∧ is_prime (n^3 + 2*n + 1) ∧
  ∀ m : ℕ, (3 ≤ m ∧ m ≠ n) → ¬ is_prime (m^3 + 2*m + 1)) :=
sorry

end count_primes_1021_eq_one_l2114_211405


namespace arithmetic_sequence_closed_form_l2114_211459

noncomputable def B_n (n : ℕ) : ℝ :=
  2 * (1 - (-2)^n) / 3

theorem arithmetic_sequence_closed_form (a_n : ℕ → ℝ) (S_n : ℕ → ℝ)
  (h1 : a_n 1 = 1) (h2 : S_n 3 = 0) :
  B_n n = 2 * (1 - (-2)^n) / 3 := sorry

end arithmetic_sequence_closed_form_l2114_211459


namespace number_of_circumcenter_quadrilaterals_l2114_211475

-- Definitions for each type of quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

def is_square (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_kite (q : Quadrilateral) : Prop := sorry
def is_trapezoid (q : Quadrilateral) : Prop := sorry
def has_circumcenter (q : Quadrilateral) : Prop := sorry

-- List of quadrilaterals
def square : Quadrilateral := sorry
def rectangle : Quadrilateral := sorry
def rhombus : Quadrilateral := sorry
def kite : Quadrilateral := sorry
def trapezoid : Quadrilateral := sorry

-- Proof that the number of quadrilaterals with a point equidistant from all vertices is 2
theorem number_of_circumcenter_quadrilaterals :
  (has_circumcenter square) ∧
  (has_circumcenter rectangle) ∧
  ¬ (has_circumcenter rhombus) ∧
  ¬ (has_circumcenter kite) ∧
  ¬ (has_circumcenter trapezoid) →
  2 = 2 :=
by
  sorry

end number_of_circumcenter_quadrilaterals_l2114_211475


namespace shark_fin_falcata_area_is_correct_l2114_211462

noncomputable def radius_large : ℝ := 3
noncomputable def center_large : ℝ × ℝ := (0, 0)

noncomputable def radius_small : ℝ := 3 / 2
noncomputable def center_small : ℝ × ℝ := (0, 3 / 2)

noncomputable def area_large_quarter_circle : ℝ := (1 / 4) * Real.pi * (radius_large ^ 2)
noncomputable def area_small_semicircle : ℝ := (1 / 2) * Real.pi * (radius_small ^ 2)

noncomputable def shark_fin_falcata_area (area_large_quarter_circle area_small_semicircle : ℝ) : ℝ := 
  area_large_quarter_circle - area_small_semicircle

theorem shark_fin_falcata_area_is_correct : 
  shark_fin_falcata_area area_large_quarter_circle area_small_semicircle = (9 * Real.pi) / 8 := 
by
  sorry

end shark_fin_falcata_area_is_correct_l2114_211462


namespace determinant_trig_matrix_eq_one_l2114_211489

theorem determinant_trig_matrix_eq_one (α θ : ℝ) :
  Matrix.det ![
  ![Real.cos α * Real.cos θ, Real.cos α * Real.sin θ, Real.sin α],
  ![Real.sin θ, -Real.cos θ, 0],
  ![Real.sin α * Real.cos θ, Real.sin α * Real.sin θ, -Real.cos α]
  ] = 1 :=
by
  sorry

end determinant_trig_matrix_eq_one_l2114_211489


namespace third_car_year_l2114_211480

theorem third_car_year (y1 y2 y3 : ℕ) (h1 : y1 = 1970) (h2 : y2 = y1 + 10) (h3 : y3 = y2 + 20) : y3 = 2000 :=
by
  sorry

end third_car_year_l2114_211480


namespace total_weight_mason_hotdogs_l2114_211499

-- Definitions from conditions
def weight_hotdog := 2
def weight_burger := 5
def weight_pie := 10
def noah_burgers := 8
def jacob_pies := noah_burgers - 3
def mason_hotdogs := 3 * jacob_pies

-- Statement to prove
theorem total_weight_mason_hotdogs : mason_hotdogs * weight_hotdog = 30 := 
by 
  sorry

end total_weight_mason_hotdogs_l2114_211499


namespace bus_total_distance_l2114_211425

theorem bus_total_distance
  (distance40 : ℝ)
  (distance60 : ℝ)
  (speed40 : ℝ)
  (speed60 : ℝ)
  (total_time : ℝ)
  (distance40_eq : distance40 = 100)
  (speed40_eq : speed40 = 40)
  (speed60_eq : speed60 = 60)
  (total_time_eq : total_time = 5)
  (time40 : ℝ)
  (time40_eq : time40 = distance40 / speed40)
  (time_equation : time40 + distance60 / speed60 = total_time) :
  distance40 + distance60 = 250 := sorry

end bus_total_distance_l2114_211425


namespace math_problem_l2114_211488

noncomputable def problem_statement (f : ℚ → ℝ) : Prop :=
  (∀ r s : ℚ, ∃ n : ℤ, f (r + s) = f r + f s + n) →
  ∃ (q : ℕ) (p : ℤ), abs (f (1 / q) - p) ≤ 1 / 2012

-- To state this problem as a theorem in Lean 4
theorem math_problem (f : ℚ → ℝ) :
  problem_statement f :=
sorry

end math_problem_l2114_211488


namespace sqrt_529000_pow_2_5_l2114_211414

theorem sqrt_529000_pow_2_5 : (529000 ^ (1 / 2) ^ (5 / 2)) = 14873193 := by
  sorry

end sqrt_529000_pow_2_5_l2114_211414


namespace rhombus_area_from_roots_l2114_211448

-- Definition of the quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 10 * x + 24 = 0

-- Define the roots of the quadratic equation
def roots (a b : ℝ) : Prop := quadratic_eq a ∧ quadratic_eq b

-- Final mathematical statement to prove
theorem rhombus_area_from_roots (a b : ℝ) (h : roots a b) :
  a * b = 24 → (1 / 2) * a * b = 12 := 
by
  sorry

end rhombus_area_from_roots_l2114_211448


namespace total_pounds_of_food_l2114_211473

-- Conditions
def chicken := 16
def hamburgers := chicken / 2
def hot_dogs := hamburgers + 2
def sides := hot_dogs / 2

-- Define the total pounds of food
def total_food := chicken + hamburgers + hot_dogs + sides

-- Theorem statement that corresponds to the problem, showing the final result
theorem total_pounds_of_food : total_food = 39 := 
by
  -- Placeholder for the proof
  sorry

end total_pounds_of_food_l2114_211473


namespace find_nm_l2114_211470

theorem find_nm :
  ∃ n m : Int, (-120 : Int) ≤ n ∧ n ≤ 120 ∧ (-120 : Int) ≤ m ∧ m ≤ 120 ∧ 
  (Real.sin (n * Real.pi / 180) = Real.sin (580 * Real.pi / 180)) ∧ 
  (Real.cos (m * Real.pi / 180) = Real.cos (300 * Real.pi / 180)) ∧ 
  n = -40 ∧ m = -60 := by
  sorry

end find_nm_l2114_211470


namespace letters_with_dot_not_line_l2114_211484

-- Definitions from conditions
def D_inter_S : ℕ := 23
def S : ℕ := 42
def Total_letters : ℕ := 70

-- Problem statement
theorem letters_with_dot_not_line : (Total_letters - S - D_inter_S) = 5 :=
by sorry

end letters_with_dot_not_line_l2114_211484


namespace base_number_of_equation_l2114_211493

theorem base_number_of_equation (n : ℕ) (h_n: n = 17)
  (h_eq: 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = some_number^18) : some_number = 2 := by
  sorry

end base_number_of_equation_l2114_211493


namespace box_count_neither_markers_nor_erasers_l2114_211407

-- Define the conditions as parameters.
def total_boxes : ℕ := 15
def markers_count : ℕ := 10
def erasers_count : ℕ := 5
def both_count : ℕ := 4

-- State the theorem to be proven in Lean 4.
theorem box_count_neither_markers_nor_erasers : 
  total_boxes - (markers_count + erasers_count - both_count) = 4 := 
sorry

end box_count_neither_markers_nor_erasers_l2114_211407


namespace max_ratio_three_digit_sum_l2114_211442

theorem max_ratio_three_digit_sum (N a b c : ℕ) (hN : N = 100 * a + 10 * b + c) (ha : 1 ≤ a) (hb : b ≤ 9) (hc : c ≤ 9) :
  (∀ (N' a' b' c' : ℕ), N' = 100 * a' + 10 * b' + c' → 1 ≤ a' → b' ≤ 9 → c' ≤ 9 → (N' : ℚ) / (a' + b' + c') ≤ 100) :=
sorry

end max_ratio_three_digit_sum_l2114_211442


namespace beads_needed_for_jewelry_l2114_211422

/-
  We define the parameters based on the problem statement.
-/

def green_beads : ℕ := 3
def purple_beads : ℕ := 5
def red_beads : ℕ := 2 * green_beads
def total_beads_per_pattern : ℕ := green_beads + purple_beads + red_beads

def repeats_per_bracelet : ℕ := 3
def repeats_per_necklace : ℕ := 5

/-
  We calculate the total number of beads for 1 bracelet and 10 necklaces.
-/

def beads_per_bracelet : ℕ := total_beads_per_pattern * repeats_per_bracelet
def beads_per_necklace : ℕ := total_beads_per_pattern * repeats_per_necklace
def total_beads_needed : ℕ := beads_per_bracelet + beads_per_necklace * 10

theorem beads_needed_for_jewelry:
  total_beads_needed = 742 :=
by 
  sorry

end beads_needed_for_jewelry_l2114_211422


namespace take_home_pay_correct_l2114_211419

noncomputable def faith_take_home_pay : Float :=
  let regular_hourly_rate := 13.50
  let regular_hours_per_day := 8
  let days_per_week := 5
  let regular_hours_per_week := regular_hours_per_day * days_per_week
  let regular_earnings_per_week := regular_hours_per_week * regular_hourly_rate

  let overtime_rate_multiplier := 1.5
  let overtime_hourly_rate := regular_hourly_rate * overtime_rate_multiplier
  let overtime_hours_per_day := 2
  let overtime_hours_per_week := overtime_hours_per_day * days_per_week
  let overtime_earnings_per_week := overtime_hours_per_week * overtime_hourly_rate

  let total_sales := 3200.0
  let commission_rate := 0.10
  let commission := total_sales * commission_rate

  let total_earnings_before_deductions := regular_earnings_per_week + overtime_earnings_per_week + commission

  let deduction_rate := 0.25
  let amount_withheld := total_earnings_before_deductions * deduction_rate
  let amount_withheld_rounded := (amount_withheld * 100).round / 100

  let take_home_pay := total_earnings_before_deductions - amount_withheld_rounded
  take_home_pay

theorem take_home_pay_correct : faith_take_home_pay = 796.87 :=
by
  /- Proof omitted -/
  sorry

end take_home_pay_correct_l2114_211419


namespace complement_of_M_in_U_l2114_211477

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {4, 5}

theorem complement_of_M_in_U : compl M ∩ U = {1, 2, 3} :=
by
  sorry

end complement_of_M_in_U_l2114_211477


namespace number_of_ordered_pairs_l2114_211450

theorem number_of_ordered_pairs (a b : ℤ) (h : a ≠ 0 ∧ b ≠ 0 ∧ (1 / a + 1 / b = 1 / 24)) : 
  ∃ n : ℕ, n = 41 :=
by
  sorry

end number_of_ordered_pairs_l2114_211450


namespace cube_vertices_faces_edges_l2114_211453

theorem cube_vertices_faces_edges (V F E : ℕ) (hv : V = 8) (hf : F = 6) (euler : V - E + F = 2) : E = 12 :=
by
  sorry

end cube_vertices_faces_edges_l2114_211453


namespace circle_radius_l2114_211421

theorem circle_radius (A B C O : Type) (AB AC : ℝ) (OA : ℝ) (r : ℝ) 
  (h1 : AB * AC = 60)
  (h2 : OA = 8) 
  (h3 : (8 + r) * (8 - r) = 60) : r = 2 :=
sorry

end circle_radius_l2114_211421


namespace number_of_math_fun_books_l2114_211424

def intelligence_challenge_cost := 18
def math_fun_cost := 8
def total_spent := 92

theorem number_of_math_fun_books (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 5) (hy : intelligence_challenge_cost * x + math_fun_cost * y = total_spent) : y = 7 := 
by
  sorry

end number_of_math_fun_books_l2114_211424


namespace true_or_false_is_true_l2114_211476

theorem true_or_false_is_true (p q : Prop) (hp : p = true) (hq : q = false) : p ∨ q = true :=
by
  sorry

end true_or_false_is_true_l2114_211476


namespace probability_of_different_cousins_name_l2114_211443

theorem probability_of_different_cousins_name :
  let total_letters := 19
  let amelia_letters := 6
  let bethany_letters := 7
  let claire_letters := 6
  let probability := 
    2 * ((amelia_letters / (total_letters : ℚ)) * (bethany_letters / (total_letters - 1 : ℚ)) +
         (amelia_letters / (total_letters : ℚ)) * (claire_letters / (total_letters - 1 : ℚ)) +
         (claire_letters / (total_letters : ℚ)) * (bethany_letters / (total_letters - 1 : ℚ)))
  probability = 40 / 57 := sorry

end probability_of_different_cousins_name_l2114_211443


namespace sum_of_natural_numbers_eq_4005_l2114_211474

theorem sum_of_natural_numbers_eq_4005 :
  ∃ n : ℕ, (n * (n + 1)) / 2 = 4005 ∧ n = 89 :=
by
  sorry

end sum_of_natural_numbers_eq_4005_l2114_211474


namespace option_A_option_B_option_C_option_D_l2114_211483

-- Define the equation of the curve
def curve (k : ℝ) (x y : ℝ) : Prop :=
  x^2 / (k + 1) + y^2 / (5 - k) = 1

-- Prove that when k=2, the curve is a circle
theorem option_A (x y : ℝ) : curve 2 x y ↔ x^2 + y^2 = 3 :=
by
  sorry

-- Prove the necessary and sufficient condition for the curve to be an ellipse
theorem option_B (k : ℝ) : (-1 < k ∧ k < 5) ↔ ∃ x y, curve k x y ∧ (k ≠ 2) :=
by
  sorry

-- Prove the condition for the curve to be a hyperbola with foci on the y-axis
theorem option_C (k : ℝ) : k < -1 ↔ ∃ x y, curve k x y ∧ (k < -1 ∧ k < 5) :=
by
  sorry

-- Prove that there does not exist a real number k such that the curve is a parabola
theorem option_D : ¬ (∃ k x y, curve k x y ∧ ∃ a b, x = a ∧ y = b) :=
by
  sorry

end option_A_option_B_option_C_option_D_l2114_211483


namespace bond_selling_price_l2114_211427

def bond_face_value : ℝ := 5000
def bond_interest_rate : ℝ := 0.06
def interest_approx : ℝ := bond_face_value * bond_interest_rate
def selling_price_interest_rate : ℝ := 0.065
def approximate_selling_price : ℝ := 4615.38

theorem bond_selling_price :
  interest_approx = selling_price_interest_rate * approximate_selling_price :=
sorry

end bond_selling_price_l2114_211427


namespace diagonals_in_nine_sided_polygon_l2114_211429

def num_diagonals_in_polygon (n : ℕ) : ℕ :=
  let total_pairs := n * (n - 1) / 2
  total_pairs - n

theorem diagonals_in_nine_sided_polygon : num_diagonals_in_polygon 9 = 27 := by
  sorry

end diagonals_in_nine_sided_polygon_l2114_211429


namespace function_graph_second_quadrant_l2114_211481

theorem function_graph_second_quadrant (b : ℝ) (h : ∀ x, 2 ^ x + b - 1 ≥ 0): b ≤ 0 :=
sorry

end function_graph_second_quadrant_l2114_211481


namespace total_amount_l2114_211498

noncomputable def A : ℝ := 396.00000000000006
noncomputable def B : ℝ := A * (3 / 2)
noncomputable def C : ℝ := B * 4

theorem total_amount (A_eq : A = 396.00000000000006) (A_B_relation : A = (2 / 3) * B) (B_C_relation : B = (1 / 4) * C) :
  396.00000000000006 + B + C = 3366.000000000001 := by
  sorry

end total_amount_l2114_211498


namespace visitors_saturday_l2114_211426

def friday_visitors : ℕ := 3575
def saturday_visitors : ℕ := 5 * friday_visitors

theorem visitors_saturday : saturday_visitors = 17875 := by
  -- proof details would go here
  sorry

end visitors_saturday_l2114_211426


namespace parabola_point_b_l2114_211433

variable {a b : ℝ}

theorem parabola_point_b (h1 : 6 = 2^2 + 2*a + b) (h2 : -14 = (-2)^2 - 2*a + b) : b = -8 :=
by
  -- sorry as a placeholder for the actual proof.
  sorry

end parabola_point_b_l2114_211433


namespace stans_average_speed_l2114_211482

/-- Given that Stan drove 420 miles in 6 hours, 480 miles in 7 hours, and 300 miles in 5 hours,
prove that his average speed for the entire trip is 1200/18 miles per hour. -/
theorem stans_average_speed :
  let total_distance := 420 + 480 + 300
  let total_time := 6 + 7 + 5
  total_distance / total_time = 1200 / 18 :=
by
  sorry

end stans_average_speed_l2114_211482


namespace new_acute_angle_ACB_l2114_211455

-- Define the initial condition: the measure of angle ACB is 50 degrees.
def measure_ACB_initial : ℝ := 50

-- Define the rotation: ray CA is rotated by 540 degrees clockwise.
def rotation_CW_degrees : ℝ := 540

-- Theorem statement: The positive measure of the new acute angle ACB.
theorem new_acute_angle_ACB : 
  ∃ (new_angle : ℝ), new_angle = 50 ∧ new_angle < 90 := 
by
  sorry

end new_acute_angle_ACB_l2114_211455


namespace additional_rows_added_l2114_211491

theorem additional_rows_added
  (initial_tiles : ℕ) (initial_rows : ℕ) (initial_columns : ℕ) (new_columns : ℕ) (new_rows : ℕ)
  (h1 : initial_tiles = 48)
  (h2 : initial_rows = 6)
  (h3 : initial_columns = initial_tiles / initial_rows)
  (h4 : new_columns = initial_columns - 2)
  (h5 : new_rows = initial_tiles / new_columns) :
  new_rows - initial_rows = 2 := by sorry

end additional_rows_added_l2114_211491


namespace problem_proof_l2114_211497

-- Define the mixed numbers and their conversions to improper fractions
def mixed_number_1 := 84 * 19 + 4  -- 1600
def mixed_number_2 := 105 * 19 + 5 -- 2000 

-- Define the improper fractions
def improper_fraction_1 := mixed_number_1 / 19
def improper_fraction_2 := mixed_number_2 / 19

-- Define the decimals and their conversions to fractions
def decimal_1 := 11 / 8  -- 1.375
def decimal_2 := 9 / 10  -- 0.9

-- Perform the multiplications
def multiplication_1 := (improper_fraction_1 * decimal_1 : ℚ)
def multiplication_2 := (improper_fraction_2 * decimal_2 : ℚ)

-- Perform the addition
def addition_result := multiplication_1 + multiplication_2

-- The final result is converted to a fraction for comparison
def final_result := 4000 / 19

-- Define and state the theorem
theorem problem_proof : addition_result = final_result := by
  sorry

end problem_proof_l2114_211497


namespace intersection_A_B_l2114_211449

-- Definition of set A based on the given inequality
def A : Set ℝ := {x | x^2 - 2 * x - 3 ≤ 0}

-- Definition of set B
def B : Set ℝ := {-3, -1, 1, 3}

-- Prove the intersection A ∩ B equals the expected set {-1, 1, 3}
theorem intersection_A_B : A ∩ B = {-1, 1, 3} := 
by
  sorry

end intersection_A_B_l2114_211449


namespace find_circle_equation_l2114_211409

-- Define the conditions and problem
def circle_standard_equation (p1 p2 : ℝ × ℝ) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (xc, yc) := center
  (x2 - xc)^2 + (y2 - yc)^2 = radius^2

-- Define the conditions as given in the problem
def point_on_circle : Prop := circle_standard_equation (2, 0) (2, 2) (2, 2) 2

-- The main theorem to prove that the standard equation of the circle holds
theorem find_circle_equation : 
  point_on_circle →
  ∃ h k r, h = 2 ∧ k = 2 ∧ r = 2 ∧ (x - h)^2 + (y - k)^2 = r^2 :=
by
  sorry

end find_circle_equation_l2114_211409


namespace sum_of_squares_ge_two_ab_l2114_211415

theorem sum_of_squares_ge_two_ab (a b : ℝ) : a^2 + b^2 ≥ 2 * a * b := 
  sorry

end sum_of_squares_ge_two_ab_l2114_211415


namespace side_of_square_is_25_l2114_211437

theorem side_of_square_is_25 (area_of_circle : ℝ) (perimeter_of_square : ℝ) (h1 : area_of_circle = 100) (h2 : area_of_circle = perimeter_of_square) : perimeter_of_square / 4 = 25 :=
by {
  -- Insert the steps here if necessary.
  sorry
}

end side_of_square_is_25_l2114_211437


namespace sum_of_integers_c_with_four_solutions_l2114_211431

noncomputable def g (x : ℝ) : ℝ :=
  ((x - 4) * (x - 2) * x * (x + 2) * (x + 4) / 120) - 2

theorem sum_of_integers_c_with_four_solutions :
  (∃ (c : ℤ), ∀ x : ℝ, -4.5 ≤ x ∧ x ≤ 4.5 → g x = c ↔ c = -2) → c = -2 :=
by
  sorry

end sum_of_integers_c_with_four_solutions_l2114_211431


namespace square_units_digit_eq_9_l2114_211444

/-- The square of which whole number has a units digit of 9? -/
theorem square_units_digit_eq_9 (n : ℕ) (h : ∃ m : ℕ, n = m^2 ∧ m % 10 = 9) : n = 3 ∨ n = 7 := by
  sorry

end square_units_digit_eq_9_l2114_211444


namespace min_value_y_l2114_211447

theorem min_value_y : ∃ x : ℝ, (y = 2 * x^2 + 8 * x + 18) ∧ (∀ x : ℝ, y ≥ 10) :=
by
  sorry

end min_value_y_l2114_211447


namespace find_n_in_sequence_l2114_211486

theorem find_n_in_sequence (n : ℕ) (a : ℕ → ℕ) (S : ℕ → ℕ) 
    (h1 : a 1 = 2) 
    (h2 : ∀ n, a (n+1) = 2 * a n) 
    (h3 : S n = 126) 
    (h4 : S n = 2^(n+1) - 2) : 
  n = 6 :=
sorry

end find_n_in_sequence_l2114_211486


namespace sqrt_meaningful_l2114_211406

theorem sqrt_meaningful (x : ℝ) : x + 1 >= 0 ↔ (∃ y : ℝ, y * y = x + 1) := by
  sorry

end sqrt_meaningful_l2114_211406


namespace range_m_l2114_211490

def A (x : ℝ) : Prop := x^2 - 3 * x - 10 ≤ 0

def B (m x : ℝ) : Prop := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem range_m (m : ℝ) :
  (∀ x, B m x → A x) ↔ -3 ≤ m ∧ m ≤ 3 :=
by
  sorry

end range_m_l2114_211490


namespace coconut_grove_l2114_211403

theorem coconut_grove (x Y : ℕ) (h1 : 3 * x ≠ 0) (h2 : (x+3) * 60 + x * Y + (x-3) * 180 = 3 * x * 100) (hx : x = 6) : Y = 120 :=
by 
  sorry

end coconut_grove_l2114_211403


namespace min_moves_l2114_211456

theorem min_moves (n : ℕ) : (n * (n + 1)) / 2 > 100 → n = 15 :=
by
  sorry

end min_moves_l2114_211456


namespace percent_difference_calculation_l2114_211420

theorem percent_difference_calculation :
  (0.80 * 45) - ((4 / 5) * 25) = 16 :=
by sorry

end percent_difference_calculation_l2114_211420


namespace polynomial_factorization_l2114_211479

theorem polynomial_factorization : (∀ x : ℤ, x^9 + x^6 + x^3 + 1 = (x^3 + 1) * (x^6 - x^3 + 1)) := by
  intro x
  sorry

end polynomial_factorization_l2114_211479


namespace intermediate_circle_radius_l2114_211410

theorem intermediate_circle_radius (r1 r3: ℝ) (h1: r1 = 5) (h2: r3 = 13) 
  (h3: π * r1 ^ 2 = π * r3 ^ 2 - π * r2 ^ 2) : r2 = 12 := sorry


end intermediate_circle_radius_l2114_211410


namespace rational_solutions_product_l2114_211446

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem rational_solutions_product :
  ∀ c : ℕ, (c > 0) → (is_perfect_square (49 - 12 * c)) → (∃ a b : ℕ, a = 4 ∧ b = 2 ∧ a * b = 8) :=
by sorry

end rational_solutions_product_l2114_211446


namespace interest_rate_proof_l2114_211485

noncomputable def remaining_interest_rate (total_investment yearly_interest part_investment interest_rate_part amount_remaining_interest : ℝ) : Prop :=
  (part_investment * interest_rate_part) + amount_remaining_interest = yearly_interest ∧
  (total_investment - part_investment) * (amount_remaining_interest / (total_investment - part_investment)) = amount_remaining_interest

theorem interest_rate_proof :
  remaining_interest_rate 3000 256 800 0.1 176 :=
by
  sorry

end interest_rate_proof_l2114_211485


namespace intersection_A_B_l2114_211496

def A : Set ℝ := { x | Real.sqrt x ≤ 3 }
def B : Set ℝ := { x | x^2 ≤ 9 }

theorem intersection_A_B : A ∩ B = { x | 0 ≤ x ∧ x ≤ 3 } :=
by
  sorry

end intersection_A_B_l2114_211496


namespace smallest_interesting_number_l2114_211412

def is_perfect_square (x : ℕ) : Prop :=
∃ y : ℕ, y * y = x

def is_perfect_cube (x : ℕ) : Prop :=
∃ y : ℕ, y * y * y = x

theorem smallest_interesting_number (n : ℕ) :
    is_perfect_square (2 * n) ∧ is_perfect_cube (15 * n) → n = 1800 :=
by {
  sorry
}

end smallest_interesting_number_l2114_211412


namespace new_average_after_doubling_l2114_211472

theorem new_average_after_doubling
  (avg : ℝ) (num_students : ℕ) (h_avg : avg = 40) (h_num_students : num_students = 10) :
  let total_marks := avg * num_students
  let new_total_marks := total_marks * 2
  let new_avg := new_total_marks / num_students
  new_avg = 80 :=
by
  sorry

end new_average_after_doubling_l2114_211472
