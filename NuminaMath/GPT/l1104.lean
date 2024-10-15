import Mathlib

namespace NUMINAMATH_GPT_sqrt_a_add_4b_eq_pm3_l1104_110480

theorem sqrt_a_add_4b_eq_pm3
  (a b : ℝ)
  (A_sol : a * (-1) + 5 * (-1) = 15)
  (B_sol : 4 * 5 - b * 2 = -2) :
  (a + 4 * b)^(1/2) = 3 ∨ (a + 4 * b)^(1/2) = -3 := by
  sorry

end NUMINAMATH_GPT_sqrt_a_add_4b_eq_pm3_l1104_110480


namespace NUMINAMATH_GPT_train_cross_bridge_time_l1104_110411

noncomputable def time_to_cross_bridge (length_of_train : ℝ) (speed_kmh : ℝ) (length_of_bridge : ℝ) : ℝ :=
  let total_distance := length_of_train + length_of_bridge
  let speed_mps := speed_kmh * (1000 / 3600)
  total_distance / speed_mps

theorem train_cross_bridge_time :
  time_to_cross_bridge 110 72 112 = 11.1 :=
by
  sorry

end NUMINAMATH_GPT_train_cross_bridge_time_l1104_110411


namespace NUMINAMATH_GPT_find_y_l1104_110470

-- Definitions based on conditions
variables (x y : ℝ)
def inversely_proportional (x y : ℝ) : Prop := ∃ k : ℝ, x * y = k

-- Lean statement capturing the problem
theorem find_y
  (h1 : inversely_proportional x y)
  (h2 : x + y = 60)
  (h3 : x = 3 * y)
  (h4 : x = -12) :
  y = -56.25 :=
sorry  -- Proof omitted

end NUMINAMATH_GPT_find_y_l1104_110470


namespace NUMINAMATH_GPT_river_width_proof_l1104_110488
noncomputable def river_width (V FR D : ℝ) : ℝ := V / (FR * D)

theorem river_width_proof :
  river_width 2933.3333333333335 33.33333333333333 4 = 22 :=
by
  simp [river_width]
  norm_num
  sorry

end NUMINAMATH_GPT_river_width_proof_l1104_110488


namespace NUMINAMATH_GPT_ratio_sub_add_l1104_110432

theorem ratio_sub_add (x y : ℝ) (h : x / y = 3 / 2) : (x - y) / (x + y) = 1 / 5 :=
sorry

end NUMINAMATH_GPT_ratio_sub_add_l1104_110432


namespace NUMINAMATH_GPT_sum_of_digits_l1104_110458

theorem sum_of_digits (A B C D : ℕ) (H1: A < B) (H2: B < C) (H3: C < D)
  (H4: A > 0) (H5: B > 0) (H6: C > 0) (H7: D > 0)
  (H8: 1000 * A + 100 * B + 10 * C + D + 1000 * D + 100 * C + 10 * B + A = 11990) : 
  (A, B, C, D) = (1, 9, 9, 9) :=
sorry

end NUMINAMATH_GPT_sum_of_digits_l1104_110458


namespace NUMINAMATH_GPT_range_of_a_l1104_110434

variable (a : ℝ)
def A (a : ℝ) : Set ℝ := {x | -2 - a < x ∧ x < a ∧ a > 0}

def p (a : ℝ) := 1 ∈ A a
def q (a : ℝ) := 2 ∈ A a

theorem range_of_a (h1 : p a ∨ q a) (h2 : ¬ (p a ∧ q a)) : 1 < a ∧ a ≤ 2 := sorry

end NUMINAMATH_GPT_range_of_a_l1104_110434


namespace NUMINAMATH_GPT_candy_division_l1104_110463

theorem candy_division (total_candy : ℕ) (students : ℕ) (per_student : ℕ) 
  (h1 : total_candy = 344) (h2 : students = 43) : 
  total_candy / students = per_student ↔ per_student = 8 := 
by 
  sorry

end NUMINAMATH_GPT_candy_division_l1104_110463


namespace NUMINAMATH_GPT_triangle_angle_inequality_l1104_110476

theorem triangle_angle_inequality (A B C : ℝ) (h : A + B + C = Real.pi) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) :
  4 / A + 1 / (B + C) ≥ 9 / Real.pi := by
  sorry

end NUMINAMATH_GPT_triangle_angle_inequality_l1104_110476


namespace NUMINAMATH_GPT_pants_cost_l1104_110452

theorem pants_cost (P : ℝ) : 
(80 + 3 * P + 300) * 0.90 = 558 → P = 80 :=
by
  sorry

end NUMINAMATH_GPT_pants_cost_l1104_110452


namespace NUMINAMATH_GPT_regular_polygon_sides_l1104_110403

theorem regular_polygon_sides (n : ℕ) (h : 2 < n)
  (interior_angle : ∀ n, (n - 2) * 180 / n = 144) : n = 10 :=
sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1104_110403


namespace NUMINAMATH_GPT_mail_per_house_l1104_110433

theorem mail_per_house (total_mail : ℕ) (total_houses : ℕ) (h_total_mail : total_mail = 48) (h_total_houses : total_houses = 8) : 
  total_mail / total_houses = 6 := 
by 
  sorry

end NUMINAMATH_GPT_mail_per_house_l1104_110433


namespace NUMINAMATH_GPT_all_weights_equal_l1104_110494

theorem all_weights_equal (w : Fin 13 → ℤ) 
  (h : ∀ (i : Fin 13), ∃ (a b : Multiset (Fin 12)),
    a + b = (Finset.univ.erase i).val ∧ Multiset.card a = 6 ∧ 
    Multiset.card b = 6 ∧ Multiset.sum (a.map w) = Multiset.sum (b.map w)) :
  ∀ i j, w i = w j :=
by sorry

end NUMINAMATH_GPT_all_weights_equal_l1104_110494


namespace NUMINAMATH_GPT_jade_pieces_left_l1104_110426

-- Define the initial number of pieces Jade has
def initial_pieces : Nat := 100

-- Define the number of pieces per level
def pieces_per_level : Nat := 7

-- Define the number of levels in the tower
def levels : Nat := 11

-- Define the resulting number of pieces Jade has left after building the tower
def pieces_left : Nat := initial_pieces - (pieces_per_level * levels)

-- The theorem stating that after building the tower, Jade has 23 pieces left
theorem jade_pieces_left : pieces_left = 23 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_jade_pieces_left_l1104_110426


namespace NUMINAMATH_GPT_domain_of_function_l1104_110498

noncomputable def is_defined (x : ℝ) : Prop :=
  (x + 4 ≥ 0) ∧ (x ≠ 0)

theorem domain_of_function :
  ∀ x : ℝ, is_defined x ↔ x ≥ -4 ∧ x ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l1104_110498


namespace NUMINAMATH_GPT_number_of_possible_values_r_l1104_110469

noncomputable def is_closest_approx (r : ℝ) : Prop :=
  (r >= 0.2857) ∧ (r < 0.2858)

theorem number_of_possible_values_r : 
  ∃ n : ℕ, (∀ r : ℝ, is_closest_approx r ↔ r = 0.2857 ∨ r = 0.2858 ∨ r = 0.2859) ∧ n = 3 :=
by
  sorry

end NUMINAMATH_GPT_number_of_possible_values_r_l1104_110469


namespace NUMINAMATH_GPT_roots_polynomial_identity_l1104_110418

theorem roots_polynomial_identity (a b c : ℝ) (h1 : a + b + c = 15) (h2 : a * b + b * c + c * a = 22) (h3 : a * b * c = 8) :
  (2 + a) * (2 + b) * (2 + c) = 120 :=
by
  sorry

end NUMINAMATH_GPT_roots_polynomial_identity_l1104_110418


namespace NUMINAMATH_GPT_range_of_a_l1104_110412

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 - a) * x + 1 else a ^ x

theorem range_of_a (a : ℝ) : (∀ x1 x2 : ℝ, x1 < x2 → f a x1 < f a x2) ↔ (3 / 2 ≤ a ∧ a < 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1104_110412


namespace NUMINAMATH_GPT_find_u_plus_v_l1104_110495

variables (u v : ℚ)

theorem find_u_plus_v (h1 : 5 * u - 6 * v = 19) (h2 : 3 * u + 5 * v = -1) : u + v = 27 / 43 := by
  sorry

end NUMINAMATH_GPT_find_u_plus_v_l1104_110495


namespace NUMINAMATH_GPT_regular_dodecahedron_has_12_faces_l1104_110446

-- Define a structure to represent a regular dodecahedron
structure RegularDodecahedron where

-- The main theorem to state that a regular dodecahedron has 12 faces
theorem regular_dodecahedron_has_12_faces (D : RegularDodecahedron) : ∃ faces : ℕ, faces = 12 := by
  sorry

end NUMINAMATH_GPT_regular_dodecahedron_has_12_faces_l1104_110446


namespace NUMINAMATH_GPT_trigonometric_identity_l1104_110444

open Real

theorem trigonometric_identity (α : ℝ) : 
  sin α * sin α + cos (π / 6 + α) * cos (π / 6 + α) + sin α * cos (π / 6 + α) = 3 / 4 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_l1104_110444


namespace NUMINAMATH_GPT_value_of_f2009_l1104_110471

noncomputable def f : ℝ → ℝ := sorry

theorem value_of_f2009 
  (h_ineq1 : ∀ x : ℝ, f x ≤ f (x+4) + 4)
  (h_ineq2 : ∀ x : ℝ, f (x+2) ≥ f x + 2)
  (h_f1 : f 1 = 0) :
  f 2009 = 2008 :=
sorry

end NUMINAMATH_GPT_value_of_f2009_l1104_110471


namespace NUMINAMATH_GPT_money_first_day_l1104_110455

-- Define the total mushrooms
def total_mushrooms : ℕ := 65

-- Define the mushrooms picked on the second day
def mushrooms_day2 : ℕ := 12

-- Define the mushrooms picked on the third day
def mushrooms_day3 : ℕ := 2 * mushrooms_day2

-- Define the price per mushroom
def price_per_mushroom : ℕ := 2

-- Prove that the amount of money made on the first day is $58
theorem money_first_day : (total_mushrooms - mushrooms_day2 - mushrooms_day3) * price_per_mushroom = 58 := 
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_money_first_day_l1104_110455


namespace NUMINAMATH_GPT_symmetric_line_eq_l1104_110438

theorem symmetric_line_eq (x y: ℝ) :
    (∃ (a b: ℝ), 3 * a - b + 2 = 0 ∧ a = 2 - x ∧ b = 2 - y) → 3 * x - y - 6 = 0 :=
by
    intro h
    sorry

end NUMINAMATH_GPT_symmetric_line_eq_l1104_110438


namespace NUMINAMATH_GPT_students_per_bus_correct_l1104_110441

def total_students : ℝ := 28
def number_of_buses : ℝ := 2.0
def students_per_bus : ℝ := 14

theorem students_per_bus_correct :
  total_students / number_of_buses = students_per_bus := 
by
  -- Proof should go here
  sorry

end NUMINAMATH_GPT_students_per_bus_correct_l1104_110441


namespace NUMINAMATH_GPT_circles_internally_tangent_l1104_110402

theorem circles_internally_tangent (R r : ℝ) (h1 : R + r = 5) (h2 : R * r = 6) (d : ℝ) (h3 : d = 1) : d = |R - r| :=
by
  -- This allows the logic of the solution to be captured as the theorem we need to prove
  sorry

end NUMINAMATH_GPT_circles_internally_tangent_l1104_110402


namespace NUMINAMATH_GPT_function_form_l1104_110454

theorem function_form (f : ℕ → ℕ) (H : ∀ (x y z : ℕ), x ≠ y → y ≠ z → z ≠ x → (∃ k : ℕ, x + y + z = k^2 ↔ ∃ m : ℕ, f x + f y + f z = m^2)) : ∃ k : ℕ, ∀ n : ℕ, f n = k^2 * n :=
by
  sorry

end NUMINAMATH_GPT_function_form_l1104_110454


namespace NUMINAMATH_GPT_factorization_correct_l1104_110461

theorem factorization_correct {x : ℝ} : (x - 15)^2 = x^2 - 30*x + 225 :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l1104_110461


namespace NUMINAMATH_GPT_three_digit_number_div_by_11_l1104_110436

theorem three_digit_number_div_by_11 (x y z n : ℕ) 
  (hx : 0 < x ∧ x < 10) 
  (hy : 0 ≤ y ∧ y < 10) 
  (hz : 0 ≤ z ∧ z < 10) 
  (hn : n = 100 * x + 10 * y + z) 
  (hq : (n / 11) = x + y + z) : 
  n = 198 :=
by
  sorry

end NUMINAMATH_GPT_three_digit_number_div_by_11_l1104_110436


namespace NUMINAMATH_GPT_ninety_percent_of_population_is_expected_number_l1104_110445

/-- Define the total population of the village -/
def total_population : ℕ := 9000

/-- Define the percentage rate as a fraction -/
def percentage_rate : ℕ := 90

/-- Define the expected number of people representing 90% of the population -/
def expected_number : ℕ := 8100

/-- The proof problem: Prove that 90% of the total population is 8100 -/
theorem ninety_percent_of_population_is_expected_number :
  (percentage_rate * total_population / 100) = expected_number :=
by
  sorry

end NUMINAMATH_GPT_ninety_percent_of_population_is_expected_number_l1104_110445


namespace NUMINAMATH_GPT_triangle_properties_l1104_110429

theorem triangle_properties
  (K : ℝ) (α β : ℝ)
  (hK : K = 62.4)
  (hα : α = 70 + 20/60 + 40/3600)
  (hβ : β = 36 + 50/60 + 30/3600) :
  ∃ (a b T : ℝ), 
    a = 16.55 ∧
    b = 30.0 ∧
    T = 260.36 :=
by
  sorry

end NUMINAMATH_GPT_triangle_properties_l1104_110429


namespace NUMINAMATH_GPT_roger_ant_l1104_110489

def expected_steps : ℚ := 11/3

theorem roger_ant (a b : ℕ) (h1 : expected_steps = a / b) (h2 : Nat.gcd a b = 1) : 100 * a + b = 1103 :=
sorry

end NUMINAMATH_GPT_roger_ant_l1104_110489


namespace NUMINAMATH_GPT_math_club_team_selection_l1104_110450

open Nat

-- Lean statement of the problem
theorem math_club_team_selection : 
  (choose 7 3) * (choose 9 3) = 2940 :=
by 
  sorry

end NUMINAMATH_GPT_math_club_team_selection_l1104_110450


namespace NUMINAMATH_GPT_percentage_difference_l1104_110486

variable (x y : ℝ)
variable (p : ℝ)  -- percentage by which x is less than y

theorem percentage_difference (h1 : y = x * 1.3333333333333333) : p = 25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_difference_l1104_110486


namespace NUMINAMATH_GPT_bogan_maggots_l1104_110400

theorem bogan_maggots (x : ℕ) (total_maggots : ℕ) (eaten_first : ℕ) (eaten_second : ℕ) (thrown_out : ℕ) 
  (h1 : eaten_first = 1) (h2 : eaten_second = 3) (h3 : total_maggots = 20) (h4 : thrown_out = total_maggots - eaten_first - eaten_second) 
  (h5 : x + eaten_first = thrown_out) : x = 15 :=
by
  -- Use the given conditions
  sorry

end NUMINAMATH_GPT_bogan_maggots_l1104_110400


namespace NUMINAMATH_GPT_work_days_of_b_l1104_110423

theorem work_days_of_b (d : ℕ) 
  (A B C : ℕ)
  (h_ratioA : A = (3 * 115) / 5)
  (h_ratioB : B = (4 * 115) / 5)
  (h_C : C = 115)
  (h_total_wages : 1702 = (A * 6) + (B * d) + (C * 4)) :
  d = 9 := 
sorry

end NUMINAMATH_GPT_work_days_of_b_l1104_110423


namespace NUMINAMATH_GPT_angle_B_max_area_triangle_l1104_110477
noncomputable section

open Real

variables {A B C a b c : ℝ}

-- Prove B = π / 3 given b sin A = √3 a cos B
theorem angle_B (h1 : b * sin A = sqrt 3 * a * cos B) : B = π / 3 :=
sorry

-- Prove if b = 2√3, the maximum area of triangle ABC is 3√3
theorem max_area_triangle (h1 : b * sin A = sqrt 3 * a * cos B) (h2 : b = 2 * sqrt 3) : 
    (1 / 2) * a * (a : ℝ) *  (sqrt 3 / 2 : ℝ) ≤ 3 * sqrt 3 :=
sorry

end NUMINAMATH_GPT_angle_B_max_area_triangle_l1104_110477


namespace NUMINAMATH_GPT_least_five_digit_perfect_square_and_cube_l1104_110415

theorem least_five_digit_perfect_square_and_cube : ∃ n : ℕ, (n >= 10000 ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 :=
by
  sorry

end NUMINAMATH_GPT_least_five_digit_perfect_square_and_cube_l1104_110415


namespace NUMINAMATH_GPT_total_cost_l1104_110462

/-- There are two types of discs, one costing 10.50 and another costing 8.50.
You bought a total of 10 discs, out of which 6 are priced at 8.50.
The task is to determine the total amount spent. -/
theorem total_cost (price1 price2 : ℝ) (num1 num2 : ℕ) 
  (h1 : price1 = 10.50) (h2 : price2 = 8.50) 
  (h3 : num1 = 6) (h4 : num2 = 10) 
  (h5 : num2 - num1 = 4) : 
  (num1 * price2 + (num2 - num1) * price1) = 93.00 := 
by
  sorry

end NUMINAMATH_GPT_total_cost_l1104_110462


namespace NUMINAMATH_GPT_total_cost_eq_57_l1104_110496

namespace CandyCost

-- Conditions
def cost_of_caramel : ℕ := 3
def cost_of_candy_bar : ℕ := 2 * cost_of_caramel
def cost_of_cotton_candy : ℕ := (4 * cost_of_candy_bar) / 2

-- Define the total cost calculation
def total_cost : ℕ :=
  (6 * cost_of_candy_bar) + (3 * cost_of_caramel) + cost_of_cotton_candy

-- Theorem we want to prove
theorem total_cost_eq_57 : total_cost = 57 :=
by
  sorry  -- Proof to be provided

end CandyCost

end NUMINAMATH_GPT_total_cost_eq_57_l1104_110496


namespace NUMINAMATH_GPT_parabola_distance_l1104_110419

theorem parabola_distance (y : ℝ) (h : y ^ 2 = 24) : |-6 - 1| = 7 :=
by { sorry }

end NUMINAMATH_GPT_parabola_distance_l1104_110419


namespace NUMINAMATH_GPT_picnic_men_count_l1104_110409

variables 
  (M W A C : ℕ)
  (h1 : M + W + C = 200) 
  (h2 : M = W + 20)
  (h3 : A = C + 20)
  (h4 : A = M + W)

theorem picnic_men_count : M = 65 :=
by
  sorry

end NUMINAMATH_GPT_picnic_men_count_l1104_110409


namespace NUMINAMATH_GPT_coupon_savings_difference_l1104_110430

theorem coupon_savings_difference {P : ℝ} (hP : P > 200)
  (couponA_savings : ℝ := 0.20 * P) 
  (couponB_savings : ℝ := 50)
  (couponC_savings : ℝ := 0.30 * (P - 200)) :
  (200 ≤ P - 200 + 50 → 200 ≤ P ∧ P ≤ 200 + 400 → 600 - 250 = 350) :=
by
  sorry

end NUMINAMATH_GPT_coupon_savings_difference_l1104_110430


namespace NUMINAMATH_GPT_abigail_money_loss_l1104_110479

theorem abigail_money_loss {initial spent remaining lost : ℤ} 
  (h1 : initial = 11) 
  (h2 : spent = 2) 
  (h3 : remaining = 3) 
  (h4 : lost = initial - spent - remaining) : 
  lost = 6 := sorry

end NUMINAMATH_GPT_abigail_money_loss_l1104_110479


namespace NUMINAMATH_GPT_number_of_sides_of_polygon_l1104_110491

-- Given definition about angles and polygons
def exterior_angle (sides: ℕ) : ℝ := 30

-- The sum of exterior angles of any polygon
def sum_exterior_angles : ℝ := 360

-- The proof statement
theorem number_of_sides_of_polygon (k : ℕ) 
  (h1 : exterior_angle k = 30) 
  (h2 : sum_exterior_angles = 360):
  k = 12 :=
sorry

end NUMINAMATH_GPT_number_of_sides_of_polygon_l1104_110491


namespace NUMINAMATH_GPT_period_of_cos_3x_l1104_110464

theorem period_of_cos_3x :
  ∃ T : ℝ, (∀ x : ℝ, (Real.cos (3 * (x + T))) = Real.cos (3 * x)) ∧ (T = (2 * Real.pi) / 3) :=
sorry

end NUMINAMATH_GPT_period_of_cos_3x_l1104_110464


namespace NUMINAMATH_GPT_increasing_condition_l1104_110420

-- Define the function f
def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 3

-- Define the derivative of f with respect to x
def f' (x a : ℝ) : ℝ := 2 * x - 2 * a

-- Prove that f is increasing on the interval [2, +∞) if and only if a ≤ 2
theorem increasing_condition (a : ℝ) : (∀ x ≥ 2, f' x a ≥ 0) ↔ (a ≤ 2) := 
sorry

end NUMINAMATH_GPT_increasing_condition_l1104_110420


namespace NUMINAMATH_GPT_extremum_range_a_l1104_110447

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - a * x^2 + x

theorem extremum_range_a :
  (∀ x : ℝ, -1 < x ∧ x < 0 → (f a x = 0 → ∃ x0 : ℝ, f a x0 = 0 ∧ -1 < x0 ∧ x0 < 0)) →
  a < -1/5 ∨ a = -1 :=
sorry

end NUMINAMATH_GPT_extremum_range_a_l1104_110447


namespace NUMINAMATH_GPT_min_distance_PS_l1104_110440

-- Definitions of the distances given in the problem
def PQ : ℝ := 12
def QR : ℝ := 7
def RS : ℝ := 5

-- Hypotheses for the problem
axiom h1 : PQ = 12
axiom h2 : QR = 7
axiom h3 : RS = 5

-- The goal is to prove that the minimum distance between P and S is 0.
theorem min_distance_PS : ∃ PS : ℝ, PS = 0 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_min_distance_PS_l1104_110440


namespace NUMINAMATH_GPT_min_value_expression_l1104_110499

theorem min_value_expression (a b : ℝ) (h1 : 2 * a + b = 1) (h2 : a > 0) (h3 : b > 0) :
  (1 / a) + ((1 - b) / b) = 2 + 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l1104_110499


namespace NUMINAMATH_GPT_age_difference_is_13_l1104_110437

variables (A B C X : ℕ)
variables (total_age_A_B total_age_B_C : ℕ)

-- Conditions
def condition1 : Prop := total_age_A_B = total_age_B_C + X
def condition2 : Prop := C = A - 13

-- Theorem statement
theorem age_difference_is_13 (h1: condition1 total_age_A_B total_age_B_C X)
                             (h2: condition2 A C) :
  X = 13 :=
sorry

end NUMINAMATH_GPT_age_difference_is_13_l1104_110437


namespace NUMINAMATH_GPT_veronica_reroll_probability_is_correct_l1104_110481

noncomputable def veronica_reroll_probability : ℚ :=
  let P := (5 : ℚ) / 54
  P

theorem veronica_reroll_probability_is_correct :
  veronica_reroll_probability = (5 : ℚ) / 54 := sorry

end NUMINAMATH_GPT_veronica_reroll_probability_is_correct_l1104_110481


namespace NUMINAMATH_GPT_number_of_red_socks_l1104_110460

-- Definitions:
def red_sock_pairs (R : ℕ) := R
def red_sock_cost (R : ℕ) := 3 * R
def blue_socks_pairs : ℕ := 6
def blue_sock_cost : ℕ := 5
def total_amount_spent := 42

-- Proof Statement
theorem number_of_red_socks (R : ℕ) (h : red_sock_cost R + blue_socks_pairs * blue_sock_cost = total_amount_spent) : 
  red_sock_pairs R = 4 :=
by 
  sorry

end NUMINAMATH_GPT_number_of_red_socks_l1104_110460


namespace NUMINAMATH_GPT_prime_square_sum_eq_square_iff_l1104_110467

theorem prime_square_sum_eq_square_iff (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q):
  (∃ x y z : ℕ, p^(2*x) + q^(2*y) = z^2) ↔ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2) := 
sorry

end NUMINAMATH_GPT_prime_square_sum_eq_square_iff_l1104_110467


namespace NUMINAMATH_GPT_correct_number_of_eggs_to_buy_l1104_110482

/-- Define the total number of eggs needed and the number of eggs given by Andrew -/
def total_eggs_needed : ℕ := 222
def eggs_given_by_andrew : ℕ := 155

/-- Define a statement asserting the correct number of eggs to buy -/
def remaining_eggs_to_buy : ℕ := total_eggs_needed - eggs_given_by_andrew

/-- The statement of the proof problem -/
theorem correct_number_of_eggs_to_buy : remaining_eggs_to_buy = 67 :=
by sorry

end NUMINAMATH_GPT_correct_number_of_eggs_to_buy_l1104_110482


namespace NUMINAMATH_GPT_find_n_l1104_110405

theorem find_n (n a b : ℕ) 
  (h1 : a > 1)
  (h2 : a ∣ n)
  (h3 : b > a)
  (h4 : b ∣ n)
  (h5 : ∀ m, 1 < m ∧ m < a → ¬ m ∣ n)
  (h6 : ∀ m, a < m ∧ m < b → ¬ m ∣ n)
  (h7 : n = a^a + b^b)
  : n = 260 :=
by sorry

end NUMINAMATH_GPT_find_n_l1104_110405


namespace NUMINAMATH_GPT_exponent_problem_l1104_110406

theorem exponent_problem 
  (a : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : a > 0) 
  (h2 : a^x = 3) 
  (h3 : a^y = 5) : 
  a^(2*x + y/2) = 9 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_exponent_problem_l1104_110406


namespace NUMINAMATH_GPT_four_digit_integer_existence_l1104_110457

theorem four_digit_integer_existence :
  ∃ (a b c d : ℕ), 
    (1000 * a + 100 * b + 10 * c + d = 4522) ∧
    (a + b + c + d = 16) ∧
    (b + c = 10) ∧
    (a - d = 3) ∧
    (1000 * a + 100 * b + 10 * c + d) % 9 = 0 :=
by sorry

end NUMINAMATH_GPT_four_digit_integer_existence_l1104_110457


namespace NUMINAMATH_GPT_complete_squares_l1104_110410

def valid_solutions (x y z : ℝ) : Prop :=
  (x = 0 ∧ y = 0 ∧ z = 0) ∨
  (x = 0 ∧ y = -2 ∧ z = 0) ∨
  (x = 0 ∧ y = 0 ∧ z = 6) ∨
  (x = 0 ∧ y = -2 ∧ z = 6) ∨
  (x = 4 ∧ y = 0 ∧ z = 0) ∨
  (x = 4 ∧ y = -2 ∧ z = 0) ∨
  (x = 4 ∧ y = 0 ∧ z = 6) ∨
  (x = 4 ∧ y = -2 ∧ z = 6)

theorem complete_squares (x y z : ℝ) : 
  (x - 2)^2 + (y + 1)^2 = 5 →
  (x - 2)^2 + (z - 3)^2 = 13 →
  (y + 1)^2 + (z - 3)^2 = 10 →
  valid_solutions x y z :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_complete_squares_l1104_110410


namespace NUMINAMATH_GPT_sum_of_digits_of_4_plus_2_pow_21_l1104_110435

theorem sum_of_digits_of_4_plus_2_pow_21 :
  let x := (4 + 2)
  (x^(21) % 100).div 10 + (x^(21) % 100).mod 10 = 6 :=
by
  let x := (4 + 2)
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_4_plus_2_pow_21_l1104_110435


namespace NUMINAMATH_GPT_annual_rent_per_square_foot_l1104_110487

theorem annual_rent_per_square_foot (length width : ℕ) (monthly_rent : ℕ)
  (h_length : length = 20) (h_width : width = 15) (h_monthly_rent : monthly_rent = 3600) :
  let area := length * width
  let annual_rent := monthly_rent * 12
  let annual_rent_per_sq_ft := annual_rent / area
  annual_rent_per_sq_ft = 144 := by
  sorry

end NUMINAMATH_GPT_annual_rent_per_square_foot_l1104_110487


namespace NUMINAMATH_GPT_Agnes_age_now_l1104_110465

variable (A : ℕ) (J : ℕ := 6)

theorem Agnes_age_now :
  (2 * (J + 13) = A + 13) → A = 25 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_Agnes_age_now_l1104_110465


namespace NUMINAMATH_GPT_similar_pentagon_area_l1104_110428

theorem similar_pentagon_area
  (K1 K2 : ℝ) (L1 L2 : ℝ)
  (h_similar : true)  -- simplifying the similarity condition as true for the purpose of this example
  (h_K1 : K1 = 18)
  (h_K2 : K2 = 24)
  (h_L1 : L1 = 8.4375) :
  L2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_similar_pentagon_area_l1104_110428


namespace NUMINAMATH_GPT_trajectory_equation_l1104_110490

theorem trajectory_equation :
  ∀ (N : ℝ × ℝ), (∃ (F : ℝ × ℝ) (P : ℝ × ℝ) (M : ℝ × ℝ), 
    F = (1, 0) ∧ 
    (∃ b : ℝ, P = (0, b)) ∧ 
    (∃ a : ℝ, a ≠ 0 ∧ M = (a, 0)) ∧ 
    (N.fst = -(M.fst) ∧ N.snd = 2 * P.snd) ∧ 
    ((-M.fst) * F.fst + (-(M.snd)) * (-(P.snd)) = 0) ∧ 
    ((-M.fst, -M.snd) + (N.fst, N.snd) = (0,0))) → 
  (N.snd)^2 = 4 * (N.fst) :=
by
  intros N h
  sorry

end NUMINAMATH_GPT_trajectory_equation_l1104_110490


namespace NUMINAMATH_GPT_total_interest_rate_l1104_110485

theorem total_interest_rate (I_total I_11: ℝ) (r_9 r_11: ℝ) (h1: I_total = 100000) (h2: I_11 = 12499.999999999998) (h3: I_11 < I_total):
  r_9 = 0.09 →
  r_11 = 0.11 →
  ( ((I_total - I_11) * r_9 + I_11 * r_11) / I_total * 100 = 9.25 ) :=
by
  sorry

end NUMINAMATH_GPT_total_interest_rate_l1104_110485


namespace NUMINAMATH_GPT_min_passengers_on_vehicle_with_no_adjacent_seats_l1104_110474

-- Define the seating arrangement and adjacency rules

structure Seat :=
(row : Fin 2) (col : Fin 5)

def adjacent (a b : Seat) : Prop :=
(a.row = b.row ∧ (a.col = b.col + 1 ∨ a.col + 1 = b.col)) ∨
(a.col = b.col ∧ (a.row = b.row + 1 ∨ a.row + 1 = b.row))

def valid_seating (seated : List Seat) : Prop :=
∀ (i j : Seat), i ∈ seated → j ∈ seated → adjacent i j → false

def min_passengers : ℕ :=
5

theorem min_passengers_on_vehicle_with_no_adjacent_seats :
∃ seated : List Seat, valid_seating seated ∧ List.length seated = min_passengers :=
sorry

end NUMINAMATH_GPT_min_passengers_on_vehicle_with_no_adjacent_seats_l1104_110474


namespace NUMINAMATH_GPT_trapezoid_shorter_base_length_l1104_110431

theorem trapezoid_shorter_base_length
  (L B : ℕ)
  (hL : L = 125)
  (hB : B = 5)
  (h : ∀ x, (L - x) / 2 = B → x = 115) :
  ∃ x, x = 115 := by
    sorry

end NUMINAMATH_GPT_trapezoid_shorter_base_length_l1104_110431


namespace NUMINAMATH_GPT_mark_reading_pages_before_injury_l1104_110478

theorem mark_reading_pages_before_injury:
  ∀ (h_increased: Nat) (pages_week: Nat), 
  (h_increased = 2 + (2 * 3/2)) ∧ (pages_week = 1750) → 100 = pages_week / 7 / h_increased * 2 := 
by
  sorry

end NUMINAMATH_GPT_mark_reading_pages_before_injury_l1104_110478


namespace NUMINAMATH_GPT_remainder_of_x_to_15_plus_1_div_by_x_plus_1_is_0_l1104_110413

def f (x : ℝ) : ℝ := x^15 + 1

theorem remainder_of_x_to_15_plus_1_div_by_x_plus_1_is_0 : f (-1) = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_of_x_to_15_plus_1_div_by_x_plus_1_is_0_l1104_110413


namespace NUMINAMATH_GPT_find_multiple_of_q_l1104_110466

-- Definitions of x and y
def x (k q : ℤ) : ℤ := 55 + k * q
def y (q : ℤ) : ℤ := 4 * q + 41

-- The proof statement
theorem find_multiple_of_q (k : ℤ) : x k 7 = y 7 → k = 2 := by
  sorry

end NUMINAMATH_GPT_find_multiple_of_q_l1104_110466


namespace NUMINAMATH_GPT_pure_imaginary_condition_l1104_110468

def z1 : ℂ := 3 - 2 * Complex.I
def z2 (m : ℝ) : ℂ := 1 + m * Complex.I

theorem pure_imaginary_condition (m : ℝ) : z1 * z2 m ∈ {z : ℂ | z.re = 0} ↔ m = -3 / 2 := by
  sorry

end NUMINAMATH_GPT_pure_imaginary_condition_l1104_110468


namespace NUMINAMATH_GPT_find_share_of_A_l1104_110453

noncomputable def investment_share_A (initial_investment_A initial_investment_B withdraw_A add_B after_months end_of_year_profit : ℝ) : ℝ :=
  let investment_months_A := (initial_investment_A * after_months) + ((initial_investment_A - withdraw_A) * (12 - after_months))
  let investment_months_B := (initial_investment_B * after_months) + ((initial_investment_B + add_B) * (12 - after_months))
  let total_investment_months := investment_months_A + investment_months_B
  let ratio_A := investment_months_A / total_investment_months
  ratio_A * end_of_year_profit

theorem find_share_of_A : 
  investment_share_A 3000 4000 1000 1000 8 630 = 240 := 
by 
  sorry

end NUMINAMATH_GPT_find_share_of_A_l1104_110453


namespace NUMINAMATH_GPT_printer_a_time_l1104_110493

theorem printer_a_time :
  ∀ (A B : ℕ), 
  B = A + 4 → 
  A + B = 12 → 
  (480 / A = 120) :=
by 
  intros A B hB hAB
  sorry

end NUMINAMATH_GPT_printer_a_time_l1104_110493


namespace NUMINAMATH_GPT_min_value_of_m_l1104_110456

theorem min_value_of_m (m : ℝ) : (∀ x : ℝ, 0 < x → x ≠ ⌊x⌋ → mx < Real.log x) ↔ m = (1 / 2) * Real.log 2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_m_l1104_110456


namespace NUMINAMATH_GPT_range_of_c_l1104_110484

theorem range_of_c (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1 / a + 4 / b = 1) : ∀ c : ℝ, c < 9 → a + b > c :=
by
  sorry

end NUMINAMATH_GPT_range_of_c_l1104_110484


namespace NUMINAMATH_GPT_christina_speed_l1104_110448

theorem christina_speed
  (d v_j v_l t : ℝ)
  (D_l : ℝ)
  (h_d : d = 360)
  (h_v_j : v_j = 5)
  (h_v_l : v_l = 12)
  (h_D_l : D_l = 360)
  (h_t : t = D_l / v_l)
  (h_distance : d = v_j * t + c * t) :
  c = 7 :=
by
  sorry

end NUMINAMATH_GPT_christina_speed_l1104_110448


namespace NUMINAMATH_GPT_max_possible_n_l1104_110439

theorem max_possible_n :
  ∃ (n : ℕ), (n < 150) ∧ (∃ (k l : ℤ), n = 9 * k - 1 ∧ n = 6 * l - 5 ∧ n = 125) :=
by 
  sorry

end NUMINAMATH_GPT_max_possible_n_l1104_110439


namespace NUMINAMATH_GPT_polar_eq_parabola_l1104_110416

/-- Prove that the curve defined by the polar equation is a parabola. -/
theorem polar_eq_parabola :
  ∀ (r θ : ℝ), r = 1 / (2 * Real.sin θ + Real.cos θ) →
    ∃ (x y : ℝ), (x = r * Real.cos θ) ∧ (y = r * Real.sin θ) ∧ (x + 2 * y = r^2) :=
by 
  sorry

end NUMINAMATH_GPT_polar_eq_parabola_l1104_110416


namespace NUMINAMATH_GPT_charge_increase_percentage_l1104_110449

variable (P R G : ℝ)

def charge_relation_1 : Prop := P = 0.45 * R
def charge_relation_2 : Prop := P = 0.90 * G

theorem charge_increase_percentage (h1 : charge_relation_1 P R) (h2 : charge_relation_2 P G) : 
  (R/G - 1) * 100 = 100 :=
by
  sorry

end NUMINAMATH_GPT_charge_increase_percentage_l1104_110449


namespace NUMINAMATH_GPT_lizzy_loan_amount_l1104_110442

noncomputable def interest_rate : ℝ := 0.20
noncomputable def initial_amount : ℝ := 30
noncomputable def final_amount : ℝ := 33

theorem lizzy_loan_amount (X : ℝ) (h : initial_amount + (1 + interest_rate) * X = final_amount) : X = 2.5 := 
by
  sorry

end NUMINAMATH_GPT_lizzy_loan_amount_l1104_110442


namespace NUMINAMATH_GPT_trapezoid_perimeter_l1104_110425

theorem trapezoid_perimeter (a b : ℝ) (h : ∃ c : ℝ, a * b = c^2) :
  ∃ K : ℝ, K = 2 * (a + b + Real.sqrt (a * b)) :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_perimeter_l1104_110425


namespace NUMINAMATH_GPT_find_f_five_l1104_110422

noncomputable def f (x : ℝ) (y : ℝ) : ℝ := 2 * x^2 + y

theorem find_f_five (y : ℝ) (h : f 2 y = 50) : f 5 y = 92 := by
  sorry

end NUMINAMATH_GPT_find_f_five_l1104_110422


namespace NUMINAMATH_GPT_a_oxen_count_l1104_110472

-- Define the conditions from the problem
def total_rent : ℝ := 210
def c_share_rent : ℝ := 54
def oxen_b : ℝ := 12
def oxen_c : ℝ := 15
def months_b : ℝ := 5
def months_c : ℝ := 3
def months_a : ℝ := 7
def oxen_c_months : ℝ := oxen_c * months_c
def total_ox_months (oxen_a : ℝ) : ℝ := (oxen_a * months_a) + (oxen_b * months_b) + oxen_c_months

-- The theorem we want to prove
theorem a_oxen_count (oxen_a : ℝ) (h : c_share_rent / total_rent = oxen_c_months / total_ox_months oxen_a) :
  oxen_a = 10 := by sorry

end NUMINAMATH_GPT_a_oxen_count_l1104_110472


namespace NUMINAMATH_GPT_movement_of_hands_of_clock_involves_rotation_l1104_110443

theorem movement_of_hands_of_clock_involves_rotation (A B C D : Prop) :
  (A ↔ (∃ p : ℝ, ∃ θ : ℝ, p ≠ θ)) → -- A condition: exists a fixed point and rotation around it
  (B ↔ ¬∃ p : ℝ, ∃ θ : ℝ, p ≠ θ) → -- B condition: does not rotate around a fixed point
  (C ↔ ¬∃ p : ℝ, ∃ θ : ℝ, p ≠ θ) → -- C condition: does not rotate around a fixed point
  (D ↔ ¬∃ p : ℝ, ∃ θ : ℝ, p ≠ θ) → -- D condition: does not rotate around a fixed point
  A :=
by
  intros hA hB hC hD
  sorry

end NUMINAMATH_GPT_movement_of_hands_of_clock_involves_rotation_l1104_110443


namespace NUMINAMATH_GPT_percentage_calculation_l1104_110408

theorem percentage_calculation : 
  (0.8 * 90) = ((P / 100) * 60.00000000000001 + 30) → P = 70 := by
  sorry

end NUMINAMATH_GPT_percentage_calculation_l1104_110408


namespace NUMINAMATH_GPT_distance_apart_after_skating_l1104_110473

theorem distance_apart_after_skating :
  let Ann_speed := 6 -- Ann's speed in miles per hour
  let Glenda_speed := 8 -- Glenda's speed in miles per hour
  let skating_time := 3 -- Time spent skating in hours
  let Distance_Ann := Ann_speed * skating_time
  let Distance_Glenda := Glenda_speed * skating_time
  let Total_Distance := Distance_Ann + Distance_Glenda
  Total_Distance = 42 :=
by
  let Ann_speed := 6
  let Glenda_speed := 8
  let skating_time := 3
  let Distance_Ann := Ann_speed * skating_time
  let Distance_Glenda := Glenda_speed * skating_time
  let Total_Distance := Distance_Ann + Distance_Glenda
  sorry

end NUMINAMATH_GPT_distance_apart_after_skating_l1104_110473


namespace NUMINAMATH_GPT_gcd_1043_2295_eq_1_l1104_110414

theorem gcd_1043_2295_eq_1 : Nat.gcd 1043 2295 = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_1043_2295_eq_1_l1104_110414


namespace NUMINAMATH_GPT_selection_assignment_schemes_l1104_110401

noncomputable def number_of_selection_schemes (males females : ℕ) : ℕ :=
  if h : males + females < 3 then 0
  else
    let total3 := Nat.choose (males + females) 3
    let all_males := if hM : males < 3 then 0 else Nat.choose males 3
    let all_females := if hF : females < 3 then 0 else Nat.choose females 3
    total3 - all_males - all_females

theorem selection_assignment_schemes :
  number_of_selection_schemes 4 3 = 30 :=
by sorry

end NUMINAMATH_GPT_selection_assignment_schemes_l1104_110401


namespace NUMINAMATH_GPT_non_periodic_decimal_l1104_110497

variable {a : ℕ → ℕ}

-- Condition definitions
def is_increasing_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

def constraint (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) ≤ 10 * a n

-- Theorem statement
theorem non_periodic_decimal (a : ℕ → ℕ) 
  (h_inc : is_increasing_sequence a) 
  (h_constraint : constraint a) : 
  ¬ (∃ T : ℕ, ∀ n : ℕ, a (n + T) = a n) :=
sorry

end NUMINAMATH_GPT_non_periodic_decimal_l1104_110497


namespace NUMINAMATH_GPT_lower_bound_of_expression_l1104_110404

theorem lower_bound_of_expression :
  ∃ L : ℤ, (∀ n : ℤ, ((-1 ≤ n ∧ n ≤ 8) → (L < 4 * n + 7 ∧ 4 * n + 7 < 40))) ∧ L = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_lower_bound_of_expression_l1104_110404


namespace NUMINAMATH_GPT_ramesh_paid_price_l1104_110451

-- Define the variables based on the conditions
variable (labelledPrice transportCost installationCost sellingPrice paidPrice : ℝ)

-- Define the specific values given in the problem
def discount : ℝ := 0.20 
def profitRate : ℝ := 0.10 
def actualSellingPrice : ℝ := 24475
def transportAmount : ℝ := 125
def installationAmount : ℝ := 250

-- Define the conditions given in the problem as Lean definitions
def selling_price_no_discount (P : ℝ) : ℝ := (1 + profitRate) * P
def discounted_price (P : ℝ) : ℝ := P * (1 - discount)
def total_cost (P : ℝ) : ℝ :=  discounted_price P + transportAmount + installationAmount

-- The problem is to prove that the price Ramesh paid for the refrigerator is Rs. 18175
theorem ramesh_paid_price : 
  ∀ (labelledPrice : ℝ), 
  selling_price_no_discount labelledPrice = actualSellingPrice → 
  paidPrice = total_cost labelledPrice → 
  paidPrice = 18175 := 
by
  intros labelledPrice h1 h2 
  sorry

end NUMINAMATH_GPT_ramesh_paid_price_l1104_110451


namespace NUMINAMATH_GPT_triangle_angle_range_l1104_110475

theorem triangle_angle_range (α β γ : ℝ)
  (h1 : α + β + γ = 180)
  (h2 : α = 2 * γ)
  (h3 : α ≥ β)
  (h4 : β ≥ γ) :
  45 ≤ β ∧ β ≤ 72 := 
sorry

end NUMINAMATH_GPT_triangle_angle_range_l1104_110475


namespace NUMINAMATH_GPT_remainder_3_pow_100_plus_5_mod_8_l1104_110424

theorem remainder_3_pow_100_plus_5_mod_8 : (3^100 + 5) % 8 = 6 := by
  sorry

end NUMINAMATH_GPT_remainder_3_pow_100_plus_5_mod_8_l1104_110424


namespace NUMINAMATH_GPT_triangle_area_proof_l1104_110492

noncomputable def triangle_area (a b c : ℝ) (B : ℝ) : ℝ :=
  0.5 * a * c * Real.sin B

theorem triangle_area_proof (a b c : ℝ) (B : ℝ) (hB : B = 2 * Real.pi / 3) (hb : b = Real.sqrt 13) (h_sum : a + c = 4) :
  triangle_area a b c B = 3 * Real.sqrt 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_proof_l1104_110492


namespace NUMINAMATH_GPT_election_winner_votes_difference_l1104_110421

theorem election_winner_votes_difference :
  ∃ W S T F, F = 199 ∧ W = S + 53 ∧ W = T + 79 ∧ W + S + T + F = 979 ∧ (W - F = 105) :=
by
  sorry

end NUMINAMATH_GPT_election_winner_votes_difference_l1104_110421


namespace NUMINAMATH_GPT_parabola_directrix_l1104_110459

theorem parabola_directrix (x : ℝ) : ∃ d : ℝ, (∀ x : ℝ, 4 * x ^ 2 - 3 = d) → d = -49 / 16 :=
by
  sorry

end NUMINAMATH_GPT_parabola_directrix_l1104_110459


namespace NUMINAMATH_GPT_smallest_integer_greater_than_20_l1104_110427

noncomputable def smallest_integer_greater_than_A : ℕ :=
  let a (n : ℕ) := 4 * n - 3
  let A := Real.sqrt (a 1580) - 1 / 4
  Nat.ceil A

theorem smallest_integer_greater_than_20 :
  smallest_integer_greater_than_A = 20 :=
sorry

end NUMINAMATH_GPT_smallest_integer_greater_than_20_l1104_110427


namespace NUMINAMATH_GPT_multiple_of_people_l1104_110417

-- Define the conditions
variable (P : ℕ) -- number of people who can do the work in 8 days

-- define a function that represents the work capacity of M * P people in days, 
-- we abstract away the solving steps into one declaration.

noncomputable def work_capacity (M P : ℕ) (days : ℕ) : ℚ :=
  M * (1/8) * days

-- Set up the problem to prove that the multiple of people is 2
theorem multiple_of_people (P : ℕ) : ∃ M : ℕ, work_capacity M P 2 = 1/2 :=
by
  use 2
  unfold work_capacity
  sorry

end NUMINAMATH_GPT_multiple_of_people_l1104_110417


namespace NUMINAMATH_GPT_length_of_segment_l1104_110407

theorem length_of_segment (x : ℝ) (h₀ : 0 < x ∧ x < Real.pi / 2)
  (h₁ : 6 * Real.cos x = 5 * Real.tan x) :
  ∃ P_1 P_2 : ℝ, P_1 = 0 ∧ P_2 = (1 / 2) * Real.sin x ∧ abs (P_2 - P_1) = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_length_of_segment_l1104_110407


namespace NUMINAMATH_GPT_nonagon_line_segments_not_adjacent_l1104_110483

def nonagon_segments (n : ℕ) : ℕ :=
(n * (n - 3)) / 2

theorem nonagon_line_segments_not_adjacent (h : ∃ n, n = 9) :
  nonagon_segments 9 = 27 :=
by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_nonagon_line_segments_not_adjacent_l1104_110483
