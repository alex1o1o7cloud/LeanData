import Mathlib

namespace math_problem_l113_113731

noncomputable def a : ℝ := 3.67
noncomputable def b : ℝ := 4.83
noncomputable def c : ℝ := 2.57
noncomputable def d : ℝ := -0.12
noncomputable def x : ℝ := 7.25
noncomputable def y : ℝ := -0.55

theorem math_problem :
  (3 * a * (4 * b - 2 * y)^2) / (5 * c * d^3 * 0.5 * x) - (2 * x * y^3) / (a * b^2 * c) = -57.179729 := 
sorry

end math_problem_l113_113731


namespace first_number_positive_l113_113616

-- Define the initial condition
def initial_pair : ℕ × ℕ := (1, 1)

-- Define the allowable transformations
def transform1 (x y : ℕ) : Prop :=
(x, y - 1) = initial_pair ∨ (x + y, y + 1) = initial_pair

def transform2 (x y : ℕ) : Prop :=
(x, x * y) = initial_pair ∨ (1 / x, y) = initial_pair

-- Define discriminant function
def discriminant (a b : ℕ) : ℤ := b ^ 2 - 4 * a

-- Define the invariants maintained by the transformations
def invariant (a b : ℕ) : Prop :=
discriminant a b < 0

-- Statement to be proven
theorem first_number_positive :
(∀ (a b : ℕ), invariant a b → a > 0) :=
by
  sorry

end first_number_positive_l113_113616


namespace toothpicks_stage_20_l113_113289

-- Definition of the toothpick sequence
def toothpicks (n : ℕ) : ℕ :=
  if n = 1 then 3
  else 3 + 3 * (n - 1)

-- Theorem statement
theorem toothpicks_stage_20 : toothpicks 20 = 60 := by
  sorry

end toothpicks_stage_20_l113_113289


namespace group_size_increase_by_4_l113_113545

theorem group_size_increase_by_4
    (N : ℕ)
    (weight_old : ℕ)
    (weight_new : ℕ)
    (average_increase : ℕ)
    (weight_increase_diff : ℕ)
    (h1 : weight_old = 55)
    (h2 : weight_new = 87)
    (h3 : average_increase = 4)
    (h4 : weight_increase_diff = weight_new - weight_old)
    (h5 : average_increase * N = weight_increase_diff) :
    N = 8 :=
by
  sorry

end group_size_increase_by_4_l113_113545


namespace solve_equation_l113_113232

noncomputable def equation (x : ℝ) : Prop :=
  2021 * x = 2022 * x ^ (2021 / 2022) - 1

theorem solve_equation : ∀ x : ℝ, equation x ↔ x = 1 :=
by
  intro x
  sorry

end solve_equation_l113_113232


namespace equal_12_mn_P_2n_Q_m_l113_113143

-- Define P and Q based on given conditions
def P (m : ℕ) : ℕ := 2 ^ m
def Q (n : ℕ) : ℕ := 3 ^ n

-- The theorem to prove
theorem equal_12_mn_P_2n_Q_m (m n : ℕ) : (12 ^ (m * n)) = (P m ^ (2 * n)) * (Q n ^ m) :=
by
  -- Proof goes here
  sorry

end equal_12_mn_P_2n_Q_m_l113_113143


namespace ambiguous_times_l113_113517

theorem ambiguous_times (h m : ℝ) : 
  (∃ k l : ℕ, 0 ≤ k ∧ k < 12 ∧ 0 ≤ l ∧ l < 12 ∧ 
              (12 * h = k * 360 + m) ∧ 
              (12 * m = l * 360 + h) ∧
              k ≠ l) → 
  (∃ n : ℕ, n = 132) := 
sorry

end ambiguous_times_l113_113517


namespace problem_statement_l113_113522

theorem problem_statement (a b c d n : Nat) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < n) (h_eq : 7 * 4^n = a^2 + b^2 + c^2 + d^2) : 
  a ≥ 2^(n-1) ∧ b ≥ 2^(n-1) ∧ c ≥ 2^(n-1) ∧ d ≥ 2^(n-1) :=
sorry

end problem_statement_l113_113522


namespace minimum_value_expr_l113_113357

variable (a b : ℝ)

theorem minimum_value_expr (h1 : 0 < a) (h2 : 0 < b) : 
  (a + 1 / b) * (a + 1 / b - 1009) + (b + 1 / a) * (b + 1 / a - 1009) ≥ -509004.5 :=
sorry

end minimum_value_expr_l113_113357


namespace sufficient_but_not_necessary_for_circle_l113_113610

theorem sufficient_but_not_necessary_for_circle (m : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 - 4 * x + 2 * y + m = 0) → (m = 0) → (∃ (a b : ℝ), (x - a)^2 + (y - b)^2 = r^2)) ∧
  ¬(∀ (x y : ℝ), (x^2 + y^2 - 4 * x + 2 * y + m = 0) → (∃ (a b : ℝ), (x - a)^2 + (y - b)^2 = r^2) → (m = 0)) := sorry

end sufficient_but_not_necessary_for_circle_l113_113610


namespace trig_sum_identity_l113_113524

theorem trig_sum_identity :
  Real.sin (47 * Real.pi / 180) * Real.cos (43 * Real.pi / 180) 
  + Real.sin (137 * Real.pi / 180) * Real.sin (43 * Real.pi / 180) = 1 :=
by
  sorry

end trig_sum_identity_l113_113524


namespace ellipse_circle_parallelogram_condition_l113_113664

theorem ellipse_circle_parallelogram_condition
  (a b : ℝ)
  (C₀ : ∀ x y : ℝ, x^2 + y^2 = 1)
  (C₁ : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (h : a > 0 ∧ b > 0 ∧ a > b) :
  1 / a^2 + 1 / b^2 = 1 := by
  sorry

end ellipse_circle_parallelogram_condition_l113_113664


namespace set_intersection_complement_l113_113725

/-- Definition of the universal set U. -/
def U := ({1, 2, 3, 4, 5} : Set ℕ)

/-- Definition of the set M. -/
def M := ({3, 4, 5} : Set ℕ)

/-- Definition of the set N. -/
def N := ({2, 3} : Set ℕ)

/-- Statement of the problem to be proven. -/
theorem set_intersection_complement :
  ((U \ N) ∩ M) = ({4, 5} : Set ℕ) :=
by
  sorry

end set_intersection_complement_l113_113725


namespace find_common_chord_l113_113371

-- Define the two circles
def C1 (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def C2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 2 = 0

-- The common chord is the line we need to prove
def CommonChord (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- The theorem stating that the common chord is the line x + 2*y - 1 = 0
theorem find_common_chord (x y : ℝ) (p : C1 x y ∧ C2 x y) : CommonChord x y :=
sorry

end find_common_chord_l113_113371


namespace minimize_cost_per_km_l113_113966

section ship_cost_minimization

variables (u v k : ℝ) (fuel_cost other_cost total_cost_per_km: ℝ)

-- Condition 1: The fuel cost per unit time is directly proportional to the cube of its speed.
def fuel_cost_eq : Prop := u = k * v^3

-- Condition 2: When the speed of the ship is 10 km/h, the fuel cost is 35 yuan per hour.
def fuel_cost_at_10 : Prop := u = 35 ∧ v = 10

-- Condition 3: The other costs are 560 yuan per hour.
def other_cost_eq : Prop := other_cost = 560

-- Condition 4: The maximum speed of the ship is 25 km/h.
def max_speed : Prop := v ≤ 25

-- Prove that the speed of the ship that minimizes the cost per kilometer is 20 km/h.
theorem minimize_cost_per_km : 
  fuel_cost_eq u v k ∧ fuel_cost_at_10 u v ∧ other_cost_eq other_cost ∧ max_speed v → v = 20 :=
by
  sorry

end ship_cost_minimization

end minimize_cost_per_km_l113_113966


namespace find_coordinates_of_b_l113_113771

theorem find_coordinates_of_b
  (x y : ℝ)
  (a : ℂ) (b : ℂ)
  (sqrt3 sqrt5 sqrt10 sqrt6 : ℝ)
  (h1 : sqrt3 = Real.sqrt 3)
  (h2 : sqrt5 = Real.sqrt 5)
  (h3 : sqrt10 = Real.sqrt 10)
  (h4 : sqrt6 = Real.sqrt 6)
  (h5 : a = ⟨sqrt3, sqrt5⟩)
  (h6 : ∃ x y : ℝ, b = ⟨x, y⟩ ∧ (sqrt3 * x + sqrt5 * y = 0) ∧ (Real.sqrt (x^2 + y^2) = 2))
  : b = ⟨- sqrt10 / 2, sqrt6 / 2⟩ ∨ b = ⟨sqrt10 / 2, - sqrt6 / 2⟩ := 
  sorry

end find_coordinates_of_b_l113_113771


namespace parcel_cost_l113_113163

theorem parcel_cost (P : ℤ) (hP : P ≥ 1) : 
  (P ≤ 5 → C = 15 + 4 * (P - 1)) ∧ (P > 5 → C = 15 + 4 * (P - 1) - 10) :=
sorry

end parcel_cost_l113_113163


namespace number_of_girls_l113_113849

-- Define the number of boys and girls as natural numbers
variable (B G : ℕ)

-- First condition: The number of girls is 458 more than the number of boys
axiom h1 : G = B + 458

-- Second condition: The total number of pupils is 926
axiom h2 : G + B = 926

-- The theorem to be proved: The number of girls is 692
theorem number_of_girls : G = 692 := by
  sorry

end number_of_girls_l113_113849


namespace exists_disk_of_radius_one_containing_1009_points_l113_113387

theorem exists_disk_of_radius_one_containing_1009_points
  (points : Fin 2017 → ℝ × ℝ)
  (h : ∀ (a b c : Fin 2017), (dist (points a) (points b) < 1) ∨ (dist (points b) (points c) < 1) ∨ (dist (points c) (points a) < 1)) :
  ∃ (center : ℝ × ℝ), ∃ (sub_points : Finset (Fin 2017)), sub_points.card ≥ 1009 ∧ ∀ p ∈ sub_points, dist (center) (points p) ≤ 1 :=
sorry

end exists_disk_of_radius_one_containing_1009_points_l113_113387


namespace pages_copied_l113_113583

theorem pages_copied (cost_per_page : ℕ) (amount_in_dollars : ℕ)
    (cents_per_dollar : ℕ) (total_cents : ℕ) 
    (pages : ℕ)
    (h1 : cost_per_page = 3)
    (h2 : amount_in_dollars = 25)
    (h3 : cents_per_dollar = 100)
    (h4 : total_cents = amount_in_dollars * cents_per_dollar)
    (h5 : total_cents = 2500)
    (h6 : pages = total_cents / cost_per_page) :
  pages = 833 := 
sorry

end pages_copied_l113_113583


namespace majka_numbers_product_l113_113532

/-- Majka created a three-digit funny and a three-digit cheerful number.
    - The funny number starts with an odd digit and alternates between odd and even.
    - The cheerful number starts with an even digit and alternates between even and odd.
    - All digits are distinct and nonzero.
    - The sum of these two numbers is 1617.
    - The product of these two numbers ends in 40.
    Prove that the product of these numbers is 635040.
-/
theorem majka_numbers_product :
  ∃ (a b c : ℕ) (D E F : ℕ),
    -- Define 3-digit funny number as (100 * a + 10 * b + c)
    -- with a and c odd, b even, and all distinct and nonzero
    (a % 2 = 1) ∧ (c % 2 = 1) ∧ (b % 2 = 0) ∧
    -- Define 3-digit cheerful number as (100 * D + 10 * E + F)
    -- with D and F even, E odd, and all distinct and nonzero
    (D % 2 = 0) ∧ (F % 2 = 0) ∧ (E % 2 = 1) ∧
    -- All digits are distinct and nonzero
    (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ D ≠ 0 ∧ E ≠ 0 ∧ F ≠ 0) ∧
    (a ≠ b ∧ a ≠ c ∧ a ≠ D ∧ a ≠ E ∧ a ≠ F ∧
     b ≠ c ∧ b ≠ D ∧ b ≠ E ∧ b ≠ F ∧
     c ≠ D ∧ c ≠ E ∧ c ≠ F ∧
     D ≠ E ∧ D ≠ F ∧ E ≠ F) ∧
    (100 * a + 10 * b + c + 100 * D + 10 * E + F = 1617) ∧
    ((100 * a + 10 * b + c) * (100 * D + 10 * E + F) = 635040) := sorry

end majka_numbers_product_l113_113532


namespace dennis_initial_money_l113_113479

def initial_money (shirt_cost: ℕ) (ten_dollar_bills: ℕ) (loose_coins: ℕ) : ℕ :=
  shirt_cost + (10 * ten_dollar_bills) + loose_coins

theorem dennis_initial_money : initial_money 27 2 3 = 50 :=
by 
  -- Here would go the proof steps based on the solution steps identified before
  sorry

end dennis_initial_money_l113_113479


namespace work_days_in_week_l113_113750

theorem work_days_in_week (total_toys_per_week : ℕ) (toys_produced_each_day : ℕ) (h1 : total_toys_per_week = 6500) (h2 : toys_produced_each_day = 1300) : 
  total_toys_per_week / toys_produced_each_day = 5 :=
by
  sorry

end work_days_in_week_l113_113750


namespace roots_condition_l113_113936

theorem roots_condition (m : ℝ) (f : ℝ → ℝ) (x1 x2 : ℝ) (h_f : ∀ x, f x = x^2 + 2*(m - 1)*x - 5*m - 2) 
  (h_roots : ∃ x1 x2, x1 < 1 ∧ 1 < x2 ∧ f x1 = 0 ∧ f x2 = 0) : 
  m > 1 := 
by
  sorry

end roots_condition_l113_113936


namespace find_y_coordinate_of_Q_l113_113361

noncomputable def y_coordinate_of_Q 
  (P R T S : ℝ × ℝ) (Q : ℝ × ℝ) (areaPentagon areaSquare : ℝ) : Prop :=
  P = (0, 0) ∧ 
  R = (0, 5) ∧ 
  T = (6, 0) ∧ 
  S = (6, 5) ∧ 
  Q.fst = 3 ∧ 
  areaSquare = 25 ∧ 
  areaPentagon = 50 ∧ 
  (1 / 2) * 6 * (Q.snd - 5) + areaSquare = areaPentagon

theorem find_y_coordinate_of_Q : 
  ∃ y_Q : ℝ, y_coordinate_of_Q (0, 0) (0, 5) (6, 0) (6, 5) (3, y_Q) 50 25 ∧ y_Q = 40 / 3 :=
sorry

end find_y_coordinate_of_Q_l113_113361


namespace total_wire_length_l113_113744

theorem total_wire_length (S : ℕ) (L : ℕ)
  (hS : S = 20) 
  (hL : L = 2 * S) : S + L = 60 :=
by
  sorry

end total_wire_length_l113_113744


namespace simplify_expression_l113_113165

theorem simplify_expression {x a : ℝ} (h1 : x > a) (h2 : x ≠ 0) (h3 : a ≠ 0) :
  (x * (x^2 - a^2)⁻¹ + 1) / (a * (x - a)⁻¹ + (x - a)^(1 / 2))
  / ((a^2 * (x + a)^(1 / 2)) / (x - (x^2 - a^2)^(1 / 2)) + 1 / (x^2 - a * x))
  = 2 / (x^2 - a^2) :=
by sorry

end simplify_expression_l113_113165


namespace intersection_complement_eq_l113_113829

-- Define the universal set U, and sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {4, 5}

-- Define the complement of B in U
def complement_B_in_U : Set ℕ := { x ∈ U | x ∉ B }

-- The main theorem statement stating the required equality
theorem intersection_complement_eq : A ∩ complement_B_in_U = {2, 3} := by
  sorry

end intersection_complement_eq_l113_113829


namespace circles_internally_tangent_l113_113360

theorem circles_internally_tangent :
  ∀ (x y : ℝ),
  (x^2 + y^2 - 6 * x + 4 * y + 12 = 0) ∧ (x^2 + y^2 - 14 * x - 2 * y + 14 = 0) →
  ∃ (C1 C2 : ℝ × ℝ) (r1 r2 : ℝ),
  C1 = (3, -2) ∧ r1 = 1 ∧
  C2 = (7, 1) ∧ r2 = 6 ∧
  dist C1 C2 = r2 - r1 :=
by
  sorry

end circles_internally_tangent_l113_113360


namespace one_of_18_consecutive_is_divisible_l113_113198

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define what it means for one number to be divisible by another
def divisible (a b : ℕ) : Prop :=
  b ≠ 0 ∧ a % b = 0

-- The main theorem
theorem one_of_18_consecutive_is_divisible : 
  ∀ (n : ℕ), 100 ≤ n ∧ n + 17 ≤ 999 → ∃ (k : ℕ), n ≤ k ∧ k ≤ (n + 17) ∧ divisible k (sum_of_digits k) :=
by
  intros n h
  sorry

end one_of_18_consecutive_is_divisible_l113_113198


namespace quadratic_roots_l113_113648

theorem quadratic_roots (m n p : ℕ) (h : m.gcd p = 1) 
  (h1 : 3 * m^2 - 8 * m * p + p^2 = p^2 * n) : n = 13 :=
by sorry

end quadratic_roots_l113_113648


namespace q_is_false_l113_113365

-- Given conditions
variables (p q : Prop)
axiom h1 : ¬ (p ∧ q)
axiom h2 : ¬ ¬ p

-- Proof that q is false
theorem q_is_false : q = False :=
by
  sorry

end q_is_false_l113_113365


namespace geometric_sequence_seventh_term_l113_113686

theorem geometric_sequence_seventh_term :
  let a := 6
  let r := -2
  (a * r^(7 - 1)) = 384 := 
by
  sorry

end geometric_sequence_seventh_term_l113_113686


namespace triangle_equilateral_of_constraints_l113_113767

theorem triangle_equilateral_of_constraints {a b c : ℝ}
  (h1 : a^4 = b^4 + c^4 - b^2 * c^2)
  (h2 : b^4 = c^4 + a^4 - a^2 * c^2) : 
  a = b ∧ b = c :=
by 
  sorry

end triangle_equilateral_of_constraints_l113_113767


namespace total_puzzle_pieces_l113_113590

theorem total_puzzle_pieces : 
  ∀ (p1 p2 p3 : ℕ), 
  p1 = 1000 → 
  p2 = p1 + p1 / 2 → 
  p3 = p1 + p1 / 2 → 
  p1 + p2 + p3 = 4000 := 
by 
  intros p1 p2 p3 
  intro h1 
  intro h2 
  intro h3 
  rw [h1, h2, h3] 
  norm_num
  sorry

end total_puzzle_pieces_l113_113590


namespace arithmetic_sequence_6000th_term_l113_113434

theorem arithmetic_sequence_6000th_term :
  ∀ (p r : ℕ), 
  (2 * p) = 2 * p → 
  (2 * p + 2 * r = 14) → 
  (14 + 2 * r = 4 * p - r) → 
  (2 * p + (6000 - 1) * 4 = 24006) :=
by 
  intros p r h h1 h2
  sorry

end arithmetic_sequence_6000th_term_l113_113434


namespace distance_between_Q_and_R_l113_113892

noncomputable def distance_QR : ℝ :=
  let DE : ℝ := 9
  let EF : ℝ := 12
  let DF : ℝ := 15
  let N : ℝ := 7.5
  let QF : ℝ := (N * DF) / EF
  let QD : ℝ := DF - QF
  let QR : ℝ := (QD * DF) / EF
  QR

theorem distance_between_Q_and_R 
  (DE EF DF N QF QD QR : ℝ )
  (h1 : DE = 9)
  (h2 : EF = 12)
  (h3 : DF = 15)
  (h4 : N = DF / 2)
  (h5 : QF = N * DF / EF)
  (h6 : QD = DF - QF)
  (h7 : QR = QD * DF / EF) :
  QR = 7.03125 :=
by
  sorry

end distance_between_Q_and_R_l113_113892


namespace find_f_3_l113_113634

def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

theorem find_f_3 (a b : ℝ) (h : f (-3) a b = 10) : f 3 a b = -26 :=
by sorry

end find_f_3_l113_113634


namespace unfolded_side_view_of_cone_is_sector_l113_113946

theorem unfolded_side_view_of_cone_is_sector 
  (shape : Type)
  (curved_side : shape)
  (straight_side1 : shape)
  (straight_side2 : shape) 
  (condition1 : ∃ (s : shape), s = curved_side) 
  (condition2 : ∃ (s1 s2 : shape), s1 = straight_side1 ∧ s2 = straight_side2)
  : shape = sector :=
sorry

end unfolded_side_view_of_cone_is_sector_l113_113946


namespace identity_proof_l113_113466

theorem identity_proof (x y : ℝ) (h1 : x + y = 5 / 11) (h2 : x - y = 1 / 55) : x^2 - y^2 = 1 / 121 :=
by 
  sorry

end identity_proof_l113_113466


namespace right_triangle_other_angle_l113_113043

theorem right_triangle_other_angle (a b c : ℝ) 
  (h_triangle_sum : a + b + c = 180) 
  (h_right_angle : a = 90) 
  (h_acute_angle : b = 60) : 
  c = 30 :=
by
  sorry

end right_triangle_other_angle_l113_113043


namespace gcd_fa_fb_l113_113752

def f (x : ℤ) : ℤ := x * x - x + 2008

def a : ℤ := 102
def b : ℤ := 103

theorem gcd_fa_fb : Int.gcd (f a) (f b) = 2 := by
  sorry

end gcd_fa_fb_l113_113752


namespace range_of_a_l113_113808

def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

noncomputable def roots (a : ℝ) : (ℝ × ℝ) :=
  (1, 3)

noncomputable def f_max (a : ℝ) :=
  -a

theorem range_of_a (a b c : ℝ) 
  (h1 : ∀ x, quadratic_function a b c x < 0 ↔ (x < 1 ∨ 3 < x))
  (h2 : f_max a < 2) : 
  -2 < a ∧ a < 0 :=
sorry

end range_of_a_l113_113808


namespace price_of_expensive_feed_l113_113577

theorem price_of_expensive_feed
  (total_weight : ℝ)
  (mix_price_per_pound : ℝ)
  (cheaper_feed_weight : ℝ)
  (cheaper_feed_price_per_pound : ℝ)
  (expensive_feed_price_per_pound : ℝ) :
  total_weight = 27 →
  mix_price_per_pound = 0.26 →
  cheaper_feed_weight = 14.2105263158 →
  cheaper_feed_price_per_pound = 0.17 →
  expensive_feed_price_per_pound = 0.36 :=
by
  intros h1 h2 h3 h4
  sorry

end price_of_expensive_feed_l113_113577


namespace range_of_a_l113_113855

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 1 then 2^x + 1 else -x^2 + a * x

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, f x a < 3) ↔ (2 ≤ a ∧ a < 2 * Real.sqrt 3) := by
  sorry

end range_of_a_l113_113855


namespace person_y_speed_in_still_water_l113_113314

theorem person_y_speed_in_still_water 
    (speed_x_in_still_water : ℝ)
    (time_meeting_towards_each_other : ℝ)
    (time_catching_up_same_direction: ℝ)
    (distance_upstream_meeting: ℝ)
    (distance_downstream_meeting: ℝ)
    (total_distance: ℝ) :
    speed_x_in_still_water = 6 →
    time_meeting_towards_each_other = 4 →
    time_catching_up_same_direction = 16 →
    distance_upstream_meeting = 4 * (6 - distance_upstream_meeting) + 4 * (10 + distance_downstream_meeting) →
    distance_downstream_meeting = 4 * (6 + distance_upstream_meeting) →
    total_distance = 4 * (6 + 10) →
    ∃ (speed_y_in_still_water : ℝ), speed_y_in_still_water = 10 :=
by
  intros h_speed_x h_time_meeting h_time_catching h_distance_upstream h_distance_downstream h_total_distance
  sorry

end person_y_speed_in_still_water_l113_113314


namespace cost_of_eight_books_l113_113014

theorem cost_of_eight_books (x : ℝ) (h : 2 * x = 34) : 8 * x = 136 :=
by
  sorry

end cost_of_eight_books_l113_113014


namespace original_number_of_men_l113_113562

theorem original_number_of_men (x : ℕ) (h : 10 * x = 7 * (x + 10)) : x = 24 := 
by 
  -- Add your proof here 
  sorry

end original_number_of_men_l113_113562


namespace circles_intersect_condition_l113_113509

theorem circles_intersect_condition (a : ℝ) (ha : a > 0) : 
  (∃ x y : ℝ, x^2 + y^2 = 1 ∧ (x - a)^2 + y^2 = 16) ↔ 3 < a ∧ a < 5 :=
by sorry

end circles_intersect_condition_l113_113509


namespace machine_produces_one_item_in_40_seconds_l113_113549

theorem machine_produces_one_item_in_40_seconds :
  (60 * 1) / 90 * 60 = 40 :=
by
  sorry

end machine_produces_one_item_in_40_seconds_l113_113549


namespace total_hair_cut_l113_113804

-- Definitions from conditions
def first_cut : ℝ := 0.375
def second_cut : ℝ := 0.5

-- The theorem stating the math problem
theorem total_hair_cut : first_cut + second_cut = 0.875 := by
  sorry

end total_hair_cut_l113_113804


namespace bungee_cord_extension_l113_113719

variables (m g H k h L₀ T_max : ℝ)
  (mass_nonzero : m ≠ 0)
  (gravity_positive : g > 0)
  (H_positive : H > 0)
  (k_positive : k > 0)
  (L₀_nonnegative : L₀ ≥ 0)
  (T_max_eq : T_max = 4 * m * g)
  (L_eq : L₀ + h = H)
  (hooke_eq : T_max = k * h)

theorem bungee_cord_extension :
  h = H / 2 := sorry

end bungee_cord_extension_l113_113719


namespace closest_to_zero_is_13_l113_113156

noncomputable def a (n : ℕ) : ℤ := 88 - 7 * n

theorem closest_to_zero_is_13 : ∀ (n : ℕ), 1 ≤ n → 81 + (n - 1) * (-7) = a n →
  (∀ m : ℕ, (m : ℤ) ≤ (88 : ℤ) / 7 → abs (a m) > abs (a 13)) :=
  sorry

end closest_to_zero_is_13_l113_113156


namespace sqrt_two_irrational_l113_113268

theorem sqrt_two_irrational :
  ¬ ∃ (a b : ℕ), (a.gcd b = 1) ∧ (b ≠ 0) ∧ (a^2 = 2 * b^2) :=
sorry

end sqrt_two_irrational_l113_113268


namespace no_solution_if_n_eq_neg_one_l113_113119

theorem no_solution_if_n_eq_neg_one (n x y z : ℝ) :
  (n * x + y + z = 2) ∧ (x + n * y + z = 2) ∧ (x + y + n * z = 2) ↔ n = -1 → false :=
by
  sorry

end no_solution_if_n_eq_neg_one_l113_113119


namespace find_largest_number_l113_113348

theorem find_largest_number (a b c : ℝ) (h1 : a + b + c = 100) (h2 : c - b = 10) (h3 : b - a = 5) : c = 41.67 := 
sorry

end find_largest_number_l113_113348


namespace number_of_boys_l113_113389

-- Definitions based on conditions
def students_in_class : ℕ := 30
def cups_brought_total : ℕ := 90
def cups_per_boy : ℕ := 5

-- Definition of boys and girls, with a constraint from the conditions
variable (B : ℕ)
def girls_in_class (B : ℕ) : ℕ := 2 * B

-- Properties from the conditions
axiom h1 : B + girls_in_class B = students_in_class
axiom h2 : B * cups_per_boy = cups_brought_total - (students_in_class - B) * 0 -- Assume no girl brought any cup

-- We state the question as a theorem to be proved
theorem number_of_boys (B : ℕ) : B = 10 :=
by
  sorry

end number_of_boys_l113_113389


namespace range_of_m_l113_113834

theorem range_of_m (m : ℝ) 
  (h : ∀ x : ℝ, 0 < x → m * x^2 + 2 * x + m ≤ 0) : m ≤ -1 :=
sorry

end range_of_m_l113_113834


namespace calc_expression_l113_113436

theorem calc_expression : 2012 * 2016 - 2014^2 = -4 := by
  sorry

end calc_expression_l113_113436


namespace mixed_number_eval_l113_113385

theorem mixed_number_eval :
  -|-(18/5 : ℚ)| - (- (12 /5 : ℚ)) + (4/5 : ℚ) = - (2 / 5 : ℚ) :=
by
  sorry

end mixed_number_eval_l113_113385


namespace perimeter_T2_l113_113870

def Triangle (a b c : ℝ) :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem perimeter_T2 (a b c : ℝ) (h : Triangle a b c) (ha : a = 10) (hb : b = 15) (hc : c = 20) : 
  let AM := a / 2
  let BN := b / 2
  let CP := c / 2
  0 < AM ∧ 0 < BN ∧ 0 < CP →
  AM + BN + CP = 22.5 :=
by
  sorry

end perimeter_T2_l113_113870


namespace inequality_proof_l113_113878

theorem inequality_proof
  (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : (6841 * x - 1) / 9973 + (9973 * y - 1) / 6841 = z) :
  x / 9973 + y / 6841 > 1 :=
sorry

end inequality_proof_l113_113878


namespace brads_running_speed_proof_l113_113105

noncomputable def brads_speed (distance_between_homes : ℕ) (maxwells_speed : ℕ) (maxwells_time : ℕ) (brad_start_delay : ℕ) : ℕ :=
  let distance_covered_by_maxwell := maxwells_speed * maxwells_time
  let distance_covered_by_brad := distance_between_homes - distance_covered_by_maxwell
  let brads_time := maxwells_time - brad_start_delay
  distance_covered_by_brad / brads_time

theorem brads_running_speed_proof :
  brads_speed 54 4 6 1 = 6 := 
by
  unfold brads_speed
  rfl

end brads_running_speed_proof_l113_113105


namespace ratio_of_shaded_area_l113_113286

-- Definitions
variable (S : Type) [Field S]
variable (square_area shaded_area : S) -- Areas of the square and the shaded regions.
variable (PX XQ : S) -- Lengths such that PX = 3 * XQ.

-- Conditions
axiom condition1 : PX = 3 * XQ
axiom condition2 : shaded_area / square_area = 0.375

-- Goal
theorem ratio_of_shaded_area (PX XQ square_area shaded_area : S) [Field S] 
  (condition1 : PX = 3 * XQ)
  (condition2 : shaded_area / square_area = 0.375) : shaded_area / square_area = 0.375 := 
  by
  sorry

end ratio_of_shaded_area_l113_113286


namespace find_girls_l113_113211

theorem find_girls (n : ℕ) (h : 1 - (1 / Nat.choose (3 + n) 3) = 34 / 35) : n = 4 :=
  sorry

end find_girls_l113_113211


namespace inequality_proof_l113_113426

theorem inequality_proof
  (n : ℕ) (hn : n ≥ 3) (x y z : ℝ) (hxyz_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (hxyz_sum : x + y + z = 1) :
  (1 / x^(n-1) - x) * (1 / y^(n-1) - y) * (1 / z^(n-1) - z) ≥ ((3^n - 1) / 3)^3 :=
by sorry

end inequality_proof_l113_113426


namespace larger_solution_quadratic_l113_113952

theorem larger_solution_quadratic (x : ℝ) : x^2 - 13 * x + 42 = 0 → x = 7 ∨ x = 6 ∧ x > 6 :=
by
  sorry

end larger_solution_quadratic_l113_113952


namespace rectangles_single_row_7_rectangles_grid_7_4_l113_113366

def rectangles_in_single_row (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

theorem rectangles_single_row_7 :
  rectangles_in_single_row 7 = 28 :=
by
  -- Add the proof here
  sorry

def rectangles_in_grid (rows cols : ℕ) : ℕ :=
  ((cols + 1) * cols / 2) * ((rows + 1) * rows / 2)

theorem rectangles_grid_7_4 :
  rectangles_in_grid 4 7 = 280 :=
by
  -- Add the proof here
  sorry

end rectangles_single_row_7_rectangles_grid_7_4_l113_113366


namespace sum_of_odd_integers_less_than_50_l113_113961

def sumOddIntegersLessThan (n : Nat) : Nat :=
  List.sum (List.filter (λ x => x % 2 = 1) (List.range n))

theorem sum_of_odd_integers_less_than_50 : sumOddIntegersLessThan 50 = 625 :=
  by
    sorry

end sum_of_odd_integers_less_than_50_l113_113961


namespace valid_square_numbers_l113_113671

noncomputable def is_valid_number (N P Q : ℕ) (q : ℕ) : Prop :=
  N = P * 10^q + Q ∧ N = 2 * P * Q

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem valid_square_numbers : 
  ∀ (N : ℕ), (∃ (P Q : ℕ) (q : ℕ), is_valid_number N P Q q) → is_perfect_square N :=
sorry

end valid_square_numbers_l113_113671


namespace triangle_classification_l113_113068

def is_obtuse_triangle (a b c : ℕ) : Prop :=
c^2 > a^2 + b^2 ∧ a < b ∧ b < c

def is_right_triangle (a b c : ℕ) : Prop :=
c^2 = a^2 + b^2 ∧ a < b ∧ b < c

def is_acute_triangle (a b c : ℕ) : Prop :=
c^2 < a^2 + b^2 ∧ a < b ∧ b < c

theorem triangle_classification :
    is_acute_triangle 10 12 14 ∧ 
    is_right_triangle 10 24 26 ∧ 
    is_obtuse_triangle 4 6 8 :=
by 
  sorry

end triangle_classification_l113_113068


namespace initial_books_correct_l113_113667

def sold_books : ℕ := 78
def left_books : ℕ := 37
def initial_books : ℕ := sold_books + left_books

theorem initial_books_correct : initial_books = 115 := by
  sorry

end initial_books_correct_l113_113667


namespace roots_of_quadratic_l113_113635

theorem roots_of_quadratic (a b : ℝ) (h₁ : a + b = 2) (h₂ : a * b = -3) : a^2 + b^2 = 10 := 
by
  -- proof steps go here, but not required as per the instruction
  sorry

end roots_of_quadratic_l113_113635


namespace solve_inequality_l113_113861

theorem solve_inequality (x : ℝ) (hx : x ≥ 0) :
  2021 * (x ^ (2020/202)) - 1 ≥ 2020 * x ↔ x = 1 :=
by sorry

end solve_inequality_l113_113861


namespace find_annual_interest_rate_l113_113125

noncomputable def compound_interest (P A : ℝ) (r : ℝ) (n t : ℕ) :=
  A = P * (1 + r / n) ^ (n * t)

theorem find_annual_interest_rate
  (P A : ℝ) (t n : ℕ) (r : ℝ)
  (hP : P = 6000)
  (hA : A = 6615)
  (ht : t = 2)
  (hn : n = 1)
  (hr : compound_interest P A r n t) :
  r = 0.05 :=
sorry

end find_annual_interest_rate_l113_113125


namespace minimize_expression_at_c_l113_113326

theorem minimize_expression_at_c (c : ℝ) : (c = 7 / 4) → (∀ x : ℝ, 2 * c^2 - 7 * c + 4 ≤ 2 * x^2 - 7 * x + 4) :=
sorry

end minimize_expression_at_c_l113_113326


namespace subtract_30_divisible_l113_113860

theorem subtract_30_divisible (n : ℕ) (d : ℕ) (r : ℕ) 
  (h1 : n = 13602) (h2 : d = 87) (h3 : r = 30) 
  (h4 : n % d = r) : (n - r) % d = 0 :=
by
  -- Skipping the proof as it's not required
  sorry

end subtract_30_divisible_l113_113860


namespace problem_eval_expression_l113_113651

theorem problem_eval_expression :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) = 5^128 - 4^128 :=
by
  sorry

end problem_eval_expression_l113_113651


namespace f_neg_9_over_2_f_in_7_8_l113_113377

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 1 then x / (x + 1) else sorry

theorem f_neg_9_over_2 (h : ∀ x : ℝ, f (x + 2) = f x) (h_odd : ∀ x : ℝ, f (-x) = -f x) : 
  f (-9 / 2) = -1 / 3 :=
by
  have h_period : f (-9 / 2) = f (-1 / 2) := by
    sorry  -- Using periodicity property
  have h_odd1 : f (-1 / 2) = -f (1 / 2) := by
    sorry  -- Using odd function property
  have h_def : f (1 / 2) = 1 / 3 := by
    sorry  -- Using the definition of f(x) for x in [0, 1)
  rw [h_period, h_odd1, h_def]
  norm_num

theorem f_in_7_8 (h : ∀ x : ℝ, f (x + 2) = f x) (h_odd : ∀ x : ℝ, f (-x) = -f x) :
  ∀ x : ℝ, (7 < x ∧ x ≤ 8) → f x = - (x - 8) / (x - 9) :=
by
  intro x hx
  have h_period : f x = f (x - 8) := by
    sorry  -- Using periodicity property
  sorry  -- Apply the negative intervals and substitution to achieve the final form

end f_neg_9_over_2_f_in_7_8_l113_113377


namespace sum_of_numbers_with_lcm_and_ratio_l113_113765

theorem sum_of_numbers_with_lcm_and_ratio 
  (a b : ℕ) 
  (h_lcm : Nat.lcm a b = 48)
  (h_ratio : a / Nat.gcd a b = 2 ∧ b / Nat.gcd a b = 3) : 
  a + b = 80 := 
by sorry

end sum_of_numbers_with_lcm_and_ratio_l113_113765


namespace part_i_part_ii_part_iii_l113_113843

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4) + f (x + 3 * Real.pi / 4)

theorem part_i : f (Real.pi / 2) = 1 :=
sorry

theorem part_ii : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 2 * Real.pi :=
sorry

theorem part_iii : ∃ x, g x = -2 :=
sorry

end part_i_part_ii_part_iii_l113_113843


namespace group_B_equal_l113_113867

noncomputable def neg_two_pow_three := (-2)^3
noncomputable def minus_two_pow_three := -(2^3)

theorem group_B_equal : neg_two_pow_three = minus_two_pow_three :=
by sorry

end group_B_equal_l113_113867


namespace sphere_surface_area_increase_l113_113906

theorem sphere_surface_area_increase (V A : ℝ) (r : ℝ)
  (hV : V = (4/3) * π * r^3)
  (hA : A = 4 * π * r^2)
  : (∃ r', (V = 8 * ((4/3) * π * r'^3)) ∧ (∃ A', A' = 4 * A)) :=
by
  sorry

end sphere_surface_area_increase_l113_113906


namespace mary_starting_weight_l113_113683

def initial_weight (final_weight lost_1 gained_2 lost_3 gained_4 : ℕ) : ℕ :=
  final_weight + (lost_3 - gained_4) + (gained_2 - lost_1) + lost_1

theorem mary_starting_weight :
  ∀ (final_weight lost_1 gained_2 lost_3 gained_4 : ℕ),
  final_weight = 81 →
  lost_1 = 12 →
  gained_2 = 2 * lost_1 →
  lost_3 = 3 * lost_1 →
  gained_4 = lost_1 / 2 →
  initial_weight final_weight lost_1 gained_2 lost_3 gained_4 = 99 :=
by
  intros final_weight lost_1 gained_2 lost_3 gained_4 h_final_weight h_lost_1 h_gained_2 h_lost_3 h_gained_4
  rw [h_final_weight, h_lost_1] at *
  rw [h_gained_2, h_lost_3, h_gained_4]
  unfold initial_weight
  sorry

end mary_starting_weight_l113_113683


namespace age_problem_l113_113656

theorem age_problem (x y : ℕ) 
  (h1 : 3 * x = 4 * y) 
  (h2 : 3 * y - x = 140) : x = 112 ∧ y = 84 := 
by 
  sorry

end age_problem_l113_113656


namespace correct_calculation_l113_113857

variable (a b : ℕ)

theorem correct_calculation : 3 * a * b - 2 * a * b = a * b := 
by sorry

end correct_calculation_l113_113857


namespace smallest_positive_integer_l113_113337

theorem smallest_positive_integer (m n : ℤ) : ∃ k : ℕ, k > 0 ∧ (∃ m n : ℤ, k = 5013 * m + 111111 * n) ∧ k = 3 :=
by {
  sorry 
}

end smallest_positive_integer_l113_113337


namespace dave_guitar_strings_l113_113967

theorem dave_guitar_strings (strings_per_night : ℕ) (shows_per_week : ℕ) (weeks : ℕ)
  (h1 : strings_per_night = 4)
  (h2 : shows_per_week = 6)
  (h3 : weeks = 24) : 
  strings_per_night * shows_per_week * weeks = 576 :=
by
  sorry

end dave_guitar_strings_l113_113967


namespace incorrect_assignment_statement_l113_113319

theorem incorrect_assignment_statement :
  ∀ (a x y : ℕ), ¬(x * y = a) := by
sorry

end incorrect_assignment_statement_l113_113319


namespace triangle_sum_of_squares_not_right_l113_113039

noncomputable def is_right_triangle (a b c : ℝ) : Prop := 
  (a^2 + b^2 = c^2) ∨ (b^2 + c^2 = a^2) ∨ (c^2 + a^2 = b^2)

theorem triangle_sum_of_squares_not_right
  (a b r : ℝ) :
  a^2 + b^2 = (2 * r)^2 → ¬ ∃ (c : ℝ), is_right_triangle a b c := 
sorry

end triangle_sum_of_squares_not_right_l113_113039


namespace perimeter_of_resulting_figure_l113_113794

def side_length := 100
def original_square_perimeter := 4 * side_length
def rectangle_width := side_length
def rectangle_height := side_length / 2
def number_of_longer_sides_of_rectangles_touching := 4

theorem perimeter_of_resulting_figure :
  let new_perimeter := 3 * side_length + number_of_longer_sides_of_rectangles_touching * rectangle_height
  new_perimeter = 500 :=
by
  sorry

end perimeter_of_resulting_figure_l113_113794


namespace original_number_divisibility_l113_113179

theorem original_number_divisibility (N : ℤ) : (∃ k : ℤ, N = 9 * k + 3) ↔ (∃ m : ℤ, (N + 3) = 9 * m) := sorry

end original_number_divisibility_l113_113179


namespace swimming_pool_width_l113_113700

theorem swimming_pool_width (length width vol depth : ℝ) 
  (H_length : length = 60) 
  (H_depth : depth = 0.5) 
  (H_vol_removal : vol = 2250 / 7.48052) 
  (H_vol_eq : vol = (length * width) * depth) : 
  width = 10.019 :=
by
  -- Assuming the correctness of floating-point arithmetic for the purpose of this example
  sorry

end swimming_pool_width_l113_113700


namespace arith_geo_seq_prop_l113_113539

theorem arith_geo_seq_prop (a1 a2 b1 b2 b3 : ℝ)
  (arith_seq_condition : 1 + 2 * (a1 - 1) = a2)
  (geo_seq_condition1 : b1 * b3 = 4)
  (geo_seq_condition2 : b1 > 0)
  (geo_seq_condition3 : 1 * b1 * b2 * b3 * 4 = (b1 * b3 * -4)) :
  (a2 - a1) / b2 = 1/2 :=
by
  sorry

end arith_geo_seq_prop_l113_113539


namespace find_abc_value_l113_113359

noncomputable def abc_value_condition (a b c : ℝ) : Prop := 
  a + b + c = 4 ∧
  b * c + c * a + a * b = 5 ∧
  a^3 + b^3 + c^3 = 10

theorem find_abc_value (a b c : ℝ) (h : abc_value_condition a b c) : a * b * c = 2 := 
sorry

end find_abc_value_l113_113359


namespace problem_statement_l113_113378

-- Definitions
def div_remainder (a b : ℕ) : ℕ × ℕ :=
  (a / b, a % b)

-- Conditions and question as Lean structures
def condition := ∀ (a b k : ℕ), k ≠ 0 → div_remainder (a * k) (b * k) = (a / b, (a % b) * k)
def question := div_remainder 4900 600 = div_remainder 49 6

-- Theorem stating the problem's conclusion
theorem problem_statement (cond : condition) : ¬question :=
by
  sorry

end problem_statement_l113_113378


namespace length_BD_l113_113586

noncomputable def length_segments (CB : ℝ) : ℝ := 4 * CB

noncomputable def circle_radius_AC (CB : ℝ) : ℝ := (4 * CB) / 2

noncomputable def circle_radius_CB (CB : ℝ) : ℝ := CB / 2

noncomputable def tangent_touch_point (CB BD : ℝ) : Prop :=
  ∃ x, CB = x ∧ BD = x

theorem length_BD (CB BD : ℝ) (h : tangent_touch_point CB BD) : BD = CB :=
by
  sorry

end length_BD_l113_113586


namespace joao_speed_l113_113544

theorem joao_speed (d : ℝ) (v1 : ℝ) (t1 t2 : ℝ) (h1 : v1 = 10) (h2 : t1 = 6 / 60) (h3 : t2 = 8 / 60) : 
  d = v1 * t1 → d = 10 * (6 / 60) → (d / t2) = 7.5 := 
by
  sorry

end joao_speed_l113_113544


namespace find_x_l113_113534

theorem find_x (x : ℝ) (h1 : x > 0) (h2 : 1/2 * (2 * x) * x = 72) : x = 6 * Real.sqrt 2 :=
by
  sorry

end find_x_l113_113534


namespace students_voted_for_meat_l113_113066

theorem students_voted_for_meat (total_votes veggies_votes : ℕ) (h_total: total_votes = 672) (h_veggies: veggies_votes = 337) :
  total_votes - veggies_votes = 335 := 
by
  -- Proof steps go here
  sorry

end students_voted_for_meat_l113_113066


namespace columbian_coffee_price_is_correct_l113_113245

-- Definitions based on the conditions
def total_mix_weight : ℝ := 100
def brazilian_coffee_price_per_pound : ℝ := 3.75
def final_mix_price_per_pound : ℝ := 6.35
def columbian_coffee_weight : ℝ := 52

-- Let C be the price per pound of the Columbian coffee
noncomputable def columbian_coffee_price_per_pound : ℝ := sorry

-- Define the Lean 4 proof problem
theorem columbian_coffee_price_is_correct :
  columbian_coffee_price_per_pound = 8.75 :=
by
  -- Total weight and calculation based on conditions
  let brazilian_coffee_weight := total_mix_weight - columbian_coffee_weight
  let total_value_of_columbian := columbian_coffee_weight * columbian_coffee_price_per_pound
  let total_value_of_brazilian := brazilian_coffee_weight * brazilian_coffee_price_per_pound
  let total_value_of_mix := total_mix_weight * final_mix_price_per_pound
  
  -- Main equation based on the mix
  have main_eq : total_value_of_columbian + total_value_of_brazilian = total_value_of_mix :=
    by sorry

  -- Solve for C (columbian coffee price per pound)
  sorry

end columbian_coffee_price_is_correct_l113_113245


namespace andy_questions_wrong_l113_113536

variable (a b c d : ℕ)

theorem andy_questions_wrong
  (h1 : a + b = c + d)
  (h2 : a + d = b + c + 6)
  (h3 : c = 7)
  (h4 : d = 9) :
  a = 10 :=
by {
  sorry  -- Proof would go here
}

end andy_questions_wrong_l113_113536


namespace percentage_spent_on_household_items_l113_113217

def Raja_income : ℝ := 37500
def clothes_percentage : ℝ := 0.20
def medicines_percentage : ℝ := 0.05
def savings_amount : ℝ := 15000

theorem percentage_spent_on_household_items : 
  (Raja_income - (clothes_percentage * Raja_income + medicines_percentage * Raja_income + savings_amount)) / Raja_income * 100 = 35 :=
  sorry

end percentage_spent_on_household_items_l113_113217


namespace four_digit_number_exists_l113_113653

-- Definitions corresponding to the conditions in the problem
def is_four_digit_number (n : ℕ) : Prop := n >= 1000 ∧ n < 10000

def follows_scheme (n : ℕ) (d : ℕ) : Prop :=
  -- Placeholder for the scheme condition
  sorry

-- The Lean statement for the proof problem
theorem four_digit_number_exists :
  ∃ n d1 d2 : ℕ, is_four_digit_number n ∧ follows_scheme n d1 ∧ follows_scheme n d2 ∧ 
  (n = 1014 ∨ n = 1035 ∨ n = 1512) :=
by {
  -- Placeholder for proof steps
  sorry
}

end four_digit_number_exists_l113_113653


namespace factory_produces_11250_products_l113_113299

noncomputable def total_products (refrigerators_per_hour coolers_per_hour hours_per_day days : ℕ) : ℕ :=
  (refrigerators_per_hour + coolers_per_hour) * (hours_per_day * days)

theorem factory_produces_11250_products :
  total_products 90 (90 + 70) 9 5 = 11250 := by
  sorry

end factory_produces_11250_products_l113_113299


namespace laticia_total_pairs_l113_113285

-- Definitions of the conditions about the pairs of socks knitted each week

-- Number of pairs knitted in the first week
def pairs_week1 : ℕ := 12

-- Number of pairs knitted in the second week
def pairs_week2 : ℕ := pairs_week1 + 4

-- Number of pairs knitted in the third week
def pairs_week3 : ℕ := (pairs_week1 + pairs_week2) / 2

-- Number of pairs knitted in the fourth week
def pairs_week4 : ℕ := pairs_week3 - 3

-- Statement: Sum of pairs over the four weeks
theorem laticia_total_pairs :
  pairs_week1 + pairs_week2 + pairs_week3 + pairs_week4 = 53 := by
  sorry

end laticia_total_pairs_l113_113285


namespace A_salary_is_3000_l113_113928

theorem A_salary_is_3000 
    (x y : ℝ) 
    (h1 : x + y = 4000)
    (h2 : 0.05 * x = 0.15 * y) 
    : x = 3000 := by
  sorry

end A_salary_is_3000_l113_113928


namespace at_least_12_boxes_l113_113368

theorem at_least_12_boxes (extra_boxes : Nat) : 
  let total_boxes := 12 + extra_boxes
  extra_boxes ≥ 0 → total_boxes ≥ 12 :=
by
  intros
  sorry

end at_least_12_boxes_l113_113368


namespace angle_intersecting_lines_l113_113112

/-- 
Given three lines intersecting at a point forming six equal angles 
around the point, each angle equals 60 degrees.
-/
theorem angle_intersecting_lines (x : ℝ) (h : 6 * x = 360) : x = 60 := by
  sorry

end angle_intersecting_lines_l113_113112


namespace product_of_odd_implies_sum_is_odd_l113_113442

theorem product_of_odd_implies_sum_is_odd (a b c : ℤ) (h : a * b * c % 2 = 1) : (a + b + c) % 2 = 1 :=
sorry

end product_of_odd_implies_sum_is_odd_l113_113442


namespace arithmetic_sequence_geometric_l113_113158

noncomputable def sequence_arith_to_geom (a1 d : ℤ) (h_d : d ≠ 0) (n : ℕ) : ℤ :=
a1 + (n - 1) * d

theorem arithmetic_sequence_geometric (a1 d : ℤ) (h_d : d ≠ 0) (n : ℕ) :
  (n = 16)
    ↔ (((a1 + 3 * d) / (a1 + 2 * d) = (a1 + 6 * d) / (a1 + 3 * d)) ∧ 
        ((a1 + 6 * d) / (a1 + 3 * d) = (a1 + (n - 1) * d) / (a1 + 6 * d))) :=
by
  sorry

end arithmetic_sequence_geometric_l113_113158


namespace income_is_20000_l113_113523

-- Definitions from conditions
def income (x : ℕ) : ℕ := 4 * x
def expenditure (x : ℕ) : ℕ := 3 * x
def savings : ℕ := 5000

-- Theorem to prove the income
theorem income_is_20000 (x : ℕ) (h : income x - expenditure x = savings) : income x = 20000 :=
by
  sorry

end income_is_20000_l113_113523


namespace find_constant_l113_113103

theorem find_constant (n : ℤ) (c : ℝ) (h1 : ∀ n ≤ 10, c * (n : ℝ)^2 ≤ 12100) : c ≤ 121 :=
sorry

end find_constant_l113_113103


namespace circle_equation_l113_113745

theorem circle_equation (x y : ℝ)
  (h_center : ∀ x y, (x - 3)^2 + (y - 1)^2 = r ^ 2)
  (h_origin : (0 - 3)^2 + (0 - 1)^2 = r ^ 2) :
  (x - 3) ^ 2 + (y - 1) ^ 2 = 10 := by
  sorry

end circle_equation_l113_113745


namespace remainder_when_multiplied_and_divided_l113_113317

theorem remainder_when_multiplied_and_divided (n k : ℤ) (h : n % 28 = 15) : (2 * n) % 14 = 2 := 
by
  sorry

end remainder_when_multiplied_and_divided_l113_113317


namespace geo_seq_sum_eq_l113_113275

variable {a : ℕ → ℝ}

-- Conditions
def is_geo_seq (a : ℕ → ℝ) : Prop := ∃ r : ℝ, ∀ n : ℕ, a (n+1) = a n * r
def positive_seq (a : ℕ → ℝ) : Prop := ∀ n : ℕ, a n > 0
def specific_eq (a : ℕ → ℝ) : Prop := a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 25

theorem geo_seq_sum_eq (a : ℕ → ℝ) (h_geo : is_geo_seq a) (h_pos : positive_seq a) (h_eq : specific_eq a) : 
  a 2 + a 4 = 5 :=
by
  sorry

end geo_seq_sum_eq_l113_113275


namespace a_mul_b_value_l113_113290

theorem a_mul_b_value (a b : ℤ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a + b = 15) (h₃ : a * b = 36) : 
  (a * b = (1/a : ℚ) + (1/b : ℚ)) ∧ (a * b = 15/36) ∧ (15 / 36 = 5 / 12) :=
by
  sorry

end a_mul_b_value_l113_113290


namespace triangle_inequality_l113_113249

theorem triangle_inequality (a b c : ℝ) (h1 : a + b + c = 2) :
  a^2 + b^2 + c^2 < 2 * (1 - a * b * c) :=
sorry

end triangle_inequality_l113_113249


namespace number_added_is_minus_168_l113_113743

theorem number_added_is_minus_168 (N : ℕ) (X : ℤ) (h1 : N = 180)
  (h2 : N + (1/2 : ℚ) * (1/3 : ℚ) * (1/5 : ℚ) * N = (1/15 : ℚ) * N) : X = -168 :=
by
  sorry

end number_added_is_minus_168_l113_113743


namespace runners_meet_again_l113_113832

-- Definitions based on the problem conditions
def track_length : ℝ := 500 
def speed_runner1 : ℝ := 4.4
def speed_runner2 : ℝ := 4.8
def speed_runner3 : ℝ := 5.0

-- The time at which runners meet again at the starting point
def time_when_runners_meet : ℝ := 2500

theorem runners_meet_again :
  ∀ t : ℝ, t = time_when_runners_meet → 
  (∀ n1 n2 n3 : ℤ, 
    ∃ k : ℤ, 
    speed_runner1 * t = n1 * track_length ∧ 
    speed_runner2 * t = n2 * track_length ∧ 
    speed_runner3 * t = n3 * track_length) :=
by 
  sorry

end runners_meet_again_l113_113832


namespace domain_of_f_l113_113895

noncomputable def f (x : ℝ) : ℝ := (Real.log (x + 1)) / (x - 2)

theorem domain_of_f : {x : ℝ | x > -1 ∧ x ≠ 2} = {x : ℝ | x ∈ Set.Ioo (-1) 2 ∪ Set.Ioi 2} :=
by {
  sorry
}

end domain_of_f_l113_113895


namespace percentage_of_females_l113_113622

theorem percentage_of_females (total_passengers : ℕ)
  (first_class_percentage : ℝ) (male_fraction_first_class : ℝ)
  (females_coach_class : ℕ) (h1 : total_passengers = 120)
  (h2 : first_class_percentage = 0.10)
  (h3 : male_fraction_first_class = 1/3)
  (h4 : females_coach_class = 40) :
  (females_coach_class + (first_class_percentage * total_passengers - male_fraction_first_class * (first_class_percentage * total_passengers))) / total_passengers * 100 = 40 :=
by
  sorry

end percentage_of_females_l113_113622


namespace germination_percentage_l113_113087

theorem germination_percentage (total_seeds_plot1 total_seeds_plot2 germinated_plot2_percentage total_germinated_percentage germinated_plot1_percentage : ℝ) 
  (plant1 : total_seeds_plot1 = 300) 
  (plant2 : total_seeds_plot2 = 200) 
  (germination2 : germinated_plot2_percentage = 0.35) 
  (total_germination : total_germinated_percentage = 0.23)
  (germinated_plot1 : germinated_plot1_percentage = 0.15) :
  (total_germinated_percentage * (total_seeds_plot1 + total_seeds_plot2) = 
    (germinated_plot2_percentage * total_seeds_plot2) + (germinated_plot1_percentage * total_seeds_plot1)) :=
by
  sorry

end germination_percentage_l113_113087


namespace inequality_solution_set_range_of_a_l113_113655

noncomputable def f (x : ℝ) := abs (2 * x - 1) - abs (x + 2)

theorem inequality_solution_set :
  { x : ℝ | f x > 0 } = { x : ℝ | x < -1 / 3 ∨ x > 3 } :=
sorry

theorem range_of_a (x0 : ℝ) (h : f x0 + 2 * a ^ 2 < 4 * a) :
  -1 / 2 < a ∧ a < 5 / 2 :=
sorry

end inequality_solution_set_range_of_a_l113_113655


namespace vector_sum_l113_113024

def a : ℝ × ℝ := (-2, 3)
def b : ℝ × ℝ := (1, -2)

theorem vector_sum:
  2 • a + b = (-3, 4) :=
by 
  sorry

end vector_sum_l113_113024


namespace minValue_expression_l113_113221

noncomputable def minValue (x y : ℝ) : ℝ :=
  4 / x^2 + 4 / (x * y) + 1 / y^2

theorem minValue_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : (x - 2 * y)^2 = (x * y)^3) :
  minValue x y = 4 * Real.sqrt 2 :=
sorry

end minValue_expression_l113_113221


namespace exists_rational_non_integer_a_not_exists_rational_non_integer_b_l113_113528

-- Define rational non-integer numbers
def is_rational_non_integer (x : ℚ) : Prop := ¬(∃ (z : ℤ), x = z)

-- (a) Proof for existance of rational non-integer numbers y and x such that 19x + 8y, 8x + 3y are integers
theorem exists_rational_non_integer_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ a b : ℤ, 19 * x + 8 * y = a ∧ 8 * x + 3 * y = b) :=
sorry

-- (b) Proof for non-existance of rational non-integer numbers y and x such that 19x² + 8y², 8x² + 3y² are integers
theorem not_exists_rational_non_integer_b :
  ¬ ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧ (∃ m n : ℤ, 19 * x^2 + 8 * y^2 = m ∧ 8 * x^2 + 3 * y^2 = n) :=
sorry

end exists_rational_non_integer_a_not_exists_rational_non_integer_b_l113_113528


namespace price_of_72_cans_is_18_36_l113_113017

def regular_price_per_can : ℝ := 0.30
def discount_percent : ℝ := 0.15
def number_of_cans : ℝ := 72

def discounted_price_per_can : ℝ := regular_price_per_can - (discount_percent * regular_price_per_can)
def total_price (num_cans : ℝ) : ℝ := num_cans * discounted_price_per_can

theorem price_of_72_cans_is_18_36 :
  total_price number_of_cans = 18.36 :=
by
  /- Proof details omitted -/
  sorry

end price_of_72_cans_is_18_36_l113_113017


namespace distinct_solution_count_l113_113980

theorem distinct_solution_count : ∀ (x : ℝ), (|x - 10| = |x + 4|) → x = 3 :=
by
  sorry

end distinct_solution_count_l113_113980


namespace translated_parabola_eq_l113_113718

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := 3 * x^2

-- Function to translate a parabola equation downward by a units
def translate_downward (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f x - a

-- Function to translate a parabola equation rightward by b units
def translate_rightward (f : ℝ → ℝ) (b : ℝ) (x : ℝ) : ℝ := f (x - b)

-- The new parabola equation after translating the given parabola downward by 3 units and rightward by 2 units
def new_parabola (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 9

-- The main theorem stating that translating the original parabola downward by 3 units and rightward by 2 units results in the new parabola equation
theorem translated_parabola_eq :
  ∀ x : ℝ, translate_rightward (translate_downward original_parabola 3) 2 x = new_parabola x :=
by
  sorry

end translated_parabola_eq_l113_113718


namespace units_digit_7_pow_103_l113_113533

theorem units_digit_7_pow_103 : Nat.mod (7 ^ 103) 10 = 3 := sorry

end units_digit_7_pow_103_l113_113533


namespace eric_bike_speed_l113_113657

def swim_distance : ℝ := 0.5
def swim_speed : ℝ := 1
def run_distance : ℝ := 2
def run_speed : ℝ := 8
def bike_distance : ℝ := 12
def total_time_limit : ℝ := 2

theorem eric_bike_speed :
  (swim_distance / swim_speed) + (run_distance / run_speed) + (bike_distance / (48/5)) < total_time_limit :=
by
  sorry

end eric_bike_speed_l113_113657


namespace will_total_clothes_l113_113955

theorem will_total_clothes (n1 n2 n3 : ℕ) (h1 : n1 = 32) (h2 : n2 = 9) (h3 : n3 = 3) : n1 + n2 * n3 = 59 := 
by
  sorry

end will_total_clothes_l113_113955


namespace poultry_count_correct_l113_113054

noncomputable def total_poultry : ℝ :=
  let hens_total := 40
  let ducks_total := 20
  let geese_total := 10
  let pigeons_total := 30

  -- Calculate males and females
  let hens_males := (2/9) * hens_total
  let hens_females := hens_total - hens_males

  let ducks_males := (1/4) * ducks_total
  let ducks_females := ducks_total - ducks_males

  let geese_males := (3/11) * geese_total
  let geese_females := geese_total - geese_males

  let pigeons_males := (1/2) * pigeons_total
  let pigeons_females := pigeons_total - pigeons_males

  -- Offspring calculations using breeding success rates
  let hens_offspring := (0.85 * hens_females) * 7
  let ducks_offspring := (0.75 * ducks_females) * 9
  let geese_offspring := (0.9 * geese_females) * 5
  let pigeons_pairs := 0.8 * (pigeons_females / 2)
  let pigeons_offspring := pigeons_pairs * 2 * 0.8

  -- Total poultry count
  (hens_total + ducks_total + geese_total + pigeons_total) + (hens_offspring + ducks_offspring + geese_offspring + pigeons_offspring)

theorem poultry_count_correct : total_poultry = 442 := by
  sorry

end poultry_count_correct_l113_113054


namespace even_integers_between_sqrt_10_and_sqrt_100_l113_113277

theorem even_integers_between_sqrt_10_and_sqrt_100 : 
  ∃ (n : ℕ), n = 4 ∧ (∀ (a : ℕ), (∃ k, (2 * k = a ∧ a > Real.sqrt 10 ∧ a < Real.sqrt 100)) ↔ 
  (a = 4 ∨ a = 6 ∨ a = 8 ∨ a = 10)) := 
by 
  sorry

end even_integers_between_sqrt_10_and_sqrt_100_l113_113277


namespace find_missing_number_l113_113636

theorem find_missing_number (x : ℕ) : (4 + 3) + (8 - 3 - x) = 11 → x = 1 :=
by
  sorry

end find_missing_number_l113_113636


namespace fraction_of_juniors_l113_113704

theorem fraction_of_juniors (J S : ℕ) (h1 : J > 0) (h2 : S > 0) (h : 1 / 2 * J = 2 / 3 * S) : J / (J + S) = 4 / 7 :=
by
  sorry

end fraction_of_juniors_l113_113704


namespace coprime_condition_exists_l113_113676

theorem coprime_condition_exists : ∃ (A B C : ℕ), (A > 0 ∧ B > 0 ∧ C > 0) ∧ (Nat.gcd (Nat.gcd A B) C = 1) ∧ 
  (A * Real.log 5 / Real.log 50 + B * Real.log 2 / Real.log 50 = C) ∧ (A + B + C = 4) :=
by {
  sorry
}

end coprime_condition_exists_l113_113676


namespace x_squared_inverse_y_fourth_l113_113471

theorem x_squared_inverse_y_fourth (x y : ℝ) (k : ℝ) (h₁ : x = 8) (h₂ : y = 2) (h₃ : (x^2) * (y^4) = k) : x^2 = 4 :=
by
  sorry

end x_squared_inverse_y_fourth_l113_113471


namespace largest_possible_n_l113_113420

theorem largest_possible_n (k : ℕ) (hk : k > 0) : ∃ n, n = 3 * k - 1 := 
  sorry

end largest_possible_n_l113_113420


namespace arthur_walks_distance_l113_113574

theorem arthur_walks_distance :
  ∀ (blocks_east blocks_north blocks_first blocks_other distance_first distance_other : ℕ)
  (fraction_first fraction_other : ℚ),
    blocks_east = 8 →
    blocks_north = 16 →
    blocks_first = 10 →
    blocks_other = (blocks_east + blocks_north) - blocks_first →
    fraction_first = 1 / 3 →
    fraction_other = 1 / 4 →
    distance_first = blocks_first * fraction_first →
    distance_other = blocks_other * fraction_other →
    (distance_first + distance_other) = 41 / 6 :=
by
  intros blocks_east blocks_north blocks_first blocks_other distance_first distance_other fraction_first fraction_other
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end arthur_walks_distance_l113_113574


namespace saffron_milk_caps_and_milk_caps_in_basket_l113_113038

structure MushroomBasket :=
  (total : ℕ)
  (saffronMilkCapCount : ℕ)
  (milkCapCount : ℕ)
  (TotalMushrooms : total = 30)
  (SaffronMilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 12 → ∃ i ∈ selected, i < saffronMilkCapCount)
  (MilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 20 → ∃ i ∈ selected, i < milkCapCount)

theorem saffron_milk_caps_and_milk_caps_in_basket
  (basket : MushroomBasket)
  (TotalMushrooms : basket.total = 30)
  (SaffronMilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 12 → ∃ i ∈ selected, i < basket.saffronMilkCapCount)
  (MilkCapCondition : ∀ (selected : Finset ℕ), selected.card = 20 → ∃ i ∈ selected, i < basket.milkCapCount) :
  basket.saffronMilkCapCount = 19 ∧ basket.milkCapCount = 11 :=
sorry

end saffron_milk_caps_and_milk_caps_in_basket_l113_113038


namespace find_a_and_solve_inequalities_l113_113085

-- Definitions as per conditions
def inequality1 (a : ℝ) (x : ℝ) : Prop := a*x^2 + 5*x - 2 > 0
def inequality2 (a : ℝ) (x : ℝ) : Prop := a*x^2 - 5*x + a^2 - 1 > 0

-- Statement of the theorem
theorem find_a_and_solve_inequalities :
  ∀ (a : ℝ),
    (∀ x, (1/2 < x ∧ x < 2) ↔ inequality1 a x) →
    a = -2 ∧
    (∀ x, (-1/2 < x ∧ x < 3) ↔ inequality2 (-2) x) :=
by
  intros a h
  sorry

end find_a_and_solve_inequalities_l113_113085


namespace trains_crossing_time_l113_113280

theorem trains_crossing_time
  (L speed1 speed2 : ℝ)
  (time_same_direction time_opposite_direction : ℝ) 
  (h1 : speed1 = 60)
  (h2 : speed2 = 40)
  (h3 : time_same_direction = 40)
  (h4 : 2 * L = (speed1 - speed2) * 5/18 * time_same_direction) :
  time_opposite_direction = 8 := 
sorry

end trains_crossing_time_l113_113280


namespace solve_absolute_value_inequality_l113_113802

theorem solve_absolute_value_inequality (x : ℝ) :
  3 ≤ |x + 3| ∧ |x + 3| ≤ 7 ↔ (-10 ≤ x ∧ x ≤ -6) ∨ (0 ≤ x ∧ x ≤ 4) :=
by
  sorry

end solve_absolute_value_inequality_l113_113802


namespace Nicky_time_before_catchup_l113_113432

-- Define the given speeds and head start time as constants
def v_C : ℕ := 5 -- Cristina's speed in meters per second
def v_N : ℕ := 3 -- Nicky's speed in meters per second
def t_H : ℕ := 12 -- Head start in seconds

-- Define the running time until catch up
def time_Nicky_run : ℕ := t_H + (36 / (v_C - v_N))

-- Prove that the time Nicky has run before Cristina catches up to him is 30 seconds
theorem Nicky_time_before_catchup : time_Nicky_run = 30 :=
by
  -- Add the steps for the proof
  sorry

end Nicky_time_before_catchup_l113_113432


namespace cube_volume_l113_113640

theorem cube_volume (d : ℝ) (s : ℝ) (h : d = 3 * Real.sqrt 3) (h_s : s * Real.sqrt 3 = d) : s ^ 3 = 27 := by
  -- Assuming h: the formula for the given space diagonal
  -- Assuming h_s: the formula connecting side length and the space diagonal
  sorry

end cube_volume_l113_113640


namespace weight_labels_correct_l113_113063

-- Noncomputable because we're dealing with theoretical weight comparisons
noncomputable section

-- Defining the weights and their properties
variables {x1 x2 x3 x4 x5 x6 : ℕ}

-- Given conditions as stated
axiom h1 : x1 + x2 + x3 = 6
axiom h2 : x6 = 6
axiom h3 : x1 + x6 < x3 + x5

theorem weight_labels_correct :
  x1 = 1 ∧ x2 = 2 ∧ x3 = 3 ∧ x4 = 4 ∧ x5 = 5 ∧ x6 = 6 :=
sorry

end weight_labels_correct_l113_113063


namespace reversed_number_increase_l113_113238

theorem reversed_number_increase (a b c : ℕ) 
  (h1 : a + b + c = 10) 
  (h2 : b = a + c)
  (h3 : a = 2 ∧ b = 5 ∧ c = 3) :
  (c * 100 + b * 10 + a) - (a * 100 + b * 10 + c) = 99 :=
by
  sorry

end reversed_number_increase_l113_113238


namespace total_expenditure_is_108_l113_113323

-- Define the costs of items and quantities purchased by Robert and Teddy
def cost_pizza := 10   -- cost of one box of pizza
def cost_soft_drink := 2  -- cost of one can of soft drink
def cost_hamburger := 3   -- cost of one hamburger

def qty_pizza_robert := 5     -- quantity of pizza boxes by Robert
def qty_soft_drink_robert := 10 -- quantity of soft drinks by Robert

def qty_hamburger_teddy := 6  -- quantity of hamburgers by Teddy
def qty_soft_drink_teddy := 10 -- quantity of soft drinks by Teddy

-- Calculate total expenditure for Robert and Teddy
def total_cost_robert := (qty_pizza_robert * cost_pizza) + (qty_soft_drink_robert * cost_soft_drink)
def total_cost_teddy := (qty_hamburger_teddy * cost_hamburger) + (qty_soft_drink_teddy * cost_soft_drink)

-- Total expenditure in all
def total_expenditure := total_cost_robert + total_cost_teddy

-- We formulate the theorem to prove that the total expenditure is $108
theorem total_expenditure_is_108 : total_expenditure = 108 :=
by 
  -- Placeholder proof
  sorry

end total_expenditure_is_108_l113_113323


namespace time_per_lice_check_l113_113851

-- Define the number of students in each grade
def kindergartners := 26
def first_graders := 19
def second_graders := 20
def third_graders := 25

-- Define the total number of students
def total_students := kindergartners + first_graders + second_graders + third_graders

-- Define the total time in minutes
def hours := 3
def minutes_per_hour := 60
def total_minutes := hours * minutes_per_hour

-- Define the correct answer for time per check
def time_per_check := total_minutes / total_students

-- Prove that the time for each check is 2 minutes
theorem time_per_lice_check : time_per_check = 2 := 
by
  sorry

end time_per_lice_check_l113_113851


namespace conditional_probability_second_sci_given_first_sci_l113_113287

-- Definitions based on the conditions
def total_questions : ℕ := 6
def science_questions : ℕ := 4
def humanities_questions : ℕ := 2
def first_draw_is_science : Prop := true

-- The statement we want to prove
theorem conditional_probability_second_sci_given_first_sci : 
    first_draw_is_science → (science_questions - 1) / (total_questions - 1) = 3 / 5 := 
by
  intro h
  have num_sci_after_first : ℕ := science_questions - 1
  have total_after_first : ℕ := total_questions - 1
  have prob_second_sci := num_sci_after_first / total_after_first
  sorry

end conditional_probability_second_sci_given_first_sci_l113_113287


namespace ben_marble_count_l113_113071

theorem ben_marble_count :
  ∃ k : ℕ, 5 * 2^k > 200 ∧ ∀ m < k, 5 * 2^m ≤ 200 :=
sorry

end ben_marble_count_l113_113071


namespace total_marks_l113_113475

variable (E S M : Nat)

-- Given conditions
def thrice_as_many_marks_in_English_as_in_Science := E = 3 * S
def ratio_of_marks_in_English_and_Maths            := M = 4 * E
def marks_in_Science                               := S = 17

-- Proof problem statement
theorem total_marks (h1 : E = 3 * S) (h2 : M = 4 * E) (h3 : S = 17) :
  E + S + M = 272 :=
by
  sorry

end total_marks_l113_113475


namespace min_value_expr_l113_113409

open Real

theorem min_value_expr (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  ∃ c, c = 4 * sqrt 3 - 6 ∧ ∀ (z w : ℝ), z = x ∧ w = y → (3 * z) / (3 * z + 2 * w) + w / (2 * z + w) ≥ c :=
by
  sorry

end min_value_expr_l113_113409


namespace sum_of_consecutive_integers_l113_113604

theorem sum_of_consecutive_integers (n : ℤ) (h : n * (n + 1) = 20412) : n + (n + 1) = 287 :=
by
  sorry

end sum_of_consecutive_integers_l113_113604


namespace negation_proposition_l113_113273

variable (n : ℕ)
variable (n_positive : n > 0)
variable (f : ℕ → ℕ)
variable (H1 : ∀ n, n > 0 → (f n) > 0 ∧ (f n) ≤ n)

theorem negation_proposition :
  (∃ n_0, n_0 > 0 ∧ ((f n_0) ≤ 0 ∨ (f n_0) > n_0)) ↔ ¬(∀ n, n > 0 → (f n) >0 ∧ (f n) ≤ n) :=
by 
  sorry

end negation_proposition_l113_113273


namespace part_I_part_II_l113_113839

noncomputable def f_I (x : ℝ) : ℝ := abs (3*x - 1) + abs (x + 3)

theorem part_I :
  ∀ x : ℝ, f_I x ≥ 4 ↔ x ≤ 0 ∨ x ≥ 1/2 :=
by sorry

noncomputable def f_II (x b c : ℝ) : ℝ := abs (x - b) + abs (x + c)

theorem part_II :
  ∀ b c : ℝ, b > 0 → c > 0 → b + c = 1 → 
  (∀ x : ℝ, f_II x b c ≥ 1) → (1 / b + 1 / c = 4) :=
by sorry

end part_I_part_II_l113_113839


namespace rahul_work_days_l113_113309

variable (R : ℕ)

theorem rahul_work_days
  (rajesh_days : ℕ := 2)
  (total_money : ℕ := 355)
  (rahul_share : ℕ := 142)
  (rajesh_share : ℕ := total_money - rahul_share)
  (payment_ratio : ℕ := rahul_share / rajesh_share)
  (work_rate_ratio : ℕ := rajesh_days / R) :
  payment_ratio = work_rate_ratio → R = 3 :=
by
  sorry

end rahul_work_days_l113_113309


namespace find_winner_votes_l113_113801

-- Define the conditions
variables (V : ℝ) (winner_votes second_votes : ℝ)
def election_conditions :=
  winner_votes = 0.468 * V ∧
  second_votes = 0.326 * V ∧
  winner_votes - second_votes = 752

-- State the theorem
theorem find_winner_votes (h : election_conditions V winner_votes second_votes) :
  winner_votes = 2479 :=
sorry

end find_winner_votes_l113_113801


namespace suraj_average_l113_113763

theorem suraj_average : 
  ∀ (A : ℝ), 
    (16 * A + 92 = 17 * (A + 4)) → 
      (A + 4) = 28 :=
by
  sorry

end suraj_average_l113_113763


namespace find_m_plus_t_l113_113817

-- Define the system of equations represented by the augmented matrix
def equation1 (m t : ℝ) : Prop := 3 * m - t = 22
def equation2 (t : ℝ) : Prop := t = 2

-- State the main theorem with the given conditions and the goal
theorem find_m_plus_t (m t : ℝ) (h1 : equation1 m t) (h2 : equation2 t) : m + t = 10 := 
by
  sorry

end find_m_plus_t_l113_113817


namespace fraction_ratio_l113_113597

theorem fraction_ratio (x : ℚ) (h1 : 2 / 5 / (3 / 7) = x / (1 / 2)) :
  x = 7 / 15 :=
by {
  -- Proof omitted
  sorry
}

end fraction_ratio_l113_113597


namespace delta_maximum_success_ratio_l113_113848

theorem delta_maximum_success_ratio (x y z w : ℕ) (h1 : 0 < x ∧ x * 5 < y * 3)
    (h2 : 0 < z ∧ z * 5 < w * 3) (h3 : y + w = 600) :
    (x + z) / 600 ≤ 359 / 600 :=
by
  sorry

end delta_maximum_success_ratio_l113_113848


namespace total_area_of_rectangles_l113_113170

/-- The combined area of two adjacent rectangular regions given their conditions -/
theorem total_area_of_rectangles (u v w z : ℝ) 
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) (hz : w < z) : 
  (u + v) * z = (u + v) * w + (u + v) * (z - w) :=
by
  sorry

end total_area_of_rectangles_l113_113170


namespace max_ab_value_l113_113706

theorem max_ab_value {a b : ℝ} (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : 6 * a + 8 * b = 72) : ab = 27 :=
by {
  sorry
}

end max_ab_value_l113_113706


namespace balls_in_boxes_l113_113714

theorem balls_in_boxes : 
  ∀ (balls boxes : ℕ), balls = 5 ∧ boxes = 3 → 
  ∃ (ways : ℕ), ways = 21 :=
by
  sorry

end balls_in_boxes_l113_113714


namespace max_gcd_sequence_l113_113151

noncomputable def a (n : ℕ) : ℕ := n^3 + 4
noncomputable def d (n : ℕ) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_gcd_sequence : (∀ n : ℕ, 0 < n → d n ≤ 433) ∧ (∃ n : ℕ, 0 < n ∧ d n = 433) :=
by sorry

end max_gcd_sequence_l113_113151


namespace ratio_of_men_to_women_l113_113581

-- Define constants
def total_people : ℕ := 60
def men_in_meeting : ℕ := 4
def women_in_meeting : ℕ := 6
def women_reduction_percentage : ℕ := 20

-- Statement of the problem
theorem ratio_of_men_to_women (total_people men_in_meeting women_in_meeting women_reduction_percentage: ℕ)
  (total_people_eq : total_people = 60)
  (men_in_meeting_eq : men_in_meeting = 4)
  (women_in_meeting_eq : women_in_meeting = 6)
  (women_reduction_percentage_eq : women_reduction_percentage = 20) :
  (men_in_meeting + ((total_people - men_in_meeting - women_in_meeting) * women_reduction_percentage / 100)) 
  = total_people / 2 :=
sorry

end ratio_of_men_to_women_l113_113581


namespace new_volume_is_correct_l113_113670

variable (l w h : ℝ)

-- Conditions given in the problem
axiom volume : l * w * h = 4320
axiom surface_area : 2 * (l * w + w * h + h * l) = 1704
axiom edge_sum : 4 * (l + w + h) = 208

-- The proposition we need to prove:
theorem new_volume_is_correct : (l + 2) * (w + 2) * (h + 2) = 6240 :=
by
  -- Placeholder for the actual proof
  sorry

end new_volume_is_correct_l113_113670


namespace jerry_can_throw_things_l113_113195

def points_for_interrupting : ℕ := 5
def points_for_insulting : ℕ := 10
def points_for_throwing : ℕ := 25
def office_points_threshold : ℕ := 100
def interruptions : ℕ := 2
def insults : ℕ := 4

theorem jerry_can_throw_things : 
  (office_points_threshold - (points_for_interrupting * interruptions + points_for_insulting * insults)) / points_for_throwing = 2 :=
by 
  sorry

end jerry_can_throw_things_l113_113195


namespace total_inflation_over_two_years_real_interest_rate_over_two_years_l113_113568

section FinancialCalculations

-- Define the known conditions
def annual_inflation_rate : ℚ := 0.025
def nominal_interest_rate : ℚ := 0.06

-- Prove the total inflation rate over two years equals 5.0625%
theorem total_inflation_over_two_years :
  ((1 + annual_inflation_rate)^2 - 1) * 100 = 5.0625 :=
sorry

-- Prove the real interest rate over two years equals 6.95%
theorem real_interest_rate_over_two_years :
  ((1 + nominal_interest_rate) * (1 + nominal_interest_rate) / (1 + (annual_inflation_rate * annual_inflation_rate)) - 1) * 100 = 6.95 :=
sorry

end FinancialCalculations

end total_inflation_over_two_years_real_interest_rate_over_two_years_l113_113568


namespace count_valid_subsets_l113_113929

open Set

theorem count_valid_subsets :
  ∀ (A : Set ℕ), (A ⊆ {1, 2, 3, 4, 5, 6, 7}) → 
  (∀ (a : ℕ), a ∈ A → (8 - a) ∈ A) → A ≠ ∅ → 
  ∃! (n : ℕ), n = 15 :=
  by
    sorry

end count_valid_subsets_l113_113929


namespace prob_yellow_and_straight_l113_113884

-- Definitions of probabilities given in the problem
def prob_green : ℚ := 2 / 3
def prob_straight : ℚ := 1 / 2

-- Derived probability of picking a yellow flower
def prob_yellow : ℚ := 1 - prob_green

-- Statement to prove
theorem prob_yellow_and_straight : prob_yellow * prob_straight = 1 / 6 :=
by
  -- sorry is used here to skip the proof.
  sorry

end prob_yellow_and_straight_l113_113884


namespace evaluate_fraction_l113_113037

theorem evaluate_fraction (a b : ℤ) (h1 : a = 5) (h2 : b = -2) : (5 : ℝ) / (a + b) = 5 / 3 :=
by
  sorry

end evaluate_fraction_l113_113037


namespace biology_books_needed_l113_113026

-- Define the problem in Lean
theorem biology_books_needed
  (B P Q R F Z₁ Z₂ : ℕ)
  (b p : ℝ)
  (H1 : B ≠ P)
  (H2 : B ≠ Q)
  (H3 : B ≠ R)
  (H4 : B ≠ F)
  (H5 : P ≠ Q)
  (H6 : P ≠ R)
  (H7 : P ≠ F)
  (H8 : Q ≠ R)
  (H9 : Q ≠ F)
  (H10 : R ≠ F)
  (H11 : 0 < B ∧ 0 < P ∧ 0 < Q ∧ 0 < R ∧ 0 < F)
  (H12 : Bb + Pp = Z₁)
  (H13 : Qb + Rp = Z₂)
  (H14 : Fb = Z₁)
  (H15 : Z₂ < Z₁) :
  F = (Q - B) / (P - R) :=
by
  sorry  -- Proof to be provided

end biology_books_needed_l113_113026


namespace intersection_A_B_l113_113345

def A : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1 / (x^2 + 1) }
def B : Set ℝ := {x | 3 * x - 2 < 7}

theorem intersection_A_B : A ∩ B = Set.Ico 1 3 := 
by
  sorry

end intersection_A_B_l113_113345


namespace part1_part2_l113_113463

variable (m x : ℝ)

-- Condition: mx - 3 > 2x + m
def inequality1 := m * x - 3 > 2 * x + m

-- Part (1) Condition: x < (m + 3) / (m - 2)
def solution_set_part1 := x < (m + 3) / (m - 2)

-- Part (2) Condition: 2x - 1 > 3 - x
def inequality2 := 2 * x - 1 > 3 - x

theorem part1 (h : ∀ x, inequality1 m x → solution_set_part1 m x) : m < 2 :=
sorry

theorem part2 (h1 : ∀ x, inequality1 m x ↔ inequality2 x) : m = 17 :=
sorry

end part1_part2_l113_113463


namespace watermelon_count_l113_113641

theorem watermelon_count (seeds_per_watermelon : ℕ) (total_seeds : ℕ)
  (h1 : seeds_per_watermelon = 100) (h2 : total_seeds = 400) : total_seeds / seeds_per_watermelon = 4 :=
by
  sorry

end watermelon_count_l113_113641


namespace csc_neg_45_eq_neg_sqrt_2_l113_113160

noncomputable def csc (θ : Real) : Real := 1 / Real.sin θ

theorem csc_neg_45_eq_neg_sqrt_2 :
  csc (-Real.pi / 4) = -Real.sqrt 2 := by
  sorry

end csc_neg_45_eq_neg_sqrt_2_l113_113160


namespace exists_a_b_k_l113_113162

theorem exists_a_b_k (m : ℕ) (hm : 0 < m) : 
  ∃ a b k : ℤ, 
    (a % 2 = 1) ∧ 
    (b % 2 = 1) ∧ 
    (0 ≤ k) ∧ 
    (2 * m = a^19 + b^99 + k * 2^1999) :=
sorry

end exists_a_b_k_l113_113162


namespace smallest_other_integer_l113_113251

-- Definitions of conditions
def gcd_condition (a b : ℕ) (x : ℕ) : Prop := 
  Nat.gcd a b = x + 5

def lcm_condition (a b : ℕ) (x : ℕ) : Prop := 
  Nat.lcm a b = x * (x + 5)

def sum_condition (a b : ℕ) : Prop := 
  a + b < 100

-- Main statement incorporating all conditions
theorem smallest_other_integer {x b : ℕ} (hx_pos : x > 0)
  (h_gcd : gcd_condition 45 b x)
  (h_lcm : lcm_condition 45 b x)
  (h_sum : sum_condition 45 b) :
  b = 12 :=
sorry

end smallest_other_integer_l113_113251


namespace analysis_method_proves_sufficient_condition_l113_113897

-- Definitions and conditions from part (a)
def analysis_method_traces_cause_from_effect : Prop := true
def analysis_method_seeks_sufficient_conditions : Prop := true
def analysis_method_finds_conditions_for_inequality : Prop := true

-- The statement to be proven
theorem analysis_method_proves_sufficient_condition :
  analysis_method_finds_conditions_for_inequality →
  analysis_method_traces_cause_from_effect →
  analysis_method_seeks_sufficient_conditions →
  (B = "Sufficient condition") :=
by 
  sorry

end analysis_method_proves_sufficient_condition_l113_113897


namespace sqrt_expression_result_l113_113592

theorem sqrt_expression_result :
  (Real.sqrt (16 - 8 * Real.sqrt 3) - Real.sqrt (16 + 8 * Real.sqrt 3)) ^ 2 = 48 := 
sorry

end sqrt_expression_result_l113_113592


namespace find_common_ratio_l113_113205

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a 1 * q ^ n

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, S n = a 1 * (1 - q ^ n) / (1 - q)

variables (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 q : ℝ)

noncomputable def a_5_condition : Prop :=
  a 5 = 2 * S 4 + 3

noncomputable def a_6_condition : Prop :=
  a 6 = 2 * S 5 + 3

theorem find_common_ratio (h1 : a_5_condition a S) (h2 : a_6_condition a S)
  (hg : geometric_sequence a q) (hs : sum_of_first_n_terms a S q) :
  q = 3 :=
sorry

end find_common_ratio_l113_113205


namespace rectangular_prism_sides_multiples_of_5_l113_113094

noncomputable def rectangular_prism_sides_multiples_product_condition 
  (l w : ℕ) (hl : l % 5 = 0) (hw : w % 5 = 0) (prod_eq_450 : l * w = 450) : Prop :=
  l ∣ 450 ∧ w ∣ 450

theorem rectangular_prism_sides_multiples_of_5
  (l w : ℕ) (hl : l % 5 = 0) (hw : w % 5 = 0) :
  rectangular_prism_sides_multiples_product_condition l w hl hw (by sorry) :=
sorry

end rectangular_prism_sides_multiples_of_5_l113_113094


namespace relationship_withdrawn_leftover_l113_113530

-- Definitions based on the problem conditions
def pie_cost : ℝ := 6
def sandwich_cost : ℝ := 3
def book_cost : ℝ := 10
def book_discount : ℝ := 0.2 * book_cost
def book_price_with_discount : ℝ := book_cost - book_discount
def total_spent_before_tax : ℝ := pie_cost + sandwich_cost + book_price_with_discount
def sales_tax_rate : ℝ := 0.05
def sales_tax : ℝ := sales_tax_rate * total_spent_before_tax
def total_spent_with_tax : ℝ := total_spent_before_tax + sales_tax

-- Given amount withdrawn and amount left after shopping
variables (X Y : ℝ)

-- Theorem statement
theorem relationship_withdrawn_leftover :
  Y = X - total_spent_with_tax :=
sorry

end relationship_withdrawn_leftover_l113_113530


namespace cone_volume_l113_113458

theorem cone_volume (l : ℝ) (h : ℝ) (r : ℝ) (V : ℝ) 
  (hl : l = 15)    -- slant height
  (hh : h = 9)     -- vertical height
  (hr : r^2 = 144) -- radius squared from Pythagorean theorem
  : V = 432 * Real.pi :=
by
  -- Proof is omitted. Hence, we write sorry to denote skipped proof.
  sorry

end cone_volume_l113_113458


namespace hexagon_angles_l113_113525

theorem hexagon_angles
  (AB CD EF BC DE FA : ℝ)
  (F A B C D E : Type*)
  (FAB ABC EFA CDE : ℝ)
  (h1 : AB = CD)
  (h2 : AB = EF)
  (h3 : BC = DE)
  (h4 : BC = FA)
  (h5 : FAB + ABC = 240)
  (h6 : FAB + EFA = 240) :
  FAB + CDE = 240 :=
sorry

end hexagon_angles_l113_113525


namespace find_q_l113_113433

theorem find_q (p q : ℕ) (hp_prime : Nat.Prime p) (hq_prime : Nat.Prime q) (hp_congr : 5 * p ≡ 3 [MOD 4]) (hq_def : q = 13 * p + 2) : q = 41 := 
sorry

end find_q_l113_113433


namespace correct_statements_l113_113645

variable (a b : ℝ)

theorem correct_statements (hab : a * b > 0) :
  (|a + b| > |a| ∧ |a + b| > |a - b|) ∧ (¬ (|a + b| < |b|)) ∧ (¬ (|a + b| < |a - b|)) :=
by
  -- The proof is omitted as per instructions
  sorry

end correct_statements_l113_113645


namespace blue_parrots_count_l113_113606

theorem blue_parrots_count (P : ℕ) (red green blue : ℕ) (h₁ : red = P / 2) (h₂ : green = P / 4) (h₃ : blue = P - red - green) (h₄ :  P + 30 = 150) : blue = 38 :=
by {
-- We will write the proof here
sorry
}

end blue_parrots_count_l113_113606


namespace megan_math_problems_l113_113762

theorem megan_math_problems (num_spelling_problems num_problems_per_hour num_hours total_problems num_math_problems : ℕ) 
  (h1 : num_spelling_problems = 28)
  (h2 : num_problems_per_hour = 8)
  (h3 : num_hours = 8)
  (h4 : total_problems = num_problems_per_hour * num_hours)
  (h5 : total_problems = num_spelling_problems + num_math_problems) :
  num_math_problems = 36 := 
by
  sorry

end megan_math_problems_l113_113762


namespace sum_evaluation_l113_113772

noncomputable def T : ℝ := ∑' k : ℕ, (2*k+1) / 5^(k+1)

theorem sum_evaluation : T = 5 / 16 := sorry

end sum_evaluation_l113_113772


namespace target_hit_probability_l113_113457

theorem target_hit_probability (prob_A_hits : ℝ) (prob_B_hits : ℝ) (hA : prob_A_hits = 0.5) (hB : prob_B_hits = 0.6) :
  (1 - (1 - prob_A_hits) * (1 - prob_B_hits)) = 0.8 := 
by 
  sorry

end target_hit_probability_l113_113457


namespace rachel_milk_correct_l113_113030

-- Define the initial amount of milk Don has
def don_milk : ℚ := 1 / 5

-- Define the fraction of milk Rachel drinks
def rachel_drinks_fraction : ℚ := 2 / 3

-- Define the total amount of milk Rachel drinks
def rachel_milk : ℚ := rachel_drinks_fraction * don_milk

-- The goal is to prove that Rachel drinks a specific amount of milk
theorem rachel_milk_correct : rachel_milk = 2 / 15 :=
by
  -- The proof would be here
  sorry

end rachel_milk_correct_l113_113030


namespace total_cost_in_dollars_l113_113398

def pencil_price := 20 -- price of one pencil in cents
def tolu_pencils := 3 -- pencils Tolu wants
def robert_pencils := 5 -- pencils Robert wants
def melissa_pencils := 2 -- pencils Melissa wants

theorem total_cost_in_dollars :
  (pencil_price * (tolu_pencils + robert_pencils + melissa_pencils)) / 100 = 2 := 
by
  sorry

end total_cost_in_dollars_l113_113398


namespace relationship_between_M_and_N_l113_113779

variable (a : ℝ)

def M : ℝ := 2 * a * (a - 2) + 4
def N : ℝ := (a - 1) * (a - 3)

theorem relationship_between_M_and_N : M a > N a :=
by sorry

end relationship_between_M_and_N_l113_113779


namespace arithmetic_sequence_sum_l113_113491

variable {a : ℕ → ℝ} 

-- Condition: Arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Condition: Given sum of specific terms in the sequence
def given_condition (a : ℕ → ℝ) : Prop :=
  a 2 + a 10 = 16

-- Problem: Proving the correct answer
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : given_condition a) :
  a 4 + a 6 + a 8 = 24 :=
by
  sorry

end arithmetic_sequence_sum_l113_113491


namespace minimize_cost_l113_113350

noncomputable def cost_function (x : ℝ) : ℝ :=
  (1 / 2) * (x + 5)^2 + 1000 / (x + 5)

theorem minimize_cost :
  (∀ x, 2 ≤ x ∧ x ≤ 8 → cost_function x ≥ 150) ∧ cost_function 5 = 150 :=
by
  sorry

end minimize_cost_l113_113350


namespace largest_angle_l113_113760

-- Assume the conditions
def angle_a : ℝ := 50
def angle_b : ℝ := 70
def angle_c (y : ℝ) : ℝ := 180 - (angle_a + angle_b)

-- State the proposition
theorem largest_angle (y : ℝ) (h : y = angle_c y) : angle_b = 70 := by
  sorry

end largest_angle_l113_113760


namespace prism_width_calculation_l113_113347

theorem prism_width_calculation 
  (l h d : ℝ) 
  (h_l : l = 4) 
  (h_h : h = 10) 
  (h_d : d = 14) :
  ∃ w : ℝ, w = 4 * Real.sqrt 5 ∧ (l^2 + w^2 + h^2 = d^2) := 
by
  use 4 * Real.sqrt 5
  sorry

end prism_width_calculation_l113_113347


namespace sum_of_first_seven_terms_l113_113757

variable {a_n : ℕ → ℝ} {d : ℝ}

-- Define the arithmetic progression condition.
def arithmetic_progression (a_n : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a_n n = a_n 0 + n * d

-- We are given that the sequence is an arithmetic progression.
axiom sequence_is_arithmetic_progression : arithmetic_progression a_n d

-- We are also given that the sum of the 3rd, 4th, and 5th terms is 12.
axiom sum_of_terms_is_12 : a_n 2 + a_n 3 + a_n 4 = 12

-- We need to prove that the sum of the first seven terms is 28.
theorem sum_of_first_seven_terms : (a_n 0) + (a_n 1) + (a_n 2) + (a_n 3) + (a_n 4) + (a_n 5) + (a_n 6) = 28 := 
  sorry

end sum_of_first_seven_terms_l113_113757


namespace tangent_line_equation_inequality_range_l113_113447

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem tangent_line_equation :
  let x := Real.exp 1
  ∀ e : ℝ, e = Real.exp 1 → 
  ∀ y : ℝ, y = f (Real.exp 1) → 
  ∀ a b : ℝ, (y = a * Real.exp 1 + b) ∧ (a = 2) ∧ (b = -e) := sorry

theorem inequality_range (x : ℝ) (hx : x > 0) :
  (f x - 1/2 ≤ (3/2) * x^2 + a * x) → ∀ a : ℝ, a ≥ -2 := sorry

end tangent_line_equation_inequality_range_l113_113447


namespace length_of_PQ_l113_113168

-- Definitions for the problem conditions
variable (XY UV PQ : ℝ)
variable (hXY_fixed : XY = 120)
variable (hUV_fixed : UV = 90)
variable (hParallel : XY = UV ∧ UV = PQ) -- Ensures XY || UV || PQ

-- The statement to prove
theorem length_of_PQ : PQ = 360 / 7 := by
  -- Definitions for similarity ratios and solving steps can be assumed here
  sorry

end length_of_PQ_l113_113168


namespace old_man_coins_l113_113449

theorem old_man_coins (x y : ℕ) (h : x ≠ y) (h_condition : x^2 - y^2 = 81 * (x - y)) : x + y = 81 := 
sorry

end old_man_coins_l113_113449


namespace simplify_expression_l113_113953

theorem simplify_expression : 8 * (15 / 9) * (-45 / 40) = -1 :=
  by
  sorry

end simplify_expression_l113_113953


namespace sum_in_base4_eq_in_base5_l113_113658

def base4_to_base5 (n : ℕ) : ℕ := sorry -- Placeholder for the conversion function

theorem sum_in_base4_eq_in_base5 :
  base4_to_base5 (203 + 112 + 321) = 2222 := 
sorry

end sum_in_base4_eq_in_base5_l113_113658


namespace grey_area_of_first_grid_is_16_grey_area_of_second_grid_is_15_white_area_of_third_grid_is_5_l113_113830

theorem grey_area_of_first_grid_is_16 (side_length : ℝ := 1) :
  let area_triangle (base height : ℝ) := 0.5 * base * height
  let area_rectangle (length width : ℝ) := length * width
  let grey_area := area_triangle 3 side_length 
                    + area_triangle 4 side_length 
                    + area_rectangle 6 side_length 
                    + area_triangle 2 side_length 
                    + area_triangle 2 side_length 
                    + area_rectangle 2 side_length 
                    + area_triangle 2 side_length 
                    + area_triangle 3 side_length
  grey_area = 16 := by
  sorry

theorem grey_area_of_second_grid_is_15 (side_length : ℝ := 1) :
  let area_triangle (base height : ℝ) := 0.5 * base * height
  let area_rectangle (length width : ℝ) := length * width
  let grey_area := area_triangle 4 side_length 
                    + area_rectangle 2 side_length
                    + area_triangle 6 side_length 
                    + area_rectangle 2 side_length 
                    + area_triangle 2 side_length 
                    + area_triangle 2 side_length 
                    + area_rectangle 4 side_length
  grey_area = 15 := by
  sorry

theorem white_area_of_third_grid_is_5 (total_rectangle_area dark_grey_area : ℝ) (grey_area1 grey_area2 : ℝ) :
    total_rectangle_area = 32 ∧ dark_grey_area = 4 ∧ grey_area1 = 16 ∧ grey_area2 = 15 →
    let total_grey_area_recounted := grey_area1 + grey_area2 - dark_grey_area
    let white_area := total_rectangle_area - total_grey_area_recounted
    white_area = 5 := by
  sorry

end grey_area_of_first_grid_is_16_grey_area_of_second_grid_is_15_white_area_of_third_grid_is_5_l113_113830


namespace linear_function_no_second_quadrant_l113_113920

theorem linear_function_no_second_quadrant (k : ℝ) :
  (∀ x : ℝ, (y : ℝ) → y = k * x - k + 3 → ¬(x < 0 ∧ y > 0)) ↔ k ≥ 3 :=
sorry

end linear_function_no_second_quadrant_l113_113920


namespace range_of_k_l113_113882

noncomputable def f (k : ℝ) (x : ℝ) := 1 - k * x^2
noncomputable def g (x : ℝ) := Real.cos x

theorem range_of_k (k : ℝ) : (∀ x : ℝ, f k x < g x) ↔ k ≥ (1 / 2) :=
by
  sorry

end range_of_k_l113_113882


namespace freight_capacity_equation_l113_113208

theorem freight_capacity_equation
  (x : ℝ)
  (h1 : ∀ (capacity_large capacity_small : ℝ), capacity_large = capacity_small + 4)
  (h2 : ∀ (n_large n_small : ℕ), (n_large : ℝ) = 80 / (x + 4) ∧ (n_small : ℝ) = 60 / x → n_large = n_small) :
  (80 / (x + 4)) = (60 / x) :=
by
  sorry

end freight_capacity_equation_l113_113208


namespace prob1_prob2_prob3_l113_113811

def star (a b : ℤ) : ℤ :=
  if a = 0 then b^2
  else if b = 0 then a^2
  else if a > 0 ∧ b > 0 then a^2 + b^2
  else if a < 0 ∧ b < 0 then a^2 + b^2
  else -(a^2 + b^2)

theorem prob1 :
  star (-1) (-1) = 2 :=
sorry

theorem prob2 :
  star (-1) (star 0 (-2)) = -17 :=
sorry

theorem prob3 (m n : ℤ) :
  star (m-1) (n+2) = -2 → (m - n = 1 ∨ m - n = 5) :=
sorry

end prob1_prob2_prob3_l113_113811


namespace FlyersDistributon_l113_113152

variable (total_flyers ryan_flyers alyssa_flyers belinda_percentage : ℕ)
variable (scott_flyers : ℕ)

theorem FlyersDistributon (H : total_flyers = 200)
  (H1 : ryan_flyers = 42)
  (H2 : alyssa_flyers = 67)
  (H3 : belinda_percentage = 20)
  (H4 : scott_flyers = total_flyers - (ryan_flyers + alyssa_flyers + (belinda_percentage * total_flyers) / 100)) :
  scott_flyers = 51 :=
by
  simp [H, H1, H2, H3] at H4
  exact H4

end FlyersDistributon_l113_113152


namespace nehas_mother_age_l113_113909

variables (N M : ℕ)

axiom age_condition1 : M - 12 = 4 * (N - 12)
axiom age_condition2 : M + 12 = 2 * (N + 12)

theorem nehas_mother_age : M = 60 :=
by
  -- Sorry added to skip the proof
  sorry

end nehas_mother_age_l113_113909


namespace regular_polygon_sides_l113_113818

theorem regular_polygon_sides (n : ℕ) (h : ∀ i < n, (interior_angle_i : ℝ) = 150) :
  (n = 12) :=
by
  sorry

end regular_polygon_sides_l113_113818


namespace rhombus_area_l113_113660

def diagonal1 : ℝ := 24
def diagonal2 : ℝ := 16

theorem rhombus_area : 0.5 * diagonal1 * diagonal2 = 192 :=
by
  sorry

end rhombus_area_l113_113660


namespace relationship_between_k_and_c_l113_113191

-- Define the functions and given conditions
def y1 (x : ℝ) (c : ℝ) : ℝ := x^2 + 2*x + c
def y2 (x : ℝ) (k : ℝ) : ℝ := k*x + 2

-- Define the vertex of y1
def vertex_y1 (c : ℝ) : ℝ × ℝ := (-1, c - 1)

-- State the main theorem
theorem relationship_between_k_and_c (k c : ℝ) (hk : k ≠ 0) :
  y2 (vertex_y1 c).1 k = (vertex_y1 c).2 → c + k = 3 :=
by
  sorry

end relationship_between_k_and_c_l113_113191


namespace monthly_rent_is_3600_rs_l113_113970

def shop_length_feet : ℕ := 20
def shop_width_feet : ℕ := 15
def annual_rent_per_square_foot_rs : ℕ := 144

theorem monthly_rent_is_3600_rs :
  (shop_length_feet * shop_width_feet) * annual_rent_per_square_foot_rs / 12 = 3600 :=
by sorry

end monthly_rent_is_3600_rs_l113_113970


namespace beats_per_week_l113_113758

def beats_per_minute : ℕ := 200
def minutes_per_hour : ℕ := 60
def hours_per_day : ℕ := 2
def days_per_week : ℕ := 7

theorem beats_per_week : beats_per_minute * minutes_per_hour * hours_per_day * days_per_week = 168000 := by
  sorry

end beats_per_week_l113_113758


namespace jason_pears_count_l113_113425

theorem jason_pears_count 
  (initial_pears : ℕ)
  (given_to_keith : ℕ)
  (received_from_mike : ℕ)
  (final_pears : ℕ)
  (h_initial : initial_pears = 46)
  (h_given : given_to_keith = 47)
  (h_received : received_from_mike = 12)
  (h_final : final_pears = 12) :
  initial_pears - given_to_keith + received_from_mike = final_pears :=
sorry

end jason_pears_count_l113_113425


namespace triangle_side_relation_triangle_perimeter_l113_113153

theorem triangle_side_relation (a b c : ℝ) (A B C : ℝ)
  (h1 : a / (Real.sin A) = b / (Real.sin B)) (h2 : a / (Real.sin A) = c / (Real.sin C))
  (h3 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) :
  2 * a ^ 2 = b ^ 2 + c ^ 2 := sorry

theorem triangle_perimeter (a b c : ℝ) (A : ℝ) (hcosA : Real.cos A = 25 / 31)
  (h1 : a / (Real.sin A) = b / (Real.sin B)) (h2 : a / (Real.sin A) = c / (Real.sin C))
  (h3 : Real.sin C * Real.sin (A - B) = Real.sin B * Real.sin (C - A)) (ha : a = 5) :
  a + b + c = 14 := sorry

end triangle_side_relation_triangle_perimeter_l113_113153


namespace initial_nickels_l113_113511

theorem initial_nickels (quarters : ℕ) (initial_nickels : ℕ) (borrowed_nickels : ℕ) (current_nickels : ℕ) 
  (H1 : initial_nickels = 87) (H2 : borrowed_nickels = 75) (H3 : current_nickels = 12) : 
  initial_nickels = current_nickels + borrowed_nickels := 
by 
  -- proof steps go here
  sorry

end initial_nickels_l113_113511


namespace card_probability_l113_113502

-- Definitions of the conditions
def is_multiple (n d : ℕ) : Prop := d ∣ n

def count_multiples (d m : ℕ) : ℕ := (m / d)

def multiples_in_range (n : ℕ) : ℕ := 
  count_multiples 2 n + count_multiples 3 n + count_multiples 5 n
  - count_multiples 6 n - count_multiples 10 n - count_multiples 15 n 
  + count_multiples 30 n

def probability_of_multiples_in_range (n : ℕ) : ℚ := 
  multiples_in_range n / n 

-- Proof statement
theorem card_probability (n : ℕ) (h : n = 120) : probability_of_multiples_in_range n = 11 / 15 :=
  sorry

end card_probability_l113_113502


namespace find_common_ratio_l113_113167

variable (a₁ : ℝ) (q : ℝ)

def S₁ (a₁ : ℝ) : ℝ := a₁
def S₃ (a₁ q : ℝ) : ℝ := a₁ + a₁ * q + a₁ * q ^ 2
def a₃ (a₁ q : ℝ) : ℝ := a₁ * q ^ 2

theorem find_common_ratio (h : 2 * S₃ a₁ q = S₁ a₁ + 2 * a₃ a₁ q) : q = -1 / 2 :=
by
  sorry

end find_common_ratio_l113_113167


namespace conference_duration_excluding_breaks_l113_113415

-- Definitions based on the conditions
def total_hours : Nat := 14
def additional_minutes : Nat := 20
def break_minutes : Nat := 15

-- Total time including breaks
def total_time_minutes : Nat := total_hours * 60 + additional_minutes
-- Number of breaks
def number_of_breaks : Nat := total_hours
-- Total break time
def total_break_minutes : Nat := number_of_breaks * break_minutes

-- Proof statement
theorem conference_duration_excluding_breaks :
  total_time_minutes - total_break_minutes = 650 := by
  sorry

end conference_duration_excluding_breaks_l113_113415


namespace custom_mul_4_3_l113_113383

-- Define the binary operation a*b = a^2 - ab + b^2
def custom_mul (a b : ℕ) : ℕ := a^2 - a*b + b^2

-- State the theorem to prove that 4 * 3 = 13
theorem custom_mul_4_3 : custom_mul 4 3 = 13 := by
  sorry -- Proof will be filled in here

end custom_mul_4_3_l113_113383


namespace problem_solution_l113_113233

theorem problem_solution (u v : ℤ) (h₁ : 0 < v) (h₂ : v < u) (h₃ : u^2 + 3 * u * v = 451) : u + v = 21 :=
sorry

end problem_solution_l113_113233


namespace find_alpha_l113_113716

theorem find_alpha (α : ℝ) :
    7 * α + 8 * α + 45 = 180 →
    α = 9 :=
by
  sorry

end find_alpha_l113_113716


namespace homework_duration_decrease_l113_113796

variable (a b x : ℝ)

theorem homework_duration_decrease (h: a * (1 - x)^2 = b) :
  a * (1 - x)^2 = b := 
by
  sorry

end homework_duration_decrease_l113_113796


namespace average_age_add_person_l113_113033

theorem average_age_add_person (n : ℕ) (h1 : (∀ T, T = n * 14 → (T + 34) / (n + 1) = 16)) : n = 9 :=
by
  sorry

end average_age_add_person_l113_113033


namespace convex_polygons_from_fifteen_points_l113_113759

theorem convex_polygons_from_fifteen_points 
    (h : ∀ (n : ℕ), n = 15) :
    ∃ (k : ℕ), k = 32192 :=
by
  sorry

end convex_polygons_from_fifteen_points_l113_113759


namespace quadratic_solution_l113_113911

theorem quadratic_solution (x : ℝ) : (x^2 + 6 * x + 8 = -2 * (x + 4) * (x + 5)) ↔ (x = -8 ∨ x = -4) :=
by
  sorry

end quadratic_solution_l113_113911


namespace find_radius_l113_113775

noncomputable def radius (π : ℝ) : Prop :=
  ∃ r : ℝ, π * r^2 + 2 * r - 2 * π * r = 12 ∧ r = Real.sqrt (12 / π)

theorem find_radius (π : ℝ) (hπ : π > 0) : 
  radius π :=
sorry

end find_radius_l113_113775


namespace nine_chapters_problem_l113_113503

theorem nine_chapters_problem (n x : ℤ) (h1 : 8 * n = x + 3) (h2 : 7 * n = x - 4) :
  (x + 3) / 8 = (x - 4) / 7 :=
  sorry

end nine_chapters_problem_l113_113503


namespace negation_of_proposition_p_l113_113974

def f : ℝ → ℝ := sorry

theorem negation_of_proposition_p :
  (¬ (∀ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) ≥ 0)) ↔ (∃ x₁ x₂ : ℝ, (f x₂ - f x₁) * (x₂ - x₁) < 0) := 
by
  sorry

end negation_of_proposition_p_l113_113974


namespace luke_initial_stickers_l113_113247

theorem luke_initial_stickers (x : ℕ) (h : x + 12 + 20 - 5 - 8 = 39) : x = 20 := 
by 
  sorry

end luke_initial_stickers_l113_113247


namespace divisor_correct_l113_113937

/--
Given that \(10^{23} - 7\) divided by \(d\) leaves a remainder 3, 
prove that \(d\) is equal to \(10^{23} - 10\).
-/
theorem divisor_correct :
  ∃ d : ℤ, (10^23 - 7) % d = 3 ∧ d = 10^23 - 10 :=
by
  sorry

end divisor_correct_l113_113937


namespace total_lifespan_l113_113943

theorem total_lifespan (B H F : ℕ)
  (hB : B = 10)
  (hH : H = B - 6)
  (hF : F = 4 * H) :
  B + H + F = 30 := by
  sorry

end total_lifespan_l113_113943


namespace find_m_l113_113916

theorem find_m (m : ℝ) (α : ℝ) (h_cos : Real.cos α = -3/5) (h_p : ((Real.cos α = m / (Real.sqrt (m^2 + 4^2)))) ∧ (Real.cos α < 0) ∧ (m < 0)) :

  m = -3 :=
by 
  sorry

end find_m_l113_113916


namespace necessary_but_not_sufficient_not_sufficient_l113_113553

def P (x : ℝ) : Prop := x < 1
def Q (x : ℝ) : Prop := (x + 2) * (x - 1) < 0

theorem necessary_but_not_sufficient (x : ℝ) : P x → Q x := by
  intro hx
  sorry

theorem not_sufficient (x : ℝ) : ¬(Q x → P x) := by
  intro hq
  sorry

end necessary_but_not_sufficient_not_sufficient_l113_113553


namespace return_trip_time_is_15_or_67_l113_113560

variable (d p w : ℝ)

-- Conditions
axiom h1 : (d / (p - w)) = 100
axiom h2 : ∃ t : ℝ, t = d / p ∧ (d / (p + w)) = t - 15

-- Correct answer to prove: time for the return trip is 15 minutes or 67 minutes
theorem return_trip_time_is_15_or_67 : (d / (p + w)) = 15 ∨ (d / (p + w)) = 67 := 
by 
  sorry

end return_trip_time_is_15_or_67_l113_113560


namespace henry_books_l113_113214

theorem henry_books (initial_books packed_boxes each_box room_books coffee_books kitchen_books taken_books : ℕ)
  (h1 : initial_books = 99)
  (h2 : packed_boxes = 3)
  (h3 : each_box = 15)
  (h4 : room_books = 21)
  (h5 : coffee_books = 4)
  (h6 : kitchen_books = 18)
  (h7 : taken_books = 12) :
  initial_books - (packed_boxes * each_box + room_books + coffee_books + kitchen_books) + taken_books = 23 :=
by
  sorry

end henry_books_l113_113214


namespace ten_percent_of_x_l113_113111

theorem ten_percent_of_x
  (x : ℝ)
  (h : 3 - (1 / 4) * 2 - (1 / 3) * 3 - (1 / 7) * x = 27) :
  0.10 * x = 17.85 :=
by
  -- theorem proof goes here
  sorry

end ten_percent_of_x_l113_113111


namespace boy_completes_work_in_nine_days_l113_113872

theorem boy_completes_work_in_nine_days :
  let M := (1 : ℝ) / 6
  let W := (1 : ℝ) / 18
  let B := (1 / 3 : ℝ) - M - W
  B = (1 : ℝ) / 9 := by
    sorry

end boy_completes_work_in_nine_days_l113_113872


namespace time_to_write_numbers_in_minutes_l113_113815

theorem time_to_write_numbers_in_minutes : 
  (1 * 5 + 2 * (99 - 10 + 1) + 3 * (105 - 100 + 1)) / 60 = 4 := 
  by
  -- Calculation steps would go here
  sorry

end time_to_write_numbers_in_minutes_l113_113815


namespace cone_base_circumference_l113_113414

theorem cone_base_circumference 
  (r : ℝ) 
  (θ : ℝ) 
  (h₁ : r = 5) 
  (h₂ : θ = 225) : 
  (θ / 360 * 2 * Real.pi * r) = (25 * Real.pi / 4) :=
by
  -- Proof skipped
  sorry

end cone_base_circumference_l113_113414


namespace range_of_a_l113_113926

open Set

def p (a : ℝ) := ∀ x : ℝ, x^2 + 2 * a * x + 4 > 0
def q (a : ℝ) := ∀ x : ℝ, x ∈ (Icc 1 2) → x^2 ≥ a

theorem range_of_a (a : ℝ) : 
  (p a ∨ q a) ∧ ¬(p a ∧ q a) ↔ a ∈ (Ioo 1 2 ∪ Iic (-2)) :=
by sorry

end range_of_a_l113_113926


namespace total_amount_is_24_l113_113228

-- Define the original price of a tub of ice cream
def original_price_ice_cream : ℕ := 12

-- Define the discount per tub of ice cream
def discount_per_tub : ℕ := 2

-- Define the discounted price of a tub of ice cream
def discounted_price_ice_cream : ℕ := original_price_ice_cream - discount_per_tub

-- Define the price for 5 cans of juice
def price_per_5_cans_of_juice : ℕ := 2

-- Define the number of cans of juice bought
def cans_of_juice_bought : ℕ := 10

-- Calculate the total cost for two tubs of ice cream and 10 cans of juice
def total_cost (p1 p2 : ℕ) : ℕ := 2 * p1 + (price_per_5_cans_of_juice * (cans_of_juice_bought / 5))

-- Prove that the total cost is $24
theorem total_amount_is_24 : total_cost discounted_price_ice_cream price_per_5_cans_of_juice = 24 := by
  sorry

end total_amount_is_24_l113_113228


namespace smallest_positive_period_of_f_range_of_f_in_interval_l113_113446

noncomputable def f (a : ℝ) (x : ℝ) := a * Real.sin x - Real.sqrt 3 * Real.cos x

theorem smallest_positive_period_of_f (a : ℝ) (h : f a (π / 3) = 0) :
  ∃ T : ℝ, T = 2 * π ∧ (∀ x, f a (x + T) = f a x) :=
sorry

theorem range_of_f_in_interval (a : ℝ) (h : f a (π / 3) = 0) :
  ∀ x ∈ Set.Icc (π / 2) (3 * π / 2), -1 ≤ f a x ∧ f a x ≤ 2 :=
sorry

end smallest_positive_period_of_f_range_of_f_in_interval_l113_113446


namespace cuboid_face_areas_l113_113659

-- Conditions
variables (a b c S : ℝ)
-- Surface area of the sphere condition
theorem cuboid_face_areas 
  (h1 : a * b = 6) 
  (h2 : b * c = 10) 
  (h3 : a^2 + b^2 + c^2 = 76) 
  (h4 : 4 * π * 38 = 152 * π) :
  a * c = 15 :=
by 
  -- Prove that the solution matches the conclusion
  sorry

end cuboid_face_areas_l113_113659


namespace combined_moment_l113_113427

-- Definitions based on given conditions
variables (P Q Z : ℝ) -- Positions of the points and center of mass
variables (p q : ℝ) -- Masses of the points
variables (Mom_s : ℝ → ℝ) -- Moment function relative to axis s

-- Given:
-- 1. Positions P and Q with masses p and q respectively
-- 2. Combined point Z with total mass p + q
-- 3. Moments relative to the axis s: Mom_s P and Mom_s Q
-- To Prove: Moment of the combined point Z relative to axis s
-- is the sum of the moments of P and Q relative to the same axis

theorem combined_moment (hZ : Z = (P * p + Q * q) / (p + q)) :
  Mom_s Z = Mom_s P + Mom_s Q :=
sorry

end combined_moment_l113_113427


namespace circle_problem_l113_113621

theorem circle_problem (P : ℝ × ℝ) (QR : ℝ) (S : ℝ × ℝ) (k : ℝ)
  (h1 : P = (5, 12))
  (h2 : QR = 5)
  (h3 : S = (0, k))
  (h4 : dist (0,0) P = 13) -- OP = 13 from the origin to point P
  (h5 : dist (0,0) S = 8) -- OQ = 8 from the origin to point S
: k = 8 ∨ k = -8 :=
by sorry

end circle_problem_l113_113621


namespace proof_not_sufficient_nor_necessary_l113_113875

noncomputable def not_sufficient_nor_necessary (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) : Prop :=
  ¬ ((a > b) → (Real.log b / Real.log a < 1)) ∧ ¬ ((Real.log b / Real.log a < 1) → (a > b))

theorem proof_not_sufficient_nor_necessary (a b: ℝ) (h₁: 0 < a) (h₂: 0 < b) :
  not_sufficient_nor_necessary a b h₁ h₂ :=
  sorry

end proof_not_sufficient_nor_necessary_l113_113875


namespace minimum_value_occurs_at_4_l113_113698

noncomputable def minimum_value_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y, f x ≤ f y

def quadratic_expression (x : ℝ) : ℝ := x^2 - 8 * x + 15

theorem minimum_value_occurs_at_4 :
  minimum_value_at quadratic_expression 4 :=
sorry

end minimum_value_occurs_at_4_l113_113698


namespace find_a_l113_113055

theorem find_a (a : ℝ) :
  (∀ x : ℝ, ((x^2 - 4 * x + a) + |x - 3| ≤ 5) → x ≤ 3) →
  (∃ x : ℝ, x = 3 ∧ ((x^2 - 4 * x + a) + |x - 3| ≤ 5)) →
  a = 2 := 
by
  sorry

end find_a_l113_113055


namespace relationship_between_abc_l113_113272

noncomputable def a : Real := (2 / 5) ^ (3 / 5)
noncomputable def b : Real := (2 / 5) ^ (2 / 5)
noncomputable def c : Real := (3 / 5) ^ (3 / 5)

theorem relationship_between_abc : a < b ∧ b < c := by
  sorry

end relationship_between_abc_l113_113272


namespace playground_area_22500_l113_113923

noncomputable def rectangle_playground_area (w l : ℕ) : ℕ :=
  w * l

theorem playground_area_22500 (w l : ℕ) (h1 : l = 2 * w + 25) (h2 : 2 * l + 2 * w = 650) :
  rectangle_playground_area w l = 22500 := by
  sorry

end playground_area_22500_l113_113923


namespace simplify_and_evaluate_expression_l113_113474

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 3) : 
  ((x^2 / (x - 2) - x - 2) / (4 * x / (x^2 - 4))) = (5 : ℝ) / 3 := 
by
  sorry

end simplify_and_evaluate_expression_l113_113474


namespace maximize_area_of_sector_l113_113006

noncomputable def area_of_sector (x y : ℝ) : ℝ := (1 / 2) * x * y

theorem maximize_area_of_sector : 
  ∃ x y : ℝ, 2 * x + y = 20 ∧ (∀ (x : ℝ), x > 0 → 
  (∀ (y : ℝ), y > 0 → 2 * x + y = 20 → area_of_sector x y ≤ area_of_sector 5 (20 - 2 * 5))) ∧ x = 5 :=
by
  sorry

end maximize_area_of_sector_l113_113006


namespace valid_q_range_l113_113248

noncomputable def polynomial_has_nonneg_root (q : ℝ) : Prop :=
  ∃ x : ℝ, x ≥ 0 ∧ (x^4 + q*x^3 + x^2 + q*x + 4 = 0)

theorem valid_q_range (q : ℝ) : polynomial_has_nonneg_root q → q ≤ -2 * Real.sqrt 2 := 
sorry

end valid_q_range_l113_113248


namespace percentage_problem_l113_113963

variable (y x z : ℝ)

def A := y * x^2 + 3 * z - 6

theorem percentage_problem (h : A y x z > 0) :
  (2 * A y x z / 5) + (3 * A y x z / 10) = (70 / 100) * A y x z :=
by
  sorry

end percentage_problem_l113_113963


namespace relay_team_orders_l113_113114

noncomputable def jordan_relay_orders : Nat :=
  let friends := [1, 2, 3] -- Differentiate friends; let's represent A by 1, B by 2, C by 3
  let choices_for_jordan_third := 2 -- Ways if Jordan runs third
  let choices_for_jordan_fourth := 2 -- Ways if Jordan runs fourth
  choices_for_jordan_third + choices_for_jordan_fourth

theorem relay_team_orders :
  jordan_relay_orders = 4 :=
by
  sorry

end relay_team_orders_l113_113114


namespace percentage_of_literate_females_is_32_5_l113_113079

noncomputable def percentage_literate_females (inhabitants : ℕ) (percent_male : ℝ) (percent_literate_males : ℝ) (percent_literate_total : ℝ) : ℝ :=
  let males := (percent_male / 100) * inhabitants
  let females := inhabitants - males
  let literate_males := (percent_literate_males / 100) * males
  let literate_total := (percent_literate_total / 100) * inhabitants
  let literate_females := literate_total - literate_males
  (literate_females / females) * 100

theorem percentage_of_literate_females_is_32_5 :
  percentage_literate_females 1000 60 20 25 = 32.5 := 
by 
  unfold percentage_literate_females
  sorry

end percentage_of_literate_females_is_32_5_l113_113079


namespace Nick_sister_age_l113_113110

theorem Nick_sister_age
  (Nick_age : ℕ := 13)
  (Bro_in_5_years : ℕ := 21)
  (H : ∃ S : ℕ, (Nick_age + S) / 2 + 5 = Bro_in_5_years) :
  ∃ S : ℕ, S = 19 :=
by
  sorry

end Nick_sister_age_l113_113110


namespace field_ratio_l113_113997

theorem field_ratio (l w : ℕ) (h_l : l = 20) (pond_side : ℕ) (h_pond_side : pond_side = 5)
  (h_area_pond : pond_side * pond_side = (1 / 8 : ℚ) * l * w) : l / w = 2 :=
by 
  sorry

end field_ratio_l113_113997


namespace fibonacci_arithmetic_sequence_l113_113154

def fibonacci : ℕ → ℕ
| 0       => 0
| 1       => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_arithmetic_sequence (a b c n : ℕ) 
  (h1 : fibonacci 1 = 1)
  (h2 : fibonacci 2 = 1)
  (h3 : ∀ n ≥ 3, fibonacci n = fibonacci (n - 1) + fibonacci (n - 2))
  (h4 : a + b + c = 2500)
  (h5 : (a, b, c) = (n, n + 3, n + 5)) :
  a = 831 := 
sorry

end fibonacci_arithmetic_sequence_l113_113154


namespace triangle_area_l113_113417

theorem triangle_area (X Y Z : ℝ) (r R : ℝ)
  (h1 : r = 7)
  (h2 : R = 25)
  (h3 : 2 * Real.cos Y = Real.cos X + Real.cos Z) :
  ∃ (p q r : ℕ), (p * Real.sqrt q / r = 133) ∧ (p + q + r = 135) :=
  sorry

end triangle_area_l113_113417


namespace ratio_of_third_to_second_l113_113469

-- Assume we have three numbers (a, b, c) where
-- 1. b = 2 * a
-- 2. c = k * b
-- 3. (a + b + c) / 3 = 165
-- 4. a = 45

theorem ratio_of_third_to_second (a b c k : ℝ) (h1 : b = 2 * a) (h2 : c = k * b) 
  (h3 : (a + b + c) / 3 = 165) (h4 : a = 45) : k = 4 := by 
  sorry

end ratio_of_third_to_second_l113_113469


namespace candy_cost_l113_113810

-- Definitions and assumptions from problem conditions
def cents_per_page := 1
def pages_per_book := 150
def books_read := 12
def leftover_cents := 300  -- $3 in cents

-- Total pages read
def total_pages_read := pages_per_book * books_read

-- Total earnings in cents
def total_cents_earned := total_pages_read * cents_per_page

-- Cost of the candy in cents
def candy_cost_cents := total_cents_earned - leftover_cents

-- Theorem statement
theorem candy_cost : candy_cost_cents = 1500 := 
  by 
    -- proof goes here
    sorry

end candy_cost_l113_113810


namespace sara_spent_on_hotdog_l113_113212

-- Define variables for the costs
def costSalad : ℝ := 5.1
def totalLunchBill : ℝ := 10.46

-- Define the cost of the hotdog
def costHotdog : ℝ := totalLunchBill - costSalad

-- The theorem we need to prove
theorem sara_spent_on_hotdog : costHotdog = 5.36 := by
  -- Proof would go here (if required)
  sorry

end sara_spent_on_hotdog_l113_113212


namespace fiona_weekly_earnings_l113_113615

theorem fiona_weekly_earnings :
  let monday_hours := 1.5
  let tuesday_hours := 1.25
  let wednesday_hours := 3.1667
  let thursday_hours := 0.75
  let hourly_wage := 4
  let total_hours := monday_hours + tuesday_hours + wednesday_hours + thursday_hours
  let total_earnings := total_hours * hourly_wage
  total_earnings = 26.67 := by
  sorry

end fiona_weekly_earnings_l113_113615


namespace dog_rabbit_age_ratio_l113_113785

-- Definitions based on conditions
def cat_age := 8
def rabbit_age := cat_age / 2
def dog_age := 12
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b

-- Theorem statement
theorem dog_rabbit_age_ratio : is_multiple dog_age rabbit_age ∧ dog_age / rabbit_age = 3 :=
by
  sorry

end dog_rabbit_age_ratio_l113_113785


namespace value_of_x_squared_plus_y_squared_l113_113824

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -6) : x^2 + y^2 = 16 :=
by
  sorry

end value_of_x_squared_plus_y_squared_l113_113824


namespace correct_exponent_calculation_l113_113473

theorem correct_exponent_calculation : 
(∀ (a b : ℝ), (a + b)^2 ≠ a^2 + b^2) ∧
(∀ (a : ℝ), a^9 / a^3 ≠ a^3) ∧
(∀ (a b : ℝ), (ab)^3 = a^3 * b^3) ∧
(∀ (a : ℝ), (a^5)^2 ≠ a^7) :=
by 
  sorry

end correct_exponent_calculation_l113_113473


namespace parallel_lines_m_eq_l113_113201

theorem parallel_lines_m_eq (m : ℝ) : 
  (∃ k : ℝ, (x y : ℝ) → 2 * x + (m + 1) * y + 4 = k * (m * x + 3 * y - 2)) → 
  (m = 2 ∨ m = -3) :=
by
  intro h
  sorry

end parallel_lines_m_eq_l113_113201


namespace sum_of_three_numbers_is_71_point_5_l113_113501

noncomputable def sum_of_three_numbers (a b c : ℝ) : ℝ :=
a + b + c

theorem sum_of_three_numbers_is_71_point_5 (a b c : ℝ) (h1 : a + b = 35) (h2 : b + c = 48) (h3 : c + a = 60) :
  sum_of_three_numbers a b c = 71.5 :=
by
  unfold sum_of_three_numbers
  sorry

end sum_of_three_numbers_is_71_point_5_l113_113501


namespace adults_had_meal_l113_113846

theorem adults_had_meal (A : ℕ) (h1 : 70 ≥ A) (h2 : ((70 - A) * 9) = (72 * 7)) : A = 14 := 
by
  sorry

end adults_had_meal_l113_113846


namespace number_of_b_values_l113_113227

-- Let's define the conditions and the final proof required.
def inequations (x b : ℤ) : Prop := 
  (3 * x > 4 * x - 4) ∧
  (4 * x - b > -8) ∧
  (5 * x < b + 13)

theorem number_of_b_values :
  (∀ x : ℤ, 1 ≤ x → x ≠ 3 → ¬ inequations x b) →
  (∃ (b_values : Finset ℤ), 
      (∀ b ∈ b_values, inequations 3 b) ∧ 
      (b_values.card = 7)) :=
sorry

end number_of_b_values_l113_113227


namespace s_neq_t_if_Q_on_DE_l113_113727

-- Conditions and Definitions
noncomputable def DQ (x : ℝ) := x
noncomputable def QE (x : ℝ) := 10 - x
noncomputable def FQ := 5 * Real.sqrt 3
noncomputable def s (x : ℝ) := (DQ x) ^ 2 + (QE x) ^ 2
noncomputable def t := 2 * FQ ^ 2

-- Lean 4 Statement
theorem s_neq_t_if_Q_on_DE (x : ℝ) : s x ≠ t :=
by
  sorry -- Provided proof step to be filled in

end s_neq_t_if_Q_on_DE_l113_113727


namespace count_prime_digit_sums_less_than_10_l113_113518

def is_prime (n : ℕ) : Prop :=
n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ := n / 10 + n % 10

def is_two_digit_number (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem count_prime_digit_sums_less_than_10 :
  ∃ count : ℕ, count = 17 ∧
  ∀ n : ℕ, is_two_digit_number n →
  (is_prime (sum_of_digits n) ∧ sum_of_digits n < 10) ↔
  n ∈ [11, 20, 12, 21, 30, 14, 23, 32, 41, 50, 16, 25, 34, 43, 52, 61, 70] :=
sorry

end count_prime_digit_sums_less_than_10_l113_113518


namespace min_k_squared_floor_l113_113626

open Nat

theorem min_k_squared_floor (n : ℕ) :
  (∀ k : ℕ, k >= 1 → k^2 + (n / k^2) ≥ 1991) ∧
  (∃ k : ℕ, k >= 1 ∧ k^2 + (n / k^2) < 1992) ↔
  1024 * 967 ≤ n ∧ n ≤ 1024 * 967 + 1023 := 
by
  sorry

end min_k_squared_floor_l113_113626


namespace value_of_a_l113_113905

noncomputable def a : ℕ := 4

def A : Set ℕ := {0, 2, a}
def B : Set ℕ := {1, a*a}
def C : Set ℕ := {0, 1, 2, 4, 16}

theorem value_of_a : A ∪ B = C → a = 4 := by
  intro h
  sorry

end value_of_a_l113_113905


namespace toms_total_profit_l113_113688

def total_earnings_mowing : ℕ := 4 * 12 + 3 * 15 + 1 * 20
def total_earnings_side_jobs : ℕ := 2 * 10 + 3 * 8 + 1 * 12
def total_earnings : ℕ := total_earnings_mowing + total_earnings_side_jobs
def total_expenses : ℕ := 17 + 5
def total_profit : ℕ := total_earnings - total_expenses

theorem toms_total_profit : total_profit = 147 := by
  -- Proof omitted
  sorry

end toms_total_profit_l113_113688


namespace polynomial_bound_swap_l113_113175

variable (a b c : ℝ)

theorem polynomial_bound_swap (h : ∀ x : ℝ, |x| ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ (x : ℝ), |x| ≤ 1 → |c * x^2 + b * x + a| ≤ 2 := by
  sorry

end polynomial_bound_swap_l113_113175


namespace kaleb_non_working_games_l113_113627

theorem kaleb_non_working_games (total_games working_game_price earning : ℕ) (h1 : total_games = 10) (h2 : working_game_price = 6) (h3 : earning = 12) :
  total_games - (earning / working_game_price) = 8 :=
by
  sorry

end kaleb_non_working_games_l113_113627


namespace john_tv_show_duration_l113_113984

def john_tv_show (seasons_before : ℕ) (episodes_per_season : ℕ) (additional_episodes : ℕ) (episode_duration : ℝ) : ℝ :=
  let total_episodes_before := seasons_before * episodes_per_season
  let last_season_episodes := episodes_per_season + additional_episodes
  let total_episodes := total_episodes_before + last_season_episodes
  total_episodes * episode_duration

theorem john_tv_show_duration :
  john_tv_show 9 22 4 0.5 = 112 := 
by
  sorry

end john_tv_show_duration_l113_113984


namespace proof_problem_l113_113140

variables (a b : ℝ)
variable (h : a ≠ b)
variable (h1 : a * Real.exp a = b * Real.exp b)
variable (p : Prop := Real.log a + a = Real.log b + b)
variable (q : Prop := (a + 1) * (b + 1) < 0)

theorem proof_problem : p ∨ q :=
sorry

end proof_problem_l113_113140


namespace largest_n_for_divisibility_l113_113617

theorem largest_n_for_divisibility :
  ∃ (n : ℕ), n = 5 ∧ 3^n ∣ (4^27000 - 82) ∧ ¬ 3^(n + 1) ∣ (4^27000 - 82) :=
by
  sorry

end largest_n_for_divisibility_l113_113617


namespace cicely_100th_birthday_l113_113924

-- Definition of the conditions
def birth_year (birthday_year : ℕ) (birthday_age : ℕ) : ℕ :=
  birthday_year - birthday_age

def birthday (birth_year : ℕ) (age : ℕ) : ℕ :=
  birth_year + age

-- The problem restatement in Lean 4
theorem cicely_100th_birthday (birthday_year : ℕ) (birthday_age : ℕ) (expected_year : ℕ) :
  birthday_year = 1939 → birthday_age = 21 → expected_year = 2018 → birthday (birth_year birthday_year birthday_age) 100 = expected_year :=
by
  intros h1 h2 h3
  rw [birthday, birth_year]
  rw [h1, h2]
  sorry

end cicely_100th_birthday_l113_113924


namespace rugged_terrain_distance_ratio_l113_113401

theorem rugged_terrain_distance_ratio (D k : ℝ) 
  (hD : D > 0) 
  (hk : k > 0) 
  (v_M v_P : ℝ) 
  (hm : v_M = 2 * k) 
  (hp : v_P = 3 * k)
  (v_Mr v_Pr : ℝ) 
  (hmr : v_Mr = k) 
  (hpr : v_Pr = 3 * k / 2) :
  ∀ (x y a b : ℝ), (x + y = D / 2) → (a + b = D / 2) → (y + b = 2 * D / 3) →
  (x / (2 * k) + y / k = a / (3 * k) + 2 * b / (3 * k)) → 
  (y / b = 1 / 3) := 
sorry

end rugged_terrain_distance_ratio_l113_113401


namespace Alissa_presents_equal_9_l113_113455

def Ethan_presents : ℝ := 31.0
def difference : ℝ := 22.0
def Alissa_presents := Ethan_presents - difference

theorem Alissa_presents_equal_9 : Alissa_presents = 9.0 := 
by sorry

end Alissa_presents_equal_9_l113_113455


namespace fifth_power_ends_with_same_digit_l113_113011

theorem fifth_power_ends_with_same_digit (a : ℕ) : a^5 % 10 = a % 10 :=
by sorry

end fifth_power_ends_with_same_digit_l113_113011


namespace greatest_integer_less_than_or_equal_to_l113_113599

theorem greatest_integer_less_than_or_equal_to (x : ℝ) (h : x = 2 + Real.sqrt 3) : 
  ⌊x^3⌋ = 51 :=
by
  have h' : x ^ 3 = (2 + Real.sqrt 3) ^ 3 := by rw [h]
  sorry

end greatest_integer_less_than_or_equal_to_l113_113599


namespace base_area_of_cuboid_eq_seven_l113_113065

-- Definitions of the conditions
def volume_of_cuboid : ℝ := 28 -- Volume is 28 cm³
def height_of_cuboid : ℝ := 4  -- Height is 4 cm

-- The theorem statement for the problem
theorem base_area_of_cuboid_eq_seven
  (Volume : ℝ)
  (Height : ℝ)
  (h1 : Volume = 28)
  (h2 : Height = 4) :
  Volume / Height = 7 := by
  sorry

end base_area_of_cuboid_eq_seven_l113_113065


namespace pigeons_in_house_l113_113240

variable (x F c : ℝ)

theorem pigeons_in_house 
  (H1 : F = (x - 75) * 20 * c)
  (H2 : F = (x + 100) * 15 * c) :
  x = 600 := by
  sorry

end pigeons_in_house_l113_113240


namespace three_digit_numbers_divisible_by_13_count_l113_113777

noncomputable def smallest_3_digit_divisible_by_13 : ℕ := 117
noncomputable def largest_3_digit_divisible_by_13 : ℕ := 988
noncomputable def common_difference : ℕ := 13

theorem three_digit_numbers_divisible_by_13_count : 
  (largest_3_digit_divisible_by_13 - smallest_3_digit_divisible_by_13) / common_difference + 1 = 68 := 
by
  sorry

end three_digit_numbers_divisible_by_13_count_l113_113777


namespace ratio_of_cream_l113_113917

def initial_coffee := 18
def cup_capacity := 22
def Emily_drank := 3
def Emily_added_cream := 4
def Ethan_added_cream := 4
def Ethan_drank := 3

noncomputable def cream_in_Emily := Emily_added_cream

noncomputable def cream_remaining_in_Ethan :=
  Ethan_added_cream - (Ethan_added_cream * Ethan_drank / (initial_coffee + Ethan_added_cream))

noncomputable def resulting_ratio := cream_in_Emily / cream_remaining_in_Ethan

theorem ratio_of_cream :
  resulting_ratio = 200 / 173 :=
by
  sorry

end ratio_of_cream_l113_113917


namespace negation_of_universal_statement_l113_113269

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, x^2 ≠ x) ↔ ∃ x : ℝ, x^2 = x :=
by sorry

end negation_of_universal_statement_l113_113269


namespace a3_pm_2b3_not_div_by_37_l113_113464

theorem a3_pm_2b3_not_div_by_37 {a b : ℤ} (ha : ¬ (37 ∣ a)) (hb : ¬ (37 ∣ b)) :
  ¬ (37 ∣ (a^3 + 2 * b^3)) ∧ ¬ (37 ∣ (a^3 - 2 * b^3)) :=
  sorry

end a3_pm_2b3_not_div_by_37_l113_113464


namespace max_value_of_linear_combination_l113_113095

theorem max_value_of_linear_combination (x y : ℝ) (h : x^2 - 3 * x + 4 * y = 7) : 
  3 * x + 4 * y ≤ 16 :=
sorry

end max_value_of_linear_combination_l113_113095


namespace total_weight_AlF3_10_moles_l113_113976

noncomputable def molecular_weight_AlF3 (atomic_weight_Al: ℝ) (atomic_weight_F: ℝ) : ℝ :=
  atomic_weight_Al + 3 * atomic_weight_F

theorem total_weight_AlF3_10_moles :
  let atomic_weight_Al := 26.98
  let atomic_weight_F := 19.00
  let num_moles := 10
  molecular_weight_AlF3 atomic_weight_Al atomic_weight_F * num_moles = 839.8 :=
by
  sorry

end total_weight_AlF3_10_moles_l113_113976


namespace interest_rate_per_annum_l113_113593

theorem interest_rate_per_annum (P A : ℝ) (T : ℝ)
  (principal_eq : P = 973.913043478261)
  (amount_eq : A = 1120)
  (time_eq : T = 3):
  (A - P) / (T * P) * 100 = 5 := 
by 
  sorry

end interest_rate_per_annum_l113_113593


namespace original_cost_price_l113_113222

theorem original_cost_price (P : ℝ) 
  (h1 : P - 0.07 * P = 0.93 * P)
  (h2 : 0.93 * P + 0.02 * 0.93 * P = 0.9486 * P)
  (h3 : 0.9486 * P * 1.05 = 0.99603 * P)
  (h4 : 0.93 * P * 0.95 = 0.8835 * P)
  (h5 : 0.8835 * P + 0.02 * 0.8835 * P = 0.90117 * P)
  (h6 : 0.99603 * P - 5 = (0.90117 * P) * 1.10)
: P = 5 / 0.004743 :=
by
  sorry

end original_cost_price_l113_113222


namespace range_of_a_l113_113506

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -2 < x ∧ x < -1 → (a + x) * (1 + x) < 0) → a > 2 :=
by
  sorry

end range_of_a_l113_113506


namespace interval_length_l113_113844

theorem interval_length (c d : ℝ) (h : ∃ x : ℝ, c ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ d)
  (length : (d - 4) / 3 - (c - 4) / 3 = 15) : d - c = 45 :=
by
  sorry

end interval_length_l113_113844


namespace election_votes_l113_113795

theorem election_votes (V : ℝ) (ha : 0.45 * V = 4860)
                       (hb : 0.30 * V = 3240)
                       (hc : 0.20 * V = 2160)
                       (hd : 0.05 * V = 540)
                       (hmaj : (0.45 - 0.30) * V = 1620) :
                       V = 10800 :=
by
  sorry

end election_votes_l113_113795


namespace division_problem_l113_113494

theorem division_problem : (5 * 8) / 10 = 4 := by
  sorry

end division_problem_l113_113494


namespace find_other_number_l113_113124

theorem find_other_number (x y : ℕ) (h_gcd : Nat.gcd x y = 22) (h_lcm : Nat.lcm x y = 5940) (h_x : x = 220) :
  y = 594 :=
sorry

end find_other_number_l113_113124


namespace optimal_route_l113_113915

-- Define the probabilities of no traffic jam on each road segment.
def P_AC : ℚ := 9 / 10
def P_CD : ℚ := 14 / 15
def P_DB : ℚ := 5 / 6
def P_CF : ℚ := 9 / 10
def P_FB : ℚ := 15 / 16
def P_AE : ℚ := 9 / 10
def P_EF : ℚ := 9 / 10
def P_FB2 : ℚ := 19 / 20  -- Alias for repeated probability

-- Define the probability of encountering a traffic jam on a route
def prob_traffic_jam (p_no_jam : ℚ) : ℚ := 1 - p_no_jam

-- Define the probabilities of encountering a traffic jam along each route.
def P_ACDB_jam : ℚ := prob_traffic_jam (P_AC * P_CD * P_DB)
def P_ACFB_jam : ℚ := prob_traffic_jam (P_AC * P_CF * P_FB)
def P_AEFB_jam : ℚ := prob_traffic_jam (P_AE * P_EF * P_FB2)

-- State the theorem to prove the optimal route
theorem optimal_route : P_ACDB_jam < P_ACFB_jam ∧ P_ACDB_jam < P_AEFB_jam :=
by { sorry }

end optimal_route_l113_113915


namespace need_to_work_24_hours_per_week_l113_113948

-- Definitions
def original_hours_per_week := 20
def total_weeks := 12
def target_income := 3000

def missed_weeks := 2
def remaining_weeks := total_weeks - missed_weeks

-- Calculation
def new_hours_per_week := (original_hours_per_week * total_weeks) / remaining_weeks

-- Statement of the theorem
theorem need_to_work_24_hours_per_week : new_hours_per_week = 24 := 
by 
  -- Adding sorry to skip the proof, focusing on the statement.
  sorry

end need_to_work_24_hours_per_week_l113_113948


namespace solve_equation_l113_113729

theorem solve_equation (x : ℝ) (h : x^2 - x + 1 ≠ 0) :
  (x^2 + x + 1 = 1 / (x^2 - x + 1)) ↔ x = 1 ∨ x = -1 :=
by sorry

end solve_equation_l113_113729


namespace remainder_when_four_times_n_minus_nine_divided_by_7_l113_113084

theorem remainder_when_four_times_n_minus_nine_divided_by_7 (n : ℤ) (h : n % 7 = 3) : (4 * n - 9) % 7 = 3 := by
  sorry

end remainder_when_four_times_n_minus_nine_divided_by_7_l113_113084


namespace continuous_stripe_probability_l113_113257

noncomputable def probability_continuous_stripe_encircle_cube : ℚ :=
  let total_combinations : ℕ := 2^6
  let favor_combinations : ℕ := 3 * 4 -- 3 pairs of parallel faces, with 4 valid combinations each
  favor_combinations / total_combinations

theorem continuous_stripe_probability :
  probability_continuous_stripe_encircle_cube = 3 / 16 :=
by
  sorry

end continuous_stripe_probability_l113_113257


namespace coordinate_equation_solution_l113_113486

theorem coordinate_equation_solution (x y : ℝ) :
  2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 →
  (y = -x - 2) ∨ (y = -2 * x + 1) :=
by
  sorry

end coordinate_equation_solution_l113_113486


namespace least_number_remainder_5_l113_113260

theorem least_number_remainder_5 (n : ℕ) : 
  n % 12 = 5 ∧ n % 15 = 5 ∧ n % 20 = 5 ∧ n % 54 = 5 → n = 545 := 
  by
  sorry

end least_number_remainder_5_l113_113260


namespace digit_is_4_l113_113064

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

theorem digit_is_4 (d : ℕ) (hd0 : is_even d) (hd1 : is_divisible_by_3 (14 + d)) : d = 4 :=
  sorry

end digit_is_4_l113_113064


namespace find_second_term_geometric_sequence_l113_113613

noncomputable def second_term_geometric_sequence (a r : ℝ) : ℝ :=
  a * r

theorem find_second_term_geometric_sequence:
  ∀ (a r : ℝ),
    a * r^2 = 12 →
    a * r^3 = 18 →
    second_term_geometric_sequence a r = 8 :=
by
  intros a r h1 h2
  sorry

end find_second_term_geometric_sequence_l113_113613


namespace distance_parallel_lines_distance_point_line_l113_113715

def line1 (x y : ℝ) : Prop := 2 * x + y - 1 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y + 1 = 0
def point : ℝ × ℝ := (0, 2)

noncomputable def distance_between_lines (A B C1 C2 : ℝ) : ℝ :=
  |C2 - C1| / Real.sqrt (A^2 + B^2)

noncomputable def distance_point_to_line (A B C x0 y0 : ℝ) : ℝ :=
  |A * x0 + B * y0 + C| / Real.sqrt (A^2 + B^2)

theorem distance_parallel_lines : distance_between_lines 2 1 (-1) 1 = (2 * Real.sqrt 5) / 5 := by
  sorry

theorem distance_point_line : distance_point_to_line 2 1 (-1) 0 2 = (Real.sqrt 5) / 5 := by
  sorry

end distance_parallel_lines_distance_point_line_l113_113715


namespace ratio_a_to_c_l113_113947

theorem ratio_a_to_c (a b c : ℕ) (h1 : a / b = 5 / 3) (h2 : b / c = 1 / 5) : a / c = 1 / 3 :=
sorry

end ratio_a_to_c_l113_113947


namespace root_of_equation_l113_113303

theorem root_of_equation (x : ℝ) : 
  169 * (157 - 77 * x)^2 + 100 * (201 - 100 * x)^2 = 26 * (77 * x - 157) * (1000 * x - 2010) ↔ x = 31 := 
by 
  sorry

end root_of_equation_l113_113303


namespace student_average_always_less_l113_113259

theorem student_average_always_less (w x y z: ℝ) (hwx: w < x) (hxy: x < y) (hyz: y < z) :
  let A' := (w + x + y + z) / 4
  let B' := (2 * w + 2 * x + y + z) / 6
  B' < A' :=
by
  intro A' B'
  sorry

end student_average_always_less_l113_113259


namespace baseball_team_groups_l113_113488

theorem baseball_team_groups
  (new_players : ℕ) 
  (returning_players : ℕ)
  (players_per_group : ℕ)
  (total_players : ℕ := new_players + returning_players) :
  new_players = 48 → 
  returning_players = 6 → 
  players_per_group = 6 → 
  total_players / players_per_group = 9 :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  sorry

end baseball_team_groups_l113_113488


namespace town_council_original_plan_count_l113_113712

theorem town_council_original_plan_count (planned_trees current_trees : ℕ) (leaves_per_tree total_leaves : ℕ)
  (h1 : leaves_per_tree = 100)
  (h2 : total_leaves = 1400)
  (h3 : current_trees = total_leaves / leaves_per_tree)
  (h4 : current_trees = 2 * planned_trees) : 
  planned_trees = 7 :=
by
  sorry

end town_council_original_plan_count_l113_113712


namespace odd_consecutive_nums_divisibility_l113_113291

theorem odd_consecutive_nums_divisibility (a b : ℕ) (h_consecutive : b = a + 2) (h_odd_a : a % 2 = 1) (h_odd_b : b % 2 = 1) : (a^b + b^a) % (a + b) = 0 := by
  sorry

end odd_consecutive_nums_divisibility_l113_113291


namespace first_half_day_wednesday_l113_113053

theorem first_half_day_wednesday (h1 : ¬(1 : ℕ) = (4 % 7) ∨ 1 % 7 != 0)
  (h2 : ∀ d : ℕ, d ≤ 31 → d % 7 = ((d + 3) % 7)) : 
  ∃ d : ℕ, d = 25 ∧ ∃ W : ℕ → Prop, W d := sorry

end first_half_day_wednesday_l113_113053


namespace sum_of_B_coordinates_l113_113737

theorem sum_of_B_coordinates 
  (x y : ℝ) 
  (A : ℝ × ℝ) 
  (M : ℝ × ℝ)
  (midpoint_x : (A.1 + x) / 2 = M.1) 
  (midpoint_y : (A.2 + y) / 2 = M.2) 
  (A_conds : A = (7, -1))
  (M_conds : M = (4, 3)) :
  x + y = 8 :=
by 
  sorry

end sum_of_B_coordinates_l113_113737


namespace functional_solutions_l113_113982

noncomputable def functional_equation (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, x * f y + y * f x = (x + y) * (f x) * (f y)

theorem functional_solutions (f : ℝ → ℝ) (h : functional_equation f) : 
  (∀ x : ℝ, f x = 0) ∨ (∃ (a : ℝ), ∀ x : ℝ, (x ≠ 0 → f x = 1) ∧ (x = 0 → f x = a)) :=
  sorry

end functional_solutions_l113_113982


namespace find_intersection_point_l113_113735

/-- Definition of the parabola -/
def parabola (y : ℝ) : ℝ := -3 * y ^ 2 - 4 * y + 7

/-- Condition for intersection at exactly one point -/
def discriminant (m : ℝ) : ℝ := 4 ^ 2 - 4 * 3 * (m - 7)

/-- Main theorem stating the proof problem -/
theorem find_intersection_point (m : ℝ) :
  (discriminant m = 0) → m = 25 / 3 :=
by
  sorry

end find_intersection_point_l113_113735


namespace max_free_squares_l113_113310

theorem max_free_squares (n : ℕ) :
  ∀ (initial_positions : ℕ), 
    (∀ i j, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → initial_positions = 2) →
    (∀ (i j : ℕ) (move1 move2 : ℕ × ℕ),
       1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n →
       move1 = (i + 1, j) ∨ move1 = (i - 1, j) ∨ move1 = (i, j + 1) ∨ move1 = (i, j - 1) →
       move2 = (i + 1, j) ∨ move2 = (i - 1, j) ∨ move2 = (i, j + 1) ∨ move2 = (i, j - 1) →
       move1 ≠ move2) →
    ∃ free_squares : ℕ, free_squares = n^2 :=
by
  sorry

end max_free_squares_l113_113310


namespace C_investment_l113_113080

def A_investment_eq : Prop :=
  ∀ (C T : ℝ), (C * T) / 36 = (1 / 6 : ℝ) * C * (1 / 6 : ℝ) * T

def B_investment_eq : Prop :=
  ∀ (C T : ℝ), (C * T) / 9 = (1 / 3 : ℝ) * C * (1 / 3 : ℝ) * T

def C_investment_eq (x : ℝ) : Prop :=
  ∀ (C T : ℝ), x * C * T = (x : ℝ) * C * T

theorem C_investment (x : ℝ) :
  (∀ (C T : ℝ), A_investment_eq) ∧
  (∀ (C T : ℝ), B_investment_eq) ∧
  (∀ (C T : ℝ), C_investment_eq x) ∧
  (∀ (C T : ℝ), 100 / 2300 = (C * T / 36) / ((C * T / 36) + (C * T / 9) + (x * C * T))) →
  x = 1 / 2 :=
by
  intros
  sorry

end C_investment_l113_113080


namespace sunflower_height_A_l113_113279

-- Define the height of sunflowers from Packet B
def height_B : ℝ := 160

-- Define that Packet A sunflowers are 20% taller than Packet B sunflowers
def height_A : ℝ := 1.2 * height_B

-- State the theorem to show that height_A equals 192 inches
theorem sunflower_height_A : height_A = 192 := by
  sorry

end sunflower_height_A_l113_113279


namespace arrangement_count_l113_113196

def no_adjacent_students_arrangements (teachers students : ℕ) : ℕ :=
  if teachers = 3 ∧ students = 3 then 144 else 0

theorem arrangement_count :
  no_adjacent_students_arrangements 3 3 = 144 :=
by
  sorry

end arrangement_count_l113_113196


namespace total_birds_in_store_l113_113472

def num_bird_cages := 4
def parrots_per_cage := 8
def parakeets_per_cage := 2
def birds_per_cage := parrots_per_cage + parakeets_per_cage
def total_birds := birds_per_cage * num_bird_cages

theorem total_birds_in_store : total_birds = 40 :=
  by sorry

end total_birds_in_store_l113_113472


namespace line_passes_through_quadrants_l113_113047

theorem line_passes_through_quadrants (a b c : ℝ) (hab : a * b < 0) (hbc : b * c < 0) : 
  ∀ (x y : ℝ), (a * x + b * y + c = 0) → (x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0) ∨ (x < 0 ∧ y < 0) :=
by {
  sorry
}

end line_passes_through_quadrants_l113_113047


namespace equal_chessboard_numbers_l113_113889

theorem equal_chessboard_numbers (n : ℕ) (board : ℕ → ℕ → ℕ) 
  (mean_property : ∀ (x y : ℕ), board x y = (board (x-1) y + board (x+1) y + board x (y-1) + board x (y+1)) / 4) : 
  ∀ (x y : ℕ), board x y = board 0 0 :=
by
  -- Proof not required
  sorry

end equal_chessboard_numbers_l113_113889


namespace inf_div_p_n2n_plus_one_n_div_3_n2n_plus_one_l113_113776

theorem inf_div_p_n2n_plus_one (p : ℕ) (hp : Nat.Prime p) (h_odd : p % 2 = 1) :
  ∃ᶠ n in at_top, p ∣ (n * 2^n + 1) :=
sorry

theorem n_div_3_n2n_plus_one :
  (∃ k : ℕ, ∀ n, n = 6 * k + 1 ∨ n = 6 * k + 2 → 3 ∣ (n * 2^n + 1)) :=
sorry

end inf_div_p_n2n_plus_one_n_div_3_n2n_plus_one_l113_113776


namespace product_of_numbers_is_86_l113_113907

-- Definitions of the two conditions
def sum_eq_24 (x y : ℝ) : Prop := x + y = 24
def sum_of_squares_eq_404 (x y : ℝ) : Prop := x^2 + y^2 = 404

-- The theorem to prove the product of the two numbers
theorem product_of_numbers_is_86 (x y : ℝ) (h1 : sum_eq_24 x y) (h2 : sum_of_squares_eq_404 x y) : x * y = 86 :=
  sorry

end product_of_numbers_is_86_l113_113907


namespace lateral_edges_in_same_plane_edges_in_planes_for_all_vertices_l113_113713

-- Define a cube with edge length a
structure Cube :=
  (a : ℝ) -- Edge length of the cube

-- Define a pyramid with a given height
structure Pyramid :=
  (h : ℝ) -- Height of the pyramid

-- The main theorem statement for part 4A
theorem lateral_edges_in_same_plane (c : Cube) (p : Pyramid) : p.h = c.a ↔ (∃ O1 O2 O3 : ℝ × ℝ × ℝ,
  O1 = (c.a / 2, c.a / 2, -p.h) ∧
  O2 = (c.a / 2, -p.h, c.a / 2) ∧
  O3 = (-p.h, c.a / 2, c.a / 2)) := sorry

-- The main theorem statement for part 4B
theorem edges_in_planes_for_all_vertices (c : Cube) (p : Pyramid) : p.h = c.a ↔ ∀ (v : ℝ × ℝ × ℝ), -- Iterate over cube vertices
  (∃ O1 O2 O3 : ℝ × ℝ × ℝ,
    O1 = (c.a / 2, c.a / 2, -p.h) ∧
    O2 = (c.a / 2, -p.h, c.a / 2) ∧
    O3 = (-p.h, c.a / 2, c.a / 2)) := sorry

end lateral_edges_in_same_plane_edges_in_planes_for_all_vertices_l113_113713


namespace cost_price_books_l113_113512

def cost_of_type_A (cost_A cost_B : ℝ) : Prop :=
  cost_A = cost_B + 15

def quantity_equal (cost_A cost_B : ℝ) : Prop :=
  675 / cost_A = 450 / cost_B

theorem cost_price_books (cost_A cost_B : ℝ) (h1 : cost_of_type_A cost_A cost_B) (h2 : quantity_equal cost_A cost_B) : 
  cost_A = 45 ∧ cost_B = 30 :=
by
  -- Proof omitted
  sorry

end cost_price_books_l113_113512


namespace volume_of_pool_l113_113710

variable (P T V C : ℝ)

/-- 
The volume of the pool is given as P * T divided by percentage C.
The question is to prove that the volume V of the pool equals 90000 cubic feet given:
  P: The hose can remove 60 cubic feet per minute.
  T: It takes 1200 minutes to drain the pool.
  C: The pool was at 80% capacity when draining started.
-/
theorem volume_of_pool (h1 : P = 60) 
                       (h2 : T = 1200) 
                       (h3 : C = 0.80) 
                       (h4 : P * T / C = V) :
  V = 90000 := 
sorry

end volume_of_pool_l113_113710


namespace find_remainder_l113_113623

noncomputable def q (x : ℝ) : ℝ := (x^2010 + x^2009 + x^2008 + x + 1)
noncomputable def s (x : ℝ) := (q x) % (x^3 + 2*x^2 + 3*x + 1)

theorem find_remainder (x : ℝ) : (|s 2011| % 500) = 357 := by
    sorry

end find_remainder_l113_113623


namespace joe_paid_4_more_than_jenny_l113_113467

theorem joe_paid_4_more_than_jenny
  (total_plain_pizza_cost : ℕ := 12) 
  (total_slices : ℕ := 12)
  (additional_cost_per_mushroom_slice : ℕ := 1) -- 0.50 dollars represented in integer (value in cents or minimal currency unit)
  (mushroom_slices : ℕ := 4) 
  (plain_slices := total_slices - mushroom_slices) -- Calculate plain slices.
  (total_additional_cost := mushroom_slices * additional_cost_per_mushroom_slice)
  (total_pizza_cost := total_plain_pizza_cost + total_additional_cost)
  (plain_slice_cost := total_plain_pizza_cost / total_slices)
  (mushroom_slice_cost := plain_slice_cost + additional_cost_per_mushroom_slice) 
  (joe_mushroom_slices := mushroom_slices) 
  (joe_plain_slices := 3) 
  (jenny_plain_slices := plain_slices - joe_plain_slices) 
  (joe_paid := (joe_mushroom_slices * mushroom_slice_cost) + (joe_plain_slices * plain_slice_cost))
  (jenny_paid := jenny_plain_slices * plain_slice_cost) : 
  joe_paid - jenny_paid = 4 := 
by {
  -- Here, we define the steps we used to calculate the cost.
  sorry -- Proof skipped as per instructions.
}

end joe_paid_4_more_than_jenny_l113_113467


namespace a_equals_2t_squared_l113_113007

theorem a_equals_2t_squared {a b : ℕ} (h1 : 0 < a) (h2 : 0 < b) (h3 : a^3 + 4 * a = b^2) :
  ∃ t : ℕ, 0 < t ∧ a = 2 * t^2 :=
sorry

end a_equals_2t_squared_l113_113007


namespace recommendation_plans_count_l113_113390

theorem recommendation_plans_count :
  let total_students := 7
  let sports_talents := 2
  let artistic_talents := 2
  let other_talents := 3
  let recommend_count := 4
  let condition_sports := recommend_count >= 1
  let condition_artistic := recommend_count >= 1
  (condition_sports ∧ condition_artistic) → 
  ∃ (n : ℕ), n = 25 := sorry

end recommendation_plans_count_l113_113390


namespace polynomial_degree_is_14_l113_113406

noncomputable def polynomial_degree (a b c d e f g h : ℝ) : ℕ :=
  if a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0 then 14 else 0

theorem polynomial_degree_is_14 (a b c d e f g h : ℝ) (h_neq0 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ f ≠ 0 ∧ g ≠ 0 ∧ h ≠ 0) :
  polynomial_degree a b c d e f g h = 14 :=
by sorry

end polynomial_degree_is_14_l113_113406


namespace length_of_AD_l113_113468

theorem length_of_AD (AB BC CD DE : ℝ) (right_angle_B right_angle_C : Prop) :
  AB = 6 → BC = 7 → CD = 25 → DE = 15 → AD = Real.sqrt 274 :=
by
  intros
  sorry

end length_of_AD_l113_113468


namespace area_of_shaded_rectangle_l113_113193

-- Definition of side length of the squares
def side_length : ℕ := 12

-- Definition of the dimensions of the overlapped rectangle
def rectangle_length : ℕ := 20
def rectangle_width : ℕ := side_length

-- Theorem stating the area of the shaded rectangle PBCS
theorem area_of_shaded_rectangle
  (squares_identical : ∀ (a b c d p q r s : ℕ),
    a = side_length → b = side_length →
    p = side_length → q = side_length →
    rectangle_width * (rectangle_length - side_length) = 48) :
  rectangle_width * (rectangle_length - side_length) = 48 :=
by sorry -- Proof omitted

end area_of_shaded_rectangle_l113_113193


namespace license_plate_increase_l113_113149

theorem license_plate_increase :
  let old_license_plates := 26^2 * 10^3
  let new_license_plates := 26^2 * 10^4
  new_license_plates / old_license_plates = 10 :=
by
  sorry

end license_plate_increase_l113_113149


namespace perfect_even_multiples_of_3_under_3000_l113_113381

theorem perfect_even_multiples_of_3_under_3000 :
  ∃ n : ℕ, n = 9 ∧ ∀ (k : ℕ), (36 * k^2 < 3000) → (36 * k^2) % 2 = 0 ∧ (36 * k^2) % 3 = 0 ∧ ∃ m : ℕ, m^2 = 36 * k^2 :=
by
  sorry

end perfect_even_multiples_of_3_under_3000_l113_113381


namespace stapler_problem_l113_113724

noncomputable def staplesLeft (initial_staples : ℕ) (dozens : ℕ) (staples_per_report : ℝ) : ℝ :=
  initial_staples - (dozens * 12) * staples_per_report

theorem stapler_problem : staplesLeft 200 7 0.75 = 137 := 
by
  sorry

end stapler_problem_l113_113724


namespace at_least_six_stones_empty_l113_113890

def frogs_on_stones (a : Fin 23 → Fin 23) (k : Nat) : Fin 22 → Fin 23 :=
  fun i => (a i + i.1 * k) % 23

theorem at_least_six_stones_empty 
  (a : Fin 22 → Fin 23) :
  ∃ k : Nat, ∀ (s : Fin 23), ∃ (j : Fin 22), frogs_on_stones (fun i => a i) k j ≠ s ↔ ∃! t : Fin 23, ∃! j, (frogs_on_stones (fun i => a i) k j) = t := 
  sorry

end at_least_six_stones_empty_l113_113890


namespace boat_sinking_weight_range_l113_113910

theorem boat_sinking_weight_range
  (L_min L_max : ℝ)
  (B_min B_max : ℝ)
  (D_min D_max : ℝ)
  (sink_rate : ℝ)
  (down_min down_max : ℝ)
  (min_weight max_weight : ℝ)
  (condition1 : 3 ≤ L_min ∧ L_max ≤ 5)
  (condition2 : 2 ≤ B_min ∧ B_max ≤ 3)
  (condition3 : 1 ≤ D_min ∧ D_max ≤ 2)
  (condition4 : sink_rate = 0.01)
  (condition5 : 0.03 ≤ down_min ∧ down_max ≤ 0.06)
  (condition6 : ∀ D, D_min ≤ D ∧ D ≤ D_max → (D - down_max) ≥ 0.5)
  (condition7 : min_weight = down_min * (10 / 0.01))
  (condition8 : max_weight = down_max * (10 / 0.01)) :
  min_weight = 30 ∧ max_weight = 60 := 
sorry

end boat_sinking_weight_range_l113_113910


namespace correct_average_of_10_numbers_l113_113520

theorem correct_average_of_10_numbers
  (incorrect_avg : ℕ)
  (n : ℕ)
  (incorrect_read : ℕ)
  (correct_read : ℕ)
  (incorrect_total_sum : ℕ) :
  incorrect_avg = 19 →
  n = 10 →
  incorrect_read = 26 →
  correct_read = 76 →
  incorrect_total_sum = incorrect_avg * n →
  (correct_total_sum : ℕ) = incorrect_total_sum - incorrect_read + correct_read →
  (correct_avg : ℕ) = correct_total_sum / n →
  correct_avg = 24 :=
by
  intros
  sorry

end correct_average_of_10_numbers_l113_113520


namespace second_player_wins_l113_113938

-- Defining the chess board and initial positions of the rooks
inductive Square : Type
| a1 | a2 | a3 | a4 | a5 | a6 | a7 | a8
| b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8
| c1 | c2 | c3 | c4 | c5 | c6 | c7 | c8
| d1 | d2 | d3 | d4 | d5 | d6 | d7 | d8
| e1 | e2 | e3 | e4 | e5 | e6 | e7 | e8
| f1 | f2 | f3 | f4 | f5 | f6 | f7 | f8
| g1 | g2 | g3 | g4 | g5 | g6 | g7 | g8
| h1 | h2 | h3 | h4 | h5 | h6 | h7 | h8
deriving DecidableEq

-- Define the initial positions of the rooks
def initial_white_rook_position : Square := Square.b2
def initial_black_rook_position : Square := Square.c4

-- Define the rules of movement: a rook can move horizontally or vertically unless blocked
def rook_can_move (start finish : Square) : Prop :=
  -- Only horizontal or vertical moves allowed
  sorry

-- Define conditions for a square being attacked by a rook at a given position
def is_attacked_by_rook (position target : Square) : Prop :=
  sorry

-- Define the condition for a player to be in a winning position if no moves are illegal
def player_can_win (white_position black_position : Square) : Prop :=
  sorry

-- The main theorem: Second player (black rook) can ensure a win
theorem second_player_wins : player_can_win initial_white_rook_position initial_black_rook_position :=
  sorry

end second_player_wins_l113_113938


namespace ranking_of_scores_l113_113708

-- Let the scores of Ann, Bill, Carol, and Dick be A, B, C, and D respectively.

variables (A B C D : ℝ)

-- Conditions
axiom cond1 : B + D = A + C
axiom cond2 : C + B > D + A
axiom cond3 : C > A + B

-- Statement of the problem
theorem ranking_of_scores : C > D ∧ D > B ∧ B > A :=
by
  -- Placeholder for proof (proof steps aren't required)
  sorry

end ranking_of_scores_l113_113708


namespace smallest_possible_k_l113_113495

def infinite_increasing_seq (a : ℕ → ℕ) : Prop :=
∀ n, a n < a (n + 1)

def divisible_by_1005_or_1006 (a : ℕ) : Prop :=
a % 1005 = 0 ∨ a % 1006 = 0

def not_divisible_by_97 (a : ℕ) : Prop :=
a % 97 ≠ 0

def diff_less_than_k (a : ℕ → ℕ) (k : ℕ) : Prop :=
∀ n, (a (n + 1) - a n) ≤ k

theorem smallest_possible_k :
  ∀ (a : ℕ → ℕ), infinite_increasing_seq a →
  (∀ n, divisible_by_1005_or_1006 (a n)) →
  (∀ n, not_divisible_by_97 (a n)) →
  (∃ k, diff_less_than_k a k) →
  (∃ k, k = 2010 ∧ diff_less_than_k a k) :=
by
  sorry

end smallest_possible_k_l113_113495


namespace tom_sawyer_bible_l113_113316

def blue_tickets_needed (yellow: ℕ) (red: ℕ) (blue: ℕ): ℕ := 
  10 * 10 * 10 * yellow + 10 * 10 * red + blue

theorem tom_sawyer_bible (y r b : ℕ) (hc : y = 8 ∧ r = 3 ∧ b = 7):
  blue_tickets_needed 10 0 0 - blue_tickets_needed y r b = 163 :=
by 
  sorry

end tom_sawyer_bible_l113_113316


namespace problem_l113_113441

theorem problem (p q : Prop) (m : ℝ):
  (p = (m > 1)) →
  (q = (-2 ≤ m ∧ m ≤ 2)) →
  (¬q = (m < -2 ∨ m > 2)) →
  (¬(p ∧ q)) →
  (p ∨ q) →
  (¬q) →
  m > 2 :=
by
  sorry

end problem_l113_113441


namespace correct_speed_l113_113264

noncomputable def distance (t : ℝ) := 50 * (t + 5 / 60)
noncomputable def distance2 (t : ℝ) := 70 * (t - 5 / 60)

theorem correct_speed : 
  ∃ r : ℝ, 
    (∀ t : ℝ, distance t = distance2 t → r = 55) := 
by
  sorry

end correct_speed_l113_113264


namespace slices_per_pie_l113_113809

variable (S : ℕ) -- Let S be the number of slices per pie

theorem slices_per_pie (h1 : 5 * S * 9 = 180) : S = 4 := by
  sorry

end slices_per_pie_l113_113809


namespace coffee_tea_soda_l113_113431

theorem coffee_tea_soda (Pcoffee Ptea Psoda Pboth_no_soda : ℝ)
  (H1 : 0.9 = Pcoffee)
  (H2 : 0.8 = Ptea)
  (H3 : 0.7 = Psoda) :
  0.0 = Pboth_no_soda :=
  sorry

end coffee_tea_soda_l113_113431


namespace percentage_error_l113_113329

theorem percentage_error (x : ℝ) (hx : x ≠ 0) :
  let correct_result := 10 * x
  let incorrect_result := x / 10
  let error := correct_result - incorrect_result
  let percentage_error := (error / correct_result) * 100
  percentage_error = 99 :=
by
  sorry

end percentage_error_l113_113329


namespace round_robin_games_l113_113949

theorem round_robin_games (x : ℕ) (h : 45 = (1 / 2) * x * (x - 1)) : (1 / 2) * x * (x - 1) = 45 :=
sorry

end round_robin_games_l113_113949


namespace tetrahedron_cut_off_vertices_l113_113933

theorem tetrahedron_cut_off_vertices :
  ∀ (V E : ℕ) (cut_effect : ℕ → ℕ),
    -- Initial conditions
    V = 4 → E = 6 →
    -- Effect of each cut (cutting one vertex introduces 3 new edges)
    (∀ v, v ≤ V → cut_effect v = 3 * v) →
    -- Prove the number of edges in the new figure
    (E + cut_effect V) = 18 :=
by
  intros V E cut_effect hV hE hcut
  sorry

end tetrahedron_cut_off_vertices_l113_113933


namespace original_number_is_45_l113_113338

theorem original_number_is_45 (x : ℕ) (h : x - 30 = x / 3) : x = 45 :=
by {
  sorry
}

end original_number_is_45_l113_113338


namespace units_digit_of_result_is_7_l113_113115

theorem units_digit_of_result_is_7 (a b c : ℕ) (h : a = c + 3) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
  let original := 100 * a + 10 * b + c
  let reversed := 100 * c + 10 * b + a
  (original - reversed) % 10 = 7 :=
by
  sorry

end units_digit_of_result_is_7_l113_113115


namespace simplify_and_evaluate_expression_l113_113194

theorem simplify_and_evaluate_expression (x y : ℝ) (h1 : x = 1/2) (h2 : y = -2) :
  ((x + 2 * y) ^ 2 - (x + y) * (x - y)) / (2 * y) = -4 := by
  sorry

end simplify_and_evaluate_expression_l113_113194


namespace f_bounded_by_inverse_l113_113202

theorem f_bounded_by_inverse (f : ℕ → ℝ) (h_pos : ∀ n, 0 < f n) (h_rec : ∀ n, (f n)^2 ≤ f n - f (n + 1)) :
  ∀ n, f n < 1 / (n + 1) :=
by
  sorry

end f_bounded_by_inverse_l113_113202


namespace arithmetic_sequence_twenty_fourth_term_l113_113215

-- Given definitions (conditions)
def third_term (a d : ℚ) : ℚ := a + 2 * d
def tenth_term (a d : ℚ) : ℚ := a + 9 * d
def twenty_fourth_term (a d : ℚ) : ℚ := a + 23 * d

-- The main theorem to be proved
theorem arithmetic_sequence_twenty_fourth_term 
  (a d : ℚ) 
  (h1 : third_term a d = 7) 
  (h2 : tenth_term a d = 27) :
  twenty_fourth_term a d = 67 := by
  sorry

end arithmetic_sequence_twenty_fourth_term_l113_113215


namespace tan_inverse_least_positive_l113_113048

variables (a b x : ℝ)

-- Condition 1: tan(x) = a / (2*b)
def condition1 : Prop := Real.tan x = a / (2 * b)

-- Condition 2: tan(2*x) = 2*b / (a + 2*b)
def condition2 : Prop := Real.tan (2 * x) = (2 * b) / (a + 2 * b)

-- The theorem stating the least positive value of x is arctan(0)
theorem tan_inverse_least_positive (h1 : condition1 a b x) (h2 : condition2 a b x) : ∃ k : ℝ, Real.arctan k = 0 :=
by
  sorry

end tan_inverse_least_positive_l113_113048


namespace jacob_ate_five_pies_l113_113008

theorem jacob_ate_five_pies (weight_hot_dog weight_burger weight_pie noah_burgers mason_hotdogs_total_weight : ℕ)
    (H1 : weight_hot_dog = 2)
    (H2 : weight_burger = 5)
    (H3 : weight_pie = 10)
    (H4 : noah_burgers = 8)
    (H5 : mason_hotdogs_total_weight = 30)
    (H6 : ∀ x, 3 * x = (mason_hotdogs_total_weight / weight_hot_dog)) :
    (∃ y, y = (mason_hotdogs_total_weight / weight_hot_dog / 3) ∧ y = 5) :=
by
  sorry

end jacob_ate_five_pies_l113_113008


namespace gcd_7854_13843_l113_113654

theorem gcd_7854_13843 : Nat.gcd 7854 13843 = 1 := 
  sorry

end gcd_7854_13843_l113_113654


namespace numDifferentSignals_l113_113091

-- Number of indicator lights in a row
def numLights : Nat := 6

-- Number of lights that light up each time
def lightsLit : Nat := 3

-- Number of colors each light can show
def numColors : Nat := 3

-- Function to calculate number of different signals
noncomputable def calculateSignals (n m k : Nat) : Nat :=
  -- Number of possible arrangements of "adjacent, adjacent, separate" and "separate, adjacent, adjacent"
  let arrangements := 4 + 4
  -- Number of color combinations for the lit lights
  let colors := k * k * k
  arrangements * colors

-- Theorem stating the total number of different signals is 324
theorem numDifferentSignals : calculateSignals numLights lightsLit numColors = 324 := 
by
  sorry

end numDifferentSignals_l113_113091


namespace max_pens_l113_113578

theorem max_pens (total_money notebook_cost pen_cost num_notebooks : ℝ) (notebook_qty pen_qty : ℕ):
  total_money = 18 ∧ notebook_cost = 3.6 ∧ pen_cost = 3 ∧ num_notebooks = 2 →
  (pen_qty = 1 ∨ pen_qty = 2 ∨ pen_qty = 3) ↔ (2 * notebook_cost + pen_qty * pen_cost ≤ total_money) :=
by {
  sorry
}

end max_pens_l113_113578


namespace silverware_probability_l113_113666

-- Definitions based on the problem conditions
def total_silverware : ℕ := 8 + 10 + 7
def total_combinations : ℕ := Nat.choose total_silverware 4

def fork_combinations : ℕ := Nat.choose 8 2
def spoon_combinations : ℕ := Nat.choose 10 1
def knife_combinations : ℕ := Nat.choose 7 1

def favorable_combinations : ℕ := fork_combinations * spoon_combinations * knife_combinations
def specific_combination_probability : ℚ := favorable_combinations / total_combinations

-- The statement to prove the given probability
theorem silverware_probability :
  specific_combination_probability = 392 / 2530 :=
by
  sorry

end silverware_probability_l113_113666


namespace distance_between_intersections_l113_113879

open Function

def cube_vertices : List (ℝ × ℝ × ℝ) :=
  [(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0), (5, 0, 5), (5, 5, 0), (5, 5, 5)]

def intersecting_points : List (ℝ × ℝ × ℝ) :=
  [(0, 3, 0), (2, 0, 0), (2, 5, 5)]

noncomputable def plane_distance_between_points : ℝ :=
  let S := (11 / 3, 0, 5)
  let T := (0, 5, 4)
  Real.sqrt ((11 / 3 - 0)^2 + (0 - 5)^2 + (5 - 4)^2)

theorem distance_between_intersections : plane_distance_between_points = Real.sqrt (355 / 9) :=
  sorry

end distance_between_intersections_l113_113879


namespace triangle_parallel_vectors_l113_113639

noncomputable def collinear {V : Type*} [AddCommGroup V] [Module ℝ V]
  (P₁ P₂ P₃ : V) : Prop :=
∃ t : ℝ, P₃ = P₁ + t • (P₂ - P₁)

theorem triangle_parallel_vectors
  (A B C C₁ A₁ B₁ C₂ A₂ B₂ : ℝ × ℝ)
  (h1 : collinear A B C₁) (h2 : collinear B C A₁) (h3 : collinear C A B₁)
  (ratio1 : ∀ (AC1 CB : ℝ), AC1 / CB = 1) (ratio2 : ∀ (BA1 AC : ℝ), BA1 / AC = 1) (ratio3 : ∀ (CB B1A : ℝ), CB / B1A = 1)
  (h4 : collinear A₁ B₁ C₂) (h5 : collinear B₁ C₁ A₂) (h6 : collinear C₁ A₁ B₂)
  (n : ℝ)
  (ratio4 : ∀ (A1C2 C2B1 : ℝ), A1C2 / C2B1 = n) (ratio5 : ∀ (B1A2 A2C1 : ℝ), B1A2 / A2C1 = n) (ratio6 : ∀ (C1B2 B2A1 : ℝ), C1B2 / B2A1 = n) :
  collinear A C A₂ ∧ collinear C B C₂ ∧ collinear B A B₂ :=
sorry

end triangle_parallel_vectors_l113_113639


namespace fewer_seats_on_right_than_left_l113_113789

theorem fewer_seats_on_right_than_left : 
  ∀ (left_seats right_seats back_seat_capacity people_per_seat bus_capacity fewer_seats : ℕ),
    left_seats = 15 →
    back_seat_capacity = 9 →
    people_per_seat = 3 →
    bus_capacity = 90 →
    right_seats = (bus_capacity - (left_seats * people_per_seat + back_seat_capacity)) / people_per_seat →
    fewer_seats = left_seats - right_seats →
    fewer_seats = 3 :=
by
  intros left_seats right_seats back_seat_capacity people_per_seat bus_capacity fewer_seats
  sorry

end fewer_seats_on_right_than_left_l113_113789


namespace variance_proof_l113_113342

noncomputable def calculate_mean (scores : List ℝ) : ℝ :=
  (scores.sum / scores.length)

noncomputable def calculate_variance (scores : List ℝ) : ℝ :=
  let mean := calculate_mean scores
  (scores.map (λ x => (x - mean)^2)).sum / scores.length

def scores_A : List ℝ := [8, 6, 9, 5, 10, 7, 4, 7, 9, 5]
def scores_B : List ℝ := [7, 6, 5, 8, 6, 9, 6, 8, 8, 7]

noncomputable def variance_A : ℝ := calculate_variance scores_A
noncomputable def variance_B : ℝ := calculate_variance scores_B

theorem variance_proof :
  variance_A = 3.6 ∧ variance_B = 1.4 ∧ variance_B < variance_A :=
by
  -- proof steps - use sorry to skip the proof
  sorry

end variance_proof_l113_113342


namespace irrigation_tank_final_amount_l113_113137

theorem irrigation_tank_final_amount : 
  let initial_amount := 300.0
  let evaporation := 1.0
  let addition := 0.3
  let days := 45
  let daily_change := addition - evaporation
  let total_change := daily_change * days
  initial_amount + total_change = 268.5 := 
by {
  -- Proof goes here
  sorry
}

end irrigation_tank_final_amount_l113_113137


namespace trapezoids_not_necessarily_congruent_l113_113484

-- Define trapezoid structure
structure Trapezoid (α : Type) [LinearOrderedField α] :=
(base1 base2 side1 side2 diag1 diag2 : α) -- sides and diagonals
(angle1 angle2 angle3 angle4 : α)        -- internal angles

-- Conditions about given trapezoids
variables {α : Type} [LinearOrderedField α]
variables (T1 T2 : Trapezoid α)

-- The condition that corresponding angles of the trapezoids are equal
def equal_angles := 
  T1.angle1 = T2.angle1 ∧ T1.angle2 = T2.angle2 ∧ 
  T1.angle3 = T2.angle3 ∧ T1.angle4 = T2.angle4

-- The condition that diagonals of the trapezoids are equal
def equal_diagonals := 
  T1.diag1 = T2.diag1 ∧ T1.diag2 = T2.diag2

-- The statement to prove
theorem trapezoids_not_necessarily_congruent :
  equal_angles T1 T2 ∧ equal_diagonals T1 T2 → ¬ (T1 = T2) := by
  sorry

end trapezoids_not_necessarily_congruent_l113_113484


namespace max_xyz_l113_113155

theorem max_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 2) 
(h5 : x^2 + y^2 + z^2 = x * z + y * z + x * y) : xyz ≤ (8 / 27) :=
sorry

end max_xyz_l113_113155


namespace river_ratio_l113_113107

theorem river_ratio (total_length straight_length crooked_length : ℕ) 
  (h1 : total_length = 80) (h2 : straight_length = 20) 
  (h3 : crooked_length = total_length - straight_length) : 
  (straight_length / Nat.gcd straight_length crooked_length) = 1 ∧ (crooked_length / Nat.gcd straight_length crooked_length) = 3 := 
by
  sorry

end river_ratio_l113_113107


namespace missing_number_in_proportion_l113_113313

/-- Given the proportion 2 : 5 = x : 3.333333333333333, prove that the missing number x is 1.3333333333333332 -/
theorem missing_number_in_proportion : ∃ x, (2 / 5 = x / 3.333333333333333) ∧ x = 1.3333333333333332 :=
  sorry

end missing_number_in_proportion_l113_113313


namespace divisible_by_42_l113_113819

theorem divisible_by_42 (n : ℕ) : 42 ∣ (n^3 * (n^6 - 1)) :=
sorry

end divisible_by_42_l113_113819


namespace sqrt_fraction_subtraction_l113_113596

theorem sqrt_fraction_subtraction :
  (Real.sqrt (9 / 2) - Real.sqrt (2 / 9)) = (7 * Real.sqrt 2 / 6) :=
by sorry

end sqrt_fraction_subtraction_l113_113596


namespace int_solutions_l113_113333

theorem int_solutions (a b : ℤ) (h : a^2 + b = b^2022) : (a, b) = (0, 0) ∨ (a, b) = (0, 1) :=
by {
  sorry
}

end int_solutions_l113_113333


namespace proportion_solution_l113_113402

theorem proportion_solution (x : ℝ) (h : 0.75 / x = 5 / 8) : x = 1.2 := 
by 
suffices h₀ : x = 6 / 5 by sorry
suffices h₁ : 6 / 5 = 1.2 by sorry
-- Proof steps go here
sorry

end proportion_solution_l113_113402


namespace range_of_abs_function_l113_113073

theorem range_of_abs_function:
  (∀ y, ∃ x : ℝ, y = |x + 3| - |x - 5|) → ∀ y, y ≤ 8 :=
by
  sorry

end range_of_abs_function_l113_113073


namespace value_of_x_l113_113896

theorem value_of_x (x y z w : ℕ) (h1 : x = y + 7) (h2 : y = z + 12) (h3 : z = w + 25) (h4 : w = 90) : x = 134 :=
by
  sorry

end value_of_x_l113_113896


namespace circle_equation_l113_113588

/-- Given that point C is above the x-axis and
    the circle C with center C is tangent to the x-axis at point A(1,0) and
    intersects with circle O: x² + y² = 4 at points P and Q such that
    the length of PQ is sqrt(14)/2, the standard equation of circle C
    is (x - 1)² + (y - 1)² = 1. -/
theorem circle_equation {C : ℝ × ℝ} (hC : C.2 > 0) (tangent_at_A : C = (1, C.2))
  (intersect_with_O : ∃ P Q : ℝ × ℝ, (P ≠ Q) ∧ (P.1 ^ 2 + P.2 ^ 2 = 4) ∧ 
  (Q.1 ^ 2 + Q.2 ^ 2 = 4) ∧ ((P.1 - 1)^2 + (P.2 - C.2)^2 = C.2^2) ∧ 
  ((Q.1 - 1)^2 + (Q.2 - C.2)^2 = C.2^2) ∧ ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 14/4)) :
  (C.2 = 1) ∧ ((x - 1)^2 + (y - 1)^2 = 1) :=
by
  sorry

end circle_equation_l113_113588


namespace range_of_a_l113_113183

theorem range_of_a (a : ℝ) : 
  (∀ x : ℕ, (1 ≤ x ∧ x ≤ 4) → ax + 4 ≥ 0) → (-1 ≤ a ∧ a < -4/5) :=
by
  sorry

end range_of_a_l113_113183


namespace trigonometric_identity_proof_l113_113527

variable (α : Real)

theorem trigonometric_identity_proof (h1 : Real.tan α = 4 / 3) (h2 : 0 < α ∧ α < Real.pi / 2) :
  Real.sin (Real.pi + α) + Real.cos (Real.pi - α) = -7 / 5 :=
by
  sorry

end trigonometric_identity_proof_l113_113527


namespace three_digit_integer_condition_l113_113145

theorem three_digit_integer_condition (n a b c : ℕ) (hn : 100 ≤ n ∧ n < 1000)
  (hdigits : n = 100 * a + 10 * b + c)
  (hdadigits : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9)
  (fact_condition : 2 * n / 3 = a.factorial * b.factorial * c.factorial) :
  n = 432 := sorry

end three_digit_integer_condition_l113_113145


namespace determine_xy_l113_113188

noncomputable section

open Real

def op_defined (ab xy : ℝ × ℝ) : ℝ × ℝ :=
  (ab.1 * xy.1 + ab.2 * xy.2, ab.1 * xy.2 + ab.2 * xy.1)

theorem determine_xy (x y : ℝ) :
  (∀ (a b : ℝ), op_defined (a, b) (x, y) = (a, b)) → (x = 1 ∧ y = 0) :=
by
  sorry

end determine_xy_l113_113188


namespace smallest_integer_inequality_l113_113306

theorem smallest_integer_inequality :
  ∃ n : ℤ, (∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ n * (x^4 + y^4 + z^4 + w^4)) ∧ 
           (∀ m : ℤ, m < n → ¬∀ x y z w : ℝ, (x^2 + y^2 + z^2 + w^2)^2 ≤ m * (x^4 + y^4 + z^4 + w^4)) :=
by
  sorry

end smallest_integer_inequality_l113_113306


namespace inequality_solution_l113_113315

theorem inequality_solution (x : ℝ) : 
  (x-20) / (x+16) ≤ 0 ↔ -16 < x ∧ x ≤ 20 := by
  sorry

end inequality_solution_l113_113315


namespace circumference_irrational_l113_113109

theorem circumference_irrational (d : ℚ) : ¬ ∃ (r : ℚ), r = π * d :=
sorry

end circumference_irrational_l113_113109


namespace area_of_intersection_l113_113721

-- Define the region M
def in_region_M (x y : ℝ) : Prop :=
  y ≥ 0 ∧ y ≤ x ∧ y ≤ 2 - x

-- Define the region N as it changes with t
def in_region_N (t x : ℝ) : Prop :=
  t ≤ x ∧ x ≤ t + 1 ∧ 0 ≤ t ∧ t ≤ 1

-- Define the function f(t) which represents the common area of M and N
noncomputable def f (t : ℝ) : ℝ :=
  -t^2 + t + 0.5

-- Prove that f(t) is correct given the above conditions
theorem area_of_intersection (t : ℝ) :
  (∀ x y : ℝ, in_region_M x y → in_region_N t x → y ≤ f t) →
  0 ≤ t ∧ t ≤ 1 →
  f t = -t^2 + t + 0.5 :=
by
  sorry

end area_of_intersection_l113_113721


namespace correct_option_l113_113941

theorem correct_option :
  (2 * Real.sqrt 5) + (3 * Real.sqrt 5) = 5 * Real.sqrt 5 :=
by sorry

end correct_option_l113_113941


namespace max_possible_value_e_l113_113695

def b (n : ℕ) : ℕ := (7^n - 1) / 6

def e (n : ℕ) : ℕ := Nat.gcd (b n) (b (n+1))

theorem max_possible_value_e (n : ℕ) : e n = 1 := by
  sorry

end max_possible_value_e_l113_113695


namespace group_product_number_l113_113074

theorem group_product_number (a : ℕ) (group_size : ℕ) (interval : ℕ) (fifth_group_product : ℕ) :
  fifth_group_product = a + 4 * interval → fifth_group_product = 94 → group_size = 5 → interval = 20 →
  (a + (1 - 1) * interval + 1 * interval) = 34 :=
by
  intros fifth_group_eq fifth_group_is_94 group_size_is_5 interval_is_20
  -- Missing steps are handled by sorry
  sorry

end group_product_number_l113_113074


namespace true_propositions_count_l113_113092

theorem true_propositions_count (a : ℝ) :
  ((a > -3 → a > -6) ∧ (a > -6 → ¬(a ≤ -3)) ∧ (a ≤ -3 → ¬(a > -6)) ∧ (a ≤ -6 → a ≤ -3)) → 
  2 = 2 := 
by
  sorry

end true_propositions_count_l113_113092


namespace abs_sum_eq_two_l113_113262

theorem abs_sum_eq_two (a b c : ℤ) (h : (a - b) ^ 10 + (a - c) ^ 10 = 1) : 
  abs (a - b) + abs (b - c) + abs (c - a) = 2 := 
sorry

end abs_sum_eq_two_l113_113262


namespace count_valid_n_l113_113305

theorem count_valid_n :
  let n_values := [50, 550, 1050, 1550, 2050]
  ( ∀ n : ℤ, (50 * ((n + 500) / 50) - 500 = n) ∧ (Int.floor (Real.sqrt (2 * n : ℝ)) = (n + 500) / 50) → n ∈ n_values ) ∧
  ((∀ n : ℤ, ∃ k : ℤ, (n = 50 * k - 500) ∧ (k = Int.floor (Real.sqrt (2 * (50 * k - 500) : ℝ))) ∧ 0 < n ) → n_values.length = 5) :=
by
  sorry

end count_valid_n_l113_113305


namespace problem_statement_l113_113438

noncomputable def x : ℝ := sorry -- Let x be a real number satisfying the condition

theorem problem_statement (x_real_cond : x + 1/x = 3) : 
  (x^12 - 7*x^8 + 2*x^4) = 44387*x - 15088 :=
sorry

end problem_statement_l113_113438


namespace range_of_a_l113_113792

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (a-1)*x^2 + a*x + 1 ≥ 0) : a ≥ 1 :=
by {
  sorry
}

end range_of_a_l113_113792


namespace correct_parameterizations_of_line_l113_113481

theorem correct_parameterizations_of_line :
  ∀ (t : ℝ),
    (∀ (x y : ℝ), ((x = 5/3) ∧ (y = 0) ∨ (x = 0) ∧ (y = -5) ∨ (x = -5/3) ∧ (y = 0) ∨ 
                   (x = 1) ∧ (y = -2) ∨ (x = -2) ∧ (y = -11)) → 
                   y = 3 * x - 5) ∧
    (∀ (a b : ℝ), ((a = 1) ∧ (b = 3) ∨ (a = 3) ∧ (b = 1) ∨ (a = -1) ∧ (b = -3) ∨
                   (a = 1/3) ∧ (b = 1)) → 
                   b = 3 * a) →
    -- Check only Options D and E
    ((x = 1) → (y = -2) → (a = 1/3) → (b = 1) → y = 3 * x - 5 ∧ b = 3 * a) ∨
    ((x = -2) → (y = -11) → (a = 1/3) → (b = 1) → y = 3 * x - 5 ∧ b = 3 * a) :=
by
  sorry

end correct_parameterizations_of_line_l113_113481


namespace expression_value_l113_113239

theorem expression_value : (100 - (1000 - 300)) - (1000 - (300 - 100)) = -1400 := by
  sorry

end expression_value_l113_113239


namespace remainder_when_sum_divided_by_29_l113_113041

theorem remainder_when_sum_divided_by_29 (c d : ℤ) (k j : ℤ) 
  (hc : c = 52 * k + 48) 
  (hd : d = 87 * j + 82) : 
  (c + d) % 29 = 22 := 
by 
  sorry

end remainder_when_sum_divided_by_29_l113_113041


namespace notebook_price_l113_113903

theorem notebook_price (students_buying_notebooks n c : ℕ) (total_students : ℕ := 36) (total_cost : ℕ := 990) :
  students_buying_notebooks > 18 ∧ c > n ∧ students_buying_notebooks * n * c = total_cost → c = 15 :=
by
  sorry

end notebook_price_l113_113903


namespace cube_side_length_l113_113691

theorem cube_side_length (n : ℕ) (h1 : 6 * n^2 = 1 / 4 * 6 * n^3) : n = 4 := 
by 
  sorry

end cube_side_length_l113_113691


namespace find_odd_number_between_30_and_50_with_remainder_2_when_divided_by_7_l113_113541

def isOdd (n : ℕ) : Prop := n % 2 = 1
def isInRange (n : ℕ) : Prop := 30 ≤ n ∧ n ≤ 50
def hasRemainderTwo (n : ℕ) : Prop := n % 7 = 2

theorem find_odd_number_between_30_and_50_with_remainder_2_when_divided_by_7 :
  ∃ n : ℕ, isInRange n ∧ isOdd n ∧ hasRemainderTwo n ∧ n = 37 :=
by
  sorry

end find_odd_number_between_30_and_50_with_remainder_2_when_divided_by_7_l113_113541


namespace complex_number_simplification_l113_113203

theorem complex_number_simplification (i : ℂ) (h : i^2 = -1) : 
  (↑(1 : ℂ) - i) / (↑(1 : ℂ) + i) ^ 2017 = -i :=
sorry

end complex_number_simplification_l113_113203


namespace suitable_M_unique_l113_113452

noncomputable def is_suitable_M (M : ℝ) : Prop :=
  ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) →
  (1 + M ≤ a + M / (a * b)) ∨ 
  (1 + M ≤ b + M / (b * c)) ∨ 
  (1 + M ≤ c + M / (c * a))

theorem suitable_M_unique : is_suitable_M (1/2) ∧ 
  (∀ (M : ℝ), is_suitable_M M → M = 1/2) :=
by
  sorry

end suitable_M_unique_l113_113452


namespace cost_of_sneakers_l113_113874

theorem cost_of_sneakers (saved money per_action_figure final_money cost : ℤ) 
  (h1 : saved = 15) 
  (h2 : money = 10) 
  (h3 : per_action_figure = 10) 
  (h4 : final_money = 25) 
  (h5 : money * per_action_figure + saved - cost = final_money) 
  : cost = 90 := 
sorry

end cost_of_sneakers_l113_113874


namespace color_ball_ratios_l113_113332

theorem color_ball_ratios (white_balls red_balls blue_balls : ℕ)
  (h_white : white_balls = 12)
  (h_red_ratio : 4 * red_balls = 3 * white_balls)
  (h_blue_ratio : 4 * blue_balls = 2 * white_balls) :
  red_balls = 9 ∧ blue_balls = 6 :=
by
  sorry

end color_ball_ratios_l113_113332


namespace union_of_A_and_B_l113_113814

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 3, 5, 7}

theorem union_of_A_and_B :
  (A ∪ B) = {1, 2, 3, 4, 5, 7} := 
by
  sorry

end union_of_A_and_B_l113_113814


namespace lcm_condition_l113_113992

theorem lcm_condition (m : ℕ) (h_m_pos : m > 0) (h1 : Nat.lcm 30 m = 90) (h2 : Nat.lcm m 45 = 180) : m = 36 :=
by
  sorry

end lcm_condition_l113_113992


namespace largest_number_l113_113281

theorem largest_number 
  (A : ℝ) (B : ℝ) (C : ℝ) (D : ℝ) (E : ℝ)
  (hA : A = 0.986)
  (hB : B = 0.9851)
  (hC : C = 0.9869)
  (hD : D = 0.9807)
  (hE : E = 0.9819)
  : C > A ∧ C > B ∧ C > D ∧ C > E :=
by
  sorry

end largest_number_l113_113281


namespace bmws_sold_l113_113799

-- Definitions stated by the problem:
def total_cars : ℕ := 300
def percentage_mercedes : ℝ := 0.20
def percentage_toyota : ℝ := 0.25
def percentage_nissan : ℝ := 0.10
def percentage_bmws : ℝ := 1 - (percentage_mercedes + percentage_toyota + percentage_nissan)

-- Statement to prove:
theorem bmws_sold : (total_cars : ℝ) * percentage_bmws = 135 := by
  sorry

end bmws_sold_l113_113799


namespace expand_expression_l113_113028

variable {R : Type _} [CommRing R] (x : R)

theorem expand_expression :
  (3*x^2 + 7*x + 4) * (5*x - 2) = 15*x^3 + 29*x^2 + 6*x - 8 :=
by
  sorry

end expand_expression_l113_113028


namespace chef_leftover_potatoes_l113_113951

-- Defining the conditions as variables
def fries_per_potato := 25
def total_potatoes := 15
def fries_needed := 200

-- Calculating the number of potatoes needed.
def potatoes_needed : ℕ :=
  fries_needed / fries_per_potato

-- Calculating the leftover potatoes.
def leftovers : ℕ :=
  total_potatoes - potatoes_needed

-- The theorem statement
theorem chef_leftover_potatoes :
  leftovers = 7 :=
by
  -- the actual proof is omitted.
  sorry

end chef_leftover_potatoes_l113_113951


namespace pyramid_cross_section_distance_l113_113684

theorem pyramid_cross_section_distance 
  (A1 A2 : ℝ) (d : ℝ) (h : ℝ) 
  (hA1 : A1 = 125 * Real.sqrt 3)
  (hA2 : A2 = 500 * Real.sqrt 3)
  (hd : d = 12) :
  h = 24 :=
by
  sorry

end pyramid_cross_section_distance_l113_113684


namespace solve_for_T_l113_113499

theorem solve_for_T : ∃ T : ℝ, (3 / 4) * (1 / 6) * T = (2 / 5) * (1 / 4) * 200 ∧ T = 80 :=
by
  use 80
  -- The proof part is omitted as instructed
  sorry

end solve_for_T_l113_113499


namespace max_planes_15_points_l113_113429

theorem max_planes_15_points (P : Finset (Fin 15)) (hP : ∀ (p1 p2 p3 : Fin 15), p1 ≠ p2 → p2 ≠ p3 → p1 ≠ p3) :
  P.card = 15 → (∃ planes : Finset (Finset (Fin 15)), planes.card = 455) := by
  sorry

end max_planes_15_points_l113_113429


namespace small_denominator_difference_l113_113614

theorem small_denominator_difference :
  ∃ (p q : ℕ), 0 < p ∧ 0 < q ∧
               (5 : ℚ) / 9 < (p : ℚ) / q ∧
               (p : ℚ) / q < 4 / 7 ∧
               (∀ r, 0 < r → (5 : ℚ) / 9 < (p : ℚ) / r → (p : ℚ) / r < 4 / 7 → q ≤ r) ∧
               q - p = 7 := 
  by
  sorry

end small_denominator_difference_l113_113614


namespace problem_statement_l113_113096

-- Definitions of the events as described in the problem conditions.
def event1 (a b : ℝ) : Prop := a * b < 0 → a + b < 0
def event2 (a b : ℝ) : Prop := a * b < 0 → a - b > 0
def event3 (a b : ℝ) : Prop := a * b < 0 → a * b > 0
def event4 (a b : ℝ) : Prop := a * b < 0 → a / b < 0

-- The problem statement combining the conditions and the conclusion.
theorem problem_statement (a b : ℝ) (h1 : a * b < 0):
  (event4 a b) ∧ ¬(event3 a b) ∧ (event1 a b ∨ ¬(event1 a b)) ∧ (event2 a b ∨ ¬(event2 a b)) :=
by
  sorry

end problem_statement_l113_113096


namespace correct_sampling_methods_l113_113135

-- Define the surveys with their corresponding conditions
structure Survey1 where
  high_income : Nat
  middle_income : Nat
  low_income : Nat
  total_households : Nat

structure Survey2 where
  total_students : Nat
  sample_students : Nat
  differences_small : Bool
  sizes_small : Bool

-- Define the conditions
def survey1_conditions (s : Survey1) : Prop :=
  s.high_income = 125 ∧ s.middle_income = 280 ∧ s.low_income = 95 ∧ s.total_households = 100

def survey2_conditions (s : Survey2) : Prop :=
  s.total_students = 15 ∧ s.sample_students = 3 ∧ s.differences_small = true ∧ s.sizes_small = true

-- Define the answer predicate
def correct_answer (method1 method2 : String) : Prop :=
  method1 = "stratified sampling" ∧ method2 = "simple random sampling"

-- The theorem statement
theorem correct_sampling_methods (s1 : Survey1) (s2 : Survey2) :
  survey1_conditions s1 → survey2_conditions s2 → correct_answer "stratified sampling" "simple random sampling" :=
by
  -- Proof skipped for problem statement purpose
  sorry

end correct_sampling_methods_l113_113135


namespace remaining_gnomes_total_l113_113931

/--
The remaining number of gnomes in the three forests after the owner takes his specified percentages.
-/
theorem remaining_gnomes_total :
  let westerville_gnomes := 20
  let ravenswood_gnomes := 4 * westerville_gnomes
  let greenwood_grove_gnomes := ravenswood_gnomes + (25 * ravenswood_gnomes) / 100
  let remaining_ravenswood := ravenswood_gnomes - (40 * ravenswood_gnomes) / 100
  let remaining_westerville := westerville_gnomes - (30 * westerville_gnomes) / 100
  let remaining_greenwood_grove := greenwood_grove_gnomes - (50 * greenwood_grove_gnomes) / 100
  remaining_ravenswood + remaining_westerville + remaining_greenwood_grove = 112 := by
  sorry

end remaining_gnomes_total_l113_113931


namespace jana_walk_distance_l113_113835

theorem jana_walk_distance :
  (1 / 20 * 15 : ℝ) = 0.8 :=
by sorry

end jana_walk_distance_l113_113835


namespace empty_seats_correct_l113_113459

def children_count : ℕ := 52
def adult_count : ℕ := 29
def total_seats : ℕ := 95

theorem empty_seats_correct :
  total_seats - (children_count + adult_count) = 14 :=
by
  sorry

end empty_seats_correct_l113_113459


namespace JuanitaDessertCost_l113_113419

-- Define costs as constants
def brownieCost : ℝ := 2.50
def regularScoopCost : ℝ := 1.00
def premiumScoopCost : ℝ := 1.25
def deluxeScoopCost : ℝ := 1.50
def syrupCost : ℝ := 0.50
def nutsCost : ℝ := 1.50
def whippedCreamCost : ℝ := 0.75
def cherryCost : ℝ := 0.25

-- Define the total cost calculation
def totalCost : ℝ := brownieCost + regularScoopCost + premiumScoopCost +
                     deluxeScoopCost + syrupCost + syrupCost + nutsCost + whippedCreamCost + cherryCost

-- The proof problem: Prove that total cost equals $9.75
theorem JuanitaDessertCost : totalCost = 9.75 :=
by
  -- Proof is omitted
  sorry

end JuanitaDessertCost_l113_113419


namespace area_of_yard_proof_l113_113569

def area_of_yard (L W : ℕ) : ℕ :=
  L * W

theorem area_of_yard_proof (L W : ℕ) (hL : L = 40) (hFence : 2 * W + L = 52) : 
  area_of_yard L W = 240 := 
by 
  sorry

end area_of_yard_proof_l113_113569


namespace find_a_from_inclination_l113_113113

open Real

theorem find_a_from_inclination (a : ℝ) :
  (∃ (k : ℝ), k = (2 - (-3)) / (1 - a) ∧ k = tan (135 * pi / 180)) → a = 6 :=
by
  sorry

end find_a_from_inclination_l113_113113


namespace find_a12_l113_113052

variable (a : ℕ → ℤ)
variable (H1 : a 1 = 1) 
variable (H2 : ∀ m n : ℕ, a (m + n) = a m + a n + m * n)

theorem find_a12 : a 12 = 78 := 
by
  sorry

end find_a12_l113_113052


namespace solution_correct_l113_113148

def mixed_number_to_fraction (a b c : ℕ) : ℚ :=
  (a * b + c) / b

def percentage_to_decimal (fraction : ℚ) : ℚ :=
  fraction / 100

def evaluate_expression : ℚ :=
  let part1 := 63 * 5 + 4
  let part2 := 48 * 7 + 3
  let part3 := 17 * 3 + 2
  let term1 := (mixed_number_to_fraction 63 5 4) * 3150
  let term2 := (mixed_number_to_fraction 48 7 3) * 2800
  let term3 := (mixed_number_to_fraction 17 3 2) * 945 / 2
  term1 - term2 + term3

theorem solution_correct :
  (percentage_to_decimal (mixed_number_to_fraction 63 5 4) * 3150) -
  (percentage_to_decimal (mixed_number_to_fraction 48 7 3) * 2800) +
  (percentage_to_decimal (mixed_number_to_fraction 17 3 2) * 945 / 2) = 737.175 := 
sorry

end solution_correct_l113_113148


namespace full_price_tickets_revenue_l113_113901

theorem full_price_tickets_revenue (f h p : ℕ) (h1 : f + h + 12 = 160) (h2 : f * p + h * (p / 2) + 12 * (2 * p) = 2514) :  f * p = 770 := 
sorry

end full_price_tickets_revenue_l113_113901


namespace equal_probability_among_children_l113_113297

theorem equal_probability_among_children
    (n : ℕ := 100)
    (p : ℝ := 0.232818)
    (k : ℕ := 18)
    (h_pos : 0 < p)
    (h_lt : p < 1)
    (num_outcomes : ℕ := 2^k) :
  ∃ (dist : Fin n → Fin num_outcomes),
    ∀ i : Fin num_outcomes, ∃ j : Fin n, dist j = i ∧ p ^ k * (1 - p) ^ (num_outcomes - k) = 1 / n :=
by
  sorry

end equal_probability_among_children_l113_113297


namespace sum_coefficients_eq_neg_one_l113_113288

theorem sum_coefficients_eq_neg_one (a a1 a2 a3 a4 a5 : ℝ) :
  (∀ x y : ℝ, (x - 2 * y)^5 = a * x^5 + a1 * x^4 * y + a2 * x^3 * y^2 + a3 * x^2 * y^3 + a4 * x * y^4 + a5 * y^5) →
  a + a1 + a2 + a3 + a4 + a5 = -1 :=
by
  sorry

end sum_coefficients_eq_neg_one_l113_113288


namespace binary_addition_is_correct_l113_113184

theorem binary_addition_is_correct :
  (0b101101 + 0b1011 + 0b11001 + 0b1110101 + 0b1111) = 0b10010001 :=
by sorry

end binary_addition_is_correct_l113_113184


namespace triangle_angle_equality_l113_113964

theorem triangle_angle_equality
  (α β γ α₁ β₁ γ₁ : ℝ)
  (hABC : α + β + γ = 180)
  (hA₁B₁C₁ : α₁ + β₁ + γ₁ = 180)
  (angle_relation : (α = α₁ ∨ α + α₁ = 180) ∧ (β = β₁ ∨ β + β₁ = 180) ∧ (γ = γ₁ ∨ γ + γ₁ = 180)) :
  α = α₁ ∧ β = β₁ ∧ γ = γ₁ :=
by {
  sorry
}

end triangle_angle_equality_l113_113964


namespace purely_imaginary_necessary_not_sufficient_l113_113296

-- Definition of a purely imaginary number
def is_purely_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

-- Theorem statement
theorem purely_imaginary_necessary_not_sufficient (a b : ℝ) :
  a = 0 → (z : ℂ) = ⟨a, b⟩ → is_purely_imaginary z ↔ (a = 0 ∧ b ≠ 0) :=
by
  sorry

end purely_imaginary_necessary_not_sufficient_l113_113296


namespace fitted_ball_volume_l113_113865

noncomputable def volume_of_fitted_ball (d_ball d_h1 r_h1 d_h2 r_h2 : ℝ) : ℝ :=
  let r_ball := d_ball / 2
  let v_ball := (4 / 3) * Real.pi * r_ball^3
  let r_hole1 := r_h1
  let r_hole2 := r_h2
  let v_hole1 := Real.pi * r_hole1^2 * d_h1
  let v_hole2 := Real.pi * r_hole2^2 * d_h2
  v_ball - 2 * v_hole1 - v_hole2

theorem fitted_ball_volume :
  volume_of_fitted_ball 24 10 (3 / 2) 10 2 = 2219 * Real.pi :=
by
  sorry

end fitted_ball_volume_l113_113865


namespace jonah_added_yellow_raisins_l113_113833

variable (y : ℝ)

theorem jonah_added_yellow_raisins (h : y + 0.4 = 0.7) : y = 0.3 := by
  sorry

end jonah_added_yellow_raisins_l113_113833


namespace eval_x_squared_minus_y_squared_l113_113243

theorem eval_x_squared_minus_y_squared (x y : ℝ) (h1 : 3 * x + 2 * y = 30) (h2 : 4 * x + 2 * y = 34) : x^2 - y^2 = -65 :=
by
  sorry

end eval_x_squared_minus_y_squared_l113_113243


namespace find_a_if_lines_parallel_l113_113972

theorem find_a_if_lines_parallel (a : ℝ) (h1 : ∃ y : ℝ, y = - (a / 4) * (1 : ℝ) + (1 / 4)) (h2 : ∃ y : ℝ, y = - (1 / a) * (1 : ℝ) + (1 / (2 * a))) : a = -2 :=
sorry

end find_a_if_lines_parallel_l113_113972


namespace device_records_720_instances_in_one_hour_l113_113566

-- Definitions
def seconds_per_hour : ℕ := 3600
def interval : ℕ := 5
def instances_per_hour := seconds_per_hour / interval

-- Theorem Statement
theorem device_records_720_instances_in_one_hour : instances_per_hour = 720 :=
by
  sorry

end device_records_720_instances_in_one_hour_l113_113566


namespace present_age_of_B_l113_113996

theorem present_age_of_B
  (A B : ℕ)
  (h1 : A = B + 5)
  (h2 : A + 30 = 2 * (B - 30)) :
  B = 95 :=
by { sorry }

end present_age_of_B_l113_113996


namespace work_rate_l113_113060

theorem work_rate (R_B : ℚ) (R_A : ℚ) (R_total : ℚ) (days : ℚ)
  (h1 : R_A = (1/2) * R_B)
  (h2 : R_B = 1 / 22.5)
  (h3 : R_total = R_A + R_B)
  (h4 : days = 1 / R_total) : 
  days = 15 := 
sorry

end work_rate_l113_113060


namespace sum_of_first_six_terms_l113_113863

def geometric_seq_sum (a r : ℕ) (n : ℕ) : ℕ :=
  a * (r^n - 1) / (r - 1)

theorem sum_of_first_six_terms (a : ℕ) (r : ℕ) (h1 : r = 2) (h2 : a * (1 + r + r^2) = 3) :
  geometric_seq_sum a r 6 = 27 :=
by
  sorry

end sum_of_first_six_terms_l113_113863


namespace total_students_in_school_l113_113594

noncomputable def small_school_students (boys girls : ℕ) (total_students : ℕ) : Prop :=
boys = 42 ∧ 
(girls : ℕ) = boys / 7 ∧
total_students = boys + girls

theorem total_students_in_school : small_school_students 42 6 48 :=
by
  sorry

end total_students_in_school_l113_113594


namespace max_ab_l113_113322

theorem max_ab (a b c : ℝ) (h1 : 3 * a + b = 1) (h2 : 0 ≤ a) (h3 : a < 1) (h4 : 0 ≤ b) 
(h5 : b < 1) (h6 : 0 ≤ c) (h7 : c < 1) (h8 : a + b + c = 1) : 
  ab ≤ 1 / 12 := by
  sorry

end max_ab_l113_113322


namespace carl_highway_miles_l113_113403

theorem carl_highway_miles
  (city_mpg : ℕ)
  (highway_mpg : ℕ)
  (city_miles : ℕ)
  (gas_cost_per_gallon : ℕ)
  (total_cost : ℕ)
  (h1 : city_mpg = 30)
  (h2 : highway_mpg = 40)
  (h3 : city_miles = 60)
  (h4 : gas_cost_per_gallon = 3)
  (h5 : total_cost = 42)
  : (total_cost - (city_miles / city_mpg) * gas_cost_per_gallon) / gas_cost_per_gallon * highway_mpg = 480 := 
by
  sorry

end carl_highway_miles_l113_113403


namespace average_of_P_and_R_l113_113573

theorem average_of_P_and_R (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 2050)
  (h2 : (Q + R) / 2 = 5250)
  (h3 : P = 3000)
  : (P + R) / 2 = 6200 := by
  sorry

end average_of_P_and_R_l113_113573


namespace largest_result_l113_113453

theorem largest_result (a b c : ℕ) (h1 : a = 0 / 100) (h2 : b = 0 * 100) (h3 : c = 100 - 0) : 
  c > a ∧ c > b :=
by
  sorry

end largest_result_l113_113453


namespace max_min_product_xy_theorem_l113_113005

noncomputable def max_min_product_xy (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) : Prop :=
  -1 ≤ x * y ∧ x * y ≤ 1/2

theorem max_min_product_xy_theorem (x y a : ℝ) (h1 : x + y = a) (h2 : x^2 + y^2 = -a^2 + 2) :
  max_min_product_xy x y a h1 h2 :=
sorry

end max_min_product_xy_theorem_l113_113005


namespace tomatoes_left_l113_113862

theorem tomatoes_left (initial_tomatoes picked_yesterday picked_today : ℕ)
    (h_initial : initial_tomatoes = 171)
    (h_picked_yesterday : picked_yesterday = 134)
    (h_picked_today : picked_today = 30) :
    initial_tomatoes - picked_yesterday - picked_today = 7 :=
by
    sorry

end tomatoes_left_l113_113862


namespace find_m_direct_proportion_l113_113505

theorem find_m_direct_proportion (m : ℝ) (h1 : m + 2 ≠ 0) (h2 : |m| - 1 = 1) : m = 2 :=
sorry

end find_m_direct_proportion_l113_113505


namespace simplify_expression_l113_113421

theorem simplify_expression : 
  2 * (1 / (2 * 3) + 1 / (3 * 4) + 1 / (4 * 5) + 1 / (5 * 6) + 1 / (6 * 7) + 1 / (7 * 8)) = 3 / 4 :=
by
  sorry

end simplify_expression_l113_113421


namespace original_number_l113_113216

theorem original_number (x : ℕ) : 
  (∃ y : ℕ, y = x + 28 ∧ (y % 5 = 0) ∧ (y % 6 = 0) ∧ (y % 4 = 0) ∧ (y % 3 = 0)) → x = 32 :=
by
  sorry

end original_number_l113_113216


namespace simplify_and_evaluate_l113_113723

theorem simplify_and_evaluate : 
    ∀ (a b : ℤ), a = 1 → b = -1 → 
    ((2 * a^2 * b - 2 * a * b^2 - b^3) / b - (a + b) * (a - b) = 3) := 
by
  intros a b ha hb
  sorry

end simplify_and_evaluate_l113_113723


namespace leila_toys_l113_113234

theorem leila_toys:
  ∀ (x : ℕ),
  (∀ l m : ℕ, l = 2 * x ∧ m = 3 * 19 ∧ m = l + 7 → x = 25) :=
by
  sorry

end leila_toys_l113_113234


namespace smallest_a_l113_113090

theorem smallest_a (a : ℕ) (h1 : a > 0) (h2 : (∀ b : ℕ, b > 0 → b < a → ∀ h3 : b > 0, ¬ (gcd b 72 > 1 ∧ gcd b 90 > 1)))
  (h3 : gcd a 72 > 1) (h4 : gcd a 90 > 1) : a = 2 :=
by
  sorry

end smallest_a_l113_113090


namespace total_candies_correct_l113_113709

-- Define the number of candies each has
def caleb_jellybeans := 3 * 12
def caleb_chocolate_bars := 5
def caleb_gummy_bears := 8
def caleb_total := caleb_jellybeans + caleb_chocolate_bars + caleb_gummy_bears

def sophie_jellybeans := (caleb_jellybeans / 2)
def sophie_chocolate_bars := 3
def sophie_gummy_bears := 12
def sophie_total := sophie_jellybeans + sophie_chocolate_bars + sophie_gummy_bears

def max_jellybeans := (2 * 12) + sophie_jellybeans
def max_chocolate_bars := 6
def max_gummy_bears := 10
def max_total := max_jellybeans + max_chocolate_bars + max_gummy_bears

-- Define the total number of candies
def total_candies := caleb_total + sophie_total + max_total

-- Theorem statement
theorem total_candies_correct : total_candies = 140 := by
  sorry

end total_candies_correct_l113_113709


namespace symmetric_points_origin_l113_113793

theorem symmetric_points_origin (a b : ℝ) (h1 : a = -(-2)) (h2 : 1 = -b) : a + b = 1 :=
by
  sorry

end symmetric_points_origin_l113_113793


namespace part1_part2_1_part2_2_l113_113046

-- Define the operation
def mul_op (x y : ℚ) : ℚ := x ^ 2 - 3 * y + 3

-- Part 1: Prove (-4) * 2 = 13 given the operation definition
theorem part1 : mul_op (-4) 2 = 13 := sorry

-- Part 2.1: Simplify (a - b) * (a - b)^2
theorem part2_1 (a b : ℚ) : mul_op (a - b) ((a - b) ^ 2) = -2 * a ^ 2 - 2 * b ^ 2 + 4 * a * b + 3 := sorry

-- Part 2.2: Find the value of the expression when a = -2 and b = 1/2
theorem part2_2 : mul_op (-2 - 1/2) ((-2 - 1/2) ^ 2) = -13 / 2 := sorry

end part1_part2_1_part2_2_l113_113046


namespace crayons_total_l113_113591

theorem crayons_total (Billy_crayons : ℝ) (Jane_crayons : ℝ)
  (h1 : Billy_crayons = 62.0) (h2 : Jane_crayons = 52.0) :
  Billy_crayons + Jane_crayons = 114.0 := 
by
  sorry

end crayons_total_l113_113591


namespace intersection_P_compl_M_l113_113797

-- Define universal set U
def U : Set ℤ := Set.univ

-- Define set M
def M : Set ℤ := {1, 2}

-- Define set P
def P : Set ℤ := {-2, -1, 0, 1, 2}

-- Define the complement of M in U
def M_compl : Set ℤ := { x | x ∉ M }

-- Define the intersection of P and the complement of M
def P_inter_M_compl : Set ℤ := P ∩ M_compl

-- The theorem we want to prove
theorem intersection_P_compl_M : P_inter_M_compl = {-2, -1, 0} := 
by {
  sorry
}

end intersection_P_compl_M_l113_113797


namespace infinitely_many_composite_values_l113_113460

theorem infinitely_many_composite_values (k m : ℕ) 
  (h_k : k ≥ 2) : 
  ∃ n : ℕ, n = 4 * k^4 ∧ ∀ m : ℕ, ∃ x y : ℕ, x > 1 ∧ y > 1 ∧ m^4 + n = x * y :=
by
  sorry

end infinitely_many_composite_values_l113_113460


namespace binom_12_9_is_220_l113_113600

def choose (n k : ℕ) : ℕ := n.choose k

theorem binom_12_9_is_220 :
  choose 12 9 = 220 :=
by {
  -- Proof is omitted
  sorry
}

end binom_12_9_is_220_l113_113600


namespace equidistant_points_quadrants_l113_113444

open Real

theorem equidistant_points_quadrants : 
  ∀ x y : ℝ, 
    (4 * x + 6 * y = 24) → (|x| = |y|) → 
    ((0 < x ∧ 0 < y) ∨ (x < 0 ∧ 0 < y)) :=
by
  sorry

end equidistant_points_quadrants_l113_113444


namespace volume_of_wall_is_16128_l113_113788

def wall_width : ℝ := 4
def wall_height : ℝ := 6 * wall_width
def wall_length : ℝ := 7 * wall_height

def wall_volume : ℝ := wall_length * wall_width * wall_height

theorem volume_of_wall_is_16128 :
  wall_volume = 16128 := by
  sorry

end volume_of_wall_is_16128_l113_113788


namespace divide_milk_l113_113081

theorem divide_milk : (3 / 5 : ℚ) = 3 / 5 := by {
    sorry
}

end divide_milk_l113_113081


namespace jenny_total_distance_seven_hops_l113_113448

noncomputable def sum_geometric_series (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r ^ n) / (1 - r)

theorem jenny_total_distance_seven_hops :
  let a := (1 / 4 : ℚ)
  let r := (3 / 4 : ℚ)
  let n := 7
  sum_geometric_series a r n = (14197 / 16384 : ℚ) :=
by
  sorry

end jenny_total_distance_seven_hops_l113_113448


namespace inradius_of_triangle_l113_113934

theorem inradius_of_triangle (A p r s : ℝ) (h1 : A = 3 * p) (h2 : A = r * s) (h3 : s = p / 2) :
  r = 6 :=
by
  sorry

end inradius_of_triangle_l113_113934


namespace shape_is_cylinder_l113_113177

def is_cylinder (c : ℝ) (r θ z : ℝ) : Prop :=
  c > 0 ∧ r = c ∧ 0 ≤ θ ∧ θ < 2 * Real.pi ∧ True

theorem shape_is_cylinder (c : ℝ) (r θ z : ℝ) (h : c > 0) :
  is_cylinder c r θ z :=
by
  -- Proof is omitted
  sorry

end shape_is_cylinder_l113_113177


namespace total_cans_given_away_l113_113379

noncomputable def total_cans_initial : ℕ := 2000
noncomputable def cans_taken_first_day : ℕ := 500
noncomputable def restocked_first_day : ℕ := 1500
noncomputable def people_second_day : ℕ := 1000
noncomputable def cans_per_person_second_day : ℕ := 2
noncomputable def restocked_second_day : ℕ := 3000

theorem total_cans_given_away :
  (cans_taken_first_day + (people_second_day * cans_per_person_second_day) = 2500) :=
by
  sorry

end total_cans_given_away_l113_113379


namespace subset_bound_l113_113500

theorem subset_bound {m n k : ℕ} (h1 : m ≥ n) (h2 : n > 1) 
  (F : Fin k → Finset (Fin m)) 
  (hF : ∀ i j, i < j → (F i ∩ F j).card ≤ 1) 
  (hcard : ∀ i, (F i).card = n) : 
  k ≤ (m * (m - 1)) / (n * (n - 1)) :=
sorry

end subset_bound_l113_113500


namespace evaluate_fraction_l113_113642

theorem evaluate_fraction (a b c : ℝ) (h : a^3 - b^3 + c^3 ≠ 0) :
  (a^6 - b^6 + c^6) / (a^3 - b^3 + c^3) = a^3 + b^3 + c^3 :=
sorry

end evaluate_fraction_l113_113642


namespace solve_quadratic_l113_113122

theorem solve_quadratic (x : ℝ) (h₁ : x > 0) (h₂ : 3 * x^2 + 7 * x - 20 = 0) : x = 5 / 3 :=
sorry

end solve_quadratic_l113_113122


namespace fish_tank_ratio_l113_113598

theorem fish_tank_ratio :
  ∀ (F1 F2 F3: ℕ),
  F1 = 15 →
  F3 = 10 →
  (F3 = (1 / 3 * F2)) →
  F2 / F1 = 2 :=
by
  intros F1 F2 F3 hF1 hF3 hF2
  sorry

end fish_tank_ratio_l113_113598


namespace tommy_gum_given_l113_113404

variable (original_gum : ℕ) (luis_gum : ℕ) (final_total_gum : ℕ)

-- Defining the conditions
def conditions := original_gum = 25 ∧ luis_gum = 20 ∧ final_total_gum = 61

-- The theorem stating that Tommy gave Maria 16 pieces of gum
theorem tommy_gum_given (t_gum : ℕ) (h : conditions original_gum luis_gum final_total_gum) :
  t_gum = final_total_gum - (original_gum + luis_gum) → t_gum = 16 :=
by
  intros h
  sorry

end tommy_gum_given_l113_113404


namespace carl_insurance_payment_percentage_l113_113295

variable (property_damage : ℝ) (medical_bills : ℝ) 
          (total_cost : ℝ) (carl_payment : ℝ) (insurance_payment_percentage : ℝ)

theorem carl_insurance_payment_percentage :
  property_damage = 40000 ∧
  medical_bills = 70000 ∧
  total_cost = property_damage + medical_bills ∧
  carl_payment = 22000 ∧
  carl_payment = 0.20 * total_cost →
  insurance_payment_percentage = 100 - 20 :=
by
  sorry

end carl_insurance_payment_percentage_l113_113295


namespace unique_friends_count_l113_113207

-- Definitions from conditions
def M : ℕ := 10
def P : ℕ := 20
def G : ℕ := 5
def M_P : ℕ := 4
def M_G : ℕ := 2
def P_G : ℕ := 0
def M_P_G : ℕ := 2

-- Theorem we need to prove
theorem unique_friends_count : (M + P + G - M_P - M_G - P_G + M_P_G) = 31 := by
  sorry

end unique_friends_count_l113_113207


namespace trinomials_real_roots_inequality_l113_113166

theorem trinomials_real_roots_inequality :
  (∃ (p q : ℤ), 1 ≤ p ∧ p ≤ 1997 ∧ 1 ≤ q ∧ q ≤ 1997 ∧ 
   ¬ (∃ m n : ℤ, (1 ≤ m ∧ m ≤ 1997) ∧ (1 ≤ n ∧ n ≤ 1997) ∧ (m + n = p) ∧ (m * n = q))) >
  (∃ (p q : ℤ), 1 ≤ p ∧ p ≤ 1997 ∧ 1 ≤ q ∧ q ≤ 1997 ∧ 
   ∃ m n : ℤ, (1 ≤ m ∧ m ≤ 1997) ∧ (1 ≤ n ∧ n ≤ 1997) ∧ (m + n = p) ∧ (m * n = q)) :=
sorry

end trinomials_real_roots_inequality_l113_113166


namespace ratio_of_books_on_each_table_l113_113630

-- Define the conditions
variables (number_of_tables number_of_books : ℕ)
variables (R : ℕ) -- Ratio we need to find

-- State the conditions
def conditions := (number_of_tables = 500) ∧ (number_of_books = 100000)

-- Mathematical Problem Statement
theorem ratio_of_books_on_each_table (h : conditions number_of_tables number_of_books) :
    100000 = 500 * R → R = 200 :=
by
  sorry

end ratio_of_books_on_each_table_l113_113630


namespace probability_of_earning_exactly_2300_in_3_spins_l113_113019

-- Definitions of the conditions
def spinner_sections : List ℕ := [0, 1000, 200, 7000, 300]
def equal_area_sections : Prop := true  -- Each section has the same area, simple condition

-- Proving the probability of earning exactly $2300 in three spins
theorem probability_of_earning_exactly_2300_in_3_spins :
  ∃ p : ℚ, p = 3 / 125 := sorry

end probability_of_earning_exactly_2300_in_3_spins_l113_113019


namespace total_students_at_year_end_l113_113921

def initial_students : ℝ := 10.0
def added_students : ℝ := 4.0
def new_students : ℝ := 42.0

theorem total_students_at_year_end : initial_students + added_students + new_students = 56.0 :=
by
  sorry

end total_students_at_year_end_l113_113921


namespace range_of_k_l113_113783

theorem range_of_k (x k : ℝ):
  (2 * x + 9 > 6 * x + 1) → (x - k < 1) → (x < 2) → k ≥ 1 :=
by 
  sorry

end range_of_k_l113_113783


namespace even_numbers_average_19_l113_113489

theorem even_numbers_average_19 (n : ℕ) (h1 : (n / 2) * (2 + 2 * n) / n = 19) : n = 18 :=
by {
  sorry
}

end even_numbers_average_19_l113_113489


namespace divisibility_by_5_l113_113480

theorem divisibility_by_5 (x y : ℤ) : (x^2 - 2 * x * y + 2 * y^2) % 5 = 0 ∨ (x^2 + 2 * x * y + 2 * y^2) % 5 = 0 ↔ (x % 5 = 0 ∧ y % 5 = 0) ∨ (x % 5 ≠ 0 ∧ y % 5 ≠ 0) := 
by
  sorry

end divisibility_by_5_l113_113480


namespace regression_equation_is_correct_l113_113531

theorem regression_equation_is_correct 
  (linear_corr : ∃ (f : ℝ → ℝ), ∀ (x : ℝ), ∃ (y : ℝ), y = f x)
  (mean_b : ℝ)
  (mean_x : ℝ)
  (mean_y : ℝ)
  (mean_b_eq : mean_b = 0.51)
  (mean_x_eq : mean_x = 61.75)
  (mean_y_eq : mean_y = 38.14) : 
  mean_y = mean_b * mean_x + 6.65 :=
sorry

end regression_equation_is_correct_l113_113531


namespace monomial_combined_l113_113343

theorem monomial_combined (n m : ℕ) (h₁ : 2 = n) (h₂ : m = 4) : n^m = 16 := by
  sorry

end monomial_combined_l113_113343


namespace simplification_of_fractional_equation_l113_113015

theorem simplification_of_fractional_equation (x : ℝ) : 
  (x / (3 - x) - 4 = 6 / (x - 3)) -> (x - 4 * (3 - x) = -6) :=
by
  sorry

end simplification_of_fractional_equation_l113_113015


namespace percentage_of_hindu_boys_l113_113380

-- Define the total number of boys in the school
def total_boys := 700

-- Define the percentage of Muslim boys
def muslim_percentage := 44 / 100

-- Define the percentage of Sikh boys
def sikh_percentage := 10 / 100

-- Define the number of boys from other communities
def other_communities_boys := 126

-- State the main theorem to prove the percentage of Hindu boys
theorem percentage_of_hindu_boys (h1 : total_boys = 700)
                                 (h2 : muslim_percentage = 44 / 100)
                                 (h3 : sikh_percentage = 10 / 100)
                                 (h4 : other_communities_boys = 126) : 
                                 ((total_boys - (total_boys * muslim_percentage + total_boys * sikh_percentage + other_communities_boys)) / total_boys) * 100 = 28 :=
by {
  sorry
}

end percentage_of_hindu_boys_l113_113380


namespace min_value_geometric_seq_l113_113991

theorem min_value_geometric_seq (a : ℕ → ℝ) (r : ℝ) (n : ℕ) 
  (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, a (n + 1) = a n * r)
  (h3 : a 5 * a 4 * a 2 * a 1 = 16) :
  a 1 + a 5 = 4 :=
sorry

end min_value_geometric_seq_l113_113991


namespace number_representation_correct_l113_113564

-- Conditions: 5 in both the tenths and hundredths places, 0 in remaining places.
def number : ℝ := 50.05

theorem number_representation_correct :
  number = 50.05 :=
by 
  -- The proof will show that the definition satisfies the condition.
  sorry

end number_representation_correct_l113_113564


namespace initial_cost_of_article_l113_113355

variable (P : ℝ)

theorem initial_cost_of_article (h1 : 0.70 * P = 2100) (h2 : 0.50 * (0.70 * P) = 1050) : P = 3000 :=
by
  sorry

end initial_cost_of_article_l113_113355


namespace find_a_plus_b_l113_113853

theorem find_a_plus_b (a b : ℝ)
  (h1 : ab^2 = 0)
  (h2 : 2 * a^2 * b = 0)
  (h3 : a^3 + b^2 = 0)
  (h4 : ab = 1) : a + b = -2 :=
sorry

end find_a_plus_b_l113_113853


namespace students_height_order_valid_after_rearrangement_l113_113800
open List

variable {n : ℕ} -- number of students in each row
variable (a b : Fin n → ℝ) -- heights of students in each row

/-- Prove Gábor's observation remains valid after rearrangement: 
    each student in the back row is taller than the student in front of them.
    Given:
    - ∀ i, b i < a i (initial condition)
    - ∀ i < j, a i ≤ a j (rearrangement condition)
    Prove:
    - ∀ i, b i < a i (remains valid after rearrangement)
-/
theorem students_height_order_valid_after_rearrangement
  (h₁ : ∀ i : Fin n, b i < a i)
  (h₂ : ∀ (i j : Fin n), i < j → a i ≤ a j) :
  ∀ i : Fin n, b i < a i :=
by sorry

end students_height_order_valid_after_rearrangement_l113_113800


namespace todd_numbers_sum_eq_l113_113276

def sum_of_todd_numbers (n : ℕ) : ℕ :=
  sorry -- This would be the implementation of the sum based on provided problem conditions

theorem todd_numbers_sum_eq :
  sum_of_todd_numbers 5000 = 1250025 :=
sorry

end todd_numbers_sum_eq_l113_113276


namespace kirin_calculations_l113_113840

theorem kirin_calculations (calculations_per_second : ℝ) (seconds : ℝ) (h1 : calculations_per_second = 10^10) (h2 : seconds = 2022) : 
    calculations_per_second * seconds = 2.022 * 10^13 := 
by
  sorry

end kirin_calculations_l113_113840


namespace erwan_spending_l113_113445

def discount (price : ℕ) (percent : ℕ) : ℕ :=
  price - (price * percent / 100)

theorem erwan_spending (shoe_original_price : ℕ := 200) 
  (shoe_discount : ℕ := 30)
  (shirt_price : ℕ := 80)
  (num_shirts : ℕ := 2)
  (pants_price : ℕ := 150)
  (second_store_discount : ℕ := 20)
  (jacket_price : ℕ := 250)
  (tie_price : ℕ := 40)
  (hat_price : ℕ := 60)
  (watch_price : ℕ := 120)
  (wallet_price : ℕ := 49)
  (belt_price : ℕ := 35)
  (belt_discount : ℕ := 25)
  (scarf_price : ℕ := 45)
  (scarf_discount : ℕ := 10)
  (rewards_points_discount : ℕ := 5)
  (sales_tax : ℕ := 8)
  (gift_card : ℕ := 50)
  (shipping_fee : ℕ := 5)
  (num_shipping_stores : ℕ := 2) :
  ∃ total : ℕ,
    total = 85429 :=
by
  have first_store := discount shoe_original_price shoe_discount
  have second_store_total := pants_price + (shirt_price * num_shirts)
  have second_store := discount second_store_total second_store_discount
  have tie_half_price := tie_price / 2
  have hat_half_price := hat_price / 2
  have third_store := jacket_price + (tie_half_price + hat_half_price)
  have fourth_store := watch_price
  have fifth_store := discount belt_price belt_discount + discount scarf_price scarf_discount
  have subtotal := first_store + second_store + third_store + fourth_store + fifth_store
  have after_rewards_points := subtotal - (subtotal * rewards_points_discount / 100)
  have after_gift_card := after_rewards_points - gift_card
  have after_shipping_fees := after_gift_card + (shipping_fee * num_shipping_stores)
  have total := after_shipping_fees + (after_shipping_fees * sales_tax / 100)
  use total / 100 -- to match the monetary value in cents
  sorry

end erwan_spending_l113_113445


namespace min_x_squared_plus_y_squared_l113_113369

theorem min_x_squared_plus_y_squared (x y : ℝ) (h : (x + 5) * (y - 5) = 0) : x^2 + y^2 ≥ 50 := by
  sorry

end min_x_squared_plus_y_squared_l113_113369


namespace inequality_solution_l113_113465

theorem inequality_solution (x : ℝ)
  (h : ∀ x, x^2 + 2 * x + 7 > 0) :
  (x - 3) / (x^2 + 2 * x + 7) ≥ 0 ↔ x ∈ Set.Ici 3 :=
by
  sorry

end inequality_solution_l113_113465


namespace general_term_formula_l113_113693

theorem general_term_formula :
  ∀ n : ℕ, (0 < n) → 
  (-1)^n * (2*n + 1) / (2*n) = ((-1) : ℝ)^n * ((2*n + 1) : ℝ) / (2*n) :=
by {
  sorry
}

end general_term_formula_l113_113693


namespace mary_age_l113_113556

theorem mary_age (x : ℤ) (n m : ℤ) : (x - 2 = n^2) ∧ (x + 2 = m^3) → x = 6 := by
  sorry

end mary_age_l113_113556


namespace cartons_per_box_l113_113973

open Nat

theorem cartons_per_box (cartons packs sticks brown_boxes total_sticks : ℕ) 
  (h1 : cartons * (packs * sticks) * brown_boxes = total_sticks) 
  (h2 : packs = 5) 
  (h3 : sticks = 3) 
  (h4 : brown_boxes = 8) 
  (h5 : total_sticks = 480) :
  cartons = 4 := 
by 
  sorry

end cartons_per_box_l113_113973


namespace dress_total_selling_price_l113_113733

theorem dress_total_selling_price (original_price discount_rate tax_rate : ℝ) 
  (h1 : original_price = 100) (h2 : discount_rate = 0.30) (h3 : tax_rate = 0.15) : 
  (original_price * (1 - discount_rate) * (1 + tax_rate)) = 80.5 := by
  sorry

end dress_total_selling_price_l113_113733


namespace jorges_total_yield_l113_113376

def total_yield (good_acres clay_acres : ℕ) (good_yield clay_yield : ℕ) : ℕ :=
  good_acres * good_yield + clay_acres * clay_yield / 2

theorem jorges_total_yield :
  let acres := 60
  let good_yield_per_acre := 400
  let clay_yield_per_acre := good_yield_per_acre / 2
  let good_acres := 2 * acres / 3
  let clay_acres := acres / 3
  total_yield good_acres clay_acres good_yield_per_acre clay_yield_per_acre = 20000 :=
by
  sorry

end jorges_total_yield_l113_113376


namespace sugar_amount_first_week_l113_113516

theorem sugar_amount_first_week (s : ℕ → ℕ) (h : s 4 = 3) (h_rec : ∀ n, s (n + 1) = s n / 2) : s 1 = 24 :=
by
  sorry

end sugar_amount_first_week_l113_113516


namespace find_number_l113_113405

theorem find_number (x : ℕ) (h : x + 15 = 96) : x = 81 := 
sorry

end find_number_l113_113405


namespace prime_implies_power_of_two_l113_113939

-- Conditions:
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

-- Problem:
theorem prime_implies_power_of_two (n : ℕ) (h : is_prime (2^n + 1)) : ∃ k : ℕ, n = 2^k := sorry

end prime_implies_power_of_two_l113_113939


namespace neither_sufficient_nor_necessary_l113_113003

variable (a b : ℝ)

theorem neither_sufficient_nor_necessary (h1 : 0 < a * b ∧ a * b < 1) : ¬ (b < 1 / a) ∨ ¬ (1 / a < b) := by
  sorry

end neither_sufficient_nor_necessary_l113_113003


namespace man_speed_l113_113058

theorem man_speed (time_in_minutes : ℕ) (distance_in_km : ℕ) 
  (h_time : time_in_minutes = 30) 
  (h_distance : distance_in_km = 5) : 
  (distance_in_km : ℝ) / (time_in_minutes / 60 : ℝ) = 10 :=
by 
  sorry

end man_speed_l113_113058


namespace range_of_a_l113_113755

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → 2 * x > x^2 + a) → a < -8 :=
by
  intro h
  -- Complete the proof by showing that 2x - x^2 has a minimum value of -8 on [-2, 3] and hence proving a < -8.
  sorry

end range_of_a_l113_113755


namespace quadrilateral_side_length_l113_113781

-- Definitions
def inscribed_quadrilateral (a b c d r : ℝ) : Prop :=
  ∃ (O : ℝ) (A B C D : ℝ), 
    O = r ∧ 
    A = a ∧ B = b ∧ C = c ∧ 
    (r^2 + r^2 = (a^2 + b^2) / 2) ∧
    (r^2 + r^2 = (b^2 + c^2) / 2) ∧
    (r^2 + r^2 = (c^2 + d^2) / 2)

-- Theorem statement
theorem quadrilateral_side_length :
  inscribed_quadrilateral 250 250 100 200 250 :=
sorry

end quadrilateral_side_length_l113_113781


namespace largest_prime_divisor_of_sum_of_cyclic_sequence_is_101_l113_113012

-- Define the sequence and its cyclic property
def cyclicSequence (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq (n + 4) = 1000 * (seq n % 10) + 100 * (seq (n + 1) % 10) + 10 * (seq (n + 2) % 10) + (seq (n + 3) % 10)

-- Define the property of T being the sum of the sequence
def sumOfSequence (seq : ℕ → ℕ) (T : ℕ) : Prop :=
  T = seq 0 + seq 1 + seq 2 + seq 3

-- Define the statement that T is always divisible by 101
theorem largest_prime_divisor_of_sum_of_cyclic_sequence_is_101
  (seq : ℕ → ℕ) (T : ℕ)
  (h1 : cyclicSequence seq)
  (h2 : sumOfSequence seq T) :
  (101 ∣ T) := 
sorry

end largest_prime_divisor_of_sum_of_cyclic_sequence_is_101_l113_113012


namespace find_other_number_l113_113605

theorem find_other_number (a b : ℕ) (h_lcm: Nat.lcm a b = 2310) (h_hcf: Nat.gcd a b = 55) (h_a: a = 210) : b = 605 := by
  sorry

end find_other_number_l113_113605


namespace train_crossing_time_l113_113108

theorem train_crossing_time (length_of_train : ℕ) (speed_kmh : ℕ) (speed_ms : ℕ) 
  (conversion_factor : speed_kmh * 1000 / 3600 = speed_ms) 
  (H1 : length_of_train = 180) 
  (H2 : speed_kmh = 72) 
  (H3 : speed_ms = 20) 
  : length_of_train / speed_ms = 9 := by
  sorry

end train_crossing_time_l113_113108


namespace mom_age_when_Jayson_born_l113_113812

theorem mom_age_when_Jayson_born
  (Jayson_age : ℕ)
  (Dad_age : ℕ)
  (Mom_age : ℕ)
  (H1 : Jayson_age = 10)
  (H2 : Dad_age = 4 * Jayson_age)
  (H3 : Mom_age = Dad_age - 2) :
  Mom_age - Jayson_age = 28 := by
  sorry

end mom_age_when_Jayson_born_l113_113812


namespace minimum_buses_required_l113_113768

-- Condition definitions
def one_way_trip_time : ℕ := 50
def stop_time : ℕ := 10
def departure_interval : ℕ := 6

-- Total round trip time
def total_round_trip_time : ℕ := 2 * one_way_trip_time + 2 * stop_time

-- The total number of buses needed to ensure the bus departs every departure_interval minutes
-- from both stations A and B.
theorem minimum_buses_required : 
  (total_round_trip_time / departure_interval) = 20 := by
  sorry

end minimum_buses_required_l113_113768


namespace congruent_triangles_count_l113_113206

open Set

variables (g l : Line) (A B C : Point)

def number_of_congruent_triangles (g l : Line) (A B C : Point) : ℕ :=
  16

theorem congruent_triangles_count (g l : Line) (A B C : Point) :
  number_of_congruent_triangles g l A B C = 16 :=
sorry

end congruent_triangles_count_l113_113206


namespace frac_addition_l113_113253

theorem frac_addition :
  (3 / 5) + (2 / 15) = 11 / 15 :=
sorry

end frac_addition_l113_113253


namespace Nicki_total_miles_run_l113_113734

theorem Nicki_total_miles_run:
  ∀ (miles_per_week_first_half miles_per_week_second_half weeks_in_year weeks_per_half_year : ℕ),
  miles_per_week_first_half = 20 →
  miles_per_week_second_half = 30 →
  weeks_in_year = 52 →
  weeks_per_half_year = weeks_in_year / 2 →
  (miles_per_week_first_half * weeks_per_half_year) + (miles_per_week_second_half * weeks_per_half_year) = 1300 :=
by
  intros miles_per_week_first_half miles_per_week_second_half weeks_in_year weeks_per_half_year
  intros h1 h2 h3 h4
  sorry

end Nicki_total_miles_run_l113_113734


namespace trigonometric_identity_l113_113983

-- Define the conditions and the target statement
theorem trigonometric_identity (α : ℝ) (h1 : Real.tan α = 2) : 
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = 1 / 3 :=
by
  sorry

end trigonometric_identity_l113_113983


namespace valid_numbers_l113_113075

def is_valid_100_digit_number (N N' : ℕ) (k m n : ℕ) (a : ℕ) : Prop :=
  0 ≤ a ∧ a < 100 ∧ 0 ≤ m ∧ m < 10^k ∧ 
  N = m + 10^k * a + 10^(k + 2) * n ∧ 
  N' = m + 10^k * n ∧
  N = 87 * N'

theorem valid_numbers : ∀ (N : ℕ), (∃ N' k m n a, is_valid_100_digit_number N N' k m n a) →
  N = 435 * 10^97 ∨ 
  N = 1305 * 10^96 ∨ 
  N = 2175 * 10^96 ∨ 
  N = 3045 * 10^96 :=
by
  sorry

end valid_numbers_l113_113075


namespace incorrect_statement_l113_113176

noncomputable def a : ℝ × ℝ := (1, -2)
noncomputable def b : ℝ × ℝ := (2, 1)
noncomputable def c : ℝ × ℝ := (-4, -2)

-- Define the incorrect vector statement D
theorem incorrect_statement :
  ¬ ∀ (d : ℝ × ℝ), ∃ (k1 k2 : ℝ), d = (k1 * b.1 + k2 * c.1, k1 * b.2 + k2 * c.2) := sorry

end incorrect_statement_l113_113176


namespace gasoline_tank_capacity_l113_113186

theorem gasoline_tank_capacity
  (y : ℝ)
  (h_initial: y * (5 / 6) - y * (1 / 3) = 20) :
  y = 40 :=
sorry

end gasoline_tank_capacity_l113_113186


namespace tank_capacity_l113_113144

theorem tank_capacity
  (w c : ℝ)
  (h1 : w / c = 1 / 3)
  (h2 : (w + 5) / c = 2 / 5) :
  c = 75 :=
by
  sorry

end tank_capacity_l113_113144


namespace smallest_population_multiple_of_3_l113_113696

theorem smallest_population_multiple_of_3 : 
  ∃ (a : ℕ), ∃ (b c : ℕ), 
  a^2 + 50 = b^2 + 1 ∧ b^2 + 51 = c^2 ∧ 
  (∃ m : ℕ, a * a = 576 ∧ 576 = 3 * m) :=
by
  sorry

end smallest_population_multiple_of_3_l113_113696


namespace surface_area_of_circumscribed_sphere_l113_113538

theorem surface_area_of_circumscribed_sphere :
  let a := 2
  let AD := Real.sqrt (a^2 - (a/2)^2)
  let r := Real.sqrt (1 + 1 + AD^2) / 2
  4 * Real.pi * r^2 = 5 * Real.pi := by
  sorry

end surface_area_of_circumscribed_sphere_l113_113538


namespace find_x_given_y_and_ratio_l113_113242

variable (x y k : ℝ)

theorem find_x_given_y_and_ratio :
  (∀ x y, (5 * x - 6) / (2 * y + 20) = k) →
  (5 * 3 - 6) / (2 * 5 + 20) = k →
  y = 15 →
  x = 21 / 5 :=
by 
  intro h1 h2 hy
  -- proof steps would go here
  sorry

end find_x_given_y_and_ratio_l113_113242


namespace number_of_correct_conclusions_l113_113270

-- Define the conditions as hypotheses
variable {a_n : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}
variable {a_1 : ℝ}
variable {n : ℕ}

-- Arithmetic sequence definition for a_n
def arithmetic_sequence (a_n : ℕ → ℝ) (a_1 d : ℝ) : Prop :=
  ∀ n, a_n n = a_1 + (n - 1) * d

-- Sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (S : ℕ → ℝ) (a_1 d : ℝ) : Prop :=
  ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d)

-- Problem statement
theorem number_of_correct_conclusions 
  (h_seq : arithmetic_sequence a_n a_1 d)
  (h_sum : sum_arithmetic_sequence S a_1 d)
  (h1 : S 5 < S 6)
  (h2 : S 6 = S 7 ∧ S 7 > S 8) :
  ∃ n, n = 3 ∧ 
       (d < 0) ∧ 
       (a_n 7 = 0) ∧ 
       ¬(S 9 = S 5) ∧ 
       (S 6 = S 7 ∧ ∀ m, m > 7 → S m < S 6) := 
sorry

end number_of_correct_conclusions_l113_113270


namespace trim_area_dodecagon_pie_l113_113062

theorem trim_area_dodecagon_pie :
  let d := 8 -- diameter of the pie
  let r := d / 2 -- radius of the pie
  let A_circle := π * r^2 -- area of the circle
  let A_dodecagon := 3 * r^2 -- area of the dodecagon
  let A_trimmed := A_circle - A_dodecagon -- area to be trimmed
  let a := 16 -- coefficient of π in A_trimmed
  let b := 48 -- constant term in A_trimmed
  a + b = 64 := 
by 
  sorry

end trim_area_dodecagon_pie_l113_113062


namespace import_tax_calculation_l113_113352

def import_tax_rate : ℝ := 0.07
def excess_value_threshold : ℝ := 1000
def total_value_item : ℝ := 2610
def correct_import_tax : ℝ := 112.7

theorem import_tax_calculation :
  (total_value_item - excess_value_threshold) * import_tax_rate = correct_import_tax :=
by
  sorry

end import_tax_calculation_l113_113352


namespace simplify_expression_l113_113492

theorem simplify_expression
  (cos : ℝ → ℝ)
  (sin : ℝ → ℝ)
  (cos_double_angle : ∀ x, cos (2 * x) = cos x * cos x - sin x * sin x)
  (sin_double_angle : ∀ x, sin (2 * x) = 2 * sin x * cos x)
  (sin_cofunction : ∀ x, sin (Real.pi / 2 - x) = cos x) :
  (cos 5 * cos 5 - sin 5 * sin 5) / (sin 40 * cos 40) = 2 := by
  sorry

end simplify_expression_l113_113492


namespace equation1_solutions_equation2_solutions_l113_113682

theorem equation1_solutions (x : ℝ) : 3 * x^2 - 6 * x = 0 ↔ (x = 0 ∨ x = 2) := by
  sorry

theorem equation2_solutions (x : ℝ) : x^2 + 4 * x - 1 = 0 ↔ (x = -2 + Real.sqrt 5 ∨ x = -2 - Real.sqrt 5) := by
  sorry

end equation1_solutions_equation2_solutions_l113_113682


namespace num_boys_and_girls_l113_113778

def num_ways_to_select (x : ℕ) := (x * (x - 1) / 2) * (8 - x) * 6

theorem num_boys_and_girls (x : ℕ) (h1 : num_ways_to_select x = 180) :
    x = 5 ∨ x = 6 :=
by
  sorry

end num_boys_and_girls_l113_113778


namespace peach_bun_weight_l113_113999

theorem peach_bun_weight (O triangle : ℕ) 
  (h1 : O = 2 * triangle + 40) 
  (h2 : O + 80 = triangle + 200) : 
  O + triangle = 280 := 
by 
  sorry

end peach_bun_weight_l113_113999


namespace total_cookies_l113_113424

-- Define the number of bags and cookies per bag
def num_bags : Nat := 37
def cookies_per_bag : Nat := 19

-- The theorem stating the total number of cookies
theorem total_cookies : num_bags * cookies_per_bag = 703 := by
  sorry

end total_cookies_l113_113424


namespace neg_exists_le_eq_forall_gt_l113_113993

open Classical

variable {n : ℕ}

theorem neg_exists_le_eq_forall_gt :
  (¬ ∃ (n : ℕ), n > 0 ∧ 2^n ≤ 2 * n + 1) ↔
  (∀ (n : ℕ), n > 0 → 2^n > 2 * n + 1) :=
by 
  sorry

end neg_exists_le_eq_forall_gt_l113_113993


namespace find_b_l113_113476

theorem find_b (b : ℚ) (h : ∃ c : ℚ, (3 * x + c)^2 = 9 * x^2 + 27 * x + b) : b = 81 / 4 := 
sorry

end find_b_l113_113476


namespace total_cost_is_correct_l113_113252

def goldfish_price := 3
def goldfish_quantity := 15
def blue_fish_price := 6
def blue_fish_quantity := 7
def neon_tetra_price := 2
def neon_tetra_quantity := 10
def angelfish_price := 8
def angelfish_quantity := 5

def total_cost := goldfish_quantity * goldfish_price 
                 + blue_fish_quantity * blue_fish_price 
                 + neon_tetra_quantity * neon_tetra_price 
                 + angelfish_quantity * angelfish_price

theorem total_cost_is_correct : total_cost = 147 :=
by
  -- Summary of the proof steps goes here
  sorry

end total_cost_is_correct_l113_113252


namespace total_cost_of_coat_l113_113559

def original_price : ℝ := 150
def sale_discount : ℝ := 0.25
def additional_discount : ℝ := 10
def sales_tax : ℝ := 0.10

theorem total_cost_of_coat :
  let sale_price := original_price * (1 - sale_discount)
  let price_after_discount := sale_price - additional_discount
  let final_price := price_after_discount * (1 + sales_tax)
  final_price = 112.75 :=
by
  -- sorry for the actual proof
  sorry

end total_cost_of_coat_l113_113559


namespace minimum_trucks_on_lot_l113_113327

variable (total_trucks : ℕ) (rented_percentage : ℝ) (max_rented_trucks : ℕ)
variable (initial_trucks : total_trucks = 24) (returned_percentage : rented_percentage = 0.5) (max_trucks_rented : max_rented_trucks = 24)

theorem minimum_trucks_on_lot (total_trucks : ℕ) (rented_percentage : ℝ) (max_rented_trucks : ℕ)
  (initial_trucks : total_trucks = 24) (returned_percentage : rented_percentage = 0.5) (max_trucks_rented : max_rented_trucks = 24) :
  max_rented_trucks / 2 = 12 :=
by sorry

end minimum_trucks_on_lot_l113_113327


namespace number_of_boys_l113_113230

-- Definitions reflecting the conditions
def total_students := 1200
def sample_size := 200
def extra_boys := 10

-- Main problem statement
theorem number_of_boys (B G b g : ℕ) 
  (h_total_students : B + G = total_students)
  (h_sample_size : b + g = sample_size)
  (h_extra_boys : b = g + extra_boys)
  (h_stratified : b * G = g * B) :
  B = 660 :=
by sorry

end number_of_boys_l113_113230


namespace prod_div_sum_le_square_l113_113354

theorem prod_div_sum_le_square (m n : ℕ) (h : (m * n) ∣ (m + n)) : m + n ≤ n^2 := sorry

end prod_div_sum_le_square_l113_113354


namespace arithmetic_sequence_sum_l113_113493

-- Let {a_n} be an arithmetic sequence.
-- Define Sn as the sum of the first n terms.
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d

def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n * (a 0 + a (n-1))) / 2

theorem arithmetic_sequence_sum
  (a : ℕ → ℤ)
  (h_arith : is_arithmetic_sequence a)
  (h_condition : 2 * a 6 = a 7 + 5) :
  S a 11 = 55 :=
sorry

end arithmetic_sequence_sum_l113_113493


namespace distances_equal_l113_113294

noncomputable def distance_from_point_to_line (x y m : ℝ) : ℝ :=
  |m * x + y + 3| / Real.sqrt (m^2 + 1)

theorem distances_equal (m : ℝ) :
  distance_from_point_to_line 3 2 m = distance_from_point_to_line (-1) 4 m ↔
  (m = 1 / 2 ∨ m = -6) := 
sorry

end distances_equal_l113_113294


namespace inequality_proof_l113_113806

theorem inequality_proof 
  {a b c : ℝ}
  (ha : 0 ≤ a)
  (hb : 0 ≤ b)
  (hc : 0 ≤ c)
  (h1 : a^2 ≤ b^2 + c^2)
  (h2 : b^2 ≤ c^2 + a^2)
  (h3 : c^2 ≤ a^2 + b^2) :
  (a + b + c) * (a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) ≥ 4 * (a^6 + b^6 + c^6) :=
sorry

end inequality_proof_l113_113806


namespace kelly_grade_correct_l113_113437

variable (Jenny Jason Bob Kelly : ℕ)

def jenny_grade : ℕ := 95
def jason_grade := jenny_grade - 25
def bob_grade := jason_grade / 2
def kelly_grade := bob_grade + (bob_grade / 5)  -- 20% of Bob's grade is (Bob's grade * 0.20), which is the same as (Bob's grade / 5)

theorem kelly_grade_correct : kelly_grade = 42 :=
by
  sorry

end kelly_grade_correct_l113_113437


namespace mean_of_remaining_three_numbers_l113_113256

variable {a b c : ℝ}

theorem mean_of_remaining_three_numbers (h1 : (a + b + c + 103) / 4 = 90) : (a + b + c) / 3 = 85.7 :=
by
  -- Sorry placeholder for the proof
  sorry

end mean_of_remaining_three_numbers_l113_113256


namespace container_volume_ratio_l113_113454

variables (A B C : ℝ)

theorem container_volume_ratio (h1 : (2 / 3) * A = (1 / 2) * B) (h2 : (1 / 2) * B = (3 / 5) * C) :
  A / C = 6 / 5 :=
sorry

end container_volume_ratio_l113_113454


namespace nat_number_solution_odd_l113_113869

theorem nat_number_solution_odd (x y z : ℕ) (h : x + y + z = 100) : 
  ∃ P : ℕ, P = 49 ∧ P % 2 = 1 := 
sorry

end nat_number_solution_odd_l113_113869


namespace range_of_a_l113_113016

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, a * x^2 + a * x - 1 ≤ 0) : -4 ≤ a ∧ a ≤ 0 := 
sorry

end range_of_a_l113_113016


namespace kim_min_pours_l113_113485

-- Define the initial conditions
def initial_volume (V : ℝ) : ℝ := V
def pour (V : ℝ) : ℝ := 0.9 * V

-- Define the remaining volume after n pours
def remaining_volume (V : ℝ) (n : ℕ) : ℝ := V * (0.9)^n

-- State the problem: After 7 pours, the remaining volume is less than half the initial volume
theorem kim_min_pours (V : ℝ) (hV : V > 0) : remaining_volume V 7 < V / 2 :=
by
  -- Because the proof is not required, we use sorry
  sorry

end kim_min_pours_l113_113485


namespace diff_squares_of_roots_l113_113462

theorem diff_squares_of_roots : ∀ α β : ℝ, (α * β = 6) ∧ (α + β = 5) -> (α - β)^2 = 1 := by
  sorry

end diff_squares_of_roots_l113_113462


namespace impossible_four_teams_tie_possible_three_teams_tie_l113_113344

-- Definitions for the conditions
def num_teams : ℕ := 4
def num_matches : ℕ := (num_teams * (num_teams - 1)) / 2
def total_possible_outcomes : ℕ := 2^num_matches
def winning_rate : ℚ := 1 / 2

-- Problem 1: It is impossible for exactly four teams to tie for first place.
theorem impossible_four_teams_tie :
  ¬ ∃ (score : ℕ), (∀ (team : ℕ) (h : team < num_teams), team = score ∧
                     (num_teams * score = num_matches / 2 ∧
                      num_teams * score + num_matches / 2 = num_matches)) := sorry

-- Problem 2: It is possible for exactly three teams to tie for first place.
theorem possible_three_teams_tie :
  ∃ (score : ℕ), (∃ (teamA teamB teamC teamD : ℕ),
  (teamA < num_teams ∧ teamB < num_teams ∧ teamC < num_teams ∧ teamD <num_teams ∧ teamA ≠ teamB ∧ teamA ≠ teamC ∧ teamA ≠ teamD ∧ 
  teamB ≠ teamC ∧ teamB ≠ teamD ∧ teamC ≠ teamD)) ∧
  (teamA = score ∧ teamB = score ∧ teamC = score ∧ teamD = 0) := sorry

end impossible_four_teams_tie_possible_three_teams_tie_l113_113344


namespace save_water_negate_l113_113450

/-- If saving 30cm^3 of water is denoted as +30cm^3, then wasting 10cm^3 of water is denoted as -10cm^3. -/
theorem save_water_negate :
  (∀ (save_waste : ℤ → ℤ), save_waste 30 = 30 → save_waste (-10) = -10) :=
by
  sorry

end save_water_negate_l113_113450


namespace cyclist_average_speed_l113_113018

noncomputable def total_distance : ℝ := 10 + 5 + 15 + 20 + 30
noncomputable def time_first_segment : ℝ := 10 / 12
noncomputable def time_second_segment : ℝ := 5 / 6
noncomputable def time_third_segment : ℝ := 15 / 16
noncomputable def time_fourth_segment : ℝ := 20 / 14
noncomputable def time_fifth_segment : ℝ := 30 / 20

noncomputable def total_time : ℝ := time_first_segment + time_second_segment + time_third_segment + time_fourth_segment + time_fifth_segment

noncomputable def average_speed : ℝ := total_distance / total_time

theorem cyclist_average_speed : average_speed = 12.93 := by
  sorry

end cyclist_average_speed_l113_113018


namespace find_number_l113_113159

axiom condition_one (x y : ℕ) : 10 * x + y = 3 * (x + y) + 7
axiom condition_two (x y : ℕ) : x^2 + y^2 - x * y = 10 * x + y

theorem find_number : 
  ∃ (x y : ℕ), (10 * x + y = 37) → (10 * x + y = 3 * (x + y) + 7 ∧ x^2 + y^2 - x * y = 10 * x + y) := 
by 
  sorry

end find_number_l113_113159


namespace farm_horses_cows_ratio_l113_113885

variable (x y : ℕ)  -- x is the base variable related to the initial counts, y is the number of horses sold (and cows bought)

theorem farm_horses_cows_ratio (h1 : 4 * x / x = 4)
    (h2 : 13 * (x + y) = 7 * (4 * x - y))
    (h3 : 4 * x - y = (x + y) + 30) :
    y = 15 := sorry

end farm_horses_cows_ratio_l113_113885


namespace oliver_learning_vowels_l113_113375

theorem oliver_learning_vowels : 
  let learn := 5
  let rest_days (n : Nat) := n
  let total_days :=
    (learn + rest_days 1) + -- For 'A'
    (learn + rest_days 2) + -- For 'E'
    (learn + rest_days 3) + -- For 'I'
    (learn + rest_days 4) + -- For 'O'
    (rest_days 5 + learn)  -- For 'U' and 'Y'
  total_days = 40 :=
by
  sorry

end oliver_learning_vowels_l113_113375


namespace octopus_shoes_needed_l113_113836

-- Defining the basic context: number of legs and current shod legs
def num_legs : ℕ := 8

-- Conditions based on the number of already shod legs for each member
def father_shod_legs : ℕ := num_legs / 2       -- Father-octopus has half of his legs shod
def mother_shod_legs : ℕ := 3                  -- Mother-octopus has 3 legs shod
def son_shod_legs : ℕ := 6                     -- Each son-octopus has 6 legs shod
def num_sons : ℕ := 2                          -- There are 2 sons

-- Calculate unshod legs for each 
def father_unshod_legs : ℕ := num_legs - father_shod_legs
def mother_unshod_legs : ℕ := num_legs - mother_shod_legs
def son_unshod_legs : ℕ := num_legs - son_shod_legs

-- Aggregate the total shoes needed based on unshod legs
def total_shoes_needed : ℕ :=
  father_unshod_legs + 
  mother_unshod_legs + 
  (son_unshod_legs * num_sons)

-- The theorem to prove
theorem octopus_shoes_needed : total_shoes_needed = 13 := 
  by 
    sorry

end octopus_shoes_needed_l113_113836


namespace find_m_l113_113044

theorem find_m : ∃ m : ℝ, 
  (∀ x : ℝ, 0 < x → (m^2 - m - 5) * x^(m - 1) = (m^2 - m - 5) * x^(m - 1) ∧ 
  (m^2 - m - 5) * (m - 1) * x^(m - 2) > 0) → m = 3 :=
by
  sorry

end find_m_l113_113044


namespace quarters_for_chips_l113_113740

def total_quarters : ℕ := 16
def quarters_for_soda : ℕ := 12

theorem quarters_for_chips : (total_quarters - quarters_for_soda) = 4 :=
  by 
    sorry

end quarters_for_chips_l113_113740


namespace total_highlighters_is_49_l113_113009

-- Define the number of highlighters of each color
def pink_highlighters : Nat := 15
def yellow_highlighters : Nat := 12
def blue_highlighters : Nat := 9
def green_highlighters : Nat := 7
def purple_highlighters : Nat := 6

-- Define the total number of highlighters
def total_highlighters : Nat := pink_highlighters + yellow_highlighters + blue_highlighters + green_highlighters + purple_highlighters

-- Statement that the total number of highlighters should be 49
theorem total_highlighters_is_49 : total_highlighters = 49 := by
  sorry

end total_highlighters_is_49_l113_113009


namespace coeff_x4_in_expansion_correct_l113_113099

noncomputable def coeff_x4_in_expansion (f g : ℕ → ℤ) := 
  ∀ (c : ℤ), c = 80 → f 4 + g 1 * g 3 = c

-- Definitions of the individual polynomials
def poly1 (x : ℤ) : ℤ := 4 * x^2 - 2 * x + 1
def poly2 (x : ℤ) : ℤ := 2 * x + 1

-- Expanded form coefficients
def coeff_poly1 : ℕ → ℤ
  | 0       => 1
  | 1       => -2
  | 2       => 4
  | _       => 0

def coeff_poly2_pow4 : ℕ → ℤ
  | 0       => 1
  | 1       => 8
  | 2       => 24
  | 3       => 32
  | 4       => 16
  | _       => 0

-- The theorem we want to prove
theorem coeff_x4_in_expansion_correct :
  coeff_x4_in_expansion coeff_poly1 coeff_poly2_pow4 := 
by
  sorry

end coeff_x4_in_expansion_correct_l113_113099


namespace polygon_perimeter_greater_than_2_l113_113001

-- Definition of the conditions
variable (polygon : Set (ℝ × ℝ))
variable (A B : ℝ × ℝ)
variable (P : ℝ)

axiom point_in_polygon (p : ℝ × ℝ) : p ∈ polygon
axiom A_in_polygon : A ∈ polygon
axiom B_in_polygon : B ∈ polygon
axiom path_length_condition (γ : ℝ → ℝ × ℝ) (γ_in_polygon : ∀ t, γ t ∈ polygon) (hA : γ 0 = A) (hB : γ 1 = B) : ∀ t₁ t₂, 0 ≤ t₁ → t₁ ≤ t₂ → t₂ ≤ 1 → dist (γ t₁) (γ t₂) > 1

-- Statement to prove
theorem polygon_perimeter_greater_than_2 : P > 2 :=
sorry

end polygon_perimeter_greater_than_2_l113_113001


namespace sin_cos_identity_second_quadrant_l113_113864

open Real

theorem sin_cos_identity_second_quadrant (α : ℝ) (hcos : cos α < 0) (hsin : sin α > 0) :
  (sin α / cos α) * sqrt ((1 / (sin α)^2) - 1) = -1 :=
sorry

end sin_cos_identity_second_quadrant_l113_113864


namespace probability_not_below_x_axis_half_l113_113067

-- Define the vertices of the parallelogram
def P : (ℝ × ℝ) := (4, 4)
def Q : (ℝ × ℝ) := (-2, -2)
def R : (ℝ × ℝ) := (-8, -2)
def S : (ℝ × ℝ) := (-2, 4)

-- Define a predicate for points within the parallelogram
def in_parallelogram (A B C D : ℝ × ℝ) (p : ℝ × ℝ) : Prop := sorry

-- Define the area function
def area_of_parallelogram (A B C D : ℝ × ℝ) : ℝ := sorry

noncomputable def probability_not_below_x_axis (A B C D : ℝ × ℝ) : ℝ :=
  let total_area := area_of_parallelogram A B C D
  let area_above_x_axis := area_of_parallelogram (0, 0) D A (0, 0) / 2
  area_above_x_axis / total_area

theorem probability_not_below_x_axis_half :
  probability_not_below_x_axis P Q R S = 1 / 2 :=
sorry

end probability_not_below_x_axis_half_l113_113067


namespace minimum_mn_l113_113022

noncomputable def f (x : ℝ) (n m : ℝ) : ℝ := Real.log x - n * x + Real.log m + 1

noncomputable def f' (x : ℝ) (n : ℝ) : ℝ := 1/x - n

theorem minimum_mn (m n x_0 : ℝ) (h_m : m > 1) (h_tangent : 2*x_0 - (f x_0 n m) + 1 = 0) :
  mn = e * ((1/x_0 - 1) ^ 2 - 1) :=
sorry

end minimum_mn_l113_113022


namespace range_of_t_l113_113150

theorem range_of_t (a b : ℝ) 
  (h1 : a^2 + a * b + b^2 = 1) 
  (h2 : ∃ t : ℝ, t = a * b - a^2 - b^2) : 
  ∀ t, t = a * b - a^2 - b^2 → -3 ≤ t ∧ t ≤ -1/3 :=
by sorry

end range_of_t_l113_113150


namespace probability_at_least_one_hit_l113_113950

-- Define probabilities of each shooter hitting the target
def P_A : ℚ := 1 / 2
def P_B : ℚ := 1 / 3
def P_C : ℚ := 1 / 4

-- Define the complementary probabilities (each shooter misses the target)
def P_A_miss : ℚ := 1 - P_A
def P_B_miss : ℚ := 1 - P_B
def P_C_miss : ℚ := 1 - P_C

-- Calculate the probability of all shooters missing the target
def P_all_miss : ℚ := P_A_miss * P_B_miss * P_C_miss

-- Calculate the probability of at least one shooter hitting the target
def P_at_least_one_hit : ℚ := 1 - P_all_miss

-- The theorem to be proved
theorem probability_at_least_one_hit : 
  P_at_least_one_hit = 3 / 4 := 
by sorry

end probability_at_least_one_hit_l113_113950


namespace range_of_sine_l113_113680

theorem range_of_sine {x : ℝ} (h1 : 0 ≤ x) (h2 : x ≤ 2 * Real.pi) (h3 : Real.sin x ≥ Real.sqrt 2 / 2) :
  Real.pi / 4 ≤ x ∧ x ≤ 3 * Real.pi / 4 :=
by
  sorry

end range_of_sine_l113_113680


namespace expected_lifetime_flashlight_l113_113399

noncomputable def E (X : ℝ) : ℝ := sorry -- Define E as the expectation operator

variables (ξ η : ℝ) -- Define ξ and η as random variables representing lifetimes of blue and red bulbs
variable (h_exi : E ξ = 2) -- Given condition E ξ = 2

theorem expected_lifetime_flashlight (h_min : ∀ x y : ℝ, min x y ≤ x) :
  E (min ξ η) ≤ 2 :=
by
  sorry

end expected_lifetime_flashlight_l113_113399


namespace partitions_equiv_l113_113918

-- Definition of partitions into distinct integers
def a (n : ℕ) : ℕ := sorry  -- Placeholder for the actual definition or count function

-- Definition of partitions into odd integers
def b (n : ℕ) : ℕ := sorry  -- Placeholder for the actual definition or count function

-- Theorem stating that the number of partitions into distinct integers equals the number of partitions into odd integers
theorem partitions_equiv (n : ℕ) : a n = b n :=
sorry

end partitions_equiv_l113_113918


namespace balloon_permutations_l113_113747

theorem balloon_permutations : 
  (Nat.factorial 7 / 
  ((Nat.factorial 1) * 
  (Nat.factorial 1) * 
  (Nat.factorial 2) * 
  (Nat.factorial 2) * 
  (Nat.factorial 1))) = 1260 := by
  sorry

end balloon_permutations_l113_113747


namespace value_a7_l113_113813

variables {a : ℕ → ℝ}

-- Condition 1: Arithmetic sequence where each term is non-zero
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variable (h1 : arithmetic_sequence a)
-- Condition 2: 2a_3 - a_1^2 + 2a_11 = 0
variable (h2 : 2 * a 3 - (a 1)^2 + 2 * a 11 = 0)
-- Condition 3: a_3 + a_11 = 2a_7
variable (h3 : a 3 + a 11 = 2 * a 7)

theorem value_a7 : a 7 = 4 := by
  sorry

end value_a7_l113_113813


namespace A_investment_is_100_l113_113602

-- Definitions directly from the conditions in a)
def A_investment (X : ℝ) := X * 12
def B_investment : ℝ := 200 * 6
def total_profit : ℝ := 100
def A_share_of_profit : ℝ := 50

-- Prove that given these conditions, A's initial investment X is 100
theorem A_investment_is_100 (X : ℝ) (h : A_share_of_profit / total_profit = A_investment X / B_investment) : X = 100 :=
by
  sorry

end A_investment_is_100_l113_113602


namespace relay_race_total_time_is_correct_l113_113235

-- Define the time taken by each runner
def time_Ainslee : ℕ := 72
def time_Bridget : ℕ := (10 * time_Ainslee) / 9
def time_Cecilia : ℕ := (3 * time_Bridget) / 4
def time_Dana : ℕ := (5 * time_Cecilia) / 6

-- Define the total time and convert to minutes and seconds
def total_time_seconds : ℕ := time_Ainslee + time_Bridget + time_Cecilia + time_Dana
def total_time_minutes := total_time_seconds / 60
def total_time_remainder := total_time_seconds % 60

theorem relay_race_total_time_is_correct :
  total_time_minutes = 4 ∧ total_time_remainder = 22 :=
by
  -- All intermediate values can be calculated using the definitions
  -- provided above correctly.
  sorry

end relay_race_total_time_is_correct_l113_113235


namespace megatek_manufacturing_percentage_l113_113142

theorem megatek_manufacturing_percentage (total_degrees sector_degrees : ℝ)
    (h_circle: total_degrees = 360)
    (h_sector: sector_degrees = 252) :
    (sector_degrees / total_degrees) * 100 = 70 :=
by
  sorry

end megatek_manufacturing_percentage_l113_113142


namespace train_length_l113_113098

theorem train_length 
  (t1 t2 : ℕ) 
  (d2 : ℕ) 
  (V L : ℝ) 
  (h1 : t1 = 11)
  (h2 : t2 = 22)
  (h3 : d2 = 120)
  (h4 : V = L / t1)
  (h5 : V = (L + d2) / t2) : 
  L = 120 := 
by 
  sorry

end train_length_l113_113098


namespace ratio_of_pens_to_pencils_l113_113293

/-
The store ordered pens and pencils:
1. The number of pens was some multiple of the number of pencils plus 300.
2. The cost of a pen was $5.
3. The cost of a pencil was $4.
4. The store ordered 15 boxes, each having 80 pencils.
5. The store paid a total of $18,300 for the stationery.
Prove that the ratio of the number of pens to the number of pencils is 2.25.
-/

variables (e p k : ℕ)
variables (cost_pen : ℕ := 5) (cost_pencil : ℕ := 4) (total_cost : ℕ := 18300)

def number_of_pencils := 15 * 80

def number_of_pens := p -- to be defined in terms of e and k

def total_cost_pens := p * cost_pen
def total_cost_pencils := e * cost_pencil

theorem ratio_of_pens_to_pencils :
  p = k * e + 300 →
  e = 1200 →
  5 * p + 4 * e = 18300 →
  (p : ℚ) / e = 2.25 :=
by
  intros hp he htotal
  sorry

end ratio_of_pens_to_pencils_l113_113293


namespace triangle_area_l113_113300

open Real

-- Define the angles A and C, side a, and state the goal as proving the area
theorem triangle_area (A C : ℝ) (a : ℝ) (hA : A = 30 * (π / 180)) (hC : C = 45 * (π / 180)) (ha : a = 2) : 
  (1 / 2) * ((sqrt 6 + sqrt 2) * (2 * sqrt 2) * sin (30 * (π / 180))) = sqrt 3 + 1 := 
by
  sorry

end triangle_area_l113_113300


namespace min_cubes_required_l113_113328

theorem min_cubes_required (length width height volume_cube : ℝ) 
  (h_length : length = 14.5) 
  (h_width : width = 17.8) 
  (h_height : height = 7.2) 
  (h_volume_cube : volume_cube = 3) : 
  ⌈(length * width * height) / volume_cube⌉ = 624 := sorry

end min_cubes_required_l113_113328


namespace find_number_l113_113790

-- Define the number 40 and the percentage 90.
def num : ℝ := 40
def percent : ℝ := 0.9

-- Define the condition that 4/5 of x is smaller than 90% of 40 by 16
def condition (x : ℝ) : Prop := (4/5 : ℝ) * x = percent * num - 16

-- Proof statement in Lean 4
theorem find_number : ∃ x : ℝ, condition x ∧ x = 25 :=
by 
  use 25
  unfold condition
  norm_num
  sorry

end find_number_l113_113790


namespace bakery_gives_away_30_doughnuts_at_end_of_day_l113_113388

def boxes_per_day (total_doughnuts doughnuts_per_box : ℕ) : ℕ :=
  total_doughnuts / doughnuts_per_box

def leftover_boxes (total_boxes sold_boxes : ℕ) : ℕ :=
  total_boxes - sold_boxes

def doughnuts_given_away (leftover_boxes doughnuts_per_box : ℕ) : ℕ :=
  leftover_boxes * doughnuts_per_box

theorem bakery_gives_away_30_doughnuts_at_end_of_day 
  (total_doughnuts doughnuts_per_box sold_boxes : ℕ) 
  (H1 : total_doughnuts = 300) (H2 : doughnuts_per_box = 10) (H3 : sold_boxes = 27) : 
  doughnuts_given_away (leftover_boxes (boxes_per_day total_doughnuts doughnuts_per_box) sold_boxes) doughnuts_per_box = 30 :=
by
  sorry

end bakery_gives_away_30_doughnuts_at_end_of_day_l113_113388


namespace older_brother_pocket_money_l113_113679

-- Definitions of the conditions
axiom sum_of_pocket_money (O Y : ℕ) : O + Y = 12000
axiom older_brother_more (O Y : ℕ) : O = Y + 1000

-- The statement to prove
theorem older_brother_pocket_money (O Y : ℕ) (h1 : O + Y = 12000) (h2 : O = Y + 1000) : O = 6500 :=
by
  exact sorry  -- Placeholder for the proof

end older_brother_pocket_money_l113_113679


namespace Amanda_car_round_trip_time_l113_113761

theorem Amanda_car_round_trip_time (bus_time : ℕ) (car_reduction : ℕ) (bus_one_way_trip : bus_time = 40) (car_time_reduction : car_reduction = 5) : 
  (2 * (bus_time - car_reduction)) = 70 := 
by
  sorry

end Amanda_car_round_trip_time_l113_113761


namespace handshakes_at_gathering_l113_113677

def total_handshakes (num_couples : ℕ) (exceptions : ℕ) : ℕ :=
  let num_people := 2 * num_couples
  let handshakes_per_person := num_people - exceptions - 1
  num_people * handshakes_per_person / 2

theorem handshakes_at_gathering : total_handshakes 6 2 = 54 := by
  sorry

end handshakes_at_gathering_l113_113677


namespace inequality_range_of_a_l113_113585

theorem inequality_range_of_a (a : ℝ) :
  (∀ x y : ℝ, (1 ≤ x ∧ x ≤ 2) ∧ (1 ≤ y ∧ y ≤ 3) → 2 * x^2 - a * x * y + y^2 ≥ 0) →
  a ≤ 2 * Real.sqrt 2 :=
by
  intros h
  sorry

end inequality_range_of_a_l113_113585


namespace max_of_2x_plus_y_l113_113258

theorem max_of_2x_plus_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x + y / 2 + 1 / x + 8 / y = 10) : 
  2 * x + y ≤ 18 :=
sorry

end max_of_2x_plus_y_l113_113258


namespace power_of_product_l113_113845

variable (a b : ℝ) (m : ℕ)
theorem power_of_product (h : 0 < m) : (a * b)^m = a^m * b^m :=
sorry

end power_of_product_l113_113845


namespace simplify_expression_l113_113292

variable (x y : ℝ)

theorem simplify_expression : 
  (3 * x^2 * y - 2 * x * y^2) - (x * y^2 - 2 * x^2 * y) - 2 * (-3 * x^2 * y - x * y^2) = 26 := by
  -- Given conditions
  let x := -1
  let y := 2
  -- Proof to be provided
  sorry

end simplify_expression_l113_113292


namespace min_value_x_plus_y_l113_113756

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 19 / x + 98 / y = 1) : 
  x + y ≥ 117 + 14 * Real.sqrt 38 :=
  sorry

end min_value_x_plus_y_l113_113756


namespace recurring_fraction_difference_l113_113557

theorem recurring_fraction_difference :
  let x := (36 / 99 : ℚ)
  let y := (36 / 100 : ℚ)
  x - y = (1 / 275 : ℚ) :=
by
  sorry

end recurring_fraction_difference_l113_113557


namespace adult_ticket_cost_l113_113171

-- Definitions based on given conditions.
def children_ticket_cost : ℝ := 7.5
def total_bill : ℝ := 138
def total_tickets : ℕ := 12
def additional_children_tickets : ℕ := 8

-- Proof statement: Prove the cost of each adult ticket.
theorem adult_ticket_cost (x : ℕ) (A : ℝ)
  (h1 : x + (x + additional_children_tickets) = total_tickets)
  (h2 : x * A + (x + additional_children_tickets) * children_ticket_cost = total_bill) :
  A = 31.50 :=
  sorry

end adult_ticket_cost_l113_113171


namespace triangle_DOE_area_l113_113570

theorem triangle_DOE_area
  (area_ABC : ℝ)
  (DO : ℝ) (OB : ℝ)
  (EO : ℝ) (OA : ℝ)
  (h_area_ABC : area_ABC = 1)
  (h_DO_OB : DO / OB = 1 / 3)
  (h_EO_OA : EO / OA = 4 / 5)
  : (1 / 4) * (4 / 9) * area_ABC = 11 / 135 := 
by 
  sorry

end triangle_DOE_area_l113_113570


namespace average_donation_l113_113579

theorem average_donation (d : ℕ) (n : ℕ) (r : ℕ) (average_donation : ℕ) 
  (h1 : d = 10)   -- $10 donated by customers
  (h2 : r = 2)    -- $2 donated by restaurant
  (h3 : n = 40)   -- number of customers
  (h4 : (r : ℕ) * n / d = 24) -- total donation by restaurant is $24
  : average_donation = 3 := 
by
  sorry

end average_donation_l113_113579


namespace find_x_plus_y_l113_113072

theorem find_x_plus_y (x y : ℚ) (h1 : 3 * x - 4 * y = 18) (h2 : x + 3 * y = -1) :
  x + y = 29 / 13 :=
sorry

end find_x_plus_y_l113_113072


namespace potato_gun_distance_l113_113373

noncomputable def length_of_football_field_in_yards : ℕ := 200
noncomputable def conversion_factor_yards_to_feet : ℕ := 3
noncomputable def length_of_football_field_in_feet : ℕ := length_of_football_field_in_yards * conversion_factor_yards_to_feet

noncomputable def dog_running_speed : ℕ := 400
noncomputable def time_for_dog_to_fetch_potato : ℕ := 9
noncomputable def total_distance_dog_runs : ℕ := dog_running_speed * time_for_dog_to_fetch_potato

noncomputable def actual_distance_to_potato : ℕ := total_distance_dog_runs / 2

noncomputable def distance_in_football_fields : ℕ := actual_distance_to_potato / length_of_football_field_in_feet

theorem potato_gun_distance :
  distance_in_football_fields = 3 :=
by
  sorry

end potato_gun_distance_l113_113373


namespace range_of_y_for_x_gt_2_l113_113034

theorem range_of_y_for_x_gt_2 :
  (∀ x : ℝ, x > 2 → 0 < 2 / x ∧ 2 / x < 1) :=
by 
  -- Proof is omitted
  sorry

end range_of_y_for_x_gt_2_l113_113034


namespace inequality_solution_set_nonempty_range_l113_113498

theorem inequality_solution_set_nonempty_range (a : ℝ) :
  (∃ x : ℝ, (a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0) ↔ (a ≤ -2 ∨ a ≥ 6 / 5) :=
by
  -- Proof is omitted
  sorry

end inequality_solution_set_nonempty_range_l113_113498


namespace rick_total_clothes_ironed_l113_113697

def rick_ironing_pieces
  (shirts_per_hour : ℕ)
  (pants_per_hour : ℕ)
  (hours_shirts : ℕ)
  (hours_pants : ℕ) : ℕ :=
  (shirts_per_hour * hours_shirts) + (pants_per_hour * hours_pants)

theorem rick_total_clothes_ironed :
  rick_ironing_pieces 4 3 3 5 = 27 :=
by
  sorry

end rick_total_clothes_ironed_l113_113697


namespace pens_bought_l113_113069

-- Define the given conditions
def num_notebooks : ℕ := 10
def cost_per_pen : ℕ := 2
def total_paid : ℕ := 30
def cost_per_notebook : ℕ := 0  -- Assumption that notebooks are free

-- Converted condition that 10N + 2P = 30 and N = 0
def equation (N P : ℕ) : Prop := (10 * N + 2 * P = total_paid)

-- Statement to prove that if notebooks are free, 15 pens were bought
theorem pens_bought (N : ℕ) (P : ℕ) (hN : N = cost_per_notebook) (h : equation N P) : P = 15 :=
by sorry

end pens_bought_l113_113069


namespace whale_consumption_l113_113826

-- Define the conditions
def first_hour_consumption (x : ℕ) := x
def second_hour_consumption (x : ℕ) := x + 3
def third_hour_consumption (x : ℕ) := x + 6
def fourth_hour_consumption (x : ℕ) := x + 9
def fifth_hour_consumption (x : ℕ) := x + 12
def sixth_hour_consumption (x : ℕ) := x + 15
def seventh_hour_consumption (x : ℕ) := x + 18
def eighth_hour_consumption (x : ℕ) := x + 21
def ninth_hour_consumption (x : ℕ) := x + 24

def total_consumed (x : ℕ) := 
  first_hour_consumption x + 
  second_hour_consumption x + 
  third_hour_consumption x + 
  fourth_hour_consumption x + 
  fifth_hour_consumption x + 
  sixth_hour_consumption x + 
  seventh_hour_consumption x + 
  eighth_hour_consumption x + 
  ninth_hour_consumption x

-- Prove that the total sum consumed equals 540
theorem whale_consumption : ∃ x : ℕ, total_consumed x = 540 ∧ sixth_hour_consumption x = 63 :=
by
  sorry

end whale_consumption_l113_113826


namespace total_jewelry_pieces_l113_113726

noncomputable def initial_necklaces : ℕ := 10
noncomputable def initial_earrings : ℕ := 15
noncomputable def bought_necklaces : ℕ := 10
noncomputable def bought_earrings : ℕ := 2 * initial_earrings / 3
noncomputable def extra_earrings_from_mother : ℕ := bought_earrings / 5

theorem total_jewelry_pieces : initial_necklaces + bought_necklaces + initial_earrings + bought_earrings + extra_earrings_from_mother = 47 :=
by
  have total_necklaces : ℕ := initial_necklaces + bought_necklaces
  have total_earrings : ℕ := initial_earrings + bought_earrings + extra_earrings_from_mother
  have total_jewelry : ℕ := total_necklaces + total_earrings
  exact Eq.refl 47
  
#check total_jewelry_pieces -- Check if the type is correct

end total_jewelry_pieces_l113_113726


namespace no_fixed_point_range_of_a_fixed_point_in_interval_l113_113702

-- Problem (1)
theorem no_fixed_point_range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + a * x + a ≠ x) →
  3 - 2 * Real.sqrt 2 < a ∧ a < 3 + 2 * Real.sqrt 2 :=
by
  sorry

-- Problem (2)
theorem fixed_point_in_interval (f : ℝ → ℝ) (n : ℤ) :
  (∀ x : ℝ, f x = -Real.log x + 3) →
  (∃ x₀ : ℝ, f x₀ = x₀ ∧ n ≤ x₀ ∧ x₀ < n + 1) →
  n = 2 :=
by
  sorry

end no_fixed_point_range_of_a_fixed_point_in_interval_l113_113702


namespace calculate_fraction_l113_113995

theorem calculate_fraction :
  (2019 + 1981)^2 / 121 = 132231 := 
  sorry

end calculate_fraction_l113_113995


namespace avg_speed_last_40_min_is_70_l113_113572

noncomputable def avg_speed_last_interval
  (total_distance : ℝ) (total_time : ℝ)
  (speed_first_40_min : ℝ) (time_first_40_min : ℝ)
  (speed_second_40_min : ℝ) (time_second_40_min : ℝ) : ℝ :=
  let time_last_40_min := total_time - (time_first_40_min + time_second_40_min)
  let distance_first_40_min := speed_first_40_min * time_first_40_min
  let distance_second_40_min := speed_second_40_min * time_second_40_min
  let distance_last_40_min := total_distance - (distance_first_40_min + distance_second_40_min)
  distance_last_40_min / time_last_40_min

theorem avg_speed_last_40_min_is_70
  (h_total_distance : total_distance = 120)
  (h_total_time : total_time = 2)
  (h_speed_first_40_min : speed_first_40_min = 50)
  (h_time_first_40_min : time_first_40_min = 2 / 3)
  (h_speed_second_40_min : speed_second_40_min = 60)
  (h_time_second_40_min : time_second_40_min = 2 / 3) :
  avg_speed_last_interval 120 2 50 (2 / 3) 60 (2 / 3) = 70 :=
by
  sorry

end avg_speed_last_40_min_is_70_l113_113572


namespace reciprocal_of_neg_one_div_2023_l113_113483

theorem reciprocal_of_neg_one_div_2023 : 1 / (-1 / (2023 : ℤ)) = -2023 := sorry

end reciprocal_of_neg_one_div_2023_l113_113483


namespace min_max_expr_l113_113543

noncomputable def expr (a b c : ℝ) : ℝ :=
  (1 / (a^2 + 1) + 1 / (b^2 + 1) + 1 / (c^2 + 1)) *
  (a^2 / (a^2 + 1) + b^2 / (b^2 + 1) + c^2 / (c^2 + 1))

theorem min_max_expr (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h_cond : a * b + b * c + c * a = 1) :
  27 / 16 ≤ expr a b c ∧ expr a b c ≤ 2 :=
sorry

end min_max_expr_l113_113543


namespace qt_q_t_neq_2_l113_113652

theorem qt_q_t_neq_2 (q t : ℕ) (hq : 0 < q) (ht : 0 < t) : q * t + q + t ≠ 2 :=
  sorry

end qt_q_t_neq_2_l113_113652


namespace solution_of_system_l113_113187

theorem solution_of_system (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : 2 * x + y = 1) : x + y = 3 :=
sorry

end solution_of_system_l113_113187


namespace cylinder_base_radius_l113_113754

theorem cylinder_base_radius (a : ℝ) (h_a_pos : 0 < a) :
  ∃ (R : ℝ), R = 7 * a * Real.sqrt 3 / 24 := 
    sorry

end cylinder_base_radius_l113_113754


namespace circle_diameter_l113_113128

theorem circle_diameter (A : ℝ) (h : A = 64 * Real.pi) : ∃ (d : ℝ), d = 16 :=
by
  sorry

end circle_diameter_l113_113128


namespace largest_multiple_of_7_l113_113567

def repeated_188 (k : Nat) : ℕ := (List.replicate k 188).foldr (λ x acc => x * 1000 + acc) 0

theorem largest_multiple_of_7 :
  ∃ n, n = repeated_188 100 ∧ ∃ m, m ≤ 303 ∧ m ≥ 0 ∧ m ≠ 300 ∧ (repeated_188 m % 7 = 0 → n ≥ repeated_188 m) :=
by
  sorry

end largest_multiple_of_7_l113_113567


namespace A_superset_B_l113_113210

open Set

variable (N : Set ℕ)
def A : Set ℕ := {x | ∃ n ∈ N, x = 2 * n}
def B : Set ℕ := {x | ∃ n ∈ N, x = 4 * n}

theorem A_superset_B : A N ⊇ B N :=
by
  -- Proof to be written
  sorry

end A_superset_B_l113_113210


namespace reciprocal_of_mixed_num_l113_113487

-- Define the fraction representation of the mixed number -1 1/2
def mixed_num_to_improper (a : ℚ) : ℚ := -3/2

-- Define the reciprocal function
def reciprocal (x : ℚ) : ℚ := 1 / x

-- Prove the statement
theorem reciprocal_of_mixed_num : reciprocal (mixed_num_to_improper (-1.5)) = -2/3 :=
by
  -- skip proof
  sorry

end reciprocal_of_mixed_num_l113_113487


namespace fraction_division_result_l113_113722

theorem fraction_division_result :
  (5/6) / (-9/10) = -25/27 := 
by
  sorry

end fraction_division_result_l113_113722


namespace girls_divisible_by_nine_l113_113825

def total_students (m c d u : ℕ) : ℕ := 1000 * m + 100 * c + 10 * d + u
def number_of_boys (m c d u : ℕ) : ℕ := m + c + d + u
def number_of_girls (m c d u : ℕ) : ℕ := total_students m c d u - number_of_boys m c d u 

theorem girls_divisible_by_nine (m c d u : ℕ) : 
  number_of_girls m c d u % 9 = 0 := 
by
    sorry

end girls_divisible_by_nine_l113_113825


namespace rectangle_width_to_length_ratio_l113_113430

theorem rectangle_width_to_length_ratio {w : ℕ} 
  (h1 : ∀ (l : ℕ), l = 10)
  (h2 : ∀ (p : ℕ), p = 32)
  (h3 : ∀ (P : ℕ), P = 2 * 10 + 2 * w) :
  (w : ℚ) / 10 = 3 / 5 :=
by
  sorry

end rectangle_width_to_length_ratio_l113_113430


namespace find_y_value_l113_113169

theorem find_y_value :
  (∃ m b : ℝ, (∀ x y : ℝ, (x = 2 ∧ y = 5) ∨ (x = 6 ∧ y = 17) ∨ (x = 10 ∧ y = 29) → y = m * x + b))
  → (∃ y : ℝ, x = 40 → y = 119) := by
  sorry

end find_y_value_l113_113169


namespace planes_1_and_6_adjacent_prob_l113_113200

noncomputable def probability_planes_adjacent (total_planes: ℕ) : ℚ :=
  if total_planes = 6 then 1/3 else 0

theorem planes_1_and_6_adjacent_prob :
  probability_planes_adjacent 6 = 1/3 := 
by
  sorry

end planes_1_and_6_adjacent_prob_l113_113200


namespace sum_of_extreme_a_l113_113077

theorem sum_of_extreme_a (a : ℝ) (h : ∀ x, x^2 - a*x - 20*a^2 < 0) (h_diff : |5*a - (-4*a)| ≤ 9) : 
  -1 ≤ a ∧ a ≤ 1 ∧ a ≠ 0 → a_min + a_max = 0 :=
by 
  sorry

end sum_of_extreme_a_l113_113077


namespace oil_flow_relationship_l113_113131

theorem oil_flow_relationship (t : ℝ) (Q : ℝ) (initial_quantity : ℝ) (flow_rate : ℝ)
  (h_initial : initial_quantity = 20) (h_flow : flow_rate = 0.2) :
  Q = initial_quantity - flow_rate * t :=
by
  -- proof to be filled in
  sorry

end oil_flow_relationship_l113_113131


namespace ben_min_sales_l113_113932

theorem ben_min_sales 
    (old_salary : ℕ := 75000) 
    (new_base_salary : ℕ := 45000) 
    (commission_rate : ℚ := 0.15) 
    (sale_amount : ℕ := 750) : 
    ∃ (n : ℕ), n ≥ 267 ∧ (old_salary ≤ new_base_salary + n * ⌊commission_rate * sale_amount⌋) :=
by 
  sorry

end ben_min_sales_l113_113932


namespace statement_A_statement_C_statement_D_l113_113989

variable (a : ℕ → ℝ) (A B : ℝ)

-- Condition: The sequence satisfies the recurrence relation
def recurrence_relation (n : ℕ) : Prop :=
  a (n + 2) = A * a (n + 1) + B * a n

-- Statement A: A=1 and B=-1 imply periodic with period 6
theorem statement_A (h : ∀ n, recurrence_relation a 1 (-1) n) :
  ∀ n, a (n + 6) = a n := 
sorry

-- Statement C: A=3 and B=-2 imply the derived sequence is geometric
theorem statement_C (h : ∀ n, recurrence_relation a 3 (-2) n) :
  ∃ r : ℝ, ∀ n, a (n + 1) - a n = r * (a n - a (n - 1)) :=
sorry

-- Statement D: A+1=B, a1=0, a2=B imply {a_{2n}} is increasing
theorem statement_D (hA : ∀ n, recurrence_relation a A (A + 1) n)
  (h1 : a 1 = 0) (h2 : a 2 = A + 1) :
  ∀ n, a (2 * (n + 1)) > a (2 * n) :=
sorry

end statement_A_statement_C_statement_D_l113_113989


namespace passengers_on_ship_l113_113035

theorem passengers_on_ship : 
  ∀ (P : ℕ), 
    P / 20 + P / 15 + P / 10 + P / 12 + P / 30 + 60 = P → 
    P = 90 :=
by 
  intros P h
  sorry

end passengers_on_ship_l113_113035


namespace symmetry_in_mathematics_l113_113335

-- Define the options
def optionA := "summation of harmonic series from 1 to 100"
def optionB := "general quadratic equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0"
def optionC := "Law of Sines: a / sin A = b / sin B = c / sin C"
def optionD := "arithmetic operation: 123456789 * 9 + 10 = 1111111111"

-- Define the symmetry property
def exhibits_symmetry (option: String) : Prop :=
  option = optionC

-- The theorem to prove
theorem symmetry_in_mathematics : ∃ option, exhibits_symmetry option := by
  use optionC
  sorry

end symmetry_in_mathematics_l113_113335


namespace two_pow_n_add_two_gt_n_sq_l113_113856

open Nat

theorem two_pow_n_add_two_gt_n_sq (n : ℕ) (h : n > 0) : 2^n + 2 > n^2 :=
by
  sorry

end two_pow_n_add_two_gt_n_sq_l113_113856


namespace min_xy_when_a_16_min_expr_when_a_0_l113_113307

-- Problem 1: Minimum value of xy when a = 16
theorem min_xy_when_a_16 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x * y = x + 4 * y + 16) : 16 ≤ x * y :=
    sorry

-- Problem 2: Minimum value of x + y + 2 / x + 1 / (2 * y) when a = 0
theorem min_expr_when_a_0 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x * y = x + 4 * y) : (11 : ℝ) / 2 ≤ x + y + 2 / x + 1 / (2 * y) :=
    sorry

end min_xy_when_a_16_min_expr_when_a_0_l113_113307


namespace minimum_xy_l113_113580

theorem minimum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 / x + 8 / y = 1) : xy ≥ 64 :=
sorry

end minimum_xy_l113_113580


namespace star_4_3_l113_113461

def star (a b : ℤ) : ℤ := a^2 - a * b + b^2

theorem star_4_3 : star 4 3 = 13 :=
by
  sorry

end star_4_3_l113_113461


namespace probability_divisible_by_three_l113_113284

noncomputable def prob_divisible_by_three : ℚ :=
  1 - (4/6)^6

theorem probability_divisible_by_three :
  prob_divisible_by_three = 665 / 729 :=
by
  sorry

end probability_divisible_by_three_l113_113284


namespace sum_of_smallest_natural_numbers_l113_113669

-- Define the problem statement
def satisfies_eq (A B : ℕ) := 360 / (A^3 / B) = 5

-- Prove that there exist natural numbers A and B such that 
-- satisfies_eq A B is true, and their sum is 9
theorem sum_of_smallest_natural_numbers :
  ∃ (A B : ℕ), satisfies_eq A B ∧ A + B = 9 :=
by
  -- Sorry is used here to indicate the proof is not given
  sorry

end sum_of_smallest_natural_numbers_l113_113669


namespace problem_statement_l113_113542

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x * (x + 4) else x * (x - 4)

theorem problem_statement (a : ℝ) (h : f a > f (8 - a)) : 4 < a :=
by sorry

end problem_statement_l113_113542


namespace blue_bead_probability_no_adjacent_l113_113440

theorem blue_bead_probability_no_adjacent :
  let total_beads := 9
  let blue_beads := 5
  let green_beads := 3
  let red_bead := 1
  let total_permutations := Nat.factorial total_beads / (Nat.factorial blue_beads * Nat.factorial green_beads * Nat.factorial red_bead)
  let valid_arrangements := (Nat.factorial 4) / (Nat.factorial 3 * Nat.factorial 1)
  let no_adjacent_valid := 4
  let probability_no_adj := (no_adjacent_valid : ℚ) / total_permutations
  probability_no_adj = (1 : ℚ) / 126 := 
by
  sorry

end blue_bead_probability_no_adjacent_l113_113440


namespace solution_set_of_inequality_l113_113894

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 3 * x - 2 ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end solution_set_of_inequality_l113_113894


namespace range_of_y_l113_113994

theorem range_of_y (y : ℝ) (h1: 1 / y < 3) (h2: 1 / y > -4) : y > 1 / 3 :=
by
  sorry

end range_of_y_l113_113994


namespace tan_2x_abs_properties_l113_113036

open Real

theorem tan_2x_abs_properties :
  (∀ x : ℝ, |tan (2 * x)| = |tan (2 * (-x))|) ∧ (∀ x : ℝ, |tan (2 * x)| = |tan (2 * (x + π / 2))|) :=
by
  sorry

end tan_2x_abs_properties_l113_113036


namespace number_of_people_for_cheaper_second_caterer_l113_113093

theorem number_of_people_for_cheaper_second_caterer : 
  ∃ (x : ℕ), (150 + 20 * x > 250 + 15 * x + 50) ∧ 
  ∀ (y : ℕ), (y < x → ¬ (150 + 20 * y > 250 + 15 * y + 50)) :=
by
  sorry

end number_of_people_for_cheaper_second_caterer_l113_113093


namespace find_prime_triplet_l113_113393

open Nat

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m ≤ n / 2 → (m ∣ n) → False

theorem find_prime_triplet :
  ∃ p q r : ℕ, is_prime p ∧ is_prime q ∧ is_prime r ∧ 
  (p + q = r) ∧ 
  (∃ k : ℕ, (r - p) * (q - p) - 27 * p = k * k) ∧ 
  (p = 2 ∧ q = 29 ∧ r = 31) := by
  sorry

end find_prime_triplet_l113_113393


namespace unripe_oranges_after_days_l113_113422

-- Definitions and Conditions
def sacks_per_day := 65
def days := 6

-- Statement to prove
theorem unripe_oranges_after_days : sacks_per_day * days = 390 := by
  sorry

end unripe_oranges_after_days_l113_113422


namespace find_side_length_of_square_l113_113701

variable (a : ℝ)

theorem find_side_length_of_square (h1 : a - 3 > 0)
                                   (h2 : 3 * a + 5 * (a - 3) = 57) :
  a = 9 := 
by
  sorry

end find_side_length_of_square_l113_113701


namespace probability_of_drawing_ball_1_is_2_over_5_l113_113236

noncomputable def probability_of_drawing_ball_1 : ℚ :=
  let total_balls := [1, 2, 3, 4, 5]
  let draw_pairs := [ (1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5) ]
  let favorable_pairs := [ (1, 2), (1, 3), (1, 4), (1, 5) ]
  (favorable_pairs.length : ℚ) / (draw_pairs.length : ℚ)

theorem probability_of_drawing_ball_1_is_2_over_5 :
  probability_of_drawing_ball_1 = 2 / 5 :=
by sorry

end probability_of_drawing_ball_1_is_2_over_5_l113_113236


namespace fraction_of_180_l113_113529

theorem fraction_of_180 : (1 / 2) * (1 / 3) * (1 / 6) * 180 = 5 := by
  sorry

end fraction_of_180_l113_113529


namespace certain_number_divided_by_two_l113_113141

theorem certain_number_divided_by_two (x : ℝ) (h : x / 2 + x + 2 = 62) : x = 40 :=
sorry

end certain_number_divided_by_two_l113_113141


namespace cody_initial_tickets_l113_113773

theorem cody_initial_tickets (T : ℕ) (h1 : T - 25 + 6 = 30) : T = 49 :=
sorry

end cody_initial_tickets_l113_113773


namespace enrollment_difference_l113_113900

theorem enrollment_difference :
  let M := 1500
  let S := 2100
  let L := 2700
  let R := 1800
  let B := 900
  max M (max S (max L (max R B))) - min M (min S (min L (min R B))) = 1800 := 
by 
  sorry

end enrollment_difference_l113_113900


namespace aquatic_reserve_total_fishes_l113_113741

-- Define the number of bodies of water
def bodies_of_water : ℕ := 6

-- Define the number of fishes per body of water
def fishes_per_body : ℕ := 175

-- Define the total number of fishes
def total_fishes : ℕ := bodies_of_water * fishes_per_body

theorem aquatic_reserve_total_fishes : bodies_of_water * fishes_per_body = 1050 := by
  -- The proof is omitted.
  sorry

end aquatic_reserve_total_fishes_l113_113741


namespace sqrt_expression_evaluation_l113_113013

theorem sqrt_expression_evaluation (sqrt48 : Real) (sqrt1div3 : Real) 
  (h1 : sqrt48 = 4 * Real.sqrt 3) (h2 : sqrt1div3 = Real.sqrt (1 / 3)) :
  (-1 / 2) * sqrt48 * sqrt1div3 = -2 :=
by 
  rw [h1, h2]
  -- Continue with the simplification steps, however
  sorry

end sqrt_expression_evaluation_l113_113013


namespace find_divisor_l113_113391

theorem find_divisor :
  ∃ d : ℕ, (4499 + 1) % d = 0 ∧ d = 2 :=
by
  sorry

end find_divisor_l113_113391


namespace find_k_l113_113827

   theorem find_k (m n : ℝ) (k : ℝ) (hm : m > 0) (hn : n > 0)
     (h1 : k = Real.log m / Real.log 2)
     (h2 : k = Real.log n / (Real.log 4))
     (h3 : k = Real.log (4 * m + 3 * n) / (Real.log 8)) :
     k = 2 :=
   by
     sorry
   
end find_k_l113_113827


namespace marked_price_of_jacket_l113_113847

variable (x : ℝ) -- Define the variable x as a real number representing the marked price.

-- Define the conditions as a Lean theorem statement
theorem marked_price_of_jacket (cost price_sold profit : ℝ) (h1 : cost = 350) (h2 : price_sold = 0.8 * x) (h3 : profit = price_sold - cost) : 
  x = 550 :=
by
  -- We would solve the proof here using provided conditions
  sorry

end marked_price_of_jacket_l113_113847


namespace matrix_vector_multiplication_correct_l113_113283

noncomputable def mat : Matrix (Fin 2) (Fin 2) ℤ := ![![3, -2], ![1, 5]]
noncomputable def vec : Fin 2 → ℤ := ![-1, 2]
noncomputable def result : Fin 2 → ℤ := ![-7, 9]

theorem matrix_vector_multiplication_correct :
  (Matrix.mulVec mat vec) = result :=
by
  sorry

end matrix_vector_multiplication_correct_l113_113283


namespace correct_answer_l113_113282

-- Define the sentence structure and the requirement for a formal object
structure SentenceStructure where
  subject : String := "I"
  verb : String := "like"
  object_placeholder : String := "_"
  clause : String := "when the weather is clear and bright"

-- Correct choices provided
inductive Choice
  | this
  | that
  | it
  | one

-- Problem formulation: Based on SentenceStructure, prove that 'it' is the correct choice
theorem correct_answer {S : SentenceStructure} : Choice.it = Choice.it :=
by
  -- Proof omitted
  sorry

end correct_answer_l113_113282


namespace sum_2001_and_1015_l113_113930

theorem sum_2001_and_1015 :
  2001 + 1015 = 3016 :=
sorry

end sum_2001_and_1015_l113_113930


namespace coordinates_on_y_axis_l113_113880

theorem coordinates_on_y_axis (a : ℝ) 
  (h : (a - 3) = 0) : 
  P = (0, -1) :=
by 
  have ha : a = 3 := by sorry
  subst ha
  sorry

end coordinates_on_y_axis_l113_113880


namespace sin_double_angle_l113_113707

open Real

theorem sin_double_angle
  {α : ℝ} (h1: tan α = -1/2) (h2: 0 < α ∧ α < π) :
  sin (2 * α) = -4/5 :=
sorry

end sin_double_angle_l113_113707


namespace find_y_given_x_zero_l113_113674

theorem find_y_given_x_zero (t : ℝ) (y : ℝ) : 
  (3 - 2 * t = 0) → (y = 3 * t + 6) → y = 21 / 2 := 
by 
  sorry

end find_y_given_x_zero_l113_113674


namespace midpoint_coordinates_l113_113998

theorem midpoint_coordinates :
  let x1 := 2
  let y1 := -3
  let z1 := 5
  let x2 := 8
  let y2 := 3
  let z2 := -1
  ( (x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2 ) = (5, 0, 2) :=
by
  sorry

end midpoint_coordinates_l113_113998


namespace range_of_c_div_a_l113_113978

-- Define the conditions and variables
variables (a b c : ℝ)

-- Define the given conditions
def conditions : Prop :=
  (a ≥ b ∧ b ≥ c) ∧ (a + b + c = 0)

-- Define the range of values for c / a
def range_for_c_div_a : Prop :=
  -2 ≤ c / a ∧ c / a ≤ -1/2

-- The theorem statement to prove
theorem range_of_c_div_a (h : conditions a b c) : range_for_c_div_a a c := 
  sorry

end range_of_c_div_a_l113_113978


namespace tiles_needed_l113_113944

-- Definitions for the problem
def width_wall : ℕ := 36
def length_wall : ℕ := 72
def width_tile : ℕ := 3
def length_tile : ℕ := 4

-- The area of the wall
def A_wall : ℕ := width_wall * length_wall

-- The area of one tile
def A_tile : ℕ := width_tile * length_tile

-- The number of tiles needed
def number_of_tiles : ℕ := A_wall / A_tile

-- Proof statement
theorem tiles_needed : number_of_tiles = 216 := by
  sorry

end tiles_needed_l113_113944


namespace outstanding_consumer_installment_credit_l113_113513

-- Given conditions
def total_consumer_installment_credit (C : ℝ) : Prop :=
  let automobile_installment_credit := 0.36 * C
  let automobile_finance_credit := 75
  let total_automobile_credit := 2 * automobile_finance_credit
  automobile_installment_credit = total_automobile_credit

-- Theorem to prove
theorem outstanding_consumer_installment_credit : ∃ (C : ℝ), total_consumer_installment_credit C ∧ C = 416.67 := 
by
  sorry

end outstanding_consumer_installment_credit_l113_113513


namespace ratio_of_boys_l113_113267

theorem ratio_of_boys (p : ℚ) (hp : p = (3 / 4) * (1 - p)) : p = 3 / 7 :=
by
  -- Proof would be provided here
  sorry

end ratio_of_boys_l113_113267


namespace store_discount_problem_l113_113887

theorem store_discount_problem (original_price : ℝ) :
  let price_after_first_discount := original_price * 0.75
  let price_after_second_discount := price_after_first_discount * 0.90
  let true_discount := 1 - price_after_second_discount / original_price
  let claimed_discount := 0.40
  let difference := claimed_discount - true_discount
  true_discount = 0.325 ∧ difference = 0.075 :=
by
  sorry

end store_discount_problem_l113_113887


namespace standard_deviation_is_one_l113_113770

def mean : ℝ := 10.5
def value : ℝ := 8.5

theorem standard_deviation_is_one (σ : ℝ) (h : value = mean - 2 * σ) : σ = 1 :=
by {
  sorry
}

end standard_deviation_is_one_l113_113770


namespace point_outside_circle_l113_113607

theorem point_outside_circle {a b : ℝ} (h : ∃ x y : ℝ, x^2 + y^2 = 1 ∧ a * x + b * y = 1) : a^2 + b^2 > 1 :=
by sorry

end point_outside_circle_l113_113607


namespace convert_to_scientific_notation_l113_113056

-- Problem statement: convert 120 million to scientific notation and validate the format.
theorem convert_to_scientific_notation :
  120000000 = 1.2 * 10^7 :=
sorry

end convert_to_scientific_notation_l113_113056


namespace john_frank_age_ratio_l113_113241

theorem john_frank_age_ratio
  (F J : ℕ)
  (h1 : F + 4 = 16)
  (h2 : J - F = 15)
  (h3 : ∃ k : ℕ, J + 3 = k * (F + 3)) :
  (J + 3) / (F + 3) = 2 :=
by
  sorry

end john_frank_age_ratio_l113_113241


namespace find_x_l113_113130

theorem find_x (x : ℕ) : 
  (∃ (students : ℕ), students = 10) ∧ 
  (∃ (selected : ℕ), selected = 6) ∧ 
  (¬ (∃ (k : ℕ), k = 5 ∧ k = x) ) ∧ 
  (1 ≤ 10 - x) ∧
  (3 ≤ x ∧ x ≤ 4) :=
by
  sorry

end find_x_l113_113130


namespace Caitlin_correct_age_l113_113121

def Aunt_Anna_age := 48
def Brianna_age := Aunt_Anna_age / 2
def Caitlin_age := Brianna_age - 7

theorem Caitlin_correct_age : Caitlin_age = 17 := by
  /- Condon: Aunt Anna is 48 years old. -/
  let ha := Aunt_Anna_age
  /- Condon: Brianna is half as old as Aunt Anna. -/
  let hb := Brianna_age
  /- Condon: Caitlin is 7 years younger than Brianna. -/
  let hc := Caitlin_age
  /- Question: How old is Caitlin? Proof: -/
  sorry

end Caitlin_correct_age_l113_113121


namespace symmetric_origin_coordinates_l113_113912

def symmetric_coordinates (x y : ℚ) (x_line y_line : ℚ) : Prop :=
  x_line - 2 * y_line + 2 = 0 ∧ y_line = -2 * x_line ∧ x = -4/5 ∧ y = 8/5

theorem symmetric_origin_coordinates :
  ∃ (x_0 y_0 : ℚ), symmetric_coordinates x_0 y_0 (-4/5) (8/5) :=
by
  use -4/5, 8/5
  sorry

end symmetric_origin_coordinates_l113_113912


namespace work_rate_problem_l113_113129

theorem work_rate_problem :
  ∃ (x : ℝ), 
    (0 < x) ∧ 
    (10 * (1 / x + 1 / 40) = 0.5833333333333334) ∧ 
    (x = 30) :=
by
  sorry

end work_rate_problem_l113_113129


namespace evaluate_expression_l113_113223

theorem evaluate_expression (a b : ℤ) (h1 : a = 4) (h2 : b = -3) :
  2 * a^2 - 3 * b^2 + 4 * a * b = -43 :=
by
  sorry

end evaluate_expression_l113_113223


namespace infinite_perfect_squares_in_arithmetic_sequence_l113_113837

theorem infinite_perfect_squares_in_arithmetic_sequence 
  (a d : ℕ) 
  (h_exists_perfect_square : ∃ (n₀ k : ℕ), a + n₀ * d = k^2) 
  : ∃ (S : ℕ → ℕ), (∀ n, ∃ t, S n = a + t * d ∧ ∃ k, S n = k^2) ∧ (∀ m n, S m = S n → m = n) :=
sorry

end infinite_perfect_squares_in_arithmetic_sequence_l113_113837


namespace percentage_of_number_l113_113881

theorem percentage_of_number (P : ℝ) (h : 0.10 * 3200 - 190 = P * 650) :
  P = 0.2 :=
sorry

end percentage_of_number_l113_113881


namespace fraction_of_fifth_set_l113_113981

theorem fraction_of_fifth_set :
  let total_match_duration := 11 * 60 + 5
  let fifth_set_duration := 8 * 60 + 11
  (fifth_set_duration : ℚ) / total_match_duration = 3 / 4 := 
sorry

end fraction_of_fifth_set_l113_113981


namespace calculate_fourth_quarter_shots_l113_113340

-- Definitions based on conditions
def first_quarters_shots : ℕ := 20
def first_quarters_successful_shots : ℕ := 12
def third_quarter_shots : ℕ := 10
def overall_accuracy : ℚ := 46 / 100
def total_shots (n : ℕ) : ℕ := first_quarters_shots + third_quarter_shots + n
def total_successful_shots (n : ℕ) : ℚ := first_quarters_successful_shots + 3 + (4 / 10 * n)


-- Main theorem to prove
theorem calculate_fourth_quarter_shots (n : ℕ) (h : (total_successful_shots n) / (total_shots n) = overall_accuracy) : 
  n = 20 :=
by {
  sorry
}

end calculate_fourth_quarter_shots_l113_113340


namespace find_a_l113_113231

theorem find_a (a b c : ℕ) (h_positive_a : 0 < a) (h_positive_b : 0 < b) (h_positive_c : 0 < c) (h_eq : (18 ^ a) * (9 ^ (3 * a - 1)) * (c ^ a) = (2 ^ 7) * (3 ^ b)) : a = 7 := by
  sorry

end find_a_l113_113231


namespace trail_mix_total_weight_l113_113395

noncomputable def peanuts : ℝ := 0.16666666666666666
noncomputable def chocolate_chips : ℝ := 0.16666666666666666
noncomputable def raisins : ℝ := 0.08333333333333333

theorem trail_mix_total_weight :
  peanuts + chocolate_chips + raisins = 0.41666666666666663 :=
by
  unfold peanuts chocolate_chips raisins
  sorry

end trail_mix_total_weight_l113_113395


namespace cups_filled_l113_113859

def total_tea : ℕ := 1050
def tea_per_cup : ℕ := 65

theorem cups_filled : Nat.floor (total_tea / (tea_per_cup : ℚ)) = 16 :=
by
  sorry

end cups_filled_l113_113859


namespace division_of_fractions_l113_113331

theorem division_of_fractions :
  (5 : ℚ) / 6 / ((2 : ℚ) / 3) = (5 : ℚ) / 4 :=
by
  sorry

end division_of_fractions_l113_113331


namespace appropriate_mass_units_l113_113004

def unit_of_mass_basket_of_eggs : String :=
  if 5 = 5 then "kilograms" else "unknown"

def unit_of_mass_honeybee : String :=
  if 5 = 5 then "grams" else "unknown"

def unit_of_mass_tank : String :=
  if 6 = 6 then "tons" else "unknown"

theorem appropriate_mass_units :
  unit_of_mass_basket_of_eggs = "kilograms" ∧
  unit_of_mass_honeybee = "grams" ∧
  unit_of_mass_tank = "tons" :=
by {
  -- skip the proof
  sorry
}

end appropriate_mass_units_l113_113004


namespace parabola_equation_l113_113000

-- Define the given conditions
def vertex : ℝ × ℝ := (3, 5)
def point_on_parabola : ℝ × ℝ := (4, 2)

-- Prove that the equation is as specified
theorem parabola_equation :
  ∃ a b c : ℝ, (a ≠ 0) ∧ (∀ x y : ℝ, (y = a * x^2 + b * x + c) ↔
     (y = -3 * x^2 + 18 * x - 22) ∧ (vertex.snd = -3 * (vertex.fst - 3)^2 + 5) ∧
     (point_on_parabola.snd = a * point_on_parabola.fst^2 + b * point_on_parabola.fst + c)) := 
sorry

end parabola_equation_l113_113000


namespace four_circles_max_parts_l113_113356

theorem four_circles_max_parts (n : ℕ) (h1 : ∀ n, n = 1 ∨ n = 2 ∨ n = 3 → ∃ k, k = 2^n) :
    n = 4 → ∃ k, k = 14 :=
by
  sorry

end four_circles_max_parts_l113_113356


namespace change_in_expression_l113_113985

theorem change_in_expression (x a : ℝ) (ha : 0 < a) :
  (x^3 - 3*x + 1) + (3*a*x^2 + 3*a^2*x + a^3 - 3*a) = (x + a)^3 - 3*(x + a) + 1 ∧
  (x^3 - 3*x + 1) + (-3*a*x^2 + 3*a^2*x - a^3 + 3*a) = (x - a)^3 - 3*(x - a) + 1 :=
by sorry

end change_in_expression_l113_113985


namespace exists_real_a_l113_113876

noncomputable def A (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }
def B : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def C : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }

theorem exists_real_a : ∃ a : ℝ, a = -2 ∧ A a ∩ C = ∅ ∧ ∅ ⊂ A a ∩ B := 
by {
  sorry
}

end exists_real_a_l113_113876


namespace remainder_mod7_l113_113711

theorem remainder_mod7 (n : ℕ) (h1 : n^2 % 7 = 1) (h2 : n^3 % 7 = 6) : n % 7 = 6 := 
by
  sorry

end remainder_mod7_l113_113711


namespace sugar_percentage_of_second_solution_l113_113873

theorem sugar_percentage_of_second_solution :
  ∀ (W : ℝ) (P : ℝ),
  (0.10 * W * (3 / 4) + P / 100 * (1 / 4) * W = 0.18 * W) → 
  (P = 42) :=
by
  intros W P h
  sorry

end sugar_percentage_of_second_solution_l113_113873


namespace odd_positive_93rd_l113_113968

theorem odd_positive_93rd : 
  (2 * 93 - 1) = 185 := 
by sorry

end odd_positive_93rd_l113_113968


namespace largest_possible_s_l113_113302

theorem largest_possible_s (r s : ℕ) (h1 : 3 ≤ s) (h2 : s ≤ r) (h3 : s < 122)
    (h4 : ∀ r s, (61 * (s - 2) * r = 60 * (r - 2) * s)) : s ≤ 121 :=
by
  sorry

end largest_possible_s_l113_113302


namespace difference_max_min_y_l113_113301

-- Define initial and final percentages of responses
def initial_yes : ℝ := 0.30
def initial_no : ℝ := 0.70
def final_yes : ℝ := 0.60
def final_no : ℝ := 0.40

-- Define the problem statement
theorem difference_max_min_y : 
  ∃ y_min y_max : ℝ, (initial_yes + initial_no = 1) ∧ (final_yes + final_no = 1) ∧
  (initial_yes + initial_no = final_yes + final_no) ∧ y_min ≤ y_max ∧ 
  y_max - y_min = 0.30 :=
sorry

end difference_max_min_y_l113_113301


namespace pow_simplification_l113_113374

theorem pow_simplification :
  9^6 * 3^3 / 27^4 = 27 :=
by
  sorry

end pow_simplification_l113_113374


namespace find_circle_center_l113_113643

noncomputable def circle_center : (ℝ × ℝ) :=
  let x_center := 5
  let y_center := 4
  (x_center, y_center)

theorem find_circle_center (x y : ℝ) (h : x^2 - 10 * x + y^2 - 8 * y = 16) :
  circle_center = (5, 4) := by
  sorry

end find_circle_center_l113_113643


namespace common_ratio_of_geometric_seq_l113_113346

-- Define the arithmetic sequence
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

-- Define the geometric sequence property
def geometric_seq_property (a2 a3 a6 : ℤ) : Prop :=
  a3 * a3 = a2 * a6

-- State the main theorem
theorem common_ratio_of_geometric_seq (a d : ℤ) (h : ¬d = 0) :
  geometric_seq_property (arithmetic_seq a d 2) (arithmetic_seq a d 3) (arithmetic_seq a d 6) →
  ∃ q : ℤ, q = 3 ∨ q = 1 :=
by
  sorry

end common_ratio_of_geometric_seq_l113_113346


namespace one_fourths_in_seven_halves_l113_113690

theorem one_fourths_in_seven_halves : (7 / 2) / (1 / 4) = 14 := by
  sorry

end one_fourths_in_seven_halves_l113_113690


namespace correctTechnologyUsedForVolcanicAshMonitoring_l113_113742

-- Define the choices
inductive Technology
| RemoteSensing : Technology
| GPS : Technology
| GIS : Technology
| DigitalEarth : Technology

-- Define the problem conditions
def primaryTechnologyUsedForVolcanicAshMonitoring := Technology.RemoteSensing

-- The statement to prove
theorem correctTechnologyUsedForVolcanicAshMonitoring : primaryTechnologyUsedForVolcanicAshMonitoring = Technology.RemoteSensing :=
by
  sorry

end correctTechnologyUsedForVolcanicAshMonitoring_l113_113742


namespace union_sets_l113_113126

def A : Set ℝ := { x | (2 / x) > 1 }
def B : Set ℝ := { x | Real.log x < 0 }

theorem union_sets : (A ∪ B) = { x : ℝ | 0 < x ∧ x < 2 } := by
  sorry

end union_sets_l113_113126


namespace simplify_expression_l113_113886

theorem simplify_expression (x : ℤ) : 
  (2 * x ^ 13 + 3 * x ^ 12 - 4 * x ^ 9 + 5 * x ^ 7) + 
  (8 * x ^ 11 - 2 * x ^ 9 + 3 * x ^ 7 + 6 * x ^ 4 - 7 * x + 9) + 
  (x ^ 13 + 4 * x ^ 12 + x ^ 11 + 9 * x ^ 9) = 
  3 * x ^ 13 + 7 * x ^ 12 + 9 * x ^ 11 + 3 * x ^ 9 + 8 * x ^ 7 + 6 * x ^ 4 - 7 * x + 9 :=
sorry

end simplify_expression_l113_113886


namespace radius_I_l113_113032

noncomputable def radius_O1 : ℝ := 3
noncomputable def radius_O2 : ℝ := 3
noncomputable def radius_O3 : ℝ := 3

axiom O1_O2_tangent : ∀ (O1 O2 : ℝ), O1 + O2 = radius_O1 + radius_O2
axiom O2_O3_tangent : ∀ (O2 O3 : ℝ), O2 + O3 = radius_O2 + radius_O3
axiom O3_O1_tangent : ∀ (O3 O1 : ℝ), O3 + O1 = radius_O3 + radius_O1

axiom I_O1_tangent : ∀ (I O1 : ℝ), I + O1 = radius_O1 + I
axiom I_O2_tangent : ∀ (I O2 : ℝ), I + O2 = radius_O2 + I
axiom I_O3_tangent : ∀ (I O3 : ℝ), I + O3 = radius_O3 + I

theorem radius_I : ∀ (I : ℝ), I = radius_O1 :=
by
  sorry

end radius_I_l113_113032


namespace find_k_l113_113220

theorem find_k (x y k : ℝ)
  (h1 : x - 4 * y + 3 ≤ 0)
  (h2 : 3 * x + 5 * y - 25 ≤ 0)
  (h3 : x ≥ 1)
  (h4 : ∃ z, z = k * x + y ∧ z = 12)
  (h5 : ∃ z', z' = k * x + y ∧ z' = 3) :
  k = 2 :=
by sorry

end find_k_l113_113220


namespace translated_parabola_expression_correct_l113_113662

-- Definitions based on the conditions
def original_parabola (x : ℝ) : ℝ := x^2 - 1
def translated_parabola (x : ℝ) : ℝ := (x + 2)^2

-- The theorem to prove
theorem translated_parabola_expression_correct :
  ∀ x : ℝ, translated_parabola x = original_parabola (x + 2) + 1 :=
by
  sorry

end translated_parabola_expression_correct_l113_113662


namespace solve_for_x_l113_113308

theorem solve_for_x : ∃ x : ℝ, 64 = 2 * (16 : ℝ)^(x - 2) ∧ x = 3.25 := by
  sorry

end solve_for_x_l113_113308


namespace a_is_multiple_of_2_l113_113871

theorem a_is_multiple_of_2 (a : ℕ) (h1 : 0 < a) (h2 : (4 ^ a) % 10 = 6) : a % 2 = 0 :=
sorry

end a_is_multiple_of_2_l113_113871


namespace prime_p_square_condition_l113_113637

theorem prime_p_square_condition (p : ℕ) (h_prime : Prime p) (h_square : ∃ n : ℤ, 5^p + 4 * p^4 = n^2) :
  p = 31 :=
sorry

end prime_p_square_condition_l113_113637


namespace find_fraction_l113_113250

-- Define the given variables and conditions
variables (x y : ℝ)
-- Assume x and y are nonzero
variable (hx : x ≠ 0)
variable (hy : y ≠ 0)
-- Assume the given condition
variable (h : (4*x + 2*y) / (2*x - 8*y) = 3)

-- Define the theorem to be proven
theorem find_fraction (h : (4*x + 2*y) / (2*x - 8*y) = 3) : (x + 4 * y) / (4 * x - y) = 1 / 3 := 
by
  sorry

end find_fraction_l113_113250


namespace constant_difference_of_equal_derivatives_l113_113774

theorem constant_difference_of_equal_derivatives
  {f g : ℝ → ℝ}
  (h : ∀ x, deriv f x = deriv g x) :
  ∃ C : ℝ, ∀ x, f x - g x = C := 
sorry

end constant_difference_of_equal_derivatives_l113_113774


namespace real_number_unique_l113_113255

variable (a x : ℝ)

theorem real_number_unique (h1 : (a + 3) * (a + 3) = x)
  (h2 : (2 * a - 9) * (2 * a - 9) = x) : x = 25 := by
  sorry

end real_number_unique_l113_113255


namespace sam_paint_cans_l113_113589

theorem sam_paint_cans : 
  ∀ (cans_per_room : ℝ) (initial_cans remaining_cans : ℕ),
    initial_cans * cans_per_room = 40 ∧
    remaining_cans * cans_per_room = 30 ∧
    initial_cans - remaining_cans = 4 →
    remaining_cans = 12 :=
by sorry

end sam_paint_cans_l113_113589


namespace percent_absent_of_students_l113_113746

theorem percent_absent_of_students
  (boys girls : ℕ)
  (total_students := boys + girls)
  (boys_absent_fraction girls_absent_fraction : ℚ)
  (boys_absent_fraction_eq : boys_absent_fraction = 1 / 8)
  (girls_absent_fraction_eq : girls_absent_fraction = 1 / 4)
  (total_students_eq : total_students = 160)
  (boys_eq : boys = 80)
  (girls_eq : girls = 80) :
  (boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students * 100 = 18.75 :=
by
  sorry

end percent_absent_of_students_l113_113746


namespace average_score_for_entire_class_l113_113647

def total_students : ℕ := 100
def assigned_day_percentage : ℝ := 0.70
def make_up_day_percentage : ℝ := 0.30
def assigned_day_avg_score : ℝ := 65
def make_up_day_avg_score : ℝ := 95

theorem average_score_for_entire_class :
  (assigned_day_percentage * total_students * assigned_day_avg_score + make_up_day_percentage * total_students * make_up_day_avg_score) / total_students = 74 := by
  sorry

end average_score_for_entire_class_l113_113647


namespace find_common_difference_find_possible_a1_l113_113555

structure ArithSeq :=
  (a : ℕ → ℤ) -- defining the sequence
  
noncomputable def S (n : ℕ) (a : ArithSeq) : ℤ :=
  (n * (2 * a.a 0 + (n - 1) * (a.a 1 - a.a 0))) / 2

axiom a4 (a : ArithSeq) : a.a 3 = 10

axiom S20 (a : ArithSeq) : S 20 a = 590

theorem find_common_difference (a : ArithSeq) (d : ℤ) : 
  (a.a 1 - a.a 0 = d) →
  d = 3 :=
sorry

theorem find_possible_a1 (a : ArithSeq) : 
  (∃a1: ℤ, a1 ∈ Set.range a.a) →
  (∀n : ℕ, S n a ≤ S 7 a) →
  Set.range a.a ∩ {n | 18 ≤ n ∧ n ≤ 20} = {18, 19, 20} :=
sorry

end find_common_difference_find_possible_a1_l113_113555


namespace range_of_a_l113_113244

noncomputable def f (a x : ℝ) : ℝ :=
  x^2 + (a^2 + 1) * x + a - 2

theorem range_of_a (a : ℝ) :
  (f a 1 < 0) ∧ (f a (-1) < 0) → -1 < a ∧ a < 0 :=
by
  sorry

end range_of_a_l113_113244


namespace cube_volume_surface_area_l113_113891

theorem cube_volume_surface_area (x : ℝ) (s : ℝ)
  (h1 : s^3 = 3 * x)
  (h2 : 6 * s^2 = 6 * x) :
  x = 3 :=
by sorry

end cube_volume_surface_area_l113_113891


namespace cost_of_camel_l113_113601

theorem cost_of_camel
  (C H O E : ℝ)
  (h1 : 10 * C = 24 * H)
  (h2 : 16 * H = 4 * O)
  (h3 : 6 * O = 4 * E)
  (h4 : 10 * E = 140000) :
  C = 5600 :=
by
  -- Skipping the proof steps
  sorry

end cost_of_camel_l113_113601


namespace parabola_x_intercepts_count_l113_113410

theorem parabola_x_intercepts_count : 
  let equation := fun y : ℝ => -3 * y^2 + 2 * y + 3
  ∃! x : ℝ, ∃ y : ℝ, y = 0 ∧ x = equation y :=
by
  sorry

end parabola_x_intercepts_count_l113_113410


namespace calc_root_difference_l113_113638

theorem calc_root_difference :
  ((81: ℝ)^(1/4) + (32: ℝ)^(1/5) - (49: ℝ)^(1/2)) = -2 :=
by
  have h1 : (81: ℝ)^(1/4) = 3 := by sorry
  have h2 : (32: ℝ)^(1/5) = 2 := by sorry
  have h3 : (49: ℝ)^(1/2) = 7 := by sorry
  rw [h1, h2, h3]
  norm_num

end calc_root_difference_l113_113638


namespace value_of_a_l113_113133

theorem value_of_a {a x : ℝ} (h1 : x > 0) (h2 : 2 * x + 1 > a * x) : a ≤ 2 :=
sorry

end value_of_a_l113_113133


namespace mart_income_percentage_of_juan_l113_113736

theorem mart_income_percentage_of_juan
  (J T M : ℝ)
  (h1 : T = 0.60 * J)
  (h2 : M = 1.60 * T) :
  M = 0.96 * J :=
by 
  sorry

end mart_income_percentage_of_juan_l113_113736


namespace middle_number_of_consecutive_sum_30_l113_113748

theorem middle_number_of_consecutive_sum_30 (n : ℕ) (h : n + (n + 1) + (n + 2) = 30) : n + 1 = 10 :=
by
  sorry

end middle_number_of_consecutive_sum_30_l113_113748


namespace least_n_satisfies_inequality_l113_113807

theorem least_n_satisfies_inequality : ∃ n : ℕ, (1 : ℚ) / n - (1 : ℚ) / (n + 1) < 1 / 15 ∧ ∀ m : ℕ, (1 : ℚ) / m - (1 : ℚ) / (m + 1) < 1 / 15 -> ¬ (m < n) := 
sorry

end least_n_satisfies_inequality_l113_113807


namespace abs_sub_lt_five_solution_set_l113_113254

theorem abs_sub_lt_five_solution_set (x : ℝ) : |x - 3| < 5 ↔ -2 < x ∧ x < 8 :=
by sorry

end abs_sub_lt_five_solution_set_l113_113254


namespace total_chairs_taken_l113_113624

def num_students : ℕ := 5
def chairs_per_trip : ℕ := 5
def num_trips : ℕ := 10

theorem total_chairs_taken :
  (num_students * chairs_per_trip * num_trips) = 250 :=
by
  sorry

end total_chairs_taken_l113_113624


namespace seven_expression_one_seven_expression_two_l113_113611

theorem seven_expression_one : 777 / 7 - 77 / 7 = 100 :=
by sorry

theorem seven_expression_two : 7 * 7 + 7 * 7 + 7 / 7 + 7 / 7 = 100 :=
by sorry

end seven_expression_one_seven_expression_two_l113_113611


namespace inequality_proof_l113_113584

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x * y + y * z + z * x = 1) :
  3 - Real.sqrt 3 + (x^2 / y) + (y^2 / z) + (z^2 / x) ≥ (x + y + z)^2 :=
by
  sorry

end inequality_proof_l113_113584


namespace age_difference_in_decades_l113_113631

-- Declare the ages of x, y, and z as real numbers
variables (x y z : ℝ)

-- Define the condition
def age_condition (x y z : ℝ) : Prop := x + y = y + z + 18

-- The proof problem statement
theorem age_difference_in_decades (h : age_condition x y z) : (x - z) / 10 = 1.8 :=
by {
  -- Proof is omitted as per instructions
  sorry
}

end age_difference_in_decades_l113_113631


namespace expected_losses_correct_l113_113224

def game_probabilities : List (ℕ × ℝ) := [
  (5, 0.6), (10, 0.75), (15, 0.4), (12, 0.85), (20, 0.5),
  (30, 0.2), (10, 0.9), (25, 0.7), (35, 0.65), (10, 0.8)
]

def expected_losses : ℝ :=
  (1 - 0.6) + (1 - 0.75) + (1 - 0.4) + (1 - 0.85) +
  (1 - 0.5) + (1 - 0.2) + (1 - 0.9) + (1 - 0.7) +
  (1 - 0.65) + (1 - 0.8)

theorem expected_losses_correct :
  expected_losses = 3.55 :=
by {
  -- Skipping the actual proof and inserting a sorry as instructed
  sorry
}

end expected_losses_correct_l113_113224


namespace triangle_inequality_l113_113877

variable {a b c : ℝ}

theorem triangle_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (habc1 : a + b > c) (habc2 : a + c > b) (habc3 : b + c > a) :
  (a / (b + c) + b / (c + a) + c / (a + b) < 2) :=
sorry

end triangle_inequality_l113_113877


namespace min_fence_posts_needed_l113_113413

-- Definitions for the problem conditions
def area_length : ℕ := 72
def regular_side : ℕ := 30
def sloped_side : ℕ := 33
def interval : ℕ := 15

-- The property we want to prove
theorem min_fence_posts_needed : 3 * ((sloped_side + interval - 1) / interval) + 3 * ((regular_side + interval - 1) / interval) = 6 := 
by
  sorry

end min_fence_posts_needed_l113_113413


namespace sum_of_three_consecutive_even_integers_l113_113225

theorem sum_of_three_consecutive_even_integers : 
  ∃ (n : ℤ), n * (n + 2) * (n + 4) = 480 → n + (n + 2) + (n + 4) = 24 :=
by
  sorry

end sum_of_three_consecutive_even_integers_l113_113225


namespace evaluate_expression_l113_113791

theorem evaluate_expression (a b : ℕ) (h1 : a = 3) (h2 : b = 2) :
  (a^3 + b)^2 - (a^3 - b)^2 = 216 :=
by sorry

end evaluate_expression_l113_113791


namespace total_servings_l113_113174

-- Definitions for the conditions

def servings_per_carrot : ℕ := 4
def plants_per_plot : ℕ := 9
def servings_multiplier_corn : ℕ := 5
def servings_multiplier_green_bean : ℤ := 2

-- Proof statement
theorem total_servings : 
  (plants_per_plot * servings_per_carrot) + 
  (plants_per_plot * (servings_per_carrot * servings_multiplier_corn)) + 
  (plants_per_plot * (servings_per_carrot * servings_multiplier_corn / servings_multiplier_green_bean)) = 
  306 :=
by
  sorry

end total_servings_l113_113174


namespace arctan_arcsin_arccos_sum_l113_113219

theorem arctan_arcsin_arccos_sum :
  (Real.arctan (Real.sqrt 3 / 3) + Real.arcsin (-1 / 2) + Real.arccos 1 = 0) :=
by
  sorry

end arctan_arcsin_arccos_sum_l113_113219


namespace right_triangle_hypotenuse_segment_ratio_l113_113304

theorem right_triangle_hypotenuse_segment_ratio
  (x : ℝ) (AB BC AC BD AD CD : ℝ)
  (h1 : AB = 4 * x) 
  (h2 : BC = 3 * x) 
  (h3 : AC = 5 * x) 
  (h4 : (BD ^ 2) = AD * CD) :
  (CD / AD) = (16 / 9) :=
by
  sorry

end right_triangle_hypotenuse_segment_ratio_l113_113304


namespace find_ab_l113_113629

theorem find_ab (a b : ℝ) (h₁ : a - b = 3) (h₂ : a^2 + b^2 = 29) : a * b = 10 :=
sorry

end find_ab_l113_113629


namespace Felicity_family_store_visits_l113_113514

theorem Felicity_family_store_visits
  (lollipop_stick : ℕ := 1)
  (fort_total_sticks : ℕ := 400)
  (fort_completion_percent : ℕ := 60)
  (weeks_collected : ℕ := 80)
  (sticks_collected : ℕ := (fort_total_sticks * fort_completion_percent) / 100)
  (store_visits_per_week : ℕ := sticks_collected / weeks_collected) :
  store_visits_per_week = 3 := by
  sorry

end Felicity_family_store_visits_l113_113514


namespace right_triangle_distance_l113_113116

theorem right_triangle_distance (x h d : ℝ) :
  x + Real.sqrt ((x + 2 * h) ^ 2 + d ^ 2) = 2 * h + d → 
  x = (h * d) / (2 * h + d) :=
by
  intros h_eq_d
  sorry

end right_triangle_distance_l113_113116


namespace ratio_of_a_b_l113_113298

-- Define the problem
theorem ratio_of_a_b (a b : ℝ) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it.
  sorry

end ratio_of_a_b_l113_113298


namespace factorization_solution_1_factorization_solution_2_factorization_solution_3_l113_113263

noncomputable def factorization_problem_1 (m : ℝ) : Prop :=
  -3 * m^3 + 12 * m = -3 * m * (m + 2) * (m - 2)

noncomputable def factorization_problem_2 (x y : ℝ) : Prop :=
  2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2

noncomputable def factorization_problem_3 (a : ℝ) : Prop :=
  a^4 + 3 * a^2 - 4 = (a^2 + 4) * (a + 1) * (a - 1)

-- Lean statements for the proofs
theorem factorization_solution_1 (m : ℝ) : factorization_problem_1 m :=
  by sorry

theorem factorization_solution_2 (x y : ℝ) : factorization_problem_2 x y :=
  by sorry

theorem factorization_solution_3 (a : ℝ) : factorization_problem_3 a :=
  by sorry

end factorization_solution_1_factorization_solution_2_factorization_solution_3_l113_113263


namespace pen_distribution_l113_113692

theorem pen_distribution:
  (∃ (fountain: ℕ) (ballpoint: ℕ), fountain = 2 ∧ ballpoint = 3) ∧
  (∃ (students: ℕ), students = 4) →
  (∀ (s: ℕ), s ≥ 1 → s ≤ 4) →
  ∃ (ways: ℕ), ways = 28 :=
by
  sorry

end pen_distribution_l113_113692


namespace grasshopper_total_distance_l113_113386

theorem grasshopper_total_distance :
  let initial := 2
  let first_jump := -3
  let second_jump := 8
  let final_jump := -1
  abs (first_jump - initial) + abs (second_jump - first_jump) + abs (final_jump - second_jump) = 25 :=
by
  sorry

end grasshopper_total_distance_l113_113386


namespace length_after_haircut_l113_113050

-- Definitions
def original_length : ℕ := 18
def cut_length : ℕ := 9

-- Target statement to prove
theorem length_after_haircut : original_length - cut_length = 9 :=
by
  -- Simplification and proof
  sorry

end length_after_haircut_l113_113050


namespace inequality_solution_l113_113229

theorem inequality_solution (x : ℝ) : 
    (x - 5) / 2 + 1 > x - 3 → x < 3 := 
by 
    sorry

end inequality_solution_l113_113229


namespace max_additional_pies_l113_113730

theorem max_additional_pies (initial_cherries used_cherries cherries_per_pie : ℕ) 
  (h₀ : initial_cherries = 500) 
  (h₁ : used_cherries = 350) 
  (h₂ : cherries_per_pie = 35) :
  (initial_cherries - used_cherries) / cherries_per_pie = 4 := 
by
  sorry

end max_additional_pies_l113_113730


namespace sum_of_cubes_l113_113728

theorem sum_of_cubes {x y : ℝ} (h₁ : x + y = 0) (h₂ : x * y = -1) : x^3 + y^3 = 0 :=
by
  sorry

end sum_of_cubes_l113_113728


namespace smallest_c_for_inverse_l113_113554

def f (x : ℝ) : ℝ := (x - 3)^2 + 4

theorem smallest_c_for_inverse :
  ∃ c, (∀ x₁ x₂, (c ≤ x₁ ∧ c ≤ x₂ ∧ f x₁ = f x₂) → x₁ = x₂) ∧
       (∀ d, (∀ x₁ x₂, (d ≤ x₁ ∧ d ≤ x₂ ∧ f x₁ = f x₂) → x₁ = x₂) → c ≤ d) ∧
       c = 3 := sorry

end smallest_c_for_inverse_l113_113554


namespace repetitions_today_l113_113118

theorem repetitions_today (yesterday_reps : ℕ) (deficit : ℤ) (today_reps : ℕ) : 
  yesterday_reps = 86 ∧ deficit = -13 → 
  today_reps = yesterday_reps + deficit →
  today_reps = 73 :=
by
  intros
  sorry

end repetitions_today_l113_113118


namespace anthony_balloon_count_l113_113958

variable (Tom Luke Anthony : ℕ)

theorem anthony_balloon_count
  (h1 : Tom = 3 * Luke)
  (h2 : Luke = Anthony / 4)
  (hTom : Tom = 33) :
  Anthony = 44 := by
    sorry

end anthony_balloon_count_l113_113958


namespace option_a_is_correct_l113_113106

theorem option_a_is_correct (a b : ℝ) : 
  (a^2 + a * b) / a = a + b := 
by sorry

end option_a_is_correct_l113_113106


namespace value_20_percent_greater_l113_113960

theorem value_20_percent_greater (x : ℝ) : (x = 88 * 1.20) ↔ (x = 105.6) :=
by
  sorry

end value_20_percent_greater_l113_113960


namespace div_c_a_l113_113057

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a / b = 3)
variable (h2 : b / c = 2 / 5)

-- State the theorem to be proven
theorem div_c_a (a b c : ℝ) (h1 : a / b = 3) (h2 : b / c = 2 / 5) : c / a = 5 / 6 := 
by 
  sorry

end div_c_a_l113_113057


namespace scientific_notation_3080000_l113_113370

theorem scientific_notation_3080000 : (∃ (a : ℝ) (b : ℤ), 1 ≤ a ∧ a < 10 ∧ (3080000 : ℝ) = a * 10^b) ∧ (3080000 : ℝ) = 3.08 * 10^6 :=
by
  sorry

end scientific_notation_3080000_l113_113370


namespace complex_number_solution_l113_113088

theorem complex_number_solution {i z : ℂ} (h : (2 : ℂ) / (1 + i) = z + i) : z = 1 + 2 * i :=
sorry

end complex_number_solution_l113_113088


namespace bike_trike_race_l113_113990

theorem bike_trike_race (P : ℕ) (B T : ℕ) (h1 : B = (3 * P) / 5) (h2 : T = (2 * P) / 5) (h3 : 2 * B + 3 * T = 96) :
  P = 40 :=
by
  sorry

end bike_trike_race_l113_113990


namespace sales_tax_difference_l113_113334

noncomputable def price_before_tax : ℝ := 50
noncomputable def sales_tax_rate_7_5_percent : ℝ := 0.075
noncomputable def sales_tax_rate_8_percent : ℝ := 0.08

theorem sales_tax_difference :
  (price_before_tax * sales_tax_rate_8_percent) - (price_before_tax * sales_tax_rate_7_5_percent) = 0.25 :=
by
  sorry

end sales_tax_difference_l113_113334


namespace age_hence_l113_113059

theorem age_hence (A x : ℕ) (h1 : A = 50)
  (h2 : 5 * (A + x) - 5 * (A - 5) = A) : x = 5 :=
by sorry

end age_hence_l113_113059


namespace union_sets_l113_113192

open Set

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {2, 4, 5, 6}

theorem union_sets : (A ∪ B) = {1, 2, 3, 4, 5, 6} :=
by
  sorry

end union_sets_l113_113192


namespace initial_weight_of_fish_l113_113362

theorem initial_weight_of_fish (B F : ℝ) 
  (h1 : B + F = 54) 
  (h2 : B + F / 2 = 29) : 
  F = 50 := 
sorry

end initial_weight_of_fish_l113_113362


namespace ln_sqrt2_lt_sqrt2_div2_ln_sin_cos_sum_l113_113519

theorem ln_sqrt2_lt_sqrt2_div2 : Real.log (Real.sqrt 2) < Real.sqrt 2 / 2 :=
sorry

theorem ln_sin_cos_sum : 2 * Real.log (Real.sin (1/8) + Real.cos (1/8)) < 1 / 4 :=
sorry

end ln_sqrt2_lt_sqrt2_div2_ln_sin_cos_sum_l113_113519


namespace log_eval_l113_113083

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_eval : log_base (Real.sqrt 10) (1000 * Real.sqrt 10) = 7 := sorry

end log_eval_l113_113083


namespace sum_a_b_eq_five_l113_113261

theorem sum_a_b_eq_five (a b : ℝ) (h : ∀ x : ℝ, 1 < x ∧ x < 2 → x^2 - a * x + b < 0) : a + b = 5 :=
sorry

end sum_a_b_eq_five_l113_113261


namespace total_cost_ice_cream_l113_113010

noncomputable def price_Chocolate : ℝ := 2.50
noncomputable def price_Vanilla : ℝ := 2.00
noncomputable def price_Strawberry : ℝ := 2.25
noncomputable def price_Mint : ℝ := 2.20
noncomputable def price_WaffleCone : ℝ := 1.50
noncomputable def price_ChocolateChips : ℝ := 1.00
noncomputable def price_Fudge : ℝ := 1.25
noncomputable def price_WhippedCream : ℝ := 0.75

def scoops_Pierre : ℕ := 3  -- 2 scoops Chocolate + 1 scoop Mint
def scoops_Mother : ℕ := 4  -- 2 scoops Vanilla + 1 scoop Strawberry + 1 scoop Mint

noncomputable def price_Pierre_BeforeOffer : ℝ :=
  2 * price_Chocolate + price_Mint + price_WaffleCone + price_ChocolateChips

noncomputable def free_Pierre : ℝ := price_Mint -- Mint is the cheapest among Pierre's choices

noncomputable def price_Pierre_AfterOffer : ℝ := price_Pierre_BeforeOffer - free_Pierre

noncomputable def price_Mother_BeforeOffer : ℝ :=
  2 * price_Vanilla + price_Strawberry + price_Mint + price_WaffleCone + price_Fudge + price_WhippedCream

noncomputable def free_Mother : ℝ := price_Vanilla -- Vanilla is the cheapest among Mother's choices

noncomputable def price_Mother_AfterOffer : ℝ := price_Mother_BeforeOffer - free_Mother

noncomputable def total_BeforeDiscount : ℝ := price_Pierre_AfterOffer + price_Mother_AfterOffer

noncomputable def discount_Amount : ℝ := total_BeforeDiscount * 0.15

noncomputable def total_AfterDiscount : ℝ := total_BeforeDiscount - discount_Amount

theorem total_cost_ice_cream : total_AfterDiscount = 14.83 := by
  sorry


end total_cost_ice_cream_l113_113010


namespace income_max_takehome_pay_l113_113866

theorem income_max_takehome_pay :
  ∃ x : ℝ, (∀ y : ℝ, 1000 * y - 5 * y^2 ≤ 1000 * x - 5 * x^2) ∧ x = 100 :=
by
  sorry

end income_max_takehome_pay_l113_113866


namespace bills_needed_can_pay_groceries_l113_113246

theorem bills_needed_can_pay_groceries 
  (cans_of_soup : ℕ := 6) (price_per_can : ℕ := 2)
  (loaves_of_bread : ℕ := 3) (price_per_loaf : ℕ := 5)
  (boxes_of_cereal : ℕ := 4) (price_per_box : ℕ := 3)
  (gallons_of_milk : ℕ := 2) (price_per_gallon : ℕ := 4)
  (apples : ℕ := 7) (price_per_apple : ℕ := 1)
  (bags_of_cookies : ℕ := 5) (price_per_bag : ℕ := 3)
  (bottles_of_olive_oil : ℕ := 1) (price_per_bottle : ℕ := 8)
  : ∃ (bills_needed : ℕ), bills_needed = 4 :=
by
  let total_cost := (cans_of_soup * price_per_can) + 
                    (loaves_of_bread * price_per_loaf) +
                    (boxes_of_cereal * price_per_box) +
                    (gallons_of_milk * price_per_gallon) +
                    (apples * price_per_apple) +
                    (bags_of_cookies * price_per_bag) +
                    (bottles_of_olive_oil * price_per_bottle)
  let bills_needed := (total_cost + 19) / 20   -- Calculating ceiling of total_cost / 20
  sorry

end bills_needed_can_pay_groceries_l113_113246


namespace sum_of_digits_is_8_l113_113828

theorem sum_of_digits_is_8 (d : ℤ) (h1 : d ≥ 0)
  (h2 : 8 * d / 5 - 80 = d) : (d / 100) + ((d % 100) / 10) + (d % 10) = 8 :=
by
  sorry

end sum_of_digits_is_8_l113_113828


namespace symmetric_conic_transform_l113_113913

open Real

theorem symmetric_conic_transform (x y : ℝ) 
  (h1 : 2 * x^2 + 4 * x * y + 5 * y^2 - 22 = 0)
  (h2 : x - y + 1 = 0) : 
  5 * x^2 + 4 * x * y + 2 * y^2 + 6 * x - 19 = 0 := 
sorry

end symmetric_conic_transform_l113_113913


namespace max_regions_1002_1000_l113_113842

def regions_through_point (n : ℕ) : ℕ := (n * (n + 1)) / 2 + 1

def max_regions (a b : ℕ) : ℕ := 
  let rB := regions_through_point b
  let first_line_through_A := rB + b + 1
  let remaining_lines_through_A := (a - 1) * (b + 2)
  first_line_through_A + remaining_lines_through_A

theorem max_regions_1002_1000 : max_regions 1002 1000 = 1504503 := by
  sorry

end max_regions_1002_1000_l113_113842


namespace like_terms_sum_three_l113_113173

theorem like_terms_sum_three (m n : ℤ) (h1 : 2 * m = 4 - n) (h2 : m = n - 1) : m + n = 3 :=
sorry

end like_terms_sum_three_l113_113173


namespace solve_fractional_equation_l113_113694

-- Define the fractional equation as a function
def fractional_equation (x : ℝ) : Prop :=
  (3 / 2) - (2 * x) / (3 * x - 1) = 7 / (6 * x - 2)

-- State the theorem we need to prove
theorem solve_fractional_equation : fractional_equation 2 :=
by
  -- Placeholder for proof
  sorry

end solve_fractional_equation_l113_113694


namespace arithmetic_series_sum_l113_113975

theorem arithmetic_series_sum : 
  let a := -41
  let d := 2
  let n := 22
  let l := 1
  let Sn := n * (a + l) / 2
  a = -41 ∧ d = 2 ∧ l = 1 ∧ n = 22 → Sn = -440 :=
by 
  intros a d n l Sn h
  sorry

end arithmetic_series_sum_l113_113975


namespace geography_book_price_l113_113139

open Real

-- Define the problem parameters
def num_english_books : ℕ := 35
def num_geography_books : ℕ := 35
def cost_english : ℝ := 7.50
def total_cost : ℝ := 630.00

-- Define the unknown we need to prove
def cost_geography : ℝ := 10.50

theorem geography_book_price :
  num_english_books * cost_english + num_geography_books * cost_geography = total_cost :=
by
  -- No need to include the proof steps
  sorry

end geography_book_price_l113_113139


namespace aunt_masha_butter_usage_l113_113397

theorem aunt_masha_butter_usage
  (x y : ℝ)
  (h1 : x + 10 * y = 600)
  (h2 : x = 5 * y) :
  (2 * x + 2 * y = 480) := 
by
  sorry

end aunt_masha_butter_usage_l113_113397


namespace peak_infection_day_l113_113132

-- Given conditions
def initial_cases : Nat := 20
def increase_rate : Nat := 50
def decrease_rate : Nat := 30
def total_infections : Nat := 8670
def total_days : Nat := 30

-- Peak Day and infections on that day
def peak_day : Nat := 12

-- Theorem stating what we want to prove
theorem peak_infection_day :
  ∃ n : Nat, n = initial_cases + increase_rate * (peak_day - 1) - decrease_rate * (30 - peak_day) :=
sorry

end peak_infection_day_l113_113132


namespace isabella_more_than_sam_l113_113199

variable (I S G : ℕ)

def Giselle_money : G = 120 := by sorry
def Isabella_more_than_Giselle : I = G + 15 := by sorry
def total_donation : I + S + G = 345 := by sorry

theorem isabella_more_than_sam : I - S = 45 := by
sorry

end isabella_more_than_sam_l113_113199


namespace part1_part2_l113_113392

-- Definition of sets A and B
def A (a : ℝ) : Set ℝ := {x | a-1 < x ∧ x < a+1}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

-- Theorem (Ⅰ)
theorem part1 (a : ℝ) : (A a ∩ B = ∅ ∧ A a ∪ B = Set.univ) → a = 2 :=
by
  sorry

-- Theorem (Ⅱ)
theorem part2 (a : ℝ) : (A a ⊆ B ∧ A a ≠ ∅) → (a ≤ 0 ∨ a ≥ 4) :=
by
  sorry

end part1_part2_l113_113392


namespace find_a_l113_113321

/-- The random variable ξ takes on all possible values 1, 2, 3, 4, 5,
and P(ξ = k) = a * k for k = 1, 2, 3, 4, 5. Given that the sum 
of probabilities for all possible outcomes of a discrete random
variable equals 1, find the value of a. -/
theorem find_a (a : ℝ) 
  (h : (a * 1) + (a * 2) + (a * 3) + (a * 4) + (a * 5) = 1) : 
  a = 1 / 15 :=
sorry

end find_a_l113_113321


namespace inequality_proof_l113_113552

theorem inequality_proof (x y z : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) (hz : 1 ≤ z) 
  (h : 1 / (x^2 - 1) + 1 / (y^2 - 1) + 1 / (z^2 - 1) = 1) :
  1 / (x + 1) + 1 / (y + 1) + 1 / (z + 1) ≤ 1 := 
by {
  sorry
}

end inequality_proof_l113_113552


namespace clarence_oranges_l113_113563

def initial_oranges := 5
def oranges_from_joyce := 3
def total_oranges := initial_oranges + oranges_from_joyce

theorem clarence_oranges : total_oranges = 8 :=
  by
  sorry

end clarence_oranges_l113_113563


namespace muffins_in_each_pack_l113_113367

-- Define the conditions as constants
def total_amount_needed : ℕ := 120
def price_per_muffin : ℕ := 2
def number_of_cases : ℕ := 5
def packs_per_case : ℕ := 3

-- Define the theorem to prove
theorem muffins_in_each_pack :
  (total_amount_needed / price_per_muffin) / (number_of_cases * packs_per_case) = 4 :=
by
  sorry

end muffins_in_each_pack_l113_113367


namespace solve_for_a_l113_113805

open Complex

theorem solve_for_a (a : ℝ) (h : ∃ x : ℝ, (2 * Complex.I - (a * Complex.I) / (1 - Complex.I) = x)) : a = 4 := 
sorry

end solve_for_a_l113_113805


namespace no_sol_x_y_pos_int_eq_2015_l113_113086

theorem no_sol_x_y_pos_int_eq_2015 (x y : ℕ) (hx : x > 0) (hy : y > 0) : ¬ (x^2 - y! = 2015) :=
sorry

end no_sol_x_y_pos_int_eq_2015_l113_113086


namespace groceries_delivered_l113_113456

variables (S C P g T G : ℝ)
theorem groceries_delivered (hS : S = 14500) (hC : C = 14600) (hP : P = 1.5) (hg : g = 0.05) (hT : T = 40) :
  G = 800 :=
by {
  sorry
}

end groceries_delivered_l113_113456


namespace wall_length_l113_113625

theorem wall_length (s : ℕ) (w : ℕ) (a_ratio : ℕ) (A_mirror : ℕ) (A_wall : ℕ) (L : ℕ) 
  (hs : s = 24) (hw : w = 42) (h_ratio : a_ratio = 2) 
  (hA_mirror : A_mirror = s * s) 
  (hA_wall : A_wall = A_mirror * a_ratio) 
  (h_area : A_wall = w * L) : L = 27 :=
  sorry

end wall_length_l113_113625


namespace find_k_and_general_term_l113_113097

noncomputable def sum_of_first_n_terms (n k : ℝ) : ℝ :=
  -n^2 + (10 + k) * n + (k - 1)

noncomputable def general_term (n : ℕ) : ℝ :=
  -2 * n + 12

theorem find_k_and_general_term :
  (∀ n k : ℝ, sum_of_first_n_terms n k = sum_of_first_n_terms n (1 : ℝ)) ∧
  (∀ n : ℕ, ∃ an : ℝ, an = general_term n) :=
by
  sorry

end find_k_and_general_term_l113_113097


namespace sum_of_cubes_is_81720_l113_113396

-- Let n be the smallest of these consecutive even integers.
def smallest_even : Int := 28

-- Assumptions given the conditions
def sum_of_squares (n : Int) : Int := n^2 + (n + 2)^2 + (n + 4)^2

-- The condition provided is that sum of the squares is 2930
lemma sum_of_squares_is_2930 : sum_of_squares smallest_even = 2930 := by
  sorry

-- To prove that the sum of the cubes of these three integers is 81720
def sum_of_cubes (n : Int) : Int := n^3 + (n + 2)^3 + (n + 4)^3

theorem sum_of_cubes_is_81720 : sum_of_cubes smallest_even = 81720 := by
  sorry

end sum_of_cubes_is_81720_l113_113396


namespace garden_area_l113_113104

-- Given conditions:
def width := 16
def length (W : ℕ) := 3 * W

-- Proof statement:
theorem garden_area (W : ℕ) (hW : W = width) : length W * W = 768 :=
by
  rw [hW]
  exact rfl

end garden_area_l113_113104


namespace jane_emily_total_accessories_l113_113959

def total_accessories : ℕ :=
  let jane_dresses := 4 * 10
  let emily_dresses := 3 * 8
  let jane_ribbons := 3 * jane_dresses
  let jane_buttons := 2 * jane_dresses
  let jane_lace_trims := 1 * jane_dresses
  let jane_beads := 4 * jane_dresses
  let emily_ribbons := 2 * emily_dresses
  let emily_buttons := 3 * emily_dresses
  let emily_lace_trims := 2 * emily_dresses
  let emily_beads := 5 * emily_dresses
  let emily_bows := 1 * emily_dresses
  jane_ribbons + jane_buttons + jane_lace_trims + jane_beads +
  emily_ribbons + emily_buttons + emily_lace_trims + emily_beads + emily_bows 

theorem jane_emily_total_accessories : total_accessories = 712 := 
by
  sorry

end jane_emily_total_accessories_l113_113959


namespace eccentricity_of_hyperbola_l113_113646

noncomputable def hyperbola_eccentricity : ℝ → ℝ → ℝ → ℝ
| p, a, b => 
  let c := p / 2
  let e := c / a
  have h₁ : 9 * e^2 - 12 * e^2 / (e^2 - 1) = 1 := sorry
  e

theorem eccentricity_of_hyperbola (p a b : ℝ) (hp : p > 0) (ha : a > 0) (hb : b > 0) :
  hyperbola_eccentricity p a b = (Real.sqrt 7 + 2) / 3 :=
sorry

end eccentricity_of_hyperbola_l113_113646


namespace fractional_equation_solution_l113_113672

theorem fractional_equation_solution (x : ℝ) (h : x = 7) : (3 / (x - 3)) - 1 = 1 / (3 - x) := by
  sorry

end fractional_equation_solution_l113_113672


namespace quadratic_eq_is_general_form_l113_113685

def quadratic_eq_general_form (x : ℝ) : Prop :=
  x^2 - 2 * (3 * x - 2) + (x + 1) = x^2 - 5 * x + 5

theorem quadratic_eq_is_general_form :
  quadratic_eq_general_form x :=
sorry

end quadratic_eq_is_general_form_l113_113685


namespace time_to_sell_all_cars_l113_113120

/-- Conditions: -/
def total_cars : ℕ := 500
def number_of_sales_professionals : ℕ := 10
def cars_per_salesperson_per_month : ℕ := 10

/-- Proof Statement: -/
theorem time_to_sell_all_cars 
  (total_cars : ℕ) 
  (number_of_sales_professionals : ℕ) 
  (cars_per_salesperson_per_month : ℕ) : 
  ((number_of_sales_professionals * cars_per_salesperson_per_month) > 0) →
  (total_cars / (number_of_sales_professionals * cars_per_salesperson_per_month)) = 5 :=
by
  sorry

end time_to_sell_all_cars_l113_113120


namespace watch_cost_price_l113_113161

open Real

theorem watch_cost_price (CP SP1 SP2 : ℝ)
    (h1 : SP1 = CP * 0.85)
    (h2 : SP2 = CP * 1.10)
    (h3 : SP2 = SP1 + 450) : CP = 1800 :=
by
  sorry

end watch_cost_price_l113_113161


namespace find_b_l113_113841

noncomputable def f (b x : ℝ) : ℝ :=
if x < 1 then 2 * x - b else 2 ^ x

theorem find_b (b : ℝ) (h : f b (f b (1 / 2)) = 4) : b = -1 :=
sorry

end find_b_l113_113841


namespace find_f_2547_l113_113384

theorem find_f_2547 (f : ℚ → ℚ)
  (h1 : ∀ x y : ℚ, f (x + y) = f x + f y + 2547)
  (h2 : f 2004 = 2547) :
  f 2547 = 2547 :=
sorry

end find_f_2547_l113_113384


namespace chess_tournament_total_players_l113_113987

theorem chess_tournament_total_players :
  ∃ (n: ℕ), 
    (∀ (players: ℕ) (points: ℕ -> ℕ), 
      (players = n + 15) ∧
      (∀ p, points p = points p / 2 + points p / 2) ∧
      (∀ i < 15, ∀ j < 15, points i = points j / 2) → 
      players = 36) :=
by
  sorry

end chess_tournament_total_players_l113_113987


namespace smallest_whole_number_larger_than_perimeter_l113_113040

theorem smallest_whole_number_larger_than_perimeter (s : ℝ) (h1 : 7 + 23 > s) (h2 : 7 + s > 23) (h3 : 23 + s > 7) : 
  60 = Int.ceil (7 + 23 + s - 1) :=
by
  sorry

end smallest_whole_number_larger_than_perimeter_l113_113040


namespace mod_multiplication_l113_113049

theorem mod_multiplication :
  (176 * 929) % 50 = 4 :=
by
  sorry

end mod_multiplication_l113_113049


namespace four_mutually_acquainted_l113_113507

theorem four_mutually_acquainted (G : SimpleGraph (Fin 9)) 
  (h : ∀ (s : Finset (Fin 9)), s.card = 3 → ∃ (u v : Fin 9), u ∈ s ∧ v ∈ s ∧ G.Adj u v) :
  ∃ (s : Finset (Fin 9)), s.card = 4 ∧ ∀ (u v : Fin 9), u ∈ s → v ∈ s → G.Adj u v :=
by
  sorry

end four_mutually_acquainted_l113_113507


namespace alcohol_to_water_ratio_l113_113749

theorem alcohol_to_water_ratio (alcohol water : ℚ) (h_alcohol : alcohol = 2/7) (h_water : water = 3/7) : alcohol / water = 2 / 3 := by
  sorry

end alcohol_to_water_ratio_l113_113749


namespace reflected_circle_center_l113_113423

theorem reflected_circle_center
  (original_center : ℝ × ℝ) 
  (reflection_line : ℝ × ℝ → ℝ × ℝ)
  (hc : original_center = (8, -3))
  (hl : ∀ (p : ℝ × ℝ), reflection_line p = (-p.2, -p.1))
  : reflection_line original_center = (3, -8) :=
sorry

end reflected_circle_center_l113_113423


namespace find_x_l113_113408

theorem find_x (x : ℝ) (h : x^2 + 75 = (x - 20)^2) : x = 8.125 :=
by
  sorry

end find_x_l113_113408


namespace rhombus_diagonal_l113_113266

theorem rhombus_diagonal (d2 : ℝ) (area : ℝ) (d1 : ℝ) : d2 = 15 → area = 127.5 → d1 = 17 :=
by
  intros h1 h2
  sorry

end rhombus_diagonal_l113_113266


namespace smallest_b_of_factored_quadratic_l113_113394

theorem smallest_b_of_factored_quadratic (r s : ℕ) (h1 : r * s = 1620) : (r + s) = 84 :=
sorry

end smallest_b_of_factored_quadratic_l113_113394


namespace find_ab_pairs_l113_113661

theorem find_ab_pairs (a b s : ℕ) (a_pos : a > 0) (b_pos : b > 0) (s_gt_one : s > 1) :
  (a = 2^s ∧ b = 2^(2*s) - 1) ↔
  (∃ p k : ℕ, Prime p ∧ (a^2 + b + 1 = p^k) ∧
   (a^2 + b + 1 ∣ b^2 - a^3 - 1) ∧
   ¬ (a^2 + b + 1 ∣ (a + b - 1)^2)) :=
sorry

end find_ab_pairs_l113_113661


namespace sum_of_ages_is_59_l113_113076

variable (juliet maggie ralph nicky lucy lily alex : ℕ)

def juliet_age := 10
def maggie_age := juliet_age - 3
def ralph_age := juliet_age + 2
def nicky_age := ralph_age / 2
def lucy_age := ralph_age + 1
def lily_age := ralph_age + 1
def alex_age := lucy_age - 5

theorem sum_of_ages_is_59 :
  maggie_age + ralph_age + nicky_age + lucy_age + lily_age + alex_age = 59 :=
by
  let maggie := 7
  let ralph := 12
  let nicky := 6
  let lucy := 13
  let lily := 13
  let alex := 8
  show maggie + ralph + nicky + lucy + lily + alex = 59
  sorry

end sum_of_ages_is_59_l113_113076


namespace inequality_integral_ln_bounds_l113_113619

-- Define the conditions
variables (x a : ℝ)
variables (hx : 0 < x) (ha : x < a)

-- First part: inequality involving integral
theorem inequality_integral (hx : 0 < x) (ha : x < a) :
  (2 * x / a) < (∫ t in a - x..a + x, 1 / t) ∧ (∫ t in a - x..a + x, 1 / t) < x * (1 / (a + x) + 1 / (a - x)) :=
sorry

-- Second part: to prove 0.68 < ln(2) < 0.71 using the result of the first part
theorem ln_bounds :
  0.68 < Real.log 2 ∧ Real.log 2 < 0.71 :=
sorry

end inequality_integral_ln_bounds_l113_113619


namespace sequences_properties_l113_113341

-- Definitions based on the problem conditions
def geom_sequence (a : ℕ → ℕ) := ∃ q : ℕ, a 1 = 2 ∧ a 3 = 18 ∧ ∀ n, a (n + 1) = a n * q
def arith_sequence (b : ℕ → ℕ) := b 1 = 2 ∧ ∃ d : ℕ, ∀ n, b (n + 1) = b n + d
def condition (a : ℕ → ℕ) (b : ℕ → ℕ) := a 1 + a 2 + a 3 > 20 ∧ a 1 + a 2 + a 3 = b 1 + b 2 + b 3 + b 4

-- Proof statement: proving the general term of the geometric sequence and the sum of the arithmetic sequence
theorem sequences_properties (a : ℕ → ℕ) (b : ℕ → ℕ) :
  geom_sequence a → arith_sequence b → condition a b →
  (∀ n, a n = 2 * 3^(n - 1)) ∧ (∀ n, S_n = 3 / 2 * n^2 + 1 / 2 * n) :=
by
  sorry

end sequences_properties_l113_113341


namespace unique_function_satisfying_conditions_l113_113898

open Nat

def satisfies_conditions (f : ℕ → ℕ) : Prop :=
  (f 1 = 1) ∧ (∀ n, f n * f (n + 2) = (f (n + 1))^2 + 1997)

theorem unique_function_satisfying_conditions :
  (∃! f : ℕ → ℕ, satisfies_conditions f) :=
sorry

end unique_function_satisfying_conditions_l113_113898


namespace division_problem_l113_113649

theorem division_problem : 75 / 0.05 = 1500 := 
  sorry

end division_problem_l113_113649


namespace fraction_problem_l113_113908

theorem fraction_problem :
  ((3 / 4 - 5 / 8) / 2) = 1 / 16 :=
by
  sorry

end fraction_problem_l113_113908


namespace snack_eaters_remaining_l113_113608

noncomputable def initial_snack_eaters := 5000 * 60 / 100
noncomputable def snack_eaters_after_1_hour := initial_snack_eaters + 25
noncomputable def snack_eaters_after_70_percent_left := snack_eaters_after_1_hour * 30 / 100
noncomputable def snack_eaters_after_2_hour := snack_eaters_after_70_percent_left + 50
noncomputable def snack_eaters_after_800_left := snack_eaters_after_2_hour - 800
noncomputable def snack_eaters_after_2_thirds_left := snack_eaters_after_800_left * 1 / 3
noncomputable def final_snack_eaters := snack_eaters_after_2_thirds_left + 100

theorem snack_eaters_remaining : final_snack_eaters = 153 :=
by
  have h1 : initial_snack_eaters = 3000 := by sorry
  have h2 : snack_eaters_after_1_hour = initial_snack_eaters + 25 := by sorry
  have h3 : snack_eaters_after_70_percent_left = snack_eaters_after_1_hour * 30 / 100 := by sorry
  have h4 : snack_eaters_after_2_hour = snack_eaters_after_70_percent_left + 50 := by sorry
  have h5 : snack_eaters_after_800_left = snack_eaters_after_2_hour - 800 := by sorry
  have h6 : snack_eaters_after_2_thirds_left = snack_eaters_after_800_left * 1 / 3 := by sorry
  have h7 : final_snack_eaters = snack_eaters_after_2_thirds_left + 100 := by sorry
  -- Prove that these equal 153 overall
  sorry

end snack_eaters_remaining_l113_113608


namespace paul_money_last_weeks_l113_113650

theorem paul_money_last_weeks (a b c: ℕ) (h1: a = 68) (h2: b = 13) (h3: c = 9) : 
  (a + b) / c = 9 := 
by 
  sorry

end paul_money_last_weeks_l113_113650


namespace five_by_five_rectangles_l113_113325

def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem five_by_five_rectangles : (choose 5 2) * (choose 5 2) = 100 :=
by
  sorry

end five_by_five_rectangles_l113_113325


namespace congruent_triangle_sides_l113_113100

variable {x y : ℕ}

theorem congruent_triangle_sides (h_congruent : ∃ (a b c d e f : ℕ), (a = x) ∧ (b = 2) ∧ (c = 6) ∧ (d = 5) ∧ (e = 6) ∧ (f = y) ∧ (a = d) ∧ (b = f) ∧ (c = e)) : 
  x + y = 7 :=
sorry

end congruent_triangle_sides_l113_113100


namespace prob_diff_colors_correct_l113_113914

def total_chips := 6 + 5 + 4 + 3

def prob_diff_colors : ℚ :=
  (6 / total_chips * (12 / total_chips) +
  5 / total_chips * (13 / total_chips) +
  4 / total_chips * (14 / total_chips) +
  3 / total_chips * (15 / total_chips))

theorem prob_diff_colors_correct :
  prob_diff_colors = 119 / 162 := by
  sorry

end prob_diff_colors_correct_l113_113914


namespace min_a_plus_b_l113_113969

theorem min_a_plus_b (a b : ℝ) (h : a^2 + 2 * b^2 = 6) : a + b ≥ -3 :=
sorry

end min_a_plus_b_l113_113969


namespace linear_term_coefficient_l113_113678

theorem linear_term_coefficient : (x - 1) * (1 / x + x) ^ 6 = a + b * x + c * x^2 + d * x^3 + e * x^4 + f * x^5 + g * x^6 →
  b = 20 :=
by
  sorry

end linear_term_coefficient_l113_113678


namespace minimize_quadratic_expression_l113_113803

theorem minimize_quadratic_expression :
  ∃ x : ℝ, x = 3 ∧ ∀ y : ℝ, (y^2 - 6*y + 8) ≥ (x^2 - 6*x + 8) := by
sorry

end minimize_quadratic_expression_l113_113803


namespace sum_series_a_eq_one_sum_series_b_eq_half_sum_series_c_eq_third_l113_113363

noncomputable def sum_series_a : ℝ :=
∑' n, (1 / (n * (n + 1)))

noncomputable def sum_series_b : ℝ :=
∑' n, (1 / ((n + 1) * (n + 2)))

noncomputable def sum_series_c : ℝ :=
∑' n, (1 / ((n + 2) * (n + 3)))

theorem sum_series_a_eq_one : sum_series_a = 1 := sorry

theorem sum_series_b_eq_half : sum_series_b = 1 / 2 := sorry

theorem sum_series_c_eq_third : sum_series_c = 1 / 3 := sorry

end sum_series_a_eq_one_sum_series_b_eq_half_sum_series_c_eq_third_l113_113363


namespace magnitude_of_z_l113_113738

open Complex

noncomputable def z : ℂ := (1 - I) / (1 + I) + 2 * I

theorem magnitude_of_z : Complex.abs z = 1 := by
  sorry

end magnitude_of_z_l113_113738


namespace total_amount_l113_113382

variable (x y z : ℝ)

def condition1 : Prop := y = 0.45 * x
def condition2 : Prop := z = 0.30 * x
def condition3 : Prop := y = 36

theorem total_amount (h1 : condition1 x y)
                     (h2 : condition2 x z)
                     (h3 : condition3 y) :
  x + y + z = 140 :=
by
  sorry

end total_amount_l113_113382


namespace difference_between_numbers_l113_113526

theorem difference_between_numbers : 
  ∃ (a : ℕ), a + 10 * a = 30000 → 9 * a = 24543 := 
by 
  sorry

end difference_between_numbers_l113_113526


namespace find_payment_y_l113_113407

variable (X Y : Real)

axiom h1 : X + Y = 570
axiom h2 : X = 1.2 * Y

theorem find_payment_y : Y = 570 / 2.2 := by
  sorry

end find_payment_y_l113_113407


namespace not_prime_sum_l113_113029

theorem not_prime_sum (a b c d : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) (h_eq : a * b = c * d) : ¬ Nat.Prime (a + b + c + d) :=
sorry

end not_prime_sum_l113_113029


namespace floor_plus_r_eq_10_3_implies_r_eq_5_3_l113_113798

noncomputable def floor (x : ℝ) : ℤ := sorry -- Assuming the function exists

theorem floor_plus_r_eq_10_3_implies_r_eq_5_3 (r : ℝ) 
  (h : floor r + r = 10.3) : r = 5.3 :=
sorry

end floor_plus_r_eq_10_3_implies_r_eq_5_3_l113_113798


namespace math_problem_l113_113181

theorem math_problem : (300 + 5 * 8) / (2^3) = 42.5 := by
  sorry

end math_problem_l113_113181


namespace base_number_is_five_l113_113675

theorem base_number_is_five (x k : ℝ) (h1 : x^k = 5) (h2 : x^(2 * k + 2) = 400) : x = 5 :=
by
  sorry

end base_number_is_five_l113_113675


namespace cleaning_time_if_anne_doubled_l113_113632

-- Definitions based on conditions
def anne_rate := 1 / 12
def combined_rate := 1 / 4
def bruce_rate := combined_rate - anne_rate
def double_anne_rate := 2 * anne_rate
def doubled_combined_rate := bruce_rate + double_anne_rate

-- Statement of the problem
theorem cleaning_time_if_anne_doubled :  1 / doubled_combined_rate = 3 :=
by sorry

end cleaning_time_if_anne_doubled_l113_113632


namespace prime_square_minus_one_divisible_by_twelve_l113_113351

theorem prime_square_minus_one_divisible_by_twelve (p : ℕ) (hp : Nat.Prime p) (hp_gt : p > 3) : 12 ∣ (p^2 - 1) :=
by
  sorry

end prime_square_minus_one_divisible_by_twelve_l113_113351


namespace time_morning_is_one_l113_113858

variable (D : ℝ)  -- Define D as the distance between the two points.

def morning_speed := 20 -- Morning speed (km/h)
def afternoon_speed := 10 -- Afternoon speed (km/h)
def time_difference := 1 -- Time difference (hour)

-- Proving that the morning time t_m is equal to 1 hour
theorem time_morning_is_one (t_m t_a : ℝ) 
  (h1 : t_m - t_a = time_difference) 
  (h2 : D = morning_speed * t_m) 
  (h3 : D = afternoon_speed * t_a) : 
  t_m = 1 := 
by
  sorry

end time_morning_is_one_l113_113858


namespace isosceles_triangle_CBD_supplement_l113_113504

/-- Given an isosceles triangle ABC with AC = BC and angle C = 50 degrees,
    and point D such that angle CBD is supplementary to angle ABC,
    prove that angle CBD is 115 degrees. -/
theorem isosceles_triangle_CBD_supplement 
  (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]
  (AC BC : ℝ) (angleBAC angleABC angleC angleCBD : ℝ)
  (isosceles : AC = BC)
  (angle_C_eq : angleC = 50)
  (supplement : angleCBD = 180 - angleABC) :
  angleCBD = 115 :=
sorry

end isosceles_triangle_CBD_supplement_l113_113504


namespace inverse_function_log3_l113_113320

noncomputable def f (x : ℝ) : ℝ := Real.logb 3 x

theorem inverse_function_log3 :
  ∀ x : ℝ, x > 0 →
  ∃ y : ℝ, f (3 ^ y) = y := 
sorry

end inverse_function_log3_l113_113320


namespace nancy_tortilla_chips_l113_113620

theorem nancy_tortilla_chips :
  ∀ (total_chips chips_brother chips_herself chips_sister : ℕ),
    total_chips = 22 →
    chips_brother = 7 →
    chips_herself = 10 →
    chips_sister = total_chips - chips_brother - chips_herself →
    chips_sister = 5 :=
by
  intros total_chips chips_brother chips_herself chips_sister
  intro h_total h_brother h_herself h_sister
  rw [h_total, h_brother, h_herself] at h_sister
  simp at h_sister
  assumption

end nancy_tortilla_chips_l113_113620


namespace wall_bricks_count_l113_113101

def alice_rate (y : ℕ) : ℕ := y / 8
def bob_rate (y : ℕ) : ℕ := y / 12
def combined_rate (y : ℕ) : ℕ := (5 * y) / 24 - 12
def effective_working_time : ℕ := 6

theorem wall_bricks_count :
  ∃ y : ℕ, (combined_rate y * effective_working_time = y) ∧ y = 288 :=
by
  sorry

end wall_bricks_count_l113_113101


namespace range_of_a_if_slope_is_obtuse_l113_113595

theorem range_of_a_if_slope_is_obtuse : 
  ∀ a : ℝ, (a^2 + 2 * a < 0) → -2 < a ∧ a < 0 :=
by
  intro a
  intro h
  sorry

end range_of_a_if_slope_is_obtuse_l113_113595


namespace wife_weekly_savings_correct_l113_113265

-- Define constants
def monthly_savings_husband := 225
def num_months := 4
def weeks_per_month := 4
def num_weeks := num_months * weeks_per_month
def stocks_per_share := 50
def num_shares := 25
def invested_amount := num_shares * stocks_per_share
def total_savings := 2 * invested_amount

-- Weekly savings amount to prove
def weekly_savings_wife := 100

-- Total savings calculation condition
theorem wife_weekly_savings_correct :
  (monthly_savings_husband * num_months + weekly_savings_wife * num_weeks) = total_savings :=
by
  sorry

end wife_weekly_savings_correct_l113_113265


namespace area_difference_of_squares_l113_113521

theorem area_difference_of_squares (d1 d2 : ℝ) (h1 : d1 = 19) (h2 : d2 = 17) : 
  let s1 := d1 / Real.sqrt 2
  let s2 := d2 / Real.sqrt 2
  let area1 := s1 * s1
  let area2 := s2 * s2
  (area1 - area2) = 36 :=
by
  sorry

end area_difference_of_squares_l113_113521


namespace xiao_ding_distance_l113_113705

variable (x y z w : ℕ)

theorem xiao_ding_distance (h1 : x = 4 * y)
                          (h2 : z = x / 2 + 20)
                          (h3 : w = 2 * z - 15)
                          (h4 : x + y + z + w = 705) : 
                          y = 60 := 
sorry

end xiao_ding_distance_l113_113705


namespace joan_spent_on_toys_l113_113400

theorem joan_spent_on_toys :
  let toy_cars := 14.88
  let toy_trucks := 5.86
  toy_cars + toy_trucks = 20.74 :=
by
  let toy_cars := 14.88
  let toy_trucks := 5.86
  sorry

end joan_spent_on_toys_l113_113400


namespace cost_of_8_dozen_oranges_l113_113739

noncomputable def cost_per_dozen (cost_5_dozen : ℝ) : ℝ :=
  cost_5_dozen / 5

noncomputable def cost_8_dozen (cost_5_dozen : ℝ) : ℝ :=
  8 * cost_per_dozen cost_5_dozen

theorem cost_of_8_dozen_oranges (cost_5_dozen : ℝ) (h : cost_5_dozen = 39) : cost_8_dozen cost_5_dozen = 62.4 :=
by
  sorry

end cost_of_8_dozen_oranges_l113_113739


namespace subsets_bound_l113_113204

variable {n : ℕ} (S : Finset (Fin n)) (m : ℕ) (A : ℕ → Finset (Fin n))

theorem subsets_bound {n : ℕ} (hn : n ≥ 2) (hA : ∀ i, 1 ≤ i ∧ i ≤ m → (A i).card ≥ 2)
  (h_inter : ∀ i j k, 1 ≤ i ∧ i ≤ m → 1 ≤ j ∧ j ≤ m → 1 ≤ k ∧ k ≤ m →
    (A i) ∩ (A j) ≠ ∅ ∧ (A i) ∩ (A k) ≠ ∅ ∧ (A j) ∩ (A k) ≠ ∅ → (A i) ∩ (A j) ∩ (A k) ≠ ∅) :
  m ≤ 2 ^ (n - 1) - 1 := 
sorry

end subsets_bound_l113_113204


namespace problem_l113_113439

theorem problem (f : ℕ → ℝ) 
  (h_def : ∀ x, f x = Real.cos (x * Real.pi / 3)) 
  (h_period : ∀ x, f (x + 6) = f x) : 
  (Finset.sum (Finset.range 2018) f) = 0 := 
by
  sorry

end problem_l113_113439


namespace calculate_expression_l113_113180

theorem calculate_expression :
  (56 * 0.57 * 0.85) / (2.8 * 19 * 1.7) = 0.3 :=
by
  sorry

end calculate_expression_l113_113180


namespace total_potatoes_l113_113687

theorem total_potatoes (Nancy_potatoes : ℕ) (Sandy_potatoes : ℕ) (Andy_potatoes : ℕ) 
  (h1 : Nancy_potatoes = 6) (h2 : Sandy_potatoes = 7) (h3 : Andy_potatoes = 9) : 
  Nancy_potatoes + Sandy_potatoes + Andy_potatoes = 22 :=
by
  -- The proof can be written here
  sorry

end total_potatoes_l113_113687


namespace right_triangle_area_l113_113720

variable (AB AC : ℝ) (angle_A : ℝ)

def is_right_triangle (AB AC : ℝ) (angle_A : ℝ) : Prop :=
  angle_A = 90

def area_of_triangle (AB AC : ℝ) : ℝ :=
  0.5 * AB * AC

theorem right_triangle_area :
  is_right_triangle AB AC angle_A →
  AB = 35 →
  AC = 15 →
  area_of_triangle AB AC = 262.5 :=
by
  intros
  simp [is_right_triangle, area_of_triangle]
  sorry

end right_triangle_area_l113_113720


namespace parabola_perpendicular_bisector_intersects_x_axis_l113_113919

theorem parabola_perpendicular_bisector_intersects_x_axis
  (x1 y1 x2 y2 : ℝ) 
  (A_on_parabola : y1^2 = 2 * x1)
  (B_on_parabola : y2^2 = 2 * x2) 
  (k m : ℝ) 
  (AB_line : ∀ x y, y = k * x + m)
  (k_not_zero : k ≠ 0) 
  (k_m_condition : (1 / k^2) - (m / k) > 0) :
  ∃ x0 : ℝ, x0 = (1 / k^2) - (m / k) + 1 ∧ x0 > 1 :=
by
  sorry

end parabola_perpendicular_bisector_intersects_x_axis_l113_113919


namespace square_difference_l113_113971

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 81) (h2 : x * y = 6) : (x - y)^2 = 57 :=
by
  sorry

end square_difference_l113_113971


namespace euler_totient_bound_l113_113816

theorem euler_totient_bound (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : (Nat.totient^[k]) n = 1) :
  n ≤ 3^k :=
sorry

end euler_totient_bound_l113_113816


namespace steve_height_end_second_year_l113_113138

noncomputable def initial_height_ft : ℝ := 5
noncomputable def initial_height_inch : ℝ := 6
noncomputable def inch_to_cm : ℝ := 2.54

noncomputable def initial_height_cm : ℝ :=
  (initial_height_ft * 12 + initial_height_inch) * inch_to_cm

noncomputable def first_growth_spurt : ℝ := 0.15
noncomputable def second_growth_spurt : ℝ := 0.07
noncomputable def height_decrease : ℝ := 0.04

noncomputable def height_after_growths : ℝ :=
  let height_after_first_growth := initial_height_cm * (1 + first_growth_spurt)
  height_after_first_growth * (1 + second_growth_spurt)

noncomputable def final_height_cm : ℝ :=
  height_after_growths * (1 - height_decrease)

theorem steve_height_end_second_year : final_height_cm = 198.03 :=
  sorry

end steve_height_end_second_year_l113_113138


namespace min_value_of_expression_l113_113147

theorem min_value_of_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ x : ℝ, x = 6 * (12 : ℝ)^(1/6) ∧
  (∀ a b c, 0 < a ∧ 0 < b ∧ 0 < c → 
  x ≤ (a + 2 * b) / c + (2 * a + c) / b + (b + 3 * c) / a) :=
sorry

end min_value_of_expression_l113_113147


namespace total_handshakes_l113_113189

-- There are 5 members on each of the two basketball teams.
def teamMembers : Nat := 5

-- There are 2 referees.
def referees : Nat := 2

-- Each player from one team shakes hands with each player from the other team.
def handshakesBetweenTeams : Nat := teamMembers * teamMembers

-- Each player shakes hands with each referee.
def totalPlayers : Nat := 2 * teamMembers
def handshakesWithReferees : Nat := totalPlayers * referees

-- Prove that the total number of handshakes is 45.
theorem total_handshakes : handshakesBetweenTeams + handshakesWithReferees = 45 := by
  -- Total handshakes is the sum of handshakes between teams and handshakes with referees.
  sorry

end total_handshakes_l113_113189


namespace heartsuit_symmetric_solution_l113_113764

def heartsuit (a b : ℝ) : ℝ :=
  a^3 * b - a^2 * b^2 + a * b^3

theorem heartsuit_symmetric_solution :
  ∀ x y : ℝ, (heartsuit x y = heartsuit y x) ↔ (x = 0 ∨ y = 0 ∨ x = y ∨ x = -y) :=
by
  sorry

end heartsuit_symmetric_solution_l113_113764


namespace hours_felt_good_l113_113172

variable (x : ℝ)

theorem hours_felt_good (h1 : 15 * x + 10 * (8 - x) = 100) : x == 4 := 
by
  sorry

end hours_felt_good_l113_113172


namespace abs_neg_2_plus_sqrt3_add_tan60_eq_2_l113_113312

theorem abs_neg_2_plus_sqrt3_add_tan60_eq_2 :
  abs (-2 + Real.sqrt 3) + Real.tan (Real.pi / 3) = 2 :=
by
  sorry

end abs_neg_2_plus_sqrt3_add_tan60_eq_2_l113_113312


namespace complex_number_solution_l113_113922

theorem complex_number_solution (z i : ℂ) (h : z * (i - i^2) = 1 + i^3) (h1 : i^2 = -1) (h2 : i^3 = -i) (h3 : i^4 = 1) : 
  z = -i := 
by 
  sorry

end complex_number_solution_l113_113922


namespace fraction_of_work_completed_in_25_days_l113_113146

def men_init : ℕ := 100
def days_total : ℕ := 50
def hours_per_day_init : ℕ := 8
def days_first : ℕ := 25
def men_add : ℕ := 60
def hours_per_day_later : ℕ := 10

theorem fraction_of_work_completed_in_25_days : 
  (men_init * days_first * hours_per_day_init) / (men_init * days_total * hours_per_day_init) = 1 / 2 :=
  by sorry

end fraction_of_work_completed_in_25_days_l113_113146


namespace correct_calculation_l113_113209

theorem correct_calculation (x : ℝ) : 
(x + x = 2 * x) ∧
(x * x = x^2) ∧
(2 * x * x^2 = 2 * x^3) ∧
(x^6 / x^3 = x^3) →
(2 * x * x^2 = 2 * x^3) := 
by
  intro h
  exact h.2.2.1

end correct_calculation_l113_113209


namespace similar_triangles_l113_113349

theorem similar_triangles (y : ℝ) 
  (h₁ : 12 / y = 9 / 6) : y = 8 :=
by {
  -- solution here
  -- currently, we just provide the theorem statement as requested
  sorry
}

end similar_triangles_l113_113349


namespace average_percentage_l113_113428

theorem average_percentage (s1 s2 : ℕ) (a1 a2 : ℕ) (n : ℕ)
  (h1 : s1 = 15) (h2 : a1 = 70) (h3 : s2 = 10) (h4 : a2 = 90) (h5 : n = 25)
  : ((s1 * a1 + s2 * a2) / n : ℕ) = 78 :=
by
  -- We include sorry to skip the proof part.
  sorry

end average_percentage_l113_113428


namespace restaurant_production_in_june_l113_113353

def cheese_pizzas_per_day (hot_dogs_per_day : ℕ) : ℕ :=
  hot_dogs_per_day + 40

def pepperoni_pizzas_per_day (cheese_pizzas_per_day : ℕ) : ℕ :=
  2 * cheese_pizzas_per_day

def hot_dogs_per_day := 60
def beef_hot_dogs_per_day := 30
def chicken_hot_dogs_per_day := 30
def days_in_june := 30

theorem restaurant_production_in_june :
  (cheese_pizzas_per_day hot_dogs_per_day * days_in_june = 3000) ∧
  (pepperoni_pizzas_per_day (cheese_pizzas_per_day hot_dogs_per_day) * days_in_june = 6000) ∧
  (beef_hot_dogs_per_day * days_in_june = 900) ∧
  (chicken_hot_dogs_per_day * days_in_june = 900) :=
by
  sorry

end restaurant_production_in_june_l113_113353


namespace scientific_notation_of_0_0000021_l113_113182

theorem scientific_notation_of_0_0000021 :
  0.0000021 = 2.1 * 10 ^ (-6) :=
sorry

end scientific_notation_of_0_0000021_l113_113182


namespace cars_people_count_l113_113178

-- Define the problem conditions
def cars_people_conditions (x y : ℕ) : Prop :=
  y = 3 * (x - 2) ∧ y = 2 * x + 9

-- Define the theorem stating that there exist numbers of cars and people that satisfy the conditions
theorem cars_people_count (x y : ℕ) : cars_people_conditions x y ↔ (y = 3 * (x - 2) ∧ y = 2 * x + 9) := by
  -- skip the proof
  sorry

end cars_people_count_l113_113178


namespace circle_equation_tangent_y_axis_center_on_line_chord_length_condition_l113_113665

theorem circle_equation_tangent_y_axis_center_on_line_chord_length_condition :
  ∃ (x₀ y₀ r : ℝ), 
  (x₀ - 3 * y₀ = 0) ∧ 
  (r = |3 * y₀|) ∧ 
  ((x₀ + 3)^2 + (y₀ - 1)^2 = r^2 ∨ (x₀ - 3)^2 + (y₀ + 1)^2 = r^2) :=
sorry

end circle_equation_tangent_y_axis_center_on_line_chord_length_condition_l113_113665


namespace range_of_a_l113_113190

noncomputable def problem (x y z : ℝ) (a : ℝ) : Prop :=
  x > 0 ∧ y > 0 ∧ z > 0 ∧ (x + y + z = 1) ∧ 
  (a / (x * y * z) = 1/x + 1/y + 1/z - 2) 

theorem range_of_a (x y z a : ℝ) (h : problem x y z a) : 
  0 < a ∧ a ≤ 7/27 :=
sorry

end range_of_a_l113_113190


namespace value_of_n_l113_113550

theorem value_of_n (n : ℕ) : (1 / 5 : ℝ) ^ n * (1 / 4 : ℝ) ^ 18 = 1 / (2 * (10 : ℝ) ^ 35) → n = 35 :=
by
  intro h
  sorry

end value_of_n_l113_113550


namespace correct_calculation_result_l113_113986

theorem correct_calculation_result (n : ℤ) (h1 : n - 59 = 43) : n - 46 = 56 :=
by {
  sorry -- Proof is omitted
}

end correct_calculation_result_l113_113986


namespace circumcircle_radius_l113_113782

-- Here we define the necessary conditions and prove the radius.
theorem circumcircle_radius
  (A B C : Type)
  (AB : ℝ)
  (angle_B : ℝ)
  (angle_A : ℝ)
  (h_AB : AB = 2)
  (h_angle_B : angle_B = 120)
  (h_angle_A : angle_A = 30) :
  ∃ R, R = 2 :=
by
  -- We will skip the proof using sorry
  sorry

end circumcircle_radius_l113_113782


namespace work_rate_proof_l113_113823

def combined_rate (a b c : ℚ) : ℚ := a + b + c

def inv (x : ℚ) : ℚ := 1 / x

theorem work_rate_proof (A B C : ℚ) (h₁ : A + B = 1/15) (h₂ : C = 1/10) :
  inv (combined_rate A B C) = 6 :=
by
  sorry

end work_rate_proof_l113_113823


namespace track_is_600_l113_113942

noncomputable def track_length (x : ℝ) : Prop :=
  ∃ (s_b s_s : ℝ), 
      s_b > 0 ∧ s_s > 0 ∧
      (∀ t, t > 0 → ((s_b * t = 120 ∧ s_s * t = x / 2 - 120) ∨ 
                     (s_s * (t + 180 / s_s) - s_s * t = x / 2 + 60 
                      ∧ s_b * (t + 180 / s_s) - s_b * t = x / 2 - 60)))

theorem track_is_600 : track_length 600 :=
sorry

end track_is_600_l113_113942


namespace east_bound_cyclist_speed_l113_113218

-- Define the speeds of the cyclists and the relationship between them
def east_bound_speed (t : ℕ) (x : ℕ) : ℕ := t * x
def west_bound_speed (t : ℕ) (x : ℕ) : ℕ := t * (x + 4)

-- Condition: After 5 hours, they are 200 miles apart
def total_distance (t : ℕ) (x : ℕ) : ℕ := east_bound_speed t x + west_bound_speed t x

theorem east_bound_cyclist_speed :
  ∃ x : ℕ, total_distance 5 x = 200 ∧ x = 18 :=
by
  sorry

end east_bound_cyclist_speed_l113_113218


namespace max_value_sqrt_abcd_l113_113042

theorem max_value_sqrt_abcd (a b c d : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) (h_sum : a + b + c + d = 1) :
  Real.sqrt (abcd) ^ (1 / 4) + Real.sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ^ (1 / 4) ≤ 1 := 
sorry

end max_value_sqrt_abcd_l113_113042


namespace additional_pots_produced_l113_113435

theorem additional_pots_produced (first_hour_time_per_pot last_hour_time_per_pot : ℕ) :
  first_hour_time_per_pot = 6 →
  last_hour_time_per_pot = 5 →
  60 / last_hour_time_per_pot - 60 / first_hour_time_per_pot = 2 :=
by
  intros
  sorry

end additional_pots_produced_l113_113435


namespace minimum_value_of_A_l113_113766

open Real

noncomputable def A (x y z : ℝ) : ℝ :=
  ((x^3 - 24) * (x + 24)^(1/3) + (y^3 - 24) * (y + 24)^(1/3) + (z^3 - 24) * (z + 24)^(1/3)) / (x * y + y * z + z * x)

theorem minimum_value_of_A (x y z : ℝ) (h : 3 ≤ x) (h2 : 3 ≤ y) (h3 : 3 ≤ z) :
  ∃ v : ℝ, (∀ a b c : ℝ, 3 ≤ a ∧ 3 ≤ b ∧ 3 ≤ c → A a b c ≥ v) ∧ v = 1 :=
sorry

end minimum_value_of_A_l113_113766


namespace y_coordinate_of_second_point_l113_113364

variable {m n k : ℝ}

theorem y_coordinate_of_second_point (h1 : m = 2 * n + 5) (h2 : k = 0.5) : (n + k) = n + 0.5 := 
by
  sorry

end y_coordinate_of_second_point_l113_113364


namespace inequality_of_sum_l113_113732

theorem inequality_of_sum 
  (a : ℕ → ℝ)
  (h : ∀ n m, 0 ≤ n → n < m → a n < a m) :
  (0 < a 1 ->
  0 < a 2 ->
  0 < a 3 ->
  0 < a 4 ->
  0 < a 5 ->
  0 < a 6 ->
  0 < a 7 ->
  0 < a 8 ->
  0 < a 9 ->
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9) / (a 3 + a 6 + a 9) < 3) :=
by
  intros
  sorry

end inequality_of_sum_l113_113732


namespace percentage_of_students_who_speak_lies_l113_113443

theorem percentage_of_students_who_speak_lies
  (T : ℝ)    -- percentage of students who speak the truth
  (I : ℝ)    -- percentage of students who speak both truth and lies
  (U : ℝ)    -- probability of a randomly selected student speaking the truth or lies
  (H_T : T = 0.3)
  (H_I : I = 0.1)
  (H_U : U = 0.4) :
  ∃ (L : ℝ), L = 0.2 :=
by
  sorry

end percentage_of_students_who_speak_lies_l113_113443


namespace find_a7_of_arithmetic_sequence_l113_113078

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + d * (n - 1)

theorem find_a7_of_arithmetic_sequence (a d : ℤ)
  (h : arithmetic_sequence a d 1 + arithmetic_sequence a d 2 +
       arithmetic_sequence a d 12 + arithmetic_sequence a d 13 = 24) :
  arithmetic_sequence a d 7 = 6 :=
by
  sorry

end find_a7_of_arithmetic_sequence_l113_113078


namespace total_fruit_count_l113_113372

-- Define the conditions as variables and equations
def apples := 4 -- based on the final deduction from the solution
def pears := 6 -- calculated from the condition of bananas
def bananas := 9 -- given in the problem

-- State the conditions
axiom h1 : pears = apples + 2
axiom h2 : bananas = pears + 3
axiom h3 : bananas = 9

-- State the proof objective
theorem total_fruit_count : apples + pears + bananas = 19 :=
by
  sorry

end total_fruit_count_l113_113372


namespace range_of_a_l113_113868

variable {a : ℝ}

def A (a : ℝ) : Set ℝ := { x | (x - 2) * (x - (a + 1)) < 0 }
def B (a : ℝ) : Set ℝ := { x | (x - 2 * a) / (x - (a^2 + 1)) < 0 }

theorem range_of_a (a : ℝ) : B a ⊆ A a ↔ (a = -1 / 2) ∨ (2 ≤ a ∧ a ≤ 3) := by
  sorry

end range_of_a_l113_113868


namespace factorize_expression_l113_113962

theorem factorize_expression (a b : ℝ) : 2 * a ^ 2 - 8 * b ^ 2 = 2 * (a + 2 * b) * (a - 2 * b) :=
by
  sorry

end factorize_expression_l113_113962


namespace triangle_base_l113_113717

noncomputable def side_length_square (p : ℕ) : ℕ := p / 4

noncomputable def area_square (s : ℕ) : ℕ := s * s

noncomputable def area_triangle (h b : ℕ) : ℕ := (h * b) / 2

theorem triangle_base (p h a b : ℕ) (hp : p = 80) (hh : h = 40) (ha : a = (side_length_square p)^2) (eq_areas : area_square (side_length_square p) = area_triangle h b) : b = 20 :=
by {
  -- Here goes the proof which we are omitting
  sorry
}

end triangle_base_l113_113717


namespace triangle_sides_inequality_triangle_sides_equality_condition_l113_113451

theorem triangle_sides_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) ≥ 0 := 
sorry

theorem triangle_sides_equality_condition (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (triangle_cond : a + b > c ∧ b + c > a ∧ c + a > b) :
  a^2 * b * (a - b) + b^2 * c * (b - c) + c^2 * a * (c - a) = 0 ↔ a = b ∧ b = c := 
sorry

end triangle_sides_inequality_triangle_sides_equality_condition_l113_113451


namespace net_emails_received_l113_113226

-- Define the conditions
def emails_received_morning : ℕ := 3
def emails_sent_morning : ℕ := 2
def emails_received_afternoon : ℕ := 5
def emails_sent_afternoon : ℕ := 1

-- Define the problem statement
theorem net_emails_received :
  emails_received_morning - emails_sent_morning + emails_received_afternoon - emails_sent_afternoon = 5 := by
  sorry

end net_emails_received_l113_113226


namespace fixed_point_at_5_75_l113_113925

-- Defining the function
def quadratic_function (k : ℝ) (x : ℝ) : ℝ := 3 * x^2 + k * x - 5 * k

-- Stating the theorem that the graph passes through the fixed point (5, 75)
theorem fixed_point_at_5_75 (k : ℝ) : quadratic_function k 5 = 75 := by
  sorry

end fixed_point_at_5_75_l113_113925


namespace sequence_length_l113_113753

theorem sequence_length :
  ∀ (n : ℕ), 
    (2 + 4 * (n - 1) = 2010) → n = 503 :=
by
    intro n
    intro h
    sorry

end sequence_length_l113_113753


namespace function_symmetry_origin_l113_113023

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - x

theorem function_symmetry_origin : ∀ x : ℝ, f (-x) = -f x := by
  sorry

end function_symmetry_origin_l113_113023


namespace points_on_circle_l113_113821

theorem points_on_circle (t : ℝ) (ht : t ≠ 0) :
  let x := (t + 1) / t ^ 2
  let y := (t - 1) / t ^ 2
  (x - 2)^2 + (y - 2)^2 = 4 :=
by
  let x := (t + 1) / t ^ 2
  let y := (t - 1) / t ^ 2
  sorry

end points_on_circle_l113_113821


namespace range_of_a_l113_113535

theorem range_of_a (a : ℝ) (x : ℝ) :
  (¬(x > a) →¬(x^2 + 2*x - 3 > 0)) → (a ≥ 1 ) :=
by
  intro h
  sorry

end range_of_a_l113_113535


namespace simplify_fraction_l113_113002

theorem simplify_fraction (x y z : ℕ) (hx : x = 5) (hy : y = 2) (hz : z = 4) :
  (10 * x^2 * y^3 * z) / (15 * x * y^2 * z^2) = 4 / 3 :=
by
  sorry

end simplify_fraction_l113_113002


namespace intersection_M_N_l113_113603

def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x > 1}

theorem intersection_M_N :
  M ∩ N = {x | 1 < x ∧ x ≤ 2} := 
sorry

end intersection_M_N_l113_113603


namespace problem_proof_l113_113508

theorem problem_proof (x : ℝ) (hx : x + 1/x = 7) : (x - 3)^2 + 49/((x - 3)^2) = 23 := by
  sorry

end problem_proof_l113_113508


namespace inequality_proof_l113_113988

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a ^ 3 / (a ^ 2 + a * b + b ^ 2)) + (b ^ 3 / (b ^ 2 + b * c + c ^ 2)) + (c ^ 3 / (c ^ 2 + c * a + a ^ 2)) ≥ (a + b + c) / 3 :=
by
  sorry

end inequality_proof_l113_113988


namespace floor_equiv_l113_113336

theorem floor_equiv {n : ℤ} (h : n > 2) : 
  Int.floor ((n * (n + 1) : ℚ) / (4 * n - 2 : ℚ)) = Int.floor ((n + 1 : ℚ) / 4) := 
sorry

end floor_equiv_l113_113336


namespace problems_finished_equals_45_l113_113852

/-- Mathematical constants and conditions -/
def ratio_finished_left (F L : ℕ) : Prop := F = 9 * (L / 4)
def total_problems (F L : ℕ) : Prop := F + L = 65

/-- Lean theorem to prove the problem statement -/
theorem problems_finished_equals_45 :
  ∃ F L : ℕ, ratio_finished_left F L ∧ total_problems F L ∧ F = 45 :=
by
  sorry

end problems_finished_equals_45_l113_113852


namespace complete_the_square_l113_113551

theorem complete_the_square (x : ℝ) : 
  (x^2 - 8 * x + 10 = 0) → 
  ((x - 4)^2 = 6) :=
sorry

end complete_the_square_l113_113551


namespace total_spider_legs_l113_113237

theorem total_spider_legs (num_legs_single_spider group_spider_count: ℕ) 
      (h1: num_legs_single_spider = 8) 
      (h2: group_spider_count = (num_legs_single_spider / 2) + 10) :
      group_spider_count * num_legs_single_spider = 112 := 
by
  sorry

end total_spider_legs_l113_113237


namespace smallest_m_plus_n_l113_113537

theorem smallest_m_plus_n (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_lt : m < n)
    (h_eq : 1978^m % 1000 = 1978^n % 1000) : m + n = 26 :=
sorry

end smallest_m_plus_n_l113_113537


namespace range_of_k_l113_113940

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (k : ℝ) : Set ℝ := {x | k < x ∧ x < 2 * k + 1}
def A_complement : Set ℝ := {x | 1 < x ∧ x < 3}

theorem range_of_k (k : ℝ) : ((A_complement ∩ (B k)) = ∅) ↔ (k ∈ Set.Iic 0 ∪ Set.Ici 3) := sorry

end range_of_k_l113_113940


namespace trader_sold_meters_l113_113117

-- Defining the context and conditions
def cost_price_per_meter : ℝ := 100
def profit_per_meter : ℝ := 5
def total_selling_price : ℝ := 8925

-- Calculating the selling price per meter
def selling_price_per_meter : ℝ := cost_price_per_meter + profit_per_meter

-- The problem statement: proving the number of meters sold is 85
theorem trader_sold_meters : (total_selling_price / selling_price_per_meter) = 85 :=
by
  sorry

end trader_sold_meters_l113_113117


namespace zoe_total_songs_l113_113850

def total_songs (country_albums pop_albums songs_per_country_album songs_per_pop_album : ℕ) : ℕ :=
  country_albums * songs_per_country_album + pop_albums * songs_per_pop_album

theorem zoe_total_songs :
  total_songs 4 7 5 6 = 62 :=
by
  sorry

end zoe_total_songs_l113_113850


namespace eighteenth_prime_l113_113571

-- Define the necessary statements
def isPrime (n : ℕ) : Prop := sorry

def primeSeq (n : ℕ) : ℕ :=
  if n = 0 then
    2
  else if n = 1 then
    3
  else
    -- Function to generate the n-th prime number
    sorry

theorem eighteenth_prime :
  primeSeq 17 = 67 := by
  sorry

end eighteenth_prime_l113_113571


namespace profit_rate_is_five_percent_l113_113477

theorem profit_rate_is_five_percent (cost_price selling_price : ℝ) (hx : 1.1 * cost_price - 10 = 210) : 
  (selling_price = 1.1 * cost_price) → 
  (selling_price - cost_price) / cost_price * 100 = 5 :=
by
  sorry

end profit_rate_is_five_percent_l113_113477


namespace sarah_total_weeds_l113_113893

theorem sarah_total_weeds :
  let tuesday_weeds := 25
  let wednesday_weeds := 3 * tuesday_weeds
  let thursday_weeds := wednesday_weeds / 5
  let friday_weeds := thursday_weeds - 10
  tuesday_weeds + wednesday_weeds + thursday_weeds + friday_weeds = 120 :=
by
  intros
  let tuesday_weeds := 25
  let wednesday_weeds := 3 * tuesday_weeds
  let thursday_weeds := wednesday_weeds / 5
  let friday_weeds := thursday_weeds - 10
  sorry

end sarah_total_weeds_l113_113893


namespace max_value_xy_l113_113575

open Real

theorem max_value_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 2 * x + 5 * y < 100) :
  ∃ (c : ℝ), c = 3703.7 ∧ ∀ (x' y' : ℝ), 0 < x' → 0 < y' → 2 * x' + 5 * y' < 100 → x' * y' * (100 - 2 * x' - 5 * y') ≤ c :=
sorry

end max_value_xy_l113_113575


namespace find_a_of_extreme_value_at_one_l113_113618

-- Define the function f(x) = x^3 - a * x
def f (x a : ℝ) : ℝ := x^3 - a * x
  
-- Define the derivative of f with respect to x
def f' (x a : ℝ) : ℝ := 3 * x^2 - a

-- The theorem statement: for f(x) having an extreme value at x = 1, the corresponding a must be 3
theorem find_a_of_extreme_value_at_one (a : ℝ) : 
  (f' 1 a = 0) ↔ (a = 3) :=
by
  sorry

end find_a_of_extreme_value_at_one_l113_113618


namespace total_fruit_pieces_correct_l113_113324

/-
  Define the quantities of each type of fruit.
-/
def red_apples : Nat := 9
def green_apples : Nat := 4
def purple_grapes : Nat := 3
def yellow_bananas : Nat := 6
def orange_oranges : Nat := 2

/-
  The total number of fruit pieces in the basket.
-/
def total_fruit_pieces : Nat := red_apples + green_apples + purple_grapes + yellow_bananas + orange_oranges

/-
  Prove that the total number of fruit pieces is 24.
-/
theorem total_fruit_pieces_correct : total_fruit_pieces = 24 := by
  sorry

end total_fruit_pieces_correct_l113_113324


namespace counted_integer_twice_l113_113582

theorem counted_integer_twice (x n : ℕ) (hn : n = 100) 
  (h_sum : (n * (n + 1)) / 2 + x = 5053) : x = 3 := by
  sorry

end counted_integer_twice_l113_113582


namespace math_equivalence_l113_113822

theorem math_equivalence (a b c : ℕ) (ha : 0 < a ∧ a < 12) (hb : 0 < b ∧ b < 12) (hc : 0 < c ∧ c < 12) (hbc : b + c = 12) :
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c := 
by 
  sorry

end math_equivalence_l113_113822


namespace distance_AO_min_distance_BM_l113_113663

open Real

-- Definition of rectangular distance
def rectangular_distance (P Q : ℝ × ℝ) : ℝ :=
  abs (P.1 - Q.1) + abs (P.2 - Q.2)

-- Point A and O
def A : ℝ × ℝ := (-1, 3)
def O : ℝ × ℝ := (0, 0)

-- Point B
def B : ℝ × ℝ := (1, 0)

-- Line "x - y + 2 = 0"
def on_line (M : ℝ × ℝ) : Prop :=
  M.1 - M.2 + 2 = 0

-- Proof statement 1: distance from A to O is 4
theorem distance_AO : rectangular_distance A O = 4 := 
sorry

-- Proof statement 2: minimum distance from B to any point on the line is 3
theorem min_distance_BM (M : ℝ × ℝ) (h : on_line M) : rectangular_distance B M = 3 := 
sorry

end distance_AO_min_distance_BM_l113_113663


namespace solution_to_abs_eq_l113_113157

theorem solution_to_abs_eq :
  ∀ x : ℤ, abs ((-5) + x) = 11 → (x = 16 ∨ x = -6) :=
by sorry

end solution_to_abs_eq_l113_113157


namespace student_passing_percentage_l113_113318

def student_marks : ℕ := 80
def shortfall_marks : ℕ := 100
def total_marks : ℕ := 600

def passing_percentage (student_marks shortfall_marks total_marks : ℕ) : ℕ :=
  (student_marks + shortfall_marks) * 100 / total_marks

theorem student_passing_percentage :
  passing_percentage student_marks shortfall_marks total_marks = 30 :=
by
  sorry

end student_passing_percentage_l113_113318


namespace f_one_value_l113_113681

noncomputable def f (x : ℝ) : ℝ := sorry

axiom h_f_defined : ∀ x, x > 0 → ∃ y, f x = y
axiom h_f_strict_increasing : ∀ x y, 0 < x → 0 < y → x < y → f x < f y
axiom h_f_eq : ∀ x, x > 0 → f x * f (f x + 1/x) = 1

theorem f_one_value : f 1 = (1 + Real.sqrt 5) / 2 := 
by
  sorry

end f_one_value_l113_113681


namespace relationship_t_s_l113_113784

theorem relationship_t_s (a b : ℝ) : 
  let t := a + 2 * b
  let s := a + b^2 + 1
  t <= s :=
by
  sorry

end relationship_t_s_l113_113784


namespace conic_curve_focus_eccentricity_l113_113089

theorem conic_curve_focus_eccentricity (m : ℝ) 
  (h : ∀ x y : ℝ, x^2 + m * y^2 = 1)
  (eccentricity_eq : ∀ a b : ℝ, a > b → m = 4/3) : m = 4/3 :=
by
  sorry

end conic_curve_focus_eccentricity_l113_113089


namespace a_13_eq_30_l113_113082

variable (a : ℕ → ℕ)
variable (d : ℕ)

-- Define arithmetic sequence condition
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom a_5_eq_6 : a 5 = 6
axiom a_8_eq_15 : a 8 = 15

-- Required proof
theorem a_13_eq_30 (h : arithmetic_sequence a d) : a 13 = 30 :=
  sorry

end a_13_eq_30_l113_113082


namespace find_x_l113_113548

-- Define the known values
def a := 6
def b := 16
def c := 8
def desired_average := 13

-- Define the target number we need to find
def target_x := 22

-- Prove that the number we need to add to get the desired average is 22
theorem find_x : (a + b + c + target_x) / 4 = desired_average :=
by
  -- The proof itself is omitted as per instructions
  sorry

end find_x_l113_113548


namespace inverse_proportion_function_range_m_l113_113558

theorem inverse_proportion_function_range_m
  (x1 x2 y1 y2 m : ℝ)
  (h_func_A : y1 = (5 * m - 2) / x1)
  (h_func_B : y2 = (5 * m - 2) / x2)
  (h_x : x1 < x2)
  (h_x_neg : x2 < 0)
  (h_y : y1 < y2) :
  m < 2 / 5 :=
sorry

end inverse_proportion_function_range_m_l113_113558


namespace hours_per_day_in_deliberation_l113_113935

noncomputable def jury_selection_days : ℕ := 2
noncomputable def trial_days : ℕ := 4 * jury_selection_days
noncomputable def total_deliberation_hours : ℕ := 6 * 24
noncomputable def total_days_on_jury_duty : ℕ := 19

theorem hours_per_day_in_deliberation :
  (total_deliberation_hours / (total_days_on_jury_duty - (jury_selection_days + trial_days))) = 16 :=
by
  sorry

end hours_per_day_in_deliberation_l113_113935


namespace domain_of_sqrt_tan_l113_113787

theorem domain_of_sqrt_tan :
  ∀ x : ℝ, (∃ k : ℤ, k * π ≤ x ∧ x < k * π + π / 2) ↔ 0 ≤ (Real.tan x) :=
sorry

end domain_of_sqrt_tan_l113_113787


namespace increase_in_daily_mess_expenses_l113_113197

theorem increase_in_daily_mess_expenses (A X : ℝ)
  (h1 : 35 * A = 420)
  (h2 : 42 * (A - 1) = 420 + X) :
  X = 42 :=
by
  sorry

end increase_in_daily_mess_expenses_l113_113197


namespace range_of_xy_l113_113888

-- Given conditions
variables {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : 2 / x + 8 / y = 1)

-- To Prove
theorem range_of_xy (hx : x > 0) (hy : y > 0) (hxy : 2 / x + 8 / y = 1) : 64 ≤ x * y :=
sorry

end range_of_xy_l113_113888


namespace proof_problem_l113_113412

noncomputable def f (x : ℝ) : ℝ := Real.exp x

noncomputable def g (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x + b

theorem proof_problem
  (a : ℝ) (b : ℝ) (x : ℝ)
  (h₀ : 0 ≤ a)
  (h₁ : a ≤ 1 / 2)
  (h₂ : b = 1)
  (h₃ : 0 ≤ x) :
  (1 / f x) + (x / g x a b) ≥ 1 := by
    sorry

end proof_problem_l113_113412


namespace simplify_abs_neg_pow_sub_l113_113470

theorem simplify_abs_neg_pow_sub (a b : ℤ) (h : a = 4) (h' : b = 6) : 
  (|-(a ^ 2) - b| = 22) := 
by
  sorry

end simplify_abs_neg_pow_sub_l113_113470


namespace paper_fold_length_l113_113956

theorem paper_fold_length (length_orig : ℝ) (h : length_orig = 12) : length_orig / 2 = 6 :=
by
  rw [h]
  norm_num

end paper_fold_length_l113_113956


namespace remainder_expression_l113_113954

theorem remainder_expression (x y u v : ℕ) (h1 : x = u * y + v) (h2 : 0 ≤ v) (h3 : v < y) : 
  (x + 3 * u * y) % y = v := 
by
  sorry

end remainder_expression_l113_113954


namespace reading_schedule_correct_l113_113965

-- Defining the conditions
def total_words : ℕ := 34685
def words_day1 (x : ℕ) : ℕ := x
def words_day2 (x : ℕ) : ℕ := 2 * x
def words_day3 (x : ℕ) : ℕ := 4 * x

-- Defining the main statement of the problem
theorem reading_schedule_correct (x : ℕ) : 
  words_day1 x + words_day2 x + words_day3 x = total_words := 
sorry

end reading_schedule_correct_l113_113965


namespace number_of_ordered_triples_l113_113786

theorem number_of_ordered_triples (x y z : ℝ) (hx : x + y = 3) (hy : xy - z^2 = 4)
  (hnn : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z) : 
  ∃! (x y z : ℝ), (x + y = 3) ∧ (xy - z^2 = 4) ∧ (0 ≤ x) ∧ (0 ≤ y) ∧ (0 ≤ z) :=
sorry

end number_of_ordered_triples_l113_113786


namespace shaded_area_of_square_with_circles_l113_113070

theorem shaded_area_of_square_with_circles :
  let side_length_square := 12
  let radius_quarter_circle := 6
  let radius_center_circle := 3
  let area_square := side_length_square * side_length_square
  let area_quarter_circles := 4 * (1 / 4) * Real.pi * (radius_quarter_circle ^ 2)
  let area_center_circle := Real.pi * (radius_center_circle ^ 2)
  area_square - area_quarter_circles - area_center_circle = 144 - 45 * Real.pi :=
by
  sorry

end shaded_area_of_square_with_circles_l113_113070


namespace max_sum_prod_48_l113_113213

theorem max_sum_prod_48 (spadesuit heartsuit : Nat) (h: spadesuit * heartsuit = 48) : spadesuit + heartsuit ≤ 49 :=
sorry

end max_sum_prod_48_l113_113213


namespace tear_paper_l113_113820

theorem tear_paper (n : ℕ) : 1 + 3 * n ≠ 2007 :=
by
  sorry

end tear_paper_l113_113820


namespace TotalGenuineItems_l113_113769

def TirzahPurses : ℕ := 26
def TirzahHandbags : ℕ := 24
def FakePurses : ℕ := TirzahPurses / 2
def FakeHandbags : ℕ := TirzahHandbags / 4
def GenuinePurses : ℕ := TirzahPurses - FakePurses
def GenuineHandbags : ℕ := TirzahHandbags - FakeHandbags

theorem TotalGenuineItems : GenuinePurses + GenuineHandbags = 31 :=
  by
    -- proof
    sorry

end TotalGenuineItems_l113_113769


namespace tory_sells_grandmother_l113_113510

theorem tory_sells_grandmother (G : ℕ)
    (total_goal : ℕ) (sold_to_uncle : ℕ) (sold_to_neighbor : ℕ) (remaining_to_sell : ℕ)
    (h_goal : total_goal = 50) (h_sold_to_uncle : sold_to_uncle = 7)
    (h_sold_to_neighbor : sold_to_neighbor = 5) (h_remaining_to_sell : remaining_to_sell = 26) :
    (G + sold_to_uncle + sold_to_neighbor + remaining_to_sell = total_goal) → G = 12 :=
by
    intros h
    -- Proof goes here
    sorry

end tory_sells_grandmother_l113_113510


namespace AndrewAge_l113_113546

noncomputable def AndrewAgeProof : Prop :=
  ∃ (a g : ℕ), g = 10 * a ∧ g - a = 45 ∧ a = 5

-- Proof is not required, so we use sorry to skip the proof.
theorem AndrewAge : AndrewAgeProof := by
  sorry

end AndrewAge_l113_113546


namespace simplify_expression_l113_113668

theorem simplify_expression :
  (2 + Real.sqrt 3)^2 - Real.sqrt 18 * Real.sqrt (2 / 3) = 7 + 2 * Real.sqrt 3 :=
by
  sorry

end simplify_expression_l113_113668


namespace complex_multiplication_l113_113497

theorem complex_multiplication (i : ℂ) (h : i^2 = -1) : i * (2 - i) = 1 + 2 * i := by
  sorry

end complex_multiplication_l113_113497


namespace math_majors_consecutive_probability_l113_113025

def twelve_people := 12
def math_majors := 5
def physics_majors := 4
def biology_majors := 3

def total_ways := Nat.choose twelve_people math_majors

-- Computes the probability that all five math majors sit in consecutive seats
theorem math_majors_consecutive_probability :
  (12 : ℕ) / (Nat.choose twelve_people math_majors) = 1 / 66 := by
  sorry

end math_majors_consecutive_probability_l113_113025


namespace firm_partners_initial_count_l113_113927

theorem firm_partners_initial_count
  (x : ℕ)
  (h1 : 2*x/(63*x + 35) = 1/34)
  (h2 : 2*x/(20*x + 10) = 1/15) :
  2*x = 14 :=
by
  sorry

end firm_partners_initial_count_l113_113927


namespace pairs_of_values_l113_113123

theorem pairs_of_values (x y : ℂ) :
  (y = (x + 2)^3 ∧ x * y + 2 * y = 2) →
  (∃ (r1 r2 i1 i2 : ℂ), (r1.im = 0 ∧ r2.im = 0) ∧ (i1.im ≠ 0 ∧ i2.im ≠ 0) ∧ 
    ((r1, (r1 + 2)^3) = (x, y) ∨ (r2, (r2 + 2)^3) = (x, y) ∨
     (i1, (i1 + 2)^3) = (x, y) ∨ (i2, (i2 + 2)^3) = (x, y))) :=
sorry

end pairs_of_values_l113_113123


namespace car_not_sold_probability_l113_113418

theorem car_not_sold_probability (a b : ℕ) (h : a = 5) (k : b = 6) : (b : ℚ) / (a + b : ℚ) = 6 / 11 :=
  by
    rw [h, k]
    norm_num

end car_not_sold_probability_l113_113418


namespace line_ellipse_common_points_l113_113540

theorem line_ellipse_common_points (m : ℝ) : (m ≥ 1 ∧ m ≠ 5) ↔ (∀ k : ℝ, ∃ x y : ℝ, y = k * x + 1 ∧ (x^2 / 5) + (y^2 / m) = 1) :=
by 
  sorry

end line_ellipse_common_points_l113_113540


namespace largest_7_10_triple_l113_113021

theorem largest_7_10_triple :
  ∃ M : ℕ, (3 * M = Nat.ofDigits 10 (Nat.digits 7 M))
  ∧ (∀ N : ℕ, (3 * N = Nat.ofDigits 10 (Nat.digits 7 N)) → N ≤ M)
  ∧ M = 335 :=
sorry

end largest_7_10_triple_l113_113021


namespace initial_students_l113_113904

variable (n : ℝ) (W : ℝ)

theorem initial_students 
  (h1 : W = n * 15)
  (h2 : W + 11 = (n + 1) * 14.8)
  (h3 : 15 * n + 11 = 14.8 * n + 14.8)
  (h4 : 0.2 * n = 3.8) :
  n = 19 :=
sorry

end initial_students_l113_113904


namespace opposite_of_neg3_squared_l113_113045

theorem opposite_of_neg3_squared : -(-3^2) = 9 :=
by
  sorry

end opposite_of_neg3_squared_l113_113045


namespace number_of_girls_l113_113274

-- Definitions from the problem conditions
def ratio_girls_boys (g b : ℕ) : Prop := 4 * b = 3 * g
def total_students (g b : ℕ) : Prop := g + b = 56

-- The proof statement
theorem number_of_girls (g b k : ℕ) (hg : 4 * k = g) (hb : 3 * k = b) (hr : ratio_girls_boys g b) (ht : total_students g b) : g = 32 :=
by sorry

end number_of_girls_l113_113274


namespace combined_tax_rate_l113_113127

theorem combined_tax_rate (Mork_income Mindy_income : ℝ) (h1 : Mindy_income = 3 * Mork_income)
  (tax_Mork tax_Mindy : ℝ) (h2 : tax_Mork = 0.10 * Mork_income) (h3 : tax_Mindy = 0.20 * Mindy_income)
  : (tax_Mork + tax_Mindy) / (Mork_income + Mindy_income) = 0.175 :=
by
  sorry

end combined_tax_rate_l113_113127


namespace sofia_total_cost_l113_113515

def shirt_cost : ℕ := 7
def shoes_cost : ℕ := shirt_cost + 3
def two_shirts_cost : ℕ := 2 * shirt_cost
def total_clothes_cost : ℕ := two_shirts_cost + shoes_cost
def bag_cost : ℕ := total_clothes_cost / 2
def total_cost : ℕ := two_shirts_cost + shoes_cost + bag_cost

theorem sofia_total_cost : total_cost = 36 := by
  sorry

end sofia_total_cost_l113_113515


namespace days_with_equal_sun_tue_l113_113185

theorem days_with_equal_sun_tue (days_in_month : ℕ) (weekdays : ℕ) (d1 d2 : ℕ) (h1 : days_in_month = 30)
  (h2 : weekdays = 7) (h3 : d1 = 4) (h4 : d2 = 2) :
  ∃ count, count = 3 := by
  sorry

end days_with_equal_sun_tue_l113_113185


namespace train_crossing_time_l113_113490

theorem train_crossing_time 
    (length : ℝ) (speed_kmph : ℝ) 
    (conversion_factor: ℝ) (speed_mps: ℝ) 
    (time : ℝ) :
  length = 400 ∧ speed_kmph = 144 ∧ conversion_factor = 1000 / 3600 ∧ speed_mps = speed_kmph * conversion_factor ∧ time = length / speed_mps → time = 10 := 
by 
  sorry

end train_crossing_time_l113_113490


namespace problem_condition_l113_113031

theorem problem_condition (m : ℝ) :
  (∀ x : ℝ, x ≤ -1 → (m^2 - m) * 4^x - 2^x < 0) → -1 < m ∧ m < 2 :=
sorry

end problem_condition_l113_113031


namespace solution_of_system_l113_113838

theorem solution_of_system :
  ∃ x y : ℝ, (x^4 + y^4 = 17) ∧ (x + y = 3) ∧ ((x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1)) :=
by
  sorry

end solution_of_system_l113_113838


namespace real_values_satisfying_inequality_l113_113136

theorem real_values_satisfying_inequality :
  { x : ℝ | (x^2 + 2*x^3 - 3*x^4) / (2*x + 3*x^2 - 4*x^3) ≥ -1 } =
  Set.Icc (-1 : ℝ) ((-3 - Real.sqrt 41) / -8) ∪ 
  Set.Ioo ((-3 - Real.sqrt 41) / -8) ((-3 + Real.sqrt 41) / -8) ∪ 
  Set.Ioo ((-3 + Real.sqrt 41) / -8) 0 ∪ 
  Set.Ioi 0 :=
by
  sorry

end real_values_satisfying_inequality_l113_113136


namespace sum_of_constants_l113_113644

theorem sum_of_constants (c d : ℝ) (h₁ : 16 = 2 * 4 + c) (h₂ : 16 = 4 * 4 + d) : c + d = 8 := by
  sorry

end sum_of_constants_l113_113644


namespace solve_for_y_l113_113278

theorem solve_for_y (x y : ℝ) : 3 * x + 5 * y = 10 → y = 2 - (3 / 5) * x :=
by 
  -- proof steps would be filled here
  sorry

end solve_for_y_l113_113278


namespace mango_distribution_l113_113979

theorem mango_distribution (friends : ℕ) (initial_mangos : ℕ) 
    (share_left : ℕ) (share_right : ℕ) 
    (eat_mango : ℕ) (pass_mango_right : ℕ)
    (H1 : friends = 100) 
    (H2 : initial_mangos = 2019)
    (H3 : share_left = 2) 
    (H4 : share_right = 1) 
    (H5 : eat_mango = 1) 
    (H6 : pass_mango_right = 1) :
    ∃ final_count, final_count = 8 :=
by
  -- Proof is omitted.
  sorry

end mango_distribution_l113_113979


namespace train_crosses_lamp_post_in_30_seconds_l113_113565

open Real

/-- Prove that given a train that crosses a 2500 m long bridge in 120 s and has a length of
    833.33 m, it takes the train 30 seconds to cross a lamp post. -/
theorem train_crosses_lamp_post_in_30_seconds (L_train : ℝ) (L_bridge : ℝ) (T_bridge : ℝ) (T_lamp_post : ℝ)
  (hL_train : L_train = 833.33)
  (hL_bridge : L_bridge = 2500)
  (hT_bridge : T_bridge = 120)
  (ht : T_lamp_post = (833.33 / ((833.33 + 2500) / 120))) :
  T_lamp_post = 30 :=
by
  sorry

end train_crosses_lamp_post_in_30_seconds_l113_113565


namespace find_k_plus_m_l113_113902

def initial_sum := 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9
def initial_count := 9

def new_list_sum (m k : ℕ) := initial_sum + 8 * m + 9 * k
def new_list_count (m k : ℕ) := initial_count + m + k

def average_eq_73 (m k : ℕ) := (new_list_sum m k : ℝ) / (new_list_count m k : ℝ) = 7.3

theorem find_k_plus_m : ∃ (m k : ℕ), average_eq_73 m k ∧ (k + m = 21) :=
by
  sorry

end find_k_plus_m_l113_113902


namespace card_average_value_l113_113957

theorem card_average_value (n : ℕ) (h : (2 * n + 1) / 3 = 2023) : n = 3034 :=
sorry

end card_average_value_l113_113957


namespace time_jogging_l113_113977

def distance := 25     -- Distance jogged (in kilometers)
def speed := 5        -- Speed (in kilometers per hour)

theorem time_jogging :
  (distance / speed) = 5 := 
by
  sorry

end time_jogging_l113_113977


namespace calculation_correct_l113_113482

theorem calculation_correct : 2 * (3 ^ 2) ^ 4 = 13122 := by
  sorry

end calculation_correct_l113_113482


namespace find_matrix_N_l113_113587

open Matrix

variable (u : Fin 3 → ℝ)

def cross_product (a b : Fin 3 → ℝ) : Fin 3 → ℝ :=
  ![a 1 * b 2 - a 2 * b 1, a 2 * b 0 - a 0 * b 2, a 0 * b 1 - a 1 * b 0]

-- Define vector v as the fixed vector in the problem
def v : Fin 3 → ℝ := ![7, 3, -9]

-- Define matrix N as the matrix to be found
def N : Matrix (Fin 3) (Fin 3) ℝ := ![![0, 9, 3], ![-9, 0, -7], ![-3, 7, 0]]

-- Define the requirement condition
theorem find_matrix_N :
  ∀ (u : Fin 3 → ℝ), (N.mulVec u) = cross_product v u :=
by
  sorry

end find_matrix_N_l113_113587


namespace arithmetic_sequence_second_term_l113_113883

theorem arithmetic_sequence_second_term (a d : ℤ)
  (h1 : a + 11 * d = 11)
  (h2 : a + 12 * d = 14) :
  a + d = -19 :=
sorry

end arithmetic_sequence_second_term_l113_113883


namespace product_of_k_values_l113_113751

theorem product_of_k_values (a b c k : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a) (h_eq : a / (1 + b) = k ∧ b / (1 + c) = k ∧ c / (1 + a) = k) : k = -1 :=
by
  sorry

end product_of_k_values_l113_113751


namespace g_eq_one_l113_113899

theorem g_eq_one (g : ℝ → ℝ) 
  (h1 : ∀ (x y : ℝ), g (x - y) = g x * g y) 
  (h2 : ∀ (x : ℝ), g x ≠ 0) : 
  g 5 = 1 :=
by
  sorry

end g_eq_one_l113_113899


namespace desired_average_score_is_correct_l113_113416

-- Conditions
def average_score_9_tests : ℕ := 82
def score_10th_test : ℕ := 92

-- Desired average score
def desired_average_score : ℕ := 83

-- Total score for 10 tests
def total_score_10_tests (avg9 : ℕ) (score10 : ℕ) : ℕ :=
  9 * avg9 + score10

-- Main theorem statement to prove
theorem desired_average_score_is_correct :
  total_score_10_tests average_score_9_tests score_10th_test / 10 = desired_average_score :=
by
  sorry

end desired_average_score_is_correct_l113_113416


namespace sector_properties_l113_113134

noncomputable def central_angle (l R : ℝ) : ℝ := l / R

noncomputable def area_of_sector (l R : ℝ) : ℝ := (1 / 2) * l * R

theorem sector_properties (R l : ℝ) (hR : R = 8) (hl : l = 12) :
  central_angle l R = 3 / 2 ∧ area_of_sector l R = 48 :=
by
  sorry

end sector_properties_l113_113134


namespace stack_of_logs_total_l113_113311

-- Define the given conditions as variables and constants in Lean
def bottom_row : Nat := 15
def top_row : Nat := 4
def rows : Nat := bottom_row - top_row + 1
def sum_arithmetic_series (a l n : Nat) : Nat := n * (a + l) / 2

-- Define the main theorem to prove
theorem stack_of_logs_total : sum_arithmetic_series top_row bottom_row rows = 114 :=
by
  -- Here you will normally provide the proof
  sorry

end stack_of_logs_total_l113_113311


namespace triangle_obtuse_l113_113854

theorem triangle_obtuse 
  (A B : ℝ)
  (hA : 0 < A ∧ A < π/2)
  (hB : 0 < B ∧ B < π/2)
  (h_cosA_gt_sinB : Real.cos A > Real.sin B) :
  π - (A + B) > π/2 ∧ π - (A + B) < π :=
by
  sorry

end triangle_obtuse_l113_113854


namespace john_started_5_days_ago_l113_113689

noncomputable def daily_wage (x : ℕ) : Prop := 250 + 10 * x = 750

theorem john_started_5_days_ago :
  ∃ x : ℕ, daily_wage x ∧ 250 / x = 5 :=
by
  sorry

end john_started_5_days_ago_l113_113689


namespace satisfies_equation_l113_113102

theorem satisfies_equation : 
  { (x, y) : ℤ × ℤ | x^2 + x = y^4 + y^3 + y^2 + y } = 
  { (0, -1), (-1, -1), (0, 0), (-1, 0), (5, 2), (-6, 2) } :=
by
  sorry

end satisfies_equation_l113_113102


namespace smallest_lambda_inequality_l113_113496

theorem smallest_lambda_inequality 
  (a b c d : ℝ) (h_pos : ∀ x ∈ [a, b, c, d], 0 < x) (h_sum : a + b + c + d = 4) :
  5 * (a*b + a*c + a*d + b*c + b*d + c*d) ≤ 8 * (a*b*c*d) + 12 :=
sorry

end smallest_lambda_inequality_l113_113496


namespace cats_more_than_spinsters_l113_113339

def ratio (a b : ℕ) := ∃ k : ℕ, a = b * k

theorem cats_more_than_spinsters (S C : ℕ) (h1 : ratio 2 9) (h2 : S = 12) (h3 : 2 * C = 108) :
  C - S = 42 := by 
  sorry

end cats_more_than_spinsters_l113_113339


namespace cone_central_angle_l113_113020

/-- Proof Problem Statement: Given the radius of the base circle of a cone (r) and the slant height of the cone (l),
    prove that the central angle (θ) of the unfolded diagram of the lateral surface of this cone is 120 degrees. -/
theorem cone_central_angle (r l : ℝ) (h_r : r = 10) (h_l : l = 30) : (360 * r) / l = 120 :=
by
  -- The proof steps are omitted
  sorry

end cone_central_angle_l113_113020


namespace plane_through_point_and_line_l113_113673

noncomputable def plane_equation (x y z : ℝ) : Prop :=
  12 * x + 67 * y + 23 * z - 26 = 0

theorem plane_through_point_and_line :
  ∃ (A B C D : ℤ), 
  (A > 0) ∧ (Int.gcd (abs A) (Int.gcd (abs B) (Int.gcd (abs C) (abs D))) = 1) ∧
  (plane_equation 1 4 (-6)) ∧  
  ∀ t : ℝ, (plane_equation (4 * t + 2)  (-t - 1) (5 * t + 3)) :=
sorry

end plane_through_point_and_line_l113_113673


namespace solve_equation_l113_113271

theorem solve_equation (x : ℝ) (h : (x - 60) / 3 = (4 - 3 * x) / 6) : x = 124 / 5 := by
  sorry

end solve_equation_l113_113271


namespace people_in_club_M_l113_113061

theorem people_in_club_M (m s z n : ℕ) (h1 : s = 18) (h2 : z = 11) (h3 : m + s + z + n = 60) (h4 : n ≤ 26) : m = 5 :=
sorry

end people_in_club_M_l113_113061


namespace max_x_value_l113_113051

theorem max_x_value (x y z : ℝ) (h1 : x + y + z = 5) (h2 : xy + xz + yz = 8) : 
  x ≤ 7 / 3 :=
sorry

end max_x_value_l113_113051


namespace ending_number_divisible_by_six_l113_113699

theorem ending_number_divisible_by_six (first_term : ℕ) (n : ℕ) (common_difference : ℕ) (sequence_length : ℕ) 
  (start : first_term = 12) 
  (diff : common_difference = 6)
  (num_terms : sequence_length = 11) :
  first_term + (sequence_length - 1) * common_difference = 72 := by
  sorry

end ending_number_divisible_by_six_l113_113699


namespace eval_expr_at_x_eq_neg6_l113_113633

-- Define the given condition
def x : ℤ := -4

-- Define the expression to be simplified and evaluated
def expr (x y : ℤ) : ℤ := ((x + y)^2 - y * (2 * x + y) - 8 * x) / (2 * x)

-- The theorem stating the result of the evaluated expression
theorem eval_expr_at_x_eq_neg6 (y : ℤ) : expr (-4) y = -6 := 
by
  sorry

end eval_expr_at_x_eq_neg6_l113_113633


namespace max_sub_min_value_l113_113609

variable {x y : ℝ}

noncomputable def expression (x y : ℝ) : ℝ :=
  (abs (x + y))^2 / ((abs x)^2 + (abs y)^2)

theorem max_sub_min_value :
  ∀ x y : ℝ, x ≠ 0 → y ≠ 0 → 
  (expression x y ≤ 2 ∧ 0 ≤ expression x y) → 
  (∃ m M, m = 0 ∧ M = 2 ∧ M - m = 2) :=
by
  sorry

end max_sub_min_value_l113_113609


namespace min_sum_of_3_digit_numbers_l113_113628

def digits : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def is_3_digit (n : ℕ) := 100 ≤ n ∧ n ≤ 999

theorem min_sum_of_3_digit_numbers : 
  ∃ (a b c : ℕ), 
    a ∈ digits.permutations.map (λ l => 100*l.head! + 10*(l.tail!.head!) + l.tail!.tail!.head!) ∧ 
    b ∈ digits.permutations.map (λ l => 100*l.head! + 10*(l.tail!.head!) + l.tail!.tail!.head!) ∧ 
    c ∈ digits.permutations.map (λ l => 100*l.head! + 10*(l.tail!.head!) + l.tail!.tail!.head!) ∧ 
    a + b = c ∧ 
    a + b + c = 459 := 
sorry

end min_sum_of_3_digit_numbers_l113_113628


namespace combined_rocket_height_l113_113612

theorem combined_rocket_height :
  let first_rocket_height := 500
  let second_rocket_height := 2 * first_rocket_height
  first_rocket_height + second_rocket_height = 1500 :=
by
  sorry

end combined_rocket_height_l113_113612


namespace find_distance_l113_113547

-- Definitions based on the given conditions
def speed_of_boat := 16 -- in kmph
def speed_of_stream := 2 -- in kmph
def total_time := 960 -- in hours
def downstream_speed := speed_of_boat + speed_of_stream
def upstream_speed := speed_of_boat - speed_of_stream

-- Prove that the distance D is 7590 km given the total time and speeds
theorem find_distance (D : ℝ) :
  (D / downstream_speed + D / upstream_speed = total_time) → D = 7590 :=
by
  sorry

end find_distance_l113_113547


namespace river_lengths_l113_113831

theorem river_lengths (x : ℝ) (dnieper don : ℝ)
  (h1 : dnieper = (5 / (19 / 3)) * x)
  (h2 : don = (6.5 / 9.5) * x)
  (h3 : dnieper - don = 300) :
  x = 2850 ∧ dnieper = 2250 ∧ don = 1950 :=
by
  sorry

end river_lengths_l113_113831


namespace common_ratio_of_increasing_geometric_sequence_l113_113780

theorem common_ratio_of_increasing_geometric_sequence 
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_inc : ∀ n, a n < a (n + 1))
  (h_a2 : a 2 = 2)
  (h_a4_a3 : a 4 - a 3 = 4) : 
  q = 2 :=
by
  -- sorry - placeholder for proof
  sorry

end common_ratio_of_increasing_geometric_sequence_l113_113780


namespace max_strips_cut_l113_113576

-- Definitions: dimensions of the paper and the strips
def length_paper : ℕ := 14
def width_paper : ℕ := 11
def length_strip : ℕ := 4
def width_strip : ℕ := 1

-- States the main theorem: Maximum number of strips that can be cut from the rectangular piece of paper
theorem max_strips_cut (L W l w : ℕ) (H1 : L = 14) (H2 : W = 11) (H3 : l = 4) (H4 : w = 1) :
  ∃ n : ℕ, n = 33 :=
by
  sorry

end max_strips_cut_l113_113576


namespace no_integer_pairs_satisfy_equation_l113_113561

theorem no_integer_pairs_satisfy_equation :
  ∀ (m n : ℤ), m^3 + 6 * m^2 + 5 * m ≠ 27 * n^3 + 27 * n^2 + 9 * n + 1 :=
by
  intros m n
  sorry

end no_integer_pairs_satisfy_equation_l113_113561


namespace contradiction_proof_l113_113164

theorem contradiction_proof (a b : ℝ) (h : a ≥ b) (h_pos : b > 0) (h_contr : a^2 < b^2) : false :=
by {
  sorry
}

end contradiction_proof_l113_113164


namespace plan1_maximizes_B_winning_probability_l113_113411

open BigOperators

-- Definitions for the conditions
def prob_A_wins : ℚ := 3/4
def prob_B_wins : ℚ := 1/4

-- Plan 1 probabilities
def prob_B_win_2_0 : ℚ := prob_B_wins^2
def prob_B_win_2_1 : ℚ := (Nat.choose 2 1) * prob_B_wins * prob_A_wins * prob_B_wins
def prob_B_win_plan1 : ℚ := prob_B_win_2_0 + prob_B_win_2_1

-- Plan 2 probabilities
def prob_B_win_3_0 : ℚ := prob_B_wins^3
def prob_B_win_3_1 : ℚ := (Nat.choose 3 1) * prob_B_wins^2 * prob_A_wins * prob_B_wins
def prob_B_win_3_2 : ℚ := (Nat.choose 4 2) * prob_B_wins^2 * prob_A_wins^2 * prob_B_wins
def prob_B_win_plan2 : ℚ := prob_B_win_3_0 + prob_B_win_3_1 + prob_B_win_3_2

-- Theorem statement
theorem plan1_maximizes_B_winning_probability :
  prob_B_win_plan1 > prob_B_win_plan2 :=
by
  sorry

end plan1_maximizes_B_winning_probability_l113_113411


namespace picture_area_l113_113703

theorem picture_area (x y : ℕ) (hx : 0 < x) (hy : 0 < y) :
  (3 * x + 4) * (y + 3) - x * y = 54 → x * y = 6 :=
by
  intros h
  sorry

end picture_area_l113_113703


namespace cos_minus_sin_eq_neg_one_fifth_l113_113330

theorem cos_minus_sin_eq_neg_one_fifth
  (α : ℝ)
  (h1 : Real.sin (2 * α) = 24 / 25)
  (h2 : π < α ∧ α < 5 * π / 4) :
  Real.cos α - Real.sin α = -1 / 5 := sorry

end cos_minus_sin_eq_neg_one_fifth_l113_113330


namespace tangent_line_value_l113_113358

theorem tangent_line_value {k : ℝ} 
  (h1 : ∃ x y : ℝ, x^2 + y^2 - 6*y + 8 = 0) 
  (h2 : ∃ P Q : ℝ, x^2 + y^2 - 6*y + 8 = 0 ∧ Q = k * P)
  (h3 : P * k < 0 ∧ P < 0 ∧ Q > 0) : 
  k = -2 * Real.sqrt 2 :=
sorry

end tangent_line_value_l113_113358


namespace find_prime_solution_l113_113945

theorem find_prime_solution :
  ∀ p x y : ℕ, Prime p → x > 0 → y > 0 →
    (p ^ x = y ^ 3 + 1) ↔ 
    ((p = 2 ∧ x = 1 ∧ y = 1) ∨ (p = 3 ∧ x = 2 ∧ y = 2)) := 
by
  sorry

end find_prime_solution_l113_113945


namespace fraction_of_income_from_tips_l113_113027

variable (S T I : ℝ)

-- Conditions
def tips_are_fraction_of_salary : Prop := T = (3/4) * S
def total_income_is_sum_of_salary_and_tips : Prop := I = S + T

-- Statement to prove
theorem fraction_of_income_from_tips (h1 : tips_are_fraction_of_salary S T) (h2 : total_income_is_sum_of_salary_and_tips S T I) :
  T / I = 3 / 7 := by
  sorry

end fraction_of_income_from_tips_l113_113027


namespace year_2049_is_Jisi_l113_113478

-- Define Heavenly Stems
def HeavenlyStems : List String := ["Jia", "Yi", "Bing", "Ding", "Wu", "Ji", "Geng", "Xin", "Ren", "Gui"]

-- Define Earthly Branches
def EarthlyBranches : List String := ["Zi", "Chou", "Yin", "Mao", "Chen", "Si", "Wu", "Wei", "Shen", "You", "Xu", "Hai"]

-- Define the indices of Ding (丁) and You (酉) based on 2017
def Ding_index : Nat := 3
def You_index : Nat := 9

-- Define the year difference
def year_difference : Nat := 2049 - 2017

-- Calculate the indices for the Heavenly Stem and Earthly Branch in 2049
def HeavenlyStem_index_2049 : Nat := (Ding_index + year_difference) % 10
def EarthlyBranch_index_2049 : Nat := (You_index + year_difference) % 12

theorem year_2049_is_Jisi : 
  HeavenlyStems[HeavenlyStem_index_2049]? = some "Ji" ∧ EarthlyBranches[EarthlyBranch_index_2049]? = some "Si" :=
by
  sorry

end year_2049_is_Jisi_l113_113478
