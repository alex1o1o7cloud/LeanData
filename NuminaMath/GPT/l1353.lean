import Mathlib

namespace smallest_positive_integer_x_for_2520x_eq_m_cubed_l1353_135349

theorem smallest_positive_integer_x_for_2520x_eq_m_cubed :
  ∃ (M x : ℕ), x > 0 ∧ 2520 * x = M^3 ∧ (∀ y, y > 0 ∧ 2520 * y = M^3 → x ≤ y) :=
sorry

end smallest_positive_integer_x_for_2520x_eq_m_cubed_l1353_135349


namespace race_order_l1353_135351

inductive Position where
| First | Second | Third | Fourth | Fifth
deriving DecidableEq, Repr

structure Statements where
  amy1 : Position → Prop
  amy2 : Position → Prop
  bruce1 : Position → Prop
  bruce2 : Position → Prop
  chris1 : Position → Prop
  chris2 : Position → Prop
  donna1 : Position → Prop
  donna2 : Position → Prop
  eve1 : Position → Prop
  eve2 : Position → Prop

def trueStatements : Statements := {
  amy1 := fun p => p = Position.Second,
  amy2 := fun p => p = Position.Third,
  bruce1 := fun p => p = Position.Second,
  bruce2 := fun p => p = Position.Fourth,
  chris1 := fun p => p = Position.First,
  chris2 := fun p => p = Position.Second,
  donna1 := fun p => p = Position.Third,
  donna2 := fun p => p = Position.Fifth,
  eve1 := fun p => p = Position.Fourth,
  eve2 := fun p => p = Position.First,
}

theorem race_order (f : Statements) :
  f.amy1 Position.Second ∧ f.amy2 Position.Third ∧
  f.bruce1 Position.First ∧ f.bruce2 Position.Fourth ∧
  f.chris1 Position.Fifth ∧ f.chris2 Position.Second ∧
  f.donna1 Position.Fourth ∧ f.donna2 Position.Fifth ∧
  f.eve1 Position.Fourth ∧ f.eve2 Position.First :=
by
  sorry

end race_order_l1353_135351


namespace land_area_l1353_135368

theorem land_area (x : ℝ) (h : (70 * x - 800) / 1.2 * 1.6 + 800 = 80 * x) : x = 20 :=
by
  sorry

end land_area_l1353_135368


namespace point_not_in_region_l1353_135364

-- Define the inequality
def inequality (x y : ℝ) : Prop := 3 * x + 2 * y < 6

-- Points definition
def point := ℝ × ℝ

-- Points to be checked
def p1 : point := (0, 0)
def p2 : point := (1, 1)
def p3 : point := (0, 2)
def p4 : point := (2, 0)

-- Conditions stating that certain points satisfy the inequality
axiom h1 : inequality p1.1 p1.2
axiom h2 : inequality p2.1 p2.2
axiom h3 : inequality p3.1 p3.2

-- Goal: Prove that point (2,0) does not satisfy the inequality
theorem point_not_in_region : ¬ inequality p4.1 p4.2 :=
sorry -- Proof omitted

end point_not_in_region_l1353_135364


namespace geometric_sequence_150th_term_l1353_135375

-- Given conditions
def a1 : ℤ := 5
def a2 : ℤ := -10

-- Computation of common ratio
def r : ℤ := a2 / a1

-- Definition of the n-th term in geometric sequence
def nth_term (n : ℕ) : ℤ :=
  a1 * r^(n-1)

-- Statement to prove
theorem geometric_sequence_150th_term :
  nth_term 150 = -5 * 2^149 :=
by
  sorry

end geometric_sequence_150th_term_l1353_135375


namespace x_intercept_of_parabola_l1353_135308

theorem x_intercept_of_parabola (a b c : ℝ)
    (h_vertex : ∀ x, (a * (x - 5)^2 + 9 = y) → (x, y) = (5, 9))
    (h_intercept : ∀ x, (a * x^2 + b * x + c = 0) → x = 0 ∨ y = 0) :
    ∃ x0 : ℝ, x0 = 10 :=
by
  sorry

end x_intercept_of_parabola_l1353_135308


namespace evaluate_expression_l1353_135343

theorem evaluate_expression : (1 - 1/4) / (1 - 2/3) + 1/6 = 29/12 :=
by
  sorry

end evaluate_expression_l1353_135343


namespace candy_cost_l1353_135319

theorem candy_cost (candy_cost_in_cents : ℕ) (pieces : ℕ) (dollar_in_cents : ℕ)
  (h1 : candy_cost_in_cents = 2) (h2 : pieces = 500) (h3 : dollar_in_cents = 100) :
  (pieces * candy_cost_in_cents) / dollar_in_cents = 10 :=
by
  sorry

end candy_cost_l1353_135319


namespace max_value_M_l1353_135374

theorem max_value_M : 
  ∃ t : ℝ, (t = (3 / (4 ^ (1 / 3)))) ∧ 
    (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → 
      a^3 + b^3 + c^3 - 3 * a * b * c ≥ t * (a * b^2 + b * c^2 + c * a^2 - 3 * a * b * c)) :=
sorry

end max_value_M_l1353_135374


namespace min_sqrt_eq_sum_sqrt_implies_param_l1353_135312

noncomputable def sqrt (x : ℝ) : ℝ := Real.sqrt x

theorem min_sqrt_eq_sum_sqrt_implies_param (a b c : ℝ) (r s t : ℝ)
    (h1 : 0 < a ∧ a ≤ 1)
    (h2 : 0 < b ∧ b ≤ 1)
    (h3 : 0 < c ∧ c ≤ 1)
    (h4 : min (sqrt ((a * b + 1) / (a * b * c))) (min (sqrt ((b * c + 1) / (a * b * c))) (sqrt ((a * c + 1) / (a * b * c)))) 
          = (sqrt ((1 - a) / a) + sqrt ((1 - b) / b) + sqrt ((1 - c) / c))) :
    ∃ r, a = 1 / (1 + r^2) ∧ b = 1 / (1 + (1 / r^2)) ∧ c = (r + 1 / r)^2 / (1 + (r + 1 / r)^2) :=
sorry

end min_sqrt_eq_sum_sqrt_implies_param_l1353_135312


namespace symmetric_abs_necessary_not_sufficient_l1353_135359

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def y_axis_symmetric (f : ℝ → ℝ) : Prop :=
  ∀ x, |f (-x)| = |f x|

theorem symmetric_abs_necessary_not_sufficient (f : ℝ → ℝ) :
  is_odd_function f → y_axis_symmetric f := sorry

end symmetric_abs_necessary_not_sufficient_l1353_135359


namespace ruby_total_classes_l1353_135300

noncomputable def average_price_per_class (pack_cost : ℝ) (pack_classes : ℕ) : ℝ :=
  pack_cost / pack_classes

noncomputable def additional_class_price (average_price : ℝ) : ℝ :=
  average_price + (1/3 * average_price)

noncomputable def total_classes_taken (total_payment : ℝ) (pack_cost : ℝ) (pack_classes : ℕ) : ℕ :=
  let avg_price := average_price_per_class pack_cost pack_classes
  let additional_price := additional_class_price avg_price
  let additional_classes := (total_payment - pack_cost) / additional_price
  pack_classes + Nat.floor additional_classes -- We use Nat.floor to convert from real to natural number of classes

theorem ruby_total_classes 
  (pack_cost : ℝ) 
  (pack_classes : ℕ) 
  (total_payment : ℝ) 
  (h_pack_cost : pack_cost = 75) 
  (h_pack_classes : pack_classes = 10) 
  (h_total_payment : total_payment = 105) :
  total_classes_taken total_payment pack_cost pack_classes = 13 :=
by
  -- The proof would go here
  sorry

end ruby_total_classes_l1353_135300


namespace part1_part2_l1353_135314

variables (a b : ℝ)

theorem part1 (h₀ : a > 0) (h₁ : b > 0) (h₂ : ab = a + b + 8) : ab ≥ 16 :=
sorry

theorem part2 (h₀ : a > 0) (h₁ : b > 0) (h₂ : ab = a + b + 8) :
  ∃ (a b : ℝ), a = 7 ∧ b = 5 / 2 ∧ a + 4 * b = 17 :=
sorry

end part1_part2_l1353_135314


namespace algebraic_expression_value_l1353_135355

-- Define the equation and its roots.
def quadratic_eq (x : ℝ) : Prop := 2 * x^2 - 3 * x + 1 = 0

def is_root (x : ℝ) : Prop := quadratic_eq x

-- The main theorem.
theorem algebraic_expression_value (x1 x2 : ℝ) (h1 : is_root x1) (h2 : is_root x2) :
  (x1 + x2) / (1 + x1 * x2) = 1 :=
sorry

end algebraic_expression_value_l1353_135355


namespace gcf_60_90_l1353_135326

theorem gcf_60_90 : Nat.gcd 60 90 = 30 := by
  sorry

end gcf_60_90_l1353_135326


namespace store_discount_l1353_135341

theorem store_discount (P : ℝ) :
  let P1 := 0.9 * P
  let P2 := 0.86 * P1
  P2 = 0.774 * P :=
by
  let P1 := 0.9 * P
  let P2 := 0.86 * P1
  sorry

end store_discount_l1353_135341


namespace prize_distribution_l1353_135340

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem prize_distribution :
  let total_ways := 
    (binomial_coefficient 7 3) * 5 * (Nat.factorial 4) + 
    (binomial_coefficient 7 2 * binomial_coefficient 5 2 / 2) * 
    (binomial_coefficient 5 2) * (Nat.factorial 3)
  total_ways = 10500 :=
by 
  sorry

end prize_distribution_l1353_135340


namespace loaned_out_books_l1353_135339

def initial_books : ℕ := 75
def added_books : ℕ := 10 + 15 + 6
def removed_books : ℕ := 3 + 2 + 4
def end_books : ℕ := 90
def return_percentage : ℝ := 0.80

theorem loaned_out_books (L : ℕ) :
  (end_books - initial_books = added_books - removed_books - ⌊(1 - return_percentage) * L⌋) →
  (L = 35) :=
sorry

end loaned_out_books_l1353_135339


namespace overall_average_score_l1353_135361

variables (average_male average_female sum_male sum_female total_sum : ℕ)
variables (count_male count_female total_count : ℕ)

def average_score (sum : ℕ) (count : ℕ) : ℕ := sum / count

theorem overall_average_score
  (average_male : ℕ := 84)
  (count_male : ℕ := 8)
  (average_female : ℕ := 92)
  (count_female : ℕ := 24)
  (sum_male : ℕ := count_male * average_male)
  (sum_female : ℕ := count_female * average_female)
  (total_sum : ℕ := sum_male + sum_female)
  (total_count : ℕ := count_male + count_female) :
  average_score total_sum total_count = 90 := 
sorry

end overall_average_score_l1353_135361


namespace candidate_failed_by_25_marks_l1353_135350

-- Define the given conditions
def maximum_marks : ℝ := 127.27
def passing_percentage : ℝ := 0.55
def marks_secured : ℝ := 45

-- Define the minimum passing marks
def minimum_passing_marks : ℝ := passing_percentage * maximum_marks

-- Define the number of failing marks the candidate missed
def failing_marks : ℝ := minimum_passing_marks - marks_secured

-- Define the main theorem to prove the candidate failed by 25 marks
theorem candidate_failed_by_25_marks :
  failing_marks = 25 := 
by
  sorry

end candidate_failed_by_25_marks_l1353_135350


namespace smallest_divisor_28_l1353_135345

theorem smallest_divisor_28 : ∃ (d : ℕ), d > 0 ∧ d ∣ 28 ∧ ∀ (d' : ℕ), d' > 0 ∧ d' ∣ 28 → d ≤ d' := by
  sorry

end smallest_divisor_28_l1353_135345


namespace sqrt_square_multiply_l1353_135313

theorem sqrt_square_multiply (a : ℝ) (h : a = 49284) :
  (Real.sqrt a)^2 * 3 = 147852 :=
by
  sorry

end sqrt_square_multiply_l1353_135313


namespace flavors_needed_this_year_l1353_135380

def num_flavors_total : ℕ := 100

def num_flavors_two_years_ago : ℕ := num_flavors_total / 4

def num_flavors_last_year : ℕ := 2 * num_flavors_two_years_ago

def num_flavors_tried_so_far : ℕ := num_flavors_two_years_ago + num_flavors_last_year

theorem flavors_needed_this_year : 
  (num_flavors_total - num_flavors_tried_so_far) = 25 := by {
  sorry
}

end flavors_needed_this_year_l1353_135380


namespace right_triangle_area_l1353_135310

theorem right_triangle_area :
  ∃ (a b c : ℕ), a + b + c = 12 ∧ a * a + b * b = c * c ∧ (1/2 : ℝ) * a * b = 6 := 
sorry

end right_triangle_area_l1353_135310


namespace derivative_of_f_at_pi_over_2_l1353_135333

noncomputable def f (x : Real) := 5 * Real.sin x

theorem derivative_of_f_at_pi_over_2 :
  deriv f (Real.pi / 2) = 0 :=
by
  -- The proof is omitted
  sorry

end derivative_of_f_at_pi_over_2_l1353_135333


namespace monthly_interest_rate_l1353_135356

-- Define the principal amount (initial amount).
def principal : ℝ := 200

-- Define the final amount after 2 months (A).
def amount_after_two_months : ℝ := 222

-- Define the number of months (n).
def months : ℕ := 2

-- Define the monthly interest rate (r) we need to prove.
def interest_rate : ℝ := 0.053

-- Main statement to prove
theorem monthly_interest_rate :
  amount_after_two_months = principal * (1 + interest_rate)^months :=
sorry

end monthly_interest_rate_l1353_135356


namespace number_of_sheep_l1353_135395

theorem number_of_sheep (s d : ℕ) 
  (h1 : s + d = 15)
  (h2 : 4 * s + 2 * d = 22 + 2 * (s + d)) : 
  s = 11 :=
by
  sorry

end number_of_sheep_l1353_135395


namespace number_of_workers_l1353_135357

theorem number_of_workers (W C : ℕ) (h1 : W * C = 300000) (h2 : W * (C + 50) = 350000) : W = 1000 :=
sorry

end number_of_workers_l1353_135357


namespace square_area_less_than_circle_area_l1353_135370

theorem square_area_less_than_circle_area (a : ℝ) (ha : 0 < a) :
    let S1 := (a / 4) ^ 2
    let r := a / (2 * Real.pi)
    let S2 := Real.pi * r^2
    (S1 < S2) := by
sorry

end square_area_less_than_circle_area_l1353_135370


namespace cone_volume_l1353_135378

theorem cone_volume (l : ℝ) (S_side : ℝ) (h r V : ℝ)
  (hl : l = 10)
  (hS : S_side = 60 * Real.pi)
  (hr : S_side = π * r * l)
  (hh : h = Real.sqrt (l^2 - r^2))
  (hV : V = (1/3) * π * r^2 * h) :
  V = 96 * Real.pi := 
sorry

end cone_volume_l1353_135378


namespace distinct_integer_roots_iff_l1353_135324

theorem distinct_integer_roots_iff (a : ℤ) :
  (∃ x y : ℤ, x ≠ y ∧ 2 * x^2 - a * x + 2 * a = 0 ∧ 2 * y^2 - a * y + 2 * a = 0) ↔ a = -2 ∨ a = 18 :=
by
  sorry

end distinct_integer_roots_iff_l1353_135324


namespace largest_integral_solution_l1353_135330

noncomputable def largest_integral_value : ℤ :=
  let a : ℚ := 1 / 4
  let b : ℚ := 7 / 11 
  let lower_bound : ℚ := 7 * a
  let upper_bound : ℚ := 7 * b
  let x := 3  -- The largest integral value within the bounds
  x

-- A theorem to prove that x = 3 satisfies the inequality conditions and is the largest integer.
theorem largest_integral_solution (x : ℤ) (h₁ : 1 / 4 < x / 7) (h₂ : x / 7 < 7 / 11) : x = 3 := by
  sorry

end largest_integral_solution_l1353_135330


namespace factorize_x_cube_minus_4x_l1353_135331

theorem factorize_x_cube_minus_4x (x : ℝ) : x^3 - 4 * x = x * (x + 2) * (x - 2) := 
by
  -- Continue the proof from here
  sorry

end factorize_x_cube_minus_4x_l1353_135331


namespace perpendicular_lines_l1353_135384

theorem perpendicular_lines (a : ℝ) : 
  (3 * y + x + 4 = 0) → 
  (4 * y + a * x + 5 = 0) → 
  (∀ x y, x ≠ 0 ∧ y ≠ 0 → - (1 / 3 : ℝ) * - (a / 4 : ℝ) = -1) → 
  a = -12 := 
by
  intros h1 h2 h_perpendicularity
  sorry

end perpendicular_lines_l1353_135384


namespace range_of_m_l1353_135327

noncomputable def p (x : ℝ) : Prop := (x^3 - 4*x) / (2*x) ≤ 0
noncomputable def q (x m : ℝ) : Prop := (x^2 - (2*m + 1)*x + m^2 + m) ≤ 0

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, p x → q x m) ∧ ¬ (∀ x : ℝ, p x → q x m) ↔ m ∈ Set.Ico (-2 : ℝ) (-1) ∪ Set.Ioc 0 (1 : ℝ) :=
by
  sorry

end range_of_m_l1353_135327


namespace proof_l_squared_l1353_135383

noncomputable def longest_line_segment (diameter : ℝ) (sectors : ℕ) : ℝ :=
  let R := diameter / 2
  let theta := (2 * Real.pi) / sectors
  2 * R * (Real.sin (theta / 2))

theorem proof_l_squared :
  let diameter := 18
  let sectors := 4
  let l := longest_line_segment diameter sectors
  l^2 = 162 := by
  let diameter := 18
  let sectors := 4
  let l := longest_line_segment diameter sectors
  have h : l^2 = 162 := sorry
  exact h

end proof_l_squared_l1353_135383


namespace volunteer_comprehensive_score_is_92_l1353_135309

noncomputable def written_score : ℝ := 90
noncomputable def trial_lecture_score : ℝ := 94
noncomputable def interview_score : ℝ := 90

noncomputable def written_weight : ℝ := 0.3
noncomputable def trial_lecture_weight : ℝ := 0.5
noncomputable def interview_weight : ℝ := 0.2

noncomputable def comprehensive_score : ℝ :=
  written_score * written_weight +
  trial_lecture_score * trial_lecture_weight +
  interview_score * interview_weight

theorem volunteer_comprehensive_score_is_92 :
  comprehensive_score = 92 := by
  sorry

end volunteer_comprehensive_score_is_92_l1353_135309


namespace percentOfNonUnionWomenIs90_l1353_135363

variable (totalEmployees : ℕ) (percentMen : ℚ) (percentUnionized : ℚ) (percentUnionizedMen : ℚ)

noncomputable def percentNonUnionWomen : ℚ :=
  let numberOfMen := percentMen * totalEmployees
  let numberOfUnionEmployees := percentUnionized * totalEmployees
  let numberOfUnionMen := percentUnionizedMen * numberOfUnionEmployees
  let numberOfNonUnionEmployees := totalEmployees - numberOfUnionEmployees
  let numberOfNonUnionMen := numberOfMen - numberOfUnionMen
  let numberOfNonUnionWomen := numberOfNonUnionEmployees - numberOfNonUnionMen
  (numberOfNonUnionWomen / numberOfNonUnionEmployees) * 100

theorem percentOfNonUnionWomenIs90
  (h1 : percentMen = 46 / 100)
  (h2 : percentUnionized = 60 / 100)
  (h3 : percentUnionizedMen = 70 / 100) : percentNonUnionWomen 100 46 60 70 = 90 :=
sorry

end percentOfNonUnionWomenIs90_l1353_135363


namespace find_alpha_l1353_135335

noncomputable def parametric_eq_line (α t : Real) : Real × Real :=
  (1 + t * Real.cos α, t * Real.sin α)

def cartesian_eq_curve (x y : Real) : Prop :=
  y^2 = 4 * x

def intersection_condition (α t₁ t₂ : Real) : Prop :=
  Real.sin α ≠ 0 ∧ 
  (1 + t₁ * Real.cos α, t₁ * Real.sin α) = (1 + t₂ * Real.cos α, t₂ * Real.sin α) ∧ 
  Real.sqrt ((t₁ + t₂)^2 - 4 * (-4 / (Real.sin α)^2)) = 8

theorem find_alpha (α : Real) (t₁ t₂ : Real) 
  (h1: 0 < α) (h2: α < π) (h3: intersection_condition α t₁ t₂) : 
  α = π/4 ∨ α = 3*π/4 :=
by 
  sorry

end find_alpha_l1353_135335


namespace fraction_comparison_l1353_135399

theorem fraction_comparison (a b c d : ℝ) 
  (h1 : (a / b) < (c / d))
  (h2 : b > d) (h3 : d > 0) :
  (a + c) / (b + d) < (1 / 2) * ((a / b) + (c / d)) :=
by
  sorry

end fraction_comparison_l1353_135399


namespace exterior_angle_decreases_l1353_135304

theorem exterior_angle_decreases (n : ℕ) (hn : n ≥ 3) (n' : ℕ) (hn' : n' ≥ n) :
  (360 : ℝ) / n' < (360 : ℝ) / n := by sorry

end exterior_angle_decreases_l1353_135304


namespace total_pieces_of_clothing_l1353_135390

def number_of_pieces_per_drawer : ℕ := 2
def number_of_drawers : ℕ := 4

theorem total_pieces_of_clothing : 
  (number_of_pieces_per_drawer * number_of_drawers = 8) :=
by sorry

end total_pieces_of_clothing_l1353_135390


namespace radicals_like_simplest_forms_l1353_135318

theorem radicals_like_simplest_forms (a b : ℝ) (h1 : 2 * a + b = 7) (h2 : a = b + 2) :
  a = 3 ∧ b = 1 :=
by
  sorry

end radicals_like_simplest_forms_l1353_135318


namespace rectangle_inscribed_circle_hypotenuse_l1353_135306

open Real

theorem rectangle_inscribed_circle_hypotenuse
  (AB BC : ℝ)
  (h_AB : AB = 20)
  (h_BC : BC = 10)
  (r : ℝ)
  (h_r : r = 10 / 3) :
  sqrt ((AB - 2 * r) ^ 2 + BC ^ 2) = 50 / 3 :=
by {
  sorry
}

end rectangle_inscribed_circle_hypotenuse_l1353_135306


namespace adults_tickets_sold_eq_1200_l1353_135305

variable (A : ℕ)
variable (S : ℕ := 300) -- Number of student tickets
variable (P_adult : ℕ := 12) -- Price per adult ticket
variable (P_student : ℕ := 6) -- Price per student ticket
variable (total_tickets : ℕ := 1500) -- Total tickets sold
variable (total_amount : ℕ := 16200) -- Total amount collected

theorem adults_tickets_sold_eq_1200
  (h1 : S = 300)
  (h2 : A + S = total_tickets)
  (h3 : P_adult * A + P_student * S = total_amount) :
  A = 1200 := by
  sorry

end adults_tickets_sold_eq_1200_l1353_135305


namespace compare_a_b_l1353_135362

def a := 1 / 3 + 1 / 4
def b := 1 / 5 + 1 / 6 + 1 / 7

theorem compare_a_b : a > b := 
  sorry

end compare_a_b_l1353_135362


namespace tangents_of_convex_quad_l1353_135322

theorem tangents_of_convex_quad (
  α β γ δ : ℝ
) (m : ℝ) (h₀ : α + β + γ + δ = 2 * Real.pi) (h₁ : 0 < α ∧ α < Real.pi) (h₂ : 0 < β ∧ β < Real.pi) 
  (h₃ : 0 < γ ∧ γ < Real.pi) (h₄ : 0 < δ ∧ δ < Real.pi) (t1 : Real.tan α = m) :
  ¬ (Real.tan β = m ∧ Real.tan γ = m ∧ Real.tan δ = m) :=
sorry

end tangents_of_convex_quad_l1353_135322


namespace time_interval_for_7_students_l1353_135352

-- Definitions from conditions
def students_per_ride : ℕ := 7
def total_students : ℕ := 21
def total_time : ℕ := 15

-- Statement of the problem
theorem time_interval_for_7_students : (total_time / (total_students / students_per_ride)) = 5 := 
by sorry

end time_interval_for_7_students_l1353_135352


namespace single_digit_solution_l1353_135366

theorem single_digit_solution :
  ∃ A : ℕ, A < 10 ∧ A^3 = 210 + A ∧ A = 6 :=
by
  existsi 6
  sorry

end single_digit_solution_l1353_135366


namespace find_equation_of_line_l1353_135307

-- Define the conditions
def line_passes_through_A (m b : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (1, 1) ∧ A.2 = -A.1 + b

def intercepts_equal (m b : ℝ) : Prop :=
  b = m

-- The goal to prove the equations of the line
theorem find_equation_of_line :
  ∃ (m b : ℝ), line_passes_through_A m b (1, 1) ∧ intercepts_equal m b ↔ 
  (∃ m b : ℝ, (m = -1 ∧ b = 2) ∨ (m = 1 ∧ b = 0)) :=
sorry

end find_equation_of_line_l1353_135307


namespace original_length_before_sharpening_l1353_135382

/-- Define the current length of the pencil after sharpening -/
def current_length : ℕ := 14

/-- Define the length of the pencil that was sharpened off -/
def sharpened_off_length : ℕ := 17

/-- Prove that the original length of the pencil before sharpening was 31 inches -/
theorem original_length_before_sharpening : current_length + sharpened_off_length = 31 := by
  sorry

end original_length_before_sharpening_l1353_135382


namespace tan_double_angle_l1353_135338

open Real

theorem tan_double_angle (α : ℝ) (h : (sin α + cos α) / (sin α - cos α) = 1 / 2) : tan (2 * α) = 3 / 4 := 
by 
  sorry

end tan_double_angle_l1353_135338


namespace find_ruv_l1353_135391

theorem find_ruv (u v : ℝ) : 
  (∃ u v : ℝ, 
    (3 + 8 * u + 5, 1 - 4 * u + 2) = (4 + -3 * v + 5, 2 + 4 * v + 2)) →
  (u = -1/2 ∧ v = -1) :=
by
  intros H
  sorry

end find_ruv_l1353_135391


namespace age_intervals_l1353_135387

theorem age_intervals (A1 A2 A3 A4 A5 : ℝ) (x : ℝ) (h1 : A1 = 7)
  (h2 : A2 = A1 + x) (h3 : A3 = A1 + 2 * x) (h4 : A4 = A1 + 3 * x) (h5 : A5 = A1 + 4 * x)
  (sum_ages : A1 + A2 + A3 + A4 + A5 = 65) :
  x = 3.7 :=
by
  -- Sketch a proof or leave 'sorry' for completeness
  sorry

end age_intervals_l1353_135387


namespace window_width_l1353_135389

theorem window_width (h_pane_height : ℕ) (h_to_w_ratio_num : ℕ) (h_to_w_ratio_den : ℕ) (gaps : ℕ) 
(border : ℕ) (columns : ℕ) 
(panes_per_row : ℕ) (pane_height : ℕ) 
(heights_equal : h_pane_height = pane_height)
(ratio : h_to_w_ratio_num * pane_height = h_to_w_ratio_den * panes_per_row)
: columns * (h_to_w_ratio_den * pane_height / h_to_w_ratio_num) + 
  gaps + 2 * border = 57 := sorry

end window_width_l1353_135389


namespace sum_of_min_max_l1353_135377

-- Define the necessary parameters and conditions
variables (n k : ℕ)
  (h_pos_nk : 0 < n ∧ 0 < k)
  (f : ℕ → ℕ)
  (h_toppings : ∀ t, (0 ≤ f t ∧ f t ≤ n) ∧ (f t + f (t + k) % (2 * k) = n))
  (m M : ℕ)
  (h_m : ∀ t, m ≤ f t)
  (h_M : ∀ t, f t ≤ M)
  (h_min_max : ∃ t_min t_max, m = f t_min ∧ M = f t_max)

-- The goal is to prove that the sum of m and M equals n
theorem sum_of_min_max (n k : ℕ) (h_pos_nk : 0 < n ∧ 0 < k)
  (f : ℕ → ℕ) (h_toppings : ∀ t, (0 ≤ f t ∧ f t ≤ n) ∧ (f t + f (t + k) % (2 * k) = n))
  (m M : ℕ) (h_m : ∀ t, m ≤ f t)
  (h_M : ∀ t, f t ≤ M)
  (h_min_max : ∃ t_min t_max, m = f t_min ∧ M = f t_max) :
  m + M = n := 
sorry

end sum_of_min_max_l1353_135377


namespace total_blocks_l1353_135379

theorem total_blocks (red_blocks yellow_blocks blue_blocks : ℕ) 
  (h1 : red_blocks = 18) 
  (h2 : yellow_blocks = red_blocks + 7) 
  (h3 : blue_blocks = red_blocks + 14) : 
  red_blocks + yellow_blocks + blue_blocks = 75 := 
by
  sorry

end total_blocks_l1353_135379


namespace find_triangle_l1353_135325

theorem find_triangle : ∀ (triangle : ℕ), (∀ (d : ℕ), 0 ≤ d ∧ d ≤ 9) → (5 * 3 + triangle = 12 * triangle + 4) → triangle = 1 :=
by
  sorry

end find_triangle_l1353_135325


namespace yellow_candy_percentage_l1353_135315

variable (b : ℝ) (y : ℝ) (r : ℝ)

-- Conditions from the problem
-- 14% more yellow candies than blue candies
axiom yellow_candies : y = 1.14 * b
-- 14% fewer red candies than blue candies
axiom red_candies : r = 0.86 * b
-- Total number of candies equals 1 (or 100%)
axiom total_candies : r + b + y = 1

-- Question to prove: The percentage of yellow candies in the jar is 38%
theorem yellow_candy_percentage  : y = 0.38 := by
  sorry

end yellow_candy_percentage_l1353_135315


namespace min_value_f_l1353_135369

theorem min_value_f
  (a b c : ℝ)
  (α β γ : ℤ)
  (hα : α = 1 ∨ α = -1)
  (hβ : β = 1 ∨ β = -1)
  (hγ : γ = 1 ∨ γ = -1)
  (h : a * α + b * β + c * γ = 0) :
  (∃ f_min : ℝ, f_min = ( ((a ^ 3 + b ^ 3 + c ^ 3) / (a * b * c)) ^ 2) ∧ f_min = 9) :=
sorry

end min_value_f_l1353_135369


namespace result_of_fractions_mult_l1353_135397

theorem result_of_fractions_mult (a b c d : ℚ) (x : ℕ) :
  a = 3 / 4 →
  b = 1 / 2 →
  c = 2 / 5 →
  d = 5100 →
  a * b * c * d = 765 := by
  sorry

end result_of_fractions_mult_l1353_135397


namespace scallops_per_pound_l1353_135336

theorem scallops_per_pound
  (cost_per_pound : ℝ)
  (scallops_per_person : ℕ)
  (number_of_people : ℕ)
  (total_cost : ℝ)
  (total_scallops : ℕ)
  (total_pounds : ℝ)
  (scallops_per_pound : ℕ)
  (h1 : cost_per_pound = 24)
  (h2 : scallops_per_person = 2)
  (h3 : number_of_people = 8)
  (h4 : total_cost = 48)
  (h5 : total_scallops = scallops_per_person * number_of_people)
  (h6 : total_pounds = total_cost / cost_per_pound)
  (h7 : scallops_per_pound = total_scallops / total_pounds) : 
  scallops_per_pound = 8 :=
sorry

end scallops_per_pound_l1353_135336


namespace arithmetic_sequence_product_l1353_135365

theorem arithmetic_sequence_product
  (a d : ℤ)
  (h1 : a + 5 * d = 17)
  (h2 : d = 2) :
  (a + 2 * d) * (a + 3 * d) = 143 :=
by
  sorry

end arithmetic_sequence_product_l1353_135365


namespace valid_third_side_length_l1353_135394

theorem valid_third_side_length (x : ℝ) : 4 < x ∧ x < 14 ↔ (((5 : ℝ) + 9 > x) ∧ (x + 5 > 9) ∧ (x + 9 > 5)) :=
by 
  sorry

end valid_third_side_length_l1353_135394


namespace diagonals_in_eight_sided_polygon_l1353_135398

-- Definitions based on the conditions
def n := 8  -- Number of sides
def right_angles := 2  -- Number of right angles

-- Calculating the number of diagonals using the formula
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Lean statement for the problem
theorem diagonals_in_eight_sided_polygon : num_diagonals n = 20 :=
by
  -- Substitute n = 8 into the formula and simplify
  sorry

end diagonals_in_eight_sided_polygon_l1353_135398


namespace jan_keeps_on_hand_l1353_135353

theorem jan_keeps_on_hand (total_length : ℕ) (section_length : ℕ) (friend_fraction : ℚ) (storage_fraction : ℚ) 
  (total_sections : ℕ) (sections_to_friend : ℕ) (remaining_sections : ℕ) (sections_in_storage : ℕ) (sections_on_hand : ℕ) :
  total_length = 1000 → section_length = 25 → friend_fraction = 1 / 4 → storage_fraction = 1 / 2 →
  total_sections = total_length / section_length →
  sections_to_friend = friend_fraction * total_sections →
  remaining_sections = total_sections - sections_to_friend →
  sections_in_storage = storage_fraction * remaining_sections →
  sections_on_hand = remaining_sections - sections_in_storage →
  sections_on_hand = 15 :=
by sorry

end jan_keeps_on_hand_l1353_135353


namespace Caden_total_money_l1353_135354

theorem Caden_total_money (p n d q : ℕ) (hp : p = 120)
    (hn : p = 3 * n) 
    (hd : n = 5 * d)
    (hq : q = 2 * d) :
    (p * 1 / 100 + n * 5 / 100 + d * 10 / 100 + q * 25 / 100) = 8 := 
by
  sorry

end Caden_total_money_l1353_135354


namespace sam_friend_points_l1353_135337

theorem sam_friend_points (sam_points total_points : ℕ) (h1 : sam_points = 75) (h2 : total_points = 87) :
  total_points - sam_points = 12 :=
by sorry

end sam_friend_points_l1353_135337


namespace value_of_fraction_l1353_135385

theorem value_of_fraction (a b c d e f : ℚ) (h1 : a / b = 1 / 3) (h2 : c / d = 1 / 3) (h3 : e / f = 1 / 3) :
  (3 * a - 2 * c + e) / (3 * b - 2 * d + f) = 1 / 3 :=
by
  sorry

end value_of_fraction_l1353_135385


namespace large_integer_value_l1353_135373

theorem large_integer_value :
  (2 + 3) * (2^2 + 3^2) * (2^4 - 3^4) * (2^8 + 3^8) * (2^16 - 3^16) * (2^32 + 3^32) * (2^64 - 3^64)
  > 0 := 
by
  sorry

end large_integer_value_l1353_135373


namespace number_of_unique_four_digit_numbers_from_2004_l1353_135372

-- Definitions representing the conditions
def is_four_digit_number (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000
def uses_digits_from_2004 (n : ℕ) : Prop := 
  ∀ d ∈ [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10], d ∈ [0, 2, 4]

-- The proposition we need to prove
theorem number_of_unique_four_digit_numbers_from_2004 :
  ∃ n : ℕ, is_four_digit_number n ∧ uses_digits_from_2004 n ∧ n = 6 := 
sorry

end number_of_unique_four_digit_numbers_from_2004_l1353_135372


namespace pipe_A_fill_time_l1353_135392

theorem pipe_A_fill_time (t : ℝ) (h1 : t > 0) (h2 : ∃ tA tB, tA = t ∧ tB = t / 6 ∧ (tA + tB) = 3) : t = 21 :=
by
  sorry

end pipe_A_fill_time_l1353_135392


namespace sale_price_for_50_percent_profit_l1353_135344

theorem sale_price_for_50_percent_profit
  (C L: ℝ)
  (h1: 892 - C = C - L)
  (h2: 1005 = 1.5 * C) :
  1.5 * C = 1005 :=
by
  sorry

end sale_price_for_50_percent_profit_l1353_135344


namespace lindsey_final_money_l1353_135311

-- Define the savings in each month
def save_sep := 50
def save_oct := 37
def save_nov := 11

-- Total savings over the three months
def total_savings := save_sep + save_oct + save_nov

-- Condition for Mom's contribution
def mom_contribution := if total_savings > 75 then 25 else 0

-- Total savings including mom's contribution
def total_with_mom := total_savings + mom_contribution

-- Amount spent on the video game
def spent := 87

-- Final amount left
def final_amount := total_with_mom - spent

-- Proof statement
theorem lindsey_final_money : final_amount = 36 := by
  sorry

end lindsey_final_money_l1353_135311


namespace other_endpoint_sum_l1353_135332

def endpoint_sum (A B M : (ℝ × ℝ)) : ℝ := 
  let (Ax, Ay) := A
  let (Mx, My) := M
  let (Bx, By) := B
  Bx + By

theorem other_endpoint_sum (A M : (ℝ × ℝ)) (hA : A = (6, 1)) (hM : M = (5, 7)) :
  ∃ B : (ℝ × ℝ), endpoint_sum A B M = 17 :=
by
  use (4, 13)
  rw [endpoint_sum, hA, hM]
  simp
  sorry

end other_endpoint_sum_l1353_135332


namespace coeff_x4_in_expansion_l1353_135386

open Nat

noncomputable def binom (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

noncomputable def coefficient_x4_term : ℕ := binom 9 4

noncomputable def constant_term : ℕ := 243 * 4

theorem coeff_x4_in_expansion : coefficient_x4_term * 972 * Real.sqrt 2 = 122472 * Real.sqrt 2 :=
by
  sorry

end coeff_x4_in_expansion_l1353_135386


namespace range_of_m_increasing_function_l1353_135320

theorem range_of_m_increasing_function :
  (2 : ℝ) ≤ m ∧ m ≤ 4 ↔ ∀ x : ℝ, (1 / 3 : ℝ) * x ^ 3 - (4 * m - 1) * x ^ 2 + (15 * m ^ 2 - 2 * m - 7) * x + 2 ≤ 
                                 ((1 / 3 : ℝ) * (x + 1) ^ 3 - (4 * m - 1) * (x + 1) ^ 2 + (15 * m ^ 2 - 2 * m - 7) * (x + 1) + 2) :=
by
  sorry

end range_of_m_increasing_function_l1353_135320


namespace unique_solution_l1353_135360

def is_valid_func (f : ℕ → ℕ) : Prop :=
  ∀ n, f (f n) + f n = 2 * n + 2001 ∨ f (f n) + f n = 2 * n + 2002

theorem unique_solution (f : ℕ → ℕ) (hf : is_valid_func f) :
  ∀ n, f n = n + 667 :=
sorry

end unique_solution_l1353_135360


namespace balls_in_boxes_l1353_135348

theorem balls_in_boxes : 
  let balls := 5
  let boxes := 4
  (∀ (ball : Fin balls), Fin boxes) → (4^5 = 1024) := 
by
  intro h
  sorry

end balls_in_boxes_l1353_135348


namespace inequality_abc_ge_1_sqrt_abcd_l1353_135358

theorem inequality_abc_ge_1_sqrt_abcd
  (a b c d : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d)
  (h_sum : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + b + c + d) / 2 ≥ 1 + Real.sqrt (a * b * c * d) :=
by
  sorry

end inequality_abc_ge_1_sqrt_abcd_l1353_135358


namespace effective_writing_speed_is_750_l1353_135388

-- Definitions based on given conditions in problem part a)
def total_words : ℕ := 60000
def total_hours : ℕ := 100
def break_hours : ℕ := 20
def effective_hours : ℕ := total_hours - break_hours
def effective_writing_speed : ℕ := total_words / effective_hours

-- Statement to be proved
theorem effective_writing_speed_is_750 : effective_writing_speed = 750 := by
  sorry

end effective_writing_speed_is_750_l1353_135388


namespace fifth_friend_paid_l1353_135342

theorem fifth_friend_paid (a b c d e : ℝ)
  (h1 : a = (1/3) * (b + c + d + e))
  (h2 : b = (1/4) * (a + c + d + e))
  (h3 : c = (1/5) * (a + b + d + e))
  (h4 : a + b + c + d + e = 120) :
  e = 40 :=
sorry

end fifth_friend_paid_l1353_135342


namespace octagon_area_equals_eight_one_plus_sqrt_two_l1353_135316

theorem octagon_area_equals_eight_one_plus_sqrt_two
  (a b : ℝ)
  (h1 : 4 * a = 8 * b)
  (h2 : a ^ 2 = 16) :
  2 * (1 + Real.sqrt 2) * b ^ 2 = 8 * (1 + Real.sqrt 2) :=
by
  sorry

end octagon_area_equals_eight_one_plus_sqrt_two_l1353_135316


namespace sqrt_extraction_count_l1353_135393

theorem sqrt_extraction_count (p : ℕ) [Fact p.Prime] : 
    ∃ k, k = (p + 1) / 2 ∧ ∀ n < p, ∃ x < p, x^2 ≡ n [MOD p] ↔ n < k := 
by
  sorry

end sqrt_extraction_count_l1353_135393


namespace extremum_condition_l1353_135381

theorem extremum_condition (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = x^3 + a * x^2 + b * x + a^2)
  (h2 : f 1 = 10)
  (h3 : deriv f 1 = 0) :
  a + b = -7 :=
sorry

end extremum_condition_l1353_135381


namespace part1_union_part1_intersection_complement_part2_necessary_sufficient_condition_l1353_135334

-- Definitions of the sets and conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -4 < x ∧ x < 1}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 2}

-- Part 1
theorem part1_union (a : ℝ) (ha : a = 1) : 
  A ∪ B a = { x | -4 < x ∧ x ≤ 3 } :=
sorry

theorem part1_intersection_complement (a : ℝ) (ha : a = 1) : 
  A ∩ (U \ B a) = { x | -4 < x ∧ x < 0 } :=
sorry

-- Part 2
theorem part2_necessary_sufficient_condition (a : ℝ) : 
  (∀ x, x ∈ B a ↔ x ∈ A) ↔ (-3 < a ∧ a < -1) :=
sorry

end part1_union_part1_intersection_complement_part2_necessary_sufficient_condition_l1353_135334


namespace isosceles_triangle_range_l1353_135367

theorem isosceles_triangle_range (x : ℝ) (h1 : 0 < x) (h2 : 2 * x + (10 - 2 * x) = 10):
  (5 / 2) < x ∧ x < 5 :=
by
  sorry

end isosceles_triangle_range_l1353_135367


namespace find_a_solve_inequality_intervals_of_monotonicity_l1353_135396

-- Problem 1: Prove a = 2 given conditions
theorem find_a (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : Real.log 3 / Real.log a > Real.log 2 / Real.log a) 
    (h₃ : Real.log (2 * a) / Real.log a - Real.log a / Real.log a = 1) : a = 2 := 
  by
  sorry

-- Problem 2: Prove the solution interval for inequality
theorem solve_inequality (x a : ℝ) (h₀ : 1 < x) (h₁ : x < 3 / 2) : 
    Real.log (x - 1) / Real.log (1 / 3) > Real.log (a - x) / Real.log (1 / 3) :=
  by
  have ha : a = 2 := sorry
  sorry

-- Problem 3: Prove intervals of monotonicity for g(x)
theorem intervals_of_monotonicity (x : ℝ) : 
  (∀ x : ℝ, 0 < x → x ≤ 2 → (|Real.log x / Real.log 2 - 1| : ℝ) = 1 - Real.log x / Real.log 2) ∧ 
  (∀ x : ℝ, x > 2 → (|Real.log x / Real.log 2 - 1| : ℝ) = Real.log x / Real.log 2 - 1) :=
  by
  sorry

end find_a_solve_inequality_intervals_of_monotonicity_l1353_135396


namespace diameter_of_tripled_volume_sphere_l1353_135321

noncomputable def volume_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem diameter_of_tripled_volume_sphere :
  let r1 := 6
  let V1 := volume_sphere r1
  let V2 := 3 * V1
  let r2 := (V2 * 3 / (4 * Real.pi))^(1 / 3)
  let D := 2 * r2
  ∃ (a b : ℕ), (D = a * (b:ℝ)^(1 / 3) ∧ b ≠ 0 ∧ ∀ n : ℕ, n^3 ∣ b → n = 1) ∧ a + b = 15 :=
by
  sorry

end diameter_of_tripled_volume_sphere_l1353_135321


namespace field_area_l1353_135323

theorem field_area (L W : ℝ) (hL : L = 20) (h_fencing : 2 * W + L = 59) :
  L * W = 390 :=
by {
  -- We will skip the proof
  sorry
}

end field_area_l1353_135323


namespace squares_circles_intersections_l1353_135346

noncomputable def number_of_intersections (p1 p2 : (ℤ × ℤ)) (square_side : ℚ) (circle_radius : ℚ) : ℕ :=
sorry -- function definition placeholder

theorem squares_circles_intersections :
  let p1 := (0, 0)
  let p2 := (1009, 437)
  let square_side := (1 : ℚ) / 4
  let circle_radius := (1 : ℚ) / 8
  (number_of_intersections p1 p2 square_side circle_radius) = 526 := by
  sorry

end squares_circles_intersections_l1353_135346


namespace original_price_l1353_135329

theorem original_price (P : ℕ) (h : (1 / 8) * P = 8) : P = 64 :=
sorry

end original_price_l1353_135329


namespace number_of_circles_is_3_l1353_135302

-- Define the radius and diameter of the circles
def radius := 4
def diameter := 2 * radius

-- Given the total horizontal length
def total_horizontal_length := 24

-- Number of circles calculated as per the given conditions
def number_of_circles := total_horizontal_length / diameter

-- The proof statement to verify
theorem number_of_circles_is_3 : number_of_circles = 3 := by
  sorry

end number_of_circles_is_3_l1353_135302


namespace geometric_sequence_condition_l1353_135376

theorem geometric_sequence_condition (a : ℕ → ℝ) :
  (∀ n ≥ 2, a n = 2 * a (n-1)) → 
  (∃ r, r = 2 ∧ ∀ n ≥ 2, a n = r * a (n-1)) ∧ 
  (∃ b, b ≠ 0 ∧ ∀ n, a n = 0) :=
sorry

end geometric_sequence_condition_l1353_135376


namespace wedge_top_half_volume_l1353_135317

theorem wedge_top_half_volume (r : ℝ) (C : ℝ) (V : ℝ) : 
  (C = 18 * π) ∧ (C = 2 * π * r) ∧ (V = (4/3) * π * r^3) ∧ 
  (V / 3 / 2) = 162 * π :=
  sorry

end wedge_top_half_volume_l1353_135317


namespace diff_baseball_soccer_l1353_135328

variable (totalBalls soccerBalls basketballs tennisBalls baseballs volleyballs : ℕ)

axiom h1 : totalBalls = 145
axiom h2 : soccerBalls = 20
axiom h3 : basketballs = soccerBalls + 5
axiom h4 : tennisBalls = 2 * soccerBalls
axiom h5 : baseballs > soccerBalls
axiom h6 : volleyballs = 30

theorem diff_baseball_soccer : baseballs - soccerBalls = 10 :=
  by {
    sorry
  }

end diff_baseball_soccer_l1353_135328


namespace box_width_is_target_width_l1353_135301

-- Defining the conditions
def cube_volume : ℝ := 27
def box_length : ℝ := 8
def box_height : ℝ := 12
def max_cubes : ℕ := 24

-- Defining the target width we want to prove
def target_width : ℝ := 6.75

-- The proof statement
theorem box_width_is_target_width :
  ∃ w : ℝ,
  (∀ v : ℝ, (v = max_cubes * cube_volume) →
   ∀ l : ℝ, (l = box_length) →
   ∀ h : ℝ, (h = box_height) →
   v = l * w * h) →
   w = target_width :=
by
  sorry

end box_width_is_target_width_l1353_135301


namespace olivia_card_value_l1353_135371

theorem olivia_card_value (x : ℝ) (hx1 : 90 < x ∧ x < 180)
  (h_sin_pos : Real.sin x > 0) (h_cos_neg : Real.cos x < 0) (h_tan_neg : Real.tan x < 0)
  (h_olivia_distinguish : ∀ (a b c : ℝ), 
    (a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x) →
    (b = Real.sin x ∨ b = Real.cos x ∨ b = Real.tan x) →
    (c = Real.sin x ∨ c = Real.cos x ∨ c = Real.tan x) →
    (a ≠ b ∧ b ≠ c ∧ c ≠ a) →
    (a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x) →
    (b = Real.sin x ∨ b = Real.cos x ∨ b = Real.tan x) →
    (c = Real.sin x ∨ c = Real.cos x ∨ c = Real.tan x) →
    (∃! a, a = Real.sin x ∨ a = Real.cos x ∨ a = Real.tan x)) :
  Real.sin 135 = Real.cos 45 := 
sorry

end olivia_card_value_l1353_135371


namespace minimum_packs_needed_l1353_135303

theorem minimum_packs_needed (n : ℕ) :
  (∃ x y z : ℕ, 30 * x + 18 * y + 9 * z = 120 ∧ x + y + z = n ∧ x ≥ 2 ∧ z' = if x ≥ 2 then z + 1 else z) → n = 4 := 
by
  sorry

end minimum_packs_needed_l1353_135303


namespace work_completed_in_5_days_l1353_135347

-- Define the rates of work for A, B, and C
def rateA : ℚ := 1 / 15
def rateB : ℚ := 1 / 14
def rateC : ℚ := 1 / 16

-- Summing their rates to get the combined rate
def combined_rate : ℚ := rateA + rateB + rateC

-- This is the statement we need to prove, i.e., the time required for A, B, and C to finish the work together is 5 days.
theorem work_completed_in_5_days (hA : rateA = 1 / 15) (hB : rateB = 1 / 14) (hC : rateC = 1 / 16) :
  (1 / combined_rate) = 5 :=
by
  sorry

end work_completed_in_5_days_l1353_135347
