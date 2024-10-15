import Mathlib

namespace NUMINAMATH_GPT_line_segments_cannot_form_triangle_l67_6700

theorem line_segments_cannot_form_triangle (a : ℕ → ℕ) (h1 : a 1 = 1) (h2 : a 7 = 21)
    (h3 : ∀ n, a n < a (n+1)) (h4 : ∀ i j k, a i + a j ≤ a k) :
    a 6 = 13 :=
    sorry

end NUMINAMATH_GPT_line_segments_cannot_form_triangle_l67_6700


namespace NUMINAMATH_GPT_find_larger_number_l67_6793

-- Definitions based on the conditions
def larger_number (L S : ℕ) : Prop :=
  L - S = 1365 ∧ L = 6 * S + 20

-- The theorem to prove
theorem find_larger_number (L S : ℕ) (h : larger_number L S) : L = 1634 :=
by
  sorry  -- Proof would go here

end NUMINAMATH_GPT_find_larger_number_l67_6793


namespace NUMINAMATH_GPT_parametric_curve_intersects_l67_6758

noncomputable def curve_crosses_itself : Prop :=
  let t1 := Real.sqrt 11
  let t2 := -Real.sqrt 11
  let x (t : ℝ) := t^3 - t + 1
  let y (t : ℝ) := t^3 - 11*t + 11
  (x t1 = 10 * Real.sqrt 11 + 1) ∧ (y t1 = 11) ∧
  (x t2 = 10 * Real.sqrt 11 + 1) ∧ (y t2 = 11)

theorem parametric_curve_intersects : curve_crosses_itself :=
by
  sorry

end NUMINAMATH_GPT_parametric_curve_intersects_l67_6758


namespace NUMINAMATH_GPT_cubic_root_expression_l67_6794

theorem cubic_root_expression (u v w : ℂ) (huvwx : u * v * w ≠ 0)
  (h1 : u^3 - 6 * u^2 + 11 * u - 6 = 0)
  (h2 : v^3 - 6 * v^2 + 11 * v - 6 = 0)
  (h3 : w^3 - 6 * w^2 + 11 * w - 6 = 0) :
  (u * v / w) + (v * w / u) + (w * u / v) = 49 / 6 :=
sorry

end NUMINAMATH_GPT_cubic_root_expression_l67_6794


namespace NUMINAMATH_GPT_questions_for_second_project_l67_6738

open Nat

theorem questions_for_second_project (days_per_week : ℕ) (first_project_q : ℕ) (questions_per_day : ℕ) 
  (total_questions : ℕ) (second_project_q : ℕ) 
  (h1 : days_per_week = 7)
  (h2 : first_project_q = 518)
  (h3 : questions_per_day = 142)
  (h4 : total_questions = days_per_week * questions_per_day)
  (h5 : second_project_q = total_questions - first_project_q) :
  second_project_q = 476 :=
by
  -- we assume the solution steps as correct
  sorry

end NUMINAMATH_GPT_questions_for_second_project_l67_6738


namespace NUMINAMATH_GPT_binomial_133_133_l67_6751

theorem binomial_133_133 : @Nat.choose 133 133 = 1 := by   
sorry

end NUMINAMATH_GPT_binomial_133_133_l67_6751


namespace NUMINAMATH_GPT_gwen_math_problems_l67_6734

-- Problem statement
theorem gwen_math_problems (m : ℕ) (science_problems : ℕ := 11) (problems_finished_at_school : ℕ := 24) (problems_left_for_homework : ℕ := 5) 
  (h1 : m + science_problems = problems_finished_at_school + problems_left_for_homework) : m = 18 := 
by {
  sorry
}

end NUMINAMATH_GPT_gwen_math_problems_l67_6734


namespace NUMINAMATH_GPT_cost_per_metre_of_carpet_l67_6716

theorem cost_per_metre_of_carpet :
  (length_of_room = 18) →
  (breadth_of_room = 7.5) →
  (carpet_width = 0.75) →
  (total_cost = 810) →
  (cost_per_metre = 4.5) :=
by
  intros length_of_room breadth_of_room carpet_width total_cost
  sorry

end NUMINAMATH_GPT_cost_per_metre_of_carpet_l67_6716


namespace NUMINAMATH_GPT_prove_collinear_prove_perpendicular_l67_6786

noncomputable def vec_a : ℝ × ℝ := (1, 3)
noncomputable def vec_b : ℝ × ℝ := (3, -4)

def collinear (k : ℝ) : Prop :=
  let v1 := (k * 1 - 3, k * 3 + 4)
  let v2 := (1 + 3, 3 - 4)
  v1.1 * v2.2 = v1.2 * v2.1

def perpendicular (k : ℝ) : Prop :=
  let v1 := (k * 1 - 3, k * 3 + 4)
  let v2 := (1 + 3, 3 - 4)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem prove_collinear : collinear (-1) :=
by
  sorry

theorem prove_perpendicular : perpendicular (16) :=
by
  sorry

end NUMINAMATH_GPT_prove_collinear_prove_perpendicular_l67_6786


namespace NUMINAMATH_GPT_inequality_proof_l67_6710

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a / Real.sqrt (a^2 + 8 * b * c)) + 
  (b / Real.sqrt (b^2 + 8 * c * a)) + 
  (c / Real.sqrt (c^2 + 8 * a * b)) ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l67_6710


namespace NUMINAMATH_GPT_percentage_reduction_in_price_l67_6763

theorem percentage_reduction_in_price (P R : ℝ) (hR : R = 2.953846153846154)
  (h_condition : ∃ P, 65 / 12 * R = 40 - 24 / P) :
  ((P - R) / P) * 100 = 33.3 := by
  sorry

end NUMINAMATH_GPT_percentage_reduction_in_price_l67_6763


namespace NUMINAMATH_GPT_john_total_skateboarded_distance_l67_6772

noncomputable def total_skateboarded_distance (to_park: ℕ) (back_home: ℕ) : ℕ :=
  to_park + back_home

theorem john_total_skateboarded_distance :
  total_skateboarded_distance 10 10 = 20 :=
by
  sorry

end NUMINAMATH_GPT_john_total_skateboarded_distance_l67_6772


namespace NUMINAMATH_GPT_expression_value_l67_6737

noncomputable def f (x : ℝ) : ℝ := sorry

axiom f_add_prop (a b : ℝ) : f (a + b) = f a * f b
axiom f_one_val : f 1 = 2

theorem expression_value : 
  (f 1 ^ 2 + f 2) / f 1 + 
  (f 2 ^ 2 + f 4) / f 3 +
  (f 3 ^ 2 + f 6) / f 5 + 
  (f 4 ^ 2 + f 8) / f 7 
  = 16 := 
sorry

end NUMINAMATH_GPT_expression_value_l67_6737


namespace NUMINAMATH_GPT_border_area_is_198_l67_6719

-- We define the dimensions of the picture and the border width
def picture_height : ℝ := 12
def picture_width : ℝ := 15
def border_width : ℝ := 3

-- We compute the entire framed height and width
def framed_height : ℝ := picture_height + 2 * border_width
def framed_width : ℝ := picture_width + 2 * border_width

-- We compute the area of the picture and framed area
def picture_area : ℝ := picture_height * picture_width
def framed_area : ℝ := framed_height * framed_width

-- We compute the area of the border
def border_area : ℝ := framed_area - picture_area

-- Now we pose the theorem to prove the area of the border is 198 square inches
theorem border_area_is_198 : border_area = 198 := by
  sorry

end NUMINAMATH_GPT_border_area_is_198_l67_6719


namespace NUMINAMATH_GPT_number_is_correct_l67_6792

theorem number_is_correct : (1 / 8) + 0.675 = 0.800 := 
by
  sorry

end NUMINAMATH_GPT_number_is_correct_l67_6792


namespace NUMINAMATH_GPT_triangle_perimeter_l67_6750

variable (y : ℝ)

theorem triangle_perimeter (h₁ : 2 * y > y) (h₂ : y > 0) :
  ∃ (P : ℝ), P = 2 * y + y * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_triangle_perimeter_l67_6750


namespace NUMINAMATH_GPT_jack_additional_sweets_is_correct_l67_6756

/-- Initial number of sweets --/
def initial_sweets : ℕ := 22

/-- Sweets taken by Paul --/
def sweets_taken_by_paul : ℕ := 7

/-- Jack's total sweets taken --/
def jack_total_sweets_taken : ℕ := initial_sweets - sweets_taken_by_paul

/-- Half of initial sweets --/
def half_initial_sweets : ℕ := initial_sweets / 2

/-- Additional sweets taken by Jack --/
def additional_sweets_taken_by_jack : ℕ := jack_total_sweets_taken - half_initial_sweets

theorem jack_additional_sweets_is_correct : additional_sweets_taken_by_jack = 4 := by
  sorry

end NUMINAMATH_GPT_jack_additional_sweets_is_correct_l67_6756


namespace NUMINAMATH_GPT_more_balloons_allan_l67_6755

theorem more_balloons_allan (allan_balloons : ℕ) (jake_initial_balloons : ℕ) (jake_bought_balloons : ℕ) 
  (h1 : allan_balloons = 6) (h2 : jake_initial_balloons = 2) (h3 : jake_bought_balloons = 3) :
  allan_balloons = jake_initial_balloons + jake_bought_balloons + 1 := 
by 
  -- Assuming Jake's total balloons after purchase
  let jake_total_balloons := jake_initial_balloons + jake_bought_balloons
  -- The proof would involve showing that Allan's balloons are one more than Jake's total balloons
  sorry

end NUMINAMATH_GPT_more_balloons_allan_l67_6755


namespace NUMINAMATH_GPT_triangle_inequality_difference_l67_6717

theorem triangle_inequality_difference :
  ∀ (x : ℕ), (x + 8 > 10) → (x + 10 > 8) → (8 + 10 > x) →
    (17 - 3 = 14) :=
by
  intros x hx1 hx2 hx3
  sorry

end NUMINAMATH_GPT_triangle_inequality_difference_l67_6717


namespace NUMINAMATH_GPT_unique_mod_inverse_l67_6722

theorem unique_mod_inverse (a n : ℤ) (coprime : Int.gcd a n = 1) : 
  ∃! b : ℤ, (a * b) % n = 1 % n := 
sorry

end NUMINAMATH_GPT_unique_mod_inverse_l67_6722


namespace NUMINAMATH_GPT_oranges_left_to_be_sold_l67_6744

-- Defining the initial conditions
def seven_dozen_oranges : ℕ := 7 * 12
def reserved_for_friend (total : ℕ) : ℕ := total / 4
def remaining_after_reserve (total reserved : ℕ) : ℕ := total - reserved
def sold_yesterday (remaining : ℕ) : ℕ := 3 * remaining / 7
def remaining_after_sale (remaining sold : ℕ) : ℕ := remaining - sold
def remaining_after_rotten (remaining : ℕ) : ℕ := remaining - 4

-- Statement to prove
theorem oranges_left_to_be_sold (total reserved remaining sold final : ℕ) :
  total = seven_dozen_oranges →
  reserved = reserved_for_friend total →
  remaining = remaining_after_reserve total reserved →
  sold = sold_yesterday remaining →
  final = remaining_after_sale remaining sold - 4 →
  final = 32 :=
by
  sorry

end NUMINAMATH_GPT_oranges_left_to_be_sold_l67_6744


namespace NUMINAMATH_GPT_sally_fries_count_l67_6739

theorem sally_fries_count (sally_initial_fries mark_initial_fries : ℕ) 
  (mark_gave_fraction : ℤ) 
  (h_sally_initial : sally_initial_fries = 14) 
  (h_mark_initial : mark_initial_fries = 36) 
  (h_mark_give : mark_gave_fraction = 1 / 3) :
  sally_initial_fries + (mark_initial_fries * mark_gave_fraction).natAbs = 26 :=
by
  sorry

end NUMINAMATH_GPT_sally_fries_count_l67_6739


namespace NUMINAMATH_GPT_initial_eggs_ben_l67_6784

-- Let's define the conditions from step a):
def eggs_morning := 4
def eggs_afternoon := 3
def eggs_left := 13

-- Define the total eggs Ben ate
def eggs_eaten := eggs_morning + eggs_afternoon

-- Now we define the initial eggs Ben had
def initial_eggs := eggs_left + eggs_eaten

-- The theorem that states the initial number of eggs
theorem initial_eggs_ben : initial_eggs = 20 :=
  by sorry

end NUMINAMATH_GPT_initial_eggs_ben_l67_6784


namespace NUMINAMATH_GPT_c_positive_when_others_negative_l67_6752

variables {a b c d e f : ℤ}

theorem c_positive_when_others_negative (h_ab_cdef_lt_0 : a * b + c * d * e * f < 0)
  (h_a_neg : a < 0) (h_b_neg : b < 0) (h_d_neg : d < 0) (h_e_neg : e < 0) (h_f_neg : f < 0) 
  : c > 0 :=
sorry

end NUMINAMATH_GPT_c_positive_when_others_negative_l67_6752


namespace NUMINAMATH_GPT_equation_of_line_l67_6799

theorem equation_of_line {x y : ℝ} (b : ℝ) (h1 : ∀ x y, (3 * x + 4 * y - 7 = 0) → (y = -3/4 * x))
  (h2 : (1 / 2) * |b| * |(4 / 3) * b| = 24) : 
  ∃ b : ℝ, ∀ x, y = -3/4 * x + b := 
sorry

end NUMINAMATH_GPT_equation_of_line_l67_6799


namespace NUMINAMATH_GPT_graph_is_hyperbola_l67_6785

def graph_equation (x y : ℝ) : Prop := x^2 - 16 * y^2 - 8 * x + 64 = 0

theorem graph_is_hyperbola : ∃ (a b : ℝ), ∀ x y : ℝ, graph_equation x y ↔ (x - a)^2 / 48 - y^2 / 3 = -1 :=
by
  sorry

end NUMINAMATH_GPT_graph_is_hyperbola_l67_6785


namespace NUMINAMATH_GPT_cheapest_salon_option_haily_l67_6721

theorem cheapest_salon_option_haily : 
  let gustran_haircut := 45
  let gustran_facial := 22
  let gustran_nails := 30
  let gustran_foot_spa := 15
  let gustran_massage := 50
  let gustran_total := gustran_haircut + gustran_facial + gustran_nails + gustran_foot_spa + gustran_massage
  let gustran_discount := 0.20
  let gustran_final := gustran_total * (1 - gustran_discount)

  let barbara_nails := 40
  let barbara_haircut := 30
  let barbara_facial := 28
  let barbara_foot_spa := 18
  let barbara_massage := 45
  let barbara_total :=
      barbara_nails + barbara_haircut + (barbara_facial * 0.5) + barbara_foot_spa + (barbara_massage * 0.5)

  let fancy_haircut := 34
  let fancy_facial := 30
  let fancy_nails := 20
  let fancy_foot_spa := 25
  let fancy_massage := 60
  let fancy_total := fancy_haircut + fancy_facial + fancy_nails + fancy_foot_spa + fancy_massage
  let fancy_discount := 15
  let fancy_final := fancy_total - fancy_discount

  let avg_haircut := (gustran_haircut + barbara_haircut + fancy_haircut) / 3
  let avg_facial := (gustran_facial + barbara_facial + fancy_facial) / 3
  let avg_nails := (gustran_nails + barbara_nails + fancy_nails) / 3
  let avg_foot_spa := (gustran_foot_spa + barbara_foot_spa + fancy_foot_spa) / 3
  let avg_massage := (gustran_massage + barbara_massage + fancy_massage) / 3

  let luxury_haircut := avg_haircut * 1.10
  let luxury_facial := avg_facial * 1.10
  let luxury_nails := avg_nails * 1.10
  let luxury_foot_spa := avg_foot_spa * 1.10
  let luxury_massage := avg_massage * 1.10
  let luxury_total := luxury_haircut + luxury_facial + luxury_nails + luxury_foot_spa + luxury_massage
  let luxury_discount := 20
  let luxury_final := luxury_total - luxury_discount

  gustran_final > barbara_total ∧ barbara_total < fancy_final ∧ barbara_total < luxury_final := 
by 
  sorry

end NUMINAMATH_GPT_cheapest_salon_option_haily_l67_6721


namespace NUMINAMATH_GPT_find_x7_l67_6733

-- Definitions for the conditions
def seq (x : ℕ → ℕ) : Prop :=
  (x 6 = 144) ∧ ∀ n, (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) → (x (n + 3) = x (n + 2) * (x (n + 1) + x n))

-- Theorem statement to prove x_7 = 3456
theorem find_x7 (x : ℕ → ℕ) (h : seq x) : x 7 = 3456 := sorry

end NUMINAMATH_GPT_find_x7_l67_6733


namespace NUMINAMATH_GPT_line_through_diameter_l67_6759

theorem line_through_diameter (P : ℝ × ℝ) (hP : P = (2, 1)) (h_circle : ∀ x y : ℝ, (x - 1)^2 + y^2 = 4) :
  ∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = -1 :=
by
  exists 1, -1, -1
  sorry

end NUMINAMATH_GPT_line_through_diameter_l67_6759


namespace NUMINAMATH_GPT_fifth_number_in_21st_row_l67_6766

theorem fifth_number_in_21st_row : 
  let nth_odd_number (n : ℕ) := 2 * n - 1 
  let sum_first_n_rows (n : ℕ) := n * (n + (n - 1))
  nth_odd_number 405 = 809 := 
by
  sorry

end NUMINAMATH_GPT_fifth_number_in_21st_row_l67_6766


namespace NUMINAMATH_GPT_find_a_parallel_find_a_perpendicular_l67_6735

open Real

def line_parallel (p1 p2 q1 q2 : (ℝ × ℝ)) : Prop :=
  let k1 := (q2.2 - q1.2) / (q2.1 - q1.1)
  let k2 := (p2.2 - p1.2) / (p2.1 - p1.1)
  k1 = k2

def line_perpendicular (p1 p2 q1 q2 : (ℝ × ℝ)) : Prop :=
  let k1 := (q2.2 - q1.2) / (q2.1 - q1.1)
  let k2 := (p2.2 - p1.2) / (p2.1 - p1.1)
  k1 * k2 = -1

theorem find_a_parallel (a : ℝ) :
  line_parallel (3, a) (a-1, 2) (1, 2) (-2, a+2) ↔ a = 1 ∨ a = 6 :=
by sorry

theorem find_a_perpendicular (a : ℝ) :
  line_perpendicular (3, a) (a-1, 2) (1, 2) (-2, a+2) ↔ a = 3 ∨ a = -4 :=
by sorry

end NUMINAMATH_GPT_find_a_parallel_find_a_perpendicular_l67_6735


namespace NUMINAMATH_GPT_find_k_l67_6778

theorem find_k (a b c k : ℤ)
  (g : ℤ → ℤ)
  (h1 : ∀ x, g x = a * x^2 + b * x + c)
  (h2 : g 2 = 0)
  (h3 : 60 < g 6 ∧ g 6 < 70)
  (h4 : 90 < g 9 ∧ g 9 < 100)
  (h5 : 10000 * k < g 50 ∧ g 50 < 10000 * (k + 1)) :
  k = 0 :=
sorry

end NUMINAMATH_GPT_find_k_l67_6778


namespace NUMINAMATH_GPT_S_rational_iff_divides_l67_6741

-- Definition of "divides" for positive integers
def divides (m k : ℕ) : Prop := ∃ j : ℕ, k = m * j

-- Definition of the series S(m, k)
noncomputable def S (m k : ℕ) : ℝ := 
  ∑' n, 1 / (n * (m * n + k))

-- Proof statement
theorem S_rational_iff_divides (m k : ℕ) (hm : 0 < m) (hk : 0 < k) : 
  (∃ r : ℚ, S m k = r) ↔ divides m k :=
sorry

end NUMINAMATH_GPT_S_rational_iff_divides_l67_6741


namespace NUMINAMATH_GPT_find_a_l67_6788

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a (a : ℝ) (h : deriv (f a) (-1) = 4) : a = 10 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_a_l67_6788


namespace NUMINAMATH_GPT_total_chestnuts_weight_l67_6761

def eunsoo_kg := 2
def eunsoo_g := 600
def mingi_g := 3700

theorem total_chestnuts_weight :
  (eunsoo_kg * 1000 + eunsoo_g + mingi_g) = 6300 :=
by
  sorry

end NUMINAMATH_GPT_total_chestnuts_weight_l67_6761


namespace NUMINAMATH_GPT_cannot_determine_if_counterfeit_coin_is_lighter_or_heavier_l67_6731

/-- 
Vasiliy has 2019 coins, one of which is counterfeit (differing in weight). 
Using balance scales without weights and immediately paying out identified genuine coins, 
it is impossible to determine whether the counterfeit coin is lighter or heavier.
-/
theorem cannot_determine_if_counterfeit_coin_is_lighter_or_heavier 
  (num_coins : ℕ)
  (num_counterfeit : ℕ)
  (balance_scale : Bool → Bool → Bool)
  (immediate_payment : Bool → Bool) :
  num_coins = 2019 →
  num_counterfeit = 1 →
  (∀ coins_w1 coins_w2, balance_scale coins_w1 coins_w2 = (coins_w1 = coins_w2)) →
  (∀ coin_p coin_q, (immediate_payment coin_p = true) → ¬ coin_p = coin_q) →
  ¬ ∃ (is_lighter_or_heavier : Bool), true :=
by
  intro h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_cannot_determine_if_counterfeit_coin_is_lighter_or_heavier_l67_6731


namespace NUMINAMATH_GPT_range_of_a_l67_6720

def line_intersects_circle (a : ℝ) : Prop :=
  let distance_from_center_to_line := |1 - a| / Real.sqrt 2
  distance_from_center_to_line ≤ Real.sqrt 2

theorem range_of_a :
  {a : ℝ | line_intersects_circle a} = {a : ℝ | -1 ≤ a ∧ a ≤ 3} :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l67_6720


namespace NUMINAMATH_GPT_domain_h_l67_6708

noncomputable def h (x : ℝ) : ℝ := (3 * x - 1) / Real.sqrt (x - 5)

theorem domain_h (x : ℝ) : h x = (3 * x - 1) / Real.sqrt (x - 5) → (x > 5) :=
by
  intro hx
  have hx_nonneg : x - 5 >= 0 := sorry
  have sqrt_nonzero : Real.sqrt (x - 5) ≠ 0 := sorry
  sorry

end NUMINAMATH_GPT_domain_h_l67_6708


namespace NUMINAMATH_GPT_quadratic_factorization_value_of_a_l67_6787

theorem quadratic_factorization_value_of_a (a : ℝ) :
  (∀ x : ℝ, 2 * x^2 - 8 * x + a = 0 ↔ 2 * (x - 2)^2 = 4) → a = 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_factorization_value_of_a_l67_6787


namespace NUMINAMATH_GPT_largest_value_of_n_l67_6730

noncomputable def largest_n_under_200000 : ℕ :=
  if h : 199999 < 200000 ∧ (8 * (199999 - 3)^5 - 2 * 199999^2 + 18 * 199999 - 36) % 7 = 0 then 199999 else 0

theorem largest_value_of_n (n : ℕ) :
  n < 200000 → (8 * (n - 3)^5 - 2 * n^2 + 18 * n - 36) % 7 = 0 → n = 199999 :=
by sorry

end NUMINAMATH_GPT_largest_value_of_n_l67_6730


namespace NUMINAMATH_GPT_greatest_whole_number_difference_l67_6782

theorem greatest_whole_number_difference (x y : ℤ) (hx1 : 7 < x) (hx2 : x < 9) (hy1 : 9 < y) (hy2 : y < 15) : y - x = 6 :=
by
  sorry

end NUMINAMATH_GPT_greatest_whole_number_difference_l67_6782


namespace NUMINAMATH_GPT_boys_in_class_l67_6724

theorem boys_in_class (g b : ℕ) 
  (h_ratio : 4 * g = 3 * b) (h_total : g + b = 28) : b = 16 :=
by
  sorry

end NUMINAMATH_GPT_boys_in_class_l67_6724


namespace NUMINAMATH_GPT_sin_ratio_in_triangle_l67_6779

theorem sin_ratio_in_triangle
  {A B C : ℝ} {a b c : ℝ}
  (h : (b + c) / (c + a) = 4 / 5 ∧ (c + a) / (a + b) = 5 / 6) :
  (Real.sin A + Real.sin C) / Real.sin B = 2 :=
sorry

end NUMINAMATH_GPT_sin_ratio_in_triangle_l67_6779


namespace NUMINAMATH_GPT_line_equation_l67_6715

theorem line_equation (x y : ℝ) : 
  (∃ (m c : ℝ), m = 3 ∧ c = 4 ∧ y = m * x + c) ↔ 3 * x - y + 4 = 0 := by
  sorry

end NUMINAMATH_GPT_line_equation_l67_6715


namespace NUMINAMATH_GPT_find_number_x_l67_6707

theorem find_number_x (x : ℝ) (h : 2500 - x / 20.04 = 2450) : x = 1002 :=
by
  -- Proof can be written here, but skipped by using sorry
  sorry

end NUMINAMATH_GPT_find_number_x_l67_6707


namespace NUMINAMATH_GPT_shaded_area_z_shape_l67_6757

theorem shaded_area_z_shape (L W s1 s2 : ℕ) (hL : L = 6) (hW : W = 4) (hs1 : s1 = 2) (hs2 : s2 = 1) :
  (L * W - (s1 * s1 + s2 * s2)) = 19 := by
  sorry

end NUMINAMATH_GPT_shaded_area_z_shape_l67_6757


namespace NUMINAMATH_GPT_sin_pow_cos_pow_sum_l67_6732

namespace ProofProblem

-- Define the condition
def trig_condition (x : ℝ) : Prop :=
  3 * (Real.sin x)^3 + (Real.cos x)^3 = 3

-- State the theorem
theorem sin_pow_cos_pow_sum (x : ℝ) (h : trig_condition x) : Real.sin x ^ 2018 + Real.cos x ^ 2018 = 1 :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_sin_pow_cos_pow_sum_l67_6732


namespace NUMINAMATH_GPT_restoration_of_axes_l67_6765

theorem restoration_of_axes (parabola : ℝ → ℝ) (h : ∀ x, parabola x = x^2) : 
  ∃ (origin : ℝ × ℝ) (x_axis y_axis : ℝ × ℝ → Prop), 
    (∀ x, x_axis (x, 0)) ∧ 
    (∀ y, y_axis (0, y)) ∧ 
    origin = (0, 0) := 
sorry

end NUMINAMATH_GPT_restoration_of_axes_l67_6765


namespace NUMINAMATH_GPT_apples_in_each_bag_l67_6709

variable (x : ℕ)
variable (total_children : ℕ)
variable (eaten_apples : ℕ)
variable (sold_apples : ℕ)
variable (remaining_apples : ℕ)

theorem apples_in_each_bag
  (h1 : total_children = 5)
  (h2 : eaten_apples = 2 * 4)
  (h3 : sold_apples = 7)
  (h4 : remaining_apples = 60)
  (h5 : total_children * x - eaten_apples - sold_apples = remaining_apples) :
  x = 15 :=
by
  sorry

end NUMINAMATH_GPT_apples_in_each_bag_l67_6709


namespace NUMINAMATH_GPT_initial_concentration_is_40_l67_6747

noncomputable def initial_concentration_fraction : ℝ := 1 / 3
noncomputable def replaced_solution_concentration : ℝ := 25
noncomputable def resulting_concentration : ℝ := 35
noncomputable def initial_concentration := 40

theorem initial_concentration_is_40 (C : ℝ) (h1 : C = (3 / 2) * (resulting_concentration - (initial_concentration_fraction * replaced_solution_concentration))) :
  C = initial_concentration :=
by sorry

end NUMINAMATH_GPT_initial_concentration_is_40_l67_6747


namespace NUMINAMATH_GPT_puppy_sleep_duration_l67_6745

-- Definitions based on the given conditions
def connor_sleep_hours : ℕ := 6
def luke_sleep_hours : ℕ := connor_sleep_hours + 2
def puppy_sleep_hours : ℕ := 2 * luke_sleep_hours

-- Theorem stating the puppy's sleep duration
theorem puppy_sleep_duration : puppy_sleep_hours = 16 :=
by
  -- ( Proof goes here )
  sorry

end NUMINAMATH_GPT_puppy_sleep_duration_l67_6745


namespace NUMINAMATH_GPT_silvia_order_total_cost_l67_6768

theorem silvia_order_total_cost :
  let quiche_price : ℝ := 15
  let croissant_price : ℝ := 3
  let biscuit_price : ℝ := 2
  let quiche_count : ℝ := 2
  let croissant_count : ℝ := 6
  let biscuit_count : ℝ := 6
  let discount_rate : ℝ := 0.10
  let pre_discount_total : ℝ := (quiche_price * quiche_count) + (croissant_price * croissant_count) + (biscuit_price * biscuit_count)
  let discount_amount : ℝ := pre_discount_total * discount_rate
  let post_discount_total : ℝ := pre_discount_total - discount_amount
  pre_discount_total > 50 → post_discount_total = 54 :=
by
  sorry

end NUMINAMATH_GPT_silvia_order_total_cost_l67_6768


namespace NUMINAMATH_GPT_arithmetic_sequence_30th_term_l67_6706

-- Defining the initial term and the common difference of the arithmetic sequence
def a : ℕ := 3
def d : ℕ := 4

-- Defining the general formula for the n-th term of the arithmetic sequence
def a_n (n : ℕ) : ℕ := a + (n - 1) * d

-- Theorem stating that the 30th term of the given sequence is 119
theorem arithmetic_sequence_30th_term : a_n 30 = 119 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_30th_term_l67_6706


namespace NUMINAMATH_GPT_algebraic_expression_value_l67_6712

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x - y = -2) 
  (h2 : 2 * x + y = -1) : 
  (x - y)^2 - (x - 2 * y) * (x + 2 * y) = 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_algebraic_expression_value_l67_6712


namespace NUMINAMATH_GPT_matrix_addition_l67_6777

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![1, 0], ![-1, 2]]
def B : Matrix (Fin 2) (Fin 2) ℤ := ![![2, 4], ![1, -3]]
def C : Matrix (Fin 2) (Fin 2) ℤ := ![![3, 4], ![0, -1]]

theorem matrix_addition : A + B = C := by
    sorry

end NUMINAMATH_GPT_matrix_addition_l67_6777


namespace NUMINAMATH_GPT_least_number_to_add_l67_6780

-- Definition of LCM for given primes
def lcm_of_primes : ℕ := 5 * 7 * 11 * 13 * 17 * 19

theorem least_number_to_add (n : ℕ) : 
  (5432 + n) % 5 = 0 ∧ 
  (5432 + n) % 7 = 0 ∧ 
  (5432 + n) % 11 = 0 ∧ 
  (5432 + n) % 13 = 0 ∧ 
  (5432 + n) % 17 = 0 ∧ 
  (5432 + n) % 19 = 0 ↔ 
  n = 1611183 :=
by sorry

end NUMINAMATH_GPT_least_number_to_add_l67_6780


namespace NUMINAMATH_GPT_triangle_right_angled_solve_system_quadratic_roots_real_l67_6748

-- Problem 1
theorem triangle_right_angled (a b c : ℝ) (h : a^2 + b^2 + c^2 - 6 * a - 8 * b - 10 * c + 50 = 0) :
  (a = 3) ∧ (b = 4) ∧ (c = 5) ∧ (a^2 + b^2 = c^2) :=
sorry

-- Problem 2
theorem solve_system (x y : ℝ) (h1 : 3 * x + 4 * y = 30) (h2 : 5 * x + 3 * y = 28) :
  (x = 2) ∧ (y = 6) :=
sorry

-- Problem 3
theorem quadratic_roots_real (m : ℝ) :
  (∃ x : ℝ, ∃ y : ℝ, 3 * x^2 + 4 * x + m = 0 ∧ 3 * y^2 + 4 * y + m = 0) ↔ (m ≤ 4 / 3) :=
sorry

end NUMINAMATH_GPT_triangle_right_angled_solve_system_quadratic_roots_real_l67_6748


namespace NUMINAMATH_GPT_luke_total_points_l67_6704

theorem luke_total_points (rounds : ℕ) (points_per_round : ℕ) (total_points : ℕ) 
  (h1 : rounds = 177) (h2 : points_per_round = 46) : 
  total_points = 8142 := by
  have h : total_points = rounds * points_per_round := by sorry
  rw [h1, h2] at h
  exact h

end NUMINAMATH_GPT_luke_total_points_l67_6704


namespace NUMINAMATH_GPT_ascending_order_l67_6701

theorem ascending_order (a b : ℝ) (ha : a < 0) (hb1 : -1 < b) (hb2 : b < 0) : a < a * b^2 ∧ a * b^2 < a * b :=
by
  sorry

end NUMINAMATH_GPT_ascending_order_l67_6701


namespace NUMINAMATH_GPT_unique_value_of_W_l67_6725

theorem unique_value_of_W (T O W F U R : ℕ) (h1 : T = 8) (h2 : O % 2 = 0) (h3 : ∀ x y, x ≠ y → x = O → y = T → x ≠ O) :
  (T + T) * 10^2 + (W + W) * 10 + (O + O) = F * 10^3 + O * 10^2 + U * 10 + R → W = 3 :=
by
  sorry

end NUMINAMATH_GPT_unique_value_of_W_l67_6725


namespace NUMINAMATH_GPT_four_times_angle_triangle_l67_6740

theorem four_times_angle_triangle (A B C : ℕ) 
  (h1 : A + B + C = 180) 
  (h2 : A = 40)
  (h3 : (A = 4 * C) ∨ (B = 4 * C) ∨ (C = 4 * A)) : 
  (B = 130 ∧ C = 10) ∨ (B = 112 ∧ C = 28) :=
by
  sorry

end NUMINAMATH_GPT_four_times_angle_triangle_l67_6740


namespace NUMINAMATH_GPT_perpendicular_lines_solve_a_l67_6702

theorem perpendicular_lines_solve_a (a : ℝ) :
  (3 * a + 2) * (5 * a - 2) + (1 - 4 * a) * (a + 4) = 0 → a = 0 ∨ a = 12 / 11 :=
by 
  sorry

end NUMINAMATH_GPT_perpendicular_lines_solve_a_l67_6702


namespace NUMINAMATH_GPT_pet_shop_ways_l67_6774

theorem pet_shop_ways (puppies : ℕ) (kittens : ℕ) (turtles : ℕ)
  (h_puppies : puppies = 10) (h_kittens : kittens = 8) (h_turtles : turtles = 5) : 
  (puppies * kittens * turtles = 400) :=
by
  sorry

end NUMINAMATH_GPT_pet_shop_ways_l67_6774


namespace NUMINAMATH_GPT_arithmetic_sequence_transformation_l67_6749

theorem arithmetic_sequence_transformation (a : ℕ → ℝ) (d c : ℝ) (h : ∀ n, a (n + 1) = a n + d) (hc : c ≠ 0) :
  ∀ n, (c * a (n + 1)) - (c * a n) = c * d := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_transformation_l67_6749


namespace NUMINAMATH_GPT_proof_supplies_proof_transportation_cost_proof_min_cost_condition_l67_6754

open Real

noncomputable def supplies_needed (a b : ℕ) := a = 200 ∧ b = 300

noncomputable def transportation_cost (x : ℝ) := 60 ≤ x ∧ x ≤ 260 ∧ ∀ w : ℝ, w = 10 * x + 10200

noncomputable def min_cost_condition (m x : ℝ) := 
  (0 < m ∧ m ≤ 8) ∧ (∀ w : ℝ, (10 - m) * x + 10200 ≥ 10320)

theorem proof_supplies : ∃ a b : ℕ, supplies_needed a b := 
by
  use 200, 300
  sorry

theorem proof_transportation_cost : ∃ x : ℝ, transportation_cost x := 
by
  use 60
  sorry

theorem proof_min_cost_condition : ∃ m x : ℝ, min_cost_condition m x := 
by
  use 8, 60
  sorry

end NUMINAMATH_GPT_proof_supplies_proof_transportation_cost_proof_min_cost_condition_l67_6754


namespace NUMINAMATH_GPT_john_total_replacement_cost_l67_6718

def cost_to_replace_all_doors
  (num_bedroom_doors : ℕ)
  (num_outside_doors : ℕ)
  (cost_outside_door : ℕ)
  (cost_bedroom_door : ℕ) : ℕ :=
  let total_cost_outside_doors := num_outside_doors * cost_outside_door
  let total_cost_bedroom_doors := num_bedroom_doors * cost_bedroom_door
  total_cost_outside_doors + total_cost_bedroom_doors

theorem john_total_replacement_cost :
  let num_bedroom_doors := 3
  let num_outside_doors := 2
  let cost_outside_door := 20
  let cost_bedroom_door := cost_outside_door / 2
  cost_to_replace_all_doors num_bedroom_doors num_outside_doors cost_outside_door cost_bedroom_door = 70 := by
  sorry

end NUMINAMATH_GPT_john_total_replacement_cost_l67_6718


namespace NUMINAMATH_GPT_initial_birds_count_l67_6753

theorem initial_birds_count (B : ℕ) (h1 : 6 = B + 3 + 1) : B = 2 :=
by
  -- Placeholder for the proof, we are not required to provide it here.
  sorry

end NUMINAMATH_GPT_initial_birds_count_l67_6753


namespace NUMINAMATH_GPT_largest_integer_x_l67_6798

theorem largest_integer_x (x : ℕ) : (1 / 4 : ℚ) + (x / 8 : ℚ) < 1 ↔ x <= 5 := sorry

end NUMINAMATH_GPT_largest_integer_x_l67_6798


namespace NUMINAMATH_GPT_monica_studied_32_67_hours_l67_6767

noncomputable def monica_total_study_time : ℚ :=
  let monday := 1
  let tuesday := 2 * monday
  let wednesday := 2
  let thursday := 3 * wednesday
  let friday := thursday / 2
  let total_weekday := monday + tuesday + wednesday + thursday + friday
  let saturday := total_weekday
  let sunday := saturday / 3
  total_weekday + saturday + sunday

theorem monica_studied_32_67_hours :
  monica_total_study_time = 32.67 := by
  sorry

end NUMINAMATH_GPT_monica_studied_32_67_hours_l67_6767


namespace NUMINAMATH_GPT_sum_tens_units_11_pow_2010_l67_6796

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

def units_digit (n : ℕ) : ℕ :=
  n % 10

def sum_tens_units_digits (n : ℕ) : ℕ :=
  tens_digit n + units_digit n

theorem sum_tens_units_11_pow_2010 :
  sum_tens_units_digits (11 ^ 2010) = 1 :=
sorry

end NUMINAMATH_GPT_sum_tens_units_11_pow_2010_l67_6796


namespace NUMINAMATH_GPT_carnival_tickets_l67_6789

theorem carnival_tickets (total_tickets friends : ℕ) (equal_share : ℕ)
  (h1 : friends = 6)
  (h2 : total_tickets = 234)
  (h3 : total_tickets % friends = 0)
  (h4 : equal_share = total_tickets / friends) : 
  equal_share = 39 := 
by
  sorry

end NUMINAMATH_GPT_carnival_tickets_l67_6789


namespace NUMINAMATH_GPT_jenny_house_value_l67_6746

/-- Jenny's property tax rate is 2% -/
def property_tax_rate : ℝ := 0.02

/-- Her house's value increases by 25% due to the new high-speed rail project -/
noncomputable def house_value_increase_rate : ℝ := 0.25

/-- Jenny can afford to spend $15,000/year on property tax -/
def max_affordable_tax : ℝ := 15000

/-- Jenny can make improvements worth $250,000 to her house -/
def improvement_value : ℝ := 250000

/-- Current worth of Jenny's house -/
noncomputable def current_house_worth : ℝ := 500000

theorem jenny_house_value :
  property_tax_rate * (current_house_worth + improvement_value) = max_affordable_tax :=
by
  sorry

end NUMINAMATH_GPT_jenny_house_value_l67_6746


namespace NUMINAMATH_GPT_smallest_possible_sum_l67_6791

theorem smallest_possible_sum (A B C D : ℤ) 
  (h1 : A + B = 2 * C)
  (h2 : B * D = C * C)
  (h3 : 3 * C = 7 * B)
  (h4 : 0 < A ∧ 0 < B ∧ 0 < C ∧ 0 < D) : 
  A + B + C + D = 76 :=
sorry

end NUMINAMATH_GPT_smallest_possible_sum_l67_6791


namespace NUMINAMATH_GPT_find_range_of_m_l67_6729

def has_two_distinct_negative_real_roots (m : ℝ) : Prop := 
  let Δ := m^2 - 4
  Δ > 0 ∧ -m > 0

def inequality_holds_for_all_real (m : ℝ) : Prop :=
  let Δ := (4 * (m - 2))^2 - 16
  Δ < 0

def problem_statement (m : ℝ) : Prop :=
  (has_two_distinct_negative_real_roots m ∨ inequality_holds_for_all_real m) ∧ 
  ¬(has_two_distinct_negative_real_roots m ∧ inequality_holds_for_all_real m)

theorem find_range_of_m (m : ℝ) : problem_statement m ↔ ((1 < m ∧ m ≤ 2) ∨ (3 ≤ m)) :=
by
  sorry

end NUMINAMATH_GPT_find_range_of_m_l67_6729


namespace NUMINAMATH_GPT_line_is_tangent_to_circle_l67_6771

theorem line_is_tangent_to_circle
  (θ : Real)
  (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop)
  (h_l : ∀ x y, l x y ↔ x * Real.sin θ + 2 * y * Real.cos θ = 1)
  (h_C : ∀ x y, C x y ↔ x^2 + y^2 = 1) :
  (∀ x y, l x y ↔ x = 1 ∨ x = -1) ↔
  (∃ x y, C x y ∧ ∀ x y, l x y → Real.sqrt ((x * Real.sin θ + 2 * y * Real.cos θ - 1)^2 / (Real.sin θ^2 + 4 * Real.cos θ^2)) = 1) :=
sorry

end NUMINAMATH_GPT_line_is_tangent_to_circle_l67_6771


namespace NUMINAMATH_GPT_f_pos_for_all_x_g_le_ax_plus_1_for_a_eq_1_l67_6775

noncomputable def f (x : ℝ) : ℝ := Real.exp x - (x + 1)^2 / 2
noncomputable def g (x : ℝ) : ℝ := 2 * Real.log (x + 1) + Real.exp (-x)

theorem f_pos_for_all_x (x : ℝ) (hx : x > -1) : f x > 0 := by
  sorry

theorem g_le_ax_plus_1_for_a_eq_1 (a : ℝ) (ha : a > 0) : (∀ x : ℝ, -1 < x → g x ≤ a * x + 1) ↔ a = 1 := by
  sorry

end NUMINAMATH_GPT_f_pos_for_all_x_g_le_ax_plus_1_for_a_eq_1_l67_6775


namespace NUMINAMATH_GPT_hockey_games_in_season_l67_6714

-- Define the conditions
def games_per_month : Nat := 13
def season_months : Nat := 14

-- Define the total number of hockey games in the season
def total_games_in_season (games_per_month : Nat) (season_months : Nat) : Nat :=
  games_per_month * season_months

-- Define the theorem to prove
theorem hockey_games_in_season :
  total_games_in_season games_per_month season_months = 182 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_hockey_games_in_season_l67_6714


namespace NUMINAMATH_GPT_combination_identity_l67_6723

-- Lean statement defining the proof problem
theorem combination_identity : Nat.choose 12 5 + Nat.choose 12 6 = Nat.choose 13 6 :=
  sorry

end NUMINAMATH_GPT_combination_identity_l67_6723


namespace NUMINAMATH_GPT_similar_iff_condition_l67_6742

-- Define the similarity of triangles and the necessary conditions.
variables {α : Type*} [LinearOrderedField α]
variables (a b c a' b' c' : α)

-- Statement of the problem in Lean 4
theorem similar_iff_condition : 
  (∃ z w : α, a' = a * z + w ∧ b' = b * z + w ∧ c' = c * z + w) ↔ 
  (a' * (b - c) + b' * (c - a) + c' * (a - b) = 0) :=
sorry

end NUMINAMATH_GPT_similar_iff_condition_l67_6742


namespace NUMINAMATH_GPT_inradius_of_right_triangle_l67_6797

theorem inradius_of_right_triangle (a b c r : ℝ) (h : a^2 + b^2 = c^2) :
  r = (1/2) * (a + b - c) :=
sorry

end NUMINAMATH_GPT_inradius_of_right_triangle_l67_6797


namespace NUMINAMATH_GPT_percentage_A_to_B_l67_6743

variable (A B : ℕ)
variable (total : ℕ := 570)
variable (B_amount : ℕ := 228)

theorem percentage_A_to_B :
  (A + B = total) →
  B = B_amount →
  (A = total - B_amount) →
  ((A / B_amount : ℚ) * 100 = 150) :=
sorry

end NUMINAMATH_GPT_percentage_A_to_B_l67_6743


namespace NUMINAMATH_GPT_cows_now_l67_6795

-- Defining all conditions
def initial_cows : ℕ := 39
def cows_died : ℕ := 25
def cows_sold : ℕ := 6
def cows_increase : ℕ := 24
def cows_bought : ℕ := 43
def cows_gift : ℕ := 8

-- Lean statement for the equivalent proof problem
theorem cows_now :
  let cows_left := initial_cows - cows_died
  let cows_after_selling := cows_left - cows_sold
  let cows_this_year_increased := cows_after_selling + cows_increase
  let cows_with_purchase := cows_this_year_increased + cows_bought
  let total_cows := cows_with_purchase + cows_gift
  total_cows = 83 :=
by
  sorry

end NUMINAMATH_GPT_cows_now_l67_6795


namespace NUMINAMATH_GPT_sin_double_angle_of_tangent_l67_6713

theorem sin_double_angle_of_tangent (α : ℝ) (h : Real.tan (π + α) = 2) : Real.sin (2 * α) = 4 / 5 := by
  sorry

end NUMINAMATH_GPT_sin_double_angle_of_tangent_l67_6713


namespace NUMINAMATH_GPT_relationship_p_q_no_linear_term_l67_6764

theorem relationship_p_q_no_linear_term (p q : ℝ) :
  (∀ x : ℝ, (x^2 - p * x + q) * (x - 3) = x^3 + (-p - 3) * x^2 + (3 * p + q) * x - 3 * q) 
  → (3 * p + q = 0) → (q + 3 * p = 0) :=
by
  intro h_expansion coeff_zero
  sorry

end NUMINAMATH_GPT_relationship_p_q_no_linear_term_l67_6764


namespace NUMINAMATH_GPT_skating_probability_given_skiing_l67_6781

theorem skating_probability_given_skiing (P_A P_B P_A_or_B : ℝ)
    (h1 : P_A = 0.6) (h2 : P_B = 0.5) (h3 : P_A_or_B = 0.7) : 
    (P_A_or_B = P_A + P_B - P_A * P_B) → 
    ((P_A * P_B) / P_B = 0.8) := 
    by
        intros
        sorry

end NUMINAMATH_GPT_skating_probability_given_skiing_l67_6781


namespace NUMINAMATH_GPT_problem1_problem2_l67_6776

theorem problem1 : ((- (5 : ℚ) / 6) + 2 / 3) / (- (7 / 12)) * (7 / 2) = 1 := 
sorry

theorem problem2 : ((1 - 1 / 6) * (-3) - (- (11 / 6)) / (- (22 / 3))) = - (11 / 4) := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l67_6776


namespace NUMINAMATH_GPT_evaluate_expression_l67_6783

theorem evaluate_expression : 5 - 7 * (8 - 3^2) * 4 = 33 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l67_6783


namespace NUMINAMATH_GPT_outer_boundary_diameter_l67_6736

theorem outer_boundary_diameter (fountain_diameter garden_width path_width : ℝ) 
(h1 : fountain_diameter = 12) 
(h2 : garden_width = 10) 
(h3 : path_width = 6) : 
2 * ((fountain_diameter / 2) + garden_width + path_width) = 44 :=
by
  -- Sorry, proof not needed for this statement
  sorry

end NUMINAMATH_GPT_outer_boundary_diameter_l67_6736


namespace NUMINAMATH_GPT_teacher_engineer_ratio_l67_6760

-- Define the context with the given conditions
variable (t e : ℕ)

-- Conditions
def avg_age (t e : ℕ) : Prop := (40 * t + 55 * e) / (t + e) = 45

-- The statement to be proved
theorem teacher_engineer_ratio
  (h : avg_age t e) :
  t / e = 2 := sorry

end NUMINAMATH_GPT_teacher_engineer_ratio_l67_6760


namespace NUMINAMATH_GPT_sphere_radius_eq_three_of_volume_eq_surface_area_l67_6773

theorem sphere_radius_eq_three_of_volume_eq_surface_area
  (r : ℝ) 
  (h1 : (4 / 3) * Real.pi * r^3 = 4 * Real.pi * r^2) : 
  r = 3 :=
sorry

end NUMINAMATH_GPT_sphere_radius_eq_three_of_volume_eq_surface_area_l67_6773


namespace NUMINAMATH_GPT_abs_x_plus_abs_y_eq_one_area_l67_6703

theorem abs_x_plus_abs_y_eq_one_area : 
  (∃ (A : ℝ), ∀ (x y : ℝ), |x| + |y| = 1 → A = 2) :=
sorry

end NUMINAMATH_GPT_abs_x_plus_abs_y_eq_one_area_l67_6703


namespace NUMINAMATH_GPT_total_seconds_eq_250200_l67_6770

def bianca_hours : ℝ := 12.5
def celeste_hours : ℝ := 2 * bianca_hours
def mcclain_hours : ℝ := celeste_hours - 8.5
def omar_hours : ℝ := bianca_hours + 3

def total_hours : ℝ := bianca_hours + celeste_hours + mcclain_hours + omar_hours
def hour_to_seconds : ℝ := 3600
def total_seconds : ℝ := total_hours * hour_to_seconds

theorem total_seconds_eq_250200 : total_seconds = 250200 := by
  sorry

end NUMINAMATH_GPT_total_seconds_eq_250200_l67_6770


namespace NUMINAMATH_GPT_find_f1_plus_g1_l67_6711

variable (f g : ℝ → ℝ)

-- Conditions
def even_function (h : ℝ → ℝ) := ∀ x : ℝ, h x = h (-x)
def odd_function (h : ℝ → ℝ) := ∀ x : ℝ, h x = -h (-x)
def function_relation := ∀ x : ℝ, f x - g x = x^3 + x^2 + 1

-- Mathematically equivalent proof problem
theorem find_f1_plus_g1
  (hf_even : even_function f)
  (hg_odd : odd_function g)
  (h_relation : function_relation f g) :
  f 1 + g 1 = 1 := by
  sorry

end NUMINAMATH_GPT_find_f1_plus_g1_l67_6711


namespace NUMINAMATH_GPT_total_dots_not_visible_eq_54_l67_6790

theorem total_dots_not_visible_eq_54 :
  let die_sum := 21
  let num_dice := 4
  let total_sum := num_dice * die_sum
  let visible_sum := 1 + 2 + 3 + 4 + 4 + 5 + 5 + 6
  total_sum - visible_sum = 54 :=
by
  let die_sum := 21
  let num_dice := 4
  let total_sum := num_dice * die_sum
  let visible_sum := 1 + 2 + 3 + 4 + 4 + 5 + 5 + 6
  show total_sum - visible_sum = 54
  sorry

end NUMINAMATH_GPT_total_dots_not_visible_eq_54_l67_6790


namespace NUMINAMATH_GPT_remainder_of_8_pow_6_plus_1_mod_7_l67_6727

theorem remainder_of_8_pow_6_plus_1_mod_7 :
  (8^6 + 1) % 7 = 2 := by
  sorry

end NUMINAMATH_GPT_remainder_of_8_pow_6_plus_1_mod_7_l67_6727


namespace NUMINAMATH_GPT_original_number_of_laborers_l67_6705

theorem original_number_of_laborers 
(L : ℕ) (h1 : L * 15 = (L - 5) * 20) : L = 15 :=
sorry

end NUMINAMATH_GPT_original_number_of_laborers_l67_6705


namespace NUMINAMATH_GPT_sum_of_missing_digits_l67_6728

-- Define the problem's conditions
def add_digits (a b c d e f g h : ℕ) := 
a + b = 18 ∧ b + c + d = 21

-- Prove the sum of the missing digits equals 7
theorem sum_of_missing_digits (a b c d e f g h : ℕ) (h1 : add_digits a b c d e f g h) : a + c = 7 := 
sorry

end NUMINAMATH_GPT_sum_of_missing_digits_l67_6728


namespace NUMINAMATH_GPT_find_annual_interest_rate_l67_6769

theorem find_annual_interest_rate (P0 P1 P2 : ℝ) (r1 r : ℝ) :
  P0 = 12000 →
  r1 = 10 →
  P1 = P0 * (1 + (r1 / 100) / 2) →
  P1 = 12600 →
  P2 = 13260 →
  P1 * (1 + (r / 200)) = P2 →
  r = 10.476 :=
by
  intros hP0 hr1 hP1 hP1val hP2 hP1P2
  sorry

end NUMINAMATH_GPT_find_annual_interest_rate_l67_6769


namespace NUMINAMATH_GPT_sum_of_real_solutions_l67_6762

theorem sum_of_real_solutions:
  (∃ (s : ℝ), ∀ x : ℝ, 
    (x - 3) / (x^2 + 6 * x + 2) = (x - 6) / (x^2 - 12 * x) → 
    s = 106 / 9) :=
  sorry

end NUMINAMATH_GPT_sum_of_real_solutions_l67_6762


namespace NUMINAMATH_GPT_solution_set_f_gt_5_range_m_f_ge_abs_2m1_l67_6726

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) + abs (x + 3)

theorem solution_set_f_gt_5 :
  {x : ℝ | f x > 5} = {x : ℝ | x < -1} ∪ {x : ℝ | x > 1} :=
by sorry

theorem range_m_f_ge_abs_2m1 :
  (∀ x : ℝ, f x ≥ abs (2 * m + 1)) ↔ -9/4 ≤ m ∧ m ≤ 5/4 :=
by sorry

end NUMINAMATH_GPT_solution_set_f_gt_5_range_m_f_ge_abs_2m1_l67_6726
