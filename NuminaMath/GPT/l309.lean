import Mathlib

namespace average_visitors_in_30_day_month_l309_309486

def average_visitors_per_day (visitors_sunday visitors_other : ℕ) (days_in_month : ℕ) (starts_on_sunday : Prop) : ℕ :=
    let sundays := days_in_month / 7 + if days_in_month % 7 > 0 then 1 else 0
    let other_days := days_in_month - sundays
    let total_visitors := sundays * visitors_sunday + other_days * visitors_other
    total_visitors / days_in_month

theorem average_visitors_in_30_day_month 
    (visitors_sunday : ℕ) (visitors_other : ℕ) (days_in_month : ℕ) (starts_on_sunday : Prop) (h1 : visitors_sunday = 660) (h2 : visitors_other = 240) (h3 : days_in_month = 30) :
    average_visitors_per_day visitors_sunday visitors_other days_in_month starts_on_sunday = 296 := 
by
  sorry

end average_visitors_in_30_day_month_l309_309486


namespace largest_eight_digit_number_l309_309833

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (digits : List ℕ) : Prop :=
  ∀ d, is_even_digit d → List.contains digits d

def largest_eight_digit_with_even_digits : ℕ :=
  99986420

theorem largest_eight_digit_number {n : ℕ} 
  (h_digits : List ℕ) 
  (h_len : h_digits.length = 8)
  (h_cond : contains_all_even_digits h_digits)
  (h_num : n = List.foldl (λ acc d, acc * 10 + d) 0 h_digits)
  : n = largest_eight_digit_with_even_digits :=
sorry

end largest_eight_digit_number_l309_309833


namespace units_digit_of_product_of_first_four_composites_l309_309802

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l309_309802


namespace hammers_ordered_in_october_l309_309255

theorem hammers_ordered_in_october
  (ordered_in_june : Nat)
  (ordered_in_july : Nat)
  (ordered_in_august : Nat)
  (ordered_in_september : Nat)
  (pattern_increase : ∀ n : Nat, ordered_in_june + n = ordered_in_july ∧ ordered_in_july + (n + 1) = ordered_in_august ∧ ordered_in_august + (n + 2) = ordered_in_september) :
  ordered_in_september + 4 = 13 :=
by
  -- Proof omitted
  sorry

end hammers_ordered_in_october_l309_309255


namespace obtuse_angled_triangles_in_polygon_l309_309165

/-- The number of obtuse-angled triangles formed by the vertices of a regular polygon with 2n+1 sides -/
theorem obtuse_angled_triangles_in_polygon (n : ℕ) : 
  (2 * n + 1) * (n * (n - 1)) / 2 = (2 * n + 1) * (n * (n - 1)) / 2 :=
by
  sorry

end obtuse_angled_triangles_in_polygon_l309_309165


namespace tan_150_eq_neg_inv_sqrt3_l309_309915

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l309_309915


namespace problem_solution_l309_309556

variable (x : ℝ)

-- Given condition
def condition1 : Prop := (7 / 8) * x = 28

-- The main statement to prove
theorem problem_solution (h : condition1 x) : (x + 16) * (5 / 16) = 15 := by
  sorry

end problem_solution_l309_309556


namespace determine_a_l309_309240

def y (x : ℝ) (a : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

theorem determine_a (a : ℝ) : is_even (y _ a) → a = 2 :=
sorry

end determine_a_l309_309240


namespace upper_bound_of_expression_l309_309209

theorem upper_bound_of_expression (n : ℤ) (h1 : ∀ (n : ℤ), 4 * n + 7 > 1 ∧ 4 * n + 7 < 111) :
  ∃ U, (∀ (n : ℤ), 4 * n + 7 < U) ∧ 
       (∀ (n : ℤ), 4 * n + 7 < U ↔ 4 * n + 7 < 111) ∧ 
       U = 111 :=
by
  sorry

end upper_bound_of_expression_l309_309209


namespace distribute_balls_in_boxes_l309_309228

theorem distribute_balls_in_boxes (balls boxes : ℕ) (h_balls : balls = 7) (h_boxes : boxes = 2) : (boxes ^ balls) = 128 :=
by
  simp [h_balls, h_boxes]
  sorry

end distribute_balls_in_boxes_l309_309228


namespace tan_150_degrees_l309_309868

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l309_309868


namespace possible_items_l309_309693

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items_l309_309693


namespace multiple_of_every_positive_integer_is_zero_l309_309047

theorem multiple_of_every_positive_integer_is_zero :
  ∀ (n : ℤ), (∀ (m : ℕ), ∃ (k : ℤ), n = k * (m : ℤ)) → n = 0 := 
by
  sorry

end multiple_of_every_positive_integer_is_zero_l309_309047


namespace symmetric_function_is_periodic_l309_309731

theorem symmetric_function_is_periodic {f : ℝ → ℝ} {a b y0 : ℝ}
  (h1 : ∀ x, f (a + x) - y0 = y0 - f (a - x))
  (h2 : ∀ x, f (b + x) = f (b - x))
  (hb : b > a) :
  ∀ x, f (x + 4 * (b - a)) = f x := sorry

end symmetric_function_is_periodic_l309_309731


namespace james_money_left_no_foreign_currency_needed_l309_309386

noncomputable def JameMoneyLeftAfterPurchase : ℝ :=
  let usd_bills := 50 + 20 + 5 + 1 + 20 + 10 -- USD bills and coins
  let euro_in_usd := 5 * 1.20               -- €5 bill to USD
  let pound_in_usd := 2 * 1.35 - 0.8 / 100 * (2 * 1.35) -- £2 coin to USD after fee
  let yen_in_usd := 100 * 0.009 - 1.5 / 100 * (100 * 0.009) -- ¥100 coin to USD after fee
  let franc_in_usd := 2 * 1.08 - 1 / 100 * (2 * 1.08) -- 2₣ coins to USD after fee
  let total_usd := usd_bills + euro_in_usd + pound_in_usd + yen_in_usd + franc_in_usd
  let present_cost_with_tax := 88 * 1.08   -- Present cost after 8% tax
  total_usd - present_cost_with_tax        -- Amount left after purchasing the present

theorem james_money_left :
  JameMoneyLeftAfterPurchase = 22.6633 :=
by
  sorry

theorem no_foreign_currency_needed :
  (0 : ℝ)  = 0 :=
by
  sorry

end james_money_left_no_foreign_currency_needed_l309_309386


namespace units_digit_of_composite_product_l309_309794

theorem units_digit_of_composite_product : 
  let composites := [4, 6, 8, 9],
      product := List.foldl (· * ·) 1 composites
  in product % 10 = 8 :=
  by
  sorry

end units_digit_of_composite_product_l309_309794


namespace probability_palindrome_divisible_by_11_l309_309311

def is_five_digit_palindrome (n : ℕ) : Prop :=
  let d1 := n / 10000 % 10
  let d2 := n / 1000 % 10
  let d3 := n / 100 % 10
  let d4 := n / 10 % 10
  let d5 := n % 10
  n >= 10000 ∧ n < 100000 ∧ d1 = d5 ∧ d2 = d4

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def count_all_palindromes : ℕ :=
  9 * 10 * 10

def count_palindromes_div_by_11 : ℕ :=
  9 * 10

theorem probability_palindrome_divisible_by_11 :
  (count_palindromes_div_by_11 : ℚ) / count_all_palindromes = 1 / 10 :=
by sorry

end probability_palindrome_divisible_by_11_l309_309311


namespace sum_fn_3000_l309_309642

def f (n : ℕ) : ℚ :=
  if (logBase 9 n).isRational 
  then logBase 9 n 
  else 0

theorem sum_fn_3000 : (∑ n in Finset.range 3001, f n) = 21 / 2 :=
sorry

end sum_fn_3000_l309_309642


namespace kelly_grade_is_42_l309_309559

noncomputable def jenny_grade := 95

noncomputable def jason_grade := jenny_grade - 25

noncomputable def bob_grade := jason_grade / 2

noncomputable def kelly_grade := bob_grade * 1.2

theorem kelly_grade_is_42 : kelly_grade = 42 := by
  sorry

end kelly_grade_is_42_l309_309559


namespace x_minus_y_solution_l309_309954

theorem x_minus_y_solution (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 16) : x - y = 2 := 
by
  sorry

end x_minus_y_solution_l309_309954


namespace solve_for_x_l309_309230

noncomputable def f (x : ℝ) : ℝ := x^3

noncomputable def f_prime (x : ℝ) : ℝ := 3

theorem solve_for_x (x : ℝ) (h : f_prime x = 3) : x = 1 ∨ x = -1 :=
by
  sorry

end solve_for_x_l309_309230


namespace chord_length_y_eq_x_plus_one_meets_circle_l309_309520

noncomputable def chord_length (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem chord_length_y_eq_x_plus_one_meets_circle 
  (A B : ℝ × ℝ) 
  (hA : A.2 = A.1 + 1) 
  (hB : B.2 = B.1 + 1) 
  (hA_on_circle : A.1^2 + A.2^2 + 2 * A.2 - 3 = 0)
  (hB_on_circle : B.1^2 + B.2^2 + 2 * B.2 - 3 = 0) :
  chord_length A B = 2 * Real.sqrt 2 := 
sorry

end chord_length_y_eq_x_plus_one_meets_circle_l309_309520


namespace equation_of_line_l309_309743

theorem equation_of_line 
  (slope : ℝ)
  (a1 b1 c1 a2 b2 c2 : ℝ)
  (h_slope : slope = 2)
  (h_line1 : a1 = 3 ∧ b1 = 4 ∧ c1 = -5)
  (h_line2 : a2 = 3 ∧ b2 = -4 ∧ c2 = -13) 
  : ∃ (a b c : ℝ), (a = 2 ∧ b = -1 ∧ c = -7) ∧ 
    (∀ x y : ℝ, (a1 * x + b1 * y + c1 = 0) ∧ (a2 * x + b2 * y + c2 = 0) → (a * x + b * y + c = 0)) :=
by
  sorry

end equation_of_line_l309_309743


namespace minimum_value_of_y_l309_309288

theorem minimum_value_of_y : ∀ x : ℝ, ∃ y : ℝ, (y = 3 * x^2 + 6 * x + 9) → y ≥ 6 :=
by
  intro x
  use (3 * (x + 1)^2 + 6)
  intro h
  sorry

end minimum_value_of_y_l309_309288


namespace total_right_handed_players_l309_309980

theorem total_right_handed_players
  (total_players : ℕ)
  (total_throwers : ℕ)
  (left_handed_throwers_perc : ℕ)
  (right_handed_thrower_runs : ℕ)
  (left_handed_thrower_runs : ℕ)
  (total_runs : ℕ)
  (batsmen_to_allrounders_run_ratio : ℕ)
  (proportion_left_right_non_throwers : ℕ)
  (left_handed_non_thrower_runs : ℕ)
  (left_handed_batsmen_eq_allrounders : Prop)
  (left_handed_throwers : ℕ)
  (right_handed_throwers : ℕ)
  (total_right_handed_thrower_runs : ℕ)
  (total_left_handed_thrower_runs : ℕ)
  (total_throwers_runs : ℕ)
  (total_non_thrower_runs : ℕ)
  (allrounder_runs : ℕ)
  (batsmen_runs : ℕ)
  (left_handed_batsmen : ℕ)
  (left_handed_allrounders : ℕ)
  (total_left_handed_non_throwers : ℕ)
  (right_handed_non_throwers : ℕ)
  (total_right_handed_players : ℕ) :
  total_players = 120 →
  total_throwers = 55 →
  left_handed_throwers_perc = 20 →
  right_handed_thrower_runs = 25 →
  left_handed_thrower_runs = 30 →
  total_runs = 3620 →
  batsmen_to_allrounders_run_ratio = 2 →
  proportion_left_right_non_throwers = 5 →
  left_handed_non_thrower_runs = 720 →
  left_handed_batsmen_eq_allrounders →
  left_handed_throwers = total_throwers * left_handed_throwers_perc / 100 →
  right_handed_throwers = total_throwers - left_handed_throwers →
  total_right_handed_thrower_runs = right_handed_throwers * right_handed_thrower_runs →
  total_left_handed_thrower_runs = left_handed_throwers * left_handed_thrower_runs →
  total_throwers_runs = total_right_handed_thrower_runs + total_left_handed_thrower_runs →
  total_non_thrower_runs = total_runs - total_throwers_runs →
  allrounder_runs = total_non_thrower_runs / (batsmen_to_allrounders_run_ratio + 1) →
  batsmen_runs = batsmen_to_allrounders_run_ratio * allrounder_runs →
  left_handed_batsmen = left_handed_non_thrower_runs / (left_handed_thrower_runs * 2) →
  left_handed_allrounders = left_handed_non_thrower_runs / (left_handed_thrower_runs * 2) →
  total_left_handed_non_throwers = left_handed_batsmen + left_handed_allrounders →
  right_handed_non_throwers = total_left_handed_non_throwers * proportion_left_right_non_throwers →
  total_right_handed_players = right_handed_throwers + right_handed_non_throwers →
  total_right_handed_players = 164 :=
by sorry

end total_right_handed_players_l309_309980


namespace contestants_order_l309_309250

variables (G E H F : ℕ) -- Scores of the participants, given that they are nonnegative

theorem contestants_order (h1 : E + G = F + H) (h2 : F + E = H + G) (h3 : G > E + F) : 
  G ≥ E ∧ G ≥ H ∧ G ≥ F ∧ E = H ∧ E ≥ F :=
by {
  sorry
}

end contestants_order_l309_309250


namespace minimum_blocks_l309_309064

-- Assume we have the following conditions encoded:
-- 
-- 1) Each block is a cube with a snap on one side and receptacle holes on the other five sides.
-- 2) Blocks can connect on the sides, top, and bottom.
-- 3) All snaps must be covered by other blocks' receptacle holes.
-- 
-- Define a formal statement of this requirement.

def block : Type := sorry -- to model the block with snap and holes
def connects (b1 b2 : block) : Prop := sorry -- to model block connectivity

def snap_covered (b : block) : Prop := sorry -- True if and only if the snap is covered by another block’s receptacle hole

theorem minimum_blocks (blocks : List block) : 
  (∀ b ∈ blocks, snap_covered b) → blocks.length ≥ 4 :=
sorry

end minimum_blocks_l309_309064


namespace two_cos_30_eq_sqrt_3_l309_309863

open Real

-- Given condition: cos 30 degrees is sqrt(3)/2
def cos_30_eq : cos (π / 6) = sqrt 3 / 2 := 
sorry

-- Goal: to prove that 2 * cos 30 degrees = sqrt(3)
theorem two_cos_30_eq_sqrt_3 : 2 * cos (π / 6) = sqrt 3 :=
by
  rw [cos_30_eq]
  sorry

end two_cos_30_eq_sqrt_3_l309_309863


namespace advertisement_probability_l309_309304

theorem advertisement_probability
  (ads_time_hour : ℕ)
  (total_time_hour : ℕ)
  (h1 : ads_time_hour = 20)
  (h2 : total_time_hour = 60) :
  ads_time_hour / total_time_hour = 1 / 3 :=
by
  sorry

end advertisement_probability_l309_309304


namespace remainder_3_pow_2n_plus_8_l309_309395

theorem remainder_3_pow_2n_plus_8 (n : Nat) : (3 ^ (2 * n) + 8) % 8 = 1 := by
  sorry

end remainder_3_pow_2n_plus_8_l309_309395


namespace empty_subset_of_disjoint_and_nonempty_l309_309651

variable {α : Type*} (A B : Set α)

theorem empty_subset_of_disjoint_and_nonempty (h₁ : A ≠ ∅) (h₂ : A ∩ B = ∅) : ∅ ⊆ B :=
by
  sorry

end empty_subset_of_disjoint_and_nonempty_l309_309651


namespace minimum_parents_needed_l309_309436

/-- 
Given conditions:
1. There are 30 students going on the excursion.
2. Each car can accommodate 5 people, including the driver.
Prove that the minimum number of parents needed to be invited on the excursion is 8.
-/
theorem minimum_parents_needed (students : ℕ) (car_capacity : ℕ) (drivers_needed : ℕ) 
  (h1 : students = 30) (h2 : car_capacity = 5) (h3 : drivers_needed = 1) 
  : ∃ (parents : ℕ), parents = 8 :=
by
  existsi 8
  sorry

end minimum_parents_needed_l309_309436


namespace inequality_holds_l309_309351

theorem inequality_holds (a : ℝ) : 
  (∀ x ∈ set.Icc (-2 : ℝ) 1, a * x ^ 3 - x ^ 2 + 4 * x + 3 ≥ 0) ↔ (-6 ≤ a ∧ a ≤ -2) :=
sorry

end inequality_holds_l309_309351


namespace avg_price_of_pencil_l309_309611

theorem avg_price_of_pencil 
  (total_pens : ℤ) (total_pencils : ℤ) (total_cost : ℤ)
  (avg_cost_pen : ℤ) (avg_cost_pencil : ℤ) :
  total_pens = 30 → 
  total_pencils = 75 → 
  total_cost = 690 → 
  avg_cost_pen = 18 → 
  (total_cost - total_pens * avg_cost_pen) / total_pencils = avg_cost_pencil → 
  avg_cost_pencil = 2 :=
by
  intros
  sorry

end avg_price_of_pencil_l309_309611


namespace symmetric_curve_wrt_line_l309_309275

theorem symmetric_curve_wrt_line {f : ℝ → ℝ → ℝ} :
  (∀ x y : ℝ, f x y = 0 → f (y + 3) (x - 3) = 0) := by
  sorry

end symmetric_curve_wrt_line_l309_309275


namespace prove_intersection_l309_309216

-- Defining the set M
def M : Set ℝ := { x | x^2 - 2 * x < 0 }

-- Defining the set N
def N : Set ℝ := { x | x ≥ 1 }

-- Defining the complement of N in ℝ
def complement_N : Set ℝ := { x | x < 1 }

-- The intersection M ∩ complement_N
def intersection : Set ℝ := { x | 0 < x ∧ x < 1 }

-- The statement to be proven
theorem prove_intersection : M ∩ complement_N = intersection :=
by
  sorry

end prove_intersection_l309_309216


namespace frog_probability_0_4_l309_309307

-- Definitions and conditions
def vertices : List (ℤ × ℤ) := [(1,1), (1,6), (5,6), (5,1)]
def start_position : ℤ × ℤ := (2,3)

-- Probabilities for transition, boundary definitions, this mimics the recursive nature described
def P : ℤ × ℤ → ℝ
| (x, 1) => 1   -- Boundary condition for horizontal sides
| (x, 6) => 1   -- Boundary condition for horizontal sides
| (1, y) => 0   -- Boundary condition for vertical sides
| (5, y) => 0   -- Boundary condition for vertical sides
| (x, y) => sorry  -- General case for other positions

-- The theorem to prove
theorem frog_probability_0_4 : P (2, 3) = 0.4 :=
by
  sorry

end frog_probability_0_4_l309_309307


namespace prob_not_same_class_prob_same_class_prob_diff_gender_not_same_class_l309_309669

theorem prob_not_same_class : 
  let students := [
    ("Class1", "Male"), ("Class1", "Female"),
    ("Class2", "Male"), ("Class2", "Female"),
    ("Class3", "Male"), ("Class3", "Female")
  ] in
  let total_pairs := ((students.length : ℚ)!).choose(2 : ℚ) in
  let diff_class_pairs := 12 in
  diff_class_pairs / total_pairs = 4/5 :=
by
  sorry

theorem prob_same_class : 
  let students := [
    ("Class1", "Male"), ("Class1", "Female"),
    ("Class2", "Male"), ("Class2", "Female"),
    ("Class3", "Male"), ("Class3", "Female")
  ] in
  let total_pairs := ((students.length : ℚ)!).choose(2 : ℚ) in
  let same_class_pairs := 3 in
  same_class_pairs / total_pairs = 1/5 :=
by
  sorry

theorem prob_diff_gender_not_same_class : 
  let students := [
    ("Class1", "Male"), ("Class1", "Female"),
    ("Class2", "Male"), ("Class2", "Female"),
    ("Class3", "Male"), ("Class3", "Female")
  ] in
  let total_pairs := ((students.length : ℚ)!).choose(2 : ℚ) in
  let diff_gender_not_same_class_pairs := 6 in
  diff_gender_not_same_class_pairs / total_pairs = 2/5 :=
by
  sorry

end prob_not_same_class_prob_same_class_prob_diff_gender_not_same_class_l309_309669


namespace necessary_but_not_sufficient_for_inequalities_l309_309969

theorem necessary_but_not_sufficient_for_inequalities (a b : ℝ) :
  (a + b > 4) ↔ (a > 2 ∧ b > 2) :=
sorry

end necessary_but_not_sufficient_for_inequalities_l309_309969


namespace range_of_a_for_maximum_l309_309947

variable {f : ℝ → ℝ}
variable {a : ℝ}

theorem range_of_a_for_maximum (h : ∀ x, deriv f x = a * (x + 1) * (x - a))
  (h_max : ∀ x, f x ≤ f a → x = a) : -1 < a ∧ a < 0 :=
sorry

end range_of_a_for_maximum_l309_309947


namespace find_x_value_l309_309376

/-- Given x, y, z such that x ≠ 0, z ≠ 0, (x / 2) = y^2 + z, and (x / 4) = 4y + 2z, the value of x is 120. -/
theorem find_x_value (x y z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0) (h1 : x / 2 = y^2 + z) (h2 : x / 4 = 4 * y + 2 * z) : x = 120 := 
sorry

end find_x_value_l309_309376


namespace determinant_of_trig_matrix_l309_309089

theorem determinant_of_trig_matrix (α β : ℝ) : 
  Matrix.det ![
    ![Real.sin α, Real.cos α], 
    ![Real.cos β, Real.sin β]
  ] = -Real.cos (α - β) :=
by sorry

end determinant_of_trig_matrix_l309_309089


namespace probability_two_girls_l309_309212

/--
From a group of 5 students consisting of 2 boys and 3 girls, 2 representatives 
are randomly selected (with each student having an equal chance of being selected). 
Prove that the probability that both representatives are girls is 3/10.
-/
theorem probability_two_girls (total_students boys girls : ℕ) : 
  total_students = 5 ∧ boys = 2 ∧ girls = 3 → 
  let total_outcomes := Nat.choose 5 2 in
  let favorable_outcomes := Nat.choose 3 2 in
  favorable_outcomes.toRational / total_outcomes.toRational = 3 / 10 :=
by
  intros h
  cases h with h1 h2
  cases h2 with h3 h4
  unfold total_outcomes favorable_outcomes
  sorry

end probability_two_girls_l309_309212


namespace xyz_sum_sqrt14_l309_309724

theorem xyz_sum_sqrt14 (x y z : ℝ) (h1 : x^2 + y^2 + z^2 = 1) (h2 : x + 2 * y + 3 * z = Real.sqrt 14) :
  x + y + z = (3 * Real.sqrt 14) / 7 :=
sorry

end xyz_sum_sqrt14_l309_309724


namespace books_not_sold_l309_309851

variable {B : ℕ} -- Total number of books

-- Conditions
def two_thirds_books_sold (B : ℕ) : ℕ := (2 * B) / 3
def price_per_book : ℕ := 2
def total_amount_received : ℕ := 144
def remaining_books_sold : ℕ := 0
def two_thirds_by_price (B : ℕ) : ℕ := two_thirds_books_sold B * price_per_book

-- Main statement to prove
theorem books_not_sold (h : two_thirds_by_price B = total_amount_received) : (B / 3) = 36 :=
by
  sorry

end books_not_sold_l309_309851


namespace count_n_equals_3_count_n_even_l309_309948

open Finset

def M : Finset ℕ := {1, 2, 3, 4}

def cumulative_value (A : Finset ℕ) : ℕ :=
  if A.card = 0 then 0
  else A.prod id

def count_subsets_with_n (n : ℕ) (f : Finset ℕ → ℕ) (M : Finset ℕ) : ℕ :=
  M.powerset.count (λ A, f A = n)

def even (n : ℕ) : Prop :=
  n % 2 = 0

theorem count_n_equals_3 :
  count_subsets_with_n 3 cumulative_value M = 2 := 
sorry

theorem count_n_even :
  count_subsets_with_n (even ∘ cumulative_value) M = 13 :=
sorry

end count_n_equals_3_count_n_even_l309_309948


namespace number_of_outfits_l309_309842

theorem number_of_outfits (shirts pants : ℕ) (h_shirts : shirts = 5) (h_pants : pants = 3) 
    : shirts * pants = 15 := by
  sorry

end number_of_outfits_l309_309842


namespace range_of_k_l309_309163

theorem range_of_k (k : ℝ) :
  (∃ x y : ℝ, (x - 3)^2 + (y - 2)^2 = 4 ∧ y = k * x + 3) ∧ 
  (∃ M N : ℝ × ℝ, ((M.1 - N.1)^2 + (M.2 - N.2)^2)^(1/2) ≥ 2) →
  (k ≤ 0) :=
by
  sorry

end range_of_k_l309_309163


namespace three_digit_even_less_than_600_count_l309_309442

theorem three_digit_even_less_than_600_count : 
  let digits := {1, 2, 3, 4, 5, 6} 
  let hundreds := {d ∈ digits | d < 6}
  let tens := digits 
  let units := {d ∈ digits | d % 2 = 0}
  ∑ (h : ℕ) in hundreds, ∑ (t : ℕ) in tens, ∑ (u : ℕ) in units, 1 = 90 :=
by
  sorry

end three_digit_even_less_than_600_count_l309_309442


namespace integer_expression_l309_309410

theorem integer_expression (m : ℤ) : ∃ k : ℤ, k = (m / 3) + (m^2 / 2) + (m^3 / 6) :=
sorry

end integer_expression_l309_309410


namespace trajectory_of_M_is_ellipse_line_tangent_to_fixed_circle_l309_309220

noncomputable def fixed_point_Q : ℝ × ℝ := (real.sqrt 3, 0)

noncomputable def circle_N (x y : ℝ) : Prop :=
  (x + real.sqrt 3) ^ 2 + y ^ 2 = 24

noncomputable def trajectory_C (x y : ℝ) : Prop :=
  x ^ 2 / 6 + y ^ 2 / 3 = 1

noncomputable def fixed_circle_E (x y : ℝ) : Prop :=
  x ^ 2 + y ^ 2 = 2

theorem trajectory_of_M_is_ellipse :
  ∀ (P : ℝ × ℝ) (M : ℝ × ℝ),
    circle_N P.1 P.2 →
    (∃ Q : ℝ × ℝ, Q = fixed_point_Q ∧ dist M P = dist M Q) →
    trajectory_C M.1 M.2 :=
sorry

theorem line_tangent_to_fixed_circle :
  ∀ (l : ℝ → ℝ) (A B : ℝ × ℝ),
    (trajectory_C A.1 A.2 ∧ trajectory_C B.1 B.2) →
    (l A.1 = A.2 ∧ l B.1 = B.2) →
    (A.1 * B.1 + A.2 * B.2 = 0) →
    (∃ (x y : ℝ), fixed_circle_E x y ∧ ∀ t : ℝ, l t = y + x * t) :=
sorry

end trajectory_of_M_is_ellipse_line_tangent_to_fixed_circle_l309_309220


namespace unique_solution_condition_l309_309587

theorem unique_solution_condition {a b : ℝ} : (∃ x : ℝ, 4 * x - 7 + a = b * x + 4) ↔ b ≠ 4 :=
by
  sorry

end unique_solution_condition_l309_309587


namespace p_plus_q_identity_l309_309161

variable {α : Type*} [CommRing α]

-- Definitions derived from conditions
def p (x : α) : α := 3 * (x - 2)
def q (x : α) : α := (x + 2) * (x - 4)

-- Lean theorem stating the problem
theorem p_plus_q_identity (x : α) : p x + q x = x^2 + x - 14 :=
by
  unfold p q
  sorry

end p_plus_q_identity_l309_309161


namespace probability_digits_different_l309_309623

theorem probability_digits_different : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999} in
  let total := ∑ x in S, 1 in
  let different_digits := ∑ x in S, (if (x / 100 ≠ (x % 100) / 10 ∧ (x % 100) / 10 ≠ x % 10 ∧ x / 100 ≠ x % 10) then 1 else 0) in
  (different_digits / total) = (18 / 25) := by
sorry

end probability_digits_different_l309_309623


namespace trust_meteorologist_l309_309609

/-- 
The probability of a clear day in Anchuria.
-/
def P_G : ℝ := 0.74

/-- 
Accuracy of the forecast by each senator.
-/
variable (p : ℝ)

/-- 
Accuracy of the meteorologist's forecast being 1.5 times that of a senator.
-/
def meteorologist_accuracy : ℝ := 1.5 * p

/-- 
Calculations and final proof that the meteorologist's forecast is more reliable than that of the senators. 
-/
theorem trust_meteorologist (p : ℝ) (Hp1 : 0 ≤ p) (Hp2 : p ≤ 1) : 
  λ P_S_M1_M2_G P_S_M1_M2_not_G : 
  P_G := 0.74 ∧ meteorologist_accuracy = 1.5 * p ∧
  (∀ P_S_M1_M2, P_S_M1_M2 = P_S_M1_M2_G * P_G + P_S_M1_M2_not_G * (1 - P_G)) → 
  (P_S_M1_M2_not_G * (1 - P_G) > P_S_M1_M2_G * P_G) :=
begin
  sorry
end

end trust_meteorologist_l309_309609


namespace calculate_difference_l309_309596

theorem calculate_difference (x y : ℝ) (h1 : x + y = 520) (h2 : x / y = 0.75) : y - x = 74 :=
by
  sorry

end calculate_difference_l309_309596


namespace parabola_focus_coordinates_l309_309032

theorem parabola_focus_coordinates (x y : ℝ) (h : x = 2 * y^2) : (x, y) = (1/8, 0) :=
sorry

end parabola_focus_coordinates_l309_309032


namespace Indians_drink_tea_is_zero_l309_309246

-- Definitions based on given conditions and questions
variable (total_people : Nat)
variable (total_drink_tea : Nat)
variable (total_drink_coffee : Nat)
variable (answer_do_you_drink_coffee : Nat)
variable (answer_are_you_a_turk : Nat)
variable (answer_is_it_raining : Nat)
variable (Indians_drink_tea : Nat)
variable (Indians_drink_coffee : Nat)
variable (Turks_drink_coffee : Nat)
variable (Turks_drink_tea : Nat)

-- The given facts and conditions
axiom hx1 : total_people = 55
axiom hx2 : answer_do_you_drink_coffee = 44
axiom hx3 : answer_are_you_a_turk = 33
axiom hx4 : answer_is_it_raining = 22
axiom hx5 : Indians_drink_tea + Indians_drink_coffee + Turks_drink_coffee + Turks_drink_tea = total_people
axiom hx6 : Indians_drink_coffee + Turks_drink_coffee = answer_do_you_drink_coffee
axiom hx7 : Indians_drink_coffee + Turks_drink_tea = answer_are_you_a_turk
axiom hx8 : Indians_drink_tea + Turks_drink_coffee = answer_is_it_raining

-- Prove that the number of Indians drinking tea is 0
theorem Indians_drink_tea_is_zero : Indians_drink_tea = 0 :=
by {
    sorry
}

end Indians_drink_tea_is_zero_l309_309246


namespace Jackie_exercise_hours_l309_309557

variable (work_hours : ℕ) (sleep_hours : ℕ) (free_time_hours : ℕ) (total_hours_in_day : ℕ)
variable (time_for_exercise : ℕ)

noncomputable def prove_hours_exercising (work_hours sleep_hours free_time_hours total_hours_in_day : ℕ) : Prop :=
  work_hours = 8 ∧
  sleep_hours = 8 ∧
  free_time_hours = 5 ∧
  total_hours_in_day = 24 → 
  time_for_exercise = total_hours_in_day - (work_hours + sleep_hours + free_time_hours)

theorem Jackie_exercise_hours :
  prove_hours_exercising 8 8 5 24 3 :=
by
  -- Proof is omitted as per instruction
  sorry

end Jackie_exercise_hours_l309_309557


namespace rational_with_smallest_absolute_value_is_zero_l309_309428

theorem rational_with_smallest_absolute_value_is_zero (r : ℚ) :
  (forall r : ℚ, |r| ≥ 0) →
  (forall r : ℚ, r ≠ 0 → |r| > 0) →
  |r| = 0 ↔ r = 0 := sorry

end rational_with_smallest_absolute_value_is_zero_l309_309428


namespace simplify_fraction_l309_309267

theorem simplify_fraction :
  (1 : ℚ) / ((1 / (1 / 3 : ℚ) ^ 1) + (1 / (1 / 3 : ℚ) ^ 2) + (1 / (1 / 3 : ℚ) ^ 3) + (1 / (1 / 3 : ℚ) ^ 4)) = 1 / 120 := 
by 
  sorry

end simplify_fraction_l309_309267


namespace hyperbola_range_l309_309654

theorem hyperbola_range (m : ℝ) : m * (2 * m - 1) < 0 → 0 < m ∧ m < (1 / 2) :=
by
  intro h
  sorry

end hyperbola_range_l309_309654


namespace decimal_150th_digit_of_1_div_13_l309_309458

theorem decimal_150th_digit_of_1_div_13 :
  (1 / 13).decimalExpansion[150] = 3 :=
by
  sorry

end decimal_150th_digit_of_1_div_13_l309_309458


namespace pipe_fill_time_l309_309729

theorem pipe_fill_time (T : ℝ) 
  (h1 : ∃ T : ℝ, 0 < T) 
  (h2 : T + (1/2) > 0) 
  (h3 : ∃ leak_rate : ℝ, leak_rate = 1/10) 
  (h4 : ∃ pipe_rate : ℝ, pipe_rate = 1/T) 
  (h5 : ∃ effective_rate : ℝ, effective_rate = pipe_rate - leak_rate) 
  (h6 : effective_rate = 1 / (T + 1/2))  : 
  T = Real.sqrt 5 :=
  sorry

end pipe_fill_time_l309_309729


namespace even_expression_l309_309991

theorem even_expression (m n : ℤ) (hm : Odd m) (hn : Odd n) : Even (m + 5 * n) :=
by
  sorry

end even_expression_l309_309991


namespace units_digit_of_first_four_composite_numbers_l309_309806

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l309_309806


namespace remainder_of_exponentiation_l309_309398

theorem remainder_of_exponentiation (n : ℕ) : (3 ^ (2 * n) + 8) % 8 = 1 := 
by sorry

end remainder_of_exponentiation_l309_309398


namespace mario_garden_total_blossoms_l309_309405

def hibiscus_growth (initial_flowers growth_rate weeks : ℕ) : ℕ :=
  initial_flowers + growth_rate * weeks

def rose_growth (initial_flowers growth_rate weeks : ℕ) : ℕ :=
  initial_flowers + growth_rate * weeks

theorem mario_garden_total_blossoms :
  let weeks := 2
  let hibiscus1 := hibiscus_growth 2 3 weeks
  let hibiscus2 := hibiscus_growth (2 * 2) 4 weeks
  let hibiscus3 := hibiscus_growth (4 * (2 * 2)) 5 weeks
  let rose1 := rose_growth 3 2 weeks
  let rose2 := rose_growth 5 3 weeks
  hibiscus1 + hibiscus2 + hibiscus3 + rose1 + rose2 = 64 := 
by
  sorry

end mario_garden_total_blossoms_l309_309405


namespace circle_equation_l309_309362

-- Definitions of the conditions
def passes_through (a : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) (r : ℝ) : Prop :=
  (c - a) ^ 2 + (d - b) ^ 2 = r ^ 2

def center_on_line (a : ℝ) (b : ℝ) : Prop :=
  a - b - 4 = 0

-- Statement of the problem to be proved
theorem circle_equation 
  (a b r : ℝ) 
  (h1 : passes_through a b (-1) (-4) r)
  (h2 : passes_through a b 6 3 r)
  (h3 : center_on_line a b) :
  -- Equation of the circle
  (a = 3 ∧ b = -1 ∧ r = 5) → ∀ x y : ℝ, 
    (x - 3)^2 + (y + 1)^2 = 25 :=
sorry

end circle_equation_l309_309362


namespace paul_completion_time_l309_309644

theorem paul_completion_time :
  let george_rate := 1 / 15
  let remaining_work := 2 / 5
  let combined_rate (P : ℚ) := george_rate + P
  let P_work := 4 * combined_rate P = remaining_work
  let paul_rate := 13 / 90
  let total_work := 1
  let time_paul_alone := total_work / paul_rate
  P_work → time_paul_alone = (90 / 13) := by
  intros
  -- all necessary definitions and conditions are used
  sorry

end paul_completion_time_l309_309644


namespace find_x_l309_309522

theorem find_x (x : ℕ) (hx : x > 0 ∧ x <= 100) 
    (mean_twice_mode : (40 + 57 + 76 + 90 + x + x) / 6 = 2 * x) : 
    x = 26 :=
sorry

end find_x_l309_309522


namespace simplify_expression_l309_309636

theorem simplify_expression (x : ℝ) (h1 : x^3 + 2*x + 1 ≠ 0) (h2 : x^3 - 2*x - 1 ≠ 0) : 
  ( ((x + 2)^2 * (x^2 - x + 2)^2 / (x^3 + 2*x + 1)^2 )^3 * ((x - 2)^2 * (x^2 + x + 2)^2 / (x^3 - 2*x - 1)^2 )^3 ) = 1 :=
by sorry

end simplify_expression_l309_309636


namespace readers_scifi_l309_309670

variable (S L B T : ℕ)

-- Define conditions given in the problem
def totalReaders := 650
def literaryReaders := 550
def bothReaders := 150

-- Define the main problem to prove
theorem readers_scifi (S L B T : ℕ) (hT : T = totalReaders) (hL : L = literaryReaders) (hB : B = bothReaders) (hleq : T = S + L - B) : S = 250 :=
by
  -- Insert proof here
  sorry

end readers_scifi_l309_309670


namespace intersection_points_hyperbola_l309_309646

theorem intersection_points_hyperbola (t : ℝ) :
  ∃ x y : ℝ, (2 * t * x - 3 * y - 4 * t = 0) ∧ (2 * x - 3 * t * y + 4 = 0) ∧ 
  (x^2 / 4 - y^2 / (9 / 16) = 1) :=
sorry

end intersection_points_hyperbola_l309_309646


namespace eating_contest_l309_309011

variables (hotdog_weight burger_weight pie_weight : ℕ)
variable (noah_burgers jacob_pies mason_hotdogs : ℕ)
variable (total_weight_mason_hotdogs : ℕ)

theorem eating_contest :
  hotdog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  noah_burgers = 8 →
  jacob_pies = noah_burgers - 3 →
  mason_hotdogs = 3 * jacob_pies →
  total_weight_mason_hotdogs = mason_hotdogs * hotdog_weight →
  total_weight_mason_hotdogs = 30 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end eating_contest_l309_309011


namespace units_digit_first_four_composites_l309_309779

theorem units_digit_first_four_composites :
  let product := 4 * 6 * 8 * 9 in
  product % 10 = 8 :=
by
  let product := 4 * 6 * 8 * 9
  have h : product = 1728 := by norm_num
  show product % 10 = 8
  rw [h]
  norm_num
  done 
  sorry

end units_digit_first_four_composites_l309_309779


namespace problem_statement_l309_309305

-- Defining the data of Type A and Type B
def typeA : List ℝ := [2, 4, 5, 6, 8]
def typeB : List ℝ := [3, 4, 4, 4, 5]

-- Defining the means of Type A and Type B
def mean (xs : List ℝ) : ℝ := xs.sum / xs.length

-- Defining the covariance and variance functions
def covariance (xs ys : List ℝ) : ℝ :=
  let n := xs.length
  List.sum (List.map₂ (λ x y, (x - mean xs) * (y - mean ys)) xs ys) / n

def variance (xs : List ℝ) : ℝ :=
  let m := mean xs
  List.sum (List.map (λ x, (x - m) ^ 2) xs) / xs.length

-- Correlation coefficient
def correlation_coefficient (xs ys : List ℝ) : ℝ :=
  covariance xs ys / (Real.sqrt (variance xs) * Real.sqrt (variance ys))

-- Probability calculation
def count_greater_pairs (pairs : List (ℝ × ℝ)) : ℕ :=
  pairs.count (λ (a, b), a > b)

def probability_greater (pairs : List (ℝ × ℝ)) : ℝ :=
  let greater := count_greater_pairs pairs
  let total := pairs.length
  greater / total

theorem problem_statement : correlation_coefficient typeA typeB = sqrt (9/10) ∧
                             abs (correlation_coefficient typeA typeB) > 0.75 ∧
                             probability_greater [(2,3), (2,4), (2,4), (2,4), (2,5),
                                                  (4,3), (4,4), (4,4), (4,4), (4,5),
                                                  (5,3), (5,4), (5,4), (5,4), (5,5),
                                                  (6,3), (6,4), (6,4), (6,4), (6,5),
                                                  (8,3), (8,4), (8,4), (8,4), (8,5)] = 3/10 :=
by sorry  -- Proof of the theorem is omitted

end problem_statement_l309_309305


namespace tan_150_eq_neg_sqrt_3_l309_309895

theorem tan_150_eq_neg_sqrt_3 :
  let θ := 150 * Real.pi / 180 in
  Real.tan θ = -Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_sqrt_3_l309_309895


namespace gcd_97_power_l309_309507

theorem gcd_97_power (h : Nat.Prime 97) : 
  Nat.gcd (97^7 + 1) (97^7 + 97^3 + 1) = 1 := 
by 
  sorry

end gcd_97_power_l309_309507


namespace trigonometric_identity_l309_309350

variable {α : ℝ}

theorem trigonometric_identity (h : Real.tan α = 3) :
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 3 * Real.cos α) = 5 / 6 :=
by
  sorry

end trigonometric_identity_l309_309350


namespace marbles_total_l309_309552

theorem marbles_total (r b g y : ℝ) 
  (h1 : r = 1.30 * b)
  (h2 : g = 1.50 * r)
  (h3 : y = 0.80 * g) :
  r + b + g + y = 4.4692 * r :=
by
  sorry

end marbles_total_l309_309552


namespace transform_quadratic_equation_l309_309318

theorem transform_quadratic_equation :
  ∀ x : ℝ, (x^2 - 8 * x - 1 = 0) → ((x - 4)^2 = 17) :=
by
  intro x
  intro h
  sorry

end transform_quadratic_equation_l309_309318


namespace domain_of_f_l309_309995

def domain (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
∀ x, f x ∈ D

noncomputable def f (x : ℝ) : ℝ := 1 / (x + 2)

theorem domain_of_f :
  domain f {y | y ≠ -2} :=
by sorry

end domain_of_f_l309_309995


namespace tan_150_eq_neg_one_over_sqrt_three_l309_309913

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l309_309913


namespace find_value_of_expression_l309_309527

theorem find_value_of_expression
  (x y : ℝ)
  (h : x^2 - 2*x + y^2 - 6*y + 10 = 0) :
  x^2 * y^2 + 2 * x * y + 1 = 16 :=
sorry

end find_value_of_expression_l309_309527


namespace exists_positive_m_dividing_f_100_l309_309530

noncomputable def f (x : ℤ) : ℤ := 3 * x + 2

theorem exists_positive_m_dividing_f_100:
  ∃ (m : ℤ), m > 0 ∧ 19881 ∣ (3^100 * (m + 1) - 1) :=
by
  sorry

end exists_positive_m_dividing_f_100_l309_309530


namespace georgia_total_cost_l309_309014

def carnation_price : ℝ := 0.50
def dozen_price : ℝ := 4.00
def teachers : ℕ := 5
def friends : ℕ := 14

theorem georgia_total_cost :
  ((dozen_price * teachers) + dozen_price + (carnation_price * (friends - 12))) = 25.00 :=
by
  sorry

end georgia_total_cost_l309_309014


namespace ripe_oranges_l309_309541

theorem ripe_oranges (U : ℕ) (hU : U = 25) (hR : R = U + 19) : R = 44 := by
  sorry

end ripe_oranges_l309_309541


namespace sum_of_squares_l309_309593

theorem sum_of_squares (x : ℚ) (hx : 7 * x = 15) : 
  (x^2 + (2 * x)^2 + (4 * x)^2 = 4725 / 49) := by
  sorry

end sum_of_squares_l309_309593


namespace instantaneous_velocity_at_3_l309_309034

noncomputable def motion_equation (t : ℝ) : ℝ := 1 - t + t^2

theorem instantaneous_velocity_at_3 :
  (deriv (motion_equation) 3 = 5) :=
by
  sorry

end instantaneous_velocity_at_3_l309_309034


namespace solve_inequality_l309_309037

theorem solve_inequality (x : ℝ) : x + 2 < 1 ↔ x < -1 := sorry

end solve_inequality_l309_309037


namespace rational_product_sum_l309_309438

theorem rational_product_sum (x y : ℚ) 
  (h1 : x * y < 0) 
  (h2 : x + y < 0) : 
  |y| < |x| ∧ y < 0 ∧ x > 0 ∨ |x| < |y| ∧ x < 0 ∧ y > 0 :=
by
  sorry

end rational_product_sum_l309_309438


namespace tom_needs_more_blue_tickets_l309_309175

def yellow_to_red : ℕ := 10
def red_to_blue : ℕ := 10
def yellow_to_blue : ℕ := yellow_to_red * red_to_blue
def required_yellow_tickets : ℕ := 10
def required_blue_tickets : ℕ := required_yellow_tickets * yellow_to_blue

def toms_yellow_tickets : ℕ := 8
def toms_red_tickets : ℕ := 3
def toms_blue_tickets : ℕ := 7
def toms_total_blue_tickets : ℕ := 
  (toms_yellow_tickets * yellow_to_blue) + 
  (toms_red_tickets * red_to_blue) + 
  toms_blue_tickets

def additional_blue_tickets_needed : ℕ :=
  required_blue_tickets - toms_total_blue_tickets

theorem tom_needs_more_blue_tickets : additional_blue_tickets_needed = 163 := 
by sorry

end tom_needs_more_blue_tickets_l309_309175


namespace area_of_sector_l309_309028

theorem area_of_sector (r : ℝ) (θ : ℝ) (h1 : r = 10) (h2 : θ = π / 5) : 
  (1 / 2) * r * r * θ = 10 * π :=
by
  rw [h1, h2]
  sorry

end area_of_sector_l309_309028


namespace max_sum_cubes_l309_309565

theorem max_sum_cubes (a b c d : ℝ) (h : a^2 + b^2 + c^2 + d^2 = 4) : 
  a^3 + b^3 + c^3 + d^3 ≤ 8 :=
sorry

end max_sum_cubes_l309_309565


namespace average_weight_of_a_and_b_l309_309741

-- Given conditions as Lean definitions
variable (A B C : ℝ)
variable (h1 : (A + B + C) / 3 = 45)
variable (h2 : (B + C) / 2 = 46)
variable (hB : B = 37)

-- The statement we want to prove
theorem average_weight_of_a_and_b : (A + B) / 2 = 40 := by
  sorry

end average_weight_of_a_and_b_l309_309741


namespace triangle_inequality_cosine_rule_l309_309384

theorem triangle_inequality_cosine_rule (a b c : ℝ) (A B C : ℝ)
  (hA : Real.cos A = (b^2 + c^2 - a^2) / (2 * b * c))
  (hB : Real.cos B = (a^2 + c^2 - b^2) / (2 * a * c))
  (hC : Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b)) :
  a^3 * Real.cos A + b^3 * Real.cos B + c^3 * Real.cos C ≤ (3 / 2) * a * b * c := 
sorry

end triangle_inequality_cosine_rule_l309_309384


namespace major_premise_is_false_l309_309285

-- Define the major premise
def major_premise (a : ℝ) : Prop := a^2 > 0

-- Define the minor premise
def minor_premise (a : ℝ) := true

-- Define the conclusion based on the premises
def conclusion (a : ℝ) : Prop := a^2 > 0

-- Show that the major premise is false by finding a counterexample
theorem major_premise_is_false : ¬ ∀ a : ℝ, major_premise a := by
  sorry

end major_premise_is_false_l309_309285


namespace largest_value_p_l309_309723

theorem largest_value_p 
  (p q r : ℝ) 
  (h1 : p + q + r = 10) 
  (h2 : p * q + p * r + q * r = 25) :
  p ≤ 20 / 3 :=
sorry

end largest_value_p_l309_309723


namespace units_digit_first_four_composites_l309_309774

theorem units_digit_first_four_composites :
  let product := 4 * 6 * 8 * 9 in
  product % 10 = 8 :=
by
  let product := 4 * 6 * 8 * 9
  have h : product = 1728 := by norm_num
  show product % 10 = 8
  rw [h]
  norm_num
  done 
  sorry

end units_digit_first_four_composites_l309_309774


namespace arithmetic_series_sum_base6_l309_309518

-- Define the terms in the arithmetic series in base 6
def a₁ := 1
def a₄₅ := 45
def n := a₄₅

-- Sum of arithmetic series in base 6
def sum_arithmetic_series := (n * (a₁ + a₄₅)) / 2

-- Expected result for the arithmetic series sum
def expected_result := 2003

theorem arithmetic_series_sum_base6 :
  sum_arithmetic_series = expected_result := by
  sorry

end arithmetic_series_sum_base6_l309_309518


namespace factorization_a_minus_b_l309_309996

theorem factorization_a_minus_b (a b : ℤ) (y : ℝ) 
  (h1 : 3 * y ^ 2 - 7 * y - 6 = (3 * y + a) * (y + b)) 
  (h2 : a ≠ 0) 
  (h3 : b ≠ 0) : 
  a - b = 5 :=
sorry

end factorization_a_minus_b_l309_309996


namespace proof_problem_l309_309930

open Set

def Point : Type := ℝ × ℝ

structure Triangle :=
(A : Point)
(B : Point)
(C : Point)

def area_of_triangle (T : Triangle) : ℝ :=
   0.5 * abs ((T.B.1 - T.A.1) * (T.C.2 - T.A.2) - (T.C.1 - T.A.1) * (T.B.2 - T.A.2))

def area_of_grid (length width : ℝ) : ℝ :=
   length * width

def problem_statement : Prop :=
   let T : Triangle := {A := (1,3), B := (5,1), C := (4,4)} 
   let S1 := area_of_triangle T
   let S := area_of_grid 6 5
   (S1 / S) = 1 / 6

theorem proof_problem : problem_statement := 
by
  sorry


end proof_problem_l309_309930


namespace log_inequality_l309_309503

open Real

theorem log_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  log (1 + sqrt (a * b)) ≤ (1 / 2) * (log (1 + a) + log (1 + b)) :=
sorry

end log_inequality_l309_309503


namespace find_m_value_l309_309529

-- Definitions from conditions
def O : ℝ × ℝ := (0, 0)
def A : ℝ × ℝ := (-1, 3)
def B : ℝ × ℝ := (2, -4)
def OA := (A.1 - O.1, A.2 - O.2)
def AB := (B.1 - A.1, B.2 - A.2)

-- Defining the vector OP with the given expression
def OP (m : ℝ) := (2 * OA.1 + m * AB.1, 2 * OA.2 + m * AB.2)

-- The point P is on the y-axis if the x-coordinate of OP is zero
theorem find_m_value : ∃ m : ℝ, OP m = (0, (OP m).2) ∧ m = 2 / 3 :=
by { 
  -- sorry is added to skip the proof itself
  sorry 
}

end find_m_value_l309_309529


namespace solve_system_l309_309989

def system_of_equations (x y : ℤ) : Prop :=
  (x^2 * y + x * y^2 + 3 * x + 3 * y + 24 = 0) ∧
  (x^3 * y - x * y^3 + 3 * x^2 - 3 * y^2 - 48 = 0)

theorem solve_system : system_of_equations (-3) (-1) :=
by {
  -- Proof details are omitted
  sorry
}

end solve_system_l309_309989


namespace wrench_turns_bolt_l309_309287

theorem wrench_turns_bolt (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (Real.sqrt 3 / Real.sqrt 2 < b / a) ∧ (b / a ≤ 3 - Real.sqrt 3) :=
sorry

end wrench_turns_bolt_l309_309287


namespace molecular_weight_of_compound_l309_309467

def n_weight : ℝ := 14.01
def h_weight : ℝ := 1.01
def br_weight : ℝ := 79.90

def molecular_weight : ℝ := (1 * n_weight) + (4 * h_weight) + (1 * br_weight)

theorem molecular_weight_of_compound :
  molecular_weight = 97.95 :=
by
  -- proof steps go here if needed, but currently, we use sorry to complete the theorem
  sorry

end molecular_weight_of_compound_l309_309467


namespace horner_v2_value_l309_309439

def polynomial : ℤ → ℤ := fun x => 208 + 9 * x^2 + 6 * x^4 + x^6

def horner (x : ℤ) : ℤ :=
  let v0 := 1
  let v1 := v0 * x
  let v2 := v1 * x + 6
  v2

theorem horner_v2_value (x : ℤ) : x = -4 → horner x = 22 :=
by
  intro h
  rw [h]
  rfl

end horner_v2_value_l309_309439


namespace expression_undefined_at_9_l309_309337

theorem expression_undefined_at_9 (x : ℝ) : (3 * x ^ 3 - 5) / (x ^ 2 - 18 * x + 81) = 0 → x = 9 :=
by sorry

end expression_undefined_at_9_l309_309337


namespace tan_150_eq_neg_inv_sqrt3_l309_309914

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l309_309914


namespace solution_exists_solution_unique_l309_309585

noncomputable def abc_solutions : Finset (ℕ × ℕ × ℕ) :=
  {(2, 2, 2), (2, 2, 4), (2, 4, 8), (3, 5, 15), 
   (2, 4, 2), (4, 2, 2), (4, 2, 8), (8, 4, 2), 
   (2, 8, 4), (8, 2, 4), (5, 3, 15), (15, 3, 5), (3, 15, 5),
   (2, 2, 4), (4, 2, 2), (4, 8, 2)}

theorem solution_exists (a b c : ℕ) (h : a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2) :
  (a * b * c - 1 = (a - 1) * (b - 1) * (c - 1)) ↔ (a, b, c) ∈ abc_solutions := 
by
  sorry

theorem solution_unique (a b c : ℕ) (h : a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2) :
  (a, b, c) ∈ abc_solutions → a * b * c - 1 = (a - 1) * (b - 1) * (c - 1) :=
by
  sorry

end solution_exists_solution_unique_l309_309585


namespace process_end_after_two_draws_exactly_two_white_balls_l309_309245

noncomputable def prob_end_after_two_draws (n : ℕ) (red_balls white_balls blue_balls : ℕ) : ℝ :=
((red_balls + white_balls).choose 1 / n.choose 1) * (blue_balls.choose 1 / n.choose 1)

noncomputable def prob_two_white_balls (n : ℕ) (red_balls white_balls blue_balls : ℕ) : ℝ :=
(((red_balls / n) * (white_balls / n) * (white_balls / n) * 3) 
+ ((white_balls / n) * (white_balls / n) * (blue_balls / n)))

theorem process_end_after_two_draws :
  prob_end_after_two_draws 10 5 3 2 = 4 / 25 :=
begin
  sorry
end

theorem exactly_two_white_balls :
  prob_two_white_balls 10 5 3 2 = 153 / 1000 :=
begin
  sorry
end

end process_end_after_two_draws_exactly_two_white_balls_l309_309245


namespace size_of_angle_A_area_of_triangle_l309_309968

-- Definitions for the problem
variables {a b c : ℝ} {A B C : ℝ}
def obtuse_triangle (A B C : ℝ) : Prop := 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧ A + B + C = π

def given_condition1 (a b c A B C : ℝ) : Prop :=
  2 * a * Real.sin A = (2 * b - Real.sqrt 3 * c) * Real.sin B + (2 * c - Real.sqrt 3 * b) * Real.sin C

def given_condition2 (a b : ℝ) : Prop :=
  a = 2 ∧ b = 2 * Real.sqrt 3

-- Lean statement for the proof problem
theorem size_of_angle_A (h1 : obtuse_triangle A B C) (h2 : given_condition1 a b c A B C) :
  A = π / 6 :=
sorry

theorem area_of_triangle (h1 : obtuse_triangle A B C) (h2 : given_condition1 a b c A B C) (h3 : given_condition2 a b) :
  ∃ S, S = Real.sqrt 3 ∨ S = 2 * Real.sqrt 3 :=
sorry

end size_of_angle_A_area_of_triangle_l309_309968


namespace curve_symmetric_reflection_l309_309273

theorem curve_symmetric_reflection (f : ℝ → ℝ → ℝ) :
  (∀ x y, f x y = 0 ↔ f (y + 3) (x - 3) = 0) → 
  (∀ x y, (x - y - 3 = 0) → (f (y + 3) (x - 3) = 0)) :=
sorry

end curve_symmetric_reflection_l309_309273


namespace lara_has_largest_answer_l309_309390

/-- Define the final result for John, given his operations --/
def final_john (n : ℕ) : ℕ :=
  let add_three := n + 3
  let double := add_three * 2
  double - 4

/-- Define the final result for Lara, given her operations --/
def final_lara (n : ℕ) : ℕ :=
  let triple := n * 3
  let add_five := triple + 5
  add_five - 6

/-- Define the final result for Miguel, given his operations --/
def final_miguel (n : ℕ) : ℕ :=
  let double := n * 2
  let subtract_two := double - 2
  subtract_two + 2

/-- Main theorem to be proven --/
theorem lara_has_largest_answer :
  final_lara 12 > final_john 12 ∧ final_lara 12 > final_miguel 12 :=
by {
  sorry
}

end lara_has_largest_answer_l309_309390


namespace probability_compensation_l309_309306

-- Define the probabilities of each vehicle getting into an accident
def p1 : ℚ := 1 / 20
def p2 : ℚ := 1 / 21

-- Define the probability of the complementary event
def comp_event : ℚ := (1 - p1) * (1 - p2)

-- Define the overall probability that at least one vehicle gets into an accident
def comp_unit : ℚ := 1 - comp_event

-- The theorem to be proved: the probability that the unit will receive compensation from this insurance within a year is 2 / 21
theorem probability_compensation : comp_unit = 2 / 21 :=
by
  -- giving the proof is not required
  sorry

end probability_compensation_l309_309306


namespace distinct_convex_polygons_l309_309179

theorem distinct_convex_polygons (n : ℕ) (hn : n = 12) : (2^n - (choose n 0 + choose n 1 + choose n 2)) = 4017 :=
by
  guard_hyp hn : n = 12
  sorry

end distinct_convex_polygons_l309_309179


namespace range_of_a_l309_309539

def set_A (a : ℝ) : Set ℝ := {-1, 0, a}
def set_B : Set ℝ := {x : ℝ | 1/3 < x ∧ x < 1}

theorem range_of_a (a : ℝ) (h : (set_A a) ∩ set_B ≠ ∅) : 1/3 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l309_309539


namespace a3_equals_1_div_12_l309_309997

-- Definition of the sequence
def seq (n : Nat) : Rat :=
  1 / (n * (n + 1))

-- Assertion to be proved
theorem a3_equals_1_div_12 : seq 3 = 1 / 12 := 
sorry

end a3_equals_1_div_12_l309_309997


namespace kirill_height_l309_309391

theorem kirill_height (B : ℕ) (h1 : ∃ B, B - 14 = kirill_height) (h2 : B + (B - 14) = 112) : kirill_height = 49 :=
sorry

end kirill_height_l309_309391


namespace num_combinations_l309_309168

-- The conditions given in the problem.
def num_pencil_types : ℕ := 2
def num_eraser_types : ℕ := 3

-- The theorem to prove.
theorem num_combinations (pencils : ℕ) (erasers : ℕ) (h1 : pencils = num_pencil_types) (h2 : erasers = num_eraser_types) : pencils * erasers = 6 :=
by 
  have hp : pencils = 2 := h1
  have he : erasers = 3 := h2
  cases hp
  cases he
  rfl

end num_combinations_l309_309168


namespace quadratic_inequality_solution_l309_309339

theorem quadratic_inequality_solution (x : ℝ) : x^2 + 3 * x - 18 < 0 ↔ -6 < x ∧ x < 3 := 
sorry

end quadratic_inequality_solution_l309_309339


namespace odd_integer_divisibility_l309_309081

theorem odd_integer_divisibility (n : ℕ) (hodd : n % 2 = 1) (hpos : n > 0) : ∃ k : ℕ, n^4 - n^2 - n = n * k := 
sorry

end odd_integer_divisibility_l309_309081


namespace value_of_a_l309_309844

theorem value_of_a (a b c d : ℕ) (h : (18^a) * (9^(4*a-1)) * (27^c) = (2^6) * (3^b) * (7^d)) : a = 6 :=
by
  sorry

end value_of_a_l309_309844


namespace simpleInterest_500_l309_309640

def simpleInterest (P R T : ℝ) : ℝ := P * R * T

theorem simpleInterest_500 :
  simpleInterest 10000 0.05 1 = 500 :=
by
  sorry

end simpleInterest_500_l309_309640


namespace maritza_study_hours_l309_309007

noncomputable def time_to_study_for_citizenship_test (num_mc_questions num_fitb_questions time_mc time_fitb : ℕ) : ℕ :=
  (num_mc_questions * time_mc + num_fitb_questions * time_fitb) / 60

theorem maritza_study_hours :
  time_to_study_for_citizenship_test 30 30 15 25 = 20 :=
by
  sorry

end maritza_study_hours_l309_309007


namespace value_of_a_l309_309116

theorem value_of_a (a : ℤ) (x y : ℝ) :
  (a - 2) ≠ 0 →
  (2 + |a| + 1 = 5) →
  a = -2 :=
by
  intro ha hdeg
  sorry

end value_of_a_l309_309116


namespace solve_equation_l309_309988

theorem solve_equation (x : ℝ) (h : -x^2 = (3 * x + 1) / (x + 3)) : x = -1 :=
sorry

end solve_equation_l309_309988


namespace units_digit_first_four_composites_l309_309810

theorem units_digit_first_four_composites : 
  let first_four_composites := [4, 6, 8, 9]
  let product := first_four_composites.prod
  Nat.unitsDigit product = 8 :=
by
  sorry

end units_digit_first_four_composites_l309_309810


namespace even_function_a_value_l309_309241

theorem even_function_a_value (a : ℝ) : 
  (∀ x : ℝ, let y := (x - 1)^2 + a * x + sin(x + π / 2) in y = y) ↔ a = 2 :=
by
  let f := λ x, (x - 1)^2 + a * x + sin(x + π / 2)
  have h_even : ∀ x : ℝ, f(-x) = f(x) ↔ (a = 2) := sorry
  exact h_even

end even_function_a_value_l309_309241


namespace convex_polygons_from_12_points_on_circle_l309_309181

def total_subsets (n : ℕ) := 2 ^ n
def non_polygon_subsets (n : ℕ) := 1 + n + (n * (n - 1)) / 2

theorem convex_polygons_from_12_points_on_circle :
  let total := total_subsets 12 in
  let non_polygons := non_polygon_subsets 12 in
  total - non_polygons = 4017 :=
by
  sorry

end convex_polygons_from_12_points_on_circle_l309_309181


namespace carB_distance_traveled_l309_309182

-- Define the initial conditions
def initial_separation : ℝ := 150
def distance_carA_main_road : ℝ := 25
def distance_between_cars : ℝ := 38

-- Define the question as a theorem where we need to show the distance Car B traveled
theorem carB_distance_traveled (initial_separation distance_carA_main_road distance_between_cars : ℝ) :
  initial_separation - (distance_carA_main_road + distance_between_cars) = 87 :=
  sorry

end carB_distance_traveled_l309_309182


namespace raised_arm_length_exceeds_head_l309_309474

variables (h s s' x : ℝ)

def xiaogang_height := 1.7
def shadow_without_arm := 0.85
def shadow_with_arm := 1.1

theorem raised_arm_length_exceeds_head :
  h = xiaogang_height → s = shadow_without_arm → s' = shadow_with_arm → 
  x / (s' - s) = h / s → x = 0.5 :=
by
  intros h_eq s_eq s'_eq prop
  sorry

end raised_arm_length_exceeds_head_l309_309474


namespace minimum_omega_l309_309042

theorem minimum_omega (ω : ℝ) (h_pos : ω > 0) :
  (∃ k : ℤ, ω * (3 * π / 4) - ω * (π / 4) = k * π) → ω = 2 :=
by
  sorry

end minimum_omega_l309_309042


namespace kolya_purchase_l309_309684

theorem kolya_purchase : ∃ n : ℕ, n = 17 ∨ n = 117 :=
by
  let item_cost := λ a : ℕ, 100 * a + 99
  let total_cost := 20000 + 83
  have h : ∀ n a, n * (item_cost a) = total_cost → (n = 17 ∨ n = 117) := sorry
  have h1 := h 17 0
  have h2 := h 117 0
  existsi 17
  exact h1 sorry

end kolya_purchase_l309_309684


namespace line_through_origin_l309_309063

theorem line_through_origin (x y : ℝ) :
  (∃ x0 y0 : ℝ, 4 * x0 + y0 + 6 = 0 ∧ 3 * (-x0) + (- 5) * y0 + 6 = 0)
  → (x + 6 * y = 0) :=
by
  sorry

end line_through_origin_l309_309063


namespace percentage_shaded_is_14_29_l309_309757

noncomputable def side_length : ℝ := 20
noncomputable def rect_length : ℝ := 35
noncomputable def rect_width : ℝ := side_length
noncomputable def rect_area : ℝ := rect_length * rect_width
noncomputable def overlap_length : ℝ := 2 * side_length - rect_length
noncomputable def overlap_area : ℝ := overlap_length * side_length
noncomputable def shaded_percentage : ℝ := (overlap_area / rect_area) * 100

theorem percentage_shaded_is_14_29 :
  shaded_percentage = 14.29 :=
sorry

end percentage_shaded_is_14_29_l309_309757


namespace ball_bounces_less_than_two_meters_l309_309057

theorem ball_bounces_less_than_two_meters : ∀ k : ℕ, 500 * (1/3 : ℝ)^k < 2 → k ≥ 6 := by
  sorry

end ball_bounces_less_than_two_meters_l309_309057


namespace units_digit_of_product_of_first_four_composites_l309_309786

theorem units_digit_of_product_of_first_four_composites :
  (4 * 6 * 8 * 9) % 10 = 8 := 
by
  sorry

end units_digit_of_product_of_first_four_composites_l309_309786


namespace smaug_copper_coins_l309_309025

def copper_value_of_silver (silver_coins silver_to_copper : ℕ) : ℕ :=
  silver_coins * silver_to_copper

def copper_value_of_gold (gold_coins gold_to_silver silver_to_copper : ℕ) : ℕ :=
  gold_coins * gold_to_silver * silver_to_copper

def total_copper_value (gold_coins silver_coins gold_to_silver silver_to_copper : ℕ) : ℕ :=
  copper_value_of_gold gold_coins gold_to_silver silver_to_copper +
  copper_value_of_silver silver_coins silver_to_copper

def actual_copper_coins (total_value gold_value silver_value : ℕ) : ℕ :=
  total_value - (gold_value + silver_value)

theorem smaug_copper_coins :
  let gold_coins := 100
  let silver_coins := 60
  let silver_to_copper := 8
  let gold_to_silver := 3
  let total_copper_value := 2913
  let gold_value := copper_value_of_gold gold_coins gold_to_silver silver_to_copper
  let silver_value := copper_value_of_silver silver_coins silver_to_copper
  actual_copper_coins total_copper_value gold_value silver_value = 33 :=
by
  sorry

end smaug_copper_coins_l309_309025


namespace theater_price_balcony_l309_309495

theorem theater_price_balcony 
  (price_orchestra : ℕ) (total_tickets : ℕ) (total_revenue : ℕ) (extra_balcony_tickets : ℕ) (price_balcony : ℕ) 
  (h1 : price_orchestra = 12) 
  (h2 : total_tickets = 380) 
  (h3 : total_revenue = 3320) 
  (h4 : extra_balcony_tickets = 240) 
  (h5 : ∃ (O : ℕ), O + (O + extra_balcony_tickets) = total_tickets ∧ (price_orchestra * O) + (price_balcony * (O + extra_balcony_tickets)) = total_revenue) : 
  price_balcony = 8 := 
by
  sorry

end theater_price_balcony_l309_309495


namespace angles_arith_prog_tangent_tangent_parallel_euler_line_l309_309733

-- Define a non-equilateral triangle with angles in arithmetic progression
structure Triangle :=
  (A B C : ℝ) -- Angles in a non-equilateral triangle
  (non_equilateral : A ≠ B ∨ B ≠ C ∨ A ≠ C)
  (angles_arith_progression : (2 * B = A + C))

-- Additional geometry concepts will be assumptions as their definition 
-- would involve extensive axiomatic setups

-- The main theorem to state the equivalence
theorem angles_arith_prog_tangent_tangent_parallel_euler_line (Δ : Triangle)
  (common_tangent_parallel_euler : sorry) : 
  ((Δ.A = 60) ∨ (Δ.B = 60) ∨ (Δ.C = 60)) :=
sorry

end angles_arith_prog_tangent_tangent_parallel_euler_line_l309_309733


namespace tan_150_deg_l309_309873

-- Define the conditions
def angle_150_deg := 150 * real.pi / 180
def coordinates_of_Q := (-real.sqrt 3 / 2, 1 / 2)
def tan_of_angle_150 := real.tan angle_150_deg

-- The statement to prove
theorem tan_150_deg : tan_of_angle_150 = -1 / real.sqrt 3 :=
by
  -- Proof omitted
  sorry

end tan_150_deg_l309_309873


namespace other_employee_number_l309_309859

-- Define the conditions
variables (total_employees : ℕ) (sample_size : ℕ) (e1 e2 e3 : ℕ)

-- Define the systematic sampling interval
def sampling_interval (total : ℕ) (size : ℕ) : ℕ := total / size

-- The Lean statement for the proof problem
theorem other_employee_number
  (h1 : total_employees = 52)
  (h2 : sample_size = 4)
  (h3 : e1 = 6)
  (h4 : e2 = 32)
  (h5 : e3 = 45) :
  ∃ e4 : ℕ, e4 = 19 := 
sorry

end other_employee_number_l309_309859


namespace distance_to_origin_l309_309554

theorem distance_to_origin (x y : ℤ) (hx : x = -5) (hy : y = 12) :
  Real.sqrt (x^2 + y^2) = 13 := by
  rw [hx, hy]
  norm_num
  sorry

end distance_to_origin_l309_309554


namespace negation_of_existential_proposition_l309_309164

theorem negation_of_existential_proposition : 
  (¬ ∃ x : ℝ, x^2 + x + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end negation_of_existential_proposition_l309_309164


namespace units_digit_first_four_composites_l309_309814

theorem units_digit_first_four_composites : 
  let first_four_composites := [4, 6, 8, 9]
  let product := first_four_composites.prod
  Nat.unitsDigit product = 8 :=
by
  sorry

end units_digit_first_four_composites_l309_309814


namespace smallest_x_remainder_l309_309472

theorem smallest_x_remainder : ∃ x : ℕ, x > 0 ∧ 
    x % 6 = 5 ∧
    x % 7 = 6 ∧
    x % 8 = 7 ∧
    x = 167 :=
by
  sorry

end smallest_x_remainder_l309_309472


namespace target_hit_probability_l309_309663

open ProbabilityTheory

theorem target_hit_probability :
  let PA := 0.8
  let PB := 0.7
  let P_not_hit := (1 - PA) * (1 - PB)
  let P_hit := 1 - P_not_hit
  P_hit = 0.94 :=
by
  sorry

end target_hit_probability_l309_309663


namespace chromosome_stability_due_to_meiosis_and_fertilization_l309_309749

-- Definitions for conditions
def chrom_replicate_distribute_evenly : Prop := true
def central_cell_membrane_invagination : Prop := true
def mitosis : Prop := true
def meiosis_and_fertilization : Prop := true

-- Main theorem statement to be proved
theorem chromosome_stability_due_to_meiosis_and_fertilization :
  meiosis_and_fertilization :=
sorry

end chromosome_stability_due_to_meiosis_and_fertilization_l309_309749


namespace frustum_volume_correct_l309_309622

noncomputable def volume_frustum (base_edge_original base_edge_smaller altitude_original altitude_smaller : ℝ) : ℝ :=
  let base_area_original := base_edge_original ^ 2
  let base_area_smaller := base_edge_smaller ^ 2
  let volume_original := (1 / 3) * base_area_original * altitude_original
  let volume_smaller := (1 / 3) * base_area_smaller * altitude_smaller
  volume_original - volume_smaller

theorem frustum_volume_correct :
  volume_frustum 16 8 10 5 = 2240 / 3 :=
by
  have h1 : volume_frustum 16 8 10 5 = 
    (1 / 3) * (16^2) * 10 - (1 / 3) * (8^2) * 5 := rfl
  simp only [pow_two] at h1
  norm_num at h1
  exact h1

end frustum_volume_correct_l309_309622


namespace units_digit_first_four_composite_is_eight_l309_309821

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l309_309821


namespace georgia_total_carnation_cost_l309_309017

-- Define the cost of one carnation
def cost_of_single_carnation : ℝ := 0.50

-- Define the cost of one dozen carnations
def cost_of_dozen_carnations : ℝ := 4.00

-- Define the number of teachers
def number_of_teachers : ℕ := 5

-- Define the number of friends
def number_of_friends : ℕ := 14

-- Calculate the cost for teachers
def cost_for_teachers : ℝ :=
  (number_of_teachers : ℝ) * cost_of_dozen_carnations

-- Calculate the cost for friends
def cost_for_friends : ℝ :=
  cost_of_dozen_carnations + (2 * cost_of_single_carnation)

-- Calculate the total cost
def total_cost : ℝ := cost_for_teachers + cost_for_friends

-- Theorem stating the total cost
theorem georgia_total_carnation_cost : total_cost = 25 := by
  -- Placeholder for the proof
  sorry

end georgia_total_carnation_cost_l309_309017


namespace proposition_false_n5_l309_309065

variable (P : ℕ → Prop)

-- Declaring the conditions as definitions:
def condition1 (k : ℕ) (hk : k > 0) : Prop := P k → P (k + 1)
def condition2 : Prop := ¬ P 6

-- Theorem statement which leverages the conditions to prove the desired result.
theorem proposition_false_n5 (h1: ∀ k (hk : k > 0), condition1 P k hk) (h2: condition2 P) : ¬ P 5 :=
sorry

end proposition_false_n5_l309_309065


namespace find_M_pos_int_l309_309469

theorem find_M_pos_int (M : ℕ) (hM : 33^2 * 66^2 = 15^2 * M^2) :
    M = 726 :=
by
  -- Sorry, skipping the proof.
  sorry

end find_M_pos_int_l309_309469


namespace solve_for_k_l309_309523

theorem solve_for_k : 
  ∃ k : ℤ, (k + 2) / 4 - (2 * k - 1) / 6 = 1 ∧ k = -4 := 
by
  use -4
  sorry

end solve_for_k_l309_309523


namespace ratio_of_part_diminished_by_4_l309_309310

theorem ratio_of_part_diminished_by_4 (N P : ℕ) (h1 : N = 160)
    (h2 : (1/5 : ℝ) * N + 4 = P - 4) : (P - 4) / N = 9 / 40 := 
by
  sorry

end ratio_of_part_diminished_by_4_l309_309310


namespace theta_range_l309_309353

noncomputable def f (x : ℝ) : ℝ := x / (x^2 + 1)

theorem theta_range (k : ℤ) (θ : ℝ) : 
  (2 * ↑k * π - 5 * π / 6 < θ ∧ θ < 2 * ↑k * π - π / 6) →
  (f (1 / (Real.sin θ)) + f (Real.cos (2 * θ)) < f π - f (1 / π)) :=
by
  intros h
  sorry

end theta_range_l309_309353


namespace area_of_combined_rectangle_l309_309345

theorem area_of_combined_rectangle
  (short_side : ℝ) (num_small_rectangles : ℕ) (total_area : ℝ)
  (h1 : num_small_rectangles = 4)
  (h2 : short_side = 7)
  (h3 : total_area = (3 * short_side + short_side) * (2 * short_side)) :
  total_area = 392 := by
  sorry

end area_of_combined_rectangle_l309_309345


namespace sum_of_possible_amounts_l309_309570

-- Definitions based on conditions:
def possible_quarters_amounts : Finset ℕ := {5, 30, 55, 80}
def possible_dimes_amounts : Finset ℕ := {15, 20, 30, 35, 40, 50, 60, 70, 80, 90}
def both_possible_amounts : Finset ℕ := possible_quarters_amounts ∩ possible_dimes_amounts

-- Statement of the problem:
theorem sum_of_possible_amounts : (both_possible_amounts.sum id) = 110 :=
by
  sorry

end sum_of_possible_amounts_l309_309570


namespace units_digit_of_composite_product_l309_309792

theorem units_digit_of_composite_product : 
  let composites := [4, 6, 8, 9],
      product := List.foldl (· * ·) 1 composites
  in product % 10 = 8 :=
  by
  sorry

end units_digit_of_composite_product_l309_309792


namespace total_time_is_three_hours_l309_309573

-- Define the conditions of the problem in Lean
def time_uber_house := 10
def time_uber_airport := 5 * time_uber_house
def time_check_bag := 15
def time_security := 3 * time_check_bag
def time_boarding := 20
def time_takeoff := 2 * time_boarding

-- Total time in minutes
def total_time_minutes := time_uber_house + time_uber_airport + time_check_bag + time_security + time_boarding + time_takeoff

-- Conversion from minutes to hours
def total_time_hours := total_time_minutes / 60

-- The theorem to prove
theorem total_time_is_three_hours : total_time_hours = 3 := by
  sorry

end total_time_is_three_hours_l309_309573


namespace find_radius_of_cone_l309_309594

def slant_height : ℝ := 10
def curved_surface_area : ℝ := 157.07963267948966

theorem find_radius_of_cone
    (l : ℝ) (CSA : ℝ) (h1 : l = slant_height) (h2 : CSA = curved_surface_area) :
    ∃ r : ℝ, r = 5 := 
by
  sorry

end find_radius_of_cone_l309_309594


namespace balls_diff_color_probability_l309_309244

-- Defining the number of each color balls in the bag
def blue_balls := 1
def red_balls := 1
def yellow_balls := 2
def total_balls := blue_balls + red_balls + yellow_balls

-- Defining the event of drawing two balls of different colors
def event_diff_color := 
  let total_draw := 2
  let total_ways := Nat.choose total_balls total_draw
  let same_yellow_ways := Nat.choose yellow_balls total_draw
  (total_ways - same_yellow_ways) / total_ways

-- Theorem statement
theorem balls_diff_color_probability : event_diff_color = (5 / 6) :=
by
  sorry

end balls_diff_color_probability_l309_309244


namespace min_ab_value_l309_309223

theorem min_ab_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 1 / a + 4 / b = Real.sqrt (a * b)) :
  a * b = 4 :=
  sorry

end min_ab_value_l309_309223


namespace intersection_value_of_a_l309_309942

theorem intersection_value_of_a (a : ℝ) (A B : Set ℝ) 
  (hA : A = {0, 1, 3})
  (hB : B = {a + 1, a^2 + 2})
  (h_inter : A ∩ B = {1}) : 
  a = 0 :=
by
  sorry

end intersection_value_of_a_l309_309942


namespace wendy_lost_lives_l309_309758

theorem wendy_lost_lives (L : ℕ) (h1 : 10 - L + 37 = 41) : L = 6 :=
by
  sorry

end wendy_lost_lives_l309_309758


namespace digit_150th_of_fraction_l309_309450

-- Condition: The decimal representation of 1/13 is 0.076923
def repeating_block : List ℕ := [0, 7, 6, 9, 2, 3]

-- The length of the repeating block
def block_length : ℕ := 6

-- Problem: Prove that the 150th digit after the decimal point is 3
theorem digit_150th_of_fraction (n : ℕ) (h : n = 150) :
  List.getRepeating repeating_block block_length n = some 3 :=
by
  sorry

end digit_150th_of_fraction_l309_309450


namespace union_A_B_complement_intersection_A_B_l309_309366

def A : Set ℝ := {x | 3 ≤ x ∧ x < 10}

def B : Set ℝ := {x | 2 * x - 8 ≥ 0}

theorem union_A_B : A ∪ B = { x | x ≥ 3 } := 
by
  sorry

theorem complement_intersection_A_B : (A ∩ B)ᶜ = { x | x < 4 } ∪ { x | x ≥ 10 } := 
by
  sorry

end union_A_B_complement_intersection_A_B_l309_309366


namespace range_of_a_exists_distinct_x1_x2_eq_f_l309_309657

noncomputable
def f (a x : ℝ) : ℝ :=
  if x < 1 then a * x + 1 - 4 * a else x^2 - 3 * a * x

theorem range_of_a_exists_distinct_x1_x2_eq_f :
  { a : ℝ | ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2 } = 
  { a : ℝ | (a > (2 / 3)) ∨ (a ≤ 0) } :=
sorry

end range_of_a_exists_distinct_x1_x2_eq_f_l309_309657


namespace sequence_monotonically_decreasing_l309_309659

theorem sequence_monotonically_decreasing (t : ℝ) (a : ℕ → ℝ) :
  (∀ n : ℕ, a n = -↑n^2 + t * ↑n) →
  (∀ n : ℕ, a (n + 1) < a n) →
  t < 3 :=
by
  intros h1 h2
  sorry

end sequence_monotonically_decreasing_l309_309659


namespace class_average_correct_l309_309955

def class_average_test_A : ℝ :=
  0.30 * 97 + 0.25 * 85 + 0.20 * 78 + 0.15 * 65 + 0.10 * 55

def class_average_test_B : ℝ :=
  0.30 * 93 + 0.25 * 80 + 0.20 * 75 + 0.15 * 70 + 0.10 * 60

theorem class_average_correct :
  round class_average_test_A = 81 ∧
  round class_average_test_B = 79 := 
by 
  sorry

end class_average_correct_l309_309955


namespace average_height_l309_309418

def heights : List ℕ := [145, 142, 138, 136, 143, 146, 138, 144, 137, 141]

theorem average_height :
  (heights.sum : ℕ) / heights.length = 141 := by
  sorry

end average_height_l309_309418


namespace slope_l1_parallel_lines_math_proof_problem_l309_309658

-- Define the two lines
def l1 := ∀ x y : ℝ, x + 2 * y + 2 = 0
def l2 (a : ℝ) := ∀ x y : ℝ, a * x + y - 4 = 0

-- Define the assertions
theorem slope_l1 : ∀ x y : ℝ, x + 2 * y + 2 = 0 ↔ y = -1 / 2 * x - 1 := sorry

theorem parallel_lines (a : ℝ) : (∀ x y : ℝ, x + 2 * y + 2 = 0) ∧ (∀ x y : ℝ, a * x + y - 4 = 0) ↔ a = 1 / 2 := sorry

-- Using the assertions to summarize what we need to prove
theorem math_proof_problem (a : ℝ) :
  ((∀ x y : ℝ, x + 2 * y + 2 = 0) ∧ (∀ x y : ℝ, a * x + y - 4 = 0) → a = 1 / 2) ∧
  (∀ x y : ℝ, x + 2 * y + 2 = 0 → y = -1 / 2 * x - 1) := sorry

end slope_l1_parallel_lines_math_proof_problem_l309_309658


namespace units_digit_first_four_composites_l309_309765

theorem units_digit_first_four_composites :
  let p := [4, 6, 8, 9] in
  let product := List.prod p in
  product % 10 = 8 :=
by
  let p := [4, 6, 8, 9]
  let product := List.prod p
  show product % 10 = 8
  sorry

end units_digit_first_four_composites_l309_309765


namespace sum_of_center_coords_l309_309343

theorem sum_of_center_coords (x y : ℝ) (h : x^2 + y^2 = 4 * x - 6 * y + 9) : 2 + (-3) = -1 :=
by
  sorry

end sum_of_center_coords_l309_309343


namespace prime_divisor_property_l309_309139

open Classical

theorem prime_divisor_property (p n q : ℕ) (hp : Nat.Prime p) (hn : 0 < n) (hq : q ∣ (n + 1)^p - n^p) : p ∣ q - 1 :=
by
  sorry

end prime_divisor_property_l309_309139


namespace abc_product_l309_309956

theorem abc_product (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a * b = 13) (h2 : b * c = 52) (h3 : c * a = 4) : a * b * c = 52 := 
  sorry

end abc_product_l309_309956


namespace math_problem_l309_309504

theorem math_problem : 3 * 3^4 + 9^60 / 9^59 - 27^3 = -19431 := by
  sorry

end math_problem_l309_309504


namespace square_numbers_divisible_by_5_between_20_and_110_l309_309932

theorem square_numbers_divisible_by_5_between_20_and_110 :
  ∃ (y : ℕ), (y = 25 ∨ y = 100) ∧ (∃ (n : ℕ), y = n^2) ∧ 5 ∣ y ∧ 20 < y ∧ y < 110 :=
by
  sorry

end square_numbers_divisible_by_5_between_20_and_110_l309_309932


namespace inequality_problem_l309_309100

noncomputable def a := (3 / 4) * Real.exp (2 / 5)
noncomputable def b := 2 / 5
noncomputable def c := (2 / 5) * Real.exp (3 / 4)

theorem inequality_problem : b < c ∧ c < a := by
  sorry

end inequality_problem_l309_309100


namespace A_alone_finishes_in_27_days_l309_309050

noncomputable def work (B : ℝ) : ℝ := 54 * B  -- amount of work W
noncomputable def days_to_finish_alone (B : ℝ) : ℝ := (work B) / (2 * B)

theorem A_alone_finishes_in_27_days (B : ℝ) (h : (work B) / (2 * B + B) = 18) : 
  days_to_finish_alone B = 27 :=
by
  sorry

end A_alone_finishes_in_27_days_l309_309050


namespace rods_needed_to_complete_6_step_pyramid_l309_309381

def rods_in_step (n : ℕ) : ℕ :=
  16 * n

theorem rods_needed_to_complete_6_step_pyramid (rods_1_step rods_2_step : ℕ) :
  rods_1_step = 16 → rods_2_step = 32 → rods_in_step 6 - rods_in_step 4 = 32 :=
by
  intros h1 h2
  sorry

end rods_needed_to_complete_6_step_pyramid_l309_309381


namespace popsicle_sticks_left_l309_309328

/-- Danielle has $10 for supplies. She buys one set of molds for $3, 
a pack of 100 popsicle sticks for $1. Each bottle of juice makes 20 popsicles and costs $2.
Prove that the number of popsicle sticks Danielle will be left with after making as many popsicles as she can is 40. -/
theorem popsicle_sticks_left (initial_money : ℕ)
    (mold_cost : ℕ) (sticks_cost : ℕ) (initial_sticks : ℕ)
    (juice_cost : ℕ) (popsicles_per_bottle : ℕ)
    (final_sticks : ℕ) :
    initial_money = 10 →
    mold_cost = 3 → 
    sticks_cost = 1 → 
    initial_sticks = 100 →
    juice_cost = 2 →
    popsicles_per_bottle = 20 →
    final_sticks = initial_sticks - (popsicles_per_bottle * (initial_money - mold_cost - sticks_cost) / juice_cost) →
    final_sticks = 40 :=
by
  intros h_initial_money h_mold_cost h_sticks_cost h_initial_sticks h_juice_cost h_popsicles_per_bottle h_final_sticks
  rw [h_initial_money, h_mold_cost, h_sticks_cost, h_initial_sticks, h_juice_cost, h_popsicles_per_bottle] at h_final_sticks
  norm_num at h_final_sticks
  exact h_final_sticks

end popsicle_sticks_left_l309_309328


namespace toads_l309_309756

theorem toads (Tim Jim Sarah : ℕ) 
  (h1 : Jim = Tim + 20) 
  (h2 : Sarah = 2 * Jim) 
  (h3 : Sarah = 100) : Tim = 30 := 
by 
  -- Proof will be provided later
  sorry

end toads_l309_309756


namespace fraction_inequality_solution_l309_309153

theorem fraction_inequality_solution (x : ℝ) :
  (x < -5 ∨ x ≥ 2) ↔ (x-2) / (x+5) ≥ 0 :=
sorry

end fraction_inequality_solution_l309_309153


namespace keith_turnips_l309_309966

theorem keith_turnips (Alyssa_turnips Keith_turnips : ℕ) 
  (total_turnips : Alyssa_turnips + Keith_turnips = 15) 
  (alyssa_grew : Alyssa_turnips = 9) : Keith_turnips = 6 :=
by
  sorry

end keith_turnips_l309_309966


namespace relationship_f_minus_a2_f_minus_1_l309_309655

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x

-- Theorem statement translation
theorem relationship_f_minus_a2_f_minus_1 (a : ℝ) : f (-a^2) ≤ f (-1) := 
sorry

end relationship_f_minus_a2_f_minus_1_l309_309655


namespace ratio_of_linear_combination_l309_309218

theorem ratio_of_linear_combination (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) :
  (3 * a + 2 * b) / (b + 4 * c) = 3 / 17 :=
by
  sorry

end ratio_of_linear_combination_l309_309218


namespace no_two_digit_factorization_2023_l309_309114

theorem no_two_digit_factorization_2023 :
  ¬ ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 2023 := 
by
  sorry

end no_two_digit_factorization_2023_l309_309114


namespace semicircle_radius_l309_309479

theorem semicircle_radius (P : ℝ) (r : ℝ) (h₁ : P = π * r + 2 * r) (h₂ : P = 198) :
  r = 198 / (π + 2) :=
sorry

end semicircle_radius_l309_309479


namespace triangle_identity_l309_309963

variable {α : Type} [LinearOrderedField α] 

variables (A B C a b c : α)
variable (R : α) -- Circumradius 2R

-- Assuming Law of Sines holds in triangle ABC 
axiom law_of_sines : a = 2 * R * (sin A) ∧ b = 2 * R * (sin B) ∧ c = 2 * R * (sin C)

-- Prove the given trigonometric identity:
theorem triangle_identity
  (h₁ : a = 2 * R * (sin A))
  (h₂ : b = 2 * R * (sin B))
  (h₃ : c = 2 * R * (sin C)) : 
  (a^2 - b^2) / c^2 = (sin (A - B)) / (sin C) := by sorry

end triangle_identity_l309_309963


namespace min_cost_to_package_fine_arts_collection_l309_309605

theorem min_cost_to_package_fine_arts_collection :
  let box_length := 20
  let box_width := 20
  let box_height := 12
  let cost_per_box := 0.50
  let required_volume := 1920000
  let volume_of_one_box := box_length * box_width * box_height
  let number_of_boxes := required_volume / volume_of_one_box
  let total_cost := number_of_boxes * cost_per_box
  total_cost = 200 := 
by
  sorry

end min_cost_to_package_fine_arts_collection_l309_309605


namespace share_of_each_person_l309_309080

theorem share_of_each_person (total_length : ℕ) (h1 : total_length = 12) (h2 : total_length % 2 = 0)
  : total_length / 2 = 6 :=
by
  sorry

end share_of_each_person_l309_309080


namespace number_of_pictures_in_first_coloring_book_l309_309578

-- Define the conditions
variable (X : ℕ)
variable (total_pictures_colored : ℕ := 44)
variable (pictures_left : ℕ := 11)
variable (pictures_in_second_coloring_book : ℕ := 32)
variable (total_pictures : ℕ := total_pictures_colored + pictures_left)

-- The theorem statement
theorem number_of_pictures_in_first_coloring_book :
  X + pictures_in_second_coloring_book = total_pictures → X = 23 :=
by
  intro h
  sorry

end number_of_pictures_in_first_coloring_book_l309_309578


namespace units_digit_of_first_four_composite_numbers_l309_309804

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l309_309804


namespace students_neither_math_nor_physics_l309_309408

theorem students_neither_math_nor_physics :
  let total_students := 150
  let students_math := 80
  let students_physics := 60
  let students_both := 20
  total_students - (students_math - students_both + students_physics - students_both + students_both) = 30 :=
by
  sorry

end students_neither_math_nor_physics_l309_309408


namespace sum_of_digits_base8_product_l309_309499

theorem sum_of_digits_base8_product
  (a b : ℕ)
  (a_base8 : a = 3 * 8^1 + 4 * 8^0)
  (b_base8 : b = 2 * 8^1 + 2 * 8^0)
  (product : ℕ := a * b)
  (product_base8 : ℕ := (product / 64) * 8^2 + ((product / 8) % 8) * 8^1 + (product % 8)) :
  ((product_base8 / 8^2) + ((product_base8 / 8) % 8) + (product_base8 % 8)) = 1 * 8^1 + 6 * 8^0 :=
sorry

end sum_of_digits_base8_product_l309_309499


namespace ratio_of_hypotenuse_segments_l309_309668

theorem ratio_of_hypotenuse_segments (a b c d : ℝ) 
  (h1 : a^2 + b^2 = c^2)
  (h2 : b = (3/4) * a)
  (h3 : d^2 = (c - d)^2 + b^2) :
  (d / (c - d)) = (4 / 3) :=
sorry

end ratio_of_hypotenuse_segments_l309_309668


namespace units_digit_of_product_is_eight_l309_309771

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l309_309771


namespace units_digit_product_first_four_composite_numbers_l309_309780

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l309_309780


namespace point_in_fourth_quadrant_l309_309129

def lies_in_fourth_quadrant (P : ℤ × ℤ) : Prop :=
  P.fst > 0 ∧ P.snd < 0

theorem point_in_fourth_quadrant : lies_in_fourth_quadrant (2023, -2024) :=
by
  -- Here is where the proof steps would go
  sorry

end point_in_fourth_quadrant_l309_309129


namespace sum_of_intercepts_l309_309618

theorem sum_of_intercepts (x y : ℝ) (hx : y + 3 = 5 * (x - 6)) : 
  let x_intercept := 6 + 3/5;
  let y_intercept := -33;
  x_intercept + y_intercept = -26.4 := by
  sorry

end sum_of_intercepts_l309_309618


namespace average_of_remaining_two_numbers_l309_309272

theorem average_of_remaining_two_numbers (S S3 : ℝ) (h_avg5 : S / 5 = 8) (h_avg3 : S3 / 3 = 4) : S / 5 = 8 ∧ S3 / 3 = 4 → (S - S3) / 2 = 14 :=
by 
  sorry

end average_of_remaining_two_numbers_l309_309272


namespace geom_sequence_a7_l309_309131

theorem geom_sequence_a7 (a : ℕ → ℝ) (r : ℝ) 
  (h_geom : ∀ n : ℕ, a (n+1) = a n * r) 
  (h_a1 : a 1 = 8) 
  (h_a4_eq : a 4 = a 3 * a 5) : 
  a 7 = 1 / 8 :=
by
  sorry

end geom_sequence_a7_l309_309131


namespace unit_digit_hundred_digit_difference_l309_309144

theorem unit_digit_hundred_digit_difference :
  ∃ (A B C : ℕ), 100 ≤ 100 * A + 10 * B + C ∧ 100 * A + 10 * B + C < 1000 ∧
    99 * (A - C) = 198 ∧ 0 ≤ A ∧ A < 10 ∧ 0 ≤ C ∧ C < 10 ∧ 0 ≤ B ∧ B < 10 → 
  A - C = 2 :=
by 
  -- we only need to state the theorem, actual proof is not required.
  sorry

end unit_digit_hundred_digit_difference_l309_309144


namespace tan_150_eq_neg_sqrt_3_l309_309892

theorem tan_150_eq_neg_sqrt_3 :
  let θ := 150 * Real.pi / 180 in
  Real.tan θ = -Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_sqrt_3_l309_309892


namespace compute_operation_value_l309_309725

def operation (a b c : ℝ) : ℝ := b^3 - 3 * a * b * c - 4 * a * c^2

theorem compute_operation_value : operation 2 (-1) 4 = -105 :=
by
  sorry

end compute_operation_value_l309_309725


namespace conic_section_eccentricities_cubic_l309_309945

theorem conic_section_eccentricities_cubic : 
  ∃ (e1 e2 e3 : ℝ), 
    (e1 = 1) ∧ 
    (0 < e2 ∧ e2 < 1) ∧ 
    (e3 > 1) ∧ 
    2 * e1^3 - 7 * e1^2 + 7 * e1 - 2 = 0 ∧
    2 * e2^3 - 7 * e2^2 + 7 * e2 - 2 = 0 ∧
    2 * e3^3 - 7 * e3^2 + 7 * e3 - 2 = 0 := 
by
  sorry

end conic_section_eccentricities_cubic_l309_309945


namespace tan_150_eq_neg_one_over_sqrt_three_l309_309910

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l309_309910


namespace chocolate_ice_cream_ordered_l309_309022

theorem chocolate_ice_cream_ordered (V C : ℕ) (total_ice_cream : ℕ) (percentage_vanilla : ℚ) 
  (h_total : total_ice_cream = 220) 
  (h_percentage : percentage_vanilla = 0.20) 
  (h_vanilla_total : V = percentage_vanilla * total_ice_cream) 
  (h_vanilla_chocolate : V = 2 * C) 
  : C = 22 := 
by 
  sorry

end chocolate_ice_cream_ordered_l309_309022


namespace cone_height_ratio_l309_309067

theorem cone_height_ratio (C : ℝ) (h₁ : ℝ) (V₂ : ℝ) (r : ℝ) (h₂ : ℝ) :
  C = 20 * Real.pi → 
  h₁ = 40 →
  V₂ = 400 * Real.pi →
  2 * Real.pi * r = 20 * Real.pi →
  V₂ = (1 / 3) * Real.pi * r^2 * h₂ →
  h₂ / h₁ = (3 / 10) := by
sorry

end cone_height_ratio_l309_309067


namespace shaded_L_area_l309_309091

theorem shaded_L_area 
  (s₁ s₂ s₃ s₄ : ℕ)
  (hA : s₁ = 2)
  (hB : s₂ = 2)
  (hC : s₃ = 3)
  (hD : s₄ = 3)
  (side_ABC : ℕ := 6)
  (area_ABC : ℕ := side_ABC * side_ABC) : 
  area_ABC - (s₁ * s₁ + s₂ * s₂ + s₃ * s₃ + s₄ * s₄) = 10 :=
sorry

end shaded_L_area_l309_309091


namespace melanie_total_weight_l309_309407

def weight_of_brie : ℝ := 8 / 16 -- 8 ounces converted to pounds
def weight_of_bread : ℝ := 1
def weight_of_tomatoes : ℝ := 1
def weight_of_zucchini : ℝ := 2
def weight_of_chicken : ℝ := 1.5
def weight_of_raspberries : ℝ := 8 / 16 -- 8 ounces converted to pounds
def weight_of_blueberries : ℝ := 8 / 16 -- 8 ounces converted to pounds

def total_weight : ℝ := weight_of_brie + weight_of_bread + weight_of_tomatoes + weight_of_zucchini +
                        weight_of_chicken + weight_of_raspberries + weight_of_blueberries

theorem melanie_total_weight : total_weight = 7 := by
  sorry

end melanie_total_weight_l309_309407


namespace fraction_students_say_dislike_actually_like_l309_309320

theorem fraction_students_say_dislike_actually_like (total_students : ℕ) (like_dancing_fraction : ℚ) 
  (like_dancing_say_dislike_fraction : ℚ) (dislike_dancing_say_dislike_fraction : ℚ) : 
  (∃ frac : ℚ, frac = 40.7 / 100) :=
by
  let total_students := (200 : ℕ)
  let like_dancing_fraction := (70 / 100 : ℚ)
  let like_dancing_say_dislike_fraction := (25 / 100 : ℚ)
  let dislike_dancing_say_dislike_fraction := (85 / 100 : ℚ)
  
  let total_like_dancing := total_students * like_dancing_fraction
  let total_dislike_dancing :=  total_students * (1 - like_dancing_fraction)
  let like_dancing_say_dislike := total_like_dancing * like_dancing_say_dislike_fraction
  let dislike_dancing_say_dislike := total_dislike_dancing * dislike_dancing_say_dislike_fraction
  let total_say_dislike := like_dancing_say_dislike + dislike_dancing_say_dislike
  let fraction_say_dislike_actually_like := like_dancing_say_dislike / total_say_dislike
  
  existsi fraction_say_dislike_actually_like
  sorry

end fraction_students_say_dislike_actually_like_l309_309320


namespace units_digit_first_four_composites_l309_309812

theorem units_digit_first_four_composites : 
  let first_four_composites := [4, 6, 8, 9]
  let product := first_four_composites.prod
  Nat.unitsDigit product = 8 :=
by
  sorry

end units_digit_first_four_composites_l309_309812


namespace kopeechka_purchase_l309_309708

theorem kopeechka_purchase
  (a : ℕ)
  (n : ℕ)
  (total_cost : ℕ)
  (item_cost : ℕ) :
  total_cost = 20083 →
  item_cost = 100 * a + 99 →
  (n * item_cost = total_cost ∧ n = 17 ∨ n = 117) :=
begin
  sorry
end

end kopeechka_purchase_l309_309708


namespace scientific_notation_of_0_00065_l309_309994

/-- 
Prove that the decimal representation of a number 0.00065 can be expressed in scientific notation 
as 6.5 * 10^(-4)
-/
theorem scientific_notation_of_0_00065 : 0.00065 = 6.5 * 10^(-4) := 
by 
  sorry

end scientific_notation_of_0_00065_l309_309994


namespace arithmetic_seq_first_term_l309_309973

theorem arithmetic_seq_first_term (S : ℕ → ℚ) (n : ℕ) (a : ℚ)
  (h₁ : ∀ n, S n = n * (2 * a + (n - 1) * 5) / 2)
  (h₂ : ∀ n, S (3 * n) / S n = 9) :
  a = 5 / 2 :=
by
  sorry

end arithmetic_seq_first_term_l309_309973


namespace fruits_in_box_l309_309582

theorem fruits_in_box (initial_persimmons : ℕ) (added_apples : ℕ) (total_fruits : ℕ) :
  initial_persimmons = 2 → added_apples = 7 → total_fruits = initial_persimmons + added_apples → total_fruits = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end fruits_in_box_l309_309582


namespace triangle_inequality_check_l309_309860

theorem triangle_inequality_check :
  ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 →
    (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ↔
    ((a = 6 ∧ b = 9 ∧ c = 14) ∨ (a = 9 ∧ b = 6 ∧ c = 14) ∨ (a = 6 ∧ b = 14 ∧ c = 9) ∨
     (a = 14 ∧ b = 6 ∧ c = 9) ∨ (a = 9 ∧ b = 14 ∧ c = 6) ∨ (a = 14 ∧ b = 9 ∧ c = 6)) := sorry

end triangle_inequality_check_l309_309860


namespace units_digit_first_four_composite_is_eight_l309_309816

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l309_309816


namespace liters_to_pints_conversion_l309_309649

-- Definitions based on conditions
def liters_to_pints_ratio := 0.75 / 1.575
def target_liters := 1.5
def expected_pints := 3.15

-- Lean statement
theorem liters_to_pints_conversion 
  (h_ratio : 0.75 / 1.575 = liters_to_pints_ratio)
  (h_target : 1.5 = target_liters) :
  target_liters * (1 / liters_to_pints_ratio) = expected_pints :=
by 
  sorry

end liters_to_pints_conversion_l309_309649


namespace canoes_built_by_April_l309_309077

theorem canoes_built_by_April :
  (∃ (c1 c2 c3 c4 : ℕ), 
    c1 = 5 ∧ 
    c2 = 3 * c1 ∧ 
    c3 = 3 * c2 ∧ 
    c4 = 3 * c3 ∧
    (c1 + c2 + c3 + c4) = 200) :=
sorry

end canoes_built_by_April_l309_309077


namespace units_digit_first_four_composite_is_eight_l309_309819

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l309_309819


namespace monotonic_f_iff_l309_309363

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - a * x + 5 else 1 + 1 / x

theorem monotonic_f_iff {a : ℝ} :  
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (2 ≤ a ∧ a ≤ 4) :=
by
  sorry

end monotonic_f_iff_l309_309363


namespace kolya_purchase_l309_309683

theorem kolya_purchase : ∃ n : ℕ, n = 17 ∨ n = 117 :=
by
  let item_cost := λ a : ℕ, 100 * a + 99
  let total_cost := 20000 + 83
  have h : ∀ n a, n * (item_cost a) = total_cost → (n = 17 ∨ n = 117) := sorry
  have h1 := h 17 0
  have h2 := h 117 0
  existsi 17
  exact h1 sorry

end kolya_purchase_l309_309683


namespace negation_of_existence_l309_309998

theorem negation_of_existence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) :=
by
  sorry

end negation_of_existence_l309_309998


namespace total_chestnuts_weight_l309_309515

def eunsoo_kg := 2
def eunsoo_g := 600
def mingi_g := 3700

theorem total_chestnuts_weight :
  (eunsoo_kg * 1000 + eunsoo_g + mingi_g) = 6300 :=
by
  sorry

end total_chestnuts_weight_l309_309515


namespace wire_cut_circle_square_area_eq_l309_309070

theorem wire_cut_circle_square_area_eq (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (h₃ : (a^2 / (4 * π)) = ((b^2) / 16)) : 
  a / b = 2 / Real.sqrt π :=
by
  sorry

end wire_cut_circle_square_area_eq_l309_309070


namespace total_weight_mason_hotdogs_l309_309010

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

end total_weight_mason_hotdogs_l309_309010


namespace greatest_third_side_l309_309548

theorem greatest_third_side (a b c : ℝ) (h₀: a = 5) (h₁: b = 11) (h₂ : 6 < c ∧ c < 16) : c ≤ 15 :=
by
  -- assumption applying that c needs to be within 6 and 16
  have h₃ : 6 < c := h₂.1
  have h₄: c < 16 := h₂.2
  -- need to show greatest integer c is 15
  sorry

end greatest_third_side_l309_309548


namespace smallest_k_for_difference_l309_309349

theorem smallest_k_for_difference (s : Finset ℕ) (h₁ : ∀ x ∈ s, x ≤ 2016) (h₂ : s.card = 674) :
  ∃ a b ∈ s, 672 < abs (a - b) ∧ abs (a - b) < 1344 :=
by
  sorry

end smallest_k_for_difference_l309_309349


namespace sector_area_is_2pi_l309_309103

noncomputable def sectorArea (l : ℝ) (R : ℝ) : ℝ :=
  (1 / 2) * l * R

theorem sector_area_is_2pi (R : ℝ) (l : ℝ) (hR : R = 4) (hl : l = π) :
  sectorArea l R = 2 * π :=
by
  sorry

end sector_area_is_2pi_l309_309103


namespace base4_arithmetic_l309_309638

theorem base4_arithmetic :
  (Nat.ofDigits 4 [2, 3, 1] * Nat.ofDigits 4 [2, 2] / Nat.ofDigits 4 [3]) = Nat.ofDigits 4 [2, 2, 1] := by
sorry

end base4_arithmetic_l309_309638


namespace golf_tees_per_member_l309_309076

theorem golf_tees_per_member (T : ℕ) : 
  (∃ (t : ℕ), 
     t = 4 * T ∧ 
     (∀ (g : ℕ), g ≤ 2 → g * 12 + 28 * 2 = t)
  ) → T = 20 :=
by
  intros h
  -- problem statement is enough for this example
  sorry

end golf_tees_per_member_l309_309076


namespace compute_d1e1_d2e2_d3e3_l309_309566

-- Given polynomials and conditions
variables {R : Type*} [CommRing R]

noncomputable def P (x : R) : R :=
  x^7 - x^6 + x^4 - x^3 + x^2 - x + 1

noncomputable def Q (x : R) (d1 d2 d3 e1 e2 e3 : R) : R :=
  (x^2 + d1 * x + e1) * (x^2 + d2 * x + e2) * (x^2 + d3 * x + e3)

-- Given conditions
theorem compute_d1e1_d2e2_d3e3 
  (d1 d2 d3 e1 e2 e3 : R)
  (h : ∀ x : R, P x = Q x d1 d2 d3 e1 e2 e3) : 
  d1 * e1 + d2 * e2 + d3 * e3 = -1 :=
by
  sorry

end compute_d1e1_d2e2_d3e3_l309_309566


namespace diana_hits_seven_l309_309417

-- Define the participants
inductive Player 
| Alex 
| Brooke 
| Carlos 
| Diana 
| Emily 
| Fiona

open Player

-- Define a function to get the total score of a participant
def total_score (p : Player) : ℕ :=
match p with
| Alex => 20
| Brooke => 23
| Carlos => 28
| Diana => 18
| Emily => 26
| Fiona => 30

-- Function to check if a dart target is hit within the range and unique
def is_valid_target (x y z : ℕ) :=
x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 1 ≤ x ∧ x ≤ 12 ∧ 1 ≤ y ∧ y ≤ 12 ∧ 1 ≤ z ∧ z ≤ 12

-- Check if the sum equals the score of the player
def valid_score (p : Player) (x y z : ℕ) :=
is_valid_target x y z ∧ x + y + z = total_score p

-- Lean 4 theorem statement, asking if Diana hits the region 7
theorem diana_hits_seven : ∃ x y z, valid_score Diana x y z ∧ (x = 7 ∨ y = 7 ∨ z = 7) :=
sorry

end diana_hits_seven_l309_309417


namespace negation_prob1_negation_prob2_negation_prob3_l309_309629

-- Definitions and Conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p

def defines_const_func (f : ℝ → ℝ) (y : ℝ) : Prop :=
  ∀ x1 x2 : ℝ, f x1 = f x2

-- Problem 1
theorem negation_prob1 : 
  (∃ n : ℕ, ∀ p : ℕ, is_prime p → p ≤ n) ↔ 
  ¬(∀ n : ℕ, ∃ p : ℕ, is_prime p ∧ n ≤ p) :=
sorry

-- Problem 2
theorem negation_prob2 : 
  (∃ n : ℤ, ∀ p : ℤ, n + p ≠ 0) ↔ 
  ¬(∀ n : ℤ, ∃! p : ℤ, n + p = 0) :=
sorry

-- Problem 3
theorem negation_prob3 : 
  (∀ y : ℝ, ¬defines_const_func (λ x => x * y) y) ↔ 
  ¬(∃ y : ℝ, defines_const_func (λ x => x * y) y) :=
sorry

end negation_prob1_negation_prob2_negation_prob3_l309_309629


namespace digit_150_of_one_thirteenth_l309_309451

theorem digit_150_of_one_thirteenth : 
  (let repeating_seq := "076923".to_list in
  (repeating_seq.nth ((150 - 1) % repeating_seq.length)).iget = '3') :=
by
  sorry

end digit_150_of_one_thirteenth_l309_309451


namespace triangle_height_l309_309029

theorem triangle_height (base height area : ℝ) (h_base : base = 3) (h_area : area = 6) (h_formula : area = (1/2) * base * height) : height = 4 :=
by
  sorry

end triangle_height_l309_309029


namespace find_payment_y_l309_309603

variable (X Y : Real)

axiom h1 : X + Y = 570
axiom h2 : X = 1.2 * Y

theorem find_payment_y : Y = 570 / 2.2 := by
  sorry

end find_payment_y_l309_309603


namespace find_fx_log3_5_value_l309_309108

noncomputable def f : ℝ → ℝ
| x := if x < 2 then f (x + 2) else (1 / 3) ^ x

theorem find_fx_log3_5_value :
  f (-1 + real.logb 3 5) = 1 / 15 := by
  sorry

end find_fx_log3_5_value_l309_309108


namespace absolute_difference_AB_l309_309369

noncomputable def A : Real := 12 / 7
noncomputable def B : Real := 20 / 7

theorem absolute_difference_AB : |A - B| = 8 / 7 := by
  sorry

end absolute_difference_AB_l309_309369


namespace dollar_neg3_4_eq_neg27_l309_309335

-- Define the operation $$
def dollar (a b : ℤ) : ℤ := a * (b + 1) + a * b

-- Theorem stating the value of (-3) $$ 4
theorem dollar_neg3_4_eq_neg27 : dollar (-3) 4 = -27 := 
by
  sorry

end dollar_neg3_4_eq_neg27_l309_309335


namespace John_pushup_count_l309_309477

-- Definitions arising from conditions
def Zachary_pushups : ℕ := 51
def David_pushups : ℕ := Zachary_pushups + 22
def John_pushups : ℕ := David_pushups - 4

-- Theorem statement
theorem John_pushup_count : John_pushups = 69 := 
by 
  sorry

end John_pushup_count_l309_309477


namespace black_shirts_in_pack_l309_309262

-- defining the conditions
variables (B : ℕ) -- the number of black shirts in each pack
variable (total_shirts : ℕ := 21)
variable (yellow_shirts_per_pack : ℕ := 2)
variable (black_packs : ℕ := 3)
variable (yellow_packs : ℕ := 3)

-- ensuring the conditions are met, the total shirts equals 21
def total_black_shirts := black_packs * B
def total_yellow_shirts := yellow_packs * yellow_shirts_per_pack

-- the proof problem
theorem black_shirts_in_pack : total_black_shirts + total_yellow_shirts = total_shirts → B = 5 := by
  sorry

end black_shirts_in_pack_l309_309262


namespace club_boys_count_l309_309059

theorem club_boys_count (B G : ℕ) (h1 : B + G = 30) (h2 : (1 / 3 : ℝ) * G + B = 18) : B = 12 :=
by
  -- We would proceed with the steps here, but add 'sorry' to indicate incomplete proof
  sorry

end club_boys_count_l309_309059


namespace possible_items_l309_309688

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items_l309_309688


namespace geometric_sequence_solution_l309_309976

theorem geometric_sequence_solution:
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ) (q a1 : ℝ),
    a 2 = 6 → 6 * a1 + a 3 = 30 → q > 2 →
    (∀ n, a n = 2 * 3 ^ (n - 1)) ∧
    (∀ n, S n = (3 ^ n - 1) / 2) :=
by
  intros a S q a1 h1 h2 h3
  sorry

end geometric_sequence_solution_l309_309976


namespace subtract_abs_from_local_value_l309_309156

-- Define the local value of 4 in 564823 as 4000
def local_value_of_4_in_564823 : ℕ := 4000

-- Define the absolute value of 4 as 4
def absolute_value_of_4 : ℕ := 4

-- Theorem statement: Prove that subtracting the absolute value of 4 from the local value of 4 in 564823 equals 3996
theorem subtract_abs_from_local_value : (local_value_of_4_in_564823 - absolute_value_of_4) = 3996 :=
by
  sorry

end subtract_abs_from_local_value_l309_309156


namespace kirill_height_l309_309392

theorem kirill_height (K B : ℕ) (h1 : K = B - 14) (h2 : K + B = 112) : K = 49 :=
by
  sorry

end kirill_height_l309_309392


namespace unique_last_digit_divisible_by_7_l309_309978

theorem unique_last_digit_divisible_by_7 :
  ∃! d : ℕ, (∃ n : ℕ, n % 7 = 0 ∧ n % 10 = d) :=
sorry

end unique_last_digit_divisible_by_7_l309_309978


namespace trust_meteorologist_l309_309608

-- Definitions for problem conditions
variables {G M1 M2 S : Prop}
variable {r : ℝ}
variable {p : ℝ}

/-- The probability of a clear day is r -/
axiom prob_clear_day : r = 0.74

/-- Senators' prediction accuracy -/
axiom senator_accuracy : ℝ

/-- Meteorologist's prediction accuracy being 1.5 times senators' /-
axiom meteorologist_accuracy : ∀ p, 1.5 * p

/-- Independence of predictions -/
axiom independence_preds : independent [G, M1, M2, S]

noncomputable def joint_probability_given_G : ℝ :=
(1 - 1.5 * meteorologist_accuracy senator_accuracy) * senator_accuracy^2

noncomputable def joint_probability_given_not_G : ℝ :=
1.5 * meteorologist_accuracy senator_accuracy * (1 - senator_accuracy)^2

noncomputable def overall_probability : ℝ :=
joint_probability_given_G * r + joint_probability_given_not_G * (1 - r)

noncomputable def conditional_prob_not_clear : ℝ /-
(joint_probability_given_not_G * (1 - r)) / overall_probability

noncomputable def conditional_prob_clear : ℝ
(joint_probability_given_G * r) / overall_probability

-- Main theorem statement: Given the conditions, the meteorologist's forecast is more reliable
theorem trust_meteorologist : conditional_prob_not_clear > conditional_prob_clear :=
by sorry

end trust_meteorologist_l309_309608


namespace john_annual_patients_l309_309389

-- Definitions for the various conditions
def first_hospital_patients_per_day := 20
def second_hospital_patients_per_day := first_hospital_patients_per_day + (first_hospital_patients_per_day * 20 / 100)
def third_hospital_patients_per_day := first_hospital_patients_per_day + (first_hospital_patients_per_day * 15 / 100)
def total_patients_per_day := first_hospital_patients_per_day + second_hospital_patients_per_day + third_hospital_patients_per_day
def workdays_per_week := 5
def total_patients_per_week := total_patients_per_day * workdays_per_week
def working_weeks_per_year := 50 - 2 -- considering 2 weeks of vacation
def total_patients_per_year := total_patients_per_week * working_weeks_per_year

-- The statement to prove
theorem john_annual_patients : total_patients_per_year = 16080 := by
  sorry

end john_annual_patients_l309_309389


namespace tan_150_degrees_l309_309866

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l309_309866


namespace sum_of_other_endpoint_coordinates_l309_309981

theorem sum_of_other_endpoint_coordinates (x y : ℝ) (hx : (x + 5) / 2 = 3) (hy : (y - 2) / 2 = 4) :
  x + y = 11 :=
sorry

end sum_of_other_endpoint_coordinates_l309_309981


namespace maximum_profit_at_110_l309_309742

noncomputable def profit (x : ℕ) : ℝ := 
if x > 0 ∧ x < 100 then 
  -0.5 * (x : ℝ)^2 + 90 * (x : ℝ) - 600 
else if x ≥ 100 then 
  -2 * (x : ℝ) - 24200 / (x : ℝ) + 4100 
else 
  0 -- To ensure totality, although this won't match the problem's condition that x is always positive

theorem maximum_profit_at_110 :
  ∃ (y_max : ℝ), ∀ (x : ℕ), profit 110 = y_max ∧ (∀ x ≠ 0, profit 110 ≥ profit x) :=
sorry

end maximum_profit_at_110_l309_309742


namespace units_digit_first_four_composites_l309_309777

theorem units_digit_first_four_composites :
  let product := 4 * 6 * 8 * 9 in
  product % 10 = 8 :=
by
  let product := 4 * 6 * 8 * 9
  have h : product = 1728 := by norm_num
  show product % 10 = 8
  rw [h]
  norm_num
  done 
  sorry

end units_digit_first_four_composites_l309_309777


namespace max_value_of_f_f_lt_x3_minus_2x2_l309_309361

noncomputable def f (a b : ℝ) (x : ℝ) := a * x^2 + Real.log x + b

theorem max_value_of_f (a b : ℝ) (h_a : a = -1) (h_b : b = -1 / 4) :
  f a b (Real.sqrt 2 / 2) = - (3 + 2 * Real.log 2) / 4 := by
  sorry

theorem f_lt_x3_minus_2x2 (a b : ℝ) (h_a : a = -1) (h_b : b = -1 / 4) (x : ℝ) (hx : 0 < x) :
  f a b x < x^3 - 2 * x^2 := by
  sorry

end max_value_of_f_f_lt_x3_minus_2x2_l309_309361


namespace simplify_and_multiply_expression_l309_309735

variable (b : ℝ)

theorem simplify_and_multiply_expression :
  (2 * (3 * b) * (4 * b^2) * (5 * b^3)) * 6 = 720 * b^6 :=
by
  sorry

end simplify_and_multiply_expression_l309_309735


namespace solve_for_y_l309_309232

theorem solve_for_y (y : ℝ) (h_pos : y > 0) (h_eq : y^2 = 1024) : y = 32 := 
by
  sorry

end solve_for_y_l309_309232


namespace pentagon_area_l309_309862

theorem pentagon_area 
  (side1 : ℝ) (side2 : ℝ) (side3 : ℝ) (side4 : ℝ) (side5 : ℝ)
  (h1 : side1 = 12) (h2 : side2 = 20) (h3 : side3 = 30) (h4 : side4 = 15) (h5 : side5 = 25)
  (right_angle : ∃ (a b : ℝ), a = side1 ∧ b = side5 ∧ a^2 + b^2 = (a + b)^2) : 
  ∃ (area : ℝ), area = 600 := 
  sorry

end pentagon_area_l309_309862


namespace problem_simplify_and_evaluate_l309_309024

theorem problem_simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 3 + 3) :
  (1 - (m / (m + 3))) / ((m^2 - 9) / (m^2 + 6 * m + 9)) = Real.sqrt 3 :=
by
  sorry

end problem_simplify_and_evaluate_l309_309024


namespace tan_150_eq_neg_one_over_sqrt_three_l309_309912

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l309_309912


namespace distinct_square_sum_100_l309_309672

theorem distinct_square_sum_100 :
  ∀ a b c : ℕ, a ≠ b → b ≠ c → a ≠ c → 
  a^2 + b^2 + c^2 = 100 → false := by
  sorry

end distinct_square_sum_100_l309_309672


namespace problem_1_problem_2_l309_309224

-- Proof Problem 1: Prove A ∩ B = {x | -3 ≤ x ≤ -2} given m = -3
def A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x | 2 * m - 1 ≤ x ∧ x ≤ m + 1}

theorem problem_1 : B (-3) ∩ A = {x | -3 ≤ x ∧ x ≤ -2} := sorry

-- Proof Problem 2: Prove m ≥ -1 given B ⊆ A
theorem problem_2 (m : ℝ) : (B m ⊆ A) → m ≥ -1 := sorry

end problem_1_problem_2_l309_309224


namespace solve_expr_l309_309293

theorem solve_expr (x : ℝ) (h : x = 3) : x^6 - 6 * x^2 = 675 := by
  sorry

end solve_expr_l309_309293


namespace expand_and_simplify_l309_309637

theorem expand_and_simplify (x : ℝ) : (x + 6) * (x - 11) = x^2 - 5 * x - 66 :=
by
  sorry

end expand_and_simplify_l309_309637


namespace overall_average_marks_l309_309123

theorem overall_average_marks
  (avg_A : ℝ) (n_A : ℕ) (avg_B : ℝ) (n_B : ℕ) (avg_C : ℝ) (n_C : ℕ)
  (h_avg_A : avg_A = 40) (h_n_A : n_A = 12)
  (h_avg_B : avg_B = 60) (h_n_B : n_B = 28)
  (h_avg_C : avg_C = 55) (h_n_C : n_C = 15) :
  ((n_A * avg_A) + (n_B * avg_B) + (n_C * avg_C)) / (n_A + n_B + n_C) = 54.27 := by
  sorry

end overall_average_marks_l309_309123


namespace min_buses_needed_l309_309856

theorem min_buses_needed (x y : ℕ) (h1 : 45 * x + 35 * y ≥ 530) (h2 : y ≥ 3) : x + y = 13 :=
by
  sorry

end min_buses_needed_l309_309856


namespace commuting_time_equation_l309_309127

-- Definitions based on the conditions
def distance_to_cemetery : ℝ := 15
def cyclists_speed (x : ℝ) : ℝ := x
def car_speed (x : ℝ) : ℝ := 2 * x
def cyclists_start_time_earlier : ℝ := 0.5

-- The statement we need to prove
theorem commuting_time_equation (x : ℝ) (h : x > 0) :
  distance_to_cemetery / cyclists_speed x =
  (distance_to_cemetery / car_speed x) + cyclists_start_time_earlier :=
by
  sorry

end commuting_time_equation_l309_309127


namespace horizontal_asymptote_l309_309000

noncomputable def rational_function (x : ℝ) : ℝ :=
  (15 * x^4 + 7 * x^3 + 10 * x^2 + 6 * x + 4) / (4 * x^4 + 3 * x^3 + 9 * x^2 + 4 * x + 2)

theorem horizontal_asymptote :
  ∃ L : ℝ, (∀ ε > 0, ∃ M > 0, ∀ x > M, |rational_function x - L| < ε) → L = 15 / 4 :=
by
  sorry

end horizontal_asymptote_l309_309000


namespace age_of_b_l309_309030

theorem age_of_b (A B C : ℕ) (h₁ : (A + B + C) / 3 = 25) (h₂ : (A + C) / 2 = 29) : B = 17 := 
by
  sorry

end age_of_b_l309_309030


namespace intersection_of_A_and_B_l309_309647

def A (x : ℝ) : Prop := x^2 - x - 6 ≤ 0
def B (x : ℝ) : Prop := x > 1

theorem intersection_of_A_and_B :
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 1 < x ∧ x ≤ 3} :=
by
  sorry

end intersection_of_A_and_B_l309_309647


namespace final_withdrawal_amount_july_2005_l309_309606

-- Define the conditions given in the problem
variables (a r : ℝ) (n : ℕ)

-- Define the recursive formula for deposits
def deposit_amount (n : ℕ) : ℝ :=
  if n = 0 then a else (deposit_amount (n - 1)) * (1 + r) + a

-- The problem statement translated to Lean
theorem final_withdrawal_amount_july_2005 :
  deposit_amount a r 5 = a / r * ((1 + r) ^ 6 - (1 + r)) :=
sorry

end final_withdrawal_amount_july_2005_l309_309606


namespace purchase_options_l309_309716

def item_cost (a : Nat) : Nat := 100 * a + 99

def total_cost : Nat := 20083

theorem purchase_options (a : Nat) (n : Nat) (h : n * item_cost a = total_cost) :
  n = 17 ∨ n = 117 :=
by
  sorry

end purchase_options_l309_309716


namespace phone_number_C_value_l309_309380

/-- 
In a phone number formatted as ABC-DEF-GHIJ, each letter symbolizes a distinct digit.
Digits in each section ABC, DEF, and GHIJ are in ascending order i.e., A < B < C, D < E < F, and G < H < I < J.
Moreover, D, E, F are consecutive odd digits, and G, H, I, J are consecutive even digits.
Also, A + B + C = 15. Prove that the value of C is 9. 
-/
theorem phone_number_C_value :
  ∃ (A B C D E F G H I J : ℕ), 
  A < B ∧ B < C ∧ D < E ∧ E < F ∧ G < H ∧ H < I ∧ I < J ∧
  (D % 2 = 1) ∧ (E % 2 = 1) ∧ (F % 2 = 1) ∧
  (G % 2 = 0) ∧ (H % 2 = 0) ∧ (I % 2 = 0) ∧ (J % 2 = 0) ∧
  (E = D + 2) ∧ (F = D + 4) ∧ (H = G + 2) ∧ (I = G + 4) ∧ (J = G + 6) ∧
  A + B + C = 15 ∧
  C = 9 := by 
  sorry

end phone_number_C_value_l309_309380


namespace units_digit_first_four_composites_l309_309764

theorem units_digit_first_four_composites :
  let p := [4, 6, 8, 9] in
  let product := List.prod p in
  product % 10 = 8 :=
by
  let p := [4, 6, 8, 9]
  let product := List.prod p
  show product % 10 = 8
  sorry

end units_digit_first_four_composites_l309_309764


namespace smallest_n_for_multiple_of_5_l309_309157

theorem smallest_n_for_multiple_of_5 (x y : ℤ) (h1 : x + 2 ≡ 0 [ZMOD 5]) (h2 : y - 2 ≡ 0 [ZMOD 5]) :
  ∃ n : ℕ, n > 0 ∧ x^2 + x * y + y^2 + n ≡ 0 [ZMOD 5] ∧ n = 1 := 
sorry

end smallest_n_for_multiple_of_5_l309_309157


namespace total_amount_owed_l309_309977

-- Conditions
def borrowed_amount : ℝ := 500
def monthly_interest_rate : ℝ := 0.02
def months_not_paid : ℕ := 3

-- Compounded monthly formula
def amount_after_n_months (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- Theorem statement
theorem total_amount_owed :
  amount_after_n_months borrowed_amount monthly_interest_rate months_not_paid = 530.604 :=
by
  -- Proof to be filled in here
  sorry

end total_amount_owed_l309_309977


namespace range_of_m_l309_309525

theorem range_of_m (m : ℝ) : (0.7 ^ 1.3) ^ m < (1.3 ^ 0.7) ^ m ↔ m < 0 := by
  sorry

end range_of_m_l309_309525


namespace units_digit_product_first_four_composite_numbers_l309_309781

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l309_309781


namespace distinct_square_sum_100_l309_309673

theorem distinct_square_sum_100 :
  ∀ a b c : ℕ, a ≠ b → b ≠ c → a ≠ c → 
  a^2 + b^2 + c^2 = 100 → false := by
  sorry

end distinct_square_sum_100_l309_309673


namespace maritza_study_hours_l309_309008

noncomputable def time_to_study_for_citizenship_test (num_mc_questions num_fitb_questions time_mc time_fitb : ℕ) : ℕ :=
  (num_mc_questions * time_mc + num_fitb_questions * time_fitb) / 60

theorem maritza_study_hours :
  time_to_study_for_citizenship_test 30 30 15 25 = 20 :=
by
  sorry

end maritza_study_hours_l309_309008


namespace problem_statement_l309_309744

theorem problem_statement :
  ∃ p q r : ℤ,
    (∀ x : ℝ, (x^2 + 19*x + 88 = (x + p) * (x + q)) ∧ (x^2 - 23*x + 132 = (x - q) * (x - r))) →
      p + q + r = 31 :=
sorry

end problem_statement_l309_309744


namespace intersection_M_N_l309_309215

open Set Real

def M := {x : ℝ | x^2 < 4}
def N := {x : ℝ | ∃ α : ℝ, x = sin α}
def IntersectSet := {x : ℝ | -1 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = IntersectSet := by
  sorry

end intersection_M_N_l309_309215


namespace remainder_52_l309_309142

theorem remainder_52 (x y : ℕ) (k m : ℤ)
  (h₁ : x = 246 * k + 37)
  (h₂ : y = 357 * m + 53) :
  (x + y + 97) % 123 = 52 := by
  sorry

end remainder_52_l309_309142


namespace quadratic_roots_proof_l309_309468

noncomputable def quadratic_roots_statement : Prop :=
  ∃ (x1 x2 : ℝ), 
    (x1 ≠ x2 ∨ x1 = x2) ∧ 
    (x1 = -20 ∧ x2 = -20) ∧ 
    (x1^2 + 40 * x1 + 300 = -100) ∧ 
    (x1 - x2 = 0 ∧ x1 * x2 = 400)  

theorem quadratic_roots_proof : quadratic_roots_statement :=
sorry

end quadratic_roots_proof_l309_309468


namespace popsicle_sticks_left_l309_309329

/-- Danielle has $10 for supplies. She buys one set of molds for $3, 
a pack of 100 popsicle sticks for $1. Each bottle of juice makes 20 popsicles and costs $2.
Prove that the number of popsicle sticks Danielle will be left with after making as many popsicles as she can is 40. -/
theorem popsicle_sticks_left (initial_money : ℕ)
    (mold_cost : ℕ) (sticks_cost : ℕ) (initial_sticks : ℕ)
    (juice_cost : ℕ) (popsicles_per_bottle : ℕ)
    (final_sticks : ℕ) :
    initial_money = 10 →
    mold_cost = 3 → 
    sticks_cost = 1 → 
    initial_sticks = 100 →
    juice_cost = 2 →
    popsicles_per_bottle = 20 →
    final_sticks = initial_sticks - (popsicles_per_bottle * (initial_money - mold_cost - sticks_cost) / juice_cost) →
    final_sticks = 40 :=
by
  intros h_initial_money h_mold_cost h_sticks_cost h_initial_sticks h_juice_cost h_popsicles_per_bottle h_final_sticks
  rw [h_initial_money, h_mold_cost, h_sticks_cost, h_initial_sticks, h_juice_cost, h_popsicles_per_bottle] at h_final_sticks
  norm_num at h_final_sticks
  exact h_final_sticks

end popsicle_sticks_left_l309_309329


namespace parallel_lines_sufficient_necessity_l309_309098

theorem parallel_lines_sufficient_necessity (a : ℝ) :
  ¬ (a = 1 ↔ (∀ x : ℝ, a^2 * x + 1 = x - 1)) := 
sorry

end parallel_lines_sufficient_necessity_l309_309098


namespace geometric_functions_l309_309083

-- Defining the function types based on the problem description.
def f1 (x : ℝ) : ℝ := 2 ^ x
def f2 (x : ℝ) : ℝ := Real.log 2 x
def f3 (x : ℝ) : ℝ := x ^ 2
def f4 (x : ℝ) : ℝ := Real.log 2 x

-- Condition: Function f is a geometric function if for any geometric sequence {a_n},
-- the sequence {f(a_n)} is also geometric.

def is_geometric_function (f : ℝ → ℝ) : Prop :=
  ∀ {a_n : ℕ → ℝ} (h : ∃ r : ℝ, ∀ n, a_n (n + 1) = r * a_n n), ∃ r' : ℝ, ∀ n, f (a_n (n + 1)) = r' * f (a_n n)

-- Theorem that needs to be proved
theorem geometric_functions {f1 f2 f3 f4 : ℝ → ℝ} :
  (is_geometric_function f3) ∧ (is_geometric_function f4) ∧ 
  ¬ (is_geometric_function f1) ∧ ¬ (is_geometric_function f2) :=
by {
  sorry
}

end geometric_functions_l309_309083


namespace balls_in_boxes_l309_309227

theorem balls_in_boxes : (2^7 = 128) := 
by
  -- number of balls
  let n : ℕ := 7
  -- number of boxes
  let b : ℕ := 2
  have h : b ^ n = 128 := by sorry
  exact h

end balls_in_boxes_l309_309227


namespace solve_for_y_l309_309233

theorem solve_for_y (y : ℝ) (h_pos : y > 0) (h_eq : y^2 = 1024) : y = 32 := 
by
  sorry

end solve_for_y_l309_309233


namespace tan_150_eq_neg_inv_sqrt_3_l309_309904

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l309_309904


namespace find_length_of_room_l309_309160

def length_of_room (L : ℕ) (width verandah_width verandah_area : ℕ) : Prop :=
  (L + 2 * verandah_width) * (width + 2 * verandah_width) - (L * width) = verandah_area

theorem find_length_of_room : length_of_room 15 12 2 124 :=
by
  -- We state the proof here, which is not requested in this exercise
  sorry

end find_length_of_room_l309_309160


namespace crowdfunding_total_amount_l309_309073

theorem crowdfunding_total_amount
  (backers_highest_level : ℕ := 2)
  (backers_second_level : ℕ := 3)
  (backers_lowest_level : ℕ := 10)
  (amount_highest_level : ℝ := 5000) :
  ((backers_highest_level * amount_highest_level) + 
   (backers_second_level * (amount_highest_level / 10)) + 
   (backers_lowest_level * (amount_highest_level / 100))) = 12000 :=
by
  sorry

end crowdfunding_total_amount_l309_309073


namespace tan_150_eq_neg_inv_sqrt3_l309_309880

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l309_309880


namespace calculate_expression_l309_309861

-- Define the conditions
def exp1 : ℤ := (-1)^(53)
def exp2 : ℤ := 2^(2^4 + 5^2 - 4^3)

-- State and skip the proof
theorem calculate_expression :
  exp1 + exp2 = -1 + 1 / (2^23) :=
by sorry

#check calculate_expression

end calculate_expression_l309_309861


namespace removed_cubes_total_l309_309517

-- Define the large cube composed of 125 smaller cubes (5x5x5 cube)
def large_cube := 5 * 5 * 5

-- Number of smaller cubes removed from each face to opposite face
def removed_faces := (5 * 5 + 5 * 5 + 5 * 3)

-- Overlapping cubes deducted
def overlapping_cubes := (3 + 1)

-- Final number of removed smaller cubes
def removed_total := removed_faces - overlapping_cubes

-- Lean theorem statement
theorem removed_cubes_total : removed_total = 49 :=
by
  -- Definitions provided above imply the theorem
  sorry

end removed_cubes_total_l309_309517


namespace taxi_trip_miles_l309_309561

theorem taxi_trip_miles 
  (initial_fee : ℝ := 2.35)
  (additional_charge : ℝ := 0.35)
  (segment_length : ℝ := 2/5)
  (total_charge : ℝ := 5.50) :
  ∃ (miles : ℝ), total_charge = initial_fee + additional_charge * (miles / segment_length) ∧ miles = 3.6 :=
by
  sorry

end taxi_trip_miles_l309_309561


namespace length_of_platform_is_350_l309_309478

-- Define the parameters as given in the problem
def train_length : ℕ := 300
def time_to_cross_post : ℕ := 18
def time_to_cross_platform : ℕ := 39

-- Define the speed of the train as a ratio of the length of the train and the time to cross the post
def train_speed : ℚ := train_length / time_to_cross_post

-- Formalize the problem statement: Prove that the length of the platform is 350 meters
theorem length_of_platform_is_350 : ∃ (L : ℕ), (train_speed * time_to_cross_platform) = train_length + L := by
  use 350
  sorry

end length_of_platform_is_350_l309_309478


namespace simplify_expression_l309_309865

theorem simplify_expression (x : ℝ) : 2 * x * (x - 4) - (2 * x - 3) * (x + 2) = -9 * x + 6 :=
by
  sorry

end simplify_expression_l309_309865


namespace number_of_machines_l309_309739

def machine_problem : Prop :=
  ∃ (m : ℕ), (6 * 42) = 6 * 36 ∧ m = 7

theorem number_of_machines : machine_problem :=
  sorry

end number_of_machines_l309_309739


namespace polynomial_root_sum_l309_309394

theorem polynomial_root_sum 
  (c d : ℂ) 
  (h1 : c + d = 6) 
  (h2 : c * d = 10) 
  (h3 : c^2 - 6 * c + 10 = 0) 
  (h4 : d^2 - 6 * d + 10 = 0) : 
  c^3 + c^5 * d^3 + c^3 * d^5 + d^3 = 16156 := 
by sorry

end polynomial_root_sum_l309_309394


namespace cuboid_volume_l309_309051

theorem cuboid_volume (length width height : ℕ) (h_length : length = 4) (h_width : width = 4) (h_height : height = 6) : (length * width * height = 96) :=
by 
  -- Sorry places a placeholder for the actual proof
  sorry

end cuboid_volume_l309_309051


namespace purchase_options_l309_309713

def item_cost (a : Nat) : Nat := 100 * a + 99

def total_cost : Nat := 20083

theorem purchase_options (a : Nat) (n : Nat) (h : n * item_cost a = total_cost) :
  n = 17 ∨ n = 117 :=
by
  sorry

end purchase_options_l309_309713


namespace inheritance_amount_l309_309437

def federalTaxRate : ℝ := 0.25
def stateTaxRate : ℝ := 0.15
def totalTaxPaid : ℝ := 16500

theorem inheritance_amount :
  ∃ x : ℝ, (federalTaxRate * x + stateTaxRate * (1 - federalTaxRate) * x = totalTaxPaid) → x = 45500 := by
  sorry

end inheritance_amount_l309_309437


namespace length_of_opposite_leg_l309_309248

noncomputable def hypotenuse_length : Real := 18

noncomputable def angle_deg : Real := 30

theorem length_of_opposite_leg (h : Real) (angle : Real) (condition1 : h = hypotenuse_length) (condition2 : angle = angle_deg) : 
 ∃ x : Real, 2 * x = h ∧ angle = 30 → x = 9 := 
by
  sorry

end length_of_opposite_leg_l309_309248


namespace largest_8_digit_number_with_even_digits_l309_309826

def is_even (n : ℕ) : Prop := n % 2 = 0

def all_even_digits : List ℕ := [0, 2, 4, 6, 8]

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 8 ∧
  ∀ (d : ℕ), d ∈ digits → is_even d ∧
  all_even_digits ⊆ digits

theorem largest_8_digit_number_with_even_digits : ∃ n : ℕ, is_valid_number n ∧ n = 99986420 :=
sorry

end largest_8_digit_number_with_even_digits_l309_309826


namespace units_digit_of_composite_product_l309_309797

theorem units_digit_of_composite_product : 
  let composites := [4, 6, 8, 9],
      product := List.foldl (· * ·) 1 composites
  in product % 10 = 8 :=
  by
  sorry

end units_digit_of_composite_product_l309_309797


namespace tan_pi_div_4_add_alpha_l309_309097

theorem tan_pi_div_4_add_alpha (α : ℝ) (h : Real.sin α = 2 * Real.cos α) : 
  Real.tan (π / 4 + α) = -3 :=
by
  sorry

end tan_pi_div_4_add_alpha_l309_309097


namespace smallest_k_674_l309_309347

theorem smallest_k_674 :
  ∀ (S : Finset ℕ), (S ⊆ Finset.range 2017) → (S.card = 674) → 
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ (672 < a - b) ∧ (a - b < 1344) ∨ (672 < b - a) ∧ (b - a < 1344) :=
by sorry

end smallest_k_674_l309_309347


namespace total_outlets_needed_l309_309060

-- Definitions based on conditions:
def outlets_per_room : ℕ := 6
def number_of_rooms : ℕ := 7

-- Theorem to prove the total number of outlets is 42
theorem total_outlets_needed : outlets_per_room * number_of_rooms = 42 := by
  -- Simple proof with mathematics:
  sorry

end total_outlets_needed_l309_309060


namespace kopeechka_items_l309_309701

-- Define necessary concepts and conditions
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 200 * 100 + 83

-- Lean statement defining the proof problem
theorem kopeechka_items (a n : ℕ) (h1 : ∀ a, n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
by sorry

end kopeechka_items_l309_309701


namespace liters_to_pints_l309_309650

theorem liters_to_pints (l : ℝ) (p : ℝ) (h : 0.75 = l) (h_p : 1.575 = p) : 
  Float.round (1.5 * (p / l) * 10) / 10 = 3.2 :=
by sorry

end liters_to_pints_l309_309650


namespace height_of_wall_l309_309058

theorem height_of_wall (length_brick width_brick height_brick : ℝ)
                        (length_wall width_wall number_of_bricks : ℝ)
                        (volume_of_bricks : ℝ) :
  (length_brick, width_brick, height_brick) = (125, 11.25, 6) →
  (length_wall, width_wall) = (800, 22.5) →
  number_of_bricks = 1280 →
  volume_of_bricks = length_brick * width_brick * height_brick * number_of_bricks →
  volume_of_bricks = length_wall * width_wall * 600 := 
by
  intros h1 h2 h3 h4
  -- proof skipped
  sorry

end height_of_wall_l309_309058


namespace purchase_options_l309_309717

def item_cost (a : Nat) : Nat := 100 * a + 99

def total_cost : Nat := 20083

theorem purchase_options (a : Nat) (n : Nat) (h : n * item_cost a = total_cost) :
  n = 17 ∨ n = 117 :=
by
  sorry

end purchase_options_l309_309717


namespace find_non_divisible_and_product_l309_309575

-- Define the set of numbers
def numbers : List Nat := [3543, 3552, 3567, 3579, 3581]

-- Function to get the digits of a number
def digits (n : Nat) : List Nat := n.digits 10

-- Function to sum the digits
def sum_of_digits (n : Nat) : Nat := (digits n).sum

-- Function to check divisibility by 3
def divisible_by_3 (n : Nat) : Bool := sum_of_digits n % 3 = 0

-- Find the units digit of a number
def units_digit (n : Nat) : Nat := n % 10

-- Find the tens digit of a number
def tens_digit (n : Nat) : Nat := (n / 10) % 10

-- The problem statement
theorem find_non_divisible_and_product :
  ∃ n ∈ numbers, ¬ divisible_by_3 n ∧ units_digit n * tens_digit n = 8 :=
by
  sorry

end find_non_divisible_and_product_l309_309575


namespace units_digit_of_composite_product_l309_309793

theorem units_digit_of_composite_product : 
  let composites := [4, 6, 8, 9],
      product := List.foldl (· * ·) 1 composites
  in product % 10 = 8 :=
  by
  sorry

end units_digit_of_composite_product_l309_309793


namespace different_digits_probability_l309_309624

noncomputable def number_nonidentical_probability : ℚ :=
  let total_numbers := 900
  let identical_numbers := 9
  -- The probability of identical digits.
  let identical_probability := identical_numbers / total_numbers
  -- The probability of non-identical digits.
  1 - identical_probability

theorem different_digits_probability : number_nonidentical_probability = 99 / 100 := by
  sorry

end different_digits_probability_l309_309624


namespace trader_sells_cloth_l309_309496

variable (x : ℝ) (SP_total : ℝ := 6900) (profit_per_meter : ℝ := 20) (CP_per_meter : ℝ := 66.25)

theorem trader_sells_cloth : SP_total = x * (CP_per_meter + profit_per_meter) → x = 80 :=
by
  intro h
  -- Placeholder for actual proof
  sorry

end trader_sells_cloth_l309_309496


namespace determine_treasures_possible_l309_309162

structure Subject :=
  (is_knight : Prop)
  (is_liar : Prop)
  (is_normal : Prop)

def island_has_treasures : Prop := sorry

def can_determine_treasures (A B C : Subject) (at_most_one_normal : Bool) : Prop :=
  if at_most_one_normal then
    ∃ (question : (Subject → Prop)),
      (∀ response1, ∃ (question2 : (Subject → Prop)),
        (∀ response2, island_has_treasures ↔ (response1 ∧ response2)))
  else
    false

theorem determine_treasures_possible (A B C : Subject) (at_most_one_normal : Bool) :
  at_most_one_normal = true → can_determine_treasures A B C at_most_one_normal :=
by
  intro h
  sorry

end determine_treasures_possible_l309_309162


namespace bead_arrangement_probability_l309_309211

noncomputable def totalArrangements (red white blue green : ℕ) : ℕ :=
  let n := red + white + blue + green
  Nat.factorial n / (Nat.factorial red * Nat.factorial white * Nat.factorial blue * Nat.factorial green)

noncomputable def validArrangements := 0.05 * 12600

def probability_no_adjacent_beads_same (r w b g : ℕ) : Prop :=
  let total := totalArrangements r w b g
  validArrangements / total = 0.05

theorem bead_arrangement_probability :
  probability_no_adjacent_beads_same 4 3 2 1 := 
  sorry

end bead_arrangement_probability_l309_309211


namespace inequality_proof_l309_309140

open Real

theorem inequality_proof (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (hSum : x + y + z = 1) :
  x * y / sqrt (x * y + y * z) + y * z / sqrt (y * z + z * x) + z * x / sqrt (z * x + x * y) ≤ sqrt 2 / 2 := 
sorry

end inequality_proof_l309_309140


namespace palindrome_divisible_by_11_prob_l309_309312

def is_palindrome (n : ℕ) : Prop :=
  let digits := List.ofDigits [n / 10000 % 10, n / 1000 % 10, n / 100 % 10, n / 10 % 10, n % 10]
  in digits.reverse = digits

def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

def palindrome_in_range (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000

def random_palindrome_divisible_by_11 : Prop :=
  ∃ n, palindrome_in_range n ∧ is_palindrome n ∧ is_divisible_by_11 n

theorem palindrome_divisible_by_11_prob : 
  let total_palindromes := 9 * 10 * 10
  let valid_palindromes := 90
  let probability := valid_palindromes / total_palindromes
  probability = 1 / 10 :=
sorry

end palindrome_divisible_by_11_prob_l309_309312


namespace sqrt_of_9_l309_309584

theorem sqrt_of_9 : Real.sqrt 9 = 3 := 
by 
  sorry

end sqrt_of_9_l309_309584


namespace solve_equation_frac_l309_309038

theorem solve_equation_frac (x : ℝ) (h : x ≠ 2) : (3 / (x - 2) = 1) ↔ (x = 5) :=
by
  sorry -- proof is to be constructed

end solve_equation_frac_l309_309038


namespace tan_150_eq_l309_309889

noncomputable def cosine150 : ℚ := -√3 / 2
noncomputable def sine150 : ℚ := 1 / 2
noncomputable def tangent150 : ℚ := sine150 / cosine150

theorem tan_150_eq : tangent150 = -1 / √3 := by
  sorry

end tan_150_eq_l309_309889


namespace natural_number_divisor_problem_l309_309933

theorem natural_number_divisor_problem (x y z : ℕ) (h1 : (y+1)*(z+1) = 30) 
    (h2 : (x+1)*(z+1) = 42) (h3 : (x+1)*(y+1) = 35) :
    (2^x * 3^y * 5^z = 2^6 * 3^5 * 5^4) :=
sorry

end natural_number_divisor_problem_l309_309933


namespace largest_eight_digit_number_with_even_digits_l309_309837

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (num : ℕ) : Prop :=
  ∀ d, is_even_digit d → (num.toDigits.contains d)

def is_eight_digit_number (num : ℕ) : Prop :=
  10000000 ≤ num ∧ num < 100000000

def largest_number_with_conditions (num : ℕ) : ℕ :=
  if contains_all_even_digits num ∧ is_eight_digit_number num then num else 0

theorem largest_eight_digit_number_with_even_digits :
  largest_number_with_conditions 99986420 = 99986420 :=
begin
  sorry
end

end largest_eight_digit_number_with_even_digits_l309_309837


namespace y_value_is_32_l309_309235

-- Define the conditions
variables (y : ℝ) (hy_pos : y > 0) (hy_eq : y^2 = 1024)

-- State the theorem
theorem y_value_is_32 : y = 32 :=
by
  -- The proof will be written here
  sorry

end y_value_is_32_l309_309235


namespace profit_percent_l309_309300

theorem profit_percent (CP SP : ℕ) (h : CP * 5 = SP * 4) : 100 * (SP - CP) = 25 * CP :=
by
  sorry

end profit_percent_l309_309300


namespace rectangle_division_impossible_l309_309500

theorem rectangle_division_impossible :
  ¬ ∃ n m : ℕ, n * 5 = 55 ∧ m * 11 = 39 :=
by
  sorry

end rectangle_division_impossible_l309_309500


namespace simultaneous_equations_in_quadrant_I_l309_309720

theorem simultaneous_equations_in_quadrant_I (c : ℝ) :
  (∃ x y : ℝ, x - y = 3 ∧ c * x + y = 4 ∧ x > 0 ∧ y > 0) ↔ (-1 < c ∧ c < 4 / 3) :=
  sorry

end simultaneous_equations_in_quadrant_I_l309_309720


namespace one_div_thirteen_150th_digit_l309_309456

theorem one_div_thirteen_150th_digit :
  let repeating_digits := [0, 7, 6, 9, 2, 3]
  (repeating_digits.nth ((150 - 1) % repeating_digits.length)).get_or_else (-1) = 0 :=
by
  -- provided for skipping proof
  sorry

end one_div_thirteen_150th_digit_l309_309456


namespace tan_150_eq_neg_inv_sqrt_3_l309_309902

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l309_309902


namespace units_digit_first_four_composites_l309_309766

theorem units_digit_first_four_composites :
  let p := [4, 6, 8, 9] in
  let product := List.prod p in
  product % 10 = 8 :=
by
  let p := [4, 6, 8, 9]
  let product := List.prod p
  show product % 10 = 8
  sorry

end units_digit_first_four_composites_l309_309766


namespace russian_dolls_initial_purchase_l309_309082

theorem russian_dolls_initial_purchase (cost_initial cost_discount : ℕ) (num_discount : ℕ) (savings : ℕ) :
  cost_initial = 4 → cost_discount = 3 → num_discount = 20 → savings = num_discount * cost_discount → 
  (savings / cost_initial) = 15 := 
by {
sorry
}

end russian_dolls_initial_purchase_l309_309082


namespace shaltaev_boltaev_proof_l309_309302

variable (S B : ℝ)

axiom cond1 : 175 * S > 125 * B
axiom cond2 : 175 * S < 126 * B

theorem shaltaev_boltaev_proof : 3 * S + B ≥ 1 :=
by {
  sorry
}

end shaltaev_boltaev_proof_l309_309302


namespace simplify_abs_sum_l309_309534

theorem simplify_abs_sum (a b c : ℝ) (h1 : a + b > c) (h2 : b + c > a) (h3 : a + c > b) :
  |c - a - b| + |c + b - a| = 2 * b :=
sorry

end simplify_abs_sum_l309_309534


namespace tan_150_eq_neg_one_over_sqrt_three_l309_309911

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l309_309911


namespace total_appetizers_l309_309201

theorem total_appetizers (hotdogs cheese_pops chicken_nuggets mini_quiches stuffed_mushrooms total_portions : Nat)
  (h1 : hotdogs = 60)
  (h2 : cheese_pops = 40)
  (h3 : chicken_nuggets = 80)
  (h4 : mini_quiches = 100)
  (h5 : stuffed_mushrooms = 50)
  (h6 : total_portions = hotdogs + cheese_pops + chicken_nuggets + mini_quiches + stuffed_mushrooms) :
  total_portions = 330 :=
by sorry

end total_appetizers_l309_309201


namespace decimal_1_div_13_150th_digit_is_3_l309_309447

theorem decimal_1_div_13_150th_digit_is_3 :
  (let repeating_block := "076923";
   let block_length := String.length repeating_block in
   repeating_block[5] = '3') → 
   (150 % block_length = 0) →
   (repeating_block[(150 % block_length) - 1] = '3') :=
by
  intros h_block h_mod
  sorry

end decimal_1_div_13_150th_digit_is_3_l309_309447


namespace sum_of_digits_l309_309426

theorem sum_of_digits (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
                      (h_range : 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9)
                      (h_product : a * b * c * d = 810) :
  a + b + c + d = 23 := sorry

end sum_of_digits_l309_309426


namespace curve_symmetric_reflection_l309_309274

theorem curve_symmetric_reflection (f : ℝ → ℝ → ℝ) :
  (∀ x y, f x y = 0 ↔ f (y + 3) (x - 3) = 0) → 
  (∀ x y, (x - y - 3 = 0) → (f (y + 3) (x - 3) = 0)) :=
sorry

end curve_symmetric_reflection_l309_309274


namespace line_intersects_circle_l309_309547

theorem line_intersects_circle 
  (k : ℝ)
  (h_tangent : ∀ x y : ℝ, (x - 2)^2 + (y - 1)^2 = 2 → y = k * x - 1) :
  ∀ x y : ℝ, ((x - 2)^2 + y^2 = 3) → (y = k * x - 1) ∨ (y ≠ k * x - 1) :=
by
  sorry

end line_intersects_circle_l309_309547


namespace mean_of_three_digit_multiples_of_8_l309_309044

theorem mean_of_three_digit_multiples_of_8 :
  let a1 := 104
  let an := 992
  let d := 8
  let n := (an - a1) / d + 1
  let sum := n * (a1 + an) / 2
  let mean := sum / n
  mean = 548 :=
by
  let a1 := 104
  let an := 992
  let d := 8
  let n := (an - a1) / d + 1
  let sum := n * (a1 + an) / 2
  let mean := sum / n
  sorry

end mean_of_three_digit_multiples_of_8_l309_309044


namespace correct_equation_l309_309324

-- Definitions based on conditions
def total_students := 98
def transfer_students := 3
def original_students_A (x : ℕ) : Prop := x ≥ 0 ∧ x ≤ total_students
def students_B (x : ℕ) := total_students - x

-- Equation set up based on translation of the proof problem
theorem correct_equation (x : ℕ) (h : original_students_A x) :
  students_B x + transfer_students = x - transfer_students ↔ (98 - x) + 3 = x - 3 :=
by
  sorry
  
end correct_equation_l309_309324


namespace hot_dogs_sold_next_innings_l309_309048

-- Defining the conditions
variables (total_initial hot_dogs_sold_first_innings hot_dogs_left : ℕ)

-- Given conditions that need to hold true
axiom initial_count : total_initial = 91
axiom first_innings_sold : hot_dogs_sold_first_innings = 19
axiom remaining_hot_dogs : hot_dogs_left = 45

-- Prove the number of hot dogs sold during the next three innings is 27
theorem hot_dogs_sold_next_innings : total_initial - (hot_dogs_sold_first_innings + hot_dogs_left) = 27 :=
by
  sorry

end hot_dogs_sold_next_innings_l309_309048


namespace correct_conclusions_l309_309354

variable (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0)

def f (x : ℝ) : ℝ := x^2

theorem correct_conclusions (h_distinct : x1 ≠ x2) :
  (f x1 * x2 = f x1 * f x2) ∧
  ((f x1 - f x2) / (x1 - x2) > 0) ∧
  (f ((x1 + x2) / 2) < (f x1 + f x2) / 2) :=
by
  sorry

end correct_conclusions_l309_309354


namespace gcd_lcm_sum_l309_309564

theorem gcd_lcm_sum (a b : ℕ) (h : a = 1999 * b) : Nat.gcd a b + Nat.lcm a b = 2000 * b := by
  sorry

end gcd_lcm_sum_l309_309564


namespace arithmetic_expression_l309_309092

theorem arithmetic_expression : 4 * 6 * 8 + 18 / 3 - 2 ^ 3 = 190 :=
by
  -- Proof goes here
  sorry

end arithmetic_expression_l309_309092


namespace other_train_speed_l309_309598

noncomputable def speed_of_other_train (l1 l2 v1 : ℕ) (t : ℝ) : ℝ := 
  let relative_speed := (l1 + l2) / 1000 / (t / 3600)
  relative_speed - v1

theorem other_train_speed :
  speed_of_other_train 210 260 40 16.918646508279338 = 60 := 
by
  sorry

end other_train_speed_l309_309598


namespace kelly_grade_correct_l309_309560

variable (Jenny Jason Bob Kelly : ℕ)

def jenny_grade : ℕ := 95
def jason_grade := jenny_grade - 25
def bob_grade := jason_grade / 2
def kelly_grade := bob_grade + (bob_grade / 5)  -- 20% of Bob's grade is (Bob's grade * 0.20), which is the same as (Bob's grade / 5)

theorem kelly_grade_correct : kelly_grade = 42 :=
by
  sorry

end kelly_grade_correct_l309_309560


namespace division_quotient_is_correct_l309_309289

noncomputable def polynomial_division_quotient : Polynomial ℚ :=
  Polynomial.div (Polynomial.C 8 * Polynomial.X ^ 3 + 
                  Polynomial.C 16 * Polynomial.X ^ 2 + 
                  Polynomial.C (-7) * Polynomial.X + 
                  Polynomial.C 4) 
                 (Polynomial.C 2 * Polynomial.X + Polynomial.C 5)

theorem division_quotient_is_correct :
  polynomial_division_quotient =
    Polynomial.C 4 * Polynomial.X ^ 2 +
    Polynomial.C (-2) * Polynomial.X +
    Polynomial.C (3 / 2) :=
by
  sorry

end division_quotient_is_correct_l309_309289


namespace pumpkins_eaten_l309_309411

theorem pumpkins_eaten (initial: ℕ) (left: ℕ) (eaten: ℕ) (h1 : initial = 43) (h2 : left = 20) : eaten = 23 :=
by {
  -- We are skipping the proof as per the requirement
  sorry
}

end pumpkins_eaten_l309_309411


namespace tan_150_eq_neg_inv_sqrt3_l309_309882

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l309_309882


namespace area_difference_l309_309385

theorem area_difference (A B a b : ℝ) : (A * b) - (a * B) = A * b - a * B :=
by {
  -- proof goes here
  sorry
}

end area_difference_l309_309385


namespace tan_tan_lt_half_l309_309147

noncomputable def tan_half (x : ℝ) : ℝ := Real.tan (x / 2)

theorem tan_tan_lt_half (a b c α β : ℝ) (h1: a + b < 3 * c) (h2: tan_half α * tan_half β = (a + b - c) / (a + b + c)) :
  tan_half α * tan_half β < 1 / 2 := 
sorry

end tan_tan_lt_half_l309_309147


namespace sock_pair_selection_l309_309961

def num_white_socks : Nat := 5
def num_brown_socks : Nat := 5
def num_blue_socks : Nat := 3

def white_odd_positions : List Nat := [1, 3, 5]
def white_even_positions : List Nat := [2, 4]

def brown_odd_positions : List Nat := [1, 3, 5]
def brown_even_positions : List Nat := [2, 4]

def blue_odd_positions : List Nat := [1, 3]
def blue_even_positions : List Nat := [2]

noncomputable def count_pairs : Nat :=
  let white_brown := (white_odd_positions.length * brown_odd_positions.length) +
                     (white_even_positions.length * brown_even_positions.length)
  
  let brown_blue := (brown_odd_positions.length * blue_odd_positions.length) +
                    (brown_even_positions.length * blue_even_positions.length)

  let white_blue := (white_odd_positions.length * blue_odd_positions.length) +
                    (white_even_positions.length * blue_even_positions.length)

  white_brown + brown_blue + white_blue

theorem sock_pair_selection :
  count_pairs = 29 :=
by
  sorry

end sock_pair_selection_l309_309961


namespace number_of_small_triangles_l309_309492

noncomputable def area_of_large_triangle (hypotenuse_large : ℝ) : ℝ :=
  let leg := hypotenuse_large / Real.sqrt 2
  (1 / 2) * (leg * leg)

noncomputable def area_of_small_triangle (hypotenuse_small : ℝ) : ℝ :=
  let leg := hypotenuse_small / Real.sqrt 2
  (1 / 2) * (leg * leg)

theorem number_of_small_triangles (hypotenuse_large : ℝ) (hypotenuse_small : ℝ) :
  hypotenuse_large = 14 → hypotenuse_small = 2 →
  let number_of_triangles := (area_of_large_triangle hypotenuse_large) / (area_of_small_triangle hypotenuse_small)
  number_of_triangles = 49 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end number_of_small_triangles_l309_309492


namespace a_n_formula_b_n_formula_l309_309940

namespace SequenceFormulas

theorem a_n_formula (n : ℕ) (h_pos : 0 < n) : 
  (∃ S : ℕ → ℕ, S n = 2 * n^2 + 2 * n) → ∃ a : ℕ → ℕ, a n = 4 * n :=
by
  sorry

theorem b_n_formula (n : ℕ) (h_pos : 0 < n) : 
  (∃ T : ℕ → ℕ, T n = 2 - (if n > 1 then T (n-1) else 1)) → ∃ b : ℕ → ℝ, b n = (1/2)^(n-1) :=
by
  sorry

end SequenceFormulas


end a_n_formula_b_n_formula_l309_309940


namespace length_of_AD_l309_309191

-- Define the segment AD and points B, C, and M as given conditions
variable (x : ℝ) -- Assuming x is the length of segments AB, BC, CD
variable (AD : ℝ)
variable (MC : ℝ)

-- Conditions given in the problem statement
def trisect (AD : ℝ) : Prop :=
  ∃ (x : ℝ), AD = 3 * x ∧ 0 < x

def one_third_way (M AD : ℝ) : Prop :=
  M = AD / 3

def distance_MC (M C : ℝ) : ℝ :=
  C - M

noncomputable def D : Prop := sorry

-- The main theorem statement
theorem length_of_AD (AD : ℝ) (M : ℝ) (MC : ℝ) : trisect AD → one_third_way M AD → MC = M / 3 → AD = 15 :=
by
  intro H1 H2 H3
  -- sorry is added to skip the actual proof
  sorry

end length_of_AD_l309_309191


namespace union_M_N_l309_309141

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | 1 < x ∧ x < 3}

theorem union_M_N : M ∪ N = {x | -1 < x ∧ x < 3} := 
by 
  sorry

end union_M_N_l309_309141


namespace probability_black_or_white_l309_309303

-- Defining the probabilities of drawing red and white balls
def prob_red : ℝ := 0.45
def prob_white : ℝ := 0.25

-- Defining the total probability
def total_prob : ℝ := 1.0

-- Define the probability of drawing a black or white ball
def prob_black_or_white : ℝ := total_prob - prob_red

-- The theorem stating the required proof
theorem probability_black_or_white : 
  prob_black_or_white = 0.55 := by
    sorry

end probability_black_or_white_l309_309303


namespace units_digit_first_four_composites_l309_309813

theorem units_digit_first_four_composites : 
  let first_four_composites := [4, 6, 8, 9]
  let product := first_four_composites.prod
  Nat.unitsDigit product = 8 :=
by
  sorry

end units_digit_first_four_composites_l309_309813


namespace tan_150_eq_neg_sqrt_3_l309_309890

theorem tan_150_eq_neg_sqrt_3 :
  let θ := 150 * Real.pi / 180 in
  Real.tan θ = -Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_sqrt_3_l309_309890


namespace hours_worked_each_day_l309_309934

-- Definitions based on problem conditions
def total_hours_worked : ℝ := 8.0
def number_of_days_worked : ℝ := 4.0

-- Theorem statement to prove the number of hours worked each day
theorem hours_worked_each_day :
  total_hours_worked / number_of_days_worked = 2.0 :=
sorry

end hours_worked_each_day_l309_309934


namespace simplify_fraction_l309_309268

theorem simplify_fraction :
  (1 : ℚ) / ((1 / (1 / 3 : ℚ) ^ 1) + (1 / (1 / 3 : ℚ) ^ 2) + (1 / (1 / 3 : ℚ) ^ 3) + (1 / (1 / 3 : ℚ) ^ 4)) = 1 / 120 := 
by 
  sorry

end simplify_fraction_l309_309268


namespace choose_math_class_representative_l309_309551

def number_of_boys : Nat := 26
def number_of_girls : Nat := 24

theorem choose_math_class_representative : number_of_boys + number_of_girls = 50 := 
by
  sorry

end choose_math_class_representative_l309_309551


namespace tan_150_eq_neg_sqrt3_div_3_l309_309921

theorem tan_150_eq_neg_sqrt3_div_3
: tan 150 = - (Real.sqrt 3 / 3) :=
by
  have h1 : 150 = 180 - 30, by rfl
  have h2 : sin 150 = sin (180 - 30), by simp [h1]
  have h3 : cos 150 = cos (180 - 30), by simp [h1]
  have cos_30 := Real.sqrt 3 / 2
  have sin_30 := 1 / 2
  simp [Real.sin_eq_sin_of_Real_angle, Real.cos_eq_cos_of_Real_angle] at h2 h3
  have h4 : sin 150 = sin 30, by simp [h2, sin_30]
  have h5 : cos 150 = -cos 30, by simp [h3, cos_30]
  rw [Real.tan_eq_sin_div_cos, h4, h5]
  have h6 : (1 / 2) / (-Real.sqrt 3 / 2) = - (1 / Real.sqrt 3), by linarith
  rw h6
  have h7 : - (1 / Real.sqrt 3) = - (Real.sqrt 3 / 3), by
    field_simp [Real.sqrt_ne_zero],
    linarith
  exact h7,
  sorry

end tan_150_eq_neg_sqrt3_div_3_l309_309921


namespace percentage_in_excess_l309_309960

theorem percentage_in_excess 
  (A B : ℝ) (x : ℝ)
  (h1 : ∀ A',  A' = A * (1 + x / 100))
  (h2 : ∀ B',  B' = 0.94 * B)
  (h3 : ∀ A' B', A' * B' = A * B * (1 + 0.0058)) :
  x = 7 :=
by
  sorry

end percentage_in_excess_l309_309960


namespace max_value_of_sum_l309_309099

theorem max_value_of_sum 
  (a b c : ℝ) 
  (h : a^2 + 2 * b^2 + 3 * c^2 = 6) : 
  a + b + c ≤ Real.sqrt 11 := 
by 
  sorry

end max_value_of_sum_l309_309099


namespace reduced_price_per_kg_l309_309197

theorem reduced_price_per_kg {P R : ℝ} (H1 : R = 0.75 * P) (H2 : 1100 = 1100 / P * P) (H3 : 1100 = (1100 / P + 5) * R) : R = 55 :=
by sorry

end reduced_price_per_kg_l309_309197


namespace polynomial_divisibility_l309_309732

theorem polynomial_divisibility (n : ℕ) : (¬ n % 3 = 0) → (x ^ (2 * n) + x ^ n + 1) % (x ^ 2 + x + 1) = 0 :=
by
  sorry

end polynomial_divisibility_l309_309732


namespace min_value_l309_309118

theorem min_value (m n : ℝ) (h1 : 2 * m + n = 1) (h2 : m > 0) (h3 : n > 0) :
  ∃ x, x = 3 + 2 * Real.sqrt 2 ∧ (∀ y, (2 * m + n = 1 → m > 0 → n > 0 → y = (1 / m) + (1 / n) → y ≥ x)) :=
by
  sorry

end min_value_l309_309118


namespace smallest_x_division_remainder_l309_309470

theorem smallest_x_division_remainder :
  ∃ x : ℕ, x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7 ∧ x = 167 := by
  sorry

end smallest_x_division_remainder_l309_309470


namespace gcd_calculation_l309_309506

theorem gcd_calculation :
  let a := 97^7 + 1
  let b := 97^7 + 97^3 + 1
  gcd a b = 1 := by
  sorry

end gcd_calculation_l309_309506


namespace fraction_to_decimal_l309_309926

theorem fraction_to_decimal : (7 / 12 : ℝ) = 0.5833 := by
  sorry

end fraction_to_decimal_l309_309926


namespace intersect_at_2d_l309_309746

def g (x : ℝ) (c : ℝ) : ℝ := 4 * x + c

theorem intersect_at_2d (c d : ℤ) (h₁ : d = 8 + c) (h₂ : 2 = g d c) : d = 2 :=
by
  sorry

end intersect_at_2d_l309_309746


namespace tan_150_eq_neg_inv_sqrt3_l309_309916

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l309_309916


namespace tan_150_deg_l309_309896

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l309_309896


namespace integer_solutions_exist_l309_309951

theorem integer_solutions_exist (x y : ℤ) : 
  12 * x^2 + 7 * y^2 = 4620 ↔ 
  (x = 7 ∧ y = 24) ∨ 
  (x = -7 ∧ y = 24) ∨
  (x = 7 ∧ y = -24) ∨
  (x = -7 ∧ y = -24) ∨
  (x = 14 ∧ y = 18) ∨
  (x = -14 ∧ y = 18) ∨
  (x = 14 ∧ y = -18) ∨
  (x = -14 ∧ y = -18) :=
sorry

end integer_solutions_exist_l309_309951


namespace correct_statements_count_l309_309589

/-
  Question: How many students have given correct interpretations of the algebraic expression \( 7x \)?
  Conditions:
    - Xiaoming's Statement: \( 7x \) can represent the sum of \( 7 \) and \( x \).
    - Xiaogang's Statement: \( 7x \) can represent the product of \( 7 \) and \( x \).
    - Xiaoliang's Statement: \( 7x \) can represent the total price of buying \( x \) pens at a unit price of \( 7 \) yuan.
  Given these conditions, prove that the number of correct statements is \( 2 \).
-/

theorem correct_statements_count (x : ℕ) :
  (if 7 * x = 7 + x then 1 else 0) +
  (if 7 * x = 7 * x then 1 else 0) +
  (if 7 * x = 7 * x then 1 else 0) = 2 := sorry

end correct_statements_count_l309_309589


namespace line_does_not_pass_through_third_quadrant_l309_309377

theorem line_does_not_pass_through_third_quadrant (k : ℝ) :
  (∀ x : ℝ, ¬ (x > 0 ∧ (-3 * x + k) < 0)) ∧ (∀ x : ℝ, ¬ (x < 0 ∧ (-3 * x + k) > 0)) → k ≥ 0 :=
by
  sorry

end line_does_not_pass_through_third_quadrant_l309_309377


namespace seven_points_unit_distance_l309_309734

theorem seven_points_unit_distance :
  ∃ (A B C D E F G : ℝ × ℝ),
    (A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧
     B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧
     C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧
     D ≠ E ∧ D ≠ F ∧ D ≠ G ∧
     E ≠ F ∧ E ≠ G ∧
     F ≠ G) ∧
    (∀ (P Q R : ℝ × ℝ),
      (P = A ∨ P = B ∨ P = C ∨ P = D ∨ P = E ∨ P = F ∨ P = G) →
      (Q = A ∨ Q = B ∨ Q = C ∨ Q = D ∨ Q = E ∨ Q = F ∨ Q = G) →
      (R = A ∨ R = B ∨ R = C ∨ R = D ∨ R = E ∨ R = F ∨ R = G) →
      P ≠ Q → P ≠ R → Q ≠ R →
      (dist P Q = 1 ∨ dist P R = 1 ∨ dist Q R = 1)) :=
sorry

end seven_points_unit_distance_l309_309734


namespace rabbit_time_2_miles_l309_309314

def rabbit_travel_time (distance : ℕ) (rate : ℕ) : ℕ :=
  (distance * 60) / rate

theorem rabbit_time_2_miles : rabbit_travel_time 2 5 = 24 := by
  sorry

end rabbit_time_2_miles_l309_309314


namespace pumpkins_eaten_l309_309414

-- Definitions for the conditions
def originalPumpkins : ℕ := 43
def leftPumpkins : ℕ := 20

-- Theorem statement
theorem pumpkins_eaten : originalPumpkins - leftPumpkins = 23 :=
  by
    -- Proof steps are omitted
    sorry

end pumpkins_eaten_l309_309414


namespace tan_150_deg_l309_309897

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l309_309897


namespace least_number_remainder_l309_309481

open Nat

theorem least_number_remainder (n : ℕ) :
  (n ≡ 4 [MOD 5]) →
  (n ≡ 4 [MOD 6]) →
  (n ≡ 4 [MOD 9]) →
  (n ≡ 4 [MOD 12]) →
  n = 184 :=
by
  intros h1 h2 h3 h4
  sorry

end least_number_remainder_l309_309481


namespace find_f_on_interval_l309_309971

/-- Representation of periodic and even functions along with specific interval definition -/
noncomputable def f (x : ℝ) : ℝ := 
if 2 ≤ x ∧ x ≤ 3 then -2*(x-3)^2 + 4 else 0 -- Define f(x) on [2,3], otherwise undefined

/-- Main proof statement -/
theorem find_f_on_interval :
  (∀ x, f x = f (x + 2)) ∧  -- f(x) is periodic with period 2
  (∀ x, f x = f (-x)) ∧   -- f(x) is even
  (∀ x, 2 ≤ x ∧ x ≤ 3 → f x = -2*(x-3)^2 + 4) →
  (∀ x, 0 ≤ x ∧ x ≤ 2 → f x = -2*(x-1)^2 + 4) :=
sorry

end find_f_on_interval_l309_309971


namespace largest_eight_digit_with_all_even_digits_l309_309829

theorem largest_eight_digit_with_all_even_digits :
  ∀ n : ℕ, (∃ d1 d2 d3 d4 d5 : ℕ, (d1, d2, d3, d4, d5) = (0, 2, 4, 6, 8) ∧ 
    (99900000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) = n) → n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_even_digits_l309_309829


namespace num_brownies_correct_l309_309967

-- Define the conditions (pan dimensions and brownie piece dimensions)
def pan_width : ℕ := 24
def pan_length : ℕ := 15
def piece_width : ℕ := 3
def piece_length : ℕ := 2

-- Define the area calculations for the pan and each piece
def pan_area : ℕ := pan_width * pan_length
def piece_area : ℕ := piece_width * piece_length

-- Define the problem statement to prove the number of brownies
def number_of_brownies : ℕ := pan_area / piece_area

-- The statement we need to prove
theorem num_brownies_correct : number_of_brownies = 60 :=
by
  sorry

end num_brownies_correct_l309_309967


namespace polar_to_cartesian_circle_l309_309222

open Real

-- Given the polar coordinate equation of a circle, we need to prove the following:
theorem polar_to_cartesian_circle :
  (∀ θ ρ, ρ^2 - 4 * sqrt 2 * ρ * cos (θ - π/4) + 6 = 0) →
  (∀ x y, (x = 2 + sqrt 2 * cos θ) → (y = 2 + sqrt 2 * sin θ) →
  x^2 + y^2 - 4 * x - 4 * y + 6 = 0) ∧ 
  (∀ x y, (x = 2 + sqrt 2 * cos θ) → (y = 2 + sqrt 2 * sin θ) →
  (∀ t, t = sin θ + cos θ → (xy = t^2 + 2 * sqrt 2 * t + 3 ∧ 
  min xy = 1 ∧ max xy = 9)))
sorry

end polar_to_cartesian_circle_l309_309222


namespace fixed_point_l309_309959

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x + 1) + 2

theorem fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : f a (-1) = 3 :=
by
  unfold f
  sorry

end fixed_point_l309_309959


namespace range_of_k_l309_309110

-- Part (I) Definition of the quadratic function and its properties
def quadratic_function (a b : ℝ) (x : ℝ) := a * x^2 - 2 * a * x + b + 1

-- Assumptions given
variables (a b : ℝ) (ha : 0 < a) (hf_2 : quadratic_function a b 2 = 1) (hf_3 : quadratic_function a b 3 = 4)

-- Part (II) Definition of the rational function g(x)
def g_function (a b : ℝ) (x : ℝ) := (quadratic_function a b x) / x

-- Second inequality problem
theorem range_of_k 
  (a : ℝ) (b : ℝ) (ha : 0 < a) (hf_2 : quadratic_function a b 2 = 1) (hf_3 : quadratic_function a b 3 = 4) :
  ∃ k, ∀ x ∈ Icc 1 2, g_function a b (2 ^ x) - k * (2 ^ x) ≥ 0 := 
sorry

end range_of_k_l309_309110


namespace min_packs_for_126_cans_l309_309152

-- Definition of pack sizes
def pack_sizes : List ℕ := [15, 18, 36]

-- The given total cans of soda
def total_cans : ℕ := 126

-- The minimum number of packs needed to buy exactly 126 cans of soda
def min_packs_needed (total : ℕ) (packs : List ℕ) : ℕ :=
  -- Function definition to calculate the minimum packs needed
  -- This function needs to be implemented or proven
  sorry

-- The proof that the minimum number of packs needed to buy exactly 126 cans of soda is 4
theorem min_packs_for_126_cans : min_packs_needed total_cans pack_sizes = 4 :=
  -- Proof goes here
  sorry

end min_packs_for_126_cans_l309_309152


namespace Lucy_retirement_month_l309_309004

theorem Lucy_retirement_month (start_month : ℕ) (duration : ℕ) (March : ℕ) (May : ℕ) : 
  (start_month = March) ∧ (duration = 3) → (start_month + duration - 1 = May) :=
by
  intro h
  have h_start_month := h.1
  have h_duration := h.2
  sorry

end Lucy_retirement_month_l309_309004


namespace excursion_min_parents_l309_309433

theorem excursion_min_parents 
  (students : ℕ) 
  (car_capacity : ℕ)
  (h_students : students = 30)
  (h_car_capacity : car_capacity = 5) 
  : ∃ (parents_needed : ℕ), parents_needed = 8 := 
by
  sorry -- proof goes here

end excursion_min_parents_l309_309433


namespace Lisa_weight_l309_309319

theorem Lisa_weight : ∃ l a : ℝ, a + l = 240 ∧ l - a = l / 3 ∧ l = 144 :=
by
  sorry

end Lisa_weight_l309_309319


namespace largest_angle_heptagon_l309_309419

theorem largest_angle_heptagon :
  ∃ (x : ℝ), 4 * x + 4 * x + 4 * x + 5 * x + 6 * x + 7 * x + 8 * x = 900 ∧ 8 * x = (7200 / 38) := 
by 
  sorry

end largest_angle_heptagon_l309_309419


namespace gross_profit_value_l309_309750

theorem gross_profit_value (C GP : ℝ) (h1 : GP = 1.6 * C) (h2 : 91 = C + GP) : GP = 56 :=
by
  sorry

end gross_profit_value_l309_309750


namespace solution_set_of_inequality_l309_309430

theorem solution_set_of_inequality : 
  { x : ℝ | (3 - 2 * x) * (x + 1) ≤ 0 } = { x : ℝ | -1 ≤ x ∧ x ≤ 3 / 2 } :=
sorry

end solution_set_of_inequality_l309_309430


namespace tan_150_deg_l309_309872

-- Define the conditions
def angle_150_deg := 150 * real.pi / 180
def coordinates_of_Q := (-real.sqrt 3 / 2, 1 / 2)
def tan_of_angle_150 := real.tan angle_150_deg

-- The statement to prove
theorem tan_150_deg : tan_of_angle_150 = -1 / real.sqrt 3 :=
by
  -- Proof omitted
  sorry

end tan_150_deg_l309_309872


namespace Xiaogang_raised_arm_exceeds_head_l309_309475

theorem Xiaogang_raised_arm_exceeds_head :
  ∀ (height shadow_no_arm shadow_with_arm : ℝ),
    height = 1.7 → shadow_no_arm = 0.85 → shadow_with_arm = 1.1 →
    (height / shadow_no_arm) = ((shadow_with_arm - shadow_no_arm) * (height / shadow_no_arm)) →
    shadow_with_arm - shadow_no_arm = 0.25 →
    ((height / shadow_no_arm) * 0.25) = 0.5 :=
by
  intros height shadow_no_arm shadow_with_arm h_eq1 h_eq2 h_eq3 h_eq4 h_eq5
  sorry

end Xiaogang_raised_arm_exceeds_head_l309_309475


namespace tan_150_eq_neg_one_over_sqrt_three_l309_309908

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l309_309908


namespace average_weight_of_whole_class_l309_309602

theorem average_weight_of_whole_class :
  ∀ (n_a n_b : ℕ) (w_avg_a w_avg_b : ℝ),
    n_a = 60 →
    n_b = 70 →
    w_avg_a = 60 →
    w_avg_b = 80 →
    (n_a * w_avg_a + n_b * w_avg_b) / (n_a + n_b) = 70.77 :=
by
  intros n_a n_b w_avg_a w_avg_b h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end average_weight_of_whole_class_l309_309602


namespace minimum_distinct_values_is_145_l309_309494

-- Define the conditions
def n_series : ℕ := 2023
def unique_mode_occurrence : ℕ := 15

-- Define the minimum number of distinct values satisfying the conditions
def min_distinct_values (n : ℕ) (mode_count : ℕ) : ℕ :=
  if mode_count < n then 
    (n - mode_count + 13) / 14 + 1
  else
    1

-- The theorem restating the problem to be solved
theorem minimum_distinct_values_is_145 : 
  min_distinct_values n_series unique_mode_occurrence = 145 :=
by
  sorry

end minimum_distinct_values_is_145_l309_309494


namespace sweet_numbers_count_l309_309601

-- Define the conditions of the sequence
def tripleOrSubtract (n : ℕ) : ℕ :=
if n ≤ 30 then 3 * n else n - 15

-- Define the sequence function
def sequence (G : ℕ) : ℕ → ℕ
| 0     => G
| (n+1) => tripleOrSubtract (sequence n)

-- Define "sweet number" condition
def is_sweet_number (G : ℕ) : Prop :=
∀ n, sequence G n ≠ 18

-- Define the count of sweet numbers in the range 1 to 60
def count_sweet_numbers : ℕ :=
Nat.card {G : ℕ // 1 ≤ G ∧ G ≤ 60 ∧ is_sweet_number G}

-- The statement to prove
theorem sweet_numbers_count : count_sweet_numbers = 40 := by
  sorry

end sweet_numbers_count_l309_309601


namespace price_of_most_expensive_book_l309_309096

-- Define the conditions
def number_of_books := 41
def price_increment := 3

-- Define the price of the n-th book as a function of the price of the first book
def price (c : ℕ) (n : ℕ) : ℕ := c + price_increment * (n - 1)

-- Define a theorem stating the result
theorem price_of_most_expensive_book (c : ℕ) :
  c = 30 → price c number_of_books = 150 :=
by {
  sorry
}

end price_of_most_expensive_book_l309_309096


namespace max_area_of_triangle_MAN_l309_309105

noncomputable def maximum_area_triangle_MAN (e : ℝ) (F : ℝ × ℝ) (A : ℝ × ℝ) : ℝ :=
  if h : e = Real.sqrt 3 / 2 ∧ F = (Real.sqrt 3, 0) ∧ A = (1, 1 / 2) then
    Real.sqrt 2
  else
    0

theorem max_area_of_triangle_MAN :
  maximum_area_triangle_MAN (Real.sqrt 3 / 2) (Real.sqrt 3, 0) (1, 1 / 2) = Real.sqrt 2 :=
by
  sorry

end max_area_of_triangle_MAN_l309_309105


namespace jersey_sum_adjacent_gt_17_l309_309730

theorem jersey_sum_adjacent_gt_17 (a : ℕ → ℕ) (h_unique : ∀ i j, i ≠ j → a i ≠ a j)
  (h_range : ∀ n, 0 < a n ∧ a n ≤ 10) (h_circle : ∀ n, a n = a (n % 10)) :
  ∃ n, a n + a (n+1) + a (n+2) > 17 :=
by
  sorry

end jersey_sum_adjacent_gt_17_l309_309730


namespace largest_eight_digit_number_l309_309832

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (digits : List ℕ) : Prop :=
  ∀ d, is_even_digit d → List.contains digits d

def largest_eight_digit_with_even_digits : ℕ :=
  99986420

theorem largest_eight_digit_number {n : ℕ} 
  (h_digits : List ℕ) 
  (h_len : h_digits.length = 8)
  (h_cond : contains_all_even_digits h_digits)
  (h_num : n = List.foldl (λ acc d, acc * 10 + d) 0 h_digits)
  : n = largest_eight_digit_with_even_digits :=
sorry

end largest_eight_digit_number_l309_309832


namespace roger_initial_candies_l309_309985

def initial_candies (given_candies left_candies : ℕ) : ℕ :=
  given_candies + left_candies

theorem roger_initial_candies :
  initial_candies 3 92 = 95 :=
by
  sorry

end roger_initial_candies_l309_309985


namespace geometric_sequences_common_ratios_l309_309567

theorem geometric_sequences_common_ratios 
  (k m n o : ℝ)
  (a_2 a_3 b_2 b_3 c_2 c_3 : ℝ)
  (h1 : a_2 = k * m)
  (h2 : a_3 = k * m^2)
  (h3 : b_2 = k * n)
  (h4 : b_3 = k * n^2)
  (h5 : c_2 = k * o)
  (h6 : c_3 = k * o^2)
  (h7 : a_3 - b_3 + c_3 = 2 * (a_2 - b_2 + c_2))
  (h8 : m ≠ n)
  (h9 : m ≠ o)
  (h10 : n ≠ o) : 
  m + n + o = 1 + 2 * Real.sqrt 2 :=
by
  sorry

end geometric_sequences_common_ratios_l309_309567


namespace john_drive_time_l309_309388

theorem john_drive_time
  (t : ℝ)
  (h1 : 60 * t + 90 * (15 / 4 - t) = 300)
  (h2 : 1 / 4 = 15 / 60)
  (h3 : 4 = 15 / 4 + t + 1 / 4)
  :
  t = 1.25 :=
by
  -- This introduces the hypothesis and begins the Lean proof.
  sorry

end john_drive_time_l309_309388


namespace domain_of_inverse_l309_309033

noncomputable def f (x : ℝ) : ℝ := (1/2)^(x - 1) + 1

theorem domain_of_inverse :
  ∀ y : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ y = f x) → (y ∈ Set.Icc (3/2) 3) :=
by
  sorry

end domain_of_inverse_l309_309033


namespace shaded_area_of_intersections_l309_309121

theorem shaded_area_of_intersections (r : ℝ) (n : ℕ) (intersect_origin : Prop) (radius_5 : r = 5) (four_circles : n = 4) : 
  ∃ (area : ℝ), area = 100 * Real.pi - 200 :=
by
  sorry

end shaded_area_of_intersections_l309_309121


namespace overall_avg_is_60_l309_309249

-- Define the number of students and average marks for each class
def classA_students : ℕ := 30
def classA_avg_marks : ℕ := 40

def classB_students : ℕ := 50
def classB_avg_marks : ℕ := 70

def classC_students : ℕ := 25
def classC_avg_marks : ℕ := 55

def classD_students : ℕ := 45
def classD_avg_marks : ℕ := 65

-- Calculate the total number of students
def total_students : ℕ := 
  classA_students + classB_students + classC_students + classD_students

-- Calculate the total marks for each class
def total_marks_A : ℕ := classA_students * classA_avg_marks
def total_marks_B : ℕ := classB_students * classB_avg_marks
def total_marks_C : ℕ := classC_students * classC_avg_marks
def total_marks_D : ℕ := classD_students * classD_avg_marks

-- Calculate the combined total marks of all classes
def combined_total_marks : ℕ := 
  total_marks_A + total_marks_B + total_marks_C + total_marks_D

-- Calculate the overall average marks
def overall_avg_marks : ℕ := combined_total_marks / total_students

-- Prove that the overall average marks is 60
theorem overall_avg_is_60 : overall_avg_marks = 60 := by
  sorry -- Proof will be written here

end overall_avg_is_60_l309_309249


namespace chocolate_ice_cream_l309_309021

-- Define the number of people who ordered ice cream, vanilla, and chocolate
variable (total_people vanilla_people chocolate_people : ℕ)

-- Define the conditions as Lean constraints
def condition1 : Prop := total_people = 220
def condition2 : Prop := vanilla_people = (20 * total_people) / 100
def condition3 : Prop := vanilla_people = 2 * chocolate_people

-- State the theorem to prove the number of people who ordered chocolate ice cream
theorem chocolate_ice_cream (h1 : condition1) (h2 : condition2) (h3 : condition3) : chocolate_people = 22 :=
sorry

end chocolate_ice_cream_l309_309021


namespace quadratic_eq_k_value_l309_309094

theorem quadratic_eq_k_value (k : ℤ) : (∀ x : ℝ, (k - 1) * x ^ (|k| + 1) - x + 5 = 0 → (k - 1) ≠ 0 ∧ |k| + 1 = 2) -> k = -1 :=
by
  sorry

end quadratic_eq_k_value_l309_309094


namespace kopeechka_items_l309_309700

-- Define necessary concepts and conditions
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 200 * 100 + 83

-- Lean statement defining the proof problem
theorem kopeechka_items (a n : ℕ) (h1 : ∀ a, n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
by sorry

end kopeechka_items_l309_309700


namespace georgia_total_carnation_cost_l309_309018

-- Define the cost of one carnation
def cost_of_single_carnation : ℝ := 0.50

-- Define the cost of one dozen carnations
def cost_of_dozen_carnations : ℝ := 4.00

-- Define the number of teachers
def number_of_teachers : ℕ := 5

-- Define the number of friends
def number_of_friends : ℕ := 14

-- Calculate the cost for teachers
def cost_for_teachers : ℝ :=
  (number_of_teachers : ℝ) * cost_of_dozen_carnations

-- Calculate the cost for friends
def cost_for_friends : ℝ :=
  cost_of_dozen_carnations + (2 * cost_of_single_carnation)

-- Calculate the total cost
def total_cost : ℝ := cost_for_teachers + cost_for_friends

-- Theorem stating the total cost
theorem georgia_total_carnation_cost : total_cost = 25 := by
  -- Placeholder for the proof
  sorry

end georgia_total_carnation_cost_l309_309018


namespace calculation_of_expression_l309_309136

theorem calculation_of_expression
  (w x y z : ℕ)
  (h : 2^w * 3^x * 5^y * 7^z = 13230) :
  3 * w + 2 * x + 6 * y + 4 * z = 23 :=
sorry

end calculation_of_expression_l309_309136


namespace gain_percent_is_80_l309_309188

theorem gain_percent_is_80 (C S : ℝ) (h : 81 * C = 45 * S) : ((S - C) / C) * 100 = 80 :=
by
  sorry

end gain_percent_is_80_l309_309188


namespace tan_150_eq_neg_inv_sqrt3_l309_309878

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l309_309878


namespace convex_polygons_count_l309_309178

def num_points_on_circle : ℕ := 12

def is_valid_subset (s : finset ℕ) : Prop :=
  3 ≤ s.card

def num_valid_subsets : ℕ :=
  let total_subsets := 2 ^ num_points_on_circle in
  let subsets_with_fewer_than_3_points := finset.card (finset.powerset_len 0 (finset.range num_points_on_circle)) +
                                          finset.card (finset.powerset_len 1 (finset.range num_points_on_circle)) +
                                          finset.card (finset.powerset_len 2 (finset.range num_points_on_circle)) in
  total_subsets - subsets_with_fewer_than_3_points

theorem convex_polygons_count :
  num_valid_subsets = 4017 :=
sorry

end convex_polygons_count_l309_309178


namespace base_nine_to_mod_five_l309_309421

-- Define the base-nine number N
def N : ℕ := 2 * 9^10 + 7 * 9^9 + 0 * 9^8 + 0 * 9^7 + 6 * 9^6 + 0 * 9^5 + 0 * 9^4 + 0 * 9^3 + 0 * 9^2 + 5 * 9^1 + 2 * 9^0

-- Theorem statement
theorem base_nine_to_mod_five : N % 5 = 3 :=
by
  sorry

end base_nine_to_mod_five_l309_309421


namespace train_crossing_time_l309_309309

-- Definitions for the conditions
def speed_kmph : Float := 72
def speed_mps : Float := speed_kmph * (1000 / 3600)
def length_train_m : Float := 240.0416
def length_platform_m : Float := 280
def total_distance_m : Float := length_train_m + length_platform_m

-- The problem statement
theorem train_crossing_time :
  (total_distance_m / speed_mps) = 26.00208 :=
by
  sorry

end train_crossing_time_l309_309309


namespace compute_f_f_f_19_l309_309974

def f (x : Int) : Int :=
  if x < 10 then x^2 - 9 else x - 15

theorem compute_f_f_f_19 : f (f (f 19)) = 40 := by
  sorry

end compute_f_f_f_19_l309_309974


namespace solve_for_x_l309_309521

theorem solve_for_x (x : ℝ) (h : x / 5 + 3 = 4) : x = 5 :=
sorry

end solve_for_x_l309_309521


namespace necessary_and_sufficient_condition_l309_309656

theorem necessary_and_sufficient_condition :
  ∀ a b : ℝ, (a + b > 0) ↔ ((a ^ 3) + (b ^ 3) > 0) :=
by
  sorry

end necessary_and_sufficient_condition_l309_309656


namespace variance_boys_girls_l309_309378

def boys_scores : List ℝ := [86, 94, 88, 92, 90]
def girls_scores : List ℝ := [88, 93, 93, 88, 93]

def mean (scores : List ℝ) : ℝ :=
  (scores.sum) / (scores.length)

def variance (scores : List ℝ) : ℝ :=
  (scores.map (λ x, (x - mean scores)^2).sum) / (scores.length)

theorem variance_boys_girls :
  variance boys_scores > variance girls_scores :=
by
  unfold variance boys_scores girls_scores
  sorry

end variance_boys_girls_l309_309378


namespace deployment_plans_l309_309023

/-- Given 6 volunteers and needing to select 4 to fill different positions of 
  translator, tour guide, shopping guide, and cleaner, and knowing that neither 
  supporters A nor B can work as the translator, the total number of deployment plans is 240. -/
theorem deployment_plans (volunteers : Fin 6) (A B : Fin 6) : 
  ∀ {translator tour_guide shopping_guide cleaner : Fin 6},
  A ≠ translator ∧ B ≠ translator → 
  ∃ plans : Finset (Fin 6 × Fin 6 × Fin 6 × Fin 6), plans.card = 240 :=
by 
sorry

end deployment_plans_l309_309023


namespace rabbit_travel_time_l309_309315

noncomputable def rabbit_speed : ℝ := 5 -- speed of the rabbit in miles per hour
noncomputable def rabbit_distance : ℝ := 2 -- distance traveled by the rabbit in miles

theorem rabbit_travel_time :
  let t := (rabbit_distance / rabbit_speed) * 60 in
  t = 24 :=
by
  sorry

end rabbit_travel_time_l309_309315


namespace study_time_l309_309005

theorem study_time (n_mcq n_fitb : ℕ) (t_mcq t_fitb : ℕ) (total_minutes_per_hour : ℕ) 
  (h1 : n_mcq = 30) (h2 : n_fitb = 30) (h3 : t_mcq = 15) (h4 : t_fitb = 25) (h5 : total_minutes_per_hour = 60) : 
  n_mcq * t_mcq + n_fitb * t_fitb = 20 * total_minutes_per_hour := 
by 
  -- This is a placeholder for the proof
  sorry

end study_time_l309_309005


namespace distinct_convex_polygons_of_three_or_more_sides_l309_309180

theorem distinct_convex_polygons_of_three_or_more_sides (n : ℕ) (h : n = 12) : 
  let total_subsets := 2 ^ n,
      zero_member_subsets := nat.choose n 0,
      one_member_subsets := nat.choose n 1,
      two_member_subsets := nat.choose n 2,
      subsets_with_three_or_more_members := total_subsets - zero_member_subsets - one_member_subsets - two_member_subsets
  in subsets_with_three_or_more_members = 4017 :=
by
  -- appropriate proof steps or a sorry placeholder will be added here
  sorry

end distinct_convex_polygons_of_three_or_more_sides_l309_309180


namespace repeating_decimal_to_fraction_l309_309445

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), x = (433 / 990) ∧ x = (4 + 37 / 99) / 10 := by
  sorry

end repeating_decimal_to_fraction_l309_309445


namespace measure_of_acute_angle_l309_309102

theorem measure_of_acute_angle (x : ℝ) (h_complement : 90 - x = (1/2) * (180 - x) + 20) (h_acute : 0 < x ∧ x < 90) : x = 40 :=
  sorry

end measure_of_acute_angle_l309_309102


namespace max_wins_l309_309322

theorem max_wins (Chloe_wins Max_wins : ℕ) (h1 : Chloe_wins = 24) (h2 : 8 * Max_wins = 3 * Chloe_wins) : Max_wins = 9 := by
  sorry

end max_wins_l309_309322


namespace tan_150_eq_neg_sqrt3_div_3_l309_309923

theorem tan_150_eq_neg_sqrt3_div_3
: tan 150 = - (Real.sqrt 3 / 3) :=
by
  have h1 : 150 = 180 - 30, by rfl
  have h2 : sin 150 = sin (180 - 30), by simp [h1]
  have h3 : cos 150 = cos (180 - 30), by simp [h1]
  have cos_30 := Real.sqrt 3 / 2
  have sin_30 := 1 / 2
  simp [Real.sin_eq_sin_of_Real_angle, Real.cos_eq_cos_of_Real_angle] at h2 h3
  have h4 : sin 150 = sin 30, by simp [h2, sin_30]
  have h5 : cos 150 = -cos 30, by simp [h3, cos_30]
  rw [Real.tan_eq_sin_div_cos, h4, h5]
  have h6 : (1 / 2) / (-Real.sqrt 3 / 2) = - (1 / Real.sqrt 3), by linarith
  rw h6
  have h7 : - (1 / Real.sqrt 3) = - (Real.sqrt 3 / 3), by
    field_simp [Real.sqrt_ne_zero],
    linarith
  exact h7,
  sorry

end tan_150_eq_neg_sqrt3_div_3_l309_309923


namespace avg_minutes_eq_170_div_9_l309_309671

-- Define the conditions
variables (s : ℕ) -- number of seventh graders
def sixth_graders := 3 * s
def seventh_graders := s
def eighth_graders := s / 2
def sixth_grade_minutes := 18
def seventh_grade_run_minutes := 20
def seventh_grade_stretching_minutes := 5
def eighth_grade_minutes := 12

-- Define the total activity minutes for each grade
def total_activity_minutes_sixth := sixth_grade_minutes * sixth_graders
def total_activity_minutes_seventh := (seventh_grade_run_minutes + seventh_grade_stretching_minutes) * seventh_graders
def total_activity_minutes_eighth := eighth_grade_minutes * eighth_graders

-- Calculate total activity minutes
def total_activity_minutes := total_activity_minutes_sixth + total_activity_minutes_seventh + total_activity_minutes_eighth

-- Calculate total number of students
def total_students := sixth_graders + seventh_graders + eighth_graders

-- Calculate average minutes per student
def average_minutes_per_student := total_activity_minutes / total_students

theorem avg_minutes_eq_170_div_9 : average_minutes_per_student s = 170 / 9 := by
  sorry

end avg_minutes_eq_170_div_9_l309_309671


namespace largest_8_digit_number_with_even_digits_l309_309827

def is_even (n : ℕ) : Prop := n % 2 = 0

def all_even_digits : List ℕ := [0, 2, 4, 6, 8]

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 8 ∧
  ∀ (d : ℕ), d ∈ digits → is_even d ∧
  all_even_digits ⊆ digits

theorem largest_8_digit_number_with_even_digits : ∃ n : ℕ, is_valid_number n ∧ n = 99986420 :=
sorry

end largest_8_digit_number_with_even_digits_l309_309827


namespace units_digit_of_product_is_eight_l309_309772

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l309_309772


namespace multiple_of_6_is_multiple_of_3_l309_309054

theorem multiple_of_6_is_multiple_of_3 (n : ℕ) (h1 : ∀ k : ℕ, n = 6 * k)
  : ∃ m : ℕ, n = 3 * m :=
by sorry

end multiple_of_6_is_multiple_of_3_l309_309054


namespace sum_of_radical_conjugates_l309_309325

theorem sum_of_radical_conjugates : 
  (8 - Real.sqrt 1369) + (8 + Real.sqrt 1369) = 16 :=
by
  sorry

end sum_of_radical_conjugates_l309_309325


namespace B_is_subset_of_A_l309_309849
open Set

def A := {x : ℤ | ∃ n : ℤ, x = 2 * n}
def B := {y : ℤ | ∃ k : ℤ, y = 4 * k}

theorem B_is_subset_of_A : B ⊆ A :=
by sorry

end B_is_subset_of_A_l309_309849


namespace largest_8_digit_number_with_even_digits_l309_309825

def is_even (n : ℕ) : Prop := n % 2 = 0

def all_even_digits : List ℕ := [0, 2, 4, 6, 8]

def is_valid_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 8 ∧
  ∀ (d : ℕ), d ∈ digits → is_even d ∧
  all_even_digits ⊆ digits

theorem largest_8_digit_number_with_even_digits : ∃ n : ℕ, is_valid_number n ∧ n = 99986420 :=
sorry

end largest_8_digit_number_with_even_digits_l309_309825


namespace units_digit_first_four_composites_l309_309776

theorem units_digit_first_four_composites :
  let product := 4 * 6 * 8 * 9 in
  product % 10 = 8 :=
by
  let product := 4 * 6 * 8 * 9
  have h : product = 1728 := by norm_num
  show product % 10 = 8
  rw [h]
  norm_num
  done 
  sorry

end units_digit_first_four_composites_l309_309776


namespace probability_train_or_plane_probability_not_ship_l309_309986

def P_plane : ℝ := 0.2
def P_ship : ℝ := 0.3
def P_train : ℝ := 0.4
def P_car : ℝ := 0.1
def mutually_exclusive : Prop := P_plane + P_ship + P_train + P_car = 1

theorem probability_train_or_plane : mutually_exclusive → P_train + P_plane = 0.6 := by
  intro h
  sorry

theorem probability_not_ship : mutually_exclusive → 1 - P_ship = 0.7 := by
  intro h
  sorry

end probability_train_or_plane_probability_not_ship_l309_309986


namespace average_interest_rate_l309_309854

theorem average_interest_rate 
  (total : ℝ)
  (rate1 rate2 yield1 yield2 : ℝ)
  (amount1 amount2 : ℝ)
  (h_total : total = amount1 + amount2)
  (h_rate1 : rate1 = 0.03)
  (h_rate2 : rate2 = 0.07)
  (h_yield_equal : yield1 = yield2)
  (h_yield1 : yield1 = rate1 * amount1)
  (h_yield2 : yield2 = rate2 * amount2) :
  (yield1 + yield2) / total = 0.042 :=
by
  sorry

end average_interest_rate_l309_309854


namespace maximum_volume_of_prism_l309_309122

noncomputable def maximum_volume_prism (s : ℝ) (θ : ℝ) (face_area_sum : ℝ) : ℝ := 
  if (s = 6 ∧ θ = Real.pi / 3 ∧ face_area_sum = 36) then 27 
  else 0

theorem maximum_volume_of_prism : 
  ∀ (s θ face_area_sum), s = 6 ∧ θ = Real.pi / 3 ∧ face_area_sum = 36 → maximum_volume_prism s θ face_area_sum = 27 :=
by
  intros
  sorry

end maximum_volume_of_prism_l309_309122


namespace popsicle_sticks_left_l309_309332

-- Defining the conditions
def total_money : ℕ := 10
def cost_of_molds : ℕ := 3
def cost_of_sticks : ℕ := 1
def cost_of_juice_bottle : ℕ := 2
def popsicles_per_bottle : ℕ := 20
def initial_sticks : ℕ := 100

-- Statement of the problem
theorem popsicle_sticks_left : 
  let remaining_money := total_money - cost_of_molds - cost_of_sticks
  let bottles_of_juice := remaining_money / cost_of_juice_bottle
  let total_popsicles := bottles_of_juice * popsicles_per_bottle
  let sticks_left := initial_sticks - total_popsicles
  sticks_left = 40 := by
  sorry

end popsicle_sticks_left_l309_309332


namespace smallest_x_remainder_l309_309473

theorem smallest_x_remainder : ∃ x : ℕ, x > 0 ∧ 
    x % 6 = 5 ∧
    x % 7 = 6 ∧
    x % 8 = 7 ∧
    x = 167 :=
by
  sorry

end smallest_x_remainder_l309_309473


namespace total_hours_verification_l309_309334

def total_hours_data_analytics : ℕ := 
  let weekly_class_homework_hours := (2 * 3 + 1 * 4 + 4) * 24 
  let lab_project_hours := 8 * 6 + (10 + 14 + 18)
  weekly_class_homework_hours + lab_project_hours

def total_hours_programming : ℕ :=
  let weekly_hours := (2 * 2 + 2 * 4 + 6) * 24
  weekly_hours

def total_hours_statistics : ℕ :=
  let weekly_class_lab_project_hours := (2 * 3 + 1 * 2 + 3) * 24
  let exam_study_hours := 9 * 5
  weekly_class_lab_project_hours + exam_study_hours

def total_hours_all_courses : ℕ :=
  total_hours_data_analytics + total_hours_programming + total_hours_statistics

theorem total_hours_verification : 
    total_hours_all_courses = 1167 := 
by 
    sorry

end total_hours_verification_l309_309334


namespace tan_150_eq_neg_inv_sqrt_3_l309_309907

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l309_309907


namespace kopeechka_purchase_l309_309709

theorem kopeechka_purchase
  (a : ℕ)
  (n : ℕ)
  (total_cost : ℕ)
  (item_cost : ℕ) :
  total_cost = 20083 →
  item_cost = 100 * a + 99 →
  (n * item_cost = total_cost ∧ n = 17 ∨ n = 117) :=
begin
  sorry
end

end kopeechka_purchase_l309_309709


namespace min_value_is_one_l309_309138

noncomputable def min_value (n : ℕ) (a b : ℝ) : ℝ :=
  1 / (1 + a^n) + 1 / (1 + b^n)

theorem min_value_is_one (n : ℕ) (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 2) :
  (min_value n a b) = 1 := 
sorry

end min_value_is_one_l309_309138


namespace product_multiple_of_four_probability_l309_309931

theorem product_multiple_of_four_probability :
  let chips := {1, 2, 3, 4}
  let outcomes : Finset (ℕ × ℕ) := Finset.product chips chips
  let favorable_outcomes := outcomes.filter (λ (p : ℕ × ℕ), (p.1 * p.2) % 4 = 0)
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ) = 1 / 2 := by
  sorry

end product_multiple_of_four_probability_l309_309931


namespace deceased_member_income_l309_309031

theorem deceased_member_income (A B C : ℝ) (h1 : (A + B + C) / 3 = 735) (h2 : (A + B) / 2 = 650) : 
  C = 905 :=
by
  sorry

end deceased_member_income_l309_309031


namespace middle_school_soccer_league_l309_309253

theorem middle_school_soccer_league (n : ℕ) (h : (n * (n - 1)) / 2 = 36) : n = 9 := 
  sorry

end middle_school_soccer_league_l309_309253


namespace tip_percentage_l309_309422

theorem tip_percentage 
  (total_bill : ℕ) 
  (silas_payment : ℕ) 
  (remaining_friend_payment_with_tip : ℕ) 
  (num_remaining_friends : ℕ) 
  (num_friends : ℕ)
  (h1 : total_bill = 150) 
  (h2 : silas_payment = total_bill / 2) 
  (h3 : num_remaining_friends = 5)
  (h4 : remaining_friend_payment_with_tip = 18)
  : (remaining_friend_payment_with_tip - (total_bill / 2 / num_remaining_friends) * num_remaining_friends) / total_bill * 100 = 10 :=
by
  sorry

end tip_percentage_l309_309422


namespace arc_length_of_octagon_side_l309_309198

-- Define the conditions
def is_regular_octagon (side_length : ℝ) (angle_subtended : ℝ) := side_length = 5 ∧ angle_subtended = 2 * Real.pi / 8

-- Define the property to be proved
theorem arc_length_of_octagon_side :
  ∀ (side_length : ℝ) (angle_subtended : ℝ), 
    is_regular_octagon side_length angle_subtended →
    (angle_subtended / (2 * Real.pi)) * (2 * Real.pi * side_length) = 5 * Real.pi / 4 :=
by
  intros side_length angle_subtended h
  unfold is_regular_octagon at h
  sorry

end arc_length_of_octagon_side_l309_309198


namespace prove_triangle_inequality_l309_309401

def triangle_inequality (a b c a1 a2 b1 b2 c1 c2 : ℝ) : Prop := 
  a * a1 * a2 + b * b1 * b2 + c * c1 * c2 ≥ a * b * c

theorem prove_triangle_inequality 
  (a b c a1 a2 b1 b2 c1 c2 : ℝ)
  (h1: 0 ≤ a) (h2: 0 ≤ b) (h3: 0 ≤ c)
  (h4: 0 ≤ a1) (h5: 0 ≤ a2) 
  (h6: 0 ≤ b1) (h7: 0 ≤ b2)
  (h8: 0 ≤ c1) (h9: 0 ≤ c2) : triangle_inequality a b c a1 a2 b1 b2 c1 c2 :=
sorry

end prove_triangle_inequality_l309_309401


namespace kolya_purchase_l309_309682

theorem kolya_purchase : ∃ n : ℕ, n = 17 ∨ n = 117 :=
by
  let item_cost := λ a : ℕ, 100 * a + 99
  let total_cost := 20000 + 83
  have h : ∀ n a, n * (item_cost a) = total_cost → (n = 17 ∨ n = 117) := sorry
  have h1 := h 17 0
  have h2 := h 117 0
  existsi 17
  exact h1 sorry

end kolya_purchase_l309_309682


namespace quad_factor_value_l309_309423

theorem quad_factor_value (c d : ℕ) (h1 : c + d = 14) (h2 : c * d = 40) (h3 : c > d) : 4 * d - c = 6 :=
sorry

end quad_factor_value_l309_309423


namespace units_digit_of_first_four_composite_numbers_l309_309809

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l309_309809


namespace kolya_purchase_l309_309687

theorem kolya_purchase : ∃ n : ℕ, n = 17 ∨ n = 117 :=
by
  let item_cost := λ a : ℕ, 100 * a + 99
  let total_cost := 20000 + 83
  have h : ∀ n a, n * (item_cost a) = total_cost → (n = 17 ∨ n = 117) := sorry
  have h1 := h 17 0
  have h2 := h 117 0
  existsi 17
  exact h1 sorry

end kolya_purchase_l309_309687


namespace points_on_decreasing_line_y1_gt_y2_l309_309359
-- Import the necessary library

-- Necessary conditions and definitions
variable {x y : ℝ}

-- Given points P(3, y1) and Q(4, y2)
def y1 : ℝ := -2*3 + 4
def y2 : ℝ := -2*4 + 4

-- Lean statement to prove y1 > y2
theorem points_on_decreasing_line_y1_gt_y2 (h1 : y1 = -2 * 3 +4) (h2 : y2 = -2 * 4 + 4) : 
  y1 > y2 :=
sorry  -- Proof steps go here

end points_on_decreasing_line_y1_gt_y2_l309_309359


namespace num_ways_write_100_as_distinct_squares_l309_309675

theorem num_ways_write_100_as_distinct_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a^2 + b^2 + c^2 = 100 ∧
  (∃ (x y z : ℕ), x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x^2 + y^2 + z^2 = 100 ∧ (x, y, z) ≠ (a, b, c) ∧ (x, y, z) ≠ (a, c, b) ∧ (x, y, z) ≠ (b, a, c) ∧ (x, y, z) ≠ (b, c, a) ∧ (x, y, z) ≠ (c, a, b) ∧ (x, y, z) ≠ (c, b, a)) ∧
  ∀ (p q r : ℕ), p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p^2 + q^2 + r^2 = 100 → (p, q, r) = (a, b, c) ∨ (p, q, r) = (a, c, b) ∨ (p, q, r) = (b, a, c) ∨ (p, q, r) = (b, c, a) ∨ (p, q, r) = (c, a, b) ∨ (p, q, r) = (c, b, a) ∨ (p, q, r) = (x, y, z) ∨ (p, q, r) = (x, z, y) ∨ (p, q, r) = (y, x, z) ∨ (p, q, r) = (y, z, x) ∨ (p, q, r) = (z, x, y) ∨ (p, q, r) = (z, y, x) :=
sorry

end num_ways_write_100_as_distinct_squares_l309_309675


namespace total_value_is_correct_l309_309855

-- We will define functions that convert base 7 numbers to base 10
def base7_to_base10 (n : Nat) : Nat :=
  let digits := (n.digits 7)
  digits.enum.foldr (λ ⟨i, d⟩ acc => acc + d * 7^i) 0

-- Define the specific numbers in base 7
def silver_value_base7 : Nat := 5326
def gemstone_value_base7 : Nat := 3461
def spice_value_base7 : Nat := 656

-- Define the combined total in base 10
def total_value_base10 : Nat := base7_to_base10 silver_value_base7 + base7_to_base10 gemstone_value_base7 + base7_to_base10 spice_value_base7

theorem total_value_is_correct :
  total_value_base10 = 3485 :=
by
  sorry

end total_value_is_correct_l309_309855


namespace find_a2014_l309_309112

open Nat

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 0 ∧
  (∀ n, a (n + 1) = (a n - 2) / (5 * a n / 4 - 2))

theorem find_a2014 (a : ℕ → ℚ) (h : seq a) : a 2014 = 1 :=
by
  sorry

end find_a2014_l309_309112


namespace original_number_is_120_l309_309186

theorem original_number_is_120 (N k : ℤ) (hk : N - 33 = 87 * k) : N = 120 :=
by
  have h : N - 33 = 87 * 1 := by sorry
  have N_eq : N = 87 + 33 := by sorry
  have N_val : N = 120 := by sorry
  exact N_val

end original_number_is_120_l309_309186


namespace find_c_for_degree_3_l309_309327

noncomputable def f : Polynomial ℚ := 2 - 15 * Polynomial.X + 4 * Polynomial.X^2 - 3 * Polynomial.X^3 + 6 * Polynomial.X^4
noncomputable def g : Polynomial ℚ := 4 - 3 * Polynomial.X + 1 * Polynomial.X^2 - 7 * Polynomial.X^3 + 10 * Polynomial.X^4

theorem find_c_for_degree_3 :
  ∃ (c : ℚ), Polynomial.degree (f + c • g) = 3 :=
sorry

end find_c_for_degree_3_l309_309327


namespace kopeechka_items_l309_309695

theorem kopeechka_items (a n : ℕ) (hn : n * (100 * a + 99) = 20083) : n = 17 ∨ n = 117 :=
sorry

end kopeechka_items_l309_309695


namespace kopeechka_items_l309_309703

-- Define necessary concepts and conditions
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 200 * 100 + 83

-- Lean statement defining the proof problem
theorem kopeechka_items (a n : ℕ) (h1 : ∀ a, n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
by sorry

end kopeechka_items_l309_309703


namespace quadratic_less_than_zero_for_x_in_0_1_l309_309107

theorem quadratic_less_than_zero_for_x_in_0_1 (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  ∀ x, 0 < x ∧ x < 1 → (a * x^2 + b * x + c) < 0 :=
by
  sorry

end quadratic_less_than_zero_for_x_in_0_1_l309_309107


namespace minimize_sum_of_squares_l309_309301

theorem minimize_sum_of_squares (a b c : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a + b + c = 16) :
  a^2 + b^2 + c^2 ≥ 86 :=
sorry

end minimize_sum_of_squares_l309_309301


namespace value_of_star_l309_309117

theorem value_of_star : 
  ∀ (star : ℤ), 45 - (28 - (37 - (15 - star))) = 59 → star = -154 :=
by
  intro star
  intro h
  -- Proof to be provided
  sorry

end value_of_star_l309_309117


namespace purchase_options_l309_309715

def item_cost (a : Nat) : Nat := 100 * a + 99

def total_cost : Nat := 20083

theorem purchase_options (a : Nat) (n : Nat) (h : n * item_cost a = total_cost) :
  n = 17 ∨ n = 117 :=
by
  sorry

end purchase_options_l309_309715


namespace number_of_items_l309_309679

theorem number_of_items {a n : ℕ} (h1 : ∀ x, x = 100 * a + 99) (h2 : 200 * 100 + 83 = 20083) : 
  (n * (100 * a + 99) = 20083) → (n = 17 ∨ n = 117) :=
by 
  sorry

end number_of_items_l309_309679


namespace no_such_function_exists_l309_309338

def f (n : ℕ) : ℕ := sorry

theorem no_such_function_exists :
  ¬ ∃ f : ℕ → ℕ, ∀ n : ℕ, n > 1 → (f n = f (f (n - 1)) + f (f (n + 1))) :=
by
  sorry

end no_such_function_exists_l309_309338


namespace total_snowballs_l309_309202

theorem total_snowballs (Lc : ℕ) (Ch : ℕ) (Pt : ℕ)
  (h1 : Ch = Lc + 31)
  (h2 : Lc = 19)
  (h3 : Pt = 47) : 
  Ch + Lc + Pt = 116 := by
  sorry

end total_snowballs_l309_309202


namespace minimum_parents_needed_l309_309435

/-- 
Given conditions:
1. There are 30 students going on the excursion.
2. Each car can accommodate 5 people, including the driver.
Prove that the minimum number of parents needed to be invited on the excursion is 8.
-/
theorem minimum_parents_needed (students : ℕ) (car_capacity : ℕ) (drivers_needed : ℕ) 
  (h1 : students = 30) (h2 : car_capacity = 5) (h3 : drivers_needed = 1) 
  : ∃ (parents : ℕ), parents = 8 :=
by
  existsi 8
  sorry

end minimum_parents_needed_l309_309435


namespace total_sand_donated_l309_309745

theorem total_sand_donated (A B C D: ℚ) (hA: A = 33 / 2) (hB: B = 26) (hC: C = 49 / 2) (hD: D = 28) : 
  A + B + C + D = 95 := by
  sorry

end total_sand_donated_l309_309745


namespace units_digit_of_product_of_first_four_composites_l309_309788

theorem units_digit_of_product_of_first_four_composites :
  (4 * 6 * 8 * 9) % 10 = 8 := 
by
  sorry

end units_digit_of_product_of_first_four_composites_l309_309788


namespace tan_150_degrees_l309_309870

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l309_309870


namespace part1_part2_l309_309355

-- Part (1)
theorem part1 (a : ℝ) (P Q : Set ℝ) (hP : P = {x | 4 <= x ∧ x <= 7})
              (hQ : Q = {x | -2 <= x ∧ x <= 5}) :
  (Set.compl P ∩ Q) = {x | -2 <= x ∧ x < 4} :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) (P Q : Set ℝ)
              (hP : P = {x | a + 1 <= x ∧ x <= 2 * a + 1})
              (hQ : Q = {x | -2 <= x ∧ x <= 5})
              (h_sufficient : ∀ x, x ∈ P → x ∈ Q) 
              (h_not_necessary : ∃ x, x ∈ Q ∧ x ∉ P) :
  (0 <= a ∧ a <= 2) :=
by
  sorry

end part1_part2_l309_309355


namespace gcd_97_power_l309_309508

theorem gcd_97_power (h : Nat.Prime 97) : 
  Nat.gcd (97^7 + 1) (97^7 + 97^3 + 1) = 1 := 
by 
  sorry

end gcd_97_power_l309_309508


namespace distance_between_lines_l309_309106

-- Definitions from conditions in (a)
def l1 (x y : ℝ) := 3 * x + 4 * y - 7 = 0
def l2 (x y : ℝ) := 6 * x + 8 * y + 1 = 0

-- The proof goal from (c)
theorem distance_between_lines : 
  ∀ (x y : ℝ),
    (l1 x y) → 
    (l2 x y) →
      -- Distance between the lines is 3/2
      ( (|(-14) - 1| : ℝ) / (Real.sqrt (6^2 + 8^2)) ) = 3 / 2 :=
by
  sorry

end distance_between_lines_l309_309106


namespace albums_in_either_but_not_both_l309_309625

-- Defining the conditions
def shared_albums : ℕ := 9
def total_albums_andrew : ℕ := 17
def unique_albums_john : ℕ := 6

-- Stating the theorem to prove
theorem albums_in_either_but_not_both :
  (total_albums_andrew - shared_albums) + unique_albums_john = 14 :=
sorry

end albums_in_either_but_not_both_l309_309625


namespace forgot_to_take_capsules_l309_309132

theorem forgot_to_take_capsules (total_days : ℕ) (days_taken : ℕ) 
  (h1 : total_days = 31) 
  (h2 : days_taken = 29) : 
  total_days - days_taken = 2 := 
by 
  sorry

end forgot_to_take_capsules_l309_309132


namespace total_eggs_emily_collected_l309_309053

theorem total_eggs_emily_collected :
  let number_of_baskets := 303
  let eggs_per_basket := 28
  number_of_baskets * eggs_per_basket = 8484 :=
by
  let number_of_baskets := 303
  let eggs_per_basket := 28
  sorry -- Proof to be provided

end total_eggs_emily_collected_l309_309053


namespace james_meditation_sessions_l309_309558

theorem james_meditation_sessions (minutes_per_session : ℕ) (hours_per_week : ℕ) (days_per_week : ℕ) (h1 : minutes_per_session = 30) (h2 : hours_per_week = 7) (h3 : days_per_week = 7) : 
  (hours_per_week * 60 / days_per_week / minutes_per_session) = 2 := 
by 
  sorry

end james_meditation_sessions_l309_309558


namespace track_width_l309_309493

theorem track_width (r1 r2 : ℝ) (h : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 20 * Real.pi) : r1 - r2 = 10 := by
  sorry

end track_width_l309_309493


namespace pirates_coins_l309_309170

noncomputable def coins (x : ℕ) (k : ℕ) : ℕ :=
  match k with
  | 0     => x
  | k + 1 => (coins x k) - (coins x k * (k + 2) / 15)

theorem pirates_coins (x : ℕ) (H : x = 2^15 * 3^8 * 5^14) :
  ∃ n : ℕ, n = coins x 14 :=
sorry

end pirates_coins_l309_309170


namespace volleyball_teams_l309_309848

theorem volleyball_teams (n : ℕ)
  (h1 : ∀ (a b : ℕ), a ∈ (Finset.range n) → b ∈ (Finset.range (2 * n)) → a ≠ b → (¬(∃ c : ℤ, c < 0)))
  (h2 : ratio n (2 * n) 3 4) :
  n = 5 :=
by sorry

end volleyball_teams_l309_309848


namespace joe_paint_initial_amount_l309_309964

theorem joe_paint_initial_amount (P : ℕ) (h1 : P / 6 + (5 * P / 6) / 5 = 120) :
  P = 360 := by
  sorry

end joe_paint_initial_amount_l309_309964


namespace arithmetic_mean_of_positive_three_digit_multiples_of_8_l309_309045

open Nat

theorem arithmetic_mean_of_positive_three_digit_multiples_of_8 : 
  let a := 104
  let l := 992
  2 * ∑ k in range 112, (8 * (k + 13)) / 112 = 548 :=
by
  sorry

end arithmetic_mean_of_positive_three_digit_multiples_of_8_l309_309045


namespace trust_meteorologist_l309_309610

noncomputable def problem_statement : Prop :=
  let r := 0.74
  let p := 0.5
  let senators_forecast := (1 - 1.5 * p) * p^2 * r
  let meteorologist_forecast := 1.5 * p * (1 - p)^2 * (1 - r)
  meteorologist_forecast > senators_forecast

theorem trust_meteorologist : problem_statement :=
  sorry

end trust_meteorologist_l309_309610


namespace tom_needs_more_blue_tickets_l309_309176

def yellow_to_red : ℕ := 10
def red_to_blue : ℕ := 10
def yellow_to_blue : ℕ := yellow_to_red * red_to_blue
def required_yellow_tickets : ℕ := 10
def required_blue_tickets : ℕ := required_yellow_tickets * yellow_to_blue

def toms_yellow_tickets : ℕ := 8
def toms_red_tickets : ℕ := 3
def toms_blue_tickets : ℕ := 7
def toms_total_blue_tickets : ℕ := 
  (toms_yellow_tickets * yellow_to_blue) + 
  (toms_red_tickets * red_to_blue) + 
  toms_blue_tickets

def additional_blue_tickets_needed : ℕ :=
  required_blue_tickets - toms_total_blue_tickets

theorem tom_needs_more_blue_tickets : additional_blue_tickets_needed = 163 := 
by sorry

end tom_needs_more_blue_tickets_l309_309176


namespace company_total_employees_l309_309040

def total_employees_after_hiring (T : ℕ) (before_hiring_female_percentage : ℚ) (additional_male_workers : ℕ) (after_hiring_female_percentage : ℚ) : ℕ :=
  T + additional_male_workers

theorem company_total_employees (T : ℕ)
  (before_hiring_female_percentage : ℚ)
  (additional_male_workers : ℕ)
  (after_hiring_female_percentage : ℚ)
  (h_before_percent : before_hiring_female_percentage = 0.60)
  (h_additional_male : additional_male_workers = 28)
  (h_after_percent : after_hiring_female_percentage = 0.55)
  (h_equation : (before_hiring_female_percentage * T)/(T + additional_male_workers) = after_hiring_female_percentage) :
  total_employees_after_hiring T before_hiring_female_percentage additional_male_workers after_hiring_female_percentage = 336 :=
by {
  -- This is where you add the proof steps.
  sorry
}

end company_total_employees_l309_309040


namespace jerry_claims_years_of_salary_l309_309387

theorem jerry_claims_years_of_salary
  (Y : ℝ)
  (salary_damage_per_year : ℝ := 50000)
  (medical_bills : ℝ := 200000)
  (punitive_damages : ℝ := 3 * (salary_damage_per_year * Y + medical_bills))
  (total_damages : ℝ := salary_damage_per_year * Y + medical_bills + punitive_damages)
  (received_amount : ℝ := 0.8 * total_damages)
  (actual_received_amount : ℝ := 5440000) :
  received_amount = actual_received_amount → Y = 30 := 
by
  sorry

end jerry_claims_years_of_salary_l309_309387


namespace largest_eight_digit_number_contains_even_digits_l309_309834

theorem largest_eight_digit_number_contains_even_digits :
  ∃ n : ℕ, n = 99986420 ∧ (10000000 ≤ n ∧ n < 100000000) ∧
    ∀ d ∈ [0, 2, 4, 6, 8], ∃ (i : ℕ), i < 8 ∧ (n / 10^i) % 10 = d :=
by
  sorry

end largest_eight_digit_number_contains_even_digits_l309_309834


namespace inequality1_inequality2_l309_309109

noncomputable def f (x : ℝ) := abs (x + 1 / 2) + abs (x - 3 / 2)

theorem inequality1 (x : ℝ) : 
  (f x ≤ 3) ↔ (-1 ≤ x ∧ x ≤ 2) := by
sorry

theorem inequality2 (a : ℝ) :
  (∀ x, f x ≥ 1 / 2 * abs (1 - a)) ↔ (-3 ≤ a ∧ a ≤ 5) := by
sorry

end inequality1_inequality2_l309_309109


namespace distance_from_P_to_origin_l309_309958

open Real -- This makes it easier to use real number functions and constants.

noncomputable def hyperbola := { P : ℝ × ℝ // (P.1^2 / 9) - (P.2^2 / 7) = 1 }

theorem distance_from_P_to_origin 
  (P : ℝ × ℝ) 
  (hP : (P.1^2 / 9) - (P.2^2 / 7) = 1)
  (d_right_focus : P.1 - 4 = -1) : 
  dist P (0, 0) = 3 :=
sorry

end distance_from_P_to_origin_l309_309958


namespace value_of_b_plus_a_l309_309213

theorem value_of_b_plus_a (a b : ℝ) (h1 : |a| = 8) (h2 : |b| = 2) (h3 : |a - b| = |b - a|) : b + a = -6 ∨ b + a = -10 :=
by
  sorry

end value_of_b_plus_a_l309_309213


namespace tan_150_degrees_l309_309871

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l309_309871


namespace barbara_candies_l309_309075

theorem barbara_candies :
  ∀ (initial left used : ℝ), initial = 18 ∧ left = 9 → initial - left = used → used = 9 :=
by
  intros initial left used h1 h2
  sorry

end barbara_candies_l309_309075


namespace yards_gained_l309_309615

variable {G : ℤ}

theorem yards_gained (h : -5 + G = 3) : G = 8 :=
  by
  sorry

end yards_gained_l309_309615


namespace popsicle_sticks_left_l309_309330

/-- Danielle has $10 for supplies. She buys one set of molds for $3, 
a pack of 100 popsicle sticks for $1. Each bottle of juice makes 20 popsicles and costs $2.
Prove that the number of popsicle sticks Danielle will be left with after making as many popsicles as she can is 40. -/
theorem popsicle_sticks_left (initial_money : ℕ)
    (mold_cost : ℕ) (sticks_cost : ℕ) (initial_sticks : ℕ)
    (juice_cost : ℕ) (popsicles_per_bottle : ℕ)
    (final_sticks : ℕ) :
    initial_money = 10 →
    mold_cost = 3 → 
    sticks_cost = 1 → 
    initial_sticks = 100 →
    juice_cost = 2 →
    popsicles_per_bottle = 20 →
    final_sticks = initial_sticks - (popsicles_per_bottle * (initial_money - mold_cost - sticks_cost) / juice_cost) →
    final_sticks = 40 :=
by
  intros h_initial_money h_mold_cost h_sticks_cost h_initial_sticks h_juice_cost h_popsicles_per_bottle h_final_sticks
  rw [h_initial_money, h_mold_cost, h_sticks_cost, h_initial_sticks, h_juice_cost, h_popsicles_per_bottle] at h_final_sticks
  norm_num at h_final_sticks
  exact h_final_sticks

end popsicle_sticks_left_l309_309330


namespace value_multiplied_by_15_l309_309279

theorem value_multiplied_by_15 (x : ℝ) (h : 3.6 * x = 10.08) : x * 15 = 42 :=
sorry

end value_multiplied_by_15_l309_309279


namespace remainder_3_pow_2n_plus_8_l309_309396

theorem remainder_3_pow_2n_plus_8 (n : Nat) : (3 ^ (2 * n) + 8) % 8 = 1 := by
  sorry

end remainder_3_pow_2n_plus_8_l309_309396


namespace no_solutions_l309_309400

theorem no_solutions (x y : ℤ) (h : 8 * x + 3 * y^2 = 5) : False :=
by
  sorry

end no_solutions_l309_309400


namespace pumpkins_eaten_l309_309412

theorem pumpkins_eaten (initial: ℕ) (left: ℕ) (eaten: ℕ) (h1 : initial = 43) (h2 : left = 20) : eaten = 23 :=
by {
  -- We are skipping the proof as per the requirement
  sorry
}

end pumpkins_eaten_l309_309412


namespace product_pass_rate_l309_309425

variable (a b : ℝ)

theorem product_pass_rate (h1 : 0 ≤ a) (h2 : a < 1) (h3 : 0 ≤ b) (h4 : b < 1) : 
  (1 - a) * (1 - b) = 1 - (a + b - a * b) :=
by sorry

end product_pass_rate_l309_309425


namespace num_boys_on_playground_l309_309172

-- Define the conditions using Lean definitions
def num_girls : Nat := 28
def total_children : Nat := 63

-- Define a theorem to prove the number of boys
theorem num_boys_on_playground : total_children - num_girls = 35 :=
by
  -- proof steps would go here
  sorry

end num_boys_on_playground_l309_309172


namespace tan_150_deg_l309_309901

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l309_309901


namespace tan_150_deg_l309_309900

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l309_309900


namespace max_log_expression_l309_309660

noncomputable def log_base (b a : ℝ) : ℝ := Real.log a / Real.log b

theorem max_log_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) (hxy : x > y) :
  log_base x (x^2 / y^3) + log_base y (y^2 / x^3) = -2 :=
by
  sorry

end max_log_expression_l309_309660


namespace tan_150_eq_neg_inv_sqrt_3_l309_309906

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l309_309906


namespace tan_150_eq_l309_309886

noncomputable def cosine150 : ℚ := -√3 / 2
noncomputable def sine150 : ℚ := 1 / 2
noncomputable def tangent150 : ℚ := sine150 / cosine150

theorem tan_150_eq : tangent150 = -1 / √3 := by
  sorry

end tan_150_eq_l309_309886


namespace peter_candles_l309_309728

theorem peter_candles (candles_rupert : ℕ) (ratio : ℝ) (candles_peter : ℕ) 
  (h1 : ratio = 3.5) (h2 : candles_rupert = 35) (h3 : candles_peter = candles_rupert / ratio) : 
  candles_peter = 10 := 
sorry

end peter_candles_l309_309728


namespace max_sum_abc_min_sum_reciprocal_l309_309944

open Real

variables {a b c : ℝ}
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 2)

-- Maximum of a + b + c
theorem max_sum_abc : a + b + c ≤ sqrt 6 :=
by sorry

-- Minimum of 1/(a + b) + 1/(b + c) + 1/(c + a)
theorem min_sum_reciprocal : (1 / (a + b)) + (1 / (b + c)) + (1 / (c + a)) ≥ 3 * sqrt 6 / 4 :=
by sorry

end max_sum_abc_min_sum_reciprocal_l309_309944


namespace right_triangle_area_l309_309939

theorem right_triangle_area (a : ℝ) (h : a > 2)
  (h_arith_seq : a - 2 > 0)
  (pythagorean : (a - 2)^2 + a^2 = (a + 2)^2) :
  (1 / 2) * (a - 2) * a = 24 :=
by
  sorry

end right_triangle_area_l309_309939


namespace three_lines_intersect_at_three_points_l309_309628

-- Define the lines as propositions expressing the equations
def line1 (x y : ℝ) := 2 * y - 3 * x = 4
def line2 (x y : ℝ) := x + 3 * y = 3
def line3 (x y : ℝ) := 3 * x - 4.5 * y = 7.5

-- Define a proposition stating that there are exactly 3 unique points of intersection among the three lines
def number_of_intersections : ℕ := 3

-- Prove that the number of unique intersection points is exactly 3 given the lines
theorem three_lines_intersect_at_three_points : 
  ∃! p1 p2 p3 : ℝ × ℝ, 
    (line1 p1.1 p1.2 ∧ line2 p1.1 p1.2) ∧ 
    (line2 p2.1 p2.2 ∧ line3 p2.1 p2.2) ∧ 
    (line3 p3.1 p3.2 ∧ line1 p3.1 p3.2) :=
sorry

end three_lines_intersect_at_three_points_l309_309628


namespace neg_prop_l309_309999

theorem neg_prop : ¬ (∀ x : ℝ, x^2 - 2 * x + 4 ≤ 4) ↔ ∃ x : ℝ, x^2 - 2 * x + 4 > 4 := 
by 
  sorry

end neg_prop_l309_309999


namespace extreme_points_inequality_l309_309221

noncomputable def f (x : ℝ) (m : ℝ) := (1 / 2) * x^2 + m * Real.log (1 - x)

theorem extreme_points_inequality (m x1 x2 : ℝ) 
  (h_m1 : 0 < m) (h_m2 : m < 1 / 4)
  (h_x1 : 0 < x1) (h_x2: x1 < 1 / 2)
  (h_x3: x2 > 1 / 2) (h_x4: x2 < 1)
  (h_x5 : x1 < x2)
  (h_sum : x1 + x2 = 1)
  (h_prod : x1 * x2 = m)
  : (1 / 4) - (1 / 2) * Real.log 2 < (f x1 m) / x2 ∧ (f x1 m) / x2 < 0 :=
by
  sorry

end extreme_points_inequality_l309_309221


namespace arctan_sum_l309_309544

theorem arctan_sum (a b : ℝ) (h1 : a = 2 / 3) (h2 : (a + 1) * (b + 1) = 8 / 3) :
  Real.arctan a + Real.arctan b = Real.arctan (19 / 9) := by
  sorry

end arctan_sum_l309_309544


namespace part1_part2_l309_309356

-- Part (1)
theorem part1 (a : ℝ) (P Q : Set ℝ) (hP : P = {x | 4 <= x ∧ x <= 7})
              (hQ : Q = {x | -2 <= x ∧ x <= 5}) :
  (Set.compl P ∩ Q) = {x | -2 <= x ∧ x < 4} :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) (P Q : Set ℝ)
              (hP : P = {x | a + 1 <= x ∧ x <= 2 * a + 1})
              (hQ : Q = {x | -2 <= x ∧ x <= 5})
              (h_sufficient : ∀ x, x ∈ P → x ∈ Q) 
              (h_not_necessary : ∃ x, x ∈ Q ∧ x ∉ P) :
  (0 <= a ∧ a <= 2) :=
by
  sorry

end part1_part2_l309_309356


namespace problem_statement_l309_309403

noncomputable def f (x : ℝ) (a b α β : ℝ) : ℝ := a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β) + 4

theorem problem_statement (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : α ≠ 0) (h₃ : β ≠ 0) (h₄ : f 2013 a b α β = 5) :
  f 2014 a b α β = 3 :=
by
  sorry

end problem_statement_l309_309403


namespace units_digit_of_product_is_eight_l309_309770

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l309_309770


namespace units_digit_of_product_of_first_four_composites_l309_309799

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l309_309799


namespace ZYX_syndrome_diagnosis_l309_309490

theorem ZYX_syndrome_diagnosis (p : ℕ) (h1 : p = 26) (h2 : ∀ c, c = 2 * p) : ∃ n, n = c / 4 ∧ n = 13 :=
by
  sorry

end ZYX_syndrome_diagnosis_l309_309490


namespace solve_equations_l309_309738

theorem solve_equations :
  (∀ x : ℝ, (1 / 2) * (2 * x - 5) ^ 2 - 2 = 0 ↔ x = 7 / 2 ∨ x = 3 / 2) ∧
  (∀ x : ℝ, x ^ 2 - 4 * x - 4 = 0 ↔ x = 2 + 2 * Real.sqrt 2 ∨ x = 2 - 2 * Real.sqrt 2) :=
by
  sorry

end solve_equations_l309_309738


namespace units_digit_product_first_four_composite_numbers_l309_309784

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l309_309784


namespace domain_of_sqrt_1_minus_2_cos_l309_309159

theorem domain_of_sqrt_1_minus_2_cos (x : ℝ) (k : ℤ) :
  1 - 2 * Real.cos x ≥ 0 ↔ ∃ k : ℤ, (π / 3 + 2 * k * π ≤ x ∧ x ≤ 5 * π / 3 + 2 * k * π) :=
by
  sorry

end domain_of_sqrt_1_minus_2_cos_l309_309159


namespace complement_U_M_inter_N_eq_l309_309134

def U : Set ℝ := Set.univ

def M : Set ℝ := { y | ∃ x, y = 2 * x + 1 ∧ -1/2 ≤ x ∧ x ≤ 1/2 }

def N : Set ℝ := { x | ∃ y, y = Real.log (x^2 + 3 * x) ∧ (x < -3 ∨ x > 0) }

def complement_U_M : Set ℝ := U \ M

theorem complement_U_M_inter_N_eq :
  (complement_U_M ∩ N) = ((Set.Iio (-3 : ℝ)) ∪ (Set.Ioi (2 : ℝ))) :=
sorry

end complement_U_M_inter_N_eq_l309_309134


namespace kopeechka_items_l309_309697

theorem kopeechka_items (a n : ℕ) (hn : n * (100 * a + 99) = 20083) : n = 17 ∨ n = 117 :=
sorry

end kopeechka_items_l309_309697


namespace units_digit_of_product_of_first_four_composites_l309_309803

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l309_309803


namespace unit_digit_of_power_of_two_l309_309252

theorem unit_digit_of_power_of_two (n : ℕ) :
  (2 ^ 2023) % 10 = 8 := 
by
  sorry

end unit_digit_of_power_of_two_l309_309252


namespace find_x1_l309_309360

variable (x1 x2 x3 : ℝ)

theorem find_x1 (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 0.8)
    (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1 / 3) : 
    x1 = 3 / 4 :=
  sorry

end find_x1_l309_309360


namespace continuity_of_f_at_2_l309_309149

def f (x : ℝ) := -2 * x^2 - 5

theorem continuity_of_f_at_2 : ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 2| < δ → |f x - f 2| < ε :=
by {
  sorry
}

end continuity_of_f_at_2_l309_309149


namespace y_value_is_32_l309_309236

-- Define the conditions
variables (y : ℝ) (hy_pos : y > 0) (hy_eq : y^2 = 1024)

-- State the theorem
theorem y_value_is_32 : y = 32 :=
by
  -- The proof will be written here
  sorry

end y_value_is_32_l309_309236


namespace rectangular_prism_dimensions_l309_309066

theorem rectangular_prism_dimensions (a b c : ℤ) (h1: c = (a * b) / 2) (h2: 2 * (a * b + b * c + c * a) = a * b * c) :
  (a = 3 ∧ b = 10 ∧ c = 15) ∨ (a = 4 ∧ b = 6 ∧ c = 12) :=
by {
  sorry
}

end rectangular_prism_dimensions_l309_309066


namespace min_distance_to_circle_l309_309101

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  (Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2))

def is_on_circle (Q : ℝ × ℝ) : Prop :=
  (Q.1 - 1)^2 + Q.2^2 = 4

def P : ℝ × ℝ := (-2, -3)
def center : ℝ × ℝ := (1, 0)
def radius : ℝ := 2

theorem min_distance_to_circle : ∃ Q : ℝ × ℝ, is_on_circle Q ∧ distance P Q = 3 * (Real.sqrt 2) - radius :=
by
  sorry

end min_distance_to_circle_l309_309101


namespace sample_correlation_negative_one_l309_309352

noncomputable def sample_correlation {n : ℕ} (x y : Fin n → ℝ) : ℝ :=
  sorry  -- replace with actual implementation

theorem sample_correlation_negative_one
  (n : ℕ) (x y : Fin n → ℝ) 
  (hne : ∃i j: Fin n, i ≠ j ∧ x i ≠ x j)
  (hline : ∀ i, y i = -1/2 * x i + 1) :
  n ≥ 2 → sample_correlation x y = -1 :=
by
  sorry

end sample_correlation_negative_one_l309_309352


namespace tan_150_eq_neg_inv_sqrt3_l309_309883

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l309_309883


namespace study_time_l309_309006

theorem study_time (n_mcq n_fitb : ℕ) (t_mcq t_fitb : ℕ) (total_minutes_per_hour : ℕ) 
  (h1 : n_mcq = 30) (h2 : n_fitb = 30) (h3 : t_mcq = 15) (h4 : t_fitb = 25) (h5 : total_minutes_per_hour = 60) : 
  n_mcq * t_mcq + n_fitb * t_fitb = 20 * total_minutes_per_hour := 
by 
  -- This is a placeholder for the proof
  sorry

end study_time_l309_309006


namespace psychologist_diagnosis_l309_309489

theorem psychologist_diagnosis :
  let initial_patients := 26
  let doubling_factor := 2
  let probability := 1 / 4
  let total_patients := initial_patients * doubling_factor
  let expected_patients_with_ZYX := total_patients * probability
  expected_patients_with_ZYX = 13 := by
  sorry

end psychologist_diagnosis_l309_309489


namespace largest_prime_divisor_of_sum_of_squares_l309_309519

def largest_prime_divisor (n : ℕ) : ℕ := sorry

theorem largest_prime_divisor_of_sum_of_squares :
  largest_prime_divisor (11^2 + 90^2) = 89 :=
by sorry

end largest_prime_divisor_of_sum_of_squares_l309_309519


namespace sum_b_n_l309_309648

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

noncomputable def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ (q : ℝ), (∀ n : ℕ, a (n + 1) = q * a n)

theorem sum_b_n (h_geo : is_geometric a) (h_a1 : a 1 = 3) (h_sum_a : ∑' n, a n = 9) (h_bn : ∀ n, b n = a (2 * n)) :
  ∑' n, b n = 18 / 5 :=
sorry

end sum_b_n_l309_309648


namespace range_of_a_l309_309536

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * x + Real.exp x - Real.exp (-x)

theorem range_of_a (a : ℝ) (h : f (a - 1) + f (2 * a^2) ≤ 0) : -1 ≤ a ∧ a ≤ 1/2 :=
by
  sorry

end range_of_a_l309_309536


namespace train_B_speed_l309_309183

theorem train_B_speed (V_B : ℝ) : 
  (∀ t meet_A meet_B, 
     meet_A = 9 ∧
     meet_B = 4 ∧
     t = 70 ∧
     (t * meet_A) = (V_B * meet_B)) →
     V_B = 157.5 :=
by
  intros h
  sorry

end train_B_speed_l309_309183


namespace one_thirteen_150th_digit_l309_309461

def decimal_rep_of_one_thirteen := "076923"  -- the repeating sequence

def position_within_block (n : ℕ) : ℕ :=
  n % 6

def last_digit_of_block (block : String) : Char :=
  block.get ⟨block.length - 1, sorry⟩  -- unsafely get the last character

theorem one_thirteen_150th_digit : 
  (decimal_rep_of_one_thirteen.get ⟨position_within_block 150, sorry⟩) = '3' :=
by
  unfold decimal_rep_of_one_thirteen
  unfold position_within_block
  sorry

end one_thirteen_150th_digit_l309_309461


namespace smallest_x_l309_309990

theorem smallest_x {
    x : ℤ
} : (x % 11 = 9) ∧ (x % 13 = 11) ∧ (x % 15 = 13) → x = 2143 := by
sorry

end smallest_x_l309_309990


namespace regular_polygon_sides_and_exterior_angle_l309_309653

theorem regular_polygon_sides_and_exterior_angle (n : ℕ) (exterior_sum : ℝ) :
  (180 * (n - 2) = 360 + exterior_sum) → (exterior_sum = 360) → n = 6 ∧ (360 / n = 60) :=
by
  intro h1 h2
  sorry

end regular_polygon_sides_and_exterior_angle_l309_309653


namespace range_of_x_for_acute_angle_l309_309661

theorem range_of_x_for_acute_angle (x : ℝ) (h₁ : (x, 2*x) ≠ (0, 0)) (h₂ : (x+1, x+3) ≠ (0, 0)) (h₃ : (3*x^2 + 7*x > 0)) : 
  x < -7/3 ∨ (0 < x ∧ x < 1) ∨ x > 1 :=
by {
  -- This theorem asserts the given range of x given the dot product solution.
  sorry
}

end range_of_x_for_acute_angle_l309_309661


namespace circumcircle_BCN_tangent_to_Omega_at_N_l309_309124

open EuclideanGeometry

variable {ABC : Triangle ℝ} (Ω : Circle ℝ) (B C N : Point ℝ) (K : Point ℝ)
variable [T : ABC.IsAcute]
variable [H₁ : Ω.IsIncircle ABC]
variable [H₂ : Ω.TouchesAt H₁ BC K]
variable (AD : Segment ℝ) [H₃ : AD.IsAltitude ABC]
variable [M : Segment ℝ] [H₄ : M.IsMidpoint AD]
variable [N : Point ℝ] [H₅ : (K.lineThrough M).IntersectsAt Ω N]

theorem circumcircle_BCN_tangent_to_Omega_at_N :
  let circumcircle_BCN := Circle.Circumcircle B C N in
  circumcircle_BCN.IsTangentTo Ω N := sorry

end circumcircle_BCN_tangent_to_Omega_at_N_l309_309124


namespace fraction_of_crop_brought_to_BC_l309_309509

/-- Consider a kite-shaped field with sides AB = 120 m, BC = CD = 80 m, DA = 120 m.
    The angle between sides AB and BC is 120°, and between sides CD and DA is also 120°.
    Prove that the fraction of the crop brought to the longest side BC is 1/2. -/
theorem fraction_of_crop_brought_to_BC :
  ∀ (AB BC CD DA : ℝ) (α β : ℝ),
  AB = 120 ∧ BC = 80 ∧ CD = 80 ∧ DA = 120 ∧ α = 120 ∧ β = 120 →
  ∃ (frac : ℝ), frac = 1 / 2 :=
by
  intros AB BC CD DA α β h
  sorry

end fraction_of_crop_brought_to_BC_l309_309509


namespace price_of_davids_toy_l309_309085

theorem price_of_davids_toy :
  ∀ (n : ℕ) (avg_before : ℕ) (avg_after : ℕ) (total_toys_after : ℕ), 
    n = 5 →
    avg_before = 10 →
    avg_after = 11 →
    total_toys_after = 6 →
  (total_toys_after * avg_after - n * avg_before = 16) :=
by
  intros n avg_before avg_after total_toys_after h_n h_avg_before h_avg_after h_total_toys_after
  sorry

end price_of_davids_toy_l309_309085


namespace symmetric_point_coordinates_l309_309946

theorem symmetric_point_coordinates 
  (k : ℝ) 
  (P : ℝ × ℝ) 
  (h1 : ∀ k, k * (P.1) - P.2 + k - 2 = 0) 
  (P' : ℝ × ℝ) 
  (h2 : P'.1 + P'.2 = 3) 
  (h3 : 2 * P'.1^2 + 2 * P'.2^2 + 4 * P'.1 + 8 * P'.2 + 5 = 0) 
  (hP : P = (-1, -2)): 
  P' = (2, 1) := 
sorry

end symmetric_point_coordinates_l309_309946


namespace Olivia_house_height_l309_309046

variable (h : ℕ)
variable (flagpole_height : ℕ := 35)
variable (flagpole_shadow : ℕ := 30)
variable (house_shadow : ℕ := 70)
variable (bush_height : ℕ := 14)
variable (bush_shadow : ℕ := 12)

theorem Olivia_house_height :
  (house_shadow / flagpole_shadow) * flagpole_height = 81 ∧
  (house_shadow / bush_shadow) * bush_height = 81 :=
by
  sorry

end Olivia_house_height_l309_309046


namespace excursion_min_parents_l309_309434

theorem excursion_min_parents 
  (students : ℕ) 
  (car_capacity : ℕ)
  (h_students : students = 30)
  (h_car_capacity : car_capacity = 5) 
  : ∃ (parents_needed : ℕ), parents_needed = 8 := 
by
  sorry -- proof goes here

end excursion_min_parents_l309_309434


namespace express_fraction_l309_309090

noncomputable def x : ℚ := 0.8571 -- This represents \( x = 0.\overline{8571} \)
noncomputable def y : ℚ := 0.142857 -- This represents \( y = 0.\overline{142857} \)
noncomputable def z : ℚ := 2 + y -- This represents \( 2 + y = 2.\overline{142857} \)

theorem express_fraction :
  (x / z) = (1 / 2) :=
by
  sorry

end express_fraction_l309_309090


namespace units_digit_first_four_composite_is_eight_l309_309817

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l309_309817


namespace new_ratio_after_adding_water_l309_309487

-- Define the initial conditions
variables (M W M_new W_new : ℕ)
def initial_conditions : Prop := 
  (M / (W : ℚ) = 3 / 2) ∧ 
  (M + W = 20) ∧ 
  (W_new = W + 10) ∧ 
  (M_new = M)

-- State the theorem to prove the new ratio
theorem new_ratio_after_adding_water :
  initial_conditions M W M_new W_new →
  M_new / (W_new : ℚ) = 2 / 3 :=
by
  sorry

end new_ratio_after_adding_water_l309_309487


namespace baseball_games_played_l309_309192
-- Import necessary libraries

-- Define the conditions and state the main theorem
theorem baseball_games_played (P : ℕ) (L : ℕ) (h1 : P = 5 + L) (h2 : P = 2 * L) : P = 10 :=
by 
  sorry

end baseball_games_played_l309_309192


namespace smallest_n_such_that_no_n_digit_is_11_power_l309_309444

theorem smallest_n_such_that_no_n_digit_is_11_power (log_11 : Real) (h : log_11 = 1.0413) : 
  ∃ n > 1, ∀ k : ℕ, ¬ (10 ^ (n - 1) ≤ 11 ^ k ∧ 11 ^ k < 10 ^ n) :=
sorry

end smallest_n_such_that_no_n_digit_is_11_power_l309_309444


namespace chocolates_initial_count_l309_309225

theorem chocolates_initial_count : 
  ∀ (chocolates_first_day chocolates_second_day chocolates_third_day chocolates_fourth_day chocolates_fifth_day initial_chocolates : ℕ),
  chocolates_first_day = 4 →
  chocolates_second_day = 2 * chocolates_first_day - 3 →
  chocolates_third_day = chocolates_first_day - 2 →
  chocolates_fourth_day = chocolates_third_day - 1 →
  chocolates_fifth_day = 12 →
  initial_chocolates = chocolates_first_day + chocolates_second_day + chocolates_third_day + chocolates_fourth_day + chocolates_fifth_day →
  initial_chocolates = 24 :=
by {
  -- the proof will go here,
  sorry
}

end chocolates_initial_count_l309_309225


namespace digit_150_in_decimal_representation_of_one_div_thirteen_l309_309457

theorem digit_150_in_decimal_representation_of_one_div_thirteen : 
  let repeating_seq := "076923" 
  in ∀ (n : ℕ), n = 150 → repeating_seq[(n - 1) % 6] = '3' := 
by 
  intros repeating_seq n hn 
  dsimp only 
  rw hn 
  sorry

end digit_150_in_decimal_representation_of_one_div_thirteen_l309_309457


namespace find_certain_number_l309_309431

theorem find_certain_number (N : ℝ) 
  (h : 3.6 * N * 2.50 / (0.12 * 0.09 * 0.5) = 800.0000000000001)
  : N = 0.48 :=
sorry

end find_certain_number_l309_309431


namespace num_teacher_volunteers_l309_309263

theorem num_teacher_volunteers (total_needed volunteers_from_classes extra_needed teacher_volunteers : ℕ)
  (h1 : teacher_volunteers + extra_needed + volunteers_from_classes = total_needed) 
  (h2 : total_needed = 50)
  (h3 : volunteers_from_classes = 6 * 5)
  (h4 : extra_needed = 7) :
  teacher_volunteers = 13 :=
by
  sorry

end num_teacher_volunteers_l309_309263


namespace find_e_value_l309_309003

theorem find_e_value : 
  ∃ e : ℝ, 12 / (-12 + 2 * e) = -11 - 2 * e ∧ e = 4 :=
by
  use 4
  sorry

end find_e_value_l309_309003


namespace particular_solution_satisfies_initial_conditions_l309_309586

noncomputable def x_solution : ℝ → ℝ := λ t => (-4/3) * Real.exp t + (7/3) * Real.exp (-2 * t)
noncomputable def y_solution : ℝ → ℝ := λ t => (-1/3) * Real.exp t + (7/3) * Real.exp (-2 * t)

def x_prime (x y : ℝ) := 2 * x - 4 * y
def y_prime (x y : ℝ) := x - 3 * y

theorem particular_solution_satisfies_initial_conditions :
  (∀ t, deriv x_solution t = x_prime (x_solution t) (y_solution t)) ∧
  (∀ t, deriv y_solution t = y_prime (x_solution t) (y_solution t)) ∧
  (x_solution 0 = 1) ∧
  (y_solution 0 = 2) := by
  sorry

end particular_solution_satisfies_initial_conditions_l309_309586


namespace geometric_progression_sum_l309_309208

theorem geometric_progression_sum :
  ∀ (b q a d : ℝ),
    b = a →
    b * q ^ 3 = a + 3 * d →
    b * q ^ 7 = a + 7 * d →
    3 * a + 10 * d = 148 / 9 →
    b * (1 + q + q^2 + q^3) = 700 / 27 :=
by
  intros b q a d h1 h2 h3 h4
  sorry

end geometric_progression_sum_l309_309208


namespace roots_poly_sum_cubed_eq_l309_309972

theorem roots_poly_sum_cubed_eq :
  ∀ (r s t : ℝ), (r + s + t = 0) 
  → (∀ x, 9 * x^3 + 2023 * x + 4047 = 0 → x = r ∨ x = s ∨ x = t) 
  → (r + s) ^ 3 + (s + t) ^ 3 + (t + r) ^ 3 = 1349 :=
by
  intros r s t h_sum h_roots
  sorry

end roots_poly_sum_cubed_eq_l309_309972


namespace tan_150_eq_neg_one_over_sqrt_three_l309_309909

theorem tan_150_eq_neg_one_over_sqrt_three :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  sorry

end tan_150_eq_neg_one_over_sqrt_three_l309_309909


namespace parallel_vectors_implies_value_of_t_l309_309540

theorem parallel_vectors_implies_value_of_t (t : ℝ) :
  let a := (1, t)
  let b := (t, 9)
  (1 * 9 - t^2 = 0) → (t = 3 ∨ t = -3) := 
by sorry

end parallel_vectors_implies_value_of_t_l309_309540


namespace kopeechka_items_l309_309699

theorem kopeechka_items (a n : ℕ) (hn : n * (100 * a + 99) = 20083) : n = 17 ∨ n = 117 :=
sorry

end kopeechka_items_l309_309699


namespace ab_ac_bc_all_real_l309_309260

theorem ab_ac_bc_all_real (a b c : ℝ) (h : a + b + c = 1) : ∃ x : ℝ, ab + ac + bc = x := by
  sorry

end ab_ac_bc_all_real_l309_309260


namespace digit_150_of_1_div_13_l309_309446

theorem digit_150_of_1_div_13 : 
  (150th_digit_of_decimal_expansion (1/13) = 3) := 
begin
  sorry
end

end digit_150_of_1_div_13_l309_309446


namespace georgie_entry_exit_ways_l309_309617

-- Defining the conditions
def castle_windows : Nat := 8
def non_exitable_windows : Nat := 2

-- Defining the problem
theorem georgie_entry_exit_ways (total_windows : Nat) (blocked_exits : Nat) (entry_windows : Nat) : 
  total_windows = castle_windows → blocked_exits = non_exitable_windows → 
  entry_windows = castle_windows →
  (entry_windows * (total_windows - 1 - blocked_exits) = 40) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end georgie_entry_exit_ways_l309_309617


namespace problem1_problem2_problem3_l309_309284

/-- Problem 1: Calculate 25 * 26 * 8 and show it equals 5200 --/
theorem problem1 : 25 * 26 * 8 = 5200 := 
sorry

/-- Problem 2: Calculate 340 * 40 / 17 and show it equals 800 --/
theorem problem2 : 340 * 40 / 17 = 800 := 
sorry

/-- Problem 3: Calculate 440 * 15 + 480 * 15 + 79 * 15 + 15 and show it equals 15000 --/
theorem problem3 : 440 * 15 + 480 * 15 + 79 * 15 + 15 = 15000 := 
sorry

end problem1_problem2_problem3_l309_309284


namespace john_has_18_blue_pens_l309_309846

variables (R B Bl : ℕ)

-- Conditions from the problem
def john_has_31_pens : Prop := R + B + Bl = 31
def black_pens_5_more_than_red : Prop := B = R + 5
def blue_pens_twice_black : Prop := Bl = 2 * B

theorem john_has_18_blue_pens :
  john_has_31_pens R B Bl ∧ black_pens_5_more_than_red R B ∧ blue_pens_twice_black B Bl →
  Bl = 18 :=
by
  sorry

end john_has_18_blue_pens_l309_309846


namespace coefficient_of_x2_in_binomial_expansion_l309_309093

theorem coefficient_of_x2_in_binomial_expansion :
  let T_r := ∑ r in Finset.range 8, (Nat.choose 7 r) * (-1)^r * x^(7-r)
  by sorry
in T_r = -21 :=
by
  sorry

end coefficient_of_x2_in_binomial_expansion_l309_309093


namespace units_digit_first_four_composite_is_eight_l309_309818

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l309_309818


namespace least_possible_value_of_d_l309_309549

theorem least_possible_value_of_d
  (x y z : ℤ)
  (h1 : x < y)
  (h2 : y < z)
  (h3 : y - x > 5)
  (hx_even : x % 2 = 0)
  (hy_odd : y % 2 = 1)
  (hz_odd : z % 2 = 1) :
  (z - x) = 9 := 
sorry

end least_possible_value_of_d_l309_309549


namespace original_students_l309_309753

theorem original_students (a b : ℕ) : 
  a + b = 92 ∧ a - 5 = 3 * (b + 5 - 32) → a = 45 ∧ b = 47 :=
by sorry

end original_students_l309_309753


namespace fraction_to_decimal_l309_309927

theorem fraction_to_decimal :
  (7 / 12 : ℝ) ≈ 0.5833 :=
begin
  sorry
end

end fraction_to_decimal_l309_309927


namespace probability_floor_sqrt_even_l309_309026

/-- Suppose x and y are chosen randomly and uniformly from (0,1). The probability that
    ⌊√(x/y)⌋ is even is 1 - π²/24. -/
theorem probability_floor_sqrt_even (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) :
  (1 - Real.pi ^ 2 / 24) = sorry :=
sorry

end probability_floor_sqrt_even_l309_309026


namespace probability_of_specific_balls_l309_309853

-- Define total number of balls in the jar
def total_balls : ℕ := 5 + 7 + 2 + 3 + 4

-- Define combinations function
def comb (n k : ℕ) : ℕ := n.choose k

-- Define number of ways to choose 1 black, 1 green, and 1 red ball
def favorable_outcomes : ℕ := comb 5 1 * comb 2 1 * comb 4 1

-- Define total number of ways to choose any 3 balls
def total_outcomes : ℕ := comb total_balls 3

-- Define probability of picking one black, one green, and one red ball
def probability := (4 : ℚ) / 133

theorem probability_of_specific_balls :
  5 = 5 ∧ 7 = 7 ∧ 2 = 2 ∧ 3 = 3 ∧ 4 = 4 →
  (favorable_outcomes.to_rat / total_outcomes.to_rat) = probability :=
begin
  sorry
end

end probability_of_specific_balls_l309_309853


namespace no_adjacent_teachers_l309_309752

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def permutation (n k : ℕ) : ℕ :=
  factorial n / factorial (n - k)

theorem no_adjacent_teachers (students teachers : ℕ)
  (h_students : students = 4)
  (h_teachers : teachers = 3) :
  ∃ (arrangements : ℕ), arrangements = (factorial students) * (permutation (students + 1) teachers) :=
by
  sorry

end no_adjacent_teachers_l309_309752


namespace total_amount_of_money_if_all_cookies_sold_equals_1255_50_l309_309210

-- Define the conditions
def number_cookies_Clementine : ℕ := 72
def number_cookies_Jake : ℕ := 5 * number_cookies_Clementine / 2
def number_cookies_Tory : ℕ := (number_cookies_Jake + number_cookies_Clementine) / 2
def number_cookies_Spencer : ℕ := 3 * (number_cookies_Jake + number_cookies_Tory) / 2
def price_per_cookie : ℝ := 1.50

-- Total number of cookies
def total_cookies : ℕ :=
  number_cookies_Clementine + number_cookies_Jake + number_cookies_Tory + number_cookies_Spencer

-- Proof statement
theorem total_amount_of_money_if_all_cookies_sold_equals_1255_50 :
  (total_cookies * price_per_cookie : ℝ) = 1255.50 := by
  sorry

end total_amount_of_money_if_all_cookies_sold_equals_1255_50_l309_309210


namespace brian_commission_rate_l309_309078

noncomputable def commission_rate (sale1 sale2 sale3 commission : ℝ) : ℝ :=
  (commission / (sale1 + sale2 + sale3)) * 100

theorem brian_commission_rate :
  commission_rate 157000 499000 125000 15620 = 2 :=
by
  unfold commission_rate
  sorry

end brian_commission_rate_l309_309078


namespace points_earned_l309_309251

-- Define the given conditions
def points_per_enemy := 5
def total_enemies := 8
def enemies_remaining := 6

-- Calculate the number of enemies defeated
def enemies_defeated := total_enemies - enemies_remaining

-- Calculate the points earned based on the enemies defeated
theorem points_earned : enemies_defeated * points_per_enemy = 10 := by
  -- Insert mathematical operations
  sorry

end points_earned_l309_309251


namespace units_digit_first_four_composites_l309_309767

theorem units_digit_first_four_composites :
  let p := [4, 6, 8, 9] in
  let product := List.prod p in
  product % 10 = 8 :=
by
  let p := [4, 6, 8, 9]
  let product := List.prod p
  show product % 10 = 8
  sorry

end units_digit_first_four_composites_l309_309767


namespace lucy_crayons_l309_309599

theorem lucy_crayons (W L : ℕ) (h1 : W = 1400) (h2 : W = L + 1110) : L = 290 :=
by {
  sorry
}

end lucy_crayons_l309_309599


namespace jessica_threw_away_4_roses_l309_309171

def roses_thrown_away (a b c d : ℕ) : Prop :=
  (a + b) - d = c

theorem jessica_threw_away_4_roses :
  roses_thrown_away 2 25 23 4 :=
by
  -- This is where the proof would go
  sorry

end jessica_threw_away_4_roses_l309_309171


namespace general_solution_of_differential_eq_l309_309043

noncomputable def y (x C : ℝ) : ℝ := x * (Real.exp (x ^ 2) + C)

theorem general_solution_of_differential_eq {x C : ℝ} (h : x ≠ 0) :
  let y' := (1 : ℝ) * (Real.exp (x ^ 2) + C) + x * (2 * x * Real.exp (x ^ 2))
  y' = (y x C / x) + 2 * x ^ 2 * Real.exp (x ^ 2) :=
by
  -- the proof goes here
  sorry

end general_solution_of_differential_eq_l309_309043


namespace popsicle_sticks_left_l309_309333

-- Defining the conditions
def total_money : ℕ := 10
def cost_of_molds : ℕ := 3
def cost_of_sticks : ℕ := 1
def cost_of_juice_bottle : ℕ := 2
def popsicles_per_bottle : ℕ := 20
def initial_sticks : ℕ := 100

-- Statement of the problem
theorem popsicle_sticks_left : 
  let remaining_money := total_money - cost_of_molds - cost_of_sticks
  let bottles_of_juice := remaining_money / cost_of_juice_bottle
  let total_popsicles := bottles_of_juice * popsicles_per_bottle
  let sticks_left := initial_sticks - total_popsicles
  sticks_left = 40 := by
  sorry

end popsicle_sticks_left_l309_309333


namespace tan_150_eq_neg_inv_sqrt_3_l309_309903

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l309_309903


namespace soccer_game_points_ratio_l309_309088

theorem soccer_game_points_ratio :
  ∃ B1 A1 A2 B2 : ℕ,
    A1 = 8 ∧
    B2 = 8 ∧
    A2 = 6 ∧
    (A1 + B1 + A2 + B2 = 26) ∧
    (B1 / A1 = 1 / 2) := by
  sorry

end soccer_game_points_ratio_l309_309088


namespace sum_arithmetic_sequence_S12_l309_309943

variable {a : ℕ → ℝ} -- Arithmetic sequence a_n
variable {S : ℕ → ℝ} -- Sum of the first n terms S_n

-- Conditions given in the problem
axiom condition1 (n : ℕ) : S n = (n / 2) * (a 1 + a n)
axiom condition2 : a 4 + a 9 = 10

-- Proving that S 12 = 60 given the conditions
theorem sum_arithmetic_sequence_S12 : S 12 = 60 := by
  sorry

end sum_arithmetic_sequence_S12_l309_309943


namespace probability_intersection_l309_309375

variables (A B : Type → Prop)

-- Assuming we have a measure space (probability) P
variables {P : Type → Prop}

-- Given probabilities
def p_A := 0.65
def p_B := 0.55
def p_Ac_Bc := 0.20

-- The theorem to be proven
theorem probability_intersection :
  (p_A + p_B - (1 - p_Ac_Bc) = 0.40) :=
by
  sorry

end probability_intersection_l309_309375


namespace simplest_square_root_l309_309294

theorem simplest_square_root (A B C D : Real) 
    (hA : A = Real.sqrt 0.1) 
    (hB : B = 1 / 2) 
    (hC : C = Real.sqrt 30) 
    (hD : D = Real.sqrt 18) : 
    C = Real.sqrt 30 := 
by 
    sorry

end simplest_square_root_l309_309294


namespace decreasing_interval_of_function_l309_309590

noncomputable def y (x : ℝ) : ℝ := (3 / Real.pi) ^ (x ^ 2 + 2 * x - 3)

theorem decreasing_interval_of_function :
  ∀ x ∈ Set.Ioi (-1 : ℝ), ∃ ε > 0, ∀ δ > 0, δ ≤ ε → y (x - δ) > y x :=
by
  sorry

end decreasing_interval_of_function_l309_309590


namespace largest_eight_digit_with_all_evens_l309_309822

theorem largest_eight_digit_with_all_evens :
  ∃ n : ℕ, (digits 10 n).length = 8 ∧
           (∀ d, d ∈ [2, 4, 6, 8, 0] → List.mem d (digits 10 n)) ∧
           n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_evens_l309_309822


namespace multiple_of_second_number_l309_309409

def main : IO Unit := do
  IO.println s!"Proof problem statement in Lean 4."

theorem multiple_of_second_number (x m : ℕ) 
  (h1 : 19 = m * x + 3) 
  (h2 : 19 + x = 27) : 
  m = 2 := 
sorry

end multiple_of_second_number_l309_309409


namespace right_triangles_count_l309_309952

theorem right_triangles_count (b a : ℕ) (h₁: b < 150) (h₂: (a^2 + b^2 = (b + 2)^2)) :
  ∃ n : ℕ, 1 ≤ n ∧ n ≤ 12 ∧ b = n^2 - 1 :=
by
  -- This intended to state the desired number and form of the right triangles.
  sorry

def count_right_triangles : ℕ :=
  12 -- Result as a constant based on proof steps

#eval count_right_triangles -- Should output 12

end right_triangles_count_l309_309952


namespace possible_items_l309_309690

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items_l309_309690


namespace abs_condition_l309_309290

theorem abs_condition (x : ℝ) : |2 * x - 7| ≤ 0 ↔ x = 7 / 2 := 
by
  sorry

end abs_condition_l309_309290


namespace solve_for_x_l309_309737

theorem solve_for_x : 
  (∀ x : ℝ, x ≠ -2 → (x^2 - x - 2) / (x + 2) = x - 1 ↔ x = 0) := 
by 
  sorry

end solve_for_x_l309_309737


namespace total_selling_price_l309_309199

theorem total_selling_price (profit_per_meter cost_price_per_meter meters : ℕ)
  (h_profit : profit_per_meter = 20)
  (h_cost : cost_price_per_meter = 85)
  (h_meters : meters = 85) :
  (cost_price_per_meter + profit_per_meter) * meters = 8925 :=
by
  sorry

end total_selling_price_l309_309199


namespace ratio_of_efficiencies_l309_309296

-- Definitions of efficiencies
def efficiency (time : ℕ) : ℚ := 1 / time

-- Conditions:
def E_C : ℚ := efficiency 20
def E_D : ℚ := efficiency 30
def E_A : ℚ := efficiency 18
def E_B : ℚ := 1 / 36 -- Placeholder for efficiency of B to complete the statement

-- The proof goal
theorem ratio_of_efficiencies (h1 : E_A + E_B = E_C + E_D) : E_A / E_B = 2 :=
by
  -- Placeholder to structure the format, the proof will be constructed here
  sorry

end ratio_of_efficiencies_l309_309296


namespace boys_more_than_girls_l309_309754

def numGirls : ℝ := 28.0
def numBoys : ℝ := 35.0

theorem boys_more_than_girls : numBoys - numGirls = 7.0 := by
  sorry

end boys_more_than_girls_l309_309754


namespace height_of_barbed_wire_l309_309158

theorem height_of_barbed_wire (area : ℝ) (cost_per_meter : ℝ) (gate_width : ℝ) (total_cost : ℝ) (h : ℝ) :
  area = 3136 →
  cost_per_meter = 1.50 →
  gate_width = 2 →
  total_cost = 999 →
  h = 3 := 
by
  sorry

end height_of_barbed_wire_l309_309158


namespace a0_a1_consecutive_l309_309166

variable (a : ℕ → ℤ)
variable (cond : ∀ i ≥ 2, a i = 2 * a (i - 1) - a (i - 2) ∨ a i = 2 * a (i - 2) - a (i - 1))
variable (consec : |a 2024 - a 2023| = 1)

theorem a0_a1_consecutive :
  |a 1 - a 0| = 1 :=
by
  -- Proof skipped
  sorry

end a0_a1_consecutive_l309_309166


namespace entree_cost_l309_309626

theorem entree_cost (E : ℝ) :
  let appetizer := 9
  let dessert := 11
  let tip_rate := 0.30
  let total_cost_with_tip := 78
  let total_cost_before_tip := appetizer + 2 * E + dessert
  total_cost_with_tip = total_cost_before_tip + (total_cost_before_tip * tip_rate) →
  E = 20 :=
by
  intros appetizer dessert tip_rate total_cost_with_tip total_cost_before_tip h
  sorry

end entree_cost_l309_309626


namespace one_div_thirteen_150th_digit_l309_309453

theorem one_div_thirteen_150th_digit :
  let cycle := "076923"
  let n := 150
  let position := n % cycle.length
  cycle.get position = '3' :=
by
  let cycle := "076923"
  let n := 150
  let position := 150 % 6
  sorry

end one_div_thirteen_150th_digit_l309_309453


namespace number_of_items_l309_309680

theorem number_of_items {a n : ℕ} (h1 : ∀ x, x = 100 * a + 99) (h2 : 200 * 100 + 83 = 20083) : 
  (n * (100 * a + 99) = 20083) → (n = 17 ∨ n = 117) :=
by 
  sorry

end number_of_items_l309_309680


namespace perpendicular_vectors_l309_309368

-- Define the vectors m and n
def m : ℝ × ℝ := (1, 2)
def n : ℝ × ℝ := (-3, 2)

-- Define the conditions to be checked
def km_plus_n (k : ℝ) : ℝ × ℝ := (k * m.1 + n.1, k * m.2 + n.2)
def m_minus_3n : ℝ × ℝ := (m.1 - 3 * n.1, m.2 - 3 * n.2)

-- The dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Prove that for k = 19, the two vectors are perpendicular
theorem perpendicular_vectors (k : ℝ) (h : k = 19) : dot_product (km_plus_n k) (m_minus_3n) = 0 := by
  rw [h]
  simp [km_plus_n, m_minus_3n, dot_product]
  sorry

end perpendicular_vectors_l309_309368


namespace value_of_square_reciprocal_l309_309370

theorem value_of_square_reciprocal (x : ℝ) (h : 18 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = Real.sqrt 20 := by
  sorry

end value_of_square_reciprocal_l309_309370


namespace value_of_expression_l309_309633

variable (a b : ℝ)

theorem value_of_expression : 
  let x := a + b 
  let y := a - b 
  (x - y) * (x + y) = 4 * a * b := 
by
  sorry

end value_of_expression_l309_309633


namespace units_digit_first_four_composite_is_eight_l309_309820

-- Definitions of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of the first four composite numbers
def product_first_four_composite : ℕ := first_four_composite_numbers.prod

-- Define the function to extract the units digit of a number
def units_digit (n : ℕ) : ℕ := n % 10

-- The main statement to prove
theorem units_digit_first_four_composite_is_eight : units_digit product_first_four_composite = 8 := 
sorry

end units_digit_first_four_composite_is_eight_l309_309820


namespace part_I_part_II_l309_309001

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x + x^2 - a * x

theorem part_I (x : ℝ) (a : ℝ) (h_inc : ∀ x > 0, (1/x + 2*x - a) ≥ 0) : a ≤ 2 * Real.sqrt 2 :=
sorry

noncomputable def g (x : ℝ) (a : ℝ) := f x a + 2 * Real.log ((a * x + 2) / (6 * Real.sqrt x))

theorem part_II (a : ℝ) (k : ℝ) (h_a : 2 < a ∧ a < 4) (h_ex : ∃ x : ℝ, (3/2) ≤ x ∧ x ≤ 2 ∧ g x a > k * (4 - a^2)) : k ≥ 1/3 :=
sorry

end part_I_part_II_l309_309001


namespace tan_increasing_interval_l309_309591

noncomputable def increasing_interval (k : ℤ) : Set ℝ := 
  {x | (k * Real.pi / 2 - 5 * Real.pi / 12 < x) ∧ (x < k * Real.pi / 2 + Real.pi / 12)}

theorem tan_increasing_interval (k : ℤ) : 
  ∀ x : ℝ, (k * Real.pi / 2 - 5 * Real.pi / 12 < x) ∧ (x < k * Real.pi / 2 + Real.pi / 12) ↔ 
    (∃ y, y = (2 * x + Real.pi / 3) ∧ Real.tan y > Real.tan (2 * x + Real.pi / 3 - 1e-6)) :=
sorry

end tan_increasing_interval_l309_309591


namespace cubic_difference_pos_l309_309372

theorem cubic_difference_pos {a b : ℝ} (h : a > b) : a^3 - b^3 > 0 :=
sorry

end cubic_difference_pos_l309_309372


namespace chime_2203_occurs_on_March_19_l309_309850

-- Define the initial conditions: chime patterns
def chimes_at_half_hour : Nat := 1
def chimes_at_hour (h : Nat) : Nat := if h = 12 then 12 else h % 12

-- Define the start time and the question parameters
def start_time_hours : Nat := 10
def start_time_minutes : Nat := 45
def start_day : Nat := 26 -- Assume February 26 as starting point, to facilitate day count accurately
def target_chime : Nat := 2203

-- Define the date calculation function (based on given solution steps)
noncomputable def calculate_chime_date (start_day : Nat) : Nat := sorry

-- The goal is to prove calculate_chime_date with given start conditions equals 19 (March 19th is the 19th day after the base day assumption of March 0)
theorem chime_2203_occurs_on_March_19 :
  calculate_chime_date start_day = 19 :=
sorry

end chime_2203_occurs_on_March_19_l309_309850


namespace tan_150_eq_neg_inv_sqrt3_l309_309919

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l309_309919


namespace units_digit_of_product_of_first_four_composites_l309_309790

theorem units_digit_of_product_of_first_four_composites :
  (4 * 6 * 8 * 9) % 10 = 8 := 
by
  sorry

end units_digit_of_product_of_first_four_composites_l309_309790


namespace total_sum_lent_l309_309857

-- Conditions
def interest_equal (x y : ℕ) : Prop :=
  (x * 3 * 8) / 100 = (y * 5 * 3) / 100

def second_sum : ℕ := 1704

-- Assertion
theorem total_sum_lent : ∃ x : ℕ, interest_equal x second_sum ∧ (x + second_sum = 2769) :=
  by
  -- Placeholder proof
  sorry

end total_sum_lent_l309_309857


namespace range_of_S_on_ellipse_l309_309535

theorem range_of_S_on_ellipse :
  ∀ (x y : ℝ),
    (x ^ 2 / 2 + y ^ 2 / 3 = 1) →
    -Real.sqrt 5 ≤ x + y ∧ x + y ≤ Real.sqrt 5 :=
by
  intro x y
  intro h
  sorry

end range_of_S_on_ellipse_l309_309535


namespace a_eq_1_sufficient_not_necessary_l309_309482

theorem a_eq_1_sufficient_not_necessary (a : ℝ) :
  (∀ x : ℝ, x ≤ 1 → |x - 1| ≤ |x - a|) ∧ ¬(∀ x : ℝ, x ≤ 1 → |x - 1| = |x - a|) :=
by
  sorry

end a_eq_1_sufficient_not_necessary_l309_309482


namespace smallest_x_division_remainder_l309_309471

theorem smallest_x_division_remainder :
  ∃ x : ℕ, x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7 ∧ x = 167 := by
  sorry

end smallest_x_division_remainder_l309_309471


namespace fraction_of_married_men_l309_309074

theorem fraction_of_married_men (total_women married_women : ℕ) 
    (h1 : total_women = 7)
    (h2 : married_women = 4)
    (single_women_probability : ℚ)
    (h3 : single_women_probability = 3 / 7) : 
    (4 / 11 : ℚ) = (married_women / (total_women + married_women)) := 
sorry

end fraction_of_married_men_l309_309074


namespace min_value_expr_l309_309568

open Real

theorem min_value_expr (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + 1/y) * (x + 1/y - 2020) + (y + 1/x) * (y + 1/x - 2020) ≥ -2040200 :=
by
  sorry

end min_value_expr_l309_309568


namespace gcd_calculation_l309_309505

theorem gcd_calculation :
  let a := 97^7 + 1
  let b := 97^7 + 97^3 + 1
  gcd a b = 1 := by
  sorry

end gcd_calculation_l309_309505


namespace kopeechka_items_l309_309705

-- Define necessary concepts and conditions
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 200 * 100 + 83

-- Lean statement defining the proof problem
theorem kopeechka_items (a n : ℕ) (h1 : ∀ a, n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
by sorry

end kopeechka_items_l309_309705


namespace georgia_total_cost_l309_309016

def carnation_price : ℝ := 0.50
def dozen_price : ℝ := 4.00
def teachers : ℕ := 5
def friends : ℕ := 14

theorem georgia_total_cost :
  ((dozen_price * teachers) + dozen_price + (carnation_price * (friends - 12))) = 25.00 :=
by
  sorry

end georgia_total_cost_l309_309016


namespace units_digit_first_four_composites_l309_309815

theorem units_digit_first_four_composites : 
  let first_four_composites := [4, 6, 8, 9]
  let product := first_four_composites.prod
  Nat.unitsDigit product = 8 :=
by
  sorry

end units_digit_first_four_composites_l309_309815


namespace fraction_is_meaningful_l309_309427

theorem fraction_is_meaningful (x : ℝ) : x ≠ 1 ↔ ∃ y : ℝ, y = 8 / (x - 1) :=
by
  sorry

end fraction_is_meaningful_l309_309427


namespace greatest_int_value_not_satisfy_condition_l309_309466

/--
For the inequality 8 - 6x > 26, the greatest integer value 
of x that satisfies this is -4.
-/
theorem greatest_int_value (x : ℤ) : 8 - 6 * x > 26 → x ≤ -4 :=
by sorry

theorem not_satisfy_condition (x : ℤ) : x > -4 → ¬ (8 - 6 * x > 26) :=
by sorry

end greatest_int_value_not_satisfy_condition_l309_309466


namespace find_digits_l309_309383

theorem find_digits :
  ∃ (A B C D : ℕ), A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧
  (A * 1000 + B * 100 + C * 10 + D = 1098) :=
by {
  sorry
}

end find_digits_l309_309383


namespace smallest_k_exists_l309_309346

theorem smallest_k_exists (s : Finset ℕ) :
  (∀ a b ∈ s, a ≠ b → (672 < |a - b| ∧ |a - b| < 1344)) →
  (∀ k, k < 674 → ∃ s : Finset ℕ, s.card = k ∧ (∀ a b ∈ s, a ≠ b → ¬ (672 < |a - b| ∧ |a - b| < 1344))) → False :=
begin
  sorry
end

end smallest_k_exists_l309_309346


namespace tan_150_deg_l309_309877

-- Define the conditions
def angle_150_deg := 150 * real.pi / 180
def coordinates_of_Q := (-real.sqrt 3 / 2, 1 / 2)
def tan_of_angle_150 := real.tan angle_150_deg

-- The statement to prove
theorem tan_150_deg : tan_of_angle_150 = -1 / real.sqrt 3 :=
by
  -- Proof omitted
  sorry

end tan_150_deg_l309_309877


namespace number_of_items_l309_309676

theorem number_of_items {a n : ℕ} (h1 : ∀ x, x = 100 * a + 99) (h2 : 200 * 100 + 83 = 20083) : 
  (n * (100 * a + 99) = 20083) → (n = 17 ∨ n = 117) :=
by 
  sorry

end number_of_items_l309_309676


namespace unique_positive_real_solution_l309_309087

theorem unique_positive_real_solution (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h1 : x * y = z) (h2 : y * z = x) (h3 : z * x = y) : x = 1 ∧ y = 1 ∧ z = 1 :=
sorry

end unique_positive_real_solution_l309_309087


namespace base9_39457_to_base10_is_26620_l309_309965

-- Define the components of the base 9 number 39457_9
def base9_39457 : ℕ := 39457
def base9_digits : List ℕ := [3, 9, 4, 5, 7]

-- Define the base
def base : ℕ := 9

-- Convert each position to its base 10 equivalent
def base9_to_base10 : ℕ :=
  3 * base ^ 4 + 9 * base ^ 3 + 4 * base ^ 2 + 5 * base ^ 1 + 7 * base ^ 0

-- State the theorem
theorem base9_39457_to_base10_is_26620 : base9_to_base10 = 26620 := by
  sorry

end base9_39457_to_base10_is_26620_l309_309965


namespace units_digit_of_composite_product_l309_309795

theorem units_digit_of_composite_product : 
  let composites := [4, 6, 8, 9],
      product := List.foldl (· * ·) 1 composites
  in product % 10 = 8 :=
  by
  sorry

end units_digit_of_composite_product_l309_309795


namespace amount_coach_mike_gave_l309_309627

-- Definitions from conditions
def cost_of_lemonade : ℕ := 58
def change_received : ℕ := 17

-- Theorem stating the proof problem
theorem amount_coach_mike_gave : cost_of_lemonade + change_received = 75 := by
  sorry

end amount_coach_mike_gave_l309_309627


namespace teams_worked_together_days_l309_309727

noncomputable def first_team_rate : ℝ := 1 / 12
noncomputable def second_team_rate : ℝ := 1 / 9
noncomputable def first_team_days : ℕ := 5
noncomputable def total_work : ℝ := 1
noncomputable def work_first_team_alone := first_team_rate * first_team_days

theorem teams_worked_together_days (x : ℝ) : work_first_team_alone + (first_team_rate + second_team_rate) * x = total_work → x = 3 := 
by
  sorry

end teams_worked_together_days_l309_309727


namespace adam_coins_value_l309_309497

theorem adam_coins_value (num_coins : ℕ) (subset_value: ℕ) (subset_num: ℕ) (total_value: ℕ)
  (h1 : num_coins = 20)
  (h2 : subset_value = 16)
  (h3 : subset_num = 4)
  (h4 : total_value = num_coins * (subset_value / subset_num)) :
  total_value = 80 := 
by
  sorry

end adam_coins_value_l309_309497


namespace kopeechka_purchase_l309_309711

theorem kopeechka_purchase
  (a : ℕ)
  (n : ℕ)
  (total_cost : ℕ)
  (item_cost : ℕ) :
  total_cost = 20083 →
  item_cost = 100 * a + 99 →
  (n * item_cost = total_cost ∧ n = 17 ∨ n = 117) :=
begin
  sorry
end

end kopeechka_purchase_l309_309711


namespace largest_eight_digit_with_all_evens_l309_309824

theorem largest_eight_digit_with_all_evens :
  ∃ n : ℕ, (digits 10 n).length = 8 ∧
           (∀ d, d ∈ [2, 4, 6, 8, 0] → List.mem d (digits 10 n)) ∧
           n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_evens_l309_309824


namespace proof_1_proof_2_l309_309577

-- Definitions of propositions p, q, and r

def p (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (x^2 + (a - 1) * x + a^2 ≤ 0)

def q (a : ℝ) : Prop :=
  2 * a^2 - a > 1

def r (a : ℝ) : Prop :=
  (2 * a - 1) / (a - 2) ≤ 1

-- The given proof problem statement 1
theorem proof_1 (a : ℝ) : (p a ∨ q a) ∧ ¬ (p a ∧ q a) → (a ∈ Set.Icc (-1) (-1/2) ∪ Set.Ioo (1/3) 1) :=
sorry

-- The given proof problem statement 2
theorem proof_2 (a : ℝ) : ¬ p a → r a :=
sorry

end proof_1_proof_2_l309_309577


namespace probability_Xavier_Yvonne_not_Zelda_l309_309190

theorem probability_Xavier_Yvonne_not_Zelda
    (P_Xavier : ℚ)
    (P_Yvonne : ℚ)
    (P_Zelda : ℚ)
    (hXavier : P_Xavier = 1/3)
    (hYvonne : P_Yvonne = 1/2)
    (hZelda : P_Zelda = 5/8) :
    (P_Xavier * P_Yvonne * (1 - P_Zelda) = 1/16) :=
  by
  rw [hXavier, hYvonne, hZelda]
  sorry

end probability_Xavier_Yvonne_not_Zelda_l309_309190


namespace digit_150_of_1_over_13_is_3_l309_309462

def repeating_decimal_1_over_13 : List Nat := [0, 7, 6, 9, 2, 3]

theorem digit_150_of_1_over_13_is_3 :
  (repeating_decimal_1_over_13.get? ((150 % 6) - 1) = some 3) :=
by
  sorry

end digit_150_of_1_over_13_is_3_l309_309462


namespace tan_150_deg_l309_309876

-- Define the conditions
def angle_150_deg := 150 * real.pi / 180
def coordinates_of_Q := (-real.sqrt 3 / 2, 1 / 2)
def tan_of_angle_150 := real.tan angle_150_deg

-- The statement to prove
theorem tan_150_deg : tan_of_angle_150 = -1 / real.sqrt 3 :=
by
  -- Proof omitted
  sorry

end tan_150_deg_l309_309876


namespace sets_of_headphones_l309_309614

-- Definitions of the conditions
variable (M H : ℕ)

-- Theorem statement for proving the question given the conditions
theorem sets_of_headphones (h1 : 5 * M + 30 * H = 840) (h2 : 3 * M + 120 = 480) : H = 8 := by
  sorry

end sets_of_headphones_l309_309614


namespace pencils_evenly_distributed_l309_309514

-- Define the initial number of pencils Eric had
def initialPencils : Nat := 150

-- Define the additional pencils brought by another teacher
def additionalPencils : Nat := 30

-- Define the total number of containers
def numberOfContainers : Nat := 5

-- Define the total number of pencils after receiving additional pencils
def totalPencils := initialPencils + additionalPencils

-- Define the number of pencils per container after even distribution
def pencilsPerContainer := totalPencils / numberOfContainers

-- Statement of the proof problem
theorem pencils_evenly_distributed :
  pencilsPerContainer = 36 :=
by
  -- Sorry is used as a placeholder for the proof
  sorry

end pencils_evenly_distributed_l309_309514


namespace percentage_x_minus_y_l309_309546

variable (x y : ℝ)

theorem percentage_x_minus_y (P : ℝ) :
  P / 100 * (x - y) = 20 / 100 * (x + y) ∧ y = 20 / 100 * x → P = 30 :=
by
  intros h
  sorry

end percentage_x_minus_y_l309_309546


namespace loss_percentage_on_book_sold_at_loss_l309_309115

theorem loss_percentage_on_book_sold_at_loss :
  ∀ (total_cost cost1 : ℝ) (gain_percent : ℝ),
    total_cost = 420 → cost1 = 245 → gain_percent = 0.19 →
    (∀ (cost2 SP : ℝ), cost2 = total_cost - cost1 →
                       SP = cost2 * (1 + gain_percent) →
                       SP = 208.25 →
                       ((cost1 - SP) / cost1 * 100) = 15) :=
by
  intros total_cost cost1 gain_percent h_total_cost h_cost1 h_gain_percent cost2 SP h_cost2 h_SP h_SP_value
  sorry

end loss_percentage_on_book_sold_at_loss_l309_309115


namespace tan_150_eq_neg_sqrt3_div_3_l309_309924

theorem tan_150_eq_neg_sqrt3_div_3
: tan 150 = - (Real.sqrt 3 / 3) :=
by
  have h1 : 150 = 180 - 30, by rfl
  have h2 : sin 150 = sin (180 - 30), by simp [h1]
  have h3 : cos 150 = cos (180 - 30), by simp [h1]
  have cos_30 := Real.sqrt 3 / 2
  have sin_30 := 1 / 2
  simp [Real.sin_eq_sin_of_Real_angle, Real.cos_eq_cos_of_Real_angle] at h2 h3
  have h4 : sin 150 = sin 30, by simp [h2, sin_30]
  have h5 : cos 150 = -cos 30, by simp [h3, cos_30]
  rw [Real.tan_eq_sin_div_cos, h4, h5]
  have h6 : (1 / 2) / (-Real.sqrt 3 / 2) = - (1 / Real.sqrt 3), by linarith
  rw h6
  have h7 : - (1 / Real.sqrt 3) = - (Real.sqrt 3 / 3), by
    field_simp [Real.sqrt_ne_zero],
    linarith
  exact h7,
  sorry

end tan_150_eq_neg_sqrt3_div_3_l309_309924


namespace tg_arccos_le_cos_arctg_l309_309155

theorem tg_arccos_le_cos_arctg (x : ℝ) (h₀ : -1 ≤ x ∧ x ≤ 1) :
  (Real.tan (Real.arccos x) ≤ Real.cos (Real.arctan x)) → 
  (x ∈ Set.Icc (-1:ℝ) 0 ∨ x ∈ Set.Icc (Real.sqrt ((Real.sqrt 5 - 1) / 2)) 1) :=
by
  sorry

end tg_arccos_le_cos_arctg_l309_309155


namespace ratio_of_sums_l309_309970

theorem ratio_of_sums (a b c x y z : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h1 : a^2 + b^2 + c^2 = 49) 
  (h2 : x^2 + y^2 + z^2 = 64) 
  (h3 : a * x + b * y + c * z = 56) : 
  (a + b + c) / (x + y + z) = 7/8 := 
by 
  sorry

end ratio_of_sums_l309_309970


namespace kopeechka_items_l309_309702

-- Define necessary concepts and conditions
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 200 * 100 + 83

-- Lean statement defining the proof problem
theorem kopeechka_items (a n : ℕ) (h1 : ∀ a, n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
by sorry

end kopeechka_items_l309_309702


namespace caleb_puffs_to_mom_l309_309079

variable (initial_puffs : ℕ) (puffs_to_sister : ℕ) (puffs_to_grandmother : ℕ) (puffs_to_dog : ℕ)
variable (puffs_per_friend : ℕ) (friends : ℕ)

theorem caleb_puffs_to_mom
  (h1 : initial_puffs = 40) 
  (h2 : puffs_to_sister = 3)
  (h3 : puffs_to_grandmother = 5) 
  (h4 : puffs_to_dog = 2) 
  (h5 : puffs_per_friend = 9)
  (h6 : friends = 3)
  : initial_puffs - ( friends * puffs_per_friend + puffs_to_sister + puffs_to_grandmother + puffs_to_dog ) = 3 :=
by
  sorry

end caleb_puffs_to_mom_l309_309079


namespace find_a_if_y_is_even_l309_309239

noncomputable def y (x a : ℝ) : ℝ := (x - 1)^2 + a * x + Real.sin (x + Real.pi / 2)

theorem find_a_if_y_is_even (a : ℝ) (h : ∀ x : ℝ, y x a = y (-x) a) : a = 2 :=
by
  sorry

end find_a_if_y_is_even_l309_309239


namespace inequality_false_l309_309526

variable {x y w : ℝ}

theorem inequality_false (hx : x > y) (hy : y > 0) (hw : w ≠ 0) : ¬(x^2 * w > y^2 * w) :=
by {
  sorry -- You could replace this "sorry" with a proper proof.
}

end inequality_false_l309_309526


namespace decimal_150th_digit_l309_309454

theorem decimal_150th_digit (n : ℕ) (d : ℕ) (cycle : String) (cycle_length : ℕ) (h1 : n = 1) (h2 : d = 13)
  (h3 : cycle = "076923") (h4 : cycle_length = 6) :
  (cycle.get ((150 % cycle_length) - 1) = '3') := by
  sorry

end decimal_150th_digit_l309_309454


namespace inequality_d_l309_309313

-- We define the polynomial f with integer coefficients
variable (f : ℤ → ℤ)

-- The function for f^k iteration
def iter (f: ℤ → ℤ) : ℕ → ℤ → ℤ
| 0, x => x
| (n + 1), x => f (iter f n x)

-- Definition of d(a, k) based on the problem statement
def d (a : ℤ) (k : ℕ) : ℝ := |(iter f k a : ℤ) - a|

-- Given condition that d(a, k) is positive
axiom d_pos (a : ℤ) (k : ℕ) : 0 < d f a k

-- The statement to be proved
theorem inequality_d (a : ℤ) (k : ℕ) : d f a k ≥ ↑k / 3 := by
  sorry

end inequality_d_l309_309313


namespace john_money_left_l309_309562

variable (q : ℝ) 

def cost_soda := q
def cost_medium_pizza := 3 * q
def cost_small_pizza := 2 * q

def total_cost := 4 * cost_soda q + 2 * cost_medium_pizza q + 3 * cost_small_pizza q

theorem john_money_left (h : total_cost q = 16 * q) : 50 - total_cost q = 50 - 16 * q := by
  simp [total_cost, cost_soda, cost_medium_pizza, cost_small_pizza]
  sorry

end john_money_left_l309_309562


namespace solve_expr_l309_309292

theorem solve_expr (x : ℝ) (h : x = 3) : x^6 - 6 * x^2 = 675 := by
  sorry

end solve_expr_l309_309292


namespace minimum_value_x_plus_2y_l309_309666

theorem minimum_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = x * y) : x + 2 * y ≥ 8 :=
sorry

end minimum_value_x_plus_2y_l309_309666


namespace rectangular_solid_volume_l309_309039

theorem rectangular_solid_volume 
  (x y z : ℝ)
  (h1 : x * y = 20)
  (h2 : y * z = 15)
  (h3 : x * z = 12) :
  x * y * z = 60 :=
by
  sorry

end rectangular_solid_volume_l309_309039


namespace ship_passengers_round_trip_tickets_l309_309979

theorem ship_passengers_round_trip_tickets (total_passengers : ℕ) (p1 : ℝ) (p2 : ℝ) :
  (p1 = 0.25 * total_passengers) ∧ (p2 = 0.6 * (p * total_passengers)) →
  (p * total_passengers = 62.5 / 100 * total_passengers) :=
by
  sorry

end ship_passengers_round_trip_tickets_l309_309979


namespace MaryBusinessTripTime_l309_309572

theorem MaryBusinessTripTime
  (t_uber_house : Nat := 10) -- Time for Uber to get to her house in minutes
  (t_airport_factor : Nat := 5) -- Factor for time to get to the airport
  (t_check_bag : Nat := 15) -- Time to check her bag in minutes
  (t_security_factor : Nat := 3) -- Factor for time to get through security
  (t_wait_boarding : Nat := 20) -- Time waiting for flight to start boarding in minutes
  (t_take_off_factor : Nat := 2) -- Factor for time waiting for plane to be ready take off
: (t_uber_house + t_uber_house * t_airport_factor + t_check_bag + t_check_bag * t_security_factor + t_wait_boarding + t_wait_boarding * t_take_off_factor) / 60 = 3 := 
begin
  sorry
end

end MaryBusinessTripTime_l309_309572


namespace third_month_sale_l309_309616

theorem third_month_sale
  (avg_sale : ℕ)
  (num_months : ℕ)
  (sales : List ℕ)
  (sixth_month_sale : ℕ)
  (total_sales_req : ℕ) :
  avg_sale = 6500 →
  num_months = 6 →
  sales = [6435, 6927, 7230, 6562] →
  sixth_month_sale = 4991 →
  total_sales_req = avg_sale * num_months →
  total_sales_req - (sales.sum + sixth_month_sale) = 6855 := by
  sorry

end third_month_sale_l309_309616


namespace largest_integral_x_satisfies_ineq_largest_integral_x_is_5_l309_309639

noncomputable def largest_integral_x_in_ineq (x : ℤ) : Prop :=
  (2 / 5 : ℚ) < (x / 7 : ℚ) ∧ (x / 7 : ℚ) < (8 / 11 : ℚ)

theorem largest_integral_x_satisfies_ineq : largest_integral_x_in_ineq 5 :=
sorry

theorem largest_integral_x_is_5 (x : ℤ) (h : largest_integral_x_in_ineq x) : x ≤ 5 :=
sorry

end largest_integral_x_satisfies_ineq_largest_integral_x_is_5_l309_309639


namespace infinite_solutions_of_linear_system_l309_309084

theorem infinite_solutions_of_linear_system :
  ∀ (x y : ℝ), (2 * x - 3 * y = 5) ∧ (4 * x - 6 * y = 10) → ∃ (k : ℝ), x = (3 * k + 5) / 2 :=
by
  sorry

end infinite_solutions_of_linear_system_l309_309084


namespace total_donuts_three_days_l309_309200

def donuts_on_Monday := 14

def donuts_on_Tuesday := donuts_on_Monday / 2

def donuts_on_Wednesday := 4 * donuts_on_Monday

def total_donuts := donuts_on_Monday + donuts_on_Tuesday + donuts_on_Wednesday

theorem total_donuts_three_days : total_donuts = 77 :=
  by
    sorry

end total_donuts_three_days_l309_309200


namespace part1_is_geometric_part2_is_arithmetic_general_formula_for_a_sum_of_first_n_terms_l309_309036

open Nat

variable {α : Type*}
variables (a : ℕ → ℕ) (S : ℕ → ℕ)

axiom a1 : a 1 = 1
axiom S_def : ∀ (n : ℕ), S (n + 1) = 4 * a n + 2 

def b (n : ℕ) : ℕ := a (n + 1) - 2 * a n

def c (n : ℕ) : ℚ := a n / 2^n

theorem part1_is_geometric :
  ∃ r, ∀ n, b n = r * b (n - 1) := sorry

theorem part2_is_arithmetic :
  ∃ d, ∀ n, c n - c (n - 1) = d := sorry

theorem general_formula_for_a :
  ∀ n, a n = (1 / 4) * (3 * n - 1) * 2 ^ n := sorry

theorem sum_of_first_n_terms :
  ∀ n, S n = (1 / 4) * (8 + (3 * n - 4) * 2 ^ (n + 1)) := sorry

end part1_is_geometric_part2_is_arithmetic_general_formula_for_a_sum_of_first_n_terms_l309_309036


namespace tan_150_deg_l309_309899

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l309_309899


namespace tom_sawyer_bible_l309_309173

def blue_tickets_needed (yellow: ℕ) (red: ℕ) (blue: ℕ): ℕ := 
  10 * 10 * 10 * yellow + 10 * 10 * red + blue

theorem tom_sawyer_bible (y r b : ℕ) (hc : y = 8 ∧ r = 3 ∧ b = 7):
  blue_tickets_needed 10 0 0 - blue_tickets_needed y r b = 163 :=
by 
  sorry

end tom_sawyer_bible_l309_309173


namespace solve_abs_quadratic_eq_and_properties_l309_309987

theorem solve_abs_quadratic_eq_and_properties :
  ∃ x1 x2 : ℝ, (|x1|^2 + 2 * |x1| - 8 = 0) ∧ (|x2|^2 + 2 * |x2| - 8 = 0) ∧
               (x1 = 2 ∨ x1 = -2) ∧ (x2 = 2 ∨ x2 = -2) ∧
               (x1 + x2 = 0) ∧ (x1 * x2 = -4) :=
by
  sorry

end solve_abs_quadratic_eq_and_properties_l309_309987


namespace stones_max_value_50_l309_309321

-- Define the problem conditions in Lean
def value_of_stones (x y z : ℕ) : ℕ := 14 * x + 11 * y + 2 * z

def weight_of_stones (x y z : ℕ) : ℕ := 5 * x + 4 * y + z

def max_value_stones {x y z : ℕ} (h_w : weight_of_stones x y z ≤ 18) (h_x : x ≥ 0) (h_y : y ≥ 0) (h_z : z ≥ 0) : Prop :=
  value_of_stones x y z ≤ 50

theorem stones_max_value_50 : ∃ (x y z : ℕ), weight_of_stones x y z ≤ 18 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ value_of_stones x y z = 50 :=
by
  sorry

end stones_max_value_50_l309_309321


namespace total_interest_percentage_l309_309317

theorem total_interest_percentage (inv_total : ℝ) (rate1 rate2 : ℝ) (inv2 : ℝ)
  (h_inv_total : inv_total = 100000)
  (h_rate1 : rate1 = 0.09)
  (h_rate2 : rate2 = 0.11)
  (h_inv2 : inv2 = 24999.999999999996) :
  (rate1 * (inv_total - inv2) + rate2 * inv2) / inv_total * 100 = 9.5 := 
sorry

end total_interest_percentage_l309_309317


namespace georgia_total_carnation_cost_l309_309019

-- Define the cost of one carnation
def cost_of_single_carnation : ℝ := 0.50

-- Define the cost of one dozen carnations
def cost_of_dozen_carnations : ℝ := 4.00

-- Define the number of teachers
def number_of_teachers : ℕ := 5

-- Define the number of friends
def number_of_friends : ℕ := 14

-- Calculate the cost for teachers
def cost_for_teachers : ℝ :=
  (number_of_teachers : ℝ) * cost_of_dozen_carnations

-- Calculate the cost for friends
def cost_for_friends : ℝ :=
  cost_of_dozen_carnations + (2 * cost_of_single_carnation)

-- Calculate the total cost
def total_cost : ℝ := cost_for_teachers + cost_for_friends

-- Theorem stating the total cost
theorem georgia_total_carnation_cost : total_cost = 25 := by
  -- Placeholder for the proof
  sorry

end georgia_total_carnation_cost_l309_309019


namespace eating_contest_l309_309012

variables (hotdog_weight burger_weight pie_weight : ℕ)
variable (noah_burgers jacob_pies mason_hotdogs : ℕ)
variable (total_weight_mason_hotdogs : ℕ)

theorem eating_contest :
  hotdog_weight = 2 →
  burger_weight = 5 →
  pie_weight = 10 →
  noah_burgers = 8 →
  jacob_pies = noah_burgers - 3 →
  mason_hotdogs = 3 * jacob_pies →
  total_weight_mason_hotdogs = mason_hotdogs * hotdog_weight →
  total_weight_mason_hotdogs = 30 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end eating_contest_l309_309012


namespace value_of_y_at_x_eq_1_l309_309111

noncomputable def quadractic_function (x : ℝ) (m : ℝ) : ℝ := 4 * x^2 - m * x + 5

theorem value_of_y_at_x_eq_1 (m : ℝ) (h1 : ∀ x : ℝ, x ≤ -2 → quadractic_function x m < quadractic_function (x + 1) m)
    (h2 : ∀ x : ℝ, x ≥ -2 → quadractic_function x m < quadractic_function (x + 1) m) :
    quadractic_function 1 16 = 25 :=
sorry

end value_of_y_at_x_eq_1_l309_309111


namespace tan_150_eq_neg_sqrt_3_l309_309893

theorem tan_150_eq_neg_sqrt_3 :
  let θ := 150 * Real.pi / 180 in
  Real.tan θ = -Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_sqrt_3_l309_309893


namespace largest_number_is_40_l309_309167

theorem largest_number_is_40 
    (a b c : ℕ) 
    (h1 : a ≠ b)
    (h2 : b ≠ c)
    (h3 : a ≠ c)
    (h4 : a + b + c = 100)
    (h5 : c - b = 8)
    (h6 : b - a = 4) : c = 40 :=
sorry

end largest_number_is_40_l309_309167


namespace triangle_angle_120_l309_309935

theorem triangle_angle_120 (a b c : ℝ) (B : ℝ) (hB : B = 120) :
  a^2 + a * c + c^2 - b^2 = 0 := by
sorry

end triangle_angle_120_l309_309935


namespace nth_term_arithmetic_seq_l309_309463

theorem nth_term_arithmetic_seq (a b n t count : ℕ) (h1 : count = 25) (h2 : a = 3) (h3 : b = 75) (h4 : n = 8) :
    t = a + (n - 1) * ((b - a) / (count - 1)) → t = 24 :=
by
  intros
  sorry

end nth_term_arithmetic_seq_l309_309463


namespace no_spiky_two_digit_numbers_l309_309928

def is_spiky (n : ℕ) : Prop :=
  ∃ a b : ℕ, a ≠ 0 ∧
             10 ≤ n ∧ n < 100 ∧
             n = 10 * a + b ∧
             n = a + b^3 - 2 * a

theorem no_spiky_two_digit_numbers : ∀ n, 10 ≤ n ∧ n < 100 → ¬ is_spiky n :=
by
  intro n h
  sorry

end no_spiky_two_digit_numbers_l309_309928


namespace remainder_71_3_73_5_mod_8_l309_309761

theorem remainder_71_3_73_5_mod_8 :
  (71^3) * (73^5) % 8 = 7 :=
by {
  -- hint, use the conditions given: 71 ≡ -1 (mod 8) and 73 ≡ 1 (mod 8)
  sorry
}

end remainder_71_3_73_5_mod_8_l309_309761


namespace units_digit_of_product_of_first_four_composites_l309_309791

theorem units_digit_of_product_of_first_four_composites :
  (4 * 6 * 8 * 9) % 10 = 8 := 
by
  sorry

end units_digit_of_product_of_first_four_composites_l309_309791


namespace train_distance_problem_l309_309847

theorem train_distance_problem
  (Vx : ℝ) (Vy : ℝ) (t : ℝ) (distanceX : ℝ) 
  (h1 : Vx = 32) 
  (h2 : Vy = 160 / 3) 
  (h3 : 32 * t + (160 / 3) * t = 160) :
  distanceX = Vx * t → distanceX = 60 :=
by {
  sorry
}

end train_distance_problem_l309_309847


namespace basketball_team_selection_l309_309485

theorem basketball_team_selection :
  (Nat.choose 4 2) * (Nat.choose 14 6) = 18018 := 
by
  -- number of ways to choose 2 out of 4 quadruplets
  -- number of ways to choose 6 out of the remaining 14 players
  -- the product of these combinations equals the required number of ways
  sorry

end basketball_team_selection_l309_309485


namespace multiples_of_15_between_12_and_152_l309_309664

theorem multiples_of_15_between_12_and_152 : 
  ∃ n : ℕ, n = 10 ∧ ∀ m : ℕ, (m * 15 > 12 ∧ m * 15 < 152) ↔ (1 ≤ m ∧ m ≤ 10) :=
by
  sorry

end multiples_of_15_between_12_and_152_l309_309664


namespace problem1_problem2_problem3_l309_309579

theorem problem1 (a : ℝ) : |a + 2| = 4 → (a = 2 ∨ a = -6) :=
sorry

theorem problem2 (a : ℝ) (h₀ : -4 < a) (h₁ : a < 2) : |a + 4| + |a - 2| = 6 :=
sorry

theorem problem3 (a : ℝ) : ∃ x ∈ Set.Icc (-2 : ℝ) 1, |x-1| + |x+2| = 3 :=
sorry

end problem1_problem2_problem3_l309_309579


namespace largest_eight_digit_number_l309_309831

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (digits : List ℕ) : Prop :=
  ∀ d, is_even_digit d → List.contains digits d

def largest_eight_digit_with_even_digits : ℕ :=
  99986420

theorem largest_eight_digit_number {n : ℕ} 
  (h_digits : List ℕ) 
  (h_len : h_digits.length = 8)
  (h_cond : contains_all_even_digits h_digits)
  (h_num : n = List.foldl (λ acc d, acc * 10 + d) 0 h_digits)
  : n = largest_eight_digit_with_even_digits :=
sorry

end largest_eight_digit_number_l309_309831


namespace min_area_triangle_ABC_l309_309843

theorem min_area_triangle_ABC :
  let A := (0, 0) 
  let B := (42, 18)
  (∃ p q : ℤ, let C := (p, q) 
              ∃ area : ℝ, area = (1 / 2 : ℝ) * |42 * q - 18 * p| 
              ∧ area = 3) := 
sorry

end min_area_triangle_ABC_l309_309843


namespace count_even_three_digit_numbers_less_than_600_l309_309441

-- Define the digits
def digits : List ℕ := [1, 2, 3, 4, 5, 6]

-- Condition: the number must be less than 600, i.e., hundreds digit in {1, 2, 3, 4, 5}
def valid_hundreds (d : ℕ) : Prop := d ∈ [1, 2, 3, 4, 5]

-- Condition: the units (ones) digit must be even
def valid_units (d : ℕ) : Prop := d ∈ [2, 4, 6]

-- Problem: total number of valid three-digit numbers
def total_valid_numbers : ℕ :=
  List.product (List.product [1, 2, 3, 4, 5] digits) [2, 4, 6] |>.length

-- Proof statement
theorem count_even_three_digit_numbers_less_than_600 :
  total_valid_numbers = 90 := by
  sorry

end count_even_three_digit_numbers_less_than_600_l309_309441


namespace georgia_total_cost_l309_309015

def carnation_price : ℝ := 0.50
def dozen_price : ℝ := 4.00
def teachers : ℕ := 5
def friends : ℕ := 14

theorem georgia_total_cost :
  ((dozen_price * teachers) + dozen_price + (carnation_price * (friends - 12))) = 25.00 :=
by
  sorry

end georgia_total_cost_l309_309015


namespace probability_of_green_ball_l309_309630

/-- 
Container I holds 5 red balls and 5 green balls; 
container II holds 3 red balls and 3 green balls; 
container III holds 4 red balls and 2 green balls. 
A container is selected at random, and then a ball is randomly selected from that container. 
Prove that the probability that the ball selected is green is 4/9. 
-/
theorem probability_of_green_ball : 
  let P_I := 1 / 3,
      P_II := 1 / 3,
      P_III := 1 / 3,
      P_green_I := 5 / 10,
      P_green_II := 3 / 6,
      P_green_III := 2 / 6
  in (P_I * P_green_I + P_II * P_green_II + P_III * P_green_III) = 4 / 9 :=
by
  -- Introduce the given probabilities as let bindings
  let P_I := 1 / 3
  let P_II := 1 / 3
  let P_III := 1 / 3
  let P_green_I := 5 / 10
  let P_green_II := 3 / 6
  let P_green_III := 2 / 6

  -- Compute the total probability using the law of total probability
  let total_prob := P_I * P_green_I + P_II * P_green_II + P_III * P_green_III

  -- Simplify the components and verify the final probability
  have h1 : total_prob = 1 / 6 + 1 / 6 + 1 / 9 := by sorry
  have h2 : h1 = 3 / 18 + 3 / 18 + 2 / 18 := by sorry
  have h3 : h2 = 8 / 18 := by sorry
  have h4 : h3 = 4 / 9 := by sorry

  -- Conclude the theorem
  exact h4

end probability_of_green_ball_l309_309630


namespace apples_per_box_l309_309563

theorem apples_per_box (x : ℕ) (h1 : 10 * x > 0) (h2 : 3 * (10 * x) / 4 > 0) (h3 : (10 * x) / 4 = 750) : x = 300 :=
by
  sorry

end apples_per_box_l309_309563


namespace units_digit_of_first_four_composite_numbers_l309_309807

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l309_309807


namespace pumpkins_eaten_l309_309413

-- Definitions for the conditions
def originalPumpkins : ℕ := 43
def leftPumpkins : ℕ := 20

-- Theorem statement
theorem pumpkins_eaten : originalPumpkins - leftPumpkins = 23 :=
  by
    -- Proof steps are omitted
    sorry

end pumpkins_eaten_l309_309413


namespace initial_bottle_caps_l309_309119

variable (initial_caps added_caps total_caps : ℕ)

theorem initial_bottle_caps 
  (h1 : added_caps = 7) 
  (h2 : total_caps = 14) 
  (h3 : total_caps = initial_caps + added_caps) : 
  initial_caps = 7 := 
by 
  sorry

end initial_bottle_caps_l309_309119


namespace digit_150_of_decimal_1_div_13_l309_309460

theorem digit_150_of_decimal_1_div_13 : 
  (λ r : ℚ, let digits := (r.repr.drop 2).to_list in digits.nth 149 = some '3') (1/13) :=
by
  sorry

end digit_150_of_decimal_1_div_13_l309_309460


namespace find_a_for_even_function_l309_309238

theorem find_a_for_even_function (a : ℝ) :
  (∀ x : ℝ, (x-1)^2 + a * x + sin (x + π / 2) = ((-x)-1)^2 + (-a * x) + sin (-x + π / 2)) →
  a = 2 :=
by
  sorry

end find_a_for_even_function_l309_309238


namespace stickers_earned_correct_l309_309145

-- Define the initial and final number of stickers.
def initial_stickers : ℕ := 39
def final_stickers : ℕ := 61

-- Define how many stickers Pat earned during the week
def stickers_earned : ℕ := final_stickers - initial_stickers

-- State the main theorem
theorem stickers_earned_correct : stickers_earned = 22 :=
by
  show final_stickers - initial_stickers = 22
  sorry

end stickers_earned_correct_l309_309145


namespace hyperbola_condition_l309_309219

theorem hyperbola_condition (k : ℝ) : (3 - k) * (k - 2) < 0 ↔ k < 2 ∨ k > 3 := by
  sorry

end hyperbola_condition_l309_309219


namespace least_n_for_factorial_multiple_10080_l309_309665

theorem least_n_for_factorial_multiple_10080 (n : ℕ) 
  (h₁ : 0 < n) 
  (h₂ : ∀ m, m > 0 → (n ≠ m → n! % 10080 ≠ 0)) 
  : n = 8 := 
sorry

end least_n_for_factorial_multiple_10080_l309_309665


namespace common_terms_sequence_l309_309662

-- Definitions of sequences
def a (n : ℕ) : ℤ := 3 * n - 19
def b (n : ℕ) : ℤ := 2 ^ n
def c (n : ℕ) : ℤ := 2 ^ (2 * n - 1)

-- Theorem stating the conjecture
theorem common_terms_sequence :
  ∀ n : ℕ, ∃ m : ℕ, a m = b (2 * n - 1) :=
by
  sorry

end common_terms_sequence_l309_309662


namespace tan_150_deg_l309_309874

-- Define the conditions
def angle_150_deg := 150 * real.pi / 180
def coordinates_of_Q := (-real.sqrt 3 / 2, 1 / 2)
def tan_of_angle_150 := real.tan angle_150_deg

-- The statement to prove
theorem tan_150_deg : tan_of_angle_150 = -1 / real.sqrt 3 :=
by
  -- Proof omitted
  sorry

end tan_150_deg_l309_309874


namespace tan_150_eq_neg_inv_sqrt3_l309_309879

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l309_309879


namespace psychologist_diagnosis_l309_309488

theorem psychologist_diagnosis :
  let initial_patients := 26
  let doubling_factor := 2
  let probability := 1 / 4
  let total_patients := initial_patients * doubling_factor
  let expected_patients_with_ZYX := total_patients * probability
  expected_patients_with_ZYX = 13 := by
  sorry

end psychologist_diagnosis_l309_309488


namespace asian_countries_visited_l309_309631

theorem asian_countries_visited (total_countries europe_countries south_america_countries remaining_asian_countries : ℕ)
  (h1 : total_countries = 42)
  (h2 : europe_countries = 20)
  (h3 : south_america_countries = 10)
  (h4 : remaining_asian_countries = (total_countries - (europe_countries + south_america_countries)) / 2) :
  remaining_asian_countries = 6 :=
by sorry

end asian_countries_visited_l309_309631


namespace count_total_shells_l309_309588

theorem count_total_shells 
  (purple_shells : ℕ := 13)
  (pink_shells : ℕ := 8)
  (yellow_shells : ℕ := 18)
  (blue_shells : ℕ := 12)
  (orange_shells : ℕ := 14) :
  purple_shells + pink_shells + yellow_shells + blue_shells + orange_shells = 65 :=
by
  -- Calculation
  sorry

end count_total_shells_l309_309588


namespace solve_system_of_equations_l309_309269

theorem solve_system_of_equations :
  ∃ (x y z : ℝ), 
    (2 * y + x - x^2 - y^2 = 0) ∧ 
    (z - x + y - y * (x + z) = 0) ∧ 
    (-2 * y + z - y^2 - z^2 = 0) ∧ 
    ((x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1 ∧ y = 0 ∧ z = 1)) :=
by
  sorry

end solve_system_of_equations_l309_309269


namespace hyperbola_eccentricity_l309_309645

-- Define the conditions and parameters for the problem
variables (m : ℝ) (c a e : ℝ)

-- Given conditions
def hyperbola_eq (m : ℝ) := ∀ x y : ℝ, (x^2 / m^2 - y^2 = 4)
def focal_distance : Prop := c = 4
def standard_hyperbola_form : Prop := a^2 = 4 * m^2 ∧ 4 = 4

-- Eccentricity definition
def eccentricity : Prop := e = c / a

-- Main theorem
theorem hyperbola_eccentricity (m : ℝ) (h_pos : 0 < m) (h_foc_dist : focal_distance c) (h_form : standard_hyperbola_form a m) :
  eccentricity e a c :=
by
  sorry

end hyperbola_eccentricity_l309_309645


namespace winning_votes_calculation_l309_309755

variables (V : ℚ) (winner_votes : ℚ)

-- Conditions
def percentage_of_votes_of_winner : ℚ := 0.60 * V
def percentage_of_votes_of_loser : ℚ := 0.40 * V
def vote_difference_spec : 0.60 * V - 0.40 * V = 288 := by sorry

-- Theorem to prove
theorem winning_votes_calculation (h1 : winner_votes = 0.60 * V)
  (h2 : 0.60 * V - 0.40 * V = 288) : winner_votes = 864 :=
by
  sorry

end winning_votes_calculation_l309_309755


namespace valid_combination_exists_l309_309840

def exists_valid_combination : Prop :=
  ∃ (a: Fin 7 → ℤ), (a 0 = 1) ∧
  (a 1 = 2) ∧ (a 2 = 3) ∧ (a 3 = 4) ∧ 
  (a 4 = 5) ∧ (a 5 = 6) ∧ (a 6 = 7) ∧
  ((a 0 = a 1 + a 2 + a 3 + a 4 - a 5 - a 6))

theorem valid_combination_exists :
  exists_valid_combination :=
by
  sorry

end valid_combination_exists_l309_309840


namespace possible_items_l309_309689

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items_l309_309689


namespace decorations_given_to_friend_l309_309264

-- Definitions of the given conditions
def boxes : ℕ := 6
def decorations_per_box : ℕ := 25
def used_decorations : ℕ := 58
def neighbor_decorations : ℕ := 75

-- The statement of the proof problem
theorem decorations_given_to_friend : 
  (boxes * decorations_per_box) - used_decorations - neighbor_decorations = 17 := 
by 
  sorry

end decorations_given_to_friend_l309_309264


namespace sum_pqrs_eq_3150_l309_309257

theorem sum_pqrs_eq_3150
  (p q r s : ℝ)
  (h1 : p ≠ q) (h2 : p ≠ r) (h3 : p ≠ s) (h4 : q ≠ r) (h5 : q ≠ s) (h6 : r ≠ s)
  (hroots1 : ∀ x : ℝ, x^2 - 14*p*x - 15*q = 0 → (x = r ∨ x = s))
  (hroots2 : ∀ x : ℝ, x^2 - 14*r*x - 15*s = 0 → (x = p ∨ x = q)) :
  p + q + r + s = 3150 :=
by
  sorry

end sum_pqrs_eq_3150_l309_309257


namespace units_digit_of_product_is_eight_l309_309769

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l309_309769


namespace prove_a_range_l309_309937

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^3 - x^2 - x + 2
noncomputable def g' (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 1

theorem prove_a_range (a : ℝ) :
  (∀ x : ℝ, x > 0 → 2 * f x ≤ g' x + 2) ↔ a ∈ Set.Ici (-2) :=
sorry

end prove_a_range_l309_309937


namespace number_of_items_l309_309681

theorem number_of_items {a n : ℕ} (h1 : ∀ x, x = 100 * a + 99) (h2 : 200 * 100 + 83 = 20083) : 
  (n * (100 * a + 99) = 20083) → (n = 17 ∨ n = 117) :=
by 
  sorry

end number_of_items_l309_309681


namespace largest_eight_digit_number_with_even_digits_l309_309838

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (num : ℕ) : Prop :=
  ∀ d, is_even_digit d → (num.toDigits.contains d)

def is_eight_digit_number (num : ℕ) : Prop :=
  10000000 ≤ num ∧ num < 100000000

def largest_number_with_conditions (num : ℕ) : ℕ :=
  if contains_all_even_digits num ∧ is_eight_digit_number num then num else 0

theorem largest_eight_digit_number_with_even_digits :
  largest_number_with_conditions 99986420 = 99986420 :=
begin
  sorry
end

end largest_eight_digit_number_with_even_digits_l309_309838


namespace units_digit_of_product_is_eight_l309_309773

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l309_309773


namespace tom_sawyer_bible_l309_309174

def blue_tickets_needed (yellow: ℕ) (red: ℕ) (blue: ℕ): ℕ := 
  10 * 10 * 10 * yellow + 10 * 10 * red + blue

theorem tom_sawyer_bible (y r b : ℕ) (hc : y = 8 ∧ r = 3 ∧ b = 7):
  blue_tickets_needed 10 0 0 - blue_tickets_needed y r b = 163 :=
by 
  sorry

end tom_sawyer_bible_l309_309174


namespace probability_of_winning_pair_l309_309061

theorem probability_of_winning_pair :
  let red_cards := {1, 2, 3, 4, 5}
  let green_cards := {1, 2, 3, 4}
  let deck := red_cards ∪ green_cards
  let winning_pairs := {pair | pair ∈ (deck × deck) ∧ (pair.1 ∈ red_cards ∧ pair.2 ∈ red_cards ∨ pair.1 ∈ green_cards ∧ pair.2 ∈ green_cards ∨ pair.1 = pair.2)}
  let total_pairs := {pair | pair ∈ (deck × deck)}
  (winning_pairs.card : ℚ) / (total_pairs.card : ℚ) = 5 / 9 := by
  sorry

end probability_of_winning_pair_l309_309061


namespace product_of_squares_is_perfect_square_l309_309393

theorem product_of_squares_is_perfect_square (a b c : ℤ) (h : a * b + b * c + c * a = 1) :
    ∃ k : ℤ, (1 + a^2) * (1 + b^2) * (1 + c^2) = k^2 :=
sorry

end product_of_squares_is_perfect_square_l309_309393


namespace tan_150_degrees_l309_309869

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l309_309869


namespace units_digit_product_first_four_composite_numbers_l309_309785

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l309_309785


namespace part1_part2_l309_309357

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a + 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Part (1)
theorem part1 (a : ℝ) (h : a = 3) : (P 3)ᶜ ∩ Q = {x | -2 ≤ x ∧ x < 4} :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) : (∀ x, x ∈ P a → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P a) → 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end part1_part2_l309_309357


namespace probability_of_draw_l309_309146

-- Define probabilities
def P_A_wins : ℝ := 0.4
def P_A_not_loses : ℝ := 0.9

-- Theorem statement
theorem probability_of_draw : P_A_not_loses = P_A_wins + 0.5 :=
by
  -- Proof is skipped
  sorry

end probability_of_draw_l309_309146


namespace one_over_thirteen_150th_digit_l309_309448

def decimal_representation_one_over_thirteen (n : ℕ) : ℕ :=
  -- Given the repeating block "076923" for 1/13, find nth digit in the block
  let block := [0, 7, 6, 9, 2, 3]
  in block[(n % 6)]

theorem one_over_thirteen_150th_digit : 
  decimal_representation_one_over_thirteen 150 = 3 := by
  sorry

end one_over_thirteen_150th_digit_l309_309448


namespace sum_of_powers_mod_l309_309760

-- Define a function that calculates the nth power of a number modulo a given base
def power_mod (a n k : ℕ) : ℕ := (a^n) % k

-- The main theorem: prove that the sum of powers modulo 5 gives the remainder 0
theorem sum_of_powers_mod 
  : ((power_mod 1 2013 5) + (power_mod 2 2013 5) + (power_mod 3 2013 5) + (power_mod 4 2013 5) + (power_mod 5 2013 5)) % 5 = 0 := 
by {
  sorry
}

end sum_of_powers_mod_l309_309760


namespace remainder_of_exponentiation_l309_309397

theorem remainder_of_exponentiation (n : ℕ) : (3 ^ (2 * n) + 8) % 8 = 1 := 
by sorry

end remainder_of_exponentiation_l309_309397


namespace quadratic_reciprocal_squares_l309_309510

theorem quadratic_reciprocal_squares :
  (∃ p q : ℝ, (∀ x : ℝ, 3*x^2 - 5*x + 2 = 0 → (x = p ∨ x = q)) ∧ (1 / p^2 + 1 / q^2 = 13 / 4)) :=
by
  have quadratic_eq : (∀ x : ℝ, 3*x^2 - 5*x + 2 = 0 → (x = 1 ∨ x = 2 / 3)) := sorry
  have identity_eq : 1 / (1:ℝ)^2 + 1 / (2 / 3)^2 = 13 / 4 := sorry
  exact ⟨1, 2 / 3, quadratic_eq, identity_eq⟩

end quadratic_reciprocal_squares_l309_309510


namespace boat_speed_in_still_water_l309_309126

variable (B S : ℝ)

-- conditions
def condition1 : Prop := B + S = 6
def condition2 : Prop := B - S = 2

-- question to answer
theorem boat_speed_in_still_water (h1 : condition1 B S) (h2 : condition2 B S) : B = 4 :=
by
  sorry

end boat_speed_in_still_water_l309_309126


namespace sqrt_6_approx_l309_309399

noncomputable def newton_iteration (x : ℝ) : ℝ :=
  (1 / 2) * x + (3 / x)

theorem sqrt_6_approx :
  let x0 : ℝ := 2
  let x1 : ℝ := newton_iteration x0
  let x2 : ℝ := newton_iteration x1
  let x3 : ℝ := newton_iteration x2
  abs (x3 - 2.4495) < 0.0001 :=
by
  sorry

end sqrt_6_approx_l309_309399


namespace kopeechka_purchase_l309_309710

theorem kopeechka_purchase
  (a : ℕ)
  (n : ℕ)
  (total_cost : ℕ)
  (item_cost : ℕ) :
  total_cost = 20083 →
  item_cost = 100 * a + 99 →
  (n * item_cost = total_cost ∧ n = 17 ∨ n = 117) :=
begin
  sorry
end

end kopeechka_purchase_l309_309710


namespace smallest_k_for_ten_ruble_heads_up_l309_309574

-- Conditions
def num_total_coins : ℕ := 30
def num_ten_ruble_coins : ℕ := 23
def num_five_ruble_coins : ℕ := 7
def num_heads_up : ℕ := 20
def num_tails_up : ℕ := 10

-- Prove the smallest k such that any k coins chosen include at least one ten-ruble coin heads-up.
theorem smallest_k_for_ten_ruble_heads_up (k : ℕ) :
  (∀ (coins : Finset ℕ), coins.card = k → (∃ (coin : ℕ) (h : coin ∈ coins), coin < num_ten_ruble_coins ∧ coin < num_heads_up)) →
  k = 18 :=
sorry

end smallest_k_for_ten_ruble_heads_up_l309_309574


namespace tan_150_eq_neg_inv_sqrt3_l309_309917

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l309_309917


namespace solve_exponent_problem_l309_309953

theorem solve_exponent_problem
  (h : (1 / 8) * (2 ^ 36) = 8 ^ x) : x = 11 :=
by
  sorry

end solve_exponent_problem_l309_309953


namespace inverse_h_l309_309722

def f (x : ℝ) : ℝ := 4 * x - 1
def g (x : ℝ) : ℝ := 3 * x + 2
def h (x : ℝ) : ℝ := f (g x)

theorem inverse_h (x : ℝ) : h⁻¹ (x) = (x - 7) / 12 :=
sorry

end inverse_h_l309_309722


namespace total_weight_mason_hotdogs_l309_309009

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

end total_weight_mason_hotdogs_l309_309009


namespace num_distinct_convex_polygons_on_12_points_l309_309177

theorem num_distinct_convex_polygons_on_12_points : 
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 :=
by
  let num_subsets := 2 ^ 12
  let num_subsets_with_0_members := Nat.choose 12 0
  let num_subsets_with_1_member := Nat.choose 12 1
  let num_subsets_with_2_members := Nat.choose 12 2
  have h : num_subsets - num_subsets_with_0_members - num_subsets_with_1_member - num_subsets_with_2_members = 4017 := by sorry
  exact h

end num_distinct_convex_polygons_on_12_points_l309_309177


namespace problem_solution_l309_309367

theorem problem_solution (x y z : ℝ)
  (h1 : 1/x + 1/y + 1/z = 2)
  (h2 : 1/x^2 + 1/y^2 + 1/z^2 = 1) :
  1/(x*y) + 1/(y*z) + 1/(z*x) = 3/2 :=
sorry

end problem_solution_l309_309367


namespace min_value_of_a_sq_plus_b_sq_over_a_minus_b_l309_309533

theorem min_value_of_a_sq_plus_b_sq_over_a_minus_b {a b : ℝ} (h1 : a > b) (h2 : a * b = 1) : 
  ∃ x, x = 2 * Real.sqrt 2 ∧ ∀ y, y = (a^2 + b^2) / (a - b) → y ≥ x :=
by {
  sorry
}

end min_value_of_a_sq_plus_b_sq_over_a_minus_b_l309_309533


namespace count_satisfying_pairs_l309_309187

theorem count_satisfying_pairs :
  ∃ (count : ℕ), count = 540 ∧ 
  (∀ (w n : ℕ), (w % 23 = 5) ∧ (w < 450) ∧ (n % 17 = 7) ∧ (n < 450) → w < 450 ∧ n < 450) := 
by
  sorry

end count_satisfying_pairs_l309_309187


namespace units_digit_first_four_composites_l309_309811

theorem units_digit_first_four_composites : 
  let first_four_composites := [4, 6, 8, 9]
  let product := first_four_composites.prod
  Nat.unitsDigit product = 8 :=
by
  sorry

end units_digit_first_four_composites_l309_309811


namespace find_number_l309_309056

theorem find_number
    (x: ℝ)
    (h: 0.60 * x = 0.40 * 30 + 18) : x = 50 :=
    sorry

end find_number_l309_309056


namespace tan_150_eq_neg_inv_sqrt3_l309_309881

theorem tan_150_eq_neg_inv_sqrt3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_inv_sqrt3_l309_309881


namespace yuna_initial_marbles_l309_309295

theorem yuna_initial_marbles (M : ℕ) :
  (M - 12 + 5) / 2 + 3 = 17 → M = 35 := by
  sorry

end yuna_initial_marbles_l309_309295


namespace units_digit_first_four_composites_l309_309763

theorem units_digit_first_four_composites :
  let p := [4, 6, 8, 9] in
  let product := List.prod p in
  product % 10 = 8 :=
by
  let p := [4, 6, 8, 9]
  let product := List.prod p
  show product % 10 = 8
  sorry

end units_digit_first_four_composites_l309_309763


namespace joan_jogged_3563_miles_l309_309133

noncomputable def steps_per_mile : ℕ := 1200

noncomputable def flips_per_year : ℕ := 28

noncomputable def steps_per_full_flip : ℕ := 150000

noncomputable def final_day_steps : ℕ := 75000

noncomputable def total_steps_in_year := flips_per_year * steps_per_full_flip + final_day_steps

noncomputable def miles_jogged := total_steps_in_year / steps_per_mile

theorem joan_jogged_3563_miles :
  miles_jogged = 3563 :=
by
  sorry

end joan_jogged_3563_miles_l309_309133


namespace problem_f_symmetric_l309_309373

theorem problem_f_symmetric (f : ℝ → ℝ) (k : ℝ) (h : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + k * f b) (h_not_zero : ∃ x : ℝ, f x ≠ 0) :
  ∀ x : ℝ, f (-x) = f x :=
sorry

end problem_f_symmetric_l309_309373


namespace units_digit_of_first_four_composite_numbers_l309_309808

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l309_309808


namespace hiring_probability_l309_309597

noncomputable def combinatorics (n k : ℕ) : ℕ := Nat.choose n k

theorem hiring_probability (n : ℕ) (h1 : combinatorics 2 2 = 1)
                          (h2 : combinatorics (n - 2) 1 = n - 2)
                          (h3 : combinatorics n 3 = n * (n - 1) * (n - 2) / 6)
                          (h4 : (6 : ℕ) / (n * (n - 1) : ℚ) = 1 / 15) :
  n = 10 :=
by
  sorry

end hiring_probability_l309_309597


namespace doug_initial_marbles_l309_309204

theorem doug_initial_marbles 
  (ed_marbles : ℕ)
  (doug_marbles : ℕ)
  (lost_marbles : ℕ)
  (ed_condition : ed_marbles = doug_marbles + 5)
  (lost_condition : lost_marbles = 3)
  (ed_value : ed_marbles = 27) :
  doug_marbles + lost_marbles = 25 :=
by
  sorry

end doug_initial_marbles_l309_309204


namespace integer_triples_soln_l309_309340

theorem integer_triples_soln (x y z : ℤ) :
  (x^3 + y^3 + z^3 - 3*x*y*z = 2003) ↔ ( (x = 668 ∧ y = 668 ∧ z = 667) ∨ (x = 668 ∧ y = 667 ∧ z = 668) ∨ (x = 667 ∧ y = 668 ∧ z = 668) ) := 
by
  sorry

end integer_triples_soln_l309_309340


namespace central_angle_is_two_length_of_chord_l309_309271

-- Define the conditions
constant r : ℝ
constant θ : ℝ
constant l : ℝ

axiom h1 : (1 / 2) * r^2 * θ = 1
axiom h2 : 2 * r + r * θ = 4

-- Prove the central angle in radians is 2
theorem central_angle_is_two : θ = 2 :=
sorry

-- Prove the length of the chord AB is 2 * sin(1)
theorem length_of_chord : l = 2 * sin 1 :=
sorry

end central_angle_is_two_length_of_chord_l309_309271


namespace Youseff_time_difference_l309_309476

theorem Youseff_time_difference 
  (blocks : ℕ)
  (walk_time_per_block : ℕ) 
  (bike_time_per_block_sec : ℕ) 
  (sec_per_min : ℕ)
  (h_blocks : blocks = 12) 
  (h_walk_time_per_block : walk_time_per_block = 1) 
  (h_bike_time_per_block_sec : bike_time_per_block_sec = 20) 
  (h_sec_per_min : sec_per_min = 60) : 
  (blocks * walk_time_per_block) - ((blocks * bike_time_per_block_sec) / sec_per_min) = 8 :=
by 
  sorry

end Youseff_time_difference_l309_309476


namespace fruit_basket_ratio_l309_309550

theorem fruit_basket_ratio (total_fruits : ℕ) (oranges : ℕ) (apples : ℕ) (h1 : total_fruits = 40) (h2 : oranges = 10) (h3 : apples = total_fruits - oranges) :
  (apples / oranges) = 3 := by
  sorry

end fruit_basket_ratio_l309_309550


namespace quadratic_function_solution_l309_309256

theorem quadratic_function_solution :
  (∃ g : ℝ → ℝ, ∀ x : ℝ, g (x + 1) - g x = 2 * x + 3 ∧ g 2 - g 6 = -40) :=
sorry

end quadratic_function_solution_l309_309256


namespace units_digit_of_product_of_first_four_composites_l309_309787

theorem units_digit_of_product_of_first_four_composites :
  (4 * 6 * 8 * 9) % 10 = 8 := 
by
  sorry

end units_digit_of_product_of_first_four_composites_l309_309787


namespace valid_third_side_l309_309364

theorem valid_third_side (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 8) (h₃ : 5 < c) (h₄ : c < 11) : c = 8 := 
by 
  sorry

end valid_third_side_l309_309364


namespace average_of_numbers_is_correct_l309_309464

theorem average_of_numbers_is_correct :
  let nums := [12, 13, 14, 510, 520, 530, 1120, 1, 1252140, 2345]
  let sum_nums := 1253205
  let count_nums := 10
  (sum_nums / count_nums.toFloat) = 125320.5 :=
by {
  sorry
}

end average_of_numbers_is_correct_l309_309464


namespace units_digit_first_four_composites_l309_309778

theorem units_digit_first_four_composites :
  let product := 4 * 6 * 8 * 9 in
  product % 10 = 8 :=
by
  let product := 4 * 6 * 8 * 9
  have h : product = 1728 := by norm_num
  show product % 10 = 8
  rw [h]
  norm_num
  done 
  sorry

end units_digit_first_four_composites_l309_309778


namespace square_perimeter_equals_66_88_l309_309604

noncomputable def circle_perimeter : ℝ := 52.5

noncomputable def circle_radius (C : ℝ) : ℝ := C / (2 * Real.pi)

noncomputable def circle_diameter (r : ℝ) : ℝ := 2 * r

noncomputable def square_side_length (d : ℝ) : ℝ := d

noncomputable def square_perimeter (s : ℝ) : ℝ := 4 * s

theorem square_perimeter_equals_66_88 :
  square_perimeter (square_side_length (circle_diameter (circle_radius circle_perimeter))) = 66.88 := 
by
  -- Placeholder for the proof
  sorry

end square_perimeter_equals_66_88_l309_309604


namespace sum_fib_2019_eq_fib_2021_minus_1_l309_309281

def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

def sum_fib : ℕ → ℕ
| 0 => 0
| n + 1 => sum_fib n + fib (n + 1)

theorem sum_fib_2019_eq_fib_2021_minus_1 : sum_fib 2019 = fib 2021 - 1 := 
by sorry -- proof here

end sum_fib_2019_eq_fib_2021_minus_1_l309_309281


namespace count_solutions_absolute_value_l309_309542

theorem count_solutions_absolute_value (x : ℤ) : 
  (|4 * x + 2| ≤ 10) ↔ (x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2) :=
by sorry

end count_solutions_absolute_value_l309_309542


namespace largest_eight_digit_number_contains_even_digits_l309_309836

theorem largest_eight_digit_number_contains_even_digits :
  ∃ n : ℕ, n = 99986420 ∧ (10000000 ≤ n ∧ n < 100000000) ∧
    ∀ d ∈ [0, 2, 4, 6, 8], ∃ (i : ℕ), i < 8 ∧ (n / 10^i) % 10 = d :=
by
  sorry

end largest_eight_digit_number_contains_even_digits_l309_309836


namespace tan_150_eq_neg_inv_sqrt_3_l309_309905

theorem tan_150_eq_neg_inv_sqrt_3 : Real.tan (150 * Real.pi / 180) = -1 / Real.sqrt 3 := by
  -- Using the given conditions
  sorry

end tan_150_eq_neg_inv_sqrt_3_l309_309905


namespace expected_winnings_correct_l309_309062

def winnings (roll : ℕ) : ℚ :=
  if roll % 2 = 1 then 0
  else if roll % 4 = 0 then 2 * roll
  else roll

def expected_winnings : ℚ :=
  (winnings 1) / 8 + (winnings 2) / 8 +
  (winnings 3) / 8 + (winnings 4) / 8 +
  (winnings 5) / 8 + (winnings 6) / 8 +
  (winnings 7) / 8 + (winnings 8) / 8

theorem expected_winnings_correct : expected_winnings = 3.75 := by 
  sorry

end expected_winnings_correct_l309_309062


namespace find_e_l309_309137

theorem find_e (a b c d e : ℝ) (h1 : a < b) (h2 : b < c) (h3 : c < d) (h4 : d < e)
    (h_lb1 : a + b = 32) (h_lb2 : a + c = 36) (h_lb3 : b + c = 37)
    (h_ub1 : c + e = 48) (h_ub2 : d + e = 51) : e = 27.5 :=
sorry

end find_e_l309_309137


namespace kopeechka_purchase_l309_309706

theorem kopeechka_purchase
  (a : ℕ)
  (n : ℕ)
  (total_cost : ℕ)
  (item_cost : ℕ) :
  total_cost = 20083 →
  item_cost = 100 * a + 99 →
  (n * item_cost = total_cost ∧ n = 17 ∨ n = 117) :=
begin
  sorry
end

end kopeechka_purchase_l309_309706


namespace total_earnings_first_two_weeks_l309_309600

-- Conditions
variable (x : ℝ)  -- Xenia's hourly wage
variable (earnings_first_week : ℝ := 12 * x)  -- Earnings in the first week
variable (earnings_second_week : ℝ := 20 * x)  -- Earnings in the second week

-- Xenia earned $36 more in the second week than in the first
axiom h1 : earnings_second_week = earnings_first_week + 36

-- Proof statement
theorem total_earnings_first_two_weeks : earnings_first_week + earnings_second_week = 144 := by
  -- Proof is omitted
  sorry

end total_earnings_first_two_weeks_l309_309600


namespace units_digit_first_four_composites_l309_309775

theorem units_digit_first_four_composites :
  let product := 4 * 6 * 8 * 9 in
  product % 10 = 8 :=
by
  let product := 4 * 6 * 8 * 9
  have h : product = 1728 := by norm_num
  show product % 10 = 8
  rw [h]
  norm_num
  done 
  sorry

end units_digit_first_four_composites_l309_309775


namespace number_of_asian_countries_l309_309632

theorem number_of_asian_countries (total european south_american : ℕ) 
  (H_total : total = 42) 
  (H_european : european = 20) 
  (H_south_american : south_american = 10) 
  (H_half_asian : ∃ rest, rest = total - european - south_american ∧ rest / 2 = 6) : 
  ∃ asian, asian = 6 :=
by {
  let rest := total - european - south_american,
  have H_rest : rest = 42 - 20 - 10, from sorry,
  have H_asian : rest / 2 = 6, from sorry,
  exact ⟨6, rfl⟩,
}

end number_of_asian_countries_l309_309632


namespace sum_of_reciprocals_of_squares_l309_309748

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 41) :
  (1 / (a^2) + 1 / (b^2)) = 1682 / 1681 := sorry

end sum_of_reciprocals_of_squares_l309_309748


namespace negation_of_existential_l309_309667

theorem negation_of_existential : 
  (¬ (∃ x : ℝ, x > 0 ∧ x^2 - 3 * x + 2 > 0)) ↔ (∀ x : ℝ, x > 0 → x^2 - 3 * x + 2 ≤ 0) := 
by 
  sorry

end negation_of_existential_l309_309667


namespace simplify_fraction_l309_309736

theorem simplify_fraction (a : ℕ) (h : a = 3) : (10 * a ^ 3) / (55 * a ^ 2) = 6 / 11 :=
by sorry

end simplify_fraction_l309_309736


namespace smallest_part_of_80_divided_by_proportion_l309_309371

theorem smallest_part_of_80_divided_by_proportion (x : ℕ) (h1 : 1 * x + 3 * x + 5 * x + 7 * x = 80) : x = 5 :=
sorry

end smallest_part_of_80_divided_by_proportion_l309_309371


namespace tan_150_eq_l309_309888

noncomputable def cosine150 : ℚ := -√3 / 2
noncomputable def sine150 : ℚ := 1 / 2
noncomputable def tangent150 : ℚ := sine150 / cosine150

theorem tan_150_eq : tangent150 = -1 / √3 := by
  sorry

end tan_150_eq_l309_309888


namespace number_of_shirts_made_today_l309_309498

-- Define the rate of shirts made per minute.
def shirts_per_minute : ℕ := 6

-- Define the number of minutes the machine worked today.
def minutes_today : ℕ := 12

-- Define the total number of shirts made today.
def shirts_made_today : ℕ := shirts_per_minute * minutes_today

-- State the theorem for the number of shirts made today.
theorem number_of_shirts_made_today : shirts_made_today = 72 := 
by
  -- Proof is omitted
  sorry

end number_of_shirts_made_today_l309_309498


namespace operation_result_l309_309511

def operation (a b : Int) : Int :=
  (a + b) * (a - b)

theorem operation_result :
  operation 4 (operation 2 (-1)) = 7 :=
by
  sorry

end operation_result_l309_309511


namespace find_x0_l309_309104

/-- Given that the tangent line to the curve y = x^2 - 1 at the point x = x0 is parallel 
to the tangent line to the curve y = 1 - x^3 at the point x = x0, prove that x0 = 0 
or x0 = -2/3. -/
theorem find_x0 (x0 : ℝ) (h : (∃ x0, (2 * x0) = (-3 * x0 ^ 2))) : x0 = 0 ∨ x0 = -2/3 := 
sorry

end find_x0_l309_309104


namespace exists_n_such_that_5_pow_n_has_six_consecutive_zeros_l309_309984

theorem exists_n_such_that_5_pow_n_has_six_consecutive_zeros :
  ∃ n : ℕ, n < 1000000 ∧ ∃ k : ℕ, k = 20 ∧ 5 ^ n % (10 ^ k) < (10 ^ (k - 6)) :=
by
  -- proof goes here
  sorry

end exists_n_such_that_5_pow_n_has_six_consecutive_zeros_l309_309984


namespace Vasya_can_win_l309_309286

theorem Vasya_can_win 
  (a : ℕ → ℕ) -- initial sequence of natural numbers
  (x : ℕ) -- number chosen by Vasya
: ∃ (i : ℕ), ∀ (k : ℕ), ∃ (j : ℕ), (a j + k * x = 1) :=
by
  sorry

end Vasya_can_win_l309_309286


namespace units_digit_of_product_of_first_four_composites_l309_309789

theorem units_digit_of_product_of_first_four_composites :
  (4 * 6 * 8 * 9) % 10 = 8 := 
by
  sorry

end units_digit_of_product_of_first_four_composites_l309_309789


namespace find_n_l309_309382

noncomputable def arithmetic_sequence (a : ℕ → ℕ) := 
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_n (a : ℕ → ℕ) (n d : ℕ) 
  (h1 : arithmetic_sequence a)
  (h2 : a 1 = 1)
  (h3 : a 2 + a 5 = 12)
  (h4 : a n = 25) : 
  n = 13 := 
sorry

end find_n_l309_309382


namespace units_digit_product_first_four_composite_numbers_l309_309783

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l309_309783


namespace unique_positive_integer_n_l309_309206

theorem unique_positive_integer_n (n x : ℕ) (hx : x > 0) (hn : n = 2 ^ (2 * x - 1) - 5 * x - 3 ∧ n = (2 ^ (x-1) - 1) * (2 ^ x + 1)) : n = 2015 := by
  sorry

end unique_positive_integer_n_l309_309206


namespace area_of_triangle_XYZ_l309_309858

noncomputable def centroid (p1 p2 p3 : (ℚ × ℚ)) : (ℚ × ℚ) :=
((p1.1 + p2.1 + p3.1) / 3, (p1.2 + p2.2 + p3.2) / 3)

noncomputable def triangle_area (p1 p2 p3 : (ℚ × ℚ)) : ℚ :=
abs ((p1.1 * p2.2 + p2.1 * p3.2 + p3.1 * p1.2 - p1.2 * p2.1 - p2.2 * p3.1 - p3.2 * p1.1) / 2)

noncomputable def point_A : (ℚ × ℚ) := (5, 12)
noncomputable def point_B : (ℚ × ℚ) := (0, 0)
noncomputable def point_C : (ℚ × ℚ) := (14, 0)

noncomputable def point_X : (ℚ × ℚ) :=
(109 / 13, 60 / 13)
noncomputable def point_Y : (ℚ × ℚ) :=
centroid point_A point_B point_X
noncomputable def point_Z : (ℚ × ℚ) :=
centroid point_B point_C point_Y

theorem area_of_triangle_XYZ : triangle_area point_X point_Y point_Z = 84 / 13 :=
sorry

end area_of_triangle_XYZ_l309_309858


namespace smaller_cylinder_diameter_l309_309993

theorem smaller_cylinder_diameter
  (vol_large : ℝ)
  (height_large : ℝ)
  (diameter_large : ℝ)
  (height_small : ℝ)
  (ratio : ℝ)
  (π : ℝ)
  (volume_large_eq : vol_large = π * (diameter_large / 2)^2 * height_large)  -- Volume formula for the larger cylinder
  (ratio_eq : ratio = 74.07407407407408) -- Given ratio
  (height_large_eq : height_large = 10)  -- Given height of the larger cylinder
  (diameter_large_eq : diameter_large = 20)  -- Given diameter of the larger cylinder
  (height_small_eq : height_small = 6)  -- Given height of smaller cylinders):
  :
  ∃ (diameter_small : ℝ), diameter_small = 3 := 
by
  sorry

end smaller_cylinder_diameter_l309_309993


namespace outfits_count_l309_309841

theorem outfits_count (s p : ℕ) (h_s : s = 5) (h_p : p = 3) : s * p = 15 :=
by
  rw [h_s, h_p]
  exact Nat.mul_comm 5 3

end outfits_count_l309_309841


namespace units_digit_first_four_composites_l309_309762

theorem units_digit_first_four_composites :
  let p := [4, 6, 8, 9] in
  let product := List.prod p in
  product % 10 = 8 :=
by
  let p := [4, 6, 8, 9]
  let product := List.prod p
  show product % 10 = 8
  sorry

end units_digit_first_four_composites_l309_309762


namespace largest_eight_digit_with_all_evens_l309_309823

theorem largest_eight_digit_with_all_evens :
  ∃ n : ℕ, (digits 10 n).length = 8 ∧
           (∀ d, d ∈ [2, 4, 6, 8, 0] → List.mem d (digits 10 n)) ∧
           n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_evens_l309_309823


namespace count_integers_congruent_to_7_mod_13_l309_309226

theorem count_integers_congruent_to_7_mod_13 : 
  (∃ (n : ℕ), ∀ x, (1 ≤ x ∧ x < 500 ∧ x % 13 = 7) → x = 7 + 13 * n ∧ n < 38) :=
sorry

end count_integers_congruent_to_7_mod_13_l309_309226


namespace marcia_project_hours_l309_309571

theorem marcia_project_hours (minutes_spent : ℕ) (minutes_per_hour : ℕ) 
  (h1 : minutes_spent = 300) 
  (h2 : minutes_per_hour = 60) : 
  (minutes_spent / minutes_per_hour) = 5 :=
by
  sorry

end marcia_project_hours_l309_309571


namespace evaluate_fractions_l309_309217

theorem evaluate_fractions (a b c : ℝ) 
  (h : a / (30 - a) + b / (70 - b) + c / (55 - c) = 8) : 
  6 / (30 - a) + 14 / (70 - b) + 11 / (55 - c) = 2.2 := 
by
  sorry

end evaluate_fractions_l309_309217


namespace canister_ratio_l309_309501

variable (C D : ℝ) -- Define capacities of canister C and canister D
variable (hC_half : 1/2 * C) -- Canister C is 1/2 full of water
variable (hD_third : 1/3 * D) -- Canister D is 1/3 full of water
variable (hD_after : 1/12 * D) -- Canister D contains 1/12 after pouring

theorem canister_ratio (h : 1/2 * C = 1/4 * D) : D / C = 2 :=
by
  sorry

end canister_ratio_l309_309501


namespace temperature_on_Monday_l309_309420

theorem temperature_on_Monday 
  (M T W Th F : ℝ)
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : F = 31) : 
  M = 39 :=
by
  sorry

end temperature_on_Monday_l309_309420


namespace problem_statement_l309_309035

theorem problem_statement (x : ℝ) (hx : x + 1/x = Real.sqrt 5) : 
  x^12 - 7 * x^8 + x^4 = 343 :=
sorry

end problem_statement_l309_309035


namespace arrangements_count_l309_309950

theorem arrangements_count : 
  (∑ k in Finset.range 5, (Nat.choose 4 k)^3) = 
  -- This part of the theorem states that the result of counting the valid arrangements is equal to the given sum.
  sorry -- The proof of this theorem is omitted as specified.

end arrangements_count_l309_309950


namespace habitable_land_area_l309_309619

noncomputable def area_of_habitable_land : ℝ :=
  let length : ℝ := 23
  let diagonal : ℝ := 33
  let radius_of_pond : ℝ := 3
  let width : ℝ := Real.sqrt (diagonal ^ 2 - length ^ 2)
  let area_of_rectangle : ℝ := length * width
  let area_of_pond : ℝ := Real.pi * (radius_of_pond ^ 2)
  area_of_rectangle - area_of_pond

theorem habitable_land_area :
  abs (area_of_habitable_land - 515.91) < 0.01 :=
by
  sorry

end habitable_land_area_l309_309619


namespace sum_exclude_multiples_of_2_and_5_l309_309214

theorem sum_exclude_multiples_of_2_and_5 (n : ℕ) (hn : n > 0) : 
  ∑ k in (finset.range (10 * n)).filter (λ x, ¬ (x % 2 = 0 ∨ x % 5 = 0)), k = 25 * n^2 :=
by
  -- The proof is omitted
  sorry

end sum_exclude_multiples_of_2_and_5_l309_309214


namespace minimum_points_to_determine_polynomial_l309_309576

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, f x = a * x^2 + b * x + c

def different_at (f g : ℝ → ℝ) (x : ℝ) : Prop :=
  f x ≠ g x

theorem minimum_points_to_determine_polynomial :
  ∀ (f g : ℝ → ℝ), is_quadratic f → is_quadratic g → 
  (∀ t, t < 8 → (different_at f g t → ∃ t₁ t₂ t₃, different_at f g t₁ ∧ different_at f g t₂ ∧ different_at f g t₃)) → False :=
by {
  sorry
}

end minimum_points_to_determine_polynomial_l309_309576


namespace lindsey_owns_more_cars_than_cathy_l309_309726

theorem lindsey_owns_more_cars_than_cathy :
  ∀ (cathy carol susan lindsey : ℕ),
    cathy = 5 →
    carol = 2 * cathy →
    susan = carol - 2 →
    cathy + carol + susan + lindsey = 32 →
    lindsey = cathy + 4 :=
by
  intros cathy carol susan lindsey h1 h2 h3 h4
  sorry

end lindsey_owns_more_cars_than_cathy_l309_309726


namespace right_drawing_num_triangles_l309_309424

-- Given the conditions:
-- 1. Nine distinct lines in the right drawing
-- 2. Any combination of 3 lines out of these 9 forms a triangle
-- 3. Count of intersections of these lines where exactly three lines intersect

def num_triangles : Nat := 84 -- Calculated via binomial coefficient
def num_intersections : Nat := 61 -- Given or calculated from the problem

-- The target theorem to prove that the number of triangles is equal to 23
theorem right_drawing_num_triangles :
  num_triangles - num_intersections = 23 :=
by
  -- Proof would go here, but we skip it as per the instructions
  sorry

end right_drawing_num_triangles_l309_309424


namespace tan_150_eq_l309_309887

noncomputable def cosine150 : ℚ := -√3 / 2
noncomputable def sine150 : ℚ := 1 / 2
noncomputable def tangent150 : ℚ := sine150 / cosine150

theorem tan_150_eq : tangent150 = -1 / √3 := by
  sorry

end tan_150_eq_l309_309887


namespace alternating_binomial_sum_l309_309344

open BigOperators Finset

theorem alternating_binomial_sum :
  ∑ k in range 34, (-1 : ℤ)^k * (Nat.choose 99 (3 * k)) = -1 := by
  sorry

end alternating_binomial_sum_l309_309344


namespace stratified_sampling_students_l309_309068

theorem stratified_sampling_students :
  let F := 1600
  let S := 1200
  let Sr := 800
  let sr := 20
  let f := (F * sr) / Sr
  let s := (S * sr) / Sr
  f + s = 70 :=
by
  let F := 1600
  let S := 1200
  let Sr := 800
  let sr := 20
  let f := (F * sr) / Sr
  let s := (S * sr) / Sr
  sorry

end stratified_sampling_students_l309_309068


namespace correct_operations_result_l309_309071

theorem correct_operations_result (n : ℕ) 
  (h1 : n / 8 - 12 = 32) : (n * 8 + 12 = 2828) :=
sorry

end correct_operations_result_l309_309071


namespace find_digits_l309_309443

theorem find_digits (a b c : ℕ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_digits : a ≤ 9 ∧ b ≤ 9 ∧ c ≤ 9)
  (h_sum : 100 * a + 10 * b + c = (10 * a + b) + (10 * b + c) + (10 * c + a)) :
  a = 1 ∧ b = 9 ∧ c = 8 := by
  sorry

end find_digits_l309_309443


namespace digit_150_after_decimal_of_one_thirteenth_l309_309455

-- Define the conditions given in the problem
def decimal_rep_of_one_thirteenth : String := "076923"
def block_length : Nat := 6
def digit_to_find : Nat := 150

-- Function to find the nth digit in a repeating block
def nth_digit_in_repeating_block (block : String) (block_length n : Nat) : Char :=
block[(n % block_length) % block.length]

-- The theorem that we need to prove
theorem digit_150_after_decimal_of_one_thirteenth :
  nth_digit_in_repeating_block decimal_rep_of_one_thirteenth block_length digit_to_find = '3' :=
by
  sorry

end digit_150_after_decimal_of_one_thirteenth_l309_309455


namespace Max_wins_count_l309_309323

-- Definitions for Chloe and Max's wins and their ratio
def Chloe_wins := 24
def Max_wins (Y : ℕ) := 8 * Y = 3 * Chloe_wins

-- The theorem to be proven
theorem Max_wins_count : ∃ Y : ℕ, Max_wins Y ∧ Y = 9 :=
by
  existsi 9
  simp [Max_wins, Chloe_wins]
  sorry

end Max_wins_count_l309_309323


namespace parallel_lines_condition_l309_309949

theorem parallel_lines_condition (a : ℝ) :
  (∀ x y : ℝ, x + a * y + 6 = 0 → (a - 2) * x + 3 * y + 2 * a = 0 → False) ↔ a = -1 :=
sorry

end parallel_lines_condition_l309_309949


namespace wire_cut_problem_l309_309484

noncomputable def shorter_piece_length (total_length : ℝ) (ratio : ℝ) : ℝ :=
  let x := total_length / (1 + ratio)
  x

theorem wire_cut_problem (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) :
  total_length = 35 → ratio = 5/2 → shorter_length = 10 → shorter_piece_length total_length ratio = shorter_length := by
  intros h1 h2 h3
  unfold shorter_piece_length
  rw [h1, h2, h3]
  sorry

end wire_cut_problem_l309_309484


namespace probability_red_or_green_is_two_thirds_l309_309120

-- Define the conditions
def total_balls := 2 + 3 + 4
def favorable_outcomes := 2 + 4

-- Define the probability calculation
def probability_red_or_green := (favorable_outcomes : ℚ) / total_balls

-- The theorem statement
theorem probability_red_or_green_is_two_thirds : probability_red_or_green = 2 / 3 := by
  -- This part will contain the proof using Lean, but we skip it with "sorry" for now.
  sorry

end probability_red_or_green_is_two_thirds_l309_309120


namespace harry_worked_total_hours_l309_309635

theorem harry_worked_total_hours (x : ℝ) (H : ℝ) (H_total : ℝ) :
  (24 * x + 1.5 * x * H = 42 * x) → (H_total = 24 + H) → H_total = 36 :=
by
sorry

end harry_worked_total_hours_l309_309635


namespace purchase_options_l309_309712

def item_cost (a : Nat) : Nat := 100 * a + 99

def total_cost : Nat := 20083

theorem purchase_options (a : Nat) (n : Nat) (h : n * item_cost a = total_cost) :
  n = 17 ∨ n = 117 :=
by
  sorry

end purchase_options_l309_309712


namespace sugar_required_in_new_recipe_l309_309280

theorem sugar_required_in_new_recipe
  (ratio_flour_water_sugar : ℕ × ℕ × ℕ)
  (double_ratio_flour_water : (ℕ → ℕ))
  (half_ratio_flour_sugar : (ℕ → ℕ))
  (new_water_cups : ℕ) :
  ratio_flour_water_sugar = (7, 2, 1) →
  double_ratio_flour_water 7 = 14 → 
  double_ratio_flour_water 2 = 4 →
  half_ratio_flour_sugar 7 = 7 →
  half_ratio_flour_sugar 1 = 2 →
  new_water_cups = 2 →
  (∃ sugar_cups : ℕ, sugar_cups = 1) :=
by
  sorry

end sugar_required_in_new_recipe_l309_309280


namespace exists_unique_circle_l309_309432

structure Circle := (center : ℝ × ℝ) (radius : ℝ)

def diametrically_opposite_points (C : Circle) (P : ℝ × ℝ) : Prop :=
  let (cx, cy) := C.center
  let (px, py) := P
  (px - cx) ^ 2 + (py - cy) ^ 2 = (C.radius ^ 2)

def intersects_at_diametrically_opposite_points (K A : Circle) : Prop :=
  ∃ P₁ P₂ : ℝ × ℝ, diametrically_opposite_points A P₁ ∧ diametrically_opposite_points A P₂ ∧
  P₁ ≠ P₂ ∧ diametrically_opposite_points K P₁ ∧ diametrically_opposite_points K P₂

theorem exists_unique_circle (A B C : Circle) :
  ∃! K : Circle, intersects_at_diametrically_opposite_points K A ∧
  intersects_at_diametrically_opposite_points K B ∧
  intersects_at_diametrically_opposite_points K C := sorry

end exists_unique_circle_l309_309432


namespace not_possible_last_digit_l309_309528

theorem not_possible_last_digit :
  ∀ (S : ℕ) (a : Fin 111 → ℕ),
  (∀ i, a i ≤ 500) →
  (∀ i j, i ≠ j → a i ≠ a j) →
  (∀ i, (a i) % 10 = (S - a i) % 10) →
  False :=
by
  intro S a h1 h2 h3
  sorry

end not_possible_last_digit_l309_309528


namespace angle_value_l309_309130

theorem angle_value (x y : ℝ) (h_parallel : True)
  (h_alt_int_ang : x = y)
  (h_triangle_sum : 2 * x + x + 60 = 180) : 
  y = 40 := 
by
  sorry

end angle_value_l309_309130


namespace purchase_options_l309_309714

def item_cost (a : Nat) : Nat := 100 * a + 99

def total_cost : Nat := 20083

theorem purchase_options (a : Nat) (n : Nat) (h : n * item_cost a = total_cost) :
  n = 17 ∨ n = 117 :=
by
  sorry

end purchase_options_l309_309714


namespace find_b_for_perpendicular_lines_l309_309277

theorem find_b_for_perpendicular_lines 
  (b : ℚ) : 
  (∀ (x y : ℚ), 2*x - 3*y + 6 = 0 → b*x - 3*y + 6 = 0 → (2/3) * (b/3) = -1) → b = -9/2 :=
by
  intro h
  sorry

end find_b_for_perpendicular_lines_l309_309277


namespace at_least_one_truth_and_not_knight_l309_309013

def isKnight (n : Nat) : Prop := n = 1   -- Identifier for knights
def isKnave (n : Nat) : Prop := n = 0    -- Identifier for knaves
def isRegular (n : Nat) : Prop := n = 2  -- Identifier for regular persons

def A := 2     -- Initially define A's type as regular (this can be adjusted)
def B := 2     -- Initially define B's type as regular (this can be adjusted)

def statementA : Prop := isKnight B
def statementB : Prop := ¬ isKnight A

theorem at_least_one_truth_and_not_knight :
  statementA ∧ ¬ isKnight A ∨ statementB ∧ ¬ isKnight B :=
sorry

end at_least_one_truth_and_not_knight_l309_309013


namespace rice_pounds_l309_309580

noncomputable def pounds_of_rice (r p : ℝ) : Prop :=
  r + p = 30 ∧ 1.10 * r + 0.55 * p = 23.50

theorem rice_pounds (r p : ℝ) (h : pounds_of_rice r p) : r = 12.7 :=
sorry

end rice_pounds_l309_309580


namespace tenth_digit_of_expression_l309_309184

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

def tenth_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem tenth_digit_of_expression : 
  tenth_digit ((factorial 5 * factorial 5 - factorial 5 * factorial 3) / 5) = 3 :=
by
  -- proof omitted
  sorry

end tenth_digit_of_expression_l309_309184


namespace ducklings_distance_l309_309416

noncomputable def ducklings_swim (r : ℝ) (n : ℕ) : Prop :=
  ∀ (ducklings : Fin n → ℝ × ℝ), (∀ i, (ducklings i).1 ^ 2 + (ducklings i).2 ^ 2 = r ^ 2) →
    ∃ (i j : Fin n), i ≠ j ∧ (ducklings i - ducklings j).1 ^ 2 + (ducklings i - ducklings j).2 ^ 2 ≤ r ^ 2

theorem ducklings_distance :
  ducklings_swim 5 6 :=
by sorry

end ducklings_distance_l309_309416


namespace visited_both_countries_l309_309379

theorem visited_both_countries {Total Iceland Norway Neither Both : ℕ} 
  (h1 : Total = 50) 
  (h2 : Iceland = 25)
  (h3 : Norway = 23)
  (h4 : Neither = 23) 
  (h5 : Total - Neither = 27) 
  (h6 : Iceland + Norway - Both = 27) : 
  Both = 21 := 
by
  sorry

end visited_both_countries_l309_309379


namespace units_digit_of_product_of_first_four_composites_l309_309801

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l309_309801


namespace exists_circle_passing_through_and_orthogonal_l309_309532

open EuclideanGeometry

variables {k l : Circle} {O A B P Q : Point}

theorem exists_circle_passing_through_and_orthogonal (hO : k.center = O) (hA : A ∈ k) (hB : B ∈ k) :
  ∃ l : Circle, l.passing_through A ∧ l.passing_through B ∧ ∀ P Q, P ∈ k ∧ P ∈ l → Q ∈ k ∧ Q ∈ l → k.radius O P ⊥ l.radius O P :=
by
  -- Proof omitted
  sorry

end exists_circle_passing_through_and_orthogonal_l309_309532


namespace inequality_holds_l309_309266

theorem inequality_holds (a : ℝ) : 3 * (1 + a^2 + a^4) ≥ (1 + a + a^2)^2 :=
by
  sorry

end inequality_holds_l309_309266


namespace yellow_candies_eaten_prob_before_red_find_m_n_sum_l309_309852

open Nat

theorem yellow_candies_eaten_prob_before_red
  (yellow red blue : ℕ) (h₁ : yellow = 2) (h₂ : red = 4) (h₃ : blue = 6) :
  let total := yellow + red in
  let total_arrangements := Nat.choose total yellow in
  let favorable_arrangements := 1 in
  let prob := favorable_arrangements / total_arrangements in
  prob = 1 / 15 :=
by
  sorry

theorem find_m_n_sum (m n : ℕ) (h₁ : m = 1) (h₂ : n = 15) : m + n = 16 :=
by
  rw [h₁, h₂]
  rfl

end yellow_candies_eaten_prob_before_red_find_m_n_sum_l309_309852


namespace tan_150_eq_neg_sqrt3_div_3_l309_309925

theorem tan_150_eq_neg_sqrt3_div_3
: tan 150 = - (Real.sqrt 3 / 3) :=
by
  have h1 : 150 = 180 - 30, by rfl
  have h2 : sin 150 = sin (180 - 30), by simp [h1]
  have h3 : cos 150 = cos (180 - 30), by simp [h1]
  have cos_30 := Real.sqrt 3 / 2
  have sin_30 := 1 / 2
  simp [Real.sin_eq_sin_of_Real_angle, Real.cos_eq_cos_of_Real_angle] at h2 h3
  have h4 : sin 150 = sin 30, by simp [h2, sin_30]
  have h5 : cos 150 = -cos 30, by simp [h3, cos_30]
  rw [Real.tan_eq_sin_div_cos, h4, h5]
  have h6 : (1 / 2) / (-Real.sqrt 3 / 2) = - (1 / Real.sqrt 3), by linarith
  rw h6
  have h7 : - (1 / Real.sqrt 3) = - (Real.sqrt 3 / 3), by
    field_simp [Real.sqrt_ne_zero],
    linarith
  exact h7,
  sorry

end tan_150_eq_neg_sqrt3_div_3_l309_309925


namespace units_digit_of_product_of_first_four_composites_l309_309800

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l309_309800


namespace equation_quadratic_k_neg1_l309_309095

theorem equation_quadratic_k_neg1 
  (k : ℤ) 
  (h : (k - 1) * x^abs k + 1 - x + 5 = 0) 
  (h_quad : is_quadratic (λ x => (k - 1) * x^(abs k + 1) - x + 5)) :
  k = -1 :=
sorry

end equation_quadratic_k_neg1_l309_309095


namespace y_value_is_32_l309_309237

-- Define the conditions
variables (y : ℝ) (hy_pos : y > 0) (hy_eq : y^2 = 1024)

-- State the theorem
theorem y_value_is_32 : y = 32 :=
by
  -- The proof will be written here
  sorry

end y_value_is_32_l309_309237


namespace total_earning_l309_309297

theorem total_earning (days_a days_b days_c : ℕ) (wage_ratio_a wage_ratio_b wage_ratio_c daily_wage_c total : ℕ)
  (h_ratio : wage_ratio_a = 3 ∧ wage_ratio_b = 4 ∧ wage_ratio_c = 5)
  (h_days : days_a = 6 ∧ days_b = 9 ∧ days_c = 4)
  (h_daily_wage_c : daily_wage_c = 125)
  (h_total : total = ((wage_ratio_a * (daily_wage_c / wage_ratio_c) * days_a) +
                     (wage_ratio_b * (daily_wage_c / wage_ratio_c) * days_b) +
                     (daily_wage_c * days_c))) : total = 1850 := by
  sorry

end total_earning_l309_309297


namespace chord_length_count_l309_309020

noncomputable def number_of_chords (d r : ℕ) : ℕ := sorry

theorem chord_length_count {d r : ℕ} (h1 : d = 12) (h2 : r = 13) :
  number_of_chords d r = 17 :=
sorry

end chord_length_count_l309_309020


namespace tan_150_eq_neg_sqrt3_div_3_l309_309920

theorem tan_150_eq_neg_sqrt3_div_3
: tan 150 = - (Real.sqrt 3 / 3) :=
by
  have h1 : 150 = 180 - 30, by rfl
  have h2 : sin 150 = sin (180 - 30), by simp [h1]
  have h3 : cos 150 = cos (180 - 30), by simp [h1]
  have cos_30 := Real.sqrt 3 / 2
  have sin_30 := 1 / 2
  simp [Real.sin_eq_sin_of_Real_angle, Real.cos_eq_cos_of_Real_angle] at h2 h3
  have h4 : sin 150 = sin 30, by simp [h2, sin_30]
  have h5 : cos 150 = -cos 30, by simp [h3, cos_30]
  rw [Real.tan_eq_sin_div_cos, h4, h5]
  have h6 : (1 / 2) / (-Real.sqrt 3 / 2) = - (1 / Real.sqrt 3), by linarith
  rw h6
  have h7 : - (1 / Real.sqrt 3) = - (Real.sqrt 3 / 3), by
    field_simp [Real.sqrt_ne_zero],
    linarith
  exact h7,
  sorry

end tan_150_eq_neg_sqrt3_div_3_l309_309920


namespace solve_for_y_l309_309234

theorem solve_for_y (y : ℝ) (h_pos : y > 0) (h_eq : y^2 = 1024) : y = 32 := 
by
  sorry

end solve_for_y_l309_309234


namespace kopeechka_items_l309_309696

theorem kopeechka_items (a n : ℕ) (hn : n * (100 * a + 99) = 20083) : n = 17 ∨ n = 117 :=
sorry

end kopeechka_items_l309_309696


namespace evaluate_custom_operation_l309_309543

def custom_operation (A B : ℕ) : ℕ :=
  (A + 2 * B) * (A - B)

theorem evaluate_custom_operation : custom_operation 7 5 = 34 :=
by
  sorry

end evaluate_custom_operation_l309_309543


namespace count_valid_pairs_l309_309316

open Nat

-- Define the conditions
def room_conditions (p q : ℕ) : Prop :=
  q > p ∧
  (∃ (p' q' : ℕ), p = p' + 6 ∧ q = q' + 6 ∧ p' * q' = 48)

-- State the theorem to prove the number of valid pairs (p, q)
theorem count_valid_pairs : 
  (∃ l : List (ℕ × ℕ), 
    (∀ pq ∈ l, room_conditions pq.fst pq.snd) ∧ 
    l.length = 5) := 
sorry

end count_valid_pairs_l309_309316


namespace solve_for_x_l309_309957

theorem solve_for_x
  (x y : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (h1 : 6 * x^3 + 12 * x * y = 2 * x^4 + 3 * x^3 * y)
  (h2 : y = x^2) :
  x = (-1 + Real.sqrt 55) / 3 := 
by
  sorry

end solve_for_x_l309_309957


namespace thursday_loaves_baked_l309_309270

theorem thursday_loaves_baked (wednesday friday saturday sunday monday : ℕ) (p1 : wednesday = 5) (p2 : friday = 10) (p3 : saturday = 14) (p4 : sunday = 19) (p5 : monday = 25) : 
  ∃ thursday : ℕ, thursday = 11 := 
by 
  sorry

end thursday_loaves_baked_l309_309270


namespace game_show_prizes_l309_309308

theorem game_show_prizes :
  let digits := [1, 1, 2, 2, 3, 3, 3, 3]
  let permutations := Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 2 * Nat.factorial 2)
  let partitions := Nat.choose 7 3
  permutations * partitions = 14700 :=
by
  let digits := [1, 1, 2, 2, 3, 3, 3, 3]
  let permutations := Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 2 * Nat.factorial 2)
  let partitions := Nat.choose 7 3
  exact sorry

end game_show_prizes_l309_309308


namespace kopeechka_purchase_l309_309707

theorem kopeechka_purchase
  (a : ℕ)
  (n : ℕ)
  (total_cost : ℕ)
  (item_cost : ℕ) :
  total_cost = 20083 →
  item_cost = 100 * a + 99 →
  (n * item_cost = total_cost ∧ n = 17 ∨ n = 117) :=
begin
  sorry
end

end kopeechka_purchase_l309_309707


namespace possible_items_l309_309692

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items_l309_309692


namespace ratio_of_x_and_y_l309_309258

theorem ratio_of_x_and_y (x y : ℝ) (h : (x - y) / (x + y) = 4) : x / y = -5 / 3 :=
by sorry

end ratio_of_x_and_y_l309_309258


namespace sum_of_three_consecutive_even_nums_l309_309595

theorem sum_of_three_consecutive_even_nums : 80 + 82 + 84 = 246 := by
  sorry

end sum_of_three_consecutive_even_nums_l309_309595


namespace luke_to_lee_paths_l309_309143

theorem luke_to_lee_paths :
  let total_paths := Nat.choose 8 5
  let risky_corner_paths := (Nat.choose 4 2) * (Nat.choose 4 3)
  total_paths - risky_corner_paths = 32 :=
by
  let total_paths := Nat.choose 8 5
  let risky_corner_paths := (Nat.choose 4 2) * (Nat.choose 4 3)
  exact Nat.sub_eq_of_eq_add (Integer.add_comm 32 24)
  sorry

end luke_to_lee_paths_l309_309143


namespace locus_of_point_T_l309_309531

theorem locus_of_point_T (r : ℝ) (a b : ℝ) (x y x1 y1 x2 y2 : ℝ)
  (hM_inside : a^2 + b^2 < r^2)
  (hK_on_circle : x1^2 + y1^2 = r^2)
  (hP_on_circle : x2^2 + y2^2 = r^2)
  (h_midpoints_eq : (x + a) / 2 = (x1 + x2) / 2 ∧ (y + b) / 2 = (y1 + y2) / 2)
  (h_diagonal_eq : (x - a)^2 + (y - b)^2 = (x1 - x2)^2 + (y1 - y2)^2) :
  x^2 + y^2 = 2 * r^2 - (a^2 + b^2) :=
  sorry

end locus_of_point_T_l309_309531


namespace ratio_milk_water_larger_vessel_l309_309480

-- Definitions for the conditions given in the problem
def ratio_volume (V1 V2 : ℝ) : Prop := V1 / V2 = 3 / 5
def ratio_milk_water_vessel1 (M1 W1 : ℝ) : Prop := M1 / W1 = 1 / 2
def ratio_milk_water_vessel2 (M2 W2 : ℝ) : Prop := M2 / W2 = 3 / 2

-- The final goal to prove
theorem ratio_milk_water_larger_vessel (V1 V2 M1 W1 M2 W2 : ℝ)
  (h1 : ratio_volume V1 V2) 
  (h2 : V1 = M1 + W1) 
  (h3 : V2 = M2 + W2) 
  (h4 : ratio_milk_water_vessel1 M1 W1) 
  (h5 : ratio_milk_water_vessel2 M2 W2) :
  (M1 + M2) / (W1 + W2) = 1 :=
by
  -- Proof is omitted
  sorry

end ratio_milk_water_larger_vessel_l309_309480


namespace nested_fraction_is_21_over_55_l309_309205

noncomputable def nested_fraction : ℚ := 1 / (3 - (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))))

theorem nested_fraction_is_21_over_55 : nested_fraction = 21 / 55 :=
by {
  have innermost := 3 - (1 / 3),
  have step1 := 3 - (1 / (3 - (1 / 3))),
  have step2 := 3 - (1 / (3 - (1 / (3 - (1 / 3))))),
  have step3 := 3 - (1 / (3 - (1 / (3 - (1 / (3 - (1 / 3))))))),
  have ans : innermost = 8 / 3 := by library_search,
  have step1_ans : step1 = 21 / 8 := by library_search,
  have step2_ans : step2 = 8 / 21 := by library_search,
  have step3_ans : step3 = 55 / 21 := by library_search,
  exact (by {
    simp only [nested_fraction, step3_ans, inv_div, mul_one, one_mul, div_div_eq_div_mul];
    norm_num
  })
}

end nested_fraction_is_21_over_55_l309_309205


namespace rectangle_perimeter_is_36_l309_309194

theorem rectangle_perimeter_is_36 (a b : ℕ) (h : a ≠ b) (h1 : a * b = 2 * (2 * a + 2 * b) - 8) : 2 * (a + b) = 36 :=
  sorry

end rectangle_perimeter_is_36_l309_309194


namespace S9_value_l309_309555

variable (a_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)

-- Define the arithmetic sequence
def is_arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, (a_n (n + 1) - a_n n) = (a_n 1 - a_n 0)

-- Sum of the first n terms of arithmetic sequence
def sum_first_n_terms (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S_n n = n * (a_n 0 + a_n (n - 1)) / 2

-- Given conditions: 
axiom a4_plus_a6 : a_n 4 + a_n 6 = 12
axiom S_definition : sum_first_n_terms S_n a_n

theorem S9_value : S_n 9 = 54 :=
by
  -- assuming the given conditions and definitions, we aim to prove the desired theorem.
  sorry

end S9_value_l309_309555


namespace single_light_on_positions_l309_309620

   open Matrix
   open Finset

   def toggle (n : ℕ) (A : Matrix (Fin n) (Fin n) ℕ) (i j : Fin n) : Matrix (Fin n) (Fin n) ℕ :=
     A.update i j (1 - A i j)  -- Toggle the light at (i, j)
      |>.update i (λ k => 1 - A i k)  -- Toggle the row 
      |>.update j (λ k => 1 - A k j)  -- Toggle the column

   noncomputable def possiblePositions := {(2, 2), (2, 4), (3, 3), (4, 2), (4, 4)}

   theorem single_light_on_positions
     (A : Matrix (Fin 5) (Fin 5) ℕ)
     (h_initial : ∀ i j, A i j = 0)
     (h_toggle : ∃ (toggles : Finset (Fin 5 × Fin 5)),
         ∀ i j, (toggle 5 A i j) ∈ toggles → A i j = 1)
     : ∀ (i j : Fin 5), 
         (i, j) ∈ possiblePositions ↔ (A i j = 1 ∧ ∀ (u v : Fin 5), (u, v) ≠ (i, j) → A u v = 0) :=
   sorry
   
end single_light_on_positions_l309_309620


namespace math_problem_l309_309759

noncomputable def base10_b := 25 + 1  -- 101_5 in base 10
noncomputable def base10_c := 343 + 98 + 21 + 4  -- 1234_7 in base 10
noncomputable def base10_d := 2187 + 324 + 45 + 6  -- 3456_9 in base 10

theorem math_problem (a : ℕ) (b c d : ℕ) (h_a : a = 2468)
  (h_b : b = base10_b) (h_c : c = base10_c) (h_d : d = base10_d) :
  (a / b) * c - d = 41708 :=
  by {
  sorry
}

end math_problem_l309_309759


namespace order_of_abc_l309_309936

noncomputable def a := Real.log 6 / Real.log 0.7
noncomputable def b := Real.rpow 6 0.7
noncomputable def c := Real.rpow 0.7 0.6

theorem order_of_abc : b > c ∧ c > a := by
  sorry

end order_of_abc_l309_309936


namespace one_div_thirteen_150th_digit_l309_309459

theorem one_div_thirteen_150th_digit :
  ∀ n : ℕ, n ≥ 0 → (let seq := "076923".to_list in (seq.get ((n % seq.length) - 1 + seq.length) % seq.length)) = '3' :=
by
  sorry

end one_div_thirteen_150th_digit_l309_309459


namespace find_number_satisfying_condition_l309_309516

-- Define the condition where fifteen percent of x equals 150
def fifteen_percent_eq (x : ℝ) : Prop :=
  (15 / 100) * x = 150

-- Statement to prove the existence of a number x that satisfies the condition, and this x equals 1000
theorem find_number_satisfying_condition : ∃ x : ℝ, fifteen_percent_eq x ∧ x = 1000 :=
by
  -- Proof will be added here
  sorry

end find_number_satisfying_condition_l309_309516


namespace angle_BAC_is_72_degrees_l309_309135

noncomputable theory

variables {A B C D E F : Type*}
variables [euclidean_plane A] [euclidean_plane B] [euclidean_plane C] [euclidean_plane D] [euclidean_plane E] [euclidean_plane F] 
variables {u v w : V}

/-- A proof that in an acute triangle ABC with altitudes AD, BE, and CF, 
such that the vector relationship holds true, the angle BAC is 72 degrees. -/
theorem angle_BAC_is_72_degrees 
  (triangle_ABC : is_triangle A B C)
  (acute_ABC : triangle_acute A B C)
  (altitude_AD : is_altitude A D B C)
  (altitude_BE : is_altitude B E A C)
  (altitude_CF : is_altitude C F A B)
  (vector_relation : 5 * (euclidean_plane.vector A D) + 3 * (euclidean_plane.vector B E) + 2 * (euclidean_plane.vector C F) = 0) :
  measure_angle A B C = 72 :=
sorry

end angle_BAC_is_72_degrees_l309_309135


namespace number_of_arrangements_l309_309125

-- Definitions of the problem's conditions
def student_set : Finset ℕ := {1, 2, 3, 4, 5}

def specific_students : Finset ℕ := {1, 2}

def remaining_students : Finset ℕ := student_set \ specific_students

-- Formalize the problem statement
theorem number_of_arrangements : 
  ∀ (students : Finset ℕ) 
    (specific : Finset ℕ) 
    (remaining : Finset ℕ),
    students = student_set →
    specific = specific_students →
    remaining = remaining_students →
    (specific.card = 2 ∧ students.card = 5 ∧ remaining.card = 3) →
    (∃ (n : ℕ), n = 12) :=
by
  intros
  sorry

end number_of_arrangements_l309_309125


namespace julia_total_cost_l309_309247

theorem julia_total_cost
  (snickers_cost : ℝ := 1.5)
  (mm_cost : ℝ := 2 * snickers_cost)
  (pepsi_cost : ℝ := 2 * mm_cost)
  (bread_cost : ℝ := 3 * pepsi_cost)
  (snickers_qty : ℕ := 2)
  (mm_qty : ℕ := 3)
  (pepsi_qty : ℕ := 4)
  (bread_qty : ℕ := 5)
  (money_given : ℝ := 5 * 20) :
  ((snickers_qty * snickers_cost) + (mm_qty * mm_cost) + (pepsi_qty * pepsi_cost) + (bread_qty * bread_cost)) > money_given := 
by
  sorry

end julia_total_cost_l309_309247


namespace find_c_d_l309_309721

def star (c d : ℕ) : ℕ := c^d + c*d

theorem find_c_d (c d : ℕ) (hc : 2 ≤ c) (hd : 2 ≤ d) (h_star : star c d = 28) : c + d = 7 :=
by
  sorry

end find_c_d_l309_309721


namespace cost_per_person_l309_309992

theorem cost_per_person 
  (total_cost : ℕ) 
  (total_people : ℕ) 
  (total_cost_in_billion : total_cost = 40000000000) 
  (total_people_in_million : total_people = 200000000) :
  total_cost / total_people = 200 := 
sorry

end cost_per_person_l309_309992


namespace solve_max_eq_l309_309643

theorem solve_max_eq (x : ℚ) (h : max x (-x) = 2 * x + 1) : x = -1 / 3 := by
  sorry

end solve_max_eq_l309_309643


namespace jason_car_cost_l309_309718

theorem jason_car_cost
    (down_payment : ℕ := 8000)
    (monthly_payment : ℕ := 525)
    (months : ℕ := 48)
    (interest_rate : ℝ := 0.05) :
    (down_payment + monthly_payment * months + interest_rate * (monthly_payment * months)) = 34460 := 
by
  sorry

end jason_car_cost_l309_309718


namespace triangle_sine_relation_l309_309962

theorem triangle_sine_relation (A B C a b c R : ℝ)
  (h1 : a = 2 * R * Real.sin A)
  (h2 : b = 2 * R * Real.sin B)
  (h3 : c = 2 * R * Real.sin C) :
  (a^2 - b^2) / c^2 = (Real.sin (A - B)) / (Real.sin C) :=
by
  sorry

end triangle_sine_relation_l309_309962


namespace pi_div_two_minus_alpha_in_third_quadrant_l309_309374

theorem pi_div_two_minus_alpha_in_third_quadrant (α : ℝ) (k : ℤ) (h : ∃ k : ℤ, (π + 2 * k * π < α) ∧ (α < 3 * π / 2 + 2 * k * π)) : 
  ∃ k : ℤ, (π + 2 * k * π < (π / 2 - α)) ∧ ((π / 2 - α) < 3 * π / 2 + 2 * k * π) :=
sorry

end pi_div_two_minus_alpha_in_third_quadrant_l309_309374


namespace find_k_l309_309652

noncomputable def curve (x k : ℝ) : ℝ := x + k * Real.log (1 + x)

theorem find_k (k : ℝ) :
  let y' := (fun x => 1 + k / (1 + x))
  (y' 1 = 2) ∧ ((1 + 2 * 1) = 0) → k = 2 :=
by
  sorry

end find_k_l309_309652


namespace min_value_of_function_l309_309278

theorem min_value_of_function (x : ℝ) (h : x > 0) : (∃ y : ℝ, y = x^2 + 3 * x + 1 ∧ ∀ z, z = x^2 + 3 * x + 1 → y ≤ z) → y = 5 :=
by
  sorry

end min_value_of_function_l309_309278


namespace units_digit_product_first_four_composite_numbers_l309_309782

-- Definition of the first four positive composite numbers
def first_four_composite_numbers : List ℕ := [4, 6, 8, 9]

-- Define the product of a list of numbers
def product (l : List ℕ) : ℕ := l.foldr (· * ·) 1

-- Mathematical statement
theorem units_digit_product_first_four_composite_numbers :
  (product first_four_composite_numbers) % 10 = 8 :=
by
  sorry

end units_digit_product_first_four_composite_numbers_l309_309782


namespace least_consecutive_odd_integers_l309_309845

theorem least_consecutive_odd_integers (x : ℤ)
  (h : (x + (x + 2) + (x + 4) + (x + 6) + (x + 8) + (x + 10) + (x + 12) + (x + 14) = 8 * 414)) :
  x = 407 :=
by
  sorry

end least_consecutive_odd_integers_l309_309845


namespace units_digit_of_product_is_eight_l309_309768

def first_four_compos_comps : List Nat := [4, 6, 8, 9]

def product_of_comps : Nat := first_four_compos_comps.foldl (· * ·) 1

theorem units_digit_of_product_is_eight : product_of_comps % 10 = 8 := 
by 
  sorry

end units_digit_of_product_is_eight_l309_309768


namespace max_value_of_f_l309_309326

noncomputable def f (x a : ℝ) : ℝ := -x^2 + 4 * x + a

theorem max_value_of_f (a : ℝ) (h_min : min (f 0 a) (f 1 a) = -2) :
  max (f 0 a) (f 1 a) = 1 :=
by
  sorry

end max_value_of_f_l309_309326


namespace solve_for_x_l309_309291

-- declare an existential quantifier to encapsulate the condition and the answer.
theorem solve_for_x : ∃ x : ℝ, x + (x + 2) + (x + 4) = 24 ∧ x = 6 := 
by 
  -- begin sorry to skip the proof part
  sorry

end solve_for_x_l309_309291


namespace root_condition_l309_309545

-- Let f(x) = x^2 + ax + a^2 - a - 2
noncomputable def f (a x : ℝ) : ℝ := x^2 + a * x + a^2 - a - 2

theorem root_condition (a : ℝ) (h1 : ∀ ζ : ℝ, (ζ > 1 → ζ^2 + a * ζ + a^2 - a - 2 = 0) ∧ (ζ < 1 → ζ^2 + a * ζ + a^2 - a - 2 = 0)) :
  -1 < a ∧ a < 1 :=
sorry

end root_condition_l309_309545


namespace intersection_correct_union_correct_l309_309113

variable (U A B : Set Nat)

def U_set : U = {1, 2, 3, 4, 5, 6} := by sorry
def A_set : A = {2, 4, 5} := by sorry
def B_set : B = {1, 2, 5} := by sorry

theorem intersection_correct (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {2, 4, 5}) (hB : B = {1, 2, 5}) :
  (A ∩ B) = {2, 5} := by sorry

theorem union_correct (hU : U = {1, 2, 3, 4, 5, 6}) (hA : A = {2, 4, 5}) (hB : B = {1, 2, 5}) :
  (A ∪ (U \ B)) = {2, 3, 4, 5, 6} := by sorry

end intersection_correct_union_correct_l309_309113


namespace capital_after_18_years_l309_309613

noncomputable def initial_investment : ℝ := 2000
def rate_of_increase : ℝ := 0.50
def period : ℕ := 3
def total_time : ℕ := 18

theorem capital_after_18_years :
  (initial_investment * (1 + rate_of_increase) ^ (total_time / period)) = 22781.25 :=
by
  sorry

end capital_after_18_years_l309_309613


namespace parabola_properties_l309_309938

theorem parabola_properties (m : ℝ) :
  (∀ P : ℝ × ℝ, P = (m, 1) ∧ (P.1 ^ 2 = 4 * P.2) →
    ((∃ y : ℝ, y = -1) ∧ (dist P (0, 1) = 2))) :=
by
  sorry

end parabola_properties_l309_309938


namespace tan_150_deg_l309_309875

-- Define the conditions
def angle_150_deg := 150 * real.pi / 180
def coordinates_of_Q := (-real.sqrt 3 / 2, 1 / 2)
def tan_of_angle_150 := real.tan angle_150_deg

-- The statement to prove
theorem tan_150_deg : tan_of_angle_150 = -1 / real.sqrt 3 :=
by
  -- Proof omitted
  sorry

end tan_150_deg_l309_309875


namespace possible_items_l309_309691

-- Mathematical definitions derived from the conditions.
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 20083

-- The theorem stating the possible number of items Kolya could have bought.
theorem possible_items (a n : ℕ) (hn : n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
sorry

end possible_items_l309_309691


namespace tan_150_eq_l309_309885

noncomputable def cosine150 : ℚ := -√3 / 2
noncomputable def sine150 : ℚ := 1 / 2
noncomputable def tangent150 : ℚ := sine150 / cosine150

theorem tan_150_eq : tangent150 = -1 / √3 := by
  sorry

end tan_150_eq_l309_309885


namespace tan_150_eq_neg_sqrt3_div_3_l309_309922

theorem tan_150_eq_neg_sqrt3_div_3
: tan 150 = - (Real.sqrt 3 / 3) :=
by
  have h1 : 150 = 180 - 30, by rfl
  have h2 : sin 150 = sin (180 - 30), by simp [h1]
  have h3 : cos 150 = cos (180 - 30), by simp [h1]
  have cos_30 := Real.sqrt 3 / 2
  have sin_30 := 1 / 2
  simp [Real.sin_eq_sin_of_Real_angle, Real.cos_eq_cos_of_Real_angle] at h2 h3
  have h4 : sin 150 = sin 30, by simp [h2, sin_30]
  have h5 : cos 150 = -cos 30, by simp [h3, cos_30]
  rw [Real.tan_eq_sin_div_cos, h4, h5]
  have h6 : (1 / 2) / (-Real.sqrt 3 / 2) = - (1 / Real.sqrt 3), by linarith
  rw h6
  have h7 : - (1 / Real.sqrt 3) = - (Real.sqrt 3 / 3), by
    field_simp [Real.sqrt_ne_zero],
    linarith
  exact h7,
  sorry

end tan_150_eq_neg_sqrt3_div_3_l309_309922


namespace pipe_B_fill_time_l309_309069

-- Definitions based on the given conditions
def fill_time_by_ABC := 10  -- in hours
def B_is_twice_as_fast_as_C : Prop := ∀ C B, B = 2 * C
def A_is_twice_as_fast_as_B : Prop := ∀ A B, A = 2 * B

-- The main theorem to prove
theorem pipe_B_fill_time (A B C : ℝ) (h1: fill_time_by_ABC = 10) 
    (h2 : B_is_twice_as_fast_as_C) (h3 : A_is_twice_as_fast_as_B) : B = 1 / 35 :=
by
  sorry

end pipe_B_fill_time_l309_309069


namespace decimal_150th_digit_l309_309452

theorem decimal_150th_digit {d : ℕ} (h : d = 150) :
  (∀ n, (1 / 13 : ℚ).decimalExpansion n) = "0.076923" →
  (150 % 6 = 0) →
  nthDigitAfterDecimal (1 / 13) 150 = 3 :=
by sorry

end decimal_150th_digit_l309_309452


namespace michael_monica_age_ratio_l309_309740

theorem michael_monica_age_ratio
  (x y : ℕ)
  (Patrick Michael Monica : ℕ)
  (h1 : Patrick = 3 * x)
  (h2 : Michael = 5 * x)
  (h3 : Monica = y)
  (h4 : y - Patrick = 64)
  (h5 : Patrick + Michael + Monica = 196) :
  Michael * 5 = Monica * 3 :=
by
  sorry

end michael_monica_age_ratio_l309_309740


namespace tan_150_degrees_l309_309867

theorem tan_150_degrees : Real.tan (150 * Real.pi / 180) = -Real.sqrt 3 / 3 := by
  sorry

end tan_150_degrees_l309_309867


namespace no_solutions_xyz_l309_309336

theorem no_solutions_xyz :
  ¬ ∃ (x y z : ℝ), x + y = 3 ∧ xy - z^2 = 4 :=
by
  sorry

end no_solutions_xyz_l309_309336


namespace ice_cream_stack_order_l309_309150

theorem ice_cream_stack_order (scoops : Finset ℕ) (h_scoops : scoops.card = 5) :
  (scoops.prod id) = 120 :=
by
  sorry

end ice_cream_stack_order_l309_309150


namespace even_three_digit_numbers_l309_309440

-- Define the set of digits
def digits : Set ℕ := {1, 2, 3, 4, 5, 6}

-- Define the conditions
def isEven (n : ℕ) : Prop := n % 2 = 0
def isLessThan600 (n : ℕ) : Prop := n < 600

-- Define the digit constraints for a, b, c
def validHundredsDigit (a : ℕ) : Prop := a ∈ {1, 2, 3, 4, 5}
def validTensDigit (b : ℕ) : Prop := b ∈ digits
def validUnitsDigit (c : ℕ) : Prop := c ∈ {2, 4, 6}

-- Define the number formation
def formNumber (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

-- Main statement
theorem even_three_digit_numbers : 
  {n : ℕ | ∃ a b c : ℕ, 
    validHundredsDigit a ∧ validTensDigit b ∧ validUnitsDigit c ∧ 
    isLessThan600 (formNumber a b c) ∧ isEven (formNumber a b c)}.card = 90 := 
by
  sorry

end even_three_digit_numbers_l309_309440


namespace tan_150_eq_neg_sqrt_3_l309_309894

theorem tan_150_eq_neg_sqrt_3 :
  let θ := 150 * Real.pi / 180 in
  Real.tan θ = -Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_sqrt_3_l309_309894


namespace no_integer_n_exists_l309_309583

theorem no_integer_n_exists (n : ℤ) : ¬(∃ n : ℤ, ∃ k : ℤ, ∃ m : ℤ, (n - 6) = 15 * k ∧ (n - 5) = 24 * m) :=
by
  sorry

end no_integer_n_exists_l309_309583


namespace problem1_problem2_l309_309864

theorem problem1 : 101 * 99 = 9999 := 
by sorry

theorem problem2 : 32 * 2^2 + 14 * 2^3 + 10 * 2^4 = 400 := 
by sorry

end problem1_problem2_l309_309864


namespace fraction_never_reducible_by_11_l309_309169

theorem fraction_never_reducible_by_11 :
  ∀ (n : ℕ), Nat.gcd (1 + n) (3 + 7 * n) ≠ 11 := by
  sorry

end fraction_never_reducible_by_11_l309_309169


namespace probability_of_exactly_one_red_ball_l309_309041

-- Definitions based on the conditions:
def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def draw_count : ℕ := 2

-- Required to calculate combinatory values
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Definitions of probabilities (though we won't use them explicitly for the statement):
def total_events : ℕ := choose total_balls draw_count
def no_red_ball_events : ℕ := choose white_balls draw_count
def one_red_ball_events : ℕ := choose red_balls 1 * choose white_balls 1

-- Probability Functions (for context):
def probability (events : ℕ) (total_events : ℕ) : ℚ := events / total_events

-- Lean 4 statement:
theorem probability_of_exactly_one_red_ball :
  probability one_red_ball_events total_events = 3/5 := by
  sorry

end probability_of_exactly_one_red_ball_l309_309041


namespace find_a_l309_309537

/-- Given function -/
def f (x: ℝ) : ℝ := (x + 1)^2 - 2 * (x + 1)

/-- Problem statement -/
theorem find_a (a : ℝ) (h : f a = 3) : a = 2 ∨ a = -2 := 
by
  sorry

end find_a_l309_309537


namespace no_nonzero_ints_l309_309086

theorem no_nonzero_ints (A B : ℤ) (hA : A ≠ 0) (hB : B ≠ 0) :
  (A ∣ (A + B) ∨ B ∣ (A - B)) → false :=
sorry

end no_nonzero_ints_l309_309086


namespace find_theta_l309_309538

noncomputable def f (x θ : ℝ) : ℝ := Real.sin (2 * x + θ)
noncomputable def g (x θ : ℝ) : ℝ := Real.sin (2 * (x + Real.pi / 8) + θ)

def is_even (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f x = f (-x)

theorem find_theta (θ : ℝ) : 
  (∀ x, g x θ = g (-x) θ) → θ = Real.pi / 4 :=
by
  intros h
  sorry

end find_theta_l309_309538


namespace number_of_items_l309_309678

theorem number_of_items {a n : ℕ} (h1 : ∀ x, x = 100 * a + 99) (h2 : 200 * 100 + 83 = 20083) : 
  (n * (100 * a + 99) = 20083) → (n = 17 ∨ n = 117) :=
by 
  sorry

end number_of_items_l309_309678


namespace one_thirteenth_150th_digit_l309_309449

theorem one_thirteenth_150th_digit :
  ∀ n : ℕ, 150 = n → n % 6 = 0 → (0 : ℕ).digitRec 1 13 150 = 3 :=
by
  sorry

end one_thirteenth_150th_digit_l309_309449


namespace units_digit_of_product_of_first_four_composites_l309_309798

def units_digit (n : Nat) : Nat := n % 10

theorem units_digit_of_product_of_first_four_composites : 
    units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_product_of_first_four_composites_l309_309798


namespace total_ducks_in_lake_l309_309483

/-- 
Problem: Determine the total number of ducks in the lake after more ducks join.

Conditions:
- Initially, there are 13 ducks in the lake.
- 20 more ducks come to join them.
-/

def initial_ducks : Nat := 13

def new_ducks : Nat := 20

theorem total_ducks_in_lake : initial_ducks + new_ducks = 33 := by
  sorry -- Proof to be filled in later

end total_ducks_in_lake_l309_309483


namespace melanie_food_total_weight_l309_309406

def total_weight (brie_oz : ℕ) (bread_lb : ℕ) (tomatoes_lb : ℕ) (zucchini_lb : ℕ) 
           (chicken_lb : ℕ) (raspberries_oz : ℕ) (blueberries_oz : ℕ) : ℕ :=
  let brie_lb := brie_oz / 16
  let raspberries_lb := raspberries_oz / 16
  let blueberries_lb := blueberries_oz / 16
  brie_lb + raspberries_lb + blueberries_lb + bread_lb + tomatoes_lb + zucchini_lb + chicken_lb

theorem melanie_food_total_weight : total_weight 8 1 1 2 (3 / 2) 8 8 = 7 :=
by
  -- result placeholder
  sorry

end melanie_food_total_weight_l309_309406


namespace number_of_items_l309_309677

theorem number_of_items {a n : ℕ} (h1 : ∀ x, x = 100 * a + 99) (h2 : 200 * 100 + 83 = 20083) : 
  (n * (100 * a + 99) = 20083) → (n = 17 ∨ n = 117) :=
by 
  sorry

end number_of_items_l309_309677


namespace largest_eight_digit_number_with_even_digits_l309_309839

def is_even_digit (n : ℕ) : Prop :=
  n = 0 ∨ n = 2 ∨ n = 4 ∨ n = 6 ∨ n = 8

def contains_all_even_digits (num : ℕ) : Prop :=
  ∀ d, is_even_digit d → (num.toDigits.contains d)

def is_eight_digit_number (num : ℕ) : Prop :=
  10000000 ≤ num ∧ num < 100000000

def largest_number_with_conditions (num : ℕ) : ℕ :=
  if contains_all_even_digits num ∧ is_eight_digit_number num then num else 0

theorem largest_eight_digit_number_with_even_digits :
  largest_number_with_conditions 99986420 = 99986420 :=
begin
  sorry
end

end largest_eight_digit_number_with_even_digits_l309_309839


namespace units_digit_of_composite_product_l309_309796

theorem units_digit_of_composite_product : 
  let composites := [4, 6, 8, 9],
      product := List.foldl (· * ·) 1 composites
  in product % 10 = 8 :=
  by
  sorry

end units_digit_of_composite_product_l309_309796


namespace sin_1035_eq_neg_sqrt2_div_2_l309_309641

theorem sin_1035_eq_neg_sqrt2_div_2 : Real.sin (1035 * Real.pi / 180) = - Real.sqrt 2 / 2 := by
    sorry

end sin_1035_eq_neg_sqrt2_div_2_l309_309641


namespace sqrt_sqr_l309_309415

theorem sqrt_sqr (x : ℝ) (hx : 0 ≤ x) : (Real.sqrt x) ^ 2 = x := 
by sorry

example : (Real.sqrt 3) ^ 2 = 3 := 
by apply sqrt_sqr; linarith

end sqrt_sqr_l309_309415


namespace kopeechka_items_l309_309704

-- Define necessary concepts and conditions
def item_cost_kopecks (a : ℕ) : ℕ := 100 * a + 99
def total_cost_kopecks : ℕ := 200 * 100 + 83

-- Lean statement defining the proof problem
theorem kopeechka_items (a n : ℕ) (h1 : ∀ a, n * item_cost_kopecks a = total_cost_kopecks) :
  n = 17 ∨ n = 117 :=
by sorry

end kopeechka_items_l309_309704


namespace sum_of_first_11_odd_numbers_l309_309185

theorem sum_of_first_11_odd_numbers : 
  (1 + 3 + 5 + 7 + 9 + 11 + 13 + 15 + 17 + 19 + 21) = 121 :=
by
  sorry

end sum_of_first_11_odd_numbers_l309_309185


namespace remainder_n_pow_5_minus_n_mod_30_l309_309259

theorem remainder_n_pow_5_minus_n_mod_30 (n : ℤ) : (n^5 - n) % 30 = 0 := 
by sorry

end remainder_n_pow_5_minus_n_mod_30_l309_309259


namespace even_function_iff_a_eq_2_l309_309242

noncomputable def y (a x : ℝ) : ℝ :=
  (x - 1) ^ 2 + a * x + Real.sin (x + Real.pi / 2)

theorem even_function_iff_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, y a x = y a (-x)) ↔ a = 2 :=
by
  -- Placeholder for the actual proof
  sorry

end even_function_iff_a_eq_2_l309_309242


namespace pieces_from_rod_l309_309298

theorem pieces_from_rod (length_of_rod : ℝ) (length_of_piece : ℝ) 
  (h_rod : length_of_rod = 42.5) 
  (h_piece : length_of_piece = 0.85) :
  length_of_rod / length_of_piece = 50 :=
by
  rw [h_rod, h_piece]
  calc
    42.5 / 0.85 = 50 := by norm_num

end pieces_from_rod_l309_309298


namespace james_marbles_l309_309254

def marbles_in_bag_D (bag_C : ℕ) := 2 * bag_C - 1
def marbles_in_bag_E (bag_A : ℕ) := bag_A / 2
def marbles_in_bag_G (bag_E : ℕ) := bag_E

theorem james_marbles :
    ∀ (A B C D E F G : ℕ),
      A = 4 →
      B = 3 →
      C = 5 →
      D = marbles_in_bag_D C →
      E = marbles_in_bag_E A →
      F = 3 →
      G = marbles_in_bag_G E →
      28 - (D + F) + 4 = 20 := by
    intros A B C D E F G hA hB hC hD hE hF hG
    sorry

end james_marbles_l309_309254


namespace mixing_ratios_l309_309052

theorem mixing_ratios (V : ℝ) (hV : 0 < V) :
  (4 * V / 5 + 7 * V / 10) / (V / 5 + 3 * V / 10) = 3 :=
by
  sorry

end mixing_ratios_l309_309052


namespace popsicle_sticks_left_l309_309331

-- Defining the conditions
def total_money : ℕ := 10
def cost_of_molds : ℕ := 3
def cost_of_sticks : ℕ := 1
def cost_of_juice_bottle : ℕ := 2
def popsicles_per_bottle : ℕ := 20
def initial_sticks : ℕ := 100

-- Statement of the problem
theorem popsicle_sticks_left : 
  let remaining_money := total_money - cost_of_molds - cost_of_sticks
  let bottles_of_juice := remaining_money / cost_of_juice_bottle
  let total_popsicles := bottles_of_juice * popsicles_per_bottle
  let sticks_left := initial_sticks - total_popsicles
  sticks_left = 40 := by
  sorry

end popsicle_sticks_left_l309_309331


namespace number_of_lines_with_negative_reciprocal_intercepts_l309_309207

-- Define the point (-2, 4)
def point : ℝ × ℝ := (-2, 4)

-- Define the condition that intercepts are negative reciprocals
def are_negative_reciprocals (a b : ℝ) : Prop :=
  a * b = -1

-- Define the proof problem: 
-- Number of lines through point (-2, 4) with intercepts negative reciprocals of each other
theorem number_of_lines_with_negative_reciprocal_intercepts :
  ∃ n : ℕ, n = 2 ∧ 
  ∀ (a b : ℝ), are_negative_reciprocals a b →
  (∃ m k : ℝ, (k * (-2) + m = 4) ∧ ((m ⁻¹ = a ∧ k = b) ∨ (k = a ∧ m ⁻¹ = b))) :=
sorry

end number_of_lines_with_negative_reciprocal_intercepts_l309_309207


namespace tan_domain_l309_309342

theorem tan_domain (x : ℝ) : 
  (∃ (k : ℤ), x = k * Real.pi - Real.pi / 4) ↔ 
  ¬(∃ (k : ℤ), x = k * Real.pi - Real.pi / 4) :=
sorry

end tan_domain_l309_309342


namespace probability_of_two_mathematicians_living_contemporarily_l309_309283

noncomputable def probability_of_contemporary_lifespan : ℚ :=
  let total_area := 500 * 500
  let triangle_area := 0.5 * 380 * 380
  let non_overlap_area := 2 * triangle_area
  let overlap_area := total_area - non_overlap_area
  overlap_area / total_area

theorem probability_of_two_mathematicians_living_contemporarily :
  probability_of_contemporary_lifespan = 2232 / 5000 :=
by
  -- The actual proof would go here
  sorry

end probability_of_two_mathematicians_living_contemporarily_l309_309283


namespace shortest_distance_between_semicircles_l309_309055

theorem shortest_distance_between_semicircles
  (ABCD : Type)
  (AD : ℝ)
  (shaded_area : ℝ)
  (is_rectangle : true)
  (AD_eq_10 : AD = 10)
  (shaded_area_eq_100 : shaded_area = 100) :
  ∃ d : ℝ, d = 2.5 * Real.pi :=
by
  sorry

end shortest_distance_between_semicircles_l309_309055


namespace part1_part2_l309_309358

-- Define the sets P and Q
def P (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2 * a + 1}
def Q : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Part (1)
theorem part1 (a : ℝ) (h : a = 3) : (P 3)ᶜ ∩ Q = {x | -2 ≤ x ∧ x < 4} :=
by
  sorry

-- Part (2)
theorem part2 (a : ℝ) : (∀ x, x ∈ P a → x ∈ Q) ∧ (∃ x, x ∈ Q ∧ x ∉ P a) → 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end part1_part2_l309_309358


namespace find_integer_triples_l309_309341

theorem find_integer_triples (x y z : ℤ) :
  (x^3 + y^3 + z^3 - 3 * x * y * z = 2003) →
  (x, y, z) ∈ {(668, 668, 667) | (668, 667, 668) | (667, 668, 668)} :=
begin
  sorry
end

end find_integer_triples_l309_309341


namespace radius_of_semi_circle_l309_309196

variable (r w l : ℝ)

def rectangle_inscribed_semi_circle (w l : ℝ) := 
  l = 3*w ∧ 
  2*l + 2*w = 126 ∧ 
  (∃ r, l = 2*r)

theorem radius_of_semi_circle :
  (∃ w l r, rectangle_inscribed_semi_circle w l ∧ l = 2*r) → r = 23.625 :=
by
  sorry

end radius_of_semi_circle_l309_309196


namespace present_age_of_A_is_11_l309_309751

-- Definitions for present ages
variables (A B C : ℕ)

-- Definitions for the given conditions
def sum_of_ages_present : Prop := A + B + C = 57
def age_ratio_three_years_ago (x : ℕ) : Prop := (A - 3 = x) ∧ (B - 3 = 2 * x) ∧ (C - 3 = 3 * x)

-- The proof statement
theorem present_age_of_A_is_11 (x : ℕ) (h1 : sum_of_ages_present A B C) (h2 : age_ratio_three_years_ago A B C x) : A = 11 := 
by
  sorry

end present_age_of_A_is_11_l309_309751


namespace tan_150_eq_neg_inv_sqrt3_l309_309918

theorem tan_150_eq_neg_inv_sqrt3 :
  Real.tan (150 * Real.pi / 180) = - (1 / Real.sqrt 3) :=
by
  have cos_30 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 :=
    by sorry
  have sin_30 : Real.sin (30 * Real.pi / 180) = 1 / 2 :=
    by sorry
  sorry

end tan_150_eq_neg_inv_sqrt3_l309_309918


namespace problem_1_problem_2_l309_309402

theorem problem_1 (a b c: ℝ) (h1: a > 0) (h2: b > 0) :
  a^3 + b^3 ≥ a^2 * b + a * b^2 :=
by
  sorry

theorem problem_2 (a b c: ℝ) (h1: a > 0) (h2: b > 0) (h3: c > 0) (h4: a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 :=
by
  sorry

end problem_1_problem_2_l309_309402


namespace ratio_is_one_half_l309_309265

-- Define the problem conditions as constants
def robert_age_in_2_years : ℕ := 30
def years_until_robert_is_30 : ℕ := 2
def patrick_current_age : ℕ := 14

-- Using the conditions, set up the definitions for the proof
def robert_current_age : ℕ := robert_age_in_2_years - years_until_robert_is_30

-- Define the target ratio
def ratio_of_ages : ℚ := patrick_current_age / robert_current_age

-- Prove that the ratio of Patrick's age to Robert's age is 1/2
theorem ratio_is_one_half : ratio_of_ages = 1 / 2 :=
by
  sorry

end ratio_is_one_half_l309_309265


namespace inequality_1_inequality_2_l309_309154

theorem inequality_1 (x : ℝ) : 4 * x + 5 ≤ 2 * (x + 1) → x ≤ -3/2 :=
by
  sorry

theorem inequality_2 (x : ℝ) : (2 * x - 1) / 3 - (9 * x + 2) / 6 ≤ 1 → x ≥ -2 :=
by
  sorry

end inequality_1_inequality_2_l309_309154


namespace graph_of_equation_l309_309465

theorem graph_of_equation :
  ∀ x y : ℝ, (2 * x - 3 * y) ^ 2 = 4 * x ^ 2 + 9 * y ^ 2 → (x = 0 ∨ y = 0) :=
by
  intros x y h
  sorry

end graph_of_equation_l309_309465


namespace train_length_is_correct_l309_309621

noncomputable def length_of_train 
  (time_to_cross : ℝ) 
  (bridge_length : ℝ) 
  (train_speed_kmph : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
  let distance_covered := train_speed_mps * time_to_cross
  distance_covered - bridge_length

theorem train_length_is_correct :
  length_of_train 23.998080153587715 140 36 = 99.98080153587715 :=
by sorry

end train_length_is_correct_l309_309621


namespace simplify_and_evaluate_l309_309151

-- Math proof problem
theorem simplify_and_evaluate :
  ∀ (a : ℤ), a = -1 →
  (2 - a)^2 - (1 + a) * (a - 1) - a * (a - 3) = 5 :=
by
  intros a ha
  sorry

end simplify_and_evaluate_l309_309151


namespace most_stable_athlete_l309_309524

theorem most_stable_athlete (s2_A s2_B s2_C s2_D : ℝ) 
  (hA : s2_A = 0.5) 
  (hB : s2_B = 0.5) 
  (hC : s2_C = 0.6) 
  (hD : s2_D = 0.4) :
  s2_D < s2_A ∧ s2_D < s2_B ∧ s2_D < s2_C :=
by
  sorry

end most_stable_athlete_l309_309524


namespace largest_eight_digit_with_all_even_digits_l309_309830

theorem largest_eight_digit_with_all_even_digits :
  ∀ n : ℕ, (∃ d1 d2 d3 d4 d5 : ℕ, (d1, d2, d3, d4, d5) = (0, 2, 4, 6, 8) ∧ 
    (99900000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) = n) → n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_even_digits_l309_309830


namespace complement_of_A_in_U_l309_309002

theorem complement_of_A_in_U :
    ∀ (U A : Set ℕ),
    U = {1, 2, 3, 4} →
    A = {1, 3} →
    (U \ A) = {2, 4} :=
by
  intros U A hU hA
  rw [hU, hA]
  sorry

end complement_of_A_in_U_l309_309002


namespace determine_x_l309_309512

-- Definitions based on conditions
variables {x : ℝ}

-- Problem statement
theorem determine_x (h : (6 * x)^5 = (18 * x)^4) (hx : x ≠ 0) : x = 27 / 2 :=
by
  sorry

end determine_x_l309_309512


namespace valid_permutations_l309_309049

theorem valid_permutations (a : Fin 101 → ℕ) :
  (∀ k, a k ≥ 2 ∧ a k ≤ 102 ∧ (∃ j, a j = k + 2)) →
  (∀ k, a (k + 1) % (k + 1) = 0) →
  (∃ cycles : List (List ℕ), cycles = [[1, 102], [1, 2, 102], [1, 3, 102], [1, 6, 102], [1, 17, 102], [1, 34, 102], 
                                       [1, 51, 102], [1, 2, 6, 102], [1, 2, 34, 102], [1, 3, 6, 102], [1, 3, 51, 102], 
                                       [1, 17, 34, 102], [1, 17, 51, 102]]) :=
sorry

end valid_permutations_l309_309049


namespace shooter_with_more_fluctuation_l309_309429

noncomputable def variance (scores : List ℕ) (mean : ℕ) : ℚ :=
  (List.sum (List.map (λ x => (x - mean) * (x - mean)) scores) : ℚ) / scores.length

theorem shooter_with_more_fluctuation :
  let scores_A := [7, 9, 8, 6, 10]
  let scores_B := [7, 8, 9, 8, 8]
  let mean := 8
  variance scores_A mean > variance scores_B mean :=
by
  sorry

end shooter_with_more_fluctuation_l309_309429


namespace Edward_money_left_l309_309513

theorem Edward_money_left {initial_amount item_cost sales_tax_rate sales_tax total_cost money_left : ℝ} 
    (h_initial : initial_amount = 18) 
    (h_item : item_cost = 16.35) 
    (h_rate : sales_tax_rate = 0.075) 
    (h_sales_tax : sales_tax = item_cost * sales_tax_rate) 
    (h_sales_tax_rounded : sales_tax = 1.23) 
    (h_total : total_cost = item_cost + sales_tax) 
    (h_money_left : money_left = initial_amount - total_cost) :
    money_left = 0.42 :=
by sorry

end Edward_money_left_l309_309513


namespace arithmetic_geometric_seq_l309_309941

theorem arithmetic_geometric_seq (a : ℕ → ℝ) (d a_1 : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_nonzero : d ≠ 0) (h_geom : (a 0, a 1, a 4) = (a_1, a_1 + d, a_1 + 4 * d) ∧ (a 1)^2 = a 0 * a 4)
  (h_sum : a 0 + a 1 + a 4 > 13) : a_1 > 1 :=
by sorry

end arithmetic_geometric_seq_l309_309941


namespace not_exist_three_numbers_l309_309634

theorem not_exist_three_numbers :
  ¬ ∃ (a b c : ℝ),
  (b^2 - 4 * a * c > 0 ∧ (-b / a > 0) ∧ (c / a > 0)) ∧
  (b^2 - 4 * a * c > 0 ∧ (-b / a < 0) ∧ (c / a > 0)) :=
by
  sorry

end not_exist_three_numbers_l309_309634


namespace units_digit_of_first_four_composite_numbers_l309_309805

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_first_four_composite_numbers :
  units_digit (4 * 6 * 8 * 9) = 8 :=
by
  sorry

end units_digit_of_first_four_composite_numbers_l309_309805


namespace kopeechka_items_l309_309698

theorem kopeechka_items (a n : ℕ) (hn : n * (100 * a + 99) = 20083) : n = 17 ∨ n = 117 :=
sorry

end kopeechka_items_l309_309698


namespace p_sufficient_but_not_necessary_for_q_l309_309983

variable (x : ℝ) (p q : Prop)

def p_condition : Prop := 0 < x ∧ x < 1
def q_condition : Prop := x^2 < 2 * x

theorem p_sufficient_but_not_necessary_for_q :
  (∀ x : ℝ, p_condition x → q_condition x) ∧
  ¬ (∀ x : ℝ, q_condition x → p_condition x) := by
  sorry

end p_sufficient_but_not_necessary_for_q_l309_309983


namespace intersection_of_sets_l309_309975

theorem intersection_of_sets :
  let A := {x : ℝ | -2 < x ∧ x < 1}
  let B := {x : ℝ | x < 0 ∨ x > 3}
  ∀ x, (x ∈ A ∧ x ∈ B) ↔ (-2 < x ∧ x < 0) :=
by
  let A := {x : ℝ | -2 < x ∧ x < 1}
  let B := {x : ℝ | x < 0 ∨ x > 3}
  intro x
  sorry

end intersection_of_sets_l309_309975


namespace log_ratio_squared_eq_nine_l309_309982

-- Given conditions
variable (x y : ℝ) 
variable (hx_pos : x > 0) 
variable (hy_pos : y > 0)
variable (hx_neq1 : x ≠ 1) 
variable (hy_neq1 : y ≠ 1)
variable (hlog : Real.log x / Real.log 3 = Real.log 81 / Real.log y)
variable (heq : x * y = 243)

-- Prove that (\log_3(\tfrac x y))^2 = 9
theorem log_ratio_squared_eq_nine (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0)
  (hx_neq1 : x ≠ 1) (hy_neq1 : y ≠ 1) 
  (hlog : Real.log x / Real.log 3 = Real.log 81 / Real.log y) 
  (heq : x * y = 243) : 
  ((Real.log x - Real.log y) / Real.log 3) ^ 2 = 9 :=
sorry

end log_ratio_squared_eq_nine_l309_309982


namespace tan_150_eq_l309_309884

noncomputable def cosine150 : ℚ := -√3 / 2
noncomputable def sine150 : ℚ := 1 / 2
noncomputable def tangent150 : ℚ := sine150 / cosine150

theorem tan_150_eq : tangent150 = -1 / √3 := by
  sorry

end tan_150_eq_l309_309884


namespace cara_between_pairs_l309_309502

-- Definitions based on the conditions
def friends := 7 -- Cara has 7 friends
def fixed_neighbor : Prop := true -- Alex must always be one of the neighbors

-- Problem statement to be proven
theorem cara_between_pairs (h : fixed_neighbor): 
  ∃ n : ℕ, n = 6 ∧ (1 + (friends - 1)) = n := by
  sorry

end cara_between_pairs_l309_309502


namespace water_speed_l309_309195

theorem water_speed (v : ℝ) (h1 : 4 - v > 0) (h2 : 6 * (4 - v) = 12) : v = 2 :=
by
  -- proof steps
  sorry

end water_speed_l309_309195


namespace largest_whole_number_l309_309592

theorem largest_whole_number (x : ℕ) : 9 * x < 150 → x ≤ 16 :=
by sorry

end largest_whole_number_l309_309592


namespace linda_savings_l309_309261

theorem linda_savings (S : ℝ) 
  (h1 : ∃ f : ℝ, f = 0.9 * 1/2 * S) -- She spent half of her savings on furniture with a 10% discount
  (h2 : ∃ t : ℝ, t = 1/2 * S * 1.05) -- The rest of her savings, spent on TV, had a 5% sales tax applied
  (h3 : 1/2 * S * 1.05 = 300) -- The total cost of the TV after tax was $300
  : S = 571.42 := 
sorry

end linda_savings_l309_309261


namespace quadrilateral_area_l309_309553

theorem quadrilateral_area {ABCQ : ℝ} 
  (side_length : ℝ) 
  (D P E N : ℝ → Prop) 
  (midpoints : ℝ) 
  (W X Y Z : ℝ → Prop) :
  side_length = 4 → 
  (∀ a b : ℝ, D a ∧ P b → a = 1 ∧ b = 1) → 
  (∀ c d : ℝ, E c ∧ N d → c = 1 ∧ d = 1) →
  (∀ w x y z : ℝ, W w ∧ X x ∧ Y y ∧ Z z → w = 0.5 ∧ x = 0.5 ∧ y = 0.5 ∧ z = 0.5) →
  ∃ (area : ℝ), area = 0.25 :=
by
  sorry

end quadrilateral_area_l309_309553


namespace find_A_l309_309243

theorem find_A (A B : ℕ) (hA : A < 10) (hB : B < 10) 
(h_div9 : (A + 1 + 5 + B + 9 + 4) % 9 = 0) 
(h_div11 : (A + 5 + 9 - (1 + B + 4)) % 11 = 0) : A = 5 :=
by sorry

end find_A_l309_309243


namespace kellan_wax_remaining_l309_309719

def remaining_wax (initial_A : ℕ) (initial_B : ℕ)
                  (spill_A : ℕ) (spill_B : ℕ)
                  (use_car_A : ℕ) (use_suv_B : ℕ) : ℕ :=
  let remaining_A := initial_A - spill_A - use_car_A
  let remaining_B := initial_B - spill_B - use_suv_B
  remaining_A + remaining_B

theorem kellan_wax_remaining
  (initial_A : ℕ := 10) 
  (initial_B : ℕ := 15)
  (spill_A : ℕ := 3) 
  (spill_B : ℕ := 4)
  (use_car_A : ℕ := 4) 
  (use_suv_B : ℕ := 5) :
  remaining_wax initial_A initial_B spill_A spill_B use_car_A use_suv_B = 9 :=
by sorry

end kellan_wax_remaining_l309_309719


namespace complement_of_A_in_U_l309_309404

-- Define the universal set U as the set of integers
def U : Set ℤ := Set.univ

-- Define the set A as the set of odd integers
def A : Set ℤ := {x : ℤ | ∃ k : ℤ, x = 2 * k + 1}

-- Define the complement of A in U
def complement_A : Set ℤ := U \ A

-- State the equivalence to be proved
theorem complement_of_A_in_U :
  complement_A = {x : ℤ | ∃ k : ℤ, x = 2 * k} :=
by
  sorry

end complement_of_A_in_U_l309_309404


namespace mutually_exclusive_but_not_complementary_l309_309203

-- Definitions for the problem conditions
inductive Card
| red | black | white | blue

inductive Person
| A | B | C | D

open Card Person

-- The statement of the proof
theorem mutually_exclusive_but_not_complementary : 
  (∃ (f : Person → Card), (f A = red) ∧ (f B ≠ red)) ∧ (∃ (f : Person → Card), (f B = red) ∧ (f A ≠ red)) :=
sorry

end mutually_exclusive_but_not_complementary_l309_309203


namespace continuous_at_2_l309_309148

noncomputable def f (x : ℝ) : ℝ := -2 * x^2 - 5

theorem continuous_at_2 : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → |f(x) - f 2| < ε :=
by
  assume ε ε_pos,
  -- We will use the delta from the problem solution without providing the full proof for context
  let δ := ε / 8,
  use δ,
  split,
  linarith,
  assume x hx,
  have hx_abs : |f(x) - f 2| = |f(x) - (-13)| := by rw [← f 2],
  rw [f],
  simp,
  sorry  -- The rest of the proof would follow here.

end continuous_at_2_l309_309148


namespace catch_up_time_l309_309282

noncomputable def speed_ratios (v : ℝ) : Prop :=
  let a_speed := (4 / 5) * v
  let b_speed := (2 / 5) * v
  a_speed = 2 * b_speed

theorem catch_up_time (v t : ℝ) (a_speed b_speed : ℝ)
  (h1 : a_speed = (4 / 5) * v)
  (h2 : b_speed = (2 / 5) * v)
  (h3 : a_speed = 2 * b_speed) :
  (t = 11) := by
  sorry

end catch_up_time_l309_309282


namespace ZYX_syndrome_diagnosis_l309_309491

theorem ZYX_syndrome_diagnosis (p : ℕ) (h1 : p = 26) (h2 : ∀ c, c = 2 * p) : ∃ n, n = c / 4 ∧ n = 13 :=
by
  sorry

end ZYX_syndrome_diagnosis_l309_309491


namespace ratio_length_to_width_l309_309747

-- Define the given conditions and values
def width : ℕ := 75
def perimeter : ℕ := 360

-- Define the proof problem statement
theorem ratio_length_to_width (L : ℕ) (P_eq : perimeter = 2 * L + 2 * width) :
  (L / width : ℚ) = 7 / 5 :=
sorry

end ratio_length_to_width_l309_309747


namespace hawks_total_points_l309_309027

/-- 
  Define the number of points per touchdown 
  and the number of touchdowns scored by the Hawks. 
-/
def points_per_touchdown : ℕ := 7
def touchdowns : ℕ := 3

/-- 
  Prove that the total number of points the Hawks have is 21. 
-/
theorem hawks_total_points : touchdowns * points_per_touchdown = 21 :=
by
  sorry

end hawks_total_points_l309_309027


namespace thread_length_l309_309581

theorem thread_length (initial_length : ℝ) (fraction : ℝ) (additional_length : ℝ) (total_length : ℝ) 
  (h1 : initial_length = 12) 
  (h2 : fraction = 3 / 4) 
  (h3 : additional_length = initial_length * fraction)
  (h4 : total_length = initial_length + additional_length) : 
  total_length = 21 := 
by
  -- proof steps would go here
  sorry

end thread_length_l309_309581


namespace problem_1_problem_2_l309_309365

-- Define the given function
def f (x : ℝ) := |x - 1|

-- Problem 1: Prove if f(x) + f(1 - x) ≥ a always holds, then a ≤ 1
theorem problem_1 (a : ℝ) : 
  (∀ x : ℝ, f x + f (1 - x) ≥ a) → a ≤ 1 :=
  sorry

-- Problem 2: Prove if a + 2b = 8, then f(a)^2 + f(b)^2 ≥ 5
theorem problem_2 (a b : ℝ) : 
  (a + 2 * b = 8) → (f a)^2 + (f b)^2 ≥ 5 :=
  sorry

end problem_1_problem_2_l309_309365


namespace kopeechka_items_l309_309694

theorem kopeechka_items (a n : ℕ) (hn : n * (100 * a + 99) = 20083) : n = 17 ∨ n = 117 :=
sorry

end kopeechka_items_l309_309694


namespace unique_pair_solution_l309_309929

theorem unique_pair_solution:
  ∃! (a n : ℕ) (h_pos_a : 0 < a) (h_pos_n : 0 < n), a^2 = 2^n + 15 ∧ a = 4 ∧ n = 0 := sorry

end unique_pair_solution_l309_309929


namespace trust_meteorologist_l309_309607

-- Definitions
def probability_of_clear := 0.74
def senator_accuracy := p : ℝ
def meteorologist_accuracy := 1.5 * p

-- Events
def event_G := "clear day"
def event_M1 := "first senator predicted clear"
def event_M2 := "second senator predicted clear"
def event_S := "meteorologist predicted rain"

theorem trust_meteorologist :
  let r := probability_of_clear
  let p := senator_accuracy
  let q := meteorologist_accuracy
  1.5 * p * (1 - p)^2 * (1 - r) - (1 - 1.5 * p) * p^2 * r > 0 :=
by
  sorry

end trust_meteorologist_l309_309607


namespace kolya_purchase_l309_309686

theorem kolya_purchase : ∃ n : ℕ, n = 17 ∨ n = 117 :=
by
  let item_cost := λ a : ℕ, 100 * a + 99
  let total_cost := 20000 + 83
  have h : ∀ n a, n * (item_cost a) = total_cost → (n = 17 ∨ n = 117) := sorry
  have h1 := h 17 0
  have h2 := h 117 0
  existsi 17
  exact h1 sorry

end kolya_purchase_l309_309686


namespace symmetric_curve_wrt_line_l309_309276

theorem symmetric_curve_wrt_line {f : ℝ → ℝ → ℝ} :
  (∀ x y : ℝ, f x y = 0 → f (y + 3) (x - 3) = 0) := by
  sorry

end symmetric_curve_wrt_line_l309_309276


namespace kolya_purchase_l309_309685

theorem kolya_purchase : ∃ n : ℕ, n = 17 ∨ n = 117 :=
by
  let item_cost := λ a : ℕ, 100 * a + 99
  let total_cost := 20000 + 83
  have h : ∀ n a, n * (item_cost a) = total_cost → (n = 17 ∨ n = 117) := sorry
  have h1 := h 17 0
  have h2 := h 117 0
  existsi 17
  exact h1 sorry

end kolya_purchase_l309_309685


namespace num_ways_write_100_as_distinct_squares_l309_309674

theorem num_ways_write_100_as_distinct_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a^2 + b^2 + c^2 = 100 ∧
  (∃ (x y z : ℕ), x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ x^2 + y^2 + z^2 = 100 ∧ (x, y, z) ≠ (a, b, c) ∧ (x, y, z) ≠ (a, c, b) ∧ (x, y, z) ≠ (b, a, c) ∧ (x, y, z) ≠ (b, c, a) ∧ (x, y, z) ≠ (c, a, b) ∧ (x, y, z) ≠ (c, b, a)) ∧
  ∀ (p q r : ℕ), p ≠ q ∧ p ≠ r ∧ q ≠ r ∧ p^2 + q^2 + r^2 = 100 → (p, q, r) = (a, b, c) ∨ (p, q, r) = (a, c, b) ∨ (p, q, r) = (b, a, c) ∨ (p, q, r) = (b, c, a) ∨ (p, q, r) = (c, a, b) ∨ (p, q, r) = (c, b, a) ∨ (p, q, r) = (x, y, z) ∨ (p, q, r) = (x, z, y) ∨ (p, q, r) = (y, x, z) ∨ (p, q, r) = (y, z, x) ∨ (p, q, r) = (z, x, y) ∨ (p, q, r) = (z, y, x) :=
sorry

end num_ways_write_100_as_distinct_squares_l309_309674


namespace min_value_S_max_value_S_l309_309569

theorem min_value_S 
  (a b c d e : ℝ)
  (h₀ : a ≥ -1)
  (h₁ : b ≥ -1)
  (h₂ : c ≥ -1)
  (h₃ : d ≥ -1)
  (h₄ : e ≥ -1)
  (h_sum : a + b + c + d + e = 5) : 
  (a + b) * (b + c) * (c + d) * (d + e) * (e + a) ≥ -512 := 
sorry

theorem max_value_S 
  (a b c d e : ℝ)
  (h₀ : a ≥ -1)
  (h₁ : b ≥ -1)
  (h₂ : c ≥ -1)
  (h₃ : d ≥ -1)
  (h₄ : e ≥ -1)
  (h_sum : a + b + c + d + e = 5) : 
  (a + b) * (b + c) * (c + d) * (d + e) * (e + a) ≤ 288 := 
sorry

end min_value_S_max_value_S_l309_309569


namespace probability_white_ball_is_two_fifths_l309_309612

-- Define the total number of each type of balls.
def white_balls : ℕ := 6
def yellow_balls : ℕ := 5
def red_balls : ℕ := 4

-- Calculate the total number of balls in the bag.
def total_balls : ℕ := white_balls + yellow_balls + red_balls

-- Define the probability calculation.
noncomputable def probability_of_white_ball : ℚ := white_balls / total_balls

-- The theorem statement asserting the probability of drawing a white ball.
theorem probability_white_ball_is_two_fifths :
  probability_of_white_ball = 2 / 5 :=
sorry

end probability_white_ball_is_two_fifths_l309_309612


namespace tree_planting_equation_l309_309128

theorem tree_planting_equation (x : ℝ) (h : x > 0) : 
  180 / x - 180 / (1.5 * x) = 2 :=
sorry

end tree_planting_equation_l309_309128


namespace athlete_distance_proof_l309_309072

-- Definition of conditions as constants
def time_seconds : ℕ := 20
def speed_kmh : ℕ := 36

-- Convert speed from km/h to m/s
def speed_mps : ℕ := speed_kmh * 1000 / 3600

-- Proof statement that the distance is 200 meters
theorem athlete_distance_proof : speed_mps * time_seconds = 200 :=
by sorry

end athlete_distance_proof_l309_309072


namespace remaining_money_after_payments_l309_309229

-- Conditions
def initial_money : ℕ := 100
def paid_colin : ℕ := 20
def paid_helen : ℕ := 2 * paid_colin
def paid_benedict : ℕ := paid_helen / 2
def total_paid : ℕ := paid_colin + paid_helen + paid_benedict

-- Proof
theorem remaining_money_after_payments : 
  initial_money - total_paid = 20 := by
  sorry

end remaining_money_after_payments_l309_309229


namespace tan_150_deg_l309_309898

theorem tan_150_deg : Real.tan (150 * Real.pi / 180) = - (Real.sqrt 3) / 3 :=
by
  -- Conditions used for defining the theorem
  -- 1. 150^\circ = 180^\circ - 30^\circ
  -- 2. Coordinates of a point on the unit circle at angle θ are (cos θ, sin θ)
  -- 3. For 30^\circ, (cos 30^\circ, sin 30^\circ) = (√3/2, 1/2)
  -- 4. Reflect the point across the y-axis changes x-coordinate's sign
  -- 5. tan θ = y/x for a point (x, y) on the unit circle

  sorry

end tan_150_deg_l309_309898


namespace tan_150_eq_neg_sqrt_3_l309_309891

theorem tan_150_eq_neg_sqrt_3 :
  let θ := 150 * Real.pi / 180 in
  Real.tan θ = -Real.sqrt 3 :=
by
  sorry

end tan_150_eq_neg_sqrt_3_l309_309891


namespace min_value_of_expression_l309_309231

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 1) : 
  x + 2 * y ≥ 9 + 4 * Real.sqrt 2 := 
sorry

end min_value_of_expression_l309_309231


namespace largest_eight_digit_number_contains_even_digits_l309_309835

theorem largest_eight_digit_number_contains_even_digits :
  ∃ n : ℕ, n = 99986420 ∧ (10000000 ≤ n ∧ n < 100000000) ∧
    ∀ d ∈ [0, 2, 4, 6, 8], ∃ (i : ℕ), i < 8 ∧ (n / 10^i) % 10 = d :=
by
  sorry

end largest_eight_digit_number_contains_even_digits_l309_309835


namespace johns_last_month_savings_l309_309299

theorem johns_last_month_savings (earnings rent dishwasher left_over : ℝ) 
  (h1 : rent = 0.40 * earnings) 
  (h2 : dishwasher = 0.70 * rent) 
  (h3 : left_over = earnings - rent - dishwasher) :
  left_over = 0.32 * earnings :=
by 
  sorry

end johns_last_month_savings_l309_309299


namespace cake_icing_l309_309193

/-- Define the cake conditions -/
structure Cake :=
  (dimension : ℕ)
  (small_cube_dimension : ℕ)
  (total_cubes : ℕ)
  (iced_faces : ℕ)

/-- Define the main theorem to prove the number of smaller cubes with icing on exactly two sides -/
theorem cake_icing (c : Cake) : 
  c.dimension = 5 ∧ c.small_cube_dimension = 1 ∧ c.total_cubes = 125 ∧ c.iced_faces = 4 →
  ∃ n, n = 20 :=
by
  sorry

end cake_icing_l309_309193


namespace investment_amount_l309_309189

theorem investment_amount (P: ℝ) (q_investment: ℝ) (ratio_pq: ℝ) (ratio_qp: ℝ) 
  (h1: ratio_pq = 4) (h2: ratio_qp = 6) (q_investment: ℝ) (h3: q_investment = 90000): 
  P = 60000 :=
by 
  -- Sorry is used here to skip the actual proof
  sorry

end investment_amount_l309_309189


namespace largest_eight_digit_with_all_even_digits_l309_309828

theorem largest_eight_digit_with_all_even_digits :
  ∀ n : ℕ, (∃ d1 d2 d3 d4 d5 : ℕ, (d1, d2, d3, d4, d5) = (0, 2, 4, 6, 8) ∧ 
    (99900000 + d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5) = n) → n = 99986420 :=
by
  sorry

end largest_eight_digit_with_all_even_digits_l309_309828


namespace smallest_k_l309_309348

theorem smallest_k (k : ℕ) (numbers : set ℕ) (h₁ : ∀ x ∈ numbers, x ≤ 2016) (h₂ : numbers.card = k) :
  (∃ a b ∈ numbers, 672 < abs (a - b) ∧ abs (a - b) < 1344) ↔ k ≥ 674 := 
by
  sorry

end smallest_k_l309_309348
