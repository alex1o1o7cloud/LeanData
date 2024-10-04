import Mathlib

namespace max_product_of_two_integers_whose_sum_is_300_l122_122496

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l122_122496


namespace negation_of_existential_l122_122280

theorem negation_of_existential:
  (¬ ∃ x_0 : ℝ, x_0^2 + 2 * x_0 + 2 = 0) ↔ ∀ x : ℝ, x^2 + 2 * x + 2 ≠ 0 :=
by
  sorry

end negation_of_existential_l122_122280


namespace range_of_k_for_obtuse_triangle_l122_122271

theorem range_of_k_for_obtuse_triangle (k : ℝ) (a b c : ℝ) (h₁ : a = k) (h₂ : b = k + 2) (h₃ : c = k + 4) : 
  2 < k ∧ k < 6 :=
by
  sorry

end range_of_k_for_obtuse_triangle_l122_122271


namespace max_product_300_l122_122540

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l122_122540


namespace volunteer_arrangements_l122_122957

theorem volunteer_arrangements : 
  ∃ (arrangements : Nat), 
    (∃ (A B C D E : Bool), -- Represent assignments as Bool: True for intersection A, False for intersection B
      (A || ¬A) && (B || ¬B) && (C || ¬C) && (D || ¬D) && (E || ¬E) && -- Each volunteer goes to one intersection
      (A + B + C + D + E ≠ 0) && (¬A + ¬B + ¬C + ¬D + ¬E ≠ 0)) && -- Each intersection has at least one volunteer
    arrangements = 30 := 
by
  sorry

end volunteer_arrangements_l122_122957


namespace sum_powers_div_5_iff_l122_122694

theorem sum_powers_div_5_iff (n : ℕ) (h : n > 0) : (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
sorry

end sum_powers_div_5_iff_l122_122694


namespace minimum_bamboo_fencing_length_l122_122954

theorem minimum_bamboo_fencing_length 
  (a b z : ℝ) 
  (h1 : a * b = 50)
  (h2 : a + 2 * b = z) : 
  z ≥ 20 := 
  sorry

end minimum_bamboo_fencing_length_l122_122954


namespace triangle_right_if_condition_l122_122798

variables (a b c : ℝ) (A B C : ℝ)
-- Condition: Given 1 + cos A = (b + c) / c
axiom h1 : 1 + Real.cos A = (b + c) / c 

-- To prove: a^2 + b^2 = c^2
theorem triangle_right_if_condition (h1 : 1 + Real.cos A = (b + c) / c) : a^2 + b^2 = c^2 :=
  sorry

end triangle_right_if_condition_l122_122798


namespace mode_is_necessary_characteristic_of_dataset_l122_122854

-- Define a dataset as a finite set of elements from any type.
variable {α : Type*} [Fintype α]

-- Define a mode for a dataset as an element that occurs most frequently.
def mode (dataset : Multiset α) : α :=
sorry  -- Mode definition and computation are omitted for this high-level example.

-- Define the theorem that mode is a necessary characteristic of a dataset.
theorem mode_is_necessary_characteristic_of_dataset (dataset : Multiset α) : 
  exists mode_elm : α, mode_elm = mode dataset :=
sorry

end mode_is_necessary_characteristic_of_dataset_l122_122854


namespace businessmen_neither_coffee_nor_tea_l122_122748

theorem businessmen_neither_coffee_nor_tea
  (total : ℕ)
  (C T : Finset ℕ)
  (hC : C.card = 15)
  (hT : T.card = 14)
  (hCT : (C ∩ T).card = 7)
  (htotal : total = 30) : 
  total - (C ∪ T).card = 8 := 
by
  sorry

end businessmen_neither_coffee_nor_tea_l122_122748


namespace blue_pens_count_l122_122600

variable (redPenCost bluePenCost totalCost totalPens : ℕ)
variable (numRedPens numBluePens : ℕ)

-- Conditions
axiom PriceOfRedPen : redPenCost = 5
axiom PriceOfBluePen : bluePenCost = 7
axiom TotalCost : totalCost = 102
axiom TotalPens : totalPens = 16
axiom PenCount : numRedPens + numBluePens = totalPens
axiom CostEquation : redPenCost * numRedPens + bluePenCost * numBluePens = totalCost

theorem blue_pens_count : numBluePens = 11 :=
by
  sorry

end blue_pens_count_l122_122600


namespace max_product_of_two_integers_whose_sum_is_300_l122_122501

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l122_122501


namespace probability_sqrt_lt_nine_two_digit_l122_122561

theorem probability_sqrt_lt_nine_two_digit :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99} in
  let T := {n : ℕ | 10 ≤ n ∧ n < 81} in
  (Fintype.card T : ℚ) / (Fintype.card S) = 71 / 90 :=
by
  sorry

end probability_sqrt_lt_nine_two_digit_l122_122561


namespace geometric_sequence_eighth_term_is_correct_l122_122670

noncomputable def geometric_sequence_eighth_term : ℚ :=
  let a1 := 2187
  let a5 := 960
  let r := (960 / 2187)^(1/4)
  let a8 := a1 * r^7
  a8

theorem geometric_sequence_eighth_term_is_correct :
  let a1 := 2187
  let a5 := 960
  let r := (960 / 2187)^(1/4)
  let a8 := a1 * r^7
  a8 = 35651584 / 4782969 := by
    sorry

end geometric_sequence_eighth_term_is_correct_l122_122670


namespace inequality_proof_l122_122682

variable (k : ℕ) (a b c : ℝ)
variables (hk : 0 < k) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

theorem inequality_proof (hk : k > 0) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * (1 - a^k) + b * (1 - (a + b)^k) + c * (1 - (a + b + c)^k) < k / (k + 1) :=
sorry

end inequality_proof_l122_122682


namespace max_product_of_sum_300_l122_122427

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l122_122427


namespace expectation_X_p_expectation_X_neg_p_l122_122949

theorem expectation_X_p (X : ℝ → ℝ≥0) (phi : ℝ → ℝ) (p : ℝ) 
  (hX_nonneg : ∀ x, 0 ≤ X x)
  (hphi : ∀ λ, λ ≥ 0 → phi λ = ∫ e^(-λ * X x) dμ)
  (h0p1 : 0 < p ∧ p < 1) :
  (E[X^p] = p / Γ(1 - p) * ∫ (1 - phi λ) / λ^(p+1) dλ)
:= sorry

theorem expectation_X_neg_p (X : ℝ → ℝ≥0) (phi : ℝ → ℝ) (p : ℝ) 
  (hX_nonneg : ∀ x, 0 ≤ X x)
  (hphi : ∀ λ, λ ≥ 0 → phi λ = ∫ e^(-λ * X x) dμ)
  (hp : p > 0) :
  (E[X^-p] = 1 / Γ(p) * ∫ phi λ * λ^(p-1) dλ)
:= sorry

end expectation_X_p_expectation_X_neg_p_l122_122949


namespace flavors_remaining_to_try_l122_122282

def total_flavors : ℕ := 100
def flavors_tried_two_years_ago (total_flavors : ℕ) : ℕ := total_flavors / 4
def flavors_tried_last_year (flavors_tried_two_years_ago : ℕ) : ℕ := 2 * flavors_tried_two_years_ago

theorem flavors_remaining_to_try
  (total_flavors : ℕ)
  (flavors_tried_two_years_ago : ℕ)
  (flavors_tried_last_year : ℕ) :
  flavors_tried_two_years_ago = total_flavors / 4 →
  flavors_tried_last_year = 2 * flavors_tried_two_years_ago →
  total_flavors - (flavors_tried_two_years_ago + flavors_tried_last_year) = 25 :=
by
  sorry

end flavors_remaining_to_try_l122_122282


namespace greatest_product_sum_300_l122_122362

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l122_122362


namespace crayons_lost_or_given_away_total_l122_122262

def initial_crayons_box1 := 479
def initial_crayons_box2 := 352
def initial_crayons_box3 := 621

def remaining_crayons_box1 := 134
def remaining_crayons_box2 := 221
def remaining_crayons_box3 := 487

def crayons_lost_or_given_away_box1 := initial_crayons_box1 - remaining_crayons_box1
def crayons_lost_or_given_away_box2 := initial_crayons_box2 - remaining_crayons_box2
def crayons_lost_or_given_away_box3 := initial_crayons_box3 - remaining_crayons_box3

def total_crayons_lost_or_given_away := crayons_lost_or_given_away_box1 + crayons_lost_or_given_away_box2 + crayons_lost_or_given_away_box3

theorem crayons_lost_or_given_away_total : total_crayons_lost_or_given_away = 610 :=
by
  -- Proof should go here
  sorry

end crayons_lost_or_given_away_total_l122_122262


namespace ratio_of_rectangles_l122_122905

noncomputable def rect_ratio (a b c d e f : ℝ) 
  (h1: a / c = 3 / 5) 
  (h2: b / d = 3 / 5) 
  (h3: a / e = 7 / 4) 
  (h4: b / f = 7 / 4) : ℝ :=
  let A_A := a * b
  let A_B := (a * 5 / 3) * (b * 5 / 3)
  let A_C := (a * 4 / 7) * (b * 4 / 7)
  let A_BC := A_B + A_C
  A_A / A_BC

theorem ratio_of_rectangles (a b c d e f : ℝ) 
  (h1: a / c = 3 / 5) 
  (h2: b / d = 3 / 5) 
  (h3: a / e = 7 / 4) 
  (h4: b / f = 7 / 4) : 
  rect_ratio a b c d e f h1 h2 h3 h4 = 441 / 1369 :=
by
  sorry

end ratio_of_rectangles_l122_122905


namespace total_legs_correct_l122_122666

variable (a b : ℕ)

def total_legs (a b : ℕ) : ℕ := 2 * a + 4 * b

theorem total_legs_correct (a b : ℕ) : total_legs a b = 2 * a + 4 * b :=
by sorry

end total_legs_correct_l122_122666


namespace normal_dist_probability_l122_122145

open MeasureTheory.ProbabilityTheory
open MeasureTheory

noncomputable def standard_normal_cdf (x : ℝ) : ℝ := sorry

theorem normal_dist_probability :
  (∀ (ξ : ℝ -> ℝ) (σ : ℝ), 
   (ξ ∼ Normal 1 σ^2) → 
   P(ξ < 2) = 0.6 →
   P(0 < ξ < 1) = 0.1) :=
begin
  sorry
end

end normal_dist_probability_l122_122145


namespace area_of_trapezium_l122_122724

-- Definitions
def length_parallel_side_1 : ℝ := 4
def length_parallel_side_2 : ℝ := 5
def perpendicular_distance : ℝ := 6

-- Statement
theorem area_of_trapezium :
  (1 / 2) * (length_parallel_side_1 + length_parallel_side_2) * perpendicular_distance = 27 :=
by
  sorry

end area_of_trapezium_l122_122724


namespace greatest_product_sum_300_l122_122414

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l122_122414


namespace multiple_of_10_and_12_within_100_l122_122074

theorem multiple_of_10_and_12_within_100 :
  ∀ (n : ℕ), n ≤ 100 → (∃ k₁ k₂ : ℕ, n = 10 * k₁ ∧ n = 12 * k₂) ↔ n = 60 :=
by
  sorry

end multiple_of_10_and_12_within_100_l122_122074


namespace find_p_q_l122_122751

theorem find_p_q (p q : ℝ) (f : ℝ → ℝ) 
  (h_f : ∀ x, f x = x^2 + p * x + q)
  (h_min : ∀ x, x = q → f x = (p + q)^2) : 
  (p = 0 ∧ q = 0) ∨ (p = -1 ∧ q = 1 / 2) :=
by
  sorry

end find_p_q_l122_122751


namespace product_of_y_coordinates_on_line_l122_122692

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem product_of_y_coordinates_on_line (y1 y2 : ℝ) (h1 : distance (4, -1) (-2, y1) = 8) (h2 : distance (4, -1) (-2, y2) = 8) :
  y1 * y2 = -27 :=
sorry

end product_of_y_coordinates_on_line_l122_122692


namespace intersection_M_N_l122_122686

def is_M (x : ℝ) : Prop := x^2 + x - 6 < 0
def is_N (x : ℝ) : Prop := abs (x - 1) <= 2

theorem intersection_M_N : {x : ℝ | is_M x} ∩ {x : ℝ | is_N x} = {x : ℝ | -1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_M_N_l122_122686


namespace paul_sandwiches_in_6_days_l122_122054

def sandwiches_eaten_in_n_days (n : ℕ) : ℕ :=
  let day1 := 2
  let day2 := 2 * day1
  let day3 := 2 * day2
  let three_day_total := day1 + day2 + day3
  three_day_total * (n / 3)

theorem paul_sandwiches_in_6_days : sandwiches_eaten_in_n_days 6 = 28 :=
by
  sorry

end paul_sandwiches_in_6_days_l122_122054


namespace area_of_triangle_ABC_l122_122746

variable (A : ℝ) -- Area of the triangle ABC
variable (S_heptagon : ℝ) -- Area of the heptagon ADECFGH
variable (S_overlap : ℝ) -- Overlapping area after folding

-- Given conditions
axiom ratio_condition : S_heptagon = (5 / 7) * A
axiom overlap_condition : S_overlap = 8

-- Proof statement
theorem area_of_triangle_ABC :
  A = 28 := by
  sorry

end area_of_triangle_ABC_l122_122746


namespace xy_identity_l122_122914

theorem xy_identity (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 := by
  sorry

end xy_identity_l122_122914


namespace arithmetic_sequence_30th_term_l122_122967

theorem arithmetic_sequence_30th_term :
  let a1 := 3
  let a2 := 13
  let a3 := 23
  let d := a2 - a1
  let n := 30
  let an := a1 + (n - 1) * d
  an = 293 :=
by
  sorry

end arithmetic_sequence_30th_term_l122_122967


namespace greatest_product_sum_300_l122_122374

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l122_122374


namespace sam_bought_nine_books_l122_122697

-- Definitions based on the conditions
def initial_money : ℕ := 79
def cost_per_book : ℕ := 7
def money_left : ℕ := 16

-- The amount spent on books
def money_spent_on_books : ℕ := initial_money - money_left

-- The number of books bought
def number_of_books (spent : ℕ) (cost : ℕ) : ℕ := spent / cost

-- Let x be the number of books bought and prove x = 9
theorem sam_bought_nine_books : number_of_books money_spent_on_books cost_per_book = 9 :=
by
  sorry

end sam_bought_nine_books_l122_122697


namespace greatest_product_l122_122524

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l122_122524


namespace cakes_left_correct_l122_122611

def number_of_cakes_left (total_cakes sold_cakes : ℕ) : ℕ :=
  total_cakes - sold_cakes

theorem cakes_left_correct :
  number_of_cakes_left 54 41 = 13 :=
by
  sorry

end cakes_left_correct_l122_122611


namespace range_of_s_triangle_l122_122663

theorem range_of_s_triangle (inequalities_form_triangle : Prop) : 
  (0 < s ∧ s ≤ 2) ∨ (s ≥ 4) ↔ inequalities_form_triangle := 
sorry

end range_of_s_triangle_l122_122663


namespace boys_in_school_l122_122166

theorem boys_in_school (B G1 G2 : ℕ) (h1 : G1 = 632) (h2 : G2 = G1 + 465) (h3 : G2 = B + 687) : B = 410 :=
by
  sorry

end boys_in_school_l122_122166


namespace hall_volume_proof_l122_122595

-- Define the given conditions.
def hall_length (l : ℝ) : Prop := l = 18
def hall_width (w : ℝ) : Prop := w = 9
def floor_ceiling_area_eq_wall_area (h l w : ℝ) : Prop := 
  2 * (l * w) = 2 * (l * h) + 2 * (w * h)

-- Define the volume calculation.
def hall_volume (l w h V : ℝ) : Prop := 
  V = l * w * h

-- The main theorem stating that the volume is 972 cubic meters.
theorem hall_volume_proof (l w h V : ℝ) 
  (length : hall_length l) 
  (width : hall_width w) 
  (fc_eq_wa : floor_ceiling_area_eq_wall_area h l w) 
  (volume : hall_volume l w h V) : 
  V = 972 :=
  sorry

end hall_volume_proof_l122_122595


namespace max_product_sum_300_l122_122486

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l122_122486


namespace max_product_of_sum_300_l122_122429

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l122_122429


namespace find_x_l122_122822

noncomputable def a : ℝ := Real.log 2 / Real.log 10
noncomputable def b : ℝ := 1 / a
noncomputable def log2_5 : ℝ := Real.log 5 / Real.log 2

theorem find_x (a₀ : a = 0.3010) : 
  ∃ x : ℝ, (log2_5 ^ 2 - a * log2_5 + x * b = 0) → 
  x = (log2_5 ^ 2 * 0.3010) :=
by
  sorry

end find_x_l122_122822


namespace max_product_of_sum_300_l122_122475

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l122_122475


namespace pretty_18_sum_div_18_l122_122252

def is_pretty_18 (n : ℕ) : Prop :=
  n % 18 = 0 ∧ ∃ d, d > 0 ∧ d = 18 ∧ ( nat.divisors n ).card = 18 

noncomputable def sum_pretty_18_below_1000 :=
  ∑ n in (finset.range 1000).filter is_pretty_18, n

theorem pretty_18_sum_div_18 : sum_pretty_18_below_1000 / 18 = 112 :=
by 
  sorry

end pretty_18_sum_div_18_l122_122252


namespace cups_per_larger_crust_l122_122171

theorem cups_per_larger_crust
  (initial_crusts : ℕ)
  (initial_flour : ℚ)
  (new_crusts : ℕ)
  (constant_flour : ℚ)
  (h1 : initial_crusts * (initial_flour / initial_crusts) = initial_flour )
  (h2 : new_crusts * (constant_flour / new_crusts) = constant_flour )
  (h3 : initial_flour = constant_flour)
  : (constant_flour / new_crusts) = (8 / 10) :=
by 
  sorry

end cups_per_larger_crust_l122_122171


namespace sample_capacity_n_l122_122603

theorem sample_capacity_n
  (n : ℕ) 
  (engineers technicians craftsmen : ℕ) 
  (total_population : ℕ)
  (stratified_interval systematic_interval : ℕ) :
  engineers = 6 →
  technicians = 12 →
  craftsmen = 18 →
  total_population = engineers + technicians + craftsmen →
  total_population = 36 →
  (∃ n : ℕ, n ∣ total_population ∧ 6 ∣ n ∧ 35 % (n + 1) = 0) →
  n = 6 :=
by
  sorry

end sample_capacity_n_l122_122603


namespace intercept_sum_l122_122598

theorem intercept_sum (x y : ℝ) :
  (y - 3 = 6 * (x - 5)) →
  (∃ x_intercept, (y = 0) ∧ (x_intercept = 4.5)) →
  (∃ y_intercept, (x = 0) ∧ (y_intercept = -27)) →
  (4.5 + (-27) = -22.5) :=
by
  intros h_eq h_xint h_yint
  sorry

end intercept_sum_l122_122598


namespace value_of_a_l122_122158

theorem value_of_a (a : ℝ) : (a^2 - 4) / (a - 2) = 0 → a ≠ 2 → a = -2 :=
by 
  intro h1 h2
  sorry

end value_of_a_l122_122158


namespace initial_wine_volume_l122_122769

theorem initial_wine_volume (x : ℝ) 
  (h₁ : ∀ k : ℝ, k = x → ∀ n : ℕ, n = 3 → 
    (∀ y : ℝ, y = k - 4 * (1 - ((k - 4) / k) ^ n) + 2.5)) :
  x = 16 := by
  sorry

end initial_wine_volume_l122_122769


namespace percentage_discount_l122_122048

-- Define the given conditions
def equal_contribution (total: ℕ) (num_people: ℕ) := total / num_people

def original_contribution (amount_paid: ℕ) (discount: ℕ) := amount_paid + discount

def total_original_cost (individual_original: ℕ) (num_people: ℕ) := individual_original * num_people

def discount_amount (original_cost: ℕ) (discounted_cost: ℕ) := original_cost - discounted_cost

def discount_percentage (discount: ℕ) (original_cost: ℕ) := (discount * 100) / original_cost

-- Given conditions
def given_total := 48
def given_num_people := 3
def amount_paid_each := equal_contribution given_total given_num_people
def discount_each := 4
def original_payment_each := original_contribution amount_paid_each discount_each
def original_total_cost := total_original_cost original_payment_each given_num_people
def paid_total := 48

-- Question: What is the percentage discount
theorem percentage_discount :
  discount_percentage (discount_amount original_total_cost paid_total) original_total_cost = 20 :=
by
  sorry

end percentage_discount_l122_122048


namespace more_time_in_swamp_l122_122186

theorem more_time_in_swamp (a b c : ℝ) 
  (h1 : a + b + c = 4) 
  (h2 : 2 * a + 4 * b + 6 * c = 15) : a > c :=
by {
  sorry
}

end more_time_in_swamp_l122_122186


namespace parabola_vertex_l122_122327

theorem parabola_vertex :
  (∃ h k : ℝ, ∀ x : ℝ, (y : ℝ) = (x - 2)^2 + 5 ∧ h = 2 ∧ k = 5) :=
sorry

end parabola_vertex_l122_122327


namespace magnitude_of_angle_A_range_of_f_C_l122_122298

noncomputable def f_C (C : ℝ) : ℝ := 1 - (2 * real.cos (2 * C)) / (1 + real.tan C)

open real

variables {a b c : ℝ}
variables (A B C : ℝ)
variables (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π/2)
variables (h_eq1 : (2 * a) * cos C + 1 * (c - 2 * b) = 0)

theorem magnitude_of_angle_A :
  A = π / 3 :=
sorry

theorem range_of_f_C :
  set.Ioo ((√3 - 1) / 2) √2 = set.range f_C :=
sorry

end magnitude_of_angle_A_range_of_f_C_l122_122298


namespace maximum_tied_teams_in_tournament_l122_122672

theorem maximum_tied_teams_in_tournament : 
  ∀ (n : ℕ), n = 8 →
  (∀ (wins : ℕ), wins = (n * (n - 1)) / 2 →
   ∃ (k : ℕ), k ≤ n ∧ (k > 7 → false) ∧ 
               (∃ (w : ℕ), k * w = wins)) :=
by
  intros n hn wins hw
  use 7
  split
  · exact (by linarith)
  · intro h
    exfalso
    exact h (by linarith)
  · use 4
    calc
      7 * 4 = 28 : by norm_num
      ... = 28 : by rw hw; linarith
  
-- The proof is omitted as per instructions ("sorry" can be used to indicate this).

end maximum_tied_teams_in_tournament_l122_122672


namespace greatest_product_sum_300_l122_122356

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l122_122356


namespace find_a_for_even_function_l122_122009

theorem find_a_for_even_function (f : ℝ → ℝ) (a : ℝ) 
  (h_even : ∀ x, f x = f (-x)) 
  (h_neg : ∀ x, x < 0 → f x = x^2 + a * x) 
  (h_value : f 3 = 3) : a = 2 :=
sorry

end find_a_for_even_function_l122_122009


namespace exterior_angle_of_octagon_is_45_degrees_l122_122296

noncomputable def exterior_angle_of_regular_octagon : ℝ :=
  let n : ℝ := 8
  let interior_angle_sum := 180 * (n - 2) -- This is the sum of interior angles of any n-gon
  let each_interior_angle := interior_angle_sum / n -- Each interior angle in a regular polygon
  let each_exterior_angle := 180 - each_interior_angle -- Exterior angle is supplement of interior angle
  each_exterior_angle

theorem exterior_angle_of_octagon_is_45_degrees :
  exterior_angle_of_regular_octagon = 45 := by
  sorry

end exterior_angle_of_octagon_is_45_degrees_l122_122296


namespace magnitude_of_complex_l122_122123

open Complex

theorem magnitude_of_complex :
  abs (Complex.mk (2/3) (-4/5)) = Real.sqrt 244 / 15 :=
by
  -- Placeholder for the actual proof
  sorry

end magnitude_of_complex_l122_122123


namespace goldie_total_earnings_l122_122654

-- Define weekly earnings based on hours and rates
def earnings_first_week (hours_dog_walking hours_medication : ℕ) : ℕ :=
  (hours_dog_walking * 5) + (hours_medication * 8)

def earnings_second_week (hours_feeding hours_cleaning hours_playing : ℕ) : ℕ :=
  (hours_feeding * 6) + (hours_cleaning * 4) + (hours_playing * 3)

-- Given conditions for hours worked each task in two weeks
def hours_dog_walking : ℕ := 12
def hours_medication : ℕ := 8
def hours_feeding : ℕ := 10
def hours_cleaning : ℕ := 15
def hours_playing : ℕ := 5

-- Proof statement: Total earnings over two weeks equals $259
theorem goldie_total_earnings : 
  (earnings_first_week hours_dog_walking hours_medication) + 
  (earnings_second_week hours_feeding hours_cleaning hours_playing) = 259 :=
by
  sorry

end goldie_total_earnings_l122_122654


namespace value_of_expression_in_third_quadrant_l122_122658

theorem value_of_expression_in_third_quadrant (α : ℝ) (h1 : 180 < α ∧ α < 270) :
  (2 * Real.sin α) / Real.sqrt (1 - Real.cos α ^ 2) = -2 := by
  sorry

end value_of_expression_in_third_quadrant_l122_122658


namespace double_24_times_10_pow_8_l122_122791

theorem double_24_times_10_pow_8 : 2 * (2.4 * 10^8) = 4.8 * 10^8 :=
by
  sorry

end double_24_times_10_pow_8_l122_122791


namespace intersection_distance_squared_l122_122063

-- Definitions for the circles
def circle1 (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 25
def circle2 (x y : ℝ) : Prop := (x - 1)^2 + (y - 4)^2 = 9

-- Statement to prove
theorem intersection_distance_squared : 
  ∃ C D : ℝ × ℝ, circle1 C.1 C.2 ∧ circle2 C.1 C.2 ∧ circle1 D.1 D.2 ∧ circle2 D.1 D.2 ∧ 
  (C ≠ D) ∧ ((C.1 - D.1)^2 + (C.2 - D.2)^2 = 224 / 9) :=
sorry

end intersection_distance_squared_l122_122063


namespace total_monkeys_l122_122112

theorem total_monkeys (x : ℕ) (h : (1 / 8 : ℝ) * x ^ 2 + 12 = x) : x = 48 :=
sorry

end total_monkeys_l122_122112


namespace max_product_two_integers_sum_300_l122_122345

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l122_122345


namespace infinite_positive_integer_solutions_l122_122623

theorem infinite_positive_integer_solutions :
  ∃ (k : ℕ), ∀ (n : ℕ), n > 24 → ∃ k > 24, k = n :=
sorry

end infinite_positive_integer_solutions_l122_122623


namespace probability_sqrt_less_than_nine_l122_122556

/-- Define the set of two-digit integers --/
def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Define the condition that the square root of the number is less than 9 --/
def sqrt_less_than_nine (n : Nat) : Prop := n < 81

/-- The number of integers from 10 to 80 --/
lemma count_satisfying_sqrt (n : Nat) : Prop :=
  is_two_digit n ∧ sqrt_less_than_nine n → n < 81

/-- Total number of two-digit integers --/
lemma count_two_digit_total (n : Nat) : Prop := is_two_digit n 

/-- The probability that a randomly selected two-digit integer's square root is less than 9. --/
theorem probability_sqrt_less_than_nine : 
  (∃ n, count_satisfying_sqrt n) / (∃ n, count_two_digit_total n) = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l122_122556


namespace subtraction_of_fractions_l122_122986

theorem subtraction_of_fractions : (5 / 9) - (1 / 6) = 7 / 18 :=
by
  sorry

end subtraction_of_fractions_l122_122986


namespace complete_square_form_l122_122992

theorem complete_square_form (a b x : ℝ) : 
  ∃ (p : ℝ) (q : ℝ), 
  (p = x ∧ q = 1 ∧ (x^2 + 2*x + 1 = (p + q)^2)) ∧ 
  (¬ ∃ (p q : ℝ), a^2 + 4 = (a + p) * (a + q)) ∧
  (¬ ∃ (p q : ℝ), a^2 + a*b + b^2 = (a + p) * (a + q)) ∧
  (¬ ∃ (p q : ℝ), a^2 + 4*a*b + b^2 = (a + p) * (a + q)) :=
  sorry

end complete_square_form_l122_122992


namespace max_product_of_sum_300_l122_122437

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l122_122437


namespace double_and_halve_is_sixteen_l122_122578

-- Definition of the initial number
def initial_number : ℕ := 16

-- Doubling the number
def doubled (n : ℕ) : ℕ := n * 2

-- Halving the number
def halved (n : ℕ) : ℕ := n / 2

-- The theorem that needs to be proven
theorem double_and_halve_is_sixteen : halved (doubled initial_number) = 16 :=
by
  /-
  We need to prove that when the number 16 is doubled and then halved, 
  the result is 16.
  -/
  sorry

end double_and_halve_is_sixteen_l122_122578


namespace red_cars_in_lot_l122_122075

theorem red_cars_in_lot (B : ℕ) (hB : B = 90) (ratio_condition : 3 * B = 8 * R) : R = 33 :=
by
  -- Given
  have h1 : B = 90 := hB
  have h2 : 3 * B = 8 * R := ratio_condition

  -- To solve
  sorry

end red_cars_in_lot_l122_122075


namespace polynomial_root_s_eq_pm1_l122_122865

theorem polynomial_root_s_eq_pm1
  (b_3 b_2 b_1 : ℤ)
  (s : ℤ)
  (h1 : s^3 ∣ 50)
  (h2 : (s^4 + b_3 * s^3 + b_2 * s^2 + b_1 * s + 50) = 0) :
  s = 1 ∨ s = -1 :=
sorry

end polynomial_root_s_eq_pm1_l122_122865


namespace kitchen_upgrade_total_cost_l122_122209

-- Defining the given conditions
def num_cabinet_knobs : ℕ := 18
def cost_per_cabinet_knob : ℚ := 2.50

def num_drawer_pulls : ℕ := 8
def cost_per_drawer_pull : ℚ := 4

-- Definition of the total cost function
def total_cost : ℚ :=
  (num_cabinet_knobs * cost_per_cabinet_knob) + (num_drawer_pulls * cost_per_drawer_pull)

-- The theorem to prove the total cost is $77.00
theorem kitchen_upgrade_total_cost : total_cost = 77 := by
  sorry

end kitchen_upgrade_total_cost_l122_122209


namespace trains_crossing_time_l122_122339

noncomputable def time_to_cross_each_other (L T1 T2 : ℝ) (H1 : L = 120) (H2 : T1 = 10) (H3 : T2 = 16) : ℝ :=
  let S1 := L / T1
  let S2 := L / T2
  let S := S1 + S2
  let D := L + L
  D / S

theorem trains_crossing_time : time_to_cross_each_other 120 10 16 (by rfl) (by rfl) (by rfl) = 240 / (12 + 7.5) :=
  sorry

end trains_crossing_time_l122_122339


namespace muffin_cost_ratio_l122_122702

theorem muffin_cost_ratio (m b : ℝ) 
  (h1 : 5 * m + 4 * b = 20)
  (h2 : 3 * (5 * m + 4 * b) = 60)
  (h3 : 3 * m + 18 * b = 60) :
  m / b = 13 / 4 :=
by
  sorry

end muffin_cost_ratio_l122_122702


namespace smallest_x_abs_eq_18_l122_122630

theorem smallest_x_abs_eq_18 : 
  ∃ x : ℝ, (|2 * x + 5| = 18) ∧ (∀ y : ℝ, (|2 * y + 5| = 18) → x ≤ y) :=
sorry

end smallest_x_abs_eq_18_l122_122630


namespace otimes_identity_l122_122621

def otimes (x y : ℝ) : ℝ := x^2 - y^2

theorem otimes_identity (h : ℝ) : otimes h (otimes h h) = h^2 :=
by
  sorry

end otimes_identity_l122_122621


namespace nonneg_integer_solution_l122_122127

theorem nonneg_integer_solution (a b c : ℕ) (h : 5^a * 7^b + 4 = 3^c) : (a, b, c) = (1, 0, 2) := 
sorry

end nonneg_integer_solution_l122_122127


namespace value_of_x2_plus_y2_l122_122911

theorem value_of_x2_plus_y2 (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 :=
by
  sorry

end value_of_x2_plus_y2_l122_122911


namespace range_of_S_l122_122821

variable {a b x : ℝ}
def f (x : ℝ) (a b : ℝ) : ℝ := a * x + b

theorem range_of_S (h1 : ∀ x ∈ Set.Icc 0 1, |f x a b| ≤ 1) :
  ∃ l u, -2 ≤ l ∧ u ≤ 9 / 4 ∧ ∀ (S : ℝ), (S = (a + 1) * (b + 1)) → l ≤ S ∧ S ≤ u :=
by
  sorry

end range_of_S_l122_122821


namespace monotonicity_and_max_of_f_g_range_of_a_l122_122901

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x^2

noncomputable def g (x a : ℝ) : ℝ := x * Real.exp x - (a - 1) * x^2 - x - 2 * Real.log x

theorem monotonicity_and_max_of_f : 
  (∀ x, 0 < x → x < 1 → f x > f (x + 1)) ∧ 
  (∀ x, x > 1 → f x < f (x - 1)) ∧ 
  (f 1 = -1) := 
by
  sorry

theorem g_range_of_a (a : ℝ) : 
  (∀ x, x > 0 → f x + g x a ≥ 0) → (a ≤ 1) := 
by
  sorry

end monotonicity_and_max_of_f_g_range_of_a_l122_122901


namespace max_product_300_l122_122549

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l122_122549


namespace number_of_dogs_in_shelter_l122_122667

variables (D C R P : ℕ)

-- Conditions
axiom h1 : 15 * C = 7 * D
axiom h2 : 9 * P = 5 * R
axiom h3 : 15 * (C + 8) = 11 * D
axiom h4 : 7 * P = 5 * (R + 6)

theorem number_of_dogs_in_shelter : D = 30 :=
by sorry

end number_of_dogs_in_shelter_l122_122667


namespace exists_subset_sum_divisible_by_2n_l122_122177

open BigOperators

theorem exists_subset_sum_divisible_by_2n (n : ℕ) (hn : n ≥ 4) (a : Fin n → ℤ)
  (h_distinct : ∀ i j : Fin n, i ≠ j → a i ≠ a j)
  (h_interval : ∀ i : Fin n, 0 < a i ∧ a i < 2 * n) :
  ∃ (s : Finset (Fin n)), (∑ i in s, a i) % (2 * n) = 0 :=
sorry

end exists_subset_sum_divisible_by_2n_l122_122177


namespace quadratic_inequality_solution_set_l122_122840

theorem quadratic_inequality_solution_set (x : ℝ) : (x + 3) * (2 - x) < 0 ↔ x < -3 ∨ x > 2 := 
sorry

end quadratic_inequality_solution_set_l122_122840


namespace additional_pencils_l122_122626

theorem additional_pencils (original_pencils new_pencils per_container distributed_pencils : ℕ)
  (h1 : original_pencils = 150)
  (h2 : per_container = 5)
  (h3 : distributed_pencils = 36)
  (h4 : new_pencils = distributed_pencils * per_container) :
  (new_pencils - original_pencils) = 30 :=
by
  -- Proof will go here
  sorry

end additional_pencils_l122_122626


namespace greatest_product_sum_300_l122_122418

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l122_122418


namespace find_divisor_l122_122584

theorem find_divisor 
    (x : ℕ) 
    (h : 83 = 9 * x + 2) : 
    x = 9 := 
  sorry

end find_divisor_l122_122584


namespace greatest_product_sum_300_l122_122370

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l122_122370


namespace greatest_product_l122_122526

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l122_122526


namespace solve_for_a_l122_122155

theorem solve_for_a (a : ℝ) (h : |2 * a + 1| = 3 * |a| - 2) : a = -1 ∨ a = 3 :=
by
  sorry

end solve_for_a_l122_122155


namespace sum_of_x_values_l122_122752

theorem sum_of_x_values (y x : ℝ) (h1 : y = 6) (h2 : x^2 + y^2 = 144) : x + (-x) = 0 :=
by
  sorry

end sum_of_x_values_l122_122752


namespace max_product_of_sum_300_l122_122428

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l122_122428


namespace negativity_of_c_plus_b_l122_122316

variable (a b c : ℝ)

def isWithinBounds : Prop := (1 < a ∧ a < 2) ∧ (0 < b ∧ b < 1) ∧ (-2 < c ∧ c < -1)

theorem negativity_of_c_plus_b (h : isWithinBounds a b c) : c + b < 0 :=
sorry

end negativity_of_c_plus_b_l122_122316


namespace greatest_product_sum_300_l122_122360

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l122_122360


namespace option_C_correct_l122_122855

theorem option_C_correct (x : ℝ) (hx : 0 < x) : x + 1 / x ≥ 2 :=
sorry

end option_C_correct_l122_122855


namespace brianna_fraction_left_l122_122874

theorem brianna_fraction_left (m n c : ℕ) (h : (1 : ℚ) / 4 * m = 1 / 2 * n * c) : 
  (m - (n * c) - (1 / 10 * m)) / m = 2 / 5 :=
by
  sorry

end brianna_fraction_left_l122_122874


namespace total_chickens_l122_122336

-- Definitions from conditions
def ducks : ℕ := 40
def rabbits : ℕ := 30
def hens : ℕ := ducks + 20
def roosters : ℕ := rabbits - 10

-- Theorem statement: total number of chickens
theorem total_chickens : hens + roosters = 80 := 
sorry

end total_chickens_l122_122336


namespace greatest_product_two_ints_sum_300_l122_122439

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l122_122439


namespace initial_outlay_is_10000_l122_122317

theorem initial_outlay_is_10000 
  (I : ℝ)
  (manufacturing_cost_per_set : ℝ := 20)
  (selling_price_per_set : ℝ := 50)
  (num_sets : ℝ := 500)
  (profit : ℝ := 5000) :
  profit = (selling_price_per_set * num_sets) - (I + manufacturing_cost_per_set * num_sets) → I = 10000 :=
by
  intro h
  sorry

end initial_outlay_is_10000_l122_122317


namespace sufficient_but_not_necessary_l122_122232

theorem sufficient_but_not_necessary (x : ℝ) : (x = 1 → x^2 = 1) ∧ (x^2 = 1 → x = 1 ∨ x = -1) :=
by
  sorry

end sufficient_but_not_necessary_l122_122232


namespace magnitude_v_l122_122321

open Complex

theorem magnitude_v (u v : ℂ) (h1 : u * v = 20 - 15 * Complex.I) (h2 : Complex.abs u = 5) :
  Complex.abs v = 5 := by
  sorry

end magnitude_v_l122_122321


namespace negation_of_universal_proposition_l122_122836
open Classical

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 > 0) → ∃ x : ℝ, ¬(x^2 > 0) :=
by
  intro h
  have := (not_forall.mp h)
  exact this

end negation_of_universal_proposition_l122_122836


namespace min_value_c_and_d_l122_122308

theorem min_value_c_and_d (c d : ℝ) (h1 : c > 0) (h2 : d > 0)
  (h3 : c^2 - 12 * d ≥ 0)
  (h4 : 9 * d^2 - 4 * c ≥ 0) :
  c + d ≥ 5.74 :=
sorry

end min_value_c_and_d_l122_122308


namespace greatest_product_l122_122525

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l122_122525


namespace correct_result_is_102357_l122_122995

-- Defining the conditions
def number (f : ℕ) : Prop := f * 153 = 102357

-- Stating the proof problem
theorem correct_result_is_102357 (f : ℕ) (h : f * 153 = 102325) (wrong_digits : ℕ) :
  (number f) :=
by
  sorry

end correct_result_is_102357_l122_122995


namespace range_of_a_l122_122783

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 2 ≤ x ∧ x ≤ 4 → x^2 + a * x - 2 < 0) → a < -1 :=
by
  sorry

end range_of_a_l122_122783


namespace fraction_zero_implies_a_eq_neg2_l122_122160

theorem fraction_zero_implies_a_eq_neg2 (a : ℝ) (h : (a^2 - 4) / (a - 2) = 0) (h2 : a ≠ 2) : a = -2 :=
sorry

end fraction_zero_implies_a_eq_neg2_l122_122160


namespace greatest_product_sum_300_l122_122423

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l122_122423


namespace square_area_from_wire_bent_as_circle_l122_122229

theorem square_area_from_wire_bent_as_circle 
  (radius : ℝ) 
  (h_radius : radius = 56)
  (π_ineq : π > 3.1415) : 
  ∃ (A : ℝ), A = 784 * π^2 := 
by 
  sorry

end square_area_from_wire_bent_as_circle_l122_122229


namespace total_pumped_volume_l122_122323

def powerJetA_rate : ℕ := 360
def powerJetB_rate : ℕ := 540
def powerJetA_time : ℕ := 30
def powerJetB_time : ℕ := 45

def pump_volume (rate : ℕ) (minutes : ℕ) : ℕ :=
  rate * (minutes / 60)

theorem total_pumped_volume : 
  pump_volume powerJetA_rate powerJetA_time + pump_volume powerJetB_rate powerJetB_time = 585 := 
by
  sorry

end total_pumped_volume_l122_122323


namespace polynomial_div_remainder_l122_122629

theorem polynomial_div_remainder (x : ℝ) : 
  (x^4 % (x^2 + 7*x + 2)) = -315*x - 94 := 
by
  sorry

end polynomial_div_remainder_l122_122629


namespace greatest_product_obtainable_l122_122400

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l122_122400


namespace kitchen_upgrade_total_cost_l122_122211

-- Defining the given conditions
def num_cabinet_knobs : ℕ := 18
def cost_per_cabinet_knob : ℚ := 2.50

def num_drawer_pulls : ℕ := 8
def cost_per_drawer_pull : ℚ := 4

-- Definition of the total cost function
def total_cost : ℚ :=
  (num_cabinet_knobs * cost_per_cabinet_knob) + (num_drawer_pulls * cost_per_drawer_pull)

-- The theorem to prove the total cost is $77.00
theorem kitchen_upgrade_total_cost : total_cost = 77 := by
  sorry

end kitchen_upgrade_total_cost_l122_122211


namespace max_product_of_sum_300_l122_122426

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l122_122426


namespace total_height_of_three_buildings_l122_122594

theorem total_height_of_three_buildings :
  let h1 := 600
  let h2 := 2 * h1
  let h3 := 3 * (h1 + h2)
  h1 + h2 + h3 = 7200 :=
by
  sorry

end total_height_of_three_buildings_l122_122594


namespace max_switches_not_exceed_comb_l122_122205

theorem max_switches_not_exceed_comb :
  ∀ (n : ℕ), ∃ (h : ℕ → ℕ),
  strict_mono h ∧
  (∀ (k : ℕ), (2 ≤ k ∧ k < n) → (h k = k + 1)) ∧
  (maximum_switches h n ≤ nat.choose n 3) :=
begin
  sorry
end

end max_switches_not_exceed_comb_l122_122205


namespace paul_sandwiches_in_6_days_l122_122053

def sandwiches_eaten_in_n_days (n : ℕ) : ℕ :=
  let day1 := 2
  let day2 := 2 * day1
  let day3 := 2 * day2
  let three_day_total := day1 + day2 + day3
  three_day_total * (n / 3)

theorem paul_sandwiches_in_6_days : sandwiches_eaten_in_n_days 6 = 28 :=
by
  sorry

end paul_sandwiches_in_6_days_l122_122053


namespace sufficient_but_not_necessary_l122_122934

theorem sufficient_but_not_necessary (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c > 0) :
  (a > b ∧ b > 0 ∧ c > 0) → (a / (a + c) > b / (b + c)) :=
by
  intros
  sorry

end sufficient_but_not_necessary_l122_122934


namespace max_product_of_sum_300_l122_122473

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l122_122473


namespace total_toys_l122_122248

theorem total_toys (bill_toys hana_toys hash_toys: ℕ) 
  (hb: bill_toys = 60)
  (hh: hana_toys = (5 * bill_toys) / 6)
  (hs: hash_toys = (hana_toys / 2) + 9) :
  (bill_toys + hana_toys + hash_toys) = 144 :=
by
  sorry

end total_toys_l122_122248


namespace parabola_vertex_origin_through_point_l122_122975

theorem parabola_vertex_origin_through_point :
  (∃ p, p > 0 ∧ x^2 = 2 * p * y ∧ (x, y) = (-4, 4) → x^2 = 4 * y) ∨
  (∃ p, p > 0 ∧ y^2 = -2 * p * x ∧ (x, y) = (-4, 4) → y^2 = -4 * x) :=
sorry

end parabola_vertex_origin_through_point_l122_122975


namespace greatest_product_two_ints_sum_300_l122_122438

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l122_122438


namespace max_product_300_l122_122537

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l122_122537


namespace age_of_son_l122_122240

theorem age_of_son (D S : ℕ) (h₁ : S = D / 4) (h₂ : D - S = 27) (h₃ : D = 36) : S = 9 :=
by
  sorry

end age_of_son_l122_122240


namespace greatest_product_of_two_integers_with_sum_300_l122_122516

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l122_122516


namespace greatest_product_two_ints_sum_300_l122_122442

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l122_122442


namespace intersection_eq_l122_122813

variable {x : ℝ}

def set_A := {x : ℝ | x^2 - 4 * x < 0}
def set_B := {x : ℝ | 1 / 3 ≤ x ∧ x ≤ 5}
def set_intersection := {x : ℝ | 1 / 3 ≤ x ∧ x < 4}

theorem intersection_eq : (set_A ∩ set_B) = set_intersection := by
  sorry

end intersection_eq_l122_122813


namespace greatest_product_sum_300_l122_122364

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l122_122364


namespace cost_price_of_watch_l122_122244

theorem cost_price_of_watch
  (C : ℝ)
  (h1 : 0.9 * C + 225 = 1.05 * C) :
  C = 1500 :=
by sorry

end cost_price_of_watch_l122_122244


namespace greatest_product_sum_300_l122_122410

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l122_122410


namespace problem_part_I_problem_part_II_l122_122033

theorem problem_part_I (A B C : ℝ)
  (h1 : 0 < A) 
  (h2 : A < π / 2)
  (h3 : 1 + (Real.sqrt 3 / 3) * Real.sin (2 * A) = 2 * (Real.sin ((B + C) / 2))^2) : 
  A = π / 3 := 
sorry

theorem problem_part_II (A B C R S : ℝ)
  (h1 : A = π / 3)
  (h2 : R = 2 * Real.sqrt 3) 
  (h3 : S = (1 / 2) * (6 * (Real.sin A)) * (Real.sqrt 3 / 2)) :
  S = 9 * Real.sqrt 3 :=
sorry

end problem_part_I_problem_part_II_l122_122033


namespace cost_of_flowers_l122_122581

theorem cost_of_flowers 
  (interval : ℕ) (perimeter : ℕ) (cost_per_flower : ℕ)
  (h_interval : interval = 30)
  (h_perimeter : perimeter = 1500)
  (h_cost : cost_per_flower = 5000) :
  (perimeter / interval) * cost_per_flower = 250000 :=
by
  sorry

end cost_of_flowers_l122_122581


namespace greatest_product_obtainable_l122_122407

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l122_122407


namespace max_product_of_sum_300_l122_122424

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l122_122424


namespace ratio_of_areas_l122_122792

def angle_X : ℝ := 60
def angle_Y : ℝ := 40
def radius_X : ℝ
def radius_Y : ℝ
def arc_length (θ r : ℝ) : ℝ := (θ / 360) * (2 * Real.pi * r)

theorem ratio_of_areas (angle_X_eq : angle_X / 360 * 2 * Real.pi * radius_X = angle_Y / 360 * 2 * Real.pi * radius_Y) :
  (Real.pi * radius_X ^ 2) / (Real.pi * radius_Y ^ 2) = 9 / 4 :=
by
  sorry

end ratio_of_areas_l122_122792


namespace jane_last_segment_speed_l122_122939

theorem jane_last_segment_speed :
  let total_distance := 120  -- in miles
  let total_time := (75 / 60)  -- in hours
  let segment_time := (25 / 60)  -- in hours
  let speed1 := 75  -- in mph
  let speed2 := 80  -- in mph
  let overall_avg_speed := total_distance / total_time
  let x := (3 * overall_avg_speed) - speed1 - speed2
  x = 133 :=
by { sorry }

end jane_last_segment_speed_l122_122939


namespace construct_origin_from_A_and_B_l122_122079

structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨1, 2⟩
def B : Point := ⟨3, 1⟩
def isAboveAndToLeft (p₁ p₂ : Point) : Prop := p₁.x < p₂.x ∧ p₁.y > p₂.y
def isOriginConstructed (A B : Point) : Prop := ∃ O : Point, O = ⟨0, 0⟩

theorem construct_origin_from_A_and_B : 
  isAboveAndToLeft A B → isOriginConstructed A B :=
by
  sorry

end construct_origin_from_A_and_B_l122_122079


namespace max_product_300_l122_122543

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l122_122543


namespace total_hours_jade_krista_driving_l122_122936

theorem total_hours_jade_krista_driving (d : ℕ) (h_jade_per_day h_krista_per_day : ℕ) :
  (d = 3) → (h_jade_per_day = 8) → (h_krista_per_day = 6) → 
  (d * h_jade_per_day + d * h_krista_per_day = 42) := 
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  calc
    3 * 8 + 3 * 6 = 24 + 18 := by norm_num
    ... = 42 := by norm_num

end total_hours_jade_krista_driving_l122_122936


namespace intersection_A_B_l122_122139

def A := { x : Real | -3 < x ∧ x < 2 }
def B := { x : Real | x^2 + 4*x - 5 ≤ 0 }

theorem intersection_A_B :
  (A ∩ B = { x : Real | -3 < x ∧ x ≤ 1 }) := by
  sorry

end intersection_A_B_l122_122139


namespace greatest_product_sum_300_l122_122420

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l122_122420


namespace geometric_arithmetic_sequence_l122_122978

theorem geometric_arithmetic_sequence (a q : ℝ) 
    (h₁ : a + a * q + a * q ^ 2 = 19) 
    (h₂ : a * (q - 1) = -1) : 
  (a = 4 ∧ q = 1.5) ∨ (a = 9 ∧ q = 2/3) :=
by
  sorry

end geometric_arithmetic_sequence_l122_122978


namespace prob_sqrt_less_than_nine_l122_122569

/-- The probability that the square root of a randomly selected 
two-digit whole number is less than nine is 71/90. -/
theorem prob_sqrt_less_than_nine : (let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99};
                                     let A := {n : ℕ | 10 ≤ n ∧ n < 81};
                                     (A.card / S.card : ℚ) = 71 / 90) :=
by
  sorry

end prob_sqrt_less_than_nine_l122_122569


namespace koala_fiber_intake_l122_122861

theorem koala_fiber_intake (x : ℝ) (h1 : 0.3 * x = 12) : x = 40 := 
by 
  sorry

end koala_fiber_intake_l122_122861


namespace handshake_count_l122_122606

theorem handshake_count {teams : Fin 4 → Fin 2 → Prop}
    (h_teams_disjoint : ∀ (i j : Fin 4) (x y : Fin 2), i ≠ j → teams i x → teams j y → x ≠ y)
    (unique_partner : ∀ (i : Fin 4) (x1 x2 : Fin 2), teams i x1 → teams i x2 → x1 = x2) : 
    24 = (∑ i : Fin 8, (∑ j : Fin 8, if i ≠ j ∧ ¬(∃ k : Fin 4, teams k i ∧ teams k j) then 1 else 0)) / 2 :=
by sorry

end handshake_count_l122_122606


namespace ratio_of_M_to_N_l122_122788

theorem ratio_of_M_to_N 
  (M Q P N : ℝ) 
  (h1 : M = 0.4 * Q) 
  (h2 : Q = 0.25 * P) 
  (h3 : N = 0.75 * P) : 
  M / N = 2 / 15 := 
sorry

end ratio_of_M_to_N_l122_122788


namespace flying_scotsman_more_carriages_l122_122993

theorem flying_scotsman_more_carriages :
  ∀ (E N No F T D : ℕ),
    E = 130 →
    E = N + 20 →
    No = 100 →
    T = 460 →
    D = F - No →
    F + E + N + No = T →
    D = 20 :=
by
  intros E N No F T D hE1 hE2 hNo hT hD hSum
  sorry

end flying_scotsman_more_carriages_l122_122993


namespace contrapositive_of_square_inequality_l122_122326

theorem contrapositive_of_square_inequality (x y : ℝ) :
  (x^2 > y^2 → x > y) ↔ (x ≤ y → x^2 ≤ y^2) :=
by
  sorry

end contrapositive_of_square_inequality_l122_122326


namespace greatest_product_l122_122530

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l122_122530


namespace max_product_of_two_integers_whose_sum_is_300_l122_122499

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l122_122499


namespace find_pairs_l122_122886
open Nat

theorem find_pairs (x p : ℕ) (hp : p.Prime) (hxp : x ≤ 2 * p) (hdiv : x^(p-1) ∣ (p-1)^x + 1) : 
  (x = 1 ∧ p.Prime) ∨ (x = 2 ∧ p = 2) ∨ (x = 1 ∧ p.Prime) ∨ (x = 3 ∧ p = 3) := 
by
  sorry


end find_pairs_l122_122886


namespace fifth_number_l122_122850

def sequence_sum (a b : ℕ) : ℕ :=
  a + b + (a + b) + (a + 2 * b) + (2 * a + 3 * b) + (3 * a + 5 * b)

theorem fifth_number (a b : ℕ) (h : sequence_sum a b = 2008) : 2 * a + 3 * b = 502 := by
  sorry

end fifth_number_l122_122850


namespace equilateral_triangle_t_gt_a_squared_l122_122176

theorem equilateral_triangle_t_gt_a_squared {a x : ℝ} (h0 : 0 ≤ x) (h1 : x ≤ a) :
  2 * x^2 - 2 * a * x + 3 * a^2 > a^2 :=
by {
  sorry
}

end equilateral_triangle_t_gt_a_squared_l122_122176


namespace alice_always_wins_l122_122870

theorem alice_always_wins (n : ℕ) (initial_coins : ℕ) (alice_first_move : ℕ) (total_coins : ℕ) :
  initial_coins = 1331 → alice_first_move = 1 → total_coins = 1331 →
  (∀ (k : ℕ), 
    let alice_total := (k * (k + 1)) / 2;
    let basilio_min_total := (k * (k - 1)) / 2;
    let basilio_max_total := (k * (k + 1)) / 2 - 1;
    k * k ≤ total_coins ∧ total_coins ≤ k * (k + 1) - 1 →
    ¬ (total_coins = k * k + k - 1 ∨ total_coins = k * (k + 1) - 1)) →
  alice_first_move = 1 ∧ initial_coins = 1331 ∧ total_coins = 1331 → alice_wins :=
sorry

end alice_always_wins_l122_122870


namespace sally_lost_two_balloons_l122_122696

-- Condition: Sally originally had 9 orange balloons.
def original_orange_balloons := 9

-- Condition: Sally now has 7 orange balloons.
def current_orange_balloons := 7

-- Problem: Prove that Sally lost 2 orange balloons.
theorem sally_lost_two_balloons : original_orange_balloons - current_orange_balloons = 2 := by
  sorry

end sally_lost_two_balloons_l122_122696


namespace find_f_three_l122_122146

variable {α : Type*} [LinearOrderedField α]

def f (a b c x : α) := a * x^5 - b * x^3 + c * x - 3

theorem find_f_three (a b c : α) (h : f a b c (-3) = 7) : f a b c 3 = -13 :=
by sorry

end find_f_three_l122_122146


namespace xyz_abs_eq_one_l122_122043

theorem xyz_abs_eq_one (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (cond : x + 1/y = y + 1/z ∧ y + 1/z = z + 1/x + 1) : |x * y * z| = 1 :=
sorry

end xyz_abs_eq_one_l122_122043


namespace tan_315_deg_l122_122256

theorem tan_315_deg : Real.tan (315 * Real.pi / 180) = -1 := sorry

end tan_315_deg_l122_122256


namespace smallest_prime_sum_l122_122265

theorem smallest_prime_sum (a b c d : ℕ) (ha : Prime a) (hb : Prime b) (hc : Prime c) (hd : Prime d)
  (hab : a ≠ b) (hac : a ≠ c) (had : a ≠ d) (hbc : b ≠ c) (hbd : b ≠ d) (hcd : c ≠ d)
  (H1 : Prime (a + b + c + d))
  (H2 : Prime (a + b)) (H3 : Prime (a + c)) (H4 : Prime (a + d)) (H5 : Prime (b + c)) (H6 : Prime (b + d)) (H7 : Prime (c + d))
  (H8 : Prime (a + b + c)) (H9 : Prime (a + b + d)) (H10 : Prime (a + c + d)) (H11 : Prime (b + c + d))
  : a + b + c + d = 31 :=
sorry

end smallest_prime_sum_l122_122265


namespace ratio_volumes_l122_122084

theorem ratio_volumes (hA rA hB rB : ℝ) (hA_def : hA = 30) (rA_def : rA = 15) (hB_def : hB = rA) (rB_def : rB = 2 * hA) :
    (1 / 3 * Real.pi * rA^2 * hA) / (1 / 3 * Real.pi * rB^2 * hB) = 1 / 24 :=
by
  -- skipping the proof
  sorry

end ratio_volumes_l122_122084


namespace primes_equal_l122_122712

def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_equal (p q r n : ℕ) (h_prime_p : is_prime p) (h_prime_q : is_prime q)
(h_prime_r : is_prime r) (h_pos_n : 0 < n)
(h1 : (p + n) % (q * r) = 0)
(h2 : (q + n) % (r * p) = 0)
(h3 : (r + n) % (p * q) = 0) : p = q ∧ q = r := by
  sorry

end primes_equal_l122_122712


namespace ones_divisible_by_d_l122_122826

theorem ones_divisible_by_d (d : ℕ) (h1 : ¬ (2 ∣ d)) (h2 : ¬ (5 ∣ d))  : 
  ∃ n, (∃ k : ℕ, n = 10^k - 1) ∧ n % d = 0 := 
sorry

end ones_divisible_by_d_l122_122826


namespace crayons_per_box_l122_122824

theorem crayons_per_box (total_crayons : ℝ) (total_boxes : ℝ) (h1 : total_crayons = 7.0) (h2 : total_boxes = 1.4) : total_crayons / total_boxes = 5 :=
by
  sorry

end crayons_per_box_l122_122824


namespace sum_of_2016_integers_positive_l122_122193

theorem sum_of_2016_integers_positive 
  (a : Finₓ 2016 → ℤ) 
  (h : ∀ S : Finset (Finₓ 2016), S.card = 1008 → ∑ i in S, a i > 0) 
  : ∑ i, a i > 0 := sorry

end sum_of_2016_integers_positive_l122_122193


namespace probability_single_trial_l122_122932

theorem probability_single_trial 
  (p : ℝ) 
  (h₁ : ∀ n : ℕ, 1 ≤ n → ∃ x : ℝ, x = (1 - (1 - p) ^ n)) 
  (h₂ : 1 - (1 - p) ^ 4 = 65 / 81) : 
  p = 1 / 3 :=
by 
  sorry

end probability_single_trial_l122_122932


namespace train_speed_l122_122111

theorem train_speed (length : ℝ) (time : ℝ) (conversion_factor : ℝ)
  (h1 : length = 500) (h2 : time = 5) (h3 : conversion_factor = 3.6) :
  (length / time) * conversion_factor = 360 :=
by
  sorry

end train_speed_l122_122111


namespace boulder_splash_width_l122_122215

theorem boulder_splash_width :
  (6 * (1/4) + 3 * (1 / 2) + 2 * b = 7) -> b = 2 := by
  sorry

end boulder_splash_width_l122_122215


namespace subtraction_of_fractions_l122_122985

theorem subtraction_of_fractions : (5 / 9) - (1 / 6) = 7 / 18 :=
by
  sorry

end subtraction_of_fractions_l122_122985


namespace scientific_notation_of_30067_l122_122190

theorem scientific_notation_of_30067 : ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 30067 = a * 10^n := by
  use 3.0067
  use 4
  sorry

end scientific_notation_of_30067_l122_122190


namespace redistribution_not_always_possible_l122_122338

theorem redistribution_not_always_possible (a b : ℕ) (h : a ≠ b) :
  ¬(∃ k : ℕ, a - k = b + k ∧ 0 ≤ k ∧ k ≤ a ∧ k ≤ b) ↔ (a + b) % 2 = 1 := 
by 
  sorry

end redistribution_not_always_possible_l122_122338


namespace scalene_triangle_angles_l122_122675

theorem scalene_triangle_angles (x y z : ℝ) (h1 : x + y + z = 180) (h2 : x ≠ y ∧ y ≠ z ∧ x ≠ z)
(h3 : x = 36 ∨ y = 36 ∨ z = 36) (h4 : x = 2 * y ∨ y = 2 * x ∨ z = 2 * x ∨ x = 2 * z ∨ y = 2 * z ∨ z = 2 * y) :
(x = 36 ∧ y = 48 ∧ z = 96) ∨ (x = 18 ∧ y = 36 ∧ z = 126) ∨ (x = 36 ∧ z = 48 ∧ y = 96) ∨ (y = 18 ∧ x = 36 ∧ z = 126) :=
sorry

end scalene_triangle_angles_l122_122675


namespace circles_are_externally_tangent_l122_122200

noncomputable def circleA : Prop := ∀ x y : ℝ, x^2 + y^2 + 4 * x + 2 * y + 1 = 0
noncomputable def circleB : Prop := ∀ x y : ℝ, x^2 + y^2 - 2 * x - 6 * y + 1 = 0

theorem circles_are_externally_tangent (hA : circleA) (hB : circleB) : 
  ∃ P Q : ℝ, (P = 5) ∧ (Q = 5) := 
by 
  -- start proving with given conditions
  sorry

end circles_are_externally_tangent_l122_122200


namespace abs_sum_less_b_l122_122789

theorem abs_sum_less_b (x : ℝ) (b : ℝ) (h : |2 * x - 8| + |2 * x - 6| < b) (hb : b > 0) : b > 2 :=
by
  sorry

end abs_sum_less_b_l122_122789


namespace greatest_product_two_ints_sum_300_l122_122447

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l122_122447


namespace value_of_a_l122_122159

theorem value_of_a (a : ℝ) : (a^2 - 4) / (a - 2) = 0 → a ≠ 2 → a = -2 :=
by 
  intro h1 h2
  sorry

end value_of_a_l122_122159


namespace laticia_total_pairs_l122_122811

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

end laticia_total_pairs_l122_122811


namespace prob_4_consecutive_baskets_prob_exactly_4_baskets_l122_122242

theorem prob_4_consecutive_baskets 
  (p : ℝ) (h : p = 1/2) : 
  (p^4 * (1 - p) + (1 - p) * p^4) = 1/16 :=
by sorry

theorem prob_exactly_4_baskets 
  (p : ℝ) (h : p = 1/2) : 
  5 * p^4 * (1 - p) = 5/32 :=
by sorry

end prob_4_consecutive_baskets_prob_exactly_4_baskets_l122_122242


namespace probability_sqrt_lt_nine_two_digit_l122_122562

theorem probability_sqrt_lt_nine_two_digit :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99} in
  let T := {n : ℕ | 10 ≤ n ∧ n < 81} in
  (Fintype.card T : ℚ) / (Fintype.card S) = 71 / 90 :=
by
  sorry

end probability_sqrt_lt_nine_two_digit_l122_122562


namespace permutation_residue_condition_l122_122126

section
variables {n : ℕ}
variables (p : Fin n → Fin n)

noncomputable def is_permutation (p : Fin n → Fin n) := 
  ∀ x, ∃! y, p y = x

def is_complete_residue_system (s : Fin n → Fin n) := 
  ∀ i j : Fin n, i ≠ j → s i ≠ s j

def sets_form_complete_residue_systems (p : Fin n → Fin n) := 
  is_complete_residue_system (λ i, (p i + i + 1) % n) ∧ 
  is_complete_residue_system (λ i, (p i - i + n.pred + 1) % n)

theorem permutation_residue_condition (hp : is_permutation p) :
  sets_form_complete_residue_systems p ↔ (n % 6 = 1 ∨ n % 6 = 5) := 
sorry

end

end permutation_residue_condition_l122_122126


namespace total_distance_apart_l122_122940

def Jay_rate : ℕ := 1 / 15 -- Jay walks 1 mile every 15 minutes
def Paul_rate : ℕ := 3 / 30 -- Paul walks 3 miles every 30 minutes
def time_in_minutes : ℕ := 120 -- 2 hours converted to minutes

def Jay_distance (rate time : ℕ) : ℕ := rate * time / 15
def Paul_distance (rate time : ℕ) : ℕ := rate * time / 30

theorem total_distance_apart : 
  Jay_distance Jay_rate time_in_minutes + Paul_distance Paul_rate time_in_minutes = 20 :=
  by
  -- Proof here
  sorry

end total_distance_apart_l122_122940


namespace find_expression_l122_122014

theorem find_expression (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = 3 * x + 2) : 
  ∀ x : ℤ, f x = 3 * x - 1 :=
sorry

end find_expression_l122_122014


namespace intersection_M_N_l122_122653

noncomputable def M : Set ℕ := { x | 0 < x ∧ x < 8 }
def N : Set ℕ := { x | ∃ n : ℕ, x = 2 * n + 1 }
def K : Set ℕ := { 1, 3, 5, 7 }

theorem intersection_M_N : M ∩ N = K :=
by sorry

end intersection_M_N_l122_122653


namespace greatest_product_sum_300_l122_122377

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l122_122377


namespace exists_root_in_interval_l122_122761

theorem exists_root_in_interval :
  ∃ (r : ℝ), 3 < r ∧ r < 3.5 ∧
  ∃ (a0 a1 a2 : ℝ), 
    a0 ≤ a1 ∧ a1 ≤ a2 ∧
    -3 ≤ a0 ∧ a0 ≤ 1 ∧ -3 ≤ a1 ∧ a1 ≤ 1 ∧ -3 ≤ a2 ∧ a2 ≤ 1 ∧
    |a2 - a1| ≤ 2 ∧ 
    ∃ (x : ℝ), x^3 + a2 * x^2 + a1 * x + a0 = 0 ∧ x = r :=
by sorry

end exists_root_in_interval_l122_122761


namespace tennis_handshakes_l122_122609

theorem tennis_handshakes :
  ∀ (teams : Fin 4 → Fin 2 → ℕ),
    (∀ i, teams i 0 ≠ teams i 1) ∧ (∀ i j a b, i ≠ j → teams i a ≠ teams j b) →
    ∃ handshakes : ℕ, handshakes = 24 :=
begin
  sorry
end

end tennis_handshakes_l122_122609


namespace greatest_product_l122_122534

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l122_122534


namespace count_possible_pairs_l122_122787

/-- There are four distinct mystery novels, three distinct fantasy novels, and three distinct biographies.
I want to choose two books with one of them being a specific mystery novel, "Mystery Masterpiece".
Prove that the number of possible pairs that include this mystery novel and one book from a different genre
is 6. -/
theorem count_possible_pairs (mystery_novels : Fin 4)
                            (fantasy_novels : Fin 3)
                            (biographies : Fin 3)
                            (MysteryMasterpiece : Fin 4):
                            (mystery_novels ≠ MysteryMasterpiece) →
                            ∀ genre : Fin 2, genre ≠ 0 ∧ genre ≠ 1 →
                            (genre = 1 → ∃ pairs : List (Fin 3), pairs.length = 3) →
                            (genre = 2 → ∃ pairs : List (Fin 3), pairs.length = 3) →
                            ∃ total_pairs : Nat, total_pairs = 6 :=
by
  intros h_ne_genres h_genres h_counts1 h_counts2
  sorry

end count_possible_pairs_l122_122787


namespace greatest_product_obtainable_l122_122404

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l122_122404


namespace find_weights_l122_122080

def item_weights (a b c d e f g h : ℕ) : Prop :=
  1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ 1 ≤ d ∧ 1 ≤ e ∧ 1 ≤ f ∧ 1 ≤ g ∧ 1 ≤ h ∧
  a > b ∧ b > c ∧ c > d ∧ d > e ∧ e > f ∧ f > g ∧ g > h ∧
  a ≤ 15 ∧ b ≤ 15 ∧ c ≤ 15 ∧ d ≤ 15 ∧ e ≤ 15 ∧ f ≤ 15 ∧ g ≤ 15 ∧ h ≤ 15

theorem find_weights (a b c d e f g h : ℕ) (hw : item_weights a b c d e f g h) 
    (h1 : d + e + f + g > a + b + c + h) 
    (h2 : e + f > d + g) 
    (h3 : e > f) : e = 11 ∧ g = 5 := sorry

end find_weights_l122_122080


namespace exponential_rule_l122_122717

theorem exponential_rule (a : ℝ) : (a ^ 3) ^ 2 = a ^ 6 :=  
  sorry

end exponential_rule_l122_122717


namespace average_mark_second_class_l122_122082

theorem average_mark_second_class
  (avg_mark_class1 : ℝ)
  (num_students_class1 : ℕ)
  (num_students_class2 : ℕ)
  (combined_avg_mark : ℝ) 
  (total_students : ℕ)
  (total_marks_combined : ℝ) :
  avg_mark_class1 * num_students_class1 + x * num_students_class2 = total_marks_combined →
  num_students_class1 + num_students_class2 = total_students →
  combined_avg_mark * total_students = total_marks_combined →
  avg_mark_class1 = 40 →
  num_students_class1 = 30 →
  num_students_class2 = 50 →
  combined_avg_mark = 58.75 →
  total_students = 80 →
  total_marks_combined = 4700 →
  x = 70 :=
by
  intros
  sorry

end average_mark_second_class_l122_122082


namespace max_product_of_sum_300_l122_122435

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l122_122435


namespace no_negatives_l122_122055

theorem no_negatives (x y : ℝ) (h : |x^2 + y^2 - 4*x - 4*y + 5| = |2*x + 2*y - 4|) : 
  ¬ (x < 0) ∧ ¬ (y < 0) :=
by
  sorry

end no_negatives_l122_122055


namespace subtraction_of_fractions_l122_122984

theorem subtraction_of_fractions : (5 / 9) - (1 / 6) = 7 / 18 :=
by
  sorry

end subtraction_of_fractions_l122_122984


namespace greatest_product_of_two_integers_with_sum_300_l122_122519

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l122_122519


namespace xy_identity_l122_122913

theorem xy_identity (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 := by
  sorry

end xy_identity_l122_122913


namespace intersection_in_fourth_quadrant_l122_122796

theorem intersection_in_fourth_quadrant (k : ℝ) :
  (∃ x y : ℝ, y = -2 * x + 3 * k + 14 ∧ x - 4 * y = -3 * k - 2 ∧ x > 0 ∧ y < 0) ↔ (-6 < k) ∧ (k < -2) :=
by
  sorry

end intersection_in_fourth_quadrant_l122_122796


namespace proof_problem_l122_122027

theorem proof_problem (x : ℕ) (h : (x - 4) / 10 = 5) : (x - 5) / 7 = 7 :=
  sorry

end proof_problem_l122_122027


namespace max_product_of_sum_300_l122_122470

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l122_122470


namespace same_speed_is_4_l122_122806

namespace SpeedProof

theorem same_speed_is_4 (x : ℝ) (h_jack_speed : x^2 - 11 * x - 22 = x - 10) (h_jill_speed : x^2 - 5 * x - 60 = (x - 10) * (x + 6)) :
  x = 14 → (x - 10) = 4 :=
by
  sorry

end SpeedProof

end same_speed_is_4_l122_122806


namespace find_a_b_l122_122773

noncomputable def z : ℂ := 1 + Complex.I
noncomputable def lhs (a b : ℝ) := (z^2 + a*z + b) / (z^2 - z + 1)
noncomputable def rhs : ℂ := 1 - Complex.I

theorem find_a_b (a b : ℝ) (h : lhs a b = rhs) : a = -1 ∧ b = 2 :=
  sorry

end find_a_b_l122_122773


namespace sequence_general_term_l122_122303

theorem sequence_general_term {a : ℕ → ℚ} 
  (h₀ : a 1 = 1) 
  (h₁ : ∀ n ≥ 2, a n = 3 * a (n - 1) / (a (n - 1) + 3)) : 
  ∀ n, a n = 3 / (n + 2) :=
by
  sorry

end sequence_general_term_l122_122303


namespace probability_sqrt_lt_9_l122_122563

theorem probability_sqrt_lt_9 : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 →
  ∃ p : ℚ, p = 71 / 90 ∧ 
  ∑ k in (Finset.range 100).filter (λ x, 10 ≤ x ∧ sqrt x < 9), 1 / 90 = p := 
sorry

end probability_sqrt_lt_9_l122_122563


namespace max_product_two_integers_l122_122457

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l122_122457


namespace xyz_abs_eq_one_l122_122044

theorem xyz_abs_eq_one (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) (h4 : x ≠ y) (h5 : y ≠ z) (h6 : z ≠ x)
  (cond : x + 1/y = y + 1/z ∧ y + 1/z = z + 1/x + 1) : |x * y * z| = 1 :=
sorry

end xyz_abs_eq_one_l122_122044


namespace change_in_expression_l122_122879

theorem change_in_expression (x b : ℝ) (hb : 0 < b) : 
    (2 * (x + b) ^ 2 + 5 - (2 * x ^ 2 + 5) = 4 * x * b + 2 * b ^ 2) ∨ 
    (2 * (x - b) ^ 2 + 5 - (2 * x ^ 2 + 5) = -4 * x * b + 2 * b ^ 2) := 
by
    sorry

end change_in_expression_l122_122879


namespace number_of_days_same_l122_122589

-- Defining volumes as given in the conditions.
def volume_project1 : ℕ := 100 * 25 * 30
def volume_project2 : ℕ := 75 * 20 * 50

-- The mathematical statement we want to prove.
theorem number_of_days_same : volume_project1 = volume_project2 → ∀ d : ℕ, d > 0 → d = d :=
by
  sorry

end number_of_days_same_l122_122589


namespace crate_stacking_probability_l122_122848

theorem crate_stacking_probability :
  ∃ (p q : ℕ), (p.gcd q = 1) ∧ (p : ℚ) / q = 170 / 6561 ∧ (total_height = 50) ∧ (number_of_crates = 12) ∧ (orientation_probability = 1 / 3) :=
sorry

end crate_stacking_probability_l122_122848


namespace x_eq_y_sufficient_not_necessary_abs_l122_122268

theorem x_eq_y_sufficient_not_necessary_abs (x y : ℝ) : (x = y → |x| = |y|) ∧ (|x| = |y| → x = y ∨ x = -y) :=
by {
  sorry
}

end x_eq_y_sufficient_not_necessary_abs_l122_122268


namespace perfect_square_append_100_digits_l122_122858

-- Define the number X consisting of 99 nines

def X : ℕ := (10^99 - 1)

theorem perfect_square_append_100_digits :
  ∃ n : ℕ, X * 10^100 ≤ n^2 ∧ n^2 < X * 10^100 + 10^100 :=
by 
  sorry

end perfect_square_append_100_digits_l122_122858


namespace compare_charges_l122_122306

/-
Travel agencies A and B have group discount methods with the original price being $200 per person.
- Agency A: Buy 4 full-price tickets, the rest are half price.
- Agency B: All customers get a 30% discount.
Prove the given relationships based on the number of travelers.
-/

def agency_a_cost (x : ℕ) : ℕ :=
  if 0 < x ∧ x < 4 then 200 * x
  else if x ≥ 4 then 100 * x + 400
  else 0

def agency_b_cost (x : ℕ) : ℕ :=
  140 * x

theorem compare_charges (x : ℕ) :
  (agency_a_cost x < agency_b_cost x -> x > 10) ∧
  (agency_a_cost x = agency_b_cost x -> x = 10) ∧
  (agency_a_cost x > agency_b_cost x -> x < 10) :=
by
  sorry

end compare_charges_l122_122306


namespace general_term_formula_of_a_l122_122645

def S (n : ℕ) : ℚ := (3 / 2) * n^2 - 2 * n

def a (n : ℕ) : ℚ :=
  if n = 1 then (3 / 2) - 2
  else 2 * (3 / 2) * n - (3 / 2) - 2

theorem general_term_formula_of_a :
  ∀ n : ℕ, n > 0 → a n = 3 * n - (7 / 2) :=
by
  intros n hn
  sorry

end general_term_formula_of_a_l122_122645


namespace problem_statement_l122_122174

theorem problem_statement (a b c : ℝ) (ha: 0 ≤ a) (hb: 0 ≤ b) (hc: 0 ≤ c) : 
  a * (a - b) * (a - 2 * b) + b * (b - c) * (b - 2 * c) + c * (c - a) * (c - 2 * a) ≥ 0 :=
by
  sorry

end problem_statement_l122_122174


namespace decreasing_interval_f_l122_122643

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 4 * x - 3

-- Statement to prove that the interval where f is monotonically decreasing is [2, +∞)
theorem decreasing_interval_f : (∀ x₁ x₂ : ℝ, 2 ≤ x₁ ∧ x₁ ≤ x₂ → f x₁ ≥ f x₂) :=
by
  sorry

end decreasing_interval_f_l122_122643


namespace pieces_left_l122_122998

def pieces_initial : ℕ := 900
def pieces_used : ℕ := 156

theorem pieces_left : pieces_initial - pieces_used = 744 := by
  sorry

end pieces_left_l122_122998


namespace smallest_c_for_inverse_l122_122045

def g (x : ℝ) : ℝ := -3 * (x - 1)^2 + 4

theorem smallest_c_for_inverse :
  ∃ c : ℝ, (∀ x y : ℝ, c ≤ x → c ≤ y → g x = g y → x = y) ∧ (∀ d : ℝ, (∀ x y : ℝ, d ≤ x → d ≤ y → g x = g y → x = y) → c ≤ d) :=
sorry

end smallest_c_for_inverse_l122_122045


namespace fraction_of_total_money_spent_on_dinner_l122_122742

-- Definitions based on conditions
def aaron_savings : ℝ := 40
def carson_savings : ℝ := 40
def total_savings : ℝ := aaron_savings + carson_savings

def ice_cream_cost_per_scoop : ℝ := 1.5
def scoops_each : ℕ := 6
def total_ice_cream_cost : ℝ := 2 * scoops_each * ice_cream_cost_per_scoop

def total_left : ℝ := 2

def total_spent : ℝ := total_savings - total_left
def dinner_cost : ℝ := total_spent - total_ice_cream_cost

-- Target statement
theorem fraction_of_total_money_spent_on_dinner : 
  (dinner_cost = 60) ∧ (total_savings = 80) → dinner_cost / total_savings = 3 / 4 :=
by
  intros h
  sorry

end fraction_of_total_money_spent_on_dinner_l122_122742


namespace correct_statement_l122_122633

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + (Real.pi / 2))
noncomputable def g (x : ℝ) : ℝ := Real.cos (x + (3 * Real.pi / 2))

theorem correct_statement (x : ℝ) : f (x - (Real.pi / 2)) = g x :=
by sorry

end correct_statement_l122_122633


namespace perimeter_of_square_l122_122587

theorem perimeter_of_square
  (s : ℝ) -- s is the side length of the square
  (h_divided_rectangles : ∀ r, r ∈ {r : ℝ × ℝ | r = (s, s / 6)} → true) -- the square is divided into six congruent rectangles
  (h_perimeter_rect : 2 * (s + s / 6) = 42) -- the perimeter of each of these rectangles is 42 inches
  : 4 * s = 72 := 
sorry

end perimeter_of_square_l122_122587


namespace max_product_two_integers_sum_300_l122_122341

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l122_122341


namespace min_draw_to_ensure_one_red_l122_122838

theorem min_draw_to_ensure_one_red (b y r : ℕ) (h1 : b + y + r = 20) (h2 : b = y / 6) (h3 : r < y) : 
  ∃ n : ℕ, n = 15 ∧ ∀ d : ℕ, d < 15 → ∀ drawn : Finset (ℕ × ℕ × ℕ), drawn.card = d → ∃ card ∈ drawn, card.2 = r := 
sorry

end min_draw_to_ensure_one_red_l122_122838


namespace probability_of_double_green_given_green_l122_122729

/-- The set of possible cards in the box and their respective sides. -/
def cards : Type :=
  | BlackBlack
  | BlackBlack
  | BlackBlack
  | BlackBlack
  | BlackGreen
  | BlackGreen
  | GreenGreen
  | GreenGreen

/-- Define the sides of each card. -/
def sides (c : cards) : list (bool × bool) :=
  match c with
  | BlackBlack => [(false, false)]
  | BlackGreen => [(false, true), (true, false)]
  | GreenGreen => [(true, true)]

/-- The probability mass function representing a uniform distribution of selecting a card. -/
def card_pmf : pmf cards :=
  pmf.uniform_of_finset {BlackBlack, BlackBlack, BlackBlack, BlackBlack, BlackGreen, BlackGreen, GreenGreen, GreenGreen} 
  (by decide)

/-- Given that one side is green, returning the probability that the other side is also green. -/
def probability_green_given_green : ℚ :=
  let all_sides := cards.enum.toList.bind (λ c, sides c.2)
  let green_sides := all_sides.filter (λ s, s.fst = true ∨ s.snd = true)
  let double_green := green_sides.countp (λ s, s.fst = true ∧ s.snd = true)
  double_green / green_sides.length

theorem probability_of_double_green_given_green :
  probability_green_given_green = 2 / 3 := 
  by 
    sorry

end probability_of_double_green_given_green_l122_122729


namespace largest_number_is_y_l122_122990

def x := 8.1235
def y := 8.12355555555555 -- 8.123\overline{5}
def z := 8.12345454545454 -- 8.123\overline{45}
def w := 8.12345345345345 -- 8.12\overline{345}
def v := 8.12345234523452 -- 8.1\overline{2345}

theorem largest_number_is_y : y > x ∧ y > z ∧ y > w ∧ y > v :=
by
-- Proof steps would go here.
sorry

end largest_number_is_y_l122_122990


namespace savings_account_after_8_weeks_l122_122935

noncomputable def initial_amount : ℕ := 43
noncomputable def weekly_allowance : ℕ := 10
noncomputable def comic_book_cost : ℕ := 3
noncomputable def saved_per_week : ℕ := weekly_allowance - comic_book_cost
noncomputable def weeks : ℕ := 8
noncomputable def savings_in_8_weeks : ℕ := saved_per_week * weeks
noncomputable def total_piggy_bank_after_8_weeks : ℕ := initial_amount + savings_in_8_weeks

theorem savings_account_after_8_weeks : total_piggy_bank_after_8_weeks = 99 :=
by
  have h1 : saved_per_week = 7 := rfl
  have h2 : savings_in_8_weeks = 56 := rfl
  have h3 : total_piggy_bank_after_8_weeks = 99 := rfl
  exact h3

end savings_account_after_8_weeks_l122_122935


namespace max_product_of_sum_300_l122_122466

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l122_122466


namespace spadesuit_calculation_l122_122263

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_calculation : spadesuit 2 (spadesuit 6 1) = -1221 := by
  sorry

end spadesuit_calculation_l122_122263


namespace oldest_child_age_l122_122966

def avg (a b c : ℕ) : ℕ := (a + b + c) / 3

theorem oldest_child_age (a b : ℕ) (h1 : avg a b x = 10) (h2 : a = 8) (h3 : b = 11) : x = 11 :=
by
  sorry

end oldest_child_age_l122_122966


namespace find_polynomials_satisfy_piecewise_l122_122260

def f (x : ℝ) : ℝ := 0
def g (x : ℝ) : ℝ := -x
def h (x : ℝ) : ℝ := -x + 2

theorem find_polynomials_satisfy_piecewise :
  ∀ x : ℝ, abs (f x) - abs (g x) + h x = 
    if x < -1 then -1
    else if x <= 0 then 2
    else -2 * x + 2 :=
by
  sorry

end find_polynomials_satisfy_piecewise_l122_122260


namespace max_product_300_l122_122548

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l122_122548


namespace complex_expression_l122_122948

theorem complex_expression (x y : ℂ) 
  (h : (x^3 + y^3) / (x^3 - y^3) + (x^3 - y^3) / (x^3 + y^3) = 1) :
  (x^9 + y^9) / (x^9 - y^9) + (x^9 - y^9) / (x^9 + y^9) = 3 / 2 :=
by 
  sorry

end complex_expression_l122_122948


namespace greatest_product_sum_300_l122_122366

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l122_122366


namespace total_amount_invested_l122_122929

variable (T : ℝ)

def income_first (T : ℝ) : ℝ :=
  0.10 * (T - 700)

def income_second : ℝ :=
  0.08 * 700

theorem total_amount_invested :
  income_first T - income_second = 74 → T = 2000 :=
by
  intros h
  sorry 

end total_amount_invested_l122_122929


namespace equation_of_line_l_l122_122776

def point (P : ℝ × ℝ) := P = (2, 1)
def parallel (x y : ℝ) : Prop := 2 * x - y + 2 = 0

theorem equation_of_line_l (c : ℝ) (x y : ℝ) :
  (parallel x y ∧ point (x, y)) →
  2 * x - y + c = 0 →
  c = -3 → 2 * x - y - 3 = 0 :=
by
  intro h1 h2 h3
  sorry

end equation_of_line_l_l122_122776


namespace greatest_product_obtainable_l122_122409

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l122_122409


namespace determine_constants_and_sum_l122_122884

theorem determine_constants_and_sum (A B C x : ℝ) (h₁ : A = 3) (h₂ : B = 5) (h₃ : C = 40 / 3)
  (h₄ : (x + B) * (A * x + 40) / ((x + C) * (x + 5)) = 3) :
  ∀ x : ℝ, x ≠ -5 → x ≠ -40 / 3 → (-(5 : ℝ) + -40 / 3 = -55 / 3) :=
sorry

end determine_constants_and_sum_l122_122884


namespace greatest_product_of_sum_eq_300_l122_122383

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l122_122383


namespace sum_of_products_is_70_l122_122335

theorem sum_of_products_is_70 (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 149) (h2 : a + b + c = 17) :
  a * b + b * c + c * a = 70 :=
by
  sorry 

end sum_of_products_is_70_l122_122335


namespace particle_paths_count_l122_122108

-- Definitions for the movement in the Cartesian plane
def valid_moves (a b : ℕ) : List (ℕ × ℕ) := [(a + 2, b), (a, b + 2), (a + 1, b + 1)]

-- The condition to count unique paths from (0,0) to (6,6)
def count_paths (start target : ℕ × ℕ) : ℕ :=
  sorry -- The exact implementation to count paths is omitted here

theorem particle_paths_count :
  count_paths (0, 0) (6, 6) = 58 :=
sorry

end particle_paths_count_l122_122108


namespace complement_intersection_l122_122952

open Set

-- Definitions of the sets U, M, and N
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {3, 4, 5}

-- The theorem we want to prove
theorem complement_intersection :
  (compl M ∩ N) = {4, 5} :=
by
  sorry

end complement_intersection_l122_122952


namespace smallest_non_lucky_multiple_of_8_l122_122130

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky_integer (n : ℕ) : Prop :=
  n % (sum_of_digits n) = 0

theorem smallest_non_lucky_multiple_of_8 : ∃ (m : ℕ), (m > 0) ∧ (m % 8 = 0) ∧ ¬ is_lucky_integer m ∧ m = 16 := sorry

end smallest_non_lucky_multiple_of_8_l122_122130


namespace find_fraction_l122_122100

def number : ℕ := 16

theorem find_fraction (f : ℚ) : f * number + 5 = 13 → f = 1 / 2 :=
by
  sorry

end find_fraction_l122_122100


namespace total_pieces_of_junk_mail_l122_122337

def houses : ℕ := 6
def pieces_per_house : ℕ := 4

theorem total_pieces_of_junk_mail : houses * pieces_per_house = 24 :=
by 
  sorry

end total_pieces_of_junk_mail_l122_122337


namespace exist_positive_abc_with_nonzero_integer_roots_l122_122625

theorem exist_positive_abc_with_nonzero_integer_roots :
  ∃ (a b c : ℤ), 0 < a ∧ 0 < b ∧ 0 < c ∧
  (∀ x y : ℤ, (a * x^2 + b * x + c = 0 → x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℤ, (a * x^2 + b * x - c = 0 → x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℤ, (a * x^2 - b * x + c = 0 → x ≠ 0 ∧ y ≠ 0)) ∧
  (∀ x y : ℤ, (a * x^2 - b * x - c = 0 → x ≠ 0 ∧ y ≠ 0)) :=
sorry

end exist_positive_abc_with_nonzero_integer_roots_l122_122625


namespace howard_items_l122_122656

theorem howard_items (a b c : ℕ) (h1 : a + b + c = 40) (h2 : 40 * a + 300 * b + 400 * c = 5000) : a = 20 :=
by
  sorry

end howard_items_l122_122656


namespace proof_problem_l122_122925

-- Define the problem space
variables (x y : ℝ)

-- Define the conditions
def satisfies_condition (x y : ℝ) : Prop :=
  (0 < x) ∧ (0 < y) ∧ (4 * Real.log x + 2 * Real.log (2 * y) ≥ x^2 + 8 * y - 4)

-- The theorem statement
theorem proof_problem (hx : 0 < x) (hy : 0 < y) (hcond : satisfies_condition x y) :
  x + 2 * y = 1/2 + Real.sqrt 2 :=
sorry

end proof_problem_l122_122925


namespace volunteer_arrangement_l122_122958

/-- The total number of different arrangements of five volunteers at two intersections,
    such that each intersection has at least one volunteer, is 30. -/
theorem volunteer_arrangement :
  let volunteers : Finset (Fin 5) := Finset.univ
  in finset.card ((volunteers.subsets (λ s, 1 ≤ s.card ∧ s.card ≤ 4)).image (λ s, (s, (volunteers \ s)))) = 30 :=
by
  sorry

end volunteer_arrangement_l122_122958


namespace length_AB_is_correct_l122_122180

noncomputable def length_of_AB (x y : ℚ) : ℚ :=
  let a := 3 * x
  let b := 2 * x
  let c := 4 * y
  let d := 5 * y
  let pq_distance := abs (c - a)
  if 5 * x = 9 * y ∧ pq_distance = 3 then 5 * x else 0

theorem length_AB_is_correct : 
  ∃ x y : ℚ, 5 * x = 9 * y ∧ (abs (4 * y - 3 * x)) = 3 ∧ length_of_AB x y = 135 / 7 := 
by
  sorry

end length_AB_is_correct_l122_122180


namespace max_product_of_sum_300_l122_122468

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l122_122468


namespace sum_of_fourth_powers_l122_122770

theorem sum_of_fourth_powers
  (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 2)
  (h3 : a^3 + b^3 + c^3 = 3) :
  a^4 + b^4 + c^4 = 25 / 6 := 
sorry

end sum_of_fourth_powers_l122_122770


namespace quadratic_inequality_solution_is_interval_l122_122829

noncomputable def quadratic_inequality_solution : Set ℝ :=
  { x : ℝ | -3*x^2 + 9*x + 12 > 0 }

theorem quadratic_inequality_solution_is_interval :
  quadratic_inequality_solution = { x : ℝ | -1 < x ∧ x < 4 } :=
sorry

end quadratic_inequality_solution_is_interval_l122_122829


namespace ellipse_focal_distance_correct_l122_122277

noncomputable def ellipse_focal_distance (x y : ℝ) (θ : ℝ) : ℝ :=
  let a := 5 -- semi-major axis
  let b := 2 -- semi-minor axis
  let c := Real.sqrt (a^2 - b^2) -- calculate focal distance
  2 * c -- return 2c

theorem ellipse_focal_distance_correct (θ : ℝ) :
  ellipse_focal_distance (-4 + 2 * Real.cos θ) (1 + 5 * Real.sin θ) θ = 2 * Real.sqrt 21 :=
by
  sorry

end ellipse_focal_distance_correct_l122_122277


namespace minimum_boxes_required_l122_122726

theorem minimum_boxes_required 
  (total_brochures : ℕ)
  (small_box_capacity : ℕ) (small_boxes_available : ℕ)
  (medium_box_capacity : ℕ) (medium_boxes_available : ℕ)
  (large_box_capacity : ℕ) (large_boxes_available : ℕ)
  (complete_fill : ∀ (box_capacity brochures : ℕ), box_capacity ∣ brochures)
  (min_boxes_required : ℕ) :
  total_brochures = 10000 →
  small_box_capacity = 50 →
  small_boxes_available = 40 →
  medium_box_capacity = 200 →
  medium_boxes_available = 25 →
  large_box_capacity = 500 →
  large_boxes_available = 10 →
  min_boxes_required = 35 :=
by
  intros
  sorry

end minimum_boxes_required_l122_122726


namespace number_of_digits_in_x_l122_122637

open Real

theorem number_of_digits_in_x
  (x y : ℝ)
  (hx_pos : 0 < x) (hy_pos : 0 < y)
  (hxy_inequality : x > y)
  (hxy_prod : x * y = 490)
  (hlog_cond : (log x - log 7) * (log y - log 7) = -143/4) :
  ∃ n : ℕ, n = 8 ∧ (10^(n - 1) ≤ x ∧ x < 10^n) :=
by
  sorry

end number_of_digits_in_x_l122_122637


namespace inequality_proof_l122_122288

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a ^ 3 / (a ^ 2 + a * b + b ^ 2)) + (b ^ 3 / (b ^ 2 + b * c + c ^ 2)) + (c ^ 3 / (c ^ 2 + c * a + a ^ 2)) ≥ (a + b + c) / 3 :=
by
  sorry

end inequality_proof_l122_122288


namespace percentage_cut_is_50_l122_122709

-- Conditions
def yearly_subscription_cost : ℝ := 940.0
def reduction_amount : ℝ := 470.0

-- Assertion to be proved
theorem percentage_cut_is_50 :
  (reduction_amount / yearly_subscription_cost) * 100 = 50 :=
by
  sorry

end percentage_cut_is_50_l122_122709


namespace radius_wire_is_4_cm_l122_122862

noncomputable def radius_of_wire_cross_section (r_sphere : ℝ) (length_wire : ℝ) : ℝ :=
  let volume_sphere := (4 / 3) * Real.pi * r_sphere^3
  let volume_wire := volume_sphere / length_wire
  Real.sqrt (volume_wire / Real.pi)

theorem radius_wire_is_4_cm :
  radius_of_wire_cross_section 12 144 = 4 :=
by
  unfold radius_of_wire_cross_section
  sorry

end radius_wire_is_4_cm_l122_122862


namespace mass_percentage_C_in_C6HxO6_indeterminate_l122_122628

-- Definition of conditions
def mass_percentage_C_in_C6H8O6 : ℚ := 40.91 / 100
def molar_mass_C : ℚ := 12.01
def molar_mass_H : ℚ := 1.01
def molar_mass_O : ℚ := 16.00

-- Formula for molar mass of C6H8O6
def molar_mass_C6H8O6 : ℚ := 6 * molar_mass_C + 8 * molar_mass_H + 6 * molar_mass_O

-- Mass of carbon in C6H8O6 is 40.91% of the total molar mass
def mass_of_C_in_C6H8O6 : ℚ := mass_percentage_C_in_C6H8O6 * molar_mass_C6H8O6

-- Hypothesis: mass percentage of carbon in C6H8O6 is given
axiom hyp_mass_percentage_C_in_C6H8O6 : mass_of_C_in_C6H8O6 = 72.06

-- Proof that we need the value of x to determine the mass percentage of C in C6HxO6
theorem mass_percentage_C_in_C6HxO6_indeterminate (x : ℚ) :
  (molar_mass_C6H8O6 = 176.14) → (mass_of_C_in_C6H8O6 = 72.06) → False :=
by
  sorry

end mass_percentage_C_in_C6HxO6_indeterminate_l122_122628


namespace aleena_vs_bob_distance_l122_122328

theorem aleena_vs_bob_distance :
  let AleenaDistance := 75
  let BobDistance := 60
  AleenaDistance - BobDistance = 15 :=
by
  let AleenaDistance := 75
  let BobDistance := 60
  show AleenaDistance - BobDistance = 15
  sorry

end aleena_vs_bob_distance_l122_122328


namespace sally_oscillation_distance_l122_122618

noncomputable def C : ℝ := 5 / 4
noncomputable def D : ℝ := 11 / 4

theorem sally_oscillation_distance :
  abs (C - D) = 3 / 2 :=
by
  sorry

end sally_oscillation_distance_l122_122618


namespace find_f_one_seventh_l122_122899

-- Define the function f
variable (f : ℝ → ℝ)

-- Given conditions
variable (monotonic_f : MonotonicOn f (Set.Ioi 0))
variable (h : ∀ x ∈ Set.Ioi (0 : ℝ), f (f x - 1 / x) = 2)

-- Define the domain
variable (x : ℝ)
variable (hx : x ∈ Set.Ioi (0 : ℝ))

-- The theorem to prove
theorem find_f_one_seventh : f (1 / 7) = 8 := by
  -- proof starts here
  sorry

end find_f_one_seventh_l122_122899


namespace max_product_two_integers_sum_300_l122_122351

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l122_122351


namespace max_product_two_integers_l122_122464

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l122_122464


namespace toothpicks_15_l122_122069

noncomputable def toothpicks : ℕ → ℕ
| 0       => 0  -- since the stage count n >= 1, stage 0 is not required, default 0.
| 1       => 5
| (n + 1) => 2 * toothpicks n + 2

theorem toothpicks_15 : toothpicks 15 = 32766 := by
  sorry

end toothpicks_15_l122_122069


namespace max_product_of_sum_300_l122_122474

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l122_122474


namespace find_y_l122_122599

theorem find_y (y : ℝ) (hy_pos : y > 0) (hy_prop : y^2 / 100 = 9) : y = 30 := by
  sorry

end find_y_l122_122599


namespace solve_for_x_l122_122657

theorem solve_for_x (x : ℝ) (h1 : 3 * x^2 - 5 * x = 0) (h2 : x ≠ 0) : x = 5 / 3 :=
by
  sorry

end solve_for_x_l122_122657


namespace tilde_tilde_tilde_47_l122_122881

def tilde (N : ℝ) : ℝ := 0.4 * N + 2

theorem tilde_tilde_tilde_47 : tilde (tilde (tilde 47)) = 6.128 := 
by
  sorry

end tilde_tilde_tilde_47_l122_122881


namespace circle_center_radius_l122_122128

def circle_equation (x y : ℝ) : Prop := x^2 + 4 * x + y^2 - 6 * y - 12 = 0

theorem circle_center_radius :
  ∃ (h k r : ℝ), (circle_equation (x : ℝ) (y: ℝ) -> (x + h)^2 + (y + k)^2 = r^2) ∧ h = -2 ∧ k = 3 ∧ r = 5 :=
sorry

end circle_center_radius_l122_122128


namespace problem_1_problem_2_l122_122312

def is_in_solution_set (x : ℝ) : Prop := -2 < |x - 1| - |x + 2| ∧ |x - 1| - |x + 2| < 0

variables {a b : ℝ}

theorem problem_1 (ha : is_in_solution_set a) (hb : is_in_solution_set b) :
  |(1 / 3) * a + (1 / 6) * b| < 1 / 4 :=
sorry

theorem problem_2 (ha : is_in_solution_set a) (hb : is_in_solution_set b) :
  |1 - 4 * a * b| > 2 * |a - b| :=
sorry

end problem_1_problem_2_l122_122312


namespace Cooper_age_l122_122845

variable (X : ℕ)
variable (Dante : ℕ)
variable (Maria : ℕ)

theorem Cooper_age (h1 : Dante = 2 * X) (h2 : Maria = 2 * X + 1) (h3 : X + Dante + Maria = 31) : X = 6 :=
by
  -- Proof is omitted as indicated
  sorry

end Cooper_age_l122_122845


namespace greatest_int_less_than_M_div_100_l122_122006

theorem greatest_int_less_than_M_div_100 (M : ℕ) : 
  (fraction_sum M -> 
   let k := M / 100 in 
   k = 5242) :=
by
  intro h
  sorry

where fraction_sum (M : ℕ) : Prop :=
  M = (Nat.factorial 20) * (
    (1 / (fact 1 * fact 19)) +
    (1 / (fact 2 * fact 18)) +
    (1 / (fact 3 * fact 17)) +
    (1 / (fact 4 * fact 16)) +
    (1 / (fact 5 * fact 15)) +
    (1 / (fact 6 * fact 14)) +
    (1 / (fact 7 * fact 13)) +
    (1 / (fact 8 * fact 12)) +
    (1 / (fact 9 * fact 11)) +
    (1 / (fact 10 * fact 10)))

end greatest_int_less_than_M_div_100_l122_122006


namespace minimum_value_problem_l122_122818

open Real

theorem minimum_value_problem (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h_sum : x + y + z = 6) :
  9 / x + 16 / y + 25 / z ≥ 24 :=
by
  sorry

end minimum_value_problem_l122_122818


namespace greatest_product_l122_122532

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l122_122532


namespace incircle_tangent_to_adc_sides_l122_122003

noncomputable def Triangle (A B C : Point) : Prop := -- defining Triangle for context
  True

noncomputable def CircleTangentToSidesAndExtensions (circle : Circle) (AC BA BC : Line) : Prop := -- tangent condition
  True

noncomputable def Parallelogram (A B C D : Point) : Prop := -- defining Parallelogram for context
  True

theorem incircle_tangent_to_adc_sides 
  (A B C D P S Q R : Point)
  (AC BA BC DA DC : Line)
  (circle : Circle) 
  (h_parallelogram : Parallelogram A B C D)
  (h_tangent : CircleTangentToSidesAndExtensions circle AC BA BC)
  (h_intersection : LineIntersectsSegmentsInPoints (line_through P S) DA DC Q R) :
  TangentToIncircleAtPoints (Triangle C D A) (incircle (Triangle C D A)) Q R :=
by
  sorry

end incircle_tangent_to_adc_sides_l122_122003


namespace probability_sqrt_less_nine_l122_122551

theorem probability_sqrt_less_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  ∃ (p : ℚ), p = (finset.card (valid_numbers.to_finset) : ℚ) / (finset.card (two_digit_numbers.to_finset) : ℚ) ∧ p = 71 / 90 :=
by
  sorry

end probability_sqrt_less_nine_l122_122551


namespace time_display_unique_digits_l122_122246

theorem time_display_unique_digits : 
  ∃ n : ℕ, n = 840 ∧ ∀ h : Fin 10, h = 5 →
  5 * 7 * 4 * 6 = n :=
by
  use 840
  simp
  sorry

end time_display_unique_digits_l122_122246


namespace distance_of_parallel_lines_l122_122005

noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  abs (C₂ - C₁) / real.sqrt (A^2 + B^2)

theorem distance_of_parallel_lines (a : ℝ)
  (h_parallel : (a ≠ -1) ∧ (a ≠ 0) ∧ ((-1 : ℝ) / a = (a - 1) / 2)) :
  distance_between_parallel_lines 1 2 (1 : ℝ) (4 : ℝ) = 3 * real.sqrt 5 / 5 :=
by
  sorry

end distance_of_parallel_lines_l122_122005


namespace cubic_inequality_solution_l122_122264

theorem cubic_inequality_solution (x : ℝ) : x^3 - 12 * x^2 + 27 * x > 0 ↔ (0 < x ∧ x < 3) ∨ (9 < x) :=
by sorry

end cubic_inequality_solution_l122_122264


namespace averagePrice_is_20_l122_122695

-- Define the conditions
def books1 : Nat := 32
def cost1 : Nat := 1500

def books2 : Nat := 60
def cost2 : Nat := 340

-- Define the total books and total cost
def totalBooks : Nat := books1 + books2
def totalCost : Nat := cost1 + cost2

-- Define the average price calculation
def averagePrice : Nat := totalCost / totalBooks

-- The statement to prove
theorem averagePrice_is_20 : averagePrice = 20 := by
  -- Sorry is used here as a placeholder for the actual proof.
  sorry

end averagePrice_is_20_l122_122695


namespace carl_owes_15300_l122_122118

def total_property_damage : ℝ := 40000
def total_medical_bills : ℝ := 70000
def insurance_coverage_property_damage : ℝ := 0.80
def insurance_coverage_medical_bills : ℝ := 0.75
def carl_responsibility : ℝ := 0.60

def carl_personally_owes : ℝ :=
  let insurance_paid_property_damage := insurance_coverage_property_damage * total_property_damage
  let insurance_paid_medical_bills := insurance_coverage_medical_bills * total_medical_bills
  let remaining_property_damage := total_property_damage - insurance_paid_property_damage
  let remaining_medical_bills := total_medical_bills - insurance_paid_medical_bills
  let carl_share_property_damage := carl_responsibility * remaining_property_damage
  let carl_share_medical_bills := carl_responsibility * remaining_medical_bills
  carl_share_property_damage + carl_share_medical_bills

theorem carl_owes_15300 :
  carl_personally_owes = 15300 := by
  sorry

end carl_owes_15300_l122_122118


namespace greatest_product_sum_300_l122_122369

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l122_122369


namespace tangent_line_parallel_l122_122067

theorem tangent_line_parallel (x y : ℝ) (h_parab : y = 2 * x^2) (h_parallel : ∃ (m b : ℝ), 4 * x - y + b = 0) : 
    (∃ b, 4 * x - y - b = 0) := 
by
  sorry

end tangent_line_parallel_l122_122067


namespace geometric_extraction_from_arithmetic_l122_122315

theorem geometric_extraction_from_arithmetic (a b : ℤ) :
  ∃ k : ℕ → ℤ, (∀ n : ℕ, k n = a * (b + 1) ^ n) ∧ (∀ n : ℕ, ∃ m : ℕ, k n = a + b * m) :=
by sorry

end geometric_extraction_from_arithmetic_l122_122315


namespace solve_k_equality_l122_122906

noncomputable def collinear_vectors (e1 e2 : ℝ) (k : ℝ) (AB CB CD : ℝ) : Prop := 
  let BD := (2 * e1 - e2) - (e1 + 3 * e2)
  BD = e1 - 4 * e2 ∧ AB = 2 * e1 + k * e2 ∧ AB = k * BD
  
theorem solve_k_equality (e1 e2 k AB CB CD : ℝ) (h_non_collinear : (e1 ≠ 0 ∨ e2 ≠ 0)) :
  collinear_vectors e1 e2 k AB CB CD → k = -8 :=
by
  intro h_collinear
  sorry

end solve_k_equality_l122_122906


namespace greatest_product_sum_300_l122_122381

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l122_122381


namespace greatest_product_two_ints_sum_300_l122_122449

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l122_122449


namespace sufficient_but_not_necessary_l122_122586

variable (p q : Prop)

theorem sufficient_but_not_necessary (h : p ∧ q) : ¬¬p :=
  by sorry -- Proof not required

end sufficient_but_not_necessary_l122_122586


namespace roger_bike_rides_total_l122_122187

theorem roger_bike_rides_total 
  (r1 : ℕ) (h1 : r1 = 2) 
  (r2 : ℕ) (h2 : r2 = 5 * r1) 
  (r : ℕ) (h : r = r1 + r2) : 
  r = 12 := 
by
  sorry

end roger_bike_rides_total_l122_122187


namespace sub_fraction_l122_122988

theorem sub_fraction (a b c d : ℚ) (h1 : a = 5) (h2 : b = 9) (h3 : c = 1) (h4 : d = 6) : (a / b) - (c / d) = 7 / 18 := 
by
  sorry

end sub_fraction_l122_122988


namespace greatest_product_sum_300_l122_122361

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l122_122361


namespace maisy_new_job_hours_l122_122953

-- Define the conditions
def current_job_earnings : ℚ := 80
def new_job_wage_per_hour : ℚ := 15
def new_job_bonus : ℚ := 35
def earnings_difference : ℚ := 15

-- Define the problem
theorem maisy_new_job_hours (h : ℚ) 
  (h1 : current_job_earnings = 80) 
  (h2 : new_job_wage_per_hour * h + new_job_bonus = current_job_earnings + earnings_difference) :
  h = 4 :=
  sorry

end maisy_new_job_hours_l122_122953


namespace compute_volume_of_cube_l122_122094

-- Define the conditions and required properties
variable (s V : ℝ)

-- Given condition: the surface area of the cube is 384 sq cm
def surface_area (s : ℝ) : Prop := 6 * s^2 = 384

-- Define the volume of the cube
def volume (s : ℝ) (V : ℝ) : Prop := V = s^3

-- Theorem statement to prove the volume is correctly computed
theorem compute_volume_of_cube (h₁ : surface_area s) : volume s 512 :=
  sorry

end compute_volume_of_cube_l122_122094


namespace speed_boat_upstream_l122_122588

-- Define the conditions provided in the problem
def V_b : ℝ := 8.5  -- Speed of the boat in still water (in km/hr)
def V_downstream : ℝ := 13 -- Speed of the boat downstream (in km/hr)
def V_s : ℝ := V_downstream - V_b  -- Speed of the stream (in km/hr), derived from V_downstream and V_b
def V_upstream (V_b : ℝ) (V_s : ℝ) : ℝ := V_b - V_s  -- Speed of the boat upstream (in km/hr)

-- Statement to prove: the speed of the boat upstream is 4 km/hr
theorem speed_boat_upstream :
  V_upstream V_b V_s = 4 :=
by
  -- This line is for illustration, replace with an actual proof
  sorry

end speed_boat_upstream_l122_122588


namespace percentage_of_first_to_second_l122_122230

theorem percentage_of_first_to_second (X : ℝ) (first second : ℝ) (h1 : first = (7 / 100) * X) (h2 : second = (14 / 100) * X) : 
(first / second) * 100 = 50 := by
  sorry

end percentage_of_first_to_second_l122_122230


namespace greatest_product_two_ints_sum_300_l122_122448

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l122_122448


namespace max_dinners_for_7_people_max_dinners_for_8_people_l122_122058

def max_dinners_with_new_neighbors (n : ℕ) : ℕ :=
  if n = 7 ∨ n = 8 then 3 else 0

theorem max_dinners_for_7_people : max_dinners_with_new_neighbors 7 = 3 := sorry

theorem max_dinners_for_8_people : max_dinners_with_new_neighbors 8 = 3 := sorry

end max_dinners_for_7_people_max_dinners_for_8_people_l122_122058


namespace point_inside_circle_l122_122797

theorem point_inside_circle (r OP : ℝ) (h₁ : r = 3) (h₂ : OP = 2) : OP < r :=
by
  sorry

end point_inside_circle_l122_122797


namespace total_juice_boxes_needed_l122_122184

-- Definitions for the conditions
def john_juice_per_week : Nat := 2 * 5
def john_school_weeks : Nat := 18 - 2 -- taking into account the holiday break

def samantha_juice_per_week : Nat := 1 * 5
def samantha_school_weeks : Nat := 16 - 2 -- taking into account after-school and holiday break

def heather_mon_wed_juice : Nat := 3 * 2
def heather_tue_thu_juice : Nat := 2 * 2
def heather_fri_juice : Nat := 1
def heather_juice_per_week : Nat := heather_mon_wed_juice + heather_tue_thu_juice + heather_fri_juice
def heather_school_weeks : Nat := 17 - 2 -- taking into account personal break and holiday break

-- Question and Answer in lean
theorem total_juice_boxes_needed : 
  (john_juice_per_week * john_school_weeks) + 
  (samantha_juice_per_week * samantha_school_weeks) + 
  (heather_juice_per_week * heather_school_weeks) = 395 := 
by
  sorry

end total_juice_boxes_needed_l122_122184


namespace complex_number_sum_zero_l122_122011

theorem complex_number_sum_zero (a b : ℝ) (i : ℂ) (h : a + b * i = 1 - i) : a + b = 0 := 
by sorry

end complex_number_sum_zero_l122_122011


namespace gcf_84_112_210_l122_122981

theorem gcf_84_112_210 : gcd (gcd 84 112) 210 = 14 := by sorry

end gcf_84_112_210_l122_122981


namespace max_product_two_integers_sum_300_l122_122353

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l122_122353


namespace max_product_300_l122_122541

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l122_122541


namespace candy_bar_cost_l122_122208

theorem candy_bar_cost (initial_amount change : ℕ) (h : initial_amount = 50) (hc : change = 5) : 
  initial_amount - change = 45 :=
by
  -- sorry is used to skip the proof
  sorry

end candy_bar_cost_l122_122208


namespace max_product_sum_300_l122_122484

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l122_122484


namespace minimum_fraction_l122_122269

theorem minimum_fraction (m n : ℕ) (hm : m > 0) (hn : n > 0) (h : m + 2 * n = 8) : 2 / m + 1 / n = 1 :=
by
  sorry

end minimum_fraction_l122_122269


namespace margaret_time_is_10_minutes_l122_122872

variable (time_billy_first_5_laps : ℕ)
variable (time_billy_next_3_laps : ℕ)
variable (time_billy_next_lap : ℕ)
variable (time_billy_final_lap : ℕ)
variable (time_difference : ℕ)

def billy_total_time := time_billy_first_5_laps + time_billy_next_3_laps + time_billy_next_lap + time_billy_final_lap

def margaret_total_time := billy_total_time + time_difference

theorem margaret_time_is_10_minutes :
  time_billy_first_5_laps = 120 ∧
  time_billy_next_3_laps = 240 ∧
  time_billy_next_lap = 60 ∧
  time_billy_final_lap = 150 ∧
  time_difference = 30 →
  margaret_total_time = 600 :=
by 
  sorry

end margaret_time_is_10_minutes_l122_122872


namespace volume_of_regular_quadrilateral_pyramid_l122_122110

noncomputable def volume_of_pyramid (a : ℝ) : ℝ :=
  let x := 1 -- A placeholder to outline the structure
  let PM := (6 * a) / 5
  let V := (2 * a^3) / 5
  V

theorem volume_of_regular_quadrilateral_pyramid
  (a PM : ℝ)
  (h1 : PM = (6 * a) / 5)
  [InstReal : Nonempty (Real)] :
  volume_of_pyramid a = (2 * a^3) / 5 :=
by
  sorry

end volume_of_regular_quadrilateral_pyramid_l122_122110


namespace circle_center_l122_122669

theorem circle_center (n : ℝ) (r : ℝ) (h1 : r = 7) (h2 : ∀ x : ℝ, x^2 + (x^2 - n)^2 = 49 → x^4 - x^2 * (2*n - 1) + n^2 - 49 = 0)
  (h3 : ∃! y : ℝ, y^2 + (1 - 2*n) * y + n^2 - 49 = 0) :
  (0, n) = (0, 197 / 4) := 
sorry

end circle_center_l122_122669


namespace find_other_number_l122_122980

theorem find_other_number (m n : ℕ) (H1 : n = 26) 
  (H2 : Nat.lcm n m = 52) (H3 : Nat.gcd n m = 8) : m = 16 := by
  sorry

end find_other_number_l122_122980


namespace floor_T_equals_150_l122_122179

variable {p q r s : ℝ}

theorem floor_T_equals_150
  (hpq_sum_of_squares : p^2 + q^2 = 2500)
  (hrs_sum_of_squares : r^2 + s^2 = 2500)
  (hpq_product : p * q = 1225)
  (hrs_product : r * s = 1225)
  (hp_plus_s : p + s = 75) :
  ∃ T : ℝ, T = p + q + r + s ∧ ⌊T⌋ = 150 :=
by
  sorry

end floor_T_equals_150_l122_122179


namespace range_of_a_l122_122685

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x > 2 then 2^x + a else x + a^2

theorem range_of_a (a : ℝ) : (∀ y : ℝ, ∃ x : ℝ, f x a = y) ↔ (a ≤ -1 ∨ a ≥ 2) :=
by
  sorry

end range_of_a_l122_122685


namespace probability_of_sqrt_lt_9_l122_122575

-- Define the set of two-digit whole numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the subset of numbers for which the square root is less than 9
def valid_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 80}

-- Define the probability calculation
noncomputable def probability_sqrt_lt_9 := (valid_numbers.to_finset.card : ℝ) / (two_digit_numbers.to_finset.card : ℝ)

-- The statement we aim to prove
theorem probability_of_sqrt_lt_9 : probability_sqrt_lt_9 = 71 / 90 := 
sorry

end probability_of_sqrt_lt_9_l122_122575


namespace max_value_exponential_and_power_functions_l122_122640

variable (a b : ℝ)

-- Given conditions
axiom condition : 0 < b ∧ b < a ∧ a < 1

-- Problem statement
theorem max_value_exponential_and_power_functions : 
  a^b = max (max (a^b) (b^a)) (max (a^a) (b^b)) :=
by
  sorry

end max_value_exponential_and_power_functions_l122_122640


namespace cooper_age_l122_122843

variable (Cooper Dante Maria : ℕ)

-- Conditions
def sum_of_ages : Prop := Cooper + Dante + Maria = 31
def dante_twice_cooper : Prop := Dante = 2 * Cooper
def maria_one_year_older : Prop := Maria = Dante + 1

theorem cooper_age (h1 : sum_of_ages Cooper Dante Maria) (h2 : dante_twice_cooper Cooper Dante) (h3 : maria_one_year_older Dante Maria) : Cooper = 6 :=
by
  sorry

end cooper_age_l122_122843


namespace greatest_product_l122_122529

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l122_122529


namespace minimum_value_problem_l122_122178

theorem minimum_value_problem (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 + 4 * x^2 + 2 * x + 1) * (y^3 + 4 * y^2 + 2 * y + 1) * (z^3 + 4 * z^2 + 2 * z + 1) / (x * y * z) ≥ 1331 :=
sorry

end minimum_value_problem_l122_122178


namespace part_I_part_II_l122_122279

-- Define the function f(x)
def f (x : ℝ) (m : ℝ) := abs (x + m) + abs (2 * x - 1)

-- Part (I)
theorem part_I (x : ℝ) : (f x (-1) ≤ 2) ↔ (0 ≤ x ∧ x ≤ (4 / 3)) :=
by sorry

-- Part (II)
theorem part_II (m : ℝ) : (∀ x, (3 / 4) ≤ x ∧ x ≤ 2 → f x m ≤ abs (2 * x + 1)) ↔ (-11 / 4) ≤ m ∧ m ≤ 0 :=
by sorry

end part_I_part_II_l122_122279


namespace value_of_p10_l122_122612

def p (d e f x : ℝ) : ℝ := d * x^2 + e * x + f

theorem value_of_p10 (d e f : ℝ) 
  (h1 : p d e f 3 = p d e f 4)
  (h2 : p d e f 2 = p d e f 5)
  (h3 : p d e f 0 = 2) :
  p d e f 10 = 2 :=
by
  sorry

end value_of_p10_l122_122612


namespace additional_savings_l122_122706

def initial_price : Float := 30
def discount1 : Float := 5
def discount2_percent : Float := 0.25

def price_after_discount1_then_discount2 : Float := 
  (initial_price - discount1) * (1 - discount2_percent)

def price_after_discount2_then_discount1 : Float := 
  initial_price * (1 - discount2_percent) - discount1

theorem additional_savings :
  price_after_discount1_then_discount2 - price_after_discount2_then_discount1 = 1.25 := by
  sorry

end additional_savings_l122_122706


namespace actual_number_of_children_l122_122722

-- Define the conditions of the problem
def condition1 (C B : ℕ) : Prop := B = 2 * C
def condition2 : ℕ := 320
def condition3 (C B : ℕ) : Prop := B = 4 * (C - condition2)

-- Define the statement to be proved
theorem actual_number_of_children (C B : ℕ) 
  (h1 : condition1 C B) (h2 : condition3 C B) : C = 640 :=
by 
  -- Proof will be added here
  sorry

end actual_number_of_children_l122_122722


namespace french_fries_cost_is_10_l122_122764

-- Define the costs as given in the problem conditions
def taco_salad_cost : ℕ := 10
def daves_single_cost : ℕ := 5
def peach_lemonade_cost : ℕ := 2
def num_friends : ℕ := 5
def friend_payment : ℕ := 11

-- Define the total amount collected from friends
def total_collected : ℕ := num_friends * friend_payment

-- Define the subtotal for the known items
def subtotal : ℕ := taco_salad_cost + (num_friends * daves_single_cost) + (num_friends * peach_lemonade_cost)

-- The total cost of french fries
def total_french_fries_cost := total_collected - subtotal

-- The proof statement:
theorem french_fries_cost_is_10 : total_french_fries_cost = 10 := by
  sorry

end french_fries_cost_is_10_l122_122764


namespace greatest_product_of_two_integers_with_sum_300_l122_122512

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l122_122512


namespace find_m_l122_122020

-- Define the functions f and g
def f (x m : ℝ) := x^2 - 2 * x + m
def g (x m : ℝ) := x^2 - 3 * x + 5 * m

-- The condition to be proved
theorem find_m (m : ℝ) : 3 * f 4 m = g 4 m → m = 10 :=
by
  sorry

end find_m_l122_122020


namespace percentage_increase_in_sales_l122_122579

theorem percentage_increase_in_sales (P S : ℝ) (hP : P > 0) (hS : S > 0) :
  (∃ X : ℝ, (0.8 * (1 + X / 100) = 1.44) ∧ X = 80) :=
sorry

end percentage_increase_in_sales_l122_122579


namespace geometric_series_sum_l122_122040

theorem geometric_series_sum 
  (a : ℝ) (r : ℝ) (s : ℝ)
  (h_a : a = 9)
  (h_r : r = -2/3)
  (h_abs_r : |r| < 1)
  (h_s : s = a / (1 - r)) : 
  s = 5.4 := by
  sorry

end geometric_series_sum_l122_122040


namespace integer_roots_sum_abs_eq_94_l122_122132

theorem integer_roots_sum_abs_eq_94 {a b c m : ℤ} :
  (∃ m, (x : ℤ) * (x : ℤ) * (x : ℤ) - 2013 * (x : ℤ) + m = 0 ∧ a + b + c = 0 ∧ ab + bc + ac = -2013) →
  |a| + |b| + |c| = 94 :=
sorry

end integer_roots_sum_abs_eq_94_l122_122132


namespace number_of_keepers_l122_122091

theorem number_of_keepers (hens goats camels : ℕ) (keepers feet heads : ℕ)
  (h_hens : hens = 50)
  (h_goats : goats = 45)
  (h_camels : camels = 8)
  (h_equation : (2 * hens + 4 * goats + 4 * camels + 2 * keepers) = (hens + goats + camels + keepers + 224))
  : keepers = 15 :=
by
sorry

end number_of_keepers_l122_122091


namespace number_of_jerseys_sold_l122_122194

-- Definitions based on conditions
def revenue_per_jersey : ℕ := 115
def revenue_per_tshirt : ℕ := 25
def tshirts_sold : ℕ := 113
def jersey_cost_difference : ℕ := 90

-- Main condition: Prove the number of jerseys sold is 113
theorem number_of_jerseys_sold : ∀ (J : ℕ), 
  (revenue_per_jersey = revenue_per_tshirt + jersey_cost_difference) →
  (J * revenue_per_jersey = tshirts_sold * revenue_per_tshirt) →
  J = 113 :=
by
  intros J h1 h2
  sorry

end number_of_jerseys_sold_l122_122194


namespace probability_sqrt_lt_9_of_two_digit_l122_122560

-- Define the set of two-digit whole numbers
def two_digit_whole_numbers : set ℕ := {n | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate that checks if the square root of a number is less than 9
def sqrt_lt_9 (n : ℕ) : Prop := (n : ℝ)^2 < (9 : ℝ)^2

-- Calculate the probability
theorem probability_sqrt_lt_9_of_two_digit :
  let eligible_numbers := { n ∈ two_digit_whole_numbers | sqrt_lt_9 n } in
  (eligible_numbers.to_finset.card : ℚ) / (two_digit_whole_numbers.to_finset.card : ℚ) =
  71 / 90 :=
by
  sorry

end probability_sqrt_lt_9_of_two_digit_l122_122560


namespace rock_paper_scissors_score_divisible_by_3_l122_122847

theorem rock_paper_scissors_score_divisible_by_3 
  (R : ℕ) 
  (rock_shown : ℕ) 
  (scissors_shown : ℕ) 
  (paper_shown : ℕ)
  (points : ℕ)
  (h_equal_shows : 3 * ((rock_shown + scissors_shown + paper_shown) / 3) = rock_shown + scissors_shown + paper_shown)
  (h_points_awarded : ∀ (r s p : ℕ), r + s + p = 3 → (r = 2 ∧ s = 1 ∧ p = 0) ∨ (r = 0 ∧ s = 2 ∧ p = 1) ∨ (r = 1 ∧ s = 0 ∧ p = 2) → points % 3 = 0) :
  points % 3 = 0 := 
sorry

end rock_paper_scissors_score_divisible_by_3_l122_122847


namespace solve_inequality_l122_122259

open Set

theorem solve_inequality :
  { x : ℝ | (2 * x - 2) / (x^2 - 5*x + 6) ≤ 3 } = Ioo (5/3) 2 ∪ Icc 3 4 :=
by
  sorry

end solve_inequality_l122_122259


namespace max_product_two_integers_sum_300_l122_122352

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l122_122352


namespace socks_knitted_total_l122_122810

def total_socks_knitted (nephew: ℕ) (first_week: ℕ) (second_week: ℕ) (third_week: ℕ) (fourth_week: ℕ) : ℕ := 
  nephew + first_week + second_week + third_week + fourth_week

theorem socks_knitted_total : 
  ∀ (nephew first_week second_week third_week fourth_week : ℕ),
  nephew = 4 → 
  first_week = 12 → 
  second_week = first_week + 4 → 
  third_week = (first_week + second_week) / 2 → 
  fourth_week = third_week - 3 → 
  total_socks_knitted nephew first_week second_week third_week fourth_week = 57 := 
by 
  intros nephew first_week second_week third_week fourth_week 
  intros Hnephew Hfirst_week Hsecond_week Hthird_week Hfourth_week 
  rw [Hnephew, Hfirst_week] 
  have h1: second_week = 16 := by rw [Hfirst_week, Hsecond_week]
  have h2: third_week = 14 := by rw [Hfirst_week, h1, Hthird_week]
  have h3: fourth_week = 11 := by rw [h2, Hfourth_week]
  rw [Hnephew, Hfirst_week, h1, h2, h3]
  exact rfl

end socks_knitted_total_l122_122810


namespace first_trial_addition_amounts_l122_122644

-- Define the range and conditions for the biological agent addition amount.
def lower_bound : ℝ := 20
def upper_bound : ℝ := 30
def golden_ratio_method : ℝ := 0.618
def first_trial_addition_amount_1 : ℝ := lower_bound + (upper_bound - lower_bound) * golden_ratio_method
def first_trial_addition_amount_2 : ℝ := upper_bound - (upper_bound - lower_bound) * golden_ratio_method

-- Prove that the possible addition amounts for the first trial are 26.18g or 23.82g.
theorem first_trial_addition_amounts :
  (first_trial_addition_amount_1 = 26.18 ∨ first_trial_addition_amount_2 = 23.82) :=
by
  -- Placeholder for the proof.
  sorry

end first_trial_addition_amounts_l122_122644


namespace no_such_functions_exist_l122_122754

open Function

theorem no_such_functions_exist : ¬ (∃ (f g : ℝ → ℝ), ∀ x : ℝ, f (g x) = x^2 ∧ g (f x) = x^3) := 
sorry

end no_such_functions_exist_l122_122754


namespace find_rth_term_l122_122766

def S (n : ℕ) : ℕ := 2 * n + 3 * (n^3)

def a (r : ℕ) : ℕ := S r - S (r - 1)

theorem find_rth_term (r : ℕ) : a r = 9 * r^2 - 9 * r + 5 := by
  sorry

end find_rth_term_l122_122766


namespace greatest_product_sum_300_l122_122373

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l122_122373


namespace solve_graph_equation_l122_122828

/- Problem:
Solve for the graph of the equation x^2(x+y+2)=y^2(x+y+2)
Given condition: equation x^2(x+y+2)=y^2(x+y+2)
Conclusion: Three lines that do not all pass through a common point
The final answer should be formally proven.
-/

theorem solve_graph_equation (x y : ℝ) :
  (x^2 * (x + y + 2) = y^2 * (x + y + 2)) →
  (∃ a b c d : ℝ,  (a = -x - 2 ∧ b = -x ∧ c = x ∧ (a ≠ b ∧ a ≠ c ∧ b ≠ c)) ∧
   (d = 0) ∧ ¬ ∀ p q r : ℝ, p = q ∧ q = r ∧ r = p) :=
by
  sorry

end solve_graph_equation_l122_122828


namespace find_OH_squared_l122_122036

variables (A B C : ℝ) (a b c R OH : ℝ)

-- Conditions
def circumcenter (O : ℝ) := true  -- Placeholder, as the actual definition relies on geometric properties
def orthocenter (H : ℝ) := true   -- Placeholder, as the actual definition relies on geometric properties

axiom eqR : R = 5
axiom sumSquares : a^2 + b^2 + c^2 = 50

-- Problem statement
theorem find_OH_squared : OH^2 = 175 :=
by
  sorry

end find_OH_squared_l122_122036


namespace jiwon_distance_to_school_l122_122059

theorem jiwon_distance_to_school
  (taehong_distance_meters jiwon_distance_meters : ℝ)
  (taehong_distance_km : ℝ := 1.05)
  (h1 : taehong_distance_meters = jiwon_distance_meters + 460)
  (h2 : taehong_distance_meters = taehong_distance_km * 1000) :
  jiwon_distance_meters / 1000 = 0.59 := 
sorry

end jiwon_distance_to_school_l122_122059


namespace probability_sqrt_less_than_nine_l122_122572

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sqrt_less_than_nine (n : ℕ) : Prop := n < 81

theorem probability_sqrt_less_than_nine :
  (∃ total_num favorable_num,
    (total_num = Finset.card (Finset.filter is_two_digit (Finset.range 100)) ∧
     favorable_num = Finset.card (Finset.filter (λ n, is_two_digit n ∧ sqrt_less_than_nine n) (Finset.range 100)) ∧
     (favorable_num : ℚ) / total_num = 71 / 90)) :=
by
  sorry

end probability_sqrt_less_than_nine_l122_122572


namespace add_decimal_l122_122852

theorem add_decimal (a b : ℝ) (h1 : a = 0.35) (h2 : b = 124.75) : a + b = 125.10 :=
by sorry

end add_decimal_l122_122852


namespace factorization_correct_l122_122759

theorem factorization_correct (x : ℝ) : 2 * x^2 - 4 * x = 2 * x * (x - 2) :=
by
  sorry

end factorization_correct_l122_122759


namespace point_P_in_first_quadrant_l122_122676

def pointInFirstQuadrant (x y : Int) : Prop := x > 0 ∧ y > 0

theorem point_P_in_first_quadrant : pointInFirstQuadrant 2 3 :=
by
  sorry

end point_P_in_first_quadrant_l122_122676


namespace value_of_x2_plus_y2_l122_122907

theorem value_of_x2_plus_y2 (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 :=
by
  sorry

end value_of_x2_plus_y2_l122_122907


namespace greatest_product_two_ints_sum_300_l122_122440

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l122_122440


namespace Cooper_age_l122_122844

variable (X : ℕ)
variable (Dante : ℕ)
variable (Maria : ℕ)

theorem Cooper_age (h1 : Dante = 2 * X) (h2 : Maria = 2 * X + 1) (h3 : X + Dante + Maria = 31) : X = 6 :=
by
  -- Proof is omitted as indicated
  sorry

end Cooper_age_l122_122844


namespace solve_inequality_system_l122_122830

theorem solve_inequality_system (x : ℝ) :
  (x + 2 < 3 * x) ∧ ((5 - x) / 2 + 1 < 0) → (x > 7) :=
by
  sorry

end solve_inequality_system_l122_122830


namespace sum_of_squares_l122_122917

theorem sum_of_squares (x y : ℝ) (h1 : (x + y) ^ 2 = 4) (h2 : x * y = -1) :
  x^2 + y^2 = 6 :=
by
  sorry

end sum_of_squares_l122_122917


namespace two_times_difference_eq_20_l122_122203

theorem two_times_difference_eq_20 (x y : ℕ) (hx : x = 30) (hy : y = 20) (hsum : x + y = 50) : 2 * (x - y) = 20 := by
  sorry

end two_times_difference_eq_20_l122_122203


namespace length_of_bridge_correct_l122_122107

noncomputable def length_of_bridge (speed_kmh : ℝ) (time_min : ℝ) : ℝ :=
  let speed_mpm := (speed_kmh * 1000) / 60  -- Convert speed from km/hr to m/min
  speed_mpm * time_min  -- Length of the bridge in meters

theorem length_of_bridge_correct :
  length_of_bridge 10 10 = 1666.7 :=
by
  sorry

end length_of_bridge_correct_l122_122107


namespace intersection_eq_l122_122136

open Set

variable (A B : Set ℝ)

def setA : A = {x | -3 < x ∧ x < 2} := sorry

def setB : B = {x | x^2 + 4*x - 5 ≤ 0} := sorry

theorem intersection_eq : A ∩ B = {x | -3 < x ∧ x ≤ 1} :=
sorry

end intersection_eq_l122_122136


namespace max_product_of_two_integers_whose_sum_is_300_l122_122505

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l122_122505


namespace even_num_students_count_l122_122088

-- Define the number of students in each school
def num_students_A : Nat := 786
def num_students_B : Nat := 777
def num_students_C : Nat := 762
def num_students_D : Nat := 819
def num_students_E : Nat := 493

-- Define a predicate to check if a number is even
def is_even (n : Nat) : Prop := n % 2 = 0

-- The theorem to state the problem
theorem even_num_students_count :
  (is_even num_students_A ∧ is_even num_students_C) ∧ ¬(is_even num_students_B ∧ is_even num_students_D ∧ is_even num_students_E) →
  2 = 2 :=
by
  sorry

end even_num_students_count_l122_122088


namespace div_246_by_73_sum_9999_999_99_9_prod_25_29_4_l122_122876

-- Define the division of 246 by 73
theorem div_246_by_73 :
  246 / 73 = 3 + 27 / 73 :=
sorry

-- Define the sum calculation
theorem sum_9999_999_99_9 :
  9999 + 999 + 99 + 9 = 11106 :=
sorry

-- Define the product calculation
theorem prod_25_29_4 :
  25 * 29 * 4 = 2900 :=
sorry

end div_246_by_73_sum_9999_999_99_9_prod_25_29_4_l122_122876


namespace max_product_sum_300_l122_122491

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l122_122491


namespace james_muffins_baked_l122_122745

theorem james_muffins_baked (arthur_muffins : ℝ) (factor : ℝ) (h1 : arthur_muffins = 115.0) (h2 : factor = 12.0) :
  (arthur_muffins / factor) = 9.5833 :=
by 
  -- using the conditions given, we would proceed to prove the result:
  -- sorry is used to indicate that the proof is omitted here
  sorry

end james_muffins_baked_l122_122745


namespace greatest_product_of_two_integers_with_sum_300_l122_122514

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l122_122514


namespace f_odd_increasing_intervals_no_max_value_extreme_points_l122_122015

open Real

namespace FunctionAnalysis

def f (x : ℝ) := x^3 - x

theorem f_odd : ∀ x : ℝ, f (-x) = -f (x) :=
by
  intro x
  show f (-x) = -f (x)
  calc
    f (-x) = (-x)^3 - (-x) : rfl
    ... = -x^3 + x : by ring
    ... = -(x^3 - x) : by ring
    ... = -f (x) : rfl

theorem increasing_intervals : ∀ x : ℝ, 
  (f' x > 0 ↔ x < -sqrt 3 / 3 ∨ x > sqrt 3 / 3) :=
by
  intro x
  have h_deriv : deriv f x = 3 * x^2 - 1 := deriv_pows x
  rw ← h_deriv
  split
  · intro h
    have : 3 * x^2 - 1 > 0 := h
    split
    · apply Or.inl
      linarith [sqrt_le_of_le_0 (@sqrt_nonneg ℝ _ 3)]
    · apply Or.inr
      linarith [sqrt_le_of_le_0 (@sqrt_nonneg ℝ _ 3)]
  · intro h
    cases h
    · linarith [sqrt_le_of_le_0 (@sqrt_nonneg ℝ _ 3)]
    · linarith [sqrt_le_of_le_0 (@sqrt_nonneg ℝ _ 3)]

theorem no_max_value : ∃ L : ℝ, ∀ x : ℝ, f(x) < L → False :=
by
  use 1
  intro x h
  have : ∀ x : ℝ, f(x) > x := λ x, by norm_num
  specialize this x
  linarith

theorem extreme_points : ∀ x : ℝ,
  (f' x = 0) ↔ (x = sqrt(3) / 3 ∨ x = -sqrt(3) / 3) :=
by
  intro x
  have h_deriv : deriv f x = 3 * x^2 - 1 := deriv_pows x
  rw ← h_deriv
  split
  · intro h
    solve_by_elim
  · intro h
    solve_by_elim

end FunctionAnalysis

end f_odd_increasing_intervals_no_max_value_extreme_points_l122_122015


namespace no_integer_solutions_l122_122125

theorem no_integer_solutions :
   ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 = 2 * x^2 * y^2 + 2 * y^2 * z^2 + 2 * z^2 * x^2 + 24 :=
by
  sorry

end no_integer_solutions_l122_122125


namespace sum_of_factors_is_17_l122_122331

theorem sum_of_factors_is_17 :
  ∃ (a b c d e f g : ℤ), 
  (16 * x^4 - 81 * y^4) =
    (a * x + b * y) * 
    (c * x^2 + d * x * y + e * y^2) * 
    (f * x + g * y) ∧ 
    a + b + c + d + e + f + g = 17 :=
by
  sorry

end sum_of_factors_is_17_l122_122331


namespace path_area_and_cost_correct_l122_122737

-- Define the given conditions
def length_field : ℝ := 75
def width_field : ℝ := 55
def path_width : ℝ := 2.5
def cost_per_sq_meter : ℝ := 7

-- Calculate new dimensions including the path
def length_including_path : ℝ := length_field + 2 * path_width
def width_including_path : ℝ := width_field + 2 * path_width

-- Calculate areas
def area_entire_field : ℝ := length_including_path * width_including_path
def area_grass_field : ℝ := length_field * width_field
def area_path : ℝ := area_entire_field - area_grass_field

-- Calculate cost
def cost_of_path : ℝ := area_path * cost_per_sq_meter

theorem path_area_and_cost_correct : 
  area_path = 675 ∧ cost_of_path = 4725 :=
by
  sorry

end path_area_and_cost_correct_l122_122737


namespace max_product_sum_300_l122_122489

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l122_122489


namespace f_odd_increasing_intervals_no_max_value_extreme_points_l122_122016

open Real

namespace FunctionAnalysis

def f (x : ℝ) := x^3 - x

theorem f_odd : ∀ x : ℝ, f (-x) = -f (x) :=
by
  intro x
  show f (-x) = -f (x)
  calc
    f (-x) = (-x)^3 - (-x) : rfl
    ... = -x^3 + x : by ring
    ... = -(x^3 - x) : by ring
    ... = -f (x) : rfl

theorem increasing_intervals : ∀ x : ℝ, 
  (f' x > 0 ↔ x < -sqrt 3 / 3 ∨ x > sqrt 3 / 3) :=
by
  intro x
  have h_deriv : deriv f x = 3 * x^2 - 1 := deriv_pows x
  rw ← h_deriv
  split
  · intro h
    have : 3 * x^2 - 1 > 0 := h
    split
    · apply Or.inl
      linarith [sqrt_le_of_le_0 (@sqrt_nonneg ℝ _ 3)]
    · apply Or.inr
      linarith [sqrt_le_of_le_0 (@sqrt_nonneg ℝ _ 3)]
  · intro h
    cases h
    · linarith [sqrt_le_of_le_0 (@sqrt_nonneg ℝ _ 3)]
    · linarith [sqrt_le_of_le_0 (@sqrt_nonneg ℝ _ 3)]

theorem no_max_value : ∃ L : ℝ, ∀ x : ℝ, f(x) < L → False :=
by
  use 1
  intro x h
  have : ∀ x : ℝ, f(x) > x := λ x, by norm_num
  specialize this x
  linarith

theorem extreme_points : ∀ x : ℝ,
  (f' x = 0) ↔ (x = sqrt(3) / 3 ∨ x = -sqrt(3) / 3) :=
by
  intro x
  have h_deriv : deriv f x = 3 * x^2 - 1 := deriv_pows x
  rw ← h_deriv
  split
  · intro h
    solve_by_elim
  · intro h
    solve_by_elim

end FunctionAnalysis

end f_odd_increasing_intervals_no_max_value_extreme_points_l122_122016


namespace green_passes_blue_at_46_l122_122768

variable {t : ℕ}
variable {k1 k2 k3 k4 : ℝ}
variable {b1 b2 b3 b4 : ℝ}

def elevator_position (k : ℝ) (b : ℝ) (t : ℕ) : ℝ := k * t + b

axiom red_catches_blue_at_36 :
  elevator_position k1 b1 36 = elevator_position k2 b2 36

axiom red_passes_green_at_42 :
  elevator_position k1 b1 42 = elevator_position k3 b3 42

axiom red_passes_yellow_at_48 :
  elevator_position k1 b1 48 = elevator_position k4 b4 48

axiom yellow_passes_blue_at_51 :
  elevator_position k4 b4 51 = elevator_position k2 b2 51

axiom yellow_catches_green_at_54 :
  elevator_position k4 b4 54 = elevator_position k3 b3 54

theorem green_passes_blue_at_46 : 
  elevator_position k3 b3 46 = elevator_position k2 b2 46 := 
sorry

end green_passes_blue_at_46_l122_122768


namespace range_g_l122_122883

noncomputable def g (x : Real) : Real := (Real.sin x)^6 + (Real.cos x)^4

theorem range_g :
  ∃ (a : Real), 
    (∀ x : Real, g x ≥ a ∧ g x ≤ 1) ∧
    (∀ y : Real, y < a → ¬∃ x : Real, g x = y) :=
sorry

end range_g_l122_122883


namespace initial_books_gathered_l122_122172

-- Conditions
def total_books_now : Nat := 59
def books_found : Nat := 26

-- Proof problem
theorem initial_books_gathered : total_books_now - books_found = 33 :=
by
  sorry -- Proof to be provided later

end initial_books_gathered_l122_122172


namespace greatest_product_of_two_integers_with_sum_300_l122_122510

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l122_122510


namespace additional_charge_fraction_of_mile_l122_122808

-- Conditions
def initial_fee : ℝ := 2.25
def additional_charge_per_mile_fraction : ℝ := 0.15
def total_charge (distance : ℝ) : ℝ := 2.25 + 0.15 * distance
def trip_distance : ℝ := 3.6
def total_cost : ℝ := 3.60

-- Question
theorem additional_charge_fraction_of_mile :
  ∃ f : ℝ, total_cost = initial_fee + additional_charge_per_mile_fraction * 3.6 ∧ f = 1 / 9 :=
by
  sorry

end additional_charge_fraction_of_mile_l122_122808


namespace find_values_l122_122635

noncomputable def a : ℝ := Real.sqrt 5 + Real.sqrt 3
noncomputable def b : ℝ := Real.sqrt 5 - Real.sqrt 3

theorem find_values :
  (a + b = 2 * Real.sqrt 5) ∧
  (a * b = 2) ∧
  (a^2 + a * b + b^2 = 18) := by
  sorry

end find_values_l122_122635


namespace incircle_tangent_points_l122_122002

theorem incircle_tangent_points {A B C D P S Q R : Point} (h1 : Parallelogram A B C D)
  (h2 : Circle ∈ TangentToSide (triangle A B C) AC WithTangentPoints (extend BA P) (extend BC S))
  (h3 : Segment PS Intersects AD At Q)
  (h4 : Segment PS Intersects DC At R) :
  Incircle (triangle C D A) IsTangentToSides AD DC AtPoints Q R :=
by sorry

end incircle_tangent_points_l122_122002


namespace moles_of_HCl_formed_l122_122762

-- Conditions: 1 mole of Methane (CH₄) and 2 moles of Chlorine (Cl₂)
def methane := 1 -- 1 mole of methane
def chlorine := 2 -- 2 moles of chlorine

-- Reaction: CH₄ + Cl₂ → CH₃Cl + HCl
-- We state that 1 mole of methane reacts with 1 mole of chlorine to form 1 mole of hydrochloric acid
def reaction (methane chlorine : ℕ) : ℕ := methane

-- Theorem: Prove 1 mole of hydrochloric acid (HCl) is formed
theorem moles_of_HCl_formed : reaction methane chlorine = 1 := by
  sorry

end moles_of_HCl_formed_l122_122762


namespace repeated_root_value_l122_122662

theorem repeated_root_value (m : ℝ) :
  (∃ x : ℝ, x ≠ 1 ∧ (2 / (x - 1) + 3 = m / (x - 1)) ∧ 
            ∀ y : ℝ, y ≠ 1 ∧ (2 / (y - 1) + 3 = m / (y - 1)) → y = x) →
  m = 2 :=
by
  sorry

end repeated_root_value_l122_122662


namespace div_condition_nat_l122_122258

theorem div_condition_nat (n : ℕ) : (n + 1) ∣ (n^2 + 1) ↔ n = 0 ∨ n = 1 :=
by
  sorry

end div_condition_nat_l122_122258


namespace octagon_area_inscribed_in_square_l122_122601

noncomputable def side_length_of_square (perimeter : ℝ) : ℝ :=
  perimeter / 4

noncomputable def trisected_segment_length (side_length : ℝ) : ℝ :=
  side_length / 3

noncomputable def area_of_removed_triangle (segment_length : ℝ) : ℝ :=
  (segment_length * segment_length) / 2

noncomputable def total_area_removed_by_triangles (area_of_triangle : ℝ) : ℝ :=
  4 * area_of_triangle

noncomputable def area_of_square (side_length : ℝ) : ℝ :=
  side_length * side_length

noncomputable def area_of_octagon (area_of_square : ℝ) (total_area_removed : ℝ) : ℝ :=
  area_of_square - total_area_removed

theorem octagon_area_inscribed_in_square (perimeter : ℝ) (H : perimeter = 144) :
  area_of_octagon (area_of_square (side_length_of_square perimeter))
    (total_area_removed_by_triangles (area_of_removed_triangle (trisected_segment_length (side_length_of_square perimeter))))
  = 1008 :=
by
  rw [H]
  -- Intermediate steps would contain calculations for side_length_of_square, trisected_segment_length, area_of_removed_triangle, total_area_removed_by_triangles, and area_of_square based on the given perimeter.
  sorry

end octagon_area_inscribed_in_square_l122_122601


namespace shortest_path_from_A_to_D_not_inside_circle_l122_122169

noncomputable def shortest_path_length : ℝ :=
  let A : ℝ × ℝ := (0, 0)
  let D : ℝ × ℝ := (18, 24)
  let O : ℝ × ℝ := (9, 12)
  let r : ℝ := 15
  15 * Real.pi

theorem shortest_path_from_A_to_D_not_inside_circle :
  let A := (0, 0)
  let D := (18, 24)
  let O := (9, 12)
  let r := 15
  shortest_path_length = 15 * Real.pi := 
by
  sorry

end shortest_path_from_A_to_D_not_inside_circle_l122_122169


namespace number_of_people_adopting_cats_l122_122680

theorem number_of_people_adopting_cats 
    (initial_cats : ℕ)
    (monday_kittens : ℕ)
    (tuesday_injured_cat : ℕ)
    (final_cats : ℕ)
    (cats_per_person_adopting : ℕ)
    (h_initial : initial_cats = 20)
    (h_monday : monday_kittens = 2)
    (h_tuesday : tuesday_injured_cat = 1)
    (h_final: final_cats = 17)
    (h_cats_per_person: cats_per_person_adopting = 2) :
    ∃ (people_adopting : ℕ), people_adopting = 3 :=
by
  sorry

end number_of_people_adopting_cats_l122_122680


namespace monotonically_increasing_function_l122_122834

open Function

theorem monotonically_increasing_function (f : ℝ → ℝ) (h_mono : ∀ x y, x < y → f x < f y) (t : ℝ) (h_t : t ≠ 0) :
    f (t^2 + t) > f t :=
by
  sorry

end monotonically_increasing_function_l122_122834


namespace cost_of_each_lunch_packet_l122_122930

-- Definitions of the variables
def num_students := 50
def total_cost := 3087

-- Variables representing the unknowns
variable (s c n : ℕ)

-- Conditions
def more_than_half_students_bought : Prop := s > num_students / 2
def apples_less_than_cost_per_packet : Prop := n < c
def total_cost_condition : Prop := s * c = total_cost

-- The statement to prove
theorem cost_of_each_lunch_packet :
  (s : ℕ) * c = total_cost ∧
  (s > num_students / 2) ∧
  (n < c)
  -> c = 9 :=
by
  sorry

end cost_of_each_lunch_packet_l122_122930


namespace tangent_line_at_1_f_positive_iff_a_leq_2_l122_122147

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem tangent_line_at_1 (a : ℝ) (h : a = 4) : 
  ∃ k b : ℝ, (k = -2) ∧ (b = 2) ∧ (∀ x : ℝ, f x a = k * (x - 1) + b) :=
sorry

theorem f_positive_iff_a_leq_2 : 
  (∀ x : ℝ, 1 < x → f x a > 0) ↔ a ≤ 2 :=
sorry

end tangent_line_at_1_f_positive_iff_a_leq_2_l122_122147


namespace product_of_perimeters_correct_l122_122195

noncomputable def area (side_length : ℝ) : ℝ := side_length * side_length

theorem product_of_perimeters_correct (x y : ℝ)
  (h1 : area x + area y = 85)
  (h2 : area x - area y = 45) :
  4 * x * 4 * y = 32 * Real.sqrt 325 :=
by sorry

end product_of_perimeters_correct_l122_122195


namespace total_hours_driven_l122_122938

/-- Jade and Krista went on a road trip for 3 days. Jade drives 8 hours each day, and Krista drives 6 hours each day. Prove the total number of hours they drove altogether is 42. -/
theorem total_hours_driven (days : ℕ) (hours_jade_per_day : ℕ) (hours_krista_per_day : ℕ)
  (h1 : days = 3) (h2 : hours_jade_per_day = 8) (h3 : hours_krista_per_day = 6) :
  3 * 8 + 3 * 6 = 42 := 
by
  sorry

end total_hours_driven_l122_122938


namespace park_available_spaces_l122_122931

theorem park_available_spaces :
  let section_A_benches := 30
  let section_A_capacity_per_bench := 4
  let section_B_benches := 20
  let section_B_capacity_per_bench := 5
  let section_C_benches := 15
  let section_C_capacity_per_bench := 6
  let section_A_people := 50
  let section_B_people := 40
  let section_C_people := 45
  let section_A_total_capacity := section_A_benches * section_A_capacity_per_bench
  let section_B_total_capacity := section_B_benches * section_B_capacity_per_bench
  let section_C_total_capacity := section_C_benches * section_C_capacity_per_bench
  let section_A_available := section_A_total_capacity - section_A_people
  let section_B_available := section_B_total_capacity - section_B_people
  let section_C_available := section_C_total_capacity - section_C_people
  let total_available_spaces := section_A_available + section_B_available + section_C_available
  total_available_spaces = 175 := 
by
  let section_A_benches := 30
  let section_A_capacity_per_bench := 4
  let section_B_benches := 20
  let section_B_capacity_per_bench := 5
  let section_C_benches := 15
  let section_C_capacity_per_bench := 6
  let section_A_people := 50
  let section_B_people := 40
  let section_C_people := 45
  let section_A_total_capacity := section_A_benches * section_A_capacity_per_bench
  let section_B_total_capacity := section_B_benches * section_B_capacity_per_bench
  let section_C_total_capacity := section_C_benches * section_C_capacity_per_bench
  let section_A_available := section_A_total_capacity - section_A_people
  let section_B_available := section_B_total_capacity - section_B_people
  let section_C_available := section_C_total_capacity - section_C_people
  let total_available_spaces := section_A_available + section_B_available + section_C_available
  sorry

end park_available_spaces_l122_122931


namespace num_integers_for_polynomial_negative_l122_122131

open Int

theorem num_integers_for_polynomial_negative :
  ∃ (set_x : Finset ℤ), set_x.card = 12 ∧ ∀ x ∈ set_x, (x^4 - 65 * x^2 + 64) < 0 :=
by
  sorry

end num_integers_for_polynomial_negative_l122_122131


namespace system1_solution_system2_solution_l122_122319

-- For Question 1

theorem system1_solution (x y : ℝ) :
  (2 * x - y = 5) ∧ (7 * x - 3 * y = 20) ↔ (x = 5 ∧ y = 5) := 
sorry

-- For Question 2

theorem system2_solution (x y : ℝ) :
  (3 * (x + y) - 4 * (x - y) = 16) ∧ ((x + y)/2 + (x - y)/6 = 1) ↔ (x = 1/3 ∧ y = 7/3) := 
sorry

end system1_solution_system2_solution_l122_122319


namespace domain_transform_l122_122019

-- Definitions based on conditions
def domain_f_x_plus_1 : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }
def domain_f_id : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def domain_f_2x_minus_1 : Set ℝ := { x | 0 ≤ x ∧ x ≤ 5/2 }

-- The theorem to prove the mathematically equivalent problem
theorem domain_transform :
  (∀ x, (x + 1) ∈ domain_f_x_plus_1) →
  (∀ y, y ∈ domain_f_2x_minus_1 ↔ 2 * y - 1 ∈ domain_f_id) :=
by
  sorry

end domain_transform_l122_122019


namespace count_integer_solutions_less_than_zero_l122_122284

theorem count_integer_solutions_less_than_zero : 
  ∃ k : ℕ, k = 4 ∧ (∀ n : ℤ, n^4 - n^3 - 3 * n^2 - 3 * n - 17 < 0 → k = 4) :=
by
  sorry

end count_integer_solutions_less_than_zero_l122_122284


namespace sum_of_squares_l122_122921

theorem sum_of_squares (x y : ℝ) (h1 : (x + y) ^ 2 = 4) (h2 : x * y = -1) :
  x^2 + y^2 = 6 :=
by
  sorry

end sum_of_squares_l122_122921


namespace combined_teaching_years_l122_122851

def Adrienne_Yrs : ℕ := 22
def Virginia_Yrs : ℕ := Adrienne_Yrs + 9
def Dennis_Yrs : ℕ := 40

theorem combined_teaching_years :
  Adrienne_Yrs + Virginia_Yrs + Dennis_Yrs = 93 := by
  -- Proof omitted
  sorry

end combined_teaching_years_l122_122851


namespace fraction_zero_implies_a_eq_neg2_l122_122161

theorem fraction_zero_implies_a_eq_neg2 (a : ℝ) (h : (a^2 - 4) / (a - 2) = 0) (h2 : a ≠ 2) : a = -2 :=
sorry

end fraction_zero_implies_a_eq_neg2_l122_122161


namespace circumference_of_cone_l122_122868

theorem circumference_of_cone (V : ℝ) (h : ℝ) (C : ℝ) 
  (hV : V = 36 * Real.pi) (hh : h = 3) : 
  C = 12 * Real.pi :=
sorry

end circumference_of_cone_l122_122868


namespace sub_fraction_l122_122987

theorem sub_fraction (a b c d : ℚ) (h1 : a = 5) (h2 : b = 9) (h3 : c = 1) (h4 : d = 6) : (a / b) - (c / d) = 7 / 18 := 
by
  sorry

end sub_fraction_l122_122987


namespace probability_Z_l122_122728

theorem probability_Z (p_X p_Y p_Z : ℚ)
  (hX : p_X = 2 / 5)
  (hY : p_Y = 1 / 4)
  (hTotal : p_X + p_Y + p_Z = 1) :
  p_Z = 7 / 20 := by sorry

end probability_Z_l122_122728


namespace paul_sandwiches_l122_122051

theorem paul_sandwiches (sandwiches_day1 sandwiches_day2 sandwiches_day3 total_sandwiches_3days total_sandwiches_6days : ℕ) 
    (h1 : sandwiches_day1 = 2) 
    (h2 : sandwiches_day2 = 2 * sandwiches_day1) 
    (h3 : sandwiches_day3 = 2 * sandwiches_day2) 
    (h4 : total_sandwiches_3days = sandwiches_day1 + sandwiches_day2 + sandwiches_day3) 
    (h5 : total_sandwiches_6days = 2 * total_sandwiches_3days) 
    : total_sandwiches_6days = 28 := 
by 
    sorry

end paul_sandwiches_l122_122051


namespace paul_sandwiches_l122_122052

theorem paul_sandwiches (sandwiches_day1 sandwiches_day2 sandwiches_day3 total_sandwiches_3days total_sandwiches_6days : ℕ) 
    (h1 : sandwiches_day1 = 2) 
    (h2 : sandwiches_day2 = 2 * sandwiches_day1) 
    (h3 : sandwiches_day3 = 2 * sandwiches_day2) 
    (h4 : total_sandwiches_3days = sandwiches_day1 + sandwiches_day2 + sandwiches_day3) 
    (h5 : total_sandwiches_6days = 2 * total_sandwiches_3days) 
    : total_sandwiches_6days = 28 := 
by 
    sorry

end paul_sandwiches_l122_122052


namespace sheep_count_l122_122707

/-- The ratio between the number of sheep and the number of horses at the Stewart farm is 2 to 7.
    Each horse is fed 230 ounces of horse food per day, and the farm needs a total of 12,880 ounces
    of horse food per day. -/
theorem sheep_count (S H : ℕ) (h_ratio : S = (2 / 7) * H)
    (h_food : H * 230 = 12880) : S = 16 :=
sorry

end sheep_count_l122_122707


namespace solve_for_x_l122_122267

def delta (x : ℝ) : ℝ := 5 * x + 6
def phi (x : ℝ) : ℝ := 6 * x + 5

theorem solve_for_x : ∀ x : ℝ, delta (phi x) = -1 → x = - 16 / 15 :=
by
  intro x
  intro h
  -- Proof skipped
  sorry

end solve_for_x_l122_122267


namespace max_product_300_l122_122547

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l122_122547


namespace grace_dimes_count_l122_122151

-- Defining the conditions
def dimes_to_pennies (d : ℕ) : ℕ := 10 * d
def nickels_to_pennies : ℕ := 10 * 5
def total_pennies (d : ℕ) : ℕ := dimes_to_pennies d + nickels_to_pennies

-- The statement of the theorem
theorem grace_dimes_count (d : ℕ) (h : total_pennies d = 150) : d = 10 := 
sorry

end grace_dimes_count_l122_122151


namespace rectangle_area_l122_122241

theorem rectangle_area {A_s A_r : ℕ} (s l w : ℕ) (h1 : A_s = 36) (h2 : A_s = s * s)
  (h3 : w = s) (h4 : l = 3 * w) (h5 : A_r = w * l) : A_r = 108 :=
by
  sorry

end rectangle_area_l122_122241


namespace triangle_inequality_l122_122000

theorem triangle_inequality (a b c : ℝ) (habc_triangle : a + b > c ∧ b + c > a ∧ a + c > b) : 
  2 * (a^2 * b^2 + b^2 * c^2 + a^2 * c^2) > (a^4 + b^4 + c^4) :=
by
  sorry

end triangle_inequality_l122_122000


namespace intersection_A_B_l122_122138

def A := { x : Real | -3 < x ∧ x < 2 }
def B := { x : Real | x^2 + 4*x - 5 ≤ 0 }

theorem intersection_A_B :
  (A ∩ B = { x : Real | -3 < x ∧ x ≤ 1 }) := by
  sorry

end intersection_A_B_l122_122138


namespace total_windows_l122_122863

theorem total_windows (installed: ℕ) (hours_per_window: ℕ) (remaining_hours: ℕ) : installed = 8 → hours_per_window = 8 → remaining_hours = 48 → 
  (installed + remaining_hours / hours_per_window) = 14 := by 
  intros h1 h2 h3
  sorry

end total_windows_l122_122863


namespace fraction_of_number_l122_122732

theorem fraction_of_number (x f : ℚ) (h1 : x = 2/3) (h2 : f * x = (64/216) * (1/x)) : f = 2/3 :=
by
  sorry

end fraction_of_number_l122_122732


namespace greatest_product_sum_300_l122_122368

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l122_122368


namespace tennis_handshakes_l122_122607

-- Define the participants and the condition
def number_of_participants := 8
def handshakes_no_partner (n : ℕ) := n * (n - 2) / 2

-- Prove that the number of handshakes is 24
theorem tennis_handshakes : handshakes_no_partner number_of_participants = 24 := by
  -- Since we are skipping the proof for now
  sorry

end tennis_handshakes_l122_122607


namespace math_problem_l122_122815

noncomputable def proof : Prop :=
  ∀ (a b : ℝ), a ≠ 0 → b ≠ 0 →
  ( (1 / a + 1 / b) / (1 / a - 1 / b) = 1001 ) →
  ((a + b) / (a - b) = 1001)

theorem math_problem : proof := 
  by
    intros a b h₁ h₂ h₃
    sorry

end math_problem_l122_122815


namespace max_product_sum_300_l122_122483

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l122_122483


namespace greatest_product_of_two_integers_with_sum_300_l122_122508

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l122_122508


namespace mutually_exclusive_events_l122_122860

/-- A group consists of 3 boys and 2 girls. Two students are to be randomly selected to participate in a speech competition. -/
def num_boys : ℕ := 3
def num_girls : ℕ := 2
def total_selected : ℕ := 2

/-- Possible events under consideration:
  A*: Exactly one boy is selected or exactly two girls are selected -/
def is_boy (s : ℕ) (boys : ℕ) : Prop := s ≤ boys 
def is_girl (s : ℕ) (girls : ℕ) : Prop := s ≤ girls
def one_boy_selected (selected : ℕ) (boys : ℕ) := selected = 1 ∧ is_boy selected boys
def two_girls_selected (selected : ℕ) (girls : ℕ) := selected = 2 ∧ is_girl selected girls

theorem mutually_exclusive_events 
  (selected_boy : ℕ) (selected_girl : ℕ) :
  one_boy_selected selected_boy num_boys ∧ selected_boy + selected_girl = total_selected 
  ∧ two_girls_selected selected_girl num_girls 
  → (one_boy_selected selected_boy num_boys ∨ two_girls_selected selected_girl num_girls) :=
by
  sorry

end mutually_exclusive_events_l122_122860


namespace hyperbola_with_foci_on_y_axis_l122_122301

variable (m n : ℝ)

-- condition stating that mn < 0
def mn_neg : Prop := m * n < 0

-- the main theorem statement
theorem hyperbola_with_foci_on_y_axis (h : mn_neg m n) : 
  (∃ a : ℝ, a > 0 ∧ ∀ x y : ℝ, m * x^2 - m * y^2 = n ↔ y^2 - x^2 = a) :=
sorry

end hyperbola_with_foci_on_y_axis_l122_122301


namespace max_product_two_integers_l122_122452

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l122_122452


namespace range_of_a_l122_122642

-- Definitions of position conditions in the 4th quadrant
def PosInFourthQuad (x y : ℝ) : Prop := (x > 0) ∧ (y < 0)

-- Statement to prove
theorem range_of_a (a : ℝ) (h : PosInFourthQuad (2 * a + 4) (3 * a - 6)) : -2 < a ∧ a < 2 :=
  sorry

end range_of_a_l122_122642


namespace stairs_climbed_l122_122188

theorem stairs_climbed (s v r : ℕ) 
  (h_s: s = 318) 
  (h_v: v = 18 + s / 2) 
  (h_r: r = 2 * v) 
  : s + v + r = 849 :=
by {
  sorry
}

end stairs_climbed_l122_122188


namespace kitchen_upgrade_cost_l122_122212

-- Define the number of cabinet knobs and their cost
def num_knobs : ℕ := 18
def cost_per_knob : ℝ := 2.50

-- Define the number of drawer pulls and their cost
def num_pulls : ℕ := 8
def cost_per_pull : ℝ := 4.00

-- Calculate the total cost of the knobs
def total_cost_knobs : ℝ := num_knobs * cost_per_knob

-- Calculate the total cost of the pulls
def total_cost_pulls : ℝ := num_pulls * cost_per_pull

-- Calculate the total cost of the kitchen upgrade
def total_cost : ℝ := total_cost_knobs + total_cost_pulls

-- Theorem statement
theorem kitchen_upgrade_cost : total_cost = 77 := by
  sorry

end kitchen_upgrade_cost_l122_122212


namespace find_m_l122_122152

theorem find_m (m : ℕ) (h : m * (m - 1) * (m - 2) * (m - 3) * (m - 4) = 2 * m * (m - 1) * (m - 2)) : m = 5 :=
sorry

end find_m_l122_122152


namespace num_packs_blue_tshirts_l122_122251

def num_white_tshirts_per_pack : ℕ := 6
def num_packs_white_tshirts : ℕ := 5
def num_blue_tshirts_per_pack : ℕ := 9
def total_num_tshirts : ℕ := 57

theorem num_packs_blue_tshirts : (total_num_tshirts - num_white_tshirts_per_pack * num_packs_white_tshirts) / num_blue_tshirts_per_pack = 3 := by
  sorry

end num_packs_blue_tshirts_l122_122251


namespace value_of_x2_plus_y2_l122_122910

theorem value_of_x2_plus_y2 (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 :=
by
  sorry

end value_of_x2_plus_y2_l122_122910


namespace machine_rate_ratio_l122_122688

theorem machine_rate_ratio (A B : ℕ) (h1 : ∃ A : ℕ, 8 * A = 8 * A)
  (h2 : ∃ W : ℕ, W = 8 * A)
  (h3 : ∃ W1 : ℕ, W1 = 6 * A)
  (h4 : ∃ W2 : ℕ, W2 = 2 * A)
  (h5 : ∃ B : ℕ, 8 * B = 2 * A) :
  (B:ℚ) / (A:ℚ) = 1 / 4 :=
by sorry

end machine_rate_ratio_l122_122688


namespace probability_sqrt_lt_9_l122_122567

theorem probability_sqrt_lt_9 : 
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
in probability = 71 / 90 :=
by
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  sorry

end probability_sqrt_lt_9_l122_122567


namespace max_product_sum_300_l122_122480

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l122_122480


namespace arithmetic_sequence_product_l122_122038

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ) 
  (h_inc : ∀ n, b (n + 1) - b n = d)
  (h_pos : d > 0)
  (h_prod : b 5 * b 6 = 21) 
  : b 4 * b 7 = -779 ∨ b 4 * b 7 = -11 :=
sorry

end arithmetic_sequence_product_l122_122038


namespace greatest_product_obtainable_l122_122408

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l122_122408


namespace simplify_expr_l122_122924

theorem simplify_expr (x : ℕ) (h : x = 2018) : x^2 + 2 * x - x * (x + 1) = x := by
  sorry

end simplify_expr_l122_122924


namespace g_triple_application_l122_122922

def g (x : ℤ) : ℤ := 7 * x - 3

theorem g_triple_application : g (g (g 3)) = 858 := by
  sorry

end g_triple_application_l122_122922


namespace count_factors_of_product_l122_122120

theorem count_factors_of_product :
  let n := 8^4 * 7^3 * 9^1 * 5^5
  ∃ (count : ℕ), count = 936 ∧ 
    ∀ f : ℕ, f ∣ n → ∃ a b c d : ℕ,
      a ≤ 12 ∧ b ≤ 2 ∧ c ≤ 5 ∧ d ≤ 3 ∧ 
      f = 2^a * 3^b * 5^c * 7^d :=
by sorry

end count_factors_of_product_l122_122120


namespace real_root_ineq_l122_122185

theorem real_root_ineq (a b : ℝ) (x₀ : ℝ) (h : x₀^4 - a * x₀^3 + 2 * x₀^2 - b * x₀ + 1 = 0) :
  a^2 + b^2 ≥ 8 :=
by
  sorry

end real_root_ineq_l122_122185


namespace rabbit_speed_l122_122029

theorem rabbit_speed (x : ℕ) :
  2 * (2 * x + 4) = 188 → x = 45 := by
  sorry

end rabbit_speed_l122_122029


namespace quadratic_form_sum_l122_122970

theorem quadratic_form_sum :
  ∃ a b c : ℝ, (∀ x : ℝ, 5 * x^2 - 45 * x - 500 = a * (x + b)^2 + c) ∧ (a + b + c = -605.75) :=
sorry

end quadratic_form_sum_l122_122970


namespace desert_area_2020_correct_desert_area_less_8_10_5_by_2023_l122_122610

-- Define initial desert area
def initial_desert_area : ℝ := 9 * 10^5

-- Define increase in desert area each year as observed
def yearly_increase (n : ℕ) : ℝ :=
  match n with
  | 1998 => 2000
  | 1999 => 4000
  | 2000 => 6001
  | 2001 => 7999
  | 2002 => 10001
  | _    => 0

-- Define arithmetic progression of increases
def common_difference : ℝ := 2000

-- Define desert area in 2020
def desert_area_2020 : ℝ :=
  initial_desert_area + 10001 + 18 * common_difference

-- Statement: Desert area by the end of 2020 is approximately 9.46 * 10^5 hm^2
theorem desert_area_2020_correct :
  desert_area_2020 = 9.46 * 10^5 :=
sorry

-- Define yearly transformation and desert increment with afforestation from 2003
def desert_area_with_afforestation (n : ℕ) : ℝ :=
  if n < 2003 then
    initial_desert_area + yearly_increase n
  else
    initial_desert_area + 10001 + (n - 2002) * (common_difference - 8000)

-- Statement: Desert area will be less than 8 * 10^5 hm^2 by the end of 2023
theorem desert_area_less_8_10_5_by_2023 :
  desert_area_with_afforestation 2023 < 8 * 10^5 :=
sorry

end desert_area_2020_correct_desert_area_less_8_10_5_by_2023_l122_122610


namespace number_of_neutrons_l122_122837

def mass_number (element : Type) : ℕ := 61
def atomic_number (element : Type) : ℕ := 27

theorem number_of_neutrons (element : Type) : mass_number element - atomic_number element = 34 :=
by
  -- Place the proof here
  sorry

end number_of_neutrons_l122_122837


namespace incircle_tangent_points_l122_122001

theorem incircle_tangent_points {A B C D P S Q R : Point} 
  (h_parallelogram : parallelogram A B C D) 
  (h_tangent_ac : tangent (circle P Q R) A C) 
  (h_tangent_ba_ext : tangent (circle P Q R) (extension B A P)) 
  (h_tangent_bc_ext : tangent (circle P Q R) (extension B C S)) 
  (h_ps_intersect_da : segment_intersect P S D A Q)
  (h_ps_intersect_dc : segment_intersect P S D C R) :
  tangent (incircle D C A) D A Q ∧ tangent (incircle D C A) D C R := sorry

end incircle_tangent_points_l122_122001


namespace time_to_ascend_non_working_escalator_l122_122714

-- Define the variables as given in the conditions
def V := 1 / 60 -- Speed of the moving escalator in units per minute
def U := (1 / 24) - (1 / 60) -- Speed of Gavrila running relative to the escalator

-- Theorem stating that the time to ascend a non-working escalator is 40 seconds
theorem time_to_ascend_non_working_escalator : 
  (1 : ℚ) = U * (40 / 60) := 
by sorry

end time_to_ascend_non_working_escalator_l122_122714


namespace total_hours_A_ascending_and_descending_l122_122235

theorem total_hours_A_ascending_and_descending
  (ascending_speed_A ascending_speed_B descending_speed_A descending_speed_B distance summit_distance : ℝ)
  (h1 : descending_speed_A = 1.5 * ascending_speed_A)
  (h2 : descending_speed_B = 1.5 * ascending_speed_B)
  (h3 : ascending_speed_A > ascending_speed_B)
  (h4 : 1/ascending_speed_A + 1/ascending_speed_B = 1/hour - 600/summit_distance)
  (h5 : 0.5 * summit_distance/ascending_speed_A = (summit_distance - 600)/ascending_speed_B) :
  (summit_distance / ascending_speed_A) + (summit_distance / descending_speed_A) = 1.5 := 
sorry

end total_hours_A_ascending_and_descending_l122_122235


namespace part1_part2_l122_122148
noncomputable def equation1 (x k : ℝ) := 3 * (2 * x - 1) = k + 2 * x
noncomputable def equation2 (x k : ℝ) := (x - k) / 2 = x + 2 * k

theorem part1 (x k : ℝ) (h1 : equation1 4 k) : equation2 x k ↔ x = -65 := sorry

theorem part2 (x k : ℝ) (h1 : equation1 x k) (h2 : equation2 x k) : k = -1 / 7 := sorry

end part1_part2_l122_122148


namespace addition_example_l122_122853

theorem addition_example : 0.4 + 56.7 = 57.1 := by
  -- Here we need to prove the main statement
  sorry

end addition_example_l122_122853


namespace greatest_product_sum_300_l122_122359

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l122_122359


namespace range_of_m_condition_l122_122278

theorem range_of_m_condition (m : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : x₁ * x₁ - 2 * m * x₁ + m - 3 = 0) 
  (h₂ : x₂ * x₂ - 2 * m * x₂ + m - 3 = 0)
  (hx₁ : x₁ > -1 ∧ x₁ < 0)
  (hx₂ : x₂ > 3) :
  m > 6 / 5 ∧ m < 3 :=
sorry

end range_of_m_condition_l122_122278


namespace max_product_300_l122_122546

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l122_122546


namespace ratio_of_circle_areas_l122_122794

noncomputable def ratio_of_areas (R_X R_Y : ℝ) : ℝ := (π * R_X^2) / (π * R_Y^2)

theorem ratio_of_circle_areas
  (R_X R_Y : ℝ)
  (h : (60 / 360) * 2 * π * R_X = (40 / 360) * 2 * π * R_Y) :
  ratio_of_areas R_X R_Y = 9 / 4 :=
by
  sorry

end ratio_of_circle_areas_l122_122794


namespace range_of_a_l122_122784

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x : ℝ, (2 * a < x ∧ x < a + 5) → (x < 6)) ↔ (1 < a ∧ a < 5) :=
by
  sorry

end range_of_a_l122_122784


namespace equilateral_triangle_path_l122_122750

noncomputable def equilateral_triangle_path_length (side_length_triangle side_length_square : ℝ) : ℝ :=
  let radius := side_length_triangle
  let rotational_path_length := 4 * 3 * 2 * Real.pi
  let diagonal_length := (Real.sqrt (side_length_square^2 + side_length_square^2))
  let linear_path_length := 2 * diagonal_length
  rotational_path_length + linear_path_length

theorem equilateral_triangle_path (side_length_triangle side_length_square : ℝ) 
  (h_triangle : side_length_triangle = 3) (h_square : side_length_square = 6) :
  equilateral_triangle_path_length side_length_triangle side_length_square = 24 * Real.pi + 12 * Real.sqrt 2 :=
by
  rw [h_triangle, h_square]
  unfold equilateral_triangle_path_length
  sorry

end equilateral_triangle_path_l122_122750


namespace smaller_circle_radius_l122_122119

noncomputable def radius_of_smaller_circles (R : ℝ) (r1 r2 r3 : ℝ) (OA OB OC : ℝ) : Prop :=
(OA = R + r1) ∧ (OB = R + 3 * r1) ∧ (OC = R + 5 * r1) ∧ 
((OB = OA + 2 * r1) ∧ (OC = OB + 2 * r1))

theorem smaller_circle_radius (r : ℝ) (R : ℝ := 2) :
  radius_of_smaller_circles R r r r (R + r) (R + 3 * r) (R + 5 * r) → r = 1 :=
by
  sorry

end smaller_circle_radius_l122_122119


namespace case_b_conditions_l122_122951

-- Definition of the polynomial
def polynomial (p q x : ℝ) : ℝ := x^2 + p * x + q

-- Main theorem
theorem case_b_conditions (p q: ℝ) (x1 x2: ℝ) (hx1: x1 ≤ 0) (hx2: x2 ≥ 2) :
    q ≤ 0 ∧ 2 * p + q + 4 ≤ 0 :=
sorry

end case_b_conditions_l122_122951


namespace reduction_percentage_toy_l122_122825

-- Definition of key parameters
def paintings_bought : ℕ := 10
def cost_per_painting : ℕ := 40
def toys_bought : ℕ := 8
def cost_per_toy : ℕ := 20
def total_cost : ℕ := (paintings_bought * cost_per_painting) + (toys_bought * cost_per_toy) -- $560
def painting_selling_price_per_unit : ℕ := cost_per_painting - (cost_per_painting * 10 / 100) -- $36
def total_loss : ℕ := 64

-- Define percentage reduction in the selling price of a wooden toy
variable {x : ℕ} -- Define x as a percentage value to be solved

-- Theorems to prove
theorem reduction_percentage_toy (x) : 
  (paintings_bought * painting_selling_price_per_unit) 
  + (toys_bought * (cost_per_toy - (cost_per_toy * x / 100))) 
  = (total_cost - total_loss) 
  → x = 15 := 
by
  sorry

end reduction_percentage_toy_l122_122825


namespace circle_center_radius_l122_122012

theorem circle_center_radius (x y : ℝ) :
  (x - 1)^2 + (y - 3)^2 = 4 → (1, 3) = (1, 3) ∧ 2 = 2 :=
by
  intro h
  exact ⟨rfl, rfl⟩

end circle_center_radius_l122_122012


namespace taylor_class_more_girls_l122_122972

theorem taylor_class_more_girls (b g : ℕ) (total : b + g = 42) (ratio : b / g = 3 / 4) : g - b = 6 := by
  sorry

end taylor_class_more_girls_l122_122972


namespace kitchen_upgrade_total_cost_l122_122210

-- Defining the given conditions
def num_cabinet_knobs : ℕ := 18
def cost_per_cabinet_knob : ℚ := 2.50

def num_drawer_pulls : ℕ := 8
def cost_per_drawer_pull : ℚ := 4

-- Definition of the total cost function
def total_cost : ℚ :=
  (num_cabinet_knobs * cost_per_cabinet_knob) + (num_drawer_pulls * cost_per_drawer_pull)

-- The theorem to prove the total cost is $77.00
theorem kitchen_upgrade_total_cost : total_cost = 77 := by
  sorry

end kitchen_upgrade_total_cost_l122_122210


namespace sum_first_n_terms_of_arithmetic_sequence_l122_122004

def arithmetic_sequence_sum (a1 d n: ℕ) : ℕ :=
  n * (2 * a1 + (n - 1) * d) / 2

theorem sum_first_n_terms_of_arithmetic_sequence :
  arithmetic_sequence_sum 2 2 n = n * (n + 1) / 2 :=
by sorry

end sum_first_n_terms_of_arithmetic_sequence_l122_122004


namespace greatest_product_sum_300_l122_122371

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l122_122371


namespace handmade_ornaments_l122_122050

noncomputable def handmade_more_than_1_sixth(O : ℕ) (h1 : (1 / 3 : ℚ) * O = 20) (h2 : (1 / 2 : ℚ) * (handmade : ℕ) = 20) : Prop :=
  handmade - (1 / 6 * O) = 20

theorem handmade_ornaments (O handmade : ℕ) (h1 : (1 / 3 : ℚ) * O = 20) (h2 : (1 / 2 : ℚ) * handmade = 20) :
  handmade_more_than_1_sixth O h1 h2 :=
by
  sorry

end handmade_ornaments_l122_122050


namespace simplify_polynomial_l122_122218

-- Define the original polynomial
def original_expr (x : ℝ) : ℝ := 3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 + 2 * x^3

-- Define the simplified version of the polynomial
def simplified_expr (x : ℝ) : ℝ := 2 * x^3 - x^2 + 23 * x - 3

-- State the theorem that the original expression is equal to the simplified one
theorem simplify_polynomial (x : ℝ) : original_expr x = simplified_expr x := 
by 
  sorry

end simplify_polynomial_l122_122218


namespace sub_fraction_l122_122989

theorem sub_fraction (a b c d : ℚ) (h1 : a = 5) (h2 : b = 9) (h3 : c = 1) (h4 : d = 6) : (a / b) - (c / d) = 7 / 18 := 
by
  sorry

end sub_fraction_l122_122989


namespace son_and_daughter_current_ages_l122_122104

theorem son_and_daughter_current_ages
  (father_age_now : ℕ)
  (son_age_5_years_ago : ℕ)
  (daughter_age_5_years_ago : ℝ)
  (h_father_son_birth : father_age_now - (son_age_5_years_ago + 5) = (son_age_5_years_ago + 5))
  (h_father_daughter_birth : father_age_now - (daughter_age_5_years_ago + 5) = (daughter_age_5_years_ago + 5))
  (h_daughter_half_son_5_years_ago : daughter_age_5_years_ago = son_age_5_years_ago / 2) :
  son_age_5_years_ago + 5 = 12 ∧ daughter_age_5_years_ago + 5 = 8.5 :=
by
  sorry

end son_and_daughter_current_ages_l122_122104


namespace value_of_a_l122_122648

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem value_of_a (a : ℝ) : 
  (∀ x, deriv (f a) x = 6 * x + 3 * a * x^2) →
  deriv (f a) (-1) = 6 → a = 4 :=
by
  -- Proof will be filled in here
  sorry

end value_of_a_l122_122648


namespace tan_identity_l122_122133

theorem tan_identity
  (α : ℝ)
  (h : Real.tan (π / 3 - α) = 1 / 3) :
  Real.tan (2 * π / 3 + α) = -1 / 3 := 
sorry

end tan_identity_l122_122133


namespace max_product_of_two_integers_whose_sum_is_300_l122_122500

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l122_122500


namespace length_AD_of_circle_l122_122693

def circle_radius : ℝ := 8
def p_A : Prop := True  -- stand-in for the point A on the circle
def p_B : Prop := True  -- stand-in for the point B on the circle
def dist_AB : ℝ := 10
def p_D : Prop := True  -- stand-in for point D opposite B

theorem length_AD_of_circle 
  (r : ℝ := circle_radius)
  (A B D : Prop)
  (h_AB : dist_AB = 10)
  (h_radius : r = 8)
  (h_opposite : D)
  : ∃ AD : ℝ, AD = Real.sqrt 252.75 :=
sorry

end length_AD_of_circle_l122_122693


namespace complement_intersection_l122_122037

open Set

def U : Set ℕ := {x | x < 6}
def A : Set ℕ := {0, 2, 4}
def B : Set ℕ := {1, 2, 5}

theorem complement_intersection :
  ((U \ A) ∩ B) = {1, 5} :=
by
  sorry

end complement_intersection_l122_122037


namespace total_height_of_buildings_l122_122591

theorem total_height_of_buildings :
  let height_first_building := 600
  let height_second_building := 2 * height_first_building
  let height_third_building := 3 * (height_first_building + height_second_building)
  height_first_building + height_second_building + height_third_building = 7200 := by
    let height_first_building := 600
    let height_second_building := 2 * height_first_building
    let height_third_building := 3 * (height_first_building + height_second_building)
    show height_first_building + height_second_building + height_third_building = 7200
    sorry

end total_height_of_buildings_l122_122591


namespace simplify_polynomial_l122_122219

-- Define the original polynomial
def original_expr (x : ℝ) : ℝ := 3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 + 2 * x^3

-- Define the simplified version of the polynomial
def simplified_expr (x : ℝ) : ℝ := 2 * x^3 - x^2 + 23 * x - 3

-- State the theorem that the original expression is equal to the simplified one
theorem simplify_polynomial (x : ℝ) : original_expr x = simplified_expr x := 
by 
  sorry

end simplify_polynomial_l122_122219


namespace intersection_eq_l122_122137

open Set

variable (A B : Set ℝ)

def setA : A = {x | -3 < x ∧ x < 2} := sorry

def setB : B = {x | x^2 + 4*x - 5 ≤ 0} := sorry

theorem intersection_eq : A ∩ B = {x | -3 < x ∧ x ≤ 1} :=
sorry

end intersection_eq_l122_122137


namespace max_product_300_l122_122538

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l122_122538


namespace greatest_product_obtainable_l122_122406

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l122_122406


namespace triangle_sides_fraction_sum_eq_one_l122_122162

theorem triangle_sides_fraction_sum_eq_one
  (a b c : ℝ)
  (h : a^2 + b^2 = c^2 + a * b) :
  a / (b + c) + b / (c + a) = 1 :=
sorry

end triangle_sides_fraction_sum_eq_one_l122_122162


namespace greatest_product_obtainable_l122_122402

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l122_122402


namespace max_product_two_integers_l122_122455

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l122_122455


namespace range_of_f_ge_1_l122_122181

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then (x + 1) ^ 2 else 4 - Real.sqrt (x - 1)

theorem range_of_f_ge_1 :
  {x : ℝ | f x ≥ 1} = {x : ℝ | x ≤ -2} ∪ {x : ℝ | 0 ≤ x ∧ x ≤ 10} :=
by
  sorry

end range_of_f_ge_1_l122_122181


namespace josanna_minimum_test_score_l122_122809

def test_scores := [90, 80, 70, 60, 85]

def target_average_increase := 3

def current_average (scores : List ℕ) : ℚ :=
  (scores.sum : ℚ) / scores.length

def sixth_test_score_needed (scores : List ℕ) (increase : ℚ) : ℚ :=
  let current_avg := current_average scores
  let target_avg := current_avg + increase
  target_avg * (scores.length + 1) - scores.sum

theorem josanna_minimum_test_score :
  sixth_test_score_needed test_scores target_average_increase = 95 := sorry

end josanna_minimum_test_score_l122_122809


namespace residues_of_f_singularities_l122_122129

open Complex

noncomputable def f (z : ℂ) : ℂ := (sin z / cos z) / (z^2 - (π / 4) * z)

theorem residues_of_f_singularities :
  residue (f) 0 = 0 ∧
  residue (f) (π / 4) = 4 / π ∧
  ∀ k : ℤ, residue (f) (π / 2 + k * π) = -1 / ((π / 2 + k * π) * (π / 4 + k * π)) :=
by
  sorry

end residues_of_f_singularities_l122_122129


namespace rationalize_and_subtract_l122_122827

theorem rationalize_and_subtract :
  (7 / (3 + Real.sqrt 15)) * (3 - Real.sqrt 15) / (3^2 - (Real.sqrt 15)^2) 
  - (1 / 2) = -4 + (7 * Real.sqrt 15) / 6 :=
by
  sorry

end rationalize_and_subtract_l122_122827


namespace greatest_product_sum_300_l122_122372

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l122_122372


namespace find_h_l122_122071

noncomputable def y1 (x h j : ℝ) := 4 * (x - h) ^ 2 + j
noncomputable def y2 (x h k : ℝ) := 3 * (x - h) ^ 2 + k

theorem find_h (h j k : ℝ)
  (C1 : y1 0 h j = 2024)
  (C2 : y2 0 h k = 2025)
  (H1 : y1 x h j = 0 → ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ * x₂ = 506)
  (H2 : y2 x h k = 0 → ∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ * x₂ = 675) :
  h = 22.5 :=
sorry

end find_h_l122_122071


namespace one_quarters_in_one_eighth_l122_122786

theorem one_quarters_in_one_eighth : (1 / 8) / (1 / 4) = 1 / 2 :=
by sorry

end one_quarters_in_one_eighth_l122_122786


namespace grogg_possible_cubes_l122_122022

theorem grogg_possible_cubes (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_prob : (a - 2) * (b - 2) * (c - 2) / (a * b * c) = 1 / 5) :
  a * b * c = 120 ∨ a * b * c = 160 ∨ a * b * c = 240 ∨ a * b * c = 360 := 
sorry

end grogg_possible_cubes_l122_122022


namespace width_of_field_l122_122106

noncomputable def field_width 
  (field_length : ℝ) 
  (rope_length : ℝ)
  (grazing_area : ℝ) : ℝ :=
if field_length > 2 * rope_length 
then rope_length
else grazing_area

theorem width_of_field 
  (field_length : ℝ := 45)
  (rope_length : ℝ := 22)
  (grazing_area : ℝ := 380.132711084365) : field_width field_length rope_length grazing_area = rope_length :=
by 
  sorry

end width_of_field_l122_122106


namespace f_odd_f_monotonic_increasing_intervals_f_no_max_value_f_extreme_points_l122_122017

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem f_odd (x : ℝ) : f (-x) = -f (x) :=
by sorry

theorem f_monotonic_increasing_intervals :
  ∀ x : ℝ, (x < -Real.sqrt 3 / 3 ∨ x > Real.sqrt 3 / 3) → f x' > f x :=
by sorry

theorem f_no_max_value :
  ∀ x : ℝ, ¬(∃ M, f x ≤ M) :=
by sorry

theorem f_extreme_points :
  f (-Real.sqrt 3 / 3) = -2 * Real.sqrt 3 / 9 ∧ f (Real.sqrt 3 / 3) = -2 * Real.sqrt 3 / 9 :=
by sorry

end f_odd_f_monotonic_increasing_intervals_f_no_max_value_f_extreme_points_l122_122017


namespace hexagon_chord_length_valid_l122_122596

def hexagon_inscribed_chord_length : ℚ := 48 / 49

theorem hexagon_chord_length_valid : 
    ∃ (p q : ℕ), gcd p q = 1 ∧ hexagon_inscribed_chord_length = p / q ∧ p + q = 529 :=
sorry

end hexagon_chord_length_valid_l122_122596


namespace max_area_of_sector_l122_122900

variable (r l S : ℝ)

theorem max_area_of_sector (h_circumference : 2 * r + l = 8) (h_area : S = (1 / 2) * l * r) : 
  S ≤ 4 :=
sorry

end max_area_of_sector_l122_122900


namespace fewer_bronze_stickers_l122_122956

theorem fewer_bronze_stickers
  (gold_stickers : ℕ)
  (silver_stickers : ℕ)
  (each_student_stickers : ℕ)
  (students : ℕ)
  (total_stickers_given : ℕ)
  (bronze_stickers : ℕ)
  (total_gold_and_silver_stickers : ℕ)
  (gold_stickers_eq : gold_stickers = 50)
  (silver_stickers_eq : silver_stickers = 2 * gold_stickers)
  (each_student_stickers_eq : each_student_stickers = 46)
  (students_eq : students = 5)
  (total_stickers_given_eq : total_stickers_given = students * each_student_stickers)
  (total_gold_and_silver_stickers_eq : total_gold_and_silver_stickers = gold_stickers + silver_stickers)
  (bronze_stickers_eq : bronze_stickers = total_stickers_given - total_gold_and_silver_stickers) :
  silver_stickers - bronze_stickers = 20 :=
by
  sorry

end fewer_bronze_stickers_l122_122956


namespace henry_age_is_29_l122_122976

-- Definitions and conditions
variable (Henry_age Jill_age : ℕ)

-- Condition 1: Sum of the present age of Henry and Jill is 48
def sum_of_ages : Prop := Henry_age + Jill_age = 48

-- Condition 2: Nine years ago, Henry was twice the age of Jill
def age_relation_nine_years_ago : Prop := Henry_age - 9 = 2 * (Jill_age - 9)

-- Theorem to prove
theorem henry_age_is_29 (H: ℕ) (J: ℕ)
  (h1 : sum_of_ages H J) 
  (h2 : age_relation_nine_years_ago H J) : H = 29 :=
by
  sorry

end henry_age_is_29_l122_122976


namespace greatest_product_two_ints_sum_300_l122_122444

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l122_122444


namespace total_movies_in_series_l122_122710

def book_count := 4
def total_books_read := 19
def movies_watched := 7
def movies_to_watch := 10

theorem total_movies_in_series : movies_watched + movies_to_watch = 17 := by
  sorry

end total_movies_in_series_l122_122710


namespace largest_4_digit_integer_congruent_to_25_mod_26_l122_122550

theorem largest_4_digit_integer_congruent_to_25_mod_26 : ∃ x : ℕ, x < 10000 ∧ x ≥ 1000 ∧ x % 26 = 25 ∧ ∀ y : ℕ, y < 10000 ∧ y ≥ 1000 ∧ y % 26 = 25 → y ≤ x := by
  sorry

end largest_4_digit_integer_congruent_to_25_mod_26_l122_122550


namespace greatest_product_l122_122522

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l122_122522


namespace hyperbola_asymptotes_l122_122650

variable (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)

theorem hyperbola_asymptotes (e : ℝ) (h_ecc : e = (Real.sqrt 5) / 2)
  (h_hyperbola : e = Real.sqrt (1 + (b^2 / a^2))) :
  (∀ x : ℝ, y = x * (b / a) ∨ y = -x * (b / a)) :=
by
  -- Here, the proof would follow logically from the given conditions.
  sorry

end hyperbola_asymptotes_l122_122650


namespace walter_zoo_time_l122_122715

theorem walter_zoo_time (S: ℕ) (H1: S + 8 * S + 13 = 130) : S = 13 :=
by sorry

end walter_zoo_time_l122_122715


namespace probability_sqrt_less_than_nine_l122_122571

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def sqrt_less_than_nine (n : ℕ) : Prop := n < 81

theorem probability_sqrt_less_than_nine :
  (∃ total_num favorable_num,
    (total_num = Finset.card (Finset.filter is_two_digit (Finset.range 100)) ∧
     favorable_num = Finset.card (Finset.filter (λ n, is_two_digit n ∧ sqrt_less_than_nine n) (Finset.range 100)) ∧
     (favorable_num : ℚ) / total_num = 71 / 90)) :=
by
  sorry

end probability_sqrt_less_than_nine_l122_122571


namespace math_problem_l122_122292

noncomputable def x : ℝ := (Real.sqrt 5 + 1) / 2
noncomputable def y : ℝ := (Real.sqrt 5 - 1) / 2

theorem math_problem :
    x^3 * y + 2 * x^2 * y^2 + x * y^3 = 5 := 
by
  sorry

end math_problem_l122_122292


namespace solve_m_range_l122_122639

-- Define the propositions
def p (m : ℝ) := m + 1 ≤ 0

def q (m : ℝ) := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- Provide the Lean statement for the problem
theorem solve_m_range (m : ℝ) (hpq_false : ¬ (p m ∧ q m)) (hpq_true : p m ∨ q m) :
  m ≤ -2 ∨ (-1 < m ∧ m < 2) :=
sorry

end solve_m_range_l122_122639


namespace width_of_wall_l122_122101

-- Define the dimensions of a single brick.
def brick_length : ℝ := 25
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the number of bricks.
def num_bricks : ℝ := 6800

-- Define the dimensions of the wall (length and height).
def wall_length : ℝ := 850
def wall_height : ℝ := 600

-- Prove that the width of the wall is 22.5 cm.
theorem width_of_wall : 
  (wall_length * wall_height * 22.5 = num_bricks * (brick_length * brick_width * brick_height)) :=
by
  sorry

end width_of_wall_l122_122101


namespace max_product_two_integers_sum_300_l122_122350

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l122_122350


namespace greatest_product_obtainable_l122_122399

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l122_122399


namespace max_product_of_two_integers_whose_sum_is_300_l122_122504

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l122_122504


namespace slower_whale_length_is_101_25_l122_122216

def length_of_slower_whale (v_i_f v_i_s a_f a_s t : ℝ) : ℝ :=
  let D_f := v_i_f * t + 0.5 * a_f * t^2
  let D_s := v_i_s * t + 0.5 * a_s * t^2
  D_f - D_s

theorem slower_whale_length_is_101_25
  (v_i_f v_i_s a_f a_s t L : ℝ)
  (h1 : v_i_f = 18)
  (h2 : v_i_s = 15)
  (h3 : a_f = 1)
  (h4 : a_s = 0.5)
  (h5 : t = 15)
  (h6 : length_of_slower_whale v_i_f v_i_s a_f a_s t = L) :
  L = 101.25 :=
by
  sorry

end slower_whale_length_is_101_25_l122_122216


namespace square_area_l122_122114

def edge1 (x : ℝ) := 5 * x - 18
def edge2 (x : ℝ) := 27 - 4 * x
def x_val : ℝ := 5

theorem square_area : edge1 x_val = edge2 x_val → (edge1 x_val) ^ 2 = 49 :=
by
  intro h
  -- Proof required here
  sorry

end square_area_l122_122114


namespace max_product_two_integers_l122_122465

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l122_122465


namespace minimize_x_expr_minimized_l122_122008

noncomputable def minimize_x_expr (x : ℝ) : ℝ :=
  x + 4 / (x + 1)

theorem minimize_x_expr_minimized 
  (hx : x > -1) 
  : x = 1 ↔ minimize_x_expr x = minimize_x_expr 1 :=
by
  sorry

end minimize_x_expr_minimized_l122_122008


namespace sum_prime_factors_1170_l122_122223

theorem sum_prime_factors_1170 : 
  let smallest_prime_factor := 2
  let largest_prime_factor := 13
  (smallest_prime_factor + largest_prime_factor) = 15 :=
by
  sorry

end sum_prime_factors_1170_l122_122223


namespace inclination_angle_of_line_l122_122926

theorem inclination_angle_of_line 
  (l : ℝ) (h : l = Real.tan (-π / 6)) : 
  ∀ θ, θ = Real.pi / 2 :=
by
  -- Placeholder proof
  sorry

end inclination_angle_of_line_l122_122926


namespace arithmetic_expression_evaluation_l122_122878

theorem arithmetic_expression_evaluation :
  2 + 8 * 3 - 4 + 7 * 6 / 3 = 36 := by
  sorry

end arithmetic_expression_evaluation_l122_122878


namespace greatest_product_sum_300_l122_122421

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l122_122421


namespace proof_problem_l122_122201

variable {ι : Type} [LinearOrderedField ι]

-- Let A be a family of sets indexed by natural numbers
variables {A : ℕ → Set ι}

-- Hypotheses
def condition1 (A : ℕ → Set ι) : Prop :=
  (⋃ i, A i) = Set.univ

def condition2 (A : ℕ → Set ι) (a : ι) : Prop :=
  ∀ i b c, b > c → b - c ≥ a ^ i → b ∈ A i → c ∈ A i

theorem proof_problem (A : ℕ → Set ι) (a : ι) :
  condition1 A → condition2 A a → 0 < a → a < 2 :=
sorry

end proof_problem_l122_122201


namespace greatest_product_sum_300_l122_122376

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l122_122376


namespace quadratic_has_real_roots_l122_122157

theorem quadratic_has_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m-2) * x^2 - 2 * x + 1 = 0) ↔ m ≤ 3 :=
by sorry

end quadratic_has_real_roots_l122_122157


namespace Jazmin_strip_width_l122_122807

theorem Jazmin_strip_width (a b c : ℕ) (ha : a = 44) (hb : b = 33) (hc : c = 55) : Nat.gcd (Nat.gcd a b) c = 11 := by
  sorry

end Jazmin_strip_width_l122_122807


namespace number_of_adults_in_family_l122_122873

-- Conditions as definitions
def total_apples : ℕ := 1200
def number_of_children : ℕ := 45
def apples_per_child : ℕ := 15
def apples_per_adult : ℕ := 5

-- Calculations based on conditions
def apples_eaten_by_children : ℕ := number_of_children * apples_per_child
def remaining_apples : ℕ := total_apples - apples_eaten_by_children
def number_of_adults : ℕ := remaining_apples / apples_per_adult

-- Proof target: number of adults in Bob's family equals 105
theorem number_of_adults_in_family : number_of_adults = 105 := by
  sorry

end number_of_adults_in_family_l122_122873


namespace snack_eaters_initial_count_l122_122237

-- Define all variables and conditions used in the problem
variables (S : ℕ) (initial_people : ℕ) (new_outsiders_1 : ℕ) (new_outsiders_2 : ℕ) (left_after_first_half : ℕ) (left_after_second_half : ℕ) (remaining_snack_eaters : ℕ)

-- Assign the specific values according to conditions
def conditions := 
  initial_people = 200 ∧
  new_outsiders_1 = 20 ∧
  new_outsiders_2 = 10 ∧
  left_after_first_half = (S + new_outsiders_1) / 2 ∧
  left_after_second_half = left_after_first_half + new_outsiders_2 - 30 ∧
  remaining_snack_eaters = left_after_second_half / 2 ∧
  remaining_snack_eaters = 20

-- State the theorem to prove
theorem snack_eaters_initial_count (S : ℕ) (initial_people new_outsiders_1 new_outsiders_2 left_after_first_half left_after_second_half remaining_snack_eaters : ℕ) :
  conditions S initial_people new_outsiders_1 new_outsiders_2 left_after_first_half left_after_second_half remaining_snack_eaters → S = 100 :=
by sorry

end snack_eaters_initial_count_l122_122237


namespace bill_milk_problem_l122_122249

theorem bill_milk_problem 
  (M : ℚ) 
  (sour_cream_milk : ℚ := M / 4)
  (butter_milk : ℚ := M / 4)
  (whole_milk : ℚ := M / 2)
  (sour_cream_gallons : ℚ := sour_cream_milk / 2)
  (butter_gallons : ℚ := butter_milk / 4)
  (butter_revenue : ℚ := butter_gallons * 5)
  (sour_cream_revenue : ℚ := sour_cream_gallons * 6)
  (whole_milk_revenue : ℚ := whole_milk * 3)
  (total_revenue : ℚ := butter_revenue + sour_cream_revenue + whole_milk_revenue)
  (h : total_revenue = 41) :
  M = 16 :=
by
  sorry

end bill_milk_problem_l122_122249


namespace probability_point_in_sphere_eq_2pi_div_3_l122_122109

open Real Topology

noncomputable def volume_of_region := 4 * 2 * 2

noncomputable def volume_of_sphere_radius_2 : ℝ :=
  (4 / 3) * π * (2 ^ 3)

noncomputable def probability_in_sphere : ℝ :=
  volume_of_sphere_radius_2 / volume_of_region

theorem probability_point_in_sphere_eq_2pi_div_3 :
  probability_in_sphere = (2 * π) / 3 :=
by
  sorry

end probability_point_in_sphere_eq_2pi_div_3_l122_122109


namespace team_b_wins_first_game_probability_l122_122060

/-- Team A and Team B play a series where the first team to win four games wins the series.
Each team is equally likely to win each game (probability 1/2), there are no ties, 
and the outcomes of the individual games are independent. 
If Team B wins the third game and Team A wins the series, 
prove that the probability that Team B wins the first game is 2/3. -/
theorem team_b_wins_first_game_probability : 
  ∀ (A B : Type) [ProbSpace A B] (win : A → B → Prop), 
  (∀ (X Y : A → B → Prop), P(X) = 1/2 ∧ P(Y) = 1/2) →
  (prob_series_wins : ∀ (X : A → B → Prop), prob_wins_series A = 4 ∧ prob_wins_games B 3) →
  (independent_games : ∀ (X Y : A → B → Prop), independent_trials X Y) →
  (no_ties : ∀ (X : A → B → Prop), X ≠ Y) →
  (P(team_b_wins_first_game | team_a_wins_series ∧ team_b_wins_third_game) = 2/3) :=
begin
  sorry
end

end team_b_wins_first_game_probability_l122_122060


namespace contrapositive_l122_122064

variable (Line Circle : Type) (distance : Line → Circle → ℝ) (radius : Circle → ℝ)
variable (is_tangent : Line → Circle → Prop)

-- Original proposition in Lean notation:
def original_proposition (l : Line) (c : Circle) : Prop :=
  distance l c ≠ radius c → ¬ is_tangent l c

-- Contrapositive of the original proposition:
theorem contrapositive (l : Line) (c : Circle) : Prop :=
  is_tangent l c → distance l c = radius c

end contrapositive_l122_122064


namespace max_product_300_l122_122536

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l122_122536


namespace f_odd_f_monotonic_increasing_intervals_f_no_max_value_f_extreme_points_l122_122018

noncomputable def f (x : ℝ) : ℝ := x^3 - x

theorem f_odd (x : ℝ) : f (-x) = -f (x) :=
by sorry

theorem f_monotonic_increasing_intervals :
  ∀ x : ℝ, (x < -Real.sqrt 3 / 3 ∨ x > Real.sqrt 3 / 3) → f x' > f x :=
by sorry

theorem f_no_max_value :
  ∀ x : ℝ, ¬(∃ M, f x ≤ M) :=
by sorry

theorem f_extreme_points :
  f (-Real.sqrt 3 / 3) = -2 * Real.sqrt 3 / 9 ∧ f (Real.sqrt 3 / 3) = -2 * Real.sqrt 3 / 9 :=
by sorry

end f_odd_f_monotonic_increasing_intervals_f_no_max_value_f_extreme_points_l122_122018


namespace probability_sqrt_lt_9_l122_122568

theorem probability_sqrt_lt_9 : 
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
in probability = 71 / 90 :=
by
  let two_digit_numbers := finset.range 100 \ finset.range 10
  let favorable_numbers := finset.Ico 10 81
  let probability := (favorable_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  sorry

end probability_sqrt_lt_9_l122_122568


namespace product_of_solutions_l122_122890

theorem product_of_solutions : (∃ x : ℝ, |x| = 3*(|x| - 2)) → (x = 3 ∨ x = -3) → 3 * -3 = -9 :=
by sorry

end product_of_solutions_l122_122890


namespace jane_average_speed_correct_l122_122070

noncomputable def jane_average_speed : ℝ :=
  let total_distance : ℝ := 250
  let total_time : ℝ := 6
  total_distance / total_time

theorem jane_average_speed_correct : jane_average_speed = 41.67 := by
  sorry

end jane_average_speed_correct_l122_122070


namespace shaded_fraction_l122_122733

theorem shaded_fraction (rectangle_length rectangle_width : ℕ) (h_length : rectangle_length = 15) (h_width : rectangle_width = 20)
                        (total_area : ℕ := rectangle_length * rectangle_width)
                        (shaded_quarter : ℕ := total_area / 4)
                        (h_shaded_quarter : shaded_quarter = total_area / 5) :
  shaded_quarter / total_area = 1 / 5 :=
by
  sorry

end shaded_fraction_l122_122733


namespace rocking_chair_legs_l122_122977

theorem rocking_chair_legs :
  let tables_4legs := 4 * 4
  let sofa_4legs := 1 * 4
  let chairs_4legs := 2 * 4
  let tables_3legs := 3 * 3
  let table_1leg := 1 * 1
  let total_legs := 40
  let accounted_legs := tables_4legs + sofa_4legs + chairs_4legs + tables_3legs + table_1leg
  ∃ rocking_chair_legs : Nat, total_legs = accounted_legs + rocking_chair_legs ∧ rocking_chair_legs = 2 :=
sorry

end rocking_chair_legs_l122_122977


namespace remainder_3042_div_29_l122_122716

theorem remainder_3042_div_29 : 3042 % 29 = 26 := by
  sorry

end remainder_3042_div_29_l122_122716


namespace problem_condition_problem_statement_l122_122880

noncomputable def a : ℕ → ℕ 
| 0     => 2
| (n+1) => 3 * a n

noncomputable def S : ℕ → ℕ
| 0     => 0
| (n+1) => S n + a n

theorem problem_condition : ∀ n, 3 * a n - 2 * S n = 2 :=
by
  sorry

theorem problem_statement (n : ℕ) (h : ∀ n, 3 * a n - 2 * S n = 2) :
  (S (n+1))^2 - (S n) * (S (n+2)) = 4 * 3^n :=
by
  sorry

end problem_condition_problem_statement_l122_122880


namespace problem1_problem2_l122_122875

theorem problem1 (a b : ℝ) : ((a * b) ^ 6 / (a * b) ^ 2 * (a * b) ^ 4) = a^8 * b^8 := 
by sorry

theorem problem2 (x : ℝ) : ((3 * x^3)^2 * x^5 - (-x^2)^6 / x) = 8 * x^11 :=
by sorry

end problem1_problem2_l122_122875


namespace zoey_finishes_on_wednesday_l122_122856

noncomputable def day_zoey_finishes (n : ℕ) : String :=
  let total_days := (n * (n + 1)) / 2
  match total_days % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | 6 => "Saturday"
  | _ => "Error"

theorem zoey_finishes_on_wednesday : day_zoey_finishes 18 = "Wednesday" :=
by
  -- Calculate that Zoey takes 171 days to read 18 books
  -- Recall that 171 mod 7 = 3, so she finishes on "Wednesday"
  sorry

end zoey_finishes_on_wednesday_l122_122856


namespace length_of_each_piece_cm_l122_122655

theorem length_of_each_piece_cm 
  (total_length : ℝ) 
  (number_of_pieces : ℕ) 
  (htotal : total_length = 17) 
  (hpieces : number_of_pieces = 20) : 
  (total_length / number_of_pieces) * 100 = 85 := 
by
  sorry

end length_of_each_piece_cm_l122_122655


namespace parabola_through_points_with_h_l122_122276

noncomputable def quadratic_parabola (a h k x : ℝ) : ℝ := a * (x - h)^2 + k

theorem parabola_through_points_with_h (
    a h k : ℝ) 
    (H0 : quadratic_parabola a h k 0 = 4)
    (H1 : quadratic_parabola a h k 6 = 5)
    (H2 : a < 0)
    (H3 : 0 < h)
    (H4 : h < 6) : 
    h = 4 := 
sorry

end parabola_through_points_with_h_l122_122276


namespace volume_of_fifth_section_l122_122965

variables {a₁ d : ℝ}

theorem volume_of_fifth_section (h1 : 4 * a₁ + 6 * d = 3) 
                               (h2 : 3 * a₁ + 21 * d = 4) :
  a₅ = a₁ + 4 * d :=
begin
  let a₅ := a₁ + 4 * d,
  have ha₅ : a₅ = a₁ + 4 * d, by refl,
  sorry
end

end volume_of_fifth_section_l122_122965


namespace trader_profit_l122_122740

theorem trader_profit (donation goal extra profit : ℝ) (half_profit : ℝ) 
  (H1 : donation = 310) (H2 : goal = 610) (H3 : extra = 180)
  (H4 : half_profit = profit / 2) 
  (H5 : half_profit + donation = goal + extra) : 
  profit = 960 := 
by
  sorry

end trader_profit_l122_122740


namespace blue_tshirts_in_pack_l122_122313

theorem blue_tshirts_in_pack
  (packs_white : ℕ := 2) 
  (white_per_pack : ℕ := 5) 
  (packs_blue : ℕ := 4)
  (cost_per_tshirt : ℕ := 3)
  (total_cost : ℕ := 66)
  (B : ℕ := 3) :
  (packs_white * white_per_pack * cost_per_tshirt) + (packs_blue * B * cost_per_tshirt) = total_cost := 
by
  sorry

end blue_tshirts_in_pack_l122_122313


namespace blueberry_pies_count_l122_122295

-- Definitions and conditions
def total_pies := 30
def ratio_parts := 10
def pies_per_part := total_pies / ratio_parts
def blueberry_ratio := 3

-- Problem statement
theorem blueberry_pies_count :
  blueberry_ratio * pies_per_part = 9 := by
  -- The solution step that leads to the proof
  sorry

end blueberry_pies_count_l122_122295


namespace greatest_product_sum_300_l122_122365

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l122_122365


namespace exists_natural_sum_of_squares_l122_122679

theorem exists_natural_sum_of_squares : ∃ n : ℕ, n^2 = 0^2 + 7^2 + 24^2 + 312^2 + 48984^2 :=
by {
  sorry
}

end exists_natural_sum_of_squares_l122_122679


namespace gcd_lcm_sum_l122_122577

theorem gcd_lcm_sum :
  Nat.gcd 44 64 + Nat.lcm 48 18 = 148 := 
by
  sorry

end gcd_lcm_sum_l122_122577


namespace proof_n_eq_neg2_l122_122699

theorem proof_n_eq_neg2 (n : ℤ) (h : |n + 6| = 2 - n) : n = -2 := 
by
  sorry

end proof_n_eq_neg2_l122_122699


namespace smallest_x_for_1980_power4_l122_122892

theorem smallest_x_for_1980_power4 (M : ℤ) (x : ℕ) (hx : x > 0) :
  (1980 * (x : ℤ)) = M^4 → x = 6006250 :=
by
  -- The proof goes here
  sorry

end smallest_x_for_1980_power4_l122_122892


namespace miley_discount_rate_l122_122049

theorem miley_discount_rate :
  let cost_per_cellphone := 800
  let number_of_cellphones := 2
  let amount_paid := 1520
  let total_cost_without_discount := cost_per_cellphone * number_of_cellphones
  let discount_amount := total_cost_without_discount - amount_paid
  let discount_rate := (discount_amount / total_cost_without_discount) * 100
  discount_rate = 5 := by
    sorry

end miley_discount_rate_l122_122049


namespace bruno_pens_l122_122117

def dozen := 12
def two_and_one_half_dozens := 2.5

theorem bruno_pens : 2.5 * dozen = 30 := sorry

end bruno_pens_l122_122117


namespace distance_proof_l122_122864

noncomputable section

open Real

-- Define the given conditions
def AB : Real := 3 * sqrt 3
def BC : Real := 2
def theta : Real := 60 -- angle in degrees
def phi : Real := 180 - theta -- supplementary angle to use in the Law of Cosines

-- Helper function to convert degrees to radians
def deg_to_rad (d : Real) : Real := d * (π / 180)

-- Define the law of cosines to compute AC
def distance_AC (AB BC θ : Real) : Real := 
  sqrt (AB^2 + BC^2 - 2 * AB * BC * cos (deg_to_rad θ))

-- The theorem to prove
theorem distance_proof : distance_AC AB BC phi = 7 :=
by
  sorry

end distance_proof_l122_122864


namespace max_product_of_two_integers_whose_sum_is_300_l122_122507

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l122_122507


namespace differentiable_difference_constant_l122_122039

variable {R : Type*} [AddCommGroup R] [Module ℝ R]

theorem differentiable_difference_constant (f g : ℝ → ℝ) 
  (hf : Differentiable ℝ f) (hg : Differentiable ℝ g) 
  (h : ∀ x, fderiv ℝ f x = fderiv ℝ g x) : 
  ∃ C : ℝ, ∀ x, f x - g x = C := 
sorry

end differentiable_difference_constant_l122_122039


namespace greatest_product_obtainable_l122_122405

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l122_122405


namespace relationship_between_a_and_b_l122_122765

variable {a b : ℝ} (n : ℕ)

theorem relationship_between_a_and_b (h₁ : a^n = a + 1) (h₂ : b^(2 * n) = b + 3 * a)
  (h₃ : 2 ≤ n) (h₄ : 1 < a) (h₅ : 1 < b) : a > b ∧ b > 1 :=
by
  sorry

end relationship_between_a_and_b_l122_122765


namespace x_must_be_even_l122_122831

theorem x_must_be_even (x : ℤ) (h : ∃ (n : ℤ), (2 * x / 3 - x / 6) = n) : ∃ (k : ℤ), x = 2 * k :=
by
  sorry

end x_must_be_even_l122_122831


namespace maximum_tied_teams_in_tournament_l122_122671

theorem maximum_tied_teams_in_tournament : 
  ∀ (n : ℕ), n = 8 →
  (∀ (wins : ℕ), wins = (n * (n - 1)) / 2 →
   ∃ (k : ℕ), k ≤ n ∧ (k > 7 → false) ∧ 
               (∃ (w : ℕ), k * w = wins)) :=
by
  intros n hn wins hw
  use 7
  split
  · exact (by linarith)
  · intro h
    exfalso
    exact h (by linarith)
  · use 4
    calc
      7 * 4 = 28 : by norm_num
      ... = 28 : by rw hw; linarith
  
-- The proof is omitted as per instructions ("sorry" can be used to indicate this).

end maximum_tied_teams_in_tournament_l122_122671


namespace cooper_age_l122_122842

variable (Cooper Dante Maria : ℕ)

-- Conditions
def sum_of_ages : Prop := Cooper + Dante + Maria = 31
def dante_twice_cooper : Prop := Dante = 2 * Cooper
def maria_one_year_older : Prop := Maria = Dante + 1

theorem cooper_age (h1 : sum_of_ages Cooper Dante Maria) (h2 : dante_twice_cooper Cooper Dante) (h3 : maria_one_year_older Dante Maria) : Cooper = 6 :=
by
  sorry

end cooper_age_l122_122842


namespace sin_sq_minus_2_cos_sq_eq_sin_minus_cos_eq_l122_122140

open Real

/-- Given that tan(α) = 2 and π < α < 3π/2, prove that sin(α)^2 - 2 * cos(α)^2 = 2/5. -/
theorem sin_sq_minus_2_cos_sq_eq (α : ℝ) (h1 : tan α = 2) (h2 : π < α ∧ α < 3 * π / 2) :
  sin α ^ 2 - 2 * cos α ^ 2 = 2 / 5 := by
  sorry

/-- Given that tan(α) = 2 and π < α < 3π/2, prove that sin(α) - cos(α) = -√5/5. -/
theorem sin_minus_cos_eq (α : ℝ) (h1 : tan α = 2) (h2 : π < α ∧ α < 3 * π / 2) :
  sin α - cos α = -√5 / 5 := by
  sorry

end sin_sq_minus_2_cos_sq_eq_sin_minus_cos_eq_l122_122140


namespace probability_sqrt_less_than_nine_l122_122555

/-- Define the set of two-digit integers --/
def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n ≤ 99

/-- Define the condition that the square root of the number is less than 9 --/
def sqrt_less_than_nine (n : Nat) : Prop := n < 81

/-- The number of integers from 10 to 80 --/
lemma count_satisfying_sqrt (n : Nat) : Prop :=
  is_two_digit n ∧ sqrt_less_than_nine n → n < 81

/-- Total number of two-digit integers --/
lemma count_two_digit_total (n : Nat) : Prop := is_two_digit n 

/-- The probability that a randomly selected two-digit integer's square root is less than 9. --/
theorem probability_sqrt_less_than_nine : 
  (∃ n, count_satisfying_sqrt n) / (∃ n, count_two_digit_total n) = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l122_122555


namespace geometric_sequence_eighth_term_l122_122619

theorem geometric_sequence_eighth_term 
  (a : ℕ) (r : ℕ) (h1 : a = 4) (h2 : r = 16 / 4) :
  a * r^(7) = 65536 :=
by
  sorry

end geometric_sequence_eighth_term_l122_122619


namespace sonic_leads_by_19_2_meters_l122_122665

theorem sonic_leads_by_19_2_meters (v_S v_D : ℝ)
  (h1 : ∀ t, t = 200 / v_S → 200 = v_S * t)
  (h2 : ∀ t, t = 184 / v_D → 184 = v_D * t)
  (h3 : v_S / v_D = 200 / 184)
  :  240 / v_S - (200 / v_S / (200 / 184) * 240) = 19.2 := by
  sorry

end sonic_leads_by_19_2_meters_l122_122665


namespace convert_to_scientific_notation_l122_122971

def original_value : ℝ := 3462.23
def scientific_notation_value : ℝ := 3.46223 * 10^3

theorem convert_to_scientific_notation : 
  original_value = scientific_notation_value :=
sorry

end convert_to_scientific_notation_l122_122971


namespace average_of_combined_results_l122_122723

theorem average_of_combined_results {avg1 avg2 n1 n2 : ℝ} (h1 : avg1 = 28) (h2 : avg2 = 55) (h3 : n1 = 55) (h4 : n2 = 28) :
  ((n1 * avg1) + (n2 * avg2)) / (n1 + n2) = 37.11 :=
by sorry

end average_of_combined_results_l122_122723


namespace minimum_value_omega_l122_122946

variable (f : ℝ → ℝ) (ω ϕ T : ℝ) (x : ℝ)
variable (h_zero : 0 < ω) (h_phi_range : 0 < ϕ ∧ ϕ < π)
variable (h_period : T = 2 * π / ω)
variable (h_f_period : f T = sqrt 3 / 2)
variable (h_zero_of_f : f (π / 9) = 0)
variable (h_f_def : ∀ x, f x = cos (ω * x + ϕ))

theorem minimum_value_omega : ω = 3 := by sorry

end minimum_value_omega_l122_122946


namespace find_a_tangent_to_curve_l122_122781

theorem find_a_tangent_to_curve (a : ℝ) :
  (∃ (x₀ : ℝ), y = x - 1 ∧ y = e^(x + a) ∧ (e^(x₀ + a) = 1)) → a = -2 :=
by
  sorry

end find_a_tangent_to_curve_l122_122781


namespace sampling_scheme_exists_l122_122254

theorem sampling_scheme_exists : 
  ∃ (scheme : List ℕ → List (List ℕ)), 
    ∀ (p : List ℕ), p.length = 100 → (scheme p).length = 20 :=
by
  sorry

end sampling_scheme_exists_l122_122254


namespace probability_sqrt_lt_9_of_two_digit_l122_122559

-- Define the set of two-digit whole numbers
def two_digit_whole_numbers : set ℕ := {n | 10 ≤ n ∧ n ≤ 99}

-- Define the predicate that checks if the square root of a number is less than 9
def sqrt_lt_9 (n : ℕ) : Prop := (n : ℝ)^2 < (9 : ℝ)^2

-- Calculate the probability
theorem probability_sqrt_lt_9_of_two_digit :
  let eligible_numbers := { n ∈ two_digit_whole_numbers | sqrt_lt_9 n } in
  (eligible_numbers.to_finset.card : ℚ) / (two_digit_whole_numbers.to_finset.card : ℚ) =
  71 / 90 :=
by
  sorry

end probability_sqrt_lt_9_of_two_digit_l122_122559


namespace second_number_is_30_l122_122996

-- Definitions from the conditions
def second_number (x : ℕ) := x
def first_number (x : ℕ) := 2 * x
def third_number (x : ℕ) := (2 * x) / 3
def sum_of_numbers (x : ℕ) := first_number x + second_number x + third_number x

-- Lean statement
theorem second_number_is_30 (x : ℕ) (h1 : sum_of_numbers x = 110) : x = 30 :=
by
  sorry

end second_number_is_30_l122_122996


namespace meal_combinations_l122_122087

theorem meal_combinations (n : ℕ) (h : n = 12) : ∃ m : ℕ, m = 132 :=
by
  -- Initialize the variables for dishes chosen by Yann and Camille
  let yann_choices := n
  let camille_choices := n - 1
  
  -- Calculate the total number of combinations
  let total_combinations := yann_choices * camille_choices
  
  -- Assert the number of combinations is equal to 132
  use total_combinations
  exact sorry

end meal_combinations_l122_122087


namespace solve_for_y_l122_122124

theorem solve_for_y (x y : ℝ) (h : 2 * y - 4 * x + 5 = 0) : y = 2 * x - 2.5 :=
sorry

end solve_for_y_l122_122124


namespace solution_l122_122290

noncomputable def prove_a_greater_than_3 : Prop :=
  ∀ (x : ℝ) (a : ℝ), (a > 0) → (|x - 2| + |x - 3| + |x - 4| < a) → a > 3

theorem solution : prove_a_greater_than_3 :=
by
  intros x a h_pos h_ineq
  sorry

end solution_l122_122290


namespace exists_increasing_triplet_l122_122035

theorem exists_increasing_triplet (f : ℕ → ℕ) (bij : Function.Bijective f) :
  ∃ (a d : ℕ), 0 < a ∧ 0 < d ∧ f a < f (a + d) ∧ f (a + d) < f (a + 2 * d) :=
by
  sorry

end exists_increasing_triplet_l122_122035


namespace max_product_of_sum_300_l122_122476

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l122_122476


namespace value_of_m_l122_122275

-- Definitions of the conditions
def base6_num (m : ℕ) : ℕ := 2 + m * 6^2
def dec_num (d : ℕ) := d = 146

-- Theorem to prove
theorem value_of_m (m : ℕ) (h1 : base6_num m = 146) : m = 4 := 
sorry

end value_of_m_l122_122275


namespace fraction_complex_z_l122_122154

theorem fraction_complex_z (z : ℂ) (hz : z = 1 - I) : 2 / z = 1 + I := by
    sorry

end fraction_complex_z_l122_122154


namespace inequality_proof_l122_122780

variable (a b : Real)
variable (θ : Real)

-- Line equation and point condition
def line_eq := ∀ x y, x / a + y / b = 1 → (x, y) = (Real.cos θ, Real.sin θ)
-- Main theorem to prove
theorem inequality_proof : (line_eq a b θ) → 1 / (a^2) + 1 / (b^2) ≥ 1 := sorry

end inequality_proof_l122_122780


namespace greatest_product_of_sum_eq_300_l122_122384

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l122_122384


namespace find_dimensions_l122_122032

-- Define the conditions
def perimeter (x y : ℕ) : Prop := (2 * (x + y) = 3996)
def divisible_parts (x y k : ℕ) : Prop := (x * y = 1998 * k) ∧ ∃ (k : ℕ), (k * 1998 = x * y) ∧ k ≠ 0

-- State the theorem
theorem find_dimensions (x y : ℕ) (k : ℕ) : perimeter x y ∧ divisible_parts x y k → (x = 1332 ∧ y = 666) ∨ (x = 666 ∧ y = 1332) :=
by
  -- This is where the proof would go.
  sorry

end find_dimensions_l122_122032


namespace greatest_product_of_sum_eq_300_l122_122395

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l122_122395


namespace sum_of_squares_l122_122920

theorem sum_of_squares (x y : ℝ) (h1 : (x + y) ^ 2 = 4) (h2 : x * y = -1) :
  x^2 + y^2 = 6 :=
by
  sorry

end sum_of_squares_l122_122920


namespace company_management_structure_l122_122102

theorem company_management_structure : 
  let num_employees := 13
  let choose (n k : ℕ) := nat.choose n k
  let ways_to_chose_CEO := num_employees
  let remaining_after_CEO := num_employees - 1
  let ways_to_choose_VPs := choose remaining_after_CEO 2
  let remaining_after_VPs := remaining_after_CEO - 2
  let ways_to_choose_managers_VP1 := choose remaining_after_VPs 3
  let remaining_after_VP1_mgrs := remaining_after_VPs - 3
  let ways_to_choose_managers_VP2 := choose remaining_after_VP1_mgrs 3
  let total_ways := ways_to_chose_CEO * ways_to_choose_VPs * ways_to_choose_managers_VP1 * ways_to_choose_managers_VP2
  total_ways = 349800 := by
    sorry

end company_management_structure_l122_122102


namespace greatest_product_of_sum_eq_300_l122_122389

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l122_122389


namespace ellipse_eccentricity_l122_122646

theorem ellipse_eccentricity (a : ℝ) :
  (∀ x y : ℝ, (x^2) / (a^2) + (y^2) / 16 = 1) ∧ (∃ e : ℝ, e = 3 / 4) ∧ (∀ c : ℝ, c = 3 / 4)
   → a = 7 :=
by
  sorry

end ellipse_eccentricity_l122_122646


namespace find_last_year_rate_l122_122324

-- Define the problem setting with types and values (conditions)
def last_year_rate (r : ℝ) : Prop := 
  -- Let r be the annual interest rate last year
  1.1 * r = 0.09

-- Define the theorem to prove the interest rate last year given this year's rate
theorem find_last_year_rate :
  ∃ r : ℝ, last_year_rate r ∧ r = 0.09 / 1.1 := 
by
  sorry

end find_last_year_rate_l122_122324


namespace train_crossing_time_l122_122741

def train_length : ℝ := 150
def train_speed : ℝ := 179.99999999999997

theorem train_crossing_time : train_length / train_speed = 0.8333333333333333 := by
  sorry

end train_crossing_time_l122_122741


namespace triangle_inequality_l122_122684

theorem triangle_inequality (a b c p q r : ℝ) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_sum_zero : p + q + r = 0) : 
  a^2 * p * q + b^2 * q * r + c^2 * r * p ≤ 0 := 
sorry

end triangle_inequality_l122_122684


namespace quadratic_inequality_l122_122652

theorem quadratic_inequality (a b c : ℝ)
  (h1 : ∀ x : ℝ, x = -2 → y = 8)
  (h2 : ∀ x : ℝ, x = -1 → y = 3)
  (h3 : ∀ x : ℝ, x = 0 → y = 0)
  (h4 : ∀ x : ℝ, x = 1 → y = -1)
  (h5 : ∀ x : ℝ, x = 2 → y = 0)
  (h6 : ∀ x : ℝ, x = 3 → y = 3)
  : ∀ x : ℝ, (y - 3 > 0) ↔ x < -1 ∨ x > 3 :=
sorry

end quadratic_inequality_l122_122652


namespace ellipse_equation_l122_122782

theorem ellipse_equation
  (a b : ℝ)
  (a_pos : a > 0)
  (b_pos : b > 0)
  (a_gt_b : a > b)
  (eccentricity : ℝ)
  (eccentricity_eq : eccentricity = (Real.sqrt 3 / 3))
  (perimeter_triangle : ℝ)
  (perimeter_eq : perimeter_triangle = 4 * Real.sqrt 3) :
  a = Real.sqrt 3 ∧ b = Real.sqrt 2 ∧ (a > b) ∧ (eccentricity = 1 / Real.sqrt 3) →
  (∀ x y : ℝ, (x^2 / 3 + y^2 / 2 = 1)) :=
by
  sorry

end ellipse_equation_l122_122782


namespace price_of_table_l122_122719

variable (C T : ℝ)

theorem price_of_table :
  2 * C + T = 0.6 * (C + 2 * T) ∧
  C + T = 96 →
  T = 84 := by
sorry

end price_of_table_l122_122719


namespace find_x_l122_122286

-- Given condition
def condition (x : ℝ) : Prop := 3 * x - 5 * x + 8 * x = 240

-- Statement (problem to prove)
theorem find_x (x : ℝ) (h : condition x) : x = 40 :=
by 
  sorry

end find_x_l122_122286


namespace no_equilateral_triangle_on_grid_regular_tetrahedron_on_grid_l122_122583

-- Define the context for part (a)
theorem no_equilateral_triangle_on_grid (x1 y1 x2 y2 x3 y3 : ℤ) :
  ¬ (x1 = x2 ∧ y1 = y2) ∧ (x2 = x3 ∧ y2 = y3) ∧ (x3 = x1 ∧ y3 = y1) ∧ -- vertices must not be the same
  ((x2 - x1)^2 + (y2 - y1)^2 = (x3 - x2)^2 + (y3 - y2)^2) ∧ -- sides must be equal
  ((x3 - x1)^2 + (y3 - y1)^2 = (x2 - x1)^2 + (y2 - y1)^2) ->
  false := 
sorry

-- Define the context for part (b)
theorem regular_tetrahedron_on_grid (x1 y1 z1 x2 y2 z2 x3 y3 z3 x4 y4 z4 : ℤ) :
  ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2 = (x3 - x2)^2 + (y3 - y2)^2 + (z3 - z2)^2) ∧ -- first condition: edge lengths equal
  ((x3 - x1)^2 + (y3 - y1)^2 + (z3 - z1)^2 = (x4 - x3)^2 + (y4 - y3)^2 + (z4 - z3)^2) ∧ -- second condition: edge lengths equal
  ((x4 - x1)^2 + (y4 - y1)^2 + (z4 - z1)^2 = (x2 - x4)^2 + (y2 - y4)^2 + (z2 - z4)^2) -> -- third condition: edge lengths equal
  true := 
sorry

end no_equilateral_triangle_on_grid_regular_tetrahedron_on_grid_l122_122583


namespace intersection_points_number_of_regions_l122_122772

-- Given n lines on a plane, any two of which are not parallel
-- and no three of which intersect at the same point,
-- prove the number of intersection points of these lines

theorem intersection_points (n : ℕ) (h_n : 0 < n) : 
  ∃ a_n : ℕ, a_n = n * (n - 1) / 2 := by
  sorry

-- Given n lines on a plane, any two of which are not parallel
-- and no three of which intersect at the same point,
-- prove the number of regions these lines form

theorem number_of_regions (n : ℕ) (h_n : 0 < n) :
  ∃ R_n : ℕ, R_n = n * (n + 1) / 2 + 1 := by
  sorry

end intersection_points_number_of_regions_l122_122772


namespace max_product_of_sum_300_l122_122425

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l122_122425


namespace second_term_arithmetic_sequence_l122_122973

theorem second_term_arithmetic_sequence 
  (a d : ℤ)
  (h1 : a + 15 * d = 8)
  (h2 : a + 16 * d = 10) : 
  a + d = -20 := 
by sorry

end second_term_arithmetic_sequence_l122_122973


namespace solve_system_of_equations_solve_fractional_equation_l122_122700

noncomputable def solution1 (x y : ℚ) := (3 * x - 5 * y = 3) ∧ (x / 2 - y / 3 = 1) ∧ (x = 8 / 3) ∧ (y = 1)

noncomputable def solution2 (x : ℚ) := (x / (x - 1) + 1 = 3 / (2 * x - 2)) ∧ (x = 5 / 4)

theorem solve_system_of_equations (x y : ℚ) : solution1 x y := by
  sorry

theorem solve_fractional_equation (x : ℚ) : solution2 x := by
  sorry

end solve_system_of_equations_solve_fractional_equation_l122_122700


namespace product_of_solutions_abs_eq_product_of_solutions_l122_122889

theorem product_of_solutions_abs_eq (x : ℝ) (h : |x| = 3 * (|x| - 2)) : x = 3 ∨ x = -3 :=
by
  sorry

theorem product_of_solutions (x1 x2 : ℝ) (h1 : |x1| = 3 * (|x1| - 2)) (h2 : |x2| = 3 * (|x2| - 2)) :
  x1 * x2 = -9 :=
by
  have hx1 : x1 = 3 ∨ x1 = -3 := product_of_solutions_abs_eq x1 h1
  have hx2 : x2 = 3 ∨ x2 = -3 := product_of_solutions_abs_eq x2 h2
  cases hx1
  case Or.inl hxl1 =>
    cases hx2
    case Or.inl hxr1 =>
      exact False.elim (by sorry)
    case Or.inr hxr2 =>
      rw [hxl1, hxr2]
      norm_num
  case Or.inr hxl2 =>
    cases hx2
    case Or.inl hxr1 =>
      rw [hxl2, hxr1]
      norm_num
    case Or.inr hxr2 =>
      exact False.elim (by sorry)

end product_of_solutions_abs_eq_product_of_solutions_l122_122889


namespace ratio_of_areas_l122_122793

theorem ratio_of_areas
  (R_X R_Y : ℝ)
  (h : (60 / 360) * 2 * Real.pi * R_X = (40 / 360) * 2 * Real.pi * R_Y) :
  (Real.pi * R_X^2) / (Real.pi * R_Y^2) = 4 / 9 :=
by
  sorry

end ratio_of_areas_l122_122793


namespace cos_double_angle_l122_122153

theorem cos_double_angle (α : ℝ) (h : Real.sin α = (Real.sqrt 3) / 2) : 
  Real.cos (2 * α) = -1 / 2 :=
by
  sorry

end cos_double_angle_l122_122153


namespace max_d_value_l122_122760

theorem max_d_value (d f : ℕ) (hd : d ∈ finset.range 10) (hf : f ∈ finset.range 10)
  (h_div3 : (18 + d + f) % 3 = 0) (h_div11 : (15 - (d + f)) % 11 = 0) :
  d ≤ 9 :=
by {
  sorry
}

end max_d_value_l122_122760


namespace product_relationship_l122_122894

variable {a_1 a_2 b_1 b_2 : ℝ}

theorem product_relationship (h1 : a_1 < a_2) (h2 : b_1 < b_2) : 
  a_1 * b_1 + a_2 * b_2 > a_1 * b_2 + a_2 * b_1 := 
sorry

end product_relationship_l122_122894


namespace percentage_x_eq_six_percent_y_l122_122156

variable {x y : ℝ}

theorem percentage_x_eq_six_percent_y (h1 : ∃ P : ℝ, (P / 100) * x = (6 / 100) * y)
  (h2 : (18 / 100) * x = (9 / 100) * y) : 
  ∃ P : ℝ, P = 12 := 
sorry

end percentage_x_eq_six_percent_y_l122_122156


namespace expression_equals_one_l122_122046

theorem expression_equals_one (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) (h_sum : a + b + c = 1) :
  (a^3 * b^3 / ((a^3 - b * c) * (b^3 - a * c)) + a^3 * c^3 / ((a^3 - b * c) * (c^3 - a * b)) +
    b^3 * c^3 / ((b^3 - a * c) * (c^3 - a * b))) = 1 :=
by
  sorry

end expression_equals_one_l122_122046


namespace circle_center_and_radius_sum_l122_122882

theorem circle_center_and_radius_sum :
  let a := -4
  let b := -8
  let r := Real.sqrt 17
  a + b + r = -12 + Real.sqrt 17 :=
by
  sorry

end circle_center_and_radius_sum_l122_122882


namespace local_odd_function_range_of_a_l122_122893

variable (f : ℝ → ℝ)
variable (a : ℝ)

def local_odd_function (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (-x₀) = -f x₀

theorem local_odd_function_range_of_a (hf : ∀ x, f x = -a * (2^x) - 4) :
  local_odd_function f → (-4 ≤ a ∧ a < 0) :=
by
  sorry

end local_odd_function_range_of_a_l122_122893


namespace least_isosceles_triangles_cover_rectangle_l122_122866

-- Define the dimensions of the rectangle
def rectangle_height : ℕ := 10
def rectangle_width : ℕ := 100

-- Define the least number of isosceles right triangles needed to cover the rectangle
def least_number_of_triangles (h w : ℕ) : ℕ :=
  if h = rectangle_height ∧ w = rectangle_width then 11 else 0

-- The theorem statement
theorem least_isosceles_triangles_cover_rectangle :
  least_number_of_triangles rectangle_height rectangle_width = 11 :=
by
  -- skip the proof
  sorry

end least_isosceles_triangles_cover_rectangle_l122_122866


namespace no_line_bisected_by_P_exists_l122_122677

theorem no_line_bisected_by_P_exists (P : ℝ × ℝ) (H : ∀ x y : ℝ, (x / 3)^2 - (y / 2)^2 = 1) : 
  P ≠ (2, 1) := 
sorry

end no_line_bisected_by_P_exists_l122_122677


namespace kitchen_upgrade_cost_l122_122213

-- Define the number of cabinet knobs and their cost
def num_knobs : ℕ := 18
def cost_per_knob : ℝ := 2.50

-- Define the number of drawer pulls and their cost
def num_pulls : ℕ := 8
def cost_per_pull : ℝ := 4.00

-- Calculate the total cost of the knobs
def total_cost_knobs : ℝ := num_knobs * cost_per_knob

-- Calculate the total cost of the pulls
def total_cost_pulls : ℝ := num_pulls * cost_per_pull

-- Calculate the total cost of the kitchen upgrade
def total_cost : ℝ := total_cost_knobs + total_cost_pulls

-- Theorem statement
theorem kitchen_upgrade_cost : total_cost = 77 := by
  sorry

end kitchen_upgrade_cost_l122_122213


namespace greatest_product_l122_122527

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l122_122527


namespace frank_hamburger_goal_l122_122266

theorem frank_hamburger_goal:
  let price_per_hamburger := 5
  let group1_hamburgers := 2 * 4
  let group2_hamburgers := 2 * 2
  let current_hamburgers := group1_hamburgers + group2_hamburgers
  let extra_hamburgers_needed := 4
  let total_hamburgers := current_hamburgers + extra_hamburgers_needed
  price_per_hamburger * total_hamburgers = 80 :=
by
  sorry

end frank_hamburger_goal_l122_122266


namespace max_product_two_integers_sum_300_l122_122343

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l122_122343


namespace triangle_perimeter_ABF_l122_122010

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 25) + (y^2 / 21) = 1

-- Define the line
def line (x : ℝ) : Prop := x = -2

-- Define the foci of the ellipse
def right_focus : ℝ := 2
def left_focus : ℝ := -2

-- Points A and B are on the ellipse and line
def point_A (x y : ℝ) : Prop := ellipse x y ∧ line x
def point_B (x y : ℝ) : Prop := ellipse x y ∧ line x

-- Point F is the right focus of the ellipse
def point_F (x y : ℝ) : Prop := x = right_focus ∧ y = 0

-- Perimeter of the triangle ABF
def perimeter (A B F : ℝ × ℝ) : ℝ :=
  sorry -- Calculation of the perimeter of triangle ABF

-- Theorem statement that perimeter is 20
theorem triangle_perimeter_ABF 
  (A B F : ℝ × ℝ) 
  (hA : point_A (A.fst) (A.snd)) 
  (hB : point_B (B.fst) (B.snd))
  (hF : point_F (F.fst) (F.snd)) :
  perimeter A B F = 20 :=
sorry

end triangle_perimeter_ABF_l122_122010


namespace hexagon_chord_problem_solution_l122_122597

noncomputable def hexagon_chord_length {p q : ℕ} (hpq_coprime : Nat.coprime p q) : ℕ :=
  let a := 4
  let b := 6
  let circ_hex := inscribed_hexagon_in_circle a b
  let chord := circ_hex.divides_into_trapezoids
  if (chord.len = p / q) then p + q else 0

theorem hexagon_chord_problem_solution :
  ∃ (p q : ℕ), Nat.coprime p q ∧ hexagon_chord_length Nat.Coprime p q = 799 :=
begin
  sorry
end

end hexagon_chord_problem_solution_l122_122597


namespace water_charge_rel_water_usage_from_charge_l122_122668

-- Define the conditions and functional relationship
theorem water_charge_rel (x : ℝ) (hx : x > 5) : y = 3.5 * x - 7.5 :=
  sorry

-- Prove the specific case where the charge y is 17 yuan
theorem water_usage_from_charge (h : 17 = 3.5 * x - 7.5) :
  x = 7 :=
  sorry

end water_charge_rel_water_usage_from_charge_l122_122668


namespace percentage_calculation_l122_122065

def percentage_less_than_50000_towns : Float := 85

def percentage_less_than_20000_towns : Float := 20
def percentage_20000_to_49999_towns : Float := 65

theorem percentage_calculation :
  percentage_less_than_50000_towns = percentage_less_than_20000_towns + percentage_20000_to_49999_towns :=
by
  sorry

end percentage_calculation_l122_122065


namespace valid_ATM_passwords_l122_122744

theorem valid_ATM_passwords : 
  let total_passwords := 10^4
  let restricted_passwords := 10
  total_passwords - restricted_passwords = 9990 :=
by
  sorry

end valid_ATM_passwords_l122_122744


namespace number_of_people_liking_at_least_one_activity_l122_122164

def total_people := 200
def people_like_books := 80
def people_like_songs := 60
def people_like_movies := 30
def people_like_books_and_songs := 25
def people_like_books_and_movies := 15
def people_like_songs_and_movies := 20
def people_like_all_three := 10

theorem number_of_people_liking_at_least_one_activity :
  total_people = 200 →
  people_like_books = 80 →
  people_like_songs = 60 →
  people_like_movies = 30 →
  people_like_books_and_songs = 25 →
  people_like_books_and_movies = 15 →
  people_like_songs_and_movies = 20 →
  people_like_all_three = 10 →
  (people_like_books + people_like_songs + people_like_movies -
   people_like_books_and_songs - people_like_books_and_movies -
   people_like_songs_and_movies + people_like_all_three) = 120 := sorry

end number_of_people_liking_at_least_one_activity_l122_122164


namespace vector_addition_example_l122_122149

theorem vector_addition_example :
  let a := (1, 2)
  let b := (-2, 1)
  a.1 + 2 * b.1 = -3 ∧ a.2 + 2 * b.2 = 4 :=
by
  sorry

end vector_addition_example_l122_122149


namespace gcd_2023_2048_l122_122221

theorem gcd_2023_2048 : Nat.gcd 2023 2048 = 1 := by
  sorry

end gcd_2023_2048_l122_122221


namespace line_of_intersection_canonical_form_l122_122997

def canonical_form_of_line (A B : ℝ) (x y z : ℝ) :=
  (x / A) = (y / B) ∧ (y / B) = (z)

theorem line_of_intersection_canonical_form :
  ∀ (x y z : ℝ),
  x + y - 2*z - 2 = 0 →
  x - y + z + 2 = 0 →
  canonical_form_of_line (-1) (-3) x (y - 2) (-2) :=
by
  intros x y z h_eq1 h_eq2
  sorry

end line_of_intersection_canonical_form_l122_122997


namespace probability_of_sqrt_lt_9_l122_122576

-- Define the set of two-digit whole numbers
def two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}

-- Define the subset of numbers for which the square root is less than 9
def valid_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 80}

-- Define the probability calculation
noncomputable def probability_sqrt_lt_9 := (valid_numbers.to_finset.card : ℝ) / (two_digit_numbers.to_finset.card : ℝ)

-- The statement we aim to prove
theorem probability_of_sqrt_lt_9 : probability_sqrt_lt_9 = 71 / 90 := 
sorry

end probability_of_sqrt_lt_9_l122_122576


namespace max_product_of_sum_300_l122_122469

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l122_122469


namespace lateral_surface_area_of_rotated_triangle_l122_122270

theorem lateral_surface_area_of_rotated_triangle :
  let AC := 3
  let BC := 4
  let AB := Real.sqrt (AC ^ 2 + BC ^ 2)
  let radius := BC
  let slant_height := AB
  let lateral_surface_area := Real.pi * radius * slant_height
  lateral_surface_area = 20 * Real.pi := by
  sorry

end lateral_surface_area_of_rotated_triangle_l122_122270


namespace probability_same_color_of_two_12_sided_dice_l122_122849

-- Define the conditions
def sides := 12
def red_sides := 3
def blue_sides := 5
def green_sides := 3
def golden_sides := 1

-- Calculate the probabilities for each color being rolled
def pr_both_red := (red_sides / sides) ^ 2
def pr_both_blue := (blue_sides / sides) ^ 2
def pr_both_green := (green_sides / sides) ^ 2
def pr_both_golden := (golden_sides / sides) ^ 2

-- Total probability calculation
def total_probability_same_color := pr_both_red + pr_both_blue + pr_both_green + pr_both_golden

theorem probability_same_color_of_two_12_sided_dice :
  total_probability_same_color = 11 / 36 := by
  sorry

end probability_same_color_of_two_12_sided_dice_l122_122849


namespace perfect_square_trinomial_m_eq_l122_122025

theorem perfect_square_trinomial_m_eq (
    m y : ℝ) (h : ∃ k : ℝ, 4*y^2 - m*y + 25 = (2*y - k)^2) :
  m = 20 ∨ m = -20 :=
by
  sorry

end perfect_square_trinomial_m_eq_l122_122025


namespace greatest_product_of_sum_eq_300_l122_122382

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l122_122382


namespace greatest_product_sum_300_l122_122355

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l122_122355


namespace legendre_symbol_two_l122_122090

theorem legendre_symbol_two (m : ℕ) [Fact (Nat.Prime m)] (hm : Odd m) :
  (legendreSym 2 m) = (-1 : ℤ) ^ ((m^2 - 1) / 8) :=
sorry

end legendre_symbol_two_l122_122090


namespace max_value_2x_minus_y_l122_122927

theorem max_value_2x_minus_y 
  (x y : ℝ) 
  (h1 : x - y + 1 ≥ 0) 
  (h2 : y + 1 ≥ 0) 
  (h3 : x + y + 1 ≤ 0) : 
  2 * x - y ≤ 1 :=
sorry

end max_value_2x_minus_y_l122_122927


namespace least_number_remainder_5_l122_122982

theorem least_number_remainder_5 (n : ℕ) : 
  n % 12 = 5 ∧ n % 15 = 5 ∧ n % 20 = 5 ∧ n % 54 = 5 → n = 545 := 
  by
  sorry

end least_number_remainder_5_l122_122982


namespace max_product_of_sum_300_l122_122434

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l122_122434


namespace exponentiation_and_division_l122_122756

theorem exponentiation_and_division (a b c : ℕ) (h : a = 6) (h₂ : b = 3) (h₃ : c = 15) :
  9^a * 3^b / 3^c = 1 := by
  sorry

end exponentiation_and_division_l122_122756


namespace simple_interest_double_l122_122163

theorem simple_interest_double (P : ℝ) (r : ℝ) (t : ℝ) (A : ℝ)
  (h1 : t = 50)
  (h2 : A = 2 * P) 
  (h3 : A - P = P * r * t / 100) :
  r = 2 :=
by
  -- Proof is omitted
  sorry

end simple_interest_double_l122_122163


namespace greatest_product_sum_300_l122_122419

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l122_122419


namespace sin_2B_value_l122_122636

-- Define the triangle's internal angles and the tangent of angles
variables (A B C : ℝ) 

-- Given conditions from the problem
def tan_sequence (tanA tanB tanC : ℝ) : Prop :=
  tanA = (1/2) * tanB ∧
  tanC = (3/2) * tanB ∧
  2 * tanB = tanC + tanB + (tanC - tanA)

-- The statement to be proven
theorem sin_2B_value (h : tan_sequence (Real.tan A) (Real.tan B) (Real.tan C)) :
  Real.sin (2 * B) = 4 / 5 :=
sorry

end sin_2B_value_l122_122636


namespace find_abs_xyz_l122_122041

noncomputable def distinct_nonzero_real (x y z : ℝ) : Prop :=
x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 

theorem find_abs_xyz
  (x y z : ℝ)
  (h1 : distinct_nonzero_real x y z)
  (h2 : x + 1/y = y + 1/z)
  (h3 : y + 1/z = z + 1/x + 1) :
  |x * y * z| = 1 :=
sorry

end find_abs_xyz_l122_122041


namespace find_m_for_one_real_solution_l122_122763

theorem find_m_for_one_real_solution (m : ℝ) (h : 4 * m * 4 = m^2) : m = 8 := sorry

end find_m_for_one_real_solution_l122_122763


namespace compute_focus_d_l122_122115

-- Define the given conditions as Lean definitions
structure Ellipse (d : ℝ) :=
  (first_quadrant : d > 0)
  (F1 : ℝ × ℝ := (4, 8))
  (F2 : ℝ × ℝ := (d, 8))
  (tangent_x_axis : (d + 4) / 2 > 0)
  (tangent_y_axis : (d + 4) / 2 > 0)

-- Define the proof problem to show d = 6 for the given conditions
theorem compute_focus_d (d : ℝ) (e : Ellipse d) : d = 6 := by
  sorry

end compute_focus_d_l122_122115


namespace river_flow_volume_l122_122739

theorem river_flow_volume (depth width : ℝ) (flow_rate_kmph : ℝ) :
  depth = 3 → width = 36 → flow_rate_kmph = 2 → 
  (depth * width) * (flow_rate_kmph * 1000 / 60) = 3599.64 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end river_flow_volume_l122_122739


namespace solution_sets_equiv_solve_l122_122928

theorem solution_sets_equiv_solve (a b : ℝ) :
  (∀ x : ℝ, (4 * x + 1) / (x + 2) < 0 ↔ -2 < x ∧ x < -1 / 4) →
  (∀ x : ℝ, a * x^2 + b * x - 2 > 0 ↔ -2 < x ∧ x < -1 / 4) →
  a = -4 ∧ b = -9 := by
  sorry

end solution_sets_equiv_solve_l122_122928


namespace find_a_plus_b_plus_c_l122_122705

noncomputable def parabola_satisfies_conditions (a b c : ℝ) : Prop :=
  (∀ x, a * x ^ 2 + b * x + c ≥ 61) ∧
  (a * (1:ℝ) ^ 2 + b * (1:ℝ) + c = 0) ∧
  (a * (3:ℝ) ^ 2 + b * (3:ℝ) + c = 0)

theorem find_a_plus_b_plus_c (a b c : ℝ) 
  (h_minimum : parabola_satisfies_conditions a b c) :
  a + b + c = 0 := 
sorry

end find_a_plus_b_plus_c_l122_122705


namespace solution_set_l122_122701

def solve_inequalities (x : ℝ) : Prop :=
  (3 * x - 2) / (x - 6) ≤ 1 ∧ 2 * x ^ 2 - x - 1 > 0

theorem solution_set : { x : ℝ | solve_inequalities x } = { x : ℝ | (-2 ≤ x ∧ x < 1/2) ∨ (1 < x ∧ x < 6) } :=
by sorry

end solution_set_l122_122701


namespace min_am_hm_l122_122307

theorem min_am_hm (a b : ℝ) (ha : a > 0) (hb : b > 0) : (a + b) * (1/a + 1/b) ≥ 4 :=
by sorry

end min_am_hm_l122_122307


namespace distinct_arrangements_apple_l122_122785

theorem distinct_arrangements_apple : 
  let n := 5
  let freq_p := 2
  let freq_a := 1
  let freq_l := 1
  let freq_e := 1
  (Nat.factorial n) / (Nat.factorial freq_p * Nat.factorial freq_a * Nat.factorial freq_l * Nat.factorial freq_e) = 60 :=
by
  sorry

end distinct_arrangements_apple_l122_122785


namespace groupB_is_conditional_control_l122_122801

-- Definitions based on conditions
def groupA_medium (nitrogen_sources : Set String) : Prop := nitrogen_sources = {"urea"}
def groupB_medium (nitrogen_sources : Set String) : Prop := nitrogen_sources = {"urea", "nitrate"}

-- The property that defines a conditional control in this context.
def conditional_control (control_sources : Set String) (experimental_sources : Set String) : Prop :=
  control_sources ≠ experimental_sources ∧ "urea" ∈ control_sources ∧ "nitrate" ∈ experimental_sources

-- Prove that Group B's experiment forms a conditional control
theorem groupB_is_conditional_control :
  ∃ nitrogen_sourcesA nitrogen_sourcesB, groupA_medium nitrogen_sourcesA ∧ groupB_medium nitrogen_sourcesB ∧
  conditional_control nitrogen_sourcesA nitrogen_sourcesB :=
by
  sorry

end groupB_is_conditional_control_l122_122801


namespace intersection_points_l122_122649

def f(x : ℝ) : ℝ := x^2 + 3*x + 2
def g(x : ℝ) : ℝ := 4*x^2 + 6*x + 2

theorem intersection_points : {p : ℝ × ℝ | ∃ x, f x = p.2 ∧ g x = p.2 ∧ p.1 = x} = { (0, 2), (-1, 0) } := 
by {
  sorry
}

end intersection_points_l122_122649


namespace john_total_amount_l122_122681

def grandpa_amount : ℕ := 30
def grandma_amount : ℕ := 3 * grandpa_amount
def aunt_amount : ℕ := 3 / 2 * grandpa_amount
def uncle_amount : ℕ := 2 / 3 * grandma_amount

def total_amount : ℕ :=
  grandpa_amount + grandma_amount + aunt_amount + uncle_amount

theorem john_total_amount : total_amount = 225 := by sorry

end john_total_amount_l122_122681


namespace gcd_323_391_l122_122198

theorem gcd_323_391 : Nat.gcd 323 391 = 17 := 
by sorry

end gcd_323_391_l122_122198


namespace max_product_two_integers_l122_122461

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l122_122461


namespace BM_passes_through_fixed_point_l122_122287

noncomputable def ellipse_eq (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

theorem BM_passes_through_fixed_point (N M B : ℝ × ℝ) (k : ℝ) :
  let ellipse_eq := λ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 in
  ellipse_eq B.1 B.2 →
  ellipse_eq M.1 M.2 →
  N = (4, 0) →
  ∃ x_0 : ℝ, x_0 = 1 ∧
  ∀ (x_k : ℝ), (y : ℝ), y = k * (x_k - x_0) → ellipse_eq x_k y → (B.1, B.2) = (x_0, 0) :=
sorry

end BM_passes_through_fixed_point_l122_122287


namespace ratio_sum_l122_122150

theorem ratio_sum {x y : ℚ} (h : x / y = 4 / 7) : (x + y) / y = 11 / 7 :=
sorry

end ratio_sum_l122_122150


namespace smallest_value_of_a_l122_122817

theorem smallest_value_of_a (a b c : ℤ) (h1 : a < b) (h2 : b < c) (h3 : 2 * b = a + c) (h4 : c^2 = a * b) : a = -4 :=
by
  sorry

end smallest_value_of_a_l122_122817


namespace restore_original_price_l122_122859

theorem restore_original_price (original_price promotional_price : ℝ) (h₀ : original_price = 1) (h₁ : promotional_price = original_price * 0.8) : (original_price - promotional_price) / promotional_price = 0.25 :=
by sorry

end restore_original_price_l122_122859


namespace distance_A_to_B_l122_122755

theorem distance_A_to_B (D_B D_C V_E V_F : ℝ) (h1 : D_B / 3 = V_E)
  (h2 : D_C / 4 = V_F) (h3 : V_E / V_F = 2.533333333333333)
  (h4 : D_B = 300 ∨ D_C = 300) : D_B = 570 :=
by
  -- Proof yet to be provided
  sorry

end distance_A_to_B_l122_122755


namespace compute_expression_l122_122819
-- Import the necessary Mathlib library to work with rational numbers and basic operations

-- Define the problem context
theorem compute_expression (a b : ℚ) (ha : a = 4/7) (hb : b = 3/4) : 
  a^2 * b^(-4) = 4096 / 3969 := by
  -- Proof goes here (we use sorry to skip the proof)
  sorry

end compute_expression_l122_122819


namespace seven_n_form_l122_122661

theorem seven_n_form (n : ℤ) (a b : ℤ) (h : 7 * n = a^2 + 3 * b^2) : 
  ∃ c d : ℤ, n = c^2 + 3 * d^2 :=
by {
  sorry
}

end seven_n_form_l122_122661


namespace road_network_possible_l122_122800

theorem road_network_possible (n : ℕ) :
  (n = 6 → true) ∧ (n = 1986 → false) :=
by {
  -- Proof of the statement goes here.
  sorry
}

end road_network_possible_l122_122800


namespace greatest_product_of_sum_eq_300_l122_122385

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l122_122385


namespace max_product_two_integers_sum_300_l122_122340

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l122_122340


namespace greatest_possible_value_of_n_greatest_possible_value_of_10_l122_122720

theorem greatest_possible_value_of_n (n : ℤ) (h : 101 * n^2 ≤ 12100) : n ≤ 10 :=
by
  sorry

theorem greatest_possible_value_of_10 (n : ℤ) (h : 101 * n^2 ≤ 12100) : n = 10 → n = 10 :=
by
  sorry

end greatest_possible_value_of_n_greatest_possible_value_of_10_l122_122720


namespace frequency_in_interval_l122_122165

-- Definitions for the sample size and frequencies in given intervals
def sample_size : ℕ := 20
def freq_10_20 : ℕ := 2
def freq_20_30 : ℕ := 3
def freq_30_40 : ℕ := 4
def freq_40_50 : ℕ := 5

-- The goal: Prove that the frequency of the sample in the interval (10, 50] is 0.7
theorem frequency_in_interval (h₁ : sample_size = 20)
                              (h₂ : freq_10_20 = 2)
                              (h₃ : freq_20_30 = 3)
                              (h₄ : freq_30_40 = 4)
                              (h₅ : freq_40_50 = 5) :
  ((freq_10_20 + freq_20_30 + freq_30_40 + freq_40_50) : ℝ) / sample_size = 0.7 := 
by
  sorry

end frequency_in_interval_l122_122165


namespace checker_move_10_cells_checker_move_11_cells_l122_122590

noncomputable def F : ℕ → Nat 
| 0 => 1
| 1 => 1
| n + 2 => F (n + 1) + F n

theorem checker_move_10_cells : F 10 = 89 := by
  sorry

theorem checker_move_11_cells : F 11 = 144 := by
  sorry

end checker_move_10_cells_checker_move_11_cells_l122_122590


namespace card_drawing_ways_l122_122217

theorem card_drawing_ways :
  (30 * 20 = 600) :=
by
  sorry

end card_drawing_ways_l122_122217


namespace bruno_pens_l122_122116

-- Define Bruno's purchase of pens
def one_dozen : Nat := 12
def half_dozen : Nat := one_dozen / 2
def two_and_half_dozens : Nat := 2 * one_dozen + half_dozen

-- State the theorem to be proved
theorem bruno_pens : two_and_half_dozens = 30 :=
by sorry

end bruno_pens_l122_122116


namespace greatest_product_sum_300_l122_122367

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l122_122367


namespace diminished_value_160_l122_122257

theorem diminished_value_160 (x : ℕ) (n : ℕ) : 
  (∀ m, m > 200 ∧ (∀ k, m = k * 180) → n = m) →
  (200 + x = n) →
  x = 160 :=
by
  sorry

end diminished_value_160_l122_122257


namespace max_product_sum_300_l122_122485

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l122_122485


namespace rectangle_difference_l122_122329

theorem rectangle_difference (L B : ℝ) (h1 : 2 * (L + B) = 266) (h2 : L * B = 4290) :
  L - B = 23 :=
sorry

end rectangle_difference_l122_122329


namespace find_asymptote_slope_l122_122969

theorem find_asymptote_slope :
  (∀ x y : ℝ, (x^2 / 144 - y^2 / 81 = 0) → (y = 3/4 * x ∨ y = -3/4 * x)) :=
by
  sorry

end find_asymptote_slope_l122_122969


namespace value_of_expression_l122_122771

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : a^2 - b^2 + 6 * b = 9 :=
by
  sorry

end value_of_expression_l122_122771


namespace greatest_product_of_two_integers_with_sum_300_l122_122513

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l122_122513


namespace secretary_work_hours_l122_122231

theorem secretary_work_hours
  (x : ℕ)
  (h_ratio : 2 * x + 3 * x + 5 * x = 110) :
  5 * x = 55 := 
by
  sorry

end secretary_work_hours_l122_122231


namespace max_product_two_integers_l122_122454

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l122_122454


namespace sequence_area_formula_l122_122747

open Real

noncomputable def S_n (n : ℕ) : ℝ := (8 / 5) - (3 / 5) * (4 / 9) ^ n

theorem sequence_area_formula (n : ℕ) :
  S_n n = (8 / 5) - (3 / 5) * (4 / 9) ^ n := sorry

end sequence_area_formula_l122_122747


namespace sum_possible_x_eq_16_5_l122_122334

open Real

noncomputable def sum_of_possible_x : Real :=
  let a := 2
  let b := -33
  let c := 87
  (-b) / (2 * a)

theorem sum_possible_x_eq_16_5 : sum_of_possible_x = 16.5 :=
  by
    -- The actual proof goes here
    sorry

end sum_possible_x_eq_16_5_l122_122334


namespace quadratic_function_value_when_x_is_zero_l122_122651

theorem quadratic_function_value_when_x_is_zero :
  (∃ h : ℝ, (∀ x : ℝ, x < -3 → (-(x + h)^2 < -(x + h + 1)^2)) ∧
            (∀ x : ℝ, x > -3 → (-(x + h)^2 > -(x + h - 1)^2)) ∧
            (y = -(0 + h)^2) → y = -9) := 
sorry

end quadratic_function_value_when_x_is_zero_l122_122651


namespace annual_rent_per_square_foot_is_156_l122_122835

-- Given conditions
def monthly_rent : ℝ := 1300
def length : ℝ := 10
def width : ℝ := 10
def area : ℝ := length * width
def annual_rent : ℝ := monthly_rent * 12

-- Proof statement: Annual rent per square foot
theorem annual_rent_per_square_foot_is_156 : 
  annual_rent / area = 156 := by
  sorry

end annual_rent_per_square_foot_is_156_l122_122835


namespace find_original_number_of_men_l122_122731

variable (M : ℕ) (W : ℕ)

-- Given conditions translated to Lean
def condition1 := M * 10 = W -- M men complete work W in 10 days
def condition2 := (M - 10) * 20 = W -- (M - 10) men complete work W in 20 days

theorem find_original_number_of_men (h1 : condition1 M W) (h2 : condition2 M W) : M = 20 :=
sorry

end find_original_number_of_men_l122_122731


namespace problem_l122_122777

variables {b1 b2 b3 a1 a2 : ℤ}

-- Condition: five numbers -9, b1, b2, b3, -1 form a geometric sequence.
def is_geometric_seq (b1 b2 b3 : ℤ) : Prop :=
b1^2 = -9 * b2 ∧ b2^2 = b1 * b3 ∧ b1 * b3 = 9

-- Condition: four numbers -9, a1, a2, -3 form an arithmetic sequence.
def is_arithmetic_seq (a1 a2 : ℤ) : Prop :=
2 * a1 = -9 + a2 ∧ 2 * a2 = a1 - 3

-- Proof problem: prove that b2(a2 - a1) = -6
theorem problem (h_geom : is_geometric_seq b1 b2 b3) (h_arith : is_arithmetic_seq a1 a2) : 
  b2 * (a2 - a1) = -6 :=
by sorry

end problem_l122_122777


namespace max_product_sum_300_l122_122487

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l122_122487


namespace problem_part_1_problem_part_2_l122_122098

theorem problem_part_1 
  (p_A : ℚ) (p_B : ℚ)
  (hA : p_A = 2 / 3) 
  (hB : p_B = 3 / 4) : 
  1 - p_A ^ 3 = 19 / 27 :=
by sorry

theorem problem_part_2 
  (p_A : ℚ) (p_B : ℚ)
  (hA : p_A = 2 / 3) 
  (hB : p_B = 3 / 4) 
  (h1 : 3 * (p_A ^ 2) * (1 - p_A) = 4 / 9)
  (h2 : 3 * p_B * ((1 - p_B) ^ 2) = 9 / 64) : 
  (4 / 9) * (9 / 64) = 1 / 16 :=
by sorry

end problem_part_1_problem_part_2_l122_122098


namespace greatest_product_of_sum_eq_300_l122_122391

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l122_122391


namespace match_sequences_count_l122_122964

-- Definitions based on the given conditions
def team_size : ℕ := 7
def total_matches : ℕ := 2 * team_size - 1

def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement: number of possible match sequences
theorem match_sequences_count : 
  2 * binomial_coefficient total_matches team_size = 3432 :=
by
  sorry

end match_sequences_count_l122_122964


namespace Cooper_age_l122_122846

variable (X : ℕ)
variable (Dante : ℕ)
variable (Maria : ℕ)

theorem Cooper_age (h1 : Dante = 2 * X) (h2 : Maria = 2 * X + 1) (h3 : X + Dante + Maria = 31) : X = 6 :=
by
  -- Proof is omitted as indicated
  sorry

end Cooper_age_l122_122846


namespace magnitude_of_vector_l122_122134

open Complex

theorem magnitude_of_vector (z : ℂ) (h : z = 1 - I) : 
  ‖(2 / z + z^2)‖ = Real.sqrt 2 :=
by
  sorry

end magnitude_of_vector_l122_122134


namespace kamal_age_problem_l122_122093

theorem kamal_age_problem (K S : ℕ) 
  (h1 : K - 8 = 4 * (S - 8)) 
  (h2 : K + 8 = 2 * (S + 8)) : 
  K = 40 := 
by sorry

end kamal_age_problem_l122_122093


namespace greatest_product_sum_300_l122_122415

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l122_122415


namespace Chris_age_l122_122703

theorem Chris_age 
  (a b c : ℝ)
  (h1 : a + b + c = 36)
  (h2 : c - 5 = a)
  (h3 : b + 4 = (3 / 4) * (a + 4)) :
  c = 15.5454545454545 :=
by
  sorry

end Chris_age_l122_122703


namespace greatest_product_obtainable_l122_122398

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l122_122398


namespace max_product_of_sum_300_l122_122467

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l122_122467


namespace greatest_product_sum_300_l122_122363

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l122_122363


namespace greatest_product_sum_300_l122_122354

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l122_122354


namespace symmetric_point_y_axis_l122_122204

theorem symmetric_point_y_axis (B : ℝ × ℝ) (hB : B = (-3, 4)) : 
  ∃ A : ℝ × ℝ, A = (3, 4) ∧ A.2 = B.2 ∧ A.1 = -B.1 :=
by
  use (3, 4)
  sorry

end symmetric_point_y_axis_l122_122204


namespace solve_equation1_solve_equation2_l122_122961

-- Proof for equation (1)
theorem solve_equation1 : ∃ x : ℝ, 2 * (2 * x + 1) - (3 * x - 4) = 2 := by
  exists -4
  sorry

-- Proof for equation (2)
theorem solve_equation2 : ∃ y : ℝ, (3 * y - 1) / 4 - 1 = (5 * y - 7) / 6 := by
  exists -1
  sorry

end solve_equation1_solve_equation2_l122_122961


namespace greatest_product_two_ints_sum_300_l122_122441

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l122_122441


namespace greatest_product_of_two_integers_with_sum_300_l122_122511

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l122_122511


namespace zero_interval_of_f_l122_122743

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 / x

theorem zero_interval_of_f :
    ∃ c, 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  sorry

end zero_interval_of_f_l122_122743


namespace max_ab_squared_l122_122816

theorem max_ab_squared (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + b = 2) :
  ∃ x, 0 < x ∧ x < 2 ∧ a = 2 - x ∧ ab^2 = x * (2 - x)^2 :=
sorry

end max_ab_squared_l122_122816


namespace greatest_product_sum_300_l122_122357

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l122_122357


namespace magnitude_of_complex_number_l122_122122

def complex_number : ℂ := (2/3 : ℚ) - (4/5 : ℚ) * complex.I

theorem magnitude_of_complex_number :
  complex.abs complex_number = real.sqrt (244) / 15 :=
by
  sorry

end magnitude_of_complex_number_l122_122122


namespace tenth_flip_head_probability_l122_122226

/-- Given that the coin is fair and the first 9 flips resulted in 6 heads,
prove that the probability that the 10th flip will result in a head is 1/2. -/
theorem tenth_flip_head_probability (fair_coin : ℙ (flip = head) = 1/2 ∧ ℙ (flip = tail) = 1/2)
: ℙ (flip = head) = 1/2 :=
by
  sorry

end tenth_flip_head_probability_l122_122226


namespace greatest_product_sum_300_l122_122412

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l122_122412


namespace butterfly_eq_roots_l122_122622

theorem butterfly_eq_roots (a b c : ℝ) (h1 : a ≠ 0) (h2 : a - b + c = 0)
    (h3 : (a + c)^2 - 4 * a * c = 0) : a = c :=
by
  sorry

end butterfly_eq_roots_l122_122622


namespace a_pow_a_b_pow_b_c_pow_c_ge_one_l122_122812

theorem a_pow_a_b_pow_b_c_pow_c_ge_one
    (a b c : ℝ)
    (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h : a + b + c = Real.rpow a (1/7) + Real.rpow b (1/7) + Real.rpow c (1/7)) :
    a^a * b^b * c^c ≥ 1 := 
by
  sorry

end a_pow_a_b_pow_b_c_pow_c_ge_one_l122_122812


namespace max_value_of_f_min_value_of_a2_4b2_min_value_of_a2_4b2_equals_l122_122141

noncomputable def f (x a b : ℝ) : ℝ := |x - a| - |x + 2 * b|

theorem max_value_of_f (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∀ x, f x a b ≤ a + 2 * b :=
by sorry

theorem min_value_of_a2_4b2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_max : a + 2 * b = 1) :
  a^2 + 4 * b^2 ≥ 1 / 2 :=
by sorry

theorem min_value_of_a2_4b2_equals (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_max : a + 2 * b = 1) :
  ∃ a b, a = 1 / 2 ∧ b = 1 / 4 ∧ (a^2 + 4 * b^2 = 1 / 2) :=
by sorry

end max_value_of_f_min_value_of_a2_4b2_min_value_of_a2_4b2_equals_l122_122141


namespace dawson_marks_l122_122121

theorem dawson_marks :
  ∀ (max_marks : ℕ) (passing_percentage : ℕ) (failed_by : ℕ) (M : ℕ),
  max_marks = 220 →
  passing_percentage = 30 →
  failed_by = 36 →
  M = (passing_percentage * max_marks / 100) - failed_by →
  M = 30 := by
  intros max_marks passing_percentage failed_by M h_max h_percent h_failed h_M
  rw [h_max, h_percent, h_failed] at h_M
  norm_num at h_M
  exact h_M

end dawson_marks_l122_122121


namespace sum_of_squares_l122_122918

theorem sum_of_squares (x y : ℝ) (h1 : (x + y) ^ 2 = 4) (h2 : x * y = -1) :
  x^2 + y^2 = 6 :=
by
  sorry

end sum_of_squares_l122_122918


namespace matrix_determinant_transformation_l122_122778

theorem matrix_determinant_transformation (p q r s : ℝ) (h : p * s - q * r = -3) :
  (p * (5 * r + 4 * s) - r * (5 * p + 4 * q)) = -12 :=
sorry

end matrix_determinant_transformation_l122_122778


namespace determine_common_difference_l122_122897

variables {a : ℕ → ℤ} {d : ℤ}

-- Definition of the arithmetic sequence
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a 1 + n * d

-- The given condition in the problem
def given_condition (a : ℕ → ℤ) (d : ℤ) : Prop :=
  3 * a 6 = a 3 + a 4 + a 5 + 6

-- The theorem to prove
theorem determine_common_difference
  (h_seq : arithmetic_seq a d)
  (h_cond : given_condition a d) :
  d = 1 :=
sorry

end determine_common_difference_l122_122897


namespace max_product_two_integers_sum_300_l122_122346

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l122_122346


namespace probability_sqrt_less_than_nine_is_correct_l122_122573

def probability_sqrt_less_than_nine : ℚ :=
  let total_two_digit_numbers := 99 - 10 + 1 in
  let satisfying_numbers := 80 - 10 + 1 in
  let probability := (satisfying_numbers : ℚ) / (total_two_digit_numbers : ℚ) in
  probability

theorem probability_sqrt_less_than_nine_is_correct :
  probability_sqrt_less_than_nine = 71 / 90 :=
by
  -- proof here
  sorry

end probability_sqrt_less_than_nine_is_correct_l122_122573


namespace additional_carpet_needed_l122_122314

-- Definitions according to the given conditions
def length_feet := 18
def width_feet := 12
def covered_area := 4 -- in square yards
def feet_per_yard := 3

-- Prove that the additional square yards needed to cover the remaining part of the floor is 20
theorem additional_carpet_needed : 
  ((length_feet / feet_per_yard) * (width_feet / feet_per_yard) - covered_area) = 20 := 
by
  sorry

end additional_carpet_needed_l122_122314


namespace amount_spent_on_shorts_l122_122057

def amount_spent_on_shirt := 12.14
def amount_spent_on_jacket := 7.43
def total_amount_spent_on_clothes := 33.56

theorem amount_spent_on_shorts : total_amount_spent_on_clothes - amount_spent_on_shirt - amount_spent_on_jacket = 13.99 :=
by
  sorry

end amount_spent_on_shorts_l122_122057


namespace set_intersection_l122_122281

open Set Real

theorem set_intersection (A : Set ℝ) (hA : A = {-1, 0, 1}) (B : Set ℝ) (hB : B = {y | ∃ x ∈ A, y = cos (π * x)}) :
  A ∩ B = {-1, 1} :=
by
  rw [hA, hB]
  -- remaining proof should go here
  sorry

end set_intersection_l122_122281


namespace min_omega_value_l122_122947

noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + φ)

theorem min_omega_value (ω T φ : ℝ) (hω : ω > 0)
  (hφ_range : 0 < φ ∧ φ < π) (hT : T = 2 * π / ω)
  (hT_value : f ω φ T = sqrt 3 / 2)
  (hx_zero : f ω φ (π / 9) = 0) :
  ω = 3 := sorry

end min_omega_value_l122_122947


namespace train_usual_time_l122_122228

theorem train_usual_time (S T_new T : ℝ) (h_speed : T_new = 7 / 6 * T) (h_delay : T_new = T + 1 / 6) : T = 1 := by
  sorry

end train_usual_time_l122_122228


namespace arun_weight_l122_122871

theorem arun_weight (W B : ℝ) (h1 : 65 < W ∧ W < 72) (h2 : B < W ∧ W < 70) (h3 : W ≤ 68) (h4 : (B + 68) / 2 = 67) : B = 66 :=
sorry

end arun_weight_l122_122871


namespace max_product_sum_300_l122_122488

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l122_122488


namespace raw_score_is_correct_l122_122690

-- Define the conditions
def points_per_correct : ℝ := 1
def points_subtracted_per_incorrect : ℝ := 0.25
def total_questions : ℕ := 85
def answered_questions : ℕ := 82
def correct_answers : ℕ := 70

-- Define the number of incorrect answers
def incorrect_answers : ℕ := answered_questions - correct_answers
-- Calculate the raw score
def raw_score : ℝ := 
  (correct_answers * points_per_correct) - (incorrect_answers * points_subtracted_per_incorrect)

-- Prove the raw score is 67
theorem raw_score_is_correct : raw_score = 67 := 
by
  sorry

end raw_score_is_correct_l122_122690


namespace max_product_two_integers_l122_122458

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l122_122458


namespace total_height_of_buildings_l122_122592

theorem total_height_of_buildings :
  let height_first_building := 600
  let height_second_building := 2 * height_first_building
  let height_third_building := 3 * (height_first_building + height_second_building)
  height_first_building + height_second_building + height_third_building = 7200 := by
    let height_first_building := 600
    let height_second_building := 2 * height_first_building
    let height_third_building := 3 * (height_first_building + height_second_building)
    show height_first_building + height_second_building + height_third_building = 7200
    sorry

end total_height_of_buildings_l122_122592


namespace max_product_of_sum_300_l122_122436

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l122_122436


namespace Q_at_1_eq_1_l122_122253

noncomputable def Q (x : ℚ) : ℚ := x^4 - 16*x^2 + 16

theorem Q_at_1_eq_1 : Q 1 = 1 := by
  sorry

end Q_at_1_eq_1_l122_122253


namespace max_product_sum_300_l122_122492

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l122_122492


namespace greatest_product_of_two_integers_with_sum_300_l122_122515

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l122_122515


namespace critical_force_rod_truncated_cone_l122_122887

-- Define the given conditions
variable (r0 : ℝ) (q : ℝ) (E : ℝ) (l : ℝ) (π : ℝ)

-- Assumptions
axiom q_positive : q > 0

-- Definition for the new radius based on q
def r1 : ℝ := r0 * (1 + q)

-- Proof problem statement
theorem critical_force_rod_truncated_cone (h : q > 0) : 
  ∃ Pkp : ℝ, Pkp = (E * π * r0^4 * 4.743 / l^2) * (1 + 2 * q) :=
sorry

end critical_force_rod_truncated_cone_l122_122887


namespace sum_weights_second_fourth_l122_122167

-- Definitions based on given conditions
noncomputable section

def weight (n : ℕ) : ℕ := 4 - (n - 1)

-- Assumption that weights form an arithmetic sequence.
-- 1st foot weighs 4 jin, 5th foot weighs 2 jin, and weights are linearly decreasing.
axiom weight_arith_seq (n : ℕ) : weight n = 4 - (n - 1)

-- Prove the sum of the weights of the second and fourth feet
theorem sum_weights_second_fourth :
  weight 2 + weight 4 = 6 :=
by
  simp [weight_arith_seq]
  sorry

end sum_weights_second_fourth_l122_122167


namespace solve_equation_l122_122192

theorem solve_equation (x : ℝ) : (x + 3)^4 + (x + 1)^4 = 82 → x = 0 ∨ x = -4 :=
by
  sorry

end solve_equation_l122_122192


namespace value_of_x2_plus_y2_l122_122909

theorem value_of_x2_plus_y2 (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 :=
by
  sorry

end value_of_x2_plus_y2_l122_122909


namespace yang_tricks_modulo_l122_122227

noncomputable def number_of_tricks_result : Nat :=
  let N := 20000
  let modulo := 100000
  N % modulo

theorem yang_tricks_modulo :
  number_of_tricks_result = 20000 :=
by
  sorry

end yang_tricks_modulo_l122_122227


namespace marcus_point_value_l122_122047

theorem marcus_point_value 
  (team_total_points : ℕ)
  (marcus_percentage : ℚ)
  (three_point_goals : ℕ)
  (num_goals_type2 : ℕ)
  (score_type1 : ℕ)
  (score_type2 : ℕ)
  (total_marcus_points : ℚ)
  (points_type2 : ℚ)
  (three_point_value : ℕ := 3):
  team_total_points = 70 →
  marcus_percentage = 0.5 →
  three_point_goals = 5 →
  num_goals_type2 = 10 →
  total_marcus_points = marcus_percentage * team_total_points →
  score_type1 = three_point_goals * three_point_value →
  points_type2 = total_marcus_points - score_type1 →
  score_type2 = points_type2 / num_goals_type2 →
  score_type2 = 2 :=
by
  intros
  sorry

end marcus_point_value_l122_122047


namespace max_product_of_sum_300_l122_122430

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l122_122430


namespace greatest_product_sum_300_l122_122358

theorem greatest_product_sum_300 : ∃ (x y : ℕ), x + y = 300 ∧ (∀ (a b : ℕ), a + b = 300 → a * b ≤ x * y) := 
sorry

end greatest_product_sum_300_l122_122358


namespace combined_work_rate_l122_122857

theorem combined_work_rate (x_rate y_rate z_rate : ℚ) (W : ℚ) :
  x_rate = W / 20 → y_rate = W / 40 → z_rate = W / 30 →
  (∀ (d : ℚ), 1 / d = (x_rate + y_rate + z_rate) / W) → d = 120 / 13 :=
by
  intros hx hy hz h
  have : x_rate + y_rate + z_rate = (6 + 3 + 4) * W / 120 := by
    rw [hx, hy, hz]
    norm_num
  rw ←this at h
  exact (inv_eq_iff.mp h).symm

end combined_work_rate_l122_122857


namespace parabola_directrix_l122_122066

theorem parabola_directrix (p : ℝ) (hp : p > 0) (H : - (p / 2) = -3) : p = 6 :=
by
  sorry

end parabola_directrix_l122_122066


namespace equation_of_line_through_point_with_given_slope_l122_122779

-- Define the condition that line L passes through point P(-2, 5) and has slope -3/4
def line_through_point_with_slope (x1 y1 m : ℚ) (x y : ℚ) : Prop :=
  y - y1 = m * (x - x1)

-- Define the specific point (-2, 5) and slope -3/4
def P : ℚ × ℚ := (-2, 5)
def m : ℚ := -3 / 4

-- The standard form equation of the line as the target
def standard_form (x y : ℚ) : Prop :=
  3 * x + 4 * y - 14 = 0

-- The theorem to prove
theorem equation_of_line_through_point_with_given_slope :
  ∀ x y : ℚ, line_through_point_with_slope (-2) 5 (-3 / 4) x y → standard_form x y :=
  by
    intros x y h
    sorry

end equation_of_line_through_point_with_given_slope_l122_122779


namespace additional_time_to_walk_1_mile_l122_122224

open Real

noncomputable def additional_time_per_mile
  (distance_child : ℝ) (time_child : ℝ)
  (distance_elderly : ℝ) (time_elderly : ℝ)
  : ℝ :=
  let speed_child := distance_child / time_child
  let time_per_mile_child := (time_child * 60) / distance_child
  let speed_elderly := distance_elderly / time_elderly
  let time_per_mile_elderly := (time_elderly * 60) / distance_elderly
  time_per_mile_elderly - time_per_mile_child

theorem additional_time_to_walk_1_mile
  (h1 : 15 = 15) (h2 : 3.5 = 3.5)
  (h3 : 10 = 10) (h4 : 4 = 4)
  : additional_time_per_mile 15 3.5 10 4 = 10 :=
  by
    sorry

end additional_time_to_walk_1_mile_l122_122224


namespace compute_fg_neg_2_l122_122945

def f (x : ℝ) : ℝ := 2 * x - 5
def g (x : ℝ) : ℝ := x^2 + 4 * x + 4

theorem compute_fg_neg_2 : f (g (-2)) = -5 :=
by
-- sorry is used to skip the proof
sorry

end compute_fg_neg_2_l122_122945


namespace eval_expression_l122_122757

theorem eval_expression (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a^2 + b^2) / (a * b) - (a^2 + a * b) / (a^2 + b^2) = (a^4 + b^4 + a^2 * b^2 - a^2 * b - a * b^2) / (a * b * (a^2 + b^2)) :=
by
  sorry

end eval_expression_l122_122757


namespace min_max_a_e_l122_122833

noncomputable def find_smallest_largest (a b c d e : ℝ) : ℝ × ℝ :=
  if a + b < c + d ∧ c + d < e + a ∧ e + a < b + c ∧ b + c < d + e
    then (a, e)
    else (-1, -1) -- using -1 to indicate invalid input

theorem min_max_a_e (a b c d e : ℝ) : a + b < c + d ∧ c + d < e + a ∧ e + a < b + c ∧ b + c < d + e → 
    find_smallest_largest a b c d e = (a, e) :=
  by
    -- Proof to be filled in by user
    sorry

end min_max_a_e_l122_122833


namespace exists_divisor_for_all_f_values_l122_122175

theorem exists_divisor_for_all_f_values (f : ℕ → ℕ) (h_f_range : ∀ n, 1 < f n) (h_f_div : ∀ m n, f (m + n) ∣ f m + f n) :
  ∃ c : ℕ, c > 1 ∧ ∀ n, c ∣ f n := 
sorry

end exists_divisor_for_all_f_values_l122_122175


namespace xy_identity_l122_122912

theorem xy_identity (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 := by
  sorry

end xy_identity_l122_122912


namespace greatest_product_sum_300_l122_122378

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l122_122378


namespace exists_two_positive_integers_dividing_3003_l122_122632

theorem exists_two_positive_integers_dividing_3003 : 
  ∃ (m1 m2 : ℕ), m1 > 0 ∧ m2 > 0 ∧ m1 ≠ m2 ∧ (3003 % (m1^2 + 2) = 0) ∧ (3003 % (m2^2 + 2) = 0) :=
by
  sorry

end exists_two_positive_integers_dividing_3003_l122_122632


namespace greatest_product_sum_300_l122_122422

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l122_122422


namespace find_divisor_l122_122239

theorem find_divisor (x y : ℝ) (hx : x > 0) (hx_val : x = 1.3333333333333333) (h : 4 * x / y = x^2) : y = 3 :=
by 
  sorry

end find_divisor_l122_122239


namespace cooper_age_l122_122841

variable (Cooper Dante Maria : ℕ)

-- Conditions
def sum_of_ages : Prop := Cooper + Dante + Maria = 31
def dante_twice_cooper : Prop := Dante = 2 * Cooper
def maria_one_year_older : Prop := Maria = Dante + 1

theorem cooper_age (h1 : sum_of_ages Cooper Dante Maria) (h2 : dante_twice_cooper Cooper Dante) (h3 : maria_one_year_older Dante Maria) : Cooper = 6 :=
by
  sorry

end cooper_age_l122_122841


namespace fliers_sent_afternoon_fraction_l122_122086

-- Definitions of given conditions
def total_fliers : ℕ := 2000
def fliers_morning_fraction : ℚ := 1 / 10
def remaining_fliers_next_day : ℕ := 1350

-- Helper definitions based on conditions
def fliers_sent_morning := total_fliers * fliers_morning_fraction
def fliers_after_morning := total_fliers - fliers_sent_morning
def fliers_sent_afternoon := fliers_after_morning - remaining_fliers_next_day

-- Theorem stating the required proof
theorem fliers_sent_afternoon_fraction :
  fliers_sent_afternoon / fliers_after_morning = 1 / 4 :=
sorry

end fliers_sent_afternoon_fraction_l122_122086


namespace complement_A_possible_set_l122_122903

variable (U A B : Set ℕ)

theorem complement_A_possible_set (hU : U = {1, 2, 3, 4, 5, 6})
  (h_union : A ∪ B = {1, 2, 3, 4, 5}) 
  (h_inter : A ∩ B = {3, 4, 5}) :
  ∃ C, C = U \ A ∧ C = {6} :=
by
  sorry

end complement_A_possible_set_l122_122903


namespace permutation_arrangement_count_l122_122624

theorem permutation_arrangement_count :
  let total_letters := 11
  let num_T := 2
  let num_A := 2
  (11.factorial / (2.factorial * 2.factorial)) = 9979200 :=
by
  sorry

end permutation_arrangement_count_l122_122624


namespace myrtle_eggs_l122_122182

theorem myrtle_eggs :
  ∀ (daily_rate per_hen : ℕ) (num_hens : ℕ) (days_away : ℕ) (eggs_taken : ℕ) (eggs_dropped : ℕ),
    daily_rate = 3 →
    num_hens = 3 →
    days_away = 7 →
    eggs_taken = 12 →
    eggs_dropped = 5 →
    (num_hens * daily_rate * days_away - eggs_taken - eggs_dropped) = 46 :=
by
  intros daily_rate per_hen num_hens days_away eggs_taken eggs_dropped
  assume h_rate h_hens h_days h_taken h_dropped
  rw [h_rate, h_hens, h_days, h_taken, h_dropped]
  calc 3 * 3 * 7 - 12 - 5 = 63 - 12 - 5 : by norm_num
                     ... = 51 - 5     : by norm_num
                     ... = 46         : by norm_num
  done

end myrtle_eggs_l122_122182


namespace triangle_angle_correct_l122_122202

noncomputable def triangle_angles
  (a b c : ℝ)
  (ha : a = 2)
  (hb : b = Real.sqrt 6)
  (hc : c = 1 + Real.sqrt 3) : ℝ × ℝ × ℝ := 
  let α := Real.acos ((b^2 + c^2 - a^2) / (2 * b * c))
  let β := Real.acos ((a^2 + c^2 - b^2) / (2 * a * c))
  let γ := Real.acos ((a^2 + b^2 - c^2) / (2 * a * b))
  (α, β, γ)

theorem triangle_angle_correct
  (α β γ : ℝ)
  (hα : α = Real.acos (1/2)) 
  (hβ : β = Real.acos (Real.sqrt 2/2))
  (hγ : γ = Real.acos ((Real.sqrt 3 + 1 - 2) / (2 * Real.sqrt 6 * (1 + Real.sqrt 3))))
  : (
    α = Real.pi / 3 ∧ 
    β = Real.pi / 4 ∧ 
    γ = 5 * Real.pi / 12
  ) :=
  by
    sorry

end triangle_angle_correct_l122_122202


namespace total_snow_volume_l122_122943

-- Definitions and conditions set up from part (a)
def driveway_length : ℝ := 30
def driveway_width : ℝ := 3
def section1_length : ℝ := 10
def section1_depth : ℝ := 1
def section2_length : ℝ := driveway_length - section1_length
def section2_depth : ℝ := 0.5

-- The theorem corresponding to part (c)
theorem total_snow_volume : 
  (section1_length * driveway_width * section1_depth) +
  (section2_length * driveway_width * section2_depth) = 60 :=
by 
  -- Proof is omitted as required
  sorry

end total_snow_volume_l122_122943


namespace greatest_product_two_ints_sum_300_l122_122450

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l122_122450


namespace increase_in_avg_commission_l122_122955

def new_avg_commission := 250
def num_sales := 6
def big_sale_commission := 1000

theorem increase_in_avg_commission :
  (new_avg_commission - (500 / (num_sales - 1))) = 150 := by
  sorry

end increase_in_avg_commission_l122_122955


namespace max_product_two_integers_l122_122456

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l122_122456


namespace greatest_product_two_ints_sum_300_l122_122446

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l122_122446


namespace inequality_inequality_proof_l122_122895

variable {x y z : ℝ}

theorem inequality_inequality_proof :
  (x >= 0) ∧ (y >= 0) ∧ (z >= 0) ∧ (y * z + z * x + x * y = 1) →
  (x * (1 - y^2) * (1 - z^2) + y * (1 - z^2) * (1 - x^2) + z * (1 - x^2) * (1 - y^2) ≤ (4 / 9) * Real.sqrt 3) := by
  intro h
  sorry

end inequality_inequality_proof_l122_122895


namespace greatest_product_sum_300_l122_122411

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l122_122411


namespace solution_set_inequality_l122_122974

theorem solution_set_inequality (x : ℝ) : (x + 3) / (x - 1) > 0 ↔ x < -3 ∨ x > 1 :=
sorry

end solution_set_inequality_l122_122974


namespace greatest_product_of_two_integers_with_sum_300_l122_122520

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l122_122520


namespace number_of_coaches_l122_122839

theorem number_of_coaches (r : ℕ) (v : ℕ) (c : ℕ) (h1 : r = 60) (h2 : v = 3) (h3 : c * 5 = 60 * 3) : c = 36 :=
by
  -- We skip the proof as per instructions
  sorry

end number_of_coaches_l122_122839


namespace shirt_cost_correct_l122_122189

-- Define the conditions
def pants_cost : ℝ := 9.24
def bill_amount : ℝ := 20
def change_received : ℝ := 2.51

-- Calculate total spent and shirt cost
def total_spent : ℝ := bill_amount - change_received
def shirt_cost : ℝ := total_spent - pants_cost

-- The theorem statement
theorem shirt_cost_correct : shirt_cost = 8.25 := by
  sorry

end shirt_cost_correct_l122_122189


namespace max_product_of_sum_300_l122_122472

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l122_122472


namespace simplify_expression_l122_122255

theorem simplify_expression (a : ℝ) (h₁ : a ≠ 1) (h₂ : a ≠ 1 / 2) :
    1 - 1 / (1 - a / (1 - a)) = -a / (1 - 2 * a) := by
  sorry

end simplify_expression_l122_122255


namespace increasing_g_on_neg_l122_122142

variable {R : Type*} [LinearOrderedField R]

-- Assumptions: 
-- 1. f is an increasing function on R
-- 2. (h_neg : ∀ x : R, f x < 0)

theorem increasing_g_on_neg (f : R → R) (h_inc : ∀ x y : R, x < y → f x < f y) (h_neg : ∀ x : R, f x < 0) :
  ∀ x y : R, x < y → x < 0 → y < 0 → (x^2 * f x < y^2 * f y) :=
by
  sorry

end increasing_g_on_neg_l122_122142


namespace remove_terms_yield_desired_sum_l122_122261

-- Define the original sum and the terms to be removed
def originalSum : ℚ := 1/3 + 1/6 + 1/9 + 1/12 + 1/15 + 1/18
def termsToRemove : List ℚ := [1/9, 1/12, 1/15, 1/18]

-- Definition of the desired remaining sum
def desiredSum : ℚ := 1/2

noncomputable def sumRemainingTerms : ℚ :=
originalSum - List.sum termsToRemove

-- Lean theorem to prove
theorem remove_terms_yield_desired_sum : sumRemainingTerms = desiredSum :=
by 
  sorry

end remove_terms_yield_desired_sum_l122_122261


namespace value_of_f_m_plus_one_l122_122289

variable (a m : ℝ)

def f (x : ℝ) : ℝ := x^2 - x + a

theorem value_of_f_m_plus_one 
  (h : f a (-m) < 0) : f a (m + 1) < 0 := by
  sorry

end value_of_f_m_plus_one_l122_122289


namespace solve_inequality_l122_122191

theorem solve_inequality (x : ℝ) : 
  let quad := (x - 2)^2 + 9
  let numerator := x - 3
  quad > 0 ∧ numerator ≥ 0 ↔ x ≥ 3 :=
by
    sorry

end solve_inequality_l122_122191


namespace probability_of_three_red_out_of_four_l122_122727

theorem probability_of_three_red_out_of_four :
  let total_marbles := 15
  let red_marbles := 6
  let blue_marbles := 3
  let white_marbles := 6
  let total_picked := 4
  let comb_total := Nat.choose total_marbles total_picked
  let comb_red := Nat.choose red_marbles 3
  let comb_non_red := Nat.choose (total_marbles - red_marbles) 1
  let successful_outcomes := comb_red * comb_non_red
  let probability := successful_outcomes / comb_total
  probability = 4 / 15 :=
by
  -- Using Lean, we represent the probability fraction and the equality to simplify the fractions.
  sorry

end probability_of_three_red_out_of_four_l122_122727


namespace greatest_product_obtainable_l122_122401

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l122_122401


namespace max_product_sum_300_l122_122490

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l122_122490


namespace taxi_fare_calculation_l122_122078

def fare_per_km : ℝ := 1.8
def starting_fare : ℝ := 8
def starting_distance : ℝ := 2
def total_distance : ℝ := 12

theorem taxi_fare_calculation : 
  (if total_distance <= starting_distance then starting_fare
   else starting_fare + (total_distance - starting_distance) * fare_per_km) = 26 := by
  sorry

end taxi_fare_calculation_l122_122078


namespace sphere_center_plane_intersection_l122_122814

theorem sphere_center_plane_intersection
  (d e f : ℝ)
  (O : ℝ × ℝ × ℝ := (0, 0, 0))
  (A B C : ℝ × ℝ × ℝ)
  (p : ℝ)
  (hA : A ≠ O)
  (hB : B ≠ O)
  (hC : C ≠ O)
  (hA_coord : A = (2 * p, 0, 0))
  (hB_coord : B = (0, 2 * p, 0))
  (hC_coord : C = (0, 0, 2 * p))
  (h_sphere : (p, p, p) = (p, p, p)) -- we know that the center is (p, p, p)
  (h_plane : d * (1 / (2 * p)) + e * (1 / (2 * p)) + f * (1 / (2 * p)) = 1) :
  d / p + e / p + f / p = 2 := sorry

end sphere_center_plane_intersection_l122_122814


namespace pass_rate_l122_122233

theorem pass_rate (total_students : ℕ) (students_not_passed : ℕ) (pass_rate : ℚ) :
  total_students = 500 → 
  students_not_passed = 40 → 
  pass_rate = (total_students - students_not_passed) / total_students * 100 →
  pass_rate = 92 :=
by 
  intros ht hs hpr 
  sorry

end pass_rate_l122_122233


namespace max_product_two_integers_sum_300_l122_122342

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l122_122342


namespace pam_number_of_bags_l122_122691

-- Definitions of the conditions
def apples_in_geralds_bag : Nat := 40
def pam_bags_ratio : Nat := 3
def total_pam_apples : Nat := 1200

-- Problem statement (Theorem)
theorem pam_number_of_bags :
  Pam_bags == total_pam_apples / (pam_bags_ratio * apples_in_geralds_bag) :=
by 
  sorry

end pam_number_of_bags_l122_122691


namespace total_pages_in_book_l122_122721

theorem total_pages_in_book (P : ℕ) 
  (h1 : 7 / 13 * P = P - 96 - 5 / 9 * (P - 7 / 13 * P))
  (h2 : 96 = 4 / 9 * (P - 7 / 13 * P)) : 
  P = 468 :=
 by 
    sorry

end total_pages_in_book_l122_122721


namespace probability_sqrt_less_than_nine_l122_122557

theorem probability_sqrt_less_than_nine :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let T := {n : ℕ | n ∈ S ∧ n < 81}
  (T.card : ℚ) / S.card = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l122_122557


namespace greatest_product_l122_122533

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l122_122533


namespace each_piece_of_paper_weight_l122_122942

noncomputable def paper_weight : ℚ :=
 sorry

theorem each_piece_of_paper_weight (w : ℚ) (n : ℚ) (envelope_weight : ℚ) (stamps_needed : ℚ) (paper_pieces : ℚ) :
  paper_pieces = 8 →
  envelope_weight = 2/5 →
  stamps_needed = 2 →
  n = paper_pieces * w + envelope_weight →
  n ≤ stamps_needed →
  w = 1/5 :=
by sorry

end each_piece_of_paper_weight_l122_122942


namespace line_circle_chord_shortest_l122_122272

noncomputable def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

noncomputable def line_l (x y m : ℝ) : Prop := (2 * m + 1) * x + (m + 1) * y - 7 * m - 4 = 0

theorem line_circle_chord_shortest (m : ℝ) :
  (∀ x y : ℝ, circle_C x y → line_l x y m → m = -3 / 4) :=
sorry

end line_circle_chord_shortest_l122_122272


namespace correct_exp_operation_l122_122991

theorem correct_exp_operation (a : ℝ) : (a^2 * a = a^3) := 
by
  -- Leave the proof as an exercise
  sorry

end correct_exp_operation_l122_122991


namespace conversion_bah_rah_yah_l122_122790

theorem conversion_bah_rah_yah (bahs rahs yahs : ℝ) 
  (h1 : 10 * bahs = 16 * rahs) 
  (h2 : 6 * rahs = 10 * yahs) :
  (10 / 16) * (6 / 10) * 500 * yahs = 187.5 * bahs :=
by sorry

end conversion_bah_rah_yah_l122_122790


namespace solve_system_l122_122994

theorem solve_system : ∀ (a b : ℝ), (∃ (x y : ℝ), x = 5 ∧ y = b ∧ 2 * x + y = a ∧ 2 * x - y = 12) → (a = 8 ∧ b = -2) :=
by
  sorry

end solve_system_l122_122994


namespace pencil_price_in_units_l122_122660

noncomputable def price_of_pencil_in_units (base_price additional_price unit_size : ℕ) : ℝ :=
  (base_price + additional_price) / unit_size

theorem pencil_price_in_units :
  price_of_pencil_in_units 5000 200 10000 = 0.52 := 
  by 
  sorry

end pencil_price_in_units_l122_122660


namespace increase_in_area_400ft2_l122_122234

theorem increase_in_area_400ft2 (l w : ℝ) (h₁ : l = 60) (h₂ : w = 20)
  (h₃ : 4 * (l + w) = 4 * (4 * (l + w) / 4 / 4 )):
  (4 * (l + w) / 4) ^ 2 - l * w = 400 := by
  sorry

end increase_in_area_400ft2_l122_122234


namespace max_product_two_integers_sum_300_l122_122347

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l122_122347


namespace probability_sqrt_lt_9_l122_122564

theorem probability_sqrt_lt_9 : 
  ∀ (n : ℕ), 10 ≤ n ∧ n ≤ 99 →
  ∃ p : ℚ, p = 71 / 90 ∧ 
  ∑ k in (Finset.range 100).filter (λ x, 10 ≤ x ∧ sqrt x < 9), 1 / 90 = p := 
sorry

end probability_sqrt_lt_9_l122_122564


namespace second_solution_salt_percent_l122_122585

theorem second_solution_salt_percent (S : ℝ) (x : ℝ) 
  (h1 : 0.14 * S - 0.14 * (S / 4) + (x / 100) * (S / 4) = 0.16 * S) : 
  x = 22 :=
by 
  -- Proof omitted
  sorry

end second_solution_salt_percent_l122_122585


namespace hannah_practice_hours_l122_122023

theorem hannah_practice_hours (weekend_hours : ℕ) (total_weekly_hours : ℕ) (more_weekday_hours : ℕ)
  (h1 : weekend_hours = 8)
  (h2 : total_weekly_hours = 33)
  (h3 : more_weekday_hours = 17) :
  (total_weekly_hours - weekend_hours) - weekend_hours = more_weekday_hours :=
by
  sorry

end hannah_practice_hours_l122_122023


namespace number_of_squares_in_figure_100_l122_122250

theorem number_of_squares_in_figure_100 :
  ∃ (a b c : ℤ), (c = 1) ∧ (a + b + c = 7) ∧ (4 * a + 2 * b + c = 19) ∧ (3 * 100^2 + 3 * 100 + 1 = 30301) :=
sorry

end number_of_squares_in_figure_100_l122_122250


namespace max_product_sum_300_l122_122493

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l122_122493


namespace ratio_of_cows_sold_l122_122236

-- Condition 1: The farmer originally has 51 cows.
def original_cows : ℕ := 51

-- Condition 2: The farmer adds 5 new cows to the herd.
def new_cows : ℕ := 5

-- Condition 3: The farmer has 42 cows left after selling a portion of the herd.
def remaining_cows : ℕ := 42

-- Defining total cows after adding new cows
def total_cows_after_addition : ℕ := original_cows + new_cows

-- Defining cows sold
def cows_sold : ℕ := total_cows_after_addition - remaining_cows

-- The theorem states the ratio of 'cows sold' to 'total cows after addition' is 1 : 4
theorem ratio_of_cows_sold : (cows_sold : ℚ) / (total_cows_after_addition : ℚ) = 1 / 4 := by
  -- Proof would go here
  sorry


end ratio_of_cows_sold_l122_122236


namespace convert_yahs_to_bahs_l122_122291

noncomputable section

def bahs_to_rahs (bahs : ℕ) : ℕ := bahs * (36/24)
def rahs_to_bahs (rahs : ℕ) : ℕ := rahs * (24/36)
def rahs_to_yahs (rahs : ℕ) : ℕ := rahs * (18/12)
def yahs_to_rahs (yahs : ℕ) : ℕ := yahs * (12/18)
def yahs_to_bahs (yahs : ℕ) : ℕ := rahs_to_bahs (yahs_to_rahs yahs)

theorem convert_yahs_to_bahs :
  yahs_to_bahs 1500 = 667 :=
sorry

end convert_yahs_to_bahs_l122_122291


namespace green_pill_cost_l122_122869

-- Definitions for the problem conditions
def number_of_days : ℕ := 21
def total_cost : ℚ := 819
def daily_cost : ℚ := total_cost / number_of_days
def cost_green_pill (x : ℚ) : ℚ := x
def cost_pink_pill (x : ℚ) : ℚ := x - 1
def total_daily_pill_cost (x : ℚ) : ℚ := cost_green_pill x + 2 * cost_pink_pill x

-- Theorem to be proven
theorem green_pill_cost : ∃ x : ℚ, total_daily_pill_cost x = daily_cost ∧ x = 41 / 3 :=
sorry

end green_pill_cost_l122_122869


namespace sale_in_third_month_l122_122105

theorem sale_in_third_month
  (sale1 sale2 sale4 sale5 sale6 avg : ℝ)
  (n : ℕ)
  (h_sale1 : sale1 = 6235)
  (h_sale2 : sale2 = 6927)
  (h_sale4 : sale4 = 7230)
  (h_sale5 : sale5 = 6562)
  (h_sale6 : sale6 = 5191)
  (h_avg : avg = 6500)
  (h_n : n = 6) :
  ∃ sale3 : ℝ, sale3 = 6855 := by
  sorry

end sale_in_third_month_l122_122105


namespace time_spent_watching_tv_excluding_breaks_l122_122687

-- Definitions based on conditions
def total_hours_watched : ℕ := 5
def breaks : List ℕ := [10, 15, 20, 25]

-- Conversion constants
def minutes_per_hour : ℕ := 60

-- Derived definitions
def total_minutes_watched : ℕ := total_hours_watched * minutes_per_hour
def total_break_minutes : ℕ := breaks.sum

-- The main theorem
theorem time_spent_watching_tv_excluding_breaks :
  total_minutes_watched - total_break_minutes = 230 := by
  sorry

end time_spent_watching_tv_excluding_breaks_l122_122687


namespace value_of_expression_l122_122753

def x : ℝ := 12
def y : ℝ := 7

theorem value_of_expression : (x - y) * (x + y) = 95 := by
  sorry

end value_of_expression_l122_122753


namespace greatest_product_two_ints_sum_300_l122_122445

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l122_122445


namespace different_colors_probability_l122_122711

noncomputable def differentColorProbability : ℚ :=
  let redChips := 7
  let greenChips := 5
  let totalChips := redChips + greenChips
  let probRedThenGreen := (redChips / totalChips) * (greenChips / totalChips)
  let probGreenThenRed := (greenChips / totalChips) * (redChips / totalChips)
  (probRedThenGreen + probGreenThenRed)

theorem different_colors_probability :
  differentColorProbability = 35 / 72 :=
by sorry

end different_colors_probability_l122_122711


namespace arithmetic_sequence_n_l122_122299

theorem arithmetic_sequence_n 
  (a : ℕ → ℕ)
  (S : ℕ → ℕ)
  (a1 : a 1 = 1)
  (a3_plus_a5 : a 3 + a 5 = 14)
  (Sn_eq_100 : S n = 100) :
  n = 10 :=
sorry

end arithmetic_sequence_n_l122_122299


namespace greatest_product_two_ints_sum_300_l122_122443

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l122_122443


namespace wavelength_scientific_notation_l122_122333

theorem wavelength_scientific_notation :
  (0.000000193 : Float) = 1.93 * (10 : Float) ^ (-7) :=
sorry

end wavelength_scientific_notation_l122_122333


namespace incorrect_conversion_l122_122085

/--
Incorrect conversion of -150° to radians.
-/
theorem incorrect_conversion : (¬(((-150 : ℝ) * (Real.pi / 180)) = (-7 * Real.pi / 6))) :=
by
  sorry

end incorrect_conversion_l122_122085


namespace gcd_2023_2048_l122_122222

theorem gcd_2023_2048 : Nat.gcd 2023 2048 = 1 := by
  sorry

end gcd_2023_2048_l122_122222


namespace a_congruent_b_mod_1008_l122_122963

theorem a_congruent_b_mod_1008 (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b - b^a = 1008) : a ≡ b [MOD 1008] :=
by
  sorry

end a_congruent_b_mod_1008_l122_122963


namespace prob_sqrt_less_than_nine_l122_122570

/-- The probability that the square root of a randomly selected 
two-digit whole number is less than nine is 71/90. -/
theorem prob_sqrt_less_than_nine : (let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99};
                                     let A := {n : ℕ | 10 ≤ n ∧ n < 81};
                                     (A.card / S.card : ℚ) = 71 / 90) :=
by
  sorry

end prob_sqrt_less_than_nine_l122_122570


namespace max_product_two_integers_l122_122463

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l122_122463


namespace max_product_two_integers_l122_122453

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l122_122453


namespace union_of_A_and_B_l122_122273

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := 
by 
  sorry

end union_of_A_and_B_l122_122273


namespace sum_of_arcsins_l122_122089

noncomputable def arcsin_sum : Prop :=
  arcsin (1 / real.sqrt 10) + 
  arcsin (1 / real.sqrt 26) + 
  arcsin (1 / real.sqrt 50) + 
  arcsin (1 / real.sqrt 65) = π / 4

theorem sum_of_arcsins : arcsin_sum := by
  sorry

end sum_of_arcsins_l122_122089


namespace max_tied_teams_round_robin_l122_122674

theorem max_tied_teams_round_robin (n : ℕ) (h: n = 8) :
  ∃ k, (k <= n) ∧ (∀ m, m > k → k * m < n * (n - 1) / 2) :=
by
  sorry

end max_tied_teams_round_robin_l122_122674


namespace first_divisor_l122_122736

theorem first_divisor (k : ℤ) (h1 : k % 5 = 2) (h2 : k % 6 = 5) (h3 : k % 7 = 3) (h4 : k < 42) (hk : k = 17) : 5 ≤ 6 ∧ 5 ≤ 7 ∧ 5 = 5 :=
by {
  sorry
}

end first_divisor_l122_122736


namespace max_product_two_integers_sum_300_l122_122344

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l122_122344


namespace smallest_n_l122_122804

variable {a : ℕ → ℝ} -- the arithmetic sequence
noncomputable def d := a 2 - a 1  -- common difference

variable {S : ℕ → ℝ}  -- sum of the first n terms

-- conditions
axiom cond1 : a 66 < 0
axiom cond2 : a 67 > 0
axiom cond3 : a 67 > abs (a 66)

-- sum of the first n terms of the arithmetic sequence
noncomputable def sum_n (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem smallest_n (n : ℕ) : S n > 0 → n = 132 :=
by
  sorry

end smallest_n_l122_122804


namespace negation_equivalence_l122_122332

theorem negation_equivalence (x : ℝ) : ¬(∀ x, x^2 - x + 2 ≥ 0) ↔ ∃ x, x^2 - x + 2 < 0 :=
sorry

end negation_equivalence_l122_122332


namespace function_passes_through_fixed_point_l122_122013

variables {a : ℝ}

/-- Given the function f(x) = a^(x-1) (a > 0 and a ≠ 1), prove that the function always passes through the point (1, 1) -/
theorem function_passes_through_fixed_point (h1 : a > 0) (h2 : a ≠ 1) :
  (a^(1-1) = 1) :=
by
  sorry

end function_passes_through_fixed_point_l122_122013


namespace probability_nine_moves_visits_all_vertices_l122_122099

noncomputable def probability_bug_visits_all_vertices : ℚ :=
  4 / 243

theorem probability_nine_moves_visits_all_vertices :
  ∀ (start_vertex : Fin 8), 
  ∀ (move_probability : ∀ (v : Fin 8), Fin 3 → ℚ),
  (∀ (v : Fin 8) (edge : Fin 3), move_probability v edge = 1 / 3) →
  Probability.all_vertices_visited (cube : Graph) start_vertex 9 move_probability 
  = probability_bug_visits_all_vertices :=
sorry

end probability_nine_moves_visits_all_vertices_l122_122099


namespace multiplier_of_first_integer_l122_122068

theorem multiplier_of_first_integer :
  ∃ m x : ℤ, x + 4 = 15 ∧ x * m = 3 + 2 * 15 ∧ m = 3 := by
  sorry

end multiplier_of_first_integer_l122_122068


namespace sum_of_squares_l122_122919

theorem sum_of_squares (x y : ℝ) (h1 : (x + y) ^ 2 = 4) (h2 : x * y = -1) :
  x^2 + y^2 = 6 :=
by
  sorry

end sum_of_squares_l122_122919


namespace arithmetic_sequence_26th_term_eq_neg48_l122_122325

def arithmetic_sequence_term (a₁ d n : ℤ) : ℤ := a₁ + (n - 1) * d

theorem arithmetic_sequence_26th_term_eq_neg48 : 
  arithmetic_sequence_term 2 (-2) 26 = -48 :=
by
  sorry

end arithmetic_sequence_26th_term_eq_neg48_l122_122325


namespace product_of_solutions_abs_eq_product_of_solutions_l122_122888

theorem product_of_solutions_abs_eq (x : ℝ) (h : |x| = 3 * (|x| - 2)) : x = 3 ∨ x = -3 :=
by
  sorry

theorem product_of_solutions (x1 x2 : ℝ) (h1 : |x1| = 3 * (|x1| - 2)) (h2 : |x2| = 3 * (|x2| - 2)) :
  x1 * x2 = -9 :=
by
  have hx1 : x1 = 3 ∨ x1 = -3 := product_of_solutions_abs_eq x1 h1
  have hx2 : x2 = 3 ∨ x2 = -3 := product_of_solutions_abs_eq x2 h2
  cases hx1
  case Or.inl hxl1 =>
    cases hx2
    case Or.inl hxr1 =>
      exact False.elim (by sorry)
    case Or.inr hxr2 =>
      rw [hxl1, hxr2]
      norm_num
  case Or.inr hxl2 =>
    cases hx2
    case Or.inl hxr1 =>
      rw [hxl2, hxr1]
      norm_num
    case Or.inr hxr2 =>
      exact False.elim (by sorry)

end product_of_solutions_abs_eq_product_of_solutions_l122_122888


namespace eqn_abs_3x_minus_2_solution_l122_122627

theorem eqn_abs_3x_minus_2_solution (x : ℝ) :
  (|x + 5| = 3 * x - 2) ↔ x = 7 / 2 :=
by
  sorry

end eqn_abs_3x_minus_2_solution_l122_122627


namespace solve_for_x_l122_122196

noncomputable def infinite_power_tower (x : ℝ) : ℝ := sorry

theorem solve_for_x (x : ℝ) 
  (h1 : infinite_power_tower x = 4) : 
  x = Real.sqrt 2 := 
sorry

end solve_for_x_l122_122196


namespace expression_not_defined_at_x_l122_122767

theorem expression_not_defined_at_x :
  ∃ (x : ℝ), x = 10 ∧ (x^3 - 30 * x^2 + 300 * x - 1000) = 0 := 
sorry

end expression_not_defined_at_x_l122_122767


namespace probability_sqrt_less_than_nine_is_correct_l122_122574

def probability_sqrt_less_than_nine : ℚ :=
  let total_two_digit_numbers := 99 - 10 + 1 in
  let satisfying_numbers := 80 - 10 + 1 in
  let probability := (satisfying_numbers : ℚ) / (total_two_digit_numbers : ℚ) in
  probability

theorem probability_sqrt_less_than_nine_is_correct :
  probability_sqrt_less_than_nine = 71 / 90 :=
by
  -- proof here
  sorry

end probability_sqrt_less_than_nine_is_correct_l122_122574


namespace swimming_club_cars_l122_122077

theorem swimming_club_cars (c : ℕ) :
  let vans := 3
  let people_per_car := 5
  let people_per_van := 3
  let max_people_per_car := 6
  let max_people_per_van := 8
  let extra_people := 17
  let total_people := 5 * c + (people_per_van * vans)
  let max_capacity := max_people_per_car * c + (max_people_per_van * vans)
  (total_people + extra_people = max_capacity) → c = 2 := by
  sorry

end swimming_club_cars_l122_122077


namespace number_of_hexagonal_faces_geq_2_l122_122698

noncomputable def polyhedron_condition (P H : ℕ) : Prop :=
  ∃ V E : ℕ, 
    V - E + (P + H) = 2 ∧ 
    3 * V = 2 * E ∧ 
    E = (5 * P + 6 * H) / 2 ∧
    P > 0 ∧ H > 0

theorem number_of_hexagonal_faces_geq_2 (P H : ℕ) (h : polyhedron_condition P H) : H ≥ 2 :=
sorry

end number_of_hexagonal_faces_geq_2_l122_122698


namespace greatest_product_sum_300_l122_122380

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l122_122380


namespace value_of_expression_l122_122634

theorem value_of_expression (m : ℝ) (h : 1 / (m - 2) = 1) : (2 / (m - 2)) - m + 2 = 1 :=
sorry

end value_of_expression_l122_122634


namespace max_product_two_integers_l122_122459

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l122_122459


namespace greatest_product_two_ints_sum_300_l122_122451

theorem greatest_product_two_ints_sum_300 :
  (∀ x y : ℤ, x + y = 300 → x * y ≤ 22500) ∧ (∃ x y : ℤ, x + y = 300 ∧ x * y = 22500) :=
by
  sorry

end greatest_product_two_ints_sum_300_l122_122451


namespace compare_times_l122_122173

variable {v : ℝ} (h_v_pos : 0 < v)

/-- 
  Jones covered a distance of 80 miles on his first trip at speed v.
  On a later trip, he traveled 360 miles at four times his original speed.
  Prove that his new time is (9/8) times his original time.
-/
theorem compare_times :
  let t1 := 80 / v
  let t2 := 360 / (4 * v)
  t2 = (9 / 8) * t1 :=
by
  sorry

end compare_times_l122_122173


namespace a_2_geometric_sequence_l122_122664

theorem a_2_geometric_sequence (a : ℝ) (n : ℕ) (S : ℕ → ℝ)
  (h1 : ∀ n ≥ 2, S n = a * 3^n - 2) : S 2 = 12 :=
by 
  sorry

end a_2_geometric_sequence_l122_122664


namespace arithmetic_geometric_mean_inequality_l122_122774

theorem arithmetic_geometric_mean_inequality (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (x + y) / 2 ≥ Real.sqrt (x * y) := 
  sorry

end arithmetic_geometric_mean_inequality_l122_122774


namespace part1_part2_l122_122143

-- Condition: x = -1 is a solution to 2a + 4x = x + 5a
def is_solution_x (a x : ℤ) : Prop := 2 * a + 4 * x = x + 5 * a

-- Part 1: Prove a = -1 given x = -1
theorem part1 (x : ℤ) (h1 : x = -1) (h2 : is_solution_x a x) : a = -1 :=
by sorry

-- Condition: a = -1
def a_value (a : ℤ) : Prop := a = -1

-- Condition: ay + 6 = 6a + 2y
def equation_in_y (a y : ℤ) : Prop := a * y + 6 = 6 * a + 2 * y

-- Part 2: Prove y = 4 given a = -1
theorem part2 (a y : ℤ) (h1 : a_value a) (h2 : equation_in_y a y) : y = 4 :=
by sorry

end part1_part2_l122_122143


namespace geom_seq_general_formula_find_range_of_lambda_l122_122322

variable {λ : ℝ}

theorem geom_seq_general_formula (a : ℕ → ℝ) (q : ℝ) 
    (h1 : 0 < q ∧ q < 1) 
    (h2 : ∀ n, a (n + 1) = a n * q) 
    (h3 : a 0 + a 0 * q = 12) 
    (h4 : 2 * (a 0 * q + 1) = a 0 + a 0 * q ^ 2) 
    : ∀ n, a n = (1 / 2)^(n - 4) := 
sorry

theorem find_range_of_lambda (a : ℕ → ℝ) (λ : ℝ) 
    (h1 : ∀ n, a n = (1 / 2) ^ (n - 4)) 
    (h2 : ∀ n, a n * (n - λ) > a (n - 1) * (n - 1 - λ)) 
    : λ < 2 := 
sorry

end geom_seq_general_formula_find_range_of_lambda_l122_122322


namespace greatest_product_l122_122528

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l122_122528


namespace greatest_product_sum_300_l122_122416

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l122_122416


namespace find_a7_of_arithmetic_sequence_l122_122300

noncomputable def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + d * (n - 1)

theorem find_a7_of_arithmetic_sequence (a d : ℤ)
  (h : arithmetic_sequence a d 1 + arithmetic_sequence a d 2 +
       arithmetic_sequence a d 12 + arithmetic_sequence a d 13 = 24) :
  arithmetic_sequence a d 7 = 6 :=
by
  sorry

end find_a7_of_arithmetic_sequence_l122_122300


namespace middle_digit_base_5_reversed_in_base_8_l122_122735

theorem middle_digit_base_5_reversed_in_base_8 (a b c : ℕ) (h₁ : 0 ≤ a ∧ a ≤ 4) (h₂ : 0 ≤ b ∧ b ≤ 4) 
  (h₃ : 0 ≤ c ∧ c ≤ 4) (h₄ : 25 * a + 5 * b + c = 64 * c + 8 * b + a) : b = 3 := 
by 
  sorry

end middle_digit_base_5_reversed_in_base_8_l122_122735


namespace find_m_value_l122_122805

noncomputable def pyramid_property (m : ℕ) : Prop :=
  let n1 := 3
  let n2 := 9
  let n3 := 6
  let r2_1 := m + n1
  let r2_2 := n1 + n2
  let r2_3 := n2 + n3
  let r3_1 := r2_1 + r2_2
  let r3_2 := r2_2 + r2_3
  let top := r3_1 + r3_2
  top = 54

theorem find_m_value : ∃ m : ℕ, pyramid_property m ∧ m = 12 := by
  sorry

end find_m_value_l122_122805


namespace quotient_of_37_div_8_l122_122580

theorem quotient_of_37_div_8 : (37 / 8) = 4 :=
by
  sorry

end quotient_of_37_div_8_l122_122580


namespace max_product_of_two_integers_whose_sum_is_300_l122_122498

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l122_122498


namespace train_crosses_lamp_post_in_30_seconds_l122_122602

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

end train_crosses_lamp_post_in_30_seconds_l122_122602


namespace stocks_higher_price_l122_122944

theorem stocks_higher_price
  (total_stocks : ℕ)
  (percent_increase : ℝ)
  (H L : ℝ)
  (H_eq : H = 1.35 * L)
  (sum_eq : H + L = 4200)
  (percent_increase_eq : percent_increase = 0.35)
  (total_stocks_eq : ↑total_stocks = 4200) :
  total_stocks = 2412 :=
by 
  sorry

end stocks_higher_price_l122_122944


namespace max_product_of_two_integers_whose_sum_is_300_l122_122495

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l122_122495


namespace probability_red_or_white_l122_122582

def total_marbles : ℕ := 50
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

theorem probability_red_or_white : 
  (red_marbles + white_marbles) / total_marbles = 9 / 10 := 
  sorry

end probability_red_or_white_l122_122582


namespace regular_polygon_sides_l122_122867

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, i < n → (180 * (n - 2) / n) = 174) : n = 60 := by
  sorry

end regular_polygon_sides_l122_122867


namespace driving_trip_hours_l122_122937

def total_driving_hours (jade_hours_per_day : ℕ) (jade_days : ℕ) (krista_hours_per_day : ℕ) (krista_days : ℕ) : ℕ :=
  (jade_hours_per_day * jade_days) + (krista_hours_per_day * krista_days)

theorem driving_trip_hours :
  total_driving_hours 8 3 6 3 = 42 :=
by
  rw [total_driving_hours, Nat.mul_add, ←Nat.mul_assoc, ←Nat.mul_assoc, Nat.add_comm (8*3), Nat.add_assoc]
  -- Additional steps to simplify (8 * 3) + (6 * 3) into 42
  sorry

end driving_trip_hours_l122_122937


namespace find_number_l122_122725

theorem find_number (x : ℝ) (h : 0.9 * x = 0.0063) : x = 0.007 := 
by {
  sorry
}

end find_number_l122_122725


namespace xy_identity_l122_122916

theorem xy_identity (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 := by
  sorry

end xy_identity_l122_122916


namespace max_product_two_integers_l122_122460

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l122_122460


namespace pastries_selection_l122_122615

/--
Clara wants to purchase six pastries from an ample supply of five types: muffins, eclairs, croissants, scones, and turnovers. 
Prove that there are 210 possible selections using the stars and bars theorem.
-/
theorem pastries_selection : ∃ (selections : ℕ), selections = (Nat.choose (6 + 5 - 1) (5 - 1)) ∧ selections = 210 := by
  sorry

end pastries_selection_l122_122615


namespace value_of_x_plus_y_l122_122294

theorem value_of_x_plus_y (x y : ℤ) (hx : x = -3) (hy : |y| = 5) : x + y = 2 ∨ x + y = -8 := by
  sorry

end value_of_x_plus_y_l122_122294


namespace find_abs_xyz_l122_122042

noncomputable def distinct_nonzero_real (x y z : ℝ) : Prop :=
x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 

theorem find_abs_xyz
  (x y z : ℝ)
  (h1 : distinct_nonzero_real x y z)
  (h2 : x + 1/y = y + 1/z)
  (h3 : y + 1/z = z + 1/x + 1) :
  |x * y * z| = 1 :=
sorry

end find_abs_xyz_l122_122042


namespace tetrahedron_angle_AB_CD_l122_122678

-- Define the points in the tetrahedron
variables {A B C D : EuclideanSpace ℝ (Fin 3)}

-- Define the length conditions
def AB := 15
def CD := 15
def BD := 20
def AC := 20
def AD := Real.sqrt 337
def BC := Real.sqrt 337

-- Define the problem statement
theorem tetrahedron_angle_AB_CD :
  ∀ (A B C D : EuclideanSpace ℝ (Fin 3)),
    dist A B = AB →
    dist C D = CD →
    dist B D = BD →
    dist A C = AC →
    dist A D = AD →
    dist B C = BC →
    ∃ (θ : ℝ), cos θ = - 7 / 25 :=
by
  intros A B C D hAB hCD hBD hAC hAD hBC
  -- skip the proof with sorry
  sorry

end tetrahedron_angle_AB_CD_l122_122678


namespace bianca_ate_candy_l122_122631

theorem bianca_ate_candy (original_candies : ℕ) (pieces_per_pile : ℕ) 
                         (number_of_piles : ℕ) 
                         (remaining_candies : ℕ) 
                         (h_original : original_candies = 78) 
                         (h_pieces_per_pile : pieces_per_pile = 8) 
                         (h_number_of_piles : number_of_piles = 6) 
                         (h_remaining : remaining_candies = pieces_per_pile * number_of_piles) :
  original_candies - remaining_candies = 30 := by
  subst_vars
  sorry

end bianca_ate_candy_l122_122631


namespace geometric_series_sum_l122_122617

variable (a r : ℤ) (n : ℕ) 

theorem geometric_series_sum :
  a = -1 ∧ r = 2 ∧ n = 10 →
  (a * (r^n - 1) / (r - 1)) = -1023 := 
by
  intro h
  rcases h with ⟨ha, hr, hn⟩
  sorry

end geometric_series_sum_l122_122617


namespace boat_speed_still_water_l122_122092

theorem boat_speed_still_water (b s : ℝ) (h1 : b + s = 21) (h2 : b - s = 9) : b = 15 := 
by 
  -- Solve the system of equations
  sorry

end boat_speed_still_water_l122_122092


namespace value_of_x2_plus_y2_l122_122908

theorem value_of_x2_plus_y2 (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 :=
by
  sorry

end value_of_x2_plus_y2_l122_122908


namespace correct_options_l122_122113

open Real

def option_A (x : ℝ) : Prop :=
  x^2 - 2*x + 1 > 0

def option_B : Prop :=
  ∃ (x : ℝ), (0 < x) ∧ (x + 4 / x = 6)

def option_C (a b : ℝ) : Prop :=
  (a ≠ 0) ∧ (b ≠ 0) → (b / a + a / b ≥ 2)

def option_D (x y : ℝ) : Prop :=
  (0 < x) ∧ (0 < y) ∧ (x + 2*y = 1) → (2 / x + 1 / y ≥ 8)

theorem correct_options :
  ¬(∀ (x : ℝ), option_A x) ∧ (option_B ∧ (∀ (a b : ℝ), option_C a b) = false ∧ 
  (∀ (x y : ℝ), option_D x y)) :=
by sorry

end correct_options_l122_122113


namespace largest_angle_of_convex_hexagon_l122_122103

theorem largest_angle_of_convex_hexagon 
  (x : ℝ) 
  (hx : (x + 2) + (2 * x - 1) + (3 * x + 1) + (4 * x - 2) + (5 * x + 3) + (6 * x - 4) = 720) :
  6 * x - 4 = 202 :=
sorry

end largest_angle_of_convex_hexagon_l122_122103


namespace max_product_sum_300_l122_122482

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l122_122482


namespace greatest_product_sum_300_l122_122417

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l122_122417


namespace part_one_union_sets_l122_122021

theorem part_one_union_sets (a : ℝ) (A B : Set ℝ) :
  (a = 2) →
  A = {x | x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0} →
  B = {x | -2 < x ∧ x < 2} →
  A ∪ B = {x | -2 < x ∧ x ≤ 3} :=
by
  sorry

end part_one_union_sets_l122_122021


namespace max_product_300_l122_122545

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l122_122545


namespace greatest_product_obtainable_l122_122397

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l122_122397


namespace abc_sum_zero_l122_122659

theorem abc_sum_zero
  (a b c : ℝ)
  (h1 : ∀ x: ℝ, (a * (c * x^2 + b * x + a)^2 + b * (c * x^2 + b * x + a) + c = x)) :
  (a + b + c = 0) :=
by
  sorry

end abc_sum_zero_l122_122659


namespace max_product_of_sum_300_l122_122479

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l122_122479


namespace greatest_product_obtainable_l122_122396

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l122_122396


namespace max_product_of_sum_300_l122_122431

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l122_122431


namespace max_tied_teams_round_robin_l122_122673

theorem max_tied_teams_round_robin (n : ℕ) (h: n = 8) :
  ∃ k, (k <= n) ∧ (∀ m, m > k → k * m < n * (n - 1) / 2) :=
by
  sorry

end max_tied_teams_round_robin_l122_122673


namespace eggs_remaining_l122_122183

-- Assign the given constants
def hens : ℕ := 3
def eggs_per_hen_per_day : ℕ := 3
def days_gone : ℕ := 7
def eggs_taken_by_neighbor : ℕ := 12
def eggs_dropped_by_myrtle : ℕ := 5

-- Calculate the expected number of eggs Myrtle should have
noncomputable def total_eggs :=
  hens * eggs_per_hen_per_day * days_gone - eggs_taken_by_neighbor - eggs_dropped_by_myrtle

-- Prove that the total number of eggs equals the correct answer
theorem eggs_remaining : total_eggs = 46 :=
by
  sorry

end eggs_remaining_l122_122183


namespace probability_even_sum_le_8_l122_122083

theorem probability_even_sum_le_8 (P : ProbabilityMassFunction (ℕ × ℕ)) :
P.support = {x | 1 ≤ x.1 ∧ x.1 ≤ 6 ∧ 1 ≤ x.2 ∧ x.2 ≤ 6} →
P.probability {x | (x.1 + x.2) % 2 = 0 ∧ (x.1 + x.2) ≤ 8} = 1 / 3 :=
by sorry

end probability_even_sum_le_8_l122_122083


namespace quadratic_equation_m_value_l122_122832

theorem quadratic_equation_m_value (m : ℝ) (h : m ≠ 2) : m = -2 :=
by
  -- details of the proof go here
  sorry

end quadratic_equation_m_value_l122_122832


namespace intersection_points_sum_l122_122197

theorem intersection_points_sum (x1 x2 x3 y1 y2 y3 A B : ℝ)
(h1 : y1 = x1^3 - 3 * x1 + 2)
(h2 : x1 + 6 * y1 = 6)
(h3 : y2 = x2^3 - 3 * x2 + 2)
(h4 : x2 + 6 * y2 = 6)
(h5 : y3 = x3^3 - 3 * x3 + 2)
(h6 : x3 + 6 * y3 = 6)
(hA : A = x1 + x2 + x3)
(hB : B = y1 + y2 + y3) :
A = 0 ∧ B = 3 := 
by
  sorry

end intersection_points_sum_l122_122197


namespace magnitude_of_v_l122_122320

theorem magnitude_of_v (u v : ℂ) (h1 : u * v = 20 - 15 * complex.i) (h2 : complex.abs u = 5) : complex.abs v = 5 :=
sorry

end magnitude_of_v_l122_122320


namespace max_product_300_l122_122539

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l122_122539


namespace max_product_300_l122_122542

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l122_122542


namespace remainder_4_power_100_div_9_l122_122983

theorem remainder_4_power_100_div_9 : (4^100) % 9 = 4 :=
by
  sorry

end remainder_4_power_100_div_9_l122_122983


namespace greatest_product_of_two_integers_with_sum_300_l122_122509

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l122_122509


namespace denomination_is_20_l122_122081

noncomputable def denomination_of_250_coins (x : ℕ) : Prop :=
  250 * x + 84 * 25 = 7100

theorem denomination_is_20 (x : ℕ) (h : denomination_of_250_coins x) : x = 20 :=
by
  sorry

end denomination_is_20_l122_122081


namespace greatest_product_sum_300_l122_122413

theorem greatest_product_sum_300 (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
by sorry

end greatest_product_sum_300_l122_122413


namespace person_B_completion_time_l122_122207

variables {A B : ℝ} (H : A + B = 1/6 ∧ (A + 10 * B = 1/6))

theorem person_B_completion_time :
    (1 / (1 - 2 * (A + B)) / B = 15) :=
by
  sorry

end person_B_completion_time_l122_122207


namespace obtuse_equilateral_triangle_impossible_l122_122225

-- Define a scalene triangle 
def is_scalene_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ A + B + C = 180

-- Define acute triangles
def is_acute_triangle (A B C : ℝ) : Prop :=
  A < 90 ∧ B < 90 ∧ C < 90

-- Define right triangles
def is_right_triangle (A B C : ℝ) : Prop :=
  A = 90 ∨ B = 90 ∨ C = 90

-- Define isosceles triangles
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∨ a = c ∨ b = c)

-- Define obtuse triangles
def is_obtuse_triangle (A B C : ℝ) : Prop :=
  A > 90 ∨ B > 90 ∨ C > 90

-- Define equilateral triangles
def is_equilateral_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = b ∧ b = c ∧ c = a ∧ A = 60 ∧ B = 60 ∧ C = 60

theorem obtuse_equilateral_triangle_impossible :
  ¬ ∃ (a b c A B C : ℝ), is_equilateral_triangle a b c A B C ∧ is_obtuse_triangle A B C :=
by
  sorry

end obtuse_equilateral_triangle_impossible_l122_122225


namespace product_of_solutions_l122_122891

theorem product_of_solutions : (∃ x : ℝ, |x| = 3*(|x| - 2)) → (x = 3 ∨ x = -3) → 3 * -3 = -9 :=
by sorry

end product_of_solutions_l122_122891


namespace triangle_area_l122_122030

/-- 
In a triangle ABC, given that ∠B=30°, AB=2√3, and AC=2, 
prove that the area of the triangle ABC is either √3 or 2√3.
 -/
theorem triangle_area (B : Real) (AB AC : Real) 
  (h_B : B = 30) (h_AB : AB = 2 * Real.sqrt 3) (h_AC : AC = 2) :
  ∃ S : Real, (S = Real.sqrt 3 ∨ S = 2 * Real.sqrt 3) := 
by 
  sorry

end triangle_area_l122_122030


namespace probability_sqrt_less_nine_l122_122552

theorem probability_sqrt_less_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  ∃ (p : ℚ), p = (finset.card (valid_numbers.to_finset) : ℚ) / (finset.card (two_digit_numbers.to_finset) : ℚ) ∧ p = 71 / 90 :=
by
  sorry

end probability_sqrt_less_nine_l122_122552


namespace sum_of_x_and_y_l122_122923

theorem sum_of_x_and_y (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
(hx15 : x < 15) (hy15 : y < 15) (h : x + y + x * y = 119) : x + y = 21 ∨ x + y = 20 := 
by
  sorry

end sum_of_x_and_y_l122_122923


namespace greatest_product_of_two_integers_with_sum_300_l122_122517

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l122_122517


namespace digits_of_result_l122_122896

theorem digits_of_result 
  (u1 u2 t1 t2 h1 h2 : ℕ) 
  (hu_condition : u1 = u2 + 6)
  (units_column : u1 - u2 = 5)
  (tens_column : t1 - t2 = 9)
  (no_borrowing : u2 < u1) 
  : (h1, u1 - u2) = (4, 5) := 
sorry

end digits_of_result_l122_122896


namespace kitchen_upgrade_cost_l122_122214

-- Define the number of cabinet knobs and their cost
def num_knobs : ℕ := 18
def cost_per_knob : ℝ := 2.50

-- Define the number of drawer pulls and their cost
def num_pulls : ℕ := 8
def cost_per_pull : ℝ := 4.00

-- Calculate the total cost of the knobs
def total_cost_knobs : ℝ := num_knobs * cost_per_knob

-- Calculate the total cost of the pulls
def total_cost_pulls : ℝ := num_pulls * cost_per_pull

-- Calculate the total cost of the kitchen upgrade
def total_cost : ℝ := total_cost_knobs + total_cost_pulls

-- Theorem statement
theorem kitchen_upgrade_cost : total_cost = 77 := by
  sorry

end kitchen_upgrade_cost_l122_122214


namespace probability_sqrt_lt_nine_l122_122553

theorem probability_sqrt_lt_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  let probability := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  probability = 71 / 90 :=
by
  let two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n : ℕ | 10 ≤ n ∧ n < 81}
  have h1 : two_digit_numbers.card = 90 := sorry
  have h2 : valid_numbers.card = 71 := sorry
  let probability : ℚ := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  rw [h1, h2]
  simp
  norm_num

end probability_sqrt_lt_nine_l122_122553


namespace find_smallest_c_plus_d_l122_122310

noncomputable def smallest_c_plus_d (c d : ℝ) :=
  c + d

theorem find_smallest_c_plus_d (c d : ℝ) (hc : 0 < c) (hd : 0 < d)
  (h1 : c ^ 2 ≥ 12 * d)
  (h2 : 9 * d ^ 2 ≥ 4 * c) :
  smallest_c_plus_d c d = 16 / 3 :=
by
  sorry

end find_smallest_c_plus_d_l122_122310


namespace find_natural_number_l122_122885

theorem find_natural_number (x : ℕ) (y z : ℤ) (hy : x = 2 * y^2 - 1) (hz : x^2 = 2 * z^2 - 1) : x = 1 ∨ x = 7 :=
sorry

end find_natural_number_l122_122885


namespace triangle_minimum_perimeter_l122_122799

/--
In a triangle ABC where sides have integer lengths such that no two sides are equal, let ω be a circle with its center at the incenter of ΔABC. Suppose one excircle is tangent to AB and internally tangent to ω, while excircles tangent to AC and BC are externally tangent to ω.
Prove that the minimum possible perimeter of ΔABC is 12.
-/
theorem triangle_minimum_perimeter {a b c : ℕ} (h1 : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
    (h2 : ∀ (r rA rB rC s : ℝ),
      rA = r * s / (s - a) → rB = r * s / (s - b) → rC = r * s / (s - c) →
      r + rA = rB ∧ r + rA = rC) :
  a + b + c = 12 :=
sorry

end triangle_minimum_perimeter_l122_122799


namespace greatest_product_of_sum_eq_300_l122_122386

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l122_122386


namespace negation_of_existential_statement_l122_122072

theorem negation_of_existential_statement :
  (¬ ∃ x : ℝ, x ≥ 1 ∨ x > 2) ↔ ∀ x : ℝ, x < 1 :=
by
  sorry

end negation_of_existential_statement_l122_122072


namespace greatest_whole_number_difference_l122_122026

theorem greatest_whole_number_difference (x y : ℤ) (hx1 : 7 < x) (hx2 : x < 9) (hy1 : 9 < y) (hy2 : y < 15) : y - x = 6 :=
by
  sorry

end greatest_whole_number_difference_l122_122026


namespace greatest_product_of_sum_eq_300_l122_122393

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l122_122393


namespace bacon_needed_l122_122247

def eggs_per_plate : ℕ := 2
def bacon_per_plate : ℕ := 2 * eggs_per_plate
def customers : ℕ := 14
def bacon_total (eggs_per_plate bacon_per_plate customers : ℕ) : ℕ := customers * bacon_per_plate

theorem bacon_needed : bacon_total eggs_per_plate bacon_per_plate customers = 56 :=
by
  sorry

end bacon_needed_l122_122247


namespace steve_oranges_count_l122_122613

variable (Brian_oranges Marcie_oranges Shawn_oranges Steve_oranges : ℝ)

def oranges_conditions : Prop :=
  (Marcie_oranges = 12) ∧
  (Brian_oranges = Marcie_oranges) ∧
  (Shawn_oranges = 1.075 * (Brian_oranges + Marcie_oranges)) ∧
  (Steve_oranges = 3 * (Marcie_oranges + Brian_oranges + Shawn_oranges))

theorem steve_oranges_count (h : oranges_conditions Brian_oranges Marcie_oranges Shawn_oranges Steve_oranges) :
  Steve_oranges = 149.4 :=
sorry

end steve_oranges_count_l122_122613


namespace total_children_on_playground_l122_122999

theorem total_children_on_playground (girls boys : ℕ) (h_girls : girls = 28) (h_boys : boys = 35) : girls + boys = 63 := 
by 
  sorry

end total_children_on_playground_l122_122999


namespace incorrect_statement_l122_122620

def vector_mult (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.2 - a.2 * b.1

theorem incorrect_statement (a b : ℝ × ℝ) : vector_mult a b ≠ vector_mult b a :=
by
  sorry

end incorrect_statement_l122_122620


namespace probability_sqrt_less_than_nine_l122_122558

theorem probability_sqrt_less_than_nine :
  let S := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let T := {n : ℕ | n ∈ S ∧ n < 81}
  (T.card : ℚ) / S.card = 71 / 90 :=
by
  sorry

end probability_sqrt_less_than_nine_l122_122558


namespace range_of_func_l122_122704

noncomputable def func (x : ℝ) : ℝ := 1 / (x - 1)

theorem range_of_func :
  (∀ y : ℝ, 
    (∃ x : ℝ, (x < 1 ∨ (2 ≤ x ∧ x < 5)) ∧ y = func x) ↔ 
    (y < 0 ∨ (1/4 < y ∧ y ≤ 1))) :=
by
  sorry

end range_of_func_l122_122704


namespace t_shaped_region_slope_divides_area_in_half_l122_122802

theorem t_shaped_region_slope_divides_area_in_half :
  ∃ (m : ℚ), (m = 4 / 11) ∧ (
    let area1 := 2 * (m * 2 * 4)
    let area2 := ((4 - m * 2) * 4) + 6
    area1 = area2
  ) :=
by
  sorry

end t_shaped_region_slope_divides_area_in_half_l122_122802


namespace problem1_coefficient_of_x_problem2_maximum_coefficient_term_l122_122096

-- Problem 1: Coefficient of x term
theorem problem1_coefficient_of_x (n : ℕ) 
  (A : ℕ := (3 + 1)^n) 
  (B : ℕ := 2^n) 
  (h1 : A + B = 272) 
  : true :=  -- Replacing true with actual condition
by sorry

-- Problem 2: Term with maximum coefficient
theorem problem2_maximum_coefficient_term (n : ℕ)
  (h : 1 + n + (n * (n - 1)) / 2 = 79) 
  : true :=  -- Replacing true with actual condition
by sorry

end problem1_coefficient_of_x_problem2_maximum_coefficient_term_l122_122096


namespace greatest_product_sum_300_l122_122379

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l122_122379


namespace expand_and_simplify_l122_122758

theorem expand_and_simplify (x : ℝ) : 6 * (x - 3) * (x + 10) = 6 * x^2 + 42 * x - 180 :=
by
  sorry

end expand_and_simplify_l122_122758


namespace government_subsidy_per_hour_l122_122614

-- Given conditions:
def cost_first_employee : ℕ := 20
def cost_second_employee : ℕ := 22
def hours_per_week : ℕ := 40
def weekly_savings : ℕ := 160

-- To prove:
theorem government_subsidy_per_hour (S : ℕ) : S = 2 :=
by
  -- Proof steps go here.
  sorry

end government_subsidy_per_hour_l122_122614


namespace max_product_sum_300_l122_122481

theorem max_product_sum_300 : ∃ (x : ℤ), x * (300 - x) = 22500 :=
by
  sorry

end max_product_sum_300_l122_122481


namespace area_of_triangle_is_right_angled_l122_122820

noncomputable def vector_a : ℝ × ℝ := (3, 4)
noncomputable def vector_b : ℝ × ℝ := (-4, 3)

theorem area_of_triangle_is_right_angled (h1 : vector_a = (3, 4)) (h2 : vector_b = (-4, 3)) : 
  let det := vector_a.1 * vector_b.2 - vector_a.2 * vector_b.1
  (1 / 2) * abs det = 12.5 :=
by
  sorry

end area_of_triangle_is_right_angled_l122_122820


namespace find_smallest_c_plus_d_l122_122311

noncomputable def smallest_c_plus_d (c d : ℝ) :=
  c + d

theorem find_smallest_c_plus_d (c d : ℝ) (hc : 0 < c) (hd : 0 < d)
  (h1 : c ^ 2 ≥ 12 * d)
  (h2 : 9 * d ^ 2 ≥ 4 * c) :
  smallest_c_plus_d c d = 16 / 3 :=
by
  sorry

end find_smallest_c_plus_d_l122_122311


namespace jerry_removed_old_figures_l122_122941

-- Let's declare the conditions
variables (initial_count added_count current_count removed_count : ℕ)
variables (h1 : initial_count = 7)
variables (h2 : added_count = 11)
variables (h3 : current_count = 8)

-- The statement to prove
theorem jerry_removed_old_figures : removed_count = initial_count + added_count - current_count :=
by
  -- The proof will go here, but we'll use sorry to skip it
  sorry

end jerry_removed_old_figures_l122_122941


namespace max_product_of_sum_300_l122_122433

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l122_122433


namespace greatest_product_of_sum_eq_300_l122_122394

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l122_122394


namespace minimum_value_l122_122683

noncomputable def min_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : ℝ :=
  (x + y + z) * (1 / (x + y) + 1 / (x + z) + 1 / (y + z))

theorem minimum_value : ∀ x y z : ℝ, 0 < x → 0 < y → 0 < z →
  (x + y + z) * (1 / (x + y) + 1 / (x + z) + 1 / (y + z)) ≥ 9 / 2 :=
by
  intro x y z hx hy hz
  sorry

end minimum_value_l122_122683


namespace greatest_product_of_two_integers_with_sum_300_l122_122521

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l122_122521


namespace max_product_of_two_integers_whose_sum_is_300_l122_122494

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l122_122494


namespace count_multiples_of_12_l122_122285

theorem count_multiples_of_12 (a b : ℤ) (h1 : 15 < a) (h2 : b < 205) (h3 : ∃ k : ℤ, a = 12 * k) (h4 : ∃ k : ℤ, b = 12 * k) : 
  ∃ n : ℕ, n = 16 := 
by 
  sorry

end count_multiples_of_12_l122_122285


namespace simplify_polynomial_l122_122220

-- Define the original polynomial
def original_expr (x : ℝ) : ℝ := 3 - 5 * x - 7 * x^2 + 9 + 11 * x - 13 * x^2 - 15 + 17 * x + 19 * x^2 + 2 * x^3

-- Define the simplified version of the polynomial
def simplified_expr (x : ℝ) : ℝ := 2 * x^3 - x^2 + 23 * x - 3

-- State the theorem that the original expression is equal to the simplified one
theorem simplify_polynomial (x : ℝ) : original_expr x = simplified_expr x := 
by 
  sorry

end simplify_polynomial_l122_122220


namespace age_ratio_l122_122708

theorem age_ratio 
    (a m s : ℕ) 
    (h1 : m = 60) 
    (h2 : m = 3 * a) 
    (h3 : s = 40) : 
    (m + a) / s = 2 :=
by
    sorry

end age_ratio_l122_122708


namespace div_sub_eq_l122_122877

theorem div_sub_eq : 0.24 / 0.004 - 0.1 = 59.9 := by
  sorry

end div_sub_eq_l122_122877


namespace greatest_product_of_sum_eq_300_l122_122388

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l122_122388


namespace paintable_wall_area_correct_l122_122304

noncomputable def paintable_wall_area : Nat :=
  let length := 15
  let width := 11
  let height := 9
  let closet_width := 3
  let closet_length := 4
  let unused_area := 70
  let room_wall_area :=
    2 * (length * height) +
    2 * (width * height)
  let closet_wall_area := 
    2 * (closet_width * height)
  let paintable_area_per_bedroom := 
    room_wall_area - (unused_area + closet_wall_area)
  4 * paintable_area_per_bedroom

theorem paintable_wall_area_correct : paintable_wall_area = 1376 := by
  sorry

end paintable_wall_area_correct_l122_122304


namespace tennis_tournament_handshakes_l122_122608

theorem tennis_tournament_handshakes :
  ∃ (number_of_handshakes : ℕ),
    let total_women := 8 in
    let handshakes_per_woman := 6 in
    let total_handshakes_counted_twice := total_women * handshakes_per_woman in
    number_of_handshakes = total_handshakes_counted_twice / 2 :=
begin
  use 24,
  unfold total_women handshakes_per_woman total_handshakes_counted_twice,
  norm_num,
end

end tennis_tournament_handshakes_l122_122608


namespace greatest_product_of_sum_eq_300_l122_122390

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l122_122390


namespace triangle_ABC_area_median_AD_length_l122_122034

-- Definitions and conditions
def AB : ℝ := 30
def AC : ℝ := 40
def angle_A : ℝ := 90 -- degrees

-- Areas to prove: area of triangle ABC and length of median AD
def area_ABC (AB AC : ℝ) : ℝ := 1 / 2 * AB * AC
def length_median (AB AC : ℝ) : ℝ := 1 / 2 * Real.sqrt (2 * (AB ^ 2 + AC ^ 2))

-- Theorem statements
theorem triangle_ABC_area : area_ABC AB AC = 600 :=
by sorry

theorem median_AD_length : length_median AB AC ≈ 35.36 :=
by sorry

end triangle_ABC_area_median_AD_length_l122_122034


namespace multiplication_is_247_l122_122168

theorem multiplication_is_247 (a b : ℕ) (ha : 10 ≤ a ∧ a < 100) (hb : 10 ≤ b ∧ b < 100) (h : a * b = 247) : 
a = 13 ∧ b = 19 :=
by sorry

end multiplication_is_247_l122_122168


namespace greatest_product_l122_122535

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l122_122535


namespace greatest_product_of_sum_eq_300_l122_122392

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l122_122392


namespace max_product_of_two_integers_whose_sum_is_300_l122_122503

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l122_122503


namespace negation_of_forall_x_gt_1_l122_122073

theorem negation_of_forall_x_gt_1 : ¬(∀ x : ℝ, x^2 > 1) ↔ (∃ x : ℝ, x^2 ≤ 1) := by
  sorry

end negation_of_forall_x_gt_1_l122_122073


namespace center_of_circle_l122_122062

theorem center_of_circle (h k : ℝ) :
  (∀ x y : ℝ, (x - 3) ^ 2 + (y - 4) ^ 2 = 10 ↔ x ^ 2 + y ^ 2 = 6 * x + 8 * y - 15) → 
  h + k = 7 :=
sorry

end center_of_circle_l122_122062


namespace max_product_two_integers_l122_122462

theorem max_product_two_integers (x y : ℤ) (h : x + y = 300) : x * y ≤ 22500 :=
sorry

end max_product_two_integers_l122_122462


namespace f_geq_expression_l122_122647

noncomputable def f (x a : ℝ) : ℝ := x^2 + (2 * a - 1 / a) * x - Real.log x

theorem f_geq_expression (a x : ℝ) (h : a < 0) : f x a ≥ (1 - 2 * a) * (a + 1) := 
  sorry

end f_geq_expression_l122_122647


namespace other_asymptote_l122_122959

/-- Problem Statement:
One of the asymptotes of a hyperbola is y = 2x. The foci have the same 
x-coordinate, which is 4. Prove that the equation of the other asymptote
of the hyperbola is y = -2x + 16.
-/
theorem other_asymptote (focus_x : ℝ) (asymptote1: ℝ → ℝ) (asymptote2 : ℝ → ℝ) :
  focus_x = 4 →
  (∀ x, asymptote1 x = 2 * x) →
  (asymptote2 4 = 8) → 
  (∀ x, asymptote2 x = -2 * x + 16) :=
sorry

end other_asymptote_l122_122959


namespace polynomial_value_l122_122024

variable (a b : ℝ)

theorem polynomial_value :
  2 * a + 3 * b = 5 → 6 * a + 9 * b - 12 = 3 :=
by
  intro h
  sorry

end polynomial_value_l122_122024


namespace jack_years_after_son_death_l122_122305

noncomputable def jackAdolescenceTime (L : Real) : Real := (1 / 6) * L
noncomputable def jackFacialHairTime (L : Real) : Real := (1 / 12) * L
noncomputable def jackMarriageTime (L : Real) : Real := (1 / 7) * L
noncomputable def jackSonBornTime (L : Real) (marriageTime : Real) : Real := marriageTime + 5
noncomputable def jackSonLifetime (L : Real) : Real := (1 / 2) * L
noncomputable def jackSonDeathTime (bornTime : Real) (sonLifetime : Real) : Real := bornTime + sonLifetime
noncomputable def yearsAfterSonDeath (L : Real) (sonDeathTime : Real) : Real := L - sonDeathTime

theorem jack_years_after_son_death : 
  yearsAfterSonDeath 84 
    (jackSonDeathTime (jackSonBornTime 84 (jackMarriageTime 84)) (jackSonLifetime 84)) = 4 :=
by
  sorry

end jack_years_after_son_death_l122_122305


namespace system1_solution_system2_solution_l122_122962

theorem system1_solution :
  ∃ (x y : ℝ), 3 * x - 2 * y = -1 ∧ 2 * x + 3 * y = 8 ∧ x = 1 ∧ y = 2 :=
by {
  -- Proof skipped
  sorry
}

theorem system2_solution :
  ∃ (x y : ℝ), 2 * x + y = 1 ∧ 2 * x - y = 7 ∧ x = 2 ∧ y = -3 :=
by {
  -- Proof skipped
  sorry
}

end system1_solution_system2_solution_l122_122962


namespace length_of_CD_l122_122960

theorem length_of_CD (x y : ℝ) (h1 : x / (3 + y) = 3 / 5) (h2 : (x + 3) / y = 4 / 7) (h3 : x + 3 + y = 273.6) : 3 + y = 273.6 :=
by
  sorry

end length_of_CD_l122_122960


namespace total_feet_is_correct_l122_122238

-- definitions according to conditions
def number_of_heads := 46
def number_of_hens := 24
def number_of_cows := number_of_heads - number_of_hens
def hen_feet := 2
def cow_feet := 4
def total_hen_feet := number_of_hens * hen_feet
def total_cow_feet := number_of_cows * cow_feet
def total_feet := total_hen_feet + total_cow_feet

-- proof statement with sorry
theorem total_feet_is_correct : total_feet = 136 :=
by
  sorry

end total_feet_is_correct_l122_122238


namespace annie_purchases_l122_122604

theorem annie_purchases (x y z : ℕ) 
  (h1 : x + y + z = 50) 
  (h2 : 20 * x + 400 * y + 500 * z = 5000) :
  x = 40 :=
by sorry

end annie_purchases_l122_122604


namespace max_product_of_two_integers_whose_sum_is_300_l122_122497

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l122_122497


namespace john_adds_and_subtracts_l122_122713

theorem john_adds_and_subtracts :
  (41^2 = 40^2 + 81) ∧ (39^2 = 40^2 - 79) :=
by {
  sorry
}

end john_adds_and_subtracts_l122_122713


namespace max_product_of_two_integers_whose_sum_is_300_l122_122502

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l122_122502


namespace greatest_product_sum_300_l122_122375

theorem greatest_product_sum_300 : ∃ x y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  use 150, 150
  constructor
  · exact rfl
  · exact rfl

end greatest_product_sum_300_l122_122375


namespace exists_point_D_iff_sin_inequality_l122_122056

-- Define assumptions about triangle ABC
variables {A B C : ℝ} -- Angles A, B, and C in radians
variables (h_triangle : A + B + C = Real.pi) -- Sum of angles in a triangle

-- Main theorem: existence of point D on side AB such that CD is the geometric mean 
-- of AD and DB if and only if the inequality holds.
theorem exists_point_D_iff_sin_inequality
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hA_sum : A + B + C = Real.pi) :
  (∃ D : ℝ, -- Existence of point D (need real definition here based on geometry, but we simplify)
      ∀ (AD DB CD : ℝ),
        CD = Real.sqrt (AD * DB) -- CD is the geometric mean of AD and DB
   ) ↔ (Real.sin A * Real.sin B ≤ Real.sin (C / 2) ^ 2) :=
sorry

end exists_point_D_iff_sin_inequality_l122_122056


namespace distance_from_x_axis_l122_122803

theorem distance_from_x_axis (a : ℝ) (h : |a| = 3) : a = 3 ∨ a = -3 := by
  sorry

end distance_from_x_axis_l122_122803


namespace max_product_two_integers_sum_300_l122_122349

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l122_122349


namespace max_product_of_two_integers_whose_sum_is_300_l122_122506

theorem max_product_of_two_integers_whose_sum_is_300 :
  ∃ (a b : ℤ), a + b = 300 ∧ a * b = 22500 :=
by
  sorry

end max_product_of_two_integers_whose_sum_is_300_l122_122506


namespace greatest_product_of_two_integers_with_sum_300_l122_122518

theorem greatest_product_of_two_integers_with_sum_300 :
  ∃ x : ℤ, (∀ y : ℤ, y * (300 - y) ≤ 22500) ∧ x * (300 - x) = 22500 := by
  sorry

end greatest_product_of_two_integers_with_sum_300_l122_122518


namespace coach_mike_change_l122_122616

theorem coach_mike_change (cost amount_given change : ℕ) 
    (h_cost : cost = 58) (h_amount_given : amount_given = 75) : 
    change = amount_given - cost → change = 17 := by
    sorry

end coach_mike_change_l122_122616


namespace sufficient_and_necessary_condition_l122_122243

theorem sufficient_and_necessary_condition (x : ℝ) :
  x^2 - 4 * x ≥ 0 ↔ x ≥ 4 ∨ x ≤ 0 :=
sorry

end sufficient_and_necessary_condition_l122_122243


namespace inequality_proof_l122_122144

theorem inequality_proof (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (h_sum : a + b + c = 1) :
    (1 - a^2)^2 + (1 - b^2)^2 + (1 - c^2)^2 ≥ 2 := 
by 
  sorry

end inequality_proof_l122_122144


namespace Bob_wins_game_l122_122933

theorem Bob_wins_game :
  ∀ (initial_set : Set ℕ),
    47 ∈ initial_set →
    2016 ∈ initial_set →
    (∀ (a b : ℕ), a ∈ initial_set → b ∈ initial_set → a > b → (a - b) ∉ initial_set → (a - b) ∈ initial_set) →
    (∀ (S : Set ℕ), S ⊆ initial_set → ∃ (n : ℕ), ∀ m ∈ S, m > n) → false :=
by
  sorry

end Bob_wins_game_l122_122933


namespace B_monthly_income_is_correct_l122_122199

variable (A_m B_m C_m : ℝ)
variable (A_annual C_m_value : ℝ)
variable (ratio_A_to_B : ℝ)

-- Given conditions
def conditions :=
  A_annual = 537600 ∧
  C_m_value = 16000 ∧
  ratio_A_to_B = 5 / 2 ∧
  A_m = A_annual / 12 ∧
  B_m = (2 / 5) * A_m ∧
  B_m = 1.12 * C_m ∧
  C_m = C_m_value

-- Prove that B's monthly income is Rs. 17920
theorem B_monthly_income_is_correct (h : conditions A_m B_m C_m A_annual C_m_value ratio_A_to_B) : 
  B_m = 17920 :=
by 
  sorry

end B_monthly_income_is_correct_l122_122199


namespace cell_population_l122_122730

variable (n : ℕ)

def a (n : ℕ) : ℕ :=
  if n = 1 then 5
  else 1 -- Placeholder for general definition

theorem cell_population (n : ℕ) : a n = 2^(n-1) + 4 := by
  sorry

end cell_population_l122_122730


namespace regular_polygon_sides_l122_122738

theorem regular_polygon_sides (exterior_angle : ℝ) (total_exterior_angle_sum : ℝ) (h1 : exterior_angle = 18) (h2 : total_exterior_angle_sum = 360) :
  let n := total_exterior_angle_sum / exterior_angle
  n = 20 :=
by
  sorry

end regular_polygon_sides_l122_122738


namespace trees_left_after_typhoon_l122_122283

theorem trees_left_after_typhoon (trees_grown : ℕ) (trees_died : ℕ) (h1 : trees_grown = 17) (h2 : trees_died = 5) : (trees_grown - trees_died = 12) :=
by
  -- The proof would go here
  sorry

end trees_left_after_typhoon_l122_122283


namespace cookies_initial_count_l122_122170

theorem cookies_initial_count (C : ℕ) (h1 : C / 8 = 8) : C = 64 :=
by
  sorry

end cookies_initial_count_l122_122170


namespace zero_cleverly_numbers_l122_122097

theorem zero_cleverly_numbers (n : ℕ) : 
  (1000 ≤ n ∧ n < 10000) ∧ (∃ a b c, n = 1000 * a + 10 * b + c ∧ b = 0 ∧ 9 * (100 * a + 10 * b + c) = n) ↔ (n = 2025 ∨ n = 4050 ∨ n = 6075) := 
sorry

end zero_cleverly_numbers_l122_122097


namespace best_model_is_model1_l122_122302

noncomputable def model_best_fitting (R1 R2 R3 R4 : ℝ) :=
  R1 = 0.975 ∧ R2 = 0.79 ∧ R3 = 0.55 ∧ R4 = 0.25

theorem best_model_is_model1 (R1 R2 R3 R4 : ℝ) (h : model_best_fitting R1 R2 R3 R4) :
  R1 = max R1 (max R2 (max R3 R4)) :=
by
  cases h with
  | intro h1 h_rest =>
    cases h_rest with
    | intro h2 h_rest2 =>
      cases h_rest2 with
      | intro h3 h4 =>
        sorry

end best_model_is_model1_l122_122302


namespace union_of_A_B_l122_122902

def A (p q : ℝ) : Set ℝ := {x | x^2 + p * x + q = 0}
def B (p q : ℝ) : Set ℝ := {x | x^2 - p * x - 2 * q = 0}

theorem union_of_A_B (p q : ℝ)
  (h1 : A p q ∩ B p q = {-1}) :
  A p q ∪ B p q = {-1, -2, 4} := by
sorry

end union_of_A_B_l122_122902


namespace max_product_two_integers_sum_300_l122_122348

theorem max_product_two_integers_sum_300 : 
  ∃ x : ℤ, 0 ≤ x ∧ x ≤ 300 ∧ (x * (300 - x) = 22500) := 
by
  sorry

end max_product_two_integers_sum_300_l122_122348


namespace probability_sqrt_two_digit_lt_nine_correct_l122_122566

noncomputable def probability_sqrt_two_digit_lt_nine : ℚ :=
  let two_digit_integers := finset.Icc 10 99
  let satisfying_integers := two_digit_integers.filter (λ n => n < 81)
  let probability := (satisfying_integers.card : ℚ) / (two_digit_integers.card : ℚ)
  probability

theorem probability_sqrt_two_digit_lt_nine_correct :
  probability_sqrt_two_digit_lt_nine = 71 / 90 := by
  sorry

end probability_sqrt_two_digit_lt_nine_correct_l122_122566


namespace chessboard_marking_checkerboard_l122_122689

theorem chessboard_marking_checkerboard (board : fin 8 × fin 8) (marked_cells : finset (fin 8 × fin 8)) :
  (∀ cell : fin 8 × fin 8, 
    marked_cells.card ≤ 32 ∧ 
    ∃! adjacent : fin 8 × fin 8, 
    adjacent ∈ marked_cells ∧ 
    (abs (cell.1 - adjacent.1) = 1 ∧ cell.2 = adjacent.2 ∨ 
     abs (cell.2 - adjacent.2) = 1 ∧ cell.1 = adjacent.1)) :=
sorry

end chessboard_marking_checkerboard_l122_122689


namespace probability_sqrt_two_digit_lt_nine_correct_l122_122565

noncomputable def probability_sqrt_two_digit_lt_nine : ℚ :=
  let two_digit_integers := finset.Icc 10 99
  let satisfying_integers := two_digit_integers.filter (λ n => n < 81)
  let probability := (satisfying_integers.card : ℚ) / (two_digit_integers.card : ℚ)
  probability

theorem probability_sqrt_two_digit_lt_nine_correct :
  probability_sqrt_two_digit_lt_nine = 71 / 90 := by
  sorry

end probability_sqrt_two_digit_lt_nine_correct_l122_122565


namespace max_product_of_sum_300_l122_122478

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l122_122478


namespace grid_problem_l122_122031

theorem grid_problem 
  (A B : ℕ) 
  (grid : (Fin 3) → (Fin 3) → ℕ)
  (h1 : ∀ i, grid 0 i ≠ grid 1 i)
  (h2 : ∀ i, grid 0 i ≠ grid 2 i)
  (h3 : ∀ i, grid 1 i ≠ grid 2 i)
  (h4 : ∀ i, (∃! x, grid x i = 1))
  (h5 : ∀ i, (∃! x, grid x i = 2))
  (h6 : ∀ i, (∃! x, grid x i = 3))
  (h7 : grid 1 2 = A)
  (h8 : grid 2 2 = B) : 
  A + B + 4 = 8 :=
by sorry

end grid_problem_l122_122031


namespace xy_identity_l122_122915

theorem xy_identity (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : x^2 + y^2 = 6 := by
  sorry

end xy_identity_l122_122915


namespace triangle_area_l122_122076

theorem triangle_area (a b c : ℝ) (h1 : a / b = 3 / 4) (h2 : b / c = 4 / 5) (h3 : a + b + c = 60) : 
  (1/2) * a * b = 150 :=
by
  sorry

end triangle_area_l122_122076


namespace greatest_product_l122_122523

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l122_122523


namespace handshake_problem_l122_122605

theorem handshake_problem (n : ℕ) (hn : n = 11) (H : n * (n - 1) / 2 = 55) : 10 = n - 1 :=
by
  sorry

end handshake_problem_l122_122605


namespace parabola_equation_l122_122330

theorem parabola_equation (a b c : ℝ) (h1 : a^2 = 3) (h2 : b^2 = 1) (h3 : c^2 = a^2 + b^2) : 
  (c = 2) → (vertex = 0) → (focus = 2) → ∀ x y, y^2 = 16 * x := 
by 
  sorry

end parabola_equation_l122_122330


namespace greatest_product_obtainable_l122_122403

theorem greatest_product_obtainable :
  ∃ x : ℤ, ∃ y : ℤ, x + y = 300 ∧ x * y = 22500 :=
by
  sorry

end greatest_product_obtainable_l122_122403


namespace range_of_a_iff_max_at_half_l122_122795

open Real

noncomputable def f (a x : ℝ) : ℝ := log x + a * x^2 - (a + 2) * x

theorem range_of_a_iff_max_at_half {a : ℝ} (ha : 0 < a ∧ a < 2) :
  ∀ x, f a x ≤ f a (1/2) := sorry

end range_of_a_iff_max_at_half_l122_122795


namespace f_1_eq_zero_l122_122898

-- Given a function f with the specified properties
variable {f : ℝ → ℝ}

-- Given 1) the domain of the function
axiom domain_f : ∀ x, (x < 0 ∨ x > 0) → true 

-- Given 2) the functional equation
axiom functional_eq_f : ∀ x₁ x₂, (x₁ < 0 ∨ x₁ > 0) ∧ (x₂ < 0 ∨ x₂ > 0) → f (x₁ * x₂) = f x₁ + f x₂

-- Prove that f(1) = 0
theorem f_1_eq_zero : f 1 = 0 := 
  sorry

end f_1_eq_zero_l122_122898


namespace find_tricycles_l122_122297

noncomputable def number_of_tricycles (w b t : ℕ) : ℕ := t

theorem find_tricycles : ∃ (w b t : ℕ), 
  (w + b + t = 10) ∧ 
  (2 * b + 3 * t = 25) ∧ 
  (number_of_tricycles w b t = 5) :=
  by 
    sorry

end find_tricycles_l122_122297


namespace arithmetic_sequence_30th_term_l122_122968

theorem arithmetic_sequence_30th_term (a1 a2 a3 : ℤ) (h1 : a1 = 3) (h2 : a2 - a1 = 10) (h3 : a3 - a2 = 10) : 
  a1 + 29 * 10 = 293 :=
by
  rw [h1, h2] -- using given conditions
  sorry -- skipping the actual arithmetic steps, placeholder to finish the proof

end arithmetic_sequence_30th_term_l122_122968


namespace max_product_of_sum_300_l122_122432

theorem max_product_of_sum_300 : 
  ∀ (x y : ℤ), x + y = 300 → (x * y) ≤ 22500 ∧ (x * y = 22500 → x = 150 ∧ y = 150) :=
by
  intros x y h
  sorry

end max_product_of_sum_300_l122_122432


namespace arithmetic_sequence_problem_l122_122641

theorem arithmetic_sequence_problem
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (a1 : ℝ)
  (d : ℝ)
  (h1 : d = 2)
  (h2 : ∀ n : ℕ, a n = a1 + (n - 1) * d)
  (h3 :  ∀ n : ℕ, S n = (n * (2 * a1 + (n - 1) * d)) / 2)
  (h4 : S 6 = 3 * S 3) :
  a 9 = 20 :=
by sorry

end arithmetic_sequence_problem_l122_122641


namespace adam_action_figures_per_shelf_l122_122245

-- Define the number of shelves and the total number of action figures
def shelves : ℕ := 4
def total_action_figures : ℕ := 44

-- Define the number of action figures per shelf
def action_figures_per_shelf : ℕ := total_action_figures / shelves

-- State the theorem to be proven
theorem adam_action_figures_per_shelf : action_figures_per_shelf = 11 :=
by sorry

end adam_action_figures_per_shelf_l122_122245


namespace math_problem_l122_122749

theorem math_problem : ((3.6 * 0.3) / 0.6 = 1.8) :=
by
  sorry

end math_problem_l122_122749


namespace hyperbola_properties_l122_122775

-- Definitions from the conditions
def line_l (x y : ℝ) : Prop := 4 * x - 3 * y + 20 = 0
def asymptote_l (x y : ℝ) : Prop := 4 * x - 3 * y = 0
def foci_on_x_axis (x y : ℝ) : Prop := y = 0

-- Standard equation of the hyperbola
def hyperbola_equation (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1

-- Define eccentricity
def eccentricity := 5 / 3

-- Proof statement
theorem hyperbola_properties :
  (∃ x y : ℝ, line_l x y ∧ foci_on_x_axis x y) →
  (∃ x y : ℝ, asymptote_l x y) →
  ∃ x y : ℝ, hyperbola_equation x y ∧ eccentricity = 5 / 3 :=
by
  sorry

end hyperbola_properties_l122_122775


namespace min_x_y_l122_122274

noncomputable def min_value (x y : ℝ) : ℝ := x + y

theorem min_x_y (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x + 16 * y = x * y) :
  min_value x y = 25 :=
sorry

end min_x_y_l122_122274


namespace distinct_non_zero_reals_square_rational_l122_122061

theorem distinct_non_zero_reals_square_rational
  {a : Fin 10 → ℝ}
  (distinct : ∀ i j, i ≠ j → a i ≠ a j)
  (non_zero : ∀ i, a i ≠ 0)
  (rational_condition : ∀ i j, ∃ (q : ℚ), a i + a j = q ∨ a i * a j = q) :
  ∀ i, ∃ (q : ℚ), (a i)^2 = q :=
by
  sorry

end distinct_non_zero_reals_square_rational_l122_122061


namespace find_third_coaster_speed_l122_122979

theorem find_third_coaster_speed
  (s1 s2 s4 s5 avg_speed n : ℕ)
  (hs1 : s1 = 50)
  (hs2 : s2 = 62)
  (hs4 : s4 = 70)
  (hs5 : s5 = 40)
  (havg_speed : avg_speed = 59)
  (hn : n = 5) : 
  ∃ s3 : ℕ, s3 = 73 :=
by
  sorry

end find_third_coaster_speed_l122_122979


namespace minimum_study_tools_l122_122718

theorem minimum_study_tools (n : Nat) : n^3 ≥ 366 → n ≥ 8 := by
  intros h
  sorry

end minimum_study_tools_l122_122718


namespace max_product_300_l122_122544

theorem max_product_300 (x : ℤ) (h : x + (300 - x) = 300) : 
  x * (300 - x) ≤ 22500 :=
by
  sorry

end max_product_300_l122_122544


namespace denomination_calculation_l122_122734

variables (total_money rs_50_count total_count rs_50_value remaining_count remaining_amount remaining_denomination_value : ℕ)

theorem denomination_calculation 
  (h1 : total_money = 10350)
  (h2 : rs_50_count = 97)
  (h3 : total_count = 108)
  (h4 : rs_50_value = 50)
  (h5 : remaining_count = total_count - rs_50_count)
  (h6 : remaining_amount = total_money - rs_50_count * rs_50_value)
  (h7 : remaining_denomination_value = remaining_amount / remaining_count) :
  remaining_denomination_value = 500 := 
sorry

end denomination_calculation_l122_122734


namespace relationship_among_a_b_c_l122_122007

noncomputable def a : ℝ := 0.99 ^ (1.01 : ℝ)
noncomputable def b : ℝ := 1.01 ^ (0.99 : ℝ)
noncomputable def c : ℝ := Real.log 0.99 / Real.log 1.01

theorem relationship_among_a_b_c : c < a ∧ a < b :=
by
  sorry

end relationship_among_a_b_c_l122_122007


namespace max_product_of_sum_300_l122_122471

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l122_122471


namespace area_of_square_with_diagonal_30_l122_122095

theorem area_of_square_with_diagonal_30 :
  ∀ (d : ℝ), d = 30 → (d * d / 2) = 450 := 
by
  intros d h
  rw [h]
  sorry

end area_of_square_with_diagonal_30_l122_122095


namespace intersection_M_N_l122_122950

open Set

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 3
noncomputable def g (x : ℝ) : ℝ := 3^x - 2

def M : Set ℝ := {x | f (g x) > 0}
def N : Set ℝ := {x | g x < 2}

theorem intersection_M_N : M ∩ N = {x : ℝ | x < 1} :=
by sorry

end intersection_M_N_l122_122950


namespace max_product_of_sum_300_l122_122477

theorem max_product_of_sum_300 (x y : ℤ) (h : x + y = 300) : 
  x * y ≤ 22500 := sorry

end max_product_of_sum_300_l122_122477


namespace greatest_product_of_sum_eq_300_l122_122387

def max_product (p : ℕ → ℕ → Prop) (a b : ℕ) : ℕ :=
if p a b then a * b else 0 

theorem greatest_product_of_sum_eq_300 : 
  ∃ x y : ℕ, (x + y = 300) ∧ (∀ a b : ℕ, a + b = 300 → a * b ≤ x * y) :=
begin
  use [150, 150],
  split, 
  { 
    -- Condition check
    exact rfl
  },
  {
    -- Proof of maximum product (we use a brief sketch here)
    intros a b h,
    have ha : a = 150 := sorry,
    have hb : b = 150 := sorry,
    rw [ha, hb]
  }
end

end greatest_product_of_sum_eq_300_l122_122387


namespace range_of_a_l122_122028

theorem range_of_a : (∀ x : ℝ, x^2 + (a-1)*x + 1 > 0) ↔ (-1 < a ∧ a < 3) := by
  sorry

end range_of_a_l122_122028


namespace simplify_expression_l122_122318

theorem simplify_expression (y : ℝ) :
  (2 * y^6 + 3 * y^5 + y^3 + 15) - (y^6 + 4 * y^5 - 2 * y^4 + 17) = 
  (y^6 - y^5 + 2 * y^4 + y^3 - 2) :=
by 
  sorry

end simplify_expression_l122_122318


namespace probability_sqrt_lt_nine_l122_122554

theorem probability_sqrt_lt_nine : 
  let two_digit_numbers := {n | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n | 10 ≤ n ∧ n < 81}
  let probability := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  probability = 71 / 90 :=
by
  let two_digit_numbers := {n : ℕ | 10 ≤ n ∧ n ≤ 99}
  let valid_numbers := {n : ℕ | 10 ≤ n ∧ n < 81}
  have h1 : two_digit_numbers.card = 90 := sorry
  have h2 : valid_numbers.card = 71 := sorry
  let probability : ℚ := (valid_numbers.card : ℚ) / (two_digit_numbers.card : ℚ)
  show probability = 71 / 90
  rw [h1, h2]
  simp
  norm_num

end probability_sqrt_lt_nine_l122_122554


namespace work_duration_l122_122293

variable (a b c : ℕ)
variable (daysTogether daysA daysB daysC : ℕ)

theorem work_duration (H1 : daysTogether = 4)
                      (H2 : daysA = 12)
                      (H3 : daysB = 18)
                      (H4: a = 1 / 12)
                      (H5: b = 1 / 18)
                      (H6: 1 / daysTogether = 1 / daysA + 1 / daysB + 1 / daysC) :
                      daysC = 9 :=
sorry

end work_duration_l122_122293


namespace total_height_of_three_buildings_l122_122593

theorem total_height_of_three_buildings :
  let h1 := 600
  let h2 := 2 * h1
  let h3 := 3 * (h1 + h2)
  h1 + h2 + h3 = 7200 :=
by
  sorry

end total_height_of_three_buildings_l122_122593


namespace algebraic_expression_value_l122_122904

theorem algebraic_expression_value
  (x : ℝ)
  (h : 2 * x^2 + 3 * x + 1 = 10) :
  4 * x^2 + 6 * x + 1 = 19 := 
by
  sorry

end algebraic_expression_value_l122_122904


namespace min_value_c_and_d_l122_122309

theorem min_value_c_and_d (c d : ℝ) (h1 : c > 0) (h2 : d > 0)
  (h3 : c^2 - 12 * d ≥ 0)
  (h4 : 9 * d^2 - 4 * c ≥ 0) :
  c + d ≥ 5.74 :=
sorry

end min_value_c_and_d_l122_122309


namespace problem1_problem2_l122_122638

-- Define the propositions p and q
def p (x a : ℝ) : Prop := x^2 + 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := (x^2 - 6 * x - 72 ≤ 0) ∧ (x^2 + x - 6 > 0)

-- Problem 1: Proving the range of x
theorem problem1 (x : ℝ) (h₁ : a = -1) (h₂ : ∀ (x : ℝ), p x a → q x) : 
  x ∈ {x : ℝ | -6 ≤ x ∧ x < -3} ∨ x ∈ {x : ℝ | 1 < x ∧ x ≤ 12} := sorry

-- Problem 2: Proving the range of a
theorem problem2 (a : ℝ) (h₃ : (∀ x, q x → p x a) ∧ ¬ (∀ x, ¬q x → ¬p x a)) : 
  -4 ≤ a ∧ a ≤ -2 := sorry

end problem1_problem2_l122_122638


namespace julia_shortfall_l122_122206

-- Definitions based on the problem conditions
def rock_and_roll_price : ℕ := 5
def pop_price : ℕ := 10
def dance_price : ℕ := 3
def country_price : ℕ := 7
def quantity : ℕ := 4
def julia_money : ℕ := 75

-- Proof problem: Prove that Julia is short $25
theorem julia_shortfall : (quantity * rock_and_roll_price + quantity * pop_price + quantity * dance_price + quantity * country_price) - julia_money = 25 := by
  sorry

end julia_shortfall_l122_122206


namespace greatest_product_l122_122531

-- Define the two integers
def two_integers (x y : ℤ) : Prop := x + y = 300

-- Define the product function
def product (x : ℤ) : ℤ := x * (300 - x)

-- State the greatest product problem
theorem greatest_product (x : ℤ) (h : two_integers x (300 - x)) : product x ≤ 22500 :=
sorry

end greatest_product_l122_122531


namespace geometric_common_ratio_l122_122823

noncomputable def geo_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem geometric_common_ratio (a₁ : ℝ) (q : ℝ) (n : ℕ) 
  (h : 2 * geo_sum a₁ q n = geo_sum a₁ q (n + 1) + geo_sum a₁ q (n + 2)) : q = -2 :=
by
  sorry

end geometric_common_ratio_l122_122823


namespace min_value_of_a_plus_2b_l122_122135

theorem min_value_of_a_plus_2b (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b - 3) :
  a + 2 * b = 4 * Real.sqrt 2 + 3 :=
sorry

end min_value_of_a_plus_2b_l122_122135
