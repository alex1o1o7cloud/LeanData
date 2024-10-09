import Mathlib

namespace triangle_abc_l1045_104572

/-!
# Problem Statement
In triangle ABC with side lengths a, b, and c opposite to vertices A, B, and C respectively, we are given that ∠A = 2 * ∠B. We need to prove that a² = b * (b + c).
-/

variables (A B C : Type) -- Define vertices of the triangle
variables (α β γ : ℝ) -- Define angles at vertices A, B, and C respectively.

-- Define sides of the triangle
variables (a b c x y : ℝ) -- Define sides opposite to the corresponding angles

-- Main statement to prove in Lean 4
theorem triangle_abc (h1 : α = 2 * β) (h2 : a = b * (2 * β)) :
  a^2 = b * (b + c) :=
sorry

end triangle_abc_l1045_104572


namespace find_original_number_l1045_104591

theorem find_original_number (x : ℚ) (h : 5 * ((3 * x + 6) / 2) = 100) : x = 34 / 3 := sorry

end find_original_number_l1045_104591


namespace transformed_parabolas_combined_l1045_104542

theorem transformed_parabolas_combined (a b c : ℝ) :
  let f (x : ℝ) := a * (x - 3) ^ 2 + b * (x - 3) + c
  let g (x : ℝ) := -a * (x + 4) ^ 2 - b * (x + 4) - c
  ∀ x, (f x + g x) = -14 * a * x - 19 * a - 7 * b :=
by
  -- This is a placeholder for the actual proof using the conditions
  sorry

end transformed_parabolas_combined_l1045_104542


namespace quartic_polynomial_root_l1045_104503

noncomputable def Q (x : ℝ) : ℝ := x^4 - 4*x^3 + 6*x^2 - 4*x - 2

theorem quartic_polynomial_root :
  Q (Real.sqrt (Real.sqrt 3) + 1) = 0 :=
by
  sorry

end quartic_polynomial_root_l1045_104503


namespace range_of_a_l1045_104539

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (a - 2) * x^2 + 2 * (a - 2) * x + 4 ≥ 0) ↔ (2 ≤ a ∧ a ≤ 6) := 
sorry

end range_of_a_l1045_104539


namespace bricks_in_wall_is_720_l1045_104584

/-- 
Two bricklayers have varying speeds: one could build a wall in 12 hours and 
the other in 15 hours if working alone. Their efficiency decreases by 12 bricks
per hour when they work together. The contractor placed them together on this 
project and the wall was completed in 6 hours.
Prove that the number of bricks in the wall is 720.
-/
def number_of_bricks_in_wall (y : ℕ) : Prop :=
  let rate1 := y / 12
  let rate2 := y / 15
  let combined_rate := rate1 + rate2 - 12
  6 * combined_rate = y

theorem bricks_in_wall_is_720 : ∃ y : ℕ, number_of_bricks_in_wall y ∧ y = 720 :=
  by sorry

end bricks_in_wall_is_720_l1045_104584


namespace minimum_value_2x_3y_l1045_104532

theorem minimum_value_2x_3y (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (hxy : x^2 * y * (4 * x + 3 * y) = 3) :
  2 * x + 3 * y ≥ 2 * Real.sqrt 3 := by
  sorry

end minimum_value_2x_3y_l1045_104532


namespace sufficient_but_not_necessary_l1045_104565

theorem sufficient_but_not_necessary (x : ℝ) (h : x > 4) :
  (x ^ 2 - 5 * x + 4 ≥ 0 ∧ ¬(∀ x, (x ^ 2 - 5 * x + 4 ≥ 0 → x > 4))) :=
by
  sorry

end sufficient_but_not_necessary_l1045_104565


namespace num_pupils_is_40_l1045_104527

-- given conditions
def incorrect_mark : ℕ := 83
def correct_mark : ℕ := 63
def mark_difference : ℕ := incorrect_mark - correct_mark
def avg_increase : ℚ := 1 / 2

-- the main problem statement to prove
theorem num_pupils_is_40 (n : ℕ) (h : (mark_difference : ℚ) / n = avg_increase) : n = 40 := 
sorry

end num_pupils_is_40_l1045_104527


namespace correct_time_fraction_l1045_104596

theorem correct_time_fraction : 
  (∀ hour : ℕ, hour < 24 → true) →
  (∀ minute : ℕ, minute < 60 → (minute ≠ 16)) →
  (fraction_of_correct_time = 59 / 60) :=
by
  intros h_hour h_minute
  sorry

end correct_time_fraction_l1045_104596


namespace vertex_of_parabola_l1045_104590

theorem vertex_of_parabola :
  (∃ x y : ℝ, y = (x - 6)^2 + 3 ↔ (x = 6 ∧ y = 3)) :=
sorry

end vertex_of_parabola_l1045_104590


namespace sum_of_squares_l1045_104583

theorem sum_of_squares (x y : ℝ) (h₁ : x + y = 40) (h₂ : x * y = 120) : x^2 + y^2 = 1360 :=
by
  sorry

end sum_of_squares_l1045_104583


namespace average_vegetables_per_week_l1045_104563

theorem average_vegetables_per_week (P Vp S W : ℕ) (h1 : P = 200) (h2 : Vp = 2) (h3 : S = 25) (h4 : W = 2) :
  (P / Vp) / S / W = 2 :=
by
  sorry

end average_vegetables_per_week_l1045_104563


namespace simplify_expression_l1045_104509

theorem simplify_expression :
  2^2 + 2^2 + 2^2 + 2^2 = 2^4 :=
sorry

end simplify_expression_l1045_104509


namespace quadratic_passes_through_neg3_n_l1045_104566

-- Definition of the quadratic function with given conditions
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Conditions provided in the problem
variables {a b c : ℝ}
axiom max_at_neg2 : ∀ x, quadratic a b c x ≤ 8
axiom value_at_neg2 : quadratic a b c (-2) = 8
axiom passes_through_1_4 : quadratic a b c 1 = 4

-- Statement to prove
theorem quadratic_passes_through_neg3_n : quadratic a b c (-3) = 68 / 9 :=
sorry

end quadratic_passes_through_neg3_n_l1045_104566


namespace line_through_point_equal_intercepts_l1045_104554

-- Definitions based on conditions
def passes_through (p : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  l p.1 p.2

def equal_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  ∃ a, a ≠ 0 ∧ (∀ x y, l x y ↔ x + y = a) ∨ (∀ x y, l x y ↔ y = 2 * x)

-- Theorem statement based on the problem
theorem line_through_point_equal_intercepts :
  ∃ l, passes_through (1, 2) l ∧ equal_intercepts l ∧
  (∀ x y, l x y ↔ 2 * x - y = 0) ∨ (∀ x y, l x y ↔ x + y - 3 = 0) :=
sorry

end line_through_point_equal_intercepts_l1045_104554


namespace find_certain_number_l1045_104559

theorem find_certain_number (n x : ℤ) (h1 : 9 - n / x = 7 + 8 / x) (h2 : x = 6) : n = 8 := by
  sorry

end find_certain_number_l1045_104559


namespace PQRS_product_l1045_104552

noncomputable def P : ℝ := (Real.sqrt 2023 + Real.sqrt 2024)
noncomputable def Q : ℝ := (-Real.sqrt 2023 - Real.sqrt 2024)
noncomputable def R : ℝ := (Real.sqrt 2023 - Real.sqrt 2024)
noncomputable def S : ℝ := (Real.sqrt 2024 - Real.sqrt 2023)

theorem PQRS_product : (P * Q * R * S) = 1 := 
by 
  sorry

end PQRS_product_l1045_104552


namespace perimeter_of_shaded_region_l1045_104592

noncomputable def circle_center : Type := sorry -- Define the object type for circle's center
noncomputable def radius_length : ℝ := 10 -- Define the radius length as 10
noncomputable def central_angle : ℝ := 270 -- Define the central angle corresponding to the arc RS

-- Function to calculate the perimeter of the shaded region
noncomputable def perimeter_shaded_region (radius : ℝ) (angle : ℝ) : ℝ :=
  2 * radius + (angle / 360) * 2 * Real.pi * radius

-- Theorem stating that the perimeter of the shaded region is 20 + 15π given the conditions
theorem perimeter_of_shaded_region : 
  perimeter_shaded_region radius_length central_angle = 20 + 15 * Real.pi :=
by
  -- skipping the actual proof
  sorry

end perimeter_of_shaded_region_l1045_104592


namespace power_of_powers_l1045_104564

theorem power_of_powers :
  (3^2)^4 = 6561 := 
by 
  sorry

end power_of_powers_l1045_104564


namespace janice_purchases_l1045_104518

theorem janice_purchases (a b c : ℕ) : 
  a + b + c = 50 ∧ 30 * a + 200 * b + 300 * c = 5000 → a = 10 :=
sorry

end janice_purchases_l1045_104518


namespace remaining_money_l1045_104505

-- Define the conditions
def num_pies : ℕ := 200
def price_per_pie : ℕ := 20
def fraction_for_ingredients : ℚ := 3 / 5

-- Define the total sales
def total_sales : ℕ := num_pies * price_per_pie

-- Define the cost for ingredients
def cost_for_ingredients : ℚ := fraction_for_ingredients * total_sales 

-- Prove the remaining money
theorem remaining_money : (total_sales : ℚ) - cost_for_ingredients = 1600 := 
by {
  -- This is where the proof would go
  sorry
}

end remaining_money_l1045_104505


namespace equation_I_consecutive_integers_equation_II_consecutive_even_integers_l1045_104512

theorem equation_I_consecutive_integers :
  ∃ (x y z : ℕ), x + y + z = 48 ∧ (x = y - 1) ∧ (z = y + 1) := sorry

theorem equation_II_consecutive_even_integers :
  ∃ (x y z w : ℕ), x + y + z + w = 52 ∧ (y = x + 2) ∧ (z = x + 4) ∧ (w = x + 6) := sorry

end equation_I_consecutive_integers_equation_II_consecutive_even_integers_l1045_104512


namespace circumference_of_circle_l1045_104525

/-- Given a circle with area 4 * π square units, prove that its circumference is 4 * π units. -/
theorem circumference_of_circle (r : ℝ) (h : π * r^2 = 4 * π) : 2 * π * r = 4 * π :=
sorry

end circumference_of_circle_l1045_104525


namespace find_x_l1045_104579

theorem find_x (x : ℕ) (h : x + 1 = 6) : x = 5 :=
sorry

end find_x_l1045_104579


namespace inequality_proof_l1045_104514

theorem inequality_proof (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  1 < (b / (a * b + b + 1) + c / (b * c + c + 1) + d / (c * d + d + 1) + a / (d * a + a + 1)) ∧
  (b / (a * b + b + 1) + c / (b * c + c + 1) + d / (c * d + d + 1) + a / (d * a + a + 1)) < 2 :=
sorry

end inequality_proof_l1045_104514


namespace subset_P1_P2_l1045_104573

def P1 (a : ℝ) : Set ℝ := {x | x^2 + a*x + 1 > 0}
def P2 (a : ℝ) : Set ℝ := {x | x^2 + a*x + 2 > 0}

theorem subset_P1_P2 (a : ℝ) : P1 a ⊆ P2 a :=
by intros x hx; sorry

end subset_P1_P2_l1045_104573


namespace second_number_multiple_of_seven_l1045_104588

theorem second_number_multiple_of_seven (x : ℕ) (h : gcd (gcd 105 x) 2436 = 7) : 7 ∣ x :=
sorry

end second_number_multiple_of_seven_l1045_104588


namespace economic_rationale_education_policy_l1045_104567

theorem economic_rationale_education_policy
  (countries : Type)
  (foreign_citizens : Type)
  (universities : Type)
  (free_or_nominal_fee : countries → Prop)
  (international_agreements : countries → Prop)
  (aging_population : countries → Prop)
  (economic_benefits : countries → Prop)
  (credit_concessions : countries → Prop)
  (reciprocity_education : countries → Prop)
  (educated_youth_contributions : countries → Prop)
  :
  (∀ c : countries, free_or_nominal_fee c ↔
    (international_agreements c ∧ (credit_concessions c ∨ reciprocity_education c)) ∨
    (aging_population c ∧ economic_benefits c ∧ educated_youth_contributions c)) := 
sorry

end economic_rationale_education_policy_l1045_104567


namespace cost_of_article_l1045_104544

theorem cost_of_article (C G : ℝ) (h1 : C + G = 348) (h2 : C + 1.05 * G = 350) : C = 308 :=
by
  sorry

end cost_of_article_l1045_104544


namespace change_calculation_l1045_104533

-- Define the initial amounts of Lee and his friend
def lee_amount : ℕ := 10
def friend_amount : ℕ := 8

-- Define the cost of items they ordered
def chicken_wings : ℕ := 6
def chicken_salad : ℕ := 4
def soda : ℕ := 1
def soda_count : ℕ := 2
def tax : ℕ := 3

-- Define the total money they initially had
def total_money : ℕ := lee_amount + friend_amount

-- Define the total cost of the food without tax
def food_cost : ℕ := chicken_wings + chicken_salad + (soda * soda_count)

-- Define the total cost including tax
def total_cost : ℕ := food_cost + tax

-- Define the change they should receive
def change : ℕ := total_money - total_cost

theorem change_calculation : change = 3 := by
  -- Note: Proof here is omitted
  sorry

end change_calculation_l1045_104533


namespace find_number_of_students_l1045_104519

-- Conditions
def john_marks_wrongly_recorded : ℕ := 82
def john_actual_marks : ℕ := 62
def sarah_marks_wrongly_recorded : ℕ := 76
def sarah_actual_marks : ℕ := 66
def emily_marks_wrongly_recorded : ℕ := 92
def emily_actual_marks : ℕ := 78
def increase_in_average : ℚ := 1 / 2

-- Proof problem
theorem find_number_of_students (n : ℕ) 
    (h1 : john_marks_wrongly_recorded = 82)
    (h2 : john_actual_marks = 62)
    (h3 : sarah_marks_wrongly_recorded = 76)
    (h4 : sarah_actual_marks = 66)
    (h5 : emily_marks_wrongly_recorded = 92)
    (h6 : emily_actual_marks = 78) 
    (h7: increase_in_average = 1 / 2):
    n = 88 :=
by 
  sorry

end find_number_of_students_l1045_104519


namespace geometric_sequence_sum_l1045_104589

-- Define the sequence and state the conditions
def geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n, a (n + 1) / a n = a 1 / a 0

-- The mathematical problem rewritten in Lean 4 statement
theorem geometric_sequence_sum (a : ℕ → ℝ) (s : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : s 2 = 7)
  (h3 : s 6 = 91)
  : ∃ s_4 : ℝ, s_4 = 28 :=
by
  sorry

end geometric_sequence_sum_l1045_104589


namespace width_of_room_l1045_104547

theorem width_of_room (C r l : ℝ) (hC : C = 18700) (hr : r = 850) (hl : l = 5.5) : 
  ∃ w, C / r / l = w ∧ w = 4 :=
by
  use 4
  sorry

end width_of_room_l1045_104547


namespace find_first_term_l1045_104560

theorem find_first_term (a : ℚ) (n : ℕ) (T : ℕ → ℚ)
  (hT : ∀ n, T n = n * (2 * a + 5 * (n - 1)) / 2)
  (h_const : ∃ c : ℚ, ∀ n > 0, T (4 * n) / T n = c) :
  a = 5 / 2 := 
sorry

end find_first_term_l1045_104560


namespace expand_expression_l1045_104557

variable {R : Type} [CommRing R]
variables (x y : R)

theorem expand_expression :
  5 * (3 * x^3 - 4 * x * y + x^2 - y^2) = 15 * x^3 - 20 * x * y + 5 * x^2 - 5 * y^2 :=
by
  sorry

end expand_expression_l1045_104557


namespace quadratic_has_two_real_roots_l1045_104556

-- Define the condition that the discriminant must be non-negative
def discriminant_nonneg (a b c : ℝ) : Prop := b * b - 4 * a * c ≥ 0

-- Define our specific quadratic equation conditions: x^2 - 2x + m = 0
theorem quadratic_has_two_real_roots (m : ℝ) :
  discriminant_nonneg 1 (-2) m → m ≤ 1 :=
by
  sorry

end quadratic_has_two_real_roots_l1045_104556


namespace parabola_focus_l1045_104586

open Real

theorem parabola_focus (a : ℝ) (h k : ℝ) (x y : ℝ) (f : ℝ) :
  (a = -1/4) → (h = 0) → (k = 0) → 
  (f = (1 / (4 * a))) →
  (y = a * (x - h) ^ 2 + k) → 
  (y = -1 / 4 * x ^ 2) → f = -1 := by
  intros h_a h_h h_k h_f parabola_eq _
  rw [h_a, h_h, h_k] at *
  sorry

end parabola_focus_l1045_104586


namespace directrix_of_parabola_l1045_104571

def parabola_eq (x : ℝ) : ℝ := -4 * x^2 + 4

theorem directrix_of_parabola : 
  ∃ y : ℝ, y = 65 / 16 :=
by
  sorry

end directrix_of_parabola_l1045_104571


namespace domain_of_v_l1045_104502

noncomputable def v (x : ℝ) : ℝ := 1 / (x - 1)^(1 / 3)

theorem domain_of_v :
  {x : ℝ | ∃ y : ℝ, y ≠ 0 ∧ y = (v x)} = {x | x ≠ 1} := by
  sorry

end domain_of_v_l1045_104502


namespace unused_combinations_eq_40_l1045_104568

-- Defining the basic parameters
def num_resources : ℕ := 6
def total_combinations : ℕ := 2 ^ num_resources
def used_combinations : ℕ := 23

-- Calculating the number of unused combinations
theorem unused_combinations_eq_40 : total_combinations - 1 - used_combinations = 40 := by
  sorry

end unused_combinations_eq_40_l1045_104568


namespace initial_students_l1045_104534

theorem initial_students {f : ℕ → ℕ} {g : ℕ → ℕ} (h_f : ∀ t, t ≥ 15 * 60 + 3 → (f t = 4 * ((t - (15 * 60 + 3)) / 3 + 1))) 
    (h_g : ∀ t, t ≥ 15 * 60 + 10 → (g t = 8 * ((t - (15 * 60 + 10)) / 10 + 1))) 
    (students_at_1544 : f 15 * 60 + 44 - g 15 * 60 + 44 + initial = 27) : 
    initial = 3 := 
sorry

end initial_students_l1045_104534


namespace committee_with_one_boy_one_girl_prob_l1045_104578

def total_members := 30
def boys := 12
def girls := 18
def committee_size := 6

theorem committee_with_one_boy_one_girl_prob :
  let total_ways := Nat.choose total_members committee_size
  let all_boys_ways := Nat.choose boys committee_size
  let all_girls_ways := Nat.choose girls committee_size
  let prob_all_boys_or_all_girls := (all_boys_ways + all_girls_ways) / total_ways
  let desired_prob := 1 - prob_all_boys_or_all_girls
  desired_prob = 19145 / 19793 :=
by
  sorry

end committee_with_one_boy_one_girl_prob_l1045_104578


namespace given_inequality_l1045_104537

theorem given_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h: 1 + a + b + c = 2 * a * b * c) :
  ab / (1 + a + b) + bc / (1 + b + c) + ca / (1 + c + a) ≥ 3 / 2 :=
sorry

end given_inequality_l1045_104537


namespace find_triplets_find_triplets_non_negative_l1045_104558

theorem find_triplets :
  ∀ (x y z : ℕ), (x > 0) ∧ (y > 0) ∧ (z > 0) →
    x^2 + y^2 + 1 = 2^z →
    (x = 1 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 1) :=
by
  sorry

theorem find_triplets_non_negative :
  ∀ (x y z : ℕ), x^2 + y^2 + 1 = 2^z →
    (x = 1 ∧ y = 0 ∧ z = 1) ∨ (x = 0 ∧ y = 1 ∧ z = 1) ∨ (x = 0 ∧ y = 0 ∧ z = 0) :=
by
  sorry

end find_triplets_find_triplets_non_negative_l1045_104558


namespace seventy_times_reciprocal_l1045_104515

theorem seventy_times_reciprocal (x : ℚ) (hx : 7 * x = 3) : 70 * (1 / x) = 490 / 3 :=
by 
  sorry

end seventy_times_reciprocal_l1045_104515


namespace birds_in_store_l1045_104555

/-- 
A pet store had a total of 180 animals, consisting of birds, dogs, and cats. 
Among the birds, 64 talked, and 13 didn't. If there were 40 dogs in the store 
and the number of birds that talked was four times the number of cats, 
prove that there were 124 birds in total.
-/
theorem birds_in_store (total_animals : ℕ) (talking_birds : ℕ) (non_talking_birds : ℕ) 
  (dogs : ℕ) (cats : ℕ) 
  (h1 : total_animals = 180)
  (h2 : talking_birds = 64)
  (h3 : non_talking_birds = 13)
  (h4 : dogs = 40)
  (h5 : talking_birds = 4 * cats) : 
  talking_birds + non_talking_birds + dogs + cats = 180 ∧ 
  talking_birds + non_talking_birds = 124 :=
by
  -- We are skipping the proof itself and focusing on the theorem statement
  sorry

end birds_in_store_l1045_104555


namespace number_of_cirrus_clouds_l1045_104521

def C_cb := 3
def C_cu := 12 * C_cb
def C_ci := 4 * C_cu

theorem number_of_cirrus_clouds : C_ci = 144 :=
by
  sorry

end number_of_cirrus_clouds_l1045_104521


namespace soccer_and_volleyball_unit_prices_max_soccer_balls_l1045_104511

-- Define the conditions and the problem
def unit_price_soccer_ball (x : ℕ) (y : ℕ) : Prop :=
  x = y + 15 ∧ 480 / x = 390 / y

def school_purchase (m : ℕ) : Prop :=
  m ≤ 70 ∧ 80 * m + 65 * (100 - m) ≤ 7550

-- Proof statement for the unit prices of soccer balls and volleyballs
theorem soccer_and_volleyball_unit_prices (x y : ℕ) (h : unit_price_soccer_ball x y) :
  x = 80 ∧ y = 65 :=
by
  sorry

-- Proof statement for the maximum number of soccer balls the school can purchase
theorem max_soccer_balls (m : ℕ) :
  school_purchase m :=
by
  sorry

end soccer_and_volleyball_unit_prices_max_soccer_balls_l1045_104511


namespace shop_profit_correct_l1045_104530

def profit_per_tire_repair : ℕ := 20 - 5
def total_tire_repairs : ℕ := 300
def profit_per_complex_repair : ℕ := 300 - 50
def total_complex_repairs : ℕ := 2
def retail_profit : ℕ := 2000
def fixed_expenses : ℕ := 4000

theorem shop_profit_correct :
  profit_per_tire_repair * total_tire_repairs +
  profit_per_complex_repair * total_complex_repairs +
  retail_profit - fixed_expenses = 3000 :=
by
  sorry

end shop_profit_correct_l1045_104530


namespace sum_of_squares_l1045_104538

theorem sum_of_squares (x y z : ℕ) (hx : x = 1) (hy : y = 2) (hz : z = 3) (h_sum : x * 1 + y * 2 + z * 3 = 12) : x^2 + y^2 + z^2 = 56 :=
by
  sorry

end sum_of_squares_l1045_104538


namespace lily_catches_up_mary_in_60_minutes_l1045_104594

theorem lily_catches_up_mary_in_60_minutes
  (mary_speed : ℝ) (lily_speed : ℝ) (initial_distance : ℝ)
  (h_mary_speed : mary_speed = 4)
  (h_lily_speed : lily_speed = 6)
  (h_initial_distance : initial_distance = 2) :
  ∃ t : ℝ, t = 60 := by
  sorry

end lily_catches_up_mary_in_60_minutes_l1045_104594


namespace pi_div_two_minus_alpha_in_third_quadrant_l1045_104553

theorem pi_div_two_minus_alpha_in_third_quadrant (α : ℝ) (k : ℤ) (h : ∃ k : ℤ, (π + 2 * k * π < α) ∧ (α < 3 * π / 2 + 2 * k * π)) : 
  ∃ k : ℤ, (π + 2 * k * π < (π / 2 - α)) ∧ ((π / 2 - α) < 3 * π / 2 + 2 * k * π) :=
sorry

end pi_div_two_minus_alpha_in_third_quadrant_l1045_104553


namespace marys_next_birthday_l1045_104593

theorem marys_next_birthday (d s m : ℝ) (h1 : s = 0.7 * d) (h2 : m = 1.3 * s) (h3 : m + s + d = 25.2) : m + 1 = 9 :=
by
  sorry

end marys_next_birthday_l1045_104593


namespace total_messages_l1045_104536

theorem total_messages (l1 l2 l3 a1 a2 a3 : ℕ)
  (h1 : l1 = 120)
  (h2 : a1 = l1 - 20)
  (h3 : l2 = l1 / 3)
  (h4 : a2 = 2 * a1)
  (h5 : l3 = l1)
  (h6 : a3 = a1) :
  l1 + l2 + l3 + a1 + a2 + a3 = 680 :=
by
  -- Proof steps would go here. Adding 'sorry' to skip proof.
  sorry

end total_messages_l1045_104536


namespace employee_Y_base_pay_l1045_104546

theorem employee_Y_base_pay (P : ℝ) (h1 : 1.2 * P + P * 1.1 + P * 1.08 + P = P * 4.38)
                            (h2 : 2 * 1.5 * 1.2 * P = 3.6 * P)
                            (h3 : P * 4.38 + 100 + 3.6 * P = 1800) :
  P = 213.03 :=
by
  sorry

end employee_Y_base_pay_l1045_104546


namespace problem1_problem2_l1045_104524

noncomputable def f (a x : ℝ) := a - (2 / x)

theorem problem1 (a : ℝ) :
  (∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → (f a x1 < f a x2)) :=
sorry

theorem problem2 (a : ℝ) :
  (∀ x : ℝ, 1 < x → (f a x < 2 * x)) → a ≤ 3 :=
sorry

end problem1_problem2_l1045_104524


namespace apples_total_l1045_104506

def benny_apples : ℕ := 2
def dan_apples : ℕ := 9
def total_apples : ℕ := benny_apples + dan_apples

theorem apples_total : total_apples = 11 :=
by
    sorry

end apples_total_l1045_104506


namespace find_S6_l1045_104570

variable {a : ℕ → ℝ} 
variable {S : ℕ → ℝ}

/-- sum_of_first_n_terms_of_geometric_sequence -/
def sum_of_first_n_terms_of_geometric_sequence (S : ℕ → ℝ) : Prop :=
  ∃ a1 r, ∀ n, S n = a1 * (1 - r^(n+1)) / (1 - r)

-- Given conditions
axiom geom_seq_positive_terms : ∀ n, a n > 0
axiom sum_S2 : S 2 = 3
axiom sum_S4 : S 4 = 15

theorem find_S6 : S 6 = 63 := by
  sorry

end find_S6_l1045_104570


namespace arithmetic_sequence_ninth_term_l1045_104543

-- Definitions and Conditions
variables {a : ℕ → ℝ} (S : ℕ → ℝ)

-- Problem Statement
theorem arithmetic_sequence_ninth_term
  (h1 : a 3 = 4)
  (h2 : S 11 = 110)
  (h3 : ∀ n, S n = (n * (a 1 + a n)) / 2) :
  a 9 = 16 :=
sorry

end arithmetic_sequence_ninth_term_l1045_104543


namespace license_plate_count_l1045_104595

def license_plate_combinations : Nat :=
  26 * Nat.choose 25 2 * Nat.choose 4 2 * 720

theorem license_plate_count :
  license_plate_combinations = 33696000 :=
by
  unfold license_plate_combinations
  sorry

end license_plate_count_l1045_104595


namespace algebraic_expression_decrease_l1045_104582

theorem algebraic_expression_decrease (x y : ℝ) :
  let original_expr := 2 * x^2 * y
  let new_expr := 2 * ((1 / 2) * x) ^ 2 * ((1 / 2) * y)
  let decrease := ((original_expr - new_expr) / original_expr) * 100
  decrease = 87.5 := by
  sorry

end algebraic_expression_decrease_l1045_104582


namespace intersection_A_complement_UB_l1045_104585

-- Definitions of the sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {2, 3, 5, 6}
def B : Set ℕ := {x ∈ U | x^2 - 5 * x ≥ 0}

-- Complement of B w.r.t. U
def complement_U_B : Set ℕ := {x ∈ U | ¬ (x ∈ B)}

-- The statement we want to prove
theorem intersection_A_complement_UB : A ∩ complement_U_B = {2, 3} := by
  sorry

end intersection_A_complement_UB_l1045_104585


namespace possible_values_for_D_l1045_104540

def distinct_digits (A B C D E : ℕ) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ C ≠ D ∧ C ≠ E ∧ D ≠ E

def digits_range (A B C D E : ℕ) : Prop :=
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧ 0 ≤ E ∧ E ≤ 9

def addition_equation (A B C D E : ℕ) : Prop :=
  A * 10000 + B * 1000 + C * 100 + D * 10 + B +
  B * 10000 + C * 1000 + A * 100 + D * 10 + E = 
  E * 10000 + D * 1000 + D * 100 + E * 10 + E

theorem possible_values_for_D : 
  ∀ (A B C D E : ℕ),
  distinct_digits A B C D E →
  digits_range A B C D E →
  addition_equation A B C D E →
  ∃ (S : Finset ℕ), (∀ d ∈ S, 0 ≤ d ∧ d ≤ 9) ∧ (S.card = 2) :=
by
  -- Proof omitted
  sorry

end possible_values_for_D_l1045_104540


namespace house_cats_initial_l1045_104507

def initial_house_cats (S A T H : ℝ) : Prop :=
  S + H + A = T

theorem house_cats_initial (S A T H : ℝ) (h1 : S = 13.0) (h2 : A = 10.0) (h3 : T = 28) :
  initial_house_cats S A T H ↔ H = 5 := by
sorry

end house_cats_initial_l1045_104507


namespace min_floodgates_to_reduce_level_l1045_104548

-- Definitions for the conditions given in the problem
def num_floodgates : ℕ := 10
def a (v : ℝ) := 30 * v
def w (v : ℝ) := 2 * v

def time_one_gate : ℝ := 30
def time_two_gates : ℝ := 10
def time_target : ℝ := 3

-- Prove that the minimum number of floodgates \(n\) that must be opened to achieve the goal
theorem min_floodgates_to_reduce_level (v : ℝ) (n : ℕ) :
  (a v + time_target * v) ≤ (n * time_target * w v) → n ≥ 6 :=
by
  sorry

end min_floodgates_to_reduce_level_l1045_104548


namespace alex_age_div_M_l1045_104513

variable {A M : ℕ}

-- Definitions provided by the conditions
def alex_age_current : ℕ := A
def sum_children_age : ℕ := A
def alex_age_M_years_ago (A M : ℕ) : ℕ := A - M
def children_age_M_years_ago (A M : ℕ) : ℕ := A - 4 * M

-- Given condition as a hypothesis
def condition (A M : ℕ) := alex_age_M_years_ago A M = 3 * children_age_M_years_ago A M

-- The theorem to prove
theorem alex_age_div_M (A M : ℕ) (h : condition A M) : A / M = 11 / 2 := 
by
  -- This is a placeholder for the actual proof.
  sorry

end alex_age_div_M_l1045_104513


namespace find_b_l1045_104551

theorem find_b (a b : ℝ) (h₁ : ∀ x y, y = 0.75 * x + 1 → (4, b) = (x, y))
                (h₂ : k = 0.75) : b = 4 :=
by sorry

end find_b_l1045_104551


namespace num_customers_after_family_l1045_104517

-- Definitions
def soft_taco_price : ℕ := 2
def hard_taco_price : ℕ := 5
def family_hard_tacos : ℕ := 4
def family_soft_tacos : ℕ := 3
def total_income : ℕ := 66

-- Intermediate values which can be derived
def family_cost : ℕ := (family_hard_tacos * hard_taco_price) + (family_soft_tacos * soft_taco_price)
def remaining_income : ℕ := total_income - family_cost

-- Proposition: Number of customers after the family
def customers_after_family : ℕ := remaining_income / (2 * soft_taco_price)

-- Theorem to prove the number of customers is 10
theorem num_customers_after_family : customers_after_family = 10 := by
  sorry

end num_customers_after_family_l1045_104517


namespace expression_divisible_by_19_l1045_104597

theorem expression_divisible_by_19 (n : ℕ) (h : n > 0) : 
  19 ∣ (5^(2*n - 1) + 3^(n - 2) * 2^(n - 1)) := 
by 
  sorry

end expression_divisible_by_19_l1045_104597


namespace range_of_a_l1045_104531

open Real

noncomputable def proposition_p (a : ℝ) : Prop := ∀ x : ℝ, x^2 + a * x + a > 0

noncomputable def proposition_q (a : ℝ) : Prop :=
  let Δ := 1 - 4 * a
  Δ ≥ 0

theorem range_of_a (a : ℝ) :
  ((proposition_p a ∨ proposition_q a) ∧ ¬(proposition_p a ∧ proposition_q a))
  ↔ (a ≤ 0 ∨ (1/4 : ℝ) < a ∧ a < 4) :=
by
  sorry

end range_of_a_l1045_104531


namespace cost_of_pencils_and_pens_l1045_104574

theorem cost_of_pencils_and_pens (a b : ℝ) (h1 : 4 * a + b = 2.60) (h2 : a + 3 * b = 2.15) : 3 * a + 2 * b = 2.63 :=
sorry

end cost_of_pencils_and_pens_l1045_104574


namespace negation_proposition_p_l1045_104516

theorem negation_proposition_p (x y : ℝ) : (¬ ((x - 1) ^ 2 + (y - 2) ^ 2 = 0) → (x ≠ 1 ∨ y ≠ 2)) :=
by
  sorry

end negation_proposition_p_l1045_104516


namespace four_digit_number_exists_l1045_104504

theorem four_digit_number_exists :
  ∃ n : ℕ, n >= 1000 ∧ n < 10000 ∧ 4 * n = (n % 10) * 1000 + ((n / 10) % 10) * 100 + ((n / 100) % 10) * 10 + (n / 1000) :=
sorry

end four_digit_number_exists_l1045_104504


namespace new_people_moved_in_l1045_104523

theorem new_people_moved_in (N : ℕ) : (∃ N, 1/16 * (780 - 400 + N : ℝ) = 60) → N = 580 := by
  intros hN
  sorry

end new_people_moved_in_l1045_104523


namespace shaded_cubes_count_l1045_104577

theorem shaded_cubes_count :
  let faces := 6
  let shaded_on_one_face := 5
  let corner_cubes := 8
  let center_cubes := 2 * 1 -- center cubes shared among opposite faces
  let total_shaded_cubes := corner_cubes + center_cubes
  faces = 6 → shaded_on_one_face = 5 → corner_cubes = 8 → center_cubes = 2 →
  total_shaded_cubes = 10 := 
by
  intros _ _ _ _ 
  sorry

end shaded_cubes_count_l1045_104577


namespace m_value_quadratic_l1045_104581

theorem m_value_quadratic (m : ℝ)
  (h1 : |m - 2| = 2)
  (h2 : m - 4 ≠ 0) :
  m = 0 :=
sorry

end m_value_quadratic_l1045_104581


namespace length_AF_is_25_l1045_104526

open Classical

noncomputable def length_AF : ℕ :=
  let AB := 5
  let AC := 11
  let DE := 8
  let EF := 4
  let BC := AC - AB
  let CD := BC / 3
  let AF := AB + BC + CD + DE + EF
  AF

theorem length_AF_is_25 :
  length_AF = 25 := by
  sorry

end length_AF_is_25_l1045_104526


namespace packets_of_chips_l1045_104580

variable (P R M : ℕ)

theorem packets_of_chips (h1: P > 0) (h2: R > 0) (h3: M > 0) :
  ((10 * M * P) / R) = (10 * M * P) / R :=
sorry

end packets_of_chips_l1045_104580


namespace professional_doctors_percentage_l1045_104535

-- Defining the context and conditions:

variable (total_percent : ℝ) (leaders_percent : ℝ) (nurses_percent : ℝ) (doctors_percent : ℝ)

-- Specifying the conditions:
def total_percentage_sum : Prop :=
  total_percent = 100

def leaders_percentage : Prop :=
  leaders_percent = 4

def nurses_percentage : Prop :=
  nurses_percent = 56

-- Stating the actual theorem to be proved:
theorem professional_doctors_percentage
  (h1 : total_percentage_sum total_percent)
  (h2 : leaders_percentage leaders_percent)
  (h3 : nurses_percentage nurses_percent) :
  doctors_percent = 100 - (leaders_percent + nurses_percent) := by
  sorry -- Proof placeholder

end professional_doctors_percentage_l1045_104535


namespace center_of_gravity_shift_center_of_gravity_shift_result_l1045_104520

variable (l s : ℝ) (s_val : s = 60)
#check (s_val : s = 60)

theorem center_of_gravity_shift : abs ((l / 2) - ((l - s) / 2)) = s / 2 := 
by sorry

theorem center_of_gravity_shift_result : (s / 2 = 30) :=
by sorry

end center_of_gravity_shift_center_of_gravity_shift_result_l1045_104520


namespace total_number_of_fish_l1045_104528

noncomputable def number_of_stingrays : ℕ := 28

noncomputable def number_of_sharks : ℕ := 2 * number_of_stingrays

theorem total_number_of_fish : number_of_sharks + number_of_stingrays = 84 :=
by
  sorry

end total_number_of_fish_l1045_104528


namespace solve_for_q_l1045_104599

theorem solve_for_q 
  (n m q : ℕ)
  (h1 : 5 / 6 = n / 60)
  (h2 : 5 / 6 = (m + n) / 90)
  (h3 : 5 / 6 = (q - m) / 150) : 
  q = 150 :=
sorry

end solve_for_q_l1045_104599


namespace marked_box_in_second_row_l1045_104501

theorem marked_box_in_second_row:
  ∀ a b c d e f g h : ℕ, 
  (e = a + b) → 
  (f = b + c) →
  (g = c + d) →
  (h = a + 2 * b + c) →
  ((a = 5) ∧ (d = 6)) →
  ((a = 3) ∨ (b = 3) ∨ (c = 3) ∨ (d = 3)) →
  (f = 3) :=
by
  sorry

end marked_box_in_second_row_l1045_104501


namespace cyclic_sum_fraction_ge_one_l1045_104575

theorem cyclic_sum_fraction_ge_one (a b c : ℝ) 
  (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0)
  (hineq : (a/(b+c+1) + b/(c+a+1) + c/(a+b+1)) ≤ 1) :
  (1/(b+c+1) + 1/(c+a+1) + 1/(a+b+1)) ≥ 1 :=
by sorry

end cyclic_sum_fraction_ge_one_l1045_104575


namespace binary_operation_correct_l1045_104500

theorem binary_operation_correct :
  let b1 := 0b11011
  let b2 := 0b1011
  let b3 := 0b11100
  let b4 := 0b10101
  let b5 := 0b1001
  b1 + b2 - b3 + b4 - b5 = 0b11110 := by
  sorry

end binary_operation_correct_l1045_104500


namespace solution_set_of_inequality_system_l1045_104510

theorem solution_set_of_inequality_system (x : ℝ) : (x - 1 < 0) ∧ (x + 1 > 0) ↔ (-1 < x ∧ x < 1) :=
by
  -- proof to be filled in
  sorry

end solution_set_of_inequality_system_l1045_104510


namespace calc1_calc2_calc3_calc4_calc5_calc6_l1045_104562

theorem calc1 : 320 + 16 * 27 = 752 :=
by
  -- Proof goes here
  sorry

theorem calc2 : 1500 - 125 * 8 = 500 :=
by
  -- Proof goes here
  sorry

theorem calc3 : 22 * 22 - 84 = 400 :=
by
  -- Proof goes here
  sorry

theorem calc4 : 25 * 8 * 9 = 1800 :=
by
  -- Proof goes here
  sorry

theorem calc5 : (25 + 38) * 15 = 945 :=
by
  -- Proof goes here
  sorry

theorem calc6 : (62 + 12) * 38 = 2812 :=
by
  -- Proof goes here
  sorry

end calc1_calc2_calc3_calc4_calc5_calc6_l1045_104562


namespace peanut_count_l1045_104598

-- Definitions
def initial_peanuts : Nat := 10
def added_peanuts : Nat := 8

-- Theorem to prove
theorem peanut_count : (initial_peanuts + added_peanuts) = 18 := 
by
  -- Proof placeholder
  sorry

end peanut_count_l1045_104598


namespace analogical_reasoning_l1045_104550

theorem analogical_reasoning {a b c : ℝ} (h1 : c ≠ 0) : 
  (a + b) * c = a * c + b * c → (a + b) / c = a / c + b / c := 
by 
  sorry

end analogical_reasoning_l1045_104550


namespace cost_of_article_l1045_104561

variable (C : ℝ) 
variable (G : ℝ)
variable (H1 : G = 380 - C)
variable (H2 : 1.05 * G = 420 - C)

theorem cost_of_article : C = 420 :=
by
  sorry

end cost_of_article_l1045_104561


namespace remainder_of_sum_of_powers_div_2_l1045_104587

theorem remainder_of_sum_of_powers_div_2 : 
  (1^1 + 2^2 + 3^3 + 4^4 + 5^5 + 6^6 + 7^7 + 8^8 + 9^9) % 2 = 1 :=
by 
  sorry

end remainder_of_sum_of_powers_div_2_l1045_104587


namespace coefficient_of_a3b2_in_expansions_l1045_104529

theorem coefficient_of_a3b2_in_expansions 
  (a b c : ℝ) :
  (1 : ℝ) * (a + b)^5 * (c + c⁻¹)^8 = 700 :=
by 
  sorry

end coefficient_of_a3b2_in_expansions_l1045_104529


namespace collete_age_ratio_l1045_104569

theorem collete_age_ratio (Ro R C : ℕ) (h1 : R = 2 * Ro) (h2 : Ro = 8) (h3 : R - C = 12) :
  C / Ro = 1 / 2 := by
sorry

end collete_age_ratio_l1045_104569


namespace dorothy_profit_l1045_104576

-- Define the conditions
def expense := 53
def number_of_doughnuts := 25
def price_per_doughnut := 3

-- Define revenue and profit calculations
def revenue := number_of_doughnuts * price_per_doughnut
def profit := revenue - expense

-- Prove the profit calculation
theorem dorothy_profit : profit = 22 := by
  sorry

end dorothy_profit_l1045_104576


namespace symmetry_origin_l1045_104541

def f (x : ℝ) : ℝ := x^3 + x

theorem symmetry_origin : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

end symmetry_origin_l1045_104541


namespace rachel_class_choices_l1045_104549

theorem rachel_class_choices : (Nat.choose 8 3) = 56 :=
by
  sorry

end rachel_class_choices_l1045_104549


namespace monotonically_decreasing_interval_l1045_104522

-- Given conditions
def f (x : ℝ) : ℝ := x^2 * (x - 3)

-- The proof problem statement
theorem monotonically_decreasing_interval :
  ∃ a b : ℝ, (0 ≤ a) ∧ (b ≤ 2) ∧ (∀ x : ℝ, a ≤ x ∧ x ≤ b → (deriv f x ≤ 0)) :=
sorry

end monotonically_decreasing_interval_l1045_104522


namespace find_an_l1045_104545

variable (a : ℕ → ℤ) (S : ℕ → ℤ)
variable (a₁ d : ℤ)

-- Conditions
def S4 : Prop := S 4 = 0
def a5 : Prop := a 5 = 5
def Sn (n : ℕ) : Prop := S n = n * (2 * a₁ + (n - 1) * d) / 2
def an (n : ℕ) : Prop := a n = a₁ + (n - 1) * d

-- Theorem statement
theorem find_an (S4_hyp : S 4 = 0) (a5_hyp : a 5 = 5) (Sn_hyp : ∀ n, S n = n * (2 * a₁ + (n - 1) * d) / 2) (an_hyp : ∀ n, a n = a₁ + (n - 1) * d) :
  ∀ n, a n = 2 * n - 5 :=
by 
  intros n

  -- Proof is omitted, added here for logical conclusion completeness
  sorry

end find_an_l1045_104545


namespace number_of_preferred_groups_l1045_104508

def preferred_group_sum_multiple_5 (n : Nat) : Nat := 
  (2^n) * ((2^(4*n) - 1) / 5 + 1) - 1

theorem number_of_preferred_groups :
  preferred_group_sum_multiple_5 400 = 2^400 * (2^1600 - 1) / 5 + 1 - 1 :=
sorry

end number_of_preferred_groups_l1045_104508
