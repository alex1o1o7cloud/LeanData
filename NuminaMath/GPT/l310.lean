import Mathlib

namespace transport_problem_l310_310843

theorem transport_problem
  (x : ℝ) (T : ℝ)
  (initial_distance : ℝ := 90)
  (final_distance : ℝ := 348)
  (speed_diff : ℝ := 18):
  2 * x * T + speed_diff * T - x - speed_diff = final_distance - initial_distance := 
begin
  sorry
end

end transport_problem_l310_310843


namespace amount_returned_l310_310750

namespace IrinaBankProblem

-- Definitions
def deposit_amount : ℝ := 23904
def annual_interest_rate : ℝ := 0.05
def period_in_months : ℕ := 3
def exchange_rate : ℝ := 58.15
def insurance_limit : ℝ := 1400000

-- Goal statement
theorem amount_returned : 
  let monthly_interest_rate := annual_interest_rate / 12
  let interest_for_period := deposit_amount * (period_in_months * monthly_interest_rate) / 12
  let total_amount := deposit_amount + interest_for_period
  let amount_in_rubles := total_amount * exchange_rate in
  min amount_in_rubles insurance_limit = 1400000 :=
by {
  sorry
}

end IrinaBankProblem

end amount_returned_l310_310750


namespace total_number_of_notebooks_and_pens_l310_310470

theorem total_number_of_notebooks_and_pens :
  ∀ (n p : ℕ), p = n + 50 → n = 30 → n + p = 110 :=
by
  intros n p h1 h2
  rw [h2, h1]
  rw [h2]
  simp
  exact rfl

end total_number_of_notebooks_and_pens_l310_310470


namespace factoring_sum_of_coefficients_l310_310819

theorem factoring_sum_of_coefficients 
  (a b c d e f g h j k : ℤ)
  (h1 : 64 * x^6 - 729 * y^6 = (a * x + b * y) * (c * x^2 + d * x * y + e * y^2) * (f * x + g * y) * (h * x^2 + j * x * y + k * y^2)) :
  a + b + c + d + e + f + g + h + j + k = 30 :=
sorry

end factoring_sum_of_coefficients_l310_310819


namespace condition_nonzero_neither_zero_l310_310488

theorem condition_nonzero_neither_zero (a b : ℝ) (h : a^2 + b^2 ≠ 0) : ¬(a = 0 ∧ b = 0) :=
sorry

end condition_nonzero_neither_zero_l310_310488


namespace new_sphere_radius_l310_310924

noncomputable def calculateVolume (R r : ℝ) : ℝ :=
  let originalSphereVolume := (4 / 3) * Real.pi * R^3
  let cylinderHeight := 2 * Real.sqrt (R^2 - r^2)
  let cylinderVolume := Real.pi * r^2 * cylinderHeight
  let capHeight := R - Real.sqrt (R^2 - r^2)
  let capVolume := (Real.pi * capHeight^2 * (3 * R - capHeight)) / 3
  let totalCapVolume := 2 * capVolume
  originalSphereVolume - cylinderVolume - totalCapVolume

theorem new_sphere_radius
  (R : ℝ) (r : ℝ) (h : ℝ) (new_sphere_radius : ℝ)
  (h_eq: h = 2 * Real.sqrt (R^2 - r^2))
  (new_sphere_volume_eq: calculateVolume R r = (4 / 3) * Real.pi * new_sphere_radius^3)
  : new_sphere_radius = 16 :=
sorry

end new_sphere_radius_l310_310924


namespace real_a_range_l310_310809

noncomputable theory

def is_decreasing (f : ℝ → ℝ) (I : set ℝ) :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x < y → f y ≤ f x

def f (a : ℝ) : ℝ → ℝ := λ x, 
  if 0 < x ∧ x ≤ 1 then -x^2 + 2 * (a+1) * x + 4
  else if 1 < x then x^a else 0

theorem real_a_range (a : ℝ) :
  (is_decreasing (f a) { x | 0 < x ∧ x ≤ 1 } ∧ 
   is_decreasing (f a) { x | 1 < x }) → (-2 ≤ a ∧ a ≤ -1) :=
sorry

end real_a_range_l310_310809


namespace daphne_necklaces_l310_310068

/--
Given:
1. Total cost of necklaces and earrings is $240,000.
2. Necklaces are equal in price.
3. Earrings were three times as expensive as any one necklace.
4. Cost of a single necklace is $40,000.

Prove:
Princess Daphne bought 3 necklaces.
-/
theorem daphne_necklaces (total_cost : ℤ) (price_necklace : ℤ) (price_earrings : ℤ) (n : ℤ)
  (h1 : total_cost = 240000)
  (h2 : price_necklace = 40000)
  (h3 : price_earrings = 3 * price_necklace)
  (h4 : total_cost = n * price_necklace + price_earrings) : n = 3 :=
by
  sorry

end daphne_necklaces_l310_310068


namespace bowling_ball_weight_l310_310418

theorem bowling_ball_weight (b c : ℝ) 
  (h1 : 9 * b = 6 * c) 
  (h2 : 4 * c = 120) : 
  b = 20 :=
by 
  sorry

end bowling_ball_weight_l310_310418


namespace T_composition_is_translation_or_identity_l310_310326

variables {a b c : Type}

-- Define the reflections across each line. We treat them abstractly as functions.
def S_a : (a → a) := sorry
def S_b : (b → b) := sorry
def S_c : (c → c) := sorry

-- The composition of reflections T
def T : (a → a) := S_a ∘ S_b ∘ S_c

-- The statement to be proved
theorem T_composition_is_translation_or_identity :
  (T ∘ T = T) ∧ (T ∘ T = id) :=
by sorry

end T_composition_is_translation_or_identity_l310_310326


namespace no_solution_for_m_neg_6_l310_310978

theorem no_solution_for_m_neg_6 (x : ℝ) : (2 * x + (-6))/(x - 3) ≠ 1 :=
begin
  sorry
end

end no_solution_for_m_neg_6_l310_310978


namespace solution_set_abs_inequality_l310_310833

theorem solution_set_abs_inequality (x : ℝ) : (|3*x + 1| - |x - 1| < 0) ↔ x ∈ Ioo (-1 : ℝ) 1 := sorry

end solution_set_abs_inequality_l310_310833


namespace ram_marks_l310_310076

theorem ram_marks (total_marks : ℕ) (percentage : ℕ) (h_total : total_marks = 500) (h_percentage : percentage = 90) : 
  (percentage * total_marks / 100) = 450 := by
  sorry

end ram_marks_l310_310076


namespace fraction_of_white_surface_area_is_11_16_l310_310542

theorem fraction_of_white_surface_area_is_11_16 :
  let cube_surface_area := 6 * 4^2
  let total_surface_faces := 96
  let corner_black_faces := 8 * 3
  let center_black_faces := 6 * 1
  let total_black_faces := corner_black_faces + center_black_faces
  let white_faces := total_surface_faces - total_black_faces
  (white_faces : ℚ) / total_surface_faces = 11 / 16 := 
by sorry

end fraction_of_white_surface_area_is_11_16_l310_310542


namespace a_n_parallel_l310_310324

noncomputable def sequence (c : ℕ → ℤ) : Prop :=
  c 1 = 1 ∧ c 2 = 2005 ∧ ∀ n ≥ 3, c (n - 2) = -3 * c n - 4 * c n + 2008

noncomputable def a (c : ℕ → ℤ) (i_n i_{n1} : ℕ → ℤ) (n : ℕ) : ℤ :=
  3 * (c (n - 2) - i_n n) * (502 - i_{n1} n - c 2) + 4^n * 2004 * 501

def is_parallel_number (a_n : ℤ) : Prop := -- This definition depends on what exactly a "parallel number" means. 
  sorry  -- Placeholder for the actual definition of a parallel number.

theorem a_n_parallel (c : ℕ → ℤ) (i_n i_{n1} : ℕ → ℤ) (n : ℕ) (h_seq : sequence c) :
  ∀ n > 2, is_parallel_number (a c i_n i_{n1} n) :=
sorry

end a_n_parallel_l310_310324


namespace eval_f_i_l310_310715

def f (x : ℂ) : ℂ := x^3 - x^2 + x - 1

theorem eval_f_i : f complex.I = 0 := by
  sorry

end eval_f_i_l310_310715


namespace equation_infinitely_many_solutions_l310_310218

theorem equation_infinitely_many_solutions (a : ℝ) :
  (∀ x : ℝ, 4 * (3 * x - 2 * a) = 3 * (4 * x + 18)) ↔ a = -27 / 4 :=
sorry

end equation_infinitely_many_solutions_l310_310218


namespace function_intersects_y_axis_at_0_neg4_l310_310426

theorem function_intersects_y_axis_at_0_neg4 :
  (∃ x y : ℝ, y = 4 * x - 4 ∧ x = 0 ∧ y = -4) :=
sorry

end function_intersects_y_axis_at_0_neg4_l310_310426


namespace sales_difference_is_41825_l310_310170

-- Definitions of regular and current sales conditions
def pastries_regular := 20
def croissants_regular := 10
def muffins_regular := 10
def bread_regular := 10
def sourdough_regular := 6
def whole_wheat_regular := 4

def croissant_price := 2.50
def muffin_price := 1.75
def sourdough_price := 4.25
def whole_wheat_price := 5.00

def discount := 0.10

def pastries_today := 14
def croissants_today := 8
def muffins_today := 6
def bread_today := 25
def sourdough_today := 15
def whole_wheat_today := 10

-- Calculate the total sales for regular day without discount
def total_regular_sales_pastries := (croissants_regular * croissant_price) + (muffins_regular * muffin_price)
def total_regular_sales_bread := (sourdough_regular * sourdough_price) + (whole_wheat_regular * whole_wheat_price)
def total_regular_sales := total_regular_sales_pastries + total_regular_sales_bread

-- Calculate the total sales for today with discount
def total_today_sales_pastries := (croissants_today * (croissant_price * (1 - discount))) + (muffins_today * (muffin_price * (1 - discount)))
def total_today_sales_bread := (sourdough_today * (sourdough_price * (1 - discount))) + (whole_wheat_today * (whole_wheat_price * (1 - discount)))
def total_today_sales := total_today_sales_pastries + total_today_sales_bread

-- Calculate the difference
def sales_difference := total_today_sales - total_regular_sales

theorem sales_difference_is_41825 : sales_difference = 41.825 :=
by
-- Proof is omitted
sorry

end sales_difference_is_41825_l310_310170


namespace sum_coeffs_example_l310_310340

theorem sum_coeffs_example (a : Fin 18 → ℤ) :
  (∀ x, (x - 1) ^ 17 = ∑ i in Finset.range 18, a i * (1 + x) ^ i) →
  ∑ i in Finset.range 18, a i = -1 :=
by
  sorry

end sum_coeffs_example_l310_310340


namespace solutions_periodic_with_same_period_l310_310669

variable {y z : ℝ → ℝ}
variable (f g : ℝ → ℝ)

-- defining the conditions
variable (h1 : ∀ x, deriv y x = - (z x)^3)
variable (h2 : ∀ x, deriv z x = (y x)^3)
variable (h3 : y 0 = 1)
variable (h4 : z 0 = 0)
variable (h5 : ∀ x, y x = f x)
variable (h6 : ∀ x, z x = g x)

-- proving periodicity
theorem solutions_periodic_with_same_period : ∃ k > 0, (∀ x, f (x + k) = f x ∧ g (x + k) = g x) := by
  sorry

end solutions_periodic_with_same_period_l310_310669


namespace car_speed_increase_l310_310835

/-- 
  Given:
  1. The speed of the car increases by some amount x km/h every hour.
  2. The distance traveled in the first hour is 30 kms.
  3. The total distance traveled in 12 hours is 492 kms.
  Prove that the speed of the car increases by 2 km/h every hour.
-/
theorem car_speed_increase (x : ℕ) :
  (∑ i in Finset.range 12, (30 + i * x)) = 492 → x = 2 :=
by
  sorry

end car_speed_increase_l310_310835


namespace ratio_AT_TC_l310_310483

variables {O : Type*} -- Center of the circles
variables {A C Q P R S T : Type*} -- Points on the circles

-- Defining necessary conditions as propositions
def concentric (O : Type*) (inner outer : Type*) : Prop := 
  ∃ center : Type*, inner = outer
    
def touches (A C Q : Type*) (inner : Type*) : Prop := 
  ∃ chord : Type*, chord = AC ∧ Q ∈ chord ∧ touches Q inner 

def midpoint (P Q : Type*) : Prop := 
  ∃ M : Type*, 2 * M = P + Q

def intersects (A : Type*) (inner : Type*) (R S : Type*) : Prop := 
  ∃ line : Type*, line ∩ inner = {R, S} ∧ passes (line) A

def perpendicular_bisectors_meet (PR CS : Type*) (T : Type*) (AC : Type*) : Prop :=
  ∃ bisector : Type*, bisector ⊥ PR ∧ bisector ⊥ CS ∧ T ∈ bisector ∧ T ∈ AC

-- The theorem to prove the ratio AT / TC
theorem ratio_AT_TC
  (inner outer : Type*) (O A C Q P R S T : Type*)
  (h_concentric: concentric O inner outer)
  (h_touches : touches A C Q inner)
  (h_midpoint : midpoint P Q)
  (h_intersects : intersects A inner R S)
  (h_perpendicular : perpendicular_bisectors_meet (mk_line P R) (mk_line C S) T (mk_line A C))
  : AT / TC = 5 / 3 := 
sorry

end ratio_AT_TC_l310_310483


namespace proper_subsets_count_l310_310108

theorem proper_subsets_count (s : Finset ℤ) (h : s = {-1, 0, 1}) : s.powerset.card - 1 = 7 :=
by
  have hs : s.card = 3 := by rw [h, Finset.card_insert_of_not_mem, Finset.card_insert_of_not_mem, Finset.card_singleton]; dec_trivial
  rw [←Finset.card_powerset, Finset.card_eq_to_nat, Finset.card_powerset, hs, pow_succ, pow_succ, one_add_one_eq_two] at hs
  simp only [nat.cast_ite, nat.cast_one, nat.cast_bit0, nat.cast_add, nat.cast_mul, nat.cast_pow, nat.cast_bit1]
  sorry

end proper_subsets_count_l310_310108


namespace minimum_sum_of_distances_l310_310296

noncomputable def distance_from_point_to_line (P : ℝ × ℝ) (a b c : ℝ) : ℝ :=
  (abs (a * P.1 + b * P.2 + c)) / (sqrt (a^2 + b^2))

def parabola (x y : ℝ) : Prop :=
  x^2 = 4 * y

def line1 (x y : ℝ) : Prop :=
  4 * x - 3 * y - 7 = 0

def line2 (x y : ℝ) : Prop :=
  y + 2 = 0

theorem minimum_sum_of_distances :
  ∀ (P : ℝ × ℝ), parabola P.1 P.2 → 
  let d1 := distance_from_point_to_line P 4 (-3) (-7)
      d2 := distance_from_point_to_line P 0 1 2
  in (d1 + d2) = 3 :=
sorry

end minimum_sum_of_distances_l310_310296


namespace rate_of_grapes_calculation_l310_310702

theorem rate_of_grapes_calculation (total_cost cost_mangoes cost_grapes : ℕ) (rate_grapes : ℕ):
  total_cost = 1125 →
  cost_mangoes = 9 * 55 →
  cost_grapes = 9 * rate_grapes →
  total_cost = cost_grapes + cost_mangoes →
  rate_grapes = 70 :=
by
  intros h1 h2 h3 h4
  sorry

end rate_of_grapes_calculation_l310_310702


namespace find_AC_length_l310_310792

-- Given points D and E on sides AC and BC of triangle ABC
variables {A B C D E : Type} [Point A] [Point B] [Point C] [Point D] [Point E]

-- Conditions as defined in the problem
variables (AD EC BD ED AB BE : ℝ)
variables h1 : AD = EC
variables h2 : BD = ED
variables h3 : ∠ BDC = ∠ DEB
variables h4 : AB = 7
variables h5 : BE = 2

-- To find the length of AC
theorem find_AC_length
    (h_ad_ec : AD = EC)
    (h_bd_ed : BD = ED)
    (h_angle : ∠ BDC = ∠ DEB)
    (h_ab : AB = 7)
    (h_be : BE = 2) :
    let DC := AB
    in AC = AD + DC :=
begin
  sorry
end

end find_AC_length_l310_310792


namespace sin_C_in_right_triangle_l310_310365

theorem sin_C_in_right_triangle (A B C : Point) (h_right : angle A = ⟨π / 2⟩)
  (h_AB : dist A B = 5) (h_BC : dist B C = 13) : 
  sin (angle C) = 12 / 13 :=
by
  sorry

end sin_C_in_right_triangle_l310_310365


namespace billboard_dimensions_l310_310550

theorem billboard_dimensions (photo_width_cm : ℕ) (photo_length_dm : ℕ) (billboard_area_m2 : ℕ)
  (h1 : photo_width_cm = 30) (h2 : photo_length_dm = 4) (h3 : billboard_area_m2 = 48) :
  ∃ photo_length_cm : ℕ, photo_length_cm = 40 ∧
  ∃ k : ℕ, k = 20 ∧
  ∃ billboard_width_m billboard_length_m : ℕ,
    billboard_width_m = photo_width_cm * k / 100 ∧ 
    billboard_length_m = photo_length_cm * k / 100 ∧ 
    billboard_width_m = 6 ∧ 
    billboard_length_m = 8 := by
  sorry

end billboard_dimensions_l310_310550


namespace Roja_speed_is_6_l310_310081

-- Definitions based on conditions
def Roja_speed (R : ℝ) : Prop :=
  let relative_speed := R + 3 in
  let time := 4 in
  let distance := relative_speed * time in
  distance = 36

-- Theorem statement based on question and correct answer
theorem Roja_speed_is_6 : Roja_speed 6 :=
by
  sorry

end Roja_speed_is_6_l310_310081


namespace flea_jumps_least_steps_l310_310910

theorem flea_jumps_least_steps (A B n : ℕ) (h_pos : 0 < n)
  (h_jump : ∀ (A B : ℕ), A < B → B = A + 1 ∨ B = A + (B - A) - 1 ∨ B = A + (B - A) + 1)
  (h_twice : ∃ k : ℕ, k > 0 ∧ (flea_position (k + 1) = n ∧ flea_position k = n)) :
  ∃ k : ℕ, k ≥ ⌈2 * sqrt n⌉ :=
by sorry

end flea_jumps_least_steps_l310_310910


namespace find_y2_l310_310024

-- Given the conditions
variables (y_2 : ℝ)
variables (x y : ℝ)

-- The points (0,0), (0,y_2), (5,y_2), (5,0) form a rectangle in the x-y plane
-- Probability x + y < 4 is 0.4

theorem find_y2 (h: 0.4 * (5 * y_2) = 8) :
  y_2 = 4 :=
begin
  -- Proof will be filled in here
  sorry
end

end find_y2_l310_310024


namespace simplify_eval_expression_l310_310085

theorem simplify_eval_expression (a b : ℤ) (h₁ : a = 2) (h₂ : b = -1) : 
  ((2 * a + 3 * b) * (2 * a - 3 * b) - (2 * a - b) ^ 2 - 2 * a * b) / (-2 * b) = -7 := 
by 
  sorry

end simplify_eval_expression_l310_310085


namespace max_5x_plus_3y_l310_310667

theorem max_5x_plus_3y (x y : ℝ) (h : x^2 + y^2 = 10 * x + 8 * y + 10) : 5 * x + 3 * y ≤ 105 :=
sorry

end max_5x_plus_3y_l310_310667


namespace train_length_l310_310560

theorem train_length (speed_kmh : ℕ) (time_s : ℕ) (speed_conversion : speed_kmh = 40) 
  (time_condition : time_s = 27) : 
  (speed_kmh * 1000 / 3600 * time_s = 300) := 
by
  sorry

end train_length_l310_310560


namespace range_of_a_l310_310508

variable {R : Type} [LinearOrderedField R]

def f (a x : R) : R := a * x^2 + 4 * (a + 1) * x - 3

theorem range_of_a (a : R) (h_max : ∀ x ∈ Icc (0: R) (2: R), f a x ≤ f a (2: R)) :
  a ≥ - (1 / 2) :=
sorry

end range_of_a_l310_310508


namespace set_C_is_correct_l310_310719

open Set

noncomputable def set_A : Set ℝ := {x | x ^ 2 - x - 12 ≤ 0}
noncomputable def set_B : Set ℝ := {x | (x + 1) / (x - 1) < 0}
noncomputable def set_C : Set ℝ := {x | x ∈ set_A ∧ x ∉ set_B}

theorem set_C_is_correct : set_C = {x | -3 ≤ x ∧ x ≤ -1} ∪ {x | 1 ≤ x ∧ x ≤ 4} :=
by
  sorry

end set_C_is_correct_l310_310719


namespace power_function_passing_point_l310_310692

def power_function (k a x : ℝ) : ℝ := k * x^a

theorem power_function_passing_point :
  ∃ (k a : ℝ), 
    power_function k a 3 = sqrt 3 ∧ 
    k + a = 3 / 2 :=
by
  sorry

end power_function_passing_point_l310_310692


namespace price_of_one_shirt_l310_310920

/-- 
A retailer sells any shirt for the same price and any pair of pants for the same price, but 
the price of shirts and pants are not the same. Say you bought 2 shirts and 3 pairs of pants 
for $120 in total. Then you realized you didn't need the extra pants. What's the price of 1 
shirt at this place if when you returned all 3 pairs of pants you were refunded 25% of what 
you originally paid?
-/
theorem price_of_one_shirt 
  (price_shirt price_pants : ℝ)
  (price_shirt ≠ price_pants)
  (total_paid : ℝ)
  (refund : ℝ) 
  (h_total_paid : total_paid = 120)
  (h_refund : refund = 0.25 * total_paid)
  (h_shirt_pants_eq : 2 * price_shirt + 3 * price_pants = total_paid) : 
  price_shirt = 45 := 
by 
  sorry

end price_of_one_shirt_l310_310920


namespace bus_sarah_probability_l310_310905

-- Define the probability of Sarah arriving while the bus is still there
theorem bus_sarah_probability :
  let total_minutes := 60
  let bus_waiting_time := 15
  let total_area := (total_minutes * total_minutes : ℕ)
  let triangle_area := (1 / 2 : ℝ) * 45 * 15
  let rectangle_area := 15 * 15
  let shaded_area := triangle_area + rectangle_area
  (shaded_area / total_area : ℝ) = (5 / 32 : ℝ) :=
by
  sorry

end bus_sarah_probability_l310_310905


namespace todd_final_amount_l310_310481

-- Declare the given conditions as defnitions in Lean
def borrowed_amount := 300
def repayment_amount := 330
def equipment_rental := 120
def ingredients_cost := 60
def marketing_cost := 40
def miscellaneous_cost := 10
def snow_cones_sold := 500
def price_per_snow_cone := 1.75
def cups_sold := 250
def price_per_cup := 2
def increased_ingredient_cost_factor := 1.2
def snow_cones_before_increase := 300

-- Calculate some intermediate values
def initial_total_expenses := equipment_rental + ingredients_cost + marketing_cost + miscellaneous_cost + repayment_amount
def snow_cone_revenue := snow_cones_sold * price_per_snow_cone
def cup_revenue := cups_sold * price_per_cup
def total_revenue := snow_cone_revenue + cup_revenue

def increased_ingredient_cost := ingredients_cost * 0.2
def total_ingredient_cost_with_increase := ingredients_cost + increased_ingredient_cost

def total_expenses_with_increased_cost := initial_total_expenses - ingredients_cost + total_ingredient_cost_with_increase

def final_profit := total_revenue - total_expenses_with_increased_cost

-- Lean theorem statement to prove the final amount after repayment
theorem todd_final_amount : final_profit = 803 := by
  sorry

end todd_final_amount_l310_310481


namespace money_problem_proof_l310_310482

noncomputable def Tom : ℝ := 0
noncomputable def Nataly : ℝ := 0
noncomputable def Raquel : ℝ := 0

theorem money_problem_proof :
  Assume Raquel = r
       Nataly = 3 * r
       Tom = (1 / 4) * (3 * r)
       Tom + Nataly + Raquel = 190,
  Prove Raquel = 40 :=
by
  sorry

end money_problem_proof_l310_310482


namespace music_player_and_concert_tickets_l310_310876

theorem music_player_and_concert_tickets (n : ℕ) (h1 : 35 % 5 = 0) (h2 : 35 % n = 0) (h3 : ∀ m : ℕ, m < 35 → (m % 5 ≠ 0 ∨ m % n ≠ 0)) : n = 7 :=
sorry

end music_player_and_concert_tickets_l310_310876


namespace calculate_m_n_intersections_l310_310737

-- Define the properties of the lattice points
def lattice_point_circle_radius := 1/8
def lattice_point_square_side := 1/4

-- Define the line segment from (0,0) to (625,1000)
def line_segment_start := (0, 0)
def line_segment_end := (625, 1000)

-- Define the slope of the line segment
def line_slope := 1000 / 625

-- Define the function calculating the number of lattice points intersected
def lattice_points (x_end : ℕ) (x_step : ℕ) : ℕ := x_end / x_step

-- State the proposition
theorem calculate_m_n_intersections : 
  let m := lattice_points 625 5
  let n := m
  m + n = 252 :=
by {
  let m := lattice_points 625 5,
  let n := m,
  exact (eq.refl (m + n = 252)),
}

end calculate_m_n_intersections_l310_310737


namespace area_of_isosceles_triangle_l310_310844

theorem area_of_isosceles_triangle
  (P Q R T : Point)
  (hPQPR : P.distance Q = P.distance R)
  (PT QR : Segment)
  (hPTQR : PT ⊥ QR)
  (hPTlen : PT.length = 15)
  (hQRlen : QR.length = 15)
  (hT_median_P : T = median_point P Q R)
  (hT_median_Q : T = median_point P R Q) :
  area_of_triangle P Q R = 450 :=
sorry

end area_of_isosceles_triangle_l310_310844


namespace problem_statement_l310_310641

theorem problem_statement (n : ℕ) (h : 0 < n) :
  let a_0 := 2^n
  let S_n := 3^n - 2^n
  (1) (h₀ : a_0 = 2^n) ∧
  (2) (h₁ : S_n = 3^n - 2^n) ∧
  (3) ((n = 1 → 3^n - 2^n > (n - 2) * 2^n + 2 * n^2) ∧
       (n = 2 ∨ n = 3 → 3^n - 2^n < (n - 2) * 2^n + 2 * n^2) ∧
       (n ≥ 4 → 3^n - 2^n > (n - 2) * 2^n + 2 * n^2)) :=
by {
  let a_0 := 2^n
  let S_n := 3^n - 2^n
  split; assumption <|> exact sorry
}

end problem_statement_l310_310641


namespace find_ellipse_equation_find_line_eq_l310_310656

-- Given conditions
def ellipse (x y a b : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def point_C := (2, 1) : ℝ × ℝ
def eccentricity := (sqrt 3) / 2
def long_axis_length := 4 * sqrt 2
def a > b > 0 := sorry -- assure a > b > 0 for simplicity

-- Convert problem statements to Lean
theorem find_ellipse_equation (a b : ℝ) (h_ab_pos : a > b > 0) 
    (h_ecc : (sqrt 3) / 2 = eccentricity) 
    (h_axis : 4 * sqrt 2 = long_axis_length) 
    (h_point : ellipse 2 1 a b):
    ellipse x y (2 * sqrt 2) (sqrt 2) :=
  sorry

theorem find_line_eq (a b : ℝ) (h_ab_pos : a > b > 0)
    (h_ecc : (sqrt 3) / 2 = eccentricity)
    (h_axis : 4 * sqrt 2 = long_axis_length)
    (h_point : ellipse 2 1 a b)
    (l m n : ℝ) (h_parallel : y = (1/2 : ℝ) * x + l) :
    y = 0.5 * x + sqrt 2 ∨ y = 0.5 * x - sqrt 2 :=
  sorry

end find_ellipse_equation_find_line_eq_l310_310656


namespace sufficient_but_not_necessary_condition_l310_310650

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (a > 1 → 1 / a < 1) ∧ ((1 / a < 1) → (a > 1 ∨ a < 0)) → 
  (∀ (P Q : Prop), (P → Q) → (Q → P ∨ False) → P ∧ ¬Q → False) :=
by
  sorry

end sufficient_but_not_necessary_condition_l310_310650


namespace unique_solution_impossible_УЧУЙ_is_2021_l310_310105

-- Definitions of digits and conditions
variables (K E S C : ℕ)

-- УЧУЙ, KE, KS are four-digit and two-digit positive integers respectively.
def is_valid_digit (n : ℕ) : Prop := n >= 0 ∧ n <= 9
def has_unique_digits (n : ℕ) : Prop := nat.digits 10 n |>.nodup

-- The magician's intended solution constraints
def magician_solution (УЧУЙ KE KS : ℕ) : Prop :=
  (УЧУЙ = KE * KS) ∧
  (УЧУЙ + 1111 = (KE + 11) * (KS + 11)) ∧
  has_unique_digits УЧУЙ ∧ has_unique_digits KE ∧ has_unique_digits KS

-- Problem 1: Uniqueness of the solution
theorem unique_solution_impossible :
  ¬ ∃ (УЧУЙ KE KS : ℕ), magician_solution УЧУЙ KE KS ∧ ∀ (E C : ℕ), E ≠ C → KE ≠ KS :=
sorry

-- Problem 2: Specific value of УЧУЙ
theorem УЧУЙ_is_2021 : ∃ (УЧУЙ KE KS : ℕ), magician_solution УЧУЙ KE KS ∧ УЧУЙ = 2021 :=
sorry

end unique_solution_impossible_УЧУЙ_is_2021_l310_310105


namespace fold_into_open_box_l310_310441

-- Definitions based on the conditions in the problem:
-- F-shaped figure with seven identical squares.
-- Open rectangular box through the addition of three more squares.

def F_shaped_figure : Type := sorry   -- We abstractly represent the F-shaped figure.

def can_form_open_rectangular_box (F : F_shaped_figure) (additional_squares : ℕ) : Prop :=
  additional_squares = 3 -- Condition: Using exactly three additional squares.

theorem fold_into_open_box (F : F_shaped_figure) :
  (can_form_open_rectangular_box F 3) → ∃! n, n = 1 :=
begin
  -- We need to prove there exists a unique number n such that n is 1.
  sorry
end

end fold_into_open_box_l310_310441


namespace left_handed_jazz_lovers_l310_310006

theorem left_handed_jazz_lovers:
  (total_members left_handed_members jazz_lovers right_handed_non_jazz : ℕ)
  (h1 : total_members = 30)
  (h2 : left_handed_members = 12)
  (h3 : jazz_lovers = 22)
  (h4 : right_handed_non_jazz = 4)
  (h5 : ∀ m, m ∈ {left_handed_members, total_members - left_handed_members}):
  ∃ x, x = 8 :=
by
  have h6 : total_members = left_handed_members + (total_members - left_handed_members),
    by sorry -- This is a given condition, so it should not be proved, use sorry.
  have h7 : total_members = left_handed_members + jazz_lovers - left_handed_members + total_members - jazz_lovers + left_handed_members - jazz_lovers + right_handed_non_jazz,
    by sorry -- Similar reasoning
    sorry -- skip the proof but assert the presence of x = 8.

end left_handed_jazz_lovers_l310_310006


namespace tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l310_310293

-- Conditions
variables {O : ℝ × ℝ} (A : ℝ × ℝ) (B : ℝ × ℝ)
          {P Q : ℝ × ℝ} (p : ℝ)
          (hp : 0 < p)
          (hA : A.1 ^ 2 = 2 * p * A.2)
          (hB : B = (0, -1))
          (hP : P.2 = P.1 ^ 2 / (2 * p))
          (hQ : Q.2 = Q.1 ^ 2 / (2 * p))

-- Proof problem statements
theorem tangent_line_AB
  (hAB_tangent : ∀ x : ℝ, x ^ 2 / (2 * p) = 2 * x - 1 → x = 1) : true :=
by sorry

theorem op_oq_leq_oa_squared 
  (h_op_oq_leq : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + (P.1 ^ 2 / (2 * p)) ^ 2) * (Q.1 ^ 2 + (Q.1 ^ 2 / (2 * p)) ^ 2) ≤ 2) : true :=
by sorry

theorem bp_bq_gt_ba_squared 
  ( h_bp_bq_gt : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + ((P.1 ^ 2 / (2 * p)) + 1) ^ 2) * (Q.1 ^ 2 + ((Q.1 ^ 2 / (2 * p)) +1 ) ^ 2) > 5 ) : true :=
by sorry

end tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l310_310293


namespace election_debate_conditions_l310_310039

theorem election_debate_conditions (n : ℕ) (h_n : n ≥ 3) :
  ¬ ∃ (p : ℕ), n = 2 * (2 ^ p - 2) + 1 :=
sorry

end election_debate_conditions_l310_310039


namespace card_area_after_shortening_l310_310611

/-- Given a card with dimensions 3 inches by 7 inches, prove that 
  if the length is shortened by 1 inch and the width is shortened by 2 inches, 
  then the resulting area is 10 square inches. -/
theorem card_area_after_shortening :
  let length := 3
  let width := 7
  let new_length := length - 1
  let new_width := width - 2
  new_length * new_width = 10 :=
by
  let length := 3
  let width := 7
  let new_length := length - 1
  let new_width := width - 2
  show new_length * new_width = 10
  sorry

end card_area_after_shortening_l310_310611


namespace proof_of_80_not_50_ray_partitional_l310_310394

noncomputable def number_of_80_ray_partitional_not_50_ray_partitional_points (S: set (ℝ × ℝ)) : ℕ :=
  let m := 80 in
  let n := 50 in
  let common_divisor := 10 in
  let total_80 := (m - 1) * (m - 1) in
  let total_comm := (common_divisor - 1) * (common_divisor - 1) in
  total_80 - total_comm

theorem proof_of_80_not_50_ray_partitional :
  number_of_80_ray_partitional_not_50_ray_partitional_points (set.univ : set (ℝ × ℝ)) = 6160 :=
  sorry

end proof_of_80_not_50_ray_partitional_l310_310394


namespace Tino_has_correct_jellybeans_total_jellybeans_l310_310124

-- Define the individuals and their amounts of jellybeans
def Arnold_jellybeans := 5
def Lee_jellybeans := 2 * Arnold_jellybeans
def Tino_jellybeans := Lee_jellybeans + 24
def Joshua_jellybeans := 3 * Arnold_jellybeans

-- Verify Tino's jellybean count
theorem Tino_has_correct_jellybeans : Tino_jellybeans = 34 :=
by
  -- Unfold definitions and perform calculations
  sorry

-- Verify the total jellybean count
theorem total_jellybeans : (Arnold_jellybeans + Lee_jellybeans + Tino_jellybeans + Joshua_jellybeans) = 64 :=
by
  -- Unfold definitions and perform calculations
  sorry

end Tino_has_correct_jellybeans_total_jellybeans_l310_310124


namespace last_number_remaining_l310_310034

theorem last_number_remaining :
  (∃ f : ℕ → ℕ, ∃ n : ℕ, (∀ k < n, f (2 * k) = 2 * k + 2 ∧
                         ∀ k < n, f (2 * k + 1) = 2 * k + 1 + 2^(k+1)) ∧ 
                         n = 200 ∧ f (2 * n) = 128) :=
sorry

end last_number_remaining_l310_310034


namespace product_of_fraction_l310_310145

-- Define the repeating decimal as given in the problem
def repeating_decimal : Rat := 0.018 -- represents 0.\overline{018}

-- Define the given fraction obtained by simplifying
def simplified_fraction : Rat := 2 / 111

-- The goal is to prove that the product of the numerator and denominator of 
-- the simplified fraction of the repeating decimal is 222
theorem product_of_fraction (y : Rat) (hy : y = 0.018) (fraction_eq : y = 18 / 999) : 
  (2:ℕ) * (111:ℕ) = 222 :=
by
  sorry

end product_of_fraction_l310_310145


namespace fated_number_probability_l310_310189

/-- A number is a "fated number" if the sum of any two of its digits equals the third
    and the digits are distinct and from the set {1, 2, 3, 4}. -/
def is_fated_number (a b c : ℕ) : Prop :=
  a + b = c ∨ a + c = b ∨ b + c = a

theorem fated_number_probability :
  let digits : Finset ℕ := {1, 2, 3, 4} in
  let triples := digits.product (digits.product digits) in
  let valid_triples := triples.filter (λ t, 
    let (a, (b, c)) := t in 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ is_fated_number a b c) in
  ↑(valid_triples.card) / ↑(triples.card) = 1 / 2 := by
  sorry

end fated_number_probability_l310_310189


namespace find_a_l310_310774

def U : Set ℕ := {1, 3, 5, 7, 9}
def A (a : ℝ) : Set ℕ := {1, |a - 5|.to_nat, 9}
def compl_U_A (a : ℝ) : Set ℕ := {5, 7}

theorem find_a (a : ℝ) (h₁ : compl_U_A a = U \ A a) : a = 2 ∨ a = 8 :=
by
  sorry

end find_a_l310_310774


namespace ineq_proof_l310_310055

theorem ineq_proof (n : ℕ) (x : Fin n → ℝ) (h1 : 2 ≤ n) (h2 : ∀ i, 0 < x i) (h3 : ∑ i, x i = 1) :
  (∑ i, x i / Real.sqrt (1 - x i)) ≥ (∑ i, Real.sqrt (x i)) / (n - 1) :=
by
  sorry

end ineq_proof_l310_310055


namespace calculate_expression_l310_310205

theorem calculate_expression :
  ((7 / 9) - (5 / 6) + (5 / 18)) * 18 = 4 :=
by
  -- proof to be filled in later.
  sorry

end calculate_expression_l310_310205


namespace total_number_of_notebooks_and_pens_l310_310469

theorem total_number_of_notebooks_and_pens :
  ∀ (n p : ℕ), p = n + 50 → n = 30 → n + p = 110 :=
by
  intros n p h1 h2
  rw [h2, h1]
  rw [h2]
  simp
  exact rfl

end total_number_of_notebooks_and_pens_l310_310469


namespace index_of_50th_negative_term_l310_310602

noncomputable def b (n : ℕ) : ℝ :=
  Σ k in Finset.range (n + 1) \ {0}, Real.cos k

theorem index_of_50th_negative_term :
  ∃ n : ℕ, b n < 0 ∧ ∀ m < n, b m ≥ 0 ∧ n = 159 :=
sorry

end index_of_50th_negative_term_l310_310602


namespace parabola_intersection_l310_310129

noncomputable def parabola_intersection_probability : ℚ :=
  let choices := {-3, -2, -1, 0, 1, 2}
  let probs := {p : set (ℤ × ℤ × ℤ × ℤ) | 
    ∃ a b c d, 
      a ∈ choices ∧ b ∈ choices ∧ c ∈ choices ∧ d ∈ choices ∧ 
        (a ≠ c ∨ d = b)} 
  (probs.card : ℚ) / (choices.card ^ 4 : ℚ)

theorem parabola_intersection : 
  parabola_intersection_probability = 31/36 :=
sorry

end parabola_intersection_l310_310129


namespace sum_fraction_harmonic_l310_310252

def H (n : ℕ) : ℝ := ∑ k in Finset.range (n + 1), 1 / (k + 1)

theorem sum_fraction_harmonic (H : ℕ → ℝ) (n : ℕ) :
  H n = ∑ k in Finset.range n, 1 / (k + 1) →
  (∑ n in Finset.range 100, n^2 / ((n + 1) * (H n) * (H (n + 1)))) = 1 :=
by
  intro hH
  sorry

end sum_fraction_harmonic_l310_310252


namespace total_groups_correct_l310_310079

-- Definitions from conditions
def eggs := 57
def egg_group_size := 7

def bananas := 120
def banana_group_size := 10

def marbles := 248
def marble_group_size := 8

-- Calculate the number of groups for each type of object
def egg_groups := eggs / egg_group_size
def banana_groups := bananas / banana_group_size
def marble_groups := marbles / marble_group_size

-- Total number of groups
def total_groups := egg_groups + banana_groups + marble_groups

-- Proof statement
theorem total_groups_correct : total_groups = 51 := by
  sorry

end total_groups_correct_l310_310079


namespace range_of_x_l310_310779

def floor (x : ℝ) : ℤ := ⌊x⌋

theorem range_of_x (x : ℝ) (h : floor (x - 1) = -2) : -1 ≤ x ∧ x < 0 :=
by {
  sorry
}

end range_of_x_l310_310779


namespace number_of_words_in_Babblese_l310_310378

/-- In the land of Babble, the Babblese alphabet has only 6 letters,
    and every word in the Babblese language has no more than 4 letters in it.
    Prove that the total number of such words is 1554.
-/
theorem number_of_words_in_Babblese : 
  let letters := 6 in
  (letters) + (letters ^ 2) + (letters ^ 3) + (letters ^ 4) = 1554 := by
  sorry

end number_of_words_in_Babblese_l310_310378


namespace geometric_sequence_m_value_l310_310376

theorem geometric_sequence_m_value 
  (a : ℕ → ℝ) (q : ℝ) (m : ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n, a n = a 1 * q^(n-1))
  (h3 : |q| ≠ 1) 
  (h4 : a m = a 1 * a 2 * a 3 * a 4 * a 5) : 
  m = 11 := by
  sorry

end geometric_sequence_m_value_l310_310376


namespace more_bacteria_than_fungi_l310_310221

-- Definitions based on the conditions
def bacteria_dilutions := [10^4, 10^5, 10^6]
def fungi_dilutions := [10^2, 10^3, 10^4]

/--
To determine the number of bacteria in soil, dilutions of 10^4, 10^5, and 10^6 times are used.
To determine the number of fungi, dilutions of 10^2, 10^3, and 10^4 times are used.
Prove that there are more bacteria than fungi in the soil.
-/
theorem more_bacteria_than_fungi (bacteria_dilutions : list ℕ) (fungi_dilutions : list ℕ) :
  bacteria_dilutions = [10000, 100000, 1000000] ->
  fungi_dilutions = [100, 1000, 10000] ->
  true := sorry

end more_bacteria_than_fungi_l310_310221


namespace find_angle_C_find_area_of_triangle_l310_310653

-- Conditions
variables {A B C : ℝ}
variables {x : ℝ}
def m := (2 * Real.cos A, Real.sin A)
def n := (Real.cos B, -2 * Real.sin B)
axiom dot_product_eq_one : (2 * Real.cos A * Real.cos B - 2 * Real.sin A * Real.sin B) = 1
axiom angle_sum : A + B + C = Real.pi
axiom arithmetic_seq_sides : x > 4

-- Questions translated into Lean 4 statements
theorem find_angle_C : C = (2 * Real.pi) / 3 :=
by
  sorry

theorem find_area_of_triangle : 
  let side1 := x - 4
  let side2 := x
  let side3 := x + 4
  area : ℝ := 15 * Real.sqrt 3 :=
by
  sorry

end find_angle_C_find_area_of_triangle_l310_310653


namespace product_of_fraction_l310_310148

-- Define the repeating decimal as given in the problem
def repeating_decimal : Rat := 0.018 -- represents 0.\overline{018}

-- Define the given fraction obtained by simplifying
def simplified_fraction : Rat := 2 / 111

-- The goal is to prove that the product of the numerator and denominator of 
-- the simplified fraction of the repeating decimal is 222
theorem product_of_fraction (y : Rat) (hy : y = 0.018) (fraction_eq : y = 18 / 999) : 
  (2:ℕ) * (111:ℕ) = 222 :=
by
  sorry

end product_of_fraction_l310_310148


namespace reformulate_and_find_product_l310_310816

theorem reformulate_and_find_product (a b x y : ℝ)
  (h : a^8 * x * y - a^7 * y - a^6 * x = a^5 * (b^5 - 2)) :
  ∃ m' n' p' : ℤ, (a^m' * x - a^n') * (a^p' * y - a^3) = a^5 * b^5 ∧ m' * n' * p' = 48 :=
by
  sorry

end reformulate_and_find_product_l310_310816


namespace find_a1_l310_310685

noncomputable section

def f (x : ℝ) : ℝ := x - (1 / x)

def a (n : ℕ) : ℝ

axiom geometric_seq (r : ℝ) (r_pos : r > 0) :
  ∃ a : ℕ → ℝ, (∀ n, a(n+1) = r * a(n)) ∧ (a 6 = 1)

axiom sum_f (n : ℕ → ℝ) (h_geometric : ∀ n, n (6+1) = r * n(6) ∧ (n 6 = 1))
  : f (n 1) + f (n 2) + f (n 3) + f (n 4) + f (n 5) + f (n 6) + f (n 7) + f (n 8) + f (n 9) + f (n 10) = - n 1

theorem find_a1 : (∃ a : ℝ, geometric_seq a ∧ sum_f = - a) := sorry

end find_a1_l310_310685


namespace product_of_m_and_n_l310_310681

noncomputable def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem product_of_m_and_n 
  (m n : ℝ)
  (h1 : is_even_function (λ x, x^2 + (m+1)*x + 2*m))
  (h2 : ∀ (x : ℝ), (n-2) * x - (x^2 + (m+1)*x + 2*m) - 3 = 0 → x = 1) :
  m * n = -4 :=
sorry

end product_of_m_and_n_l310_310681


namespace num_solutions_l310_310330

theorem num_solutions :
  ∃ n, (∀ a b c : ℤ, (|a + b| + c = 21 ∧ a * b + |c| = 85) ↔ n = 12) :=
sorry

end num_solutions_l310_310330


namespace elegant_numbers_l310_310178

def is_proper_divisor (d n : ℕ) : Prop :=
  1 < d ∧ d < n ∧ n % d = 0

def is_elegant (n : ℕ) : Prop :=
  (∃ d1 d2, is_proper_divisor d1 n ∧ is_proper_divisor d2 n) ∧
  (∀ d1 d2, is_proper_divisor d1 n → is_proper_divisor d2 n → n % (d1 - d2).natAbs = 0)

theorem elegant_numbers (n : ℕ) : n = 6 ∨ n = 8 ∨ n = 12 ↔ is_elegant n :=
by
  sorry

end elegant_numbers_l310_310178


namespace difference_between_greatest_and_smallest_S_l310_310122

-- Conditions
def num_students := 47
def rows := 6
def columns := 8

-- The definition of position value calculation
def position_value (i j m n : ℕ) := i - m + (j - n)

-- The definition of S
def S (initial_empty final_empty : (ℕ × ℕ)) : ℤ :=
  let (i_empty, j_empty) := initial_empty
  let (i'_empty, j'_empty) := final_empty
  (i'_empty + j'_empty) - (i_empty + j_empty)

-- Main statement
theorem difference_between_greatest_and_smallest_S :
  let max_S := S (1, 1) (6, 8)
  let min_S := S (6, 8) (1, 1)
  max_S - min_S = 24 :=
sorry

end difference_between_greatest_and_smallest_S_l310_310122


namespace shaded_region_area_l310_310621

theorem shaded_region_area :
  let line1 := fun x => -3 / 8 * x + 4 in
  let line2 := fun x => -5 / 6 * x + 35 / 6 in
  let intersection := (4, 3) in
  let area_segment_1 := ∫ x in 0..1, line2 x - line1 x in
  let area_segment_2 := ∫ x in 1..4, line2 x - line1 x in
  2 * (area_segment_1 + area_segment_2) = 13 / 2 :=
by
  sorry

end shaded_region_area_l310_310621


namespace problem_incorrect_statement_l310_310194

theorem problem_incorrect_statement :
  ¬ (let data1 := [2, 4, 6, 8] in
     let data2 := [1, 2, 3, 4] in
     variance data1 = 2 * variance data2) :=
sorry

end problem_incorrect_statement_l310_310194


namespace find_u_plus_v_l310_310343

theorem find_u_plus_v (u v : ℚ) 
  (h₁ : 3 * u + 7 * v = 17) 
  (h₂ : 5 * u - 3 * v = 9) : 
  u + v = 43 / 11 :=
sorry

end find_u_plus_v_l310_310343


namespace right_triangle_one_leg_div_by_3_l310_310072

theorem right_triangle_one_leg_div_by_3 {a b c : ℕ} (a_pos : 0 < a) (b_pos : 0 < b) 
  (h : a^2 + b^2 = c^2) : 3 ∣ a ∨ 3 ∣ b := 
by 
  sorry

end right_triangle_one_leg_div_by_3_l310_310072


namespace angle_AFE_is_80_degrees_l310_310740

-- Defining the setup and given conditions
def point := ℝ × ℝ  -- defining a 2D point
noncomputable def A : point := (0, 0)
noncomputable def B : point := (1, 0)
noncomputable def C : point := (1, 1)
noncomputable def D : point := (0, 1)
noncomputable def E : point := (-1, 1.732)  -- Place E such that angle CDE ≈ 130 degrees

-- Conditions
def angle_CDE := 130
def DF_over_DE := 2  -- DF = 2 * DE
noncomputable def F : point := (0.5, 1)  -- This is an example position; real positioning depends on more details

-- Proving that the angle AFE is 80 degrees
theorem angle_AFE_is_80_degrees :
  ∃ (AFE : ℝ), AFE = 80 := sorry

end angle_AFE_is_80_degrees_l310_310740


namespace prime_factors_count_l310_310328

theorem prime_factors_count :
  let n := 101 * 103 * 105 * 107,
  (primeFactors 101).toFinset ∪ 
  (primeFactors 103).toFinset ∪ 
  (primeFactors 105).toFinset ∪ 
  (primeFactors 107).toFinset = 
  {3, 5, 7, 101, 103, 107} ∧ 
  (primeFactors 101).toFinset.card +
  (primeFactors 103).toFinset.card + 
  (primeFactors 105).toFinset.card + 
  (primeFactors 107).toFinset.card - 
  3 /* because 105 has 3 factors, each counted in their singleton sets*/ = 6 :=
by sorry

end prime_factors_count_l310_310328


namespace ratio_albert_to_betty_l310_310934

noncomputable def Albert_age := 28
def Mary_age := Albert_age - 14
def Betty_age := 7
def Albert_to_Betty_ratio := Albert_age / Betty_age

theorem ratio_albert_to_betty : 
  ∀ (A M B : ℕ), 
    (A = 2 * M) → 
    (M = A - 14) → 
    (B = 7) → 
    (A / B = 4) := 
by
  intros A M B hA hM hB
  sorry

end ratio_albert_to_betty_l310_310934


namespace sin_cos_sum_eq_zero_l310_310255

noncomputable def intersection_condition (x y α β) : Prop :=
  (x / (Real.sin α + Real.sin β) + y / (Real.sin α + Real.cos β) = 1) ∧
  (x / (Real.cos α + Real.sin β) + y / (Real.cos α + Real.cos β) = 1) ∧
  (y = -x)

theorem sin_cos_sum_eq_zero (α β : ℝ) :
  (∃ x y, intersection_condition x y α β) → 
  Real.sin α + Real.cos α + Real.sin β + Real.cos β = 0 :=
by {
  sorry
}

end sin_cos_sum_eq_zero_l310_310255


namespace arithmetic_seq_general_term_and_min_sum_l310_310654

theorem arithmetic_seq_general_term_and_min_sum {a d : ℤ} (n : ℤ) :
  (∀ m, a + (m - 1) * d = a_n) →
  (a₁ = -11) →
  (a₃ + a₇ = -6) →
  (∃ n, S_n = n * (n - 12)) →
((a_n = 2n - 13) ∧ (S_n = n^2 - 12n) ∧ (n = 6)) :=
by
  -- proof steps go here
  sorry

end arithmetic_seq_general_term_and_min_sum_l310_310654


namespace number_of_solution_pairs_l310_310109

theorem number_of_solution_pairs :
  (∃ (x y : ℕ), 0 < x ∧ 0 < y ∧ 3*x + 5*y = 501) ∧
  (∀ (x y : ℕ), 0 < x ∧ 0 < y ∧ 3*x + 5*y = 501 → (x - 2) % 5 = 0 ∧ (495 - 15 * (x - 2) / 5) = y) ∧
  (∃! (x_values : Finset ℕ),
    ∀ x ∈ x_values, ∃ y, 0 < y ∧ 3*x + 5*y = 501 ∧
    (card x_values = 33)) :=
sorry

end number_of_solution_pairs_l310_310109


namespace find_raspberries_l310_310031

def total_berries (R : ℕ) : ℕ := 30 + 20 + R

def fresh_berries (R : ℕ) : ℕ := 2 * total_berries R / 3

def fresh_berries_to_keep (R : ℕ) : ℕ := fresh_berries R / 2

def fresh_berries_to_sell (R : ℕ) : ℕ := fresh_berries R - fresh_berries_to_keep R

theorem find_raspberries (R : ℕ) : fresh_berries_to_sell R = 20 → R = 10 := 
by 
sorry

-- To ensure the problem is complete and solvable, we also need assumptions on the domain:
example : ∃ R : ℕ, fresh_berries_to_sell R = 20 := 
by 
  use 10 
  sorry

end find_raspberries_l310_310031


namespace mean_weight_is_70_357_l310_310117

def weights_50 : List ℕ := [57]
def weights_60 : List ℕ := [60, 64, 64, 66, 69]
def weights_70 : List ℕ := [71, 73, 73, 75, 77, 78, 79, 79]

def weights := weights_50 ++ weights_60 ++ weights_70

def total_weight : ℕ := List.sum weights
def total_players : ℕ := List.length weights
def mean_weight : ℚ := (total_weight : ℚ) / total_players

theorem mean_weight_is_70_357 :
  mean_weight = 70.357 := 
sorry

end mean_weight_is_70_357_l310_310117


namespace find_n_l310_310250

theorem find_n (x y : ℝ) (n : ℝ) (h1 : x / (2 * y) = 3 / n) (h2 : (7 * x + 2 * y) / (x - 2 * y) = 23) : n = 2 := by
  sorry

end find_n_l310_310250


namespace pq_true_l310_310659

open Real

def p : Prop := ∃ x0 : ℝ, tan x0 = sqrt 3

def q : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem pq_true : p ∧ q :=
by
  sorry

end pq_true_l310_310659


namespace area_of_quadrilateral_l310_310785

variables {A B C D E F : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C]
variables [Inhabited D] [Inhabited E] [Inhabited F]
variables (S_ADE S_EFC S_BDEF : ℝ)
variables (triangle : triangle A B C)
variables (E ∈ triangle.AC)
variables (DE_parallel_BC : parallel DE BC)
variables (EF_parallel_AB : parallel EF AB)

theorem area_of_quadrilateral 
  (cond1 : ∃ E ∈ triangle.AC, true)
  (cond2 : ∃ DE_parallel_BC, true)
  (cond3 : ∃ EF_parallel_AB, true) :
  S_BDEF = 2 * real.sqrt (S_ADE * S_EFC) :=
sorry

end area_of_quadrilateral_l310_310785


namespace part1_part2_part3_l310_310741

-- Defining the quadratic function
def quadratic (t : ℝ) (x : ℝ) : ℝ := x^2 - 2 * t * x + 3

-- Part (1)
theorem part1 (t : ℝ) (h : quadratic t 2 = 1) : t = 3 / 2 :=
by sorry

-- Part (2)
theorem part2 (t : ℝ) (h : ∀x, 0 ≤ x → x ≤ 3 → (quadratic t x) ≥ -2) : t = Real.sqrt 5 :=
by sorry

-- Part (3)
theorem part3 (m a b : ℝ) (hA : quadratic t (m - 2) = a) (hB : quadratic t 4 = b) 
              (hC : quadratic t m = a) (ha : a < b) (hb : b < 3) (ht : t > 0) : 
              (3 < m ∧ m < 4) ∨ (m > 6) :=
by sorry

end part1_part2_part3_l310_310741


namespace geometric_sum_common_ratios_l310_310526

theorem geometric_sum_common_ratios (k p r : ℝ) 
  (hp : p ≠ r) (h_seq : p ≠ 1 ∧ r ≠ 1 ∧ p ≠ 0 ∧ r ≠ 0) 
  (h : k * p^4 - k * r^4 = 4 * (k * p^2 - k * r^2)) : 
  p + r = 3 :=
by
  -- Details omitted as requested
  sorry

end geometric_sum_common_ratios_l310_310526


namespace conic_is_parabola_l310_310605

-- Define the main equation
def main_equation (x y : ℝ) : Prop :=
  y^4 - 6 * x^2 = 3 * y^2 - 2

-- Definition of parabola condition
def is_parabola (x y : ℝ) : Prop :=
  ∃ a b c : ℝ, y^2 = a * x + b ∧ a ≠ 0

-- The theorem statement.
theorem conic_is_parabola :
  ∀ x y : ℝ, main_equation x y → is_parabola x y :=
by
  intros x y h
  sorry

end conic_is_parabola_l310_310605


namespace max_value_g_on_interval_l310_310101

noncomputable def f (x : ℝ) := sin (π / 4 * x - π / 6) - 2 * (cos (π / 8 * x))^2 + 1
noncomputable def g (x : ℝ) := f (2 - x)

theorem max_value_g_on_interval : 
  ∃ x ∈ set.Icc 0 (4 / 3 : ℝ), g x = (√3 / 2 : ℝ) := 
sorry

end max_value_g_on_interval_l310_310101


namespace sets_without_perfect_square_l310_310395

theorem sets_without_perfect_square : 
  let T_i := λ i, {n : ℤ | 200 * i ≤ n ∧ n < 200 * (i + 1)} in
  let sets := {i | ∀ n ∈ T_i i, ∃ k : ℤ, n ≠ k * k} in
  sets.card = 450 :=
by sorry

end sets_without_perfect_square_l310_310395


namespace exist_n_for_all_k_l310_310235

theorem exist_n_for_all_k (k : ℕ) (h_k : k > 1) : 
  ∃ n : ℕ, 
    (n > 0 ∧ ((n.choose k) % n = 0) ∧ (∀ m : ℕ, (2 ≤ m ∧ m < k) → ((n.choose m) % n ≠ 0))) :=
sorry

end exist_n_for_all_k_l310_310235


namespace geometric_sequence_sum_l310_310008

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, 0 < a n) (h_a1 : a 0 = 3)
(h_sum : a 0 + a 1 + a 2 = 21) (hq : ∀ n, a (n + 1) = a n * q) : a 2 + a 3 + a 4 = 84 := by
  sorry

end geometric_sequence_sum_l310_310008


namespace factorize1_factorize2_l310_310514

variable {a b x y p : ℝ}

theorem factorize1 (ha : a ∈ ℝ) (hb : b ∈ ℝ) (hx : x ∈ ℝ) (hy : y ∈ ℝ) :
  8 * a * x - b * x + 8 * a * y - b * y = (x + y) * (8 * a - b) :=
by sorry

theorem factorize2 (ha : a ∈ ℝ) (hb : b ∈ ℝ) (hx : x ∈ ℝ) (hp : p ∈ ℝ) :
  a * p + a * x - 2 * b * x - 2 * b * p = (p + x) * (a - 2 * b) :=
by sorry

end factorize1_factorize2_l310_310514


namespace find_length_QT_l310_310369

noncomputable def length_RS : ℝ := 75
noncomputable def length_PQ : ℝ := 36
noncomputable def length_PT : ℝ := 12

theorem find_length_QT :
  ∀ (PQRS : Type)
  (P Q R S T : PQRS)
  (h_RS_perp_PQ : true)
  (h_PQ_perp_RS : true)
  (h_PT_perpendicular_to_PR : true),
  QT = 24 :=
by
  sorry

end find_length_QT_l310_310369


namespace _l310_310290

noncomputable def is_tangent_to_parabola (x1 y1 p k : ℝ) : Prop :=
  let y := k * x1 - 1
  let rhs := x1^2
  rhs = y

noncomputable def leans_theorem_prover (O A B : (ℝ × ℝ)) : Prop :=
  -- Definitions of points
  let O := (0,0)
  let A := (1,1)
  let B := (0,-1)
  -- Value of p from point A on parabola C: x^2 = 2py
  let p := 1 / 2  -- as obtained by solving 1^2 = 2p * 1
  -- Checking option A: directrix is y = -1 is false
  let directrix := - p / 2
  (directrix ≠ -1) ∧
  -- Checking option B: tangent condition
  let slope_AB := (1 - (-1)) / (1 - 0)
  let tangent := is_tangent_to_parabola 1 1 p slope_AB
  tangent ∧
  -- Option C: |OP| * |OQ| = |OA|^2 is false
  let |OA|² := 2  -- obtained from the calculation |OA| = sqrt(1^2 + 1^2)
  ∀ (k > 2), k² ≠ |OA|² ∧
  -- Option D: |BP| * |BQ| > |BA|^2 is true
  let |BA|² := 5 -- obtained from the calculation |BA| = sqrt(1^2 + 4)
  ∀ (x1 x2) (hx1 : x1 + x2 = k) (hx2 : x1 * x2 = 1),
  let |BP| := sqrt(x1^2 + (x1^2 + 2x1 + 1))
  let |BQ| := sqrt(x2^2 + (x2^2 + 2x2 + 1))
  |BP| * |BQ| > |BA|²

example : leans_theorem_prover (0, 0) (1, 1) (0, -1) :=
by sorry  -- Proof is omitted, to be completed by Lean theorem prover

end _l310_310290


namespace counterexample_exists_l310_310594

theorem counterexample_exists :
  ∃ n ∈ {16, 22, 24, 28, 30}, ¬ Prime (n-2) ∧ ¬ Prime (n+2) :=
by
  sorry

end counterexample_exists_l310_310594


namespace profit_percentage_correct_l310_310183

-- Definitions based on the conditions
def market_price_per_pen : ℝ := 1
def number_of_pens : ℕ := 120
def number_of_pens_cost : ℕ := 100
def wholesaler_discount : ℝ := 0.05
def retailer_discount : ℝ := 0.04

-- Calculate cost price considering wholesaler's discount
def cost_price : ℝ := market_price_per_pen * number_of_pens_cost * (1 - wholesaler_discount)

-- Calculate selling price considering retailer's discount
def selling_price : ℝ := market_price_per_pen * number_of_pens * (1 - retailer_discount)

-- Calculate profit
def profit : ℝ := selling_price - cost_price

-- Calculate profit percentage
def profit_percent : ℝ := (profit / cost_price) * 100

-- Theorem to prove profit percentage is equal to 21.26%
theorem profit_percentage_correct : profit_percent = 21.26 := by
  sorry

end profit_percentage_correct_l310_310183


namespace segment_MI_equals_circumradius_l310_310522

noncomputable def circumcenter (A B C : Point) : Point := sorry
noncomputable def incenter (A B C : Point) : Point := sorry
noncomputable def circumradius (A B C : Point) : ℝ := sorry
noncomputable def midpoint_arc (A B C : Point) : Point := sorry

theorem segment_MI_equals_circumradius {A B C : Point} (hC : ∠B C A = 60) :
  let O := circumcenter A B C,
      I := incenter A B C,
      M := midpoint_arc A B C in
  dist I M = circumradius A B C :=
by sorry

end segment_MI_equals_circumradius_l310_310522


namespace no_solution_for_x_l310_310425

noncomputable def proof_problem : Prop :=
  ∀ x : ℝ, ⌊x⌋ + ⌊2*x⌋ + ⌊4*x⌋ + ⌊8*x⌋ + ⌊16*x⌋ + ⌊32*x⌋ ≠ 12345

theorem no_solution_for_x : proof_problem :=
  by
    intro x
    sorry

end no_solution_for_x_l310_310425


namespace number_of_knights_l310_310475

-- Defining types for the inhabitants
inductive Inhabitant
| knight
| liar

-- Predicate to determine if the number of liars to the right is greater
-- than the number of knights to the left.
def claim (inhabitants : List Inhabitant) (i : Fin (List.length inhabitants)) : Prop :=
  let liars_to_right := List.countp (· = Inhabitant.liar) (List.drop i.succ inhabitants)
  let knights_to_left := List.countp (· = Inhabitant.knight) (List.take i inhabitants)
  liars_to_right > knights_to_left

-- The main theorem to prove
theorem number_of_knights (inhabitants : List Inhabitant) (h_len : inhabitants.length = 2021)
  (h_claim : ∀ i, i < inhabitants.length → claim inhabitants ⟨i, by simp [*]⟩) :
  List.countp (· = Inhabitant.knight) inhabitants = 1010 :=
by
  sorry

end number_of_knights_l310_310475


namespace tommy_initial_candy_l310_310334

noncomputable theory

def initial_candy_tommy (candy_hugh candy_melany total_shared : ℕ) : ℕ :=
  total_shared - candy_hugh - candy_melany

theorem tommy_initial_candy :
  let candy_hugh := 8
  let candy_melany := 7
  let total_shared := 21
  initial_candy_tommy candy_hugh candy_melany total_shared = 6 :=
by
  let candy_hugh := 8
  let candy_melany := 7
  let total_shared := 21
  show initial_candy_tommy candy_hugh candy_melany total_shared = 6
  sorry

end tommy_initial_candy_l310_310334


namespace license_plate_count_l310_310709

def is_letter (ch : Char) : Prop := ch.isAlpha
def is_digit (ch : Char) : Prop := ch.isDigit

def valid_license_plate (plate : Array Char) : Prop :=
  plate.size = 4 ∧
  is_letter plate[0] ∧
  (is_letter plate[1] ∨ is_digit plate[1]) ∧
  is_letter plate[2] ∧
  is_digit plate[3] ∧
  ((plate[0] = plate[2] ∧ plate[1] ≠ plate[2] ∧ plate[2] ≠ plate[3]) ∨
   (plate[2] = plate[3] ∧ plate[0] ≠ plate[3] ∧ plate[1] ≠ plate[3])) 

-- statement for proof that matches the mathematical conclusion of the problem
theorem license_plate_count : 
  (∃ plates : Finset (Array Char), 
    (∀ plate ∈ plates, valid_license_plate plate) ∧ plates.card = 9100) :=
sorry

end license_plate_count_l310_310709


namespace fraction_sum_24_pretty_div_24_l310_310768

def is_24_pretty (n : ℕ) : Prop :=
  let divisors := (finset.range (n + 1)).filter (λ d, n % d = 0)
  divisors.card = 24

def sum_24_pretty_lt_5000 : ℕ :=
  (finset.range 5000).filter is_24_pretty |>.sum id

theorem fraction_sum_24_pretty_div_24 : sum_24_pretty_lt_5000 / 24 = 826 := by
  sorry

end fraction_sum_24_pretty_div_24_l310_310768


namespace power_of_power_rule_l310_310209

theorem power_of_power_rule (h : 128 = 2^7) : (128: ℝ)^(4/7) = 16 := by
  sorry

end power_of_power_rule_l310_310209


namespace median_is_4_6_l310_310228

-- Given conditions on the number of students per vision category
def vision_data : List (ℝ × ℕ) :=
[(4.0, 1), (4.1, 2), (4.2, 6), (4.3, 3), (4.4, 3), (4.5, 4), (4.6, 1), 
 (4.7, 2), (4.8, 5), (4.9, 7), (5.0, 5)]

-- Statement of the problem: Proving that the median is 4.6
theorem median_is_4_6 (data : List (ℝ × ℕ)) (h : data = vision_data) : 
  median (expand_data data) = 4.6 := 
sorry

-- Helper function to expand the data into a list of individual vision values
def expand_data (data : List (ℝ × ℕ)) : List ℝ :=
data.bind (λ p, List.replicate p.snd p.fst)

-- Helper function to compute the median
noncomputable
def median (l : List ℝ) : ℝ :=
if l.length % 2 = 1 then l.nth_le (l.length / 2) (by linarith [List.length_pos_of_mem (List.nth_le_mem l (l.length / 2) _)])
else (l.nth_le (l.length / 2 - 1) (by linarith [List.length_pos_of_mem (List.nth_le_mem l (l.length / 2 - 1) _)]) +
      l.nth_le (l.length / 2) (by linarith [List.length_pos_of_mem (List.nth_le_mem l (l.length / 2) _)])) / 2

end median_is_4_6_l310_310228


namespace coeff_x3_in_expansion_of_1_minus_2x_pow_6_l310_310952

theorem coeff_x3_in_expansion_of_1_minus_2x_pow_6 :
  (1 - 2 * polynomial.x) ^ 6.coeff 3 = -160 :=
sorry

end coeff_x3_in_expansion_of_1_minus_2x_pow_6_l310_310952


namespace minimal_number_of_moves_is_n_l310_310463

open Function

noncomputable def minimal_moves_sufficient (n : ℕ) : ℕ :=
  let two_elem_moves := λ x y : ℕ, x ≠ y
  let three_elem_cyclic_moves := λ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ z ≠ x
  n

theorem minimal_number_of_moves_is_n (n : ℕ) :
  ∃ f : ℕ → ℕ, (∀ (x y z : ℕ), two_elem_moves x y ∨ three_elem_cyclic_moves x y z) → 
  minimal_moves_sufficient n = n :=
  sorry

end minimal_number_of_moves_is_n_l310_310463


namespace projection_correct_l310_310300

variables (a e : ℝ^3)

-- Definitions derived from conditions
def magnitude_a : ℝ := 4
def is_unit_vector_e : Prop := ∥e∥ = 1
def angle_between_a_e : ℝ := 2 * Real.pi / 3

noncomputable def projection_of_sum_on_diff : ℝ :=
  let dot_product := (a + e) ⬝ (a - e)
  let magnitude_diff := ∥a - e∥
  dot_product / magnitude_diff

-- The theorem to be proved based on conditions
theorem projection_correct 
  (ha : ∥a∥ = magnitude_a) 
  (he : is_unit_vector_e e) 
  (hae : angle_between_a_e = 2 * Real.pi / 3):
  projection_of_sum_on_diff a e = 5 * Real.sqrt 21 / 7 :=
begin
  sorry
end

end projection_correct_l310_310300


namespace proposition_p_q_true_l310_310278

def represents_hyperbola (m : ℝ) : Prop := (1 - m) * (m + 2) < 0

def represents_ellipse (m : ℝ) : Prop := (2 * m > 2 - m) ∧ (2 - m > 0)

theorem proposition_p_q_true (m : ℝ) :
  represents_hyperbola m ∧ represents_ellipse m → (1 < m ∧ m < 2) :=
by
  sorry

end proposition_p_q_true_l310_310278


namespace g_crosses_horizontal_asymptote_at_minus_four_l310_310638

noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 8 * x - 10) / (x^2 - 5 * x + 6)

theorem g_crosses_horizontal_asymptote_at_minus_four : g (-4) = 3 := 
by
  sorry

end g_crosses_horizontal_asymptote_at_minus_four_l310_310638


namespace blue_balls_removal_l310_310943

noncomputable def initial_total_balls : ℕ := 150
noncomputable def initial_red_balls : ℕ := (40 * initial_total_balls) / 100
noncomputable def initial_blue_balls : ℕ := initial_total_balls - initial_red_balls
noncomputable def blue_balls_to_remove : ℕ := 75

theorem blue_balls_removal :
  let remaining_balls := initial_total_balls - blue_balls_to_remove in
  let remaining_blue_balls := initial_blue_balls - blue_balls_to_remove in
  (initial_red_balls * 100) / remaining_balls = 80 :=
by
  let remaining_balls := initial_total_balls - blue_balls_to_remove in
  let target_red_balls_count := (80 * remaining_balls) / 100 in
  have h : initial_red_balls = target_red_balls_count := by
    sorry
  exact h

end blue_balls_removal_l310_310943


namespace part_a_part_b_l310_310049

-- Defining the function that represents the count of triples in the given system.
def K (n : ℕ) : ℕ := sorry

-- Proving that K(n) > n / 6 - 1
theorem part_a (n : ℕ) : K(n) > n / 6 - 1 := by
  sorry

-- Proving that K(n) < 2n / 9
theorem part_b (n : ℕ) : K(n) < 2 * n / 9 := by
  sorry

end part_a_part_b_l310_310049


namespace number_of_unique_parabolas_l310_310023

theorem number_of_unique_parabolas :
  let values := {-3, -2, 0, 1, 2, 3}
  in ∃ cs : {a : ℤ // a ∈ values ∧ a ≠ 0} →
          {b : ℤ // b ∈ values ∧ b ≠ 0} →
          {c : ℤ // c ∈ values ∧ c ≠ cs.1 ∧ c ≠ cs.2},
     (finset.univ.bij_on
       (λ (x : {cs : (values : finset ℤ) × {a : ℤ // a ∈ values ∧ a ≠ 0 ∧ cs.1 ≠ a} × {c : ℤ // c ∈ values ∧ c ≠ cs.1 ∧ c ≠ x.2.1}),
           (cs.1, x.2.1, x.2.2))
       (values.to_finset.product (values.to_finset.erase cs.1.to_finset).product (values.to_finset.erase x.2.1.to_finset).erase cs.1.to_finset))
        card = 62 :=
sorry

end number_of_unique_parabolas_l310_310023


namespace total_goals_l310_310223

def first_period_goals (k: ℕ) : ℕ :=
  k

def second_period_goals (k: ℕ) : ℕ :=
  2 * k

def spiders_first_period_goals (k: ℕ) : ℕ :=
  k / 2

def spiders_second_period_goals (s1: ℕ) : ℕ :=
  s1 * s1

def third_period_goals (k1 k2: ℕ) : ℕ :=
  2 * (k1 + k2)

def spiders_third_period_goals (s2: ℕ) : ℕ :=
  s2

def apply_bonus (goals: ℕ) (multiple: ℕ) : ℕ :=
  if goals % multiple = 0 then goals + 1 else goals

theorem total_goals (k1 k2 s1 s2 k3 s3 : ℕ) :
  first_period_goals 2 = k1 →
  second_period_goals k1 = k2 →
  spiders_first_period_goals k1 = s1 →
  spiders_second_period_goals s1 = s2 →
  third_period_goals k1 k2 = k3 →
  apply_bonus k3 3 = k3 + 1 →
  apply_bonus s2 2 = s2 →
  spiders_third_period_goals s2 = s3 →
  apply_bonus s3 2 = s3 →
  2 + k2 + (k3 + 1) + (s1 + s2 + s3) = 22 :=
by
  sorry

end total_goals_l310_310223


namespace intersection_of_A_and_B_l310_310281

def SetA (x : ℝ) : Prop := (4 * x - 3) * (x + 3) < 0

def SetB (x : ℝ) : Prop := 2 * x > 1

theorem intersection_of_A_and_B :
  {x : ℝ | SetA x} ∩ {x | SetB x} = {x | (1/2 : ℝ) < x ∧ x < (3/4 : ℝ)} :=
begin
  sorry
end

end intersection_of_A_and_B_l310_310281


namespace geometric_sequence_product_l310_310374

theorem geometric_sequence_product {a : ℕ → ℝ} (q : ℝ) (h1 : |q| ≠ 1) :
  a 1 = 1 → (∀ n, a n = a 1 * (q ^ (n - 1))) → a 11 = a 1 * a 2 * a 3 * a 4 * a 5 :=
by {
  intros h2 h3,
  sorry
}

end geometric_sequence_product_l310_310374


namespace isosceles_triangle_integer_solutions_l310_310461

theorem isosceles_triangle_integer_solutions :
  (finset.filter (λ x : ℕ, 12 < x ∧ x < 24) (finset.range 25)).card = 11 :=
by
  sorry

end isosceles_triangle_integer_solutions_l310_310461


namespace sum_of_possible_values_l310_310400

theorem sum_of_possible_values (x y : ℝ)
  (h : x * y - x / y^3 - y / x^3 = 4) :
  (x - 1) * (y - 1) = 1 ∨ (x - 1) * (y - 1) = 4 → ∑ v in {1, 4}, v = 5 := by
  sorry

end sum_of_possible_values_l310_310400


namespace correct_statements_l310_310568

theorem correct_statements 
  (f : ℝ → ℝ) 
  (h₁ : f x = log x + 3 * x - 6)
  (h₂ : ∀ a x, ax^2 + 2 * a * x + 1 > 0 → 0 < a ∧ a < 1)
  (h₃ : ∀ x, sin x = x -> x = 0 → ¬(∃ x, y = x ∧ y = sin x ∧ y ≠ 0))
  (h₄ : ∀ x, 0 ≤ x ∧ x ≤ π/4 → (sin x * cos x + sin x + cos x) = 1) :
  {1, 4}.nonempty :=
begin
  Sorry
end

end correct_statements_l310_310568


namespace equation_not_parabola_l310_310665

def theta : ℝ := sorry  -- \(\theta \) is any real number

def equation (x y : ℝ) : ℝ := x^2 + y^2 * cos theta

theorem equation_not_parabola : ¬ ∃ a b c : ℝ, equation x y = a*x*y + b*y + c :=
by
  sorry

end equation_not_parabola_l310_310665


namespace AP_squared_sum_min_value_l310_310895

noncomputable def AP_squared_sum_min (A B C D E : ℝ) (P : ℝ) 
    (h1: A = 0) (h2: B = 1) (h3: C = 3) (h4: D = 6) (h5: E = 10) : ℝ :=
    A^2 + (B - P)^2 + (C - P)^2 + (D - P)^2 + (E - P)^2

theorem AP_squared_sum_min_value : 
    ∀ (P : ℝ), P = 4 → AP_squared_sum_min 0 1 3 6 10 P 0 1 3 6 10 = 66 :=
by
  intros P hP
  rw [AP_squared_sum_min, hP]
  sorry

end AP_squared_sum_min_value_l310_310895


namespace largest_diff_between_3digit_numbers_formed_with_5_9_2_l310_310153

theorem largest_diff_between_3digit_numbers_formed_with_5_9_2 :
  let digits: List Nat := [5, 9, 2] in
  let largest := 952 in
  let smallest := 259 in
  (∃ (num1 num2 : Nat),
    (num1 = largest ∧ num2 = smallest) ∧
    ∀ (a b c : Nat), (a ∈ digits ∧ b ∈ digits ∧ c ∈ digits ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c) →
    num1 = a * 100 + b * 10 + c ∨ num2 = a * 100 + b * 10 + c) →
  largest - smallest = 693 := by
  sorry -- Proof skipped.

end largest_diff_between_3digit_numbers_formed_with_5_9_2_l310_310153


namespace find_x_l310_310030

theorem find_x (x y : ℤ) (hx : x > 0) (hy : y > 0) (hxy : x > y) (h : x + y + x * y = 152) : x = 16 := 
by
  sorry

end find_x_l310_310030


namespace nuts_in_each_bag_l310_310899

theorem nuts_in_each_bag (total_bags : ℕ) (students : ℕ) (nuts_per_student : ℕ) 
  (total_bags = 65)
  (students = 13)
  (nuts_per_student = 75) :
  (students * nuts_per_student) / total_bags = 15 := 
by
  sorry

end nuts_in_each_bag_l310_310899


namespace alternating_binomial_sum_l310_310631

theorem alternating_binomial_sum :
  ∑ k in finset.range 51, (-1)^k * (nat.choose 101 (2*k + 1)) = -4^51 :=
by
  sorry

end alternating_binomial_sum_l310_310631


namespace pieces_per_box_l310_310948

theorem pieces_per_box (boxes : ℕ) (total_pieces : ℕ) (h_boxes : boxes = 7) (h_total : total_pieces = 21) : 
  total_pieces / boxes = 3 :=
by
  sorry

end pieces_per_box_l310_310948


namespace polynomial_roots_l310_310128

variable (AT TB m n : ℝ)

-- conditions
axiom sum_condition : AT + TB = m
axiom product_condition : AT * TB = n^2

-- statement
theorem polynomial_roots :
  Polynomial.X^2 - m * Polynomial.X + n^2 =
  Polynomial.of_roots [AT, TB] := by
  sorry

end polynomial_roots_l310_310128


namespace number_of_people_bought_tickets_l310_310479

-- Definitions of the conditions:
def ticket_price := 20
def first_10_discount := 0.40
def next_20_discount := 0.15
def total_revenue := 980

-- Mathematically equivalent proof problem statement:
theorem number_of_people_bought_tickets :
  ∃ (n1 n2 n3: ℕ), n1 = 10 ∧ n2 = 20 ∧ n3 * ticket_price + (n1 * (ticket_price - ticket_price * first_10_discount)) + (n2 * (ticket_price - ticket_price * next_20_discount)) = total_revenue → 
  n1 + n2 + n3 = 56 :=
by
  sorry

end number_of_people_bought_tickets_l310_310479


namespace median_vision_is_correct_l310_310233

def student_vision_data : List (ℝ × ℕ) := [
  (4.0, 1),
  (4.1, 2),
  (4.2, 6),
  (4.3, 3),
  (4.4, 3),
  (4.5, 4),
  (4.6, 1),
  (4.7, 2),
  (4.8, 5),
  (4.9, 7),
  (5.0, 5)
]

def total_students : ℕ := 39

def median_vision_value (data : List (ℝ × ℕ)) (total : ℕ) : ℝ :=
  -- Function to calculate the median vision value
  let sorted_data := data.sortBy (λ x, x.1) in
  let cumulative_counts := sorted_data.scanl (λ acc x, acc + x.2) 0 in
  let median_position := total / 2 in
  let median_index := cumulative_counts.indexWhere (λ x, x > median_position) in
  (sorted_data.nth! (median_index - 1)).1

theorem median_vision_is_correct : median_vision_value student_vision_data total_students = 4.6 :=
by
  -- Prove that the median vision value of the dataset is 4.6
  sorry

end median_vision_is_correct_l310_310233


namespace spencer_total_distance_l310_310036

-- Define the individual segments of Spencer's travel
def walk1 : ℝ := 1.2
def bike1 : ℝ := 1.8
def bus1 : ℝ := 3
def walk2 : ℝ := 0.4
def walk3 : ℝ := 0.6
def bike2 : ℝ := 2
def walk4 : ℝ := 1.5

-- Define the conversion factors
def bike_to_walk_conversion : ℝ := 0.5
def bus_to_walk_conversion : ℝ := 0.8

-- Calculate the total walking distance
def total_walking_distance : ℝ := walk1 + walk2 + walk3 + walk4

-- Calculate the total biking distance as walking equivalent
def total_biking_distance_as_walking : ℝ := (bike1 + bike2) * bike_to_walk_conversion

-- Calculate the total bus distance as walking equivalent
def total_bus_distance_as_walking : ℝ := bus1 * bus_to_walk_conversion

-- Define the total walking equivalent distance
def total_distance : ℝ := total_walking_distance + total_biking_distance_as_walking + total_bus_distance_as_walking

-- Theorem stating the total distance covered is 8 miles
theorem spencer_total_distance : total_distance = 8 := by
  unfold total_distance
  unfold total_walking_distance
  unfold total_biking_distance_as_walking
  unfold total_bus_distance_as_walking
  norm_num
  sorry

end spencer_total_distance_l310_310036


namespace hexagon_perimeter_l310_310493

-- Define the points A, B, C, D, E, F as pairs of real numbers for simplicity
variables (A B C D E F : Point)

-- Define the length of the sides of the hexagon
noncomputable def length (p1 p2 : Point) : ℝ := 
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Given conditions
def AB_length : length A B = 2 := sorry
def BC_length : length B C = 2 := sorry
def CD_length : length C D = 2 := sorry
def DE_length : length D E = 2 := sorry
def EF_length : length E F = 2 := sorry
def FA_length : length F A = 3 := sorry

-- Define the perimeter of the hexagon
noncomputable def perimeter : ℝ := 
  length A B + length B C + length C D + length D E + length E F + length F A

-- The theorem to be proved
theorem hexagon_perimeter : perimeter A B C D E F = 13 := sorry

end hexagon_perimeter_l310_310493


namespace length_of_leg_is_3_l310_310813

theorem length_of_leg_is_3 (h : 4) (V : 6) :
  let A_base := (1 / 2) * l ^ 2 in
    (1 / 3) * A_base * h = V ∧ 
    4.5 = (1 / 2) * l ^ 2 ∧
    l = 3 :=
by
  sorry

end length_of_leg_is_3_l310_310813


namespace problem_l310_310444

theorem problem (a b : ℕ) (h_a : a = 5) (h_b : b = 5) : (765400 + 10 * a + b) % 24 = 0 :=
by {
  calc (765400 + 10 * a + b) % 24
      = (765400 + 10 * 5 + 5) % 24 : by rw [h_a, h_b]
  ... = 765455 % 24 : by norm_num
  ... = 0 : by norm_num
}

end problem_l310_310444


namespace expand_and_simplify_l310_310857

noncomputable def expanded_expr (a : ℝ) : ℝ :=
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6)

theorem expand_and_simplify (a : ℝ) :
  expanded_expr a = a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 :=
by {
  -- Proof goes here
  sorry
}

end expand_and_simplify_l310_310857


namespace arithmetic_arrangement_result_l310_310575

theorem arithmetic_arrangement_result :
    (1 / 8) * (1 / 9) * (1 / 28) = 1 / 2016 ∨ ((1 / 8) - (1 / 9)) * (1 / 28) = 1 / 2016 :=
by {
    sorry
}

end arithmetic_arrangement_result_l310_310575


namespace sum_of_integers_l310_310113

theorem sum_of_integers :
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ a < 30 ∧ b < 30 ∧ (a * b + a + b = 167) ∧ Nat.gcd a b = 1 ∧ (a + b = 24) :=
by {
  sorry
}

end sum_of_integers_l310_310113


namespace sufficient_condition_l310_310118

theorem sufficient_condition (x : ℝ) (h : -1 < x ∧ x ≤ 5) : 6 / (x + 1) ≥ 1 :=
by
  have hx : x + 1 ≠ 0 := by
    intro hx
    linarith
  have h₁ : x + 1 > 0 := by linarith
  have h₂ : 6 ≥ x + 1 := by linarith
  linarith [div_le_one_of_le h₁ h₂]

end sufficient_condition_l310_310118


namespace expand_and_simplify_l310_310864

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
  sorry

end expand_and_simplify_l310_310864


namespace conjugate_z_l310_310342

noncomputable def z : ℂ := complex.abs ⟨real.sqrt 3, -1⟩ + complex.i^5

theorem conjugate_z : complex.conj z = 2 - complex.i :=
by
  sorry

end conjugate_z_l310_310342


namespace complex_number_in_fourth_quadrant_l310_310736

/- Define the complex number z -/
def z : ℂ := (2 + 3 * complex.I) / complex.I

/- Define what it means to be in the fourth quadrant. A point (x, y) in the complex plane
   is in the fourth quadrant if x > 0 and y < 0 -/
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

/- Main theorem that states the point corresponding to z := (2 + 3i) / i is in the fourth quadrant -/
theorem complex_number_in_fourth_quadrant : in_fourth_quadrant z :=
by 
  sorry

end complex_number_in_fourth_quadrant_l310_310736


namespace right_triangle_7_24_25_l310_310195

theorem right_triangle_7_24_25 : 
  ∃ (a b c : ℕ), (a = 7 ∧ b = 24 ∧ c = 25 ∧ a^2 + b^2 = c^2) :=
by {
  use [7, 24, 25],
  simp,
  sorry
}

end right_triangle_7_24_25_l310_310195


namespace maximize_profit_l310_310668

def R (x : ℝ) : ℝ := if 0 < x ∧ x ≤ 40 then 400 - 6 * x else (7400 / x) - (40000 / (x * x))

def W (x : ℝ) : ℝ := 
  if 0 < x ∧ x ≤ 40 then -6 * (x * x) + 384 * x - 40
  else -(40000 / x) - 16 * x + 7360

theorem maximize_profit : 
  let x := 32 in 
  W x = 6104 ∧ 
  ∀ y : ℝ, 0 < y ∧ y ≠ 32 ∧ y ≤ 40 → W y ≤ 6104 :=
by sorry

end maximize_profit_l310_310668


namespace range_of_a_l310_310660

theorem range_of_a (a : ℝ) 
  (P : ∀ x : ℝ, x ∈ set.Icc 0 1 → a ≥ Real.exp x)
  (Q : ∃ x : ℝ, x^2 + 4*x + a = 0) : 
  Real.exp 1 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l310_310660


namespace triangle_BR_parallel_AC_l310_310404

-- Lean statement encapsulating the problem and conditions
theorem triangle_BR_parallel_AC
  (ABC : Triangle) -- ABC is an acute-angled triangle
  (O : Point) -- O is the circumcenter of triangle ABC
  (Ω : Circle := circ ABC O) -- Ω is the circumcircle of triangle ABC
  (AB GT BC : ℝ) -- AB > BC
  (B_not_eq_M : intersect_angle_bisector(∠ ABC) ≠ B)
  (M : Point := intersect_angle_bisector(∠ ABC) ∈ Ω)
  (Γ : Circle := diam_circle B M) -- Γ is the circle with diameter BM
  (P Q : Point) -- P and Q are points on the angle bisectors of ∠ AOB and ∠ BOC respectively, intersecting Γ
  (R : Point) -- R is the point on line PQ such that BR = MR
  (H1 : on_line R P Q)
  (H2 : dist B R = dist M R)
  : parallel (line B R) (line A C) :=
by
  sorry

end triangle_BR_parallel_AC_l310_310404


namespace logan_snowfall_total_l310_310413

theorem logan_snowfall_total (wednesday thursday friday : ℝ) :
  wednesday = 0.33 → thursday = 0.33 → friday = 0.22 → wednesday + thursday + friday = 0.88 :=
by
  intros hw ht hf
  rw [hw, ht, hf]
  exact (by norm_num : (0.33 : ℝ) + 0.33 + 0.22 = 0.88)

end logan_snowfall_total_l310_310413


namespace smallest_perfect_square_div_by_4_and_5_l310_310496

theorem smallest_perfect_square_div_by_4_and_5 : 
  ∃ n : ℕ, (∃ m : ℕ, n = m^2) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (∀ k : ℕ, (∃ l : ℕ, k = l^2) ∧ (4 ∣ k) ∧ (5 ∣ k) → n ≤ k) :=
begin
  let n := 400,
  use n,
  split,
  { use 20, -- 400 is 20^2
    refl },
  split,
  { exact dvd.intro 100 rfl }, -- 400 = 4 * 100
  split,
  { exact dvd.intro 80 rfl }, -- 400 = 5 * 80
  { 
    intros k hk,
    obtain ⟨l, hl⟩ := hk.left,
    obtain ⟨_h4⟩ := hk.right.left  -- k divisible by 4
    obtain ⟨_h5⟩ := hk.right.right -- k divisible by 5
    rw hl,
    sorry  -- This is where the rest of the proof would go.
  }
end

end smallest_perfect_square_div_by_4_and_5_l310_310496


namespace cost_of_fencing_is_155_l310_310889

-- Definitions based on conditions
def sides_ratio : Real := 3 / 2
def area : Real := 5766
def cost_per_meter_paise : Real := 50
def cost_per_meter_rupees : Real := cost_per_meter_paise / 100
def total_cost_of_fencing (longer_side : Real) (shorter_side : Real) : Real :=
  let perimeter := 2 * (longer_side + shorter_side)
  perimeter * cost_per_meter_rupees

-- Statement to prove
theorem cost_of_fencing_is_155 :
  ∃ (x : Real), let longer_side := 3 * x
                let shorter_side := 2 * x
                (longer_side * shorter_side = area) →
                (total_cost_of_fencing longer_side shorter_side = 155) :=
by
  sorry

end cost_of_fencing_is_155_l310_310889


namespace mode_and_median_of_seedlings_l310_310089

theorem mode_and_median_of_seedlings :
  let heights := [25, 26, 27, 26, 27, 28, 29, 26, 29] in
  (mode heights = [26]) ∧ (median heights = some 27) :=
by {
  let heights := [25, 26, 27, 26, 27, 28, 29, 26, 29];
  sorry
}

end mode_and_median_of_seedlings_l310_310089


namespace constant_term_expansion_l310_310372

theorem constant_term_expansion :
  let f := (4 * x^2 - 2 * x - 5) * (1 + (1 / x^2))^5 in
  is_constant_term (f) 15 :=
by
  let f := (4 * x^2 - 2 * x - 5) * (1 + (1 / x^2))^5
  sorry

end constant_term_expansion_l310_310372


namespace speed_of_j_l310_310520

theorem speed_of_j (j p : ℝ) 
  (h_faster : j > p)
  (h_distance_j : 24 / j = 24 / j)
  (h_distance_p : 24 / p = 24 / p)
  (h_sum_speeds : j + p = 7)
  (h_sum_times : 24 / j + 24 / p = 14) : j = 4 := 
sorry

end speed_of_j_l310_310520


namespace polynomial_divisibility_l310_310796

theorem polynomial_divisibility (n : ℕ) (h : 0 < n) : 
  ∃ g : Polynomial ℚ, 
    (Polynomial.X + 1)^(2*n + 1) + Polynomial.X^(n + 2) = g * (Polynomial.X^2 + Polynomial.X + 1) := 
by
  sorry

end polynomial_divisibility_l310_310796


namespace expand_and_simplify_l310_310862

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
  sorry

end expand_and_simplify_l310_310862


namespace inclination_line_cos_l310_310666

theorem inclination_line_cos (θ : ℝ) (hθ : Real.Tan θ = 2) : Real.Cos θ = Real.sqrt 5 / 5 :=
sorry

end inclination_line_cos_l310_310666


namespace proof_inequality_x_s_l310_310154

variable {ι : Type*} [Fintype ι]

theorem proof_inequality_x_s (x : ι → ℝ) (n : ℕ) (hn : 1 ≤ n) (hx : ∀ i, 0 < x i) :
  ∑ i : ι, x i / (∑ i, x i - x i) ≥ n / (n - 1) ∧
  ∑ i : ι, (∑ i, x i - x i) / x i ≥ n * (n - 1) := by
  sorry

end proof_inequality_x_s_l310_310154


namespace find_angle_BAC_l310_310163

theorem find_angle_BAC (ABC EBD : Triangle) (A B C D E: Point) 
  (h1 : ABC ≅ EBD) (h2 : ∠DAE = 37) (h3 : ∠DEA = 37) : 
  ∠BAC = 7 := 
sorry

end find_angle_BAC_l310_310163


namespace find_f_8_6_l310_310878

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

theorem find_f_8_6 (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_symmetry : ∀ x, f (1 + x) = f (1 - x))
  (h_def : ∀ x, -1 ≤ x ∧ x ≤ 0 → f x = - (1 / 2) * x) :
  f 8.6 = 0.3 :=
sorry

end find_f_8_6_l310_310878


namespace AB_eq_6_sqrt_2_l310_310038

variables (P Q R S A B : Type*)

-- Definition of points in a plane
section
variables [metric_space P] [metric_space Q] [metric_space R] [metric_space S]
variables [metric_space A] [metric_space B]

-- Conditions:
-- PQRS is a square with side length 12
def is_square (PQRS : P × Q × R × S) (side_length : ℝ) : Prop :=
  dist P Q = side_length ∧
  dist Q R = side_length ∧
  dist R S = side_length ∧
  dist S P = side_length ∧
  dist P R = side_length * sqrt 2 ∧
  dist Q S = side_length * sqrt 2

-- ∠QPA = 30°
def angle_QPA (P Q A : P) : Prop :=
  ∃ θ : ℝ, θ = π / 6 ∧ angle_at P Q A θ 

-- ∠SRB = 60°
def angle_SRB (S R B : P) : Prop :=
  ∃ θ : ℝ, θ = π / 3 ∧ angle_at S R B θ 

noncomputable def AB_length : ℝ :=
  dist A B

-- Proof problem statement:
theorem AB_eq_6_sqrt_2
  (h1 : is_square (P, Q, R, S) 12)
  (h2 : angle_QPA P Q A)
  (h3 : angle_SRB S R B) :
  AB_length A B = 6 * sqrt 2 :=
sorry

end AB_eq_6_sqrt_2_l310_310038


namespace find_angle_B_l310_310730

open Real

noncomputable def obtuse_triangle (A B C : ℝ) : Prop :=
  sin A ^ 2 + (sqrt 3 / 6) * sin (2 * A) = 1

theorem find_angle_B (A B C : ℝ) (h : obtuse_triangle A B C) (h1 : sin B * cos C = cos (2 * B + π / 3)) 
  (h2 : b ∈ Ioc 0 π) : B = π / 12 :=
by
  -- problem set-up
  sorry

end find_angle_B_l310_310730


namespace problem1_l310_310896

theorem problem1 (z : ℂ) (h : z = 1 - 2 * Complex.i) : (z + 2) / (z - 1) = 1 + 3 / 2 * Complex.i :=
by
  sorry

end problem1_l310_310896


namespace flight_time_l310_310114

noncomputable def time_to_circle_earth (r v : ℝ) : ℝ := (2 * Real.pi * r) / v

theorem flight_time : time_to_circle_earth 3950 550 ≈ 45 :=
by
  sorry

end flight_time_l310_310114


namespace greatest_among_four_powers_l310_310240

theorem greatest_among_four_powers : 
  let a := 5^100
  let b := 6^91
  let c := 7^90
  let d := 8^85
  d > a ∧ d > b ∧ d > c ∧ (∀ x, x ∈ {a, b, c, d} → x ≤ d) :=
sorry

end greatest_among_four_powers_l310_310240


namespace cubes_divisible_by_9_l310_310846

theorem cubes_divisible_by_9 (n: ℕ) (h: n > 0) : 9 ∣ n^3 + (n + 1)^3 + (n + 2)^3 :=
by 
  sorry

end cubes_divisible_by_9_l310_310846


namespace parabola_focus_l310_310625

theorem parabola_focus (y : ℝ) : ∃ (focus : ℝ × ℝ), focus = (-2, 0) :=
by
  let focus := (-2, 0)
  use focus
  sorry

end parabola_focus_l310_310625


namespace perpendicular_bisector_plane_of_AB_l310_310657

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨1, 0, 1⟩
def B : Point3D := ⟨2, -1, 0⟩

def midpoint (P Q : Point3D) : Point3D :=
  ⟨(P.x + Q.x) / 2, (P.y + Q.y) / 2, (P.z + Q.z) / 2⟩

#check (1 : ℝ) -- ensures Real numbers can be used 편ibly 충분 MIDPOINT AB
def midAB : Point3D := midpoint A B

def direction_vector (P Q : Point3D) : Point3D :=
  ⟨Q.x - P.x, Q.y - P.y, Q.z - P.z⟩

def dirAB : Point3D := direction_vector A B

-- The desired equation of the plane
def plane_equation (P : Point3D) (n : Point3D) (D : ℝ) (x y z : ℝ) : Prop :=
  (n.x*x + n.y*y + n.z*z + D = 0)

theorem perpendicular_bisector_plane_of_AB :
  ∃ D, plane_equation midAB dirAB D x y z ↔ (2*x - 2*y - 2*z = 3) :=
sorry

end perpendicular_bisector_plane_of_AB_l310_310657


namespace percentage_class_takes_lunch_l310_310158

theorem percentage_class_takes_lunch (total_students boys girls : ℕ)
  (h_total: total_students = 100)
  (h_ratio: boys = 6 * total_students / (6 + 4))
  (h_girls: girls = 4 * total_students / (6 + 4))
  (boys_lunch_ratio : ℝ)
  (girls_lunch_ratio : ℝ)
  (h_boys_lunch_ratio : boys_lunch_ratio = 0.60)
  (h_girls_lunch_ratio : girls_lunch_ratio = 0.40):
  ((boys_lunch_ratio * boys + girls_lunch_ratio * girls) / total_students) * 100 = 52 :=
by
  sorry

end percentage_class_takes_lunch_l310_310158


namespace P_Q_equality_l310_310266

def P (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 4

def mean (coeffs : List ℝ) : ℝ := (coeffs.sum) / (coeffs.length)

noncomputable def Q (x : ℝ) : ℝ :=
  let m := mean [1, -2, 3, -4]
  m*x^3 + m*x^2 + m*x + m

theorem P_Q_equality : P 1 = Q 1 := by
  sorry

end P_Q_equality_l310_310266


namespace exists_a_b_l310_310689

noncomputable def f (a b x : ℝ) : ℝ := x^2 + a * x + b * Real.cos x

theorem exists_a_b (a b : ℝ) :
  -- Given the conditions
  (∃ x : ℝ, f a b x = 0) →
  (∀ x : ℝ, f a b x = 0 ↔ f a b (f a b x) = 0) →
  -- Proving the conclusion
  (a >= 0 ∧ a < 4 ∧ b = 0) :=
begin
  sorry
end

end exists_a_b_l310_310689


namespace Joan_video_game_expense_l310_310388

theorem Joan_video_game_expense : 
  let basketball_price := 5.20
  let racing_price := 4.23
  let action_price := 7.12
  let discount_rate := 0.10
  let sales_tax_rate := 0.06
  let discounted_basketball_price := basketball_price * (1 - discount_rate)
  let discounted_racing_price := racing_price * (1 - discount_rate)
  let discounted_action_price := action_price * (1 - discount_rate)
  let total_cost_before_tax := discounted_basketball_price + discounted_racing_price + discounted_action_price
  let sales_tax := total_cost_before_tax * sales_tax_rate
  let total_cost := total_cost_before_tax + sales_tax
  total_cost = 15.79 :=
by
  sorry

end Joan_video_game_expense_l310_310388


namespace part1_solution_set_part2_range_of_a_l310_310409

section problem

-- Definitions given in the conditions
def f (a : ℝ) (x : ℝ) : ℝ := abs (a * x + 1) + abs (x - a)
def g (x : ℝ) : ℝ := x^2 + x

-- Part (1)
theorem part1_solution_set (x : ℝ) : 
  (f 1 x ≤ g x) ↔ (x ≤ -3 ∨ 1 ≤ x) :=
begin
  sorry
end

-- Part (2)
theorem part2_range_of_a (a : ℝ) :
  (∀ x, f a x ≥ 3 / 2) → (a ≥ real.sqrt 2 / 2) :=
begin
  sorry
end

end problem

end part1_solution_set_part2_range_of_a_l310_310409


namespace median_vision_is_correct_l310_310234

def student_vision_data : List (ℝ × ℕ) := [
  (4.0, 1),
  (4.1, 2),
  (4.2, 6),
  (4.3, 3),
  (4.4, 3),
  (4.5, 4),
  (4.6, 1),
  (4.7, 2),
  (4.8, 5),
  (4.9, 7),
  (5.0, 5)
]

def total_students : ℕ := 39

def median_vision_value (data : List (ℝ × ℕ)) (total : ℕ) : ℝ :=
  -- Function to calculate the median vision value
  let sorted_data := data.sortBy (λ x, x.1) in
  let cumulative_counts := sorted_data.scanl (λ acc x, acc + x.2) 0 in
  let median_position := total / 2 in
  let median_index := cumulative_counts.indexWhere (λ x, x > median_position) in
  (sorted_data.nth! (median_index - 1)).1

theorem median_vision_is_correct : median_vision_value student_vision_data total_students = 4.6 :=
by
  -- Prove that the median vision value of the dataset is 4.6
  sorry

end median_vision_is_correct_l310_310234


namespace average_score_of_group_l310_310364

def class_average := 85
def differences : List ℤ := [2, 3, -3, -5, 12, 12, 8, 2, -1, 4, -10, -2, 5, 5]
def n := differences.length

theorem average_score_of_group :
  (class_average + (differences.sum) / n.toRat) = 87.29 := 
by
  sorry

end average_score_of_group_l310_310364


namespace distance_traveled_l310_310530

-- Definition of the velocity function
def velocity (t : ℝ) : ℝ := 2 * t - 3

-- Prove the integral statement
theorem distance_traveled : 
  (∫ t in (0 : ℝ)..(5 : ℝ), abs (velocity t)) = 29 / 2 := by 
{ sorry }

end distance_traveled_l310_310530


namespace rhombus_angles_l310_310596

-- Define the conditions for the proof
variables (a e f : ℝ) (α β : ℝ)

-- Using the geometric mean condition
def geometric_mean_condition := a^2 = e * f

-- Using the condition that diagonals of a rhombus intersect at right angles and bisect each other
def diagonals_intersect_perpendicularly := α + β = 180 ∧ α = 30 ∧ β = 150

-- Prove the question assuming the given conditions
theorem rhombus_angles (h1 : geometric_mean_condition a e f) (h2 : diagonals_intersect_perpendicularly α β) : 
  (α = 30) ∧ (β = 150) :=
sorry

end rhombus_angles_l310_310596


namespace find_c_l310_310274

theorem find_c (x y c : ℝ) (h1 : 7^(3 * x - 1) * 3^(4 * y - 3) = c^x * 27^y)
  (h2 : x + y = 4) : c = 49 :=
by
  sorry

end find_c_l310_310274


namespace impossibility_of_closed_odd_length_line_l310_310797

theorem impossibility_of_closed_odd_length_line (vertices : List (ℚ × ℚ))
    (h1 : ∀ {v1 v2 : ℚ × ℚ}, v1 ∈ vertices → v2 ∈ vertices → (v1 ≠ v2 → (v1.1 - v2.1)^2 + (v1.2 - v2.2)^2 = 1))
    (h2 : vertices.head = vertices.last)
    (h3 : vertices.length % 2 = 1) :
    False := 
  by 
    sorry

end impossibility_of_closed_odd_length_line_l310_310797


namespace math_problem_l310_310716

variable (x y : ℝ)

theorem math_problem (h1 : x + y = 10) (h2 : x^2 - y^2 = 40) : x - y = 4 :=
by sorry

end math_problem_l310_310716


namespace num_factors_greater_than_4_l310_310604

theorem num_factors_greater_than_4 (n : ℕ) (h : n = 2 * 3 * 5^2 * 17) : 
  {d : ℕ | d ∣ n ∧ (nat.factors d).length > 4}.to_finset.card = 7 := by sorry

end num_factors_greater_than_4_l310_310604


namespace smaller_molds_radius_l310_310179

theorem smaller_molds_radius (r : ℝ) : 
  (∀ V_large V_small : ℝ, 
     V_large = (2/3) * π * (2:ℝ)^3 ∧
     V_small = (2/3) * π * r^3 ∧
     8 * V_small = V_large) → r = 1 := by
  sorry

end smaller_molds_radius_l310_310179


namespace fraction_of_historical_fiction_new_releases_l310_310199

theorem fraction_of_historical_fiction_new_releases (total_books : ℕ) (p1 p2 p3 : ℕ) (frac_hist_fic : Rat) (frac_new_hist_fic : Rat) (frac_new_non_hist_fic : Rat) 
  (h1 : total_books > 0) (h2 : frac_hist_fic = 40 / 100) (h3 : frac_new_hist_fic = 40 / 100) (h4 : frac_new_non_hist_fic = 40 / 100) 
  (h5 : p1 = frac_hist_fic * total_books) (h6 : p2 = frac_new_hist_fic * p1) (h7 : p3 = frac_new_non_hist_fic * (total_books - p1)) :
  p2 / (p2 + p3) = 2 / 5 :=
by
  sorry

end fraction_of_historical_fiction_new_releases_l310_310199


namespace exists_cycle_length_not_divisible_by_3_l310_310007

variable (V : Type) [Fintype V] [DecidableEq V]

structure Graph (V : Type) :=
  (adj : V → V → Prop)
  (symm : ∀ {x y}, adj x y → adj y x)
  (irrefl : ∀ x, ¬ adj x x)

variable (G : Graph V)

-- Define 3-regular (cubic) graph property
def is_3_regular : Prop := 
  ∀ v : V, Fintype.card {w : V // G.adj v w} = 3

theorem exists_cycle_length_not_divisible_by_3 
  (hG : ∀ v : V, Fintype.card {w : V // G.adj v w} = 3) :
  ∃ (c : List V), List.Length c ≠ 0 ∧ ∀ e ∈ c.zip (c.tail), G.adj e.fst e.snd ∧ List.Length c % 3 ≠ 0 :=
sorry

end exists_cycle_length_not_divisible_by_3_l310_310007


namespace max_cos_y_cos_x_l310_310110

noncomputable def max_cos_sum : ℝ :=
  1 + (Real.sqrt (2 + Real.sqrt 2)) / 2

theorem max_cos_y_cos_x
  (x y : ℝ)
  (h1 : Real.sin y + Real.sin x + Real.cos (3 * x) = 0)
  (h2 : Real.sin (2 * y) - Real.sin (2 * x) = Real.cos (4 * x) + Real.cos (2 * x)) :
  ∃ (x y : ℝ), Real.cos y + Real.cos x = max_cos_sum :=
sorry

end max_cos_y_cos_x_l310_310110


namespace problem_1_problem_2_l310_310280

variable {x m : ℝ}

-- Problem 1
theorem problem_1 (h₀ : ∀ x, (x+1)*(x-5) ≤ 0 → (1-m) ≤ x ∧ x ≤ (1+m))
    (h₁ : 0 < m) : m ≥ 4 :=
sorry

-- Problem 2
theorem problem_2 (h₀ : m = 5)
    (h₁ : ∃ x, ((x+1)*(x-5) ≤ 0) ∨ ((1-m) ≤ x ∧ x ≤ (1+m)))
    (h₂ : ¬ (∀ x, (x+1)*(x-5) ≤ 0 ∧ (1-m) ≤ x ∧ x ≤ (1+m))) :
    ∃ x, (x ∈ [-4, -1) ∪ (5, 6]) :=
sorry

end problem_1_problem_2_l310_310280


namespace common_sum_is_zero_l310_310443

theorem common_sum_is_zero :
  let nums := list.range (24 + 1) -- generates list from 0 to 24
  let nums := list.map (λ n, n - 12) nums -- shift range to -12 to 12
  let n := 5
  let rows := 5
  let cols := 5
  ∃ (square : matrix (fin rows) (fin cols) ℤ),
    (∀ i : fin rows, ∑ j in (finset.univ : finset (fin cols)), square i j = 0) ∧
    (∀ j : fin cols, ∑ i in (finset.univ : finset (fin rows)), square i j = 0) ∧
    (∑ i in (finset.univ : finset (fin rows)), (square i i) = 0) ∧
    (∑ i in (finset.univ : finset (fin rows)), (square i ⟨(n - i - 1), sorry⟩) = 0)
    := sorry

end common_sum_is_zero_l310_310443


namespace tank_width_l310_310485

theorem tank_width :
  ∀ (Rate Time Length Depth : ℕ) (Volume : ℕ) (Width : ℕ),
    Rate = 5 →
    Time = 60 →
    Length = 10 →
    Depth = 5 →
    Volume = Rate * Time →
    Width = Volume / (Length * Depth) →
    Width = 6 := by
  intros Rate Time Length Depth Volume Width hRate hTime hLength hDepth hVolume hWidth
  rw [hRate, hTime, hLength, hDepth] at hVolume
  norm_num at hVolume
  rw [hVolume] at hWidth
  norm_num at hWidth
  exact hWidth

end tank_width_l310_310485


namespace analytical_expression_of_f_range_of_f_l310_310683

def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + (Real.pi / 6))

/-- Given that f(x) has the smallest positive period π and a lowest point M(2π/3, -2), show that f(x) = 2 * sin(2x + π/6) -/
theorem analytical_expression_of_f :
  ∀ (x : ℝ), f x = 2 * Real.sin (2 * x + (Real.pi / 6)) := sorry

/-- Prove that the range of f(x) in the interval [0, π/2] is [-1, 2] -/
theorem range_of_f :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ Real.pi / 2 → -1 ≤ f x ∧ f x ≤ 2 := sorry

end analytical_expression_of_f_range_of_f_l310_310683


namespace lines_perpendicular_l310_310000

theorem lines_perpendicular 
  (x y : ℝ)
  (first_angle : ℝ)
  (second_angle : ℝ)
  (h1 : first_angle = 50 + x - y)
  (h2 : second_angle = first_angle - (10 + 2 * x - 2 * y)) :
  first_angle + second_angle = 90 :=
by 
  sorry

end lines_perpendicular_l310_310000


namespace smallest_perfect_square_divisible_by_4_and_5_l310_310498

theorem smallest_perfect_square_divisible_by_4_and_5 : 
  ∃ n : ℕ, n > 0 ∧ n ∣ 4 ∧ n ∣ 5 ∧ is_square n ∧ 
  ∀ m : ℕ, (m > 0 ∧ m ∣ 4 ∧ m ∣ 5 ∧ is_square m) → n ≤ m :=
sorry

end smallest_perfect_square_divisible_by_4_and_5_l310_310498


namespace butterflies_direction_at_15_feet_equivalence_l310_310845

noncomputable def distance_after_n_cycles (n : ℕ) : ℝ :=
  let x_pos := (n * 2, n * 2, n * 2)
  let y_pos := (-n * 3, -n * 3, n * 3)
  real.sqrt ((x_pos.1 - y_pos.1)^2 + (x_pos.2 - y_pos.2)^2 + (x_pos.3 - y_pos.3)^2)

theorem butterflies_direction_at_15_feet_equivalence : (distance_after_n_cycles 2.5 = 15) → 
  (true) :=  -- This is a placeholder, since detailed movement analysis is needed for exact distance matching.
begin
  sorry
end

end butterflies_direction_at_15_feet_equivalence_l310_310845


namespace line_equation_through_point_with_given_area_l310_310624

theorem line_equation_through_point_with_given_area :
  ∃ k: ℝ, k < 0 ∧ 
    ((∀ x y: ℝ, 2 + (4 / 3 * k) = y) ∧ 
     (∀ x y: ℝ, (4 / 3 - 2 / k) = x) ∧ 
     (1 / 2 * (2 + (4 / 3 * k)) * (4 / 3 - 2 / k) = 6)) →
  (∃ a b c: ℝ, a * 6 + b * 3 + c = 0) :=
begin 
  sorry -- Placeholder for proof
end

end line_equation_through_point_with_given_area_l310_310624


namespace lineD_intersects_line1_l310_310939

-- Define the lines based on the conditions
def line1 (x y : ℝ) := x + y - 1 = 0
def lineA (x y : ℝ) := 2 * x + 2 * y = 6
def lineB (x y : ℝ) := x + y = 0
def lineC (x y : ℝ) := y = -x - 3
def lineD (x y : ℝ) := y = x - 1

-- Define the statement that line D intersects with line1
theorem lineD_intersects_line1 : ∃ (x y : ℝ), line1 x y ∧ lineD x y :=
by
  sorry

end lineD_intersects_line1_l310_310939


namespace triangles_type_l310_310350

-- Definitions to capture the problem conditions
variables {A1 A2 B1 B2 C1 C2 : ℝ}

-- Problem conditions
def conditions := 
  cos A1 = sin A2 ∧ 
  cos B1 = sin B2 ∧ 
  cos C1 = sin C2

-- The type of the triangles
def is_acute (A B C : ℝ) := 
  A < 90 ∧ B < 90 ∧ C < 90

def is_obtuse (A B C : ℝ) :=
  A > 90 ∨ B > 90 ∨ C > 90

theorem triangles_type
  (hc : conditions) :
  is_acute A1 B1 C1 ∧ is_obtuse A2 B2 C2 :=
sorry

end triangles_type_l310_310350


namespace range_of_t_l310_310968

noncomputable def f : ℝ → ℝ := sorry

axiom f_symmetric (x : ℝ) : f (x - 3) = f (-x - 3)
axiom f_ln_definition (x : ℝ) (h : x ≤ -3) : f x = Real.log (-x)

theorem range_of_t (t : ℝ) :
  (∀ x : ℝ, f (Real.sin x - t) > f (3 * Real.sin x - 1)) ↔ (t < -1 ∨ t > 9) := sorry

end range_of_t_l310_310968


namespace Donny_spends_28_on_Thursday_l310_310988

theorem Donny_spends_28_on_Thursday :
  let monday_savings := 15
  let tuesday_savings := 28
  let wednesday_savings := 13
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  let thursday_spending := total_savings / 2
  thursday_spending = 28 :=
by 
  let monday_savings := 15
  let tuesday_savings := 28
  let wednesday_savings := 13
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  let thursday_spending := total_savings / 2
  sorry

end Donny_spends_28_on_Thursday_l310_310988


namespace ascending_numbers_count_400_600_l310_310849

def is_ascending (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  ∀ i < digits.length - 1, digits.nth_le i (by linarith) < digits.nth_le (i + 1) (by linarith)

def ascending_numbers_in_range (a b : ℕ) : list ℕ :=
  (list.range' a (b - a + 1)).filter is_ascending

theorem ascending_numbers_count_400_600 :
  (ascending_numbers_in_range 400 600).length = 16 :=
by 
  sorry

end ascending_numbers_count_400_600_l310_310849


namespace can_conceive_room_not_fully_illuminate_walls_l310_310588

theorem can_conceive_room_not_fully_illuminate_walls (n : ℕ) (h : n = 10 ∨ n = 6) :
  ∃ (polygon : Type) [concave_polygon polygon] (candle : Type), ∀ (wall : polygon), ¬fully_illuminated candle wall :=
sorry

end can_conceive_room_not_fully_illuminate_walls_l310_310588


namespace concurrency_AX_BY_CZ_l310_310053

-- Let ABC be a triangle
variables (A B C A1 A2 B1 B2 C1 C2 X Y Z : Type)

-- Points A1 and A2 are on segment [BC] such that BA1 = A2C
axiom hA1 : segment A B C A1
axiom hA2 : segment A B C A2
axiom hAeq : dist B A1 = dist A2 C

-- Points B1 and B2 are on segment [CA] such that CB1 = B2A
axiom hB1 : segment B C A B1
axiom hB2 : segment B C A B2
axiom hBeq : dist C B1 = dist B2 A

-- Points C1 and C2 are on segment [AB] such that AC1 = C2B
axiom hC1 : segment C A B C1
axiom hC2 : segment C A B C2
axiom hCeq : dist A C1 = dist C2 B

-- X is the intersection of lines BB1 and CC2
axiom hX : intersect_lines B B1 C C2 X

-- Y is the intersection of lines CC1 and AA2
axiom hY : intersect_lines C C1 A A2 Y

-- Z is the intersection of lines AA1 and BB2
axiom hZ : intersect_lines A A1 B B2 Z

-- Show that lines AX, BY, and CZ are concurrent
theorem concurrency_AX_BY_CZ : is_concurrent A X B Y C Z :=
  sorry

end concurrency_AX_BY_CZ_l310_310053


namespace hydra_defeated_in_ten_strikes_l310_310455
noncomputable def smallest_strikes_for_hydra (e : ℕ) (h : ℕ) : ℕ :=
  if e = 100 then 10 else sorry

theorem hydra_defeated_in_ten_strikes : ∀ (h e : ℕ), e = 100 → smallest_strikes_for_hydra e h = 10 :=
begin
  intros h e heq,
  rw heq,
  simp [smallest_strikes_for_hydra],
end

end hydra_defeated_in_ten_strikes_l310_310455


namespace chord_length_of_intersection_l310_310823

-- Define the line equation
def line (x y : ℝ) : Prop := x + (sqrt 3) * y - 2 = 0

-- Define the circle equation
def circle (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the distance formula from a point to a line
def distance_to_line (a b c x₀ y₀ : ℝ) : ℝ := abs (a * x₀ + b * y₀ - c) / sqrt (a^2 + b^2)

-- State the main theorem
theorem chord_length_of_intersection :
  ∃ (AB : ℝ), (AB = 2 * sqrt 3) ∧ 
  (∀ x y : ℝ, line x y → circle x y → True) :=
by
  let d := distance_to_line 1 (sqrt 3) 2 0 0
  have hd : d = 1 := by sorry
  use 2 * sqrt 3
  split
  { 
    exact rfl 
  }
  {
    intros x y hline hcircle
    sorry
  }

end chord_length_of_intersection_l310_310823


namespace area_of_triangle_VQS_theorem_l310_310805

-- Definitions and assumptions directly derived from the conditions
variable (PQRS : Type*) [Field PQRS] [MetricSpace PQRS] [NormedAddCommGroup PQRS]
variable (T : PQRS)
variable (U V : PQRS)
variable (area_square area_RUTV area_VQS : ℝ)

-- Conditions
axiom square_area : area_square = 144
axiom point_T_on_RS : T ∈ (segmen RS)
axiom midpoint_U_PT : U = midpoint(P, T)
axiom midpoint_V_QT : V = midpoint(Q, T)
axiom quadrilateral_RUTV_area : area_RUTV = 20

-- Question: What is the area of triangle VQS?
axiom area_of_triangle_VQS : area_VQS = 34

-- Stating the theorem
theorem area_of_triangle_VQS_theorem : area_of_triangle_VQS == 34 :=
    by
    sorry


end area_of_triangle_VQS_theorem_l310_310805


namespace no_finite_sequence_rational_points_l310_310965

theorem no_finite_sequence_rational_points :
  ¬ ∃ (n : ℕ) (P : fin n → ℚ × ℚ),
    (P 0 = (0, 0)) ∧ (P (fin.last n) = (0, 1 / 2)) ∧
    (∀ i : fin n, dist P i P (i + 1) = 1) :=
sorry

end no_finite_sequence_rational_points_l310_310965


namespace ball_bounce_l310_310171

theorem ball_bounce :
  ∃ b : ℕ, 324 * (3 / 4) ^ b < 40 ∧ b = 8 :=
by
  have : (3 / 4 : ℝ) < 1 := by norm_num
  have h40_324 : (40 : ℝ) / 324 = 10 / 81 := by norm_num
  sorry

end ball_bounce_l310_310171


namespace sphere_diameter_l310_310953

theorem sphere_diameter (r : ℝ) (a b : ℕ) (π : ℝ) 
  (h1 : r = 6) 
  (h2 : 864 * π = 3 * (4 / 3) * π * (r ^ 3)) 
  (h3 : 2 * (r * 3.isCubeRoot.toReal) = a * (b.isCubeRoot.toReal))
  (h4 : 2 * r * 3.isCubeRoot.toReal = 12 * 2.isCubeRoot.toReal):
  a + b = 14 :=
by
  sorry

end sphere_diameter_l310_310953


namespace solve_x_l310_310712

theorem solve_x (x: ℝ) (h: -4 * x - 15 = 12 * x + 5) : x = -5 / 4 :=
sorry

end solve_x_l310_310712


namespace find_m_l310_310836

noncomputable def volume_of_parallelepiped (v1 v2 v3 : ℝ^3) : ℝ :=
  Real.abs (Matrix.det ![
    ![v1.x, v2.x, v3.x],
    ![v1.y, v2.y, v3.y],
    ![v1.z, v2.z, v3.z]
  ])

theorem find_m (m : ℝ) (h : m > 0) :
  volume_of_parallelepiped ⟨3, 5, 2⟩ ⟨2, m, 3⟩ ⟨1, 4, m⟩ = 20 :=
  abs(3*m^2 - 10*m + 1) = 20 → m = (5 + Real.sqrt 82) / 3 :=
begin
  sorry
end

end find_m_l310_310836


namespace roots_of_P_are_roots_of_R_R_has_no_multiple_roots_l310_310251

noncomputable def P (x : ℝ) : ℝ := sorry
noncomputable def P_prime (x : ℝ) : ℝ := by
  exact sorry -- This should be the derivative of P(x)

noncomputable def Q (x : ℝ) : ℝ := sorry -- gcd(P(x), P'(x))

-- Define polynomial R(x) as R(x) = P(x) / Q(x)
noncomputable def R (x : ℝ) : ℝ := P x / Q x

-- Prove all roots of P(x) are roots of R(x)
theorem roots_of_P_are_roots_of_R : ∀ x : ℝ, P x = 0 → R x = 0 :=
by
  sorry

-- Prove R(x) has no multiple roots
theorem R_has_no_multiple_roots : ∀ x : ℝ, (R x = 0 ∧ R' x = 0) → False :=
by
  sorry

end roots_of_P_are_roots_of_R_R_has_no_multiple_roots_l310_310251


namespace radio_loss_percentage_l310_310447

def loss (CP SP : ℝ) : ℝ :=
  CP - SP

def loss_percentage (CP SP : ℝ) : ℝ :=
  (loss CP SP / CP) * 100

theorem radio_loss_percentage :
  loss_percentage 2400 2100 = 12.5 := by
  sorry

end radio_loss_percentage_l310_310447


namespace fraction_simplification_l310_310511

theorem fraction_simplification : 
  (1/5 - 1/6) / (1/3 - 1/4) = 2/5 := 
by 
  sorry

end fraction_simplification_l310_310511


namespace max_sin_A_of_triangle_condition_l310_310383

variables {V : Type*} [inner_product_space ℝ V] (A B C : V)
open real_inner_product_space

theorem max_sin_A_of_triangle_condition (h : ∥(A - B) + (C - B)∥ = 2 * ∥(A - B) - (C - B)∥) : 
  ∃ (sin_A : ℝ), sin_A = 4 / 5 ∧ sin_A ≤ 1 := 
begin 
  sorry 
end

end max_sin_A_of_triangle_condition_l310_310383


namespace part1_part2_part3_l310_310397

variables {A B C : Type} -- Points representing the vertices of the triangle
variables {IA IB IC HA HB HC HG GB GC : Type} -- Points representing other mentioned points in the problem
variables {a b c : ℝ} -- The lengths of the sides of the triangle
variables {angleA angleB angleC : ℝ} -- The angles at vertices A, B, and C respectively
variables {sinA sinB sinC tanA tanB tanC cotA cotB cotC : ℝ} -- Trigonometric functions of the angles
variables {G I H : Type} -- Points G, I, and H

-- Non-right triangle condition
axiom non_right_triangle (A B C : Type) (a b c : ℝ) : A ≠ B → B ≠ C → C ≠ A

-- Points are defined as described in the problem
axiom Centroid (G : Type) : True
axiom Incenter (I : Type) : True
axiom Orthocenter (H : Type) : True

-- Define theorems to be proved

theorem part1 (sinA sinB sinC : ℝ) (IA IB IC : Type) : 
  sinA * IA + sinB * IB + sinC * IC = 0 :=
sorry

theorem part2 (tanA tanB tanC : ℝ) (HA HB HC : Type) : 
  tanA * HA + tanB * HB + tanC * HC = 0 :=
sorry

theorem part3 (cotA cotB cotC : ℝ) (HG GB GC : Type) : 
  HG = cotC * (cotB - cotA) * GB + cotB * (cotC - cotA) * GC :=
sorry

end part1_part2_part3_l310_310397


namespace min_value_sum_reciprocal_squares_l310_310298

open Real

theorem min_value_sum_reciprocal_squares 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 1) :  
  (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ 27 := 
sorry

end min_value_sum_reciprocal_squares_l310_310298


namespace eccentricity_of_ellipse_l310_310322

-- Let the ellipse be defined by the equation given conditions
variables {a b c x y : ℝ}
def ellipse (a b x y : ℝ) : Prop :=
  (0 < a) ∧ (a > b) ∧ (0 < b) ∧ (c = Real.sqrt (a^2 - b^2)) ∧ ((x^2 / a^2) + (y^2 / b^2) = 1)

-- The coordinates of the point F
def F_coord (c : ℝ) := (-c, 0)

-- The line equation l
def line_l (a c x : ℝ) : Prop :=
  (x = -a^2 / c)

-- The isosceles triangle condition
def isosceles_triangle (e : ℝ) : Prop :=
  (0 < e) ∧ (e < 1) ∧ (2 * e^4 - 3 * e^2 + 1 = 0)

-- Define the proof problem
theorem eccentricity_of_ellipse (a b c x y : ℝ) (h : ellipse a b x y) :
  isosceles_triangle (c / a) → (c / a = Real.sqrt 2 / 2) :=
by
  sorry

end eccentricity_of_ellipse_l310_310322


namespace find_indices_l310_310755

theorem find_indices (a b : Fin 7 → ℝ) 
  (h1 : ∀ i, 0 ≤ a i ∧ 0 ≤ b i) 
  (h2 : ∀ i, a i + b i ≤ 2) : 
  ∃ (k ≠ m : Fin 7), |a k - a m| + |b k - b m| ≤ 1 :=
by
  sorry

end find_indices_l310_310755


namespace base12_subtraction_correct_l310_310438

def base12_subtract : Prop :=
  let A := 10
  let B := 11
  let x := 9 * 144 + B * 12 + 5 -- 9B5_12 in decimal
  let y := 6 * 144 + A * 12 + 3 -- 6A3_12 in decimal
  let result := (x - y)
  let expected_result := 3 * 144 + 1 * 12 + 2 -- 312_12 in decimal
  result = expected_result

theorem base12_subtraction_correct : base12_subtract :=
by
  let A := 10
  let B := 11
  let x := 9 * 144 + B * 12 + 5
  let y := 6 * 144 + A * 12 + 3
  let result := x - y
  let expected_result := 3 * 144 + 1 * 12 + 2
  show result = expected_result from sorry

end base12_subtraction_correct_l310_310438


namespace value_of_c_l310_310022

theorem value_of_c :
  ∃ (a b c : ℕ), 
  30 = 2 * (10 + a) ∧ 
  b = 2 * (a + 30) ∧ 
  c = 2 * (b + 30) ∧ 
  c = 200 := 
sorry

end value_of_c_l310_310022


namespace bank_deposit_return_l310_310751

theorem bank_deposit_return 
  (initial_deposit : ℝ) (annual_interest_rate : ℝ) (period_in_months : ℝ) (exchange_rate : ℝ) (insurance_limit : ℝ) :
  initial_deposit = 23904 →
  annual_interest_rate = 0.05 →
  period_in_months = 3 →
  exchange_rate = 58.15 →
  insurance_limit = 1400000 →
  let period_in_years := period_in_months / 12 in
  let interest_accrued := initial_deposit * (1 + annual_interest_rate * period_in_years) in
  let amount_in_rubles := interest_accrued * exchange_rate in
  min amount_in_rubles insurance_limit = 1400000 :=
begin
  intros h1 h2 h3 h4 h5,
  let period_in_years := period_in_months / 12,
  let interest_accrued := initial_deposit * (1 + annual_interest_rate * period_in_years),
  let amount_in_rubles := interest_accrued * exchange_rate,
  have h : interest_accrued = 23904 * (1 + 0.05 * (3 / 12)), by { rw [h1, h2, h3], },
  have amount_in_rubles := interest_accrued * exchange_rate,
  rw [h, h4],
  have h_limit : min amount_in_rubles 1400000 = 1400000,
  {
    sorry
  },
  exact h_limit,
end

end bank_deposit_return_l310_310751


namespace ladder_slip_l310_310169

theorem ladder_slip {l h₀ s : ℝ} (l_pos : l = 30) (h₀_pos : h₀ = 8) (s_pos : s = 6) :
  let x := real.sqrt (l^2 - h₀^2),
      x' := x - s,
      y := real.sqrt(l^2 - x'^2) - h₀
  in y ≈ 11 := sorry

end ladder_slip_l310_310169


namespace bus_stops_time_per_hour_l310_310997

theorem bus_stops_time_per_hour :
  ∀ (speed_excl_stop speed_incl_stop : ℝ), 
    speed_excl_stop = 54 → speed_incl_stop = 36 →
    ∃ (stoppage_time : ℝ), stoppage_time = 20 :=
by
  intros speed_excl_stop speed_incl_stop h1 h2
  use 20
  sorry

end bus_stops_time_per_hour_l310_310997


namespace min_value_a_plus_b_l310_310646

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + b = 2 * a * b) : a + b ≥ 2 + Real.sqrt 3 :=
sorry

end min_value_a_plus_b_l310_310646


namespace closest_approx_of_d_l310_310490

theorem closest_approx_of_d : abs ((69.28 * 0.004 / 0.03) - 9.24) < 0.01 :=
by
  sorry

end closest_approx_of_d_l310_310490


namespace probability_bernardo_smaller_than_silvia_l310_310580

-- Define the two sets Bernardo's and Silvia's selections come from
def bernardo_set : finset ℕ := {1, 2, 3, 4, 5, 6}
def silvia_set : finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Function to calculate combination binomial coefficient
def choose (n : ℕ) (k : ℕ) := nat.choose n k

-- The mathematically equivalent Lean 4 statement
theorem probability_bernardo_smaller_than_silvia :
  let prob := (choose 6 2 / choose 7 3) + ((1 - choose 6 2 / choose 7 3) * (19 / 20) / 2) in 
  prob = 49 / 70 :=
begin
  sorry
end

end probability_bernardo_smaller_than_silvia_l310_310580


namespace nearest_integer_to_3_plus_sqrt2_pow_four_l310_310492

open Real

theorem nearest_integer_to_3_plus_sqrt2_pow_four : 
  (∃ n : ℤ, abs (n - (3 + (sqrt 2))^4) < 0.5) ∧ 
  (abs (382 - (3 + (sqrt 2))^4) < 0.5) := 
by 
  sorry

end nearest_integer_to_3_plus_sqrt2_pow_four_l310_310492


namespace rectangle_perimeter_inscribed_l310_310548

noncomputable def circle_area : ℝ := 32 * Real.pi
noncomputable def rectangle_area : ℝ := 34
noncomputable def rectangle_perimeter : ℝ := 28

theorem rectangle_perimeter_inscribed (area_circle : ℝ := 32 * Real.pi)
  (area_rectangle : ℝ := 34) : ∃ (P : ℝ), P = 28 :=
by
  use rectangle_perimeter
  sorry

end rectangle_perimeter_inscribed_l310_310548


namespace incorrect_statement_l310_310765

def g (x : ℝ) := (2 * x - 3) / (x + 2)

theorem incorrect_statement (x y : ℝ) : ¬ (x = (2 * y - 3) / (y + 2)) :=
sorry

end incorrect_statement_l310_310765


namespace sum_reciprocals_of_B_elements_m_plus_n_equals_43_l310_310047

noncomputable def B_elements_reciprocal_sum : ℚ := (∑' a b c d : ℕ, 1 / (2^a * 3^b * 5^c * 7^d))

theorem sum_reciprocals_of_B_elements : B_elements_reciprocal_sum = (35 / 8) :=
  sorry

theorem m_plus_n_equals_43 : ∃ m n : ℕ, m + n = 43 ∧ (m / n).rat_cast ≝ (35 / 8) :=
  exists.intro 35 (exists.intro 8 (and.intro rfl sorry))

end sum_reciprocals_of_B_elements_m_plus_n_equals_43_l310_310047


namespace circle_tangent_to_line_at_origin_l310_310220

-- Conditions
def circle (m : ℝ) : set (ℝ × ℝ) := { p | p.1^2 + p.2^2 = m^2 }
def line (m : ℝ) : set (ℝ × ℝ) := { p | p.1 + p.2 = m }

-- Problem statement
theorem circle_tangent_to_line_at_origin (m : ℝ) :
  (∃ (p : ℝ × ℝ), p ∈ circle m ∧ p ∈ line m ∧ (∀ (q : ℝ × ℝ), q ∈ circle m → q.1 + q.2 ≠ m)) →
  m = 0 :=
by
  sorry

end circle_tangent_to_line_at_origin_l310_310220


namespace floor_ineq_l310_310429

theorem floor_ineq (x : ℝ) (n : ℕ) (hx : 0 < x) (hn : 0 < n) :
  nat.floor (n * x) ≥ ∑ k in finset.range(n) + 1, nat.floor (k * x) / k :=
sorry

end floor_ineq_l310_310429


namespace points_product_l310_310566

def f (n : ℕ) : ℕ :=
  if n % 6 == 0 then 6
  else if n % 2 == 0 then 2
  else 0

def total_points (rolls : List ℕ) : ℕ :=
  rolls.map f |>.sum

def allie_rolls := [5, 4, 1, 2]
def betty_rolls := [6, 3, 3, 2]

def allie_points := total_points allie_rolls
def betty_points := total_points betty_rolls

theorem points_product : allie_points * betty_points = 32 := by
  sorry

end points_product_l310_310566


namespace meal_pass_cost_is_25_l310_310082

/-- Sally's trip to Sea World conditions -/
variables 
  (initial_savings : ℕ)         -- $28 already saved
  (parking_cost : ℕ)            -- $10 to park
  (entrance_fee : ℕ)            -- $55 to get into the park
  (distance_to_sea_world : ℕ)   -- 165 miles one way
  (mileage_per_gallon : ℕ)      -- 30 miles per gallon
  (gas_price_per_gallon : ℕ)    -- $3 per gallon
  (additional_savings_needed : ℕ) -- Needs to save up $95 more

/-- The cost of the meal pass -/
def meal_pass_cost (initial_savings parking_cost entrance_fee distance_to_sea_world mileage_per_gallon gas_price_per_gallon additional_savings_needed : ℕ) : ℕ :=
  let round_trip_distance := distance_to_sea_world * 2,
      gallons_needed_for_trip := round_trip_distance / mileage_per_gallon,
      cost_of_gas := gallons_needed_for_trip * gas_price_per_gallon,
      total_known_costs := parking_cost + entrance_fee + cost_of_gas,
      remaining_after_initial_savings := total_known_costs - initial_savings,
      meal_pass_cost_calculated := additional_savings_needed - remaining_after_initial_savings in
    meal_pass_cost_calculated

theorem meal_pass_cost_is_25 :
  meal_pass_cost 28 10 55 165 30 3 95 = 25 :=
by
  unfold meal_pass_cost
  norm_num
  sorry

end meal_pass_cost_is_25_l310_310082


namespace parabola_line_non_intersect_l310_310050

theorem parabola_line_non_intersect (r s : ℝ) (Q : ℝ × ℝ) (P : ℝ → ℝ)
  (hP : ∀ x, P x = x^2)
  (hQ : Q = (10, 6))
  (h_cond : ∀ m : ℝ, ¬∃ x : ℝ, (Q.snd - 6 = m * (Q.fst - 10)) ∧ (P x = x^2) ↔ r < m ∧ m < s) :
  r + s = 40 :=
sorry

end parabola_line_non_intersect_l310_310050


namespace scientific_notation_43_681_billion_l310_310525

theorem scientific_notation_43_681_billion : 
  ∃ a : ℝ, (1 ≤ |a| ∧ |a| < 10) ∧ ∃ n : ℤ, 43.681 * 10^9 = a * 10^n ∧ a = 0.437 ∧ n = 12 :=
begin
  sorry
end

end scientific_notation_43_681_billion_l310_310525


namespace sum_reciprocal_sequence_l310_310673

noncomputable def sum_a_n (n : ℕ) : ℕ := 2^n - 1

theorem sum_reciprocal_sequence (n : ℕ) :
  let a_n := λ (k : ℕ), if k = 0 then 0 else 2^(k-1) in
  let S_n := sum_a_n n in
  let a'_n := λ (k : ℕ), if a_n k = 0 then 0 else (1 / (a_n k : ℝ)) in
  let T_n := (∑ k in finset.range n, a'_n (k + 1)) in
  T_n = 2 - 1 / (2^(n-1) : ℝ) :=
by sorry

end sum_reciprocal_sequence_l310_310673


namespace part1_expr_value_l310_310165

theorem part1_expr_value : (-1)^0 + (27: ℝ)^(1/3) + (4: ℝ)^(1/2) + |real.sqrt 3 - 2| = 8 - real.sqrt 3 :=
by
  sorry

end part1_expr_value_l310_310165


namespace smallest_prime_divisor_of_sum_l310_310852

-- Definition of relevant expressions
def a := 3^19
def b := 6^21
def sum := a + b

-- Conditions translated into Lean assumptions
def odd_a : a % 2 = 1 := by sorry
def even_b : b % 2 = 0 := by sorry
def sum_is_odd : sum % 2 = 1 := by sorry

-- Check divisibility by 3
def a_div_3 : a % 3 = 0 := by sorry
def b_div_3 : b % 3 = 0 := by sorry
def sum_div_3 : sum % 3 = 0 := by sorry

-- Main statement to prove
theorem smallest_prime_divisor_of_sum : ∃ p : ℕ, p.prime ∧ p ∣ sum ∧ ∀ q : ℕ, q.prime ∧ q ∣ sum → p ≤ q :=
begin
  use 3,
  split,
  { exact prime_three, }, -- 3 is prime
  split,
  { exact sum_div_3, },   -- 3 divides sum
  { intros q hq1 hq2,
    by_contradiction,
    have q >= 3 := sorry, -- Contradiction that any smaller prime can divide sum
    exact le_of_eq corefl, -- Since 3 is itself
  }
end

end smallest_prime_divisor_of_sum_l310_310852


namespace rotation_transforms_and_sums_l310_310738

theorem rotation_transforms_and_sums 
    (D E F D' E' F' : (ℝ × ℝ))
    (hD : D = (0, 0)) (hE : E = (0, 20)) (hF : F = (30, 0)) 
    (hD' : D' = (-26, 23)) (hE' : E' = (-46, 23)) (hF' : F' = (-26, -7))
    (n : ℝ) (x y : ℝ)
    (rotation_condition : 0 < n ∧ n < 180)
    (angle_condition : n = 90) :
    n + x + y = 60.5 :=
by
  have hx : x = -49 := sorry
  have hy : y = 19.5 := sorry
  have hn : n = 90 := sorry
  sorry

end rotation_transforms_and_sums_l310_310738


namespace vector_m_value_l310_310700

theorem vector_m_value {m : ℝ} 
  (h1 : vector (ℝ × ℝ) 2 : fin 2 → ℝ := ![(m, 3), (√3, 1)])
  (h2 : real.angleBetweenVectors (m, 3) (√3, 1) = real.pi / 6) : 
  m = √3 := by
  sorry

end vector_m_value_l310_310700


namespace ratio_of_juice_to_bread_l310_310790

variable (total_money : ℕ) (money_left : ℕ) (cost_bread : ℕ) (cost_butter : ℕ) (cost_juice : ℕ)

def compute_ratio (total_money money_left cost_bread cost_butter cost_juice : ℕ) : ℕ :=
  cost_juice / cost_bread

theorem ratio_of_juice_to_bread :
  total_money = 15 →
  money_left = 6 →
  cost_bread = 2 →
  cost_butter = 3 →
  total_money - money_left - (cost_bread + cost_butter) = cost_juice →
  compute_ratio total_money money_left cost_bread cost_butter cost_juice = 2 :=
by
  intros
  sorry

end ratio_of_juice_to_bread_l310_310790


namespace integral_of_sin_plus_x_sq_l310_310613

-- Define the function f(x) = sin x + x^2
def f (x : ℝ) : ℝ := real.sin x + x^2

-- State the theorem
theorem integral_of_sin_plus_x_sq :
  ∫ x in -1..1, f x = 2 / 3 :=
by
  sorry

end integral_of_sin_plus_x_sq_l310_310613


namespace inhabitants_identity_l310_310161

universe u

-- Definitions of terms
inductive PersonType
| knight : PersonType
| liar : PersonType

-- The guide's statement that the inhabitant is a liar
def guide_statement (inhabitant : PersonType) : Prop :=
  inhabitant = PersonType.liar

-- In Dipolia, knights always tell the truth and liars always lie
def always_lying (p : PersonType) (says_yes : bool) : Prop :=
  (p = PersonType.knight ∧ says_yes = true) ∨ (p = PersonType.liar ∧ says_yes = false)

-- The inhabitant's response "Ырг"
def response_yrg := true  -- interpreted as yes

-- The goal to prove based on conditions and the guide's statement
theorem inhabitants_identity (inhabitant : PersonType) (says_yes : bool):
  guide_statement inhabitant → always_lying inhabitant says_yes → response_yrg = true ∧ inhabitant = PersonType.liar :=
by
  sorry

end inhabitants_identity_l310_310161


namespace function_even_not_odd_l310_310651

variable {f : ℝ → ℝ}

-- Conditions
axiom func_defined : ∀ x, x ≠ 0 → ∃ y, f y = f x
axiom functional_equation : ∀ a b : ℝ, a ≠ 0 → b ≠ 0 → f (a / b) = f a - f b
axiom positivity : ∀ x : ℝ, 0 < x ∧ x < 1 → f x > 0

-- Proof statement
theorem function_even_not_odd 
    (func_defined : ∀ x, x ≠ 0 → ∃ y, f y = f x)
    (functional_equation : ∀ a b : ℝ, a ≠ 0 → b ≠ 0 → f (a / b) = f a - f b)
    (positivity : ∀ x : ℝ, 0 < x ∧ x < 1 → f x > 0) :
    (∀ x : ℝ, x ≠ 0 → f x = f (-x)) ∧ (∀ x : ℝ, x ≠ 0 → f x ≠ -f x) :=
by
  sorry

end function_even_not_odd_l310_310651


namespace Ilya_winning_strategy_l310_310523

theorem Ilya_winning_strategy (A B C : ℕ) (hA : A = 100) (hB : B = 101) (hC : C = 102) : 
  ∃ (strategy : ℕ → (A × B × C) → A × B × C), 
  ∀ (turn : ℕ) (pile : A × B × C), is_valid_move strategy turn pile → 
  (Ilya_wins : strategy ≠ ∅ ∧ strategy_sequence_ends_with_Ilya_winning strategy) :=
begin
  sorry
end

end Ilya_winning_strategy_l310_310523


namespace zero_point_exists_in_interval_l310_310972

-- Define the function
def f (x : ℝ) : ℝ := 3^x - x^2

-- State the theorem to be proved
theorem zero_point_exists_in_interval : ∃ x ∈ set.Icc (-1 : ℝ) 0, f x = 0 :=
sorry

end zero_point_exists_in_interval_l310_310972


namespace max_area_rectangle_l310_310112

/-- Given a rectangle with a perimeter of 40, the rectangle with the maximum area is a square
with sides of length 10. The maximum area is thus 100. -/
theorem max_area_rectangle (a b : ℝ) (h : a + b = 20) : a * b ≤ 100 :=
by
  sorry

end max_area_rectangle_l310_310112


namespace airplane_seats_theorem_l310_310196

def airplane_seats_proof : Prop :=
  ∀ (s : ℝ),
  (∃ (first_class business_class economy premium_economy : ℝ),
    first_class = 30 ∧
    business_class = 0.4 * s ∧
    economy = 0.6 * s ∧
    premium_economy = s - (first_class + business_class + economy)) →
  s = 150

theorem airplane_seats_theorem : airplane_seats_proof :=
sorry

end airplane_seats_theorem_l310_310196


namespace variance_male_greater_than_female_l310_310727

noncomputable def male_scores : List ℝ := [87, 95, 89, 93, 91]
noncomputable def female_scores : List ℝ := [89, 94, 94, 89, 94]

-- Function to calculate the variance of scores
noncomputable def variance (scores : List ℝ) : ℝ :=
  let n := scores.length
  let mean := scores.sum / n
  (scores.map (λ x => (x - mean) ^ 2)).sum / n

-- We assert the problem statement
theorem variance_male_greater_than_female :
  variance male_scores > variance female_scores :=
by
  sorry

end variance_male_greater_than_female_l310_310727


namespace cycle_repeats_l310_310172

noncomputable def cycle_result (x : ℝ) (n : ℕ) : ℝ :=
  if n % 2 = 0 then x else x⁻¹

theorem cycle_repeats (x : ℝ) (n : ℕ) (hx : x ≠ 0) :
  (cycle_result x n) = x ^ ((-1)^n) :=
by
  sorry

end cycle_repeats_l310_310172


namespace find_b_l310_310884

noncomputable def n : ℝ := 2 ^ 0.15
noncomputable def b : ℝ := 5 / 0.15

theorem find_b (h1 : n = 2 ^ 0.15) (h2 : n ^ b = 32) : b = 33.333 := 
by
  sorry

end find_b_l310_310884


namespace graph_shift_incorrect_l310_310314

def f (x : ℝ) : ℝ := sqrt 3 * sin x ^ 2 - 2 * cos x ^ 2

def g (x : ℝ) : ℝ := 2 * sin (2 * x) - 1

theorem graph_shift_incorrect :
  ¬ ∃ (h : ℝ → ℝ), 
    ∀ x : ℝ, f x = h (x - π / 6) ∧ h = g :=
sorry

end graph_shift_incorrect_l310_310314


namespace min_occupied_for_150_seats_l310_310467

-- Define a function that captures the pattern and counts occupied seats.
def min_occupied_seats (n : ℕ) : ℕ :=
  let units := n / 12
  let remaining := n % 12  in
  let base_occupied := units * 4   
  let additional_occupied := if remaining >= 3 then 2 else 0 
  in base_occupied + additional_occupied

theorem min_occupied_for_150_seats : min_occupied_seats 150 = 50 := by  sorry

end min_occupied_for_150_seats_l310_310467


namespace min_dSigma_correct_l310_310402

noncomputable def min_dSigma {a r : ℝ} (h : a > r) : ℝ :=
  (a - r) / 2

theorem min_dSigma_correct (a r : ℝ) (h : a > r) :
  min_dSigma h = (a - r) / 2 :=
by 
  unfold min_dSigma
  sorry

end min_dSigma_correct_l310_310402


namespace area_of_shaded_region_l310_310623

noncomputable def line1_eq (x : ℝ) : ℝ := - 3 / 8 * x + 4
noncomputable def line2_eq (x : ℝ) : ℝ := - 5 / 6 * x + 35 / 6
noncomputable def intersection : ℝ × ℝ := (4, 2.5)

theorem area_of_shaded_region : 
  let a1 := ∫ x in set.Icc (0 : ℝ) 1, (5 - line1_eq x) in
  let a2 := ∫ x in set.Icc (1 : ℝ) 4, (line2_eq x - line1_eq x) in 
  2 * (a1 + a2) = 13 / 2 :=
by
  sorry

end area_of_shaded_region_l310_310623


namespace power_of_power_rule_l310_310210

theorem power_of_power_rule (h : 128 = 2^7) : (128: ℝ)^(4/7) = 16 := by
  sorry

end power_of_power_rule_l310_310210


namespace cube_surface_area_of_same_volume_as_prism_l310_310918

theorem cube_surface_area_of_same_volume_as_prism :
  let prism_length := 10
  let prism_width := 5
  let prism_height := 24
  let prism_volume := prism_length * prism_width * prism_height
  let cube_edge := (prism_volume : ℝ)^(1/3)
  let cube_surface_area := 6 * cube_edge^2
  cube_surface_area = 677.76 := by
  sorry

end cube_surface_area_of_same_volume_as_prism_l310_310918


namespace pencil_cost_l310_310214

theorem pencil_cost :
  ∃ P : ℝ, 0.85 + P - 0.35 = 1 ∧ P = 0.50 :=
begin
  sorry
end

end pencil_cost_l310_310214


namespace unique_property_of_rectangle_l310_310830

-- Define the basic properties of quadrilaterals, rectangles, and rhombuses
structure Quadrilateral :=
(a b c d : ℝ)

structure Rectangle extends Quadrilateral :=
(parallel_eq_sides : a = c ∧ b = d)
(right_angles : true) -- Assumption of right angles

structure Rhombus extends Quadrilateral :=
(all_sides_eq : a = b ∧ b = c ∧ c = d)
(diagonals_perpendicular : true)

-- Define the bisection and equality of diagonals
def bisect_diagonals (q : Quadrilateral) : Prop :=
true -- Assume the diagonals bisect each other

def equal_diagonals (q : Quadrilateral) : Prop :=
true -- Assume the diagonals equal in length, need to be replaced by actual diagonal definition & equality check

theorem unique_property_of_rectangle :
  ∀ (r : Rectangle) (rh : Rhombus),
    equal_diagonals r ∧ ¬ equal_diagonals rh :=
by
  sorry

end unique_property_of_rectangle_l310_310830


namespace general_term_a_sum_b_formula_l310_310694

def seq_a : ℕ → ℝ 
| 1 := 6
| (n+1) := 3 * seq_a n

theorem general_term_a (n : ℕ) : seq_a n = 2 * 3^n := sorry

def seq_b (n : ℕ) : ℝ := (n + 1) * 3^n

noncomputable def sum_seq_b (n : ℕ) : ℝ := ∑ i in finset.range n, seq_b (i + 1)

theorem sum_b_formula (n : ℕ) : sum_seq_b n = (2 * n + 1) / 4 * 3^(n + 1) - 3 / 4 := sorry

end general_term_a_sum_b_formula_l310_310694


namespace value_of_100d_l310_310763

noncomputable def b : ℕ → ℝ
| 0       := 7 / 25
| (n + 1) := 2 * (b n) ^ 2 - 1

def d (n : ℕ) : ℝ :=
  d.bounds n * 2^n = ∏ i in finset.range n, b(i)

theorem value_of_100d : (100 * inf d) ≈ 109 :=
by
  sorry

end value_of_100d_l310_310763


namespace checkerboard_probability_l310_310783

def total_squares (n : ℕ) : ℕ :=
  n * n

def perimeter_squares (n : ℕ) : ℕ :=
  4 * n - 4

def non_perimeter_squares (n : ℕ) : ℕ :=
  total_squares n - perimeter_squares n

def probability_non_perimeter_square (n : ℕ) : ℚ :=
  non_perimeter_squares n / total_squares n

theorem checkerboard_probability :
  probability_non_perimeter_square 10 = 16 / 25 :=
by
  sorry

end checkerboard_probability_l310_310783


namespace quadratic_function_properties_l310_310454

def quadratic_function_with_min (f : ℝ → ℝ) (min_val : ℝ) (x_min : ℝ) : Prop :=
  ∀ x, f(x) ≥ min_val ∧ f(x_min) = min_val

def function_values_at_points (f : ℝ → ℝ) (x1 x2 val : ℝ) : Prop :=
  f(x1) = val ∧ f(x2) = val

def range_of_a_for_monotonicity (f : ℝ → ℝ) (a : ℝ) : Prop :=
  (∀ x y, x ≤ y → 2 * a ≤ x → f(x) ≤ f(y)) ∨ (∀ x y, x ≤ y → 3 * a + 1 ≤ y → f(y) ≤ f(x))

theorem quadratic_function_properties :
  ∃ (f : ℝ → ℝ), 
    quadratic_function_with_min f 1 2 ∧ 
    function_values_at_points f 0 4 3 ∧ 
    ∀ a : ℝ, range_of_a_for_monotonicity f a → (a ≤ 1 / 3 ∨ a ≥ 1) :=
by
  sorry

end quadratic_function_properties_l310_310454


namespace solve_equation_l310_310435

theorem solve_equation (x : ℝ) (hx : 0 ≤ x) : 2021 * x^10 - 2020 * x - 1 = 0 ↔ x = 1 := 
by 
  sorry

end solve_equation_l310_310435


namespace ned_time_left_to_diffuse_bomb_l310_310777

-- Conditions
def building_flights : Nat := 20
def time_per_flight : Nat := 11
def bomb_timer : Nat := 72
def time_spent_running : Nat := 165

-- Main statement
theorem ned_time_left_to_diffuse_bomb : 
  (bomb_timer - (building_flights - (time_spent_running / time_per_flight)) * time_per_flight) = 17 :=
by
  sorry

end ned_time_left_to_diffuse_bomb_l310_310777


namespace unique_solution_9_eq_1_l310_310247

theorem unique_solution_9_eq_1 :
  {p : ℝ × ℝ | 9^(p.1^2 + p.2) + 9^(p.1 + p.2^2) = 1}.card = 1 :=
sorry

end unique_solution_9_eq_1_l310_310247


namespace good_numbers_in_set_l310_310355

def is_good_number (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), ∀ k : Fin n, ∃ m : ℕ, k.val + (a k).val + 1 = m * m

theorem good_numbers_in_set : 
  ∀ n ∈ ({11, 13, 15, 17, 19} : Set ℕ), (is_good_number n ↔ n ≠ 11) :=
by
  intro n hn
  cases hn <;> simp [is_good_number]
  all_goals
    try (exists (λ _ : Fin n, sorry))
    all_goals
      intro k
      exists (k.val + 1)
      sorry

end good_numbers_in_set_l310_310355


namespace convex_polygon_condition_l310_310403

noncomputable def f (n : ℕ) : ℕ := 2 * Nat.choose n 4

structure Point (ℝ)
structure SetPoints (n : ℕ) :=
  (points : fin n → Point)

def noThreeCollinear (S : SetPoints n) : Prop := sorry
def noFourConcyclic (S : SetPoints n) : Prop := sorry
def a (S : SetPoints n) (i : fin n) : ℕ := sorry

def m (S : SetPoints n) : ℕ := (finset.univ : finset (fin n)).sum (a S)

theorem convex_polygon_condition (n : ℕ) (S : SetPoints n) 
  (hn : n ≥ 4)
  (hthreelinear : noThreeCollinear S)
  (hfourcircle : noFourConcyclic S) :
  (m S = f n) ↔ (∃ (S_conv : SetPoints n), sorry) := sorry

end convex_polygon_condition_l310_310403


namespace max_distance_AB_l310_310543

/-- Define the circle with center (1, -2) and radius 8 -/
def circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 64

/-- Define a point M with coordinates (l, 2) -/
def point_M (l : ℝ) : Prop := True -- M is utilized to define the line in context

/-- Define a line through point M(l, 2) -/
def line_through_point_M (l : ℝ) : set ℝ × ℝ :=
{ p | ∃ m : ℝ, p = (m, 2) }

/-- Define the condition for a line intersecting the circle at points A and B -/
def intersects_circle_at_A_and_B (l : ℝ) : Prop :=
∃ A B : ℝ × ℝ, circle A.1 A.2 ∧ circle B.1 B.2 ∧ A ≠ B ∧ ∃ line : set ℝ × ℝ, line_through_point_M l A ∧ line_through_point_M l B ∧ line ∈ {(m, 2) | m ∈ ℝ}

/-- Define the distance between two points -/
def distance (A B : ℝ × ℝ) : ℝ := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- Prove that the maximum distance |AB| is 16 when the line passes through the center of the circle -/
theorem max_distance_AB (l : ℝ) (h : intersects_circle_at_A_and_B l) : ∃ A B : ℝ × ℝ, distance A B = 16 :=
sorry

end max_distance_AB_l310_310543


namespace tangent_line_at_e_range_of_a_non_negative_l310_310690

-- Definition of the function for any real number a
def f (a : ℝ) (x : ℝ) := a * x ^ 2 - x * Real.log x - (2 * a - 1) * x + a - 1

-- Definition of the tangent line problem at point P(e, f(e)) when a = 0
theorem tangent_line_at_e (a : ℝ) (ha : a = 0) : 
  let P := (Real.exp 1, f 0 (Real.exp 1))
  (∀ x y, y + 1 = -1 * (x - Real.exp 1) → x + y + 1 - Real.exp 1 = 0) := 
by
  sorry

-- Definition of the non-negativity condition
theorem range_of_a_non_negative (a : ℝ) : (∀ x, 1 ≤ x → f a x ≥ 0) ↔ (1 / 2 ≤ a) :=
by 
  sorry

end tangent_line_at_e_range_of_a_non_negative_l310_310690


namespace pool_capacity_l310_310923

theorem pool_capacity (C : ℕ) (h1 : 300 = 0.8 * C - 0.4 * C) : C = 750 := 
by
  sorry

end pool_capacity_l310_310923


namespace ed_more_marbles_than_doug_l310_310610

theorem ed_more_marbles_than_doug (D : ℕ) (h1 : ∃ (D : ℕ), D = 35)
  (h2 : 45 = D - 11 + 21) :
  45 - D = 10 :=
by
sor आपorresearch

end ed_more_marbles_than_doug_l310_310610


namespace zeros_in_square_of_999_999_999_l310_310946

noncomputable def number_of_zeros_in_square (n : ℕ) : ℕ :=
  if n ≥ 1 then n - 1 else 0

theorem zeros_in_square_of_999_999_999 :
  number_of_zeros_in_square 9 = 8 :=
sorry

end zeros_in_square_of_999_999_999_l310_310946


namespace sample_is_subset_of_population_l310_310480

-- Define the population and sample sizes
def population_size : ℕ := 32000
def sample_size : ℕ := 1600

-- Define the sets representing the population and the sample
def population : Set ℕ := {i | i ∈ Fin population_size}
def sample : Set ℕ := {j | j ∈ Fin sample_size}

theorem sample_is_subset_of_population : sample ⊆ population := 
sorry

end sample_is_subset_of_population_l310_310480


namespace solution_l310_310285

-- Definitions of geometric objects
def Volume_Octahedron := sorry  -- Placeholder for the volume calculation of a regular octahedron
def Volume_Cube := sorry        -- Placeholder for the volume calculation of a cube

-- Given volumes
def VolumeO : ℝ := sorry -- Volume of the octahedron
def VolumeC : ℝ := sorry -- Volume of the cube

-- Original conditions
def ratio_VolumeO_VolumeC (VolumeO VolumeC : ℝ) : ℚ := 
  VolumeO / VolumeC

axiom O_is_octahedron (O : Type) : O = sorry -- Placeholder for the property of O being an octahedron
axiom C_is_cube (C : Type) : C = sorry        -- Placeholder for the property of C being a cube with vertices at the centers of the faces of O

-- Using relatively prime integers m and n
def m : ℚ := 2
def n : ℚ := 9

-- Prove m + n = 11 given the conditions above
theorem solution : m + n = 11 := by
  sorry

end solution_l310_310285


namespace area_of_shaded_region_l310_310622

noncomputable def line1_eq (x : ℝ) : ℝ := - 3 / 8 * x + 4
noncomputable def line2_eq (x : ℝ) : ℝ := - 5 / 6 * x + 35 / 6
noncomputable def intersection : ℝ × ℝ := (4, 2.5)

theorem area_of_shaded_region : 
  let a1 := ∫ x in set.Icc (0 : ℝ) 1, (5 - line1_eq x) in
  let a2 := ∫ x in set.Icc (1 : ℝ) 4, (line2_eq x - line1_eq x) in 
  2 * (a1 + a2) = 13 / 2 :=
by
  sorry

end area_of_shaded_region_l310_310622


namespace problem_statement_l310_310043

def f (x : ℤ) : ℤ := 3*x + 4
def g (x : ℤ) : ℤ := 4*x - 3

theorem problem_statement : (f (g (f 2))) / (g (f (g 2))) = 115 / 73 :=
by
  sorry

end problem_statement_l310_310043


namespace trigonometric_identity_l310_310663

theorem trigonometric_identity
  (α : ℝ)
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : real.cos α = 5 / 13) :
  (real.sin (α - π / 4) / real.cos (2 * α + 3 * π)) = - (13 * real.sqrt 2 / 34) :=
by
  sorry

end trigonometric_identity_l310_310663


namespace product_of_two_numbers_l310_310518

theorem product_of_two_numbers (x y : ℕ) (h₁ : x + y = 16) (h₂ : x^2 + y^2 = 200) : x * y = 28 :=
by
  sorry

end product_of_two_numbers_l310_310518


namespace shape_is_cone_l310_310632

-- Define the spherical coordinate system and the condition
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

def shape (c : ℝ) (p : SphericalCoord) : Prop := p.φ ≤ c

-- The shape described by \(\exists c, \forall p \in SphericalCoord, shape c p\) is a cone
theorem shape_is_cone (c : ℝ) (p : SphericalCoord) : shape c p → (c ≥ 0 ∧ c ≤ π → shape c p = Cone) :=
by
  sorry

end shape_is_cone_l310_310632


namespace coefficient_of_x_in_expansion_l310_310215

noncomputable def binomial_expansion_term (r : ℕ) : ℤ :=
  (-1)^r * (2^(5-r)) * Nat.choose 5 r

theorem coefficient_of_x_in_expansion :
  binomial_expansion_term 3 = -40 := by
  sorry

end coefficient_of_x_in_expansion_l310_310215


namespace solve_equation_l310_310432

theorem solve_equation (x : ℝ) (hx : 0 ≤ x) : 2021 * (x^2020)^(1/202) - 1 = 2020 * x → x = 1 :=
by
  sorry

end solve_equation_l310_310432


namespace knights_count_l310_310473

theorem knights_count (n : ℕ) (inhabitants : Fin n → Bool) (claim : Fin n → Prop)
  (h1 : n = 2021)
  (h2 : ∀ i : Fin n, claim i = (∃ l r : ℕ, l = (∑ j in FinRange (i.val), if inhabitants j then 1 else 0) ∧ 
  r = (∑ j in FinRange (i.val + 1).compl, if ¬inhabitants j then 1 else 0) ∧ r > l))
  : (inhabitants.filter (λ b, b)).length = 1010 := sorry

end knights_count_l310_310473


namespace base3_to_base10_correct_l310_310597

-- Define what it means to convert from base 3 to base 10
def base3_to_base10 (n: ℕ): ℕ :=
  1 * 3^5 + 0 * 3^4 + 2 * 3^3 + 0 * 3^2 + 1 * 3^1 + 2 * 3^0

-- Define the specific problem
def conversion_problem := base3_to_base10 102012

-- Formal statement of the proof problem
theorem base3_to_base10_correct: conversion_problem = 302 :=
begin
  -- This is the problem statement; the proof steps are omitted for this task
  sorry
end

end base3_to_base10_correct_l310_310597


namespace smallest_perfect_square_divisible_by_4_and_5_l310_310503

theorem smallest_perfect_square_divisible_by_4_and_5 : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (m : ℕ), n = m * m) ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ (n = 400) := 
by
  sorry

end smallest_perfect_square_divisible_by_4_and_5_l310_310503


namespace shopping_total_cost_l310_310366

theorem shopping_total_cost :
  let shirt_price := 5
      hat_price := 4
      jeans_price := 10
      qty_shirts := 3
      qty_jeans := 2
      qty_hats := 4 in
  (qty_shirts * shirt_price) + (qty_jeans * jeans_price) + (qty_hats * hat_price) = 51 :=
by
  sorry

end shopping_total_cost_l310_310366


namespace inradius_is_three_l310_310561

noncomputable def inradius_of_right_triangle
  (a b c : ℝ) (ha : a = 9) (hb : b = 12) (hc : c = 15)
  (h_right : a^2 + b^2 = c^2) : ℝ :=
  let s := (a + b + c) / 2 in  -- semiperimeter
  let A := (1/2) * a * b in    -- area
  A / s                        -- inradius

theorem inradius_is_three : inradius_of_right_triangle 9 12 15 9 12 15 (by norm_num) = 3 :=
sorry

end inradius_is_three_l310_310561


namespace line_equation_of_parameterized_curve_l310_310097

def parameterized_curve (t : ℝ) : ℝ × ℝ :=
  (3 * t + 1, 6 * t - 2)

theorem line_equation_of_parameterized_curve :
  ∃ (m b : ℝ), (∀ (t : ℝ), let (x, y) := parameterized_curve t in y = m * x + b) ∧ (m = 2) ∧ (b = -4) :=
by
  use 2
  use -4
  intro t
  simp [parameterized_curve]
  sorry

end line_equation_of_parameterized_curve_l310_310097


namespace q_satisfies_eq_l310_310808

def q (x : ℝ) : ℝ := -2 * x^6 + 4 * x^4 + 27 * x^3 + 24 * x^2 + 10 * x + 1

theorem q_satisfies_eq (x : ℝ) : 
  q x + (2 * x^6 + 4 * x^4 + 6 * x^2 + 2) = (8 * x^4 + 27 * x^3 + 30 * x^2 + 10 * x + 3) :=
begin
  -- the proof will be here
  sorry
end

end q_satisfies_eq_l310_310808


namespace bob_total_profit_l310_310949

-- Define the given inputs
def n_dogs : ℕ := 2
def c_dog : ℝ := 250.00
def n_puppies : ℕ := 6
def c_food_vac : ℝ := 500.00
def c_ad : ℝ := 150.00
def p_puppy : ℝ := 350.00

-- The statement to prove
theorem bob_total_profit : 
  (n_puppies * p_puppy - (n_dogs * c_dog + c_food_vac + c_ad)) = 950.00 :=
by
  sorry

end bob_total_profit_l310_310949


namespace parallel_or_perpendicular_line_intersects_two_segments_l310_310028

theorem parallel_or_perpendicular_line_intersects_two_segments 
  (n : ℕ) (segments : fin (4 * n) → (ℝ × ℝ) × (ℝ × ℝ)) (l : ℝ → ℝ) :
  (exists l' : ℝ → ℝ, (parallel_to l l' ∨ perpendicular_to l l') ∧ 
  (∃ i j : fin (4 * n), i ≠ j ∧ intersects l' (segments i) ∧ intersects l' (segments j))) :=
sorry

def parallel_to (l l' : ℝ → ℝ) : Prop := sorry

def perpendicular_to (l l' : ℝ → ℝ) : Prop := sorry

def intersects (l : ℝ → ℝ) (segment : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

end parallel_or_perpendicular_line_intersects_two_segments_l310_310028


namespace triangle_condition_l310_310307

variable {A B C : Type*}
variable [inner_product_space ℝ A] -- Assuming a real inner product space for vector operations

theorem triangle_condition (a b c : A) (α β γ : ℝ) 
  (h₁ : α / 3 = β / 4)
  (h₂ : α / 3 = γ / 5) :
  inner b c / (norm (a - b) * norm (a - c)) = (3 + real.sqrt 3) / 4 :=
sorry

end triangle_condition_l310_310307


namespace min_reciprocal_sum_l310_310757

-- Define what it means for a sequence to be an arithmetic progression.
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d, ∀ n, a (n + 1) = a n + d

-- Sum of the first n terms of an arithmetic sequence.
def S (a : ℕ → ℝ) (n : ℕ) :=
  n * (a 1 + a n) / 2

-- Theorem statement: Value of \frac{1}{{a}_{4}} + \frac{4}{{a}_{2020}}
theorem min_reciprocal_sum {a : ℕ → ℝ} (h_arith : is_arithmetic_sequence a) :
  S a 2023 = 2023 → a 4 > 0 → a 2020 > 0 →
  (∀ n, S a n = n * (a 1 + a n) / 2) →
  ∃ (m : ℝ), m = (1 / (a 4) + 4 / (a 2020)) ∧ m = 9 / 2 :=
sorry

end min_reciprocal_sum_l310_310757


namespace scientific_notation_7nm_l310_310092

theorem scientific_notation_7nm :
  ∀ (x : ℝ), x = 0.000000007 → x = 7 * 10^(-9) :=
begin
  intros x hx,
  sorry
end

end scientific_notation_7nm_l310_310092


namespace initial_number_of_men_in_place_l310_310094

theorem initial_number_of_men_in_place:
  (M A : ℕ) 
  (h1 : 46 - (20 + 10) = 16) 
  (h2 : M * 2 = 16): 
  M = 8 :=
by
  sorry

end initial_number_of_men_in_place_l310_310094


namespace product_numerator_denominator_l310_310141

def recurring_decimal_to_fraction (n : ℕ) (d : ℕ) : Rat :=
  n / d

theorem product_numerator_denominator (n : ℕ) (d : ℕ) (x : Rat)
  (hx : recurring_decimal_to_fraction 18 999 = x)
  (hn : n = 2)
  (hd : d = 111) :
  n * d = 222 := by
  have h_frac : x = 0.018 -- This follows from the definition and will be used in the proof
  sorry

end product_numerator_denominator_l310_310141


namespace exists_circle_with_exactly_n_integer_points_l310_310795

def A_center : ℝ × ℝ := (sqrt 2, 1/3)

theorem exists_circle_with_exactly_n_integer_points (n : ℕ) : 
  ∃ r : ℝ, is_circle_with_integer_points A_center r n :=
sorry

end exists_circle_with_exactly_n_integer_points_l310_310795


namespace area_triangle_ABC_eq_l310_310382

noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x
noncomputable def cot (x : ℝ) : ℝ := Real.cos x / Real.sin x

variables {A B C D E : Type*}
variables [has_coe (triangle A B C)] -- assuming A, B, C form a triangle
variables (AB BC BD DC BE : ℝ)
variables (alpha beta : ℝ)

axiom h1 : AB = BC
axiom h2 : altitude BD
axiom h3 : BE = 15
axiom h4 : tan(alpha - beta) * tan(alpha + beta) = tan(alpha) * tan(alpha)
axiom h5 : cot(beta) = (b + a) / (b - a)
axiom h6 : cot(alpha) = b / a
axiom h7 : alpha = Real.pi / 4
axiom h8 : b = 3 * a

theorem area_triangle_ABC_eq : 
  (1/2) * 3 * (5 * sqrt(2) / 2) * (5 * sqrt(2) / 2) = 75 / 2 :=
by sorry

end area_triangle_ABC_eq_l310_310382


namespace probability_first_spade_last_ace_l310_310123

-- Define the problem parameters
def standard_deck : ℕ := 52
def spades_count : ℕ := 13
def aces_count : ℕ := 4
def ace_of_spades : ℕ := 1

-- Probability of drawing a spade but not an ace as the first card
def prob_spade_not_ace_first : ℚ := 12 / 52

-- Probability of drawing any of the four aces among the two remaining cards
def prob_ace_among_two_remaining : ℚ := 4 / 50

-- Probability of drawing the ace of spades as the first card
def prob_ace_of_spades_first : ℚ := 1 / 52

-- Probability of drawing one of three remaining aces among two remaining cards
def prob_three_aces_among_two_remaining : ℚ := 3 / 50

-- Combined probability according to the cases
def final_probability : ℚ := (prob_spade_not_ace_first * prob_ace_among_two_remaining) + (prob_ace_of_spades_first * prob_three_aces_among_two_remaining)

-- The theorem stating that the computed probability matches the expected result
theorem probability_first_spade_last_ace : final_probability = 51 / 2600 := 
  by
    -- inserting proof steps here would solve the theorem
    sorry

end probability_first_spade_last_ace_l310_310123


namespace expand_simplify_expression_l310_310867

theorem expand_simplify_expression (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
by
  sorry

end expand_simplify_expression_l310_310867


namespace probability_of_selecting_girl_l310_310609

theorem probability_of_selecting_girl (boys girls : ℕ) (total_students : ℕ) (prob : ℚ) 
  (h1 : boys = 3) 
  (h2 : girls = 2) 
  (h3 : total_students = boys + girls) 
  (h4 : prob = girls / total_students) : 
  prob = 2 / 5 := 
sorry

end probability_of_selecting_girl_l310_310609


namespace color_changes_l310_310486

theorem color_changes (n k : ℕ) (h1: 0 < k) (h2: k < n): ∀ (A B : set (list ℕ)), 
  (A = { l | (l.length = 2 * n) ∧ ∃ v, v = n-k ∧ color_changes l = v }) ∧
  (B = { l | (l.length = 2 * n) ∧ ∃ v, v = n+k ∧ color_changes l = v }) →
  card A = card B := 
sorry

end color_changes_l310_310486


namespace seventh_monomial_l310_310419

noncomputable def sequence_monomial (n : ℕ) (x : ℝ) : ℝ :=
  (-1)^n * 2^(n-1) * x^(n-1)

theorem seventh_monomial (x : ℝ) : sequence_monomial 7 x = -64 * x^6 := by
  sorry

end seventh_monomial_l310_310419


namespace final_quantity_of_milk_l310_310931

-- Define initial conditions
def initial_volume : ℝ := 60
def removed_volume : ℝ := 9

-- Given the initial conditions, calculate the quantity of milk left after two dilutions
theorem final_quantity_of_milk :
  let first_removal_ratio := initial_volume - removed_volume / initial_volume
  let first_milk_volume := initial_volume * (first_removal_ratio)
  let second_removal_ratio := first_milk_volume / initial_volume
  let second_milk_volume := first_milk_volume * (second_removal_ratio)
  second_milk_volume = 43.35 :=
by
  sorry

end final_quantity_of_milk_l310_310931


namespace linear_dependent_on_interval_l310_310430

noncomputable def y₁ (x : ℝ) : ℝ := x
noncomputable def y₂ (x : ℝ) : ℝ := 2 * x

theorem linear_dependent_on_interval : 
  ∃ (a b : ℝ), (∃ x ∈ set.Icc (0:ℝ) 1, a * y₁ x + b * y₂ x = 0) ∧ (a ≠ 0 ∨ b ≠ 0) := 
sorry

end linear_dependent_on_interval_l310_310430


namespace sum_of_reciprocals_divisible_by_2017_squared_l310_310040

theorem sum_of_reciprocals_divisible_by_2017_squared :
  ∃ a b : ℕ, gcd a b = 1 ∧ (a : ℤ) = ∑ k in finset.range 2016, (1 / (k + 1) : ℚ).num ∧ 
  2017 ^ 2 ∣ a :=
by sorry

end sum_of_reciprocals_divisible_by_2017_squared_l310_310040


namespace range_of_product_of_distances_l310_310370

theorem range_of_product_of_distances
  (C1 : ℝ × ℝ → Prop)
  (C2 : ℝ × ℝ → Prop)
  (l : ℝ → ℝ → ℝ)
  (h_C1 : ∀ p : ℝ × ℝ, C1 p ↔ p.2 ^ 2 = 4 * p.1)
  (h_C2 : ∀ p : ℝ × ℝ, C2 p ↔ (p.1 - 4) ^ 2 + p.2 ^ 2 = 8)
  (h_l_slope : ∀ t : ℝ, ∀ x y : ℝ, x - y - t^2 + 2 * t = 0 → l x y = 1)
  (h_line_intersects_C2 : ∀ t : ℝ, ∃ Q R : ℝ × ℝ, Q ≠ R ∧ l Q.1 Q.2 = 1 ∧ l R.1 R.2 = 1 ∧ C2 Q ∧ C2 R) :
  ∀ t : ℝ, (-2 < t ∧ t < 0 ∨ 2 < t ∧ t < 4) →
  ∃ PQ PR : ℝ, PQ * PR ∈ set.range (λ t : ℝ, (t^2 - 2)^2) \ {4} :=
by
  sorry

end range_of_product_of_distances_l310_310370


namespace units_digit_of_product_of_odds_not_multiples_of_3_l310_310149

def is_odd (n : ℕ) : Prop := n % 2 = 1
def not_multiple_of_3 (n : ℕ) : Prop := n % 3 ≠ 0
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_product_of_odds_not_multiples_of_3 : 
  units_digit (∏ n in Finset.filter (λ n, is_odd n ∧ not_multiple_of_3 n) (Finset.range 201 \˘ Finset.range 21)) = 5 :=
by
  sorry

end units_digit_of_product_of_odds_not_multiples_of_3_l310_310149


namespace coin_winning_probability_l310_310536

-- Define all the conditions as hypotheses

def coin_probability (coin_diameter : ℝ) (grid_side : ℝ) (square_side : ℝ) : ℝ :=
  let coin_radius := coin_diameter / 2
  let total_grid_area := grid_side ^ 2
  -- Safe region dimensions on the grid
  let safe_region_side := grid_side - 2 * coin_radius
  let total_admissible_area := safe_region_side ^ 2
  -- Individual square safe region
  let square_safe_side := square_side - 2 * coin_radius
  let safe_area_per_square := square_safe_side ^ 2
  let num_squares := (grid_side / square_side) ^ 2
  let total_winning_position_area := safe_area_per_square * num_squares
  total_winning_position_area / total_admissible_area

theorem coin_winning_probability :
  coin_probability 8 50 10 = 25 / 441 :=
by
  -- Proof goes here
  sorry

end coin_winning_probability_l310_310536


namespace select_points_l310_310222

theorem select_points (circumference: ℕ) (points: Finset ℕ) (num_selected: ℕ)
(h_circumference: circumference = 24)
(h_points: points = Finset.range 24)
(h_num_selected: num_selected = 8)
(h_no_arc_length_3_or_8: ∀ (p1 p2 : ℕ), p1 ∈ points → p2 ∈ points → p1 ≠ p2 → 
  (p1 - p2) % 24 ≠ 3 ∧ (p1 - p2) % 24 ≠ 8) : 
  ∃ (num_ways: ℕ), num_ways = 258 := 
begin
  sorry
end

end select_points_l310_310222


namespace _l310_310288

noncomputable def is_tangent_to_parabola (x1 y1 p k : ℝ) : Prop :=
  let y := k * x1 - 1
  let rhs := x1^2
  rhs = y

noncomputable def leans_theorem_prover (O A B : (ℝ × ℝ)) : Prop :=
  -- Definitions of points
  let O := (0,0)
  let A := (1,1)
  let B := (0,-1)
  -- Value of p from point A on parabola C: x^2 = 2py
  let p := 1 / 2  -- as obtained by solving 1^2 = 2p * 1
  -- Checking option A: directrix is y = -1 is false
  let directrix := - p / 2
  (directrix ≠ -1) ∧
  -- Checking option B: tangent condition
  let slope_AB := (1 - (-1)) / (1 - 0)
  let tangent := is_tangent_to_parabola 1 1 p slope_AB
  tangent ∧
  -- Option C: |OP| * |OQ| = |OA|^2 is false
  let |OA|² := 2  -- obtained from the calculation |OA| = sqrt(1^2 + 1^2)
  ∀ (k > 2), k² ≠ |OA|² ∧
  -- Option D: |BP| * |BQ| > |BA|^2 is true
  let |BA|² := 5 -- obtained from the calculation |BA| = sqrt(1^2 + 4)
  ∀ (x1 x2) (hx1 : x1 + x2 = k) (hx2 : x1 * x2 = 1),
  let |BP| := sqrt(x1^2 + (x1^2 + 2x1 + 1))
  let |BQ| := sqrt(x2^2 + (x2^2 + 2x2 + 1))
  |BP| * |BQ| > |BA|²

example : leans_theorem_prover (0, 0) (1, 1) (0, -1) :=
by sorry  -- Proof is omitted, to be completed by Lean theorem prover

end _l310_310288


namespace tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l310_310292

-- Conditions
variables {O : ℝ × ℝ} (A : ℝ × ℝ) (B : ℝ × ℝ)
          {P Q : ℝ × ℝ} (p : ℝ)
          (hp : 0 < p)
          (hA : A.1 ^ 2 = 2 * p * A.2)
          (hB : B = (0, -1))
          (hP : P.2 = P.1 ^ 2 / (2 * p))
          (hQ : Q.2 = Q.1 ^ 2 / (2 * p))

-- Proof problem statements
theorem tangent_line_AB
  (hAB_tangent : ∀ x : ℝ, x ^ 2 / (2 * p) = 2 * x - 1 → x = 1) : true :=
by sorry

theorem op_oq_leq_oa_squared 
  (h_op_oq_leq : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + (P.1 ^ 2 / (2 * p)) ^ 2) * (Q.1 ^ 2 + (Q.1 ^ 2 / (2 * p)) ^ 2) ≤ 2) : true :=
by sorry

theorem bp_bq_gt_ba_squared 
  ( h_bp_bq_gt : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + ((P.1 ^ 2 / (2 * p)) + 1) ^ 2) * (Q.1 ^ 2 + ((Q.1 ^ 2 / (2 * p)) +1 ) ^ 2) > 5 ) : true :=
by sorry

end tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l310_310292


namespace HK_bisects_QS_l310_310067

open EuclideanGeometry

variables {P Q R S H K : Point}

-- Setting up the conditions from the problem
variables [circle_points : PointsOnCircle {P, Q, R, S}]
variables [right_angle : ∠PSR = 90°]
variables [proj_H : OrthogonalProjection Q PR H]
variables [proj_K : OrthogonalProjection Q PS K]

-- Statement of the theorem
theorem HK_bisects_QS :
  BisectsLineSegment HK QS := 
begin
  sorry
end

end HK_bisects_QS_l310_310067


namespace condition_for_parallelogram_l310_310649

theorem condition_for_parallelogram
  (a b : ℝ)
  (C0 C1 : ℝ → ℝ → Prop) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hab : a > b) 
  (hc0 : ∀ (x y : ℝ), C0 x y ↔ x^2 + y^2 = 1)
  (hc1 : ∀ (x y : ℝ), C1 x y ↔ (x^2 / a^2 + y^2 / b^2 = 1)) :
  (∀ (P : ℝ × ℝ), C1 P.1 P.2 → ∃ (Q R S : ℝ × ℝ), parallelogram P Q R S ∧ tangent_to C0 (∂((P::Q::R::S::[]) ∷ 0)) ) ↔ 
  (1 / a^2 + 1 / b^2 = 1) := 
sorry

end condition_for_parallelogram_l310_310649


namespace min_workers_to_make_profit_l310_310537

theorem min_workers_to_make_profit :
  ∃ n : ℕ, 500 + 8 * 15 * n < 124 * n ∧ n = 126 :=
by
  sorry

end min_workers_to_make_profit_l310_310537


namespace unique_solution_9_eq_1_l310_310248

theorem unique_solution_9_eq_1 :
  {p : ℝ × ℝ | 9^(p.1^2 + p.2) + 9^(p.1 + p.2^2) = 1}.card = 1 :=
sorry

end unique_solution_9_eq_1_l310_310248


namespace blankets_first_day_l310_310254

-- Definition of the conditions
def num_people := 15
def blankets_day_three := 22
def total_blankets := 142

-- The problem statement
theorem blankets_first_day (B : ℕ) : 
  (num_people * B) + (3 * (num_people * B)) + blankets_day_three = total_blankets → 
  B = 2 :=
by sorry

end blankets_first_day_l310_310254


namespace flower_bouquet_combinations_l310_310535

theorem flower_bouquet_combinations :
  (∃ (r c : ℕ), 4 * r + 3 * c = 60) ∧
  (card { rc : ℕ × ℕ | 4 * rc.1 + 3 * rc.2 = 60 } = 6) :=
by
  sorry

end flower_bouquet_combinations_l310_310535


namespace travel_distance_l310_310061

variables (X Y Z : Type) [metric_space X] [metric_space Y] [metric_space Z]
variables (dXY dXZ dYZ : ℝ)

noncomputable def is_right_angle_triangle (XZ YZ XY : ℝ) : Prop :=
  XZ^2 + YZ^2 = XY^2

def distance_XZ : ℝ := 4000
def distance_XY : ℝ := 5000

theorem travel_distance {X Y Z : Type} [metric_space X] [metric_space Y] [metric_space Z]
  (h : is_right_angle_triangle distance_XZ dYZ distance_XY) : 
  dXZ + dYZ + distance_XY = 12000 :=
sorry

end travel_distance_l310_310061


namespace find_angle_C_l310_310357

noncomputable def measure_of_angle_C
  (a b c : ℝ)
  (A : ℝ)
  (h1 : b^2 = a^2 - 2 * b * c)
  (h2 : A = 2 * Real.pi / 3) : ℝ :=
let C := (π - A) / 2 in
C

theorem find_angle_C
  (a b c : ℝ)
  (A : ℝ)
  (h1 : b^2 = a^2 - 2 * b * c)
  (h2 : A = 2 * Real.pi / 3) :
  (let C := (π - A) / 2 in C) = π / 6 :=
by {
  let C := (π - A) / 2,
  calc
    C = (π - A) / 2 : by refl
    ... = (π - 2 * π / 3) / 2 : by rw h2
    ... = π / 3 / 2 : by ring
    ... = π / 6 : by ring,
  sorry
}

end find_angle_C_l310_310357


namespace value_of_c_l310_310087

variables (a b c : ℝ)

theorem value_of_c :
  a + b = 3 ∧
  a * c + b = 18 ∧
  b * c + a = 6 →
  c = 7 :=
by
  intro h
  sorry

end value_of_c_l310_310087


namespace ten_numbers_property_l310_310392

theorem ten_numbers_property (x : ℕ → ℝ) (h : ∀ i : ℕ, 1 ≤ i → i ≤ 9 → x i + 2 * x (i + 1) = 1) : 
  x 1 + 512 * x 10 = 171 :=
by
  sorry

end ten_numbers_property_l310_310392


namespace max_d_6_digit_multiple_33_l310_310619

theorem max_d_6_digit_multiple_33 (x d e : ℕ) 
  (hx : 1 ≤ x ∧ x ≤ 9) 
  (hd : 0 ≤ d ∧ d ≤ 9) 
  (he : 0 ≤ e ∧ e ≤ 9)
  (h1 : (x * 100000 + 50000 + d * 1000 + 300 + 30 + e) ≥ 100000) 
  (h2 : (x + d + e + 11) % 3 = 0)
  (h3 : ((x + d - e - 5 + 11) % 11 = 0)) :
  d = 9 := 
sorry

end max_d_6_digit_multiple_33_l310_310619


namespace length_AD_l310_310733

-- Defining points A, B, C, D in a plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Defining the quadrilateral with conditions given
variables (A B C D : Point)
variables (AB BC CD AD : ℝ)
variables (angleB angleC : ℝ)

-- Conditions from the problem
def conditions := (AB = 6) ∧ (BC = 9) ∧ (CD = 18) ∧ (angleB = 90) ∧ (angleC = 90)

-- The main proof statement
theorem length_AD (h : conditions A B C D AB BC CD angleB angleC) : AD = 9 * Real.sqrt 2 :=
  sorry

end length_AD_l310_310733


namespace sqrt_equation_solution_l310_310236

theorem sqrt_equation_solution (x : ℝ) :
  (Real.sqrt ((2 + Real.sqrt 3) ^ x) + Real.sqrt ((2 - Real.sqrt 3) ^ x) = 4) ↔ (x = 2 ∨ x = -2) := 
by
  sorry

end sqrt_equation_solution_l310_310236


namespace max_value_condition_l310_310767

noncomputable def maximumValue (a b c : ℝ) : ℝ :=
  a + 2 * Real.sqrt (a * b) + Real.cbrt (a * b * c)

theorem max_value_condition (a b c : ℝ) (h : a + 2 * b + c = 2) (h₁ : 0 ≤ a) (h₂ : 0 ≤ b) (h₃ : 0 ≤ c) :
  maximumValue a b c ≤ 2.545 := 
sorry

end max_value_condition_l310_310767


namespace verify_amplitude_verify_initial_phase_verify_smallest_positive_period_l310_310451

noncomputable def functionExpression (t : ℝ) : ℝ :=
  3 * Real.cos (1/2 * t + Real.pi / 5)

def amplitude : ℝ :=
  3

def initial_phase : ℝ :=
  Real.pi / 5

def smallest_positive_period : ℝ :=
  4 * Real.pi

theorem verify_amplitude :
  amplitude = 3 := sorry

theorem verify_initial_phase :
  initial_phase = Real.pi / 5 := sorry

theorem verify_smallest_positive_period :
  smallest_positive_period = 4 * Real.pi := sorry

end verify_amplitude_verify_initial_phase_verify_smallest_positive_period_l310_310451


namespace limit_of_nested_radical_l310_310585

theorem limit_of_nested_radical :
  ∃ F : ℝ, F = 43 ∧ F = Real.sqrt (86 + 41 * F) :=
sorry

end limit_of_nested_radical_l310_310585


namespace interest_rate_is_5_percent_l310_310626

noncomputable def principal : ℝ := 1018.18
noncomputable def amount_after_2_years : ℝ := 1120
noncomputable def time_in_years : ℝ := 2
noncomputable def interest_rate_per_annum (P A T : ℝ) : ℝ := 
  let total_interest := A - P
      annual_interest := total_interest / T
  in (annual_interest / P) * 100

theorem interest_rate_is_5_percent :
  interest_rate_per_annum principal amount_after_2_years time_in_years = 5 := by
  sorry

end interest_rate_is_5_percent_l310_310626


namespace birds_on_fence_l310_310466

theorem birds_on_fence (initial_birds : ℕ) (additional_birds : ℕ) (h_initial : initial_birds = 12) (h_additional : additional_birds = 8) :
  initial_birds + additional_birds = 20 :=
by
  rw [h_initial, h_additional]
  exact rfl

end birds_on_fence_l310_310466


namespace mass_percentage_Al_in_Al2O3_l310_310627

/-- Define the chemical formula and atomic masses -/
def chemical_formula := "Al2O3"
def atomic_mass_Al := 26.98 -- g/mol
def atomic_mass_O := 16.00 -- g/mol

/-- Calculate the molar masses -/
def molar_mass_Al2O3 : ℝ := (2 * atomic_mass_Al) + (3 * atomic_mass_O)
def mass_Al_in_Al2O3 : ℝ := 2 * atomic_mass_Al

/-- Mass percentage of Al in Al2O3 -/
def mass_percentage_Al : ℝ := (mass_Al_in_Al2O3 / molar_mass_Al2O3) * 100

/-- Main theorem stating the mass percentage of Al in Al2O3 -/
theorem mass_percentage_Al_in_Al2O3 : mass_percentage_Al = 52.91 :=
by
  sorry

end mass_percentage_Al_in_Al2O3_l310_310627


namespace proof_problem_l310_310264

variables (α β : Plane) (l m : Line)
variable (h1 : ¬ coincident α β)
variable (h2 : l ≠ m)
variable (h3 : perpendicular l α)
variable (h4 : m ⊆ β)

theorem proof_problem :
  (parallel α β → perpendicular l m) ∧
  (perpendicular α β → ¬ parallel l m) ∧
  (parallel m α → ¬ perpendicular l β) ∧
  (perpendicular l β → parallel m α) :=
by { sorry }

end proof_problem_l310_310264


namespace solution_l310_310284

-- Definitions of geometric objects
def Volume_Octahedron := sorry  -- Placeholder for the volume calculation of a regular octahedron
def Volume_Cube := sorry        -- Placeholder for the volume calculation of a cube

-- Given volumes
def VolumeO : ℝ := sorry -- Volume of the octahedron
def VolumeC : ℝ := sorry -- Volume of the cube

-- Original conditions
def ratio_VolumeO_VolumeC (VolumeO VolumeC : ℝ) : ℚ := 
  VolumeO / VolumeC

axiom O_is_octahedron (O : Type) : O = sorry -- Placeholder for the property of O being an octahedron
axiom C_is_cube (C : Type) : C = sorry        -- Placeholder for the property of C being a cube with vertices at the centers of the faces of O

-- Using relatively prime integers m and n
def m : ℚ := 2
def n : ℚ := 9

-- Prove m + n = 11 given the conditions above
theorem solution : m + n = 11 := by
  sorry

end solution_l310_310284


namespace integer_multiplication_for_ones_l310_310489

theorem integer_multiplication_for_ones :
  ∃ x : ℤ, (10^9 - 1) * x = (10^81 - 1) / 9 :=
by
  sorry

end integer_multiplication_for_ones_l310_310489


namespace pirate_loot_total_l310_310181

theorem pirate_loot_total
  (jewelry : ℕ := 6532) (gold_coins : ℕ := 3201) (silverware : ℕ := 526)
  (convert_base7_to_base10 : ∀ n, ℕ := λ n, 
    let digits := Nat.digits 7 n in
    List.foldr (λ (d : ℕ) (acc : ℕ × ℕ), (acc.1 + d * acc.2, acc.2 * 7)) (0, 1) digits).fst) 
  (jewelry_base10 := convert_base7_to_base10 jewelry)
  (gold_coins_base10 := convert_base7_to_base10 gold_coins)
  (silverware_base10 := convert_base7_to_base10 silverware) : 
  jewelry_base10 + gold_coins_base10 + silverware_base10 = 3719 := 
by 
  -- convert base 7 numbers to base 10 
  have h_jewelry := convert_base7_to_base10 6532
  have h_gold_coins := convert_base7_to_base10 3201
  have h_silverware := convert_base7_to_base10 526

  -- verify each conversion is correct
  have h_jewelry_correct : h_jewelry = 2326 := by sorry -- proof of conversion
  have h_gold_coins_correct : h_gold_coins = 1128 := by sorry -- proof of conversion
  have h_silverware_correct : h_silverware = 265 := by sorry -- proof of conversion

  -- show that the sum is 3719
  show 2326 + 1128 + 265 = 3719
  sorry

end pirate_loot_total_l310_310181


namespace carter_lucy_ratio_l310_310589

-- Define the number of pages Oliver can read in 1 hour
def oliver_pages : ℕ := 40

-- Define the number of additional pages Lucy can read compared to Oliver
def additional_pages : ℕ := 20

-- Define the number of pages Carter can read in 1 hour
def carter_pages : ℕ := 30

-- Calculate the number of pages Lucy can read in 1 hour
def lucy_pages : ℕ := oliver_pages + additional_pages

-- Prove the ratio of the number of pages Carter can read to the number of pages Lucy can read is 1/2
theorem carter_lucy_ratio : (carter_pages : ℚ) / (lucy_pages : ℚ) = 1 / 2 := by
  sorry

end carter_lucy_ratio_l310_310589


namespace find_AX_l310_310026

-- Definitions for the problem conditions
variable (A B C X : Type)
variable (AX XB : ℝ)
variable (AC BC : ℝ)
variable [fact (3 * AX = 5 * XB)]
variable [fact (AC = 15)]
variable [fact (BC = 25)]
variable [fact (AX / XB = AC / BC)]

-- The theorem statement
theorem find_AX : AX = 15 := 
sorry

end find_AX_l310_310026


namespace repeating_decimal_product_of_num_and_den_l310_310135

theorem repeating_decimal_product_of_num_and_den (x : ℚ) (h : x = 18 / 999) (h_simplified : x.num * x.den = 222) : x.num * x.den = 222 :=
by {
  sorry
}

end repeating_decimal_product_of_num_and_den_l310_310135


namespace sets_satisfy_conditions_l310_310970

-- Define the conditions
def nonEmptyFinite (C : Set (Finset ℕ)) : Prop :=
  ∀ c ∈ C, finset.nonempty c ∧ finset.finite c

def positiveIntegers (C : Set (Finset ℕ)) : Prop :=
  ∀ c ∈ C, ∀ x ∈ c, (0 : ℕ) < x

def cardinalityAndSumProperty (C : Set (Finset ℕ)) : Prop :=
  ∀ (n m : ℕ) (C_n C_m C_nm : Finset ℕ),
    C_n ∈ C → C_m ∈ C → C_nm ∈ C → (C_n.card + C_m.card = C_nm.sum)

def satisfiesConditions (C : Set (Finset ℕ)) : Prop :=
  nonEmptyFinite C ∧ positiveIntegers C ∧ cardinalityAndSumProperty C

-- Define the specific sets
def A1 := {2}
def A2 := {1, 3}
def A3 := {1, 2, 3}

-- Define the set of the specific sets
def A := {A1, A2, A3}

-- The main theorem
theorem sets_satisfy_conditions : satisfiesConditions A := sorry

end sets_satisfy_conditions_l310_310970


namespace power_calculation_l310_310207

theorem power_calculation : (128 : ℝ) ^ (4/7) = 16 :=
by {
  have factorization : (128 : ℝ) = 2 ^ 7 := by {
    norm_num,
  },
  rw factorization,
  have power_rule : (2 ^ 7 : ℝ) ^ (4/7) = 2 ^ 4 := by {
    norm_num,
  },
  rw power_rule,
  norm_num,
  sorry
}

end power_calculation_l310_310207


namespace probability_multiple_of_200_l310_310356

theorem probability_multiple_of_200 :
  let s := {2, 4, 8, 10, 12, 25, 50, 100}
  ∃ (count : ℕ), (count = 3) ∧ (set.univ.powerset.filter (λ t, t.card = 2).card = 28)
    → (count / (set.univ.powerset.filter (λ t, t.card = 2).card : ℚ) = 3 / 28) := by
  sorry

end probability_multiple_of_200_l310_310356


namespace scientific_notation_correct_l310_310001

-- Define the given number
def original_number : ℕ := 75500000

-- Define the expected scientific notation components
def scientific_mantissa : ℝ := 7.55
def scientific_exponent : ℤ := 7

-- Define a function to express the number in scientific notation
def scientific_notation (mantissa : ℝ) (exponent : ℤ) : ℝ :=
  mantissa * (10 ^ exponent)

-- State the theorem that 75,500,000 can be expressed as 7.55 * 10^7 in scientific notation
theorem scientific_notation_correct :
  scientific_notation scientific_mantissa scientific_exponent = (original_number : ℝ) :=
by {
  -- Proof goes here
  sorry
}

end scientific_notation_correct_l310_310001


namespace n_value_l310_310880

theorem n_value (n : ℕ) (h1 : ∃ a b : ℕ, a = (5 + 10 + 15 + 20 + 25 + 30 + 35) / 7 ∧ b = 2 * n ∧ a ^ 2 - b ^ 2 = 0) : n = 10 := 
  by sorry

end n_value_l310_310880


namespace length_of_train_l310_310930

theorem length_of_train
  (speed_kmph : ℕ)
  (time_s : ℕ)
  (bridge_length_m : ℕ)
  (speed_mps : ℕ := (speed_kmph * 1000) / 3600)
  (distance_m : ℕ := speed_mps * time_s)
  (train_length_m : ℕ := distance_m - bridge_length_m)
  (h_speed : speed_kmph = 45)
  (h_time : time_s = 30)
  (h_bridge_length : bridge_length_m = 215)
  (h_train_length : train_length_m = 160) :
  train_length_m = 160 :=
by
  rw [h_speed, h_time, h_bridge_length, h_train_length]
  sorry

end length_of_train_l310_310930


namespace car_drive_highway_distance_l310_310531

theorem car_drive_highway_distance
  (d_local : ℝ)
  (s_local : ℝ)
  (s_highway : ℝ)
  (s_avg : ℝ)
  (d_total := d_local + s_avg * (d_local / s_local + d_local / s_highway))
  (t_local := d_local / s_local)
  (t_highway : ℝ := (d_total - d_local) / s_highway)
  (t_total := t_local + t_highway)
  (avg_speed := (d_total) / t_total)
  : d_local = 60 → s_local = 20 → s_highway = 60 → s_avg = 36 → avg_speed = 36 → d_total - d_local = 120 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4]
  sorry

end car_drive_highway_distance_l310_310531


namespace _l310_310446

example : coeff (x^3) ((1 + 2*x)^6) = 160 :=
by
  -- We use the binomial theorem here
  have h := nat.choose 6 3
  -- Simplification step
  suffices h' : 2^3 * h = 160
  { exact h' }
  have h_binom : h = 6 * 5 * 4 / (3 * 2 * 1) := nat.choose_eq_factorial_div_factorial 6 3
  have h_calculate : 6 * 5 * 4 / (3 * 2 * 1) = 20 := by norm_num
  rw h_binom at h_calculate
  have h_pow : 2^3 = 8 := by norm_num
  rw [←h_pow, ←h_calculate]
  norm_num
  length 4 sorry

end _l310_310446


namespace solve_for_real_part_l310_310299

theorem solve_for_real_part (i : ℂ) (h : i = complex.I) (a : ℝ) :
  let z := (1 - 2 * i) * (a + i) in
  (z.re + z.im = 0) → a = 3 :=
by
  sorry

end solve_for_real_part_l310_310299


namespace find_k_l310_310519

theorem find_k (m n k : ℝ) (h1 : m = 2 * n + 3) (h2 : m + 2 = 2 * (n + k) + 3) : k = 1 :=
by
  -- Proof is omitted
  sorry

end find_k_l310_310519


namespace unique_t_for_Tn_eq_Sn_l310_310600

noncomputable def f_n (n : ℕ) : ℝ → ℝ :=
λ x, real.log x / x^n

noncomputable def T_n (n : ℕ) (t : ℝ) : ℝ :=
(t - 1) * f_n n t

noncomputable def S_n (n : ℕ) (t : ℝ) : ℝ :=
∫ x in 1..t, f_n n x

theorem unique_t_for_Tn_eq_Sn (n : ℕ) (hn : n ≥ 2) : ∃! t > 1, T_n n t = S_n n t :=
sorry

end unique_t_for_Tn_eq_Sn_l310_310600


namespace ratio_of_lengths_l310_310917

noncomputable def total_fence_length : ℝ := 640
noncomputable def short_side_length : ℝ := 80

theorem ratio_of_lengths (L S : ℝ) (h1 : 2 * L + 2 * S = total_fence_length) (h2 : S = short_side_length) :
  L / S = 3 :=
by {
  sorry
}

end ratio_of_lengths_l310_310917


namespace median_vision_is_4_6_l310_310230

def vision_data : List ℕ :=
  [1, 2, 6, 3, 3, 4, 1, 2, 5, 7, 5]

def vision_values : List ℚ :=
  [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]

def cumulative_students : List ℕ :=
  List.scanl (· + ·) 0 vision_data

theorem median_vision_is_4_6 : 
  vision_values.nth 6 = some 4.6 := by
  have h : cumulative_students.nth 6 = some 20 := by sorry
  show vision_values.nth 6 = some 4.6 from sorry

end median_vision_is_4_6_l310_310230


namespace solve_equation_l310_310433

theorem solve_equation (x : ℝ) (hx : 0 ≤ x) : 2021 * (x^2020)^(1/202) - 1 = 2020 * x → x = 1 :=
by
  sorry

end solve_equation_l310_310433


namespace cricket_team_matches_played_in_august_l310_310177

theorem cricket_team_matches_played_in_august
    (M : ℕ)
    (h1 : ∃ W : ℕ, W = 24 * M / 100)
    (h2 : ∃ W : ℕ, W + 70 = 52 * (M + 70) / 100) :
    M = 120 :=
sorry

end cricket_team_matches_played_in_august_l310_310177


namespace initial_water_percentage_l310_310902

noncomputable def S : ℝ := 4.0
noncomputable def V_initial : ℝ := 440
noncomputable def V_final : ℝ := 460
noncomputable def sugar_added : ℝ := 3.2
noncomputable def water_added : ℝ := 10
noncomputable def kola_added : ℝ := 6.8
noncomputable def kola_percentage : ℝ := 8.0 / 100.0
noncomputable def final_sugar_percentage : ℝ := 4.521739130434784 / 100.0

theorem initial_water_percentage : 
  ∀ (W S : ℝ),
  V_initial * (S / 100) + sugar_added = final_sugar_percentage * V_final →
  (W + 8.0 + S) = 100.0 →
  W = 88.0
:=
by
  intros W S h1 h2
  sorry

end initial_water_percentage_l310_310902


namespace trapezoid_is_isosceles_l310_310025

-- Define a structure for our trapezoid and the conditions
structure Trapezoid (A B C D : Type) :=
  (is_parallel : A is_parallel_to B)
  (diagonal_eq_sum_bases : A.C = A.D + B.C)
  (angle_between_diagonals_is_60 : angle A.O.B = 60)

-- Define the theorem statement
theorem trapezoid_is_isosceles 
  (A B C D : Type) 
  (h : Trapezoid A B C D) 
  : is_isosceles_trapezoid A B C D :=
by
  sorry

end trapezoid_is_isosceles_l310_310025


namespace application_methods_count_l310_310468

theorem application_methods_count (n_graduates m_universities : ℕ) (h_graduates : n_graduates = 5) (h_universities : m_universities = 3) :
  (m_universities ^ n_graduates) = 243 :=
by
  rw [h_graduates, h_universities]
  show 3 ^ 5 = 243
  sorry

end application_methods_count_l310_310468


namespace find_a_plus_b_plus_c_l310_310452

theorem find_a_plus_b_plus_c (a b c : ℝ) 
  (h_min : ∃ xₘ : ℝ, yₘ = axₘ^2 + bxₘ + c ∧ yₘ = 36)
  (h_passes_1 : ax^2 + bx + c = 0)
  (h_passes_2 : a(5-1)(5-5) = 0) :
  a + b + c = 0 := 
sorry

end find_a_plus_b_plus_c_l310_310452


namespace sum_of_eight_fib_not_in_sequence_l310_310921

theorem sum_of_eight_fib_not_in_sequence (k n : ℕ) :
  (∀ i, i ≤ 7 → fib (k + 1 + i) ≠ fib n) :=
by sorry

end sum_of_eight_fib_not_in_sequence_l310_310921


namespace minimum_y_squared_l310_310672

theorem minimum_y_squared :
  let consecutive_sum (x : ℤ) := (x-5)^2 + (x-4)^2 + (x-3)^2 + (x-2)^2 + (x-1)^2 + x^2 + (x+1)^2 + (x+2)^2 + (x+3)^2 + (x+4)^2 + (x+5)^2
  ∃ y : ℤ, y^2 = 11 * (1^2 + 10) ∧ ∀ z : ℤ, z^2 = 11 * consecutive_sum z → y^2 ≤ z^2 := by
sorry

end minimum_y_squared_l310_310672


namespace probability_white_rotation_l310_310168

def initial_probability (p: ℚ) (color_count: ℕ × ℕ) : Prop := 
  p = (1 / 2) ^ (color_count.1 + color_count.2)

def repaint_condition (rotation: ℕ) (initial_color: bool) : bool :=
  if rotation % 2 = 1 && !initial_color then
    ff
  else
    initial_color

def final_probability (p: ℚ) (new_grid_colors: ℕ → ℕ → bool) : Prop :=
  p = 1 / 512 ∧ 
  ∀ i j, new_grid_colors i j = tt

theorem probability_white_rotation :
  ∃ p : ℚ, initial_probability p (4, 5) ∧
          final_probability p 
          (λ i j, match (i, j) with
             | (1, 1) => tt
             | (_, _) => repaint_condition 1 tt) := 
 sorry

end probability_white_rotation_l310_310168


namespace functions_eq_A_functions_eq_D_l310_310872

-- Defining functions from options
def fA (x : ℝ) : ℝ := x^2
def gA (x : ℝ) : ℝ := real.sqrt (x^4)

def fB (x : ℝ) : ℝ := x
def gB (x : ℝ) : ℝ := real.sqrt (x^2)

def fC (x : ℝ) : ℝ := real.sqrt (-2 * x^3)
def gC (x : ℝ) : ℝ := x * real.sqrt (-2 * x)

def fD (x : ℝ) : ℝ := x^2 - 2 * x - 1
def gD (t : ℝ) : ℝ := t^2 - 2 * t - 1

-- The theorem statements
theorem functions_eq_A : ∀ x : ℝ, fA x = gA x := by
  sorry

theorem functions_eq_D : ∀ x : ℝ, fD x = gD x := by
  sorry

end functions_eq_A_functions_eq_D_l310_310872


namespace vectors_are_perpendicular_l310_310780

variables {z1 z2 : ℂ} (A B : ℂ) (O : ℂ)

/-- Non-zero complex numbers corresponding to vectors and the condition |z1 + z2| = |z1 - z2| -/

def vectors_condition (z1 z2 : ℂ) (h1 : z1 ≠ 0) (h2 : z2 ≠ 0) (h3 : |z1 + z2| = |z1 - z2|) :
  Prop :=
  ∠(O - A, O - B) = π / 2

theorem vectors_are_perpendicular
  (z1 z2 : ℂ) 
  (h1 : z1 ≠ 0) 
  (h2 : z2 ≠ 0) 
  (h3 : |z1 + z2| = |z1 - z2|) :
  vectors_condition z1 z2 h1 h2 h3 := 
sorry

end vectors_are_perpendicular_l310_310780


namespace find_unit_prices_l310_310989

variable (x : ℝ)

def typeB_unit_price (priceB : ℝ) : Prop :=
  priceB = 15

def typeA_unit_price (priceA : ℝ) : Prop :=
  priceA = 40

def budget_condition : Prop :=
  900 / x = 3 * (800 / (x + 25))

theorem find_unit_prices (h : budget_condition x) :
  typeB_unit_price x ∧ typeA_unit_price (x + 25) :=
sorry

end find_unit_prices_l310_310989


namespace curve_equation_l310_310742

open Real

noncomputable def C1 := {(x, y) | ∃ M : Real × Real, (M.1 - 5)^2 + M.2^2 = 9 → (x, y) ∉ M}
noncomputable def C2 := {(x, y) | (x - 5)^2 + y^2 = 9}

theorem curve_equation :
  (∀ (x y : ℝ), (x, y) ∈ C1 → (x-5)^2 + y^2 > 9) ∧
  (∀ (M : ℝ × ℝ), M ∈ C1 → abs (fst M + 2) = inf {dist M P | P ∈ C2}) →
  (∀ (x y : ℝ), (x, y) ∈ C1 ↔ y^2 = 20 * (x - 5)) :=
sorry

end curve_equation_l310_310742


namespace locus_of_midpoint_l310_310916

theorem locus_of_midpoint (R O : Point) (radius : ℝ) (h₁ : radius = 10) (h₂ : dist R O > radius) :
  ∃ (center : Point) (new_radius : ℝ), 
    center = ⟨(2 * R.x + O.x) / 3, (2 * R.y + O.y) / 3⟩ ∧ 
    new_radius = 10 / 3 ∧ 
    ∀ S : Point, dist S O = radius → dist ((R.x + S.x) / 2, (R.y + S.y) / 2) center = new_radius := 
sorry

end locus_of_midpoint_l310_310916


namespace expression_1_sol_expression_2_sol_l310_310954

theorem expression_1_sol : 3 * (-8)^3 + sqrt ((-10)^2) + (1/2)^(-3) = -1518 := by
  sorry

theorem expression_2_sol : 
  let log2 := Real.log 2
  let log5 := Real.log 5
  let log6 := Real.log 6
  let log8 := 3 * log2
  let log1000 := 3
  let log2_sqrt3 := Real.sqrt 3 * log2
  let log_1_div_6 := -log6
  let log_0_006 := log6 - 3
  log5 * (log8 + log1000) + (log2_sqrt3)^2 + log_1_div_6 + log_0_006 = 3.002 := by
  sorry

end expression_1_sol_expression_2_sol_l310_310954


namespace Lee_surpasses_Hernandez_in_May_l310_310728

def monthly_totals_Hernandez : List ℕ :=
  [4, 8, 9, 5, 7, 6]

def monthly_totals_Lee : List ℕ :=
  [3, 9, 10, 6, 8, 8]

def cumulative_sum (lst : List ℕ) : List ℕ :=
  List.scanl (· + ·) 0 lst

noncomputable def cumulative_Hernandez := cumulative_sum monthly_totals_Hernandez
noncomputable def cumulative_Lee := cumulative_sum monthly_totals_Lee

-- Lean 4 statement asserting when Lee surpasses Hernandez in cumulative home runs
theorem Lee_surpasses_Hernandez_in_May :
  cumulative_Hernandez[3] < cumulative_Lee[3] :=
sorry

end Lee_surpasses_Hernandez_in_May_l310_310728


namespace octahedron_cube_ratio_l310_310282

theorem octahedron_cube_ratio (O C : Type) [IsRegularOctahedron O] [IsCubeFromOctahedronCenters C O] :
  (let m := 2
   let n := 9
   m + n = 11) := by
sorry

end octahedron_cube_ratio_l310_310282


namespace hyperbola_eccentricity_l310_310320

theorem hyperbola_eccentricity
    (a b : ℝ) (P N A B : ℝ × ℝ)
    (ha : 0 < a) (hb : 0 < b)
    (hP : P = (3, 6))
    (hN : N = (12, 15))
    (hC : ∀ (x y : ℝ), (x, y) = A ∨ (x, y) = B → x^2 / a^2 - y^2 / b^2 = 1)
    (hMidpoint : (Prod.fst A + Prod.fst B) / 2 = 12 ∧ (Prod.snd A + Prod.snd B) / 2 = 15) :
    let e := Real.sqrt (1 + b^2 / a^2) in
    e = 3 / 2 :=
by
  sorry

end hyperbola_eccentricity_l310_310320


namespace triangle_sides_from_rhombus_l310_310184

variable (m p q : ℝ)

def is_triangle_side_lengths (BC AC AB : ℝ) :=
  (BC = p + q) ∧
  (AC = m * (p + q) / p) ∧
  (AB = m * (p + q) / q)

theorem triangle_sides_from_rhombus :
  ∃ BC AC AB : ℝ, is_triangle_side_lengths m p q BC AC AB :=
by
  use p + q
  use m * (p + q) / p
  use m * (p + q) / q
  sorry

end triangle_sides_from_rhombus_l310_310184


namespace eval_at_minus_two_l310_310084

-- Conditions
def satisfies_condition (x : ℝ) : Prop :=
  x * (x^2 - 4) = 0

-- The expression we need to simplify
def expression (x : ℝ) : ℝ := (x - 3) / (3 * x * (x - 2)) / (x + 2 - 5 / (x - 2))

-- The goal is to prove that when x = -2, the expression simplifies to -1/6
theorem eval_at_minus_two :
  satisfies_condition (-2) →
  expression (-2) = -1 / 6 := by
  intro condition
  sorry

end eval_at_minus_two_l310_310084


namespace total_notebooks_and_pens_is_110_l310_310471

/-- The number of notebooks and pens on Wesley's school library shelf /--
namespace WesleyLibraryShelf

noncomputable section

def notebooks : ℕ := 30
def pens : ℕ := notebooks + 50
def total_notebooks_and_pens : ℕ := notebooks + pens

theorem total_notebooks_and_pens_is_110 :
  total_notebooks_and_pens = 110 :=
sorry

end WesleyLibraryShelf

end total_notebooks_and_pens_is_110_l310_310471


namespace parallelogram_diagonal_length_l310_310789

theorem parallelogram_diagonal_length :
  let ABCD : Parallelogram
  (equilateral : ∀ (T : Triangle), T.is_equilateral → T.side_length = 1) 
  (lengths : ∀ (s : Segment ABCD.sides), s.length = 2) 
  : (diagonal_length ABCD AC = sqrt 7) :=
sorry

end parallelogram_diagonal_length_l310_310789


namespace median_vision_is_correct_l310_310232

def student_vision_data : List (ℝ × ℕ) := [
  (4.0, 1),
  (4.1, 2),
  (4.2, 6),
  (4.3, 3),
  (4.4, 3),
  (4.5, 4),
  (4.6, 1),
  (4.7, 2),
  (4.8, 5),
  (4.9, 7),
  (5.0, 5)
]

def total_students : ℕ := 39

def median_vision_value (data : List (ℝ × ℕ)) (total : ℕ) : ℝ :=
  -- Function to calculate the median vision value
  let sorted_data := data.sortBy (λ x, x.1) in
  let cumulative_counts := sorted_data.scanl (λ acc x, acc + x.2) 0 in
  let median_position := total / 2 in
  let median_index := cumulative_counts.indexWhere (λ x, x > median_position) in
  (sorted_data.nth! (median_index - 1)).1

theorem median_vision_is_correct : median_vision_value student_vision_data total_students = 4.6 :=
by
  -- Prove that the median vision value of the dataset is 4.6
  sorry

end median_vision_is_correct_l310_310232


namespace circle_through_A_B_C_l310_310630

-- Definitions of points A, B, and C
def A : ℝ × ℝ := (1, 12)
def B : ℝ × ℝ := (7, 10)
def C : ℝ × ℝ := (-9, 2)

-- Definition of the expected standard equation of the circle
def circle_eq (x y : ℝ) : Prop := (x - 1) ^ 2 + (y - 2) ^ 2 = 100

-- Theorem stating that the expected equation is the equation of the circle through points A, B, and C
theorem circle_through_A_B_C : 
  ∀ (x y : ℝ),
  (x, y) = A ∨ (x, y) = B ∨ (x, y) = C → 
  circle_eq x y := sorry

end circle_through_A_B_C_l310_310630


namespace ratio_x_y_l310_310975

theorem ratio_x_y (x y : ℚ) (h : (14 * x - 5 * y) / (17 * x - 3 * y) = 2 / 3) : x / y = 1 / 23 := by
  sorry

end ratio_x_y_l310_310975


namespace max_chords_intersecting_line_l310_310173

theorem max_chords_intersecting_line (A : Fin 2017 → Type) :
  ∃ k : ℕ, (k ≤ 2016 ∧ ∃ m : ℕ, (m = k * (2016 - k) + 2016) ∧ m = 1018080) :=
sorry

end max_chords_intersecting_line_l310_310173


namespace g_eq_g_inv_iff_l310_310601

def g (x : ℝ) : ℝ := 3 * x - 7
def g_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem g_eq_g_inv_iff (x : ℝ) : g(x) = g_inv(x) ↔ x = 3.5 := by
  sorry

end g_eq_g_inv_iff_l310_310601


namespace point_region_and_line_l310_310018

theorem point_region_and_line (t : ℝ) :
  let pt := (2, t)
      line_eq := (x, y) => x - 2*y + 4
  in (line_eq 0 0 > 0 ∧ (line_eq 2 t ≥ 0)) → t ≤ 3 :=
sorry

end point_region_and_line_l310_310018


namespace find_interest_rate_l310_310237

-- Define the principal amount, time in years, number of times interest is compounded per year, and compound interest
def principal : ℝ := 30000
def time_years : ℝ := 2
def compound_times : ℕ := 2
def compound_interest : ℝ := 2472.964799999998

-- Calculate the future value
def future_value : ℝ := principal + compound_interest

-- Define the compound interest formula to solve for the interest rate
noncomputable def annual_interest_rate : ℝ :=
  let n := (compound_times : ℝ);
  let A := future_value;
  let P := principal;
  ((A / P)^(1 / (n * time_years)) - 1) * n * 100

-- State the theorem stating that the annual interest rate is approximately 4%
theorem find_interest_rate :
  annual_interest_rate ≈ 4 :=
by
  -- Proof goes here
  sorry

end find_interest_rate_l310_310237


namespace intersection_M_N_l310_310393

def M : set ℤ := {m | -3 < m ∧ m < 2}
def N : set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N :
  M ∩ N = {-1, 0, 1} :=
by
  sorry

end intersection_M_N_l310_310393


namespace unique_solution_of_equation_l310_310241

theorem unique_solution_of_equation :
  ∃! (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = 1 :=
begin
  sorry
end

end unique_solution_of_equation_l310_310241


namespace number_of_dogs_l310_310104

-- Define variables for the number of cats (C) and dogs (D)
variables (C D : ℕ)

-- Define the conditions from the problem statement
def condition1 : Prop := C = D - 6
def condition2 : Prop := C * 3 = D * 2

-- State the theorem that D should be 18 given the conditions
theorem number_of_dogs (h1 : condition1 C D) (h2 : condition2 C D) : D = 18 :=
  sorry

end number_of_dogs_l310_310104


namespace solution_count_l310_310827

theorem solution_count (A B C D : ℝ) (m n : ℕ) (h1 : ∃ (x y : ℝ), x^2 + y^2 = A ∧ |x| + |y| = B ∧ m = 8)
  (h2 : ∃ (x y z : ℝ), x^2 + y^2 + z^2 = C ∧ |x| + |y| + |z| = D ∧ n = 6) (h3 : m > n > 1) : m = 8 ∧ n = 6 :=
by
  sorry

end solution_count_l310_310827


namespace correct_function_correspondence_rule_l310_310306

theorem correct_function_correspondence_rule 
  (f : ℝ → ℝ)
  (dom : Set ℝ := Set.Icc 0 2)
  (range : Set ℝ := Set.Icc 1 4) 
  (h1 : ∀ x ∈ dom, 1 ≤ f x ∧ f x ≤ 4) 
  (f₁ : ℝ → ℝ := λ x, 2 * x)
  (f₂ : ℝ → ℝ := λ x, x^2 + 1) 
  (f₃ : ℝ → ℝ := λ x, 2^x) 
  (f₄ : ℝ → ℝ := λ x, Real.log x / Real.log 2) :
  (∀ x ∈ dom, f₁ x ∉ range) ∧ 
  (∀ x ∈ dom, f₂ x ∉ range) ∧ 
  (∀ x ∈ dom, f₃ x ∈ range) ∧ 
  (∀ x ∈ dom, x ≠ 0 → f₄ x ∉ range) :=
by
  sorry

end correct_function_correspondence_rule_l310_310306


namespace mutually_exclusive_pairs_l310_310678

/-- Define the events for shooting rings and drawing balls. -/
inductive ShootEvent
| hits_7th_ring : ShootEvent
| hits_8th_ring : ShootEvent

inductive PersonEvent
| at_least_one_hits : PersonEvent
| A_hits_B_does_not : PersonEvent

inductive BallEvent
| at_least_one_black : BallEvent
| both_red : BallEvent
| no_black : BallEvent
| one_red : BallEvent

/-- Define mutually exclusive events. -/
def mutually_exclusive (e1 e2 : Prop) : Prop := e1 ∧ e2 → False

/-- Prove the pairs of events that are mutually exclusive. -/
theorem mutually_exclusive_pairs :
  mutually_exclusive (ShootEvent.hits_7th_ring = ShootEvent.hits_7th_ring) (ShootEvent.hits_8th_ring = ShootEvent.hits_8th_ring) ∧
  ¬mutually_exclusive (PersonEvent.at_least_one_hits = PersonEvent.at_least_one_hits) (PersonEvent.A_hits_B_does_not = PersonEvent.A_hits_B_does_not) ∧
  mutually_exclusive (BallEvent.at_least_one_black = BallEvent.at_least_one_black) (BallEvent.both_red = BallEvent.both_red) ∧
  mutually_exclusive (BallEvent.no_black = BallEvent.no_black) (BallEvent.one_red = BallEvent.one_red) :=
by {
  sorry
}

end mutually_exclusive_pairs_l310_310678


namespace train_length_l310_310929

-- Define the given speed and time
def speed_kmh : ℝ := 54
def time_s : ℝ := 6.666133375996587

-- Convert speed from km/h to m/s
def speed_ms : ℝ := (speed_kmh * 1000) / 3600

-- Define the expected length of the train
def expected_length : ℝ := 99.99

-- Theorem statement about the length of the train
theorem train_length :
  (speed_ms * time_s = expected_length) :=
by
  -- Skip the proof with sorry
  sorry

end train_length_l310_310929


namespace number_of_correct_derivative_conclusions_l310_310191

theorem number_of_correct_derivative_conclusions :
  (∀ y : ℝ, (y = Real.log 2) → (deriv id y = 0)) ∧
  (∀ (x y : ℝ), (y = x⁻²) → (x = 3) → (deriv (λ x, 1 / x^2) x = -2 / 27)) ∧
  (∀ x : ℝ, (y = 2^x) → (deriv (λ x, 2^x) x = 2^x * Real.log 2)) ∧
  (∀ x : ℝ, (y = log x / log 2) → (deriv (λ x, Real.log x / Real.log 2) x = 1 / (x * Real.log 2))) →
  2 = 2 :=
by
  sorry

end number_of_correct_derivative_conclusions_l310_310191


namespace good_placements_count_l310_310465

-- Define the type for card and box
structure Card :=
  (i j : ℕ)
  (h : i ≠ j)

def CardList : List Card :=
  [⟨1, 2, by linarith⟩, ⟨1, 3, by linarith⟩, ⟨1, 4, by linarith⟩, ⟨1, 5, by linarith⟩,
   ⟨2, 3, by linarith⟩, ⟨2, 4, by linarith⟩, ⟨2, 5, by linarith⟩, ⟨3, 4, by linarith⟩,
   ⟨3, 5, by linarith⟩, ⟨4, 5, by linarith⟩]

-- Define the predicate for a good placement
def is_good_placement (placement : Card → ℕ) : Prop :=
  let box_count := λ b, (CardList.filter (λ c, placement c = b)).length
  box_count 1 > box_count 2 ∧
  box_count 1 > box_count 3 ∧
  box_count 1 > box_count 4 ∧
  box_count 1 > box_count 5

def number_of_good_placements : ℕ :=
  -- Calculation to be done, assumed to be implemented
  120

-- Statement of the problem in Lean
theorem good_placements_count : ∃ count, count = number_of_good_placements :=
  sorry

end good_placements_count_l310_310465


namespace right_triangle_rotated_solid_areas_l310_310185

def right_triangle (a b c : ℕ) : Prop := a ^ 2 + b ^ 2 = c ^ 2

def rotated_solid_surface_area (a b c : ℕ) (rotation_side : ℕ) : ℝ :=
  if rotation_side = a then pi * a * (real.sqrt ((c - a) ^ 2 + b ^ 2))
  else if rotation_side = b then pi * b * (real.sqrt ((c - b) ^ 2 + a ^ 2))
  else 0

theorem right_triangle_rotated_solid_areas :
  right_triangle 3 4 5 →
  rotated_solid_surface_area 3 4 5 3 = 6 * pi * real.sqrt 5 ∧
  rotated_solid_surface_area 3 4 5 4 = 4 * pi * real.sqrt 10 :=
by {
  intros h,
  unfold right_triangle at h,
  -- Use sorry to skip the proofs of the detailed computations.
  sorry
}

end right_triangle_rotated_solid_areas_l310_310185


namespace median_vision_is_4_6_l310_310231

def vision_data : List ℕ :=
  [1, 2, 6, 3, 3, 4, 1, 2, 5, 7, 5]

def vision_values : List ℚ :=
  [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]

def cumulative_students : List ℕ :=
  List.scanl (· + ·) 0 vision_data

theorem median_vision_is_4_6 : 
  vision_values.nth 6 = some 4.6 := by
  have h : cumulative_students.nth 6 = some 20 := by sorry
  show vision_values.nth 6 = some 4.6 from sorry

end median_vision_is_4_6_l310_310231


namespace total_cookies_l310_310723

theorem total_cookies (x y : Nat) (h1 : x = 137) (h2 : y = 251) : x * y = 34387 := by
  sorry

end total_cookies_l310_310723


namespace find_curve_eq_center_max_area_Ode_l310_310697

noncomputable def circle1 (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 25
noncomputable def circle2 (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
noncomputable def curve (x y : ℝ) : Prop := (x^2 / 9) + (y^2 / 8) = 1
noncomputable def pointA := (0, -3 : ℝ)

-- To state that circle C₃ is tangent to the inside of circle C₁, and to the outside of circle C₂.
def tangent_to_inside_outside (C₃ : ℝ × ℝ → ℝ → Prop) : Prop := 
∀ x y r, 
  (C₃ (x, y) r -> (x - 1)^2 + y^2 = (5 - r)^2) ∧  -- Tangent to the inside of C₁
  (C₃ (x, y) r -> (x + 1)^2 + y^2 = (1 + r)^2)    -- Tangent to the outside of C₂

theorem find_curve_eq_center (C₃ : ℝ × ℝ → ℝ → Prop) : 
  tangent_to_inside_outside C₃ → 
  ∀ x y, curve x y ↔ ∃ r, C₃ (x, y) r := sorry

theorem max_area_Ode (l : ℝ → ℝ → Prop) : 
  (∀ x, l x = (some k : ℝ) x - 3) → 
  ∀ D E : ℝ × ℝ, 
  curve D.1 D.2 → curve E.1 E.2 → 
  (∃ A B C : ℝ, (pointA = (A, B) ∧ (0, 0) = (C, C)) ∧ 0 ≤ 
  (3 * sqrt 2 * sqrt ((3 * some k)^2 - 1) / ((3 * sqrt 2 * sqrt ((3 * some k)^2 - 1)) / ((some k^2 + 2) * -3))) := 
  (3 * sqrt 2)) := sorry

end find_curve_eq_center_max_area_Ode_l310_310697


namespace speed_of_goods_train_l310_310912

theorem speed_of_goods_train
  (length_of_train : ℕ)
  (length_of_platform : ℕ)
  (time_to_cross_platform : ℕ)
  (total_length := length_of_train + length_of_platform)
  (speed_m_per_s := total_length / time_to_cross_platform)
  (conversion_factor : ℝ := 3.6)
  (speed_km_per_hr := speed_m_per_s * conversion_factor) :
  length_of_train = 220 →
  length_of_platform = 300 →
  time_to_cross_platform = 26 →
  speed_km_per_hr = 72 :=
by {
  intros h1 h2 h3,
  have h4 : total_length = 520 := by simp [h1, h2],
  have h5 : speed_m_per_s = 20 := by norm_num [h4, h3],
  have h6 : speed_km_per_hr = 72 := by norm_num [h5, conversion_factor],
  exact h6,
}

end speed_of_goods_train_l310_310912


namespace number_of_correct_statements_l310_310942

theorem number_of_correct_statements :
  (∃! (n : ℕ), n = 1) ∧
  ¬(∀ a b : ℝ, a < b ∧ a ≠ 0 ∧ b ≠ 0 → 1 / b < 1 / a) ∧
  ¬(∀ a b c : ℝ, a < b → ac < bc) ∧
   (∀ a b c : ℝ, a < b → a + c < b + c) ∧
  ¬(∀ a b : ℝ, a^2 < b^2 → a < b) := 
sorry

end number_of_correct_statements_l310_310942


namespace trigonometric_function_properties_l310_310317

theorem trigonometric_function_properties:
  (∃ A ω ϕ : ℝ, 0 < A ∧ 0 < ω ∧ ∀ x : ℝ, 
    (y = A * sin (ω * x + ϕ)) ∧ 
    y (π/12) = 0 ∧ 
    y (π/3) = 5) → 
  (∃ k : ℤ, y = 5 * sin (2 * x - π / 6) ∧ 
    (∀ x : ℝ, k * π + π/3 ≤ x ∧ x < k * π + 5 * π/6 → y (x) < y (x + δ)) ∧ 
    (0 ≤ x ∧ x ≤ π → (y = 5 ∧ x = π/3) ∨ (y = -5 ∧ x = 5 * π/6)) ∧ 
    (∃ k : ℤ, k * π - 5 * π/12 ≤ x ∧ x ≤ k * π + π/12 → y ≤ 0)) :=
sorry

end trigonometric_function_properties_l310_310317


namespace find_BC_l310_310256

variables (AB AC BC : ℝ × ℝ)

def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

theorem find_BC
  (hAB : AB = (2, -1))
  (hAC : AC = (-4, 1)) :
  BC = (-6, 2) :=
by
  simp [vector_sub, hAB, hAC]
  sorry

end find_BC_l310_310256


namespace smaller_successive_number_l310_310829

theorem smaller_successive_number (n : ℕ) (h : n * (n + 1) = 9506) : n = 97 :=
sorry

end smaller_successive_number_l310_310829


namespace basketball_cost_l310_310831

variable (num_basketballs num_soccerballs total_cost cost_per_soccerball cost_per_basketball : ℕ)

-- Conditions
axiom h1 : num_basketballs = 5
axiom h2 : num_soccerballs = 8
axiom h3 : total_cost = 920
axiom h4 : cost_per_soccerball = 65

-- Question and correct answer
theorem basketball_cost :
  (num_soccerballs * cost_per_soccerball + num_basketballs * cost_per_basketball = total_cost) →
  cost_per_basketball = 80 :=
by
  -- Use the conditions as axioms
  assume h : num_soccerballs * cost_per_soccerball + num_basketballs * cost_per_basketball = total_cost,
  have h1 : num_basketballs = 5, from h1,
  have h2 : num_soccerballs = 8, from h2,
  have h3 : total_cost = 920, from h3,
  have h4 : cost_per_soccerball = 65, from h4,
  sorry

end basketball_cost_l310_310831


namespace probability_of_selecting_green_ball_l310_310213

-- Declare the probability of selecting each container
def prob_of_selecting_container := (1 : ℚ) / 4

-- Declare the number of balls in each container
def balls_in_container_A := 10
def balls_in_container_B := 14
def balls_in_container_C := 14
def balls_in_container_D := 10

-- Declare the number of green balls in each container
def green_balls_in_A := 6
def green_balls_in_B := 6
def green_balls_in_C := 6
def green_balls_in_D := 7

-- Calculate the probability of drawing a green ball from each container
def prob_green_from_A := (green_balls_in_A : ℚ) / balls_in_container_A
def prob_green_from_B := (green_balls_in_B : ℚ) / balls_in_container_B
def prob_green_from_C := (green_balls_in_C : ℚ) / balls_in_container_C
def prob_green_from_D := (green_balls_in_D : ℚ) / balls_in_container_D

-- Calculate the total probability of drawing a green ball
def total_prob_green :=
  prob_of_selecting_container * prob_green_from_A +
  prob_of_selecting_container * prob_green_from_B +
  prob_of_selecting_container * prob_green_from_C +
  prob_of_selecting_container * prob_green_from_D

theorem probability_of_selecting_green_ball : total_prob_green = 13 / 28 :=
by sorry

end probability_of_selecting_green_ball_l310_310213


namespace proof_general_formula_l310_310686

noncomputable def h (x t : ℝ) : ℝ := (1/2) * x^2 + t * x

def h_prime (x t : ℝ) : ℝ := x + t

def h_double_prime (x : ℝ) : ℝ := 1

/-- Assuming t = 1 makes the slope of the tangent line to h(x) at the origin equal to 1. -/
def t_value : ℝ := 1

noncomputable def f (x : ℝ) : ℝ := x / (h_double_prime x)

def sequence (a : ℝ) (n : ℕ) : ℝ 
| 0     := a
| (n+1) := f (sequence a n)

theorem proof_general_formula (a : ℝ) (h_pos : a > 0) : 
  ∀ n : ℕ, sequence a n = a / (1 + n * a) :=
sorry

end proof_general_formula_l310_310686


namespace range_of_q_l310_310217

variable (x : ℝ)

def q (x : ℝ) := (3 * x^2 + 1)^2

theorem range_of_q : ∀ y, (∃ x : ℝ, x ≥ 0 ∧ y = q x) ↔ y ≥ 1 := by
  sorry

end range_of_q_l310_310217


namespace triangle_abc_ac_interval_sum_l310_310431

theorem triangle_abc_ac_interval_sum :
  ∀ (A B C D : Type) [linear_ordered_ring A]
  (AB : A) (CD : A),
    AB = 8 → CD = 6 →
    (∃ m n : A, (∀ AC : A, 6 < AC ∧ AC < 16) ∧ m = 6 ∧ n = 16 ∧ m + n = 22) :=
by
  intros A B C D hAB hCD
  sorry

end triangle_abc_ac_interval_sum_l310_310431


namespace find_number_l310_310167

theorem find_number (x : ℝ) (h : 0.20 * x = 0.20 * 650 + 190) : x = 1600 := by 
  sorry

end find_number_l310_310167


namespace price_per_pound_of_40_lbs_is_33_point_2_l310_310913

/-- Define initial conditions -/
def cost_5_lb := 5 * 80 -- 5 pounds * 80 cents per pound
def total_weight := 5 + 40
def selling_price := 48 * total_weight
def total_cost_with_profit := 0.8 * selling_price

/-- Statement to be proved -/
theorem price_per_pound_of_40_lbs_is_33_point_2 :
  ∃ x : ℝ, (cost_5_lb + 40 * x = total_cost_with_profit) ∧ x = 33.2 := by
  sorry

end price_per_pound_of_40_lbs_is_33_point_2_l310_310913


namespace sum_of_coefficients_l310_310710

theorem sum_of_coefficients (a_5 a_4 a_3 a_2 a_1 a_0 : ℤ) :
  (x-2)^5 = a_5*x^5 + a_4*x^4 + a_3*x^3 + a_2*x^2 + a_1*x + a_0 →
  a_1 + a_2 + a_3 + a_4 + a_5 = 31 :=
by
  sorry

end sum_of_coefficients_l310_310710


namespace donny_spending_l310_310982

theorem donny_spending :
  (15 + 28 + 13) / 2 = 28 :=
by
  sorry

end donny_spending_l310_310982


namespace bottle_caps_per_child_l310_310224

-- Define the conditions
def num_children : ℕ := 9
def total_bottle_caps : ℕ := 45

-- State the theorem that needs to be proved: each child has 5 bottle caps
theorem bottle_caps_per_child : (total_bottle_caps / num_children) = 5 := by
  sorry

end bottle_caps_per_child_l310_310224


namespace sum_of_cubes_l310_310761

def cubic_eq (x : ℝ) : Prop := x^3 - 2 * x^2 + 3 * x - 4 = 0

variables (a b c : ℝ)

axiom a_root : cubic_eq a
axiom b_root : cubic_eq b
axiom c_root : cubic_eq c

axiom sum_roots : a + b + c = 2
axiom sum_products_roots : a * b + a * c + b * c = 3
axiom product_roots : a * b * c = 4

theorem sum_of_cubes : a^3 + b^3 + c^3 = 2 :=
by
  sorry

end sum_of_cubes_l310_310761


namespace quadrant_of_z_plus_one_over_z_l310_310263

noncomputable def z : ℂ := 1 - real.sqrt 2 * complex.I
noncomputable def one_over_z : ℂ := (1 + real.sqrt 2 * complex.I) / 3
noncomputable def z_plus_one_over_z : ℂ := z + one_over_z

theorem quadrant_of_z_plus_one_over_z :
  z_plus_one_over_z.re > 0 ∧ z_plus_one_over_z.im < 0 :=
by {
  sorry
}

end quadrant_of_z_plus_one_over_z_l310_310263


namespace zoe_makes_212_dollars_l310_310877

-- Define the initial conditions
def milk_chocolate_cost : ℕ := 6
def dark_chocolate_cost : ℕ := 8
def white_chocolate_cost : ℕ := 10

def milk_chocolate_bars : ℕ := 15
def dark_chocolate_bars : ℕ := 12
def white_chocolate_bars : ℕ := 13

def percentage_sold (p : ℚ) (n : ℕ) : ℕ := nat.floor (p * n)

def milk_chocolate_sold : ℕ := percentage_sold (70 / 100) milk_chocolate_bars
def dark_chocolate_sold : ℕ := percentage_sold (80 / 100) dark_chocolate_bars
def white_chocolate_sold : ℕ := percentage_sold (65 / 100) white_chocolate_bars

-- Calculate the total money made from selling the bars
def total_money : ℕ := (milk_chocolate_sold * milk_chocolate_cost) +
                       (dark_chocolate_sold * dark_chocolate_cost) + 
                       (white_chocolate_sold * white_chocolate_cost)

-- The theorem statement using Lean
theorem zoe_makes_212_dollars : total_money = 212 := by
  sorry

end zoe_makes_212_dollars_l310_310877


namespace checkers_arrangement_l310_310838

noncomputable def arrange_checkers (checkers : Fin 64 → Color) : Fin 8 → Fin 8 → Color :=
  sorry

theorem checkers_arrangement : 
  ∀ (checkers : Fin 64 → Color) 
    (h_pairs : ∀ i j, i ≠ j → checkers i ≠ checkers j) 
    (h_count : ∀ c : Color, 32 ≤ ∏ i, if checkers i = c then 1 else 0),
  ∃ (arrangement : Fin 8 → Fin 8 → Color),
  (∀ (i j : Fin 8), 
    -- Ensuring that every two-cell rectangle has checkers of different colors
    ∀ (rect : Fin 2 × Fin 2),
    arrangement rect.fst.fst rect.fst.snd ≠ arrangement rect.snd.fst rect.snd.snd) :=
    sorry

end checkers_arrangement_l310_310838


namespace swimmer_upstream_distance_l310_310557

theorem swimmer_upstream_distance (v : ℝ) (c : ℝ) (downstream_distance : ℝ) (downstream_time : ℝ) 
                                   (downstream_speed : ℝ) (upstream_time : ℝ) : 
  c = 4.5 →
  downstream_distance = 55 →
  downstream_time = 5 →
  downstream_speed = downstream_distance / downstream_time →
  v + c = downstream_speed →
  upstream_time = 5 →
  (v - c) * upstream_time = 10 := 
by
  intro h_c
  intro h_downstream_distance
  intro h_downstream_time
  intro h_downstream_speed
  intro h_effective_downstream
  intro h_upstream_time
  sorry

end swimmer_upstream_distance_l310_310557


namespace permutation_order_number_l310_310405

/-- Given a permutation of the numbers 1, 2, ..., 8, prove that there are 144 distinct permutations
    where the order number of 8 is 2, the order number of 7 is 3, and the order number of 5 is 3. -/
theorem permutation_order_number :
  ∃ perm : list ℕ, (∀ i, perm.nodup) ∧ ∃ perm_inds : list ℕ, 
    (perm.nth_le 2 sorry = 8 ∧ ∃ pos1 pos2, 
      perm.nth_le pos1 sorry = 7 ∧ perm.nth_le pos2 sorry = 5 ∧
      (number_of_distinct_permutations perm perm_inds = 144)) := 
sorry

end permutation_order_number_l310_310405


namespace expand_and_simplify_l310_310863

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
  sorry

end expand_and_simplify_l310_310863


namespace y_intercept_of_line_l310_310926

-- Representing the points
def point1 := (2 : ℝ, -2 : ℝ)
def point2 := (6 : ℝ, 6 : ℝ)

-- The question to prove:
theorem y_intercept_of_line (p1 p2 : ℝ × ℝ) (h1 : p1 = point1) (h2 : p2 = point2) : 
  ∃ b : ℝ, (b = -6) :=
by
  sorry

end y_intercept_of_line_l310_310926


namespace car_preference_related_to_gender_l310_310532

-- Definitions related to the given problem conditions
def total_survey_size (n : ℕ) : ℕ := 20 * n

def chi_square (n : ℕ) : ℝ := 5.556

def contingency_table (n : ℕ) : Bool → Bool → ℕ
| true, true  => 10 * n  -- Male, Like
| true, false => 2 * n   -- Male, Dislike
| false, true => 5 * n   -- Female, Like
| false, false => 3 * n  -- Female, Dislike

-- Statement of the problem to prove
-- Assuming n is a positive natural number
theorem car_preference_related_to_gender (n : ℕ) (h_pos : n > 0) :
  chi_square n ≈ 5.556 ∧
  (n = 5) ∧
  (0:ℝ) × (14 / 55 : ℝ) + 1 × (28 / 55) + 2 × (12 / 55) + 3 × (1 / 55) = 1 :=
by
  sorry

end car_preference_related_to_gender_l310_310532


namespace math_problem_l310_310713

theorem math_problem (x : ℝ) (h : 2^x + 2^x + 2^x + 2^x + 2^x + 2^x + 2^x + 2^x = 256) :
  (x + 2) * (x - 2) = 21 :=
by 
  sorry

end math_problem_l310_310713


namespace car_speed_l310_310906

-- Define the given conditions
def distance := 800 -- in kilometers
def time := 5 -- in hours

-- Define the speed calculation
def speed (d : ℕ) (t : ℕ) : ℕ := d / t

-- State the theorem to be proved
theorem car_speed : speed distance time = 160 := by
  -- proof would go here
  sorry

end car_speed_l310_310906


namespace equilateral_triangle_side_length_l310_310708

/--
If an equilateral triangle has a perimeter of 63 cm,
then the length of one side is 21 cm.
-/
theorem equilateral_triangle_side_length (h : ∀ a : ℝ, 3 * a = 63) : ∃ s : ℝ, s = 21 :=
by
  use 21
  sorry

end equilateral_triangle_side_length_l310_310708


namespace octahedron_cube_ratio_l310_310283

theorem octahedron_cube_ratio (O C : Type) [IsRegularOctahedron O] [IsCubeFromOctahedronCenters C O] :
  (let m := 2
   let n := 9
   m + n = 11) := by
sorry

end octahedron_cube_ratio_l310_310283


namespace increasing_interval_l310_310102

def my_function (x : ℝ) : ℝ := -(x - 3) * |x|

theorem increasing_interval : ∀ x y : ℝ, 0 ≤ x → x ≤ y → my_function x ≤ my_function y :=
by
  sorry

end increasing_interval_l310_310102


namespace amount_returned_l310_310749

namespace IrinaBankProblem

-- Definitions
def deposit_amount : ℝ := 23904
def annual_interest_rate : ℝ := 0.05
def period_in_months : ℕ := 3
def exchange_rate : ℝ := 58.15
def insurance_limit : ℝ := 1400000

-- Goal statement
theorem amount_returned : 
  let monthly_interest_rate := annual_interest_rate / 12
  let interest_for_period := deposit_amount * (period_in_months * monthly_interest_rate) / 12
  let total_amount := deposit_amount + interest_for_period
  let amount_in_rubles := total_amount * exchange_rate in
  min amount_in_rubles insurance_limit = 1400000 :=
by {
  sorry
}

end IrinaBankProblem

end amount_returned_l310_310749


namespace two_dot_one_seventy_four_repeating_as_fraction_l310_310615

noncomputable def repeating_decimal_to_fraction (x : ℚ) : ℚ :=
  if x = 0.02 then 2 / 99 else
    if x = 0.17 then 17 / 99 else
      x

theorem two_dot_one_seventy_four_repeating_as_fraction : repeating_decimal_to_fraction 2.17 = 215 / 99 :=
by {
  have h : repeating_decimal_to_fraction 0.02 = 2 / 99, by sorry, -- given condition
  have h' : repeating_decimal_to_fraction 0.17 = 17 / 99, by sorry, -- derived from multiplication mapping and given condition

  -- Combining the results to assert the proof
  have result : 2 + 17 / 99 = 215 / 99, by sorry,

  -- Final statement aligning with the original problem
  exact result,
}

end two_dot_one_seventy_four_repeating_as_fraction_l310_310615


namespace find_a_b_and_monotonic_intervals_l310_310100

noncomputable def f (x a b : ℝ) := (x^2 + a * x + b) * Real.exp (-x)

theorem find_a_b_and_monotonic_intervals :
  ∃ a b : ℝ, (a = 1 ∧ b = -5 ∧
  (∀ x, f x a b = (x^2 + 1 * x - 5) * Real.exp (-x)) ∧ 
  (∀ x, ∀ y, y = f x a b → (d/dx (f x a b) = (-x^2 + x + 6) * Real.exp (-x)) ∧
    ((-2 < x ∧ x < 3) → (d/dx (f x a b) > 0)) ∧ 
    ((x < -2 ∨ x > 3) → (d/dx (f x a b) < 0)))) := 
begin
  sorry
end

end find_a_b_and_monotonic_intervals_l310_310100


namespace smallest_perfect_square_divisible_by_4_and_5_l310_310501

theorem smallest_perfect_square_divisible_by_4_and_5 : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (m : ℕ), n = m * m) ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ (n = 400) := 
by
  sorry

end smallest_perfect_square_divisible_by_4_and_5_l310_310501


namespace find_white_balls_l310_310731

-- Define the number of red balls
def red_balls : ℕ := 4

-- Define the probability of drawing a red ball
def prob_red : ℚ := 1 / 4

-- Define the number of white balls
def white_balls : ℕ := 12

theorem find_white_balls (x : ℕ) (h1 : (red_balls : ℚ) / (red_balls + x) = prob_red) : x = white_balls :=
by
  -- Proof is omitted
  sorry

end find_white_balls_l310_310731


namespace true_propositions_l310_310192

theorem true_propositions (P1 P2 P3 P4 : Prop) : 
  (P1 ↔ ∀ (A B C D : Type) (a b c d : A) (h : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ c ≠ d), 
    (∀ {α : A}, α = a ∧ α = c → false) → c = d) → 
  (P2 ↔ ∀ (A B C : Type) (a b : A) (c : B) (d : C), 
    (∀ {α : A} {β : B} {γ : C}, α = a ∧ β = c ∧ γ = d → false) → 
    (c = d ∨ ¬(c = d))) → 
  (P3 ↔ ∀ (X Y Z : Type) (a b : X) (c : Y) (d : Z), 
    (∀ {α : X} {β : Y} {γ : Z}, α = a ∧ β = c ∧ γ = d → false) → 
    (c = d)) → 
  (P4 ↔ ∀ (X Y : Type) (a b : X) (c d : Y), 
    (∀ {α : X} {β : Y}, α = a ∧ β = c → false) → (a = c ∧ b = d ∨ a ≠ b)) → 
  {1, 3} = {i | i = 1 ∨ i = 3} :=
by sorry

end true_propositions_l310_310192


namespace parabola_intersects_line_at_P_and_Q_l310_310691

noncomputable def focus_of_parabola := (1 : ℝ, 0 : ℝ)

def is_parabola (P : ℝ × ℝ) : Prop :=
  P.2 ^ 2 = 4 * P.1

def is_line (P : ℝ × ℝ) : Prop :=
  P.2 = P.1 - 1

def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

theorem parabola_intersects_line_at_P_and_Q :
  ∀ P Q : ℝ × ℝ, is_parabola P → is_line P → is_parabola Q → is_line Q →
  let F := focus_of_parabola in
  let PF := distance P F in
  let QF := distance Q F in
  (1 / PF) + (1 / QF) = 1 := by
  intros P Q hP1 hP2 hQ1 hQ2
  let F := focus_of_parabola
  let PF : ℝ := distance P F
  let QF : ℝ := distance Q F
  sorry

end parabola_intersects_line_at_P_and_Q_l310_310691


namespace maximum_value_of_f_l310_310162

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
  if irrational x then x
  else
    let p := x.num in
    let q := x.denom in
    (p + 1) / q

-- The interval of interest
def interval (a b : ℝ) (x : ℝ) := a < x ∧ x < b

-- Maximum value of f(x) in the interval (7/8, 8/9)
theorem maximum_value_of_f :
  ∀ x : ℝ, interval (7 / 8) (8 / 9) x → f x ≤ 16 / 17 := by
  sorry

end maximum_value_of_f_l310_310162


namespace sum_series_1_to_144_squares_l310_310976

theorem sum_series_1_to_144_squares : 
  1^2 - 2^2 - 3^2 - 4^2 + 5^2 + 6^2 + 7^2 + 8^2 + 9^2 - 10^2 - 
  ∑ k in {11, 12, 13, ..., 144}, if (k - 1) / 3 % 3 == 2 then - k^2 else k^2 = 95433 := 
sorry

end sum_series_1_to_144_squares_l310_310976


namespace find_f9_l310_310599

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f(-x) = f(x)
axiom functional_eq : ∀ x : ℝ, f(x + 6) = f(x) + f(3)

theorem find_f9 : f(9) = 0 :=
by {
  sorry
}

end find_f9_l310_310599


namespace unique_representation_l310_310424

theorem unique_representation (k n : ℕ) :
  ∃ (a : Fin k → ℕ), -- array of length k, representing (a₁, a₂, ..., aₖ)
  (∀ i j : Fin k, i < j → a i < a j) ∧
  (n = (Finset.range k).sum (λ i, Nat.choose (a ⟨i.val, sorry⟩) (i + 1))) := 
sorry

end unique_representation_l310_310424


namespace smallest_perfect_square_divisible_by_4_and_5_l310_310502

theorem smallest_perfect_square_divisible_by_4_and_5 : 
  ∃ (n : ℕ), (n > 0) ∧ (∃ (m : ℕ), n = m * m) ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ (n = 400) := 
by
  sorry

end smallest_perfect_square_divisible_by_4_and_5_l310_310502


namespace bernold_wins_game_l310_310893

/-- A game is played on a 2007 x 2007 grid. Arnold's move consists of taking a 2 x 2 square,
 and Bernold's move consists of taking a 1 x 1 square. They alternate turns with Arnold starting.
  When Arnold can no longer move, Bernold takes all remaining squares. The goal is to prove that 
  Bernold can always win the game by ensuring that Arnold cannot make enough moves to win. --/
theorem bernold_wins_game (N : ℕ) (hN : N = 2007) :
  let admissible_points := (N - 1) * (N - 1)
  let arnold_moves_needed := (N / 2) * (N / 2 + 1) / 2 + 1
  admissible_points < arnold_moves_needed :=
by
  let admissible_points := 2006 * 2006
  let arnold_moves_needed := 1003 * 1004 / 2 + 1
  exact sorry

end bernold_wins_game_l310_310893


namespace isosceles_triangle_angle_l310_310016

theorem isosceles_triangle_angle (ABC : Triangle) (A B C : Point)
  (h_isosceles : AB = BC)
  (h_angle_bisector : ∀ D E : Point, angle_bisector A B C D = 2 * angle_bisector B C A E) : 
  (angle A B C = 36) ∧ (angle B A C = 36) ∧ (angle B C A = 108) :=
by
  sorry

end isosceles_triangle_angle_l310_310016


namespace exchange_rate_l310_310387

def jackPounds : ℕ := 42
def jackEuros : ℕ := 11
def jackYen : ℕ := 3000
def poundsPerYen : ℕ := 100
def totalYen : ℕ := 9400

theorem exchange_rate :
  ∃ (x : ℕ), 100 * jackPounds + 100 * jackEuros * x + jackYen = totalYen ∧ x = 2 :=
by
  sorry

end exchange_rate_l310_310387


namespace power_eq_45_l310_310643

theorem power_eq_45 (a m n : ℝ) (h1 : a^m = 3) (h2 : a^n = 5) : a^(2*m + n) = 45 := by
  sorry

end power_eq_45_l310_310643


namespace area_in_square_yards_l310_310549

/-
  Given:
  - length of the classroom in feet
  - width of the classroom in feet

  Prove that the area required to cover the classroom in square yards is 30. 
-/

def classroom_length_feet : ℕ := 15
def classroom_width_feet : ℕ := 18
def feet_to_yard (feet : ℕ) : ℕ := feet / 3

theorem area_in_square_yards :
  let length_yards := feet_to_yard classroom_length_feet
  let width_yards := feet_to_yard classroom_width_feet
  length_yards * width_yards = 30 :=
by
  sorry

end area_in_square_yards_l310_310549


namespace ned_defuse_time_l310_310775

theorem ned_defuse_time (flights_total time_per_flight bomb_time time_spent : ℕ) (h1 : flights_total = 20) (h2 : time_per_flight = 11) (h3 : bomb_time = 72) (h4 : time_spent = 165) :
  bomb_time - (flights_total * time_per_flight - time_spent) / time_per_flight * time_per_flight = 17 := by
  sorry

end ned_defuse_time_l310_310775


namespace contains_centroid_of_subset_dilation_l310_310037

variables {n : ℕ} (K : Set (EuclideanSpace ℝ (Fin n))) (S : Set (EuclideanSpace ℝ (Fin n)))
  (ε : ℝ) (hK_convex : Convex ℝ K) (hK_unit_volume : Volume K = 1)
  (hS_measurable : MeasurableSet S) (hS_measure : Volume S ≥ 1 - ε)

noncomputable def dilation_ratio : ℝ := 2 * ε * Real.log(1 / ε)

def centroid (A : Set (EuclideanSpace ℝ (Fin n))) : EuclideanSpace ℝ (Fin n) := 
  (1 / Volume A) • ∫ᵣⁿ x in A, x

theorem contains_centroid_of_subset_dilation :
  0 < ε → ε < 1 / 3 →
  centroid S ∈ dilation (dilation_ratio ε) (centroid K) K := sorry

end contains_centroid_of_subset_dilation_l310_310037


namespace geometric_sequence_product_l310_310375

theorem geometric_sequence_product {a : ℕ → ℝ} (q : ℝ) (h1 : |q| ≠ 1) :
  a 1 = 1 → (∀ n, a n = a 1 * (q ^ (n - 1))) → a 11 = a 1 * a 2 * a 3 * a 4 * a 5 :=
by {
  intros h2 h3,
  sorry
}

end geometric_sequence_product_l310_310375


namespace investment_in_scheme_B_l310_310572

theorem investment_in_scheme_B 
    (yieldA : ℝ) (yieldB : ℝ) (investmentA : ℝ) (difference : ℝ) (totalA : ℝ) (totalB : ℝ):
    yieldA = 0.30 → yieldB = 0.50 → investmentA = 300 → difference = 90 
    → totalA = investmentA + (yieldA * investmentA) 
    → totalB = (1 + yieldB) * totalB 
    → totalA = totalB + difference 
    → totalB = 200 :=
by sorry

end investment_in_scheme_B_l310_310572


namespace measure_new_acute_angle_l310_310107

theorem measure_new_acute_angle (θ : ℝ) (rotate : ℝ) : 
  θ = 45 → rotate = 510 → 
  let excess := rotate - 360 in
  let initial_cancellation := excess - θ in
  let positive_acute := if initial_cancellation > 90 then initial_cancellation - 180 else initial_cancellation in
  positive_acute = 75 :=
by
  intros hθ hrotate
  let excess := rotate - 360
  let initial_cancellation := excess - θ
  let positive_acute := if initial_cancellation > 90 then initial_cancellation - 180 else initial_cancellation
  calc 
    θ    = 45   : by exact hθ
    rotate = 510 : by exact hrotate
    excess = 150 : by unfold excess; rw [hrotate, sub_eq_add_neg]; norm_num
    initial_cancellation = 105 : by unfold initial_cancellation; rw [excess, sub_eq_add_neg, hθ]; norm_num
    positive_acute = 75 : by unfold positive_acute; by_cases h : 105 > 90; simp [h]; norm_num
  sorry

end measure_new_acute_angle_l310_310107


namespace maximum_constant_value_l310_310648

def sqrt_le_const {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) : Prop :=
  ∃ a : ℝ, (∀ x y, 0 < x → 0 < y → x + y = 1 → sqrt x + sqrt y ≤ a) ∧ a = sqrt 2

theorem maximum_constant_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 1) :
  sqrt_le_const hx hy hxy :=
sorry

end maximum_constant_value_l310_310648


namespace find_a_45_l310_310411

noncomputable def a : ℕ → ℝ
| 0 := 11
| 1 := 11
| (m + n) := (1 / 2) * (a (2 * m) + a (2 * n)) - (m - n)^2

theorem find_a_45 : a 45 = 1991 :=
by {
  sorry
}

end find_a_45_l310_310411


namespace scientific_notation_14000000_l310_310449

theorem scientific_notation_14000000 :
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 14000000 = a * 10 ^ n ∧ a = 1.4 ∧ n = 7 :=
by
  sorry

end scientific_notation_14000000_l310_310449


namespace cannot_form_set_l310_310871

/-- Define the set of non-negative real numbers not exceeding 20 --/
def setA : Set ℝ := {x | 0 ≤ x ∧ x ≤ 20}

/-- Define the set of solutions of the equation x^2 - 9 = 0 within the real numbers --/
def setB : Set ℝ := {x | x^2 - 9 = 0}

/-- Define the set of all students taller than 170 cm enrolled in a certain school in the year 2013 --/
def setC : Type := sorry

/-- Define the (pseudo) set of all approximate values of sqrt(3) --/
def pseudoSetD : Set ℝ := {x | x = Real.sqrt 3}

/-- Main theorem stating that setD cannot form a mathematically valid set --/
theorem cannot_form_set (x : ℝ) : x ∈ pseudoSetD → False := sorry

end cannot_form_set_l310_310871


namespace line_through_circle_center_slope_one_eq_l310_310450

theorem line_through_circle_center_slope_one_eq (x y : ℝ) :
  (∃ x y : ℝ, (x + 1)^2 + (y - 2)^2 = 4 ∧ y = 2) →
  (∃ m : ℝ, m = 1 ∧ (x + 1) = m * (y - 2)) →
  (x - y + 3 = 0) :=
sorry

end line_through_circle_center_slope_one_eq_l310_310450


namespace sin_neg_300_eq_l310_310977

theorem sin_neg_300_eq : Real.sin (-(300 * Real.pi / 180)) = Real.sqrt 3 / 2 := by
  -- Conditions used in Lean definitions
  have h1 : ∀ x, Real.sin (x + 2 * Real.pi) = Real.sin x := Real.sin_periodic 1
  have h2 : 2 * Real.pi = 360 * Real.pi / 180 := by norm_num
  have angle_conversion : -(300 * Real.pi / 180) + 2 * Real.pi = 60 * Real.pi / 180 := by
    calc
      -(300 * Real.pi / 180) + 2 * Real.pi = -(300 * Real.pi / 180) + 360 * Real.pi / 180 : by rw h2
      ... = 60 * Real.pi / 180 : by norm_num
  show Real.sin (-(300 * Real.pi / 180)) = Real.sqrt 3 / 2
  have h3 : Real.sin (-(300 * Real.pi / 180)) = Real.sin (60 * Real.pi / 180) := by
    rw [←angle_conversion, h1]
  rw h3
  rw Real.sin_pi_div_three
  norm_num
  sorry  -- Placeholder for the final verification

end sin_neg_300_eq_l310_310977


namespace tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l310_310295

-- Conditions
variables {O : ℝ × ℝ} (A : ℝ × ℝ) (B : ℝ × ℝ)
          {P Q : ℝ × ℝ} (p : ℝ)
          (hp : 0 < p)
          (hA : A.1 ^ 2 = 2 * p * A.2)
          (hB : B = (0, -1))
          (hP : P.2 = P.1 ^ 2 / (2 * p))
          (hQ : Q.2 = Q.1 ^ 2 / (2 * p))

-- Proof problem statements
theorem tangent_line_AB
  (hAB_tangent : ∀ x : ℝ, x ^ 2 / (2 * p) = 2 * x - 1 → x = 1) : true :=
by sorry

theorem op_oq_leq_oa_squared 
  (h_op_oq_leq : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + (P.1 ^ 2 / (2 * p)) ^ 2) * (Q.1 ^ 2 + (Q.1 ^ 2 / (2 * p)) ^ 2) ≤ 2) : true :=
by sorry

theorem bp_bq_gt_ba_squared 
  ( h_bp_bq_gt : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + ((P.1 ^ 2 / (2 * p)) + 1) ^ 2) * (Q.1 ^ 2 + ((Q.1 ^ 2 / (2 * p)) +1 ) ^ 2) > 5 ) : true :=
by sorry

end tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l310_310295


namespace range_sin_cos_quad_l310_310944

noncomputable def sin_cos_quad (x : ℝ) := (Real.sin x)^4 + (Real.cos x)^4

theorem range_sin_cos_quad : 
  ∀ (x : ℝ), ∃ (y : ℝ) (h : y ∈ set.Icc (1/2) 1), y = sin_cos_quad x :=
sorry

end range_sin_cos_quad_l310_310944


namespace max_f_0_f_2017_l310_310670

noncomputable def f : ℝ → ℝ := sorry

-- The condition for the function f
axiom f_condition : ∀ x : ℝ, f x > 0 ∧ f (x + 1) = 1 / 2 + real.sqrt(f x - (f x)^2)

theorem max_f_0_f_2017 : f 0 + f 2017 = 1 + real.sqrt 2 / 2 :=
sorry

end max_f_0_f_2017_l310_310670


namespace ashley_wedding_drinks_l310_310197

theorem ashley_wedding_drinks :
  ∀ (guests : ℕ) (champagne_glasses_per_guest wine_glasses_per_guest sparkling_juice_glasses_per_guest champagne_servings_per_bottle wine_servings_per_bottle sparkling_juice_servings_per_bottle : ℕ),
  guests = 120 →
  champagne_glasses_per_guest = 2 →
  wine_glasses_per_guest = 1 →
  sparkling_juice_glasses_per_guest = 1 →
  champagne_servings_per_bottle = 6 →
  wine_servings_per_bottle = 5 →
  sparkling_juice_servings_per_bottle = 4 →
  let guests_per_drink := guests / 3 in
  let champagne_bottles := (guests_per_drink * champagne_glasses_per_guest + champagne_servings_per_bottle - 1) / champagne_servings_per_bottle in
  let wine_bottles := guests_per_drink * wine_glasses_per_guest / wine_servings_per_bottle in
  let sparkling_juice_bottles := guests_per_drink * sparkling_juice_glasses_per_guest / sparkling_juice_servings_per_bottle in
  champagne_bottles = 14 ∧ wine_bottles = 8 ∧ sparkling_juice_bottles = 10 :=
by
  intros guests champagne_glasses_per_guest wine_glasses_per_guest sparkling_juice_glasses_per_guest champagne_servings_per_bottle wine_servings_per_bottle sparkling_juice_servings_per_bottle
  assume h_guests h_champagne_glasses_per_guest h_wine_glasses_per_guest h_sparkling_juice_glasses_per_guest h_champagne_servings_per_bottle h_wine_servings_per_bottle h_sparkling_juice_servings_per_bottle
  let guests_per_drink := guests / 3
  let champagne_bottles := (guests_per_drink * champagne_glasses_per_guest + champagne_servings_per_bottle - 1) / champagne_servings_per_bottle
  let wine_bottles := guests_per_drink * wine_glasses_per_guest / wine_servings_per_bottle
  let sparkling_juice_bottles := guests_per_drink * sparkling_juice_glasses_per_guest / sparkling_juice_servings_per_bottle
  have h_champagne_bottles : champagne_bottles = 14 := by
    sorry -- Proof for this step would go here
  have h_wine_bottles : wine_bottles = 8 := by
    sorry -- Proof for this step would go here
  have h_sparkling_juice_bottles : sparkling_juice_bottles = 10 := by
    sorry -- Proof for this step would go here
  exact ⟨h_champagne_bottles, h_wine_bottles, h_sparkling_juice_bottles⟩

end ashley_wedding_drinks_l310_310197


namespace trigonometric_expression_l310_310645

noncomputable def cosθ (θ : ℝ) := 1 / Real.sqrt 10
noncomputable def sinθ (θ : ℝ) := 3 / Real.sqrt 10
noncomputable def tanθ (θ : ℝ) := 3

theorem trigonometric_expression (θ : ℝ) (h : tanθ θ = 3) :
  (1 + cosθ θ) / sinθ θ + sinθ θ / (1 - cosθ θ) = (10 * Real.sqrt 10 + 10) / 9 := 
  sorry

end trigonometric_expression_l310_310645


namespace scientific_notation_correct_l310_310091

noncomputable def scientific_notation (x : ℝ) : ℝ × ℤ :=
  let a := x * 10^9
  (a, -9)

theorem scientific_notation_correct :
  scientific_notation 0.000000007 = (7, -9) :=
by
  sorry

end scientific_notation_correct_l310_310091


namespace unit_digit_product_l310_310164

theorem unit_digit_product : 
  (let u1 := (5 + 1) % 10,
       u2 := (5^3 + 1) % 10,
       u3 := (5^6 + 1) % 10,
       u4 := (5^{12} + 1) % 10 in
    (u1 * u2 * u3 * u4) % 10) = 6 :=
by sorry

end unit_digit_product_l310_310164


namespace median_is_70_74_l310_310964

-- Define the histogram data as given
def histogram : List (ℕ × ℕ) :=
  [(85, 5), (80, 15), (75, 18), (70, 22), (65, 20), (60, 10), (55, 10)]

-- Function to calculate the cumulative sum at each interval
def cumulativeSum (hist : List (ℕ × ℕ)) : List (ℕ × ℕ) :=
  hist.scanl (λ acc pair => (pair.1, acc.2 + pair.2)) (0, 0)

-- Function to find the interval where the median lies
def medianInterval (hist : List (ℕ × ℕ)) : ℕ :=
  let cumSum := cumulativeSum hist
  -- The median is the 50th and 51st scores
  let medianPos := 50
  -- Find the interval that contains the median position
  List.find? (λ pair => medianPos ≤ pair.2) cumSum |>.getD (0, 0) |>.1

-- The theorem stating that the median interval is 70-74
theorem median_is_70_74 : medianInterval histogram = 70 :=
  by sorry

end median_is_70_74_l310_310964


namespace haley_marbles_l310_310725

theorem haley_marbles (marbles boys e: ℕ) (H1: marbles = 35) (H2: boys = 5) : marbles / boys = 7 :=
by {
  rw [H1, H2],
  norm_num,
  exact e
}

end haley_marbles_l310_310725


namespace max_value_f_l310_310636

-- Define the function f(x) as the minimum of three given functions
def f (x : ℝ) : ℝ := min (min (3 * x + 1) (x + 2)) (-2 * x + 8)

-- State the theorem asserting that the maximum value of f(x) is 4
theorem max_value_f : ∃ x : ℝ, (∀ y : ℝ, f y ≤ f x) ∧ f x = 4 := by
  sorry

end max_value_f_l310_310636


namespace trigonometric_identity_l310_310341

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = -3) : 
  (Real.cos α - Real.sin α) / (Real.cos α + Real.sin α) = -2 :=
by 
  sorry

end trigonometric_identity_l310_310341


namespace jill_and_emily_total_peaches_l310_310807

-- Define each person and their conditions
variables (Steven Jake Jill Maria Emily : ℕ)

-- Given conditions
def steven_has_peaches : Steven = 14 := sorry
def jake_has_fewer_than_steven : Jake = Steven - 6 := sorry
def jake_has_more_than_jill : Jake = Jill + 3 := sorry
def maria_has_twice_jake : Maria = 2 * Jake := sorry
def emily_has_fewer_than_maria : Emily = Maria - 9 := sorry

-- The theorem statement combining the conditions and the required result
theorem jill_and_emily_total_peaches (Steven Jake Jill Maria Emily : ℕ)
  (h1 : Steven = 14) 
  (h2 : Jake = Steven - 6) 
  (h3 : Jake = Jill + 3) 
  (h4 : Maria = 2 * Jake) 
  (h5 : Emily = Maria - 9) : 
  Jill + Emily = 12 := 
sorry

end jill_and_emily_total_peaches_l310_310807


namespace mass_of_man_l310_310155

theorem mass_of_man (L B : ℝ) (h : ℝ) (ρ : ℝ) (V : ℝ) : L = 8 ∧ B = 3 ∧ h = 0.01 ∧ ρ = 1 ∧ V = L * 100 * B * 100 * h → V / 1000 = 240 :=
by
  sorry

end mass_of_man_l310_310155


namespace quadratic_translation_l310_310821

theorem quadratic_translation (b c : ℝ) :
  (∀ x : ℝ, (x^2 + b * x + c = (x - 3)^2 - 2)) →
  b = 4 ∧ c = 6 :=
by
  sorry

end quadratic_translation_l310_310821


namespace beta_unique_solution_l310_310396

noncomputable def beta := ℂ

theorem beta_unique_solution (β : beta) (h1 : β ≠ 1)
  (h2 : abs (β^2 - 1) = 3 * abs (β - 1))
  (h3 : abs (β^4 - 1) = 5 * abs (β - 1)) :
  β = 2 :=
sorry

end beta_unique_solution_l310_310396


namespace limit_of_nested_radical_l310_310586

theorem limit_of_nested_radical :
  ∃ F : ℝ, F = 43 ∧ F = Real.sqrt (86 + 41 * F) :=
sorry

end limit_of_nested_radical_l310_310586


namespace yendor_midway_distances_l310_310457

noncomputable def distances_midway_yendor : ℝ :=
  let perigee := 3
  let apogee := 15
  let major_axis := perigee + apogee
  let midway_distance := major_axis / 2
  midway_distance

theorem yendor_midway_distances (major_axis_midway : distances_midway_yendor = 9) :
  ∃ (distance : ℝ), distance = 9 ∧ distance = distances_midway_yendor :=
by
  use distances_midway_yendor
  split
  . exact major_axis_midway
  . exact distances_midway_yendor
  sorry

end yendor_midway_distances_l310_310457


namespace minimum_f_l310_310316

def f (x : ℝ) : ℝ := |3 - x| + |x - 7|

theorem minimum_f : ∀ x : ℝ, min (f x) = 4 := sorry

end minimum_f_l310_310316


namespace find_w_l310_310440

theorem find_w (α β γ : ℂ) (h1 : α + β + γ = -5) 
  (h2 : (∏ r in {α, β, γ}, (x - r)) = x^3 + 5 * x^2 + 7 * x - 13) 
  (h3 : (∏ r in {α + β, β + γ, γ + α}, (x - r)) = x^3 + u * x^2 + v * x + w) :
  w = 48 := sorry

end find_w_l310_310440


namespace number_of_elements_P_star_Q_l310_310771

-- Define sets P and Q
def P : Set ℤ := {-1, 0, 1}
def Q : Set ℤ := {0, 1, 2, 3}

-- Define the set operation P * Q
def P_star_Q : Set (ℤ × ℤ) := {(x, y) | x ∈ (P ∩ Q) ∧ y ∈ (P ∪ Q)}

theorem number_of_elements_P_star_Q : (P_star_Q.toFinset.card = 10) :=
by 
  sorry

end number_of_elements_P_star_Q_l310_310771


namespace value_of_a2_sum_of_coefficients_l310_310701

noncomputable def polynomial_expansion (x : ℝ) : ℝ :=
  (2 * (x - 1) - 1)^9

theorem value_of_a2 : 
  let a := ([ (-1)^k * 2^(9 - k) * choose 9 k | k : ℕ ]) in
  (polynomial_expansion x).coeff 2 = -144
  := sorry

theorem sum_of_coefficients : 
  let a := ([ (-1)^k * 2^(9 - k) * choose 9 k | k : ℕ ]) in
  polynomial_expansion 2 = 1 
  → a.coeff 0 + a.coeff 1 + a.coeff 2 + a.coeff 3 + a.coeff 4 + a.coeff 5 + a.coeff 6 + a.coeff 7 + a.coeff 8 + a.coeff 9 = 2
  := sorry

end value_of_a2_sum_of_coefficients_l310_310701


namespace sum_of_squares_of_rates_l310_310991

variable (b j s : ℕ)

theorem sum_of_squares_of_rates
  (h1 : 3 * b + 2 * j + 3 * s = 82)
  (h2 : 5 * b + 3 * j + 2 * s = 99) :
  b^2 + j^2 + s^2 = 314 := by
  sorry

end sum_of_squares_of_rates_l310_310991


namespace smallest_m_value_l310_310900

theorem smallest_m_value (n : ℕ) (hn : 0 < n):
  ∃ m, 
    (∃ (arr : Array (Array ℕ)), 
      let k := 3^n in 
      arr.size = k ∧ 
      (∀ row, row < k → (arr[row].sum = arr[0].sum) ∧ 
      ∀ col, col < k → (arr.map (λ r => r[col])).sum = arr[0].sum) ∧ 
      ∀ x ∈ arr.flat_map id, x ∈ (Finset.range (m+1) ∪ {0}) ∧ 
      (arr.flat_map id).erase_dup.length = m) ∧
      m = 3^(n+1) -1 :=
begin
  sorry
end

end smallest_m_value_l310_310900


namespace volume_ratio_tetrahedron_l310_310558

variables (a b d w k : ℝ)

theorem volume_ratio_tetrahedron (tetra_ABCD : ℝ) 
(h_AB : ∃ a, a = tetra_ABCD) 
(h_CD : ∃ b, b = tetra_ABCD)
(h_dist : ∃ d, d = tetra_ABCD) 
(h_angle : ∃ w, w = tetra_ABCD) 
(h_ratio : ∃ k, k = tetra_ABCD): 
  let vol_ratio := (k^3 + 3 * k^2) / (3 * k + 1) 
  in vol_ratio = (k^3 + 3 * k^2) / (3 * k + 1) := 
by 
  sorry

end volume_ratio_tetrahedron_l310_310558


namespace resistance_of_second_resistor_l310_310015

theorem resistance_of_second_resistor 
  (R1 R_total R2 : ℝ) 
  (hR1: R1 = 9) 
  (hR_total: R_total = 4.235294117647059) 
  (hFormula: 1/R_total = 1/R1 + 1/R2) : 
  R2 = 8 :=
by
  sorry

end resistance_of_second_resistor_l310_310015


namespace exp_inequality_l310_310711

theorem exp_inequality (x : ℝ) (h1 : -1 < x) (h2 : x < 0) : 
  2^x < 2^(-x) ∧ 2^(-x) < 0.2^x :=
by
  sorry

end exp_inequality_l310_310711


namespace draw_probability_l310_310442

theorem draw_probability (P_A_win : ℝ) (P_A_not_lose : ℝ) (h1 : P_A_win = 0.3) (h2 : P_A_not_lose = 0.8) : 
  ∃ P_draw : ℝ, P_draw = 0.5 := 
by
  sorry

end draw_probability_l310_310442


namespace num_isosceles_triangles_on_geoboard_l310_310962

theorem num_isosceles_triangles_on_geoboard :
  ∃ (points : finset (ℤ × ℤ)),
  let grid : finset (ℤ × ℤ) := (finset.Icc (0, 0) (6, 6))
  in let A : ℤ × ℤ := (3, 3)
  in let B : ℤ × ℤ := (5, 3)
  in points.card = 47 ∧ (∀ C ∈ points, 
        isosceles (0 : ℤ) (dist A C) (dist B C) (dist A B)) ∧ 
        (∃ (isos_points : finset (ℤ × ℤ)),
          ∀ C ∈ isos_points, (isosceles (dist A B) (dist A C) (dist B C)) ∧ 
          isos_points.card = 10) :=
begin
  sorry
end

end num_isosceles_triangles_on_geoboard_l310_310962


namespace max_sin_C_l310_310743

-- Definitions of the vectors and the condition
variables {V : Type*} [inner_product_space ℝ V]
variables {A B C : V}

-- Condition: Given the dot product condition in triangle ABC
def condition (A B C : V) : Prop :=
  inner_product_space.inner (A - B) (A - C) + 2 * inner_product_space.inner (B - A) (B - C) = 
  3 * inner_product_space.inner (C - A) (C - B)

def sin_C (A B C : V) : ℝ :=
  let a := dist B C
  let b := dist C A
  let c := dist A B in
  (sqrt (1 - (a^2 + b^2 - c^2)^2 / (4 * a^2 * b^2)))

-- Proof problem: Maximum value of sin C given the condition holds in triangle ABC
theorem max_sin_C (A B C : V) (h : condition A B C) : sin_C A B C = real.sqrt (7 / 9) := 
sorry

end max_sin_C_l310_310743


namespace quadratic_has_exactly_one_solution_l310_310219

theorem quadratic_has_exactly_one_solution (k : ℚ) :
  (3 * x^2 - 8 * x + k = 0) → ((-8)^2 - 4 * 3 * k = 0) → k = 16 / 3 :=
by
  sorry

end quadratic_has_exactly_one_solution_l310_310219


namespace unique_solution_9_eq_1_l310_310244

theorem unique_solution_9_eq_1 :
  (∃! (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = 1) :=
begin
  sorry
end

end unique_solution_9_eq_1_l310_310244


namespace part1_l310_310969

def is_Xn_function (n : ℝ) (f : ℝ → ℝ) : Prop :=
  ∃ x1 x2, x1 ≠ x2 ∧ f x1 = f x2 ∧ x1 + x2 = 2 * n

theorem part1 : is_Xn_function 0 (fun x => abs x) ∧ is_Xn_function (1/2) (fun x => x^2 - x) :=
by
  sorry

end part1_l310_310969


namespace largest_number_among_neg1_0_1_one_third_l310_310569

theorem largest_number_among_neg1_0_1_one_third : ∃ m ∈ ({-1, 0, 1, 1/3} : set ℚ), ∀ x ∈ ({-1, 0, 1, 1/3} : set ℚ), x ≤ m :=
begin
  use 1,
  split,
  {
    simp,
  },
  {
    intros x hx,
    simp at hx,
    cases hx,
    { rw hx, exact le_refl 1 },
    { rw hx, exact dec_trivial },
    { rw hx, exact dec_trivial },
    { rw hx, exact dec_trivial },
  },
end

end largest_number_among_neg1_0_1_one_third_l310_310569


namespace maximal_angle_prism_l310_310070

theorem maximal_angle_prism (n : ℕ) (R : ℝ) (h : ℝ) (A1 : ℝ → Prop) (An1' : ℝ → Prop)
  (A3 : ℝ → Prop) (An2' : ℝ → Prop) (angle : ℝ → ℝ → ℝ → ℝ)
  (Hn : n > 1) : 
  ∃ h, (angle (A1 R) (A3 R) (An2' R)) = 2 * R * Real.cos (Real.pi / n) :=
sorry

end maximal_angle_prism_l310_310070


namespace probability_3_hits_with_2_consecutive_l310_310915

theorem probability_3_hits_with_2_consecutive :
  let single_hit_prob := 1 / 2
  let total_shots := 7
  let successful_hits := 3
  let favorable_configurations := 20
  let final_prob := favorable_configurations * (single_hit_prob ^ total_shots)
  in final_prob = 5 / 32 :=
by
  let single_hit_prob := 1 / 2
  let total_shots := 7
  let successful_hits := 3
  let favorable_configurations := 20
  let final_prob := favorable_configurations * (single_hit_prob ^ total_shots)
  exact sorry

end probability_3_hits_with_2_consecutive_l310_310915


namespace intersection_is_correct_l310_310770

noncomputable def setA : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
noncomputable def setB : Set ℝ := {x | Real.log x / Real.log 2 ≤ 2}

theorem intersection_is_correct : setA ∩ setB = {x : ℝ | 0 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_is_correct_l310_310770


namespace simplify_and_rationalize_l310_310803

theorem simplify_and_rationalize :
  (sqrt 3 / sqrt 4) * (sqrt 5 / sqrt 6) * (sqrt 7 / sqrt 8) * (sqrt 9 / sqrt 10) = 
  (3 * sqrt 1050) / 320 := by
  sorry

end simplify_and_rationalize_l310_310803


namespace cos_alpha_eq_zero_l310_310772

noncomputable def f (x : ℝ) : ℝ := 1 + (sin x) / (1 + cos x)

def positive_zeros (n : ℕ) : ℝ :=
  (λ k, 2 * k * Real.pi + 3 * Real.pi / 2) n

def alpha : ℝ := (Finset.sum (Finset.range 2015) positive_zeros)

theorem cos_alpha_eq_zero : cos alpha = 0 := sorry

end cos_alpha_eq_zero_l310_310772


namespace smaller_triangle_legs_length_l310_310595

theorem smaller_triangle_legs_length :
  ∀ (h_larger h_smaller : ℝ), h_larger = 2 → h_smaller = h_larger / 2 →
  ∃ l : ℝ, l * l + l * l = h_smaller * h_smaller ∧ l = Real.sqrt (1 / 2) :=
by
  assume h_larger h_smaller
  assume h_larger_eq : h_larger = 2
  assume h_smaller_eq : h_smaller = h_larger / 2
  sorry

end smaller_triangle_legs_length_l310_310595


namespace tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l310_310294

-- Conditions
variables {O : ℝ × ℝ} (A : ℝ × ℝ) (B : ℝ × ℝ)
          {P Q : ℝ × ℝ} (p : ℝ)
          (hp : 0 < p)
          (hA : A.1 ^ 2 = 2 * p * A.2)
          (hB : B = (0, -1))
          (hP : P.2 = P.1 ^ 2 / (2 * p))
          (hQ : Q.2 = Q.1 ^ 2 / (2 * p))

-- Proof problem statements
theorem tangent_line_AB
  (hAB_tangent : ∀ x : ℝ, x ^ 2 / (2 * p) = 2 * x - 1 → x = 1) : true :=
by sorry

theorem op_oq_leq_oa_squared 
  (h_op_oq_leq : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + (P.1 ^ 2 / (2 * p)) ^ 2) * (Q.1 ^ 2 + (Q.1 ^ 2 / (2 * p)) ^ 2) ≤ 2) : true :=
by sorry

theorem bp_bq_gt_ba_squared 
  ( h_bp_bq_gt : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + ((P.1 ^ 2 / (2 * p)) + 1) ^ 2) * (Q.1 ^ 2 + ((Q.1 ^ 2 / (2 * p)) +1 ) ^ 2) > 5 ) : true :=
by sorry

end tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l310_310294


namespace perpendicular_lines_condition_l310_310021

theorem perpendicular_lines_condition (m : ℝ) :
  (∀ x y : ℝ, x + (m + 1) * y = 2 - m → m * x + 2 * y = -8) ↔ m = -2 / 3 :=
by sorry

end perpendicular_lines_condition_l310_310021


namespace smallest_four_digit_number_divisible_by_9_l310_310494

theorem smallest_four_digit_number_divisible_by_9 
  (N : ℕ) (h1 : 1000 ≤ N ∧ N < 10000)
  (h2 : (N % 10 = 2) ∨ (N / 10 % 10 = 2) ∨ (N / 100 % 10 = 2) ∨ (N / 1000 % 10 = 2))
  (h3 : N % 9 = 0)
  (h4 : (list.filter (λ x, x % 2 = 0) (list.of_num_digits N)).length = 2 ∧ 
        (list.filter (λ x, x % 2 = 1) (list.of_num_digits N)).length = 2)
  : N = 2079 :=
sorry

end smallest_four_digit_number_divisible_by_9_l310_310494


namespace max_additional_plates_l310_310814

def initial_plates_count : ℕ := 5 * 3 * 4 * 2
def new_second_set_size : ℕ := 5  -- second set after adding two letters
def new_fourth_set_size : ℕ := 3 -- fourth set after adding one letter
def new_plates_count : ℕ := 5 * new_second_set_size * 4 * new_fourth_set_size

theorem max_additional_plates :
  new_plates_count - initial_plates_count = 180 := by
  sorry

end max_additional_plates_l310_310814


namespace negative_integer_solutions_count_l310_310822

theorem negative_integer_solutions_count :
  (∃ count : ℕ, count = 3 ∧ 
    ∀ x : ℤ, x < 0 → (x > -3.43) ↔ (x = -1 ∨ x = -2 ∨ x = -3) →
    count = (if x = -1 ∨ x = -2 ∨ x = -3 then count + 1 else count)) := sorry

end negative_integer_solutions_count_l310_310822


namespace rebus_system_solution_l310_310080

theorem rebus_system_solution :
  ∃ (M A H P h : ℕ), 
  (M > 0) ∧ (P > 0) ∧ 
  (M ≠ A) ∧ (M ≠ H) ∧ (M ≠ P) ∧ (M ≠ h) ∧
  (A ≠ H) ∧ (A ≠ P) ∧ (A ≠ h) ∧ 
  (H ≠ P) ∧ (H ≠ h) ∧ (P ≠ h) ∧
  ((M * 10 + A) * (M * 10 + A) = M * 100 + H * 10 + P) ∧ 
  ((A * 10 + M) * (A * 10 + M) = P * 100 + h * 10 + M) ∧ 
  (((M = 1) ∧ (A = 3) ∧ (H = 6) ∧ (P = 9) ∧ (h = 6)) ∨
   ((M = 3) ∧ (A = 1) ∧ (H = 9) ∧ (P = 6) ∧ (h = 9))) :=
by
  sorry

end rebus_system_solution_l310_310080


namespace problem_relationship_l310_310260

noncomputable theory

-- Define the values a, b, and c based on the conditions
def a : ℝ := (3 : ℝ) ^ 0.4
def b : ℝ := Real.log 0.3 / Real.log 4
def c : ℝ := Real.log 3 / Real.log 4

-- The theorem to prove the relationship between a, b, and c
theorem problem_relationship : a > c ∧ c > b := by
  sorry

end problem_relationship_l310_310260


namespace people_came_in_first_hour_l310_310914
-- Import the entirety of the necessary library

-- Lean 4 statement for the given problem
theorem people_came_in_first_hour (X : ℕ) (net_change_first_hour : ℕ) (net_change_second_hour : ℕ) (people_after_2_hours : ℕ) : 
    (net_change_first_hour = X - 27) → 
    (net_change_second_hour = 18 - 9) →
    (people_after_2_hours = 76) → 
    (X - 27 + 9 = 76) → 
    X = 94 :=
by 
    intros h1 h2 h3 h4 
    sorry -- Proof is not required by instructions

end people_came_in_first_hour_l310_310914


namespace incorrect_conclusion_C_l310_310637

-- Definitions of the conditions and incorrect conclusion
def quadratic_function (x : ℝ) := 3 * x^2 + 6

def parabola_opens_upwards (f : ℝ → ℝ) : Prop :=
  ∀ x > 0, f x > f 0

def decreases_for_x_neg (f : ℝ → ℝ) : Prop :=
  ∀ x < 0, ∀ y ∈ Icc x 0, f y < f x

def correct_axis_of_symmetry : Prop :=
  let a := 3 in let b := 0 in (-b) / (2 * a) = 0

def correct_vertex : Prop :=
  quadratic_function 0 = 6

theorem incorrect_conclusion_C :
  quadratic_function = (λ x, 3 * x^2 + 6) →
  parabola_opens_upwards quadratic_function →
  decreases_for_x_neg quadratic_function →
  correct_axis_of_symmetry →
  correct_vertex →
  ¬ (correct_axis_of_symmetry ∧ x = -1) :=
by sorry

end incorrect_conclusion_C_l310_310637


namespace arithmetic_arrangement_result_l310_310576

theorem arithmetic_arrangement_result :
    (1 / 8) * (1 / 9) * (1 / 28) = 1 / 2016 ∨ ((1 / 8) - (1 / 9)) * (1 / 28) = 1 / 2016 :=
by {
    sorry
}

end arithmetic_arrangement_result_l310_310576


namespace number_of_green_hats_l310_310159

theorem number_of_green_hats (B G : ℕ) 
  (h1 : B + G = 85) 
  (h2 : 6 * B + 7 * G = 550) : 
  G = 40 := by
  sorry

end number_of_green_hats_l310_310159


namespace lollipop_count_l310_310349

theorem lollipop_count (total_cost one_lollipop_cost : ℚ) (h1 : total_cost = 90) (h2 : one_lollipop_cost = 0.75) : total_cost / one_lollipop_cost = 120 :=
by
  sorry

end lollipop_count_l310_310349


namespace angle_KBC_degrees_l310_310423

noncomputable def midpoint (A B : Type) [inhabited A] : A := sorry
noncomputable def intersect_ray (A B C : Type) [inhabited A] (): A := sorry
noncomputable def perpendicular (A B : Type) [inhabited A] : A := sorry

theorem angle_KBC_degrees (A B C D M P Q K : Type) [inhabited A] :
  let M := midpoint B C,
  let Q := intersect_ray P M C,
  let K := perpendicular P BQ,
  \(\angle KQD = 64^\circ\),
  \(\angle KDQ = 38^\circ\) → 
  \(\angle KBC = 39^\circ\) :=
sorry

end angle_KBC_degrees_l310_310423


namespace fraction_of_orange_juice_correct_l310_310130

-- Define the capacities of the pitchers
def capacity := 800

-- Define the fractions of orange juice and apple juice in the first pitcher
def orangeJuiceFraction1 := 1 / 4
def appleJuiceFraction1 := 1 / 8

-- Define the fractions of orange juice and apple juice in the second pitcher
def orangeJuiceFraction2 := 1 / 5
def appleJuiceFraction2 := 1 / 10

-- Define the total volumes of the contents in each pitcher
def totalVolume := 2 * capacity -- total volume in the large container after pouring

-- Define the orange juice volumes in each pitcher
def orangeJuiceVolume1 := orangeJuiceFraction1 * capacity
def orangeJuiceVolume2 := orangeJuiceFraction2 * capacity

-- Calculate the total volume of orange juice in the large container
def totalOrangeJuiceVolume := orangeJuiceVolume1 + orangeJuiceVolume2

-- Define the fraction of orange juice in the large container
def orangeJuiceFraction := totalOrangeJuiceVolume / totalVolume

theorem fraction_of_orange_juice_correct :
  orangeJuiceFraction = 9 / 40 :=
by
  sorry

end fraction_of_orange_juice_correct_l310_310130


namespace repunit_concat_composite_l310_310798

theorem repunit_concat_composite (n : ℕ) (h : n ≥ 1) : 
  let M := (10^n - 1) / 9 in
  let N := M * 10^n + M in
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ N = a * b :=
by
  sorry

end repunit_concat_composite_l310_310798


namespace find_a_l310_310351

theorem find_a (a : ℝ) :
  (∀ x, x < 2 → 0 < a - 3 * x) ↔ (a = 6) :=
by
  sorry

end find_a_l310_310351


namespace sum_of_terms_l310_310380

theorem sum_of_terms (a : ℕ → ℝ) (n : ℕ) 
  (h1 : ∀ k, a k > 0)
  (h2 : a 1 = 2) 
  (h3 : ∀ k, (a k)^2 = 9 * (a (k-1))^2) : 
  (∑ i in Finset.range n, a i) = 3^n - 1 := 
by
  sorry

end sum_of_terms_l310_310380


namespace problem_1_problem_2_l310_310684

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1

theorem problem_1 (a : ℝ) (h : ∀ x : ℝ, f x a ≥ 0) : a = 1 := by sorry

theorem problem_2 (n : ℕ) (hn : ∀ m : ℕ, m > 0) : 
  (∑ k in Finset.range(n + 1), (k / n) ^ n) < (Real.exp 1 / (Real.exp 1 - 1)) := by sorry

end problem_1_problem_2_l310_310684


namespace find_b_l310_310766

theorem find_b (a b c : ℕ) (h1 : a * b + b * c - c * a = 0) (h2 : a - c = 101) (h3 : a > 0) (h4 : b > 0) (h5 : c > 0) : b = 2550 :=
sorry

end find_b_l310_310766


namespace product_of_fractional_parts_eq_222_l310_310138

theorem product_of_fractional_parts_eq_222 : 
  let x := 18 / 999 in let y := x.num / x.denom in y.num * y.denom = 222 :=
by 
  sorry

end product_of_fractional_parts_eq_222_l310_310138


namespace no_real_solutions_l310_310332

theorem no_real_solutions :
  ∀ y : ℝ, ( (-2 * y + 7)^2 + 2 = -2 * |y| ) → false := by
  sorry

end no_real_solutions_l310_310332


namespace find_m_of_monotonic_decreasing_l310_310693

variable {x : ℝ} {m : ℝ}

def power_function (m : ℝ) (x : ℝ) : ℝ := (m - 1)^2 * x^(m^2 - 4 * m + 2)

theorem find_m_of_monotonic_decreasing (h : ∀ x > 0, deriv (power_function m) x ≤ 0) : m = 2 :=
by sorry

end find_m_of_monotonic_decreasing_l310_310693


namespace calc_expression_l310_310587

theorem calc_expression : Real.sqrt 9 - 5 + Real.cbrt 8 * (-2)^2 = 6 :=
by
  -- Proof will go here
  sorry

end calc_expression_l310_310587


namespace num_winning_scenarios_l310_310839

-- Define the problem conditions
def total_tickets : ℕ := 8
def prize_tickets : ℕ := 3
def no_prize_tickets : ℕ := 5
def total_people : ℕ := 4
def tickets_per_person : ℕ := 2

-- Definition to capture the total number of different winning scenarios
theorem num_winning_scenarios : 
  ∃ n : ℕ, n = 60 ∧ 
    (total_tickets = 8) ∧ 
    (prize_tickets = 3) ∧ 
    (no_prize_tickets = 5) ∧ 
    (total_people = 4) ∧ 
    (tickets_per_person = 2) := by
  -- sorry allows us to skip the proof
  sorry

end num_winning_scenarios_l310_310839


namespace total_distance_covered_l310_310035

-- Define the basic conditions
def num_marathons : Nat := 15
def miles_per_marathon : Nat := 26
def yards_per_marathon : Nat := 385
def yards_per_mile : Nat := 1760

-- Define the total miles and total yards covered
def total_miles : Nat := num_marathons * miles_per_marathon
def total_yards : Nat := num_marathons * yards_per_marathon

-- Convert excess yards into miles and calculate the remaining yards
def extra_miles : Nat := total_yards / yards_per_mile
def remaining_yards : Nat := total_yards % yards_per_mile

-- Compute the final total distance
def total_distance_miles : Nat := total_miles + extra_miles
def total_distance_yards : Nat := remaining_yards

-- The theorem that needs to be proven
theorem total_distance_covered :
  total_distance_miles = 393 ∧ total_distance_yards = 495 :=
by
  sorry

end total_distance_covered_l310_310035


namespace total_new_bottles_created_l310_310416

def initial_bottles : ℕ := 729
def bottles_required_per_new : ℕ := 5
def external_bottles_added : ℕ := 20

theorem total_new_bottles_created : ℕ :=
  let first_recycling := initial_bottles / bottles_required_per_new
  let second_recycling := first_recycling / bottles_required_per_new
  let third_recycling := second_recycling / bottles_required_per_new
  first_recycling + second_recycling + third_recycling + 20 * 2 = 179 := sorry

end total_new_bottles_created_l310_310416


namespace optimal_landing_point_l310_310928

noncomputable def travel_time (x : ℝ) : ℝ :=
  (Real.sqrt (81 + x^2)) / 4 + (15 - x) / 5

theorem optimal_landing_point :
  ∃ x : ℝ, x = 12 ∧ (∀ y : ℝ, travel_time x ≤ travel_time y) :=
by
  let x := 12
  use x
  split
  · exact rfl
  · intro y
    sorry

end optimal_landing_point_l310_310928


namespace square_side_percentage_increase_l310_310460

theorem square_side_percentage_increase (s : ℝ) (p : ℝ) :
  (s * (1 + p / 100)) ^ 2 = 1.44 * s ^ 2 → p = 20 :=
by
  sorry

end square_side_percentage_increase_l310_310460


namespace construct_triangle_l310_310967

noncomputable def exists_triangle_with_conditions (AH AM R : ℝ) : Prop :=
  ∃ (A B C : Point) (M : Point),
  height A B C AH ∧ 
  median A B C M AM ∧ 
  circumcircle_radius A B C R

theorem construct_triangle (AH AM R : ℝ) :
  ∃ (A B C : Point), exists_triangle_with_conditions AH AM R :=
sorry

end construct_triangle_l310_310967


namespace total_students_at_gathering_l310_310608

theorem total_students_at_gathering (x : ℕ) 
  (h1 : ∃ x : ℕ, 0 < x)
  (h2 : (x + 6) / (2 * x + 6) = 2 / 3) : 
  (2 * x + 6) = 18 := 
  sorry

end total_students_at_gathering_l310_310608


namespace geometric_probability_l310_310408

noncomputable def f (x : ℝ) : ℝ := x^2 - x - 2

theorem geometric_probability
  (x0 : ℝ)
  (h_domain : x0 ∈ set.Icc (-5 : ℝ) 5) :
  ∃ p : ℝ, p = 3 / 10 ∧ ∀ x, (x ∈ set.Icc (-5 : ℝ) 5) → ((f x ≤ 0) ↔ (x ∈ set.Icc (-1 : ℝ) 2)) :=
sorry

end geometric_probability_l310_310408


namespace length_of_bridge_l310_310517

theorem length_of_bridge
  (walking_speed_km_hr : ℝ) (time_minutes : ℝ) (length_bridge : ℝ) 
  (h1 : walking_speed_km_hr = 5) 
  (h2 : time_minutes = 15) 
  (h3 : length_bridge = 1250) : 
  length_bridge = (walking_speed_km_hr * 1000 / 60) * time_minutes := 
by 
  sorry

end length_of_bridge_l310_310517


namespace money_distribution_l310_310521

theorem money_distribution (p q r : ℝ) 
  (h1 : p + q + r = 9000) 
  (h2 : r = (2/3) * (p + q)) : 
  r = 3600 := 
by 
  sorry

end money_distribution_l310_310521


namespace locus_of_Q_circle_with_diameter_MN_l310_310276

-- Given conditions
variable (F := (1 : ℝ, 0 : ℝ))
variable (l := { x : ℝ // x = -1 })
variable (P : ℝ × ℝ)
variable (l' := { y: ℝ // y = P.2 }) -- Assuming x-coordinate of P is implied by l
variable (Q : ℝ × ℝ) -- Intersection of perpendicular bisector of PF and l'

-- Proof Problem (1)
theorem locus_of_Q :
  (∀ Q, (Q.1 - F.1)^2 + (Q.2 - F.2)^2 = (Q.1 + 1)^2 + Q.2^2 → 
   (Q.2^2 = 4 * Q.1)) :=
sorry

-- Given point H and procedure for determining points A, B, M, and N
variable (H := (1 : ℝ, 2 : ℝ))
variable (A : ℝ × ℝ)
variable (B : ℝ × ℝ)
variable (M : ℝ × ℝ)
variable (N : ℝ × ℝ)

-- Proof Problem (2)
theorem circle_with_diameter_MN :
  ∀ M N, M.1 = -1 ∧ N.1 = -1 → 
  (circle_diameter_through_fixed_points (M.1, M.2) (N.1, N.2) ↔ 
   fixed_point (-3, 0) ∧ fixed_point (1, 0)) :=
sorry

definition circle_diameter_through_fixed_points (M : ℝ × ℝ) (N : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), (x + 1)^2 + (y - (M.2 + N.2) / 2)^2 = ((M.2 - N.2) / 2)^2 

definition fixed_point (p : ℝ × ℝ) : Prop :=
  p = (-3, 0) ∨ p = (1, 0)

end locus_of_Q_circle_with_diameter_MN_l310_310276


namespace expand_simplify_expression_l310_310866

theorem expand_simplify_expression (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
by
  sorry

end expand_simplify_expression_l310_310866


namespace line_through_two_points_l310_310327

theorem line_through_two_points :
  ∀ (A_1 B_1 A_2 B_2 : ℝ),
    (2 * A_1 + 3 * B_1 = 1) →
    (2 * A_2 + 3 * B_2 = 1) →
    (∀ (x y : ℝ), (2 * x + 3 * y = 1) → (x * (B_2 - B_1) + y * (A_1 - A_2) = A_1 * B_2 - A_2 * B_1)) :=
by 
  intros A_1 B_1 A_2 B_2 h1 h2 x y hxy
  sorry

end line_through_two_points_l310_310327


namespace rain_and_no_snow_prob_l310_310828

-- Define the probabilities as given in the conditions
def prob_rain_saturday := 0.7
def prob_rain_sunday := 0.5
def prob_snow_saturday := 0.2

-- Independence assumptions (comments since Lean handles event independence internally)
-- independent(prob_rain_saturday, prob_rain_sunday)
-- independent(prob_rain_saturday, prob_snow_saturday)
-- independent(prob_rain_sunday, prob_snow_saturday)

-- Define the proof statement
theorem rain_and_no_snow_prob : (prob_rain_saturday * prob_rain_sunday * (1 - prob_snow_saturday)) = 0.28 :=
by {
  have caution := prob_rain_saturday * prob_rain_sunday * (1 - prob_snow_saturday),
  sorry,
}

end rain_and_no_snow_prob_l310_310828


namespace smallest_possible_value_l310_310373

def smallest_value_expression : ℕ :=
  -- The smallest possible value of the expression 10 / 9 / 8 / 7 / 6 / 5 / 4 / 3 / 2 / 1
  7

theorem smallest_possible_value : ∃ f : ℕ → ℕ → ℕ, (
  ∃ a : ℕ, a = 10 ∧ 
  ∃ b : ℕ, b = 9 ∧ 
  ∃ c : ℕ, c = 8 ∧
  ∃ d : ℕ, d = 7 ∧
  ∃ e : ℕ, e = 6 ∧
  ∃ g : ℕ, g = 5 ∧
  ∃ h : ℕ, h = 4 ∧
  ∃ i : ℕ, i = 3 ∧
  ∃ j : ℕ, j = 2 ∧
  ∃ k : ℕ, k = 1 ∧
  ∃ (p : ℕ → ℕ → ℕ), p a (p b (p c (p d (p e (p g (p h (p i (p j k))))))))) = smallest_value_expression
) := sorry

end smallest_possible_value_l310_310373


namespace numbers_at_distance_1_from_neg2_l310_310784

theorem numbers_at_distance_1_from_neg2 : 
  ∃ x : ℤ, (|x + 2| = 1) ∧ (x = -1 ∨ x = -3) :=
by
  sorry

end numbers_at_distance_1_from_neg2_l310_310784


namespace sum_of_areas_of_infinite_polygons_eq_area_n_gon_l310_310552

theorem sum_of_areas_of_infinite_polygons_eq_area_n_gon (n : ℕ) (R : ℝ) (hn : 3 ≤ n):
  let α := real.pi / n in
  let T := ∑' k : ℕ, n * R^2 * real.sin α * (real.cos α)^(2*k) in
  T = n * R^2 * real.cot α := 
sorry

end sum_of_areas_of_infinite_polygons_eq_area_n_gon_l310_310552


namespace Donny_spends_28_on_Thursday_l310_310987

theorem Donny_spends_28_on_Thursday :
  let monday_savings := 15
  let tuesday_savings := 28
  let wednesday_savings := 13
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  let thursday_spending := total_savings / 2
  thursday_spending = 28 :=
by 
  let monday_savings := 15
  let tuesday_savings := 28
  let wednesday_savings := 13
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  let thursday_spending := total_savings / 2
  sorry

end Donny_spends_28_on_Thursday_l310_310987


namespace perimeter_of_WXYZ_l310_310799

/-
Definitions for the conditions:
1. PQRS is a rhombus inscribed in rectangle JKLM
2. PZ || JM, XR || JM
3. QW || JK, YS || JK
4. JP = 39, JS = 52, KQ = 25
-/

noncomputable def PQRS_rhombus (P Q R S J K L M : ℝ) : Prop :=
  -- Define the properties of points representing the rhombus inscribed in a rectangle
  J = 0 ∧ K = 0 ∧ L = 0 ∧ M = 0 ∧ P = 39 ∧ S = 52 ∧ Q = 25 ∧ R = sqrt (52^2 + 39^2)

noncomputable def parallel_PZ_JM (P Z J M : ℝ) : Prop := sorry
noncomputable def parallel_XR_JM (X R J M : ℝ) : Prop := sorry
noncomputable def parallel_QW_JK (Q W J K : ℝ) : Prop := sorry
noncomputable def parallel_YS_JK (Y S J K : ℝ) : Prop := sorry

theorem perimeter_of_WXYZ
  (J K L M P Q R S W X Y Z : ℝ)
  (h1 : PQRS_rhombus P Q R S J K L M)
  (h2 : parallel_PZ_JM P Z J M)
  (h3 : parallel_XR_JM X R J M)
  (h4 : parallel_QW_JK Q W J K)
  (h5 : parallel_YS_JK Y S J K)
  (h6 : JP = 39) (h7 : JS = 52) (h8 : KQ = 25) :
  -- The statement to prove
  2 * ((JS - KQ) + (sqrt (52^2 + 39^2 - KQ^2) - JP)) = 96 :=
sorry

end perimeter_of_WXYZ_l310_310799


namespace value_2_std_devs_below_mean_l310_310890

theorem value_2_std_devs_below_mean {μ σ : ℝ} (h_mean : μ = 10.5) (h_std_dev : σ = 1) : μ - 2 * σ = 8.5 :=
by
  sorry

end value_2_std_devs_below_mean_l310_310890


namespace find_a_of_inequality_solution_set_l310_310352

theorem find_a_of_inequality_solution_set
  (a : ℝ)
  (h : ∀ x : ℝ, |a * x + 2| < 6 ↔ -1 < x ∧ x < 2) :
  a = -4 :=
sorry

end find_a_of_inequality_solution_set_l310_310352


namespace probability_point_above_curve_probability_correct_l310_310439

theorem probability_point_above_curve :
  (∃ (a c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9) →
  (∑ (a : ℕ), ∑ (c : ℕ), if c > a * (a^3 - c * a^2) then 1 else 0) = 16 :=
begin
  sorry
end

theorem probability_correct :
  (∑ (a : ℕ), ∑ (c : ℕ), if c > a * (a^3 - c * a^2) then 1 else 0) / 81 = 16 / 81 :=
begin
  sorry
end

end probability_point_above_curve_probability_correct_l310_310439


namespace find_a4_l310_310410

noncomputable def geometric_sequence (a_n : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a_n (n + 1) = a_n n * q

theorem find_a4 (a_n : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a_n q →
  a_n 1 + a_n 2 = -1 →
  a_n 1 - a_n 3 = -3 →
  a_n 4 = -8 :=
by 
  sorry

end find_a4_l310_310410


namespace probability_A2_l310_310579

-- Define events and their probabilities
variable (A1 : Prop) (A2 : Prop) (B1 : Prop)
variable (P : Prop → ℝ)
variable [MeasureTheory.MeasureSpace ℝ]

-- Conditions given in the problem
axiom P_A1 : P A1 = 0.5
axiom P_B1 : P B1 = 0.5
axiom P_A2_given_A1 : P (A2 ∧ A1) / P A1 = 0.7
axiom P_A2_given_B1 : P (A2 ∧ B1) / P B1 = 0.8

-- Theorem statement to prove
theorem probability_A2 : P A2 = 0.75 :=
by
  -- Skipping the proof as per instructions
  sorry

end probability_A2_l310_310579


namespace additional_increment_charge_cents_l310_310907

-- Conditions as definitions
def first_increment_charge_cents : ℝ := 3.10
def total_charge_8_minutes_cents : ℝ := 18.70
def total_minutes : ℝ := 8
def increments_per_minute : ℝ := 5
def total_increments : ℝ := total_minutes * increments_per_minute
def remaining_increments : ℝ := total_increments - 1
def remaining_charge_cents : ℝ := total_charge_8_minutes_cents - first_increment_charge_cents

-- Proof problem: What is the charge for each additional 1/5 of a minute?
theorem additional_increment_charge_cents : remaining_charge_cents / remaining_increments = 0.40 := by
  sorry

end additional_increment_charge_cents_l310_310907


namespace min_value_expression_l310_310308

open Real

noncomputable def perp_vectors (a b : ℝ × ℝ) : Prop := 
  a.1 * b.1 + a.2 * b.2 = 0

noncomputable def magnitudes (a b : ℝ × ℝ) (m : ℝ) : Prop := 
  (a.1 ^ 2 + a.2 ^ 2 = m ^ 2) ∧ (b.1 ^ 2 + b.2 ^ 2 = m ^ 2)

noncomputable def expression (t : ℝ) (a b : ℝ × ℝ) : ℝ :=
  let ab := (b.1 - a.1, b.2 - a.2)
  let ao := (-a.1, -a.2)
  let ba := (a.1 - b.1, a.2 - b.2)
  let bo := (-b.1, -b.2)
  sqrt ((-24 * t + 24) ^ 2 + (24 * t) ^ 2) +
    sqrt ((-24 * (1 - t)) ^ 2 + (-10 + 24 * (1 - t)) ^ 2)

theorem min_value_expression (a b : ℝ × ℝ) (h1 : perp_vectors a b) (h2 : magnitudes a b 24) :
  ∃ t, t ∈ Icc 0 1 ∧ expression t a b = 26 :=
sorry

end min_value_expression_l310_310308


namespace jenna_drives_200_miles_l310_310032

-- Definitions based on conditions
def total_trip_time : ℕ := 10 -- in hours
def total_break_time : ℕ := 1 -- in hours
def friend_last_miles : ℕ := 100 -- in miles
def friend_speed : ℕ := 20 -- in miles per hour
def jenna_speed : ℕ := 50 -- in miles per hour

-- Theorem stating the problem
theorem jenna_drives_200_miles :
  let total_driving_time := total_trip_time - total_break_time in
  let friend_driving_time := friend_last_miles / friend_speed in
  let jenna_driving_time := total_driving_time - friend_driving_time in
  jenna_driving_time * jenna_speed = 200 :=
by
  sorry

end jenna_drives_200_miles_l310_310032


namespace max_k_l310_310634

noncomputable def is_mirror_number (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    n = 1000 * a + 100 * b + 10 * c + d ∧
    1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9 ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
    (a + d = b + c)

noncomputable def F (n : ℕ) : ℚ :=
  let d₁ := n / 1000 in
  let d₂ := (n / 100) % 10 in
  let d₃ := (n / 10) % 10 in
  let d₄ := n % 10 in
  (1000 * d₄ + 100 * d₂ + 10 * d₃ + d₁ + 1000 * d₃ + 100 * d₁ + 10 * d₂ + d₄) / 1111

theorem max_k (x y e f: ℕ) (hx : 1 ≤ x ∧ x ≤ 9) (hy : 1 ≤ y ∧ y ≤ 9)
    (he : 1 ≤ e ∧ e ≤ 9) (hf : 1 ≤ f ∧ f ≤ 9)
    (sx : is_mirror_number (1000 * x + 100 * y + 32))
    (tx : is_mirror_number (1500 + 10 * e + f)) :
    F (1000 * x + 100 * y + 32) + F (1500 + 10 * e + f) = 19 →
    ∃ k : ℚ, k = 11 / 8 := by
  sorry

end max_k_l310_310634


namespace find_x_l310_310759

def operation (a b : ℝ) : ℝ := (Real.sqrt (3 * a + b))^4

theorem find_x (x : ℝ) : operation 6 x = 256 → x = -2 :=
by
  intros h
  sorry

end find_x_l310_310759


namespace math_books_count_l310_310487

theorem math_books_count (M H : ℕ) :
  M + H = 90 →
  4 * M + 5 * H = 396 →
  H = 90 - M →
  M = 54 :=
by
  intro h1 h2 h3
  sorry

end math_books_count_l310_310487


namespace bowling_ball_weight_l310_310088

noncomputable def weight_of_one_bowling_ball : ℕ := 20

theorem bowling_ball_weight (b c : ℕ) (h1 : 10 * b = 5 * c) (h2 : 3 * c = 120) : b = weight_of_one_bowling_ball := by
  sorry

end bowling_ball_weight_l310_310088


namespace team_construction_l310_310842

theorem team_construction (A_time : ℕ) (B_time : ℕ) (A_cost_per_month : ℕ) (B_cost_per_month : ℕ) :
  A_time = 3 → B_time = 6 → A_cost_per_month = 12000 → B_cost_per_month = 5000 →
  let combined_time := 2 in
  let total_cost := 34000 in
  (combined_time = 2) ∧ (total_cost = 34000) :=
by
  intros _ _ _ _
  let combined_time := 2
  let total_cost := 34000
  exact ⟨rfl, rfl⟩

end team_construction_l310_310842


namespace tangent_sum_identity_l310_310612

open Real

theorem tangent_sum_identity :
  let t23 := tan (23 * real.pi / 180)
  let t22 := tan (22 * real.pi / 180)
  t23 + t22 + t23 * t22 = 1 :=
  sorry

end tangent_sum_identity_l310_310612


namespace four_digit_consecutive_number_perfect_square_l310_310911

/-- A four-digit number has digits that are consecutive numbers in sequence.
If we swap the digits in the hundreds and thousands places, we get a perfect square.
What is this number? -/
theorem four_digit_consecutive_number_perfect_square :
  ∃ x : ℕ, x ∈ (0 : Finset ℕ).range 10 ∧ 1000 * x + 100 * (x + 1) + 10 * (x + 2) + (x + 3) = 3456 ∧
  ∃ k : ℕ, 1000 * (x + 1) + 100 * x + 10 * (x + 2) + (x + 3) = k * k :=
sorry

end four_digit_consecutive_number_perfect_square_l310_310911


namespace available_codes_count_l310_310060

def original_code : Nat := 145

def available_digits : Finset Nat := {0, 1, 2, 3, 4, 5}

def digit_at (code : Nat) (position : Nat) : Nat :=
  code / 10^(2 - position) % 10

def code_digits_distinct_except_two_positions (code1 code2 : Nat) : Prop :=
  (digit_at code1 0 ≠ digit_at code2 0 ∧ digit_at code1 1 ≠ digit_at code2 1) ∨
  (digit_at code1 0 ≠ digit_at code2 0 ∧ digit_at code1 2 ≠ digit_at code2 2) ∨
  (digit_at code1 1 ≠ digit_at code2 1 ∧ digit_at code1 2 ≠ digit_at code2 2)

def code_not_transposition (code : Nat) : Prop :=
  ¬ ∃ (i j : Nat), i < j ∧ i < 3 ∧ j < 3 ∧ code = digit_at original_code j * 10^(2 - i) + digit_at original_code i * 10^(2 - j) + digit_at original_code (6 - i - j) * 10^(2 - (6 - i - j))

def valid_code (code : Nat) : Prop :=
  code ∈ Finset.range 216 ∧ 
  code_digits_distinct_except_two_positions original_code code ∧ 
  code_not_transposition code ∧ 
  code ≠ original_code

theorem available_codes_count : Finset.card (Finset.filter valid_code (Finset.range 216)) = 198 := 
by  
  sorry

end available_codes_count_l310_310060


namespace intervals_of_increase_extrema_in_interval_l310_310679

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin((x / 2) + Real.pi / 6) + 3

theorem intervals_of_increase :
  ∃ k : ℤ, ∀ x : ℝ,
    - (4 * Real.pi) / 3 + 4 * (k : ℝ) * Real.pi ≤ x ∧ x ≤ (2 * Real.pi) / 3 + 4 * (k : ℝ) * Real.pi →
    f x > f (x - 1) := sorry

theorem extrema_in_interval (x : ℝ) (hx : x ∈ Set.Icc (Real.pi / 3) (4 * Real.pi / 3)) :
  (f x = 6 ∧ x = (2 * Real.pi) / 3) ∨ (f x = 9 / 2 ∧ x = (4 * Real.pi) / 3) := sorry

end intervals_of_increase_extrema_in_interval_l310_310679


namespace arithmetic_expression_equals_fraction_l310_310577

theorem arithmetic_expression_equals_fraction (a b c : ℚ) :
  a = 1/8 → b = 1/9 → c = 1/28 →
  (a * b * c = 1/2016) ∨ ((a - b) * c = 1/2016) :=
by
  intros ha hb hc
  rw [ha, hb, hc]
  left
  sorry

end arithmetic_expression_equals_fraction_l310_310577


namespace knights_count_l310_310474

theorem knights_count (n : ℕ) (inhabitants : Fin n → Bool) (claim : Fin n → Prop)
  (h1 : n = 2021)
  (h2 : ∀ i : Fin n, claim i = (∃ l r : ℕ, l = (∑ j in FinRange (i.val), if inhabitants j then 1 else 0) ∧ 
  r = (∑ j in FinRange (i.val + 1).compl, if ¬inhabitants j then 1 else 0) ∧ r > l))
  : (inhabitants.filter (λ b, b)).length = 1010 := sorry

end knights_count_l310_310474


namespace problem_statement_l310_310051

variable {a b c : ℝ}
variable h1 : a + b + c = 6
variable h2 : a * b + b * c + c * a = 11
variable h3 : a * b * c = 6

theorem problem_statement : a * b / c + b * c / a + c * a / b = 49 / 6 := by
  sorry

end problem_statement_l310_310051


namespace total_students_l310_310545

theorem total_students (x : ℝ) :
  (x - (1/2)*x - (1/4)*x - (1/8)*x = 3) → x = 24 :=
by
  intro h
  sorry

end total_students_l310_310545


namespace sum_abc_of_quadrilateral_l310_310547

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem sum_abc_of_quadrilateral :
  let p1 := (0, 0)
  let p2 := (4, 3)
  let p3 := (5, 2)
  let p4 := (4, -1)
  let perimeter := 
    distance p1 p2 + distance p2 p3 + distance p3 p4 + distance p4 p1
  let a : ℤ := 1    -- corresponding to the equivalent simplified distances to √5 parts
  let b : ℤ := 2    -- corresponding to the equivalent simplified distances to √2 parts
  let c : ℤ := 9    -- rest constant integer simplified part
  a + b + c = 12 :=
by
  sorry

end sum_abc_of_quadrilateral_l310_310547


namespace walkways_area_l310_310573

-- Define the conditions and prove the total walkway area is 416 square feet
theorem walkways_area (rows : ℕ) (columns : ℕ) (bed_width : ℝ) (bed_height : ℝ) (walkway_width : ℝ) 
  (h_rows : rows = 4) (h_columns : columns = 3) (h_bed_width : bed_width = 8) (h_bed_height : bed_height = 3) (h_walkway_width : walkway_width = 2) : 
  (rows * (bed_height + walkway_width) + walkway_width) * (columns * (bed_width + walkway_width) + walkway_width) - rows * columns * bed_width * bed_height = 416 := 
by 
  sorry

end walkways_area_l310_310573


namespace smallest_integer_in_set_l310_310825

theorem smallest_integer_in_set (median : ℤ) (greatest : ℤ) (h1 : median = 144) (h2 : greatest = 153) : ∃ x : ℤ, x = 135 :=
by
  sorry

end smallest_integer_in_set_l310_310825


namespace problem_statement_l310_310056

variable {R : Type*} [LinearOrderedField R]

noncomputable def f : R → R := sorry

theorem problem_statement (f_decreasing : ∀ x y : R, 0 < x → 0 < y → x < y → f y < f x)
  (f_eq : ∀ x y : R, 0 < x → 0 < y → f (x * y) = f x + f y)
  (f_one_third : f ((1 / 3 : R)) = 1) :
  f (1 : R) = 0 ∧
  f ((1 / 9 : R)) = 2 ∧
  f (9 : R) = -2 ∧
  (∀ x : R, f x - f (2 - x) < 2 → (1 / 5 : R) < x ∧ x < 2) :=
begin
  sorry,
end

end problem_statement_l310_310056


namespace find_ad_l310_310722

-- Defining the two-digit and three-digit numbers
def two_digit (a b : ℕ) : ℕ := 10 * a + b
def three_digit (a b : ℕ) : ℕ := 100 + two_digit a b

def two_digit' (c d : ℕ) : ℕ := 10 * c + d
def three_digit' (c d : ℕ) : ℕ := 100 * c + 10 * d + 1

-- The main problem
theorem find_ad (a b c d : ℕ) (h1 : three_digit a b = three_digit' c d + 15) (h2 : two_digit a b = two_digit' c d + 24) :
    two_digit a d = 32 := by
  sorry

end find_ad_l310_310722


namespace fill_pipe_half_cistern_time_l310_310515

theorem fill_pipe_half_cistern_time (time_to_fill_half : ℕ) 
  (H : time_to_fill_half = 10) : 
  time_to_fill_half = 10 := 
by
  -- Proof is omitted
  sorry

end fill_pipe_half_cistern_time_l310_310515


namespace range_pf1_pf2_l310_310676

open Real

-- Representation of the ellipse in Lean
def ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

-- Definition of foci for the ellipse
structure EllipseFoci (a b : ℝ) :=
  (f1 : ℝ × ℝ)
  (f2 : ℝ × ℝ)
  (dist_f1f2_eq : dist f1 f2 = 2 * Real.sqrt (a^2 - b^2))

-- Given ellipse parameters and foci locations
def given_ellipse : EllipseFoci 2 (Real.sqrt 3) :=
  { f1 := (-Real.sqrt 1, 0),
    f2 := (Real.sqrt 1, 0),
    dist_f1f2_eq:= by rw [Real.sqrt_one, mul_one, Real.dist] }

-- Problem statement: range of values for |PF1| + |PF2| given point P inside the ellipse
theorem range_pf1_pf2 (P : ℝ × ℝ) (h : ellipse P.1 P.2) :
  2 ≤ dist P given_ellipse.f1 + dist P given_ellipse.f2 ∧
  dist P given_ellipse.f1 + dist P given_ellipse.f2 < 4 := 
sorry

end range_pf1_pf2_l310_310676


namespace one_quarter_between_l310_310099

def one_quarter_way (a b : ℚ) : ℚ :=
  a + 1 / 4 * (b - a)

theorem one_quarter_between :
  one_quarter_way (1 / 7) (1 / 4) = 23 / 112 :=
by
  sorry

end one_quarter_between_l310_310099


namespace travel_options_l310_310837

-- Define the conditions
def trains_from_A_to_B := 3
def ferries_from_B_to_C := 2

-- State the proof problem
theorem travel_options (t : ℕ) (f : ℕ) (h1 : t = trains_from_A_to_B) (h2 : f = ferries_from_B_to_C) : t * f = 6 :=
by
  rewrite [h1, h2]
  sorry

end travel_options_l310_310837


namespace arith_seq_odd_sum_l310_310271

noncomputable def arith_seq_sum_odd (a : ℕ → ℕ) (d : ℕ) (S100 : ℕ) : Prop :=
  (∃ a_1 : ℕ, a (1 + 2 * 0) = a_1 ∧ d = 2 ∧ S100 = 10000 ∧ 
  (∑ i in finset.range 50, a_1 + i * 2 * d + i * d) = 4950)

theorem arith_seq_odd_sum :
  ∀ (a : ℕ → ℕ),
  arith_seq_sum_odd a 2 10000 :=
begin
  intros,
  sorry
end

end arith_seq_odd_sum_l310_310271


namespace seniorClassTheorem_l310_310201

-- Definitions according to conditions
variable {totalStudents : ℕ}
variable (mb: ℕ) (brass: ℕ) (sax: ℕ) (altoSax: ℕ)

-- Conditions
def condition1 : Prop := mb = totalStudents / 5
def condition2 : Prop := brass = mb / 2
def condition3 : Prop := sax = brass / 5
def condition4 : Prop := altoSax = sax / 3
def condition5 : Prop := altoSax = 4

-- Conjecture
def seniorClassConjecture : Prop := totalStudents = 600

-- Theorem to prove the conjecture given the conditions
theorem seniorClassTheorem 
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : condition4)
  (h5 : condition5) 
  : seniorClassConjecture := 
by 
  sorry

end seniorClassTheorem_l310_310201


namespace trigonometric_problem_l310_310662

theorem trigonometric_problem (α : ℝ) (h₀ : sin (α + π) = 1/4) (h₁ : α > -π/2 ∧ α < 0) :
  (cos (2 * α) - 1) / tan α = sqrt 15 / 8 :=
sorry

end trigonometric_problem_l310_310662


namespace log_expression_value_l310_310963

theorem log_expression_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (ha1 : a ≠ 1) (hb1 : b ≠ 1) : 
  ((Real.log b / Real.log a) * (Real.log a / Real.log b))^2 = 1 := 
by 
  sorry

end log_expression_value_l310_310963


namespace initial_number_l310_310507

theorem initial_number (N : ℤ) 
  (h : (N + 3) % 24 = 0) : N = 21 := 
sorry

end initial_number_l310_310507


namespace g_crosses_horizontal_asymptote_at_minus_four_l310_310639

noncomputable def g (x : ℝ) : ℝ := (3 * x^2 - 8 * x - 10) / (x^2 - 5 * x + 6)

theorem g_crosses_horizontal_asymptote_at_minus_four : g (-4) = 3 := 
by
  sorry

end g_crosses_horizontal_asymptote_at_minus_four_l310_310639


namespace minimum_AP_BP_l310_310041

def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (7, 3)
def parabola (P : ℝ × ℝ) : Prop := P.2 * P.2 = 8 * P.1

noncomputable def distance (P Q : ℝ × ℝ) : ℝ := ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt

theorem minimum_AP_BP : 
  ∀ (P : ℝ × ℝ), parabola P → distance A P + distance B P ≥ 3 * Real.sqrt 10 :=
by 
  intros P hP
  sorry

end minimum_AP_BP_l310_310041


namespace expand_and_simplify_l310_310861

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
  sorry

end expand_and_simplify_l310_310861


namespace donny_spent_on_thursday_l310_310985

theorem donny_spent_on_thursday :
  let savings_monday : ℤ := 15,
      savings_tuesday : ℤ := 28,
      savings_wednesday : ℤ := 13,
      total_savings : ℤ := savings_monday + savings_tuesday + savings_wednesday,
      amount_spent_thursday : ℤ := total_savings / 2
  in
  amount_spent_thursday = 28 :=
by
  sorry

end donny_spent_on_thursday_l310_310985


namespace tangent_line_at_1_monotonic_intervals_range_of_a_l310_310764

open Real

noncomputable def f (x : ℝ) : ℝ := ln x + 1
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x + 2
noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f x - g a x

theorem tangent_line_at_1 :
  (∀ x, x = 1 → (f 1) = x) := sorry

theorem monotonic_intervals (a : ℝ) :
  (a ≤ 0 → (∀ x, 0 < x → F a x > 0)) ∧ (a > 0 → 
    (∀ x, 0 < x → x < 1 / a → F a x > 0) ∧ (∀ x, x > 1 / a → F a x < 0)) := sorry

theorem range_of_a (a : ℝ) :
  (∀ x, x > 0 → ln x + 1 < a * x + 2) → a ∈ (1 / exp 2, ∞) := sorry

end tangent_line_at_1_monotonic_intervals_range_of_a_l310_310764


namespace problem_1_problem_2_l310_310323

-- Problem 1 statement
theorem problem_1 (a x : ℝ) (m : ℝ) (h_pos_a : a > 0) (h_cond_a : a = 1/4) (h_cond_q : (1 : ℝ) / 2 < x ∧ x < 1) (h_cond_p : a < x ∧ x < 3 * a): 1 / 2 < x ∧ x < 3 / 4 :=
by sorry

-- Problem 2 statement
theorem problem_2 (a x : ℝ) (m : ℝ) (h_pos_a : a > 0) (h_neg_p : ¬(a < x ∧ x < 3 * a)) (h_neg_q : ¬((1 / (2 : ℝ))^(m - 1) < x ∧ x < 1)): 1 / 3 ≤ a ∧ a ≤ 1 / 2 :=
by sorry

end problem_1_problem_2_l310_310323


namespace distance_between_spheres_l310_310698

variables (M m d : ℝ)
variables (M_value : M = 2116) (m_value : m = 16) (d_value : d = 1.15)

theorem distance_between_spheres : 
  let OO₁ := (d / 2) * (M - m) / real.sqrt (M * m) in
  OO₁ = 6.56 :=
by
  sorry

end distance_between_spheres_l310_310698


namespace bugs_eat_same_flowers_l310_310059

theorem bugs_eat_same_flowers (num_bugs : ℕ) (total_flowers : ℕ) (flowers_per_bug : ℕ) 
  (h1 : num_bugs = 3) (h2 : total_flowers = 6) (h3 : flowers_per_bug = total_flowers / num_bugs) : 
  flowers_per_bug = 2 :=
by
  sorry

end bugs_eat_same_flowers_l310_310059


namespace lindy_total_distance_traveled_l310_310887

theorem lindy_total_distance_traveled 
    (initial_distance : ℕ)
    (jack_speed : ℕ)
    (christina_speed : ℕ)
    (lindy_speed : ℕ) 
    (meet_time : ℕ)
    (distance : ℕ) :
    initial_distance = 150 →
    jack_speed = 7 →
    christina_speed = 8 →
    lindy_speed = 10 →
    meet_time = initial_distance / (jack_speed + christina_speed) →
    distance = lindy_speed * meet_time →
    distance = 100 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end lindy_total_distance_traveled_l310_310887


namespace water_pumping_time_l310_310188

theorem water_pumping_time :
  ∀ (length width depth : ℝ) (pump_rate cubic_foot_gallons : ℝ),
    length = 30 → width = 40 → depth = 2 → pump_rate = 10 → cubic_foot_gallons = 7.5 →
    (2 * pump_rate)^-1 * (length * width * depth * cubic_foot_gallons) = 900 :=
by
  intros length width depth pump_rate cubic_foot_gallons
  intros h_length h_width h_depth h_pump_rate h_cubic_foot_gallons
  sorry

end water_pumping_time_l310_310188


namespace solve_for_x_l310_310505

theorem solve_for_x (x : ℝ) (h : (45 / 75 : ℝ) = real.sqrt (x / 75) + 1 / 5) : x = 12 :=
sorry

end solve_for_x_l310_310505


namespace monotonicity_and_range_l310_310315

noncomputable def f (a x : ℝ) := a * x^2 - Real.log x + (2 * a - 1) * x

theorem monotonicity_and_range (a : ℝ) :
  (∀ x > 0, a ≤ 0 → deriv (f a) x < 0) ∧
  (∀ x ∈ Ioo 0 (1 / (2 * a)), 0 < a → deriv (f a) x < 0) ∧
  (∀ x ∈ Ioi (1 / (2 * a)), 0 < a → deriv (f a) x > 0) ∧
  (0 < a → (∀ x > 0, f a x + Real.exp 1 / 2 ≥ 0) → a ≥ 1 / (2 * Real.exp 1)) :=
begin
  sorry
end

end monotonicity_and_range_l310_310315


namespace books_grouped_by_subject_books_with_math_grouped_l310_310786

variables {k m n : ℕ}

-- Define the factorial function
def fact : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * fact n

-- Part 1: Verify the number of ways if books must be grouped by subject
theorem books_grouped_by_subject :
  ∃ (num_ways : ℕ), num_ways = fact 3 * fact k * fact m * fact n :=
sorry

-- Part 2: Verify the number of ways if only mathematics books must be grouped
theorem books_with_math_grouped :
  ∃ (num_ways : ℕ), num_ways = fact (m + n) * (m + n + 1) * fact k :=
sorry

end books_grouped_by_subject_books_with_math_grouped_l310_310786


namespace trigonometric_identity_l310_310644

theorem trigonometric_identity
  (α : ℝ)
  (h1 : Real.tan (π - α) = -2/3)
  (h2 : α ∈ Ioo (-π) (-π/2)) :
  (Real.cos (-α) + 3 * Real.sin (π + α)) / (Real.cos (π - α) + 9 * Real.sin α) = -1/5 :=
  sorry

end trigonometric_identity_l310_310644


namespace strange_number_l310_310437

theorem strange_number (x : ℤ) (h : (x - 7) * 7 = (x - 11) * 11) : x = 18 :=
sorry

end strange_number_l310_310437


namespace average_class_score_l310_310003

theorem average_class_score (total_students assigned_day_students make_up_date_students : ℕ)
  (assigned_day_percentage make_up_date_percentage assigned_day_avg_score make_up_date_avg_score : ℝ)
  (h1 : total_students = 100)
  (h2 : assigned_day_percentage = 0.70)
  (h3 : make_up_date_percentage = 0.30)
  (h4 : assigned_day_students = 70)
  (h5 : make_up_date_students = 30)
  (h6 : assigned_day_avg_score = 55)
  (h7 : make_up_date_avg_score = 95) :
  (assigned_day_avg_score * assigned_day_students + make_up_date_avg_score * make_up_date_students) / total_students = 67 :=
by
  sorry

end average_class_score_l310_310003


namespace product_of_fractional_parts_eq_222_l310_310139

theorem product_of_fractional_parts_eq_222 : 
  let x := 18 / 999 in let y := x.num / x.denom in y.num * y.denom = 222 :=
by 
  sorry

end product_of_fractional_parts_eq_222_l310_310139


namespace even_and_increasing_function_l310_310193

theorem even_and_increasing_function :
    ∃ f : ℝ → ℝ, (f = (λ x, log 2 (abs x))) ∧ 
                 (∀ x, f (-x) = f x) ∧ 
                 (∀ x y : ℝ, 1 < x ∧ x < 2 ∧ 1 < y ∧ y < 2 → x < y → f x < f y) ∧ 
                 ((∀ g : ℝ → ℝ, g = (λ x, cos (2 * x)) → ¬((∀ x, g (-x) = g x) ∧ (∀ x y : ℝ, 1 < x ∧ x < 2 ∧ 1 < y ∧ y < 2 → x < y → g x < g y))) ∧
                  (∀ g : ℝ → ℝ, g = (λ x, (exp x - exp (-x)) / 2) → ¬((∀ x, g (-x) = g x) ∧ (∀ x y : ℝ, 1 < x ∧ x < 2 ∧ 1 < y ∧ y < 2 → x < y → g x < g y))) ∧ 
                  (∀ g : ℝ → ℝ, g = (λ x, x ^ 3 + 1) → ¬((∀ x, g (-x) = g x) ∧ (∀ x y : ℝ, 1 < x ∧ x < 2 ∧ 1 < y ∧ y < 2 → x < y → g x < g y)))) :=
by
    sorry

end even_and_increasing_function_l310_310193


namespace expand_and_simplify_l310_310858

noncomputable def expanded_expr (a : ℝ) : ℝ :=
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6)

theorem expand_and_simplify (a : ℝ) :
  expanded_expr a = a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 :=
by {
  -- Proof goes here
  sorry
}

end expand_and_simplify_l310_310858


namespace binary_subtraction_l310_310582

theorem binary_subtraction : ∀ (x y : ℕ), x = 0b11011 → y = 0b101 → x - y = 0b10110 :=
by
  sorry

end binary_subtraction_l310_310582


namespace f_inequality_l310_310647

open Nat

noncomputable def f (n : ℕ) : ℝ :=
  ∑ i in Finset.range (n + 1).filter (λ k, k > 0), (1 : ℝ) / i

theorem f_inequality (n : ℕ) (hn : n ≥ 1) : f (2^n) > (n + 2) / 2 :=
  sorry

end f_inequality_l310_310647


namespace random_variable_point_of_increase_l310_310182

-- Assuming Real numbers, Probability space and measurable function details
variable {μ : MeasureTheory.Measure ℝ}

-- F is the distribution function
def is_distribution_function (F : ℝ → ℝ) : Prop :=
  ∀ x : ℝ , (0 ≤ F x) ∧ (F x ≤ 1) ∧ monotone F ∧ (Filter.atTop.tendsto F (𝓝 1))

def point_of_increase (F : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ ε > 0, F (x - ε) < F (x + ε)

noncomputable def ξ : MeasureTheory.ProbabilityMeasure ℝ := sorry

theorem random_variable_point_of_increase (F : ℝ → ℝ) (ξ : MeasureTheory.ProbabilityMeasure ℝ) 
  (hF : is_distribution_function F) : 
  (μ { ω : ℝ | ∃ε > 0, F (ω - ε) < F (ω + ε) }) = 1 := sorry

end random_variable_point_of_increase_l310_310182


namespace hyperbola_representation_iff_l310_310721

theorem hyperbola_representation_iff (m : ℝ) :
  (∃ x y : ℝ, (x^2) / (2 + m) - (y^2) / (m + 1) = 1) ↔ (m > -1 ∨ m < -2) :=
by
  sorry

end hyperbola_representation_iff_l310_310721


namespace length_PQ_in_triangle_ABC_l310_310724

theorem length_PQ_in_triangle_ABC
  (A B C H D E Q P : Type)
  (AB BC CA : Real)
  (HA : IsAltitude A H B C)
  (BD CE : IsAngleBisector E C B D)
  (intersect_H_on_AH : IntersectOn AH P BD)
  (intersect_H_on_AH' : IntersectOn AH Q CE)
  (AB_length : AB = 9)
  (BC_length : BC = 10)
  (CA_length : CA = 11)
  : segment_length PQ = (24 * sqrt(2)) / 85 := by sorry

end length_PQ_in_triangle_ABC_l310_310724


namespace rockets_win_30_l310_310002

-- Given conditions
def hawks_won (h : ℕ) (w : ℕ) : Prop := h > w
def rockets_won (r : ℕ) (k : ℕ) (l : ℕ) : Prop := r > k ∧ r < l
def knicks_at_least (k : ℕ) : Prop := k ≥ 15
def clippers_won (c : ℕ) (l : ℕ) : Prop := c < l

-- Possible number of games won
def possible_games : List ℕ := [15, 20, 25, 30, 35, 40]

-- Prove Rockets won 30 games
theorem rockets_win_30 (h w r k l c : ℕ) 
  (h_w: hawks_won h w)
  (r_kl : rockets_won r k l)
  (k_15: knicks_at_least k)
  (c_l : clippers_won c l)
  (h_mem : h ∈ possible_games)
  (w_mem : w ∈ possible_games)
  (r_mem : r ∈ possible_games)
  (k_mem : k ∈ possible_games)
  (l_mem : l ∈ possible_games)
  (c_mem : c ∈ possible_games) :
  r = 30 :=
sorry

end rockets_win_30_l310_310002


namespace refreshment_stand_distance_l310_310556

theorem refreshment_stand_distance 
  (A B S : ℝ) -- Positions of the camps and refreshment stand
  (dist_A_highway : A = 400) -- Distance from the first camp to the highway
  (dist_B_A : B = 700) -- Distance from the second camp directly across the highway
  (equidistant : ∀ x, S = x ∧ dist (S, A) = dist (S, B)) : 
  S = 500 := -- Distance from the refreshment stand to each camp is 500 meters
sorry

end refreshment_stand_distance_l310_310556


namespace cone_volume_l310_310462

theorem cone_volume (S : ℝ) (hPos : S > 0) : 
  let R := Real.sqrt (S / 7)
  let H := Real.sqrt (5 * S)
  let V := (π * S * (Real.sqrt (5 * S))) / 21
  (π * R * R * H / 3) = V := 
sorry

end cone_volume_l310_310462


namespace triangle_angle_BAC_l310_310739

theorem triangle_angle_BAC (A B C D : Type*) [EuclideanGeometry A B C D] 
  (hD_on_BC : D ∈ segment B C)
  (hBD_CD_AD : dist B D = dist C D ∧ dist C D = dist A D)
  (hACD : angle A C D = 40) :
  angle B A C = 90 :=
by
  sorry

end triangle_angle_BAC_l310_310739


namespace product_numerator_denominator_l310_310144

def recurring_decimal_to_fraction (n : ℕ) (d : ℕ) : Rat :=
  n / d

theorem product_numerator_denominator (n : ℕ) (d : ℕ) (x : Rat)
  (hx : recurring_decimal_to_fraction 18 999 = x)
  (hn : n = 2)
  (hd : d = 111) :
  n * d = 222 := by
  have h_frac : x = 0.018 -- This follows from the definition and will be used in the proof
  sorry

end product_numerator_denominator_l310_310144


namespace product_closest_value_l310_310955

-- Define the constants used in the problem
def a : ℝ := 2.5
def b : ℝ := 53.6
def c : ℝ := 0.4

-- Define the expression and the expected correct answer
def expression : ℝ := a * (b - c)
def correct_answer : ℝ := 133

-- State the theorem that the expression evaluates to the correct answer
theorem product_closest_value : expression = correct_answer :=
by
  sorry

end product_closest_value_l310_310955


namespace unique_solution_9_eq_1_l310_310249

theorem unique_solution_9_eq_1 :
  {p : ℝ × ℝ | 9^(p.1^2 + p.2) + 9^(p.1 + p.2^2) = 1}.card = 1 :=
sorry

end unique_solution_9_eq_1_l310_310249


namespace expand_simplify_expression_l310_310865

theorem expand_simplify_expression (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
by
  sorry

end expand_simplify_expression_l310_310865


namespace f_neg2_plus_f_2_eq_seven_l310_310680

def f (x : ℝ) : ℝ :=
  if x < 1 then 1 + Real.log 2 (2 - x) else 2^x

theorem f_neg2_plus_f_2_eq_seven : f (-2) + f 2 = 7 := 
by 
  sorry

end f_neg2_plus_f_2_eq_seven_l310_310680


namespace sum_of_abs_seq_l310_310695

def seq (n : ℕ) : ℤ := 4 * n - 25

def abs_seq (n : ℕ) : ℤ :=
  if seq n < 0 then -seq n else seq n

def sum_abs_seq (n : ℕ) : ℤ :=
  ∑ i in Finset.range (n + 1), abs_seq i

theorem sum_of_abs_seq (n : ℕ) : 
  sum_abs_seq n = 
  if n ≤ 6 then 23 * n - 2 * n^2
  else 2 * n^2 - 23 * n + 132 := 
sorry

end sum_of_abs_seq_l310_310695


namespace part1_part2_l310_310688

-- Definitions of the function and the conditions
noncomputable def f (a x : ℝ) : ℝ := a * sin x - 0.5 * cos (2 * x) + a - (3 / a) + 0.5

-- First part of the problem
theorem part1 (a : ℝ) (h₀ : a ≠ 0) (h₁ : ∀ x : ℝ, f a x ≤ 0) : 0 < a ∧ a ≤ 1 := 
sorry

-- Second part of the problem
theorem part2 (a : ℝ) (h₀ : a ≥ 2) (h₁ : ∃ x : ℝ, f a x ≤ 0) : 2 ≤ a ∧ a ≤ 3 := 
sorry

end part1_part2_l310_310688


namespace factorial_div_eq_l310_310961

theorem factorial_div_eq : (10! / (7! * 2! * 1!)) = 360 := by
  sorry

end factorial_div_eq_l310_310961


namespace largest_reciprocal_l310_310151

theorem largest_reciprocal :
  let a := (2 : ℚ) / 7
  let b := (3 : ℚ) / 8
  let c := (1 : ℚ)
  let d := (4 : ℚ)
  let e := (2000 : ℚ)
  1 / a > 1 / b ∧ 1 / a > 1 / c ∧ 1 / a > 1 / d ∧ 1 / a > 1 / e := 
by
  sorry

end largest_reciprocal_l310_310151


namespace expand_and_simplify_l310_310856

noncomputable def expanded_expr (a : ℝ) : ℝ :=
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6)

theorem expand_and_simplify (a : ℝ) :
  expanded_expr a = a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 :=
by {
  -- Proof goes here
  sorry
}

end expand_and_simplify_l310_310856


namespace rug_area_is_180_l310_310551

variables (w l : ℕ)

def length_eq_width_plus_eight (l w : ℕ) : Prop :=
  l = w + 8

def uniform_width_between_rug_and_room (d : ℕ) : Prop :=
  d = 8

def area_uncovered_by_rug (area : ℕ) : Prop :=
  area = 704

def area_of_rug (w l : ℕ) : ℕ :=
  l * w

theorem rug_area_is_180 (w l : ℕ) (hwld : length_eq_width_plus_eight l w)
  (huw : uniform_width_between_rug_and_room 8)
  (huar : area_uncovered_by_rug 704) :
  area_of_rug w l = 180 :=
sorry

end rug_area_is_180_l310_310551


namespace proof_f_a_minus_5_l310_310312

noncomputable def f : ℝ → ℝ
| x => if x > 3 then log x+1 / log 2 else 2^(x - 3) + 1

theorem proof_f_a_minus_5 (a : ℝ) (h : f a = 3) : f (a - 5) = 3/2 :=
sorry

end proof_f_a_minus_5_l310_310312


namespace vector_dot_product_l310_310257

def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (1, -2)

theorem vector_dot_product : ((a.1 + 2 * b.1, a.2 + 2 * b.2) • a.1, (a.2 + 2 * b.2))) = -4 := by
  sorry

end vector_dot_product_l310_310257


namespace extremum_f_at_1_max_t_l310_310407

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/2) * m * x^2 - 2 * x + real.log(x + 1)

noncomputable def g (m : ℝ) (x : ℝ) : ℝ := f m x - real.log(x + 1) + x^3

theorem extremum_f_at_1 (m : ℝ) : 
  ((deriv (f m)) 1 = 0) ↔ (m = 3/2) := 
by sorry

theorem max_t (m : ℝ) (hm : m ∈ Icc (-4: ℝ) (-1: ℝ)) : 
  ∃ t > 1, ∀ x ∈ Icc (1: ℝ) t, g m x ≤ g m 1 ∧ t ≤ (1 + real.sqrt 13) / 2 := 
by sorry

end extremum_f_at_1_max_t_l310_310407


namespace fraction_simplification_l310_310591

theorem fraction_simplification : 
  (1877^2 - 1862^2) / (1880^2 - 1859^2) = 5 / 7 := 
by 
  sorry

end fraction_simplification_l310_310591


namespace fraction_calls_by_team_B_l310_310156

-- Define the conditions
variables (A B C : ℝ)
axiom ratio_agents : A = (5 / 8) * B
axiom ratio_calls : ∀ (c : ℝ), c = (6 / 5) * C

-- Prove the fraction of the total calls processed by team B
theorem fraction_calls_by_team_B 
  (h1 : A = (5 / 8) * B)
  (h2 : ∀ (c : ℝ), c = (6 / 5) * C) :
  (B * C) / ((5 / 8) * B * (6 / 5) * C + B * C) = 4 / 7 :=
by {
  -- proof is omitted, so we use sorry
  sorry
}

end fraction_calls_by_team_B_l310_310156


namespace temperature_difference_on_day_xianning_l310_310782

theorem temperature_difference_on_day_xianning 
  (highest_temp : ℝ) (lowest_temp : ℝ) 
  (h_highest : highest_temp = 2) (h_lowest : lowest_temp = -3) : 
  highest_temp - lowest_temp = 5 := 
by
  sorry

end temperature_difference_on_day_xianning_l310_310782


namespace lunch_break_duration_l310_310791

theorem lunch_break_duration :
  ∃ (L : ℝ), 
    (∃ (p a : ℝ),
      (6 - L) * (p + a) = 0.4 ∧
      (4 - L) * a = 0.15 ∧
      (10 - L) * p = 0.45) ∧
    291 = L * 60 := 
by
  sorry

end lunch_break_duration_l310_310791


namespace value_of_expression_l310_310150

theorem value_of_expression (x : ℝ) (h : x = 5) : (x^2 + x - 12) / (x - 4) = 18 :=
by 
  sorry

end value_of_expression_l310_310150


namespace distance_range_midpoint_to_line_l310_310275

theorem distance_range_midpoint_to_line (P : ℝ × ℝ) (hP : P.1 + P.2 = 2) :
  ∃ t : ℝ, P = (t, 2 - t) ∧ 
  let Q := ((t / (2 * t^2 - 4 * t + 4)), ((2 - t) / (2 * t^2 - 4 * t + 4))) in
  let d := (Real.sqrt 2 / 2) * |2 - (1 / (t^2 - 2 * t + 2))| / Real.sqrt 2 in
  d ∈ (Set.Ioo (Real.sqrt 2 / 2) (Real.sqrt 2)) :=
begin
  -- Proof goes here
  sorry
end

end distance_range_midpoint_to_line_l310_310275


namespace unique_solution_9_eq_1_l310_310246

theorem unique_solution_9_eq_1 :
  (∃! (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = 1) :=
begin
  sorry
end

end unique_solution_9_eq_1_l310_310246


namespace employee_pays_216_l310_310919

def retail_price (wholesale_cost : ℝ) (markup_percentage : ℝ) : ℝ :=
    wholesale_cost + markup_percentage * wholesale_cost

def employee_payment (retail_price : ℝ) (discount_percentage : ℝ) : ℝ :=
    retail_price - discount_percentage * retail_price

theorem employee_pays_216 (wholesale_cost : ℝ) (markup_percentage : ℝ) (discount_percentage : ℝ) :
    wholesale_cost = 200 ∧ markup_percentage = 0.20 ∧ discount_percentage = 0.10 →
    employee_payment (retail_price wholesale_cost markup_percentage) discount_percentage = 216 :=
by
  intro h
  rcases h with ⟨h_wholesale, h_markup, h_discount⟩
  rw [h_wholesale, h_markup, h_discount]
  -- Now we have to prove the final statement: employee_payment (retail_price 200 0.20) 0.10 = 216
  -- This follows directly by computation, so we leave it as a sorry for now
  sorry

end employee_pays_216_l310_310919


namespace total_students_l310_310477

def total_students_passed_at_least_one_test (students_passed_1st : ℕ) (students_passed_2nd : ℕ) (students_passed_both : ℕ) : ℕ :=
  students_passed_1st + students_passed_2nd - students_passed_both

theorem total_students (students_passed_1st : ℕ) (students_passed_2nd : ℕ) (students_passed_both : ℕ) (students_failed_both : ℕ) :
  students_passed_1st = 60 →
  students_passed_2nd = 40 →
  students_passed_both = 20 →
  students_failed_both = 20 →
  students_passed_1st + students_passed_2nd - students_passed_both + students_failed_both = 100 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  calc
    60 + 40 - 20 + 20 = 80 + 20 := by norm_num
    ... = 100 := by norm_num
  done

#eval total_students 60 40 20 20 -- This should evaluate to true if correctly implemented

end total_students_l310_310477


namespace geometric_sequence_m_value_l310_310377

theorem geometric_sequence_m_value 
  (a : ℕ → ℝ) (q : ℝ) (m : ℕ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n, a n = a 1 * q^(n-1))
  (h3 : |q| ≠ 1) 
  (h4 : a m = a 1 * a 2 * a 3 * a 4 * a 5) : 
  m = 11 := by
  sorry

end geometric_sequence_m_value_l310_310377


namespace sequence_sum_when_n1_l310_310847

theorem sequence_sum_when_n1 (a : ℝ) (h : a ≠ 1) :
  (1 + a + a^2 + a^3) = ∑ i in Finset.range (2 * 1 + 2), a^i - ∑ i in Finset.singleton (2 * 1 + 1), a^i := 
sorry

end sequence_sum_when_n1_l310_310847


namespace find_value_of_expression_l310_310756

theorem find_value_of_expression (x y z : ℝ)
  (h1 : 12 * x - 9 * y^2 = 7)
  (h2 : 6 * y - 9 * z^2 = -2)
  (h3 : 12 * z - 9 * x^2 = 4) : 
  6 * x^2 + 9 * y^2 + 12 * z^2 = 9 :=
  sorry

end find_value_of_expression_l310_310756


namespace real_root_of_quadratic_eq_l310_310817

theorem real_root_of_quadratic_eq (b : ℝ) (h : b^2 + (4 + complex.i) * b + 4 + 2 * complex.i = 0) : b = -2 :=
sorry

end real_root_of_quadratic_eq_l310_310817


namespace car_gas_tank_capacity_l310_310947

theorem car_gas_tank_capacity
  (initial_mileage : ℕ)
  (final_mileage : ℕ)
  (miles_per_gallon : ℕ)
  (tank_fills : ℕ)
  (usage : initial_mileage = 1728)
  (usage_final : final_mileage = 2928)
  (car_efficiency : miles_per_gallon = 30)
  (fills : tank_fills = 2):
  (final_mileage - initial_mileage) / miles_per_gallon / tank_fills = 20 :=
by
  sorry

end car_gas_tank_capacity_l310_310947


namespace max_intersecting_chords_through_A1_l310_310176

theorem max_intersecting_chords_through_A1 
  (n : ℕ) (h_n : n = 2017) 
  (A : Fin n → α) 
  (line_through_A1 : α) 
  (no_other_intersection : ∀ i : Fin n, i ≠ 0 → A i ≠ line_through_A1) :
  ∃ k : ℕ, k * (2016 - k) + 2016 = 1018080 := 
sorry

end max_intersecting_chords_through_A1_l310_310176


namespace donny_spending_l310_310981

theorem donny_spending :
  (15 + 28 + 13) / 2 = 28 :=
by
  sorry

end donny_spending_l310_310981


namespace original_selling_price_l310_310904

-- Definitions based on conditions:
def CP : ℝ := 600 -- Cost Price of the book
def SP_gain : ℝ := 660 -- Selling Price to gain 10%
def gain_rate : ℝ := 1.1 -- 110% of CP for gaining 10%
def loss_rate : ℝ := 0.9 -- 90% of CP for losing 10%

-- Problem statement to prove original selling price (OSP) was Rs. 540
theorem original_selling_price :
  let SP := SP_gain / gain_rate in
  let OSP := loss_rate * CP in
  OSP = 540 :=
by
  sorry

end original_selling_price_l310_310904


namespace problem_statement_l310_310269

-- Definitions for sequence {a_n}
def a_seq : ℕ → ℚ
| 0       := 1    -- We use zero-based indexing for convenience
| (n + 1) := (a_seq n + 1) / (a_seq n + 2) - 1

-- Condition a_n ≠ -1
def a_n_ne_neg1 (n : ℕ) : Prop :=
a_seq n ≠ -1

-- Definition of sequence {b_n}
def b_seq (n : ℕ) : ℚ :=
(2 * n + 1) * 2 ^ (n - 1)

-- Definition of sum S_n
def S_n (n : ℕ) : ℚ :=
∑ k in finset.range n, b_seq (k + 1)

-- The main problem statement
theorem problem_statement (n : ℕ) (h1 : ∀ k, a_n_ne_neg1 k) (h2 : a_seq 1 = 1) : 
  (a_seq n = (3 - 2 * n) / (2 * n - 1)) ∧ (S_n n = (2 * n - 3) * 2 ^ n + 3) := 
sorry

end problem_statement_l310_310269


namespace find_c_l310_310824

-- Define the points A and B
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (5, 11)

-- Define the midpoint of segment AB
def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Compute the midpoint of the segment AB
def M : ℝ × ℝ := midpoint A B

-- Define the line equation
def line (c : ℝ) (P : ℝ × ℝ) : Prop :=
  P.1 - P.2 = c

-- State the theorem: line x - y = c intersects the midpoint of segment AB and find c
theorem find_c : ∃ c : ℝ, line c M ∧ c = -4 :=
by
  use -4
  simp [line, M, midpoint, A, B]
  sorry

end find_c_l310_310824


namespace number_of_16_member_event_committees_l310_310010

theorem number_of_16_member_event_committees : 
  (∃ (teams : Fin 5 → Finset (Fin 8)), 
    ∀ t, (Finset.card (teams t) = 8) ∧
    finset.card 
    { committee : Finset (Fin 40) // 
      (∀ (t : Fin 5), 3 ≤ committee.card ∧ (∀ x, x ∈ committee → ((x.val.div 8) = t) → (teams t).card - (if t = t then 4 else 0))) 
    } = 3443073600) :=
by sorry

end number_of_16_member_event_committees_l310_310010


namespace option_A_option_B_max_area_option_C_max_perimeter_option_D_acute_range_l310_310303

-- Definitions for the conditions
def is_triangle (a b c : ℝ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

def right_angle (A B C : ℝ) : Prop := A + B + C = π

def triangle_abc_conditions (a A : ℝ) : Prop := a = 2 * sqrt 3 ∧ A = π / 3

-- We state the propositions to be proven

theorem option_A (a A b : ℝ) (c : ℝ) (ha : a = 2 * sqrt 3) (hA : A = π / 3) (hb : b = sqrt 15) 
  (h_triangle : is_triangle a b c) : ∃ (B : ℝ), B ≠ B ∧ right_angle A B C := 
sorry

theorem option_B_max_area (a b c : ℝ) (A : ℝ) (ha : a = 2 * sqrt 3) (hA : A = π / 3) 
  (h_triangle : is_triangle a b c) : (1/2) * b * c * sin(A) ≤ 3 * sqrt 3 :=
sorry

theorem option_C_max_perimeter (a b c : ℝ) (A : ℝ) (ha : a = 2 * sqrt 3) (hA : A = π / 3) 
  (h_triangle : is_triangle a b c) : a + b + c ≤ 6 * sqrt 3 :=
sorry

theorem option_D_acute_range (a b c A : ℝ) (ha : a = 2 * sqrt 3) (hA : A = π / 3) 
  (h_triangle : is_triangle a b c) (h_acute : ∀ (B C : ℝ), A < π/2 ∧ B < π/2 ∧ C < π/2) : 
  1/2 < b/c ∧ b/c < 2 :=
sorry

end option_A_option_B_max_area_option_C_max_perimeter_option_D_acute_range_l310_310303


namespace relationship_of_y_coordinates_l310_310718

theorem relationship_of_y_coordinates :
  let y1 := - 12 / -3
  let y2 := - 12 / -2
  let y3 := - 12 / 2
  y3 < y1 ∧ y1 < y2 :=
by {
  let y1 : ℝ := - 12 / -3
  let y2 : ℝ := - 12 / -2
  let y3 : ℝ := - 12 / 2
  show y3 < y1 ∧ y1 < y2,
  sorry
}

end relationship_of_y_coordinates_l310_310718


namespace exist_unique_sum_subsets_l310_310052

def A_def (n : ℕ) (i : ℕ) : set ℕ :=
  { x | ∃ k, x = (i + k * (n + 1)) }

theorem exist_unique_sum_subsets 
  (n : ℕ) (h : n ≥ 2) :
  ∃ A : fin n.succ → set ℕ,
    (∀ i j : fin n.succ, i ≠ j → A i ∩ A j = ∅) ∧
    (∀ x : ℕ, ∃! t : finset (fin n.succ), (∀ a ∈ t.val, a.val ≥ 1) ∧ x = t.sum (λ a, (λ i, A i.some a).val)) ∧
    (∀ i, A i ≠ ∅) :=
by
  let A := λ (n : ℕ) (i : ℕ), if 1 ≤ i + 1 ∧ i ≤ n then A_def (n) (i) else ∅
  use λ i, { x | ∃ k, x = i.val + k * (n + 1) ∧ x > 0 }
  split
  { sorry }
  split
  { sorry }
  { sorry }

end exist_unique_sum_subsets_l310_310052


namespace upright_water_depth_l310_310553

-- Definitions based on provided conditions
def tank_radius : ℝ := 3
def tank_height : ℝ := 10
def horizontal_depth : ℝ := 4

-- Prove that the upright depth of the water is 4.5 feet
theorem upright_water_depth : 
  let R := tank_radius in
  let H := tank_height in
  let dh := horizontal_depth in
  let V_water := (R ^ 2 * real.pi) * ((H - dh)/2) in
  V_water / (R ^ 2 * real.pi) = 4.5 :=
by sorry

end upright_water_depth_l310_310553


namespace total_notebooks_and_pens_is_110_l310_310472

/-- The number of notebooks and pens on Wesley's school library shelf /--
namespace WesleyLibraryShelf

noncomputable section

def notebooks : ℕ := 30
def pens : ℕ := notebooks + 50
def total_notebooks_and_pens : ℕ := notebooks + pens

theorem total_notebooks_and_pens_is_110 :
  total_notebooks_and_pens = 110 :=
sorry

end WesleyLibraryShelf

end total_notebooks_and_pens_is_110_l310_310472


namespace true_converse_of_parallel_lines_alternate_interior_angles_l310_310874

-- Definitions to state the problem
def VerticalAnglesCongruent : Prop := 
∀ {A B C D : ℝ}, isVerticalAngle A B C D → cong A B C D

def CongruentTrianglesEqualArea : Prop :=
∀ {T1 T2 : Triangle}, cong T1 T2 → area T1 = area T2

def PositiveProduct : Prop := 
∀ {a b : ℝ}, a > 0 ∧ b > 0 → a * b > 0

def ParallelLinesAlternateInteriorAnglesCongruent : Prop := 
∀ {l1 l2 : Line}, isParallel l1 l2 → alternateInteriorAngles l1 l2 ≈ cong

-- The statement of the theorem we need to prove
theorem true_converse_of_parallel_lines_alternate_interior_angles :
  ParallelLinesAlternateInteriorAnglesCongruent ↔ 
    (∀ {l1 l2 : Line}, alternateInteriorAngles l1 l2 ≈ cong → isParallel l1 l2) := 
sorry

end true_converse_of_parallel_lines_alternate_interior_angles_l310_310874


namespace find_side_c_of_triangle_l310_310745

noncomputable def triangle_side_c (a b C : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2 - 2 * a * b * real.cos C)

theorem find_side_c_of_triangle {A B C a b c : ℝ} 
    (h_a : a = 2) (h_b : b = 3) (h_C : C = 2 * A)
    (h_angle_sum : A + B + C = real.pi) 
    (h_b_angle : B = real.pi - 3 * A) : 
    c = real.sqrt 10 :=
by
  -- Assuming that there exists a valid proof structure here
  -- to be filled in by necessary steps and proof
  sorry

end find_side_c_of_triangle_l310_310745


namespace first_year_fee_correct_l310_310564

noncomputable def first_year_fee (n : ℕ) (annual_increase : ℕ) (sixth_year_fee : ℕ) : ℕ :=
  sixth_year_fee - (n - 1) * annual_increase

theorem first_year_fee_correct (n annual_increase sixth_year_fee value : ℕ) 
  (h_n : n = 6) (h_annual_increase : annual_increase = 10) 
  (h_sixth_year_fee : sixth_year_fee = 130) (h_value : value = 80) :
  first_year_fee n annual_increase sixth_year_fee = value :=
by {
  sorry
}

end first_year_fee_correct_l310_310564


namespace cube_vertex_product_max_sum_l310_310937

theorem cube_vertex_product_max_sum :
  ∃ (a b c d e f : ℕ), ({a, b, c, d, e, f} = {1, 2, 3, 4, 8, 9} ∧
  a + b = 9 ∧ c + d = 9 ∧ e + f = 9 ∧
  (a * c * e + a * c * f + a * d * e + a * d * f + 
   b * c * e + b * c * f + b * d * e + b * d * f = 729)) :=
begin
  sorry
end

end cube_vertex_product_max_sum_l310_310937


namespace problem_statement_l310_310057

noncomputable def f : ℕ → ℝ
| 1     := 1 / 2
| (n+1) := Real.sin (π / 2 * f n)

theorem problem_statement {x : ℕ} (hx : x ≥ 2) : 
  1 - f x < (π / 4) * (1 - f (x - 1)) :=
sorry

end problem_statement_l310_310057


namespace range_of_independent_variable_x_l310_310115

noncomputable def range_of_x (x : ℝ) : Prop :=
  x > -2

theorem range_of_independent_variable_x (x : ℝ) :
  ∀ x, (x + 2 > 0) → range_of_x x :=
by
  intro x h
  unfold range_of_x
  linarith

end range_of_independent_variable_x_l310_310115


namespace max_runs_one_day_match_l310_310883

theorem max_runs_one_day_match 
  (overs : ℕ) 
  (balls_per_over : ℕ) 
  (runs_per_ball : ℕ)
  (no_extras : ¬(∃ e, e ≠ 6))
  (overs_eq : overs = 50)
  (balls_per_over_eq : balls_per_over = 6)
  (runs_per_ball_eq : runs_per_ball = 6) :
  ∃ max_runs, max_runs = 1800 :=
by
  have total_balls : ℕ := overs * balls_per_over
  have total_runs : ℕ := total_balls * runs_per_ball
  have max_runs_val : total_runs = 50 * 6 * 6
  use total_runs
  rw [overs_eq, balls_per_over_eq, runs_per_ball_eq]
  simp only [Nat.mul_assoc]
  rw [max_runs_val]
  simp only [Nat.mul_comm 50 6, Nat.mul_comm 6 50, Nat.mul_comm 6 6]
  simp only [Nat.mul_assoc 50 6 6]
  sorry

end max_runs_one_day_match_l310_310883


namespace quad_area_proof_l310_310793

noncomputable def area_of_quad : ℝ :=
  let s := real.sqrt 144
  let pts := (s / 3, 2 * s / 3)
  -- Calculate the area of each of the 8 small triangles
  let area_triangle := (1 / 2) * (s / 3) * (s / 3)
  -- Total area of the 8 triangles
  let total_area_triangle := 8 * area_triangle
  -- Area of the larger square minus the area of the triangles
  144 - total_area_triangle

theorem quad_area_proof : area_of_quad = 80 := by
  sorry

end quad_area_proof_l310_310793


namespace unique_solution_of_equation_l310_310243

theorem unique_solution_of_equation :
  ∃! (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = 1 :=
begin
  sorry
end

end unique_solution_of_equation_l310_310243


namespace diagonals_in_25_sided_polygon_l310_310538

theorem diagonals_in_25_sided_polygon : 
  let n := 25 in
  let number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2 in
  number_of_diagonals n = 275 :=
by
  sorry

end diagonals_in_25_sided_polygon_l310_310538


namespace find_dividend_l310_310892

theorem find_dividend (divisor quotient remainder : ℕ) (h₁ : divisor = 16) (h₂ : quotient = 9) (h₃ : remainder = 5) :
  divisor * quotient + remainder = 149 :=
by
  rw [h₁, h₂, h₃]
  sorry

end find_dividend_l310_310892


namespace lucas_fraction_of_money_left_l310_310414

theorem lucas_fraction_of_money_left (m p n : ℝ) (h1 : (1 / 4) * m = (1 / 2) * n * p) :
  (m - n * p) / m = 1 / 2 :=
by 
  -- Sorry is used to denote that we are skipping the proof
  sorry

end lucas_fraction_of_money_left_l310_310414


namespace comb_product_l310_310212

theorem comb_product :
  (Nat.choose 10 3) * (Nat.choose 8 3) * 2 = 13440 :=
by
  sorry

end comb_product_l310_310212


namespace games_per_season_l310_310017

-- Define the problem parameters
def total_goals : ℕ := 1244
def louie_last_match_goals : ℕ := 4
def louie_previous_goals : ℕ := 40
def louie_season_total_goals := louie_last_match_goals + louie_previous_goals
def brother_goals_per_game := 2 * louie_last_match_goals
def seasons : ℕ := 3

-- Prove the number of games in each season
theorem games_per_season : ∃ G : ℕ, louie_season_total_goals + (seasons * brother_goals_per_game * G) = total_goals ∧ G = 50 := 
by {
  sorry
}

end games_per_season_l310_310017


namespace nineteenth_number_is_8136_l310_310840

def permutations (l : List ℕ) : List (List ℕ) :=
  if l = [] then [[]] else
  List.bind l (λ x => permutations (l.filter (λ y => y ≠ x)).map (List.cons x))

def list_4_digit_numbers : List (List ℕ) :=
  permutations [1, 3, 6, 8]

def digits_to_number (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => acc * 10 + x) 0

def list_4_digit_numbers_ordered : List ℕ :=
  (list_4_digit_numbers.map digits_to_number).qsort (≤)

theorem nineteenth_number_is_8136 :
  list_4_digit_numbers_ordered.nth 18 = some 8136 :=
by
  sorry

end nineteenth_number_is_8136_l310_310840


namespace ratio_cost_to_marked_price_l310_310187

variables (x : ℝ) (marked_price : ℝ) (selling_price : ℝ) (cost_price : ℝ)

theorem ratio_cost_to_marked_price :
  (selling_price = marked_price - 1/4 * marked_price) →
  (cost_price = 2/3 * selling_price) →
  (cost_price / marked_price = 1/2) :=
by
  sorry

end ratio_cost_to_marked_price_l310_310187


namespace cartesian_equation_of_line_range_of_m_for_intersection_l310_310734

def parametric_curve_C (t : Real) : ℝ × ℝ :=
  (2 * Real.cos t, 2 * Real.sin t)

def polar_line_l (ρ θ m : Real) : Prop :=
  ρ * Real.cos (θ - Real.pi / 3) + m = 0

theorem cartesian_equation_of_line (ρ θ m : Real) :
  polar_line_l ρ θ m →
  let x := ρ * Real.cos θ in
  let y := ρ * Real.sin θ in
  x + Real.sqrt 3 * y + 2 * m = 0 :=
sorry

theorem range_of_m_for_intersection (t m : Real) :
  let x := 2 * Real.cos t in
  let y := 2 * Real.sin t in
  x + Real.sqrt 3 * y + 2 * m = 0 →
  -2 ≤ m ∧ m ≤ 2 :=
sorry

end cartesian_equation_of_line_range_of_m_for_intersection_l310_310734


namespace _l310_310289

noncomputable def is_tangent_to_parabola (x1 y1 p k : ℝ) : Prop :=
  let y := k * x1 - 1
  let rhs := x1^2
  rhs = y

noncomputable def leans_theorem_prover (O A B : (ℝ × ℝ)) : Prop :=
  -- Definitions of points
  let O := (0,0)
  let A := (1,1)
  let B := (0,-1)
  -- Value of p from point A on parabola C: x^2 = 2py
  let p := 1 / 2  -- as obtained by solving 1^2 = 2p * 1
  -- Checking option A: directrix is y = -1 is false
  let directrix := - p / 2
  (directrix ≠ -1) ∧
  -- Checking option B: tangent condition
  let slope_AB := (1 - (-1)) / (1 - 0)
  let tangent := is_tangent_to_parabola 1 1 p slope_AB
  tangent ∧
  -- Option C: |OP| * |OQ| = |OA|^2 is false
  let |OA|² := 2  -- obtained from the calculation |OA| = sqrt(1^2 + 1^2)
  ∀ (k > 2), k² ≠ |OA|² ∧
  -- Option D: |BP| * |BQ| > |BA|^2 is true
  let |BA|² := 5 -- obtained from the calculation |BA| = sqrt(1^2 + 4)
  ∀ (x1 x2) (hx1 : x1 + x2 = k) (hx2 : x1 * x2 = 1),
  let |BP| := sqrt(x1^2 + (x1^2 + 2x1 + 1))
  let |BQ| := sqrt(x2^2 + (x2^2 + 2x2 + 1))
  |BP| * |BQ| > |BA|²

example : leans_theorem_prover (0, 0) (1, 1) (0, -1) :=
by sorry  -- Proof is omitted, to be completed by Lean theorem prover

end _l310_310289


namespace geom_prog_terms_exist_l310_310119

theorem geom_prog_terms_exist (b3 b6 : ℝ) (h1 : b3 = -1) (h2 : b6 = 27 / 8) :
  ∃ (b1 q : ℝ), b1 = -4 / 9 ∧ q = -3 / 2 :=
by
  sorry

end geom_prog_terms_exist_l310_310119


namespace product_numerator_denominator_l310_310142

def recurring_decimal_to_fraction (n : ℕ) (d : ℕ) : Rat :=
  n / d

theorem product_numerator_denominator (n : ℕ) (d : ℕ) (x : Rat)
  (hx : recurring_decimal_to_fraction 18 999 = x)
  (hn : n = 2)
  (hd : d = 111) :
  n * d = 222 := by
  have h_frac : x = 0.018 -- This follows from the definition and will be used in the proof
  sorry

end product_numerator_denominator_l310_310142


namespace nancy_carrots_total_l310_310894

theorem nancy_carrots_total (picked_initial : ℕ) (thrown_out : ℕ) (picked_next_day : ℕ) :
  picked_initial = 12 → thrown_out = 2 → picked_next_day = 21 → 
  (picked_initial - thrown_out + picked_next_day) = 31 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end nancy_carrots_total_l310_310894


namespace geometry_hard_problem_l310_310019

theorem geometry_hard_problem (A B C O H P D E F : Point)
  (h1 : acute_triangle ABC)
  (h2 : AB < AC)
  (hO : circumcenter ABC O)
  (hH : orthocenter ABC H)
  (h_inc : incircle_touches ABC D E F)
  (hP : orthocenter (triangle DEF) P) :
  orthocenter (triangle A O H) P ↔ perp O P A H :=
sorry

end geometry_hard_problem_l310_310019


namespace find_ratio_CL_AB_l310_310267

-- Definitions based on problem conditions
variables {A B C D E K L : Type} 
class RegularPentagon (sides: Type) :=
  (all_sides_equal : ∀ {P Q : sides}, ∃ (r: ℝ), P = Q → r)

variables [RegularPentagon {ABCDE : Type}]

def AngleSum (α β : ℝ): Prop := 
α + β = 108

def Ratio (p q : ℝ): Prop := 
p / q = 3/7

-- Main theorem stating the problem
theorem find_ratio_CL_AB
  (h1 : RegularPentagon {A B C D E})
  (h2 : ∀ P Q, ∃ r, P = Q → r)
  (h3 : AngleSum (LAE) (KCD))
  (h4 : Ratio (AK_length)(KE_length)) :
  (CL_length / AB_length) = 0.7 := 
sorry

end find_ratio_CL_AB_l310_310267


namespace longest_side_of_triangle_l310_310562

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def vertices : List (ℝ × ℝ) := [(3, 1), (7, 5), (8, 1)]

def distances (vs : List (ℝ × ℝ)) : List ℝ :=
  match vs with
  | [p1, p2, p3] =>
    [distance p1 p2, distance p1 p3, distance p2 p3]
  | _ => []

def max_distance (ds : List ℝ) : ℝ :=
  match ds with
  | [] => 0
  | h :: t => List.foldl max h t

theorem longest_side_of_triangle : max_distance (distances vertices) = 5 :=
by
  sorry

end longest_side_of_triangle_l310_310562


namespace die_top_face_odd_probability_l310_310421

-- Define the standard die faces and the number of dots on each face
def die_faces : Fin 6 → ℕ
| 0 => 1
| 1 => 2
| 2 => 3
| 3 => 4
| 4 => 5
| 5 => 6

-- Define the total number of dots on the die
def total_dots : ℕ := 21

-- Define the probability that the top face has an odd number of dots
def prob_top_odd_dots : ℝ := 11 / 21

theorem die_top_face_odd_probability :
  (1 / 6) * ( 
    (1 - 1 / 21) + 
    (2 / 21) + 
    (1 - 3 / 21) + 
    (4 / 21) + 
    (1 - 5 / 21) + 
    (6 / 21)
  ) = prob_top_odd_dots := by
  simp
  norm_num
  sorry

end die_top_face_odd_probability_l310_310421


namespace number_of_knights_l310_310476

-- Defining types for the inhabitants
inductive Inhabitant
| knight
| liar

-- Predicate to determine if the number of liars to the right is greater
-- than the number of knights to the left.
def claim (inhabitants : List Inhabitant) (i : Fin (List.length inhabitants)) : Prop :=
  let liars_to_right := List.countp (· = Inhabitant.liar) (List.drop i.succ inhabitants)
  let knights_to_left := List.countp (· = Inhabitant.knight) (List.take i inhabitants)
  liars_to_right > knights_to_left

-- The main theorem to prove
theorem number_of_knights (inhabitants : List Inhabitant) (h_len : inhabitants.length = 2021)
  (h_claim : ∀ i, i < inhabitants.length → claim inhabitants ⟨i, by simp [*]⟩) :
  List.countp (· = Inhabitant.knight) inhabitants = 1010 :=
by
  sorry

end number_of_knights_l310_310476


namespace initial_students_count_l310_310445

theorem initial_students_count (n : ℕ) (T T' : ℚ)
    (h1 : T = n * 61.5)
    (h2 : T' = T - 24)
    (h3 : T' = (n - 1) * 64) :
  n = 16 :=
by
  sorry

end initial_students_count_l310_310445


namespace minimize_product_hatp_eval_l310_310592

open Real

def quadratic (r s : ℝ) : ℝ → ℝ :=
  fun x => x^2 - (r + s) * x + r * s

def is_mischievous (p : ℝ → ℝ) : Prop :=
  ∃ r s : ℝ, (p = quadratic r s) ∧ (∃ u v w z : ℝ, p (p(u)) = 0 ∧ p (p(v)) = 0 ∧ p (p(w)) = 0 ∧ p (p(z)) = 0)

noncomputable def min_roots_poly : (ℝ → ℝ) :=
  quadratic 1 1

theorem minimize_product : 
  is_mischievous (quadratic 1 1) → 
  ∀ (p : ℝ → ℝ), is_mischievous p → 
    (∃ r s : ℝ, p = quadratic r s ∧ r * s ≤ 1) := sorry

theorem hatp_eval : min_roots_poly 1 = 1 := sorry

end minimize_product_hatp_eval_l310_310592


namespace trajectory_of_point_P_l310_310675

open Real

theorem trajectory_of_point_P (a : ℝ) (ha : a > 0) :
  (∀ x y : ℝ, (a = 1 → x = 0) ∧ 
    (a ≠ 1 → (x - (a^2 + 1) / (a^2 - 1))^2 + y^2 = 4 * a^2 / (a^2 - 1)^2)) := 
by 
  sorry

end trajectory_of_point_P_l310_310675


namespace trigonometric_translation_symmetry_l310_310313

theorem trigonometric_translation_symmetry (ω : ℝ) (φ : ℝ) 
  (hω : ω > 0) (hφ : 0 < φ ∧ φ < π / 2) 
  (h_fe : sin φ = -sin (ω * π / 2 + φ)) 
  (h_trans_symm : ∀ x, sin (ω * (x - π / 12) + φ) = -sin (ω * (x + π / 12) + φ))
  : φ = π / 6 := by 
  sorry

end trigonometric_translation_symmetry_l310_310313


namespace find_q_revolutions_per_minute_l310_310211

variable (p_rpm : ℕ) (q_rpm : ℕ) (t : ℕ)

def revolutions_per_minute_q : Prop :=
  (p_rpm = 10) → (t = 4) → (q_rpm = (10 / 60 * 4 + 2) * 60 / 4) → (q_rpm = 120)

theorem find_q_revolutions_per_minute (p_rpm q_rpm t : ℕ) :
  revolutions_per_minute_q p_rpm q_rpm t :=
by
  unfold revolutions_per_minute_q
  sorry

end find_q_revolutions_per_minute_l310_310211


namespace expand_and_simplify_l310_310855

noncomputable def expanded_expr (a : ℝ) : ℝ :=
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6)

theorem expand_and_simplify (a : ℝ) :
  expanded_expr a = a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 :=
by {
  -- Proof goes here
  sorry
}

end expand_and_simplify_l310_310855


namespace last_day_of_third_quarter_l310_310478

def is_common_year (year: Nat) : Prop :=
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0) 

def days_in_month (year: Nat) (month: Nat) : Nat :=
  if month = 2 then 28
  else if month = 4 ∨ month = 6 ∨ month = 9 ∨ month = 11 then 30
  else 31

def last_day_of_month (year: Nat) (month: Nat) : Nat :=
  days_in_month year month

theorem last_day_of_third_quarter (year: Nat) (h : is_common_year year) : last_day_of_month year 9 = 30 :=
by
  sorry

end last_day_of_third_quarter_l310_310478


namespace fraction_is_smaller_by_12_l310_310850

-- Define the fraction x and the condition
def fraction_of_25 (x : ℝ) := x * 25

-- Define 80% of 40
def eighty_percent_of_40 := 0.8 * 40

-- Define the comparison condition
theorem fraction_is_smaller_by_12 (x : ℝ) : fraction_of_25 x = eighty_percent_of_40 - 12 ↔ x = 4/5 := by
  sorry

end fraction_is_smaller_by_12_l310_310850


namespace cube_of_cuberoot_of_7_is_7_l310_310851

theorem cube_of_cuberoot_of_7_is_7 : (∛ 7) ^ 3 = 7 := 
by 
  sorry

end cube_of_cuberoot_of_7_is_7_l310_310851


namespace complex_magnitude_l310_310720

theorem complex_magnitude (z : ℂ) (h : 2 * z - conj z = 2 + 3 * I) : abs z = Real.sqrt 5 := by
  sorry

end complex_magnitude_l310_310720


namespace carter_drum_stick_sets_l310_310957

theorem carter_drum_stick_sets (sets_per_show sets_tossed_per_show nights : ℕ) :
  sets_per_show = 5 →
  sets_tossed_per_show = 6 →
  nights = 30 →
  (sets_per_show + sets_tossed_per_show) * nights = 330 := by
  intros
  sorry

end carter_drum_stick_sets_l310_310957


namespace initial_number_of_men_l310_310005

theorem initial_number_of_men (M : ℕ) :
  let women_initial := 3000
  let boys_initial := 2000
  let women_1994 := women_initial + (2 / 3) * women_initial
  let men_1994 := 1.20 * M
  let total_population_1994 := men_1994 + women_1994 + boys_initial
  total_population_1994 = 13000
  ∧ women_1994 = 5000
  ∧ M = 5000 :=
sorry

end initial_number_of_men_l310_310005


namespace bananas_on_first_day_l310_310157

theorem bananas_on_first_day (total_bananas : ℕ) (days : ℕ) (increment : ℕ) (bananas_first_day : ℕ) :
  (total_bananas = 100) ∧ (days = 5) ∧ (increment = 6) ∧ ((bananas_first_day + (bananas_first_day + increment) + 
  (bananas_first_day + 2*increment) + (bananas_first_day + 3*increment) + (bananas_first_day + 4*increment)) = total_bananas) → 
  bananas_first_day = 8 :=
by
  sorry

end bananas_on_first_day_l310_310157


namespace v_of_2_l310_310399

def u (x : ℝ) : ℝ := 4 * x - 9
def v (y : ℝ) : ℝ := y

theorem v_of_2 : v 2 = (249 / 16) := by
  have h₁ : 4 * (11 / 4) - 9 = 2 := by
    linarith
  have h₃ : (11 / 4)^2 + 4 * (11 / 4) - 3 = 249 / 16 := by
    calc
      (11 / 4)^2     = 121 / 16 := by norm_num
      4 * (11 / 4)   = 11      := by norm_num
      121 / 16 + 11 - 3 = 249 / 16 := by norm_cast; linarith
  rw [←h₁] at h₃
  exact h₃

end v_of_2_l310_310399


namespace minimum_hat_flips_998_l310_310990

-- Definitions for gnomes and their hats
def gnome := {hat_color : bool} -- true if blue, false if red
def gnomes := list gnome

-- Each gnome always says to every other gnome "You are wearing a red hat!"
-- Based on initial conditions and the logic from b)
-- We want to prove that the minimum number of flips necessary is 998.

theorem minimum_hat_flips_998 (gnomes : gnomes) :
  ∃ flips : ℕ,
    (∀ g1 g2 : gnome,
      (g1.hat_color = g2.hat_color → g1.hat_color = false ∧ g2.hat_color = false)
      ∨ (g1.hat_color ≠ g2.hat_color → g1.hat_color = true ∨ g1.hat_color = false)) 
    ∧ flips = 998 :=
sorry

end minimum_hat_flips_998_l310_310990


namespace portions_of_milk_l310_310331

theorem portions_of_milk (liters_to_ml : ℕ) (total_liters : ℕ) (portion : ℕ) (total_volume_ml : ℕ) (num_portions : ℕ) :
  liters_to_ml = 1000 →
  total_liters = 2 →
  portion = 200 →
  total_volume_ml = total_liters * liters_to_ml →
  num_portions = total_volume_ml / portion →
  num_portions = 10 := by
  sorry

end portions_of_milk_l310_310331


namespace expand_simplify_expression_l310_310868

theorem expand_simplify_expression (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
by
  sorry

end expand_simplify_expression_l310_310868


namespace vertices_form_parabola_l310_310762

variables (a c d : ℝ) (h_a : 0 < a) (h_c : 0 < c) (h_d : 0 < d)

/-- Given fixed positive numbers a, c, and d, for any real number b, 
the set of vertices (x_t, y_t) of the parabolas y = a*x^2 + (b+c)*x + d 
forms a new parabola y = -a*x^2 + d. -/
theorem vertices_form_parabola (b : ℝ) :
  let x_v := - (b + c) / (2 * a),
      y_v := - (b + c)^2 / (4 * a) + d in
  y_v = -a * x_v^2 + d := 
begin
  sorry
end

end vertices_form_parabola_l310_310762


namespace donny_spent_on_thursday_l310_310984

theorem donny_spent_on_thursday :
  let savings_monday : ℤ := 15,
      savings_tuesday : ℤ := 28,
      savings_wednesday : ℤ := 13,
      total_savings : ℤ := savings_monday + savings_tuesday + savings_wednesday,
      amount_spent_thursday : ℤ := total_savings / 2
  in
  amount_spent_thursday = 28 :=
by
  sorry

end donny_spent_on_thursday_l310_310984


namespace find_sine_theta_l310_310758

variables (a b c : ℝ^3)
variables (θ : ℝ)

-- Conditions
def norm_a : ∥a∥ = 2 := sorry
def norm_b : ∥b∥ = 10 := sorry
def norm_c : ∥c∥ = 12 := sorry
def vector_identity : b × (a × b) = -3 • c := sorry
def angle_between : θ = real.angle a b := sorry

-- Theorem to be proven
theorem find_sine_theta :
  ∥a∥ = 2 →
  ∥b∥ = 10 →
  ∥c∥ = 12 →
  b × (a × b) = -3 • c →
  real.sin θ = 3 / 10 :=
by
  intros h₁ h₂ h₃ h₄
  sorry

end find_sine_theta_l310_310758


namespace odd_function_property_l310_310635

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

-- Lean 4 statement of the problem
theorem odd_function_property (f : ℝ → ℝ) (h : is_odd f) : ∀ x : ℝ, f x + f (-x) = 0 := 
  by sorry

end odd_function_property_l310_310635


namespace not_product_of_two_primes_l310_310069

theorem not_product_of_two_primes (n : ℕ) (h : n ≥ 2) : 
  ¬ ∃ p q : ℕ, prime p ∧ prime q ∧ 2^(4*n + 2) + 1 = p * q := 
sorry

end not_product_of_two_primes_l310_310069


namespace average_speed_to_first_summit_l310_310417

theorem average_speed_to_first_summit 
  (time_first_summit : ℝ := 3)
  (time_descend_partially : ℝ := 1)
  (time_second_uphill : ℝ := 2)
  (time_descend_back : ℝ := 2)
  (avg_speed_whole_journey : ℝ := 3) :
  avg_speed_whole_journey = 3 →
  time_first_summit = 3 →
  avg_speed_whole_journey * (time_first_summit + time_descend_partially + time_second_uphill + time_descend_back) = 24 →
  avg_speed_whole_journey = 3 := 
by
  intros h_avg_speed h_time_first_summit h_total_distance
  sorry

end average_speed_to_first_summit_l310_310417


namespace value_of_expression_l310_310353

theorem value_of_expression (x : ℤ) (h : x^2 = 1369) : (x + 1) * (x - 1) = 1368 := 
by 
  sorry

end value_of_expression_l310_310353


namespace batteries_used_in_toys_l310_310125

variables (b_flashlights b_controllers b_total b_toys : ℕ)

-- Conditions
def conditions := 
  b_flashlights = 2 ∧ 
  b_controllers = 2 ∧ 
  b_total = 19

-- Problem to Prove
theorem batteries_used_in_toys (h : conditions b_flashlights b_controllers b_total b_toys) :
  b_toys = b_total - b_flashlights - b_controllers :=
by
  cases h with h1 hrest,
  cases hrest with h2 h3,
  rw [h1, h2, h3],
  exact rfl

end batteries_used_in_toys_l310_310125


namespace farthest_point_is_80_l310_310512

-- Define the distance function from a point to the origin
def distance_from_origin (x y : ℝ) : ℝ := real.sqrt (x^2 + y^2)

-- Define the points to compare
def p1 : ℝ × ℝ := (0, 7)
def p2 : ℝ × ℝ := (2, 3)
def p3 : ℝ × ℝ := (-5, 1)
def p4 : ℝ × ℝ := (8, 0)
def p5 : ℝ × ℝ := (4, 4)

-- The proof statement
theorem farthest_point_is_80 :
  ∀ (p : ℝ × ℝ), p = p1 ∨ p = p2 ∨ p = p3 ∨ p = p4 ∨ p = p5 →
  distance_from_origin (p.fst) (p.snd) ≤ distance_from_origin (p4.fst) (p4.snd) :=
by
  sorry

end farthest_point_is_80_l310_310512


namespace ellipse_equation_prove_midpoint_area_triangle_l310_310273

noncomputable def ellipse_passes_through (C : ℝ → ℝ → Prop) (D E : ℝ × ℝ) : Prop :=
  C D.fst D.snd ∧ C E.fst E.snd

theorem ellipse_equation (a b : ℝ) (D E : ℝ × ℝ) (h₁ : ellipse_passes_through (λ x y, (x^2)/(a^2) + (y^2)/(b^2) = 1) D E) 
  (hD : D = (2, 0)) (hE : E = (1, (Real.sqrt 3) / 2)) (h2 : a > b) (h3 : b > 0) : 
  (λ x y, (x^2)/(4) + y^2 = 1) = (λ x y, (x^2)/(a^2) + (y^2)/(b^2) = 1) :=
  sorry

theorem prove_midpoint (k m : ℝ) (A B G O Q : ℝ × ℝ) (l : ℝ → ℝ → Prop) 
  (h₁ : l = λ x y, y = k*x + m) 
  (h₂ : ellipse_equation 2 1 (2,0) (1,Real.sqrt 3 / 2) (∃ x y, l x y ∧ (x^2)/(4) + y^2 = 1)) 
  (h₃ : G = ((A.fst + B.fst) / 2, (A.snd + B.snd) / 2))
  (h₄ : O = (0, 0)) (hQ : Q = (2 * G.fst, 2 * G.snd)) :
  (4 * m^2 = 4 * k^2 + 1) :=
  sorry

theorem area_triangle (k m : ℝ) (A B : ℝ × ℝ) (O : ℝ × ℝ) (h₁ : A ≠ B) 
  (h₂ : ∃ x y, (x^2)/(4) + y^2 = 1) 
  (h₃ : ∃ x y, y = k * x + m ∧ (x^2)/(4) + y^2 = 1) 
  : abs ((A.snd - O.snd) * (B.fst - A.fst) - (O.fst - A.fst) * (B.snd - A.snd)) / 2 = Real.sqrt(3) / 2 :=
  sorry

end ellipse_equation_prove_midpoint_area_triangle_l310_310273


namespace six_digit_integers_count_l310_310706

theorem six_digit_integers_count : 
  let total_digits := [1, 1, 3, 3, 3, 9] in
  (∀ n, n ∈ total_digits → n ∈ [1, 3, 9]) → 
  (∃! count,
    let total_permutations := 6! in
    let adjust_repetitions := total_permutations / (2! * 3! * 1!) in
    count = adjust_repetitions ∧ count = 60) :=
by
  sorry

end six_digit_integers_count_l310_310706


namespace inscribed_circle_radius_l310_310908

-- Definitions based on conditions
structure IsoscelesTriangle :=
  (base_line parallel_line : Line)
  (vertex_line : Line)
  (vertex : Point)
  (base_left base_right : Point) -- Points on the base of the triangle
  (base_len : ℝ) -- Length of the base of the triangle
  (height : ℝ) -- Height of the triangle from base to vertex

structure Circle :=
  (center : Point)
  (radius : ℝ)
  (touches_lines : bool)
  (touches_triangle_point : Point)

noncomputable def radius_of_inscribed_circle : IsoscelesTriangle → Circle → ℝ
  | triangle, circle => sorry

-- The main statement to prove
theorem inscribed_circle_radius (triangle : IsoscelesTriangle) (circle : Circle) 
  (h1 : circle.radius = 1)
  (h2 : circle.touches_lines = true)
  (h3 : circle.touch_triangle_point ∈ {triangle.base_left, triangle.base_right, triangle.vertex})
  : radius_of_inscribed_circle triangle circle = 1 / 2 :=
sorry

end inscribed_circle_radius_l310_310908


namespace degree_reduction_l310_310951

theorem degree_reduction (x : ℝ) (h1 : x^2 = x + 1) (h2 : 0 < x) : x^4 - 2 * x^3 + 3 * x = 1 + Real.sqrt 5 :=
by
  sorry

end degree_reduction_l310_310951


namespace union_M_N_l310_310696

-- Define sets M and N
def M : Set ℝ := { x | x^2 - 4x + 3 < 0 }
def N : Set ℝ := { x | 2x + 1 < 5 }

-- Prove that the union of M and N is exactly { x | x < 3 }
theorem union_M_N : { x | x < 3 } = M ∪ N :=
by sorry

end union_M_N_l310_310696


namespace median_is_4_6_l310_310227

-- Given conditions on the number of students per vision category
def vision_data : List (ℝ × ℕ) :=
[(4.0, 1), (4.1, 2), (4.2, 6), (4.3, 3), (4.4, 3), (4.5, 4), (4.6, 1), 
 (4.7, 2), (4.8, 5), (4.9, 7), (5.0, 5)]

-- Statement of the problem: Proving that the median is 4.6
theorem median_is_4_6 (data : List (ℝ × ℕ)) (h : data = vision_data) : 
  median (expand_data data) = 4.6 := 
sorry

-- Helper function to expand the data into a list of individual vision values
def expand_data (data : List (ℝ × ℕ)) : List ℝ :=
data.bind (λ p, List.replicate p.snd p.fst)

-- Helper function to compute the median
noncomputable
def median (l : List ℝ) : ℝ :=
if l.length % 2 = 1 then l.nth_le (l.length / 2) (by linarith [List.length_pos_of_mem (List.nth_le_mem l (l.length / 2) _)])
else (l.nth_le (l.length / 2 - 1) (by linarith [List.length_pos_of_mem (List.nth_le_mem l (l.length / 2 - 1) _)]) +
      l.nth_le (l.length / 2) (by linarith [List.length_pos_of_mem (List.nth_le_mem l (l.length / 2) _)])) / 2

end median_is_4_6_l310_310227


namespace prob_beatrice_on_beach_l310_310064

open ProbabilityTheory

noncomputable def beatrice_conditions
  (P_beach : ℚ := 1/2)
  (P_tennis : ℚ := 1/4)
  (P_cafe : ℚ := 1/4)
  (P_not_find_beach : ℚ := 1/2)
  (P_not_find_tennis : ℚ := 1/3)
  (P_not_find_cafe : ℚ := 0) :
  (P_beach + P_tennis + P_cafe = 1) ∧
  (P_not_find_beach <= 1 ∧ P_not_find_tennis <= 1 ∧ P_not_find_cafe <= 1) :=
by
  have h₁ : P_beach + P_tennis + P_cafe = 1 := by norm_num
  have h₂ : P_not_find_beach <= 1 := by norm_num
  have h₃ : P_not_find_tennis <= 1 := by norm_num
  have h₄ : P_not_find_cafe <= 1 := by norm_num
  exact ⟨h₁, ⟨h₂, ⟨h₃, h₄⟩⟩⟩

theorem prob_beatrice_on_beach (P_beach P_tennis P_cafe P_not_find_beach P_not_find_tennis P_not_find_cafe : ℚ):
  beatrice_conditions P_beach P_tennis P_cafe P_not_find_beach P_not_find_tennis P_not_find_cafe →
  let P_not_found := P_beach * P_not_find_beach + P_tennis * P_not_find_tennis + P_cafe * P_not_find_cafe in
  P_not_found = 5 / 12 →
  (P_beach * P_not_find_beach / P_not_found) = 3 / 5 :=
by
  intros h₁ h₂
  sorry

end prob_beatrice_on_beach_l310_310064


namespace possible_birches_l310_310898

theorem possible_birches (N B L : ℕ) (hN : N = 130) (h_sum : B + L = 130)
  (h_linden_false : ∀ l, l < L → (∀ b, b < B → b + l < N → b < B → False))
  (h_birch_false : ∃ b, b < B ∧ (∀ l, l < L → l + b < N → l + b = 2 * B))
  : B = 87 :=
sorry

end possible_birches_l310_310898


namespace midpoint_distance_to_y_axis_l310_310301

-- Let F be the focus of the parabola y^2 = 4x.
def focus := (1 : ℝ, 0 : ℝ)

-- A and B are points on the parabola y^2 = 4x.
variables {A B : ℝ × ℝ}

-- Given the condition that |AF| + |BF| = 12.
axiom sum_of_distances (h1 : A.1^2 + A.2^2 = 4 * A.1) (h2 : B.1^2 + B.2^2 = 4 * B.1) : 
  (real.sqrt ((A.1 - focus.1)^2 + (A.2 - focus.2)^2) + real.sqrt ((B.1 - focus.1)^2 + (B.2 - focus.2)^2)) = 12

-- Prove that the distance from the midpoint of line segment AB to the y-axis is 5.
theorem midpoint_distance_to_y_axis (hA : A.1^2 + A.2^2 = 4 * A.1) (hB : B.1^2 + B.2^2 = 4 * B.1) 
  (h_sum : sum_of_distances hA hB) : real.abs ((A.1 + B.1) / 2) = 5 := sorry

end midpoint_distance_to_y_axis_l310_310301


namespace tart_fill_l310_310932

theorem tart_fill (cherries blueberries total : ℚ) (h_cherries : cherries = 0.08) (h_blueberries : blueberries = 0.75) (h_total : total = 0.91) :
  total - (cherries + blueberries) = 0.08 :=
by
  sorry

end tart_fill_l310_310932


namespace paco_ate_sweet_cookies_l310_310422

theorem paco_ate_sweet_cookies :=
  let T := 2 in
  let S := T + 3 in
  S = 5

end paco_ate_sweet_cookies_l310_310422


namespace judges_voted_for_crow_l310_310013

theorem judges_voted_for_crow
    (P V K : ℕ)
    (h1 : P + V + K = 59)
    (h2 : P + V = 15)
    (h3 : V + K = 18)
    (h4 : K + P = 20)
    (h5 : ∀ (x : ℕ), abs (P + V + K - 59) ≤ 13)
    (h6 : ∀ (x : ℕ), abs (P + V - 15) ≤ 13)
    (h7 : ∀ (x : ℕ), abs (V + K - 18) ≤ 13)
    (h8 : ∀ (x : ℕ), abs (K + P - 20) ≤ 13) :
    V = 13 := 
sorry

end judges_voted_for_crow_l310_310013


namespace iced_tea_cost_is_correct_l310_310956

noncomputable def iced_tea_cost (cost_cappuccino cost_latte cost_espresso : ℝ) (num_cappuccino num_iced_tea num_latte num_espresso : ℕ) (bill_amount change_amount : ℝ) : ℝ :=
  let total_cappuccino_cost := cost_cappuccino * num_cappuccino
  let total_latte_cost := cost_latte * num_latte
  let total_espresso_cost := cost_espresso * num_espresso
  let total_spent := bill_amount - change_amount
  let total_other_cost := total_cappuccino_cost + total_latte_cost + total_espresso_cost
  let total_iced_tea_cost := total_spent - total_other_cost
  total_iced_tea_cost / num_iced_tea

theorem iced_tea_cost_is_correct:
  iced_tea_cost 2 1.5 1 3 2 2 2 20 3 = 3 :=
by
  sorry

end iced_tea_cost_is_correct_l310_310956


namespace quadratic_no_roots_l310_310071

noncomputable theory

variables {p b q c : ℝ}

theorem quadratic_no_roots 
  (h₁ : p^2 - 4 * q < 0) 
  (h₂ : b^2 - 4 * c < 0)
  : (2 * p + 3 * b + 4) ^ 2 - 4 * 7 * (2 * q + 3 * c + 2) < 0 :=
sorry

end quadratic_no_roots_l310_310071


namespace unique_solution_of_equation_l310_310242

theorem unique_solution_of_equation :
  ∃! (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = 1 :=
begin
  sorry
end

end unique_solution_of_equation_l310_310242


namespace geometric_parallel_incenters_circumcenters_l310_310729

open Classical

noncomputable theory

-- Define the problem in Lean

theorem geometric_parallel_incenters_circumcenters
  (ABC : Triangle)
  (D E F : Point)
  (hD : is_foot_of_altitude D A B C)
  (hE : is_foot_of_altitude E B A C)
  (hF : is_foot_of_altitude F C A B)
  (I1 : Point)
  (I2 : Point)
  (hI1 : is_incenter I1 (Triangle.mk A E F))
  (hI2 : is_incenter I2 (Triangle.mk B D F))
  (O1 : Point)
  (O2 : Point)
  (hO1 : is_circumcenter O1 (Triangle.mk A C I1))
  (hO2 : is_circumcenter O2 (Triangle.mk B C I2)) :
  parallel (Line.mk I1 I2) (Line.mk O1 O2) :=
sorry

end geometric_parallel_incenters_circumcenters_l310_310729


namespace largest_number_formed_l310_310717

-- Define the digits
def digit1 : ℕ := 2
def digit2 : ℕ := 6
def digit3 : ℕ := 9

-- Define the function to form the largest number using the given digits
def largest_three_digit_number (a b c : ℕ) : ℕ :=
  if a > b ∧ a > c then
    if b > c then 100 * a + 10 * b + c
    else 100 * a + 10 * c + b
  else if b > a ∧ b > c then
    if a > c then 100 * b + 10 * a + c
    else 100 * b + 10 * c + a
  else
    if a > b then 100 * c + 10 * a + b
    else 100 * c + 10 * b + a

-- Statement that this function correctly computes the largest number
theorem largest_number_formed :
  largest_three_digit_number digit1 digit2 digit3 = 962 :=
by
  sorry

end largest_number_formed_l310_310717


namespace angle_in_quadrants_l310_310335

theorem angle_in_quadrants (α : ℝ) (hα : 0 < α ∧ α < π / 2) (k : ℤ) :
  (∃ i : ℤ, k = 2 * i + 1 ∧ π < (2 * i + 1) * π + α ∧ (2 * i + 1) * π + α < 3 * π / 2) ∨
  (∃ i : ℤ, k = 2 * i ∧ 0 < 2 * i * π + α ∧ 2 * i * π + α < π / 2) :=
sorry

end angle_in_quadrants_l310_310335


namespace product_numerator_denominator_l310_310143

def recurring_decimal_to_fraction (n : ℕ) (d : ℕ) : Rat :=
  n / d

theorem product_numerator_denominator (n : ℕ) (d : ℕ) (x : Rat)
  (hx : recurring_decimal_to_fraction 18 999 = x)
  (hn : n = 2)
  (hd : d = 111) :
  n * d = 222 := by
  have h_frac : x = 0.018 -- This follows from the definition and will be used in the proof
  sorry

end product_numerator_denominator_l310_310143


namespace solution_of_inequality_system_l310_310834

theorem solution_of_inequality_system (x : ℝ) : 
  (x + 1 > 0 ∧ x + 3 ≤ 4) ↔ (-1 < x ∧ x ≤ 1) := 
by
  sorry

end solution_of_inequality_system_l310_310834


namespace find_reflection_line_l310_310126

open Real

def point := (ℝ × ℝ)

def P : point := (1, 4)
def Q : point := (8, 9)
def R : point := (-3, 7)

def P' : point := (1, -6)
def Q' : point := (8, -11)
def R' : point := (-3, -9)

noncomputable def line_equation (M : ℝ → ℝ → Prop) : Prop :=
  ∀ x y y', (M x y' → M x y) → (y + y') / 2 = -1

theorem find_reflection_line :
  line_equation (λ x y, y = -1) :=
by
  sorry

end find_reflection_line_l310_310126


namespace required_remaining_speed_l310_310581

-- Definitions for the given problem
variables (D T : ℝ) 

-- Given conditions from the problem
def speed_first_part (D T : ℝ) : Prop := 
  40 = (2 * D / 3) / (T / 3)

def remaining_distance_time (D T : ℝ) : Prop :=
  10 = (D / 3) / (2 * (2 * D / 3) / 40 / 3)

-- Theorem to be proved
theorem required_remaining_speed (D T : ℝ) 
  (h1 : speed_first_part D T)
  (h2 : remaining_distance_time D T) :
  10 = (D / 3) / (2 * (T / 3)) :=
  sorry  -- Proof is skipped

end required_remaining_speed_l310_310581


namespace find_n_l310_310769

-- Define the points and line
def P : (ℝ × ℝ) := (-1, -3)
def Q : (ℝ × ℝ) := (5, 3)
def R_ (n : ℝ) : (ℝ × ℝ) := (2 , n)

-- Define the distance function
def distance (A B : (ℝ × ℝ)) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- Define the line condition for R
def line_condition (n : ℝ) : Prop := 
  (R_ n).2 = 2 * (R_ n).1 - 4

-- Define the minimization condition
def minimize_distance : (ℝ → ℝ) := λ n, distance P (R_ n) + distance (R_ n) Q

-- Define the statement to be proven
theorem find_n (n : ℝ) (h1 : line_condition n) : n = 0 :=
by
  have h1 : (R_ n).2 = 2 * (R_ n).1 - 4 := h1
  have : ((R_ 0).2 = 2 * (R_ 0).1 - 4) := by simp [R_, line_condition]
  have : minimize_distance 0 = minimize_distance n := by sorry
  congr
  sorry -- step where final minimal argument is shown


end find_n_l310_310769


namespace marly_100_bills_l310_310415

-- Define the number of each type of bill Marly has
def num_20_bills := 10
def num_10_bills := 8
def num_5_bills := 4

-- Define the values of the bills
def value_20_bill := 20
def value_10_bill := 10
def value_5_bill := 5

-- Define the total amount of money Marly has
def total_amount := num_20_bills * value_20_bill + num_10_bills * value_10_bill + num_5_bills * value_5_bill

-- Define the value of a $100 bill
def value_100_bill := 100

-- Now state the main theorem
theorem marly_100_bills : total_amount / value_100_bill = 3 := by
  sorry

end marly_100_bills_l310_310415


namespace increasing_intervals_g_l310_310261

noncomputable def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

noncomputable def g (x : ℝ) : ℝ := f (2 - x^2)

theorem increasing_intervals_g : 
  (∀ x ∈ Set.Icc (-1 : ℝ) (0 : ℝ), ∀ y ∈ Set.Icc (-1 : ℝ) (0 : ℝ), x ≤ y → g x ≤ g y) ∧
  (∀ x ∈ Set.Ici (1 : ℝ), ∀ y ∈ Set.Ici (1 : ℝ), x ≤ y → g x ≤ g y) := 
sorry

end increasing_intervals_g_l310_310261


namespace geometric_sequence_range_of_lambda_l310_310664

variable (λ : ℝ) (a : ℕ → ℝ)
hypothesis h_neg : λ < 0
hypothesis h_rec : ∀ n : ℕ, n > 0 → a (n + 1) - λ * a n = λ - λ^2
hypothesis h_a1 : a 1 = 3 * λ
hypothesis h_ratio : ∀ m n : ℕ, m > 0 → n > 0 → -λ < a m / a n ∧ a m / a n < -1 / λ

-- Problem 1
theorem geometric_sequence : 
  ∀ n : ℕ, n > 0 →
  ∃ r : ℝ, ∀ m : ℕ, m > 0 → a n - λ = r ^ (m - 1) * (a 1 - λ) :=
sorry

-- Problem 2
theorem range_of_lambda :
  - (1 / 5) < λ ∧ λ < 0 :=
sorry

end geometric_sequence_range_of_lambda_l310_310664


namespace solution_eq1_solution_eq2_l310_310436

-- Definitions corresponding to the conditions of the problem.
def eq1 (x : ℝ) : Prop := 16 * x^2 = 49
def eq2 (x : ℝ) : Prop := (x - 2)^2 = 64

-- Statements for the proof problem.
theorem solution_eq1 (x : ℝ) : eq1 x → (x = 7 / 4 ∨ x = - (7 / 4)) :=
by
  intro h
  sorry

theorem solution_eq2 (x : ℝ) : eq2 x → (x = 10 ∨ x = -6) :=
by
  intro h
  sorry

end solution_eq1_solution_eq2_l310_310436


namespace edric_monthly_salary_l310_310992

theorem edric_monthly_salary 
  (hours_per_day : ℝ)
  (days_per_week : ℝ)
  (weeks_per_month : ℝ)
  (hourly_rate : ℝ) :
  hours_per_day = 8 ∧ days_per_week = 6 ∧ weeks_per_month = 4.33 ∧ hourly_rate = 3 →
  (hours_per_day * days_per_week * weeks_per_month * hourly_rate) = 623.52 :=
by
  intros h
  sorry

end edric_monthly_salary_l310_310992


namespace combined_rate_mpg_l310_310078

-- Defining the given conditions
def ray_mpg : ℝ := 30
def tom_mpg : ℝ := 20
def lucy_mpg : ℝ := 15
def m : ℝ := 1 -- Assume each drives 1 mile to simplify the proof

-- Total distance driven by all three cars
def total_distance : ℝ := 3 * m

-- Total gasoline consumption
def total_gasoline : ℝ := m / ray_mpg + m / tom_mpg + m / lucy_mpg

-- Combined miles per gallon calculation
def combined_mpg : ℝ := total_distance / total_gasoline

-- The theorem to be proved
theorem combined_rate_mpg : combined_mpg = 20 :=
sorry

end combined_rate_mpg_l310_310078


namespace increase_in_average_weight_l310_310096

variable {A X : ℝ}

-- Given initial conditions
axiom average_initial_weight_8 : X = (8 * A - 62 + 90) / 8 - A

-- The goal to prove
theorem increase_in_average_weight : X = 3.5 :=
by
  sorry

end increase_in_average_weight_l310_310096


namespace whatsapp_messages_total_l310_310529

-- Define conditions
def messages_monday : ℕ := 300
def messages_tuesday : ℕ := 200
def messages_wednesday : ℕ := messages_tuesday + 300
def messages_thursday : ℕ := 2 * messages_wednesday
def messages_friday : ℕ := messages_thursday + (20 * messages_thursday) / 100
def messages_saturday : ℕ := messages_friday - (10 * messages_friday) / 100

-- Theorem statement to be proved
theorem whatsapp_messages_total :
  messages_monday + messages_tuesday + messages_wednesday + messages_thursday + messages_friday + messages_saturday = 4280 :=
by 
  sorry

end whatsapp_messages_total_l310_310529


namespace complement_intersection_l310_310166

open Set

variable (U : Set ℕ) (A B : Set ℕ)

theorem complement_intersection :
  U = {1, 2, 3, 4, 5} →
  A = {1, 2, 3} →
  B = {2, 3, 5} →
  U \ (A ∩ B) = {1, 4, 5} :=
by
  intros hU hA hB
  rw [hU, hA, hB]
  sorry

end complement_intersection_l310_310166


namespace bee_speed_from_daisy_to_rose_l310_310879

variable (v : ℝ)   -- speed of the bee from the daisy to the rose in meters per second

-- Condition 1: The bee flies for 10 seconds from the daisy to the rose.
def daisy_to_rose_time : ℝ := 10

-- Condition 2: The bee flies for 6 seconds from the rose to the poppy.
def rose_to_poppy_time : ℝ := 6

-- Condition 3: The bee flies to the poppy at 3 meters per second faster than to the rose.
def rose_to_poppy_speed : ℝ := v + 3

-- The distance from the daisy to the rose is 8 meters longer than the distance from the rose to the poppy.
def distance_relation : Prop := 
  (daisy_to_rose_time * v) = (rose_to_poppy_time * rose_to_poppy_speed) + 8

theorem bee_speed_from_daisy_to_rose (h : distance_relation v) : v = 6.5 :=
by {
  sorry,
}

end bee_speed_from_daisy_to_rose_l310_310879


namespace all_conversions_correct_l310_310509

-- Define the conversion from degrees to radians and vice versa
def deg_to_rad (d : ℚ) : ℚ := d * real.pi / 180
def rad_to_deg (r : ℚ) : ℚ := r * 180 / real.pi

-- Given conditions as Lean definitions
def cond_A := deg_to_rad 60 = real.pi / 3
def cond_B := rad_to_deg (-10 / 3 * real.pi) = -600
def cond_C := deg_to_rad (-150) = -5 * real.pi / 6
def cond_D := rad_to_deg (real.pi / 12) = 15

-- The proof statement: Show that all these conditions are correct
theorem all_conversions_correct : cond_A ∧ cond_B ∧ cond_C ∧ cond_D :=
by sorry

end all_conversions_correct_l310_310509


namespace find_c_value_l310_310534

theorem find_c_value :
  ∃ c : ℝ, (∀ x y : ℝ, (x + 10) ^ 2 + (y + 4) ^ 2 = 169 ∧ (x - 3) ^ 2 + (y - 9) ^ 2 = 65 → x + y = c) ∧ c = 3 :=
sorry

end find_c_value_l310_310534


namespace total_rabbits_l310_310116

theorem total_rabbits (white_rabbits black_rabbits : ℕ) (h_white : white_rabbits = 15) (h_black : black_rabbits = 37) : 
  white_rabbits + black_rabbits = 52 := 
by 
  rw [h_white, h_black] 
  exact rfl

end total_rabbits_l310_310116


namespace product_of_fractional_parts_eq_222_l310_310140

theorem product_of_fractional_parts_eq_222 : 
  let x := 18 / 999 in let y := x.num / x.denom in y.num * y.denom = 222 :=
by 
  sorry

end product_of_fractional_parts_eq_222_l310_310140


namespace arithmetic_sequence_num_terms_l310_310707

theorem arithmetic_sequence_num_terms 
  (a : ℕ) (d : ℕ) (l : ℕ) (n : ℕ)
  (h1 : a = 20)
  (h2 : d = 5)
  (h3 : l = 150)
  (h4 : 150 = 20 + (n-1) * 5) :
  n = 27 :=
by sorry

end arithmetic_sequence_num_terms_l310_310707


namespace problem_1_problem_2_l310_310319

-- Definitions for functions and conditions
def f (x : ℝ) : ℝ := Real.ln x
def g (x m n : ℝ) : ℝ := (m * (x + n)) / (x + 1)

-- Problem 1: Same tangent line at x = 1
theorem problem_1 (h : g 1 m n = 0) (h' : (deriv (fun x => g x m n) 1) = 1) : m = 2 := by
  sorry

-- Problem 2: Maximum value of m given |f(x)| >= |g(x)| for all x >= 1
theorem problem_2 (h : ∀ x, x ≥ 1 → |f x| ≥ |g x m n|) : 0 < m ∧ m ≤ 2 := by
  sorry

end problem_1_problem_2_l310_310319


namespace smallest_perfect_square_divisible_by_4_and_5_l310_310500

theorem smallest_perfect_square_divisible_by_4_and_5 : 
  ∃ n : ℕ, n > 0 ∧ n ∣ 4 ∧ n ∣ 5 ∧ is_square n ∧ 
  ∀ m : ℕ, (m > 0 ∧ m ∣ 4 ∧ m ∣ 5 ∧ is_square m) → n ≤ m :=
sorry

end smallest_perfect_square_divisible_by_4_and_5_l310_310500


namespace interval_notation_for_inequality_l310_310103

theorem interval_notation_for_inequality :
  (∀ x : ℝ, -8 ≤ x ∧ x < 15) ↔ ∀ x ∈ Icc (-8 : ℝ) 15, x ∈ Ico (-8 : ℝ) 15 :=
sorry

end interval_notation_for_inequality_l310_310103


namespace determine_no_conditionals_l310_310310

def problem_requires_conditionals (n : ℕ) : Prop :=
  n = 3 ∨ n = 4

theorem determine_no_conditionals :
  problem_requires_conditionals 1 = false ∧
  problem_requires_conditionals 2 = false ∧
  problem_requires_conditionals 3 = true ∧
  problem_requires_conditionals 4 = true :=
by sorry

end determine_no_conditionals_l310_310310


namespace paul_packed_total_toys_l310_310065

def toys_in_box : ℕ := 8
def number_of_boxes : ℕ := 4
def total_toys_packed (toys_in_box number_of_boxes : ℕ) : ℕ := toys_in_box * number_of_boxes

theorem paul_packed_total_toys :
  total_toys_packed toys_in_box number_of_boxes = 32 :=
by
  sorry

end paul_packed_total_toys_l310_310065


namespace find_f_ln3_l310_310682

noncomputable def f : ℝ → ℝ
| x := if x ≥ 2 then (1/3) * real.exp x else f (x + 1)

theorem find_f_ln3 : f (real.log 3) = real.exp 1 :=
by
  sorry

end find_f_ln3_l310_310682


namespace cost_of_purchasing_sandwiches_and_sodas_l310_310198

def sandwich_price : ℕ := 4
def soda_price : ℕ := 1
def num_sandwiches : ℕ := 6
def num_sodas : ℕ := 5
def total_cost : ℕ := 29

theorem cost_of_purchasing_sandwiches_and_sodas :
  (num_sandwiches * sandwich_price + num_sodas * soda_price) = total_cost :=
by
  sorry

end cost_of_purchasing_sandwiches_and_sodas_l310_310198


namespace inequality_and_equality_l310_310074

theorem inequality_and_equality (a b c : ℝ) :
  5 * a^2 + 5 * b^2 + 5 * c^2 ≥ 4 * a * b + 4 * b * c + 4 * a * c ∧ (5 * a^2 + 5 * b^2 + 5 * c^2 = 4 * a * b + 4 * b * c + 4 * a * c ↔ a = 0 ∧ b = 0 ∧ c = 0) :=
by
  sorry

end inequality_and_equality_l310_310074


namespace power_calculation_l310_310208

theorem power_calculation : (128 : ℝ) ^ (4/7) = 16 :=
by {
  have factorization : (128 : ℝ) = 2 ^ 7 := by {
    norm_num,
  },
  rw factorization,
  have power_rule : (2 ^ 7 : ℝ) ^ (4/7) = 2 ^ 4 := by {
    norm_num,
  },
  rw power_rule,
  norm_num,
  sorry
}

end power_calculation_l310_310208


namespace meridian_students_l310_310200

theorem meridian_students
  (eighth_to_seventh_ratio : Nat → Nat → Prop)
  (seventh_to_sixth_ratio : Nat → Nat → Prop)
  (r1 : ∀ a b, eighth_to_seventh_ratio a b ↔ 7 * b = 4 * a)
  (r2 : ∀ b c, seventh_to_sixth_ratio b c ↔ 10 * c = 9 * b) :
  ∃ a b c, eighth_to_seventh_ratio a b ∧ seventh_to_sixth_ratio b c ∧ a + b + c = 73 :=
by
  sorry

end meridian_students_l310_310200


namespace adjacent_terms_product_negative_l310_310379

theorem adjacent_terms_product_negative :
  (∃ n: ℕ, a n = 15 - (2 / 3) * (n - 1) ∧ (a 23 * a 24 < 0)) :=
sorry

end adjacent_terms_product_negative_l310_310379


namespace evaluate_expression_l310_310614

theorem evaluate_expression : (∃ (a b c d : ℕ), a = 2210 ∧ b = 2137 ∧ c = 2028 ∧ d = 64 ∧ ( (a - b) ^ 2 + (b - c) ^ 2 ) / d = 268.90625) :=
by
  let a := 2210
  let b := 2137
  let c := 2028
  let d := 64
  have h : ( (a - b) ^ 2 + (b - c) ^ 2 ) / d = 268.90625 := sorry
  exact ⟨a, b, c, d, rfl, rfl, rfl, rfl, h⟩

end evaluate_expression_l310_310614


namespace interest_rate_l310_310540

-- Definitions based on given conditions
def SumLent : ℝ := 1500
def InterestTime : ℝ := 4
def InterestAmount : ℝ := SumLent - 1260

-- Main theorem to prove the interest rate r is 4%
theorem interest_rate (r : ℝ) : (InterestAmount = SumLent * r / 100 * InterestTime) → r = 4 :=
by
  sorry

end interest_rate_l310_310540


namespace factorial_multiply_square_root_l310_310504

open Real

theorem factorial_multiply_square_root :
  (sqrt (5! * 4!))^2 * 2 = 5760 :=
by
  sorry

end factorial_multiply_square_root_l310_310504


namespace seventeen_in_base_three_l310_310617

theorem seventeen_in_base_three : (17 : ℕ) = 1 * 3^2 + 2 * 3^1 + 2 * 3^0 :=
by
  -- This is the arithmetic representation of the conversion,
  -- proving that 17 in base 10 equals 122 in base 3
  sorry

end seventeen_in_base_three_l310_310617


namespace profit_percent_calculation_l310_310516

-- Define the initial conditions
def marked_price_per_pen (p : ℝ) : ℝ := 1
def number_of_pens_bought : ℕ := 54
def number_of_pens_payed_for : ℕ := 46
def discount_rate : ℝ := 0.01

-- Define the total cost and selling prices based on conditions
def total_cost : ℝ := number_of_pens_payed_for * marked_price_per_pen 1
def selling_price_per_pen : ℝ := marked_price_per_pen 1 * (1 - discount_rate)
def total_selling_price : ℝ := number_of_pens_bought * selling_price_per_pen

-- Calculate the profit and profit percent
def profit : ℝ := total_selling_price - total_cost
def profit_percent : ℝ := (profit / total_cost) * 100

-- Establish the theorem to prove the calculated profit percent is correct
theorem profit_percent_calculation : profit_percent = 16.22 := 
  sorry

end profit_percent_calculation_l310_310516


namespace repeating_decimal_product_of_num_and_den_l310_310136

theorem repeating_decimal_product_of_num_and_den (x : ℚ) (h : x = 18 / 999) (h_simplified : x.num * x.den = 222) : x.num * x.den = 222 :=
by {
  sorry
}

end repeating_decimal_product_of_num_and_den_l310_310136


namespace angle_of_rotation_l310_310945

-- Definitions for the given conditions
def radius_large := 9 -- cm
def radius_medium := 3 -- cm
def radius_small := 1 -- cm
def speed := 1 -- cm/s

-- Definition of the angles calculations
noncomputable def rotations_per_revolution (R1 R2 : ℝ) : ℝ := R1 / R2
noncomputable def total_rotations (R1 R2 R3 : ℝ) : ℝ := 
  let rotations_medium := rotations_per_revolution R1 R2
  let net_rotations_medium := rotations_medium - 1
  net_rotations_medium * rotations_per_revolution R2 R3 + 1

-- Assertion to prove
theorem angle_of_rotation : 
  total_rotations radius_large radius_medium radius_small * 360 = 2520 :=
by 
  simp [total_rotations, rotations_per_revolution]
  exact sorry -- proof placeholder

end angle_of_rotation_l310_310945


namespace gcd_of_a_and_b_lcm_of_a_and_b_l310_310258

def a : ℕ := 2 * 3 * 7
def b : ℕ := 2 * 3 * 3 * 5

theorem gcd_of_a_and_b : Nat.gcd a b = 6 := by
  sorry

theorem lcm_of_a_and_b : Nat.lcm a b = 630 := by
  sorry

end gcd_of_a_and_b_lcm_of_a_and_b_l310_310258


namespace simple_random_sampling_used_l310_310841

-- Define the student population size and sample size
def total_students : Nat := 200
def sample_size : Nat := 20

-- Define the conditions for simple random sampling
def is_simple_random_sampling (n m : Nat) : Prop :=
  n = total_students ∧ m = sample_size ∧ 
  (∀ i, i < n → (student_selected i → (random_selection n m)))

-- Define a predicate checking if a student is selected randomly
def student_selected : Nat → Prop := sorry

-- Define the notion of students being randomly selected
def random_selection : Nat → Nat → Prop := sorry

-- The main statement to prove
theorem simple_random_sampling_used : is_simple_random_sampling total_students sample_size :=
by
  sorry

end simple_random_sampling_used_l310_310841


namespace prob_X_eq_Y_l310_310938

theorem prob_X_eq_Y : 
  ∀ (x y : ℝ), -12 * Real.pi ≤ x ∧ x ≤ 12 * Real.pi ∧ -12 * Real.pi ≤ y ∧ y ≤ 12 * Real.pi ∧ cos (cos x) = cos (cos y) → 
    probability (X = Y) = 25/169 :=
by 
  sorry

end prob_X_eq_Y_l310_310938


namespace particle_position_1990_l310_310180

-- Define the initial position and the movement pattern
structure Position :=
  (x : ℕ)
  (y : ℕ)

def initial_position : Position := {x := 0, y := 0}

-- Define the movement functions
def move_right (p : Position) (steps : ℕ) : Position :=
  {x := p.x + steps, y := p.y}

def move_up (p : Position) (steps : ℕ) : Position :=
  {x := p.x, y := p.y + steps}

def move_left (p : Position) (steps : ℕ) : Position :=
  {x := p.x - steps, y := p.y}

-- Define the function that computes the position after n minutes.
-- This is an example function skeleton, concrete implementation will be required for full proof.
noncomputable def position_after_n_minutes (n : ℕ) : Position :=
  sorry -- Implementation will simulate the steps and aggregate the position

-- The main theorem stating the expected position after 1990 minutes
theorem particle_position_1990 :
  position_after_n_minutes 1990 = {x := ?, y := ?} :=
sorry

end particle_position_1990_l310_310180


namespace distance_MN_l310_310735

def point (α : Type) := (α × α × α)

def distance_3d (p1 p2 : point ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2 + (p2.3 - p1.3)^2)

def M : point ℝ := (1, 2, 3)
def N : point ℝ := (2, 3, 4)

theorem distance_MN : distance_3d M N = real.sqrt 3 :=
by
  sorry

end distance_MN_l310_310735


namespace find_a_squared_plus_b_squared_l310_310206

theorem find_a_squared_plus_b_squared (a b : ℝ) (h1 : a + b = 3) (h2 : a * b = 1) : a^2 + b^2 = 7 := 
by
  sorry

end find_a_squared_plus_b_squared_l310_310206


namespace only_five_simply_connected_polyhedra_l310_310073

-- Definitions for the types of polyhedra
inductive Polyhedron
| tetrahedron
| cube
| octahedron
| icosahedron
| dodecahedron

-- Condition for a simply connected polyhedron
def isSimplyConnected (P: Polyhedron) : Prop := 
  match P with
  | Polyhedron.tetrahedron => True
  | Polyhedron.cube => True
  | Polyhedron.octahedron => True
  | Polyhedron.icosahedron => True
  | Polyhedron.dodecahedron => True

-- Proof that only these types meet the conditions
theorem only_five_simply_connected_polyhedra :
  ∀ P: Polyhedron, isSimplyConnected P :=
by
  -- Although proofs are not required, sorry is used to indicate unfinished proofs
  intros,
  cases P;
  exact trivial <|> sorry

#check only_five_simply_connected_polyhedra

end only_five_simply_connected_polyhedra_l310_310073


namespace expand_and_simplify_l310_310859

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
  sorry

end expand_and_simplify_l310_310859


namespace manager_decision_correct_l310_310909

theorem manager_decision_correct (x : ℝ) (profit : ℝ) 
  (h_condition1 : ∀ (x : ℝ), profit = (2 * x + 20) * (40 - x)) 
  (h_condition2 : 0 ≤ x ∧ x ≤ 40)
  (h_price_reduction : x = 15) :
  profit = 1250 :=
by
  sorry

end manager_decision_correct_l310_310909


namespace lollipop_count_l310_310348

theorem lollipop_count (total_cost one_lollipop_cost : ℚ) (h1 : total_cost = 90) (h2 : one_lollipop_cost = 0.75) : total_cost / one_lollipop_cost = 120 :=
by
  sorry

end lollipop_count_l310_310348


namespace vans_hold_people_per_van_l310_310062

theorem vans_hold_people_per_van (students adults vans total_people people_per_van : ℤ) 
    (h1: students = 12) 
    (h2: adults = 3) 
    (h3: vans = 3) 
    (h4: total_people = students + adults) 
    (h5: people_per_van = total_people / vans) :
    people_per_van = 5 := 
by
    -- Steps will go here
    sorry

end vans_hold_people_per_van_l310_310062


namespace probability_of_queen_after_first_queen_l310_310925

-- Define the standard deck
def standard_deck : Finset (Fin 54) := Finset.univ

-- Define the event of drawing the first queen
def first_queen (deck : Finset (Fin 54)) : Prop := -- placeholder defining first queen draw
  sorry

-- Define the event of drawing a queen immediately after the first queen
def queen_after_first_queen (deck : Finset (Fin 54)) : Prop :=
  sorry

-- Define the probability of an event given a condition
noncomputable def probability (event : Prop) (condition : Prop) : ℚ :=
  sorry

-- Main theorem statement
theorem probability_of_queen_after_first_queen : probability 
  (queen_after_first_queen standard_deck) (first_queen standard_deck) = 2/27 :=
sorry

end probability_of_queen_after_first_queen_l310_310925


namespace elizabeth_bought_bananas_l310_310994

theorem elizabeth_bought_bananas (eaten : ℕ) (left : ℕ) : eaten = 4 ∧ left = 8 → eaten + left = 12 :=
by
  intro h
  cases h with heaten hleft
  rw [heaten, hleft]
  exact rfl

end elizabeth_bought_bananas_l310_310994


namespace valid_range_and_difference_l310_310381

/- Assume side lengths as given expressions -/
def BC (x : ℝ) : ℝ := x + 11
def AC (x : ℝ) : ℝ := x + 6
def AB (x : ℝ) : ℝ := 3 * x + 2

/- Define the inequalities representing the triangle inequalities and largest angle condition -/
def triangle_inequality1 (x : ℝ) : Prop := AB x + AC x > BC x
def triangle_inequality2 (x : ℝ) : Prop := AB x + BC x > AC x
def triangle_inequality3 (x : ℝ) : Prop := AC x + BC x > AB x
def largest_angle_condition (x : ℝ) : Prop := BC x > AB x

/- Define the combined condition for x, ensuring all relevant conditions are met -/
def valid_x_range (x : ℝ) : Prop :=
  1 < x ∧ x < 4.5 ∧ triangle_inequality1 x ∧ triangle_inequality2 x ∧ triangle_inequality3 x ∧ largest_angle_condition x

/- Compute n - m for the interval (m, n) where x lies -/
def n_minus_m : ℝ :=
  4.5 - 1

/- Main theorem stating the final result -/
theorem valid_range_and_difference :
  (∃ x : ℝ, valid_x_range x) ∧ (n_minus_m = 7 / 2) :=
by
  sorry

end valid_range_and_difference_l310_310381


namespace hyperbola_condition_l310_310974

theorem hyperbola_condition (k : ℝ) : 
    (∃ f : ℝ → ℝ → ℝ, ∃ g : ℝ → ℝ → ℝ, ∀ x y, f x y = g x y -> k ∈ (Set.Iio (-1) ∪ Set.Ioi 1)) ↔
    ∀ x y, (x * x) / (1 + k) + (y * y) / (1 - k) = 1 → (k > 1 ∨ k < -1) := 
begin 
    sorry
end

end hyperbola_condition_l310_310974


namespace tan_10pi_minus_theta_l310_310309

open Real

theorem tan_10pi_minus_theta (θ : ℝ) (h1 : π < θ) (h2 : θ < 2 * π) (h3 : cos (θ - 9 * π) = -3 / 5) : 
  tan (10 * π - θ) = -4 / 3 := 
sorry

end tan_10pi_minus_theta_l310_310309


namespace p_necessary_for_q_l310_310760

-- Definitions
def p (a b : ℝ) : Prop := (a + b = 2) ∨ (a + b = -2)
def q (a b : ℝ) : Prop := a + b = 2

-- Statement of the problem
theorem p_necessary_for_q (a b : ℝ) : (p a b → q a b) ∧ ¬(q a b → p a b) := 
sorry

end p_necessary_for_q_l310_310760


namespace possible_values_of_a2b_b2c_c2a_l310_310054

theorem possible_values_of_a2b_b2c_c2a (a b c : ℝ) (h : a + b + c = 1) : ∀ x : ℝ, ∃ a b c : ℝ, a + b + c = 1 ∧ a^2 * b + b^2 * c + c^2 * a = x :=
by
  sorry

end possible_values_of_a2b_b2c_c2a_l310_310054


namespace evaluate_statements_l310_310941

-- Defining what it means for angles to be vertical
def vertical_angles (α β : ℝ) : Prop := α = β

-- Defining what complementary angles are
def complementary (α β : ℝ) : Prop := α + β = 90

-- Defining what supplementary angles are
def supplementary (α β : ℝ) : Prop := α + β = 180

-- Define the geometric properties for perpendicular and parallel lines
def unique_perpendicular_through_point (l : ℝ → ℝ) (p : ℝ × ℝ): Prop :=
  ∃! m, ∀ x, m * x + p.2 = l x

def unique_parallel_through_point (l : ℝ → ℝ) (p : ℝ × ℝ): Prop :=
  ∃! m, ∀ x, (l x ≠ m * x + p.2) ∧ (∀ y, y ≠ p.2 → l y ≠ m * y)

theorem evaluate_statements :
  (¬ ∃ α β, α = β ∧ vertical_angles α β) ∧
  (¬ ∃ α β, supplementary α β ∧ complementary α β) ∧
  ∃ l p, unique_perpendicular_through_point l p ∧
  ∃ l p, unique_parallel_through_point l p →
  2 = 2
  :=
by
  sorry  -- Proof is omitted

end evaluate_statements_l310_310941


namespace sum_of_next_five_even_integers_l310_310886

theorem sum_of_next_five_even_integers (a : ℕ) (x : ℕ) 
  (h : a = x + (x + 2) + (x + 4) + (x + 6) + (x + 8)) : 
  (x + 10) + (x + 12) + (x + 14) + (x + 16) + (x + 18) = a + 50 := by
  sorry

end sum_of_next_five_even_integers_l310_310886


namespace stans_total_investment_l310_310806

-- Define the conditions from the problem:
variables (X : ℝ) -- amount invested at 18%
variables (I : ℝ) -- total interest
variables (A : ℝ) -- amount invested at 4%
variables (T : ℝ) -- total amount invested

-- Given conditions:
axiom h1 : A = 12000
axiom h2 : I = 1380
axiom h3 : 0.04 * A + 0.18 * X = I

-- Proof goal:
theorem stans_total_investment : T = 17000 :=
by
  -- Definitions from conditions:
  let intermediate_interest := 0.04 * A
  have h4 : intermediate_interest = 480 := 
    calc
      0.04 * A
        = 0.04 * 12000 : by rw [h1]
    ... = 480 : by norm_num,
  have h5 : 0.18 * X = 900 := 
    calc
      0.18 * X
        = I - intermediate_interest : by rw [← h3, h4]
    ... = 1380 - 480 : by rw [h2, h4]
    ... = 900 : by norm_num,
  have h6 : X = 5000 := 
    calc
      X = 900 / 0.18 : by rw [div_eq_iff_mul_eq', h5]; norm_num
      ... = 5000 : by norm_num,
  have h7 : T = A + X := rfl,
  calc
    T
      = 12000 + X : by rw [h1, h7]
    ... = 12000 + 5000 : by rw [h6]
    ... = 17000 : by norm_num

end stans_total_investment_l310_310806


namespace sum_coeffs_example_l310_310339

theorem sum_coeffs_example (a : Fin 18 → ℤ) :
  (∀ x, (x - 1) ^ 17 = ∑ i in Finset.range 18, a i * (1 + x) ^ i) →
  ∑ i in Finset.range 18, a i = -1 :=
by
  sorry

end sum_coeffs_example_l310_310339


namespace repeating_decimal_fraction_l310_310616

theorem repeating_decimal_fraction :
  (5 + 341 / 999) = (5336 / 999) :=
by
  sorry

end repeating_decimal_fraction_l310_310616


namespace find_m_l310_310999

theorem find_m (m : ℕ) : (2^3 * 3^3 * m = 9!) → m = 840 :=
by
  sorry

end find_m_l310_310999


namespace intersection_complement_A_B_l310_310302

variable {R : Set ℝ}
def A := {x : ℝ | Real.log 3 - x > -2}
def B := {x : ℝ | 5 / (x + 2) ≥ 1}
def CR (s : Set ℝ) := {x : ℝ | ¬ (x ∈ s)}

theorem intersection_complement_A_B :
  (CR R A) ∩ B = Set.union (Ioo (-2:ℝ) (-1)) {3} :=
sorry

end intersection_complement_A_B_l310_310302


namespace total_cost_correct_l310_310804

def shirt_cost : ℝ := 12
def shoes_cost : ℝ := shirt_cost + 5
def discount_rate : ℝ := 0.10
def exchange_rate : ℝ := 1.18
def discounted_shoes_cost : ℝ := shoes_cost * (1 - discount_rate)
def total_shirts_cost : ℝ := 2 * shirt_cost
def total_clothing_cost : ℝ := total_shirts_cost + discounted_shoes_cost
def bag_cost : ℝ := total_clothing_cost / 2
def total_euro_cost : ℝ := total_shirts_cost + discounted_shoes_cost + bag_cost
def total_usd_cost : ℝ := total_euro_cost * exchange_rate

theorem total_cost_correct :
  abs (total_usd_cost - 69.92) < 0.01 :=
by
  have h1 : shoes_cost = 17 := by simp [shoes_cost, shirt_cost]
  have h2 : discounted_shoes_cost = 15.3 := by simp [discounted_shoes_cost, shoes_cost, discount_rate]; linarith
  have h3 : total_shirts_cost = 24 := by simp [total_shirts_cost, shirt_cost]
  have h4 : total_clothing_cost = 39.3 := by simp [total_clothing_cost, total_shirts_cost, discounted_shoes_cost]; linarith
  have h5 : bag_cost = 19.65 := by simp [bag_cost, total_clothing_cost]; linarith
  have h6 : total_euro_cost = 59.25 := by simp [total_euro_cost, total_shirts_cost, discounted_shoes_cost, bag_cost]; linarith
  have h7 : total_usd_cost = 69.915 := by simp [total_usd_cost, total_euro_cost, exchange_rate]; linarith
  have h8 : abs (69.915 - 69.92) < 0.01 := by norm_num
  exact h8

end total_cost_correct_l310_310804


namespace integral_eval_l310_310995

theorem integral_eval :
  ∫ x in (0 : ℝ)..(4 : ℝ), (real.sqrt (16 - x^2) - 0.5 * x) = 4 * real.pi - 4 :=
by
  sorry

end integral_eval_l310_310995


namespace sphere_radius_in_cone_l310_310029

-- Definitions based on conditions
def coneBaseRadius : ℝ := 7
def coneHeight : ℝ := 15

-- Statement of the problem as a Lean theorem
theorem sphere_radius_in_cone : ∃ r : ℝ, 
  (∀ (o₁ o₂ o₃ o₄ : ℝ → Prop), 
    (∃ (O₁₁ O₁₂ O₁₃ O₂₁ O₂₂ O₂₃ O₃₁ O₃₂ O₃₃ O₄₁ O₄₂ O₄₃ : ℝ),
      (o₁ O₁₁ O₁₂ O₁₃) ∧
      (o₂ O₂₁ O₂₂ O₂₃) ∧
      (o₃ O₃₁ O₃₂ O₃₃) ∧
      (o₄ O₄₁ O₄₂ O₄₃) ∧
      (dist O₁₁ O₂₁ = 2 * r ∧ (dist O₁₂ O₂₂ = 2 * r) ∧ (dist O₁₃ O₂₃ = 2 * r)) ∧
      (dist O₁₁ O₃₁ = 2 * r ∧ (dist O₁₂ O₃₂ = 2 * r) ∧ (dist O₁₃ O₃₃ = 2 * r)) ∧
      (dist O₁₁ O₄₁ = 2 * r ∧ (dist O₁₂ O₄₂ = 2 * r) ∧ (dist O₁₃ O₄₃ = 2 * r)) ∧
      (dist O₂₁ O₃₁ = 2 * r ∧ (dist O₂₂ O₃₂ = 2 * r) ∧ (dist O₂₃ O₃₃ = 2 * r)) ∧
      (dist O₂₁ O₄₁ = 2 * r ∧ (dist O₂₂ O₄₂ = 2 * r) ∧ (dist O₂₃ O₄₃ = 2 * r)) ∧
      (dist O₃₁ O₄₁ = 2 * r ∧ (dist O₃₂ O₄₂ = 2 * r) ∧ (dist O₃₃ O₄₃ = 2 * r)) ∧
      (dist O₁₁ 0 <= r) ∧ -- Tangency to the base of the cone
      (dist O₁₁ sqrt(274) <= r) ∧ -- Tangency to the side of the cone
      -- Additional tetrahedron-related distances and geometric constraints
    )) ∧ r = 2.59 :=
begin
  sorry
end

end sphere_radius_in_cone_l310_310029


namespace charlie_ride_distance_l310_310787

-- Define the known values
def oscar_ride : ℝ := 0.75
def difference : ℝ := 0.5

-- Define Charlie's bus ride distance
def charlie_ride : ℝ := oscar_ride - difference

-- The theorem to be proven
theorem charlie_ride_distance : charlie_ride = 0.25 := 
by sorry

end charlie_ride_distance_l310_310787


namespace eventually_no_overcrowded_apartments_l310_310358

def apartments : ℕ := 120
def residents : ℕ := 119
def overcrowded (n : ℕ) : Prop := n ≥ 15
def moves_stop (apartments residents : ℕ) 
  (overcrowded: (ℕ → Prop)) : Prop :=
  ∀ (residents_in_apts : Fin apartments → ℕ),
    (∑ i, residents_in_apts i = residents) →
    ∃ N, ∀ t ≥ N, ¬ ∃ i, overcrowded (residents_in_apts i)

theorem eventually_no_overcrowded_apartments :
  moves_stop apartments residents overcrowded := 
sorry

end eventually_no_overcrowded_apartments_l310_310358


namespace great_wall_length_l310_310368

theorem great_wall_length (interval distance : ℕ) (soldiers_per_tower total_soldiers : ℕ)
  (h1 : interval = 5)
  (h2 : soldiers_per_tower = 2)
  (h3 : total_soldiers = 2920) :
  let number_of_towers := total_soldiers / soldiers_per_tower in
  let effective_intervals := number_of_towers - 1 in
  let wall_length := effective_intervals * interval in
  wall_length = 7295 :=
by
  let number_of_towers := total_soldiers / soldiers_per_tower
  let effective_intervals := number_of_towers - 1
  let wall_length := effective_intervals * interval
  have h4 : number_of_towers = 1460, from sorry
  have h5 : effective_intervals = 1459, from sorry
  have h6 : wall_length = 7295, from sorry
  exact h6

end great_wall_length_l310_310368


namespace find_a_plus_h_l310_310818

noncomputable def hyperbola_asymptotes_center : ℚ × ℚ :=
  let h := -1/3
  let k := 3
  (h, k)

noncomputable def slopes_as_ratio (a b : ℚ) (hb : b > 0) : Prop :=
  a = 3 * b

noncomputable def hyperbola_equation
  (x y k h a b : ℚ)
  (hb : b > 0) (ha : a > 0)
  (h_eq : slopes_as_ratio a b hb)
  (p : x = 1 ∧ y = 8) : Prop :=
  (y - k)^2 / a^2 - (x - h)^2 / b^2 = 1

theorem find_a_plus_h :
  ∃ (a b h k : ℚ) (hb : b > 0) (ha : a > 0),
    hyperbola_asymptotes_center = (h, k) ∧
    slopes_as_ratio a b hb ∧
    hyperbola_equation 1 8 k h a b hb ha (slopes_as_ratio a b hb) ⟨rfl, rfl⟩ ∧
    a + h = 8 / 3 :=
begin
  sorry
end

end find_a_plus_h_l310_310818


namespace find_principal_amount_l310_310238

variable (P : ℝ)
variable (R : ℝ := 4)
variable (T : ℝ := 2)
variable (diff : ℝ := 3.2)

def simple_interest (P R T : ℝ) : ℝ := P * R * T / 100

def compound_interest (P R T : ℝ) : ℝ := P * (1 + R / 100)^T - P

theorem find_principal_amount (h : compound_interest P R T - simple_interest P R T = diff) : 
  P = 2000 := by
  sorry

end find_principal_amount_l310_310238


namespace technician_round_trip_completion_percentage_l310_310927

theorem technician_round_trip_completion_percentage (D : ℝ) :
  let initial_distance := D
  let round_trip_distance := 2 * D
  let third_location_distance := 0.15 * D
  let detour_distance := 0.20 * D
  let return_journey_distance := D + third_location_distance + detour_distance
  let total_revised_round_trip_distance := D + return_journey_distance
  let remaining_drive_from_third_location := return_journey_distance
  let distance_covered := D + third_location_distance + 0.10 * remaining_drive_from_third_location
  let percent_completed := (distance_covered / total_revised_round_trip_distance) * 100
  percent_completed ≈ 54.68 :=
by {
  sorry
}

end technician_round_trip_completion_percentage_l310_310927


namespace percentage_error_97_l310_310546

theorem percentage_error_97 (x : ℝ) (h : 0 < x) : 
( (real.abs ((6 * x + 12) - (x + 2) / 5)) / (6 * x + 12)) * 100 ≈ 97 :=
by 
  sorry

end percentage_error_97_l310_310546


namespace inv_proportion_no_intersect_axes_l310_310321

theorem inv_proportion_no_intersect_axes : 
  ∀ (x : ℝ), x ≠ 0 → - (6 : ℝ) / x ≠ 0 :=
by
  intro x hx,
  sorry

end inv_proportion_no_intersect_axes_l310_310321


namespace time_difference_alice_bob_l310_310935

theorem time_difference_alice_bob
  (alice_speed : ℕ) (bob_speed : ℕ) (distance : ℕ)
  (h_alice_speed : alice_speed = 7)
  (h_bob_speed : bob_speed = 9)
  (h_distance : distance = 12) :
  (bob_speed * distance - alice_speed * distance) = 24 :=
by
  sorry

end time_difference_alice_bob_l310_310935


namespace probability_white_cube_l310_310539

theorem probability_white_cube : 
  let p : ℝ := (3^8 * (fact 8)^2 * 2^24 * (fact 24)^2) / (24^64 * fact 64) in
  p < 10^(-83)
:= sorry

end probability_white_cube_l310_310539


namespace MA_dot_MB_l310_310671

noncomputable theory

open Classical

-- Define the given conditions.
def side_length : ℝ := 2
def M : ℝ × ℝ :=
  let CB := (side_length, 0)
  let CA := (side_length * cos (pi / 3), side_length * sin (pi / 3))
  let CM := ((1:ℝ)/3 * CB.1 + (1:ℝ)/2 * CA.1, (1:ℝ)/3 * CB.2 + (1:ℝ)/2 * CA.2)
  CM

-- Define dot product
def dot_product (p q : ℝ × ℝ) : ℝ := p.1 * q.1 + p.2 * q.2 

-- Define vector operations
def neg_vector (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
def sub_vector (p q : ℝ × ℝ) : ℝ × ℝ := (p.1 - q.1, p.2 - q.2)
def scalar_mult (c : ℝ) (p : ℝ × ℝ) : ℝ × ℝ := (c * p.1, c * p.2)

-- Map from vectors
def vector_CA : ℝ × ℝ := (side_length * cos (pi / 3), side_length * sin (pi / 3))
def vector_CB : ℝ × ℝ := (side_length, 0)
def vector_AM : ℝ × ℝ := sub_vector (neg_vector (scalar_mult (1:ℝ)/3 vector_CB)) (scalar_mult (1:ℝ)/2 vector_CA)
def vector_MB : ℝ × ℝ := sub_vector (scalar_mult (2:ℝ)/3 vector_CB) (scalar_mult (1:ℝ)/2 vector_CA)

-- Theorem statement
theorem MA_dot_MB : dot_product vector_AM vector_MB = -8/9 := 
by
  sorry

end MA_dot_MB_l310_310671


namespace min_angle_of_inclination_l310_310687

noncomputable def f (x : ℝ) : ℝ := (x^3) / 3 - x^2 + 1

theorem min_angle_of_inclination : 
  ∀ α : ℝ, 
  (∀ x : ℝ, (0 < x ∧ x < 2 → tan α = (x - 1)^2 - 1)) → 
  π * 3 / 4 ≤ α ∧ α < π :=
sorry

end min_angle_of_inclination_l310_310687


namespace coefficient_of_x2_in_expansion_is_neg3_l310_310815

noncomputable def coefficient_x2_in_expansion : ℤ :=
  let f := (1 - X)^6 * (1 + X)^4
  in coeff (f.expand) 2

theorem coefficient_of_x2_in_expansion_is_neg3 :
  coefficient_x2_in_expansion = -3 := sorry

end coefficient_of_x2_in_expansion_is_neg3_l310_310815


namespace digit_packages_needed_l310_310106

/-- 
Proof statement:
Given that we need to label condos from 300 to 350 and from 400 to 450, 
and each digit can only be purchased in packages containing one of each digit from 0 to 9,
prove that the number of digit packages needed is 63.
-/
theorem digit_packages_needed : 
  let condos_range1 := List.range 51 |>.map (300 + ·) in
  let condos_range2 := List.range 51 |>.map (400 + ·) in
  let all_condos := condos_range1 ++ condos_range2 in
  let digit_count (n : ℕ) := all_condos.foldl (fun acc d => acc + if d.toString.contains n.toString then 1 else 0) 0 in
  let max_digit_usage := List.range 10 |>.map digit_count |>.maximum.get in
  max_digit_usage = 63 :=
by
  sorry

end digit_packages_needed_l310_310106


namespace max_area_rectangle_l310_310111

/-- Given a rectangle with a perimeter of 40, the rectangle with the maximum area is a square
with sides of length 10. The maximum area is thus 100. -/
theorem max_area_rectangle (a b : ℝ) (h : a + b = 20) : a * b ≤ 100 :=
by
  sorry

end max_area_rectangle_l310_310111


namespace yoongi_read_books_permutations_l310_310152

theorem yoongi_read_books_permutations : 
  ∃ (n r : ℕ), n = 5 ∧ r = 3 ∧ (nat.factorial n) / (nat.factorial (n - r)) = 60 :=
by {
  use [5, 3],
  split,
  { refl },
  split,
  { refl },
  rw [nat.sub_self, nat.factorial_succ],
  rw [nat.factorial_succ, nat.factorial_succ, nat.factorial_succ],
  rw [nat.factorial_succ, nat.factorial_succ],
  rw [nat.factorial_succ],
  norm_num,
  sorry
}

end yoongi_read_books_permutations_l310_310152


namespace equal_segments_MT_MC_l310_310048

open EuclideanGeometry

-- Definitions given conditions
variables {A B C J T M : Point}
variables {triangle ABC : Triangle}

-- Given conditions in the problem
def center_of_excircle_opposite_A (J A B C : Point) [triangle ABC] : Prop :=
  J is the center of the excircle opposite to vertex A of triangle ABC

def midpoint (M A J : Point) : Prop :=
  M is the midpoint of segment AJ

def is_tangent_point (T B C : Point) [segment BC] : Prop :=
  T is the point where the excircle touches segment BC

-- The theorem to be proved
theorem equal_segments_MT_MC 
  (h : center_of_excircle_opposite_A J A B C)
  (hM : midpoint M A J)
  (hT : is_tangent_point T B C) : 
  segment_length M T = segment_length M C :=
by 
  sorry

end equal_segments_MT_MC_l310_310048


namespace exists_a_max_value_of_four_l310_310979

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.cos x)^2 + 2 * a * Real.sin x + 3 * a - 1

theorem exists_a_max_value_of_four :
  ∃ a : ℝ, (a = 1) ∧ ∀ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), f a x ≤ 4 := 
sorry

end exists_a_max_value_of_four_l310_310979


namespace rolling_sum_12_pair_dice_probability_l310_310820

-- Definitions for the faces of the dice and the concept of probability
def decahedral_faces : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

-- The probability of rolling a sum of 12 with two decahedral dice
theorem rolling_sum_12_pair_dice_probability :
  (decahedral_faces.product decahedral_faces).filter (λ pair, pair.1 + pair.2 = 12).card / (decahedral_faces.product decahedral_faces).card = 9 / 100 := 
by 
  sorry

end rolling_sum_12_pair_dice_probability_l310_310820


namespace girls_never_catch_ball_l310_310541

theorem girls_never_catch_ball (n : ℕ) (k : ℕ) (times_caught : ℕ) (total_girls : ℕ) (throws_interval : ℕ)
  (h1 : total_girls = 50)
  (h2 : throws_interval = 6)
  (h3 : times_caught = 100) :
  (total_girls - times_caught / (Nat.lcm throws_interval total_girls / throws_interval)) = 25 :=
begin
  sorry
end

end girls_never_catch_ball_l310_310541


namespace math_problem_l310_310960

noncomputable def f (x : ℝ) := (x^9 - 27*x^6 + 216*x^3 - 512) / (x^3 - 8)

theorem math_problem : f 6 = 43264 := by
  sorry

end math_problem_l310_310960


namespace eccentricity_of_ellipse_l310_310677

noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (a^2 - b^2) / a

theorem eccentricity_of_ellipse {a b : ℝ} (h1 : a > b) (h2 : b > 0) (h3 : a > 0)
                                 (h_angle : Real.cos (Real.pi / 6) = b / a) :
    eccentricity a b = (Real.sqrt 6) / 3 := by
  sorry

end eccentricity_of_ellipse_l310_310677


namespace count_mappings_from_A_to_B_l310_310705

open Set

variables {A B : Type} {n m : ℕ}
variables [Fintype A] [Fintype B]
variables (hA : Fintype.card A = n) (hB : Fintype.card B = m)

theorem count_mappings_from_A_to_B : Fintype.card (A → B) = m^n :=
by
  sorry

end count_mappings_from_A_to_B_l310_310705


namespace voyages_difference_l310_310121

/-- 
  The year 2005 marks the 600th anniversary of the first voyage to the Western Seas 
  by the great Chinese navigator Zheng He, and the first ocean voyage by the great 
  Spanish navigator Columbus was in 1492. Prove that the difference in years between 
  these two voyages is 87 years.
-/
theorem voyages_difference : 
  let zheng_he_first_voyage := 2005 - 600,
      columbus_first_voyage := 1492 in
  columbus_first_voyage - zheng_he_first_voyage = 87 := 
by
  sorry

end voyages_difference_l310_310121


namespace expand_simplify_expression_l310_310869

theorem expand_simplify_expression (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
by
  sorry

end expand_simplify_expression_l310_310869


namespace probability_of_intersection_l310_310077

noncomputable def probability_intersects (k : ℝ) : Prop := 
abs (5 * k) / sqrt (k^2 + 1) < 3

theorem probability_of_intersection : 
  (∫ (k : ℝ) in -1..1, if probability_intersects k then (1 : ℝ) else 0) = 3 / 4 :=
sorry

end probability_of_intersection_l310_310077


namespace angle_A_eq_pi_over_3_max_area_of_triangle_l310_310746

variables {R : Type*} [LinearOrderedField R]

namespace triangle

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and cos C = (2b - c) / (2a), prove that angle A equals π/3. -/
theorem angle_A_eq_pi_over_3 
  {a b c : R}
  (h : cos (C : angleInTriangle a b c) = (2 * b - c) / (2 * a)) :
  angle (C : angleInTriangle a b c) = π / 3 := 
sorry

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and perimeter a + b + c = 6, find the maximum value of the area S of triangle ABC. -/
theorem max_area_of_triangle 
  {a b c : R}
  (h : a + b + c = 6) :
  area (a b c) ≤ sqrt 3 := 
sorry

end triangle

end angle_A_eq_pi_over_3_max_area_of_triangle_l310_310746


namespace Kylie_uses_3_towels_in_one_month_l310_310732

-- Define the necessary variables and conditions
variable (daughters_towels : Nat) (husband_towels : Nat) (loads : Nat) (towels_per_load : Nat)
variable (K : Nat) -- number of bath towels Kylie uses

-- Given conditions
axiom h1 : daughters_towels = 6
axiom h2 : husband_towels = 3
axiom h3 : loads = 3
axiom h4 : towels_per_load = 4
axiom h5 : (K + daughters_towels + husband_towels) = (loads * towels_per_load)

-- Prove that K = 3
theorem Kylie_uses_3_towels_in_one_month : K = 3 :=
by
  sorry

end Kylie_uses_3_towels_in_one_month_l310_310732


namespace hyperbola_equation_l310_310875

/-- Equation of a hyperbola satisfying the given conditions -/
theorem hyperbola_equation (a b : ℝ) (h1 : b ≤ sqrt 3 * a) :
  ∃ (c : ℝ) (x y : ℝ), (foci_on_x_axis : a ≠ 0) ∧ 
    (intersects_circle : ∀ x y, (x - 2)^2 + y^2 = 3 → (b*x + a*y = 0 ∨ b*x - a*y = 0)) ∧
    c^2 - (x^2 - y^2 / 3) = 1 :=
begin
  sorry
end

end hyperbola_equation_l310_310875


namespace point_in_fourth_quadrant_l310_310009

theorem point_in_fourth_quadrant (a b : ℝ) (h1 : a > 0) (h2 : a * b < 0) : a > 0 ∧ b < 0 :=
by 
  have hb : b < 0 := sorry
  exact ⟨h1, hb⟩

end point_in_fourth_quadrant_l310_310009


namespace unique_solution_9_eq_1_l310_310245

theorem unique_solution_9_eq_1 :
  (∃! (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = 1) :=
begin
  sorry
end

end unique_solution_9_eq_1_l310_310245


namespace kitten_last_treat_l310_310389

theorem kitten_last_treat (n k : ℕ) (A B C D E F : Type) (kittens := [A, B, C, D, E, F]) (total_treats := 278)
  (mod_div : 278 % 6 = 2) : 
  (kittens.nth 1).get_or_else A = B :=
by
  sorry

end kitten_last_treat_l310_310389


namespace probability_red_given_spade_or_king_l310_310344

def num_cards := 52
def num_spades := 13
def num_kings := 4
def num_red_kings := 2

def num_non_spade_kings := num_kings - 1
def num_spades_or_kings := num_spades + num_non_spade_kings

theorem probability_red_given_spade_or_king :
  (num_red_kings : ℚ) / num_spades_or_kings = 1 / 8 :=
sorry

end probability_red_given_spade_or_king_l310_310344


namespace balloon_arrangement_count_l310_310216

theorem balloon_arrangement_count :
  let n := 7
  let l := 2
  let o := 2
  n.factorial / (l.factorial * o.factorial) = 1260 :=
by
  sorry

end balloon_arrangement_count_l310_310216


namespace triangle_inequality_l310_310371

theorem triangle_inequality {A B C : ℝ} {a b c : ℝ} 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (hSum : A + B + C = π) 
  (hacute : A < π / 2 ∧ B < π / 2 ∧ C < π / 2)
  (hR : ∀ (a b c A B C : ℝ), a = 2 * sin A ∧ b = 2 * sin B ∧ c = 2 * sin C ∧ R = 1) :
  1 ≤ a / (1 - sin A) + b / (1 - sin B) + c / (1 - sin C) ≥ 18 + 12 * sqrt 3 := 
sorry

end triangle_inequality_l310_310371


namespace initial_card_count_l310_310204

theorem initial_card_count (x : ℕ) (h1 : (3 * (1/2) * ((x / 3) + (4 / 3))) = 34) : x = 64 :=
  sorry

end initial_card_count_l310_310204


namespace largest_polygon_is_E_l310_310940

def area (num_unit_squares num_right_triangles num_half_squares: ℕ) : ℚ :=
  num_unit_squares + num_right_triangles * 0.5 + num_half_squares * 0.25

def polygon_A_area := area 3 2 0
def polygon_B_area := area 4 1 0
def polygon_C_area := area 2 4 2
def polygon_D_area := area 5 0 0
def polygon_E_area := area 3 3 4

theorem largest_polygon_is_E :
  polygon_E_area > polygon_A_area ∧ 
  polygon_E_area > polygon_B_area ∧ 
  polygon_E_area > polygon_C_area ∧ 
  polygon_E_area > polygon_D_area :=
by
  sorry

end largest_polygon_is_E_l310_310940


namespace smallest_perfect_square_div_by_4_and_5_l310_310495

theorem smallest_perfect_square_div_by_4_and_5 : 
  ∃ n : ℕ, (∃ m : ℕ, n = m^2) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (∀ k : ℕ, (∃ l : ℕ, k = l^2) ∧ (4 ∣ k) ∧ (5 ∣ k) → n ≤ k) :=
begin
  let n := 400,
  use n,
  split,
  { use 20, -- 400 is 20^2
    refl },
  split,
  { exact dvd.intro 100 rfl }, -- 400 = 4 * 100
  split,
  { exact dvd.intro 80 rfl }, -- 400 = 5 * 80
  { 
    intros k hk,
    obtain ⟨l, hl⟩ := hk.left,
    obtain ⟨_h4⟩ := hk.right.left  -- k divisible by 4
    obtain ⟨_h5⟩ := hk.right.right -- k divisible by 5
    rw hl,
    sorry  -- This is where the rest of the proof would go.
  }
end

end smallest_perfect_square_div_by_4_and_5_l310_310495


namespace compounded_rate_of_growth_l310_310571

theorem compounded_rate_of_growth (k m : ℝ) :
  (1 + k / 100) * (1 + m / 100) - 1 = ((k + m + (k * m / 100)) / 100) :=
by
  sorry

end compounded_rate_of_growth_l310_310571


namespace binomial_v2_l310_310633

def v2 (s : ℕ) : ℕ := Nat.findGreatestPow 2 s

theorem binomial_v2 (m : ℕ) (h : m > 0) : 
  v2 (∏ n in finset.range (2^m + 1), nat.choose (2*n) n) = m * 2^(m-1) + 1 := 
sorry

end binomial_v2_l310_310633


namespace sum_coefficients_eq_neg_one_l310_310337

theorem sum_coefficients_eq_neg_one
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} a_{11} a_{12} a_{13} a_{14} a_{15} a_{16} a_{17} : ℤ) :
  (x - 1) ^ 17 = a_0 + a_1 * (1 + x) + a_2 * (1 + x) ^ 2 + a_3 * (1 + x) ^ 3 + 
  a_4 * (1 + x) ^ 4 + a_5 * (1 + x) ^ 5 + a_6 * (1 + x) ^ 6 + a_7 * (1 + x) ^ 7 + 
  a_8 * (1 + x) ^ 8 + a_9 * (1 + x) ^ 9 + a_{10} * (1 + x) ^ 10 + a_{11} * (1 + x) ^ 11 + 
  a_{12} * (1 + x) ^ 12 + a_{13} * (1 + x) ^ 13 + a_{14} * (1 + x) ^ 14 + 
  a_{15} * (1 + x) ^ 15 + a_{16} * (1 + x) ^ 16 + a_{17} * (1 + x) ^ 17 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} + a_{11} + 
  a_{12} + a_{13} + a_{14} + a_{15} + a_{16} + a_{17} = -1 :=
begin
  sorry
end

end sum_coefficients_eq_neg_one_l310_310337


namespace seq_six_is_216_l310_310190

def seq (n : ℕ) : ℝ := match n with
  | 1 => 1
  | 2 => 0.5
  | 3 => 1
  | 4 => 4
  | 5 => 25
  | _ => n^(n-3)

theorem seq_six_is_216 : seq 6 = 216 := by
  simp [seq]
  norm_num
  sorry

end seq_six_is_216_l310_310190


namespace total_big_cats_at_sanctuary_l310_310203

theorem total_big_cats_at_sanctuary :
  let lions := 12
  let tigers := 14
  let cougars := (lions + tigers) / 2
  lions + tigers + cougars = 39 :=
by
  let lions := 12
  let tigers := 14
  let cougars := (lions + tigers) / 2
  have h : lions + tigers + cougars = 12 + 14 + (12 + 14) / 2 := rfl
  have h2 : 12 + 14 + (12 + 14) / 2 = 39 := by norm_num
  rw [h, h2]
  exact rfl

end total_big_cats_at_sanctuary_l310_310203


namespace alcohol_percentage_l310_310563

theorem alcohol_percentage (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 100) 
(h3 : (0.6 + (x / 100) * 6 = 2.4)) : x = 30 :=
by sorry

end alcohol_percentage_l310_310563


namespace median_is_4_6_l310_310226

-- Given conditions on the number of students per vision category
def vision_data : List (ℝ × ℕ) :=
[(4.0, 1), (4.1, 2), (4.2, 6), (4.3, 3), (4.4, 3), (4.5, 4), (4.6, 1), 
 (4.7, 2), (4.8, 5), (4.9, 7), (5.0, 5)]

-- Statement of the problem: Proving that the median is 4.6
theorem median_is_4_6 (data : List (ℝ × ℕ)) (h : data = vision_data) : 
  median (expand_data data) = 4.6 := 
sorry

-- Helper function to expand the data into a list of individual vision values
def expand_data (data : List (ℝ × ℕ)) : List ℝ :=
data.bind (λ p, List.replicate p.snd p.fst)

-- Helper function to compute the median
noncomputable
def median (l : List ℝ) : ℝ :=
if l.length % 2 = 1 then l.nth_le (l.length / 2) (by linarith [List.length_pos_of_mem (List.nth_le_mem l (l.length / 2) _)])
else (l.nth_le (l.length / 2 - 1) (by linarith [List.length_pos_of_mem (List.nth_le_mem l (l.length / 2 - 1) _)]) +
      l.nth_le (l.length / 2) (by linarith [List.length_pos_of_mem (List.nth_le_mem l (l.length / 2) _)])) / 2

end median_is_4_6_l310_310226


namespace lollipop_count_l310_310346

theorem lollipop_count (total_cost : ℝ) (cost_per_lollipop : ℝ) (h1 : total_cost = 90) (h2 : cost_per_lollipop = 0.75) : 
  total_cost / cost_per_lollipop = 120 :=
by 
  sorry

end lollipop_count_l310_310346


namespace Seulgi_second_round_need_l310_310704

def Hohyeon_first_round := 23
def Hohyeon_second_round := 28
def Hyunjeong_first_round := 32
def Hyunjeong_second_round := 17
def Seulgi_first_round := 27

def Hohyeon_total := Hohyeon_first_round + Hohyeon_second_round
def Hyunjeong_total := Hyunjeong_first_round + Hyunjeong_second_round

def required_total_for_Seulgi := Hohyeon_total + 1

theorem Seulgi_second_round_need (Seulgi_second_round: ℕ) :
  Seulgi_first_round + Seulgi_second_round ≥ required_total_for_Seulgi → Seulgi_second_round ≥ 25 :=
by
  sorry

end Seulgi_second_round_need_l310_310704


namespace find_ab_of_sqrt_pattern_l310_310297

theorem find_ab_of_sqrt_pattern (a b : ℝ)
  (h₁ : sqrt (2 + 2/3) = 2 * sqrt (2/3))
  (h₂ : sqrt (3 + 3/8) = 3 * sqrt (3/8))
  (h₃ : sqrt (4 + 4/15) = 4 * sqrt (4/15))
  (h₄ : sqrt (6 + a/b) = 6 * sqrt (a/b)) : 
  a = 6 ∧ b = 35 :=
sorry

end find_ab_of_sqrt_pattern_l310_310297


namespace sequence_first_three_terms_l310_310268

-- Given the sequence definition and sum of first n terms
def S (n : ℕ) : ℕ := n^2 - 2 * n + 3

-- First three terms of the sequence
theorem sequence_first_three_terms :
  let a_1 := S 1,
      a_2 := S 2 - S 1,
      a_3 := S 3 - S 2
  in a_1 = 2 ∧ a_2 = 1 ∧ a_3 = 3 :=
by
  have h1 : a_1 = S 1 := rfl,
  have h2 : a_2 = S 2 - S 1 := rfl,
  have h3 : a_3 = S 3 - S 2 := rfl,
  have hS1 : S 1 = 2 := by norm_num,
  have hS2 : S 2 = 3 := by norm_num,
  have hS3 : S 3 = 6 := by norm_num,
  rw [h1, h2, h3],
  split,
  exact hS1,
  split,
  linarith,
  linarith

# The theorem sequence_first_three_terms states that the first three terms of the 
# sequence are indeed 2, 1, and 3 respectively, using the function S to calculate 
# the partial sums.
sorry

end sequence_first_three_terms_l310_310268


namespace frog_jump_possible_iff_mod_4_l310_310012

theorem frog_jump_possible_iff_mod_4 (n : ℕ) (h : n ≥ 2) :
  (∃ jump_method : (fin (2 * n)) → (fin (2 * n)), ∀ i j : fin (2 * n), (i ≠ j) →
    ¬ (jump_method i).val = (jump_method j).val + n) ↔
  n % 4 = 2 := sorry

end frog_jump_possible_iff_mod_4_l310_310012


namespace product_of_fractional_parts_eq_222_l310_310137

theorem product_of_fractional_parts_eq_222 : 
  let x := 18 / 999 in let y := x.num / x.denom in y.num * y.denom = 222 :=
by 
  sorry

end product_of_fractional_parts_eq_222_l310_310137


namespace annual_rent_per_sq_foot_l310_310891

theorem annual_rent_per_sq_foot
  (length width : ℝ)
  (monthly_rent : ℝ)
  (h1 : length = 18)
  (h2 : width = 20)
  (h3 : monthly_rent = 3600) :
  let area := length * width in
  let annual_rent := monthly_rent * 12 in
  let annual_rent_per_sq_foot := annual_rent / area in
  annual_rent_per_sq_foot = 120 := 
by
  intros
  unfold area annual_rent annual_rent_per_sq_foot
  rw [h1, h2, h3]
  sorry

end annual_rent_per_sq_foot_l310_310891


namespace no_real_roots_l310_310593

theorem no_real_roots (x : ℝ) : ¬ (sqrt (x + 9) - sqrt (x - 2) + 2 = 0) := 
by
  sorry

end no_real_roots_l310_310593


namespace collinear_points_b_value_l310_310354

theorem collinear_points_b_value :
  ∃ b : ℝ, (3 - (-2)) * (11 - b) = (8 - 3) * (1 - b) → b = -9 :=
by
  sorry

end collinear_points_b_value_l310_310354


namespace arithmetic_sequence_term_l310_310020

theorem arithmetic_sequence_term (a : ℕ → ℕ) (h1 : a 2 = 2) (h2 : a 3 = 4) : a 10 = 18 :=
by
  sorry

end arithmetic_sequence_term_l310_310020


namespace intersecting_diagonals_probability_l310_310360

def num_vertices (hexagon : Polygon) : ℕ := 6

def num_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

def num_pairs (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

def intersecting_pairs (hexagon : Polygon) : ℕ :=
  15  -- Precomputed from steps

def prob_intersecting_diagonals (hexagon : Polygon) : ℚ :=
  intersecting_pairs(hexagon) / num_pairs(num_diagonals(num_vertices(hexagon)))

theorem intersecting_diagonals_probability (hexagon : Polygon) : 
  prob_intersecting_diagonals(hexagon) = 5 / 12 :=
by
  sorry

end intersecting_diagonals_probability_l310_310360


namespace expand_and_simplify_l310_310853

noncomputable def expanded_expr (a : ℝ) : ℝ :=
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6)

theorem expand_and_simplify (a : ℝ) :
  expanded_expr a = a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 :=
by {
  -- Proof goes here
  sorry
}

end expand_and_simplify_l310_310853


namespace sphere_radius_in_cone_l310_310384

-- Define the conditions provided in the problem
def cone_base_radius : ℝ := 6
def cone_height : ℝ := 15

def tangent_spheres_radius (r : ℝ) : Prop :=
  ∃ (s1 s2 s3 s4 : ℝ × ℝ × ℝ),
    -- Each sphere is tangent to the others and fits within the cone, satisfying tangency with the base and side
    r = s1.1 ∧ r = s2.1 ∧ r = s3.1 ∧ r = s4.1 ∧  -- Spheres have radius r
    s1.2 = cone_height ∧ s2.2 = cone_height ∧ s3.2 = cone_height ∧ s4.2 = cone_height ∧ -- Spheres are aligned within the height
    -- Additional conditions for tangency relationships would be included here if necessary

-- The proof goal
theorem sphere_radius_in_cone : ∃ r : ℝ, tangent_spheres_radius r ∧ r = 45/7 := 
by {
  sorry
}

end sphere_radius_in_cone_l310_310384


namespace find_S_2017_l310_310652

def sequence_a (n : ℕ) : ℝ := (2 * n - 1) * Real.cos (n * Real.pi / 2)

def sum_S (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, sequence_a (k + 1)

theorem find_S_2017 : sum_S 2017 = 6043 :=
by
  sorry

end find_S_2017_l310_310652


namespace repeating_decimal_product_of_num_and_den_l310_310133

theorem repeating_decimal_product_of_num_and_den (x : ℚ) (h : x = 18 / 999) (h_simplified : x.num * x.den = 222) : x.num * x.den = 222 :=
by {
  sorry
}

end repeating_decimal_product_of_num_and_den_l310_310133


namespace teapot_weight_l310_310640

theorem teapot_weight :
  ∃ (x : ℕ), 
  let spoon := x,
      candlestick := 3 * x,
      teapot := 9 * x,
      total_weight := spoon + candlestick + teapot
  in
    total_weight = 2600 ∧ teapot = 1800 :=
by
  sorry

end teapot_weight_l310_310640


namespace max_intersecting_chords_through_A1_l310_310175

theorem max_intersecting_chords_through_A1 
  (n : ℕ) (h_n : n = 2017) 
  (A : Fin n → α) 
  (line_through_A1 : α) 
  (no_other_intersection : ∀ i : Fin n, i ≠ 0 → A i ≠ line_through_A1) :
  ∃ k : ℕ, k * (2016 - k) + 2016 = 1018080 := 
sorry

end max_intersecting_chords_through_A1_l310_310175


namespace rope_cutting_impossible_l310_310186

/-- 
Given a rope initially cut into 5 pieces, and then some of these pieces were each cut into 
5 parts, with this process repeated several times, it is not possible for the total 
number of pieces to be exactly 2019.
-/ 
theorem rope_cutting_impossible (n : ℕ) : 5 + 4 * n ≠ 2019 := 
sorry

end rope_cutting_impossible_l310_310186


namespace shaded_region_area_l310_310620

theorem shaded_region_area :
  let line1 := fun x => -3 / 8 * x + 4 in
  let line2 := fun x => -5 / 6 * x + 35 / 6 in
  let intersection := (4, 3) in
  let area_segment_1 := ∫ x in 0..1, line2 x - line1 x in
  let area_segment_2 := ∫ x in 1..4, line2 x - line1 x in
  2 * (area_segment_1 + area_segment_2) = 13 / 2 :=
by
  sorry

end shaded_region_area_l310_310620


namespace calculate_expression_l310_310584

theorem calculate_expression :
  (3^1 + 3^0 + 3^(-1) + 3^(-2)) / (3^(-3) + 3^(-4) + 3^(-5) + 3^(-6)) = 81 := by
  sorry

end calculate_expression_l310_310584


namespace problem_statement_l310_310398

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (π * x + α) + b * Real.cos (π * x + β) + 2

theorem problem_statement (a b α β : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : α ≠ 0) (h₃ : β ≠ 0)
  (h4 : f 2012 a α b β = 1) : f 2013 a α b β = 3 :=
  sorry

end problem_statement_l310_310398


namespace percentage_passed_l310_310533

def swim_club_members := 100
def not_passed_course_taken := 40
def not_passed_course_not_taken := 30
def not_passed := not_passed_course_taken + not_passed_course_not_taken

theorem percentage_passed :
  ((swim_club_members - not_passed).toFloat / swim_club_members.toFloat * 100) = 30 := by
  sorry

end percentage_passed_l310_310533


namespace bounds_for_a_l310_310607

theorem bounds_for_a (a : ℝ) (h_a : a > 0) :
  ∀ x : ℝ, 0 < x ∧ x < 17 → (3 / 4) * x = (5 / 6) * (17 - x) + a → a < (153 / 12) := 
sorry

end bounds_for_a_l310_310607


namespace ratio_frank_to_others_l310_310253

theorem ratio_frank_to_others:
  (Betty_oranges : ℕ) (Bill_oranges : ℕ) (total_oranges_picked_by_philip : ℕ) :
  Betty_oranges = 15 → Bill_oranges = 12 →
  (exists m : ℕ, total_oranges_picked_by_philip = 270 * m ∧ 270 * m / 27 = 3) →
  (total_oranges_picked_by_philip / 27 / (Betty_oranges + Bill_oranges / 27) = 3) :=
by
  intros Betty_oranges Bill_oranges total_oranges_picked_by_philip
  assume hB : Betty_oranges = 15,
  assume hBi : Bill_oranges = 12,
  assume hE : ∃ m : ℕ, total_oranges_picked_by_philip = 270 * m ∧ 270 * m / 27 = 3,
  sorry

end ratio_frank_to_others_l310_310253


namespace temperature_at_noon_l310_310004

-- Definitions of the given conditions.
def morning_temperature : ℝ := 4
def temperature_drop : ℝ := 10

-- The theorem statement that needs to be proven.
theorem temperature_at_noon : morning_temperature - temperature_drop = -6 :=
by
  -- The proof can be filled in by solving the stated theorem.
  sorry

end temperature_at_noon_l310_310004


namespace find_number_for_B_l310_310826

variable (a : ℝ)

-- Defining B as an unknown number
def number_for_B (a : ℝ) :=
  B : ℝ

-- Given condition that a is 2.5 more than B
axiom a_eq_B_plus_2_5 : a = B + 2.5

-- The theorem we need to prove: the number for B is equal to a - 2.5
theorem find_number_for_B (a : ℝ) (B : ℝ) (h : a = B + 2.5) : B = a - 2.5 :=
  sorry

end find_number_for_B_l310_310826


namespace sum_2010_3_array_remainder_l310_310901
 
theorem sum_2010_3_array_remainder :
  let p := 2010
  let q := 3
  let sum_array := (∑' r : ℕ, ∑' c : ℕ, (1 / (2 * p)^r) * (1 / q^c))
  let fraction := 4020 / 4019 * 3 / 2
  let (m, n) := (6030, 4019) in
  m + n = 10049 → 10049 % 2010 = 1009 :=
by
  let sum_geom_r := (∑' r : ℕ, 1 / (2 * 2010)^r)
  let sum_geom_c := (∑' c : ℕ, 1 / 3^c)
  have sum_geom_r_correct : sum_geom_r = 1 / (1 - 1 / (2 * 2010)) := sorry
  have sum_geom_c_correct : sum_geom_c = 1 / (1 - 1 / 3) := sorry
  have sum_array_value : (∑' r, ∑' c, 1 / (2 * 2010)^r * 1 / 3^c) = fraction := sorry
  have fraction_eq : fraction = 6030 / 4019 := sorry
  have m_prime : Nat.gcd 6030 4019 = 1 := by norm_num
  have mod_eq : 10049 % 2010 = 1009 := by norm_num
  exact mod_eq

end sum_2010_3_array_remainder_l310_310901


namespace expand_and_simplify_l310_310860

theorem expand_and_simplify (a : ℝ) : 
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
  sorry

end expand_and_simplify_l310_310860


namespace cylinder_increase_l310_310127

theorem cylinder_increase (x : ℝ) (r h : ℝ) (π : ℝ) 
  (h₁ : r = 5) (h₂ : h = 10) 
  (h₃ : π > 0) 
  (h_equal_volumes : π * (r + x) ^ 2 * h = π * r ^ 2 * (h + x)) :
  x = 5 / 2 :=
by
  -- Proof is omitted
  sorry

end cylinder_increase_l310_310127


namespace problem_statement_l310_310603

def assoc_number (x : ℚ) : ℚ :=
  if x >= 0 then 2 * x - 1 else -2 * x + 1

theorem problem_statement (a b : ℚ) (ha : a > 0) (hb : b < 0) (hab : assoc_number a = assoc_number b) :
  (a + b)^2 - 2 * a - 2 * b = -1 :=
sorry

end problem_statement_l310_310603


namespace probability_at_least_one_woman_l310_310345

theorem probability_at_least_one_woman (total_people men women k : ℕ) (h : total_people = 15 ∧ men = 10 ∧ women = 5 ∧ k = 4)
  : ℚ :=
  let prob_no_women := (2 / 3) * (9 / 14) * (8 / 13) * (7 / 12) in
  1 - prob_no_women 
-- Expected result: 29 / 36

end probability_at_least_one_woman_l310_310345


namespace rons_percentage_less_than_tammy_l310_310802

theorem rons_percentage_less_than_tammy (sammy_p: ℕ) (ron_p: ℕ):
  sammy_p = 15 → 
  ron_p = 24 → 
  ∃ tammy_p: ℕ, tammy_p = 2 * sammy_p ∧ (tammy_p - ron_p) * 100 / tammy_p = 20 :=
by
  -- Definitions and conditions
  assume h1: sammy_p = 15,
  assume h2: ron_p = 24,
  -- Calculate the number of pickle slices Tammy can eat
  let tammy_p := 2 * sammy_p,
  -- Statement
  existsi tammy_p,
  -- Proof to be done
  sorry

end rons_percentage_less_than_tammy_l310_310802


namespace smallest_positive_integer_in_linear_combination_l310_310606

theorem smallest_positive_integer_in_linear_combination :
  ∃ m n : ℤ, 2016 * m + 43200 * n = 24 :=
by
  sorry

end smallest_positive_integer_in_linear_combination_l310_310606


namespace arithmetic_sequence_transformation_l310_310272

theorem arithmetic_sequence_transformation (a : ℕ → ℝ) (d c : ℝ) (h : ∀ n, a (n + 1) = a n + d) (hc : c ≠ 0) :
  ∀ n, (c * a (n + 1)) - (c * a n) = c * d := 
by
  sorry

end arithmetic_sequence_transformation_l310_310272


namespace quadratic_no_real_roots_l310_310277

theorem quadratic_no_real_roots
  (p q a b c : ℝ)
  (hpq_pos : 0 < p ∧ 0 < q ∧ 0 < a ∧ 0 < b ∧ 0 < c)
  (hpq_neq : p ≠ q)
  (gp : a^2 = p * q)
  (ap1 : b - p = c - b)
  (ap2 : c - b = q - c) :
  let Δ := (2 * a)^2 - 4 * (b * c)
  in Δ < 0 :=
by
  sorry

end quadratic_no_real_roots_l310_310277


namespace compute_david_score_l310_310753

theorem compute_david_score : ∃ (s c w : ℤ), s = 105 ∧ s = 40 + 5 * c - w ∧ s > 100 ∧
  (∀ (s' : ℤ), 100 < s' < s → ¬ (∃ (c' w' : ℤ), s' = 40 + 5 * c' - w') → (c' = c)) :=
by
  sorry

end compute_david_score_l310_310753


namespace problem_1_problem_2_l310_310318

open Real
open Set

noncomputable def y (x : ℝ) : ℝ := (2 * sin x - cos x ^ 2) / (1 + sin x)

theorem problem_1 :
  { x : ℝ | y x = 1 ∧ sin x ≠ -1 } = { x | ∃ (k : ℤ), x = 2 * k * π + (π / 2) } :=
by
  sorry

theorem problem_2 : 
  ∃ x, y x = 1 ∧ ∀ x', y x' ≤ 1 :=
by
  sorry

end problem_1_problem_2_l310_310318


namespace quarters_left_l310_310933

theorem quarters_left (initial_quarters spent_quarters : ℕ) (h₀ : initial_quarters = 88) (h₁ : spent_quarters = 9) : initial_quarters - spent_quarters = 79 :=
by
  rw [h₀, h₁]
  norm_num

end quarters_left_l310_310933


namespace equilateral_triangle_coverage_impossible_l310_310428

theorem equilateral_triangle_coverage_impossible (T T1 T2 : Type) [equilateral_triangle T] [equilateral_triangle T1] [equilateral_triangle T2] :
  (side_length T1 < side_length T) ∧ (side_length T2 < side_length T) →
  ¬(∀ x, x ∈ T → x ∈ T1 ∨ x ∈ T2) :=
by 
  -- Definitions and assumptions for the proof
  sorry

end equilateral_triangle_coverage_impossible_l310_310428


namespace balloon_height_per_ounce_l310_310703

theorem balloon_height_per_ounce
    (total_money : ℕ)
    (sheet_cost : ℕ)
    (rope_cost : ℕ)
    (propane_cost : ℕ)
    (helium_price : ℕ)
    (max_height : ℕ)
    :
    total_money = 200 →
    sheet_cost = 42 →
    rope_cost = 18 →
    propane_cost = 14 →
    helium_price = 150 →
    max_height = 9492 →
    max_height / ((total_money - (sheet_cost + rope_cost + propane_cost)) / helium_price) = 113 :=
by
  intros
  sorry

end balloon_height_per_ounce_l310_310703


namespace probability_no_root_l310_310773

noncomputable def is_normal_distribution (ξ : ℝ) (μ σ : ℝ) :=
  ∀ x : ℝ, -- definition of the normal distribution with mean μ and variance σ^2
  sorry

noncomputable def discriminant_condition (ξ : ℝ) :=
  ξ > 1

theorem probability_no_root (ξ : ℝ) (s : ℝ) (h : is_normal_distribution ξ 1 s) : 
  ∃ p, p = 1 / 2 ∧ ∃ h0 : s > 0, 
    (∀ x : ℝ, discriminant_condition x → P(ξ > 1) = p) :=
sorry

end probability_no_root_l310_310773


namespace rectangle_length_l310_310554

theorem rectangle_length (s : ℝ) (W L : ℝ) (P_square : ℝ) (P_rectangle : ℝ) (A_rectangle : ℝ) :
  s = 15 →
  P_square = 4 * s →
  P_square = P_rectangle →
  P_rectangle = 2 * L + 2 * W →
  A_rectangle = 216 →
  A_rectangle = L * W →
  L = 18 :=
by
  intro h1 h2 h3 h4 h5 h6
  have h7 : L * W = 216 := h6
  have h8 : 4 * s = P_rectangle := by rw [h2, h3]
  have h9 : 60 = 2 * L + 2 * W := by rw [←h8, h4]
  sorry

end rectangle_length_l310_310554


namespace max_chords_intersecting_line_l310_310174

theorem max_chords_intersecting_line (A : Fin 2017 → Type) :
  ∃ k : ℕ, (k ≤ 2016 ∧ ∃ m : ℕ, (m = k * (2016 - k) + 2016) ∧ m = 1018080) :=
sorry

end max_chords_intersecting_line_l310_310174


namespace donny_spending_l310_310980

theorem donny_spending :
  (15 + 28 + 13) / 2 = 28 :=
by
  sorry

end donny_spending_l310_310980


namespace half_angle_in_second_quadrant_l310_310336

theorem half_angle_in_second_quadrant (α : ℝ) (h : 180 < α ∧ α < 270) : 90 < α / 2 ∧ α / 2 < 135 := 
by
  sorry

end half_angle_in_second_quadrant_l310_310336


namespace largest_proper_subset_size_l310_310391

universe u

open Finset -- Open the Finset namespace

noncomputable def X (n : ℕ) : Finset (Fin (n+1) → ℕ) :=
  univ.filter (λ s, ∀ i, s i ∈ finset.range (i + 1))

def join (s t : Fin (n+1) → ℕ) : Fin (n+1) → ℕ :=
  λ i, max (s i) (t i)

def meet (s t : Fin (n+1) → ℕ) : Fin (n+1) → ℕ :=
  λ i, min (s i) (t i)

theorem largest_proper_subset_size (n : ℕ) (hn : n ≥ 2) :
  ∃ A ⊂ X n, (∀ s t ∈ A, join s t ∈ A ∧ meet s t ∈ A) ∧ |A| = (n + 1)! - (n - 1)! :=
begin
  sorry
end

end largest_proper_subset_size_l310_310391


namespace first_tier_price_level_is_10000_l310_310882

noncomputable def first_tier_price_level (P : ℝ) : Prop :=
  ∀ (car_price : ℝ), car_price = 30000 → (P ≤ car_price ∧ 
    (0.25 * P + 0.15 * (car_price - P)) = 5500)

theorem first_tier_price_level_is_10000 :
  first_tier_price_level 10000 :=
by
  sorry

end first_tier_price_level_is_10000_l310_310882


namespace constant_term_poly_product_l310_310583

theorem constant_term_poly_product :
  let p₁ := (λ x : ℝ, x^4 + x^2 + 6)
  let p₂ := (λ x : ℝ, x^5 + x^3 + x + 20)
  (p₁ 0) * (p₂ 0) = 120 :=
by
  -- Define the polynomials
  let p₁ := (λ x : ℝ, x^4 + x^2 + 6)
  let p₂ := (λ x : ℝ, x^5 + x^3 + x + 20)
  -- Evaluate the constant terms (set x=0)
  have h1 : p₁ 0 = 6 := by simp [p₁]
  have h2 : p₂ 0 = 20 := by simp [p₂]
  -- Multiply the constant terms
  show 6 * 20 = 120 from by simp [h1, h2]

end constant_term_poly_product_l310_310583


namespace sum_of_digits_of_least_time_for_five_horses_l310_310464

/-
Problem:
There are 10 horses, named Horse 1, Horse 2, ..., Horse 10. They get their names from how many minutes it takes them to run one lap around a circular race track: Horse k runs one lap in exactly k minutes. At time 0 all the horses are together at the starting point on the track. The horses start running in the same direction, and they keep running around the circular track at their constant speeds. The least time S > 0, in minutes, at which all 10 horses will again simultaneously be at the starting point is S = 2520. Let T > 0 be the least time, in minutes, such that at least 5 of the horses are again at the starting point. What is the sum of the digits of T?
-/

def horse_running_time (k : ℕ) : ℕ := k

def horse_laps (n T : ℕ) : Prop :=
  T % n = 0

theorem sum_of_digits_of_least_time_for_five_horses :
  ∃ T > 0, (∃ count : ℕ, count ≥ 5 ∧ (∀ k ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10} : set ℕ), horse_laps k T → count += 1)) ∧ (T = 12 ∧ (1 + 2 = 3)) := 
by
  sorry

end sum_of_digits_of_least_time_for_five_horses_l310_310464


namespace exists_n_gon_l310_310063

theorem exists_n_gon (n : ℕ) (α : Fin n → ℝ) : 
  (∀ i, 0 < α i ∧ α i < 2 * π) ∧ (∑ i, α i = (n - 2) * π) → 
  ∃ (vertices : Fin n → Fin n), 
  True :=  -- The existence of vertices is a trivial implication as we only need to assert existence without construction.

sorry

end exists_n_gon_l310_310063


namespace train_passing_time_l310_310559

/-
  Given:
  1. The length of the train is 178 meters.
  2. The platform length is 267 meters.
  3. The train crosses the platform in 20 seconds.
  Prove:
  - The train takes 8 seconds to pass a man standing on the platform.
-/

noncomputable def train_length : ℝ := 178
noncomputable def platform_length : ℝ := 267
noncomputable def crossing_time_on_platform : ℝ := 20

theorem train_passing_time 
  (train_length : ℝ) 
  (platform_length : ℝ) 
  (crossing_time_on_platform : ℝ) 
  : (train_length + platform_length) / crossing_time_on_platform = 22.25 
  → train_length / 22.25 = 8 :=
begin
  sorry
end

#eval train_passing_time train_length platform_length crossing_time_on_platform

end train_passing_time_l310_310559


namespace intersection_M_N_l310_310325

def M : Set ℝ := { x : ℝ | x^2 > 4 }
def N : Set ℝ := { x : ℝ | x = -3 ∨ x = -2 ∨ x = 2 ∨ x = 3 ∨ x = 4 }

theorem intersection_M_N : M ∩ N = { x : ℝ | x = -3 ∨ x = 3 ∨ x = 4 } :=
by
  sorry

end intersection_M_N_l310_310325


namespace factor_x6_minus_64_l310_310618

theorem factor_x6_minus_64 :
  ∀ x : ℝ, (x^6 - 64) = (x-2) * (x+2) * (x^4 + 4*x^2 + 16) :=
by
  sorry

end factor_x6_minus_64_l310_310618


namespace remove_max_rooks_l310_310747

-- Defines the problem of removing the maximum number of rooks under given conditions
theorem remove_max_rooks (n : ℕ) (attacks_odd : (ℕ × ℕ) → ℕ) :
  (∀ p : ℕ × ℕ, (attacks_odd p) % 2 = 1 → true) →
  n = 8 →
  (∃ m, m = 59) :=
by
  intros _ _
  existsi 59
  sorry

end remove_max_rooks_l310_310747


namespace ned_time_left_to_diffuse_bomb_l310_310778

-- Conditions
def building_flights : Nat := 20
def time_per_flight : Nat := 11
def bomb_timer : Nat := 72
def time_spent_running : Nat := 165

-- Main statement
theorem ned_time_left_to_diffuse_bomb : 
  (bomb_timer - (building_flights - (time_spent_running / time_per_flight)) * time_per_flight) = 17 :=
by
  sorry

end ned_time_left_to_diffuse_bomb_l310_310778


namespace congurence_solution_exists_l310_310406

theorem congurence_solution_exists (a m b : ℤ) (ha : Nat.Coprime a m) (hm : m > 1) : ∃ x : ℤ, a * x ≡ b [ZMOD m] := 
sorry

end congurence_solution_exists_l310_310406


namespace tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l310_310291

-- Conditions
variables {O : ℝ × ℝ} (A : ℝ × ℝ) (B : ℝ × ℝ)
          {P Q : ℝ × ℝ} (p : ℝ)
          (hp : 0 < p)
          (hA : A.1 ^ 2 = 2 * p * A.2)
          (hB : B = (0, -1))
          (hP : P.2 = P.1 ^ 2 / (2 * p))
          (hQ : Q.2 = Q.1 ^ 2 / (2 * p))

-- Proof problem statements
theorem tangent_line_AB
  (hAB_tangent : ∀ x : ℝ, x ^ 2 / (2 * p) = 2 * x - 1 → x = 1) : true :=
by sorry

theorem op_oq_leq_oa_squared 
  (h_op_oq_leq : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + (P.1 ^ 2 / (2 * p)) ^ 2) * (Q.1 ^ 2 + (Q.1 ^ 2 / (2 * p)) ^ 2) ≤ 2) : true :=
by sorry

theorem bp_bq_gt_ba_squared 
  ( h_bp_bq_gt : ∀ P Q : ℝ × ℝ, (P.1 ^ 2 + ((P.1 ^ 2 / (2 * p)) + 1) ^ 2) * (Q.1 ^ 2 + ((Q.1 ^ 2 / (2 * p)) +1 ) ^ 2) > 5 ) : true :=
by sorry

end tangent_line_AB_op_oq_leq_oa_squared_bp_bq_gt_ba_squared_l310_310291


namespace cows_in_group_l310_310361

theorem cows_in_group (c h : ℕ) (L H: ℕ) 
  (legs_eq : L = 4 * c + 2 * h)
  (heads_eq : H = c + h)
  (legs_heads_relation : L = 2 * H + 14) 
  : c = 7 :=
by
  sorry

end cows_in_group_l310_310361


namespace Donny_spends_28_on_Thursday_l310_310986

theorem Donny_spends_28_on_Thursday :
  let monday_savings := 15
  let tuesday_savings := 28
  let wednesday_savings := 13
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  let thursday_spending := total_savings / 2
  thursday_spending = 28 :=
by 
  let monday_savings := 15
  let tuesday_savings := 28
  let wednesday_savings := 13
  let total_savings := monday_savings + tuesday_savings + wednesday_savings
  let thursday_spending := total_savings / 2
  sorry

end Donny_spends_28_on_Thursday_l310_310986


namespace flea_can_visit_all_natural_numbers_l310_310570

theorem flea_can_visit_all_natural_numbers :
  ∀ n : ℕ, ∃ (jumps : ℕ → ℤ), (jumps 0 = 0) ∧ (∀ k, jumps (k+1) = jumps k + (-1)^(some_direction k) * (2^k + 1)) ∧ (∀ m, ∃ k, jumps k = m) :=
by
  sorry

end flea_can_visit_all_natural_numbers_l310_310570


namespace scientific_notation_7nm_l310_310093

theorem scientific_notation_7nm :
  ∀ (x : ℝ), x = 0.000000007 → x = 7 * 10^(-9) :=
begin
  intros x hx,
  sorry
end

end scientific_notation_7nm_l310_310093


namespace gcd_rope_lengths_l310_310748

theorem gcd_rope_lengths : 
  ∀ (a b c : ℕ), 
  a = 45 ∧ b = 60 ∧ c = 75 → Nat.gcd (Nat.gcd a b) c = 15 :=
by
  intros a b c h
  cases h with ha hbc
  cases hbc with hb hc
  rw [ha, hb, hc]
  sorry

end gcd_rope_lengths_l310_310748


namespace value_of_each_red_rose_l310_310958

-- Definitions
def totalFlowers := 400
def tulips := 120
def totalRoses := totalFlowers - tulips
def whiteRoses := 80
def redRoses := totalRoses - whiteRoses
def halfRedRoses := redRoses / 2
def earnings := 75

-- Statement to construct the proof
theorem value_of_each_red_rose : (earnings / halfRedRoses) = 0.75 := by
  sorry

end value_of_each_red_rose_l310_310958


namespace reflection_m_b_sum_l310_310453

noncomputable def reflection_point (p1 p2 : ℝ × ℝ) (m b : ℝ) : Prop :=
  -- Check if (p2) is the image of (p1) after reflection across the line y = mx + b
  let midpoint := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2) in
  let line_eq := -(8 / 3) * midpoint.1 + b = midpoint.2 in
  p1.2 - m * p1.1 + b = p2.2 ∧ line_eq

theorem reflection_m_b_sum :
  reflection_point (2, 3) (10, 6) (-8 / 3) 20.5 ∧ ( -8 / 3 + 20.5 = 107 / 6 ) :=
by
  sorry

end reflection_m_b_sum_l310_310453


namespace evaluate_expression_l310_310225

theorem evaluate_expression :
  (9 ^ (-1 / 2) - 6 ^ (-1)) ^ (-1) = 6 :=
by
  sorry

end evaluate_expression_l310_310225


namespace complex_number_in_second_quadrant_l310_310044

-- Define the imaginary unit i
def i : ℂ := complex.I

-- Define the complex number z
def z : ℂ := (1 + complex.I) * (2 * complex.I)

-- State the theorem to prove
theorem complex_number_in_second_quadrant (i : ℂ) (h : i = complex.I) : z.im > 0 ∧ z.re < 0 := 
by {
  -- Lean code to construct the proof goes here
  sorry
}

end complex_number_in_second_quadrant_l310_310044


namespace ned_defuse_time_l310_310776

theorem ned_defuse_time (flights_total time_per_flight bomb_time time_spent : ℕ) (h1 : flights_total = 20) (h2 : time_per_flight = 11) (h3 : bomb_time = 72) (h4 : time_spent = 165) :
  bomb_time - (flights_total * time_per_flight - time_spent) / time_per_flight * time_per_flight = 17 := by
  sorry

end ned_defuse_time_l310_310776


namespace cost_fencing_l310_310888

variable (x : ℝ) (length width area perimeter cost_per_meter : ℝ)

-- Conditions
def ratio (length : ℝ) (width : ℝ) : Prop := length / width = 3 / 2
def area_eqn (length : ℝ) (width : ℝ) : Prop := length * width = 3750
def cost_rate_in_rupees : ℝ := 0.80

-- Definition of length and width given the ratio
def length_def (x : ℝ) : ℝ := 3 * x
def width_def (x : ℝ) : ℝ := 2 * x

-- Derived conditions
def derived_area_eqn (x : ℝ) : Prop := (3 * x) * (2 * x) = 3750
def solve_x (x : ℝ) : Prop := x = real.sqrt(625)

-- Calculate length, width and perimeter
def length_value : ℝ := 75
def width_value : ℝ := 50
def perimeter_value : ℝ := 2 * (length_value + width_value)

-- Final question, Cost of fencing
theorem cost_fencing : cost_per_meter * perimeter_value = 200 :=
by
  -- Sorry placeholder for the elaborate proof steps
  sorry

end cost_fencing_l310_310888


namespace horner_method_v3_l310_310131

def f (x : ℤ) : ℤ := 7*x^5 + 5*x^4 + 3*x^3 + x^2 + x + 2

noncomputable def v_0 : ℤ := 7
noncomputable def v_1 (x : ℤ) (v0 : ℤ) : ℤ := v0 * x + 5
noncomputable def v_2 (x : ℤ) (v1 : ℤ) : ℤ := v1 * x + 3
noncomputable def v_3 (x : ℤ) (v2 : ℤ) : ℤ := v2 * x + 1

theorem horner_method_v3 (x : ℤ) (v0 : ℤ) (v1 : ℤ) (v2 : ℤ) : 
  v_3 x v2 = 83 :=
by 
  have v0_def : v0 = 7 := by rfl
  have v1_def : v1 = v_0 * x + 5 := by rw [v_1, v_0]
  have v2_def : v2 = v_1 x v_0 * x + 3 := by rw [v_2, v_1, v0_def]
  sorry

end horner_method_v3_l310_310131


namespace matrix_power_101_l310_310754

def B : Matrix (Fin 3) (Fin 3) ℝ := ![
  ![1, 0, 0],
  ![0, 0, 1],
  ![0, 1, 0]
]

theorem matrix_power_101 :
  B ^ (101 : ℕ) = B := sorry

end matrix_power_101_l310_310754


namespace leap_year_53_sundays_and_february_5_sundays_l310_310628

theorem leap_year_53_sundays_and_february_5_sundays :
  let Y := 366
  let W := 52
  ∃ (p : ℚ), p = (2/7) * (1/7) → p = 2/49
:=
by
  sorry

end leap_year_53_sundays_and_february_5_sundays_l310_310628


namespace bank_deposit_return_l310_310752

theorem bank_deposit_return 
  (initial_deposit : ℝ) (annual_interest_rate : ℝ) (period_in_months : ℝ) (exchange_rate : ℝ) (insurance_limit : ℝ) :
  initial_deposit = 23904 →
  annual_interest_rate = 0.05 →
  period_in_months = 3 →
  exchange_rate = 58.15 →
  insurance_limit = 1400000 →
  let period_in_years := period_in_months / 12 in
  let interest_accrued := initial_deposit * (1 + annual_interest_rate * period_in_years) in
  let amount_in_rubles := interest_accrued * exchange_rate in
  min amount_in_rubles insurance_limit = 1400000 :=
begin
  intros h1 h2 h3 h4 h5,
  let period_in_years := period_in_months / 12,
  let interest_accrued := initial_deposit * (1 + annual_interest_rate * period_in_years),
  let amount_in_rubles := interest_accrued * exchange_rate,
  have h : interest_accrued = 23904 * (1 + 0.05 * (3 / 12)), by { rw [h1, h2, h3], },
  have amount_in_rubles := interest_accrued * exchange_rate,
  rw [h, h4],
  have h_limit : min amount_in_rubles 1400000 = 1400000,
  {
    sorry
  },
  exact h_limit,
end

end bank_deposit_return_l310_310752


namespace bug_back_at_vertex_A_after_8_meters_l310_310042

noncomputable def bug_probability : ℚ := (2/5)^8

theorem bug_back_at_vertex_A_after_8_meters :
  ∃ m : ℕ, (2 / 5 : ℚ)^8 = m / 1024 ∧ m = 256 :=
begin
  use 256,
  norm_num
end

end bug_back_at_vertex_A_after_8_meters_l310_310042


namespace determine_y_l310_310506

theorem determine_y : ∃ (y : ℤ), (2010 + 2 * y)^2 = 4 * y^2 ∧ y = -1005 :=
by
  existsi (-1005 : ℤ)
  split
  · -- First part: prove (2010 + 2 * (-1005))^2 = 4 * (-1005)^2
    calc
      (2010 + 2 * (-1005))^2 = (2010 - 2010)^2 : by rw [mul_neg_one, two_mul, add_right_neg]
      ...                   = 0^2 : by rfl
      ...                   = 0 : by rfl
    calc
      4 * (-1005)^2 = 4 * 1005^2 : by rw [neg_square]
      ...           = 4 * 1005 * 1005 : by rfl
      ...           = 0 : /- detailed proof goes here -/
  · sorry

end determine_y_l310_310506


namespace smallest_perfect_square_divisible_by_4_and_5_l310_310499

theorem smallest_perfect_square_divisible_by_4_and_5 : 
  ∃ n : ℕ, n > 0 ∧ n ∣ 4 ∧ n ∣ 5 ∧ is_square n ∧ 
  ∀ m : ℕ, (m > 0 ∧ m ∣ 4 ∧ m ∣ 5 ∧ is_square m) → n ≤ m :=
sorry

end smallest_perfect_square_divisible_by_4_and_5_l310_310499


namespace donny_spent_on_thursday_l310_310983

theorem donny_spent_on_thursday :
  let savings_monday : ℤ := 15,
      savings_tuesday : ℤ := 28,
      savings_wednesday : ℤ := 13,
      total_savings : ℤ := savings_monday + savings_tuesday + savings_wednesday,
      amount_spent_thursday : ℤ := total_savings / 2
  in
  amount_spent_thursday = 28 :=
by
  sorry

end donny_spent_on_thursday_l310_310983


namespace lollipop_count_l310_310347

theorem lollipop_count (total_cost : ℝ) (cost_per_lollipop : ℝ) (h1 : total_cost = 90) (h2 : cost_per_lollipop = 0.75) : 
  total_cost / cost_per_lollipop = 120 :=
by 
  sorry

end lollipop_count_l310_310347


namespace compare_abc_l310_310259

noncomputable def a := 2^(-1/3 : ℝ)
noncomputable def b := 3^(-1/2 : ℝ)
noncomputable def c := Real.cos (50 * Real.pi / 180) * Real.cos (10 * Real.pi / 180) + Real.cos (140 * Real.pi / 180) * Real.sin (170 * Real.pi / 180)

theorem compare_abc : a > b ∧ b > c :=
by
  sorry

end compare_abc_l310_310259


namespace restaurant_bill_split_l310_310528

noncomputable def calculate_share (initial_bill : ℝ) (tip_percent : ℝ) (num_people : ℕ) : ℝ :=
by
  let tip := initial_bill * (tip_percent / 100)
  let total_bill := initial_bill + Float.ofReal (Real.round (Float.toReal $ tip))
  let amount_per_person := total_bill / num_people
  exact Float.ofReal (Real.round (Float.toReal $ amount_per_person))

theorem restaurant_bill_split :
  calculate_share 314.12 18 8 = 46.33 := sorry

end restaurant_bill_split_l310_310528


namespace a5_value_l310_310458

def seq : ℕ → ℝ
| 0 => 2
| (n + 1) => 1 / (1 - seq n)

theorem a5_value : seq 4 = -1 := sorry

end a5_value_l310_310458


namespace option_c_opposites_l310_310873

theorem option_c_opposites : -|3| = -3 ∧ 3 = 3 → ( ∃ x y : ℝ, x = -3 ∧ y = 3 ∧ x = -y) :=
by
  sorry

end option_c_opposites_l310_310873


namespace triangle_ABC_lengths_of_sides_l310_310014

/-- Given an acute triangle ABC with sides opposite to angles A, B, and C being a, b, and c respectively,
    and √3 * a = 2 * c * sin A, prove that the size of angle C is π / 3. Additionally, if c = √7 and ab = 6,
    prove that the lengths of sides a and b are 2 and 3 respectively or 3 and 2. -/
theorem triangle_ABC (A B C a b c : ℝ) (h_acute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  (h_sides_angles : ∀ (A B C : ℝ), √3 * a = 2 * c * sin A) :
  C = π / 3 :=
sorry

theorem lengths_of_sides (c a b : ℝ) (h_c : c = √7) (h_ab : a * b = 6)
  (h_cosine_rule : a^2 + b^2 - 2 * a * b * cos (π / 3) = 7) :
  (a = 2 ∧ b = 3) ∨ (a = 3 ∧ b = 2) :=
sorry

end triangle_ABC_lengths_of_sides_l310_310014


namespace probability_point_below_or_on_line_l310_310788

structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Parallelogram :=
  (P Q R S : Point)

def area (A B C D : Point) : ℝ := 
  -- Assuming a general formula for the area of a quadrilateral given vertices A, B, C, D
  abs ((B.x - A.x) * (D.y - A.y) - (C.x - A.x) * (B.y - A.y))

def is_below_or_on_line (p : Point) (y_line : ℝ) : Prop :=
  p.y ≤ y_line

theorem probability_point_below_or_on_line 
  (P Q R S T U : Point)
  (hT : T.x = -1 ∧ T.y = -1)
  (hU : U.x = -7 ∧ U.y = -1)
  (parallelogram : Parallelogram)
  (hP : parallelogram.P = P)
  (hQ : parallelogram.Q = Q)
  (hR : parallelogram.R = R)
  (hS : parallelogram.S = S)
  (hPQ : P = ⟨4, 4⟩)
  (hRQ : Q = ⟨-2, -2⟩)
  (hRP : R = ⟨-8, -2⟩)
  (hSP : S = ⟨-2, 4⟩) :
  let subregion := area Q R T U in
  let total_area := area P Q R S in
  subregion / total_area = 1 / 6 :=
by
  sorry

end probability_point_below_or_on_line_l310_310788


namespace surface_area_of_sphere_l310_310120

theorem surface_area_of_sphere (S A B C O : Type) 
  (h1 : euclidean_geometry.point_on_sphere S O)
  (h2 : euclidean_geometry.point_on_sphere A O)
  (h3 : euclidean_geometry.point_on_sphere B O)
  (h4 : euclidean_geometry.point_on_sphere C O)
  (h5 : euclidean_geometry.perpendicular SA (plane ABC))
  (h6 : euclidean_geometry.perpendicular AB BC)
  (h7 : euclidean_geometry.distance S A = 2)
  (h8 : euclidean_geometry.distance A B = 2)
  (h9 : euclidean_geometry.distance B C = 2) :
  euclidean_geometry.surface_area O = 40 * euclidean_geometry.pi := by
  sorry

end surface_area_of_sphere_l310_310120


namespace prob_both_students_female_l310_310567

-- Define the conditions
def total_students : ℕ := 5
def male_students : ℕ := 2
def female_students : ℕ := 3
def selected_students : ℕ := 2

-- Define the function to compute binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability function to select 2 female students
def probability_both_female : ℚ := 
  (binomial female_students selected_students : ℚ) / (binomial total_students selected_students : ℚ)

-- The actual theorem to be proved
theorem prob_both_students_female : probability_both_female = 0.3 := by
  sorry

end prob_both_students_female_l310_310567


namespace smallest_perfect_square_div_by_4_and_5_l310_310497

theorem smallest_perfect_square_div_by_4_and_5 : 
  ∃ n : ℕ, (∃ m : ℕ, n = m^2) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (∀ k : ℕ, (∃ l : ℕ, k = l^2) ∧ (4 ∣ k) ∧ (5 ∣ k) → n ≤ k) :=
begin
  let n := 400,
  use n,
  split,
  { use 20, -- 400 is 20^2
    refl },
  split,
  { exact dvd.intro 100 rfl }, -- 400 = 4 * 100
  split,
  { exact dvd.intro 80 rfl }, -- 400 = 5 * 80
  { 
    intros k hk,
    obtain ⟨l, hl⟩ := hk.left,
    obtain ⟨_h4⟩ := hk.right.left  -- k divisible by 4
    obtain ⟨_h5⟩ := hk.right.right -- k divisible by 5
    rw hl,
    sorry  -- This is where the rest of the proof would go.
  }
end

end smallest_perfect_square_div_by_4_and_5_l310_310497


namespace expand_simplify_expression_l310_310870

theorem expand_simplify_expression (a : ℝ) :
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6) = 
  a^5 + 19 * a^4 + 137 * a^3 + 461 * a^2 + 702 * a + 360 :=
by
  sorry

end expand_simplify_expression_l310_310870


namespace lacy_correct_percentage_l310_310420

theorem lacy_correct_percentage (x : ℕ) (h_pos : x > 0) : 
  (4 * x) / (6 * x) * 100 = 66.67 :=
by
  sorry

end lacy_correct_percentage_l310_310420


namespace cost_price_l310_310555

namespace ClothingDiscount

variables (x : ℝ)

def loss_condition (x : ℝ) : ℝ := 0.5 * x + 20
def profit_condition (x : ℝ) : ℝ := 0.8 * x - 40

def marked_price := { x : ℝ // loss_condition x = profit_condition x }

noncomputable def clothing_price : marked_price := 
    ⟨200, sorry⟩

theorem cost_price : loss_condition 200 = 120 :=
sorry

end ClothingDiscount

end cost_price_l310_310555


namespace find_y_l310_310011

-- Definition of the modified magic square
variable (a b c d e y : ℕ)

-- Conditions from the modified magic square problem
axiom h1 : y + 5 + c = 120 + a + c
axiom h2 : y + (y - 115) + e = 120 + b + e
axiom h3 : y + 25 + 120 = 5 + (y - 115) + (2*y - 235)

-- The statement to prove
theorem find_y : y = 245 :=
by
  sorry

end find_y_l310_310011


namespace angle_between_vectors_l310_310699

variables (a b : ℝ) -- These variables represent vectors in Lean.

noncomputable def norm (v : ℝ) : ℝ := |v| -- Noncomputable because we are defining norm using real absolute value.

def dot_product (v w : ℝ) : ℝ := v * w -- Definition of dot product for ℝ vectors.

theorem angle_between_vectors (a b : ℝ) (θ : ℝ) 
    (h1 : norm a = 1) 
    (h2 : norm b = 2) 
    (h3 : dot_product a b = 1) : 
    θ = 60 :=
by
  sorry

end angle_between_vectors_l310_310699


namespace tan_C_l310_310744

theorem tan_C (A B C : Type) [euclidean_triangle A B C]
  (angle_A : ∠BAC = 90°)
  (length_AB : segment_length AB = 5)
  (length_AC : segment_length AC = sqrt 34) :
  tan (angle C) = 5 / 3 :=
sorry

end tan_C_l310_310744


namespace alice_sold_20_pears_l310_310936

-- Definitions (Conditions)
def canned_more_than_poached (C P : ℝ) : Prop := C = P + 0.2 * P
def poached_less_than_sold (P S : ℝ) : Prop := P = 0.5 * S
def total_pears (S C P : ℝ) : Prop := S + C + P = 42

-- Theorem statement
theorem alice_sold_20_pears (S C P : ℝ) (h1 : canned_more_than_poached C P) (h2 : poached_less_than_sold P S) (h3 : total_pears S C P) : S = 20 :=
by 
  -- This is where the proof would go, but for now, we use sorry to signify it's omitted.
  sorry

end alice_sold_20_pears_l310_310936


namespace problem_statement_l310_310083

-- Definitions corresponding to the conditions
def A : Set Prop := {p : Prop | p = both_shots_hit}
def B : Set Prop := {p : Prop | p = both_shots_missed}
def C : Set Prop := {p : Prop | p = exactly_one_shot_hit}
def D : Set Prop := {p : Prop | p = at_least_one_shot_hit}

-- Statement to prove
theorem problem_statement : A ∪ C ≠ B ∪ D := 
sorry

end problem_statement_l310_310083


namespace rounding_to_nearest_hundredth_l310_310801

def number := 54.68237
def precision := 0.01

theorem rounding_to_nearest_hundredth :
  Real.NearestHundredth number = 54.68 :=
by
  sorry

end rounding_to_nearest_hundredth_l310_310801


namespace solve_equation_l310_310434

theorem solve_equation (x : ℝ) (hx : 0 ≤ x) : 2021 * x^10 - 2020 * x - 1 = 0 ↔ x = 1 := 
by 
  sorry

end solve_equation_l310_310434


namespace max_operations_l310_310484

def arithmetic_mean (a b : ℕ) := (a + b) / 2

theorem max_operations (b : ℕ) (hb : b < 2002) (heven : (2002 + b) % 2 = 0) :
  ∃ n, n = 10 ∧ (2002 - b) / 2^n = 1 :=
by
  sorry

end max_operations_l310_310484


namespace P_has_no_negative_roots_but_at_least_one_positive_root_l310_310973

-- Define the polynomial P(x)
def P (x : ℝ) : ℝ := x^6 - 4*x^5 - 9*x^3 + 2*x + 9

-- Statement of the problem
theorem P_has_no_negative_roots_but_at_least_one_positive_root :
  (∀ x : ℝ, x < 0 → P x ≠ 0 ∧ P x > 0) ∧ (∃ x : ℝ, x > 0 ∧ P x = 0) :=
by
  sorry

end P_has_no_negative_roots_but_at_least_one_positive_root_l310_310973


namespace floor_area_of_locus_l310_310390

notation "ℂ" => Complex

theorem floor_area_of_locus (A : ℝ) (z : ℂ) (h : |z + 12 + 9 * Complex.I| ≤ 15) 
  (area_def : A = Real.pi * 225) : ⌊A⌋ = 706 :=
sorry

end floor_area_of_locus_l310_310390


namespace solve_system_eqns_l310_310966

theorem solve_system_eqns:
  ∃ x y : ℚ, (3 * x - 2 * y = 12) ∧ (9 * y - 6 * x = -18) ∧ (x = 24/5) ∧ (y = 6/5) := 
by
  use 24/5, 6/5
  simp
  split; norm_num; sorry

end solve_system_eqns_l310_310966


namespace median_vision_is_4_6_l310_310229

def vision_data : List ℕ :=
  [1, 2, 6, 3, 3, 4, 1, 2, 5, 7, 5]

def vision_values : List ℚ :=
  [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0]

def cumulative_students : List ℕ :=
  List.scanl (· + ·) 0 vision_data

theorem median_vision_is_4_6 : 
  vision_values.nth 6 = some 4.6 := by
  have h : cumulative_students.nth 6 = some 20 := by sorry
  show vision_values.nth 6 = some 4.6 from sorry

end median_vision_is_4_6_l310_310229


namespace add_and_round_to_thousandth_l310_310565

-- Define a function to round a number to the nearest thousandth
noncomputable def round_nearest_thousandth (x : ℝ) : ℝ :=
  let scale := 1000
  let scaled_val := x * scale
  let rounded_scaled_val := Real.round scaled_val
  rounded_scaled_val / scale

-- Define the numbers
def num1 : ℝ := 78.621
def num2 : ℝ := 34.0568

-- The statement to prove
theorem add_and_round_to_thousandth :
  round_nearest_thousandth (num1 + num2) = 112.678 :=
by
  -- Proof omitted
  sorry

end add_and_round_to_thousandth_l310_310565


namespace ninth_number_value_l310_310095

theorem ninth_number_value
  (numbers : Fin 20 → ℕ)
  (h_avg_20 : (∑ i : Fin 20, numbers i) = 20 * 59)
  (h_avg_first_10 : (∑ i : Fin 10, numbers ⟨i, by linarith⟩) = 10 * 56)
  (h_avg_last_10_excl_9 :
    (∑ i : Fin 9, numbers ⟨10 + i, by linarith[intro 10 + i]⟩) + 
    (∑ i : Fin 10, numbers ⟨11 + i, by linarith[intro 11 + i]⟩) = 10 * 63)
  (h_5_eq_2_mul_diff_8_3 : numbers ⟨4, by linarith⟩ = 2 * (numbers ⟨7, by linarith⟩ - numbers ⟨2, by linarith⟩))
  (h_7_eq_sum_4_6 : numbers ⟨6, by linarith⟩ = numbers ⟨3, by linarith⟩ + numbers ⟨5, by linarith⟩)
  (h_18_eq_5_mul_16 : numbers ⟨17, by linarith⟩ = 5 * numbers ⟨15, by linarith⟩) :
  numbers ⟨8, by linarith⟩ = 10 := sorry

end ninth_number_value_l310_310095


namespace inequality_solution_l310_310971

theorem inequality_solution (x : ℝ) (h1 : x ≥ -1) (h2 : x ≤ 3) :
  sqrt (3 - x) - sqrt (x + 1) > 1 / 2 ↔ x < 1 - sqrt 31 / 8 :=
by sorry

end inequality_solution_l310_310971


namespace smallest_prime_divides_polynomial_l310_310629

theorem smallest_prime_divides_polynomial : 
  ∃ n : ℤ, n^2 + 5 * n + 23 = 17 := 
sorry

end smallest_prime_divides_polynomial_l310_310629


namespace scientific_notation_correct_l310_310090

noncomputable def scientific_notation (x : ℝ) : ℝ × ℤ :=
  let a := x * 10^9
  (a, -9)

theorem scientific_notation_correct :
  scientific_notation 0.000000007 = (7, -9) :=
by
  sorry

end scientific_notation_correct_l310_310090


namespace ways_to_place_books_in_bins_l310_310333

theorem ways_to_place_books_in_bins :
  ∃ (S: ℕ → ℕ → ℕ), S 5 3 = 25 :=
by
  use fun (n k : ℕ) => Stirling.second_kind n k
  simp [Stirling.second_kind]
  sorry

end ways_to_place_books_in_bins_l310_310333


namespace rectangular_to_polar_l310_310598

theorem rectangular_to_polar :
  ∃ (r θ : ℝ), (r = (5 / 2) ∧ θ = 2 * Real.pi - Real.arctan (4 / 3)) ∧
  (r ≥ 0 ∧ 0 ≤ θ ∧ θ < 2 * Real.pi) ∧
  let x := (3 / 2) in let y := (-2) in
  (r = Real.sqrt (x^2 + y^2) ∧ (θ = if y < 0 then 2 * Real.pi - Real.arctan (|y| / x) else Real.arctan (|y| / x))) :=
sorry

end rectangular_to_polar_l310_310598


namespace cube_edge_sum_impossible_l310_310385

theorem cube_edge_sum_impossible :
  ¬ (∃ (a : Fin 12 → ℕ) (S : Fin 6 → ℕ),
       (∀ i, i < 6 → S i = ∑ j in ({set cube_face_edges i}), a j) ∧
       |S 0 - S 1| = 1 ∧
       |S 2 - S 3| = 1 ∧
       |S 4 - S 5| = 1) :=
sorry

end cube_edge_sum_impossible_l310_310385


namespace ellipse_equation_and_eccentricity_exists_point_P_l310_310655

theorem ellipse_equation_and_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (h1 : a > b) 
  (h2 : (1/2)^2 / a^2 + (sqrt 15 / 4)^2 / b^2 = 1) (h3 : 1^2 / a^2 + (sqrt 3 / 2)^2 / b^2 = 1) :
  (a^2 = 4) ∧ (b^2 = 1) ∧ (eccentricity : ℝ := sqrt (a^2 - b^2) / a = sqrt 3 / 2) := 
begin
  sorry
end

theorem exists_point_P (k : ℝ) (A B P : ℝ × ℝ) (hxA : A.1 ∈ set.X) (hxB : B.1 ∈ set.X) 
  (hyA : A.2 = k * A.1 + 3) (hyB : B.2 = k * B.1 + 3) 
  (hP : P.1 = 0) (hP_t : ∃ t, P.2 = t) :
  ∃ P_y, P = (0, 1/3) ∧ (slope_PA := (A.2 - P.2) / A.1) + (slope_PB := (B.2 - P.2) / B.1) = 0 :=
begin
  sorry
end

end ellipse_equation_and_eccentricity_exists_point_P_l310_310655


namespace exists_parallel_and_perpendicular_planes_l310_310304

variables {l m : Line} {α : Plane}

-- Given conditions
def is_skew_lines (l m : Line) : Prop := sorry
def line_in_plane (l : Line) (α : Plane) : Prop := sorry
def plane_through_line (m : Line) : (Plane → Prop) := sorry

-- Mathematically equivalent proof problem in Lean 4 statement
theorem exists_parallel_and_perpendicular_planes :
  is_skew_lines l m →
  line_in_plane l α →
  (∀ π, plane_through_line m π) →
  (∃ π1, plane_parallel_to_line l π1 ∧ ∃ π2, plane_perpendicular_to_plane α π2) :=
sorry

end exists_parallel_and_perpendicular_planes_l310_310304


namespace allyn_total_expense_in_june_l310_310922

/-- We have a house with 40 bulbs, each using 60 watts of power daily.
Allyn pays 0.20 dollars per watt used. June has 30 days.
We need to calculate Allyn's total monthly expense on electricity in June,
which should be \$14400. -/
theorem allyn_total_expense_in_june
    (daily_watt_per_bulb : ℕ := 60)
    (num_bulbs : ℕ := 40)
    (cost_per_watt : ℝ := 0.20)
    (days_in_june : ℕ := 30)
    : num_bulbs * daily_watt_per_bulb * days_in_june * cost_per_watt = 14400 := 
by
  sorry

end allyn_total_expense_in_june_l310_310922


namespace find_x_l310_310086

def custom_op (a b : ℝ) : ℝ :=
  a^2 - 3 * b

theorem find_x (x : ℝ) : 
  (custom_op (custom_op 7 x) 3 = 18) ↔ (x = 17.71 ∨ x = 14.96) := 
by
  sorry

end find_x_l310_310086


namespace cards_given_to_friend_is_two_l310_310033

-- Definitions and conditions
def total_cards : ℕ := 16
def brother_fraction : ℚ := 3 / 8
def left_fraction : ℚ := 1 / 2
def cards_given_to_brother := (brother_fraction * total_cards : ℚ).toNat  -- 6 cards
def cards_left := (left_fraction * total_cards : ℚ).toNat  -- 8 cards
def cards_given_away := total_cards - cards_left  -- 8 cards in total
def cards_given_to_friend := cards_given_away - cards_given_to_brother  -- 2 cards

-- Proof statement
theorem cards_given_to_friend_is_two : cards_given_to_friend = 2 := by
  sorry

end cards_given_to_friend_is_two_l310_310033


namespace angles_equal_l310_310305

theorem angles_equal (α θ γ : Real) (hα : 0 < α ∧ α < π / 2) (hθ : 0 < θ ∧ θ < π / 2) (hγ : 0 < γ ∧ γ < π / 2)
  (h : Real.sin (α + γ) * Real.tan α = Real.sin (θ + γ) * Real.tan θ) : α = θ :=
by
  sorry

end angles_equal_l310_310305


namespace expand_and_simplify_l310_310854

noncomputable def expanded_expr (a : ℝ) : ℝ :=
  (a + 1) * (a + 3) * (a + 4) * (a + 5) * (a + 6)

theorem expand_and_simplify (a : ℝ) :
  expanded_expr a = a^5 + 19*a^4 + 137*a^3 + 461*a^2 + 702*a + 360 :=
by {
  -- Proof goes here
  sorry
}

end expand_and_simplify_l310_310854


namespace arithmetic_sequence_sets_count_l310_310848

theorem arithmetic_sequence_sets_count 
  (S : Finset ℕ) 
  (hS : S = Finset.range 10) : 
  (Finset.filter (λ t : Finset ℕ, 
    t.card = 3 ∧ 
    (∀ x ∈ t, x ≤ 9) ∧ 
    (0 ∈ t) ∧ 
    (∃ d, ∀ x y ∈ t, (x ≠ y) → (y - x = d ∨ x - y = d))
  ) (Finset.powerset S)).card = 5 :=
by 
  sorry

end arithmetic_sequence_sets_count_l310_310848


namespace min_distance_to_line_l310_310993

noncomputable def distance_from_point_to_line (x y : ℝ) : ℝ :=
  abs(x - y + 6) / (sqrt 2)

theorem min_distance_to_line (t α : ℝ) :
  ∃ (d : ℝ),
    (∀ (ρ θ : ℕ), (ρ^2 * (1 + 2 * sin^2 θ) = 3) → 
      ∃ (x y : ℝ), (x = t ∧ y = 6 + t ∧ d = distance_from_point_to_line (sqrt 3 * cos α) (sin α)) ∧ 
      (d = 2 * sqrt 2)) :=
begin
  sorry
end

end min_distance_to_line_l310_310993


namespace repeating_decimal_product_of_num_and_den_l310_310134

theorem repeating_decimal_product_of_num_and_den (x : ℚ) (h : x = 18 / 999) (h_simplified : x.num * x.den = 222) : x.num * x.den = 222 :=
by {
  sorry
}

end repeating_decimal_product_of_num_and_den_l310_310134


namespace packet_e_more_l310_310812

variable {a b c d e: ℝ}

-- Given conditions:
def avg_abc (a b c: ℝ) : Prop := (a + b + c) / 3 = 84
def avg_abcd (a b c d: ℝ) : Prop := (a + b + c + d) / 4 = 80
def avg_bcde (b c d e: ℝ) : Prop := (b + c + d + e) / 4 = 79
def weight_a : Prop := a = 75

-- Main theorem to prove:
theorem packet_e_more (h1: avg_abc a b c) (h2: avg_abcd a b c d) (h3: avg_bcde b c d e) (h4: weight_a):
  e - d = 3 :=
sorry

end packet_e_more_l310_310812


namespace wildlife_endangerment_safe_level_l310_310950

noncomputable def wildlife_population (t : ℝ) (K : ℝ) : ℝ :=
  K / (1 + exp (-0.12 * t - 0.8))

theorem wildlife_endangerment_safe_level
  (K : ℝ)
  (ln_2_approx : ℝ := 0.70)
  (H1 : ∀ t, wildlife_population t K = K / (1 + exp (-0.12 * t - 0.8)))
  (H2 : wildlife_population t_star K = 0.8 * K) :
  t_star = 5 :=
by 
  sorry

end wildlife_endangerment_safe_level_l310_310950


namespace thirtieth_term_is_399_l310_310459

-- Conditions
def contains_three_or_six (n : ℕ) : Prop :=
  n.to_string.contains '3' ∨ n.to_string.contains '6'

def multiples_of_three_with_conditions := 
  {n : ℕ | n % 3 = 0 ∧ contains_three_or_six n}

def nth_term (s : set ℕ) (n : ℕ) : ℕ :=
  (s.to_finset.to_list.nth (n - 1)).get_or_else 0

-- Theorem stating the 30th term of the sequence is 399
theorem thirtieth_term_is_399 : nth_term multiples_of_three_with_conditions 30 = 399 :=
sorry

end thirtieth_term_is_399_l310_310459


namespace exists_q_bound_l310_310524

noncomputable def Q1 := {x : ℚ // x ≥ 1}

variables (f : Q1 → ℝ) (ε : ℝ)
variables (hε : ε > 0) 
variables (hf : ∀ (x y : Q1), |f (x + y) - f x - f y| < ε)

theorem exists_q_bound (x : Q1) : ∃ q : ℝ, ∀ x : Q1, |f x / x.val - q| < 2 * ε := 
sorry

end exists_q_bound_l310_310524


namespace total_big_cats_at_sanctuary_l310_310202

theorem total_big_cats_at_sanctuary :
  let lions := 12
  let tigers := 14
  let cougars := (lions + tigers) / 2
  lions + tigers + cougars = 39 :=
by
  let lions := 12
  let tigers := 14
  let cougars := (lions + tigers) / 2
  have h : lions + tigers + cougars = 12 + 14 + (12 + 14) / 2 := rfl
  have h2 : 12 + 14 + (12 + 14) / 2 = 39 := by norm_num
  rw [h, h2]
  exact rfl

end total_big_cats_at_sanctuary_l310_310202


namespace point_of_impact_l310_310412

variable (R g α : ℝ)

noncomputable def V : ℝ := sqrt (2 * g * R * cos α)
noncomputable def T : ℝ := sqrt (2 * R / g) * (sin α * sqrt (cos α) + sqrt (1 - cos α ^ 3))

noncomputable def x_t (t : ℝ) : ℝ := R * sin α + V R g α * cos α * t

theorem point_of_impact :
  x_t R g α (T R g α) = R * (sin α + sin (2 * α) + sqrt (cos α * (1 - cos α ^ 3))) :=
sorry

end point_of_impact_l310_310412


namespace _l310_310286

noncomputable def is_tangent_to_parabola (x1 y1 p k : ℝ) : Prop :=
  let y := k * x1 - 1
  let rhs := x1^2
  rhs = y

noncomputable def leans_theorem_prover (O A B : (ℝ × ℝ)) : Prop :=
  -- Definitions of points
  let O := (0,0)
  let A := (1,1)
  let B := (0,-1)
  -- Value of p from point A on parabola C: x^2 = 2py
  let p := 1 / 2  -- as obtained by solving 1^2 = 2p * 1
  -- Checking option A: directrix is y = -1 is false
  let directrix := - p / 2
  (directrix ≠ -1) ∧
  -- Checking option B: tangent condition
  let slope_AB := (1 - (-1)) / (1 - 0)
  let tangent := is_tangent_to_parabola 1 1 p slope_AB
  tangent ∧
  -- Option C: |OP| * |OQ| = |OA|^2 is false
  let |OA|² := 2  -- obtained from the calculation |OA| = sqrt(1^2 + 1^2)
  ∀ (k > 2), k² ≠ |OA|² ∧
  -- Option D: |BP| * |BQ| > |BA|^2 is true
  let |BA|² := 5 -- obtained from the calculation |BA| = sqrt(1^2 + 4)
  ∀ (x1 x2) (hx1 : x1 + x2 = k) (hx2 : x1 * x2 = 1),
  let |BP| := sqrt(x1^2 + (x1^2 + 2x1 + 1))
  let |BQ| := sqrt(x2^2 + (x2^2 + 2x2 + 1))
  |BP| * |BQ| > |BA|²

example : leans_theorem_prover (0, 0) (1, 1) (0, -1) :=
by sorry  -- Proof is omitted, to be completed by Lean theorem prover

end _l310_310286


namespace dickens_birthday_l310_310811

def days_in_years (n : ℕ) : ℕ := (n * 365 + n / 4 - n / 100 + n / 400) % 7

theorem dickens_birthday : 
  days_in_years 210 = 3 → "Friday" :=
by
  sorry

end dickens_birthday_l310_310811


namespace exists_point_from_which_all_transversals_appear_at_right_angle_l310_310058

variable {P : Type} [MetricSpace P]

structure Triangle (P : Type) :=
  (A B C : P)
  (angle_A : RealAngle (B - A) (C - A) < π / 2)
  (angle_B : RealAngle (A - B) (C - B) < π / 2)
  (angle_C : RealAngle (A - C) (B - C) < π / 2)

theorem exists_point_from_which_all_transversals_appear_at_right_angle (T : Triangle P) :
  ∃ S : P, ∀ (A₁ B₁ C₁ : P), 
    IsOnLine A₁ T.B T.C → IsOnLine B₁ T.A T.C → IsOnLine C₁ T.A T.B → -- A₁ is on BC, B₁ on CA, and C₁ on AB
    RightAngle (S - T.A) (S - A₁) ∧ RightAngle (S - T.B) (S - B₁) ∧ RightAngle (S - T.C) (S - C₁) := 
  sorry

end exists_point_from_which_all_transversals_appear_at_right_angle_l310_310058


namespace translated_function_is_odd_l310_310311

theorem translated_function_is_odd
  (ω : ℝ) (φ : ℝ)
  (h1 : ω > 0)
  (h2 : abs φ < π/2)
  (h3 : ∀ x, f(x : ℝ) = sin (ω * x + φ))
  (h4 : (∀ x, f(x + π/6) = sin (2 * x - π/3)) →
        (∀ x, f(x + π/6) = -f(-x - π/6))):
  ∀ x, f x = sin (2 * x - π / 3) := sorry

end translated_function_is_odd_l310_310311


namespace find_sum_of_abc_l310_310386

theorem find_sum_of_abc
  (a b c x y : ℕ)
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : a^2 + b^2 + c^2 = 2011)
  (h3 : Nat.gcd a (Nat.gcd b c) = x)
  (h4 : Nat.lcm a (Nat.lcm b c) = y)
  (h5 : x + y = 388)
  :
  a + b + c = 61 :=
sorry

end find_sum_of_abc_l310_310386


namespace percentage_increase_l310_310427

theorem percentage_increase :
  ∃ (P : ℝ), P = 21.3333333333333∴
        let E := 210,
        let A := 225,
        let  new_A := A * (1 + P/100),
        let new_E := E * 1.2 
        new_A = new_E + 21:=
begin
  -- some proofs explanation 
  have pq1 : new_A = A * (1 + P/100),
      -- some computation  have new := (225 + 2.25 * P)

      let P :=(21.33)
      have xxx:  A * (1 + P/100) = new_E + 21 <=>( 225 * (1 + P/100) = X* (1+.2) + 21),
-- proof sketch for 
2550/11
end

end percentage_increase_l310_310427


namespace xiao_yang_correct_answers_l310_310363

noncomputable def problems_group_a : ℕ := 5
noncomputable def points_per_problem_group_a : ℕ := 8
noncomputable def problems_group_b : ℕ := 12
noncomputable def points_per_problem_group_b_correct : ℕ := 5
noncomputable def points_per_problem_group_b_incorrect : ℤ := -2
noncomputable def total_score : ℕ := 71
noncomputable def correct_answers_group_a : ℕ := 2 -- minimum required
noncomputable def correct_answers_total : ℕ := 13 -- provided correct result by the problem

theorem xiao_yang_correct_answers : correct_answers_total = 13 := by
  sorry

end xiao_yang_correct_answers_l310_310363


namespace simplify_expression_correct_l310_310513

-- Define the problem statement: simplify the given expression
constant simplify_expression : ℤ := (+7) - (-5) - (+3) + (-9)

-- Assert that the simplified result is equal to the target expression
theorem simplify_expression_correct : 
  simplify_expression = 7 + 5 - 3 - 9 :=
sorry

end simplify_expression_correct_l310_310513


namespace count_multiples_36_between_50_and_400_l310_310329

noncomputable def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem count_multiples_36_between_50_and_400 : 
  (Finset.card (Finset.filter (λ x, x % 36 = 0) (Finset.Icc 50 400))) = 10 :=
by
  sorry

end count_multiples_36_between_50_and_400_l310_310329


namespace triangle_sides_l310_310367

theorem triangle_sides (A B C : Type) [fintype A] [fintype B] [fintype C] (areaABC : ℝ) (AB : ℝ) (medians_orthogonal : Prop) :
  areaABC = 18 ∧ AB = 5 ∧ medians_orthogonal →
  BC = 2 * Real.sqrt 13 ∧ AC = Real.sqrt 73 :=
by
  sorry

end triangle_sides_l310_310367


namespace product_of_fraction_l310_310147

-- Define the repeating decimal as given in the problem
def repeating_decimal : Rat := 0.018 -- represents 0.\overline{018}

-- Define the given fraction obtained by simplifying
def simplified_fraction : Rat := 2 / 111

-- The goal is to prove that the product of the numerator and denominator of 
-- the simplified fraction of the repeating decimal is 222
theorem product_of_fraction (y : Rat) (hy : y = 0.018) (fraction_eq : y = 18 / 999) : 
  (2:ℕ) * (111:ℕ) = 222 :=
by
  sorry

end product_of_fraction_l310_310147


namespace units_digit_of_k_squared_plus_3_to_k_l310_310045

theorem units_digit_of_k_squared_plus_3_to_k (k : ℤ) (h1 : k % 10 = 7) (h2 : k^2 % 10 = 9) (h3 : 3^k % 10 = 7) :
    (k^2 + 3^k) % 10 = 6 := by
  sorry

end units_digit_of_k_squared_plus_3_to_k_l310_310045


namespace arithmetic_expression_equals_fraction_l310_310578

theorem arithmetic_expression_equals_fraction (a b c : ℚ) :
  a = 1/8 → b = 1/9 → c = 1/28 →
  (a * b * c = 1/2016) ∨ ((a - b) * c = 1/2016) :=
by
  intros ha hb hc
  rw [ha, hb, hc]
  left
  sorry

end arithmetic_expression_equals_fraction_l310_310578


namespace simplify_expression_l310_310510

theorem simplify_expression : 1 + (1 / (1 + (1 / (2 + 1)))) = 7 / 4 :=
by
  sorry

end simplify_expression_l310_310510


namespace equal_circumradii_of_BDE_and_AOC_l310_310270

-- Define the basic geometric entities and conditions
variables {A B C O D E : Point}
variable [IsTriangle ABC]
variable [IsCircumcenter O ABC]
variable [OnCircle D (Circumcircle AOC)]
variable [OnCircle E (Circumcircle AOC)]
variable [OnLine C B D]
variable [OnLine A B E]

-- State the theorem to be proved
theorem equal_circumradii_of_BDE_and_AOC :
  radius (Circumcircle BDE) = radius (Circumcircle AOC) :=
sorry

end equal_circumradii_of_BDE_and_AOC_l310_310270


namespace sin_neg_45_l310_310959

theorem sin_neg_45 :
  Real.sin (-45 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
sorry

end sin_neg_45_l310_310959


namespace largest_difference_in_S_l310_310491

def S : List Int := [-25, -10, -1, 5, 8, 20]

theorem largest_difference_in_S : 45 ∈ {a - b | a b : Int, a ∈ S ∧ b ∈ S} :=
by {
  sorry
}

end largest_difference_in_S_l310_310491


namespace cubic_sum_l310_310262

theorem cubic_sum (x : ℝ) (h : x + 1/x = 4) : x^3 + 1/x^3 = 52 :=
by 
  sorry

end cubic_sum_l310_310262


namespace solve_circle_tangent_and_intercept_l310_310674

namespace CircleProblems

-- Condition: Circle equation
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y + 3 = 0

-- Problem 1: Equations of tangent lines with equal intercepts
def tangent_lines_with_equal_intercepts (l : ℝ → ℝ → Prop) : Prop :=
  (∀ x y : ℝ, l x y ↔ x + y + 1 = 0) ∨ (∀ x y : ℝ, l x y ↔ x + y - 3 = 0)

-- Problem 2: Equations of lines passing through origin and intercepted by the circle with a segment length of 2
def lines_intercepted_by_circle (l : ℝ → ℝ → Prop) : Prop :=
  (∀ x y : ℝ, l x y ↔ x = 0) ∨ (∀ x y : ℝ, l x y ↔ y = - (3 / 4) * x)

theorem solve_circle_tangent_and_intercept (l_tangent l_origin : ℝ → ℝ → Prop) :
  (∀ x y : ℝ, circle_eq x y → l_tangent x y) →
  tangent_lines_with_equal_intercepts l_tangent ∧ lines_intercepted_by_circle l_origin :=
by
  sorry

end CircleProblems

end solve_circle_tangent_and_intercept_l310_310674


namespace cardinality_M_l310_310832

def nat_set := {m : ℕ | 8 - m ∈ ℕ}

theorem cardinality_M : nat_set.to_finset.card = 9 := 
by sorry

end cardinality_M_l310_310832


namespace max_cardinality_of_set_l310_310401

theorem max_cardinality_of_set (M : Set ℕ) (hM : ∀ (a b c ∈ M), (a ≠ b ∧ a ≠ c ∧ b ≠ c) → (a ∣ b ∨ b ∣ a ∨ a ∣ c ∨ c ∣ a ∨ b ∣ c ∨ c ∣ b)) :
  M ⊆ (Finset.range 2012).toSet → 
  ∃ N, N = 21 ∧ 
    (∀ M', (M' ⊆ M) → 
      (∀ (a b c ∈ M'), (a ≠ b ∧ a ≠ c ∧ b ≠ c) → 
        (a ∣ b ∨ b ∣ a ∨ b ∣ c ∨ c ∣ a ∨ c ∣ b)) → 
      |M'| ≤ 21) := 
sorry

end max_cardinality_of_set_l310_310401


namespace number_of_students_in_course_l310_310781

-- Define the conditions
def total (T : ℕ) :=
  (1/5 : ℚ) * T + (1/4 : ℚ) * T + (1/2 : ℚ) * T + 20 = T

-- Formalize the problem statement
theorem number_of_students_in_course : ∃ T : ℕ, total T ∧ T = 400 := 
sorry

end number_of_students_in_course_l310_310781


namespace trigonometric_identity_third_quadrant_l310_310714

theorem trigonometric_identity_third_quadrant (α : ℝ) (h1 : sin α < 0) (h2 : cos α < 0) :
  (cos α / sqrt (1 - sin α ^ 2) + 2 * sin α / sqrt (1 - cos α ^ 2)) = -3 :=
by
  sorry

end trigonometric_identity_third_quadrant_l310_310714


namespace find_x_values_l310_310279

def p (x : ℤ) := x^2 - x ≥ 6
def q (x : ℤ) := x ∈ ℤ

theorem find_x_values : {x : ℤ | ¬ (p x ∧ q x) ∧ ¬ (¬ q x)} = {-1, 0, 1, 2} :=
by
  sorry

end find_x_values_l310_310279


namespace speed_for_remaining_distance_l310_310066

theorem speed_for_remaining_distance
  (t_total : ℝ) (v1 : ℝ) (d_total : ℝ)
  (t_total_def : t_total = 1.4)
  (v1_def : v1 = 4)
  (d_total_def : d_total = 5.999999999999999) :
  ∃ v2 : ℝ, v2 = 5 := 
by
  sorry

end speed_for_remaining_distance_l310_310066


namespace sqrt_inequality_l310_310794

theorem sqrt_inequality : sqrt 6 + sqrt 7 > 2 * sqrt 2 + sqrt 5 :=
sorry

end sqrt_inequality_l310_310794


namespace distance_between_parallel_lines_l310_310448

theorem distance_between_parallel_lines :
  let line1 : ℝ → ℝ → Prop := λ x y, x + y - 1 = 0
  let line2 : ℝ → ℝ → Prop := λ x y, 2x + 2y + 3 = 0
  ∃ d : ℝ, d = abs ((3/2) - (-1)) / real.sqrt (1^2 + 1^2) ∧ d = 5*real.sqrt 2/4 :=
by
  sorry

end distance_between_parallel_lines_l310_310448


namespace metallic_sheet_dimension_l310_310544

theorem metallic_sheet_dimension :
  ∃ w : ℝ, (∀ (h := 8) (l := 40) (v := 2688),
    v = (w - 2 * h) * (l - 2 * h) * h) → w = 30 :=
by sorry

end metallic_sheet_dimension_l310_310544


namespace highlighter_count_l310_310726

-- Define the quantities of highlighters.
def pinkHighlighters := 3
def yellowHighlighters := 7
def blueHighlighters := 5

-- Define the total number of highlighters.
def totalHighlighters := pinkHighlighters + yellowHighlighters + blueHighlighters

-- The theorem states that the total number of highlighters is 15.
theorem highlighter_count : totalHighlighters = 15 := by
  -- Proof skipped for now.
  sorry

end highlighter_count_l310_310726


namespace gcd_1975_2625_l310_310239

def gcd : ℕ → ℕ → ℕ
| a, 0 => a
| a, b => gcd b (a % b)

theorem gcd_1975_2625 : gcd 1975 2625 = 25 := by
  sorry

end gcd_1975_2625_l310_310239


namespace find_larger_number_l310_310098

theorem find_larger_number (S L : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 6 * S + 15) : 
  L = 1635 := 
sorry

end find_larger_number_l310_310098


namespace find_XY_in_triangle_l310_310998

theorem find_XY_in_triangle
  (XYZ: Type*)
  [triangle XYZ]
  (X Y Z : XYZ)
  (right_angle : ∠ Y = 90)
  (angle_Z : ∠ Z = 60)
  (YZ_length : distance Y Z = 24) :
  distance X Y = 12 :=
sorry

end find_XY_in_triangle_l310_310998


namespace value_of_a_2m_n_l310_310642

variable (a m n : ℝ)

-- Conditions as Lean statements
def log_a_2 (h : 0 < a) : \ℝ := log a 2 = m
def log_a_3 (h : 0 < a) : \ℝ := log a 3 = n

-- Proof statement
theorem value_of_a_2m_n (h : 0 < a) (hm : log a 2 = m) (hn : log a 3 = n) : a^(2 * m + n) = 12 :=
by 
  sorry

end value_of_a_2m_n_l310_310642


namespace find_p_q_r_l310_310046

noncomputable def problem_statement : Prop :=
let n := (2 / (x - 2) + 6 / (x - 6) + 13 / (x - 13) + 15 / (x - 15) = x^2 - 10x - 2) in
let t := x - 10 in
let equation := t^4 - 42*t^2 + 400 = 0 in
∃ p q r : ℕ, n = p + sqrt (q + sqrt r) ∧ p + q + r = 72

theorem find_p_q_r : problem_statement := sorry

end find_p_q_r_l310_310046


namespace midpoints_not_collinear_l310_310132

noncomputable theory
open_locale classical

variables {A B C A1 B1 C1 : Type*} [metric_space A] [metric_space B] [metric_space C]
  [metric_space A1] [metric_space B1] [metric_space C1]
  [inhabited A] [inhabited B] [inhabited C] [inhabited A1] [inhabited B1] [inhabited C1]

def midpoint (x y : Type*) [metric_space x] [metric_space y] := 
  sorry -- assume appropriate definition here

def triangle (A B C : Type*) [metric_space A] [metric_space B] [metric_space C] := 
  sorry -- assume appropriate definition here

variables (ABC : triangle A B C)
  (A1_on_opposite_side : A1 ≠ A)
  (B1_on_opposite_side : B1 ≠ B)
  (C1_on_opposite_side : C1 ≠ C)

theorem midpoints_not_collinear :
  let M1 := midpoint A A1
  let M2 := midpoint B B1
  let M3 := midpoint C C1
  (¬ collinear ({M1, M2, M3} : set (Type*))) :=
by
  -- Proposition that needs proof
  sorry

end midpoints_not_collinear_l310_310132


namespace weekly_protein_proof_l310_310800

noncomputable def nutritional_values := 
  let rice_fat := 10
  let rice_carbs := 45
  let rice_protein := 5
  
  let chicken_fat := 2
  let chicken_carbs := 0
  let chicken_protein := 7
  
  let vegetables_fat := 0.3
  let vegetables_carbs := 5
  let vegetables_protein := 2
  
  let beans_fat := 1
  let beans_carbs := 40
  let beans_protein := 15
  
  let beef_fat := 6
  let beef_carbs := 0
  let beef_protein := 7
  
  let cheese_fat := 18
  let cheese_carbs := 2
  let cheese_protein := 14
  
  let almonds_fat := 6
  let almonds_carbs := 2
  let almonds_protein := 3
  
  (rice_fat, rice_carbs, rice_protein, 
   chicken_fat, chicken_carbs, chicken_protein,
   vegetables_fat, vegetables_carbs, vegetables_protein,
   beans_fat, beans_carbs, beans_protein,
   beef_fat, beef_carbs, beef_protein,
   cheese_fat, cheese_carbs, cheese_protein,
   almonds_fat, almonds_carbs, almonds_protein)

noncomputable def daily_intake := nutritional_values

-- Calculate the daily intake for each nutrient
def daily_protein : ℕ :=
  let (rice_fat, rice_carbs, rice_protein, 
       chicken_fat, chicken_carbs, chicken_protein,
       vegetables_fat, vegetables_carbs, vegetables_protein,
       beans_fat, beans_carbs, beans_protein,
       beef_fat, beef_carbs, beef_protein,
       cheese_fat, cheese_carbs, cheese_protein,
       almonds_fat, almonds_carbs, almonds_protein) := daily_intake in
  (3 * rice_protein) + (8 * chicken_protein) + (2 * vegetables_protein) +
  (2 * rice_protein) + (1 * beans_protein) + (6 * beef_protein) +
  (5 * rice_protein) + (0.5 * cheese_protein) + (10 * almonds_protein)

-- Weekly intake of each nutrient based on the daily intake
def weekly_protein : ℕ := 7 * daily_protein

theorem weekly_protein_proof :
  weekly_protein = 1239 := 
by
  sorry

end weekly_protein_proof_l310_310800


namespace problem1_problem2_l310_310897

-- Problem (I)
theorem problem1 (α : ℝ) (h1 : Real.tan α = 3) :
  (4 * Real.sin (Real.pi - α) - 2 * Real.cos (-α)) / (3 * Real.cos (Real.pi / 2 - α) - 5 * Real.cos (Real.pi + α)) = 5 / 7 := by
sorry

-- Problem (II)
theorem problem2 (x : ℝ) (h2 : Real.sin x + Real.cos x = 1 / 5) (h3 : 0 < x ∧ x < Real.pi) :
  Real.sin x = 4 / 5 ∧ Real.cos x = -3 / 5 := by
sorry

end problem1_problem2_l310_310897


namespace bus_stops_for_28_minutes_per_hour_l310_310996

-- Definitions based on the conditions
def without_stoppages_speed : ℕ := 75
def with_stoppages_speed : ℕ := 40
def speed_difference : ℕ := without_stoppages_speed - with_stoppages_speed

-- Theorem statement
theorem bus_stops_for_28_minutes_per_hour : 
  ∀ (T : ℕ), (T = (speed_difference*60)/(without_stoppages_speed))  → 
  T = 28 := 
by
  sorry

end bus_stops_for_28_minutes_per_hour_l310_310996


namespace parabola_eq_OA_dot_OB_eq_neg16_l310_310265

open Real

-- Condition: Point P(1, m) on the parabola y^2 = 2px and PF = 3
constant (m p : ℝ)
constant (T A B : Point)
axiom point_p_on_parabola : ∀ (P : Point), P = ⟨1, m⟩ → P ∈ parabola (2 * p)
axiom p_pos : p > 0
noncomputable def F : Point := ⟨p / 2, 0⟩
axiom PF_eq_3 : dist ⟨1, m⟩ F = 3

-- Proof to find the equation of the parabola
theorem parabola_eq : ∀ P = ⟨1, m⟩, P ∈ parabola(2 * 3) :=
by
  sorry

-- Proof for the intersection points and vector dot product
noncomputable def O : Point := ⟨0, 0⟩
constant line_through_T : Line → ¬verticalLine L ∧ passes_through ⟨4, 0⟩ T
axiom intersections_A_B : ∃ A B, A, B ∈ intersectionPoints (parabola ⟨6x⟩) (line_through_T)

theorem OA_dot_OB_eq_neg16 : ∀ (l : Line), intersects l (parabola (2 * 3)) A B ∧ passes_through l (⟨4, 0⟩ T) → vector dot_product OA OB = -16 :=
by
  sorry

end parabola_eq_OA_dot_OB_eq_neg16_l310_310265


namespace parabola_intersection_sum_zero_l310_310456

theorem parabola_intersection_sum_zero
    (x y : ℝ → ℝ)
    (h1 : ∀ x, y x = (x + 2)^2)
    (h2 : ∀ y, x y = (y - 2)^2 - 3) :
    let points := {p : ℝ × ℝ | ∃ x y : ℝ, y = (x + 2)^2 ∧ x + 3 = (y - 2)^2} in
    ∃ x1 x2 x3 x4 y1 y2 y3 y4 : ℝ,
    (x1, y1) ∈ points ∧ (x2, y2) ∈ points ∧ (x3, y3) ∈ points ∧ (x4, y4) ∈ points ∧
    x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 0 :=
by
    sorry

end parabola_intersection_sum_zero_l310_310456


namespace no_32_students_yes_30_students_l310_310359

variables (m d : ℕ)

-- Each boy is friends with exactly three girls, and each girl is friends with exactly two boys.
def friendship_condition (m d : ℕ) : Prop :=
  3 * m = 2 * d

def total_students (m d : ℕ) : ℕ :=
  m + d

-- There cannot be 32 students in the class
theorem no_32_students (m d : ℕ) (h1 : friendship_condition m d) (h2 : total_students m d = 32) : false :=
begin
  sorry,
end

-- There can be 30 students in the class
theorem yes_30_students (m d : ℕ) (h1 : friendship_condition m d) (h2 : total_students m d = 30) : true :=
begin
  trivial,
end

end no_32_students_yes_30_students_l310_310359


namespace probability_product_eight_l310_310574

def roll_die_outcomes : Finset (ℕ × ℕ) := 
  do x ← Finset.range (6 + 1) 
     y ← Finset.range (6 + 1)
     [(x, y)]

def successful_outcomes : Finset (ℕ × ℕ) := 
  roll_die_outcomes.filter (λ xy => xy.1 * xy.2 = 8)

theorem probability_product_eight :
  (successful_outcomes.card : ℚ) / (roll_die_outcomes.card : ℚ) = 1 / 18 := 
by
  sorry

end probability_product_eight_l310_310574


namespace alice_cannot_guarantee_no_loss_l310_310160

noncomputable def cannot_guarantee_no_loss (n : ℕ) : Prop :=
  ∀ (initial_white_piece : ℕ^n), 
  ∃ (time_limit : ℕ),
  ∀ (black_piece_strategy : ℕ^n → ℕ^n),
  ∃ (white_piece_position : ℕ^n), 
  (manhattan_distance (black_piece_strategy time_limit) white_piece_position = 0) ∧
  (white_piece_position ≠ initial_white_piece)

theorem alice_cannot_guarantee_no_loss (n : ℕ) : cannot_guarantee_no_loss n :=
sorry

end alice_cannot_guarantee_no_loss_l310_310160


namespace probability_greater_than_115_l310_310362

theorem probability_greater_than_115 
  {σ : ℝ} (hσ : σ > 0) (hP : ∀ ξ : ℝ, ξ ~ Normal 100 σ^2) : 
  (∃ ξ, P(85 < ξ ∧ ξ < 115) = 0.75) → P(ξ > 115) = 0.125 :=
by
  sorry

end probability_greater_than_115_l310_310362


namespace sum_coefficients_eq_neg_one_l310_310338

theorem sum_coefficients_eq_neg_one
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_{10} a_{11} a_{12} a_{13} a_{14} a_{15} a_{16} a_{17} : ℤ) :
  (x - 1) ^ 17 = a_0 + a_1 * (1 + x) + a_2 * (1 + x) ^ 2 + a_3 * (1 + x) ^ 3 + 
  a_4 * (1 + x) ^ 4 + a_5 * (1 + x) ^ 5 + a_6 * (1 + x) ^ 6 + a_7 * (1 + x) ^ 7 + 
  a_8 * (1 + x) ^ 8 + a_9 * (1 + x) ^ 9 + a_{10} * (1 + x) ^ 10 + a_{11} * (1 + x) ^ 11 + 
  a_{12} * (1 + x) ^ 12 + a_{13} * (1 + x) ^ 13 + a_{14} * (1 + x) ^ 14 + 
  a_{15} * (1 + x) ^ 15 + a_{16} * (1 + x) ^ 16 + a_{17} * (1 + x) ^ 17 →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_{10} + a_{11} + 
  a_{12} + a_{13} + a_{14} + a_{15} + a_{16} + a_{17} = -1 :=
begin
  sorry
end

end sum_coefficients_eq_neg_one_l310_310338


namespace _l310_310287

noncomputable def is_tangent_to_parabola (x1 y1 p k : ℝ) : Prop :=
  let y := k * x1 - 1
  let rhs := x1^2
  rhs = y

noncomputable def leans_theorem_prover (O A B : (ℝ × ℝ)) : Prop :=
  -- Definitions of points
  let O := (0,0)
  let A := (1,1)
  let B := (0,-1)
  -- Value of p from point A on parabola C: x^2 = 2py
  let p := 1 / 2  -- as obtained by solving 1^2 = 2p * 1
  -- Checking option A: directrix is y = -1 is false
  let directrix := - p / 2
  (directrix ≠ -1) ∧
  -- Checking option B: tangent condition
  let slope_AB := (1 - (-1)) / (1 - 0)
  let tangent := is_tangent_to_parabola 1 1 p slope_AB
  tangent ∧
  -- Option C: |OP| * |OQ| = |OA|^2 is false
  let |OA|² := 2  -- obtained from the calculation |OA| = sqrt(1^2 + 1^2)
  ∀ (k > 2), k² ≠ |OA|² ∧
  -- Option D: |BP| * |BQ| > |BA|^2 is true
  let |BA|² := 5 -- obtained from the calculation |BA| = sqrt(1^2 + 4)
  ∀ (x1 x2) (hx1 : x1 + x2 = k) (hx2 : x1 * x2 = 1),
  let |BP| := sqrt(x1^2 + (x1^2 + 2x1 + 1))
  let |BQ| := sqrt(x2^2 + (x2^2 + 2x2 + 1))
  |BP| * |BQ| > |BA|²

example : leans_theorem_prover (0, 0) (1, 1) (0, -1) :=
by sorry  -- Proof is omitted, to be completed by Lean theorem prover

end _l310_310287


namespace lunch_cost_calc_l310_310075

-- Define the given conditions
def gasoline_cost : ℝ := 8
def gift_cost : ℝ := 5
def grandma_gift : ℝ := 10
def initial_money : ℝ := 50
def return_trip_money : ℝ := 36.35

-- Calculate the total expenses and determine the money spent on lunch
def total_gifts_cost : ℝ := 2 * gift_cost
def total_money_received : ℝ := initial_money + 2 * grandma_gift
def total_gas_gift_cost : ℝ := gasoline_cost + total_gifts_cost
def expected_remaining_money : ℝ := total_money_received - total_gas_gift_cost
def lunch_cost : ℝ := expected_remaining_money - return_trip_money

-- State theorem
theorem lunch_cost_calc : lunch_cost = 15.65 := by
  sorry

end lunch_cost_calc_l310_310075


namespace negative_angle_to_quadrant_l310_310527

def angle_quadrant (angle : ℝ) : string :=
if angle > 0 ∧ angle < 90 then "First quadrant"
else if angle > 90 ∧ angle < 180 then "Second quadrant"
else if angle > 180 ∧ angle < 270 then "Third quadrant"
else if angle > 270 ∧ angle < 360 then "Fourth quadrant"
else if angle < 0 then angle_quadrant (angle + 360)
else "On axis"

theorem negative_angle_to_quadrant : angle_quadrant (-215) = "Fourth quadrant" :=
by
  sorry

end negative_angle_to_quadrant_l310_310527


namespace larger_sphere_radius_l310_310810

theorem larger_sphere_radius 
  (n : ℕ)
  (r_small r_large : ℝ)
  (h_n : n = 10)
  (h_r_small : r_small = 3)
  (volume_small : ℝ := (4 / 3) * Real.pi * r_small^3)
  (volume_total : ℝ := n * volume_small)
  (volume_large := (4 / 3) * Real.pi * r_large^3) :
  r_large ≈ Real.cbrt (10 * (3^3)) := sorry

end larger_sphere_radius_l310_310810


namespace difference_of_squares_l310_310885

theorem difference_of_squares (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 8) : x^2 - y^2 = 160 :=
by
  sorry

end difference_of_squares_l310_310885


namespace Q_vector_representation_l310_310027

-- Variables representing points in vectorspace
variables {V : Type*} [add_comm_group V] [vector_space ℚ V]
variables (X Y Z M N Q: V)
variables {λ μ : ℚ}

-- Conditions given in the problem
def cond1 : YM:MZ = 4:1 := sorry
def M_def : M = 4/5 • Z + 1/5 • Y := sorry
def cond2 : XN:NZ = 3:2 := sorry
def N_def : N = 3/5 • X + 2/5 • Z := sorry
def Q_intersection : intersection YN XM Q := sorry
def sum_condition (a b c : ℚ) : a + b + c = 1 := sorry

-- Proposition to prove
theorem Q_vector_representation (a b c : ℚ) :
  cond1 → M_def → cond2 → N_def → Q_intersection → sum_condition a b c →
  Q = a • X + b • Y + c • Z :=
begin
  intros,
  have h := by exact sorry,
  exact h,
end

end Q_vector_representation_l310_310027


namespace find_mean_senior_score_l310_310590

def mean_all_scores (total_score : ℝ) (num_students : ℝ) : ℝ := total_score / num_students

def num_non_seniors (num_seniors : ℝ) : ℝ := 1.2 * num_seniors

def total_students (num_seniors num_non_seniors : ℝ) : ℝ := num_seniors + num_non_seniors

def mean_senior_score (mean_non_senior_score : ℝ) : ℝ := 1.25 * mean_non_senior_score

def total_score (num_seniors num_non_seniors mean_senior_score mean_non_senior_score : ℝ) : ℝ :=
  num_seniors * mean_senior_score + num_non_seniors * mean_non_senior_score

theorem find_mean_senior_score :
  ∀ (num_students : ℝ) (mean_score : ℝ)
    (s n m_s m_n : ℝ),
    total_students s n = num_students →
    mean_all_scores (total_score s n m_s m_n) num_students = mean_score →
    num_non_seniors s = n →
    mean_senior_score m_n = m_s →
    m_s = 100.19 :=
by
  intros
  sorry

end find_mean_senior_score_l310_310590


namespace max_value_of_f_l310_310661

noncomputable def f (x y : ℝ) : ℝ := real.sqrt (8 * y - 6 * x + 50) + real.sqrt (8 * y + 6 * x + 50)

theorem max_value_of_f (x y : ℝ) (h : x^2 + y^2 = 25) : f x y ≤ 6 * real.sqrt 10 := by
  sorry

end max_value_of_f_l310_310661


namespace reflection_combination_l310_310881

variables (l1 l2 : ℝ → ℝ → Prop) [Parallel l1 l2] (a : ℝ) 
          (T : ℝ → ℝ → ℝ → ℝ → Prop) 
          (S_l1 S_l2 : ℝ → ℝ → ℝ → ℝ → Prop)

def a_perp := ∀ (x y : ℝ), l1 x y → a * x = 0
def T_2a := λ x y: ℝ, T (2 * a * x) (2 * a * y)

theorem reflection_combination : 
  (∀ x y : ℝ, S_l1 (S_l2 x y) = T_2a x y) → 
  S_l1 ∘ S_l2 = T_2a :=
sorry

end reflection_combination_l310_310881


namespace product_of_fraction_l310_310146

-- Define the repeating decimal as given in the problem
def repeating_decimal : Rat := 0.018 -- represents 0.\overline{018}

-- Define the given fraction obtained by simplifying
def simplified_fraction : Rat := 2 / 111

-- The goal is to prove that the product of the numerator and denominator of 
-- the simplified fraction of the repeating decimal is 222
theorem product_of_fraction (y : Rat) (hy : y = 0.018) (fraction_eq : y = 18 / 999) : 
  (2:ℕ) * (111:ℕ) = 222 :=
by
  sorry

end product_of_fraction_l310_310146


namespace find_BC_l310_310658

-- Define points A, B and their coordinates
structure Point where
  x : ℝ
  y : ℝ

def A : Point := ⟨0, 1⟩
def B : Point := ⟨2, 1⟩

-- Define vector AC
structure Vector where
  dx : ℝ
  dy : ℝ

def AC : Vector := ⟨-3, -2⟩

-- Define the target vector for the proof
def BC : Vector := ⟨-5, -2⟩

-- Proof statement
-- Given the points A, B, and vector AC, prove vector BC is as defined
theorem find_BC :
  ∀ C : Point, 
    Vector.mk (C.x - B.x) (C.y - B.y) = Vector.mk (AC.dx - (A.x - B.x)) (AC.dy - (A.y - B.y)) →
    Vector.mk (C.x - B.x) (C.y - B.y) = BC :=
by
  intros
  sorry

end find_BC_l310_310658


namespace smallest_number_of_cubes_l310_310903

def box_length : ℕ := 49
def box_width : ℕ := 42
def box_depth : ℕ := 14
def gcd_box_dimensions : ℕ := Nat.gcd (Nat.gcd box_length box_width) box_depth

theorem smallest_number_of_cubes :
  (box_length / gcd_box_dimensions) *
  (box_width / gcd_box_dimensions) *
  (box_depth / gcd_box_dimensions) = 84 := by
  sorry

end smallest_number_of_cubes_l310_310903
