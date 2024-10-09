import Mathlib

namespace calculate_monthly_rent_l318_31880

theorem calculate_monthly_rent (P : ℝ) (R : ℝ) (T : ℝ) (M : ℝ) (rent : ℝ) :
  P = 12000 →
  R = 0.06 →
  T = 400 →
  M = 0.1 →
  rent = 103.70 :=
by
  intros hP hR hT hM
  sorry

end calculate_monthly_rent_l318_31880


namespace total_green_and_yellow_peaches_in_basket_l318_31846

def num_red_peaches := 5
def num_yellow_peaches := 14
def num_green_peaches := 6

theorem total_green_and_yellow_peaches_in_basket :
  num_yellow_peaches + num_green_peaches = 20 :=
by
  sorry

end total_green_and_yellow_peaches_in_basket_l318_31846


namespace quadratic_roots_abs_eq_l318_31897

theorem quadratic_roots_abs_eq (x1 x2 m : ℝ) (h1 : x1 > 0) (h2 : x2 < 0) 
  (h_eq_roots : ∀ x, x^2 - (x1 + x2)*x + x1*x2 = 0) : 
  ∃ q : ℝ, q = x^2 - (1 - 4*m)/x + 2 := 
by
  sorry

end quadratic_roots_abs_eq_l318_31897


namespace ratio_of_a_to_b_and_c_l318_31820

theorem ratio_of_a_to_b_and_c (A B C : ℝ) (h1 : A = 160) (h2 : A + B + C = 400) (h3 : B = (2/3) * (A + C)) :
  A / (B + C) = 2 / 3 :=
by
  sorry

end ratio_of_a_to_b_and_c_l318_31820


namespace age_difference_l318_31867

variable (A B C : ℕ)

theorem age_difference (h1 : A + B > B + C) (h2 : C = A - 13) : (A + B) - (B + C) = 13 := by
  sorry

end age_difference_l318_31867


namespace probability_of_selecting_cooking_l318_31855

def total_courses : ℕ := 4
def favorable_outcomes : ℕ := 1

theorem probability_of_selecting_cooking : (favorable_outcomes : ℚ) / total_courses = 1 / 4 := 
by 
  sorry

end probability_of_selecting_cooking_l318_31855


namespace necessary_and_sufficient_condition_l318_31809

theorem necessary_and_sufficient_condition (t : ℝ) :
  ((t + 1) * (1 - |t|) > 0) ↔ (t < 1 ∧ t ≠ -1) :=
by
  sorry

end necessary_and_sufficient_condition_l318_31809


namespace kayak_rental_cost_l318_31802

theorem kayak_rental_cost (F : ℝ) (C : ℝ) (h1 : ∀ t : ℝ, C = F + 5 * t)
  (h2 : C = 30) : C = 45 :=
sorry

end kayak_rental_cost_l318_31802


namespace total_eggs_l318_31869

noncomputable def total_eggs_in_all_containers (n : ℕ) (f l : ℕ) : ℕ :=
  n * (f * l)

theorem total_eggs (f l : ℕ) :
  (f = 14 + 20 - 1) →
  (l = 3 + 2 - 1) →
  total_eggs_in_all_containers 28 f l = 3696 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end total_eggs_l318_31869


namespace molecular_weight_of_one_mole_l318_31839

theorem molecular_weight_of_one_mole 
  (total_weight : ℝ) (n_moles : ℝ) (mw_per_mole : ℝ)
  (h : total_weight = 792) (h2 : n_moles = 9) 
  (h3 : total_weight = n_moles * mw_per_mole) 
  : mw_per_mole = 88 :=
by
  sorry

end molecular_weight_of_one_mole_l318_31839


namespace other_diagonal_of_rhombus_l318_31829

noncomputable def calculate_other_diagonal (area d1 : ℝ) : ℝ :=
  (area * 2) / d1

theorem other_diagonal_of_rhombus {a1 a2 : ℝ} (area_eq : a1 = 21.46) (d1_eq : a2 = 7.4) : calculate_other_diagonal a1 a2 = 5.8 :=
by
  rw [area_eq, d1_eq]
  norm_num
  -- The next step would involve proving that (21.46 * 2) / 7.4 = 5.8 in a formal proof.
  sorry

end other_diagonal_of_rhombus_l318_31829


namespace largest_possible_perimeter_l318_31806

theorem largest_possible_perimeter
  (a b c : ℕ)
  (h1 : a > 2 ∧ b > 2 ∧ c > 2)  -- sides are greater than 2
  (h2 : a = c ∨ b = c ∨ a = b)  -- at least two polygons are congruent
  (h3 : (a - 2) * (b - 2) = 8 ∨ (a - 2) * (c - 2) = 8 ∨ (b - 2) * (c - 2) = 8)  -- possible factorizations
  (h4 : (a - 2) + (b - 2) + (c - 2) = 12)  -- sum of interior angles at A is 360 degrees
  : 2 * a + 2 * b + 2 * c - 6 ≤ 21 :=
sorry

end largest_possible_perimeter_l318_31806


namespace domain_range_equal_l318_31865

noncomputable def f (a b x : ℝ) : ℝ := Real.sqrt (a * x^2 + b * x)

theorem domain_range_equal {a b : ℝ} (hb : b > 0) :
  (∀ y, ∃ x, f a b x = y) ↔ (a = -4 ∨ a = 0) :=
sorry

end domain_range_equal_l318_31865


namespace packing_heights_difference_l318_31824

-- Definitions based on conditions
def diameter := 8   -- Each pipe has a diameter of 8 cm
def num_pipes := 160 -- Each crate contains 160 pipes

-- Heights of the crates based on the given packing methods
def height_crate_A := 128 -- Calculated height for Crate A

noncomputable def height_crate_B := 8 + 60 * Real.sqrt 3 -- Calculated height for Crate B

-- Positive difference in the total heights of the two packings
noncomputable def delta_height := height_crate_A - height_crate_B

-- The goal to prove
theorem packing_heights_difference :
  delta_height = 120 - 60 * Real.sqrt 3 :=
sorry

end packing_heights_difference_l318_31824


namespace remaining_amount_is_16_l318_31801

-- Define initial amount of money Sam has.
def initial_amount : ℕ := 79

-- Define cost per book.
def cost_per_book : ℕ := 7

-- Define the number of books.
def number_of_books : ℕ := 9

-- Define the total cost of books.
def total_cost : ℕ := cost_per_book * number_of_books

-- Define the remaining amount of money after buying the books.
def remaining_amount : ℕ := initial_amount - total_cost

-- Prove the remaining amount is 16 dollars.
theorem remaining_amount_is_16 : remaining_amount = 16 := by
  rfl

end remaining_amount_is_16_l318_31801


namespace abigail_money_loss_l318_31882

theorem abigail_money_loss
  (initial_amount : ℕ)
  (spent_amount : ℕ)
  (remaining_amount : ℕ)
  (h1 : initial_amount = 11)
  (h2 : spent_amount = 2)
  (h3 : remaining_amount = 3) :
  initial_amount - spent_amount - remaining_amount = 6 :=
by sorry

end abigail_money_loss_l318_31882


namespace loss_percentage_l318_31805

theorem loss_percentage (CP SP : ℝ) (h_CP : CP = 1300) (h_SP : SP = 1040) :
  ((CP - SP) / CP) * 100 = 20 :=
by
  sorry

end loss_percentage_l318_31805


namespace directrix_of_parabola_l318_31815

-- Define the equation of the parabola and what we need to prove
def parabola_equation (x : ℝ) : ℝ := 2 * x^2 + 6

-- Theorem stating the directrix of the given parabola
theorem directrix_of_parabola :
  ∀ x : ℝ, y = parabola_equation x → y = 47 / 8 := 
by
  sorry

end directrix_of_parabola_l318_31815


namespace inequality_solution_l318_31899

theorem inequality_solution (x : ℝ) : 
  (2 * x) / (x + 2) ≤ 3 ↔ x ∈ Set.Iic (-6) ∪ Set.Ioi (-2) :=
by
  sorry

end inequality_solution_l318_31899


namespace purchase_gifts_and_have_money_left_l318_31835

/-
  We start with 5000 forints in our pocket to buy gifts, visiting three stores.
  In each store, we find a gift that we like and purchase it if we have enough money. 
  The prices in each store are independently 1000, 1500, or 2000 forints, each with a probability of 1/3. 
  What is the probability that we can purchase gifts from all three stores 
  and still have money left (i.e., the total expenditure is at most 4500 forints)?
-/

def giftProbability (totalForints : ℕ) (prices : List ℕ) : ℚ :=
  let outcomes := prices |>.product prices |>.product prices
  let favorable := outcomes.filter (λ ((p1, p2), p3) => p1 + p2 + p3 <= totalForints)
  favorable.length / outcomes.length

theorem purchase_gifts_and_have_money_left :
  giftProbability 4500 [1000, 1500, 2000] = 17 / 27 :=
sorry

end purchase_gifts_and_have_money_left_l318_31835


namespace range_of_m_l318_31807

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then 2^x + 1 else 1 - Real.log (x) / Real.log 2

-- The problem is to find the range of m such that f(1 - m^2) > f(2m - 2). We assert the range of m as given in the correct answer.
theorem range_of_m : {m : ℝ | f (1 - m^2) > f (2 * m - 2)} = 
  {m : ℝ | -3 < m ∧ m < 1} ∪ {m : ℝ | m > 3 / 2} :=
sorry

end range_of_m_l318_31807


namespace total_hats_l318_31866

theorem total_hats (B G : ℕ) (cost_blue cost_green total_cost green_quantity : ℕ)
  (h1 : cost_blue = 6)
  (h2 : cost_green = 7)
  (h3 : total_cost = 530)
  (h4 : green_quantity = 20)
  (h5 : G = green_quantity)
  (h6 : total_cost = B * cost_blue + G * cost_green) :
  B + G = 85 :=
by
  sorry

end total_hats_l318_31866


namespace mat_radius_increase_l318_31841

theorem mat_radius_increase (C1 C2 : ℝ) (h1 : C1 = 40) (h2 : C2 = 50) :
  let r1 := C1 / (2 * Real.pi)
  let r2 := C2 / (2 * Real.pi)
  (r2 - r1) = 5 / Real.pi := by
  sorry

end mat_radius_increase_l318_31841


namespace problem_I_problem_II_l318_31814

open Set

variable (a x : ℝ)

def p : Prop := ∀ x ∈ Icc (1 : ℝ) 2, x^2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

theorem problem_I (hp : p a) : a ≤ 1 :=
  sorry

theorem problem_II (hpq : ¬ (p a ∧ q a)) : a ∈ Ioo (-2 : ℝ) (1 : ℝ) ∪ Ioi 1 :=
  sorry

end problem_I_problem_II_l318_31814


namespace sum_3n_terms_l318_31862

variable {a_n : ℕ → ℝ} -- Definition of the sequence
variable {S : ℕ → ℝ} -- Definition of the sum function

-- Conditions
axiom sum_n_terms (n : ℕ) : S n = 3
axiom sum_2n_terms (n : ℕ) : S (2 * n) = 15

-- Question and correct answer
theorem sum_3n_terms (n : ℕ) : S (3 * n) = 63 := 
sorry -- Proof to be provided

end sum_3n_terms_l318_31862


namespace cone_water_fill_percentage_l318_31816

noncomputable def volumeFilledPercentage (h r : ℝ) : ℝ :=
  let original_cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_cone_volume := (1 / 3) * Real.pi * ((2 / 3) * r)^2 * ((2 / 3) * h)
  let ratio := water_cone_volume / original_cone_volume
  ratio * 100


theorem cone_water_fill_percentage (h r : ℝ) :
  volumeFilledPercentage h r = 29.6296 :=
by
  sorry

end cone_water_fill_percentage_l318_31816


namespace find_m_l318_31819

open Set

def A : Set ℕ := {1, 3, 5}
def B (m : ℕ) : Set ℕ := {1, m}
def C (m : ℕ) : Set ℕ := {1, m}

theorem find_m (m : ℕ) (h : A ∩ B m = C m) : m = 3 ∨ m = 5 :=
sorry

end find_m_l318_31819


namespace rick_iron_clothing_l318_31858

theorem rick_iron_clothing :
  let shirts_per_hour := 4
  let pants_per_hour := 3
  let jackets_per_hour := 2
  let hours_shirts := 3
  let hours_pants := 5
  let hours_jackets := 2
  let total_clothing := (shirts_per_hour * hours_shirts) + (pants_per_hour * hours_pants) + (jackets_per_hour * hours_jackets)
  total_clothing = 31 := by
  sorry

end rick_iron_clothing_l318_31858


namespace interest_earned_l318_31803

theorem interest_earned (P : ℝ) (r : ℝ) (n : ℕ) (A : ℝ) : 
  P = 2000 → r = 0.05 → n = 5 → 
  A = P * (1 + r)^n → 
  A - P = 552.56 :=
by
  intro hP hr hn hA
  rw [hP, hr, hn] at hA
  sorry

end interest_earned_l318_31803


namespace rational_with_smallest_absolute_value_is_zero_l318_31851

theorem rational_with_smallest_absolute_value_is_zero (r : ℚ) :
  (forall r : ℚ, |r| ≥ 0) →
  (forall r : ℚ, r ≠ 0 → |r| > 0) →
  |r| = 0 ↔ r = 0 := sorry

end rational_with_smallest_absolute_value_is_zero_l318_31851


namespace geometric_sequence_b_value_l318_31833

theorem geometric_sequence_b_value (b : ℝ) 
  (h1 : ∃ r : ℝ, 30 * r = b ∧ b * r = 9 / 4)
  (h2 : b > 0) : b = 3 * Real.sqrt 30 :=
by
  sorry

end geometric_sequence_b_value_l318_31833


namespace fraction_equals_one_l318_31876

/-- Given the fraction (12-11+10-9+8-7+6-5+4-3+2-1) / (1-2+3-4+5-6+7-8+9-10+11),
    prove that its value is equal to 1. -/
theorem fraction_equals_one :
  (12 - 11 + 10 - 9 + 8 - 7 + 6 - 5 + 4 - 3 + 2 - 1) /
  (1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11) = 1 := by
  sorry

end fraction_equals_one_l318_31876


namespace max_profit_correctness_l318_31811

noncomputable def daily_purchase_max_profit := 
  let purchase_price := 4.2
  let selling_price := 6
  let return_price := 1.2
  let days_sold_10kg := 10
  let days_sold_6kg := 20
  let days_in_month := 30
  let profit_function (x : ℝ) := 
    10 * x * (selling_price - purchase_price) + 
    days_sold_6kg * 6 * (selling_price - purchase_price) + 
    days_sold_6kg * (x - 6) * (return_price - purchase_price)
  (6, profit_function 6)

theorem max_profit_correctness : daily_purchase_max_profit = (6, 324) :=
  sorry

end max_profit_correctness_l318_31811


namespace cos_pi_plus_alpha_l318_31885

-- Define the angle α and conditions given
variable (α : Real) (h1 : 0 < α) (h2 : α < π/2)

-- Given condition sine of α
variable (h3 : Real.sin α = 4/5)

-- Define the cosine identity to prove the assertion
theorem cos_pi_plus_alpha (h1 : 0 < α) (h2 : α < π/2) (h3 : Real.sin α = 4/5) :
  Real.cos (π + α) = -3/5 :=
sorry

end cos_pi_plus_alpha_l318_31885


namespace clothing_discounted_to_fraction_of_original_price_l318_31868

-- Given conditions
variable (P : ℝ) (f : ℝ)

-- Price during first sale is fP, price during second sale is 0.5P
-- Price decreased by 40% from first sale to second sale
def price_decrease_condition : Prop :=
  f * P - (1/2) * P = 0.4 * (f * P)

-- The main theorem to prove
theorem clothing_discounted_to_fraction_of_original_price (h : price_decrease_condition P f) :
  f = 5/6 :=
sorry

end clothing_discounted_to_fraction_of_original_price_l318_31868


namespace domain_of_v_l318_31894

noncomputable def v (x : ℝ) : ℝ := 1 / (x ^ (1/3) + x^2 - 1)

theorem domain_of_v : ∀ x, x ≠ 1 → x ^ (1/3) + x^2 - 1 ≠ 0 :=
by
  sorry

end domain_of_v_l318_31894


namespace eighth_binomial_term_l318_31817

theorem eighth_binomial_term :
  let n := 10
  let a := 2 * x
  let b := 1
  let k := 7
  (Nat.choose n k) * (a ^ k) * (b ^ (n - k)) = 960 * (x ^ 3) := by
  sorry

end eighth_binomial_term_l318_31817


namespace g_g_2_eq_394_l318_31864

def g (x : ℝ) : ℝ := 4 * x ^ 2 - 6

theorem g_g_2_eq_394 : g (g 2) = 394 :=
by
  sorry

end g_g_2_eq_394_l318_31864


namespace first_generation_tail_length_l318_31842

theorem first_generation_tail_length
  (length_first_gen : ℝ)
  (H : (1.25:ℝ) * (1.25:ℝ) * length_first_gen = 25) :
  length_first_gen = 16 := by
  sorry

end first_generation_tail_length_l318_31842


namespace pentagonal_tiles_count_l318_31893

theorem pentagonal_tiles_count (a b : ℕ) (h1 : a + b = 30) (h2 : 3 * a + 5 * b = 120) : b = 15 :=
by
  sorry

end pentagonal_tiles_count_l318_31893


namespace max_n_for_factoring_l318_31844

theorem max_n_for_factoring (n : ℤ) :
  (∃ A B : ℤ, (5 * B + A = n) ∧ (A * B = 90)) → n = 451 :=
by
  sorry

end max_n_for_factoring_l318_31844


namespace beverage_distribution_l318_31890

theorem beverage_distribution (total_cans : ℕ) (number_of_children : ℕ) (hcans : total_cans = 5) (hchildren : number_of_children = 8) :
  (total_cans / number_of_children : ℚ) = 5 / 8 :=
by
  -- Given the conditions
  have htotal_cans : total_cans = 5 := hcans
  have hnumber_of_children : number_of_children = 8 := hchildren
  
  -- we need to show the beverage distribution
  rw [htotal_cans, hnumber_of_children]
  exact by norm_num

end beverage_distribution_l318_31890


namespace exists_root_between_l318_31887

-- Given definitions and conditions
variables (a b c : ℝ)
variables (ha : a ≠ 0)
variables (x1 x2 : ℝ)
variable (h1 : a * x1^2 + b * x1 + c = 0)    -- root of the first equation
variable (h2 : -a * x2^2 + b * x2 + c = 0)   -- root of the second equation

-- Proof statement
theorem exists_root_between (a b c : ℝ) (ha : a ≠ 0) (x1 x2 : ℝ)
    (h1 : a * x1^2 + b * x1 + c = 0) (h2 : -a * x2^2 + b * x2 + c = 0) :
    ∃ x3 : ℝ, 
      (x1 ≤ x3 ∧ x3 ≤ x2 ∨ x1 ≥ x3 ∧ x3 ≥ x2) ∧ 
      (1 / 2 * a * x3^2 + b * x3 + c = 0) :=
sorry

end exists_root_between_l318_31887


namespace find_multiple_l318_31884

theorem find_multiple (x y m : ℕ) (h1 : y + x = 50) (h2 : y = m * x - 43) (h3 : y = 31) : m = 4 :=
by
  sorry

end find_multiple_l318_31884


namespace proof_age_gladys_l318_31850

-- Definitions of ages
def age_gladys : ℕ := 30
def age_lucas : ℕ := 5
def age_billy : ℕ := 10

-- Conditions
def condition1 : Prop := age_gladys = 2 * (age_billy + age_lucas)
def condition2 : Prop := age_gladys = 3 * age_billy
def condition3 : Prop := age_lucas + 3 = 8

-- Theorem to prove the correct age of Gladys
theorem proof_age_gladys (G L B : ℕ)
  (h1 : G = 2 * (B + L))
  (h2 : G = 3 * B)
  (h3 : L + 3 = 8) :
  G = 30 :=
sorry

end proof_age_gladys_l318_31850


namespace quadratic_inequality_l318_31870

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) 
  (h₁ : quadratic_function a b c 1 = quadratic_function a b c 3) 
  (h₂ : quadratic_function a b c 1 > quadratic_function a b c 4) : 
  a < 0 ∧ 4 * a + b = 0 :=
by
  sorry

end quadratic_inequality_l318_31870


namespace at_least_two_inequalities_hold_l318_31871

theorem at_least_two_inequalities_hold 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h : a + b + c ≥ a * b * c) : 
  (2 / a + 3 / b + 6 / c ≥ 6 ∨ 2 / b + 3 / c + 6 / a ≥ 6) ∨ (2 / b + 3 / c + 6 / a ≥ 6 ∨ 2 / c + 3 / a + 6 / b ≥ 6) ∨ (2 / c + 3 / a + 6 / b ≥ 6 ∨ 2 / a + 3 / b + 6 / c ≥ 6) := 
sorry

end at_least_two_inequalities_hold_l318_31871


namespace num_arrangement_options_l318_31827

def competition_events := ["kicking shuttlecocks", "jumping rope", "tug-of-war", "pushing the train", "multi-person multi-foot"]

def is_valid_arrangement (arrangement : List String) : Prop :=
  arrangement.length = 5 ∧
  arrangement.getLast? = some "tug-of-war" ∧
  arrangement.get? 0 ≠ some "multi-person multi-foot"

noncomputable def count_valid_arrangements : ℕ :=
  let positions := ["kicking shuttlecocks", "jumping rope", "pushing the train"]
  3 * positions.permutations.length

theorem num_arrangement_options : count_valid_arrangements = 18 :=
by
  sorry

end num_arrangement_options_l318_31827


namespace total_wet_surface_area_is_correct_l318_31877

noncomputable def wet_surface_area (cistern_length cistern_width water_depth platform_length platform_width platform_height : ℝ) : ℝ :=
  let two_longer_walls := 2 * (cistern_length * water_depth)
  let two_shorter_walls := 2 * (cistern_width * water_depth)
  let area_walls := two_longer_walls + two_shorter_walls
  let area_bottom := cistern_length * cistern_width
  let submerged_height := water_depth - platform_height
  let two_longer_sides_platform := 2 * (platform_length * submerged_height)
  let two_shorter_sides_platform := 2 * (platform_width * submerged_height)
  let area_platform_sides := two_longer_sides_platform + two_shorter_sides_platform
  area_walls + area_bottom + area_platform_sides

theorem total_wet_surface_area_is_correct :
  wet_surface_area 8 4 1.25 1 0.5 0.75 = 63.5 :=
by
  -- The proof goes here
  sorry

end total_wet_surface_area_is_correct_l318_31877


namespace zarnin_staffing_l318_31852

theorem zarnin_staffing (n total unsuitable : ℕ) (unsuitable_factor : ℕ) (job_openings : ℕ)
  (h1 : total = 30) 
  (h2 : unsuitable_factor = 2 / 3) 
  (h3 : unsuitable = unsuitable_factor * total) 
  (h4 : n = total - unsuitable)
  (h5 : job_openings = 5) :
  (n - 0) * (n - 1) * (n - 2) * (n - 3) * (n - 4) = 30240 := by
    sorry

end zarnin_staffing_l318_31852


namespace more_customers_after_lunch_rush_l318_31804

-- Definitions for conditions
def initial_customers : ℝ := 29.0
def added_customers : ℝ := 20.0
def total_customers : ℝ := 83.0

-- The number of additional customers that came in after the lunch rush
def additional_customers (initial additional total : ℝ) : ℝ :=
  total - (initial + additional)

-- Statement to prove
theorem more_customers_after_lunch_rush :
  additional_customers initial_customers added_customers total_customers = 34.0 :=
by
  sorry

end more_customers_after_lunch_rush_l318_31804


namespace average_speed_is_correct_l318_31847

namespace CyclistTrip

-- Define the trip parameters
def distance_north := 10 -- kilometers
def speed_north := 15 -- kilometers per hour
def rest_time := 10 / 60 -- hours
def distance_south := 10 -- kilometers
def speed_south := 20 -- kilometers per hour

-- The total trip distance
def total_distance := distance_north + distance_south -- kilometers

-- Calculate the time for each segment
def time_north := distance_north / speed_north -- hours
def time_south := distance_south / speed_south -- hours

-- Total time for the trip
def total_time := time_north + rest_time + time_south -- hours

-- Calculate the average speed
def average_speed := total_distance / total_time -- kilometers per hour

theorem average_speed_is_correct : average_speed = 15 := by
  sorry

end CyclistTrip

end average_speed_is_correct_l318_31847


namespace brenda_bought_stones_l318_31873

-- Given Conditions
def n_bracelets : ℕ := 3
def n_stones_per_bracelet : ℕ := 12

-- Problem Statement: Prove Betty bought the correct number of stone-shaped stars
theorem brenda_bought_stones :
  let n_total_stones := n_bracelets * n_stones_per_bracelet
  n_total_stones = 36 := 
by 
  -- proof goes here, but we omit it with sorry
  sorry

end brenda_bought_stones_l318_31873


namespace quadratic_ineq_real_solutions_l318_31889

theorem quadratic_ineq_real_solutions (d : ℝ) (h₀ : 0 < d) :
  (∀ x : ℝ, x^2 - 8 * x + d < 0 → 0 < d ∧ d < 16) :=
by
  sorry

end quadratic_ineq_real_solutions_l318_31889


namespace general_formula_constant_c_value_l318_31859

-- Definition of an arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) - a n = d

-- Given sequence {a_n}
variables {a : ℕ → ℝ} (S : ℕ → ℝ) (d : ℝ)
-- Conditions
variables (h1 : a 3 * a 4 = 117) (h2 : a 2 + a 5 = 22) (hd_pos : d > 0)
-- Proof that the general formula for the sequence {a_n} is a_n = 4n - 3
theorem general_formula :
  (∀ n, a n = 4 * n - 3) :=
sorry

-- Given new sequence {b_n}
variables (b : ℕ → ℕ → ℝ) {c : ℝ} (hc : c ≠ 0)
-- New condition that bn is an arithmetic sequence
variables (h_b1 : b 1 = S 1 / (1 + c)) (h_b2 : b 2 = S 2 / (2 + c)) (h_b3 : b 3 = S 3 / (3 + c))
-- Proof that c = -1/2 is the correct constant
theorem constant_c_value :
  (c = -1 / 2) :=
sorry

end general_formula_constant_c_value_l318_31859


namespace train_passes_jogger_in_37_seconds_l318_31872

-- Define the parameters
def jogger_speed_kmph : ℝ := 9
def train_speed_kmph : ℝ := 45
def headstart : ℝ := 250
def train_length : ℝ := 120

-- Convert speeds from km/h to m/s
noncomputable def jogger_speed_mps : ℝ := jogger_speed_kmph * 1000 / 3600
noncomputable def train_speed_mps : ℝ := train_speed_kmph * 1000 / 3600

-- Calculate relative speed in m/s
noncomputable def relative_speed : ℝ :=
  train_speed_mps - jogger_speed_mps

-- Calculate total distance to be covered in meters
def total_distance : ℝ :=
  headstart + train_length

-- Calculate time taken to pass the jogger in seconds
noncomputable def time_to_pass : ℝ :=
  total_distance / relative_speed

theorem train_passes_jogger_in_37_seconds :
  time_to_pass = 37 :=
by
  -- Proof would be here
  sorry

end train_passes_jogger_in_37_seconds_l318_31872


namespace min_value_exp_l318_31881

theorem min_value_exp (a b : ℝ) (h_condition : a - 3 * b + 6 = 0) : 
  ∃ (m : ℝ), m = 2^a + 1 / 8^b ∧ m ≥ (1 / 4) :=
by
  sorry

end min_value_exp_l318_31881


namespace equal_vectors_implies_collinear_l318_31832

-- Definitions for vectors and their properties
variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def collinear (u v : V) : Prop := ∃ (a : ℝ), v = a • u 

def equal_vectors (u v : V) : Prop := u = v

theorem equal_vectors_implies_collinear (u v : V)
  (h : equal_vectors u v) : collinear u v :=
by sorry

end equal_vectors_implies_collinear_l318_31832


namespace rectangle_area_from_perimeter_l318_31810

theorem rectangle_area_from_perimeter
  (a : ℝ)
  (shorter_side := 12 * a)
  (longer_side := 22 * a)
  (P := 2 * (shorter_side + longer_side))
  (hP : P = 102) :
  (shorter_side * longer_side = 594) := by
  sorry

end rectangle_area_from_perimeter_l318_31810


namespace avg_speed_including_stoppages_l318_31826

theorem avg_speed_including_stoppages (speed_without_stoppages : ℝ) (stoppage_time_per_hour : ℝ) 
  (h₁ : speed_without_stoppages = 60) (h₂ : stoppage_time_per_hour = 0.5) : 
  (speed_without_stoppages * (1 - stoppage_time_per_hour)) / 1 = 30 := 
  by 
  sorry

end avg_speed_including_stoppages_l318_31826


namespace calculate_expression_l318_31849

-- Theorem statement for the provided problem
theorem calculate_expression :
  ((18 ^ 15 / 18 ^ 14)^3 * 8 ^ 3) / 4 ^ 5 = 2916 := by
  sorry

end calculate_expression_l318_31849


namespace pascal_triangle_43rd_element_in_51_row_l318_31857

theorem pascal_triangle_43rd_element_in_51_row :
  (Nat.choose 50 42) = 10272278170 :=
  by
  -- proof construction here
  sorry

end pascal_triangle_43rd_element_in_51_row_l318_31857


namespace sum_of_coordinates_A_l318_31834

theorem sum_of_coordinates_A (a b : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∃ x y : ℝ, y = a * x + 4 ∧ y = 2 * x + b ∧ y = (a / 2) * x + 8) :
  ∃ y : ℝ, y = 13 ∨ y = 20 :=
by
  sorry

end sum_of_coordinates_A_l318_31834


namespace partitions_distinct_parts_eq_odd_parts_l318_31854

def num_partitions_into_distinct_parts (n : ℕ) : ℕ := sorry
def num_partitions_into_odd_parts (n : ℕ) : ℕ := sorry

theorem partitions_distinct_parts_eq_odd_parts (n : ℕ) :
  num_partitions_into_distinct_parts n = num_partitions_into_odd_parts n :=
  sorry

end partitions_distinct_parts_eq_odd_parts_l318_31854


namespace range_of_a_l318_31874

def A (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 5
def B (x : ℝ) (a : ℝ) : Prop := x > a

theorem range_of_a (a : ℝ) : (∀ x : ℝ, A x → B x a) → a < -2 :=
by
  sorry

end range_of_a_l318_31874


namespace incorrect_conclusion_l318_31886

theorem incorrect_conclusion (p q : ℝ) (h1 : p < 0) (h2 : q < 0) : ¬ ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x3 ≠ x1 ∧ (x1 * |x1| + p * x1 + q = 0) ∧ (x2 * |x2| + p * x2 + q = 0) ∧ (x3 * |x3| + p * x3 + q = 0) :=
by
  sorry

end incorrect_conclusion_l318_31886


namespace tan_alpha_plus_pi_over_4_sin_2alpha_expr_l318_31878

open Real

theorem tan_alpha_plus_pi_over_4 (α : ℝ) (h : tan α = 2) : tan (α + π / 4) = -3 :=
by
  sorry

theorem sin_2alpha_expr (α : ℝ) (h : tan α = 2) :
  (sin (2 * α)) / (sin (α) ^ 2 + sin (α) * cos (α)) = 2 / 3 :=
by
  sorry

end tan_alpha_plus_pi_over_4_sin_2alpha_expr_l318_31878


namespace limit_of_sequence_l318_31823

theorem limit_of_sequence (a_n : ℕ → ℝ) (a : ℝ) :
  (∀ n : ℕ, a_n n = (2 * (n ^ 3)) / ((n ^ 3) - 2)) →
  a = 2 →
  (∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |a_n n - a| < ε) :=
by
  intros h1 h2 ε hε
  sorry

end limit_of_sequence_l318_31823


namespace int_pairs_satisfy_eq_l318_31848

theorem int_pairs_satisfy_eq (x y : ℤ) : (x^2 = y^2 + 2 * y + 13) ↔ ((x = 4 ∧ y = 1) ∨ (x = -4 ∧ y = -5)) :=
by 
  sorry

end int_pairs_satisfy_eq_l318_31848


namespace proposition_equivalence_l318_31856

open Classical

theorem proposition_equivalence
  (p q : Prop) :
  ¬(p ∨ q) ↔ (¬p ∧ ¬q) :=
by sorry

end proposition_equivalence_l318_31856


namespace inequality_holds_l318_31813

theorem inequality_holds (a b : ℝ) (h : a < b) (h₀ : b < 0) : - (1 / a) < - (1 / b) :=
sorry

end inequality_holds_l318_31813


namespace doug_initial_marbles_l318_31896

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

end doug_initial_marbles_l318_31896


namespace roots_of_polynomial_l318_31808

theorem roots_of_polynomial : 
  (∀ x : ℝ, (x^3 - 6*x^2 + 11*x - 6) * (x - 2) = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) :=
by
  intro x
  sorry

end roots_of_polynomial_l318_31808


namespace mary_saw_total_snakes_l318_31800

theorem mary_saw_total_snakes :
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  totalSnakes = 36 :=
by
  /- Definitions -/ 
  let breedingBalls := 3
  let snakesPerBall := 8
  let pairsOfSnakes := 6
  let snakesPerPair := 2
  let totalSnakes := breedingBalls * snakesPerBall + pairsOfSnakes * snakesPerPair
  /- Main proof statement -/
  show totalSnakes = 36
  sorry

end mary_saw_total_snakes_l318_31800


namespace winding_clock_available_time_l318_31837

theorem winding_clock_available_time
    (minute_hand_restriction_interval: ℕ := 5) -- Each interval the minute hand restricts
    (hour_hand_restriction_interval: ℕ := 60) -- Each interval the hour hand restricts
    (intervals_per_12_hours: ℕ := 2) -- Number of restricted intervals in each 12-hour cycle
    (minutes_in_day: ℕ := 24 * 60) -- Total minutes in 24 hours
    : (minutes_in_day - ((minute_hand_restriction_interval * intervals_per_12_hours * 12) + 
                         (hour_hand_restriction_interval * intervals_per_12_hours * 2))) = 1080 :=
by
  -- Skipping the proof steps
  sorry

end winding_clock_available_time_l318_31837


namespace remainder_when_divided_by_x_plus_2_l318_31822

-- Define the polynomial q(x)
def q (M N D x : ℝ) : ℝ := M * x^4 + N * x^2 + D * x - 5

-- Define the given conditions
def cond1 (M N D : ℝ) : Prop := q M N D 2 = 15

-- The theorem statement we want to prove
theorem remainder_when_divided_by_x_plus_2 (M N D : ℝ) (h1 : cond1 M N D) : q M N D (-2) = 15 :=
sorry

end remainder_when_divided_by_x_plus_2_l318_31822


namespace find_numbers_with_conditions_l318_31892

theorem find_numbers_with_conditions (n : ℕ) (hn1 : n % 100 = 0) (hn2 : (n.divisors).card = 12) : 
  n = 200 ∨ n = 500 :=
by
  sorry

end find_numbers_with_conditions_l318_31892


namespace length_of_train_l318_31840

theorem length_of_train (V L : ℝ) (h1 : L = V * 18) (h2 : L + 250 = V * 33) : L = 300 :=
by
  sorry

end length_of_train_l318_31840


namespace packs_of_red_balls_l318_31891

/-
Julia bought some packs of red balls, R packs.
Julia bought 10 packs of yellow balls.
Julia bought 8 packs of green balls.
There were 19 balls in each package.
Julia bought 399 balls in total.
The goal is to prove that the number of packs of red balls Julia bought, R, is equal to 3.
-/

theorem packs_of_red_balls (R : ℕ) (balls_per_pack : ℕ) (packs_yellow : ℕ) (packs_green : ℕ) (total_balls : ℕ) 
  (h1 : balls_per_pack = 19) (h2 : packs_yellow = 10) (h3 : packs_green = 8) (h4 : total_balls = 399) 
  (h5 : total_balls = R * balls_per_pack + (packs_yellow + packs_green) * balls_per_pack) : 
  R = 3 :=
by
  -- Proof goes here
  sorry

end packs_of_red_balls_l318_31891


namespace greatest_possible_x_for_equation_l318_31853

theorem greatest_possible_x_for_equation :
  ∃ x, (x = (9 : ℝ) / 5) ∧ 
  ((5 * x - 20) / (4 * x - 5))^2 + ((5 * x - 20) / (4 * x - 5)) = 20 := by
  sorry

end greatest_possible_x_for_equation_l318_31853


namespace ticket_cost_l318_31888

noncomputable def calculate_cost (x : ℝ) : ℝ :=
  6 * (1.1 * x) + 5 * (x / 2)

theorem ticket_cost (x : ℝ) (h : 4 * (1.1 * x) + 3 * (x / 2) = 28.80) : 
  calculate_cost x = 44.41 := by
  sorry

end ticket_cost_l318_31888


namespace inequality_proof_l318_31863

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b + b * c + c * a ≥ 1) :
    (1 / a^2) + (1 / b^2) + (1 / c^2) ≥ (Real.sqrt 3) / (a * b * c) :=
by
  sorry

end inequality_proof_l318_31863


namespace pentagon_diagonals_l318_31830

def number_of_sides_pentagon : ℕ := 5
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem pentagon_diagonals : number_of_diagonals number_of_sides_pentagon = 5 := by
  sorry

end pentagon_diagonals_l318_31830


namespace tan_quadruple_angle_l318_31895

theorem tan_quadruple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (4 * θ) = -24 / 7 :=
sorry

end tan_quadruple_angle_l318_31895


namespace min_x_plus_4y_min_value_l318_31836

noncomputable def min_x_plus_4y (x y: ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(2 * y) = 1) : ℝ :=
  x + 4 * y

theorem min_x_plus_4y_min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 1/(2 * y) = 1) :
  min_x_plus_4y x y hx hy h = 3 + 2 * Real.sqrt 2 :=
sorry

end min_x_plus_4y_min_value_l318_31836


namespace find_g_inv_neg_fifteen_sixtyfour_l318_31860

noncomputable def g (x : ℝ) : ℝ := (x^6 - 1) / 4

theorem find_g_inv_neg_fifteen_sixtyfour : g⁻¹ (-15/64) = 1/2 :=
by
  sorry  -- Proof is not required

end find_g_inv_neg_fifteen_sixtyfour_l318_31860


namespace pounds_of_beef_l318_31821

theorem pounds_of_beef (meals_price : ℝ) (total_sales : ℝ) (meat_per_meal : ℝ) (relationship : ℝ) (total_meat_used : ℝ) (beef_pounds : ℝ) :
  (total_sales = 400) → (meals_price = 20) → (meat_per_meal = 1.5) → (relationship = 0.5) → (20 * meals_price = total_sales) → (total_meat_used = 30) →
  (beef_pounds + beef_pounds * relationship = total_meat_used) → beef_pounds = 20 :=
by
  intros
  sorry

end pounds_of_beef_l318_31821


namespace sequence_statements_correct_l318_31845

theorem sequence_statements_correct (S : ℕ → ℝ) (a : ℕ → ℝ) (T : ℕ → ℝ) 
(h_S_nonzero : ∀ n, n > 0 → S n ≠ 0)
(h_S_T_relation : ∀ n, n > 0 → S n + T n = S n * T n) :
  (a 1 = 2) ∧ (∀ n, n > 0 → T n - T (n - 1) = 1) ∧ (∀ n, n > 0 → S n = (n + 1) / n) :=
by
  sorry

end sequence_statements_correct_l318_31845


namespace solve_for_x_l318_31831

theorem solve_for_x (x : ℝ) (h : (4/7) * (2/5) * x = 8) : x = 35 :=
sorry

end solve_for_x_l318_31831


namespace abs_z_bounds_l318_31843

open Complex

theorem abs_z_bounds (z : ℂ) (h : abs (z + 1/z) = 1) : 
  (Real.sqrt 5 - 1) / 2 ≤ abs z ∧ abs z ≤ (Real.sqrt 5 + 1) / 2 := 
sorry

end abs_z_bounds_l318_31843


namespace num_two_digit_prime_with_units_digit_3_eq_6_l318_31875

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l318_31875


namespace students_remaining_l318_31812

theorem students_remaining (students_showed_up : ℕ) (students_checked_out : ℕ) (students_left : ℕ) :
  students_showed_up = 16 → students_checked_out = 7 → students_left = students_showed_up - students_checked_out → students_left = 9 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end students_remaining_l318_31812


namespace tangent_sum_l318_31883

theorem tangent_sum :
  (Finset.sum (Finset.range 2019) (λ k => Real.tan ((k + 1) * Real.pi / 47) * Real.tan ((k + 2) * Real.pi / 47))) = -2021 :=
by
  -- proof will be completed here
  sorry

end tangent_sum_l318_31883


namespace isosceles_triangle_l318_31838

def shape_of_triangle (A B C : Real) (h : 2 * Real.sin A * Real.cos B = Real.sin C) : Prop :=
  A = B

theorem isosceles_triangle {A B C : Real} (h : 2 * Real.sin A * Real.cos B = Real.sin C) :
  shape_of_triangle A B C h := 
  sorry

end isosceles_triangle_l318_31838


namespace find_smaller_number_l318_31898

theorem find_smaller_number (x : ℕ) (hx : x + 4 * x = 45) : x = 9 :=
by
  sorry

end find_smaller_number_l318_31898


namespace triangle_perimeter_l318_31879

-- Given conditions
def inradius : ℝ := 2.5
def area : ℝ := 40

-- The formula relating inradius, area, and perimeter
def perimeter_formula (r a p : ℝ) : Prop := a = r * p / 2

-- Prove the perimeter p of the triangle
theorem triangle_perimeter : ∃ (p : ℝ), perimeter_formula inradius area p ∧ p = 32 := by
  sorry

end triangle_perimeter_l318_31879


namespace find_wall_width_l318_31861

-- Define the dimensions of the brick in meters
def brick_length : ℝ := 0.20
def brick_width : ℝ := 0.1325
def brick_height : ℝ := 0.08

-- Define the dimensions of the wall in meters
def wall_length : ℝ := 7
def wall_height : ℝ := 15.5
def number_of_bricks : ℝ := 4094.3396226415093

-- Volume of one brick
def brick_volume : ℝ := brick_length * brick_width * brick_height

-- Total volume of bricks used
def total_brick_volume : ℝ := number_of_bricks * brick_volume

-- Wall volume in terms of width W
def wall_volume (W : ℝ) : ℝ := wall_length * W * wall_height

-- The theorem we want to prove
theorem find_wall_width (W : ℝ) (h : wall_volume W = total_brick_volume) : W = 0.08 := by
  sorry

end find_wall_width_l318_31861


namespace range_of_m_l318_31828

-- Definitions of propositions
def is_circle (m : ℝ) : Prop :=
  ∃ x y : ℝ, (x - m)^2 + y^2 = 2 * m - m^2 ∧ 2 * m - m^2 > 0

def is_hyperbola_eccentricity_in_interval (m : ℝ) : Prop :=
  1 < Real.sqrt (1 + m / 5) ∧ Real.sqrt (1 + m / 5) < 2

-- Proving the main statement
theorem range_of_m (m : ℝ) (h1 : is_circle m ∨ is_hyperbola_eccentricity_in_interval m)
  (h2 : ¬ (is_circle m ∧ is_hyperbola_eccentricity_in_interval m)) : 2 ≤ m ∧ m < 15 :=
sorry

end range_of_m_l318_31828


namespace gerbils_left_l318_31825

theorem gerbils_left (initial count sold : ℕ) (h_initial : count = 85) (h_sold : sold = 69) : 
  count - sold = 16 := 
by 
  sorry

end gerbils_left_l318_31825


namespace parabola_functions_eq_l318_31818

noncomputable def f (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := x^2 + b * x + c
noncomputable def g (x : ℝ) (c : ℝ) (b : ℝ) : ℝ := x^2 + c * x + b

theorem parabola_functions_eq : ∀ (x₁ x₂ : ℝ), 
  (∃ t : ℝ, (f t b c = g t c b) ∧ (t = 1)) → 
    (f x₁ 2 (-3) = x₁^2 + 2 * x₁ - 3) ∧ (g x₂ (-3) 2 = x₂^2 - 3 * x₂ + 2) :=
sorry

end parabola_functions_eq_l318_31818
