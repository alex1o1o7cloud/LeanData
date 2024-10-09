import Mathlib

namespace chipmunk_families_went_away_l1820_182055

theorem chipmunk_families_went_away :
  ∀ (total_families left_families went_away_families : ℕ),
  total_families = 86 →
  left_families = 21 →
  went_away_families = total_families - left_families →
  went_away_families = 65 :=
by
  intros total_families left_families went_away_families ht hl hw
  rw [ht, hl] at hw
  exact hw

end chipmunk_families_went_away_l1820_182055


namespace find_minimum_a_l1820_182080

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x

theorem find_minimum_a (a : ℝ) :
  (∀ x, 1 ≤ x → 0 ≤ 3 * x^2 + a) ↔ a ≥ -3 :=
by
  sorry

end find_minimum_a_l1820_182080


namespace inequality_proof_l1820_182085

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_prod : a * b * c = 1) :
  (2 * a^2) / (1 + a + a * b)^2 + (2 * b^2) / (1 + b + b * c)^2 + (2 * c^2) / (1 + c + c * a)^2 +
  9 / ((1 + a + a * b) * (1 + b + b * c) * (1 + c + c * a)) ≥ 1 :=
by {
  sorry -- The proof goes here
}

end inequality_proof_l1820_182085


namespace Oates_reunion_l1820_182097

-- Declare the conditions as variables
variables (total_guests both_reunions yellow_reunion : ℕ)
variables (H1 : total_guests = 100)
variables (H2 : both_reunions = 7)
variables (H3 : yellow_reunion = 65)

-- The proof problem statement
theorem Oates_reunion (O : ℕ) (H4 : total_guests = O + yellow_reunion - both_reunions) : O = 42 :=
sorry

end Oates_reunion_l1820_182097


namespace six_times_more_coats_l1820_182072

/-- The number of lab coats is 6 times the number of uniforms. --/
def coats_per_uniforms (c u : ℕ) : Prop := c = 6 * u

/-- There are 12 uniforms. --/
def uniforms : ℕ := 12

/-- Each lab tech gets 14 coats and uniforms in total. --/
def total_per_tech : ℕ := 14

/-- Show that the number of lab coats is 6 times the number of uniforms. --/
theorem six_times_more_coats (c u : ℕ) (h1 : coats_per_uniforms c u) (h2 : u = 12) :
  c / u = 6 :=
by
  sorry

end six_times_more_coats_l1820_182072


namespace value_in_box_l1820_182014

theorem value_in_box (x : ℤ) (h : 5 + x = 10 + 20) : x = 25 := by
  sorry

end value_in_box_l1820_182014


namespace each_monkey_gets_bananas_l1820_182081

-- Define the conditions
def total_monkeys : ℕ := 12
def total_piles : ℕ := 10
def first_piles : ℕ := 6
def first_pile_hands : ℕ := 9
def first_hand_bananas : ℕ := 14
def remaining_piles : ℕ := total_piles - first_piles
def remaining_pile_hands : ℕ := 12
def remaining_hand_bananas : ℕ := 9

-- Define the number of bananas in each type of pile
def bananas_in_first_piles : ℕ := first_piles * first_pile_hands * first_hand_bananas
def bananas_in_remaining_piles : ℕ := remaining_piles * remaining_pile_hands * remaining_hand_bananas
def total_bananas : ℕ := bananas_in_first_piles + bananas_in_remaining_piles

-- Define the main theorem to be proved
theorem each_monkey_gets_bananas : total_bananas / total_monkeys = 99 := by
  sorry

end each_monkey_gets_bananas_l1820_182081


namespace milkman_pure_milk_l1820_182074

theorem milkman_pure_milk (x : ℝ) 
  (h_cost : 3.60 * x = 3 * (x + 5)) : x = 25 :=
  sorry

end milkman_pure_milk_l1820_182074


namespace sum_of_lengths_of_edges_geometric_progression_l1820_182058

theorem sum_of_lengths_of_edges_geometric_progression :
  ∃ (a r : ℝ), (a / r) * a * (a * r) = 8 ∧ 2 * (a / r * a + a * a * r + a * r * a / r) = 32 ∧ 
  4 * ((a / r) + a + (a * r)) = 32 :=
by
  sorry

end sum_of_lengths_of_edges_geometric_progression_l1820_182058


namespace p_sufficient_not_necessary_for_q_l1820_182012

-- Define the propositions p and q
def is_ellipse (m : ℝ) : Prop := (1 / 4 < m) ∧ (m < 1)
def is_hyperbola (m : ℝ) : Prop := (0 < m) ∧ (m < 1)

-- Define the theorem to prove the relationship between p and q
theorem p_sufficient_not_necessary_for_q (m : ℝ) :
  (is_ellipse m → is_hyperbola m) ∧ ¬(is_hyperbola m → is_ellipse m) :=
sorry

end p_sufficient_not_necessary_for_q_l1820_182012


namespace problem_solution_l1820_182073

theorem problem_solution :
  3 ^ (0 ^ (2 ^ 2)) + ((3 ^ 1) ^ 0) ^ 2 = 2 :=
by
  sorry

end problem_solution_l1820_182073


namespace monotonic_solution_l1820_182033

-- Definition of a monotonic function
def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

-- The main theorem
theorem monotonic_solution (f : ℝ → ℝ) 
  (mon : monotonic f) 
  (h : ∀ x y : ℝ, f (f x - y) + f (x + y) = 0) : 
  (∀ x, f x = 0) ∨ (∀ x, f x = -x) :=
sorry

end monotonic_solution_l1820_182033


namespace arrangement_count_l1820_182035

/-- April has five different basil plants and five different tomato plants. --/
def basil_plants : ℕ := 5
def tomato_plants : ℕ := 5

/-- All tomato plants must be placed next to each other. --/
def tomatoes_next_to_each_other := true

/-- The row must start with a basil plant. --/
def starts_with_basil := true

/-- The number of ways to arrange the plants in a row under the given conditions is 11520. --/
theorem arrangement_count :
  basil_plants = 5 ∧ tomato_plants = 5 ∧ tomatoes_next_to_each_other ∧ starts_with_basil → 
  ∃ arrangements : ℕ, arrangements = 11520 :=
by 
  sorry

end arrangement_count_l1820_182035


namespace length_of_unfenced_side_l1820_182089

theorem length_of_unfenced_side :
  ∃ L W : ℝ, L * W = 320 ∧ 2 * W + L = 56 ∧ L = 40 :=
by
  sorry

end length_of_unfenced_side_l1820_182089


namespace car_speeds_l1820_182005

theorem car_speeds (u v w : ℝ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) :
  (3 / (1 / u + 1 / v + 1 / w)) ≤ ((u + v) / 2) :=
sorry

end car_speeds_l1820_182005


namespace pounds_of_fudge_sold_l1820_182000

variable (F : ℝ)
variable (price_fudge price_truffles price_pretzels total_revenue : ℝ)

def conditions := 
  price_fudge = 2.50 ∧
  price_truffles = 60 * 1.50 ∧
  price_pretzels = 36 * 2.00 ∧
  total_revenue = 212 ∧
  total_revenue = (price_fudge * F) + price_truffles + price_pretzels

theorem pounds_of_fudge_sold (F : ℝ) (price_fudge price_truffles price_pretzels total_revenue : ℝ) 
  (h : conditions F price_fudge price_truffles price_pretzels total_revenue ) :
  F = 20 :=
by
  sorry

end pounds_of_fudge_sold_l1820_182000


namespace part1_part2_part3_l1820_182067

def folklore {a b m n : ℤ} (h1 : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2) : Prop :=
  a = m ^ 2 + 3 * n ^ 2 ∧ b = 2 * m * n

theorem part1 (a b m n : ℤ) (h : a + b * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2) : a = m ^ 2 + 3 * n ^ 2 ∧ b = 2 * m * n :=
by sorry

theorem part2 : 13 + 4 * Real.sqrt 3 = (1 + 2 * Real.sqrt 3) ^ 2 :=
by sorry

theorem part3 (a m n : ℤ) (h1 : 4 = 2 * m * n) (h2 : a + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3) ^ 2) : a = 7 ∨ a = 13 :=
by sorry

end part1_part2_part3_l1820_182067


namespace rectangle_area_eq_l1820_182095

theorem rectangle_area_eq (d : ℝ) (w : ℝ) (h1 : w = d / (2 * (5 : ℝ) ^ (1/2))) (h2 : 3 * w = (3 * d) / (2 * (5 : ℝ) ^ (1/2))) : 
  (3 * w^2) = (3 / 10) * d^2 := 
by sorry

end rectangle_area_eq_l1820_182095


namespace undefined_sum_slope_y_intercept_of_vertical_line_l1820_182082

theorem undefined_sum_slope_y_intercept_of_vertical_line :
  ∀ (C D : ℝ × ℝ), C.1 = 8 → D.1 = 8 → C.2 ≠ D.2 →
  ∃ (m b : ℝ), false :=
by
  intros
  sorry

end undefined_sum_slope_y_intercept_of_vertical_line_l1820_182082


namespace base12_remainder_l1820_182088

def base12_to_base10 (a : ℕ) (b : ℕ) (c : ℕ) (d : ℕ) : ℕ :=
  a * 12^3 + b * 12^2 + c * 12^1 + d * 12^0

theorem base12_remainder (a b c d : ℕ) 
  (h1531 : base12_to_base10 a b c d = 1 * 12^3 + 5 * 12^2 + 3 * 12^1 + 1 * 12^0):
  (base12_to_base10 a b c d) % 8 = 5 :=
by
  unfold base12_to_base10 at h1531
  sorry

end base12_remainder_l1820_182088


namespace bankers_gain_correct_l1820_182087

def PW : ℝ := 600
def R : ℝ := 0.10
def n : ℕ := 2

def A : ℝ := PW * (1 + R)^n
def BG : ℝ := A - PW

theorem bankers_gain_correct : BG = 126 :=
by
  sorry

end bankers_gain_correct_l1820_182087


namespace jacket_cost_l1820_182017

noncomputable def cost_of_shorts : ℝ := 13.99
noncomputable def cost_of_shirt : ℝ := 12.14
noncomputable def total_spent : ℝ := 33.56
noncomputable def cost_of_jacket : ℝ := total_spent - (cost_of_shorts + cost_of_shirt)

theorem jacket_cost : cost_of_jacket = 7.43 := by
  sorry

end jacket_cost_l1820_182017


namespace dani_pants_after_5_years_l1820_182057

theorem dani_pants_after_5_years :
  ∀ (pairs_per_year : ℕ) (pants_per_pair : ℕ) (initial_pants : ℕ) (years : ℕ),
  pairs_per_year = 4 →
  pants_per_pair = 2 →
  initial_pants = 50 →
  years = 5 →
  initial_pants + years * (pairs_per_year * pants_per_pair) = 90 :=
by sorry

end dani_pants_after_5_years_l1820_182057


namespace find_abc_unique_solution_l1820_182021

theorem find_abc_unique_solution (N a b c : ℕ) 
  (hN : N > 3 ∧ N % 2 = 1)
  (h_eq : a^N = b^N + 2^N + a * b * c)
  (h_c : c ≤ 5 * 2^(N-1)) : 
  N = 5 ∧ a = 3 ∧ b = 1 ∧ c = 70 := 
sorry

end find_abc_unique_solution_l1820_182021


namespace max_value_of_function_l1820_182086

theorem max_value_of_function (x : ℝ) (h : 0 < x ∧ x < 1.5) : 
  ∃ m, ∀ y, y = 4 * x * (3 - 2 * x) → m = 9 / 2 :=
sorry

end max_value_of_function_l1820_182086


namespace no_consecutive_primes_sum_65_l1820_182071

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def consecutive_primes (p q : ℕ) : Prop := 
  is_prime p ∧ is_prime q ∧ (q = p + 2 ∨ q = p - 2)

theorem no_consecutive_primes_sum_65 : 
  ¬ ∃ p q : ℕ, consecutive_primes p q ∧ p + q = 65 :=
by 
  sorry

end no_consecutive_primes_sum_65_l1820_182071


namespace sales_tax_paid_l1820_182018

theorem sales_tax_paid 
  (total_spent : ℝ) 
  (tax_free_cost : ℝ) 
  (tax_rate : ℝ) 
  (cost_of_taxable_items : ℝ) 
  (sales_tax : ℝ) 
  (h1 : total_spent = 40) 
  (h2 : tax_free_cost = 34.7) 
  (h3 : tax_rate = 0.06) 
  (h4 : cost_of_taxable_items = 5) 
  (h5 : sales_tax = 0.3) 
  (h6 : 1.06 * cost_of_taxable_items + tax_free_cost = total_spent) : 
  sales_tax = tax_rate * cost_of_taxable_items :=
sorry

end sales_tax_paid_l1820_182018


namespace sum_six_terms_l1820_182031

variable (S : ℕ → ℝ)
variable (n : ℕ)
variable (S_2 S_4 S_6 : ℝ)

-- Given conditions
axiom sum_two_terms : S 2 = 4
axiom sum_four_terms : S 4 = 16

-- Problem statement
theorem sum_six_terms : S 6 = 52 :=
by
  -- Insert the proof here
  sorry

end sum_six_terms_l1820_182031


namespace interest_rate_first_part_l1820_182036

theorem interest_rate_first_part (A A1 A2 I : ℝ) (r : ℝ) :
  A = 3200 →
  A1 = 800 →
  A2 = A - A1 →
  I = 144 →
  (A1 * r / 100 + A2 * 5 / 100 = I) →
  r = 3 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end interest_rate_first_part_l1820_182036


namespace green_toads_per_acre_l1820_182075

theorem green_toads_per_acre (brown_toads spotted_brown_toads green_toads : ℕ) 
  (h1 : ∀ g, 25 * g = brown_toads) 
  (h2 : spotted_brown_toads = brown_toads / 4) 
  (h3 : spotted_brown_toads = 50) : 
  green_toads = 8 :=
by
  sorry

end green_toads_per_acre_l1820_182075


namespace prime_divisor_of_form_l1820_182065

theorem prime_divisor_of_form (a p : ℕ) (hp1 : a > 0) (hp2 : Prime p) (hp3 : p ∣ (a^3 - 3 * a + 1)) (hp4 : p ≠ 3) :
  ∃ k : ℤ, p = 9 * k + 1 ∨ p = 9 * k - 1 :=
by
  sorry

end prime_divisor_of_form_l1820_182065


namespace Brad_age_l1820_182016

theorem Brad_age (shara_age : ℕ) (h_shara : shara_age = 10)
  (jaymee_age : ℕ) (h_jaymee : jaymee_age = 2 * shara_age + 2)
  (brad_age : ℕ) (h_brad : brad_age = (shara_age + jaymee_age) / 2 - 3) : brad_age = 13 := by
  sorry

end Brad_age_l1820_182016


namespace cones_to_cylinder_volume_ratio_l1820_182048

theorem cones_to_cylinder_volume_ratio :
  let π := Real.pi
  let r_cylinder := 4
  let h_cylinder := 18
  let r_cone := 4
  let h_cone1 := 6
  let h_cone2 := 9
  let V_cylinder := π * r_cylinder^2 * h_cylinder
  let V_cone1 := (1 / 3) * π * r_cone^2 * h_cone1
  let V_cone2 := (1 / 3) * π * r_cone^2 * h_cone2
  let V_totalCones := V_cone1 + V_cone2
  V_totalCones / V_cylinder = 5 / 18 :=
by
  sorry

end cones_to_cylinder_volume_ratio_l1820_182048


namespace reciprocal_of_neg3_l1820_182006

theorem reciprocal_of_neg3 : ∃ x : ℝ, -3 * x = 1 ∧ x = -1/3 :=
by
  sorry

end reciprocal_of_neg3_l1820_182006


namespace elmer_saving_percent_l1820_182038

theorem elmer_saving_percent (x c : ℝ) (hx : x > 0) (hc : c > 0) :
  let old_car_fuel_efficiency := x
  let new_car_fuel_efficiency := 1.6 * x
  let gasoline_cost := c
  let diesel_cost := 1.25 * c
  let trip_distance := 300
  let old_car_fuel_needed := trip_distance / old_car_fuel_efficiency
  let new_car_fuel_needed := trip_distance / new_car_fuel_efficiency
  let old_car_cost := old_car_fuel_needed * gasoline_cost
  let new_car_cost := new_car_fuel_needed * diesel_cost
  let cost_saving := old_car_cost - new_car_cost
  let percent_saving := (cost_saving / old_car_cost) * 100
  percent_saving = 21.875 :=
by
  sorry

end elmer_saving_percent_l1820_182038


namespace remainder_1235678_div_127_l1820_182008

theorem remainder_1235678_div_127 : 1235678 % 127 = 69 := by
  sorry

end remainder_1235678_div_127_l1820_182008


namespace teddy_bears_per_shelf_l1820_182096

theorem teddy_bears_per_shelf :
  (98 / 14 = 7) := 
by
  sorry

end teddy_bears_per_shelf_l1820_182096


namespace rick_ironed_27_pieces_l1820_182034

def pieces_of_clothing_ironed (dress_shirts_per_hour : ℕ) (hours_ironing_shirts : ℕ) 
                              (dress_pants_per_hour : ℕ) (hours_ironing_pants : ℕ) : ℕ :=
  dress_shirts_per_hour * hours_ironing_shirts + dress_pants_per_hour * hours_ironing_pants

theorem rick_ironed_27_pieces :
  pieces_of_clothing_ironed 4 3 3 5 = 27 :=
by sorry

end rick_ironed_27_pieces_l1820_182034


namespace smith_boxes_l1820_182077

theorem smith_boxes (initial_markers : ℕ) (markers_per_box : ℕ) (total_markers : ℕ)
  (h1 : initial_markers = 32) (h2 : markers_per_box = 9) (h3 : total_markers = 86) :
  total_markers - initial_markers = 6 * markers_per_box :=
by
  -- We state our assumptions explicitly for better clarity
  have h_total : total_markers = 86 := h3
  have h_initial : initial_markers = 32 := h1
  have h_box : markers_per_box = 9 := h2
  sorry

end smith_boxes_l1820_182077


namespace min_shots_for_probability_at_least_075_l1820_182002

theorem min_shots_for_probability_at_least_075 (hit_rate : ℝ) (target_probability : ℝ) :
  hit_rate = 0.25 → target_probability = 0.75 → ∃ n : ℕ, n = 4 ∧ (1 - hit_rate)^n ≤ 1 - target_probability := by
  intros h_hit_rate h_target_probability
  sorry

end min_shots_for_probability_at_least_075_l1820_182002


namespace consecutive_integers_greatest_l1820_182010

theorem consecutive_integers_greatest (n : ℤ) (h : n + 2 = 8) : 
  (n + 2 = 8) → (max n (max (n + 1) (n + 2)) = 8) :=
by {
  sorry
}

end consecutive_integers_greatest_l1820_182010


namespace decimal_to_base7_l1820_182013

-- Define the decimal number
def decimal_number : ℕ := 2011

-- Define the base-7 conversion function
def to_base7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else to_base7 (n / 7) ++ [n % 7]

-- Calculate the base-7 representation of 2011
def base7_representation : List ℕ := to_base7 decimal_number

-- Prove that the base-7 representation of 2011 is [5, 6, 0, 2]
theorem decimal_to_base7 : base7_representation = [5, 6, 0, 2] :=
  by sorry

end decimal_to_base7_l1820_182013


namespace prime_pairs_square_l1820_182090

noncomputable def is_square (n : ℤ) : Prop := ∃ m : ℤ, m * m = n

def is_prime (n : ℤ) : Prop := n > 1 ∧ ∀ m : ℤ, m ∣ n → m = 1 ∨ m = n

theorem prime_pairs_square (a b : ℤ) (ha : is_prime a) (hb : is_prime b) :
  is_square (3 * a^2 * b + 16 * a * b^2) ↔ (a = 19 ∧ b = 19) ∨ (a = 2 ∧ b = 3) :=
by
  sorry

end prime_pairs_square_l1820_182090


namespace negation_of_proposition_l1820_182084

theorem negation_of_proposition (x : ℝ) :
  ¬ (x > 1 → x ^ 2 > x) ↔ (x ≤ 1 → x ^ 2 ≤ x) :=
by 
  sorry

end negation_of_proposition_l1820_182084


namespace xy_inequality_l1820_182045

theorem xy_inequality (x y : ℝ) (h: x^8 + y^8 ≤ 2) : 
  x^2 * y^2 + |x^2 - y^2| ≤ π / 2 :=
sorry

end xy_inequality_l1820_182045


namespace theo_cookie_price_l1820_182051

theorem theo_cookie_price :
  (∃ (dough_amount total_earnings per_cookie_earnings_carla per_cookie_earnings_theo : ℕ) 
     (cookies_carla cookies_theo : ℝ), 
  dough_amount = 120 ∧ 
  cookies_carla = 20 ∧ 
  per_cookie_earnings_carla = 50 ∧ 
  cookies_theo = 15 ∧ 
  total_earnings = cookies_carla * per_cookie_earnings_carla ∧ 
  per_cookie_earnings_theo = total_earnings / cookies_theo ∧ 
  per_cookie_earnings_theo = 67) :=
sorry

end theo_cookie_price_l1820_182051


namespace new_oranges_added_l1820_182030

-- Defining the initial conditions
def initial_oranges : Nat := 40
def thrown_away_oranges : Nat := 37
def total_oranges_now : Nat := 10
def remaining_oranges : Nat := initial_oranges - thrown_away_oranges
def new_oranges := total_oranges_now - remaining_oranges

-- The theorem we want to prove
theorem new_oranges_added : new_oranges = 7 := by
  sorry

end new_oranges_added_l1820_182030


namespace acute_triangle_tangent_sum_geq_3_sqrt_3_l1820_182093

theorem acute_triangle_tangent_sum_geq_3_sqrt_3 {α β γ : ℝ} (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) (h_sum : α + β + γ = π)
  (acute_α : α < π / 2) (acute_β : β < π / 2) (acute_γ : γ < π / 2) :
  Real.tan α + Real.tan β + Real.tan γ >= 3 * Real.sqrt 3 :=
sorry

end acute_triangle_tangent_sum_geq_3_sqrt_3_l1820_182093


namespace min_a2_b2_l1820_182050

noncomputable def minimum_a2_b2 (a b : ℝ) : Prop :=
  (∃ a b : ℝ, (|(-2*a - 2*b + 4)|) / (Real.sqrt (a^2 + (2*b)^2)) = 2) → (a^2 + b^2 = 2)

theorem min_a2_b2 : minimum_a2_b2 a b :=
by
  sorry

end min_a2_b2_l1820_182050


namespace train_crossing_time_l1820_182028

/-- Prove the time it takes for a train of length 50 meters running at 60 km/hr to cross a pole is 3 seconds. -/
theorem train_crossing_time
  (speed_kmh : ℝ)
  (length_m : ℝ)
  (conversion_factor : ℝ)
  (time_seconds : ℝ) :
  speed_kmh = 60 →
  length_m = 50 →
  conversion_factor = 1000 / 3600 →
  time_seconds = 3 →
  time_seconds = length_m / (speed_kmh * conversion_factor) := 
by
  intros
  sorry

end train_crossing_time_l1820_182028


namespace truncated_pyramid_volume_ratio_l1820_182069

/-
Statement: Given a truncated triangular pyramid with a plane drawn through a side of the upper base parallel to the opposite lateral edge,
and the corresponding sides of the bases in the ratio 1:2, prove that the volume of the truncated pyramid is divided in the ratio 3:4.
-/

theorem truncated_pyramid_volume_ratio (S1 S2 h : ℝ) 
  (h_ratio : S1 = 4 * S2) :
  (h * S2) / ((7 * h * S2) / 3 - h * S2) = 3 / 4 :=
by
  sorry

end truncated_pyramid_volume_ratio_l1820_182069


namespace mabel_marble_ratio_l1820_182066

variable (A K M : ℕ)

-- Conditions
def condition1 : Prop := A + 12 = 2 * K
def condition2 : Prop := M = 85
def condition3 : Prop := M = A + 63

-- The main statement to prove
theorem mabel_marble_ratio (h1 : condition1 A K) (h2 : condition2 M) (h3 : condition3 A M) : M / K = 5 :=
by
  sorry

end mabel_marble_ratio_l1820_182066


namespace partition_count_l1820_182049

theorem partition_count (A B : Finset ℕ) :
  (∀ n, n ∈ A ∨ n ∈ B) ∧ 
  (∀ n, n ∈ A → 1 ≤ n ∧ n ≤ 9) ∧ 
  (∀ n, n ∈ B → 1 ≤ n ∧ n ≤ 9) ∧ 
  (A ∩ B = ∅) ∧ 
  (A ∪ B = {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ 
  (8 * A.sum id = B.sum id) ∧ 
  (A.sum id + B.sum id = 45) → 
  ∃! (num_ways : ℕ), num_ways = 3 :=
sorry

end partition_count_l1820_182049


namespace smallest_N_sum_of_digits_eq_six_l1820_182001

def bernardo_wins (N : ℕ) : Prop :=
  let b1 := 3 * N
  let s1 := b1 - 30
  let b2 := 3 * s1
  let s2 := b2 - 30
  let b3 := 3 * s2
  let s3 := b3 - 30
  let b4 := 3 * s3
  b4 < 800

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n
  else sum_of_digits (n / 10) + (n % 10)

theorem smallest_N_sum_of_digits_eq_six :
  ∃ N : ℕ, bernardo_wins N ∧ sum_of_digits N = 6 :=
by
  sorry

end smallest_N_sum_of_digits_eq_six_l1820_182001


namespace last_digit_base5_89_l1820_182059

theorem last_digit_base5_89 (n : ℕ) (h : n = 89) : (n % 5) = 4 :=
by 
  sorry

end last_digit_base5_89_l1820_182059


namespace b_minus_a_equals_two_l1820_182015

open Set

variables {a b : ℝ}

theorem b_minus_a_equals_two (h₀ : {1, a + b, a} = ({0, b / a, b} : Finset ℝ)) (h₁ : a ≠ 0) : b - a = 2 :=
sorry

end b_minus_a_equals_two_l1820_182015


namespace min_value_expression_l1820_182056

variable {a b : ℝ}

theorem min_value_expression
  (h1 : a > 0) 
  (h2 : b > 0)
  (h3 : a + b = 4) : 
  (∃ C, (∀ a b, a > 0 → b > 0 → a + b = 4 → (b / a + 4 / b) ≥ C) ∧ 
         (∀ a b, a > 0 → b > 0 → a + b = 4 → (b / a + 4 / b) = C)) ∧ 
         C = 3 :=
  by sorry

end min_value_expression_l1820_182056


namespace min_value_16_l1820_182009

noncomputable def min_value_expr (a b : ℝ) : ℝ :=
  1 / a + 3 / b

theorem min_value_16 (a b : ℝ) (h : a > 0 ∧ b > 0) (h_constraint : a + 3 * b = 1) :
  min_value_expr a b ≥ 16 :=
sorry

end min_value_16_l1820_182009


namespace mark_donates_cans_l1820_182024

-- Definitions coming directly from the conditions
def num_shelters : ℕ := 6
def people_per_shelter : ℕ := 30
def cans_per_person : ℕ := 10

-- The final statement to be proven
theorem mark_donates_cans : (num_shelters * people_per_shelter * cans_per_person) = 1800 :=
by sorry

end mark_donates_cans_l1820_182024


namespace symmetric_axis_and_vertex_l1820_182003

theorem symmetric_axis_and_vertex (x : ℝ) : 
  (∀ x y, y = (1 / 2) * (x - 1)^2 + 6 → x = 1) 
  ∧ (1, 6) = (1, 6) :=
by 
  sorry

end symmetric_axis_and_vertex_l1820_182003


namespace prime_sum_of_primes_unique_l1820_182070

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_sum_of_primes_unique (p q : ℕ) (hp : is_prime p) (hq : is_prime q) (h_sum_prime : is_prime (p^q + q^p)) :
  p = 2 ∧ q = 3 :=
sorry

end prime_sum_of_primes_unique_l1820_182070


namespace smallest_angle_in_icosagon_l1820_182062

-- Definitions for the conditions:
def sum_of_interior_angles (n : ℕ) : ℕ := (n - 2) * 180
def average_angle (n : ℕ) (sum_of_angles : ℕ) : ℕ := sum_of_angles / n
def is_convex (angle : ℕ) : Prop := angle < 180
def arithmetic_sequence_smallest_angle (n : ℕ) (average : ℕ) (d : ℕ) : ℕ := average - 9 * d

theorem smallest_angle_in_icosagon
  (d : ℕ)
  (d_condition : d = 1)
  (convex_condition : ∀ i, is_convex (162 + (i - 1) * 2 * d))
  : arithmetic_sequence_smallest_angle 20 162 d = 153 := by
  sorry

end smallest_angle_in_icosagon_l1820_182062


namespace credibility_of_relationship_l1820_182042

theorem credibility_of_relationship
  (sample_size : ℕ)
  (chi_squared_value : ℝ)
  (table : ℕ → ℝ × ℝ)
  (h_sample : sample_size = 5000)
  (h_chi_squared : chi_squared_value = 6.109)
  (h_table : table 5 = (5.024, 0.025) ∧ table 6 = (6.635, 0.010)) :
  credible_percent = 97.5 :=
by
  sorry

end credibility_of_relationship_l1820_182042


namespace pow_mod_remainder_l1820_182004

theorem pow_mod_remainder (x : ℕ) (h : x = 3) : x^1988 % 8 = 1 := by
  sorry

end pow_mod_remainder_l1820_182004


namespace compare_polynomials_l1820_182068

noncomputable def f (x : ℝ) : ℝ := 2*x^2 + 5*x + 3
noncomputable def g (x : ℝ) : ℝ := x^2 + 4*x + 2

theorem compare_polynomials (x : ℝ) : f x > g x :=
by sorry

end compare_polynomials_l1820_182068


namespace largest_n_for_two_digit_quotient_l1820_182098

-- Lean statement for the given problem.
theorem largest_n_for_two_digit_quotient (n : ℕ) (h₀ : 0 ≤ n) (h₃ : n ≤ 9) :
  (10 ≤ (n * 100 + 5) / 5 ∧ (n * 100 + 5) / 5 < 100) ↔ n = 4 :=
by sorry

end largest_n_for_two_digit_quotient_l1820_182098


namespace remainder_when_two_pow_thirty_three_div_nine_l1820_182047

-- Define the base and the exponent
def base : ℕ := 2
def exp : ℕ := 33
def modulus : ℕ := 9

-- The main statement to prove
theorem remainder_when_two_pow_thirty_three_div_nine :
  (base ^ exp) % modulus = 8 :=
by
  sorry

end remainder_when_two_pow_thirty_three_div_nine_l1820_182047


namespace radius_increase_l1820_182053

/-- Proving that the radius increases by 7/π inches when the circumference increases from 50 inches to 64 inches -/
theorem radius_increase (C₁ C₂ : ℝ) (h₁ : C₁ = 50) (h₂ : C₂ = 64) :
  (C₂ / (2 * Real.pi) - C₁ / (2 * Real.pi)) = 7 / Real.pi :=
by
  sorry

end radius_increase_l1820_182053


namespace bus_seats_capacity_l1820_182076

-- Define the conditions
variable (x : ℕ) -- number of people each seat can hold
def left_side_seats := 15
def right_side_seats := left_side_seats - 3
def back_seat_capacity := 7
def total_capacity := left_side_seats * x + right_side_seats * x + back_seat_capacity

-- State the theorem
theorem bus_seats_capacity :
  total_capacity x = 88 → x = 3 := by
  sorry

end bus_seats_capacity_l1820_182076


namespace company_fund_initial_amount_l1820_182019

theorem company_fund_initial_amount (n : ℕ) (fund_initial : ℤ) 
  (h1 : ∃ n, fund_initial = 60 * n - 10)
  (h2 : ∃ n, 55 * n + 120 = fund_initial + 130)
  : fund_initial = 1550 := 
sorry

end company_fund_initial_amount_l1820_182019


namespace xyz_inequality_l1820_182037

theorem xyz_inequality (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  3 * (x^2 * y^2 + x^2 * z^2 + y^2 * z^2) - 2 * x * y * z * (x + y + z) ≤ 3 := by
  sorry

end xyz_inequality_l1820_182037


namespace difference_of_integers_l1820_182064

theorem difference_of_integers : ∃ (x y : ℕ), x + y = 20 ∧ x * y = 96 ∧ (x - y = 4 ∨ y - x = 4) :=
by
  sorry

end difference_of_integers_l1820_182064


namespace no_absolute_winner_l1820_182099

noncomputable def A_wins_B_probability : ℝ := 0.6
noncomputable def B_wins_V_probability : ℝ := 0.4

def no_absolute_winner_probability (A_wins_B B_wins_V : ℝ) (V_wins_A : ℝ) : ℝ :=
  let scenario1 := A_wins_B * B_wins_V * V_wins_A
  let scenario2 := A_wins_B * (1 - B_wins_V) * (1 - V_wins_A)
  scenario1 + scenario2

theorem no_absolute_winner (V_wins_A : ℝ) : no_absolute_winner_probability A_wins_B_probability B_wins_V_probability V_wins_A = 0.36 :=
  sorry

end no_absolute_winner_l1820_182099


namespace restaurant_sales_l1820_182054

theorem restaurant_sales :
  let meals_sold_8 := 10
  let price_per_meal_8 := 8
  let meals_sold_10 := 5
  let price_per_meal_10 := 10
  let meals_sold_4 := 20
  let price_per_meal_4 := 4
  let total_sales := meals_sold_8 * price_per_meal_8 + meals_sold_10 * price_per_meal_10 + meals_sold_4 * price_per_meal_4
  total_sales = 210 :=
by
  sorry

end restaurant_sales_l1820_182054


namespace toothpicks_at_20th_stage_l1820_182078

def toothpicks_in_stage (n : ℕ) : ℕ :=
  4 + 3 * (n - 1)

theorem toothpicks_at_20th_stage : toothpicks_in_stage 20 = 61 :=
by 
  sorry

end toothpicks_at_20th_stage_l1820_182078


namespace proof_problem_l1820_182022

noncomputable def a : ℝ := (11 + Real.sqrt 337) ^ (1 / 3)
noncomputable def b : ℝ := (11 - Real.sqrt 337) ^ (1 / 3)
noncomputable def x : ℝ := a + b

theorem proof_problem : x^3 + 18 * x = 22 := by
  sorry

end proof_problem_l1820_182022


namespace arithmetic_sequence_problem_l1820_182039

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ)
  (h_seq : arithmetic_sequence a)
  (h1 : a 2 + a 3 = 4)
  (h2 : a 4 + a 5 = 6) :
  a 9 + a 10 = 11 :=
sorry

end arithmetic_sequence_problem_l1820_182039


namespace even_factors_count_l1820_182092

theorem even_factors_count (n : ℕ) (h : n = 2^3 * 3 * 7^2 * 5) : 
  ∃ k, k = 36 ∧ 
       (∀ a b c d : ℕ, 1 ≤ a ∧ a ≤ 3 ∧ 0 ≤ b ∧ b ≤ 1 ∧ 0 ≤ c ∧ c ≤ 2 ∧ 0 ≤ d ∧ d ≤ 1 →
       ∃ m, m = 2^a * 3^b * 7^c * 5^d ∧ 2 ∣ m ∧ m ∣ n) := sorry

end even_factors_count_l1820_182092


namespace minimum_value_of_expression_l1820_182046

theorem minimum_value_of_expression (x : ℝ) (hx : x ≠ 0) : 
  (x^2 + 1 / x^2) ≥ 2 ∧ (x^2 + 1 / x^2 = 2 ↔ x = 1 ∨ x = -1) := 
by
  sorry

end minimum_value_of_expression_l1820_182046


namespace train_speed_correct_l1820_182083

noncomputable def train_speed : ℝ :=
  let distance := 120 -- meters
  let time := 5.999520038396929 -- seconds
  let speed_m_s := distance / time -- meters per second
  speed_m_s * 3.6 -- converting to km/hr

theorem train_speed_correct : train_speed = 72.004800384 := by
  simp [train_speed]
  sorry

end train_speed_correct_l1820_182083


namespace find_factor_l1820_182041

theorem find_factor (n f : ℤ) (h₁ : n = 124) (h₂ : n * f - 138 = 110) : f = 2 := by
  sorry

end find_factor_l1820_182041


namespace alice_meets_bob_at_25_km_l1820_182044

-- Define variables for times, speeds, and distances
variables (t : ℕ) (d : ℕ)

-- Conditions
def distance_between_homes := 41
def alice_speed := 5
def bob_speed := 4
def alice_start_time := 1

-- Relating the distances covered by Alice and Bob when they meet
def alice_walk_distance := alice_speed * (t + alice_start_time)
def bob_walk_distance := bob_speed * t
def total_walk_distance := alice_walk_distance + bob_walk_distance

-- Alexander walks 25 kilometers before meeting Bob
theorem alice_meets_bob_at_25_km :
  total_walk_distance = distance_between_homes → alice_walk_distance = 25 :=
by
  sorry

end alice_meets_bob_at_25_km_l1820_182044


namespace range_of_a_l1820_182063

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ (ax^2 - ax + 1 = 0)) ↔ (0 ≤ a ∧ a < 4) :=
sorry

end range_of_a_l1820_182063


namespace carbon_atoms_in_compound_l1820_182094

theorem carbon_atoms_in_compound 
    (molecular_weight : ℕ := 65)
    (carbon_weight : ℕ := 12)
    (hydrogen_weight : ℕ := 1)
    (oxygen_weight : ℕ := 16)
    (hydrogen_atoms : ℕ := 1)
    (oxygen_atoms : ℕ := 1) :
    ∃ (carbon_atoms : ℕ), molecular_weight = (carbon_atoms * carbon_weight) + (hydrogen_atoms * hydrogen_weight) + (oxygen_atoms * oxygen_weight) ∧ carbon_atoms = 4 :=
by
  sorry

end carbon_atoms_in_compound_l1820_182094


namespace cost_of_pencil_l1820_182026

theorem cost_of_pencil (x y : ℕ) (h1 : 4 * x + 3 * y = 224) (h2 : 2 * x + 5 * y = 154) : y = 12 := 
by
  sorry

end cost_of_pencil_l1820_182026


namespace children_absent_on_independence_day_l1820_182007

theorem children_absent_on_independence_day
  (total_children : ℕ)
  (bananas_per_child : ℕ)
  (extra_bananas : ℕ)
  (total_possible_children : total_children = 780)
  (bananas_distributed : bananas_per_child = 2)
  (additional_bananas : extra_bananas = 2) :
  ∃ (A : ℕ), A = 390 := 
sorry

end children_absent_on_independence_day_l1820_182007


namespace n_mult_n_plus_1_eq_square_l1820_182043

theorem n_mult_n_plus_1_eq_square (n : ℤ) : (∃ k : ℤ, n * (n + 1) = k^2) ↔ (n = 0 ∨ n = -1) := 
by sorry

end n_mult_n_plus_1_eq_square_l1820_182043


namespace find_f_2006_l1820_182029

-- Assuming an odd periodic function f with period 3(3x+1), defining the conditions.
def f : ℤ → ℤ := sorry -- Definition of f is not provided.

-- Conditions
axiom odd_function : ∀ x : ℤ, f (-x) = -f x
axiom period_3_function : ∀ x : ℤ, f (3 * x + 1) = f (3 * (x + 1) + 1)
axiom value_at_1 : f 1 = -1

-- Question: What is f(2006)?
theorem find_f_2006 : f 2006 = 1 := sorry

end find_f_2006_l1820_182029


namespace sarah_age_is_26_l1820_182027

theorem sarah_age_is_26 (mark_age billy_age ana_age : ℕ) (sarah_age : ℕ) 
  (h1 : sarah_age = 3 * mark_age - 4)
  (h2 : mark_age = billy_age + 4)
  (h3 : billy_age = ana_age / 2)
  (h4 : ana_age = 15 - 3) :
  sarah_age = 26 := 
sorry

end sarah_age_is_26_l1820_182027


namespace part1_part2_l1820_182032

def A (t : ℝ) : Prop :=
  ∀ x : ℝ, (t+2)*x^2 + 2*x + 1 > 0

def B (a x : ℝ) : Prop :=
  (a*x - 1)*(x + a) > 0

theorem part1 (t : ℝ) : A t ↔ t < -1 :=
sorry

theorem part2 (a : ℝ) : (∀ t : ℝ, t < -1 → ∀ x : ℝ, B a x) → (0 ≤ a ∧ a ≤ 1) :=
sorry

end part1_part2_l1820_182032


namespace percentage_paid_to_x_l1820_182061

theorem percentage_paid_to_x (X Y : ℕ) (h₁ : Y = 350) (h₂ : X + Y = 770) :
  (X / Y) * 100 = 120 :=
by
  sorry

end percentage_paid_to_x_l1820_182061


namespace true_statement_l1820_182025

theorem true_statement :
  -8 < -2 := 
sorry

end true_statement_l1820_182025


namespace solve_inequality_system_l1820_182091

theorem solve_inequality_system (x : ℝ) :
  (x / 3 + 2 > 0) ∧ (2 * x + 5 ≥ 3) ↔ (x ≥ -1) :=
by
  sorry

end solve_inequality_system_l1820_182091


namespace jaguars_total_games_l1820_182052

-- Defining constants for initial conditions
def initial_win_rate : ℚ := 0.55
def additional_wins : ℕ := 8
def additional_losses : ℕ := 2
def final_win_rate : ℚ := 0.6

-- Defining the main problem statement
theorem jaguars_total_games : 
  ∃ y x : ℕ, (x = initial_win_rate * y) ∧ (x + additional_wins = final_win_rate * (y + (additional_wins + additional_losses))) ∧ (y + (additional_wins + additional_losses) = 50) :=
sorry

end jaguars_total_games_l1820_182052


namespace solve_for_k_l1820_182011

theorem solve_for_k (k : ℕ) (h : 16 / k = 4) : k = 4 :=
sorry

end solve_for_k_l1820_182011


namespace symmetrical_line_range_l1820_182023

theorem symmetrical_line_range {k : ℝ} :
  (∀ x y : ℝ, (y = k * x - 1) ∧ (x + y - 1 = 0) → y ≠ -x + 1) → k > 1 ↔ k > 1 :=
by
  sorry

end symmetrical_line_range_l1820_182023


namespace cafeteria_orders_green_apples_l1820_182020

theorem cafeteria_orders_green_apples (G : ℕ) (h1 : 6 + G = 5 + 16) : G = 15 :=
by
  sorry

end cafeteria_orders_green_apples_l1820_182020


namespace attendance_proof_l1820_182060

noncomputable def next_year_attendance (this_year: ℕ) := 2 * this_year
noncomputable def last_year_attendance (next_year: ℕ) := next_year - 200
noncomputable def total_attendance (last_year this_year next_year: ℕ) := last_year + this_year + next_year

theorem attendance_proof (this_year: ℕ) (h1: this_year = 600):
    total_attendance (last_year_attendance (next_year_attendance this_year)) this_year (next_year_attendance this_year) = 2800 :=
by
  sorry

end attendance_proof_l1820_182060


namespace parrot_seeds_consumed_l1820_182079

theorem parrot_seeds_consumed (H1 : ∃ T : ℝ, 0.40 * T = 8) : 
  (∃ T : ℝ, 0.40 * T = 8 ∧ 2 * T = 40) :=
sorry

end parrot_seeds_consumed_l1820_182079


namespace travel_speed_is_four_l1820_182040
-- Import the required library

-- Define the conditions
def jacksSpeed (x : ℝ) : ℝ := x^2 - 13 * x - 26
def jillsDistance (x : ℝ) : ℝ := x^2 - 5 * x - 66
def jillsTime (x : ℝ) : ℝ := x + 8

-- Prove the equivalent statement
theorem travel_speed_is_four (x : ℝ) (h : x = 15) :
  jillsDistance x / jillsTime x = 4 ∧ jacksSpeed x = 4 := 
by sorry

end travel_speed_is_four_l1820_182040
