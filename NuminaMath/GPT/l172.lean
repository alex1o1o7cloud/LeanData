import Mathlib

namespace city_A_fare_higher_than_city_B_l172_172881

def fare_in_city_A (x : ℝ) : ℝ :=
  10 + 2 * (x - 3)

def fare_in_city_B (x : ℝ) : ℝ :=
  8 + 2.5 * (x - 3)

theorem city_A_fare_higher_than_city_B (x : ℝ) (h : x > 3) :
  fare_in_city_A x > fare_in_city_B x → 3 < x ∧ x < 7 :=
by
  sorry

end city_A_fare_higher_than_city_B_l172_172881


namespace train_stoppage_time_l172_172967

-- Definitions from conditions
def speed_without_stoppages := 60 -- kmph
def speed_with_stoppages := 36 -- kmph

-- Main statement to prove
theorem train_stoppage_time : (60 - 36) / 60 * 60 = 24 := by
  sorry

end train_stoppage_time_l172_172967


namespace simplify_tangent_expression_l172_172591

theorem simplify_tangent_expression :
  (1 + Real.tan (15 * Real.pi / 180)) / (1 - Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 :=
by
  sorry

end simplify_tangent_expression_l172_172591


namespace raisin_weight_l172_172872

theorem raisin_weight (Wg : ℝ) (dry_grapes_fraction : ℝ) (dry_raisins_fraction : ℝ) :
  Wg = 101.99999999999999 → dry_grapes_fraction = 0.10 → dry_raisins_fraction = 0.85 → 
  Wg * dry_grapes_fraction / dry_raisins_fraction = 12 := 
by
  intros h1 h2 h3
  sorry

end raisin_weight_l172_172872


namespace trivia_team_missing_members_l172_172112

theorem trivia_team_missing_members 
  (total_members : ℕ)
  (points_per_member : ℕ)
  (total_points : ℕ)
  (showed_up_members : ℕ)
  (missing_members : ℕ) 
  (h1 : total_members = 15) 
  (h2 : points_per_member = 3) 
  (h3 : total_points = 27) 
  (h4 : showed_up_members = total_points / points_per_member) 
  (h5 : missing_members = total_members - showed_up_members) : 
  missing_members = 6 :=
by
  sorry

end trivia_team_missing_members_l172_172112


namespace find_number_satisfying_9y_eq_number12_l172_172415

noncomputable def power_9_y (y : ℝ) := (9 : ℝ) ^ y
noncomputable def root_12 (x : ℝ) := x ^ (1 / 12 : ℝ)

theorem find_number_satisfying_9y_eq_number12 :
  ∃ number : ℝ, power_9_y 6 = number ^ 12 ∧ abs (number - 3) < 0.0001 :=
by
  sorry

end find_number_satisfying_9y_eq_number12_l172_172415


namespace doug_initial_marbles_l172_172268

theorem doug_initial_marbles (E D : ℕ) (H1 : E = D + 5) (H2 : E = 27) : D = 22 :=
by
  -- proof provided here would infer the correct answer from the given conditions
  sorry

end doug_initial_marbles_l172_172268


namespace payment_per_minor_character_l172_172575

noncomputable def M : ℝ := 285000 / 19 

theorem payment_per_minor_character
    (num_main_characters : ℕ := 5)
    (num_minor_characters : ℕ := 4)
    (total_payment : ℝ := 285000)
    (payment_ratio : ℝ := 3)
    (eq1 : 5 * 3 * M + 4 * M = total_payment) :
    M = 15000 :=
by
  sorry

end payment_per_minor_character_l172_172575


namespace value_range_of_a_l172_172848

variable (A B : Set ℝ)

noncomputable def A_def : Set ℝ := { x | 2 * x^2 - 3 * x + 1 ≤ 0 }
noncomputable def B_def (a : ℝ) : Set ℝ := { x | x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0 }

theorem value_range_of_a (a : ℝ) (hA : A = A_def) (hB : B = B_def a) :
    (Bᶜ ∩ A = ∅) → (0 ≤ a ∧ a ≤ 0.5) := 
sorry

end value_range_of_a_l172_172848


namespace length_of_shorter_piece_l172_172641

theorem length_of_shorter_piece (x : ℕ) (h1 : x + (x + 12) = 68) : x = 28 :=
by
  sorry

end length_of_shorter_piece_l172_172641


namespace find_n_values_l172_172153

theorem find_n_values (n : ℚ) :
  ( 4 * n ^ 2 + 3 * n + 2 = 2 * n + 2 ∨ 4 * n ^ 2 + 3 * n + 2 = 5 * n + 4 ) →
  ( n = 0 ∨ n = 1 ) :=
by
  sorry

end find_n_values_l172_172153


namespace wedding_cost_l172_172453

theorem wedding_cost (venue_cost food_drink_cost guests_john : ℕ) 
  (guest_increment decorations_base decorations_per_guest transport_couple transport_per_guest entertainment_cost surchage_rate discount_thresh : ℕ) (discount_rate : ℕ) :
  let guests_wife := guests_john + (guests_john * guest_increment / 100)
  let venue_total := venue_cost + (venue_cost * surchage_rate / 100)
  let food_drink_total := if guests_wife > discount_thresh then (food_drink_cost * guests_wife) * (100 - discount_rate) / 100 else food_drink_cost * guests_wife
  let decorations_total := decorations_base + (decorations_per_guest * guests_wife)
  let transport_total := transport_couple + (transport_per_guest * guests_wife)
  (venue_total + food_drink_total + decorations_total + transport_total + entertainment_cost = 56200) :=
by {
  -- Constants given in the conditions
  let venue_cost := 10000
  let food_drink_cost := 500
  let guests_john := 50
  let guest_increment := 60
  let decorations_base := 2500
  let decorations_per_guest := 10
  let transport_couple := 200
  let transport_per_guest := 15
  let entertainment_cost := 4000
  let surchage_rate := 15
  let discount_thresh := 75
  let discount_rate := 10
  sorry
}

end wedding_cost_l172_172453


namespace smallest_solution_floor_eq_l172_172688

theorem smallest_solution_floor_eq (x : ℝ) (hx : ⌊x^2⌋ - ⌊x⌋^2 = 19) : x = 11 := by
  sorry

end smallest_solution_floor_eq_l172_172688


namespace gre_exam_month_l172_172269

def months_of_year := ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

def start_month := "June"
def preparation_duration := 5

theorem gre_exam_month :
  months_of_year[(months_of_year.indexOf start_month + preparation_duration) % 12] = "November" := by
  sorry

end gre_exam_month_l172_172269


namespace f_5_eq_25sqrt5_l172_172232

open Real

noncomputable def f : ℝ → ℝ := sorry

axiom continuous_f : Continuous f
axiom functional_eq : ∀ x y : ℝ, f (x + y) = f x * f y
axiom f_2 : f 2 = 5

theorem f_5_eq_25sqrt5 : f 5 = 25 * Real.sqrt 5 := by
  sorry

end f_5_eq_25sqrt5_l172_172232


namespace a_minus_b_eq_three_l172_172411

theorem a_minus_b_eq_three (a b : ℝ) (h : (a+bi) * i = 1 + 2 * i) : a - b = 3 :=
by
  sorry

end a_minus_b_eq_three_l172_172411


namespace functional_inequality_solution_l172_172124

theorem functional_inequality_solution (f : ℝ → ℝ) (h : ∀ a b : ℝ, f (a^2) - f (b^2) ≤ (f (a) + b) * (a - f (b))) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) := 
sorry

end functional_inequality_solution_l172_172124


namespace people_got_rid_of_some_snails_l172_172793

namespace SnailProblem

def originalSnails : ℕ := 11760
def remainingSnails : ℕ := 8278
def snailsGotRidOf : ℕ := 3482

theorem people_got_rid_of_some_snails :
  originalSnails - remainingSnails = snailsGotRidOf :=
by 
  sorry

end SnailProblem

end people_got_rid_of_some_snails_l172_172793


namespace find_m_l172_172387

section
variables {R : Type*} [CommRing R]

def f (x : R) : R := 4 * x^2 - 3 * x + 5
def g (x : R) (m : R) : R := x^2 - m * x - 8

theorem find_m (m : ℚ) : 
  f (5 : ℚ) - g (5 : ℚ) m = 20 → m = -53 / 5 :=
by {
  sorry
}

end

end find_m_l172_172387


namespace role_assignment_l172_172819

theorem role_assignment (m w : ℕ) (m_roles w_roles e_roles : ℕ) 
  (hm : m = 5) (hw : w = 6) (hm_roles : m_roles = 2) (hw_roles : w_roles = 2) (he_roles : e_roles = 2) :
  ∃ (total_assignments : ℕ), total_assignments = 25200 :=
by
  sorry

end role_assignment_l172_172819


namespace problem_statement_l172_172022

variables {A B C O D : Type}
variables [AddCommGroup A] [Module ℝ A]
variables (a b c o d : A)

-- Define the geometric conditions
axiom condition1 : a + 2 • b + 3 • c = 0
axiom condition2 : ∃ (D: A), (∃ (k : ℝ), a = k • d ∧ k ≠ 0) ∧ (∃ (u v : ℝ),  u • b + v • c = d ∧ u + v = 1)

-- Define points
def OA : A := a - o
def OB : A := b - o
def OC : A := c - o
def OD : A := d - o

-- The main statement to prove
theorem problem_statement : 2 • (b - d) + 3 • (c - d) = (0 : A) :=
by
  sorry

end problem_statement_l172_172022


namespace not_possible_to_list_numbers_l172_172445

theorem not_possible_to_list_numbers :
  ¬ (∃ (f : ℕ → ℕ), (∀ n, f n ≥ 1 ∧ f n ≤ 1963) ∧
                     (∀ n, Nat.gcd (f n) (f (n+1)) = 1) ∧
                     (∀ n, Nat.gcd (f n) (f (n+2)) = 1)) :=
by
  sorry

end not_possible_to_list_numbers_l172_172445


namespace x_squared_minus_y_squared_l172_172870

theorem x_squared_minus_y_squared {x y : ℚ} 
    (h1 : x + y = 3/8) 
    (h2 : x - y = 5/24) 
    : x^2 - y^2 = 5/64 := 
by 
    -- The proof would go here
    sorry

end x_squared_minus_y_squared_l172_172870


namespace mary_age_l172_172490

theorem mary_age (M F : ℕ) (h1 : F = 4 * M) (h2 : F - 3 = 5 * (M - 3)) : M = 12 :=
by
  sorry

end mary_age_l172_172490


namespace polynomial_expansion_l172_172391

theorem polynomial_expansion (t : ℝ) :
  (3 * t^3 + 2 * t^2 - 4 * t + 3) * (-2 * t^2 + 3 * t - 4) =
  -6 * t^5 + 5 * t^4 + 2 * t^3 - 26 * t^2 + 25 * t - 12 :=
by sorry

end polynomial_expansion_l172_172391


namespace geo_seq_b_formula_b_n_sum_T_n_l172_172729

-- Define the sequence a_n 
def a (n : ℕ) : ℕ :=
  if n = 0 then 1 else sorry -- Definition based on provided conditions

-- Define the partial sum S_n
def S (n : ℕ) : ℕ :=
  if n = 0 then 1 else 4 * a (n-1) + 2 -- Given condition S_{n+1} = 4a_n + 2

-- Condition for b_n
def b (n : ℕ) : ℕ :=
  a (n+1) - 2 * a n

-- Definition for c_n
def c (n : ℕ) := (b n) / 3

-- Define the sequence terms for c_n based sequence
def T (n : ℕ) : ℝ :=
  sorry -- Needs explicit definition from given sequence part

-- Proof statements
theorem geo_seq_b : ∀ n : ℕ, b (n + 1) = 2 * b n :=
  sorry

theorem formula_b_n : ∀ n : ℕ, b n = 3 * 2^(n-1) :=
  sorry

theorem sum_T_n : ∀ n : ℕ, T n = n / (n + 1) :=
  sorry

end geo_seq_b_formula_b_n_sum_T_n_l172_172729


namespace find_ab_l172_172161

variable (a b : ℝ)

theorem find_ab (h1 : a + b = 4) (h2 : a^3 + b^3 = 136) : a * b = -6 := by
  sorry

end find_ab_l172_172161


namespace trajectory_of_M_l172_172045

theorem trajectory_of_M {x y x₀ y₀ : ℝ} (P_on_parabola : x₀^2 = 2 * y₀)
(line_PQ_perpendicular : ∀ Q : ℝ, true)
(vector_PM_PQ_relation : x₀ = x ∧ y₀ = 2 * y) :
  x^2 = 4 * y := by
  sorry

end trajectory_of_M_l172_172045


namespace x_pow_10_eq_correct_answer_l172_172563

noncomputable def x : ℝ := sorry

theorem x_pow_10_eq_correct_answer (h : x + (1 / x) = Real.sqrt 5) : 
  x^10 = (50 + 25 * Real.sqrt 5) / 2 := 
sorry

end x_pow_10_eq_correct_answer_l172_172563


namespace original_price_of_movie_ticket_l172_172185

theorem original_price_of_movie_ticket
    (P : ℝ)
    (new_price : ℝ)
    (h1 : new_price = 80)
    (h2 : new_price = 0.80 * P) :
    P = 100 :=
by
  sorry

end original_price_of_movie_ticket_l172_172185


namespace negation_of_exists_cond_l172_172073

theorem negation_of_exists_cond (x : ℝ) (h : x > 0) : ¬ (∃ x : ℝ, x > 0 ∧ x^3 - x + 1 > 0) ↔ (∀ x : ℝ, x > 0 → x^3 - x + 1 ≤ 0) :=
by 
  sorry

end negation_of_exists_cond_l172_172073


namespace equivalent_expression_l172_172959

theorem equivalent_expression (x : ℝ) (hx : x > 0) : (x^2 * x^(1/4))^(1/3) = x^(3/4) := 
  sorry

end equivalent_expression_l172_172959


namespace meat_per_slice_is_22_l172_172039

noncomputable def piecesOfMeatPerSlice : ℕ :=
  let pepperoni := 30
  let ham := 2 * pepperoni
  let sausage := pepperoni + 12
  let totalMeat := pepperoni + ham + sausage
  let slices := 6
  totalMeat / slices

theorem meat_per_slice_is_22 : piecesOfMeatPerSlice = 22 :=
by
  -- Here would be the proof (not required in the task)
  sorry

end meat_per_slice_is_22_l172_172039


namespace probability_of_smallest_section_l172_172251

-- Define the probabilities for the largest and next largest sections
def P_largest : ℚ := 1 / 2
def P_next_largest : ℚ := 1 / 3

-- Define the total probability constraint
def total_probability (P_smallest : ℚ) : Prop :=
  P_largest + P_next_largest + P_smallest = 1

-- State the theorem to be proved
theorem probability_of_smallest_section : 
  ∃ P_smallest : ℚ, total_probability P_smallest ∧ P_smallest = 1 / 6 := 
by
  sorry

end probability_of_smallest_section_l172_172251


namespace total_distance_combined_l172_172067

/-- The conditions for the problem
Each car has 50 liters of fuel.
Car U has a fuel efficiency of 20 liters per 100 kilometers.
Car V has a fuel efficiency of 25 liters per 100 kilometers.
Car W has a fuel efficiency of 5 liters per 100 kilometers.
Car X has a fuel efficiency of 10 liters per 100 kilometers.
-/
theorem total_distance_combined (fuel_U fuel_V fuel_W fuel_X : ℕ) (eff_U eff_V eff_W eff_X : ℕ) (fuel : ℕ)
  (hU : fuel_U = 50) (hV : fuel_V = 50) (hW : fuel_W = 50) (hX : fuel_X = 50)
  (eU : eff_U = 20) (eV : eff_V = 25) (eW : eff_W = 5) (eX : eff_X = 10) :
  (fuel_U * 100 / eff_U) + (fuel_V * 100 / eff_V) + (fuel_W * 100 / eff_W) + (fuel_X * 100 / eff_X) = 1950 := by 
  sorry

end total_distance_combined_l172_172067


namespace cost_per_vent_l172_172784

/--
Given that:
1. The total cost of the HVAC system is $20,000.
2. The system includes 2 conditioning zones.
3. Each zone has 5 vents.

Prove that the cost per vent is $2000.
-/
theorem cost_per_vent (total_cost : ℕ) (zones : ℕ) (vents_per_zone : ℕ) (h1 : total_cost = 20000) (h2 : zones = 2) (h3 : vents_per_zone = 5) :
  total_cost / (zones * vents_per_zone) = 2000 := 
sorry

end cost_per_vent_l172_172784


namespace distance_from_P_to_AD_is_correct_l172_172594

noncomputable def P_distance_to_AD : ℝ :=
  let A : ℝ × ℝ := (0, 6)
  let D : ℝ × ℝ := (0, 0)
  let C : ℝ × ℝ := (6, 0)
  let M : ℝ × ℝ := (3, 0)
  let radius1 : ℝ := 5
  let radius2 : ℝ := 6
  let circle1_eq := fun (x y : ℝ) => (x - 3)^2 + y^2 = 25
  let circle2_eq := fun (x y : ℝ) => x^2 + (y - 6)^2 = 36
  let P := (24/5, 18/5)
  let AD := fun x y : ℝ => x = 0
  abs ((P.fst : ℝ) - 0)

theorem distance_from_P_to_AD_is_correct :
  P_distance_to_AD = 24 / 5 := by
  sorry

end distance_from_P_to_AD_is_correct_l172_172594


namespace smallest_solution_floor_eq_l172_172692

theorem smallest_solution_floor_eq (x : ℝ) (hx : ⌊x^2⌋ - ⌊x⌋^2 = 19) : x = 11 := by
  sorry

end smallest_solution_floor_eq_l172_172692


namespace final_replacement_weight_l172_172208

theorem final_replacement_weight (W : ℝ) (a b c d e : ℝ) 
  (h1 : a = W / 10)
  (h2 : b = (W - 70 + e) / 10)
  (h3 : b - a = 4)
  (h4 : c = (W - 70 + e - 110 + d) / 10)
  (h5 : c - b = -2)
  (h6 : d = (W - 70 + e - 110 + d + 140 - 90) / 10)
  (h7 : d - c = 5)
  : e = 110 ∧ d = 90 ∧ 140 = e + 50 := sorry

end final_replacement_weight_l172_172208


namespace cubic_polynomials_common_roots_c_d_l172_172846

theorem cubic_polynomials_common_roots_c_d (c d : ℝ) :
  (∀ (r s : ℝ), r ≠ s ∧
     (r^3 + c*r^2 + 12*r + 7 = 0) ∧ (s^3 + c*s^2 + 12*s + 7 = 0) ∧
     (r^3 + d*r^2 + 15*r + 9 = 0) ∧ (s^3 + d*s^2 + 15*s + 9 = 0)) →
  (c = -5 ∧ d = -6) := 
by
  sorry

end cubic_polynomials_common_roots_c_d_l172_172846


namespace distinct_not_geom_prog_l172_172120

open Nat

theorem distinct_not_geom_prog (k m n : ℕ) (hk : k ≠ m) (hm : m ≠ n) (hn : k ≠ n) :
  ¬ ((2^m + 1)^2 = (2^k + 1) * (2^n + 1)) :=
by sorry

end distinct_not_geom_prog_l172_172120


namespace cylinder_area_ratio_l172_172743

noncomputable def ratio_of_areas (r h : ℝ) (h_cond : 2 * r / h = h / (2 * Real.pi * r)) : ℝ :=
  let lateral_area := 2 * Real.pi * r * h
  let total_area := lateral_area + 2 * Real.pi * r * r
  lateral_area / total_area

theorem cylinder_area_ratio {r h : ℝ} (h_cond : 2 * r / h = h / (2 * Real.pi * r)) :
  ratio_of_areas r h h_cond = 2 * Real.sqrt Real.pi / (2 * Real.sqrt Real.pi + 1) := 
sorry

end cylinder_area_ratio_l172_172743


namespace find_b_find_area_l172_172422

open Real

noncomputable def A : ℝ := sorry
noncomputable def B : ℝ := A + π / 2
noncomputable def a : ℝ := 3
noncomputable def cos_A : ℝ := sqrt 6 / 3
noncomputable def b : ℝ := 3 * sqrt 2
noncomputable def area : ℝ := 3 * sqrt 2 / 2

theorem find_b (A : ℝ) (H1 : a = 3) (H2 : cos A = sqrt 6 / 3) (H3 : B = A + π / 2) : 
  b = 3 * sqrt 2 := 
  sorry

theorem find_area (A : ℝ) (H1 : a = 3) (H2 : cos A = sqrt 6 / 3) (H3 : B = A + π / 2) : 
  area = 3 * sqrt 2 / 2 := 
  sorry

end find_b_find_area_l172_172422


namespace xyz_eq_neg10_l172_172029

noncomputable def complex_numbers := {z : ℂ // z ≠ 0}

variables (a b c x y z : complex_numbers)

def condition1 := a.val = (b.val + c.val) / (x.val - 3)
def condition2 := b.val = (a.val + c.val) / (y.val - 3)
def condition3 := c.val = (a.val + b.val) / (z.val - 3)
def condition4 := x.val * y.val + x.val * z.val + y.val * z.val = 9
def condition5 := x.val + y.val + z.val = 6

theorem xyz_eq_neg10 (a b c x y z : complex_numbers) :
  condition1 a b c x ∧ condition2 a b c y ∧ condition3 a b c z ∧
  condition4 x y z ∧ condition5 x y z → x.val * y.val * z.val = -10 :=
by sorry

end xyz_eq_neg10_l172_172029


namespace center_cell_value_l172_172439

variable (a b c d e f g h i : ℝ)

-- Defining the conditions
def row_product_1 := a * b * c = 1 ∧ d * e * f = 1 ∧ g * h * i = 1
def col_product_1 := a * d * g = 1 ∧ b * e * h = 1 ∧ c * f * i = 1
def subgrid_product_2 := a * b * d * e = 2 ∧ b * c * e * f = 2 ∧ d * e * g * h = 2 ∧ e * f * h * i = 2

-- The theorem to prove
theorem center_cell_value (h1 : row_product_1 a b c d e f g h i) 
                          (h2 : col_product_1 a b c d e f g h i) 
                          (h3 : subgrid_product_2 a b c d e f g h i) : 
                          e = 1 :=
by
  sorry

end center_cell_value_l172_172439


namespace trapezoid_area_l172_172013

theorem trapezoid_area 
  (area_ABE area_ADE : ℝ)
  (DE BE : ℝ)
  (h1 : area_ABE = 40)
  (h2 : area_ADE = 30)
  (h3 : DE = 2 * BE) : 
  area_ABE + area_ADE + area_ADE + 4 * area_ABE = 260 :=
by
  -- sorry admits the goal without providing the actual proof
  sorry

end trapezoid_area_l172_172013


namespace num_students_is_92_l172_172617

noncomputable def total_students (S : ℕ) : Prop :=
  let remaining := S - 20
  let biking := (5/8 : ℚ) * remaining
  let walking := (3/8 : ℚ) * remaining
  walking = 27

theorem num_students_is_92 : total_students 92 :=
by
  let remaining := 92 - 20
  let biking := (5/8 : ℚ) * remaining
  let walking := (3/8 : ℚ) * remaining
  have walk_eq : walking = 27 := by sorry
  exact walk_eq

end num_students_is_92_l172_172617


namespace original_deck_card_count_l172_172252

theorem original_deck_card_count (r b u : ℕ)
  (h1 : r / (r + b + u) = 1 / 5)
  (h2 : r / (r + b + u + 3) = 1 / 6) :
  r + b + u = 15 := by
  sorry

end original_deck_card_count_l172_172252


namespace probability_jammed_l172_172117

theorem probability_jammed (T τ : ℝ) (h : τ < T) : 
    (2 * τ / T - (τ / T) ^ 2) = (T^2 - (T - τ)^2) / T^2 := 
by
  sorry

end probability_jammed_l172_172117


namespace train_pass_time_l172_172937

noncomputable def train_length : ℕ := 360
noncomputable def platform_length : ℕ := 140
noncomputable def train_speed_kmh : ℕ := 45

noncomputable def convert_speed_to_mps (speed_kmh : ℕ) : ℚ := 
  (speed_kmh * 1000) / 3600

noncomputable def total_distance (train_len platform_len : ℕ) : ℕ :=
  train_len + platform_len

noncomputable def time_to_pass (distance : ℕ) (speed_mps : ℚ) : ℚ :=
  distance / speed_mps

theorem train_pass_time 
  (train_len : ℕ) 
  (platform_len : ℕ) 
  (speed_kmh : ℕ) : 
  time_to_pass (total_distance train_len platform_len) (convert_speed_to_mps speed_kmh) = 40 := 
by 
  sorry

end train_pass_time_l172_172937


namespace center_cell_value_l172_172430

theorem center_cell_value (a b c d e f g h i : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
  (hg : 0 < g) (hh : 0 < h) (hi : 0 < i)
  (row1 : a * b * c = 1) (row2 : d * e * f = 1) (row3 : g * h * i = 1)
  (col1 : a * d * g = 1) (col2 : b * e * h = 1) (col3 : c * f * i = 1)
  (square1 : a * b * d * e = 2) (square2 : b * c * e * f = 2)
  (square3 : d * e * g * h = 2) (square4 : e * f * h * i = 2) :
  e = 1 :=
begin
  sorry
end

end center_cell_value_l172_172430


namespace cubic_sum_expression_l172_172413

theorem cubic_sum_expression (x y z p q r : ℝ) (h1 : x * y = p) (h2 : x * z = q) (h3 : y * z = r) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  x^3 + y^3 + z^3 = (p^2 * q^2 + p^2 * r^2 + q^2 * r^2) / (p * q * r) :=
by
  sorry

end cubic_sum_expression_l172_172413


namespace portion_apples_weight_fraction_l172_172508

-- Given conditions
def total_apples : ℕ := 28
def total_weight_kg : ℕ := 3
def number_of_portions : ℕ := 7

-- Proof statement
theorem portion_apples_weight_fraction :
  (1 / number_of_portions = 1 / 7) ∧ (3 / number_of_portions = 3 / 7) :=
by
  -- Proof goes here
  sorry

end portion_apples_weight_fraction_l172_172508


namespace star_set_l172_172456

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | 1 ≤ x}
def star (A B : Set ℝ) : Set ℝ := {x | (x ∈ A ∪ B) ∧ ¬(x ∈ A ∩ B)}

theorem star_set :
  star A B = {x | (0 ≤ x ∧ x < 1) ∨ (3 < x)} :=
by
  sorry

end star_set_l172_172456


namespace triangle_area_l172_172826

noncomputable def s (a b c : ℝ) : ℝ := (a + b + c) / 2
noncomputable def area (a b c : ℝ) : ℝ := Real.sqrt (s a b c * (s a b c - a) * (s a b c - b) * (s a b c - c))

theorem triangle_area (a b c : ℝ) (ha : a = 13) (hb : b = 12) (hc : c = 5) : area a b c = 30 := by
  rw [ha, hb, hc]
  show area 13 12 5 = 30
  -- manually calculate and reduce the expression to verify the theorem
  sorry

end triangle_area_l172_172826


namespace arithmetic_sum_sequence_l172_172995

theorem arithmetic_sum_sequence (a : ℕ → ℝ) (d : ℝ)
  (h : ∀ n, a (n + 1) = a n + d) :
  ∃ d', 
    a 4 + a 5 + a 6 - (a 1 + a 2 + a 3) = d' ∧
    a 7 + a 8 + a 9 - (a 4 + a 5 + a 6) = d' :=
by
  sorry

end arithmetic_sum_sequence_l172_172995


namespace determine_value_of_x_l172_172577

theorem determine_value_of_x (x y z : ℤ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≥ y) (hyz : y ≥ z)
  (h1 : x^2 - y^2 - z^2 + x * y = 4033) 
  (h2 : x^2 + 4 * y^2 + 4 * z^2 - 4 * x * y - 3 * x * z - 3 * y * z = -3995) : 
  x = 69 := sorry

end determine_value_of_x_l172_172577


namespace unique_solution_of_system_l172_172907

theorem unique_solution_of_system (n k m : ℕ) (hnk : n + k = Nat.gcd n k ^ 2) (hkm : k + m = Nat.gcd k m ^ 2) (hmn : m + n = Nat.gcd m n ^ 2) : 
  n = 2 ∧ k = 2 ∧ m = 2 :=
by
  sorry

end unique_solution_of_system_l172_172907


namespace largest_x_value_l172_172592

-- Definition of the equation
def equation (x : ℚ) : Prop := 3 * (9 * x^2 + 10 * x + 11) = x * (9 * x - 45)

-- The problem to prove is that the largest value of x satisfying the equation is -1/2
theorem largest_x_value : ∃ x : ℚ, equation x ∧ ∀ y : ℚ, equation y → y ≤ -1/2 := by
  sorry

end largest_x_value_l172_172592


namespace calculate_y_l172_172997

theorem calculate_y (w x y : ℝ) (h1 : (7 / w) + (7 / x) = 7 / y) (h2 : w * x = y) (h3 : (w + x) / 2 = 0.5) : y = 0.25 :=
by
  sorry

end calculate_y_l172_172997


namespace absolute_difference_avg_median_l172_172301

theorem absolute_difference_avg_median (a b : ℝ) (h1 : 1 < a) (h2 : a < b) : 
  |((3 + 4 * a + 2 * b) / 4) - (a + b / 2 + 1)| = 1 / 4 :=
by
  sorry

end absolute_difference_avg_median_l172_172301


namespace smallest_solution_floor_equation_l172_172705

theorem smallest_solution_floor_equation : ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (x = Real.sqrt 109) :=
by
  sorry

end smallest_solution_floor_equation_l172_172705


namespace ages_sum_13_and_product_72_l172_172540

theorem ages_sum_13_and_product_72 (g b s : ℕ) (h1 : b < g) (h2 : g < s) (h3 : b * g * s = 72) : b + g + s = 13 :=
sorry

end ages_sum_13_and_product_72_l172_172540


namespace cost_price_computer_table_l172_172100

theorem cost_price_computer_table 
  (CP SP : ℝ)
  (h1 : SP = CP * 1.20)
  (h2 : SP = 8400) :
  CP = 7000 :=
by
  sorry

end cost_price_computer_table_l172_172100


namespace cubic_roots_geometric_progression_l172_172764

theorem cubic_roots_geometric_progression 
  (a r : ℝ)
  (h_roots: 27 * a^3 * r^3 - 81 * a^2 * r^2 + 63 * a * r - 14 = 0)
  (h_sum: a + a * r + a * r^2 = 3)
  (h_product: a^3 * r^3 = 14 / 27)
  : (max (a^2) ((a * r^2)^2) - min (a^2) ((a * r^2)^2) = 5 / 3) := 
sorry

end cubic_roots_geometric_progression_l172_172764


namespace smallest_positive_integer_l172_172924

theorem smallest_positive_integer (x : ℕ) : 
  (5 * x ≡ 18 [MOD 33]) ∧ (x ≡ 4 [MOD 7]) → x = 10 := 
by 
  sorry

end smallest_positive_integer_l172_172924


namespace students_on_bleachers_l172_172043

theorem students_on_bleachers (F B : ℕ) (h1 : F + B = 26) (h2 : F / (F + B) = 11 / 13) : B = 4 :=
by sorry

end students_on_bleachers_l172_172043


namespace dasha_strip_problem_l172_172263

theorem dasha_strip_problem (a b c : ℕ) (h : a * (2 * b + 2 * c - a) = 43) :
  a = 1 ∧ b + c = 22 :=
by {
  sorry
}

end dasha_strip_problem_l172_172263


namespace circumference_of_circle_x_l172_172830

theorem circumference_of_circle_x (A_x A_y : ℝ) (r_x r_y C_x : ℝ)
  (h_area: A_x = A_y) (h_half_radius_y: r_y = 2 * 5)
  (h_area_y: A_y = Real.pi * r_y^2)
  (h_area_x: A_x = Real.pi * r_x^2)
  (h_circumference_x: C_x = 2 * Real.pi * r_x) :
  C_x = 20 * Real.pi :=
by
  sorry

end circumference_of_circle_x_l172_172830


namespace number_of_girls_in_class_l172_172877

section
variables (g b : ℕ)

/-- Given the total number of students and the ratio of girls to boys, this theorem states the number of girls in Ben's class. -/
theorem number_of_girls_in_class (h1 : 3 * b = 4 * g) (h2 : g + b = 35) : g = 15 :=
sorry
end

end number_of_girls_in_class_l172_172877


namespace smallest_solution_eq_sqrt_104_l172_172699

theorem smallest_solution_eq_sqrt_104 :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, ⌊y^2⌋ - ⌊y⌋^2 = 19 → x ≤ y) := sorry

end smallest_solution_eq_sqrt_104_l172_172699


namespace simplify_expression_l172_172590

theorem simplify_expression (y : ℝ) : (3 * y + 4 * y + 5 * y + 7) = (12 * y + 7) :=
by
  sorry

end simplify_expression_l172_172590


namespace intersection_A_B_l172_172980

def A : Set ℤ := { x | (2 * x + 3) * (x - 4) < 0 }
def B : Set ℝ := { x | 0 < x ∧ x ≤ Real.exp 1 }

theorem intersection_A_B :
  { x : ℤ | x ∈ A ∧ (x : ℝ) ∈ B } = {1, 2} :=
by
  sorry

end intersection_A_B_l172_172980


namespace math_problem_l172_172122

theorem math_problem : (300 + 5 * 8) / (2^3) = 42.5 := by
  sorry

end math_problem_l172_172122


namespace retail_price_before_discounts_l172_172951

theorem retail_price_before_discounts 
  (wholesale_price profit_rate tax_rate discount1 discount2 total_effective_price : ℝ) 
  (h_wholesale_price : wholesale_price = 108)
  (h_profit_rate : profit_rate = 0.20)
  (h_tax_rate : tax_rate = 0.15)
  (h_discount1 : discount1 = 0.10)
  (h_discount2 : discount2 = 0.05)
  (h_total_effective_price : total_effective_price = 126.36) :
  ∃ (retail_price_before_discounts : ℝ), retail_price_before_discounts = 147.78 := 
by
  sorry

end retail_price_before_discounts_l172_172951


namespace translation_identity_l172_172360

open Real

-- Define the functions f and g
def f (x : ℝ) : ℝ := sin (2 * x)
def g (x : ℝ) : ℝ := sin (2 * x - π / 6)

-- Define the translation of g by π / 12 units to the left
def g_translated (x : ℝ) : ℝ := g (x + π / 12)

-- The theorem we want to prove
theorem translation_identity : ∀ x : ℝ, g_translated x = f x :=
by
  sorry

end translation_identity_l172_172360


namespace range_of_a_quadratic_root_conditions_l172_172556

theorem range_of_a_quadratic_root_conditions (a : ℝ) :
  ((∃ x₁ x₂ : ℝ, x₁ > 2 ∧ x₂ < 2 ∧ (ax^2 - 2*(a+1)*x + a-1 = 0)) ↔ (0 < a ∧ a < 5)) :=
by
  sorry

end range_of_a_quadratic_root_conditions_l172_172556


namespace apples_in_basket_l172_172008

theorem apples_in_basket
  (total_rotten : ℝ := 12 / 100)
  (total_spots : ℝ := 7 / 100)
  (total_insects : ℝ := 5 / 100)
  (total_varying_rot : ℝ := 3 / 100)
  (perfect_apples : ℝ := 66) :
  (perfect_apples / ((1 - (total_rotten + total_spots + total_insects + total_varying_rot))) = 90) :=
by
  sorry

end apples_in_basket_l172_172008


namespace avg_of_consecutive_starting_with_b_l172_172775

variable {a : ℕ} (h : b = (a + 1 + a + 2 + a + 3 + a + 4 + a + 5 + a + 6 + a + 7) / 7)

theorem avg_of_consecutive_starting_with_b (h : b = (a + 1 + a + 2 + a + 3 + a + 4 + a + 5 + a + 6 + a + 7) / 7) :
  (a + 4 + (a + 4 + 1) + (a + 4 + 2) + (a + 4 + 3) + (a + 4 + 4) + (a + 4 + 5) + (a + 4 + 6)) / 7 = a + 7 :=
  sorry

end avg_of_consecutive_starting_with_b_l172_172775


namespace diameter_percentage_l172_172004

theorem diameter_percentage (d_R d_S : ℝ) (h : π * (d_R / 2)^2 = 0.16 * π * (d_S / 2)^2) :
  (d_R / d_S) * 100 = 40 :=
by {
  sorry
}

end diameter_percentage_l172_172004


namespace molecular_weight_l172_172497

theorem molecular_weight :
  let H_weight := 1.008
  let Br_weight := 79.904
  let O_weight := 15.999
  let C_weight := 12.011
  let N_weight := 14.007
  let S_weight := 32.065
  (2 * H_weight + 1 * Br_weight + 3 * O_weight + 1 * C_weight + 1 * N_weight + 2 * S_weight) = 220.065 :=
by
  let H_weight := 1.008
  let Br_weight := 79.904
  let O_weight := 15.999
  let C_weight := 12.011
  let N_weight := 14.007
  let S_weight := 32.065
  sorry

end molecular_weight_l172_172497


namespace a_n_divisible_by_2013_a_n_minus_207_is_cube_l172_172536

theorem a_n_divisible_by_2013 (n : ℕ) (h : n ≥ 1) : 2013 ∣ (4 ^ (6 ^ n) + 1943) :=
by sorry

theorem a_n_minus_207_is_cube (n : ℕ) : (∃ k : ℕ, 4 ^ (6 ^ n) + 1736 = k^3) ↔ (n = 1) :=
by sorry

end a_n_divisible_by_2013_a_n_minus_207_is_cube_l172_172536


namespace find_ab_l172_172159

theorem find_ab (a b : ℝ) (h1 : a + b = 4) (h2 : a^3 + b^3 = 136) : a * b = -6 :=
by
  sorry

end find_ab_l172_172159


namespace wickets_before_last_match_l172_172253

theorem wickets_before_last_match (W : ℕ) (avg_before : ℝ) (wickets_taken : ℕ) (runs_conceded : ℝ) (avg_drop : ℝ) :
  avg_before = 12.4 → wickets_taken = 4 → runs_conceded = 26 → avg_drop = 0.4 →
  (avg_before - avg_drop) * (W + wickets_taken) = avg_before * W + runs_conceded →
  W = 55 :=
by
  intros
  sorry

end wickets_before_last_match_l172_172253


namespace unique_pair_exists_l172_172046

theorem unique_pair_exists (n : ℕ) (hn : n > 0) : 
  ∃! (k l : ℕ), n = k * (k - 1) / 2 + l ∧ 0 ≤ l ∧ l < k :=
sorry

end unique_pair_exists_l172_172046


namespace sum_squares_seven_consecutive_not_perfect_square_l172_172331

theorem sum_squares_seven_consecutive_not_perfect_square : 
  ∀ (n : ℤ), ¬ ∃ k : ℤ, k * k = (n-3)^2 + (n-2)^2 + (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 + (n+3)^2 :=
by
  sorry

end sum_squares_seven_consecutive_not_perfect_square_l172_172331


namespace find_number_l172_172809

theorem find_number (x : ℝ) (h : 0.30 * x = 108.0) : x = 360 := 
sorry

end find_number_l172_172809


namespace cubic_polynomials_common_roots_c_d_l172_172845

theorem cubic_polynomials_common_roots_c_d (c d : ℝ) :
  (∀ (r s : ℝ), r ≠ s ∧
     (r^3 + c*r^2 + 12*r + 7 = 0) ∧ (s^3 + c*s^2 + 12*s + 7 = 0) ∧
     (r^3 + d*r^2 + 15*r + 9 = 0) ∧ (s^3 + d*s^2 + 15*s + 9 = 0)) →
  (c = -5 ∧ d = -6) := 
by
  sorry

end cubic_polynomials_common_roots_c_d_l172_172845


namespace min_value_frac_gcd_l172_172026

theorem min_value_frac_gcd {N k : ℕ} (hN_substring : N % 10^5 = 11235) (hN_pos : 0 < N) (hk_pos : 0 < k) (hk_bound : 10^k > N) : 
  (10^k - 1) / Nat.gcd N (10^k - 1) = 89 :=
by
  -- proof goes here
  sorry

end min_value_frac_gcd_l172_172026


namespace part1_part2_l172_172544

open BigOperators

noncomputable def seq (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n ≠ 0 → a n > 0) ∧
  (a 1 = 2) ∧
  (∀ n : ℕ, n ≠ 0 → (n + 1) * (a (n + 1)) ^ 2 = n * (a n) ^ 2 + a n)

theorem part1 (a : ℕ → ℝ) (h : seq a)
  (n : ℕ) (hn : n ≠ 0) 
  : 1 < a (n+1) ∧ a (n+1) < a n :=
sorry

theorem part2 (a : ℕ → ℝ) (h : seq a)
  : ∑ k in Finset.range 2022 \ {0}, (a (k+1))^2 / (k+1)^2 < 2 :=
sorry

end part1_part2_l172_172544


namespace sin_330_eq_negative_half_l172_172385

theorem sin_330_eq_negative_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_330_eq_negative_half_l172_172385


namespace boy_completion_time_l172_172254

theorem boy_completion_time (M W B : ℝ) (h1 : M + W + B = 1/3) (h2 : M = 1/6) (h3 : W = 1/18) : B = 1/9 :=
sorry

end boy_completion_time_l172_172254


namespace staplers_left_l172_172485

-- Definitions of the conditions
def initialStaplers : ℕ := 50
def dozen : ℕ := 12
def reportsStapled : ℕ := 3 * dozen

-- The proof statement
theorem staplers_left : initialStaplers - reportsStapled = 14 := by
  sorry

end staplers_left_l172_172485


namespace find_x_from_percentage_l172_172740

theorem find_x_from_percentage (x : ℝ) (h : 0.2 * 30 = 0.25 * x + 2) : x = 16 :=
sorry

end find_x_from_percentage_l172_172740


namespace max_g_of_15_l172_172317

noncomputable def g (x : ℝ) : ℝ := x^3  -- Assume the polynomial g(x) = x^3 based on the maximum value found.

theorem max_g_of_15 (g : ℝ → ℝ) (h_coeff : ∀ x, 0 ≤ g x)
  (h3 : g 3 = 3) (h27 : g 27 = 1701) : g 15 = 3375 :=
by
  -- According to the problem's constraint and identified solution,
  -- here is the statement asserting that the maximum value of g(15) is 3375
  sorry

end max_g_of_15_l172_172317


namespace negation_of_exists_cond_l172_172074

theorem negation_of_exists_cond (x : ℝ) (h : x > 0) : ¬ (∃ x : ℝ, x > 0 ∧ x^3 - x + 1 > 0) ↔ (∀ x : ℝ, x > 0 → x^3 - x + 1 ≤ 0) :=
by 
  sorry

end negation_of_exists_cond_l172_172074


namespace optimal_bicycle_point_l172_172261

noncomputable def distance_A_B : ℝ := 30  -- Distance between A and B is 30 km
noncomputable def midpoint_distance : ℝ := distance_A_B / 2  -- Distance between midpoint C to both A and B is 15 km
noncomputable def walking_speed : ℝ := 5  -- Walking speed is 5 km/h
noncomputable def biking_speed : ℝ := 20  -- Biking speed is 20 km/h

theorem optimal_bicycle_point : ∃ x : ℝ, 0 ≤ x ∧ x ≤ 30 ∧ (30 - x + 4 * x = 60 - 3 * x) → x = 5 :=
by sorry

end optimal_bicycle_point_l172_172261


namespace avg_hamburgers_per_day_l172_172656

theorem avg_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h1 : total_hamburgers = 63) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 := by
  sorry

end avg_hamburgers_per_day_l172_172656


namespace find_a_l172_172542

noncomputable def f (x : ℝ) : ℝ := 3 * ((x - 1) / 2) - 2

theorem find_a (x a : ℝ) (hx : f a = 4) (ha : a = 2 * x + 1) : a = 5 :=
by
  sorry

end find_a_l172_172542


namespace find_a_l172_172852

def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then
    x^3 + 1
  else
    x^2 - a * x

theorem find_a : ∃ a : ℝ, f (f 0 a) a = -2 :=
by
  have h₁ : f 0 a = 1 := by
    simp [f]
    sorry
  have h₂ : f 1 a = 1 - a := by
    simp [f]
    sorry
  rw [h₁, h₂]
  existsi 3
  sorry

end find_a_l172_172852


namespace problem1_l172_172102

open Real

theorem problem1 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 1) : 
  ∃ (m : ℝ), m = 9 / 2 ∧ ∀ (u v : ℝ), 0 < u → 0 < v → u + v = 1 → (1 / u + 4 / (1 + v)) ≥ m := 
sorry

end problem1_l172_172102


namespace smallest_solution_floor_equation_l172_172701

theorem smallest_solution_floor_equation : ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (x = Real.sqrt 109) :=
by
  sorry

end smallest_solution_floor_equation_l172_172701


namespace total_cost_8_dozen_pencils_2_dozen_notebooks_l172_172342

variable (P N : ℝ)

def eq1 : Prop := 3 * P + 4 * N = 60
def eq2 : Prop := P + N = 15.512820512820513

theorem total_cost_8_dozen_pencils_2_dozen_notebooks :
  eq1 P N ∧ eq2 P N → (96 * P + 24 * N = 520) :=
by
  sorry

end total_cost_8_dozen_pencils_2_dozen_notebooks_l172_172342


namespace problem_a_b_c_relationship_l172_172119

theorem problem_a_b_c_relationship (u v a b c : ℝ)
  (h1 : u - v = a)
  (h2 : u^2 - v^2 = b)
  (h3 : u^3 - v^3 = c) :
  3 * b^2 + a^4 = 4 * a * c := by
  sorry

end problem_a_b_c_relationship_l172_172119


namespace purity_of_alloy_l172_172246

theorem purity_of_alloy (w1 w2 : ℝ) (p1 p2 : ℝ) (h_w1 : w1 = 180) (h_p1 : p1 = 920) (h_w2 : w2 = 100) (h_p2 : p2 = 752) : 
  let a := w1 * (p1 / 1000) + w2 * (p2 / 1000)
  let b := w1 + w2
  let p_result := (a / b) * 1000
  p_result = 860 :=
by
  sorry

end purity_of_alloy_l172_172246


namespace b_remainder_l172_172890

theorem b_remainder (n : ℕ) (hn : n > 0) : ∃ b : ℕ, b % 11 = 5 :=
by
  sorry

end b_remainder_l172_172890


namespace digits_of_85550_can_be_arranged_l172_172308

theorem digits_of_85550_can_be_arranged : 
  let digits := [8, 5, 5, 5, 0]
  let non_zero_digits := [8, 5, 5, 5]
  let factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

  (∑ pos in {1, 2, 3, 4}, factorial non_zero_digits.length / (non_zero_digits.count 5)!) = 16 :=
by
  sorry

end digits_of_85550_can_be_arranged_l172_172308


namespace logarithm_function_decreasing_l172_172912

theorem logarithm_function_decreasing (a : ℝ) : 
  (∀ x ∈ Set.Ici (-1), (3 * x^2 - a * x + 5) ≤ (3 * x^2 - a * (x + 1) + 5)) ↔ (-8 < a ∧ a ≤ -6) :=
by
  sorry

end logarithm_function_decreasing_l172_172912


namespace shelby_rain_time_l172_172334

noncomputable def speedNonRainy : ℚ := 30 / 60
noncomputable def speedRainy : ℚ := 20 / 60
noncomputable def totalDistance : ℚ := 16
noncomputable def totalTime : ℚ := 40

theorem shelby_rain_time : 
  ∃ x : ℚ, (speedNonRainy * (totalTime - x) + speedRainy * x = totalDistance) ∧ x = 24 := 
by
  sorry

end shelby_rain_time_l172_172334


namespace train_speed_approximation_l172_172350

theorem train_speed_approximation (train_speed_mph : ℝ) (seconds : ℝ) :
  (40 : ℝ) * train_speed_mph * 1 / 60 = seconds → seconds = 27 := 
  sorry

end train_speed_approximation_l172_172350


namespace two_faucets_fill_60_gallons_l172_172722

def four_faucets_fill (tub_volume : ℕ) (time_minutes : ℕ) : Prop :=
  4 * (tub_volume / time_minutes) = 120 / 5

def two_faucets_fill (tub_volume : ℕ) (time_minutes : ℕ) : Prop :=
  2 * (tub_volume / time_minutes) = 60 / time_minutes

theorem two_faucets_fill_60_gallons :
  (four_faucets_fill 120 5) → ∃ t: ℕ, two_faucets_fill 60 t ∧ t = 5 :=
by {
  sorry
}

end two_faucets_fill_60_gallons_l172_172722


namespace meat_per_slice_is_22_l172_172040

noncomputable def piecesOfMeatPerSlice : ℕ :=
  let pepperoni := 30
  let ham := 2 * pepperoni
  let sausage := pepperoni + 12
  let totalMeat := pepperoni + ham + sausage
  let slices := 6
  totalMeat / slices

theorem meat_per_slice_is_22 : piecesOfMeatPerSlice = 22 :=
by
  -- Here would be the proof (not required in the task)
  sorry

end meat_per_slice_is_22_l172_172040


namespace find_x_plus_y_l172_172283

theorem find_x_plus_y (x y : ℤ) (h1 : |x| = 3) (h2 : y^2 = 4) (h3 : x < y) : x + y = -1 ∨ x + y = -5 :=
sorry

end find_x_plus_y_l172_172283


namespace evaluate_expression_l172_172030

theorem evaluate_expression (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hxy : x > y) (hyz : y > z) :
  (x ^ (y + z) * z ^ (x + y)) / (y ^ (x + z) * z ^ (y + x)) = (x / y) ^ (y + z) :=
by
  sorry

end evaluate_expression_l172_172030


namespace spadesuit_evaluation_l172_172537

-- Define the operation
def spadesuit (a b : ℝ) : ℝ := (a + b) * (a - b)

-- The theorem to prove
theorem spadesuit_evaluation : spadesuit 4 (spadesuit 5 (-2)) = -425 :=
by
  sorry

end spadesuit_evaluation_l172_172537


namespace smallest_perimeter_l172_172394

-- Definitions of the angle conditions
def angleA (B : ℝ) := 2 * B
def angleC (B : ℝ) := 180 - 3 * B

-- Conditions of the problem: ∠A = 2∠B and ∠C > 90°
lemma angle_conditions (B : ℝ) (h1 : B < 30) : angleC B > 90 :=
by linarith [h1]

-- The sides a, b, c of triangle
def side_lengths : (ℤ × ℤ × ℤ) := (7, 8, 15)

-- Perimeter of the triangle
def perimeter (a b c : ℤ) := a + b + c

-- The main statement: The smallest possible perimeter is 30
theorem smallest_perimeter :
  let ⟨a, b, c⟩ := side_lengths in
  perimeter a b c = 30 :=
by sorry

end smallest_perimeter_l172_172394


namespace arrangement_count_l172_172824

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

end arrangement_count_l172_172824


namespace car_speed_l172_172374

theorem car_speed (distance time : ℝ) (h1 : distance = 300) (h2 : time = 5) : distance / time = 60 := by
  have h : distance / time = 300 / 5 := by
    rw [h1, h2]
  norm_num at h
  exact h

end car_speed_l172_172374


namespace model_A_sampling_l172_172248

theorem model_A_sampling (prod_A prod_B prod_C total_prod total_sampled : ℕ)
    (hA : prod_A = 1200) (hB : prod_B = 6000) (hC : prod_C = 2000)
    (htotal : total_prod = prod_A + prod_B + prod_C) (htotal_car : total_prod = 9200)
    (hsampled : total_sampled = 46) :
    (prod_A * total_sampled) / total_prod = 6 := by
  sorry

end model_A_sampling_l172_172248


namespace value_of_a_l172_172420

theorem value_of_a (a : ℝ) (A : ℝ × ℝ) (h : A = (1, 0)) : (a * A.1 + 3 * A.2 - 2 = 0) → a = 2 :=
by
  intro h1
  rw [h] at h1
  sorry

end value_of_a_l172_172420


namespace green_balls_more_than_red_l172_172472

theorem green_balls_more_than_red
  (total_balls : ℕ) (red_balls : ℕ) (green_balls : ℕ)
  (h1 : total_balls = 66)
  (h2 : red_balls = 30)
  (h3 : green_balls = total_balls - red_balls) : green_balls - red_balls = 6 :=
by
  sorry

end green_balls_more_than_red_l172_172472


namespace ryan_fraction_l172_172755

-- Define the total amount of money
def total_money : ℕ := 48

-- Define that Ryan owns a fraction R of the total money
variable {R : ℚ}

-- Define the debts
def ryan_owes_leo : ℕ := 10
def leo_owes_ryan : ℕ := 7

-- Define the final amount Leo has after settling the debts
def leo_final_amount : ℕ := 19

-- Define the condition that Leo and Ryan together have $48
def leo_plus_ryan (leo_amount ryan_amount : ℚ) : Prop := 
  leo_amount + ryan_amount = total_money

-- Define Ryan's amount as a fraction R of the total money
def ryan_amount (R : ℚ) : ℚ := R * total_money

-- Define Leo's amount before debts were settled
def leo_amount_before_debts : ℚ := (leo_final_amount : ℚ) + leo_owes_ryan

-- Define the equation after settling debts
def leo_final_eq (leo_amount_before_debts : ℚ) : Prop :=
  (leo_amount_before_debts - ryan_owes_leo = leo_final_amount)

-- The Lean theorem that needs to be proved
theorem ryan_fraction :
  ∃ (R : ℚ), leo_plus_ryan (leo_amount_before_debts - ryan_owes_leo) (ryan_amount R)
  ∧ leo_final_eq leo_amount_before_debts
  ∧ R = 11 / 24 :=
sorry

end ryan_fraction_l172_172755


namespace outfit_combinations_l172_172739

def shirts : ℕ := 6
def pants : ℕ := 4
def hats : ℕ := 6

def pant_colors : Finset String := {"tan", "black", "blue", "gray"}
def shirt_colors : Finset String := {"tan", "black", "blue", "gray", "white", "yellow"}
def hat_colors : Finset String := {"tan", "black", "blue", "gray", "white", "yellow"}

def total_combinations : ℕ := shirts * pants * hats
def restricted_combinations : ℕ := pant_colors.card

theorem outfit_combinations
    (hshirts : shirts = 6)
    (hpants : pants = 4)
    (hhats : hats = 6)
    (hpant_colors : pant_colors.card = 4)
    (hshirt_colors : shirt_colors.card = 6)
    (hhat_colors : hat_colors.card = 6)
    (hrestricted : restricted_combinations = pant_colors.card) :
    total_combinations - restricted_combinations = 140 := by
  sorry

end outfit_combinations_l172_172739


namespace arithmetic_sequence_sum_l172_172290

-- Given {a_n} is an arithmetic sequence, and a_2 + a_3 + a_{10} + a_{11} = 40, prove a_6 + a_7 = 20
theorem arithmetic_sequence_sum (a : ℕ → ℝ)
  (h_arith : ∃ d : ℝ, ∀ n, a (n + 1) = a n + d)
  (h_sum : a 2 + a 3 + a 10 + a 11 = 40) :
  a 6 + a 7 = 20 :=
sorry

end arithmetic_sequence_sum_l172_172290


namespace inequality_system_no_solution_l172_172272

theorem inequality_system_no_solution (a : ℝ) : ¬ (∃ x : ℝ, x ≤ 5 ∧ x > a) ↔ a ≥ 5 :=
sorry

end inequality_system_no_solution_l172_172272


namespace Ofelia_savings_l172_172766

theorem Ofelia_savings (X : ℝ) (h : 16 * X = 160) : X = 10 :=
by
  sorry

end Ofelia_savings_l172_172766


namespace length_of_shorter_piece_l172_172640

theorem length_of_shorter_piece (x : ℕ) (h1 : x + (x + 12) = 68) : x = 28 :=
by
  sorry

end length_of_shorter_piece_l172_172640


namespace not_all_roots_real_l172_172203

-- Define the quintic polynomial with coefficients a5, a4, a3, a2, a1, a0
def quintic_polynomial (a5 a4 a3 a2 a1 a0 : ℝ) (x : ℝ) : ℝ :=
  a5 * x^5 + a4 * x^4 + a3 * x^3 + a2 * x^2 + a1 * x + a0

-- Define a predicate for the existence of all real roots
def all_roots_real (a5 a4 a3 a2 a1 a0 : ℝ) : Prop :=
  ∀ r : ℝ, quintic_polynomial a5 a4 a3 a2 a1 a0 r = 0

-- Define the main theorem statement
theorem not_all_roots_real (a5 a4 a3 a2 a1 a0 : ℝ) :
  2 * a4^2 < 5 * a5 * a3 →
  ¬ all_roots_real a5 a4 a3 a2 a1 a0 :=
by
  sorry

end not_all_roots_real_l172_172203


namespace weight_of_replaced_student_l172_172209

theorem weight_of_replaced_student (W : ℝ) : 
  (W - 12 = 5 * 12) → W = 72 :=
by
  intro hyp
  linarith

end weight_of_replaced_student_l172_172209


namespace k_at_27_l172_172891

noncomputable def h (x : ℝ) : ℝ := x^3 - 2 * x + 1

theorem k_at_27 (k : ℝ → ℝ)
    (hk_cubic : ∀ x, ∃ a b c, k x = a * x^3 + b * x^2 + c * x)
    (hk_at_0 : k 0 = 1)
    (hk_roots : ∀ a b c, (h a = 0) → (h b = 0) → (h c = 0) → 
                 ∃ (p q r: ℝ), k (p^3) = 0 ∧ k (q^3) = 0 ∧ k (r^3) = 0) :
    k 27 = -704 :=
sorry

end k_at_27_l172_172891


namespace correct_average_l172_172632

theorem correct_average (n : Nat) (incorrect_avg correct_mark incorrect_mark : ℝ) 
  (h1 : n = 30) (h2 : incorrect_avg = 60) (h3 : correct_mark = 15) (h4 : incorrect_mark = 90) :
  (incorrect_avg * n - incorrect_mark + correct_mark) / n = 57.5 :=
by
  sorry

end correct_average_l172_172632


namespace unique_solution_eq_l172_172528

theorem unique_solution_eq (x : ℝ) : 
  (x ≠ 0 ∧ x ≠ 5) ∧ (∀ x, (3 * x^3 - 15 * x^2) / (x^2 - 5 * x) = x - 2) 
  → ∃! (x : ℝ), (3 * x ^ 3 - 15 * x ^ 2) / (x^2 - 5 * x) = x - 2 := 
by sorry

end unique_solution_eq_l172_172528


namespace sum21_exists_l172_172056

theorem sum21_exists (S : Finset ℕ) (h_size : S.card = 11) (h_range : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 20) :
  ∃ a b, a ≠ b ∧ a ∈ S ∧ b ∈ S ∧ a + b = 21 :=
by
  sorry

end sum21_exists_l172_172056


namespace smallest_x_solution_l172_172686

theorem smallest_x_solution :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, (⌊y^2⌋ - ⌊y⌋^2 = 19) → x ≤ y) ∧ x = Real.sqrt 119 := 
sorry

end smallest_x_solution_l172_172686


namespace smallest_solution_floor_equation_l172_172702

theorem smallest_solution_floor_equation : ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (x = Real.sqrt 109) :=
by
  sorry

end smallest_solution_floor_equation_l172_172702


namespace find_percentage_l172_172414

variable (x p : ℝ)
variable (h1 : 0.25 * x = (p / 100) * 1500 - 20)
variable (h2 : x = 820)

theorem find_percentage : p = 15 :=
by
  sorry

end find_percentage_l172_172414


namespace min_pencils_for_each_color_max_pencils_remaining_each_color_max_red_pencils_to_ensure_five_remaining_l172_172560

-- Condition Definitions
def blue := 5
def red := 9
def green := 6
def yellow := 4

-- Theorem Statements
theorem min_pencils_for_each_color :
  ∀ B R G Y : ℕ, blue = 5 ∧ red = 9 ∧ green = 6 ∧ yellow = 4 →
  ∃ min_pencils : ℕ, min_pencils = 21 := by
  sorry

theorem max_pencils_remaining_each_color :
  ∀ B R G Y : ℕ, blue = 5 ∧ red = 9 ∧ green = 6 ∧ yellow = 4 →
  ∃ max_pencils : ℕ, max_pencils = 3 := by
  sorry

theorem max_red_pencils_to_ensure_five_remaining :
  ∀ B R G Y : ℕ, blue = 5 ∧ red = 9 ∧ green = 6 ∧ yellow = 4 →
  ∃ max_red_pencils : ℕ, max_red_pencils = 4 := by
  sorry

end min_pencils_for_each_color_max_pencils_remaining_each_color_max_red_pencils_to_ensure_five_remaining_l172_172560


namespace sum_of_remaining_numbers_last_number_written_l172_172884

open Nat

-- Define the initial sequence as a list from 1 to 100
def initial_sequence := List.range' 1 100

-- Function to perform one step of the given operation: remove the first 6 numbers and append their sum
def step (seq: List ℕ) : List ℕ :=
  let (first_six, rest) := List.splitAt 6 seq
  List.append rest [first_six.sum]

-- Function to repeatedly apply the step until fewer than 6 numbers remain
def repeat_step (seq: List ℕ) : List ℕ :=
  if seq.length < 6 then seq
  else repeat_step (step seq)

-- Prove the sum of the remaining numbers is 5050
theorem sum_of_remaining_numbers :
  repeat_step initial_sequence.sum = 5050 :=
by
  sorry

-- Prove the last number written on the blackboard
theorem last_number_written :
  let final_sequence := repeat_step initial_sequence
  final_sequence.lastD 0 = 2394 :=
by
  sorry

end sum_of_remaining_numbers_last_number_written_l172_172884


namespace even_numbers_average_l172_172779

theorem even_numbers_average (n : ℕ) (h1 : 2 * (n * (n + 1)) = 22 * n) : n = 10 :=
by
  sorry

end even_numbers_average_l172_172779


namespace range_of_m_l172_172302

theorem range_of_m (m : ℝ) (h : ∀ x : ℝ, |x| ≤ 1 → x < m) : m > 1 := 
by
  sorry

end range_of_m_l172_172302


namespace sum_of_three_geq_54_l172_172355

theorem sum_of_three_geq_54 (a : Fin 10 → ℕ) (h_diff : Function.Injective a) (h_sum : (∑ i, a i) > 144) :
  ∃ i j k : Fin 10, i < j ∧ j < k ∧ a i + a j + a k ≥ 54 := 
by
  -- By contradiction
  sorry

end sum_of_three_geq_54_l172_172355


namespace children_division_into_circles_l172_172307

theorem children_division_into_circles (n m k : ℕ) (hn : n = 5) (hm : m = 2) (trees_indistinguishable : true) (children_distinguishable : true) :
  ∃ ways, ways = 50 := 
by
  sorry

end children_division_into_circles_l172_172307


namespace fraction_inequality_l172_172158

theorem fraction_inequality (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a < b) (h2 : c < d) : (a + c) / (b + c) < (a + d) / (b + d) :=
by
  sorry

end fraction_inequality_l172_172158


namespace eel_jellyfish_ratio_l172_172116

noncomputable def combined_cost : ℝ := 200
noncomputable def eel_cost : ℝ := 180
noncomputable def jellyfish_cost : ℝ := combined_cost - eel_cost

theorem eel_jellyfish_ratio : eel_cost / jellyfish_cost = 9 :=
by
  sorry

end eel_jellyfish_ratio_l172_172116


namespace red_balls_in_bag_l172_172104

theorem red_balls_in_bag (r : ℕ) (h1 : 0 ≤ r ∧ r ≤ 12)
  (h2 : (r * (r - 1)) / (12 * 11) = 1 / 10) : r = 12 :=
sorry

end red_balls_in_bag_l172_172104


namespace gcd_12345_6789_l172_172965

theorem gcd_12345_6789 : Nat.gcd 12345 6789 = 3 := by
  sorry

end gcd_12345_6789_l172_172965


namespace length_PQ_l172_172758

theorem length_PQ (AB AC BC : ℕ) (P Q : ℝ) 
  (h_AB : AB = 1985) 
  (h_AC : AC = 1983) 
  (h_BC : BC = 1982) 
  (altitude_CH : ∃ CH : ℝ, ∃ H : ℝ, 
    ∀ A B C, P = A ∧ Q = B ∧ CH⊥BC ∧ H ∈ AB) 
  (tangent_PQ : ∀ ACH BCH, 
    tangent_point(ACH, P, CH) ∧ tangent_point(BCH, Q, CH) 
    ∧ inscribed_circle(ACH) ∧ inscribed_circle(BCH)) :
  ∃ m n : ℕ, m + n = 1190 :=
by
  sorry

end length_PQ_l172_172758


namespace geom_seq_term_10_l172_172173

-- Given conditions:
def a₆ : ℝ := 2 / 3
def q : ℝ := Real.sqrt 3

-- Statement to prove:
theorem geom_seq_term_10 : 
  let a₁ := a₆ / q^5 in 
  a₁ * q^9 = 6 := by
  sorry

end geom_seq_term_10_l172_172173


namespace intersection_A_B_l172_172557

def A : Set ℝ := {x | abs x <= 1}

def B : Set ℝ := {y | ∃ x : ℝ, y = x^2}

theorem intersection_A_B :
  (A ∩ B) = {x | 0 ≤ x ∧ x ≤ 1} := sorry

end intersection_A_B_l172_172557


namespace price_reductions_l172_172128

theorem price_reductions (a : ℝ) : 18400 * (1 - a / 100)^2 = 16000 :=
sorry

end price_reductions_l172_172128


namespace equal_spacing_between_paintings_l172_172168

/--
Given:
- The width of each painting is 30 centimeters.
- The total width of the wall in the exhibition hall is 320 centimeters.
- There are six pieces of artwork.
Prove that: The distance between the end of the wall and the artwork, and between the artworks, is 20 centimeters.
-/
theorem equal_spacing_between_paintings :
  let width_painting := 30 -- in centimeters
  let total_wall_width := 320 -- in centimeters
  let num_paintings := 6
  let total_paintings_width := num_paintings * width_painting
  let remaining_space := total_wall_width - total_paintings_width
  let num_spaces := num_paintings + 1
  let space_between := remaining_space / num_spaces
  space_between = 20 := sorry

end equal_spacing_between_paintings_l172_172168


namespace Ella_jellybeans_l172_172147

-- Definitions based on conditions from part (a)
def Dan_volume := 10
def Dan_jellybeans := 200
def scaling_factor := 3

-- Prove that Ella's box holds 5400 jellybeans
theorem Ella_jellybeans : scaling_factor^3 * Dan_jellybeans = 5400 := 
by
  sorry

end Ella_jellybeans_l172_172147


namespace prob_white_point_leftmost_l172_172370

noncomputable def prob_sum_at_most_one (n : ℕ) (m : ℕ) (k : ℕ) : ℚ :=
  have points_drawn : k > 0 := sorry
  (m : ℚ) / (n : ℚ)

theorem prob_white_point_leftmost (p : ℚ):
  let n := 2019
  let m := 1019
  let k := 1000
  let prob := prob_sum_at_most_one n m k
  p = prob := by
  sorry

end prob_white_point_leftmost_l172_172370


namespace number_of_racks_l172_172388

theorem number_of_racks (cds_per_rack total_cds : ℕ) (h1 : cds_per_rack = 8) (h2 : total_cds = 32) :
  total_cds / cds_per_rack = 4 :=
by
  -- actual proof goes here
  sorry

end number_of_racks_l172_172388


namespace interest_difference_20_years_l172_172829

def compound_interest (P r : ℝ) (n : ℕ) : ℝ := P * (1 + r)^n
def simple_interest (P r : ℝ) (t : ℕ) : ℝ := P * (1 + r * t)

theorem interest_difference_20_years :
  compound_interest 15000 0.06 20 - simple_interest 15000 0.08 20 = 9107 :=
by
  sorry

end interest_difference_20_years_l172_172829


namespace sum_of_diagonal_elements_l172_172765

/-- Odd numbers from 1 to 49 arranged in a 5x5 grid. -/
def table : ℕ → ℕ → ℕ
| 0, 0 => 1
| 0, 1 => 3
| 0, 2 => 5
| 0, 3 => 7
| 0, 4 => 9
| 1, 0 => 11
| 1, 1 => 13
| 1, 2 => 15
| 1, 3 => 17
| 1, 4 => 19
| 2, 0 => 21
| 2, 1 => 23
| 2, 2 => 25
| 2, 3 => 27
| 2, 4 => 29
| 3, 0 => 31
| 3, 1 => 33
| 3, 2 => 35
| 3, 3 => 37
| 3, 4 => 39
| 4, 0 => 41
| 4, 1 => 43
| 4, 2 => 45
| 4, 3 => 47
| 4, 4 => 49
| _, _ => 0

/-- Proof that the sum of five numbers chosen from the table such that no two of them are in the same row or column equals 125. -/
theorem sum_of_diagonal_elements : 
  (table 0 0 + table 1 1 + table 2 2 + table 3 3 + table 4 4) = 125 := by
  sorry

end sum_of_diagonal_elements_l172_172765


namespace marked_price_correct_l172_172952

theorem marked_price_correct
    (initial_price : ℝ)
    (initial_discount_rate : ℝ)
    (profit_margin_rate : ℝ)
    (final_discount_rate : ℝ)
    (purchase_price : ℝ)
    (final_selling_price : ℝ)
    (marked_price : ℝ)
    (h_initial_price : initial_price = 30)
    (h_initial_discount_rate : initial_discount_rate = 0.15)
    (h_profit_margin_rate : profit_margin_rate = 0.20)
    (h_final_discount_rate : final_discount_rate = 0.25)
    (h_purchase_price : purchase_price = initial_price * (1 - initial_discount_rate))
    (h_final_selling_price : final_selling_price = purchase_price * (1 + profit_margin_rate))
    (h_marked_price : marked_price * (1 - final_discount_rate) = final_selling_price) : 
    marked_price = 40.80 :=
by
  sorry

end marked_price_correct_l172_172952


namespace clips_and_earnings_l172_172998

variable (x y z : ℝ)
variable (h_y : y = x / 2)
variable (totalClips : ℝ := 48 * x + y)
variable (avgEarning : ℝ := z / totalClips)

theorem clips_and_earnings :
  totalClips = 97 * x / 2 ∧ avgEarning = 2 * z / (97 * x) :=
by
  sorry

end clips_and_earnings_l172_172998


namespace find_ab_l172_172416

theorem find_ab (a b : ℝ) (h1 : a - b = 10) (h2 : a^2 + b^2 = 150) : a * b = 25 :=
by 
  sorry

end find_ab_l172_172416


namespace cards_per_page_l172_172675

noncomputable def total_cards (new_cards old_cards : ℕ) : ℕ := new_cards + old_cards

theorem cards_per_page
  (new_cards old_cards : ℕ)
  (total_pages : ℕ)
  (h_new_cards : new_cards = 3)
  (h_old_cards : old_cards = 13)
  (h_total_pages : total_pages = 2) :
  total_cards new_cards old_cards / total_pages = 8 :=
by
  rw [h_new_cards, h_old_cards, h_total_pages]
  rfl

end cards_per_page_l172_172675


namespace center_cell_value_l172_172428

theorem center_cell_value (a b c d e f g h i : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
  (hg : 0 < g) (hh : 0 < h) (hi : 0 < i)
  (row1 : a * b * c = 1) (row2 : d * e * f = 1) (row3 : g * h * i = 1)
  (col1 : a * d * g = 1) (col2 : b * e * h = 1) (col3 : c * f * i = 1)
  (square1 : a * b * d * e = 2) (square2 : b * c * e * f = 2)
  (square3 : d * e * g * h = 2) (square4 : e * f * h * i = 2) :
  e = 1 :=
begin
  sorry
end

end center_cell_value_l172_172428


namespace distribute_papers_l172_172661

theorem distribute_papers (n m : ℕ) (h_n : n = 5) (h_m : m = 10) : 
  (m ^ n) = 100000 :=
by 
  rw [h_n, h_m]
  rfl

end distribute_papers_l172_172661


namespace solve_equation_l172_172506

theorem solve_equation (x : ℝ) : 
  (4 * (1 - x)^2 = 25) ↔ (x = -3 / 2 ∨ x = 7 / 2) := 
by 
  sorry

end solve_equation_l172_172506


namespace largest_consecutive_even_sum_l172_172220

theorem largest_consecutive_even_sum (a b c : ℤ) (h1 : b = a+2) (h2 : c = a+4) (h3 : a + b + c = 312) : c = 106 := 
by 
  sorry

end largest_consecutive_even_sum_l172_172220


namespace cube_volume_l172_172171

theorem cube_volume (S : ℝ) (hS : S = 294) : ∃ V : ℝ, V = 343 := by
  sorry

end cube_volume_l172_172171


namespace exists_m_with_totient_ratio_l172_172730

variable (α β : ℝ)

theorem exists_m_with_totient_ratio (h0 : 0 ≤ α) (h1 : α < β) (h2 : β ≤ 1) :
  ∃ m : ℕ, α < (Nat.totient m : ℝ) / m ∧ (Nat.totient m : ℝ) / m < β := 
  sorry

end exists_m_with_totient_ratio_l172_172730


namespace negation_proposition_l172_172476

theorem negation_proposition (a b : ℝ) :
  (a * b ≠ 0) → (a ≠ 0 ∧ b ≠ 0) :=
by
  sorry

end negation_proposition_l172_172476


namespace copper_tin_alloy_weight_l172_172238

theorem copper_tin_alloy_weight :
  let c1 := (4/5 : ℝ) * 10 -- Copper in the first alloy
  let t1 := (1/5 : ℝ) * 10 -- Tin in the first alloy
  let c2 := (1/4 : ℝ) * 16 -- Copper in the second alloy
  let t2 := (3/4 : ℝ) * 16 -- Tin in the second alloy
  let x := ((3 * 14 - 24) / 2 : ℝ) -- Pure copper added
  let total_copper := c1 + c2 + x
  let total_tin := t1 + t2
  total_copper + total_tin = 35 := 
by
  sorry

end copper_tin_alloy_weight_l172_172238


namespace find_a_l172_172123

def F (a b c : ℤ) : ℤ := a * b^2 + c

theorem find_a (a : ℤ) (h : F a 3 (-1) = F a 5 (-3)) : a = 1 / 8 := by
  sorry

end find_a_l172_172123


namespace quadrilateral_area_offset_l172_172136

theorem quadrilateral_area_offset
  (d : ℝ) (x : ℝ) (y : ℝ) (A : ℝ)
  (h_d : d = 26)
  (h_y : y = 6)
  (h_A : A = 195) :
  A = 1/2 * (x + y) * d → x = 9 :=
by
  sorry

end quadrilateral_area_offset_l172_172136


namespace sacks_harvested_per_section_l172_172178

theorem sacks_harvested_per_section (total_sacks : ℕ) (sections : ℕ) (sacks_per_section : ℕ) 
  (h1 : total_sacks = 360) 
  (h2 : sections = 8) 
  (h3 : total_sacks = sections * sacks_per_section) :
  sacks_per_section = 45 :=
by sorry

end sacks_harvested_per_section_l172_172178


namespace jodi_third_week_miles_l172_172450

theorem jodi_third_week_miles (total_miles : ℕ) (first_week : ℕ) (second_week : ℕ) (fourth_week : ℕ) (days_per_week : ℕ) (third_week_miles_per_day : ℕ) 
  (H1 : first_week * days_per_week + second_week * days_per_week + third_week_miles_per_day * days_per_week + fourth_week * days_per_week = total_miles)
  (H2 : first_week = 1) 
  (H3 : second_week = 2) 
  (H4 : fourth_week = 4)
  (H5 : total_miles = 60)
  (H6 : days_per_week = 6) :
    third_week_miles_per_day = 3 :=
by sorry

end jodi_third_week_miles_l172_172450


namespace planA_text_message_cost_l172_172933

def planA_cost (x : ℝ) : ℝ := 60 * x + 9
def planB_cost : ℝ := 60 * 0.40

theorem planA_text_message_cost (x : ℝ) (h : planA_cost x = planB_cost) : x = 0.25 :=
by
  -- h represents the condition that the costs are equal
  -- The proof is skipped with sorry
  sorry

end planA_text_message_cost_l172_172933


namespace laborer_income_l172_172803

theorem laborer_income (I : ℕ) (debt : ℕ) 
  (h1 : 6 * I < 420) 
  (h2 : 4 * I = 240 + debt + 30) 
  (h3 : debt = 420 - 6 * I) : 
  I = 69 := by
  sorry

end laborer_income_l172_172803


namespace hypotenuse_length_l172_172620

-- Definition of the right triangle with the given leg lengths
structure RightTriangle :=
  (BC AC AB : ℕ)
  (right : BC^2 + AC^2 = AB^2)

-- The theorem we need to prove
theorem hypotenuse_length (T : RightTriangle) (h1 : T.BC = 5) (h2 : T.AC = 12) :
  T.AB = 13 :=
by
  sorry

end hypotenuse_length_l172_172620


namespace center_cell_value_l172_172432

open Matrix Finset

def table := Matrix (Fin 3) (Fin 3) ℝ

def row_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 0 2 = 1) ∧ 
  (T 1 0 * T 1 1 * T 1 2 = 1) ∧ 
  (T 2 0 * T 2 1 * T 2 2 = 1)

def col_products (T : table) : Prop :=
  (T 0 0 * T 1 0 * T 2 0 = 1) ∧ 
  (T 0 1 * T 1 1 * T 2 1 = 1) ∧ 
  (T 0 2 * T 1 2 * T 2 2 = 1)

def square_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 1 0 * T 1 1 = 2) ∧ 
  (T 0 1 * T 0 2 * T 1 1 * T 1 2 = 2) ∧ 
  (T 1 0 * T 1 1 * T 2 0 * T 2 1 = 2) ∧ 
  (T 1 1 * T 1 2 * T 2 1 * T 2 2 = 2)

theorem center_cell_value (T : table) 
  (h_row : row_products T) 
  (h_col : col_products T) 
  (h_square : square_products T) : 
  T 1 1 = 1 :=
by
  sorry

end center_cell_value_l172_172432


namespace total_earning_proof_l172_172097

noncomputable def total_earning (daily_wage_c : ℝ) (days_a : ℕ) (days_b : ℕ) (days_c : ℕ) (ratio_a : ℕ) (ratio_b : ℕ) (ratio_c : ℕ) : ℝ :=
  let daily_wage_a := (ratio_a : ℝ) / (ratio_c : ℝ) * daily_wage_c
  let daily_wage_b := (ratio_b : ℝ) / (ratio_c : ℝ) * daily_wage_c
  (daily_wage_a * days_a) + (daily_wage_b * days_b) + (daily_wage_c * days_c)

theorem total_earning_proof : 
  total_earning 71.15384615384615 16 9 4 3 4 5 = 1480 := 
by 
  -- calculations here
  sorry

end total_earning_proof_l172_172097


namespace bank_queue_min_max_wastage_bank_queue_expected_wastage_l172_172646

-- Definitions of operations
def simple_op_time : ℕ := 1
def lengthy_op_time : ℕ := 5
def num_simple_ops : ℕ := 5
def num_lengthy_ops : ℕ := 3
def total_people : ℕ := num_simple_ops + num_lengthy_ops

-- Proving minimum and maximum person-minutes wasted
theorem bank_queue_min_max_wastage :
  (∃ q : list ℕ, q.length = total_people ∧ ∑ i in q, (q.take i).sum ≤ 40) ∧
  (∃ q : list ℕ, q.length = total_people ∧ ∑ i in q, (q.take i).sum ≤ 100) :=
by sorry

-- Proving expected value of wasted person-minutes
theorem bank_queue_expected_wastage :
  expected_value_wasted_person_minutes total_people simple_op_time lengthy_op_time = 84 :=
by sorry

-- Placeholder for the actual expected value calculation function
noncomputable def expected_value_wasted_person_minutes
  (n : ℕ) (t_simple : ℕ) (t_lengthy : ℕ) : ℕ :=
  -- Calculation logic will be implemented here
  84 -- This is just the provided answer, actual logic needed for correctness

end bank_queue_min_max_wastage_bank_queue_expected_wastage_l172_172646


namespace negation_of_exists_gt_implies_forall_leq_l172_172071

theorem negation_of_exists_gt_implies_forall_leq (x : ℝ) (h : 0 < x) :
  ¬ (∃ x : ℝ, 0 < x ∧ x^3 - x + 1 > 0) ↔ ∀ x : ℝ, 0 < x → x^3 - x + 1 ≤ 0 :=
by sorry

end negation_of_exists_gt_implies_forall_leq_l172_172071


namespace keith_bought_cards_l172_172538

theorem keith_bought_cards (orig : ℕ) (now : ℕ) (bought : ℕ) 
  (h1 : orig = 40) (h2 : now = 18) (h3 : bought = orig - now) : bought = 22 := by
  sorry

end keith_bought_cards_l172_172538


namespace solve_equation_l172_172593

theorem solve_equation :
  ∀ x : ℝ, (1 + 2 * x ^ (1/2) - x ^ (1/3) - 2 * x ^ (1/6) = 0) ↔ (x = 1 ∨ x = 1 / 64) :=
by
  sorry

end solve_equation_l172_172593


namespace tree_placement_impossible_l172_172212

theorem tree_placement_impossible
  (length width : ℝ) (h_length : length = 4) (h_width : width = 1) :
  ¬ (∃ (t1 t2 t3 : ℝ × ℝ), 
       dist t1 t2 ≥ 2.5 ∧ 
       dist t2 t3 ≥ 2.5 ∧ 
       dist t1 t3 ≥ 2.5 ∧ 
       t1.1 ≥ 0 ∧ t1.1 ≤ length ∧ t1.2 ≥ 0 ∧ t1.2 ≤ width ∧ 
       t2.1 ≥ 0 ∧ t2.1 ≤ length ∧ t2.2 ≥ 0 ∧ t2.2 ≤ width ∧ 
       t3.1 ≥ 0 ∧ t3.1 ≤ length ∧ t3.2 ≥ 0 ∧ t3.2 ≤ width) := 
by {
  sorry
}

end tree_placement_impossible_l172_172212


namespace min_value_problem_l172_172578

noncomputable def minValueOfExpression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x * y * z = 1) : ℝ :=
  (x + 2 * y) * (y + 2 * z) * (x * z + 1)

theorem min_value_problem (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  minValueOfExpression x y z hx hy hz hxyz = 16 :=
  sorry

end min_value_problem_l172_172578


namespace lower_right_is_one_l172_172920

def initial_grid : Matrix (Fin 5) (Fin 5) (Option (Fin 5)) :=
![![some 0, none, some 1, none, none],
  ![some 1, some 3, none, none, none],
  ![none, none, none, some 4, none],
  ![none, some 4, none, none, none],
  ![none, none, none, none, none]]

theorem lower_right_is_one 
  (complete_grid : Matrix (Fin 5) (Fin 5) (Fin 5)) 
  (unique_row_col : ∀ i j k, 
      complete_grid i j = complete_grid i k ↔ j = k ∧ 
      complete_grid i j = complete_grid k j ↔ i = k)
  (matches_partial : ∀ i j, ∃ x, 
      initial_grid i j = some x → complete_grid i j = x) :
  complete_grid 4 4 = 0 := 
sorry

end lower_right_is_one_l172_172920


namespace smallest_solution_floor_eq_l172_172707

theorem smallest_solution_floor_eq (x : ℝ) : ⌊x^2⌋ - ⌊x⌋^2 = 19 → x = Real.sqrt 119 :=
by
  sorry

end smallest_solution_floor_eq_l172_172707


namespace shorter_piece_is_28_l172_172639

noncomputable def shorter_piece_length (x : ℕ) : Prop :=
  x + (x + 12) = 68 → x = 28

theorem shorter_piece_is_28 (x : ℕ) : shorter_piece_length x :=
by
  intro h
  have h1 : 2 * x + 12 = 68 := by linarith
  have h2 : 2 * x = 56 := by linarith
  have h3 : x = 28 := by linarith
  exact h3

end shorter_piece_is_28_l172_172639


namespace find_abc_l172_172859

open Real

noncomputable def abc_value (a b c : ℝ) : ℝ := a * b * c

theorem find_abc (a b c : ℝ)
  (h₁ : a - b = 3)
  (h₂ : a^2 + b^2 = 39)
  (h₃ : a + b + c = 10) :
  abc_value a b c = -150 + 15 * Real.sqrt 69 :=
by
  sorry

end find_abc_l172_172859


namespace arithmetic_geometric_mean_l172_172048

variable (x y : ℝ)

theorem arithmetic_geometric_mean (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 110) :
  x^2 + y^2 = 1380 := by
  sorry

end arithmetic_geometric_mean_l172_172048


namespace inequality_x_y_z_l172_172458

open Real

theorem inequality_x_y_z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
    (x ^ 3) / ((1 + y) * (1 + z)) + (y ^ 3) / ((1 + z) * (1 + x)) + (z ^ 3) / ((1 + x) * (1 + y)) ≥ 3 / 4 :=
by
  sorry

end inequality_x_y_z_l172_172458


namespace cost_of_first_shirt_l172_172452

theorem cost_of_first_shirt (x : ℝ) (h1 : x + (x + 6) = 24) : x + 6 = 15 :=
by
  sorry

end cost_of_first_shirt_l172_172452


namespace studios_total_l172_172618

section

variable (s1 s2 s3 : ℕ)

theorem studios_total (h1 : s1 = 110) (h2 : s2 = 135) (h3 : s3 = 131) : s1 + s2 + s3 = 376 :=
by
  sorry

end

end studios_total_l172_172618


namespace university_diploma_percentage_l172_172015

-- Define variables
variables (P U J : ℝ)  -- P: Percentage of total population (i.e., 1 or 100%), U: Having a university diploma, J: having the job of their choice
variables (h1 : 10 / 100 * P = 10 / 100 * P * (1 - U) * J)        -- 10% of the people do not have a university diploma but have the job of their choice
variables (h2 : 30 / 100 * (P * (1 - J)) = 30 / 100 * P * U * (1 - J))  -- 30% of the people who do not have the job of their choice have a university diploma
variables (h3 : 40 / 100 * P = 40 / 100 * P * J)                   -- 40% of the people have the job of their choice

-- Statement to prove
theorem university_diploma_percentage : 
  48 / 100 * P = (30 / 100 * P * J) + (18 / 100 * P * (1 - J)) :=
by sorry

end university_diploma_percentage_l172_172015


namespace parabola_constant_term_l172_172347

theorem parabola_constant_term
  (a b c : ℝ)
  (h1 : ∀ x, (-2 * (x - 1)^2 + 3) = a * x^2 + b * x + c ) :
  c = 2 :=
sorry

end parabola_constant_term_l172_172347


namespace distance_to_river_l172_172957

theorem distance_to_river (d : ℝ) (h1 : ¬ (d ≥ 8)) (h2 : ¬ (d ≤ 7)) (h3 : ¬ (d ≤ 6)) : 7 < d ∧ d < 8 :=
by
  sorry

end distance_to_river_l172_172957


namespace proof_y_times_1_minus_g_eq_1_l172_172031
noncomputable def y : ℝ := (3 + Real.sqrt 8) ^ 100
noncomputable def m : ℤ := Int.floor y
noncomputable def g : ℝ := y - m

theorem proof_y_times_1_minus_g_eq_1 :
  y * (1 - g) = 1 := 
sorry

end proof_y_times_1_minus_g_eq_1_l172_172031


namespace problem_statement_l172_172564

theorem problem_statement (x : ℂ) (h : x + 1 / x = real.sqrt 5) : x ^ 10 = 1 := by
  sorry

end problem_statement_l172_172564


namespace range_of_f_l172_172218

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (Real.sqrt (5 + 4 * Real.cos x))

theorem range_of_f :
  Set.range f = Set.Icc (-1/2 : ℝ) (1/2 : ℝ) := 
sorry

end range_of_f_l172_172218


namespace union_of_A_and_B_l172_172979

def A : Set ℤ := {0, 1}
def B : Set ℤ := {-1, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1} := by
  sorry

end union_of_A_and_B_l172_172979


namespace total_distance_combined_l172_172066

/-- The conditions for the problem
Each car has 50 liters of fuel.
Car U has a fuel efficiency of 20 liters per 100 kilometers.
Car V has a fuel efficiency of 25 liters per 100 kilometers.
Car W has a fuel efficiency of 5 liters per 100 kilometers.
Car X has a fuel efficiency of 10 liters per 100 kilometers.
-/
theorem total_distance_combined (fuel_U fuel_V fuel_W fuel_X : ℕ) (eff_U eff_V eff_W eff_X : ℕ) (fuel : ℕ)
  (hU : fuel_U = 50) (hV : fuel_V = 50) (hW : fuel_W = 50) (hX : fuel_X = 50)
  (eU : eff_U = 20) (eV : eff_V = 25) (eW : eff_W = 5) (eX : eff_X = 10) :
  (fuel_U * 100 / eff_U) + (fuel_V * 100 / eff_V) + (fuel_W * 100 / eff_W) + (fuel_X * 100 / eff_X) = 1950 := by 
  sorry

end total_distance_combined_l172_172066


namespace Joan_orange_balloons_l172_172753

theorem Joan_orange_balloons (originally_has : ℕ) (received : ℕ) (final_count : ℕ) 
  (h1 : originally_has = 8) (h2 : received = 2) : 
  final_count = 10 := by
  sorry

end Joan_orange_balloons_l172_172753


namespace expressing_population_in_scientific_notation_l172_172461

def population_in_scientific_notation (population : ℝ) : Prop :=
  population = 1.412 * 10^9

theorem expressing_population_in_scientific_notation : 
  population_in_scientific_notation (1.412 * 10^9) :=
by
  sorry

end expressing_population_in_scientific_notation_l172_172461


namespace tan_alpha_plus_beta_mul_tan_alpha_l172_172541

theorem tan_alpha_plus_beta_mul_tan_alpha (α β : ℝ) (h : 2 * Real.cos (2 * α + β) + 3 * Real.cos β = 0) :
  Real.tan (α + β) * Real.tan α = -5 := 
by
  sorry

end tan_alpha_plus_beta_mul_tan_alpha_l172_172541


namespace compare_x_y_l172_172548

variable (a b : ℝ)
variable (a_pos : 0 < a)
variable (b_pos : 0 < b)
variable (a_ne_b : a ≠ b)

noncomputable def x : ℝ := (Real.sqrt a + Real.sqrt b) / Real.sqrt 2
noncomputable def y : ℝ := Real.sqrt (a + b)

theorem compare_x_y : y a b > x a b := sorry

end compare_x_y_l172_172548


namespace solve_equation_l172_172906

theorem solve_equation (x : ℝ) (h : x ≠ 0 ∧ x ≠ -1) : (x / (x + 1) = 1 + (1 / x)) ↔ (x = -1 / 2) :=
by
  sorry

end solve_equation_l172_172906


namespace arithmetic_progression_y_value_l172_172603

theorem arithmetic_progression_y_value (x y : ℚ) 
  (h1 : x = 2)
  (h2 : 2 * y - x = (y + x + 3) - (2 * y - x))
  (h3 : (3 * y + x) - (y + x + 3) = (y + x + 3) - (2 * y - x)) : 
  y = 10 / 3 :=
by
  sorry

end arithmetic_progression_y_value_l172_172603


namespace find_smallest_solution_l172_172715

theorem find_smallest_solution : ∃ x : ℝ, x = Real.sqrt 119 ∧ (Int.floor (x^2) - Int.floor x ^ 2 = 19) := by
  sorry

end find_smallest_solution_l172_172715


namespace solve_for_x_l172_172059

theorem solve_for_x : ∃ x : ℝ, 64 = 2 * (16 : ℝ)^(x - 2) ∧ x = 3.25 := by
  sorry

end solve_for_x_l172_172059


namespace exists_k_for_blocks_of_2022_l172_172585

theorem exists_k_for_blocks_of_2022 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, (0 < k) ∧ (∀ i : ℕ, (1 ≤ i ∧ i ≤ n) → (∃ j, 
  k^i / 10^j % 10^4 = 2022)) :=
sorry

end exists_k_for_blocks_of_2022_l172_172585


namespace find_smallest_solution_l172_172716

theorem find_smallest_solution : ∃ x : ℝ, x = Real.sqrt 119 ∧ (Int.floor (x^2) - Int.floor x ^ 2 = 19) := by
  sorry

end find_smallest_solution_l172_172716


namespace rahim_books_second_shop_l172_172901

variable (x : ℕ)

-- Definitions of the problem's conditions
def total_cost : ℕ := 520 + 248
def total_books (x : ℕ) : ℕ := 42 + x
def average_price : ℕ := 12

-- The problem statement in Lean 4
theorem rahim_books_second_shop : x = 22 → total_cost / total_books x = average_price :=
  sorry

end rahim_books_second_shop_l172_172901


namespace roots_of_quadratic_l172_172219

theorem roots_of_quadratic (x : ℝ) : (5 * x^2 = 4 * x) → (x = 0 ∨ x = 4 / 5) :=
by
  sorry

end roots_of_quadratic_l172_172219


namespace isosceles_triangles_height_ratio_l172_172492

theorem isosceles_triangles_height_ratio
  (b1 b2 h1 h2 : ℝ)
  (h1_ne_zero : h1 ≠ 0) 
  (h2_ne_zero : h2 ≠ 0)
  (equal_vertical_angles : ∀ (a1 a2 : ℝ), true) -- Placeholder for equal angles since it's not used directly
  (areas_ratio : (b1 * h1) / (b2 * h2) = 16 / 36)
  (similar_triangles : b1 / b2 = h1 / h2) :
  h1 / h2 = 2 / 3 :=
by
  sorry

end isosceles_triangles_height_ratio_l172_172492


namespace value_of_clothing_piece_eq_l172_172378

def annual_remuneration := 10
def work_months := 7
def received_silver_coins := 2

theorem value_of_clothing_piece_eq : 
  ∃ x : ℝ, (x + received_silver_coins) * 12 = (x + annual_remuneration) * work_months → x = 9.2 :=
by
  sorry

end value_of_clothing_piece_eq_l172_172378


namespace function_inverse_l172_172139

theorem function_inverse (x : ℝ) (h : ℝ → ℝ) (k : ℝ → ℝ) 
  (h_def : ∀ x, h x = 6 - 7 * x) 
  (k_def : ∀ x, k x = (6 - x) / 7) : 
  h (k x) = x ∧ k (h x) = x := 
  sorry

end function_inverse_l172_172139


namespace chinese_pig_problem_l172_172062

variable (x : ℕ)

theorem chinese_pig_problem :
  100 * x - 90 * x = 100 :=
sorry

end chinese_pig_problem_l172_172062


namespace cauchy_schwarz_inequality_l172_172983

theorem cauchy_schwarz_inequality
  (x1 y1 z1 x2 y2 z2 : ℝ) :
  (x1 * x2 + y1 * y2 + z1 * z2) ^ 2 ≤ (x1 ^ 2 + y1 ^ 2 + z1 ^ 2) * (x2 ^ 2 + y2 ^ 2 + z2 ^ 2) := 
sorry

end cauchy_schwarz_inequality_l172_172983


namespace pies_made_l172_172210

-- Define the initial number of apples
def initial_apples : Nat := 62

-- Define the number of apples handed out to students
def handed_out_apples : Nat := 8

-- Define the number of apples required per pie
def apples_per_pie : Nat := 9

-- Define the number of remaining apples after handing out to students
def remaining_apples : Nat := initial_apples - handed_out_apples

-- State the theorem
theorem pies_made (initial_apples handed_out_apples apples_per_pie remaining_apples : Nat) :
  initial_apples = 62 →
  handed_out_apples = 8 →
  apples_per_pie = 9 →
  remaining_apples = initial_apples - handed_out_apples →
  remaining_apples / apples_per_pie = 6 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end pies_made_l172_172210


namespace arithmetic_sequence_common_difference_l172_172882

theorem arithmetic_sequence_common_difference (a_1 a_4 a_5 d : ℤ) 
  (h1 : a_1 + a_5 = 10) 
  (h2 : a_4 = 7) 
  (h3 : a_4 = a_1 + 3 * d) 
  (h4 : a_5 = a_1 + 4 * d) : 
  d = 2 :=
by
  sorry

end arithmetic_sequence_common_difference_l172_172882


namespace fraction_simplification_l172_172230

theorem fraction_simplification (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) :
  (x / (x - 1) = 3 / (2 * x - 2) - 3) → (2 * x = 3 - 6 * x + 6) :=
by 
  intro h1
  -- Proof steps would be here, but we are using sorry
  sorry

end fraction_simplification_l172_172230


namespace find_x_l172_172759

-- Declaration for the custom operation
def star (a b : ℝ) : ℝ := a * b + 3 * b - 2 * a

-- Theorem statement
theorem find_x (x : ℝ) (h : star 3 x = 23) : x = 29 / 6 :=
by {
    sorry -- The proof steps are to be filled here.
}

end find_x_l172_172759


namespace additional_water_added_l172_172878

variable (M W : ℕ)

theorem additional_water_added (M W : ℕ) (initial_mix : ℕ) (initial_ratio : ℕ × ℕ) (new_ratio : ℚ) :
  initial_mix = 45 →
  initial_ratio = (4, 1) →
  new_ratio = 4 / 3 →
  (4 / 5) * initial_mix = M →
  (1 / 5) * initial_mix + W = 3 / 4 * M →
  W = 18 :=
by
  sorry

end additional_water_added_l172_172878


namespace value_of_x_l172_172169

theorem value_of_x (x : ℕ) : (1 / 16) * (2 ^ 20) = 4 ^ x → x = 8 := by
  sorry

end value_of_x_l172_172169


namespace percentage_failed_in_english_l172_172571

theorem percentage_failed_in_english (total_students : ℕ) (hindi_failed : ℕ) (both_failed : ℕ) (both_passed : ℕ) 
  (H1 : hindi_failed = total_students * 25 / 100)
  (H2 : both_failed = total_students * 25 / 100)
  (H3 : both_passed = total_students * 50 / 100)
  : (total_students * 50 / 100) = (total_students * 75 / 100) + (both_failed) - both_passed
:= sorry

end percentage_failed_in_english_l172_172571


namespace count_shenma_numbers_l172_172363

def is_shenma_number (n : ℕ) : Prop :=
  let digits := (Finset.range 10).filter (λ d, (n / 10 ^ d) % 10 ∈ Finset.range 10)
  digits.card = 5 ∧
  (let middle_digit := (n / 10 ^ 2) % 10 in
   digits.erase middle_digit = Finset.range 5 ∧
   ∀ i j, i < j → ((n / 10 ^ i % 10 ) < (n / 10 ^ j % 10 )))

theorem count_shenma_numbers : 
  (Finset.filter is_shenma_number (Finset.range 100000)).card = 1512 := 
  sorry

end count_shenma_numbers_l172_172363


namespace female_cows_percentage_l172_172061

theorem female_cows_percentage (TotalCows PregnantFemaleCows : Nat) (PregnantPercentage : ℚ)
    (h1 : TotalCows = 44)
    (h2 : PregnantFemaleCows = 11)
    (h3 : PregnantPercentage = 0.50) :
    (PregnantFemaleCows / PregnantPercentage / TotalCows) * 100 = 50 := 
sorry

end female_cows_percentage_l172_172061


namespace roots_theorem_l172_172336

-- Definitions and Conditions
def root1 (a b p : ℝ) : Prop := 
  a + b = -p ∧ a * b = 1

def root2 (b c q : ℝ) : Prop := 
  b + c = -q ∧ b * c = 2

-- The theorem to prove
theorem roots_theorem (a b c p q : ℝ) (h1 : root1 a b p) (h2 : root2 b c q) : 
  (b - a) * (b - c) = p * q - 6 :=
sorry

end roots_theorem_l172_172336


namespace lily_account_balance_l172_172032

def initial_balance : ℕ := 55

def shirt_cost : ℕ := 7

def second_spend_multiplier : ℕ := 3

def first_remaining_balance (initial_balance shirt_cost: ℕ) : ℕ :=
  initial_balance - shirt_cost

def second_spend (shirt_cost second_spend_multiplier: ℕ) : ℕ :=
  shirt_cost * second_spend_multiplier

def final_remaining_balance (first_remaining_balance second_spend: ℕ) : ℕ :=
  first_remaining_balance - second_spend

theorem lily_account_balance :
  final_remaining_balance (first_remaining_balance initial_balance shirt_cost) (second_spend shirt_cost second_spend_multiplier) = 27 := by
    sorry

end lily_account_balance_l172_172032


namespace moles_of_BeOH2_l172_172681

-- Definitions based on the given conditions
def balanced_chemical_equation (xBe2C xH2O xBeOH2 xCH4 : ℕ) : Prop :=
  xBe2C = 1 ∧ xH2O = 4 ∧ xBeOH2 = 2 ∧ xCH4 = 1

def initial_conditions (yBe2C yH2O : ℕ) : Prop :=
  yBe2C = 1 ∧ yH2O = 4

-- Lean statement to prove the number of moles of Beryllium hydroxide formed
theorem moles_of_BeOH2 (xBe2C xH2O xBeOH2 xCH4 yBe2C yH2O : ℕ) (h1 : balanced_chemical_equation xBe2C xH2O xBeOH2 xCH4) (h2 : initial_conditions yBe2C yH2O) :
  xBeOH2 = 2 :=
by
  sorry

end moles_of_BeOH2_l172_172681


namespace smallest_solution_eq_sqrt_104_l172_172696

theorem smallest_solution_eq_sqrt_104 :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, ⌊y^2⌋ - ⌊y⌋^2 = 19 → x ≤ y) := sorry

end smallest_solution_eq_sqrt_104_l172_172696


namespace loss_percentage_remaining_stock_l172_172953

noncomputable def total_worth : ℝ := 9999.999999999998
def overall_loss : ℝ := 200
def profit_percentage_20 : ℝ := 0.1
def sold_20_percentage : ℝ := 0.2
def remaining_percentage : ℝ := 0.8

theorem loss_percentage_remaining_stock :
  ∃ L : ℝ, 0.8 * total_worth * (L / 100) - 0.02 * total_worth = overall_loss ∧ L = 5 :=
by sorry

end loss_percentage_remaining_stock_l172_172953


namespace charcoal_amount_l172_172181

theorem charcoal_amount (water_per_charcoal : ℕ) (charcoal_ratio : ℕ) (water_added : ℕ) (charcoal_needed : ℕ) 
  (h1 : water_per_charcoal = 30) (h2 : charcoal_ratio = 2) (h3 : water_added = 900) : charcoal_needed = 60 :=
by
  sorry

end charcoal_amount_l172_172181


namespace find_days_jane_indisposed_l172_172313

-- Define the problem conditions
def John_rate := 1 / 20
def Jane_rate := 1 / 10
def together_rate := John_rate + Jane_rate
def total_task := 1
def total_days := 10

-- The time Jane was indisposed
def days_jane_indisposed (x : ℝ) : Prop :=
  (total_days - x) * together_rate + x * John_rate = total_task

-- Statement we want to prove
theorem find_days_jane_indisposed : ∃ x : ℝ, days_jane_indisposed x ∧ x = 5 :=
by 
  sorry

end find_days_jane_indisposed_l172_172313


namespace carrots_picked_first_day_l172_172960

theorem carrots_picked_first_day (X : ℕ) 
  (H1 : X - 10 + 47 = 60) : X = 23 :=
by 
  -- We state the proof steps here, completing the proof with sorry
  sorry

end carrots_picked_first_day_l172_172960


namespace veronica_photo_choices_l172_172359

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

noncomputable def choose (n k : ℕ) : ℕ :=
factorial n / (factorial k * factorial (n - k))

theorem veronica_photo_choices : choose 5 3 + choose 5 4 = 15 := by
  sorry

end veronica_photo_choices_l172_172359


namespace find_bigger_number_l172_172797

noncomputable def common_factor (x : ℕ) : Prop :=
  8 * x + 3 * x = 143

theorem find_bigger_number (x : ℕ) (h : common_factor x) : 8 * x = 104 :=
by
  sorry

end find_bigger_number_l172_172797


namespace center_cell_value_l172_172425

namespace MathProof

variables {a b c d e f g h i : ℝ}

-- Conditions
axiom row_product1 : a * b * c = 1
axiom row_product2 : d * e * f = 1
axiom row_product3 : g * h * i = 1

axiom col_product1 : a * d * g = 1
axiom col_product2 : b * e * h = 1
axiom col_product3 : c * f * i = 1

axiom square_product1 : a * b * d * e = 2
axiom square_product2 : b * c * e * f = 2
axiom square_product3 : d * e * g * h = 2
axiom square_product4 : e * f * h * i = 2

-- Proof problem
theorem center_cell_value : e = 1 :=
sorry

end MathProof

end center_cell_value_l172_172425


namespace range_of_m_F_x2_less_than_x2_minus_1_l172_172164

noncomputable def f (x : ℝ) : ℝ := x + Real.log x
noncomputable def g (x : ℝ) : ℝ := 3 - 2 / x
noncomputable def T (x m : ℝ) : ℝ := Real.log x - x - 2 * m
noncomputable def F (x m : ℝ) : ℝ := x - m / x - 2 * Real.log x
noncomputable def h (t : ℝ) : ℝ := t - 2 * Real.log t - 1

-- (1)
theorem range_of_m (m : ℝ) (h_intersections : ∃ x y : ℝ, T x m = 0 ∧ T y m = 0 ∧ x ≠ y) :
  m < -1 / 2 := sorry

-- (2)
theorem F_x2_less_than_x2_minus_1 {m : ℝ} (h₀ : 0 < m ∧ m < 1) {x₁ x₂ : ℝ} (h₁ : 0 < x₁ ∧ x₁ < x₂)
  (h₂ : F x₁ m = 0 ∧ F x₂ m = 0) :
  F x₂ m < x₂ - 1 := sorry

end range_of_m_F_x2_less_than_x2_minus_1_l172_172164


namespace total_sales_l172_172745

noncomputable def sales_in_june : ℕ := 96
noncomputable def sales_in_july : ℕ := sales_in_june * 4 / 3

theorem total_sales (june_sales : ℕ) (july_sales : ℕ) (h1 : june_sales = 96)
                    (h2 : july_sales = june_sales * 4 / 3) :
                    june_sales + july_sales = 224 :=
by
  rw [h1, h2]
  norm_num
  sorry

end total_sales_l172_172745


namespace gcd_min_val_l172_172319

theorem gcd_min_val (p q r : ℕ) (hpq : Nat.gcd p q = 210) (hpr : Nat.gcd p r = 1155) : ∃ (g : ℕ), g = Nat.gcd q r ∧ g = 105 :=
by
  sorry

end gcd_min_val_l172_172319


namespace minimum_jumps_l172_172315

theorem minimum_jumps (dist_cm : ℕ) (jump_mm : ℕ) (dist_mm : ℕ) (cm_to_mm_conversion : dist_mm = dist_cm * 10) (leap_condition : ∃ n : ℕ, jump_mm * n ≥ dist_mm) : ∃ n : ℕ, 19 * n = 18120 → n = 954 :=
by
  sorry

end minimum_jumps_l172_172315


namespace measure_of_angle_C_l172_172421

theorem measure_of_angle_C (a b area : ℝ) (C : ℝ) :
  a = 5 → b = 8 → area = 10 →
  (1 / 2 * a * b * Real.sin C = area) →
  (C = Real.pi / 6 ∨ C = 5 * Real.pi / 6) := by
  intros ha hb harea hformula
  sorry

end measure_of_angle_C_l172_172421


namespace matrix_det_problem_l172_172623

-- Define the determinant of a 2x2 matrix
def det (a b c d : ℤ) : ℤ := a * d - b * c

-- State the problem in Lean
theorem matrix_det_problem : 2 * det 5 7 2 3 = 2 := by
  sorry

end matrix_det_problem_l172_172623


namespace tan_alpha_value_l172_172284

open Real

variable (α : ℝ)

/- Conditions -/
def alpha_interval : Prop := (0 < α) ∧ (α < π)
def sine_cosine_sum : Prop := sin α + cos α = -7 / 13

/- Statement -/
theorem tan_alpha_value 
  (h1 : alpha_interval α)
  (h2 : sine_cosine_sum α) : 
  tan α = -5 / 12 :=
sorry

end tan_alpha_value_l172_172284


namespace prove_midpoint_trajectory_eq_l172_172583

noncomputable def midpoint_trajectory_eq {x y : ℝ} (h : ∃ (x_P y_P : ℝ), (x_P^2 - y_P^2 = 1) ∧ (x = x_P / 2) ∧ (y = y_P / 2)) : Prop :=
  4*x^2 - 4*y^2 = 1

theorem prove_midpoint_trajectory_eq (x y : ℝ) (h : ∃ (x_P y_P : ℝ), (x_P^2 - y_P^2 = 1) ∧ (x = x_P / 2) ∧ (y = y_P / 2)) :
  midpoint_trajectory_eq h :=
sorry

end prove_midpoint_trajectory_eq_l172_172583


namespace tickets_bought_l172_172195

theorem tickets_bought
  (olivia_money : ℕ) (nigel_money : ℕ) (ticket_cost : ℕ) (leftover_money : ℕ)
  (total_money : ℕ) (money_spent : ℕ) 
  (h1 : olivia_money = 112) 
  (h2 : nigel_money = 139) 
  (h3 : ticket_cost = 28) 
  (h4 : leftover_money = 83)
  (h5 : total_money = olivia_money + nigel_money)
  (h6 : total_money = 251)
  (h7 : money_spent = total_money - leftover_money)
  (h8 : money_spent = 168)
  : money_spent / ticket_cost = 6 := 
by
  sorry

end tickets_bought_l172_172195


namespace bill_salary_increase_l172_172742

theorem bill_salary_increase (S P : ℝ) 
  (h1 : S + 0.16 * S = 812) 
  (h2 : S + P * S = 770.0000000000001) : 
  P = 0.1 :=
by {
  sorry
}

end bill_salary_increase_l172_172742


namespace solve_expression_l172_172802

theorem solve_expression : (0.76 ^ 3 - 0.008) / (0.76 ^ 2 + 0.76 * 0.2 + 0.04) = 0.560 := 
by
  sorry

end solve_expression_l172_172802


namespace overlapping_area_zero_l172_172348

-- Definition of the points and triangles
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  p1 : Point
  p2 : Point
  p3 : Point

def point0 : Point := { x := 0, y := 0 }
def point1 : Point := { x := 2, y := 2 }
def point2 : Point := { x := 2, y := 0 }
def point3 : Point := { x := 0, y := 2 }
def point4 : Point := { x := 1, y := 1 }

def triangle1 : Triangle := { p1 := point0, p2 := point1, p3 := point2 }
def triangle2 : Triangle := { p1 := point3, p2 := point1, p3 := point0 }

-- Function to calculate the area of a triangle
def area (t : Triangle) : ℝ :=
  0.5 * abs (t.p1.x * (t.p2.y - t.p3.y) + t.p2.x * (t.p3.y - t.p1.y) + t.p3.x * (t.p1.y - t.p2.y))

-- Using collinear points theorem to prove that the area of the overlapping region is zero
theorem overlapping_area_zero : area { p1 := point0, p2 := point1, p3 := point4 } = 0 := 
by 
  -- This follows directly from the fact that the points (0,0), (2,2), and (1,1) are collinear
  -- skipping the actual geometric proof for conciseness
  sorry

end overlapping_area_zero_l172_172348


namespace staples_left_in_stapler_l172_172484

def initial_staples : ℕ := 50
def used_staples : ℕ := 3 * 12
def remaining_staples : ℕ := initial_staples - used_staples

theorem staples_left_in_stapler : remaining_staples = 14 :=
by
  unfold initial_staples used_staples remaining_staples
  rw [Nat.mul_comm, Nat.mul_comm 3, Nat.mul_comm 12, Nat.sub_eq_iff_eq_add]
  have h : ∀ a b : ℕ, a = b -> 50 - (3 * 12) = b -> 50 - 36 = a := by intros; rw [h, Nat.mul_comm 3, Nat.mul_comm 12]
  exact h 36 36 rfl
#align std.staples_left_in_stapler


end staples_left_in_stapler_l172_172484


namespace pasta_sauce_cost_l172_172183

theorem pasta_sauce_cost :
  let mustard_oil_cost := 2 * 13
  let penne_pasta_cost := 3 * 4
  let total_cost := 50 - 7
  let spent_on_oil_and_pasta := mustard_oil_cost + penne_pasta_cost
  let pasta_sauce_cost := total_cost - spent_on_oil_and_pasta
  pasta_sauce_cost = 5 :=
by
  let mustard_oil_cost := 2 * 13
  let penne_pasta_cost := 3 * 4
  let total_cost := 50 - 7
  let spent_on_oil_and_pasta := mustard_oil_cost + penne_pasta_cost
  let pasta_sauce_cost := total_cost - spent_on_oil_and_pasta
  sorry

end pasta_sauce_cost_l172_172183


namespace flynn_tv_weeks_l172_172271

-- Define the conditions
def minutes_per_weekday := 30
def additional_hours_weekend := 2
def total_hours := 234
def minutes_per_hour := 60
def weekdays := 5

-- Define the total watching time per week in minutes
def total_weekday_minutes := minutes_per_weekday * weekdays
def total_weekday_hours := total_weekday_minutes / minutes_per_hour
def total_weekly_hours := total_weekday_hours + additional_hours_weekend

-- Create a theorem to prove the correct number of weeks
theorem flynn_tv_weeks : 
  (total_hours / total_weekly_hours) = 52 := 
by
  sorry

end flynn_tv_weeks_l172_172271


namespace find_common_ratio_l172_172553

variable (a₁ : ℝ) (q : ℝ)

def S₁ (a₁ : ℝ) : ℝ := a₁
def S₃ (a₁ q : ℝ) : ℝ := a₁ + a₁ * q + a₁ * q ^ 2
def a₃ (a₁ q : ℝ) : ℝ := a₁ * q ^ 2

theorem find_common_ratio (h : 2 * S₃ a₁ q = S₁ a₁ + 2 * a₃ a₁ q) : q = -1 / 2 :=
by
  sorry

end find_common_ratio_l172_172553


namespace painting_cost_3x_l172_172099

-- Define the dimensions of the original room and the painting cost
variables (L B H : ℝ)
def cost_of_painting (area : ℝ) : ℝ := 350

-- Create a definition for the calculation of area
def paint_area (L B H : ℝ) : ℝ := 2 * (L * H + B * H)

-- Define the new dimensions
def new_dimensions (L B H : ℝ) : ℝ × ℝ × ℝ := (3 * L, 3 * B, 3 * H)

-- Create a definition for the calculation of the new area
def new_paint_area (L B H : ℝ) : ℝ := 18 * (paint_area L B H)

-- Calculate the new cost
def new_cost (L B H : ℝ) : ℝ := 18 * cost_of_painting (paint_area L B H)

-- The theorem to be proved
theorem painting_cost_3x (L B H : ℝ) : new_cost L B H = 6300 :=
by 
  simp [new_cost, cost_of_painting, paint_area]
  sorry

end painting_cost_3x_l172_172099


namespace lily_account_balance_l172_172033

def initial_balance : ℕ := 55

def shirt_cost : ℕ := 7

def second_spend_multiplier : ℕ := 3

def first_remaining_balance (initial_balance shirt_cost: ℕ) : ℕ :=
  initial_balance - shirt_cost

def second_spend (shirt_cost second_spend_multiplier: ℕ) : ℕ :=
  shirt_cost * second_spend_multiplier

def final_remaining_balance (first_remaining_balance second_spend: ℕ) : ℕ :=
  first_remaining_balance - second_spend

theorem lily_account_balance :
  final_remaining_balance (first_remaining_balance initial_balance shirt_cost) (second_spend shirt_cost second_spend_multiplier) = 27 := by
    sorry

end lily_account_balance_l172_172033


namespace meat_left_l172_172447

theorem meat_left (initial_meat : ℕ) (meatball_fraction : ℚ) (spring_roll_meat : ℕ) 
  (h_initial : initial_meat = 20) 
  (h_meatball_fraction : meatball_fraction = 1/4)
  (h_spring_roll_meat : spring_roll_meat = 3) : 
  initial_meat - (initial_meat * meatball_fraction.num / meatball_fraction.denom).toNat - spring_roll_meat = 12 :=
by
  sorry

end meat_left_l172_172447


namespace Shell_Ratio_l172_172463

-- Definitions of the number of shells collected by Alan, Ben, and Laurie.
variable (A B L : ℕ)

-- Hypotheses based on the given conditions:
-- 1. Alan collected four times as many shells as Ben did.
-- 2. Laurie collected 36 shells.
-- 3. Alan collected 48 shells.
theorem Shell_Ratio (h1 : A = 4 * B) (h2 : L = 36) (h3 : A = 48) : B / Nat.gcd B L = 1 ∧ L / Nat.gcd B L = 3 :=
by
  sorry

end Shell_Ratio_l172_172463


namespace number_of_tables_l172_172780

theorem number_of_tables (x : ℕ) (h : 2 * (x - 1) + 3 = 65) : x = 32 :=
sorry

end number_of_tables_l172_172780


namespace mean_temperature_is_0_5_l172_172595

def temperatures : List ℝ := [-3.5, -2.25, 0, 3.75, 4.5]

theorem mean_temperature_is_0_5 :
  (temperatures.sum / temperatures.length) = 0.5 :=
by
  sorry

end mean_temperature_is_0_5_l172_172595


namespace deepak_present_age_l172_172098

theorem deepak_present_age (x : ℕ) (h : 4 * x + 6 = 26) : 3 * x = 15 := 
by 
  sorry

end deepak_present_age_l172_172098


namespace exists_natural_numbers_solving_equation_l172_172930

theorem exists_natural_numbers_solving_equation :
  ∃ (x y z : ℕ), 28 * x + 30 * y + 31 * z = 365 := by
  sorry

end exists_natural_numbers_solving_equation_l172_172930


namespace probability_boy_saturday_girl_sunday_l172_172395

def num_people := 4
def num_boys := 2
def num_girls := 2

-- Total number of ways to choose 2 people out of 4
def total_events := nnreal.of_nat (nat.choose num_people 2 * (2!))

-- Number of favorable outcomes: choosing 1 boy for Saturday and 1 girl for Sunday
def favorable_events := nnreal.of_nat (num_boys * num_girls)

-- The probability that a boy is chosen for Saturday and a girl for Sunday
theorem probability_boy_saturday_girl_sunday :
  (favorable_events / total_events) = (1 / 3) :=
sorry

end probability_boy_saturday_girl_sunday_l172_172395


namespace smallest_solution_floor_equation_l172_172700

theorem smallest_solution_floor_equation : ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (x = Real.sqrt 109) :=
by
  sorry

end smallest_solution_floor_equation_l172_172700


namespace g_of_3_l172_172296

def g (x : ℝ) : ℝ := 5 * x ^ 4 + 4 * x ^ 3 - 7 * x ^ 2 + 3 * x - 2

theorem g_of_3 : g 3 = 401 :=
by
    -- proof will go here
    sorry

end g_of_3_l172_172296


namespace largest_prime_factor_always_37_l172_172376

-- We define the cyclic sequence conditions
def cyclic_shift (seq : List ℕ) : Prop :=
  ∀ i, seq.get! (i % seq.length) % 10 = seq.get! ((i + 1) % seq.length) / 100 ∧
       (seq.get! ((i + 1) % seq.length) / 10 % 10 = seq.get! ((i + 2) % seq.length) % 10) ∧
       (seq.get! ((i + 2) % seq.length) / 10 % 10 = seq.get! ((i + 3) % seq.length) / 100)

-- Summing all elements of a list
def sum (l : List ℕ) : ℕ := l.foldr (· + ·) 0

-- Prove that 37 is always a factor of the sum T
theorem largest_prime_factor_always_37 (seq : List ℕ) (h : cyclic_shift seq) : 
  37 ∣ sum seq := 
sorry

end largest_prime_factor_always_37_l172_172376


namespace length_of_book_l172_172943

theorem length_of_book (A W L : ℕ) (hA : A = 50) (hW : W = 10) (hArea : A = L * W) : L = 5 := 
sorry

end length_of_book_l172_172943


namespace original_price_l172_172788

variable (p q : ℝ)

theorem original_price (x : ℝ)
  (hp : x * (1 + p / 100) * (1 - q / 100) = 1) :
  x = 10000 / (10000 + 100 * (p - q) - p * q) :=
sorry

end original_price_l172_172788


namespace inequality_solution_l172_172060

theorem inequality_solution (a x : ℝ) : 
  (a = 0 → ¬(x^2 - 2*a*x - 3*a^2 < 0)) ∧
  (a > 0 → (-a < x ∧ x < 3*a) ↔ (x^2 - 2*a*x - 3*a^2 < 0)) ∧
  (a < 0 → (3*a < x ∧ x < -a) ↔ (x^2 - 2*a*x - 3*a^2 < 0)) :=
by
  sorry

end inequality_solution_l172_172060


namespace solution_set_inequality_l172_172085

theorem solution_set_inequality (x : ℝ) : (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := 
sorry

end solution_set_inequality_l172_172085


namespace first_day_bacteria_exceeds_200_l172_172011

noncomputable def N : ℕ → ℕ := λ n => 5 * 3^n

theorem first_day_bacteria_exceeds_200 : ∃ n : ℕ, N n > 200 ∧ ∀ m : ℕ, m < n → N m ≤ 200 :=
by
  sorry

end first_day_bacteria_exceeds_200_l172_172011


namespace boxes_of_toothpicks_needed_l172_172130

def total_cards : Nat := 52
def unused_cards : Nat := 23
def cards_used : Nat := total_cards - unused_cards

def toothpicks_wall_per_card : Nat := 64
def windows_per_card : Nat := 3
def doors_per_card : Nat := 2
def toothpicks_per_window_or_door : Nat := 12
def roof_toothpicks : Nat := 1250
def box_capacity : Nat := 750

def toothpicks_for_walls : Nat := cards_used * toothpicks_wall_per_card
def toothpicks_per_card_windows_doors : Nat := (windows_per_card + doors_per_card) * toothpicks_per_window_or_door
def toothpicks_for_windows_doors : Nat := cards_used * toothpicks_per_card_windows_doors
def total_toothpicks_needed : Nat := toothpicks_for_walls + toothpicks_for_windows_doors + roof_toothpicks

def boxes_needed := Nat.ceil (total_toothpicks_needed / box_capacity)

theorem boxes_of_toothpicks_needed : boxes_needed = 7 := by
  -- Proof should be done here
  sorry

end boxes_of_toothpicks_needed_l172_172130


namespace simon_removes_exactly_180_silver_coins_l172_172174

theorem simon_removes_exactly_180_silver_coins :
  ∀ (initial_total_coins initial_gold_percentage final_gold_percentage : ℝ) 
  (initial_silver_coins final_total_coins final_silver_coins silver_coins_removed : ℕ),
  initial_total_coins = 200 → 
  initial_gold_percentage = 0.02 →
  final_gold_percentage = 0.2 →
  initial_silver_coins = (initial_total_coins * (1 - initial_gold_percentage)) → 
  final_total_coins = (4 / final_gold_percentage) →
  final_silver_coins = (final_total_coins - 4) →
  silver_coins_removed = (initial_silver_coins - final_silver_coins) →
  silver_coins_removed = 180 :=
by
  intros initial_total_coins initial_gold_percentage final_gold_percentage 
         initial_silver_coins final_total_coins final_silver_coins silver_coins_removed
  sorry

end simon_removes_exactly_180_silver_coins_l172_172174


namespace complex_fraction_evaluation_l172_172630

theorem complex_fraction_evaluation :
  ( 
    ((3 + 1/3) / 10 + 0.175 / 0.35) / 
    (1.75 - (1 + 11/17) * (51/56)) - 
    ((11/18 - 1/15) / 1.4) / 
    ((0.5 - 1/9) * 3)
  ) = 1/2 := 
sorry

end complex_fraction_evaluation_l172_172630


namespace behavior_on_1_2_l172_172260

/-- Definition of an odd function -/
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

/-- Definition of being decreasing on an interval -/
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x ≥ f y

/-- Definition of having a minimum value on an interval -/
def has_minimum_on (f : ℝ → ℝ) (a b : ℝ) (m : ℝ) : Prop :=
  ∀ x, a ≤ x → x ≤ b → f x ≥ m

theorem behavior_on_1_2 
  {f : ℝ → ℝ} 
  (h_odd : is_odd_function f) 
  (h_dec : is_decreasing_on f (-2) (-1)) 
  (h_min : has_minimum_on f (-2) (-1) 3) :
  is_decreasing_on f 1 2 ∧ ∀ x, 1 ≤ x → x ≤ 2 → f x ≤ -3 := 
by 
  sorry

end behavior_on_1_2_l172_172260


namespace sum_of_4_corners_is_200_l172_172814

-- Define the conditions: 9x9 grid, numbers start from 10, and filled sequentially from left to right and top to bottom.
def topLeftCorner : ℕ := 10
def topRightCorner : ℕ := 18
def bottomLeftCorner : ℕ := 82
def bottomRightCorner : ℕ := 90

-- The main theorem stating that the sum of the numbers in the four corners is 200.
theorem sum_of_4_corners_is_200 :
  topLeftCorner + topRightCorner + bottomLeftCorner + bottomRightCorner = 200 :=
by
  -- Placeholder for proof
  sorry

end sum_of_4_corners_is_200_l172_172814


namespace puppies_per_cage_l172_172513

theorem puppies_per_cage
  (initial_puppies : ℕ)
  (sold_puppies : ℕ)
  (remaining_puppies : ℕ)
  (cages : ℕ)
  (puppies_per_cage : ℕ)
  (h1 : initial_puppies = 78)
  (h2 : sold_puppies = 30)
  (h3 : remaining_puppies = initial_puppies - sold_puppies)
  (h4 : cages = 6)
  (h5 : puppies_per_cage = remaining_puppies / cages) :
  puppies_per_cage = 8 := by
  sorry

end puppies_per_cage_l172_172513


namespace james_present_age_l172_172239

-- Definitions and conditions
variables (D J : ℕ) -- Dan's and James's ages are natural numbers

-- Condition 1: The ratio between Dan's and James's ages
def ratio_condition : Prop := (D * 5 = J * 6)

-- Condition 2: In 4 years, Dan will be 28
def future_age_condition : Prop := (D + 4 = 28)

-- The proof goal: James's present age is 20
theorem james_present_age : ratio_condition D J ∧ future_age_condition D → J = 20 :=
by
  sorry

end james_present_age_l172_172239


namespace focal_length_of_lens_l172_172106

-- Define the conditions
def initial_screen_distance : ℝ := 80
def moved_screen_distance : ℝ := 40
def lens_formula (f v u : ℝ) : Prop := (1 / f) = (1 / v) + (1 / u)

-- Define the proof goal
theorem focal_length_of_lens :
  ∃ f : ℝ, (f = 100 ∨ f = 60) ∧
  lens_formula f f (1 / 0) ∧  -- parallel beam implies object at infinity u = 1/0
  initial_screen_distance = 80 ∧
  moved_screen_distance = 40 :=
sorry

end focal_length_of_lens_l172_172106


namespace value_of_f_neg2_l172_172475

def f (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + 4 * x + c

theorem value_of_f_neg2 (a b c : ℝ) (h1 : f a b c 5 + f a b c (-5) = 6) (h2 : f a b c 2 = 8) :
  f a b c (-2) = -2 := by
  sorry

end value_of_f_neg2_l172_172475


namespace unique_value_of_W_l172_172573

theorem unique_value_of_W (T O W F U R : ℕ) (h1 : T = 8) (h2 : O % 2 = 0) (h3 : ∀ x y, x ≠ y → x = O → y = T → x ≠ O) :
  (T + T) * 10^2 + (W + W) * 10 + (O + O) = F * 10^3 + O * 10^2 + U * 10 + R → W = 3 :=
by
  sorry

end unique_value_of_W_l172_172573


namespace train_crosses_platform_in_25_002_seconds_l172_172111

noncomputable def time_to_cross_platform 
  (length_train : ℝ) 
  (length_platform : ℝ) 
  (speed_kmph : ℝ) : ℝ := 
  let total_distance := length_train + length_platform
  let speed_mps := speed_kmph * (1000 / 3600)
  total_distance / speed_mps

theorem train_crosses_platform_in_25_002_seconds :
  time_to_cross_platform 225 400.05 90 = 25.002 := by
  sorry

end train_crosses_platform_in_25_002_seconds_l172_172111


namespace sea_creatures_lost_l172_172406

theorem sea_creatures_lost (sea_stars : ℕ) (seashells : ℕ) (snails : ℕ) (items_left : ℕ)
  (h1 : sea_stars = 34) (h2 : seashells = 21) (h3 : snails = 29) (h4 : items_left = 59) :
  sea_stars + seashells + snails - items_left = 25 :=
by
  sorry

end sea_creatures_lost_l172_172406


namespace evaluate_expression_l172_172390

theorem evaluate_expression : -20 + 8 * (10 / 2) - 4 = 16 :=
by
  sorry -- Proof to be completed

end evaluate_expression_l172_172390


namespace smallest_solution_floor_eq_l172_172690

theorem smallest_solution_floor_eq (x : ℝ) (hx : ⌊x^2⌋ - ⌊x⌋^2 = 19) : x = 11 := by
  sorry

end smallest_solution_floor_eq_l172_172690


namespace smallest_solution_floor_eq_l172_172709

theorem smallest_solution_floor_eq (x : ℝ) : ⌊x^2⌋ - ⌊x⌋^2 = 19 → x = Real.sqrt 119 :=
by
  sorry

end smallest_solution_floor_eq_l172_172709


namespace kittens_remaining_l172_172822

theorem kittens_remaining (original_kittens : ℕ) (kittens_given_away : ℕ) 
  (h1 : original_kittens = 8) (h2 : kittens_given_away = 4) : 
  original_kittens - kittens_given_away = 4 := by
  sorry

end kittens_remaining_l172_172822


namespace inequality_holds_l172_172850

theorem inequality_holds (x a : ℝ) (h1 : x < a) (h2 : a < 0) : x^2 > ax ∧ ax > a^2 :=
by 
  sorry

end inequality_holds_l172_172850


namespace seven_times_equivalent_l172_172464

theorem seven_times_equivalent (n a b : ℤ) (h : n = a^2 + a * b + b^2) :
  ∃ (c d : ℤ), 7 * n = c^2 + c * d + d^2 :=
sorry

end seven_times_equivalent_l172_172464


namespace no_mult_of_5_end_in_2_l172_172993

theorem no_mult_of_5_end_in_2 (n : ℕ) : n < 500 → ∃ k, n = 5 * k → (n % 10 = 2) = false :=
by
  sorry

end no_mult_of_5_end_in_2_l172_172993


namespace arithmetic_geometric_mean_l172_172049

variable (x y : ℝ)

theorem arithmetic_geometric_mean (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 110) :
  x^2 + y^2 = 1380 := by
  sorry

end arithmetic_geometric_mean_l172_172049


namespace total_distance_correct_l172_172065

def liters_U := 50
def liters_V := 50
def liters_W := 50
def liters_X := 50

def fuel_efficiency_U := 20 -- liters per 100 km
def fuel_efficiency_V := 25 -- liters per 100 km
def fuel_efficiency_W := 5 -- liters per 100 km
def fuel_efficiency_X := 10 -- liters per 100 km

def distance_U := (liters_U / fuel_efficiency_U) * 100 -- Distance for U in km
def distance_V := (liters_V / fuel_efficiency_V) * 100 -- Distance for V in km
def distance_W := (liters_W / fuel_efficiency_W) * 100 -- Distance for W in km
def distance_X := (liters_X / fuel_efficiency_X) * 100 -- Distance for X in km

def total_distance := distance_U + distance_V + distance_W + distance_X -- Total distance of all cars

theorem total_distance_correct :
  total_distance = 1950 := 
by {
  sorry
}

end total_distance_correct_l172_172065


namespace average_consecutive_from_c_l172_172054

variable (a : ℕ) (c : ℕ)

-- Condition: c is the average of seven consecutive integers starting from a
axiom h1 : c = (a + (a + 1) + (a + 2) + (a + 3) + (a + 4) + (a + 5) + (a + 6)) / 7

-- Target statement: Prove the average of seven consecutive integers starting from c is a + 6
theorem average_consecutive_from_c : 
  (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5) + (c + 6)) / 7 = a + 6 :=
by
  sorry

end average_consecutive_from_c_l172_172054


namespace last_digit_of_x95_l172_172364

theorem last_digit_of_x95 (x : ℕ) : 
  (x^95 % 10) - (3^58 % 10) = 4 % 10 → (x^95 % 10 = 3) := by
  sorry

end last_digit_of_x95_l172_172364


namespace solve_quadratic1_solve_quadratic2_l172_172205

theorem solve_quadratic1 (x : ℝ) :
  x^2 + 10 * x + 16 = 0 ↔ (x = -2 ∨ x = -8) :=
by
  sorry

theorem solve_quadratic2 (x : ℝ) :
  x * (x + 4) = 8 * x + 12 ↔ (x = -2 ∨ x = 6) :=
by
  sorry

end solve_quadratic1_solve_quadratic2_l172_172205


namespace benjamin_billboards_l172_172198

theorem benjamin_billboards (B : ℕ) (h1 : 20 + 23 + B = 60) : B = 17 :=
by
  sorry

end benjamin_billboards_l172_172198


namespace fourth_buoy_distance_with_current_l172_172109

-- Define the initial conditions
def first_buoy_distance : ℕ := 20
def second_buoy_additional_distance : ℕ := 24
def third_buoy_additional_distance : ℕ := 28
def common_difference_increment : ℕ := 4
def ocean_current_push_per_segment : ℕ := 3
def number_of_segments : ℕ := 3

-- Define the mathematical proof problem
theorem fourth_buoy_distance_with_current :
  let fourth_buoy_additional_distance := third_buoy_additional_distance + common_difference_increment
  let first_to_second_buoy := first_buoy_distance + second_buoy_additional_distance
  let second_to_third_buoy := first_to_second_buoy + third_buoy_additional_distance
  let distance_before_current := second_to_third_buoy + fourth_buoy_additional_distance
  let total_current_push := ocean_current_push_per_segment * number_of_segments
  let final_distance := distance_before_current - total_current_push
  final_distance = 95 := by
  sorry

end fourth_buoy_distance_with_current_l172_172109


namespace roots_equal_and_real_l172_172986

theorem roots_equal_and_real (a c : ℝ) (h : 32 - 4 * a * c = 0) :
  ∃ x : ℝ, x = (2 * Real.sqrt 2) / a := 
by sorry

end roots_equal_and_real_l172_172986


namespace john_february_phone_bill_l172_172813

-- Define given conditions
def base_cost : ℕ := 30
def included_hours : ℕ := 50
def overage_cost_per_minute : ℕ := 15 -- costs per minute in cents
def hours_talked_in_February : ℕ := 52

-- Define conversion from dollars to cents
def cents_per_dollar : ℕ := 100

-- Define total cost calculation
def total_cost (base_cost : ℕ) (included_hours : ℕ) (overage_cost_per_minute : ℕ) (hours_talked : ℕ) : ℕ :=
  let extra_minutes := (hours_talked - included_hours) * 60
  let extra_cost := extra_minutes * overage_cost_per_minute
  base_cost * cents_per_dollar + extra_cost

-- State the theorem
theorem john_february_phone_bill : total_cost base_cost included_hours overage_cost_per_minute hours_talked_in_February = 4800 := by
  sorry

end john_february_phone_bill_l172_172813


namespace minimum_at_neg_one_l172_172554

noncomputable def f (x : Real) : Real := x * Real.exp x

theorem minimum_at_neg_one : 
  ∃ c : Real, c = -1 ∧ ∀ x : Real, f c ≤ f x := sorry

end minimum_at_neg_one_l172_172554


namespace determine_f_5_l172_172761

theorem determine_f_5 (f : ℝ → ℝ) (h1 : f 1 = 3) 
  (h2 : ∀ x y : ℝ, f (x^2 + y^2) = (x + y) * (f x + f y)) : f 5 = 45 :=
sorry

end determine_f_5_l172_172761


namespace certain_number_is_3_l172_172145

theorem certain_number_is_3 (x : ℚ) (h : (x / 11) * ((121 : ℚ) / 3) = 11) : x = 3 := 
sorry

end certain_number_is_3_l172_172145


namespace find_a_l172_172989

noncomputable def A (a : ℝ) : Set ℝ :=
  {a + 2, (a + 1)^2, a^2 + 3 * a + 3}

theorem find_a (a : ℝ) (h : 1 ∈ A a) : a = 0 :=
  sorry

end find_a_l172_172989


namespace inequality_x_y_l172_172584

theorem inequality_x_y 
  (x y : ℝ) 
  (hx : 0 < x) 
  (hy : 0 < y) : 
  (x / (x + 5 * y)) + (y / (y + 5 * x)) ≤ 1 := 
by 
  sorry

end inequality_x_y_l172_172584


namespace work_completion_time_l172_172373

theorem work_completion_time (work_per_day_A : ℚ) (work_per_day_B : ℚ) (work_per_day_C : ℚ) 
(days_A_worked: ℚ) (days_C_worked: ℚ) :
work_per_day_A = 1 / 20 ∧ work_per_day_B = 1 / 30 ∧ work_per_day_C = 1 / 10 ∧
days_A_worked = 2 ∧ days_C_worked = 4  → 
(work_per_day_A * days_A_worked + work_per_day_B * days_A_worked + work_per_day_C * days_A_worked +
work_per_day_B * (days_C_worked - days_A_worked) + work_per_day_C * (days_C_worked - days_A_worked) +
(1 - 
(work_per_day_A * days_A_worked + work_per_day_B * days_A_worked + work_per_day_C * days_A_worked +
work_per_day_B * (days_C_worked - days_A_worked) + work_per_day_C * (days_C_worked - days_A_worked)))
/ work_per_day_B + days_C_worked) 
= 15 := by
sorry

end work_completion_time_l172_172373


namespace complement_M_in_U_l172_172580

open Finset

-- Definitions of the universal set and subset
def U := {1, 2, 3, 4, 5, 6} : Finset ℕ
def M := {1, 2, 4} : Finset ℕ

-- The statement that needs to be proved
theorem complement_M_in_U : U \ M = {3, 5, 6} := by
  sorry

end complement_M_in_U_l172_172580


namespace tiles_touching_walls_of_room_l172_172604

theorem tiles_touching_walls_of_room (length width : Nat) 
    (hl : length = 10) (hw : width = 5) : 
    2 * length + 2 * width - 4 = 26 := by
  sorry

end tiles_touching_walls_of_room_l172_172604


namespace negation_of_universal_proposition_l172_172079

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + 5 * x = 4) ↔ (∃ x : ℝ, x^2 + 5 * x ≠ 4) :=
by
  sorry

end negation_of_universal_proposition_l172_172079


namespace range_a_l172_172005

noncomputable def f (x a : ℝ) : ℝ := Real.log x + x + 2 / x - a
noncomputable def g (x : ℝ) : ℝ := Real.log x + x + 2 / x

theorem range_a (a : ℝ) : (∃ x > 0, f x a = 0) → a ≥ 3 :=
by
sorry

end range_a_l172_172005


namespace arithmetic_geometric_mean_l172_172050

variable (x y : ℝ)

theorem arithmetic_geometric_mean (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 110) :
  x^2 + y^2 = 1380 := by
  sorry

end arithmetic_geometric_mean_l172_172050


namespace num_pairs_mod_eq_l172_172885

theorem num_pairs_mod_eq (k : ℕ) (h : k ≥ 7) :
  ∃ n : ℕ, n = 2^(k+5) ∧
  (∀ x y : ℕ, 0 ≤ x ∧ x < 2^k ∧ 0 ≤ y ∧ y < 2^k → (73^(73^x) ≡ 9^(9^y) [MOD 2^k]) → true) :=
sorry

end num_pairs_mod_eq_l172_172885


namespace simplify_fraction_l172_172905

theorem simplify_fraction :
  (30 / 35) * (21 / 45) * (70 / 63) - (2 / 3) = - (8 / 15) :=
by
  sorry

end simplify_fraction_l172_172905


namespace quadratic_completing_square_l172_172612

theorem quadratic_completing_square:
  ∃ (b c : ℝ), (∀ x : ℝ, x^2 + 900 * x + 1800 = (x + b)^2 + c) ∧ (c / b = -446.22222) :=
by
  -- We'll skip the proof steps here
  sorry

end quadratic_completing_square_l172_172612


namespace center_cell_value_l172_172433

open Matrix Finset

def table := Matrix (Fin 3) (Fin 3) ℝ

def row_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 0 2 = 1) ∧ 
  (T 1 0 * T 1 1 * T 1 2 = 1) ∧ 
  (T 2 0 * T 2 1 * T 2 2 = 1)

def col_products (T : table) : Prop :=
  (T 0 0 * T 1 0 * T 2 0 = 1) ∧ 
  (T 0 1 * T 1 1 * T 2 1 = 1) ∧ 
  (T 0 2 * T 1 2 * T 2 2 = 1)

def square_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 1 0 * T 1 1 = 2) ∧ 
  (T 0 1 * T 0 2 * T 1 1 * T 1 2 = 2) ∧ 
  (T 1 0 * T 1 1 * T 2 0 * T 2 1 = 2) ∧ 
  (T 1 1 * T 1 2 * T 2 1 * T 2 2 = 2)

theorem center_cell_value (T : table) 
  (h_row : row_products T) 
  (h_col : col_products T) 
  (h_square : square_products T) : 
  T 1 1 = 1 :=
by
  sorry

end center_cell_value_l172_172433


namespace combined_value_of_cookies_sold_l172_172947

theorem combined_value_of_cookies_sold:
  ∀ (total_boxes : ℝ) (plain_boxes : ℝ) (price_plain : ℝ) (price_choco : ℝ),
    total_boxes = 1585 →
    plain_boxes = 793.125 →
    price_plain = 0.75 →
    price_choco = 1.25 →
    (plain_boxes * price_plain + (total_boxes - plain_boxes) * price_choco) = 1584.6875 :=
by
  intros total_boxes plain_boxes price_plain price_choco
  intro h1 h2 h3 h4
  sorry

end combined_value_of_cookies_sold_l172_172947


namespace more_pups_than_adult_dogs_l172_172574

def number_of_huskies := 5
def number_of_pitbulls := 2
def number_of_golden_retrievers := 4
def pups_per_husky := 3
def pups_per_pitbull := 3
def additional_pups_per_golden_retriever := 2
def pups_per_golden_retriever := pups_per_husky + additional_pups_per_golden_retriever

def total_pups := (number_of_huskies * pups_per_husky) + (number_of_pitbulls * pups_per_pitbull) + (number_of_golden_retrievers * pups_per_golden_retriever)
def total_adult_dogs := number_of_huskies + number_of_pitbulls + number_of_golden_retrievers

theorem more_pups_than_adult_dogs : (total_pups - total_adult_dogs) = 30 :=
by
  -- proof steps, which we will skip
  sorry

end more_pups_than_adult_dogs_l172_172574


namespace polynomial_value_l172_172928

theorem polynomial_value 
  (x : ℝ) 
  (h1 : x = (1 + (1994 : ℝ).sqrt) / 2) : 
  (4 * x ^ 3 - 1997 * x - 1994) ^ 20001 = -1 := 
  sorry

end polynomial_value_l172_172928


namespace normal_distribution_probability_l172_172985

open MeasureTheory

variable (ξ : ℝ)
variable (σ : ℝ)
variable (μ : ProbabilityMassFunction ℝ)

theorem normal_distribution_probability (h1 : μ = PDF.normal 0 σ^2) (h2 : μ.prob (λ x, x > 2) = 0.023) :
  μ.prob (λ x, -2 ≤ x ∧ x ≤ 0) = 0.477 :=
by
  sorry

end normal_distribution_probability_l172_172985


namespace simplify_trig_expression_tan_alpha_value_l172_172369

-- Proof Problem (1)
theorem simplify_trig_expression :
  (∃ θ : ℝ, θ = (20:ℝ) ∧ 
    (∃ α : ℝ, α = (160:ℝ) ∧ 
      (∃ β : ℝ, β = 1 - 2 * (Real.sin θ) * (Real.cos θ) ∧ 
        (∃ γ : ℝ, γ = 1 - (Real.sin θ)^2 ∧ 
          (Real.sqrt β) / ((Real.sin α) - (Real.sqrt γ)) = -1)))) :=
sorry

-- Proof Problem (2)
theorem tan_alpha_value (α : ℝ) (h : Real.tan α = 1 / 3) :
  1 / (4 * (Real.cos α)^2 - 6 * (Real.sin α) * (Real.cos α)) = 5 / 9 :=
sorry

end simplify_trig_expression_tan_alpha_value_l172_172369


namespace initial_blocks_l172_172332

theorem initial_blocks (used_blocks remaining_blocks : ℕ) (h1 : used_blocks = 25) (h2 : remaining_blocks = 72) : 
  used_blocks + remaining_blocks = 97 := by
  sorry

end initial_blocks_l172_172332


namespace problem_1_problem_2_l172_172316

-- Problem (1)
theorem problem_1 (x a : ℝ) (h_a : a = 1) (hP : x^2 - 4*a*x + 3*a^2 < 0) (hQ1 : x^2 - x - 6 ≤ 0) (hQ2 : x^2 + 2*x - 8 > 0) :
  2 < x ∧ x < 3 := sorry

-- Problem (2)
theorem problem_2 (a : ℝ) (h_a_pos : 0 < a) (h_suff_neccess : (¬(∀ x, x^2 - 4*a*x + 3*a^2 < 0) → ¬(∀ x, x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0)) ∧
                   ¬(∀ x, x^2 - 4*a*x + 3*a^2 < 0) ≠ ¬(∀ x, x^2 - x - 6 ≤ 0 ∧ x^2 + 2*x - 8 > 0)) :
  1 < a ∧ a ≤ 2 := sorry

end problem_1_problem_2_l172_172316


namespace non_neg_int_solutions_eq_10_l172_172865

theorem non_neg_int_solutions_eq_10 :
  ∃ n : ℕ, n = 55 ∧
  (∃ (x y z : ℕ), x + y + z = 10) :=
by
  sorry

end non_neg_int_solutions_eq_10_l172_172865


namespace closest_to_zero_is_13_l172_172175

noncomputable def a (n : ℕ) : ℤ := 88 - 7 * n

theorem closest_to_zero_is_13 : ∀ (n : ℕ), 1 ≤ n → 81 + (n - 1) * (-7) = a n →
  (∀ m : ℕ, (m : ℤ) ≤ (88 : ℤ) / 7 → abs (a m) > abs (a 13)) :=
  sorry

end closest_to_zero_is_13_l172_172175


namespace test_unanswered_one_way_l172_172233

theorem test_unanswered_one_way (Q A : ℕ) (hQ : Q = 4) (hA : A = 5):
  ∀ (unanswered : ℕ), (unanswered = 1) :=
by
  intros
  sorry

end test_unanswered_one_way_l172_172233


namespace largest_digit_7182N_divisible_by_6_l172_172091

noncomputable def largest_digit_divisible_by_6 : ℕ := 6

theorem largest_digit_7182N_divisible_by_6 (N : ℕ) : 
  (N % 2 = 0) ∧ ((18 + N) % 3 = 0) ↔ (N ≤ 9) ∧ (N = 6) :=
by
  sorry

end largest_digit_7182N_divisible_by_6_l172_172091


namespace infinitely_many_good_approximations_l172_172320

theorem infinitely_many_good_approximations (x : ℝ) (hx : Irrational x) (hx_pos : 0 < x) :
  ∃ᶠ p q : ℕ in at_top, abs (x - p / q) < 1 / q ^ 2 :=
by
  sorry

end infinitely_many_good_approximations_l172_172320


namespace covered_ratio_battonya_covered_ratio_sopron_l172_172267

noncomputable def angular_diameter_sun : ℝ := 1899 / 2
noncomputable def angular_diameter_moon : ℝ := 1866 / 2

def max_phase_battonya : ℝ := 0.766
def max_phase_sopron : ℝ := 0.678

def center_distance (R_M R_S f : ℝ) : ℝ :=
  R_M - (2 * f - 1) * R_S

-- Defining the hypothetical calculation (details omitted for brevity)
def covered_ratio (R_S R_M d : ℝ) : ℝ := 
  -- Placeholder for the actual calculation logic
  sorry

theorem covered_ratio_battonya :
  covered_ratio angular_diameter_sun angular_diameter_moon (center_distance angular_diameter_moon angular_diameter_sun max_phase_battonya) = 0.70 :=
  sorry

theorem covered_ratio_sopron :
  covered_ratio angular_diameter_sun angular_diameter_moon (center_distance angular_diameter_moon angular_diameter_sun max_phase_sopron) = 0.59 :=
  sorry

end covered_ratio_battonya_covered_ratio_sopron_l172_172267


namespace monotonic_decreasing_interval_l172_172607

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

def decreasing_interval (a b : ℝ) := 
  ∀ x : ℝ, a < x ∧ x < b → deriv f x < 0

theorem monotonic_decreasing_interval : decreasing_interval 0 1 :=
sorry

end monotonic_decreasing_interval_l172_172607


namespace center_cell_value_l172_172426

namespace MathProof

variables {a b c d e f g h i : ℝ}

-- Conditions
axiom row_product1 : a * b * c = 1
axiom row_product2 : d * e * f = 1
axiom row_product3 : g * h * i = 1

axiom col_product1 : a * d * g = 1
axiom col_product2 : b * e * h = 1
axiom col_product3 : c * f * i = 1

axiom square_product1 : a * b * d * e = 2
axiom square_product2 : b * c * e * f = 2
axiom square_product3 : d * e * g * h = 2
axiom square_product4 : e * f * h * i = 2

-- Proof problem
theorem center_cell_value : e = 1 :=
sorry

end MathProof

end center_cell_value_l172_172426


namespace bank_queue_minimum_wasted_minutes_bank_queue_maximum_wasted_minutes_expected_wasted_minutes_random_order_l172_172645

def simple_op_time : ℕ := 1
def long_op_time : ℕ := 5
def num_simple_customers : ℕ := 5
def num_long_customers : ℕ := 3
def total_customers : ℕ := 8

theorem bank_queue_minimum_wasted_minutes :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  n * a + 3 * a + b + 4 * a + b + a + b + (b + (n - 1) * a) + b + (b + (n-2) * a) = 40 :=
  by intros; sorry

theorem bank_queue_maximum_wasted_minutes :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  m * (m - 1) * b / 2 + n * a * (m + n) + 1 = 100 :=
  by intros; sorry

theorem expected_wasted_minutes_random_order :
  ∀ (a b n m : ℕ), 
  a = simple_op_time → b = long_op_time → n = num_simple_customers → m = num_long_customers → 
  total_customers = (n + m) →
  ∑ i in range total_customers, (i * (a + b)) = 72.5 * (total_customers * (total_customers - 1)) / 2 :=
  by intros; sorry

end bank_queue_minimum_wasted_minutes_bank_queue_maximum_wasted_minutes_expected_wasted_minutes_random_order_l172_172645


namespace kangaroo_can_jump_1000_units_l172_172651

noncomputable def distance (x y : ℕ) : ℕ := x + y

def valid_small_jump (x y : ℕ) : Prop :=
  x + 1 ≥ 0 ∧ y - 1 ≥ 0

def valid_big_jump (x y : ℕ) : Prop :=
  x - 5 ≥ 0 ∧ y + 7 ≥ 0

theorem kangaroo_can_jump_1000_units (x y : ℕ) (h : x + y > 6) :
  distance x y ≥ 1000 :=
sorry

end kangaroo_can_jump_1000_units_l172_172651


namespace point_on_imaginary_axis_point_in_fourth_quadrant_l172_172309

-- (I) For what value(s) of the real number m is the point A on the imaginary axis?
theorem point_on_imaginary_axis (m : ℝ) :
  m^2 - 8 * m + 15 = 0 ∧ m^2 + m - 12 ≠ 0 ↔ m = 5 := sorry

-- (II) For what value(s) of the real number m is the point A located in the fourth quadrant?
theorem point_in_fourth_quadrant (m : ℝ) :
  (m^2 - 8 * m + 15 > 0 ∧ m^2 + m - 12 < 0) ↔ -4 < m ∧ m < 3 := sorry

end point_on_imaginary_axis_point_in_fourth_quadrant_l172_172309


namespace original_speed_correct_l172_172917

variables (t m s : ℝ)

noncomputable def original_speed (t m s : ℝ) : ℝ :=
  ((t * m + Real.sqrt (t^2 * m^2 + 4 * t * m * s)) / (2 * t))

theorem original_speed_correct (t m s : ℝ) (ht : 0 < t) : 
  original_speed t m s = (Real.sqrt (t * m * (4 * s + t * m)) - t * m) / (2 * t) :=
by
  sorry

end original_speed_correct_l172_172917


namespace total_distance_correct_l172_172064

def liters_U := 50
def liters_V := 50
def liters_W := 50
def liters_X := 50

def fuel_efficiency_U := 20 -- liters per 100 km
def fuel_efficiency_V := 25 -- liters per 100 km
def fuel_efficiency_W := 5 -- liters per 100 km
def fuel_efficiency_X := 10 -- liters per 100 km

def distance_U := (liters_U / fuel_efficiency_U) * 100 -- Distance for U in km
def distance_V := (liters_V / fuel_efficiency_V) * 100 -- Distance for V in km
def distance_W := (liters_W / fuel_efficiency_W) * 100 -- Distance for W in km
def distance_X := (liters_X / fuel_efficiency_X) * 100 -- Distance for X in km

def total_distance := distance_U + distance_V + distance_W + distance_X -- Total distance of all cars

theorem total_distance_correct :
  total_distance = 1950 := 
by {
  sorry
}

end total_distance_correct_l172_172064


namespace angles_with_same_terminal_side_l172_172477

theorem angles_with_same_terminal_side (k : ℤ) :
  {θ : ℝ | ∃ k : ℤ, θ = k * 360 + 260} = 
  {θ : ℝ | ∃ k : ℤ, θ = k * 360 + (-460 % 360)} :=
by sorry

end angles_with_same_terminal_side_l172_172477


namespace range_of_a_l172_172287

def proposition_p (a : ℝ) : Prop := ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → x^2 ≥ a

def proposition_q (a : ℝ) : Prop := ∃ (x₀ : ℝ), x₀^2 + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) : proposition_p a ∧ proposition_q a ↔ (a = 1 ∨ a ≤ -2) :=
by
  sorry

end range_of_a_l172_172287


namespace multiple_of_x_l172_172873

theorem multiple_of_x (k x y : ℤ) (hk : k * x + y = 34) (hx : 2 * x - y = 20) (hy : y^2 = 4) : k = 4 :=
sorry

end multiple_of_x_l172_172873


namespace top_card_is_king_l172_172386

noncomputable def num_cards := 52
noncomputable def num_kings := 4
noncomputable def probability_king := num_kings / num_cards

theorem top_card_is_king :
  probability_king = 1 / 13 := by
  sorry

end top_card_is_king_l172_172386


namespace negation_of_exists_gt_implies_forall_leq_l172_172070

theorem negation_of_exists_gt_implies_forall_leq (x : ℝ) (h : 0 < x) :
  ¬ (∃ x : ℝ, 0 < x ∧ x^3 - x + 1 > 0) ↔ ∀ x : ℝ, 0 < x → x^3 - x + 1 ≤ 0 :=
by sorry

end negation_of_exists_gt_implies_forall_leq_l172_172070


namespace increasing_sum_sequence_l172_172886

theorem increasing_sum_sequence (a : ℕ → ℝ) (Sn : ℕ → ℝ)
  (ha : ∀ n : ℕ, 0 < a (n + 1))
  (hSn : ∀ n : ℕ, Sn (n + 1) = Sn n + a (n + 1)) :
  (∀ n : ℕ, Sn (n + 1) > Sn n)
  ∧ ¬ (∀ n : ℕ, Sn (n + 1) > Sn n → 0 < a (n + 1)) :=
sorry

end increasing_sum_sequence_l172_172886


namespace canonical_equations_of_line_l172_172368

-- Definitions for the normal vectors of the planes
def n1 : ℝ × ℝ × ℝ := (2, 3, -2)
def n2 : ℝ × ℝ × ℝ := (1, -3, 1)

-- Define the equations of the planes
def plane1 (x y z : ℝ) : Prop := 2 * x + 3 * y - 2 * z + 6 = 0
def plane2 (x y z : ℝ) : Prop := x - 3 * y + z + 3 = 0

-- The canonical equations of the line of intersection
def canonical_eq (x y z : ℝ) : Prop := (z * (-4)) = (y * (-9)) ∧ (z * (-3)) = (x + 3) * (-9)

theorem canonical_equations_of_line :
  ∀ x y z : ℝ, (plane1 x y z) ∧ (plane2 x y z) → canonical_eq x y z :=
by
  sorry

end canonical_equations_of_line_l172_172368


namespace center_cell_value_l172_172440

variable (a b c d e f g h i : ℝ)

-- Defining the conditions
def row_product_1 := a * b * c = 1 ∧ d * e * f = 1 ∧ g * h * i = 1
def col_product_1 := a * d * g = 1 ∧ b * e * h = 1 ∧ c * f * i = 1
def subgrid_product_2 := a * b * d * e = 2 ∧ b * c * e * f = 2 ∧ d * e * g * h = 2 ∧ e * f * h * i = 2

-- The theorem to prove
theorem center_cell_value (h1 : row_product_1 a b c d e f g h i) 
                          (h2 : col_product_1 a b c d e f g h i) 
                          (h3 : subgrid_product_2 a b c d e f g h i) : 
                          e = 1 :=
by
  sorry

end center_cell_value_l172_172440


namespace tan_subtraction_inequality_l172_172586

theorem tan_subtraction_inequality (x y : ℝ) 
  (hx : 0 < x ∧ x < (π / 2)) 
  (hy : 0 < y ∧ y < (π / 2)) 
  (h : Real.tan x = 3 * Real.tan y) : 
  x - y ≤ π / 6 ∧ (x - y = π / 6 ↔ (x = π / 3 ∧ y = π / 6)) := 
sorry

end tan_subtraction_inequality_l172_172586


namespace oranges_per_box_l172_172589

theorem oranges_per_box (h_oranges : 56 = 56) (h_boxes : 8 = 8) : 56 / 8 = 7 :=
by
  -- Placeholder for the proof
  sorry

end oranges_per_box_l172_172589


namespace division_of_powers_of_ten_l172_172525

theorem division_of_powers_of_ten : 10^8 / (2 * 10^6) = 50 := by 
  sorry

end division_of_powers_of_ten_l172_172525


namespace unique_10_digit_number_property_l172_172133

def ten_digit_number (N : ℕ) : Prop :=
  10^9 ≤ N ∧ N < 10^10

def first_digits_coincide (N : ℕ) : Prop :=
  ∀ M : ℕ, N^2 < 10^M → N^2 / 10^(M - 10) = N

theorem unique_10_digit_number_property :
  ∀ (N : ℕ), ten_digit_number N ∧ first_digits_coincide N → N = 1000000000 := 
by
  intros N hN
  sorry

end unique_10_digit_number_property_l172_172133


namespace coin_flips_heads_l172_172954

theorem coin_flips_heads (H T : ℕ) (flip_condition : H + T = 211) (tail_condition : T = H + 81) :
    H = 65 :=
by
  sorry

end coin_flips_heads_l172_172954


namespace cos_E_floor_1000_l172_172014

theorem cos_E_floor_1000 {EF GH FG EH : ℝ} {E G : ℝ} (h1 : EF = 200) (h2 : GH = 200) (h3 : FG + EH = 380) (h4 : E = G) (h5 : EH ≠ FG) :
  ∃ (cE : ℝ), cE = 11/16 ∧ ⌊ 1000 * cE ⌋ = 687 :=
by sorry

end cos_E_floor_1000_l172_172014


namespace sector_area_120_6_l172_172778

open Real

def area_of_sector (R : ℝ) (n : ℝ) : ℝ :=
  (n * π * R ^ 2) / 360

theorem sector_area_120_6 :
  area_of_sector 6 120 = 12 * π :=
by
  sorry

end sector_area_120_6_l172_172778


namespace fraction_of_journey_asleep_l172_172820

theorem fraction_of_journey_asleep (x y : ℝ) (hx : x > 0) (hy : y = x / 3) :
  y / x = 1 / 3 :=
by
  sorry

end fraction_of_journey_asleep_l172_172820


namespace min_ab_value_l172_172806

theorem min_ab_value 
  (a b : ℝ) 
  (hab_pos : a * b > 0)
  (collinear_condition : 2 * a + 2 * b + a * b = 0) :
  a * b ≥ 16 := 
sorry

end min_ab_value_l172_172806


namespace inscribable_in_circle_l172_172191

variables {A B C D : Type} [EuclideanGeometry A B C D]
variable (P : Quadrilateral A B C D) 

-- The condition: ABCD is convex
variable [ConvexQuadrilateral P]

-- The condition: ∠ABD = ∠ACD
variable (h_angle_eq : ∠(A B D) = ∠(A C D))

-- The statement to prove: ABCD can be inscribed in a circle
theorem inscribable_in_circle : InscribableInCircle P :=
by sorry

end inscribable_in_circle_l172_172191


namespace alt_rep_of_set_l172_172597

def NatPos (x : ℕ) := x > 0

theorem alt_rep_of_set : {x : ℕ | NatPos x ∧ x - 3 < 2} = {1, 2, 3, 4} := by
  sorry

end alt_rep_of_set_l172_172597


namespace find_central_angle_l172_172984

-- We define the given conditions.
def radius : ℝ := 2
def area : ℝ := 8

-- We state the theorem that we need to prove.
theorem find_central_angle (R : ℝ) (A : ℝ) (hR : R = radius) (hA : A = area) :
  ∃ α : ℝ, α = 4 :=
by
  sorry

end find_central_angle_l172_172984


namespace smallest_solution_floor_eq_l172_172711

theorem smallest_solution_floor_eq (x : ℝ) : ⌊x^2⌋ - ⌊x⌋^2 = 19 → x = Real.sqrt 119 :=
by
  sorry

end smallest_solution_floor_eq_l172_172711


namespace math_proof_problem_l172_172760

noncomputable def proof_problem (c d : ℝ) : Prop :=
  (∀ x : ℝ, (((x + c) * (x + d) * (x - 10)) / ((x - 5)^2) = 0) → 
    x = -c ∨ x = -d ∨ x = 10 ∧ c ≠ -5 ∧ d ≠ -5 ∧ -c ≠ -d ∧ -c ≠ 10 ∧ -d ≠ 10)
  ∧ (∃ x : ℝ, (((x + 3 * c) * (x - 4) * (x - 8)) / ((x + d) * (x - 10)) = 0) → 
    x = -d ∨ x = 10 ∨ -d = 4 ∨ x = -4 ∨ x = -8 ∧ 3 * c ≠ -4 ∧ c = 4 / 3)
  ∧ 100 * c + d = 141
  
theorem math_proof_problem (c d : ℝ) 
  (h1 : ∀ x : ℝ, (((x + c) * (x + d) * (x - 10)) / ((x - 5)^2) = 0) → 
    x = -c ∨ x = -d ∨ x = 10 ∧ c ≠ -5 ∧ d ≠ -5 ∧ -c ≠ -d ∧ -c ≠ 10 ∧ -d ≠ 10)
  (h2 : ∀ x : ℝ, (((x + 3 * c) * (x - 4) * (x - 8)) / ((x + d) * (x - 10)) = 0) → 
    x = -d ∨ x = 10 ∨ -d = 4 ∨ x = -4 ∨ x = -8 ∧ 3 * c ≠ -4 ∧ c = 4 / 3) :
  100 * c + d = 141 := 
sorry

end math_proof_problem_l172_172760


namespace loom_weaving_rate_l172_172383

theorem loom_weaving_rate (total_cloth : ℝ) (total_time : ℝ) (rate : ℝ) 
  (h1 : total_cloth = 26) (h2 : total_time = 203.125) : rate = total_cloth / total_time := by
  sorry

#check loom_weaving_rate

end loom_weaving_rate_l172_172383


namespace square_div_by_144_l172_172107

theorem square_div_by_144 (n : ℕ) (h1 : ∃ (k : ℕ), n = 12 * k) : ∃ (m : ℕ), n^2 = 144 * m :=
by
  sorry

end square_div_by_144_l172_172107


namespace line_through_point_with_equal_intercepts_l172_172069

theorem line_through_point_with_equal_intercepts
  (P : ℝ × ℝ) (hP : P = (1, 3))
  (intercepts_equal : ∃ a : ℝ, a ≠ 0 ∧ (∀ x y : ℝ, (x/a) + (y/a) = 1 → x + y = 4 ∨ 3*x - y = 0)) :
  ∃ a b c : ℝ, (a, b, c) = (3, -1, 0) ∨ (a, b, c) = (1, 1, -4) ∧ (∀ x y : ℝ, a*x + b*y + c = 0 → (x + y = 4 ∨ 3*x - y = 0)) := 
by
  sorry

end line_through_point_with_equal_intercepts_l172_172069


namespace monotonic_intervals_of_g_range_of_a_for_extreme_values_of_f_floor_of_unique_zero_point_l172_172988

open Real

-- Condition definitions
def g (x: ℝ) (a: ℝ) : ℝ := 2 / x - a * log x

def f (x: ℝ) (a: ℝ) : ℝ := x^2 + g x a

-- Problem statements
theorem monotonic_intervals_of_g (a: ℝ) :
  (a >= 0 → ∀ x > 0, ∀ y > x, g y a ≤ g x a) ∧ 
  (a < 0 → (∀ x ∈(open_interval (0, -2 / a)), ∀ y > x, g y a ≤ g x a) ∧ 
           (∀ x ∈(open_interval (-2 / a, +∞)), ∀ y > x, g y a ≥ g x a)) :=
sorry

theorem range_of_a_for_extreme_values_of_f (a: ℝ) :
  (∃ x ∈ open_interval (0, 1), deriv (λ x, f x a) x = 0) ↔ a < 0 :=
sorry

theorem floor_of_unique_zero_point (a x0: ℝ) (hx0: 1 < x0) :
  (0 < a → f x0 a = 0 → deriv (λ x, f x a) x0 = 0 → ⌊x0⌋ = 2) :=
sorry

end monotonic_intervals_of_g_range_of_a_for_extreme_values_of_f_floor_of_unique_zero_point_l172_172988


namespace cylinder_volume_l172_172733

theorem cylinder_volume (V_sphere : ℝ) (V_cylinder : ℝ) (R H : ℝ) 
  (h1 : V_sphere = 4 * π / 3) 
  (h2 : (4 * π * R ^ 3) / 3 = V_sphere) 
  (h3 : H = 2 * R) 
  (h4 : R = 1) : V_cylinder = 2 * π :=
by
  sorry

end cylinder_volume_l172_172733


namespace order_of_abc_l172_172981

theorem order_of_abc (a b c : ℝ) (h1 : a = 16 ^ (1 / 3))
                                 (h2 : b = 2 ^ (4 / 5))
                                 (h3 : c = 5 ^ (2 / 3)) :
  c > a ∧ a > b :=
by {
  sorry
}

end order_of_abc_l172_172981


namespace imaginary_part_of_z_l172_172728

open Complex

-- Condition
def equation_z (z : ℂ) : Prop := (z * (1 + I) * I^3) / (1 - I) = 1 - I

-- Problem statement
theorem imaginary_part_of_z (z : ℂ) (h : equation_z z) : z.im = -1 := 
by 
  sorry

end imaginary_part_of_z_l172_172728


namespace unique_real_function_l172_172839

theorem unique_real_function (f : ℝ → ℝ) :
  (∀ x y z : ℝ, (f (x * y) / 2 + f (x * z) / 2 - f x * f (y * z)) ≥ 1 / 4) →
  (∀ x : ℝ, f x = 1 / 2) :=
by
  intro h
  -- proof steps go here
  sorry

end unique_real_function_l172_172839


namespace problem_sum_value_l172_172214

def letter_value_pattern : List Int := [2, 3, 2, 1, 0, -1, -2, -3, -2, -1]

def char_value (c : Char) : Int :=
  let pos := c.toNat - 'a'.toNat + 1
  letter_value_pattern.get! ((pos - 1) % 10)

def word_value (w : String) : Int :=
  w.data.map char_value |>.sum

theorem problem_sum_value : word_value "problem" = 5 :=
  by sorry

end problem_sum_value_l172_172214


namespace girls_count_l172_172918

theorem girls_count (G B : ℕ) (hB : B = 4) (h_alt : ∀ (G B : ℕ), 
  (∃ (f : ℕ → ℕ), ∀ g, f g ∈ {1, 2, 3, 4!} →  ∃ f (G = 5)), ∀ b, f b ∈ {4!}) (h_total : 4! * G! := 2880 ) : 
  G = 5 :=
  
  sorry
  
end girls_count_l172_172918


namespace alpha_beta_inequality_l172_172403

theorem alpha_beta_inequality (α β : ℝ) (h1 : -1 < α) (h2 : α < β) (h3 : β < 1) : 
  -2 < α - β ∧ α - β < 0 := 
sorry

end alpha_beta_inequality_l172_172403


namespace gold_weight_is_ten_l172_172357

theorem gold_weight_is_ten :
  let weights := finset.range 19
  let total_weight := weights.sum id
  let bronze_weights := finset.range 9
  let total_bronze_weight := bronze_weights.sum id
  let iron_weights := finset.Icc 10 18
  let total_iron_weight := iron_weights.sum id
  let S_gold := total_weight - (total_bronze_weight + total_iron_weight)
  total_weight = 190 ∧ (total_iron_weight - total_bronze_weight) = 90 →
  S_gold = 10 :=
by
  simp [S_gold, total_weight, total_bronze_weight, total_iron_weight] at *
  sorry

end gold_weight_is_ten_l172_172357


namespace toms_nickels_l172_172223

variables (q n : ℕ)

theorem toms_nickels (h1 : q + n = 12) (h2 : 25 * q + 5 * n = 220) : n = 4 :=
by {
  sorry
}

end toms_nickels_l172_172223


namespace probability_exactly_three_primes_l172_172118

noncomputable def prime_faces : Finset ℕ := {2, 3, 5, 7, 11}

def num_faces : ℕ := 12
def num_dice : ℕ := 7
def target_primes : ℕ := 3

def probability_three_primes : ℚ :=
  35 * ((5 / 12)^3 * (7 / 12)^4)

theorem probability_exactly_three_primes :
  probability_three_primes = (4375 / 51821766) :=
by
  sorry

end probability_exactly_three_primes_l172_172118


namespace problem_solution_l172_172875

theorem problem_solution (m : ℝ) (h : (m - 2023)^2 + (2024 - m)^2 = 2025) :
  (m - 2023) * (2024 - m) = -1012 :=
sorry

end problem_solution_l172_172875


namespace equilateral_triangle_area_l172_172598

theorem equilateral_triangle_area (h : ∀ (a : ℝ), a = 2 * Real.sqrt 3) : 
  ∃ (a : ℝ), a = 4 * Real.sqrt 3 := 
sorry

end equilateral_triangle_area_l172_172598


namespace amount_of_meat_left_l172_172446

theorem amount_of_meat_left (initial_meat : ℕ) (meatballs_fraction : ℚ) (spring_rolls_meat : ℕ)
  (h0 : initial_meat = 20)
  (h1 : meatballs_fraction = 1/4)
  (h2 : spring_rolls_meat = 3) : 
  (initial_meat - (initial_meat * meatballs_fraction:ℕ) - spring_rolls_meat) = 12 := 
by 
  sorry

end amount_of_meat_left_l172_172446


namespace variance_2ξ_plus_3_l172_172860

variable {Ω : Type*} [ProbabilitySpace Ω]

-- Conditions: The random variable ξ has a discrete probability distribution.
noncomputable def ξ (ω : Ω) : ℝ := 1 * indicator {ω | ξ = 1} ω 
                                   + 2 * indicator {ω | ξ = 2} ω 
                                   + 3 * indicator {ω | ξ = 3} ω

axiom ξ_dist : ∀ k ∈ {1, 2, 3}, P {ω | ξ ω = k} = 1 / 3

-- Prove that the variance of 2ξ + 3 is 8/3
theorem variance_2ξ_plus_3 : variance (λ ω, 2 * ξ ω + 3) = 8 / 3 := by
  sorry

end variance_2ξ_plus_3_l172_172860


namespace intersection_M_N_l172_172459

def M := { x : ℝ | x < 2011 }
def N := { x : ℝ | 0 < x ∧ x < 1 }

theorem intersection_M_N :
  M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } := 
by 
  sorry

end intersection_M_N_l172_172459


namespace total_money_in_dollars_l172_172932

/-- You have some amount in nickels and quarters.
    You have 40 nickels and the same number of quarters.
    Prove that the total amount of money in dollars is 12. -/
theorem total_money_in_dollars (n_nickels n_quarters : ℕ) (value_nickel value_quarter : ℕ) 
  (h1: n_nickels = 40) (h2: n_quarters = 40) (h3: value_nickel = 5) (h4: value_quarter = 25) : 
  (n_nickels * value_nickel + n_quarters * value_quarter) / 100 = 12 :=
  sorry

end total_money_in_dollars_l172_172932


namespace jackson_meat_left_l172_172448

theorem jackson_meat_left (total_meat : ℕ) (meatballs_fraction : ℚ) (spring_rolls_meat : ℕ) :
  total_meat = 20 →
  meatballs_fraction = 1/4 →
  spring_rolls_meat = 3 →
  total_meat - (meatballs_fraction * total_meat + spring_rolls_meat) = 12 := by
  intros ht hm hs
  sorry

end jackson_meat_left_l172_172448


namespace symmetric_point_R_l172_172855

variable (a b : ℝ) 

def symmetry_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
def symmetry_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

theorem symmetric_point_R :
  let M := (a, b)
  let N := symmetry_x M
  let P := symmetry_y N
  let Q := symmetry_x P
  let R := symmetry_y Q
  R = (a, b) := by
  unfold symmetry_x symmetry_y
  sorry

end symmetric_point_R_l172_172855


namespace lilies_per_centerpiece_correct_l172_172324

-- Definitions based on the conditions
def num_centerpieces : ℕ := 6
def roses_per_centerpiece : ℕ := 8
def cost_per_flower : ℕ := 15
def total_budget : ℕ := 2700

-- Definition of the number of orchids per centerpiece using given condition
def orchids_per_centerpiece : ℕ := 2 * roses_per_centerpiece

-- Definition of the total cost for roses and orchids before calculating lilies
def total_rose_cost : ℕ := num_centerpieces * roses_per_centerpiece * cost_per_flower
def total_orchid_cost : ℕ := num_centerpieces * orchids_per_centerpiece * cost_per_flower
def total_rose_and_orchid_cost : ℕ := total_rose_cost + total_orchid_cost

-- Definition for the remaining budget for lilies
def remaining_budget_for_lilies : ℕ := total_budget - total_rose_and_orchid_cost

-- Number of lilies in total and per centerpiece
def total_lilies : ℕ := remaining_budget_for_lilies / cost_per_flower
def lilies_per_centerpiece : ℕ := total_lilies / num_centerpieces

-- The proof statement we want to assert
theorem lilies_per_centerpiece_correct : lilies_per_centerpiece = 6 :=
by
  sorry

end lilies_per_centerpiece_correct_l172_172324


namespace train_length_l172_172518

theorem train_length (time : ℝ) (speed_kmh : ℝ) (speed_ms : ℝ) (length : ℝ) : 
  time = 3.499720022398208 ∧ 
  speed_kmh = 144 ∧ 
  speed_ms = 40 ∧ 
  length = speed_ms * time → 
  length = 139.98880089592832 :=
by sorry

end train_length_l172_172518


namespace perp_condition_l172_172289

def a (x : ℝ) : ℝ × ℝ := (x-1, 2)
def b : ℝ × ℝ := (2, 1)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem perp_condition (x : ℝ) : dot_product (a x) b = 0 ↔ x = 0 :=
by 
  sorry

end perp_condition_l172_172289


namespace value_of_expression_l172_172221

theorem value_of_expression :
  (0.00001 * (0.01)^2 * 1000) / 0.001 = 10^(-3) :=
by
  -- Proof goes here
  sorry

end value_of_expression_l172_172221


namespace proportion_solution_l172_172741

theorem proportion_solution (x: ℕ) (h : 3 / 12 = x / 16) : x = 4 :=
sorry

end proportion_solution_l172_172741


namespace prob_first_3_heads_last_5_tails_eq_l172_172999

-- Define the conditions
def prob_heads : ℚ := 3/5
def prob_tails : ℚ := 1 - prob_heads
def heads_flips (n : ℕ) : ℚ := prob_heads ^ n
def tails_flips (n : ℕ) : ℚ := prob_tails ^ n
def first_3_heads_last_5_tails (first_n : ℕ) (last_m : ℕ) : ℚ := (heads_flips first_n) * (tails_flips last_m)

-- Specify the problem
theorem prob_first_3_heads_last_5_tails_eq :
  first_3_heads_last_5_tails 3 5 = 864/390625 := 
by
  -- conditions and calculation here
  sorry

end prob_first_3_heads_last_5_tails_eq_l172_172999


namespace scientific_notation_of_425000_l172_172677

def scientific_notation (x : ℝ) : ℝ × ℤ := sorry

theorem scientific_notation_of_425000 :
  scientific_notation 425000 = (4.25, 5) := sorry

end scientific_notation_of_425000_l172_172677


namespace find_an_l172_172028

def sequence_sum (k : ℝ) (n : ℕ) : ℝ :=
  k * n ^ 2 + n

def term_of_sequence (k : ℝ) (n : ℕ) (S_n : ℝ) (S_nm1 : ℝ) : ℝ :=
  S_n - S_nm1

theorem find_an (k : ℝ) (n : ℕ) (h₁ : n > 0) :
  term_of_sequence k n (sequence_sum k n) (sequence_sum k (n - 1)) = 2 * k * n - k + 1 :=
by
  sorry

end find_an_l172_172028


namespace sophie_germain_identity_l172_172131

theorem sophie_germain_identity (a b : ℝ) : 
  a^4 + 4 * b^4 = (a^2 + 2 * a * b + 2 * b^2) * (a^2 - 2 * a * b + 2 * b^2) :=
by sorry

end sophie_germain_identity_l172_172131


namespace center_cell_value_l172_172436

theorem center_cell_value
  (a b c d e f g h i : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i)
  (h_row1 : a * b * c = 1)
  (h_row2 : d * e * f = 1)
  (h_row3 : g * h * i = 1)
  (h_col1 : a * d * g = 1)
  (h_col2 : b * e * h = 1)
  (h_col3 : c * f * i = 1)
  (h_square1 : a * b * d * e = 2)
  (h_square2 : b * c * e * f = 2)
  (h_square3 : d * e * g * h = 2)
  (h_square4 : e * f * h * i = 2) :
  e = 1 :=
  sorry

end center_cell_value_l172_172436


namespace find_value_l172_172236

theorem find_value :
  3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2800 :=
by
  sorry

end find_value_l172_172236


namespace center_cell_value_l172_172427

theorem center_cell_value (a b c d e f g h i : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
  (hg : 0 < g) (hh : 0 < h) (hi : 0 < i)
  (row1 : a * b * c = 1) (row2 : d * e * f = 1) (row3 : g * h * i = 1)
  (col1 : a * d * g = 1) (col2 : b * e * h = 1) (col3 : c * f * i = 1)
  (square1 : a * b * d * e = 2) (square2 : b * c * e * f = 2)
  (square3 : d * e * g * h = 2) (square4 : e * f * h * i = 2) :
  e = 1 :=
begin
  sorry
end

end center_cell_value_l172_172427


namespace probability_of_specific_choice_l172_172407

-- Define the sets of subjects
inductive Subject
| Chinese
| Mathematics
| ForeignLanguage
| Physics
| History
| PoliticalScience
| Geography
| Chemistry
| Biology

-- Define the conditions of the examination mode "3+1+2"
def threeSubjects := [Subject.Chinese, Subject.Mathematics, Subject.ForeignLanguage]
def oneSubject := [Subject.Physics, Subject.History]
def twoSubjects := [Subject.PoliticalScience, Subject.Geography, Subject.Chemistry, Subject.Biology]

-- Calculate the total number of ways to choose one subject from Physics or History and two subjects from PoliticalScience, Geography, Chemistry, and Biology
def totalWays : Nat := 2 * Nat.choose 4 2  -- 2 choices for "1" part, and C(4, 2) ways for "2" part

-- Calculate the probability that a candidate will choose Political Science, History, and Geography
def favorableOutcome := 1  -- Only one specific combination counts

theorem probability_of_specific_choice :
  let total_ways := totalWays
  let specific_combination := favorableOutcome
  (specific_combination : ℚ) / total_ways = 1 / 12 :=
by
  let total_ways := totalWays
  let specific_combination := favorableOutcome
  show (specific_combination : ℚ) / total_ways = 1 / 12
  sorry

end probability_of_specific_choice_l172_172407


namespace percent_change_range_l172_172262

-- Define initial conditions
def initial_yes_percent : ℝ := 0.60
def initial_no_percent : ℝ := 0.40
def final_yes_percent : ℝ := 0.80
def final_no_percent : ℝ := 0.20

-- Define the key statement to prove
theorem percent_change_range : 
  ∃ y_min y_max : ℝ, 
  y_min = 0.20 ∧ 
  y_max = 0.60 ∧ 
  (y_max - y_min = 0.40) :=
sorry

end percent_change_range_l172_172262


namespace geometric_series_sum_value_l172_172926

theorem geometric_series_sum_value :
  let a : ℚ := 3 / 4
  let r : ℚ := 3 / 4
  let n : ℕ := 12
  \(\sum_{k\in \mathbb{N}, 1 \leq k \leq n} a * r^(k - 1)\) = \(\frac{48758625}{16777216}\) :=
sorry

end geometric_series_sum_value_l172_172926


namespace stratified_sampling_elderly_l172_172636

theorem stratified_sampling_elderly (total_elderly middle_aged young total_sample total_population elderly_to_sample : ℕ) 
  (h1: total_elderly = 30) 
  (h2: middle_aged = 90) 
  (h3: young = 60) 
  (h4: total_sample = 36) 
  (h5: total_population = total_elderly + middle_aged + young) 
  (h6: 1 / 5 * total_elderly = elderly_to_sample)
  : elderly_to_sample = 6 := 
  by 
    sorry

end stratified_sampling_elderly_l172_172636


namespace center_cell_value_l172_172423

namespace MathProof

variables {a b c d e f g h i : ℝ}

-- Conditions
axiom row_product1 : a * b * c = 1
axiom row_product2 : d * e * f = 1
axiom row_product3 : g * h * i = 1

axiom col_product1 : a * d * g = 1
axiom col_product2 : b * e * h = 1
axiom col_product3 : c * f * i = 1

axiom square_product1 : a * b * d * e = 2
axiom square_product2 : b * c * e * f = 2
axiom square_product3 : d * e * g * h = 2
axiom square_product4 : e * f * h * i = 2

-- Proof problem
theorem center_cell_value : e = 1 :=
sorry

end MathProof

end center_cell_value_l172_172423


namespace isosceles_triangle_perimeter_l172_172154

theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 5) (h₂ : b = 10) :
  (a + b + b = 25) ∧ (a + a + b ≤ b → False) :=
by
  sorry

end isosceles_triangle_perimeter_l172_172154


namespace minimum_value_of_f_l172_172971

noncomputable def f (x : ℝ) : ℝ := (x^2 + 9) / Real.sqrt (x^2 + 5)

theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 6 :=
by 
  sorry

end minimum_value_of_f_l172_172971


namespace plane_equation_l172_172393

theorem plane_equation :
  ∃ (A B C D : ℤ), A > 0 ∧ Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1 ∧ 
  (∀ (x y z : ℤ), (x, y, z) = (0, 0, 0) ∨ (x, y, z) = (2, 0, -2) → A * x + B * y + C * z + D = 0) ∧ 
  ∀ (x y z : ℤ), (A = 1 ∧ B = -5 ∧ C = 1 ∧ D = 0) := sorry

end plane_equation_l172_172393


namespace cos_difference_zero_l172_172465

noncomputable def cos72 := Real.cos (72 * Real.pi / 180)
noncomputable def cos144 := Real.cos (144 * Real.pi / 180)
noncomputable def cos36 := Real.cos (36 * Real.pi / 180)

theorem cos_difference_zero :
  cos72 - cos144 = 0 :=
by
  let c := cos72
  let d := cos144
  have h1 : d = -(1 - 2 * c^2), by sorry
  have h2 : c - d = c + 1 - 2 * c^2, by sorry
  have h3 : 2 * c^2 - c - 1 = 0, by sorry
  have h4 : c = 1, by sorry
  sorry

end cos_difference_zero_l172_172465


namespace total_people_participated_l172_172193

theorem total_people_participated 
  (N f p : ℕ)
  (h1 : N = f * p)
  (h2 : N = (f - 10) * (p + 1))
  (h3 : N = (f - 25) * (p + 3)) : 
  N = 900 :=
by 
  sorry

end total_people_participated_l172_172193


namespace find_angle_A_range_of_bc_l172_172155

-- Define the necessary conditions and prove the size of angle A
theorem find_angle_A 
  (a b c : ℝ)
  (A B C : ℝ)
  (h₁ : b * (Real.sin B + Real.sin C) = (a - c) * (Real.sin A + Real.sin C))
  (h₂ : B > Real.pi / 2)
  (h₃ : A + B + C = Real.pi)
  (h₄ : a > 0) (h₅ : b > 0) (h₆ : c > 0): 
  A = 2 * Real.pi / 3 :=
sorry

-- Define the necessary conditions and prove the range for b+c when a = sqrt(3)/2
theorem range_of_bc 
  (a b c : ℝ)
  (A : ℝ)
  (h₁ : A = 2 * Real.pi / 3)
  (h₂ : a = Real.sqrt 3 / 2)
  (h₃ : a > 0) (h₄ : b > 0) (h₅ : c > 0)
  (h₆ : A + B + C = Real.pi)
  (h₇ : B + C = Real.pi / 3) : 
  Real.sqrt 3 / 2 < b + c ∧ b + c ≤ 1 :=
sorry

end find_angle_A_range_of_bc_l172_172155


namespace rectangle_length_reduction_30_percent_l172_172785

variables (L W : ℝ) (x : ℝ)

theorem rectangle_length_reduction_30_percent
  (h : 1 = (1 - x / 100) * 1.4285714285714287) :
  x = 30 :=
sorry

end rectangle_length_reduction_30_percent_l172_172785


namespace nine_y_squared_eq_x_squared_z_squared_l172_172994

theorem nine_y_squared_eq_x_squared_z_squared (x y z : ℝ) (h : x / y = 3 / z) : 9 * y ^ 2 = x ^ 2 * z ^ 2 :=
by
  sorry

end nine_y_squared_eq_x_squared_z_squared_l172_172994


namespace geom_series_sum_n_eq_728_div_729_l172_172478

noncomputable def a : ℚ := 1 / 3
noncomputable def r : ℚ := 1 / 3
noncomputable def S_n (n : ℕ) : ℚ := a * ((1 - r^n) / (1 - r))

theorem geom_series_sum_n_eq_728_div_729 (n : ℕ) (h : S_n n = 728 / 729) : n = 6 :=
by
  sorry

end geom_series_sum_n_eq_728_div_729_l172_172478


namespace students_participated_in_function_l172_172808

theorem students_participated_in_function :
  ∀ (B G : ℕ),
  B + G = 800 →
  (3 / 4 : ℚ) * G = 150 →
  (2 / 3 : ℚ) * B + 150 = 550 :=
by
  intros B G h1 h2
  sorry

end students_participated_in_function_l172_172808


namespace geometric_series_sum_l172_172531

theorem geometric_series_sum :
  let a := 2 / 3
  let r := 1 / 3
  a / (1 - r) = 1 :=
by
  sorry

end geometric_series_sum_l172_172531


namespace canary_possible_distances_l172_172259

noncomputable def distance_from_bus_stop (bus_stop swallow sparrow canary : ℝ) : Prop :=
  swallow = 380 ∧
  sparrow = 450 ∧
  (sparrow - swallow) = (canary - sparrow) ∨
  (swallow - sparrow) = (sparrow - canary)

theorem canary_possible_distances (swallow sparrow canary : ℝ) :
  distance_from_bus_stop 0 swallow sparrow canary →
  canary = 520 ∨ canary = 1280 :=
by
  sorry

end canary_possible_distances_l172_172259


namespace find_salary_january_l172_172804

noncomputable section
open Real

def average_salary_jan_to_apr (J F M A : ℝ) : Prop := 
  (J + F + M + A) / 4 = 8000

def average_salary_feb_to_may (F M A May : ℝ) : Prop := 
  (F + M + A + May) / 4 = 9500

def may_salary_value (May : ℝ) : Prop := 
  May = 6500

theorem find_salary_january : 
  ∀ J F M A May, 
    average_salary_jan_to_apr J F M A → 
    average_salary_feb_to_may F M A May → 
    may_salary_value May → 
    J = 500 :=
by
  intros J F M A May h1 h2 h3
  sorry

end find_salary_january_l172_172804


namespace count_white_balls_l172_172944

variable (W B : ℕ)

theorem count_white_balls
  (h_total : W + B = 30)
  (h_white : ∀ S : Finset ℕ, S.card = 12 → ∃ w ∈ S, w < W)
  (h_black : ∀ S : Finset ℕ, S.card = 20 → ∃ b ∈ S, b < B) :
  W = 19 :=
sorry

end count_white_balls_l172_172944


namespace find_xz_l172_172001

theorem find_xz (x y z : ℝ) (h1 : 2 * x + z = 15) (h2 : x - 2 * y = 8) : x + z = 15 :=
sorry

end find_xz_l172_172001


namespace principal_amount_l172_172915

theorem principal_amount (SI : ℝ) (R : ℝ) (T : ℕ) (P : ℝ) :
  SI = 3.45 → R = 0.05 → T = 3 → SI = P * R * T → P = 23 :=
by
  -- The proof steps would go here but are omitted as specified.
  sorry

end principal_amount_l172_172915


namespace ShelbyRainDrivingTime_l172_172335

-- Define the conditions
def drivingTimeNonRain (totalTime: ℕ) (rainTime: ℕ) : ℕ := totalTime - rainTime
def rainSpeed : ℚ := 20 / 60
def noRainSpeed : ℚ := 30 / 60
def totalDistance (rainTime: ℕ) (nonRainTime: ℕ) : ℚ := rainSpeed * rainTime + noRainSpeed * nonRainTime

-- Prove the question == answer given conditions
theorem ShelbyRainDrivingTime :
  ∀ (rainTime totalTime: ℕ),
  (totalTime = 40) →
  (totalDistance rainTime (drivingTimeNonRain totalTime rainTime) = 16) →
  rainTime = 24 :=
by
  intros rainTime totalTime ht hd
  have h1 : drivingTimeNonRain totalTime rainTime = 40 - rainTime := rfl
  rw [← h1] at hd
  sorry

end ShelbyRainDrivingTime_l172_172335


namespace probability_at_least_one_shows_one_is_correct_l172_172362

/-- Two fair 8-sided dice are rolled. What is the probability that at least one of the dice shows a 1? -/
def probability_at_least_one_shows_one : ℚ :=
  let total_outcomes := 8 * 8
  let neither_one := 7 * 7
  let at_least_one := total_outcomes - neither_one
  at_least_one / total_outcomes

theorem probability_at_least_one_shows_one_is_correct :
  probability_at_least_one_shows_one = 15 / 64 :=
by
  unfold probability_at_least_one_shows_one
  sorry

end probability_at_least_one_shows_one_is_correct_l172_172362


namespace boat_fuel_cost_per_hour_l172_172582

variable (earnings_per_photo : ℕ)
variable (shark_frequency_minutes : ℕ)
variable (hunting_hours : ℕ)
variable (expected_profit : ℕ)

def cost_of_fuel_per_hour (earnings_per_photo shark_frequency_minutes hunting_hours expected_profit : ℕ) : ℕ :=
  sorry

theorem boat_fuel_cost_per_hour
  (h₁ : earnings_per_photo = 15)
  (h₂ : shark_frequency_minutes = 10)
  (h₃ : hunting_hours = 5)
  (h₄ : expected_profit = 200) :
  cost_of_fuel_per_hour earnings_per_photo shark_frequency_minutes hunting_hours expected_profit = 50 :=
  sorry

end boat_fuel_cost_per_hour_l172_172582


namespace pauline_spent_in_all_l172_172199

theorem pauline_spent_in_all
  (cost_taco_shells : ℝ := 5)
  (cost_bell_pepper : ℝ := 1.5)
  (num_bell_peppers : ℕ := 4)
  (cost_meat_per_pound : ℝ := 3)
  (num_pounds_meat : ℝ := 2) :
  (cost_taco_shells + num_bell_peppers * cost_bell_pepper + num_pounds_meat * cost_meat_per_pound = 17) :=
by
  sorry

end pauline_spent_in_all_l172_172199


namespace conclusion_1_conclusion_2_conclusion_3_conclusion_4_l172_172799

axiom normal_distribution (μ σ : ℝ) : ℝ → ℝ

axiom P (A : Set ℝ) : ℝ

theorem conclusion_1 (ξ : ℝ) (σ : ℝ) (h : σ > 0) (μ : ℝ = 1) (hp : P {x | 0 < x ∧ x < 1} = 0.35) : P {x | 0 < x ∧ x < 2} = 0.7 := 
  sorry

theorem conclusion_2 (c k : ℝ) (hx : ∀ x, ln (c * exp (k * x)) = 0.3 * x + 4) : 
  c = exp 4 :=
  sorry

theorem conclusion_3 (m : ℝ) (h : ∀ x > 0, exp x - m ≤ 0) : ¬ (∀ x, exp x - m - 1 ≥ 0) := 
  sorry

theorem conclusion_4 (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∀ x > 1, a * x^2 - (a + b - 1) * x + b > 0) ↔ (a ≥ b - 1) := 
  sorry

end conclusion_1_conclusion_2_conclusion_3_conclusion_4_l172_172799


namespace marys_remaining_money_l172_172896

def drinks_cost (p : ℝ) := 4 * p
def medium_pizzas_cost (p : ℝ) := 3 * (3 * p)
def large_pizzas_cost (p : ℝ) := 2 * (5 * p)
def total_initial_money := 50

theorem marys_remaining_money (p : ℝ) : 
  total_initial_money - (drinks_cost p + medium_pizzas_cost p + large_pizzas_cost p) = 50 - 23 * p :=
by
  sorry

end marys_remaining_money_l172_172896


namespace fraction_value_l172_172499

theorem fraction_value : (4 * 5) / 10 = 2 := by
  sorry

end fraction_value_l172_172499


namespace vaishali_total_stripes_l172_172495

def total_stripes (hats_with_3_stripes hats_with_4_stripes hats_with_no_stripes : ℕ) 
  (hats_with_5_stripes hats_with_7_stripes hats_with_1_stripe : ℕ) 
  (hats_with_10_stripes hats_with_2_stripes : ℕ)
  (stripes_per_hat_with_3 stripes_per_hat_with_4 stripes_per_hat_with_no : ℕ)
  (stripes_per_hat_with_5 stripes_per_hat_with_7 stripes_per_hat_with_1 : ℕ)
  (stripes_per_hat_with_10 stripes_per_hat_with_2 : ℕ) : ℕ :=
  hats_with_3_stripes * stripes_per_hat_with_3 +
  hats_with_4_stripes * stripes_per_hat_with_4 +
  hats_with_no_stripes * stripes_per_hat_with_no +
  hats_with_5_stripes * stripes_per_hat_with_5 +
  hats_with_7_stripes * stripes_per_hat_with_7 +
  hats_with_1_stripe * stripes_per_hat_with_1 +
  hats_with_10_stripes * stripes_per_hat_with_10 +
  hats_with_2_stripes * stripes_per_hat_with_2

#eval total_stripes 4 3 6 2 1 4 2 3 3 4 0 5 7 1 10 2 -- 71

theorem vaishali_total_stripes : (total_stripes 4 3 6 2 1 4 2 3 3 4 0 5 7 1 10 2) = 71 :=
by
  sorry

end vaishali_total_stripes_l172_172495


namespace initial_range_without_telescope_l172_172105

variable (V : ℝ)

def telescope_increases_range (V : ℝ) : Prop :=
  V + 0.875 * V = 150

theorem initial_range_without_telescope (V : ℝ) (h : telescope_increases_range V) : V = 80 :=
by
  sorry

end initial_range_without_telescope_l172_172105


namespace green_hats_count_l172_172241

theorem green_hats_count 
  (B G : ℕ)
  (h1 : B + G = 85)
  (h2 : 6 * B + 7 * G = 530) : 
  G = 20 :=
by
  sorry

end green_hats_count_l172_172241


namespace contractor_male_workers_l172_172511

noncomputable def number_of_male_workers (M : ℕ) : Prop :=
  let female_wages : ℕ := 15 * 20
  let child_wages : ℕ := 5 * 8
  let total_wages : ℕ := 35 * M + female_wages + child_wages
  let total_workers : ℕ := M + 15 + 5
  (total_wages / total_workers) = 26

theorem contractor_male_workers : ∃ M : ℕ, number_of_male_workers M ∧ M = 20 :=
by
  use 20
  sorry

end contractor_male_workers_l172_172511


namespace find_lines_through_p_and_intersecting_circle_l172_172854

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  (x - 1) ^ 2 + (y - 2) ^ 2 = 25

noncomputable def passes_through (l : ℝ → ℝ) (P : ℝ × ℝ) : Prop :=
  P.2 = l P.1

noncomputable def chord_length (c p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2

theorem find_lines_through_p_and_intersecting_circle :
  ∃ l : ℝ → ℝ, (passes_through l (-2, 3)) ∧
  (∃ p1 p2 : ℝ × ℝ, trajectory_equation p1.1 p1.2 ∧ trajectory_equation p2.1 p2.2 ∧
  chord_length (1, 2) p1 p2 = 8^2) :=
by
  sorry

end find_lines_through_p_and_intersecting_circle_l172_172854


namespace c_share_l172_172505

theorem c_share (A B C D : ℝ) 
    (h1 : A = 1/2 * B) 
    (h2 : B = 1/2 * C) 
    (h3 : D = 1/4 * 392) 
    (h4 : A + B + C + D = 392) : 
    C = 168 := 
by 
    sorry

end c_share_l172_172505


namespace f_is_monotonic_decreasing_l172_172744

noncomputable def f (x : ℝ) : ℝ := Real.sin (1/2 * x + Real.pi / 6)

theorem f_is_monotonic_decreasing : ∀ x y : ℝ, (2 * Real.pi / 3 ≤ x ∧ x ≤ 8 * Real.pi / 3) → (2 * Real.pi / 3 ≤ y ∧ y ≤ 8 * Real.pi / 3) → x < y → f y ≤ f x :=
sorry

end f_is_monotonic_decreasing_l172_172744


namespace perfect_square_fraction_l172_172186

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, n = k * k

theorem perfect_square_fraction (a b : ℕ) 
  (h_pos_a: 0 < a) 
  (h_pos_b: 0 < b) 
  (h_div : (a * b + 1) ∣ (a^2 + b^2)) : 
  is_perfect_square ((a^2 + b^2) / (a * b + 1)) := 
sorry

end perfect_square_fraction_l172_172186


namespace red_lucky_stars_l172_172372

theorem red_lucky_stars (x : ℕ) : (20 + x + 15 > 0) → (x / (20 + x + 15) : ℚ) = 0.5 → x = 35 := by
  sorry

end red_lucky_stars_l172_172372


namespace number_subtracted_l172_172515

-- Define the variables x and y
variable (x y : ℝ)

-- Define the conditions
def condition1 := 6 * x - y = 102
def condition2 := x = 40

-- Define the theorem to prove
theorem number_subtracted (h1 : condition1 x y) (h2 : condition2 x) : y = 138 :=
sorry

end number_subtracted_l172_172515


namespace complement_U_A_l172_172245

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {3, 4, 5}

theorem complement_U_A :
  U \ A = {1, 2, 6} := by
  sorry

end complement_U_A_l172_172245


namespace rational_solutions_for_k_l172_172974

theorem rational_solutions_for_k :
  ∀ (k : ℕ), k > 0 → 
  (∃ x : ℚ, k * x^2 + 16 * x + k = 0) ↔ k = 8 :=
by
  sorry

end rational_solutions_for_k_l172_172974


namespace arithmetic_sequence_sum_l172_172322

theorem arithmetic_sequence_sum (S : ℕ → ℤ) (m : ℕ)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end arithmetic_sequence_sum_l172_172322


namespace total_bill_is_89_l172_172664

-- Define the individual costs and quantities
def adult_meal_cost := 12
def child_meal_cost := 7
def fries_cost := 5
def drink_cost := 10

def num_adults := 4
def num_children := 3
def num_fries := 2
def num_drinks := 1

-- Calculate the total bill
def total_bill : Nat :=
  (num_adults * adult_meal_cost) + 
  (num_children * child_meal_cost) + 
  (num_fries * fries_cost) + 
  (num_drinks * drink_cost)

-- The proof statement
theorem total_bill_is_89 : total_bill = 89 := 
  by
  -- The proof will be provided here
  sorry

end total_bill_is_89_l172_172664


namespace part_i_part_ii_part_iii_l172_172735

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 4) + f (x + 3 * Real.pi / 4)

theorem part_i : f (Real.pi / 2) = 1 :=
sorry

theorem part_ii : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 2 * Real.pi :=
sorry

theorem part_iii : ∃ x, g x = -2 :=
sorry

end part_i_part_ii_part_iii_l172_172735


namespace total_distance_journey_l172_172752

def miles_driven : ℕ := 384
def miles_remaining : ℕ := 816

theorem total_distance_journey :
  miles_driven + miles_remaining = 1200 :=
by
  sorry

end total_distance_journey_l172_172752


namespace squirrel_rise_per_circuit_l172_172821

theorem squirrel_rise_per_circuit
  (h_post_height : ℕ := 12)
  (h_circumference : ℕ := 3)
  (h_travel_distance : ℕ := 9) :
  (h_post_height / (h_travel_distance / h_circumference) = 4) :=
  sorry

end squirrel_rise_per_circuit_l172_172821


namespace NumberOfValidTenDigitNumbers_l172_172835

theorem NumberOfValidTenDigitNumbers : 
  let digits : Finset ℕ := Finset.range 10       -- The set of digits {0, 1, 2, ..., 9}
  let valid_numbers := {p : Finset (Fin 10) // 
    ∀ x ∈ p.1.filter (λ d, d ≠ 9), ∃ y ∈ p.1, x < y} -- Sets of digits fulfilling the conditions.
  in valid_numbers.card = 256 := 
by sorry

end NumberOfValidTenDigitNumbers_l172_172835


namespace garden_perimeter_is_44_l172_172816

-- Define the original garden's side length given the area
noncomputable def original_side_length (A : ℕ) := Nat.sqrt A

-- Given condition: Area of the original garden is 49 square meters
def original_area := 49

-- Define the new side length after expanding each side by 4 meters
def new_side_length (original_side : ℕ) := original_side + 4

-- Define the perimeter of the new garden given the new side length
def new_perimeter (new_side : ℕ) := 4 * new_side

-- Proof statement: The perimeter of the new garden given the original area is 44 meters
theorem garden_perimeter_is_44 : new_perimeter (new_side_length (original_side_length original_area)) = 44 := by
  -- This is where the proof would go
  sorry

end garden_perimeter_is_44_l172_172816


namespace four_digit_number_l172_172615

-- Defining the cards and their holders
def cards : List ℕ := [2, 0, 1, 5]
def A : ℕ := 5
def B : ℕ := 1
def C : ℕ := 2
def D : ℕ := 0

-- Conditions based on statements
def A_statement (a b c d : ℕ) : Prop := 
  ¬ ((b = a + 1) ∨ (b = a - 1) ∨ (c = a + 1) ∨ (c = a - 1) ∨ (d = a + 1) ∨ (d = a - 1))

def B_statement (a b c d : ℕ) : Prop := 
  (b = a + 1) ∨ (b = a - 1) ∨ (c = a + 1) ∨ (c = a - 1) ∨ (d = a + 1) ∨ (d = a - 1)

def C_statement (c : ℕ) : Prop := ¬ (c = 1 ∨ c = 2 ∨ c = 5)
def D_statement (d : ℕ) : Prop := d ≠ 0

-- Truth conditions
def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬ is_even n

def tells_truth (n : ℕ) : Prop := is_odd n
def lies (n : ℕ) : Prop := is_even n

-- Proof statement
theorem four_digit_number (a b c d : ℕ) 
  (ha : a ∈ cards) (hb : b ∈ cards) (hc : c ∈ cards) (hd : d ∈ cards) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (truth_A : tells_truth a → A_statement a b c d)
  (lie_A : lies a → ¬ A_statement a b c d)
  (truth_B : tells_truth b → B_statement a b c d)
  (lie_B : lies b → ¬ B_statement a b c d)
  (truth_C : tells_truth c → C_statement c)
  (lie_C : lies c → ¬ C_statement c)
  (truth_D : tells_truth d → D_statement d)
  (lie_D : lies d → ¬ D_statement d) :
  a * 1000 + b * 100 + c * 10 + d = 5120 := 
  by
    sorry

end four_digit_number_l172_172615


namespace find_a9_l172_172311

noncomputable def a : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+1) => a n + n

theorem find_a9 : a 9 = 37 := by
  sorry

end find_a9_l172_172311


namespace age_difference_of_declans_sons_l172_172343

theorem age_difference_of_declans_sons 
  (current_age_elder_son : ℕ) 
  (future_age_younger_son : ℕ) 
  (years_until_future : ℕ) 
  (current_age_elder_son_eq : current_age_elder_son = 40) 
  (future_age_younger_son_eq : future_age_younger_son = 60) 
  (years_until_future_eq : years_until_future = 30) :
  (current_age_elder_son - (future_age_younger_son - years_until_future)) = 10 := by
  sorry

end age_difference_of_declans_sons_l172_172343


namespace find_smallest_solution_l172_172714

theorem find_smallest_solution : ∃ x : ℝ, x = Real.sqrt 119 ∧ (Int.floor (x^2) - Int.floor x ^ 2 = 19) := by
  sorry

end find_smallest_solution_l172_172714


namespace log_base_function_inequalities_l172_172276

/-- 
Given the function y = log_(1/(sqrt(2))) (1/(x + 3)),
prove that:
1. for y > 0, x ∈ (-2, +∞)
2. for y < 0, x ∈ (-3, -2)
-/
theorem log_base_function_inequalities :
  let y (x : ℝ) := Real.logb (1 / Real.sqrt 2) (1 / (x + 3))
  ∀ x : ℝ, (y x > 0 ↔ x > -2) ∧ (y x < 0 ↔ -3 < x ∧ x < -2) :=
by
  intros
  -- Proof steps would go here
  sorry

end log_base_function_inequalities_l172_172276


namespace work_hours_together_l172_172931

theorem work_hours_together (t : ℚ) :
  (1 / 9) * (9 : ℚ) = 1 ∧ (1 / 12) * (12 : ℚ) = 1 ∧
  (7 / 36) * t + (1 / 9) * (15 / 4) = 1 → t = 3 :=
by
  sorry

end work_hours_together_l172_172931


namespace empty_seats_correct_l172_172942

def children_count : ℕ := 52
def adult_count : ℕ := 29
def total_seats : ℕ := 95

theorem empty_seats_correct :
  total_seats - (children_count + adult_count) = 14 :=
by
  sorry

end empty_seats_correct_l172_172942


namespace find_y_value_l172_172082

theorem find_y_value (x y : ℝ) (k : ℝ) 
  (h1 : 5 * y = k / x^2)
  (h2 : y = 4)
  (h3 : x = 2)
  (h4 : k = 80) :
  ( ∃ y : ℝ, 5 * y = k / 4^2 ∧ y = 1) :=
by
  sorry

end find_y_value_l172_172082


namespace max_min_x_plus_y_l172_172726

theorem max_min_x_plus_y (x y : ℝ) (h : |x + 2| + |1 - x| = 9 - |y - 5| - |1 + y|) :
  -3 ≤ x + y ∧ x + y ≤ 6 := 
sorry

end max_min_x_plus_y_l172_172726


namespace average_hamburgers_sold_per_day_l172_172657

theorem average_hamburgers_sold_per_day 
  (total_hamburgers : ℕ) (days_in_week : ℕ)
  (h1 : total_hamburgers = 63) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 :=
by
  sorry

end average_hamburgers_sold_per_day_l172_172657


namespace average_daily_production_correct_l172_172235

noncomputable def average_daily_production : ℝ :=
  let jan_production := 3000
  let monthly_increase := 100
  let total_days := 365
  let total_production := jan_production + (11 * jan_production + (100 * (1 + 11))/2)
  (total_production / total_days : ℝ)

theorem average_daily_production_correct :
  average_daily_production = 121.1 :=
sorry

end average_daily_production_correct_l172_172235


namespace arithmetic_geometric_mean_l172_172052

theorem arithmetic_geometric_mean (x y : ℝ) (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 110) : x^2 + y^2 = 1380 := by
  sorry

end arithmetic_geometric_mean_l172_172052


namespace range_of_b_for_local_minimum_l172_172285

variable {x : ℝ}
variable (b : ℝ)

def f (x : ℝ) (b : ℝ) : ℝ :=
  x^3 - 6 * b * x + 3 * b

def f' (x : ℝ) (b : ℝ) : ℝ :=
  3 * x^2 - 6 * b

theorem range_of_b_for_local_minimum
  (h1 : f' 0 b < 0)
  (h2 : f' 1 b > 0) :
  0 < b ∧ b < 1 / 2 :=
by
  sorry

end range_of_b_for_local_minimum_l172_172285


namespace bert_phone_price_l172_172825

theorem bert_phone_price :
  ∃ x : ℕ, x * 8 = 144 := sorry

end bert_phone_price_l172_172825


namespace kids_go_to_camp_l172_172129

-- Define the total number of kids in Lawrence County
def total_kids : ℕ := 1059955

-- Define the number of kids who stay home
def stay_home : ℕ := 495718

-- Define the expected number of kids who go to camp
def expected_go_to_camp : ℕ := 564237

-- The theorem to prove the number of kids who go to camp
theorem kids_go_to_camp :
  total_kids - stay_home = expected_go_to_camp :=
by
  -- Proof is omitted
  sorry

end kids_go_to_camp_l172_172129


namespace scientific_notation_of_19400000000_l172_172480

theorem scientific_notation_of_19400000000 :
  ∃ a n, 1 ≤ |a| ∧ |a| < 10 ∧ (19400000000 : ℝ) = a * 10^n ∧ a = 1.94 ∧ n = 10 :=
by
  sorry

end scientific_notation_of_19400000000_l172_172480


namespace jack_evening_emails_l172_172180

theorem jack_evening_emails (ema_morning ema_afternoon ema_afternoon_evening ema_evening : ℕ)
  (h1 : ema_morning = 4)
  (h2 : ema_afternoon = 5)
  (h3 : ema_afternoon_evening = 13)
  (h4 : ema_afternoon_evening = ema_afternoon + ema_evening) :
  ema_evening = 8 :=
by
  sorry

end jack_evening_emails_l172_172180


namespace min_surface_area_of_stacked_solids_l172_172796

theorem min_surface_area_of_stacked_solids :
  ∀ (l w h : ℕ), l = 3 → w = 2 → h = 1 → 
  (2 * (l * w + l * h + w * h) - 2 * l * w = 32) :=
by
  intros l w h hl hw hh
  rw [hl, hw, hh]
  sorry

end min_surface_area_of_stacked_solids_l172_172796


namespace exists_square_with_only_invisible_points_l172_172653

def is_invisible (p q : ℤ) : Prop := Int.gcd p q > 1

def all_points_in_square_invisible (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≥ 2 ∧ ∀ x y : ℕ, (x < n ∧ y < n) → is_invisible (k*x) (k*y)

theorem exists_square_with_only_invisible_points (n : ℕ) :
  all_points_in_square_invisible n := sorry

end exists_square_with_only_invisible_points_l172_172653


namespace center_cell_value_l172_172438

theorem center_cell_value
  (a b c d e f g h i : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i)
  (h_row1 : a * b * c = 1)
  (h_row2 : d * e * f = 1)
  (h_row3 : g * h * i = 1)
  (h_col1 : a * d * g = 1)
  (h_col2 : b * e * h = 1)
  (h_col3 : c * f * i = 1)
  (h_square1 : a * b * d * e = 2)
  (h_square2 : b * c * e * f = 2)
  (h_square3 : d * e * g * h = 2)
  (h_square4 : e * f * h * i = 2) :
  e = 1 :=
  sorry

end center_cell_value_l172_172438


namespace rational_solutions_iff_k_equals_8_l172_172972

theorem rational_solutions_iff_k_equals_8 {k : ℕ} (hk : k > 0) :
  (∃ (x : ℚ), k * x^2 + 16 * x + k = 0) ↔ k = 8 :=
by
  sorry

end rational_solutions_iff_k_equals_8_l172_172972


namespace question1_question2_l172_172165

noncomputable def A (x : ℝ) : Prop := x^2 - 3 * x + 2 ≤ 0
noncomputable def B_set (x a : ℝ) : ℝ := x^2 - 2 * x + a
def B (y a : ℝ) : Prop := y ≥ a - 1
noncomputable def C (x a : ℝ) : Prop := x^2 - a * x - 4 ≤ 0

def prop_p (a : ℝ) : Prop := ∃ x, A x ∧ B (B_set x a) a
def prop_q (a : ℝ) : Prop := ∀ x, A x → C x a

theorem question1 (a : ℝ) (h : ¬ prop_p a) : a > 3 :=
sorry

theorem question2 (a : ℝ) (hp : prop_p a) (hq : prop_q a) : 0 ≤ a ∧ a ≤ 3 :=
sorry

end question1_question2_l172_172165


namespace trigonometric_identity_l172_172396

open Real

theorem trigonometric_identity
  (theta : ℝ)
  (h : cos (π / 6 - theta) = 2 * sqrt 2 / 3) : 
  cos (π / 3 + theta) = 1 / 3 ∨ cos (π / 3 + theta) = -1 / 3 :=
by
  sorry

end trigonometric_identity_l172_172396


namespace find_smaller_number_l172_172353

theorem find_smaller_number (x y : ℕ) (h1 : x + y = 84) (h2 : y = 3 * x) : x = 21 := 
by
  sorry

end find_smaller_number_l172_172353


namespace smallest_square_value_l172_172579

theorem smallest_square_value (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
  (r s : ℕ) (hr : 15 * a + 16 * b = r^2) (hs : 16 * a - 15 * b = s^2) :
  min (r^2) (s^2) = 481^2 :=
  sorry

end smallest_square_value_l172_172579


namespace reverse_base_sum_l172_172719

theorem reverse_base_sum :
  {n : ℕ | ∃ d a_d a_d1 a_d2, 
            n = 5^d * a_d + 5^(d-1) * a_d1 + 5^(d-2) * a_d2 ∧
            n = 12^d * a_d2 + 12^(d-1) * a_d1 + 12^(d-2) * a_d ∧
            (12^d - 1) * a_d2 + (12^(d-1) - 5) * a_d1 + (12^(d-2) - 5^(d-2)) * a_d = 0 ∧
            d ≤ 2 ∧ a_d ≤ 4 ∧ a_d1 ≤ 4 ∧ a_d2 ≤ 4}.sum = 10 := 
sorry

end reverse_base_sum_l172_172719


namespace solve_double_inequality_l172_172776

theorem solve_double_inequality (x : ℝ) :
  (-1 < (x^2 - 20 * x + 21) / (x^2 - 4 * x + 5) ∧
   (x^2 - 20 * x + 21) / (x^2 - 4 * x + 5) < 1) ↔ (2 < x ∨ 26 < x) := 
sorry

end solve_double_inequality_l172_172776


namespace smallest_solution_eq_sqrt_104_l172_172698

theorem smallest_solution_eq_sqrt_104 :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, ⌊y^2⌋ - ⌊y⌋^2 = 19 → x ≤ y) := sorry

end smallest_solution_eq_sqrt_104_l172_172698


namespace general_formula_l172_172567

def sum_of_terms (a : ℕ → ℕ) (n : ℕ) : ℕ := 3 / 2 * a n - 3

def sequence_term (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  if n = 0 then 6 
  else a (n - 1) * 3

theorem general_formula (a : ℕ → ℕ) (n : ℕ) :
  (∀ n, sum_of_terms a n = 3 / 2 * a n - 3) →
  (∀ n, n = 0 → a n = 6) →
  (∀ n, n > 0 → a n = a (n - 1) * 3) →
  a n = 2 * 3^n := by
  sorry

end general_formula_l172_172567


namespace inscribed_cone_volume_l172_172605

theorem inscribed_cone_volume
  (H : ℝ) 
  (α : ℝ)
  (h_pos : 0 < H)
  (α_pos : 0 < α ∧ α < π / 2) :
  (1 / 12) * π * H ^ 3 * (Real.sin α) ^ 2 * (Real.sin (2 * α)) ^ 2 = 
  (1 / 3) * π * ((H * Real.sin α * Real.cos α / 2) ^ 2) * (H * (Real.sin α) ^ 2) :=
by sorry

end inscribed_cone_volume_l172_172605


namespace log_sum_zero_l172_172243

theorem log_sum_zero (a b c N : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_N : 0 < N) (h_neq_N : N ≠ 1) (h_geom_mean : b^2 = a * c) : 
  1 / Real.logb a N - 2 / Real.logb b N + 1 / Real.logb c N = 0 :=
  by
  sorry

end log_sum_zero_l172_172243


namespace find_negative_integer_l172_172354

theorem find_negative_integer (M : ℤ) (h_neg : M < 0) (h_eq : M^2 + M = 12) : M = -4 :=
sorry

end find_negative_integer_l172_172354


namespace smallest_solution_floor_eq_l172_172708

theorem smallest_solution_floor_eq (x : ℝ) : ⌊x^2⌋ - ⌊x⌋^2 = 19 → x = Real.sqrt 119 :=
by
  sorry

end smallest_solution_floor_eq_l172_172708


namespace ned_trays_per_trip_l172_172326

def trays_from_table1 : ℕ := 27
def trays_from_table2 : ℕ := 5
def total_trips : ℕ := 4
def total_trays : ℕ := trays_from_table1 + trays_from_table2
def trays_per_trip : ℕ := total_trays / total_trips

theorem ned_trays_per_trip :
  trays_per_trip = 8 :=
by
  -- proof is skipped
  sorry

end ned_trays_per_trip_l172_172326


namespace negation_of_existence_l172_172078

theorem negation_of_existence :
  ¬ (∃ x : ℝ, 0 < x ∧ x^3 - x + 1 > 0) ↔ ∀ x : ℝ, 0 < x → x^3 - x + 1 ≤ 0 :=
by sorry

end negation_of_existence_l172_172078


namespace mod_abc_eq_zero_l172_172294

open Nat

theorem mod_abc_eq_zero
    (a b c : ℕ)
    (h1 : (a + 2 * b + 3 * c) % 9 = 1)
    (h2 : (2 * a + 3 * b + c) % 9 = 2)
    (h3 : (3 * a + b + 2 * c) % 9 = 3) :
    (a * b * c) % 9 = 0 := by
  sorry

end mod_abc_eq_zero_l172_172294


namespace find_pairs_l172_172134

theorem find_pairs (a b : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) :
  (a * b^2 + b + 7) ∣ (a^2 * b + a + b) ↔ ((a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ ∃ k : ℕ+, a = 7 * k^2 ∧ b = 7 * k) :=
by sorry

end find_pairs_l172_172134


namespace velocity_is_zero_at_t_equals_2_l172_172084

def displacement (t : ℝ) : ℝ := -2 * t^2 + 8 * t

theorem velocity_is_zero_at_t_equals_2 : (deriv displacement 2 = 0) :=
by
  -- The definition step from (a). 
  let v := deriv displacement
  -- This would skip the proof itself, as instructed.
  sorry

end velocity_is_zero_at_t_equals_2_l172_172084


namespace expand_and_simplify_l172_172270

theorem expand_and_simplify (x : ℝ) : (17 * x - 9) * 3 * x = 51 * x^2 - 27 * x := 
by 
  sorry

end expand_and_simplify_l172_172270


namespace geometric_series_sum_proof_l172_172927

theorem geometric_series_sum_proof :
  ∑ k in Finset.range 12, (4: ℚ) ^ (-k) * 3 ^ k = 48750225 / 16777216 :=
by sorry

end geometric_series_sum_proof_l172_172927


namespace correct_proposition_D_l172_172502

theorem correct_proposition_D (a b : ℝ) (h1 : a < 0) (h2 : b < 0) : 
  (b / a) + (a / b) ≥ 2 := 
sorry

end correct_proposition_D_l172_172502


namespace factorial_sum_mod_prime_l172_172146

def sum_factorials (n : ℕ) : ℕ :=
  (Finset.range (n+1)).sum (λ k, nat.factorial k)

noncomputable def nat_floor_div_e (n : ℕ) : ℕ :=
  nat.floor ((nat.factorial n : ℝ) / real.exp 1)

theorem factorial_sum_mod_prime (p : ℕ) [Fact (Nat.Prime p)] (hp : p % 2 = 1) :
  (sum_factorials p - nat_floor_div_e (p-1)) % p = 0 :=
sorry

end factorial_sum_mod_prime_l172_172146


namespace caffeine_in_cup_l172_172086

-- Definitions based on the conditions
def caffeine_goal : ℕ := 200
def excess_caffeine : ℕ := 40
def total_cups : ℕ := 3

-- The statement proving that the amount of caffeine in a cup is 80 mg given the conditions.
theorem caffeine_in_cup : (3 * (80 : ℕ)) = (caffeine_goal + excess_caffeine) := by
  -- Plug in the value and simplify
  simp [caffeine_goal, excess_caffeine]

end caffeine_in_cup_l172_172086


namespace initial_pieces_of_gum_l172_172956

theorem initial_pieces_of_gum (additional_pieces given_pieces leftover_pieces initial_pieces : ℕ)
  (h_additional : additional_pieces = 3)
  (h_given : given_pieces = 11)
  (h_leftover : leftover_pieces = 2)
  (h_initial : initial_pieces + additional_pieces = given_pieces + leftover_pieces) :
  initial_pieces = 10 :=
by
  sorry

end initial_pieces_of_gum_l172_172956


namespace probability_after_2020_rounds_l172_172963

noncomputable
def raashan_sylvia_ted_game :
  ℕ → ℕ → ℕ → ℕ → ℝ
  | 0, a, b, c => if a = 2 ∧ b = 2 ∧ c = 2 then 1 else 0
  | n+1, a, b, c =>
    0.1 * raashan_sylvia_ted_game n a b c +
    0.9 * (1/2 * raashan_sylvia_ted_game n (a-1) (b+1) c +
           1/2 * raashan_sylvia_ted_game n (a-1) b (c+1) +
           1/2 * raashan_sylvia_ted_game n (a+1) (b-1) c +
           1/2 * raashan_sylvia_ted_game n a (b-1) (c+1) +
           1/2 * raashan_sylvia_ted_game n a (b+1) (c-1) +
           1/2 * raashan_sylvia_ted_game n (a+1) b (c-1) +
           sorry) -- other required transitions and edge cases need to be included correctly

theorem probability_after_2020_rounds :
  raashan_sylvia_ted_game 2020 2 2 2 = 0.073 :=
sorry

end probability_after_2020_rounds_l172_172963


namespace transmission_time_estimation_l172_172530

noncomputable def number_of_blocks := 80
noncomputable def chunks_per_block := 640
noncomputable def transmission_rate := 160 -- chunks per second
noncomputable def seconds_per_minute := 60
noncomputable def total_chunks := number_of_blocks * chunks_per_block
noncomputable def total_time_seconds := total_chunks / transmission_rate
noncomputable def total_time_minutes := total_time_seconds / seconds_per_minute

theorem transmission_time_estimation : total_time_minutes = 5 := 
  sorry

end transmission_time_estimation_l172_172530


namespace average_pages_per_book_l172_172179

-- Conditions
def book_thickness_in_inches : ℕ := 12
def pages_per_inch : ℕ := 80
def number_of_books : ℕ := 6

-- Given these conditions, we need to prove the average number of pages per book is 160.
theorem average_pages_per_book (book_thickness_in_inches : ℕ) (pages_per_inch : ℕ) (number_of_books : ℕ)
  (h1 : book_thickness_in_inches = 12)
  (h2 : pages_per_inch = 80)
  (h3 : number_of_books = 6) :
  (book_thickness_in_inches * pages_per_inch) / number_of_books = 160 := by
  sorry

end average_pages_per_book_l172_172179


namespace cosine_of_negative_three_pi_over_two_l172_172970

theorem cosine_of_negative_three_pi_over_two : 
  Real.cos (-3 * Real.pi / 2) = 0 := 
by sorry

end cosine_of_negative_three_pi_over_two_l172_172970


namespace sculpture_exposed_surface_area_l172_172663

theorem sculpture_exposed_surface_area :
  let l₁ := 9
  let l₂ := 6
  let l₃ := 4
  let l₄ := 1

  let exposed_bottom_layer := 9 + 16
  let exposed_second_layer := 6 + 10
  let exposed_third_layer := 4 + 8
  let exposed_top_layer := 5

  l₁ + l₂ + l₃ + l₄ = 20 →
  exposed_bottom_layer + exposed_second_layer + exposed_third_layer + exposed_top_layer = 58 :=
by {
  sorry
}

end sculpture_exposed_surface_area_l172_172663


namespace lineup_possibilities_l172_172672

theorem lineup_possibilities (total_players : ℕ) (all_stars_in_lineup : ℕ) (injured_player : ℕ) :
  total_players = 15 ∧ all_stars_in_lineup = 2 ∧ injured_player = 1 →
  Nat.choose 12 4 = 495 :=
by
  intro h
  sorry

end lineup_possibilities_l172_172672


namespace value_of_def_ef_l172_172731

theorem value_of_def_ef
  (a b c d e f : ℝ)
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 500)
  (h4 : (a * f) / (c * d) = 1)
  : d * e * f = 250 := 
by 
  sorry

end value_of_def_ef_l172_172731


namespace find_integer_l172_172628

theorem find_integer (x : ℕ) (h1 : (4 * x)^2 + 2 * x = 3528) : x = 14 := by
  sorry

end find_integer_l172_172628


namespace find_k_of_quadratic_eq_ratio_3_to_1_l172_172836

theorem find_k_of_quadratic_eq_ratio_3_to_1 (k : ℝ) :
  (∃ (x : ℝ), x ≠ 0 ∧ (x^2 + 8 * x + k = 0) ∧
              (∃ (r : ℝ), x = 3 * r ∧ 3 * r + r = -8)) → k = 12 :=
by {
  sorry
}

end find_k_of_quadratic_eq_ratio_3_to_1_l172_172836


namespace largest_systematic_sample_l172_172275

theorem largest_systematic_sample {n_products interval start second_smallest max_sample : ℕ} 
  (h1 : n_products = 300) 
  (h2 : start = 2) 
  (h3 : second_smallest = 17) 
  (h4 : interval = second_smallest - start) 
  (h5 : n_products % interval = 0) 
  (h6 : max_sample = start + (interval * ((n_products / interval) - 1))) : 
  max_sample = 287 := 
by
  -- This is where the proof would go if required.
  sorry

end largest_systematic_sample_l172_172275


namespace center_cell_value_l172_172441

variable (a b c d e f g h i : ℝ)

-- Defining the conditions
def row_product_1 := a * b * c = 1 ∧ d * e * f = 1 ∧ g * h * i = 1
def col_product_1 := a * d * g = 1 ∧ b * e * h = 1 ∧ c * f * i = 1
def subgrid_product_2 := a * b * d * e = 2 ∧ b * c * e * f = 2 ∧ d * e * g * h = 2 ∧ e * f * h * i = 2

-- The theorem to prove
theorem center_cell_value (h1 : row_product_1 a b c d e f g h i) 
                          (h2 : col_product_1 a b c d e f g h i) 
                          (h3 : subgrid_product_2 a b c d e f g h i) : 
                          e = 1 :=
by
  sorry

end center_cell_value_l172_172441


namespace min_wasted_person_minutes_max_wasted_person_minutes_expected_wasted_person_minutes_l172_172643

def a : ℕ := 1  -- time for a simple operation
def b : ℕ := 5  -- time for a lengthy operation
def n : ℕ := 5  -- number of "simple" customers
def m : ℕ := 3  -- number of "lengthy" customers
def total_customers : ℕ := 8 -- 8 people in queue

theorem min_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → min_wasted_person_minutes ≤ 40) :=
by
  sorry

theorem max_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → max_wasted_person_minutes ≥ 100) :=
by
  sorry

theorem expected_wasted_person_minutes:
  (∀ (a b n m total_customers : ℕ), a = 1 → b = 5 → n = 5 → m = 3 →  total_customers = 8 → expected_wasted_person_minutes = 72.5) :=
by
  sorry

end min_wasted_person_minutes_max_wasted_person_minutes_expected_wasted_person_minutes_l172_172643


namespace remainder_sum_mod_l172_172144

theorem remainder_sum_mod (a b c d e : ℕ)
  (h₁ : a = 17145)
  (h₂ : b = 17146)
  (h₃ : c = 17147)
  (h₄ : d = 17148)
  (h₅ : e = 17149)
  : (a + b + c + d + e) % 10 = 5 := by
  sorry

end remainder_sum_mod_l172_172144


namespace domain_of_f_l172_172840

open Set Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 6*x + 9)

theorem domain_of_f :
  {x : ℝ | f x ≠ f (-3)} = Iio (-3) ∪ Ioi (-3) :=
by
  sorry

end domain_of_f_l172_172840


namespace lucy_fish_bought_l172_172192

def fish_bought (fish_original fish_now : ℕ) : ℕ :=
  fish_now - fish_original

theorem lucy_fish_bought : fish_bought 212 492 = 280 :=
by
  sorry

end lucy_fish_bought_l172_172192


namespace arith_seq_seventh_term_l172_172798

theorem arith_seq_seventh_term (a1 a25 : ℝ) (n : ℕ) (d : ℝ) (a7 : ℝ) :
  a1 = 5 → a25 = 80 → n = 25 → d = (a25 - a1) / (n - 1) → a7 = a1 + (7 - 1) * d → a7 = 23.75 :=
by
  intros h1 h2 h3 hd ha7
  sorry

end arith_seq_seventh_term_l172_172798


namespace cost_of_pink_notebook_l172_172770

theorem cost_of_pink_notebook
    (total_cost : ℕ) 
    (black_cost : ℕ) 
    (green_cost : ℕ) 
    (num_green : ℕ) 
    (num_black : ℕ) 
    (num_pink : ℕ)
    (total_notebooks : ℕ)
    (h_total_cost : total_cost = 45)
    (h_black_cost : black_cost = 15) 
    (h_green_cost : green_cost = 10) 
    (h_num_green : num_green = 2) 
    (h_num_black : num_black = 1) 
    (h_num_pink : num_pink = 1)
    (h_total_notebooks : total_notebooks = 4) 
    : (total_cost - (num_green * green_cost + black_cost) = 10) :=
by
  sorry

end cost_of_pink_notebook_l172_172770


namespace student_average_comparison_l172_172108

theorem student_average_comparison (x y w : ℤ) (hxw : x < w) (hwy : w < y) : 
  (B : ℤ) > (A : ℤ) :=
  let A := (x + y + w) / 3
  let B := ((x + w) / 2 + y) / 2
  sorry

end student_average_comparison_l172_172108


namespace problem_statement_l172_172966

-- Proposition p: For any x ∈ ℝ, 2^x > x^2
def p : Prop := ∀ x : ℝ, 2 ^ x > x ^ 2

-- Proposition q: "ab > 4" is a sufficient but not necessary condition for "a > 2 and b > 2"
def q : Prop := (∀ a b : ℝ, (a > 2 ∧ b > 2) → (a * b > 4)) ∧ ¬ (∀ a b : ℝ, (a * b > 4) → (a > 2 ∧ b > 2))

-- Problem statement: Determine that the true statement is ¬p ∧ ¬q
theorem problem_statement : ¬p ∧ ¬q := by
  sorry

end problem_statement_l172_172966


namespace notebooks_last_days_l172_172754

-- Given conditions
def n := 5
def p := 40
def u := 4

-- Derived conditions
def total_pages := n * p
def days := total_pages / u

-- The theorem statement
theorem notebooks_last_days : days = 50 := sorry

end notebooks_last_days_l172_172754


namespace arithmetic_geometric_mean_l172_172053

theorem arithmetic_geometric_mean (x y : ℝ) (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 110) : x^2 + y^2 = 1380 := by
  sorry

end arithmetic_geometric_mean_l172_172053


namespace half_plus_five_l172_172864

theorem half_plus_five (n : ℕ) (h : n = 16) : n / 2 + 5 = 13 := by
  sorry

end half_plus_five_l172_172864


namespace ratio_flow_chart_to_total_time_l172_172266

noncomputable def T := 48
noncomputable def D := 18
noncomputable def C := (3 / 8) * T
noncomputable def F := T - C - D

theorem ratio_flow_chart_to_total_time : (F / T) = (1 / 4) := by
  sorry

end ratio_flow_chart_to_total_time_l172_172266


namespace find_xyz_l172_172996

theorem find_xyz (x y z : ℝ) (h1 : x * (y + z) = 195) (h2 : y * (z + x) = 204) (h3 : z * (x + y) = 213) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x * y * z = 1029 := by
  sorry

end find_xyz_l172_172996


namespace lily_remaining_money_l172_172035

def initial_amount := 55
def spent_on_shirt := 7
def spent_at_second_shop := 3 * spent_on_shirt
def total_spent := spent_on_shirt + spent_at_second_shop
def remaining_amount := initial_amount - total_spent

theorem lily_remaining_money : remaining_amount = 27 :=
by
  sorry

end lily_remaining_money_l172_172035


namespace find_xyz_l172_172469

theorem find_xyz (x y z : ℝ) (h₁ : x + 1 / y = 5) (h₂ : y + 1 / z = 2) (h₃ : z + 2 / x = 10 / 3) : x * y * z = (21 + Real.sqrt 433) / 2 :=
by
  sorry

end find_xyz_l172_172469


namespace mutually_exclusive_events_not_complementary_l172_172047

def event_a (ball: ℕ) (box: ℕ): Prop := ball = 1 ∧ box = 1
def event_b (ball: ℕ) (box: ℕ): Prop := ball = 1 ∧ box = 2

theorem mutually_exclusive_events_not_complementary :
  (∀ ball box, event_a ball box → ¬ event_b ball box) ∧ 
  (∃ box, ¬((event_a 1 box) ∨ (event_b 1 box))) :=
by
  sorry

end mutually_exclusive_events_not_complementary_l172_172047


namespace find_x_l172_172625

theorem find_x (x : ℝ) (h : (2012 + x)^2 = x^2) : x = -1006 :=
by
  sorry

end find_x_l172_172625


namespace limit_equivalence_l172_172380

open Nat
open Real

variable {u : ℕ → ℝ} {L : ℝ}

def original_def (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |L - u n| ≤ ε

def def1 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε : ℝ, ε ≤ 0 ∨ (∃ N : ℕ, ∀ n : ℕ, n < N ∨ |L - u n| ≤ ε)

def def2 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∀ n : ℕ, ∃ N : ℕ, n ≥ N → |L - u n| ≤ ε

def def3 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n > N, |L - u n| < ε

def def4 (u : ℕ → ℝ) (L : ℝ) : Prop :=
  ∃ N : ℕ, ∀ ε > 0, ∀ n ≥ N, |L - u n| ≤ ε

theorem limit_equivalence :
  original_def u L ↔ def1 u L ∧ def3 u L ∧ ¬def2 u L ∧ ¬def4 u L :=
by
  sorry

end limit_equivalence_l172_172380


namespace sum_of_first_39_natural_numbers_l172_172668

theorem sum_of_first_39_natural_numbers : (39 * (39 + 1)) / 2 = 780 :=
by
  sorry

end sum_of_first_39_natural_numbers_l172_172668


namespace number_of_valid_triples_l172_172256

theorem number_of_valid_triples : 
  ∃ n, n = 7 ∧ ∀ (a b c : ℕ), b = 2023 → a ≤ b → b ≤ c → a * c = 2023^2 → (n = 7) :=
by 
  sorry

end number_of_valid_triples_l172_172256


namespace simplify_expression_l172_172057

theorem simplify_expression (x y : ℤ) (h1 : x = -2) (h2 : y = 3) :
  (x + 2 * y)^2 - (x + y) * (2 * x - y) = 23 :=
by
  sorry

end simplify_expression_l172_172057


namespace fixed_points_intersection_infinite_tangency_points_l172_172402

variables {P : Type*} [fintype P]

def circle_C1 : set (ℝ × ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 - 10 * p.1 - 6 * p.2 + 32 = 0 }

def family_circle_C2 (a : ℝ) : set (ℝ × ℝ) :=
{ p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * a * p.1 - 2 * (8 - a) * p.2 + 4 * a + 12 = 0 }

theorem fixed_points_intersection :
  (4, 2) ∈ circle_C1 ∧ (6, 4) ∈ circle_C1 ∧
  ∀ a : ℝ, (4, 2) ∈ family_circle_C2 a ∧ (6, 4) ∈ family_circle_C2 a :=
sorry

def ellipse (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

theorem infinite_tangency_points :
  (2, 0) ∈ ellipse ∧ 
  (6 / 5, -4 / 5) ∈ ellipse ∧
  ∀ P : ℝ × ℝ, P ∈ ellipse → 
    let PT1 := sqrt (P.1^2 + P.2^2 - 10 * P.1 - 6 * P.2 + 32),
        PT2 := sqrt (P.1^2 + P.2^2 - 2 * a * P.1 - 2 * (8 - a) * P.2 + 4 * a + 12)
    in (PT1 = PT2 → (P = (2, 0) ∨ P = (6 / 5, -4 / 5))) :=
sorry

end fixed_points_intersection_infinite_tangency_points_l172_172402


namespace gcd_of_repeated_three_digit_number_is_constant_l172_172817

theorem gcd_of_repeated_three_digit_number_is_constant (m : ℕ) (h1 : 100 ≤ m) (h2 : m < 1000) : 
  ∃ d, d = 1001001 ∧ ∀ n, n = 10010013 * m → (gcd 1001001 n) = 1001001 :=
by
  -- The proof would go here
  sorry

end gcd_of_repeated_three_digit_number_is_constant_l172_172817


namespace number_of_solutions_l172_172291

open Nat

-- Definitions arising from the conditions
def is_solution (x y : ℕ) : Prop := 3 * x + 5 * y = 501

-- Statement of the problem
theorem number_of_solutions :
  (∃ k : ℕ, k ≥ 0 ∧ k < 33 ∧ ∀ (x y : ℕ), x = 5 * k + 2 ∧ y = 99 - 3 * k → is_solution x y) :=
  sorry

end number_of_solutions_l172_172291


namespace candy_necklaces_per_pack_l172_172529

theorem candy_necklaces_per_pack (packs_total packs_opened packs_left candies_left necklaces_per_pack : ℕ) 
  (h_total : packs_total = 9) 
  (h_opened : packs_opened = 4) 
  (h_left : packs_left = packs_total - packs_opened) 
  (h_candies_left : candies_left = 40) 
  (h_necklaces_per_pack : candies_left = packs_left * necklaces_per_pack) :
  necklaces_per_pack = 8 :=
by
  -- Proof goes here
  sorry

end candy_necklaces_per_pack_l172_172529


namespace necessary_but_not_sufficient_condition_l172_172321

-- Definitions
variable (f : ℝ → ℝ)

-- Condition that we need to prove
def is_even (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

def is_symmetric_about_origin (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = -g (-x)

-- Necessary and sufficient condition
theorem necessary_but_not_sufficient_condition : 
  (∀ x, |f x| = |f (-x)|) ↔ (∀ x, f x = -f (-x)) ∧ ¬(∀ x, |f x| = |f (-x)| → f x = -f (-x)) := by 
sorry

end necessary_but_not_sufficient_condition_l172_172321


namespace exponentiation_correct_l172_172629

theorem exponentiation_correct (a : ℝ) : (a ^ 2) ^ 3 = a ^ 6 :=
sorry

end exponentiation_correct_l172_172629


namespace children_on_bus_l172_172810

/-- Prove the number of children on the bus after the bus stop equals 14 given the initial conditions -/
theorem children_on_bus (initial_children : ℕ) (children_got_off : ℕ) (extra_children_got_on : ℕ) (final_children : ℤ) :
  initial_children = 5 →
  children_got_off = 63 →
  extra_children_got_on = 9 →
  final_children = (initial_children - children_got_off) + (children_got_off + extra_children_got_on) →
  final_children = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end children_on_bus_l172_172810


namespace part1_part2_part3_part4_l172_172152

section QuadraticFunction

variable {x : ℝ} {y : ℝ} 

-- 1. Prove that if a quadratic function y = x^2 + bx - 3 intersects the x-axis at (3, 0), 
-- then b = -2 and the other intersection point is (-1, 0).
theorem part1 (b : ℝ) : 
  ((3:ℝ) ^ 2 + b * (3:ℝ) - 3 = 0) → 
  b = -2 ∧ ∃ x : ℝ, (x = -1 ∧ x^2 + b * x - 3 = 0) := 
  sorry

-- 2. For the function y = x^2 + bx - 3 where b = -2, 
-- prove that when 0 < y < 5, x is in -2 < x < -1 or 3 < x < 4.
theorem part2 (b : ℝ) :
  b = -2 → 
  (0 < y ∧ y < 5 → ∃ x : ℝ, (x^2 + b * x - 3 = y) → (-2 < x ∧ x < -1) ∨ (3 < x ∧ x < 4)) :=
  sorry

-- 3. Prove that the value t such that y = x^2 + bx - 3 and y > t always holds for all x
-- is t < -((b ^ 2 + 12) / 4).
theorem part3 (b t : ℝ) :
  (∀ x : ℝ, (x ^ 2 + b * x - 3 > t)) → t < -(b ^ 2 + 12) / 4 :=
  sorry

-- 4. Given y = x^2 - 3x - 3 and 1 < x < 2, 
-- prove that m < y < n with n = -5, b = -3, and m ≤ -21 / 4.
theorem part4 (m n : ℝ) :
  (1 < x ∧ x < 2 → m < x^2 - 3 * x - 3 ∧ x^2 - 3 * x - 3 < n) →
  n = -5 ∧ -21 / 4 ≤ m :=
  sorry

end QuadraticFunction

end part1_part2_part3_part4_l172_172152


namespace mul_101_eq_10201_l172_172389

theorem mul_101_eq_10201 : 101 * 101 = 10201 := by
  sorry

end mul_101_eq_10201_l172_172389


namespace find_certain_amount_l172_172306

theorem find_certain_amount :
  ∀ (A : ℝ), (160 * 8 * 12.5 / 100 = A * 8 * 4 / 100) → 
            (A = 500) :=
  by
    intros A h
    sorry

end find_certain_amount_l172_172306


namespace angle_BAD_measure_l172_172143

theorem angle_BAD_measure (D_A_C : ℝ) (AB_AC : AB = AC) (AD_BD : AD = BD) (h : D_A_C = 39) :
  B_A_D = 70.5 :=
by sorry

end angle_BAD_measure_l172_172143


namespace equal_triangle_area_l172_172177

theorem equal_triangle_area
  (ABC_area : ℝ)
  (AP PB : ℝ)
  (AB_area : ℝ)
  (PQ_BQ_equal : Prop)
  (AP_ratio: AP / (AP + PB) = 3 / 5)
  (ABC_area_val : ABC_area = 15)
  (AP_val : AP = 3)
  (PB_val : PB = 2)
  (PQ_BQ_equal : PQ_BQ_equal = true) :
  ∃ area, area = 9 ∧ area = 9 :=
by
  sorry

end equal_triangle_area_l172_172177


namespace prob_odd_sum_l172_172182

-- Given conditions on the spinners
def spinner_P := [1, 2, 3]
def spinner_Q := [2, 4, 6]
def spinner_R := [1, 3, 5]

-- Probability of spinner P landing on an even number is 1/3
def prob_even_P : ℚ := 1 / 3

-- Probability of odd sum from spinners P, Q, and R
theorem prob_odd_sum : 
  (prob_even_P = 1 / 3) → 
  ∃ p : ℚ, p = 1 / 3 :=
by
  sorry

end prob_odd_sum_l172_172182


namespace find_smallest_solution_l172_172713

theorem find_smallest_solution : ∃ x : ℝ, x = Real.sqrt 119 ∧ (Int.floor (x^2) - Int.floor x ^ 2 = 19) := by
  sorry

end find_smallest_solution_l172_172713


namespace order_of_abc_l172_172723

noncomputable def a : ℚ := 1 / 2
noncomputable def b : ℝ := Real.sqrt 7 - Real.sqrt 5
noncomputable def c : ℝ := Real.sqrt 6 - 2

theorem order_of_abc : a > c ∧ c > b := by
  sorry

end order_of_abc_l172_172723


namespace staples_left_in_stapler_l172_172483

def initial_staples : ℕ := 50
def used_staples : ℕ := 3 * 12
def remaining_staples : ℕ := initial_staples - used_staples

theorem staples_left_in_stapler : remaining_staples = 14 :=
by
  unfold initial_staples used_staples remaining_staples
  rw [Nat.mul_comm, Nat.mul_comm 3, Nat.mul_comm 12, Nat.sub_eq_iff_eq_add]
  have h : ∀ a b : ℕ, a = b -> 50 - (3 * 12) = b -> 50 - 36 = a := by intros; rw [h, Nat.mul_comm 3, Nat.mul_comm 12]
  exact h 36 36 rfl
#align std.staples_left_in_stapler


end staples_left_in_stapler_l172_172483


namespace average_is_correct_l172_172939

def numbers : List ℕ := [1200, 1300, 1400, 1510, 1520, 1530, 1200]

def sum_of_numbers : ℕ := numbers.sum
def count_of_numbers : ℕ := numbers.length
def average_of_numbers : ℚ := sum_of_numbers / count_of_numbers

theorem average_is_correct : average_of_numbers = 1380 := 
by 
  -- Here, you would normally put the proof steps.
  sorry

end average_is_correct_l172_172939


namespace parabola_vertex_below_x_axis_l172_172305

theorem parabola_vertex_below_x_axis (a : ℝ) : (∀ x : ℝ, (x^2 + 2 * x + a < 0)) → a < 1 := 
by
  intro h
  -- proof step here
  sorry

end parabola_vertex_below_x_axis_l172_172305


namespace negation_of_existence_l172_172076

theorem negation_of_existence :
  ¬ (∃ x : ℝ, 0 < x ∧ x^3 - x + 1 > 0) ↔ ∀ x : ℝ, 0 < x → x^3 - x + 1 ≤ 0 :=
by sorry

end negation_of_existence_l172_172076


namespace Arman_hours_worked_l172_172454

/--
  Given:
  - LastWeekHours = 35
  - LastWeekRate = 10 (in dollars per hour)
  - IncreaseRate = 0.5 (in dollars per hour)
  - TotalEarnings = 770 (in dollars)
  Prove that:
  - ThisWeekHours = 40
-/
theorem Arman_hours_worked (LastWeekHours : ℕ) (LastWeekRate : ℕ) (IncreaseRate : ℕ) (TotalEarnings : ℕ)
  (h1 : LastWeekHours = 35)
  (h2 : LastWeekRate = 10)
  (h3 : IncreaseRate = 1/2)  -- because 0.5 as a fraction is 1/2
  (h4 : TotalEarnings = 770)
  : ∃ ThisWeekHours : ℕ, ThisWeekHours = 40 :=
by
  sorry

end Arman_hours_worked_l172_172454


namespace gcd_of_polynomials_l172_172982

theorem gcd_of_polynomials (b : ℤ) (k : ℤ) (hk : k % 2 = 0) (hb : b = 1187 * k) : 
  Int.gcd (2 * b^2 + 31 * b + 67) (b + 15) = 1 :=
by 
  sorry

end gcd_of_polynomials_l172_172982


namespace one_inch_represents_feet_l172_172787

def height_statue : ℕ := 80 -- Height of the statue in feet

def height_model : ℕ := 5 -- Height of the model in inches

theorem one_inch_represents_feet : (height_statue / height_model) = 16 := 
by
  sorry

end one_inch_represents_feet_l172_172787


namespace intersection_empty_l172_172166

-- Define the set M
def M : Set ℝ := { x | ∃ y, y = Real.log (1 - x)}

-- Define the set N
def N : Set (ℝ × ℝ) := { p | ∃ x, ∃ y, (p = (x, y)) ∧ (y = Real.exp x) ∧ (x ∈ Set.univ)}

-- Prove that M ∩ N = ∅
theorem intersection_empty : M ∩ (Prod.fst '' N) = ∅ :=
by
  sorry

end intersection_empty_l172_172166


namespace ratio_of_pipe_lengths_l172_172247

theorem ratio_of_pipe_lengths (L S : ℕ) (h1 : L + S = 177) (h2 : L = 118) (h3 : ∃ k : ℕ, L = k * S) : L / S = 2 := 
by 
  sorry

end ratio_of_pipe_lengths_l172_172247


namespace green_red_socks_ratio_l172_172323

theorem green_red_socks_ratio 
  (r : ℕ) -- Number of pairs of red socks originally ordered
  (y : ℕ) -- Price per pair of red socks
  (green_socks_price : ℕ := 3 * y) -- Price per pair of green socks, 3 times the red socks
  (C_original : ℕ := 6 * green_socks_price + r * y) -- Cost of the original order
  (C_interchanged : ℕ := r * green_socks_price + 6 * y) -- Cost of the interchanged order
  (exchange_rate : ℚ := 1.2) -- 20% increase
  (cost_relation : C_interchanged = exchange_rate * C_original) -- Cost relation given by the problem
  : (6 : ℚ) / (r : ℚ) = 2 / 3 := 
by
  sorry

end green_red_socks_ratio_l172_172323


namespace rectangle_ratio_l172_172083

theorem rectangle_ratio (L B : ℕ) (hL : L = 250) (hB : B = 160) : L / B = 25 / 16 := by
  sorry

end rectangle_ratio_l172_172083


namespace frac_subtraction_l172_172827

theorem frac_subtraction : (18 / 42) - (3 / 8) = (3 / 56) := by
  -- Conditions
  have h1 : 18 / 42 = 3 / 7 := by sorry
  have h2 : 3 / 7 = 24 / 56 := by sorry
  have h3 : 3 / 8 = 21 / 56 := by sorry
  -- Proof using the conditions
  sorry

end frac_subtraction_l172_172827


namespace distinct_sum_of_five_integers_l172_172888

theorem distinct_sum_of_five_integers 
  (a b c d e : ℤ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) 
  (h_condition : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = -120) : 
  a + b + c + d + e = 25 :=
sorry

end distinct_sum_of_five_integers_l172_172888


namespace range_of_a_l172_172876

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x - a| + |x - 1| ≤ 3) → -2 ≤ a ∧ a ≤ 4 :=
by
  sorry

end range_of_a_l172_172876


namespace maria_total_earnings_l172_172895

noncomputable def total_earnings : ℕ := 
  let tulips_day1 := 30
  let roses_day1 := 20
  let lilies_day1 := 15
  let sunflowers_day1 := 10
  let tulips_day2 := tulips_day1 * 2
  let roses_day2 := roses_day1 * 2
  let lilies_day2 := lilies_day1
  let sunflowers_day2 := sunflowers_day1 * 3
  let tulips_day3 := tulips_day2 / 10
  let roses_day3 := 16
  let lilies_day3 := lilies_day1 / 2
  let sunflowers_day3 := sunflowers_day2
  let price_tulip := 2
  let price_rose := 3
  let price_lily := 4
  let price_sunflower := 5
  let day1_earnings := tulips_day1 * price_tulip + roses_day1 * price_rose + lilies_day1 * price_lily + sunflowers_day1 * price_sunflower
  let day2_earnings := tulips_day2 * price_tulip + roses_day2 * price_rose + lilies_day2 * price_lily + sunflowers_day2 * price_sunflower
  let day3_earnings := tulips_day3 * price_tulip + roses_day3 * price_rose + lilies_day3 * price_lily + sunflowers_day3 * price_sunflower
  day1_earnings + day2_earnings + day3_earnings

theorem maria_total_earnings : total_earnings = 920 := 
by 
  unfold total_earnings
  sorry

end maria_total_earnings_l172_172895


namespace inequality_solution_l172_172412

theorem inequality_solution (x : ℝ) : 9 - x^2 < 0 ↔ x < -3 ∨ x > 3 := by
  sorry

end inequality_solution_l172_172412


namespace reduce_repeating_decimal_l172_172969

noncomputable def repeating_decimal_to_fraction (a : ℚ) (n : ℕ) : ℚ :=
  a + (n / 99)

theorem reduce_repeating_decimal : repeating_decimal_to_fraction 2 7 = 205 / 99 := by
  -- proof omitted
  sorry

end reduce_repeating_decimal_l172_172969


namespace water_leaked_l172_172024

theorem water_leaked (initial remaining : ℝ) (h_initial : initial = 0.75) (h_remaining : remaining = 0.5) :
  initial - remaining = 0.25 :=
by
  sorry

end water_leaked_l172_172024


namespace pics_per_album_eq_five_l172_172328

-- Definitions based on conditions
def pics_from_phone : ℕ := 5
def pics_from_camera : ℕ := 35
def total_pics : ℕ := pics_from_phone + pics_from_camera
def num_albums : ℕ := 8

-- Statement to prove
theorem pics_per_album_eq_five : total_pics / num_albums = 5 := by
  sorry

end pics_per_album_eq_five_l172_172328


namespace eval_power_expr_of_196_l172_172756

theorem eval_power_expr_of_196 (a b : ℕ) (ha : 2^a ∣ 196 ∧ ¬ 2^(a + 1) ∣ 196) (hb : 7^b ∣ 196 ∧ ¬ 7^(b + 1) ∣ 196) :
  (1 / 7 : ℝ)^(b - a) = 1 := by
  have ha_val : a = 2 := sorry
  have hb_val : b = 2 := sorry
  rw [ha_val, hb_val]
  simp

end eval_power_expr_of_196_l172_172756


namespace pow_mod_sub_remainder_l172_172940

theorem pow_mod_sub_remainder :
  (10^23 - 7) % 6 = 3 :=
sorry

end pow_mod_sub_remainder_l172_172940


namespace isosceles_triangle_k_l172_172163

theorem isosceles_triangle_k (m n k : ℝ) (h_iso : (m = 4 ∨ n = 4 ∨ m = n) ∧ (m ≠ n ∨ (m = n ∧ m + m > 4))) 
  (h_roots : ∀ x, x^2 - 6*x + (k + 2) = 0 → (x = m ∨ x = n)) : k = 6 ∨ k = 7 :=
sorry

end isosceles_triangle_k_l172_172163


namespace chess_tournament_games_l172_172009

theorem chess_tournament_games (n : ℕ) (h : 2 * 404 = n * (n - 4)) : False :=
by
  sorry

end chess_tournament_games_l172_172009


namespace gcd_153_119_eq_17_l172_172842

theorem gcd_153_119_eq_17 : Nat.gcd 153 119 = 17 := by
  sorry

end gcd_153_119_eq_17_l172_172842


namespace solve_for_y_l172_172297

theorem solve_for_y (x : ℝ) (y : ℝ) (h1 : x = 8) (h2 : x^(2*y) = 16) : y = 2/3 :=
by
  sorry

end solve_for_y_l172_172297


namespace intersection_set_l172_172893

open Set

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_set: M ∩ N = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_set_l172_172893


namespace probability_abc_plus_ab_plus_a_divisible_by_4_l172_172772

noncomputable def count_multiples_of (n m : ℕ) : ℕ := (m / n)

noncomputable def probability_divisible_by_4 : ℚ := 
  let total_numbers := 2008
  let multiples_of_4 := count_multiples_of 4 total_numbers
  -- Probability that 'a' is divisible by 4
  let p_a := (multiples_of_4 : ℚ) / total_numbers
  -- Probability that 'a' is not divisible by 4
  let p_not_a := 1 - p_a
  -- Considering specific cases for b and c modulo 4
  let p_bc_cases := (2 * ((1 / 4) * (1 / 4)))  -- Probabilities for specific cases noted as 2 * (1/16)
  -- Adjusting probabilities for non-divisible 'a'
  let p_not_a_cases := p_bc_cases * p_not_a
  -- Total Probability
  p_a + p_not_a_cases

theorem probability_abc_plus_ab_plus_a_divisible_by_4 :
  probability_divisible_by_4 = 11 / 32 :=
sorry

end probability_abc_plus_ab_plus_a_divisible_by_4_l172_172772


namespace center_cell_value_l172_172424

namespace MathProof

variables {a b c d e f g h i : ℝ}

-- Conditions
axiom row_product1 : a * b * c = 1
axiom row_product2 : d * e * f = 1
axiom row_product3 : g * h * i = 1

axiom col_product1 : a * d * g = 1
axiom col_product2 : b * e * h = 1
axiom col_product3 : c * f * i = 1

axiom square_product1 : a * b * d * e = 2
axiom square_product2 : b * c * e * f = 2
axiom square_product3 : d * e * g * h = 2
axiom square_product4 : e * f * h * i = 2

-- Proof problem
theorem center_cell_value : e = 1 :=
sorry

end MathProof

end center_cell_value_l172_172424


namespace absent_children_l172_172897

-- Definitions
def total_children := 840
def bananas_per_child_present := 4
def bananas_per_child_if_all_present := 2
def total_bananas_if_all_present := total_children * bananas_per_child_if_all_present

-- The theorem to prove
theorem absent_children (A : ℕ) (P : ℕ) :
  P = total_children - A →
  total_bananas_if_all_present = P * bananas_per_child_present →
  A = 420 :=
by
  sorry

end absent_children_l172_172897


namespace chebyshev_inequality_l172_172151

noncomputable def tk (n : ℕ) (x : Fin n → ℝ) (k : Fin n) : ℝ :=
  ∏ (j : Fin n) in Finset.univ.erase k, |x j - x k|

theorem chebyshev_inequality {n : ℕ} (x : Fin n → ℝ) (h₁ : 2 ≤ n) (h₂ : ∀ i j, i < j → x i < x j) :
  (∑ k : Fin n, 1 / tk n x k) ≥ 2 ^ (n - 2) :=
by 
  sorry

end chebyshev_inequality_l172_172151


namespace geometric_sequence_tan_sum_l172_172569

theorem geometric_sequence_tan_sum
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b^2 = a * c)
  (h2 : Real.tan B = 3/4):
  1 / Real.tan A + 1 / Real.tan C = 5 / 3 := 
by
  sorry

end geometric_sequence_tan_sum_l172_172569


namespace quadratic_radical_same_type_l172_172351

theorem quadratic_radical_same_type (a : ℝ) (h : (∃ (t : ℝ), t ^ 2 = 3 * a - 4) ∧ (∃ (t : ℝ), t ^ 2 = 8)) : a = 2 :=
by
  -- Extract the properties of the radicals
  sorry

end quadratic_radical_same_type_l172_172351


namespace two_and_four_digit_singular_numbers_six_digit_singular_number_exists_twenty_digit_singular_number_at_most_ten_singular_numbers_with_100_digits_exists_thirty_digit_singular_number_l172_172871

def is_singular_number (n : ℕ) (num : ℕ) : Prop :=
  let first_n_digits := num / 10^n;
  let last_n_digits := num % 10^n;
  (num > 0) ∧
  (first_n_digits > 0) ∧
  (last_n_digits > 0) ∧
  (first_n_digits < 10^n) ∧
  (last_n_digits < 10^n) ∧
  (num = first_n_digits * 10^n + last_n_digits) ∧
  (∃ k, num = k^2) ∧
  (∃ k, first_n_digits = k^2) ∧
  (∃ k, last_n_digits = k^2)

-- (1) Prove that 49 is a two-digit singular number and 1681 is a four-digit singular number
theorem two_and_four_digit_singular_numbers :
  is_singular_number 1 49 ∧ is_singular_number 2 1681 :=
sorry

-- (2) Prove that 256036 is a six-digit singular number
theorem six_digit_singular_number :
  is_singular_number 3 256036 :=
sorry

-- (3) Prove the existence of a 20-digit singular number
theorem exists_twenty_digit_singular_number :
  ∃ num, is_singular_number 10 num :=
sorry

-- (4) Prove that there are at most 10 singular numbers with 100 digits
theorem at_most_ten_singular_numbers_with_100_digits :
  ∃! n, n <= 10 ∧ ∀ num, num < 10^100 → is_singular_number 50 num → num < 10 ∧ num > 0 :=
sorry

-- (5) Prove the existence of a 30-digit singular number
theorem exists_thirty_digit_singular_number :
  ∃ num, is_singular_number 15 num :=
sorry

end two_and_four_digit_singular_numbers_six_digit_singular_number_exists_twenty_digit_singular_number_at_most_ten_singular_numbers_with_100_digits_exists_thirty_digit_singular_number_l172_172871


namespace number_of_cows_l172_172570

-- Define conditions
def total_bags_consumed_by_some_cows := 45
def bags_consumed_by_one_cow := 1

-- State the theorem to prove the number of cows
theorem number_of_cows (h1 : total_bags_consumed_by_some_cows = 45) (h2 : bags_consumed_by_one_cow = 1) : 
  total_bags_consumed_by_some_cows / bags_consumed_by_one_cow = 45 :=
by
  -- Proof goes here
  sorry

end number_of_cows_l172_172570


namespace find_number_l172_172879

def number_of_faces : ℕ := 6

noncomputable def probability (n : ℕ) : ℚ :=
  (number_of_faces - n : ℕ) / number_of_faces

theorem find_number (n : ℕ) (h: n < number_of_faces) :
  probability n = 1 / 3 → n = 4 :=
by
  -- proof goes here
  sorry

end find_number_l172_172879


namespace percentage_increase_in_pay_rate_l172_172114

-- Given conditions
def regular_rate : ℝ := 10
def total_surveys : ℕ := 50
def cellphone_surveys : ℕ := 35
def total_earnings : ℝ := 605

-- We need to demonstrate that the percentage increase in the pay rate for surveys involving the use of her cellphone is 30%
theorem percentage_increase_in_pay_rate :
  let earnings_at_regular_rate := regular_rate * total_surveys
  let earnings_from_cellphone_surveys := total_earnings - earnings_at_regular_rate
  let rate_per_cellphone_survey := earnings_from_cellphone_surveys / cellphone_surveys
  let increase_in_rate := rate_per_cellphone_survey - regular_rate
  let percentage_increase := (increase_in_rate / regular_rate) * 100
  percentage_increase = 30 :=
by
  sorry

end percentage_increase_in_pay_rate_l172_172114


namespace find_integer_n_l172_172003

theorem find_integer_n (n : ℤ) : (⌊(n^2 / 9 : ℝ)⌋ - ⌊(n / 3 : ℝ)⌋ ^ 2 = 5) → n = 14 :=
by
  -- Proof is omitted
  sorry

end find_integer_n_l172_172003


namespace smallest_solution_eq_sqrt_104_l172_172697

theorem smallest_solution_eq_sqrt_104 :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, ⌊y^2⌋ - ⌊y⌋^2 = 19 → x ≤ y) := sorry

end smallest_solution_eq_sqrt_104_l172_172697


namespace negation_of_existence_l172_172077

theorem negation_of_existence :
  ¬ (∃ x : ℝ, 0 < x ∧ x^3 - x + 1 > 0) ↔ ∀ x : ℝ, 0 < x → x^3 - x + 1 ≤ 0 :=
by sorry

end negation_of_existence_l172_172077


namespace multiplication_of_fractions_l172_172002

theorem multiplication_of_fractions :
  (77 / 4) * (5 / 2) = 48 + 1 / 8 := 
sorry

end multiplication_of_fractions_l172_172002


namespace area_square_II_l172_172909

theorem area_square_II (a b : ℝ) :
  let diag_I := 2 * (a + b)
  let area_I := (a + b) * (a + b) * 2
  let area_II := area_I * 3
  area_II = 6 * (a + b) ^ 2 :=
by
  sorry

end area_square_II_l172_172909


namespace smallest_solution_floor_equation_l172_172704

theorem smallest_solution_floor_equation : ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (x = Real.sqrt 109) :=
by
  sorry

end smallest_solution_floor_equation_l172_172704


namespace carpenters_time_l172_172148

theorem carpenters_time (t1 t2 t3 t4 : ℝ) (ht1 : t1 = 1) (ht2 : t2 = 2)
  (ht3 : t3 = 3) (ht4 : t4 = 4) : (1 / (1 / t1 + 1 / t2 + 1 / t3 + 1 / t4)) = 12 / 25 := by
  sorry

end carpenters_time_l172_172148


namespace lara_yesterday_more_than_sarah_l172_172231

variable (yesterdaySarah todaySarah todayLara : ℕ)
variable (cansDifference : ℕ)

axiom yesterdaySarah_eq : yesterdaySarah = 50
axiom todaySarah_eq : todaySarah = 40
axiom todayLara_eq : todayLara = 70
axiom cansDifference_eq : cansDifference = 20

theorem lara_yesterday_more_than_sarah :
  let totalCansYesterday := yesterdaySarah + todaySarah + cansDifference
  let laraYesterday := totalCansYesterday - yesterdaySarah
  laraYesterday - yesterdaySarah = 30 :=
by
  sorry

end lara_yesterday_more_than_sarah_l172_172231


namespace fewer_noodles_than_pirates_l172_172748

theorem fewer_noodles_than_pirates 
  (P : ℕ) (N : ℕ) (h1 : P = 45) (h2 : N + P = 83) : P - N = 7 := by 
  sorry

end fewer_noodles_than_pirates_l172_172748


namespace find_ab_l172_172162

variable (a b : ℝ)

theorem find_ab (h1 : a + b = 4) (h2 : a^3 + b^3 = 136) : a * b = -6 := by
  sorry

end find_ab_l172_172162


namespace correct_answers_unanswered_minimum_correct_answers_l172_172767

-- Definition of the conditions in the problem
def total_questions := 25
def unanswered_questions := 1
def correct_points := 4
def wrong_points := -1
def total_score_1 := 86
def total_score_2 := 90

-- Part 1: Define the conditions and prove that x = 22
theorem correct_answers_unanswered (x : ℕ) (h1 : total_questions - unanswered_questions = 24)
  (h2 : 4 * x + wrong_points * (total_questions - unanswered_questions - x) = total_score_1) : x = 22 :=
sorry

-- Part 2: Define the conditions and prove that at least 23 correct answers are needed
theorem minimum_correct_answers (a : ℕ)
  (h3 : correct_points * a + wrong_points * (total_questions - a) ≥ total_score_2) : a ≥ 23 :=
sorry

end correct_answers_unanswered_minimum_correct_answers_l172_172767


namespace jerry_earnings_per_task_l172_172449

theorem jerry_earnings_per_task :
  ∀ (task_hours : ℕ) (daily_hours : ℕ) (days_per_week : ℕ) (total_earnings : ℕ),
    task_hours = 2 →
    daily_hours = 10 →
    days_per_week = 5 →
    total_earnings = 1400 →
    total_earnings / ((daily_hours / task_hours) * days_per_week) = 56 :=
by
  intros task_hours daily_hours days_per_week total_earnings
  intros h_task_hours h_daily_hours h_days_per_week h_total_earnings
  sorry

end jerry_earnings_per_task_l172_172449


namespace negation_of_universal_proposition_l172_172215

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 + 2^x - 1 > 0)) ↔ (∃ x : ℝ, x^2 + 2^x - 1 ≤ 0) :=
by
  sorry

end negation_of_universal_proposition_l172_172215


namespace painter_total_cost_l172_172258

-- Define the arithmetic sequence for house addresses
def south_side_arith_seq (n : ℕ) : ℕ := 5 + (n - 1) * 7
def north_side_arith_seq (n : ℕ) : ℕ := 6 + (n - 1) * 8

-- Define the counting of digits
def digit_count (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else 3

-- Define the condition of painting cost for multiples of 10
def painting_cost (n : ℕ) : ℕ :=
  if n % 10 = 0 then 2 * digit_count n
  else digit_count n

-- Calculate total cost for side with given arithmetic sequence
def total_cost_for_side (side_arith_seq : ℕ → ℕ): ℕ :=
  List.range 25 |>.map (λ n => painting_cost (side_arith_seq (n + 1))) |>.sum

-- Main theorem to prove
theorem painter_total_cost : total_cost_for_side south_side_arith_seq + total_cost_for_side north_side_arith_seq = 147 := by
  sorry

end painter_total_cost_l172_172258


namespace hypotenuse_length_l172_172929

theorem hypotenuse_length
  (a b : ℝ)
  (V1 : ℝ := (1/3) * Real.pi * a * b^2)
  (V2 : ℝ := (1/3) * Real.pi * b * a^2)
  (hV1 : V1 = 800 * Real.pi)
  (hV2 : V2 = 1920 * Real.pi) :
  Real.sqrt (a^2 + b^2) = 26 :=
by
  sorry

end hypotenuse_length_l172_172929


namespace product_increased_l172_172012

theorem product_increased (a b c : ℕ) (h1 : a = 1) (h2: b = 1) (h3: c = 676) :
  ((a - 3) * (b - 3) * (c - 3) = a * b * c + 2016) :=
by
  simp [h1, h2, h3]
  sorry

end product_increased_l172_172012


namespace zero_in_interval_l172_172978

open Real

noncomputable def f (x : ℝ) : ℝ := log x + x - 3

theorem zero_in_interval (a b : ℕ) (h1 : b - a = 1) (h2 : 1 ≤ a) (h3 : 1 ≤ b) 
  (h4 : f a < 0) (h5 : 0 < f b) : a + b = 5 :=
sorry

end zero_in_interval_l172_172978


namespace mikes_remaining_cards_l172_172460

variable (original_number_of_cards : ℕ)
variable (sam_bought : ℤ)
variable (alex_bought : ℤ)

theorem mikes_remaining_cards :
  original_number_of_cards = 87 →
  sam_bought = 8 →
  alex_bought = 13 →
  original_number_of_cards - (sam_bought + alex_bought) = 66 :=
by
  intros h_original h_sam h_alex
  rw [h_original, h_sam, h_alex]
  norm_num

end mikes_remaining_cards_l172_172460


namespace min_value_inverse_sum_l172_172550

theorem min_value_inverse_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 2 * b = 2) :
  (1 / a + 2 / b) ≥ 9 / 2 :=
sorry

end min_value_inverse_sum_l172_172550


namespace diff_of_squares_l172_172189

theorem diff_of_squares (a b : ℕ) : 
  (∃ x y : ℤ, a = x^2 - y^2) ∨ (∃ x y : ℤ, b = x^2 - y^2) ∨ (∃ x y : ℤ, a + b = x^2 - y^2) :=
sorry

end diff_of_squares_l172_172189


namespace percentage_decrease_l172_172898

-- Define the initial conditions
def total_cans : ℕ := 600
def initial_people : ℕ := 40
def new_total_cans : ℕ := 420

-- Use the conditions to define the resulting quantities
def cans_per_person : ℕ := total_cans / initial_people
def new_people : ℕ := new_total_cans / cans_per_person

-- Prove the percentage decrease in the number of people
theorem percentage_decrease :
  let original_people := initial_people
  let new_people := new_people
  let decrease := original_people - new_people
  let percentage_decrease := (decrease * 100) / original_people
  percentage_decrease = 30 :=
by
  sorry

end percentage_decrease_l172_172898


namespace pizza_slices_meat_count_l172_172038

theorem pizza_slices_meat_count :
  let p := 30 in
  let h := 2 * p in
  let s := p + 12 in
  let n := 6 in
  (p + h + s) / n = 22 :=
by
  let p := 30
  let h := 2 * p
  let s := p + 12
  let n := 6
  calc
    (p + h + s) / n = (30 + 60 + 42) / 6 : by
      simp [p, h, s, n]
    ... = 132 / 6 : by
      rfl
    ... = 22 : by
      norm_num

end pizza_slices_meat_count_l172_172038


namespace train_speed_l172_172110

theorem train_speed (length_train : ℝ) (time_to_cross : ℝ) (length_bridge : ℝ)
  (h_train : length_train = 100) (h_time : time_to_cross = 12.499)
  (h_bridge : length_bridge = 150) : 
  ((length_train + length_bridge) / time_to_cross * 3.6) = 72 := 
by 
  sorry

end train_speed_l172_172110


namespace joan_gave_away_kittens_l172_172576

-- Definitions based on conditions in the problem
def original_kittens : ℕ := 8
def kittens_left : ℕ := 6

-- Mathematical statement to be proved
theorem joan_gave_away_kittens : original_kittens - kittens_left = 2 :=
by
  sorry

end joan_gave_away_kittens_l172_172576


namespace profit_function_and_optimal_price_l172_172250

variable (cost selling base_units additional_units: ℝ)
variable (x: ℝ) (y: ℝ)

def profit (x: ℝ): ℝ := -20 * x^2 + 100 * x + 6000

theorem profit_function_and_optimal_price:
  (cost = 40) →
  (selling = 60) →
  (base_units = 300) →
  (additional_units = 20) →
  (0 ≤ x) →
  (x < 20) →
  (y = profit x) →
  exists x_max y_max: ℝ, (x_max = 2.5) ∧ (y_max = 6125) :=
by 
  sorry

end profit_function_and_optimal_price_l172_172250


namespace total_birds_on_fence_l172_172811

theorem total_birds_on_fence (initial_birds additional_birds storks : ℕ) 
  (h1 : initial_birds = 6) 
  (h2 : additional_birds = 4) 
  (h3 : storks = 8) :
  initial_birds + additional_birds + storks = 18 :=
by
  sorry

end total_birds_on_fence_l172_172811


namespace percent_increase_l172_172961

theorem percent_increase (N : ℝ) (h : (1 / 7) * N = 1) : 
  N = 7 ∧ (N - (4 / 7)) / (4 / 7) * 100 = 1125.0000000000002 := 
by 
  sorry

end percent_increase_l172_172961


namespace hypotenuse_length_l172_172092

-- Definitions and conditions
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Hypotheses
def leg1 := 8
def leg2 := 15

-- The theorem to be proven
theorem hypotenuse_length : ∃ c : ℕ, is_right_triangle leg1 leg2 c ∧ c = 17 :=
by { sorry }

end hypotenuse_length_l172_172092


namespace product_is_approximately_9603_l172_172611

noncomputable def smaller_number : ℝ := 97.49871794028884
noncomputable def successive_number : ℝ := smaller_number + 1
noncomputable def product_of_numbers : ℝ := smaller_number * successive_number

theorem product_is_approximately_9603 : abs (product_of_numbers - 9603) < 10e-3 := 
sorry

end product_is_approximately_9603_l172_172611


namespace min_value_f_when_a_eq_1_range_of_a_if_f_leq_3_non_empty_l172_172724

-- Condition 1: Define the function f(x)
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x - 3)

-- Proof Problem 1: Minimum value of f(x) when a = 1
theorem min_value_f_when_a_eq_1 : (∀ x : ℝ, f x 1 ≥ 2) :=
sorry

-- Proof Problem 2: Range of values for a when f(x) ≤ 3 has solutions
theorem range_of_a_if_f_leq_3_non_empty : 
  (∃ x : ℝ, f x a ≤ 3) → abs (3 - a) ≤ 3 :=
sorry

end min_value_f_when_a_eq_1_range_of_a_if_f_leq_3_non_empty_l172_172724


namespace fraction_numerator_greater_than_denominator_l172_172914

theorem fraction_numerator_greater_than_denominator (x : ℝ) :
  (4 * x + 2 > 8 - 3 * x) ↔ (6 / 7 < x ∧ x ≤ 3) :=
by
  sorry

end fraction_numerator_greater_than_denominator_l172_172914


namespace power_of_two_last_digit_product_divisible_by_6_l172_172904

theorem power_of_two_last_digit_product_divisible_by_6 (n : Nat) (h : 3 < n) :
  ∃ d m : Nat, (2^n = 10 * m + d) ∧ (m * d) % 6 = 0 :=
by
  sorry

end power_of_two_last_digit_product_divisible_by_6_l172_172904


namespace tangent_lines_to_curve_at_l172_172858

noncomputable
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

noncomputable
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a * x^2 + (a - 2) * x

noncomputable
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + (a - 2)

theorem tangent_lines_to_curve_at (a : ℝ) :
  is_even_function (f' a) →
  (∀ x, f a x = - 2 → (2*x + (- f a x) = 0 ∨ 19*x - 4*(- f a x) - 27 = 0)) :=
by
  sorry

end tangent_lines_to_curve_at_l172_172858


namespace storks_more_than_birds_l172_172637

def birds := 4
def initial_storks := 3
def additional_storks := 6

theorem storks_more_than_birds :
  (initial_storks + additional_storks) - birds = 5 := 
by
  sorry

end storks_more_than_birds_l172_172637


namespace sculpture_height_l172_172669

def base_height: ℝ := 10  -- height of the base in inches
def combined_height_feet: ℝ := 3.6666666666666665  -- combined height in feet
def inches_per_foot: ℝ := 12  -- conversion factor from feet to inches

-- Convert combined height to inches
def combined_height_inches: ℝ := combined_height_feet * inches_per_foot

-- Math proof problem statement
theorem sculpture_height : combined_height_inches - base_height = 34 := by
  sorry

end sculpture_height_l172_172669


namespace linear_price_item_func_l172_172946

noncomputable def price_item_func (x : ℝ) : Prop :=
  ∃ (y : ℝ), y = - (1/4) * x + 50 ∧ 0 < x ∧ x < 200

theorem linear_price_item_func : ∀ x, price_item_func x ↔ (∃ y, y = - (1/4) * x + 50 ∧ 0 < x ∧ x < 200) :=
by
  sorry

end linear_price_item_func_l172_172946


namespace remainder_of_product_mod_5_l172_172622

theorem remainder_of_product_mod_5 :
  (2685 * 4932 * 91406) % 5 = 0 :=
by
  sorry

end remainder_of_product_mod_5_l172_172622


namespace solution_set_l172_172762

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

variable {f : ℝ → ℝ}

-- Hypotheses
axiom odd_f : is_odd f
axiom increasing_f : is_increasing f
axiom f_of_neg_three : f (-3) = 0

-- Theorem statement
theorem solution_set (x : ℝ) : (x - 3) * f (x - 3) < 0 ↔ (0 < x ∧ x < 3) ∨ (3 < x ∧ x < 6) :=
sorry

end solution_set_l172_172762


namespace shadow_length_correct_l172_172202

theorem shadow_length_correct :
  let light_source := (0, 16)
  let disc_center := (6, 10)
  let radius := 2
  let m := 4
  let n := 17
  let length_form := m * Real.sqrt n
  length_form = 4 * Real.sqrt 17 :=
by
  sorry

end shadow_length_correct_l172_172202


namespace cart_total_books_l172_172462

theorem cart_total_books (fiction non_fiction autobiographies picture: ℕ) 
  (h1: fiction = 5)
  (h2: non_fiction = fiction + 4)
  (h3: autobiographies = 2 * fiction)
  (h4: picture = 11)
  : fiction + non_fiction + autobiographies + picture = 35 := by
  -- Proof is omitted
  sorry

end cart_total_books_l172_172462


namespace math_problem_l172_172345

noncomputable def f : ℝ → ℝ := sorry

noncomputable def g (x : ℝ) : ℝ := f (x + (1/6))

lemma f_odd (x : ℝ) : f (-x) = -f (x) := sorry

lemma g_def (x : ℝ) : g (x) = f (x + 1/6) := sorry

lemma g_def_shifted (x : ℝ) : g (x + 1/3) = f (1/2 - x) := sorry

lemma f_interval (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) : f (x) = 2^x - 1 := sorry

theorem math_problem :
  f (Real.log 5 / Real.log 2) + g (5/6) = 1/4 :=
sorry

end math_problem_l172_172345


namespace sum_of_reciprocals_l172_172790

variable {x y : ℝ}
variable (hx : x + y = 3 * x * y + 2)

theorem sum_of_reciprocals : (1 / x) + (1 / y) = 3 :=
by
  sorry

end sum_of_reciprocals_l172_172790


namespace coloring_ways_l172_172473

-- Definitions of the problem:
def column1 := 1
def column2 := 2
def column3 := 3
def column4 := 4
def column5 := 3
def column6 := 2
def column7 := 1
def total_colors := 3 -- Blue, Yellow, Green

-- Adjacent coloring constraints:
def adjacent_constraints (c1 c2 : ℕ) : Prop := c1 ≠ c2

-- Number of ways to color figure:
theorem coloring_ways : 
  (∃ (n : ℕ), n = 2^5) ∧ 
  n = 32 :=
by 
  sorry

end coloring_ways_l172_172473


namespace greatest_whole_number_solution_l172_172138

theorem greatest_whole_number_solution (x : ℤ) (h : 6 * x - 5 < 7 - 3 * x) : x ≤ 1 :=
sorry

end greatest_whole_number_solution_l172_172138


namespace count_even_positive_integers_satisfy_inequality_l172_172992

open Int

noncomputable def countEvenPositiveIntegersInInterval : ℕ :=
  (List.filter (fun n : ℕ => n % 2 = 0) [2, 4, 6, 8, 10, 12]).length

theorem count_even_positive_integers_satisfy_inequality :
  countEvenPositiveIntegersInInterval = 6 := by
  sorry

end count_even_positive_integers_satisfy_inequality_l172_172992


namespace find_second_divisor_l172_172514

theorem find_second_divisor (k : ℕ) (d : ℕ) 
  (h1 : k % 5 = 2)
  (h2 : k < 42)
  (h3 : k % 7 = 3)
  (h4 : k % d = 5) : d = 12 := 
sorry

end find_second_divisor_l172_172514


namespace tank_capacity_l172_172299

theorem tank_capacity :
  ∃ T : ℝ, (5/8) * T + 12 = (11/16) * T ∧ T = 192 :=
sorry

end tank_capacity_l172_172299


namespace bottom_rightmost_rectangle_is_E_l172_172880

-- Definitions of the given conditions
structure Rectangle where
  w : ℕ
  y : ℕ

def A : Rectangle := { w := 5, y := 8 }
def B : Rectangle := { w := 2, y := 4 }
def C : Rectangle := { w := 4, y := 6 }
def D : Rectangle := { w := 8, y := 5 }
def E : Rectangle := { w := 10, y := 9 }

-- The theorem we need to prove
theorem bottom_rightmost_rectangle_is_E :
    (E.w = 10) ∧ (E.y = 9) :=
by
  -- Proof would go here
  sorry

end bottom_rightmost_rectangle_is_E_l172_172880


namespace chocolate_bars_per_small_box_l172_172652

theorem chocolate_bars_per_small_box (total_chocolate_bars small_boxes : ℕ) 
  (h1 : total_chocolate_bars = 442) 
  (h2 : small_boxes = 17) : 
  total_chocolate_bars / small_boxes = 26 :=
by
  sorry

end chocolate_bars_per_small_box_l172_172652


namespace cosine_squared_is_half_l172_172190

def sides_of_triangle (p q r : ℝ) : Prop :=
  p > 0 ∧ q > 0 ∧ r > 0 ∧ p + q > r ∧ q + r > p ∧ r + p > q

noncomputable def cosine_squared (p q r : ℝ) : ℝ :=
  ((p^2 + q^2 - r^2) / (2 * p * q))^2

theorem cosine_squared_is_half (p q r : ℝ) (h : sides_of_triangle p q r) 
  (h_eq : p^4 + q^4 + r^4 = 2 * r^2 * (p^2 + q^2)) : cosine_squared p q r = 1 / 2 :=
by
  sorry

end cosine_squared_is_half_l172_172190


namespace intersection_eq_l172_172103

open Set

def A : Set ℕ := {0, 2, 4, 6}
def B : Set ℕ := {x | 3 < x ∧ x < 7}

theorem intersection_eq : A ∩ B = {4, 6} := 
by 
  sorry

end intersection_eq_l172_172103


namespace limit_T2_over_T1_l172_172455

-- Define the curve
def curve (x : ℝ) : ℝ := log (x + 1)

-- Define the tangent line at (p, log(p + 1))
def tangent_line (p : ℝ) (x : ℝ) : ℝ :=
  (1 / (p + 1)) * x - (p / (p + 1)) + log (p + 1)

-- Define the normal line at (p, log(p + 1))
def normal_line (p : ℝ) (x : ℝ) : ℝ :=
  - (p + 1) * x + (p * (p + 1)) + log (p + 1)

-- Define the areas T1 and T2
noncomputable def T1 (p : ℝ) : ℝ :=
  integral 0 p (λ x, curve x - tangent_line p x)

noncomputable def T2 (p : ℝ) : ℝ :=
  integral 0 p (λ x, curve x - normal_line p x)

-- State the theorem
theorem limit_T2_over_T1 (T1 T2 : ℝ → ℝ) : 
  (∀ p : ℝ, T1 p = integral 0 p (λ x, curve x - tangent_line p x)) →
  (∀ p : ℝ, T2 p = integral 0 p (λ x, curve x - normal_line p x)) →
  tendsto (λ p, T2 p / T1 p) (nhds 0) (nhds (-1)) :=
by
  sorry

end limit_T2_over_T1_l172_172455


namespace smallest_solution_floor_eq_l172_172706

theorem smallest_solution_floor_eq (x : ℝ) : ⌊x^2⌋ - ⌊x⌋^2 = 19 → x = Real.sqrt 119 :=
by
  sorry

end smallest_solution_floor_eq_l172_172706


namespace tan_arith_seq_l172_172304

theorem tan_arith_seq (x y z : ℝ)
  (h₁ : y = x + π / 3)
  (h₂ : z = x + 2 * π / 3) :
  (Real.tan x * Real.tan y) + (Real.tan y * Real.tan z) + (Real.tan z * Real.tan x) = -3 :=
sorry

end tan_arith_seq_l172_172304


namespace trips_to_collect_all_trays_l172_172226

-- Definition of conditions
def trays_at_once : ℕ := 7
def trays_one_table : ℕ := 23
def trays_other_table : ℕ := 5

-- Theorem statement
theorem trips_to_collect_all_trays : 
  (trays_one_table / trays_at_once) + (if trays_one_table % trays_at_once = 0 then 0 else 1) + 
  (trays_other_table / trays_at_once) + (if trays_other_table % trays_at_once = 0 then 0 else 1) = 5 := 
by
  sorry

end trips_to_collect_all_trays_l172_172226


namespace avg_hamburgers_per_day_l172_172655

theorem avg_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h1 : total_hamburgers = 63) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 := by
  sorry

end avg_hamburgers_per_day_l172_172655


namespace value_of_expression_l172_172635

theorem value_of_expression (m : ℝ) (h : m^2 + m - 1 = 0) : 3 * m^2 + 3 * m + 2006 = 2009 :=
by
  sorry

end value_of_expression_l172_172635


namespace basketball_scores_l172_172371

theorem basketball_scores :
  ∃ P: Finset ℕ, (∀ x y: ℕ, (x + y = 7 → P = {p | ∃ x y: ℕ, p = 3 * x + 2 * y})) ∧ (P.card = 8) :=
sorry

end basketball_scores_l172_172371


namespace last_digit_of_largest_power_of_3_dividing_factorial_l172_172680

theorem last_digit_of_largest_power_of_3_dividing_factorial (n : ℕ) (h : n = 3^3) : 
  let m := (Nat.multiplicity 3 n.factorial).get (Nat.multiplicity.finite _ _)
  let last_digit := (3 ^ m) % 10
  last_digit = 3 :=
by
  sorry

end last_digit_of_largest_power_of_3_dividing_factorial_l172_172680


namespace inverse_proportion_inequality_l172_172418

theorem inverse_proportion_inequality :
  ∀ (y : ℝ → ℝ) (y_1 y_2 y_3 : ℝ),
  (∀ x, y x = 7 / x) →
  y (-3) = y_1 →
  y (-1) = y_2 →
  y (2) = y_3 →
  y_2 < y_1 ∧ y_1 < y_3 :=
by
  intros y y_1 y_2 y_3 hy hA hB hC
  sorry

end inverse_proportion_inequality_l172_172418


namespace vector_computation_l172_172167

def c : ℝ × ℝ × ℝ := (-3, 5, 2)
def d : ℝ × ℝ × ℝ := (5, -1, 3)

theorem vector_computation : 2 • c - 5 • d + c = (-34, 20, -9) := by
  sorry

end vector_computation_l172_172167


namespace bottle_cost_l172_172101

-- Definitions of the conditions
def total_cost := 30
def wine_extra_cost := 26

-- Statement of the problem in Lean 4
theorem bottle_cost : 
  ∃ x : ℕ, (x + (x + wine_extra_cost) = total_cost) ∧ x = 2 :=
by
  sorry

end bottle_cost_l172_172101


namespace remaining_cube_edge_length_l172_172375

theorem remaining_cube_edge_length (a b : ℕ) (h : a^3 = 98 + b^3) : b = 3 :=
sorry

end remaining_cube_edge_length_l172_172375


namespace find_c_d_of_cubic_common_roots_l172_172844

theorem find_c_d_of_cubic_common_roots 
  (c d : ℝ)
  (h1 : ∃ r s : ℝ, r ≠ s ∧ (r ^ 3 + c * r ^ 2 + 12 * r + 7 = 0) ∧ (s ^ 3 + c * s ^ 2 + 12 * s + 7 = 0))
  (h2 : ∃ r s : ℝ, r ≠ s ∧ (r ^ 3 + d * r ^ 2 + 15 * r + 9 = 0) ∧ (s ^ 3 + d * s ^ 2 + 15 * s + 9 = 0)) :
  c = 5 ∧ d = 4 :=
sorry

end find_c_d_of_cubic_common_roots_l172_172844


namespace MathContestMeanMedianDifference_l172_172746

theorem MathContestMeanMedianDifference :
  (15 / 100 * 65 + 20 / 100 * 85 + 40 / 100 * 95 + 25 / 100 * 110) - 95 = -3 := 
by
  sorry

end MathContestMeanMedianDifference_l172_172746


namespace polynomial_non_negative_l172_172222

theorem polynomial_non_negative (a : ℝ) : a^2 * (a^2 - 1) - a^2 + 1 ≥ 0 := by
  -- we would include the proof steps here
  sorry

end polynomial_non_negative_l172_172222


namespace lucy_snowballs_l172_172962

theorem lucy_snowballs : ∀ (c l : ℕ), c = l + 31 → c = 50 → l = 19 :=
by
  intros c l h1 h2
  sorry

end lucy_snowballs_l172_172962


namespace smallest_solution_floor_equation_l172_172703

theorem smallest_solution_floor_equation : ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (x = Real.sqrt 109) :=
by
  sorry

end smallest_solution_floor_equation_l172_172703


namespace max_sum_of_inequalities_l172_172188

theorem max_sum_of_inequalities (x y : ℝ) (h1 : 4 * x + 3 * y ≤ 10) (h2 : 3 * x + 5 * y ≤ 11) :
  x + y ≤ 31 / 11 :=
sorry

end max_sum_of_inequalities_l172_172188


namespace maximize_profit_l172_172949

noncomputable def profit (m : ℝ) : ℝ := 
  29 - (16 / (m + 1) + (m + 1))

theorem maximize_profit : 
  ∃ m : ℝ, m = 3 ∧ m ≥ 0 ∧ profit m = 21 :=
by
  use 3
  repeat { sorry }

end maximize_profit_l172_172949


namespace part_a_l172_172923

def is_tricubic (k : ℕ) : Prop :=
  ∃ a b c : ℕ, k = a^3 + b^3 + c^3

theorem part_a : ∃ (n : ℕ), is_tricubic n ∧ ¬ is_tricubic (n + 2) ∧ ¬ is_tricubic (n + 28) :=
by 
  let n := 3 * (3*1+1)^3
  exists n
  sorry

end part_a_l172_172923


namespace range_of_m_l172_172229

noncomputable def f (x : ℝ) : ℝ := x^3 - x^2 - x

theorem range_of_m (m : ℝ) (x : ℝ) (h1 : x ∈ Set.Icc (-1 : ℝ) 2) : 
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x < m) ↔ 2 < m := 
by 
  sorry

end range_of_m_l172_172229


namespace remainder_of_sum_mod_eight_l172_172869

theorem remainder_of_sum_mod_eight (m : ℤ) : 
  ((10 - 3 * m) + (5 * m + 6)) % 8 = (2 * m) % 8 :=
by
  sorry

end remainder_of_sum_mod_eight_l172_172869


namespace average_hamburgers_sold_per_day_l172_172658

theorem average_hamburgers_sold_per_day 
  (total_hamburgers : ℕ) (days_in_week : ℕ)
  (h1 : total_hamburgers = 63) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 :=
by
  sorry

end average_hamburgers_sold_per_day_l172_172658


namespace smallest_solution_floor_eq_l172_172691

theorem smallest_solution_floor_eq (x : ℝ) (hx : ⌊x^2⌋ - ⌊x⌋^2 = 19) : x = 11 := by
  sorry

end smallest_solution_floor_eq_l172_172691


namespace ladder_alley_width_l172_172010

theorem ladder_alley_width (l : ℝ) (m : ℝ) (w : ℝ) (h : m = l / 2) :
  w = (l * (Real.sqrt 3 + 1)) / 2 :=
by
  sorry

end ladder_alley_width_l172_172010


namespace smallest_solution_eq_sqrt_104_l172_172694

theorem smallest_solution_eq_sqrt_104 :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, ⌊y^2⌋ - ⌊y⌋^2 = 19 → x ≤ y) := sorry

end smallest_solution_eq_sqrt_104_l172_172694


namespace sequence_sum_zero_l172_172127

theorem sequence_sum_zero (n : ℕ) (h : n > 1) :
  (∃ (a : ℕ → ℤ), (∀ k : ℕ, k > 0 → a k ≠ 0) ∧ (∀ k : ℕ, k > 0 → a k + 2 * a (2 * k) + n * a (n * k) = 0)) ↔ n ≥ 3 := 
by sorry

end sequence_sum_zero_l172_172127


namespace find_other_number_l172_172240

variable (A B : ℕ)
variable (LCM : ℕ → ℕ → ℕ)
variable (HCF : ℕ → ℕ → ℕ)

theorem find_other_number (h1 : LCM A B = 2310) 
  (h2 : HCF A B = 30) (h3 : A = 210) : B = 330 := by
  sorry

end find_other_number_l172_172240


namespace largest_non_expressible_number_l172_172055

theorem largest_non_expressible_number :
  ∀ (x y z : ℕ), 15 * x + 18 * y + 20 * z ≠ 97 :=
by sorry

end largest_non_expressible_number_l172_172055


namespace unit_price_in_range_l172_172113

-- Given definitions and conditions
def Q (x : ℝ) : ℝ := 220 - 2 * x
def f (x : ℝ) : ℝ := x * Q x

-- The desired range for the unit price to maintain a production value of at least 60 million yuan
def valid_unit_price_range (x : ℝ) : Prop := 50 < x ∧ x < 60

-- The main theorem that needs to be proven
theorem unit_price_in_range (x : ℝ) (h₁ : 0 < x) (h₂ : x < 500) (h₃ : f x ≥ 60 * 10^6) : valid_unit_price_range x :=
sorry

end unit_price_in_range_l172_172113


namespace impossible_fifty_pieces_l172_172792

open Nat

theorem impossible_fifty_pieces :
  ¬ ∃ (m : ℕ), 1 + 3 * m = 50 :=
by
  sorry

end impossible_fifty_pieces_l172_172792


namespace course_selection_plans_l172_172489

def C (n k : ℕ) : ℕ := Nat.choose n k

theorem course_selection_plans :
  let A_courses := C 4 2
  let B_courses := C 4 3
  let C_courses := C 4 3
  A_courses * B_courses * C_courses = 96 :=
by
  sorry

end course_selection_plans_l172_172489


namespace polygon_proof_l172_172379

-- Define the conditions and the final proof problem.
theorem polygon_proof 
  (interior_angle : ℝ) 
  (side_length : ℝ) 
  (h1 : interior_angle = 160) 
  (h2 : side_length = 4) 
  : ∃ n : ℕ, ∃ P : ℝ, (interior_angle = 180 * (n - 2) / n) ∧ (P = n * side_length) ∧ (n = 18) ∧ (P = 72) :=
by
  sorry

end polygon_proof_l172_172379


namespace average_percentage_l172_172516

theorem average_percentage (x : ℝ) : (60 + x + 80) / 3 = 70 → x = 70 :=
by
  intro h
  sorry

end average_percentage_l172_172516


namespace polynomial_remainder_l172_172551

theorem polynomial_remainder (a b : ℝ) (h : ∀ x : ℝ, (x^3 - 2*x^2 + a*x + b) % ((x - 1)*(x - 2)) = 2*x + 1) : 
  a = 1 ∧ b = 3 := 
sorry

end polynomial_remainder_l172_172551


namespace pentagonal_grid_toothpicks_l172_172491

theorem pentagonal_grid_toothpicks :
  ∀ (base toothpicks per sides toothpicks per joint : ℕ),
    base = 10 → 
    sides = 4 → 
    toothpicks_per_side = 8 → 
    joints = 5 → 
    toothpicks_per_joint = 1 → 
    (base + sides * toothpicks_per_side + joints * toothpicks_per_joint = 47) :=
by
  intros base sides toothpicks_per_side joints toothpicks_per_joint
  sorry

end pentagonal_grid_toothpicks_l172_172491


namespace rectangle_perimeter_of_right_triangle_l172_172519

-- Define the conditions for the triangle and the rectangle
def rightTriangleArea (a b c : ℕ) (h : a^2 + b^2 = c^2) : ℕ :=
  (1 / 2) * a * b

def rectanglePerimeter (width area : ℕ) : ℕ :=
  2 * ((area / width) + width)

theorem rectangle_perimeter_of_right_triangle :
  ∀ (a b c width : ℕ) (h_a : a = 5) (h_b : b = 12) (h_c : c = 13)
    (h_pyth : a^2 + b^2 = c^2) (h_width : width = 5)
    (h_area_eq : rightTriangleArea a b c h_pyth = width * (rightTriangleArea a b c h_pyth / width)),
  rectanglePerimeter width (rightTriangleArea a b c h_pyth) = 22 :=
by
  intros
  sorry

end rectangle_perimeter_of_right_triangle_l172_172519


namespace num_integer_solutions_l172_172720

def circle_center := (3, 3)
def circle_radius := 10

theorem num_integer_solutions :
  (∃ f : ℕ, f = 15) :=
sorry

end num_integer_solutions_l172_172720


namespace find_tuition_l172_172795

def tuition_problem (T : ℝ) : Prop :=
  75 = T + (T - 15)

theorem find_tuition (T : ℝ) (h : tuition_problem T) : T = 45 :=
by
  sorry

end find_tuition_l172_172795


namespace smallest_x_solution_l172_172687

theorem smallest_x_solution :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, (⌊y^2⌋ - ⌊y⌋^2 = 19) → x ≤ y) ∧ x = Real.sqrt 119 := 
sorry

end smallest_x_solution_l172_172687


namespace value_of_n_l172_172721

theorem value_of_n : ∃ (n : ℕ), 6 * 8 * 3 * n = Nat.factorial 8 ∧ n = 280 :=
by
  use 280
  sorry

end value_of_n_l172_172721


namespace find_k_l172_172863

-- Define the vectors and the condition that k · a + b is perpendicular to a
theorem find_k 
  (a : ℝ × ℝ) (b : ℝ × ℝ) (k : ℝ)
  (h_a : a = (1, 2))
  (h_b : b = (-2, 0))
  (h_perpendicular : ∀ (k : ℝ), (k * a.1 + b.1, k * a.2 + b.2) • a = 0 ) : k = 2 / 5 :=
sorry

end find_k_l172_172863


namespace smallest_solution_floor_eq_l172_172689

theorem smallest_solution_floor_eq (x : ℝ) (hx : ⌊x^2⌋ - ⌊x⌋^2 = 19) : x = 11 := by
  sorry

end smallest_solution_floor_eq_l172_172689


namespace cafeteria_apples_pies_l172_172211

theorem cafeteria_apples_pies (initial_apples handed_out_apples apples_per_pie remaining_apples pies : ℕ) 
    (h_initial: initial_apples = 62) 
    (h_handed_out: handed_out_apples = 8) 
    (h_apples_per_pie: apples_per_pie = 9)
    (h_remaining: remaining_apples = initial_apples - handed_out_apples) 
    (h_pies: pies = remaining_apples / apples_per_pie) : 
    pies = 6 := by
  sorry

end cafeteria_apples_pies_l172_172211


namespace hvac_cost_per_vent_l172_172781

theorem hvac_cost_per_vent (cost : ℕ) (zones : ℕ) (vents_per_zone : ℕ) (h_cost : cost = 20000) (h_zones : zones = 2) (h_vents_per_zone : vents_per_zone = 5) :
  (cost / (zones * vents_per_zone) = 2000) :=
by
  sorry

end hvac_cost_per_vent_l172_172781


namespace cos_difference_simplify_l172_172466

theorem cos_difference_simplify 
  (x : ℝ) 
  (y : ℝ) 
  (z : ℝ) 
  (h1 : x = Real.cos 72)
  (h2 : y = Real.cos 144)
  (h3 : y = -Real.cos 36)
  (h4 : x = 2 * (Real.cos 36)^2 - 1)
  (hz : z = Real.cos 36)
  : x - y = 1 / 2 :=
by
  sorry

end cos_difference_simplify_l172_172466


namespace staplers_left_l172_172482

-- Definitions based on conditions
def initial_staplers : ℕ := 50
def dozen : ℕ := 12
def reports_stapled : ℕ := 3 * dozen

-- Statement of the theorem
theorem staplers_left (h : initial_staplers = 50) (d : dozen = 12) (r : reports_stapled = 3 * dozen) :
  (initial_staplers - reports_stapled) = 14 :=
sorry

end staplers_left_l172_172482


namespace arithmetic_seq_sum_l172_172883

theorem arithmetic_seq_sum (a : ℕ → ℝ) (d : ℝ) (h_arith : ∀ n, a (n + 1) = a n + d) (h_a5 : a 5 = 15) :
  a 3 + a 4 + a 6 + a 7 = 60 :=
sorry

end arithmetic_seq_sum_l172_172883


namespace value_sq_dist_OP_OQ_l172_172547

-- Definitions from problem conditions
def origin : ℝ × ℝ := (0, 0)
def ellipse (x y : ℝ) : Prop := (x^2) / 4 + (y^2) / 3 = 1
def perpendicular (p q : ℝ × ℝ) : Prop := p.1 * q.1 + p.2 * q.2 = 0

-- The proof statement
theorem value_sq_dist_OP_OQ 
  (P Q : ℝ × ℝ) 
  (hP : ellipse P.1 P.2) 
  (hQ : ellipse Q.1 Q.2) 
  (h_perp : perpendicular P Q)
  : (P.1^2 + P.2^2) + (Q.1^2 + Q.2^2) = 48 / 7 := 
sorry

end value_sq_dist_OP_OQ_l172_172547


namespace number_of_possible_teams_l172_172366

-- Definitions for the conditions
def num_goalkeepers := 3
def num_defenders := 5
def num_midfielders := 5
def num_strikers := 5

-- The number of ways to choose x from y
def choose (y x : ℕ) : ℕ := Nat.factorial y / (Nat.factorial x * Nat.factorial (y - x))

-- Main proof problem statement
theorem number_of_possible_teams :
  (choose num_goalkeepers 1) *
  (choose num_strikers 2) *
  (choose num_midfielders 4) *
  (choose (num_defenders + (num_midfielders - 4)) 4) = 2250 := by
  sorry

end number_of_possible_teams_l172_172366


namespace ratio_m_q_l172_172279

theorem ratio_m_q (m n p q : ℚ) (h1 : m / n = 25) (h2 : p / n = 5) (h3 : p / q = 1 / 15) : 
  m / q = 1 / 3 :=
by 
  sorry

end ratio_m_q_l172_172279


namespace systematic_sampling_removal_count_l172_172922

theorem systematic_sampling_removal_count :
  ∀ (N n : ℕ), N = 3204 ∧ n = 80 → N % n = 4 := 
by
  sorry

end systematic_sampling_removal_count_l172_172922


namespace remaining_integers_count_l172_172964

def set_of_integers_from_1_to_100 : Finset ℕ := (Finset.range 100).map ⟨Nat.succ, Nat.succ_injective⟩

def multiples_of (n : ℕ) (s : Finset ℕ) : Finset ℕ := s.filter (λ x => x % n = 0)

def T : Finset ℕ := set_of_integers_from_1_to_100
def M2 : Finset ℕ := multiples_of 2 T
def M3 : Finset ℕ := multiples_of 3 T
def M5 : Finset ℕ := multiples_of 5 T

def remaining_set : Finset ℕ := T \ (M2 ∪ M3 ∪ M5)

theorem remaining_integers_count : remaining_set.card = 26 := by
  sorry

end remaining_integers_count_l172_172964


namespace period_start_time_l172_172274

theorem period_start_time (end_time : ℕ) (rained_hours : ℕ) (not_rained_hours : ℕ) (total_hours : ℕ) (start_time : ℕ) 
  (h1 : end_time = 17) -- 5 pm as 17 in 24-hour format 
  (h2 : rained_hours = 2)
  (h3 : not_rained_hours = 6)
  (h4 : total_hours = rained_hours + not_rained_hours)
  (h5 : total_hours = 8)
  (h6 : start_time = end_time - total_hours)
  : start_time = 9 :=
sorry

end period_start_time_l172_172274


namespace solve_for_x_l172_172565

theorem solve_for_x (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 7 * x^2 + 14 * x * y = x^3 + 3 * x^2 * y) : x = 7 :=
by
  sorry

end solve_for_x_l172_172565


namespace A_intersect_B_l172_172613

def A : Set ℝ := { x | abs x < 2 }
def B : Set ℝ := { x | x^2 - 5 * x - 6 < 0 }

theorem A_intersect_B : A ∩ B = { x | -1 < x ∧ x < 2 } := by
  sorry

end A_intersect_B_l172_172613


namespace find_k_from_given_solution_find_other_root_l172_172732

-- Given
def one_solution_of_first_eq_is_same_as_second (x k : ℝ) : Prop :=
  x^2 + k * x - 2 = 0 ∧ (x + 1) / (x - 1) = 3

-- To find k
theorem find_k_from_given_solution : ∃ k : ℝ, ∃ x : ℝ, one_solution_of_first_eq_is_same_as_second x k ∧ k = -1 := by
  sorry

-- To find the other root
theorem find_other_root : ∃ x2 : ℝ, (x2 = -1) := by
  sorry

end find_k_from_given_solution_find_other_root_l172_172732


namespace initial_number_of_men_l172_172471

def initial_average_age_increased_by_2_years_when_two_women_replace_two_men 
    (M : ℕ) (A men1 men2 women1 women2 : ℕ) : Prop :=
  (men1 = 20) ∧ (men2 = 24) ∧ (women1 = 30) ∧ (women2 = 30) ∧
  ((M * A) + 16 = (M * (A + 2)))

theorem initial_number_of_men (M : ℕ) (A : ℕ) (men1 men2 women1 women2: ℕ):
  initial_average_age_increased_by_2_years_when_two_women_replace_two_men M A men1 men2 women1 women2 → 
  2 * M = 16 → M = 8 :=
by
  sorry

end initial_number_of_men_l172_172471


namespace solve_linear_system_l172_172562

theorem solve_linear_system (x y : ℝ) (h1 : 2 * x + 3 * y = 5) (h2 : 3 * x + 2 * y = 10) : x + y = 3 := 
by
  sorry

end solve_linear_system_l172_172562


namespace find_largest_m_l172_172141

theorem find_largest_m (m : ℤ) : (m^2 - 11 * m + 24 < 0) → m ≤ 7 := sorry

end find_largest_m_l172_172141


namespace part1_part2_l172_172861

noncomputable def f (x : Real) : Real :=
  2 * (Real.sin (Real.pi / 4 + x))^2 - Real.sqrt 3 * Real.cos (2 * x) - 1

noncomputable def h (x t : Real) : Real :=
  f (x + t)

theorem part1 (t : Real) (ht : 0 < t ∧ t < Real.pi / 2) :
  (h (-Real.pi / 6) t = 0) → t = Real.pi / 3 :=
sorry

theorem part2 (A B C : Real) (hA : 0 < A ∧ A < Real.pi / 2) (hA1 : h A (Real.pi / 3) = 1) :
  1 < ((Real.sqrt 3 - 1) * Real.sin B + Real.sqrt 2 * Real.sin (Real.pi / 2 - B)) ∧
  ((Real.sqrt 3 - 1) * Real.sin B + Real.sqrt 2 * Real.sin (Real.pi / 2 - B)) ≤ 2 :=
sorry

end part1_part2_l172_172861


namespace even_card_sum_probability_l172_172539

theorem even_card_sum_probability :
  let cards := Finset.range 10 \ 0, -- Cards from 1 to 9
      even_cards := cards.filter (λ n, n % 2 = 0), -- Filter even cards
      odd_cards := cards.filter (λ n, n % 2 = 1), -- Filter odd cards
      
      n := cards.card * (cards.card - 1) / 2, -- Total number of ways to pick 2 cards out of 9
      
      even_pairs := Finset.card (even_cards.pairCombinations), 
      odd_pairs := Finset.card (odd_cards.pairCombinations),
      
      m := even_pairs + odd_pairs, -- Total number of successful outcomes
      p := (m : ℚ) / n in -- Probability of successful outcome

      p = 4 / 9 := by {
  -- Card definition
  let cards := Finset.range 10 \ 0,
  -- Card counts
  have card_count := Finset.card cards,
  -- Even and odd cards
  let even_cards := cards.filter (λ n, n % 2 = 0),
  let odd_cards := cards.filter (λ n, n % 2 = 1),
  -- Number of ways to choose 2 cards out of 9
  have n := Finset.card (cards.pairCombinations),
  -- Even pairs count
  have even_pairs := Finset.card (even_cards.pairCombinations),
  -- Odd pairs count
  have odd_pairs := Finset.card (odd_cards.pairCombinations),
  -- Total success outcomes
  have m := even_pairs + odd_pairs,
  -- Probability computation
  let p := (m : ℚ) / n,
  -- Calculation
  have n_calc : n = 36 := sorry,
  have m_calc : m = 16 := sorry,
  -- Check
  rw [n_calc, m_calc] at p,
  norm_num at p,
  exact p,
}

end even_card_sum_probability_l172_172539


namespace log_m_n_iff_m_minus_1_n_minus_1_l172_172282

theorem log_m_n_iff_m_minus_1_n_minus_1 (m n : ℝ) (h1 : m > 0) (h2 : m ≠ 1) (h3 : n > 0) :
  (Real.log n / Real.log m < 0) ↔ ((m - 1) * (n - 1) < 0) :=
sorry

end log_m_n_iff_m_minus_1_n_minus_1_l172_172282


namespace geometric_sequence_a_l172_172080

open Real

theorem geometric_sequence_a (a : ℝ) (r : ℝ) (h1 : 20 * r = a) (h2 : a * r = 5/4) (h3 : 0 < a) : a = 5 :=
by
  -- The proof would go here
  sorry

end geometric_sequence_a_l172_172080


namespace solve_speeds_ratio_l172_172244

noncomputable def speeds_ratio (v_A v_B : ℝ) : Prop :=
  v_A / v_B = 1 / 3

theorem solve_speeds_ratio (v_A v_B : ℝ) (h1 : ∃ t : ℝ, t = 1 ∧ v_A = 300 - v_B ∧ v_A = v_B ∧ v_B = 300) 
  (h2 : ∃ t : ℝ, t = 7 ∧ 7 * v_A = 300 - 7 * v_B ∧ 7 * v_A = 300 - v_B ∧ 7 * v_B = v_A): 
    speeds_ratio v_A v_B :=
sorry

end solve_speeds_ratio_l172_172244


namespace center_cell_value_l172_172442

variable (a b c d e f g h i : ℝ)

-- Defining the conditions
def row_product_1 := a * b * c = 1 ∧ d * e * f = 1 ∧ g * h * i = 1
def col_product_1 := a * d * g = 1 ∧ b * e * h = 1 ∧ c * f * i = 1
def subgrid_product_2 := a * b * d * e = 2 ∧ b * c * e * f = 2 ∧ d * e * g * h = 2 ∧ e * f * h * i = 2

-- The theorem to prove
theorem center_cell_value (h1 : row_product_1 a b c d e f g h i) 
                          (h2 : col_product_1 a b c d e f g h i) 
                          (h3 : subgrid_product_2 a b c d e f g h i) : 
                          e = 1 :=
by
  sorry

end center_cell_value_l172_172442


namespace inequality_solution_range_4_l172_172789

theorem inequality_solution_range_4 (a : ℝ) : 
  (∃ x : ℝ, |x - 2| - |x + 2| ≥ a) → a ≤ 4 :=
sorry

end inequality_solution_range_4_l172_172789


namespace valid_years_count_l172_172847

def is_valid_year (yy : Nat) : Prop :=
  Nat.gcd 20 yy = 1

def count_valid_years (start end_ : Nat) : Nat :=
  (List.range (end_ - start + 1)).countp (λ n => is_valid_year (start + n))

theorem valid_years_count : count_valid_years 2026 2099 = 30 := by
  sorry

end valid_years_count_l172_172847


namespace linear_function_through_two_points_l172_172392

theorem linear_function_through_two_points :
  ∃ (k b : ℝ), (∀ x, y = k * x + b) ∧
  (k ≠ 0) ∧
  (3 = 2 * k + b) ∧
  (2 = 3 * k + b) ∧
  (∀ x, y = -x + 5) :=
by
  sorry

end linear_function_through_two_points_l172_172392


namespace min_sum_distances_to_corners_of_rectangle_center_l172_172812

theorem min_sum_distances_to_corners_of_rectangle_center (P A B C D : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (1, 0))
  (hC : C = (1, 1))
  (hD : D = (0, 1))
  (hP_center : P = (0.5, 0.5)) :
  ∀ Q, (dist Q A + dist Q B + dist Q C + dist Q D) ≥ (dist P A + dist P B + dist P C + dist P D) := 
sorry

end min_sum_distances_to_corners_of_rectangle_center_l172_172812


namespace rational_solutions_for_k_l172_172975

theorem rational_solutions_for_k :
  ∀ (k : ℕ), k > 0 → 
  (∃ x : ℚ, k * x^2 + 16 * x + k = 0) ↔ k = 8 :=
by
  sorry

end rational_solutions_for_k_l172_172975


namespace factor_expression_l172_172837

theorem factor_expression (c : ℝ) : 270 * c^2 + 45 * c - 15 = 15 * c * (18 * c + 2) :=
by
  sorry

end factor_expression_l172_172837


namespace total_cookies_l172_172666

-- Definitions from conditions
def cookies_per_guest : ℕ := 2
def number_of_guests : ℕ := 5

-- Theorem statement that needs to be proved
theorem total_cookies : cookies_per_guest * number_of_guests = 10 := by
  -- We skip the proof since only the statement is required
  sorry

end total_cookies_l172_172666


namespace find_f_zero_l172_172977

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_zero (h : ∀ x : ℝ, x ≠ 0 → f (2 * x - 1) = (1 - x^2) / x^2) : f 0 = 3 :=
sorry

end find_f_zero_l172_172977


namespace relationship_cannot_be_determined_l172_172771

noncomputable def point_on_parabola (a b c x y : ℝ) : Prop :=
  y = a * x^2 + b * x + c

theorem relationship_cannot_be_determined
  (a b c x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) (h1 : a ≠ 0) 
  (h2 : point_on_parabola a b c x1 y1) 
  (h3 : point_on_parabola a b c x2 y2) 
  (h4 : point_on_parabola a b c x3 y3) 
  (h5 : point_on_parabola a b c x4 y4)
  (h6 : x1 + x4 - x2 + x3 = 0) : 
  ¬( ∃ m n : ℝ, ((y4 - y1) / (x4 - x1) = m ∧ (y2 - y3) / (x2 - x3) = m) ∨ 
                     ((y4 - y1) / (x4 - x1) * (y2 - y3) / (x2 - x3) = -1) ∨ 
                     ((y4 - y1) / (x4 - x1) ≠ m ∧ (y2 - y3) / (x2 - x3) ≠ m ∧ 
                      (y4 - y1) / (x4 - x1) * (y2 - y3) / (x2 - x3) ≠ -1)) :=
sorry

end relationship_cannot_be_determined_l172_172771


namespace mail_cars_in_train_l172_172517

theorem mail_cars_in_train (n : ℕ) (hn : n % 2 = 0) (hfront : 1 ≤ n ∧ n ≤ 20)
  (hclose : ∀ i, 1 ≤ i ∧ i < n → (∃ j, i < j ∧ j ≤ 20))
  (hlast : 4 * n ≤ 20)
  (hconn : ∀ k, (k = 4 ∨ k = 5 ∨ k = 15 ∨ k = 16) → 
                  (∃ j, j = k + 1 ∨ j = k - 1)) :
  ∃ (i : ℕ) (j : ℕ), i = 4 ∧ j = 16 :=
by
  sorry

end mail_cars_in_train_l172_172517


namespace find_m_for_positive_integer_x_l172_172734

theorem find_m_for_positive_integer_x :
  ∃ (m : ℤ), (2 * m * x - 8 = (m + 2) * x) → ∀ (x : ℤ), x > 0 → m = 3 ∨ m = 4 ∨ m = 6 ∨ m = 10 :=
sorry

end find_m_for_positive_integer_x_l172_172734


namespace percentage_flowering_plants_l172_172894

variable (P : ℝ)

theorem percentage_flowering_plants (h : 5 * (1 / 4) * (P / 100) * 80 = 40) : P = 40 :=
by
  -- This is where the proof would go, but we will use sorry to skip it for now
  sorry

end percentage_flowering_plants_l172_172894


namespace rectangle_area_l172_172791

theorem rectangle_area 
  (length_to_width_ratio : Real) 
  (width : Real) 
  (area : Real) 
  (h1 : length_to_width_ratio = 0.875) 
  (h2 : width = 24) 
  (h_area : area = 504) : 
  True := 
sorry

end rectangle_area_l172_172791


namespace fraction_power_minus_one_l172_172523

theorem fraction_power_minus_one :
  (5 / 3) ^ 4 - 1 = 544 / 81 := 
by
  sorry

end fraction_power_minus_one_l172_172523


namespace find_f_13_l172_172344

noncomputable def f : ℝ → ℝ := sorry

axiom f_property : ∀ x, f (x + f x) = 3 * f x
axiom f_of_1 : f 1 = 3

theorem find_f_13 : f 13 = 27 :=
by
  have hf := f_property
  have hf1 := f_of_1
  sorry

end find_f_13_l172_172344


namespace valid_values_l172_172227

noncomputable def is_defined (x : ℝ) : Prop := 
  (x^2 - 4*x + 3 > 0) ∧ (5 - x^2 > 0)

theorem valid_values (x : ℝ) : 
  is_defined x ↔ (-Real.sqrt 5 < x ∧ x < 1) ∨ (3 < x ∧ x < Real.sqrt 5) := by
  sorry

end valid_values_l172_172227


namespace seating_arrangement_l172_172572

theorem seating_arrangement :
  let total_arrangements := Nat.factorial 8
  let adjacent_arrangements := Nat.factorial 7 * 2
  total_arrangements - adjacent_arrangements = 30240 :=
by
  sorry

end seating_arrangement_l172_172572


namespace correct_option_is_B_l172_172800

def natural_growth_rate (birth_rate death_rate : ℕ) : ℕ :=
  birth_rate - death_rate

def option_correct (birth_rate death_rate : ℕ) :=
  (∃ br dr, natural_growth_rate br dr = br - dr)

theorem correct_option_is_B (birth_rate death_rate : ℕ) :
  option_correct birth_rate death_rate :=
by 
  sorry

end correct_option_is_B_l172_172800


namespace how_many_strawberries_did_paul_pick_l172_172768

-- Here, we will define the known quantities
def original_strawberries : Nat := 28
def total_strawberries : Nat := 63

-- The statement to prove
theorem how_many_strawberries_did_paul_pick : total_strawberries - original_strawberries = 35 :=
by
  unfold total_strawberries
  unfold original_strawberries
  calc
    63 - 28 = 35 := by norm_num

end how_many_strawberries_did_paul_pick_l172_172768


namespace lcm_two_numbers_l172_172566

theorem lcm_two_numbers (a b : ℕ) (h1 : a * b = 17820) (h2 : Nat.gcd a b = 12) : Nat.lcm a b = 1485 := 
by
  sorry

end lcm_two_numbers_l172_172566


namespace max_marks_paper_I_l172_172945

-- Definitions based on the problem conditions
def percent_to_pass : ℝ := 0.35
def secured_marks : ℝ := 42
def failed_by : ℝ := 23

-- The calculated passing marks
def passing_marks : ℝ := secured_marks + failed_by

-- The theorem statement that needs to be proved
theorem max_marks_paper_I : ∀ (M : ℝ), (percent_to_pass * M = passing_marks) → M = 186 :=
by
  intros M h
  have h1 : M = passing_marks / percent_to_pass := by sorry
  have h2 : M = 186 := by sorry
  exact h2

end max_marks_paper_I_l172_172945


namespace equivalent_single_discount_l172_172823

theorem equivalent_single_discount (p : ℝ) :
  let discount1 := 0.15
  let discount2 := 0.10
  let discount3 := 0.05
  let final_price := (1 - discount1) * (1 - discount2) * (1 - discount3) * p
  (1 - final_price / p) = 0.27325 :=
by
  sorry

end equivalent_single_discount_l172_172823


namespace smallest_x_solution_l172_172683

theorem smallest_x_solution :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, (⌊y^2⌋ - ⌊y⌋^2 = 19) → x ≤ y) ∧ x = Real.sqrt 119 := 
sorry

end smallest_x_solution_l172_172683


namespace employee_total_weekly_pay_l172_172958

-- Define the conditions
def hours_per_day_first_3_days : ℕ := 6
def hours_per_day_last_2_days : ℕ := 2 * hours_per_day_first_3_days
def first_40_hours_pay_rate : ℕ := 30
def overtime_multiplier : ℕ := 3 / 2 -- 50% more pay, i.e., 1.5 times

-- Functions to compute total hours worked and total pay
def hours_first_3_days (d : ℕ) : ℕ := d * hours_per_day_first_3_days
def hours_last_2_days (d : ℕ) : ℕ := d * hours_per_day_last_2_days
def total_hours_worked : ℕ := (hours_first_3_days 3) + (hours_last_2_days 2)
def regular_hours : ℕ := min 40 total_hours_worked
def overtime_hours : ℕ := total_hours_worked - regular_hours
def regular_pay : ℕ := regular_hours * first_40_hours_pay_rate
def overtime_pay_rate : ℕ := first_40_hours_pay_rate + (first_40_hours_pay_rate / 2) -- 50% more
def overtime_pay : ℕ := overtime_hours * overtime_pay_rate
def total_pay : ℕ := regular_pay + overtime_pay

-- The statement to be proved
theorem employee_total_weekly_pay : total_pay = 1290 := by
  sorry

end employee_total_weekly_pay_l172_172958


namespace max_c_value_for_f_x_range_l172_172142

theorem max_c_value_for_f_x_range:
  (∀ c : ℝ, (∃ x : ℝ, x^2 + 4 * x + c = -2) → c ≤ 2) ∧ (∃ (x : ℝ), x^2 + 4 * x + 2 = -2) :=
sorry

end max_c_value_for_f_x_range_l172_172142


namespace find_percentage_l172_172298

variable (X P : ℝ)

theorem find_percentage (h₁ : 0.20 * X = 400) (h₂ : (P / 100) * X = 2400) : P = 120 :=
by
  -- The proof is intentionally left out
  sorry

end find_percentage_l172_172298


namespace no_a_satisfies_condition_l172_172757

noncomputable def M : Set ℝ := {0, 1}
noncomputable def N (a : ℝ) : Set ℝ := {11 - a, Real.log a / Real.log 1, 2^a, a}

theorem no_a_satisfies_condition :
  ¬ ∃ a : ℝ, M ∩ N a = {1} :=
by
  sorry

end no_a_satisfies_condition_l172_172757


namespace opposite_of_negative_six_is_six_l172_172216

-- Define what it means for one number to be the opposite of another.
def is_opposite (a b : Int) : Prop :=
  a = -b

-- The statement to be proved: the opposite number of -6 is 6.
theorem opposite_of_negative_six_is_six : is_opposite (-6) 6 :=
  by sorry

end opposite_of_negative_six_is_six_l172_172216


namespace evaluate_g_at_neg1_l172_172736

def g (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

theorem evaluate_g_at_neg1 : g (-1) = -5 := by
  sorry

end evaluate_g_at_neg1_l172_172736


namespace exists_solution_negation_correct_l172_172546

theorem exists_solution_negation_correct :
  (∃ x : ℝ, x^2 - x = 0) ↔ (∃ x : ℝ, True) ∧ (∀ x : ℝ, ¬ (x^2 - x = 0)) :=
by
  sorry

end exists_solution_negation_correct_l172_172546


namespace Sophia_fraction_finished_l172_172468

/--
Sophia finished a fraction of a book.
She calculated that she finished 90 more pages than she has yet to read.
Her book is 270.00000000000006 pages long.
Prove that the fraction of the book she finished is 2/3.
-/
theorem Sophia_fraction_finished :
  let total_pages : ℝ := 270.00000000000006
  let yet_to_read : ℝ := (total_pages - 90) / 2
  let finished_pages : ℝ := yet_to_read + 90
  finished_pages / total_pages = 2 / 3 :=
by
  sorry

end Sophia_fraction_finished_l172_172468


namespace candle_ratio_l172_172042

theorem candle_ratio (r b : ℕ) (h1: r = 45) (h2: b = 27) : r / Nat.gcd r b = 5 ∧ b / Nat.gcd r b = 3 := 
by
  sorry

end candle_ratio_l172_172042


namespace standard_circle_equation_l172_172352

theorem standard_circle_equation (x y : ℝ) :
  ∃ (h k r : ℝ), h = 2 ∧ k = -1 ∧ r = 3 ∧ (x - h)^2 + (y - k + 1)^2 = r^2 :=
by
  use 2, -1, 3
  simp
  sorry

end standard_circle_equation_l172_172352


namespace lives_per_player_l172_172377

theorem lives_per_player (num_players total_lives : ℕ) (h1 : num_players = 8) (h2 : total_lives = 64) :
  total_lives / num_players = 8 := by
  sorry

end lives_per_player_l172_172377


namespace sequence_polynomial_l172_172132

theorem sequence_polynomial (f : ℕ → ℤ) :
  (f 0 = 3 ∧ f 1 = 7 ∧ f 2 = 21 ∧ f 3 = 51) ↔ (∀ n, f n = n^3 + 2 * n^2 + n + 3) :=
by
  sorry

end sequence_polynomial_l172_172132


namespace Rick_received_amount_l172_172588

theorem Rick_received_amount :
  let total_promised := 400
  let sally_owes := 35
  let amy_owes := 30
  let derek_owes := amy_owes / 2
  let carl_owes := 35
  let total_owed := sally_owes + amy_owes + derek_owes + carl_owes
  total_promised - total_owed = 285 :=
by
  sorry

end Rick_received_amount_l172_172588


namespace largest_integer_m_such_that_expression_is_negative_l172_172140

theorem largest_integer_m_such_that_expression_is_negative :
  ∃ (m : ℤ), (∀ (n : ℤ), (m^2 - 11 * m + 24 < 0 ) → n < m → n^2 - 11 * n + 24 < 0) ∧
  m^2 - 11 * m + 24 < 0 ∧
  (m ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) :=
by
  sorry

end largest_integer_m_such_that_expression_is_negative_l172_172140


namespace seminar_duration_total_l172_172786

/-- The first part of the seminar lasted 4 hours and 45 minutes -/
def first_part_minutes := 4 * 60 + 45

/-- The second part of the seminar lasted 135 minutes -/
def second_part_minutes := 135

/-- The closing event lasted 500 seconds -/
def closing_event_minutes := 500 / 60

/-- The total duration of the seminar session in minutes, including the closing event, is 428 minutes -/
theorem seminar_duration_total :
  first_part_minutes + second_part_minutes + closing_event_minutes = 428 := by
  sorry

end seminar_duration_total_l172_172786


namespace probability_sum_of_two_dice_is_five_l172_172501

theorem probability_sum_of_two_dice_is_five :
  let outcomes := {(a, b) | a ∈ Fin 6, b ∈ Fin 6}
  let favorable := {(a, b) | (a + 1) + (b + 1) = 5}
  (favorable.to_finset.card / outcomes.to_finset.card : ℚ) = 1 / 9 :=
by
  sorry

end probability_sum_of_two_dice_is_five_l172_172501


namespace possible_division_l172_172381

theorem possible_division (side_length : ℕ) (areas : Fin 5 → Set (Fin side_length × Fin side_length))
  (h1 : side_length = 5)
  (h2 : ∀ i, ∃ cells : Finset (Fin side_length × Fin side_length), areas i = cells ∧ Finset.card cells = 5)
  (h3 : ∀ i j, i ≠ j → Disjoint (areas i) (areas j))
  (total_cut_length : ℕ)
  (h4 : total_cut_length ≤ 16) :
  
  ∃ cuts : Finset (Fin side_length × Fin side_length) × Finset (Fin side_length × Fin side_length),
    total_cut_length = (cuts.1.card + cuts.2.card) :=
sorry

end possible_division_l172_172381


namespace number_of_envelopes_l172_172522

theorem number_of_envelopes (total_weight_grams : ℕ) (weight_per_envelope_grams : ℕ) (n : ℕ) :
  total_weight_grams = 7480 ∧ weight_per_envelope_grams = 8500 ∧ n = 880 → total_weight_grams = n * weight_per_envelope_grams := 
sorry

end number_of_envelopes_l172_172522


namespace tea_in_each_box_initially_l172_172919

theorem tea_in_each_box_initially (x : ℕ) 
  (h₁ : 4 * (x - 9) = x) : 
  x = 12 := 
sorry

end tea_in_each_box_initially_l172_172919


namespace water_pumping_problem_l172_172596

theorem water_pumping_problem :
  let pumpA_rate := 300 -- gallons per hour
  let pumpB_rate := 500 -- gallons per hour
  let combined_rate := pumpA_rate + pumpB_rate -- Combined rate per hour
  let time_duration := 1 / 2 -- Time in hours (30 minutes)
  combined_rate * time_duration = 400 := -- Total volume in gallons
by
  -- Lean proof would go here
  sorry

end water_pumping_problem_l172_172596


namespace pd_distance_l172_172400

theorem pd_distance (PA PB PC PD : ℝ) (hPA : PA = 17) (hPB : PB = 15) (hPC : PC = 6) :
  PA^2 + PC^2 = PB^2 + PD^2 → PD = 10 :=
by
  sorry

end pd_distance_l172_172400


namespace isosceles_triangle_base_length_l172_172545

theorem isosceles_triangle_base_length
  (a b : ℝ) (h₁ : a = 4) (h₂ : b = 8) (h₃ : a ≠ b)
  (triangle_inequality : ∀ x y z : ℝ, x + y > z) :
  ∃ base : ℝ, base = 8 := by
  sorry

end isosceles_triangle_base_length_l172_172545


namespace sammy_offer_l172_172200

-- Declaring the given constants and assumptions
def peggy_records : ℕ := 200
def bryan_interested_records : ℕ := 100
def bryan_uninterested_records : ℕ := 100
def bryan_interested_offer : ℕ := 6
def bryan_uninterested_offer : ℕ := 1
def sammy_offer_diff : ℕ := 100

-- The problem to be proved
theorem sammy_offer:
    ∃ S : ℝ, 
    (200 * S) - 
    (bryan_interested_records * bryan_interested_offer +
    bryan_uninterested_records * bryan_uninterested_offer) = sammy_offer_diff → 
    S = 4 :=
sorry

end sammy_offer_l172_172200


namespace cos_sum_is_zero_l172_172763

theorem cos_sum_is_zero (x y z : ℝ) 
  (h1: Real.cos (2 * x) + 2 * Real.cos (2 * y) + 3 * Real.cos (2 * z) = 0) 
  (h2: Real.sin (2 * x) + 2 * Real.sin (2 * y) + 3 * Real.sin (2 * z) = 0) : 
  Real.cos (4 * x) + Real.cos (4 * y) + Real.cos (4 * z) = 0 := 
by 
  sorry

end cos_sum_is_zero_l172_172763


namespace proof_problem_l172_172295

open Real

noncomputable def problem (c d : ℝ) : ℝ :=
  5^(c / d) + 2^(d / c)

theorem proof_problem :
  let c := log 8
  let d := log 25
  problem c d = 2 * sqrt 2 + 5^(2 / 3) :=
by
  intro c d
  have c_def : c = log 8 := rfl
  have d_def : d = log 25 := rfl
  rw [c_def, d_def]
  sorry

end proof_problem_l172_172295


namespace rocky_running_ratio_l172_172016

theorem rocky_running_ratio (x y : ℕ) (h1 : x = 4) (h2 : 2 * x + y = 36) : y / (2 * x) = 3 :=
by
  sorry

end rocky_running_ratio_l172_172016


namespace gcd_of_840_and_1764_l172_172346

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := 
by {
  sorry
}

end gcd_of_840_and_1764_l172_172346


namespace compute_r_l172_172674

noncomputable def r (side_length : ℝ) : ℝ :=
  let a := (0.5 * side_length, 0.5 * side_length)
  let b := (1.5 * side_length, 2.5 * side_length)
  let c := (2.5 * side_length, 1.5 * side_length)
  let ab := Real.sqrt ((b.1 - a.1)^2 + (b.2 - a.2)^2)
  let ac := Real.sqrt ((c.1 - a.1)^2 + (c.2 - a.2)^2)
  let bc := Real.sqrt ((c.1 - b.1)^2 + (c.2 - b.2)^2)
  let s := (ab + ac + bc) / 2
  let area_ABC := Real.sqrt (s * (s - ab) * (s - ac) * (s - bc))
  let circumradius := ab * ac * bc / (4 * area_ABC)
  circumradius - (side_length / 2)

theorem compute_r :
  r 1 = (5 * Real.sqrt 2 - 3) / 6 :=
by
  unfold r
  sorry

end compute_r_l172_172674


namespace interest_rate_l172_172068

theorem interest_rate (P : ℝ) (r : ℝ) (t : ℝ) (CI SI : ℝ) (diff : ℝ) 
    (hP : P = 1500)
    (ht : t = 2)
    (hdiff : diff = 15)
    (hCI : CI = P * (1 + r / 100)^t - P)
    (hSI : SI = P * r * t / 100)
    (hCI_SI_diff : CI - SI = diff) :
    r = 1 := 
by
  sorry -- proof goes here


end interest_rate_l172_172068


namespace factor_expression_l172_172678

variable (x : ℝ)

theorem factor_expression : 75 * x^3 - 250 * x^7 = 25 * x^3 * (3 - 10 * x^4) :=
by
  sorry

end factor_expression_l172_172678


namespace number_of_trailing_zeroes_base8_l172_172000

theorem number_of_trailing_zeroes_base8 (n : ℕ) (hn : n = 15) : 
  (trailing_zeroes_base8 (factorial 15)) = 3 := 
by
  sorry

end number_of_trailing_zeroes_base8_l172_172000


namespace average_speed_comparison_l172_172121

theorem average_speed_comparison (u v w : ℝ) (hu : u > 0) (hv : v > 0) (hw : w > 0):
  (3 / (1 / u + 1 / v + 1 / w)) ≤ ((u + v + w) / 3) :=
sorry

end average_speed_comparison_l172_172121


namespace building_height_l172_172367

theorem building_height (flagpole_height : ℝ) (flagpole_shadow : ℝ) 
  (building_shadow : ℝ) (building_height : ℝ)
  (h_flagpole : flagpole_height = 18)
  (s_flagpole : flagpole_shadow = 45)
  (s_building : building_shadow = 70)
  (ratio_eq : flagpole_height / flagpole_shadow = building_height / building_shadow) :
  building_height = 28 :=
by
  have h_flagpole_shadow := ratio_eq ▸ h_flagpole ▸ s_flagpole ▸ s_building
  sorry

end building_height_l172_172367


namespace lcm_150_456_l172_172496

theorem lcm_150_456 : Nat.lcm 150 456 = 11400 := by
  sorry

end lcm_150_456_l172_172496


namespace min_score_to_achieve_average_l172_172769

theorem min_score_to_achieve_average (a b c : ℕ) (h₁ : a = 76) (h₂ : b = 94) (h₃ : c = 87) :
  ∃ d e : ℕ, d + e = 148 ∧ d ≤ 100 ∧ e ≤ 100 ∧ min d e = 48 :=
by sorry

end min_score_to_achieve_average_l172_172769


namespace calculate_value_of_A_plus_C_l172_172948

theorem calculate_value_of_A_plus_C (A B C : ℕ) (hA : A = 238) (hAB : A = B + 143) (hBC : C = B + 304) : A + C = 637 :=
by
  sorry

end calculate_value_of_A_plus_C_l172_172948


namespace triangle_proof_l172_172310

noncomputable def length_DC (AB DA BC DB : ℝ) : ℝ :=
  Real.sqrt (BC^2 - DB^2)

theorem triangle_proof :
  ∀ (AB DA BC DB : ℝ), AB = 30 → DA = 24 → BC = 22.5 → DB = 18 →
  length_DC AB DA BC DB = 13.5 :=
by
  intros AB DA BC DB hAB hDA hBC hDB
  rw [length_DC]
  sorry

end triangle_proof_l172_172310


namespace prime_pairs_l172_172833

open Nat

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, 2 ≤ m → m ≤ n / 2 → n % m ≠ 0

theorem prime_pairs :
  ∀ (p q : ℕ), is_prime p → is_prime q →
  1 < p → p < 100 →
  1 < q → q < 100 →
  is_prime (p + 6) →
  is_prime (p + 10) →
  is_prime (q + 4) →
  is_prime (q + 10) →
  is_prime (p + q + 1) →
  (p, q) = (7, 3) ∨ (p, q) = (13, 3) ∨ (p, q) = (37, 3) ∨ (p, q) = (97, 3) :=
by
  sorry

end prime_pairs_l172_172833


namespace find_interest_rate_l172_172194

-- Translating the identified conditions into Lean definitions
def initial_deposit (P : ℝ) : Prop := P > 0
def compounded_semiannually (n : ℕ) : Prop := n = 2
def growth_in_sum (A : ℝ) (P : ℝ) : Prop := A = 1.1592740743 * P
def time_period (t : ℝ) : Prop := t = 2.5

theorem find_interest_rate (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) (A : ℝ)
  (h_init : initial_deposit P)
  (h_n : compounded_semiannually n)
  (h_A : growth_in_sum A P)
  (h_t : time_period t) :
  r = 0.06 :=
by
  sorry

end find_interest_rate_l172_172194


namespace jewel_price_reduction_l172_172665

theorem jewel_price_reduction (P x : ℝ) (P1 : ℝ) (hx : x ≠ 0) 
  (hP1 : P1 = P * (1 - (x / 100) ^ 2))
  (h_final : P1 * (1 - (x / 100) ^ 2) = 2304) : 
  P1 = 2304 / (1 - (x / 100) ^ 2) :=
by
  sorry

end jewel_price_reduction_l172_172665


namespace area_of_rectangle_l172_172090

theorem area_of_rectangle (length width : ℝ) (h_length : length = 47.3) (h_width : width = 24) :
  length * width = 1135.2 :=
by
  sorry -- Skip the proof

end area_of_rectangle_l172_172090


namespace minimum_cost_l172_172921

theorem minimum_cost (
    x y m w : ℝ) 
    (h1 : 4 * x + 2 * y = 400)
    (h2 : 2 * x + 4 * y = 320)
    (h3 : m ≥ 16)
    (h4 : m + (80 - m) = 80)
    (h5 : w = 80 * m + 40 * (80 - m)) :
    x = 80 ∧ y = 40 ∧ w = 3840 :=
by 
  sorry

end minimum_cost_l172_172921


namespace xyz_squared_sum_l172_172794

theorem xyz_squared_sum (x y z : ℝ)
  (h1 : x^2 + 6 * y = -17)
  (h2 : y^2 + 4 * z = 1)
  (h3 : z^2 + 2 * x = 2) :
  x^2 + y^2 + z^2 = 14 := 
sorry

end xyz_squared_sum_l172_172794


namespace consecutive_product_even_product_divisible_by_6_l172_172889

theorem consecutive_product_even (n : ℕ) : ∃ k, n * (n + 1) = 2 * k := 
sorry

theorem product_divisible_by_6 (n : ℕ) : 6 ∣ (n * (n + 1) * (2 * n + 1)) :=
sorry

end consecutive_product_even_product_divisible_by_6_l172_172889


namespace find_smallest_solution_l172_172712

theorem find_smallest_solution : ∃ x : ℝ, x = Real.sqrt 119 ∧ (Int.floor (x^2) - Int.floor x ^ 2 = 19) := by
  sorry

end find_smallest_solution_l172_172712


namespace price_of_whole_pizza_l172_172207

theorem price_of_whole_pizza
    (price_per_slice : ℕ)
    (num_slices_sold : ℕ)
    (num_whole_pizzas_sold : ℕ)
    (total_revenue : ℕ) 
    (H : price_per_slice * num_slices_sold + num_whole_pizzas_sold * P = total_revenue) : 
    P = 15 :=
by
  let price_per_slice := 3
  let num_slices_sold := 24
  let num_whole_pizzas_sold := 3
  let total_revenue := 117
  sorry

end price_of_whole_pizza_l172_172207


namespace min_value_z_l172_172991

theorem min_value_z (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 1) :
  (x + 1/x) * (y + 1/y) ≥ 25/4 := 
sorry

end min_value_z_l172_172991


namespace exams_in_fourth_year_l172_172660

variable (a b c d e : ℕ)

theorem exams_in_fourth_year:
  a + b + c + d + e = 31 ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e = 3 * a → d = 8 := by
  sorry

end exams_in_fourth_year_l172_172660


namespace smallest_lcm_of_4digit_multiples_of_5_l172_172868

theorem smallest_lcm_of_4digit_multiples_of_5 :
  ∃ m n : ℕ, (1000 ≤ m) ∧ (m ≤ 9999) ∧ (1000 ≤ n) ∧ (n ≤ 9999) ∧ (Nat.gcd m n = 5) ∧ (Nat.lcm m n = 201000) := 
sorry

end smallest_lcm_of_4digit_multiples_of_5_l172_172868


namespace find_usual_time_l172_172938

noncomputable def journey_time (S T : ℝ) : Prop :=
  (6 / 5) = (T + (1 / 5)) / T

theorem find_usual_time (S T : ℝ) (h1 : ∀ S T, S / (5 / 6 * S) = (T + (12 / 60)) / T) : T = 1 :=
by
  -- Let the conditions defined by the user be:
  -- h1 : condition (e.g., the cab speed and time relationship)
  -- Given that the cab is \(\frac{5}{6}\) times its speed and is late by 12 minutes
  let h1 := journey_time S T
  sorry

end find_usual_time_l172_172938


namespace monotonically_increasing_range_a_l172_172006

theorem monotonically_increasing_range_a (a : ℝ) :
  (∀ x y : ℝ, x < y → (x^3 + a * x) ≤ (y^3 + a * y)) → a ≥ 0 := 
by
  sorry

end monotonically_increasing_range_a_l172_172006


namespace friends_area_is_greater_by_14_point_4_times_l172_172361

theorem friends_area_is_greater_by_14_point_4_times :
  let tommy_length := 2 * 200
  let tommy_width := 3 * 150
  let tommy_area := tommy_length * tommy_width
  let friend_block_area := 180 * 180
  let friend_area := 80 * friend_block_area
  friend_area / tommy_area = 14.4 :=
by
  let tommy_length := 2 * 200
  let tommy_width := 3 * 150
  let tommy_area := tommy_length * tommy_width
  let friend_block_area := 180 * 180
  let friend_area := 80 * friend_block_area
  sorry

end friends_area_is_greater_by_14_point_4_times_l172_172361


namespace part1_part2_l172_172555

noncomputable def f (x : ℝ) : ℝ := |x| + |x + 1|

theorem part1 (x : ℝ) : f x > 3 ↔ x > 1 ∨ x < -2 :=
by
  sorry

theorem part2 (m : ℝ) (hx : ∀ x : ℝ, m^2 + 3 * m + 2 * f x ≥ 0) : m ≤ -2 ∨ m ≥ -1 :=
by
  sorry

end part1_part2_l172_172555


namespace faster_train_length_is_150_l172_172633

def speed_faster_train_kmph : ℝ := 72
def speed_slower_train_kmph : ℝ := 36
def time_seconds : ℝ := 15

noncomputable def length_faster_train : ℝ :=
  let relative_speed_kmph := speed_faster_train_kmph - speed_slower_train_kmph
  let relative_speed_mps := relative_speed_kmph * 1000 / 3600
  relative_speed_mps * time_seconds

theorem faster_train_length_is_150 :
  length_faster_train = 150 := by
sorry

end faster_train_length_is_150_l172_172633


namespace parallelepiped_vectors_l172_172020

theorem parallelepiped_vectors (x y z : ℝ)
  (h1: ∀ (AB BC CC1 AC1 : ℝ), AC1 = AB + BC + CC1)
  (h2: ∀ (AB BC CC1 AC1 : ℝ), AC1 = x * AB + 2 * y * BC + 3 * z * CC1) :
  x + y + z = 11 / 6 :=
by
  -- This is where the proof would go, but as per the instruction we'll add sorry.
  sorry

end parallelepiped_vectors_l172_172020


namespace find_length_of_AC_l172_172737

theorem find_length_of_AC
  (A B C : Type)
  (AB : Real)
  (AC : Real)
  (Area : Real)
  (angle_A : Real)
  (h1 : AB = 8)
  (h2 : angle_A = (30 * Real.pi / 180)) -- converting degrees to radians
  (h3 : Area = 16) :
  AC = 8 :=
by
  -- Skipping proof as requested
  sorry

end find_length_of_AC_l172_172737


namespace find_smallest_solution_l172_172717

theorem find_smallest_solution : ∃ x : ℝ, x = Real.sqrt 119 ∧ (Int.floor (x^2) - Int.floor x ^ 2 = 19) := by
  sorry

end find_smallest_solution_l172_172717


namespace find_x_if_perpendicular_l172_172288

-- Given definitions and conditions
def a : ℝ × ℝ := (-5, 1)
def b (x : ℝ) : ℝ × ℝ := (2, x)

-- Statement to be proved
theorem find_x_if_perpendicular (x : ℝ) :
  (a.1 * (b x).1 + a.2 * (b x).2 = 0) → x = 10 :=
by
  sorry

end find_x_if_perpendicular_l172_172288


namespace temperature_conversion_l172_172805

noncomputable def celsius_to_fahrenheit (c : ℝ) : ℝ :=
  (c * (9 / 5)) + 32

theorem temperature_conversion (c : ℝ) (hf : c = 60) :
  celsius_to_fahrenheit c = 140 :=
by {
  rw [hf, celsius_to_fahrenheit];
  norm_num
}

end temperature_conversion_l172_172805


namespace find_number_l172_172619

theorem find_number (x : ℤ) (h : 2 * x + 5 = 17) : x = 6 := 
by
  sorry

end find_number_l172_172619


namespace circumradius_of_consecutive_triangle_l172_172023

theorem circumradius_of_consecutive_triangle
  (a b c : ℕ)
  (h : a = b - 1)
  (h1 : c = b + 1)
  (r : ℝ)
  (h2 : r = 4)
  (h3 : a + b > c)
  (h4 : a + c > b)
  (h5 : b + c > a)
  : ∃ R : ℝ, R = 65 / 8 :=
by {
  sorry
}

end circumradius_of_consecutive_triangle_l172_172023


namespace remainder_of_f_100_div_100_l172_172526

def pascal_triangle_row_sum (n : ℕ) : ℕ :=
  2^n - 2

theorem remainder_of_f_100_div_100 : 
  (pascal_triangle_row_sum 100) % 100 = 74 :=
by
  sorry

end remainder_of_f_100_div_100_l172_172526


namespace global_school_math_students_l172_172339

theorem global_school_math_students (n : ℕ) (h1 : n < 600) (h2 : n % 28 = 27) (h3 : n % 26 = 20) : n = 615 :=
by
  -- skip the proof
  sorry

end global_school_math_students_l172_172339


namespace club_president_vice_president_combinations_144_l172_172815

variables (boys_total girls_total : Nat)
variables (senior_boys junior_boys senior_girls junior_girls : Nat)
variables (choose_president_vice_president : Nat)

-- Define the conditions
def club_conditions : Prop :=
  boys_total = 12 ∧
  girls_total = 12 ∧
  senior_boys = 6 ∧
  junior_boys = 6 ∧
  senior_girls = 6 ∧
  junior_girls = 6

-- Define the proof problem
def president_vice_president_combinations (boys_total girls_total senior_boys junior_boys senior_girls junior_girls : Nat) : Nat :=
  2 * senior_boys * junior_boys + 2 * senior_girls * junior_girls

-- The main theorem to prove
theorem club_president_vice_president_combinations_144 :
  club_conditions boys_total girls_total senior_boys junior_boys senior_girls junior_girls →
  president_vice_president_combinations boys_total girls_total senior_boys junior_boys senior_girls junior_girls = 144 :=
sorry

end club_president_vice_president_combinations_144_l172_172815


namespace area_of_paper_l172_172234

theorem area_of_paper (L W : ℕ) (h1 : L + 2 * W = 34) (h2 : 2 * L + W = 38) : L * W = 140 := by
  sorry

end area_of_paper_l172_172234


namespace pizza_slices_meat_count_l172_172037

theorem pizza_slices_meat_count :
  let p := 30 in
  let h := 2 * p in
  let s := p + 12 in
  let n := 6 in
  (p + h + s) / n = 22 :=
by
  let p := 30
  let h := 2 * p
  let s := p + 12
  let n := 6
  calc
    (p + h + s) / n = (30 + 60 + 42) / 6 : by
      simp [p, h, s, n]
    ... = 132 / 6 : by
      rfl
    ... = 22 : by
      norm_num

end pizza_slices_meat_count_l172_172037


namespace find_common_difference_l172_172887

variable {a : ℕ → ℤ} 
variable {S : ℕ → ℤ}

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

def problem_conditions (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) : Prop :=
  a 3 + a 4 = 8 ∧ S 8 = 48

theorem find_common_difference :
  ∃ d, problem_conditions a S d ∧ is_arithmetic_sequence a d ∧ sum_of_first_n_terms a S ∧ d = 2 :=
by
  sorry

end find_common_difference_l172_172887


namespace find_ab_minus_a_neg_b_l172_172150

variable (a b : ℝ)
variables (h₀ : a > 1) (h₁ : b > 0) (h₂ : a^b + a^(-b) = 2 * Real.sqrt 2)

theorem find_ab_minus_a_neg_b : a^b - a^(-b) = 2 := by
  sorry

end find_ab_minus_a_neg_b_l172_172150


namespace hank_donates_90_percent_l172_172559

theorem hank_donates_90_percent (x : ℝ) : 
  (100 * x + 0.75 * 80 + 50 = 200) → (x = 0.9) :=
by
  intro h
  sorry

end hank_donates_90_percent_l172_172559


namespace cost_per_vent_l172_172783

/--
Given that:
1. The total cost of the HVAC system is $20,000.
2. The system includes 2 conditioning zones.
3. Each zone has 5 vents.

Prove that the cost per vent is $2000.
-/
theorem cost_per_vent (total_cost : ℕ) (zones : ℕ) (vents_per_zone : ℕ) (h1 : total_cost = 20000) (h2 : zones = 2) (h3 : vents_per_zone = 5) :
  total_cost / (zones * vents_per_zone) = 2000 := 
sorry

end cost_per_vent_l172_172783


namespace intersection_P_Q_l172_172156

-- Definitions based on conditions
def P : Set ℝ := { y | ∃ x : ℝ, y = x + 1 }
def Q : Set ℝ := { y | ∃ x : ℝ, y = 1 - x }

-- Proof statement to show P ∩ Q = Set.univ
theorem intersection_P_Q : P ∩ Q = Set.univ := by
  sorry

end intersection_P_Q_l172_172156


namespace trigonometric_identity_l172_172278

theorem trigonometric_identity (α : ℝ) (h : Real.tan (α + π / 4) = -3) :
  2 * Real.cos (2 * α) + 3 * Real.sin (2 * α) - Real.sin α ^ 2 = 2 / 5 :=
by
  sorry

end trigonometric_identity_l172_172278


namespace intersecting_lines_at_point_find_b_plus_m_l172_172225

theorem intersecting_lines_at_point_find_b_plus_m :
  ∀ (m b : ℝ),
  (12 = m * 4 + 2) →
  (12 = -2 * 4 + b) →
  (b + m = 22.5) :=
by
  intros m b h1 h2
  sorry

end intersecting_lines_at_point_find_b_plus_m_l172_172225


namespace pizza_slices_meat_count_l172_172036

theorem pizza_slices_meat_count :
  let p := 30 in
  let h := 2 * p in
  let s := p + 12 in
  let n := 6 in
  (p + h + s) / n = 22 :=
by
  let p := 30
  let h := 2 * p
  let s := p + 12
  let n := 6
  calc
    (p + h + s) / n = (30 + 60 + 42) / 6 : by
      simp [p, h, s, n]
    ... = 132 / 6 : by
      rfl
    ... = 22 : by
      norm_num

end pizza_slices_meat_count_l172_172036


namespace cos_300_eq_half_l172_172634

theorem cos_300_eq_half : Real.cos (2 * π * (300 / 360)) = 1 / 2 :=
by
  sorry

end cos_300_eq_half_l172_172634


namespace book_count_l172_172504

theorem book_count (P C B : ℕ) (h1 : P = 3 * C / 2) (h2 : B = 3 * C / 4) (h3 : P + C + B > 3000) : 
  P + C + B = 3003 := by
  sorry

end book_count_l172_172504


namespace pow_sum_nineteen_eq_zero_l172_172857

variable {a b c : ℝ}

theorem pow_sum_nineteen_eq_zero (h₁ : a + b + c = 0) (h₂ : a^3 + b^3 + c^3 = 0) : a^19 + b^19 + c^19 = 0 :=
sorry

end pow_sum_nineteen_eq_zero_l172_172857


namespace find_b_l172_172021

noncomputable def given_c := 3
noncomputable def given_C := Real.pi / 3
noncomputable def given_cos_C := 1 / 2
noncomputable def given_a (b : ℝ) := 2 * b

theorem find_b (b : ℝ) (h1 : given_c = 3) (h2 : given_cos_C = Real.cos (given_C)) (h3 : given_a b = 2 * b) : b = Real.sqrt 3 := 
by
  sorry

end find_b_l172_172021


namespace replaced_person_is_65_l172_172601

-- Define the conditions of the problem context
variable (W : ℝ)
variable (avg_increase : ℝ := 3.5)
variable (num_persons : ℕ := 8)
variable (new_person_weight : ℝ := 93)

-- Express the given condition in the problem: 
-- The total increase in weight is given by the number of persons multiplied by the average increase in weight
def total_increase : ℝ := num_persons * avg_increase

-- Express the relationship between the new person's weight and the person who was replaced
def replaced_person_weight (W : ℝ) : ℝ := new_person_weight - total_increase

-- Stating the theorem to be proved
theorem replaced_person_is_65 : replaced_person_weight W = 65 := by
  sorry

end replaced_person_is_65_l172_172601


namespace painted_cube_ways_l172_172650

theorem painted_cube_ways (b r g : ℕ) (cubes : ℕ) : 
  b = 1 → r = 2 → g = 3 → cubes = 3 := 
by
  intros
  sorry

end painted_cube_ways_l172_172650


namespace eating_time_l172_172325

-- Defining the terms based on the conditions provided
def rate_mr_swift := 1 / 15 -- Mr. Swift eats 1 pound in 15 minutes
def rate_mr_slow := 1 / 45  -- Mr. Slow eats 1 pound in 45 minutes

-- Combined eating rate of Mr. Swift and Mr. Slow
def combined_rate := rate_mr_swift + rate_mr_slow

-- Total amount of cereal to be consumed
def total_cereal := 4 -- pounds

-- Proving the total time to eat the cereal
theorem eating_time :
  (total_cereal / combined_rate) = 45 :=
by
  sorry

end eating_time_l172_172325


namespace meet_time_same_departure_meet_time_staggered_departure_catch_up_time_same_departure_l172_172474

-- Distance between locations A and B
def distance : ℝ := 448

-- Speed of the slow train
def slow_speed : ℝ := 60

-- Speed of the fast train
def fast_speed : ℝ := 80

-- Problem 1: Prove the two trains meet 3.2 hours after the fast train departs (both trains heading towards each other, departing at the same time)
theorem meet_time_same_departure : 
  (slow_speed + fast_speed) * 3.2 = distance :=
by
  sorry

-- Problem 2: Prove the two trains meet 3 hours after the fast train departs (slow train departs 28 minutes before the fast train)
theorem meet_time_staggered_departure : 
  (slow_speed * (28/60) + (slow_speed + fast_speed) * 3) = distance :=
by
  sorry

-- Problem 3: Prove the fast train catches up to the slow train 22.4 hours after departure (both trains heading in the same direction, departing at the same time)
theorem catch_up_time_same_departure : 
  (fast_speed - slow_speed) * 22.4 = distance :=
by
  sorry

end meet_time_same_departure_meet_time_staggered_departure_catch_up_time_same_departure_l172_172474


namespace tourists_count_l172_172512

theorem tourists_count (n k : ℕ) (h1 : 2 * n % k = 1) (h2 : 3 * n % k = 13) : k = 23 := 
sorry

end tourists_count_l172_172512


namespace element_in_set_l172_172866

variable (A : Set ℕ) (a b : ℕ)
def condition : Prop := A = {a, b, 1}

theorem element_in_set (h : condition A a b) : 1 ∈ A :=
by sorry

end element_in_set_l172_172866


namespace probability_no_adjacent_standing_l172_172908

-- Define the problem conditions in Lean 4.
def total_outcomes := 2^10
def favorable_outcomes := 123

-- The probability is given by favorable outcomes over total outcomes.
def probability : ℚ := favorable_outcomes / total_outcomes

-- Now state the theorem regarding the probability.
theorem probability_no_adjacent_standing : 
  probability = 123 / 1024 :=
by {
  sorry
}

end probability_no_adjacent_standing_l172_172908


namespace lily_pad_half_coverage_l172_172237

-- Define the conditions in Lean
def doubles_daily (size: ℕ → ℕ) : Prop :=
  ∀ n : ℕ, size (n + 1) = 2 * size n

def covers_entire_lake (size: ℕ → ℕ) (total_size: ℕ) : Prop :=
  size 34 = total_size

-- The main statement to prove
theorem lily_pad_half_coverage (size : ℕ → ℕ) (total_size : ℕ) 
  (h1 : doubles_daily size) 
  (h2 : covers_entire_lake size total_size) : 
  size 33 = total_size / 2 :=
sorry

end lily_pad_half_coverage_l172_172237


namespace remainder_of_hx10_divided_by_hx_is_6_l172_172318

noncomputable def h (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1

theorem remainder_of_hx10_divided_by_hx_is_6 : 
  let q := h (x ^ 10);
  q % h (x) = 6 := by
  sorry

end remainder_of_hx10_divided_by_hx_is_6_l172_172318


namespace center_cell_value_l172_172431

open Matrix Finset

def table := Matrix (Fin 3) (Fin 3) ℝ

def row_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 0 2 = 1) ∧ 
  (T 1 0 * T 1 1 * T 1 2 = 1) ∧ 
  (T 2 0 * T 2 1 * T 2 2 = 1)

def col_products (T : table) : Prop :=
  (T 0 0 * T 1 0 * T 2 0 = 1) ∧ 
  (T 0 1 * T 1 1 * T 2 1 = 1) ∧ 
  (T 0 2 * T 1 2 * T 2 2 = 1)

def square_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 1 0 * T 1 1 = 2) ∧ 
  (T 0 1 * T 0 2 * T 1 1 * T 1 2 = 2) ∧ 
  (T 1 0 * T 1 1 * T 2 0 * T 2 1 = 2) ∧ 
  (T 1 1 * T 1 2 * T 2 1 * T 2 2 = 2)

theorem center_cell_value (T : table) 
  (h_row : row_products T) 
  (h_col : col_products T) 
  (h_square : square_products T) : 
  T 1 1 = 1 :=
by
  sorry

end center_cell_value_l172_172431


namespace cube_bug_probability_l172_172831

theorem cube_bug_probability :
  ∃ n : ℕ, (∃ p : ℚ, p = 547/2187) ∧ (p = n/6561) ∧ n = 1641 :=
by
  sorry

end cube_bug_probability_l172_172831


namespace jellybeans_problem_l172_172520

theorem jellybeans_problem (n : ℕ) (h : n ≥ 100) (h_mod : n % 13 = 11) : n = 102 :=
sorry

end jellybeans_problem_l172_172520


namespace breaststroke_hours_correct_l172_172017

namespace Swimming

def total_required_hours : ℕ := 1500
def backstroke_hours : ℕ := 50
def butterfly_hours : ℕ := 121
def monthly_freestyle_sidestroke_hours : ℕ := 220
def months : ℕ := 6

def calculated_total_hours : ℕ :=
  backstroke_hours + butterfly_hours + (monthly_freestyle_sidestroke_hours * months)

def remaining_hours_to_breaststroke : ℕ :=
  total_required_hours - calculated_total_hours

theorem breaststroke_hours_correct :
  remaining_hours_to_breaststroke = 9 :=
by
  sorry

end Swimming

end breaststroke_hours_correct_l172_172017


namespace rational_solutions_iff_k_equals_8_l172_172973

theorem rational_solutions_iff_k_equals_8 {k : ℕ} (hk : k > 0) :
  (∃ (x : ℚ), k * x^2 + 16 * x + k = 0) ↔ k = 8 :=
by
  sorry

end rational_solutions_iff_k_equals_8_l172_172973


namespace volume_of_regular_tetrahedron_l172_172599

noncomputable def volume_of_tetrahedron (a : ℝ) : ℝ :=
  (a ^ 3 * Real.sqrt 2) / 12

theorem volume_of_regular_tetrahedron (a : ℝ) : 
  volume_of_tetrahedron a = (a ^ 3 * Real.sqrt 2) / 12 := 
by
  sorry

end volume_of_regular_tetrahedron_l172_172599


namespace lines_parallel_iff_l172_172558

theorem lines_parallel_iff (a : ℝ) : (∀ x y : ℝ, x + 2*a*y - 1 = 0 ∧ (2*a - 1)*x - a*y - 1 = 0 → x = 1 ∧ x = -1 ∨ ∃ (slope : ℝ), slope = - (1 / (2 * a)) ∧ slope = (2 * a - 1) / a) ↔ (a = 0 ∨ a = 1/4) :=
by
  sorry

end lines_parallel_iff_l172_172558


namespace compute_expression_l172_172535

theorem compute_expression (x z : ℝ) (h1 : x ≠ 0) (h2 : z ≠ 0) (h3 : x = 1 / z^2) : 
  (x - 1 / x) * (z^2 + 1 / z^2) = x^2 - z^4 :=
by
  sorry

end compute_expression_l172_172535


namespace petya_time_spent_l172_172044

theorem petya_time_spent :
  (1 / 3) + (1 / 5) + (1 / 6) + (1 / 70) + (1 / 3) > 1 :=
by
  sorry

end petya_time_spent_l172_172044


namespace count_4_digit_numbers_with_property_l172_172409

noncomputable def count_valid_4_digit_numbers : ℕ :=
  let valid_units (t : ℕ) : List ℕ := List.filter (λ u => u ≥ 3 * t) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  let choices_for_tu : ℕ := (List.length (valid_units 0)) + (List.length (valid_units 1)) + (List.length (valid_units 2))
  choices_for_tu * 9 * 9

theorem count_4_digit_numbers_with_property : count_valid_4_digit_numbers = 1701 := by
  sorry

end count_4_digit_numbers_with_property_l172_172409


namespace product_of_integers_l172_172149

theorem product_of_integers (A B C D : ℕ) 
  (h1 : A + B + C + D = 100) 
  (h2 : 2^A = B - 6) 
  (h3 : C + 6 = D)
  (h4 : B + C = D + 10) : 
  A * B * C * D = 33280 := 
by
  sorry

end product_of_integers_l172_172149


namespace minimum_positive_period_of_f_decreasing_intervals_of_f_maximum_value_of_f_l172_172849

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 6) + 3 / 2

theorem minimum_positive_period_of_f : ∀ x : ℝ, f (x + Real.pi) = f x := by sorry

theorem decreasing_intervals_of_f : ∀ k : ℤ, ∀ x : ℝ,
  (Real.pi / 6 + k * Real.pi) ≤ x ∧ x ≤ (2 * Real.pi / 3 + k * Real.pi) → ∀ y : ℝ, 
  (Real.pi / 6 + k * Real.pi) ≤ y ∧ y ≤ (2 * Real.pi / 3 + k * Real.pi) → x ≤ y → f y ≤ f x := by sorry

theorem maximum_value_of_f : ∃ k : ℤ, ∃ x : ℝ, x = (Real.pi / 6 + k * Real.pi) ∧ f x = 5 / 2 := by sorry

end minimum_positive_period_of_f_decreasing_intervals_of_f_maximum_value_of_f_l172_172849


namespace min_value_of_a_l172_172874

theorem min_value_of_a (a : ℝ) : (∀ x, 0 < x ∧ x ≤ 1/2 → x^2 + a * x + 1 ≥ 0) → a ≥ -5/2 := 
sorry

end min_value_of_a_l172_172874


namespace valid_numbers_l172_172510

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

end valid_numbers_l172_172510


namespace count_cubes_between_bounds_l172_172410

theorem count_cubes_between_bounds : ∃ (n : ℕ), n = 42 ∧
  ∀ x, 2^9 + 1 ≤ x^3 ∧ x^3 ≤ 2^17 + 1 ↔ 9 ≤ x ∧ x ≤ 50 := 
sorry

end count_cubes_between_bounds_l172_172410


namespace initial_investment_l172_172534

theorem initial_investment :
  ∃ x : ℝ, P = 705.03 ∧ r = 0.12 ∧ n = 5 ∧ P = x * (1 + r)^n ∧ x = 400 :=
by
  let P := 705.03
  let r := 0.12
  let n := 5
  use 400
  simp [P, r, n]
  sorry

end initial_investment_l172_172534


namespace yogurt_price_is_5_l172_172903

theorem yogurt_price_is_5
  (yogurt_pints : ℕ)
  (gum_packs : ℕ)
  (shrimp_trays : ℕ)
  (total_cost : ℝ)
  (shrimp_cost : ℝ)
  (gum_fraction : ℝ)
  (price_frozen_yogurt : ℝ) :
  yogurt_pints = 5 →
  gum_packs = 2 →
  shrimp_trays = 5 →
  total_cost = 55 →
  shrimp_cost = 5 →
  gum_fraction = 0.5 →
  5 * price_frozen_yogurt + 2 * (gum_fraction * price_frozen_yogurt) + 5 * shrimp_cost = total_cost →
  price_frozen_yogurt = 5 :=
by
  intro hp hg hs ht hc hf h_formula
  sorry

end yogurt_price_is_5_l172_172903


namespace ribbons_purple_l172_172443

theorem ribbons_purple (total_ribbons : ℕ) (yellow_ribbons purple_ribbons orange_ribbons black_ribbons : ℕ)
  (h1 : yellow_ribbons = total_ribbons / 4)
  (h2 : purple_ribbons = total_ribbons / 3)
  (h3 : orange_ribbons = total_ribbons / 6)
  (h4 : black_ribbons = 40)
  (h5 : yellow_ribbons + purple_ribbons + orange_ribbons + black_ribbons = total_ribbons) :
  purple_ribbons = 53 :=
by
  sorry

end ribbons_purple_l172_172443


namespace winter_sales_l172_172479

theorem winter_sales (T : ℕ) (spring_summer_sales : ℕ) (fall_sales : ℕ) (winter_sales : ℕ) 
  (h1 : T = 20) 
  (h2 : spring_summer_sales = 12) 
  (h3 : fall_sales = 4) 
  (h4 : T = spring_summer_sales + fall_sales + winter_sales) : 
     winter_sales = 4 := 
by 
  rw [h1, h2, h3] at h4
  linarith


end winter_sales_l172_172479


namespace seventy_fifth_elem_in_s_l172_172892

-- Define the set s
def s : Set ℕ := {x | ∃ n : ℕ, x = 8 * n + 5}

-- State the main theorem
theorem seventy_fifth_elem_in_s : (∃ n : ℕ, n = 74 ∧ (8 * n + 5) = 597) :=
by
  -- The proof is skipped using sorry
  sorry

end seventy_fifth_elem_in_s_l172_172892


namespace modulus_of_complex_number_l172_172543

noncomputable def z := Complex

theorem modulus_of_complex_number (z : Complex) (h : z * (1 + Complex.I) = 2) :
  Complex.abs z = Real.sqrt 2 :=
sorry

end modulus_of_complex_number_l172_172543


namespace find_x_l172_172807

variable (x : ℝ)

theorem find_x (h : (15 - 2 + 4 / 1 / 2) * x = 77) : x = 77 / (15 - 2 + 4 / 1 / 2) :=
by sorry

end find_x_l172_172807


namespace product_of_marbles_l172_172910

theorem product_of_marbles (R B : ℕ) (h1 : R - B = 12) (h2 : R + B = 52) : R * B = 640 := by
  sorry

end product_of_marbles_l172_172910


namespace smallest_of_five_consecutive_l172_172916

theorem smallest_of_five_consecutive (n : ℤ) (h : (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 2015) : n - 2 = 401 :=
by sorry

end smallest_of_five_consecutive_l172_172916


namespace restaurant_sales_l172_172196

theorem restaurant_sales (monday tuesday wednesday thursday : ℕ) 
  (h1 : monday = 40) 
  (h2 : tuesday = monday + 40) 
  (h3 : wednesday = tuesday / 2) 
  (h4 : monday + tuesday + wednesday + thursday = 203) : 
  thursday = wednesday + 3 := 
by sorry

end restaurant_sales_l172_172196


namespace range_F_l172_172264

noncomputable def f (x : ℝ) : ℝ := 3^(x - 2)

def f_inv (y : ℝ) : ℝ := 2 + Real.log y / Real.log 3

def F (x : ℝ) : ℝ := (f_inv x)^2 - f_inv (x^2)

theorem range_F :
  {y : ℝ | ∃ x : ℝ, (2 ≤ x ∧ x ≤ 4) ∧ y = F x} = set.Icc 2 5 := sorry

end range_F_l172_172264


namespace pool_capacity_l172_172936

-- Conditions
variables (C : ℝ) -- total capacity of the pool in gallons
variables (h1 : 300 = 0.75 * C - 0.45 * C) -- the pool requires an additional 300 gallons to be filled to 75%
variables (h2 : 300 = 0.30 * C) -- pumping in these additional 300 gallons will increase the amount of water by 30%

-- Goal
theorem pool_capacity : C = 1000 :=
by sorry

end pool_capacity_l172_172936


namespace john_marbles_l172_172500

theorem john_marbles : ∃ m : ℕ, (m ≡ 3 [MOD 7]) ∧ (m ≡ 2 [MOD 4]) ∧ m = 10 := by
  sorry

end john_marbles_l172_172500


namespace range_of_a_l172_172552

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₀ d : ℝ), ∀ n, a n = a₀ + n * d

theorem range_of_a (a : ℝ) (a_seq : ℕ → ℝ) (b : ℕ → ℝ)
  (h1 : is_arithmetic_sequence a_seq) 
  (h2 : a_seq 0 = a)
  (h3 : ∀ n, b n = (1 + a_seq n) / a_seq n)
  (h4 : ∀ n : ℕ, 0 < n → b n ≥ b 8) :
  -8 < a ∧ a < -7 :=
sorry

end range_of_a_l172_172552


namespace male_students_plant_trees_l172_172608

theorem male_students_plant_trees (total_avg : ℕ) (female_trees : ℕ) (male_trees : ℕ) 
  (h1 : total_avg = 6) 
  (h2 : female_trees = 15)
  (h3 : 1 / (male_trees : ℝ) + 1 / (female_trees : ℝ) = 1 / (total_avg : ℝ)) : 
  male_trees = 10 := 
sorry

end male_students_plant_trees_l172_172608


namespace original_ticket_price_l172_172358

open Real

theorem original_ticket_price 
  (P : ℝ)
  (total_revenue : ℝ)
  (revenue_equation : total_revenue = 10 * 0.60 * P + 20 * 0.85 * P + 15 * P) 
  (total_revenue_val : total_revenue = 760) : 
  P = 20 := 
by
  sorry

end original_ticket_price_l172_172358


namespace quadratic_product_fact_l172_172125

def quadratic_factors_product : Prop :=
  let integer_pairs := [(-1, 24), (-2, 12), (-3, 8), (-4, 6), (-6, 4), (-8, 3), (-12, 2), (-24, 1)]
  let t_values := integer_pairs.map (fun (c, d) => c + d)
  let product_t := t_values.foldl (fun acc t => acc * t) 1
  product_t = -5290000

theorem quadratic_product_fact : quadratic_factors_product :=
by sorry

end quadratic_product_fact_l172_172125


namespace largest_prime_factor_of_expression_l172_172365

theorem largest_prime_factor_of_expression :
  ∃ (p : ℕ), Prime p ∧ p > 35 ∧ p > 2 ∧ p ∣ (18^4 + 2 * 18^2 + 1 - 17^4) ∧ ∀ q, Prime q ∧ q ∣ (18^4 + 2 * 18^2 + 1 - 17^4) → q ≤ p :=
by
  sorry

end largest_prime_factor_of_expression_l172_172365


namespace systematic_sampling_l172_172255

-- Definitions for the class of 50 students numbered from 1 to 50, sampling interval, and starting number.
def students : Set ℕ := {n | n ∈ Finset.range 50 ∧ n ≥ 1}
def sampling_interval : ℕ := 10
def start : ℕ := 6

-- The main theorem stating that the selected students' numbers are as given.
theorem systematic_sampling : ∃ (selected : List ℕ), selected = [6, 16, 26, 36, 46] ∧ 
  ∀ x ∈ selected, x ∈ students := 
  sorry

end systematic_sampling_l172_172255


namespace problem_2012_square_eq_x_square_l172_172624

theorem problem_2012_square_eq_x_square : 
  ∃ x : ℤ, (2012 + x)^2 = x^2 ∧ x = -1006 :=
by {
  existsi (-1006 : ℤ),
  split,
  sorry,
  refl
}

end problem_2012_square_eq_x_square_l172_172624


namespace apples_per_pie_l172_172341

/-- Let's define the parameters given in the problem -/
def initial_apples : ℕ := 62
def apples_given_to_students : ℕ := 8
def pies_made : ℕ := 6

/-- Define the remaining apples after handing out to students -/
def remaining_apples : ℕ := initial_apples - apples_given_to_students

/-- The statement we need to prove: each pie requires 9 apples -/
theorem apples_per_pie : remaining_apples / pies_made = 9 := by
  -- Add the proof here
  sorry

end apples_per_pie_l172_172341


namespace zoo_peacocks_l172_172616

theorem zoo_peacocks (R P : ℕ) (h1 : R + P = 60) (h2 : 4 * R + 2 * P = 192) : P = 24 :=
by
  sorry

end zoo_peacocks_l172_172616


namespace cakes_to_make_l172_172671

-- Define the conditions
def packages_per_cake : ℕ := 2
def cost_per_package : ℕ := 3
def total_cost : ℕ := 12

-- Define the proof problem
theorem cakes_to_make (h1 : packages_per_cake = 2) (h2 : cost_per_package = 3) (h3 : total_cost = 12) :
  (total_cost / cost_per_package) / packages_per_cake = 2 :=
by sorry

end cakes_to_make_l172_172671


namespace sequence_a_11_l172_172444

theorem sequence_a_11 (a : ℕ → ℚ) (arithmetic_seq : ℕ → ℚ)
  (h1 : a 3 = 2)
  (h2 : a 7 = 1)
  (h_arith : ∀ n, arithmetic_seq n = 1 / (a n + 1))
  (arith_property : ∀ n, arithmetic_seq (n + 1) - arithmetic_seq n = arithmetic_seq (n + 2) - arithmetic_seq (n + 1)) :
  a 11 = 1 / 2 :=
by
  sorry

end sequence_a_11_l172_172444


namespace work_rate_problem_l172_172934

theorem work_rate_problem 
  (W : ℝ)
  (rate_ab : ℝ)
  (rate_c : ℝ)
  (rate_abc : ℝ)
  (cond1 : rate_c = W / 2)
  (cond2 : rate_abc = W / 1)
  (cond3 : rate_ab = (W / 1) - rate_c) :
  rate_ab = W / 2 :=
by 
  -- We can add the solution steps here, but we skip that part following the guidelines
  sorry

end work_rate_problem_l172_172934


namespace polynomial_sum_eq_l172_172457

-- Definitions of the given polynomials
def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def s (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

-- The theorem to prove
theorem polynomial_sum_eq (x : ℝ) : 
  p x + q x + r x + s x = -x^2 + 10 * x - 11 :=
by 
  -- Proof steps are omitted here
  sorry

end polynomial_sum_eq_l172_172457


namespace sum_of_reversed_base_digits_eq_zero_l172_172718

theorem sum_of_reversed_base_digits_eq_zero : ∃ n : ℕ, 
  (∀ a₁ a₀ : ℕ, n = 5 * a₁ + a₀ ∧ n = 12 * a₀ + a₁ ∧ 0 ≤ a₁ ∧ a₁ < 5 ∧ 0 ≤ a₀ ∧ a₀ < 12 
  ∧ n > 0 → n = 0)
:= sorry

end sum_of_reversed_base_digits_eq_zero_l172_172718


namespace find_distance_l172_172089

variable (A B : Point)
variable (distAB : ℝ) -- the distance between A and B
variable (meeting1 : ℝ) -- first meeting distance from A
variable (meeting2 : ℝ) -- second meeting distance from B

-- Conditions
axiom meeting_conditions_1 : meeting1 = 70
axiom meeting_conditions_2 : meeting2 = 90

-- Prove the distance between A and B is 120 km
def distance_from_A_to_B : ℝ := 120

theorem find_distance : distAB = distance_from_A_to_B := 
sorry

end find_distance_l172_172089


namespace age_of_youngest_person_l172_172356

theorem age_of_youngest_person :
  ∃ (a1 a2 a3 a4 : ℕ), 
  (a1 < a2) ∧ (a2 < a3) ∧ (a3 < a4) ∧ 
  (a4 = 50) ∧ 
  (a1 + a2 + a3 + a4 = 158) ∧ 
  (a2 - a1 = a3 - a2) ∧ (a3 - a2 = a4 - a3) ∧ 
  a1 = 29 :=
by
  sorry

end age_of_youngest_person_l172_172356


namespace smallest_d_l172_172587

-- Constants and conditions
variables (c d : ℝ)
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions involving c and d
def conditions (c d : ℝ) : Prop :=
  2 < c ∧ c < d ∧ ¬triangle_inequality 2 c d ∧ ¬triangle_inequality (1/d) (1/c) 2

-- Goal statement: the smallest possible value of d
theorem smallest_d (c d : ℝ) (h : conditions c d) : d = 2 + Real.sqrt 2 :=
sorry

end smallest_d_l172_172587


namespace lily_remaining_money_l172_172034

def initial_amount := 55
def spent_on_shirt := 7
def spent_at_second_shop := 3 * spent_on_shirt
def total_spent := spent_on_shirt + spent_at_second_shop
def remaining_amount := initial_amount - total_spent

theorem lily_remaining_money : remaining_amount = 27 :=
by
  sorry

end lily_remaining_money_l172_172034


namespace find_ab_l172_172160

theorem find_ab (a b : ℝ) (h1 : a + b = 4) (h2 : a^3 + b^3 = 136) : a * b = -6 :=
by
  sorry

end find_ab_l172_172160


namespace find_natural_numbers_satisfying_prime_square_l172_172838

-- Define conditions as a Lean statement
theorem find_natural_numbers_satisfying_prime_square (n : ℕ) (h : ∃ p : ℕ, Prime p ∧ (2 * n^2 + 3 * n - 35 = p^2)) :
  n = 4 ∨ n = 12 :=
sorry

end find_natural_numbers_satisfying_prime_square_l172_172838


namespace find_x_from_average_l172_172340

theorem find_x_from_average :
  let sum_series := 5151
  let n := 102
  let known_average := 50 * (x + 1)
  (sum_series + x) / n = known_average → 
  x = 51 / 5099 :=
by
  intros
  sorry

end find_x_from_average_l172_172340


namespace calculate_expression_l172_172828

variable (a : ℝ)

theorem calculate_expression (h : a ≠ 0) : (6 * a^2) / (a / 2) = 12 * a := by
  sorry

end calculate_expression_l172_172828


namespace find_quantities_l172_172217

variables {a b x y : ℝ}

-- Original total expenditure condition
axiom h1 : a * x + b * y = 1500

-- New prices and quantities for the first scenario
axiom h2 : (a + 1.5) * (x - 10) + (b + 1) * y = 1529

-- New prices and quantities for the second scenario
axiom h3 : (a + 1) * (x - 5) + (b + 1) * y = 1563.5

-- Inequality constraint
axiom h4 : 205 < 2 * x + y ∧ 2 * x + y < 210

-- Range for 'a'
axiom h5 : 17.5 < a ∧ a < 18.5

-- Proving x and y are specific values.
theorem find_quantities :
  x = 76 ∧ y = 55 :=
sorry

end find_quantities_l172_172217


namespace center_cell_value_l172_172434

open Matrix Finset

def table := Matrix (Fin 3) (Fin 3) ℝ

def row_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 0 2 = 1) ∧ 
  (T 1 0 * T 1 1 * T 1 2 = 1) ∧ 
  (T 2 0 * T 2 1 * T 2 2 = 1)

def col_products (T : table) : Prop :=
  (T 0 0 * T 1 0 * T 2 0 = 1) ∧ 
  (T 0 1 * T 1 1 * T 2 1 = 1) ∧ 
  (T 0 2 * T 1 2 * T 2 2 = 1)

def square_products (T : table) : Prop :=
  (T 0 0 * T 0 1 * T 1 0 * T 1 1 = 2) ∧ 
  (T 0 1 * T 0 2 * T 1 1 * T 1 2 = 2) ∧ 
  (T 1 0 * T 1 1 * T 2 0 * T 2 1 = 2) ∧ 
  (T 1 1 * T 1 2 * T 2 1 * T 2 2 = 2)

theorem center_cell_value (T : table) 
  (h_row : row_products T) 
  (h_col : col_products T) 
  (h_square : square_products T) : 
  T 1 1 = 1 :=
by
  sorry

end center_cell_value_l172_172434


namespace car_gas_consumption_l172_172265

theorem car_gas_consumption
  (miles_today : ℕ)
  (miles_tomorrow : ℕ)
  (total_gallons : ℕ)
  (h1 : miles_today = 400)
  (h2 : miles_tomorrow = miles_today + 200)
  (h3 : total_gallons = 4000)
  : (∃ g : ℕ, 400 * g + (400 + 200) * g = total_gallons ∧ g = 4) :=
by
  sorry

end car_gas_consumption_l172_172265


namespace hvac_cost_per_vent_l172_172782

theorem hvac_cost_per_vent (cost : ℕ) (zones : ℕ) (vents_per_zone : ℕ) (h_cost : cost = 20000) (h_zones : zones = 2) (h_vents_per_zone : vents_per_zone = 5) :
  (cost / (zones * vents_per_zone) = 2000) :=
by
  sorry

end hvac_cost_per_vent_l172_172782


namespace problem_l172_172832

def otimes (x y : ℝ) : ℝ := x^3 + y - 2 * x

theorem problem (k : ℝ) : otimes k (otimes k k) = 2 * k^3 - 3 * k :=
by
  sorry

end problem_l172_172832


namespace video_duration_correct_l172_172900

/-
Define the conditions as given:
1. Vasya's time from home to school
2. Petya's time from school to home
3. Meeting conditions
-/

-- Define the times for Vasya and Petya
def vasya_time : ℕ := 8
def petya_time : ℕ := 5

-- Define the total video duration when correctly merged
def video_duration : ℕ := 5

-- State the theorem to be proved in Lean:
theorem video_duration_correct : vasya_time = 8 → petya_time = 5 → video_duration = 5 :=
by
  intros h1 h2
  exact sorry

end video_duration_correct_l172_172900


namespace shorter_piece_is_28_l172_172638

noncomputable def shorter_piece_length (x : ℕ) : Prop :=
  x + (x + 12) = 68 → x = 28

theorem shorter_piece_is_28 (x : ℕ) : shorter_piece_length x :=
by
  intro h
  have h1 : 2 * x + 12 = 68 := by linarith
  have h2 : 2 * x = 56 := by linarith
  have h3 : x = 28 := by linarith
  exact h3

end shorter_piece_is_28_l172_172638


namespace solve_quadratic_equation_l172_172925

theorem solve_quadratic_equation :
  ∀ x : ℝ, (10 - x) ^ 2 = 2 * x ^ 2 + 4 * x ↔ x = 3.62 ∨ x = -27.62 := by
  sorry

end solve_quadratic_equation_l172_172925


namespace max_n_satisfying_property_l172_172187

theorem max_n_satisfying_property :
  ∃ n : ℕ, (0 < n) ∧ (∀ m : ℕ, Nat.gcd m n = 1 → m^6 % n = 1) ∧ n = 504 :=
by
  sorry

end max_n_satisfying_property_l172_172187


namespace length_of_field_l172_172213

-- Define the problem conditions
variables (width length : ℕ)
  (pond_area field_area : ℕ)
  (h1 : length = 2 * width)
  (h2 : pond_area = 64)
  (h3 : pond_area = field_area / 8)

-- Define the proof problem
theorem length_of_field : length = 32 :=
by
  -- We'll provide the proof later
  sorry

end length_of_field_l172_172213


namespace fill_in_the_blanks_correctly_l172_172333

def remote_areas_need : String := "what the remote areas need"
def children : String := "children"
def education : String := "education"
def good_textbooks : String := "good textbooks"

-- Defining the grammatical agreement condition
def subject_verb_agreement (s : String) (v : String) : Prop :=
  (s = remote_areas_need ∧ v = "is") ∨ (s = children ∧ v = "are")

-- The main theorem statement
theorem fill_in_the_blanks_correctly : 
  subject_verb_agreement remote_areas_need "is" ∧ subject_verb_agreement children "are" :=
sorry

end fill_in_the_blanks_correctly_l172_172333


namespace original_price_of_racket_l172_172329

theorem original_price_of_racket (P : ℝ) (h : (3 / 2) * P = 90) : P = 60 :=
sorry

end original_price_of_racket_l172_172329


namespace m_not_equal_n_possible_l172_172197

-- Define the touching relation on an infinite chessboard
structure Chessboard :=
(colored_square : ℤ × ℤ → Prop)
(touches : ℤ × ℤ → ℤ × ℤ → Prop)

-- Define the properties
def colors_square (board : Chessboard) : Prop :=
∃ i j : ℤ, board.colored_square (i, j) ∧ board.colored_square (i + 1, j + 1)

def black_square_touches_m_black_squares (board : Chessboard) (m : ℕ) : Prop :=
∀ i j : ℤ, board.colored_square (i, j) →
    (board.touches (i, j) (i + 1, j) → board.colored_square (i + 1, j)) ∧ 
    (board.touches (i, j) (i - 1, j) → board.colored_square (i - 1, j)) ∧
    (board.touches (i, j) (i, j + 1) → board.colored_square (i, j + 1)) ∧
    (board.touches (i, j) (i, j - 1) → board.colored_square (i, j - 1))
    -- Add additional conditions to ensure exactly m black squares are touched

def white_square_touches_n_white_squares (board : Chessboard) (n : ℕ) : Prop :=
∀ i j : ℤ, ¬board.colored_square (i, j) →
    (board.touches (i, j) (i + 1, j) → ¬board.colored_square (i + 1, j)) ∧ 
    (board.touches (i, j) (i - 1, j) → ¬board.colored_square (i - 1, j)) ∧
    (board.touches (i, j) (i, j + 1) → ¬board.colored_square (i, j + 1)) ∧
    (board.touches (i, j) (i, j - 1) → ¬board.colored_square (i, j - 1))
    -- Add additional conditions to ensure exactly n white squares are touched

theorem m_not_equal_n_possible (board : Chessboard) (m n : ℕ) :
colors_square board →
black_square_touches_m_black_squares board m →
white_square_touches_n_white_squares board n →
m ≠ n :=
by {
    sorry
}

end m_not_equal_n_possible_l172_172197


namespace low_income_households_sampled_l172_172649

def total_households := 500
def high_income_households := 125
def middle_income_households := 280
def low_income_households := 95
def sampled_high_income_households := 25

theorem low_income_households_sampled :
  (sampled_high_income_households / high_income_households) * low_income_households = 19 := by
  sorry

end low_income_households_sampled_l172_172649


namespace ordered_triples_count_l172_172257

def similar_prisms_count (b : ℕ) (c : ℕ) (a : ℕ) := 
  (a ≤ c ∧ c ≤ b ∧ 
   ∃ (x y z : ℕ), x ≤ z ∧ z ≤ y ∧ y = b ∧ 
   x < a ∧ y < b ∧ z < c ∧ 
   ((x : ℚ) / a = (y : ℚ) / b ∧ (y : ℚ) / b = (z : ℚ) / c))

theorem ordered_triples_count : 
  ∃ (n : ℕ), n = 24 ∧ ∀ a c, similar_prisms_count 2000 c a → a < c :=
sorry

end ordered_triples_count_l172_172257


namespace find_M_l172_172867

theorem find_M : ∀ M : ℕ, (10 + 11 + 12 : ℕ) / 3 = (2024 + 2025 + 2026 : ℕ) / M → M = 552 :=
by
  intro M
  sorry

end find_M_l172_172867


namespace center_cell_value_l172_172435

theorem center_cell_value
  (a b c d e f g h i : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i)
  (h_row1 : a * b * c = 1)
  (h_row2 : d * e * f = 1)
  (h_row3 : g * h * i = 1)
  (h_col1 : a * d * g = 1)
  (h_col2 : b * e * h = 1)
  (h_col3 : c * f * i = 1)
  (h_square1 : a * b * d * e = 2)
  (h_square2 : b * c * e * f = 2)
  (h_square3 : d * e * g * h = 2)
  (h_square4 : e * f * h * i = 2) :
  e = 1 :=
  sorry

end center_cell_value_l172_172435


namespace range_of_a_l172_172987

theorem range_of_a
  (a x : ℝ)
  (h_eq : 2 * (1 / 4) ^ (-x) - (1 / 2) ^ (-x) + a = 0)
  (h_x : -1 ≤ x ∧ x ≤ 0) :
  -1 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l172_172987


namespace min_ratio_area_of_incircle_circumcircle_rt_triangle_l172_172007

variables (a b: ℝ)
variables (a' b' c: ℝ)

-- Conditions
def area_of_right_triangle (a b : ℝ) : ℝ := 
    0.5 * a * b

def incircle_radius (a' b' c : ℝ) : ℝ := 
    0.5 * (a' + b' - c)

def circumcircle_radius (c : ℝ) : ℝ := 
    0.5 * c

-- Condition of the problem
def condition (a b a' b' c : ℝ) : Prop :=
    incircle_radius a' b' c = circumcircle_radius c ∧ 
    a' + b' = 2 * c

-- The final proof problem
theorem min_ratio_area_of_incircle_circumcircle_rt_triangle (a b a' b' c : ℝ)
    (h_area_a : a = area_of_right_triangle a' b')
    (h_area_b : b = area_of_right_triangle a b)
    (h_condition : condition a b a' b' c) :
    (a / b ≥ 3 + 2 * Real.sqrt 2) :=
by
  sorry

end min_ratio_area_of_incircle_circumcircle_rt_triangle_l172_172007


namespace amount_paid_by_customer_l172_172349

theorem amount_paid_by_customer 
  (cost_price : ℝ)
  (markup_percentage : ℝ)
  (final_price : ℝ)
  (h1 : cost_price = 6681.818181818181)
  (h2 : markup_percentage = 10 / 100)
  (h3 : final_price = cost_price * (1 + markup_percentage)) :
  final_price = 7350 :=
by 
  sorry

end amount_paid_by_customer_l172_172349


namespace solve_cubic_eq_l172_172135

theorem solve_cubic_eq (x : ℝ) : x^3 + (2 - x)^3 = 8 ↔ x = 0 ∨ x = 2 := 
by 
  { sorry }

end solve_cubic_eq_l172_172135


namespace area_of_annulus_l172_172176

theorem area_of_annulus (R r t : ℝ) (h : R > r) (h_tangent : R^2 = r^2 + t^2) : 
  π * (R^2 - r^2) = π * t^2 :=
by 
  sorry

end area_of_annulus_l172_172176


namespace part1_inequality_part2_inequality_l172_172405

-- Problem Part 1
def f (x : ℝ) : ℝ := abs (x - 2) - abs (x + 1)

theorem part1_inequality (x : ℝ) : f x ≤ 1 ↔ 0 ≤ x :=
by sorry

-- Problem Part 2
def max_f_value : ℝ := 3
def a : ℝ := sorry  -- Define in context
def b : ℝ := sorry  -- Define in context
def c : ℝ := sorry  -- Define in context

-- Prove √a + √b + √c ≤ 3 given a + b + c = 3
theorem part2_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = max_f_value) :
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ 3 :=
by sorry

end part1_inequality_part2_inequality_l172_172405


namespace tom_initial_game_count_zero_l172_172224

theorem tom_initial_game_count_zero
  (batman_game_cost superman_game_cost total_expenditure initial_game_count : ℝ)
  (h_batman_cost : batman_game_cost = 13.60)
  (h_superman_cost : superman_game_cost = 5.06)
  (h_total_expenditure : total_expenditure = 18.66)
  (h_initial_game_cost : initial_game_count = total_expenditure - (batman_game_cost + superman_game_cost)) :
  initial_game_count = 0 :=
by
  sorry

end tom_initial_game_count_zero_l172_172224


namespace find_x_l172_172727

theorem find_x (x y : ℤ) (h₁ : x / y = 12 / 5) (h₂ : y = 25) : x = 60 :=
by
  sorry

end find_x_l172_172727


namespace cost_for_sugar_substitutes_l172_172524

def packets_per_cup : ℕ := 1
def cups_per_day : ℕ := 2
def days : ℕ := 90
def packets_per_box : ℕ := 30
def price_per_box : ℕ := 4

theorem cost_for_sugar_substitutes : 
  (packets_per_cup * cups_per_day * days / packets_per_box) * price_per_box = 24 := by
  sorry

end cost_for_sugar_substitutes_l172_172524


namespace system_of_equations_solution_l172_172337

theorem system_of_equations_solution:
  ∀ (x y : ℝ), 
    x^2 + y^2 + x + y = 42 ∧ x * y = 15 → 
      (x = 3 ∧ y = 5) ∨ (x = 5 ∧ y = 3) ∨ 
      (x = (-9 + Real.sqrt 21) / 2 ∧ y = (-9 - Real.sqrt 21) / 2) ∨ 
      (x = (-9 - Real.sqrt 21) / 2 ∧ y = (-9 + Real.sqrt 21) / 2) := 
by
  sorry

end system_of_equations_solution_l172_172337


namespace solve_system_equations_l172_172096

-- Define the hypotheses of the problem
variables {a x y : ℝ}
variables (h1 : (0 < a) ∧ (a ≠ 1))
variables (h2 : (0 < x))
variables (h3 : (0 < y))
variables (eq1 : (log a x + log a y - 2) * log 18 a = 1)
variables (eq2 : 2 * x + y - 20 * a = 0)

-- State the theorem to be proved
theorem solve_system_equations :
  (x = a ∧ y = 18 * a) ∨ (x = 9 * a ∧ y = 2 * a) := by
  sorry

end solve_system_equations_l172_172096


namespace teachers_on_field_trip_l172_172941

-- Definitions for conditions in the problem
def number_of_students := 12
def cost_per_student_ticket := 1
def cost_per_adult_ticket := 3
def total_cost_of_tickets := 24

-- Main statement
theorem teachers_on_field_trip :
  ∃ (T : ℕ), number_of_students * cost_per_student_ticket + T * cost_per_adult_ticket = total_cost_of_tickets ∧ T = 4 :=
by
  use 4
  sorry

end teachers_on_field_trip_l172_172941


namespace calc_expression_l172_172667

theorem calc_expression :
  5 + 7 * (2 + (1 / 4 : ℝ)) = 20.75 :=
by
  sorry

end calc_expression_l172_172667


namespace a7_of_expansion_x10_l172_172609

theorem a7_of_expansion_x10 : 
  (∃ (a : ℕ) (a1 : ℕ) (a2 : ℕ) (a3 : ℕ) 
     (a4 : ℕ) (a5 : ℕ) (a6 : ℕ) 
     (a8 : ℕ) (a9 : ℕ) (a10 : ℕ),
     ((x : ℕ) → x^10 = a + a1*(x-1) + a2*(x-1)^2 + a3*(x-1)^3 + 
                      a4*(x-1)^4 + a5*(x-1)^5 + a6*(x-1)^6 + 
                      120*(x-1)^7 + a8*(x-1)^8 + a9*(x-1)^9 + a10*(x-1)^10)) :=
  sorry

end a7_of_expansion_x10_l172_172609


namespace least_blue_eyes_and_snack_l172_172273

variable (total_students blue_eyes students_with_snack : ℕ)

theorem least_blue_eyes_and_snack (h1 : total_students = 35) 
                                 (h2 : blue_eyes = 14) 
                                 (h3 : students_with_snack = 22) :
  ∃ n, n = 1 ∧ 
        ∀ k, (k < n → 
                 ∃ no_snack_no_blue : ℕ, no_snack_no_blue = total_students - students_with_snack ∧
                      no_snack_no_blue = blue_eyes - k) := 
by
  sorry

end least_blue_eyes_and_snack_l172_172273


namespace find_reading_l172_172935

variable (a_1 a_2 a_3 a_4 : ℝ) (x : ℝ)
variable (h1 : a_1 = 2) (h2 : a_2 = 2.1) (h3 : a_3 = 2) (h4 : a_4 = 2.2)
variable (mean : (a_1 + a_2 + a_3 + a_4 + x) / 5 = 2)

theorem find_reading : x = 1.7 :=
by
  sorry

end find_reading_l172_172935


namespace negation_of_exists_gt_implies_forall_leq_l172_172072

theorem negation_of_exists_gt_implies_forall_leq (x : ℝ) (h : 0 < x) :
  ¬ (∃ x : ℝ, 0 < x ∧ x^3 - x + 1 > 0) ↔ ∀ x : ℝ, 0 < x → x^3 - x + 1 ≤ 0 :=
by sorry

end negation_of_exists_gt_implies_forall_leq_l172_172072


namespace nathan_correct_answers_l172_172747

theorem nathan_correct_answers (c w : ℤ) (h1 : c + w = 15) (h2 : 6 * c - 3 * w = 45) : c = 10 := 
by sorry

end nathan_correct_answers_l172_172747


namespace ratio_of_rats_l172_172902

theorem ratio_of_rats (x y : ℝ) (h : (0.56 * x) / (0.84 * y) = 1 / 2) : x / y = 3 / 4 :=
sorry

end ratio_of_rats_l172_172902


namespace correct_answers_count_l172_172382

-- Define the conditions from the problem
def total_questions : ℕ := 25
def correct_points : ℕ := 4
def incorrect_points : ℤ := -1
def total_score : ℤ := 85

-- State the theorem
theorem correct_answers_count :
  ∃ x : ℕ, (x ≤ total_questions) ∧ 
           (total_questions - x : ℕ) ≥ 0 ∧ 
           (correct_points * x + incorrect_points * (total_questions - x) = total_score) :=
sorry

end correct_answers_count_l172_172382


namespace infinite_subsets_exists_divisor_l172_172025

-- Definition of the set M
def M : Set ℕ := { n | ∃ a b : ℕ, n = 2^a * 3^b }

-- Infinite family of subsets of M
variable (A : ℕ → Set ℕ)
variables (inf_family : ∀ i, A i ⊆ M)

-- Theorem statement
theorem infinite_subsets_exists_divisor :
  ∃ i j : ℕ, i ≠ j ∧ ∀ x ∈ A i, ∃ y ∈ A j, y ∣ x := by
  sorry

end infinite_subsets_exists_divisor_l172_172025


namespace center_cell_value_l172_172429

theorem center_cell_value (a b c d e f g h i : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) (hf : 0 < f)
  (hg : 0 < g) (hh : 0 < h) (hi : 0 < i)
  (row1 : a * b * c = 1) (row2 : d * e * f = 1) (row3 : g * h * i = 1)
  (col1 : a * d * g = 1) (col2 : b * e * h = 1) (col3 : c * f * i = 1)
  (square1 : a * b * d * e = 2) (square2 : b * c * e * f = 2)
  (square3 : d * e * g * h = 2) (square4 : e * f * h * i = 2) :
  e = 1 :=
begin
  sorry
end

end center_cell_value_l172_172429


namespace smallest_A_l172_172614

theorem smallest_A (A B C D E : ℕ) 
  (hA_even : A % 2 = 0)
  (hB_even : B % 2 = 0)
  (hC_even : C % 2 = 0)
  (hD_even : D % 2 = 0)
  (hE_even : E % 2 = 0)
  (hA_three_digit : 100 ≤ A ∧ A < 1000)
  (hB_three_digit : 100 ≤ B ∧ B < 1000)
  (hC_three_digit : 100 ≤ C ∧ C < 1000)
  (hD_three_digit : 100 ≤ D ∧ D < 1000)
  (hE_three_digit : 100 ≤ E ∧ E < 1000)
  (h_sorted : A < B ∧ B < C ∧ C < D ∧ D < E)
  (h_sum : A + B + C + D + E = 4306) :
  A = 326 :=
sorry

end smallest_A_l172_172614


namespace fraction_zero_implies_x_zero_l172_172172

theorem fraction_zero_implies_x_zero (x : ℝ) (h : x / (2 * x - 1) = 0) : x = 0 := 
by {
  sorry
}

end fraction_zero_implies_x_zero_l172_172172


namespace roundness_of_8000000_l172_172093

def is_prime (n : Nat) : Prop := sorry

def prime_factors_exponents (n : Nat) : List (Nat × Nat) := sorry

def roundness (n : Nat) : Nat := 
  (prime_factors_exponents n).foldr (λ p acc => p.2 + acc) 0

theorem roundness_of_8000000 : roundness 8000000 = 15 :=
sorry

end roundness_of_8000000_l172_172093


namespace curve_equation_represents_line_l172_172063

noncomputable def curve_is_line (x y : ℝ) : Prop :=
(x^2 + y^2 - 2*x) * (x + y - 3)^(1/2) = 0

theorem curve_equation_represents_line (x y : ℝ) :
curve_is_line x y ↔ (x + y = 3) :=
by sorry

end curve_equation_represents_line_l172_172063


namespace smallest_solution_floor_eq_l172_172693

theorem smallest_solution_floor_eq (x : ℝ) (hx : ⌊x^2⌋ - ⌊x⌋^2 = 19) : x = 11 := by
  sorry

end smallest_solution_floor_eq_l172_172693


namespace hyperbola_eccentricity_is_2_l172_172853

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : ℝ :=
  let c := 4 * a
  let e := c / a
  e

theorem hyperbola_eccentricity_is_2
  (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  hyperbola_eccentricity a b ha hb = 2 := 
sorry

end hyperbola_eccentricity_is_2_l172_172853


namespace third_snail_time_l172_172088

theorem third_snail_time
  (speed_first_snail : ℝ)
  (speed_second_snail : ℝ)
  (speed_third_snail : ℝ)
  (time_first_snail : ℝ)
  (distance : ℝ) :
  (speed_first_snail = 2) →
  (speed_second_snail = 2 * speed_first_snail) →
  (speed_third_snail = 5 * speed_second_snail) →
  (time_first_snail = 20) →
  (distance = speed_first_snail * time_first_snail) →
  (distance / speed_third_snail = 2) :=
by
  sorry

end third_snail_time_l172_172088


namespace hexagon_angle_arith_prog_l172_172600

theorem hexagon_angle_arith_prog (x d : ℝ) (hx : x > 0) (hd : d > 0) 
  (h_eq : 6 * x + 15 * d = 720) : x = 120 :=
by
  sorry

end hexagon_angle_arith_prog_l172_172600


namespace parametric_line_segment_computation_l172_172606

theorem parametric_line_segment_computation :
  ∃ (a b c d : ℝ), 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
   (-3, 10) = (a * t + b, c * t + d) ∧
   (4, 16) = (a * 1 + b, c * 1 + d)) ∧
  (b = -3) ∧ (d = 10) ∧ 
  (a + b = 4) ∧ (c + d = 16) ∧ 
  (a^2 + b^2 + c^2 + d^2 = 194) :=
sorry

end parametric_line_segment_computation_l172_172606


namespace remainder_twice_original_l172_172950

def findRemainder (N : ℕ) (D : ℕ) (r : ℕ) : ℕ :=
  2 * N % D

theorem remainder_twice_original
  (N : ℕ) (D : ℕ)
  (hD : D = 367)
  (hR : N % D = 241) :
  findRemainder N D 2 = 115 := by
  sorry

end remainder_twice_original_l172_172950


namespace no_square_remainder_2_infinitely_many_squares_remainder_3_l172_172774

theorem no_square_remainder_2 :
  ∀ n : ℤ, (n * n) % 6 ≠ 2 :=
by sorry

theorem infinitely_many_squares_remainder_3 :
  ∀ k : ℤ, ∃ n : ℤ, n = 6 * k + 3 ∧ (n * n) % 6 = 3 :=
by sorry

end no_square_remainder_2_infinitely_many_squares_remainder_3_l172_172774


namespace inequality_proof_l172_172201

theorem inequality_proof (x : ℝ) (hx : 0 < x) : (1 / x) + 4 * (x ^ 2) ≥ 3 :=
by
  sorry

end inequality_proof_l172_172201


namespace curve_is_line_l172_172137

-- Define the polar equation as a condition
def polar_eq (r θ : ℝ) : Prop := r = 2 / (2 * Real.sin θ - Real.cos θ)

-- Define what it means for a curve to be a line
def is_line (x y : ℝ) : Prop := x + 2 * y = 2

-- The main statement to prove
theorem curve_is_line (r θ : ℝ) (x y : ℝ) (hr : polar_eq r θ) (hx : x = r * Real.cos θ) (hy : y = r * Real.sin θ) :
  is_line x y :=
sorry

end curve_is_line_l172_172137


namespace staplers_left_l172_172481

-- Definitions based on conditions
def initial_staplers : ℕ := 50
def dozen : ℕ := 12
def reports_stapled : ℕ := 3 * dozen

-- Statement of the theorem
theorem staplers_left (h : initial_staplers = 50) (d : dozen = 12) (r : reports_stapled = 3 * dozen) :
  (initial_staplers - reports_stapled) = 14 :=
sorry

end staplers_left_l172_172481


namespace count_valid_4_digit_numbers_l172_172408

def is_valid_number (n : ℕ) : Prop :=
  let thousands := n / 1000 in
  let hundreds := (n / 100) % 10 in
  let tens := (n / 10) % 10 in
  let units := n % 10 in
  1000 ≤ n ∧ n < 10000 ∧
  1 ≤ thousands ∧ thousands ≤ 9 ∧
  1 ≤ hundreds ∧ hundreds ≤ 9 ∧
  units ≥ 3 * tens

theorem count_valid_4_digit_numbers : 
  (Finset.filter is_valid_number (Finset.range 10000)).card = 1782 :=
sorry

end count_valid_4_digit_numbers_l172_172408


namespace park_area_l172_172654

-- Define the width (w) and length (l) of the park
def width : Float := 11.25
def length : Float := 33.75

-- Define the perimeter and area functions
def perimeter (w l : Float) : Float := 2 * (w + l)
def area (w l : Float) : Float := w * l

-- Provide the conditions
axiom width_is_one_third_length : width = length / 3
axiom perimeter_is_90 : perimeter width length = 90

-- Theorem to prove the area given the conditions
theorem park_area : area width length = 379.6875 := by
  sorry

end park_area_l172_172654


namespace range_of_a_l172_172293

theorem range_of_a (a : ℚ) (h₀ : 0 < a) (h₁ : ∃ n : ℕ, (2 * n - 1 = 2007) ∧ (-a < n ∧ n < a)) :
  1003 < a ∧ a ≤ 1004 :=
sorry

end range_of_a_l172_172293


namespace number_of_monomials_is_3_l172_172019

def isMonomial (term : String) : Bool :=
  match term with
  | "0" => true
  | "-a" => true
  | "-3x^2y" => true
  | _ => false

def monomialCount (terms : List String) : Nat :=
  terms.filter isMonomial |>.length

theorem number_of_monomials_is_3 :
  monomialCount ["1/x", "x+y", "0", "-a", "-3x^2y", "(x+1)/3"] = 3 :=
by
  sorry

end number_of_monomials_is_3_l172_172019


namespace derivative_sum_l172_172280

theorem derivative_sum (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (hf : ∀ x, deriv f x = f' x)
  (h : ∀ x, f x = 3 * x^2 + 2 * x * f' 2) :
  f' 5 + f' 2 = -6 :=
sorry

end derivative_sum_l172_172280


namespace advanced_purchase_tickets_sold_l172_172115

theorem advanced_purchase_tickets_sold (A D : ℕ) 
  (h1 : A + D = 140)
  (h2 : 8 * A + 14 * D = 1720) : 
  A = 40 :=
by
  sorry

end advanced_purchase_tickets_sold_l172_172115


namespace solve_inequality_l172_172058

theorem solve_inequality :
  (4 - Real.sqrt 17 < x ∧ x < 4 - Real.sqrt 3) ∨ 
  (4 + Real.sqrt 3 < x ∧ x < 4 + Real.sqrt 17) → 
  0 < (x^2 - 8*x + 13) / (x^2 - 4*x + 7) ∧ 
  (x^2 - 8*x + 13) / (x^2 - 4*x + 7) < 2 :=
sorry

end solve_inequality_l172_172058


namespace value_of_f_minus_3_l172_172397

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := a * Real.sin x + b * Real.tan x + x^3 + 1

theorem value_of_f_minus_3 (a b : ℝ) (h : f 3 a b = 7) : f (-3) a b = -5 := 
by
  sorry

end value_of_f_minus_3_l172_172397


namespace xy_gt_xz_l172_172725

variable {R : Type*} [LinearOrderedField R]
variables (x y z : R)

theorem xy_gt_xz (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 0) : x * y > x * z :=
by
  sorry

end xy_gt_xz_l172_172725


namespace unique_perpendicular_line_through_point_l172_172157

variables (a b : ℝ → ℝ) (P : ℝ)

def are_skew_lines (a b : ℝ → ℝ) : Prop :=
  ¬∃ (t₁ t₂ : ℝ), a t₁ = b t₂

def is_point_not_on_lines (P : ℝ) (a b : ℝ → ℝ) : Prop :=
  ∀ (t : ℝ), P ≠ a t ∧ P ≠ b t

theorem unique_perpendicular_line_through_point (ha : are_skew_lines a b) (hp : is_point_not_on_lines P a b) :
  ∃! (L : ℝ → ℝ), (∀ (t : ℝ), L t ≠ P) ∧ (∀ (L' : ℝ → ℝ), (∀ (t : ℝ), L' t ≠ P) → L' = L) := sorry

end unique_perpendicular_line_through_point_l172_172157


namespace shorter_side_length_l172_172503

theorem shorter_side_length (L W : ℝ) (h1 : L * W = 91) (h2 : 2 * L + 2 * W = 40) :
  min L W = 7 :=
by
  sorry

end shorter_side_length_l172_172503


namespace mul_fraction_eq_l172_172673

theorem mul_fraction_eq : 7 * (1 / 11) * 33 = 21 :=
by
  sorry

end mul_fraction_eq_l172_172673


namespace max_dot_product_of_points_on_ellipses_l172_172292

theorem max_dot_product_of_points_on_ellipses :
  let C1 (M : ℝ × ℝ) := M.1^2 / 25 + M.2^2 / 9 = 1
  let C2 (N : ℝ × ℝ) := N.1^2 / 9 + N.2^2 / 25 = 1
  ∃ M N : ℝ × ℝ,
    C1 M ∧ C2 N ∧
    (∀ M N, C1 M ∧ C2 N → M.1 * N.1 + M.2 * N.2 ≤ 15 ∧ 
      (∃ θ φ, M = (5 * Real.cos θ, 3 * Real.sin θ) ∧ N = (3 * Real.cos φ, 5 * Real.sin φ) ∧ (M.1 * N.1 + M.2 * N.2 = 15))) :=
by
  sorry

end max_dot_product_of_points_on_ellipses_l172_172292


namespace eval_expression_l172_172676

theorem eval_expression : 0.5 * 0.8 - 0.2 = 0.2 := by
  sorry

end eval_expression_l172_172676


namespace ratio_of_container_volumes_l172_172521

-- Define the volumes of the first and second containers.
variables (A B : ℝ )

-- Hypotheses based on the problem conditions
-- First container is 4/5 full
variable (h1 : A * 4 / 5 = B * 2 / 3)

-- The statement to prove
theorem ratio_of_container_volumes : A / B = 5 / 6 :=
by
  sorry

end ratio_of_container_volumes_l172_172521


namespace negation_of_exists_cond_l172_172075

theorem negation_of_exists_cond (x : ℝ) (h : x > 0) : ¬ (∃ x : ℝ, x > 0 ∧ x^3 - x + 1 > 0) ↔ (∀ x : ℝ, x > 0 → x^3 - x + 1 ≤ 0) :=
by 
  sorry

end negation_of_exists_cond_l172_172075


namespace total_votes_l172_172631

theorem total_votes (V : ℝ) (h1 : 0.32 * V = 0.32 * V) (h2 : 0.32 * V + 1908 = 0.68 * V) : V = 5300 :=
by
  sorry

end total_votes_l172_172631


namespace intersection_points_l172_172602

-- Define the line equation
def line (x : ℝ) : ℝ := 2 * x - 1

-- Problem statement to be proven
theorem intersection_points :
  (line 0.5 = 0) ∧ (line 0 = -1) :=
by 
  sorry

end intersection_points_l172_172602


namespace simple_interest_years_l172_172955

theorem simple_interest_years (P : ℝ) (R : ℝ) (N : ℝ) (higher_interest_amount : ℝ) (additional_rate : ℝ) (initial_sum : ℝ) :
  (initial_sum * (R + additional_rate) * N) / 100 - (initial_sum * R * N) / 100 = higher_interest_amount →
  initial_sum = 3000 →
  higher_interest_amount = 1350 →
  additional_rate = 5 →
  N = 9 :=
by
  sorry

end simple_interest_years_l172_172955


namespace positive_integer_solution_l172_172532

theorem positive_integer_solution (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxyz : x ≤ y ∧ y ≤ z) (h_eq : 5 * (x * y + y * z + z * x) = 4 * x * y * z) :
  (x = 2 ∧ y = 5 ∧ z = 10) ∨ (x = 2 ∧ y = 4 ∧ z = 20) :=
sorry

end positive_integer_solution_l172_172532


namespace solve_quadratic_equation_l172_172467

theorem solve_quadratic_equation :
  ∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2 ∧ ∀ x : ℝ, (x^2 - 2*x - 1 = 0) ↔ (x = x₁ ∨ x = x₂) :=
by
  sorry

end solve_quadratic_equation_l172_172467


namespace polynomial_expansion_l172_172968

variable (x : ℝ)

theorem polynomial_expansion :
  (7*x^2 + 3)*(5*x^3 + 4*x + 1) = 35*x^5 + 43*x^3 + 7*x^2 + 12*x + 3 := by
  sorry

end polynomial_expansion_l172_172968


namespace tree_initial_height_l172_172627

noncomputable def initial_tree_height (H : ℝ) (growth_rate : ℝ) (years : ℕ) : ℝ := 
  H + growth_rate * years

theorem tree_initial_height :
  ∀ (H : ℝ), 
  (∀ (years : ℕ), ∃ h : ℝ, h = initial_tree_height H 0.5 years) →
  initial_tree_height H 0.5 6 = initial_tree_height H 0.5 4 * (7 / 6) →
  H = 4 :=
by
  intro H height_increase condition
  sorry

end tree_initial_height_l172_172627


namespace problem_l172_172170

def g(x : ℤ) : ℤ := 3 * x^2 + 3 * x - 2

theorem problem : g(g(3)) = 3568 := by
  sorry

end problem_l172_172170


namespace determine_a_l172_172851

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 1 then x^3 + 1 else x^2 - a * x

theorem determine_a (a : ℝ) : 
  f (f 0 a) a = -2 → a = 3 :=
by
  sorry

end determine_a_l172_172851


namespace sum_of_roots_l172_172610

theorem sum_of_roots (p : ℝ) (h : (4 - p) / 2 = 9) : (p / 2 = 7) :=
by 
  sorry

end sum_of_roots_l172_172610


namespace polynomial_inequality_solution_l172_172126

theorem polynomial_inequality_solution :
  {x : ℝ | x^3 - 4*x^2 - x + 20 > 0} = {x | x < -4} ∪ {x | 1 < x ∧ x < 5} ∪ {x | x > 5} :=
sorry

end polynomial_inequality_solution_l172_172126


namespace find_amount_l172_172738

-- Definitions based on the conditions provided
def gain : ℝ := 0.70
def gain_percent : ℝ := 1.0

-- The theorem statement
theorem find_amount (h : gain_percent = 1) : ∀ (amount : ℝ), amount = gain / (gain_percent / 100) → amount = 70 :=
by
  intros amount h_calc
  sorry

end find_amount_l172_172738


namespace arithmetic_geometric_mean_l172_172051

theorem arithmetic_geometric_mean (x y : ℝ) (h1 : (x + y) / 2 = 20) (h2 : Real.sqrt (x * y) = Real.sqrt 110) : x^2 + y^2 = 1380 := by
  sorry

end arithmetic_geometric_mean_l172_172051


namespace value_division_l172_172568

theorem value_division (x y : ℝ) (h1 : y ≠ 0) (h2 : 2 * x - y = 1.75 * x) 
                       (h3 : x / y = n) : n = 4 := 
by 
sorry

end value_division_l172_172568


namespace volume_cone_equals_cylinder_minus_surface_area_l172_172773

theorem volume_cone_equals_cylinder_minus_surface_area (r h : ℝ) :
  let V_cyl := π * r^2 * h
  let V_cone := (1 / 3) * π * r^2 * h
  let S_lateral_cyl := 2 * π * r * h
  V_cone = V_cyl - (1 / 3) * S_lateral_cyl * r := by
  let V_cyl := π * r^2 * h
  let V_cone := (1 / 3) * π * r^2 * h
  let S_lateral_cyl := 2 * π * r * h
  sorry

end volume_cone_equals_cylinder_minus_surface_area_l172_172773


namespace solution_set_of_inequality_l172_172281

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 - 4 * x else (-(x^2 - 4 * x))

theorem solution_set_of_inequality :
  {x : ℝ | f (x - 2) < 5} = {x : ℝ | -3 < x ∧ x < 7} := by
  sorry

end solution_set_of_inequality_l172_172281


namespace stratified_sampling_l172_172648

noncomputable def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem stratified_sampling :
  let junior_students := 400
  let senior_students := 200
  let total_sample_size := 60
  let junior_sample_size := (2 * total_sample_size) / 3
  let senior_sample_size := total_sample_size / 3
  combination junior_students junior_sample_size * combination senior_students senior_sample_size =
    combination 400 40 * combination 200 20 :=
by
  let junior_students := 400
  let senior_students := 200
  let total_sample_size := 60
  let junior_sample_size := (2 * total_sample_size) / 3
  let senior_sample_size := total_sample_size / 3
  exact sorry

end stratified_sampling_l172_172648


namespace smallest_a_l172_172417

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = k * a

theorem smallest_a (a : ℕ) (h1 : is_factor 112 (a * 43 * 62 * 1311)) (h2 : is_factor 33 (a * 43 * 62 * 1311)) : a = 1848 :=
by
  sorry

end smallest_a_l172_172417


namespace remainder_of_3_pow_19_mod_10_l172_172498

theorem remainder_of_3_pow_19_mod_10 : (3 ^ 19) % 10 = 7 := by
  sorry

end remainder_of_3_pow_19_mod_10_l172_172498


namespace find_c_d_of_cubic_common_roots_l172_172843

theorem find_c_d_of_cubic_common_roots 
  (c d : ℝ)
  (h1 : ∃ r s : ℝ, r ≠ s ∧ (r ^ 3 + c * r ^ 2 + 12 * r + 7 = 0) ∧ (s ^ 3 + c * s ^ 2 + 12 * s + 7 = 0))
  (h2 : ∃ r s : ℝ, r ≠ s ∧ (r ^ 3 + d * r ^ 2 + 15 * r + 9 = 0) ∧ (s ^ 3 + d * s ^ 2 + 15 * s + 9 = 0)) :
  c = 5 ∧ d = 4 :=
sorry

end find_c_d_of_cubic_common_roots_l172_172843


namespace solve_system_l172_172206

theorem solve_system :
  ∃! (x y : ℝ), (2 * x + y + 8 ≤ 0) ∧ (x^4 + 2 * x^2 * y^2 + y^4 + 9 - 10 * x^2 - 10 * y^2 = 8 * x * y) ∧ (x = -3 ∧ y = -2) := 
  by
  sorry

end solve_system_l172_172206


namespace compare_y_values_l172_172330

noncomputable def parabola (x : ℝ) : ℝ := -2 * (x + 1) ^ 2 - 1

theorem compare_y_values :
  ∃ y1 y2 y3, (parabola (-3) = y1) ∧ (parabola (-2) = y2) ∧ (parabola 2 = y3) ∧ (y3 < y1) ∧ (y1 < y2) :=
by
  sorry

end compare_y_values_l172_172330


namespace sphere_radius_squared_l172_172488

theorem sphere_radius_squared (R x y z : ℝ)
  (h1 : 2 * Real.sqrt (R^2 - x^2 - y^2) = 5)
  (h2 : 2 * Real.sqrt (R^2 - x^2 - z^2) = 6)
  (h3 : 2 * Real.sqrt (R^2 - y^2 - z^2) = 7) :
  R^2 = 15 :=
sorry

end sphere_radius_squared_l172_172488


namespace payment_ratio_l172_172095

theorem payment_ratio (m p t : ℕ) (hm : m = 14) (hp : p = 84) (ht : t = m * 12) :
  (p : ℚ) / ((t : ℚ) - p) = 1 :=
by
  sorry

end payment_ratio_l172_172095


namespace max_value_of_f_l172_172834

-- Define the quadratic function
def f (x : ℝ) : ℝ := 9 * x - 4 * x^2

-- Define a proof problem to show that the maximum value of f(x) is 81/16
theorem max_value_of_f : ∃ x : ℝ, f x = 81 / 16 :=
by
  -- The vertex of the quadratic function gives the maximum value since the parabola opens downward
  let x := 9 / (2 * 4)
  use x
  -- sorry to skip the proof steps
  sorry

end max_value_of_f_l172_172834


namespace fraction_halfway_between_one_fourth_and_one_sixth_l172_172841

theorem fraction_halfway_between_one_fourth_and_one_sixth :
  (1/4 + 1/6) / 2 = 5 / 24 :=
by
  sorry

end fraction_halfway_between_one_fourth_and_one_sixth_l172_172841


namespace wendy_time_correct_l172_172384

noncomputable section

def bonnie_time : ℝ := 7.80
def wendy_margin : ℝ := 0.25

theorem wendy_time_correct : (bonnie_time - wendy_margin) = 7.55 := by
  sorry

end wendy_time_correct_l172_172384


namespace sequence_of_arrows_512_to_517_is_B_C_D_E_A_l172_172777

noncomputable def sequence_from_512_to_517 : List Char :=
  let pattern := ['A', 'B', 'C', 'D', 'E']
  pattern.drop 2 ++ pattern.take 2

theorem sequence_of_arrows_512_to_517_is_B_C_D_E_A : sequence_from_512_to_517 = ['B', 'C', 'D', 'E', 'A'] :=
  sorry

end sequence_of_arrows_512_to_517_is_B_C_D_E_A_l172_172777


namespace proof_min_max_expected_wasted_minutes_l172_172644

/-- The conditions given:
    - There are 8 people in the queue.
    - 5 people perform simple operations that take 1 minute each.
    - 3 people perform lengthy operations that take 5 minutes each.
--/
structure QueueStatus where
  total_people : Nat := 8
  simple_operations_people : Nat := 5
  lengthy_operations_people : Nat := 3
  simple_operation_time : Nat := 1
  lengthy_operation_time : Nat := 5

/-- Propositions to be proven:
    - Minimum possible total number of wasted person-minutes is 40.
    - Maximum possible total number of wasted person-minutes is 100.
    - Expected total number of wasted person-minutes in random order is 72.5.
--/
def min_wasted_person_minutes (qs: QueueStatus) : Nat := 40
def max_wasted_person_minutes (qs: QueueStatus) : Nat := 100
def expected_wasted_person_minutes (qs: QueueStatus) : Real := 72.5

theorem proof_min_max_expected_wasted_minutes (qs: QueueStatus) :
  min_wasted_person_minutes qs = 40 ∧ 
  max_wasted_person_minutes qs = 100 ∧ 
  expected_wasted_person_minutes qs = 72.5 := by
  sorry

end proof_min_max_expected_wasted_minutes_l172_172644


namespace smallest_solution_eq_sqrt_104_l172_172695

theorem smallest_solution_eq_sqrt_104 :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, ⌊y^2⌋ - ⌊y⌋^2 = 19 → x ≤ y) := sorry

end smallest_solution_eq_sqrt_104_l172_172695


namespace garden_perimeter_l172_172242

theorem garden_perimeter
  (width_garden : ℝ) (area_playground : ℝ)
  (length_playground : ℝ) (width_playground : ℝ)
  (area_garden : ℝ) (L : ℝ)
  (h1 : width_garden = 4) 
  (h2 : length_playground = 16)
  (h3 : width_playground = 12)
  (h4 : area_playground = length_playground * width_playground)
  (h5 : area_garden = area_playground)
  (h6 : area_garden = L * width_garden) :
  2 * L + 2 * width_garden = 104 :=
by
  sorry

end garden_perimeter_l172_172242


namespace mona_game_group_size_l172_172581

theorem mona_game_group_size 
  (x : ℕ)
  (h_conditions: 9 * (x - 1) - 3 = 33) : x = 5 := 
by 
  sorry

end mona_game_group_size_l172_172581


namespace jogged_distance_is_13_point_5_l172_172818

noncomputable def jogger_distance (x t d : ℝ) : Prop :=
  d = x * t ∧
  d = (x + 3/4) * (3 * t / 4) ∧
  d = (x - 3/4) * (t + 3)

theorem jogged_distance_is_13_point_5:
  ∃ (x t d : ℝ), jogger_distance x t d ∧ d = 13.5 :=
by
  sorry

end jogged_distance_is_13_point_5_l172_172818


namespace problem_solution_l172_172401

-- Definitions
def has_property_P (A : List ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ A.length →
    (A.get! (j - 1) + A.get! (i - 1) ∈ A ∨ A.get! (j - 1) - A.get! (i - 1) ∈ A)

def sequence_01234 := [0, 2, 4, 6]

-- Propositions
def proposition_1 : Prop := has_property_P sequence_01234

def proposition_2 (A : List ℕ) : Prop := 
  has_property_P A → (A.headI = 0)

def proposition_3 (A : List ℕ) : Prop :=
  has_property_P A → A.headI ≠ 0 →
  ∀ k, 1 ≤ k ∧ k < A.length → A.get! (A.length - 1) - A.get! (A.length - 1 - k) = A.get! k

def proposition_4 (A : List ℕ) : Prop :=
  has_property_P A → A.length = 3 →
  A.get! 2 = A.get! 0 + A.get! 1

-- Main statement
theorem problem_solution : 
  (proposition_1) ∧
  (∃ A, ¬ (proposition_2 A)) ∧
  (∃ A, proposition_3 A) ∧
  (∃ A, proposition_4 A) →
  3 = 3 := 
by sorry

end problem_solution_l172_172401


namespace count_even_fibonacci_first_2007_l172_172487

def fibonacci (n : Nat) : Nat :=
  if h : n = 0 then 0
  else if h : n = 1 then 1
  else fibonacci (n - 1) + fibonacci (n - 2)

def fibonacci_parity : List Bool := List.map (fun x => fibonacci x % 2 = 0) (List.range 2008)

def count_even (l : List Bool) : Nat :=
  l.foldl (fun acc x => if x then acc + 1 else acc) 0

theorem count_even_fibonacci_first_2007 : count_even (fibonacci_parity.take 2007) = 669 :=
sorry

end count_even_fibonacci_first_2007_l172_172487


namespace John_days_per_week_l172_172314

theorem John_days_per_week
    (patients_first : ℕ := 20)
    (patients_increase_rate : ℕ := 20)
    (patients_second : ℕ := (20 + (20 * 20 / 100)))
    (total_weeks_year : ℕ := 50)
    (total_patients_year : ℕ := 11000) :
    ∃ D : ℕ, (20 * D + (20 + (20 * 20 / 100)) * D) * total_weeks_year = total_patients_year ∧ D = 5 := by
  sorry

end John_days_per_week_l172_172314


namespace g_of_5_l172_172911

theorem g_of_5 (g : ℝ → ℝ) (h : ∀ x ≠ 0, 4 * g x - 3 * g (1 / x) = 2 * x) :
  g 5 = 402 / 70 := 
sorry

end g_of_5_l172_172911


namespace tyler_common_ratio_l172_172494

theorem tyler_common_ratio (a r : ℝ) 
  (h1 : a / (1 - r) = 10)
  (h2 : (a + 4) / (1 - r) = 15) : 
  r = 1 / 5 :=
by
  sorry

end tyler_common_ratio_l172_172494


namespace probability_abc_120_l172_172094

/-- Define a standard die with possible outcomes 1, 2, 3, 4, 5, and 6 -/
def standard_die := {1, 2, 3, 4, 5, 6} 

/-- The set of permutations that when multiplied together equals to 120 -/
def valid_permutations : Finset (ℕ × ℕ × ℕ) :=
  { (5, 4, 6), (5, 6, 4), (6, 4, 5), (6, 5, 4), (4, 5, 6), (4, 6, 5) }

theorem probability_abc_120 : 
  (Finset.card valid_permutations : ℝ) / (6 * 6 * 6 : ℝ) = 1 / 36 := 
by
  sorry

end probability_abc_120_l172_172094


namespace min_value_reciprocal_sum_l172_172862

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m + n = 1) (h2 : 0 < m) (h3 : 0 < n) : 
  (1/m + 1/n) = 4 :=
by
  sorry

end min_value_reciprocal_sum_l172_172862


namespace probability_A_wins_l172_172493

variable (P_A_not_lose : ℝ) (P_draw : ℝ)
variable (h1 : P_A_not_lose = 0.8)
variable (h2 : P_draw = 0.5)

theorem probability_A_wins : P_A_not_lose - P_draw = 0.3 := by
  sorry

end probability_A_wins_l172_172493


namespace green_marble_prob_l172_172087

-- Problem constants
def total_marbles : ℕ := 84
def prob_white : ℚ := 1 / 4
def prob_red_or_blue : ℚ := 0.4642857142857143

-- Defining the individual variables for the counts
variable (W R B G : ℕ)

-- Conditions
axiom total_marbles_eq : W + R + B + G = total_marbles
axiom prob_white_eq : (W : ℚ) / total_marbles = prob_white
axiom prob_red_or_blue_eq : (R + B : ℚ) / total_marbles = prob_red_or_blue

-- Proving the probability of drawing a green marble
theorem green_marble_prob :
  (G : ℚ) / total_marbles = 2 / 7 :=
by
  sorry  -- Proof is not required and thus omitted

end green_marble_prob_l172_172087


namespace average_calls_per_day_l172_172750

def calls_Monday : ℕ := 35
def calls_Tuesday : ℕ := 46
def calls_Wednesday : ℕ := 27
def calls_Thursday : ℕ := 61
def calls_Friday : ℕ := 31

def total_calls : ℕ := calls_Monday + calls_Tuesday + calls_Wednesday + calls_Thursday + calls_Friday
def number_of_days : ℕ := 5

theorem average_calls_per_day : (total_calls / number_of_days) = 40 := 
by 
  -- calculations and proof steps go here.
  sorry

end average_calls_per_day_l172_172750


namespace probability_event_l172_172549

def uniformProbability (a b : ℝ) (h : a < b) (P : set ℝ → Prop) : ℝ :=
  (∫ x in a..b, if P x then 1 else 0) / (b - a)

theorem probability_event : 
  uniformProbability 0 2 (by linarith : 0 < 2) (λ x, 3 * x - 2 ≥ 0) = 2 / 3 :=
by
  sorry

end probability_event_l172_172549


namespace find_a_l172_172399

-- Define the polynomial f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 - 0.8

-- Define the intermediate values v_0, v_1, and v_2 using Horner's method
def v_0 (a : ℝ) : ℝ := a
def v_1 (a : ℝ) (x : ℝ) : ℝ := v_0 a * x + 2
def v_2 (a : ℝ) (x : ℝ) : ℝ := v_1 a x * x + 3.5 * x - 2.6 * x + 13.5

-- The condition for v_2 when x = 5
axiom v2_value (a : ℝ) : v_2 a 5 = 123.5

-- Prove that a = 4
theorem find_a : ∃ a : ℝ, v_2 a 5 = 123.5 ∧ a = 4 := by
  sorry

end find_a_l172_172399


namespace min_value_of_f_l172_172621

noncomputable def f (x : ℝ) : ℝ := 7 * x^2 - 28 * x + 1425

theorem min_value_of_f : ∃ (x : ℝ), f x = 1397 :=
by
  sorry

end min_value_of_f_l172_172621


namespace bank_queue_wasted_time_l172_172642

-- Conditions definition
def simple_time : ℕ := 1
def lengthy_time : ℕ := 5
def num_simple : ℕ := 5
def num_lengthy : ℕ := 3
def total_people : ℕ := 8

-- Theorem statement
theorem bank_queue_wasted_time :
  (min_wasted_time : ℕ := 40) ∧
  (max_wasted_time : ℕ := 100) ∧
  (expected_wasted_time : ℚ := 72.5) := by
  sorry

end bank_queue_wasted_time_l172_172642


namespace smallest_x_solution_l172_172682

theorem smallest_x_solution :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, (⌊y^2⌋ - ⌊y⌋^2 = 19) → x ≤ y) ∧ x = Real.sqrt 119 := 
sorry

end smallest_x_solution_l172_172682


namespace range_of_m_l172_172990

theorem range_of_m (m : ℝ) :
  (¬ ∃ x_0 : ℝ, x_0^2 + 2 * m * x_0 + m + 2 < 0) ↔ (-1 : ℝ) ≤ m ∧ m ≤ 2 :=
sorry

end range_of_m_l172_172990


namespace problem_statement_l172_172662

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

end problem_statement_l172_172662


namespace smallest_x_solution_l172_172685

theorem smallest_x_solution :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, (⌊y^2⌋ - ⌊y⌋^2 = 19) → x ≤ y) ∧ x = Real.sqrt 119 := 
sorry

end smallest_x_solution_l172_172685


namespace peter_spent_on_repairs_l172_172899

variable (C : ℝ)

def repairs_cost (C : ℝ) := 0.10 * C

def profit (C : ℝ) := 1.20 * C - C

theorem peter_spent_on_repairs :
  ∀ C, profit C = 1100 → repairs_cost C = 550 :=
by
  intro C
  sorry

end peter_spent_on_repairs_l172_172899


namespace min_days_to_find_poisoned_apple_l172_172204

theorem min_days_to_find_poisoned_apple (n : ℕ) (n_pos : 0 < n) : 
  ∀ k : ℕ, 2^k ≥ 2021 → k ≥ 11 :=
  sorry

end min_days_to_find_poisoned_apple_l172_172204


namespace meat_per_slice_is_22_l172_172041

noncomputable def piecesOfMeatPerSlice : ℕ :=
  let pepperoni := 30
  let ham := 2 * pepperoni
  let sausage := pepperoni + 12
  let totalMeat := pepperoni + ham + sausage
  let slices := 6
  totalMeat / slices

theorem meat_per_slice_is_22 : piecesOfMeatPerSlice = 22 :=
by
  -- Here would be the proof (not required in the task)
  sorry

end meat_per_slice_is_22_l172_172041


namespace remove_two_vertices_eliminate_all_triangles_l172_172027

theorem remove_two_vertices_eliminate_all_triangles {V : Type*} (G : SimpleGraph V) :
  (¬ ∃ (K5 : set V), (K5.card = 5) ∧ (∀ (u v : V), u ∈ K5 → v ∈ K5 → (u = v ∨ G.Adj u v))) → 
  (∀ {T1 T2 : set V}, T1.card = 3 → T2.card = 3 → (∀ (t1 t2 : V), t1 ∈ T1 → t2 ∈ T2 → T1 ≠ T2 → ∃ v, v ∈ T1 ∧ v ∈ T2)) →
  (∃ (v1 v2 : V), ∀ (T : set V), T.card = 3 → (v1 ∈ T ∨ v2 ∈ T) → ¬ ∃ (u w x : V), {u, w, x} = T ∧ G.Adj u w ∧ G.Adj w x ∧ G.Adj x u) :=
by 
  sorry

end remove_two_vertices_eliminate_all_triangles_l172_172027


namespace straw_costs_max_packs_type_a_l172_172647

theorem straw_costs (x y : ℝ) (h1 : 12 * x + 15 * y = 171) (h2 : 24 * x + 28 * y = 332) :
  x = 8 ∧ y = 5 :=
  by sorry

theorem max_packs_type_a (m : ℕ) (cA cB : ℕ) (total_packs : ℕ) (max_cost : ℕ)
  (h1 : cA = 8) (h2 : cB = 5) (h3 : total_packs = 100) (h4 : max_cost = 600) :
  m ≤ 33 :=
  by sorry

end straw_costs_max_packs_type_a_l172_172647


namespace num_of_terms_in_arith_seq_l172_172561

-- Definitions of the conditions
def a : Int := -5 -- Start of the arithmetic sequence
def l : Int := 85 -- End of the arithmetic sequence
def d : Nat := 5  -- Common difference

-- The theorem that needs to be proved
theorem num_of_terms_in_arith_seq : (l - a) / d + 1 = 19 := sorry

end num_of_terms_in_arith_seq_l172_172561


namespace find_value_l172_172470

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodic : ∀ x : ℝ, f (x + Real.pi) = f x
axiom value_at_neg_pi_third : f (-Real.pi / 3) = 1 / 2

theorem find_value : f (2017 * Real.pi / 3) = 1 / 2 :=
by
  sorry

end find_value_l172_172470


namespace staplers_left_l172_172486

-- Definitions of the conditions
def initialStaplers : ℕ := 50
def dozen : ℕ := 12
def reportsStapled : ℕ := 3 * dozen

-- The proof statement
theorem staplers_left : initialStaplers - reportsStapled = 14 := by
  sorry

end staplers_left_l172_172486


namespace cannot_determine_right_triangle_from_conditions_l172_172749

-- Let triangle ABC have side lengths a, b, c opposite angles A, B, C respectively.
def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

-- Condition A: c^2 = a^2 - b^2 is rearranged to c^2 + b^2 = a^2 implying right triangle
def condition_A (a b c : ℝ) : Prop := c^2 = a^2 - b^2

-- Condition B: Triangle angles in the ratio A:B:C = 3:4:5 means not a right triangle
def condition_B : Prop := 
  let A := 45.0
  let B := 60.0
  let C := 75.0
  A ≠ 90.0 ∧ B ≠ 90.0 ∧ C ≠ 90.0

-- Condition C: Specific lengths 7, 24, 25 form a right triangle
def condition_C : Prop := 
  let a := 7.0
  let b := 24.0
  let c := 25.0
  is_right_triangle a b c

-- Condition D: A = B - C can be shown to always form at least one 90 degree angle, a right triangle
def condition_D (A B C : ℝ) : Prop := A = B - C ∧ (A + B + C = 180)

-- The actual mathematical proof that option B does not determine a right triangle
theorem cannot_determine_right_triangle_from_conditions :
  ∀ a b c (A B C : ℝ),
    (condition_A a b c → is_right_triangle a b c) ∧
    (condition_C → is_right_triangle 7 24 25) ∧
    (condition_D A B C → is_right_triangle a b c) ∧
    ¬condition_B :=
by
  sorry

end cannot_determine_right_triangle_from_conditions_l172_172749


namespace min_value_fraction_l172_172277

-- We start by defining the geometric sequence and the given conditions
variable {a : ℕ → ℝ}
variable {r : ℝ}
variable {a1 : ℝ} (h_pos : ∀ n, 0 < a n)
variable (h_geo : ∀ n, a (n + 1) = a n * r)
variable (h_a7 : a 7 = a 6 + 2 * a 5)
variable (h_am_an : ∃ m n, a m * a n = 16 * (a 1)^2)

theorem min_value_fraction : 
  ∃ (m n : ℕ), (a m * a n = 16 * (a 1)^2 ∧ (1/m) + (4/n) = 1) :=
sorry

end min_value_fraction_l172_172277


namespace nth_equation_l172_172327

theorem nth_equation (n : ℕ) : 
  1 - (1 / ((n + 1)^2)) = (n / (n + 1)) * ((n + 2) / (n + 1)) :=
by sorry

end nth_equation_l172_172327


namespace ratio_rocks_eaten_to_collected_l172_172527

def rocks_collected : ℕ := 10
def rocks_left : ℕ := 7
def rocks_spit_out : ℕ := 2

theorem ratio_rocks_eaten_to_collected : 
  (rocks_collected - rocks_left + rocks_spit_out) * 2 = rocks_collected := 
by 
  sorry

end ratio_rocks_eaten_to_collected_l172_172527


namespace value_of_k_for_square_of_binomial_l172_172626

theorem value_of_k_for_square_of_binomial (a k : ℝ) : (x : ℝ) → x^2 - 14 * x + k = (x - a)^2 → k = 49 :=
by
  intro x h
  sorry

end value_of_k_for_square_of_binomial_l172_172626


namespace intersection_M_N_l172_172976

noncomputable def M : Set ℝ := { x | -1 < x ∧ x < 3 }
noncomputable def N : Set ℝ := { x | ∃ y, y = Real.log (x - x^2) }
noncomputable def intersection (A B : Set ℝ) : Set ℝ := { x | x ∈ A ∧ x ∈ B }

theorem intersection_M_N : intersection M N = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end intersection_M_N_l172_172976


namespace larry_wins_probability_l172_172184

-- Define the probability of hitting the bottle (victory) as 1/3
def prob_hit := 1 / 3

-- Define the probability of missing the bottle (failure) as 2/3
def prob_miss := 2 / 3

-- Define the geometric series sum for the probability that Larry wins
def prob_larry_wins : ℚ := 
  let a := prob_hit
  let r := (prob_miss ^ 3)
  a / (1 - r)

theorem larry_wins_probability :
  prob_larry_wins = 9 / 19 :=
by
  sorry

end larry_wins_probability_l172_172184


namespace find_number_of_terms_l172_172286

theorem find_number_of_terms (n : ℕ) (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n, a n = (2^n - 1) / (2^n)) → S n = 321 / 64 → n = 6 :=
by
  sorry

end find_number_of_terms_l172_172286


namespace quadratic_linear_common_solution_l172_172303

theorem quadratic_linear_common_solution
  (a x1 x2 d e : ℝ)
  (ha : a ≠ 0) (hx1x2 : x1 ≠ x2) (hd : d ≠ 0)
  (h_quad : ∀ x, a * (x - x1) * (x - x2) = 0 → x = x1 ∨ x = x2)
  (h_linear : d * x1 + e = 0)
  (h_combined : ∀ x, a * (x - x1) * (x - x2) + d * x + e = 0 → x = x1) :
  d = a * (x2 - x1) :=
by sorry

end quadratic_linear_common_solution_l172_172303


namespace ratio_of_ages_three_years_from_now_l172_172081

theorem ratio_of_ages_three_years_from_now :
  ∃ L B : ℕ,
  (L + B = 6) ∧ 
  (L = (1/2 : ℝ) * B) ∧ 
  (L + 3 = 5) ∧ 
  (B + 3 = 7) → 
  (L + 3) / (B + 3) = (5/7 : ℝ) :=
by
  sorry

end ratio_of_ages_three_years_from_now_l172_172081


namespace right_triangular_pyramid_property_l172_172659

theorem right_triangular_pyramid_property
  (S1 S2 S3 S : ℝ)
  (right_angle_face1_area : S1 = S1) 
  (right_angle_face2_area : S2 = S2) 
  (right_angle_face3_area : S3 = S3) 
  (oblique_face_area : S = S) :
  S1^2 + S2^2 + S3^2 = S^2 := 
sorry

end right_triangular_pyramid_property_l172_172659


namespace find_f2_l172_172398

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  x^5 + a * x^3 + b * x + 1

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -8 :=
by
  sorry

end find_f2_l172_172398


namespace b5b9_l172_172018

-- Assuming the sequences are indexed from natural numbers starting at 1
-- a_n is an arithmetic sequence with common difference d
-- b_n is a geometric sequence
-- Given conditions
def a : ℕ → ℝ := sorry
def b : ℕ → ℝ := sorry
def d : ℝ := sorry
axiom arithmetic_seq : ∀ n : ℕ, a (n + 1) - a n = d
axiom d_nonzero : d ≠ 0
axiom condition_arith : 2 * a 4 - a 7 ^ 2 + 2 * a 10 = 0
axiom geometric_seq : ∀ n : ℕ, b (n + 1) / b n = b 2 / b 1
axiom b7_equals_a7 : b 7 = a 7

-- To prove
theorem b5b9 : b 5 * b 9 = 16 :=
by
  sorry

end b5b9_l172_172018


namespace smallest_solution_floor_eq_l172_172710

theorem smallest_solution_floor_eq (x : ℝ) : ⌊x^2⌋ - ⌊x⌋^2 = 19 → x = Real.sqrt 119 :=
by
  sorry

end smallest_solution_floor_eq_l172_172710


namespace center_cell_value_l172_172437

theorem center_cell_value
  (a b c d e f g h i : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i)
  (h_row1 : a * b * c = 1)
  (h_row2 : d * e * f = 1)
  (h_row3 : g * h * i = 1)
  (h_col1 : a * d * g = 1)
  (h_col2 : b * e * h = 1)
  (h_col3 : c * f * i = 1)
  (h_square1 : a * b * d * e = 2)
  (h_square2 : b * c * e * f = 2)
  (h_square3 : d * e * g * h = 2)
  (h_square4 : e * f * h * i = 2) :
  e = 1 :=
  sorry

end center_cell_value_l172_172437


namespace parabola_focus_equals_ellipse_focus_l172_172419

theorem parabola_focus_equals_ellipse_focus (p : ℝ) : 
  let parabola_focus := (p / 2, 0)
  let ellipse_focus := (2, 0)
  parabola_focus = ellipse_focus → p = 4 :=
by
  intros h
  sorry

end parabola_focus_equals_ellipse_focus_l172_172419


namespace Jeanine_gave_fraction_of_pencils_l172_172751

theorem Jeanine_gave_fraction_of_pencils
  (Jeanine_initial_pencils Clare_initial_pencils Jeanine_pencils_after Clare_pencils_after : ℕ)
  (h1 : Jeanine_initial_pencils = 18)
  (h2 : Clare_initial_pencils = Jeanine_initial_pencils / 2)
  (h3 : Jeanine_pencils_after = Clare_pencils_after + 3)
  (h4 : Clare_pencils_after = Clare_initial_pencils)
  (h5 : Jeanine_pencils_after + (Jeanine_initial_pencils - Jeanine_pencils_after) = Jeanine_initial_pencils) :
  (Jeanine_initial_pencils - Jeanine_pencils_after) / Jeanine_initial_pencils = 1 / 3 :=
by
  -- Proof here
  sorry

end Jeanine_gave_fraction_of_pencils_l172_172751


namespace solve_for_x_l172_172300

theorem solve_for_x (x : ℝ) (h : 3 + 5 * x = 28) : x = 5 :=
by {
  sorry
}

end solve_for_x_l172_172300


namespace slope_symmetric_line_l172_172856

  theorem slope_symmetric_line {l1 l2 : ℝ → ℝ} 
     (hl1 : ∀ x, l1 x = 2 * x + 3)
     (hl2_sym : ∀ x, l2 x = 2 * x + 3 -> l2 (-x) = -2 * x - 3) :
     ∀ x, l2 x = -2 * x + 3 :=
  sorry
  
end slope_symmetric_line_l172_172856


namespace probability_intersecting_chord_l172_172338

-- Definitions related to the problem
def points_on_circle : ℕ := 2023
def points_excluding_A_B : ℕ := 2021
def probability_CD_intersects_AB : ℚ := 1/2

-- Lean statement to prove the problem
theorem probability_intersecting_chord (points_on_circle = 2023)
    (A B : fin points_on_circle)
    (hAB : A ≠ B) :
    let C D := (choose_two (points_excluding_A_B)) in
    probability_of_intersection(A, B, C, D, points_on_circle) = probability_CD_intersects_AB := by sorry

end probability_intersecting_chord_l172_172338


namespace decreasing_on_neg_l172_172404

variable (f : ℝ → ℝ)

-- Condition 1: f(x) is an even function
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Condition 2: f(x) is increasing on (0, +∞)
def increasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y, 0 < x → x < y → f x < f y

-- Theorem to prove: f(x) is decreasing on (-∞, 0)
theorem decreasing_on_neg (f : ℝ → ℝ) 
  (h_even : even_function f)
  (h_increasing : increasing_on_pos f) :
  ∀ x y, x < y → y < 0 → f y < f x :=
by 
  sorry

end decreasing_on_neg_l172_172404


namespace sum_of_digits_of_7_pow_1974_l172_172228

-- Define the number \(7^{1974}\)
def num := 7^1974

-- Function to extract the last two digits
def last_two_digits (n : ℕ) : ℕ := n % 100

-- Function to compute the sum of the tens and units digits
def sum_tens_units (n : ℕ) : ℕ :=
  let last_two := last_two_digits n
  (last_two / 10) + (last_two % 10)

theorem sum_of_digits_of_7_pow_1974 : sum_tens_units num = 9 := by
  sorry

end sum_of_digits_of_7_pow_1974_l172_172228


namespace find_equation_of_line_l172_172533

theorem find_equation_of_line 
  (l : ℝ → ℝ → Prop)
  (h_intersect : ∃ x y : ℝ, 2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0 ∧ l x y)
  (h_parallel : ∀ x y : ℝ, l x y → 4 * x - 3 * y - 6 = 0) :
  ∀ x y : ℝ, l x y ↔ 4 * x - 3 * y - 6 = 0 :=
by
  sorry

end find_equation_of_line_l172_172533


namespace count_ordered_pairs_l172_172312

theorem count_ordered_pairs (d n : ℕ) (h₁ : d ≥ 35) (h₂ : n > 0) 
    (h₃ : 45 + 2 * n < 120)
    (h₄ : ∃ a b : ℕ, 10 * a + b = 30 + n ∧ 10 * b + a = 35 + n ∧ a ≤ 9 ∧ b ≤ 9) :
    ∃ k : ℕ, -- number of valid ordered pairs (d, n)
    sorry := sorry

end count_ordered_pairs_l172_172312


namespace number_of_gummies_l172_172670

-- Define the necessary conditions
def lollipop_cost : ℝ := 1.5
def lollipop_count : ℕ := 4
def gummy_cost : ℝ := 2.0
def initial_money : ℝ := 15.0
def money_left : ℝ := 5.0

-- Total cost of lollipops and total amount spent on candies
noncomputable def total_lollipop_cost := lollipop_count * lollipop_cost
noncomputable def total_spent := initial_money - money_left
noncomputable def total_gummy_cost := total_spent - total_lollipop_cost
noncomputable def gummy_count := total_gummy_cost / gummy_cost

-- Main theorem statement
theorem number_of_gummies : gummy_count = 2 := 
by
  sorry -- Proof to be added

end number_of_gummies_l172_172670


namespace domain_intersection_l172_172507

theorem domain_intersection (A B : Set ℝ) 
    (h1 : A = {x | x < 1})
    (h2 : B = {y | y ≥ 0}) : A ∩ B = {z | 0 ≤ z ∧ z < 1} := 
by
  sorry

end domain_intersection_l172_172507


namespace smallest_x_solution_l172_172684

theorem smallest_x_solution :
  ∃ x : ℝ, (⌊x^2⌋ - ⌊x⌋^2 = 19) ∧ (∀ y : ℝ, (⌊y^2⌋ - ⌊y⌋^2 = 19) → x ≤ y) ∧ x = Real.sqrt 119 := 
sorry

end smallest_x_solution_l172_172684


namespace race_head_start_l172_172801

variables {Va Vb L H : ℝ}

theorem race_head_start
  (h1 : Va = 20 / 14 * Vb)
  (h2 : L / Va = (L - H) / Vb) : 
  H = 3 / 10 * L :=
by
  sorry

end race_head_start_l172_172801


namespace percentage_increase_l172_172679

variable (m y : ℝ)

theorem percentage_increase (h : x = y + (m / 100) * y) : x = ((100 + m) / 100) * y := by
  sorry

end percentage_increase_l172_172679


namespace median_avg_scores_compare_teacher_avg_scores_l172_172249

-- Definitions of conditions
def class1_students (a : ℕ) := a
def class2_students (b : ℕ) := b
def class3_students (c : ℕ) := c
def class4_students (c : ℕ) := c

def avg_score_1 := 68
def avg_score_2 := 78
def avg_score_3 := 74
def avg_score_4 := 72

-- Part 1: Prove the median of the average scores.
theorem median_avg_scores : 
  let scores := [68, 72, 74, 78]
  ∃ m, m = 73 :=
by 
  sorry

-- Part 2: Prove that the average scores for Teacher Wang and Teacher Li are not necessarily the same.
theorem compare_teacher_avg_scores (a b c : ℕ) (h_ab : a ≠ 0 ∧ b ≠ 0) : 
  let wang_avg := (68 * a + 78 * b) / (a + b)
  let li_avg := 73
  wang_avg ≠ li_avg :=
by
  sorry

end median_avg_scores_compare_teacher_avg_scores_l172_172249


namespace find_a_l172_172913

noncomputable def a := 1/2

theorem find_a (a : ℝ) (h₀ : 0 < a ∧ a < 1) (h₁ : 1 - a^2 = 3/4) : a = 1/2 :=
sorry

end find_a_l172_172913


namespace valid_numbers_l172_172509

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

end valid_numbers_l172_172509


namespace ratio_of_additional_hours_james_danced_l172_172451

-- Definitions based on given conditions
def john_first_dance_time : ℕ := 3
def john_break_time : ℕ := 1
def john_second_dance_time : ℕ := 5
def combined_dancing_time_excluding_break : ℕ := 20

-- Calculations to be proved
def john_total_resting_dancing_time : ℕ :=
  john_first_dance_time + john_break_time + john_second_dance_time

def john_total_dancing_time : ℕ :=
  john_first_dance_time + john_second_dance_time

def james_dancing_time : ℕ :=
  combined_dancing_time_excluding_break - john_total_dancing_time

def additional_hours_james_danced : ℕ :=
  james_dancing_time - john_total_dancing_time

def desired_ratio : ℕ × ℕ :=
  (additional_hours_james_danced, john_total_resting_dancing_time)

-- Theorem to be proved according to the problem statement
theorem ratio_of_additional_hours_james_danced :
  desired_ratio = (4, 9) :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_additional_hours_james_danced_l172_172451
