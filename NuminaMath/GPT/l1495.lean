import Mathlib

namespace large_pyramid_tiers_l1495_149585

def surface_area_pyramid (n : ℕ) : ℕ :=
  4 * n^2 + 2 * n

theorem large_pyramid_tiers :
  (∃ n : ℕ, surface_area_pyramid n = 42) →
  (∃ n : ℕ, surface_area_pyramid n = 2352) →
  ∃ n : ℕ, surface_area_pyramid n = 2352 ∧ n = 24 :=
by
  sorry

end large_pyramid_tiers_l1495_149585


namespace students_in_front_of_Yuna_l1495_149515

-- Defining the total number of students
def total_students : ℕ := 25

-- Defining the number of students behind Yuna
def students_behind_Yuna : ℕ := 9

-- Defining Yuna's position from the end of the line
def Yuna_position_from_end : ℕ := students_behind_Yuna + 1

-- Statement to prove the number of students in front of Yuna
theorem students_in_front_of_Yuna : (total_students - Yuna_position_from_end) = 15 := by
  sorry

end students_in_front_of_Yuna_l1495_149515


namespace inscribed_sphere_radius_l1495_149596

theorem inscribed_sphere_radius 
  (a : ℝ) 
  (h_angle : ∀ (lateral_face : ℝ), lateral_face = 60) : 
  ∃ (r : ℝ), r = a * (Real.sqrt 3) / 6 :=
by
  sorry

end inscribed_sphere_radius_l1495_149596


namespace triangular_pyramid_volume_l1495_149584

theorem triangular_pyramid_volume
  (b : ℝ) (h : ℝ) (H : ℝ)
  (b_pos : b = 4.5) (h_pos : h = 6) (H_pos : H = 8) :
  let base_area := (b * h) / 2
  let volume := (base_area * H) / 3
  volume = 36 := by
  sorry

end triangular_pyramid_volume_l1495_149584


namespace inequality_proof_l1495_149525

/-- Given a and b are positive and satisfy the inequality ab > 2007a + 2008b,
    prove that a + b > (sqrt 2007 + sqrt 2008)^2 -/
theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b > 2007 * a + 2008 * b) :
  a + b > (Real.sqrt 2007 + Real.sqrt 2008) ^ 2 :=
by
  sorry

end inequality_proof_l1495_149525


namespace rug_shorter_side_l1495_149531

theorem rug_shorter_side (x : ℝ) :
  (64 - x * 7) / 64 = 0.78125 → x = 2 :=
by
  sorry

end rug_shorter_side_l1495_149531


namespace ellipse_problem_l1495_149534

theorem ellipse_problem :
  (∃ (k : ℝ) (a θ : ℝ), 
    (∀ x y : ℝ, y = k * (x + 3) → (x^2 / 25 + y^2 / 16 = 1)) ∧
    (a > -3) ∧
    (∃ x y : ℝ, (x = - (25 / 3) ∧ y = k * (x + 3)) ∧ 
                 (x = D_fst ∧ y = D_snd) ∧ -- Point D(a, θ)
                 (x = M_fst ∧ y = M_snd) ∧ -- Point M
                 (x = N_fst ∧ y = N_snd)) ∧ -- Point N
    (∃ x y : ℝ, (x = -3 ∧ y = 0))) → 
    a = 5 :=
sorry

end ellipse_problem_l1495_149534


namespace combined_weight_of_three_boxes_l1495_149576

theorem combined_weight_of_three_boxes (a b c d : ℕ) (h₁ : a + b = 132) (h₂ : a + c = 136) (h₃ : b + c = 138) (h₄ : d = 60) : 
  a + b + c = 203 :=
sorry

end combined_weight_of_three_boxes_l1495_149576


namespace degree_g_of_degree_f_and_h_l1495_149574

noncomputable def degree (p : ℕ) := p -- definition to represent degree of polynomials

theorem degree_g_of_degree_f_and_h (f g : ℕ → ℕ) (h : ℕ → ℕ) 
  (deg_h : ℕ) (deg_f : ℕ) (deg_10 : deg_h = 10) (deg_3 : deg_f = 3) 
  (h_eq : ∀ x, degree (h x) = degree (f (g x)) + degree x ^ 5) :
  degree (g 0) = 4 :=
by
  sorry

end degree_g_of_degree_f_and_h_l1495_149574


namespace intersection_of_A_and_B_l1495_149507

def setA : Set ℝ := {-1, 1, 2, 4}
def setB : Set ℝ := {x | abs (x - 1) ≤ 1}

theorem intersection_of_A_and_B : setA ∩ setB = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l1495_149507


namespace total_notes_count_l1495_149558

theorem total_notes_count :
  ∀ (rows : ℕ) (notes_per_row : ℕ) (blue_notes_per_red : ℕ) (additional_blue_notes : ℕ),
  rows = 5 →
  notes_per_row = 6 →
  blue_notes_per_red = 2 →
  additional_blue_notes = 10 →
  (rows * notes_per_row + (rows * notes_per_row * blue_notes_per_red + additional_blue_notes)) = 100 := by
  intros rows notes_per_row blue_notes_per_red additional_blue_notes
  sorry

end total_notes_count_l1495_149558


namespace PB_distance_eq_l1495_149560

theorem PB_distance_eq {
  A B C D P : Type
} (PA PD PC : ℝ) (hPA: PA = 6) (hPD: PD = 8) (hPC: PC = 10)
  (h_equidistant: ∃ y : ℝ, PA^2 + y^2 = PB^2 ∧ PD^2 + y^2 = PC^2) :
  ∃ PB : ℝ, PB = 6 * Real.sqrt 2 := 
by
  sorry

end PB_distance_eq_l1495_149560


namespace irreducible_fractions_properties_l1495_149539

theorem irreducible_fractions_properties : 
  let f1 := 11 / 2
  let f2 := 11 / 6
  let f3 := 11 / 3
  let reciprocal_sum := (2 / 11) + (6 / 11) + (3 / 11)
  (f1 + f2 + f3 = 11) ∧ (reciprocal_sum = 1) :=
by
  sorry

end irreducible_fractions_properties_l1495_149539


namespace line_through_two_points_l1495_149502

theorem line_through_two_points :
  ∃ (m b : ℝ), (∀ x y : ℝ, (x, y) = (-2, 4) ∨ (x, y) = (-1, 3) → y = m * x + b) ∧ b = 2 ∧ m = -1 :=
by
  sorry

end line_through_two_points_l1495_149502


namespace total_unique_items_l1495_149548

-- Define the conditions
def shared_albums : ℕ := 12
def total_andrew_albums : ℕ := 23
def exclusive_andrew_memorabilia : ℕ := 5
def exclusive_john_albums : ℕ := 8

-- Define the number of unique items in Andrew's and John's collection 
def unique_andrew_albums : ℕ := total_andrew_albums - shared_albums
def unique_total_items : ℕ := unique_andrew_albums + exclusive_john_albums + exclusive_andrew_memorabilia

-- The proof goal
theorem total_unique_items : unique_total_items = 24 := by
  -- Proof steps would go here
  sorry

end total_unique_items_l1495_149548


namespace volume_ratio_l1495_149503

def volume_of_cube (side_length : ℕ) : ℕ :=
  side_length * side_length * side_length

theorem volume_ratio 
  (hyungjin_side_length_cm : ℕ)
  (kyujun_side_length_m : ℕ)
  (h1 : hyungjin_side_length_cm = 100)
  (h2 : kyujun_side_length_m = 2) :
  volume_of_cube (kyujun_side_length_m * 100) = 8 * volume_of_cube hyungjin_side_length_cm :=
by
  sorry

end volume_ratio_l1495_149503


namespace age_problem_l1495_149535

variable (A B : ℕ)

theorem age_problem (h1 : A + 10 = 2 * (B - 10)) (h2 : A = B + 5) : B = 35 := by
  sorry

end age_problem_l1495_149535


namespace pure_alcohol_addition_l1495_149522

theorem pure_alcohol_addition (x : ℝ) (h1 : 3 / 10 * 10 = 3)
    (h2 : 60 / 100 * (10 + x) = (3 + x) ) : x = 7.5 :=
sorry

end pure_alcohol_addition_l1495_149522


namespace find_a_in_triangle_l1495_149518

theorem find_a_in_triangle (a b c : ℝ) (A B C : ℝ) 
  (h1 : b^2 - c^2 + 2 * a = 0) 
  (h2 : Real.tan C / Real.tan B = 3) 
  : a = 4 :=
  sorry

end find_a_in_triangle_l1495_149518


namespace valid_N_values_l1495_149556

def N_values (N : ℕ) : Prop :=
  (∀ k, k = N - 8 → (22 < N ∧ N ≤ 25))

-- Main theorem statement without proof
theorem valid_N_values (N : ℕ) (h : ∀ k, k = N - 8 → N_values N) : 
  (N = 23 ∨ N = 24 ∨ N = 25) :=
by
  sorry

end valid_N_values_l1495_149556


namespace spending_total_march_to_july_l1495_149554

/-- Given the conditions:
  1. Total amount spent by the beginning of March is 1.2 million,
  2. Total amount spent by the end of July is 5.4 million,
  Prove that the total amount spent during March, April, May, June, and July is 4.2 million. -/
theorem spending_total_march_to_july
  (spent_by_end_of_feb : ℝ)
  (spent_by_end_of_july : ℝ)
  (h1 : spent_by_end_of_feb = 1.2)
  (h2 : spent_by_end_of_july = 5.4) :
  spent_by_end_of_july - spent_by_end_of_feb = 4.2 :=
by
  sorry

end spending_total_march_to_july_l1495_149554


namespace rectangle_area_x_l1495_149511

theorem rectangle_area_x (x : ℕ) (h1 : x > 0) (h2 : 5 * x = 45) : x = 9 := 
by
  -- proof goes here
  sorry

end rectangle_area_x_l1495_149511


namespace inequality_holds_iff_l1495_149580

theorem inequality_holds_iff (a : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → x^2 + (a - 4) * x + 4 > 0) ↔ a > 0 :=
by
  sorry

end inequality_holds_iff_l1495_149580


namespace simplify_fraction_l1495_149508

theorem simplify_fraction :
  (1 / ((1 / (Real.sqrt 2 + 1)) + (2 / (Real.sqrt 3 - 1)) + (3 / (Real.sqrt 5 + 2)))) =
  (1 / (Real.sqrt 2 + 2 * Real.sqrt 3 + 3 * Real.sqrt 5 - 5)) :=
by
  sorry

end simplify_fraction_l1495_149508


namespace eccentricity_of_ellipse_l1495_149559

theorem eccentricity_of_ellipse : 
  ∀ (a b c e : ℝ), a^2 = 16 → b^2 = 8 → c^2 = a^2 - b^2 → e = c / a → e = (Real.sqrt 2) / 2 := 
by 
  intros a b c e ha hb hc he
  sorry

end eccentricity_of_ellipse_l1495_149559


namespace rocking_chair_legs_l1495_149598

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

end rocking_chair_legs_l1495_149598


namespace ordered_quadruple_solution_exists_l1495_149547

theorem ordered_quadruple_solution_exists (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) : 
  a^2 * b = c ∧ b * c^2 = a ∧ c * a^2 = b ∧ a + b + c = d → (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 3) :=
by
  sorry

end ordered_quadruple_solution_exists_l1495_149547


namespace ellipse_foci_coordinates_l1495_149552

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ),
    x^2 / 16 + y^2 / 25 = 1 → (x = 0 ∧ y = 3) ∨ (x = 0 ∧ y = -3) :=
by
  sorry

end ellipse_foci_coordinates_l1495_149552


namespace cost_of_show_dogs_l1495_149578

noncomputable def cost_per_dog : ℕ → ℕ → ℕ → ℕ
| total_revenue, total_profit, number_of_dogs => (total_revenue - total_profit) / number_of_dogs

theorem cost_of_show_dogs {revenue_per_puppy number_of_puppies profit number_of_dogs : ℕ}
  (h_puppies: number_of_puppies = 6)
  (h_revenue_per_puppy : revenue_per_puppy = 350)
  (h_profit : profit = 1600)
  (h_number_of_dogs : number_of_dogs = 2)
:
  cost_per_dog (number_of_puppies * revenue_per_puppy) profit number_of_dogs = 250 :=
by
  sorry

end cost_of_show_dogs_l1495_149578


namespace effective_average_speed_l1495_149514

def rowing_speed_with_stream := 16 -- km/h
def rowing_speed_against_stream := 6 -- km/h
def stream1_effect := 2 -- km/h
def stream2_effect := -1 -- km/h
def stream3_effect := 3 -- km/h
def opposing_wind := 1 -- km/h

theorem effective_average_speed :
  ((rowing_speed_with_stream + stream1_effect - opposing_wind) + 
   (rowing_speed_against_stream + stream2_effect - opposing_wind) + 
   (rowing_speed_with_stream + stream3_effect - opposing_wind)) / 3 = 13 := 
by
  sorry

end effective_average_speed_l1495_149514


namespace mark_charged_more_hours_l1495_149555

variable {p k m : ℕ}

theorem mark_charged_more_hours (h1 : p + k + m = 216)
                                (h2 : p = 2 * k)
                                (h3 : p = m / 3) :
                                m - k = 120 :=
sorry

end mark_charged_more_hours_l1495_149555


namespace sum_9_to_12_l1495_149505

variable {a : ℕ → ℝ} -- Define the arithmetic sequence
variables {S : ℕ → ℝ} -- Define the sum function of the sequence

-- Define the conditions given in the problem
def S_4 : ℝ := 8
def S_8 : ℝ := 20

-- The goal is to show that the sum of the 9th to 12th terms is 16
theorem sum_9_to_12 : (a 9) + (a 10) + (a 11) + (a 12) = 16 :=
by
  sorry

end sum_9_to_12_l1495_149505


namespace range_of_a_l1495_149562

noncomputable def f (x : ℝ) : ℝ := (x^2 + x + 16) / x

theorem range_of_a (a : ℝ) (h1 : 2 ≤ a) (h2 : (∀ x, 2 ≤ x ∧ x ≤ a → 9 ≤ f x ∧ f x ≤ 11)) : 4 ≤ a ∧ a ≤ 8 := by
  sorry

end range_of_a_l1495_149562


namespace merck_hourly_rate_l1495_149536

-- Define the relevant data from the problem
def hours_donaldsons : ℕ := 7
def hours_merck : ℕ := 6
def hours_hille : ℕ := 3
def total_earnings : ℕ := 273

-- Define the total hours based on the conditions
def total_hours : ℕ := hours_donaldsons + hours_merck + hours_hille

-- Define what we want to prove:
def hourly_rate := total_earnings / total_hours

theorem merck_hourly_rate : hourly_rate = 273 / (7 + 6 + 3) := by
  sorry

end merck_hourly_rate_l1495_149536


namespace adoption_time_l1495_149553

theorem adoption_time
  (p0 : ℕ) (p1 : ℕ) (rate : ℕ)
  (p0_eq : p0 = 10) (p1_eq : p1 = 15) (rate_eq : rate = 7) :
  Nat.ceil ((p0 + p1) / rate) = 4 := by
  sorry

end adoption_time_l1495_149553


namespace problem_inequality_l1495_149500

theorem problem_inequality {n : ℕ} {a : ℕ → ℕ} (h : ∀ i j : ℕ, 1 ≤ i → i < j → j ≤ n → a i < a j → (a j - a i) ∣ a i) 
  (h_sorted : ∀ i j : ℕ, 1 ≤ i → i < j → j ≤ n → a i < a j)
  (h_pos : ∀ i : ℕ, 1 ≤ i → i ≤ n → 0 < a i) 
  (i j : ℕ) (hi : 1 ≤ i) (hij : i < j) (hj : j ≤ n) : i * a j ≤ j * a i := 
sorry

end problem_inequality_l1495_149500


namespace average_age_when_youngest_born_l1495_149542

theorem average_age_when_youngest_born (n : ℕ) (current_average_age youngest age_difference total_ages : ℝ)
  (hc1 : n = 7)
  (hc2 : current_average_age = 30)
  (hc3 : youngest = 6)
  (hc4 : age_difference = youngest * 6)
  (hc5 : total_ages = n * current_average_age - age_difference) :
  total_ages / n = 24.857
:= sorry

end average_age_when_youngest_born_l1495_149542


namespace b_value_rational_polynomial_l1495_149529

theorem b_value_rational_polynomial (a b : ℚ) :
  (Polynomial.aeval (2 + Real.sqrt 3) (Polynomial.C (-15) + Polynomial.C b * X + Polynomial.C a * X^2 + X^3 : Polynomial ℚ) = 0) →
  b = -44 :=
by
  sorry

end b_value_rational_polynomial_l1495_149529


namespace find_dividend_and_divisor_l1495_149561

theorem find_dividend_and_divisor (quotient : ℕ) (remainder : ℕ) (total : ℕ) (dividend divisor : ℕ) :
  quotient = 13 ∧ remainder = 6 ∧ total = 137 ∧ (dividend + divisor + quotient + remainder = total)
  ∧ dividend = 13 * divisor + remainder → 
  dividend = 110 ∧ divisor = 8 :=
by
  intro h
  sorry

end find_dividend_and_divisor_l1495_149561


namespace xiaomings_possible_score_l1495_149533

def average_score_class_A : ℤ := 87
def average_score_class_B : ℤ := 82

theorem xiaomings_possible_score (x : ℤ) :
  (average_score_class_B < x ∧ x < average_score_class_A) → x = 85 :=
by sorry

end xiaomings_possible_score_l1495_149533


namespace inequality_S_l1495_149583

def S (n m : ℕ) : ℕ := sorry

theorem inequality_S (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  S (2015 * n) n * S (2015 * m) m ≥ S (2015 * n) m * S (2015 * m) n :=
sorry

end inequality_S_l1495_149583


namespace Johnson_farm_budget_l1495_149532

variable (total_land : ℕ) (corn_cost_per_acre : ℕ) (wheat_cost_per_acre : ℕ)
variable (acres_wheat : ℕ) (acres_corn : ℕ)

def total_money (total_land corn_cost_per_acre wheat_cost_per_acre acres_wheat acres_corn : ℕ) : ℕ :=
  acres_corn * corn_cost_per_acre + acres_wheat * wheat_cost_per_acre

theorem Johnson_farm_budget :
  total_land = 500 ∧
  corn_cost_per_acre = 42 ∧
  wheat_cost_per_acre = 30 ∧
  acres_wheat = 200 ∧
  acres_corn = total_land - acres_wheat →
  total_money total_land corn_cost_per_acre wheat_cost_per_acre acres_wheat acres_corn = 18600 := by
  sorry

end Johnson_farm_budget_l1495_149532


namespace mikey_jelly_beans_l1495_149504

theorem mikey_jelly_beans :
  let napoleon_jelly_beans := 17
  let sedrich_jelly_beans := napoleon_jelly_beans + 4
  let total_jelly_beans := napoleon_jelly_beans + sedrich_jelly_beans
  let twice_sum := 2 * total_jelly_beans
  ∃ mikey_jelly_beans, 4 * mikey_jelly_beans = twice_sum → mikey_jelly_beans = 19 :=
by
  intro napoleon_jelly_beans
  intro sedrich_jelly_beans
  intro total_jelly_beans
  intro twice_sum
  use 19
  sorry

end mikey_jelly_beans_l1495_149504


namespace range_of_a_l1495_149509

theorem range_of_a (a x : ℝ) (p : 0.5 ≤ x ∧ x ≤ 1) (q : (x - a) * (x - a - 1) > 0) :
  (0 ≤ a ∧ a ≤ 0.5) :=
by 
  sorry

end range_of_a_l1495_149509


namespace base_addition_is_10_l1495_149579

-- The problem states that adding two numbers in a particular base results in a third number in the same base.
def valid_base_10_addition (n m k b : ℕ) : Prop :=
  let n_b := n / b^2 * b^2 + (n / b % b) * b + n % b
  let m_b := m / b^2 * b^2 + (m / b % b) * b + m % b
  let k_b := k / b^2 * b^2 + (k / b % b) * b + k % b
  n_b + m_b = k_b

theorem base_addition_is_10 : valid_base_10_addition 172 156 340 10 :=
  sorry

end base_addition_is_10_l1495_149579


namespace sqrt_rational_rational_l1495_149521

theorem sqrt_rational_rational 
  (a b : ℚ) 
  (h : ∃ r : ℚ, r = (a : ℝ).sqrt + (b : ℝ).sqrt) : 
  (∃ p : ℚ, p = (a : ℝ).sqrt) ∧ (∃ q : ℚ, q = (b : ℝ).sqrt) := 
sorry

end sqrt_rational_rational_l1495_149521


namespace quad_roots_sum_l1495_149563

theorem quad_roots_sum {x₁ x₂ : ℝ} (h1 : x₁ + x₂ = 5) (h2 : x₁ * x₂ = -6) :
  1 / x₁ + 1 / x₂ = -5 / 6 :=
by
  sorry

end quad_roots_sum_l1495_149563


namespace find_ab_l1495_149517

theorem find_ab (a b : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 2 →
  (3 * x - 2 < a + 1 ∧ 6 - 2 * x < b + 2)) →
  a = 3 ∧ b = 6 :=
by
  sorry

end find_ab_l1495_149517


namespace find_a_plus_d_l1495_149526

theorem find_a_plus_d (a b c d : ℕ)
  (h1 : a + b = 14)
  (h2 : b + c = 9)
  (h3 : c + d = 3) : 
  a + d = 2 :=
by sorry

end find_a_plus_d_l1495_149526


namespace quadratic_eqns_mod_7_l1495_149530

/-- Proving the solutions for quadratic equations in arithmetic modulo 7. -/
theorem quadratic_eqns_mod_7 :
  (¬ ∃ x : ℤ, (5 * x^2 + 3 * x + 1) % 7 = 0) ∧
  (∃! x : ℤ, (x^2 + 3 * x + 4) % 7 = 0 ∧ x % 7 = 2) ∧
  (∃ x1 x2 : ℤ, (x1 ^ 2 - 2 * x1 - 3) % 7 = 0 ∧ (x2 ^ 2 - 2 * x2 - 3) % 7 = 0 ∧ 
              x1 % 7 = 3 ∧ x2 % 7 = 6) :=
by
  sorry

end quadratic_eqns_mod_7_l1495_149530


namespace least_possible_value_l1495_149523

theorem least_possible_value (x y : ℝ) : (3 * x * y - 1)^2 + (x - y)^2 ≥ 1 := sorry

end least_possible_value_l1495_149523


namespace coats_count_l1495_149540

def initial_minks : Nat := 30
def babies_per_mink : Nat := 6
def minks_per_coat : Nat := 15

def total_minks : Nat := initial_minks + (initial_minks * babies_per_mink)
def remaining_minks : Nat := total_minks / 2

theorem coats_count : remaining_minks / minks_per_coat = 7 := by
  -- Proof goes here
  sorry

end coats_count_l1495_149540


namespace relay_race_arrangements_l1495_149581

noncomputable def number_of_arrangements (athletes : Finset ℕ) (a b : ℕ) : ℕ :=
  (athletes.erase a).card.factorial * ((athletes.erase b).card.factorial - 2) * (athletes.card.factorial / ((athletes.card - 4).factorial)) / 4

theorem relay_race_arrangements :
  let athletes := {0, 1, 2, 3, 4, 5}
  number_of_arrangements athletes 0 1 = 252 := 
by
  sorry

end relay_race_arrangements_l1495_149581


namespace second_train_speed_l1495_149512

theorem second_train_speed (v : ℝ) :
  (∃ t : ℝ, 20 * t = v * t + 75 ∧ 20 * t + v * t = 675) → v = 16 :=
by
  sorry

end second_train_speed_l1495_149512


namespace find_n_l1495_149557

theorem find_n (n : ℕ) (h : (1 + n + (n * (n - 1)) / 2) / 2^n = 7 / 32) : n = 6 :=
sorry

end find_n_l1495_149557


namespace distance_between_centers_of_tangent_circles_l1495_149524

theorem distance_between_centers_of_tangent_circles
  (R r d : ℝ) (h1 : R = 8) (h2 : r = 3) (h3 : d = R + r) : d = 11 :=
by
  -- Insert proof here
  sorry

end distance_between_centers_of_tangent_circles_l1495_149524


namespace increasing_function_on_interval_l1495_149577

section
  variable (a b : ℝ)
  def f (x : ℝ) : ℝ := |x^2 - 2*a*x + b|

  theorem increasing_function_on_interval (h : a^2 - b ≤ 0) :
    ∀ x y : ℝ, a ≤ x → x ≤ y → f x ≤ f y := 
  sorry
end

end increasing_function_on_interval_l1495_149577


namespace weighted_averages_correct_l1495_149594

def group_A_boys : ℕ := 20
def group_B_boys : ℕ := 25
def group_C_boys : ℕ := 15

def group_A_weight : ℝ := 50.25
def group_B_weight : ℝ := 45.15
def group_C_weight : ℝ := 55.20

def group_A_height : ℝ := 160
def group_B_height : ℝ := 150
def group_C_height : ℝ := 165

def group_A_age : ℝ := 15
def group_B_age : ℝ := 14
def group_C_age : ℝ := 16

def group_A_athletic : ℝ := 0.60
def group_B_athletic : ℝ := 0.40
def group_C_athletic : ℝ := 0.75

noncomputable def total_boys : ℕ := group_A_boys + group_B_boys + group_C_boys

noncomputable def weighted_average_height : ℝ := 
    (group_A_boys * group_A_height + group_B_boys * group_B_height + group_C_boys * group_C_height) / total_boys

noncomputable def weighted_average_weight : ℝ := 
    (group_A_boys * group_A_weight + group_B_boys * group_B_weight + group_C_boys * group_C_weight) / total_boys

noncomputable def weighted_average_age : ℝ := 
    (group_A_boys * group_A_age + group_B_boys * group_B_age + group_C_boys * group_C_age) / total_boys

noncomputable def weighted_average_athletic : ℝ := 
    (group_A_boys * group_A_athletic + group_B_boys * group_B_athletic + group_C_boys * group_C_athletic) / total_boys

theorem weighted_averages_correct :
  weighted_average_height = 157.08 ∧
  weighted_average_weight = 49.36 ∧
  weighted_average_age = 14.83 ∧
  weighted_average_athletic = 0.5542 := 
  by
    sorry

end weighted_averages_correct_l1495_149594


namespace geometric_sequence_condition_l1495_149592

-- Given the sum of the first n terms of the sequence {a_n} is S_n = 2^n + c,
-- we need to prove that the sequence {a_n} is a geometric sequence if and only if c = -1.
theorem geometric_sequence_condition (c : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) :
  (∀ n, S n = 2^n + c) →
  (∀ n ≥ 2, a n = S n - S (n - 1)) →
  (∃ q, ∀ n ≥ 1, a n = a 1 * q ^ (n - 1)) ↔ (c = -1) :=
by
  -- Proof skipped
  sorry

end geometric_sequence_condition_l1495_149592


namespace white_balls_in_bag_l1495_149588

theorem white_balls_in_bag:
  ∀ (total balls green yellow red purple : Nat),
  total = 60 →
  green = 18 →
  yellow = 8 →
  red = 5 →
  purple = 7 →
  (1 - 0.8) = (red + purple : ℚ) / total →
  (W + green + yellow = total - (red + purple : ℚ)) →
  W = 22 :=
by
  intros total balls green yellow red purple ht hg hy hr hp hprob heqn
  sorry

end white_balls_in_bag_l1495_149588


namespace quadratic_real_roots_range_l1495_149545

theorem quadratic_real_roots_range (m : ℝ) :
  (∀ x : ℝ, (m - 1) * x^2 + 2 * x + 1 = 0 → 
    (∃ x1 x2 : ℝ, x = x1 ∧ x = x2 ∧ x1 = x2 → true)) → 
    m ≤ 2 ∧ m ≠ 1 :=
by
  sorry

end quadratic_real_roots_range_l1495_149545


namespace pathway_area_ratio_l1495_149586

theorem pathway_area_ratio (AB AD: ℝ) (r: ℝ) (A_rectangle A_circles: ℝ):
  AB = 24 → (AD / AB) = (4 / 3) → r = AB / 2 → 
  A_rectangle = AD * AB → A_circles = π * r^2 →
  (A_rectangle / A_circles) = 16 / (3 * π) :=
by
  sorry

end pathway_area_ratio_l1495_149586


namespace algebra_expression_correct_l1495_149587

theorem algebra_expression_correct {x y : ℤ} (h : x + 2 * y + 1 = 3) : 2 * x + 4 * y + 1 = 5 :=
  sorry

end algebra_expression_correct_l1495_149587


namespace complex_division_l1495_149516

theorem complex_division (i : ℂ) (hi : i^2 = -1) : (2 * i) / (1 + i) = 1 + i :=
by
  sorry

end complex_division_l1495_149516


namespace find_width_fabric_width_is_3_l1495_149546

variable (Area Length : ℝ)
variable (Width : ℝ)

theorem find_width (h1 : Area = 24) (h2 : Length = 8) :
  Width = Area / Length :=
sorry

theorem fabric_width_is_3 (h1 : Area = 24) (h2 : Length = 8) :
  (Area / Length) = 3 :=
by
  have h : Area / Length = 3 := by sorry
  exact h

end find_width_fabric_width_is_3_l1495_149546


namespace pebbles_ratio_l1495_149544

variable (S : ℕ)

theorem pebbles_ratio :
  let initial_pebbles := 18
  let skipped_pebbles := 9
  let additional_pebbles := 30
  let final_pebbles := 39
  initial_pebbles - skipped_pebbles + additional_pebbles = final_pebbles →
  (skipped_pebbles : ℚ) / initial_pebbles = 1 / 2 :=
by
  intros
  sorry

end pebbles_ratio_l1495_149544


namespace complement_U_A_l1495_149543

open Set

-- Definitions of the universal set U and the set A
def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1, 2}

-- Proof statement: the complement of A with respect to U is {3}
theorem complement_U_A : U \ A = {3} :=
by
  sorry

end complement_U_A_l1495_149543


namespace parabola_focus_distance_l1495_149537

theorem parabola_focus_distance (x y : ℝ) (h1 : y^2 = 4 * x) (h2 : (x - 1)^2 + y^2 = 100) : x = 9 :=
sorry

end parabola_focus_distance_l1495_149537


namespace solution_l1495_149589

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x)

def is_monotonically_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x → x < y → y ≤ b → f x ≤ f y

theorem solution (a b : ℝ) (H : a = 5 * Real.pi / 8 ∧ b = 7 * Real.pi / 8) :
  is_monotonically_increasing g a b :=
sorry

end solution_l1495_149589


namespace inequality_solver_l1495_149593

variable {m n x : ℝ}

-- Main theorem statement validating the instances described above.
theorem inequality_solver (h : 2 * m * x + 3 < 3 * x + n) :
  (2 * m - 3 > 0 ∧ x < (n - 3) / (2 * m - 3)) ∨ 
  (2 * m - 3 < 0 ∧ x > (n - 3) / (2 * m - 3)) ∨ 
  (m = 3 / 2 ∧ n > 3 ∧ ∀ x : ℝ, true) ∨ 
  (m = 3 / 2 ∧ n ≤ 3 ∧ ∀ x : ℝ, false) :=
sorry

end inequality_solver_l1495_149593


namespace p_twice_q_in_future_years_l1495_149527

-- We define the ages of p and q
def p_current_age : ℕ := 33
def q_current_age : ℕ := 11

-- Third condition that is redundant given the values we already defined
def age_relation : Prop := (p_current_age = 3 * q_current_age)

-- Number of years in the future when p will be twice as old as q
def future_years_when_twice : ℕ := 11

-- Prove that in future_years_when_twice years, p will be twice as old as q
theorem p_twice_q_in_future_years :
  ∀ t : ℕ, t = future_years_when_twice → (p_current_age + t = 2 * (q_current_age + t)) := by
  sorry

end p_twice_q_in_future_years_l1495_149527


namespace max_lg_sum_eq_one_min_inv_sum_eq_specific_value_l1495_149568

theorem max_lg_sum_eq_one {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y = 20) :
  ∀ u, u = Real.log x + Real.log y → u ≤ 1 :=
sorry

theorem min_inv_sum_eq_specific_value {x y : ℝ} (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 5 * y = 20) :
  ∀ v, v = (1 / x) + (1 / y) → v ≥ (7 + 2 * Real.sqrt 10) / 20 :=
sorry

end max_lg_sum_eq_one_min_inv_sum_eq_specific_value_l1495_149568


namespace age_difference_64_l1495_149566

variables (Patrick Michael Monica : ℕ)
axiom age_ratio_1 : ∃ (x : ℕ), Patrick = 3 * x ∧ Michael = 5 * x
axiom age_ratio_2 : ∃ (y : ℕ), Michael = 3 * y ∧ Monica = 5 * y
axiom age_sum : Patrick + Michael + Monica = 196

theorem age_difference_64 : Monica - Patrick = 64 :=
by {
  sorry
}

end age_difference_64_l1495_149566


namespace integer_roots_of_polynomial_l1495_149564

theorem integer_roots_of_polynomial :
  ∀ x : ℤ, (x^3 - 3 * x^2 - 13 * x + 15 = 0) → (x = -3 ∨ x = 1 ∨ x = 5) :=
by
  sorry

end integer_roots_of_polynomial_l1495_149564


namespace sequence_a3_equals_1_over_3_l1495_149591

theorem sequence_a3_equals_1_over_3 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 1) 
  (h2 : ∀ n ≥ 2, a n = 1 - 1 / (a (n - 1) + 1)) : 
  a 3 = 1 / 3 :=
sorry

end sequence_a3_equals_1_over_3_l1495_149591


namespace combined_distance_proof_l1495_149569

/-- Define the distances walked by Lionel, Esther, and Niklaus in their respective units -/
def lionel_miles : ℕ := 4
def esther_yards : ℕ := 975
def niklaus_feet : ℕ := 1287

/-- Define the conversion factors -/
def miles_to_feet : ℕ := 5280
def yards_to_feet : ℕ := 3

/-- The total combined distance in feet -/
def total_distance_feet : ℕ :=
  (lionel_miles * miles_to_feet) + (esther_yards * yards_to_feet) + niklaus_feet

theorem combined_distance_proof : total_distance_feet = 24332 := by
  -- expand definitions and calculations here...
  -- lionel = 4 * 5280 = 21120
  -- esther = 975 * 3 = 2925
  -- niklaus = 1287
  -- sum = 21120 + 2925 + 1287 = 24332
  sorry

end combined_distance_proof_l1495_149569


namespace jeremy_goal_product_l1495_149597

theorem jeremy_goal_product 
  (g1 g2 g3 g4 g5 : ℕ) 
  (total5 : g1 + g2 + g3 + g4 + g5 = 13)
  (g6 g7 : ℕ) 
  (h6 : g6 < 10) 
  (h7 : g7 < 10) 
  (avg6 : (13 + g6) % 6 = 0) 
  (avg7 : (13 + g6 + g7) % 7 = 0) :
  g6 * g7 = 15 := 
sorry

end jeremy_goal_product_l1495_149597


namespace find_flag_count_l1495_149506

-- Definitions of conditions
inductive Color
| purple
| gold
| silver

-- Function to count valid flags
def countValidFlags : Nat :=
  let first_stripe_choices := 3
  let second_stripe_choices := 2
  let third_stripe_choices := 2
  first_stripe_choices * second_stripe_choices * third_stripe_choices

-- Statement to prove
theorem find_flag_count : countValidFlags = 12 := by
  sorry

end find_flag_count_l1495_149506


namespace twelve_xy_leq_fourx_1_y_9y_1_x_l1495_149582

theorem twelve_xy_leq_fourx_1_y_9y_1_x
  (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hxy : x + y ≤ 1) :
  12 * x * y ≤ 4 * x * (1 - y) + 9 * y * (1 - x) :=
  sorry

end twelve_xy_leq_fourx_1_y_9y_1_x_l1495_149582


namespace x_coordinate_of_second_point_l1495_149570

variable (m n : ℝ)

theorem x_coordinate_of_second_point
  (h1 : m = 2 * n + 5)
  (h2 : (m + 5) = 2 * (n + 2.5) + 5) :
  (m + 5) = m + 5 :=
by
  sorry

end x_coordinate_of_second_point_l1495_149570


namespace sum_of_tens_l1495_149571

theorem sum_of_tens (n : ℕ) (h : n = 100^10) : ∃ k : ℕ, n = 10 * k ∧ k = 10^19 :=
by
  sorry

end sum_of_tens_l1495_149571


namespace find_angle_x_l1495_149501

def angle_ABC := 124
def angle_BAD := 30
def angle_BDA := 28
def angle_ABD := 180 - angle_ABC
def angle_x := 180 - (angle_BAD + angle_ABD)

theorem find_angle_x : angle_x = 94 :=
by
  repeat { sorry }

end find_angle_x_l1495_149501


namespace find_value_of_expression_l1495_149549

noncomputable def quadratic_function (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 2

theorem find_value_of_expression (a b : ℝ) (h : quadratic_function a b (-1) = 0) :
  2 * a - 2 * b = -4 :=
sorry

end find_value_of_expression_l1495_149549


namespace average_score_for_girls_l1495_149567

variable (A a B b : ℕ)
variable (h1 : 71 * A + 76 * a = 74 * (A + a))
variable (h2 : 81 * B + 90 * b = 84 * (B + b))
variable (h3 : 71 * A + 81 * B = 79 * (A + B))

theorem average_score_for_girls
  (h1 : 71 * A + 76 * a = 74 * (A + a))
  (h2 : 81 * B + 90 * b = 84 * (B + b))
  (h3 : 71 * A + 81 * B = 79 * (A + B))
  : (76 * a + 90 * b) / (a + b) = 84 := by
  sorry

end average_score_for_girls_l1495_149567


namespace find_a2018_l1495_149541

-- Definitions based on given conditions
def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 0.5 ∧ ∀ n, a (n + 1) = 1 - 1 / (a n)

-- The statement to prove
theorem find_a2018 (a : ℕ → ℝ) (h : seq a) : a 2018 = -1 := by
  sorry

end find_a2018_l1495_149541


namespace parabola_equation_max_slope_OQ_l1495_149528

-- Definition of the problem for part (1)
theorem parabola_equation (p : ℝ) (hp : p = 2) : (∀ x y : ℝ, y^2 = 2 * p * x ↔ y^2 = 4 * x) :=
by {
  sorry
}

-- Definition of the problem for part (2)
theorem max_slope_OQ (m n : ℝ) (hp : y^2 = 4 * x)
  (h_relate : ∀ P Q F : (ℝ × ℝ), P.1 * Q.1 + P.2 * Q.2 = 9 * (Q.1 - F.1) * (Q.2 - F.2))
  : (∀ Q : (ℝ × ℝ), max (Q.2 / Q.1) = 1/3) :=
by {
  sorry
}

end parabola_equation_max_slope_OQ_l1495_149528


namespace maximize_profit_l1495_149551

def cups_sold (p : ℝ) : ℝ :=
  150 - 4 * p

def revenue (p : ℝ) : ℝ :=
  p * cups_sold p

def cost : ℝ :=
  200

def profit (p : ℝ) : ℝ :=
  revenue p - cost

theorem maximize_profit (p : ℝ) (h : p ≤ 30) : p = 19 → profit p = 1206.25 :=
by
  sorry

end maximize_profit_l1495_149551


namespace max_ahead_distance_l1495_149595

noncomputable def distance_run_by_alex (initial_distance ahead1 ahead_max_runs final_ahead : ℝ) : ℝ :=
  initial_distance + ahead1 + ahead_max_runs + final_ahead

theorem max_ahead_distance :
  let initial_distance := 200
  let ahead1 := 300
  let final_ahead := 440
  let total_road := 5000
  let distance_remaining := 3890
  let distance_run_alex := total_road - distance_remaining
  ∃ X : ℝ, distance_run_by_alex initial_distance ahead1 X final_ahead = distance_run_alex ∧ X = 170 :=
by
  intro initial_distance ahead1 final_ahead total_road distance_remaining distance_run_alex
  use 170
  simp [initial_distance, ahead1, final_ahead, total_road, distance_remaining, distance_run_alex, distance_run_by_alex]
  sorry

end max_ahead_distance_l1495_149595


namespace sequence_correctness_l1495_149573

def sequence_a (n : ℕ) : ℤ :=
  if n = 1 then -2
  else -(2^(n - 1))

def partial_sum_S (n : ℕ) : ℤ := -2^n

theorem sequence_correctness (n : ℕ) (h : n ≥ 1) :
  (sequence_a 1 = -2) ∧ (∀ n ≥ 2, sequence_a (n + 1) = partial_sum_S n) ∧
  (sequence_a n = -(2^(n - 1))) ∧ (partial_sum_S n = -2^n) :=
by
  sorry

end sequence_correctness_l1495_149573


namespace original_cost_l1495_149519

theorem original_cost (SP : ℝ) (C : ℝ) (h1 : SP = 540) (h2 : SP = C + 0.35 * C) : C = 400 :=
by {
  sorry
}

end original_cost_l1495_149519


namespace library_visitors_on_sunday_l1495_149538

def avg_visitors_sundays (S : ℕ) : Prop :=
  let total_days := 30
  let avg_other_days := 240
  let avg_total := 285
  let sundays := 5
  let other_days := total_days - sundays
  (S * sundays) + (avg_other_days * other_days) = avg_total * total_days

theorem library_visitors_on_sunday (S : ℕ) (h : avg_visitors_sundays S) : S = 510 :=
by
  sorry

end library_visitors_on_sunday_l1495_149538


namespace acute_triangle_sec_csc_inequality_l1495_149510

theorem acute_triangle_sec_csc_inequality (A B C : ℝ) (h : A + B + C = π) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hA90 : A < π / 2) (hB90 : B < π / 2) (hC90 : C < π / 2) :
  (1 / Real.cos A) + (1 / Real.cos B) + (1 / Real.cos C) ≥
  (1 / Real.sin (A / 2)) + (1 / Real.sin (B / 2)) + (1 / Real.sin (C / 2)) :=
by sorry

end acute_triangle_sec_csc_inequality_l1495_149510


namespace container_capacity_l1495_149590

theorem container_capacity (C : ℝ) (h₁ : C > 15) (h₂ : 0 < (81 : ℝ)) (h₃ : (337 : ℝ) > 0) :
  ((C - 15) / C) ^ 4 = 81 / 337 :=
sorry

end container_capacity_l1495_149590


namespace reciprocal_opposites_l1495_149575

theorem reciprocal_opposites (a b : ℝ) (h1 : 1 / a = -8) (h2 : 1 / -b = 8) : a = b :=
sorry

end reciprocal_opposites_l1495_149575


namespace ordered_pair_unique_l1495_149565

theorem ordered_pair_unique (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h1 : x^y + 1 = y^x) (h2 : 2 * x^y = y^x + 13) : (x, y) = (1, 14) :=
by
  sorry

end ordered_pair_unique_l1495_149565


namespace quadratic_equation_with_product_of_roots_20_l1495_149572

theorem quadratic_equation_with_product_of_roots_20
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : c / a = 20) :
  ∃ b : ℝ, ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  use 1
  use 20
  sorry

end quadratic_equation_with_product_of_roots_20_l1495_149572


namespace slope_tangent_at_point_l1495_149599

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 - 2

theorem slope_tangent_at_point : (deriv f 1) = 1 := 
by
  sorry

end slope_tangent_at_point_l1495_149599


namespace felicity_used_5_gallons_less_l1495_149520

def adhesion_gas_problem : Prop :=
  ∃ A x : ℕ, (A + 23 = 30) ∧ (4 * A - x = 23) ∧ (x = 5)
  
theorem felicity_used_5_gallons_less :
  adhesion_gas_problem :=
by
  sorry

end felicity_used_5_gallons_less_l1495_149520


namespace functional_equation_f2023_l1495_149550

theorem functional_equation_f2023 (f : ℝ → ℝ) (h_add : ∀ x y : ℝ, f (x + y) = f x + f y) (h_one : f 1 = 1) :
  f 2023 = 2023 := sorry

end functional_equation_f2023_l1495_149550


namespace two_digit_factors_count_l1495_149513

-- Definition of the expression 10^8 - 1
def expr : ℕ := 10^8 - 1

-- Factorization of 10^8 - 1
def factored_expr : List ℕ := [73, 137, 101, 11, 3^2]

-- Define the condition for being a two-digit factor
def is_two_digit (n : ℕ) : Bool := n > 9 ∧ n < 100

-- Count the number of positive two-digit factors in the factorization of 10^8 - 1
def num_two_digit_factors : ℕ := List.length (factored_expr.filter is_two_digit)

-- The theorem stating our proof problem
theorem two_digit_factors_count : num_two_digit_factors = 2 := by
  sorry

end two_digit_factors_count_l1495_149513
