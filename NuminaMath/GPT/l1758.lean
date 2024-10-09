import Mathlib

namespace bose_einstein_distribution_diagrams_fermi_dirac_distribution_diagrams_l1758_175833

/-- 
To prove the number of distinct distribution diagrams for particles following the 
Bose-Einstein distribution, satisfying the given conditions. 
-/
theorem bose_einstein_distribution_diagrams (n_particles : ℕ) (total_energy : ℕ) 
  (energy_unit : ℕ) : n_particles = 4 → total_energy = 4 * energy_unit → 
  ∃ (distinct_diagrams : ℕ), distinct_diagrams = 72 := 
  by
  sorry

/-- 
To prove the number of distinct distribution diagrams for particles following the 
Fermi-Dirac distribution, satisfying the given conditions. 
-/
theorem fermi_dirac_distribution_diagrams (n_particles : ℕ) (total_energy : ℕ) 
  (energy_unit : ℕ) : n_particles = 4 → total_energy = 4 * energy_unit → 
  ∃ (distinct_diagrams : ℕ), distinct_diagrams = 246 := 
  by
  sorry

end bose_einstein_distribution_diagrams_fermi_dirac_distribution_diagrams_l1758_175833


namespace min_buses_needed_l1758_175809

theorem min_buses_needed (n : ℕ) (h1 : 45 * n ≥ 500) (h2 : n ≥ 2) : n = 12 :=
sorry

end min_buses_needed_l1758_175809


namespace g_inv_zero_solution_l1758_175897

noncomputable def g (a b x : ℝ) : ℝ := 1 / (2 * a * x + b)

theorem g_inv_zero_solution (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hab : a ≠ b) :
  g a b (g a b 0) = 0 ↔ g a b 0 = 1 / b :=
by
  sorry

end g_inv_zero_solution_l1758_175897


namespace factorable_b_even_l1758_175826

-- Defining the conditions
def is_factorable (b : ℤ) : Prop :=
  ∃ (m n p q : ℤ), 
    m * p = 15 ∧ n * q = 15 ∧ b = m * q + n * p

-- The theorem to be stated
theorem factorable_b_even (b : ℤ) : is_factorable b ↔ ∃ k : ℤ, b = 2 * k :=
sorry

end factorable_b_even_l1758_175826


namespace cos_double_angle_l1758_175819

theorem cos_double_angle (α : ℝ) (h : Real.cos α = 4 / 5) : Real.cos (2 * α) = 7 / 25 := 
by
  sorry

end cos_double_angle_l1758_175819


namespace impossible_odd_n_m_even_sum_l1758_175853

theorem impossible_odd_n_m_even_sum (n m : ℤ) (h : (n^2 + m^2 + n*m) % 2 = 0) : ¬ (n % 2 = 1 ∧ m % 2 = 1) :=
by sorry

end impossible_odd_n_m_even_sum_l1758_175853


namespace usual_time_to_school_l1758_175859

-- Define the conditions
variables (R T : ℝ) (h1 : 0 < T) (h2 : 0 < R)
noncomputable def boy_reaches_school_early : Prop :=
  (7/6 * R) * (T - 5) = R * T

-- The theorem stating the usual time to reach the school
theorem usual_time_to_school (h : boy_reaches_school_early R T) : T = 35 :=
by
  sorry

end usual_time_to_school_l1758_175859


namespace simplify_expression_l1758_175843

variable (x : ℝ)

theorem simplify_expression :
  (3 * x^6 + 2 * x^5 + x^4 + x - 5) - (x^6 + 3 * x^5 + 2 * x^3 + 6) = 2 * x^6 - x^5 + x^4 - 2 * x^3 + x + 1 := by
  sorry

end simplify_expression_l1758_175843


namespace has_minimum_value_iff_l1758_175895

noncomputable def f (a x : ℝ) : ℝ :=
if x < a then -a * x + 4 else (x - 2) ^ 2

theorem has_minimum_value_iff (a : ℝ) : (∃ m, ∀ x, f a x ≥ m) ↔ 0 ≤ a ∧ a ≤ 2 :=
sorry

end has_minimum_value_iff_l1758_175895


namespace second_discount_percentage_l1758_175814

-- Definitions for the given conditions
def original_price : ℝ := 33.78
def first_discount_rate : ℝ := 0.25
def final_price : ℝ := 19.0

-- Intermediate calculations based on the conditions
def first_discount : ℝ := first_discount_rate * original_price
def price_after_first_discount : ℝ := original_price - first_discount
def second_discount_amount : ℝ := price_after_first_discount - final_price

-- Lean theorem statement
theorem second_discount_percentage : (second_discount_amount / price_after_first_discount) * 100 = 25 := by
  sorry

end second_discount_percentage_l1758_175814


namespace convex_quadrilateral_inequality_l1758_175847

variable (A B C D : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]

theorem convex_quadrilateral_inequality
    (AB CD BC AD AC BD : ℝ)
    (h : AB * CD + BC * AD >= AC * BD)
    (convex_quadrilateral : Prop) :
  AB * CD + BC * AD >= AC * BD :=
by
  sorry

end convex_quadrilateral_inequality_l1758_175847


namespace last_locker_opened_2046_l1758_175815

def last_locker_opened (n : ℕ) : ℕ :=
  n - (n % 3)

theorem last_locker_opened_2046 : last_locker_opened 2048 = 2046 := by
  sorry

end last_locker_opened_2046_l1758_175815


namespace fraction_value_l1758_175818

theorem fraction_value (x y z : ℝ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4) : (x + y + z) / (2 * z) = 9 / 8 :=
by
  sorry

end fraction_value_l1758_175818


namespace max_product_h_k_l1758_175854

theorem max_product_h_k {h k : ℝ → ℝ} (h_bound : ∀ x, -3 ≤ h x ∧ h x ≤ 5) (k_bound : ∀ x, -1 ≤ k x ∧ k x ≤ 4) :
  ∃ x y, h x * k y = 20 :=
by
  sorry

end max_product_h_k_l1758_175854


namespace find_principal_l1758_175820

theorem find_principal (A : ℝ) (R : ℝ) (T : ℝ) (P : ℝ) (h1 : A = 1456) (h2 : R = 0.05) (h3 : T = 2.4) :
  A = P + P * R * T → P = 1300 :=
by {
  sorry
}

end find_principal_l1758_175820


namespace third_generation_tail_length_is_25_l1758_175801

def first_generation_tail_length : ℝ := 16
def growth_rate : ℝ := 0.25

def second_generation_tail_length : ℝ := first_generation_tail_length * (1 + growth_rate)
def third_generation_tail_length : ℝ := second_generation_tail_length * (1 + growth_rate)

theorem third_generation_tail_length_is_25 :
  third_generation_tail_length = 25 := by
  sorry

end third_generation_tail_length_is_25_l1758_175801


namespace fuel_remaining_l1758_175890

-- Definitions given in the conditions of the original problem
def initial_fuel : ℕ := 48
def fuel_consumption_rate : ℕ := 8

-- Lean 4 statement of the mathematical proof problem
theorem fuel_remaining (x : ℕ) : 
  ∃ y : ℕ, y = initial_fuel - fuel_consumption_rate * x :=
sorry

end fuel_remaining_l1758_175890


namespace susie_pizza_sales_l1758_175881

theorem susie_pizza_sales :
  ∃ x : ℕ, 
    (24 * 3 + 15 * x = 117) ∧ 
    x = 3 := 
by
  sorry

end susie_pizza_sales_l1758_175881


namespace race_lead_distance_l1758_175852

theorem race_lead_distance :
  ∀ (d12 d13 : ℝ) (s1 s2 s3 t : ℝ), 
  d12 = 2 →
  d13 = 4 →
  t > 0 →
  s1 = (d12 / t + s2) →
  s1 = (d13 / t + s3) →
  s2 * t - s3 * t = 2.5 :=
by
  sorry

end race_lead_distance_l1758_175852


namespace carly_trimmed_nails_correct_l1758_175857

-- Definitions based on the conditions
def total_dogs : Nat := 11
def three_legged_dogs : Nat := 3
def paws_per_four_legged_dog : Nat := 4
def paws_per_three_legged_dog : Nat := 3
def nails_per_paw : Nat := 4

-- Mathematically equivalent proof problem in Lean 4 statement
theorem carly_trimmed_nails_correct :
  let four_legged_dogs := total_dogs - three_legged_dogs
  let nails_per_four_legged_dog := paws_per_four_legged_dog * nails_per_paw
  let nails_per_three_legged_dog := paws_per_three_legged_dog * nails_per_paw
  let total_nails_trimmed :=
    (four_legged_dogs * nails_per_four_legged_dog) +
    (three_legged_dogs * nails_per_three_legged_dog)
  total_nails_trimmed = 164 := by
  sorry

end carly_trimmed_nails_correct_l1758_175857


namespace initial_books_l1758_175841

-- Define the variables and conditions
def B : ℕ := 75
def loaned_books : ℕ := 60
def returned_books : ℕ := (70 * loaned_books) / 100
def not_returned_books : ℕ := loaned_books - returned_books
def end_of_month_books : ℕ := 57

-- State the theorem
theorem initial_books (h1 : returned_books = 42)
                      (h2 : end_of_month_books = 57)
                      (h3 : loaned_books = 60) :
  B = end_of_month_books + not_returned_books :=
by sorry

end initial_books_l1758_175841


namespace speed_upstream_calculation_l1758_175856

def speed_boat_still_water : ℝ := 60
def speed_current : ℝ := 17

theorem speed_upstream_calculation : speed_boat_still_water - speed_current = 43 := by
  sorry

end speed_upstream_calculation_l1758_175856


namespace increment_M0_to_M1_increment_M0_to_M2_increment_M0_to_M3_l1758_175871

-- Define the function z = x * y
def z (x y : ℝ) : ℝ := x * y

-- Initial point M0
def M0 : ℝ × ℝ := (1, 2)

-- Points to which we move
def M1 : ℝ × ℝ := (1.1, 2)
def M2 : ℝ × ℝ := (1, 1.9)
def M3 : ℝ × ℝ := (1.1, 2.2)

-- Proofs for the increments
theorem increment_M0_to_M1 : z M1.1 M1.2 - z M0.1 M0.2 = 0.2 :=
by sorry

theorem increment_M0_to_M2 : z M2.1 M2.2 - z M0.1 M0.2 = -0.1 :=
by sorry

theorem increment_M0_to_M3 : z M3.1 M3.2 - z M0.1 M0.2 = 0.42 :=
by sorry

end increment_M0_to_M1_increment_M0_to_M2_increment_M0_to_M3_l1758_175871


namespace cross_shape_rectangle_count_l1758_175873

def original_side_length := 30
def smallest_square_side_length := 1
def cut_corner_length := 10
def N : ℕ := sorry  -- total number of rectangles in the resultant graph paper
def result : ℕ := 14413

theorem cross_shape_rectangle_count :
  (1/10 : ℚ) * N = result := 
sorry

end cross_shape_rectangle_count_l1758_175873


namespace sum_of_two_digit_integers_l1758_175879

theorem sum_of_two_digit_integers :
  let a := 10
  let l := 99
  let d := 1
  let n := (l - a) / d + 1
  let S := n * (a + l) / 2
  S = 4905 :=
by
  sorry

end sum_of_two_digit_integers_l1758_175879


namespace books_before_grant_l1758_175889

-- Define the conditions 
def books_purchased_with_grant : ℕ := 2647
def total_books_now : ℕ := 8582

-- Prove the number of books before the grant
theorem books_before_grant : 
  (total_books_now - books_purchased_with_grant = 5935) := 
by
  sorry

end books_before_grant_l1758_175889


namespace lemon_bag_mass_l1758_175824

variable (m : ℝ)  -- mass of one bag of lemons in kg

-- Conditions
def max_load := 900  -- maximum load in kg
def num_bags := 100  -- number of bags
def extra_load := 100  -- additional load in kg

-- Proof statement (target)
theorem lemon_bag_mass : num_bags * m + extra_load = max_load → m = 8 :=
by
  sorry

end lemon_bag_mass_l1758_175824


namespace average_speed_l1758_175804

/--
On the first day of her vacation, Louisa traveled 100 miles.
On the second day, traveling at the same average speed, she traveled 175 miles.
If the 100-mile trip took 3 hours less than the 175-mile trip,
prove that her average speed (in miles per hour) was 25.
-/
theorem average_speed (v : ℝ) (h1 : 100 / v + 3 = 175 / v) : v = 25 :=
by 
  sorry

end average_speed_l1758_175804


namespace solve_for_x_l1758_175813

variable (a b x : ℝ)
variable (a_pos : a > 0) (b_pos : b > 0) (x_pos : x > 0)

theorem solve_for_x : (3 * a) ^ (3 * b) = (a ^ b) * (x ^ b) → x = 27 * a ^ 2 :=
by
  intro h_eq
  sorry

end solve_for_x_l1758_175813


namespace second_divisor_l1758_175811

theorem second_divisor (N : ℤ) (k : ℤ) (D : ℤ) (m : ℤ) 
  (h1 : N = 39 * k + 20) 
  (h2 : N = D * m + 7) : 
  D = 13 := sorry

end second_divisor_l1758_175811


namespace ratio_small_to_large_is_one_to_one_l1758_175899

theorem ratio_small_to_large_is_one_to_one
  (total_beads : ℕ)
  (large_beads_per_bracelet : ℕ)
  (bracelets_count : ℕ)
  (small_beads : ℕ)
  (large_beads : ℕ)
  (small_beads_per_bracelet : ℕ) :
  total_beads = 528 →
  large_beads_per_bracelet = 12 →
  bracelets_count = 11 →
  large_beads = total_beads / 2 →
  large_beads >= bracelets_count * large_beads_per_bracelet →
  small_beads = total_beads / 2 →
  small_beads_per_bracelet = small_beads / bracelets_count →
  small_beads_per_bracelet / large_beads_per_bracelet = 1 :=
by sorry

end ratio_small_to_large_is_one_to_one_l1758_175899


namespace abc_mod_n_l1758_175802

theorem abc_mod_n (n : ℕ) (a b c : ℤ) (hn : 0 < n)
  (h1 : a * b ≡ 1 [ZMOD n])
  (h2 : c ≡ b [ZMOD n]) : (a * b * c) ≡ 1 [ZMOD n] := sorry

end abc_mod_n_l1758_175802


namespace probability_of_pairing_with_friends_l1758_175832

theorem probability_of_pairing_with_friends (n : ℕ) (f : ℕ) (h1 : n = 32) (h2 : f = 2):
  (f / (n - 1) : ℚ) = 2 / 31 :=
by
  rw [h1, h2]
  norm_num

end probability_of_pairing_with_friends_l1758_175832


namespace abs_neg_four_minus_six_l1758_175884

theorem abs_neg_four_minus_six : abs (-4 - 6) = 10 := 
by
  sorry

end abs_neg_four_minus_six_l1758_175884


namespace baseball_team_grouping_l1758_175845

theorem baseball_team_grouping (new_players returning_players : ℕ) (group_size : ℕ) 
  (h_new : new_players = 4) (h_returning : returning_players = 6) (h_group : group_size = 5) : 
  (new_players + returning_players) / group_size = 2 := 
  by 
  sorry

end baseball_team_grouping_l1758_175845


namespace domain_of_f_l1758_175829

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (-x) + Real.sqrt (x * (x + 1))

theorem domain_of_f :
  {x : ℝ | -x ≥ 0 ∧ x * (x + 1) ≥ 0} = {x : ℝ | x ≤ -1 ∨ x = 0} :=
by
  sorry

end domain_of_f_l1758_175829


namespace xiaoli_estimate_larger_l1758_175882

variable (x y z w : ℝ)
variable (hxy : x > y) (hy0 : y > 0) (hz1 : z > 1) (hw0 : w > 0)

theorem xiaoli_estimate_larger : (x + w) - (y - w) * z > x - y * z :=
by sorry

end xiaoli_estimate_larger_l1758_175882


namespace estimate_time_pm_l1758_175834

-- Definitions from the conditions
def school_start_time : ℕ := 12
def classes : List String := ["Maths", "History", "Geography", "Science", "Music"]
def class_time : ℕ := 45  -- in minutes
def break_time : ℕ := 15  -- in minutes
def classes_up_to_science : List String := ["Maths", "History", "Geography", "Science"]
def total_classes_time : ℕ := classes_up_to_science.length * (class_time + break_time)

-- Lean statement to prove that given the conditions, the time is 4 pm
theorem estimate_time_pm :
  school_start_time + (total_classes_time / 60) = 16 :=
by
  sorry

end estimate_time_pm_l1758_175834


namespace pyramid_volume_formula_l1758_175868

noncomputable def pyramid_volume (a α β : ℝ) : ℝ :=
  (1/6) * a^3 * (Real.sin (α/2)) * (Real.tan β)

theorem pyramid_volume_formula (a α β : ℝ) :
  (base_is_isosceles_triangle : Prop) → (lateral_edges_inclined : Prop) → 
  pyramid_volume a α β = (1/6) * a^3 * (Real.sin (α/2)) * (Real.tan β) :=
by
  intros c1 c2
  exact sorry

end pyramid_volume_formula_l1758_175868


namespace R_and_D_expenditure_l1758_175805

theorem R_and_D_expenditure (R_D_t : ℝ) (Delta_APL_t_plus_2 : ℝ) (ratio : ℝ) :
  R_D_t = 3013.94 → Delta_APL_t_plus_2 = 3.29 → ratio = 916 →
  R_D_t / Delta_APL_t_plus_2 = ratio :=
by
  intros hR hD hRto
  rw [hR, hD, hRto]
  sorry

end R_and_D_expenditure_l1758_175805


namespace find_m_l1758_175878

theorem find_m (x y m : ℝ) (h₁ : x - 2 * y = m) (h₂ : x = 2) (h₃ : y = 1) : m = 0 :=
by 
  -- Proof omitted
  sorry

end find_m_l1758_175878


namespace remainder_when_divided_by_7_l1758_175864

theorem remainder_when_divided_by_7 
  {k : ℕ} 
  (h1 : k % 5 = 2) 
  (h2 : k % 6 = 5) 
  (h3 : k < 41) : 
  k % 7 = 3 := 
sorry

end remainder_when_divided_by_7_l1758_175864


namespace difference_largest_smallest_l1758_175825

def num1 : ℕ := 10
def num2 : ℕ := 11
def num3 : ℕ := 12

theorem difference_largest_smallest :
  (max num1 (max num2 num3)) - (min num1 (min num2 num3)) = 2 :=
by
  -- Proof can be filled here
  sorry

end difference_largest_smallest_l1758_175825


namespace radian_measure_sector_l1758_175866

theorem radian_measure_sector (r l : ℝ) (h1 : 2 * r + l = 12) (h2 : (1 / 2) * l * r = 8) :
  l / r = 1 ∨ l / r = 4 := by
  sorry

end radian_measure_sector_l1758_175866


namespace fourth_person_height_l1758_175896

-- Definitions based on conditions
def h1 : ℕ := 73  -- height of first person
def h2 : ℕ := h1 + 2  -- height of second person
def h3 : ℕ := h2 + 2  -- height of third person
def h4 : ℕ := h3 + 6  -- height of fourth person

theorem fourth_person_height : h4 = 83 :=
by
  -- calculation to check the average height and arriving at h1
  -- (all detailed calculations are skipped using "sorry")
  sorry

end fourth_person_height_l1758_175896


namespace conversion_base10_to_base7_l1758_175800

-- Define the base-10 number
def num_base10 : ℕ := 1023

-- Define the conversion base
def base : ℕ := 7

-- Define the expected base-7 representation as a function of the base
def expected_base7 (b : ℕ) : ℕ := 2 * b^3 + 6 * b^2 + 6 * b^1 + 1 * b^0

-- Statement to prove
theorem conversion_base10_to_base7 : expected_base7 base = num_base10 :=
by 
  -- Sorry is a placeholder for the proof
  sorry

end conversion_base10_to_base7_l1758_175800


namespace simplify_expression_l1758_175831

variable (b : ℝ)

theorem simplify_expression : 3 * b * (3 * b^3 + 2 * b^2) - 2 * b^2 + 5 = 9 * b^4 + 6 * b^3 - 2 * b^2 + 5 := sorry

end simplify_expression_l1758_175831


namespace shanille_probability_l1758_175883

-- Defining the probability function according to the problem's conditions.
def hit_probability (n k : ℕ) : ℚ :=
  if n = 100 ∧ k = 50 then 1 / 99 else 0

-- Prove that the probability Shanille hits exactly 50 of her first 100 shots is 1/99.
theorem shanille_probability :
  hit_probability 100 50 = 1 / 99 :=
by
  -- proof omitted
  sorry

end shanille_probability_l1758_175883


namespace ziggy_song_requests_l1758_175869

theorem ziggy_song_requests :
  ∃ T : ℕ, 
    (T = (1/2) * T + (1/6) * T + 5 + 2 + 1 + 2) →
    T = 30 :=
by 
  sorry

end ziggy_song_requests_l1758_175869


namespace find_a_n_geo_b_find_S_2n_l1758_175887
noncomputable def S : ℕ → ℚ
| n => (n^2 + n + 1) / 2

def a (n : ℕ) : ℚ :=
  if n = 1 then 3/2
  else n

theorem find_a_n (n : ℕ) : a n = if n = 1 then 3/2 else n :=
by
  sorry

def b (n : ℕ) : ℚ :=
  a (2 * n - 1) + a (2 * n)

theorem geo_b (n : ℕ) : b (n + 1) = 3 * b n :=
by
  sorry

theorem find_S_2n (n : ℕ) : S (2 * n) = 3/2 * (3^n - 1) :=
by
  sorry

end find_a_n_geo_b_find_S_2n_l1758_175887


namespace hours_per_trainer_l1758_175816

-- Define the conditions from part (a)
def number_of_dolphins : ℕ := 4
def hours_per_dolphin : ℕ := 3
def number_of_trainers : ℕ := 2

-- Define the theorem we want to prove using the answer from part (b)
theorem hours_per_trainer : (number_of_dolphins * hours_per_dolphin) / number_of_trainers = 6 :=
by
  -- Proof goes here
  sorry

end hours_per_trainer_l1758_175816


namespace modulus_of_z_l1758_175865

section complex_modulus
open Complex

theorem modulus_of_z (z : ℂ) (h : z * (2 + I) = 10 - 5 * I) : Complex.abs z = 5 :=
by
  sorry
end complex_modulus

end modulus_of_z_l1758_175865


namespace ice_cream_sandwiches_l1758_175806

theorem ice_cream_sandwiches (n : ℕ) (x : ℕ) (h1 : n = 11) (h2 : x = 13) : (n * x = 143) := 
by
  sorry

end ice_cream_sandwiches_l1758_175806


namespace simplify_expression_l1758_175892

noncomputable def a : ℝ := Real.sqrt 3 - 1

theorem simplify_expression : 
  ( (a - 1) / (a^2 - 2 * a + 1) / ( (a^2 + a) / (a^2 - 1) + 1 / (a - 1) ) = Real.sqrt 3 / 3 ) :=
by
  sorry

end simplify_expression_l1758_175892


namespace add_and_subtract_l1758_175876

theorem add_and_subtract (a b c : ℝ) (h1 : a = 0.45) (h2 : b = 52.7) (h3 : c = 0.25) : 
  (a + b) - c = 52.9 :=
by 
  sorry

end add_and_subtract_l1758_175876


namespace tangent_line_at_x_is_2_l1758_175828

noncomputable def curve (x : ℝ) : ℝ := (1/4) * x^2 - 3 * Real.log x

theorem tangent_line_at_x_is_2 :
  ∃ x₀ : ℝ, (x₀ > 0) ∧ ((1/2) * x₀ - (3 / x₀) = -1/2) ∧ x₀ = 2 :=
by
  sorry

end tangent_line_at_x_is_2_l1758_175828


namespace n_not_both_perfect_squares_l1758_175862

open Int

theorem n_not_both_perfect_squares (n x y : ℤ) (h1 : n > 0) :
  ¬ ((n + 1 = x^2) ∧ (4 * n + 1 = y^2)) :=
by {
  -- Problem restated in Lean, proof not required
  sorry
}

end n_not_both_perfect_squares_l1758_175862


namespace math_city_police_officers_needed_l1758_175867

def number_of_streets : Nat := 10
def initial_intersections : Nat := Nat.choose number_of_streets 2
def non_intersections : Nat := 2
def effective_intersections : Nat := initial_intersections - non_intersections

theorem math_city_police_officers_needed :
  effective_intersections = 43 := by
  sorry

end math_city_police_officers_needed_l1758_175867


namespace necessary_not_sufficient_l1758_175875

theorem necessary_not_sufficient (m : ℝ) (x : ℝ) (h₁ : m > 0) (h₂ : 0 < x ∧ x < m) (h₃ : x / (x - 1) < 0) 
: m = 1 / 2 := 
sorry

end necessary_not_sufficient_l1758_175875


namespace find_value_of_m_l1758_175874

variables (x y m : ℝ)

theorem find_value_of_m (h1 : y ≥ x) (h2 : x + 3 * y ≤ 4) (h3 : x ≥ m) (hz_max : ∀ z, (z = x - 3 * y) → z ≤ 8) :
  m = -4 :=
sorry

end find_value_of_m_l1758_175874


namespace greatest_three_digit_multiple_of_17_l1758_175846

theorem greatest_three_digit_multiple_of_17 :
  ∃ n : ℕ, (100 ≤ n ∧ n < 1000) ∧ (17 ∣ n) ∧ (∀ m : ℕ, (100 ≤ m ∧ m < 1000 ∧ 17 ∣ m) → m ≤ n) ∧ n = 986 :=
by
  sorry

end greatest_three_digit_multiple_of_17_l1758_175846


namespace tan_17pi_over_4_l1758_175837

theorem tan_17pi_over_4 : Real.tan (17 * Real.pi / 4) = 1 := by
  sorry

end tan_17pi_over_4_l1758_175837


namespace first_rectangle_dimensions_second_rectangle_dimensions_l1758_175863

theorem first_rectangle_dimensions (x y : ℕ) (h : x * y = 2 * (x + y) + 1) : (x = 7 ∧ y = 3) ∨ (x = 3 ∧ y = 7) :=
sorry

theorem second_rectangle_dimensions (a b : ℕ) (h : a * b = 2 * (a + b) - 1) : (a = 5 ∧ b = 3) ∨ (a = 3 ∧ b = 5) :=
sorry

end first_rectangle_dimensions_second_rectangle_dimensions_l1758_175863


namespace geometric_progression_fourth_term_l1758_175835

theorem geometric_progression_fourth_term :
  ∀ (a₁ a₂ a₃ a₄ : ℝ), a₁ = 2^(1/2) ∧ a₂ = 2^(1/4) ∧ a₃ = 2^(1/6) ∧ (a₂ / a₁ = r) ∧ (a₃ = a₂ * r⁻¹) ∧ (a₄ = a₃ * r) → a₄ = 2^(1/8) := by
intro a₁ a₂ a₃ a₄
intro h
sorry

end geometric_progression_fourth_term_l1758_175835


namespace eval_expr_l1758_175855

theorem eval_expr (x y : ℕ) (h1 : x = 2) (h2 : y = 1) : 2 * x - 3 * y = 1 := by
  sorry

end eval_expr_l1758_175855


namespace intersection_x_value_l1758_175877

theorem intersection_x_value : ∃ x y : ℝ, y = 3 * x + 7 ∧ 3 * x - 2 * y = -4 ∧ x = -10 / 3 :=
by
  sorry

end intersection_x_value_l1758_175877


namespace Connor_spends_36_dollars_l1758_175893

def tickets_cost := 10.00
def combo_meal_cost := 11.00
def candy_cost := 2.50
def total_cost := tickets_cost * 2 + combo_meal_cost + candy_cost * 2

theorem Connor_spends_36_dollars : total_cost = 36.00 := 
by 
  sorry

end Connor_spends_36_dollars_l1758_175893


namespace average_brown_mms_l1758_175851

def brown_smiley_counts : List Nat := [9, 12, 8, 8, 3]
def brown_star_counts : List Nat := [7, 14, 11, 6, 10]

def average (lst : List Nat) : Float :=
  (lst.foldl (· + ·) 0).toFloat / lst.length.toFloat
  
theorem average_brown_mms :
  average brown_smiley_counts = 8 ∧
  average brown_star_counts = 9.6 :=
by 
  sorry

end average_brown_mms_l1758_175851


namespace sky_falls_distance_l1758_175849

def distance_from_city (x : ℕ) (y : ℕ) : Prop := 50 * x = y

theorem sky_falls_distance :
    ∃ D_s : ℕ, distance_from_city D_s 400 ∧ D_s = 8 :=
by
  sorry

end sky_falls_distance_l1758_175849


namespace find_a20_l1758_175830

variables {a : ℕ → ℤ} {S : ℕ → ℤ}
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℤ) (S : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, S n = n * a 1 + (n * (n - 1) / 2) * (a 2 - a 1)

theorem find_a20 (h_arith : is_arithmetic_sequence a)
  (h_sum : sum_of_first_n_terms a S)
  (h_a1 : a 1 = -1)
  (h_S10 : S 10 = 35) :
  a 20 = 18 :=
sorry

end find_a20_l1758_175830


namespace range_of_a_l1758_175894

variable (a : ℝ)
def f (x : ℝ) := x^2 + 2 * (a - 1) * x + 2
def f_deriv (x : ℝ) := 2 * x + 2 * (a - 1)

theorem range_of_a (h : ∀ x ≥ -4, f_deriv a x ≥ 0) : a ≥ 5 :=
sorry

end range_of_a_l1758_175894


namespace master_zhang_must_sell_100_apples_l1758_175836

-- Define the given conditions
def buying_price_per_apple : ℚ := 1 / 4 -- 1 yuan for 4 apples
def selling_price_per_apple : ℚ := 2 / 5 -- 2 yuan for 5 apples
def profit_per_apple : ℚ := selling_price_per_apple - buying_price_per_apple

-- Define the target profit
def target_profit : ℚ := 15

-- Define the number of apples to sell
def apples_to_sell : ℚ := target_profit / profit_per_apple

-- The theorem statement: Master Zhang must sell 100 apples to achieve the target profit of 15 yuan
theorem master_zhang_must_sell_100_apples :
  apples_to_sell = 100 :=
sorry

end master_zhang_must_sell_100_apples_l1758_175836


namespace solve_linear_equation_l1758_175810

theorem solve_linear_equation :
  ∀ x : ℝ, (2 * x + 1) / 3 - (5 * x - 1) / 6 = 1 → x = -3 :=
by
  sorry

end solve_linear_equation_l1758_175810


namespace triangle_shape_l1758_175880

open Real

noncomputable def triangle (a b c A B C S : ℝ) :=
  ∃ (a b c A B C S : ℝ),
    a = 2 * sqrt 3 ∧
    A = π / 3 ∧
    S = 2 * sqrt 3 ∧
    (S = (1 / 2) * b * c * sin A) ∧
    (a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * cos A) ∧
    (b = 2 ∧ c = 4 ∨ b = 4 ∧ c = 2)

theorem triangle_shape (A B C : ℝ) (h : sin (C - B) = sin (2 * B) - sin A):
    (B = π / 2 ∨ C = B) :=
sorry

end triangle_shape_l1758_175880


namespace probability_difference_l1758_175840

noncomputable def Ps (red black : ℕ) : ℚ :=
  let total := red + black
  (red * (red - 1) + black * (black - 1)) / (total * (total - 1))

noncomputable def Pd (red black : ℕ) : ℚ :=
  let total := red + black
  (red * black * 2) / (total * (total - 1))

noncomputable def abs_diff (Ps Pd : ℚ) : ℚ :=
  |Ps - Pd|

theorem probability_difference :
  let red := 1200
  let black := 800
  let total := red + black
  abs_diff (Ps red black) (Pd red black) = 789 / 19990 := by
  sorry

end probability_difference_l1758_175840


namespace parabola_x_intercept_unique_l1758_175827

theorem parabola_x_intercept_unique : ∃! (x : ℝ), ∀ (y : ℝ), x = -y^2 + 2*y + 3 → x = 3 :=
by
  sorry

end parabola_x_intercept_unique_l1758_175827


namespace odd_power_sum_divisible_l1758_175891

theorem odd_power_sum_divisible (x y : ℤ) (n : ℕ) (h_odd : ∃ k : ℕ, n = 2 * k + 1) :
  (x ^ n + y ^ n) % (x + y) = 0 := 
sorry

end odd_power_sum_divisible_l1758_175891


namespace cos_5alpha_eq_sin_5alpha_eq_l1758_175842

noncomputable def cos_five_alpha (α : ℝ) : ℝ := 16 * (Real.cos α) ^ 5 - 20 * (Real.cos α) ^ 3 + 5 * (Real.cos α)
noncomputable def sin_five_alpha (α : ℝ) : ℝ := 16 * (Real.sin α) ^ 5 - 20 * (Real.sin α) ^ 3 + 5 * (Real.sin α)

theorem cos_5alpha_eq (α : ℝ) : Real.cos (5 * α) = cos_five_alpha α :=
by sorry

theorem sin_5alpha_eq (α : ℝ) : Real.sin (5 * α) = sin_five_alpha α :=
by sorry

end cos_5alpha_eq_sin_5alpha_eq_l1758_175842


namespace percent_both_correct_l1758_175850

-- Definitions of the given percentages
def A : ℝ := 75
def B : ℝ := 25
def N : ℝ := 20

-- The proof problem statement
theorem percent_both_correct (A B N : ℝ) (hA : A = 75) (hB : B = 25) (hN : N = 20) : A + B - N - 100 = 20 :=
by
  sorry

end percent_both_correct_l1758_175850


namespace find_A_for_diamond_eq_85_l1758_175812

def diamond (A B : ℝ) : ℝ := 4 * A + B^2 + 7

theorem find_A_for_diamond_eq_85 :
  ∃ (A : ℝ), diamond A 3 = 85 ∧ A = 17.25 :=
by
  sorry

end find_A_for_diamond_eq_85_l1758_175812


namespace sufficiency_and_necessity_of_p_and_q_l1758_175839

noncomputable def p : Prop := ∀ k, k = Real.sqrt 3
noncomputable def q : Prop := ∀ k, ∃ y x, y = k * x + 2 ∧ x^2 + y^2 = 1

theorem sufficiency_and_necessity_of_p_and_q : (p → q) ∧ (¬ (q → p)) := by
  sorry

end sufficiency_and_necessity_of_p_and_q_l1758_175839


namespace factor_expression_l1758_175858

theorem factor_expression (x : ℝ) :
  84 * x ^ 5 - 210 * x ^ 9 = -42 * x ^ 5 * (5 * x ^ 4 - 2) :=
by
  sorry

end factor_expression_l1758_175858


namespace jason_games_planned_last_month_l1758_175898

-- Define the conditions
variable (games_planned_this_month : Nat) (games_missed : Nat) (games_attended : Nat)

-- Define what we want to prove
theorem jason_games_planned_last_month (h1 : games_planned_this_month = 11)
                                        (h2 : games_missed = 16)
                                        (h3 : games_attended = 12) :
                                        (games_attended + games_missed - games_planned_this_month = 17) := 
by
  sorry

end jason_games_planned_last_month_l1758_175898


namespace binom_n_2_l1758_175872

theorem binom_n_2 (n : ℕ) (h : 1 ≤ n) : Nat.choose n 2 = (n * (n - 1)) / 2 :=
by sorry

end binom_n_2_l1758_175872


namespace problem_statement_l1758_175838

theorem problem_statement :
  ∀ k : Nat, (∃ r s : Nat, r > 0 ∧ s > 0 ∧ (k^2 - 6 * k + 11)^(r - 1) = (2 * k - 7)^s) ↔ (k = 2 ∨ k = 3 ∨ k = 4 ∨ k = 8) :=
by
  sorry

end problem_statement_l1758_175838


namespace bob_grade_is_35_l1758_175888

-- Define the conditions
def jenny_grade : ℕ := 95
def jason_grade : ℕ := jenny_grade - 25
def bob_grade : ℕ := jason_grade / 2

-- State the theorem
theorem bob_grade_is_35 : bob_grade = 35 := by
  sorry

end bob_grade_is_35_l1758_175888


namespace find_y_value_l1758_175822

theorem find_y_value : (12 : ℕ)^3 * (6 : ℕ)^2 / 432 = 144 := by
  -- assumptions and computations are not displayed in the statement
  sorry

end find_y_value_l1758_175822


namespace no_solution_inequality_C_l1758_175823

theorem no_solution_inequality_C : ¬∃ x : ℝ, 2 * x - x^2 > 5 := by
  -- There is no need to include the other options in the Lean theorem, as the proof focuses on the condition C directly.
  sorry

end no_solution_inequality_C_l1758_175823


namespace feed_cost_l1758_175808

theorem feed_cost (total_birds ducks_fraction chicken_feed_cost : ℕ) (h1 : total_birds = 15) (h2 : ducks_fraction = 1/3) (h3 : chicken_feed_cost = 2) :
  15 * (1 - 1/3) * 2 = 20 :=
by
  sorry

end feed_cost_l1758_175808


namespace geom_sequence_sum_of_first4_l1758_175807

noncomputable def geom_sum_first4_terms (a : ℕ → ℝ) (common_ratio : ℝ) (a0 a1 a4 : ℝ) : ℝ :=
  a0 + a0 * common_ratio + a0 * common_ratio^2 + a0 * common_ratio^3

theorem geom_sequence_sum_of_first4 {a : ℕ → ℝ} (a1 a4 : ℝ) (r : ℝ)
  (h1 : a 1 = a1) (h4 : a 4 = a4) 
  (h_geom : ∀ n, a (n + 1) = a n * r) :
  geom_sum_first4_terms a (r) a1 (a 0) (a 4) = 120 :=
by sorry

end geom_sequence_sum_of_first4_l1758_175807


namespace tan_neg_585_eq_neg_1_l1758_175885

theorem tan_neg_585_eq_neg_1 : Real.tan (-585 * Real.pi / 180) = -1 := by
  sorry

end tan_neg_585_eq_neg_1_l1758_175885


namespace teal_sales_l1758_175870

theorem teal_sales
  (pumpkin_pie_slices : ℕ := 8)
  (custard_pie_slices : ℕ := 6)
  (pumpkin_pie_price : ℕ := 5)
  (custard_pie_price : ℕ := 6)
  (pumpkin_pies_sold : ℕ := 4)
  (custard_pies_sold : ℕ := 5) :
  let total_pumpkin_slices := pumpkin_pie_slices * pumpkin_pies_sold
  let total_custard_slices := custard_pie_slices * custard_pies_sold
  let total_pumpkin_sales := total_pumpkin_slices * pumpkin_pie_price
  let total_custard_sales := total_custard_slices * custard_pie_price
  let total_sales := total_pumpkin_sales + total_custard_sales
  total_sales = 340 :=
by
  sorry

end teal_sales_l1758_175870


namespace geometric_series_sum_l1758_175821

theorem geometric_series_sum (a r : ℚ) (h_a : a = 1) (h_r : r = 1 / 3) : 
  (∑' n : ℕ, a * r ^ n) = 3 / 2 := 
by
  sorry

end geometric_series_sum_l1758_175821


namespace no_analytic_roots_l1758_175803

theorem no_analytic_roots : ¬∃ x : ℝ, (x - 2) * (x + 5)^3 * (5 - x) = 8 := 
sorry

end no_analytic_roots_l1758_175803


namespace find_last_number_of_consecutive_even_numbers_l1758_175861

theorem find_last_number_of_consecutive_even_numbers (x : ℕ) (h : 8 * x + 2 + 4 + 6 + 8 + 10 + 12 + 14 = 424) : x + 14 = 60 :=
sorry

end find_last_number_of_consecutive_even_numbers_l1758_175861


namespace count_zero_vectors_l1758_175848

variable {V : Type} [AddCommGroup V]

variables (A B C D M O : V)

def vector_expressions_1 := (A - B) + (B - C) + (C - A) = 0
def vector_expressions_2 := (A - B) + (M - B) + (B - O) + (O - M) ≠ 0
def vector_expressions_3 := (A - B) - (A - C) + (B - D) - (C - D) = 0
def vector_expressions_4 := (O - A) + (O - C) + (B - O) + (C - O) ≠ 0

theorem count_zero_vectors :
  (vector_expressions_1 A B C) ∧
  (vector_expressions_2 A B M O) ∧
  (vector_expressions_3 A B C D) ∧
  (vector_expressions_4 O A C B) →
  (2 = 2) :=
sorry

end count_zero_vectors_l1758_175848


namespace ab_bc_ca_leq_zero_l1758_175844

theorem ab_bc_ca_leq_zero (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end ab_bc_ca_leq_zero_l1758_175844


namespace extra_time_A_to_reach_destination_l1758_175860

theorem extra_time_A_to_reach_destination (speed_ratio : ℕ -> ℕ -> Prop) (t_A t_B : ℝ)
  (h_ratio : speed_ratio 3 4)
  (time_A : t_A = 2)
  (distance_constant : ∀ a b : ℝ, a / b = (3 / 4)) :
  (t_A - t_B) * 60 = 30 :=
by
  sorry

end extra_time_A_to_reach_destination_l1758_175860


namespace domain_of_f_l1758_175817

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1)

theorem domain_of_f : { x : ℝ | x > 1 } = { x : ℝ | ∃ y, f y = f x } :=
by sorry

end domain_of_f_l1758_175817


namespace total_amount_paid_l1758_175886

-- Define the conditions
def chicken_nuggets_ordered : ℕ := 100
def nuggets_per_box : ℕ := 20
def cost_per_box : ℕ := 4

-- Define the hypothesis on the amount of money paid for the chicken nuggets
theorem total_amount_paid :
  (chicken_nuggets_ordered / nuggets_per_box) * cost_per_box = 20 :=
by
  sorry

end total_amount_paid_l1758_175886
