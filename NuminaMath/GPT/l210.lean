import Mathlib

namespace amount_paid_out_l210_210416

theorem amount_paid_out 
  (amount : ℕ) 
  (h1 : amount % 50 = 0) 
  (h2 : ∃ (n : ℕ), n ≥ 15 ∧ amount = n * 5000 ∨ amount = n * 1000)
  (h3 : ∃ (n : ℕ), n ≥ 35 ∧ amount = n * 1000) : 
  amount = 29950 :=
by 
  sorry

end amount_paid_out_l210_210416


namespace largest_possible_value_of_s_l210_210854

theorem largest_possible_value_of_s (r s : Nat) (h1 : r ≥ s) (h2 : s ≥ 3)
  (h3 : (r - 2) * s * 61 = (s - 2) * r * 60) : s = 121 :=
sorry

end largest_possible_value_of_s_l210_210854


namespace sum_of_first_9000_terms_of_geometric_sequence_l210_210894

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_9000_terms_of_geometric_sequence 
  (a r : ℝ)
  (h₁ : geometric_sum a r 3000 = 500)
  (h₂ : geometric_sum a r 6000 = 950)
  : geometric_sum a r 9000 = 1355 :=
sorry

end sum_of_first_9000_terms_of_geometric_sequence_l210_210894


namespace application_schemes_eq_l210_210905

noncomputable def number_of_application_schemes (graduates : ℕ) (universities : ℕ) : ℕ :=
  universities ^ graduates

theorem application_schemes_eq : 
  number_of_application_schemes 5 3 = 3 ^ 5 := 
by 
  -- proof goes here
  sorry

end application_schemes_eq_l210_210905


namespace smallest_multiple_of_18_all_digits_9_or_0_l210_210272

theorem smallest_multiple_of_18_all_digits_9_or_0 :
  ∃ (m : ℕ), (m > 0) ∧ (m % 18 = 0) ∧ (∀ d ∈ (m.digits 10), d = 9 ∨ d = 0) ∧ (m / 18 = 5) :=
sorry

end smallest_multiple_of_18_all_digits_9_or_0_l210_210272


namespace mary_flour_total_l210_210712

-- Definitions for conditions
def initial_flour : ℝ := 7.0
def extra_flour : ℝ := 2.0
def total_flour (x y : ℝ) : ℝ := x + y

-- The statement we want to prove
theorem mary_flour_total : total_flour initial_flour extra_flour = 9.0 := 
by sorry

end mary_flour_total_l210_210712


namespace value_range_f_l210_210424

noncomputable def f (x : ℝ) : ℝ := Real.logb 3 (x^2 - 2*x + 10)

theorem value_range_f :
  (∀ x : ℝ, x^2 - 2*x + 10 ≥ 9) ->
  (∀ y : ℝ, (∃ x : ℝ, f x = y) ↔ y ≥ 2) :=
by
  sorry

end value_range_f_l210_210424


namespace jordan_width_l210_210003

-- Definitions based on conditions
def area_of_carols_rectangle : ℝ := 15 * 20
def jordan_length_feet : ℝ := 6
def feet_to_inches (feet: ℝ) : ℝ := feet * 12
def jordan_length_inches : ℝ := feet_to_inches jordan_length_feet

-- Main statement
theorem jordan_width :
  ∃ w : ℝ, w = 300 / 72 :=
sorry

end jordan_width_l210_210003


namespace probability_maxim_born_in_2008_l210_210254

def days_in_month : ℕ → ℕ → ℕ 
| 2007 9 := 29
| 2007 10 := 31
| 2007 11 := 30
| 2007 12 := 31
| 2008 1 := 31
| 2008 2 := 29   -- leap year
| 2008 3 := 31
| 2008 4 := 30
| 2008 5 := 31
| 2008 6 := 30
| 2008 7 := 31
| 2008 8 := 31
| _ _ := 0

def total_days : ℕ :=
list.sum [days_in_month 2007 9, days_in_month 2007 10, days_in_month 2007 11, days_in_month 2007 12,
          days_in_month 2008 1, days_in_month 2008 2, days_in_month 2008 3, days_in_month 2008 4,
          days_in_month 2008 5, days_in_month 2008 6, days_in_month 2008 7, days_in_month 2008 8]

def days_in_2008 : ℕ :=
list.sum [days_in_month 2008 1, days_in_month 2008 2, days_in_month 2008 3, days_in_month 2008 4,
          days_in_month 2008 5, days_in_month 2008 6, days_in_month 2008 7, days_in_month 2008 8]

theorem probability_maxim_born_in_2008 : 
  (days_in_2008 : ℚ) / (total_days : ℚ) = 244 / 365 :=
by
  -- Placeholder for actual proof. This theorem is set properly and ready for proof development.
  sorry

end probability_maxim_born_in_2008_l210_210254


namespace min_value_dot_product_l210_210957

-- Side length of the square
def side_length: ℝ := 1

-- Definition of points in vector space
variables {A B C D O M N P: Type}

-- Definitions assuming standard Euclidean geometry
variables (O P : ℝ) (a b c : ℝ)

-- Points M and N on the edges AD and BC respectively, line MN passes through O
-- Point P satisfies 2 * vector OP = l * vector OA + (1-l) * vector OB
theorem min_value_dot_product (l : ℝ) (O P M N : ℝ) :
  (2 * (O + P)) = l * (O - a) + (1 - l) * (b + c) ∧
  ((O - P) * (O + P) - ((l^2 - l + 1/2) / 4) = -7/16) :=
by
  sorry

end min_value_dot_product_l210_210957


namespace workman_problem_l210_210304

theorem workman_problem (A B : ℝ) (h1 : A = B / 2) (h2 : (A + B) * 10 = 1) : B = 1 / 15 := by
  sorry

end workman_problem_l210_210304


namespace difference_in_perimeter_is_50_cm_l210_210210

-- Define the lengths of the four ribbons
def ribbon_lengths (x : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  (x, x + 25, x + 50, x + 75)

-- Define the perimeter of the first shape
def perimeter_first_shape (x : ℕ) : ℕ :=
  2 * x + 230

-- Define the perimeter of the second shape
def perimeter_second_shape (x : ℕ) : ℕ :=
  2 * x + 280

-- Define the main theorem that the difference in perimeter is 50 cm
theorem difference_in_perimeter_is_50_cm (x : ℕ) :
  perimeter_second_shape x - perimeter_first_shape x = 50 := by
  sorry

end difference_in_perimeter_is_50_cm_l210_210210


namespace prime_cubed_plus_seven_composite_l210_210213

theorem prime_cubed_plus_seven_composite (P : ℕ) (hP_prime : Nat.Prime P) (hP3_plus_5_prime : Nat.Prime (P ^ 3 + 5)) : ¬ Nat.Prime (P ^ 3 + 7) :=
by
  sorry

end prime_cubed_plus_seven_composite_l210_210213


namespace Tyler_CDs_after_giveaway_and_purchase_l210_210739

theorem Tyler_CDs_after_giveaway_and_purchase :
  (∃ cds_initial cds_giveaway_fraction cds_bought cds_final, 
     cds_initial = 21 ∧ 
     cds_giveaway_fraction = 1 / 3 ∧ 
     cds_bought = 8 ∧ 
     cds_final = cds_initial - (cds_initial * cds_giveaway_fraction) + cds_bought ∧
     cds_final = 22) := 
sorry

end Tyler_CDs_after_giveaway_and_purchase_l210_210739


namespace garden_length_l210_210224

theorem garden_length (columns : ℕ) (distance_between_trees : ℕ) (boundary_distance : ℕ) (h_columns : columns = 12) (h_distance_between_trees : distance_between_trees = 2) (h_boundary_distance : boundary_distance = 5) : 
  ((columns - 1) * distance_between_trees + 2 * boundary_distance) = 32 :=
by 
  sorry

end garden_length_l210_210224


namespace michael_eggs_count_l210_210713

-- Define the conditions: crates bought on Tuesday, crates given to Susan, crates bought on Thursday, and eggs per crate.
def crates_bought_on_tuesday : ℕ := 6
def crates_given_to_susan : ℕ := 2
def crates_bought_on_thursday : ℕ := 5
def eggs_per_crate : ℕ := 30

-- State the theorem to prove.
theorem michael_eggs_count :
  let crates_left = crates_bought_on_tuesday - crates_given_to_susan
  let total_crates = crates_left + crates_bought_on_thursday
  total_crates * eggs_per_crate = 270 :=
by
  -- Proof goes here
  sorry

end michael_eggs_count_l210_210713


namespace x_equals_l210_210492

variable (x y: ℝ)

theorem x_equals:
  (x / (x - 2) = (y^2 + 3 * y + 1) / (y^2 + 3 * y - 1)) → x = 2 * y^2 + 6 * y + 2 := by
  sorry

end x_equals_l210_210492


namespace c_alone_finishes_in_6_days_l210_210309

theorem c_alone_finishes_in_6_days (a b c : ℝ) (W : ℝ) :
  (1 / 36) * W + (1 / 18) * W + (1 / c) * W = (1 / 4) * W → c = 6 :=
by
  intros h
  simp at h
  sorry

end c_alone_finishes_in_6_days_l210_210309


namespace faye_earned_total_l210_210457

-- Definitions of the necklace sales
def bead_necklaces := 3
def bead_price := 7
def gemstone_necklaces := 7
def gemstone_price := 10
def pearl_necklaces := 2
def pearl_price := 12
def crystal_necklaces := 5
def crystal_price := 15

-- Total amount calculation
def total_amount := 
  bead_necklaces * bead_price + 
  gemstone_necklaces * gemstone_price + 
  pearl_necklaces * pearl_price + 
  crystal_necklaces * crystal_price

-- Proving the total amount equals $190
theorem faye_earned_total : total_amount = 190 := by
  sorry

end faye_earned_total_l210_210457


namespace sample_capacity_l210_210161

theorem sample_capacity (f : ℕ) (r : ℚ) (n : ℕ) (h₁ : f = 40) (h₂ : r = 0.125) (h₃ : r * n = f) : n = 320 :=
sorry

end sample_capacity_l210_210161


namespace Janet_initial_crayons_l210_210407

variable (Michelle_initial Janet_initial Michelle_final : ℕ)

theorem Janet_initial_crayons (h1 : Michelle_initial = 2) (h2 : Michelle_final = 4) (h3 : Michelle_final = Michelle_initial + Janet_initial) :
  Janet_initial = 2 :=
by
  sorry

end Janet_initial_crayons_l210_210407


namespace donation_addition_median_mode_l210_210645

def initial_donations : list ℕ := [5, 3, 6, 5, 10]

def sorted_initial_donations : list ℕ := initial_donations.qsort (≤)

-- Proof that a = 1 or a = 2 maintains the median and mode.
def median_donation (l : list ℕ) : ℕ :=
l[(l.length / 2 : ℕ)] -- Assuming list is sorted

def mode_donation (l : list ℕ) : list ℕ :=
l.foldl (λ acc x, if list.count l x > list.count l (list.head! acc) then [x] else if list.count l x = list.count l (list.head! acc) && x ≠ list.head! acc then x :: acc else acc) [list.head! l]

theorem donation_addition_median_mode (a : ℕ) :
  (a = 1 ∨ a = 2) ↔ 
  median_donation ([4, 5, 5, 6, 10].qsort (≤)) = median_donation sorted_initial_donations ∧ 
  mode_donation ([4, 5, 5, 6, 10].qsort (≤)) = mode_donation sorted_initial_donations ∨ 
  median_donation ([5, 5, 5, 6, 10].qsort (≤)) = median_donation sorted_initial_donations ∧ 
  mode_donation ([5, 5, 5, 6, 10].qsort (≤)) = mode_donation sorted_initial_donations ∧ 
  median_donation updated_donations = 5 ∧ mode_donation updated_donations = [5] :=
  sorry

end donation_addition_median_mode_l210_210645


namespace inequality_solution_set_no_positive_a_b_exists_l210_210474

def f (x : ℝ) := abs (2 * x - 1) - abs (2 * x - 2)
def k := 1

theorem inequality_solution_set :
  { x : ℝ | f x ≥ x } = { x : ℝ | x ≤ -1 ∨ x = 1 } :=
sorry

theorem no_positive_a_b_exists (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) :
  ¬ (a + 2 * b = k ∧ 2 / a + 1 / b = 4 - 1 / (a * b)) :=
sorry

end inequality_solution_set_no_positive_a_b_exists_l210_210474


namespace intersection_A_B_is_1_4_close_l210_210245

noncomputable def A : Set ℝ := {x | 1 < x ∧ x < 5}
noncomputable def B : Set ℝ := {x | x^2 - 3 * x - 4 ≤ 0}
noncomputable def intersection_set : Set ℝ := A ∩ B

theorem intersection_A_B_is_1_4_close : intersection_set = {x | 1 < x ∧ x ≤ 4} :=
by
  sorry

end intersection_A_B_is_1_4_close_l210_210245


namespace smallest_gcd_for_system_l210_210465

theorem smallest_gcd_for_system :
  ∃ n : ℕ, n > 0 ∧ 
    (∀ a b c : ℤ,
     gcd (gcd a b) c = n →
     ∃ x y z : ℤ, 
       (x + 2*y + 3*z = a) ∧ 
       (2*x + y - 2*z = b) ∧ 
       (3*x + y + 5*z = c)) ∧ 
  n = 28 :=
sorry

end smallest_gcd_for_system_l210_210465


namespace combination_8_5_is_56_l210_210506

theorem combination_8_5_is_56 : nat.choose 8 5 = 56 :=
by sorry

end combination_8_5_is_56_l210_210506


namespace Josephine_sold_10_liters_l210_210260

def milk_sold (n1 n2 n3 : ℕ) (v1 v2 v3 : ℝ) : ℝ :=
  (v1 * n1) + (v2 * n2) + (v3 * n3)

theorem Josephine_sold_10_liters :
  milk_sold 3 2 5 2 0.75 0.5 = 10 :=
by
  sorry

end Josephine_sold_10_liters_l210_210260


namespace triangle_perimeter_l210_210659

-- Define the given quadratic equation
def quadratic_equation (m : ℝ) (x : ℝ) : ℝ :=
  x^2 - (5 + m) * x + 5 * m

-- Define the isosceles triangle with sides given by the roots of the equation
def is_isosceles_triangle (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

-- Defining the fact that 2 is a root of the given quadratic equation with an unknown m
lemma two_is_root (m : ℝ) : quadratic_equation m 2 = 0 := sorry

-- Prove that the perimeter of triangle ABC is 12 given the conditions
theorem triangle_perimeter (α β γ : ℝ) (m : ℝ) (h1 : quadratic_equation m α = 0) 
  (h2 : quadratic_equation m β = 0) 
  (h3 : is_isosceles_triangle α β γ) : α + β + γ = 12 := sorry

end triangle_perimeter_l210_210659


namespace work_done_in_one_day_l210_210588

theorem work_done_in_one_day (A_days B_days : ℝ) (hA : A_days = 6) (hB : B_days = A_days / 2) : 
  (1 / A_days + 1 / B_days) = 1 / 2 := by
  sorry

end work_done_in_one_day_l210_210588


namespace trig_identity_l210_210195

theorem trig_identity (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α - Real.cos α) / (2 * Real.cos α + Real.sin α) = 3 / 4 :=
by
  sorry

end trig_identity_l210_210195


namespace initial_weights_of_apples_l210_210329

variables {A B : ℕ}

theorem initial_weights_of_apples (h₁ : A + B = 75) (h₂ : A - 5 = (B + 5) + 7) :
  A = 46 ∧ B = 29 :=
by
  sorry

end initial_weights_of_apples_l210_210329


namespace sufficient_drivers_and_correct_time_l210_210152

-- Conditions definitions
def one_way_minutes := 2 * 60 + 40  -- 2 hours 40 minutes in minutes
def round_trip_minutes := 2 * one_way_minutes  -- round trip in minutes
def rest_minutes := 60  -- mandatory rest period in minutes

-- Time checks for drivers
def driver_a_return := 12 * 60 + 40  -- Driver A returns at 12:40 PM in minutes
def driver_a_next_trip := driver_a_return + rest_minutes  -- Driver A's next trip time
def driver_d_departure := 13 * 60 + 5  -- Driver D departs at 13:05 in minutes

-- Verify sufficiency of four drivers and time correctness
theorem sufficient_drivers_and_correct_time : 
  4 = 4 ∧ (driver_a_next_trip + round_trip_minutes = 21 * 60 + 30) :=
by
  -- Explain the reasoning path that leads to this conclusion within this block
  sorry

end sufficient_drivers_and_correct_time_l210_210152


namespace geometric_sequence_a9_value_l210_210963

theorem geometric_sequence_a9_value {a : ℕ → ℝ} (q a1 : ℝ) 
  (h_geom : ∀ n, a n = a1 * q ^ n)
  (h_a3 : a 3 = 2)
  (S : ℕ → ℝ)
  (h_S : ∀ n, S n = a1 * (1 - q ^ n) / (1 - q))
  (h_sum : S 12 = 4 * S 6) : a 9 = 2 := 
by 
  sorry

end geometric_sequence_a9_value_l210_210963


namespace problem_l210_210675

theorem problem (x a : ℝ) (h : x^5 - x^3 + x = a) : x^6 ≥ 2 * a - 1 := 
by 
  sorry

end problem_l210_210675


namespace zamena_correct_l210_210776

variables {M E H A : ℕ}

def valid_digits := {1, 2, 3, 4, 5}

-- Define the conditions as hypotheses
def conditions : Prop :=
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
  M ∈ valid_digits ∧ E ∈ valid_digits ∧ H ∈ valid_digits ∧ A ∈ valid_digits ∧
  ∀ x y, x ∈ {M, E, H, A} ∧ y ∈ {M, E, H, A} → x ≠ y

-- Define the proof problem that these conditions imply the correct answer
theorem zamena_correct : conditions → M = 1 ∧ E = 2 ∧ H = 3 ∧ A = 4 ∧ 10000 * 5 + 1000 * A + 100 * M + 10 * E + H = 541234 :=
by sorry

end zamena_correct_l210_210776


namespace sparrows_among_non_pigeons_l210_210222

theorem sparrows_among_non_pigeons (perc_sparrows perc_pigeons perc_parrots perc_crows : ℝ)
  (h_sparrows : perc_sparrows = 0.40)
  (h_pigeons : perc_pigeons = 0.20)
  (h_parrots : perc_parrots = 0.15)
  (h_crows : perc_crows = 0.25) :
  (perc_sparrows / (1 - perc_pigeons) * 100) = 50 :=
by
  sorry

end sparrows_among_non_pigeons_l210_210222


namespace octahedron_common_sum_is_39_l210_210275

-- Define the vertices of the regular octahedron with numbers from 1 to 12
def vertices : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Define the property that the sum of four numbers at the vertices of each triangle face is the same
def common_sum (faces : List (List ℕ)) (k : ℕ) : Prop :=
  ∀ face ∈ faces, face.sum = k

-- Define the faces of the regular octahedron
def faces : List (List ℕ) := [
  [1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [1, 5, 9, 6],
  [2, 6, 10, 7], [3, 7, 11, 8], [4, 8, 12, 5], [1, 9, 2, 10]
]

-- Prove that the common sum is 39
theorem octahedron_common_sum_is_39 : common_sum faces 39 :=
  sorry

end octahedron_common_sum_is_39_l210_210275


namespace vertical_asymptote_once_l210_210456

theorem vertical_asymptote_once (c : ℝ) : 
  (∀ x : ℝ, (x^2 + 2*x + c) / (x^2 - x - 12) = (x^2 + 2*x + c) / ((x - 4) * (x + 3))) → 
  (c = -24 ∨ c = -3) :=
by 
  sorry

end vertical_asymptote_once_l210_210456


namespace tom_sold_price_l210_210121

noncomputable def original_price : ℝ := 200
noncomputable def tripled_price (price : ℝ) : ℝ := 3 * price
noncomputable def sold_price (price : ℝ) : ℝ := 0.4 * price

theorem tom_sold_price : sold_price (tripled_price original_price) = 240 := 
by
  sorry

end tom_sold_price_l210_210121


namespace isosceles_trapezoid_rotation_produces_frustum_l210_210721

-- Definitions based purely on conditions
structure IsoscelesTrapezoid :=
(a b c d : ℝ) -- sides
(ha : a = c) -- isosceles property
(hb : b ≠ d) -- non-parallel sides

def rotateAroundSymmetryAxis (shape : IsoscelesTrapezoid) : Type :=
-- We need to define what the rotation of the trapezoid produces
sorry

theorem isosceles_trapezoid_rotation_produces_frustum (shape : IsoscelesTrapezoid) :
  rotateAroundSymmetryAxis shape = Frustum :=
sorry

end isosceles_trapezoid_rotation_produces_frustum_l210_210721


namespace calculate_total_cost_l210_210711

def num_chicken_nuggets := 100
def num_per_box := 20
def cost_per_box := 4

theorem calculate_total_cost :
  (num_chicken_nuggets / num_per_box) * cost_per_box = 20 := by
  sorry

end calculate_total_cost_l210_210711


namespace principal_amount_is_26_l210_210112

-- Define the conditions
def rate : Real := 0.07
def time : Real := 6
def simple_interest : Real := 10.92

-- Define the simple interest formula
def simple_interest_formula (P R T : Real) : Real := P * R * T

-- State the theorem to prove
theorem principal_amount_is_26 : 
  ∃ (P : Real), simple_interest_formula P rate time = simple_interest ∧ P = 26 :=
by
  sorry

end principal_amount_is_26_l210_210112


namespace lydia_ate_24_ounces_l210_210284

theorem lydia_ate_24_ounces (total_fruit_pounds : ℕ) (mario_oranges_ounces : ℕ) (nicolai_peaches_pounds : ℕ) (total_fruit_ounces mario_oranges_ounces_in_ounces nicolai_peaches_ounces_in_ounces : ℕ) :
  total_fruit_pounds = 8 →
  mario_oranges_ounces = 8 →
  nicolai_peaches_pounds = 6 →
  total_fruit_ounces = total_fruit_pounds * 16 →
  mario_oranges_ounces_in_ounces = mario_oranges_ounces →
  nicolai_peaches_ounces_in_ounces = nicolai_peaches_pounds * 16 →
  (total_fruit_ounces - mario_oranges_ounces_in_ounces - nicolai_peaches_ounces_in_ounces) = 24 :=
by
  sorry

end lydia_ate_24_ounces_l210_210284


namespace gray_region_area_is_96pi_l210_210124

noncomputable def smaller_circle_diameter : ℝ := 4

noncomputable def smaller_circle_radius : ℝ := smaller_circle_diameter / 2

noncomputable def larger_circle_radius : ℝ := 5 * smaller_circle_radius

noncomputable def area_of_larger_circle : ℝ := Real.pi * (larger_circle_radius ^ 2)

noncomputable def area_of_smaller_circle : ℝ := Real.pi * (smaller_circle_radius ^ 2)

noncomputable def area_of_gray_region : ℝ := area_of_larger_circle - area_of_smaller_circle

theorem gray_region_area_is_96pi : area_of_gray_region = 96 * Real.pi := by
  sorry

end gray_region_area_is_96pi_l210_210124


namespace find_length_of_AC_in_triangle_ABC_l210_210390

noncomputable def length_AC_in_triangle_ABC
  (AB BC : ℝ) (angle_B : ℝ) (h_AB : AB = 1) (h_BC : BC = 2) (h_angle_B : angle_B = Real.pi / 3) :
  ℝ :=
  let cos_B := Real.cos (Real.pi / 3)
  let AC_squared := AB^2 + BC^2 - 2 * AB * BC * cos_B
  Real.sqrt AC_squared

theorem find_length_of_AC_in_triangle_ABC :
  ∃ AC : ℝ, ∀ (AB BC : ℝ) (angle_B : ℝ) (h_AB : AB = 1) (h_BC : BC = 2) (h_angle_B : angle_B = Real.pi / 3),
    length_AC_in_triangle_ABC AB BC angle_B h_AB h_BC h_angle_B = Real.sqrt 3 :=
by sorry

end find_length_of_AC_in_triangle_ABC_l210_210390


namespace math_problem_l210_210917

theorem math_problem :
  3^(5+2) + 4^(1+3) = 39196 ∧
  2^(9+2) - 3^(4+1) = 3661 ∧
  1^(8+6) + 3^(2+3) = 250 ∧
  6^(5+4) - 4^(5+1) = 409977 → 
  5^(7+2) - 2^(5+3) = 1952869 :=
by
  sorry

end math_problem_l210_210917


namespace part1_part2_l210_210481
-- Import the entire Mathlib library for broader usage

-- Definition of the given vectors
def a : ℝ × ℝ := (4, 7)
def b (x : ℝ) : ℝ × ℝ := (x, x + 6)

-- Part 1: Prove the dot product when x = -1 is 31
theorem part1 : (a.1 * (-1) + a.2 * (5)) = 31 := by
  sorry

-- Part 2: Prove the value of x when the vectors are parallel
theorem part2 : (4 : ℝ) / x = (7 : ℝ) / (x + 6) → x = 8 := by
  sorry

end part1_part2_l210_210481


namespace intersection_of_sets_l210_210858

def A (x : ℝ) : Prop := x^2 - 2 * x - 3 ≥ 0
def B (x : ℝ) : Prop := -2 ≤ x ∧ x < 2

theorem intersection_of_sets :
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | -2 ≤ x ∧ x ≤ -1} :=
by
  sorry

end intersection_of_sets_l210_210858


namespace find_m_of_hyperbola_l210_210973

noncomputable def eccen_of_hyperbola (a b : ℝ) : ℝ := Real.sqrt (1 + (b^2) / (a^2))

theorem find_m_of_hyperbola :
  ∃ (m : ℝ), (m > 0) ∧ (eccen_of_hyperbola 2 m = Real.sqrt 3) ∧ (m = 2 * Real.sqrt 2) :=
by
  sorry

end find_m_of_hyperbola_l210_210973


namespace students_in_front_l210_210757

theorem students_in_front (total_students : ℕ) (students_behind : ℕ) (students_total : total_students = 25) (behind_Yuna : students_behind = 9) :
  (total_students - (students_behind + 1)) = 15 :=
by
  sorry

end students_in_front_l210_210757


namespace range_f_x_negative_l210_210816

-- We define the conditions: f is an even function, increasing on (-∞, 0), and f(2) = 0.
variables {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

def increasing_on_neg_infinity_to_zero (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → x < 0 ∧ y < 0 → f x ≤ f y

def f_at_2_is_zero (f : ℝ → ℝ) : Prop :=
  f 2 = 0

-- The theorem to be proven.
theorem range_f_x_negative (hf_even : even_function f)
  (hf_incr : increasing_on_neg_infinity_to_zero f)
  (hf_at2 : f_at_2_is_zero f) :
  ∀ x, f x < 0 ↔ x < -2 ∨ x > 2 :=
by
  sorry

end range_f_x_negative_l210_210816


namespace greatest_int_with_conditions_l210_210567

noncomputable def is_valid_integer (n : ℤ) : Prop :=
  0 < n ∧ n < 200 ∧ Int.gcd n 18 = 6

theorem greatest_int_with_conditions : 
  ∃ n, is_valid_integer n ∧ ∀ m, is_valid_integer m → m ≤ 174 :=
begin
  sorry
end

end greatest_int_with_conditions_l210_210567


namespace base5_representation_three_consecutive_digits_l210_210518

theorem base5_representation_three_consecutive_digits :
  ∃ (digits : ℕ), 
    (digits = 3) ∧ 
    (∃ (a1 a2 a3 : ℕ), 
      94 = a1 * 5^2 + a2 * 5^1 + a3 * 5^0 ∧
      a1 = 3 ∧ a2 = 3 ∧ a3 = 4 ∧
      (a1 = a3 + 1) ∧ (a2 = a3 + 2)) := 
    sorry

end base5_representation_three_consecutive_digits_l210_210518


namespace plane_equation_l210_210639

theorem plane_equation (p q r : ℝ × ℝ × ℝ)
  (h₁ : p = (2, -1, 3))
  (h₂ : q = (0, -1, 5))
  (h₃ : r = (-1, -3, 4)) :
  ∃ A B C D : ℤ, A = 1 ∧ B = 2 ∧ C = -1 ∧ D = 3 ∧
               A > 0 ∧ Int.gcd (Int.gcd (Int.gcd A B) C) D = 1 ∧
               ∀ x y z : ℝ, A * x + B * y + C * z + D = 0 ↔
                             (x, y, z) = p ∨ (x, y, z) = q ∨ (x, y, z) = r :=
by
  sorry

end plane_equation_l210_210639


namespace scientific_notation_correct_l210_210941

def num : ℝ := 1040000000

theorem scientific_notation_correct : num = 1.04 * 10^9 := by
  sorry

end scientific_notation_correct_l210_210941


namespace sin_600_eq_neg_sqrt_3_div_2_l210_210756

theorem sin_600_eq_neg_sqrt_3_div_2 : Real.sin (600 * Real.pi / 180) = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_600_eq_neg_sqrt_3_div_2_l210_210756


namespace min_n_consecutive_integers_sum_of_digits_is_multiple_of_8_l210_210680

theorem min_n_consecutive_integers_sum_of_digits_is_multiple_of_8 
: ∃ n : ℕ, (∀ (nums : Fin n.succ → ℕ), 
              (∀ i j, i < j → nums i < nums j → nums j = nums i + 1) →
              ∃ i, (nums i) % 8 = 0) ∧ n = 15 := 
sorry

end min_n_consecutive_integers_sum_of_digits_is_multiple_of_8_l210_210680


namespace distribute_fruits_l210_210386

theorem distribute_fruits (n m k : ℕ) (h_n : n = 3) (h_m : m = 6) (h_k : k = 1) :
  ((3 ^ n) * (Finset.card ((Finset.Icc 0 m).subsetsOfCard 2).attach)) = 756 :=
by
  sorry

end distribute_fruits_l210_210386


namespace number_of_sides_of_polygon_l210_210061

theorem number_of_sides_of_polygon (n : ℕ) :
  ((n - 2) * 180 = 3 * 360) → n = 8 := 
by
  sorry

end number_of_sides_of_polygon_l210_210061


namespace calculate_brick_height_cm_l210_210596

noncomputable def wall_length_cm : ℕ := 1000  -- 10 m converted to cm
noncomputable def wall_width_cm : ℕ := 800   -- 8 m converted to cm
noncomputable def wall_height_cm : ℕ := 2450 -- 24.5 m converted to cm

noncomputable def wall_volume_cm3 : ℕ := wall_length_cm * wall_width_cm * wall_height_cm

noncomputable def brick_length_cm : ℕ := 20
noncomputable def brick_width_cm : ℕ := 10
noncomputable def number_of_bricks : ℕ := 12250

noncomputable def brick_area_cm2 : ℕ := brick_length_cm * brick_width_cm

theorem calculate_brick_height_cm (h : ℕ) : brick_area_cm2 * h * number_of_bricks = wall_volume_cm3 → 
  h = wall_volume_cm3 / (brick_area_cm2 * number_of_bricks) := by
  sorry

end calculate_brick_height_cm_l210_210596


namespace find_stickers_before_birthday_l210_210523

variable (stickers_received : ℕ) (total_stickers : ℕ)

def stickers_before_birthday (stickers_received total_stickers : ℕ) : ℕ :=
  total_stickers - stickers_received

theorem find_stickers_before_birthday (h1 : stickers_received = 22) (h2 : total_stickers = 61) : 
  stickers_before_birthday stickers_received total_stickers = 39 :=
by 
  have h1 : stickers_received = 22 := h1
  have h2 : total_stickers = 61 := h2
  rw [h1, h2]
  rfl

end find_stickers_before_birthday_l210_210523


namespace candidate_percentage_l210_210595

theorem candidate_percentage (P : ℝ) (l : ℝ) (V : ℝ) : 
  l = 5000.000000000007 ∧ 
  V = 25000.000000000007 ∧ 
  V - 2 * (P / 100) * V = l →
  P = 40 :=
by
  sorry

end candidate_percentage_l210_210595


namespace cells_at_end_of_8th_day_l210_210760

theorem cells_at_end_of_8th_day :
  let initial_cells := 5
  let factor := 3
  let toxin_factor := 1 / 2
  let cells_after_toxin := (initial_cells * factor * factor * factor * toxin_factor : ℤ)
  let final_cells := cells_after_toxin * factor 
  final_cells = 201 :=
by
  sorry

end cells_at_end_of_8th_day_l210_210760


namespace supplement_of_complementary_angle_of_35_deg_l210_210056

theorem supplement_of_complementary_angle_of_35_deg :
  let A := 35
  let C := 90 - A
  let S := 180 - C
  S = 125 :=
by
  let A := 35
  let C := 90 - A
  let S := 180 - C
  -- we need to prove S = 125
  sorry

end supplement_of_complementary_angle_of_35_deg_l210_210056


namespace num_proper_subsets_of_A_l210_210107

open Set

def A : Finset ℕ := {2, 3}

theorem num_proper_subsets_of_A : (A.powerset \ {A, ∅}).card = 3 := by
  sorry

end num_proper_subsets_of_A_l210_210107


namespace exists_arithmetic_progression_with_11_numbers_exists_arithmetic_progression_with_10000_numbers_not_exists_infinite_arithmetic_progression_l210_210591

open Nat

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_arithmetic_progression_with_11_numbers :
  ∃ a d : ℕ, ∀ i j : ℕ, i < 11 → j < 11 → i < j → a + i * d < a + j * d ∧ 
  sum_of_digits (a + i * d) < sum_of_digits (a + j * d) := by
  sorry

theorem exists_arithmetic_progression_with_10000_numbers :
  ∃ a d : ℕ, ∀ i j : ℕ, i < 10000 → j < 10000 → i < j → a + i * d < a + j * d ∧
  sum_of_digits (a + i * d) < sum_of_digits (a + j * d) := by
  sorry

theorem not_exists_infinite_arithmetic_progression :
  ¬ (∃ a d : ℕ, ∀ i j : ℕ, i < j → a + i * d < a + j * d ∧
  sum_of_digits (a + i * d) < sum_of_digits (a + j * d)) := by
  sorry

end exists_arithmetic_progression_with_11_numbers_exists_arithmetic_progression_with_10000_numbers_not_exists_infinite_arithmetic_progression_l210_210591


namespace number_of_squares_l210_210099

theorem number_of_squares (total_streetlights squares_streetlights unused_streetlights : ℕ) 
  (h1 : total_streetlights = 200) 
  (h2 : squares_streetlights = 12) 
  (h3 : unused_streetlights = 20) : 
  (∃ S : ℕ, total_streetlights = squares_streetlights * S + unused_streetlights ∧ S = 15) :=
by
  sorry

end number_of_squares_l210_210099


namespace fourth_polygon_is_square_l210_210594

theorem fourth_polygon_is_square
  (angle_triangle angle_square angle_hexagon : ℕ)
  (h_triangle : angle_triangle = 60)
  (h_square : angle_square = 90)
  (h_hexagon : angle_hexagon = 120)
  (h_total : angle_triangle + angle_square + angle_hexagon = 270) :
  ∃ angle_fourth : ℕ, angle_fourth = 90 ∧ (angle_fourth + angle_triangle + angle_square + angle_hexagon = 360) :=
sorry

end fourth_polygon_is_square_l210_210594


namespace greatest_4_digit_number_l210_210561

theorem greatest_4_digit_number
  (n : ℕ)
  (h1 : n % 5 = 3)
  (h2 : n % 9 = 2)
  (h3 : 1000 ≤ n)
  (h4 : n < 10000) :
  n = 9962 := 
sorry

end greatest_4_digit_number_l210_210561


namespace jerry_total_logs_l210_210396

def logs_from_trees (p m w : Nat) : Nat :=
  80 * p + 60 * m + 100 * w

theorem jerry_total_logs :
  logs_from_trees 8 3 4 = 1220 :=
by
  -- Proof here
  sorry

end jerry_total_logs_l210_210396


namespace base6_arithmetic_l210_210787

def base6_to_base10 (n : ℕ) : ℕ :=
  let d0 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  let n2 := n1 / 10
  let d2 := n2 % 10
  let n3 := n2 / 10
  let d3 := n3 % 10
  let n4 := n3 / 10
  let d4 := n4 % 10
  d4 * 6^4 + d3 * 6^3 + d2 * 6^2 + d1 * 6^1 + d0

def base10_to_base6 (n : ℕ) : ℕ :=
  let b4 := n / 6^4
  let r4 := n % 6^4
  let b3 := r4 / 6^3
  let r3 := r4 % 6^3
  let b2 := r3 / 6^2
  let r2 := r3 % 6^2
  let b1 := r2 / 6^1
  let b0 := r2 % 6^1
  b4 * 10000 + b3 * 1000 + b2 * 100 + b1 * 10 + b0

theorem base6_arithmetic : 
  base10_to_base6 ((base6_to_base10 45321 - base6_to_base10 23454) + base6_to_base10 14553) = 45550 :=
by
  sorry

end base6_arithmetic_l210_210787


namespace smallest_natural_number_with_50_squares_in_interval_l210_210030

theorem smallest_natural_number_with_50_squares_in_interval (a : ℕ) :
  (∀ n : ℕ, n^2 ≤ a → (∃ n_plus_50 : ℕ, (n_plus_50)^2 < 3 * a ∧ 
  (λ n, n^2).range.count_upto (3 * a) - (λ n, n^2).range.count_upto (a) = 50)) → a = 4486 := 
sorry

end smallest_natural_number_with_50_squares_in_interval_l210_210030


namespace conic_sections_of_equation_l210_210174

noncomputable def is_parabola (s : Set (ℝ × ℝ)) : Prop :=
∃ a b c : ℝ, ∀ x y : ℝ, (x, y) ∈ s ↔ y ≠ 0 ∧ y = a * x^3 + b * x + c

theorem conic_sections_of_equation :
  let eq := { p : ℝ × ℝ | p.2^6 - 9 * p.1^6 = 3 * p.2^3 - 1 }
  (is_parabola eq1) → (is_parabola eq2) → (eq = eq1 ∪ eq2) :=
by sorry

end conic_sections_of_equation_l210_210174


namespace joan_final_oranges_l210_210527

def joan_oranges_initial := 75
def tom_oranges := 42
def sara_sold := 40
def christine_added := 15

theorem joan_final_oranges : joan_oranges_initial + tom_oranges - sara_sold + christine_added = 92 :=
by 
  sorry

end joan_final_oranges_l210_210527


namespace even_function_increasing_l210_210206

noncomputable def example_function (x m : ℝ) : ℝ := (m - 1) * x^2 - 2 * m * x + 3

theorem even_function_increasing (m : ℝ) (h : ∀ x : ℝ, example_function x m = example_function (-x) m) :
  (∀ x y : ℝ, x < 0 ∧ y < 0 ∧ x < y → example_function x m < example_function y m) :=
by 
  sorry

end even_function_increasing_l210_210206


namespace num_5_digit_numbers_is_six_l210_210919

-- Define that we have the digits 2, 45, and 68
def digits : List Nat := [2, 45, 68]

-- Function to generate all permutations of given digits
def permute : List Nat → List (List Nat)
| [] => [[]]
| (x::xs) =>
  List.join (List.map (λ ys =>
    List.map (λ zs => x :: zs) (permute xs)) (permute xs))

-- Calculate the number of distinct 5-digit numbers
def numberOf5DigitNumbers : Int := 
  (permute digits).length

-- Theorem to prove the number of distinct 5-digit numbers formed
theorem num_5_digit_numbers_is_six : numberOf5DigitNumbers = 6 := by
  sorry

end num_5_digit_numbers_is_six_l210_210919


namespace min_value_expression_l210_210949

theorem min_value_expression 
  (a b c : ℝ)
  (h1 : a + b + c = -1)
  (h2 : a * b * c ≤ -3) : 
  (ab + 1) / (a + b) + (bc + 1) / (b + c) + (ca + 1) / (c + a) ≥ 3 :=
sorry

end min_value_expression_l210_210949


namespace decimal_to_fraction_l210_210307

theorem decimal_to_fraction (h : 0.36 = 36 / 100): (36 / 100 = 9 / 25) := by
    sorry

end decimal_to_fraction_l210_210307


namespace num_5_digit_even_div_by_5_l210_210489

theorem num_5_digit_even_div_by_5 : ∃! (n : ℕ), n = 500 ∧ ∀ (d : ℕ), 
  10000 ≤ d ∧ d ≤ 99999 → 
  (∀ i, i ∈ [0, 1, 2, 3, 4] → ((d / 10^i) % 10) % 2 = 0) ∧
  (d % 10 = 0) → 
  n = 500 := sorry

end num_5_digit_even_div_by_5_l210_210489


namespace average_speed_return_trip_l210_210615

/--
A train travels from Albany to Syracuse, a distance of 120 miles,
at an average rate of 50 miles per hour. The train then continues
to Rochester, which is 90 miles from Syracuse, before returning
to Albany. On its way to Rochester, the train's average speed is
60 miles per hour. Finally, the train travels back to Albany from
Rochester, with the total travel time of the train, including all
three legs of the journey, being 9 hours and 15 minutes. What was
the average rate of speed of the train on the return trip from
Rochester to Albany?
-/
theorem average_speed_return_trip :
  let dist_Albany_Syracuse := 120 -- miles
  let speed_Albany_Syracuse := 50 -- miles per hour
  let dist_Syracuse_Rochester := 90 -- miles
  let speed_Syracuse_Rochester := 60 -- miles per hour
  let total_travel_time := 9.25 -- hours (9 hours 15 minutes)
  let time_Albany_Syracuse := dist_Albany_Syracuse / speed_Albany_Syracuse
  let time_Syracuse_Rochester := dist_Syracuse_Rochester / speed_Syracuse_Rochester
  let total_time_so_far := time_Albany_Syracuse + time_Syracuse_Rochester
  let time_return_trip := total_travel_time - total_time_so_far
  let dist_return_trip := dist_Albany_Syracuse + dist_Syracuse_Rochester
  let average_speed_return := dist_return_trip / time_return_trip
  average_speed_return = 39.25 :=
by
  -- sorry placeholder for the actual proof
  sorry

end average_speed_return_trip_l210_210615


namespace pete_numbers_count_l210_210541

theorem pete_numbers_count :
  ∃ x_values : Finset Nat, x_values.card = 4 ∧
  ∀ x ∈ x_values, ∃ y z : Nat, 
  0 < x ∧ 0 < y ∧ 0 < z ∧ (x + y) * z = 14 ∧ (x * y) + z = 14 :=
by
  sorry

end pete_numbers_count_l210_210541


namespace pencils_per_student_l210_210117

-- Define the number of pens
def numberOfPens : ℕ := 1001

-- Define the number of pencils
def numberOfPencils : ℕ := 910

-- Define the maximum number of students
def maxNumberOfStudents : ℕ := 91

-- Using the given conditions, prove that each student gets 10 pencils
theorem pencils_per_student :
  (numberOfPencils / maxNumberOfStudents) = 10 :=
by sorry

end pencils_per_student_l210_210117


namespace regular_price_of_polo_shirt_l210_210585

/--
Zane purchases 2 polo shirts from the 40% off rack at the men's store. 
The polo shirts are priced at a certain amount at the regular price. 
He paid $60 for the shirts. 
Prove that the regular price of each polo shirt is $50.
-/
theorem regular_price_of_polo_shirt (P : ℝ) 
  (h1 : ∀ (x : ℝ), x = 0.6 * P → 2 * x = 60) : 
  P = 50 :=
sorry

end regular_price_of_polo_shirt_l210_210585


namespace rectangular_prism_has_8_vertices_l210_210459

def rectangular_prism_vertices := 8

theorem rectangular_prism_has_8_vertices : rectangular_prism_vertices = 8 := by
  sorry

end rectangular_prism_has_8_vertices_l210_210459


namespace goods_train_speed_is_52_l210_210607

def man_train_speed : ℕ := 60 -- speed of the man's train in km/h
def goods_train_length : ℕ := 280 -- length of the goods train in meters
def time_to_pass : ℕ := 9 -- time for the goods train to pass the man in seconds
def relative_speed_kmph : ℕ := (goods_train_length * 3600) / (time_to_pass * 1000) -- relative speed in km/h, calculated as (0.28 km / (9/3600) h)
def goods_train_speed : ℕ := relative_speed_kmph - man_train_speed -- speed of the goods train in km/h

theorem goods_train_speed_is_52 : goods_train_speed = 52 := by
  sorry

end goods_train_speed_is_52_l210_210607


namespace lunks_needed_for_20_apples_l210_210054

-- Define the conditions as given in the problem
def lunks_to_kunks (lunks : ℤ) : ℤ := (4 * lunks) / 7
def kunks_to_apples (kunks : ℤ) : ℤ := (5 * kunks) / 3

-- Define the target function to calculate the number of lunks needed for given apples
def apples_to_lunks (apples : ℤ) : ℤ := 
  let kunks := (3 * apples) / 5
  let lunks := (7 * kunks) / 4
  lunks

-- Prove the given problem
theorem lunks_needed_for_20_apples : apples_to_lunks 20 = 21 := by
  sorry

end lunks_needed_for_20_apples_l210_210054


namespace probability_of_sum_23_is_7_over_200_l210_210762

/-
A fair, twenty-faced die has 19 of its faces numbered from 1 through 18 and 20, and has one blank face.
Another fair, twenty-faced die has 19 of its faces numbered from 1 through 7 and 9 through 20, and has one blank face.
When the two dice are rolled, what is the probability that the sum of the two numbers facing up will be 23?
-/

def die_1_faces := {x | (1 ≤ x ∧ x ≤ 18) ∨ (x = 20)}
def die_2_faces := {x | (1 ≤ x ∧ x ≠ 8 ∧ x ≤ 20)}

def num_ways_to_sum_23 : ℕ :=
  -- List pairs that sum to 23, taking into account the missing faces
  let pairs := [(3, 20), (5, 18), (6, 17), (7, 16), (9, 14), (10, 13), (11, 12), (12, 11), 
                (13, 10), (14, 9), (16, 7), (17, 6), (18, 5), (20, 3)] 
  in pairs.length

def total_possible_outcomes : ℕ := 20 * 20

def probability_sum_23 : ℚ := num_ways_to_sum_23 / total_possible_outcomes

theorem probability_of_sum_23_is_7_over_200 :
  probability_sum_23 = 7 / 200 := by sorry

end probability_of_sum_23_is_7_over_200_l210_210762


namespace simplify_fraction_l210_210343

theorem simplify_fraction (x y z : ℝ) (h : x + 2 * y + z ≠ 0) :
  (x^2 + y^2 - 4 * z^2 + 2 * x * y) / (x^2 + 4 * y^2 - z^2 + 2 * x * z) = (x + y - 2 * z) / (x + z - 2 * y) :=
by
  sorry

end simplify_fraction_l210_210343


namespace time_spent_on_type_a_problems_l210_210914

theorem time_spent_on_type_a_problems 
  (total_problems : ℕ)
  (exam_time_minutes : ℕ)
  (type_a_problems : ℕ)
  (type_b_problem_time : ℕ)
  (total_time_type_a : ℕ)
  (h1 : total_problems = 200)
  (h2 : exam_time_minutes = 180)
  (h3 : type_a_problems = 50)
  (h4 : ∀ x : ℕ, type_b_problem_time = 2 * x)
  (h5 : ∀ x : ℕ, total_time_type_a = type_a_problems * type_b_problem_time)
  : total_time_type_a = 72 := 
by
  sorry

end time_spent_on_type_a_problems_l210_210914


namespace rational_linear_independent_sqrt_prime_l210_210912

theorem rational_linear_independent_sqrt_prime (p : ℕ) (hp : Nat.Prime p) (m n m1 n1 : ℚ) :
  m + n * Real.sqrt p = m1 + n1 * Real.sqrt p → m = m1 ∧ n = n1 :=
sorry

end rational_linear_independent_sqrt_prime_l210_210912


namespace value_of_expression_l210_210132

theorem value_of_expression (x : ℤ) (h : x = -2) : (3 * x + 4)^2 = 4 :=
by
  rw [h]
  norm_num
  sorry

end value_of_expression_l210_210132


namespace number_of_cars_sold_next_four_days_cars_sold_each_day_next_four_days_l210_210806

def cars_sold_each_day_first_three_days : ℕ := 5
def days_first_period : ℕ := 3
def quota : ℕ := 50
def cars_remaining_after_next_four_days : ℕ := 23
def days_next_period : ℕ := 4

theorem number_of_cars_sold_next_four_days :
  (quota - cars_sold_each_day_first_three_days * days_first_period) - cars_remaining_after_next_four_days = 12 :=
by
  sorry

theorem cars_sold_each_day_next_four_days :
  (quota - cars_sold_each_day_first_three_days * days_first_period - cars_remaining_after_next_four_days) / days_next_period = 3 :=
by
  sorry

end number_of_cars_sold_next_four_days_cars_sold_each_day_next_four_days_l210_210806


namespace total_notebooks_l210_210583

-- Definitions from the conditions
def Yoongi_notebooks : Nat := 3
def Jungkook_notebooks : Nat := 3
def Hoseok_notebooks : Nat := 3

-- The proof problem
theorem total_notebooks : Yoongi_notebooks + Jungkook_notebooks + Hoseok_notebooks = 9 := 
by 
  sorry

end total_notebooks_l210_210583


namespace derivative_at_pi_over_4_l210_210350

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem derivative_at_pi_over_4 : (deriv f (Real.pi / 4)) = -2 :=
by
  sorry

end derivative_at_pi_over_4_l210_210350


namespace jack_received_more_emails_l210_210696

-- Definitions representing the conditions
def morning_emails : ℕ := 6
def afternoon_emails : ℕ := 8

-- The theorem statement
theorem jack_received_more_emails : afternoon_emails - morning_emails = 2 := 
by 
  sorry

end jack_received_more_emails_l210_210696


namespace mean_score_classes_is_82_l210_210089

theorem mean_score_classes_is_82
  (F S : ℕ)
  (f s : ℕ)
  (hF : F = 90)
  (hS : S = 75)
  (hf_ratio : f * 6 = s * 5)
  (hf_total : f + s = 66) :
  ((F * f + S * s) / (f + s) : ℚ) = 82 :=
by
  sorry

end mean_score_classes_is_82_l210_210089


namespace rainy_days_l210_210123

theorem rainy_days
  (rain_on_first_day : ℕ) (rain_on_second_day : ℕ) (rain_on_third_day : ℕ) (sum_of_first_two_days : ℕ)
  (h1 : rain_on_first_day = 4)
  (h2 : rain_on_second_day = 5 * rain_on_first_day)
  (h3 : sum_of_first_two_days = rain_on_first_day + rain_on_second_day)
  (h4 : rain_on_third_day = sum_of_first_two_days - 6) :
  rain_on_third_day = 18 :=
by
  sorry

end rainy_days_l210_210123


namespace power_of_product_l210_210449

variable (x y: ℝ)

theorem power_of_product :
  (-2 * x * y^3)^2 = 4 * x^2 * y^6 := 
by
  sorry

end power_of_product_l210_210449


namespace zamena_inequalities_l210_210777

theorem zamena_inequalities :
  ∃ (M E H A : ℕ), 
    M ≠ E ∧ M ≠ H ∧ M ≠ A ∧ 
    E ≠ H ∧ E ≠ A ∧ H ≠ A ∧
    (∃ Z, Z = 5 ∧ -- Assume Z is 5 based on the conclusion
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ A ∧ A ≤ 5 ∧
    3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
    -- ZAMENA evaluates to 541234
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) :=
sorry

end zamena_inequalities_l210_210777


namespace locus_of_P_is_ellipse_locus_of_P_is_hyperbola_l210_210479

variable {A O B P : Point}
variable {R : ℝ}

-- Scenario 1: Prove that the locus of point P is an ellipse with O and A as foci and OB as the major axis length
theorem locus_of_P_is_ellipse (hA_inside_O : A ∈ circle O R) (hB_on_O : B ∈ circle O R)
  (hP_eq_bisector : is_perpendicular_bisector(O, B, P)) :
  is_ellipse_with_foci_and_major_axis_length P O A OB := by
  sorry

-- Scenario 2: Prove that the locus of point P is a hyperbola with O and A as foci and OB as the length of the real axis
theorem locus_of_P_is_hyperbola (hA_outside_O : A ∉ interior (circle O R)) (hB_on_O : B ∈ circle O R)
  (hP_eq_bisector : is_perpendicular_bisector(O, B, P)) :
  is_hyperbola_with_foci_and_real_axis_length P O A OB := by
  sorry

end locus_of_P_is_ellipse_locus_of_P_is_hyperbola_l210_210479


namespace triangle_equilateral_l210_210239

noncomputable def point := (ℝ × ℝ)

noncomputable def D : point := (0, 0)
noncomputable def E : point := (2, 0)
noncomputable def F : point := (1, Real.sqrt 3)

noncomputable def dist (p1 p2 : point) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

noncomputable def D' (l : ℝ) : point :=
  let ED := dist E D
  (D.1 + l * ED * (Real.sqrt 3), D.2 + l * ED)

noncomputable def E' (l : ℝ) : point :=
  let DF := dist D F
  (E.1 + l * DF * (Real.sqrt 3), E.2 + l * DF)

noncomputable def F' (l : ℝ) : point :=
  let DE := dist D E
  (F.1 - 2 * l * DE, F.2 + (Real.sqrt 3 - l * DE))

theorem triangle_equilateral (l : ℝ) (h : l = 1 / Real.sqrt 3) :
  let DD' := dist D (D' l)
  let EE' := dist E (E' l)
  let FF' := dist F (F' l)
  dist (D' l) (E' l) = dist (E' l) (F' l) ∧ dist (E' l) (F' l) = dist (F' l) (D' l) ∧ dist (F' l) (D' l) = dist (D' l) (E' l) := sorry

end triangle_equilateral_l210_210239


namespace smallest_a_has_50_perfect_squares_l210_210023

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def perfect_squares_in_interval (a b : ℕ) : ℕ :=
  (Finset.filter (λ x, is_perfect_square x) (Finset.range ((b + 1) - a))).card

theorem smallest_a_has_50_perfect_squares :
  ∃ a : ℕ, a = 4486 ∧ perfect_squares_in_interval (a + 1) (3 * a - 1) = 50 :=
by
  sorry

end smallest_a_has_50_perfect_squares_l210_210023


namespace sum_of_first_9000_terms_of_geometric_sequence_l210_210892

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_9000_terms_of_geometric_sequence 
  (a r : ℝ)
  (h₁ : geometric_sum a r 3000 = 500)
  (h₂ : geometric_sum a r 6000 = 950)
  : geometric_sum a r 9000 = 1355 :=
sorry

end sum_of_first_9000_terms_of_geometric_sequence_l210_210892


namespace smallest_natural_number_with_50_squares_in_interval_l210_210029

theorem smallest_natural_number_with_50_squares_in_interval (a : ℕ) :
  (∀ n : ℕ, n^2 ≤ a → (∃ n_plus_50 : ℕ, (n_plus_50)^2 < 3 * a ∧ 
  (λ n, n^2).range.count_upto (3 * a) - (λ n, n^2).range.count_upto (a) = 50)) → a = 4486 := 
sorry

end smallest_natural_number_with_50_squares_in_interval_l210_210029


namespace probability_maxim_born_in_2008_l210_210255

def days_in_month : ℕ → ℕ → ℕ 
| 2007 9 := 29
| 2007 10 := 31
| 2007 11 := 30
| 2007 12 := 31
| 2008 1 := 31
| 2008 2 := 29   -- leap year
| 2008 3 := 31
| 2008 4 := 30
| 2008 5 := 31
| 2008 6 := 30
| 2008 7 := 31
| 2008 8 := 31
| _ _ := 0

def total_days : ℕ :=
list.sum [days_in_month 2007 9, days_in_month 2007 10, days_in_month 2007 11, days_in_month 2007 12,
          days_in_month 2008 1, days_in_month 2008 2, days_in_month 2008 3, days_in_month 2008 4,
          days_in_month 2008 5, days_in_month 2008 6, days_in_month 2008 7, days_in_month 2008 8]

def days_in_2008 : ℕ :=
list.sum [days_in_month 2008 1, days_in_month 2008 2, days_in_month 2008 3, days_in_month 2008 4,
          days_in_month 2008 5, days_in_month 2008 6, days_in_month 2008 7, days_in_month 2008 8]

theorem probability_maxim_born_in_2008 : 
  (days_in_2008 : ℚ) / (total_days : ℚ) = 244 / 365 :=
by
  -- Placeholder for actual proof. This theorem is set properly and ready for proof development.
  sorry

end probability_maxim_born_in_2008_l210_210255


namespace probability_area_less_than_perimeter_l210_210611

def valid_sum (s : ℕ) : Prop := 2 ≤ s ∧ s ≤ 3

def dice_sum (d1 d2 : ℕ) : ℕ := d1 + d2

theorem probability_area_less_than_perimeter :
  (finset.filter (λ s, valid_sum s) (finset.Icc 2 14)).card / 48 = 1 / 16 :=
by
  sorry

end probability_area_less_than_perimeter_l210_210611


namespace arithmetic_sequence_sum_neq_l210_210039

theorem arithmetic_sequence_sum_neq (a : ℕ → ℤ) (d : ℤ) (S : ℕ → ℤ)
    (h_arith : ∀ n, a (n + 1) = a n + d)
    (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2)
    (h_abs_eq : abs (a 3) = abs (a 9))
    (h_d_neg : d < 0) : S 5 ≠ S 6 := by
  sorry

end arithmetic_sequence_sum_neq_l210_210039


namespace solutions_of_system_l210_210947

theorem solutions_of_system (x y z : ℝ) :
    (x^2 - y = z^2) → (y^2 - z = x^2) → (z^2 - x = y^2) →
    (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
    (x = 1 ∧ y = 0 ∧ z = -1) ∨ 
    (x = 0 ∧ y = -1 ∧ z = 1) ∨ 
    (x = -1 ∧ y = 1 ∧ z = 0) := by
  sorry

end solutions_of_system_l210_210947


namespace max_non_managers_l210_210435

theorem max_non_managers (n_mngrs n_non_mngrs : ℕ) (hmngrs : n_mngrs = 8) 
                (h_ratio : (5 : ℚ) / 24 < (n_mngrs : ℚ) / n_non_mngrs) :
                n_non_mngrs ≤ 38 :=
by {
  sorry
}

end max_non_managers_l210_210435


namespace geometric_seq_sum_l210_210818

theorem geometric_seq_sum (a : ℝ) (q : ℝ) (ha : a ≠ 0) (hq : q ≠ 1) 
    (hS4 : a * (1 - q^4) / (1 - q) = 1) 
    (hS12 : a * (1 - q^12) / (1 - q) = 13) 
    : a * q^12 * (1 + q + q^2 + q^3) = 27 := 
by
  sorry

end geometric_seq_sum_l210_210818


namespace multiples_of_7_units_digit_7_l210_210823

theorem multiples_of_7_units_digit_7 (n : ℕ) (h1 : n < 150) (h2 : ∃ (k : ℕ), n = 7 * k) (h3 : n % 10 = 7) : 
    ∃ (m : ℕ), m = 2 := 
by
  sorry

end multiples_of_7_units_digit_7_l210_210823


namespace no_such_integers_exist_l210_210635

theorem no_such_integers_exist :
  ¬ ∃ (a b : ℕ), a ≥ 1 ∧ b ≥ 1 ∧ ∃ k₁ k₂ : ℕ, (a^5 * b + 3 = k₁^3) ∧ (a * b^5 + 3 = k₂^3) :=
by
  sorry

end no_such_integers_exist_l210_210635


namespace unique_solution_m_n_eq_l210_210007

theorem unique_solution_m_n_eq (m n : ℕ) (h : m^2 = (10 * n + 1) * n + 2) : (m, n) = (11, 7) := by
  sorry

end unique_solution_m_n_eq_l210_210007


namespace min_value_expression_l210_210716
-- We begin by importing the necessary mathematical libraries

-- Definitions based on the conditions
def X_distribution : ℝ → ℝ := sorry -- we assume a normal distribution N(10, σ^2)

-- Define the probabilities given in the conditions
def P_X_gt_12 : ℝ := sorry -- P(X > 12) = m
def P_8_le_X_le_10 : ℝ := sorry -- P(8 ≤ X ≤ 10) = n

-- Define the expressions we are interested in
def m : ℝ := P_X_gt_12
def n : ℝ := P_8_le_X_le_10
def expression := (2 / m) + (1 / n)

-- State the theorem to be proven
theorem min_value_expression : expression = 6 + 4*real.sqrt 2 := sorry

end min_value_expression_l210_210716


namespace solution_in_quadrant_II_l210_210532

theorem solution_in_quadrant_II (k x y : ℝ) (h1 : 2 * x + y = 6) (h2 : k * x - y = 4) : x < 0 ∧ y > 0 ↔ k < -2 :=
by
  sorry

end solution_in_quadrant_II_l210_210532


namespace karen_cases_pickup_l210_210216

theorem karen_cases_pickup (total_boxes cases_per_box: ℕ) (h1 : total_boxes = 36) (h2 : cases_per_box = 12):
  total_boxes / cases_per_box = 3 :=
by
  -- We insert a placeholder to skip the proof here
  sorry

end karen_cases_pickup_l210_210216


namespace cos_double_angle_l210_210810

variable (θ : ℝ)

theorem cos_double_angle (h : Real.tan (θ + Real.pi / 4) = 3) : Real.cos (2 * θ) = 3 / 5 :=
sorry

end cos_double_angle_l210_210810


namespace Annette_more_than_Sara_l210_210933

variable (A C S : ℕ)

-- Define the given conditions as hypotheses
def Annette_Caitlin_weight : Prop := A + C = 95
def Caitlin_Sara_weight : Prop := C + S = 87

-- The theorem to prove: Annette weighs 8 pounds more than Sara
theorem Annette_more_than_Sara (h1 : Annette_Caitlin_weight A C)
                               (h2 : Caitlin_Sara_weight C S) :
  A - S = 8 := by
  sorry

end Annette_more_than_Sara_l210_210933


namespace carrots_planted_per_hour_l210_210487

theorem carrots_planted_per_hour (rows plants_per_row hours : ℕ) (h1 : rows = 400) (h2 : plants_per_row = 300) (h3 : hours = 20) :
  (rows * plants_per_row) / hours = 6000 := by
  sorry

end carrots_planted_per_hour_l210_210487


namespace factor_expr_l210_210178

theorem factor_expr (x : ℝ) : 81 - 27 * x^3 = 27 * (3 - x) * (9 + 3 * x + x^2) := 
sorry

end factor_expr_l210_210178


namespace pentagon_same_parity_l210_210141

open Classical

theorem pentagon_same_parity (vertices : Fin 5 → ℤ × ℤ) : 
  ∃ i j : Fin 5, i ≠ j ∧ (vertices i).1 % 2 = (vertices j).1 % 2 ∧ (vertices i).2 % 2 = (vertices j).2 % 2 :=
by
  sorry

end pentagon_same_parity_l210_210141


namespace find_a_b_g_increasing_l210_210970

noncomputable def f (a b x : ℝ) : ℝ := (1 + a * x^2) / (x + b)

noncomputable def g (a b x : ℝ) : ℝ := x * f a b x

theorem find_a_b (a b : ℝ) (h₁ : f a b 1 = 3) (h₂ : ∀ x : ℝ, g a b x = g a b (-x)) :
  a = 2 ∧ b = 0 := 
  sorry

theorem g_increasing (a b : ℝ) (h₁ : f a b 1 = 3) (h₂ : ∀ x : ℝ, g a b x = g a b (-x)) 
  (ha : a = 2) (hb : b = 0) : 
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → g a b x₁ < g a b x₂ := 
  sorry

end find_a_b_g_increasing_l210_210970


namespace find_x_solution_l210_210010

theorem find_x_solution :
  ∃ x, 2 ^ (x / 2) * (Real.sqrt ((2 + Real.sqrt 3) ^ x) + Real.sqrt ((2 - Real.sqrt 3) ^ x)) = 6 ∧
       x = 2 * Real.log 1.5 / Real.log 2 := by
  sorry

end find_x_solution_l210_210010


namespace choose_five_from_eight_l210_210509

theorem choose_five_from_eight : Nat.choose 8 5 = 56 :=
by
  sorry 

end choose_five_from_eight_l210_210509


namespace directrix_of_parabola_l210_210872

theorem directrix_of_parabola (p : ℝ) (hp : 2 * p = 4) : 
  (∃ x : ℝ, x = -1) :=
by
  sorry

end directrix_of_parabola_l210_210872


namespace kim_probability_red_and_blue_shoe_l210_210235

noncomputable def probability_red_and_blue_shoe : ℚ :=
  if Kim_has_10_pairs_of_shoes : ∃ s : Finset (Fin 20), s.card = 20 ∧ (∀ p ∈ powerset_len 2 s, pair_color_differs p) -- Conditions Kim_has_10_pairs_of_shoes
  then
    let red_probability := 2/20 in  -- Probability of picking a red shoe
    let blue_probability := 2/19 in -- Probability of picking a blue shoe given a red shoe has been picked
    red_probability * blue_probability  -- Combined probability of both events
  else
    0

theorem kim_probability_red_and_blue_shoe (Kim_has_10_pairs_of_shoes : ∃ s : Finset (Fin 20), s.card = 20 ∧ (∀ p ∈ powerset_len 2 s, pair_color_differs p)) :
  probability_red_and_blue_shoe = 1 / 95 :=
by
  swap  -- This theorem proves the equality
  sorry

end kim_probability_red_and_blue_shoe_l210_210235


namespace solution_set_inequality_l210_210458

open Set

theorem solution_set_inequality :
  {x : ℝ | (x+1)/(x-4) ≥ 3} = Iio 4 ∪ Ioo 4 (13/2) ∪ {13/2} :=
by
  sorry

end solution_set_inequality_l210_210458


namespace ship_lighthouse_distance_l210_210928

-- Definitions for conditions
def speed : ℝ := 15 -- speed of the ship in km/h
def time : ℝ := 4  -- time the ship sails eastward in hours
def angle_A : ℝ := 60 -- angle at point A in degrees
def angle_C : ℝ := 30 -- angle at point C in degrees

-- Main theorem statement
theorem ship_lighthouse_distance (d_A_C : ℝ) (d_C_B : ℝ) : d_A_C = speed * time → d_C_B = 60 := 
by sorry

end ship_lighthouse_distance_l210_210928


namespace functional_equation_solution_l210_210436

noncomputable def func_form (f : ℝ → ℝ) : Prop :=
  ∃ α β : ℝ, (α = 1 ∨ α = -1 ∨ α = 0) ∧ (∀ x, f x = α * x + β ∨ f x = α * x ^ 3 + β)

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ a b c : ℝ, (f a - f b) * (f b - f c) * (f c - f a) = f (a * b ^ 2 + b * c ^ 2 + c * a ^ 2) - f (a ^ 2 * b + b ^ 2 * c + c ^ 2 * a)) →
  func_form f :=
sorry

end functional_equation_solution_l210_210436


namespace count_multiples_of_7_with_units_digit_7_less_than_150_l210_210831

-- Definition of the problem's conditions and expected result
theorem count_multiples_of_7_with_units_digit_7_less_than_150 : 
  let multiples_of_7 := { n : Nat // n < 150 ∧ n % 7 = 0 }
  let units_digit_7_multiples := { n : Nat // n < 150 ∧ n % 7 = 0 ∧ n % 10 = 7 }
  List.length (units_digit_7_multiples.members) = 3 :=
by
  sorry

end count_multiples_of_7_with_units_digit_7_less_than_150_l210_210831


namespace jacob_calories_l210_210698

theorem jacob_calories (goal : ℕ) (breakfast : ℕ) (lunch : ℕ) (dinner : ℕ) 
  (h_goal : goal = 1800) 
  (h_breakfast : breakfast = 400) 
  (h_lunch : lunch = 900) 
  (h_dinner : dinner = 1100) : 
  (breakfast + lunch + dinner) - goal = 600 :=
by 
  sorry

end jacob_calories_l210_210698


namespace greatest_integer_gcd_l210_210562

theorem greatest_integer_gcd (n : ℕ) (h1 : n < 200) (h2 : gcd n 18 = 6) : n = 192 :=
sorry

end greatest_integer_gcd_l210_210562


namespace probability_divisor_l210_210294

theorem probability_divisor (n : ℕ) (hn : n = 30) :
  (Nat.card {x // x ∣ n ∧ 1 ≤ x ∧ x ≤ n}.to_finset) / (Nat.card {x // 1 ≤ x ∧ x ≤ n}.to_finset : ℚ) = 4 / 15 :=
by
  sorry

end probability_divisor_l210_210294


namespace glass_heavier_than_plastic_l210_210279

-- Define the conditions
def condition1 (G : ℕ) : Prop := 3 * G = 600
def condition2 (G P : ℕ) : Prop := 4 * G + 5 * P = 1050

-- Define the theorem to prove
theorem glass_heavier_than_plastic (G P : ℕ) (h1 : condition1 G) (h2 : condition2 G P) : G - P = 150 :=
by
  sorry

end glass_heavier_than_plastic_l210_210279


namespace sum_of_roots_l210_210191

theorem sum_of_roots (x y : ℝ) (h : ∀ z, z^2 + 2023 * z - 2024 = 0 → z = x ∨ z = y) : x + y = -2023 := 
by
  sorry

end sum_of_roots_l210_210191


namespace find_number_l210_210143

theorem find_number (x : ℝ) (h : 0.45 * x = 162) : x = 360 :=
sorry

end find_number_l210_210143


namespace javier_average_hits_per_game_l210_210078

theorem javier_average_hits_per_game (total_games_first_part : ℕ) (average_hits_first_part : ℕ) 
  (remaining_games : ℕ) (average_hits_remaining : ℕ) : 
  total_games_first_part = 20 → average_hits_first_part = 2 → 
  remaining_games = 10 → average_hits_remaining = 5 →
  (total_games_first_part * average_hits_first_part + 
  remaining_games * average_hits_remaining) /
  (total_games_first_part + remaining_games) = 3 := 
by intros h1 h2 h3 h4;
   sorry

end javier_average_hits_per_game_l210_210078


namespace sufficient_drivers_and_completion_time_l210_210156

noncomputable def one_way_trip_minutes : ℕ := 2 * 60 + 40
noncomputable def round_trip_minutes : ℕ := 2 * one_way_trip_minutes
noncomputable def rest_period_minutes : ℕ := 60
noncomputable def twelve_forty_pm : ℕ := 12 * 60 + 40 -- in minutes from midnight
noncomputable def one_forty_pm : ℕ := twelve_forty_pm + rest_period_minutes
noncomputable def thirteen_five_pm : ℕ := 13 * 60 + 5 -- 1:05 PM
noncomputable def sixteen_ten_pm : ℕ := 16 * 60 + 10 -- 4:10 PM
noncomputable def sixteen_pm : ℕ := 16 * 60 -- 4:00 PM
noncomputable def seventeen_thirty_pm : ℕ := 17 * 60 + 30 -- 5:30 PM
noncomputable def twenty_one_thirty_pm : ℕ := sixteen_ten_pm + round_trip_minutes -- 9:30 PM (21:30)

theorem sufficient_drivers_and_completion_time :
  4 = 4 ∧ twenty_one_thirty_pm = 21 * 60 + 30 := by
  sorry 

end sufficient_drivers_and_completion_time_l210_210156


namespace distance_between_points_l210_210629

theorem distance_between_points (a b c d m k : ℝ) 
  (h1 : b = 2 * m * a + k) (h2 : d = -m * c + k) : 
  (Real.sqrt ((c - a)^2 + (d - b)^2)) = Real.sqrt ((1 + m^2) * (c - a)^2) := 
by {
  sorry
}

end distance_between_points_l210_210629


namespace factor_81_sub_27x3_l210_210180

theorem factor_81_sub_27x3 (x : ℝ) : 81 - 27 * x^3 = 3 * (3 - x) * (81 + 27 * x + 9 * x^2) :=
sorry

end factor_81_sub_27x3_l210_210180


namespace polygon_num_sides_l210_210988

theorem polygon_num_sides (s : ℕ) (h : 180 * (s - 2) > 2790) : s = 18 :=
sorry

end polygon_num_sides_l210_210988


namespace fraction_of_garden_occupied_by_triangle_beds_l210_210770

theorem fraction_of_garden_occupied_by_triangle_beds :
  ∀ (rect_height rect_width trapezoid_short_base trapezoid_long_base : ℝ) 
    (num_triangles : ℕ) 
    (triangle_leg_length : ℝ)
    (total_area_triangles : ℝ)
    (total_garden_area : ℝ)
    (fraction : ℝ),
  rect_height = 10 → rect_width = 30 →
  trapezoid_short_base = 20 → trapezoid_long_base = 30 → num_triangles = 3 →
  triangle_leg_length = 10 / 3 →
  total_area_triangles = 3 * (1 / 2 * (triangle_leg_length ^ 2)) →
  total_garden_area = rect_height * rect_width →
  fraction = total_area_triangles / total_garden_area →
  fraction = 1 / 18 := by
  intros rect_height rect_width trapezoid_short_base trapezoid_long_base
         num_triangles triangle_leg_length total_area_triangles
         total_garden_area fraction
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end fraction_of_garden_occupied_by_triangle_beds_l210_210770


namespace min_cost_correct_l210_210018

noncomputable def min_cost_to_feed_group : ℕ :=
  let main_courses := 50
  let salads := 30
  let soups := 15
  let price_salad := 200
  let price_soup_main := 350
  let price_salad_main := 350
  let price_all_three := 500
  17000

theorem min_cost_correct : min_cost_to_feed_group = 17000 :=
by
  sorry

end min_cost_correct_l210_210018


namespace total_legs_in_farm_l210_210116

theorem total_legs_in_farm (total_animals : ℕ) (total_cows : ℕ) (cow_legs : ℕ) (duck_legs : ℕ) 
  (h_total_animals : total_animals = 15) (h_total_cows : total_cows = 6) 
  (h_cow_legs : cow_legs = 4) (h_duck_legs : duck_legs = 2) :
  total_cows * cow_legs + (total_animals - total_cows) * duck_legs = 42 :=
by
  sorry

end total_legs_in_farm_l210_210116


namespace domain_of_f_l210_210292

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f : { x : ℝ | f x = (1 / ((x - 3) + (x - 9))) } = {x : ℝ | x ∈ (-∞, 6) ∪ (6, ∞)} :=
by
  sorry

end domain_of_f_l210_210292


namespace pradeep_failed_marks_l210_210092

theorem pradeep_failed_marks
    (total_marks : ℕ)
    (obtained_marks : ℕ)
    (pass_percentage : ℕ)
    (pass_marks : ℕ)
    (fail_marks : ℕ)
    (total_marks_eq : total_marks = 2075)
    (obtained_marks_eq : obtained_marks = 390)
    (pass_percentage_eq : pass_percentage = 20)
    (pass_marks_eq : pass_marks = (pass_percentage * total_marks) / 100)
    (fail_marks_eq : fail_marks = pass_marks - obtained_marks) :
    fail_marks = 25 :=
by
  rw [total_marks_eq, obtained_marks_eq, pass_percentage_eq] at *
  sorry

end pradeep_failed_marks_l210_210092


namespace stratified_sampling_total_sample_size_l210_210768

-- Definitions based on conditions
def pure_milk_brands : ℕ := 30
def yogurt_brands : ℕ := 10
def infant_formula_brands : ℕ := 35
def adult_milk_powder_brands : ℕ := 25
def sampled_infant_formula_brands : ℕ := 7

-- The goal is to prove that the total sample size n is 20.
theorem stratified_sampling_total_sample_size : 
  let total_brands := pure_milk_brands + yogurt_brands + infant_formula_brands + adult_milk_powder_brands
  let sampling_fraction := sampled_infant_formula_brands / infant_formula_brands
  let pure_milk_samples := pure_milk_brands * sampling_fraction
  let yogurt_samples := yogurt_brands * sampling_fraction
  let adult_milk_samples := adult_milk_powder_brands * sampling_fraction
  let n := pure_milk_samples + yogurt_samples + sampled_infant_formula_brands + adult_milk_samples
  n = 20 :=
by
  sorry

end stratified_sampling_total_sample_size_l210_210768


namespace abs_inequality_solution_l210_210867

theorem abs_inequality_solution (x : ℝ) : |x - 1| + |x - 3| < 8 ↔ -2 < x ∧ x < 6 :=
by sorry

end abs_inequality_solution_l210_210867


namespace page_mistakenly_added_twice_l210_210732

theorem page_mistakenly_added_twice (n k: ℕ) (h₁: n = 77) (h₂: (n * (n + 1)) / 2 + k = 3050) : k = 47 :=
by
  -- sorry here to indicate the proof is not needed
  sorry

end page_mistakenly_added_twice_l210_210732


namespace symmetric_point_correct_l210_210469

def symmetric_point (P A : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x₁, y₁, z₁) := P
  let (x₀, y₀, z₀) := A
  (2 * x₀ - x₁, 2 * y₀ - y₁, 2 * z₀ - z₁)

def P : ℝ × ℝ × ℝ := (3, -2, 4)
def A : ℝ × ℝ × ℝ := (0, 1, -2)
def expected_result : ℝ × ℝ × ℝ := (-3, 4, -8)

theorem symmetric_point_correct : symmetric_point P A = expected_result :=
  by
    sorry

end symmetric_point_correct_l210_210469


namespace pos_int_divides_l210_210637

theorem pos_int_divides (n : ℕ) (h₀ : 0 < n) (h₁ : (n - 1) ∣ (n^3 + 4)) : n = 2 ∨ n = 6 :=
by sorry

end pos_int_divides_l210_210637


namespace arithmetic_sequence_sum_l210_210040

theorem arithmetic_sequence_sum 
  (a : ℕ → ℤ) 
  (h_arith : ∀ n : ℕ, a (n+1) - a n = a 1 - a 0) 
  (h_sum : a 2 + a 4 + a 6 = 12) : 
  (a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 = 28) :=
sorry

end arithmetic_sequence_sum_l210_210040


namespace part1_part2_l210_210581

-- Define the main condition of the farthest distance formula
def distance_formula (S h : ℝ) : Prop := S^2 = 1.7 * h

-- Define part 1: Given h = 1.7, prove S = 1.7
theorem part1
  (h : ℝ)
  (hyp : h = 1.7)
  : ∃ S : ℝ, distance_formula S h ∧ S = 1.7 :=
by
  sorry
  
-- Define part 2: Given S = 6.8 and height of eyes to ground 1.5, prove the height of tower = 25.7
theorem part2
  (S : ℝ)
  (h1 : ℝ)
  (height_eyes_to_ground : ℝ)
  (hypS : S = 6.8)
  (height_eyes_to_ground_eq : height_eyes_to_ground = 1.5)
  : ∃ h : ℝ, distance_formula S h ∧ (h - height_eyes_to_ground) = 25.7 :=
by
  sorry

end part1_part2_l210_210581


namespace ball_initial_height_l210_210446

theorem ball_initial_height (c : ℝ) (d : ℝ) (h : ℝ) 
  (H1 : c = 4 / 5) 
  (H2 : d = 1080) 
  (H3 : d = h + 2 * h * c / (1 - c)) : 
  h = 216 :=
sorry

end ball_initial_height_l210_210446


namespace count_valid_prime_pairs_l210_210204

theorem count_valid_prime_pairs (x y : ℕ) (h₁ : Prime x) (h₂ : Prime y) (h₃ : x ≠ y) (h₄ : (621 * x * y) % (x + y) = 0) : 
  ∃ p, p = 6 := by
  sorry

end count_valid_prime_pairs_l210_210204


namespace solve_for_x_l210_210370

theorem solve_for_x (x : ℤ) (h : 3 * x = 2 * x + 6) : x = 6 := by
  sorry

end solve_for_x_l210_210370


namespace white_washing_cost_l210_210417

theorem white_washing_cost
    (length width height : ℝ)
    (door_width door_height window_width window_height : ℝ)
    (num_doors num_windows : ℝ)
    (paint_cost : ℝ)
    (extra_paint_fraction : ℝ)
    (perimeter := 2 * (length + width))
    (door_area := num_doors * (door_width * door_height))
    (window_area := num_windows * (window_width * window_height))
    (wall_area := perimeter * height)
    (paint_area := wall_area - door_area - window_area)
    (total_area := paint_area * (1 + extra_paint_fraction))
    : total_area * paint_cost = 6652.8 :=
by sorry

end white_washing_cost_l210_210417


namespace linear_function_expression_l210_210359

def linear_function_through_point_and_parallel (f : ℝ → ℝ) : Prop :=
  (f 0 = 5) ∧ (∀ x, f x = x + 5)

theorem linear_function_expression :
  ∃ f : ℝ → ℝ, (f 0 = 5) ∧ (∀ x, f x = x + 5) :=
sorry

end linear_function_expression_l210_210359


namespace A_profit_share_l210_210617

theorem A_profit_share (A_shares : ℚ) (B_shares : ℚ) (C_shares : ℚ) (D_shares : ℚ) (total_profit : ℚ) (A_profit : ℚ) :
  A_shares = 1/3 → B_shares = 1/4 → C_shares = 1/5 → 
  D_shares = 1 - (A_shares + B_shares + C_shares) → total_profit = 2445 → A_profit = 815 →
  A_shares * total_profit = A_profit :=
by sorry

end A_profit_share_l210_210617


namespace bus_routes_in_city_l210_210223

noncomputable def bus_routes_problem (n : ℕ) : ℕ :=
  if n = 3 then 1 else if n = 7 then 7 else 0

theorem bus_routes_in_city {n : ℕ} (h1 : ∀ route, route ∈ {3}) (h2 : ∀ stops, ∃ route, route has_bus_route stops)
    (h3 : ∀ route1 route2, route1 ≠ route2 → ∃ common_stop, common_stop ∈ route1 ∧ common_stop ∈ route2) :
  bus_routes_problem n = 1 ∨ bus_routes_problem n = 7 :=
sorry

end bus_routes_in_city_l210_210223


namespace total_eggs_michael_has_l210_210714

-- Define the initial number of crates
def initial_crates : ℕ := 6

-- Define the number of crates given to Susan
def crates_given_to_susan : ℕ := 2

-- Define the number of crates bought on Thursday
def crates_bought_thursday : ℕ := 5

-- Define the number of eggs per crate
def eggs_per_crate : ℕ := 30

-- Theorem stating the total number of eggs Michael has now
theorem total_eggs_michael_has :
  (initial_crates - crates_given_to_susan + crates_bought_thursday) * eggs_per_crate = 270 :=
sorry

end total_eggs_michael_has_l210_210714


namespace domain_h_l210_210638

noncomputable def h (x : ℝ) : ℝ := (3 * x - 1) / Real.sqrt (x - 5)

theorem domain_h (x : ℝ) : h x = (3 * x - 1) / Real.sqrt (x - 5) → (x > 5) :=
by
  intro hx
  have hx_nonneg : x - 5 >= 0 := sorry
  have sqrt_nonzero : Real.sqrt (x - 5) ≠ 0 := sorry
  sorry

end domain_h_l210_210638


namespace reduced_price_per_kg_of_oil_l210_210319

/-- The reduced price per kg of oil is approximately Rs. 48 -
given a 30% reduction in price and the ability to buy 5 kgs more
for Rs. 800. -/
theorem reduced_price_per_kg_of_oil
  (P R : ℝ)
  (h1 : R = 0.70 * P)
  (h2 : 800 / R = (800 / P) + 5) : 
  R = 48 :=
sorry

end reduced_price_per_kg_of_oil_l210_210319


namespace smallest_a_with_50_perfect_squares_l210_210033

theorem smallest_a_with_50_perfect_squares : 
  ∃ a : ℕ, (∀ n : ℕ, (n^2 > a → (n^2 < 3 * a → 50 ∃ k, (k^2 >= a ∧ k^2 < 3 * a) ↔ k = (n+1)) → a = 4486

end smallest_a_with_50_perfect_squares_l210_210033


namespace correct_calculation_l210_210848

theorem correct_calculation :
  (- (4 + 2 / 3) - (1 + 5 / 6) - (- (18 + 1 / 2)) + (- (13 + 3 / 4))) = - (7 / 4) :=
by 
  sorry

end correct_calculation_l210_210848


namespace scientific_notation_of_1040000000_l210_210942

theorem scientific_notation_of_1040000000 : (1.04 * 10^9 = 1040000000) :=
by
  -- Math proof steps can be added here
  sorry

end scientific_notation_of_1040000000_l210_210942


namespace barneys_grocery_store_items_left_l210_210621

theorem barneys_grocery_store_items_left 
    (ordered_items : ℕ) 
    (sold_items : ℕ) 
    (storeroom_items : ℕ) 
    (damaged_percentage : ℝ)
    (h1 : ordered_items = 4458) 
    (h2 : sold_items = 1561) 
    (h3 : storeroom_items = 575) 
    (h4 : damaged_percentage = 5/100) : 
    ordered_items - (sold_items + ⌊damaged_percentage * ordered_items⌋) + storeroom_items = 3250 :=
by
    sorry

end barneys_grocery_store_items_left_l210_210621


namespace star_7_2_l210_210455

def star (a b : ℕ) := 4 * a - 4 * b

theorem star_7_2 : star 7 2 = 20 := 
by
  sorry

end star_7_2_l210_210455


namespace find_w_l210_210937

variables (w x y z : ℕ)

-- conditions
def condition1 : Prop := x = w / 2
def condition2 : Prop := y = w + x
def condition3 : Prop := z = 400
def condition4 : Prop := w + x + y + z = 1000

-- problem to prove
theorem find_w (h1 : condition1 w x) (h2 : condition2 w x y) (h3 : condition3 z) (h4 : condition4 w x y z) : w = 200 :=
by sorry

end find_w_l210_210937


namespace sequence_general_term_l210_210873

theorem sequence_general_term (n : ℕ) (h : n > 0) : 
  ∃ a : ℕ → ℚ, (∀ n, a n = 1 / n) :=
by
  sorry

end sequence_general_term_l210_210873


namespace find_k_l210_210207

variable {x y k : ℝ}

theorem find_k (h1 : 3 * x + 4 * y = k + 2) 
             (h2 : 2 * x + y = 4) 
             (h3 : x + y = 2) :
  k = 4 := 
by
  sorry

end find_k_l210_210207


namespace linear_function_product_neg_l210_210660

theorem linear_function_product_neg (a1 b1 a2 b2 : ℝ) (hP : b1 = -3 * a1 + 4) (hQ : b2 = -3 * a2 + 4) :
  (a1 - a2) * (b1 - b2) < 0 :=
by
  sorry

end linear_function_product_neg_l210_210660


namespace regions_bounded_by_blue_lines_l210_210519

theorem regions_bounded_by_blue_lines (n : ℕ) : 
  (2 * n^2 + 3 * n + 2) -(n - 1) * (2 * n + 1) ≥ 4 * n + 2 :=
by
  sorry

end regions_bounded_by_blue_lines_l210_210519


namespace maxim_birth_probability_l210_210251

open Nat

def interval_days (start_date end_date : ℕ) : ℕ :=
  end_date - start_date + 1

def total_days_2007_2008 : ℕ :=
  interval_days 245 2735 -- total days from Sep 2, 2007, to Aug 31, 2008

def days_in_2008 : ℕ :=
  interval_days 305 548  -- total days from Jan 1, 2008, to Aug 31, 2008

def probability_born_in_2008 : ℚ :=
  (days_in_2008 : ℚ) / (total_days_2007_2008 : ℚ)

theorem maxim_birth_probability: probability_born_in_2008 = 244 / 365 := 
  sorry

end maxim_birth_probability_l210_210251


namespace sector_arc_length_l210_210200

noncomputable def arc_length (R : ℝ) (θ : ℝ) : ℝ :=
  θ / 180 * Real.pi * R

theorem sector_arc_length
  (central_angle : ℝ) (area : ℝ) (arc_length_answer : ℝ)
  (h1 : central_angle = 120)
  (h2 : area = 300 * Real.pi) :
  arc_length_answer = 20 * Real.pi :=
by
  sorry

end sector_arc_length_l210_210200


namespace div_by_72_l210_210843

theorem div_by_72 (x : ℕ) (y : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : x = 4)
    (h3 : 0 ≤ y ∧ y ≤ 9) (h4 : y = 6) : 
    72 ∣ (9834800 + 1000 * x + 10 * y) :=
by 
  sorry

end div_by_72_l210_210843


namespace jacob_calories_l210_210697

theorem jacob_calories (goal : ℕ) (breakfast : ℕ) (lunch : ℕ) (dinner : ℕ) 
  (h_goal : goal = 1800) 
  (h_breakfast : breakfast = 400) 
  (h_lunch : lunch = 900) 
  (h_dinner : dinner = 1100) : 
  (breakfast + lunch + dinner) - goal = 600 :=
by 
  sorry

end jacob_calories_l210_210697


namespace greatest_integer_gcd_6_l210_210572

theorem greatest_integer_gcd_6 (n : ℕ) (h₁ : n < 200) (h₂ : Nat.gcd n 18 = 6) : n = 192 :=
by
  sorry

end greatest_integer_gcd_6_l210_210572


namespace no_intersection_abs_value_graphs_l210_210211

theorem no_intersection_abs_value_graphs : 
  ∀ (x : ℝ), ¬ (|3 * x + 6| = -|4 * x - 1|) :=
by
  intro x
  sorry

end no_intersection_abs_value_graphs_l210_210211


namespace total_hangers_l210_210062

theorem total_hangers (pink green blue yellow orange purple red : ℕ) 
  (h_pink : pink = 7)
  (h_green : green = 4)
  (h_blue : blue = green - 1)
  (h_yellow : yellow = blue - 1)
  (h_orange : orange = 2 * pink)
  (h_purple : purple = yellow + 3)
  (h_red : red = purple / 2) :
  pink + green + blue + yellow + orange + purple + red = 37 :=
sorry

end total_hangers_l210_210062


namespace part1_tangent_circles_part2_chords_l210_210201

theorem part1_tangent_circles (t : ℝ) : 
  t = 1 → 
  ∃ (a b : ℝ), 
    (x + 1)^2 + y^2 = 1 ∨ 
    (x + (2/5))^2 + (y - (9/5))^2 = (1 : ℝ) :=
by
  sorry

theorem part2_chords (t : ℝ) : 
  (∀ (k1 k2 : ℝ), 
    k1 + k2 = -3 * t / 4 ∧ 
    k1 * k2 = (t^2 - 1) / 8 ∧ 
    |k1 - k2| = 3 / 4) → 
    t = 1 ∨ t = -1 :=
by
  sorry

end part1_tangent_circles_part2_chords_l210_210201


namespace solution_to_equation_l210_210345

-- Define the given conditions
noncomputable def given_equation (x : ℝ) : Prop := 
  real.cbrt (3 - x) + real.sqrt (x - 1) = 2

-- Prove that x = 2 satisfies the given conditions
theorem solution_to_equation : given_equation 2 :=
  sorry -- proof omitted

end solution_to_equation_l210_210345


namespace correct_calculation_l210_210301

theorem correct_calculation (a b : ℝ) : (-3 * a^3 * b)^2 = 9 * a^6 * b^2 := 
by sorry

end correct_calculation_l210_210301


namespace find_value_of_A_l210_210366

theorem find_value_of_A (x : ℝ) (h₁ : x - 3 * (x - 2) ≥ 2) (h₂ : 4 * x - 2 < 5 * x - 1) (h₃ : x ≠ 1) (h₄ : x ≠ -1) (h₅ : x ≠ 0) (hx : x = 2) :
  let A := (3 * x / (x - 1) - x / (x + 1)) / (x / (x^2 - 1))
  A = 8 :=
by
  -- Proof will be filled in
  sorry

end find_value_of_A_l210_210366


namespace Maxim_born_in_2008_probability_l210_210249

def birth_interval_start := ⟨2007, 9, 2⟩ -- September 2, 2007
def birth_interval_end := ⟨2008, 8, 31⟩ -- August 31, 2008
def days_in_interval := 365
def days_in_2008 := 244

theorem Maxim_born_in_2008_probability :
  (days_in_2008 : ℝ) / (days_in_interval : ℝ) = 244 / 365 :=
sorry

end Maxim_born_in_2008_probability_l210_210249


namespace RachelFurnitureAssemblyTime_l210_210094

/-- Rachel bought seven new chairs and three new tables for her house.
    She spent four minutes on each piece of furniture putting it together.
    Prove that it took her 40 minutes to finish putting together all the furniture. -/
theorem RachelFurnitureAssemblyTime :
  let chairs := 7
  let tables := 3
  let time_per_piece := 4
  let total_time := (chairs + tables) * time_per_piece
  total_time = 40 := by
    sorry

end RachelFurnitureAssemblyTime_l210_210094


namespace solve_fraction_eq_l210_210267

theorem solve_fraction_eq :
  ∀ x : ℝ, (x - 3 ≠ 0) → ((x + 6) / (x - 3) = 4) → x = 6 :=
by
  intros x h_nonzero h_eq
  sorry

end solve_fraction_eq_l210_210267


namespace factor_81_minus_27_x_cubed_l210_210183

theorem factor_81_minus_27_x_cubed (x : ℝ) : 
  81 - 27 * x ^ 3 = 27 * (3 - x) * (9 + 3 * x + x ^ 2) :=
by sorry

end factor_81_minus_27_x_cubed_l210_210183


namespace no_two_items_share_color_l210_210674

theorem no_two_items_share_color (shirts pants hats : Fin 5) :
  ∃ num_outfits : ℕ, num_outfits = 60 :=
by
  sorry

end no_two_items_share_color_l210_210674


namespace find_atomic_weight_of_Na_l210_210188

def atomic_weight_of_Na_is_correct : Prop :=
  ∃ (atomic_weight_of_Na : ℝ),
    (atomic_weight_of_Na + 35.45 + 16.00 = 74) ∧ (atomic_weight_of_Na = 22.55)

theorem find_atomic_weight_of_Na : atomic_weight_of_Na_is_correct :=
by
  sorry

end find_atomic_weight_of_Na_l210_210188


namespace find_g_of_5_l210_210418

theorem find_g_of_5 (g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, g (x * y) = g x * g y) 
  (h2 : g 1 = 2) : 
  g 5 = 32 := 
by 
  sorry

end find_g_of_5_l210_210418


namespace max_visible_cubes_from_point_l210_210336

theorem max_visible_cubes_from_point (n : ℕ) (h : n = 12) :
  let total_cubes := n^3
  let face_cube_count := n * n
  let edge_count := n
  let visible_face_count := 3 * face_cube_count
  let double_counted_edges := 3 * (edge_count - 1)
  let corner_cube_count := 1
  visible_face_count - double_counted_edges + corner_cube_count = 400 := by
  sorry

end max_visible_cubes_from_point_l210_210336


namespace average_of_N_l210_210290

theorem average_of_N (N : ℤ) (h1 : (1:ℚ)/3 < N/90) (h2 : N/90 < (2:ℚ)/5) : 31 ≤ N ∧ N ≤ 35 → (N = 31 ∨ N = 32 ∨ N = 33 ∨ N = 34 ∨ N = 35) → (31 + 32 + 33 + 34 + 35) / 5 = 33 := by
  sorry

end average_of_N_l210_210290


namespace ratio_of_mustang_models_length_l210_210863

theorem ratio_of_mustang_models_length :
  ∀ (full_size_length mid_size_length smallest_model_length : ℕ),
    full_size_length = 240 →
    mid_size_length = full_size_length / 10 →
    smallest_model_length = 12 →
    smallest_model_length / mid_size_length = 1/2 :=
by
  intros full_size_length mid_size_length smallest_model_length h1 h2 h3
  sorry

end ratio_of_mustang_models_length_l210_210863


namespace geometric_sequence_sum_l210_210899

noncomputable theory

-- Define the conditions and the problem statement
theorem geometric_sequence_sum (a r : ℝ) (h₁ : a * (1 - r^3000) / (1 - r) = 500) 
  (h₂ : a * (1 - r^6000) / (1 - r) = 950) : 
  a * (1 - r^9000) / (1 - r) = 1355 := 
sorry

end geometric_sequence_sum_l210_210899


namespace number_of_M_subsets_l210_210491

def P : Set ℤ := {0, 1, 2}
def Q : Set ℤ := {0, 2, 4}

theorem number_of_M_subsets (M : Set ℤ) (hP : M ⊆ P) (hQ : M ⊆ Q) : 
  ∃ n : ℕ, n = 4 :=
by
  sorry

end number_of_M_subsets_l210_210491


namespace liam_markers_liam_first_markers_over_500_l210_210859

def seq (n : ℕ) : ℕ := 5 * 3^n

theorem liam_markers (n : ℕ) (h1 : seq 0 = 5) (h2 : seq 1 = 10) (h3 : ∀ k < n, 5 * 3^k ≤ 500) : 
  seq n > 500 := by sorry

theorem liam_first_markers_over_500 (h1 : seq 0 = 5) (h2 : seq 1 = 10) :
  ∃ n, seq n > 500 ∧ ∀ k < n, seq k ≤ 500 := by sorry

end liam_markers_liam_first_markers_over_500_l210_210859


namespace intersection_sums_l210_210733

def parabola1 (x : ℝ) : ℝ := (x - 2)^2
def parabola2 (y : ℝ) : ℝ := (y - 2)^2 - 6

theorem intersection_sums (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ) 
  (h1 : y1 = parabola1 x1) (h2 : y2 = parabola1 x2)
  (h3 : y3 = parabola1 x3) (h4 : y4 = parabola1 x4)
  (k1 : x1 + 6 = y1^2 - 4*y1 + 4) (k2 : x2 + 6 = y2^2 - 4*y2 + 4)
  (k3 : x3 + 6 = y3^2 - 4*y3 + 4) (k4 : x4 + 6 = y4^2 - 4*y4 + 4) :
  x1 + x2 + x3 + x4 + y1 + y2 + y3 + y4 = 16 := 
sorry

end intersection_sums_l210_210733


namespace cannot_determine_remaining_pictures_l210_210586

theorem cannot_determine_remaining_pictures (taken_pics : ℕ) (dolphin_show_pics : ℕ) (total_pics : ℕ) :
  taken_pics = 28 → dolphin_show_pics = 16 → total_pics = 44 → 
  (∀ capacity : ℕ, ¬ (total_pics + x = capacity)) → 
  ¬ ∃ remaining_pics : ℕ, remaining_pics = capacity - total_pics :=
by {
  sorry
}

end cannot_determine_remaining_pictures_l210_210586


namespace transformation_1_transformation_2_l210_210072

theorem transformation_1 (x y x' y' : ℝ) 
  (h1 : x' = x / 2) 
  (h2 : y' = y / 3) 
  (eq1 : 5 * x + 2 * y = 0) : 
  5 * x' + 3 * y' = 0 := 
sorry

theorem transformation_2 (x y x' y' : ℝ) 
  (h1 : x' = x / 2) 
  (h2 : y' = y / 3) 
  (eq2 : x^2 + y^2 = 1) : 
  4 * x' ^ 2 + 9 * y' ^ 2 = 1 := 
sorry

end transformation_1_transformation_2_l210_210072


namespace profit_relationship_profit_range_max_profit_l210_210727

noncomputable def profit (x : ℝ) : ℝ :=
  -20 * x ^ 2 + 100 * x + 6000

theorem profit_relationship (x : ℝ) :
  profit (x) = (60 - x) * (300 + 20 * x) - 40 * (300 + 20 * x) :=
by
  sorry
  
theorem profit_range (x : ℝ) (h : 0 ≤ x ∧ x < 20) : 
  0 ≤ profit (x) :=
by
  sorry

theorem max_profit (x : ℝ) :
  (2.5 ≤ x ∧ x < 2.6) → profit (x) ≤ 6125 := 
by
  sorry  

end profit_relationship_profit_range_max_profit_l210_210727


namespace min_max_f_l210_210012

theorem min_max_f (a b x y z t : ℝ) (ha : 0 < a) (hb : 0 < b)
  (hxz : x + z = 1) (hyt : y + t = 1) (hx : 0 ≤ x) (hy : 0 ≤ y)
  (hz : 0 ≤ z) (ht : 0 ≤ t) :
  1 ≤ ( (a * x^2 + b * y^2) / (a * x + b * y) + (a * z^2 + b * t^2) / (a * z + b * t) ) ∧
  ( (a * x^2 + b * y^2) / (a * x + b * y) + (a * z^2 + b * t^2) / (a * z + b * t) ) ≤ 2 :=
sorry

end min_max_f_l210_210012


namespace trigonometric_comparison_l210_210240

theorem trigonometric_comparison :
  let a := Real.sin (33 * Real.pi / 180)
  let b := Real.cos (58 * Real.pi / 180)
  let c := Real.tan (34 * Real.pi / 180)
  c > a ∧ a > b := by
    let a := Real.sin (33 * Real.pi / 180)
    let b := Real.cos (58 * Real.pi / 180)
    let c := Real.tan (34 * Real.pi / 180)
    have hb : b = Real.sin (32 * Real.pi / 180) := by
      rw [Real.cos_eq_sin_pi_div_two_sub]
      norm_num
    
    have hab : a > b := by
      rw [hb]
      exact Real.sin_monotone (lt_add_of_pos_right _ (by norm_num))
    
    have hac : c > a := by
      have hc : c = Real.sin (34 * Real.pi / 180) / Real.cos (34 * Real.pi / 180) := by
        rfl
      have hsc : Real.sin (34 * Real.pi / 180) > Real.sin (33 * Real.pi / 180) := 
        Real.sin_monotone (lt_add_of_pos_right _ (by norm_num))
      have hcc : 0 < Real.cos (34 * Real.pi / 180) := by
        exact Real.cos_pos_of_mem_Ioo (by norm_num; linarith [pi_pos])
      rw [hc]
      exact (div_lt_one hcc).mpr hsc
    
    exact ⟨hac, hab⟩ := sorry

end trigonometric_comparison_l210_210240


namespace value_of_f_at_2_l210_210371

def f (x : ℝ) : ℝ :=
  x^3 - x - 1

theorem value_of_f_at_2 : f 2 = 5 := by
  -- Proof goes here
  sorry

end value_of_f_at_2_l210_210371


namespace common_pts_above_curve_l210_210853

open Real

theorem common_pts_above_curve {x y t : ℝ} (h1 : 0 ≤ x ∧ x ≤ 1) (h2 : 0 ≤ y ∧ y ≤ 1) (h3 : 0 < t ∧ t < 1) :
  (∀ t, y ≥ (t-1)/t * x + 1 - t) ↔ (sqrt x + sqrt y ≥ 1) := 
by
  sorry

end common_pts_above_curve_l210_210853


namespace frequency_group_5_l210_210599

theorem frequency_group_5 (total_students : ℕ) (freq1 freq2 freq3 freq4 : ℕ)
  (h1 : total_students = 45)
  (h2 : freq1 = 12)
  (h3 : freq2 = 11)
  (h4 : freq3 = 9)
  (h5 : freq4 = 4) :
  ((total_students - (freq1 + freq2 + freq3 + freq4)) / total_students : ℚ) = 0.2 := 
sorry

end frequency_group_5_l210_210599


namespace greatest_integer_less_than_200_with_gcd_18_l210_210577

theorem greatest_integer_less_than_200_with_gcd_18 : ∃ n: ℕ, n < 200 ∧ Nat.gcd n 18 = 6 ∧ ∀ m: ℕ, m < 200 ∧ Nat.gcd m 18 = 6 → m ≤ n := 
by
  sorry

end greatest_integer_less_than_200_with_gcd_18_l210_210577


namespace combine_monomials_x_plus_y_l210_210975

theorem combine_monomials_x_plus_y : ∀ (x y : ℤ),
  7 * x = 2 - 4 * y →
  y + 7 = 2 * x →
  x + y = -1 :=
by
  intros x y h1 h2
  sorry

end combine_monomials_x_plus_y_l210_210975


namespace find_mini_cupcakes_l210_210735

-- Definitions of the conditions
def number_of_donut_holes := 12
def number_of_students := 13
def desserts_per_student := 2

-- Statement of the theorem to prove the number of mini-cupcakes is 14
theorem find_mini_cupcakes :
  let D := number_of_donut_holes
  let N := number_of_students
  let total_desserts := N * desserts_per_student
  let C := total_desserts - D
  C = 14 :=
by
  sorry

end find_mini_cupcakes_l210_210735


namespace find_q_minus_p_values_l210_210553

theorem find_q_minus_p_values (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn : 0 < n) 
    (h : (p * (q + 1) + q * (p + 1)) * (n + 2) = 2 * n * p * q) : 
    q - p = 2 ∨ q - p = 3 ∨ q - p = 5 :=
sorry

end find_q_minus_p_values_l210_210553


namespace ratio_of_female_democrats_l210_210906

theorem ratio_of_female_democrats 
    (M F : ℕ) 
    (H1 : M + F = 990)
    (H2 : M / 4 + 165 = 330) 
    (H3 : 165 = 165) : 
    165 / F = 1 / 2 := 
sorry

end ratio_of_female_democrats_l210_210906


namespace total_profit_correct_l210_210540

def natasha_money : ℤ := 60
def carla_money : ℤ := natasha_money / 3
def cosima_money : ℤ := carla_money / 2
def sergio_money : ℤ := (3 * cosima_money) / 2

def natasha_items : ℤ := 4
def carla_items : ℤ := 6
def cosima_items : ℤ := 5
def sergio_items : ℤ := 3

def natasha_profit_margin : ℚ := 0.10
def carla_profit_margin : ℚ := 0.15
def cosima_sergio_profit_margin : ℚ := 0.12

def natasha_item_cost : ℚ := (natasha_money : ℚ) / natasha_items
def carla_item_cost : ℚ := (carla_money : ℚ) / carla_items
def cosima_item_cost : ℚ := (cosima_money : ℚ) / cosima_items
def sergio_item_cost : ℚ := (sergio_money : ℚ) / sergio_items

def natasha_profit : ℚ := natasha_items * natasha_item_cost * natasha_profit_margin
def carla_profit : ℚ := carla_items * carla_item_cost * carla_profit_margin
def cosima_profit : ℚ := cosima_items * cosima_item_cost * cosima_sergio_profit_margin
def sergio_profit : ℚ := sergio_items * sergio_item_cost * cosima_sergio_profit_margin

def total_profit : ℚ := natasha_profit + carla_profit + cosima_profit + sergio_profit

theorem total_profit_correct : total_profit = 11.99 := 
by sorry

end total_profit_correct_l210_210540


namespace correct_operation_l210_210433

theorem correct_operation :
  ¬ ( (-3 : ℤ) * x ^ 2 * y ) ^ 3 = -9 * (x ^ 6) * y ^ 3 ∧
  ¬ (a + b) * (a + b) = (a ^ 2 + b ^ 2) ∧
  (4 * x ^ 3 * y ^ 2) * (x ^ 2 * y ^ 3) = (4 * x ^ 5 * y ^ 5) ∧
  ¬ ((-a) + b) * (a - b) = (a ^ 2 - b ^ 2) :=
by
  sorry

end correct_operation_l210_210433


namespace Ali_winning_strategy_l210_210085

def Ali_and_Mohammad_game (m n : ℕ) (a : Fin m → ℕ) : Prop :=
∃ (k l : ℕ), k > 0 ∧ l > 0 ∧ (∃ p : ℕ, Nat.Prime p ∧ m = p^k ∧ n = p^l)

theorem Ali_winning_strategy (m n : ℕ) (a : Fin m → ℕ) :
  Ali_and_Mohammad_game m n a :=
sorry

end Ali_winning_strategy_l210_210085


namespace power_mod_remainder_l210_210743

theorem power_mod_remainder (a : ℕ) (n : ℕ) (h1 : 3^5 % 11 = 1) (h2 : 221 % 5 = 1) : 3^221 % 11 = 3 :=
by
  sorry

end power_mod_remainder_l210_210743


namespace find_area_of_triangle_formed_by_centers_l210_210264

-- Define the problem conditions
def isosceles_right_triangle (a b c : ℝ) : Prop := 
  a = b ∧ c = a * Real.sqrt 2

def centers_of_squares (a : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  let d := (a / 2, a / 2)
  let e := (a + a / 2, a / 2)
  let f := (a * (1 + Real.sqrt 2) / 2, a * (1 + Real.sqrt 2) / 2)
  (d, e, f)

def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := 
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  0.5 * Real.abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- The main theorem to prove
theorem find_area_of_triangle_formed_by_centers (a b c : ℝ) (h : isosceles_right_triangle a b c) : 
  triangle_area (centers_of_squares a).1 (centers_of_squares a).2.1 (centers_of_squares a).2.2 = c^2 / 2 := 
sorry

end find_area_of_triangle_formed_by_centers_l210_210264


namespace num_students_earning_B_l210_210064

variables (nA nB nC nF : ℕ)

-- Conditions from the problem
def condition1 := nA = 6 * nB / 10
def condition2 := nC = 15 * nB / 10
def condition3 := nF = 4 * nB / 10
def condition4 := nA + nB + nC + nF = 50

-- The theorem to prove
theorem num_students_earning_B (nA nB nC nF : ℕ) : 
  condition1 nA nB → 
  condition2 nC nB → 
  condition3 nF nB → 
  condition4 nA nB nC nF → 
  nB = 14 :=
by
  sorry

end num_students_earning_B_l210_210064


namespace find_slope_l210_210468

theorem find_slope (k b x y y2 : ℝ) (h1 : y = k * x + b) (h2 : y2 = k * (x + 3) + b) (h3 : y2 - y = -2) : k = -2 / 3 := by
  sorry

end find_slope_l210_210468


namespace trees_died_in_typhoon_l210_210821

theorem trees_died_in_typhoon :
  ∀ (original_trees left_trees died_trees : ℕ), 
  original_trees = 20 → 
  left_trees = 4 → 
  died_trees = original_trees - left_trees → 
  died_trees = 16 :=
by
  intros original_trees left_trees died_trees h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end trees_died_in_typhoon_l210_210821


namespace sum_zero_of_cubic_identity_l210_210380

theorem sum_zero_of_cubic_identity (a b c : ℝ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c) (h4 : a^3 + b^3 + c^3 = 3 * a * b * c) : 
  a + b + c = 0 :=
by
  sorry

end sum_zero_of_cubic_identity_l210_210380


namespace sum_of_geometric_sequence_first_9000_terms_l210_210883

noncomputable def geomSum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

theorem sum_of_geometric_sequence_first_9000_terms (a r : ℝ) (h1 : geomSum a r 3000 = 500) (h2 : geomSum a r 6000 = 950) :
  geomSum a r 9000 = 1355 :=
by
  sorry

end sum_of_geometric_sequence_first_9000_terms_l210_210883


namespace relationship_between_a_b_c_l210_210838

noncomputable def a := 33
noncomputable def b := 5 * 6^1 + 2 * 6^0
noncomputable def c := 1 * 2^4 + 1 * 2^3 + 1 * 2^2 + 1 * 2^1 + 1 * 2^0

theorem relationship_between_a_b_c : a > b ∧ b > c := by
  sorry

end relationship_between_a_b_c_l210_210838


namespace student_count_l210_210440

open Nat

theorem student_count :
  ∃ n : ℕ, n < 60 ∧ n % 8 = 5 ∧ n % 6 = 2 ∧ n = 53 :=
by {
  -- placeholder for the proof
  sorry
}

end student_count_l210_210440


namespace quadrilateral_area_l210_210693

-- Define the angles in the quadrilateral ABCD
def ABD : ℝ := 20
def DBC : ℝ := 60
def ADB : ℝ := 30
def BDC : ℝ := 70

-- Define the side lengths
variables (AB CD AD BC AC BD : ℝ)

-- Prove that the area of the quadrilateral ABCD is half the product of its sides
theorem quadrilateral_area (h1 : ABD = 20) (h2 : DBC = 60) (h3 : ADB = 30) (h4 : BDC = 70)
  : (1 / 2) * (AB * CD + AD * BC) = (1 / 2) * (AB * CD + AD * BC) :=
by
  sorry

end quadrilateral_area_l210_210693


namespace area_of_EFCD_l210_210389

noncomputable def area_of_quadrilateral (AB CD altitude: ℝ) :=
  let sum_bases_half := (AB + CD) / 2
  let small_altitude := altitude / 2
  small_altitude * (sum_bases_half + CD) / 2

theorem area_of_EFCD
  (AB CD altitude : ℝ)
  (AB_len : AB = 10)
  (CD_len : CD = 24)
  (altitude_len : altitude = 15)
  : area_of_quadrilateral AB CD altitude = 153.75 :=
by
  rw [AB_len, CD_len, altitude_len]
  simp [area_of_quadrilateral]
  sorry

end area_of_EFCD_l210_210389


namespace prime_factors_of_69_l210_210109

theorem prime_factors_of_69 
  (prime : ℕ → Prop)
  (is_prime : ∀ n, prime n ↔ n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 11 ∨ 
                        n = 13 ∨ n = 17 ∨ n = 19 ∨ n = 23)
  (x y : ℕ)
  (h1 : 15 < 69)
  (h2 : 69 < 70)
  (h3 : prime y)
  (h4 : 13 < y)
  (h5 : y < 25)
  (h6 : 69 = x * y)
  : prime x ∧ x = 3 := 
sorry

end prime_factors_of_69_l210_210109


namespace average_mpg_highway_l210_210324

variable (mpg_city : ℝ) (H mpg : ℝ) (gallons : ℝ) (max_distance : ℝ)

noncomputable def SUV_fuel_efficiency : Prop :=
  mpg_city  = 7.6 ∧
  gallons = 20 ∧
  max_distance = 244 ∧
  H * gallons = max_distance

theorem average_mpg_highway (h1 : mpg_city = 7.6) (h2 : gallons = 20) (h3 : max_distance = 244) :
  SUV_fuel_efficiency mpg_city H gallons max_distance → H = 12.2 :=
by
  intros h
  cases h
  sorry

end average_mpg_highway_l210_210324


namespace problem_solution_l210_210354

def p1 : Prop := ∃ x : ℝ, x^2 + x + 1 < 0
def p2 : Prop := ∀ x : ℝ, -1 ≤ x ∧ x ≤ 2 → x^2 - 1 ≥ 0

theorem problem_solution : (¬ p1) ∨ (¬ p2) :=
by
  sorry

end problem_solution_l210_210354


namespace ferry_time_difference_l210_210647

theorem ferry_time_difference :
  ∃ (t : ℕ), (∀ (dP : ℕ) (sP : ℕ) (sQ : ℕ), dP = sP * 3 →
   dP = 24 →
   sP = 8 →
   sQ = sP + 1 →
   t = (dP * 3) / sQ - 3) ∧ t = 5 := 
  sorry

end ferry_time_difference_l210_210647


namespace total_pieces_correct_l210_210175

-- Definitions based on conditions
def rods_in_row (n : ℕ) : ℕ := 3 * n
def connectors_in_row (n : ℕ) : ℕ := n

-- Sum of natural numbers
def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Total rods in ten rows
def total_rods : ℕ := 3 * sum_first_n 10

-- Total connectors in eleven rows
def total_connectors : ℕ := sum_first_n 11

-- Total pieces
def total_pieces : ℕ := total_rods + total_connectors

-- Theorem to prove
theorem total_pieces_correct : total_pieces = 231 :=
by
  sorry

end total_pieces_correct_l210_210175


namespace intersection_M_N_l210_210478

def M := {p : ℝ × ℝ | p.snd = 2 - p.fst}
def N := {p : ℝ × ℝ | p.fst - p.snd = 4}
def intersection := {p : ℝ × ℝ | p = (3, -1)}

theorem intersection_M_N : M ∩ N = intersection := 
by sorry

end intersection_M_N_l210_210478


namespace numbers_represented_3_units_from_A_l210_210411

theorem numbers_represented_3_units_from_A (A : ℝ) (x : ℝ) (h : A = -2) : 
  abs (x + 2) = 3 ↔ x = 1 ∨ x = -5 := by
  sorry

end numbers_represented_3_units_from_A_l210_210411


namespace geometric_sequence_sum_l210_210897

noncomputable theory

-- Define the conditions and the problem statement
theorem geometric_sequence_sum (a r : ℝ) (h₁ : a * (1 - r^3000) / (1 - r) = 500) 
  (h₂ : a * (1 - r^6000) / (1 - r) = 950) : 
  a * (1 - r^9000) / (1 - r) = 1355 := 
sorry

end geometric_sequence_sum_l210_210897


namespace connie_marbles_l210_210453

theorem connie_marbles (j c : ℕ) (h1 : j = 498) (h2 : j = c + 175) : c = 323 :=
by
  -- Placeholder for the proof
  sorry

end connie_marbles_l210_210453


namespace distinguishable_arrangements_l210_210368

-- Define number of each type of tiles
def brown_tiles := 2
def purple_tiles := 1
def green_tiles := 3
def yellow_tiles := 2
def total_tiles := brown_tiles + purple_tiles + green_tiles + yellow_tiles

theorem distinguishable_arrangements :
  (Nat.factorial total_tiles) / 
  ((Nat.factorial green_tiles) * 
   (Nat.factorial brown_tiles) * 
   (Nat.factorial yellow_tiles) * 
   (Nat.factorial purple_tiles)) = 1680 := by
  sorry

end distinguishable_arrangements_l210_210368


namespace distance_along_stream_l210_210850
-- Define the problem in Lean 4

noncomputable def speed_boat_still : ℝ := 11   -- Speed of the boat in still water
noncomputable def distance_against_stream : ℝ := 9  -- Distance traveled against the stream in one hour

theorem distance_along_stream : 
  ∃ (v_s : ℝ), (speed_boat_still - v_s = distance_against_stream) ∧ (11 + v_s) * 1 = 13 := 
by
  use 2
  sorry

end distance_along_stream_l210_210850


namespace distinct_exponentiation_values_l210_210009

theorem distinct_exponentiation_values : 
  let a := 3^(3^(3^3))
  let b := 3^((3^3)^3)
  let c := ((3^3)^3)^3
  let d := 3^((3^3)^(3^2))
  (a ≠ b) → (a ≠ c) → (a ≠ d) → (b ≠ c) → (b ≠ d) → (c ≠ d) → 
  ∃ n, n = 3 := 
sorry

end distinct_exponentiation_values_l210_210009


namespace systematic_sampling_first_group_l210_210429

theorem systematic_sampling_first_group (S : ℕ) (n : ℕ) (students_per_group : ℕ) (group_number : ℕ)
(h1 : n = 160)
(h2 : students_per_group = 8)
(h3 : group_number = 16)
(h4 : S + (group_number - 1) * students_per_group = 126)
: S = 6 := by
  sorry

end systematic_sampling_first_group_l210_210429


namespace cost_per_game_l210_210326

theorem cost_per_game 
  (x : ℝ)
  (shoe_rent : ℝ := 0.50)
  (total_money : ℝ := 12.80)
  (games : ℕ := 7)
  (h1 : total_money - shoe_rent = 12.30)
  (h2 : 7 * x = 12.30) :
  x = 1.76 := 
sorry

end cost_per_game_l210_210326


namespace minimum_value_of_m_plus_n_l210_210218

-- Define the conditions and goals as a Lean 4 statement with a proof goal.
theorem minimum_value_of_m_plus_n (m n : ℝ) (h : m * n > 0) (hA : m + n = 3 * m * n) : m + n = 4 / 3 :=
sorry

end minimum_value_of_m_plus_n_l210_210218


namespace combination_8_5_is_56_l210_210508

theorem combination_8_5_is_56 : nat.choose 8 5 = 56 :=
by sorry

end combination_8_5_is_56_l210_210508


namespace time_for_c_l210_210137

theorem time_for_c (a b work_completion: ℝ) (ha : a = 16) (hb : b = 6) (habc : work_completion = 3.2) : 
  (12 : ℝ) = 
  (48 * work_completion - 48) / 4 := 
sorry

end time_for_c_l210_210137


namespace shaded_area_correct_l210_210325

-- Define the side lengths of the squares
def side_length_large_square : ℕ := 14
def side_length_small_square : ℕ := 10

-- Define the areas of the squares
def area_large_square : ℕ := side_length_large_square * side_length_large_square
def area_small_square : ℕ := side_length_small_square * side_length_small_square

-- Define the area of the shaded regions
def area_shaded_regions : ℕ := area_large_square - area_small_square

-- State the theorem
theorem shaded_area_correct : area_shaded_regions = 49 := by
  sorry

end shaded_area_correct_l210_210325


namespace sufficient_drivers_and_completion_time_l210_210157

noncomputable def one_way_trip_minutes : ℕ := 2 * 60 + 40
noncomputable def round_trip_minutes : ℕ := 2 * one_way_trip_minutes
noncomputable def rest_period_minutes : ℕ := 60
noncomputable def twelve_forty_pm : ℕ := 12 * 60 + 40 -- in minutes from midnight
noncomputable def one_forty_pm : ℕ := twelve_forty_pm + rest_period_minutes
noncomputable def thirteen_five_pm : ℕ := 13 * 60 + 5 -- 1:05 PM
noncomputable def sixteen_ten_pm : ℕ := 16 * 60 + 10 -- 4:10 PM
noncomputable def sixteen_pm : ℕ := 16 * 60 -- 4:00 PM
noncomputable def seventeen_thirty_pm : ℕ := 17 * 60 + 30 -- 5:30 PM
noncomputable def twenty_one_thirty_pm : ℕ := sixteen_ten_pm + round_trip_minutes -- 9:30 PM (21:30)

theorem sufficient_drivers_and_completion_time :
  4 = 4 ∧ twenty_one_thirty_pm = 21 * 60 + 30 := by
  sorry 

end sufficient_drivers_and_completion_time_l210_210157


namespace sum_real_imag_parts_eq_l210_210057

noncomputable def z (a b : ℂ) : ℂ := a / b

theorem sum_real_imag_parts_eq (z : ℂ) (h : z * (2 + I) = 2 * I - 1) : 
  (z.re + z.im) = 1 / 5 :=
sorry

end sum_real_imag_parts_eq_l210_210057


namespace math_problem_l210_210463

theorem math_problem (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 1 / y) :
  (x - 1 / x) * (y + 1 / y) = x^2 - y^2 :=
by
  sorry

end math_problem_l210_210463


namespace boxes_per_class_l210_210904

variable (boxes : ℕ) (classes : ℕ)

theorem boxes_per_class (h1 : boxes = 3) (h2 : classes = 4) : 
  (boxes : ℚ) / (classes : ℚ) = 3 / 4 :=
by
  rw [h1, h2]
  norm_num

end boxes_per_class_l210_210904


namespace zamena_solution_l210_210785

def digit_assignment (A M E H Z N : ℕ) : Prop :=
  1 ≤ A ∧ A ≤ 5 ∧ 1 ≤ M ∧ M ≤ 5 ∧ 1 ≤ E ∧ E ≤ 5 ∧ 1 ≤ H ∧ H ≤ 5 ∧ 1 ≤ Z ∧ Z ≤ 5 ∧ 1 ≤ N ∧ N ≤ 5 ∧
  A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ A ≠ Z ∧ A ≠ N ∧
  M ≠ E ∧ M ≠ H ∧ M ≠ Z ∧ M ≠ N ∧
  E ≠ H ∧ E ≠ Z ∧ E ≠ N ∧
  H ≠ Z ∧ H ≠ N ∧
  Z ≠ N ∧
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A 

theorem zamena_solution : 
  ∃ (A M E H Z N : ℕ), digit_assignment A M E H Z N ∧ (Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) := 
by
  sorry

end zamena_solution_l210_210785


namespace flower_beds_l210_210091

theorem flower_beds (seeds_per_bed total_seeds flower_beds : ℕ) 
  (h1 : seeds_per_bed = 10) (h2 : total_seeds = 60) : 
  flower_beds = total_seeds / seeds_per_bed := by
  rw [h1, h2]
  sorry

end flower_beds_l210_210091


namespace walnut_trees_initially_in_park_l210_210428

def initial_trees_in_park (final_trees planted_trees : ℕ) : ℕ :=
  final_trees - planted_trees

theorem walnut_trees_initially_in_park (final_trees planted_trees initial_trees : ℕ) 
  (h1 : final_trees = 55) 
  (h2 : planted_trees = 33)
  (h3 : initial_trees = initial_trees_in_park final_trees planted_trees) :
  initial_trees = 22 :=
by
  rw [initial_trees_in_park, h1, h2]
  simp
  exact h3
  sorry

end walnut_trees_initially_in_park_l210_210428


namespace find_c_l210_210855

-- Definitions of r and s
def r (x : ℝ) : ℝ := 4 * x - 9
def s (x : ℝ) (c : ℝ) : ℝ := 5 * x - c

-- Given and proved statement
theorem find_c (c : ℝ) : r (s 2 c) = 11 → c = 5 := 
by 
  sorry

end find_c_l210_210855


namespace complex_root_condition_l210_210534

open Complex

theorem complex_root_condition (u v : ℂ) 
    (h1 : 3 * abs (u + 1) * abs (v + 1) ≥ abs (u * v + 5 * u + 5 * v + 1))
    (h2 : abs (u + v) = abs (u * v + 1)) :
    u = 1 ∨ v = 1 :=
sorry

end complex_root_condition_l210_210534


namespace lucas_1500th_day_is_sunday_l210_210405

def days_in_week : ℕ := 7

def start_day : ℕ := 5  -- 0: Monday, 1: Tuesday, ..., 5: Friday

def nth_day_of_life (n : ℕ) : ℕ :=
  (n - 1 + start_day) % days_in_week

theorem lucas_1500th_day_is_sunday : nth_day_of_life 1500 = 0 :=
by
  sorry

end lucas_1500th_day_is_sunday_l210_210405


namespace multiples_of_7_units_digit_7_l210_210824

theorem multiples_of_7_units_digit_7 (n : ℕ) (h1 : n < 150) (h2 : ∃ (k : ℕ), n = 7 * k) (h3 : n % 10 = 7) : 
    ∃ (m : ℕ), m = 2 := 
by
  sorry

end multiples_of_7_units_digit_7_l210_210824


namespace solve_ZAMENA_l210_210784

noncomputable def ZAMENA : ℕ := 541234

theorem solve_ZAMENA :
  ∃ (A M E H : ℕ), 
    1 ≤ A ∧ A ≤ 5 ∧
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    3 > A ∧ 
    A > M ∧ 
    M < E ∧ 
    E < H ∧ 
    H < A ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ H ∧
    M ≠ E ∧ M ≠ H ∧
    E ≠ H ∧
    ZAMENA = 541234 :=
by {
  have h1 : ∃ (A M E H : ℕ), 
    3 > A ∧ 
    A > M ∧ 
    M < E ∧ 
    E < H ∧ 
    H < A ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ H ∧
    M ≠ E ∧ M ≠ H ∧
    E ≠ H, sorry,
  exact ⟨2, 1, 2, 3, h1⟩,
  repeat { sorry }
}

end solve_ZAMENA_l210_210784


namespace jeans_original_price_l210_210612

theorem jeans_original_price 
  (discount : ℝ -> ℝ)
  (original_price : ℝ)
  (discount_percentage : ℝ)
  (final_price : ℝ) 
  (customer_payment : ℝ) : 
  discount_percentage = 0.10 -> 
  discount x = x * (1 - discount_percentage) -> 
  final_price = discount (2 * original_price) + original_price -> 
  customer_payment = 112 -> 
  final_price = 112 -> 
  original_price = 40 := 
by
  intros
  sorry

end jeans_original_price_l210_210612


namespace rabbits_ate_three_potatoes_l210_210537

variable (initial_potatoes remaining_potatoes eaten_potatoes : ℕ)

-- Definitions from the conditions
def mary_initial_potatoes : initial_potatoes = 8 := sorry
def mary_remaining_potatoes : remaining_potatoes = 5 := sorry

-- The goal to prove
theorem rabbits_ate_three_potatoes :
  initial_potatoes - remaining_potatoes = 3 := sorry

end rabbits_ate_three_potatoes_l210_210537


namespace gcd_547_323_l210_210431

theorem gcd_547_323 : Nat.gcd 547 323 = 1 := 
by
  sorry

end gcd_547_323_l210_210431


namespace club_committee_probability_l210_210600

noncomputable def probability_at_least_two_boys_and_two_girls (total_members boys girls committee_size : ℕ) : ℚ :=
  let total_ways := Nat.choose total_members committee_size
  let ways_fewer_than_two_boys := (Nat.choose girls committee_size) + (boys * Nat.choose girls (committee_size - 1))
  let ways_fewer_than_two_girls := (Nat.choose boys committee_size) + (girls * Nat.choose boys (committee_size - 1))
  let ways_invalid := ways_fewer_than_two_boys + ways_fewer_than_two_girls
  (total_ways - ways_invalid) / total_ways

theorem club_committee_probability :
  probability_at_least_two_boys_and_two_girls 30 12 18 6 = 457215 / 593775 :=
by
  sorry

end club_committee_probability_l210_210600


namespace find_a2_plus_b2_l210_210060

theorem find_a2_plus_b2 (a b : ℝ) :
  (∀ x, |a * Real.sin x + b * Real.cos x - 1| + |b * Real.sin x - a * Real.cos x| ≤ 11)
  → a^2 + b^2 = 50 :=
by
  sorry

end find_a2_plus_b2_l210_210060


namespace cylinder_volume_transformation_l210_210983

theorem cylinder_volume_transformation (π : ℝ) (r h : ℝ) (V : ℝ) (V_new : ℝ)
  (hV : V = π * r^2 * h) (hV_initial : V = 20) : V_new = π * (3 * r)^2 * (4 * h) :=
by
sorry

end cylinder_volume_transformation_l210_210983


namespace sin_double_angle_identity_l210_210960

open Real

theorem sin_double_angle_identity {α : ℝ} (h1 : π / 2 < α ∧ α < π) 
    (h2 : sin (α + π / 6) = 1 / 3) :
  sin (2 * α + π / 3) = -4 * sqrt 2 / 9 := 
by 
  sorry

end sin_double_angle_identity_l210_210960


namespace count_multiples_of_7_with_units_digit_7_less_than_150_l210_210829

-- Definition of the problem's conditions and expected result
theorem count_multiples_of_7_with_units_digit_7_less_than_150 : 
  let multiples_of_7 := { n : Nat // n < 150 ∧ n % 7 = 0 }
  let units_digit_7_multiples := { n : Nat // n < 150 ∧ n % 7 = 0 ∧ n % 10 = 7 }
  List.length (units_digit_7_multiples.members) = 3 :=
by
  sorry

end count_multiples_of_7_with_units_digit_7_less_than_150_l210_210829


namespace sum_of_cubes_three_consecutive_divisible_by_three_l210_210093

theorem sum_of_cubes_three_consecutive_divisible_by_three (n : ℤ) : 
  (n^3 + (n+1)^3 + (n+2)^3) % 3 = 0 := 
by 
  sorry

end sum_of_cubes_three_consecutive_divisible_by_three_l210_210093


namespace smallest_n_for_sum_exceed_10_pow_5_l210_210423

def a₁ : ℕ := 9
def r : ℕ := 10
def S (n : ℕ) : ℕ := 5 * n^2 + 4 * n
def target_sum : ℕ := 10^5

theorem smallest_n_for_sum_exceed_10_pow_5 : 
  ∃ n : ℕ, S n > target_sum ∧ ∀ m < n, ¬(S m > target_sum) := 
sorry

end smallest_n_for_sum_exceed_10_pow_5_l210_210423


namespace subset_relation_l210_210533

def P := {x : ℝ | x < 2}
def Q := {y : ℝ | y < 1}

theorem subset_relation : Q ⊆ P := 
by {
  sorry
}

end subset_relation_l210_210533


namespace amount_on_table_A_l210_210098

-- Definitions based on conditions
variables (A B C : ℝ)
variables (h1 : B = 2 * C)
variables (h2 : C = A + 20)
variables (h3 : A + B + C = 220)

-- Theorem statement
theorem amount_on_table_A : A = 40 :=
by
  -- This is expected to be filled in with the proof steps, but we skip it with 'sorry'
  sorry

end amount_on_table_A_l210_210098


namespace second_number_deduction_l210_210101

theorem second_number_deduction
  (x : ℝ)
  (h1 : (10 * 16 = 10 * x + (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)))
  (h2 : 2.5 + (x+1 - y) + 6.5 + 8.5 + 10.5 + 12.5 + 14.5 + 16.5 + 18.5 + 20.5 = 115)
  : y = 8 :=
by
  -- This is where the proof would go, but we'll leave it as 'sorry' for now.
  sorry

end second_number_deduction_l210_210101


namespace slope_of_asymptotes_is_one_l210_210535

-- Given definitions and axioms
variables (a b c : ℝ)
  (h1 : a > 0)
  (h2 : b > 0)
  (A1 : ℝ × ℝ := (-a, 0))
  (A2 : ℝ × ℝ := (a, 0))
  (B : ℝ × ℝ := (c, b^2 / a))
  (C : ℝ × ℝ := (c, -b^2 / a))
  (Perp : (b^2 / a) / (c + a) * -(b^2 / a) / (c - a) = -1)

-- Proof goal
theorem slope_of_asymptotes_is_one : a = b → (∀ m : ℝ, m = (b / a) ∨ m = -(b / a)) ↔ ∀ m : ℝ, m = 1 ∨ m = -1 :=
by
  sorry

end slope_of_asymptotes_is_one_l210_210535


namespace find_constants_l210_210801

theorem find_constants (t s : ℤ) :
  (∀ x : ℤ, (3 * x^2 - 4 * x + 9) * (5 * x^2 + t * x + s) = 15 * x^4 - 22 * x^3 + (41 + s) * x^2 - 34 * x + 9 * s) →
  t = -2 ∧ s = s :=
by
  intros h
  sorry

end find_constants_l210_210801


namespace people_eat_only_vegetarian_l210_210502

def number_of_people_eat_only_veg (total_veg : ℕ) (both_veg_nonveg : ℕ) : ℕ :=
  total_veg - both_veg_nonveg

theorem people_eat_only_vegetarian
  (total_veg : ℕ) (both_veg_nonveg : ℕ)
  (h1 : total_veg = 28)
  (h2 : both_veg_nonveg = 12)
  : number_of_people_eat_only_veg total_veg both_veg_nonveg = 16 := by
  sorry

end people_eat_only_vegetarian_l210_210502


namespace eating_possible_values_l210_210300

def A : Set ℝ := {-1, 1 / 2, 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x ^ 2 = 1 ∧ a ≥ 0}

-- Full eating relation: B ⊆ A
-- Partial eating relation: (B ∩ A).Nonempty ∧ ¬(B ⊆ A ∨ A ⊆ B)

def is_full_eating (a : ℝ) : Prop := B a ⊆ A
def is_partial_eating (a : ℝ) : Prop :=
  (B a ∩ A).Nonempty ∧ ¬(B a ⊆ A ∨ A ⊆ B a)

theorem eating_possible_values :
  {a : ℝ | is_full_eating a ∨ is_partial_eating a} = {0, 1, 4} :=
by
  sorry

end eating_possible_values_l210_210300


namespace find_c_value_l210_210550

theorem find_c_value (x c : ℝ) (h₁ : 3 * x + 8 = 5) (h₂ : c * x + 15 = 3) : c = 12 :=
by
  -- This is where the proof steps would go, but we will use sorry for now.
  sorry

end find_c_value_l210_210550


namespace change_factor_l210_210869

theorem change_factor (avg1 avg2 : ℝ) (n : ℕ) (h_avg1 : avg1 = 40) (h_n : n = 10) (h_avg2 : avg2 = 80) : avg2 * (n : ℝ) / (avg1 * (n : ℝ)) = 2 :=
by
  sorry

end change_factor_l210_210869


namespace smallest_x_l210_210298

theorem smallest_x :
  ∃ (x : ℕ), x % 4 = 3 ∧ x % 5 = 4 ∧ x % 6 = 5 ∧ ∀ y : ℕ, (y % 4 = 3 ∧ y % 5 = 4 ∧ y % 6 = 5) → y ≥ x := 
sorry

end smallest_x_l210_210298


namespace almond_croissant_price_l210_210484

theorem almond_croissant_price (R : ℝ) (T : ℝ) (W : ℕ) (total_spent : ℝ) (regular_price : ℝ) (weeks_in_year : ℕ) :
  R = 3.50 →
  T = 468 →
  W = 52 →
  (total_spent = 468) →
  (weekly_regular : ℝ) = 52 * 3.50 →
  (almond_total_cost : ℝ) = (total_spent - weekly_regular) →
  (A : ℝ) = (almond_total_cost / 52) →
  A = 5.50 := by
  intros hR hT hW htotal_spent hweekly_regular halmond_total_cost hA
  sorry

end almond_croissant_price_l210_210484


namespace scientific_notation_correct_l210_210940

def num : ℝ := 1040000000

theorem scientific_notation_correct : num = 1.04 * 10^9 := by
  sorry

end scientific_notation_correct_l210_210940


namespace expression_bounds_l210_210703

theorem expression_bounds (p q r s : ℝ) (hp : 0 ≤ p ∧ p ≤ 1) (hq : 0 ≤ q ∧ q ≤ 1) (hr : 0 ≤ r ∧ r ≤ 1) (hs : 0 ≤ s ∧ s ≤ 1) :
  2 * Real.sqrt 2 ≤ (Real.sqrt (p^2 + (1 - q)^2) + Real.sqrt (q^2 + (1 - r)^2) + Real.sqrt (r^2 + (1 - s)^2) + Real.sqrt (s^2 + (1 - p)^2)) ∧
  (Real.sqrt (p^2 + (1 - q)^2) + Real.sqrt (q^2 + (1 - r)^2) + Real.sqrt (r^2 + (1 - s)^2) + Real.sqrt (s^2 + (1 - p)^2)) ≤ 4 :=
by
  sorry

end expression_bounds_l210_210703


namespace power_function_through_point_l210_210364

noncomputable def f (x k α : ℝ) : ℝ := k * x ^ α

theorem power_function_through_point (k α : ℝ) (h : f (1/2) k α = Real.sqrt 2) : 
  k + α = 1/2 := 
by 
  sorry

end power_function_through_point_l210_210364


namespace prime_p_sum_of_squares_l210_210655

theorem prime_p_sum_of_squares (p : ℕ) (hp : p.Prime) 
  (h : ∃ (a : ℕ), 2 * p = a^2 + (a + 1)^2 + (a + 2)^2 + (a + 3)^2) : 
  36 ∣ (p - 7) :=
by 
  sorry

end prime_p_sum_of_squares_l210_210655


namespace minimum_tan_product_l210_210993

theorem minimum_tan_product 
  {A B C : ℝ} 
  (h1 : A + B + C = π)
  (h2 : A < π / 2)
  (h3 : B < π / 2)
  (h4 : C < π / 2)
  (h5 : ∃ a b c : ℝ, 
        b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C)
  (h6 : Real.sin A = 1/2):
  ∃ (B C : ℝ), 
  (π / 6 < B) ∧ (B < π / 2) ∧ (π / 6 < C) ∧ (C < π / 2) ∧ 
  tan A * tan B * tan C = (12 + 7 * Real.sqrt 3) / 3 := 
by
  sorry

end minimum_tan_product_l210_210993


namespace find_x_l210_210840

theorem find_x (x : ℕ) (h : 2^x - 2^(x - 2) = 3 * 2^10) : x = 12 :=
by 
  sorry

end find_x_l210_210840


namespace dot_product_parallel_vectors_is_minus_ten_l210_210670

-- Definitions from the conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -4)
def are_parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2) ∨ v = (k * u.1, k * u.2)

theorem dot_product_parallel_vectors_is_minus_ten (x : ℝ) (h : are_parallel vector_a (vector_b x)) : (vector_a.1 * (vector_b x).1 + vector_a.2 * (vector_b x).2) = -10 :=
by
  sorry

end dot_product_parallel_vectors_is_minus_ten_l210_210670


namespace negation_of_existential_l210_210274

theorem negation_of_existential :
  ¬ (∃ x : ℝ, x^2 - 2 * x - 3 < 0) ↔ ∀ x : ℝ, x^2 - 2 * x - 3 ≥ 0 :=
by sorry

end negation_of_existential_l210_210274


namespace factor_polynomial_l210_210186

theorem factor_polynomial : 
  (x : ℝ) → x^4 - 4 * x^2 + 16 = (x^2 - 4 * x + 4) * (x^2 + 2 * x + 4) :=
by
sorry

end factor_polynomial_l210_210186


namespace harry_total_cost_l210_210604

-- Define the price of each type of seed packet
def pumpkin_price : ℝ := 2.50
def tomato_price : ℝ := 1.50
def chili_pepper_price : ℝ := 0.90
def zucchini_price : ℝ := 1.20
def eggplant_price : ℝ := 1.80

-- Define the quantities Harry wants to buy
def pumpkin_qty : ℕ := 4
def tomato_qty : ℕ := 6
def chili_pepper_qty : ℕ := 7
def zucchini_qty : ℕ := 3
def eggplant_qty : ℕ := 5

-- Calculate the total cost
def total_cost : ℝ :=
  pumpkin_qty * pumpkin_price +
  tomato_qty * tomato_price +
  chili_pepper_qty * chili_pepper_price +
  zucchini_qty * zucchini_price +
  eggplant_qty * eggplant_price

-- The proof problem
theorem harry_total_cost : total_cost = 38.90 := by
  sorry

end harry_total_cost_l210_210604


namespace line_perpendicular_to_plane_implies_parallel_l210_210035

-- Definitions for lines and planes in space
axiom Line : Type
axiom Plane : Type

-- Relation of perpendicularity between a line and a plane
axiom perp : Line → Plane → Prop

-- Relation of parallelism between two lines
axiom parallel : Line → Line → Prop

-- The theorem to be proved
theorem line_perpendicular_to_plane_implies_parallel (x y : Line) (z : Plane) :
  perp x z → perp y z → parallel x y :=
by sorry

end line_perpendicular_to_plane_implies_parallel_l210_210035


namespace find_four_digit_numbers_l210_210008

def isFourDigitNumber (n : ℕ) : Prop := (1000 ≤ n) ∧ (n < 10000)

noncomputable def solveABCD (AB CD : ℕ) : ℕ := 100 * AB + CD

theorem find_four_digit_numbers :
  ∀ (AB CD : ℕ),
    isFourDigitNumber (solveABCD AB CD) →
    solveABCD AB CD = AB * CD + AB ^ 2 →
      solveABCD AB CD = 1296 ∨ solveABCD AB CD = 3468 :=
by
  intros AB CD h1 h2
  sorry

end find_four_digit_numbers_l210_210008


namespace truck_tank_percentage_increase_l210_210847

-- Declaration of the initial conditions (as given in the problem)
def service_cost : ℝ := 2.20
def fuel_cost_per_liter : ℝ := 0.70
def num_minivans : ℕ := 4
def num_trucks : ℕ := 2
def total_cost : ℝ := 395.40
def minivan_tank_size : ℝ := 65.0

-- Proof statement with the conditions declared above
theorem truck_tank_percentage_increase :
  ∃ p : ℝ, p = 120 ∧ (minivan_tank_size * (p + 100) / 100 = 143) :=
sorry

end truck_tank_percentage_increase_l210_210847


namespace probability_born_in_2008_l210_210256

/-- Define the relevant dates and days calculation -/
def num_days_between : ℕ :=
  29 + 31 + 30 + 31 + 31 + 29 + 31 + 30 + 31 + 30 + 31 + 31

def num_days_2008 : ℕ :=
  31 + 29 + 31 + 30 + 31 + 30 + 31 + 31

theorem probability_born_in_2008 :
  (num_days_2008 : ℚ) / num_days_between = 244 / 365 :=
by
  -- The proof is omitted for brevity
  sorry

end probability_born_in_2008_l210_210256


namespace ab_value_in_triangle_l210_210230

theorem ab_value_in_triangle (a b c : ℝ) (C : ℝ) (h1 : (a + b)^2 - c^2 = 4) (h2 : C = 60) :
  a * b = 4 / 3 :=
by sorry

end ab_value_in_triangle_l210_210230


namespace smallest_value_l210_210531

noncomputable def smallest_possible_value (a b : ℝ) : ℝ := 2 * a + b

theorem smallest_value (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 ≥ 3 * b) (h4 : b^2 ≥ (8 / 9) * a) :
  smallest_possible_value a b = 5.602 :=
sorry

end smallest_value_l210_210531


namespace leila_total_cakes_l210_210084

def cakes_monday : ℕ := 6
def cakes_friday : ℕ := 9
def cakes_saturday : ℕ := 3 * cakes_monday
def total_cakes : ℕ := cakes_monday + cakes_friday + cakes_saturday

theorem leila_total_cakes : total_cakes = 33 := by
  sorry

end leila_total_cakes_l210_210084


namespace frustum_lateral_surface_area_l210_210765

theorem frustum_lateral_surface_area (r1 r2 h : ℝ) (r1_eq : r1 = 10) (r2_eq : r2 = 4) (h_eq : h = 6) :
  let s := Real.sqrt (h^2 + (r1 - r2)^2)
  let A := Real.pi * (r1 + r2) * s
  A = 84 * Real.pi * Real.sqrt 2 :=
by
  sorry

end frustum_lateral_surface_area_l210_210765


namespace find_y_in_terms_of_x_l210_210214

theorem find_y_in_terms_of_x (p : ℝ) (x y : ℝ) (h1 : x = 1 + 3^p) (h2 : y = 1 + 3^(-p)) : y = x / (x - 1) :=
by
  sorry

end find_y_in_terms_of_x_l210_210214


namespace measure_Z_is_19_6_l210_210231

def measure_angle_X : ℝ := 72
def measure_Y (measure_Z : ℝ) : ℝ := 4 * measure_Z + 10
def angle_sum_condition (measure_Z : ℝ) : Prop :=
  measure_angle_X + (measure_Y measure_Z) + measure_Z = 180

theorem measure_Z_is_19_6 :
  ∃ measure_Z : ℝ, measure_Z = 19.6 ∧ angle_sum_condition measure_Z :=
by
  sorry

end measure_Z_is_19_6_l210_210231


namespace father_age_l210_210441

variable (F S x : ℕ)

-- Conditions
axiom h1 : F + S = 75
axiom h2 : F = 8 * (S - x)
axiom h3 : F - x = S

-- Theorem to prove
theorem father_age : F = 48 :=
sorry

end father_age_l210_210441


namespace smallest_positive_integer_multiple_of_6_and_15_is_30_l210_210642

theorem smallest_positive_integer_multiple_of_6_and_15_is_30 :
  ∃ b : ℕ, b > 0 ∧ (6 ∣ b) ∧ (15 ∣ b) ∧ ∀ n, (n > 0 ∧ (6 ∣ n) ∧ (15 ∣ n)) → b ≤ n :=
  let b := 30 in
  ⟨b, by simp [b, dvd_refl, nat.succ_pos'], sorry⟩

end smallest_positive_integer_multiple_of_6_and_15_is_30_l210_210642


namespace legs_paws_in_pool_l210_210454

def total_legs_paws (num_humans : Nat) (human_legs : Nat) (num_dogs : Nat) (dog_paws : Nat) : Nat :=
  (num_humans * human_legs) + (num_dogs * dog_paws)

theorem legs_paws_in_pool :
  total_legs_paws 2 2 5 4 = 24 := by
  sorry

end legs_paws_in_pool_l210_210454


namespace roots_of_equation_l210_210795

theorem roots_of_equation : ∃ x₁ x₂ : ℝ, (3 ^ x₁ = Real.log (x₁ + 9) / Real.log 3) ∧ 
                                     (3 ^ x₂ = Real.log (x₂ + 9) / Real.log 3) ∧ 
                                     (x₁ < 0) ∧ (x₂ > 0) := 
by {
  sorry
}

end roots_of_equation_l210_210795


namespace sphere_surface_area_l210_210059

theorem sphere_surface_area 
  (a b c : ℝ) 
  (h1 : a = 1)
  (h2 : b = 2)
  (h3 : c = 2)
  (h_spherical_condition : ∃ R : ℝ, ∀ (x y z : ℝ), x^2 + y^2 + z^2 = (2 * R)^2) :
  4 * Real.pi * ((3 / 2)^2) = 9 * Real.pi :=
by
  sorry

end sphere_surface_area_l210_210059


namespace geo_series_sum_eight_terms_l210_210950

theorem geo_series_sum_eight_terms :
  let a_0 := 1 / 3
  let r := 1 / 3 
  let S_8 := a_0 * (1 - r^8) / (1 - r)
  S_8 = 3280 / 6561 :=
by
  /- :: Proof Steps Omitted. -/
  sorry

end geo_series_sum_eight_terms_l210_210950


namespace total_fruit_weight_l210_210723

def melon_weight : Real := 0.35
def berries_weight : Real := 0.48
def grapes_weight : Real := 0.29
def pineapple_weight : Real := 0.56
def oranges_weight : Real := 0.17

theorem total_fruit_weight : melon_weight + berries_weight + grapes_weight + pineapple_weight + oranges_weight = 1.85 :=
by
  unfold melon_weight berries_weight grapes_weight pineapple_weight oranges_weight
  sorry

end total_fruit_weight_l210_210723


namespace difference_in_cans_l210_210672

-- Definitions of the conditions
def total_cans_collected : ℕ := 9
def cans_in_bag : ℕ := 7

-- Statement of the proof problem
theorem difference_in_cans :
  total_cans_collected - cans_in_bag = 2 := by
  sorry

end difference_in_cans_l210_210672


namespace average_visitors_per_day_correct_l210_210750

-- Define the average number of visitors on Sundays
def avg_visitors_sunday : ℕ := 660

-- Define the average number of visitors on other days
def avg_visitors_other : ℕ := 240

-- Define the number of Sundays in a 30-day month starting with a Sunday
def num_sundays_in_month : ℕ := 5

-- Define the number of other days in a 30-day month starting with a Sunday
def num_other_days_in_month : ℕ := 25

-- Calculate the total number of visitors in the month
def total_visitors_in_month : ℕ :=
  (num_sundays_in_month * avg_visitors_sunday) + (num_other_days_in_month * avg_visitors_other)

-- Define the number of days in the month
def days_in_month : ℕ := 30

-- Define the average number of visitors per day
def avg_visitors_per_day := total_visitors_in_month / days_in_month

-- State the theorem to be proved
theorem average_visitors_per_day_correct :
  avg_visitors_per_day = 310 :=
by
  sorry

end average_visitors_per_day_correct_l210_210750


namespace smallest_int_k_for_64_pow_k_l210_210744

theorem smallest_int_k_for_64_pow_k (k : ℕ) (base : ℕ) (h₁ : k = 7) : 
  64^k > base^20 → base = 4 := by
  sorry

end smallest_int_k_for_64_pow_k_l210_210744


namespace q_minus_r_l210_210340

noncomputable def problem (x : ℝ) : Prop :=
  (5 * x - 15) / (x^2 + x - 20) = x + 3

def q_and_r (q r : ℝ) : Prop :=
  q ≠ r ∧ problem q ∧ problem r ∧ q > r

theorem q_minus_r (q r : ℝ) (h : q_and_r q r) : q - r = 2 :=
  sorry

end q_minus_r_l210_210340


namespace four_drivers_suffice_l210_210151

theorem four_drivers_suffice
  (one_way_trip_time : ℕ := 160) -- in minutes
  (round_trip_time : ℕ := 320) -- in minutes
  (rest_time : ℕ := 60) -- in minutes
  (time_A_returns : ℕ := 760) -- 12:40 PM in minutes from midnight
  (time_A_next_start : ℕ := 820) -- 1:40 PM in minutes from midnight
  (time_D_departs : ℕ := 785) -- 1:05 PM in minutes from midnight
  (time_A_fifth_depart : ℕ := 970) -- 4:10 PM in minutes from midnight
  (time_B_returns : ℕ := 960) -- 4:00 PM in minutes from midnight
  (time_B_sixth_depart : ℕ := 1050) -- 5:30 PM in minutes from midnight
  (time_A_fifth_complete: ℕ := 1290) -- 9:30 PM in minutes from midnight
  : 4_drivers_sufficient : ℕ :=
    if time_A_fifth_complete = 1290 then 1 else 0
-- The theorem states that if the calculated trip completion time is 9:30 PM, then 4 drivers are sufficient.
  sorry

end four_drivers_suffice_l210_210151


namespace zamena_correct_l210_210775

variables {M E H A : ℕ}

def valid_digits := {1, 2, 3, 4, 5}

-- Define the conditions as hypotheses
def conditions : Prop :=
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
  M ∈ valid_digits ∧ E ∈ valid_digits ∧ H ∈ valid_digits ∧ A ∈ valid_digits ∧
  ∀ x y, x ∈ {M, E, H, A} ∧ y ∈ {M, E, H, A} → x ≠ y

-- Define the proof problem that these conditions imply the correct answer
theorem zamena_correct : conditions → M = 1 ∧ E = 2 ∧ H = 3 ∧ A = 4 ∧ 10000 * 5 + 1000 * A + 100 * M + 10 * E + H = 541234 :=
by sorry

end zamena_correct_l210_210775


namespace ratio_lena_kevin_after_5_more_l210_210236

variables (L K N : ℕ)

def lena_initial_candy : ℕ := 16
def lena_gets_more : ℕ := 5
def kevin_candy_less_than_nicole : ℕ := 4
def lena_more_than_nicole : ℕ := 5

theorem ratio_lena_kevin_after_5_more
  (lena_initial : L = lena_initial_candy)
  (lena_to_multiple_of_kevin : L + lena_gets_more = K * 3) 
  (kevin_less_than_nicole : K = N - kevin_candy_less_than_nicole)
  (lena_more_than_nicole_condition : L = N + lena_more_than_nicole) :
  (L + lena_gets_more) / K = 3 :=
sorry

end ratio_lena_kevin_after_5_more_l210_210236


namespace johns_total_expenditure_l210_210528

-- Conditions
def treats_first_15_days : ℕ := 3 * 15
def treats_next_15_days : ℕ := 4 * 15
def total_treats : ℕ := treats_first_15_days + treats_next_15_days
def cost_per_treat : ℝ := 0.10
def discount_threshold : ℕ := 50
def discount_rate : ℝ := 0.10

-- Intermediate calculations
def total_cost_without_discount : ℝ := total_treats * cost_per_treat
def discounted_cost_per_treat : ℝ := cost_per_treat * (1 - discount_rate)
def total_cost_with_discount : ℝ := total_treats * discounted_cost_per_treat

-- Main theorem statement
theorem johns_total_expenditure : total_cost_with_discount = 9.45 :=
by
  -- Place proof here
  sorry

end johns_total_expenditure_l210_210528


namespace symmetric_point_yaxis_correct_l210_210522

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def symmetric_yaxis (P : Point3D) : Point3D :=
  { x := -P.x, y := P.y, z := P.z }

theorem symmetric_point_yaxis_correct (P : Point3D) (P' : Point3D) :
  P = {x := 1, y := 2, z := -1} → 
  P' = symmetric_yaxis P → 
  P' = {x := -1, y := 2, z := -1} :=
by
  intros hP hP'
  rw [hP] at hP'
  simp [symmetric_yaxis] at hP'
  exact hP'

end symmetric_point_yaxis_correct_l210_210522


namespace find_m_if_z_is_pure_imaginary_l210_210981

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

theorem find_m_if_z_is_pure_imaginary (m : ℝ) (z : ℂ) (i : ℂ) (h_i_unit : i^2 = -1) (h_z : z = (1 + i) / (1 - i) + m * (1 - i)) :
  is_pure_imaginary z → m = 0 := 
by
  sorry

end find_m_if_z_is_pure_imaginary_l210_210981


namespace domain_of_f_l210_210291

noncomputable def f (x : ℝ) : ℝ := 1 / ((x - 3) + (x - 9))

theorem domain_of_f : 
  {x : ℝ | ¬ ((x - 3) + (x - 9) = 0)} = 
  {x : ℝ | x ≠ 6} := 
by
  sorry

end domain_of_f_l210_210291


namespace find_third_number_l210_210127

theorem find_third_number (N : ℤ) :
  (1274 % 12 = 2) ∧ (1275 % 12 = 3) ∧ (1285 % 12 = 1) ∧ ((1274 * 1275 * N * 1285) % 12 = 6) →
  N % 12 = 1 :=
by
  sorry

end find_third_number_l210_210127


namespace james_has_43_oreos_l210_210077

def james_oreos (jordan : ℕ) : ℕ := 7 + 4 * jordan

theorem james_has_43_oreos (jordan : ℕ) (total : ℕ) (h1 : total = jordan + james_oreos jordan) (h2 : total = 52) : james_oreos jordan = 43 :=
by
  sorry

end james_has_43_oreos_l210_210077


namespace value_of_expression_l210_210131

theorem value_of_expression (x : ℝ) (hx : x = -2) : (3 * x + 4) ^ 2 = 4 :=
by
  sorry

end value_of_expression_l210_210131


namespace bucket_water_l210_210946

theorem bucket_water (oz1 oz2 oz3 oz4 oz5 total1 total2: ℕ) 
  (h1 : oz1 = 11)
  (h2 : oz2 = 13)
  (h3 : oz3 = 12)
  (h4 : oz4 = 16)
  (h5 : oz5 = 10)
  (h_total : total1 = oz1 + oz2 + oz3 + oz4 + oz5)
  (h_second_bucket : total2 = 39)
  : total1 - total2 = 23 :=
sorry

end bucket_water_l210_210946


namespace scientific_notation_of_1040000000_l210_210943

theorem scientific_notation_of_1040000000 : (1.04 * 10^9 = 1040000000) :=
by
  -- Math proof steps can be added here
  sorry

end scientific_notation_of_1040000000_l210_210943


namespace reena_interest_paid_l210_210862

-- Definitions based on conditions
def principal : ℝ := 1200
def rate : ℝ := 0.03
def time : ℝ := 3

-- Definition of simple interest calculation based on conditions
def simple_interest (P R T : ℝ) : ℝ := P * R * T

-- Statement to prove that Reena paid $108 as interest
theorem reena_interest_paid : simple_interest principal rate time = 108 := by
  sorry

end reena_interest_paid_l210_210862


namespace sum_g_squared_l210_210462

noncomputable def g (n : ℕ) : ℝ :=
  ∑' m, if m ≥ 3 then 1 / (m : ℝ)^n else 0

theorem sum_g_squared :
  (∑' n, if n ≥ 3 then (g n)^2 else 0) = 1 / 288 :=
by
  sorry

end sum_g_squared_l210_210462


namespace at_least_two_babies_speak_l210_210844

theorem at_least_two_babies_speak (p : ℚ) (h : p = 1/5) :
  let q := 1 - p in
  let none_speak := q^7 in
  let one_speaks := (7.choose 1) * p * q^6 in
  let at_most_one_speaks := none_speak + one_speaks in
  let result := 1 - at_most_one_speaks in
  result = 50477 / 78125 :=
by
  simp [h]
  sorry

end at_least_two_babies_speak_l210_210844


namespace price_of_lemonade_l210_210323

def costOfIngredients : ℝ := 20
def numberOfCups : ℕ := 50
def desiredProfit : ℝ := 80

theorem price_of_lemonade (price_per_cup : ℝ) :
  (costOfIngredients + desiredProfit) / numberOfCups = price_per_cup → price_per_cup = 2 :=
by
  sorry

end price_of_lemonade_l210_210323


namespace real_solutions_of_equation_l210_210636

theorem real_solutions_of_equation (x : ℝ) :
  (1 / ((x - 1) * (x - 3)) + 1 / ((x - 3) * (x - 5)) + 1 / ((x - 5) * (x - 7)) = 1 / 12) ↔ (x = 13 ∨ x = -5) :=
by
  sorry

end real_solutions_of_equation_l210_210636


namespace range_of_y_l210_210814

theorem range_of_y (m n k y : ℝ)
  (h₁ : 0 ≤ m)
  (h₂ : 0 ≤ n)
  (h₃ : 0 ≤ k)
  (h₄ : m - k + 1 = 1)
  (h₅ : 2 * k + n = 1)
  (h₆ : y = 2 * k^2 - 8 * k + 6)
  : 5 / 2 ≤ y ∧ y ≤ 6 :=
by
  sorry

end range_of_y_l210_210814


namespace relationship_between_a_and_b_l210_210196

theorem relationship_between_a_and_b 
  (a b : ℝ) 
  (h₁ : a > 0)
  (h₂ : b > 0)
  (h₃ : Real.exp a + 2 * a = Real.exp b + 3 * b) : 
  a > b :=
sorry

end relationship_between_a_and_b_l210_210196


namespace only_number_smaller_than_zero_l210_210774

theorem only_number_smaller_than_zero : ∀ (x : ℝ), (x = 5 ∨ x = 2 ∨ x = 0 ∨ x = -Real.sqrt 2) → x < 0 → x = -Real.sqrt 2 :=
by
  intro x hx h
  sorry

end only_number_smaller_than_zero_l210_210774


namespace find_function_l210_210185

theorem find_function (f : ℚ → ℚ) (h1 : f 1 = 2) 
  (h2 : ∀ x y : ℚ, f (x * y) = f x * f y - f (x + y) + 1) :
  ∀ x : ℚ, f x = x + 1 :=
by
  sorry

end find_function_l210_210185


namespace inequality_proof_equality_condition_l210_210715

theorem inequality_proof (a : ℝ) : (a^2 + 5)^2 + 4 * a * (10 - a) ≥ 8 * a^3  :=
by sorry

theorem equality_condition (a : ℝ) : ((a^2 + 5)^2 + 4 * a * (10 - a) = 8 * a^3) ↔ (a = 5 ∨ a = -1) :=
by sorry

end inequality_proof_equality_condition_l210_210715


namespace percentage_calculation_l210_210678

theorem percentage_calculation 
  (number : ℝ)
  (h1 : 0.035 * number = 700) :
  0.024 * (1.5 * number) = 720 := 
by
  sorry

end percentage_calculation_l210_210678


namespace carpenter_material_cost_l210_210759

theorem carpenter_material_cost (total_estimate hourly_rate num_hours : ℝ) 
    (h1 : total_estimate = 980)
    (h2 : hourly_rate = 28)
    (h3 : num_hours = 15) : 
    total_estimate - hourly_rate * num_hours = 560 := 
by
  sorry

end carpenter_material_cost_l210_210759


namespace distance_swim_downstream_correct_l210_210926

def speed_man_still_water : ℝ := 7
def time_taken : ℝ := 5
def distance_upstream : ℝ := 25

lemma distance_swim_downstream (V_m : ℝ) (t : ℝ) (d_up : ℝ) : 
  t * ((V_m + (V_m - d_up / t)) / 2) = 45 :=
by
  have h_speed_upstream : (V_m - (d_up / t)) = d_up / t := by sorry
  have h_speed_stream : (d_up / t) = (V_m - (d_up / t)) := by sorry
  have h_distance_downstream : t * ((V_m + (V_m - (d_up / t)) / 2)) = t * (V_m + (V_m - (V_m - d_up / t))) := by sorry
  sorry

noncomputable def distance_swim_downstream_value : ℝ :=
  9 * 5

theorem distance_swim_downstream_correct :
  distance_swim_downstream_value = 45 :=
by
  sorry

end distance_swim_downstream_correct_l210_210926


namespace remainder_of_prime_powers_l210_210662

theorem remainder_of_prime_powers (p q : ℕ) (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) :
  (p^(q-1) + q^(p-1)) % (p * q) = 1 := 
sorry

end remainder_of_prime_powers_l210_210662


namespace ZAMENA_correct_l210_210780

-- Define the digits and assigning them to the letters
noncomputable def A : ℕ := 2
noncomputable def M : ℕ := 1
noncomputable def E : ℕ := 2
noncomputable def H : ℕ := 3

-- Define the number ZAMENA
noncomputable def ZAMENA : ℕ :=
  5 * 10^5 + A * 10^4 + M * 10^3 + E * 10^2 + H * 10 + A

-- Define the inequalities and constraints
lemma satisfies_inequalities  : 3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A := by
  unfold A M E H
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

-- Define uniqueness of each digit set
lemma unique_digits: A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ M ≠ E ∧ M ≠ H ∧ E ≠ H := by
  unfold A M E H
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

-- Prove that the number ZAMENA matches the specified number
theorem ZAMENA_correct : ZAMENA = 541234 := by
  unfold ZAMENA A M E H
  norm_num

end ZAMENA_correct_l210_210780


namespace max_P_value_l210_210464

noncomputable def P (a : ℝ) : ℝ :=
   ∫ x in 0..a, ∫ y in 0..1, if (Real.sin (π * x))^2 + (Real.sin (π * y))^2 > 1 then 1 else 0

theorem max_P_value : 
   ∃ (a : ℝ) (h : 0 ≤ a ∧ a ≤ 1), P 1 = 2 - Real.sqrt 2 :=
by {
  use 1,
  split,
  { exact le_refl 1 },
  { exact le_of_eq (eq.refl 1) },
  sorry
}

end max_P_value_l210_210464


namespace cost_price_of_article_l210_210589

theorem cost_price_of_article :
  ∃ (C : ℝ), 
  (∃ (G : ℝ), C + G = 500 ∧ C + 1.15 * G = 570) ∧ 
  C = (100 / 3) :=
by sorry

end cost_price_of_article_l210_210589


namespace sum_of_coefficients_l210_210349

theorem sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 a_9 a_10 : ℝ) :
  (∀ x : ℝ, (x^3 - 1) * (x + 1)^7 = a_0 + a_1 * (x + 3) + 
           a_2 * (x + 3)^2 + a_3 * (x + 3)^3 + a_4 * (x + 3)^4 + 
           a_5 * (x + 3)^5 + a_6 * (x + 3)^6 + a_7 * (x + 3)^7 + 
           a_8 * (x + 3)^8 + a_9 * (x + 3)^9 + a_10 * (x + 3)^10) →
  a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 + a_9 + a_10 = 9 := 
by
  -- proof steps skipped
  sorry

end sum_of_coefficients_l210_210349


namespace sum_of_squares_l210_210285

theorem sum_of_squares (x : ℕ) (h : 2 * x = 14) : (3 * x)^2 + (2 * x)^2 + (5 * x)^2 = 1862 := 
by 
  sorry

end sum_of_squares_l210_210285


namespace geometric_sequence_product_l210_210676

variable {a b c : ℝ}

theorem geometric_sequence_product (h : ∃ r : ℝ, r ≠ 0 ∧ -4 = c * r ∧ c = b * r ∧ b = a * r ∧ a = -1 * r) (hb : b < 0) : a * b * c = -8 :=
by
  sorry

end geometric_sequence_product_l210_210676


namespace spotlight_distance_l210_210934

open Real

-- Definitions for the ellipsoid parameters
def ellipsoid_parameters (a b c : ℝ) : Prop :=
  a^2 = b^2 + c^2 ∧ a - c = 1.5

-- Given conditions as input parameters
variables (a b c : ℝ)
variables (h_a : a = 2.7) -- semi-major axis half length
variables (h_c : c = 1.5) -- focal point distance

-- Prove that the distance from F2 to F1 is 12 cm
theorem spotlight_distance (h : ellipsoid_parameters a b c) : 2 * a - (a - c) = 12 :=
by sorry

end spotlight_distance_l210_210934


namespace chad_savings_correct_l210_210169

variable (earnings_mowing : ℝ := 600)
variable (earnings_birthday : ℝ := 250)
variable (earnings_video_games : ℝ := 150)
variable (earnings_odd_jobs : ℝ := 150)
variable (tax_rate : ℝ := 0.10)

noncomputable def total_earnings : ℝ := 
  earnings_mowing + earnings_birthday + earnings_video_games + earnings_odd_jobs

noncomputable def taxes : ℝ := 
  tax_rate * total_earnings

noncomputable def money_after_taxes : ℝ := 
  total_earnings - taxes

noncomputable def savings_mowing : ℝ := 
  0.50 * earnings_mowing

noncomputable def savings_birthday : ℝ := 
  0.30 * earnings_birthday

noncomputable def savings_video_games : ℝ := 
  0.40 * earnings_video_games

noncomputable def savings_odd_jobs : ℝ := 
  0.20 * earnings_odd_jobs

noncomputable def total_savings : ℝ := 
  savings_mowing + savings_birthday + savings_video_games + savings_odd_jobs

theorem chad_savings_correct : total_savings = 465 := by
  sorry

end chad_savings_correct_l210_210169


namespace probability_all_choose_paper_l210_210907

-- Given conditions
def probability_choice_is_paper := 1 / 3

-- The theorem to be proved
theorem probability_all_choose_paper :
  probability_choice_is_paper ^ 3 = 1 / 27 :=
sorry

end probability_all_choose_paper_l210_210907


namespace smallest_b_factors_l210_210015

theorem smallest_b_factors (p q b : ℤ) (hpq : p * q = 1764) (hb : b = p + q) (hposp : p > 0) (hposq : q > 0) :
  b = 84 :=
by
  sorry

end smallest_b_factors_l210_210015


namespace gcd_98_63_l210_210740

theorem gcd_98_63 : Nat.gcd 98 63 = 7 :=
by
  sorry

end gcd_98_63_l210_210740


namespace possible_values_of_a_l210_210070

theorem possible_values_of_a (a : ℝ) : (2 < a ∧ a < 3 ∨ 3 < a ∧ a < 5) → (a = 5/2 ∨ a = 4) := 
by
  sorry

end possible_values_of_a_l210_210070


namespace bucket_capacity_l210_210142

theorem bucket_capacity :
  (∃ (x : ℝ), 30 * x = 45 * 9) → 13.5 = 13.5 :=
by
  -- proof needed
  sorry

end bucket_capacity_l210_210142


namespace combination_eight_choose_five_l210_210504

theorem combination_eight_choose_five : nat.choose 8 5 = 56 :=
by sorry

end combination_eight_choose_five_l210_210504


namespace divisor_of_1076_plus_least_addend_l210_210741

theorem divisor_of_1076_plus_least_addend (a d : ℕ) (h1 : 1076 + a = 1081) (h2 : d ∣ 1081) (ha : a = 5) : d = 13 := 
sorry

end divisor_of_1076_plus_least_addend_l210_210741


namespace Maxim_born_in_2008_probability_l210_210248

def birth_interval_start := ⟨2007, 9, 2⟩ -- September 2, 2007
def birth_interval_end := ⟨2008, 8, 31⟩ -- August 31, 2008
def days_in_interval := 365
def days_in_2008 := 244

theorem Maxim_born_in_2008_probability :
  (days_in_2008 : ℝ) / (days_in_interval : ℝ) = 244 / 365 :=
sorry

end Maxim_born_in_2008_probability_l210_210248


namespace sum_of_roots_l210_210192

theorem sum_of_roots : 
  ∀ x1 x2 : ℝ, 
  (x1^2 + 2023*x1 = 2024 ∧ x2^2 + 2023*x2 = 2024) → 
  x1 + x2 = -2023 := 
by 
  sorry

end sum_of_roots_l210_210192


namespace number_of_intersection_points_l210_210385

noncomputable section

-- Define a type for Points in the plane
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Defining the five points
variables (A B C D E : Point)

-- Define the conditions that no three points are collinear
def no_three_collinear (A B C : Point) : Prop :=
  (B.x - A.x) * (C.y - A.y) ≠ (C.x - A.x) * (B.y - A.y)

-- Define the theorem statement
theorem number_of_intersection_points (h1 : no_three_collinear A B C)
  (h2 : no_three_collinear A B D)
  (h3 : no_three_collinear A B E)
  (h4 : no_three_collinear A C D)
  (h5 : no_three_collinear A C E)
  (h6 : no_three_collinear A D E)
  (h7 : no_three_collinear B C D)
  (h8 : no_three_collinear B C E)
  (h9 : no_three_collinear B D E)
  (h10 : no_three_collinear C D E) :
  ∃ (N : ℕ), N = 40 :=
  sorry

end number_of_intersection_points_l210_210385


namespace match_processes_count_l210_210452

-- Define the sets and the number of interleavings
def team_size : ℕ := 4 -- Each team has 4 players

-- Define the problem statement
theorem match_processes_count :
  (Nat.choose (2 * team_size) team_size) = 70 := by
  -- This is where the proof would go, but we'll use sorry as specified
  sorry

end match_processes_count_l210_210452


namespace range_and_period_range_of_m_l210_210971

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.cos (x + Real.pi / 3) * (Real.sin (x + Real.pi / 3) - Real.sqrt 3 * Real.cos (x + Real.pi / 3))

theorem range_and_period (x : ℝ) :
  (Set.range f = Set.Icc (-2 - Real.sqrt 3) (2 - Real.sqrt 3)) ∧ (∀ x, f (x + Real.pi) = f x) := sorry

theorem range_of_m (x m : ℝ) (h1 : 0 ≤ x ∧ x ≤ Real.pi / 6) (h2 : m * (f x + Real.sqrt 3) + 2 = 0) :
  m ∈ Set.Icc (- 2 * Real.sqrt 3 / 3) (-1) := sorry

end range_and_period_range_of_m_l210_210971


namespace find_factor_l210_210163

theorem find_factor (f : ℝ) : (120 * f - 138 = 102) → f = 2 :=
by
  sorry

end find_factor_l210_210163


namespace complementary_angle_decrease_l210_210878

theorem complementary_angle_decrease :
  (ratio : ℚ := 3 / 7) →
  let total_angle := 90
  let small_angle := (ratio * total_angle) / (1+ratio)
  let large_angle := total_angle - small_angle
  let new_small_angle := small_angle * 1.2
  let new_large_angle := total_angle - new_small_angle
  let decrease_percent := (large_angle - new_large_angle) / large_angle * 100
  decrease_percent = 8.57 :=
by
  sorry

end complementary_angle_decrease_l210_210878


namespace bonnie_roark_wire_length_ratio_l210_210167

-- Define the conditions
def bonnie_wire_pieces : ℕ := 12
def bonnie_wire_length_per_piece : ℕ := 8
def roark_wire_length_per_piece : ℕ := 2
def bonnie_cube_volume : ℕ := 8 * 8 * 8
def roark_total_cube_volume : ℕ := bonnie_cube_volume
def roark_unit_cube_volume : ℕ := 1
def roark_unit_cube_wires : ℕ := 12

-- Calculate Bonnie's total wire length
noncomputable def bonnie_total_wire_length : ℕ := bonnie_wire_pieces * bonnie_wire_length_per_piece

-- Calculate the number of Roark's unit cubes
noncomputable def roark_number_of_unit_cubes : ℕ := roark_total_cube_volume / roark_unit_cube_volume

-- Calculate the total wire used by Roark
noncomputable def roark_total_wire_length : ℕ := roark_number_of_unit_cubes * roark_unit_cube_wires * roark_wire_length_per_piece

-- Calculate the ratio of Bonnie's total wire length to Roark's total wire length
noncomputable def wire_length_ratio : ℚ := bonnie_total_wire_length / roark_total_wire_length

-- State the theorem
theorem bonnie_roark_wire_length_ratio : wire_length_ratio = 1 / 128 := 
by 
  sorry

end bonnie_roark_wire_length_ratio_l210_210167


namespace find_ff_of_five_half_l210_210046

noncomputable def f (x : ℝ) : ℝ :=
if x <= 1 then 2^x - 2 else Real.log x / Real.log 2

theorem find_ff_of_five_half : f (f (5/2)) = -1/2 := by
  sorry

end find_ff_of_five_half_l210_210046


namespace negation_of_p_l210_210365

-- Defining the proposition 'p'
def p : Prop := ∃ x : ℝ, x^3 > x

-- Stating the theorem
theorem negation_of_p : ¬p ↔ ∀ x : ℝ, x^3 ≤ x :=
by
  sorry

end negation_of_p_l210_210365


namespace farmer_children_l210_210602

noncomputable def numberOfChildren 
  (totalLeft : ℕ)
  (eachChildCollected : ℕ)
  (eatenApples : ℕ)
  (soldApples : ℕ) : ℕ :=
  let totalApplesEaten := eatenApples * 2
  let initialCollection := eachChildCollected * (totalLeft + totalApplesEaten + soldApples) / eachChildCollected
  initialCollection / eachChildCollected

theorem farmer_children (totalLeft : ℕ) (eachChildCollected : ℕ) (eatenApples : ℕ) (soldApples : ℕ) : 
  totalLeft = 60 → eachChildCollected = 15 → eatenApples = 4 → soldApples = 7 → 
  numberOfChildren totalLeft eachChildCollected eatenApples soldApples = 5 := 
by
  intro h_totalLeft h_eachChildCollected h_eatenApples h_soldApples
  unfold numberOfChildren
  simp
  sorry

end farmer_children_l210_210602


namespace parkway_elementary_students_l210_210997

/-- The total number of students in the fifth grade at Parkway Elementary School is 420,
given the following conditions:
1. There are 312 boys.
2. 250 students are playing soccer.
3. 78% of the students that play soccer are boys.
4. There are 53 girl students not playing soccer. -/
theorem parkway_elementary_students (boys : ℕ) (playing_soccer : ℕ) (percent_boys_playing : ℝ) (girls_not_playing_soccer : ℕ)
  (h1 : boys = 312)
  (h2 : playing_soccer = 250)
  (h3 : percent_boys_playing = 0.78)
  (h4 : girls_not_playing_soccer = 53) :
  ∃ total_students : ℕ, total_students = 420 :=
by
  sorry

end parkway_elementary_students_l210_210997


namespace find_a_b_l210_210348

theorem find_a_b (a b : ℝ) (h1 : b - a = -7) (h2 : 64 * (a + b) = 20736) :
  a = 165.5 ∧ b = 158.5 :=
by
  sorry

end find_a_b_l210_210348


namespace arithmetic_progression_l210_210432

-- Define the general formula for the nth term of an arithmetic progression
def nth_term (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

-- Define the conditions given in the problem
def condition1 (a1 d : ℤ) : Prop := nth_term a1 d 13 = 3 * nth_term a1 d 3
def condition2 (a1 d : ℤ) : Prop := nth_term a1 d 18 = 2 * nth_term a1 d 7 + 8

-- The main proof problem statement
theorem arithmetic_progression (a1 d : ℤ) (h1 : condition1 a1 d) (h2 : condition2 a1 d) : a1 = 12 ∧ d = 4 :=
by
  sorry

end arithmetic_progression_l210_210432


namespace ratio_volumes_of_spheres_l210_210985

theorem ratio_volumes_of_spheres (r R : ℝ) (hratio : (4 * π * r^2) / (4 * π * R^2) = 4 / 9) :
    (4 / 3 * π * r^3) / (4 / 3 * π * R^3) = 8 / 27 := 
by {
  sorry
}

end ratio_volumes_of_spheres_l210_210985


namespace eunji_received_900_won_l210_210748

-- Define the conditions
def eunji_pocket_money (X : ℝ) : Prop :=
  (X / 2 + 550 = 1000)

-- Define the theorem to prove the question equals the correct answer
theorem eunji_received_900_won {X : ℝ} (h : eunji_pocket_money X) : X = 900 :=
  by
    sorry

end eunji_received_900_won_l210_210748


namespace inverse_function_domain_l210_210219

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem inverse_function_domain :
  ∃ (g : ℝ → ℝ), (∀ x, 0 ≤ x → f (g x) = x) ∧ (∀ y, 0 ≤ y → g (f y) = y) ∧ (∀ x, 0 ≤ x ↔ 0 ≤ g x) :=
by
  sorry

end inverse_function_domain_l210_210219


namespace smallest_a_with_50_perfect_squares_l210_210027

theorem smallest_a_with_50_perfect_squares : ∃ a : ℕ, (∀ b : ℕ, (b = a) → (∀ n : ℕ, (n * n ≤ a ∧ a < (n + 1) * (n + 1)) ∧ (count_squares_in_interval (a, 3 * a) = 50)) ∧ a = 4486) :=
by
  sorry

def count_squares_in_interval (I : set ℤ) : ℕ :=
  sorry

end smallest_a_with_50_perfect_squares_l210_210027


namespace true_propositions_count_l210_210493

theorem true_propositions_count {a b c : ℝ} (h : a ≤ b) : 
  (if (c^2 ≥ 0 ∧ a * c^2 ≤ b * c^2) then 1 else 0) +
  (if (c^2 ≥ 0 ∧ a * c^2 > b * c^2) then 1 else 0) +
  (if (c^2 ≥ 0 ∧ ¬(a * c^2 ≤ b * c^2) → ¬(a ≤ b)) then 1 else 0) +
  (if (c^2 ≥ 0 ∧ ¬(a ≤ b) → ¬(a * c^2 ≤ b * c^2)) then 1 else 0) = 2 :=
sorry

end true_propositions_count_l210_210493


namespace simplify_and_evaluate_expression_l210_210725

variables (m n : ℚ)

theorem simplify_and_evaluate_expression (h1 : m = -1) (h2 : n = 1 / 2) :
  ( (2 / (m - n) - 1 / (m + n)) / ((m * n + 3 * n ^ 2) / (m ^ 3 - m * n ^ 2)) ) = -2 :=
by
  sorry

end simplify_and_evaluate_expression_l210_210725


namespace find_value_of_expression_l210_210215

-- Conditions translated to Lean 4 definitions
variable (a b : ℝ)
axiom h1 : (a^2 * b^3) / 5 = 1000
axiom h2 : a * b = 2

-- The theorem stating what we need to prove
theorem find_value_of_expression :
  (a^3 * b^2) / 3 = 2 / 705 :=
by
  sorry

end find_value_of_expression_l210_210215


namespace positive_abc_l210_210651

theorem positive_abc (a b c : ℝ) (h1 : a + b + c > 0) (h2 : ab + bc + ca > 0) (h3 : abc > 0) : a > 0 ∧ b > 0 ∧ c > 0 := 
by
  sorry

end positive_abc_l210_210651


namespace range_of_a_l210_210966

noncomputable def f (a x : ℝ) : ℝ := 2 * x^2 - a * x + 5

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → f a x ≤ f a y) → a ≤ 4 :=
by
  sorry

end range_of_a_l210_210966


namespace distance_from_circle_to_line_l210_210071

def polar_circle (ρ θ : ℝ) : Prop := ρ = 4 * Real.sin θ
def polar_line (θ : ℝ) : Prop := θ = Real.pi / 6

theorem distance_from_circle_to_line : 
  ∃ d : ℝ, polar_circle ρ θ ∧ polar_line θ → d = Real.sqrt 3 := 
by
  sorry

end distance_from_circle_to_line_l210_210071


namespace find_4a_plus_8b_l210_210373

def quadratic_equation_x_solution (a b : ℝ) : Prop :=
  (1 : ℝ)^2 + a * (1 : ℝ) + 2 * b = 0

theorem find_4a_plus_8b (a b : ℝ) (h : quadratic_equation_x_solution a b) : 4 * a + 8 * b = -4 := 
  by
    sorry

end find_4a_plus_8b_l210_210373


namespace two_box_even_sum_probability_l210_210177

theorem two_box_even_sum_probability : 
  let chips := {1, 2, 4}
  let draws := { (a, b) | a ∈ chips ∧ b ∈ chips }
  let even_sum := { (a, b) ∈ draws | (a + b) % 2 = 0 }
  (|even_sum| : ℚ) / (|draws| : ℚ) = 5 / 9 :=
by
  have chips_def : chips = {1, 2, 4} := rfl
  have draws_def : draws = { (a, b) | a ∈ chips ∧ b ∈ chips } := rfl
  have even_sum_def : even_sum = { (a, b) | (a + b) % 2 = 0 } := rfl
  sorry

end two_box_even_sum_probability_l210_210177


namespace collinear_vectors_l210_210968

theorem collinear_vectors (x : ℝ) :
  (∃ k : ℝ, (2, 4) = (k * 2, k * 4) ∧ (k * 2 = x ∧ k * 4 = 6)) → x = 3 :=
by
  intros h
  sorry

end collinear_vectors_l210_210968


namespace binom_eight_three_l210_210334

theorem binom_eight_three : Nat.choose 8 3 = 56 := by
  sorry

end binom_eight_three_l210_210334


namespace sum_first_9000_terms_l210_210887

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_9000_terms (a r : ℝ) :
  geometric_sequence_sum a r 3000 = 500 →
  geometric_sequence_sum a r 6000 = 950 →
  geometric_sequence_sum a r 9000 = 1355 :=
by
  intros h1 h2
  sorry

end sum_first_9000_terms_l210_210887


namespace rectangle_area_proof_l210_210682

def rectangle_area (L W : ℝ) : ℝ := L * W

theorem rectangle_area_proof (L W : ℝ) (h1 : L + W = 23) (h2 : L^2 + W^2 = 289) : rectangle_area L W = 120 := by
  sorry

end rectangle_area_proof_l210_210682


namespace geometric_sequence_sum_l210_210896

noncomputable theory

-- Define the conditions and the problem statement
theorem geometric_sequence_sum (a r : ℝ) (h₁ : a * (1 - r^3000) / (1 - r) = 500) 
  (h₂ : a * (1 - r^6000) / (1 - r) = 950) : 
  a * (1 - r^9000) / (1 - r) = 1355 := 
sorry

end geometric_sequence_sum_l210_210896


namespace range_a_part1_range_a_part2_l210_210048

def A (x : ℝ) : Prop := x^2 - 3*x + 2 ≤ 0
def B (x a : ℝ) : Prop := x = x^2 - 4*x + a
def C (x a : ℝ) : Prop := x^2 - a*x - 4 ≤ 0

def p (a : ℝ) : Prop := ∃ x : ℝ, A x ∧ B x a
def q (a : ℝ) : Prop := ∀ x : ℝ, A x → C x a

theorem range_a_part1 : ¬(p a) → a > 6 := sorry

theorem range_a_part2 : p a ∧ q a → 0 ≤ a ∧ a ≤ 6 := sorry

end range_a_part1_range_a_part2_l210_210048


namespace find_points_on_number_line_l210_210958

noncomputable def numbers_are_opposite (x y : ℝ) : Prop :=
  x = -y

theorem find_points_on_number_line (A B : ℝ) 
  (h1 : numbers_are_opposite A B) 
  (h2 : |A - B| = 8) 
  (h3 : A < B) : 
  (A = -4 ∧ B = 4) :=
by
  sorry

end find_points_on_number_line_l210_210958


namespace carly_shipping_cost_l210_210939

noncomputable def total_shipping_cost (flat_fee cost_per_pound weight : ℝ) : ℝ :=
flat_fee + cost_per_pound * weight

theorem carly_shipping_cost : 
  total_shipping_cost 5 0.80 5 = 9 :=
by 
  unfold total_shipping_cost
  norm_num

end carly_shipping_cost_l210_210939


namespace ads_not_blocked_not_interesting_l210_210976

theorem ads_not_blocked_not_interesting:
  (let A_blocks := 0.75
   let B_blocks := 0.85
   let C_blocks := 0.95
   let A_let_through := 1 - A_blocks
   let B_let_through := 1 - B_blocks
   let C_let_through := 1 - C_blocks
   let all_let_through := A_let_through * B_let_through * C_let_through
   let interesting := 0.15
   let not_interesting := 1 - interesting
   (all_let_through * not_interesting) = 0.00159375) :=
  sorry

end ads_not_blocked_not_interesting_l210_210976


namespace smallest_a_with_50_perfect_squares_l210_210028

theorem smallest_a_with_50_perfect_squares : ∃ a : ℕ, (∀ b : ℕ, (b = a) → (∀ n : ℕ, (n * n ≤ a ∧ a < (n + 1) * (n + 1)) ∧ (count_squares_in_interval (a, 3 * a) = 50)) ∧ a = 4486) :=
by
  sorry

def count_squares_in_interval (I : set ℤ) : ℕ :=
  sorry

end smallest_a_with_50_perfect_squares_l210_210028


namespace six_digit_squares_l210_210173

theorem six_digit_squares (x y : ℕ) 
  (h1 : y < 1000)
  (h2 : (1000 * x + y) < 1000000)
  (h3 : y * (y - 1) = 1000 * x)
  (mod8 : y * (y - 1) ≡ 0 [MOD 8])
  (mod125 : y * (y - 1) ≡ 0 [MOD 125]) :
  (1000 * x + y = 390625 ∨ 1000 * x + y = 141376) :=
sorry

end six_digit_squares_l210_210173


namespace power_mod_eq_five_l210_210546

theorem power_mod_eq_five
  (m : ℕ)
  (h₀ : 0 ≤ m)
  (h₁ : m < 8)
  (h₂ : 13^5 % 8 = m) : m = 5 :=
by 
  sorry

end power_mod_eq_five_l210_210546


namespace rectangle_area_formula_l210_210276

-- Define the given conditions: perimeter is 20, one side length is x
def rectangle_perimeter (P x : ℝ) (w : ℝ) : Prop := P = 2 * (x + w)
def rectangle_area (x w : ℝ) : ℝ := x * w

-- The theorem to prove
theorem rectangle_area_formula (x : ℝ) (h_perimeter : rectangle_perimeter 20 x (10 - x)) : 
  rectangle_area x (10 - x) = x * (10 - x) := 
by 
  sorry

end rectangle_area_formula_l210_210276


namespace sum_first_9000_terms_l210_210884

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_9000_terms (a r : ℝ) :
  geometric_sequence_sum a r 3000 = 500 →
  geometric_sequence_sum a r 6000 = 950 →
  geometric_sequence_sum a r 9000 = 1355 :=
by
  intros h1 h2
  sorry

end sum_first_9000_terms_l210_210884


namespace frood_game_least_n_l210_210695

theorem frood_game_least_n (n : ℕ) (h : n > 0) (drop_score : ℕ := n * (n + 1) / 2) (eat_score : ℕ := 15 * n) 
  : drop_score > eat_score ↔ n ≥ 30 :=
by
  sorry

end frood_game_least_n_l210_210695


namespace smallest_b_multiple_of_6_and_15_is_30_l210_210643

theorem smallest_b_multiple_of_6_and_15_is_30 : ∃ b : ℕ, b > 0 ∧ (6 ∣ b) ∧ (15 ∣ b) ∧ b = 30 :=
by
  use 30
  split
  . trivial
  . split
    . sorry
    . sorry

end smallest_b_multiple_of_6_and_15_is_30_l210_210643


namespace expectation_of_2ξ_plus_1_variance_of_2ξ_plus_1_l210_210817

variable (ξ : ℝ)

-- Given conditions
def E_ξ : ℝ := 3
def D_ξ : ℝ := 4

-- Questions and corresponding correct answers
theorem expectation_of_2ξ_plus_1 : 
  E (2 * ξ + 1) = 7 := by sorry

theorem variance_of_2ξ_plus_1 : 
  D (2 * ξ + 1) = 16 := by sorry

end expectation_of_2ξ_plus_1_variance_of_2ξ_plus_1_l210_210817


namespace multiples_of_7_with_unit_digit_7_and_lt_150_l210_210832

/--
Given a positive integer n,
Let \(7n\) be a multiple of 7,
\(7n < 150\),
and \(7n \mod 10 = 7\),
prove there are 11 such multiples of 7.
-/
theorem multiples_of_7_with_unit_digit_7_and_lt_150 :
  ∃! (n : ℕ), n ≤ 11 ∧ ∀ (m : ℕ), (1 ≤ m ∧ m ≤ n) → (7 * (2 * m - 1) < 150 ∧ (7 * (2 * m - 1)) % 10 = 7) := by
sorry

end multiples_of_7_with_unit_digit_7_and_lt_150_l210_210832


namespace different_answers_due_to_different_cuts_l210_210631

noncomputable def problem_89914 (bub : Type) (cut : bub → (bub × bub)) (is_log_cut : bub → Prop) (is_halved_log : bub × bub → Prop) : Prop :=
  ∀ b : bub, (is_log_cut b) → is_halved_log (cut b)

noncomputable def problem_89915 (bub : Type) (cut : bub → (bub × bub)) (is_sector_cut : bub → Prop) (is_sectors : bub × bub → Prop) : Prop :=
  ∀ b : bub, (is_sector_cut b) → is_sectors (cut b)

theorem different_answers_due_to_different_cuts
  (bub : Type)
  (cut : bub → (bub × bub))
  (is_log_cut : bub → Prop)
  (is_halved_log : bub × bub → Prop)
  (is_sector_cut : bub → Prop)
  (is_sectors : bub × bub → Prop) :
  problem_89914 bub cut is_log_cut is_halved_log ∧ problem_89915 bub cut is_sector_cut is_sectors →
  ∃ b : bub, (is_log_cut b ∧ ¬ is_sector_cut b) ∨ (¬ is_log_cut b ∧ is_sector_cut b) := sorry

end different_answers_due_to_different_cuts_l210_210631


namespace pie_eating_contest_l210_210118

theorem pie_eating_contest :
  let a := 5 / 6
  let b := 7 / 8
  let c := 2 / 3
  let max_pie := max a (max b c)
  let min_pie := min a (min b c)
  max_pie - min_pie = 5 / 24 :=
by
  sorry

end pie_eating_contest_l210_210118


namespace number_of_families_l210_210996

theorem number_of_families (x : ℕ) (h1 : x + x / 3 = 100) : x = 75 :=
sorry

end number_of_families_l210_210996


namespace greatest_integer_gcd_l210_210563

theorem greatest_integer_gcd (n : ℕ) (h1 : n < 200) (h2 : gcd n 18 = 6) : n = 192 :=
sorry

end greatest_integer_gcd_l210_210563


namespace smallest_n_l210_210666

theorem smallest_n (n : ℕ) :
  (∀ m : ℤ, 0 < m ∧ m < 2001 →
    ∃ k : ℤ, (m : ℚ) / 2001 < (k : ℚ) / n ∧ (k : ℚ) / n < (m + 1 : ℚ) / 2002) ↔ n = 4003 :=
by
  sorry

end smallest_n_l210_210666


namespace class_funding_reached_l210_210498

-- Definition of the conditions
def students : ℕ := 45
def goal : ℝ := 3000
def full_payment_students : ℕ := 25
def full_payment_amount : ℝ := 60
def merit_students : ℕ := 10
def merit_payment_per_student_euro : ℝ := 40
def euro_to_usd : ℝ := 1.20
def financial_needs_students : ℕ := 7
def financial_needs_payment_per_student_pound : ℝ := 30
def pound_to_usd : ℝ := 1.35
def discount_students : ℕ := 3
def discount_payment_per_student_cad : ℝ := 68
def cad_to_usd : ℝ := 0.80
def administrative_fee_yen : ℝ := 10000
def yen_to_usd : ℝ := 0.009

-- Definitions of amounts
def full_payment_amount_total : ℝ := full_payment_students * full_payment_amount
def merit_payment_amount_total : ℝ := merit_students * merit_payment_per_student_euro * euro_to_usd
def financial_needs_payment_amount_total : ℝ := financial_needs_students * financial_needs_payment_per_student_pound * pound_to_usd
def discount_payment_amount_total : ℝ := discount_students * discount_payment_per_student_cad * cad_to_usd
def administrative_fee_usd : ℝ := administrative_fee_yen * yen_to_usd

-- Definition of total collected
def total_collected : ℝ := 
  full_payment_amount_total + 
  merit_payment_amount_total + 
  financial_needs_payment_amount_total + 
  discount_payment_amount_total - 
  administrative_fee_usd

-- The final theorem statement
theorem class_funding_reached : total_collected = 2427.70 ∧ goal - total_collected = 572.30 := by
  sorry

end class_funding_reached_l210_210498


namespace original_population_l210_210613

variable (n : ℝ)

theorem original_population
  (h1 : n + 1500 - 0.15 * (n + 1500) = n - 45) :
  n = 8800 :=
sorry

end original_population_l210_210613


namespace ratio_length_to_width_l210_210609

def garden_length := 80
def garden_perimeter := 240

theorem ratio_length_to_width : ∃ W, 2 * garden_length + 2 * W = garden_perimeter ∧ garden_length / W = 2 := by
  sorry

end ratio_length_to_width_l210_210609


namespace find_n_l210_210184

theorem find_n (n : ℕ) (hn : n * n! - n! = 5040 - n!) : n = 7 :=
by
  sorry

end find_n_l210_210184


namespace geometric_sequence_sum_l210_210891

theorem geometric_sequence_sum
  (a r : ℝ)
  (h1 : a * (1 - r ^ 3000) / (1 - r) = 500)
  (h2 : a * (1 - r ^ 6000) / (1 - r) = 950) :
  a * (1 - r ^ 9000) / (1 - r) = 1355 :=
sorry

end geometric_sequence_sum_l210_210891


namespace angle_relationship_l210_210998

open Real

variables (A B C D : Point)
variables (AB AC AD : ℝ)
variables (CAB DAC BDC DBC : ℝ)
variables (k : ℝ)

-- Given conditions
axiom h1 : AB = AC
axiom h2 : AC = AD
axiom h3 : DAC = k * CAB

-- Proof to be shown
theorem angle_relationship : DBC = k * BDC :=
  sorry

end angle_relationship_l210_210998


namespace total_distance_walked_l210_210140

noncomputable def desk_to_fountain_distance : ℕ := 30
noncomputable def number_of_trips : ℕ := 4

theorem total_distance_walked :
  2 * desk_to_fountain_distance * number_of_trips = 240 :=
by
  sorry

end total_distance_walked_l210_210140


namespace symmetric_words_l210_210792

def symmetric_words_count : ℕ := 12

theorem symmetric_words :
  let positions_nat_total := 6
  let positions_n := 2
  let symmetric_property (s : String) : Prop := s = s.reverse
  (∃ (arrangements : Finset (Finset ℕ)). 
    arrangements.card = 3 
    ∧ (∀ w ∈ arrangements, symmetric_property w) 
    ∧ arrangements.card * 4 = symmetric_words_count) := sorry

end symmetric_words_l210_210792


namespace sum_of_cubes_consecutive_divisible_by_9_l210_210865

theorem sum_of_cubes_consecutive_divisible_by_9 (n : ℤ) : 9 ∣ (n-1)^3 + n^3 + (n+1)^3 :=
  sorry

end sum_of_cubes_consecutive_divisible_by_9_l210_210865


namespace painters_work_l210_210391

theorem painters_work (w1 w2 : ℕ) (d1 d2 : ℚ) (C : ℚ) (h1 : w1 * d1 = C) (h2 : w2 * d2 = C) (p : w1 = 5) (t : d1 = 1.6) (a : w2 = 4) : d2 = 2 := 
by
  sorry

end painters_work_l210_210391


namespace pen_and_pencil_total_cost_l210_210442

theorem pen_and_pencil_total_cost :
  ∀ (pen pencil : ℕ), pen = 4 → pen = 2 * pencil → pen + pencil = 6 :=
by
  intros pen pencil
  intro h1
  intro h2
  sorry

end pen_and_pencil_total_cost_l210_210442


namespace range_of_z_l210_210648

theorem range_of_z (a b : ℝ) (h1 : 2 < a) (h2 : a < 3) (h3 : -2 < b) (h4 : b < -1) :
  5 < 2 * a - b ∧ 2 * a - b < 8 :=
by
  sorry

end range_of_z_l210_210648


namespace convert_deg_to_rad_l210_210170

theorem convert_deg_to_rad (deg_to_rad : ℝ → ℝ) (conversion_factor : deg_to_rad 1 = π / 180) :
  deg_to_rad (-300) = - (5 * π) / 3 :=
by
  sorry

end convert_deg_to_rad_l210_210170


namespace original_numbers_product_l210_210282

theorem original_numbers_product (a b c d x : ℕ) 
  (h1 : a + b + c + d = 243)
  (h2 : a + 8 = x)
  (h3 : b - 8 = x)
  (h4 : c * 8 = x)
  (h5 : d / 8 = x) : 
  (min (min a (min b (min c d))) * max a (max b (max c d))) = 576 :=
by 
  sorry

end original_numbers_product_l210_210282


namespace slopes_product_of_tangents_l210_210856

theorem slopes_product_of_tangents 
  (x₀ y₀ : ℝ) 
  (h_hyperbola : (2 * x₀^2) / 3 - y₀^2 / 6 = 1) 
  (h_outside_circle : x₀^2 + y₀^2 > 2) :
  ∃ (k₁ k₂ : ℝ), 
    k₁ * k₂ = 4 ∧ 
    (y₀ - k₁ * x₀)^2 + k₁^2 = 2 ∧ 
    (y₀ - k₂ * x₀)^2 + k₂^2 = 2 :=
by {
  -- this proof will use the properties of tangents to a circle and the constraints given
  -- we don't need to implement it now, but we aim to show the correct relationship
  sorry
}

end slopes_product_of_tangents_l210_210856


namespace polynomial_abs_value_at_neg_one_l210_210402

theorem polynomial_abs_value_at_neg_one:
  ∃ g : Polynomial ℝ, 
  (∀ x ∈ ({0, 1, 2, 4, 5, 6} : Set ℝ), |g.eval x| = 15) → 
  |g.eval (-1)| = 75 :=
by
  sorry

end polynomial_abs_value_at_neg_one_l210_210402


namespace find_printer_price_l210_210557

variable (C P M : ℝ)

theorem find_printer_price
  (h1 : C + P + M = 3000)
  (h2 : P = (1/4) * (C + P + M + 800)) :
  P = 950 :=
sorry

end find_printer_price_l210_210557


namespace point_on_xoz_plane_l210_210229

def Point := ℝ × ℝ × ℝ

def lies_on_plane_xoz (p : Point) : Prop :=
  p.2 = 0

theorem point_on_xoz_plane :
  lies_on_plane_xoz (-2, 0, 3) :=
by
  sorry

end point_on_xoz_plane_l210_210229


namespace sin_pi_over_4_l210_210520

theorem sin_pi_over_4 (α : ℝ) (hα1 : sin α = 4/5) (hα2 : cos α = -3/5) :
  sin (α + real.pi / 4) = real.sqrt 2 / 10 :=
by
  sorry

end sin_pi_over_4_l210_210520


namespace max_min_value_d_l210_210198

-- Definitions of the given conditions
def circle_eqn (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Definition of the distance squared from a point to a fixed point
def dist_sq (P Q : ℝ × ℝ) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Definition of the function d
def d (P : ℝ × ℝ) : ℝ := dist_sq P A + dist_sq P B

-- The main theorem that we need to prove
theorem max_min_value_d :
  (∀ P : ℝ × ℝ, circle_eqn P.1 P.2 → d P ≤ 74) ∧
  (∃ P : ℝ × ℝ, circle_eqn P.1 P.2 ∧ d P = 74) ∧
  (∀ P : ℝ × ℝ, circle_eqn P.1 P.2 → 34 ≤ d P) ∧
  (∃ P : ℝ × ℝ, circle_eqn P.1 P.2 ∧ d P = 34) :=
sorry

end max_min_value_d_l210_210198


namespace largest_percent_error_l210_210397
noncomputable def max_percent_error (d : ℝ) (d_err : ℝ) (r_err : ℝ) : ℝ :=
  let d_min := d - d * d_err
  let d_max := d + d * d_err
  let r := d / 2
  let r_min := r - r * r_err
  let r_max := r + r * r_err
  let area_actual := Real.pi * r^2
  let area_d_min := Real.pi * (d_min / 2)^2
  let area_d_max := Real.pi * (d_max / 2)^2
  let area_r_min := Real.pi * r_min^2
  let area_r_max := Real.pi * r_max^2
  let error_d_min := (area_actual - area_d_min) / area_actual * 100
  let error_d_max := (area_d_max - area_actual) / area_actual * 100
  let error_r_min := (area_actual - area_r_min) / area_actual * 100
  let error_r_max := (area_r_max - area_actual) / area_actual * 100
  max (max error_d_min error_d_max) (max error_r_min error_r_max)

theorem largest_percent_error 
  (d : ℝ) (d_err : ℝ) (r_err : ℝ) 
  (h_d : d = 30) (h_d_err : d_err = 0.15) (h_r_err : r_err = 0.10) : 
  max_percent_error d d_err r_err = 31.57 := by
  sorry

end largest_percent_error_l210_210397


namespace marble_221_is_green_l210_210320

def marble_sequence_color (n : ℕ) : String :=
  let cycle_length := 15
  let red_count := 6
  let green_start := red_count + 1
  let green_end := red_count + 5
  let position := n % cycle_length
  if position ≠ 0 then
    let cycle_position := position
    if cycle_position <= red_count then "red"
    else if cycle_position <= green_end then "green"
    else "blue"
  else "blue"

theorem marble_221_is_green : marble_sequence_color 221 = "green" :=
by
  -- proof to be filled in
  sorry

end marble_221_is_green_l210_210320


namespace ab_value_l210_210482

theorem ab_value (a b : ℤ) (h1 : |a| = 7) (h2 : b = 5) (h3 : a + b < 0) : a * b = -35 := 
by
  sorry

end ab_value_l210_210482


namespace cheryl_needed_first_material_l210_210271

noncomputable def cheryl_material (x : ℚ) : ℚ :=
  x + 1 / 3 - 3 / 8

theorem cheryl_needed_first_material
  (h_total_used : 0.33333333333333326 = 1 / 3) :
  cheryl_material x = 1 / 3 → x = 3 / 8 :=
by
  intros
  rw [h_total_used] at *
  sorry

end cheryl_needed_first_material_l210_210271


namespace Joe_spent_800_on_hotel_l210_210234

noncomputable def Joe'sExpenses : Prop :=
  let S := 6000 -- Joe's total savings
  let F := 1200 -- Expense on the flight
  let FD := 3000 -- Expense on food
  let R := 1000 -- Remaining amount after all expenses
  let H := S - R - (F + FD) -- Calculating hotel expense
  H = 800 -- We need to prove the hotel expense equals $800

theorem Joe_spent_800_on_hotel : Joe'sExpenses :=
by {
  -- Proof goes here; currently skipped
  sorry
}

end Joe_spent_800_on_hotel_l210_210234


namespace find_a_b_find_extreme_point_g_num_zeros_h_l210_210964

-- (1) Proving the values of a and b
theorem find_a_b (a b : ℝ)
  (h1 : (3 + 2 * a + b = 0))
  (h2 : (3 - 2 * a + b = 0)) : 
  a = 0 ∧ b = -3 :=
sorry

-- (2) Proving the extreme points of g(x)
theorem find_extreme_point_g (x : ℝ) : 
  x = -2 :=
sorry

-- (3) Proving the number of zeros of h(x)
theorem num_zeros_h (c : ℝ) (h : -2 ≤ c ∧ c ≤ 2) :
  (|c| = 2 → ∃ y, y = 5) ∧ (|c| < 2 → ∃ y, y = 9) :=
sorry

end find_a_b_find_extreme_point_g_num_zeros_h_l210_210964


namespace count_multiples_of_7_with_units_digit_7_less_than_150_l210_210830

-- Definition of the problem's conditions and expected result
theorem count_multiples_of_7_with_units_digit_7_less_than_150 : 
  let multiples_of_7 := { n : Nat // n < 150 ∧ n % 7 = 0 }
  let units_digit_7_multiples := { n : Nat // n < 150 ∧ n % 7 = 0 ∧ n % 10 = 7 }
  List.length (units_digit_7_multiples.members) = 3 :=
by
  sorry

end count_multiples_of_7_with_units_digit_7_less_than_150_l210_210830


namespace father_children_problem_l210_210100

theorem father_children_problem {F C n : ℕ} 
  (hF_C : F = C) 
  (sum_ages_after_15_years : C + 15 * n = 2 * (F + 15)) 
  (father_age : F = 75) : 
  n = 7 :=
by
  sorry

end father_children_problem_l210_210100


namespace final_temperature_is_58_32_l210_210525

-- Initial temperature
def T₀ : ℝ := 40

-- Sequence of temperature adjustments
def T₁ : ℝ := 2 * T₀
def T₂ : ℝ := T₁ - 30
def T₃ : ℝ := T₂ * (1 - 0.30)
def T₄ : ℝ := T₃ + 24
def T₅ : ℝ := T₄ * (1 - 0.10)
def T₆ : ℝ := T₅ + 8
def T₇ : ℝ := T₆ * (1 + 0.20)
def T₈ : ℝ := T₇ - 15

-- Proof statement
theorem final_temperature_is_58_32 : T₈ = 58.32 :=
by sorry

end final_temperature_is_58_32_l210_210525


namespace geometric_sequence_sum_l210_210888

theorem geometric_sequence_sum
  (a r : ℝ)
  (h1 : a * (1 - r ^ 3000) / (1 - r) = 500)
  (h2 : a * (1 - r ^ 6000) / (1 - r) = 950) :
  a * (1 - r ^ 9000) / (1 - r) = 1355 :=
sorry

end geometric_sequence_sum_l210_210888


namespace batsman_average_after_12th_innings_l210_210749

theorem batsman_average_after_12th_innings (A : ℤ) :
  (∀ A : ℤ, (11 * A + 60 = 12 * (A + 2))) → (A = 36) → (A + 2 = 38) := 
by
  intro h_avg_increase h_init_avg
  sorry

end batsman_average_after_12th_innings_l210_210749


namespace value_of_expression_when_x_is_neg2_l210_210129

theorem value_of_expression_when_x_is_neg2 : 
  ∀ (x : ℤ), x = -2 → (3 * x + 4) ^ 2 = 4 :=
by
  sorry

end value_of_expression_when_x_is_neg2_l210_210129


namespace maxim_birth_probability_l210_210252

open Nat

def interval_days (start_date end_date : ℕ) : ℕ :=
  end_date - start_date + 1

def total_days_2007_2008 : ℕ :=
  interval_days 245 2735 -- total days from Sep 2, 2007, to Aug 31, 2008

def days_in_2008 : ℕ :=
  interval_days 305 548  -- total days from Jan 1, 2008, to Aug 31, 2008

def probability_born_in_2008 : ℚ :=
  (days_in_2008 : ℚ) / (total_days_2007_2008 : ℚ)

theorem maxim_birth_probability: probability_born_in_2008 = 244 / 365 := 
  sorry

end maxim_birth_probability_l210_210252


namespace calculate_X_value_l210_210836

theorem calculate_X_value : 
  let M := (2025 : ℝ) / 3
  let N := M / 4
  let X := M - N
  X = 506.25 :=
by 
  sorry

end calculate_X_value_l210_210836


namespace rectangle_area_invariant_l210_210548

theorem rectangle_area_invariant
    (x y : ℕ)
    (h1 : x * y = (x + 3) * (y - 1))
    (h2 : x * y = (x - 3) * (y + 2)) :
    x * y = 15 :=
by sorry

end rectangle_area_invariant_l210_210548


namespace solve_for_a_l210_210979

theorem solve_for_a (x a : ℤ) (h : x = 3) (heq : 2 * x - 10 = 4 * a) : a = -1 := by
  sorry

end solve_for_a_l210_210979


namespace percentage_increase_l210_210989

variable {x y : ℝ}
variable {P : ℝ} -- percentage

theorem percentage_increase (h1 : y = x * (1 + P / 100)) (h2 : x = y * 0.5882352941176471) : P = 70 := 
by
  sorry

end percentage_increase_l210_210989


namespace remainder_division_l210_210189

theorem remainder_division (x : ℝ) :
  (x ^ 2021 + 1) % (x ^ 12 - x ^ 9 + x ^ 6 - x ^ 3 + 1) = -x ^ 4 + 1 :=
sorry

end remainder_division_l210_210189


namespace curve_line_and_circle_l210_210549

theorem curve_line_and_circle : 
  ∀ x y : ℝ, (x^3 + x * y^2 = 2 * x) ↔ (x = 0 ∨ x^2 + y^2 = 2) :=
by
  sorry

end curve_line_and_circle_l210_210549


namespace thabo_books_l210_210868

/-- Thabo's book count puzzle -/
theorem thabo_books (H P F : ℕ) (h1 : P = H + 20) (h2 : F = 2 * P) (h3 : H + P + F = 200) : H = 35 :=
by
  -- sorry is used to skip the proof, only state the theorem.
  sorry

end thabo_books_l210_210868


namespace solve_for_x_l210_210278

theorem solve_for_x (x : ℝ) : 3 * x^2 + 15 = 3 * (2 * x + 20) → (x = 5 ∨ x = -3) :=
sorry

end solve_for_x_l210_210278


namespace percentage_reduction_in_women_l210_210864

theorem percentage_reduction_in_women
    (total_people : Nat) (men_in_office : Nat) (women_in_office : Nat)
    (men_in_meeting : Nat) (women_in_meeting : Nat)
    (even_men_women : men_in_office = women_in_office)
    (total_people_condition : total_people = men_in_office + women_in_office)
    (meeting_condition : total_people = 60)
    (men_meeting_condition : men_in_meeting = 4)
    (women_meeting_condition : women_in_meeting = 6) :
    ((women_in_meeting * 100) / women_in_office) = 20 :=
by
  sorry

end percentage_reduction_in_women_l210_210864


namespace calc1_calc2_calc3_calc4_l210_210789

theorem calc1 : (-16) - 25 + (-43) - (-39) = -45 := by
  sorry

theorem calc2 : (-3 / 4)^2 * (-8 + 1 / 3) = -69 / 16 := by
  sorry

theorem calc3 : 16 / (- (1 / 2)) * (3 / 8) - | -45 | / 9 = -17 := by
  sorry

theorem calc4 : -1 ^ 2024 - (2 - 0.75) * (2 / 7) * (4 - (-5)^2) = 13 / 2 := by
  sorry

end calc1_calc2_calc3_calc4_l210_210789


namespace line_through_point_l210_210105

theorem line_through_point (b : ℚ) :
  (∃ x y,
    (x = 3) ∧ (y = -7) ∧ (b * x + (b - 1) * y = b + 3))
  → (b = 4 / 5) :=
begin
  sorry
end
 
end line_through_point_l210_210105


namespace mixed_doubles_teams_l210_210341

theorem mixed_doubles_teams (males females : ℕ) (hm : males = 6) (hf : females = 7) : (males * females) = 42 :=
by
  sorry

end mixed_doubles_teams_l210_210341


namespace f_monotonically_decreasing_range_of_a_tangent_intersection_l210_210972

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 - a * x^2 + 2
noncomputable def f' (x : ℝ) (a : ℝ) : ℝ := 3 * x^2 - 2 * a * x

-- Part (I)
theorem f_monotonically_decreasing (a : ℝ) (x : ℝ) :
  (a > 0 → 0 < x ∧ x < (2 / 3) * a → f' x a < 0) ∧
  (a = 0 → ¬∃ x, f' x a < 0) ∧
  (a < 0 → (2 / 3) * a < x ∧ x < 0 → f' x a < 0) :=
sorry

-- Part (II)
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f' x a ≥ abs x - 3 / 4) → (-1 ≤ a ∧ a ≤ 1) :=
sorry

-- Part (III)
theorem tangent_intersection (a : ℝ) :
  (a = 0 → ∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ ∃ t : ℝ, (t - x1^3 - 2 = 3 * x1^2 * (2 - x1)) ∧
  (t - x2^3 - 2 = 3 * x2^2 * (2 - x2)) ∧ 2 ≤ t ∧ t ≤ 10 ∧
  ∀ t', (t' - x1^3 - 2 = 3 * x1^2 * (2 - x1)) ∧
  (t' - x2^3 - 2 = 3 * x2^2 * (2 - x2)) → t' ≤ 10) :=
sorry

end f_monotonically_decreasing_range_of_a_tangent_intersection_l210_210972


namespace largest_ordered_pair_exists_l210_210005

-- Define the condition for ordered pairs (a, b)
def ordered_pair_condition (a b : ℤ) : Prop :=
  1 ≤ a ∧ a ≤ b ∧ b ≤ 100 ∧ ∃ (k : ℤ), (a + b) * (a + b + 1) = k * a * b

-- Define the specific ordered pair to be checked
def specific_pair (a b : ℤ) : Prop :=
  a = 35 ∧ b = 90

-- The main statement to be proven
theorem largest_ordered_pair_exists : specific_pair 35 90 ∧ ordered_pair_condition 35 90 :=
by
  sorry

end largest_ordered_pair_exists_l210_210005


namespace log_relationship_l210_210808

noncomputable def log_m (m x : ℝ) : ℝ := Real.log x / Real.log m

theorem log_relationship (m : ℝ) (h₀ : 0 < m) (h₁ : m < 1) :
  log_m m 0.3 > log_m m 0.5 :=
by
  sorry

end log_relationship_l210_210808


namespace min_perimeter_triangle_l210_210846

theorem min_perimeter_triangle (a b c : ℝ) (cosC : ℝ) :
  a + b = 10 ∧ cosC = -1/2 ∧ c^2 = (a - 5)^2 + 75 →
  a + b + c = 10 + 5 * Real.sqrt 3 :=
by
  sorry

end min_perimeter_triangle_l210_210846


namespace molecular_weight_correct_l210_210147

namespace MolecularWeight

-- Define the atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_Cl : ℝ := 35.45

-- Define the number of each atom in the compound
def n_N : ℝ := 1
def n_H : ℝ := 4
def n_Cl : ℝ := 1

-- Calculate the molecular weight of the compound
def molecular_weight : ℝ := (n_N * atomic_weight_N) + (n_H * atomic_weight_H) + (n_Cl * atomic_weight_Cl)

theorem molecular_weight_correct : molecular_weight = 53.50 := by
  -- Proof is omitted
  sorry

end MolecularWeight

end molecular_weight_correct_l210_210147


namespace sum_first_9000_terms_l210_210885

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_9000_terms (a r : ℝ) :
  geometric_sequence_sum a r 3000 = 500 →
  geometric_sequence_sum a r 6000 = 950 →
  geometric_sequence_sum a r 9000 = 1355 :=
by
  intros h1 h2
  sorry

end sum_first_9000_terms_l210_210885


namespace smallest_n_l210_210665

theorem smallest_n (n : ℕ) :
  (∀ m : ℤ, 0 < m ∧ m < 2001 →
    ∃ k : ℤ, (m : ℚ) / 2001 < (k : ℚ) / n ∧ (k : ℚ) / n < (m + 1 : ℚ) / 2002) ↔ n = 4003 :=
by
  sorry

end smallest_n_l210_210665


namespace unique_common_element_l210_210813

variable (A B : Set ℝ)
variable (a : ℝ)

theorem unique_common_element :
  A = {1, 3, a} → 
  B = {4, 5} →
  A ∩ B = {4} →
  a = 4 := 
by
  intro hA hB hAB
  sorry

end unique_common_element_l210_210813


namespace area_of_ABCD_l210_210692

theorem area_of_ABCD (area_AMOP area_CNOQ : ℝ) 
  (h1: area_AMOP = 8) (h2: area_CNOQ = 24.5) : 
  ∃ (area_ABCD : ℝ), area_ABCD = 60.5 :=
by
  sorry

end area_of_ABCD_l210_210692


namespace valid_song_distribution_l210_210619

open Finset

noncomputable def count_ways : ℕ :=
  let AB := {1}  -- At least one song liked by Amy and Beth but not Jo
  let BC := {2}  -- At least one song liked by Beth and Jo but not Amy
  let CA := {3}  -- At least one song liked by Jo and Amy but not Beth
  let remaining_songs := {4, 5}
  let choices := {AB, BC, CA, ∅, {4}, {5}}
  (4^2 + 4 * 3) -- Case 1: Remaining two in {N, A, B, C}; Case 2: One in {N, A, B, C} and one more in {AB, BC, CA}

theorem valid_song_distribution :
  count_ways = 28 :=
by {
  sorry
}

end valid_song_distribution_l210_210619


namespace sin_eq_product_one_eighth_l210_210803

open Real

theorem sin_eq_product_one_eighth :
  (∀ (n k m : ℕ), 1 ≤ n → n ≤ 5 → 1 ≤ k → k ≤ 5 → 1 ≤ m → m ≤ 5 →
    sin (π * n / 12) * sin (π * k / 12) * sin (π * m / 12) = 1 / 8) ↔ (n = 2 ∧ k = 2 ∧ m = 2) := by
  sorry

end sin_eq_product_one_eighth_l210_210803


namespace percentage_40_number_l210_210681

theorem percentage_40_number (x y z P : ℝ) (hx : x = 93.75) (hy : y = 0.40 * x) (hz : z = 6) (heq : (P / 100) * y = z) :
  P = 16 :=
sorry

end percentage_40_number_l210_210681


namespace sector_area_angle_1_sector_max_area_l210_210045

-- The definition and conditions
variable (c : ℝ) (r l : ℝ)

-- 1) Proof that the area of the sector when the central angle is 1 radian is c^2 / 18
-- given 2r + l = c
theorem sector_area_angle_1 (h : 2 * r + l = c) (h1: l = r) :
  (1/2 * l * r = (c^2 / 18)) :=
by sorry

-- 2) Proof that the central angle that maximizes the area is 2 radians and the maximum area is c^2 / 16
-- given 2r + l = c
theorem sector_max_area (h : 2 * r + l = c) :
  ∃ l r, 2 * r = l ∧ 1/2 * l * r = (c^2 / 16) :=
by sorry

end sector_area_angle_1_sector_max_area_l210_210045


namespace four_drivers_suffice_l210_210150

theorem four_drivers_suffice
  (one_way_trip_time : ℕ := 160) -- in minutes
  (round_trip_time : ℕ := 320) -- in minutes
  (rest_time : ℕ := 60) -- in minutes
  (time_A_returns : ℕ := 760) -- 12:40 PM in minutes from midnight
  (time_A_next_start : ℕ := 820) -- 1:40 PM in minutes from midnight
  (time_D_departs : ℕ := 785) -- 1:05 PM in minutes from midnight
  (time_A_fifth_depart : ℕ := 970) -- 4:10 PM in minutes from midnight
  (time_B_returns : ℕ := 960) -- 4:00 PM in minutes from midnight
  (time_B_sixth_depart : ℕ := 1050) -- 5:30 PM in minutes from midnight
  (time_A_fifth_complete: ℕ := 1290) -- 9:30 PM in minutes from midnight
  : 4_drivers_sufficient : ℕ :=
    if time_A_fifth_complete = 1290 then 1 else 0
-- The theorem states that if the calculated trip completion time is 9:30 PM, then 4 drivers are sufficient.
  sorry

end four_drivers_suffice_l210_210150


namespace leonard_younger_than_nina_by_4_l210_210399

variable (L N J : ℕ)

-- Conditions based on conditions from the problem
axiom h1 : L = 6
axiom h2 : N = 1 / 2 * J
axiom h3 : L + N + J = 36

-- Statement to prove
theorem leonard_younger_than_nina_by_4 : N - L = 4 :=
by 
  sorry

end leonard_younger_than_nina_by_4_l210_210399


namespace opposite_face_is_D_l210_210944

-- Define the six faces
inductive Face
| A | B | C | D | E | F

open Face

-- Define the adjacency relation
def is_adjacent (x y : Face) : Prop :=
(y = B ∧ x = A) ∨ (y = F ∧ x = A) ∨ (y = C ∧ x = A) ∨ (y = E ∧ x = A)

-- Define the problem statement in Lean
theorem opposite_face_is_D : 
  (∀ (x : Face), is_adjacent A x ↔ x = B ∨ x = F ∨ x = C ∨ x = E) →
  (¬ (is_adjacent A D)) →
  True :=
by
  intro adj_relation non_adj_relation
  sorry

end opposite_face_is_D_l210_210944


namespace number_with_specific_places_l210_210317

theorem number_with_specific_places :
  ∃ (n : Real), 
    (n / 10 % 10 = 6) ∧ -- tens place
    (n / 1 % 10 = 0) ∧  -- ones place
    (n * 10 % 10 = 0) ∧  -- tenths place
    (n * 100 % 10 = 6) →  -- hundredths place
    n = 60.06 :=
by
  sorry

end number_with_specific_places_l210_210317


namespace find_n_solution_l210_210017

theorem find_n_solution : ∃ n : ℤ, (1 / (n + 1 : ℝ) + 2 / (n + 1 : ℝ) + (n : ℝ) / (n + 1 : ℝ) = 3) :=
by
  use 0
  sorry

end find_n_solution_l210_210017


namespace maximum_b_value_l210_210208

noncomputable def f (a x : ℝ) := (1 / 2) * x ^ 2 + a * x
noncomputable def g (a b x : ℝ) := 2 * a ^ 2 * Real.log x + b

theorem maximum_b_value (a b : ℝ) (h_a : 0 < a) :
  (∃ x : ℝ, f a x = g a b x ∧ (deriv (f a) x = deriv (g a b) x))
  → b ≤ Real.exp (1 / 2) := 
sorry

end maximum_b_value_l210_210208


namespace angle_E_measure_l210_210927

theorem angle_E_measure (H F G E : ℝ) 
  (h1 : E = 2 * F) (h2 : F = 2 * G) (h3 : G = 1.25 * H) 
  (h4 : E + F + G + H = 360) : E = 150 := by
  sorry

end angle_E_measure_l210_210927


namespace total_profit_correct_l210_210752

noncomputable def total_profit (a b c : ℕ) (c_share : ℕ) : ℕ :=
  let ratio := a + b + c
  let part_value := c_share / c
  ratio * part_value

theorem total_profit_correct (h_a : ℕ := 5000) (h_b : ℕ := 8000) (h_c : ℕ := 9000) (h_c_share : ℕ := 36000) :
  total_profit h_a h_b h_c h_c_share = 88000 :=
by
  sorry

end total_profit_correct_l210_210752


namespace solve_for_a_l210_210199

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if h : x >= 0 then 4^x else 2^(a - x)

theorem solve_for_a (a : ℝ) (h : a ≠ 1) (h_f_eq : f a (1 - a) = f a (a - 1)) : a = 1 / 2 :=
by
  sorry

end solve_for_a_l210_210199


namespace find_k_exists_p3_p5_no_number_has_p2_and_p4_l210_210980

def has_prop_pk (n k : ℕ) : Prop := ∃ lst : List ℕ, (∀ x ∈ lst, x > 1) ∧ (lst.length = k) ∧ (lst.prod = n)

theorem find_k_exists_p3_p5 :
  ∃ (k : ℕ), (k = 3) ∧ ∃ (n : ℕ), has_prop_pk n k ∧ has_prop_pk n (k + 2) :=
by {
  sorry
}

theorem no_number_has_p2_and_p4 :
  ¬ ∃ (n : ℕ), has_prop_pk n 2 ∧ has_prop_pk n 4 :=
by {
  sorry
}

end find_k_exists_p3_p5_no_number_has_p2_and_p4_l210_210980


namespace average_test_score_45_percent_l210_210374

theorem average_test_score_45_percent (x : ℝ) 
  (h1 : 0.45 * x + 0.50 * 78 + 0.05 * 60 = 84.75) : 
  x = 95 :=
by sorry

end average_test_score_45_percent_l210_210374


namespace parallel_lines_parallel_lines_solution_l210_210376

theorem parallel_lines (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0) ∧ (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) → a = -1 ∨ a = 2 :=
sorry

theorem parallel_lines_solution (a : ℝ) :
  (∀ x y : ℝ, a * x + 2 * y + 6 = 0) ∧ (∀ x y : ℝ, x + (a - 1) * y + (a^2 - 1) = 0) ∧ 
  ((a = -1 → ∀ x y : ℝ, a * x + 2 * y + 6 = 0 ∧ (x + (a - 1) * y + (a^2 - 1) = 0)) ∨ 
  (a = 2 → ∀ x y : ℝ, a * x + 2 * y + 6 = 0 ∧ (x + (a - 1) * y + (a^2 - 1) = 0))) :=
sorry

end parallel_lines_parallel_lines_solution_l210_210376


namespace length_of_box_l210_210420

theorem length_of_box (v : ℝ) (w : ℝ) (h : ℝ) (l : ℝ) (conversion_factor : ℝ) (v_gallons : ℝ)
  (h_inch : ℝ) (conversion_inches_feet : ℝ) :
  v_gallons / conversion_factor = v → 
  h_inch / conversion_inches_feet = h →
  v = l * w * h →
  w = 25 →
  v_gallons = 4687.5 →
  conversion_factor = 7.5 →
  h_inch = 6 →
  conversion_inches_feet = 12 →
  l = 50 :=
by
  sorry

end length_of_box_l210_210420


namespace average_growth_rate_inequality_l210_210226

theorem average_growth_rate_inequality (p q x : ℝ) (h₁ : (1+x)^2 = (1+p)*(1+q)) (h₂ : p ≠ q) :
  x < (p + q) / 2 :=
sorry

end average_growth_rate_inequality_l210_210226


namespace no_fermat_in_sequence_l210_210694

def sequence_term (n k : ℕ) : ℕ :=
  (k - 2) * n * (n - 1) / 2 + n

def is_fermat_number (a : ℕ) : Prop :=
  ∃ m : ℕ, a = 2^(2^m) + 1

theorem no_fermat_in_sequence (k n : ℕ) (hk : k > 2) (hn : n > 2) :
  ¬ is_fermat_number (sequence_term n k) :=
sorry

end no_fermat_in_sequence_l210_210694


namespace smallest_a_has_50_perfect_squares_l210_210022

def is_perfect_square (x : ℕ) : Prop :=
  ∃ n : ℕ, n * n = x

def count_perfect_squares_in_interval (low high : ℕ) : ℕ :=
  (List.range' low (high - low)).filter is_perfect_square |>.length

theorem smallest_a_has_50_perfect_squares : (a : ℕ) (h : a = 4486) →
  count_perfect_squares_in_interval (a + 1) (3 * a) = 50 :=
by
  intros a h
  rw h
  sorry

end smallest_a_has_50_perfect_squares_l210_210022


namespace salad_quantity_percentage_difference_l210_210580

noncomputable def Tom_rate := 2/3 -- Tom's rate (lb/min)
noncomputable def Tammy_rate := 3/2 -- Tammy's rate (lb/min)
noncomputable def Total_salad := 65 -- Total salad chopped (lb)
noncomputable def Time_to_chop := Total_salad / (Tom_rate + Tammy_rate) -- Time to chop 65 lb (min)
noncomputable def Tom_chop := Time_to_chop * Tom_rate -- Total chopped by Tom (lb)
noncomputable def Tammy_chop := Time_to_chop * Tammy_rate -- Total chopped by Tammy (lb)
noncomputable def Percent_difference := (Tammy_chop - Tom_chop) / Tom_chop * 100 -- Percent difference

theorem salad_quantity_percentage_difference : Percent_difference = 125 :=
by
  sorry

end salad_quantity_percentage_difference_l210_210580


namespace combination_eight_choose_five_l210_210503

theorem combination_eight_choose_five : nat.choose 8 5 = 56 :=
by sorry

end combination_eight_choose_five_l210_210503


namespace compare_powers_l210_210559

theorem compare_powers : 2^24 < 10^8 ∧ 10^8 < 5^12 :=
by 
  -- proofs omitted
  sorry

end compare_powers_l210_210559


namespace find_length_AB_l210_210106

-- Define the parabola y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the line y = x - 1
def line (x y : ℝ) : Prop := y = x - 1

-- Define the intersection length |AB|
noncomputable def length_AB (x1 x2 : ℝ) : ℝ := x1 + x2 + 2

-- Main theorem statement
theorem find_length_AB (x1 x2 : ℝ)
  (h₁ : parabola x1 (x1 - 1))
  (h₂ : parabola x2 (x2 - 1))
  (hx : x1 + x2 = 6) :
  length_AB x1 x2 = 8 := sorry

end find_length_AB_l210_210106


namespace x3_y3_sum_sq_sq_leq_4a10_equality_holds_when_x_eq_y_l210_210212

-- Conditions: x, y are positive real numbers and x + y = 2a
variables {x y a : ℝ}
variable (hxy : x + y = 2 * a)
variable (hx_pos : 0 < x)
variable (hy_pos : 0 < y)

-- Math proof problem: Prove the inequality
theorem x3_y3_sum_sq_sq_leq_4a10 : 
  x^3 * y^3 * (x^2 + y^2)^2 ≤ 4 * a^10 :=
by sorry

-- Equality condition: Equality holds when x = y
theorem equality_holds_when_x_eq_y (h : x = y) :
  x^3 * y^3 * (x^2 + y^2)^2 = 4 * a^10 :=
by sorry

end x3_y3_sum_sq_sq_leq_4a10_equality_holds_when_x_eq_y_l210_210212


namespace solve_proof_problem_l210_210959

noncomputable def proof_problem (alpha : ℝ) :=
  𝚜𝚎𝚌𝚘𝚗𝚍𝚀𝚞𝚊𝚍𝚛𝚊𝚗𝚝 : ∃ k : ℤ, α ∈ set.Ioo (k * (2 * Real.pi) + Real.pi / 2) (k * (2 * Real.pi) + Real.pi) ∧
  (h : Real.sin (alpha + Real.pi / 6) = 1 / 3),
  Real.sin (2 * alpha + Real.pi / 3) = -4 * Real.sqrt 2 / 9

theorem solve_proof_problem (alpha : ℝ) (h1 : ∃ k : ℤ, alpha ∈ set.Ioo (k * (2 * Real.pi) + Real.pi / 2) (k * (2 * Real.pi) + Real.pi)) 
                             (h2 : Real.sin (alpha + Real.pi / 6) = 1 / 3) :
  Real.sin (2 * alpha + Real.pi / 3) = -4 * Real.sqrt 2 / 9 :=
sorry

end solve_proof_problem_l210_210959


namespace jerry_total_logs_l210_210393

-- Given conditions
def pine_logs_per_tree := 80
def maple_logs_per_tree := 60
def walnut_logs_per_tree := 100

def pine_trees_cut := 8
def maple_trees_cut := 3
def walnut_trees_cut := 4

-- Formulate the problem
theorem jerry_total_logs : 
  pine_logs_per_tree * pine_trees_cut + 
  maple_logs_per_tree * maple_trees_cut + 
  walnut_logs_per_tree * walnut_trees_cut = 1220 := 
by 
  -- Placeholder for the actual proof
  sorry

end jerry_total_logs_l210_210393


namespace find_e_l210_210004

noncomputable def Q (x : ℝ) (d e f : ℝ) : ℝ := 3 * x^3 + d * x^2 + e * x + f

theorem find_e (d e f : ℝ) (h1 : 3 + d + e + f = -6)
  (h2 : - f / 3 = -6)
  (h3 : 9 = f)
  (h4 : - d / 3 = -18) : e = -72 :=
by
  sorry

end find_e_l210_210004


namespace avg_visitors_is_correct_l210_210305

-- Define the number of days in the month
def days_in_month : ℕ := 30

-- Define the average number of visitors on Sundays
def avg_visitors_sunday : ℕ := 510

-- Define the average number of visitors on other days
def avg_visitors_other_days : ℕ := 240

-- Define the number of Sundays in the month
def sundays_in_month : ℕ := 4

-- Define the number of other days in the month
def other_days_in_month : ℕ := days_in_month - sundays_in_month

-- Define the total visitors on Sundays
def total_visitors_sundays : ℕ := sundays_in_month * avg_visitors_sunday

-- Define the total visitors on other days
def total_visitors_other_days : ℕ := other_days_in_month * avg_visitors_other_days

-- Define the total number of visitors in the month
def total_visitors : ℕ := total_visitors_sundays + total_visitors_other_days

-- Define the average number of visitors per day
def avg_visitors_per_day : ℕ := total_visitors / days_in_month

-- The theorem to prove
theorem avg_visitors_is_correct : avg_visitors_per_day = 276 := by
  sorry

end avg_visitors_is_correct_l210_210305


namespace work_days_together_l210_210605

theorem work_days_together (A B : Type) (R_A R_B : ℝ) 
  (h1 : R_A = 1/2 * R_B) (h2 : R_B = 1 / 27) : 
  (1 / (R_A + R_B)) = 18 :=
by
  sorry

end work_days_together_l210_210605


namespace sum_of_squares_not_divisible_by_5_or_13_l210_210243

-- Definition of the set T
def T (n : ℤ) : ℤ :=
  (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2

-- The theorem to prove
theorem sum_of_squares_not_divisible_by_5_or_13 (n : ℤ) :
  ¬ (T n % 5 = 0) ∧ ¬ (T n % 13 = 0) :=
by
  sorry

end sum_of_squares_not_divisible_by_5_or_13_l210_210243


namespace expression_varies_l210_210626

variables {x : ℝ}

noncomputable def expression (x : ℝ) : ℝ :=
  (3 * x^2 + 4 * x - 5) / ((x + 1) * (x - 3)) - (8 + x) / ((x + 1) * (x - 3))

theorem expression_varies (h1 : x ≠ -1) (h2 : x ≠ 3) : 
  ∃ x₀ x₁ : ℝ, x₀ ≠ x₁ ∧ 
  expression x₀ ≠ expression x₁ :=
by
  sorry

end expression_varies_l210_210626


namespace solution_when_a_is_1_solution_for_arbitrary_a_l210_210475

-- Let's define the inequality and the solution sets
def inequality (a x : ℝ) : Prop :=
  ((a + 1) * x - 3) / (x - 1) < 1

def solutionSet_a_eq_1 (x : ℝ) : Prop :=
  1 < x ∧ x < 2

def solutionSet_a_eq_0 (x : ℝ) : Prop :=
  1 < x
  
def solutionSet_a_lt_0 (a x : ℝ) : Prop :=
  x < (2 / a) ∨ 1 < x

def solutionSet_0_lt_a_lt_2 (a x : ℝ) : Prop :=
  1 < x ∧ x < (2 / a)

def solutionSet_a_eq_2 : Prop :=
  false

def solutionSet_a_gt_2 (a x : ℝ) : Prop :=
  (2 / a) < x ∧ x < 1

-- Prove the solution for a = 1
theorem solution_when_a_is_1 : ∀ (x : ℝ), inequality 1 x ↔ solutionSet_a_eq_1 x :=
by sorry

-- Prove the solution for arbitrary real number a
theorem solution_for_arbitrary_a : ∀ (a x : ℝ),
  (a < 0 → inequality a x ↔ solutionSet_a_lt_0 a x) ∧
  (a = 0 → inequality a x ↔ solutionSet_a_eq_0 x) ∧
  (0 < a ∧ a < 2 → inequality a x ↔ solutionSet_0_lt_a_lt_2 a x) ∧
  (a = 2 → inequality a x → solutionSet_a_eq_2) ∧
  (a > 2 → inequality a x ↔ solutionSet_a_gt_2 a x) :=
by sorry

end solution_when_a_is_1_solution_for_arbitrary_a_l210_210475


namespace maximum_at_vertex_l210_210807

def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem maximum_at_vertex (a b c x_0 : ℝ) (h_a : a < 0) (h_x0 : 2 * a * x_0 + b = 0) :
  ∀ x : ℝ, quadratic_function a b c x ≤ quadratic_function a b c x_0 :=
sorry

end maximum_at_vertex_l210_210807


namespace p_and_q_and_not_not_p_or_q_l210_210490

theorem p_and_q_and_not_not_p_or_q (p q : Prop) (h1 : ¬(p ∧ q)) (h2 : ¬(¬p ∨ q)) : p ∧ ¬q :=
by
  sorry

end p_and_q_and_not_not_p_or_q_l210_210490


namespace child_grandmother_ratio_l210_210875

def grandmother_weight (G D C : ℝ) : Prop :=
  G + D + C = 160

def daughter_child_weight (D C : ℝ) : Prop :=
  D + C = 60

def daughter_weight (D : ℝ) : Prop :=
  D = 40

theorem child_grandmother_ratio (G D C : ℝ) (h1 : grandmother_weight G D C) (h2 : daughter_child_weight D C) (h3 : daughter_weight D) :
  C / G = 1 / 5 :=
sorry

end child_grandmother_ratio_l210_210875


namespace positive_difference_l210_210900

-- Define the conditions given in the problem
def conditions (x y : ℝ) : Prop :=
  x + y = 40 ∧ 3 * y - 4 * x = 20

-- The theorem to prove
theorem positive_difference (x y : ℝ) (h : conditions x y) : abs (y - x) = 11.42 :=
by
  sorry -- proof omitted

end positive_difference_l210_210900


namespace average_price_per_racket_l210_210929

theorem average_price_per_racket (total_amount : ℕ) (pairs_sold : ℕ) (expected_average : ℚ) 
  (h1 : total_amount = 637) (h2 : pairs_sold = 65) : 
  expected_average = total_amount / pairs_sold := 
by
  sorry

end average_price_per_racket_l210_210929


namespace num_white_black_balls_prob_2_black_balls_dist_exp_black_balls_l210_210920

-- Problem 1: Number of white and black balls
theorem num_white_black_balls (n m : ℕ) (h1 : n + m = 10)
  (h2 : (10 - m) = 4) : n = 4 ∧ m = 6 :=
by sorry

-- Problem 2: Probability of drawing exactly 2 black balls with replacement
theorem prob_2_black_balls (p_black_draw : ℕ → ℕ → ℚ)
  (h1 : ∀ n m, p_black_draw n m = (6/10)^(n-m) * (4/10)^m)
  (h2 : p_black_draw 2 3 = 54/125) : p_black_draw 2 3 = 54 / 125 :=
by sorry

-- Problem 3: Distribution and Expectation of number of black balls drawn without replacement
theorem dist_exp_black_balls (prob_X : ℕ → ℚ) (expect_X : ℚ)
  (h1 : prob_X 0 = 2/15) (h2 : prob_X 1 = 8/15) (h3 : prob_X 2 = 1/3)
  (h4 : expect_X = 6 / 5) : ∀ k, prob_X k = match k with
    | 0 => 2/15
    | 1 => 8/15
    | 2 => 1/3
    | _ => 0 :=
by sorry

end num_white_black_balls_prob_2_black_balls_dist_exp_black_balls_l210_210920


namespace inverse_function_correct_l210_210126

noncomputable def f (x : ℝ) : ℝ := 3 - 7 * x

noncomputable def g (x : ℝ) : ℝ := (3 - x) / 7

theorem inverse_function_correct : ∀ x : ℝ, f (g x) = x ∧ g (f x) = x :=
by
  intro x
  sorry

end inverse_function_correct_l210_210126


namespace arithmetic_expression_value_l210_210451

theorem arithmetic_expression_value :
  15 * 36 + 15 * 3^3 = 945 :=
by
  sorry

end arithmetic_expression_value_l210_210451


namespace two_rides_combinations_l210_210702

-- Define the number of friends
def num_friends : ℕ := 7

-- Define the size of the group for one ride
def ride_group_size : ℕ := 4

-- Define the number of combinations of choosing 'ride_group_size' out of 'num_friends'
def combinations_first_ride : ℕ := Nat.choose num_friends ride_group_size

-- Define the number of friends left for the second ride
def remaining_friends : ℕ := num_friends - ride_group_size

-- Define the number of combinations of choosing 'ride_group_size' out of 'remaining_friends' friends
def combinations_second_ride : ℕ := Nat.choose remaining_friends ride_group_size

-- Define the total number of possible combinations for two rides
def total_combinations : ℕ := combinations_first_ride * combinations_second_ride

-- The final theorem stating the total number of combinations is equal to 525
theorem two_rides_combinations : total_combinations = 525 := by
  -- Placeholder for proof
  sorry

end two_rides_combinations_l210_210702


namespace retail_price_l210_210771

theorem retail_price (W M : ℝ) (hW : W = 20) (hM : M = 80) : W + (M / 100) * W = 36 := by
  sorry

end retail_price_l210_210771


namespace parabola_focus_l210_210422

theorem parabola_focus (h : ∀ x y : ℝ, y ^ 2 = -12 * x → True) : (-3, 0) = (-3, 0) :=
  sorry

end parabola_focus_l210_210422


namespace smallest_n_for_inequality_l210_210663

theorem smallest_n_for_inequality :
  ∃ (n : ℕ), n = 4003 ∧ (∀ m : ℤ, (0 < m ∧ m < 2001) →
  ∃ k : ℤ, (m / 2001 : ℚ) < (k / n : ℚ) ∧ (k / n : ℚ) < ((m + 1) / 2002 : ℚ)) :=
sorry

end smallest_n_for_inequality_l210_210663


namespace domain_of_f_l210_210339

theorem domain_of_f (c : ℝ) :
  (∀ x : ℝ, -7 * x^2 + 5 * x + c ≠ 0) ↔ c < -25 / 28 :=
by
  sorry

end domain_of_f_l210_210339


namespace quadratic_function_graph_opens_downwards_l210_210302

-- Definition of the quadratic function
def quadratic_function (x : ℝ) : ℝ := -x^2 + 3

-- The problem statement to prove
theorem quadratic_function_graph_opens_downwards :
  (∀ x : ℝ, (quadratic_function (x + 1) - quadratic_function x) < (quadratic_function x - quadratic_function (x - 1))) :=
by
  -- Proof omitted
  sorry

end quadratic_function_graph_opens_downwards_l210_210302


namespace arithmetic_sequence_a10_gt_0_l210_210353

variable {α : Type*} [LinearOrderedField α]

-- Definitions of the conditions
def arithmetic_sequence (a : ℕ → α) := ∀ n1 n2, a n1 - a n2 = (n1 - n2) * (a 1 - a 0)
def a9_lt_0 (a : ℕ → α) := a 9 < 0
def a1_add_a18_gt_0 (a : ℕ → α) := a 1 + a 18 > 0

-- The proof statement
theorem arithmetic_sequence_a10_gt_0 
  (a : ℕ → α) 
  (h_arith : arithmetic_sequence a) 
  (h_a9 : a9_lt_0 a) 
  (h_a1_a18 : a1_add_a18_gt_0 a) : 
  a 10 > 0 := 
sorry

end arithmetic_sequence_a10_gt_0_l210_210353


namespace tangent_perpendicular_l210_210969

open Real

def curve (m : ℝ) (x : ℝ) : ℝ := exp x - m * x + 1

theorem tangent_perpendicular (m : ℝ) :
  (∃ x : ℝ, deriv (curve m) x = deriv (λ x, exp x) x ∧ 
    deriv (curve m) x = -1 / deriv (λ x, exp x) x) → 
  m > 1 / exp 1 := by
  sorry

end tangent_perpendicular_l210_210969


namespace sum_of_first_9000_terms_of_geometric_sequence_l210_210895

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_9000_terms_of_geometric_sequence 
  (a r : ℝ)
  (h₁ : geometric_sum a r 3000 = 500)
  (h₂ : geometric_sum a r 6000 = 950)
  : geometric_sum a r 9000 = 1355 :=
sorry

end sum_of_first_9000_terms_of_geometric_sequence_l210_210895


namespace hazel_ratio_is_correct_father_ratio_is_correct_l210_210822

variables (hazelA hazelB fatherA fatherB : ℕ)
variables (hazelRatio fatherRatio : ℚ)

-- Conditions for Hazel
def hazel_conditions : Prop :=
  hazelA = 48 ∧ hazelB = 32

-- Condition for Hazel's ratio
def hazel_ratio_conditions : Prop :=
  hazelRatio = hazelA / hazelB

-- Conditions for Hazel's father
def father_conditions : Prop :=
  fatherA = 46 ∧ fatherB = 24

-- Condition for Hazel's father's ratio
def father_ratio_conditions : Prop :=
  fatherRatio = fatherA / fatherB

theorem hazel_ratio_is_correct (hA : hazelA = 48) (hB : hazelB = 32) : hazelRatio = 3 / 2 :=
  sorry

theorem father_ratio_is_correct (fA : fatherA = 46) (fB : fatherB = 24) : fatherRatio = 23 / 12 :=
  sorry

end hazel_ratio_is_correct_father_ratio_is_correct_l210_210822


namespace minimum_value_of_f_l210_210730

noncomputable def f (x : ℝ) : ℝ := x + 1 / (x - 2)

theorem minimum_value_of_f :
  (∀ x : ℝ, x > 2 → f x ≥ 4) ∧ (∃ x : ℝ, x > 2 ∧ f x = 4) :=
by {
  sorry
}

end minimum_value_of_f_l210_210730


namespace willie_initial_bananas_l210_210746

/-- Given that Willie will have 13 bananas, we need to prove that the initial number of bananas Willie had was some specific number X. --/
theorem willie_initial_bananas (initial_bananas : ℕ) (final_bananas : ℕ) 
    (h : final_bananas = 13) : initial_bananas = initial_bananas :=
by
  sorry

end willie_initial_bananas_l210_210746


namespace race_distance_correct_l210_210265

noncomputable def solve_race_distance : ℝ :=
  let Vq := 1          -- assume Vq as some positive real number (could be normalized to 1 for simplicity)
  let Vp := 1.20 * Vq  -- P runs 20% faster
  let head_start := 300 -- Q's head start in meters
  let Dp := 1800       -- distance P runs

  -- time taken by P
  let time_p := Dp / Vp
  -- time taken by Q, given it has a 300 meter head start
  let Dq := Dp - head_start
  let time_q := Dq / Vq

  Dp

theorem race_distance_correct :
  let Vq := 1          -- assume Vq as some positive real number (could be normalized to 1 for simplicity)
  let Vp := 1.20 * Vq  -- P runs 20% faster
  let head_start := 300 -- Q's head start in meters
  let Dp := 1800       -- distance P runs
  -- time taken by P
  let time_p := Dp / Vp
  -- time taken by Q, given it has a 300 meter head start
  let Dq := Dp - head_start
  let time_q := Dq / Vq

  time_p = time_q := by
  sorry

end race_distance_correct_l210_210265


namespace division_multiplication_eval_l210_210342

theorem division_multiplication_eval : (18 / (5 + 2 - 3)) * 4 = 18 := 
by
  sorry

end division_multiplication_eval_l210_210342


namespace potato_bag_weight_l210_210587

-- Defining the weight of the bag of potatoes as a variable W
variable (W : ℝ)

-- Given condition: The weight of the bag is described by the equation
def weight_condition (W : ℝ) := W = 12 / (W / 2)

-- Proving the weight of the bag of potatoes is 12 lbs:
theorem potato_bag_weight : weight_condition W → W = 12 :=
by
  sorry

end potato_bag_weight_l210_210587


namespace jamesOreos_count_l210_210075

noncomputable def jamesOreos (jordanOreos : ℕ) : ℕ := 4 * jordanOreos + 7

theorem jamesOreos_count (J : ℕ) (h1 : J + jamesOreos J = 52) : jamesOreos J = 43 :=
by
  sorry

end jamesOreos_count_l210_210075


namespace average_temp_tues_to_fri_l210_210415

theorem average_temp_tues_to_fri (T W Th : ℕ) 
  (h1: (42 + T + W + Th) / 4 = 48) 
  (mon: 42 = 42) 
  (fri: 10 = 10) :
  (T + W + Th + 10) / 4 = 40 := by
  sorry

end average_temp_tues_to_fri_l210_210415


namespace factor_81_minus_27_x_cubed_l210_210182

theorem factor_81_minus_27_x_cubed (x : ℝ) : 
  81 - 27 * x ^ 3 = 27 * (3 - x) * (9 + 3 * x + x ^ 2) :=
by sorry

end factor_81_minus_27_x_cubed_l210_210182


namespace find_original_number_l210_210377

theorem find_original_number (x : ℕ) (h1 : 10 * x + 9 + 2 * x = 633) : x = 52 :=
by
  sorry

end find_original_number_l210_210377


namespace min_value_exp_sum_eq_4sqrt2_l210_210268

theorem min_value_exp_sum_eq_4sqrt2 {a b : ℝ} (h : a + b = 3) : 2^a + 2^b ≥ 4 * Real.sqrt 2 :=
by
  sorry

end min_value_exp_sum_eq_4sqrt2_l210_210268


namespace problem_statement_l210_210706

noncomputable def x : ℝ := (3 + Real.sqrt 8) ^ 30
noncomputable def n : ℤ := Int.floor x
noncomputable def f : ℝ := x - n

theorem problem_statement : x * (1 - f) = 1 := 
by 
  sorry

end problem_statement_l210_210706


namespace cricket_initial_average_l210_210726

theorem cricket_initial_average (A : ℕ) (h1 : ∀ A, A * 20 + 137 = 21 * (A + 5)) : A = 32 := by
  -- assumption and proof placeholder
  sorry

end cricket_initial_average_l210_210726


namespace maxim_birth_probability_l210_210250

open Nat

def interval_days (start_date end_date : ℕ) : ℕ :=
  end_date - start_date + 1

def total_days_2007_2008 : ℕ :=
  interval_days 245 2735 -- total days from Sep 2, 2007, to Aug 31, 2008

def days_in_2008 : ℕ :=
  interval_days 305 548  -- total days from Jan 1, 2008, to Aug 31, 2008

def probability_born_in_2008 : ℚ :=
  (days_in_2008 : ℚ) / (total_days_2007_2008 : ℚ)

theorem maxim_birth_probability: probability_born_in_2008 = 244 / 365 := 
  sorry

end maxim_birth_probability_l210_210250


namespace Josephine_sold_10_liters_l210_210259

def milk_sold (n1 n2 n3 : ℕ) (v1 v2 v3 : ℝ) : ℝ :=
  (v1 * n1) + (v2 * n2) + (v3 * n3)

theorem Josephine_sold_10_liters :
  milk_sold 3 2 5 2 0.75 0.5 = 10 :=
by
  sorry

end Josephine_sold_10_liters_l210_210259


namespace lake_circumference_ratio_l210_210313

theorem lake_circumference_ratio 
    (D C : ℝ) 
    (hD : D = 100) 
    (hC : C = 314) : 
    C / D = 3.14 := 
sorry

end lake_circumference_ratio_l210_210313


namespace value_of_expression_l210_210351

theorem value_of_expression (x : ℝ) (h : |x| = x + 2) : 19 * x ^ 99 + 3 * x + 27 = 5 :=
by
  have h1: x ≥ -2 := sorry
  have h2: x = -1 := sorry
  sorry

end value_of_expression_l210_210351


namespace middle_group_frequency_l210_210384

theorem middle_group_frequency (capacity : ℕ) (n_rectangles : ℕ) (A_mid A_other : ℝ) 
  (h_capacity : capacity = 300)
  (h_rectangles : n_rectangles = 9)
  (h_areas : A_mid = 1 / 5 * A_other)
  (h_total_area : A_mid + A_other = 1) : 
  capacity * A_mid = 50 := by
  sorry

end middle_group_frequency_l210_210384


namespace union_inter_distrib_inter_union_distrib_l210_210306

section
variables {α : Type*} (A B C : Set α)

-- Problem (a)
theorem union_inter_distrib : A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C) :=
sorry

-- Problem (b)
theorem inter_union_distrib : A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) :=
sorry
end

end union_inter_distrib_inter_union_distrib_l210_210306


namespace maximum_profit_l210_210761

noncomputable def R (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then
  10.8 - (1/30) * x^2
else
  108 / x - 1000 / (3 * x^2)

noncomputable def W (x : ℝ) : ℝ :=
if h : 0 < x ∧ x ≤ 10 then
  x * R x - (10 + 2.7 * x)
else
  x * R x - (10 + 2.7 * x)

theorem maximum_profit : 
  ∃ x : ℝ, (0 < x ∧ x ≤ 10 → W x = 8.1 * x - (x^3 / 30) - 10) ∧ 
           (x > 10 → W x = 98 - 1000 / (3 * x) - 2.7 * x) ∧ 
           (∃ xmax : ℝ, xmax = 9 ∧ W 9 = 38.6) := 
sorry

end maximum_profit_l210_210761


namespace hyperbola_equation_correct_l210_210473

noncomputable def hyperbola_equation (a b : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (focal_len : Real.sqrt (a^2 + b^2) = 5) (asymptote_slope : b = 2 * a) :=
  (x y : ℝ) -> (x^2 / 5) - (y^2 / 20) = 1

theorem hyperbola_equation_correct {a b : ℝ} (a_pos : 0 < a) (b_pos : 0 < b) 
  (focal_len : Real.sqrt (a^2 + b^2) = 5) (asymptote_slope : b = 2 * a) :
  hyperbola_equation a b a_pos b_pos focal_len asymptote_slope :=
by {
  sorry
}

end hyperbola_equation_correct_l210_210473


namespace mario_hibiscus_l210_210406

def hibiscus_flowers (F : ℕ) : Prop :=
  let F2 := 2 * F
  let F3 := 4 * F2
  F + F2 + F3 = 22 → F = 2

theorem mario_hibiscus (F : ℕ) : hibiscus_flowers F :=
  sorry

end mario_hibiscus_l210_210406


namespace greatest_int_with_conditions_l210_210568

noncomputable def is_valid_integer (n : ℤ) : Prop :=
  0 < n ∧ n < 200 ∧ Int.gcd n 18 = 6

theorem greatest_int_with_conditions : 
  ∃ n, is_valid_integer n ∧ ∀ m, is_valid_integer m → m ≤ 174 :=
begin
  sorry
end

end greatest_int_with_conditions_l210_210568


namespace sin_div_one_minus_tan_eq_neg_three_fourths_l210_210961

variable (α : ℝ)

theorem sin_div_one_minus_tan_eq_neg_three_fourths
  (h : Real.sin (α - Real.pi / 4) = Real.sqrt 2 / 4) :
  (Real.sin α) / (1 - Real.tan α) = -3 / 4 := sorry

end sin_div_one_minus_tan_eq_neg_three_fourths_l210_210961


namespace distribution_ways_l210_210447

theorem distribution_ways :
  let friends := 12
  let problems := 6
  (friends ^ problems = 2985984) :=
by
  sorry

end distribution_ways_l210_210447


namespace expression_evaluates_to_2023_l210_210296

theorem expression_evaluates_to_2023 : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 1) / (2023 * 2024) = 2023 :=
by 
  sorry

end expression_evaluates_to_2023_l210_210296


namespace find_number_l210_210312

theorem find_number (x : ℝ) (h : 0.3 * x + 0.1 * 0.5 = 0.29) : x = 0.8 :=
by
  sorry

end find_number_l210_210312


namespace nice_string_properties_l210_210790

open List

-- Define what it means for a string to be nice.
def is_nice (s : List Char) : Prop :=
  (∀ c ∈ ['A'.. 'Z'], c ∈ s) ∧
  (∀ π : List Char, 
    Multiset.card ((subsequences s).filter (λ t, t = π)) = 
      Multiset.card ((subsequences s).filter (λ t, t = ['A'..'Z'])))

-- The main theorem to prove
theorem nice_string_properties :
  (∃ s : List Char, is_nice s) ∧ 
  (∀ s : List Char, is_nice s → length s ≥ 2022) :=
by
  sorry

end nice_string_properties_l210_210790


namespace quadratic_function_correct_value_l210_210470

noncomputable def quadratic_function_value (a b x x1 x2 : ℝ) :=
  a * x^2 + b * x + 5

theorem quadratic_function_correct_value
  (a b x1 x2 : ℝ)
  (h_a : a ≠ 0)
  (h_A : quadratic_function_value a b x1 x1 x2 = 2002)
  (h_B : quadratic_function_value a b x2 x1 x2 = 2002) :
  quadratic_function_value a b (x1 + x2) x1 x2 = 5 :=
by
  sorry

end quadratic_function_correct_value_l210_210470


namespace find_smallest_a_l210_210032

theorem find_smallest_a (n : ℕ) (a : ℕ) :
  (n^2 ≤ a) ∧ (a < (n+1)^2) →
  (∀ k : ℕ, (n^2 ≤ a) ∧ (a < k^2) ∧ (k^2 < 3 * a) → ((n+1)^2 ≤ k^2 ∧ k^2 ≤ (n+50)^2)) →
  ((n + 50)^2 < 3 * a) →
  a = 4486 :=
begin
  sorry
end

end find_smallest_a_l210_210032


namespace ratio_first_term_common_difference_l210_210948

theorem ratio_first_term_common_difference
  (a d : ℚ)
  (h : (15 / 2) * (2 * a + 14 * d) = 4 * (8 / 2) * (2 * a + 7 * d)) :
  a / d = -7 / 17 := 
by {
  sorry
}

end ratio_first_term_common_difference_l210_210948


namespace solve_equation1_solve_equation2_l210_210096

-- Problem for Equation (1)
theorem solve_equation1 (x : ℝ) : x * (x - 6) = 2 * (x - 8) → x = 4 := by
  sorry

-- Problem for Equation (2)
theorem solve_equation2 (x : ℝ) : (2 * x - 1)^2 + 3 * (2 * x - 1) + 2 = 0 → x = 0 ∨ x = -1 / 2 := by
  sorry

end solve_equation1_solve_equation2_l210_210096


namespace cone_lateral_surface_area_l210_210601

theorem cone_lateral_surface_area (r : ℝ) (theta : ℝ) (h_r : r = 3) (h_theta : theta = 90) : 
  let base_circumference := 2 * Real.pi * r
  let R := 12
  let lateral_surface_area := (1 / 2) * base_circumference * R 
  lateral_surface_area = 36 * Real.pi :=
by
  sorry

end cone_lateral_surface_area_l210_210601


namespace number_of_5_dollar_coins_l210_210736

-- Define the context and the proof problem
theorem number_of_5_dollar_coins (x y : ℕ) (h1 : x + y = 40) (h2 : 2 * x + 5 * y = 125) : y = 15 :=
by sorry

end number_of_5_dollar_coins_l210_210736


namespace probability_blue_is_approx_50_42_l210_210001

noncomputable def probability_blue_second_pick : ℚ :=
  let yellow := 30
  let green := yellow / 3
  let red := 2 * green
  let total_marbles := 120
  let blue := total_marbles - (yellow + green + red)
  let total_after_first_pick := total_marbles - 1
  let blue_probability := (blue : ℚ) / total_after_first_pick
  blue_probability * 100

theorem probability_blue_is_approx_50_42 :
  abs (probability_blue_second_pick - 50.42) < 0.005 := -- Approximately checking for equality due to possible floating-point precision issues
sorry

end probability_blue_is_approx_50_42_l210_210001


namespace value_of_m_l210_210839

theorem value_of_m (m x : ℝ) (h : x = 3) (h_eq : 3 * m - 2 * x = 6) : m = 4 := by
  -- Given x = 3
  subst h
  -- Now we have to show m = 4
  sorry

end value_of_m_l210_210839


namespace greatest_integer_gcd_l210_210564

theorem greatest_integer_gcd (n : ℕ) (h1 : n < 200) (h2 : gcd n 18 = 6) : n = 192 :=
sorry

end greatest_integer_gcd_l210_210564


namespace main_world_population_transition_l210_210171

noncomputable def world_population_reproduction_transition
  (developing_majority : Prop)
  (developing_transition : Prop)
  (developed_small : Prop)
  (developed_modern : Prop)
  (minor_impact : Prop) : Prop :=
  developing_majority ∧ developing_transition ∧ developed_small ∧ developed_modern ∧ minor_impact →
  "Traditional" = "Traditional" ∧ "Modern" = "Modern" →
  "Traditional" = "Traditional"

theorem main_world_population_transition
  (developing_majority : Prop)
  (developing_transition : Prop)
  (developed_small : Prop)
  (developed_modern : Prop)
  (minor_impact : Prop) :
  developing_majority ∧ developing_transition ∧ developed_small ∧ developed_modern ∧ minor_impact →
  "Traditional" = "Traditional" ∧ "Modern" = "Modern" →
  "Traditional" = "Traditional" :=
by
  sorry

end main_world_population_transition_l210_210171


namespace greatest_integer_less_than_200_with_gcd_18_l210_210574

theorem greatest_integer_less_than_200_with_gcd_18 : ∃ n: ℕ, n < 200 ∧ Nat.gcd n 18 = 6 ∧ ∀ m: ℕ, m < 200 ∧ Nat.gcd m 18 = 6 → m ≤ n := 
by
  sorry

end greatest_integer_less_than_200_with_gcd_18_l210_210574


namespace find_line_eqn_from_bisected_chord_l210_210043

noncomputable def line_eqn_from_bisected_chord (x y : ℝ) : Prop :=
  2 * x + y - 3 = 0

theorem find_line_eqn_from_bisected_chord (
  A B : ℝ × ℝ) 
  (hA :  (A.1^2) / 2 + (A.2^2) / 4 = 1)
  (hB :  (B.1^2) / 2 + (B.2^2) / 4 = 1)
  (h_mid : (A.1 + B.1) / 2 = 1 ∧ (A.2 + B.2) / 2 = 1) :
  line_eqn_from_bisected_chord 1 1 :=
by 
  sorry

end find_line_eqn_from_bisected_chord_l210_210043


namespace center_of_gravity_shift_center_of_gravity_shift_result_l210_210616

variable (l s : ℝ) (s_val : s = 60)
#check (s_val : s = 60)

theorem center_of_gravity_shift : abs ((l / 2) - ((l - s) / 2)) = s / 2 := 
by sorry

theorem center_of_gravity_shift_result : (s / 2 = 30) :=
by sorry

end center_of_gravity_shift_center_of_gravity_shift_result_l210_210616


namespace a5_eq_neg3_l210_210388

-- Define arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℤ) :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the sequence with given conditions
def a (n : ℕ) : ℤ :=
  if n = 2 then -5
  else if n = 8 then 1
  else sorry  -- Placeholder for other values

axiom a3_eq_neg5 : a 2 = -5
axiom a9_eq_1 : a 8 = 1
axiom a_is_arithmetic : is_arithmetic_sequence a

-- Statement to prove
theorem a5_eq_neg3 : a 4 = -3 :=
by
  sorry

end a5_eq_neg3_l210_210388


namespace smallest_a_with_50_perfect_squares_l210_210034

theorem smallest_a_with_50_perfect_squares : 
  ∃ a : ℕ, (∀ n : ℕ, (n^2 > a → (n^2 < 3 * a → 50 ∃ k, (k^2 >= a ∧ k^2 < 3 * a) ↔ k = (n+1)) → a = 4486

end smallest_a_with_50_perfect_squares_l210_210034


namespace problem_1_problem_2_l210_210047

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2
noncomputable def g (x : ℝ) : ℝ := 1 / 2 + Real.sqrt 3 * Real.sin x * Real.cos x
noncomputable def h (x : ℝ) : ℝ := f x + g x

-- Proof Problem 1
theorem problem_1 (a : ℝ) (h_symmetry : ∃ k : ℤ, a = k * Real.pi / 2) : g (2 * a) = 1 / 2 := by
  sorry

-- Proof Problem 2
theorem problem_2 (x : ℝ) (h_range : 0 ≤ x ∧ x ≤ Real.pi / 2) : 
  ∃ y : ℝ, y = h x ∧ 1/2 ≤ y ∧ y ≤ 2 := by
  sorry

end problem_1_problem_2_l210_210047


namespace solve_for_x_l210_210347

theorem solve_for_x (x : ℚ) :
  (x^2 - 4*x + 3) / (x^2 - 7*x + 6) = (x^2 - 3*x - 10) / (x^2 - 2*x - 15) →
  x = -3 / 4 :=
by
  intro h
  sorry

end solve_for_x_l210_210347


namespace river_depth_mid_may_l210_210690

variable (DepthMidMay DepthMidJune DepthMidJuly : ℕ)

theorem river_depth_mid_may :
  (DepthMidJune = DepthMidMay + 10) →
  (DepthMidJuly = 3 * DepthMidJune) →
  (DepthMidJuly = 45) →
  DepthMidMay = 5 :=
by
  intros h1 h2 h3
  sorry

end river_depth_mid_may_l210_210690


namespace sum_of_first_9000_terms_of_geometric_sequence_l210_210893

noncomputable def geometric_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_of_first_9000_terms_of_geometric_sequence 
  (a r : ℝ)
  (h₁ : geometric_sum a r 3000 = 500)
  (h₂ : geometric_sum a r 6000 = 950)
  : geometric_sum a r 9000 = 1355 :=
sorry

end sum_of_first_9000_terms_of_geometric_sequence_l210_210893


namespace apollo_total_cost_l210_210164

def hephaestus_first_half_months : ℕ := 6
def hephaestus_first_half_rate : ℕ := 3
def hephaestus_second_half_rate : ℕ := hephaestus_first_half_rate * 2

def athena_rate : ℕ := 5
def athena_months : ℕ := 12

def ares_first_period_months : ℕ := 9
def ares_first_period_rate : ℕ := 4
def ares_second_period_months : ℕ := 3
def ares_second_period_rate : ℕ := 6

def total_cost := hephaestus_first_half_months * hephaestus_first_half_rate
               + hephaestus_first_half_months * hephaestus_second_half_rate
               + athena_months * athena_rate
               + ares_first_period_months * ares_first_period_rate
               + ares_second_period_months * ares_second_period_rate

theorem apollo_total_cost : total_cost = 168 := by
  -- placeholder for the proof
  sorry

end apollo_total_cost_l210_210164


namespace train_speed_correct_l210_210606

noncomputable def jogger_speed_km_per_hr := 9
noncomputable def jogger_speed_m_per_s := 9 * 1000 / 3600
noncomputable def train_speed_km_per_hr := 45
noncomputable def distance_ahead_m := 270
noncomputable def train_length_m := 120
noncomputable def total_distance_m := distance_ahead_m + train_length_m
noncomputable def time_seconds := 39

theorem train_speed_correct :
  let relative_speed_m_per_s := total_distance_m / time_seconds
  let train_speed_m_per_s := relative_speed_m_per_s + jogger_speed_m_per_s
  let train_speed_km_per_hr_calculated := train_speed_m_per_s * 3600 / 1000
  train_speed_km_per_hr_calculated = train_speed_km_per_hr :=
by
  sorry

end train_speed_correct_l210_210606


namespace multiples_of_7_with_units_digit_7_number_of_multiples_of_7_with_units_digit_7_l210_210828

theorem multiples_of_7_with_units_digit_7 (n : ℕ) : 
  (n < 150 ∧ ∃ k : ℕ, n = 7 * k ∧ n % 10 = 7) ↔ 
  n = 7 ∨ n = 77 ∨ n = 147 := 
by 
  sorry

theorem number_of_multiples_of_7_with_units_digit_7 : 
  ∃ m, m = 3 ∧ ∀ n : ℕ, (n < 150 ∧ ∃ k : ℕ, n = 7 * k ∧ n % 10 = 7) → n = 7 ∨ n = 77 ∨ n = 147 := 
by 
  use 3
  intros n
  intros hn
  split
  intro h
  cases h
  sorry

end multiples_of_7_with_units_digit_7_number_of_multiples_of_7_with_units_digit_7_l210_210828


namespace gcd_459_357_polynomial_at_neg4_l210_210592

-- Statement for the GCD problem
theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

-- Definition of the polynomial
def f (x : Int) : Int :=
  12 + 35 * x - 8 * x^2 + 79 * x^3 + 6 * x^4 + 5 * x^5 + 3 * x^6

-- Statement for the polynomial evaluation problem
theorem polynomial_at_neg4 : f (-4) = 3392 := by
  sorry

end gcd_459_357_polynomial_at_neg4_l210_210592


namespace number_of_purchasing_schemes_l210_210383

def total_cost (a : Nat) (b : Nat) : Nat := 8 * a + 10 * b

def valid_schemes : List (Nat × Nat) :=
  [(4, 4), (4, 5), (4, 6), (4, 7),
   (5, 4), (5, 5), (5, 6),
   (6, 4), (6, 5),
   (7, 4)]

theorem number_of_purchasing_schemes : valid_schemes.length = 9 := sorry

end number_of_purchasing_schemes_l210_210383


namespace simplify_and_rationalize_l210_210544

theorem simplify_and_rationalize :
  ( (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 11) * (Real.sqrt 9 / Real.sqrt 13) = 
    (3 * Real.sqrt 15015) / 1001 ) :=
by
  sorry

end simplify_and_rationalize_l210_210544


namespace not_in_M_4n2_l210_210242

def M : Set ℤ := {a | ∃ x y : ℤ, a = x^2 - y^2}

theorem not_in_M_4n2 (n : ℤ) : ¬ (4 * n + 2 ∈ M) :=
by
sorry

end not_in_M_4n2_l210_210242


namespace geometric_sequence_product_l210_210228

noncomputable def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q

theorem geometric_sequence_product (a : ℕ → ℝ) (h_geo : is_geometric_sequence a)
  (h : a 3 = -1) : a 1 * a 2 * a 3 * a 4 * a 5 = -1 :=
by
  sorry

end geometric_sequence_product_l210_210228


namespace tangent_slope_of_cubic_l210_210556

theorem tangent_slope_of_cubic (P : ℝ × ℝ) (tangent_at_P : ℝ) (h1 : P.snd = P.fst ^ 3)
  (h2 : tangent_at_P = 3) : P = (1,1) ∨ P = (-1,-1) :=
by
  sorry

end tangent_slope_of_cubic_l210_210556


namespace trigonometric_identity_l210_210466

theorem trigonometric_identity (α : ℝ) (h1 : Real.tan α = Real.sqrt 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.cos α - Real.sin α = (-1 + Real.sqrt 3) / 2 :=
sorry

end trigonometric_identity_l210_210466


namespace total_cost_of_dishes_l210_210623

theorem total_cost_of_dishes
  (e t b : ℝ)
  (h1 : 4 * e + 5 * t + 2 * b = 8.20)
  (h2 : 6 * e + 3 * t + 4 * b = 9.40) :
  5 * e + 6 * t + 3 * b = 12.20 := 
sorry

end total_cost_of_dishes_l210_210623


namespace total_results_count_l210_210426

theorem total_results_count (N : ℕ) (S : ℕ) 
  (h1 : S = 50 * N) 
  (h2 : (12 * 14) + (12 * 17) = 372)
  (h3 : S = 372 + 878) : N = 25 := 
by 
  sorry

end total_results_count_l210_210426


namespace trapezoid_segment_ratio_l210_210321

theorem trapezoid_segment_ratio (s l : ℝ) (h₁ : 3 * s + l = 1) (h₂ : 2 * l + 6 * s = 2) :
  l = 2 * s :=
by
  sorry

end trapezoid_segment_ratio_l210_210321


namespace boys_in_art_class_l210_210110

noncomputable def number_of_boys (ratio_girls_to_boys : ℕ × ℕ) (total_students : ℕ) : ℕ :=
  let (g, b) := ratio_girls_to_boys
  let k := total_students / (g + b)
  b * k

theorem boys_in_art_class (h : number_of_boys (4, 3) 35 = 15) : true := 
  sorry

end boys_in_art_class_l210_210110


namespace smallest_a_has_50_perfect_squares_l210_210024

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def perfect_squares_in_interval (a b : ℕ) : ℕ :=
  (Finset.filter (λ x, is_perfect_square x) (Finset.range ((b + 1) - a))).card

theorem smallest_a_has_50_perfect_squares :
  ∃ a : ℕ, a = 4486 ∧ perfect_squares_in_interval (a + 1) (3 * a - 1) = 50 :=
by
  sorry

end smallest_a_has_50_perfect_squares_l210_210024


namespace greatest_integer_less_than_200_with_gcd_18_l210_210576

theorem greatest_integer_less_than_200_with_gcd_18 : ∃ n: ℕ, n < 200 ∧ Nat.gcd n 18 = 6 ∧ ∀ m: ℕ, m < 200 ∧ Nat.gcd m 18 = 6 → m ≤ n := 
by
  sorry

end greatest_integer_less_than_200_with_gcd_18_l210_210576


namespace candidate_function_is_odd_and_increasing_l210_210618

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_increasing_function (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

def candidate_function (x : ℝ) : ℝ := x * |x|

theorem candidate_function_is_odd_and_increasing :
  is_odd_function candidate_function ∧ is_increasing_function candidate_function :=
by
  sorry

end candidate_function_is_odd_and_increasing_l210_210618


namespace three_digit_subtraction_l210_210902

theorem three_digit_subtraction (c d : ℕ) (H1 : 0 ≤ c ∧ c ≤ 9) (H2 : 0 ≤ d ∧ d ≤ 9) :
  (745 - (300 + c * 10 + 4) = (400 + d * 10 + 1)) ∧ ((4 + 1) - d % 11 = 0) → 
  c + d = 14 := 
sorry

end three_digit_subtraction_l210_210902


namespace least_number_to_add_to_246835_l210_210134

-- Define relevant conditions and computations
def lcm_of_169_and_289 : ℕ := Nat.lcm 169 289
def remainder_246835_mod_lcm : ℕ := 246835 % lcm_of_169_and_289
def least_number_to_add : ℕ := lcm_of_169_and_289 - remainder_246835_mod_lcm

-- The theorem statement
theorem least_number_to_add_to_246835 : least_number_to_add = 52 :=
by
  sorry

end least_number_to_add_to_246835_l210_210134


namespace amount_to_add_l210_210438

theorem amount_to_add (students : ℕ) (total_cost : ℕ) (h1 : students = 9) (h2 : total_cost = 143) : 
  ∃ k : ℕ, total_cost + k = students * (total_cost / students + 1) :=
by
  sorry

end amount_to_add_l210_210438


namespace james_has_43_oreos_l210_210076

def james_oreos (jordan : ℕ) : ℕ := 7 + 4 * jordan

theorem james_has_43_oreos (jordan : ℕ) (total : ℕ) (h1 : total = jordan + james_oreos jordan) (h2 : total = 52) : james_oreos jordan = 43 :=
by
  sorry

end james_has_43_oreos_l210_210076


namespace percentage_of_girls_after_changes_l210_210067

theorem percentage_of_girls_after_changes :
  let boys_classA := 15
  let girls_classA := 20
  let boys_classB := 25
  let girls_classB := 35
  let boys_transferAtoB := 3
  let girls_transferAtoB := 2
  let boys_joiningA := 4
  let girls_joiningA := 6

  let boys_classA_after := boys_classA - boys_transferAtoB + boys_joiningA
  let girls_classA_after := girls_classA - girls_transferAtoB + girls_joiningA
  let boys_classB_after := boys_classB + boys_transferAtoB
  let girls_classB_after := girls_classB + girls_transferAtoB

  let total_students := boys_classA_after + girls_classA_after + boys_classB_after + girls_classB_after
  let total_girls := girls_classA_after + girls_classB_after 

  (total_girls / total_students : ℝ) * 100 = 58.095 := by
  sorry

end percentage_of_girls_after_changes_l210_210067


namespace find_height_of_pyramid_l210_210870

noncomputable def volume (B h : ℝ) : ℝ := (1/3) * B * h
noncomputable def area_of_isosceles_right_triangle (leg : ℝ) : ℝ := (1/2) * leg * leg

theorem find_height_of_pyramid (leg : ℝ) (h : ℝ) (V : ℝ) (B : ℝ)
  (Hleg : leg = 3)
  (Hvol : V = 6)
  (Hbase : B = area_of_isosceles_right_triangle leg)
  (Hvol_eq : V = volume B h) :
  h = 4 :=
by
  sorry

end find_height_of_pyramid_l210_210870


namespace range_of_a_l210_210058

def inequality_system_has_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, (x + a ≥ 0) ∧ (1 - 2 * x > x - 2)

theorem range_of_a (a : ℝ) : inequality_system_has_solution a ↔ a > -1 :=
by
  sorry

end range_of_a_l210_210058


namespace red_balls_count_l210_210063

theorem red_balls_count (R W N_1 N_2 : ℕ) 
  (h1 : R - 2 * N_1 = 18) 
  (h2 : W = 3 * N_1) 
  (h3 : R - 5 * N_2 = 0) 
  (h4 : W - 3 * N_2 = 18)
  : R = 50 :=
sorry

end red_balls_count_l210_210063


namespace Jacob_eats_more_calories_than_planned_l210_210699

theorem Jacob_eats_more_calories_than_planned 
  (planned_calories : ℕ) (actual_calories : ℕ)
  (h1 : planned_calories < 1800) 
  (h2 : actual_calories = 400 + 900 + 1100)
  : actual_calories - planned_calories = 600 := by
  sorry

end Jacob_eats_more_calories_than_planned_l210_210699


namespace greatest_divisor_6215_7373_l210_210913

theorem greatest_divisor_6215_7373 : 
  Nat.gcd (6215 - 23) (7373 - 29) = 144 := by
  sorry

end greatest_divisor_6215_7373_l210_210913


namespace polynomial_identity_l210_210369

theorem polynomial_identity (a : ℝ) (h₁ : a^5 + 5 * a^4 + 10 * a^3 + 3 * a^2 - 9 * a - 6 = 0) (h₂ : a ≠ -1) : (a + 1)^3 = 7 :=
sorry

end polynomial_identity_l210_210369


namespace functional_equation_solution_l210_210799

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  1 / (1 + a * x)

theorem functional_equation_solution (a : ℝ) (x y : ℝ)
  (ha : 0 < a) (hx : 0 < x) (hy : 0 < y) :
  f a x * f a (y * f a x) = f a (x + y) :=
sorry

end functional_equation_solution_l210_210799


namespace sin_identity_proof_l210_210962

theorem sin_identity_proof (x : ℝ) (h : Real.sin (x + π / 6) = 1 / 4) :
  Real.sin (5 * π / 6 - x) + Real.sin (π / 3 - x) ^ 2 = 19 / 16 :=
by
  sorry

end sin_identity_proof_l210_210962


namespace total_reams_l210_210209

theorem total_reams (h_r : ℕ) (s_r : ℕ) : h_r = 2 → s_r = 3 → h_r + s_r = 5 :=
by
  intro h_eq s_eq
  sorry

end total_reams_l210_210209


namespace rationalize_denominator_div_l210_210718

theorem rationalize_denominator_div (h : 343 = 7 ^ 3) : 7 / Real.sqrt 343 = Real.sqrt 7 / 7 := 
by 
  sorry

end rationalize_denominator_div_l210_210718


namespace transformed_inequality_l210_210909

theorem transformed_inequality (x : ℝ) : 
  (x - 3) / 3 < (2 * x + 1) / 2 - 1 ↔ 2 * (x - 3) < 3 * (2 * x + 1) - 6 :=
by
  sorry

end transformed_inequality_l210_210909


namespace chris_pennies_count_l210_210055

theorem chris_pennies_count (a c : ℤ) 
  (h1 : c + 2 = 4 * (a - 2)) 
  (h2 : c - 2 = 3 * (a + 2)) : 
  c = 62 := 
by 
  -- The actual proof is omitted
  sorry

end chris_pennies_count_l210_210055


namespace total_selling_price_l210_210614

theorem total_selling_price (cost_price_per_metre profit_per_metre : ℝ)
  (total_metres_sold : ℕ) :
  cost_price_per_metre = 58.02564102564102 → 
  profit_per_metre = 29 → 
  total_metres_sold = 78 →
  (cost_price_per_metre + profit_per_metre) * total_metres_sold = 6788 :=
by 
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  -- backend calculation, checking computation level;
  sorry

end total_selling_price_l210_210614


namespace smallest_b_factors_l210_210016

theorem smallest_b_factors (p q b : ℤ) (hpq : p * q = 1764) (hb : b = p + q) (hposp : p > 0) (hposq : q > 0) :
  b = 84 :=
by
  sorry

end smallest_b_factors_l210_210016


namespace isosceles_right_triangle_area_l210_210656

/--
Given an isosceles right triangle with a hypotenuse of 6√2 units, prove that the area
of this triangle is 18 square units.
-/
theorem isosceles_right_triangle_area (h : ℝ) (l : ℝ) (hyp : h = 6 * Real.sqrt 2) 
  (isosceles : h = l * Real.sqrt 2) : 
  (1/2) * l^2 = 18 :=
by
  sorry

end isosceles_right_triangle_area_l210_210656


namespace evaluate_expression_l210_210579

theorem evaluate_expression : 
    (1 / ( (-5 : ℤ) ^ 4) ^ 2 ) * (-5 : ℤ) ^ 9 = -5 :=
by sorry

end evaluate_expression_l210_210579


namespace largest_angle_of_convex_pentagon_l210_210315

theorem largest_angle_of_convex_pentagon :
  ∀ (x : ℝ), (x + 2) + (2 * x + 3) + (3 * x + 4) + (4 * x + 5) + (5 * x + 6) = 540 →
  5 * (104 / 3 : ℝ) + 6 = 538 / 3 := 
by
  intro x
  intro h
  sorry

end largest_angle_of_convex_pentagon_l210_210315


namespace amc_proposed_by_Dorlir_Ahmeti_Albania_l210_210804

-- Define the problem statement, encapsulating the conditions and the final inequality.
theorem amc_proposed_by_Dorlir_Ahmeti_Albania
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h_cond : a * b + b * c + c * a = 3) :
  (a / Real.sqrt (a^3 + 5) + b / Real.sqrt (b^3 + 5) + c / Real.sqrt (c^3 + 5) ≤ Real.sqrt 6 / 2) := 
by 
  sorry -- Proof steps go here, which are omitted as per the requirement.

end amc_proposed_by_Dorlir_Ahmeti_Albania_l210_210804


namespace john_more_needed_l210_210852

def john_needs : ℝ := 2.5
def john_has : ℝ := 0.75
def john_needs_more : ℝ := 1.75

theorem john_more_needed : (john_needs - john_has) = john_needs_more :=
by
  sorry

end john_more_needed_l210_210852


namespace average_speed_of_train_l210_210444

theorem average_speed_of_train
  (d1 d2 : ℝ) (t1 t2 : ℝ)
  (h1 : d1 = 290) (h2 : d2 = 400) (h3 : t1 = 4.5) (h4 : t2 = 5.5) :
  ((d1 + d2) / (t1 + t2)) = 69 :=
by
  -- proof steps can be filled in later
  sorry

end average_speed_of_train_l210_210444


namespace average_of_remaining_two_nums_l210_210102

theorem average_of_remaining_two_nums (S S4 : ℕ) (h1 : S / 6 = 8) (h2 : S4 / 4 = 5) :
  ((S - S4) / 2 = 14) :=
by 
  sorry

end average_of_remaining_two_nums_l210_210102


namespace susan_can_drive_with_50_l210_210547

theorem susan_can_drive_with_50 (car_efficiency : ℕ) (gas_price : ℕ) (money_available : ℕ) 
  (h1 : car_efficiency = 40) (h2 : gas_price = 5) (h3 : money_available = 50) : 
  car_efficiency * (money_available / gas_price) = 400 :=
by
  sorry

end susan_can_drive_with_50_l210_210547


namespace probability_at_least_one_spade_and_no_hearts_l210_210766

theorem probability_at_least_one_spade_and_no_hearts :
  let P_not_spade := (39 / 52 : ℝ)
  let P_spade := (1 - P_not_spade)
  let P_no_hearts_5 := P_not_spade ^ 5
  let P_at_least_one_spade := 1 - P_not_spade ^ 5
  let P_no_hearts_and_spade := P_at_least_one_spade * P_no_hearts_5
  P_no_hearts_and_spade = 189723 / 1048576 :=
sorry

end probability_at_least_one_spade_and_no_hearts_l210_210766


namespace value_of_expression_when_x_is_neg2_l210_210128

theorem value_of_expression_when_x_is_neg2 : 
  ∀ (x : ℤ), x = -2 → (3 * x + 4) ^ 2 = 4 :=
by
  sorry

end value_of_expression_when_x_is_neg2_l210_210128


namespace multiples_of_7_units_digit_7_l210_210825

theorem multiples_of_7_units_digit_7 (n : ℕ) (h1 : n < 150) (h2 : ∃ (k : ℕ), n = 7 * k) (h3 : n % 10 = 7) : 
    ∃ (m : ℕ), m = 2 := 
by
  sorry

end multiples_of_7_units_digit_7_l210_210825


namespace graveling_cost_is_correct_l210_210911

noncomputable def cost_of_graveling (lawn_length : ℕ) (lawn_breadth : ℕ) 
(road_width : ℕ) (cost_per_sq_m : ℕ) : ℕ :=
  let area_road_parallel_to_length := road_width * lawn_breadth
  let area_road_parallel_to_breadth := road_width * lawn_length
  let area_overlap := road_width * road_width
  let total_area := area_road_parallel_to_length + area_road_parallel_to_breadth - area_overlap
  total_area * cost_per_sq_m

theorem graveling_cost_is_correct : cost_of_graveling 90 60 10 3 = 4200 := by
  sorry

end graveling_cost_is_correct_l210_210911


namespace point_on_line_l210_210139

theorem point_on_line (m n k : ℝ) (h1 : m = 2 * n + 5) (h2 : m + 4 = 2 * (n + k) + 5) : k = 2 := by
  sorry

end point_on_line_l210_210139


namespace parallel_lines_l210_210667

theorem parallel_lines :
  (∃ m: ℚ, (∀ x y: ℚ, (4 * y - 3 * x = 16 → y = m * x + (16 / 4)) ∧
                      (-3 * x - 4 * y = 15 → y = -m * x - (15 / 4)) ∧
                      (4 * y + 3 * x = 16 → y = -m * x + (16 / 4)) ∧
                      (3 * y + 4 * x = 15) → False)) :=
sorry

end parallel_lines_l210_210667


namespace gcd_product_eq_gcd_l210_210088

theorem gcd_product_eq_gcd {a b c : ℤ} (hab : Int.gcd a b = 1) : Int.gcd a (b * c) = Int.gcd a c := 
by 
  sorry

end gcd_product_eq_gcd_l210_210088


namespace fuel_cost_equation_l210_210646

theorem fuel_cost_equation (x : ℝ) (h : (x / 4) - (x / 6) = 8) : x = 96 :=
sorry

end fuel_cost_equation_l210_210646


namespace mike_daily_work_hours_l210_210539

def total_hours_worked : ℕ := 15
def number_of_days_worked : ℕ := 5

theorem mike_daily_work_hours : total_hours_worked / number_of_days_worked = 3 :=
by
  sorry

end mike_daily_work_hours_l210_210539


namespace non_empty_solution_set_l210_210677

theorem non_empty_solution_set (a : ℝ) (h : a > 0) : (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 :=
by
  sorry

end non_empty_solution_set_l210_210677


namespace speed_in_terms_of_time_l210_210707

variable (a b x : ℝ)

-- Conditions
def condition1 : Prop := 1000 = a * x
def condition2 : Prop := 833 = b * x

-- The theorem to prove
theorem speed_in_terms_of_time (h1 : condition1 a x) (h2 : condition2 b x) :
  a = 1000 / x ∧ b = 833 / x :=
by
  sorry

end speed_in_terms_of_time_l210_210707


namespace greatest_integer_less_than_200_with_gcd_18_l210_210575

theorem greatest_integer_less_than_200_with_gcd_18 : ∃ n: ℕ, n < 200 ∧ Nat.gcd n 18 = 6 ∧ ∀ m: ℕ, m < 200 ∧ Nat.gcd m 18 = 6 → m ≤ n := 
by
  sorry

end greatest_integer_less_than_200_with_gcd_18_l210_210575


namespace remainder_g10_div_g_l210_210087

-- Conditions/Definitions
def g (x : ℝ) : ℝ := x^5 + x^4 + x^3 + x^2 + x + 1
def g10 (x : ℝ) : ℝ := (g (x^10))

-- Theorem/Question
theorem remainder_g10_div_g : (g10 x) % (g x) = 6 :=
by
  sorry

end remainder_g10_div_g_l210_210087


namespace find_theta_plus_3phi_l210_210356

variables (θ φ : ℝ)

-- The conditions
variables (h1 : 0 < θ ∧ θ < π / 2) (h2 : 0 < φ ∧ φ < π / 2)
variables (h3 : Real.tan θ = 1 / 3) (h4 : Real.sin φ = 3 / 5)

theorem find_theta_plus_3phi :
  θ + 3 * φ = π - Real.arctan (199 / 93) :=
sorry

end find_theta_plus_3phi_l210_210356


namespace ryan_sandwiches_l210_210722

theorem ryan_sandwiches (sandwich_slices : ℕ) (total_slices : ℕ) (h1 : sandwich_slices = 3) (h2 : total_slices = 15) :
  total_slices / sandwich_slices = 5 :=
by
  sorry

end ryan_sandwiches_l210_210722


namespace certain_number_sixth_powers_l210_210158

theorem certain_number_sixth_powers :
  ∃ N, (∀ n : ℕ, n < N → ∃ a : ℕ, n = a^6) ∧
       (∃ m ≤ N, (∀ n < m, ∃ k : ℕ, n = k^6) ∧ ¬ ∃ k : ℕ, m = k^6) :=
sorry

end certain_number_sixth_powers_l210_210158


namespace correct_sum_of_integers_l210_210860

theorem correct_sum_of_integers (a b : ℕ) (h1 : a - b = 4) (h2 : a * b = 63) : a + b = 18 := 
  sorry

end correct_sum_of_integers_l210_210860


namespace what_to_do_first_l210_210709

-- Definition of the conditions
def eat_or_sleep_to_survive (days_without_eat : ℕ) (days_without_sleep : ℕ) : Prop :=
  (days_without_eat = 7 → days_without_sleep ≠ 7) ∨ (days_without_sleep = 7 → days_without_eat ≠ 7)

-- Theorem statement based on the problem and its conditions
theorem what_to_do_first (days_without_eat days_without_sleep : ℕ) :
  days_without_eat = 7 ∨ days_without_sleep = 7 →
  eat_or_sleep_to_survive days_without_eat days_without_sleep :=
by sorry

end what_to_do_first_l210_210709


namespace combination_8_5_l210_210517

theorem combination_8_5 : (Nat.choose 8 5) = 56 := by
  sorry

end combination_8_5_l210_210517


namespace find_k_l210_210138

theorem find_k
  (t k : ℝ)
  (h1 : t = 5 / 9 * (k - 32))
  (h2 : t = 20) :
  k = 68 := 
by
  sorry

end find_k_l210_210138


namespace smallest_a_with_50_squares_l210_210020


theorem smallest_a_with_50_squares : ∃ (a : ℕ), (∀ b : ℕ, (b < a)) → 
(a > 0) ∧ (a * 3 > a) ∧ 
let count_squares := λ (a : ℕ), set.to_finset {n : ℕ | n^2 ∈ (set.Ioo (a : ℝ) (3 * a : ℝ))}.card
in count_squares a = 50 ∧ a = 4486 := by {
  sorry -- proof omitted
}

end smallest_a_with_50_squares_l210_210020


namespace ned_mowed_in_summer_l210_210409

def mowed_in_summer (total_mows spring_mows summer_mows : ℕ) : Prop :=
  total_mows = spring_mows + summer_mows

theorem ned_mowed_in_summer :
  ∀ (total_mows spring_mows summer_mows : ℕ),
  total_mows = 11 →
  spring_mows = 6 →
  mowed_in_summer total_mows spring_mows summer_mows →
  summer_mows = 5 :=
by
  intros total_mows spring_mows summer_mows h_total h_spring h_mowed
  sorry

end ned_mowed_in_summer_l210_210409


namespace prove_ZAMENA_l210_210781

theorem prove_ZAMENA :
  ∃ (A M E H Z : ℕ),
    1 ≤ A ∧ A ≤ 5 ∧
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ Z ∧ Z ≤ 5 ∧
    (3 > A) ∧ (A > M) ∧ (M < E) ∧ (E < H) ∧ (H < A) ∧
    (A ≠ M) ∧ (A ≠ E) ∧ (A ≠ H) ∧ (A ≠ Z) ∧
    (M ≠ E) ∧ (M ≠ H) ∧ (M ≠ Z) ∧
    (E ≠ H) ∧ (E ≠ Z) ∧
    (H ≠ Z) ∧
    Z = 5 ∧
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234 :=
begin
  sorry
end

end prove_ZAMENA_l210_210781


namespace quadratic_distinct_real_roots_l210_210496

theorem quadratic_distinct_real_roots (k : ℝ) : 
  (∀ (x : ℝ), (k - 1) * x^2 + 4 * x + 1 = 0 → False) ↔ (k < 5 ∧ k ≠ 1) :=
by
  sorry

end quadratic_distinct_real_roots_l210_210496


namespace min_distance_from_curve_to_focus_l210_210273

noncomputable def minDistanceToFocus (x y θ : ℝ) : ℝ :=
  let a := 3
  let b := 2
  let c := Real.sqrt (a^2 - b^2)
  a - c

theorem min_distance_from_curve_to_focus :
  ∀ θ : ℝ, minDistanceToFocus (2 * Real.cos θ) (3 * Real.sin θ) θ = 3 - Real.sqrt 5 :=
by
  sorry

end min_distance_from_curve_to_focus_l210_210273


namespace max_m_l210_210379

noncomputable def f (x a : ℝ) : ℝ := 2 ^ |x + a|

theorem max_m (a m : ℝ) (H1 : ∀ x, f (3 + x) a = f (3 - x) a) 
(H2 : ∀ x y, x ≤ y → y ≤ m → f x a ≥ f y a) : 
  m = 3 :=
by
  sorry

end max_m_l210_210379


namespace turquoise_more_green_l210_210593

-- Definitions based on given conditions
def total_people : ℕ := 150
def more_blue : ℕ := 90
def both_blue_green : ℕ := 40
def neither_blue_green : ℕ := 20

-- Theorem statement to prove the number of people who believe turquoise is more green
theorem turquoise_more_green : (total_people - neither_blue_green - (more_blue - both_blue_green) - both_blue_green) + both_blue_green = 80 := by
  sorry

end turquoise_more_green_l210_210593


namespace binomial_coeff_8_3_l210_210335

theorem binomial_coeff_8_3 : nat.choose 8 3 = 56 := by
  sorry

end binomial_coeff_8_3_l210_210335


namespace factor_54x5_135x9_l210_210632

theorem factor_54x5_135x9 (x : ℝ) :
  54 * x ^ 5 - 135 * x ^ 9 = -27 * x ^ 5 * (5 * x ^ 4 - 2) :=
by 
  sorry

end factor_54x5_135x9_l210_210632


namespace spring_spending_l210_210729

theorem spring_spending (end_of_feb : ℝ) (end_of_may : ℝ) (h_end_of_feb : end_of_feb = 0.8) (h_end_of_may : end_of_may = 2.5)
  : (end_of_may - end_of_feb) = 1.7 :=
by
  have spending_end_of_feb : end_of_feb = 0.8 := h_end_of_feb
  have spending_end_of_may : end_of_may = 2.5 := h_end_of_may
  sorry

end spring_spending_l210_210729


namespace Jamir_swims_more_l210_210701

def Julien_distance_per_day : ℕ := 50
def Sarah_distance_per_day (J : ℕ) : ℕ := 2 * J
def combined_distance_per_week (J S M : ℕ) : ℕ := 7 * (J + S + M)

theorem Jamir_swims_more :
  let J := Julien_distance_per_day
  let S := Sarah_distance_per_day J
  ∃ M, combined_distance_per_week J S M = 1890 ∧ (M - S = 20) := by
    let J := Julien_distance_per_day
    let S := Sarah_distance_per_day J
    use 120
    sorry

end Jamir_swims_more_l210_210701


namespace cistern_emptying_time_l210_210314

theorem cistern_emptying_time (R L : ℝ) (h1 : R * 8 = 1) (h2 : (R - L) * 10 = 1) : 1 / L = 40 :=
by
  -- proof omitted
  sorry

end cistern_emptying_time_l210_210314


namespace heath_plants_per_hour_l210_210486

theorem heath_plants_per_hour (rows : ℕ) (plants_per_row : ℕ) (hours : ℕ) (total_plants : ℕ) :
  rows = 400 ∧ plants_per_row = 300 ∧ hours = 20 ∧ total_plants = rows * plants_per_row →
  total_plants / hours = 6000 :=
by
  sorry

end heath_plants_per_hour_l210_210486


namespace frustum_volume_fraction_l210_210162

noncomputable def volume_pyramid (base_edge height : ℝ) : ℝ :=
  (1 / 3) * (base_edge ^ 2) * height

noncomputable def fraction_of_frustum (base_edge height : ℝ) : ℝ :=
  let original_volume := volume_pyramid base_edge height
  let smaller_volume := volume_pyramid (base_edge / 5) (height / 5)
  let frustum_volume := original_volume - smaller_volume
  frustum_volume / original_volume

theorem frustum_volume_fraction :
  fraction_of_frustum 40 20 = 63 / 64 :=
by sorry

end frustum_volume_fraction_l210_210162


namespace coin_flip_prob_l210_210097

theorem coin_flip_prob : 
  let outcomes := 2^5 in
  let success_cases := 8 in
  (success_cases / outcomes) = 1 / 4 :=
by
  sorry

end coin_flip_prob_l210_210097


namespace horizontal_length_tv_screen_l210_210246

theorem horizontal_length_tv_screen : 
  ∀ (a b : ℝ), (a / b = 4 / 3) → (a ^ 2 + b ^ 2 = 27 ^ 2) → a = 21.5 := 
by 
  sorry

end horizontal_length_tv_screen_l210_210246


namespace find_r_given_conditions_l210_210842

theorem find_r_given_conditions (p c r : ℝ) (h1 : p * r = 360) (h2 : 6 * c * r = 15) (h3 : r = 4) : r = 4 :=
by
  sorry

end find_r_given_conditions_l210_210842


namespace find_k_values_l210_210362

theorem find_k_values (a b k : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : b % a = 0) 
  (h₄ : ∀ (m : ℤ), (a : ℤ) = k * (a : ℤ) + m ∧ (8 * (b : ℤ)) = k * (b : ℤ) + m) :
  k = 9 ∨ k = 15 :=
by
  { sorry }

end find_k_values_l210_210362


namespace problem1_problem2_l210_210331

-- Problem 1
theorem problem1 (x y : ℝ) :
  2 * x^2 * y - 3 * x * y + 2 - x^2 * y + 3 * x * y = x^2 * y + 2 :=
by sorry

-- Problem 2
theorem problem2 (m n : ℝ) :
  9 * m^2 - 4 * (2 * m^2 - 3 * m * n + n^2) + 4 * n^2 = m^2 + 12 * m * n :=
by sorry

end problem1_problem2_l210_210331


namespace find_m_l210_210671

theorem find_m {m : ℝ} (a b : ℝ × ℝ) (H : a = (3, m) ∧ b = (2, -1)) (H_dot : a.1 * b.1 + a.2 * b.2 = 0) : m = 6 := 
by
  sorry

end find_m_l210_210671


namespace harry_change_l210_210977

theorem harry_change (a : ℕ) :
  (∃ k : ℕ, a = 50 * k + 2 ∧ a < 100) ∧ (∃ m : ℕ, a = 5 * m + 4 ∧ a < 100) →
  a = 52 :=
by sorry

end harry_change_l210_210977


namespace largest_n_for_perfect_square_l210_210494

theorem largest_n_for_perfect_square :
  ∃ n : ℕ, 4 ^ 27 + 4 ^ 500 + 4 ^ n = k ^ 2 ∧ ∀ m : ℕ, 4 ^ 27 + 4 ^ 500 + 4 ^ m = l ^ 2 → m ≤ n  → n = 972 :=
sorry

end largest_n_for_perfect_square_l210_210494


namespace probability_maxim_born_in_2008_l210_210253

def days_in_month : ℕ → ℕ → ℕ 
| 2007 9 := 29
| 2007 10 := 31
| 2007 11 := 30
| 2007 12 := 31
| 2008 1 := 31
| 2008 2 := 29   -- leap year
| 2008 3 := 31
| 2008 4 := 30
| 2008 5 := 31
| 2008 6 := 30
| 2008 7 := 31
| 2008 8 := 31
| _ _ := 0

def total_days : ℕ :=
list.sum [days_in_month 2007 9, days_in_month 2007 10, days_in_month 2007 11, days_in_month 2007 12,
          days_in_month 2008 1, days_in_month 2008 2, days_in_month 2008 3, days_in_month 2008 4,
          days_in_month 2008 5, days_in_month 2008 6, days_in_month 2008 7, days_in_month 2008 8]

def days_in_2008 : ℕ :=
list.sum [days_in_month 2008 1, days_in_month 2008 2, days_in_month 2008 3, days_in_month 2008 4,
          days_in_month 2008 5, days_in_month 2008 6, days_in_month 2008 7, days_in_month 2008 8]

theorem probability_maxim_born_in_2008 : 
  (days_in_2008 : ℚ) / (total_days : ℚ) = 244 / 365 :=
by
  -- Placeholder for actual proof. This theorem is set properly and ready for proof development.
  sorry

end probability_maxim_born_in_2008_l210_210253


namespace complement_M_l210_210480

def U : Set ℝ := Set.univ
def M : Set ℝ := { x | x^2 - 4 ≤ 0 }

theorem complement_M (U M : Set ℝ) : (U \ M) = { x : ℝ | x < -2 ∨ x > 2 } :=
by
  sorry

end complement_M_l210_210480


namespace cost_one_dozen_pens_l210_210590

variable (cost_of_pen cost_of_pencil : ℝ)
variable (ratio : ℝ)
variable (dozen_pens_cost : ℝ)

axiom cost_equation : 3 * cost_of_pen + 5 * cost_of_pencil = 200
axiom ratio_pen_pencil : cost_of_pen = 5 * cost_of_pencil

theorem cost_one_dozen_pens : dozen_pens_cost = 12 * cost_of_pen := 
  by
    sorry

end cost_one_dozen_pens_l210_210590


namespace max_value_quadratic_l210_210742

theorem max_value_quadratic : ∀ s : ℝ, ∃ M : ℝ, (∀ s : ℝ, -3 * s^2 + 54 * s - 27 ≤ M) ∧ M = 216 :=
by
  sorry

end max_value_quadratic_l210_210742


namespace problem_statement_l210_210837

noncomputable def P1 (α : ℝ) : ℝ × ℝ := (Real.cos α, Real.sin α)
noncomputable def P2 (β : ℝ) : ℝ × ℝ := (Real.cos β, -Real.sin β)
noncomputable def P3 (α β : ℝ) : ℝ × ℝ := (Real.cos (α + β), Real.sin (α + β))
noncomputable def A : ℝ × ℝ := (1, 0)

theorem problem_statement (α β : ℝ) :
  (Prod.fst (P1 α))^2 + (Prod.snd (P1 α))^2 = 1 ∧
  (Prod.fst (P2 β))^2 + (Prod.snd (P2 β))^2 = 1 ∧
  (Prod.fst (P1 α) * Prod.fst (P2 β) + Prod.snd (P1 α) * Prod.snd (P2 β)) = Real.cos (α + β) :=
by
  sorry

end problem_statement_l210_210837


namespace black_balls_number_l210_210811

-- Define the given conditions and the problem statement as Lean statements
theorem black_balls_number (n : ℕ) (h : (2 : ℝ) / (n + 2 : ℝ) = 0.4) : n = 3 :=
by
  sorry

end black_balls_number_l210_210811


namespace milk_for_flour_l210_210805

theorem milk_for_flour (milk flour use_flour : ℕ) (h1 : milk = 75) (h2 : flour = 300) (h3 : use_flour = 900) : (use_flour/flour * milk) = 225 :=
by sorry

end milk_for_flour_l210_210805


namespace min_max_x_l210_210237

theorem min_max_x (n : ℕ) (hn : 0 < n) (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = n * x + n * y) : 
  n + 1 ≤ x ∧ x ≤ n * (n + 1) :=
by {
  sorry  -- Proof goes here
}

end min_max_x_l210_210237


namespace riley_outside_fraction_l210_210080

theorem riley_outside_fraction
  (awake_jonsey : ℚ := 2 / 3)
  (jonsey_outside_fraction : ℚ := 1 / 2)
  (awake_riley : ℚ := 3 / 4)
  (total_inside_time : ℚ := 10)
  (hours_per_day : ℕ := 24) :
  let jonsey_inside_time := 1 / 3 * hours_per_day
  let riley_inside_time := (1 - (8 / 9)) * (3 / 4) * hours_per_day
  jonsey_inside_time + riley_inside_time = total_inside_time :=
by
  sorry

end riley_outside_fraction_l210_210080


namespace kindergarten_children_l210_210225

theorem kindergarten_children (x y z n : ℕ) 
  (h1 : 2 * x + 3 * y + 4 * z = n)
  (h2 : x + y + z = 26)
  : n = 24 := 
sorry

end kindergarten_children_l210_210225


namespace quadratic_no_real_solution_l210_210649

theorem quadratic_no_real_solution (a : ℝ) : (∀ x : ℝ, x^2 - x + a ≠ 0) → a > 1 / 4 :=
by
  intro h
  sorry

end quadratic_no_real_solution_l210_210649


namespace smallest_multiple_of_6_and_15_l210_210641

theorem smallest_multiple_of_6_and_15 : ∃ b : ℕ, b > 0 ∧ b % 6 = 0 ∧ b % 15 = 0 ∧ ∀ b' : ℕ, b' > 0 ∧ b' % 6 = 0 ∧ b' % 15 = 0 → b ≤ b' :=
sorry

end smallest_multiple_of_6_and_15_l210_210641


namespace binomial_variance_example_l210_210684

noncomputable def variance {n : ℕ} {p : ℝ} (X : ℕ → ℝ) [Binomial X n p] : ℝ :=
  n * p * (1 - p)

theorem binomial_variance_example :
  let X : ℕ → ℝ := sorry in
  ∀ (X : ℕ → ℝ), Binomial X 10 (2 / 3) → variance X = 20 / 9 :=
by
  intros X hXh
  rw [variance, hXh]
  sorry

end binomial_variance_example_l210_210684


namespace sphere_cross_section_area_l210_210011

theorem sphere_cross_section_area (R : ℝ) (d : ℝ) (hR : R = 3) (hd : d = 2) : 
    let r := Real.sqrt (R^2 - d^2) in
    let area := Real.pi * r^2 in
    area = 5 * Real.pi := 
  by
  -- Introduce the values
  rw [hR, hd]
  -- Let r be the radius of the cross-section circle
  let r := Real.sqrt (3^2 - 2^2)
  have hr : r = Real.sqrt 5 := by
    -- Calculation details:
    calc
      r = Real.sqrt (3^2 - 2^2) : by rw [hR, hd]
      ... = Real.sqrt (9 - 4)    : by norm_num
      ... = Real.sqrt 5         : by norm_num
  -- Hence the area of the cross-section circle
  let area := Real.pi * r^2
  have harea : area = 5 * Real.pi := by
    calc
      area = Real.pi * (Real.sqrt 5)^2 : by rw hr
      ...  = Real.pi * 5               : by norm_num
      ...  = 5 * Real.pi               : by ring
  exact harea

end sphere_cross_section_area_l210_210011


namespace determine_dresses_and_shoes_colors_l210_210737

variables (dress_color shoe_color : String → String)
variables (Tamara Valya Lida : String)

-- Conditions
def condition_1 : Prop := ∀ x : String, x ≠ Tamara → dress_color x ≠ shoe_color x
def condition_2 : Prop := shoe_color Valya = "white"
def condition_3 : Prop := dress_color Lida ≠ "red" ∧ shoe_color Lida ≠ "red"
def condition_4 : Prop := ∀ x : String, dress_color x ∈ ["white", "red", "blue"] ∧ shoe_color x ∈ ["white", "red", "blue"]

-- Desired conclusion
def conclusion : Prop :=
  dress_color Valya = "blue" ∧ shoe_color Valya = "white" ∧
  dress_color Lida = "white" ∧ shoe_color Lida = "blue" ∧
  dress_color Tamara = "red" ∧ shoe_color Tamara = "red"

theorem determine_dresses_and_shoes_colors
  (Tamara Valya Lida : String)
  (h1 : condition_1 dress_color shoe_color Tamara)
  (h2 : condition_2 shoe_color Valya)
  (h3 : condition_3 dress_color shoe_color Lida)
  (h4 : condition_4 dress_color shoe_color) :
  conclusion dress_color shoe_color Valya Lida Tamara :=
sorry

end determine_dresses_and_shoes_colors_l210_210737


namespace truck_speed_in_mph_l210_210879

-- Definitions based on the conditions
def truck_length : ℝ := 66  -- Truck length in feet
def tunnel_length : ℝ := 330  -- Tunnel length in feet
def exit_time : ℝ := 6  -- Exit time in seconds
def feet_to_miles : ℝ := 5280  -- Feet per mile

-- Problem statement
theorem truck_speed_in_mph :
  ((tunnel_length + truck_length) / exit_time) * (3600 / feet_to_miles) = 45 := 
sorry

end truck_speed_in_mph_l210_210879


namespace greatest_integer_gcd_l210_210565

theorem greatest_integer_gcd (n : ℕ) (h1 : n < 200) (h2 : gcd n 18 = 6) : n = 192 :=
sorry

end greatest_integer_gcd_l210_210565


namespace eating_relationship_l210_210299

open Set

-- Definitions of the sets A and B
def A : Set ℝ := {-1, 1/2, 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | a ≥ 0 ∧ a * x^2 = 1}

-- Definitions of relationships
def full_eating (A B : Set ℝ) : Prop := A ⊆ B ∨ B ⊆ A
def partial_eating (A B : Set ℝ) : Prop := ∃ x, x ∈ A ∧ x ∈ B ∧ ¬ full_eating A B

-- Main theorem
theorem eating_relationship (a : ℝ) :
  (full_eating A (B a) ∨ partial_eating A (B a)) ↔ (a = 0 ∨ a = 1 ∨ a = 4) := by
  sorry

end eating_relationship_l210_210299


namespace harry_to_sue_nuts_ratio_l210_210448

-- Definitions based on conditions
def sue_nuts : ℕ := 48
def bill_nuts (harry_nuts : ℕ) : ℕ := 6 * harry_nuts
def total_nuts (harry_nuts : ℕ) : ℕ := bill_nuts harry_nuts + harry_nuts

-- Proving the ratio
theorem harry_to_sue_nuts_ratio (H : ℕ) (h1 : sue_nuts = 48) (h2 : bill_nuts H + H = 672) : H / sue_nuts = 2 :=
by
  sorry

end harry_to_sue_nuts_ratio_l210_210448


namespace combination_8_5_l210_210516

theorem combination_8_5 : (Nat.choose 8 5) = 56 := by
  sorry

end combination_8_5_l210_210516


namespace product_of_factors_l210_210330

theorem product_of_factors : (2.1 * (53.2 - 0.2) = 111.3) := by
  sorry

end product_of_factors_l210_210330


namespace project_completion_time_l210_210720

theorem project_completion_time (Renu_time Suma_time Arun_time : ℚ)
  (hR: Renu_time = 5)
  (hS: Suma_time = 8)
  (hA: Arun_time = 10) :
  (1 / Renu_time + 1 / Suma_time + 1 / Arun_time)⁻¹ = 40 / 17 :=
by
  rw [hR, hS, hA]
  -- Here we calculate the combined rate and then find its reciprocal
  calc
    (1 / 5 + 1 / 8 + 1 / 10)⁻¹
        = (1 / 5 + 1 / 8 + 1 / 10)⁻¹ : rfl
    ... = (8 / 40 + 5 / 40 + 4 / 40)⁻¹ : by norm_num; rfl
    ... = (17 / 40)⁻¹ : by norm_num
    ... = 40 / 17 : by norm_num

end project_completion_time_l210_210720


namespace rationalize_denominator_of_seven_over_sqrt_343_l210_210719

noncomputable def rationalize_denominator (x : ℝ) : ℝ := x * sqrt 7 / 7

theorem rationalize_denominator_of_seven_over_sqrt_343 :
  (343 = 7^3) → (sqrt 343 = 7 * sqrt 7) →
  (7 / sqrt 343 = sqrt 7 / 7) :=
by
  intros h1 h2
  sorry

end rationalize_denominator_of_seven_over_sqrt_343_l210_210719


namespace probability_prime_and_greater_than_4_l210_210758

namespace Probability

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5

def successful_outcomes_6_sided := { x : ℕ | x ∈ {2, 3, 5} }.card
def successful_outcomes_8_sided := { x : ℕ | x ∈ {5, 6, 7, 8} }.card

theorem probability_prime_and_greater_than_4 :
  (successful_outcomes_6_sided * successful_outcomes_8_sided : ℚ) / (6 * 8) = 1 / 4 := by
sorry

end Probability

end probability_prime_and_greater_than_4_l210_210758


namespace sandy_receives_correct_change_l210_210000

-- Define the costs of each item
def cost_cappuccino : ℕ := 2
def cost_iced_tea : ℕ := 3
def cost_cafe_latte : ℝ := 1.5
def cost_espresso : ℕ := 1

-- Define the quantities ordered
def qty_cappuccino : ℕ := 3
def qty_iced_tea : ℕ := 2
def qty_cafe_latte : ℕ := 2
def qty_espresso : ℕ := 2

-- Calculate the total cost
def total_cost : ℝ := (qty_cappuccino * cost_cappuccino) + 
                      (qty_iced_tea * cost_iced_tea) + 
                      (qty_cafe_latte * cost_cafe_latte) + 
                      (qty_espresso * cost_espresso)

-- Define the amount paid
def amount_paid : ℝ := 20

-- Calculate the change
def change : ℝ := amount_paid - total_cost

theorem sandy_receives_correct_change : change = 3 := by
  -- Detailed steps would go here
  sorry

end sandy_receives_correct_change_l210_210000


namespace comb_8_5_eq_56_l210_210513

theorem comb_8_5_eq_56 : nat.choose 8 5 = 56 :=
by {
  sorry
}

end comb_8_5_eq_56_l210_210513


namespace paul_spent_81_90_l210_210266

-- Define the original price of each racket
def originalPrice : ℝ := 60

-- Define the discount rates
def firstDiscount : ℝ := 0.20
def secondDiscount : ℝ := 0.50

-- Define the sales tax rate
def salesTax : ℝ := 0.05

-- Define the prices after discount
def firstRacketPrice : ℝ := originalPrice * (1 - firstDiscount)
def secondRacketPrice : ℝ := originalPrice * (1 - secondDiscount)

-- Define the total price before tax
def totalPriceBeforeTax : ℝ := firstRacketPrice + secondRacketPrice

-- Define the total sales tax
def totalSalesTax : ℝ := totalPriceBeforeTax * salesTax

-- Define the total amount spent
def totalAmountSpent : ℝ := totalPriceBeforeTax + totalSalesTax

-- The statement to prove
theorem paul_spent_81_90 : totalAmountSpent = 81.90 := 
by
  sorry

end paul_spent_81_90_l210_210266


namespace each_person_gets_after_taxes_l210_210233

-- Definitions based strictly on problem conditions
def house_price : ℝ := 500000
def market_multiplier : ℝ := 1.2
def brothers_count : ℕ := 3
def tax_rate : ℝ := 0.1

-- Derived conditions
def selling_price : ℝ := house_price * market_multiplier
def total_people : ℕ := 1 + brothers_count
def share_before_taxes : ℝ := selling_price / total_people
def tax_amount_per_person : ℝ := share_before_taxes * tax_rate
def final_amount_per_person : ℝ := share_before_taxes - tax_amount_per_person

-- Problem: Prove the final amount each person receives
theorem each_person_gets_after_taxes : final_amount_per_person = 135000 := by
  sorry

end each_person_gets_after_taxes_l210_210233


namespace initial_necklaces_15_l210_210767

variable (N E : ℕ)
variable (initial_necklaces : ℕ) (initial_earrings : ℕ) (store_necklaces : ℕ) (store_earrings : ℕ) (mother_earrings : ℕ) (total_jewelry : ℕ)

axiom necklaces_eq_initial : N = initial_necklaces
axiom earrings_eq_15 : E = initial_earrings
axiom initial_earrings_15 : initial_earrings = 15
axiom store_necklaces_eq_initial : store_necklaces = initial_necklaces
axiom store_earrings_eq_23_initial : store_earrings = 2 * initial_earrings / 3
axiom mother_earrings_eq_115_store : mother_earrings = 1 * store_earrings / 5
axiom total_jewelry_is_57 : total_jewelry = 57
axiom jewelry_pieces_eq : 2 * initial_necklaces + initial_earrings + store_earrings + mother_earrings = total_jewelry

theorem initial_necklaces_15 : initial_necklaces = 15 := by
  sorry

end initial_necklaces_15_l210_210767


namespace geometric_sequence_a5_l210_210044

theorem geometric_sequence_a5 {a : ℕ → ℝ} 
  (h_geom : ∃ r, ∀ n, a (n + 1) = r * a n) 
  (h_a3 : a 3 = -4) 
  (h_a7 : a 7 = -16) : 
  a 5 = -8 := 
sorry

end geometric_sequence_a5_l210_210044


namespace maurice_earnings_l210_210538

theorem maurice_earnings (bonus_per_10_tasks : ℕ → ℕ) (num_tasks : ℕ) (total_earnings : ℕ) :
  (∀ n, n * (bonus_per_10_tasks n) = 6 * n) →
  num_tasks = 30 →
  total_earnings = 78 →
  bonus_per_10_tasks num_tasks / 10 = 3 →
  (total_earnings - (bonus_per_10_tasks num_tasks / 10) * 6) / num_tasks = 2 :=
by
  intros h_bonus h_num_tasks h_total_earnings h_bonus_count
  sorry

end maurice_earnings_l210_210538


namespace journey_total_distance_l210_210841

/--
Given:
- A person covers 3/5 of their journey by train.
- A person covers 7/20 of their journey by bus.
- A person covers 3/10 of their journey by bicycle.
- A person covers 1/50 of their journey by taxi.
- The rest of the journey (4.25 km) is covered by walking.

Prove:
  D = 15.74 km
where D is the total distance of the journey.
-/
theorem journey_total_distance :
  ∀ (D : ℝ), 3/5 * D + 7/20 * D + 3/10 * D + 1/50 * D + 4.25 = D → D = 15.74 :=
by
  intro D
  sorry

end journey_total_distance_l210_210841


namespace units_digit_of_product_l210_210295

/-
Problem: What is the units digit of the product of the first three even positive composite numbers?
Conditions: 
- The first three even positive composite numbers are 4, 6, and 8.
Proof: Prove that the units digit of their product is 2.
-/

def even_positive_composite_numbers := [4, 6, 8]
def product := List.foldl (· * ·) 1 even_positive_composite_numbers
def units_digit (n : Nat) := n % 10

theorem units_digit_of_product : units_digit product = 2 := by
  sorry

end units_digit_of_product_l210_210295


namespace least_positive_integer_n_l210_210476

theorem least_positive_integer_n (n : ℕ) (hn : n = 10) :
  (2:ℝ)^(1 / 5 * (n * (n + 1) / 2)) > 1000 :=
by
  sorry

end least_positive_integer_n_l210_210476


namespace geometric_sequence_terms_sum_l210_210194

theorem geometric_sequence_terms_sum :
  ∀ (a_n : ℕ → ℝ) (q : ℝ),
    (∀ n, a_n (n + 1) = a_n n * q) ∧ a_n 1 = 3 ∧
    (a_n 1 + a_n 2 + a_n 3) = 21 →
    (a_n (1 + 2) + a_n (1 + 3) + a_n (1 + 4)) = 84 :=
by
  intros a_n q h
  sorry

end geometric_sequence_terms_sum_l210_210194


namespace area_of_sector_l210_210915

/-- The area of a sector of a circle with radius 10 meters and central angle 42 degrees is 35/3 * pi square meters. -/
theorem area_of_sector (r θ : ℕ) (h_r : r = 10) (h_θ : θ = 42) : 
  (θ / 360 : ℝ) * (Real.pi : ℝ) * (r : ℝ)^2 = (35 / 3 : ℝ) * (Real.pi : ℝ) :=
by {
  sorry
}

end area_of_sector_l210_210915


namespace sum_coords_A_eq_neg9_l210_210529

variable (A B C : ℝ × ℝ)
variable (h1 : (C.1 - A.1)^2 + (C.2 - A.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 / 3)
variable (h2 : (C.1 - B.1)^2 + (C.2 - B.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 / 3)
variable (hB : B = (2, 5))
variable (hC : C = (4, 11))

theorem sum_coords_A_eq_neg9 
  (A B C : ℝ × ℝ)
  (h1 : (C.1 - A.1)^2 + (C.2 - A.2)^2 = (B.1 - A.1)^2 + (B.2 - A.2)^2 / 3)
  (h2 : (C.1 - B.1)^2 + (C.2 - B.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 / 3)
  (hB : B = (2, 5))
  (hC : C = (4, 11)) : 
  A.1 + A.2 = -9 :=
  sorry

end sum_coords_A_eq_neg9_l210_210529


namespace committee_count_l210_210922

theorem committee_count (students : Finset ℕ) (Alice : ℕ) (hAlice : Alice ∈ students) (hCard : students.card = 7) :
  ∃ committees : Finset (Finset ℕ), (∀ c ∈ committees, Alice ∈ c ∧ c.card = 4) ∧ committees.card = 20 :=
sorry

end committee_count_l210_210922


namespace next_month_eggs_l210_210408

-- Given conditions definitions
def eggs_left_last_month : ℕ := 27
def eggs_after_buying : ℕ := 58
def eggs_eaten_this_month : ℕ := 48

-- Calculate number of eggs mother buys each month
def eggs_bought_each_month : ℕ := eggs_after_buying - eggs_left_last_month

-- Remaining eggs before next purchase
def eggs_left_before_next_purchase : ℕ := eggs_after_buying - eggs_eaten_this_month

-- Final amount of eggs after mother buys next month's supply
def total_eggs_next_month : ℕ := eggs_left_before_next_purchase + eggs_bought_each_month

-- Prove the total number of eggs next month equals 41
theorem next_month_eggs : total_eggs_next_month = 41 := by
  sorry

end next_month_eggs_l210_210408


namespace length_of_train_l210_210797

-- Conditions
variable (L E T : ℝ)
axiom h1 : 300 * E = L + 300 * T
axiom h2 : 90 * E = L - 90 * T

-- The statement to be proved
theorem length_of_train : L = 200 * E :=
by
  sorry

end length_of_train_l210_210797


namespace reduced_price_is_55_l210_210751

variables (P R : ℝ) (X : ℕ)

-- Conditions
def condition1 : R = 0.75 * P := sorry
def condition2 : P * X = 1100 := sorry
def condition3 : 0.75 * P * (X + 5) = 1100 := sorry

-- Theorem
theorem reduced_price_is_55 (P R : ℝ) (X : ℕ) (h1 : R = 0.75 * P) (h2 : P * X = 1100) (h3 : 0.75 * P * (X + 5) = 1100) :
  R = 55 :=
sorry

end reduced_price_is_55_l210_210751


namespace chess_tournament_l210_210310

theorem chess_tournament (n : ℕ) (h1 : 10 * 9 * n / 2 = 90) : n = 2 :=
by
  sorry

end chess_tournament_l210_210310


namespace exceeds_500_bacteria_l210_210686

noncomputable def bacteria_count (n : Nat) : Nat :=
  4 * 3^n

theorem exceeds_500_bacteria (n : Nat) (h : 4 * 3^n > 500) : n ≥ 6 :=
by
  sorry

end exceeds_500_bacteria_l210_210686


namespace tan_theta_solution_l210_210835

theorem tan_theta_solution (θ : ℝ)
  (h : 2 * Real.sin (θ + Real.pi / 3) = 3 * Real.sin (Real.pi / 3 - θ)) :
  Real.tan θ = Real.sqrt 3 / 5 := sorry

end tan_theta_solution_l210_210835


namespace triangle_dimensions_l210_210772

theorem triangle_dimensions (a b c : ℕ) (h1 : a > b) (h2 : b > c)
  (h3 : a = 2 * c) (h4 : b - 2 = c) (h5 : 2 * a / 3 = b) :
  a = 12 ∧ b = 8 ∧ c = 6 :=
by
  sorry

end triangle_dimensions_l210_210772


namespace greatest_integer_gcd_6_l210_210571

theorem greatest_integer_gcd_6 (n : ℕ) (h₁ : n < 200) (h₂ : Nat.gcd n 18 = 6) : n = 192 :=
by
  sorry

end greatest_integer_gcd_6_l210_210571


namespace simplify_expression_l210_210413

theorem simplify_expression (a : ℝ) (h : a = -2) : 
  (1 - a / (a + 1)) / (1 / (1 - a ^ 2)) = 1 / 3 :=
by
  subst h
  sorry

end simplify_expression_l210_210413


namespace tagged_fish_ratio_l210_210990

theorem tagged_fish_ratio (tagged_first_catch : ℕ) 
(tagged_second_catch : ℕ) (total_second_catch : ℕ) 
(h1 : tagged_first_catch = 30) (h2 : tagged_second_catch = 2) 
(h3 : total_second_catch = 50) : tagged_second_catch / total_second_catch = 1 / 25 :=
by
  sorry

end tagged_fish_ratio_l210_210990


namespace find_f_minus_two_l210_210793

noncomputable def f : ℝ → ℝ := sorry

axiom f_add : ∀ x y : ℝ, f (x + y) = f x + f y + 2 * x * y
axiom f_one : f 1 = 2

theorem find_f_minus_two : f (-2) = 2 :=
by sorry

end find_f_minus_two_l210_210793


namespace sum_a1_a5_l210_210036

theorem sum_a1_a5 (S : ℕ → ℕ) (a : ℕ → ℕ)
  (hS : ∀ n, S n = n^2 + 1)
  (ha : ∀ n, a (n + 1) = S (n + 1) - S n) :
  a 1 + a 5 = 11 :=
sorry

end sum_a1_a5_l210_210036


namespace speed_against_current_l210_210136

theorem speed_against_current (V_m V_c : ℝ) (h1 : V_m + V_c = 20) (h2 : V_c = 1) :
  V_m - V_c = 18 :=
by
  sorry

end speed_against_current_l210_210136


namespace find_DY_length_l210_210992

noncomputable def angle_bisector_theorem (DE DY EF FY : ℝ) : ℝ :=
  (DE * FY) / EF

theorem find_DY_length :
  ∀ (DE EF FY : ℝ), DE = 26 → EF = 34 → FY = 30 →
  angle_bisector_theorem DE DY EF FY = 22.94 := 
by
  intros
  sorry

end find_DY_length_l210_210992


namespace find_triples_of_positive_integers_l210_210800

theorem find_triples_of_positive_integers :
  ∀ (x y z : ℕ), 2 * (x + y + z + 2 * x * y * z)^2 = (2 * x * y + 2 * y * z + 2 * z * x + 1)^2 + 2023 ↔ 
  (x = 3 ∧ y = 3 ∧ z = 2) ∨
  (x = 3 ∧ y = 2 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = 3) ∨
  (x = 2 ∧ y = 3 ∧ z = 3) ∨
  (x = 3 ∧ y = 2 ∧ z = 3) ∨
  (x = 3 ∧ y = 3 ∧ z = 2) := 
by 
  sorry

end find_triples_of_positive_integers_l210_210800


namespace sphere_volume_ratio_l210_210987

theorem sphere_volume_ratio
  (r R : ℝ)
  (h : (4:ℝ) * π * r^2 / (4 * π * R^2) = (4:ℝ) / 9) : 
  (r^3 / R^3 = (8:ℝ) / 27) := by
  sorry

end sphere_volume_ratio_l210_210987


namespace delta_f_l210_210945

open BigOperators

def f (n : ℕ) : ℕ := ∑ i in Finset.range n, (i + 1) * (n - i)

theorem delta_f (k : ℕ) : f (k + 1) - f k = ∑ i in Finset.range (k + 1), (i + 1) :=
by
  sorry

end delta_f_l210_210945


namespace final_height_of_helicopter_total_fuel_consumed_l210_210796

noncomputable def height_changes : List Float := [4.1, -2.3, 1.6, -0.9, 1.1]

def total_height_change (changes : List Float) : Float :=
  changes.foldl (λ acc x => acc + x) 0

theorem final_height_of_helicopter :
  total_height_change height_changes = 3.6 :=
by
  sorry

noncomputable def fuel_consumption (changes : List Float) : Float :=
  changes.foldl (λ acc x => if x > 0 then acc + 5 * x else acc + 3 * -x) 0

theorem total_fuel_consumed :
  fuel_consumption height_changes = 43.6 :=
by
  sorry

end final_height_of_helicopter_total_fuel_consumed_l210_210796


namespace initial_amount_saved_l210_210338

noncomputable section

def cost_of_couch : ℝ := 750
def cost_of_table : ℝ := 100
def cost_of_lamp : ℝ := 50
def amount_still_owed : ℝ := 400

def total_cost : ℝ := cost_of_couch + cost_of_table + cost_of_lamp

theorem initial_amount_saved (initial_amount : ℝ) :
  initial_amount = total_cost - amount_still_owed ↔ initial_amount = 500 :=
by
  -- the proof is omitted
  sorry

end initial_amount_saved_l210_210338


namespace part1_part2_l210_210086

open Set

def A (a : ℝ) : Set ℝ := { x | x^2 - a * x + a^2 - 19 = 0 }
def B : Set ℝ := { x | x^2 - 5 * x + 6 = 0 }
def C : Set ℝ := { x | x^2 + 2 * x - 8 = 0 }

theorem part1 (a : ℝ) : (A a ∩ B = A a ∪ B) → a = 5 :=
by
  sorry

theorem part2 (a : ℝ) : (A a ∩ B ≠ ∅) ∧ (A a ∩ C = ∅) → a = -2 :=
by
  sorry

end part1_part2_l210_210086


namespace solution_verification_l210_210802

-- Define the differential equation
def diff_eq (y y' y'': ℝ → ℝ) : Prop :=
  ∀ x, y'' x - 4 * y' x + 5 * y x = 2 * Real.cos x + 6 * Real.sin x

-- General solution form
def general_solution (C₁ C₂ : ℝ) (y: ℝ → ℝ) : Prop :=
  ∀ x, y x = Real.exp (2 * x) * (C₁ * Real.cos x + C₂ * Real.sin x) + Real.cos x + 1/2 * Real.sin x

-- Proof problem statement
theorem solution_verification (C₁ C₂ : ℝ) (y y' y'': ℝ → ℝ) :
  (∀ x, y' x = deriv y x) →
  (∀ x, y'' x = deriv (deriv y) x) →
  diff_eq y y' y'' →
  general_solution C₁ C₂ y :=
by
  intros h1 h2 h3
  sorry

end solution_verification_l210_210802


namespace garden_area_l210_210221

theorem garden_area (P b l: ℕ) (hP: P = 900) (hb: b = 190) (hl: l = P / 2 - b):
  l * b = 49400 := 
by
  sorry

end garden_area_l210_210221


namespace log_condition_necessary_not_sufficient_l210_210400

noncomputable def base_of_natural_logarithm := Real.exp 1

variable (a b : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : 0 < b) (h4 : b ≠ 1)

theorem log_condition_necessary_not_sufficient (h : 0 < a ∧ a < b ∧ b < 1) :
  (Real.log 2 / Real.log a > Real.log base_of_natural_logarithm / Real.log b) :=
sorry

end log_condition_necessary_not_sufficient_l210_210400


namespace arithmetic_sequence_max_min_b_l210_210624

-- Define the sequence a_n
def S (n : ℕ) : ℚ := (1/2) * n^2 - 2 * n
def a (n : ℕ) : ℚ := S n - S (n - 1)

-- Question 1: Prove that {a_n} is an arithmetic sequence with a common difference of 1
theorem arithmetic_sequence (n : ℕ) (hn : n ≥ 2) : 
  a n - a (n - 1) = 1 :=
sorry

-- Define the sequence b_n
def b (n : ℕ) : ℚ := (a n + 1) / a n

-- Question 2: Prove that b_3 is the maximum value and b_2 is the minimum value in {b_n}
theorem max_min_b (hn2 : 2 ≥ 1) (hn3 : 3 ≥ 1) : 
  b 3 = 3 ∧ b 2 = -1 :=
sorry

end arithmetic_sequence_max_min_b_l210_210624


namespace parabola_latus_rectum_l210_210472

theorem parabola_latus_rectum (p : ℝ) (H : ∀ y : ℝ, y^2 = 2 * p * -2) : p = 4 :=
sorry

end parabola_latus_rectum_l210_210472


namespace benjamin_distance_l210_210938

def speed := 10  -- Speed in kilometers per hour
def time := 8    -- Time in hours

def distance (s t : ℕ) := s * t  -- Distance formula

theorem benjamin_distance : distance speed time = 80 :=
by
  -- proof omitted
  sorry

end benjamin_distance_l210_210938


namespace odd_f_l210_210357

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 2^x else if x < 0 then -x^2 + 2^(-x) else 0

theorem odd_f (x : ℝ) : (f (-x) = -f x) :=
by
  sorry

end odd_f_l210_210357


namespace value_of_k_h_10_l210_210372

def h (x : ℝ) : ℝ := 4 * x - 5
def k (x : ℝ) : ℝ := 2 * x + 6

theorem value_of_k_h_10 : k (h 10) = 76 := by
  -- We provide only the statement as required, skipping the proof
  sorry

end value_of_k_h_10_l210_210372


namespace desiredCircleEquation_l210_210205

-- Definition of the given circle
def givenCircle (x y : ℝ) : Prop := x^2 + y^2 + x - 6*y + 3 = 0

-- Definition of the given line
def givenLine (x y : ℝ) : Prop := x + 2*y - 3 = 0

-- The required proof problem statement
theorem desiredCircleEquation :
  (∀ P Q : ℝ × ℝ, givenCircle P.1 P.2 ∧ givenLine P.1 P.2 → givenCircle Q.1 Q.2 ∧ givenLine Q.1 Q.2 →
  (P ≠ Q) → 
  (∃ x y : ℝ, x^2 + y^2 + 2*x - 4*y = 0)) :=
by
  -- Proof omitted
  sorry

end desiredCircleEquation_l210_210205


namespace specific_value_correct_l210_210119

noncomputable def specific_value (x : ℝ) : ℝ :=
  (3 / 5) * (x ^ 2)

theorem specific_value_correct :
  specific_value 14.500000000000002 = 126.15000000000002 :=
by
  sorry

end specific_value_correct_l210_210119


namespace rationalize_denominator_l210_210717

theorem rationalize_denominator (a b : ℝ) (h : b = 343) (h_nonzero : b ≠ 0) : (a = 7) → (\sqrt b = 7 * \sqrt 7) → \frac{a}{\sqrt b} = \frac{\sqrt 7}{7} :=
by
  sorry

end rationalize_denominator_l210_210717


namespace compound_interest_time_l210_210187

theorem compound_interest_time (P r CI : ℝ) (n : ℕ) (A : ℝ) :
  P = 16000 ∧ r = 0.15 ∧ CI = 6218 ∧ n = 1 ∧ A = P + CI →
  t = 2 :=
by
  sorry

end compound_interest_time_l210_210187


namespace zamena_inequalities_l210_210778

theorem zamena_inequalities :
  ∃ (M E H A : ℕ), 
    M ≠ E ∧ M ≠ H ∧ M ≠ A ∧ 
    E ≠ H ∧ E ≠ A ∧ H ≠ A ∧
    (∃ Z, Z = 5 ∧ -- Assume Z is 5 based on the conclusion
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ A ∧ A ≤ 5 ∧
    3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A ∧
    -- ZAMENA evaluates to 541234
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) :=
sorry

end zamena_inequalities_l210_210778


namespace exists_rank_with_profit_2016_l210_210708

theorem exists_rank_with_profit_2016 : ∃ n : ℕ, n * (n + 1) / 2 = 2016 :=
by 
  sorry

end exists_rank_with_profit_2016_l210_210708


namespace find_b_l210_210731

def nabla (a b : ℤ) (h : a ≠ b) : ℤ := (a + b) / (a - b)

theorem find_b (b : ℤ) (h : 3 ≠ b) (h_eq : nabla 3 b h = -4) : b = 5 :=
sorry

end find_b_l210_210731


namespace problem_l210_210974

variable (U : Set ℕ) (M : Set ℕ) (N : Set ℕ)

axiom universal_set : U = {1, 2, 3, 4, 5, 6, 7}
axiom set_M : M = {3, 4, 5}
axiom set_N : N = {1, 3, 6}

def complement (U M : Set ℕ) : Set ℕ := {x | x ∈ U ∧ x ∉ M}

theorem problem :
  {1, 6} = (complement U M) ∩ N :=
by
  sorry

end problem_l210_210974


namespace focus_of_parabola_l210_210953

theorem focus_of_parabola :
  (∃ (x y : ℝ), y = 4 * x ^ 2 - 8 * x - 12 ∧ x = 1 ∧ y = -15.9375) :=
by
  sorry

end focus_of_parabola_l210_210953


namespace min_value_of_y_l210_210794

noncomputable def y (x : ℝ) : ℝ := x^2 + 26 * x + 7

theorem min_value_of_y : ∃ x : ℝ, y x = -162 :=
by
  use -13
  sorry

end min_value_of_y_l210_210794


namespace sum_of_coefficients_shifted_function_l210_210916

def original_function (x : ℝ) : ℝ :=
  3*x^2 - 2*x + 6

def shifted_function (x : ℝ) : ℝ :=
  original_function (x + 5)

theorem sum_of_coefficients_shifted_function : 
  let a := 3
  let b := 28
  let c := 71
  a + b + c = 102 :=
by
  -- Placeholder for the proof
  sorry

end sum_of_coefficients_shifted_function_l210_210916


namespace min_h_for_circle_l210_210653

theorem min_h_for_circle (h : ℝ) :
  (∀ x y : ℝ, (x - h)^2 + (y - 1)^2 = 1 → x + y + 1 ≥ 0) →
  h = Real.sqrt 2 - 2 :=
sorry

end min_h_for_circle_l210_210653


namespace books_inequality_system_l210_210630

theorem books_inequality_system (x : ℕ) (n : ℕ) (h1 : x = 5 * n + 6) (h2 : (1 ≤ x - 7 * (x - 6) / 5 + 7)) :
  1 ≤ x - 7 * (x - 6) / 5 + 7 ∧ x - 7 * (x - 6) / 5 + 7 < 7 := 
by
  sorry

end books_inequality_system_l210_210630


namespace product_of_digits_l210_210679

theorem product_of_digits (n A B : ℕ) (h1 : n % 6 = 0) (h2 : A + B = 12) (h3 : n = 10 * A + B) : 
  (A * B = 32 ∨ A * B = 36) :=
by 
  sorry

end product_of_digits_l210_210679


namespace snack_eaters_left_after_second_newcomers_l210_210923

theorem snack_eaters_left_after_second_newcomers
  (initial_snackers : ℕ)
  (new_outsiders_1 : ℕ)
  (half_left_1 : ℕ)
  (new_outsiders_2 : ℕ)
  (final_snackers : ℕ)
  (H1 : initial_snackers = 100)
  (H2 : new_outsiders_1 = 20)
  (H3 : half_left_1 = (initial_snackers + new_outsiders_1) / 2)
  (H4 : new_outsiders_2 = 10)
  (H5 : final_snackers = 20)
  : (initial_snackers + new_outsiders_1 - half_left_1 + new_outsiders_2 - (initial_snackers + new_outsiders_1 - half_left_1 + new_outsiders_2 - final_snackers * 2)) = 30 :=
by 
  sorry

end snack_eaters_left_after_second_newcomers_l210_210923


namespace y_exceeds_x_by_100_percent_l210_210753

theorem y_exceeds_x_by_100_percent (x y : ℝ) (h : x = 0.5 * y) : (y - x) / x = 1 := by
sorry

end y_exceeds_x_by_100_percent_l210_210753


namespace volume_tetrahedron_O_l210_210558

variable {d e f : ℝ}

theorem volume_tetrahedron_O DEF : 
  d^2 + e^2 = 64 →
  e^2 + f^2 = 100 →
  f^2 + d^2 = 144 →
  (1 / 6 * real.sqrt d * real.sqrt e * real.sqrt f : ℝ) = 110 / 3 :=
by
  intros h₁ h₂ h₃
  sorry

end volume_tetrahedron_O_l210_210558


namespace multiples_of_7_with_unit_digit_7_and_lt_150_l210_210834

/--
Given a positive integer n,
Let \(7n\) be a multiple of 7,
\(7n < 150\),
and \(7n \mod 10 = 7\),
prove there are 11 such multiples of 7.
-/
theorem multiples_of_7_with_unit_digit_7_and_lt_150 :
  ∃! (n : ℕ), n ≤ 11 ∧ ∀ (m : ℕ), (1 ≤ m ∧ m ≤ n) → (7 * (2 * m - 1) < 150 ∧ (7 * (2 * m - 1)) % 10 = 7) := by
sorry

end multiples_of_7_with_unit_digit_7_and_lt_150_l210_210834


namespace factor_81_sub_27x3_l210_210181

theorem factor_81_sub_27x3 (x : ℝ) : 81 - 27 * x^3 = 3 * (3 - x) * (81 + 27 * x + 9 * x^2) :=
sorry

end factor_81_sub_27x3_l210_210181


namespace frog_climb_time_l210_210149

-- Definitions related to the problem
def well_depth : ℕ := 12
def climb_per_cycle : ℕ := 3
def slip_per_cycle : ℕ := 1
def effective_climb_per_cycle : ℕ := climb_per_cycle - slip_per_cycle

-- Time taken for each activity
def time_to_climb : ℕ := 10 -- given as t
def time_to_slip : ℕ := time_to_climb / 3
def total_time_per_cycle : ℕ := time_to_climb + time_to_slip

-- Condition specifying the observed frog position at a certain time
def observed_time : ℕ := 17 -- minutes since 8:00
def observed_position : ℕ := 9 -- meters climbed since it's 3 meters from the top of the well (well_depth - 3)

-- The main theorem stating the total time taken to climb to the top of the well
theorem frog_climb_time : 
  ∃ (k : ℕ), k * effective_climb_per_cycle + climb_per_cycle = well_depth ∧ k * total_time_per_cycle + time_to_climb = 22 := 
sorry

end frog_climb_time_l210_210149


namespace maximum_xy_l210_210495

theorem maximum_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 4 * x^2 + 9 * y^2 + 3 * x * y = 30) : xy ≤ 2 :=
sorry

end maximum_xy_l210_210495


namespace unique_positive_integer_solution_l210_210628

-- Definitions of the given points
def P1 : ℚ × ℚ := (4, 11)
def P2 : ℚ × ℚ := (16, 1)

-- Definition for the line equation in standard form
def line_equation (x y : ℤ) : Prop := 5 * x + 6 * y = 43

-- Proof for the existence of only one solution with positive integer coordinates
theorem unique_positive_integer_solution :
  ∃ P : ℤ × ℤ, P.1 > 0 ∧ P.2 > 0 ∧ line_equation P.1 P.2 ∧ (∀ Q : ℤ × ℤ, line_equation Q.1 Q.2 → Q.1 > 0 ∧ Q.2 > 0 → Q = (5, 3)) :=
by 
  sorry

end unique_positive_integer_solution_l210_210628


namespace reliability_is_correct_l210_210069

-- Define the probabilities of each switch functioning properly.
def P_A : ℝ := 0.9
def P_B : ℝ := 0.8
def P_C : ℝ := 0.7

-- Define the system reliability.
def reliability : ℝ := P_A * P_B * P_C

-- The theorem stating the reliability of the system.
theorem reliability_is_correct : reliability = 0.504 := by
  sorry

end reliability_is_correct_l210_210069


namespace visits_exactly_two_friends_l210_210006

theorem visits_exactly_two_friends (a_visits b_visits c_visits vacation_period : ℕ) (full_period days : ℕ)
(h_a : a_visits = 4)
(h_b : b_visits = 5)
(h_c : c_visits = 6)
(h_vacation : vacation_period = 30)
(h_full_period : full_period = Nat.lcm (Nat.lcm a_visits b_visits) c_visits)
(h_days : days = 360)
(h_start_vacation : ∀ n, ∃ k, n = k * vacation_period + 30):
  ∃ n, n = 24 :=
by {
  sorry
}

end visits_exactly_two_friends_l210_210006


namespace average_speed_l210_210113

-- Define the conditions given in the problem
def distance_first_hour : ℕ := 50 -- distance traveled in the first hour
def distance_second_hour : ℕ := 60 -- distance traveled in the second hour
def total_distance : ℕ := distance_first_hour + distance_second_hour -- total distance traveled

-- Define the total time
def total_time : ℕ := 2 -- total time in hours

-- The problem statement: proving the average speed
theorem average_speed : total_distance / total_time = 55 := by
  unfold total_distance total_time
  sorry

end average_speed_l210_210113


namespace Maddie_bought_palettes_l210_210710

-- Defining constants and conditions as per the problem statement.
def cost_per_palette : ℝ := 15
def number_of_lipsticks : ℝ := 4
def cost_per_lipstick : ℝ := 2.50
def number_of_hair_boxes : ℝ := 3
def cost_per_hair_box : ℝ := 4
def total_paid : ℝ := 67

-- Defining the condition which we need to prove for number of makeup palettes bought.
theorem Maddie_bought_palettes (P : ℝ) :
  (number_of_lipsticks * cost_per_lipstick) +
  (number_of_hair_boxes * cost_per_hair_box) +
  (cost_per_palette * P) = total_paid →
  P = 3 :=
sorry

end Maddie_bought_palettes_l210_210710


namespace harkamal_payment_l210_210308

theorem harkamal_payment :
  let grapes_kg := 9
  let grape_rate_per_kg := 70
  let mangoes_kg := 9
  let mango_rate_per_kg := 55
  let cost_of_grapes := grapes_kg * grape_rate_per_kg
  let cost_of_mangoes := mangoes_kg * mango_rate_per_kg
  let total_payment := cost_of_grapes + cost_of_mangoes
  total_payment = 1125 :=
by
  let grapes_kg := 9
  let grape_rate_per_kg := 70
  let mangoes_kg := 9
  let mango_rate_per_kg := 55
  let cost_of_grapes := grapes_kg * grape_rate_per_kg
  let cost_of_mangoes := mangoes_kg * mango_rate_per_kg
  let total_payment := cost_of_grapes + cost_of_mangoes
  sorry

end harkamal_payment_l210_210308


namespace boat_speed_in_still_water_l210_210439

def speed_of_boat (V_b : ℝ) : Prop :=
  let stream_speed := 4  -- speed of the stream in km/hr
  let downstream_distance := 168  -- distance traveled downstream in km
  let time := 6  -- time taken to travel downstream in hours
  (downstream_distance = (V_b + stream_speed) * time)

theorem boat_speed_in_still_water : ∃ V_b, speed_of_boat V_b ∧ V_b = 24 := 
by
  exists 24
  unfold speed_of_boat
  simp
  sorry

end boat_speed_in_still_water_l210_210439


namespace find_b_l210_210104

theorem find_b (b : ℚ) (H : ∃ x y : ℚ, x = 3 ∧ y = -7 ∧ b * x + (b - 1) * y = b + 3) : 
  b = 4 / 5 := 
by
  sorry

end find_b_l210_210104


namespace gamesNextMonth_l210_210392

def gamesThisMonth : ℕ := 11
def gamesLastMonth : ℕ := 17
def totalPlannedGames : ℕ := 44

theorem gamesNextMonth :
  (totalPlannedGames - (gamesThisMonth + gamesLastMonth) = 16) :=
by
  unfold totalPlannedGames
  unfold gamesThisMonth
  unfold gamesLastMonth
  sorry

end gamesNextMonth_l210_210392


namespace karl_total_income_is_53_l210_210081

noncomputable def compute_income (tshirt_price pant_price skirt_price sold_tshirts sold_pants sold_skirts sold_refurbished_tshirts: ℕ) : ℝ :=
  let tshirt_income := 2 * tshirt_price
  let pant_income := sold_pants * pant_price
  let skirt_income := sold_skirts * skirt_price
  let refurbished_tshirt_price := (tshirt_price : ℝ) / 2
  let refurbished_tshirt_income := sold_refurbished_tshirts * refurbished_tshirt_price
  tshirt_income + pant_income + skirt_income + refurbished_tshirt_income

theorem karl_total_income_is_53 : compute_income 5 4 6 2 1 4 6 = 53 := by
  sorry

end karl_total_income_is_53_l210_210081


namespace multiples_of_7_with_unit_digit_7_and_lt_150_l210_210833

/--
Given a positive integer n,
Let \(7n\) be a multiple of 7,
\(7n < 150\),
and \(7n \mod 10 = 7\),
prove there are 11 such multiples of 7.
-/
theorem multiples_of_7_with_unit_digit_7_and_lt_150 :
  ∃! (n : ℕ), n ≤ 11 ∧ ∀ (m : ℕ), (1 ≤ m ∧ m ≤ n) → (7 * (2 * m - 1) < 150 ∧ (7 * (2 * m - 1)) % 10 = 7) := by
sorry

end multiples_of_7_with_unit_digit_7_and_lt_150_l210_210833


namespace river_depth_mid_may_l210_210688

-- Definitions corresponding to the conditions
def depth_mid_june (D : ℕ) : ℕ := D + 10
def depth_mid_july (D : ℕ) : ℕ := 3 * (depth_mid_june D)

-- The theorem statement
theorem river_depth_mid_may (D : ℕ) (h : depth_mid_july D = 45) : D = 5 :=
by
  sorry

end river_depth_mid_may_l210_210688


namespace sand_exchange_impossible_to_achieve_l210_210381

-- Let G and P be the initial weights of gold and platinum sand, respectively
def initial_G : ℕ := 1 -- 1 kg
def initial_P : ℕ := 1 -- 1 kg

-- Initial values for g and p
def initial_g : ℕ := 1001
def initial_p : ℕ := 1001

-- Daily reduction of either g or p
axiom decrease_g_or_p (g p : ℕ) : g > 1 ∨ p > 1 → (g = g - 1 ∨ p = p - 1) ∧ (g ≥ 1) ∧ (p ≥ 1)

-- Final condition: after 2000 days, g and p both equal to 1
axiom final_g_p_after_2000_days : ∀ (g p : ℕ), (g = initial_g - 2000) ∧ (p = initial_p - 2000) → g = 1 ∧ p = 1

-- State of the system, defined as S = G * p + P * g
def S (G P g p : ℕ) : ℕ := G * p + P * g

-- Prove that after 2000 days, the banker cannot have at least 2 kg of each type of sand
theorem sand_exchange_impossible_to_achieve (G P g p : ℕ) (h : G = initial_G) (h1 : P = initial_P) 
  (h2 : g = initial_g) (h3 : p = initial_p) : 
  ∀ (d : ℕ), (d = 2000) → (g = 1) ∧ (p = 1) 
    → (S G P g p < 4) :=
by
  sorry

end sand_exchange_impossible_to_achieve_l210_210381


namespace faster_train_cross_time_l210_210288

/-- Statement of the problem in Lean 4 -/
theorem faster_train_cross_time :
  let speed_faster_train_kmph := 72
  let speed_slower_train_kmph := 36
  let length_faster_train_m := 180
  let relative_speed_kmph := speed_faster_train_kmph - speed_slower_train_kmph
  let relative_speed_mps := relative_speed_kmph * (5 / 18 : ℝ)
  let time_taken := length_faster_train_m / relative_speed_mps
  time_taken = 18 :=
by
  sorry

end faster_train_cross_time_l210_210288


namespace ratio_volumes_of_spheres_l210_210984

theorem ratio_volumes_of_spheres (r R : ℝ) (hratio : (4 * π * r^2) / (4 * π * R^2) = 4 / 9) :
    (4 / 3 * π * r^3) / (4 / 3 * π * R^3) = 8 / 27 := 
by {
  sorry
}

end ratio_volumes_of_spheres_l210_210984


namespace find_a_l210_210467

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x > 2 then x^2 - 4 else |x - 3| + a

theorem find_a (a : ℝ) (h : f (f (Real.sqrt 6) a) a = 3) : a = 2 := by
  sorry

end find_a_l210_210467


namespace choose_five_from_eight_l210_210511

theorem choose_five_from_eight : Nat.choose 8 5 = 56 :=
by
  sorry 

end choose_five_from_eight_l210_210511


namespace SomeAthletesNotHonorSociety_l210_210935

variable (Athletes HonorSociety : Type)
variable (Discipline : Athletes → Prop)
variable (isMember : Athletes → HonorSociety → Prop)

-- Some athletes are not disciplined
axiom AthletesNotDisciplined : ∃ a : Athletes, ¬Discipline a

-- All members of the honor society are disciplined
axiom AllHonorSocietyDisciplined : ∀ h : HonorSociety, ∀ a : Athletes, isMember a h → Discipline a

-- The theorem to be proved
theorem SomeAthletesNotHonorSociety : ∃ a : Athletes, ∀ h : HonorSociety, ¬isMember a h :=
  sorry

end SomeAthletesNotHonorSociety_l210_210935


namespace roots_of_quadratic_l210_210845

theorem roots_of_quadratic (b c : ℝ) (h1 : 1 + -2 = -b) (h2 : 1 * -2 = c) : b = 1 ∧ c = -2 :=
by
  sorry

end roots_of_quadratic_l210_210845


namespace simplify_eval_l210_210095

variable (x : ℝ)
def expr := 8 * x^2 - (x - 2) * (3 * x + 1) - 2 * (x + 1) * (x - 1)

theorem simplify_eval (h : x = -2) : expr x = 6 := by
  sorry

end simplify_eval_l210_210095


namespace factor_expression_l210_210633

theorem factor_expression (y : ℝ) : 49 - 16*y^2 + 8*y = (7 - 4*y)*(7 + 4*y) := 
sorry

end factor_expression_l210_210633


namespace prove_ZAMENA_l210_210782

theorem prove_ZAMENA :
  ∃ (A M E H Z : ℕ),
    1 ≤ A ∧ A ≤ 5 ∧
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    1 ≤ Z ∧ Z ≤ 5 ∧
    (3 > A) ∧ (A > M) ∧ (M < E) ∧ (E < H) ∧ (H < A) ∧
    (A ≠ M) ∧ (A ≠ E) ∧ (A ≠ H) ∧ (A ≠ Z) ∧
    (M ≠ E) ∧ (M ≠ H) ∧ (M ≠ Z) ∧
    (E ≠ H) ∧ (E ≠ Z) ∧
    (H ≠ Z) ∧
    Z = 5 ∧
    Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234 :=
begin
  sorry
end

end prove_ZAMENA_l210_210782


namespace combined_sum_of_interior_numbers_of_eighth_and_ninth_rows_l210_210277

theorem combined_sum_of_interior_numbers_of_eighth_and_ninth_rows :
  (2 ^ (8 - 1) - 2) + (2 ^ (9 - 1) - 2) = 380 :=
by
  -- The steps of the proof would go here, but for the purpose of this task:
  sorry

end combined_sum_of_interior_numbers_of_eighth_and_ninth_rows_l210_210277


namespace relationship_between_vars_l210_210053

variable {α : Type*} [LinearOrderedAddCommGroup α]

theorem relationship_between_vars (a b : α) 
  (h1 : a + b < 0) 
  (h2 : b > 0) : a < -b ∧ -b < b ∧ b < -a :=
by
  sorry

end relationship_between_vars_l210_210053


namespace school_pupils_l210_210066

def girls : ℕ := 868
def difference : ℕ := 281
def boys (g b : ℕ) : Prop := g = b + difference
def total_pupils (g b t : ℕ) : Prop := t = g + b

theorem school_pupils : 
  ∃ b t, boys girls b ∧ total_pupils girls b t ∧ t = 1455 :=
by
  sorry

end school_pupils_l210_210066


namespace tan_20_plus_tan_40_sin_50_times_1_plus_sqrt3_tan_10_l210_210724

noncomputable def proof1 : Prop :=
  tan (20 * (π / 180)) + tan (40 * (π / 180)) + real.sqrt 3 * tan (20 * (π / 180)) * tan (40 * (π / 180)) = real.sqrt 3

noncomputable def proof2 : Prop :=
  sin (50 * (π / 180)) * (1 + real.sqrt 3 * tan (10 * (π / 180))) = 1

theorem tan_20_plus_tan_40 (h1 : proof1) : true :=
by sorry

theorem sin_50_times_1_plus_sqrt3_tan_10 (h2 : proof2) : true :=
by sorry

end tan_20_plus_tan_40_sin_50_times_1_plus_sqrt3_tan_10_l210_210724


namespace professors_women_tenured_or_both_l210_210165

variable (professors : ℝ) -- Total number of professors as percentage
variable (women tenured men_tenured tenured_women : ℝ) -- Given percentages

-- Conditions
variables (hw : women = 0.69 * professors) 
          (ht : tenured = 0.7 * professors)
          (hm_t : men_tenured = 0.52 * (1 - women) * professors)
          (htw : tenured_women = tenured - men_tenured)
          
-- The statement to prove
theorem professors_women_tenured_or_both :
  women + tenured - tenured_women = 0.8512 * professors :=
by
  sorry

end professors_women_tenured_or_both_l210_210165


namespace monotone_decreasing_interval_l210_210363

noncomputable def f (x : ℝ) : ℝ := -2 * real.sin (2 * x + π / 4)

theorem monotone_decreasing_interval :
  f(π / 8) = -2 →
  ∃ a b : ℝ, [a, b] = [π / 8, 5 * π / 8] ∧
    ∀ x y : ℝ, a ≤ x ∧ x ≤ y ∧ y ≤ b → f(y) ≤ f(x) :=
begin
  intros h,
  use [π / 8, 5 * π / 8],
  split,
  refl,
  intros x y hx hy hxy,
  sorry  -- Proof to be completed
end

end monotone_decreasing_interval_l210_210363


namespace gcd_of_powers_of_two_minus_one_l210_210560

theorem gcd_of_powers_of_two_minus_one : 
  gcd (2^1015 - 1) (2^1020 - 1) = 1 :=
sorry

end gcd_of_powers_of_two_minus_one_l210_210560


namespace financing_amount_correct_l210_210367

-- Define the conditions
def monthly_payment : ℕ := 150
def years : ℕ := 5
def months_per_year : ℕ := 12

-- Define the total financed amount
def total_financed : ℕ := monthly_payment * years * months_per_year

-- The statement that we need to prove
theorem financing_amount_correct : total_financed = 9000 := 
by
  sorry

end financing_amount_correct_l210_210367


namespace quadratic_distinct_roots_l210_210497

theorem quadratic_distinct_roots (k : ℝ) : 
  k < 5 ∧ k ≠ 1 ↔ ∃ x : ℝ, (k-1)*x^2 + 4*x + 1 = 0 ∧ ∃ y : ℝ, y ≠ x ∧ (k-1)*y^2 + 4*y + 1 = 0 :=
begin
  sorry
end

end quadratic_distinct_roots_l210_210497


namespace ZAMENA_correct_l210_210779

-- Define the digits and assigning them to the letters
noncomputable def A : ℕ := 2
noncomputable def M : ℕ := 1
noncomputable def E : ℕ := 2
noncomputable def H : ℕ := 3

-- Define the number ZAMENA
noncomputable def ZAMENA : ℕ :=
  5 * 10^5 + A * 10^4 + M * 10^3 + E * 10^2 + H * 10 + A

-- Define the inequalities and constraints
lemma satisfies_inequalities  : 3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A := by
  unfold A M E H
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

-- Define uniqueness of each digit set
lemma unique_digits: A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ M ≠ E ∧ M ≠ H ∧ E ≠ H := by
  unfold A M E H
  exact ⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩

-- Prove that the number ZAMENA matches the specified number
theorem ZAMENA_correct : ZAMENA = 541234 := by
  unfold ZAMENA A M E H
  norm_num

end ZAMENA_correct_l210_210779


namespace cost_of_12_cheaper_fruits_l210_210524

-- Defining the price per 10 apples in cents.
def price_per_10_apples : ℕ := 200

-- Defining the price per 5 oranges in cents.
def price_per_5_oranges : ℕ := 150

-- No bulk discount means per item price is just total cost divided by the number of items
def price_per_apple := price_per_10_apples / 10
def price_per_orange := price_per_5_oranges / 5

-- Given the calculation steps, we have to prove that the cost for 12 cheaper fruits (apples) is 240
theorem cost_of_12_cheaper_fruits : 12 * price_per_apple = 240 := by
  -- This step performs the proof, which we skip with sorry
  sorry

end cost_of_12_cheaper_fruits_l210_210524


namespace combination_eight_choose_five_l210_210505

theorem combination_eight_choose_five : nat.choose 8 5 = 56 :=
by sorry

end combination_eight_choose_five_l210_210505


namespace isosceles_triangle_large_angles_l210_210994

theorem isosceles_triangle_large_angles (y : ℝ) (h : 2 * y + 40 = 180) : y = 70 :=
by
  sorry

end isosceles_triangle_large_angles_l210_210994


namespace geometric_sequence_sum_l210_210898

noncomputable theory

-- Define the conditions and the problem statement
theorem geometric_sequence_sum (a r : ℝ) (h₁ : a * (1 - r^3000) / (1 - r) = 500) 
  (h₂ : a * (1 - r^6000) / (1 - r) = 950) : 
  a * (1 - r^9000) / (1 - r) = 1355 := 
sorry

end geometric_sequence_sum_l210_210898


namespace riding_time_fraction_l210_210582

-- Definitions for conditions
def M : ℕ := 6
def total_days : ℕ := 6
def max_time_days : ℕ := 2
def part_time_days : ℕ := 2
def fixed_time : ℝ := 1.5
def total_riding_time : ℝ := 21

-- Prove the statement
theorem riding_time_fraction :
  ∃ F : ℝ, 2 * M + 2 * fixed_time + 2 * F * M = total_riding_time ∧ F = 0.5 :=
by
  exists 0.5
  sorry

end riding_time_fraction_l210_210582


namespace value_of_expression_l210_210130

theorem value_of_expression (x : ℝ) (hx : x = -2) : (3 * x + 4) ^ 2 = 4 :=
by
  sorry

end value_of_expression_l210_210130


namespace find_angle_l210_210952

theorem find_angle (A : ℝ) (deg_to_rad : ℝ) :
  (1/2 * Real.sin (A / 2 * deg_to_rad) + Real.cos (A / 2 * deg_to_rad) = 1) →
  (A = 360) :=
sorry

end find_angle_l210_210952


namespace rain_in_both_areas_l210_210500

variable (P1 P2 : ℝ)
variable (hP1 : 0 < P1 ∧ P1 < 1)
variable (hP2 : 0 < P2 ∧ P2 < 1)

theorem rain_in_both_areas :
  ∀ P1 P2, (0 < P1 ∧ P1 < 1) → (0 < P2 ∧ P2 < 1) → (1 - P1) * (1 - P2) = (1 - P1) * (1 - P2) :=
by
  intros P1 P2 hP1 hP2
  sorry

end rain_in_both_areas_l210_210500


namespace intersection_complement_M_N_eq_456_l210_210668

def UniversalSet := { n : ℕ | 1 ≤ n ∧ n < 9 }
def M : Set ℕ := { 1, 2, 3 }
def N : Set ℕ := { 3, 4, 5, 6 }

theorem intersection_complement_M_N_eq_456 : 
  (UniversalSet \ M) ∩ N = { 4, 5, 6 } :=
by
  sorry

end intersection_complement_M_N_eq_456_l210_210668


namespace rectangle_area_error_percent_l210_210849

theorem rectangle_area_error_percent 
  (L W : ℝ)
  (hL: L > 0)
  (hW: W > 0) :
  let original_area := L * W
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let measured_area := measured_length * measured_width
  let error := measured_area - original_area
  let error_percent := (error / original_area) * 100
  error_percent = 0.7 := by
  let original_area := L * W
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let measured_area := measured_length * measured_width
  let error := measured_area - original_area
  let error_percent := (error / original_area) * 100
  sorry

end rectangle_area_error_percent_l210_210849


namespace forum_members_l210_210769

theorem forum_members (M : ℕ)
  (h1 : ∀ q a, a = 3 * q)
  (h2 : ∀ h d, q = 3 * h * d)
  (h3 : 24 * (M * 3 * (24 + 3 * 72)) = 57600) : M = 200 :=
by
  sorry

end forum_members_l210_210769


namespace chef_initial_potatoes_l210_210597

theorem chef_initial_potatoes (fries_per_potato : ℕ) (total_fries_needed : ℕ) (leftover_potatoes : ℕ) 
  (H1 : fries_per_potato = 25) 
  (H2 : total_fries_needed = 200) 
  (H3 : leftover_potatoes = 7) : 
  (total_fries_needed / fries_per_potato + leftover_potatoes = 15) :=
by
  sorry

end chef_initial_potatoes_l210_210597


namespace C_must_be_2_l210_210337

-- Define the given digits and their sum conditions
variables (A B C D : ℤ)

-- The sum of known digits for the first number
def sum1_known_digits := 7 + 4 + 5 + 2

-- The sum of known digits for the second number
def sum2_known_digits := 3 + 2 + 6 + 5

-- The first number must be divisible by 3
def divisible_by_3 (n : ℤ) : Prop := n % 3 = 0

-- Conditions for the divisibility by 3 of both numbers
def conditions := divisible_by_3 (sum1_known_digits + A + B + D) ∧ 
                  divisible_by_3 (sum2_known_digits + A + B + C)

-- The statement of the theorem
theorem C_must_be_2 (A B D : ℤ) (h : conditions A B 2 D) : C = 2 :=
  sorry

end C_must_be_2_l210_210337


namespace fruit_basket_count_l210_210978

theorem fruit_basket_count :
  let apples := 6
  let oranges := 8
  let min_apples := 2
  let min_fruits := 1
  (0 <= oranges ∧ oranges <= 8) ∧ (min_apples <= apples ∧ apples <= 6) ∧ (min_fruits <= (apples + oranges)) →
  (5 * 9 = 45) :=
by
  intro h
  sorry

end fruit_basket_count_l210_210978


namespace thirty_percent_greater_l210_210434

theorem thirty_percent_greater (x : ℝ) (h : x = 1.3 * 88) : x = 114.4 :=
sorry

end thirty_percent_greater_l210_210434


namespace union_when_m_equals_4_subset_implies_m_range_l210_210404

-- Define the sets and conditions
def set_A := { x : ℝ | -2 ≤ x ∧ x ≤ 5 }
def set_B (m : ℝ) := { x : ℝ | m + 1 ≤ x ∧ x ≤ 2 * m - 1 }

-- Problem 1: When m = 4, find the union of A and B
theorem union_when_m_equals_4 : ∀ x, x ∈ set_A ∪ set_B 4 ↔ -2 ≤ x ∧ x ≤ 7 :=
by sorry

-- Problem 2: If B ⊆ A, find the range of the real number m
theorem subset_implies_m_range (m : ℝ) : (∀ x, x ∈ set_B m → x ∈ set_A) ↔ m ≤ 3 :=
by sorry

end union_when_m_equals_4_subset_implies_m_range_l210_210404


namespace jason_current_money_l210_210398

/-- Definition of initial amounts and earnings -/
def fred_initial : ℕ := 49
def jason_initial : ℕ := 3
def fred_current : ℕ := 112
def jason_earned : ℕ := 60

/-- The main theorem -/
theorem jason_current_money : jason_initial + jason_earned = 63 := 
by
  -- proof omitted for this example
  sorry

end jason_current_money_l210_210398


namespace left_handed_rock_music_lovers_l210_210501

theorem left_handed_rock_music_lovers (total_club_members left_handed_members rock_music_lovers right_handed_dislike_rock: ℕ)
  (h1 : total_club_members = 25)
  (h2 : left_handed_members = 10)
  (h3 : rock_music_lovers = 18)
  (h4 : right_handed_dislike_rock = 3)
  (h5 : total_club_members = left_handed_members + (total_club_members - left_handed_members))
  : (∃ x : ℕ, x = 6 ∧ x + (left_handed_members - x) + (rock_music_lovers - x) + right_handed_dislike_rock = total_club_members) :=
sorry

end left_handed_rock_music_lovers_l210_210501


namespace negation_of_proposition_l210_210876

theorem negation_of_proposition :
  (¬ ∃ x : ℝ, x^2 - x - 1 = 0) ↔ (∀ x : ℝ, x^2 - x - 1 ≠ 0) :=
by sorry

end negation_of_proposition_l210_210876


namespace weight_loss_comparison_l210_210620

-- Define the conditions
def weight_loss_Barbi : ℝ := 1.5 * 24
def weight_loss_Luca : ℝ := 9 * 15
def weight_loss_Kim : ℝ := (2 * 12) + (3 * 60)

-- Define the combined weight loss of Luca and Kim
def combined_weight_loss_Luca_Kim : ℝ := weight_loss_Luca + weight_loss_Kim

-- Define the difference in weight loss between Luca and Kim combined and Barbi
def weight_loss_difference : ℝ := combined_weight_loss_Luca_Kim - weight_loss_Barbi

-- State the theorem to be proved
theorem weight_loss_comparison : weight_loss_difference = 303 := by
  sorry

end weight_loss_comparison_l210_210620


namespace Tyler_CDs_count_l210_210738

-- Definitions using the conditions
def initial_CDs := 21
def fraction_given_away := 1 / 3
def CDs_bought := 8

-- The problem is to prove the final number of CDs Tyler has
theorem Tyler_CDs_count (initial_CDs : ℕ) 
  (fraction_given_away : ℚ) -- using ℚ for fractions
  (CDs_bought : ℕ) : 
  let CDs_given_away := initial_CDs * fraction_given_away in
  let remaining_CDs := initial_CDs - CDs_given_away in
  let final_CDs := remaining_CDs + CDs_bought in
  final_CDs = 22 := 
by 
  sorry

end Tyler_CDs_count_l210_210738


namespace arithmetic_sequence_geometric_sum_l210_210705

theorem arithmetic_sequence_geometric_sum (a₁ a₂ d : ℕ) (h₁ : d ≠ 0) 
    (h₂ : (2 * a₁ + d)^2 = a₁ * (4 * a₁ + 6 * d)) :
    a₂ = 3 * a₁ :=
by
  sorry

end arithmetic_sequence_geometric_sum_l210_210705


namespace function_properties_l210_210303

theorem function_properties (f : ℝ → ℝ) : 
  (∀ x1 x2, f (x1 * x2) = f x1 * f x2) ∧
  (∀ x, 0 < x → (deriv f x) > 0) ∧
  (∀ x, deriv f (-x) = -(deriv f x)) → 
  f = (λ x, x ^ 2)
by
  sorry

end function_properties_l210_210303


namespace find_two_digit_number_l210_210634

theorem find_two_digit_number : ∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 10 * x + y = x^3 + y^2 ∧ 10 * x + y = 24 := by
  sorry

end find_two_digit_number_l210_210634


namespace periodic_odd_function_value_at_7_l210_210241

noncomputable def f : ℝ → ℝ := sorry -- Need to define f appropriately, skipped for brevity

theorem periodic_odd_function_value_at_7
    (f_odd : ∀ x : ℝ, f (-x) = -f x)
    (f_periodic : ∀ x : ℝ, f (x + 4) = f x)
    (f_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x) :
    f 7 = -1 := sorry

end periodic_odd_function_value_at_7_l210_210241


namespace parallel_lines_solution_l210_210669

theorem parallel_lines_solution (a : ℝ) :
  (∃ (k1 k2 : ℝ), k1 ≠ 0 ∧ k2 ≠ 0 ∧ 
  ∀ x y : ℝ, x + a^2 * y + 6 = 0 → k1*y = x ∧ 
             (a-2) * x + 3 * a * y + 2 * a = 0 → k2*y = x) 
  → (a = -1 ∨ a = 0) :=
by
  sorry

end parallel_lines_solution_l210_210669


namespace points_satisfying_clubsuit_l210_210627

def clubsuit (a b : ℝ) : ℝ := a^2 * b + a * b^2

theorem points_satisfying_clubsuit (x y : ℝ) :
  clubsuit x y = clubsuit y x ↔ (x = 0 ∨ y = 0 ∨ x + y = 0) :=
by
  sorry

end points_satisfying_clubsuit_l210_210627


namespace josephine_total_milk_l210_210262

-- Define the number of containers and the amount of milk they hold
def cnt_1 : ℕ := 3
def qty_1 : ℚ := 2

def cnt_2 : ℕ := 2
def qty_2 : ℚ := 0.75

def cnt_3 : ℕ := 5
def qty_3 : ℚ := 0.5

-- Define the total amount of milk sold
def total_milk_sold : ℚ := cnt_1 * qty_1 + cnt_2 * qty_2 + cnt_3 * qty_3

theorem josephine_total_milk : total_milk_sold = 10 := by
  -- This is the proof placeholder
  sorry

end josephine_total_milk_l210_210262


namespace sphere_volume_ratio_l210_210986

theorem sphere_volume_ratio
  (r R : ℝ)
  (h : (4:ℝ) * π * r^2 / (4 * π * R^2) = (4:ℝ) / 9) : 
  (r^3 / R^3 = (8:ℝ) / 27) := by
  sorry

end sphere_volume_ratio_l210_210986


namespace sum_of_roots_l210_210193

theorem sum_of_roots : 
  ∀ x1 x2 : ℝ, 
  (x1^2 + 2023*x1 = 2024 ∧ x2^2 + 2023*x2 = 2024) → 
  x1 + x2 = -2023 := 
by 
  sorry

end sum_of_roots_l210_210193


namespace num_correct_props_geometric_sequence_l210_210360

-- Define what it means to be a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Original Proposition P
def Prop_P (a : ℕ → ℝ) :=
  a 1 < a 2 ∧ a 2 < a 3 → ∀ n : ℕ, a n < a (n + 1)

-- Converse of Proposition P
def Conv_Prop_P (a : ℕ → ℝ) :=
  ( ∀ n : ℕ, a n < a (n + 1) ) → a 1 < a 2 ∧ a 2 < a 3

-- Inverse of Proposition P
def Inv_Prop_P (a : ℕ → ℝ) :=
  ¬(a 1 < a 2 ∧ a 2 < a 3) → ¬( ∀ n : ℕ, a n < a (n + 1) )

-- Contrapositive of Proposition P
def Contra_Prop_P (a : ℕ → ℝ) :=
  ¬( ∀ n : ℕ, a n < a (n + 1) ) → ¬(a 1 < a 2 ∧ a 2 < a 3)

-- Main theorem to be proved
theorem num_correct_props_geometric_sequence (a : ℕ → ℝ) :
  is_geometric_sequence a → 
  Prop_P a ∧ Conv_Prop_P a ∧ Inv_Prop_P a ∧ Contra_Prop_P a := by
  sorry

end num_correct_props_geometric_sequence_l210_210360


namespace total_amount_of_money_l210_210311

theorem total_amount_of_money (P1 : ℝ) (interest_total : ℝ)
  (hP1 : P1 = 299.99999999999994) (hInterest : interest_total = 144) :
  ∃ T : ℝ, T = 3000 :=
by
  sorry

end total_amount_of_money_l210_210311


namespace coeff_sum_eq_minus_243_l210_210375

theorem coeff_sum_eq_minus_243 (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x y : ℝ, (x - 2 * y) ^ 5 = a * (x + 2 * y) ^ 5 + a₁ * (x + 2 * y)^4 * y + a₂ * (x + 2 * y)^3 * y^2 
             + a₃ * (x + 2 * y)^2 * y^3 + a₄ * (x + 2 * y) * y^4 + a₅ * y^5) →
  a + a₁ + a₂ + a₃ + a₄ + a₅ = -243 :=
by
  intros h
  sorry

end coeff_sum_eq_minus_243_l210_210375


namespace alberto_spent_2457_l210_210322

-- Define the expenses by Samara on each item
def oil_expense : ℕ := 25
def tires_expense : ℕ := 467
def detailing_expense : ℕ := 79

-- Define the additional amount Alberto spent more than Samara
def additional_amount : ℕ := 1886

-- Total amount spent by Samara
def samara_total_expense : ℕ := oil_expense + tires_expense + detailing_expense

-- The amount spent by Alberto
def alberto_expense := samara_total_expense + additional_amount

-- Theorem stating the amount spent by Alberto
theorem alberto_spent_2457 :
  alberto_expense = 2457 :=
by {
  -- Include the actual proof here if necessary
  sorry
}

end alberto_spent_2457_l210_210322


namespace curve_intersects_every_plane_l210_210542

theorem curve_intersects_every_plane (A B C D : ℝ) (h : A ≠ 0 ∨ B ≠ 0 ∨ C ≠ 0) :
  ∃ t : ℝ, A * t + B * t^3 + C * t^5 + D = 0 :=
by
  sorry

end curve_intersects_every_plane_l210_210542


namespace number_of_donut_selections_l210_210412

-- Definitions for the problem
def g : ℕ := sorry
def c : ℕ := sorry
def p : ℕ := sorry

-- Condition: Pat wants to buy four donuts from three types
def equation : Prop := g + c + p = 4

-- Question: Prove the number of different selections possible
theorem number_of_donut_selections : (∃ n, n = 15) := 
by 
  -- Use combinatorial method to establish this
  sorry

end number_of_donut_selections_l210_210412


namespace central_angle_of_sector_l210_210610

-- Given conditions as hypotheses
variable (r θ : ℝ)
variable (h₁ : (1/2) * θ * r^2 = 1)
variable (h₂ : 2 * r + θ * r = 4)

-- The goal statement to be proved
theorem central_angle_of_sector :
  θ = 2 :=
by sorry

end central_angle_of_sector_l210_210610


namespace simplify_and_evaluate_l210_210866

theorem simplify_and_evaluate (x : ℝ) (h : x = 3) : ((x - 2) / (x - 1)) / ((x + 1) - (3 / (x - 1))) = 1 / 5 :=
by
  sorry

end simplify_and_evaluate_l210_210866


namespace carrots_planted_per_hour_l210_210488

theorem carrots_planted_per_hour (rows plants_per_row hours : ℕ) (h1 : rows = 400) (h2 : plants_per_row = 300) (h3 : hours = 20) :
  (rows * plants_per_row) / hours = 6000 := by
  sorry

end carrots_planted_per_hour_l210_210488


namespace negative_870_in_third_quadrant_l210_210901

noncomputable def angle_in_third_quadrant (theta : ℝ) : Prop :=
  180 < theta ∧ theta < 270

theorem negative_870_in_third_quadrant:
  angle_in_third_quadrant 210 :=
by
  sorry

end negative_870_in_third_quadrant_l210_210901


namespace Jacob_eats_more_calories_than_planned_l210_210700

theorem Jacob_eats_more_calories_than_planned 
  (planned_calories : ℕ) (actual_calories : ℕ)
  (h1 : planned_calories < 1800) 
  (h2 : actual_calories = 400 + 900 + 1100)
  : actual_calories - planned_calories = 600 := by
  sorry

end Jacob_eats_more_calories_than_planned_l210_210700


namespace negation_of_implication_iff_l210_210877

variable (a : ℝ)

theorem negation_of_implication_iff (p : a > 1 → a^2 > 1) :
  ¬(a > 1 → a^2 > 1) ↔ (a ≤ 1 → a^2 ≤ 1) :=
by sorry

end negation_of_implication_iff_l210_210877


namespace ratio_wheelbarrow_to_earnings_l210_210763

theorem ratio_wheelbarrow_to_earnings :
  let duck_price := 10
  let chicken_price := 8
  let chickens_sold := 5
  let ducks_sold := 2
  let resale_earn := 60
  let total_earnings := chickens_sold * chicken_price + ducks_sold * duck_price
  let wheelbarrow_cost := resale_earn / 2
  (wheelbarrow_cost / total_earnings = 1 / 2) :=
by
  sorry

end ratio_wheelbarrow_to_earnings_l210_210763


namespace percentage_decrease_is_correct_l210_210108

variable (P : ℝ)

-- Condition 1: After the first year, the price increased by 30%
def price_after_first_year : ℝ := 1.30 * P

-- Condition 2: At the end of the 2-year period, the price of the painting is 110.5% of the original price
def price_after_second_year : ℝ := 1.105 * P

-- Condition 3: Let D be the percentage decrease during the second year
def D : ℝ := 0.15

-- Goal: Prove that the percentage decrease during the second year is 15%
theorem percentage_decrease_is_correct : 
  1.30 * P - D * 1.30 * P = 1.105 * P → D = 0.15 :=
by
  sorry

end percentage_decrease_is_correct_l210_210108


namespace chameleons_to_blue_l210_210263

-- Define a function that simulates the biting between chameleons and their resulting color changes
def color_transition (color_biter : ℕ) (color_bitten : ℕ) : ℕ :=
  if color_bitten = 1 then color_biter + 1
  else if color_bitten = 2 then color_biter + 2
  else if color_bitten = 3 then color_biter + 3
  else if color_bitten = 4 then color_biter + 4
  else 5  -- Once it reaches color 5 (blue), it remains blue.

-- Define the main theorem statement that given 5 red chameleons, all can be turned to blue.
theorem chameleons_to_blue : ∀ (red_chameleons : ℕ), red_chameleons = 5 → 
  ∃ (sequence_of_bites : ℕ → (ℕ × ℕ)), (∀ (c : ℕ), c < 5 → color_transition c (sequence_of_bites c).fst = 5) :=
by sorry

end chameleons_to_blue_l210_210263


namespace polynomial_evaluation_l210_210421

theorem polynomial_evaluation (P : ℕ → ℝ) (n : ℕ) 
  (h_degree : ∀ k : ℕ, k ≤ n → P k = k / (k + 1)) 
  (h_poly : ∀ k : ℕ, ∃ a : ℝ, P k = a * k ^ n) : 
  P (n + 1) = (n + 1 + (-1) ^ (n + 1)) / (n + 2) :=
by 
  sorry

end polynomial_evaluation_l210_210421


namespace sufficient_but_not_necessary_condition_m_sufficient_but_not_necessary_l210_210355

noncomputable def y (x m : ℝ) : ℝ := x^2 + m / x
noncomputable def y_prime (x m : ℝ) : ℝ := 2 * x - m / x^2

theorem sufficient_but_not_necessary_condition (m : ℝ) :
  (∀ x ≥ 1, y_prime x m ≥ 0) ↔ m ≤ 2 :=
sorry  -- Proof skipped as instructed

-- Now, state that m < 1 is a sufficient but not necessary condition
theorem m_sufficient_but_not_necessary (m : ℝ) :
  m < 1 → (∀ x ≥ 1, y_prime x m ≥ 0) :=
sorry  -- Proof skipped as instructed

end sufficient_but_not_necessary_condition_m_sufficient_but_not_necessary_l210_210355


namespace sophia_read_more_pages_l210_210414

variable (total_pages : ℝ) (finished_fraction : ℝ)
variable (pages_read : ℝ) (pages_left : ℝ) (pages_more : ℝ)

theorem sophia_read_more_pages :
  total_pages = 269.99999999999994 ∧
  finished_fraction = 2/3 ∧
  pages_read = finished_fraction * total_pages ∧
  pages_left = total_pages - pages_read →
  pages_more = pages_read - pages_left →
  pages_more = 90 := 
by
  intro h
  sorry

end sophia_read_more_pages_l210_210414


namespace plane_tiled_squares_triangles_percentage_l210_210734

theorem plane_tiled_squares_triangles_percentage :
    (percent_triangle_area : ℚ) = 625 / 10000 := sorry

end plane_tiled_squares_triangles_percentage_l210_210734


namespace two_lines_perpendicular_to_same_plane_are_parallel_l210_210650

variables {Plane Line : Type} 
variables (perp : Line → Plane → Prop) (parallel : Line → Line → Prop)

theorem two_lines_perpendicular_to_same_plane_are_parallel
  (a b : Line) (α : Plane) (ha : perp a α) (hb : perp b α) : parallel a b :=
sorry

end two_lines_perpendicular_to_same_plane_are_parallel_l210_210650


namespace flower_shop_ratio_l210_210499

theorem flower_shop_ratio (V C T R : ℕ) 
(total_flowers : V + C + T + R > 0)
(tulips_ratio : T = V / 4)
(roses_tulips_equal : R = T)
(carnations_fraction : C = 2 / 3 * (V + T + R + C)) 
: V / C = 1 / 3 := 
by
  -- Proof omitted
  sorry

end flower_shop_ratio_l210_210499


namespace percentage_of_hexagon_area_is_closest_to_17_l210_210318

noncomputable def tiling_area_hexagon_percentage : Real :=
  let total_area := 2 * 3
  let square_area := 1 * 1 
  let squares_count := 5 -- Adjusted count from 8 to fit total area properly
  let square_total_area := squares_count * square_area
  let hexagon_area := total_area - square_total_area
  let percentage := (hexagon_area / total_area) * 100
  percentage

theorem percentage_of_hexagon_area_is_closest_to_17 :
  abs (tiling_area_hexagon_percentage - 17) < 1 :=
sorry

end percentage_of_hexagon_area_is_closest_to_17_l210_210318


namespace expression_evaluates_to_2023_l210_210297

theorem expression_evaluates_to_2023 : 
  (2023^3 - 2 * 2023^2 * 2024 + 3 * 2023 * 2024^2 - 2024^3 + 1) / (2023 * 2024) = 2023 :=
by 
  sorry

end expression_evaluates_to_2023_l210_210297


namespace intersection_eq_l210_210202

-- Define the sets A and B
def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1, 2}

-- Prove the intersection of A and B is {0, 1}
theorem intersection_eq : A ∩ B = {0, 1} := by
  sorry

end intersection_eq_l210_210202


namespace terminating_decimal_l210_210344

-- Define the given fraction
def frac : ℚ := 21 / 160

-- Define the decimal representation
def dec : ℚ := 13125 / 100000

-- State the theorem to be proved
theorem terminating_decimal : frac = dec := by
  sorry

end terminating_decimal_l210_210344


namespace smallest_natural_with_50_perfect_squares_l210_210025

theorem smallest_natural_with_50_perfect_squares (a : ℕ) (h : a = 4486) :
  (∃ n, n^2 ≤ a ∧ (n+50)^2 < 3 * a ∧ (∀ b, n^2 ≤ b^2 ∧ b^2 < 3 * a → n ≤ b-1 ∧ b-1 ≤ n+49)) :=
by {
  sorry
}

end smallest_natural_with_50_perfect_squares_l210_210025


namespace geometric_sequence_sum_l210_210890

theorem geometric_sequence_sum
  (a r : ℝ)
  (h1 : a * (1 - r ^ 3000) / (1 - r) = 500)
  (h2 : a * (1 - r ^ 6000) / (1 - r) = 950) :
  a * (1 - r ^ 9000) / (1 - r) = 1355 :=
sorry

end geometric_sequence_sum_l210_210890


namespace order_of_abc_l210_210652

noncomputable def a : ℝ := (1 / 3) * Real.logb 2 (1 / 4)
noncomputable def b : ℝ := 1 - Real.logb 2 3
noncomputable def c : ℝ := Real.cos (5 * Real.pi / 6)

theorem order_of_abc : c < a ∧ a < b := by
  sorry

end order_of_abc_l210_210652


namespace sum_first_9000_terms_l210_210886

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_9000_terms (a r : ℝ) :
  geometric_sequence_sum a r 3000 = 500 →
  geometric_sequence_sum a r 6000 = 950 →
  geometric_sequence_sum a r 9000 = 1355 :=
by
  intros h1 h2
  sorry

end sum_first_9000_terms_l210_210886


namespace farm_owns_60_more_horses_than_cows_l210_210410

-- Let x be the number of cows initially
-- The number of horses initially is 4x
-- After selling 15 horses and buying 15 cows, the ratio of horses to cows becomes 7:3

theorem farm_owns_60_more_horses_than_cows (x : ℕ) (h_pos : 0 < x)
  (h_ratio : (4 * x - 15) / (x + 15) = 7 / 3) :
  (4 * x - 15) - (x + 15) = 60 :=
by
  sorry

end farm_owns_60_more_horses_than_cows_l210_210410


namespace river_depth_mid_may_l210_210689

variable (DepthMidMay DepthMidJune DepthMidJuly : ℕ)

theorem river_depth_mid_may :
  (DepthMidJune = DepthMidMay + 10) →
  (DepthMidJuly = 3 * DepthMidJune) →
  (DepthMidJuly = 45) →
  DepthMidMay = 5 :=
by
  intros h1 h2 h3
  sorry

end river_depth_mid_may_l210_210689


namespace karl_total_income_is_53_l210_210082

noncomputable def compute_income (tshirt_price pant_price skirt_price sold_tshirts sold_pants sold_skirts sold_refurbished_tshirts: ℕ) : ℝ :=
  let tshirt_income := 2 * tshirt_price
  let pant_income := sold_pants * pant_price
  let skirt_income := sold_skirts * skirt_price
  let refurbished_tshirt_price := (tshirt_price : ℝ) / 2
  let refurbished_tshirt_income := sold_refurbished_tshirts * refurbished_tshirt_price
  tshirt_income + pant_income + skirt_income + refurbished_tshirt_income

theorem karl_total_income_is_53 : compute_income 5 4 6 2 1 4 6 = 53 := by
  sorry

end karl_total_income_is_53_l210_210082


namespace probability_born_in_2008_l210_210257

/-- Define the relevant dates and days calculation -/
def num_days_between : ℕ :=
  29 + 31 + 30 + 31 + 31 + 29 + 31 + 30 + 31 + 30 + 31 + 31

def num_days_2008 : ℕ :=
  31 + 29 + 31 + 30 + 31 + 30 + 31 + 31

theorem probability_born_in_2008 :
  (num_days_2008 : ℚ) / num_days_between = 244 / 365 :=
by
  -- The proof is omitted for brevity
  sorry

end probability_born_in_2008_l210_210257


namespace greatest_int_with_conditions_l210_210569

noncomputable def is_valid_integer (n : ℤ) : Prop :=
  0 < n ∧ n < 200 ∧ Int.gcd n 18 = 6

theorem greatest_int_with_conditions : 
  ∃ n, is_valid_integer n ∧ ∀ m, is_valid_integer m → m ≤ 174 :=
begin
  sorry
end

end greatest_int_with_conditions_l210_210569


namespace ants_movement_impossible_l210_210283

theorem ants_movement_impossible (initial_positions final_positions : Fin 3 → ℝ × ℝ) :
  initial_positions 0 = (0,0) ∧ initial_positions 1 = (0,1) ∧ initial_positions 2 = (1,0) →
  final_positions 0 = (-1,0) ∧ final_positions 1 = (0,1) ∧ final_positions 2 = (1,0) →
  (∀ t : ℕ, ∃ m : Fin 3, 
    ∀ i : Fin 3, (i ≠ m → ∃ k l : ℝ, 
      (initial_positions i).2 - l * (initial_positions i).1 = 0 ∧ 
      ∀ (p : ℕ → ℝ × ℝ), p 0 = initial_positions i ∧ p t = final_positions i → 
      (p 0).1 + k * (p 0).2 = 0)) →
  false :=
by 
  sorry

end ants_movement_impossible_l210_210283


namespace combination_8_5_l210_210515

theorem combination_8_5 : (Nat.choose 8 5) = 56 := by
  sorry

end combination_8_5_l210_210515


namespace evaluate_polynomial_l210_210798

theorem evaluate_polynomial (x : ℝ) (h : x^2 - 3 * x - 9 = 0) (hx : 0 < x) : 
  x^4 - 3 * x^3 - 9 * x^2 + 27 * x - 8 = -8 := 
sorry

end evaluate_polynomial_l210_210798


namespace stable_equilibrium_condition_l210_210924

theorem stable_equilibrium_condition
  (a b : ℝ)
  (h_condition1 : a > b)
  (h_condition2 : ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1)
  : (b / a) < (1 / Real.sqrt 2) :=
sorry

end stable_equilibrium_condition_l210_210924


namespace isoperimetric_inequality_l210_210543

theorem isoperimetric_inequality (S : ℝ) (P : ℝ) : S ≤ P^2 / (4 * Real.pi) :=
sorry

end isoperimetric_inequality_l210_210543


namespace probability_born_in_2008_l210_210258

/-- Define the relevant dates and days calculation -/
def num_days_between : ℕ :=
  29 + 31 + 30 + 31 + 31 + 29 + 31 + 30 + 31 + 30 + 31 + 31

def num_days_2008 : ℕ :=
  31 + 29 + 31 + 30 + 31 + 30 + 31 + 31

theorem probability_born_in_2008 :
  (num_days_2008 : ℚ) / num_days_between = 244 / 365 :=
by
  -- The proof is omitted for brevity
  sorry

end probability_born_in_2008_l210_210258


namespace votes_ratio_l210_210065

theorem votes_ratio (joey_votes barry_votes marcy_votes : ℕ) 
  (h1 : joey_votes = 8) 
  (h2 : barry_votes = 2 * (joey_votes + 3)) 
  (h3 : marcy_votes = 66) : 
  (marcy_votes : ℚ) / barry_votes = 3 / 1 := 
by
  sorry

end votes_ratio_l210_210065


namespace combination_8_5_is_56_l210_210507

theorem combination_8_5_is_56 : nat.choose 8 5 = 56 :=
by sorry

end combination_8_5_is_56_l210_210507


namespace find_k_l210_210745

theorem find_k (x y k : ℝ) 
  (line1 : y = 3 * x + 2) 
  (line2 : y = -4 * x - 14) 
  (line3 : y = 2 * x + k) :
  k = -2 / 7 := 
by {
  sorry
}

end find_k_l210_210745


namespace solution_set_inequality_l210_210871

variable {f : ℝ → ℝ}

-- Declare the conditions as definitions and assumptions
def condition1 : Prop := ∀ x, f x + deriv f x < 2
def condition2 : Prop := f 1 = 3

theorem solution_set_inequality (h1 : condition1) (h2 : condition2) :
  {x : ℝ | exp x * f x > 2 * exp x + exp 1} = {x : ℝ | x < 1} :=
sorry

end solution_set_inequality_l210_210871


namespace a_in_range_l210_210820

noncomputable def kOM (t : ℝ) : ℝ := (Real.log t) / t
noncomputable def kON (a t : ℝ) : ℝ := (a + a * t - t^2) / t

theorem a_in_range (a : ℝ) : 
  (∀ t ∈ Set.Ici 1, 0 ≤ (1 - Real.log t + a) / t^2 + 1) →
  a ∈ Set.Ici (-2) := 
by
  sorry

end a_in_range_l210_210820


namespace inequality_solution_l210_210657

theorem inequality_solution 
  (a b c d e f : ℕ) 
  (h1 : a * d * f > b * c * f)
  (h2 : c * f * b > d * e * b) 
  (h3 : a * f - b * e = 1) 
  : d ≥ b + f := by
  -- Proof goes here
  sorry

end inequality_solution_l210_210657


namespace parabola_directrix_eq_l210_210728

theorem parabola_directrix_eq (x y : ℝ) : x^2 + 12 * y = 0 → y = 3 := 
by sorry

end parabola_directrix_eq_l210_210728


namespace find_a_range_l210_210819

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then |x| + 2 else x + 2 / x

theorem find_a_range (a : ℝ) :
  (∀ x : ℝ, f x ≥ |x / 2 + a|) ↔ (-2 ≤ a ∧ a ≤ 2) :=
by
  sorry

end find_a_range_l210_210819


namespace range_of_a_for_distinct_real_roots_l210_210982

theorem range_of_a_for_distinct_real_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (a - 1) * x^2 - 2 * x + 1 = 0 ∧ (a - 1) * y^2 - 2 * y + 1 = 0) ↔ (a < 2 ∧ a ≠ 1) :=
by
  sorry

end range_of_a_for_distinct_real_roots_l210_210982


namespace total_number_of_flowers_is_correct_l210_210280

-- Define the conditions
def number_of_pots : ℕ := 544
def flowers_per_pot : ℕ := 32
def total_flowers : ℕ := number_of_pots * flowers_per_pot

-- State the theorem to be proved
theorem total_number_of_flowers_is_correct :
  total_flowers = 17408 :=
by
  sorry

end total_number_of_flowers_is_correct_l210_210280


namespace smallest_n_for_inequality_l210_210664

theorem smallest_n_for_inequality :
  ∃ (n : ℕ), n = 4003 ∧ (∀ m : ℤ, (0 < m ∧ m < 2001) →
  ∃ k : ℤ, (m / 2001 : ℚ) < (k / n : ℚ) ∧ (k / n : ℚ) < ((m + 1) / 2002 : ℚ)) :=
sorry

end smallest_n_for_inequality_l210_210664


namespace all_positive_integers_occur_in_sequence_l210_210172

noncomputable def a : ℕ → ℕ
| 1     := 1
| 2     := 2
| (n+1) := (List.range (n+2)).find (λ m, m > 0 ∧ (∀ k < n, a k ≠ m) ∧ Nat.gcd m (a n) ≠ 1)

theorem all_positive_integers_occur_in_sequence : ∀ m : ℕ, ∃ n : ℕ, a (n + 1) = m :=
sorry

end all_positive_integers_occur_in_sequence_l210_210172


namespace initial_walnut_trees_l210_210427

theorem initial_walnut_trees (total_trees_after_planting : ℕ) (trees_planted_today : ℕ) (initial_trees : ℕ) : 
  (total_trees_after_planting = 55) → (trees_planted_today = 33) → (initial_trees + trees_planted_today = total_trees_after_planting) → (initial_trees = 22) :=
by
  sorry

end initial_walnut_trees_l210_210427


namespace part1_part2_l210_210658

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

def p (m : ℝ) : Prop :=
  let Δ := discriminant 1 m 1
  Δ > 0 ∧ -m / 2 < 0

def q (m : ℝ) : Prop :=
  let Δ := discriminant 4 (4 * (m - 2)) 1
  Δ < 0

theorem part1 (m : ℝ) (hp : p m) : m > 2 := 
sorry

theorem part2 (m : ℝ) (h : ¬(p m ∧ q m) ∧ (p m ∨ q m)) : (m ≥ 3) ∨ (1 < m ∧ m ≤ 2) := 
sorry

end part1_part2_l210_210658


namespace range_of_k_for_distinct_roots_l210_210220
-- Import necessary libraries

-- Define the quadratic equation and conditions
noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Define the property of having distinct real roots
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  discriminant a b c > 0

-- Define the specific problem instance and range condition
theorem range_of_k_for_distinct_roots (k : ℝ) :
  has_two_distinct_real_roots 1 2 k ↔ k < 1 :=
by
  sorry

end range_of_k_for_distinct_roots_l210_210220


namespace boundary_length_of_pattern_l210_210930

theorem boundary_length_of_pattern (area : ℝ) (num_points : ℕ) 
(points_per_side : ℕ) : 
area = 144 → num_points = 4 → points_per_side = 4 →
∃ length : ℝ, length = 92.5 :=
by
  intros
  sorry

end boundary_length_of_pattern_l210_210930


namespace comb_8_5_eq_56_l210_210512

theorem comb_8_5_eq_56 : nat.choose 8 5 = 56 :=
by {
  sorry
}

end comb_8_5_eq_56_l210_210512


namespace fill_cistern_time_l210_210287

theorem fill_cistern_time (A B C : ℕ) (hA : A = 10) (hB : B = 12) (hC : C = 50) :
    1 / (1 / A + 1 / B - 1 / C) = 300 / 49 :=
by
  sorry

end fill_cistern_time_l210_210287


namespace find_fourth_student_number_l210_210685

theorem find_fourth_student_number 
  (total_students : ℕ) 
  (sample_size : ℕ) 
  (student1_num : ℕ) 
  (student2_num : ℕ) 
  (student3_num : ℕ) 
  (student4_num : ℕ)
  ( H1 : total_students = 52 )
  ( H2 : sample_size = 4 )
  ( H3 : student1_num = 6 )
  ( H4 : student2_num = 32 )
  ( H5 : student3_num = 45 ) :
  student4_num = 19 :=
sorry

end find_fourth_student_number_l210_210685


namespace find_smallest_a_l210_210031

theorem find_smallest_a (n : ℕ) (a : ℕ) :
  (n^2 ≤ a) ∧ (a < (n+1)^2) →
  (∀ k : ℕ, (n^2 ≤ a) ∧ (a < k^2) ∧ (k^2 < 3 * a) → ((n+1)^2 ≤ k^2 ∧ k^2 ≤ (n+50)^2)) →
  ((n + 50)^2 < 3 * a) →
  a = 4486 :=
begin
  sorry
end

end find_smallest_a_l210_210031


namespace alcohol_water_ratio_l210_210755

theorem alcohol_water_ratio 
  (P_alcohol_pct : ℝ) (Q_alcohol_pct : ℝ) 
  (P_volume : ℝ) (Q_volume : ℝ) 
  (mixture_alcohol : ℝ) (mixture_water : ℝ)
  (h1 : P_alcohol_pct = 62.5)
  (h2 : Q_alcohol_pct = 87.5)
  (h3 : P_volume = 4)
  (h4 : Q_volume = 4)
  (ha : mixture_alcohol = (P_volume * (P_alcohol_pct / 100)) + (Q_volume * (Q_alcohol_pct / 100)))
  (hm : mixture_water = (P_volume + Q_volume) - mixture_alcohol) :
  mixture_alcohol / mixture_water = 3 :=
by
  sorry

end alcohol_water_ratio_l210_210755


namespace curved_surface_area_cone_l210_210654

theorem curved_surface_area_cone :
  let r := 8  -- base radius in cm
  let l := 19  -- lateral edge length in cm
  let π := Real.pi
  let CSA := π * r * l
  477.5 < CSA ∧ CSA < 478 := by
  sorry

end curved_surface_area_cone_l210_210654


namespace hockey_championship_max_k_volleyball_championship_max_k_l210_210425

theorem hockey_championship_max_k : ∃ (k : ℕ), 0 < k ∧ k ≤ 20 ∧ k = 18 :=
by
  -- proof goes here
  sorry

theorem volleyball_championship_max_k : ∃ (k : ℕ), 0 < k ∧ k ≤ 20 ∧ k = 15 :=
by
  -- proof goes here
  sorry

end hockey_championship_max_k_volleyball_championship_max_k_l210_210425


namespace quadratic_is_perfect_square_l210_210903

theorem quadratic_is_perfect_square (a b c : ℤ) 
  (h : ∀ x : ℤ, ∃ d e : ℤ, a*x^2 + b*x + c = (d*x + e)^2) : 
  ∃ d e : ℤ, a = d^2 ∧ b = 2*d*e ∧ c = e^2 :=
sorry

end quadratic_is_perfect_square_l210_210903


namespace arith_seq_general_term_sum_b_n_l210_210037

-- Definitions and conditions
structure ArithSeq (f : ℕ → ℕ) :=
  (d : ℕ)
  (d_ne_zero : d ≠ 0)
  (Sn : ℕ → ℕ)
  (a3_plus_S5 : f 3 + Sn 5 = 42)
  (geom_seq : (f 4)^2 = (f 1) * (f 13))

-- Given the definitions and conditions, prove the general term formula of the sequence
theorem arith_seq_general_term (a_n : ℕ → ℕ) (S_n : ℕ → ℕ) (d : ℕ) 
  (d_ne_zero : d ≠ 0) (a3_plus_S5 : a_n 3 + S_n 5 = 42)
  (geom_seq : (a_n 4)^2 = (a_n 1) * (a_n 13)) :
  ∀ n : ℕ, a_n n = 2 * n - 1 :=
sorry

-- Prove the sum of the first n terms of the sequence b_n
theorem sum_b_n (a_n : ℕ → ℕ) (b_n : ℕ → ℕ) (T_n : ℕ → ℕ) (n : ℕ):
  b_n n = 1 / (a_n (n - 1) * a_n n) →
  T_n n = (1 / 2) * (1 - (1 / (2 * n - 1))) →
  T_n n = (n - 1) / (2 * n - 1) :=
sorry

end arith_seq_general_term_sum_b_n_l210_210037


namespace factor_expr_l210_210179

theorem factor_expr (x : ℝ) : 81 - 27 * x^3 = 27 * (3 - x) * (9 + 3 * x + x^2) := 
sorry

end factor_expr_l210_210179


namespace tom_games_value_l210_210120

/--
  Tom bought his games for $200. The value tripled and he sold 40% of them.
  Prove that Tom sold the games for $240.
-/
theorem tom_games_value : (price: ℤ) -> (tripled_value: ℤ) -> (percentage_sold: ℤ) -> (sold_value: ℤ) ->
  price = 200 -> tripled_value = 3 * price -> percentage_sold = 40 -> sold_value = tripled_value * percentage_sold / 100 -> sold_value = 240 :=
by
  assume price tripled_value percentage_sold sold_value
  assume h1: price = 200
  assume h2: tripled_value = 3 * price
  assume h3: percentage_sold = 40
  assume h4: sold_value = tripled_value * percentage_sold / 100
  sorry

end tom_games_value_l210_210120


namespace fraction_uninterested_students_interested_l210_210936

theorem fraction_uninterested_students_interested 
  (students : Nat)
  (interest_ratio : ℚ)
  (express_interest_ratio_if_interested : ℚ)
  (express_disinterest_ratio_if_not_interested : ℚ) 
  (h1 : students > 0)
  (h2 : interest_ratio = 0.70)
  (h3 : express_interest_ratio_if_interested = 0.75)
  (h4 : express_disinterest_ratio_if_not_interested = 0.85) :
  let interested_students := students * interest_ratio
  let not_interested_students := students * (1 - interest_ratio)
  let express_interest_and_interested := interested_students * express_interest_ratio_if_interested
  let not_express_interest_and_interested := interested_students * (1 - express_interest_ratio_if_interested)
  let express_disinterest_and_not_interested := not_interested_students * express_disinterest_ratio_if_not_interested
  let express_interest_and_not_interested := not_interested_students * (1 - express_disinterest_ratio_if_not_interested)
  let not_express_interest_total := not_express_interest_and_interested + express_disinterest_and_not_interested
  let fraction := not_express_interest_and_interested / not_express_interest_total
  fraction = 0.407 := 
by
  sorry

end fraction_uninterested_students_interested_l210_210936


namespace solve_trig_eqn_solution_set_l210_210460

theorem solve_trig_eqn_solution_set :
  {x : ℝ | ∃ k : ℤ, x = 3 * k * Real.pi + Real.pi / 4 ∨ x = 3 * k * Real.pi + 5 * Real.pi / 4} =
  {x : ℝ | 2 * Real.sin ((2 / 3) * x) = 1} :=
by
  sorry

end solve_trig_eqn_solution_set_l210_210460


namespace disk_max_areas_l210_210146

-- Conditions Definition
def disk_divided (n : ℕ) : ℕ :=
  let radii := 3 * n
  let secant_lines := 2
  let total_areas := 9 * n
  total_areas

theorem disk_max_areas (n : ℕ) : disk_divided n = 9 * n :=
by
  sorry

end disk_max_areas_l210_210146


namespace integral_sqrt_1_minus_x_sq_plus_2x_l210_210788

theorem integral_sqrt_1_minus_x_sq_plus_2x :
  ∫ x in (0 : Real)..1, (Real.sqrt (1 - x^2) + 2 * x) = (Real.pi + 4) / 4 := by
  sorry

end integral_sqrt_1_minus_x_sq_plus_2x_l210_210788


namespace find_original_price_l210_210083

noncomputable def original_price_per_bottle (P : ℝ) : Prop :=
  let discounted_price := 0.80 * P
  let final_price_per_bottle := discounted_price - 2.00
  3 * final_price_per_bottle = 30

theorem find_original_price : ∃ P : ℝ, original_price_per_bottle P ∧ P = 15 :=
by
  sorry

end find_original_price_l210_210083


namespace range_of_k_l210_210683

theorem range_of_k {x y k : ℝ} :
  (∀ x y, 2 * x - y ≤ 1 ∧ x + y ≥ 2 ∧ y - x ≤ 2) →
  (z = k * x + 2 * y) →
  (∀ (x y : ℝ), z = k * x + 2 * y → (x = 1) ∧ (y = 1)) →
  -4 < k ∧ k < 2 :=
by
  sorry

end range_of_k_l210_210683


namespace area_of_gray_region_is_96π_l210_210125

noncomputable def area_gray_region (d_small : ℝ) (r_ratio : ℝ) : ℝ :=
  let r_small := d_small / 2
  let r_large := r_ratio * r_small
  let area_large := π * r_large^2
  let area_small := π * r_small^2
  area_large - area_small

theorem area_of_gray_region_is_96π :
  ∀ (d_small : ℝ) (r_ratio : ℝ), d_small = 4 → r_ratio = 5 → area_gray_region d_small r_ratio = 96 * π :=
by
  intros d_small r_ratio h1 h2
  have : d_small = 4 := h1
  have : r_ratio = 5 := h2
  sorry

end area_of_gray_region_is_96π_l210_210125


namespace intersection_of_A_and_B_l210_210042

open Set

def A : Set ℤ := {x | 1 ≤ x ∧ x ≤ 3}
def B : Set ℤ := {-1, 0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_of_A_and_B_l210_210042


namespace compare_neg_rational_decimal_l210_210332

theorem compare_neg_rational_decimal : 
  -3 / 4 > -0.8 := 
by 
  sorry

end compare_neg_rational_decimal_l210_210332


namespace cistern_fill_time_l210_210910

theorem cistern_fill_time (F : ℝ) (E : ℝ) (net_rate : ℝ) (time : ℝ)
  (h_F : F = 1 / 4)
  (h_E : E = 1 / 8)
  (h_net : net_rate = F - E)
  (h_time : time = 1 / net_rate) :
  time = 8 := 
sorry

end cistern_fill_time_l210_210910


namespace betty_red_beads_l210_210166

theorem betty_red_beads (r b : ℕ) (h_ratio : r / b = 3 / 2) (h_blue_beads : b = 20) : r = 30 :=
by
  sorry

end betty_red_beads_l210_210166


namespace rancher_unique_solution_l210_210747

-- Defining the main problem statement
theorem rancher_unique_solution : ∃! (b h : ℕ), 30 * b + 32 * h = 1200 ∧ b > h := by
  sorry

end rancher_unique_solution_l210_210747


namespace solve_ZAMENA_l210_210783

noncomputable def ZAMENA : ℕ := 541234

theorem solve_ZAMENA :
  ∃ (A M E H : ℕ), 
    1 ≤ A ∧ A ≤ 5 ∧
    1 ≤ M ∧ M ≤ 5 ∧
    1 ≤ E ∧ E ≤ 5 ∧
    1 ≤ H ∧ H ≤ 5 ∧
    3 > A ∧ 
    A > M ∧ 
    M < E ∧ 
    E < H ∧ 
    H < A ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ H ∧
    M ≠ E ∧ M ≠ H ∧
    E ≠ H ∧
    ZAMENA = 541234 :=
by {
  have h1 : ∃ (A M E H : ℕ), 
    3 > A ∧ 
    A > M ∧ 
    M < E ∧ 
    E < H ∧ 
    H < A ∧
    A ≠ M ∧ A ≠ E ∧ A ≠ H ∧
    M ≠ E ∧ M ≠ H ∧
    E ≠ H, sorry,
  exact ⟨2, 1, 2, 3, h1⟩,
  repeat { sorry }
}

end solve_ZAMENA_l210_210783


namespace count_injective_functions_with_unique_descent_l210_210918

open Nat

theorem count_injective_functions_with_unique_descent 
  (n m : ℕ) (h₁ : 2 ≤ n) (h₂ : n ≤ m) :
  (∃ (f : Fin n → Fin m), (Injective f) ∧ (∃! i : Fin (n - 1), f i > f (i + 1)))
  = Nat.choose m n * (2^n - (n+1)) :=
sorry

end count_injective_functions_with_unique_descent_l210_210918


namespace triangular_weight_l210_210555

noncomputable def rectangular_weight := 90
variables {C T : ℕ}

-- Conditions
axiom cond1 : C + T = 3 * C
axiom cond2 : 4 * C + T = T + C + rectangular_weight

-- Question: How much does the triangular weight weigh?
theorem triangular_weight : T = 60 :=
sorry

end triangular_weight_l210_210555


namespace complement_intersection_l210_210857

open Set

variable (A B U : Set ℕ) 

theorem complement_intersection (hA : A = {1, 2, 3}) (hB : B = {2, 3, 4, 5}) (hU : U = A ∪ B) :
  (U \ A) ∩ B = {4, 5} :=
by sorry

end complement_intersection_l210_210857


namespace circle_area_difference_l210_210754

noncomputable def difference_of_circle_areas (C1 C2 : ℝ) : ℝ :=
  let π := Real.pi
  let r1 := C1 / (2 * π)
  let r2 := C2 / (2 * π)
  let A1 := π * r1 ^ 2
  let A2 := π * r2 ^ 2
  A2 - A1

theorem circle_area_difference :
  difference_of_circle_areas 396 704 = 26948.4 :=
by
  sorry

end circle_area_difference_l210_210754


namespace problem1_problem2_l210_210809

def f (x a : ℝ) := |x - 1| + |x - a|

/-
  Problem 1:
  Prove that if a = 3, the solution set to the inequality f(x) ≥ 4 is 
  {x | x ≤ 0 ∨ x ≥ 4}.
-/
theorem problem1 (f : ℝ → ℝ → ℝ) (a : ℝ) (h : a = 3) : 
  {x : ℝ | f x a ≥ 4} = {x : ℝ | x ≤ 0 ∨ x ≥ 4} := 
sorry

/-
  Problem 2:
  Prove that for any x₁ ∈ ℝ, if f(x₁) ≥ 2 holds true, the range of values for
  a is {a | a ≥ 3 ∨ a ≤ -1}.
-/
theorem problem2 (f : ℝ → ℝ → ℝ) (x₁ : ℝ) :
  (∀ x₁ : ℝ, f x₁ a ≥ 2) ↔ (a ≥ 3 ∨ a ≤ -1) :=
sorry

end problem1_problem2_l210_210809


namespace find_S9_l210_210625

-- Setting up basic definitions for arithmetic sequence and the sum of its terms
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + (n - 1) * d
def sum_arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := n * (a + arithmetic_seq a d n) / 2

-- Given conditions
variables (a d : ℤ)
axiom h : 2 * arithmetic_seq a d 3 = 3 + a

-- Theorem to prove
theorem find_S9 : sum_arithmetic_seq a d 9 = 27 :=
by {
  sorry
}

end find_S9_l210_210625


namespace jerry_total_logs_l210_210395

def logs_from_trees (p m w : Nat) : Nat :=
  80 * p + 60 * m + 100 * w

theorem jerry_total_logs :
  logs_from_trees 8 3 4 = 1220 :=
by
  -- Proof here
  sorry

end jerry_total_logs_l210_210395


namespace zamena_solution_l210_210786

def digit_assignment (A M E H Z N : ℕ) : Prop :=
  1 ≤ A ∧ A ≤ 5 ∧ 1 ≤ M ∧ M ≤ 5 ∧ 1 ≤ E ∧ E ≤ 5 ∧ 1 ≤ H ∧ H ≤ 5 ∧ 1 ≤ Z ∧ Z ≤ 5 ∧ 1 ≤ N ∧ N ≤ 5 ∧
  A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ A ≠ Z ∧ A ≠ N ∧
  M ≠ E ∧ M ≠ H ∧ M ≠ Z ∧ M ≠ N ∧
  E ≠ H ∧ E ≠ Z ∧ E ≠ N ∧
  H ≠ Z ∧ H ≠ N ∧
  Z ≠ N ∧
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A 

theorem zamena_solution : 
  ∃ (A M E H Z N : ℕ), digit_assignment A M E H Z N ∧ (Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) := 
by
  sorry

end zamena_solution_l210_210786


namespace fraction_equivalence_l210_210956

theorem fraction_equivalence (a b : ℝ) (h : ((1 / a) + (1 / b)) / ((1 / a) - (1 / b)) = 2020) : (a + b) / (a - b) = 2020 :=
sorry

end fraction_equivalence_l210_210956


namespace percent_freshmen_psychology_majors_l210_210327

-- Define the total number of students in our context
def total_students : ℕ := 100

-- Define what 80% of total students being freshmen means
def freshmen (total : ℕ) : ℕ := 8 * total / 10

-- Define what 60% of freshmen being in the school of liberal arts means
def freshmen_in_liberal_arts (total : ℕ) : ℕ := 6 * freshmen total / 10

-- Define what 50% of freshmen in the school of liberal arts being psychology majors means
def freshmen_psychology_majors (total : ℕ) : ℕ := 5 * freshmen_in_liberal_arts total / 10

theorem percent_freshmen_psychology_majors :
  (freshmen_psychology_majors total_students : ℝ) / total_students * 100 = 24 :=
by
  sorry

end percent_freshmen_psychology_majors_l210_210327


namespace analysis_error_l210_210661

theorem analysis_error (x : ℝ) (h1 : x + 1 / x ≥ 2) : 
  x + 1 / x ≥ 2 :=
by {
  sorry
}

end analysis_error_l210_210661


namespace relationship_between_abc_l210_210168

theorem relationship_between_abc (u v a b c : ℝ)
  (h1 : u - v = a) 
  (h2 : u^2 - v^2 = b)
  (h3 : u^3 - v^3 = c) : 
  3 * b ^ 2 + a ^ 4 = 4 * a * c :=
sorry

end relationship_between_abc_l210_210168


namespace sufficient_drivers_and_correct_time_l210_210153

-- Conditions definitions
def one_way_minutes := 2 * 60 + 40  -- 2 hours 40 minutes in minutes
def round_trip_minutes := 2 * one_way_minutes  -- round trip in minutes
def rest_minutes := 60  -- mandatory rest period in minutes

-- Time checks for drivers
def driver_a_return := 12 * 60 + 40  -- Driver A returns at 12:40 PM in minutes
def driver_a_next_trip := driver_a_return + rest_minutes  -- Driver A's next trip time
def driver_d_departure := 13 * 60 + 5  -- Driver D departs at 13:05 in minutes

-- Verify sufficiency of four drivers and time correctness
theorem sufficient_drivers_and_correct_time : 
  4 = 4 ∧ (driver_a_next_trip + round_trip_minutes = 21 * 60 + 30) :=
by
  -- Explain the reasoning path that leads to this conclusion within this block
  sorry

end sufficient_drivers_and_correct_time_l210_210153


namespace fraction_sum_l210_210135

theorem fraction_sum : ((10 : ℚ) / 9 + (9 : ℚ) / 10 = 2.0 + (0.1 + 0.1 / 9)) :=
by sorry

end fraction_sum_l210_210135


namespace min_distance_between_points_on_circles_l210_210403

theorem min_distance_between_points_on_circles :
  let C₁ := {p : ℝ × ℝ | p.1^2 + p.2^2 + 4*p.1 + 2*p.2 + 1 = 0}
  let C₂ := {p : ℝ × ℝ | p.1^2 + p.2^2 - 4*p.1 - 4*p.2 + 6 = 0}
  ∃ P Q ∈ ℝ × ℝ, P ∈ C₁ ∧ Q ∈ C₂ ∧ dist P Q = 3 - Real.sqrt 2 :=
begin
  sorry
end

end min_distance_between_points_on_circles_l210_210403


namespace parallel_line_plane_l210_210378

-- Define vectors
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Dot product definition
def dotProduct (u v : Vector3D) : ℝ :=
  u.x * v.x + u.y * v.y + u.z * v.z

-- Options given
def optionA : Vector3D × Vector3D := (⟨1, 0, 0⟩, ⟨-2, 0, 0⟩)
def optionB : Vector3D × Vector3D := (⟨1, 3, 5⟩, ⟨1, 0, 1⟩)
def optionC : Vector3D × Vector3D := (⟨0, 2, 1⟩, ⟨-1, 0, -1⟩)
def optionD : Vector3D × Vector3D := (⟨1, -1, 3⟩, ⟨0, 3, 1⟩)

-- Main theorem
theorem parallel_line_plane :
  (dotProduct (optionA.fst) (optionA.snd) ≠ 0) ∧
  (dotProduct (optionB.fst) (optionB.snd) ≠ 0) ∧
  (dotProduct (optionC.fst) (optionC.snd) ≠ 0) ∧
  (dotProduct (optionD.fst) (optionD.snd) = 0) :=
by
  -- Using sorry to skip the proof
  sorry

end parallel_line_plane_l210_210378


namespace geometric_sequence_arithmetic_progression_l210_210361

open Nat

/--
Given a geometric sequence \( \{a_n\} \) where \( a_1 = 1 \) and the sequence terms
\( 4a_1 \), \( 2a_2 \), \( a_3 \) form an arithmetic progression, prove that
the common ratio \( q = 2 \) and the sum of the first four terms \( S_4 = 15 \).
-/
theorem geometric_sequence_arithmetic_progression (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h₀ : a 1 = 1)
    (h₁ : ∀ n, S n = (1 - q^n) / (1 - q)) 
    (h₂ : ∀ k n, a (k + n) = a k * q ^ n) 
    (h₃ : 4 * a 1 + a 3 = 4 * a 2) :
  q = 2 ∧ S 4 = 15 := 
sorry

end geometric_sequence_arithmetic_progression_l210_210361


namespace Maxim_born_in_2008_probability_l210_210247

def birth_interval_start := ⟨2007, 9, 2⟩ -- September 2, 2007
def birth_interval_end := ⟨2008, 8, 31⟩ -- August 31, 2008
def days_in_interval := 365
def days_in_2008 := 244

theorem Maxim_born_in_2008_probability :
  (days_in_2008 : ℝ) / (days_in_interval : ℝ) = 244 / 365 :=
sorry

end Maxim_born_in_2008_probability_l210_210247


namespace max_value_frac_l210_210358

theorem max_value_frac (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 5) :
  ∃ z, z = (x + y) / x ∧ z ≤ 2 / 3 := by
  sorry

end max_value_frac_l210_210358


namespace find_x_value_l210_210461

theorem find_x_value (x : ℝ) (h1 : 0 < x) (h2 : x < 180) :
  (Real.tan (150 - x * Real.pi / 180) = 
   (Real.sin (150 * Real.pi / 180) - Real.sin (x * Real.pi / 180)) /
   (Real.cos (150 * Real.pi / 180) - Real.cos (x * Real.pi / 180))) → 
  x = 110 := 
by 
  sorry

end find_x_value_l210_210461


namespace evaluate_expression_l210_210578

theorem evaluate_expression : 
    (1 / ( (-5 : ℤ) ^ 4) ^ 2 ) * (-5 : ℤ) ^ 9 = -5 :=
by sorry

end evaluate_expression_l210_210578


namespace choir_final_score_l210_210227

theorem choir_final_score (content_score sing_score spirit_score : ℕ)
  (content_weight sing_weight spirit_weight : ℝ)
  (h_content : content_weight = 0.30) 
  (h_sing : sing_weight = 0.50) 
  (h_spirit : spirit_weight = 0.20) 
  (h_content_score : content_score = 90)
  (h_sing_score : sing_score = 94)
  (h_spirit_score : spirit_score = 95) :
  content_weight * content_score + sing_weight * sing_score + spirit_weight * spirit_score = 93 := by
  sorry

end choir_final_score_l210_210227


namespace win_sector_area_l210_210598

theorem win_sector_area (r : ℝ) (p_win : ℝ) (area_total : ℝ) 
  (h1 : r = 8)
  (h2 : p_win = 3 / 8)
  (h3 : area_total = π * r^2) :
  ∃ area_win, area_win = 24 * π ∧ area_win = p_win * area_total :=
by
  sorry

end win_sector_area_l210_210598


namespace B_completes_work_in_18_days_l210_210445

variable {A B : ℝ}
variable (x : ℝ)

-- Conditions provided
def A_works_twice_as_fast_as_B (h1 : A = 2 * B) : Prop := true
def together_finish_work_in_6_days (h2 : B = 1 / x ∧ A = 2 / x ∧ 1 / 6 = (A + B)) : Prop := true

-- Theorem to prove: It takes B 18 days to complete the work independently
theorem B_completes_work_in_18_days (h1 : A = 2 * B) (h2 : B = 1 / x ∧ A = 2 / x ∧ 1 / 6 = (A + B)) : x = 18 := by
  sorry

end B_completes_work_in_18_days_l210_210445


namespace min_b_for_factorization_l210_210014

theorem min_b_for_factorization : 
  ∃ b : ℕ, (∀ p q : ℤ, (p + q = b) ∧ (p * q = 1764) → x^2 + b * x + 1764 = (x + p) * (x + q)) 
  ∧ b = 84 :=
sorry

end min_b_for_factorization_l210_210014


namespace right_triangle_x_value_l210_210521

theorem right_triangle_x_value (BM MA BC CA: ℝ) (M_is_altitude: BM + MA = BC + CA)
  (x: ℝ) (h: ℝ) (d: ℝ) (M: BM = x) (CB: BC = h) (CA: CA = d) :
  x = (2 * h * d - d ^ 2 / 4) / (2 * d + 2 * h) := by
  sorry

end right_triangle_x_value_l210_210521


namespace cost_of_pen_l210_210608

-- define the conditions
def notebook_cost (pen_cost : ℝ) : ℝ := 3 * pen_cost
def total_cost (notebook_cost : ℝ) : ℝ := 4 * notebook_cost

-- theorem stating the problem we need to prove
theorem cost_of_pen (pen_cost : ℝ) (h1 : total_cost (notebook_cost pen_cost) = 18) : pen_cost = 1.5 :=
by
  -- proof to be constructed
  sorry

end cost_of_pen_l210_210608


namespace distinct_integers_sum_441_l210_210471

-- Define the variables and conditions
variables (a b c d : ℕ)

-- State the conditions: a, b, c, d are distinct positive integers and their product is 441
def distinct_positive_integers (a b c d : ℕ) : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 
def positive_integers (a b c d : ℕ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

-- Define the main statement to be proved
theorem distinct_integers_sum_441 (a b c d : ℕ) (h_distinct : distinct_positive_integers a b c d) 
(h_positive : positive_integers a b c d) 
(h_product : a * b * c * d = 441) : a + b + c + d = 32 :=
by
  sorry

end distinct_integers_sum_441_l210_210471


namespace only_one_tuple_exists_l210_210640

theorem only_one_tuple_exists :
  ∃! (x : Fin 15 → ℝ),
    (1 - x 0)^2 + (x 0 - x 1)^2 + (x 1 - x 2)^2 + (x 2 - x 3)^2 + (x 3 - x 4)^2
    + (x 4 - x 5)^2 + (x 5 - x 6)^2 + (x 6 - x 7)^2 + (x 7 - x 8)^2 + (x 8 - x 9)^2
    + (x 9 - x 10)^2 + (x 10 - x 11)^2 + (x 11 - x 12)^2 + (x 12 - x 13)^2
    + (x 13 - x 14)^2 + (x 14)^2 = 1 / 16 := by
  sorry

end only_one_tuple_exists_l210_210640


namespace trigonometric_expression_identity_l210_210874

theorem trigonometric_expression_identity :
  (2 * Real.sin (100 * Real.pi / 180) - Real.cos (70 * Real.pi / 180)) / Real.cos (20 * Real.pi / 180)
  = 2 * Real.sqrt 3 - 1 :=
sorry

end trigonometric_expression_identity_l210_210874


namespace part_I_part_II_l210_210387

noncomputable def curve_M (theta : ℝ) : ℝ := 4 * Real.cos theta

noncomputable def line_l (t m alpha : ℝ) : ℝ × ℝ :=
  let x := m + t * Real.cos alpha
  let y := t * Real.sin alpha
  (x, y)

theorem part_I (varphi : ℝ) :
  let OB := curve_M (varphi + π / 4)
  let OC := curve_M (varphi - π / 4)
  let OA := curve_M varphi
  OB + OC = Real.sqrt 2 * OA := by
  sorry

theorem part_II (m alpha : ℝ) :
  let varphi := π / 12
  let B := (1, Real.sqrt 3)
  let C := (3, -Real.sqrt 3)
  exists t1 t2, line_l t1 m alpha = B ∧ line_l t2 m alpha = C :=
  have hα : alpha = 2 * π / 3 := by sorry
  have hm : m = 2 := by sorry
  sorry

end part_I_part_II_l210_210387


namespace fraction_of_total_students_l210_210382

variables (G B T : ℕ) (F : ℚ)

-- Given conditions
axiom ratio_boys_to_girls : (7 : ℚ) / 3 = B / G
axiom total_students : T = B + G
axiom fraction_equals_two_thirds_girls : (2 : ℚ) / 3 * G = F * T

-- Proof goal
theorem fraction_of_total_students : F = 1 / 5 :=
by
  sorry

end fraction_of_total_students_l210_210382


namespace sum_of_geometric_sequence_first_9000_terms_l210_210882

noncomputable def geomSum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

theorem sum_of_geometric_sequence_first_9000_terms (a r : ℝ) (h1 : geomSum a r 3000 = 500) (h2 : geomSum a r 6000 = 950) :
  geomSum a r 9000 = 1355 :=
by
  sorry

end sum_of_geometric_sequence_first_9000_terms_l210_210882


namespace multiples_of_7_with_units_digit_7_number_of_multiples_of_7_with_units_digit_7_l210_210827

theorem multiples_of_7_with_units_digit_7 (n : ℕ) : 
  (n < 150 ∧ ∃ k : ℕ, n = 7 * k ∧ n % 10 = 7) ↔ 
  n = 7 ∨ n = 77 ∨ n = 147 := 
by 
  sorry

theorem number_of_multiples_of_7_with_units_digit_7 : 
  ∃ m, m = 3 ∧ ∀ n : ℕ, (n < 150 ∧ ∃ k : ℕ, n = 7 * k ∧ n % 10 = 7) → n = 7 ∨ n = 77 ∨ n = 147 := 
by 
  use 3
  intros n
  intros hn
  split
  intro h
  cases h
  sorry

end multiples_of_7_with_units_digit_7_number_of_multiples_of_7_with_units_digit_7_l210_210827


namespace probability_exactly_one_solves_problem_l210_210144

-- Define the context in which A and B solve the problem with given probabilities.
variables (p1 p2 : ℝ)

-- Define the constraint that the probabilities are between 0 and 1
axiom prob_A_nonneg : 0 ≤ p1
axiom prob_A_le_one : p1 ≤ 1
axiom prob_B_nonneg : 0 ≤ p2
axiom prob_B_le_one : p2 ≤ 1

-- Define the context that A and B solve the problem independently.
axiom A_and_B_independent : true

-- The theorem statement to prove the desired probability of exactly one solving the problem.
theorem probability_exactly_one_solves_problem : (p1 * (1 - p2) + p2 * (1 - p1)) =  p1 * (1 - p2) + p2 * (1 - p1) :=
by
  sorry

end probability_exactly_one_solves_problem_l210_210144


namespace min_b_for_factorization_l210_210013

theorem min_b_for_factorization : 
  ∃ b : ℕ, (∀ p q : ℤ, (p + q = b) ∧ (p * q = 1764) → x^2 + b * x + 1764 = (x + p) * (x + q)) 
  ∧ b = 84 :=
sorry

end min_b_for_factorization_l210_210013


namespace complement_A_in_U_l210_210049

def U : Set ℕ := {1, 3, 5, 7, 9}
def A : Set ℕ := {1, 5, 7}

theorem complement_A_in_U : (U \ A) = {3, 9} := 
by sorry

end complement_A_in_U_l210_210049


namespace sum_of_interior_angles_6_find_n_from_300_degrees_l210_210812

-- Definitions and statement for part 1:
def sum_of_interior_angles (n : ℕ) : ℕ :=
  (n - 2) * 180

theorem sum_of_interior_angles_6 :
  sum_of_interior_angles 6 = 720 := 
by
  sorry

-- Definitions and statement for part 2:
def find_n_from_angles (angle : ℕ) : ℕ := 
  (angle / 180) + 2

theorem find_n_from_300_degrees :
  find_n_from_angles 900 = 7 :=
by
  sorry

end sum_of_interior_angles_6_find_n_from_300_degrees_l210_210812


namespace max_value_expression_l210_210244

theorem max_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (eq_condition : x^2 - 3 * x * y + 4 * y^2 - z = 0) : 
  ∃ (M : ℝ), M = 1 ∧ (∀ x y z : ℝ, 0 < x → 0 < y → 0 < z → x^2 - 3 * x * y + 4 * y^2 - z = 0 → (2/x + 1/y - 2/z) ≤ M) := 
by
  sorry

end max_value_expression_l210_210244


namespace drivers_sufficient_l210_210154

theorem drivers_sufficient
  (round_trip_duration : ℕ := 320)
  (rest_duration : ℕ := 60)
  (return_time_A : ℕ := 12 * 60 + 40)
  (depart_time_D : ℕ := 13 * 60 + 5)
  (next_depart_time_A : ℕ := 13 * 60 + 40)
  : (4 : ℕ) ∧ (21 * 60 + 30 = 1290) := 
  sorry

end drivers_sufficient_l210_210154


namespace problem_statement_l210_210197

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : ∃ x1 x2 x3 : ℝ, (x1 < x2 ∧ x2 < x3) ∧ (x3 = b) ∧
    (|x1|^(1/2) + |x1 + a|^(1/2) = b) ∧
    (|x2|^(1/2) + |x2 + a|^(1/2) = b) ∧
    (|x3|^(1/2) + |x3 + a|^(1/2) = b ∧ x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3)) :
  a + b = 144 := sorry

end problem_statement_l210_210197


namespace seats_per_bus_l210_210111

-- Conditions
def total_students : ℕ := 180
def total_buses : ℕ := 3

-- Theorem Statement
theorem seats_per_bus : (total_students / total_buses) = 60 := 
by 
  sorry

end seats_per_bus_l210_210111


namespace repeating_decimal_product_l210_210346

theorem repeating_decimal_product (x : ℚ) (h : x = 4 / 9) : x * 9 = 4 := 
by
  sorry

end repeating_decimal_product_l210_210346


namespace car_p_less_hours_l210_210114

theorem car_p_less_hours (distance : ℕ) (speed_r : ℕ) (speed_p : ℕ) (time_r : ℕ) (time_p : ℕ) (h1 : distance = 600) (h2 : speed_r = 50) (h3 : speed_p = speed_r + 10) (h4 : time_r = distance / speed_r) (h5 : time_p = distance / speed_p) : time_r - time_p = 2 := 
by
  sorry

end car_p_less_hours_l210_210114


namespace carina_coffee_l210_210002

def total_coffee (t f : ℕ) : ℕ := 10 * t + 5 * f

theorem carina_coffee (t : ℕ) (h1 : t = 3) (f : ℕ) (h2 : f = t + 2) : total_coffee t f = 55 := by
  sorry

end carina_coffee_l210_210002


namespace base_b_square_l210_210552

theorem base_b_square (b : ℕ) (h : b > 2) : ∃ k : ℕ, 121 = k ^ 2 :=
by
  sorry

end base_b_square_l210_210552


namespace sum_non_solution_values_l210_210238

theorem sum_non_solution_values (A B C : ℝ) (h : ∀ x : ℝ, (x+B) * (A*x+36) / ((x+C) * (x+9)) = 4) :
  ∃ M : ℝ, M = - (B + 9) := 
sorry

end sum_non_solution_values_l210_210238


namespace last_two_digits_of_1976_pow_100_l210_210419

theorem last_two_digits_of_1976_pow_100 :
  (1976 ^ 100) % 100 = 76 :=
by
  sorry

end last_two_digits_of_1976_pow_100_l210_210419


namespace slices_eaten_l210_210269

theorem slices_eaten (total_slices : Nat) (slices_left : Nat) (expected_slices_eaten : Nat) :
  total_slices = 32 →
  slices_left = 7 →
  expected_slices_eaten = 25 →
  total_slices - slices_left = expected_slices_eaten :=
by
  intros
  sorry

end slices_eaten_l210_210269


namespace sufficient_but_not_necessary_not_necessary_l210_210401

variable (x y : ℝ)

theorem sufficient_but_not_necessary (h1: x ≥ 2) (h2: y ≥ 2): x^2 + y^2 ≥ 4 :=
by
  sorry

theorem not_necessary (hx4 : x^2 + y^2 ≥ 4) : ¬ (x ≥ 2 ∧ y ≥ 2) → ∃ x y, (x^2 + y^2 ≥ 4) ∧ (¬ (x ≥ 2) ∨ ¬ (y ≥ 2)) :=
by
  sorry

end sufficient_but_not_necessary_not_necessary_l210_210401


namespace solve_equation1_solve_equation2_l210_210545

noncomputable def solutions_equation1 : Set ℝ := { x | x^2 - 2 * x - 8 = 0 }
noncomputable def solutions_equation2 : Set ℝ := { x | x^2 - 2 * x - 5 = 0 }

theorem solve_equation1 :
  solutions_equation1 = {4, -2} := 
by
  sorry

theorem solve_equation2 :
  solutions_equation2 = {1 + Real.sqrt 6, 1 - Real.sqrt 6} :=
by
  sorry

end solve_equation1_solve_equation2_l210_210545


namespace find_number_l210_210954

theorem find_number (x k : ℕ) (h₁ : x / k = 4) (h₂ : k = 6) : x = 24 := by
  sorry

end find_number_l210_210954


namespace probability_x_lt_2y_is_2_over_5_l210_210443

noncomputable def rectangle_area : ℝ :=
  5 * 2

noncomputable def triangle_area : ℝ :=
  1 / 2 * 4 * 2

noncomputable def probability_x_lt_2y : ℝ :=
  triangle_area / rectangle_area

theorem probability_x_lt_2y_is_2_over_5 :
  probability_x_lt_2y = 2 / 5 := by
  sorry

end probability_x_lt_2y_is_2_over_5_l210_210443


namespace value_of_expression_l210_210133

theorem value_of_expression (x : ℤ) (h : x = -2) : (3 * x + 4)^2 = 4 :=
by
  rw [h]
  norm_num
  sorry

end value_of_expression_l210_210133


namespace sum_of_geometric_sequence_first_9000_terms_l210_210881

noncomputable def geomSum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

theorem sum_of_geometric_sequence_first_9000_terms (a r : ℝ) (h1 : geomSum a r 3000 = 500) (h2 : geomSum a r 6000 = 950) :
  geomSum a r 9000 = 1355 :=
by
  sorry

end sum_of_geometric_sequence_first_9000_terms_l210_210881


namespace smallest_a_with_50_squares_l210_210019


theorem smallest_a_with_50_squares : ∃ (a : ℕ), (∀ b : ℕ, (b < a)) → 
(a > 0) ∧ (a * 3 > a) ∧ 
let count_squares := λ (a : ℕ), set.to_finset {n : ℕ | n^2 ∈ (set.Ioo (a : ℝ) (3 * a : ℝ))}.card
in count_squares a = 50 ∧ a = 4486 := by {
  sorry -- proof omitted
}

end smallest_a_with_50_squares_l210_210019


namespace total_area_of_triangles_l210_210991

theorem total_area_of_triangles :
    let AB := 12
    let DE := 8 * Real.sqrt 2
    let area_ABC := (1 / 2) * AB * AB
    let area_DEF := (1 / 2) * DE * DE * 2
    area_ABC + area_DEF = 136 := by
  sorry

end total_area_of_triangles_l210_210991


namespace yogurt_combinations_l210_210773

theorem yogurt_combinations (flavors toppings : ℕ) (hflavors : flavors = 5) (htoppings : toppings = 8) :
  (flavors * Nat.choose toppings 3 = 280) :=
by
  rw [hflavors, htoppings]
  sorry

end yogurt_combinations_l210_210773


namespace years_since_mothers_death_l210_210526

noncomputable def jessica_age_at_death (x : ℕ) : ℕ := 40 - x
noncomputable def mother_age_at_death (x : ℕ) : ℕ := 2 * jessica_age_at_death x

theorem years_since_mothers_death (x : ℕ) : mother_age_at_death x + x = 70 ↔ x = 10 :=
by
  sorry

end years_since_mothers_death_l210_210526


namespace common_difference_of_arithmetic_sequence_l210_210038

noncomputable def a_n (a1 d n : ℕ) : ℕ := a1 + (n - 1) * d

noncomputable def S_n (a1 d n : ℕ) : ℕ := n * (2 * a1 + (n - 1) * d) / 2

theorem common_difference_of_arithmetic_sequence (a1 d : ℕ) (h1 : a_n a1 d 3 = 8) (h2 : S_n a1 d 6 = 54) : d = 2 :=
  sorry

end common_difference_of_arithmetic_sequence_l210_210038


namespace value_of_place_ratio_l210_210999

theorem value_of_place_ratio :
  let d8_pos := 10000
  let d6_pos := 0.1
  d8_pos = 100000 * d6_pos :=
by
  let d8_pos := 10000
  let d6_pos := 0.1
  sorry

end value_of_place_ratio_l210_210999


namespace probability_at_least_one_trip_l210_210951

theorem probability_at_least_one_trip (p_A_trip : ℚ) (p_B_trip : ℚ)
  (h1 : p_A_trip = 1/4) (h2 : p_B_trip = 1/5) :
  (1 - ((1 - p_A_trip) * (1 - p_B_trip))) = 2/5 :=
by
  sorry

end probability_at_least_one_trip_l210_210951


namespace sweatshirt_cost_l210_210483

/--
Hannah bought 3 sweatshirts and 2 T-shirts.
Each T-shirt cost $10.
Hannah spent $65 in total.
Prove that the cost of each sweatshirt is $15.
-/
theorem sweatshirt_cost (S : ℝ) (h1 : 3 * S + 2 * 10 = 65) : S = 15 :=
by
  sorry

end sweatshirt_cost_l210_210483


namespace log_relation_l210_210052

theorem log_relation (a b : ℝ) 
  (h₁ : a = Real.log 1024 / Real.log 16) 
  (h₂ : b = Real.log 32 / Real.log 2) : 
  a = 1 / 2 * b := 
by 
  sorry

end log_relation_l210_210052


namespace tank_capacities_l210_210115

theorem tank_capacities (x y z : ℕ) 
  (h1 : x + y + z = 1620)
  (h2 : z = x + y / 5) 
  (h3 : z = y + x / 3) :
  x = 540 ∧ y = 450 ∧ z = 630 := 
by 
  sorry

end tank_capacities_l210_210115


namespace cone_prism_ratio_l210_210159

theorem cone_prism_ratio 
  (a b h_c h_p : ℝ) (hb_lt_a : b < a) : 
  (π * b * h_c) / (12 * a * h_p) = (1 / 3 * π * b^2 * h_c) / (4 * a * b * h_p) :=
by
  sorry

end cone_prism_ratio_l210_210159


namespace smallest_a_has_50_perfect_squares_l210_210021

def is_perfect_square (x : ℕ) : Prop :=
  ∃ n : ℕ, n * n = x

def count_perfect_squares_in_interval (low high : ℕ) : ℕ :=
  (List.range' low (high - low)).filter is_perfect_square |>.length

theorem smallest_a_has_50_perfect_squares : (a : ℕ) (h : a = 4486) →
  count_perfect_squares_in_interval (a + 1) (3 * a) = 50 :=
by
  intros a h
  rw h
  sorry

end smallest_a_has_50_perfect_squares_l210_210021


namespace truck_gas_consumption_l210_210622

theorem truck_gas_consumption :
  ∀ (initial_gasoline total_distance remaining_gasoline : ℝ),
    initial_gasoline = 12 →
    total_distance = (2 * 5 + 2 + 2 * 2 + 6) →
    remaining_gasoline = 2 →
    (initial_gasoline - remaining_gasoline) ≠ 0 →
    (total_distance / (initial_gasoline - remaining_gasoline)) = 2.2 :=
by
  intros initial_gasoline total_distance remaining_gasoline
  intro h_initial_gas h_total_distance h_remaining_gas h_non_zero
  sorry

end truck_gas_consumption_l210_210622


namespace line_parabola_intersection_l210_210644

theorem line_parabola_intersection (k : ℝ) :
  (∀ x y, y = k * x + 1 ∧ y^2 = 4 * x → y = 1 ∧ x = 1 / 4) ∨
  (∀ x y, y = k * x + 1 ∧ y^2 = 4 * x → (k^2 * x^2 + (2 * k - 4) * x + 1 = 0) ∧ (4 * k * k - 16 * k + 16 - 4 * k * k = 0) → k = 1) :=
sorry

end line_parabola_intersection_l210_210644


namespace trains_at_starting_positions_after_2016_minutes_l210_210551

-- Definitions corresponding to conditions
def round_trip_minutes (line: String) : Nat :=
  if line = "red" then 14
  else if line = "blue" then 16
  else if line = "green" then 18
  else 0

def is_multiple_of (n m : Nat) : Prop :=
  n % m = 0

-- Formalize the statement to be proven
theorem trains_at_starting_positions_after_2016_minutes :
  ∀ (line: String), 
  line = "red" ∨ line = "blue" ∨ line = "green" →
  is_multiple_of 2016 (round_trip_minutes line) :=
by
  intro line h
  cases h with
  | inl red =>
    sorry
  | inr hb =>
    cases hb with
    | inl blue =>
      sorry
    | inr green =>
      sorry

end trains_at_starting_positions_after_2016_minutes_l210_210551


namespace ab_plus_cd_value_l210_210041

theorem ab_plus_cd_value (a b c d : ℝ)
  (h1 : a + b + c = 5)
  (h2 : a + b + d = 1)
  (h3 : a + c + d = 12)
  (h4 : b + c + d = 7) :
  a * b + c * d = 176 / 9 := 
sorry

end ab_plus_cd_value_l210_210041


namespace tom_dimes_count_l210_210122

def originalDimes := 15
def dimesFromDad := 33
def dimesSpent := 11

theorem tom_dimes_count : originalDimes + dimesFromDad - dimesSpent = 37 := by
  sorry

end tom_dimes_count_l210_210122


namespace concurrency_of_ceva_l210_210554

open EuclideanGeometry

theorem concurrency_of_ceva
  (A B C P Q R A1 B1 C1 : Point)
  (ω : Circle)
  (hA1 : A1 ∈ ω)
  (hB1 : B1 ∈ ω)
  (hC1 : C1 ∈ ω)
  (h1 : ∠(B, A1, P) = ∠(C, A1, Q))
  (h2 : ∠(C, B1, P) = ∠(A, B1, R))
  (h3 : ∠(A, C1, R) = ∠(B, C1, Q)) :
  Concurrency (Line.mk A A1) (Line.mk B B1) (Line.mk C C1) :=
by
  sorry

end concurrency_of_ceva_l210_210554


namespace farmer_children_count_l210_210603

def apples_each_bag := 15
def eaten_each := 4
def sold_apples := 7
def apples_left := 60

theorem farmer_children_count : 
  ∃ (n : ℕ), 15 * n - (2 * 4 + 7) = 60 ∧ n = 5 :=
by
  use 5
  sorry

end farmer_children_count_l210_210603


namespace ratio_of_girls_more_than_boys_l210_210090

theorem ratio_of_girls_more_than_boys 
  (B : ℕ := 50) 
  (P : ℕ := 123) 
  (driver_assistant_teacher := 3) 
  (h : P = driver_assistant_teacher + B + (P - driver_assistant_teacher - B)) : 
  (P - driver_assistant_teacher - B) - B = 21 → 
  (P - driver_assistant_teacher - B) % B = 21 / 50 := 
sorry

end ratio_of_girls_more_than_boys_l210_210090


namespace jerry_total_logs_l210_210394

-- Given conditions
def pine_logs_per_tree := 80
def maple_logs_per_tree := 60
def walnut_logs_per_tree := 100

def pine_trees_cut := 8
def maple_trees_cut := 3
def walnut_trees_cut := 4

-- Formulate the problem
theorem jerry_total_logs : 
  pine_logs_per_tree * pine_trees_cut + 
  maple_logs_per_tree * maple_trees_cut + 
  walnut_logs_per_tree * walnut_trees_cut = 1220 := 
by 
  -- Placeholder for the actual proof
  sorry

end jerry_total_logs_l210_210394


namespace lines_perpendicular_l210_210217

-- Definition of lines and their relationships
def Line : Type := ℝ × ℝ × ℝ → Prop

variables (a b c : Line)

-- Condition 1: a is perpendicular to b
axiom perp (a b : Line) : Prop
-- Condition 2: b is parallel to c
axiom parallel (b c : Line) : Prop

-- Theorem to prove: 
theorem lines_perpendicular (h1 : perp a b) (h2 : parallel b c) : perp a c :=
sorry

end lines_perpendicular_l210_210217


namespace intersection_eq_set_l210_210203

-- Define set A based on the inequality
def A : Set ℝ := {x | x^2 + x - 2 < 0}

-- Define set B based on the inequality
def B : Set ℝ := {x | 0 ≤ Real.log (x + 1) / Real.log 2 ∧ Real.log (x + 1) / Real.log 2 < 2}

-- Translate the question to a lean theorem
theorem intersection_eq_set : (A ∩ B) = {x | 0 ≤ x ∧ x < 1} := 
sorry

end intersection_eq_set_l210_210203


namespace fruit_seller_price_l210_210764

theorem fruit_seller_price (CP SP : ℝ) (h1 : SP = 0.90 * CP) (h2 : 1.10 * CP = 13.444444444444445) : 
  SP = 11 :=
sorry

end fruit_seller_price_l210_210764


namespace geometric_sequence_sum_l210_210889

theorem geometric_sequence_sum
  (a r : ℝ)
  (h1 : a * (1 - r ^ 3000) / (1 - r) = 500)
  (h2 : a * (1 - r ^ 6000) / (1 - r) = 950) :
  a * (1 - r ^ 9000) / (1 - r) = 1355 :=
sorry

end geometric_sequence_sum_l210_210889


namespace comb_8_5_eq_56_l210_210514

theorem comb_8_5_eq_56 : nat.choose 8 5 = 56 :=
by {
  sorry
}

end comb_8_5_eq_56_l210_210514


namespace greatest_integer_gcd_6_l210_210570

theorem greatest_integer_gcd_6 (n : ℕ) (h₁ : n < 200) (h₂ : Nat.gcd n 18 = 6) : n = 192 :=
by
  sorry

end greatest_integer_gcd_6_l210_210570


namespace smallest_natural_with_50_perfect_squares_l210_210026

theorem smallest_natural_with_50_perfect_squares (a : ℕ) (h : a = 4486) :
  (∃ n, n^2 ≤ a ∧ (n+50)^2 < 3 * a ∧ (∀ b, n^2 ≤ b^2 ∧ b^2 < 3 * a → n ≤ b-1 ∧ b-1 ≤ n+49)) :=
by {
  sorry
}

end smallest_natural_with_50_perfect_squares_l210_210026


namespace distance_light_travels_250_years_l210_210815

def distance_light_travels_one_year : ℝ := 5.87 * 10^12
def years : ℝ := 250

theorem distance_light_travels_250_years :
  distance_light_travels_one_year * years = 1.4675 * 10^15 :=
by
  sorry

end distance_light_travels_250_years_l210_210815


namespace probability_factor_of_30_correct_l210_210293

noncomputable def probability_factor_of_30 : ℚ :=
  let n := 30
  let divisors_of_30 : Finset ℕ := {d ∈ (Finset.range (n + 1)) | n % d = 0}
  (divisors_of_30.card : ℚ) / n

theorem probability_factor_of_30_correct :
  probability_factor_of_30 = 4 / 15 :=
by
  sorry

end probability_factor_of_30_correct_l210_210293


namespace probability_white_black_l210_210281

variable (a b : ℕ)

theorem probability_white_black (h_a_pos : 0 < a) (h_b_pos : 0 < b) :
  (2 * a * b) / (a + b) / (a + b - 1) = (2 * (a * b) : ℝ) / ((a + b) * (a + b - 1): ℝ) :=
by sorry

end probability_white_black_l210_210281


namespace find_a15_l210_210851

namespace ArithmeticSequence

variable {a : ℕ → ℝ}

def arithmetic_sequence (an : ℕ → ℝ) := ∃ (a₁ d : ℝ), ∀ n, an n = a₁ + (n - 1) * d

theorem find_a15
  (h_seq : arithmetic_sequence a)
  (h_a3 : a 3 = 4)
  (h_a9 : a 9 = 10) :
  a 15 = 16 := 
sorry

end ArithmeticSequence

end find_a15_l210_210851


namespace calculate_expression_l210_210450

/-- Calculate the expression 2197 + 180 ÷ 60 × 3 - 197. -/
theorem calculate_expression : 2197 + (180 / 60) * 3 - 197 = 2009 := by
  sorry

end calculate_expression_l210_210450


namespace jerry_total_mean_l210_210079

def receivedFromAunt : ℕ := 9
def receivedFromUncle : ℕ := 9
def receivedFromBestFriends : List ℕ := [22, 23, 22, 22]
def receivedFromSister : ℕ := 7

def totalAmountReceived : ℕ :=
  receivedFromAunt + receivedFromUncle +
  receivedFromBestFriends.sum + receivedFromSister

def totalNumberOfGifts : ℕ :=
  1 + 1 + receivedFromBestFriends.length + 1

def meanAmountReceived : ℚ :=
  totalAmountReceived / totalNumberOfGifts

theorem jerry_total_mean :
  meanAmountReceived = 16.29 := by
sorry

end jerry_total_mean_l210_210079


namespace total_surface_area_of_resulting_solid_is_12_square_feet_l210_210931

noncomputable def height_of_D :=
  let h_A := 1 / 4
  let h_B := 1 / 5
  let h_C := 1 / 8
  2 - (h_A + h_B + h_C)

theorem total_surface_area_of_resulting_solid_is_12_square_feet :
  let h_A := 1 / 4
  let h_B := 1 / 5
  let h_C := 1 / 8
  let h_D := 2 - (h_A + h_B + h_C)
  let top_and_bottom_area := 4 * 2
  let side_area := 2 * (h_A + h_B + h_C + h_D)
  top_and_bottom_area + side_area = 12 := by
  sorry

end total_surface_area_of_resulting_solid_is_12_square_feet_l210_210931


namespace sum_of_roots_l210_210190

theorem sum_of_roots (x y : ℝ) (h : ∀ z, z^2 + 2023 * z - 2024 = 0 → z = x ∨ z = y) : x + y = -2023 := 
by
  sorry

end sum_of_roots_l210_210190


namespace find_m_l210_210967

theorem find_m (x m : ℝ) (h1 : 3 * x - 2 * m = 4) (h2 : x = m) : m = 4 :=
sorry

end find_m_l210_210967


namespace jamesOreos_count_l210_210074

noncomputable def jamesOreos (jordanOreos : ℕ) : ℕ := 4 * jordanOreos + 7

theorem jamesOreos_count (J : ℕ) (h1 : J + jamesOreos J = 52) : jamesOreos J = 43 :=
by
  sorry

end jamesOreos_count_l210_210074


namespace drivers_sufficient_l210_210155

theorem drivers_sufficient
  (round_trip_duration : ℕ := 320)
  (rest_duration : ℕ := 60)
  (return_time_A : ℕ := 12 * 60 + 40)
  (depart_time_D : ℕ := 13 * 60 + 5)
  (next_depart_time_A : ℕ := 13 * 60 + 40)
  : (4 : ℕ) ∧ (21 * 60 + 30 = 1290) := 
  sorry

end drivers_sufficient_l210_210155


namespace toms_friend_decks_l210_210286

theorem toms_friend_decks
  (cost_per_deck : ℕ)
  (tom_decks : ℕ)
  (total_spent : ℕ)
  (h1 : cost_per_deck = 8)
  (h2 : tom_decks = 3)
  (h3 : total_spent = 64) :
  (total_spent - tom_decks * cost_per_deck) / cost_per_deck = 5 := by
  sorry

end toms_friend_decks_l210_210286


namespace linda_savings_l210_210536

theorem linda_savings :
  let original_price_per_notebook := 3.75
  let discount_rate := 0.15
  let quantity := 12
  let total_price_without_discount := quantity * original_price_per_notebook
  let discount_amount_per_notebook := original_price_per_notebook * discount_rate
  let discounted_price_per_notebook := original_price_per_notebook - discount_amount_per_notebook
  let total_price_with_discount := quantity * discounted_price_per_notebook
  let total_savings := total_price_without_discount - total_price_with_discount
  total_savings = 6.75 :=
by {
  sorry
}

end linda_savings_l210_210536


namespace share_per_person_is_135k_l210_210232

noncomputable def calculate_share : ℝ :=
  (0.90 * (500000 * 1.20)) / 4

theorem share_per_person_is_135k : calculate_share = 135000 :=
by
  sorry

end share_per_person_is_135k_l210_210232


namespace largest_square_side_l210_210584

variable (length width : ℕ)
variable (h_length : length = 54)
variable (h_width : width = 20)
variable (num_squares : ℕ)
variable (h_num_squares : num_squares = 3)

theorem largest_square_side : (length : ℝ) / num_squares = 18 := by
  sorry

end largest_square_side_l210_210584


namespace one_in_M_l210_210477

def M : Set ℕ := {1, 2, 3}

theorem one_in_M : 1 ∈ M := sorry

end one_in_M_l210_210477


namespace allocation_schemes_l210_210176

theorem allocation_schemes (students venues : ℕ) (H_students : students = 4) (H_venues : venues = 3) (H_nonempty : ∀ v, v < 3 → ∃ s, s < 4 ∧ ∃ v' : fin 3, v' = v) :
  ∃ allocation_schemes, allocation_schemes = 36 :=
by {
  -- Omitted proof
  sorry,
}

end allocation_schemes_l210_210176


namespace height_of_room_is_twelve_l210_210270

-- Defining the dimensions of the room
def length : ℝ := 25
def width : ℝ := 15

-- Defining the dimensions of the door and windows
def door_area : ℝ := 6 * 3
def window_area : ℝ := 3 * (4 * 3)

-- Total cost of whitewashing
def total_cost : ℝ := 5436

-- Cost per square foot for whitewashing
def cost_per_sqft : ℝ := 6

-- The equation to solve for height
def height_equation (h : ℝ) : Prop :=
  cost_per_sqft * (2 * (length + width) * h - (door_area + window_area)) = total_cost

theorem height_of_room_is_twelve : ∃ h : ℝ, height_equation h ∧ h = 12 := by
  -- Proof would go here
  sorry

end height_of_room_is_twelve_l210_210270


namespace solve_quadratic_eq_l210_210352

theorem solve_quadratic_eq (a : ℝ) (x : ℝ) 
  (h : a ∈ ({-1, 1, a^2} : Set ℝ)) : 
  (x^2 - (1 - a) * x - 2 = 0) → (x = 2 ∨ x = -1) := by
  sorry

end solve_quadratic_eq_l210_210352


namespace allowance_amount_l210_210932

variable (initial_money spent_money final_money : ℕ)

theorem allowance_amount (initial_money : ℕ) (spent_money : ℕ) (final_money : ℕ) (h1: initial_money = 5) (h2: spent_money = 2) (h3: final_money = 8) : (final_money - (initial_money - spent_money)) = 5 := 
by 
  sorry

end allowance_amount_l210_210932


namespace arccos_one_eq_zero_l210_210333

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  -- the proof will go here
  sorry

end arccos_one_eq_zero_l210_210333


namespace Nishita_preferred_shares_l210_210861

variable (P : ℕ)

def preferred_share_dividend : ℕ := 5 * P
def common_share_dividend : ℕ := 3500 * 3  -- 3.5 * 1000

theorem Nishita_preferred_shares :
  preferred_share_dividend P + common_share_dividend = 16500 → P = 1200 :=
by
  unfold preferred_share_dividend common_share_dividend
  intro h
  sorry

end Nishita_preferred_shares_l210_210861


namespace sine_five_l210_210430

noncomputable def sine_value (x : ℝ) : ℝ :=
  Real.sin (5 * x)

theorem sine_five : sine_value 1 = -0.959 := 
  by
  sorry

end sine_five_l210_210430


namespace sarah_saves_5_dollars_l210_210148

noncomputable def price_per_pair : ℕ := 40

noncomputable def promotion_A_price (n : ℕ) : ℕ :=
if n % 2 = 0 then price_per_pair * n / 2 else price_per_pair

noncomputable def promotion_B_price (n : ℕ) : ℕ :=
if n % 2 = 0 then price_per_pair * n - (15 * (n / 2)) else price_per_pair

noncomputable def total_price_promotion_A : ℕ :=
price_per_pair + (price_per_pair / 2)

noncomputable def total_price_promotion_B : ℕ :=
price_per_pair + (price_per_pair - 15)

theorem sarah_saves_5_dollars : total_price_promotion_B - total_price_promotion_A = 5 :=
by
  rw [total_price_promotion_B, total_price_promotion_A]
  norm_num
  sorry

end sarah_saves_5_dollars_l210_210148


namespace no_psafe_numbers_l210_210955

def is_psafe (n p : ℕ) : Prop := 
  ¬ (n % p = 0 ∨ n % p = 1 ∨ n % p = 2 ∨ n % p = 3 ∨ n % p = p - 3 ∨ n % p = p - 2 ∨ n % p = p - 1)

theorem no_psafe_numbers (N : ℕ) (hN : N = 10000) :
  ∀ n, (n ≤ N ∧ is_psafe n 5 ∧ is_psafe n 7 ∧ is_psafe n 11) → false :=
by
  sorry

end no_psafe_numbers_l210_210955


namespace a_minus_two_sufficient_but_not_necessary_for_pure_imaginary_l210_210437

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

def complex_from_a (a : ℝ) : ℂ :=
  (a^2 - 4 : ℝ) + (a + 1 : ℝ) * Complex.I

theorem a_minus_two_sufficient_but_not_necessary_for_pure_imaginary :
  (is_pure_imaginary (complex_from_a (-2))) ∧ ¬ (∀ (a : ℝ), is_pure_imaginary (complex_from_a a) → a = -2) :=
by
  sorry

end a_minus_two_sufficient_but_not_necessary_for_pure_imaginary_l210_210437


namespace prove_ratio_l210_210704

noncomputable def box_dimensions : ℝ × ℝ × ℝ := (2, 3, 5)
noncomputable def d := (2 * 3 * 5 : ℝ)
noncomputable def a := ((4 * Real.pi) / 3 : ℝ)
noncomputable def b := (10 * Real.pi : ℝ)
noncomputable def c := (62 : ℝ)

theorem prove_ratio :
  (b * c) / (a * d) = (15.5 : ℝ) :=
by
  unfold a b c d
  sorry

end prove_ratio_l210_210704


namespace greatest_integer_gcd_6_l210_210573

theorem greatest_integer_gcd_6 (n : ℕ) (h₁ : n < 200) (h₂ : Nat.gcd n 18 = 6) : n = 192 :=
by
  sorry

end greatest_integer_gcd_6_l210_210573


namespace arithmetic_seq_sum_ratio_l210_210530

theorem arithmetic_seq_sum_ratio (a1 d : ℝ) (S : ℕ → ℝ) 
  (hSn : ∀ n, S n = n * a1 + d * (n * (n - 1) / 2))
  (h_ratio : S 3 / S 6 = 1 / 3) :
  S 9 / S 6 = 2 :=
by
  sorry

end arithmetic_seq_sum_ratio_l210_210530


namespace root_quadratic_expression_value_l210_210965

theorem root_quadratic_expression_value (m : ℝ) (h : m^2 - m - 3 = 0) : 2023 - m^2 + m = 2020 := 
by 
  sorry

end root_quadratic_expression_value_l210_210965


namespace lengths_of_diagonals_and_t_value_l210_210691

-- Define the coordinates of points A, B, and C
def A : ℝ × ℝ := ⟨-1, -2⟩
def B : ℝ × ℝ := ⟨2, 3⟩
def C : ℝ × ℝ := ⟨-2, -1⟩

-- Define vectors AB and AC
def AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
def AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
def OC : ℝ × ℝ := C

-- Define dot product for pairs of real numbers
def dot (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Define magnitudes of vectors AB + AC and AB - AC
def len_diag1 : ℝ := Real.sqrt ((AB.1 + AC.1)^2 + (AB.2 + AC.2)^2)
def len_diag2 : ℝ := Real.sqrt ((AB.1 - AC.1)^2 + (AB.2 - AC.2)^2)

-- Define expression for t
def t_eqn (t : ℝ) : ℝ × ℝ := (AB.1 + t * OC.1, AB.2 + t * OC.2)

-- Lean statement for the proof problem
theorem lengths_of_diagonals_and_t_value :
  len_diag1 = 2 * Real.sqrt(10) ∧ len_diag2 = 4 * Real.sqrt(2) ∧
  (∃ t : ℝ, dot (AB.1 - t * OC.1, AB.2 - t * OC.2) OC = 0 ∧ t = -11 / 5) :=
by
  sorry

end lengths_of_diagonals_and_t_value_l210_210691


namespace runner_distance_l210_210160

theorem runner_distance :
  ∃ x t d : ℕ,
    d = x * t ∧
    d = (x + 1) * (2 * t / 3) ∧
    d = (x - 1) * (t + 3) ∧
    d = 6 :=
by
  sorry

end runner_distance_l210_210160


namespace measure_of_RPS_l210_210068

-- Assume the elements of the problem
variables {Q R P S : Type}

-- Angles in degrees
def angle_PQS := 35
def angle_QPR := 80
def angle_PSQ := 40

-- Define the angles and the straight line condition
def QRS_straight_line : Prop := true  -- This definition is trivial for a straight line

-- Measure of angle QPS using sum of angles in triangle
noncomputable def angle_QPS : ℝ := 180 - angle_PQS - angle_PSQ

-- Measure of angle RPS derived from the previous steps
noncomputable def angle_RPS : ℝ := angle_QPS - angle_QPR

-- The statement of the problem in Lean
theorem measure_of_RPS : angle_RPS = 25 := by
  sorry

end measure_of_RPS_l210_210068


namespace triangle_equilateral_if_arithmetic_sequences_l210_210073

theorem triangle_equilateral_if_arithmetic_sequences
  (A B C : ℝ) (a b c : ℝ)
  (h_angles : A + B + C = 180)
  (h_angle_seq : ∃ (N : ℝ), A = B - N ∧ C = B + N)
  (h_sides : ∃ (n : ℝ), a = b - n ∧ c = b + n) :
  A = B ∧ B = C ∧ a = b ∧ b = c :=
sorry

end triangle_equilateral_if_arithmetic_sequences_l210_210073


namespace field_division_l210_210316

theorem field_division
  (total_area : ℕ)
  (part_area : ℕ)
  (diff : ℕ → ℕ)
  (X : ℕ)
  (h_total : total_area = 900)
  (h_part : part_area = 405)
  (h_diff : diff (total_area - part_area - part_area) = (1 / 5 : ℚ) * X)
  : X = 450 := 
sorry

end field_division_l210_210316


namespace ab_times_65_eq_48ab_l210_210051

theorem ab_times_65_eq_48ab (a b : ℕ) (h_ab : 0 ≤ a ∧ a < 10) (h_b : 0 ≤ b ∧ b < 10) :
  (10 * a + b) * 65 = 4800 + 10 * a + b ↔ 10 * a + b = 75 := by
sorry

end ab_times_65_eq_48ab_l210_210051


namespace river_depth_mid_may_l210_210687

-- Definitions corresponding to the conditions
def depth_mid_june (D : ℕ) : ℕ := D + 10
def depth_mid_july (D : ℕ) : ℕ := 3 * (depth_mid_june D)

-- The theorem statement
theorem river_depth_mid_may (D : ℕ) (h : depth_mid_july D = 45) : D = 5 :=
by
  sorry

end river_depth_mid_may_l210_210687


namespace convert_base5_412_to_base7_l210_210791

def base5_to_dec (n : Nat) : Nat :=
  let d2 := n % 10
  let n1 := n / 10
  let d1 := n1 % 10
  let n0 := n1 / 10
  let d0 := n0 % 10
  d0 * 25 + d1 * 5 + d2

def dec_to_base7 (n : Nat) : Nat :=
  let r2 := n % 7
  let n1 := n / 7
  let r1 := n1 % 7
  let n0 := n1 / 7
  let r0 := n0 % 7
  r0 * 100 + r1 * 10 + r2

theorem convert_base5_412_to_base7 : 
  dec_to_base7 (base5_to_dec 412) = 212 :=
by
  sorry

end convert_base5_412_to_base7_l210_210791


namespace total_amount_paid_l210_210673

-- Define the quantities and rates as constants
def quantity_grapes : ℕ := 8
def rate_grapes : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 55

-- Define the cost functions
def cost_grapes (q : ℕ) (r : ℕ) : ℕ := q * r
def cost_mangoes (q : ℕ) (r : ℕ) : ℕ := q * r

-- Define the total cost function
def total_cost (c1 : ℕ) (c2 : ℕ) : ℕ := c1 + c2

-- State the proof problem
theorem total_amount_paid :
  total_cost (cost_grapes quantity_grapes rate_grapes) (cost_mangoes quantity_mangoes rate_mangoes) = 1055 :=
by
  sorry

end total_amount_paid_l210_210673


namespace mean_and_variance_l210_210328

def scores_A : List ℝ := [8, 9, 14, 15, 15, 16, 21, 22]
def scores_B : List ℝ := [7, 8, 13, 15, 15, 17, 22, 23]

noncomputable def mean (l : List ℝ) : ℝ := (l.sum) / (l.length)
noncomputable def variance (l : List ℝ) : ℝ := mean (l.map (λ x => (x - (mean l)) ^ 2))

theorem mean_and_variance :
  (mean scores_A = mean scores_B) ∧ (variance scores_A < variance scores_B) :=
by
  sorry

end mean_and_variance_l210_210328


namespace fractionOf_Product_Of_Fractions_l210_210908

noncomputable def fractionOfProductOfFractions := 
  let a := (2 : ℚ) / 9 * (5 : ℚ) / 6 -- Define the product of the fractions
  let b := (3 : ℚ) / 4 -- Define another fraction
  a / b = 20 / 81 -- Statement to be proven

theorem fractionOf_Product_Of_Fractions: fractionOfProductOfFractions :=
by sorry

end fractionOf_Product_Of_Fractions_l210_210908


namespace multiples_of_7_with_units_digit_7_number_of_multiples_of_7_with_units_digit_7_l210_210826

theorem multiples_of_7_with_units_digit_7 (n : ℕ) : 
  (n < 150 ∧ ∃ k : ℕ, n = 7 * k ∧ n % 10 = 7) ↔ 
  n = 7 ∨ n = 77 ∨ n = 147 := 
by 
  sorry

theorem number_of_multiples_of_7_with_units_digit_7 : 
  ∃ m, m = 3 ∧ ∀ n : ℕ, (n < 150 ∧ ∃ k : ℕ, n = 7 * k ∧ n % 10 = 7) → n = 7 ∨ n = 77 ∨ n = 147 := 
by 
  use 3
  intros n
  intros hn
  split
  intro h
  cases h
  sorry

end multiples_of_7_with_units_digit_7_number_of_multiples_of_7_with_units_digit_7_l210_210826


namespace greatest_int_with_conditions_l210_210566

noncomputable def is_valid_integer (n : ℤ) : Prop :=
  0 < n ∧ n < 200 ∧ Int.gcd n 18 = 6

theorem greatest_int_with_conditions : 
  ∃ n, is_valid_integer n ∧ ∀ m, is_valid_integer m → m ≤ 174 :=
begin
  sorry
end

end greatest_int_with_conditions_l210_210566


namespace voltage_relationship_l210_210050

variables (x y z : ℝ) -- Coordinates representing positions on the lines
variables (I R U : ℝ) -- Representing current, resistance, and voltage respectively

-- Conditions translated into Lean
def I_def := I = 10^x
def R_def := R = 10^(-2 * y)
def U_def := U = 10^(-z)
def coord_relation := x + z = 2 * y

-- The final theorem to prove V = I * R under given conditions
theorem voltage_relationship : I = 10^x → R = 10^(-2 * y) → U = 10^(-z) → (x + z = 2 * y) → U = I * R :=
by 
  intros hI hR hU hXYZ
  sorry

end voltage_relationship_l210_210050


namespace sum_of_geometric_sequence_first_9000_terms_l210_210880

noncomputable def geomSum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

theorem sum_of_geometric_sequence_first_9000_terms (a r : ℝ) (h1 : geomSum a r 3000 = 500) (h2 : geomSum a r 6000 = 950) :
  geomSum a r 9000 = 1355 :=
by
  sorry

end sum_of_geometric_sequence_first_9000_terms_l210_210880


namespace josephine_total_milk_l210_210261

-- Define the number of containers and the amount of milk they hold
def cnt_1 : ℕ := 3
def qty_1 : ℚ := 2

def cnt_2 : ℕ := 2
def qty_2 : ℚ := 0.75

def cnt_3 : ℕ := 5
def qty_3 : ℚ := 0.5

-- Define the total amount of milk sold
def total_milk_sold : ℚ := cnt_1 * qty_1 + cnt_2 * qty_2 + cnt_3 * qty_3

theorem josephine_total_milk : total_milk_sold = 10 := by
  -- This is the proof placeholder
  sorry

end josephine_total_milk_l210_210261


namespace correct_flowchart_requirement_l210_210103

def flowchart_requirement (option : String) : Prop := 
  option = "From left to right, from top to bottom" ∨
  option = "From right to left, from top to bottom" ∨
  option = "From left to right, from bottom to top" ∨
  option = "From right to left, from bottom to top"

theorem correct_flowchart_requirement : 
  (∀ option, flowchart_requirement option → option = "From left to right, from top to bottom") :=
by
  sorry

end correct_flowchart_requirement_l210_210103


namespace heath_plants_per_hour_l210_210485

theorem heath_plants_per_hour (rows : ℕ) (plants_per_row : ℕ) (hours : ℕ) (total_plants : ℕ) :
  rows = 400 ∧ plants_per_row = 300 ∧ hours = 20 ∧ total_plants = rows * plants_per_row →
  total_plants / hours = 6000 :=
by
  sorry

end heath_plants_per_hour_l210_210485


namespace gain_percent_l210_210925

theorem gain_percent (CP SP : ℕ) (h1 : CP = 20) (h2 : SP = 25) : 
  (SP - CP) * 100 / CP = 25 := by
  sorry

end gain_percent_l210_210925


namespace small_branches_count_l210_210921

theorem small_branches_count (x : ℕ) (h : x^2 + x + 1 = 91) : x = 9 := 
  sorry

end small_branches_count_l210_210921


namespace find_certain_number_l210_210289

theorem find_certain_number 
  (x : ℝ) 
  (h : ( (x + 2 - 6) * 3 ) / 4 = 3) 
  : x = 8 :=
by
  sorry

end find_certain_number_l210_210289


namespace choose_five_from_eight_l210_210510

theorem choose_five_from_eight : Nat.choose 8 5 = 56 :=
by
  sorry 

end choose_five_from_eight_l210_210510


namespace virus_affected_computers_l210_210995

theorem virus_affected_computers (m n : ℕ) (h1 : 5 * m + 2 * n = 52) : m = 8 :=
by
  sorry

end virus_affected_computers_l210_210995


namespace work_done_in_one_day_l210_210145

theorem work_done_in_one_day (A_time B_time : ℕ) (hA : A_time = 4) (hB : B_time = A_time / 2) : 
  (1 / A_time + 1 / B_time) = (3 / 4) :=
by
  -- Here we are setting up the conditions as per our identified steps
  rw [hA, hB]
  -- The remaining steps to prove will be omitted as per instructions
  sorry

end work_done_in_one_day_l210_210145
