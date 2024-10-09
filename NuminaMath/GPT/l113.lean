import Mathlib

namespace chess_club_members_l113_11369

theorem chess_club_members {n : ℤ} (h10 : n % 10 = 6) (h11 : n % 11 = 6) (rng : 300 ≤ n ∧ n ≤ 400) : n = 336 :=
  sorry

end chess_club_members_l113_11369


namespace inscribed_circle_distance_l113_11300

-- description of the geometry problem
theorem inscribed_circle_distance (r : ℝ) (AB : ℝ):
  r = 4 →
  AB = 4 →
  ∃ d : ℝ, d = 6.4 :=
by
  intros hr hab
  -- skipping proof steps
  let a := 2*r
  let PQ := 2 * r * (Real.sqrt 3 / 2)
  use PQ
  sorry

end inscribed_circle_distance_l113_11300


namespace second_prime_is_23_l113_11329

-- Define the conditions
def is_prime (n : ℕ) : Prop := Nat.Prime n
def x := 69
def p : ℕ := 3
def q : ℕ := 23

-- State the theorem
theorem second_prime_is_23 (h1 : is_prime p) (h2 : 2 < p ∧ p < 6) (h3 : is_prime q) (h4 : x = p * q) : q = 23 := 
by 
  sorry

end second_prime_is_23_l113_11329


namespace curve_C2_equation_l113_11355

theorem curve_C2_equation (x y : ℝ) :
  (∀ x, y = 2 * Real.sin (2 * x + π / 3) → 
    y = 2 * Real.sin (4 * (( x - π / 6) / 2))) := 
  sorry

end curve_C2_equation_l113_11355


namespace pregnant_dogs_count_l113_11397

-- Definitions as conditions stated in the problem
def total_puppies (P : ℕ) : ℕ := 4 * P
def total_shots (P : ℕ) : ℕ := 2 * total_puppies P
def total_cost (P : ℕ) : ℕ := total_shots P * 5

-- Proof statement without proof
theorem pregnant_dogs_count : ∃ P : ℕ, total_cost P = 120 → P = 3 :=
by sorry

end pregnant_dogs_count_l113_11397


namespace monotonic_increasing_m_ge_neg4_l113_11386

def is_monotonic_increasing (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y : ℝ, x ≥ a → y > x → f y ≥ f x

def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * x - 2

theorem monotonic_increasing_m_ge_neg4 (m : ℝ) :
  is_monotonic_increasing (f m) 2 → m ≥ -4 :=
by
  sorry

end monotonic_increasing_m_ge_neg4_l113_11386


namespace total_surface_area_excluding_bases_l113_11356

def lower_base_radius : ℝ := 8
def upper_base_radius : ℝ := 5
def frustum_height : ℝ := 6
def cylinder_section_height : ℝ := 2
def cylinder_section_radius : ℝ := 5

theorem total_surface_area_excluding_bases :
  let l := Real.sqrt (frustum_height ^ 2 + (lower_base_radius - upper_base_radius) ^ 2)
  let lateral_surface_area_frustum := π * (lower_base_radius + upper_base_radius) * l
  let lateral_surface_area_cylinder := 2 * π * cylinder_section_radius * cylinder_section_height
  lateral_surface_area_frustum + lateral_surface_area_cylinder = 39 * π * Real.sqrt 5 + 20 * π :=
by
  sorry

end total_surface_area_excluding_bases_l113_11356


namespace total_pieces_of_junk_mail_l113_11354

def pieces_per_block : ℕ := 48
def num_blocks : ℕ := 4

theorem total_pieces_of_junk_mail : (pieces_per_block * num_blocks) = 192 := by
  sorry

end total_pieces_of_junk_mail_l113_11354


namespace pork_price_increase_l113_11363

variable (x : ℝ)
variable (P_aug P_oct : ℝ)
variable (P_aug := 32)
variable (P_oct := 64)

theorem pork_price_increase :
  P_aug * (1 + x) ^ 2 = P_oct :=
sorry

end pork_price_increase_l113_11363


namespace total_amount_spent_l113_11302

-- Definitions based on the conditions
def games_this_month := 11
def cost_per_ticket_this_month := 25
def total_cost_this_month := games_this_month * cost_per_ticket_this_month

def games_last_month := 17
def cost_per_ticket_last_month := 30
def total_cost_last_month := games_last_month * cost_per_ticket_last_month

def games_next_month := 16
def cost_per_ticket_next_month := 35
def total_cost_next_month := games_next_month * cost_per_ticket_next_month

-- Lean statement for the proof problem
theorem total_amount_spent :
  total_cost_this_month + total_cost_last_month + total_cost_next_month = 1345 :=
by
  -- proof goes here
  sorry

end total_amount_spent_l113_11302


namespace sin_b_in_triangle_l113_11333

theorem sin_b_in_triangle (a b : ℝ) (sin_A sin_B : ℝ) (h₁ : a = 2) (h₂ : b = 1) (h₃ : sin_A = 1 / 3) 
  (h₄ : sin_B = (b * sin_A) / a) : sin_B = 1 / 6 :=
by
  have h₅ : sin_B = 1 / 6 := by 
    sorry
  exact h₅

end sin_b_in_triangle_l113_11333


namespace mean_temperature_is_correct_l113_11399

-- Defining the list of temperatures
def temperatures : List ℝ := [75, 74, 76, 77, 80, 81, 83, 85, 83, 85]

-- Lean statement asserting the mean temperature is 79.9
theorem mean_temperature_is_correct : temperatures.sum / (temperatures.length: ℝ) = 79.9 := 
by
  sorry

end mean_temperature_is_correct_l113_11399


namespace arithmetic_geometric_seq_sum_5_l113_11307

-- Define the arithmetic-geometric sequence a_n
def a (n : ℕ) : ℤ := sorry

-- Define the sum S_n of the first n terms of the sequence a_n
def S (n : ℕ) : ℤ := sorry

-- Condition: a_1 = 1
axiom a1 : a 1 = 1

-- Condition: a_{n+2} + a_{n+1} - 2 * a_{n} = 0 for all n ∈ ℕ_+
axiom recurrence (n : ℕ) : a (n + 2) + a (n + 1) - 2 * a n = 0

-- Prove that S_5 = 11
theorem arithmetic_geometric_seq_sum_5 : S 5 = 11 := 
by
  sorry

end arithmetic_geometric_seq_sum_5_l113_11307


namespace max_value_expression_l113_11376

variable (a b : ℝ)

theorem max_value_expression (h : a^2 + b^2 = 3 + a * b) : 
  ∃ a b : ℝ, (2 * a - 3 * b)^2 + (a + 2 * b) * (a - 2 * b) = 22 :=
by
  -- This is a placeholder for the actual proof
  sorry

end max_value_expression_l113_11376


namespace right_triangle_side_lengths_l113_11358

theorem right_triangle_side_lengths :
  ¬ (4^2 + 5^2 = 6^2) ∧
  (12^2 + 16^2 = 20^2) ∧
  ¬ (5^2 + 10^2 = 13^2) ∧
  ¬ (8^2 + 40^2 = 41^2) := by
  sorry

end right_triangle_side_lengths_l113_11358


namespace hyperbola_focal_length_l113_11323

/--
In the Cartesian coordinate system \( xOy \),
let the focal length of the hyperbola \( \frac{x^{2}}{2m^{2}} - \frac{y^{2}}{3m} = 1 \) be 6.
Prove that the set of all real numbers \( m \) that satisfy this condition is {3/2}.
-/
theorem hyperbola_focal_length (m : ℝ) (h1 : 2 * m^2 > 0) (h2 : 3 * m > 0) (h3 : 2 * m^2 + 3 * m = 9) :
  m = 3 / 2 :=
sorry

end hyperbola_focal_length_l113_11323


namespace num_three_digit_perfect_cubes_divisible_by_16_l113_11347

-- define what it means for an integer to be a three-digit number
def is_three_digit (n : ℤ) : Prop := 100 ≤ n ∧ n ≤ 999

-- define what it means for an integer to be a perfect cube
def is_perfect_cube (n : ℤ) : Prop := ∃ m : ℤ, m^3 = n

-- define what it means for an integer to be divisible by 16
def is_divisible_by_sixteen (n : ℤ) : Prop := n % 16 = 0

-- define the main theorem that combines these conditions
theorem num_three_digit_perfect_cubes_divisible_by_16 : 
  ∃ n, n = 2 := sorry

end num_three_digit_perfect_cubes_divisible_by_16_l113_11347


namespace second_train_length_is_120_l113_11327

noncomputable def length_of_second_train
  (speed_train1_kmph : ℝ) 
  (speed_train2_kmph : ℝ) 
  (crossing_time : ℝ) 
  (length_train1_m : ℝ) : ℝ :=
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600
  let speed_train2_mps := speed_train2_kmph * 1000 / 3600
  let relative_speed := speed_train1_mps + speed_train2_mps
  let distance := relative_speed * crossing_time
  distance - length_train1_m

theorem second_train_length_is_120 :
  length_of_second_train 60 40 6.119510439164867 50 = 120 :=
by
  -- Here's where the proof would go
  sorry

end second_train_length_is_120_l113_11327


namespace replace_batteries_in_December_16_years_later_l113_11314

theorem replace_batteries_in_December_16_years_later :
  ∀ (n : ℕ), n = 30 → ∃ (years : ℕ) (months : ℕ), years = 16 ∧ months = 11 :=
by
  sorry

end replace_batteries_in_December_16_years_later_l113_11314


namespace scramble_words_count_l113_11342

-- Definitions based on the conditions
def alphabet_size : Nat := 25
def alphabet_size_no_B : Nat := 24

noncomputable def num_words_with_B : Nat :=
  let total_without_restriction := 25^1 + 25^2 + 25^3 + 25^4 + 25^5
  let total_without_B := 24^1 + 24^2 + 24^3 + 24^4 + 24^5
  total_without_restriction - total_without_B

-- Lean statement to prove the result
theorem scramble_words_count : num_words_with_B = 1692701 :=
by
  sorry

end scramble_words_count_l113_11342


namespace find_A_d_minus_B_d_l113_11304

-- Definitions of the conditions
def is_digit_in_base (x : ℕ) (d : ℕ) : Prop := x < d

def ab_aa_sum_to_172 (A B d : ℕ) : Prop :=
  is_digit_in_base A d ∧ is_digit_in_base B d ∧ d > 7 ∧ (d * A + B) + (d * A + A) = d^2 + 7 * d + 2

-- The final theorem statement
theorem find_A_d_minus_B_d (A B d : ℕ) (h : ab_aa_sum_to_172 A B d) : A - B = 5 :=
by sorry

end find_A_d_minus_B_d_l113_11304


namespace heather_total_oranges_l113_11348

-- Define the initial conditions
def initial_oranges : ℝ := 60.0
def additional_oranges : ℝ := 35.0

-- Define the total number of oranges
def total_oranges : ℝ := initial_oranges + additional_oranges

-- State the theorem that needs to be proven
theorem heather_total_oranges : total_oranges = 95.0 := 
by
  sorry

end heather_total_oranges_l113_11348


namespace necessarily_negative_l113_11396

theorem necessarily_negative (x y z : ℝ) 
  (hx : -1 < x ∧ x < 0) 
  (hy : 0 < y ∧ y < 1) 
  (hz : -2 < z ∧ z < -1) : 
  y + z < 0 := 
sorry

end necessarily_negative_l113_11396


namespace triangle_ABC_properties_l113_11338

theorem triangle_ABC_properties 
  (a b c : ℝ) (A B C : ℝ)
  (h1 : a * Real.sin C = Real.sqrt 3 * c * Real.cos A)
  (h2 : a = Real.sqrt 13)
  (h3 : c = 3)
  (h_angle_range : A > 0 ∧ A < Real.pi) : 
  A = Real.pi / 3 ∧ (1 / 2) * b * c * Real.sin A = 3 * Real.sqrt 3 := 
by
  sorry

end triangle_ABC_properties_l113_11338


namespace relationship_y1_y2_l113_11375

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem relationship_y1_y2 :
  ∀ (a b c x₀ x₁ x₂ : ℝ),
    (quadratic_function a b c 0 = 4) →
    (quadratic_function a b c 1 = 1) →
    (quadratic_function a b c 2 = 0) →
    1 < x₁ → 
    x₁ < 2 → 
    3 < x₂ → 
    x₂ < 4 → 
    (quadratic_function a b c x₁ < quadratic_function a b c x₂) :=
by 
  sorry

end relationship_y1_y2_l113_11375


namespace cistern_emptied_fraction_l113_11343

variables (minutes : ℕ) (fractionA fractionB fractionC : ℚ)

def pipeA_rate := 1 / 2 / 12
def pipeB_rate := 1 / 3 / 15
def pipeC_rate := 1 / 4 / 20

def time_active := 5

def emptiedA := pipeA_rate * time_active
def emptiedB := pipeB_rate * time_active
def emptiedC := pipeC_rate * time_active

def total_emptied := emptiedA + emptiedB + emptiedC

theorem cistern_emptied_fraction :
  total_emptied = 55 / 144 := by
  sorry

end cistern_emptied_fraction_l113_11343


namespace find_x2_plus_y2_l113_11360

open Real

theorem find_x2_plus_y2 (x y : ℝ) 
  (h1 : (x + y) ^ 4 + (x - y) ^ 4 = 4112)
  (h2 : x ^ 2 - y ^ 2 = 16) :
  x ^ 2 + y ^ 2 = 34 := 
sorry

end find_x2_plus_y2_l113_11360


namespace max_value_of_fraction_l113_11341

theorem max_value_of_fraction (a b : ℝ) (ha : a > 0) (hb : b > 1) (h_discriminant : a^2 = 4 * (b - 1)) :
  a = 2 → b = 2 → (3 * a + 2 * b) / (a + b) = 5 / 2 :=
by
  intro ha_eq
  intro hb_eq
  sorry

end max_value_of_fraction_l113_11341


namespace jessica_needs_stamps_l113_11372

-- Define the weights and conditions
def weight_of_paper := 1 / 5
def total_papers := 8
def weight_of_envelope := 2 / 5
def stamps_per_ounce := 1

-- Calculate the total weight and determine the number of stamps needed
theorem jessica_needs_stamps : 
  total_papers * weight_of_paper + weight_of_envelope = 2 :=
by
  sorry

end jessica_needs_stamps_l113_11372


namespace domain_of_f_l113_11340

noncomputable def f (x: ℝ): ℝ := 1 / Real.sqrt (x - 2)

theorem domain_of_f:
  {x: ℝ | 2 < x} = {x: ℝ | f x = 1 / Real.sqrt (x - 2)} :=
by
  sorry

end domain_of_f_l113_11340


namespace percent_millet_mix_correct_l113_11310

-- Define the necessary percentages
def percent_BrandA_in_mix : ℝ := 0.60
def percent_BrandB_in_mix : ℝ := 0.40
def percent_millet_in_BrandA : ℝ := 0.60
def percent_millet_in_BrandB : ℝ := 0.65

-- Define the overall percentage of millet in the mix
def percent_millet_in_mix : ℝ :=
  percent_BrandA_in_mix * percent_millet_in_BrandA +
  percent_BrandB_in_mix * percent_millet_in_BrandB

-- State the theorem
theorem percent_millet_mix_correct :
  percent_millet_in_mix = 0.62 :=
  by
    -- Here, we would provide the proof, but we use sorry as instructed.
    sorry

end percent_millet_mix_correct_l113_11310


namespace distance_C_to_D_l113_11317

noncomputable def side_length_smaller_square (perimeter : ℝ) : ℝ := perimeter / 4
noncomputable def side_length_larger_square (area : ℝ) : ℝ := Real.sqrt area

theorem distance_C_to_D 
  (perimeter_smaller : ℝ) (area_larger : ℝ) (h1 : perimeter_smaller = 8) (h2 : area_larger = 36) :
  let s_smaller := side_length_smaller_square perimeter_smaller
  let s_larger := side_length_larger_square area_larger 
  let leg1 := s_larger 
  let leg2 := s_larger - 2 * s_smaller 
  Real.sqrt (leg1 ^ 2 + leg2 ^ 2) = 2 * Real.sqrt 10 :=
by
  sorry

end distance_C_to_D_l113_11317


namespace probability_of_picking_letter_in_mathematics_l113_11390

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}

def unique_letters_in_mathematics : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

theorem probability_of_picking_letter_in_mathematics :
  (unique_letters_in_mathematics.card : ℚ) / (alphabet.card : ℚ) = 4 / 13 :=
by
  sorry

end probability_of_picking_letter_in_mathematics_l113_11390


namespace fraction_of_desks_full_l113_11335

-- Define the conditions
def restroom_students : ℕ := 2
def absent_students : ℕ := (3 * restroom_students) - 1
def total_students : ℕ := 23
def desks_per_row : ℕ := 6
def number_of_rows : ℕ := 4
def total_desks : ℕ := desks_per_row * number_of_rows
def students_in_classroom : ℕ := total_students - absent_students - restroom_students

-- Prove the fraction of desks that are full
theorem fraction_of_desks_full : (students_in_classroom : ℚ) / (total_desks : ℚ) = 2 / 3 :=
by
    sorry

end fraction_of_desks_full_l113_11335


namespace correct_conclusions_l113_11367

variable (f : ℝ → ℝ)

def condition_1 := ∀ x : ℝ, f (x + 2) = f (2 - (x + 2))
def condition_2 := ∀ x : ℝ, f (-2*x - 1) = -f (2*x + 1)

theorem correct_conclusions 
  (h1 : condition_1 f) 
  (h2 : condition_2 f) : 
  f 1 = f 3 ∧ 
  f 2 + f 4 = 0 ∧ 
  f (-1 / 2) * f (11 / 2) ≤ 0 := 
by 
  sorry

end correct_conclusions_l113_11367


namespace intersection_of_A_and_B_l113_11385

def A : Set (ℝ × ℝ) := {p | p.snd = 3 * p.fst - 2}
def B : Set (ℝ × ℝ) := {p | p.snd = p.fst ^ 2}

theorem intersection_of_A_and_B :
  {p : ℝ × ℝ | p ∈ A ∧ p ∈ B} = {(1, 1), (2, 4)} :=
by
  sorry

end intersection_of_A_and_B_l113_11385


namespace sum_of_powers_of_i_l113_11391

-- Define the imaginary unit and its property
def i : ℂ := Complex.I -- ℂ represents the complex numbers, Complex.I is the imaginary unit

-- The statement we need to prove
theorem sum_of_powers_of_i : i + i^2 + i^3 + i^4 = 0 := 
by {
  -- Lean requires the proof, but we will use sorry to skip it.
  -- Define the properties of i directly or use in-built properties
  sorry
}

end sum_of_powers_of_i_l113_11391


namespace encyclopedia_pages_count_l113_11334

theorem encyclopedia_pages_count (digits_used : ℕ) (h : digits_used = 6869) : ∃ pages : ℕ, pages = 1994 :=
by 
  sorry

end encyclopedia_pages_count_l113_11334


namespace hyperbola_standard_equation_l113_11344

theorem hyperbola_standard_equation :
  (∃ c : ℝ, c = Real.sqrt 5) →
  (∃ a b : ℝ, b / a = 2 ∧ a ^ 2 + b ^ 2 = 5) →
  (∃ a b : ℝ, a = 1 ∧ b = 2 ∧ (x ^ 2 / a ^ 2 - y ^ 2 / b ^ 2 = 1)) :=
by
  sorry

end hyperbola_standard_equation_l113_11344


namespace least_possible_xy_l113_11383

theorem least_possible_xy (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h : 1 / x + 1 / (3 * y) = 1 / 6) : x * y = 48 :=
by
  sorry

end least_possible_xy_l113_11383


namespace missing_coins_l113_11351

-- Definition representing the total number of coins Charlie received
variable (y : ℚ)

-- Conditions
def initial_lost_coins (y : ℚ) := (1 / 3) * y
def recovered_coins (y : ℚ) := (2 / 9) * y

-- Main Theorem
theorem missing_coins (y : ℚ) :
  y - (y * (8 / 9)) = y * (1 / 9) :=
by
  sorry

end missing_coins_l113_11351


namespace andrews_age_l113_11362

-- Define Andrew's age
variable (a g : ℚ)

-- Problem conditions
axiom condition1 : g = 10 * a
axiom condition2 : g - (a + 2) = 57

theorem andrews_age : a = 59 / 9 := 
by
  -- Set the proof steps aside for now
  sorry

end andrews_age_l113_11362


namespace J_of_given_values_l113_11388

def J (a b c : ℚ) : ℚ := a / b + b / c + c / a

theorem J_of_given_values : J 3 (-15) 10 = 49 / 30 := 
by 
  sorry

end J_of_given_values_l113_11388


namespace product_of_roots_l113_11377

theorem product_of_roots (a b c : ℤ) (h_eq : a = 24 ∧ b = 60 ∧ c = -600) :
  ∀ x : ℂ, (a * x^2 + b * x + c = 0) → (x * (-b - x) = -25) := sorry

end product_of_roots_l113_11377


namespace yacht_actual_cost_l113_11352

theorem yacht_actual_cost
  (discount_percentage : ℝ)
  (amount_paid : ℝ)
  (original_cost : ℝ)
  (h1 : discount_percentage = 0.72)
  (h2 : amount_paid = 3200000)
  (h3 : amount_paid = (1 - discount_percentage) * original_cost) :
  original_cost = 11428571.43 :=
by
  sorry

end yacht_actual_cost_l113_11352


namespace fraction_in_classroom_l113_11395

theorem fraction_in_classroom (total_students absent_fraction canteen_students present_students class_students : ℕ) 
  (h_total : total_students = 40)
  (h_absent_fraction : absent_fraction = 1 / 10)
  (h_canteen_students : canteen_students = 9)
  (h_absent_students : absent_fraction * total_students = 4)
  (h_present_students : present_students = total_students - absent_fraction * total_students)
  (h_class_students : class_students = present_students - canteen_students) :
  class_students / present_students = 3 / 4 := 
by {
  sorry
}

end fraction_in_classroom_l113_11395


namespace abs_eq_iff_mul_nonpos_l113_11365

theorem abs_eq_iff_mul_nonpos (a b : ℝ) : |a - b| = |a| + |b| ↔ a * b ≤ 0 :=
sorry

end abs_eq_iff_mul_nonpos_l113_11365


namespace longest_diagonal_length_l113_11321

theorem longest_diagonal_length (A : ℝ) (d1 d2 : ℝ) (h1 : A = 150) (h2 : d1 / d2 = 4 / 3) : d1 = 20 :=
by
  -- Skipping the proof here
  sorry

end longest_diagonal_length_l113_11321


namespace fraction_ratio_l113_11336

theorem fraction_ratio (x : ℚ) : 
  (x : ℚ) / (2/6) = (3/4) / (1/2) -> (x = 1/2) :=
by {
  sorry
}

end fraction_ratio_l113_11336


namespace polynomial_roots_l113_11320

theorem polynomial_roots :
  Polynomial.roots (3 * X^4 + 11 * X^3 - 28 * X^2 + 10 * X) = {0, 1/3, 2, -5} :=
sorry

end polynomial_roots_l113_11320


namespace sum_first_20_terms_l113_11303

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the conditions stated in the problem
variables {a : ℕ → ℤ}
variables (h_arith : is_arithmetic_sequence a)
variables (h_sum_first_three : a 1 + a 2 + a 3 = -24)
variables (h_sum_18_to_20 : a 18 + a 19 + a 20 = 78)

-- State the theorem to prove
theorem sum_first_20_terms : (Finset.range 20).sum a = 180 :=
by
  sorry

end sum_first_20_terms_l113_11303


namespace total_problems_completed_l113_11387

variables (p t : ℕ)
variables (hp_pos : 15 < p) (ht_pos : 0 < t)
variables (eq1 : (3 * p - 6) * (t - 3) = p * t)

theorem total_problems_completed : p * t = 120 :=
by sorry

end total_problems_completed_l113_11387


namespace outfit_choices_l113_11361

theorem outfit_choices (tops pants : ℕ) (TopsCount : tops = 4) (PantsCount : pants = 3) :
  tops * pants = 12 := by
  sorry

end outfit_choices_l113_11361


namespace sum_of_squares_of_distances_is_constant_l113_11316

variable {r1 r2 : ℝ}
variable {x y : ℝ}

theorem sum_of_squares_of_distances_is_constant
  (h1 : r1 < r2)
  (h2 : x^2 + y^2 = r1^2) :
  let PA := (x - r2)^2 + y^2
  let PB := (x + r2)^2 + y^2
  PA + PB = 2 * r1^2 + 2 * r2^2 :=
by
  sorry

end sum_of_squares_of_distances_is_constant_l113_11316


namespace vector_BC_is_correct_l113_11313

-- Given points B(1,2) and C(4,5)
def point_B := (1, 2)
def point_C := (4, 5)

-- Define the vector BC
def vector_BC (B C : ℕ × ℕ) : ℕ × ℕ :=
  (C.1 - B.1, C.2 - B.2)

-- Prove that the vector BC is (3, 3)
theorem vector_BC_is_correct : vector_BC point_B point_C = (3, 3) :=
  sorry

end vector_BC_is_correct_l113_11313


namespace angle_complement_half_supplement_is_zero_l113_11306

theorem angle_complement_half_supplement_is_zero (x : ℝ) 
  (h_complement: x - 90 = (1 / 2) * (x - 180)) : x = 0 := 
sorry

end angle_complement_half_supplement_is_zero_l113_11306


namespace probability_closer_to_center_radius6_eq_1_4_l113_11357

noncomputable def probability_closer_to_center (radius : ℝ) (r_inner : ℝ) :=
    let area_outer := Real.pi * radius ^ 2
    let area_inner := Real.pi * r_inner ^ 2
    area_inner / area_outer

theorem probability_closer_to_center_radius6_eq_1_4 :
    probability_closer_to_center 6 3 = 1 / 4 := by
    sorry

end probability_closer_to_center_radius6_eq_1_4_l113_11357


namespace fuchsia_to_mauve_l113_11371

theorem fuchsia_to_mauve (F : ℝ) :
  (5 / 8) * F + (3 * 26.67 : ℝ) = (3 / 8) * F + (5 / 8) * F →
  F = 106.68 :=
by
  intro h
  -- Step to implement the solution would go here
  sorry

end fuchsia_to_mauve_l113_11371


namespace part_one_l113_11374

theorem part_one (m : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = m * Real.exp x - x - 2) :
  (∀ x : ℝ, f x > 0) → m > Real.exp 1 :=
sorry

end part_one_l113_11374


namespace exists_x_y_for_2021_pow_n_l113_11318

theorem exists_x_y_for_2021_pow_n (n : ℕ) :
  (∃ x y : ℤ, 2021 ^ n = x ^ 4 - 4 * y ^ 4) ↔ ∃ m : ℕ, n = 4 * m := 
sorry

end exists_x_y_for_2021_pow_n_l113_11318


namespace students_not_making_cut_l113_11325

theorem students_not_making_cut
  (girls boys called_back : ℕ) 
  (h1 : girls = 39) 
  (h2 : boys = 4) 
  (h3 : called_back = 26) :
  (girls + boys) - called_back = 17 := 
by sorry

end students_not_making_cut_l113_11325


namespace solve_system_l113_11350

theorem solve_system (x y : ℚ) 
  (h1 : 3 * (x - 1) = y + 6) 
  (h2 : x / 2 + y / 3 = 2) : 
  x = 10 / 3 ∧ y = 1 := 
by 
  sorry

end solve_system_l113_11350


namespace median_of_consecutive_integers_l113_11381

theorem median_of_consecutive_integers (n : ℕ) (S : ℤ) (h1 : n = 35) (h2 : S = 1225) : 
  n % 2 = 1 → S / n = 35 := 
sorry

end median_of_consecutive_integers_l113_11381


namespace temperature_on_wednesday_l113_11311

theorem temperature_on_wednesday
  (T_sunday   : ℕ)
  (T_monday   : ℕ)
  (T_tuesday  : ℕ)
  (T_thursday : ℕ)
  (T_friday   : ℕ)
  (T_saturday : ℕ)
  (average_temperature : ℕ)
  (h_sunday   : T_sunday = 40)
  (h_monday   : T_monday = 50)
  (h_tuesday  : T_tuesday = 65)
  (h_thursday : T_thursday = 82)
  (h_friday   : T_friday = 72)
  (h_saturday : T_saturday = 26)
  (h_avg_temp : (T_sunday + T_monday + T_tuesday + W + T_thursday + T_friday + T_saturday) / 7 = average_temperature)
  (h_avg_val  : average_temperature = 53) :
  W = 36 :=
by { sorry }

end temperature_on_wednesday_l113_11311


namespace intersection_of_sets_l113_11373

noncomputable def A : Set ℝ := {x | -1 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 5}
noncomputable def B : Set ℝ := {x | 2 < x ∧ x < 4}

theorem intersection_of_sets : A ∩ B = {x : ℝ | 2 < x ∧ x ≤ 3} := 
by
  sorry

end intersection_of_sets_l113_11373


namespace typing_speed_ratio_l113_11319

-- Defining the conditions for the problem
def typing_speeds (T M : ℝ) : Prop :=
  (T + M = 12) ∧ (T + 1.25 * M = 14)

-- Stating the theorem with conditions and the expected result
theorem typing_speed_ratio (T M : ℝ) (h : typing_speeds T M) : M / T = 2 :=
by
  cases h
  sorry

end typing_speed_ratio_l113_11319


namespace exponentiation_properties_l113_11309

theorem exponentiation_properties
  (a : ℝ) (m n : ℕ) (hm : a^m = 9) (hn : a^n = 3) : a^(m - n) = 3 :=
by
  sorry

end exponentiation_properties_l113_11309


namespace misha_problem_l113_11368

theorem misha_problem (N : ℕ) (h : ∀ a, a ∈ {a | a > 1 → ∃ b > 0, b ∈ {b' | b' < a ∧ a % b' = 0}}) :
  (∀ t : ℕ, (t > 1) → (1 / t ^ 2) < (1 / t * (t - 1))) →
  (∃ (n : ℕ), n = 1) → (N = 1 ↔ ∃ (k : ℕ), k = N^2) :=
by
  sorry

end misha_problem_l113_11368


namespace video_files_count_l113_11332

-- Definitions for the given conditions
def total_files : ℝ := 48.0
def music_files : ℝ := 4.0
def picture_files : ℝ := 23.0

-- The proposition to prove
theorem video_files_count : total_files - (music_files + picture_files) = 21.0 :=
by
  sorry

end video_files_count_l113_11332


namespace find_M_l113_11389

theorem find_M :
  (∃ M: ℕ, (10 + 11 + 12) / 3 = (2022 + 2023 + 2024) / M) → M = 551 :=
by
  sorry

end find_M_l113_11389


namespace initial_numbers_is_five_l113_11328

theorem initial_numbers_is_five : 
  ∀ (n S : ℕ), 
    (12 * n = S) →
    (10 * (n - 1) = S - 20) → 
    n = 5 := 
by sorry

end initial_numbers_is_five_l113_11328


namespace population_growth_l113_11331

theorem population_growth :
  let scale_factor1 := 1 + 10 / 100
  let scale_factor2 := 1 + 20 / 100
  let k := 2 * 20
  let scale_factor3 := 1 + k / 100
  let combined_scale := scale_factor1 * scale_factor2 * scale_factor3
  (combined_scale - 1) * 100 = 84.8 :=
by
  sorry

end population_growth_l113_11331


namespace correct_statements_l113_11315

def studentsPopulation : Nat := 70000
def sampleSize : Nat := 1000
def isSamplePopulation (s : Nat) (p : Nat) : Prop := s < p
def averageSampleEqualsPopulation (sampleAvg populationAvg : ℕ) : Prop := sampleAvg = populationAvg
def isPopulation (p : Nat) : Prop := p = studentsPopulation

theorem correct_statements (p s : ℕ) (h1 : isSamplePopulation s p) (h2 : isPopulation p) 
  (h4 : s = sampleSize) : 
  (isSamplePopulation s p ∧ ¬averageSampleEqualsPopulation 1 1 ∧ isPopulation p ∧ s = sampleSize) := 
by
  sorry

end correct_statements_l113_11315


namespace g_1993_at_4_l113_11301

def g (x : ℚ) : ℚ := (2 + x) / (2 - 4 * x)

def g_n : ℕ → ℚ → ℚ
  | 0, x     => x
  | (n+1), x => g (g_n n x)

theorem g_1993_at_4 : g_n 1993 4 = 11 / 20 :=
by
  sorry

end g_1993_at_4_l113_11301


namespace baguettes_sold_third_batch_l113_11312

-- Definitions of the conditions
def daily_batches : ℕ := 3
def baguettes_per_batch : ℕ := 48
def baguettes_sold_first_batch : ℕ := 37
def baguettes_sold_second_batch : ℕ := 52
def baguettes_left : ℕ := 6

theorem baguettes_sold_third_batch : 
  daily_batches * baguettes_per_batch - (baguettes_sold_first_batch + baguettes_sold_second_batch + baguettes_left) = 49 :=
by sorry

end baguettes_sold_third_batch_l113_11312


namespace same_function_representation_l113_11330

theorem same_function_representation : 
  ∀ (f g : ℝ → ℝ), 
    (∀ x, f x = x^2 - 2*x - 1) ∧ (∀ m, g m = m^2 - 2*m - 1) →
    (f = g) :=
by
  sorry

end same_function_representation_l113_11330


namespace arithmetic_sequence_geometric_subsequence_l113_11379

theorem arithmetic_sequence_geometric_subsequence (a : ℕ → ℕ)
  (h1 : ∀ n, a (n + 1) = a n + 1)
  (h2 : (a 3)^2 = a 1 * a 7) :
  a 5 = 6 :=
sorry

end arithmetic_sequence_geometric_subsequence_l113_11379


namespace num_starting_lineups_l113_11346

def total_players := 15
def chosen_players := 3 -- Ace, Zeppo, Buddy already chosen
def remaining_players := total_players - chosen_players
def players_to_choose := 2 -- remaining players to choose

noncomputable def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem num_starting_lineups : combinations remaining_players players_to_choose = 66 := by
  sorry

end num_starting_lineups_l113_11346


namespace ivan_spent_fraction_l113_11324

theorem ivan_spent_fraction (f : ℝ) (h1 : 10 - 10 * f - 5 = 3) : f = 1 / 5 :=
by
  sorry

end ivan_spent_fraction_l113_11324


namespace find_m_value_l113_11382

def vectors_parallel (a1 a2 b1 b2 : ℝ) : Prop := a1 * b2 = a2 * b1

theorem find_m_value (m : ℝ) :
  let a := (6, 3)
  let b := (m, 2)
  vectors_parallel a.1 a.2 b.1 b.2 ↔ m = 4 :=
by
  intro H
  obtain ⟨_, _⟩ := H
  sorry

end find_m_value_l113_11382


namespace combined_reach_l113_11394

theorem combined_reach (barry_reach : ℝ) (larry_height : ℝ) (shoulder_ratio : ℝ) :
  barry_reach = 5 → larry_height = 5 → shoulder_ratio = 0.80 → 
  (larry_height * shoulder_ratio + barry_reach) = 9 :=
by
  intros h1 h2 h3
  sorry

end combined_reach_l113_11394


namespace f_divisible_by_64_l113_11339

theorem f_divisible_by_64 (n : ℕ) (h : n > 0) : 64 ∣ (3^(2*n + 2) - 8*n - 9) :=
sorry

end f_divisible_by_64_l113_11339


namespace min_value_problem_l113_11326

theorem min_value_problem (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 4) :
  (x + 1) * (2 * y + 1) / (x * y) ≥ 9 / 2 :=
by
  sorry

end min_value_problem_l113_11326


namespace Qing_Dynasty_Problem_l113_11366

variable {x y : ℕ}

theorem Qing_Dynasty_Problem (h1 : 4 * x + 6 * y = 48) (h2 : 2 * x + 5 * y = 38) :
  (4 * x + 6 * y = 48) ∧ (2 * x + 5 * y = 38) := by
  exact ⟨h1, h2⟩

end Qing_Dynasty_Problem_l113_11366


namespace ones_digit_sum_l113_11349

theorem ones_digit_sum : 
  (1 + 2 ^ 2023 + 3 ^ 2023 + 4 ^ 2023 + 5 : ℕ) % 10 = 5 := 
by 
  sorry

end ones_digit_sum_l113_11349


namespace false_statement_E_l113_11322

theorem false_statement_E
  (A B C : Type)
  (a b c : ℝ)
  (ha_gt_hb : a > b)
  (hb_gt_hc : b > c)
  (AB BC : ℝ)
  (hAB : AB = a - b → True)
  (hBC : BC = b + c → True)
  (hABC : AB + BC > a + b + c → True)
  (hAC : AB + BC > a - c → True) : False := sorry

end false_statement_E_l113_11322


namespace product_of_real_values_l113_11364

theorem product_of_real_values (r : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (1 / (3 * x)) = (r - x) / 8 → (3 * x * x - 3 * r * x + 8 = 0)) →
  r = 4 * Real.sqrt 6 / 3 ∨ r = -(4 * Real.sqrt 6 / 3) →
  r * -r = -32 / 3 :=
by
  intro h_x
  intro h_r
  sorry

end product_of_real_values_l113_11364


namespace remainder_when_divided_by_30_l113_11337

theorem remainder_when_divided_by_30 (n k R m : ℤ) (h1 : 0 ≤ R ∧ R < 30) (h2 : 2 * n % 15 = 2) (h3 : n = 30 * k + R) : R = 1 := by
  sorry

end remainder_when_divided_by_30_l113_11337


namespace brianna_books_gift_l113_11359

theorem brianna_books_gift (books_per_month : ℕ) (months_per_year : ℕ) (books_bought : ℕ) 
  (borrow_difference : ℕ) (books_reread : ℕ) (total_books_needed : ℕ) : 
  (books_per_month * months_per_year = total_books_needed) →
  ((books_per_month * months_per_year) - books_reread - 
  (books_bought + (books_bought - borrow_difference)) = 
  books_given) →
  books_given = 6 := 
by
  intro h1 h2
  sorry

end brianna_books_gift_l113_11359


namespace total_toothpicks_correct_l113_11353

noncomputable def total_toothpicks_in_grid 
  (height : ℕ) (width : ℕ) (partition_interval : ℕ) : ℕ :=
  let horizontal_lines := height + 1
  let vertical_lines := width + 1
  let num_partitions := height / partition_interval
  (horizontal_lines * width) + (vertical_lines * height) + (num_partitions * width)

theorem total_toothpicks_correct :
  total_toothpicks_in_grid 25 15 5 = 850 := 
by 
  sorry

end total_toothpicks_correct_l113_11353


namespace sum_of_g_31_values_l113_11370

def f (x : ℝ) : ℝ := 4 * x^2 - 3
def g (y : ℝ) : ℝ := y ^ 2 - y + 2

theorem sum_of_g_31_values :
  g 31 + g 31 = 21 := sorry

end sum_of_g_31_values_l113_11370


namespace contrapositive_of_equality_square_l113_11305

theorem contrapositive_of_equality_square (a b : ℝ) (h : a^2 ≠ b^2) : a ≠ b := 
by 
  sorry

end contrapositive_of_equality_square_l113_11305


namespace victor_wins_ratio_l113_11345

theorem victor_wins_ratio (victor_wins friend_wins : ℕ) (hvw : victor_wins = 36) (fw : friend_wins = 20) : (victor_wins : ℚ) / friend_wins = 9 / 5 :=
by
  sorry

end victor_wins_ratio_l113_11345


namespace not_necessarily_divisible_by_20_l113_11393

theorem not_necessarily_divisible_by_20 (k : ℤ) (h : ∃ k : ℤ, 5 ∣ k * (k+1) * (k+2)) : ¬ ∀ k : ℤ, 20 ∣ k * (k+1) * (k+2) :=
by
  sorry

end not_necessarily_divisible_by_20_l113_11393


namespace unique_H_value_l113_11378

theorem unique_H_value :
  ∀ (T H R E F I V S : ℕ),
    T = 8 →
    E % 2 = 1 →
    E ≠ T ∧ E ≠ H ∧ E ≠ R ∧ E ≠ F ∧ E ≠ I ∧ E ≠ V ∧ E ≠ S ∧ 
    H ≠ T ∧ H ≠ R ∧ H ≠ F ∧ H ≠ I ∧ H ≠ V ∧ H ≠ S ∧
    F ≠ T ∧ F ≠ I ∧ F ≠ V ∧ F ≠ S ∧
    I ≠ T ∧ I ≠ V ∧ I ≠ S ∧
    V ≠ T ∧ V ≠ S ∧
    S ≠ T ∧
    (8 + 8) = 10 + F ∧
    (E + E) % 10 = 6 →
    H + H = 10 + 4 →
    H = 7 := 
sorry

end unique_H_value_l113_11378


namespace sale_in_fifth_month_l113_11380

theorem sale_in_fifth_month (a1 a2 a3 a4 a5 a6 avg : ℝ)
  (h1 : a1 = 5420) (h2 : a2 = 5660) (h3 : a3 = 6200) (h4 : a4 = 6350) (h6 : a6 = 6470) (h_avg : avg = 6100) :
  a5 = 6500 :=
by
  sorry

end sale_in_fifth_month_l113_11380


namespace max_value_l113_11398

variable (a b c d : ℝ)

theorem max_value 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : b ≠ c) 
  (h5 : b ≠ d) (h6 : c ≠ d)
  (cond1 : a / b + b / c + c / d + d / a = 4)
  (cond2 : a * c = b * d) :
  (a / c + b / d + c / a + d / b) ≤ -12 :=
sorry

end max_value_l113_11398


namespace abs_sin_diff_le_abs_sin_sub_l113_11392

theorem abs_sin_diff_le_abs_sin_sub (A B : ℝ) (hA : 0 ≤ A) (hA' : A ≤ π) (hB : 0 ≤ B) (hB' : B ≤ π) :
  |Real.sin A - Real.sin B| ≤ |Real.sin (A - B)| :=
by
  -- Proof would go here
  sorry

end abs_sin_diff_le_abs_sin_sub_l113_11392


namespace solve_congruence_l113_11308

theorem solve_congruence : ∃ (a m : ℕ), 10 * x + 3 ≡ 7 [MOD 18] ∧ x ≡ a [MOD m] ∧ a < m ∧ m ≥ 2 ∧ a + m = 13 := 
sorry

end solve_congruence_l113_11308


namespace find_integer_pairs_l113_11384

theorem find_integer_pairs (x y : ℕ) (h : x ^ 5 = y ^ 5 + 10 * y ^ 2 + 20 * y + 1) : (x, y) = (1, 0) :=
  sorry

end find_integer_pairs_l113_11384
