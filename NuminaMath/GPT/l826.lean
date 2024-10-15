import Mathlib

namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l826_82637

noncomputable def common_ratio_q (a1 a5 a : ℕ) (q : ℕ) : Prop :=
  a1 * a5 = 16 ∧ a1 > 0 ∧ a5 > 0 ∧ a = 2 ∧ q = 2

theorem geometric_sequence_common_ratio : ∀ (a1 a5 a q : ℕ), 
  common_ratio_q a1 a5 a q → q = 2 :=
by
  intros a1 a5 a q h
  have h1 : a1 * a5 = 16 := h.1
  have h2 : a1 > 0 := h.2.1
  have h3 : a5 > 0 := h.2.2.1
  have h4 : a = 2 := h.2.2.2.1
  have h5 : q = 2 := h.2.2.2.2
  exact h5

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l826_82637


namespace NUMINAMATH_GPT_novel_to_history_ratio_l826_82634

-- Define the conditions
def history_book_pages : ℕ := 300
def science_book_pages : ℕ := 600
def novel_pages := science_book_pages / 4

-- Define the target ratio to prove
def target_ratio := (novel_pages : ℚ) / (history_book_pages : ℚ)

theorem novel_to_history_ratio :
  target_ratio = (1 : ℚ) / (2 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_novel_to_history_ratio_l826_82634


namespace NUMINAMATH_GPT_required_run_rate_l826_82696

theorem required_run_rate (target : ℝ) (initial_run_rate : ℝ) (initial_overs : ℕ) (remaining_overs : ℕ) :
  target = 282 → initial_run_rate = 3.8 → initial_overs = 10 → remaining_overs = 40 →
  (target - initial_run_rate * initial_overs) / remaining_overs = 6.1 :=
by
  intros
  sorry

end NUMINAMATH_GPT_required_run_rate_l826_82696


namespace NUMINAMATH_GPT_find_central_cell_l826_82621

variable (a b c d e f g h i : ℝ)

def condition_1 : Prop :=
  a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10

def condition_2 : Prop :=
  a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10

def condition_3 : Prop :=
  a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3

theorem find_central_cell (h1 : condition_1 a b c d e f g h i)
                          (h2 : condition_2 a b c d e f g h i)
                          (h3 : condition_3 a b c d e f g h i) : 
  e = 0.00081 := 
sorry

end NUMINAMATH_GPT_find_central_cell_l826_82621


namespace NUMINAMATH_GPT_area_of_triangle_l826_82648

-- Define the function to calculate the area of a right isosceles triangle given the side lengths of squares
theorem area_of_triangle (a b c : ℝ) (h1 : a = 10) (h2 : b = 8) (h3 : c = 10) (right_isosceles : true) :
  (1 / 2) * a * c = 50 :=
by
  -- We state the theorem but leave the proof as sorry.
  sorry

end NUMINAMATH_GPT_area_of_triangle_l826_82648


namespace NUMINAMATH_GPT_sin_double_angle_l826_82676

theorem sin_double_angle (x : ℝ)
  (h : Real.sin (x + Real.pi / 4) = 4 / 5) :
  Real.sin (2 * x) = 7 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l826_82676


namespace NUMINAMATH_GPT_number_of_photographs_is_twice_the_number_of_paintings_l826_82622

theorem number_of_photographs_is_twice_the_number_of_paintings (P Q : ℕ) :
  (Q * (Q - 1) * P) = 2 * (P * (Q * (Q - 1)) / 2) := by
  sorry

end NUMINAMATH_GPT_number_of_photographs_is_twice_the_number_of_paintings_l826_82622


namespace NUMINAMATH_GPT_area_of_quadrilateral_l826_82653

def Quadrilateral (A B C D : Type) :=
  ∃ (ABC_deg : ℝ) (ADC_deg : ℝ) (AD : ℝ) (DC : ℝ) (AB : ℝ) (BC : ℝ),
  (ABC_deg = 90) ∧ (ADC_deg = 90) ∧ (AD = DC) ∧ (AB + BC = 20)

theorem area_of_quadrilateral (A B C D : Type) (h : Quadrilateral A B C D) : 
  ∃ (area : ℝ), area = 100 := 
sorry

end NUMINAMATH_GPT_area_of_quadrilateral_l826_82653


namespace NUMINAMATH_GPT_function_satisfies_condition_l826_82615

noncomputable def f : ℕ → ℕ := sorry

theorem function_satisfies_condition (f : ℕ → ℕ) (h : ∀ n : ℕ, 0 < n → f (n + 1) > (f n + f (f n)) / 2) :
  (∃ b : ℕ, ∀ n : ℕ, (n < b → f n = n) ∧ (n ≥ b → f n = n + 1)) :=
sorry

end NUMINAMATH_GPT_function_satisfies_condition_l826_82615


namespace NUMINAMATH_GPT_molly_gift_cost_l826_82609

noncomputable def cost_per_package : ℕ := 5
noncomputable def num_parents : ℕ := 2
noncomputable def num_brothers : ℕ := 3
noncomputable def num_sisters_in_law : ℕ := num_brothers -- each brother is married
noncomputable def num_children_per_brother : ℕ := 2
noncomputable def num_nieces_nephews : ℕ := num_brothers * num_children_per_brother
noncomputable def total_relatives : ℕ := num_parents + num_brothers + num_sisters_in_law + num_nieces_nephews

theorem molly_gift_cost : (total_relatives * cost_per_package) = 70 := by
  sorry

end NUMINAMATH_GPT_molly_gift_cost_l826_82609


namespace NUMINAMATH_GPT_ferry_tourist_total_l826_82649

theorem ferry_tourist_total :
  let number_of_trips := 8
  let a := 120 -- initial number of tourists
  let d := -2  -- common difference
  let total_tourists := (number_of_trips * (2 * a + (number_of_trips - 1) * d)) / 2
  total_tourists = 904 := 
by {
  sorry
}

end NUMINAMATH_GPT_ferry_tourist_total_l826_82649


namespace NUMINAMATH_GPT_maria_purse_value_l826_82604

def value_of_nickels (num_nickels : ℕ) : ℕ := num_nickels * 5
def value_of_dimes (num_dimes : ℕ) : ℕ := num_dimes * 10
def value_of_quarters (num_quarters : ℕ) : ℕ := num_quarters * 25
def total_value (num_nickels num_dimes num_quarters : ℕ) : ℕ := 
  value_of_nickels num_nickels + value_of_dimes num_dimes + value_of_quarters num_quarters
def percentage_of_dollar (value_cents : ℕ) : ℕ := value_cents * 100 / 100

theorem maria_purse_value : percentage_of_dollar (total_value 2 3 2) = 90 := by
  sorry

end NUMINAMATH_GPT_maria_purse_value_l826_82604


namespace NUMINAMATH_GPT_sequence_a_n_l826_82636

theorem sequence_a_n {a : ℕ → ℤ}
  (h1 : a 2 = 5)
  (h2 : a 1 = 1)
  (h3 : ∀ n ≥ 2, a (n+1) - 2 * a n + a (n-1) = 7) :
  a 17 = 905 :=
  sorry

end NUMINAMATH_GPT_sequence_a_n_l826_82636


namespace NUMINAMATH_GPT_remove_remaining_wallpaper_time_l826_82600

noncomputable def time_per_wall : ℕ := 2
noncomputable def walls_dining_room : ℕ := 4
noncomputable def walls_living_room : ℕ := 4
noncomputable def walls_completed : ℕ := 1

theorem remove_remaining_wallpaper_time : 
    time_per_wall * (walls_dining_room - walls_completed) + time_per_wall * walls_living_room = 14 :=
by
  sorry

end NUMINAMATH_GPT_remove_remaining_wallpaper_time_l826_82600


namespace NUMINAMATH_GPT_probability_of_x_greater_than_3y_l826_82646

noncomputable def area_triangle (base height : ℝ) : ℝ := 0.5 * base * height
noncomputable def area_rectangle (width height : ℝ) : ℝ := width * height
noncomputable def probability (area_triangle area_rectangle : ℝ) : ℝ := area_triangle / area_rectangle

theorem probability_of_x_greater_than_3y :
  let base := 2010
  let height_triangle := base / 3
  let height_rectangle := 2011
  let area_triangle := area_triangle base height_triangle
  let area_rectangle := area_rectangle base height_rectangle
  let prob := probability area_triangle area_rectangle
  prob = (335 : ℝ) / 2011 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_x_greater_than_3y_l826_82646


namespace NUMINAMATH_GPT_problem_statement_l826_82629

noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 ≤ x ∧ x ≤ 1 then 2 * x * (1 - x)
  else -- define elsewhere based on periodicity and oddness properties
    sorry 

theorem problem_statement : 
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 4) = f x) → f 2015.5 = -0.5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_problem_statement_l826_82629


namespace NUMINAMATH_GPT_smallest_cube_with_divisor_l826_82656

theorem smallest_cube_with_divisor (p q r : ℕ) (hp : p.Prime) (hq : q.Prime) (hr : r.Prime) (hpq : p ≠ q) (hpr : p ≠ r) (hqr : q ≠ r) :
  ∃ (m : ℕ), m = (p * q * r^2) ^ 3 ∧ (p * q^3 * r^5 ∣ m) :=
by
  sorry

end NUMINAMATH_GPT_smallest_cube_with_divisor_l826_82656


namespace NUMINAMATH_GPT_half_plus_five_l826_82642

theorem half_plus_five (n : ℕ) (h : n = 16) : n / 2 + 5 = 13 := by
  sorry

end NUMINAMATH_GPT_half_plus_five_l826_82642


namespace NUMINAMATH_GPT_proof_problem_l826_82662

-- Definitions for the given conditions in the problem
def equations (a x y : ℝ) : Prop :=
(x + 5 * y = 4 - a) ∧ (x - y = 3 * a)

-- The conclusions from the problem
def conclusion1 (a x y : ℝ) : Prop :=
a = 1 → x + y = 4 - a

def conclusion2 (a x y : ℝ) : Prop :=
a = -2 → x = -y

def conclusion3 (a x y : ℝ) : Prop :=
2 * x + 7 * y = 6

def conclusion4 (a x y : ℝ) : Prop :=
x ≤ 1 → y > 4 / 7

-- The main theorem to be proven
theorem proof_problem (a x y : ℝ) :
  equations a x y →
  (¬ conclusion1 a x y ∨ ¬ conclusion2 a x y ∨ ¬ conclusion3 a x y ∨ ¬ conclusion4 a x y) →
  (∃ n : ℕ, n = 2 ∧ ((conclusion1 a x y ∨ conclusion2 a x y ∨ conclusion3 a x y ∨ conclusion4 a x y) → false)) :=
by {
  sorry
}

end NUMINAMATH_GPT_proof_problem_l826_82662


namespace NUMINAMATH_GPT_inequalities_count_three_l826_82697

theorem inequalities_count_three
  (x y a b : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (h1 : x^2 < a^2) (h2 : y^3 < b^3) :
  (x^2 + y^2 < a^2 + b^2) ∧ ¬(x^2 - y^2 < a^2 - b^2) ∧ (x^2 * y^3 < a^2 * b^3) ∧ (x^2 / y^3 < a^2 / b^3) := 
sorry

end NUMINAMATH_GPT_inequalities_count_three_l826_82697


namespace NUMINAMATH_GPT_tiffany_initial_lives_l826_82647

variable (x : ℝ) -- Define the variable x representing the initial number of lives

-- Define the conditions
def condition1 : Prop := x + 14.0 + 27.0 = 84.0

-- Prove the initial number of lives
theorem tiffany_initial_lives (h : condition1 x) : x = 43.0 := by
  sorry

end NUMINAMATH_GPT_tiffany_initial_lives_l826_82647


namespace NUMINAMATH_GPT_selling_price_correct_l826_82665

noncomputable def cost_price : ℝ := 100
noncomputable def gain_percent : ℝ := 0.15
noncomputable def profit : ℝ := gain_percent * cost_price
noncomputable def selling_price : ℝ := cost_price + profit

theorem selling_price_correct : selling_price = 115 := by
  sorry

end NUMINAMATH_GPT_selling_price_correct_l826_82665


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l826_82644

-- Define sets M and N as given in the conditions
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

-- The theorem statement to prove the intersection of M and N is {2, 3}
theorem intersection_of_M_and_N : M ∩ N = {2, 3} := 
by sorry  -- The proof is skipped with 'sorry'

end NUMINAMATH_GPT_intersection_of_M_and_N_l826_82644


namespace NUMINAMATH_GPT_sector_area_l826_82657

theorem sector_area (α r : ℝ) (hα : α = 2) (h_r : r = 1 / Real.sin 1) : 
  (1 / 2) * r^2 * α = 1 / (Real.sin 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_sector_area_l826_82657


namespace NUMINAMATH_GPT_sin_cos_15_deg_l826_82623

noncomputable def sin_deg (deg : ℝ) : ℝ := Real.sin (deg * Real.pi / 180)
noncomputable def cos_deg (deg : ℝ) : ℝ := Real.cos (deg * Real.pi / 180)

theorem sin_cos_15_deg :
  (sin_deg 15 + cos_deg 15) * (sin_deg 15 - cos_deg 15) = -Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_cos_15_deg_l826_82623


namespace NUMINAMATH_GPT_find_speed_of_second_boy_l826_82675

theorem find_speed_of_second_boy
  (v : ℝ)
  (speed_first_boy : ℝ)
  (distance_apart : ℝ)
  (time_taken : ℝ)
  (h1 : speed_first_boy = 5.3)
  (h2 : distance_apart = 10.5)
  (h3 : time_taken = 35) :
  v = 5.6 :=
by {
  -- translation of the steps to work on the proof
  -- sorry is used to indicate that the proof is not provided here
  sorry
}

end NUMINAMATH_GPT_find_speed_of_second_boy_l826_82675


namespace NUMINAMATH_GPT_min_value_sqrt_expression_l826_82608

open Real

theorem min_value_sqrt_expression : ∃ x : ℝ, ∀ y : ℝ, 
  sqrt (y^2 + (2 - y)^2) + sqrt ((y - 1)^2 + (y + 2)^2) ≥ sqrt 17 :=
by
  sorry

end NUMINAMATH_GPT_min_value_sqrt_expression_l826_82608


namespace NUMINAMATH_GPT_investment_interest_min_l826_82679

theorem investment_interest_min (x y : ℝ) (hx : x + y = 25000) (hmax : x ≤ 11000) : 
  0.07 * x + 0.12 * y ≥ 2450 :=
by
  sorry

end NUMINAMATH_GPT_investment_interest_min_l826_82679


namespace NUMINAMATH_GPT_root_exists_in_interval_l826_82690

noncomputable def f (x : ℝ) : ℝ := x^3 + 2 * x - 1

theorem root_exists_in_interval : 
  ∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0 := 
by
  sorry

end NUMINAMATH_GPT_root_exists_in_interval_l826_82690


namespace NUMINAMATH_GPT_rita_hours_per_month_l826_82625

theorem rita_hours_per_month :
  let t := 1500
  let h_backstroke := 50
  let h_breaststroke := 9
  let h_butterfly := 121
  let m := 6
  let h_completed := h_backstroke + h_breaststroke + h_butterfly
  let h_remaining := t - h_completed
  let h := h_remaining / m
  h = 220
:= by 
  let t := 1500
  let h_backstroke := 50
  let h_breaststroke := 9
  let h_butterfly := 121
  let m := 6
  let h_completed := h_backstroke + h_breaststroke + h_butterfly
  have h_remaining := t - h_completed
  have h := h_remaining / m
  sorry

end NUMINAMATH_GPT_rita_hours_per_month_l826_82625


namespace NUMINAMATH_GPT_assume_proof_by_contradiction_l826_82669

theorem assume_proof_by_contradiction (a b : ℤ) (hab : ∃ k : ℤ, ab = 3 * k) :
  (¬ (∃ k : ℤ, a = 3 * k) ∧ ¬ (∃ k : ℤ, b = 3 * k)) :=
sorry

end NUMINAMATH_GPT_assume_proof_by_contradiction_l826_82669


namespace NUMINAMATH_GPT_furniture_definition_based_on_vocabulary_study_l826_82671

theorem furniture_definition_based_on_vocabulary_study (term : String) (h : term = "furniture") :
  term = "furniture" :=
by
  sorry

end NUMINAMATH_GPT_furniture_definition_based_on_vocabulary_study_l826_82671


namespace NUMINAMATH_GPT_find_side_lengths_l826_82677

variable (a b : ℝ)

-- Conditions
def diff_side_lengths := a - b = 2
def diff_areas := a^2 - b^2 = 40

-- Theorem to prove
theorem find_side_lengths (h1 : diff_side_lengths a b) (h2 : diff_areas a b) :
  a = 11 ∧ b = 9 := by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_find_side_lengths_l826_82677


namespace NUMINAMATH_GPT_smallest_add_to_2002_l826_82602

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def next_palindrome_after (n : ℕ) : ℕ :=
  -- a placeholder function for the next palindrome calculation
  -- implementation logic is skipped
  2112

def smallest_add_to_palindrome (n target : ℕ) : ℕ :=
  target - n

theorem smallest_add_to_2002 :
  let target := next_palindrome_after 2002
  ∃ k, is_palindrome (2002 + k) ∧ (2002 < 2002 + k) ∧ target = 2002 + k ∧ k = 110 := 
by
  use 110
  sorry

end NUMINAMATH_GPT_smallest_add_to_2002_l826_82602


namespace NUMINAMATH_GPT_moon_iron_percentage_l826_82643

variables (x : ℝ) -- percentage of iron in the moon

-- Given conditions
def carbon_percentage_of_moon : ℝ := 0.20
def mass_of_moon : ℝ := 250
def mass_of_mars : ℝ := 2 * mass_of_moon
def mass_of_other_elements_on_mars : ℝ := 150
def composition_same (m : ℝ) (x : ℝ) := 
  (x / 100 * m + carbon_percentage_of_moon * m + (100 - x - 20) / 100 * m) = m

-- Theorem statement
theorem moon_iron_percentage : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_moon_iron_percentage_l826_82643


namespace NUMINAMATH_GPT_solve_equation_l826_82641

theorem solve_equation (x : ℝ) (h : (x^2 - x + 2) / (x - 1) = x + 3) (h1 : x ≠ 1) : 
  x = 5 / 3 :=
sorry

end NUMINAMATH_GPT_solve_equation_l826_82641


namespace NUMINAMATH_GPT_hash_difference_l826_82666

def hash (x y : ℕ) : ℤ := x * y - 3 * x + y

theorem hash_difference : (hash 8 5) - (hash 5 8) = -12 := by
  sorry

end NUMINAMATH_GPT_hash_difference_l826_82666


namespace NUMINAMATH_GPT_hiking_committee_selection_l826_82668

def comb (n k : ℕ) : ℕ := n.choose k

theorem hiking_committee_selection :
  comb 10 3 = 120 :=
by
  sorry

end NUMINAMATH_GPT_hiking_committee_selection_l826_82668


namespace NUMINAMATH_GPT_prime_saturated_96_l826_82685

def is_prime_saturated (d : ℕ) : Prop :=
  let prime_factors := [2, 3]  -- list of the different positive prime factors of 96
  prime_factors.prod < d       -- the product of prime factors should be less than d

theorem prime_saturated_96 : is_prime_saturated 96 :=
by
  sorry

end NUMINAMATH_GPT_prime_saturated_96_l826_82685


namespace NUMINAMATH_GPT_triangle_at_most_one_obtuse_l826_82674

theorem triangle_at_most_one_obtuse (A B C : ℝ) (h1 : 0 < A ∧ A < 180) (h2 : 0 < B ∧ B < 180) (h3 : 0 < C ∧ C < 180) (h4 : A + B + C = 180) : A ≤ 90 ∨ B ≤ 90 ∨ C ≤ 90 :=
by
  sorry

end NUMINAMATH_GPT_triangle_at_most_one_obtuse_l826_82674


namespace NUMINAMATH_GPT_longest_badminton_match_duration_l826_82619

theorem longest_badminton_match_duration :
  let hours := 12
  let minutes := 25
  (hours * 60 + minutes = 745) :=
by
  sorry

end NUMINAMATH_GPT_longest_badminton_match_duration_l826_82619


namespace NUMINAMATH_GPT_line_tangent_to_parabola_l826_82686

theorem line_tangent_to_parabola (d : ℝ) :
  (∀ x y: ℝ, y = 3 * x + d ↔ y^2 = 12 * x) → d = 1 :=
by
  sorry

end NUMINAMATH_GPT_line_tangent_to_parabola_l826_82686


namespace NUMINAMATH_GPT_imaginary_part_of_quotient_l826_82660

noncomputable def imaginary_part_of_complex (z : ℂ) : ℂ := z.im

theorem imaginary_part_of_quotient :
  imaginary_part_of_complex (i / (1 - i)) = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_imaginary_part_of_quotient_l826_82660


namespace NUMINAMATH_GPT_number_of_lattice_points_in_triangle_l826_82691

theorem number_of_lattice_points_in_triangle (N L S : ℕ) (A B O : (ℕ × ℕ)) :
  (A = (0, 30)) →
  (B = (20, 10)) →
  (O = (0, 0)) →
  (S = 300) →
  (L = 60) →
  S = N + L / 2 - 1 →
  N = 271 :=
by
  intros hA hB hO hS hL hPick
  sorry

end NUMINAMATH_GPT_number_of_lattice_points_in_triangle_l826_82691


namespace NUMINAMATH_GPT_find_n_after_folding_l826_82650

theorem find_n_after_folding (n : ℕ) (h : 2 ^ n = 128) : n = 7 := by
  sorry

end NUMINAMATH_GPT_find_n_after_folding_l826_82650


namespace NUMINAMATH_GPT_quadratic_other_x_intercept_l826_82654

theorem quadratic_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 5 → a * x^2 + b * x + c = -3)
  (h_intercept : a * 1^2 + b * 1 + c = 0) : 
  ∃ x0 : ℝ, x0 = 9 ∧ a * x0^2 + b * x0 + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_other_x_intercept_l826_82654


namespace NUMINAMATH_GPT_height_difference_l826_82661

variables (H1 H2 H3 : ℕ)
variable (x : ℕ)
variable (h_ratio : H1 = 4 * x ∧ H2 = 5 * x ∧ H3 = 6 * x)
variable (h_lightest : H1 = 120)

theorem height_difference :
  (H1 + H3) - H2 = 150 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_height_difference_l826_82661


namespace NUMINAMATH_GPT_vacationers_city_correctness_l826_82638

noncomputable def vacationer_cities : Prop :=
  ∃ (city : String → String),
    (city "Amelie" = "Acapulco" ∨ city "Amelie" = "Brest" ∨ city "Amelie" = "Madrid") ∧
    (city "Benoit" = "Acapulco" ∨ city "Benoit" = "Brest" ∨ city "Benoit" = "Madrid") ∧
    (city "Pierre" = "Paris" ∨ city "Pierre" = "Brest" ∨ city "Pierre" = "Madrid") ∧
    (city "Melanie" = "Acapulco" ∨ city "Melanie" = "Brest" ∨ city "Melanie" = "Madrid") ∧
    (city "Charles" = "Acapulco" ∨ city "Charles" = "Brest" ∨ city "Charles" = "Madrid") ∧
    -- Conditions stated by participants
    ((city "Amelie" = "Acapulco") ∨ (city "Amelie" ≠ "Acapulco" ∧ city "Benoit" = "Acapulco" ∧ city "Pierre" = "Paris")) ∧
    ((city "Benoit" = "Brest") ∨ (city "Benoit" ≠ "Brest" ∧ city "Charles" = "Brest" ∧ city "Pierre" = "Paris")) ∧
    ((city "Pierre" ≠ "France") ∨ (city "Pierre" = "Paris" ∧ city "Amelie" ≠ "France" ∧ city "Melanie" = "Madrid")) ∧
    ((city "Melanie" = "Clermont-Ferrand") ∨ (city "Melanie" ≠ "Clermont-Ferrand" ∧ city "Amelie" = "Acapulco" ∧ city "Pierre" = "Paris")) ∧
    ((city "Charles" = "Clermont-Ferrand") ∨ (city "Charles" ≠ "Clermont-Ferrand" ∧ city "Amelie" = "Acapulco" ∧ city "Benoit" = "Acapulco"))

theorem vacationers_city_correctness : vacationer_cities :=
  sorry

end NUMINAMATH_GPT_vacationers_city_correctness_l826_82638


namespace NUMINAMATH_GPT_interval_length_condition_l826_82693

theorem interval_length_condition (c : ℝ) (x : ℝ) (H1 : 3 ≤ 5 * x - 4) (H2 : 5 * x - 4 ≤ c) 
                                  (H3 : (c + 4) / 5 - 7 / 5 = 15) : c - 3 = 75 := 
sorry

end NUMINAMATH_GPT_interval_length_condition_l826_82693


namespace NUMINAMATH_GPT_point_a_coordinates_l826_82692

open Set

theorem point_a_coordinates (A B : ℝ × ℝ) :
  B = (2, 4) →
  (A.1 = B.1 + 3 ∨ A.1 = B.1 - 3) ∧ A.2 = B.2 →
  dist A B = 3 →
  A = (5, 4) ∨ A = (-1, 4) :=
by
  intros hB hA hDist
  sorry

end NUMINAMATH_GPT_point_a_coordinates_l826_82692


namespace NUMINAMATH_GPT_A_is_guilty_l826_82687

-- Define the conditions
variables (A B C : Prop)  -- A, B, C are the propositions that represent the guilt of the individuals A, B, and C
variable  (car : Prop)    -- car represents the fact that the crime involved a car
variable  (C_never_alone : C → A)  -- C never commits a crime without A

-- Facts:
variables (crime_committed : A ∨ B ∨ C) -- the crime was committed by A, B, or C (or a combination)
variable  (B_knows_drive : B → car)     -- B knows how to drive

-- The proof goal: Show that A is guilty.
theorem A_is_guilty : A :=
sorry

end NUMINAMATH_GPT_A_is_guilty_l826_82687


namespace NUMINAMATH_GPT_nature_of_roots_of_quadratic_l826_82611

theorem nature_of_roots_of_quadratic (k : ℝ) (h1 : k > 0) (h2 : 3 * k^2 - 2 = 10) :
  let a := 1
  let b := -(4 * k - 3)
  let c := 3 * k^2 - 2
  let Δ := b^2 - 4 * a * c
  Δ < 0 :=
by
  sorry

end NUMINAMATH_GPT_nature_of_roots_of_quadratic_l826_82611


namespace NUMINAMATH_GPT_compare_abc_l826_82683

theorem compare_abc 
  (a : ℝ := 1 / 11) 
  (b : ℝ := Real.sqrt (1 / 10)) 
  (c : ℝ := Real.log (11 / 10)) : 
  b > c ∧ c > a := 
by
  sorry

end NUMINAMATH_GPT_compare_abc_l826_82683


namespace NUMINAMATH_GPT_is_not_innovative_54_l826_82688

def is_innovative (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 0 < b ∧ b < a ∧ n = a^2 - b^2

theorem is_not_innovative_54 : ¬ is_innovative 54 :=
sorry

end NUMINAMATH_GPT_is_not_innovative_54_l826_82688


namespace NUMINAMATH_GPT_inequality_on_abc_l826_82613

theorem inequality_on_abc (α β γ : ℝ) (h : α^2 + β^2 + γ^2 = 1) :
  -1/2 ≤ α * β + β * γ + γ * α ∧ α * β + β * γ + γ * α ≤ 1 :=
by {
  sorry -- Proof to be added
}

end NUMINAMATH_GPT_inequality_on_abc_l826_82613


namespace NUMINAMATH_GPT_equation_of_perpendicular_line_through_point_l826_82633

theorem equation_of_perpendicular_line_through_point :
  ∃ (a : ℝ) (b : ℝ) (c : ℝ), (a = 3) ∧ (b = 1) ∧ (x - 2 * y - 3 = 0 → y = (-(1/2)) * x + 3/2) ∧ (2 * a + b - 7 = 0) := sorry

end NUMINAMATH_GPT_equation_of_perpendicular_line_through_point_l826_82633


namespace NUMINAMATH_GPT_inequality_proof_l826_82684

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (h : x^2 + y^2 + z^2 = 2 * (x * y + y * z + z * x)) :
  (x + y + z) / 3 ≥ (2 * x * y * z)^(1/3 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l826_82684


namespace NUMINAMATH_GPT_volume_increase_factor_l826_82673

variable (π : ℝ) (r h : ℝ)

def original_volume : ℝ := π * r^2 * h

def new_volume : ℝ := π * (2 * r)^2 * (3 * h)

theorem volume_increase_factor : new_volume π r h = 12 * original_volume π r h :=
by
  -- Here we would include the proof that new_volume = 12 * original_volume
  sorry

end NUMINAMATH_GPT_volume_increase_factor_l826_82673


namespace NUMINAMATH_GPT_number_of_different_ways_is_18_l826_82603

-- Define the problem conditions
def number_of_ways_to_place_balls : ℕ :=
  let total_balls := 9
  let boxes := 3
  -- Placeholder function to compute the requirement
  -- The actual function would involve combinatorial logic
  -- Let us define it as an axiom for now.
  sorry

-- The theorem to be proven
theorem number_of_different_ways_is_18 :
  number_of_ways_to_place_balls = 18 :=
sorry

end NUMINAMATH_GPT_number_of_different_ways_is_18_l826_82603


namespace NUMINAMATH_GPT_distance_from_origin_to_line_l826_82680

theorem distance_from_origin_to_line : 
  let a := 1
  let b := 2
  let c := -5
  let x0 := 0
  let y0 := 0
  let distance := (|a * x0 + b * y0 + c|) / (Real.sqrt (a^2 + b^2))
  distance = Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_origin_to_line_l826_82680


namespace NUMINAMATH_GPT_find_number_of_terms_l826_82624

theorem find_number_of_terms (n : ℕ) (a : ℕ → ℚ) (S : ℕ → ℚ) :
  (∀ n, a n = (2^n - 1) / (2^n)) → S n = 321 / 64 → n = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_terms_l826_82624


namespace NUMINAMATH_GPT_Q1_Intersection_Q1_Union_Q2_l826_82631

namespace Example

def A : Set ℝ := {x | x ≤ -1 ∨ x ≥ 5}

def B (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x ≤ a + 2}

-- Question 1: 
theorem Q1_Intersection (a : ℝ) (ha : a = -1) : 
  A ∩ B a = {x | -2 ≤ x ∧ x ≤ -1} :=
sorry

theorem Q1_Union (a : ℝ) (ha : a = -1) :
  A ∪ B a = {x | x ≤ 1 ∨ x ≥ 5} :=
sorry

-- Question 2:
theorem Q2 (a : ℝ) :
  (A ∩ B a = B a) ↔ (a ≤ -3 ∨ a > 2) :=
sorry

end Example

end NUMINAMATH_GPT_Q1_Intersection_Q1_Union_Q2_l826_82631


namespace NUMINAMATH_GPT_jim_anne_mary_paul_report_time_l826_82639

def typing_rate_jim := 1 / 12
def typing_rate_anne := 1 / 20
def combined_typing_rate := typing_rate_jim + typing_rate_anne
def typing_time := 1 / combined_typing_rate

def editing_rate_mary := 1 / 30
def editing_rate_paul := 1 / 10
def combined_editing_rate := editing_rate_mary + editing_rate_paul
def editing_time := 1 / combined_editing_rate

theorem jim_anne_mary_paul_report_time : 
  typing_time + editing_time = 15 := by
  sorry

end NUMINAMATH_GPT_jim_anne_mary_paul_report_time_l826_82639


namespace NUMINAMATH_GPT_g_is_odd_function_l826_82655

noncomputable def g (x : ℝ) := 5 / (3 * x^5 - 7 * x)

theorem g_is_odd_function : ∀ x : ℝ, g (-x) = -g x :=
by
  intro x
  unfold g
  sorry

end NUMINAMATH_GPT_g_is_odd_function_l826_82655


namespace NUMINAMATH_GPT_check_prime_large_number_l826_82614

def large_number := 23021^377 - 1

theorem check_prime_large_number : ¬ Prime large_number := by
  sorry

end NUMINAMATH_GPT_check_prime_large_number_l826_82614


namespace NUMINAMATH_GPT_expression_always_positive_l826_82645

theorem expression_always_positive (x : ℝ) : x^2 + |x| + 1 > 0 :=
by 
  sorry

end NUMINAMATH_GPT_expression_always_positive_l826_82645


namespace NUMINAMATH_GPT_remainder_when_4x_div_7_l826_82630

theorem remainder_when_4x_div_7 (x : ℤ) (h : x % 7 = 5) : (4 * x) % 7 = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_4x_div_7_l826_82630


namespace NUMINAMATH_GPT_find_m3_minus_2mn_plus_n3_l826_82627

theorem find_m3_minus_2mn_plus_n3 (m n : ℝ) (h1 : m^2 = n + 2) (h2 : n^2 = m + 2) (h3 : m ≠ n) : m^3 - 2 * m * n + n^3 = -2 := by
  sorry

end NUMINAMATH_GPT_find_m3_minus_2mn_plus_n3_l826_82627


namespace NUMINAMATH_GPT_fingers_game_conditions_l826_82681

noncomputable def minNForWinningSubset (N : ℕ) : Prop :=
  N ≥ 220

-- To state the probability condition, we need to express it in terms of actual probabilities
noncomputable def probLeaderWins (N : ℕ) : ℝ := 
  1 / N

noncomputable def leaderWinProbabilityTendsToZero : Prop :=
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, probLeaderWins n < ε

theorem fingers_game_conditions (N : ℕ) (probLeaderWins : ℕ → ℝ) :
  (minNForWinningSubset N) ∧ leaderWinProbabilityTendsToZero :=
by
  sorry

end NUMINAMATH_GPT_fingers_game_conditions_l826_82681


namespace NUMINAMATH_GPT_six_hundred_billion_in_scientific_notation_l826_82651

theorem six_hundred_billion_in_scientific_notation (billion : ℕ) (h_billion : billion = 10^9) : 
  600 * billion = 6 * 10^11 :=
by
  rw [h_billion]
  sorry

end NUMINAMATH_GPT_six_hundred_billion_in_scientific_notation_l826_82651


namespace NUMINAMATH_GPT_arcsin_one_half_eq_pi_six_l826_82698

theorem arcsin_one_half_eq_pi_six : Real.arcsin (1 / 2) = Real.pi / 6 := 
by
  sorry

end NUMINAMATH_GPT_arcsin_one_half_eq_pi_six_l826_82698


namespace NUMINAMATH_GPT_range_of_m_l826_82659

noncomputable def quadratic_inequality_solution_set_is_R (m : ℝ) : Prop :=
  ∀ x : ℝ, (m - 1) * x^2 + (m - 1) * x + 2 > 0

theorem range_of_m :
  { m : ℝ | quadratic_inequality_solution_set_is_R m } = { m : ℝ | 1 ≤ m ∧ m < 9 } :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l826_82659


namespace NUMINAMATH_GPT_math_problem_l826_82640

-- Conditions
variables {f g : ℝ → ℝ}
axiom f_zero : f 0 = 0
axiom inequality : ∀ x y : ℝ, g (x - y) ≥ f x * f y + g x * g y

-- Problem Statement
theorem math_problem : ∀ x : ℝ, f x ^ 2008 + g x ^ 2008 ≤ 1 :=
by
  sorry

end NUMINAMATH_GPT_math_problem_l826_82640


namespace NUMINAMATH_GPT_hydrochloric_acid_required_l826_82678

-- Define the quantities for the balanced reaction equation
def molesOfAgNO3 : ℕ := 2
def molesOfHNO3 : ℕ := 2
def molesOfHCl : ℕ := 2

-- Define the condition for the reaction (balances the equation)
def balanced_reaction (x y z w : ℕ) : Prop :=
  x = y ∧ x = z ∧ y = w

-- The goal is to prove that the number of moles of HCl needed is 2
theorem hydrochloric_acid_required :
  balanced_reaction molesOfAgNO3 molesOfHCl molesOfHNO3 2 →
  molesOfHCl = 2 :=
by sorry

end NUMINAMATH_GPT_hydrochloric_acid_required_l826_82678


namespace NUMINAMATH_GPT_expected_disease_count_l826_82612

/-- Define the probability of an American suffering from the disease. -/
def probability_of_disease := 1 / 3

/-- Define the sample size of Americans surveyed. -/
def sample_size := 450

/-- Calculate the expected number of individuals suffering from the disease in the sample. -/
noncomputable def expected_number := probability_of_disease * sample_size

/-- State the theorem: the expected number of individuals suffering from the disease is 150. -/
theorem expected_disease_count : expected_number = 150 :=
by
  -- Proof is required but skipped using sorry.
  sorry

end NUMINAMATH_GPT_expected_disease_count_l826_82612


namespace NUMINAMATH_GPT_polygon_expected_value_l826_82699

def polygon_expected_sides (area_square : ℝ) (flower_prob : ℝ) (area_flower : ℝ) (hex_sides : ℝ) (pent_sides : ℝ) : ℝ :=
  hex_sides * flower_prob + pent_sides * (area_square - flower_prob)

theorem polygon_expected_value :
  polygon_expected_sides 1 (π - 1) (π - 1) 6 5 = π + 4 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_polygon_expected_value_l826_82699


namespace NUMINAMATH_GPT_rect_plot_length_more_than_breadth_l826_82626

theorem rect_plot_length_more_than_breadth (b x : ℕ) (cost_per_m : ℚ)
  (length_eq : b + x = 56)
  (fencing_cost : (4 * b + 2 * x) * cost_per_m = 5300)
  (cost_rate : cost_per_m = 26.50) : x = 12 :=
by
  sorry

end NUMINAMATH_GPT_rect_plot_length_more_than_breadth_l826_82626


namespace NUMINAMATH_GPT_Mildred_final_oranges_l826_82670

def initial_oranges : ℕ := 215
def father_oranges : ℕ := 3 * initial_oranges
def total_after_father : ℕ := initial_oranges + father_oranges
def sister_takes_away : ℕ := 174
def after_sister : ℕ := total_after_father - sister_takes_away
def final_oranges : ℕ := 2 * after_sister

theorem Mildred_final_oranges : final_oranges = 1372 := by
  sorry

end NUMINAMATH_GPT_Mildred_final_oranges_l826_82670


namespace NUMINAMATH_GPT_jenny_spent_625_dollars_l826_82616

def adoption_fee := 50
def vet_visits_cost := 500
def monthly_food_cost := 25
def toys_cost := 200
def year_months := 12

def jenny_adoption_vet_share := (adoption_fee + vet_visits_cost) / 2
def jenny_food_share := (monthly_food_cost * year_months) / 2
def jenny_total_cost := jenny_adoption_vet_share + jenny_food_share + toys_cost

theorem jenny_spent_625_dollars :
  jenny_total_cost = 625 := by
  sorry

end NUMINAMATH_GPT_jenny_spent_625_dollars_l826_82616


namespace NUMINAMATH_GPT_garden_roller_length_l826_82695

/-- The length of a garden roller with diameter 1.4m,
covering 52.8m² in 6 revolutions, and using π = 22/7,
is 2 meters. -/
theorem garden_roller_length
  (diameter : ℝ)
  (total_area_covered : ℝ)
  (revolutions : ℕ)
  (approx_pi : ℝ)
  (circumference : ℝ := approx_pi * diameter)
  (area_per_revolution : ℝ := total_area_covered / (revolutions : ℝ))
  (length : ℝ := area_per_revolution / circumference) :
  diameter = 1.4 ∧ total_area_covered = 52.8 ∧ revolutions = 6 ∧ approx_pi = (22 / 7) → length = 2 :=
by
  sorry

end NUMINAMATH_GPT_garden_roller_length_l826_82695


namespace NUMINAMATH_GPT_find_C_l826_82620

theorem find_C (A B C : ℕ) :
  (8 + 5 + 6 + 3 + 2 + A + B) % 3 = 0 →
  (4 + 3 + 7 + 5 + A + B + C) % 3 = 0 →
  C = 2 :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_find_C_l826_82620


namespace NUMINAMATH_GPT_min_value_function_l826_82635

theorem min_value_function (x y: ℝ) (hx: x > 2) (hy: y > 2) : 
  (∃c: ℝ, c = (x^3/(y - 2) + y^3/(x - 2)) ∧ ∀x y: ℝ, x > 2 → y > 2 → (x^3/(y - 2) + y^3/(x - 2)) ≥ c) ∧ c = 96 :=
sorry

end NUMINAMATH_GPT_min_value_function_l826_82635


namespace NUMINAMATH_GPT_max_gcd_b_n_b_n_plus_1_max_possible_value_of_e_n_l826_82607

def b_n (n : ℕ) : ℤ := (10 ^ n - 9) / 3
def e_n (n : ℕ) : ℤ := Int.gcd (b_n n) (b_n (n + 1))

theorem max_gcd_b_n_b_n_plus_1 : ∀ n : ℕ, e_n n ≤ 3 :=
by
  -- Provide the proof here
  sorry

theorem max_possible_value_of_e_n : ∃ n : ℕ, e_n n = 3 :=
by
  -- Provide the proof here
  sorry

end NUMINAMATH_GPT_max_gcd_b_n_b_n_plus_1_max_possible_value_of_e_n_l826_82607


namespace NUMINAMATH_GPT_value_in_half_dollars_percentage_l826_82664

theorem value_in_half_dollars_percentage (n h q : ℕ) (hn : n = 75) (hh : h = 40) (hq : q = 30) : 
  (h * 50 : ℕ) / (n * 5 + h * 50 + q * 25 : ℕ) * 100 = 64 := by
  sorry

end NUMINAMATH_GPT_value_in_half_dollars_percentage_l826_82664


namespace NUMINAMATH_GPT_max_min_values_l826_82605

-- Define the function f(x) = x^2 - 2ax + 1
def f (x a : ℝ) : ℝ := x ^ 2 - 2 * a * x + 1

-- Define the interval [0, 2]
def interval : Set ℝ := Set.Icc 0 2

theorem max_min_values (a : ℝ) : 
  (a > 2 → (∀ x ∈ interval, f x a ≤ 1) ∧ (∃ x ∈ interval, f x a = 5 - 4 * a))
  ∧ (1 ≤ a ∧ a ≤ 2 → (∀ x ∈ interval, f x a ≤ 5 - 4 * a) ∧ (∃ x ∈ interval, f x a = -a^2 + 1))
  ∧ (0 ≤ a ∧ a < 1 → (∀ x ∈ interval, f x a ≤ 1) ∧ (∃ x ∈ interval, f x a = -a^2 + 1))
  ∧ (a < 0 → (∀ x ∈ interval, f x a ≤ 5 - 4 * a) ∧ (∃ x ∈ interval, f x a = 1)) := by
  sorry

end NUMINAMATH_GPT_max_min_values_l826_82605


namespace NUMINAMATH_GPT_max_height_of_ball_l826_82610

def h (t : ℝ) : ℝ := -20 * t^2 + 70 * t + 20

theorem max_height_of_ball : ∃ t₀, h t₀ = 81.25 ∧ ∀ t, h t ≤ 81.25 :=
by
  sorry

end NUMINAMATH_GPT_max_height_of_ball_l826_82610


namespace NUMINAMATH_GPT_roots_of_polynomial_l826_82682

theorem roots_of_polynomial :
  (∃ (r : List ℤ), r = [1, 3, 4] ∧ 
    (∀ x : ℤ, x ∈ r → x^3 - 8*x^2 + 19*x - 12 = 0)) ∧ 
  (∀ x, x^3 - 8*x^2 + 19*x - 12 = 0 → x ∈ [1, 3, 4]) := 
sorry

end NUMINAMATH_GPT_roots_of_polynomial_l826_82682


namespace NUMINAMATH_GPT_marbles_per_boy_l826_82606

theorem marbles_per_boy (boys marbles : ℕ) (h1 : boys = 5) (h2 : marbles = 35) : marbles / boys = 7 := by
  sorry

end NUMINAMATH_GPT_marbles_per_boy_l826_82606


namespace NUMINAMATH_GPT_bake_sale_total_money_l826_82672

def dozens_to_pieces (dozens : Nat) : Nat :=
  dozens * 12

def total_money_raised
  (betty_chocolate_chip_dozen : Nat)
  (betty_oatmeal_raisin_dozen : Nat)
  (betty_brownies_dozen : Nat)
  (paige_sugar_cookies_dozen : Nat)
  (paige_blondies_dozen : Nat)
  (paige_cream_cheese_brownies_dozen : Nat)
  (price_per_cookie : Rat)
  (price_per_brownie_blondie : Rat) : Rat :=
let betty_cookies := dozens_to_pieces betty_chocolate_chip_dozen + dozens_to_pieces betty_oatmeal_raisin_dozen
let paige_cookies := dozens_to_pieces paige_sugar_cookies_dozen
let total_cookies := betty_cookies + paige_cookies
let betty_brownies := dozens_to_pieces betty_brownies_dozen
let paige_brownies_blondies := dozens_to_pieces paige_blondies_dozen + dozens_to_pieces paige_cream_cheese_brownies_dozen
let total_brownies_blondies := betty_brownies + paige_brownies_blondies
(total_cookies * price_per_cookie) + (total_brownies_blondies * price_per_brownie_blondie)

theorem bake_sale_total_money :
  total_money_raised 4 6 2 6 3 5 1 2 = 432 :=
by
  sorry

end NUMINAMATH_GPT_bake_sale_total_money_l826_82672


namespace NUMINAMATH_GPT_probability_different_colors_l826_82663

/-- There are 5 blue chips and 3 yellow chips in a bag. One chip is drawn from the bag and placed
back into the bag. A second chip is then drawn. Prove that the probability of the two selected chips
being of different colors is 15/32. -/
theorem probability_different_colors : 
  let total_chips := 8
  let blue_chips := 5
  let yellow_chips := 3
  let prob_blue_then_yellow := (blue_chips/total_chips) * (yellow_chips/total_chips)
  let prob_yellow_then_blue := (yellow_chips/total_chips) * (blue_chips/total_chips)
  prob_blue_then_yellow + prob_yellow_then_blue = 15/32 := by
  sorry

end NUMINAMATH_GPT_probability_different_colors_l826_82663


namespace NUMINAMATH_GPT_fractional_part_of_students_who_walk_home_l826_82694

theorem fractional_part_of_students_who_walk_home 
  (students_by_bus : ℚ)
  (students_by_car : ℚ)
  (students_by_bike : ℚ)
  (students_by_skateboard : ℚ)
  (h_bus : students_by_bus = 1/3)
  (h_car : students_by_car = 1/5)
  (h_bike : students_by_bike = 1/8)
  (h_skateboard : students_by_skateboard = 1/15)
  : 1 - (students_by_bus + students_by_car + students_by_bike + students_by_skateboard) = 11/40 := 
by
  sorry

end NUMINAMATH_GPT_fractional_part_of_students_who_walk_home_l826_82694


namespace NUMINAMATH_GPT_no_such_integers_exist_l826_82667

theorem no_such_integers_exist : ¬ ∃ (n k : ℕ), n > 0 ∧ k > 0 ∧ (n ∣ (k ^ n - 1)) ∧ (n.gcd (k - 1) = 1) :=
by
  sorry

end NUMINAMATH_GPT_no_such_integers_exist_l826_82667


namespace NUMINAMATH_GPT_abc_value_l826_82632

theorem abc_value {a b c : ℂ} 
  (h1 : a * b + 5 * b + 20 = 0) 
  (h2 : b * c + 5 * c + 20 = 0) 
  (h3 : c * a + 5 * a + 20 = 0) : 
  a * b * c = 100 := 
by 
  sorry

end NUMINAMATH_GPT_abc_value_l826_82632


namespace NUMINAMATH_GPT_age_of_replaced_person_l826_82652

theorem age_of_replaced_person (avg_age x : ℕ) (h1 : 10 * avg_age - 10 * (avg_age - 3) = x - 18) : x = 48 := 
by
  -- The proof goes here, but we are omitting it as per instruction.
  sorry

end NUMINAMATH_GPT_age_of_replaced_person_l826_82652


namespace NUMINAMATH_GPT_cherry_ratio_l826_82628

theorem cherry_ratio (total_lollipops cherry_lollipops watermelon_lollipops sour_apple_lollipops grape_lollipops : ℕ) 
  (h_total : total_lollipops = 42) 
  (h_rest_equally_distributed : watermelon_lollipops = sour_apple_lollipops ∧ sour_apple_lollipops = grape_lollipops) 
  (h_grape : grape_lollipops = 7) 
  (h_total_sum : cherry_lollipops + watermelon_lollipops + sour_apple_lollipops + grape_lollipops = total_lollipops) : 
  cherry_lollipops = 21 ∧ (cherry_lollipops : ℚ) / total_lollipops = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_cherry_ratio_l826_82628


namespace NUMINAMATH_GPT_sum_of_squares_eq_power_l826_82601

theorem sum_of_squares_eq_power (n : ℕ) : ∃ x y z : ℕ, x^2 + y^2 = z^n :=
sorry

end NUMINAMATH_GPT_sum_of_squares_eq_power_l826_82601


namespace NUMINAMATH_GPT_find_number_l826_82689

theorem find_number (x : ℤ) (h : ((x * 2) - 37 + 25) / 8 = 5) : x = 26 :=
sorry  -- Proof placeholder

end NUMINAMATH_GPT_find_number_l826_82689


namespace NUMINAMATH_GPT_find_x_for_fx_neg_half_l826_82617

open Function 

theorem find_x_for_fx_neg_half (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_period : ∀ x : ℝ, f (x + 2) = -f x)
  (h_interval : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → f x = 1/2 * x) :
  {x : ℝ | f x = -1/2} = {x : ℝ | ∃ n : ℤ, x = 4 * n - 1} :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_fx_neg_half_l826_82617


namespace NUMINAMATH_GPT_find_fourth_student_number_l826_82618

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

end NUMINAMATH_GPT_find_fourth_student_number_l826_82618


namespace NUMINAMATH_GPT_binom_12_10_eq_66_l826_82658

theorem binom_12_10_eq_66 : Nat.choose 12 10 = 66 :=
by
  sorry

end NUMINAMATH_GPT_binom_12_10_eq_66_l826_82658
