import Mathlib

namespace NUMINAMATH_GPT_gcd_18222_24546_66364_eq_2_l1877_187747

/-- Definition of three integers a, b, c --/
def a : ℕ := 18222 
def b : ℕ := 24546
def c : ℕ := 66364

/-- Proof of the gcd of the three integers being 2 --/
theorem gcd_18222_24546_66364_eq_2 : Nat.gcd (Nat.gcd a b) c = 2 := by
  sorry

end NUMINAMATH_GPT_gcd_18222_24546_66364_eq_2_l1877_187747


namespace NUMINAMATH_GPT_quadratic_solution_l1877_187772

theorem quadratic_solution (a c: ℝ) (h1 : a + c = 7) (h2 : a < c) (h3 : 36 - 4 * a * c = 0) : 
  a = (7 - Real.sqrt 13) / 2 ∧ c = (7 + Real.sqrt 13) / 2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_l1877_187772


namespace NUMINAMATH_GPT_factorize_x_squared_minus_sixteen_l1877_187721

theorem factorize_x_squared_minus_sixteen (x : ℝ) : x^2 - 16 = (x + 4) * (x - 4) :=
by
  sorry

end NUMINAMATH_GPT_factorize_x_squared_minus_sixteen_l1877_187721


namespace NUMINAMATH_GPT_xiao_li_more_stable_l1877_187724

def average_xiao_li : ℝ := 95
def average_xiao_zhang : ℝ := 95

def variance_xiao_li : ℝ := 0.55
def variance_xiao_zhang : ℝ := 1.35

theorem xiao_li_more_stable : 
  variance_xiao_li < variance_xiao_zhang :=
by
  sorry

end NUMINAMATH_GPT_xiao_li_more_stable_l1877_187724


namespace NUMINAMATH_GPT_divide_square_into_smaller_squares_l1877_187752

-- Definition of the property P(n)
def P (n : ℕ) : Prop := ∃ (f : ℕ → ℕ), ∀ i, i < n → (f i > 0)

-- Proposition for the problem
theorem divide_square_into_smaller_squares (n : ℕ) (h : n > 5) : P n :=
sorry

end NUMINAMATH_GPT_divide_square_into_smaller_squares_l1877_187752


namespace NUMINAMATH_GPT_dispatch_3_male_2_female_dispatch_at_least_2_male_l1877_187732

-- Define the number of male and female drivers
def male_drivers : ℕ := 5
def female_drivers : ℕ := 4
def total_drivers_needed : ℕ := 5

-- Define the combination formula (binomial coefficient)
def combination (n k : ℕ) : ℕ := Nat.choose n k

-- First part of the problem
theorem dispatch_3_male_2_female : 
  combination male_drivers 3 * combination female_drivers 2 = 60 :=
by sorry

-- Second part of the problem
theorem dispatch_at_least_2_male : 
  combination male_drivers 2 * combination female_drivers 3 + 
  combination male_drivers 3 * combination female_drivers 2 + 
  combination male_drivers 4 * combination female_drivers 1 + 
  combination male_drivers 5 * combination female_drivers 0 = 121 :=
by sorry

end NUMINAMATH_GPT_dispatch_3_male_2_female_dispatch_at_least_2_male_l1877_187732


namespace NUMINAMATH_GPT_rational_sum_of_cubic_roots_inverse_l1877_187788

theorem rational_sum_of_cubic_roots_inverse 
  (p q r : ℚ) 
  (h1 : p ≠ 0) 
  (h2 : q ≠ 0) 
  (h3 : r ≠ 0) 
  (h4 : ∃ a b c : ℚ, a = (pq^2)^(1/3) ∧ b = (qr^2)^(1/3) ∧ c = (rp^2)^(1/3) ∧ a + b + c ≠ 0) 
  : ∃ s : ℚ, s = 1/((pq^2)^(1/3)) + 1/((qr^2)^(1/3)) + 1/((rp^2)^(1/3)) :=
sorry

end NUMINAMATH_GPT_rational_sum_of_cubic_roots_inverse_l1877_187788


namespace NUMINAMATH_GPT_range_of_a_l1877_187723

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem range_of_a (a : ℝ) (h0 : a > 0) (h1 : a ≠ 1) 
  (h2 : log_base a (a^2 + 1) < log_base a (2 * a))
  (h3 : log_base a (2 * a) < 0) : a ∈ Set.Ioo (0.5) 1 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1877_187723


namespace NUMINAMATH_GPT_books_read_in_eight_hours_l1877_187790

-- Definitions to set up the problem
def reading_speed : ℕ := 120
def book_length : ℕ := 360
def available_time : ℕ := 8

-- Theorem statement
theorem books_read_in_eight_hours : (available_time * reading_speed) / book_length = 2 := 
by
  sorry

end NUMINAMATH_GPT_books_read_in_eight_hours_l1877_187790


namespace NUMINAMATH_GPT_ordered_pairs_squares_diff_150_l1877_187777

theorem ordered_pairs_squares_diff_150 (m n : ℕ) (hm_pos : 0 < m) (hn_pos : 0 < n) (hmn : m ≥ n) (h_diff : m^2 - n^2 = 150) : false :=
by {
    sorry
}

end NUMINAMATH_GPT_ordered_pairs_squares_diff_150_l1877_187777


namespace NUMINAMATH_GPT_next_meeting_time_l1877_187793

noncomputable def perimeter (AB BC CD DA : ℝ) : ℝ :=
  AB + BC + CD + DA

theorem next_meeting_time 
  (AB BC CD AD : ℝ) 
  (v_human v_dog : ℝ) 
  (initial_meeting_time : ℝ) :
  AB = 100 → BC = 200 → CD = 100 → AD = 200 →
  initial_meeting_time = 2 →
  v_human + v_dog = 300 →
  ∃ next_time : ℝ, next_time = 14 := 
by
  sorry

end NUMINAMATH_GPT_next_meeting_time_l1877_187793


namespace NUMINAMATH_GPT_coin_toss_tails_count_l1877_187783

theorem coin_toss_tails_count (flips : ℕ) (frequency_heads : ℝ) (h_flips : flips = 20) (h_frequency_heads : frequency_heads = 0.45) : 
  (20 : ℝ) * (1 - 0.45) = 11 := 
by
  sorry

end NUMINAMATH_GPT_coin_toss_tails_count_l1877_187783


namespace NUMINAMATH_GPT_convert_point_cylindrical_to_rectangular_l1877_187716

noncomputable def cylindrical_to_rectangular_coordinates (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem convert_point_cylindrical_to_rectangular :
  cylindrical_to_rectangular_coordinates 6 (5 * Real.pi / 3) (-3) = (3, -3 * Real.sqrt 3, -3) :=
by
  sorry

end NUMINAMATH_GPT_convert_point_cylindrical_to_rectangular_l1877_187716


namespace NUMINAMATH_GPT_sports_club_members_l1877_187755

theorem sports_club_members (N B T : ℕ) (h_total : N = 30) (h_badminton : B = 18) (h_tennis : T = 19) (h_neither : N - (B + T - 9) = 2) : B + T - 9 = 28 :=
by
  sorry

end NUMINAMATH_GPT_sports_club_members_l1877_187755


namespace NUMINAMATH_GPT_total_profit_correct_l1877_187736

noncomputable def total_profit (a b c : ℕ) (c_share : ℕ) : ℕ :=
  let ratio := a + b + c
  let part_value := c_share / c
  ratio * part_value

theorem total_profit_correct (h_a : ℕ := 5000) (h_b : ℕ := 8000) (h_c : ℕ := 9000) (h_c_share : ℕ := 36000) :
  total_profit h_a h_b h_c h_c_share = 88000 :=
by
  sorry

end NUMINAMATH_GPT_total_profit_correct_l1877_187736


namespace NUMINAMATH_GPT_problem1_problem2_l1877_187761

variable {x : ℝ} (hx : x > 0)

theorem problem1 : (2 / (3 * x)) * Real.sqrt (9 * x^3) + 6 * Real.sqrt (x / 4) - 2 * x * Real.sqrt (1 / x) = 3 * Real.sqrt x := 
by sorry

theorem problem2 : (Real.sqrt 24 + Real.sqrt 6) / Real.sqrt 3 + (Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) = 3 * Real.sqrt 2 + 2 := 
by sorry

end NUMINAMATH_GPT_problem1_problem2_l1877_187761


namespace NUMINAMATH_GPT_sequence_property_implies_geometric_progression_l1877_187781

theorem sequence_property_implies_geometric_progression {p : ℝ} {a : ℕ → ℝ}
  (h_p : (2 / (Real.sqrt 5 + 1) ≤ p) ∧ (p < 1))
  (h_a : ∀ (e : ℕ → ℤ), (∀ n, (e n = 0) ∨ (e n = 1) ∨ (e n = -1)) →
    (∑' n, (e n) * (p ^ n)) = 0 → (∑' n, (e n) * (a n)) = 0) :
  ∃ c : ℝ, ∀ n, a n = c * (p ^ n) := by
  sorry

end NUMINAMATH_GPT_sequence_property_implies_geometric_progression_l1877_187781


namespace NUMINAMATH_GPT_abs_diff_roots_eq_3_l1877_187713

theorem abs_diff_roots_eq_3 : ∀ (r1 r2 : ℝ), (r1 ≠ r2) → (r1 + r2 = 7) → (r1 * r2 = 10) → |r1 - r2| = 3 :=
by
  intros r1 r2 hneq hsum hprod
  sorry

end NUMINAMATH_GPT_abs_diff_roots_eq_3_l1877_187713


namespace NUMINAMATH_GPT_i_pow_2016_eq_one_l1877_187791
open Complex

theorem i_pow_2016_eq_one : (Complex.I ^ 2016) = 1 := by
  have h : Complex.I ^ 4 = 1 :=
    by rw [Complex.I_pow_four]
  exact sorry

end NUMINAMATH_GPT_i_pow_2016_eq_one_l1877_187791


namespace NUMINAMATH_GPT_weight_of_B_l1877_187739

theorem weight_of_B (A B C : ℕ) (h1 : A + B + C = 90) (h2 : A + B = 50) (h3 : B + C = 56) : B = 16 := 
sorry

end NUMINAMATH_GPT_weight_of_B_l1877_187739


namespace NUMINAMATH_GPT_solve_quadratic_eq_l1877_187737

theorem solve_quadratic_eq (x : ℝ) :
  (x^2 + (x - 1) * (x + 3) = 3 * x + 5) ↔ (x = -2 ∨ x = 2) :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l1877_187737


namespace NUMINAMATH_GPT_inequality_AM_GM_HM_l1877_187745

variable {x y k : ℝ}

-- Define the problem conditions
def is_positive (a : ℝ) : Prop := a > 0
def is_unequal (a b : ℝ) : Prop := a ≠ b
def positive_constant_lessthan_two (c : ℝ) : Prop := c > 0 ∧ c < 2

-- State the theorem to be proven
theorem inequality_AM_GM_HM (h₁ : is_positive x) 
                             (h₂ : is_positive y) 
                             (h₃ : is_unequal x y) 
                             (h₄ : positive_constant_lessthan_two k) :
  ( ( ( (x + y) / 2 )^k > ( (x * y)^(1/2) )^k ) ∧ 
    ( ( (x * y)^(1/2) )^k > ( ( 2 * x * y ) / ( x + y ) )^k ) ) :=
by
  sorry

end NUMINAMATH_GPT_inequality_AM_GM_HM_l1877_187745


namespace NUMINAMATH_GPT_neg_number_is_A_l1877_187786

def A : ℤ := -(3 ^ 2)
def B : ℤ := (-3) ^ 2
def C : ℤ := abs (-3)
def D : ℤ := -(-3)

theorem neg_number_is_A : A < 0 := 
by sorry

end NUMINAMATH_GPT_neg_number_is_A_l1877_187786


namespace NUMINAMATH_GPT_team_t_speed_l1877_187740

theorem team_t_speed (v t : ℝ) (h1 : 300 = v * t) (h2 : 300 = (v + 5) * (t - 3)) : v = 20 :=
by 
  sorry

end NUMINAMATH_GPT_team_t_speed_l1877_187740


namespace NUMINAMATH_GPT_total_time_equiv_7_75_l1877_187762

def acclimation_period : ℝ := 1
def learning_basics : ℝ := 2
def research_time_without_sabbatical : ℝ := learning_basics + 0.75 * learning_basics
def sabbatical : ℝ := 0.5
def research_time_with_sabbatical : ℝ := research_time_without_sabbatical + sabbatical
def dissertation_without_conference : ℝ := 0.5 * acclimation_period
def conference : ℝ := 0.25
def dissertation_with_conference : ℝ := dissertation_without_conference + conference
def total_time : ℝ := acclimation_period + learning_basics + research_time_with_sabbatical + dissertation_with_conference

theorem total_time_equiv_7_75 : total_time = 7.75 := by
  sorry

end NUMINAMATH_GPT_total_time_equiv_7_75_l1877_187762


namespace NUMINAMATH_GPT_order_of_numbers_l1877_187710

variables (a b : ℚ)

theorem order_of_numbers (ha_pos : a > 0) (hb_neg : b < 0) (habs : |a| < |b|) :
  b < -a ∧ -a < a ∧ a < -b :=
by { sorry }

end NUMINAMATH_GPT_order_of_numbers_l1877_187710


namespace NUMINAMATH_GPT_problem_statement_l1877_187774

open Set

noncomputable def U : Set ℝ := univ
noncomputable def M : Set ℝ := { x : ℝ | abs x < 2 }
noncomputable def N : Set ℝ := { y : ℝ | ∃ x : ℝ, y = 2^x - 1 }

theorem problem_statement :
  compl M ∪ compl N = Iic (-1) ∪ Ici 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l1877_187774


namespace NUMINAMATH_GPT_prime_div_p_sq_minus_one_l1877_187741

theorem prime_div_p_sq_minus_one {p : ℕ} (hp : p ≥ 7) (hp_prime : Nat.Prime p) : 
  (p % 10 = 1 ∨ p % 10 = 9) → 40 ∣ (p^2 - 1) :=
sorry

end NUMINAMATH_GPT_prime_div_p_sq_minus_one_l1877_187741


namespace NUMINAMATH_GPT_relationship_between_x_and_z_l1877_187717

-- Definitions of the given conditions
variable {x y z : ℝ}

-- Statement of the theorem
theorem relationship_between_x_and_z (h1 : x = 1.027 * y) (h2 : y = 0.45 * z) : x = 0.46215 * z :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_x_and_z_l1877_187717


namespace NUMINAMATH_GPT_movie_watching_l1877_187779

theorem movie_watching :
  let total_duration := 120 
  let watched1 := 35
  let watched2 := 20
  let watched3 := 15
  let total_watched := watched1 + watched2 + watched3
  total_duration - total_watched = 50 :=
by
  sorry

end NUMINAMATH_GPT_movie_watching_l1877_187779


namespace NUMINAMATH_GPT_total_food_each_day_l1877_187769

-- Conditions
def num_dogs : ℕ := 2
def food_per_dog : ℝ := 0.125
def total_food : ℝ := num_dogs * food_per_dog

-- Proof statement
theorem total_food_each_day : total_food = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_total_food_each_day_l1877_187769


namespace NUMINAMATH_GPT_factorize1_factorize2_factorize3_l1877_187725

theorem factorize1 (x : ℝ) : x^3 + 6 * x^2 + 9 * x = x * (x + 3)^2 := 
  sorry

theorem factorize2 (x y : ℝ) : 16 * x^2 - 9 * y^2 = (4 * x - 3 * y) * (4 * x + 3 * y) := 
  sorry

theorem factorize3 (x y : ℝ) : (3 * x + y)^2 - (x - 3 * y) * (3 * x + y) = 2 * (3 * x + y) * (x + 2 * y) := 
  sorry

end NUMINAMATH_GPT_factorize1_factorize2_factorize3_l1877_187725


namespace NUMINAMATH_GPT_middle_angle_of_triangle_l1877_187759

theorem middle_angle_of_triangle (α β γ : ℝ) 
  (h1 : 0 < β) (h2 : β < 90) 
  (h3 : α ≤ β) (h4 : β ≤ γ) 
  (h5 : α + β + γ = 180) :
  True :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_middle_angle_of_triangle_l1877_187759


namespace NUMINAMATH_GPT_slope_of_BC_l1877_187756

theorem slope_of_BC
  (h₁ : ∀ x y : ℝ, (x^2 / 8) + (y^2 / 2) = 1)
  (h₂ : ∀ A : ℝ × ℝ, A = (2, 1))
  (h₃ : ∀ k₁ k₂ : ℝ, k₁ + k₂ = 0) :
  ∃ k : ℝ, k = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_slope_of_BC_l1877_187756


namespace NUMINAMATH_GPT_sphere_has_circular_views_l1877_187705

-- Define the geometric shapes
inductive Shape
| cuboid
| cylinder
| cone
| sphere

-- Define a function that describes the views of a shape
def views (s: Shape) : (String × String × String) :=
match s with
| Shape.cuboid   => ("Rectangle", "Rectangle", "Rectangle")
| Shape.cylinder => ("Rectangle", "Rectangle", "Circle")
| Shape.cone     => ("Isosceles Triangle", "Isosceles Triangle", "Circle")
| Shape.sphere   => ("Circle", "Circle", "Circle")

-- Define the property of having circular views in all perspectives
def has_circular_views (s: Shape) : Prop :=
views s = ("Circle", "Circle", "Circle")

-- The theorem to prove
theorem sphere_has_circular_views :
  ∀ (s : Shape), has_circular_views s ↔ s = Shape.sphere :=
by sorry

end NUMINAMATH_GPT_sphere_has_circular_views_l1877_187705


namespace NUMINAMATH_GPT_john_new_salary_after_raise_l1877_187773

theorem john_new_salary_after_raise (original_salary : ℝ) (percentage_increase : ℝ) (h1 : original_salary = 60) (h2 : percentage_increase = 0.8333333333333334) : 
  original_salary * (1 + percentage_increase) = 110 := 
sorry

end NUMINAMATH_GPT_john_new_salary_after_raise_l1877_187773


namespace NUMINAMATH_GPT_capacity_of_new_vessel_is_10_l1877_187738

-- Define the conditions
def first_vessel_capacity : ℕ := 2
def first_vessel_concentration : ℚ := 0.25
def second_vessel_capacity : ℕ := 6
def second_vessel_concentration : ℚ := 0.40
def total_liquid_combined : ℕ := 8
def new_mixture_concentration : ℚ := 0.29
def total_alcohol_content : ℚ := (first_vessel_capacity * first_vessel_concentration) + (second_vessel_capacity * second_vessel_concentration)
def desired_vessel_capacity : ℚ := total_alcohol_content / new_mixture_concentration

-- The theorem we want to prove
theorem capacity_of_new_vessel_is_10 : desired_vessel_capacity = 10 := by
  sorry

end NUMINAMATH_GPT_capacity_of_new_vessel_is_10_l1877_187738


namespace NUMINAMATH_GPT_rahim_pillows_l1877_187757

theorem rahim_pillows (x T : ℕ) (h1 : T = 5 * x) (h2 : (T + 10) / (x + 1) = 6) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_rahim_pillows_l1877_187757


namespace NUMINAMATH_GPT_stratified_sampling_number_of_boys_stratified_sampling_probability_of_boy_l1877_187728

theorem stratified_sampling_number_of_boys (total_students : Nat) (num_girls : Nat) (selected_students : Nat)
  (h1 : total_students = 125) (h2 : num_girls = 50) (h3 : selected_students = 25) :
  (total_students - num_girls) * selected_students / total_students = 15 :=
  sorry

theorem stratified_sampling_probability_of_boy (total_students : Nat) (selected_students : Nat)
  (h1 : total_students = 125) (h2 : selected_students = 25) :
  selected_students / total_students = 1 / 5 :=
  sorry

end NUMINAMATH_GPT_stratified_sampling_number_of_boys_stratified_sampling_probability_of_boy_l1877_187728


namespace NUMINAMATH_GPT_gross_profit_percentage_without_discount_l1877_187751

theorem gross_profit_percentage_without_discount (C P : ℝ)
  (discount : P * 0.9 = C * 1.2)
  (discount_profit : C * 0.2 = P * 0.9 - C) :
  (P - C) / C * 100 = 33.3 :=
by
  sorry

end NUMINAMATH_GPT_gross_profit_percentage_without_discount_l1877_187751


namespace NUMINAMATH_GPT_line_equation_l1877_187785

open Real

-- Define the points A, B, and C
def A : ℝ × ℝ := ⟨1, 4⟩
def B : ℝ × ℝ := ⟨3, 2⟩
def C : ℝ × ℝ := ⟨2, -1⟩

-- Definition for a line passing through point C
-- and having equal distance to points A and B
def is_line_equation (l : ℝ → ℝ → Prop) :=
  ∀ x y, (l x y ↔ (x + y - 1 = 0 ∨ x - 2 = 0))

-- Our main statement
theorem line_equation :
  ∃ l : ℝ → ℝ → Prop, is_line_equation l ∧ (l 2 (-1)) :=
by
  sorry  -- Proof goes here.

end NUMINAMATH_GPT_line_equation_l1877_187785


namespace NUMINAMATH_GPT_sean_has_45_whistles_l1877_187735

variable (Sean Charles : ℕ)

def sean_whistles (Charles : ℕ) : ℕ :=
  Charles + 32

theorem sean_has_45_whistles
    (Charles_whistles : Charles = 13) 
    (Sean_whistles_condition : Sean = sean_whistles Charles) :
    Sean = 45 := by
  sorry

end NUMINAMATH_GPT_sean_has_45_whistles_l1877_187735


namespace NUMINAMATH_GPT_find_height_of_cuboid_l1877_187715

-- Define the cuboid structure and its surface area formula
structure Cuboid where
  length : ℝ
  width : ℝ
  height : ℝ

def surface_area (c : Cuboid) : ℝ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

-- Given conditions
def given_cuboid : Cuboid := { length := 12, width := 14, height := 7 }
def given_surface_area : ℝ := 700

-- The theorem to prove
theorem find_height_of_cuboid :
  surface_area given_cuboid = given_surface_area :=
by
  sorry

end NUMINAMATH_GPT_find_height_of_cuboid_l1877_187715


namespace NUMINAMATH_GPT_clive_change_l1877_187746

theorem clive_change (total_money : ℝ) (num_olives_needed : ℕ) (olives_per_jar : ℕ) (cost_per_jar : ℝ)
  (h1 : total_money = 10)
  (h2 : num_olives_needed = 80)
  (h3 : olives_per_jar = 20)
  (h4 : cost_per_jar = 1.5) : total_money - (num_olives_needed / olives_per_jar) * cost_per_jar = 4 := by
  sorry

end NUMINAMATH_GPT_clive_change_l1877_187746


namespace NUMINAMATH_GPT_steps_to_Madison_eq_991_l1877_187775

variable (steps_down steps_to_Madison : ℕ)

def total_steps (steps_down steps_to_Madison : ℕ) : ℕ :=
  steps_down + steps_to_Madison

theorem steps_to_Madison_eq_991 (h1 : steps_down = 676) (h2 : steps_to_Madison = 315) :
  total_steps steps_down steps_to_Madison = 991 :=
by
  sorry

end NUMINAMATH_GPT_steps_to_Madison_eq_991_l1877_187775


namespace NUMINAMATH_GPT_parametric_circle_eqn_l1877_187798

variables (t x y : ℝ)

theorem parametric_circle_eqn (h1 : y = t * x) (h2 : x^2 + y^2 - 4 * y = 0) :
  x = 4 * t / (1 + t^2) ∧ y = 4 * t^2 / (1 + t^2) :=
by
  sorry

end NUMINAMATH_GPT_parametric_circle_eqn_l1877_187798


namespace NUMINAMATH_GPT_Merrill_and_Elliot_have_fewer_marbles_than_Selma_l1877_187797

variable (Merrill_marbles Elliot_marbles Selma_marbles total_marbles fewer_marbles : ℕ)

-- Conditions
def Merrill_has_30_marbles : Merrill_marbles = 30 := by sorry

def Elliot_has_half_of_Merrill's_marbles : Elliot_marbles = Merrill_marbles / 2 := by sorry

def Selma_has_50_marbles : Selma_marbles = 50 := by sorry

def Merrill_and_Elliot_together_total_marbles : total_marbles = Merrill_marbles + Elliot_marbles := by sorry

def number_of_fewer_marbles : fewer_marbles = Selma_marbles - total_marbles := by sorry

-- Goal
theorem Merrill_and_Elliot_have_fewer_marbles_than_Selma :
  fewer_marbles = 5 := by
  sorry

end NUMINAMATH_GPT_Merrill_and_Elliot_have_fewer_marbles_than_Selma_l1877_187797


namespace NUMINAMATH_GPT_matrix_vector_addition_l1877_187748

def A : Matrix (Fin 2) (Fin 2) ℤ := ![![4, -2], ![-5, 6]]
def v : Fin 2 → ℤ := ![5, -2]
def w : Fin 2 → ℤ := ![1, -1]

theorem matrix_vector_addition :
  (A.mulVec v + w) = ![25, -38] :=
by
  sorry

end NUMINAMATH_GPT_matrix_vector_addition_l1877_187748


namespace NUMINAMATH_GPT_friend_cutoff_fraction_l1877_187766

-- Definitions based on problem conditions
def biking_time : ℕ := 30
def bus_time : ℕ := biking_time + 10
def days_biking : ℕ := 1
def days_bus : ℕ := 3
def days_friend : ℕ := 1
def total_weekly_commuting_time : ℕ := 160

-- Lean theorem statement
theorem friend_cutoff_fraction (F : ℕ) (hF : days_biking * biking_time + days_bus * bus_time + days_friend * F = total_weekly_commuting_time) :
  (biking_time - F) / biking_time = 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_friend_cutoff_fraction_l1877_187766


namespace NUMINAMATH_GPT_fruit_mix_apples_count_l1877_187794

variable (a o b p : ℕ)

theorem fruit_mix_apples_count :
  a + o + b + p = 240 →
  o = 3 * a →
  b = 2 * o →
  p = 5 * b →
  a = 6 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_fruit_mix_apples_count_l1877_187794


namespace NUMINAMATH_GPT_find_a12_a14_l1877_187764

noncomputable def S (n : ℕ) (a_n : ℕ → ℝ) (b : ℝ) : ℝ := a_n n ^ 2 + b * n

noncomputable def is_arithmetic_sequence (a_n : ℕ → ℝ) :=
  ∃ (a1 : ℝ) (c : ℝ), ∀ n : ℕ, a_n n = a1 + (n - 1) * c

theorem find_a12_a14
  (a_n : ℕ → ℝ)
  (b : ℝ)
  (S : ℕ → ℝ)
  (h1 : ∀ n, S n = a_n n ^ 2 + b * n)
  (h2 : S 25 = 100)
  (h3 : is_arithmetic_sequence a_n) :
  a_n 12 + a_n 14 = 5 :=
sorry

end NUMINAMATH_GPT_find_a12_a14_l1877_187764


namespace NUMINAMATH_GPT_h_h_3_eq_3568_l1877_187765

def h (x : ℤ) : ℤ := 3 * x * x + 3 * x - 2

theorem h_h_3_eq_3568 : h (h 3) = 3568 := by
  sorry

end NUMINAMATH_GPT_h_h_3_eq_3568_l1877_187765


namespace NUMINAMATH_GPT_geometric_series_sum_l1877_187729

theorem geometric_series_sum :
  let a := (1 : ℝ) / 5
  let r := -(1 : ℝ) / 5
  let n := 5
  let S_n := (a * (1 - r ^ n)) / (1 - r)
  S_n = 521 / 3125 := by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1877_187729


namespace NUMINAMATH_GPT_minimize_distances_is_k5_l1877_187767

-- Define the coordinates of points A, B, and D
def A : ℝ × ℝ := (4, 3)
def B : ℝ × ℝ := (1, 2)
def D : ℝ × ℝ := (0, 5)

-- Define C as a point vertically below D, implying the x-coordinate is the same as that of D and y = k
def C (k : ℝ) : ℝ × ℝ := (0, k)

-- Prove that the value of k that minimizes the distances over AC and BC is k = 5
theorem minimize_distances_is_k5 : ∃ k : ℝ, (C k = (0, 5)) ∧ k = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_minimize_distances_is_k5_l1877_187767


namespace NUMINAMATH_GPT_solve_2x2_minus1_eq_3x_l1877_187750
noncomputable def solve_quadratic (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let sqrt_discriminant := Real.sqrt discriminant
  let root1 := (-b + sqrt_discriminant) / (2 * a)
  let root2 := (-b - sqrt_discriminant) / (2 * a)
  (root1, root2)

theorem solve_2x2_minus1_eq_3x :
  solve_quadratic 2 (-3) (-1) = ( (3 + Real.sqrt 17) / 4, (3 - Real.sqrt 17) / 4 ) :=
by
  let roots := solve_quadratic 2 (-3) (-1)
  have : roots = ( (3 + Real.sqrt 17) / 4, (3 - Real.sqrt 17) / 4) := by sorry
  exact this

end NUMINAMATH_GPT_solve_2x2_minus1_eq_3x_l1877_187750


namespace NUMINAMATH_GPT_non_shaded_perimeter_6_l1877_187719

theorem non_shaded_perimeter_6 
  (area_shaded : ℝ) (area_large_rect : ℝ) (area_extension : ℝ) (total_area : ℝ)
  (non_shaded_area : ℝ) (perimeter : ℝ) :
  area_shaded = 104 → 
  area_large_rect = 12 * 8 → 
  area_extension = 5 * 2 → 
  total_area = area_large_rect + area_extension → 
  non_shaded_area = total_area - area_shaded → 
  non_shaded_area = 2 → 
  perimeter = 2 * (2 + 1) → 
  perimeter = 6 := 
by 
  sorry

end NUMINAMATH_GPT_non_shaded_perimeter_6_l1877_187719


namespace NUMINAMATH_GPT_correct_factorization_l1877_187727

theorem correct_factorization (a x m : ℝ) :
  (ax^2 - a = a * (x^2 - 1)) ∨
  (m^3 + m = m * (m^2 + 1)) ∨
  (x^2 + 2*x - 3 = x*(x+2) - 3) ∨
  (x^2 + 2*x - 3 = (x-3)*(x+1)) :=
by sorry

end NUMINAMATH_GPT_correct_factorization_l1877_187727


namespace NUMINAMATH_GPT_total_cost_proof_l1877_187706

def sandwich_cost : ℝ := 2.49
def soda_cost : ℝ := 1.87
def num_sandwiches : ℕ := 2
def num_sodas : ℕ := 4
def total_cost : ℝ := 12.46

theorem total_cost_proof : (num_sandwiches * sandwich_cost + num_sodas * soda_cost) = total_cost :=
by
  sorry

end NUMINAMATH_GPT_total_cost_proof_l1877_187706


namespace NUMINAMATH_GPT_units_digit_17_pow_17_l1877_187789

theorem units_digit_17_pow_17 : (17^17 % 10) = 7 := by
  sorry

end NUMINAMATH_GPT_units_digit_17_pow_17_l1877_187789


namespace NUMINAMATH_GPT_additional_toothpicks_needed_l1877_187753

def three_step_toothpicks := 18
def four_step_toothpicks := 26

theorem additional_toothpicks_needed : 
  (∃ (f : ℕ → ℕ), f 3 = three_step_toothpicks ∧ f 4 = four_step_toothpicks ∧ (f 6 - f 4) = 22) :=
by {
  -- Assume f is a function that gives the number of toothpicks for a n-step staircase
  sorry
}

end NUMINAMATH_GPT_additional_toothpicks_needed_l1877_187753


namespace NUMINAMATH_GPT_max_value_5x_minus_25x_l1877_187787

open Real

theorem max_value_5x_minus_25x : 
  ∃ x : ℝ, ∀ y : ℝ, (y = 5^x) → (y - y^2) ≤ 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_max_value_5x_minus_25x_l1877_187787


namespace NUMINAMATH_GPT_field_area_is_13_point854_hectares_l1877_187749

noncomputable def area_of_field_in_hectares (cost_fencing: ℝ) (rate_per_meter: ℝ): ℝ :=
  let length_of_fence := cost_fencing / rate_per_meter
  let radius := length_of_fence / (2 * Real.pi)
  let area_in_square_meters := Real.pi * (radius * radius)
  area_in_square_meters / 10000

theorem field_area_is_13_point854_hectares :
  area_of_field_in_hectares 6202.75 4.70 = 13.854 :=
by
  sorry

end NUMINAMATH_GPT_field_area_is_13_point854_hectares_l1877_187749


namespace NUMINAMATH_GPT_island_perimeter_l1877_187707

-- Defining the properties of the island
def width : ℕ := 4
def length : ℕ := 7

-- The main theorem stating the condition to be proved
theorem island_perimeter : 2 * (length + width) = 22 := by
  sorry

end NUMINAMATH_GPT_island_perimeter_l1877_187707


namespace NUMINAMATH_GPT_cars_given_by_mum_and_dad_l1877_187760

-- Define the conditions given in the problem
def initial_cars : ℕ := 150
def final_cars : ℕ := 196
def cars_by_auntie : ℕ := 6
def cars_more_than_uncle : ℕ := 1
def cars_given_by_family (uncle : ℕ) (grandpa : ℕ) (auntie : ℕ) : ℕ :=
  uncle + grandpa + auntie

-- Prove the required statement
theorem cars_given_by_mum_and_dad :
  ∃ (uncle grandpa : ℕ), grandpa = 2 * uncle ∧ auntie = uncle + cars_more_than_uncle ∧ 
    auntie = cars_by_auntie ∧
    final_cars - initial_cars - cars_given_by_family uncle grandpa auntie = 25 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_cars_given_by_mum_and_dad_l1877_187760


namespace NUMINAMATH_GPT_exponential_decreasing_l1877_187702

theorem exponential_decreasing (a : ℝ) (h : ∀ x y : ℝ, x < y → (a+1)^x > (a+1)^y) : -1 < a ∧ a < 0 :=
sorry

end NUMINAMATH_GPT_exponential_decreasing_l1877_187702


namespace NUMINAMATH_GPT_orthocenter_ABC_l1877_187718

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

def A : Point2D := ⟨5, -1⟩
def B : Point2D := ⟨4, -8⟩
def C : Point2D := ⟨-4, -4⟩

def isOrthocenter (H : Point2D) (A B C : Point2D) : Prop := sorry  -- Define this properly according to the geometric properties in actual formalization.

theorem orthocenter_ABC : ∃ H : Point2D, isOrthocenter H A B C ∧ H = ⟨3, -5⟩ := 
by 
  sorry  -- Proof omitted

end NUMINAMATH_GPT_orthocenter_ABC_l1877_187718


namespace NUMINAMATH_GPT_contrapositive_correct_l1877_187708

-- Conditions and the proposition
def prop1 (a : ℝ) : Prop := a = -1 → a^2 = 1

-- The contrapositive of the proposition
def contrapositive (a : ℝ) : Prop := a^2 ≠ 1 → a ≠ -1

-- The proof problem statement
theorem contrapositive_correct (a : ℝ) : prop1 a ↔ contrapositive a :=
by sorry

end NUMINAMATH_GPT_contrapositive_correct_l1877_187708


namespace NUMINAMATH_GPT_ratio_problem_l1877_187780

variable {a b c d : ℚ}

theorem ratio_problem (h₁ : a / b = 5) (h₂ : c / b = 3) (h₃ : c / d = 2) :
  d / a = 3 / 10 :=
sorry

end NUMINAMATH_GPT_ratio_problem_l1877_187780


namespace NUMINAMATH_GPT_exponent_power_identity_l1877_187704

theorem exponent_power_identity (y : ℕ) (h : 3^y = 81) : 3^(y+3) = 2187 :=
sorry

end NUMINAMATH_GPT_exponent_power_identity_l1877_187704


namespace NUMINAMATH_GPT_reduced_price_l1877_187795

noncomputable def reduced_price_per_dozen (P : ℝ) : ℝ := 12 * (P / 2)

theorem reduced_price (X P : ℝ) (h1 : X * P = 50) (h2 : (X + 50) * (P / 2) = 50) : reduced_price_per_dozen P = 6 :=
sorry

end NUMINAMATH_GPT_reduced_price_l1877_187795


namespace NUMINAMATH_GPT_fraction_meaningful_l1877_187730

theorem fraction_meaningful (x : ℝ) : (x - 1 ≠ 0) ↔ (∃ (y : ℝ), y = 3 / (x - 1)) :=
by sorry

end NUMINAMATH_GPT_fraction_meaningful_l1877_187730


namespace NUMINAMATH_GPT_complex_quadrant_l1877_187771

theorem complex_quadrant (θ : ℝ) (hθ : θ ∈ Set.Ioo (3/4 * Real.pi) (5/4 * Real.pi)) :
  let z := Complex.mk (Real.cos θ + Real.sin θ) (Real.sin θ - Real.cos θ)
  z.re < 0 ∧ z.im > 0 :=
by
  sorry

end NUMINAMATH_GPT_complex_quadrant_l1877_187771


namespace NUMINAMATH_GPT_total_profit_l1877_187733

-- Define the variables for the subscriptions and profits
variables {A B C : ℕ} -- Subscription amounts
variables {profit : ℕ} -- Total profit

-- Given conditions
def conditions (A B C : ℕ) (profit : ℕ) :=
  50000 = A + B + C ∧
  A = B + 4000 ∧
  B = C + 5000 ∧
  A * profit = 29400 * 50000

-- Statement of the theorem
theorem total_profit (A B C : ℕ) (profit : ℕ) (h : conditions A B C profit) :
  profit = 70000 :=
sorry

end NUMINAMATH_GPT_total_profit_l1877_187733


namespace NUMINAMATH_GPT_red_car_speed_l1877_187711

/-- Dale owns 4 sports cars where:
1. The red car can travel at twice the speed of the green car.
2. The green car can travel at 8 times the speed of the blue car.
3. The blue car can travel at a speed of 80 miles per hour.
We need to determine the speed of the red car. --/
theorem red_car_speed (r g b: ℕ) (h1: r = 2 * g) (h2: g = 8 * b) (h3: b = 80) : 
  r = 1280 :=
by
  sorry

end NUMINAMATH_GPT_red_car_speed_l1877_187711


namespace NUMINAMATH_GPT_geometric_sum_n_eq_4_l1877_187799

theorem geometric_sum_n_eq_4 :
  ∃ n : ℕ, (n = 4) ∧ 
  ((1 : ℚ) * (1 - (1 / 4 : ℚ) ^ n) / (1 - (1 / 4 : ℚ)) = (85 / 64 : ℚ)) :=
by
  use 4
  simp
  sorry

end NUMINAMATH_GPT_geometric_sum_n_eq_4_l1877_187799


namespace NUMINAMATH_GPT_sapling_height_relationship_l1877_187768

-- Definition to state the conditions
def initial_height : ℕ := 100
def growth_per_year : ℕ := 50
def height_after_years (years : ℕ) : ℕ := initial_height + growth_per_year * years

-- The theorem statement that should be proved
theorem sapling_height_relationship (x : ℕ) : height_after_years x = 50 * x + 100 := 
by
  sorry

end NUMINAMATH_GPT_sapling_height_relationship_l1877_187768


namespace NUMINAMATH_GPT_find_salary_J_l1877_187734

variables {J F M A May : ℝ}
variables (h1 : (J + F + M + A) / 4 = 8000)
variables (h2 : (F + M + A + May) / 4 = 8200)
variables (h3 : May = 6500)

theorem find_salary_J : J = 5700 :=
by
  sorry

end NUMINAMATH_GPT_find_salary_J_l1877_187734


namespace NUMINAMATH_GPT_count_solutions_absolute_value_l1877_187782

theorem count_solutions_absolute_value (x : ℤ) : 
  (|4 * x + 2| ≤ 10) ↔ (x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2) :=
by sorry

end NUMINAMATH_GPT_count_solutions_absolute_value_l1877_187782


namespace NUMINAMATH_GPT_cos_function_max_value_l1877_187731

theorem cos_function_max_value (k : ℤ) : (2 * Real.cos (2 * k * Real.pi) - 1) = 1 :=
by
  -- Proof not included
  sorry

end NUMINAMATH_GPT_cos_function_max_value_l1877_187731


namespace NUMINAMATH_GPT_three_a_in_S_implies_a_in_S_l1877_187763

def S := {n | ∃ x y : ℤ, n = x^2 + 2 * y^2}

theorem three_a_in_S_implies_a_in_S (a : ℤ) (h : 3 * a ∈ S) : a ∈ S := 
sorry

end NUMINAMATH_GPT_three_a_in_S_implies_a_in_S_l1877_187763


namespace NUMINAMATH_GPT_value_of_c_l1877_187796

theorem value_of_c (c : ℝ) : (∀ x : ℝ, (-x^2 + c * x + 10 < 0) ↔ (x < 2 ∨ x > 8)) → c = 10 :=
by
  sorry

end NUMINAMATH_GPT_value_of_c_l1877_187796


namespace NUMINAMATH_GPT_biggest_number_in_ratio_l1877_187778

theorem biggest_number_in_ratio (x : ℕ) (h_sum : 2 * x + 3 * x + 4 * x + 5 * x = 1344) : 5 * x = 480 := 
by
  sorry

end NUMINAMATH_GPT_biggest_number_in_ratio_l1877_187778


namespace NUMINAMATH_GPT_log_addition_closed_l1877_187722

def is_log_of_nat (n : ℝ) : Prop := ∃ k : ℕ, k > 0 ∧ n = Real.log k

theorem log_addition_closed (a b : ℝ) (ha : is_log_of_nat a) (hb : is_log_of_nat b) : is_log_of_nat (a + b) :=
by
  sorry

end NUMINAMATH_GPT_log_addition_closed_l1877_187722


namespace NUMINAMATH_GPT_opposite_of_point_one_l1877_187754

theorem opposite_of_point_one : ∃ x : ℝ, 0.1 + x = 0 ∧ x = -0.1 :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_point_one_l1877_187754


namespace NUMINAMATH_GPT_floor_cube_neg_seven_four_l1877_187744

theorem floor_cube_neg_seven_four :
  (Int.floor ((-7 / 4 : ℚ) ^ 3) = -6) :=
by
  sorry

end NUMINAMATH_GPT_floor_cube_neg_seven_four_l1877_187744


namespace NUMINAMATH_GPT_arithmetic_mean_of_fractions_l1877_187714

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 5
  let b := (5 : ℚ) / 7
  (a + b) / 2 = (23 : ℚ) / 35 := 
by 
  sorry 

end NUMINAMATH_GPT_arithmetic_mean_of_fractions_l1877_187714


namespace NUMINAMATH_GPT_third_player_matches_l1877_187758

theorem third_player_matches (first_player second_player third_player : ℕ) (h1 : first_player = 10) (h2 : second_player = 21) :
  third_player = 11 :=
by
  sorry

end NUMINAMATH_GPT_third_player_matches_l1877_187758


namespace NUMINAMATH_GPT_total_seedlings_transferred_l1877_187701

-- Define the number of seedlings planted on the first day
def seedlings_day_1 : ℕ := 200

-- Define the number of seedlings planted on the second day
def seedlings_day_2 : ℕ := 2 * seedlings_day_1

-- Define the total number of seedlings planted on both days
def total_seedlings : ℕ := seedlings_day_1 + seedlings_day_2

-- The theorem statement
theorem total_seedlings_transferred : total_seedlings = 600 := by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_total_seedlings_transferred_l1877_187701


namespace NUMINAMATH_GPT_possible_values_of_ab_plus_ac_plus_bc_l1877_187770

theorem possible_values_of_ab_plus_ac_plus_bc (a b c : ℝ) (h : a + b + 2 * c = 0) :
  ∃ x ∈ Set.Iic 0, ab + ac + bc = x :=
sorry

end NUMINAMATH_GPT_possible_values_of_ab_plus_ac_plus_bc_l1877_187770


namespace NUMINAMATH_GPT_tim_sarah_age_ratio_l1877_187709

theorem tim_sarah_age_ratio :
  ∀ (x : ℕ), ∃ (t s : ℕ),
    t = 23 ∧ s = 11 ∧
    (23 + x) * 2 = (11 + x) * 3 → x = 13 :=
by
  sorry

end NUMINAMATH_GPT_tim_sarah_age_ratio_l1877_187709


namespace NUMINAMATH_GPT_a_eq_3_suff_not_nec_l1877_187742

theorem a_eq_3_suff_not_nec (a : ℝ) : (a = 3 → a^2 = 9) ∧ (a^2 = 9 → ∃ b : ℝ, b = a ∧ (b = 3 ∨ b = -3)) :=
by
  sorry

end NUMINAMATH_GPT_a_eq_3_suff_not_nec_l1877_187742


namespace NUMINAMATH_GPT_find_cost_of_article_l1877_187792

-- Define the given conditions and the corresponding proof statement.
theorem find_cost_of_article
  (tax_rate : ℝ) (selling_price1 : ℝ)
  (selling_price2 : ℝ) (profit_increase_rate : ℝ)
  (cost : ℝ) : tax_rate = 0.05 →
              selling_price1 = 360 →
              selling_price2 = 340 →
              profit_increase_rate = 0.05 →
              (selling_price1 / (1 + tax_rate) - cost = 1.05 * (selling_price2 / (1 + tax_rate) - cost)) →
              cost = 57.13 :=
by sorry

end NUMINAMATH_GPT_find_cost_of_article_l1877_187792


namespace NUMINAMATH_GPT_OReilly_triple_8_49_x_l1877_187743

def is_OReilly_triple (a b x : ℕ) : Prop :=
  (a : ℝ)^(1/3) + (b : ℝ)^(1/2) = x

theorem OReilly_triple_8_49_x (x : ℕ) (h : is_OReilly_triple 8 49 x) : x = 9 := by
  sorry

end NUMINAMATH_GPT_OReilly_triple_8_49_x_l1877_187743


namespace NUMINAMATH_GPT_g_sum_zero_l1877_187776

def g (x : ℝ) : ℝ := x^2 - 2013 * x

theorem g_sum_zero (a b : ℝ) (h₁ : g a = g b) (h₂ : a ≠ b) : g (a + b) = 0 :=
sorry

end NUMINAMATH_GPT_g_sum_zero_l1877_187776


namespace NUMINAMATH_GPT_square_side_length_l1877_187700

theorem square_side_length(area_sq_cm : ℕ) (h : area_sq_cm = 361) : ∃ side_length : ℕ, side_length ^ 2 = area_sq_cm ∧ side_length = 19 := 
by 
  use 19
  sorry

end NUMINAMATH_GPT_square_side_length_l1877_187700


namespace NUMINAMATH_GPT_isosceles_trapezoid_area_l1877_187703

-- Defining the problem characteristics
variables {a b c d h θ : ℝ}

-- The area formula for an isosceles trapezoid with given bases and height
theorem isosceles_trapezoid_area (h : ℝ) (c d : ℝ) : 
  (1 / 2) * (c + d) * h = (1 / 2) * (c + d) * h := 
sorry

end NUMINAMATH_GPT_isosceles_trapezoid_area_l1877_187703


namespace NUMINAMATH_GPT_students_selecting_water_l1877_187784

-- Definitions of percentages and given values.
def p : ℝ := 0.7
def q : ℝ := 0.1
def n : ℕ := 140

-- The Lean statement to prove the number of students who selected water.
theorem students_selecting_water (p_eq : p = 0.7) (q_eq : q = 0.1) (n_eq : n = 140) :
  ∃ w : ℕ, w = (q / p) * n ∧ w = 20 :=
by sorry

end NUMINAMATH_GPT_students_selecting_water_l1877_187784


namespace NUMINAMATH_GPT_positive_rationals_in_S_l1877_187726

variable (S : Set ℚ)

-- Conditions
axiom closed_under_addition (a b : ℚ) (ha : a ∈ S) (hb : b ∈ S) : a + b ∈ S
axiom closed_under_multiplication (a b : ℚ) (ha : a ∈ S) (hb : b ∈ S) : a * b ∈ S
axiom zero_rule : ∀ r : ℚ, r ∈ S ∨ -r ∈ S ∨ r = 0

-- Prove that S is the set of positive rational numbers
theorem positive_rationals_in_S : S = {r : ℚ | 0 < r} :=
by
  sorry

end NUMINAMATH_GPT_positive_rationals_in_S_l1877_187726


namespace NUMINAMATH_GPT_binary_operation_result_l1877_187720

theorem binary_operation_result :
  let a := 0b1101
  let b := 0b111
  let c := 0b1010
  let d := 0b1001
  a + b - c + d = 0b10011 :=
by {
  sorry
}

end NUMINAMATH_GPT_binary_operation_result_l1877_187720


namespace NUMINAMATH_GPT_mary_total_zoom_time_l1877_187712

noncomputable def timeSpentDownloadingMac : ℝ := 10
noncomputable def timeSpentDownloadingWindows : ℝ := 3 * timeSpentDownloadingMac
noncomputable def audioGlitchesCount : ℝ := 2
noncomputable def audioGlitchDuration : ℝ := 4
noncomputable def totalAudioGlitchTime : ℝ := audioGlitchesCount * audioGlitchDuration
noncomputable def videoGlitchDuration : ℝ := 6
noncomputable def totalGlitchTime : ℝ := totalAudioGlitchTime + videoGlitchDuration
noncomputable def glitchFreeTalkingTime : ℝ := 2 * totalGlitchTime

theorem mary_total_zoom_time : 
  timeSpentDownloadingMac + timeSpentDownloadingWindows + totalGlitchTime + glitchFreeTalkingTime = 82 :=
by sorry

end NUMINAMATH_GPT_mary_total_zoom_time_l1877_187712
