import Mathlib

namespace NUMINAMATH_GPT_least_positive_integer_condition_l636_63677

theorem least_positive_integer_condition
  (a : ℤ) (ha1 : a % 4 = 1) (ha2 : a % 5 = 2) (ha3 : a % 6 = 3) :
  a > 0 → a = 57 :=
by
  intro ha_pos
  -- Proof omitted for brevity
  sorry

end NUMINAMATH_GPT_least_positive_integer_condition_l636_63677


namespace NUMINAMATH_GPT_pow_three_not_sum_of_two_squares_l636_63618

theorem pow_three_not_sum_of_two_squares (k : ℕ) (hk : 0 < k) : 
  ¬ ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^2 + y^2 = 3^k :=
by
  sorry

end NUMINAMATH_GPT_pow_three_not_sum_of_two_squares_l636_63618


namespace NUMINAMATH_GPT_measure_of_AED_l636_63664

-- Importing the necessary modules for handling angles and geometry
variables {A B C D E : Type}
noncomputable def angle (p q r : Type) : ℝ := sorry -- Definition to represent angles in general

-- Given conditions
variables
  (hD_on_AC : D ∈ line_segment A C)
  (hE_on_BC : E ∈ line_segment B C)
  (h_angle_ABD : angle A B D = 30)
  (h_angle_BAE : angle B A E = 60)
  (h_angle_CAE : angle C A E = 20)
  (h_angle_CBD : angle C B D = 30)

-- The goal to prove
theorem measure_of_AED :
  angle A E D = 20 :=
by
  -- Proof details will go here
  sorry

end NUMINAMATH_GPT_measure_of_AED_l636_63664


namespace NUMINAMATH_GPT_find_cos_alpha_l636_63617

theorem find_cos_alpha (α : ℝ) (h0 : 0 ≤ α ∧ α ≤ π / 2) (h1 : Real.sin (α - π / 6) = 3 / 5) : 
  Real.cos α = (4 * Real.sqrt 3 - 3) / 10 :=
sorry

end NUMINAMATH_GPT_find_cos_alpha_l636_63617


namespace NUMINAMATH_GPT_solve_for_x_and_y_l636_63687

theorem solve_for_x_and_y : 
  (∃ x y : ℝ, 0.65 * 900 = 0.40 * x ∧ 0.35 * 1200 = 0.25 * y) → 
  ∃ x y : ℝ, x + y = 3142.5 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_and_y_l636_63687


namespace NUMINAMATH_GPT_M_subset_N_l636_63656

-- Define M and N using the given conditions
def M : Set ℝ := {α | ∃ (k : ℤ), α = k * 90} ∪ {α | ∃ (k : ℤ), α = k * 180 + 45}
def N : Set ℝ := {α | ∃ (k : ℤ), α = k * 45}

-- Prove that M is a subset of N
theorem M_subset_N : M ⊆ N :=
by
  sorry

end NUMINAMATH_GPT_M_subset_N_l636_63656


namespace NUMINAMATH_GPT_range_of_a_l636_63613

def p (a : ℝ) : Prop := (a + 2) > 1
def q (a : ℝ) : Prop := (4 - 4 * a) ≥ 0
def prop_and (a : ℝ) : Prop := p a ∧ q a
def prop_or (a : ℝ) : Prop := p a ∨ q a
def valid_a (a : ℝ) : Prop := (a ∈ Set.Iic (-1)) ∨ (a ∈ Set.Ioi 1)

theorem range_of_a (a : ℝ) (h_and : ¬ prop_and a) (h_or : prop_or a) : valid_a a := 
sorry

end NUMINAMATH_GPT_range_of_a_l636_63613


namespace NUMINAMATH_GPT_complex_sum_to_zero_l636_63663

noncomputable def z : ℂ := sorry

theorem complex_sum_to_zero 
  (h₁ : z ^ 3 = 1) 
  (h₂ : z ≠ 1) : 
  z ^ 103 + z ^ 104 + z ^ 105 + z ^ 106 + z ^ 107 + z ^ 108 = 0 :=
sorry

end NUMINAMATH_GPT_complex_sum_to_zero_l636_63663


namespace NUMINAMATH_GPT_negate_exponential_inequality_l636_63632

theorem negate_exponential_inequality :
  ¬ (∀ x : ℝ, Real.exp x > x) ↔ ∃ x : ℝ, Real.exp x ≤ x :=
by
  sorry

end NUMINAMATH_GPT_negate_exponential_inequality_l636_63632


namespace NUMINAMATH_GPT_cos_double_angle_l636_63641

theorem cos_double_angle (x : ℝ) (h : Real.sin (x + Real.pi / 2) = 1 / 3) : Real.cos (2 * x) = -7 / 9 :=
sorry

end NUMINAMATH_GPT_cos_double_angle_l636_63641


namespace NUMINAMATH_GPT_calculate_s_at_2_l636_63690

-- Given definitions
def t (x : ℝ) : ℝ := 2 * x^2 - 5 * x + 1
def s (p : ℝ) : ℝ := p^3 - 4 * p^2 + p + 6

-- The target statement
theorem calculate_s_at_2 : s 2 = ((5 + Real.sqrt 33) / 4)^3 - 4 * ((5 + Real.sqrt 33) / 4)^2 + ((5 + Real.sqrt 33) / 4) + 6 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_s_at_2_l636_63690


namespace NUMINAMATH_GPT_triangle_ABC_properties_l636_63653

noncomputable def is_arithmetic_sequence (α β γ : ℝ) : Prop :=
γ - β = β - α

theorem triangle_ABC_properties
  (A B C a c : ℝ)
  (b : ℝ := Real.sqrt 3)
  (h1 : a * Real.cos C + c * Real.cos A = 2 * b * Real.cos B) :
  is_arithmetic_sequence A B C ∧
  ∃ (max_area : ℝ), max_area = (3 * Real.sqrt 3) / 4 := by sorry

end NUMINAMATH_GPT_triangle_ABC_properties_l636_63653


namespace NUMINAMATH_GPT_functional_equation_solution_l636_63654

noncomputable def function_nat_nat (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, f (x + y) = f x + f y

theorem functional_equation_solution :
  ∀ f : ℕ → ℕ, function_nat_nat f → ∃ a : ℕ, ∀ x : ℕ, f x = a * x :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l636_63654


namespace NUMINAMATH_GPT_total_protest_days_l636_63691

-- Definitions for the problem conditions
def first_protest_days : ℕ := 4
def second_protest_days : ℕ := first_protest_days + (first_protest_days / 4)

-- The proof statement
theorem total_protest_days : first_protest_days + second_protest_days = 9 := sorry

end NUMINAMATH_GPT_total_protest_days_l636_63691


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_l636_63619

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a > 2 → a^2 > 2 * a) ∧ ¬(a^2 > 2 * a → a > 2) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_l636_63619


namespace NUMINAMATH_GPT_biology_marks_l636_63675

theorem biology_marks 
  (e : ℕ) (m : ℕ) (p : ℕ) (c : ℕ) (a : ℕ) (n : ℕ) (b : ℕ) 
  (h_e : e = 96) (h_m : m = 95) (h_p : p = 82) (h_c : c = 97) (h_a : a = 93) (h_n : n = 5)
  (h_total : e + m + p + c + b = a * n) :
  b = 95 :=
by 
  sorry

end NUMINAMATH_GPT_biology_marks_l636_63675


namespace NUMINAMATH_GPT_unused_square_is_teal_l636_63621

-- Define the set of colors
inductive Color
| Cyan
| Magenta
| Lime
| Purple
| Teal
| Silver
| Violet

open Color

-- Define the condition that Lime is opposite Purple in the cube
def opposite (a b : Color) : Prop :=
  (a = Lime ∧ b = Purple) ∨ (a = Purple ∧ b = Lime)

-- Define the problem: seven squares are colored and one color remains unused.
def seven_squares_set (hinge : List Color) : Prop :=
  hinge.length = 6 ∧ 
  opposite Lime Purple ∧
  Color.Cyan ∈ hinge ∧
  Color.Magenta ∈ hinge ∧ 
  Color.Lime ∈ hinge ∧ 
  Color.Purple ∈ hinge ∧ 
  Color.Teal ∈ hinge ∧ 
  Color.Silver ∈ hinge ∧ 
  Color.Violet ∈ hinge

theorem unused_square_is_teal :
  ∃ hinge : List Color, seven_squares_set hinge ∧ ¬ (Teal ∈ hinge) := 
by sorry

end NUMINAMATH_GPT_unused_square_is_teal_l636_63621


namespace NUMINAMATH_GPT_find_sachins_age_l636_63682

variable (S R : ℕ)

theorem find_sachins_age (h1 : R = S + 8) (h2 : S * 9 = R * 7) : S = 28 := by
  sorry

end NUMINAMATH_GPT_find_sachins_age_l636_63682


namespace NUMINAMATH_GPT_rhombus_diagonal_BD_equation_rhombus_diagonal_AD_equation_l636_63688

theorem rhombus_diagonal_BD_equation (A C : ℝ × ℝ) (AB_eq : ∀ x y : ℝ, 3 * x - y + 2 = 0) : 
  A = (0, 2) ∧ C = (4, 6) → ∃ k b : ℝ, k = 1 ∧ b = 6 ∧ ∀ x y : ℝ, x + y - 6 = 0 := by
  sorry

theorem rhombus_diagonal_AD_equation (A C : ℝ × ℝ) (AB_eq BD_eq : ∀ x y : ℝ, 3 * x - y + 2 = 0 ∧ x + y - 6 = 0) : 
  A = (0, 2) ∧ C = (4, 6) → ∃ k b : ℝ, k = 3 ∧ b = 14 ∧ ∀ x y : ℝ, x - 3 * y + 14 = 0 := by
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_BD_equation_rhombus_diagonal_AD_equation_l636_63688


namespace NUMINAMATH_GPT_workshop_output_comparison_l636_63648

theorem workshop_output_comparison (a x : ℝ)
  (h1 : ∀n:ℕ, n ≥ 0 → (1 + n * a) = (1 + x)^n) :
  (1 + 3 * a) > (1 + x)^3 := sorry

end NUMINAMATH_GPT_workshop_output_comparison_l636_63648


namespace NUMINAMATH_GPT_fraction_of_smaller_jar_l636_63609

theorem fraction_of_smaller_jar (S L : ℝ) (W : ℝ) (F : ℝ) 
  (h1 : W = F * S) 
  (h2 : W = 1/2 * L) 
  (h3 : 2 * W = 2/3 * L) 
  (h4 : S = 2/3 * L) :
  F = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_smaller_jar_l636_63609


namespace NUMINAMATH_GPT_number_of_guests_l636_63639

-- Defining the given conditions
def appetizers_per_guest : ℕ := 6
def deviled_eggs_dozen : ℕ := 3
def pigs_in_blanket_dozen : ℕ := 2
def kebabs_dozen : ℕ := 2
def additional_appetizers_dozen : ℕ := 8

-- The main theorem to prove the number of guests Patsy is expecting
theorem number_of_guests : 
  (deviled_eggs_dozen + pigs_in_blanket_dozen + kebabs_dozen + additional_appetizers_dozen) * 12 / appetizers_per_guest = 30 :=
by
  sorry

end NUMINAMATH_GPT_number_of_guests_l636_63639


namespace NUMINAMATH_GPT_percentage_decrease_equivalent_l636_63667

theorem percentage_decrease_equivalent :
  ∀ (P D : ℝ), 
    (D = 10) →
    ((1.25 * P) - (D / 100) * (1.25 * P) = 1.125 * P) :=
by
  intros P D h
  rw [h]
  sorry

end NUMINAMATH_GPT_percentage_decrease_equivalent_l636_63667


namespace NUMINAMATH_GPT_problem_statement_l636_63620

theorem problem_statement (a b : ℝ) (h1 : a - b > 0) (h2 : a + b < 0) : b < 0 ∧ |b| > |a| :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l636_63620


namespace NUMINAMATH_GPT_sibling_of_5_over_7_child_unique_parent_one_over_2008_descendent_of_one_l636_63637

-- Part (a):
def sibling_of_frac (x : ℚ) : Prop :=
  x = 5/7

theorem sibling_of_5_over_7 : ∃ (y : ℚ), sibling_of_frac (y / (y + 1)) ∧ y + 1 = 7/2 :=
  sorry

-- Part (b):
def child (x y : ℚ) : Prop :=
  y = x + 1 ∨ y = x / (x + 1)

theorem child_unique_parent (x y z : ℚ) (hx : 0 < x) (hz : 0 < z) (hyx : child x y) (hyz : child z y) : x = z :=
  sorry

-- Part (c):
def descendent (x y : ℚ) : Prop :=
  ∃ n : ℕ, y = 1 / (x + n)

theorem one_over_2008_descendent_of_one : descendent 1 (1 / 2008) :=
  sorry

end NUMINAMATH_GPT_sibling_of_5_over_7_child_unique_parent_one_over_2008_descendent_of_one_l636_63637


namespace NUMINAMATH_GPT_johns_average_speed_last_hour_l636_63614

theorem johns_average_speed_last_hour
  (total_distance : ℕ)
  (total_time : ℕ)
  (speed_first_hour : ℕ)
  (speed_second_hour : ℕ)
  (distance_last_hour : ℕ)
  (average_speed_last_hour : ℕ)
  (H1 : total_distance = 120)
  (H2 : total_time = 3)
  (H3 : speed_first_hour = 40)
  (H4 : speed_second_hour = 50)
  (H5 : distance_last_hour = total_distance - (speed_first_hour + speed_second_hour))
  (H6 : average_speed_last_hour = distance_last_hour / 1)
  : average_speed_last_hour = 30 := 
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_johns_average_speed_last_hour_l636_63614


namespace NUMINAMATH_GPT_inequality_solution_set_l636_63686

noncomputable def increasing_function (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

theorem inequality_solution_set (f : ℝ → ℝ)
  (h_increasing : increasing_function f)
  (h_A : f 0 = -2)
  (h_B : f 3 = 2) :
  {x : ℝ | |f (x+1)| ≥ 2} = {x | x ≤ -1} ∪ {x | x ≥ 2} :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l636_63686


namespace NUMINAMATH_GPT_conic_section_is_ellipse_l636_63626

theorem conic_section_is_ellipse :
  ∀ x y : ℝ, 4 * x^2 + y^2 - 12 * x - 2 * y + 4 = 0 →
  ∃ a b h k : ℝ, a > 0 ∧ b > 0 ∧ (a * (x - h)^2 + b * (y - k)^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_conic_section_is_ellipse_l636_63626


namespace NUMINAMATH_GPT_smallest_k_for_a_n_digital_l636_63685

theorem smallest_k_for_a_n_digital (a n : ℕ) (h : 10^2013 ≤ a^n ∧ a^n < 10^2014) : 
  ∀ k : ℕ, (∀ b : ℕ, 10^(k-1) ≤ b → b < 10^k → (¬(10^2013 ≤ b^n ∧ b^n < 10^2014))) ↔ k = 2014 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_k_for_a_n_digital_l636_63685


namespace NUMINAMATH_GPT_area_of_rectangle_l636_63661

-- Define the lengths in meters
def length : ℝ := 1.2
def width : ℝ := 0.5

-- Define the function to calculate the area of a rectangle
def area (l w : ℝ) : ℝ := l * w

-- Prove that the area of the rectangle with given length and width is 0.6 square meters
theorem area_of_rectangle :
  area length width = 0.6 := by
  -- This is just the statement. We omit the proof with sorry.
  sorry

end NUMINAMATH_GPT_area_of_rectangle_l636_63661


namespace NUMINAMATH_GPT_value_of_M_l636_63665

theorem value_of_M (x y z M : ℚ) : 
  (x + y + z = 48) ∧ (x - 5 = M) ∧ (y + 9 = M) ∧ (z / 5 = M) → M = 52 / 7 :=
by
  sorry

end NUMINAMATH_GPT_value_of_M_l636_63665


namespace NUMINAMATH_GPT_javier_time_outlining_l636_63666

variable (O : ℕ)
variable (W : ℕ := O + 28)
variable (P : ℕ := (O + 28) / 2)
variable (total_time : ℕ := O + W + P)

theorem javier_time_outlining
  (h1 : total_time = 117)
  (h2 : W = O + 28)
  (h3 : P = (O + 28) / 2)
  : O = 30 := by 
  sorry

end NUMINAMATH_GPT_javier_time_outlining_l636_63666


namespace NUMINAMATH_GPT_position_of_point_l636_63698

theorem position_of_point (a b : ℝ) (h_tangent: (a ≠ 0 ∨ b ≠ 0) ∧ (a^2 + b^2 = 1)) : a^2 + b^2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_position_of_point_l636_63698


namespace NUMINAMATH_GPT_odd_function_negative_value_l636_63678

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_value {f : ℝ → ℝ} (h_odd : is_odd_function f) :
  (∀ x, 0 < x → f x = x^2 - x - 1) → (∀ x, x < 0 → f x = -x^2 - x + 1) :=
by
  sorry

end NUMINAMATH_GPT_odd_function_negative_value_l636_63678


namespace NUMINAMATH_GPT_sample_size_l636_63628

variable (x n : ℕ)

-- Conditions as definitions
def staff_ratio : Prop := 15 * x + 3 * x + 2 * x = 20 * x
def sales_staff : Prop := 30 / n = 15 / 20

-- Main statement to prove
theorem sample_size (h1: staff_ratio x) (h2: sales_staff n) : n = 40 := by
  sorry

end NUMINAMATH_GPT_sample_size_l636_63628


namespace NUMINAMATH_GPT_triangle_area_l636_63649

noncomputable def a := 5
noncomputable def b := 4
noncomputable def s := (13 : ℝ) / 2 -- semi-perimeter
noncomputable def area := Real.sqrt (s * (s - a) * (s - b) * (s - b))

theorem triangle_area :
  a + 2 * b = 13 →
  (a > 0) → (b > 0) →
  (a < 2 * b) →
  (a + b > b) → 
  (a + b > b) →
  area = Real.sqrt 61.09375 :=
by
  intros h1 h2 h3 h4 h5 h6
  -- We assume validity of these conditions and skip the proof for brevity.
  sorry

end NUMINAMATH_GPT_triangle_area_l636_63649


namespace NUMINAMATH_GPT_part_i_l636_63606

theorem part_i (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : (a^2 + b^2 + c^2)^2 > 2 * (a^4 + b^4 + c^4)) : 
  a + b > c ∧ a + c > b ∧ b + c > a := sorry

end NUMINAMATH_GPT_part_i_l636_63606


namespace NUMINAMATH_GPT_howard_rewards_l636_63627

theorem howard_rewards (initial_bowls : ℕ) (customers : ℕ) (customers_bought_20 : ℕ) 
                       (bowls_remaining : ℕ) (rewards_per_bowl : ℕ) :
  initial_bowls = 70 → 
  customers = 20 → 
  customers_bought_20 = 10 → 
  bowls_remaining = 30 → 
  rewards_per_bowl = 2 →
  ∀ (bowls_bought_per_customer : ℕ), bowls_bought_per_customer = 20 → 
  2 * (200 / 20) = 10 := 
by 
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_howard_rewards_l636_63627


namespace NUMINAMATH_GPT_triangle_medians_and_area_l636_63634

/-- Given a triangle with side lengths 13, 14, and 15,
    prove that the sum of the squares of the lengths of the medians is 385
    and the area of the triangle is 84. -/
theorem triangle_medians_and_area :
  let a := 13
  let b := 14
  let c := 15
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let m_a := Real.sqrt (2 * b^2 + 2 * c^2 - a^2) / 2
  let m_b := Real.sqrt (2 * c^2 + 2 * a^2 - b^2) / 2
  let m_c := Real.sqrt (2 * a^2 + 2 * b^2 - c^2) / 2
  m_a^2 + m_b^2 + m_c^2 = 385 ∧ area = 84 := sorry

end NUMINAMATH_GPT_triangle_medians_and_area_l636_63634


namespace NUMINAMATH_GPT_fishing_tomorrow_l636_63631

-- Conditions
def every_day_fishers : Nat := 7
def every_other_day_fishers : Nat := 8
def every_three_days_fishers : Nat := 3
def yesterday_fishers : Nat := 12
def today_fishers : Nat := 10

-- Determine the number who will fish tomorrow
def fishers_tomorrow : Nat :=
  let every_day_tomorrow := every_day_fishers
  let every_three_day_tomorrow := every_three_days_fishers
  let every_other_day_yesterday := yesterday_fishers - every_day_fishers
  let every_other_day_tomorrow := every_other_day_fishers - every_other_day_yesterday
  every_day_tomorrow + every_three_day_tomorrow + every_other_day_tomorrow

theorem fishing_tomorrow : fishers_tomorrow = 15 :=
  by
    -- skipping the actual proof with sorry
    sorry

end NUMINAMATH_GPT_fishing_tomorrow_l636_63631


namespace NUMINAMATH_GPT_solve_inequality_l636_63671

-- Define the domain and inequality conditions
def inequality_condition (x : ℝ) : Prop := (1 / (x - 1)) > 1
def domain_condition (x : ℝ) : Prop := x ≠ 1

-- State the theorem to be proved.
theorem solve_inequality (x : ℝ) : domain_condition x → inequality_condition x → 1 < x ∧ x < 2 :=
by
  intros h_domain h_ineq
  sorry

end NUMINAMATH_GPT_solve_inequality_l636_63671


namespace NUMINAMATH_GPT_amelia_distance_l636_63674

theorem amelia_distance (total_distance amelia_monday_distance amelia_tuesday_distance : ℕ) 
  (h1 : total_distance = 8205) 
  (h2 : amelia_monday_distance = 907) 
  (h3 : amelia_tuesday_distance = 582) : 
  total_distance - (amelia_monday_distance + amelia_tuesday_distance) = 6716 := 
by 
  sorry

end NUMINAMATH_GPT_amelia_distance_l636_63674


namespace NUMINAMATH_GPT_man_l636_63692

theorem man's_speed_downstream (v : ℝ) (speed_of_stream : ℝ) (speed_upstream : ℝ) : 
  speed_upstream = v - speed_of_stream ∧ speed_of_stream = 1.5 ∧ speed_upstream = 8 → v + speed_of_stream = 11 :=
by
  sorry

end NUMINAMATH_GPT_man_l636_63692


namespace NUMINAMATH_GPT_cristina_nicky_head_start_l636_63625

theorem cristina_nicky_head_start (s_c s_n : ℕ) (t d : ℕ) 
  (h1 : s_c = 5) 
  (h2 : s_n = 3) 
  (h3 : t = 30)
  (h4 : d = s_n * t):
  d = 90 := 
by
  sorry

end NUMINAMATH_GPT_cristina_nicky_head_start_l636_63625


namespace NUMINAMATH_GPT_sequence_count_l636_63679

theorem sequence_count :
  ∃ (a : ℕ → ℕ), 
    a 10 = 3 * a 1 ∧ 
    a 2 + a 8 = 2 * a 5 ∧ 
    (∀ i, 1 ≤ i ∧ i ≤ 9 → a (i + 1) = 1 + a i ∨ a (i + 1) = 2 + a i) ∧ 
    (∃ n, n = 80) :=
sorry

end NUMINAMATH_GPT_sequence_count_l636_63679


namespace NUMINAMATH_GPT_find_exponent_l636_63668

theorem find_exponent (m x y a : ℝ) (h : y = m * x ^ a) (hx : x = 1 / 4) (hy : y = 1 / 2) : a = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_exponent_l636_63668


namespace NUMINAMATH_GPT_math_problem_l636_63695

theorem math_problem (x t : ℝ) (h1 : 6 * x + t = 4 * x - 9) (h2 : t = 7) : x + 4 = -4 := by
  sorry

end NUMINAMATH_GPT_math_problem_l636_63695


namespace NUMINAMATH_GPT_speed_of_stream_l636_63635

-- Definitions based on the conditions
def upstream_speed (c v : ℝ) : Prop := c - v = 4
def downstream_speed (c v : ℝ) : Prop := c + v = 12

-- Main theorem to prove
theorem speed_of_stream (c v : ℝ) (h1 : upstream_speed c v) (h2 : downstream_speed c v) : v = 4 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_stream_l636_63635


namespace NUMINAMATH_GPT_waiter_slices_l636_63660

theorem waiter_slices (total_slices : ℕ) (buzz_ratio waiter_ratio : ℕ)
  (h_total_slices : total_slices = 78)
  (h_ratios : buzz_ratio = 5 ∧ waiter_ratio = 8) :
  20 < (waiter_ratio * (total_slices / (buzz_ratio + waiter_ratio))) →
  28 = waiter_ratio * (total_slices / (buzz_ratio + waiter_ratio)) - 20 :=
by
  sorry

end NUMINAMATH_GPT_waiter_slices_l636_63660


namespace NUMINAMATH_GPT_simplify_fraction_l636_63681

theorem simplify_fraction : (2 / (1 - (2 / 3))) = 6 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l636_63681


namespace NUMINAMATH_GPT_find_G_8_l636_63608

noncomputable def G : Polynomial ℝ := sorry 

variable (x : ℝ)

theorem find_G_8 :
  G.eval 4 = 8 ∧ 
  (∀ x, (G.eval (2*x)) / (G.eval (x+2)) = 4 - (16 * x) / (x^2 + 2 * x + 2)) →
  G.eval 8 = 40 := 
sorry

end NUMINAMATH_GPT_find_G_8_l636_63608


namespace NUMINAMATH_GPT_right_triangle_area_l636_63650

/-- Given a right triangle with one leg of length 3 and the hypotenuse of length 5,
    the area of the triangle is 6. -/
theorem right_triangle_area (a b c : ℝ) (h₁ : a = 3) (h₂ : c = 5) (h₃ : c^2 = a^2 + b^2) :
  (1 / 2) * a * b = 6 := 
sorry

end NUMINAMATH_GPT_right_triangle_area_l636_63650


namespace NUMINAMATH_GPT_inequality_solution_l636_63644

theorem inequality_solution (m : ℝ) : 
  (∀ x : ℝ, 2 * x + 7 > 3 * x + 2 ∧ 2 * x - 2 < 2 * m → x < 5) → m ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l636_63644


namespace NUMINAMATH_GPT_unique_positive_integers_exists_l636_63612

theorem unique_positive_integers_exists (p : ℕ) (hp_prime : Nat.Prime p) (hp_odd : p % 2 = 1) : 
  ∃! m n : ℕ, m^2 = n * (n + p) ∧ m = (p^2 - 1) / 2 ∧ n = (p - 1)^2 / 4 := by
  sorry

end NUMINAMATH_GPT_unique_positive_integers_exists_l636_63612


namespace NUMINAMATH_GPT_cuboid_surface_area_l636_63607

-- Definition of the problem with given conditions and the statement we need to prove.
theorem cuboid_surface_area (h l w: ℝ) (H1: 4 * (2 * h) + 4 * (2 * h) + 4 * h = 100)
                            (H2: l = 2 * h)
                            (H3: w = 2 * h) :
                            (2 * (l * w + l * h + w * h) = 400) :=
by
  sorry

end NUMINAMATH_GPT_cuboid_surface_area_l636_63607


namespace NUMINAMATH_GPT_find_g_l636_63645

open Function

def linear_system (a b c d e f g : ℚ) :=
  a + b + c + d + e = 1 ∧
  b + c + d + e + f = 2 ∧
  c + d + e + f + g = 3 ∧
  d + e + f + g + a = 4 ∧
  e + f + g + a + b = 5 ∧
  f + g + a + b + c = 6 ∧
  g + a + b + c + d = 7

theorem find_g (a b c d e f g : ℚ) (h : linear_system a b c d e f g) : 
  g = 13 / 3 :=
sorry

end NUMINAMATH_GPT_find_g_l636_63645


namespace NUMINAMATH_GPT_polynomial_root_exists_l636_63689

theorem polynomial_root_exists
  (P : ℝ → ℝ)
  (a1 a2 a3 b1 b2 b3 : ℝ)
  (h_nonzero : a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0)
  (h_eq : ∀ x : ℝ, P (a1 * x + b1) + P (a2 * x + b2) = P (a3 * x + b3)) :
  ∃ r : ℝ, P r = 0 :=
sorry

end NUMINAMATH_GPT_polynomial_root_exists_l636_63689


namespace NUMINAMATH_GPT_term_10_of_sequence_l636_63640

theorem term_10_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n, S n = n * (2 * n + 1)) →
  (∀ n, a n = S n - S (n - 1)) →
  a 10 = 39 :=
by
  intros hS ha
  sorry

end NUMINAMATH_GPT_term_10_of_sequence_l636_63640


namespace NUMINAMATH_GPT_side_length_of_square_l636_63683

noncomputable def area_of_circle : ℝ := 3848.4510006474966
noncomputable def pi : ℝ := Real.pi

theorem side_length_of_square :
  ∃ s : ℝ, (∃ r : ℝ, area_of_circle = pi * r * r ∧ 2 * r = s) ∧ s = 70 := 
by
  sorry

end NUMINAMATH_GPT_side_length_of_square_l636_63683


namespace NUMINAMATH_GPT_sum_of_cubes_l636_63651

theorem sum_of_cubes (x y : ℝ) (hx : x + y = 10) (hxy : x * y = 12) : x^3 + y^3 = 640 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_l636_63651


namespace NUMINAMATH_GPT_interest_rate_l636_63629

-- Definitions based on given conditions
def SumLent : ℝ := 1500
def InterestTime : ℝ := 4
def InterestAmount : ℝ := SumLent - 1260

-- Main theorem to prove the interest rate r is 4%
theorem interest_rate (r : ℝ) : (InterestAmount = SumLent * r / 100 * InterestTime) → r = 4 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_l636_63629


namespace NUMINAMATH_GPT_divisor_is_3_l636_63602

theorem divisor_is_3 (divisor quotient remainder : ℕ) (h_dividend : 22 = (divisor * quotient) + remainder) 
  (h_quotient : quotient = 7) (h_remainder : remainder = 1) : divisor = 3 :=
by
  sorry

end NUMINAMATH_GPT_divisor_is_3_l636_63602


namespace NUMINAMATH_GPT_range_of_f_find_a_l636_63659

-- Define the function f
def f (a x : ℝ) : ℝ := -a^2 * x - 2 * a * x + 1

-- Define the proposition for part (1)
theorem range_of_f (a : ℝ) (h : a > 1) : Set.range (f a) = Set.Iio 1 := sorry

-- Define the proposition for part (2)
theorem find_a (a : ℝ) (h : a > 1) (min_value : ∀ x, x ∈ Set.Icc (-2 : ℝ) 1 → f a x ≥ -7) : a = 2 :=
sorry

end NUMINAMATH_GPT_range_of_f_find_a_l636_63659


namespace NUMINAMATH_GPT_range_of_a_l636_63652

-- Definitions from conditions 
def p (x : ℝ) : Prop := |x + 1| > 2
def q (x a : ℝ) : Prop := x > a

-- The Lean statement for the problem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 1 → x ≤ a) → a ≥ 1 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l636_63652


namespace NUMINAMATH_GPT_carol_savings_l636_63697

theorem carol_savings (S : ℝ) (h1 : ∀ t : ℝ, t = S - (2/3) * S) (h2 : S + (S - (2/3) * S) = 1/4) : S = 3/16 :=
by {
  sorry
}

end NUMINAMATH_GPT_carol_savings_l636_63697


namespace NUMINAMATH_GPT_range_of_PF1_minus_PF2_l636_63672

noncomputable def ellipse_property (x0 : ℝ) (h1 : 0 < x0) (h2 : x0 < Real.sqrt 5) : Prop :=
  ∃ f : ℝ, f = (2 * Real.sqrt 5 / 5) * x0 ∧ f > 0 ∧ f < 2

theorem range_of_PF1_minus_PF2 (x0 : ℝ) (h1 : 0 < x0) (h2 : x0 < Real.sqrt 5) : 
  ellipse_property x0 h1 h2 := by
  sorry

end NUMINAMATH_GPT_range_of_PF1_minus_PF2_l636_63672


namespace NUMINAMATH_GPT_cube_volume_from_surface_area_l636_63643

theorem cube_volume_from_surface_area (s : ℝ) (h : 6 * s^2 = 54) : s^3 = 27 :=
sorry

end NUMINAMATH_GPT_cube_volume_from_surface_area_l636_63643


namespace NUMINAMATH_GPT_percent_only_cats_l636_63696

def total_students := 500
def total_cats := 120
def total_dogs := 200
def both_cats_and_dogs := 40
def only_cats := total_cats - both_cats_and_dogs

theorem percent_only_cats:
  (only_cats : ℕ) / (total_students : ℕ) * 100 = 16 := 
by 
  sorry

end NUMINAMATH_GPT_percent_only_cats_l636_63696


namespace NUMINAMATH_GPT_probability_of_stopping_after_2nd_shot_l636_63604

-- Definitions based on the conditions
def shootingProbability : ℚ := 2 / 3

noncomputable def scoring (n : ℕ) : ℕ := 12 - n

def stopShootingProbabilityAfterNthShot (n : ℕ) (probOfShooting : ℚ) : ℚ :=
  if n = 2 then (1 / 3) * (2 / 3) * sorry -- Note: Here, filling in the remaining calculation steps according to problem logic.
  else sorry -- placeholder for other cases

theorem probability_of_stopping_after_2nd_shot :
  stopShootingProbabilityAfterNthShot 2 shootingProbability = 8 / 729 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_stopping_after_2nd_shot_l636_63604


namespace NUMINAMATH_GPT_animal_costs_l636_63623

theorem animal_costs (S K L : ℕ) (h1 : K = 4 * S) (h2 : L = 4 * K) (h3 : S + 2 * K + L = 200) :
  S = 8 ∧ K = 32 ∧ L = 128 :=
by
  sorry

end NUMINAMATH_GPT_animal_costs_l636_63623


namespace NUMINAMATH_GPT_deposit_amount_l636_63694

theorem deposit_amount (P : ℝ) (h₀ : 0.1 * P + 720 = P) : 0.1 * P = 80 :=
by
  sorry

end NUMINAMATH_GPT_deposit_amount_l636_63694


namespace NUMINAMATH_GPT_value_of_A_l636_63699

theorem value_of_A (A B C D : ℕ) (h1 : A * B = 60) (h2 : C * D = 60) (h3 : A - B = C + D) (h4 : A ≠ B) (h5 : A ≠ C) (h6 : A ≠ D) (h7 : B ≠ C) (h8 : B ≠ D) (h9 : C ≠ D) : A = 20 :=
by sorry

end NUMINAMATH_GPT_value_of_A_l636_63699


namespace NUMINAMATH_GPT_john_ate_half_package_l636_63611

def fraction_of_package_john_ate (servings : ℕ) (calories_per_serving : ℕ) (calories_consumed : ℕ) : ℚ :=
  calories_consumed / (servings * calories_per_serving : ℚ)

theorem john_ate_half_package (servings : ℕ) (calories_per_serving : ℕ) (calories_consumed : ℕ) 
    (h_servings : servings = 3) (h_calories_per_serving : calories_per_serving = 120) (h_calories_consumed : calories_consumed = 180) :
    fraction_of_package_john_ate servings calories_per_serving calories_consumed = 1 / 2 :=
by
  -- Replace the actual proof with sorry to ensure the statement compiles.
  sorry

end NUMINAMATH_GPT_john_ate_half_package_l636_63611


namespace NUMINAMATH_GPT_rationalize_denominator_sum_l636_63615

theorem rationalize_denominator_sum :
  let A := 3
  let B := -9
  let C := -9
  let D := 9
  let E := 165
  let F := 51
  A + B + C + D + E + F = 210 :=
by
  let A := 3
  let B := -9
  let C := -9
  let D := 9
  let E := 165
  let F := 51
  show 3 + -9 + -9 + 9 + 165 + 51 = 210
  sorry

end NUMINAMATH_GPT_rationalize_denominator_sum_l636_63615


namespace NUMINAMATH_GPT_ratio_M_N_l636_63670

-- Definitions of M, Q and N based on the given conditions
variables (M Q P N : ℝ)
variable (h1 : M = 0.40 * Q)
variable (h2 : Q = 0.30 * P)
variable (h3 : N = 0.50 * P)

theorem ratio_M_N : M / N = 6 / 25 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_ratio_M_N_l636_63670


namespace NUMINAMATH_GPT_function_quadrants_l636_63658

theorem function_quadrants (a b : ℝ) (h_a : a > 1) (h_b : b < -1) :
  (∀ x : ℝ, a^x + b > 0 → ∃ x1 : ℝ, a^x1 + b < 0 → ∃ x2 : ℝ, a^x2 + b < 0) :=
sorry

end NUMINAMATH_GPT_function_quadrants_l636_63658


namespace NUMINAMATH_GPT_color_change_probability_l636_63605

-- Definitions based directly on conditions in a)
def light_cycle_duration := 93
def change_intervals_duration := 15
def expected_probability := 5 / 31

-- The Lean 4 statement for the proof problem
theorem color_change_probability :
  (change_intervals_duration / light_cycle_duration) = expected_probability :=
by
  sorry

end NUMINAMATH_GPT_color_change_probability_l636_63605


namespace NUMINAMATH_GPT_four_pow_four_mul_five_pow_four_l636_63657

theorem four_pow_four_mul_five_pow_four : (4 ^ 4) * (5 ^ 4) = 160000 := by
  sorry

end NUMINAMATH_GPT_four_pow_four_mul_five_pow_four_l636_63657


namespace NUMINAMATH_GPT_geometric_seq_problem_l636_63610

theorem geometric_seq_problem
  (a : Nat → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h_cond : a 1 * a 99 = 16) :
  a 20 * a 80 = 16 := 
sorry

end NUMINAMATH_GPT_geometric_seq_problem_l636_63610


namespace NUMINAMATH_GPT_problem_l636_63638

noncomputable def f (ω φ : ℝ) (x : ℝ) := 4 * Real.sin (ω * x + φ)

theorem problem (ω : ℝ) (φ : ℝ) (x1 x2 α : ℝ) (hω : 0 < ω) (hφ : |φ| < Real.pi / 2)
  (h0 : f ω φ 0 = 2 * Real.sqrt 3)
  (hx1 : f ω φ x1 = 0) (hx2 : f ω φ x2 = 0) (hx1x2 : |x1 - x2| = Real.pi / 2)
  (hα : α ∈ Set.Ioo (Real.pi / 12) (Real.pi / 2)) :
  f 2 (Real.pi / 3) α = 12 / 5 ∧ Real.sin (2 * α) = (3 + 4 * Real.sqrt 3) / 10 :=
sorry

end NUMINAMATH_GPT_problem_l636_63638


namespace NUMINAMATH_GPT_circle_equation_solution_l636_63673

theorem circle_equation_solution (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2 * m * x - 2 * m * y + 2 * m^2 + m - 1 = 0) ↔ m < 1 :=
sorry

end NUMINAMATH_GPT_circle_equation_solution_l636_63673


namespace NUMINAMATH_GPT_average_speed_is_one_l636_63646

-- Definition of distance and time
def distance : ℕ := 1800
def time_in_minutes : ℕ := 30
def time_in_seconds : ℕ := time_in_minutes * 60

-- Definition of average speed as distance divided by time
def average_speed (distance : ℕ) (time : ℕ) : ℚ :=
  distance / time

-- Theorem: Given the distance and time, the average speed is 1 meter per second
theorem average_speed_is_one : average_speed distance time_in_seconds = 1 :=
  by
    sorry

end NUMINAMATH_GPT_average_speed_is_one_l636_63646


namespace NUMINAMATH_GPT_num_intersection_points_l636_63693

-- Define the equations of the lines as conditions
def line1 (x y : ℝ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℝ) : Prop := x + 3 * y = 3
def line3 (x y : ℝ) : Prop := 6 * x - 4 * y = 8

-- The theorem to prove the number of intersection points
theorem num_intersection_points :
  ∃! p : ℝ × ℝ, (line1 p.1 p.2 ∧ line2 p.1 p.2) ∨ (line2 p.1 p.2 ∧ line3 p.1 p.2) :=
sorry

end NUMINAMATH_GPT_num_intersection_points_l636_63693


namespace NUMINAMATH_GPT_distance_AB_l636_63684

def C1_polar (ρ θ : Real) : Prop :=
  ρ = 2 * Real.cos θ

def C2_polar (ρ θ : Real) : Prop :=
  ρ^2 * (1 + (Real.sin θ)^2) = 2

def ray_polar (θ : Real) : Prop :=
  θ = Real.pi / 6

theorem distance_AB :
  let ρ1 := 2 * Real.cos (Real.pi / 6)
  let ρ2 := Real.sqrt 10 * 2 / 5
  |ρ1 - ρ2| = Real.sqrt 3 - (2 * Real.sqrt 10) / 5 :=
by
  sorry

end NUMINAMATH_GPT_distance_AB_l636_63684


namespace NUMINAMATH_GPT_flower_team_participation_l636_63676

-- Definitions based on the conditions in the problem
def num_rows : ℕ := 60
def first_row_people : ℕ := 40
def people_increment : ℕ := 1

-- Statement to be proved in Lean
theorem flower_team_participation (x : ℕ) (hx : 1 ≤ x ∧ x ≤ num_rows) : 
  ∃ y : ℕ, y = first_row_people - people_increment + x :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_flower_team_participation_l636_63676


namespace NUMINAMATH_GPT_berries_per_bird_per_day_l636_63642

theorem berries_per_bird_per_day (birds : ℕ) (total_berries : ℕ) (days : ℕ) (berries_per_bird_per_day : ℕ) 
  (h_birds : birds = 5)
  (h_total_berries : total_berries = 140)
  (h_days : days = 4) :
  berries_per_bird_per_day = 7 :=
  sorry

end NUMINAMATH_GPT_berries_per_bird_per_day_l636_63642


namespace NUMINAMATH_GPT_supplementary_angle_measure_l636_63647

theorem supplementary_angle_measure (a b : ℝ) 
  (h1 : a + b = 180) 
  (h2 : a / 5 = b / 4) : b = 80 :=
by
  sorry

end NUMINAMATH_GPT_supplementary_angle_measure_l636_63647


namespace NUMINAMATH_GPT_isosceles_triangle_circumradius_l636_63624

theorem isosceles_triangle_circumradius (b : ℝ) (s : ℝ) (R : ℝ) (hb : b = 6) (hs : s = 5) :
  R = 25 / 8 :=
by 
  sorry

end NUMINAMATH_GPT_isosceles_triangle_circumradius_l636_63624


namespace NUMINAMATH_GPT_letter_puzzle_solutions_l636_63603

theorem letter_puzzle_solutions (A B : ℕ) : 
  (1 ≤ A ∧ A < 10) ∧ (1 ≤ B ∧ B < 10) ∧ (A ≠ B) ∧ (A^B = 10 * B + A) → 
  (A = 2 ∧ B = 5) ∨ (A = 6 ∧ B = 2) ∨ (A = 4 ∧ B = 3) :=
by
  sorry

end NUMINAMATH_GPT_letter_puzzle_solutions_l636_63603


namespace NUMINAMATH_GPT_age_of_b_l636_63600

theorem age_of_b (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 47) : b = 18 :=
by
  sorry

end NUMINAMATH_GPT_age_of_b_l636_63600


namespace NUMINAMATH_GPT_negate_proposition_l636_63636

theorem negate_proposition :
  (¬(∀ x : ℝ, x^2 + x + 1 ≠ 0)) ↔ (∃ x : ℝ, x^2 + x + 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_negate_proposition_l636_63636


namespace NUMINAMATH_GPT_sweater_markup_l636_63680

-- Conditions
variables (W R : ℝ)
axiom h1 : 0.40 * R = 1.20 * W

-- Theorem statement
theorem sweater_markup (W R : ℝ) (h1 : 0.40 * R = 1.20 * W) : (R - W) / W * 100 = 200 :=
sorry

end NUMINAMATH_GPT_sweater_markup_l636_63680


namespace NUMINAMATH_GPT_polynomial_divisibility_l636_63662

theorem polynomial_divisibility 
  (a b c : ℤ)
  (P : ℤ → ℤ)
  (root_condition : ∃ u v : ℤ, u * v * (u + v) = -c ∧ u * v = b) 
  (P_def : ∀ x, P x = x^3 + a * x^2 + b * x + c) :
  2 * P (-1) ∣ (P 1 + P (-1) - 2 * (1 + P 0)) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_divisibility_l636_63662


namespace NUMINAMATH_GPT_tan_identity_l636_63616

theorem tan_identity (α β γ : ℝ) (h : α + β + γ = 45 * π / 180) :
  (1 + Real.tan α) * (1 + Real.tan β) * (1 + Real.tan γ) / (1 + Real.tan α * Real.tan β * Real.tan γ) = 2 :=
by
  sorry

end NUMINAMATH_GPT_tan_identity_l636_63616


namespace NUMINAMATH_GPT_solution_range_l636_63669

-- Given conditions from the table
variable (a b c : ℝ)

def f (x : ℝ) : ℝ := a * x^2 + b * x + c

axiom h₁ : f a b c 1.1 = -0.59
axiom h₂ : f a b c 1.2 = 0.84
axiom h₃ : f a b c 1.3 = 2.29
axiom h₄ : f a b c 1.4 = 3.76

theorem solution_range (a b c : ℝ) : 
  ∃ x : ℝ, 1.3 < x ∧ x < 1.4 ∧ f a b c x = 3 :=
sorry

end NUMINAMATH_GPT_solution_range_l636_63669


namespace NUMINAMATH_GPT_chessboard_queen_placements_l636_63655

theorem chessboard_queen_placements :
  ∃ (n : ℕ), n = 864 ∧
  (∀ (qpos : Finset (Fin 8 × Fin 8)), 
    qpos.card = 3 ∧
    (∀ (q1 q2 q3 : Fin 8 × Fin 8), 
      q1 ∈ qpos ∧ q2 ∈ qpos ∧ q3 ∈ qpos ∧ q1 ≠ q2 ∧ q2 ≠ q3 ∧ q1 ≠ q3 → 
      (q1.1 = q2.1 ∨ q1.2 = q2.2 ∨ abs (q1.1 - q2.1) = abs (q1.2 - q2.2)) ∧ 
      (q1.1 = q3.1 ∨ q1.2 = q3.2 ∨ abs (q1.1 - q3.1) = abs (q1.2 - q3.2)) ∧ 
      (q2.1 = q3.1 ∨ q2.2 = q3.2 ∨ abs (q2.1 - q3.1) = abs (q2.2 - q3.2)))) ↔ n = 864
:=
by
  sorry

end NUMINAMATH_GPT_chessboard_queen_placements_l636_63655


namespace NUMINAMATH_GPT_max_reciprocal_sum_eq_2_l636_63601

theorem max_reciprocal_sum_eq_2 (r1 r2 t q : ℝ) (h1 : r1 + r2 = t) (h2 : r1 * r2 = q)
  (h3 : ∀ n : ℕ, n > 0 → r1 + r2 = r1^n + r2^n) :
  1 / r1^2010 + 1 / r2^2010 = 2 :=
by
  sorry

end NUMINAMATH_GPT_max_reciprocal_sum_eq_2_l636_63601


namespace NUMINAMATH_GPT_jerry_total_cost_l636_63633

-- Definition of the costs and quantities
def cost_color : ℕ := 32
def cost_bw : ℕ := 27
def num_color : ℕ := 3
def num_bw : ℕ := 1

-- Definition of the total cost
def total_cost : ℕ := (cost_color * num_color) + (cost_bw * num_bw)

-- The theorem that needs to be proved
theorem jerry_total_cost : total_cost = 123 :=
by
  sorry

end NUMINAMATH_GPT_jerry_total_cost_l636_63633


namespace NUMINAMATH_GPT_number_of_intersections_l636_63630

theorem number_of_intersections : 
  (∃ p : ℝ × ℝ, p.1^2 + 9 * p.2^2 = 9 ∧ 9 * p.1^2 + p.2^2 = 1) 
  ∧ (∃! p₁ p₂ : ℝ × ℝ, p₁ ≠ p₂ ∧ p₁.1^2 + 9 * p₁.2^2 = 9 ∧ 9 * p₁.1^2 + p₁.2^2 = 1 ∧
    p₂.1^2 + 9 * p₂.2^2 = 9 ∧ 9 * p₂.1^2 + p₂.2^2 = 1) :=
by
  -- The proof will be here
  sorry

end NUMINAMATH_GPT_number_of_intersections_l636_63630


namespace NUMINAMATH_GPT_jessica_and_sibling_age_l636_63622

theorem jessica_and_sibling_age
  (J M S : ℕ)
  (h1 : J = M / 2)
  (h2 : M + 10 = 70)
  (h3 : S = J + ((70 - M) / 2)) :
  J = 40 ∧ S = 45 :=
by
  sorry

end NUMINAMATH_GPT_jessica_and_sibling_age_l636_63622
