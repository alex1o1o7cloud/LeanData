import Mathlib

namespace NUMINAMATH_GPT_jane_chickens_l577_57704

-- Conditions
def eggs_per_chicken_per_week : ℕ := 6
def egg_price_per_dozen : ℕ := 2
def total_income_in_2_weeks : ℕ := 20

-- Mathematical problem
theorem jane_chickens : (total_income_in_2_weeks / egg_price_per_dozen) * 12 / (eggs_per_chicken_per_week * 2) = 10 :=
by
  sorry

end NUMINAMATH_GPT_jane_chickens_l577_57704


namespace NUMINAMATH_GPT_ways_to_place_letters_l577_57788

-- defining the conditions of the problem
def num_letters : Nat := 4
def num_mailboxes : Nat := 3

-- the theorem we need to prove
theorem ways_to_place_letters : 
  (num_mailboxes ^ num_letters) = 81 := 
by 
  sorry

end NUMINAMATH_GPT_ways_to_place_letters_l577_57788


namespace NUMINAMATH_GPT_find_f_ln6_l577_57709

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def condition1 (f : ℝ → ℝ) : Prop :=
  ∀ x, x < 0 → f x = x - Real.exp (-x)

noncomputable def given_function_value : ℝ := Real.log 6

theorem find_f_ln6 (f : ℝ → ℝ)
  (h1 : odd_function f)
  (h2 : condition1 f) :
  f given_function_value = given_function_value + 6 :=
by
  sorry

end NUMINAMATH_GPT_find_f_ln6_l577_57709


namespace NUMINAMATH_GPT_years_in_future_l577_57732

theorem years_in_future (Shekhar Shobha : ℕ) (h1 : Shekhar / Shobha = 4 / 3) (h2 : Shobha = 15) (h3 : Shekhar + t = 26)
  : t = 6 :=
by
  sorry

end NUMINAMATH_GPT_years_in_future_l577_57732


namespace NUMINAMATH_GPT_geometric_sequence_problem_l577_57779

variable {a : ℕ → ℝ}

theorem geometric_sequence_problem (h1 : a 5 * a 7 = 2) (h2 : a 2 + a 10 = 3) : 
  (a 12 / a 4 = 1 / 2) ∨ (a 12 / a 4 = 2) := 
sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l577_57779


namespace NUMINAMATH_GPT_negation_universal_proposition_l577_57795

theorem negation_universal_proposition : 
  (¬ ∀ x : ℝ, x^2 - x < 0) = ∃ x : ℝ, x^2 - x ≥ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_universal_proposition_l577_57795


namespace NUMINAMATH_GPT_range_of_positive_integers_in_list_H_l577_57764

noncomputable def list_H_lower_bound : Int := -15
noncomputable def list_H_length : Nat := 30

theorem range_of_positive_integers_in_list_H :
  ∃(r : Nat), list_H_lower_bound + list_H_length - 1 = 14 ∧ r = 14 - 1 := 
by
  let upper_bound := list_H_lower_bound + Int.ofNat list_H_length - 1
  use (upper_bound - 1).toNat
  sorry

end NUMINAMATH_GPT_range_of_positive_integers_in_list_H_l577_57764


namespace NUMINAMATH_GPT_red_mushrooms_bill_l577_57720

theorem red_mushrooms_bill (R : ℝ) : 
  (2/3) * R + 6 + 3 = 17 → R = 12 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_red_mushrooms_bill_l577_57720


namespace NUMINAMATH_GPT_consecutive_days_sum_l577_57728

theorem consecutive_days_sum (x : ℕ) (h : 3 * x + 3 = 33) : x = 10 ∧ x + 1 = 11 ∧ x + 2 = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_consecutive_days_sum_l577_57728


namespace NUMINAMATH_GPT_circle_division_parts_l577_57721

-- Define the number of parts a circle is divided into by the chords.
noncomputable def numberOfParts (n : ℕ) : ℚ :=
  (n^4 - 6*n^3 + 23*n^2 - 18*n + 24) / 24

-- Prove that the number of parts is given by the defined function.
theorem circle_division_parts (n : ℕ) : numberOfParts n = (n^4 - 6*n^3 + 23*n^2 - 18*n + 24) / 24 := by
  sorry

end NUMINAMATH_GPT_circle_division_parts_l577_57721


namespace NUMINAMATH_GPT_km_per_gallon_proof_l577_57740

-- Define the given conditions
def distance := 100
def gallons := 10

-- Define what we need to prove the correct answer
def kilometers_per_gallon := distance / gallons

-- Prove that the calculated kilometers per gallon is equal to 10
theorem km_per_gallon_proof : kilometers_per_gallon = 10 := by
  sorry

end NUMINAMATH_GPT_km_per_gallon_proof_l577_57740


namespace NUMINAMATH_GPT_vertical_asymptote_at_5_l577_57760

noncomputable def f (x : ℝ) : ℝ := (x^3 + 3*x^2 + 2*x + 10) / (x - 5)

theorem vertical_asymptote_at_5 : ∃ a : ℝ, (a = 5) ∧ ∀ δ > 0, ∃ ε > 0, ∀ x : ℝ, 0 < |x - a| ∧ |x - a| < ε → |f x| > δ :=
by
  sorry

end NUMINAMATH_GPT_vertical_asymptote_at_5_l577_57760


namespace NUMINAMATH_GPT_area_of_estate_l577_57703

theorem area_of_estate (side_length_in_inches : ℝ) (scale : ℝ) (real_side_length : ℝ) (area : ℝ) :
  side_length_in_inches = 12 →
  scale = 100 →
  real_side_length = side_length_in_inches * scale →
  area = real_side_length ^ 2 →
  area = 1440000 :=
by
  sorry

end NUMINAMATH_GPT_area_of_estate_l577_57703


namespace NUMINAMATH_GPT_correct_proposition_l577_57785

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then 1 + x else 1 - x

def prop_A := ∀ x : ℝ, f (Real.sin x) = -f (Real.sin (-x)) ∧ (∃ T > 0, ∀ x, f (Real.sin (x + T)) = f (Real.sin x))
def prop_B := ∀ x : ℝ, f (Real.sin x) = f (Real.sin (-x)) ∧ ¬(∃ T > 0, ∀ x, f (Real.sin (x + T)) = f (Real.sin x))
def prop_C := ∀ x : ℝ, f (Real.sin (1 / x)) = f (Real.sin (-1 / x)) ∧ ¬(∃ T > 0, ∀ x, f (Real.sin (1 / (x + T))) = f (Real.sin (1 / x)))
def prop_D := ∀ x : ℝ, f (Real.sin (1 / x)) = f (Real.sin (-1 / x)) ∧ (∃ T > 0, ∀ x, f (Real.sin (1 / (x + T))) = f (Real.sin (1 / x)))

theorem correct_proposition :
  (¬ prop_A ∧ ¬ prop_B ∧ prop_C ∧ ¬ prop_D) :=
sorry

end NUMINAMATH_GPT_correct_proposition_l577_57785


namespace NUMINAMATH_GPT_complex_quadrant_l577_57774

open Complex

-- Let complex number i be the imaginary unit
noncomputable def purely_imaginary (z : ℂ) : Prop := 
  z.re = 0

theorem complex_quadrant (z : ℂ) (a : ℂ) (hz : purely_imaginary z) (h : (2 + I) * z = 1 + a * I ^ 3) :
  (a + z).re > 0 ∧ (a + z).im < 0 :=
by 
  sorry

end NUMINAMATH_GPT_complex_quadrant_l577_57774


namespace NUMINAMATH_GPT_work_completion_days_l577_57770

theorem work_completion_days (A_time : ℝ) (A_efficiency : ℝ) (B_time : ℝ) (B_efficiency : ℝ) (C_time : ℝ) (C_efficiency : ℝ) :
  A_time = 60 → A_efficiency = 1.5 → B_time = 20 → B_efficiency = 1 → C_time = 30 → C_efficiency = 0.75 → 
  (1 / (A_efficiency / A_time + B_efficiency / B_time + C_efficiency / C_time)) = 10 := 
by
  intros A_time_eq A_efficiency_eq B_time_eq B_efficiency_eq C_time_eq C_efficiency_eq
  rw [A_time_eq, A_efficiency_eq, B_time_eq, B_efficiency_eq, C_time_eq, C_efficiency_eq]
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_work_completion_days_l577_57770


namespace NUMINAMATH_GPT_tyre_flattening_time_l577_57719

theorem tyre_flattening_time (R1 R2 : ℝ) (hR1 : R1 = 1 / 9) (hR2 : R2 = 1 / 6) : 
  1 / (R1 + R2) = 3.6 :=
by 
  sorry

end NUMINAMATH_GPT_tyre_flattening_time_l577_57719


namespace NUMINAMATH_GPT_arithmetic_geometric_properties_l577_57762

noncomputable def arithmetic_seq (a₁ a₂ a₃ : ℝ) :=
  ∃ d : ℝ, a₂ = a₁ + d ∧ a₃ = a₂ + d

noncomputable def geometric_seq (b₁ b₂ b₃ : ℝ) :=
  ∃ q : ℝ, q ≠ 0 ∧ b₂ = b₁ * q ∧ b₃ = b₂ * q

theorem arithmetic_geometric_properties (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) :
  arithmetic_seq a₁ a₂ a₃ →
  geometric_seq b₁ b₂ b₃ →
  ¬(a₁ < a₂ ∧ a₂ > a₃) ∧
  (b₁ < b₂ ∧ b₂ > b₃) ∧
  (a₁ + a₂ < 0 → ¬(a₂ + a₃ < 0)) ∧
  (b₁ * b₂ < 0 → b₂ * b₃ < 0) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_properties_l577_57762


namespace NUMINAMATH_GPT_Ram_money_l577_57756

theorem Ram_money (R G K : ℕ) (h1 : R = 7 * G / 17) (h2 : G = 7 * K / 17) (h3 : K = 4046) : R = 686 := by
  sorry

end NUMINAMATH_GPT_Ram_money_l577_57756


namespace NUMINAMATH_GPT_bouquet_count_l577_57758

theorem bouquet_count : ∃ n : ℕ, n = 9 ∧ ∀ (r c : ℕ), 3 * r + 2 * c = 50 → n = 9 :=
by
  sorry

end NUMINAMATH_GPT_bouquet_count_l577_57758


namespace NUMINAMATH_GPT_common_difference_arithmetic_sequence_l577_57706

theorem common_difference_arithmetic_sequence (d : ℝ) :
  (∀ (n : ℝ) (a_1 : ℝ), a_1 = 9 ∧
  (∃ a₄ a₈ : ℝ, a₄ = a_1 + 3 * d ∧ a₈ = a_1 + 7 * d ∧ a₄ = (a_1 * a₈)^(1/2)) →
  d = 1) :=
sorry

end NUMINAMATH_GPT_common_difference_arithmetic_sequence_l577_57706


namespace NUMINAMATH_GPT_num_people_at_gathering_l577_57723

noncomputable def total_people_at_gathering : ℕ :=
  let wine_soda := 12
  let wine_juice := 10
  let wine_coffee := 6
  let wine_tea := 4
  let soda_juice := 8
  let soda_coffee := 5
  let soda_tea := 3
  let juice_coffee := 7
  let juice_tea := 2
  let coffee_tea := 4
  let wine_soda_juice := 3
  let wine_soda_coffee := 1
  let wine_soda_tea := 2
  let wine_juice_coffee := 3
  let wine_juice_tea := 1
  let wine_coffee_tea := 2
  let soda_juice_coffee := 3
  let soda_juice_tea := 1
  let soda_coffee_tea := 2
  let juice_coffee_tea := 3
  let all_five := 1
  wine_soda + wine_juice + wine_coffee + wine_tea +
  soda_juice + soda_coffee + soda_tea + juice_coffee +
  juice_tea + coffee_tea + wine_soda_juice + wine_soda_coffee +
  wine_soda_tea + wine_juice_coffee + wine_juice_tea +
  wine_coffee_tea + soda_juice_coffee + soda_juice_tea +
  soda_coffee_tea + juice_coffee_tea + all_five

theorem num_people_at_gathering : total_people_at_gathering = 89 := by
  sorry

end NUMINAMATH_GPT_num_people_at_gathering_l577_57723


namespace NUMINAMATH_GPT_least_number_to_subtract_l577_57708

theorem least_number_to_subtract (x : ℕ) (h : x = 7538 % 14) : (7538 - x) % 14 = 0 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_least_number_to_subtract_l577_57708


namespace NUMINAMATH_GPT_units_digit_m_squared_plus_3_pow_m_l577_57705

def m := 2023^2 + 3^2023

theorem units_digit_m_squared_plus_3_pow_m : 
  (m^2 + 3^m) % 10 = 5 := sorry

end NUMINAMATH_GPT_units_digit_m_squared_plus_3_pow_m_l577_57705


namespace NUMINAMATH_GPT_drivers_schedule_l577_57777

/--
  Given the conditions:
  1. One-way trip duration: 2 hours 40 minutes.
  2. Round trip duration: 5 hours 20 minutes.
  3. Rest period after trip: 1 hour.
  4. Driver A returns at 12:40 PM.
  5. Driver A cannot start next trip until 1:40 PM.
  6. Driver D departs at 1:05 PM.
  7. Driver A departs on fifth trip at 4:10 PM.
  8. Driver B departs on sixth trip at 5:30 PM.
  9. Driver B performs the trip from 5:30 PM to 10:50 PM.

  Prove that:
  1. The number of drivers required is 4.
  2. Ivan Petrovich departs on the second trip at 10:40 AM.
-/
theorem drivers_schedule (dep_a_fifth_trip : 16 * 60 + 10 = 970)
(dep_b_sixth_trip : 17 * 60 + 30 = 1050)
(dep_from_1730_to_2250 : 17 * 60 + 30 = 1050 ∧ 22 * 60 + 50 = 1370)
(dep_second_trip : "Ivan_Petrovich" = "10:40 AM") :
  ∃ (drivers : Nat), drivers = 4 ∧ "Ivan_Petrovich" = "10:40 AM" :=
sorry

end NUMINAMATH_GPT_drivers_schedule_l577_57777


namespace NUMINAMATH_GPT_marquita_garden_width_l577_57745

theorem marquita_garden_width
  (mancino_gardens : ℕ) (marquita_gardens : ℕ)
  (mancino_length mancnio_width marquita_length total_area : ℕ)
  (h1 : mancino_gardens = 3)
  (h2 : mancino_length = 16)
  (h3 : mancnio_width = 5)
  (h4 : marquita_gardens = 2)
  (h5 : marquita_length = 8)
  (h6 : total_area = 304) :
  ∃ (marquita_width : ℕ), marquita_width = 4 :=
by
  sorry

end NUMINAMATH_GPT_marquita_garden_width_l577_57745


namespace NUMINAMATH_GPT_find_f_2022_l577_57712

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3

variables (f : ℝ → ℝ)
  (h_condition : satisfies_condition f)
  (h_f1 : f 1 = 1)
  (h_f4 : f 4 = 7)

theorem find_f_2022 : f 2022 = 4043 :=
  sorry

end NUMINAMATH_GPT_find_f_2022_l577_57712


namespace NUMINAMATH_GPT_initial_alloy_weight_l577_57724

theorem initial_alloy_weight
  (x : ℝ)  -- Weight of the initial alloy in ounces
  (h1 : 0.80 * (x + 24) = 0.50 * x + 24)  -- Equation derived from conditions
: x = 16 := 
sorry

end NUMINAMATH_GPT_initial_alloy_weight_l577_57724


namespace NUMINAMATH_GPT_initial_number_of_girls_is_31_l577_57749

-- Define initial number of boys and girls
variables (b g : ℕ)

-- Conditions
def first_condition (g b : ℕ) : Prop := b = 3 * (g - 18)
def second_condition (g b : ℕ) : Prop := 4 * (b - 36) = g - 18

-- Theorem statement
theorem initial_number_of_girls_is_31 (b g : ℕ) (h1 : first_condition g b) (h2 : second_condition g b) : g = 31 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_girls_is_31_l577_57749


namespace NUMINAMATH_GPT_subtraction_digits_l577_57763

theorem subtraction_digits (a b c : ℕ) (h1 : c - a = 2) (h2 : b = c - 1) (h3 : 100 * a + 10 * b + c - (100 * c + 10 * b + a) = 802) :
a = 0 ∧ b = 1 ∧ c = 2 :=
by {
  -- The detailed proof steps will go here
  sorry
}

end NUMINAMATH_GPT_subtraction_digits_l577_57763


namespace NUMINAMATH_GPT_quadratic_equation_unique_l577_57700

/-- Prove that among the given options, the only quadratic equation in \( x \) is \( x^2 - 3x = 0 \). -/
theorem quadratic_equation_unique (A B C D : ℝ → ℝ) :
  A = (3 * x + 2) →
  B = (x^2 - 3 * x) →
  C = (x + 3 * x * y - 1) →
  D = (1 / x - 4) →
  ∃! (eq : ℝ → ℝ), eq = B := by
  sorry

end NUMINAMATH_GPT_quadratic_equation_unique_l577_57700


namespace NUMINAMATH_GPT_total_pages_in_book_l577_57751

variable (p1 p2 p_total : ℕ)
variable (read_first_four_days : p1 = 4 * 45)
variable (read_next_three_days : p2 = 3 * 52)
variable (total_until_last_day : p_total = p1 + p2 + 15)

theorem total_pages_in_book : p_total = 351 :=
by
  -- Introduce the conditions
  rw [read_first_four_days, read_next_three_days] at total_until_last_day
  sorry

end NUMINAMATH_GPT_total_pages_in_book_l577_57751


namespace NUMINAMATH_GPT_polyhedron_volume_l577_57722

-- Define the properties of the polygons
def isosceles_right_triangle (a : ℝ) := a ≠ 0 ∧ ∀ (x y : ℝ), x = y

def square (side : ℝ) := side = 2

def equilateral_triangle (side : ℝ) := side = 2 * Real.sqrt 2

-- Define the conditions
def condition_AE : Prop := isosceles_right_triangle 2
def condition_B : Prop := square 2
def condition_C : Prop := square 2
def condition_D : Prop := square 2
def condition_G : Prop := equilateral_triangle (2 * Real.sqrt 2)

-- Define the polyhedron volume calculation problem
theorem polyhedron_volume (hA : condition_AE) (hE : condition_AE) (hF : condition_AE) (hB : condition_B) (hC : condition_C) (hD : condition_D) (hG : condition_G) : 
  ∃ V : ℝ, V = 16 := 
sorry

end NUMINAMATH_GPT_polyhedron_volume_l577_57722


namespace NUMINAMATH_GPT_number_of_packets_l577_57739

def ounces_in_packet : ℕ := 16 * 16 + 4
def ounces_in_ton : ℕ := 2500 * 16
def gunny_bag_capacity_in_ounces : ℕ := 13 * ounces_in_ton

theorem number_of_packets : gunny_bag_capacity_in_ounces / ounces_in_packet = 2000 :=
by
  sorry

end NUMINAMATH_GPT_number_of_packets_l577_57739


namespace NUMINAMATH_GPT_intersection_A_B_l577_57767

-- Define the sets A and B
def set_A : Set ℝ := { x | x^2 ≤ 1 }
def set_B : Set ℝ := { -2, -1, 0, 1, 2 }

-- The goal is to prove that the intersection of A and B is {-1, 0, 1}
theorem intersection_A_B : set_A ∩ set_B = ({-1, 0, 1} : Set ℝ) :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l577_57767


namespace NUMINAMATH_GPT_radius_of_larger_circle_l577_57737

theorem radius_of_larger_circle (R1 R2 : ℝ) (α : ℝ) (h1 : α = 60) (h2 : R1 = 24) (h3 : R2 = 3 * R1) : 
  R2 = 72 := 
by
  sorry

end NUMINAMATH_GPT_radius_of_larger_circle_l577_57737


namespace NUMINAMATH_GPT_M_eq_N_l577_57726

def M : Set ℝ := {x | ∃ m : ℤ, x = Real.sin ((2 * m - 3) * Real.pi / 6)}
def N : Set ℝ := {y | ∃ n : ℤ, y = Real.cos (n * Real.pi / 3)}

theorem M_eq_N : M = N := 
by 
  sorry

end NUMINAMATH_GPT_M_eq_N_l577_57726


namespace NUMINAMATH_GPT_area_of_circle_given_circumference_l577_57761

theorem area_of_circle_given_circumference (C : ℝ) (hC : C = 18 * Real.pi) (k : ℝ) :
  ∃ r : ℝ, C = 2 * Real.pi * r ∧ k * Real.pi = Real.pi * r^2 → k = 81 :=
by
  sorry

end NUMINAMATH_GPT_area_of_circle_given_circumference_l577_57761


namespace NUMINAMATH_GPT_simple_interest_years_l577_57734

theorem simple_interest_years (P : ℝ) (R : ℝ) (T : ℝ) 
  (h1 : P = 300) 
  (h2 : (P * ((R + 6) / 100) * T) = (P * (R / 100) * T + 90)) : 
  T = 5 := 
by 
  -- Necessary proof steps go here
  sorry

end NUMINAMATH_GPT_simple_interest_years_l577_57734


namespace NUMINAMATH_GPT_find_x_l577_57778

noncomputable def x : ℝ := 80 / 9

theorem find_x
  (hx_pos : 0 < x)
  (hx_condition : x * (⌊x⌋₊ : ℝ) = 80) :
  x = 80 / 9 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l577_57778


namespace NUMINAMATH_GPT_curve_of_polar_equation_is_line_l577_57727

theorem curve_of_polar_equation_is_line (r θ : ℝ) :
  (r = 1 / (Real.sin θ - Real.cos θ)) →
  ∃ (a b c : ℝ), (a ≠ 0 ∨ b ≠ 0) ∧ ∀ (x y : ℝ), r * (Real.sin θ) = y ∧ r * (Real.cos θ) = x → a * x + b * y = c :=
by
  sorry

end NUMINAMATH_GPT_curve_of_polar_equation_is_line_l577_57727


namespace NUMINAMATH_GPT_relationship_xyz_l577_57738

theorem relationship_xyz (x y z : ℝ) (h1 : x = Real.log x) (h2 : y = Real.logb 5 2) (h3 : z = Real.exp (-0.5)) : x > z ∧ z > y :=
by
  sorry

end NUMINAMATH_GPT_relationship_xyz_l577_57738


namespace NUMINAMATH_GPT_three_units_away_from_neg_one_l577_57766

def is_three_units_away (x : ℝ) (y : ℝ) : Prop := abs (x - y) = 3

theorem three_units_away_from_neg_one :
  { x : ℝ | is_three_units_away x (-1) } = {2, -4} := 
by
  sorry

end NUMINAMATH_GPT_three_units_away_from_neg_one_l577_57766


namespace NUMINAMATH_GPT_smallest_N_l577_57718

theorem smallest_N (N : ℕ) : 
  (N = 484) ∧ 
  (∃ k : ℕ, 484 = 4 * k) ∧
  (∃ k : ℕ, 485 = 25 * k) ∧
  (∃ k : ℕ, 486 = 9 * k) ∧
  (∃ k : ℕ, 487 = 121 * k) :=
by
  -- Proof omitted (replaced by sorry)
  sorry

end NUMINAMATH_GPT_smallest_N_l577_57718


namespace NUMINAMATH_GPT_problem1_problem2_l577_57710

variable (x a : ℝ)

def P := x^2 - 5*a*x + 4*a^2 < 0
def Q := (x^2 - 2*x - 8 <= 0) ∧ (x^2 + 3*x - 10 > 0)

theorem problem1 (h : 1 = a) (hP : P x a) (hQ : Q x) : 2 < x ∧ x ≤ 4 :=
sorry

theorem problem2 (h1 : ∀ x, ¬P x a → ¬Q x) (h2 : ∃ x, P x a ∧ ¬Q x) : 1 < a ∧ a ≤ 2 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l577_57710


namespace NUMINAMATH_GPT_min_value_of_reciprocals_l577_57716

theorem min_value_of_reciprocals {x y a b : ℝ} 
  (h1 : 8 * x - y - 4 ≤ 0)
  (h2 : x + y + 1 ≥ 0)
  (h3 : y - 4 * x ≤ 0)
  (h4 : 2 = a * (1 / 2) + b * 1)
  (ha : a > 0)
  (hb : b > 0) :
  (1 / a) + (1 / b) = 9 / 2 :=
sorry

end NUMINAMATH_GPT_min_value_of_reciprocals_l577_57716


namespace NUMINAMATH_GPT_extra_amount_spent_on_shoes_l577_57771

theorem extra_amount_spent_on_shoes (total_cost shirt_cost shoes_cost: ℝ) 
  (h1: total_cost = 300) (h2: shirt_cost = 97) 
  (h3: shoes_cost > 2 * shirt_cost)
  (h4: shirt_cost + shoes_cost = total_cost): 
  shoes_cost - 2 * shirt_cost = 9 :=
by
  sorry

end NUMINAMATH_GPT_extra_amount_spent_on_shoes_l577_57771


namespace NUMINAMATH_GPT_weight_of_3_moles_of_CaI2_is_881_64_l577_57776

noncomputable def molar_mass_Ca : ℝ := 40.08
noncomputable def molar_mass_I : ℝ := 126.90
noncomputable def molar_mass_CaI2 : ℝ := molar_mass_Ca + 2 * molar_mass_I
noncomputable def weight_3_moles_CaI2 : ℝ := 3 * molar_mass_CaI2

theorem weight_of_3_moles_of_CaI2_is_881_64 :
  weight_3_moles_CaI2 = 881.64 :=
by sorry

end NUMINAMATH_GPT_weight_of_3_moles_of_CaI2_is_881_64_l577_57776


namespace NUMINAMATH_GPT_range_of_values_for_a_l577_57799

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * Real.sin x - (1 / 2) * Real.cos (2 * x) + a - (3 / a) + (1 / 2)

theorem range_of_values_for_a (a : ℝ) (ha : a ≠ 0) : 
  (∀ x : ℝ, f x a ≤ 0) ↔ (0 < a ∧ a ≤ 1) :=
by 
  let g (t : ℝ) : ℝ := t^2 + a * t + a - (3 / a)
  have h1 : g (-1) ≤ 0 := by sorry
  have h2 : g (1) ≤ 0 := by sorry
  sorry

end NUMINAMATH_GPT_range_of_values_for_a_l577_57799


namespace NUMINAMATH_GPT_quadratic_function_increasing_l577_57765

theorem quadratic_function_increasing (x : ℝ) : ((x - 1)^2 + 2 < (x + 1 - 1)^2 + 2) ↔ (x > 1) := by
  sorry

end NUMINAMATH_GPT_quadratic_function_increasing_l577_57765


namespace NUMINAMATH_GPT_girls_in_class_l577_57786

theorem girls_in_class (g b : ℕ) (h1 : g + b = 28) (h2 : g * 4 = b * 3) : g = 12 := by
  sorry

end NUMINAMATH_GPT_girls_in_class_l577_57786


namespace NUMINAMATH_GPT_count_positive_integers_l577_57752

theorem count_positive_integers (x : ℤ) : 
  (25 < x^2 + 6 * x + 8) → (x^2 + 6 * x + 8 < 50) → (x > 0) → (x = 3 ∨ x = 4) :=
by sorry

end NUMINAMATH_GPT_count_positive_integers_l577_57752


namespace NUMINAMATH_GPT_contrapositive_geometric_sequence_l577_57736

def geometric_sequence (a b c : ℝ) : Prop :=
  b^2 = a * c

theorem contrapositive_geometric_sequence (a b c : ℝ) :
  (b^2 ≠ a * c) → ¬geometric_sequence a b c :=
by
  intros h
  unfold geometric_sequence
  assumption

end NUMINAMATH_GPT_contrapositive_geometric_sequence_l577_57736


namespace NUMINAMATH_GPT_minimum_value_of_expression_l577_57798

noncomputable def monotonic_function_property
    (f : ℝ → ℝ)
    (h_monotonic : ∀ x y, (x ≤ y → f x ≤ f y) ∨ (x ≥ y → f x ≥ f y))
    (h_additive : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂)
    (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : f a + f (2 * b - 1) = 0): Prop :=
    (1 : ℝ) / a + 8 / b = 25

theorem minimum_value_of_expression 
    (f : ℝ → ℝ)
    (h_monotonic : ∀ x y, (x ≤ y → f x ≤ f y) ∨ (x ≥ y → f x ≥ f y))
    (h_additive : ∀ x₁ x₂, f (x₁ + x₂) = f x₁ + f x₂)
    (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : f a + f (2 * b - 1) = 0) :
    (1 : ℝ) / a + 8 / b = 25 := 
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l577_57798


namespace NUMINAMATH_GPT_solve_equation_l577_57759

theorem solve_equation :
  ∀ (x : ℝ), 
    x^3 + (Real.log 25 + Real.log 32 + Real.log 53) * x = (Real.log 23 + Real.log 35 + Real.log 52) * x^2 + 1 ↔ 
    x = Real.log 23 ∨ x = Real.log 35 ∨ x = Real.log 52 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l577_57759


namespace NUMINAMATH_GPT_range_of_f_l577_57711

noncomputable def f (x : ℝ) : ℝ := 
  Real.cos (2 * x - Real.pi / 3) + 2 * Real.sin (x - Real.pi / 4) * Real.sin (x + Real.pi / 4)

theorem range_of_f : ∀ x ∈ Set.Icc (-Real.pi / 12) (Real.pi / 2), 
  -Real.sqrt 3 / 2 ≤ f x ∧ f x ≤ 1 := by
  sorry

end NUMINAMATH_GPT_range_of_f_l577_57711


namespace NUMINAMATH_GPT_minimum_value_of_expression_l577_57775

noncomputable def expression (x y z : ℝ) : ℝ :=
  (x * y / z + z * x / y + y * z / x) * (x / (y * z) + y / (z * x) + z / (x * y))

theorem minimum_value_of_expression (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) :
  expression x y z ≥ 9 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l577_57775


namespace NUMINAMATH_GPT_smallest_E_of_positive_reals_l577_57746

noncomputable def E (a b c : ℝ) : ℝ :=
  (a^3) / (1 - a^2) + (b^3) / (1 - b^2) + (c^3) / (1 - c^2)

theorem smallest_E_of_positive_reals (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a + b + c = 1) : 
  E a b c = 1 / 8 := 
sorry

end NUMINAMATH_GPT_smallest_E_of_positive_reals_l577_57746


namespace NUMINAMATH_GPT_max_students_late_all_three_days_l577_57755

theorem max_students_late_all_three_days (A B C total l: ℕ) 
  (hA: A = 20) 
  (hB: B = 13) 
  (hC: C = 7) 
  (htotal: total = 30) 
  (hposA: 0 ≤ A) (hposB: 0 ≤ B) (hposC: 0 ≤ C) 
  (hpostotal: 0 ≤ total) 
  : l = 5 := by
  sorry

end NUMINAMATH_GPT_max_students_late_all_three_days_l577_57755


namespace NUMINAMATH_GPT_gamesNextMonth_l577_57793

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

end NUMINAMATH_GPT_gamesNextMonth_l577_57793


namespace NUMINAMATH_GPT_quadrilateral_area_is_48_l577_57790

structure Quadrilateral :=
  (PQ QR RS SP : ℝ)
  (angle_QRS angle_SPQ : ℝ)

def quadrilateral_example : Quadrilateral :=
{ PQ := 11, QR := 7, RS := 9, SP := 3, angle_QRS := 90, angle_SPQ := 90 }

noncomputable def area_of_quadrilateral (Q : Quadrilateral) : ℝ :=
  (1/2 * Q.PQ * Q.SP) + (1/2 * Q.QR * Q.RS)

theorem quadrilateral_area_is_48 (Q : Quadrilateral) (h1 : Q.PQ = 11) (h2 : Q.QR = 7) (h3 : Q.RS = 9) (h4 : Q.SP = 3) (h5 : Q.angle_QRS = 90) (h6 : Q.angle_SPQ = 90) :
  area_of_quadrilateral Q = 48 :=
by
  -- Here would be the proof
  sorry

end NUMINAMATH_GPT_quadrilateral_area_is_48_l577_57790


namespace NUMINAMATH_GPT_select_two_subsets_union_six_elements_l577_57707

def f (n : ℕ) : ℕ :=
  if n = 0 then 1 else 3 * f (n - 1) - 1

theorem select_two_subsets_union_six_elements :
  f 6 = 365 :=
by
  sorry

end NUMINAMATH_GPT_select_two_subsets_union_six_elements_l577_57707


namespace NUMINAMATH_GPT_functions_unique_l577_57733

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

theorem functions_unique (f g: ℝ → ℝ) :
  (∀ x : ℝ, x < 0 → (f (g x) = x / (x * f x - 2)) ∧ (g (f x) = x / (x * g x - 2))) →
  (∀ x : ℝ, 0 < x → (f x = 3 / x ∧ g x = 3 / x)) :=
by
  sorry

end NUMINAMATH_GPT_functions_unique_l577_57733


namespace NUMINAMATH_GPT_total_students_count_l577_57714

theorem total_students_count (n1 n2 n: ℕ) (avg1 avg2 avg_tot: ℝ)
  (h1: n1 = 15) (h2: avg1 = 70) (h3: n2 = 10) (h4: avg2 = 90) (h5: avg_tot = 78)
  (h6: (n1 * avg1 + n2 * avg2) / (n1 + n2) = avg_tot) :
  n = 25 :=
by
  sorry

end NUMINAMATH_GPT_total_students_count_l577_57714


namespace NUMINAMATH_GPT_flowers_lost_l577_57729

theorem flowers_lost 
  (time_per_flower : ℕ)
  (gathered_time : ℕ) 
  (additional_time : ℕ) 
  (classmates : ℕ) 
  (collected_flowers : ℕ) 
  (total_needed : ℕ)
  (lost_flowers : ℕ) 
  (H1 : time_per_flower = 10)
  (H2 : gathered_time = 120)
  (H3 : additional_time = 210)
  (H4 : classmates = 30)
  (H5 : collected_flowers = gathered_time / time_per_flower)
  (H6 : total_needed = classmates + (additional_time / time_per_flower))
  (H7 : lost_flowers = total_needed - classmates) :
lost_flowers = 3 := 
sorry

end NUMINAMATH_GPT_flowers_lost_l577_57729


namespace NUMINAMATH_GPT_range_of_m_if_solution_set_empty_solve_inequality_y_geq_m_l577_57757

noncomputable def quadratic_function (m x : ℝ) : ℝ :=
  (m + 1) * x^2 - m * x + m - 1

-- Part 1
theorem range_of_m_if_solution_set_empty (m : ℝ) :
  (∀ x : ℝ, quadratic_function m x < 0 → false) ↔ m ≥ 2 * Real.sqrt 3 / 3 := sorry

-- Part 2
theorem solve_inequality_y_geq_m (m x : ℝ) (h : m > -2) :
  (quadratic_function m x ≥ m) ↔ 
  (m = -1 → x ≥ 1) ∧
  (m > -1 → x ≤ -1/(m+1) ∨ x ≥ 1) ∧
  (m > -2 ∧ m < -1 → 1 ≤ x ∧ x ≤ -1/(m+1)) := sorry

end NUMINAMATH_GPT_range_of_m_if_solution_set_empty_solve_inequality_y_geq_m_l577_57757


namespace NUMINAMATH_GPT_part_I_part_II_l577_57796

def f (x a : ℝ) : ℝ := abs (3 * x + 2) - abs (2 * x + a)

theorem part_I (a : ℝ) : (∀ x : ℝ, f x a ≥ 0) ↔ a = 4 / 3 :=
by
  sorry

theorem part_II (a : ℝ) : (∃ x : ℝ, 1 ≤ x ∧ x ≤ 2 ∧ f x a ≤ 0) ↔ (3 ≤ a ∨ a ≤ -7) :=
by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l577_57796


namespace NUMINAMATH_GPT_star_j_l577_57787

def star (x y : ℝ) : ℝ := x^3 - x * y

theorem star_j (j : ℝ) : star j (star j j) = 2 * j^3 - j^4 := 
by
  sorry

end NUMINAMATH_GPT_star_j_l577_57787


namespace NUMINAMATH_GPT_solution_in_quadrants_I_and_II_l577_57782

theorem solution_in_quadrants_I_and_II (x y : ℝ) :
  (y > 3 * x) ∧ (y > 6 - 2 * x) → ((x > 0 ∧ y > 0) ∨ (x < 0 ∧ y > 0)) :=
by
  sorry

end NUMINAMATH_GPT_solution_in_quadrants_I_and_II_l577_57782


namespace NUMINAMATH_GPT_remainder_when_divided_by_13_l577_57730

theorem remainder_when_divided_by_13 (N : ℕ) (k : ℕ) (hk : N = 39 * k + 15) : N % 13 = 2 :=
sorry

end NUMINAMATH_GPT_remainder_when_divided_by_13_l577_57730


namespace NUMINAMATH_GPT_count_even_three_digit_numbers_less_than_600_l577_57725

-- Define the digits
def digits : List ℕ := [1, 2, 3, 4, 5, 6]

-- Condition: the number must be less than 600, i.e., hundreds digit in {1, 2, 3, 4, 5}
def valid_hundreds (d : ℕ) : Prop := d ∈ [1, 2, 3, 4, 5]

-- Condition: the units (ones) digit must be even
def valid_units (d : ℕ) : Prop := d ∈ [2, 4, 6]

-- Problem: total number of valid three-digit numbers
def total_valid_numbers : ℕ :=
  List.product (List.product [1, 2, 3, 4, 5] digits) [2, 4, 6] |>.length

-- Proof statement
theorem count_even_three_digit_numbers_less_than_600 :
  total_valid_numbers = 90 := by
  sorry

end NUMINAMATH_GPT_count_even_three_digit_numbers_less_than_600_l577_57725


namespace NUMINAMATH_GPT_two_om_2om5_l577_57797

def om (a b : ℕ) : ℕ := a^b - b^a

theorem two_om_2om5 : om 2 (om 2 5) = 79 := by
  sorry

end NUMINAMATH_GPT_two_om_2om5_l577_57797


namespace NUMINAMATH_GPT_train_speed_is_126_kmh_l577_57772

noncomputable def train_speed_proof : Prop :=
  let length_meters := 560 / 1000           -- Convert length to kilometers
  let time_hours := 16 / 3600               -- Convert time to hours
  let speed := length_meters / time_hours   -- Calculate the speed
  speed = 126                               -- The speed should be 126 km/h

theorem train_speed_is_126_kmh : train_speed_proof := by 
  sorry

end NUMINAMATH_GPT_train_speed_is_126_kmh_l577_57772


namespace NUMINAMATH_GPT_g_600_l577_57702

def g : ℕ → ℕ := sorry

axiom g_mul (x y : ℕ) (hx : x > 0) (hy : y > 0) : g (x * y) = g x + g y
axiom g_12 : g 12 = 18
axiom g_48 : g 48 = 26

theorem g_600 : g 600 = 36 :=
by 
  sorry

end NUMINAMATH_GPT_g_600_l577_57702


namespace NUMINAMATH_GPT_meet_time_approx_l577_57741

noncomputable def length_of_track : ℝ := 1800 -- in meters
noncomputable def speed_first_woman : ℝ := 10 * 1000 / 3600 -- in meters per second
noncomputable def speed_second_woman : ℝ := 20 * 1000 / 3600 -- in meters per second
noncomputable def relative_speed : ℝ := speed_first_woman + speed_second_woman

theorem meet_time_approx (ε : ℝ) (hε : ε = 216.048) :
  ∃ t : ℝ, t = length_of_track / relative_speed ∧ abs (t - ε) < 0.001 :=
by
  sorry

end NUMINAMATH_GPT_meet_time_approx_l577_57741


namespace NUMINAMATH_GPT_largest_band_members_l577_57717

def band_formation (m r x : ℕ) : Prop :=
  m < 100 ∧ m = r * x + 2 ∧ (r - 2) * (x + 1) = m ∧ r - 2 * x = 4

theorem largest_band_members : ∃ (r x m : ℕ), band_formation m r x ∧ m = 98 := 
  sorry

end NUMINAMATH_GPT_largest_band_members_l577_57717


namespace NUMINAMATH_GPT_expected_dietary_restriction_l577_57743

theorem expected_dietary_restriction (n : ℕ) (p : ℚ) (sample_size : ℕ) (expected : ℕ) :
  p = 1 / 4 ∧ sample_size = 300 ∧ expected = sample_size * p → expected = 75 := by
  sorry

end NUMINAMATH_GPT_expected_dietary_restriction_l577_57743


namespace NUMINAMATH_GPT_complex_division_example_l577_57780

-- Given conditions
def i : ℂ := Complex.I

-- The statement we need to prove
theorem complex_division_example : (1 + 3 * i) / (1 + i) = 2 + i :=
by
  sorry

end NUMINAMATH_GPT_complex_division_example_l577_57780


namespace NUMINAMATH_GPT_nap_time_l577_57713

-- Definitions of given conditions
def flight_duration : ℕ := 680
def reading_time : ℕ := 120
def movie_time : ℕ := 240
def dinner_time : ℕ := 30
def radio_time : ℕ := 40
def game_time : ℕ := 70

def total_activity_time : ℕ := reading_time + movie_time + dinner_time + radio_time + game_time

-- Theorem statement
theorem nap_time : (flight_duration - total_activity_time) / 60 = 3 := by
  -- Here would go the proof steps verifying the equality
  sorry

end NUMINAMATH_GPT_nap_time_l577_57713


namespace NUMINAMATH_GPT_find_ages_l577_57794

theorem find_ages (P F M : ℕ) 
  (h1 : F - P = 31)
  (h2 : (F + 8) + (P + 8) = 69)
  (h3 : F - M = 4)
  (h4 : (P + 5) + (M + 5) = 65) :
  P = 11 ∧ F = 42 ∧ M = 38 :=
by
  sorry

end NUMINAMATH_GPT_find_ages_l577_57794


namespace NUMINAMATH_GPT_p_or_q_is_false_implies_p_and_q_is_false_l577_57781

theorem p_or_q_is_false_implies_p_and_q_is_false (p q : Prop) :
  (¬ (p ∨ q) → ¬ (p ∧ q)) ∧ ((¬ (p ∧ q) → (p ∨ q ∨ ¬ (p ∨ q)))) := sorry

end NUMINAMATH_GPT_p_or_q_is_false_implies_p_and_q_is_false_l577_57781


namespace NUMINAMATH_GPT_range_of_a_l577_57753

noncomputable def f (x a : ℝ) : ℝ := x * Real.log x + a * x^2 - (2 * a + 1) * x + 1

theorem range_of_a (a : ℝ) (h_a : 0 < a ∧ a ≤ 1/2) : 
  ∀ x : ℝ, x ∈ Set.Ici a → f x a ≥ a^3 - a - 1/8 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l577_57753


namespace NUMINAMATH_GPT_percentage_income_spent_on_clothes_l577_57768

-- Define the assumptions
def monthly_income : ℝ := 90000
def household_expenses : ℝ := 0.5 * monthly_income
def medicine_expenses : ℝ := 0.15 * monthly_income
def savings : ℝ := 9000

-- Define the proof statement
theorem percentage_income_spent_on_clothes :
  ∃ (clothes_expenses : ℝ),
    clothes_expenses = monthly_income - household_expenses - medicine_expenses - savings ∧
    (clothes_expenses / monthly_income) * 100 = 25 := 
sorry

end NUMINAMATH_GPT_percentage_income_spent_on_clothes_l577_57768


namespace NUMINAMATH_GPT_cube_volume_from_surface_area_l577_57735

theorem cube_volume_from_surface_area (S : ℝ) (h : S = 150) : (S / 6) ^ (3 / 2) = 125 := by
  sorry

end NUMINAMATH_GPT_cube_volume_from_surface_area_l577_57735


namespace NUMINAMATH_GPT_hyperbola_condition_l577_57744

theorem hyperbola_condition (m : ℝ) : (m > 0) ↔ (2 + m > 0 ∧ 1 + m > 0) :=
by sorry

end NUMINAMATH_GPT_hyperbola_condition_l577_57744


namespace NUMINAMATH_GPT_fencing_problem_l577_57773

noncomputable def fencingRequired (L A W F : ℝ) := (A = L * W) → (F = 2 * W + L)

theorem fencing_problem :
  fencingRequired 25 880 35.2 95.4 :=
by
  sorry

end NUMINAMATH_GPT_fencing_problem_l577_57773


namespace NUMINAMATH_GPT_bisection_method_example_l577_57747

noncomputable def f (x : ℝ) : ℝ := x^3 - 6 * x^2 + 4

theorem bisection_method_example :
  (∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0) →
  (∃ x : ℝ, (1 / 2) < x ∧ x < 1 ∧ f x = 0) :=
by
  sorry

end NUMINAMATH_GPT_bisection_method_example_l577_57747


namespace NUMINAMATH_GPT_winning_percentage_l577_57754

theorem winning_percentage (total_games first_games remaining_games : ℕ) 
                           (first_win_percent remaining_win_percent : ℝ)
                           (total_games_eq : total_games = 60)
                           (first_games_eq : first_games = 30)
                           (remaining_games_eq : remaining_games = 30)
                           (first_win_percent_eq : first_win_percent = 0.40)
                           (remaining_win_percent_eq : remaining_win_percent = 0.80) :
                           (first_win_percent * (first_games : ℝ) +
                            remaining_win_percent * (remaining_games : ℝ)) /
                           (total_games : ℝ) * 100 = 60 := sorry

end NUMINAMATH_GPT_winning_percentage_l577_57754


namespace NUMINAMATH_GPT_compute_product_l577_57783

-- Define the conditions
variables {x y : ℝ} (h1 : x - y = 5) (h2 : x^3 - y^3 = 35)

-- Define the theorem to be proved
theorem compute_product (h1 : x - y = 5) (h2 : x^3 - y^3 = 35) : x * y = 190 / 9 := 
sorry

end NUMINAMATH_GPT_compute_product_l577_57783


namespace NUMINAMATH_GPT_distinct_solutions_diff_l577_57791

theorem distinct_solutions_diff (r s : ℝ) 
  (h1 : r ≠ s) 
  (h2 : (5*r - 15)/(r^2 + 3*r - 18) = r + 3) 
  (h3 : (5*s - 15)/(s^2 + 3*s - 18) = s + 3) 
  (h4 : r > s) : 
  r - s = 13 :=
sorry

end NUMINAMATH_GPT_distinct_solutions_diff_l577_57791


namespace NUMINAMATH_GPT_greg_books_difference_l577_57748

theorem greg_books_difference (M K G X : ℕ)
  (hM : M = 32)
  (hK : K = M / 4)
  (hG : G = 2 * K + X)
  (htotal : M + K + G = 65) :
  X = 9 :=
by
  sorry

end NUMINAMATH_GPT_greg_books_difference_l577_57748


namespace NUMINAMATH_GPT_sabrina_basil_leaves_l577_57750

-- Definitions of variables
variables (S B V : ℕ)

-- Conditions as definitions in Lean
def condition1 : Prop := B = 2 * S
def condition2 : Prop := S = V - 5
def condition3 : Prop := B + S + V = 29

-- Problem statement
theorem sabrina_basil_leaves (h1 : condition1 S B) (h2 : condition2 S V) (h3 : condition3 S B V) : B = 12 :=
by {
  sorry
}

end NUMINAMATH_GPT_sabrina_basil_leaves_l577_57750


namespace NUMINAMATH_GPT_range_of_m_l577_57769

theorem range_of_m (m : ℝ) (hm : m > 0) :
  (∀ x, (x^2 + 1) * (x^2 - 8 * x - 20) ≤ 0 → (x^2 - 2 * x + (1 - m^2)) ≤ 0) →
  m ≥ 9 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l577_57769


namespace NUMINAMATH_GPT_average_steps_per_day_l577_57731

theorem average_steps_per_day (total_steps : ℕ) (h : total_steps = 56392) : 
  (total_steps / 7 : ℚ) = 8056.00 :=
by
  sorry

end NUMINAMATH_GPT_average_steps_per_day_l577_57731


namespace NUMINAMATH_GPT_number_of_boys_l577_57742

theorem number_of_boys (M W B : ℕ) (X : ℕ) 
  (h1 : 5 * M = W) 
  (h2 : W = B) 
  (h3 : 5 * M * 12 + W * X + B * X = 180) 
  : B = 15 := 
by sorry

end NUMINAMATH_GPT_number_of_boys_l577_57742


namespace NUMINAMATH_GPT_inequality_holds_for_m_l577_57789

theorem inequality_holds_for_m (n : ℕ) (m : ℕ) :
  (∀ a b : ℝ, (0 < a ∧ 0 < b) ∧ (a + b = 2) → (1 / a^n + 1 / b^n ≥ a^m + b^m)) ↔ (m = n ∨ m = n + 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_m_l577_57789


namespace NUMINAMATH_GPT_least_positive_t_geometric_progression_l577_57784

noncomputable def least_positive_t( α : ℝ ) ( h : 0 < α ∧ α < Real.pi / 2 ) : ℝ :=
  9 - 4 * Real.sqrt 5

theorem least_positive_t_geometric_progression ( α t : ℝ ) ( h : 0 < α ∧ α < Real.pi / 2 ) :
  least_positive_t α h = t ↔
  ∃ r : ℝ, r > 0 ∧
    Real.arcsin (Real.sin α) = α ∧
    Real.arcsin (Real.sin (2 * α)) = 2 * α ∧
    Real.arcsin (Real.sin (7 * α)) = 7 * α ∧
    Real.arcsin (Real.sin (t * α)) = t * α ∧
    (α * r = 2 * α) ∧
    (2 * α * r = 7 * α ) ∧
    (7 * α * r = t * α) :=
sorry

end NUMINAMATH_GPT_least_positive_t_geometric_progression_l577_57784


namespace NUMINAMATH_GPT_mean_weight_of_cats_l577_57701

def weight_list : List ℝ :=
  [87, 90, 93, 95, 95, 98, 104, 106, 106, 107, 109, 110, 111, 112]

noncomputable def total_weight : ℝ := weight_list.sum

noncomputable def mean_weight : ℝ := total_weight / weight_list.length

theorem mean_weight_of_cats : mean_weight = 101.64 := by
  sorry

end NUMINAMATH_GPT_mean_weight_of_cats_l577_57701


namespace NUMINAMATH_GPT_volume_of_cylinder_l577_57792

theorem volume_of_cylinder (r h : ℝ) (hr : r = 1) (hh : h = 2) (A : r * h = 4) : (π * r^2 * h = 2 * π) :=
by
  sorry

end NUMINAMATH_GPT_volume_of_cylinder_l577_57792


namespace NUMINAMATH_GPT_total_apples_l577_57715

-- Definitions and Conditions
variable (a : ℕ) -- original number of apples in the first pile (scaled integer type)
variable (n m : ℕ) -- arbitrary positions in the sequence

-- Arithmetic sequence of initial piles
def initial_piles := [a, 2*a, 3*a, 4*a, 5*a, 6*a]

-- Given condition transformations
def after_removal_distribution (initial_piles : List ℕ) (k : ℕ) : List ℕ :=
  match k with
  | 0 => [0, 2*a + 10, 3*a + 20, 4*a + 30, 5*a + 40, 6*a + 50]
  | 1 => [a + 10, 0, 3*a + 20, 4*a + 30, 5*a + 40, 6*a + 50]
  | 2 => [a + 10, 2*a + 20, 0, 4*a + 30, 5*a + 40, 6*a + 50]
  | 3 => [a + 10, 2*a + 20, 3*a + 30, 0, 5*a + 40, 6*a + 50]
  | 4 => [a + 10, 2*a + 20, 3*a + 30, 4*a + 40, 0, 6*a + 50]
  | _ => [a + 10, 2*a + 20, 3*a + 30, 4*a + 40, 5*a + 50, 0]

-- Prove the total number of apples
theorem total_apples : (a = 35) → (a + 2 * a + 3 * a + 4 * a + 5 * a + 6 * a = 735) :=
by
  intros h1
  sorry

end NUMINAMATH_GPT_total_apples_l577_57715
