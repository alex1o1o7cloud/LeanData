import Mathlib

namespace NUMINAMATH_GPT_moles_of_water_produced_l1819_181925

theorem moles_of_water_produced (H₃PO₄ NaOH NaH₂PO₄ H₂O : ℝ) (h₁ : H₃PO₄ = 3) (h₂ : NaOH = 3) (h₃ : NaH₂PO₄ = 3) (h₄ : NaH₂PO₄ / H₂O = 1) : H₂O = 3 :=
by
  sorry

end NUMINAMATH_GPT_moles_of_water_produced_l1819_181925


namespace NUMINAMATH_GPT_polynomial_at_3_l1819_181935

def f (x : ℝ) : ℝ := 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

theorem polynomial_at_3 : f 3 = 1641 := 
by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_polynomial_at_3_l1819_181935


namespace NUMINAMATH_GPT_cost_price_of_watch_l1819_181978

theorem cost_price_of_watch (C SP1 SP2 : ℝ)
    (h1 : SP1 = 0.90 * C)
    (h2 : SP2 = 1.02 * C)
    (h3 : SP2 = SP1 + 140) :
    C = 1166.67 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_of_watch_l1819_181978


namespace NUMINAMATH_GPT_group_friends_opponents_l1819_181908

theorem group_friends_opponents (n m : ℕ) (h₀ : 2 ≤ n) (h₁ : (n - 1) * m = 15) :
  n * m = 16 ∨ n * m = 18 ∨ n * m = 20 ∨ n * m = 30 :=
by
  sorry

end NUMINAMATH_GPT_group_friends_opponents_l1819_181908


namespace NUMINAMATH_GPT_quadratic_difference_l1819_181999

theorem quadratic_difference (f : ℝ → ℝ) (hpoly : ∃ c d e : ℤ, ∀ x, f x = c*x^2 + d*x + e) 
(h : f (Real.sqrt 3) - f (Real.sqrt 2) = 4) : 
f (Real.sqrt 10) - f (Real.sqrt 7) = 12 := sorry

end NUMINAMATH_GPT_quadratic_difference_l1819_181999


namespace NUMINAMATH_GPT_parallel_vectors_x_value_l1819_181931

/-
Given that \(\overrightarrow{a} = (1,2)\) and \(\overrightarrow{b} = (2x, -3)\) are parallel vectors, prove that \(x = -\frac{3}{4}\).
-/
theorem parallel_vectors_x_value (x : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (2 * x, -3)) 
  (h_parallel : (a.1 * b.2) - (a.2 * b.1) = 0) : 
  x = -3 / 4 := by
  sorry

end NUMINAMATH_GPT_parallel_vectors_x_value_l1819_181931


namespace NUMINAMATH_GPT_common_root_l1819_181948

theorem common_root (a b x : ℝ) (h1 : a > b) (h2 : b > 0) 
  (eq1 : x^2 + a * x + b = 0) (eq2 : x^3 + b * x + a = 0) : x = -1 :=
by
  sorry

end NUMINAMATH_GPT_common_root_l1819_181948


namespace NUMINAMATH_GPT_focus_of_parabola_eq_l1819_181975

-- Define the given parabola equation
def parabola (x : ℝ) : ℝ := -5 * x^2 + 10 * x - 2

-- Statement of the theorem to find the focus of the given parabola
theorem focus_of_parabola_eq (x : ℝ) : 
  let vertex_x := 1
  let vertex_y := 3
  let a := -5
  ∃ focus_x focus_y, 
    focus_x = vertex_x ∧ 
    focus_y = vertex_y - (1 / (4 * a)) ∧
    focus_x = 1 ∧
    focus_y = 59 / 20 := 
  sorry

end NUMINAMATH_GPT_focus_of_parabola_eq_l1819_181975


namespace NUMINAMATH_GPT_quadratic_nonnegative_l1819_181913

theorem quadratic_nonnegative (x y : ℝ) : x^2 + x * y + y^2 ≥ 0 :=
by sorry

end NUMINAMATH_GPT_quadratic_nonnegative_l1819_181913


namespace NUMINAMATH_GPT_complement_A_complement_U_range_of_a_empty_intersection_l1819_181941

open Set Real

noncomputable def complement_A_in_U := { x : ℝ | ¬ (x < -1 ∨ x > 3) }

theorem complement_A_complement_U
  {A : Set ℝ} (hA : A = {x | x^2 - 2 * x - 3 > 0}) :
  (complement_A_in_U = (Icc (-1) 3)) :=
by sorry

theorem range_of_a_empty_intersection
  {B : Set ℝ} {a : ℝ}
  (hB : B = {x | abs (x - a) > 3})
  (h_empty : (Icc (-1) 3) ∩ B = ∅) :
  (0 ≤ a ∧ a ≤ 2) :=
by sorry

end NUMINAMATH_GPT_complement_A_complement_U_range_of_a_empty_intersection_l1819_181941


namespace NUMINAMATH_GPT_logs_quadratic_sum_l1819_181980

theorem logs_quadratic_sum (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b)
  (h_roots : ∀ x, 2 * x^2 + 4 * x + 1 = 0 → (x = Real.log a) ∨ (x = Real.log b)) :
  (Real.log a)^2 + Real.log (a^2) + a * b = 1 / Real.exp 2 - 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_logs_quadratic_sum_l1819_181980


namespace NUMINAMATH_GPT_james_earnings_l1819_181957

-- Define the conditions
def rain_gallons_per_inch : ℕ := 15
def rain_monday : ℕ := 4
def rain_tuesday : ℕ := 3
def price_per_gallon : ℝ := 1.2

-- State the theorem to be proved
theorem james_earnings : (rain_monday * rain_gallons_per_inch + rain_tuesday * rain_gallons_per_inch) * price_per_gallon = 126 :=
by
  sorry

end NUMINAMATH_GPT_james_earnings_l1819_181957


namespace NUMINAMATH_GPT_multiples_of_eleven_ending_in_seven_l1819_181916

theorem multiples_of_eleven_ending_in_seven (n : ℕ) : 
  (∀ k : ℕ, n > 0 ∧ n < 2000 ∧ (∃ m : ℕ, n = 11 * m) ∧ n % 10 = 7) → ∃ c : ℕ, c = 18 := 
by
  sorry

end NUMINAMATH_GPT_multiples_of_eleven_ending_in_seven_l1819_181916


namespace NUMINAMATH_GPT_intersection_S_T_l1819_181961

open Set

def S : Set ℝ := { x | x ≥ 1 }
def T : Set ℝ := { -2, -1, 0, 1, 2 }

theorem intersection_S_T : S ∩ T = { 1, 2 } := by
  sorry

end NUMINAMATH_GPT_intersection_S_T_l1819_181961


namespace NUMINAMATH_GPT_lcm_is_600_l1819_181949

def lcm_of_24_30_40_50_60 : ℕ :=
  Nat.lcm 24 (Nat.lcm 30 (Nat.lcm 40 (Nat.lcm 50 60)))

theorem lcm_is_600 : lcm_of_24_30_40_50_60 = 600 := by
  sorry

end NUMINAMATH_GPT_lcm_is_600_l1819_181949


namespace NUMINAMATH_GPT_die_vanishing_probability_and_floor_value_l1819_181974

/-
Given conditions:
1. The die has four faces labeled 0, 1, 2, 3.
2. When the die lands on a face labeled:
   - 0: the die vanishes.
   - 1: nothing happens (one die remains).
   - 2: the die replicates into 2 dice.
   - 3: the die replicates into 3 dice.
3. All dice (original and replicas) will continuously be rolled.
Prove:
  The value of ⌊10/p⌋ is 24, where p is the probability that all dice will eventually disappear.
-/

theorem die_vanishing_probability_and_floor_value : 
  ∃ (p : ℝ), 
  (p^3 + p^2 - 3 * p + 1 = 0 ∧ 0 ≤ p ∧ p ≤ 1 ∧ p = Real.sqrt 2 - 1) 
  ∧ ⌊10 / p⌋ = 24 := 
    sorry

end NUMINAMATH_GPT_die_vanishing_probability_and_floor_value_l1819_181974


namespace NUMINAMATH_GPT_descent_property_l1819_181984

def quadratic_function (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

theorem descent_property (x : ℝ) (h : x < 3) : (quadratic_function (x + 1) < quadratic_function x) :=
sorry

end NUMINAMATH_GPT_descent_property_l1819_181984


namespace NUMINAMATH_GPT_find_divisor_l1819_181906

theorem find_divisor (n : ℕ) (d : ℕ) (h1 : n = 105829) (h2 : d = 10) (h3 : ∃ k, n - d = k * d) : d = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_divisor_l1819_181906


namespace NUMINAMATH_GPT_max_area_of_garden_l1819_181983

theorem max_area_of_garden
  (w : ℕ) (l : ℕ)
  (h1 : l = 2 * w)
  (h2 : l + 2 * w = 480) : l * w = 28800 :=
sorry

end NUMINAMATH_GPT_max_area_of_garden_l1819_181983


namespace NUMINAMATH_GPT_jack_emails_morning_l1819_181962

-- Definitions from conditions
def emails_evening : ℕ := 7
def additional_emails_morning : ℕ := 2
def emails_morning : ℕ := emails_evening + additional_emails_morning

-- The proof problem
theorem jack_emails_morning : emails_morning = 9 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_jack_emails_morning_l1819_181962


namespace NUMINAMATH_GPT_each_group_has_145_bananas_l1819_181971

theorem each_group_has_145_bananas (total_bananas : ℕ) (groups_bananas : ℕ) : 
  total_bananas = 290 ∧ groups_bananas = 2 → total_bananas / groups_bananas = 145 := 
by 
  sorry

end NUMINAMATH_GPT_each_group_has_145_bananas_l1819_181971


namespace NUMINAMATH_GPT_number_of_female_only_child_students_l1819_181901

def students : Finset ℕ := Finset.range 21 -- Set of students with attendance numbers from 1 to 20

def female_students : Finset ℕ := {1, 3, 4, 6, 7, 10, 11, 13, 16, 17, 18, 20}

def only_child_students : Finset ℕ := {1, 4, 5, 8, 11, 14, 17, 20}

def common_students : Finset ℕ := female_students ∩ only_child_students

theorem number_of_female_only_child_students :
  common_students.card = 5 :=
by
  sorry

end NUMINAMATH_GPT_number_of_female_only_child_students_l1819_181901


namespace NUMINAMATH_GPT_six_rational_right_triangles_same_perimeter_l1819_181967

theorem six_rational_right_triangles_same_perimeter :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ a₃ b₃ c₃ a₄ b₄ c₄ a₅ b₅ c₅ a₆ b₆ c₆ : ℕ),
    a₁^2 + b₁^2 = c₁^2 ∧ a₂^2 + b₂^2 = c₂^2 ∧ a₃^2 + b₃^2 = c₃^2 ∧
    a₄^2 + b₄^2 = c₄^2 ∧ a₅^2 + b₅^2 = c₅^2 ∧ a₆^2 + b₆^2 = c₆^2 ∧
    a₁ + b₁ + c₁ = 720 ∧ a₂ + b₂ + c₂ = 720 ∧ a₃ + b₃ + c₃ = 720 ∧
    a₄ + b₄ + c₄ = 720 ∧ a₅ + b₅ + c₅ = 720 ∧ a₆ + b₆ + c₆ = 720 ∧
    (a₁, b₁, c₁) ≠ (a₂, b₂, c₂) ∧ (a₁, b₁, c₁) ≠ (a₃, b₃, c₃) ∧
    (a₁, b₁, c₁) ≠ (a₄, b₄, c₄) ∧ (a₁, b₁, c₁) ≠ (a₅, b₅, c₅) ∧
    (a₁, b₁, c₁) ≠ (a₆, b₆, c₆) ∧ (a₂, b₂, c₂) ≠ (a₃, b₃, c₃) ∧
    (a₂, b₂, c₂) ≠ (a₄, b₄, c₄) ∧ (a₂, b₂, c₂) ≠ (a₅, b₅, c₅) ∧
    (a₂, b₂, c₂) ≠ (a₆, b₆, c₆) ∧ (a₃, b₃, c₃) ≠ (a₄, b₄, c₄) ∧
    (a₃, b₃, c₃) ≠ (a₅, b₅, c₅) ∧ (a₃, b₃, c₃) ≠ (a₆, b₆, c₆) ∧
    (a₄, b₄, c₄) ≠ (a₅, b₅, c₅) ∧ (a₄, b₄, c₄) ≠ (a₆, b₆, c₆) ∧
    (a₅, b₅, c₅) ≠ (a₆, b₆, c₆) :=
sorry

end NUMINAMATH_GPT_six_rational_right_triangles_same_perimeter_l1819_181967


namespace NUMINAMATH_GPT_greatest_possible_value_of_q_minus_r_l1819_181930

noncomputable def max_difference (q r : ℕ) : ℕ :=
  if q < r then r - q else q - r

theorem greatest_possible_value_of_q_minus_r (q r : ℕ) (x y : ℕ) (hq : q = 10 * x + y) (hr : r = 10 * y + x) (cond : q ≠ r) (hqr : max_difference q r < 20) : q - r = 18 :=
  sorry

end NUMINAMATH_GPT_greatest_possible_value_of_q_minus_r_l1819_181930


namespace NUMINAMATH_GPT_solution_inequality_l1819_181955

-- Define the condition as a predicate
def inequality_condition (x : ℝ) : Prop :=
  (x - 1) * (x + 1) < 0

-- State the theorem that we need to prove
theorem solution_inequality : ∀ x : ℝ, inequality_condition x → (-1 < x ∧ x < 1) :=
by
  intro x hx
  sorry

end NUMINAMATH_GPT_solution_inequality_l1819_181955


namespace NUMINAMATH_GPT_recurring_decimal_36_exceeds_decimal_35_l1819_181947

-- Definition of recurring decimal 0.36...
def recurring_decimal_36 : ℚ := 36 / 99

-- Definition of 0.35 as fraction
def decimal_35 : ℚ := 7 / 20

-- Statement of the math proof problem
theorem recurring_decimal_36_exceeds_decimal_35 :
  recurring_decimal_36 - decimal_35 = 3 / 220 := by
  sorry

end NUMINAMATH_GPT_recurring_decimal_36_exceeds_decimal_35_l1819_181947


namespace NUMINAMATH_GPT_denominator_of_second_fraction_l1819_181945

theorem denominator_of_second_fraction (y x : ℝ) (h_cond : y > 0) (h_eq : (y / 20) + (3 * y / x) = 0.35 * y) : x = 10 :=
sorry

end NUMINAMATH_GPT_denominator_of_second_fraction_l1819_181945


namespace NUMINAMATH_GPT_intersection_sets_l1819_181969

theorem intersection_sets :
  let A := { x : ℝ | x^2 - 1 ≥ 0 }
  let B := { x : ℝ | 1 ≤ x ∧ x < 3 }
  A ∩ B = { x : ℝ | 1 ≤ x ∧ x < 3 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_sets_l1819_181969


namespace NUMINAMATH_GPT_no_purchase_count_l1819_181934

def total_people : ℕ := 15
def people_bought_tvs : ℕ := 9
def people_bought_computers : ℕ := 7
def people_bought_both : ℕ := 3

theorem no_purchase_count : total_people - (people_bought_tvs - people_bought_both) - (people_bought_computers - people_bought_both) - people_bought_both = 2 := by
  sorry

end NUMINAMATH_GPT_no_purchase_count_l1819_181934


namespace NUMINAMATH_GPT_cylinder_radius_l1819_181927

open Real

theorem cylinder_radius (r : ℝ) 
  (h₁ : ∀(V₁ : ℝ), V₁ = π * (r + 4)^2 * 3)
  (h₂ : ∀(V₂ : ℝ), V₂ = π * r^2 * 9)
  (h₃ : ∀(V₁ V₂ : ℝ), V₁ = V₂) :
  r = 2 + 2 * sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_cylinder_radius_l1819_181927


namespace NUMINAMATH_GPT_domain_of_f_l1819_181988

noncomputable def f (x : ℝ) : ℝ := (2*x + 3) / Real.sqrt (3*x - 9)

theorem domain_of_f : ∀ x : ℝ, (3 < x) ↔ (∃ y : ℝ, f y ≠ y) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1819_181988


namespace NUMINAMATH_GPT_triangle_area_correct_l1819_181900

def vector_a : ℝ × ℝ := (4, -3)
def vector_b : ℝ × ℝ := (-6, 5)
def vector_c : ℝ × ℝ := (2 * -6, 2 * 5)

def area_of_triangle (a b c : ℝ × ℝ) : ℝ :=
  0.5 * |a.1 * c.2 - a.2 * c.1|

theorem triangle_area_correct :
  area_of_triangle (4, -3) (0, 0) (-12, 10) = 2 := by
  sorry

end NUMINAMATH_GPT_triangle_area_correct_l1819_181900


namespace NUMINAMATH_GPT_paid_amount_divisible_by_11_l1819_181953

-- Define the original bill amount and the increased bill amount
def original_bill (x : ℕ) : ℕ := x
def paid_amount (x : ℕ) : ℕ := (11 * x) / 10

-- Theorem: The paid amount is divisible by 11
theorem paid_amount_divisible_by_11 (x : ℕ) (h : x % 10 = 0) : paid_amount x % 11 = 0 :=
by
  sorry

end NUMINAMATH_GPT_paid_amount_divisible_by_11_l1819_181953


namespace NUMINAMATH_GPT_Zhang_Hai_average_daily_delivery_is_37_l1819_181917

theorem Zhang_Hai_average_daily_delivery_is_37
  (d1_packages : ℕ) (d1_count : ℕ)
  (d2_packages : ℕ) (d2_count : ℕ)
  (d3_packages : ℕ) (d3_count : ℕ)
  (total_days : ℕ) 
  (h1 : d1_packages = 41) (h2 : d1_count = 1)
  (h3 : d2_packages = 35) (h4 : d2_count = 2)
  (h5 : d3_packages = 37) (h6 : d3_count = 4)
  (h7 : total_days = 7) :
  (d1_count * d1_packages + d2_count * d2_packages + d3_count * d3_packages) / total_days = 37 := 
by sorry

end NUMINAMATH_GPT_Zhang_Hai_average_daily_delivery_is_37_l1819_181917


namespace NUMINAMATH_GPT_binom_coeff_div_prime_l1819_181972

open Nat

theorem binom_coeff_div_prime (p n : ℕ) (hp : Nat.Prime p) (hn : n ≥ p) :
  p ∣ Nat.choose n p :=
by
  sorry

end NUMINAMATH_GPT_binom_coeff_div_prime_l1819_181972


namespace NUMINAMATH_GPT_handshakes_at_gathering_l1819_181911

-- Define the number of couples
def couples := 6

-- Define the total number of people
def total_people := 2 * couples

-- Each person shakes hands with 10 others (excluding their spouse)
def handshakes_per_person := 10

-- Total handshakes counted with pairs counted twice
def total_handshakes := total_people * handshakes_per_person / 2

-- The theorem to prove the number of handshakes
theorem handshakes_at_gathering : total_handshakes = 60 :=
by
  sorry

end NUMINAMATH_GPT_handshakes_at_gathering_l1819_181911


namespace NUMINAMATH_GPT_blue_pill_cost_l1819_181995

theorem blue_pill_cost
  (days : Int := 10)
  (total_expenditure : Int := 430)
  (daily_cost : Int := total_expenditure / days) :
  ∃ (y : Int), y + (y - 3) = daily_cost ∧ y = 23 := by
  sorry

end NUMINAMATH_GPT_blue_pill_cost_l1819_181995


namespace NUMINAMATH_GPT_max_capacity_per_car_l1819_181958

-- Conditions
def num_cars : ℕ := 2
def num_vans : ℕ := 3
def people_per_car : ℕ := 5
def people_per_van : ℕ := 3
def max_people_per_van : ℕ := 8
def additional_people : ℕ := 17

-- Theorem to prove maximum capacity of each car is 6 people
theorem max_capacity_per_car (num_cars num_vans people_per_car people_per_van max_people_per_van additional_people : ℕ) : 
  (num_cars = 2 ∧ num_vans = 3 ∧ people_per_car = 5 ∧ people_per_van = 3 ∧ max_people_per_van = 8 ∧ additional_people = 17) →
  ∃ max_people_per_car, max_people_per_car = 6 :=
by
  sorry

end NUMINAMATH_GPT_max_capacity_per_car_l1819_181958


namespace NUMINAMATH_GPT_compute_f_six_l1819_181928

def f (x : Int) : Int :=
  if x ≥ 0 then -x^2 - 1 else x + 10

theorem compute_f_six (x : Int) : f (f (f (f (f (f 1))))) = -35 :=
by
  sorry

end NUMINAMATH_GPT_compute_f_six_l1819_181928


namespace NUMINAMATH_GPT_intersection_A_B_l1819_181968

def A := { x : ℝ | -5 < x ∧ x < 2 }
def B := { x : ℝ | x^2 - 9 < 0 }
def AB := { x : ℝ | -3 < x ∧ x < 2 }

theorem intersection_A_B : A ∩ B = AB := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1819_181968


namespace NUMINAMATH_GPT_volume_of_bottle_l1819_181924

theorem volume_of_bottle (r h : ℝ) (π : ℝ) (h₀ : π > 0)
  (h₁ : r^2 * h + (4 / 3) * r^3 = 625) :
  π * (r^2 * h + (4 / 3) * r^3) = 625 * π :=
by sorry

end NUMINAMATH_GPT_volume_of_bottle_l1819_181924


namespace NUMINAMATH_GPT_area_triangle_AMB_l1819_181937

def parabola (x : ℝ) : ℝ := x^2 + 2*x + 3

def point_A : ℝ × ℝ := (0, parabola 0)

def rotated_parabola (x : ℝ) : ℝ := -(x + 1)^2 + 2

def point_B : ℝ × ℝ := (0, rotated_parabola 0)

def vertex_M : ℝ × ℝ := (-1, 2)

def area_of_triangle (A B M : ℝ × ℝ) : ℝ :=
  0.5 * (A.2 - M.2) * (M.1 - B.1)

theorem area_triangle_AMB : area_of_triangle point_A point_B vertex_M = 1 :=
  sorry

end NUMINAMATH_GPT_area_triangle_AMB_l1819_181937


namespace NUMINAMATH_GPT_correct_calculation_l1819_181966

theorem correct_calculation :
  -4^2 / (-2)^3 * (-1 / 8) = -1 / 4 := by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1819_181966


namespace NUMINAMATH_GPT_baker_cakes_remaining_l1819_181951

def InitialCakes : ℕ := 48
def SoldCakes : ℕ := 44
def RemainingCakes (initial sold : ℕ) : ℕ := initial - sold

theorem baker_cakes_remaining : RemainingCakes InitialCakes SoldCakes = 4 := 
by {
  -- placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_baker_cakes_remaining_l1819_181951


namespace NUMINAMATH_GPT_diminished_value_l1819_181992

theorem diminished_value (x y : ℝ) (h1 : x = 160)
  (h2 : x / 5 + 4 = x / 4 - y) : y = 4 :=
by
  sorry

end NUMINAMATH_GPT_diminished_value_l1819_181992


namespace NUMINAMATH_GPT_value_of_expression_l1819_181907

theorem value_of_expression :
  4 * 5 + 5 * 4 = 40 :=
sorry

end NUMINAMATH_GPT_value_of_expression_l1819_181907


namespace NUMINAMATH_GPT_rhombus_area_l1819_181919

theorem rhombus_area (d₁ d₂ : ℕ) (h₁ : d₁ = 6) (h₂ : d₂ = 8) : 
  (1 / 2 : ℝ) * d₁ * d₂ = 24 := 
by
  sorry

end NUMINAMATH_GPT_rhombus_area_l1819_181919


namespace NUMINAMATH_GPT_distance_preserving_l1819_181982

variables {Point : Type} {d : Point → Point → ℕ} {f : Point → Point}

axiom distance_one (A B : Point) : d A B = 1 → d (f A) (f B) = 1

theorem distance_preserving :
  ∀ (A B : Point) (n : ℕ), n > 0 → d A B = n → d (f A) (f B) = n :=
by
  sorry

end NUMINAMATH_GPT_distance_preserving_l1819_181982


namespace NUMINAMATH_GPT_surface_area_ratio_l1819_181918

-- Defining conditions
variable (V_E V_J : ℝ) (A_E A_J : ℝ)
variable (volume_ratio : V_J = 30 * (Real.sqrt 30) * V_E)

-- Statement to prove
theorem surface_area_ratio (h : V_J = 30 * (Real.sqrt 30) * V_E) :
  A_J = 30 * A_E :=
by
  sorry

end NUMINAMATH_GPT_surface_area_ratio_l1819_181918


namespace NUMINAMATH_GPT_intersection_A_B_l1819_181979

def A : Set ℕ := {70, 1946, 1997, 2003}
def B : Set ℕ := {1, 10, 70, 2016}

theorem intersection_A_B : A ∩ B = {70} := by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l1819_181979


namespace NUMINAMATH_GPT_focus_of_parabola_l1819_181965

noncomputable def parabola_focus (a h k : ℝ) : ℝ × ℝ :=
  (h, k + 1 / (4 * a))

theorem focus_of_parabola :
  parabola_focus 9 (-1/3) (-3) = (-1/3, -107/36) := 
  sorry

end NUMINAMATH_GPT_focus_of_parabola_l1819_181965


namespace NUMINAMATH_GPT_siblings_water_intake_l1819_181922

theorem siblings_water_intake 
  (theo_daily : ℕ := 8) 
  (mason_daily : ℕ := 7) 
  (roxy_daily : ℕ := 9) 
  (days_in_week : ℕ := 7) 
  : (theo_daily + mason_daily + roxy_daily) * days_in_week = 168 := 
by 
  sorry

end NUMINAMATH_GPT_siblings_water_intake_l1819_181922


namespace NUMINAMATH_GPT_split_into_similar_heaps_l1819_181920

noncomputable def similar_sizes (x y : ℕ) : Prop :=
  x ≤ 2 * y

theorem split_into_similar_heaps (n : ℕ) (h : n > 0) : 
  ∃ f : ℕ → ℕ, (∀ k, k < n → similar_sizes (f (k + 1)) (f k)) ∧ f (n - 1) = n := by
  sorry

end NUMINAMATH_GPT_split_into_similar_heaps_l1819_181920


namespace NUMINAMATH_GPT_jakes_present_weight_l1819_181926

theorem jakes_present_weight:
  ∃ J S : ℕ, J - 15 = 2 * S ∧ J + S = 132 ∧ J = 93 :=
by
  sorry

end NUMINAMATH_GPT_jakes_present_weight_l1819_181926


namespace NUMINAMATH_GPT_similar_triangle_shortest_side_l1819_181981

theorem similar_triangle_shortest_side (a b c : ℕ) (p : ℕ) (h : a = 8 ∧ b = 10 ∧ c = 12 ∧ p = 150) :
  ∃ x : ℕ, (x = p / (a + b + c) ∧ 8 * x = 40) :=
by
  sorry

end NUMINAMATH_GPT_similar_triangle_shortest_side_l1819_181981


namespace NUMINAMATH_GPT_pages_per_day_read_l1819_181956

theorem pages_per_day_read (start_date : ℕ) (end_date : ℕ) (total_pages : ℕ) (fraction_covered : ℚ) (pages_read : ℕ) (days : ℕ) :
  start_date = 1 →
  end_date = 12 →
  total_pages = 144 →
  fraction_covered = 2/3 →
  pages_read = fraction_covered * total_pages →
  days = end_date - start_date + 1 →
  pages_read / days = 8 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_pages_per_day_read_l1819_181956


namespace NUMINAMATH_GPT_area_of_circle_l1819_181915

theorem area_of_circle (C : ℝ) (hC : C = 30 * Real.pi) : ∃ k : ℝ, (Real.pi * (C / (2 * Real.pi))^2 = k * Real.pi) ∧ k = 225 :=
by
  sorry

end NUMINAMATH_GPT_area_of_circle_l1819_181915


namespace NUMINAMATH_GPT_brown_rabbit_hop_distance_l1819_181909

theorem brown_rabbit_hop_distance
  (w : ℕ) (b : ℕ) (t : ℕ)
  (h1 : w = 15)
  (h2 : t = 135)
  (hop_distance_in_5_minutes : w * 5 + b * 5 = t) : 
  b = 12 :=
by
  sorry

end NUMINAMATH_GPT_brown_rabbit_hop_distance_l1819_181909


namespace NUMINAMATH_GPT_cornbread_pieces_count_l1819_181940

-- Define the dimensions of the pan and the pieces of cornbread
def pan_length := 24
def pan_width := 20
def piece_length := 3
def piece_width := 2
def margin := 1

-- Define the effective width after considering the margin
def effective_width := pan_width - margin

-- Prove the number of pieces of cornbread is 72
theorem cornbread_pieces_count :
  (pan_length / piece_length) * (effective_width / piece_width) = 72 :=
by
  sorry

end NUMINAMATH_GPT_cornbread_pieces_count_l1819_181940


namespace NUMINAMATH_GPT_not_detecting_spy_probability_l1819_181976

-- Definitions based on conditions
def forest_size : ℝ := 10
def detection_radius : ℝ := 10

-- Inoperative detector - assuming NE corner
def detector_NE_inoperative : Prop := true

-- Probability calculation result
def probability_not_detected : ℝ := 0.087

-- Theorem to prove
theorem not_detecting_spy_probability :
  (forest_size = 10) ∧ (detection_radius = 10) ∧ detector_NE_inoperative →
  probability_not_detected = 0.087 :=
by
  sorry

end NUMINAMATH_GPT_not_detecting_spy_probability_l1819_181976


namespace NUMINAMATH_GPT_dunkers_lineup_count_l1819_181964

theorem dunkers_lineup_count (players : Finset ℕ) (h_players : players.card = 15) (alice zen : ℕ) 
  (h_alice : alice ∈ players) (h_zen : zen ∈ players) (h_distinct : alice ≠ zen) :
  (∃ (S : Finset (Finset ℕ)), S.card = 2717 ∧ ∀ s ∈ S, s.card = 5 ∧ ¬ (alice ∈ s ∧ zen ∈ s)) :=
by
  sorry

end NUMINAMATH_GPT_dunkers_lineup_count_l1819_181964


namespace NUMINAMATH_GPT_max_intersections_between_quadrilateral_and_pentagon_l1819_181914

-- Definitions based on the conditions
def quadrilateral_sides : ℕ := 4
def pentagon_sides : ℕ := 5

-- Theorem statement based on the problem
theorem max_intersections_between_quadrilateral_and_pentagon 
  (qm_sides : ℕ := quadrilateral_sides) 
  (pm_sides : ℕ := pentagon_sides) : 
  (∀ (n : ℕ), n = qm_sides →
    ∀ (m : ℕ), m = pm_sides →
      ∀ (intersection_points : ℕ), 
        intersection_points = (n * m) →
        intersection_points = 20) :=
sorry

end NUMINAMATH_GPT_max_intersections_between_quadrilateral_and_pentagon_l1819_181914


namespace NUMINAMATH_GPT_math_problem_l1819_181902

variable (a b c : ℝ)

theorem math_problem 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a^2 + b^2 + c^2 = 1) : 
  (ab / c + bc / a + ca / b) ≥ Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l1819_181902


namespace NUMINAMATH_GPT_angle_D_calculation_l1819_181952

theorem angle_D_calculation (A B E C D : ℝ)
  (h1 : A + B = 180)
  (h2 : C = D)
  (h3 : A = 50)
  (h4 : E = 60)
  (h5 : A + B + E = 180)
  (h6 : B + C + D = 180) :
  D = 55 :=
by
  sorry

end NUMINAMATH_GPT_angle_D_calculation_l1819_181952


namespace NUMINAMATH_GPT_identify_fraction_l1819_181993

variable {a b : ℚ}

def is_fraction (x : ℚ) (y : ℚ) := ∃ (n : ℚ), x = n / y

theorem identify_fraction :
  is_fraction 2 a ∧ ¬ is_fraction (2 * a) 3 ∧ ¬ is_fraction (-b) 2 ∧ ¬ is_fraction (3 * a + 1) 2 :=
by
  sorry

end NUMINAMATH_GPT_identify_fraction_l1819_181993


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l1819_181990

theorem part_a (p q : ℝ) : q < p^2 → ∃ r1 r2 : ℝ, r1 ≠ r2 ∧ (r1 + r2 = 2 * p) ∧ (r1 * r2 = q) :=
by
  sorry

theorem part_b (p q : ℝ) : q = 4 * p - 4 → (2^2 - 2 * p * 2 + q = 0) :=
by
  sorry

theorem part_c (p q : ℝ) : q = p^2 ∧ q = 4 * p - 4 → (p = 2 ∧ q = 4) :=
by
  sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l1819_181990


namespace NUMINAMATH_GPT_Rogers_expense_fraction_l1819_181912

variables (B m s p : ℝ)

theorem Rogers_expense_fraction (h1 : m = 0.25 * (B - s))
                              (h2 : s = 0.10 * (B - m))
                              (h3 : p = 0.10 * (m + s)) :
  m + s + p = 0.34 * B :=
by
  sorry

end NUMINAMATH_GPT_Rogers_expense_fraction_l1819_181912


namespace NUMINAMATH_GPT_joe_new_average_score_after_dropping_lowest_l1819_181921

theorem joe_new_average_score_after_dropping_lowest 
  (initial_average : ℕ)
  (lowest_score : ℕ)
  (num_tests : ℕ)
  (new_num_tests : ℕ)
  (total_points : ℕ)
  (new_total_points : ℕ)
  (new_average : ℕ) :
  initial_average = 70 →
  lowest_score = 55 →
  num_tests = 4 →
  new_num_tests = 3 →
  total_points = num_tests * initial_average →
  new_total_points = total_points - lowest_score →
  new_average = new_total_points / new_num_tests →
  new_average = 75 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_joe_new_average_score_after_dropping_lowest_l1819_181921


namespace NUMINAMATH_GPT_particle_probability_at_2_3_after_5_moves_l1819_181946

theorem particle_probability_at_2_3_after_5_moves:
  ∃ (C : ℕ), C = Nat.choose 5 2 ∧
  (1/2 ^ 5 * C) = (Nat.choose 5 2) * ((1/2: ℝ) ^ 5) := by
sorry

end NUMINAMATH_GPT_particle_probability_at_2_3_after_5_moves_l1819_181946


namespace NUMINAMATH_GPT_books_into_bags_l1819_181932

def books := Finset.range 5
def bags := Finset.range 4

noncomputable def arrangement_count : ℕ :=
  -- definition of arrangement_count can be derived from the solution logic
  sorry

theorem books_into_bags : arrangement_count = 51 := 
  sorry

end NUMINAMATH_GPT_books_into_bags_l1819_181932


namespace NUMINAMATH_GPT_exists_x_f_lt_g_l1819_181987

noncomputable def f (x : ℝ) := (2 / Real.exp 1) ^ x

noncomputable def g (x : ℝ) := (Real.exp 1 / 3) ^ x

theorem exists_x_f_lt_g : ∃ x : ℝ, f x < g x := by
  sorry

end NUMINAMATH_GPT_exists_x_f_lt_g_l1819_181987


namespace NUMINAMATH_GPT_largest_mersenne_prime_lt_1000_l1819_181910

def is_prime (p : ℕ) : Prop := ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

def is_mersenne_prime (n : ℕ) : Prop :=
  is_prime n ∧ ∃ p : ℕ, is_prime p ∧ n = 2^p - 1

theorem largest_mersenne_prime_lt_1000 : ∃ (n : ℕ), is_mersenne_prime n ∧ n < 1000 ∧ ∀ (m : ℕ), is_mersenne_prime m ∧ m < 1000 → m ≤ n :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_largest_mersenne_prime_lt_1000_l1819_181910


namespace NUMINAMATH_GPT_total_bins_sum_l1819_181998

def total_bins_soup : ℝ := 0.2
def total_bins_vegetables : ℝ := 0.35
def total_bins_fruits : ℝ := 0.15
def total_bins_pasta : ℝ := 0.55
def total_bins_canned_meats : ℝ := 0.275
def total_bins_beans : ℝ := 0.175

theorem total_bins_sum :
  total_bins_soup + total_bins_vegetables + total_bins_fruits + total_bins_pasta + total_bins_canned_meats + total_bins_beans = 1.7 :=
by
  sorry

end NUMINAMATH_GPT_total_bins_sum_l1819_181998


namespace NUMINAMATH_GPT_simplified_expression_evaluate_at_zero_l1819_181996

noncomputable def simplify_expr (x : ℝ) : ℝ :=
  (x^2 / (x + 1) - x + 1) / ((x^2 - 1) / (x^2 + 2 * x + 1))

theorem simplified_expression (x : ℝ) (h₁ : x ≠ -1) (h₂ : x ≠ 1) : 
  simplify_expr x = 1 / (x - 1) :=
by sorry

theorem evaluate_at_zero (h₁ : (0 : ℝ) ≠ -1) (h₂ : (0 : ℝ) ≠ 1) : 
  simplify_expr 0 = -1 :=
by sorry

end NUMINAMATH_GPT_simplified_expression_evaluate_at_zero_l1819_181996


namespace NUMINAMATH_GPT_calculate_expression_l1819_181923

theorem calculate_expression : 
  (3 * 7.5 * (6 + 4) / 2.5) = 90 := 
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l1819_181923


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1819_181905

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∃ r > 0, ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : ∀ n, a n > 0)
  (h3 : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) : 
  a 3 + a 5 = 5 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1819_181905


namespace NUMINAMATH_GPT_negation_of_p_is_neg_p_l1819_181936

-- Define the proposition p
def p : Prop := ∃ m : ℝ, m > 0 ∧ ∃ x : ℝ, m * x^2 + x - 2 * m = 0

-- Define the negation of p
def neg_p : Prop := ∀ m : ℝ, m > 0 → ¬ ∃ x : ℝ, m * x^2 + x - 2 * m = 0

-- The theorem statement
theorem negation_of_p_is_neg_p : (¬ p) = neg_p := 
by
  sorry

end NUMINAMATH_GPT_negation_of_p_is_neg_p_l1819_181936


namespace NUMINAMATH_GPT_weighted_average_correct_l1819_181954

-- Define the marks
def english_marks : ℝ := 76
def mathematics_marks : ℝ := 65
def physics_marks : ℝ := 82
def chemistry_marks : ℝ := 67
def biology_marks : ℝ := 85

-- Define the weightages
def english_weightage : ℝ := 0.20
def mathematics_weightage : ℝ := 0.25
def physics_weightage : ℝ := 0.25
def chemistry_weightage : ℝ := 0.15
def biology_weightage : ℝ := 0.15

-- Define the weighted sum calculation
def weighted_sum : ℝ :=
  english_marks * english_weightage + 
  mathematics_marks * mathematics_weightage + 
  physics_marks * physics_weightage + 
  chemistry_marks * chemistry_weightage + 
  biology_marks * biology_weightage

-- Define the theorem statement: the weighted average marks
theorem weighted_average_correct : weighted_sum = 74.75 :=
by
  sorry

end NUMINAMATH_GPT_weighted_average_correct_l1819_181954


namespace NUMINAMATH_GPT_peaches_left_at_stand_l1819_181963

def initial_peaches : ℝ := 34.0
def picked_peaches : ℝ := 86.0
def spoiled_peaches : ℝ := 12.0
def sold_peaches : ℝ := 27.0

theorem peaches_left_at_stand :
  initial_peaches + picked_peaches - spoiled_peaches - sold_peaches = 81.0 :=
by
  -- initial_peaches + picked_peaches - spoiled_peaches - sold_peaches = 84.0
  sorry

end NUMINAMATH_GPT_peaches_left_at_stand_l1819_181963


namespace NUMINAMATH_GPT_find_circles_tangent_to_axes_l1819_181950

def tangent_to_axes_and_passes_through (R : ℝ) (P : ℝ × ℝ) :=
  let center := (R, R)
  (P.1 - R) ^ 2 + (P.2 - R) ^ 2 = R ^ 2

theorem find_circles_tangent_to_axes (x y : ℝ) :
  (tangent_to_axes_and_passes_through 1 (2, 1) ∧ tangent_to_axes_and_passes_through 1 (x, y)) ∨
  (tangent_to_axes_and_passes_through 5 (2, 1) ∧ tangent_to_axes_and_passes_through 5 (x, y)) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_circles_tangent_to_axes_l1819_181950


namespace NUMINAMATH_GPT_infinite_non_congruent_integers_l1819_181960

theorem infinite_non_congruent_integers (a : ℕ → ℤ) (m : ℕ → ℤ) (k : ℕ)
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ k → 2 ≤ m i)
  (h2 : ∀ i : ℕ, 1 ≤ i ∧ i < k → 2 * m i ≤ m (i + 1)) :
  ∃ (x : ℕ), ∀ i : ℕ, 1 ≤ i ∧ i ≤ k → ¬ (x % (m i) = a i % (m i)) :=
sorry

end NUMINAMATH_GPT_infinite_non_congruent_integers_l1819_181960


namespace NUMINAMATH_GPT_min_val_expression_l1819_181989

theorem min_val_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a^2 * b + b^2 * c + c^2 * a = 3) : 
  a^7 * b + b^7 * c + c^7 * a + a * b^3 + b * c^3 + c * a^3 ≥ 6 :=
sorry

end NUMINAMATH_GPT_min_val_expression_l1819_181989


namespace NUMINAMATH_GPT_smaller_root_of_equation_l1819_181929

theorem smaller_root_of_equation :
  ∀ x : ℚ, (x - 7 / 8)^2 + (x - 1/4) * (x - 7 / 8) = 0 → x = 9 / 16 :=
by
  intro x
  intro h
  sorry

end NUMINAMATH_GPT_smaller_root_of_equation_l1819_181929


namespace NUMINAMATH_GPT_range_of_a_l1819_181986

theorem range_of_a (a : ℝ) :
  (∀ (x y : ℝ), (1 ≤ x ∧ x ≤ 2) ∧ (2 ≤ y ∧ y ≤ 3) → (x * y ≤ a * x^2 + 2 * y^2)) →
  a ≥ -1 :=
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l1819_181986


namespace NUMINAMATH_GPT_prob_sum_divisible_by_4_is_1_4_l1819_181991

/-- 
  Given two wheels each with numbers from 1 to 8, 
  the probability that the sum of two selected numbers from the wheels is divisible by 4.
-/
noncomputable def prob_sum_divisible_by_4 : ℚ :=
  let outcomes : ℕ := 8 * 8
  let favorable_outcomes : ℕ := 16
  favorable_outcomes / outcomes

theorem prob_sum_divisible_by_4_is_1_4 : prob_sum_divisible_by_4 = 1 / 4 := 
  by
    -- Statement is left as sorry as the proof steps are not required.
    sorry

end NUMINAMATH_GPT_prob_sum_divisible_by_4_is_1_4_l1819_181991


namespace NUMINAMATH_GPT_CauchySchwarz_l1819_181938

theorem CauchySchwarz' (a b x y : ℝ) : (a^2 + b^2) * (x^2 + y^2) ≥ (a * x + b * y)^2 := by
  sorry

end NUMINAMATH_GPT_CauchySchwarz_l1819_181938


namespace NUMINAMATH_GPT_walmart_pot_stacking_l1819_181959

theorem walmart_pot_stacking :
  ∀ (total_pots pots_per_set shelves : ℕ),
    total_pots = 60 →
    pots_per_set = 5 →
    shelves = 4 →
    (total_pots / pots_per_set / shelves) = 3 :=
by 
  intros total_pots pots_per_set shelves h1 h2 h3
  sorry

end NUMINAMATH_GPT_walmart_pot_stacking_l1819_181959


namespace NUMINAMATH_GPT_current_rate_l1819_181994

variable (c : ℝ)

def still_water_speed : ℝ := 3.6

axiom rowing_time_ratio (c : ℝ) : (2 : ℝ) * (still_water_speed - c) = still_water_speed + c

theorem current_rate : c = 1.2 :=
by
  sorry

end NUMINAMATH_GPT_current_rate_l1819_181994


namespace NUMINAMATH_GPT_coordinates_of_M_l1819_181944

-- Let M be a point in the 2D Cartesian plane
variable {x y : ℝ}

-- Definition of the conditions
def distance_from_x_axis (y : ℝ) : Prop := abs y = 1
def distance_from_y_axis (x : ℝ) : Prop := abs x = 2

-- Theorem to prove
theorem coordinates_of_M (hx : distance_from_y_axis x) (hy : distance_from_x_axis y) :
  (x = 2 ∧ y = 1) ∨ (x = 2 ∧ y = -1) ∨ (x = -2 ∧ y = 1) ∨ (x = -2 ∧ y = -1) :=
sorry

end NUMINAMATH_GPT_coordinates_of_M_l1819_181944


namespace NUMINAMATH_GPT_find_inverse_l1819_181943

theorem find_inverse :
  ∀ (f : ℝ → ℝ), (∀ x, f x = 3 * x ^ 3 + 9) → (f⁻¹ 90 = 3) :=
by
  intros f h
  sorry

end NUMINAMATH_GPT_find_inverse_l1819_181943


namespace NUMINAMATH_GPT_arithmetic_sequence_monotone_l1819_181903

theorem arithmetic_sequence_monotone (a : ℕ → ℝ) (d : ℝ) (h_arithmetic : ∀ n, a (n + 1) - a n = d) :
  (a 2 > a 1) ↔ (∀ n, a (n + 1) > a n) :=
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_monotone_l1819_181903


namespace NUMINAMATH_GPT_Eunji_has_most_marbles_l1819_181939

-- Declare constants for each person's marbles
def Minyoung_marbles : ℕ := 4
def Yujeong_marbles : ℕ := 2
def Eunji_marbles : ℕ := Minyoung_marbles + 1

-- Theorem: Eunji has the most marbles
theorem Eunji_has_most_marbles :
  Eunji_marbles > Minyoung_marbles ∧ Eunji_marbles > Yujeong_marbles :=
by
  sorry

end NUMINAMATH_GPT_Eunji_has_most_marbles_l1819_181939


namespace NUMINAMATH_GPT_rajan_income_l1819_181973

theorem rajan_income : 
  ∀ (x y : ℕ), 
  7 * x - 6 * y = 1000 → 
  6 * x - 5 * y = 1000 → 
  7 * x = 7000 := 
by 
  intros x y h1 h2
  sorry

end NUMINAMATH_GPT_rajan_income_l1819_181973


namespace NUMINAMATH_GPT_cylinder_increase_l1819_181977

theorem cylinder_increase (x : ℝ) (r h : ℝ) (π : ℝ) 
  (h₁ : r = 5) (h₂ : h = 10) 
  (h₃ : π > 0) 
  (h_equal_volumes : π * (r + x) ^ 2 * h = π * r ^ 2 * (h + x)) :
  x = 5 / 2 :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_cylinder_increase_l1819_181977


namespace NUMINAMATH_GPT_number_of_beavers_in_second_group_l1819_181985

-- Define the number of beavers and the time for the first group
def numBeavers1 := 20
def time1 := 3

-- Define the time for the second group
def time2 := 5

-- Define the total work done (which is constant)
def work := numBeavers1 * time1

-- Define the number of beavers in the second group
def numBeavers2 := 12

-- Theorem stating the mathematical equivalence
theorem number_of_beavers_in_second_group : numBeavers2 * time2 = work :=
by
  -- remaining proof steps would go here
  sorry

end NUMINAMATH_GPT_number_of_beavers_in_second_group_l1819_181985


namespace NUMINAMATH_GPT_polynomial_factorization_l1819_181933

theorem polynomial_factorization (m n : ℤ) (h₁ : (x^2 + m * x + 6 : ℤ) = (x - 2) * (x + n)) : m = -5 := by
  sorry

end NUMINAMATH_GPT_polynomial_factorization_l1819_181933


namespace NUMINAMATH_GPT_length_diff_width_8m_l1819_181942

variables (L W : ℝ)

theorem length_diff_width_8m (h1: W = (1/2) * L) (h2: L * W = 128) : L - W = 8 :=
by sorry

end NUMINAMATH_GPT_length_diff_width_8m_l1819_181942


namespace NUMINAMATH_GPT_negation_of_exists_l1819_181970

theorem negation_of_exists (x : ℕ) : (¬ ∃ x : ℕ, x^2 ≤ x) := 
by 
  sorry

end NUMINAMATH_GPT_negation_of_exists_l1819_181970


namespace NUMINAMATH_GPT_lara_puts_flowers_in_vase_l1819_181997

theorem lara_puts_flowers_in_vase : 
  ∀ (total_flowers mom_flowers flowers_given_more : ℕ), 
    total_flowers = 52 →
    mom_flowers = 15 →
    flowers_given_more = 6 →
  (total_flowers - (mom_flowers + (mom_flowers + flowers_given_more))) = 16 :=
by
  intros total_flowers mom_flowers flowers_given_more h1 h2 h3
  sorry

end NUMINAMATH_GPT_lara_puts_flowers_in_vase_l1819_181997


namespace NUMINAMATH_GPT_area_of_shaded_region_l1819_181904

-- Define the vertices of the larger square
def large_square_vertices : List (ℝ × ℝ) := [(0, 0), (40, 0), (40, 40), (0, 40)]

-- Define the vertices of the polygon forming the shaded area
def shaded_polygon_vertices : List (ℝ × ℝ) := [(0, 0), (20, 0), (40, 30), (40, 40), (10, 40), (0, 10)]

-- Provide the area of the larger square for reference
def large_square_area : ℝ := 1600

-- Provide the area of the triangles subtracted
def triangles_area : ℝ := 450

-- The main theorem stating the problem:
theorem area_of_shaded_region :
  let shaded_area := large_square_area - triangles_area
  shaded_area = 1150 :=
by
  sorry

end NUMINAMATH_GPT_area_of_shaded_region_l1819_181904
