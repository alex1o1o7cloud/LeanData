import Mathlib

namespace NUMINAMATH_GPT_inequality_may_not_hold_l1079_107923

theorem inequality_may_not_hold (a b : ℝ) (h : 0 < b ∧ b < a) :
  ¬(∀ x y : ℝ,  x = 1 / (a - b) → y = 1 / b → x > y) :=
sorry

end NUMINAMATH_GPT_inequality_may_not_hold_l1079_107923


namespace NUMINAMATH_GPT_find_a_minus_b_l1079_107931

theorem find_a_minus_b (a b c d : ℤ) 
  (h1 : (a - b) + c - d = 19) 
  (h2 : a - b - c - d = 9) : 
  a - b = 14 :=
sorry

end NUMINAMATH_GPT_find_a_minus_b_l1079_107931


namespace NUMINAMATH_GPT_ratio_of_money_spent_on_clothes_is_1_to_2_l1079_107922

-- Definitions based on conditions
def allowance1 : ℕ := 5
def weeks1 : ℕ := 8
def allowance2 : ℕ := 6
def weeks2 : ℕ := 6
def cost_video : ℕ := 35
def remaining_money : ℕ := 3

-- Calculations
def total_saved : ℕ := (allowance1 * weeks1) + (allowance2 * weeks2)
def total_expended : ℕ := cost_video + remaining_money
def spent_on_clothes : ℕ := total_saved - total_expended

-- Prove the ratio of money spent on clothes to the total money saved is 1:2
theorem ratio_of_money_spent_on_clothes_is_1_to_2 :
  (spent_on_clothes : ℚ) / (total_saved : ℚ) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_money_spent_on_clothes_is_1_to_2_l1079_107922


namespace NUMINAMATH_GPT_wesley_breenah_ages_l1079_107960

theorem wesley_breenah_ages (w b : ℕ) (h₁ : w = 15) (h₂ : b = 7) (h₃ : w + b = 22) :
  ∃ n : ℕ, 2 * (w + b) = (w + n) + (b + n) := by
  exists 11
  sorry

end NUMINAMATH_GPT_wesley_breenah_ages_l1079_107960


namespace NUMINAMATH_GPT_joe_height_l1079_107948

theorem joe_height (S J : ℕ) (h1 : S + J = 120) (h2 : J = 2 * S + 6) : J = 82 :=
by
  sorry

end NUMINAMATH_GPT_joe_height_l1079_107948


namespace NUMINAMATH_GPT_intersection_of_P_and_Q_l1079_107993

def P : Set ℝ := {x | 1 ≤ x}
def Q : Set ℝ := {x | x < 2}

theorem intersection_of_P_and_Q : P ∩ Q = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_P_and_Q_l1079_107993


namespace NUMINAMATH_GPT_pentagon_inequality_l1079_107959

-- Definitions
variables {S R1 R2 R3 R4 R5 : ℝ}
noncomputable def sine108 := Real.sin (108 * Real.pi / 180)

-- Theorem statement
theorem pentagon_inequality (h_area : S > 0) (h_radii : R1 > 0 ∧ R2 > 0 ∧ R3 > 0 ∧ R4 > 0 ∧ R5 > 0) :
  R1^4 + R2^4 + R3^4 + R4^4 + R5^4 ≥ (4 / (5 * sine108^2)) * S^2 :=
by
  sorry

end NUMINAMATH_GPT_pentagon_inequality_l1079_107959


namespace NUMINAMATH_GPT_avg_variance_stability_excellent_performance_probability_l1079_107937

-- Define the scores of players A and B in seven games
def scores_A : List ℕ := [26, 28, 32, 22, 37, 29, 36]
def scores_B : List ℕ := [26, 29, 32, 28, 39, 29, 27]

-- Define the mean and variance calculations
def mean (scores : List ℕ) : ℚ := (scores.sum : ℚ) / scores.length
def variance (scores : List ℕ) : ℚ := 
  (scores.map (λ x => (x - mean scores) ^ 2)).sum / scores.length

theorem avg_variance_stability :
  mean scores_A = 30 ∧ mean scores_B = 30 ∧
  variance scores_A = 174 / 7 ∧ variance scores_B = 116 / 7 ∧
  variance scores_A > variance scores_B := 
by
  sorry

-- Define the probabilities of scoring higher than 30
def probability_excellent (scores : List ℕ) : ℚ := 
  (scores.filter (λ x => x > 30)).length / scores.length

theorem excellent_performance_probability :
  probability_excellent scores_A = 3 / 7 ∧ probability_excellent scores_B = 2 / 7 ∧
  (probability_excellent scores_A * probability_excellent scores_B = 6 / 49) :=
by
  sorry

end NUMINAMATH_GPT_avg_variance_stability_excellent_performance_probability_l1079_107937


namespace NUMINAMATH_GPT_cost_of_12_cheaper_fruits_l1079_107967

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

end NUMINAMATH_GPT_cost_of_12_cheaper_fruits_l1079_107967


namespace NUMINAMATH_GPT_triangle_third_side_l1079_107984

theorem triangle_third_side (a b x : ℝ) (h : (a - 3) ^ 2 + |b - 4| = 0) :
  x = 5 ∨ x = Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_triangle_third_side_l1079_107984


namespace NUMINAMATH_GPT_time_to_pass_pole_l1079_107953

def train_length : ℕ := 250
def platform_length : ℕ := 1250
def time_to_pass_platform : ℕ := 60

theorem time_to_pass_pole : 
  (train_length + platform_length) / time_to_pass_platform * train_length = 10 :=
by
  sorry

end NUMINAMATH_GPT_time_to_pass_pole_l1079_107953


namespace NUMINAMATH_GPT_penthouse_units_l1079_107906

theorem penthouse_units (total_floors : ℕ) (regular_units_per_floor : ℕ) (penthouse_floors : ℕ) (total_units : ℕ) :
  total_floors = 23 →
  regular_units_per_floor = 12 →
  penthouse_floors = 2 →
  total_units = 256 →
  (total_units - (total_floors - penthouse_floors) * regular_units_per_floor) / penthouse_floors = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_penthouse_units_l1079_107906


namespace NUMINAMATH_GPT_carrots_total_l1079_107969

theorem carrots_total 
  (picked_1 : Nat) 
  (thrown_out : Nat) 
  (picked_2 : Nat) 
  (total_carrots : Nat) 
  (h_picked1 : picked_1 = 23) 
  (h_thrown_out : thrown_out = 10) 
  (h_picked2 : picked_2 = 47) : 
  total_carrots = 60 := 
by
  sorry

end NUMINAMATH_GPT_carrots_total_l1079_107969


namespace NUMINAMATH_GPT_consecutive_even_integer_bases_l1079_107903

/-- Given \(X\) and \(Y\) are consecutive even positive integers and the equation
\[ 241_X + 36_Y = 94_{X+Y} \]
this theorem proves that \(X + Y = 22\). -/
theorem consecutive_even_integer_bases (X Y : ℕ) (h1 : X > 0) (h2 : Y = X + 2)
    (h3 : 2 * X^2 + 4 * X + 1 + 3 * Y + 6 = 9 * (X + Y) + 4) : 
    X + Y = 22 :=
by sorry

end NUMINAMATH_GPT_consecutive_even_integer_bases_l1079_107903


namespace NUMINAMATH_GPT_fraction_check_l1079_107939

variable (a b x y : ℝ)
noncomputable def is_fraction (expr : ℝ) : Prop :=
∃ n d : ℝ, d ≠ 0 ∧ expr = n / d ∧ ∃ var : ℝ, d = var

theorem fraction_check :
  is_fraction ((x + 3) / x) :=
sorry

end NUMINAMATH_GPT_fraction_check_l1079_107939


namespace NUMINAMATH_GPT_cost_percentage_l1079_107952

-- Define the original and new costs
def original_cost (t b : ℝ) : ℝ := t * b^4
def new_cost (t b : ℝ) : ℝ := t * (2 * b)^4

-- Define the theorem to prove the percentage relationship
theorem cost_percentage (t b : ℝ) (C R : ℝ) (h1 : C = original_cost t b) (h2 : R = new_cost t b) :
  (R / C) * 100 = 1600 :=
by sorry

end NUMINAMATH_GPT_cost_percentage_l1079_107952


namespace NUMINAMATH_GPT_quadratic_roots_real_and_equal_l1079_107988

theorem quadratic_roots_real_and_equal (m : ℤ) :
  (∀ x : ℝ, 3 * x^2 + (2 - m) * x + 12 = 0 →
   (∃ r, x = r ∧ 3 * r^2 + (2 - m) * r + 12 = 0)) →
   (m = -10 ∨ m = 14) :=
sorry

end NUMINAMATH_GPT_quadratic_roots_real_and_equal_l1079_107988


namespace NUMINAMATH_GPT_whitney_total_cost_l1079_107930

-- Definitions of the number of items and their costs
def w := 15
def c_w := 14
def f := 12
def c_f := 13
def s := 5
def c_s := 10
def m := 8
def c_m := 3

-- The total cost Whitney spent
theorem whitney_total_cost :
  w * c_w + f * c_f + s * c_s + m * c_m = 440 := by
  sorry

end NUMINAMATH_GPT_whitney_total_cost_l1079_107930


namespace NUMINAMATH_GPT_solve_for_x_l1079_107976

theorem solve_for_x (x : ℝ) (h : 3 - 1 / (1 - x) = 2 * (1 / (1 - x))) : x = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l1079_107976


namespace NUMINAMATH_GPT_num_girls_on_trip_l1079_107997

/-- Given the conditions: 
  * Three adults each eating 3 eggs.
  * Ten boys each eating one more egg than each girl.
  * A total of 36 eggs.
  Prove that there are 7 girls on the trip. -/
theorem num_girls_on_trip (adults boys girls eggs : ℕ) 
  (H1 : adults = 3)
  (H2 : boys = 10)
  (H3 : eggs = 36)
  (H4 : ∀ g, (girls * g) + (boys * (g + 1)) + (adults * 3) = eggs)
  (H5 : ∀ g, g = 1) :
  girls = 7 :=
by
  sorry

end NUMINAMATH_GPT_num_girls_on_trip_l1079_107997


namespace NUMINAMATH_GPT_greatest_possible_triangle_perimeter_l1079_107978

noncomputable def triangle_perimeter (x : ℕ) : ℕ :=
  x + 2 * x + 18

theorem greatest_possible_triangle_perimeter :
  (∃ (x : ℕ), 7 ≤ x ∧ x < 18 ∧ ∀ y : ℕ, (7 ≤ y ∧ y < 18) → triangle_perimeter y ≤ triangle_perimeter x) ∧
  triangle_perimeter 17 = 69 :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_triangle_perimeter_l1079_107978


namespace NUMINAMATH_GPT_grape_rate_per_kg_l1079_107956

theorem grape_rate_per_kg (G : ℝ) : 
    (8 * G) + (9 * 55) = 1055 → G = 70 := by
  sorry

end NUMINAMATH_GPT_grape_rate_per_kg_l1079_107956


namespace NUMINAMATH_GPT_length_of_chord_l1079_107971

theorem length_of_chord 
  (a : ℝ)
  (h_sym : ∀ (x y : ℝ), (x^2 + y^2 - 2*x + 4*y = 0) → (3*x - a*y - 11 = 0))
  (h_line : 3 * 1 - a * (-2) - 11 = 0)
  (h_midpoint : (1 : ℝ) = (a / 4) ∧ (-1 : ℝ) = (-a / 4)) :
  let r := Real.sqrt 5
  let d := Real.sqrt ((1 - 1)^2 + (-1 + 2)^2)
  (2 * Real.sqrt (r^2 - d^2)) = 4 :=
by {
  -- Variables and assumptions would go here
  sorry
}

end NUMINAMATH_GPT_length_of_chord_l1079_107971


namespace NUMINAMATH_GPT_max_notebooks_no_more_than_11_l1079_107974

noncomputable def maxNotebooks (money : ℕ) (cost_single : ℕ) (cost_pack4 : ℕ) (cost_pack7 : ℕ) (max_pack7 : ℕ) : ℕ :=
if money >= cost_pack7 then
  if (money - cost_pack7) >= cost_pack4 then 7 + 4
  else if (money - cost_pack7) >= cost_single then 7 + 1
  else 7
else if money >= cost_pack4 then
  if (money - cost_pack4) >= cost_pack4 then 4 + 4
  else if (money - cost_pack4) >= cost_single then 4 + 1
  else 4
else
  money / cost_single

theorem max_notebooks_no_more_than_11 :
  maxNotebooks 15 2 6 9 1 = 11 :=
by
  sorry

end NUMINAMATH_GPT_max_notebooks_no_more_than_11_l1079_107974


namespace NUMINAMATH_GPT_largest_side_of_enclosure_l1079_107945

-- Definitions for the conditions
def perimeter (l w : ℝ) : ℝ := 2 * l + 2 * w
def area (l w : ℝ) : ℝ := l * w

theorem largest_side_of_enclosure (l w : ℝ) (h_fencing : perimeter l w = 240) (h_area : area l w = 12 * 240) : l = 86.83 ∨ w = 86.83 :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_side_of_enclosure_l1079_107945


namespace NUMINAMATH_GPT_sandwiches_difference_l1079_107965

theorem sandwiches_difference :
  let monday_lunch := 3
  let monday_dinner := 2 * monday_lunch
  let monday_total := monday_lunch + monday_dinner

  let tuesday_lunch := 4
  let tuesday_dinner := tuesday_lunch / 2
  let tuesday_total := tuesday_lunch + tuesday_dinner

  let wednesday_lunch := 2 * tuesday_lunch
  let wednesday_dinner := 3 * tuesday_lunch
  let wednesday_total := wednesday_lunch + wednesday_dinner

  let total_mw := monday_total + tuesday_total + wednesday_total

  let thursday_lunch := 3 * 2
  let thursday_dinner := 5
  let thursday_total := thursday_lunch + thursday_dinner

  total_mw - thursday_total = 24 :=
by
  sorry

end NUMINAMATH_GPT_sandwiches_difference_l1079_107965


namespace NUMINAMATH_GPT_find_3a_plus_3b_l1079_107963

theorem find_3a_plus_3b (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 8 * a + 2 * b = 50) :
  3 * a + 3 * b = 73 / 2 := 
sorry

end NUMINAMATH_GPT_find_3a_plus_3b_l1079_107963


namespace NUMINAMATH_GPT_sqrt_floor_squared_l1079_107980

theorem sqrt_floor_squared (h1 : 7^2 = 49) (h2 : 8^2 = 64) (h3 : 7 < Real.sqrt 50) (h4 : Real.sqrt 50 < 8) : (Int.floor (Real.sqrt 50))^2 = 49 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_floor_squared_l1079_107980


namespace NUMINAMATH_GPT_customs_days_l1079_107940

-- Definitions from the problem conditions
def navigation_days : ℕ := 21
def transport_days : ℕ := 7
def total_days : ℕ := 30

-- Proposition we need to prove
theorem customs_days (expected_days: ℕ) (ship_departure_days : ℕ) : expected_days = 2 → ship_departure_days = 30 → (navigation_days + expected_days + transport_days = total_days) → expected_days = 2 :=
by
  intros h_expected h_departure h_eq
  sorry

end NUMINAMATH_GPT_customs_days_l1079_107940


namespace NUMINAMATH_GPT_stratified_sampling_second_year_students_l1079_107944

theorem stratified_sampling_second_year_students 
  (total_athletes : ℕ) 
  (first_year_students : ℕ) 
  (sample_size : ℕ) 
  (second_year_students_in_sample : ℕ)
  (h1 : total_athletes = 98) 
  (h2 : first_year_students = 56) 
  (h3 : sample_size = 28)
  (h4 : second_year_students_in_sample = (42 * sample_size) / total_athletes) :
  second_year_students_in_sample = 4 := 
sorry

end NUMINAMATH_GPT_stratified_sampling_second_year_students_l1079_107944


namespace NUMINAMATH_GPT_simplify_expression_l1079_107936

theorem simplify_expression :
  8 * (15 / 4) * (-45 / 50) = - (12 / 25) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1079_107936


namespace NUMINAMATH_GPT_find_expression_l1079_107919

theorem find_expression : 1^567 + 3^5 / 3^3 - 2 = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_expression_l1079_107919


namespace NUMINAMATH_GPT_max_sequence_is_ten_l1079_107926

noncomputable def max_int_sequence_length : Prop :=
  ∀ (a : ℕ → ℤ), 
    (∀ i : ℕ, a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) > 0) ∧
    (∀ i : ℕ, a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) + a (i+5) + a (i+6) < 0) →
    (∃ n ≤ 10, ∀ i ≥ n, a i = 0)

theorem max_sequence_is_ten : max_int_sequence_length :=
sorry

end NUMINAMATH_GPT_max_sequence_is_ten_l1079_107926


namespace NUMINAMATH_GPT_unique_triple_l1079_107949

def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def find_triples (x y z : ℕ) : Prop :=
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  is_prime x ∧ is_prime y ∧ is_prime z ∧
  is_prime (x - y) ∧ is_prime (y - z) ∧ is_prime (x - z)

theorem unique_triple :
  ∀ (x y z : ℕ), find_triples x y z → (x, y, z) = (7, 5, 2) :=
by
  sorry

end NUMINAMATH_GPT_unique_triple_l1079_107949


namespace NUMINAMATH_GPT_avg_hamburgers_per_day_l1079_107902

theorem avg_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h1 : total_hamburgers = 63) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 := by
  sorry

end NUMINAMATH_GPT_avg_hamburgers_per_day_l1079_107902


namespace NUMINAMATH_GPT_inequality_non_empty_solution_set_l1079_107917

theorem inequality_non_empty_solution_set (a : ℝ) : ∃ x : ℝ, ax^2 - (a-2)*x - 2 ≤ 0 :=
sorry

end NUMINAMATH_GPT_inequality_non_empty_solution_set_l1079_107917


namespace NUMINAMATH_GPT_initial_milk_amount_l1079_107955

theorem initial_milk_amount (M : ℝ) (H1 : 0.05 * M = 0.02 * (M + 15)) : M = 10 :=
by
  sorry

end NUMINAMATH_GPT_initial_milk_amount_l1079_107955


namespace NUMINAMATH_GPT_product_percent_x_l1079_107929

variables {x y z w : ℝ}
variables (h1 : 0.45 * z = 1.2 * y) 
variables (h2 : y = 0.75 * x) 
variables (h3 : z = 0.8 * w)

theorem product_percent_x :
  (w * y) / x = 1.875 :=
by 
  sorry

end NUMINAMATH_GPT_product_percent_x_l1079_107929


namespace NUMINAMATH_GPT_unused_sector_angle_l1079_107985

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h
noncomputable def slant_height (r h : ℝ) : ℝ := Real.sqrt (r^2 + h^2)
noncomputable def central_angle (r base_circumference : ℝ) : ℝ := (base_circumference / (2 * Real.pi * r)) * 360
noncomputable def unused_angle (total_degrees used_angle : ℝ) : ℝ := total_degrees - used_angle

theorem unused_sector_angle (R : ℝ)
  (cone_radius := 15)
  (cone_volume := 675 * Real.pi)
  (total_circumference := 2 * Real.pi * R)
  (cone_height := 9)
  (slant_height := Real.sqrt (cone_radius^2 + cone_height^2))
  (base_circumference := 2 * Real.pi * cone_radius)
  (used_angle := central_angle slant_height base_circumference) :

  unused_angle 360 used_angle = 164.66 := by
  sorry

end NUMINAMATH_GPT_unused_sector_angle_l1079_107985


namespace NUMINAMATH_GPT_negation_of_P_is_there_exists_x_ge_0_l1079_107989

-- Define the proposition P
def P : Prop := ∀ x : ℝ, x^2 + x - 1 < 0

-- State the theorem of the negation of P
theorem negation_of_P_is_there_exists_x_ge_0 : ¬P ↔ ∃ x : ℝ, x^2 + x - 1 ≥ 0 :=
by sorry

end NUMINAMATH_GPT_negation_of_P_is_there_exists_x_ge_0_l1079_107989


namespace NUMINAMATH_GPT_find_angle_ACB_l1079_107999

theorem find_angle_ACB
    (convex_quadrilateral : Prop)
    (angle_BAC : ℝ)
    (angle_CAD : ℝ)
    (angle_ADB : ℝ)
    (angle_BDC : ℝ)
    (h1 : convex_quadrilateral)
    (h2 : angle_BAC = 20)
    (h3 : angle_CAD = 60)
    (h4 : angle_ADB = 50)
    (h5 : angle_BDC = 10)
    : ∃ angle_ACB : ℝ, angle_ACB = 80 :=
by
  -- Here use sorry to skip the proof.
  sorry

end NUMINAMATH_GPT_find_angle_ACB_l1079_107999


namespace NUMINAMATH_GPT_range_of_a_l1079_107968

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x + y = 2 ∧ 
    (if x > 1 then (x^2 + 1) / x else Real.log (x + a)) = 
    (if y > 1 then (y^2 + 1) / y else Real.log (y + a))) ↔ 
    a > Real.exp 2 - 1 :=
by sorry

end NUMINAMATH_GPT_range_of_a_l1079_107968


namespace NUMINAMATH_GPT_mass_percentage_of_Cl_in_NaClO_l1079_107950

noncomputable def molarMassNa : ℝ := 22.99
noncomputable def molarMassCl : ℝ := 35.45
noncomputable def molarMassO : ℝ := 16.00

noncomputable def molarMassNaClO : ℝ := molarMassNa + molarMassCl + molarMassO

theorem mass_percentage_of_Cl_in_NaClO : 
  (molarMassCl / molarMassNaClO) * 100 = 47.61 :=
by 
  sorry

end NUMINAMATH_GPT_mass_percentage_of_Cl_in_NaClO_l1079_107950


namespace NUMINAMATH_GPT_tax_percentage_first_40000_l1079_107982

theorem tax_percentage_first_40000 (P : ℝ) :
  (0 < P) → 
  (P / 100) * 40000 + 0.20 * 10000 = 8000 →
  P = 15 :=
by
  intros hP h
  sorry

end NUMINAMATH_GPT_tax_percentage_first_40000_l1079_107982


namespace NUMINAMATH_GPT_inheritance_amount_l1079_107938

theorem inheritance_amount (x : ℝ) 
  (federal_tax : ℝ := 0.25 * x) 
  (state_tax : ℝ := 0.15 * (x - federal_tax)) 
  (city_tax : ℝ := 0.05 * (x - federal_tax - state_tax)) 
  (total_tax : ℝ := 20000) :
  (federal_tax + state_tax + city_tax = total_tax) → 
  x = 50704 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_inheritance_amount_l1079_107938


namespace NUMINAMATH_GPT_pieces_after_cuts_l1079_107995

theorem pieces_after_cuts (n : ℕ) : 
  (∃ n, (8 * n + 1 = 2009)) ↔ (n = 251) :=
by 
  sorry

end NUMINAMATH_GPT_pieces_after_cuts_l1079_107995


namespace NUMINAMATH_GPT_regression_line_is_y_eq_x_plus_1_l1079_107983

theorem regression_line_is_y_eq_x_plus_1 :
  let points : List (ℝ × ℝ) := [(1, 2), (2, 3), (3, 4), (4, 5)]
  ∃ (m b : ℝ), (∀ (x y : ℝ), (x, y) ∈ points → y = m * x + b) ∧ m = 1 ∧ b = 1 :=
by
  sorry 

end NUMINAMATH_GPT_regression_line_is_y_eq_x_plus_1_l1079_107983


namespace NUMINAMATH_GPT_total_amount_divided_l1079_107947

-- Define the conditions
variables (A B C : ℕ)
axiom h1 : 4 * A = 5 * B
axiom h2 : 4 * A = 10 * C
axiom h3 : C = 160

-- Define the theorem to prove the total amount
theorem total_amount_divided (h1 : 4 * A = 5 * B) (h2 : 4 * A = 10 * C) (h3 : C = 160) : 
  A + B + C = 880 :=
sorry

end NUMINAMATH_GPT_total_amount_divided_l1079_107947


namespace NUMINAMATH_GPT_distance_hyperbola_focus_to_line_l1079_107927

def hyperbola_right_focus : Type := { x : ℝ // x = 3 } -- Right focus is at (3, 0)
def line : Type := { x // x + 2 * (0 : ℝ) - 8 = 0 } -- Represents the line x + 2y - 8 = 0

theorem distance_hyperbola_focus_to_line : Real.sqrt 5 = 
  abs (1 * 3 + 2 * 0 - 8) / Real.sqrt (1^2 + 2^2) := 
by
  sorry

end NUMINAMATH_GPT_distance_hyperbola_focus_to_line_l1079_107927


namespace NUMINAMATH_GPT_max_k_l1079_107935

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

theorem max_k (k : ℤ) : (∀ x : ℝ, 1 < x → f x - k * x + k > 0) → k ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_max_k_l1079_107935


namespace NUMINAMATH_GPT_angle_measure_l1079_107921

theorem angle_measure (x : ℝ) 
  (h : x = 2 * (90 - x) - 60) : 
  x = 40 := 
  sorry

end NUMINAMATH_GPT_angle_measure_l1079_107921


namespace NUMINAMATH_GPT_set_intersection_l1079_107972

open Set

universe u

variables {U : Type u} (A B : Set ℝ) (x : ℝ)

def universal_set : Set ℝ := univ
def set_A : Set ℝ := {x | abs x < 1}
def set_B : Set ℝ := {x | x > -1/2}
def complement_B : Set ℝ := {x | x ≤ -1/2}
def intersection : Set ℝ := {x | -1 < x ∧ x ≤ -1/2}

theorem set_intersection :
  (universal_set \ set_B) ∩ set_A = {x | -1 < x ∧ x ≤ -1/2} :=
by 
  -- The actual proof steps would go here
  sorry

end NUMINAMATH_GPT_set_intersection_l1079_107972


namespace NUMINAMATH_GPT_max_value_of_a_l1079_107941

theorem max_value_of_a (a b c : ℕ) (h : a + b + c = Nat.gcd a b + Nat.gcd b c + Nat.gcd c a + 120) : a ≤ 240 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_a_l1079_107941


namespace NUMINAMATH_GPT_age_ratio_in_1_year_l1079_107990

variable (j m x : ℕ)

-- Conditions
def condition1 (j m : ℕ) : Prop :=
  j - 3 = 2 * (m - 3)

def condition2 (j m : ℕ) : Prop :=
  j - 5 = 3 * (m - 5)

-- Question
def age_ratio (j m x : ℕ) : Prop :=
  (j + x) * 2 = 3 * (m + x)

theorem age_ratio_in_1_year (j m x : ℕ) :
  condition1 j m → condition2 j m → age_ratio j m 1 :=
by
  sorry

end NUMINAMATH_GPT_age_ratio_in_1_year_l1079_107990


namespace NUMINAMATH_GPT_circular_garden_radius_l1079_107996

theorem circular_garden_radius (r : ℝ) (h : 2 * Real.pi * r = (1 / 8) * Real.pi * r^2) : r = 16 :=
sorry

end NUMINAMATH_GPT_circular_garden_radius_l1079_107996


namespace NUMINAMATH_GPT_hyperbola_eccentricity_condition_l1079_107943

theorem hyperbola_eccentricity_condition (m : ℝ) (h : m > 0) : 
  (∃ e : ℝ, e = Real.sqrt (1 + m) ∧ e > Real.sqrt 2) → m > 1 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_condition_l1079_107943


namespace NUMINAMATH_GPT_work_efficiency_l1079_107998

theorem work_efficiency (orig_time : ℝ) (new_time : ℝ) (work : ℝ) 
  (h1 : orig_time = 1)
  (h2 : new_time = orig_time * (1 - 0.20))
  (h3 : work = 1) :
  (orig_time / new_time) * 100 = 125 :=
by
  sorry

end NUMINAMATH_GPT_work_efficiency_l1079_107998


namespace NUMINAMATH_GPT_license_plates_count_l1079_107914

theorem license_plates_count :
  let letters := 26
  let digits := 10
  let odd_digits := 5
  let even_digits := 5
  (letters^3) * digits * (odd_digits + even_digits) = 878800 := by
  sorry

end NUMINAMATH_GPT_license_plates_count_l1079_107914


namespace NUMINAMATH_GPT_total_vehicles_l1079_107981

-- Define the conditions
def num_trucks_per_lane := 60
def num_lanes := 4
def total_trucks := num_trucks_per_lane * num_lanes
def num_cars_per_lane := 2 * total_trucks
def total_cars := num_cars_per_lane * num_lanes

-- Prove the total number of vehicles in all lanes
theorem total_vehicles : total_trucks + total_cars = 2160 := by
  sorry

end NUMINAMATH_GPT_total_vehicles_l1079_107981


namespace NUMINAMATH_GPT_avg_b_c_weight_l1079_107916

theorem avg_b_c_weight (a b c : ℝ) (H1 : (a + b + c) / 3 = 45) (H2 : (a + b) / 2 = 40) (H3 : b = 39) : (b + c) / 2 = 47 :=
by
  sorry

end NUMINAMATH_GPT_avg_b_c_weight_l1079_107916


namespace NUMINAMATH_GPT_marta_sold_on_saturday_l1079_107951

-- Definitions of conditions
def initial_shipment : ℕ := 1000
def rotten_tomatoes : ℕ := 200
def second_shipment : ℕ := 2000
def tomatoes_on_tuesday : ℕ := 2500
def x := 300

-- Total tomatoes on Monday after the second shipment
def tomatoes_on_monday (sold_tomatoes : ℕ) : ℕ :=
  initial_shipment - sold_tomatoes - rotten_tomatoes + second_shipment

-- Theorem statement to prove
theorem marta_sold_on_saturday : (tomatoes_on_monday x = tomatoes_on_tuesday) -> (x = 300) :=
by 
  intro h
  sorry

end NUMINAMATH_GPT_marta_sold_on_saturday_l1079_107951


namespace NUMINAMATH_GPT_ellipse_with_foci_on_x_axis_l1079_107918

theorem ellipse_with_foci_on_x_axis (a : ℝ) :
  (∀ x y : ℝ, (x^2) / (a - 5) + (y^2) / 2 = 1 →  
   (∃ cx cy : ℝ, ∀ x', cx - x' = a - 5 ∧ cy = 2)) → 
  a > 7 :=
by sorry

end NUMINAMATH_GPT_ellipse_with_foci_on_x_axis_l1079_107918


namespace NUMINAMATH_GPT_simplify_expression_l1079_107907

theorem simplify_expression (x y : ℝ) (h : y = x / (1 - 2 * x)) :
    (2 * x - 3 * x * y - 2 * y) / (y + x * y - x) = -7 / 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_simplify_expression_l1079_107907


namespace NUMINAMATH_GPT_Namjoon_gave_Yoongi_9_pencils_l1079_107924

theorem Namjoon_gave_Yoongi_9_pencils
  (stroke_pencils : ℕ)
  (strokes : ℕ)
  (pencils_left : ℕ)
  (total_pencils : ℕ := stroke_pencils * strokes)
  (given_pencils : ℕ := total_pencils - pencils_left) :
  stroke_pencils = 12 →
  strokes = 2 →
  pencils_left = 15 →
  given_pencils = 9 := by
  sorry

end NUMINAMATH_GPT_Namjoon_gave_Yoongi_9_pencils_l1079_107924


namespace NUMINAMATH_GPT_length_of_field_l1079_107920

def width : ℝ := 13.5

def length (w : ℝ) : ℝ := 2 * w - 3

theorem length_of_field : length width = 24 :=
by
  -- full proof goes here
  sorry

end NUMINAMATH_GPT_length_of_field_l1079_107920


namespace NUMINAMATH_GPT_max_value_90_l1079_107958

noncomputable def max_value_expression (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_90 (a b c d : ℝ) (h₁ : -4.5 ≤ a) (h₂ : a ≤ 4.5)
                                   (h₃ : -4.5 ≤ b) (h₄ : b ≤ 4.5)
                                   (h₅ : -4.5 ≤ c) (h₆ : c ≤ 4.5)
                                   (h₇ : -4.5 ≤ d) (h₈ : d ≤ 4.5) :
  max_value_expression a b c d ≤ 90 :=
sorry

end NUMINAMATH_GPT_max_value_90_l1079_107958


namespace NUMINAMATH_GPT_remainder_of_7_pow_7_pow_7_pow_7_mod_500_l1079_107992

theorem remainder_of_7_pow_7_pow_7_pow_7_mod_500 :
    (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 := 
sorry

end NUMINAMATH_GPT_remainder_of_7_pow_7_pow_7_pow_7_mod_500_l1079_107992


namespace NUMINAMATH_GPT_paint_cost_of_cube_l1079_107915

def cube_side_length : ℝ := 10
def paint_cost_per_quart : ℝ := 3.20
def coverage_per_quart : ℝ := 1200
def number_of_faces : ℕ := 6

theorem paint_cost_of_cube : 
  (number_of_faces * (cube_side_length^2) / coverage_per_quart) * paint_cost_per_quart = 3.20 :=
by 
  sorry

end NUMINAMATH_GPT_paint_cost_of_cube_l1079_107915


namespace NUMINAMATH_GPT_discounted_price_per_bag_l1079_107961

theorem discounted_price_per_bag
  (cost_per_bag : ℝ)
  (num_bags : ℕ)
  (initial_price : ℝ)
  (num_sold_initial : ℕ)
  (net_profit : ℝ)
  (discounted_revenue : ℝ)
  (discounted_price : ℝ) :
  cost_per_bag = 3.0 →
  num_bags = 20 →
  initial_price = 6.0 →
  num_sold_initial = 15 →
  net_profit = 50 →
  discounted_revenue = (net_profit + (num_bags * cost_per_bag) - (num_sold_initial * initial_price) ) →
  discounted_price = (discounted_revenue / (num_bags - num_sold_initial)) →
  discounted_price = 4.0 :=
by
  sorry

end NUMINAMATH_GPT_discounted_price_per_bag_l1079_107961


namespace NUMINAMATH_GPT_distance_covered_is_9_17_miles_l1079_107913

noncomputable def totalDistanceCovered 
  (walkingTimeInMinutes : ℕ) (walkingRate : ℝ)
  (runningTimeInMinutes : ℕ) (runningRate : ℝ)
  (cyclingTimeInMinutes : ℕ) (cyclingRate : ℝ) : ℝ :=
  (walkingRate * (walkingTimeInMinutes / 60.0)) + 
  (runningRate * (runningTimeInMinutes / 60.0)) + 
  (cyclingRate * (cyclingTimeInMinutes / 60.0))

theorem distance_covered_is_9_17_miles :
  totalDistanceCovered 30 3 20 8 25 12 = 9.17 := 
by 
  sorry

end NUMINAMATH_GPT_distance_covered_is_9_17_miles_l1079_107913


namespace NUMINAMATH_GPT_two_pow_2023_add_three_pow_2023_mod_seven_not_zero_l1079_107908

theorem two_pow_2023_add_three_pow_2023_mod_seven_not_zero : (2^2023 + 3^2023) % 7 ≠ 0 := 
by sorry

end NUMINAMATH_GPT_two_pow_2023_add_three_pow_2023_mod_seven_not_zero_l1079_107908


namespace NUMINAMATH_GPT_equalize_foma_ierema_l1079_107901

variables 
  (F E Y : ℕ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)

def foma_should_give_ierema : ℕ := (F - E) / 2

theorem equalize_foma_ierema (F E Y : ℕ) (h1 : F - 70 = E + 70) (h2 : F - 40 = Y) :
  foma_should_give_ierema F E = 55 := 
by
  sorry

end NUMINAMATH_GPT_equalize_foma_ierema_l1079_107901


namespace NUMINAMATH_GPT_find_a_l1079_107933

theorem find_a (a : ℝ) (α : ℝ) (P : ℝ × ℝ) 
  (h_P : P = (3 * a, 4)) 
  (h_cos : Real.cos α = -3/5) : 
  a = -1 := 
by
  sorry

end NUMINAMATH_GPT_find_a_l1079_107933


namespace NUMINAMATH_GPT_second_year_associates_l1079_107962

theorem second_year_associates (total_associates : ℕ) (not_first_year : ℕ) (more_than_two_years : ℕ) 
  (h1 : not_first_year = 60 * total_associates / 100) 
  (h2 : more_than_two_years = 30 * total_associates / 100) :
  not_first_year - more_than_two_years = 30 * total_associates / 100 :=
by
  sorry

end NUMINAMATH_GPT_second_year_associates_l1079_107962


namespace NUMINAMATH_GPT_four_digit_cubes_divisible_by_16_l1079_107942

theorem four_digit_cubes_divisible_by_16 (n : ℕ) : 
  1000 ≤ (4 * n)^3 ∧ (4 * n)^3 ≤ 9999 ∧ (4 * n)^3 % 16 = 0 ↔ n = 4 ∨ n = 5 := 
sorry

end NUMINAMATH_GPT_four_digit_cubes_divisible_by_16_l1079_107942


namespace NUMINAMATH_GPT_movie_theater_loss_l1079_107979

theorem movie_theater_loss :
  let capacity := 50
  let ticket_price := 8.0
  let tickets_sold := 24
  (capacity * ticket_price - tickets_sold * ticket_price) = 208 := by
  sorry

end NUMINAMATH_GPT_movie_theater_loss_l1079_107979


namespace NUMINAMATH_GPT_expression_eval_l1079_107970

theorem expression_eval : (-4)^7 / 4^5 + 5^3 * 2 - 7^2 = 185 := by
  sorry

end NUMINAMATH_GPT_expression_eval_l1079_107970


namespace NUMINAMATH_GPT_exists_positive_integer_m_l1079_107905

theorem exists_positive_integer_m (a b c d : ℝ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0) (hpos_d : d > 0) (h_cd : c * d = 1) : 
  ∃ m : ℕ, (a * b ≤ ↑m * ↑m) ∧ (↑m * ↑m ≤ (a + c) * (b + d)) :=
by
  sorry

end NUMINAMATH_GPT_exists_positive_integer_m_l1079_107905


namespace NUMINAMATH_GPT_john_ingrid_combined_weighted_average_tax_rate_l1079_107932

noncomputable def john_employment_income : ℕ := 57000
noncomputable def john_employment_tax_rate : ℚ := 0.30
noncomputable def john_rental_income : ℕ := 11000
noncomputable def john_rental_tax_rate : ℚ := 0.25

noncomputable def ingrid_employment_income : ℕ := 72000
noncomputable def ingrid_employment_tax_rate : ℚ := 0.40
noncomputable def ingrid_investment_income : ℕ := 4500
noncomputable def ingrid_investment_tax_rate : ℚ := 0.15

noncomputable def combined_weighted_average_tax_rate : ℚ :=
  let john_total_tax := john_employment_income * john_employment_tax_rate + john_rental_income * john_rental_tax_rate
  let john_total_income := john_employment_income + john_rental_income
  let ingrid_total_tax := ingrid_employment_income * ingrid_employment_tax_rate + ingrid_investment_income * ingrid_investment_tax_rate
  let ingrid_total_income := ingrid_employment_income + ingrid_investment_income
  let combined_total_tax := john_total_tax + ingrid_total_tax
  let combined_total_income := john_total_income + ingrid_total_income
  (combined_total_tax / combined_total_income) * 100

theorem john_ingrid_combined_weighted_average_tax_rate :
  combined_weighted_average_tax_rate = 34.14 := by
  sorry

end NUMINAMATH_GPT_john_ingrid_combined_weighted_average_tax_rate_l1079_107932


namespace NUMINAMATH_GPT_sum_proper_divisors_243_l1079_107904

theorem sum_proper_divisors_243 : (1 + 3 + 9 + 27 + 81) = 121 :=
by
  sorry

end NUMINAMATH_GPT_sum_proper_divisors_243_l1079_107904


namespace NUMINAMATH_GPT_NaNO3_moles_l1079_107928

theorem NaNO3_moles (moles_NaCl moles_HNO3 moles_NaNO3 : ℝ) (h_HNO3 : moles_HNO3 = 2) (h_ratio : moles_NaNO3 = moles_NaCl) (h_NaNO3 : moles_NaNO3 = 2) :
  moles_NaNO3 = 2 :=
sorry

end NUMINAMATH_GPT_NaNO3_moles_l1079_107928


namespace NUMINAMATH_GPT_value_of_a_l1079_107975

-- Definitions of sets A and B
def A : Set ℝ := {x | x^2 = 1}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

-- The main theorem statement
theorem value_of_a (a : ℝ) (H : B a ⊆ A) : a = 0 ∨ a = 1 ∨ a = -1 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_a_l1079_107975


namespace NUMINAMATH_GPT_new_person_weight_l1079_107964

theorem new_person_weight (W : ℝ) (N : ℝ) (avg_increase : ℝ := 2.5) (replaced_weight : ℝ := 35) :
  (W - replaced_weight + N) = (W + (8 * avg_increase)) → N = 55 := sorry

end NUMINAMATH_GPT_new_person_weight_l1079_107964


namespace NUMINAMATH_GPT_value_of_w_l1079_107954

theorem value_of_w (x : ℝ) (hx : x + 1/x = 5) : x^2 + (1/x)^2 = 23 :=
by
  sorry

end NUMINAMATH_GPT_value_of_w_l1079_107954


namespace NUMINAMATH_GPT_increasing_interval_when_a_neg_increasing_and_decreasing_intervals_when_a_pos_l1079_107911

noncomputable def f (a x : ℝ) : ℝ := x - a / x

theorem increasing_interval_when_a_neg {a : ℝ} (h : a < 0) :
  ∀ x : ℝ, x > 0 → f a x > 0 :=
sorry

theorem increasing_and_decreasing_intervals_when_a_pos {a : ℝ} (h : a > 0) :
  (∀ x : ℝ, 0 < x → x < Real.sqrt a → f a x < 0) ∧
  (∀ x : ℝ, x > Real.sqrt a → f a x > 0) :=
sorry

end NUMINAMATH_GPT_increasing_interval_when_a_neg_increasing_and_decreasing_intervals_when_a_pos_l1079_107911


namespace NUMINAMATH_GPT_find_root_of_equation_l1079_107900

theorem find_root_of_equation (a b c d x : ℕ) (h_ad : a + d = 2016) (h_bc : b + c = 2016) (h_ac : a ≠ c) :
  (x - a) * (x - b) = (x - c) * (x - d) → x = 1008 :=
by
  sorry

end NUMINAMATH_GPT_find_root_of_equation_l1079_107900


namespace NUMINAMATH_GPT_arithmetic_sequence_solution_l1079_107977

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
∀ n : ℕ, a n = a1 + n * d

noncomputable def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
n * a 0 + (n * (n - 1) / 2) * (a 1 - a 0)

theorem arithmetic_sequence_solution :
  ∃ d : ℤ,
  (∀ n : ℕ, n > 0 ∧ n < 10 → a n = 23 + n * d) ∧
  (23 + 5 * d > 0) ∧
  (23 + 6 * d < 0) ∧
  d = -4 ∧
  S a 6 = 78 ∧
  ∀ n : ℕ, S a n > 0 → n ≤ 12 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_solution_l1079_107977


namespace NUMINAMATH_GPT_lawrence_walking_speed_l1079_107994

theorem lawrence_walking_speed :
  let distance := 4
  let time := (4 : ℝ) / 3
  let speed := distance / time
  speed = 3 := 
by
  sorry

end NUMINAMATH_GPT_lawrence_walking_speed_l1079_107994


namespace NUMINAMATH_GPT_henrys_friend_money_l1079_107957

theorem henrys_friend_money (h1 h2 : ℕ) (T : ℕ) (f : ℕ) : h1 = 5 → h2 = 2 → T = 20 → h1 + h2 + f = T → f = 13 :=
by
  intros h1_eq h2_eq T_eq total_eq
  rw [h1_eq, h2_eq, T_eq] at total_eq
  sorry

end NUMINAMATH_GPT_henrys_friend_money_l1079_107957


namespace NUMINAMATH_GPT_packets_of_sugar_per_week_l1079_107934

theorem packets_of_sugar_per_week (total_grams : ℕ) (packet_weight : ℕ) (total_packets : ℕ) :
  total_grams = 2000 →
  packet_weight = 100 →
  total_packets = total_grams / packet_weight →
  total_packets = 20 := 
  by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3 

end NUMINAMATH_GPT_packets_of_sugar_per_week_l1079_107934


namespace NUMINAMATH_GPT_points_on_parabola_l1079_107986

theorem points_on_parabola (s : ℝ) : ∃ (u v : ℝ), u = 3^s - 4 ∧ v = 9^s - 7 * 3^s - 2 ∧ v = u^2 + u - 14 :=
by
  sorry

end NUMINAMATH_GPT_points_on_parabola_l1079_107986


namespace NUMINAMATH_GPT_razorback_tshirts_sold_l1079_107991

variable (T : ℕ) -- Number of t-shirts sold
variable (price_per_tshirt : ℕ := 62) -- Price of each t-shirt
variable (total_revenue : ℕ := 11346) -- Total revenue from t-shirts

theorem razorback_tshirts_sold :
  (price_per_tshirt * T = total_revenue) → T = 183 :=
by
  sorry

end NUMINAMATH_GPT_razorback_tshirts_sold_l1079_107991


namespace NUMINAMATH_GPT_find_other_number_l1079_107910

theorem find_other_number (a b : ℕ) (h1 : a + b = 62) (h2 : b - a = 12) (h3 : a = 25) : b = 37 :=
sorry

end NUMINAMATH_GPT_find_other_number_l1079_107910


namespace NUMINAMATH_GPT_expected_rice_yield_l1079_107925

theorem expected_rice_yield (x : ℝ) (y : ℝ) (h : y = 5 * x + 250) (hx : x = 80) : y = 650 :=
by
  sorry

end NUMINAMATH_GPT_expected_rice_yield_l1079_107925


namespace NUMINAMATH_GPT_quadratic_has_sum_r_s_l1079_107966

/-
  Define the quadratic equation 6x^2 - 24x - 54 = 0
-/
def quadratic_eq (x : ℝ) : Prop :=
  6 * x^2 - 24 * x - 54 = 0

/-
  Define the value 11 which is the sum r + s when completing the square
  for the above quadratic equation  
-/
def result_value := -2 + 13

/-
  State the proof that r + s = 11 given the quadratic equation.
-/
theorem quadratic_has_sum_r_s : ∀ x : ℝ, quadratic_eq x → -2 + 13 = 11 :=
by
  intros
  exact rfl

end NUMINAMATH_GPT_quadratic_has_sum_r_s_l1079_107966


namespace NUMINAMATH_GPT_range_of_m_l1079_107987

noncomputable def A := {x : ℝ | x^2 - 3 * x + 2 = 0}
noncomputable def B (m : ℝ) := {x : ℝ | x^2 - m * x + 2 = 0}

theorem range_of_m (m : ℝ) (h : ∀ x, x ∈ B m → x ∈ A) : m = 3 ∨ -2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1079_107987


namespace NUMINAMATH_GPT_football_team_progress_l1079_107946

theorem football_team_progress : 
  ∀ {loss gain : ℤ}, loss = 5 → gain = 11 → gain - loss = 6 :=
by
  intros loss gain h_loss h_gain
  rw [h_loss, h_gain]
  sorry

end NUMINAMATH_GPT_football_team_progress_l1079_107946


namespace NUMINAMATH_GPT_cost_calculation_l1079_107909

variables (H M F : ℝ)

theorem cost_calculation 
  (h1 : 3 * H + 5 * M + F = 23.50) 
  (h2 : 5 * H + 9 * M + F = 39.50) : 
  2 * H + 2 * M + 2 * F = 15.00 :=
sorry

end NUMINAMATH_GPT_cost_calculation_l1079_107909


namespace NUMINAMATH_GPT_semicircle_perimeter_l1079_107912

-- Assuming π as 3.14 for approximation
def π_approx : ℝ := 3.14

-- Radius of the semicircle
def radius : ℝ := 2.1

-- Half of the circumference
def half_circumference (r : ℝ) : ℝ := π_approx * r

-- Diameter of the semicircle
def diameter (r : ℝ) : ℝ := 2 * r

-- Perimeter of the semicircle
def perimeter (r : ℝ) : ℝ := half_circumference r + diameter r

-- Theorem stating the perimeter of the semicircle with given radius
theorem semicircle_perimeter : perimeter radius = 10.794 := by
  sorry

end NUMINAMATH_GPT_semicircle_perimeter_l1079_107912


namespace NUMINAMATH_GPT_average_possible_k_l1079_107973

theorem average_possible_k (k : ℕ) (r1 r2 : ℕ) (h : r1 * r2 = 24) (h_pos : r1 > 0 ∧ r2 > 0) (h_eq_k : r1 + r2 = k) : 
  (25 + 14 + 11 + 10) / 4 = 15 :=
by 
  sorry

end NUMINAMATH_GPT_average_possible_k_l1079_107973
