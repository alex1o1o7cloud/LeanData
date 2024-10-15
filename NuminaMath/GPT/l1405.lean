import Mathlib

namespace NUMINAMATH_GPT_find_angle4_l1405_140549

theorem find_angle4 (angle1 angle2 angle3 angle4 : ℝ)
                    (h1 : angle1 + angle2 = 180)
                    (h2 : angle3 = 2 * angle4)
                    (h3 : angle1 = 50)
                    (h4 : angle3 + angle4 = 130) : 
                    angle4 = 130 / 3 := by 
    sorry

end NUMINAMATH_GPT_find_angle4_l1405_140549


namespace NUMINAMATH_GPT_angle_in_fourth_quadrant_l1405_140521

theorem angle_in_fourth_quadrant (θ : ℝ) (hθ : θ = 300) : 270 < θ ∧ θ < 360 :=
by
  -- theta equals 300
  have h1 : θ = 300 := hθ
  -- check that 300 degrees lies between 270 and 360
  sorry

end NUMINAMATH_GPT_angle_in_fourth_quadrant_l1405_140521


namespace NUMINAMATH_GPT_max_xy_l1405_140582

theorem max_xy : 
  ∃ x y : ℕ, 5 * x + 3 * y = 100 ∧ x > 0 ∧ y > 0 ∧ x * y = 165 :=
by
  sorry

end NUMINAMATH_GPT_max_xy_l1405_140582


namespace NUMINAMATH_GPT_lily_pads_doubling_l1405_140558

theorem lily_pads_doubling (patch_half_day: ℕ) (doubling_rate: ℝ)
  (H1: patch_half_day = 49)
  (H2: doubling_rate = 2): (patch_half_day + 1) = 50 :=
by 
  sorry

end NUMINAMATH_GPT_lily_pads_doubling_l1405_140558


namespace NUMINAMATH_GPT_mark_gpa_probability_l1405_140593

theorem mark_gpa_probability :
  let A_points := 4
  let B_points := 3
  let C_points := 2
  let D_points := 1
  let GPA_required := 3.5
  let total_subjects := 4
  let total_points_required := GPA_required * total_subjects
  -- Points from guaranteed A's in Mathematics and Science
  let guaranteed_points := 8
  -- Required points from Literature and History
  let points_needed := total_points_required - guaranteed_points
  -- Probabilities for grades in Literature
  let prob_A_Lit := 1 / 3
  let prob_B_Lit := 1 / 3
  let prob_C_Lit := 1 / 3
  -- Probabilities for grades in History
  let prob_A_Hist := 1 / 5
  let prob_B_Hist := 1 / 4
  let prob_C_Hist := 11 / 20
  -- Combinations of grades to achieve the required points
  let prob_two_As := prob_A_Lit * prob_A_Hist
  let prob_A_Lit_B_Hist := prob_A_Lit * prob_B_Hist
  let prob_B_Lit_A_Hist := prob_B_Lit * prob_A_Hist
  let prob_two_Bs := prob_B_Lit * prob_B_Hist
  -- Total probability of achieving at least the required GPA
  let total_probability := prob_two_As + prob_A_Lit_B_Hist + prob_B_Lit_A_Hist + prob_two_Bs
  total_probability = 3 / 10 := sorry

end NUMINAMATH_GPT_mark_gpa_probability_l1405_140593


namespace NUMINAMATH_GPT_max_minutes_sleep_without_missing_happy_moment_l1405_140596

def isHappyMoment (h m : ℕ) : Prop :=
  (h = 4 * m ∨ m = 4 * h) ∧ h < 24 ∧ m < 60

def sleepDurationMax : ℕ :=
  239

theorem max_minutes_sleep_without_missing_happy_moment :
  ∀ (sleepDuration : ℕ), sleepDuration ≤ 239 :=
sorry

end NUMINAMATH_GPT_max_minutes_sleep_without_missing_happy_moment_l1405_140596


namespace NUMINAMATH_GPT_service_center_location_l1405_140502

theorem service_center_location : 
  ∀ (milepost4 milepost9 : ℕ), 
  milepost4 = 30 → milepost9 = 150 → 
  (∃ milepost_service_center : ℕ, milepost_service_center = milepost4 + ((milepost9 - milepost4) / 2)) → 
  milepost_service_center = 90 :=
by
  intros milepost4 milepost9 h4 h9 hsc
  sorry

end NUMINAMATH_GPT_service_center_location_l1405_140502


namespace NUMINAMATH_GPT_new_volume_l1405_140581

theorem new_volume (l w h : ℝ) 
  (h1: l * w * h = 3000) 
  (h2: l * w + w * h + l * h = 690) 
  (h3: l + w + h = 40) : 
  (l + 2) * (w + 2) * (h + 2) = 4548 := 
  sorry

end NUMINAMATH_GPT_new_volume_l1405_140581


namespace NUMINAMATH_GPT_guess_probability_l1405_140539

-- Definitions based on the problem conditions
def even_digits : Set ℕ := {0, 2, 4, 6, 8}

def possible_attempts : ℕ := (5 * 4) -- A^2_5

def favorable_outcomes : ℕ := (4 * 2) -- C^1_4 * A^2_2

noncomputable def probability_correct_guess : ℝ :=
  (favorable_outcomes : ℝ) / (possible_attempts : ℝ)

-- Lean statement for the proof problem
theorem guess_probability : probability_correct_guess = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_guess_probability_l1405_140539


namespace NUMINAMATH_GPT_part_I_part_II_l1405_140516

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) - (a * x) / (x + 1)

theorem part_I (a : ℝ) : (∀ x, f a 0 ≤ f a x) → a = 1 := by
  sorry

theorem part_II (a : ℝ) : (∀ x > 0, f a x > 0) → a ≤ 1 := by
  sorry

end NUMINAMATH_GPT_part_I_part_II_l1405_140516


namespace NUMINAMATH_GPT_range_of_a_l1405_140517

-- Define the propositions p and q
def p (a : ℝ) := ∀ x : ℝ, 0 ≤ x → x ≤ 1 → a ≥ Real.exp x
def q (a : ℝ) := ∃ x : ℝ, x^2 + 4 * x + a = 0

-- The proof statement
theorem range_of_a (a : ℝ) : (p a ∧ q a) → a ∈ Set.Icc (Real.exp 1) 4 := by
  intro h
  sorry

end NUMINAMATH_GPT_range_of_a_l1405_140517


namespace NUMINAMATH_GPT_caesars_rental_fee_l1405_140570

theorem caesars_rental_fee (C : ℕ) 
  (hc : ∀ (n : ℕ), n = 60 → C + 30 * n = 500 + 35 * n) : 
  C = 800 :=
by
  sorry

end NUMINAMATH_GPT_caesars_rental_fee_l1405_140570


namespace NUMINAMATH_GPT_find_m_for_even_function_l1405_140513

def quadratic_function (m : ℝ) (x : ℝ) : ℝ :=
  m * x^2 + (m + 2) * m * x + 2

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem find_m_for_even_function :
  ∃ m : ℝ, is_even_function (quadratic_function m) ∧ m = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_m_for_even_function_l1405_140513


namespace NUMINAMATH_GPT_original_number_of_people_l1405_140500

/-- Initially, one-third of the people in a room left.
Then, one-fourth of those remaining started to dance.
There were then 18 people who were not dancing.
What was the original number of people in the room? -/
theorem original_number_of_people (x : ℕ) 
  (h_one_third_left : ∀ y : ℕ, 2 * y / 3 = x) 
  (h_one_fourth_dancing : ∀ y : ℕ, y / 4 = x) 
  (h_non_dancers : x / 2 = 18) : 
  x = 36 :=
sorry

end NUMINAMATH_GPT_original_number_of_people_l1405_140500


namespace NUMINAMATH_GPT_min_ticket_gates_l1405_140525

theorem min_ticket_gates (a x y : ℕ) (h_pos: a > 0) :
  (a = 30 * x) ∧ (y = 2 * x) → ∃ n : ℕ, (n ≥ 4) ∧ (a + 5 * x ≤ 5 * n * y) :=
by
  sorry

end NUMINAMATH_GPT_min_ticket_gates_l1405_140525


namespace NUMINAMATH_GPT_min_deliveries_l1405_140532

theorem min_deliveries (cost_per_delivery_income: ℕ) (cost_per_delivery_gas: ℕ) (van_cost: ℕ) (d: ℕ) : 
  (d * (cost_per_delivery_income - cost_per_delivery_gas) ≥ van_cost) ↔ (d ≥ van_cost / (cost_per_delivery_income - cost_per_delivery_gas)) :=
by
  sorry

def john_deliveries : ℕ := 7500 / (15 - 5)

example : john_deliveries = 750 :=
by
  sorry

end NUMINAMATH_GPT_min_deliveries_l1405_140532


namespace NUMINAMATH_GPT_tim_total_spending_l1405_140589

def lunch_cost : ℝ := 50.50
def dessert_cost : ℝ := 8.25
def beverage_cost : ℝ := 3.75
def lunch_discount : ℝ := 0.10
def dessert_tax : ℝ := 0.07
def beverage_tax : ℝ := 0.05
def lunch_tip_rate : ℝ := 0.20
def other_items_tip_rate : ℝ := 0.15

def total_spending : ℝ := 
  let lunch_after_discount := lunch_cost * (1 - lunch_discount)
  let dessert_after_tax := dessert_cost * (1 + dessert_tax)
  let beverage_after_tax := beverage_cost * (1 + beverage_tax)
  let tip_on_lunch := lunch_after_discount * lunch_tip_rate
  let combined_other_items := dessert_after_tax + beverage_after_tax
  let tip_on_other_items := combined_other_items * other_items_tip_rate
  lunch_after_discount + dessert_after_tax + beverage_after_tax + tip_on_lunch + tip_on_other_items

theorem tim_total_spending :
  total_spending = 69.23 :=
by
  sorry

end NUMINAMATH_GPT_tim_total_spending_l1405_140589


namespace NUMINAMATH_GPT_log_domain_l1405_140559

theorem log_domain (x : ℝ) : x + 2 > 0 ↔ x ∈ Set.Ioi (-2) :=
by
  sorry

end NUMINAMATH_GPT_log_domain_l1405_140559


namespace NUMINAMATH_GPT_point_coordinates_l1405_140599

namespace CoordinateProof

structure Point where
  x : ℝ
  y : ℝ

def isSecondQuadrant (P : Point) : Prop := P.x < 0 ∧ P.y > 0
def distToXAxis (P : Point) : ℝ := |P.y|
def distToYAxis (P : Point) : ℝ := |P.x|

theorem point_coordinates (P : Point) (h1 : isSecondQuadrant P) (h2 : distToXAxis P = 3) (h3 : distToYAxis P = 7) : P = ⟨-7, 3⟩ :=
by
  sorry

end CoordinateProof

end NUMINAMATH_GPT_point_coordinates_l1405_140599


namespace NUMINAMATH_GPT_profit_percentage_is_40_l1405_140598

-- Define the given conditions
def total_cost : ℚ := 44 * 150 + 36 * 125  -- Rs 11100
def total_weight : ℚ := 44 + 36            -- 80 kg
def selling_price_per_kg : ℚ := 194.25     -- Rs 194.25
def total_selling_price : ℚ := total_weight * selling_price_per_kg  -- Rs 15540
def profit : ℚ := total_selling_price - total_cost  -- Rs 4440

-- Define the statement about the profit percentage
def profit_percentage : ℚ := (profit / total_cost) * 100

-- State the theorem
theorem profit_percentage_is_40 :
  profit_percentage = 40 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_profit_percentage_is_40_l1405_140598


namespace NUMINAMATH_GPT_missing_fraction_is_correct_l1405_140573

theorem missing_fraction_is_correct :
  (1 / 3 + 1 / 2 + -5 / 6 + 1 / 5 + -9 / 20 + -9 / 20) = 0.45 - (23 / 20) :=
by
  sorry

end NUMINAMATH_GPT_missing_fraction_is_correct_l1405_140573


namespace NUMINAMATH_GPT_fraction_division_l1405_140531

theorem fraction_division:
  (1 / 4) / (1 / 8) = 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_division_l1405_140531


namespace NUMINAMATH_GPT_volume_surface_area_ratio_l1405_140579

theorem volume_surface_area_ratio
  (V : ℕ := 9)
  (S : ℕ := 34)
  (shape_conditions : ∃ n : ℕ, n = 9 ∧ ∃ m : ℕ, m = 2) :
  V / S = 9 / 34 :=
by
  sorry

end NUMINAMATH_GPT_volume_surface_area_ratio_l1405_140579


namespace NUMINAMATH_GPT_solve_equation_l1405_140534

theorem solve_equation (x : ℝ) (h : x ≠ -1) :
  (x = -1 / 2 ∨ x = 2) ↔ (∃ x : ℝ, x ≠ -1 ∧ (x^3 - x^2)/(x^2 + 2*x + 1) + x = -2) :=
sorry

end NUMINAMATH_GPT_solve_equation_l1405_140534


namespace NUMINAMATH_GPT_digit_H_value_l1405_140503

theorem digit_H_value (E F G H : ℕ) (h1 : E < 10) (h2 : F < 10) (h3 : G < 10) (h4 : H < 10)
  (cond1 : 10 * E + F + 10 * G + E = 10 * H + E)
  (cond2 : 10 * E + F - (10 * G + E) = E)
  (cond3 : E + G = H + 1) : H = 8 :=
sorry

end NUMINAMATH_GPT_digit_H_value_l1405_140503


namespace NUMINAMATH_GPT_merchant_articles_l1405_140506

theorem merchant_articles (N CP SP : ℝ) (h1 : N * CP = 16 * SP) (h2 : SP = CP * 1.0625) (h3 : CP ≠ 0) : N = 17 :=
by
  sorry

end NUMINAMATH_GPT_merchant_articles_l1405_140506


namespace NUMINAMATH_GPT_number_of_boys_l1405_140535

-- Define the conditions given in the problem
def total_people := 41
def total_amount := 460
def boy_amount := 12
def girl_amount := 8

-- Define the proof statement that needs to be proven
theorem number_of_boys (B G : ℕ) (h1 : B + G = total_people) (h2 : boy_amount * B + girl_amount * G = total_amount) : B = 33 := 
by {
  -- The actual proof will go here
  sorry
}

end NUMINAMATH_GPT_number_of_boys_l1405_140535


namespace NUMINAMATH_GPT_solution_l1405_140545

noncomputable def polynomial_has_real_root (a : ℝ) : Prop :=
  ∃ x : ℝ, x^4 - a * x^2 + a * x - 1 = 0

theorem solution (a : ℝ) : polynomial_has_real_root a :=
sorry

end NUMINAMATH_GPT_solution_l1405_140545


namespace NUMINAMATH_GPT_average_age_of_combined_rooms_l1405_140587

theorem average_age_of_combined_rooms
  (num_people_A : ℕ) (avg_age_A : ℕ)
  (num_people_B : ℕ) (avg_age_B : ℕ)
  (num_people_C : ℕ) (avg_age_C : ℕ)
  (hA : num_people_A = 8) (hAA : avg_age_A = 35)
  (hB : num_people_B = 5) (hBB : avg_age_B = 30)
  (hC : num_people_C = 7) (hCC : avg_age_C = 50) :
  ((num_people_A * avg_age_A + num_people_B * avg_age_B + num_people_C * avg_age_C) / 
  (num_people_A + num_people_B + num_people_C) = 39) :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_combined_rooms_l1405_140587


namespace NUMINAMATH_GPT_total_number_of_boys_in_camp_l1405_140585

theorem total_number_of_boys_in_camp (T : ℕ)
  (hA1 : ∃ (boysA : ℕ), boysA = 20 * T / 100)
  (hA2 : ∀ (boysS : ℕ) (boysM : ℕ), boysS = 30 * boysA / 100 ∧ boysM = 40 * boysA / 100)
  (hB1 : ∃ (boysB : ℕ), boysB = 30 * T / 100)
  (hB2 : ∀ (boysS : ℕ) (boysM : ℕ), boysS = 25 * boysB / 100 ∧ boysM = 35 * boysB / 100)
  (hC1 : ∃ (boysC : ℕ), boysC = 50 * T / 100)
  (hC2 : ∀ (boysS : ℕ) (boysM : ℕ), boysS = 15 * boysC / 100 ∧ boysM = 45 * boysC / 100)
  (hA_no_SM : 77 = 70 * boysA / 100)
  (hB_no_SM : 72 = 60 * boysB / 100)
  (hC_no_SM : 98 = 60 * boysC / 100) :
  T = 535 :=
by
  sorry

end NUMINAMATH_GPT_total_number_of_boys_in_camp_l1405_140585


namespace NUMINAMATH_GPT_sin_phi_value_l1405_140551

theorem sin_phi_value 
  (φ α : ℝ)
  (hφ : φ = 2 * α)
  (hα1 : Real.sin α = (Real.sqrt 5) / 5)
  (hα2 : Real.cos α = 2 * (Real.sqrt 5) / 5) 
  : Real.sin φ = 4 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_sin_phi_value_l1405_140551


namespace NUMINAMATH_GPT_range_of_m_l1405_140591

variable (m : ℝ)

/-- Proposition p: For any x in ℝ, x^2 + 1 > m -/
def p := ∀ x : ℝ, x^2 + 1 > m

/-- Proposition q: The linear function f(x) = (2 - m) * x + 1 is an increasing function -/
def q := (2 - m) > 0

theorem range_of_m (h1 : p m ∨ q m) (h2 : ¬(p m ∧ q m)) : 1 < m ∧ m < 2 := 
sorry

end NUMINAMATH_GPT_range_of_m_l1405_140591


namespace NUMINAMATH_GPT_judah_crayons_l1405_140541

theorem judah_crayons (karen beatrice gilbert judah : ℕ) 
  (h1 : karen = 128)
  (h2 : karen = 2 * beatrice)
  (h3 : beatrice = 2 * gilbert)
  (h4 : gilbert = 4 * judah) : 
  judah = 8 :=
by
  sorry

end NUMINAMATH_GPT_judah_crayons_l1405_140541


namespace NUMINAMATH_GPT_no_nat_numbers_m_n_satisfy_eq_l1405_140511

theorem no_nat_numbers_m_n_satisfy_eq (m n : ℕ) : ¬ (m^2 = n^2 + 2014) := sorry

end NUMINAMATH_GPT_no_nat_numbers_m_n_satisfy_eq_l1405_140511


namespace NUMINAMATH_GPT_parallelogram_area_example_l1405_140512

def point := (ℚ × ℚ)
def parallelogram_area (A B C D : point) : ℚ :=
  let base := B.1 - A.1
  let height := C.2 - A.2
  base * height

theorem parallelogram_area_example : 
  parallelogram_area (1, 1) (7, 1) (4, 9) (10, 9) = 48 := by
  sorry

end NUMINAMATH_GPT_parallelogram_area_example_l1405_140512


namespace NUMINAMATH_GPT_candies_problem_max_children_l1405_140533

theorem candies_problem_max_children (u v : ℕ → ℕ) (n : ℕ) :
  (∀ i : ℕ, u i = v i + 2) →
  (∀ i : ℕ, u i + 2 = u (i + 1)) →
  (u (n - 1) / u 0 = 13) →
  n = 25 :=
by
  -- Proof not required as per the instructions.
  sorry

end NUMINAMATH_GPT_candies_problem_max_children_l1405_140533


namespace NUMINAMATH_GPT_bus_sarah_probability_l1405_140514

-- Define the probability of Sarah arriving while the bus is still there
theorem bus_sarah_probability :
  let total_minutes := 60
  let bus_waiting_time := 15
  let total_area := (total_minutes * total_minutes : ℕ)
  let triangle_area := (1 / 2 : ℝ) * 45 * 15
  let rectangle_area := 15 * 15
  let shaded_area := triangle_area + rectangle_area
  (shaded_area / total_area : ℝ) = (5 / 32 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_bus_sarah_probability_l1405_140514


namespace NUMINAMATH_GPT_k_range_for_two_zeros_of_f_l1405_140595

noncomputable def f (x k : ℝ) : ℝ := x^2 - x * (Real.log x) - k * (x + 2) + 2

theorem k_range_for_two_zeros_of_f :
  ∀ k : ℝ, (∃ x1 x2 : ℝ, (1/2 < x1) ∧ (x1 < x2) ∧ f x1 k = 0 ∧ f x2 k = 0) ↔ 1 < k ∧ k ≤ (9 + 2 * Real.log 2) / 10 :=
by
  sorry

end NUMINAMATH_GPT_k_range_for_two_zeros_of_f_l1405_140595


namespace NUMINAMATH_GPT_general_formula_sum_b_l1405_140528

-- Define the arithmetic sequence
def arithmetic_sequence (a d: ℕ) (n: ℕ) := a + (n - 1) * d

-- Given conditions
def a1 : ℕ := 1
def d : ℕ := 2
def a (n : ℕ) : ℕ := arithmetic_sequence a1 d n
def b (n : ℕ) : ℕ := 2 ^ a n

-- Formula for the arithmetic sequence
theorem general_formula (n : ℕ) : a n = 2 * n - 1 := 
by sorry

-- Sum of the first n terms of b_n
theorem sum_b (n : ℕ) : (Finset.range n).sum b = (2 / 3) * (4 ^ n - 1) :=
by sorry

end NUMINAMATH_GPT_general_formula_sum_b_l1405_140528


namespace NUMINAMATH_GPT_machine_loan_repaid_in_5_months_l1405_140574

theorem machine_loan_repaid_in_5_months :
  ∀ (loan cost selling_price tax_percentage products_per_month profit_per_product months : ℕ),
    loan = 22000 →
    cost = 5 →
    selling_price = 8 →
    tax_percentage = 10 →
    products_per_month = 2000 →
    profit_per_product = (selling_price - cost - (selling_price * tax_percentage / 100)) →
    (products_per_month * months * profit_per_product) ≥ loan →
    months = 5 :=
by
  intros loan cost selling_price tax_percentage products_per_month profit_per_product months
  sorry

end NUMINAMATH_GPT_machine_loan_repaid_in_5_months_l1405_140574


namespace NUMINAMATH_GPT_smallest_number_to_end_in_four_zeros_l1405_140554

theorem smallest_number_to_end_in_four_zeros (x : ℕ) :
  let n1 := 225
  let n2 := 525
  let factor_needed := 16
  (∃ y : ℕ, y = n1 * n2 * x) ∧ (10^4 ∣ n1 * n2 * x) ↔ x = factor_needed :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_to_end_in_four_zeros_l1405_140554


namespace NUMINAMATH_GPT_regular_polygon_sides_l1405_140540

theorem regular_polygon_sides (angle : ℝ) (h1 : angle = 150) : ∃ n : ℕ, (180 * (n - 2)) / n = angle ∧ n = 12 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1405_140540


namespace NUMINAMATH_GPT_average_second_pair_l1405_140542

theorem average_second_pair 
  (avg_six : ℝ) (avg_first_pair : ℝ) (avg_third_pair : ℝ) (avg_second_pair : ℝ) 
  (h1 : avg_six = 3.95) 
  (h2 : avg_first_pair = 4.2) 
  (h3 : avg_third_pair = 3.8000000000000007) : 
  avg_second_pair = 3.85 :=
by
  sorry

end NUMINAMATH_GPT_average_second_pair_l1405_140542


namespace NUMINAMATH_GPT_sum_radical_conjugate_l1405_140552

theorem sum_radical_conjugate : (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 := 
by 
  sorry

end NUMINAMATH_GPT_sum_radical_conjugate_l1405_140552


namespace NUMINAMATH_GPT_number_of_baskets_l1405_140547

-- Define the conditions
def total_peaches : Nat := 10
def red_peaches_per_basket : Nat := 4
def green_peaches_per_basket : Nat := 6
def peaches_per_basket : Nat := red_peaches_per_basket + green_peaches_per_basket

-- The goal is to prove that the number of baskets is 1 given the conditions

theorem number_of_baskets (h1 : total_peaches = 10)
                           (h2 : peaches_per_basket = red_peaches_per_basket + green_peaches_per_basket)
                           (h3 : red_peaches_per_basket = 4)
                           (h4 : green_peaches_per_basket = 6) : 
                           total_peaches / peaches_per_basket = 1 := by
                            sorry

end NUMINAMATH_GPT_number_of_baskets_l1405_140547


namespace NUMINAMATH_GPT_billy_apples_l1405_140597

def num_apples_eaten (monday_apples tuesday_apples wednesday_apples thursday_apples friday_apples total_apples : ℕ) : Prop :=
  monday_apples = 2 ∧
  tuesday_apples = 2 * monday_apples ∧
  wednesday_apples = 9 ∧
  friday_apples = monday_apples / 2 ∧
  thursday_apples = 4 * friday_apples ∧
  total_apples = monday_apples + tuesday_apples + wednesday_apples + thursday_apples + friday_apples

theorem billy_apples : num_apples_eaten 2 4 9 4 1 20 := 
by
  unfold num_apples_eaten
  sorry

end NUMINAMATH_GPT_billy_apples_l1405_140597


namespace NUMINAMATH_GPT_problem1_problem2_l1405_140562

variable {a b x : ℝ}

theorem problem1 (h₀ : a ≠ b) (h₁ : a ≠ -b) :
  (a / (a - b)) - (b / (a + b)) = (a^2 + b^2) / (a^2 - b^2) :=
sorry

theorem problem2 (h₀ : x ≠ 2) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  ((x - 2) / (x - 1)) / ((x^2 - 4 * x + 4) / (x^2 - 1)) + ((1 - x) / (x - 2)) = 2 / (x - 2) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1405_140562


namespace NUMINAMATH_GPT_velocity_of_point_C_l1405_140583

variable (a T R L x : ℝ)
variable (a_pos : a > 0) (T_pos : T > 0) (R_pos : R > 0) (L_pos : L > 0)
variable (h_eq : a * T / (a * T - R) = (L + x) / x)

theorem velocity_of_point_C : a * (L / R) = x / T := by
  sorry

end NUMINAMATH_GPT_velocity_of_point_C_l1405_140583


namespace NUMINAMATH_GPT_value_of_m_l1405_140553

theorem value_of_m (m : ℤ) : (|m| = 1) ∧ (m + 1 ≠ 0) → m = 1 := by
  sorry

end NUMINAMATH_GPT_value_of_m_l1405_140553


namespace NUMINAMATH_GPT_compute_difference_l1405_140538

def distinct_solutions (p q : ℝ) : Prop :=
  (p ≠ q) ∧ (∃ (x : ℝ), (x = p ∨ x = q) ∧ (x-3)*(x+3) = 21*x - 63) ∧
  (p > q)

theorem compute_difference (p q : ℝ) (h : distinct_solutions p q) : p - q = 15 :=
by
  sorry

end NUMINAMATH_GPT_compute_difference_l1405_140538


namespace NUMINAMATH_GPT_range_of_a_I_minimum_value_of_a_II_l1405_140550

open Real

def f (x a : ℝ) : ℝ := abs (x - a)

theorem range_of_a_I (a : ℝ) :
  (∀ x, -1 ≤ x → x ≤ 3 → f x a ≤ 3) ↔ 0 ≤ a ∧ a ≤ 2 := sorry

theorem minimum_value_of_a_II :
  ∀ a : ℝ, (∀ x : ℝ, f (x - a) a + f (x + a) a ≥ 1 - 2 * a) ↔ a ≥ (1 / 4) :=
sorry

end NUMINAMATH_GPT_range_of_a_I_minimum_value_of_a_II_l1405_140550


namespace NUMINAMATH_GPT_cos_half_pi_plus_double_alpha_l1405_140556

theorem cos_half_pi_plus_double_alpha (α : ℝ) (h : Real.tan α = 1 / 3) : 
  Real.cos (Real.pi / 2 + 2 * α) = -3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_half_pi_plus_double_alpha_l1405_140556


namespace NUMINAMATH_GPT_min_trips_required_l1405_140515

def masses : List ℕ := [150, 62, 63, 66, 70, 75, 79, 84, 95, 96, 99]
def load_capacity : ℕ := 190

theorem min_trips_required :
  ∃ (trips : ℕ), 
  (∀ partition : List (List ℕ), (∀ group : List ℕ, group ∈ partition → 
  group.sum ≤ load_capacity) ∧ partition.join = masses → 
  partition.length ≥ 6) :=
sorry

end NUMINAMATH_GPT_min_trips_required_l1405_140515


namespace NUMINAMATH_GPT_quadratic_real_roots_a_condition_l1405_140576

theorem quadratic_real_roots_a_condition (a : ℝ) (h : ∃ x : ℝ, (a - 5) * x^2 - 4 * x - 1 = 0) :
  a ≥ 1 ∧ a ≠ 5 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_a_condition_l1405_140576


namespace NUMINAMATH_GPT_golu_distance_travelled_l1405_140519

theorem golu_distance_travelled 
  (b : ℝ) (c : ℝ) (h : c^2 = x^2 + b^2) : x = 8 := by
  sorry

end NUMINAMATH_GPT_golu_distance_travelled_l1405_140519


namespace NUMINAMATH_GPT_k_range_m_range_l1405_140564

noncomputable def f (x : ℝ) : ℝ := 1 - (2 / (2^x + 1))

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem k_range (k : ℝ) : (∃ x : ℝ, g x = (2^x + 1) * f x + k) → k < 1 :=
by
  sorry

theorem m_range (m : ℝ) : (∀ x1 : ℝ, 0 < x1 ∧ x1 < 1 → 
                        ∃ x2 : ℝ, -Real.pi / 4 ≤ x2 ∧ x2 ≤ Real.pi / 6 ∧ f x1 - m * 2^x1 > g x2) 
                       → m ≤ 7 / 6 :=
by
  sorry

end NUMINAMATH_GPT_k_range_m_range_l1405_140564


namespace NUMINAMATH_GPT_mark_brings_in_148_cans_l1405_140560

-- Define the given conditions
variable (R : ℕ) (Mark Jaydon Sophie : ℕ)

-- Conditions
def jaydon_cans := 2 * R + 5
def mark_cans := 4 * jaydon_cans
def unit_ratio := mark_cans / 4
def sophie_cans := 2 * unit_ratio

-- Condition: Total cans
def total_cans := mark_cans + jaydon_cans + sophie_cans

-- Condition: Each contributes at least 5 cans
axiom each_contributes_at_least_5 : R ≥ 5

-- Condition: Total cans is an odd number not less than 250
axiom total_odd_not_less_than_250 : ∃ k : ℕ, total_cans = 2 * k + 1 ∧ total_cans ≥ 250

-- Theorem: Prove Mark brings in 148 cans under the conditions
theorem mark_brings_in_148_cans (h : R = 16) : mark_cans = 148 :=
by sorry

end NUMINAMATH_GPT_mark_brings_in_148_cans_l1405_140560


namespace NUMINAMATH_GPT_approximate_number_of_fish_in_pond_l1405_140522

-- Define the conditions as hypotheses.
def tagged_fish_caught_first : ℕ := 50
def total_fish_caught_second : ℕ := 50
def tagged_fish_found_second : ℕ := 5

-- Define total fish in the pond.
def total_fish_in_pond (N : ℝ) : Prop :=
  tagged_fish_found_second / total_fish_caught_second = tagged_fish_caught_first / N

-- The statement to be proved.
theorem approximate_number_of_fish_in_pond (N : ℝ) (h : total_fish_in_pond N) : N = 500 :=
sorry

end NUMINAMATH_GPT_approximate_number_of_fish_in_pond_l1405_140522


namespace NUMINAMATH_GPT_tsunami_added_sand_l1405_140563

noncomputable def dig_rate : ℝ := 8 / 4 -- feet per hour
noncomputable def sand_after_storm : ℝ := 8 / 2 -- feet
noncomputable def time_to_dig_up_treasure : ℝ := 3 -- hours
noncomputable def total_sand_dug_up : ℝ := dig_rate * time_to_dig_up_treasure -- feet

theorem tsunami_added_sand :
  total_sand_dug_up - sand_after_storm = 2 :=
by
  sorry

end NUMINAMATH_GPT_tsunami_added_sand_l1405_140563


namespace NUMINAMATH_GPT_length_PR_l1405_140536

variable (P Q R : Type) [Inhabited P] [Inhabited Q] [Inhabited R]
variable {xPR xQR xsinR : ℝ}
variable (hypotenuse_opposite_ratio : xsinR = (3/5))
variable (sideQR : xQR = 9)
variable (rightAngle : ∀ (P Q R : Type), P ≠ Q → Q ∈ line_through Q R)

theorem length_PR : (∃ xPR : ℝ, xPR = 15) :=
by
  sorry

end NUMINAMATH_GPT_length_PR_l1405_140536


namespace NUMINAMATH_GPT_eq_sin_intersect_16_solutions_l1405_140537

theorem eq_sin_intersect_16_solutions :
  ∃ S : Finset ℝ, (∀ x ∈ S, 0 ≤ x ∧ x ≤ 50 ∧ (x / 50 = Real.sin x)) ∧ (S.card = 16) :=
  sorry

end NUMINAMATH_GPT_eq_sin_intersect_16_solutions_l1405_140537


namespace NUMINAMATH_GPT_program_output_l1405_140544

theorem program_output :
  ∃ a b : ℕ, a = 10 ∧ b = a - 8 ∧ a = a - b ∧ a = 8 :=
by
  let a := 10
  let b := a - 8
  let a := a - b
  use a
  use b
  sorry

end NUMINAMATH_GPT_program_output_l1405_140544


namespace NUMINAMATH_GPT_sum_of_differences_of_7_in_657932657_l1405_140572

theorem sum_of_differences_of_7_in_657932657 :
  let numeral := 657932657
  let face_value (d : Nat) := d
  let local_value (d : Nat) (pos : Nat) := d * 10 ^ pos
  let indices_of_7 := [6, 0]
  let differences := indices_of_7.map (fun pos => local_value 7 pos - face_value 7)
  differences.sum = 6999993 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_differences_of_7_in_657932657_l1405_140572


namespace NUMINAMATH_GPT_probability_different_colors_l1405_140509

theorem probability_different_colors :
  let total_chips := 16
  let prob_blue := (7 : ℚ) / total_chips
  let prob_yellow := (5 : ℚ) / total_chips
  let prob_red := (4 : ℚ) / total_chips
  let prob_blue_then_nonblue := prob_blue * ((prob_yellow + prob_red) : ℚ)
  let prob_yellow_then_non_yellow := prob_yellow * ((prob_blue + prob_red) : ℚ)
  let prob_red_then_non_red := prob_red * ((prob_blue + prob_yellow) : ℚ)
  let total_prob := prob_blue_then_nonblue + prob_yellow_then_non_yellow + prob_red_then_non_red
  total_prob = (83 : ℚ) / 128 := 
by
  sorry

end NUMINAMATH_GPT_probability_different_colors_l1405_140509


namespace NUMINAMATH_GPT_cabbages_produced_l1405_140566

theorem cabbages_produced (x y : ℕ) (h1 : y = x + 1) (h2 : x^2 + 199 = y^2) : y^2 = 10000 :=
by
  sorry

end NUMINAMATH_GPT_cabbages_produced_l1405_140566


namespace NUMINAMATH_GPT_savings_by_buying_gallon_l1405_140580

def gallon_to_ounces : ℕ := 128
def bottle_volume_ounces : ℕ := 16
def cost_gallon : ℕ := 8
def cost_bottle : ℕ := 3

theorem savings_by_buying_gallon :
  (cost_bottle * (gallon_to_ounces / bottle_volume_ounces)) - cost_gallon = 16 := 
by
  sorry

end NUMINAMATH_GPT_savings_by_buying_gallon_l1405_140580


namespace NUMINAMATH_GPT_percentage_of_boys_to_girls_l1405_140548

theorem percentage_of_boys_to_girls
  (boys : ℕ) (girls : ℕ)
  (h1 : boys = 20)
  (h2 : girls = 26) :
  (boys / girls : ℝ) * 100 = 76.9 := by
  sorry

end NUMINAMATH_GPT_percentage_of_boys_to_girls_l1405_140548


namespace NUMINAMATH_GPT_xy_addition_equals_13_l1405_140520

theorem xy_addition_equals_13 (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hx_lt_15 : x < 15) (hy_lt_15 : y < 15) (hxy : x + y + x * y = 49) : x + y = 13 :=
by
  sorry

end NUMINAMATH_GPT_xy_addition_equals_13_l1405_140520


namespace NUMINAMATH_GPT_sum_of_fractions_l1405_140592

theorem sum_of_fractions :
  (3 / 15) + (6 / 15) + (9 / 15) + (12 / 15) + (15 / 15) + 
  (18 / 15) + (21 / 15) + (24 / 15) + (27 / 15) + (75 / 15) = 14 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l1405_140592


namespace NUMINAMATH_GPT_inequality_solution_set_l1405_140575

theorem inequality_solution_set :
  ∀ x : ℝ, (1 / (x^2 + 1) > 5 / x + 21 / 10) ↔ x ∈ Set.Ioo (-2 : ℝ) (0 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_set_l1405_140575


namespace NUMINAMATH_GPT_third_number_hcf_lcm_l1405_140504

theorem third_number_hcf_lcm (N : ℕ) 
  (HCF : Nat.gcd (Nat.gcd 136 144) N = 8)
  (LCM : Nat.lcm (Nat.lcm 136 144) N = 2^4 * 3^2 * 17 * 7) : 
  N = 7 := 
  sorry

end NUMINAMATH_GPT_third_number_hcf_lcm_l1405_140504


namespace NUMINAMATH_GPT_completing_the_square_l1405_140530

theorem completing_the_square (x : ℝ) : x^2 + 8*x + 7 = 0 → (x + 4)^2 = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_completing_the_square_l1405_140530


namespace NUMINAMATH_GPT_solve_marble_problem_l1405_140590

noncomputable def marble_problem : Prop :=
  ∃ k : ℕ, k ≥ 0 ∧ k ≤ 50 ∧ 
  (∀ initial_white initial_black : ℕ, initial_white = 50 ∧ initial_black = 50 → 
  ∃ w b : ℕ, w = 50 + k - initial_black ∧ b = 50 - k ∧ (w, b) = (2, 0))

theorem solve_marble_problem: marble_problem :=
sorry

end NUMINAMATH_GPT_solve_marble_problem_l1405_140590


namespace NUMINAMATH_GPT_floor_problem_solution_l1405_140524

noncomputable def floor_problem (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/2⌋ = ⌊x + 3⌋

theorem floor_problem_solution :
  { x : ℝ | floor_problem x } = { x : ℝ | 2 ≤ x ∧ x < 7 / 3 } :=
by sorry

end NUMINAMATH_GPT_floor_problem_solution_l1405_140524


namespace NUMINAMATH_GPT_find_a45_l1405_140557

theorem find_a45 :
  ∃ (a : ℕ → ℝ), 
    a 0 = 11 ∧ a 1 = 11 ∧ 
    (∀ m n : ℕ, a (m + n) = (1/2) * (a (2 * m) + a (2 * n)) - (m - n)^2) ∧ 
    a 45 = 1991 := by
  sorry

end NUMINAMATH_GPT_find_a45_l1405_140557


namespace NUMINAMATH_GPT_train_more_passengers_l1405_140577

def one_train_car_capacity : ℕ := 60
def one_airplane_capacity : ℕ := 366
def number_of_train_cars : ℕ := 16
def number_of_airplanes : ℕ := 2

theorem train_more_passengers {one_train_car_capacity : ℕ} 
                               {one_airplane_capacity : ℕ} 
                               {number_of_train_cars : ℕ} 
                               {number_of_airplanes : ℕ} :
  (number_of_train_cars * one_train_car_capacity) - (number_of_airplanes * one_airplane_capacity) = 228 :=
by
  sorry

end NUMINAMATH_GPT_train_more_passengers_l1405_140577


namespace NUMINAMATH_GPT_square_area_is_81_l1405_140561

def square_perimeter (s : ℕ) : ℕ := 4 * s
def square_area (s : ℕ) : ℕ := s * s

theorem square_area_is_81 (s : ℕ) (h : square_perimeter s = 36) : square_area s = 81 :=
by {
  sorry
}

end NUMINAMATH_GPT_square_area_is_81_l1405_140561


namespace NUMINAMATH_GPT_new_average_weight_is_27_3_l1405_140586

-- Define the given conditions as variables/constants in Lean
noncomputable def original_students : ℕ := 29
noncomputable def original_average_weight : ℝ := 28
noncomputable def new_student_weight : ℝ := 7

-- The total weight of the original students
noncomputable def original_total_weight : ℝ := original_students * original_average_weight
-- The new total number of students
noncomputable def new_total_students : ℕ := original_students + 1
-- The new total weight after new student is added
noncomputable def new_total_weight : ℝ := original_total_weight + new_student_weight

-- The theorem to prove that the new average weight is 27.3 kg
theorem new_average_weight_is_27_3 : (new_total_weight / new_total_students) = 27.3 := 
by
  sorry -- The proof will be provided here

end NUMINAMATH_GPT_new_average_weight_is_27_3_l1405_140586


namespace NUMINAMATH_GPT_units_digit_17_pow_35_l1405_140529

theorem units_digit_17_pow_35 : (17 ^ 35) % 10 = 3 := by
sorry

end NUMINAMATH_GPT_units_digit_17_pow_35_l1405_140529


namespace NUMINAMATH_GPT_quadratic_root_shift_l1405_140571

theorem quadratic_root_shift (d e : ℝ) :
  (∀ r s : ℝ, (r^2 - 2 * r + 0.5 = 0) → (r-3)^2 + (r-3) * (s-3) * d + e = 0) → e = 3.5 := 
by
  intros
  sorry

end NUMINAMATH_GPT_quadratic_root_shift_l1405_140571


namespace NUMINAMATH_GPT_tracy_candies_l1405_140555

variable (x : ℕ) -- number of candies Tracy started with

theorem tracy_candies (h1: x % 4 = 0)
                      (h2 : 46 ≤ x / 2 - 40 ∧ x / 2 - 40 ≤ 50) 
                      (h3 : ∃ k, 2 ≤ k ∧ k ≤ 6 ∧ x / 2 - 40 - k = 4) 
                      (h4 : ∃ n, x = 4 * n) : x = 96 :=
by
  sorry

end NUMINAMATH_GPT_tracy_candies_l1405_140555


namespace NUMINAMATH_GPT_probability_of_yellow_jelly_bean_l1405_140567

theorem probability_of_yellow_jelly_bean (P_red P_orange P_green P_yellow : ℝ)
  (h_red : P_red = 0.1)
  (h_orange : P_orange = 0.4)
  (h_green : P_green = 0.25)
  (h_total : P_red + P_orange + P_green + P_yellow = 1) :
  P_yellow = 0.25 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_yellow_jelly_bean_l1405_140567


namespace NUMINAMATH_GPT_second_athlete_high_jump_eq_eight_l1405_140510

theorem second_athlete_high_jump_eq_eight :
  let first_athlete_long_jump := 26
  let first_athlete_triple_jump := 30
  let first_athlete_high_jump := 7
  let second_athlete_long_jump := 24
  let second_athlete_triple_jump := 34
  let winner_average_jump := 22
  (first_athlete_long_jump + first_athlete_triple_jump + first_athlete_high_jump) / 3 < winner_average_jump →
  ∃ (second_athlete_high_jump : ℝ), 
    second_athlete_high_jump = 
    (winner_average_jump * 3 - (second_athlete_long_jump + second_athlete_triple_jump)) ∧ 
    second_athlete_high_jump = 8 :=
by
  intros 
  sorry

end NUMINAMATH_GPT_second_athlete_high_jump_eq_eight_l1405_140510


namespace NUMINAMATH_GPT_mod_pow_difference_l1405_140568

theorem mod_pow_difference (a b n : ℕ) (h1 : a ≡ 47 [MOD n]) (h2 : b ≡ 22 [MOD n]) (h3 : n = 8) : (a ^ 2023 - b ^ 2023) % n = 1 :=
by
  sorry

end NUMINAMATH_GPT_mod_pow_difference_l1405_140568


namespace NUMINAMATH_GPT_real_roots_of_polynomial_l1405_140526

theorem real_roots_of_polynomial :
  {x : ℝ | (x^4 - 4*x^3 + 5*x^2 - 2*x + 2) = 0} = {1, -1} :=
sorry

end NUMINAMATH_GPT_real_roots_of_polynomial_l1405_140526


namespace NUMINAMATH_GPT_polynomial_roots_expression_l1405_140508

theorem polynomial_roots_expression 
  (a b α β γ δ : ℝ)
  (h1 : α^2 - a*α - 1 = 0)
  (h2 : β^2 - a*β - 1 = 0)
  (h3 : γ^2 - b*γ - 1 = 0)
  (h4 : δ^2 - b*δ - 1 = 0) :
  ((α - γ)^2 * (β - γ)^2 * (α + δ)^2 * (β + δ)^2) = (b^2 - a^2)^2 :=
sorry

end NUMINAMATH_GPT_polynomial_roots_expression_l1405_140508


namespace NUMINAMATH_GPT_geometric_sequence_a7_a8_l1405_140569

-- Define the geometric sequence {a_n}
variable {a : ℕ → ℝ}

-- {a_n} is a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Conditions
axiom h1 : is_geometric_sequence a
axiom h2 : a 1 + a 2 = 40
axiom h3 : a 3 + a 4 = 60

-- Proof problem: Find a_7 + a_8
theorem geometric_sequence_a7_a8 :
  a 7 + a 8 = 135 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a7_a8_l1405_140569


namespace NUMINAMATH_GPT_geometric_progression_theorem_l1405_140543

variables {a b c : ℝ} {n : ℕ} {q : ℝ}

-- Define the terms in the geometric progression
def nth_term (a q : ℝ) (n : ℕ) : ℝ := a * q^n
def second_nth_term (a q : ℝ) (n : ℕ) : ℝ := a * q^(2 * n)
def fourth_nth_term (a q : ℝ) (n : ℕ) : ℝ := a * q^(4 * n)

-- Conditions
axiom nth_term_def : b = nth_term a q n
axiom second_nth_term_def : b = second_nth_term a q n
axiom fourth_nth_term_def : c = fourth_nth_term a q n

-- Statement to prove
theorem geometric_progression_theorem :
  b * (b^2 - a^2) = a^2 * (c - b) :=
sorry

end NUMINAMATH_GPT_geometric_progression_theorem_l1405_140543


namespace NUMINAMATH_GPT_GlobalConnect_more_cost_effective_if_x_300_l1405_140594

def GlobalConnectCost (x : ℕ) : ℝ := 50 + 0.4 * x
def QuickConnectCost (x : ℕ) : ℝ := 0.6 * x

theorem GlobalConnect_more_cost_effective_if_x_300 : 
  GlobalConnectCost 300 < QuickConnectCost 300 :=
by
  sorry

end NUMINAMATH_GPT_GlobalConnect_more_cost_effective_if_x_300_l1405_140594


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1405_140584

theorem simplify_and_evaluate_expression (a b : ℝ) (h₁ : a = 2 + Real.sqrt 3) (h₂ : b = 2 - Real.sqrt 3) :
  (a^2 - b^2) / a / (a - (2 * a * b - b^2) / a) = 2 * Real.sqrt 3 / 3 :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1405_140584


namespace NUMINAMATH_GPT_find_k_l1405_140507

theorem find_k (k : ℝ) : (∀ x y : ℝ, (x + k * y - 2 * k = 0) → (k * x - (k - 2) * y + 1 = 0) → x * k + y * (-1 / k) + y * 2 = 0) →
  (k = 0 ∨ k = 3) :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1405_140507


namespace NUMINAMATH_GPT_find_k_l1405_140578

-- Define the conditions
def parabola (k : ℝ) (x : ℝ) : ℝ := x^2 + 2 * x + k

-- Theorem statement
theorem find_k (k : ℝ) : (∀ x : ℝ, parabola k x = 0 → x = -1) → k = 1 :=
by
  sorry

end NUMINAMATH_GPT_find_k_l1405_140578


namespace NUMINAMATH_GPT_product_divisible_by_49_l1405_140505

theorem product_divisible_by_49 (a b : ℕ) (h : (a^2 + b^2) % 7 = 0) : (a * b) % 49 = 0 :=
sorry

end NUMINAMATH_GPT_product_divisible_by_49_l1405_140505


namespace NUMINAMATH_GPT_matrix_problem_l1405_140565

def A : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![20 / 3, 4 / 3],
  ![-8 / 3, 8 / 3]
]
def B : Matrix (Fin 2) (Fin 2) ℚ := ![
  ![0, 0], -- Correct values for B can be computed from conditions if needed
  ![0, 0]
]

theorem matrix_problem (A B : Matrix (Fin 2) (Fin 2) ℚ)
  (h1 : A + B = A * B)
  (h2 : A * B = ![
  ![20 / 3, 4 / 3],
  ![-8 / 3, 8 / 3]
]) :
  B * A = ![
    ![20 / 3, 4 / 3],
    ![-8 / 3, 8 / 3]
  ] :=
sorry

end NUMINAMATH_GPT_matrix_problem_l1405_140565


namespace NUMINAMATH_GPT_max_distance_l1405_140546

theorem max_distance (x y : ℝ) (u v w : ℝ)
  (h1 : u = Real.sqrt (x^2 + y^2))
  (h2 : v = Real.sqrt ((x - 1)^2 + y^2))
  (h3 : w = Real.sqrt ((x - 1)^2 + (y - 1)^2))
  (h4 : u^2 + v^2 = w^2) :
  ∃ (P : ℝ), P = 2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_max_distance_l1405_140546


namespace NUMINAMATH_GPT_instantaneous_rate_of_change_at_e_l1405_140518

open Real

noncomputable def f (x : ℝ) : ℝ := (log x) / x

theorem instantaneous_rate_of_change_at_e : deriv f e = 0 := by
  sorry

end NUMINAMATH_GPT_instantaneous_rate_of_change_at_e_l1405_140518


namespace NUMINAMATH_GPT_triangle_at_most_one_obtuse_l1405_140527

theorem triangle_at_most_one_obtuse 
  (A B C : ℝ)
  (h_sum : A + B + C = 180) 
  (h_obtuse_A : A > 90) 
  (h_obtuse_B : B > 90) 
  (h_obtuse_C : C > 90) :
  false :=
by 
  sorry

end NUMINAMATH_GPT_triangle_at_most_one_obtuse_l1405_140527


namespace NUMINAMATH_GPT_tom_tim_typing_ratio_l1405_140501

theorem tom_tim_typing_ratio (T M : ℝ) (h1 : T + M = 12) (h2 : T + 1.3 * M = 15) :
  M / T = 5 :=
by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_tom_tim_typing_ratio_l1405_140501


namespace NUMINAMATH_GPT_triangle_side_length_l1405_140523

theorem triangle_side_length (P Q R : Type) (cos_Q : ℝ) (PQ QR : ℝ) 
  (sin_Q : ℝ) (h_cos_Q : cos_Q = 0.6) (h_PQ : PQ = 10) (h_sin_Q : sin_Q = 0.8) : 
  QR = 50 / 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_l1405_140523


namespace NUMINAMATH_GPT_infinitely_many_a_not_prime_l1405_140588

theorem infinitely_many_a_not_prime (a: ℤ) (n: ℤ) : ∃ (b: ℤ), b ≥ 0 ∧ (∃ (N: ℕ) (a: ℤ), a = 4*(N:ℤ)^4 ∧ ∀ (n: ℤ), ¬Prime (n^4 + a)) :=
by { sorry }

end NUMINAMATH_GPT_infinitely_many_a_not_prime_l1405_140588
