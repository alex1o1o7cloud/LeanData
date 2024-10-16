import Mathlib

namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l745_74597

theorem gcd_of_three_numbers : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l745_74597


namespace NUMINAMATH_CALUDE_diana_statue_painting_l745_74577

/-- The number of statues that can be painted given a certain amount of paint and paint required per statue -/
def statues_paintable (paint_available : ℚ) (paint_per_statue : ℚ) : ℚ :=
  paint_available / paint_per_statue

/-- Theorem: Given 1/2 gallon of paint and 1/4 gallon required per statue, 2 statues can be painted -/
theorem diana_statue_painting :
  statues_paintable (1/2) (1/4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_diana_statue_painting_l745_74577


namespace NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l745_74507

/-- The imaginary unit -/
noncomputable def i : ℂ := Complex.I

/-- The main theorem -/
theorem complex_fraction_equals_neg_i : (1 - i) / (1 + i) = -i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_neg_i_l745_74507


namespace NUMINAMATH_CALUDE_no_valid_cookie_count_l745_74584

theorem no_valid_cookie_count : ¬ ∃ (N : ℕ), N < 120 ∧ N % 13 = 3 ∧ N % 8 = 5 := by
  sorry

end NUMINAMATH_CALUDE_no_valid_cookie_count_l745_74584


namespace NUMINAMATH_CALUDE_percentage_problem_l745_74560

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 800 = (20 / 100) * 650 + 190 → P = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l745_74560


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l745_74537

theorem simplify_and_evaluate_expression (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 2) (h3 : a ≠ -2) (h4 : a ≠ -1) (h5 : a = 1) :
  1 - (a - 2) / a / ((a^2 - 4) / (a^2 + a)) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l745_74537


namespace NUMINAMATH_CALUDE_min_k_value_l745_74564

-- Define the function f(x) = x(ln x + 1) / (x - 2)
noncomputable def f (x : ℝ) : ℝ := x * (Real.log x + 1) / (x - 2)

-- State the theorem
theorem min_k_value : 
  (∃ x₀ : ℝ, x₀ > 2 ∧ ∃ k : ℕ, k > 0 ∧ k * (x₀ - 2) > x₀ * (Real.log x₀ + 1)) → 
  (∀ k : ℕ, k > 0 → (∃ x : ℝ, x > 2 ∧ k * (x - 2) > x * (Real.log x + 1)) → k ≥ 5) ∧
  (∃ x : ℝ, x > 2 ∧ 5 * (x - 2) > x * (Real.log x + 1)) :=
sorry

end NUMINAMATH_CALUDE_min_k_value_l745_74564


namespace NUMINAMATH_CALUDE_line_intersects_at_least_one_l745_74530

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the necessary relations
variable (contained_in : Line → Plane → Prop)
variable (intersects : Line → Line → Prop)
variable (skew : Line → Line → Prop)
variable (plane_intersection : Plane → Plane → Line → Prop)

-- State the theorem
theorem line_intersects_at_least_one 
  (a b l : Line) (α β : Plane) :
  skew a b →
  contained_in a α →
  contained_in b β →
  plane_intersection α β l →
  (intersects l a) ∨ (intersects l b) :=
sorry

end NUMINAMATH_CALUDE_line_intersects_at_least_one_l745_74530


namespace NUMINAMATH_CALUDE_mean_of_six_numbers_with_sum_one_third_l745_74542

theorem mean_of_six_numbers_with_sum_one_third :
  ∀ (a b c d e f : ℚ),
  a + b + c + d + e + f = 1/3 →
  (a + b + c + d + e + f) / 6 = 1/18 := by
sorry

end NUMINAMATH_CALUDE_mean_of_six_numbers_with_sum_one_third_l745_74542


namespace NUMINAMATH_CALUDE_modular_inverse_five_mod_twentysix_l745_74531

theorem modular_inverse_five_mod_twentysix :
  ∃! x : ℕ, x ∈ Finset.range 26 ∧ (5 * x) % 26 = 1 :=
by
  use 21
  sorry

end NUMINAMATH_CALUDE_modular_inverse_five_mod_twentysix_l745_74531


namespace NUMINAMATH_CALUDE_only_sqrt_6_is_quadratic_radical_l745_74524

-- Define what it means for an expression to be a quadratic radical
def is_quadratic_radical (x : ℝ) : Prop :=
  ∃ y : ℝ, y ≥ 0 ∧ x = Real.sqrt y

-- Theorem statement
theorem only_sqrt_6_is_quadratic_radical :
  is_quadratic_radical (Real.sqrt 6) ∧
  ¬is_quadratic_radical (Real.sqrt (-5)) ∧
  ¬is_quadratic_radical (8 ^ (1/3 : ℝ)) ∧
  ¬∀ a : ℝ, is_quadratic_radical (Real.sqrt a) :=
by sorry

end NUMINAMATH_CALUDE_only_sqrt_6_is_quadratic_radical_l745_74524


namespace NUMINAMATH_CALUDE_amanda_kitchen_upgrade_cost_l745_74585

/-- The cost of Amanda's kitchen upgrade -/
def kitchen_upgrade_cost (num_knobs : ℕ) (knob_cost : ℚ) (num_pulls : ℕ) (pull_cost : ℚ) : ℚ :=
  num_knobs * knob_cost + num_pulls * pull_cost

/-- Theorem stating the total cost of Amanda's kitchen upgrade -/
theorem amanda_kitchen_upgrade_cost :
  kitchen_upgrade_cost 18 2.5 8 4 = 77 := by
  sorry

end NUMINAMATH_CALUDE_amanda_kitchen_upgrade_cost_l745_74585


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_and_negatives_l745_74582

/-- The sum of the fourth powers of the first n natural numbers -/
def sum_of_fourth_powers (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) * (3 * n^2 + 3 * n - 1) / 30

/-- The sum of the fourth powers of the first 50 natural numbers and their negatives -/
theorem sum_of_fourth_powers_and_negatives : 
  2 * (sum_of_fourth_powers 50) = 1301700 := by sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_and_negatives_l745_74582


namespace NUMINAMATH_CALUDE_defeat_dragon_l745_74528

def dragonHeads (n : ℕ) : ℕ → ℕ
  | 0 => n
  | m + 1 => 
    let remaining := dragonHeads n m - 5
    if remaining ≤ 5 then 0
    else remaining + (remaining % 9)

theorem defeat_dragon (initialHeads : ℕ) (swings : ℕ) : 
  initialHeads = 198 →
  (∀ k < swings, dragonHeads initialHeads k > 5) →
  dragonHeads initialHeads swings ≤ 5 →
  swings = 40 :=
sorry

#check defeat_dragon

end NUMINAMATH_CALUDE_defeat_dragon_l745_74528


namespace NUMINAMATH_CALUDE_max_product_xyz_l745_74513

theorem max_product_xyz (x y z : ℝ) 
  (hx : x ≥ 20) (hy : y ≥ 40) (hz : z ≥ 1675) (hsum : x + y + z = 2015) : 
  x * y * z ≤ 721480000 / 27 := by
  sorry

end NUMINAMATH_CALUDE_max_product_xyz_l745_74513


namespace NUMINAMATH_CALUDE_collinear_vectors_l745_74553

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

theorem collinear_vectors (k : ℝ) :
  (∃ t : ℝ, (a.1 + k * c.1, a.2 + k * c.2) = t • (2 * b.1 - a.1, 2 * b.2 - a.2)) →
  k = -16/13 := by
sorry

end NUMINAMATH_CALUDE_collinear_vectors_l745_74553


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l745_74543

/-- The speed of a boat in still water, given downstream travel information and stream speed. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_distance : ℝ) 
  (downstream_time : ℝ) 
  (h1 : stream_speed = 6)
  (h2 : downstream_distance = 72)
  (h3 : downstream_time = 3.6) :
  downstream_distance / downstream_time - stream_speed = 14 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l745_74543


namespace NUMINAMATH_CALUDE_divisibility_of_factorial_products_l745_74512

theorem divisibility_of_factorial_products (a b : ℕ) : 
  Nat.Prime (a + b + 1) → 
  (∃ k : ℤ, (k = a.factorial * b.factorial + 1 ∨ k = a.factorial * b.factorial - 1) ∧ 
   (a + b + 1 : ℤ) ∣ k) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_factorial_products_l745_74512


namespace NUMINAMATH_CALUDE_chess_tournament_participants_l745_74563

theorem chess_tournament_participants (total_games : ℕ) (h : total_games = 231) :
  ∃ (n : ℕ), n * (n - 1) / 2 = total_games ∧ n = 22 ∧ n - 1 = 21 := by
sorry

end NUMINAMATH_CALUDE_chess_tournament_participants_l745_74563


namespace NUMINAMATH_CALUDE_no_real_roots_iff_m_less_than_neg_one_l745_74539

theorem no_real_roots_iff_m_less_than_neg_one (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - m ≠ 0) ↔ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_iff_m_less_than_neg_one_l745_74539


namespace NUMINAMATH_CALUDE_cosine_equation_solution_l745_74580

theorem cosine_equation_solution (A ω φ b : ℝ) (h_A : A > 0) :
  (∀ x, 2 * (Real.cos (x + Real.sin (2 * x)))^2 = A * Real.sin (ω * x + φ) + b) →
  A = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equation_solution_l745_74580


namespace NUMINAMATH_CALUDE_sock_pair_combinations_l745_74578

theorem sock_pair_combinations (white brown blue : ℕ) 
  (h_white : white = 5) 
  (h_brown : brown = 5) 
  (h_blue : blue = 2) 
  (h_total : white + brown + blue = 12) : 
  (white.choose 2) + (brown.choose 2) + (blue.choose 2) = 21 := by
  sorry

end NUMINAMATH_CALUDE_sock_pair_combinations_l745_74578


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l745_74532

/-- Given a quarter circle sector with radius 5 cm, the radius of the inscribed circle
    tangent to both radii and the arc is 5√2 - 5 cm. -/
theorem inscribed_circle_radius (r : ℝ) : r = 5 * Real.sqrt 2 - 5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l745_74532


namespace NUMINAMATH_CALUDE_last_digit_3_count_l745_74500

/-- The number of terms in the sequence 7^1, 7^2, ..., 7^2008 that have a last digit of 3 -/
def count_last_digit_3 : ℕ := 502

/-- The length of the sequence 7^1, 7^2, ..., 7^2008 -/
def sequence_length : ℕ := 2008

theorem last_digit_3_count :
  count_last_digit_3 = sequence_length / 4 :=
sorry

end NUMINAMATH_CALUDE_last_digit_3_count_l745_74500


namespace NUMINAMATH_CALUDE_fraction_equality_l745_74591

theorem fraction_equality : (900 ^ 2 : ℝ) / (306 ^ 2 - 294 ^ 2) = 112.5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l745_74591


namespace NUMINAMATH_CALUDE_quadratic_real_root_condition_l745_74533

theorem quadratic_real_root_condition (b : ℝ) : 
  (∃ x : ℝ, x^2 + b*x + 25 = 0) ↔ b ≤ -10 ∨ b ≥ 10 := by
sorry

end NUMINAMATH_CALUDE_quadratic_real_root_condition_l745_74533


namespace NUMINAMATH_CALUDE_BD_range_l745_74521

/-- Triangle ABC with median AD to side BC -/
structure Triangle :=
  (A B C D : ℝ × ℝ)
  (AB : ℝ)
  (AC : ℝ)
  (BC : ℝ)
  (BD : ℝ)
  (is_median : BD = BC / 2)
  (AB_eq : AB = 5)
  (AC_eq : AC = 7)

/-- The length of BD in a triangle ABC with median AD to side BC, 
    where AB = 5 and AC = 7, satisfies 1 < BD < 6 -/
theorem BD_range (t : Triangle) : 1 < t.BD ∧ t.BD < 6 := by
  sorry

end NUMINAMATH_CALUDE_BD_range_l745_74521


namespace NUMINAMATH_CALUDE_student_presentations_periods_class_presentation_periods_l745_74541

/-- Calculates the number of periods needed for all student presentations --/
theorem student_presentations_periods (total_students : ℕ) (period_length : ℕ) 
  (individual_presentation_time : ℕ) (individual_qa_time : ℕ) 
  (group_presentations : ℕ) (group_presentation_time : ℕ) : ℕ :=
  let individual_students := total_students - group_presentations
  let individual_time := individual_students * (individual_presentation_time + individual_qa_time)
  let group_time := group_presentations * group_presentation_time
  let total_time := individual_time + group_time
  (total_time + period_length - 1) / period_length

/-- The number of periods needed for the given class presentation scenario is 7 --/
theorem class_presentation_periods : 
  student_presentations_periods 32 40 5 3 4 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_student_presentations_periods_class_presentation_periods_l745_74541


namespace NUMINAMATH_CALUDE_candy_necklaces_per_pack_l745_74567

theorem candy_necklaces_per_pack (total_packs : ℕ) (opened_packs : ℕ) (leftover_necklaces : ℕ) 
  (h1 : total_packs = 9)
  (h2 : opened_packs = 4)
  (h3 : leftover_necklaces = 40) :
  leftover_necklaces / (total_packs - opened_packs) = 8 := by
  sorry

end NUMINAMATH_CALUDE_candy_necklaces_per_pack_l745_74567


namespace NUMINAMATH_CALUDE_parallel_vectors_l745_74538

def a (n : ℝ) : ℝ × ℝ := (n, -1)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (-1, 2)

theorem parallel_vectors (n : ℝ) : 
  (∃ k : ℝ, a n + b = k • c) → n = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_l745_74538


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l745_74529

/-- Given a circle with equation x^2 + y^2 = 6x + 8y - 15, 
    prove that the sum of the x and y coordinates of its center is 7. -/
theorem circle_center_coordinate_sum : 
  ∀ (h k : ℝ), 
  (∀ (x y : ℝ), x^2 + y^2 = 6*x + 8*y - 15 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 6*h - 8*k + 15)) →
  h + k = 7 := by
sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l745_74529


namespace NUMINAMATH_CALUDE_dakota_medical_bill_l745_74552

/-- Calculates the total medical bill for Dakota's hospital stay -/
def total_medical_bill (
  days : ℕ)
  (bed_charge_per_day : ℕ)
  (specialist_fee_per_hour : ℕ)
  (specialist_time_minutes : ℕ)
  (ambulance_ride_cost : ℕ)
  (iv_cost : ℕ)
  (surgery_duration_hours : ℕ)
  (surgeon_fee_per_hour : ℕ)
  (assistant_fee_per_hour : ℕ)
  (physical_therapy_fee_per_hour : ℕ)
  (physical_therapy_duration_hours : ℕ)
  (medication_a_times_per_day : ℕ)
  (medication_a_cost_per_pill : ℕ)
  (medication_b_duration_hours : ℕ)
  (medication_b_cost_per_hour : ℕ)
  (medication_c_times_per_day : ℕ)
  (medication_c_cost_per_injection : ℕ) : ℕ :=
  let bed_charges := days * bed_charge_per_day
  let specialist_fees := 2 * (specialist_fee_per_hour * specialist_time_minutes / 60) * days
  let iv_charges := days * iv_cost
  let surgery_costs := surgery_duration_hours * (surgeon_fee_per_hour + assistant_fee_per_hour)
  let physical_therapy_fees := physical_therapy_fee_per_hour * physical_therapy_duration_hours * days
  let medication_a_cost := medication_a_times_per_day * medication_a_cost_per_pill * days
  let medication_b_cost := medication_b_duration_hours * medication_b_cost_per_hour * days
  let medication_c_cost := medication_c_times_per_day * medication_c_cost_per_injection * days
  bed_charges + specialist_fees + ambulance_ride_cost + iv_charges + surgery_costs + 
  physical_therapy_fees + medication_a_cost + medication_b_cost + medication_c_cost

/-- Theorem stating that the total medical bill for Dakota's hospital stay is $11,635 -/
theorem dakota_medical_bill : 
  total_medical_bill 3 900 250 15 1800 200 2 1500 800 300 1 3 20 2 45 2 35 = 11635 := by
  sorry

end NUMINAMATH_CALUDE_dakota_medical_bill_l745_74552


namespace NUMINAMATH_CALUDE_vegetable_ghee_mixture_l745_74592

/-- The weight of one liter of brand 'a' vegetable ghee in grams -/
def weight_a : ℝ := 900

/-- The ratio of brand 'a' to brand 'b' in the mixture by volume -/
def ratio_a : ℝ := 3
def ratio_b : ℝ := 2

/-- The total volume of the mixture in liters -/
def total_volume : ℝ := 4

/-- The total weight of the mixture in grams -/
def total_weight : ℝ := 3440

/-- The weight of one liter of brand 'b' vegetable ghee in grams -/
def weight_b : ℝ := 370

theorem vegetable_ghee_mixture :
  weight_a * (ratio_a * total_volume / (ratio_a + ratio_b)) +
  weight_b * (ratio_b * total_volume / (ratio_a + ratio_b)) = total_weight :=
sorry

end NUMINAMATH_CALUDE_vegetable_ghee_mixture_l745_74592


namespace NUMINAMATH_CALUDE_distance_from_negative_two_l745_74511

theorem distance_from_negative_two : 
  {x : ℝ | |x - (-2)| = 1} = {-3, -1} := by sorry

end NUMINAMATH_CALUDE_distance_from_negative_two_l745_74511


namespace NUMINAMATH_CALUDE_maxRegions_correct_maxRegions_is_maximal_l745_74558

/-- The maximal number of regions a circle can be divided into by segments joining n points on its boundary -/
def maxRegions (n : ℕ) : ℕ :=
  Nat.choose n 4 + Nat.choose n 2 + 1

/-- Theorem stating that maxRegions gives the correct number of regions -/
theorem maxRegions_correct (n : ℕ) : 
  maxRegions n = Nat.choose n 4 + Nat.choose n 2 + 1 := by
  sorry

/-- Theorem stating that maxRegions indeed gives the maximal number of regions -/
theorem maxRegions_is_maximal (n : ℕ) :
  ∀ k : ℕ, k ≤ maxRegions n := by
  sorry

end NUMINAMATH_CALUDE_maxRegions_correct_maxRegions_is_maximal_l745_74558


namespace NUMINAMATH_CALUDE_total_wheels_in_parking_lot_l745_74586

/-- The number of wheels for a car -/
def car_wheels : ℕ := 4

/-- The number of wheels for a bike -/
def bike_wheels : ℕ := 2

/-- The number of cars in the parking lot -/
def num_cars : ℕ := 14

/-- The number of bikes in the parking lot -/
def num_bikes : ℕ := 10

/-- Theorem: The total number of wheels in the parking lot is 76 -/
theorem total_wheels_in_parking_lot : 
  num_cars * car_wheels + num_bikes * bike_wheels = 76 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_in_parking_lot_l745_74586


namespace NUMINAMATH_CALUDE_complex_equation_solution_l745_74547

theorem complex_equation_solution (z : ℂ) : (3 - z) * Complex.I = 2 → z = 3 + 2 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l745_74547


namespace NUMINAMATH_CALUDE_cost_of_dozen_pens_l745_74575

/-- Given the cost of 3 pens and 5 pencils is Rs. 150, and the ratio of the cost of one pen
    to one pencil is 5:1, prove that the cost of one dozen pens is Rs. 450. -/
theorem cost_of_dozen_pens (pen_cost pencil_cost : ℝ) 
  (h1 : 3 * pen_cost + 5 * pencil_cost = 150)
  (h2 : pen_cost = 5 * pencil_cost) : 
  12 * pen_cost = 450 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_dozen_pens_l745_74575


namespace NUMINAMATH_CALUDE_original_number_proof_l745_74555

theorem original_number_proof : 
  ∃! x : ℤ, ∃ y : ℤ, x + y = 859560 ∧ x % 456 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l745_74555


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l745_74506

theorem magnitude_of_complex_power : 
  Complex.abs ((2 + 2*Complex.I)^(3+3)) = 512 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l745_74506


namespace NUMINAMATH_CALUDE_expression_value_l745_74556

theorem expression_value (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l745_74556


namespace NUMINAMATH_CALUDE_sum_digits_first_1998_even_l745_74596

/-- The number of digits in a positive integer -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The sum of digits used to write all even integers from 2 to n -/
def sum_digits_even (n : ℕ) : ℕ := sorry

/-- The 1998th positive even integer -/
def n_1998 : ℕ := 3996

theorem sum_digits_first_1998_even : sum_digits_even n_1998 = 7440 := by sorry

end NUMINAMATH_CALUDE_sum_digits_first_1998_even_l745_74596


namespace NUMINAMATH_CALUDE_three_lines_intersection_l745_74525

/-- Three lines intersecting at a point -/
structure ThreeLines where
  a : ℝ
  l₁ : ℝ → ℝ → Prop
  l₂ : ℝ → ℝ → Prop
  l₃ : ℝ → ℝ → Prop
  h₁ : l₁ x y ↔ a * x + 2 * y + 6 = 0
  h₂ : l₂ x y ↔ x + y - 4 = 0
  h₃ : l₃ x y ↔ 2 * x - y + 1 = 0
  intersection : ∃ (x y : ℝ), l₁ x y ∧ l₂ x y ∧ l₃ x y

/-- The value of a when three lines intersect at a point -/
theorem three_lines_intersection (t : ThreeLines) : t.a = -12 := by
  sorry

end NUMINAMATH_CALUDE_three_lines_intersection_l745_74525


namespace NUMINAMATH_CALUDE_triangle_ratio_l745_74545

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →
  b = 1 →
  (1 / 2) * b * c * Real.sin A = Real.sqrt 3 →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 :=
sorry


end NUMINAMATH_CALUDE_triangle_ratio_l745_74545


namespace NUMINAMATH_CALUDE_range_of_m_for_sqrt_function_l745_74581

/-- Given a function f(x) = √(x² - 2x + 2m - 1) with domain ℝ, 
    prove that the range of m is [1, ∞) -/
theorem range_of_m_for_sqrt_function (m : ℝ) : 
  (∀ x, ∃ y, y = Real.sqrt (x^2 - 2*x + 2*m - 1)) → m ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_for_sqrt_function_l745_74581


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l745_74559

theorem probability_of_white_ball (total_balls : Nat) (red_balls white_balls : Nat) :
  total_balls = red_balls + white_balls + 1 →
  red_balls = 2 →
  white_balls = 3 →
  (white_balls : ℚ) / (total_balls - 1 : ℚ) = 3 / 5 := by
sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l745_74559


namespace NUMINAMATH_CALUDE_two_digit_numbers_property_l745_74520

-- Define a function to calculate the truncated square of a number
def truncatedSquare (n : ℕ) : ℕ := n * n + n * (n % 10) + (n % 10) * (n % 10)

-- Define the property for a two-digit number
def satisfiesProperty (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧ n = truncatedSquare (n / 10 + n % 10)

-- Theorem statement
theorem two_digit_numbers_property :
  satisfiesProperty 13 ∧
  satisfiesProperty 63 ∧
  63 - 13 = 50 ∧
  (∃ (x : ℕ), satisfiesProperty x ∧ x ≠ 13 ∧ x ≠ 63) :=
by sorry

end NUMINAMATH_CALUDE_two_digit_numbers_property_l745_74520


namespace NUMINAMATH_CALUDE_baseball_cards_pages_l745_74589

def organize_baseball_cards (new_cards : ℕ) (old_cards : ℕ) (cards_per_page : ℕ) : ℕ :=
  (new_cards + old_cards) / cards_per_page

theorem baseball_cards_pages :
  organize_baseball_cards 8 10 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_pages_l745_74589


namespace NUMINAMATH_CALUDE_min_distance_squared_l745_74566

theorem min_distance_squared (a b c d : ℝ) 
  (h : (b + 2 * a^2 - 6 * Real.log a)^2 + |2 * c - d + 6| = 0) :
  ∃ (m : ℝ), m = 20 ∧ ∀ (x y : ℝ), (x - c)^2 + (y - d)^2 ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_distance_squared_l745_74566


namespace NUMINAMATH_CALUDE_jordans_weight_loss_l745_74593

/-- Calculates Jordan's final weight after 13 weeks of an exercise program --/
theorem jordans_weight_loss (initial_weight : ℕ) : 
  initial_weight = 250 →
  (initial_weight 
    - (3 * 4)  -- Weeks 1-4
    - 5        -- Week 5
    - (2 * 4)  -- Weeks 6-9
    + 3        -- Week 10
    - (4 * 3)) -- Weeks 11-13
  = 216 := by
  sorry

#check jordans_weight_loss

end NUMINAMATH_CALUDE_jordans_weight_loss_l745_74593


namespace NUMINAMATH_CALUDE_consecutive_even_sum_l745_74598

theorem consecutive_even_sum (n k : ℕ) (hn : n > 2) (hk : k > 2) :
  ∃ a : ℤ, n * (n - 1)^(k - 1) = n * (2 * a + (n - 1)) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_sum_l745_74598


namespace NUMINAMATH_CALUDE_sum_of_xyz_l745_74518

theorem sum_of_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 40) (hxz : x * z = 80) (hyz : y * z = 120) :
  x + y + z = 22 * Real.sqrt 15 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l745_74518


namespace NUMINAMATH_CALUDE_negative_45_same_terminal_side_as_315_l745_74590

def has_same_terminal_side (α β : ℝ) : Prop :=
  ∃ k : ℤ, α = β + k * 360

theorem negative_45_same_terminal_side_as_315 :
  has_same_terminal_side (-45 : ℝ) 315 :=
sorry

end NUMINAMATH_CALUDE_negative_45_same_terminal_side_as_315_l745_74590


namespace NUMINAMATH_CALUDE_birthday_crayons_l745_74595

/-- The number of crayons Paul had left at the end of the school year -/
def crayons_left : ℕ := 134

/-- The number of crayons Paul lost or gave away -/
def crayons_lost : ℕ := 345

/-- The total number of crayons Paul got for his birthday -/
def total_crayons : ℕ := crayons_left + crayons_lost

theorem birthday_crayons : total_crayons = 479 := by
  sorry

end NUMINAMATH_CALUDE_birthday_crayons_l745_74595


namespace NUMINAMATH_CALUDE_greatest_third_side_length_l745_74519

theorem greatest_third_side_length (a b : ℝ) (ha : a = 5) (hb : b = 11) :
  ∃ (c : ℕ), c = 15 ∧ 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) ∧
  ∀ (d : ℕ), d > c → ¬((a + b > d) ∧ (a + d > b) ∧ (b + d > a)) :=
by sorry

end NUMINAMATH_CALUDE_greatest_third_side_length_l745_74519


namespace NUMINAMATH_CALUDE_yellow_ball_fraction_l745_74514

theorem yellow_ball_fraction (total : ℕ) (green blue white yellow : ℕ) : 
  (green : ℚ) / total = 1 / 4 →
  (blue : ℚ) / total = 1 / 8 →
  white = 26 →
  blue = 6 →
  total = green + blue + white + yellow →
  (yellow : ℚ) / total = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_yellow_ball_fraction_l745_74514


namespace NUMINAMATH_CALUDE_soccer_team_starters_l745_74594

def total_players : ℕ := 16
def quadruplets : ℕ := 4
def starters : ℕ := 7

theorem soccer_team_starters : 
  (Nat.choose (total_players - quadruplets) (starters - quadruplets)) = 220 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_starters_l745_74594


namespace NUMINAMATH_CALUDE_largest_circumference_error_l745_74544

theorem largest_circumference_error (actual_radius : ℝ) (error_margin : ℝ) :
  actual_radius = 15 →
  error_margin = 0.25 →
  let min_radius := actual_radius * (1 - error_margin)
  let max_radius := actual_radius * (1 + error_margin)
  let actual_circumference := 2 * Real.pi * actual_radius
  let min_circumference := 2 * Real.pi * min_radius
  let max_circumference := 2 * Real.pi * max_radius
  let min_error := (actual_circumference - min_circumference) / actual_circumference
  let max_error := (max_circumference - actual_circumference) / actual_circumference
  max min_error max_error = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_largest_circumference_error_l745_74544


namespace NUMINAMATH_CALUDE_area_of_U_l745_74516

noncomputable section

/-- A regular octagon in the complex plane -/
def RegularOctagon : Set ℂ :=
  sorry

/-- The region outside the regular octagon -/
def T : Set ℂ :=
  { z : ℂ | z ∉ RegularOctagon }

/-- The region U, which is the image of T under the transformation z ↦ 1/z -/
def U : Set ℂ :=
  { w : ℂ | ∃ z ∈ T, w = 1 / z }

/-- The area of a set in the complex plane -/
def area : Set ℂ → ℝ :=
  sorry

/-- The main theorem: The area of region U is 4 + 4π -/
theorem area_of_U : area U = 4 + 4 * Real.pi :=
  sorry

end NUMINAMATH_CALUDE_area_of_U_l745_74516


namespace NUMINAMATH_CALUDE_proposition_2_l745_74576

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Theorem statement
theorem proposition_2 
  (m n : Line) (α β γ : Plane) 
  (h1 : m ≠ n) 
  (h2 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) 
  (h3 : perpendicular m β) 
  (h4 : parallel m α) : 
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_proposition_2_l745_74576


namespace NUMINAMATH_CALUDE_julia_car_rental_cost_l745_74565

/-- Calculates the total cost of a car rental given the daily rate, per-mile charge, days rented, and miles driven. -/
def carRentalCost (dailyRate : ℝ) (perMileCharge : ℝ) (daysRented : ℕ) (milesDriven : ℝ) : ℝ :=
  dailyRate * daysRented + perMileCharge * milesDriven

/-- Proves that Julia's car rental cost is $46.12 given the specific conditions. -/
theorem julia_car_rental_cost :
  let dailyRate : ℝ := 29
  let perMileCharge : ℝ := 0.08
  let daysRented : ℕ := 1
  let milesDriven : ℝ := 214.0
  carRentalCost dailyRate perMileCharge daysRented milesDriven = 46.12 := by
  sorry

end NUMINAMATH_CALUDE_julia_car_rental_cost_l745_74565


namespace NUMINAMATH_CALUDE_min_shift_sinusoidal_graphs_l745_74535

open Real

theorem min_shift_sinusoidal_graphs : 
  let f (x : ℝ) := 2 * sin (x + π/6)
  let g (x : ℝ) := 2 * sin (x - π/3)
  ∃ φ : ℝ, φ > 0 ∧ (∀ x : ℝ, f (x - φ) = g x) ∧
    (∀ ψ : ℝ, ψ > 0 ∧ (∀ x : ℝ, f (x - ψ) = g x) → φ ≤ ψ) ∧
    φ = π/2 :=
by sorry

end NUMINAMATH_CALUDE_min_shift_sinusoidal_graphs_l745_74535


namespace NUMINAMATH_CALUDE_rectangle_with_three_tangent_circles_l745_74574

/-- Represents a circle with a center point and a radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a rectangle with width and length -/
structure Rectangle where
  width : ℝ
  length : ℝ

/-- Checks if two circles are tangent to each other -/
def are_circles_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is tangent to the sides of a rectangle -/
def is_circle_tangent_to_rectangle (c : Circle) (r : Rectangle) : Prop :=
  c.radius ≤ r.width / 2 ∧ c.radius ≤ r.length / 2

/-- Main theorem: If a rectangle contains three tangent circles (two smaller equal ones and one larger),
    and the width of the rectangle is 4, then its length is 3 + √8 -/
theorem rectangle_with_three_tangent_circles 
  (r : Rectangle) 
  (c1 c2 c3 : Circle) : 
  r.width = 4 →
  c1.radius = c2.radius →
  c1.radius < c3.radius →
  are_circles_tangent c1 c2 →
  are_circles_tangent c1 c3 →
  are_circles_tangent c2 c3 →
  is_circle_tangent_to_rectangle c1 r →
  is_circle_tangent_to_rectangle c2 r →
  is_circle_tangent_to_rectangle c3 r →
  r.length = 3 + Real.sqrt 8 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_with_three_tangent_circles_l745_74574


namespace NUMINAMATH_CALUDE_tangent_line_at_x_1_l745_74549

/-- The equation of the tangent line to y = x³ + 2x + 1 at x = 1 is 5x - y - 1 = 0 -/
theorem tangent_line_at_x_1 : 
  let f (x : ℝ) := x^3 + 2*x + 1
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let k : ℝ := (3 * x₀^2 + 2)
  ∀ x y : ℝ, (y - y₀ = k * (x - x₀)) ↔ (5*x - y - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_x_1_l745_74549


namespace NUMINAMATH_CALUDE_proposition_falsity_l745_74554

theorem proposition_falsity (P : ℕ → Prop) 
  (h_induction : ∀ k : ℕ, k > 0 → P k → P (k + 1))
  (h_false_5 : ¬ P 5) : 
  ¬ P 4 := by
  sorry

end NUMINAMATH_CALUDE_proposition_falsity_l745_74554


namespace NUMINAMATH_CALUDE_profession_assignment_l745_74573

/-- Represents the three people mentioned in the problem -/
inductive Person
  | Kondratyev
  | Davydov
  | Fedorov

/-- Represents the three professions mentioned in the problem -/
inductive Profession
  | Carpenter
  | Painter
  | Plumber

/-- Represents the age relation between two people -/
def OlderThan (a b : Person) : Prop := sorry

/-- Represents that one person has never heard of another -/
def NeverHeardOf (a b : Person) : Prop := sorry

/-- Represents the assignment of professions to people -/
def ProfessionAssignment := Person → Profession

/-- The carpenter was repairing the plumber's house -/
def CarpenterRepairingPlumbersHouse (assignment : ProfessionAssignment) : Prop := sorry

/-- The painter needed help from the carpenter -/
def PainterNeededHelpFromCarpenter (assignment : ProfessionAssignment) : Prop := sorry

/-- Main theorem: Given the conditions, prove the correct profession assignment -/
theorem profession_assignment :
  ∀ (assignment : ProfessionAssignment),
    (∀ p : Profession, ∃! person : Person, assignment person = p) →
    OlderThan Person.Davydov Person.Kondratyev →
    NeverHeardOf Person.Fedorov Person.Davydov →
    CarpenterRepairingPlumbersHouse assignment →
    PainterNeededHelpFromCarpenter assignment →
    (∀ p1 p2 : Person, assignment p1 = Profession.Plumber ∧ assignment p2 = Profession.Painter → OlderThan p1 p2) →
    assignment Person.Kondratyev = Profession.Carpenter ∧
    assignment Person.Davydov = Profession.Painter ∧
    assignment Person.Fedorov = Profession.Plumber := by
  sorry


end NUMINAMATH_CALUDE_profession_assignment_l745_74573


namespace NUMINAMATH_CALUDE_cube_edge_length_in_water_l745_74550

/-- Theorem: Edge length of a cube immersed in water --/
theorem cube_edge_length_in_water 
  (base_length : ℝ) (base_width : ℝ) (water_rise : ℝ) (a : ℝ) :
  base_length = 20 →
  base_width = 15 →
  water_rise = 11.25 →
  a^3 = base_length * base_width * water_rise →
  a = 15 :=
by sorry

end NUMINAMATH_CALUDE_cube_edge_length_in_water_l745_74550


namespace NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l745_74505

/-- The value of a pig in dollars -/
def pig_value : ℕ := 400

/-- The value of a goat in dollars -/
def goat_value : ℕ := 250

/-- A debt is resolvable if it can be expressed as a linear combination of pig and goat values -/
def is_resolvable (debt : ℕ) : Prop :=
  ∃ (p g : ℤ), debt = pig_value * p + goat_value * g

/-- The smallest positive resolvable debt -/
def smallest_resolvable_debt : ℕ := 50

theorem smallest_resolvable_debt_is_correct :
  (is_resolvable smallest_resolvable_debt) ∧
  (∀ d : ℕ, 0 < d → d < smallest_resolvable_debt → ¬(is_resolvable d)) :=
sorry

end NUMINAMATH_CALUDE_smallest_resolvable_debt_is_correct_l745_74505


namespace NUMINAMATH_CALUDE_sequence_general_term_l745_74583

theorem sequence_general_term (a : ℕ → ℝ) :
  a 1 = 1 ∧ (∀ n : ℕ, a (n + 1) = 3 * a n + 2) →
  ∀ n : ℕ, n ≥ 1 → a n = 2 * 3^(n - 1) - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l745_74583


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l745_74501

theorem boys_to_girls_ratio (total : ℕ) (diff : ℕ) : 
  total = 36 → 
  diff = 6 → 
  ∃ (boys girls : ℕ), 
    boys = girls + diff ∧ 
    boys + girls = total ∧ 
    boys * 5 = girls * 7 := by
sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l745_74501


namespace NUMINAMATH_CALUDE_perimeter_of_arranged_rectangles_l745_74561

theorem perimeter_of_arranged_rectangles :
  let small_length : ℕ := 9
  let small_width : ℕ := 3
  let horizontal_count : ℕ := 8
  let vertical_count : ℕ := 4
  let additional_edges : ℕ := 2 * 3
  let large_length : ℕ := small_length * horizontal_count
  let large_width : ℕ := small_width * vertical_count
  let perimeter : ℕ := 2 * (large_length + large_width) + additional_edges
  perimeter = 180 := by sorry

end NUMINAMATH_CALUDE_perimeter_of_arranged_rectangles_l745_74561


namespace NUMINAMATH_CALUDE_quadratic_always_has_real_root_l745_74510

theorem quadratic_always_has_real_root (b : ℝ) : 
  ∃ x : ℝ, x^2 + b*x - 20 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_always_has_real_root_l745_74510


namespace NUMINAMATH_CALUDE_complex_number_properties_l745_74557

theorem complex_number_properties (z : ℂ) (h : (2 + I) * z = 1 + 3 * I) : 
  Complex.abs z = Real.sqrt 2 ∧ z^2 - 2*z + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_properties_l745_74557


namespace NUMINAMATH_CALUDE_perpendicular_slope_l745_74551

theorem perpendicular_slope (x y : ℝ) :
  let given_line := {(x, y) | 5 * x - 2 * y = 10}
  let given_slope := 5 / 2
  let perpendicular_slope := -1 / given_slope
  perpendicular_slope = -2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l745_74551


namespace NUMINAMATH_CALUDE_infinitely_many_circled_l745_74588

/-- A sequence of natural numbers -/
def Sequence := ℕ → ℕ

/-- Predicate that checks if a number in the sequence is circled -/
def IsCircled (a : Sequence) (n : ℕ) : Prop := a n ≥ n

/-- The main theorem stating that infinitely many numbers are circled -/
theorem infinitely_many_circled (a : Sequence) : 
  Set.Infinite {n : ℕ | IsCircled a n} := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_circled_l745_74588


namespace NUMINAMATH_CALUDE_electric_water_ratio_l745_74570

def monthly_earnings : ℚ := 6000
def house_rental : ℚ := 640
def food_expense : ℚ := 380
def insurance_ratio : ℚ := 1 / 5
def remaining_money : ℚ := 2280

theorem electric_water_ratio :
  let insurance_cost := insurance_ratio * monthly_earnings
  let total_expenses := house_rental + food_expense + insurance_cost
  let electric_water_bill := monthly_earnings - total_expenses - remaining_money
  electric_water_bill / monthly_earnings = 1 / 4 := by sorry

end NUMINAMATH_CALUDE_electric_water_ratio_l745_74570


namespace NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l745_74534

theorem complex_number_in_second_quadrant : 
  let z : ℂ := 2 * Complex.I * (Complex.I + 1) + 1
  (z.re < 0) ∧ (z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_in_second_quadrant_l745_74534


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l745_74546

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are equal. -/
def symmetric_wrt_y_axis (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ = -x₂ ∧ y₁ = y₂

/-- Given point A(2, -3) is symmetric to point A'(a, b) with respect to the y-axis, prove that a + b = -5. -/
theorem symmetric_points_sum (a b : ℝ) 
  (h : symmetric_wrt_y_axis 2 (-3) a b) : a + b = -5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l745_74546


namespace NUMINAMATH_CALUDE_product_of_squares_l745_74571

theorem product_of_squares (x : ℝ) : 
  (Real.sqrt (6 + x) + Real.sqrt (21 - x) = 8) → 
  ((6 + x) * (21 - x) = 1369 / 4) := by
sorry

end NUMINAMATH_CALUDE_product_of_squares_l745_74571


namespace NUMINAMATH_CALUDE_sqrt_12_equals_2_sqrt_3_l745_74522

theorem sqrt_12_equals_2_sqrt_3 : Real.sqrt 12 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_equals_2_sqrt_3_l745_74522


namespace NUMINAMATH_CALUDE_monotonically_increasing_iff_a_geq_one_third_l745_74526

def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x - 5

theorem monotonically_increasing_iff_a_geq_one_third :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x < f a y) ↔ a ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_monotonically_increasing_iff_a_geq_one_third_l745_74526


namespace NUMINAMATH_CALUDE_claire_final_balloons_l745_74508

/-- The number of balloons Claire has at the end of the fair --/
def final_balloon_count (initial : ℕ) (floated_away : ℕ) (given_away : ℕ) (received : ℕ) : ℕ :=
  initial - floated_away - given_away + received

/-- Theorem stating that Claire ends up with 40 balloons --/
theorem claire_final_balloons :
  final_balloon_count 50 12 9 11 = 40 := by
  sorry

end NUMINAMATH_CALUDE_claire_final_balloons_l745_74508


namespace NUMINAMATH_CALUDE_transistors_2010_count_l745_74527

/-- The number of transistors in a CPU triples every two years -/
def tripling_period : ℕ := 2

/-- The initial number of transistors in 1990 -/
def initial_transistors : ℕ := 500000

/-- The number of years between 1990 and 2010 -/
def years_passed : ℕ := 20

/-- Calculate the number of transistors in 2010 -/
def transistors_2010 : ℕ := initial_transistors * (3 ^ (years_passed / tripling_period))

/-- Theorem stating that the number of transistors in 2010 is 29,524,500,000 -/
theorem transistors_2010_count : transistors_2010 = 29524500000 := by
  sorry

end NUMINAMATH_CALUDE_transistors_2010_count_l745_74527


namespace NUMINAMATH_CALUDE_min_operations_to_300_l745_74569

def Calculator (n : ℕ) : Set ℕ :=
  { m | ∃ (ops : List (ℕ → ℕ)), 
    (∀ op ∈ ops, op = (· + 1) ∨ op = (· * 2)) ∧
    ops.foldl (λ acc f => f acc) 1 = m ∧
    ops.length = n }

theorem min_operations_to_300 :
  (∀ n < 11, 300 ∉ Calculator n) ∧ 300 ∈ Calculator 11 :=
sorry

end NUMINAMATH_CALUDE_min_operations_to_300_l745_74569


namespace NUMINAMATH_CALUDE_batting_average_is_60_l745_74536

/-- A batsman's batting statistics -/
structure BattingStats where
  total_innings : ℕ
  highest_score : ℕ
  lowest_score : ℕ
  average_excluding_extremes : ℚ

/-- The batting average for all innings -/
def batting_average (stats : BattingStats) : ℚ :=
  let total_runs := stats.average_excluding_extremes * (stats.total_innings - 2 : ℚ) + stats.highest_score + stats.lowest_score
  total_runs / stats.total_innings

/-- Theorem stating the batting average for the given conditions -/
theorem batting_average_is_60 (stats : BattingStats) 
    (h1 : stats.total_innings = 46)
    (h2 : stats.highest_score = 194)
    (h3 : stats.highest_score - stats.lowest_score = 180)
    (h4 : stats.average_excluding_extremes = 58) :
    batting_average stats = 60 := by
  sorry

end NUMINAMATH_CALUDE_batting_average_is_60_l745_74536


namespace NUMINAMATH_CALUDE_absolute_value_eq_four_sum_of_absolute_values_min_value_of_sum_min_value_is_three_l745_74523

-- Problem 1
theorem absolute_value_eq_four (a : ℝ) : 
  |a + 2| = 4 ↔ a = -6 ∨ a = 2 := by sorry

-- Problem 2
theorem sum_of_absolute_values (a : ℝ) :
  -4 < a ∧ a < 2 → |a + 4| + |a - 2| = 6 := by sorry

-- Problem 3
theorem min_value_of_sum (a : ℝ) :
  (∀ x : ℝ, |x - 1| + |x + 2| ≥ |a - 1| + |a + 2|) ↔ -2 ≤ a ∧ a ≤ 1 := by sorry

theorem min_value_is_three (a : ℝ) :
  -2 ≤ a ∧ a ≤ 1 → |a - 1| + |a + 2| = 3 := by sorry

end NUMINAMATH_CALUDE_absolute_value_eq_four_sum_of_absolute_values_min_value_of_sum_min_value_is_three_l745_74523


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l745_74540

theorem complex_fraction_evaluation : 
  (((10/3 / 10 + 0.175 / 0.35) / (1.75 - (28/17) * (51/56))) - 
   ((11/18 - 1/15) / 1.4) / ((0.5 - 1/9) * 3)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l745_74540


namespace NUMINAMATH_CALUDE_second_candidate_percentage_l745_74504

theorem second_candidate_percentage : ∀ (total_marks : ℕ) (passing_mark : ℕ),
  passing_mark = 160 →
  (0.4 : ℝ) * total_marks = passing_mark - 40 →
  let second_candidate_marks := passing_mark + 20
  ((second_candidate_marks : ℝ) / total_marks) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_second_candidate_percentage_l745_74504


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l745_74515

/-- Given a hyperbola with equation x²/4 - y² = 1, prove that its asymptotes
    are described by the equations y = x/2 and y = -x/2 -/
theorem hyperbola_asymptotes :
  ∀ x y : ℝ,
  (x^2 / 4 - y^2 = 1) →
  (∃ t : ℝ, y = t*x ∧ (t = 1/2 ∨ t = -1/2)) :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l745_74515


namespace NUMINAMATH_CALUDE_table_rearrangement_l745_74548

/-- Represents a table with n rows and n columns -/
def Table (α : Type) (n : ℕ) := Fin n → Fin n → α

/-- Predicate to check if a row has no repeated elements -/
def NoRepeatsInRow {α : Type} [DecidableEq α] (row : Fin n → α) : Prop :=
  ∀ i j : Fin n, i ≠ j → row i ≠ row j

/-- Predicate to check if a table has no repeated elements in any row -/
def NoRepeatsInRows {α : Type} [DecidableEq α] (T : Table α n) : Prop :=
  ∀ i : Fin n, NoRepeatsInRow (T i)

/-- Predicate to check if two rows are permutations of each other -/
def RowsArePermutations {α : Type} [DecidableEq α] (row1 row2 : Fin n → α) : Prop :=
  ∀ x : α, (∃ i : Fin n, row1 i = x) ↔ (∃ j : Fin n, row2 j = x)

/-- Predicate to check if a column has no repeated elements -/
def NoRepeatsInColumn {α : Type} [DecidableEq α] (T : Table α n) (j : Fin n) : Prop :=
  ∀ i k : Fin n, i ≠ k → T i j ≠ T k j

/-- The main theorem statement -/
theorem table_rearrangement {α : Type} [DecidableEq α] (n : ℕ) (T : Table α n) 
  (h : NoRepeatsInRows T) :
  ∃ T_star : Table α n,
    (∀ i : Fin n, RowsArePermutations (T i) (T_star i)) ∧
    (∀ j : Fin n, NoRepeatsInColumn T_star j) :=
  sorry

end NUMINAMATH_CALUDE_table_rearrangement_l745_74548


namespace NUMINAMATH_CALUDE_employed_females_percentage_l745_74517

theorem employed_females_percentage (total_population employed_population employed_males : ℝ) :
  employed_population / total_population = 0.7 →
  employed_males / total_population = 0.21 →
  (employed_population - employed_males) / employed_population = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_employed_females_percentage_l745_74517


namespace NUMINAMATH_CALUDE_hyperbola_equation_l745_74587

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes is parallel to the line x + 3y + 2√5 = 0
    and one of its foci lies on this line, then a² = 18 and b² = 2 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (-b / a = -1 / 3) →
  (∃ (x : ℝ), x + 3 * 0 + 2 * Real.sqrt 5 = 0 ∧ x^2 = 4 * 5) →
  a^2 = 18 ∧ b^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l745_74587


namespace NUMINAMATH_CALUDE_correct_average_after_adjustment_l745_74509

theorem correct_average_after_adjustment (numbers : Finset ℝ) (initial_sum : ℝ) :
  Finset.card numbers = 25 →
  initial_sum = 337.5 →
  let adjusted_sum := initial_sum + 10 + 8.5 + 10 + 1
  adjusted_sum / Finset.card numbers = 367 / 25 := by
  sorry

#check correct_average_after_adjustment

end NUMINAMATH_CALUDE_correct_average_after_adjustment_l745_74509


namespace NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l745_74502

theorem smallest_perfect_square_divisible_by_2_3_5 :
  ∃ n : ℕ, n > 0 ∧ 
  (∃ m : ℕ, n = m^2) ∧
  2 ∣ n ∧ 3 ∣ n ∧ 5 ∣ n ∧
  (∀ k : ℕ, k > 0 → (∃ l : ℕ, k = l^2) → 2 ∣ k → 3 ∣ k → 5 ∣ k → k ≥ n) ∧
  n = 900 :=
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_divisible_by_2_3_5_l745_74502


namespace NUMINAMATH_CALUDE_largest_number_l745_74568

theorem largest_number (a b c d e : ℝ) : 
  a = 12345 + 1/5678 →
  b = 12345 - 1/5678 →
  c = 12345 * 1/5678 →
  d = 12345 / (1/5678) →
  e = 12345.5678 →
  d > a ∧ d > b ∧ d > c ∧ d > e :=
by sorry

end NUMINAMATH_CALUDE_largest_number_l745_74568


namespace NUMINAMATH_CALUDE_hyperbola_focus_distance_l745_74599

/-- The hyperbola with equation x²/16 - y²/20 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 20 = 1

/-- The left focus of the hyperbola -/
def F₁ : ℝ × ℝ := sorry

/-- The right focus of the hyperbola -/
def F₂ : ℝ × ℝ := sorry

/-- The distance between two points in ℝ² -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

theorem hyperbola_focus_distance (P : ℝ × ℝ) :
  hyperbola P.1 P.2 → distance P F₁ = 9 → distance P F₂ = 17 := by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_distance_l745_74599


namespace NUMINAMATH_CALUDE_projectile_max_height_l745_74579

/-- The height function of the projectile --/
def h (t : ℝ) : ℝ := -16 * t^2 + 64 * t + 36

/-- The maximum height reached by the projectile --/
theorem projectile_max_height :
  ∃ (max : ℝ), max = 100 ∧ ∀ (t : ℝ), h t ≤ max :=
by sorry

end NUMINAMATH_CALUDE_projectile_max_height_l745_74579


namespace NUMINAMATH_CALUDE_longest_length_is_three_smallest_square_is_1444_l745_74503

/-- A number is a perfect square with n identical non-zero last digits if it's
    a square and its last n digits in base 10 are the same and non-zero. -/
def is_perfect_square_with_n_identical_last_digits (x n : ℕ) : Prop :=
  ∃ k : ℕ, x = k^2 ∧
  ∃ d : ℕ, d ≠ 0 ∧ d < 10 ∧
  ∀ i : ℕ, i < n → (x / 10^i) % 10 = d

/-- The longest possible length for which a perfect square ends with
    n identical non-zero digits is 3. -/
theorem longest_length_is_three :
  (∀ n : ℕ, ∃ x : ℕ, is_perfect_square_with_n_identical_last_digits x n) →
  (∀ m : ℕ, m > 3 → ¬∃ x : ℕ, is_perfect_square_with_n_identical_last_digits x m) :=
sorry

/-- The smallest perfect square with 3 identical non-zero last digits is 1444. -/
theorem smallest_square_is_1444 :
  is_perfect_square_with_n_identical_last_digits 1444 3 ∧
  ∀ x : ℕ, x < 1444 → ¬is_perfect_square_with_n_identical_last_digits x 3 :=
sorry

end NUMINAMATH_CALUDE_longest_length_is_three_smallest_square_is_1444_l745_74503


namespace NUMINAMATH_CALUDE_star_calculation_l745_74572

-- Define the star operation
def star (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem star_calculation :
  star (star (star 3 5) 2) 7 = -11/10 :=
by sorry

end NUMINAMATH_CALUDE_star_calculation_l745_74572


namespace NUMINAMATH_CALUDE_difference_of_x_and_y_l745_74562

theorem difference_of_x_and_y (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (diff_squares : x^2 - y^2 = 24) : 
  x - y = 3 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_x_and_y_l745_74562
