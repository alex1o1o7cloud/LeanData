import Mathlib

namespace NUMINAMATH_GPT_part1_part2_l915_91587

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + a

theorem part1 (tangent_at_e : ∀ x : ℝ, f x e = 2 * e) : a = e := sorry

theorem part2 (m : ℝ) (a : ℝ) (hm : 0 < m) :
  (if m ≤ 1 / (2 * Real.exp 1) then 
     ∀ x ∈ Set.Icc m (2 * m), f x a ≥ f (2 * m) a 
   else if 1 / (2 * Real.exp 1) < m ∧ m < 1 / (Real.exp 1) then 
     ∀ x ∈ Set.Icc m (2 * m), f x a ≥ f (1 / (Real.exp 1)) a 
   else 
     ∀ x ∈ Set.Icc m (2 * m), f x a ≥ f m a) :=
  sorry

end NUMINAMATH_GPT_part1_part2_l915_91587


namespace NUMINAMATH_GPT_guilt_proof_l915_91534

theorem guilt_proof (X Y : Prop) (h1 : X ∨ Y) (h2 : ¬X) : Y :=
by
  sorry

end NUMINAMATH_GPT_guilt_proof_l915_91534


namespace NUMINAMATH_GPT_symmetry_proof_l915_91543

-- Define the initial point P and its reflection P' about the x-axis
def P : ℝ × ℝ := (-1, 2)
def P' : ℝ × ℝ := (-1, -2)

-- Define the property of symmetry about the x-axis
def symmetric_about_x_axis (P P' : ℝ × ℝ) : Prop :=
  P'.fst = P.fst ∧ P'.snd = -P.snd

-- The theorem to prove that point P' is symmetric to point P about the x-axis
theorem symmetry_proof : symmetric_about_x_axis P P' :=
  sorry

end NUMINAMATH_GPT_symmetry_proof_l915_91543


namespace NUMINAMATH_GPT_minimum_value_of_expr_l915_91539

noncomputable def expr (x : ℝ) : ℝ := x + (1 / (x - 5))

theorem minimum_value_of_expr : ∀ (x : ℝ), x > 5 → expr x ≥ 7 ∧ (expr x = 7 ↔ x = 6) := 
by 
  sorry

end NUMINAMATH_GPT_minimum_value_of_expr_l915_91539


namespace NUMINAMATH_GPT_prob_at_least_one_multiple_of_4_60_l915_91532

def num_multiples_of_4 (n : ℕ) : ℕ :=
  n / 4

def total_numbers_in_range (n : ℕ) : ℕ :=
  n

def num_not_multiples_of_4 (n : ℕ) : ℕ :=
  total_numbers_in_range n - num_multiples_of_4 n

def prob_no_multiple_of_4 (n : ℕ) : ℚ :=
  let p := num_not_multiples_of_4 n / total_numbers_in_range n
  p * p

def prob_at_least_one_multiple_of_4 (n : ℕ) : ℚ :=
  1 - prob_no_multiple_of_4 n

theorem prob_at_least_one_multiple_of_4_60 :
  prob_at_least_one_multiple_of_4 60 = 7 / 16 :=
by
  -- Proof is skipped.
  sorry

end NUMINAMATH_GPT_prob_at_least_one_multiple_of_4_60_l915_91532


namespace NUMINAMATH_GPT_cylindrical_to_rectangular_l915_91566

theorem cylindrical_to_rectangular (r θ z : ℝ) (hr : r = 10) (hθ : θ = 3 * Real.pi / 4) (hz : z = 2) :
    ∃ (x y z' : ℝ), (x = r * Real.cos θ) ∧ (y = r * Real.sin θ) ∧ (z' = z) ∧ (x = -5 * Real.sqrt 2) ∧ (y = 5 * Real.sqrt 2) ∧ (z' = 2) :=
by
  sorry

end NUMINAMATH_GPT_cylindrical_to_rectangular_l915_91566


namespace NUMINAMATH_GPT_trapezoid_area_division_l915_91535

/-- Given a trapezoid where one base is 150 units longer than the other base and the segment joining the midpoints of the legs divides the trapezoid into two regions whose areas are in the ratio 3:4, prove that the greatest integer less than or equal to (x^2 / 150) is 300, where x is the length of the segment that joins the midpoints of the legs and divides the trapezoid into two equal areas. -/
theorem trapezoid_area_division (b h x : ℝ) (h_b : b = 112.5) (h_x : x = 150) :
  ⌊x^2 / 150⌋ = 300 :=
by
  sorry

end NUMINAMATH_GPT_trapezoid_area_division_l915_91535


namespace NUMINAMATH_GPT_difference_between_sevens_l915_91536

-- Define the numeral
def numeral : ℕ := 54179759

-- Define a function to find the place value of a digit at a specific position in a number
def place_value (n : ℕ) (pos : ℕ) : ℕ :=
  let digit := (n / 10^pos) % 10
  digit * 10^pos

-- Define specific place values for the two sevens
def first_seven_place : ℕ := place_value numeral 4  -- Ten-thousands place
def second_seven_place : ℕ := place_value numeral 1 -- Tens place

-- Define their values
def first_seven_value : ℕ := 7 * 10^4  -- 70,000
def second_seven_value : ℕ := 7 * 10^1  -- 70

-- Prove the difference between these place values
theorem difference_between_sevens : first_seven_value - second_seven_value = 69930 := by
  sorry

end NUMINAMATH_GPT_difference_between_sevens_l915_91536


namespace NUMINAMATH_GPT_calories_per_strawberry_l915_91596

theorem calories_per_strawberry (x : ℕ) :
  (12 * x + 6 * 17 = 150) → x = 4 := by
  sorry

end NUMINAMATH_GPT_calories_per_strawberry_l915_91596


namespace NUMINAMATH_GPT_express_h_l915_91510

variable (a b S h : ℝ)
variable (h_formula : S = 1/2 * (a + b) * h)
variable (h_nonzero : a + b ≠ 0)

theorem express_h : h = 2 * S / (a + b) := 
by 
  sorry

end NUMINAMATH_GPT_express_h_l915_91510


namespace NUMINAMATH_GPT_intersection_M_N_l915_91598

def M : Set ℝ := { x | Real.exp (x - 1) > 1 }
def N : Set ℝ := { x | x^2 - 2*x - 3 < 0 }

theorem intersection_M_N :
  (M ∩ N : Set ℝ) = { x | 1 < x ∧ x < 3 } := 
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l915_91598


namespace NUMINAMATH_GPT_inequality_holds_for_all_y_l915_91513

theorem inequality_holds_for_all_y (x : ℝ) :
  (∀ y : ℝ, y^2 - (5^x - 1) * (y - 1) > 0) ↔ (0 < x ∧ x < 1) :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_for_all_y_l915_91513


namespace NUMINAMATH_GPT_amoeba_growth_after_5_days_l915_91514

theorem amoeba_growth_after_5_days : (3 : ℕ)^5 = 243 := by
  sorry

end NUMINAMATH_GPT_amoeba_growth_after_5_days_l915_91514


namespace NUMINAMATH_GPT_problem_product_of_areas_eq_3600x6_l915_91556

theorem problem_product_of_areas_eq_3600x6 
  (x : ℝ) 
  (bottom_area : ℝ) 
  (side_area : ℝ) 
  (front_area : ℝ)
  (bottom_area_eq : bottom_area = 12 * x ^ 2)
  (side_area_eq : side_area = 15 * x ^ 2)
  (front_area_eq : front_area = 20 * x ^ 2)
  (dimensions_proportional : ∃ a b c : ℝ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x 
                            ∧ bottom_area = a * b ∧ side_area = a * c ∧ front_area = b * c)
  : bottom_area * side_area * front_area = 3600 * x ^ 6 :=
by 
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_problem_product_of_areas_eq_3600x6_l915_91556


namespace NUMINAMATH_GPT_cyclists_meet_at_starting_point_l915_91516

/--
Given a circular track of length 1200 meters, and three cyclists with speeds of 36 kmph, 54 kmph, and 72 kmph,
prove that all three cyclists will meet at the starting point for the first time after 4 minutes.
-/
theorem cyclists_meet_at_starting_point :
  let track_length := 1200
  let speed_a_kmph := 36
  let speed_b_kmph := 54
  let speed_c_kmph := 72
  
  let speed_a_m_per_min := speed_a_kmph * 1000 / 60
  let speed_b_m_per_min := speed_b_kmph * 1000 / 60
  let speed_c_m_per_min := speed_c_kmph * 1000 / 60
  
  let time_a := track_length / speed_a_m_per_min
  let time_b := track_length / speed_b_m_per_min
  let time_c := track_length / speed_c_m_per_min
  
  let lcm := (2 : ℚ)

  (time_a = 2) ∧ (time_b = 4 / 3) ∧ (time_c = 1) → 
  ∀ t, t = lcm * 3 → t = 12 / 3 → t = 4 :=
by
  sorry

end NUMINAMATH_GPT_cyclists_meet_at_starting_point_l915_91516


namespace NUMINAMATH_GPT_even_quadratic_iff_b_zero_l915_91505

-- Define a quadratic function
def quadratic (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- State the theorem
theorem even_quadratic_iff_b_zero (a b c : ℝ) : 
  (∀ x : ℝ, quadratic a b c x = quadratic a b c (-x)) ↔ b = 0 := 
by
  sorry

end NUMINAMATH_GPT_even_quadratic_iff_b_zero_l915_91505


namespace NUMINAMATH_GPT_gym_class_students_l915_91583

theorem gym_class_students :
  ∃ n : ℕ, 150 ≤ n ∧ n ≤ 300 ∧ n % 6 = 3 ∧ n % 8 = 5 ∧ n % 9 = 2 ∧ (n = 165 ∨ n = 237) :=
by
  sorry

end NUMINAMATH_GPT_gym_class_students_l915_91583


namespace NUMINAMATH_GPT_number_of_members_l915_91519

theorem number_of_members (n : ℕ) (h1 : n * n = 5929) : n = 77 :=
sorry

end NUMINAMATH_GPT_number_of_members_l915_91519


namespace NUMINAMATH_GPT_inverse_linear_intersection_l915_91582

theorem inverse_linear_intersection (m n : ℝ) 
  (h1 : n = 2 / m) 
  (h2 : n = m + 3) 
  : (1 / m) - (1 / n) = 3 / 2 := 
by sorry

end NUMINAMATH_GPT_inverse_linear_intersection_l915_91582


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l915_91550

variable (b r d v t : ℝ)

theorem boat_speed_in_still_water (hr : r = 3) 
                                 (hd : d = 3.6) 
                                 (ht : t = 1/5) 
                                 (hv : v = b + r) 
                                 (dist_eq : d = v * t) : 
  b = 15 := 
by
  sorry

end NUMINAMATH_GPT_boat_speed_in_still_water_l915_91550


namespace NUMINAMATH_GPT_sum_of_bases_l915_91563

theorem sum_of_bases (R_1 R_2 : ℕ) 
  (hF1 : (4 * R_1 + 8) / (R_1 ^ 2 - 1) = (3 * R_2 + 6) / (R_2 ^ 2 - 1))
  (hF2 : (8 * R_1 + 4) / (R_1 ^ 2 - 1) = (6 * R_2 + 3) / (R_2 ^ 2 - 1)) : 
  R_1 + R_2 = 21 :=
sorry

end NUMINAMATH_GPT_sum_of_bases_l915_91563


namespace NUMINAMATH_GPT_sum_of_squares_l915_91574

theorem sum_of_squares (b j s : ℕ) (h : b + j + s = 34) : b^2 + j^2 + s^2 = 406 :=
sorry

end NUMINAMATH_GPT_sum_of_squares_l915_91574


namespace NUMINAMATH_GPT_bijection_condition_l915_91537

variable {n m : ℕ}
variable (f : Fin n → Fin n)

theorem bijection_condition (h_even : m % 2 = 0)
(h_prime : Nat.Prime (n + 1))
(h_bij : Function.Bijective f) :
  ∀ x y : Fin n, (n : ℕ) ∣ (m * x - y : ℕ) → (n + 1) ∣ (f x).val ^ m - (f y).val := sorry

end NUMINAMATH_GPT_bijection_condition_l915_91537


namespace NUMINAMATH_GPT_num_of_valid_three_digit_numbers_l915_91529

def valid_three_digit_numbers : ℕ :=
  let valid_numbers : List (ℕ × ℕ × ℕ) :=
    [(2, 3, 4), (4, 6, 8)]
  valid_numbers.length

theorem num_of_valid_three_digit_numbers :
  valid_three_digit_numbers = 2 :=
by
  sorry

end NUMINAMATH_GPT_num_of_valid_three_digit_numbers_l915_91529


namespace NUMINAMATH_GPT_rachel_makes_money_l915_91557

theorem rachel_makes_money (cost_per_bar total_bars remaining_bars : ℕ) (h_cost : cost_per_bar = 2) (h_total : total_bars = 13) (h_remaining : remaining_bars = 4) :
  cost_per_bar * (total_bars - remaining_bars) = 18 :=
by 
  sorry

end NUMINAMATH_GPT_rachel_makes_money_l915_91557


namespace NUMINAMATH_GPT_linear_equation_a_neg2_l915_91581

theorem linear_equation_a_neg2 (a : ℝ) :
  (∃ x y : ℝ, (a - 2) * x ^ (|a| - 1) + 3 * y = 1) ∧
  (∀ x : ℝ, x ≠ 0 → x ^ (|a| - 1) ≠ 1) →
  a = -2 :=
by
  sorry

end NUMINAMATH_GPT_linear_equation_a_neg2_l915_91581


namespace NUMINAMATH_GPT_adam_tickets_left_l915_91540

-- Define the initial number of tickets, cost per ticket, and total spending on the ferris wheel
def initial_tickets : ℕ := 13
def cost_per_ticket : ℕ := 9
def total_spent : ℕ := 81

-- Define the number of tickets Adam has after riding the ferris wheel
def tickets_left (initial_tickets cost_per_ticket total_spent : ℕ) : ℕ :=
  initial_tickets - (total_spent / cost_per_ticket)

-- Proposition to prove that Adam has 4 tickets left
theorem adam_tickets_left : tickets_left initial_tickets cost_per_ticket total_spent = 4 :=
by
  sorry

end NUMINAMATH_GPT_adam_tickets_left_l915_91540


namespace NUMINAMATH_GPT_cake_angle_between_adjacent_pieces_l915_91562

theorem cake_angle_between_adjacent_pieces 
  (total_angle : ℝ := 360)
  (total_pieces : ℕ := 10)
  (eaten_pieces : ℕ := 1)
  (angle_per_piece := total_angle / total_pieces)
  (remaining_pieces := total_pieces - eaten_pieces)
  (new_angle_per_piece := total_angle / remaining_pieces) :
  (new_angle_per_piece - angle_per_piece = 4) := 
by
  sorry

end NUMINAMATH_GPT_cake_angle_between_adjacent_pieces_l915_91562


namespace NUMINAMATH_GPT_right_triangle_perimeter_l915_91553

def right_triangle_circumscribed_perimeter (r c : ℝ) (a b : ℝ) : ℝ :=
  a + b + c

theorem right_triangle_perimeter : 
  ∀ (a b : ℝ),
  (4 : ℝ) * (a + b + (26 : ℝ)) = a * b ∧ a^2 + b^2 = (26 : ℝ)^2 →
  right_triangle_circumscribed_perimeter 4 26 a b = 60 := sorry

end NUMINAMATH_GPT_right_triangle_perimeter_l915_91553


namespace NUMINAMATH_GPT_students_in_only_one_subject_l915_91597

variables (A B C : ℕ) 
variables (A_inter_B A_inter_C B_inter_C A_inter_B_inter_C : ℕ)

def students_in_one_subject (A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C : ℕ) : ℕ :=
  A + B + C - A_inter_B - A_inter_C - B_inter_C + A_inter_B_inter_C - 2 * A_inter_B_inter_C

theorem students_in_only_one_subject :
  ∀ (A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C : ℕ),
    A = 29 →
    B = 28 →
    C = 27 →
    A_inter_B = 13 →
    A_inter_C = 12 →
    B_inter_C = 11 →
    A_inter_B_inter_C = 5 →
    students_in_one_subject A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C = 27 :=
by
  intros A B C A_inter_B A_inter_C B_inter_C A_inter_B_inter_C hA hB hC hAB hAC hBC hABC
  unfold students_in_one_subject
  rw [hA, hB, hC, hAB, hAC, hBC, hABC]
  norm_num
  sorry

end NUMINAMATH_GPT_students_in_only_one_subject_l915_91597


namespace NUMINAMATH_GPT_proof_problem_l915_91541

open Set

variable (U : Set ℕ)
variable (P : Set ℕ)
variable (Q : Set ℕ)

noncomputable def problem_statement : Set ℕ :=
  compl (P ∪ Q) ∩ U

theorem proof_problem :
  U = {1, 2, 3, 4} →
  P = {1, 2} →
  Q = {2, 3} →
  compl (P ∪ Q) ∩ U = {4} :=
by
  intros hU hP hQ
  rw [hU, hP, hQ]
  sorry

end NUMINAMATH_GPT_proof_problem_l915_91541


namespace NUMINAMATH_GPT_solve_problems_l915_91595

theorem solve_problems (x y : ℕ) (hx : x + y = 14) (hy : 7 * x - 12 * y = 60) : x = 12 :=
sorry

end NUMINAMATH_GPT_solve_problems_l915_91595


namespace NUMINAMATH_GPT_equal_cylinder_volumes_l915_91552

theorem equal_cylinder_volumes (x : ℝ) (hx : x > 0) :
  π * (5 + x) ^ 2 * 4 = π * 25 * (4 + x) → x = 35 / 4 :=
by
  sorry

end NUMINAMATH_GPT_equal_cylinder_volumes_l915_91552


namespace NUMINAMATH_GPT_gross_pay_calculation_l915_91530

theorem gross_pay_calculation
    (NetPay : ℕ) (Taxes : ℕ) (GrossPay : ℕ) 
    (h1 : NetPay = 315) 
    (h2 : Taxes = 135) 
    (h3 : GrossPay = NetPay + Taxes) : 
    GrossPay = 450 :=
by
    -- We need to prove this part
    sorry

end NUMINAMATH_GPT_gross_pay_calculation_l915_91530


namespace NUMINAMATH_GPT_ant_to_vertices_probability_l915_91526

noncomputable def event_A_probability : ℝ :=
  1 - (Real.sqrt 3 * Real.pi / 24)

theorem ant_to_vertices_probability :
  let side_length := 4
  let event_A := "the distance from the ant to all three vertices is more than 1"
  event_A_probability = 1 - Real.sqrt 3 * Real.pi / 24
:=
sorry

end NUMINAMATH_GPT_ant_to_vertices_probability_l915_91526


namespace NUMINAMATH_GPT_fraction_calculation_l915_91568

theorem fraction_calculation : (36 - 12) / (12 - 4) = 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_calculation_l915_91568


namespace NUMINAMATH_GPT_crab_ratio_l915_91590

theorem crab_ratio 
  (oysters_day1 : ℕ) 
  (crabs_day1 : ℕ) 
  (total_days : ℕ) 
  (oysters_ratio : ℕ) 
  (oysters_day2 : ℕ) 
  (total_oysters_crabs : ℕ) 
  (crabs_day2 : ℕ) 
  (ratio : ℚ) :
  oysters_day1 = 50 →
  crabs_day1 = 72 →
  oysters_ratio = 2 →
  oysters_day2 = oysters_day1 / oysters_ratio →
  total_oysters_crabs = 195 →
  total_oysters_crabs = oysters_day1 + crabs_day1 + oysters_day2 + crabs_day2 →
  crabs_day2 = total_oysters_crabs - (oysters_day1 + crabs_day1 + oysters_day2) →
  ratio = (crabs_day2 : ℚ) / crabs_day1 →
  ratio = 2 / 3 :=
by
  intro h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_crab_ratio_l915_91590


namespace NUMINAMATH_GPT_pastries_and_juices_count_l915_91524

theorem pastries_and_juices_count 
  (budget : ℕ) 
  (cost_per_pastry : ℕ) 
  (cost_per_juice : ℕ) 
  (total_money : budget = 50)
  (pastry_cost : cost_per_pastry = 7) 
  (juice_cost : cost_per_juice = 2) : 
  ∃ (p j : ℕ), 7 * p + 2 * j ≤ 50 ∧ p + j = 7 :=
by
  sorry

end NUMINAMATH_GPT_pastries_and_juices_count_l915_91524


namespace NUMINAMATH_GPT_triangle_side_ratio_l915_91572

variable (A B C : ℝ)  -- angles in radians
variable (a b c : ℝ)  -- sides of triangle

theorem triangle_side_ratio
  (h : a * Real.sin A * Real.sin B + b * Real.cos A ^ 2 = Real.sqrt 2 * a) :
  b / a = Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_triangle_side_ratio_l915_91572


namespace NUMINAMATH_GPT_exists_three_distinct_integers_in_A_l915_91560

noncomputable def A (m n : ℤ) : Set ℤ := { x^2 + m * x + n | x : ℤ }

theorem exists_three_distinct_integers_in_A (m n : ℤ) :
  ∃ a b c : ℤ, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a ∈ A m n ∧ b ∈ A m n ∧ c ∈ A m n ∧ a = b * c :=
by
  sorry

end NUMINAMATH_GPT_exists_three_distinct_integers_in_A_l915_91560


namespace NUMINAMATH_GPT_systematic_sampling_first_group_l915_91571

theorem systematic_sampling_first_group
  (a : ℕ → ℕ)
  (d : ℕ)
  (n : ℕ)
  (a₁ : ℕ)
  (a₁₆ : ℕ)
  (h₁ : d = 8)
  (h₂ : a 16 = a₁₆)
  (h₃ : a₁₆ = 125)
  (h₄ : a n = a₁ + (n - 1) * d) :
  a 1 = 5 :=
by
  sorry

end NUMINAMATH_GPT_systematic_sampling_first_group_l915_91571


namespace NUMINAMATH_GPT_simplify_fraction_l915_91559

theorem simplify_fraction (a b m : ℝ) (h1 : (a / b) ^ m = (a^m) / (b^m)) (h2 : (-1 : ℝ) ^ (0 : ℝ) = 1) :
  ( (81 / 16) ^ (3 / 4) ) - 1 = 19 / 8 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l915_91559


namespace NUMINAMATH_GPT_average_cost_is_70_l915_91578

noncomputable def C_before_gratuity (total_bill : ℝ) (gratuity_rate : ℝ) : ℝ :=
  total_bill / (1 + gratuity_rate)

noncomputable def average_cost_per_individual (C : ℝ) (total_people : ℝ) : ℝ :=
  C / total_people

theorem average_cost_is_70 :
  let total_bill := 756
  let gratuity_rate := 0.20
  let total_people := 9
  average_cost_per_individual (C_before_gratuity total_bill gratuity_rate) total_people = 70 :=
by
  sorry

end NUMINAMATH_GPT_average_cost_is_70_l915_91578


namespace NUMINAMATH_GPT_negation_is_false_l915_91525

-- Define the proposition and its negation
def proposition (x y : ℝ) : Prop := (x > 2 ∧ y > 3) → (x + y > 5)
def negation_proposition (x y : ℝ) : Prop := ¬ proposition x y

-- The proposition and its negation
theorem negation_is_false : ∀ (x y : ℝ), negation_proposition x y = false :=
by sorry

end NUMINAMATH_GPT_negation_is_false_l915_91525


namespace NUMINAMATH_GPT_elena_pen_cost_l915_91564

theorem elena_pen_cost (cost_X : ℝ) (cost_Y : ℝ) (total_pens : ℕ) (brand_X_pens : ℕ) 
    (purchased_X_cost : cost_X = 4.0) (purchased_Y_cost : cost_Y = 2.8)
    (total_pens_condition : total_pens = 12) (brand_X_pens_condition : brand_X_pens = 8) :
    cost_X * brand_X_pens + cost_Y * (total_pens - brand_X_pens) = 43.20 :=
    sorry

end NUMINAMATH_GPT_elena_pen_cost_l915_91564


namespace NUMINAMATH_GPT_replaced_person_is_65_l915_91554

-- Define the conditions of the problem context
variable (W : ℝ)
variable (avg_increase : ℝ := 3.5)
variable (num_persons : ℕ := 8)
variable (new_person_weight : ℝ := 93)

-- Express the given condition in the problem: 
-- The total increase in weight is given by the number of persons multiplied by the average increase in weight
def total_increase : ℝ := num_persons * avg_increase

-- Express the relationship between the new person's weight and the person who was replaced
def replaced_person_weight (W : ℝ) : ℝ := new_person_weight - total_increase

-- Stating the theorem to be proved
theorem replaced_person_is_65 : replaced_person_weight W = 65 := by
  sorry

end NUMINAMATH_GPT_replaced_person_is_65_l915_91554


namespace NUMINAMATH_GPT_abc_cubed_sum_l915_91500

theorem abc_cubed_sum (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
    (h_eq : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) : 
    a^3 + b^3 + c^3 = -36 :=
by sorry

end NUMINAMATH_GPT_abc_cubed_sum_l915_91500


namespace NUMINAMATH_GPT_composite_shape_sum_l915_91588

def triangular_prism_faces := 5
def triangular_prism_edges := 9
def triangular_prism_vertices := 6

def pentagonal_prism_additional_faces := 7
def pentagonal_prism_additional_edges := 10
def pentagonal_prism_additional_vertices := 5

def pyramid_additional_faces := 5
def pyramid_additional_edges := 5
def pyramid_additional_vertices := 1

def resulting_shape_faces := triangular_prism_faces - 1 + pentagonal_prism_additional_faces + pyramid_additional_faces
def resulting_shape_edges := triangular_prism_edges + pentagonal_prism_additional_edges + pyramid_additional_edges
def resulting_shape_vertices := triangular_prism_vertices + pentagonal_prism_additional_vertices + pyramid_additional_vertices

def sum_faces_edges_vertices := resulting_shape_faces + resulting_shape_edges + resulting_shape_vertices

theorem composite_shape_sum : sum_faces_edges_vertices = 51 :=
by
  unfold sum_faces_edges_vertices resulting_shape_faces resulting_shape_edges resulting_shape_vertices
  unfold triangular_prism_faces triangular_prism_edges triangular_prism_vertices
  unfold pentagonal_prism_additional_faces pentagonal_prism_additional_edges pentagonal_prism_additional_vertices
  unfold pyramid_additional_faces pyramid_additional_edges pyramid_additional_vertices
  simp
  sorry

end NUMINAMATH_GPT_composite_shape_sum_l915_91588


namespace NUMINAMATH_GPT_total_eyes_correct_l915_91576

-- Conditions
def boys := 21 * 2 + 2 * 1
def girls := 15 * 2 + 3 * 1
def cats := 8 * 2 + 2 * 1
def spiders := 4 * 8 + 1 * 6

-- Total count of eyes
def total_eyes := boys + girls + cats + spiders

theorem total_eyes_correct: total_eyes = 133 :=
by 
  -- Here the proof steps would go, which we are skipping
  sorry

end NUMINAMATH_GPT_total_eyes_correct_l915_91576


namespace NUMINAMATH_GPT_two_person_subcommittees_from_eight_l915_91567

theorem two_person_subcommittees_from_eight :
  (Nat.choose 8 2) = 28 :=
by
  sorry

end NUMINAMATH_GPT_two_person_subcommittees_from_eight_l915_91567


namespace NUMINAMATH_GPT_games_played_in_tournament_l915_91542

def number_of_games (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem games_played_in_tournament : number_of_games 18 = 153 :=
  by
    sorry

end NUMINAMATH_GPT_games_played_in_tournament_l915_91542


namespace NUMINAMATH_GPT_cupcakes_left_at_home_correct_l915_91551

-- Definitions of the conditions
def total_cupcakes_baked : ℕ := 53
def boxes_given_away : ℕ := 17
def cupcakes_per_box : ℕ := 3

-- Calculate the total number of cupcakes given away
def total_cupcakes_given_away := boxes_given_away * cupcakes_per_box

-- Calculate the number of cupcakes left at home
def cupcakes_left_at_home := total_cupcakes_baked - total_cupcakes_given_away

-- Prove that the number of cupcakes left at home is 2
theorem cupcakes_left_at_home_correct : cupcakes_left_at_home = 2 := by
  sorry

end NUMINAMATH_GPT_cupcakes_left_at_home_correct_l915_91551


namespace NUMINAMATH_GPT_solve_for_y_l915_91521

noncomputable def log5 (x : ℝ) : ℝ := (Real.log x) / (Real.log 5)

theorem solve_for_y (y : ℝ) (h₀ : log5 ((2 * y + 10) / (3 * y - 6)) + log5 ((3 * y - 6) / (y - 4)) = 3) : 
  y = 170 / 41 :=
sorry

end NUMINAMATH_GPT_solve_for_y_l915_91521


namespace NUMINAMATH_GPT_bekah_days_left_l915_91548

theorem bekah_days_left 
  (total_pages : ℕ)
  (pages_read : ℕ)
  (pages_per_day : ℕ)
  (remaining_pages : ℕ := total_pages - pages_read)
  (days_left : ℕ := remaining_pages / pages_per_day) :
  total_pages = 408 →
  pages_read = 113 →
  pages_per_day = 59 →
  days_left = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_bekah_days_left_l915_91548


namespace NUMINAMATH_GPT_remainder_is_zero_l915_91501

def f (x : ℝ) : ℝ := x^3 - 5 * x^2 + 2 * x + 8

theorem remainder_is_zero : f 2 = 0 := by
  sorry

end NUMINAMATH_GPT_remainder_is_zero_l915_91501


namespace NUMINAMATH_GPT_find_age_l915_91544

theorem find_age (x : ℕ) (h : 5 * (x + 5) - 5 * (x - 5) = x) : x = 50 :=
by
  sorry

end NUMINAMATH_GPT_find_age_l915_91544


namespace NUMINAMATH_GPT_compute_product_l915_91586

theorem compute_product : (100 - 5) * (100 + 5) = 9975 := by
  sorry

end NUMINAMATH_GPT_compute_product_l915_91586


namespace NUMINAMATH_GPT_triangle_side_length_l915_91520

variable (A C : ℝ) (a c b : ℝ)

theorem triangle_side_length (h1 : c = 48) (h2 : a = 27) (h3 : C = 3 * A) : b = 35 := by
  sorry

end NUMINAMATH_GPT_triangle_side_length_l915_91520


namespace NUMINAMATH_GPT_impossibility_exchange_l915_91509

theorem impossibility_exchange :
  ¬ ∃ (x y z : ℕ), (x + y + z = 10) ∧ (x + 3 * y + 5 * z = 25) := 
by
  sorry

end NUMINAMATH_GPT_impossibility_exchange_l915_91509


namespace NUMINAMATH_GPT_angle_AC_B₁C₁_is_60_l915_91517

-- Redefine the conditions of the problem using Lean definitions
-- We define a regular triangular prism, equilateral triangle condition,
-- and parallel lines relation.

structure TriangularPrism :=
  (A B C A₁ B₁ C₁ : Type)
  (is_regular : Prop) -- Property stating it is a regular triangular prism
  (base_is_equilateral : Prop) -- Property stating the base is an equilateral triangle
  (B₁C₁_parallel_to_BC : Prop) -- Property stating B₁C₁ is parallel to BC

-- Assume a regular triangular prism with the given properties
variable (prism : TriangularPrism)
axiom isRegularPrism : prism.is_regular
axiom baseEquilateral : prism.base_is_equilateral
axiom parallelLines : prism.B₁C₁_parallel_to_BC

-- Define the angle calculation statement in Lean 4
theorem angle_AC_B₁C₁_is_60 :
  ∃ (angle : ℝ), angle = 60 :=
by
  -- Proof is omitted using sorry
  exact ⟨60, sorry⟩

end NUMINAMATH_GPT_angle_AC_B₁C₁_is_60_l915_91517


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l915_91502

variables {Point Line Plane : Type} 

-- Definitions for the problem conditions
def is_subset_of (a : Line) (α : Plane) : Prop := sorry
def parallel_plane (a : Line) (β : Plane) : Prop := sorry
def parallel_lines (a b : Line) : Prop := sorry
def parallel_planes (α β : Plane) : Prop := sorry

-- The statement of the problem
theorem necessary_but_not_sufficient_condition (a b : Line) (α β : Plane) 
  (h1 : is_subset_of a α) (h2 : is_subset_of b β) :
  (parallel_plane a β ∧ parallel_plane b α) ↔ 
  (¬ parallel_planes α β ∧ sorry) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l915_91502


namespace NUMINAMATH_GPT_max_subjects_per_teacher_l915_91549

theorem max_subjects_per_teacher (math_teachers physics_teachers chemistry_teachers min_teachers : ℕ)
  (h_math : math_teachers = 4)
  (h_physics : physics_teachers = 3)
  (h_chemistry : chemistry_teachers = 3)
  (h_min_teachers : min_teachers = 5) :
  (math_teachers + physics_teachers + chemistry_teachers) / min_teachers = 2 :=
by
  sorry

end NUMINAMATH_GPT_max_subjects_per_teacher_l915_91549


namespace NUMINAMATH_GPT_complement_union_in_universe_l915_91506

variable (U : Set ℕ := {1, 2, 3, 4, 5})
variable (M : Set ℕ := {1, 3})
variable (N : Set ℕ := {1, 2})

theorem complement_union_in_universe :
  (U \ (M ∪ N)) = {4, 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_in_universe_l915_91506


namespace NUMINAMATH_GPT_not_and_implication_l915_91555

variable (p q : Prop)

theorem not_and_implication : ¬ (p ∧ q) → (¬ p ∨ ¬ q) :=
by
  sorry

end NUMINAMATH_GPT_not_and_implication_l915_91555


namespace NUMINAMATH_GPT_score_recording_l915_91592

theorem score_recording (avg : ℤ) (h : avg = 0) : 
  (9 = avg + 9) ∧ (-18 = avg - 18) ∧ (-2 = avg - 2) :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_score_recording_l915_91592


namespace NUMINAMATH_GPT_road_completion_l915_91512

/- 
  The company "Roga and Kopyta" undertook a project to build a road 100 km long. 
  The construction plan is: 
  - In the first month, 1 km of the road will be built.
  - Subsequently, if by the beginning of some month A km is already completed, then during that month an additional 1 / A^10 km of road will be constructed.
  Prove that the road will be completed within 100^11 months.
-/

theorem road_completion (L : ℕ → ℝ) (h1 : L 1 = 1)
  (h2 : ∀ n ≥ 1, L (n + 1) = L n + 1 / (L n) ^ 10) :
  ∃ m ≤ 100 ^ 11, L m ≥ 100 := 
  sorry

end NUMINAMATH_GPT_road_completion_l915_91512


namespace NUMINAMATH_GPT_rectangle_area_excluding_hole_l915_91546

theorem rectangle_area_excluding_hole (x : ℝ) (h : x > 5 / 3) :
  let A_large := (2 * x + 4) * (x + 7)
  let A_hole := (x + 2) * (3 * x - 5)
  A_large - A_hole = -x^2 + 17 * x + 38 :=
by
  let A_large := (2 * x + 4) * (x + 7)
  let A_hole := (x + 2) * (3 * x - 5)
  sorry

end NUMINAMATH_GPT_rectangle_area_excluding_hole_l915_91546


namespace NUMINAMATH_GPT_variable_v_value_l915_91593

theorem variable_v_value (w x v : ℝ) (h1 : 2 / w + 2 / x = 2 / v) (h2 : w * x = v) (h3 : (w + x) / 2 = 0.5) :
  v = 0.25 :=
sorry

end NUMINAMATH_GPT_variable_v_value_l915_91593


namespace NUMINAMATH_GPT_speed_of_stream_l915_91538

theorem speed_of_stream (downstream_speed upstream_speed : ℝ) (h1 : downstream_speed = 11) (h2 : upstream_speed = 8) : 
    (downstream_speed - upstream_speed) / 2 = 1.5 :=
by
  rw [h1, h2]
  simp
  norm_num

end NUMINAMATH_GPT_speed_of_stream_l915_91538


namespace NUMINAMATH_GPT_dave_earnings_l915_91570

def total_games : Nat := 10
def non_working_games : Nat := 2
def price_per_game : Nat := 4
def working_games : Nat := total_games - non_working_games
def money_earned : Nat := working_games * price_per_game

theorem dave_earnings : money_earned = 32 := by
  sorry

end NUMINAMATH_GPT_dave_earnings_l915_91570


namespace NUMINAMATH_GPT_probability_number_greater_than_3_from_0_5_l915_91579

noncomputable def probability_number_greater_than_3_in_0_5 : ℝ :=
  let total_interval_length := 5 - 0
  let event_interval_length := 5 - 3
  event_interval_length / total_interval_length

theorem probability_number_greater_than_3_from_0_5 :
  probability_number_greater_than_3_in_0_5 = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_probability_number_greater_than_3_from_0_5_l915_91579


namespace NUMINAMATH_GPT_necessary_condition_for_q_implies_m_bounds_necessary_but_not_sufficient_condition_for_not_q_l915_91584

-- Problem 1
theorem necessary_condition_for_q_implies_m_bounds (m : ℝ) :
  (∀ x : ℝ, x^2 - 8 * x - 20 ≤ 0 → 1 - m^2 ≤ x ∧ x ≤ 1 + m^2) → (- Real.sqrt 3 ≤ m ∧ m ≤ Real.sqrt 3) :=
sorry

-- Problem 2
theorem necessary_but_not_sufficient_condition_for_not_q (m : ℝ) :
  (∀ x : ℝ, ¬ (x^2 - 8 * x - 20 ≤ 0) → ¬ (1 - m^2 ≤ x ∧ x ≤ 1 + m^2)) → (m ≥ 3 ∨ m ≤ -3) :=
sorry

end NUMINAMATH_GPT_necessary_condition_for_q_implies_m_bounds_necessary_but_not_sufficient_condition_for_not_q_l915_91584


namespace NUMINAMATH_GPT_cost_per_meter_l915_91558

-- Defining the parameters and their relationships
def length : ℝ := 58
def breadth : ℝ := length - 16
def total_cost : ℝ := 5300
def perimeter : ℝ := 2 * (length + breadth)

-- Proving the cost per meter of fencing
theorem cost_per_meter : total_cost / perimeter = 26.50 := 
by
  sorry

end NUMINAMATH_GPT_cost_per_meter_l915_91558


namespace NUMINAMATH_GPT_smallest_possible_value_of_N_l915_91511

-- Conditions definition:
variable (a b c d : ℕ)
variable (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
variable (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
variable (gcd_ab : Int.gcd a b = 1)
variable (gcd_ac : Int.gcd a c = 2)
variable (gcd_ad : Int.gcd a d = 4)
variable (gcd_bc : Int.gcd b c = 5)
variable (gcd_bd : Int.gcd b d = 3)
variable (gcd_cd : Int.gcd c d = N)
variable (hN : N > 5)

-- Statement to prove:
theorem smallest_possible_value_of_N : N = 14 := sorry

end NUMINAMATH_GPT_smallest_possible_value_of_N_l915_91511


namespace NUMINAMATH_GPT_reciprocal_inequality_pos_reciprocal_inequality_neg_l915_91580

theorem reciprocal_inequality_pos {a b : ℝ} (h : a < b) (ha : 0 < a) : (1 / a) > (1 / b) :=
sorry

theorem reciprocal_inequality_neg {a b : ℝ} (h : a < b) (hb : b < 0) : (1 / a) < (1 / b) :=
sorry

end NUMINAMATH_GPT_reciprocal_inequality_pos_reciprocal_inequality_neg_l915_91580


namespace NUMINAMATH_GPT_smallest_inverse_defined_l915_91565

theorem smallest_inverse_defined (n : ℤ) : n = 5 :=
by sorry

end NUMINAMATH_GPT_smallest_inverse_defined_l915_91565


namespace NUMINAMATH_GPT_years_since_marriage_l915_91527

theorem years_since_marriage (x : ℕ) (ave_age_husband_wife_at_marriage : ℕ)
  (total_family_age_now : ℕ) (child_age : ℕ) (family_members : ℕ) :
  ave_age_husband_wife_at_marriage = 23 →
  total_family_age_now = 19 →
  child_age = 1 →
  family_members = 3 →
  (46 + 2 * x) + child_age = 57 →
  x = 5 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_years_since_marriage_l915_91527


namespace NUMINAMATH_GPT_problem_statement_l915_91504

theorem problem_statement : (2^2015 + 2^2011) / (2^2015 - 2^2011) = 17 / 15 :=
by
  -- Prove the equivalence as outlined above.
  sorry

end NUMINAMATH_GPT_problem_statement_l915_91504


namespace NUMINAMATH_GPT_find_cost_price_l915_91594

theorem find_cost_price (C : ℝ) (h1 : C * 1.05 = C + 0.05 * C)
  (h2 : 0.95 * C = C - 0.05 * C)
  (h3 : 1.05 * C - 4 = 1.045 * C) :
  C = 800 := sorry

end NUMINAMATH_GPT_find_cost_price_l915_91594


namespace NUMINAMATH_GPT_perfect_square_trinomial_l915_91523

theorem perfect_square_trinomial (m : ℝ) :
  (∃ (a : ℝ), (x^2 + mx + 1) = (x + a)^2) ↔ (m = 2 ∨ m = -2) := sorry

end NUMINAMATH_GPT_perfect_square_trinomial_l915_91523


namespace NUMINAMATH_GPT_intersection_of_M_and_complement_N_l915_91585

def M : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def N : Set ℝ := { x | 2 * x < 2 }
def complement_N : Set ℝ := { x | x ≥ 1 }

theorem intersection_of_M_and_complement_N : M ∩ complement_N = { x | 1 ≤ x ∧ x < 3 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_complement_N_l915_91585


namespace NUMINAMATH_GPT_Fran_speed_l915_91547

def Joann_speed : ℝ := 15
def Joann_time : ℝ := 4
def Fran_time : ℝ := 3.5

theorem Fran_speed :
  ∀ (s : ℝ), (s * Fran_time = Joann_speed * Joann_time) → (s = 120 / 7) :=
by
  intro s h
  sorry

end NUMINAMATH_GPT_Fran_speed_l915_91547


namespace NUMINAMATH_GPT_find_integer_x_l915_91503

theorem find_integer_x (x : ℤ) :
  (2 * x > 70 → x > 35 ∧ 4 * x > 25 → x > 6) ∧
  (x < 100 → x = 6) ∧ -- Given valid inequality relations and restrictions
  ((2 * x <= 70 ∨ x >= 100) ∨ (4 * x <= 25 ∨ x <= 5)) := -- Certain contradictions inherent

  by sorry

end NUMINAMATH_GPT_find_integer_x_l915_91503


namespace NUMINAMATH_GPT_no_equal_prob_for_same_color_socks_l915_91599

theorem no_equal_prob_for_same_color_socks :
  ∀ (n m : ℕ), n + m = 2009 → (n * (n - 1) + m * (m - 1) = (n + m) * (n + m - 1) / 2) → false :=
by
  intro n m h_total h_prob
  sorry

end NUMINAMATH_GPT_no_equal_prob_for_same_color_socks_l915_91599


namespace NUMINAMATH_GPT_range_of_a_l915_91573

noncomputable section

open Real

def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ∈ Set.Icc 0 3 then a - x else a * log x / log 2

theorem range_of_a (a : ℝ) (h : f a 2 < f a 4) : a > -2 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l915_91573


namespace NUMINAMATH_GPT_average_age_nine_students_l915_91531

theorem average_age_nine_students (total_age_15_students : ℕ)
                                (total_age_5_students : ℕ)
                                (age_15th_student : ℕ)
                                (h1 : total_age_15_students = 225)
                                (h2 : total_age_5_students = 65)
                                (h3 : age_15th_student = 16) :
                                (total_age_15_students - total_age_5_students - age_15th_student) / 9 = 16 := by
  sorry

end NUMINAMATH_GPT_average_age_nine_students_l915_91531


namespace NUMINAMATH_GPT_cost_price_is_1500_l915_91508

-- Definitions for the given conditions
def selling_price : ℝ := 1200
def loss_percentage : ℝ := 20

-- Define the cost price such that the loss percentage condition is satisfied
def cost_price (C : ℝ) : Prop :=
  loss_percentage = ((C - selling_price) / C) * 100

-- The proof problem to be solved: 
-- Prove that the cost price of the radio is Rs. 1500
theorem cost_price_is_1500 : ∃ C, cost_price C ∧ C = 1500 :=
by
  sorry

end NUMINAMATH_GPT_cost_price_is_1500_l915_91508


namespace NUMINAMATH_GPT_range_of_ab_l915_91528

noncomputable def f (x : ℝ) : ℝ := abs (2 - x^2)

theorem range_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : f a = f b) : 0 < a * b ∧ a * b < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_ab_l915_91528


namespace NUMINAMATH_GPT_solution_set_for_composed_function_l915_91518

theorem solution_set_for_composed_function :
  ∀ x : ℝ, (∀ y : ℝ, y = 2 * x - 1 → (2 * y - 1) ≥ 1) ↔ x ≥ 1 := by
  sorry

end NUMINAMATH_GPT_solution_set_for_composed_function_l915_91518


namespace NUMINAMATH_GPT_certain_number_l915_91561

theorem certain_number (x : ℝ) (h : 0.65 * 40 = (4/5) * x + 6) : x = 25 :=
sorry

end NUMINAMATH_GPT_certain_number_l915_91561


namespace NUMINAMATH_GPT_express_q_as_polynomial_l915_91545

def q (x : ℝ) : ℝ := x^3 + 4

theorem express_q_as_polynomial (x : ℝ) : 
  q x + (2 * x^6 + x^5 + 4 * x^4 + 6 * x^2) = (5 * x^4 + 10 * x^3 - x^2 + 8 * x + 15) → 
  q x = -2 * x^6 - x^5 + x^4 + 10 * x^3 - 7 * x^2 + 8 * x + 15 := by
  sorry

end NUMINAMATH_GPT_express_q_as_polynomial_l915_91545


namespace NUMINAMATH_GPT_x_zero_sufficient_not_necessary_for_sin_zero_l915_91591

theorem x_zero_sufficient_not_necessary_for_sin_zero :
  (∀ x : ℝ, x = 0 → Real.sin x = 0) ∧ (∃ y : ℝ, Real.sin y = 0 ∧ y ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_x_zero_sufficient_not_necessary_for_sin_zero_l915_91591


namespace NUMINAMATH_GPT_total_cost_of_shirt_and_sweater_l915_91589

-- Define the given conditions
def price_of_shirt := 36.46
def diff_price_shirt_sweater := 7.43
def price_of_sweater := price_of_shirt + diff_price_shirt_sweater

-- Statement to prove
theorem total_cost_of_shirt_and_sweater :
  price_of_shirt + price_of_sweater = 80.35 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_total_cost_of_shirt_and_sweater_l915_91589


namespace NUMINAMATH_GPT_pete_bus_ride_blocks_l915_91522

theorem pete_bus_ride_blocks : 
  ∀ (total_walk_blocks bus_blocks total_blocks : ℕ), 
  total_walk_blocks = 10 → 
  total_blocks = 50 → 
  total_walk_blocks + 2 * bus_blocks = total_blocks → 
  bus_blocks = 20 :=
by
  intros total_walk_blocks bus_blocks total_blocks h1 h2 h3
  sorry

end NUMINAMATH_GPT_pete_bus_ride_blocks_l915_91522


namespace NUMINAMATH_GPT_work_completion_time_l915_91575

noncomputable def work_rate_A : ℚ := 1 / 12
noncomputable def work_rate_B : ℚ := 1 / 14

theorem work_completion_time : 
  (work_rate_A + work_rate_B)⁻¹ = 84 / 13 := by
  sorry

end NUMINAMATH_GPT_work_completion_time_l915_91575


namespace NUMINAMATH_GPT_find_a_b_l915_91507

theorem find_a_b (a b : ℝ) (h : (a - 2) ^ 2 + |b + 4| = 0) : a + b = -2 :=
sorry

end NUMINAMATH_GPT_find_a_b_l915_91507


namespace NUMINAMATH_GPT_time_to_pass_platform_is_correct_l915_91577

noncomputable def train_length : ℝ := 250 -- meters
noncomputable def time_to_pass_pole : ℝ := 10 -- seconds
noncomputable def time_to_pass_platform : ℝ := 60 -- seconds

-- Speed of the train
noncomputable def train_speed := train_length / time_to_pass_pole -- meters/second

-- Length of the platform
noncomputable def platform_length := train_speed * time_to_pass_platform - train_length -- meters

-- Proving the time to pass the platform is 50 seconds
theorem time_to_pass_platform_is_correct : 
  (platform_length / train_speed) = 50 :=
by
  sorry

end NUMINAMATH_GPT_time_to_pass_platform_is_correct_l915_91577


namespace NUMINAMATH_GPT_find_m_values_l915_91515

-- Defining the sets and conditions
def A : Set ℝ := { x | x ^ 2 - 9 * x - 10 = 0 }
def B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }

-- Stating the proof problem
theorem find_m_values : {m | A ∪ B m = A} = {0, 1, -1 / 10} :=
by
  sorry

end NUMINAMATH_GPT_find_m_values_l915_91515


namespace NUMINAMATH_GPT_circles_tangent_l915_91533

theorem circles_tangent (m : ℝ) :
  (∀ (x y : ℝ), (x - m)^2 + (y + 2)^2 = 9 → 
                (x + 1)^2 + (y - m)^2 = 4 →
                ∃ m, m = -1 ∨ m = -2) := 
sorry

end NUMINAMATH_GPT_circles_tangent_l915_91533


namespace NUMINAMATH_GPT_solution_value_a_l915_91569

theorem solution_value_a (x a : ℝ) (h₁ : x = 2) (h₂ : 2 * x + a = 3) : a = -1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_solution_value_a_l915_91569
