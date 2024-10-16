import Mathlib

namespace NUMINAMATH_CALUDE_range_of_m_l3318_331866

def A : Set ℝ := {x | |x - 1| < 2}
def B (m : ℝ) : Set ℝ := {x | -1 < x ∧ x < m + 1}

theorem range_of_m :
  (∀ m : ℝ, (∀ x : ℝ, x ∈ B m → x ∈ A) ∧ (∃ x : ℝ, x ∈ A ∧ x ∉ B m)) ↔ 
  (∀ m : ℝ, m > 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l3318_331866


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_fourteen_l3318_331835

theorem sum_of_roots_equals_fourteen : 
  let f : ℝ → ℝ := λ x => (x - 7)^2 - 16
  ∃ x₁ x₂ : ℝ, (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) ∧ x₁ + x₂ = 14 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_fourteen_l3318_331835


namespace NUMINAMATH_CALUDE_g_of_3_equals_79_l3318_331800

/-- Given a function g(x) = 5x^3 - 7x^2 + 3x - 2, prove that g(3) = 79 -/
theorem g_of_3_equals_79 :
  let g : ℝ → ℝ := λ x ↦ 5 * x^3 - 7 * x^2 + 3 * x - 2
  g 3 = 79 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_equals_79_l3318_331800


namespace NUMINAMATH_CALUDE_hash_difference_l3318_331864

def hash (x y : ℤ) : ℤ := 2 * x * y - 3 * x + y

theorem hash_difference : (hash 6 4) - (hash 4 6) = -8 := by
  sorry

end NUMINAMATH_CALUDE_hash_difference_l3318_331864


namespace NUMINAMATH_CALUDE_symmetric_line_theorem_l3318_331802

/-- The equation of a line symmetric to another line with respect to a vertical line. -/
def symmetric_line_equation (a b c : ℝ) (k : ℝ) : ℝ → ℝ → Prop :=
  fun x y => 2 * (k - a) * x + b * y + (c - 2 * k * (k - a)) = 0

/-- The original line equation. -/
def original_line (x y : ℝ) : Prop := x - 2 * y + 1 = 0

/-- The line of symmetry. -/
def symmetry_line (x : ℝ) : Prop := x = 3

theorem symmetric_line_theorem :
  symmetric_line_equation 1 (-2) 1 3 = fun x y => 2 * x + y - 8 = 0 :=
sorry

end NUMINAMATH_CALUDE_symmetric_line_theorem_l3318_331802


namespace NUMINAMATH_CALUDE_one_distinct_real_root_l3318_331841

theorem one_distinct_real_root :
  ∃! x : ℝ, x ≠ 0 ∧ (abs x - 4 / x = 3 * abs x / x) :=
by sorry

end NUMINAMATH_CALUDE_one_distinct_real_root_l3318_331841


namespace NUMINAMATH_CALUDE_boxes_with_pans_is_eight_l3318_331846

/-- Represents the arrangement of teacups and boxes. -/
structure TeacupArrangement where
  total_boxes : Nat
  cups_per_box : Nat
  cups_broken_per_box : Nat
  cups_left : Nat

/-- Calculates the number of boxes containing pans. -/
def boxes_with_pans (arrangement : TeacupArrangement) : Nat :=
  let teacup_boxes := arrangement.cups_left / (arrangement.cups_per_box - arrangement.cups_broken_per_box)
  let remaining_boxes := arrangement.total_boxes - teacup_boxes
  remaining_boxes / 2

/-- Theorem stating that the number of boxes with pans is 8. -/
theorem boxes_with_pans_is_eight : 
  boxes_with_pans { total_boxes := 26
                  , cups_per_box := 20
                  , cups_broken_per_box := 2
                  , cups_left := 180 } = 8 := by
  sorry


end NUMINAMATH_CALUDE_boxes_with_pans_is_eight_l3318_331846


namespace NUMINAMATH_CALUDE_sample_size_is_140_l3318_331858

/-- Represents a school with students and a height measurement study -/
structure School where
  total_students : ℕ
  measured_students : ℕ
  measured_students_le_total : measured_students ≤ total_students

/-- The sample size of a height measurement study in a school -/
def sample_size (s : School) : ℕ := s.measured_students

/-- Theorem stating that for a school with 1740 students and 140 measured students, the sample size is 140 -/
theorem sample_size_is_140 (s : School) 
  (h1 : s.total_students = 1740) 
  (h2 : s.measured_students = 140) : 
  sample_size s = 140 := by sorry

end NUMINAMATH_CALUDE_sample_size_is_140_l3318_331858


namespace NUMINAMATH_CALUDE_isabela_spent_2800_l3318_331816

/-- The total amount Isabela spent on cucumbers and pencils -/
def total_spent (cucumber_price : ℝ) (pencil_price : ℝ) (cucumber_count : ℕ) 
  (pencil_discount : ℝ) : ℝ :=
  let pencil_count := cucumber_count / 2
  let pencil_cost := pencil_count * pencil_price * (1 - pencil_discount)
  let cucumber_cost := cucumber_count * cucumber_price
  pencil_cost + cucumber_cost

/-- Theorem stating that Isabela spent $2800 on cucumbers and pencils -/
theorem isabela_spent_2800 : 
  total_spent 20 20 100 0.2 = 2800 := by
  sorry

end NUMINAMATH_CALUDE_isabela_spent_2800_l3318_331816


namespace NUMINAMATH_CALUDE_residue_problem_l3318_331892

theorem residue_problem : Int.mod (Int.mod (-1043) 36) 10 = 1 := by sorry

end NUMINAMATH_CALUDE_residue_problem_l3318_331892


namespace NUMINAMATH_CALUDE_matt_bike_ride_distance_l3318_331832

theorem matt_bike_ride_distance 
  (distance_to_first_sign : ℕ)
  (distance_between_signs : ℕ)
  (distance_after_second_sign : ℕ)
  (h1 : distance_to_first_sign = 350)
  (h2 : distance_between_signs = 375)
  (h3 : distance_after_second_sign = 275) :
  distance_to_first_sign + distance_between_signs + distance_after_second_sign = 1000 :=
by sorry

end NUMINAMATH_CALUDE_matt_bike_ride_distance_l3318_331832


namespace NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l3318_331830

/-- A curve represented by the equation mx^2 + ny^2 = 1 is an ellipse -/
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

theorem mn_positive_necessary_not_sufficient :
  (∀ m n : ℝ, is_ellipse m n → m * n > 0) ∧
  (∃ m n : ℝ, m * n > 0 ∧ ¬is_ellipse m n) :=
sorry

end NUMINAMATH_CALUDE_mn_positive_necessary_not_sufficient_l3318_331830


namespace NUMINAMATH_CALUDE_container_volume_ratio_l3318_331828

theorem container_volume_ratio :
  ∀ (volume_container1 volume_container2 : ℚ),
  volume_container1 > 0 →
  volume_container2 > 0 →
  (2 : ℚ) / 3 * volume_container1 = (1 : ℚ) / 2 * volume_container2 →
  volume_container1 / volume_container2 = (3 : ℚ) / 4 := by
sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l3318_331828


namespace NUMINAMATH_CALUDE_integral_equation_solution_l3318_331873

theorem integral_equation_solution (k : ℝ) : 
  (∫ x in (0:ℝ)..1, x - k) = (3/2 : ℝ) → k = -1 := by
  sorry

end NUMINAMATH_CALUDE_integral_equation_solution_l3318_331873


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l3318_331872

-- Define the function
def f (x : ℝ) : ℝ := 2 * |x - 3| + 5

-- State the theorem
theorem minimum_point_of_translated_graph :
  ∃! (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = 5 ∧ x = 3 :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l3318_331872


namespace NUMINAMATH_CALUDE_largest_integer_in_interval_l3318_331818

theorem largest_integer_in_interval : 
  ∃ (x : ℤ), x = 4 ∧ (2 : ℚ) / 7 < (x : ℚ) / 6 ∧ (x : ℚ) / 6 < 7 / 9 ∧
  ∀ (y : ℤ), y > x → ¬((2 : ℚ) / 7 < (y : ℚ) / 6 ∧ (y : ℚ) / 6 < 7 / 9) :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_in_interval_l3318_331818


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l3318_331848

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n |>.reverse

def binary_1101101 : List Bool := [true, true, false, true, true, false, true]
def binary_1101 : List Bool := [true, true, false, true]
def binary_result : List Bool := [true, false, false, true, false, true, false, true, false, false, false, true]

theorem binary_multiplication_theorem :
  nat_to_binary ((binary_to_nat binary_1101101) * (binary_to_nat binary_1101)) = binary_result := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l3318_331848


namespace NUMINAMATH_CALUDE_stratified_sample_theorem_l3318_331807

/-- Calculates the number of female students in a stratified sample -/
def stratified_sample_females (total_students : ℕ) (female_students : ℕ) (sample_size : ℕ) : ℕ :=
  (female_students * sample_size) / total_students

/-- Theorem: In a class of 54 students with 18 females, a stratified sample of 9 students contains 3 females -/
theorem stratified_sample_theorem :
  stratified_sample_females 54 18 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_theorem_l3318_331807


namespace NUMINAMATH_CALUDE_edward_book_expense_l3318_331862

/-- The amount Edward spent on books -/
def amount_spent (num_books : ℕ) (cost_per_book : ℝ) : ℝ :=
  num_books * cost_per_book

/-- Theorem stating that Edward spent $6 on books -/
theorem edward_book_expense : amount_spent 2 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_edward_book_expense_l3318_331862


namespace NUMINAMATH_CALUDE_no_single_intersection_point_l3318_331813

theorem no_single_intersection_point :
  ¬ ∃ (b : ℝ), b ≠ 0 ∧
    (∃! (x : ℝ), bx^2 + 2*x - 3 = 2*x + 5) :=
sorry

end NUMINAMATH_CALUDE_no_single_intersection_point_l3318_331813


namespace NUMINAMATH_CALUDE_min_abs_alpha_plus_gamma_l3318_331898

theorem min_abs_alpha_plus_gamma :
  ∀ (α γ : ℂ),
  let g := λ (z : ℂ) => (3 + I) * z^2 + α * z + γ
  (g 1).im = 0 →
  (g I).im = 0 →
  ∃ (α₀ γ₀ : ℂ),
    (let g₀ := λ (z : ℂ) => (3 + I) * z^2 + α₀ * z + γ₀
     (g₀ 1).im = 0 ∧
     (g₀ I).im = 0 ∧
     Complex.abs α₀ + Complex.abs γ₀ = Real.sqrt 2 ∧
     ∀ (α' γ' : ℂ),
       (let g' := λ (z : ℂ) => (3 + I) * z^2 + α' * z + γ'
        (g' 1).im = 0 ∧
        (g' I).im = 0 →
        Complex.abs α' + Complex.abs γ' ≥ Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_min_abs_alpha_plus_gamma_l3318_331898


namespace NUMINAMATH_CALUDE_N_is_composite_l3318_331833

/-- N is defined as 7 × 9 × 13 + 2020 × 2018 × 2014 -/
def N : ℕ := 7 * 9 * 13 + 2020 * 2018 * 2014

/-- Theorem stating that N is composite -/
theorem N_is_composite : ¬ Nat.Prime N := by sorry

end NUMINAMATH_CALUDE_N_is_composite_l3318_331833


namespace NUMINAMATH_CALUDE_rectangular_sheet_fold_ratio_l3318_331859

theorem rectangular_sheet_fold_ratio :
  ∀ (l s : ℝ), l > s ∧ s > 0 →
  l^2 = s^2 + (s^2 / l)^2 →
  l / s = Real.sqrt (2 / (Real.sqrt 5 - 1)) := by
sorry

end NUMINAMATH_CALUDE_rectangular_sheet_fold_ratio_l3318_331859


namespace NUMINAMATH_CALUDE_factorization_sum_l3318_331852

theorem factorization_sum (a b : ℤ) :
  (∀ x : ℝ, 25 * x^2 - 125 * x - 100 = (5 * x + a) * (5 * x + b)) →
  a + b = -25 :=
by sorry

end NUMINAMATH_CALUDE_factorization_sum_l3318_331852


namespace NUMINAMATH_CALUDE_vector_operation_l3318_331863

/-- Given vectors a = (-4, 3) and b = (5, 6), prove that 3|a|^2 - 4(a · b) = 83 -/
theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (-4, 3)) (h2 : b = (5, 6)) :
  3 * ‖a‖^2 - 4 * (a.1 * b.1 + a.2 * b.2) = 83 := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_l3318_331863


namespace NUMINAMATH_CALUDE_height_difference_l3318_331871

-- Define the heights in inches
def dog_height : ℕ := 24
def carter_height : ℕ := 2 * dog_height
def betty_height : ℕ := 3 * 12  -- 3 feet converted to inches

-- Theorem statement
theorem height_difference : carter_height - betty_height = 12 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l3318_331871


namespace NUMINAMATH_CALUDE_black_balls_count_l3318_331831

theorem black_balls_count (red_balls : ℕ) (prob_red : ℚ) (black_balls : ℕ) : 
  red_balls = 3 → prob_red = 1/4 → black_balls = 9 → 
  (red_balls : ℚ) / (red_balls + black_balls : ℚ) = prob_red :=
by sorry

end NUMINAMATH_CALUDE_black_balls_count_l3318_331831


namespace NUMINAMATH_CALUDE_football_team_throwers_l3318_331806

/-- Represents the number of throwers on a football team -/
def num_throwers (total_players right_handed_players : ℕ) : ℕ :=
  total_players - (3 * right_handed_players - 2 * total_players)

theorem football_team_throwers :
  let total_players : ℕ := 70
  let right_handed_players : ℕ := 57
  num_throwers total_players right_handed_players = 31 :=
by sorry

end NUMINAMATH_CALUDE_football_team_throwers_l3318_331806


namespace NUMINAMATH_CALUDE_inches_in_foot_l3318_331808

theorem inches_in_foot (room_side : ℝ) (room_area_sq_inches : ℝ) :
  room_side = 10 →
  room_area_sq_inches = 14400 →
  ∃ (inches_per_foot : ℝ), inches_per_foot = 12 ∧ 
    (room_side * inches_per_foot)^2 = room_area_sq_inches :=
by sorry

end NUMINAMATH_CALUDE_inches_in_foot_l3318_331808


namespace NUMINAMATH_CALUDE_laundry_cost_is_11_l3318_331837

/-- The cost of Samantha's laundry given the specified conditions. -/
def laundry_cost : ℚ :=
  let washer_cost : ℚ := 4
  let dryer_cost_per_10_min : ℚ := 1/4
  let wash_loads : ℕ := 2
  let dryer_count : ℕ := 3
  let dryer_time : ℕ := 40

  let wash_cost : ℚ := washer_cost * wash_loads
  let dryer_intervals : ℕ := dryer_time / 10
  let dryer_cost : ℚ := (dryer_cost_per_10_min * dryer_intervals) * dryer_count

  wash_cost + dryer_cost

/-- Theorem stating that the total cost of Samantha's laundry is $11. -/
theorem laundry_cost_is_11 : laundry_cost = 11 := by
  sorry

end NUMINAMATH_CALUDE_laundry_cost_is_11_l3318_331837


namespace NUMINAMATH_CALUDE_integral_of_even_function_l3318_331811

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The main theorem -/
theorem integral_of_even_function (a : ℝ) :
  let f := fun x => a * x^2 + (a - 2) * x + a^2
  IsEven f →
  ∫ x in (-a)..a, (x^2 + x + Real.sqrt (4 - x^2)) = 16/3 + 2 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_integral_of_even_function_l3318_331811


namespace NUMINAMATH_CALUDE_triangle_theorem_l3318_331822

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_theorem (t : Triangle) 
  (h1 : 2 * t.b * Real.cos t.A = t.c * Real.cos t.A + t.a * Real.cos t.C)
  (h2 : t.b + t.c = Real.sqrt 2 * t.a)
  (h3 : t.a * t.b * Real.sin t.A / 2 = Real.sqrt 3 / 12) : 
  t.A = Real.pi / 3 ∧ t.a = 1 := by
  sorry


end NUMINAMATH_CALUDE_triangle_theorem_l3318_331822


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3318_331842

theorem complex_magnitude_problem (z : ℂ) : z = Complex.I * (Complex.I - 1) → Complex.abs (z - 1) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3318_331842


namespace NUMINAMATH_CALUDE_expression_value_l3318_331861

theorem expression_value (a b c d m : ℚ) 
  (h1 : a = -b) 
  (h2 : c * d = 1) 
  (h3 : |m| = 4) : 
  2*a - 5*c*d - m + 2*b = -9 ∨ 2*a - 5*c*d - m + 2*b = -1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3318_331861


namespace NUMINAMATH_CALUDE_chromium_percentage_in_mixed_alloy_l3318_331869

/-- Given two alloys with different chromium percentages and weights, 
    calculates the chromium percentage in the resulting alloy when mixed. -/
theorem chromium_percentage_in_mixed_alloy 
  (chromium_percent1 chromium_percent2 : ℝ)
  (weight1 weight2 : ℝ)
  (h1 : chromium_percent1 = 15)
  (h2 : chromium_percent2 = 8)
  (h3 : weight1 = 15)
  (h4 : weight2 = 35) :
  let total_chromium := (chromium_percent1 / 100 * weight1) + (chromium_percent2 / 100 * weight2)
  let total_weight := weight1 + weight2
  (total_chromium / total_weight) * 100 = 10.1 := by
sorry

end NUMINAMATH_CALUDE_chromium_percentage_in_mixed_alloy_l3318_331869


namespace NUMINAMATH_CALUDE_problem_solution_l3318_331834

theorem problem_solution (α β : Real) 
  (h1 : α ∈ Set.Ioo 0 (Real.pi / 2))
  (h2 : Real.cos (2 * α) = 4 / 5)
  (h3 : β ∈ Set.Ioo (Real.pi / 2) Real.pi)
  (h4 : 5 * Real.sin (2 * α + β) = Real.sin β) : 
  (Real.sin α + Real.cos α = 2 * Real.sqrt 10 / 5) ∧ 
  (β = 3 * Real.pi / 4) := by
sorry


end NUMINAMATH_CALUDE_problem_solution_l3318_331834


namespace NUMINAMATH_CALUDE_representative_selection_cases_l3318_331887

def number_of_female_students : ℕ := 4
def number_of_male_students : ℕ := 6

theorem representative_selection_cases :
  (number_of_female_students * number_of_male_students) = 24 :=
by sorry

end NUMINAMATH_CALUDE_representative_selection_cases_l3318_331887


namespace NUMINAMATH_CALUDE_multiplication_mistake_problem_l3318_331827

theorem multiplication_mistake_problem :
  ∃ x : ℝ, (493 * x - 394 * x = 78426) ∧ (x = 792) := by
  sorry

end NUMINAMATH_CALUDE_multiplication_mistake_problem_l3318_331827


namespace NUMINAMATH_CALUDE_carpet_width_calculation_l3318_331825

theorem carpet_width_calculation (room_length room_width carpet_cost_per_sqm total_cost : ℝ) 
  (h1 : room_length = 13)
  (h2 : room_width = 9)
  (h3 : carpet_cost_per_sqm = 12)
  (h4 : total_cost = 1872) : 
  (total_cost / carpet_cost_per_sqm / room_length) * 100 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_carpet_width_calculation_l3318_331825


namespace NUMINAMATH_CALUDE_max_product_with_digits_1_to_5_l3318_331899

def digit := Fin 5

def valid_number (n : ℕ) : Prop :=
  ∃ (d₁ d₂ d₃ : digit), n = d₁.val + 1 + 10 * (d₂.val + 1) + 100 * (d₃.val + 1)

def valid_product (p : ℕ) : Prop :=
  ∃ (n₁ n₂ : ℕ), valid_number n₁ ∧ valid_number n₂ ∧ p = n₁ * n₂

theorem max_product_with_digits_1_to_5 :
  ∀ p, valid_product p → p ≤ 22412 :=
sorry

end NUMINAMATH_CALUDE_max_product_with_digits_1_to_5_l3318_331899


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l3318_331821

theorem floor_ceiling_sum_seven (x : ℝ) : 
  (⌊x⌋ : ℝ) + (⌈x⌉ : ℝ) = 7 ↔ 3 < x ∧ x < 4 :=
by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_seven_l3318_331821


namespace NUMINAMATH_CALUDE_cos_330_deg_l3318_331819

/-- Cosine of 330 degrees is equal to sqrt(3)/2 -/
theorem cos_330_deg : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_330_deg_l3318_331819


namespace NUMINAMATH_CALUDE_calculation_proof_l3318_331868

theorem calculation_proof : 
  (3/5 : ℚ) * 200 + (456/1000 : ℚ) * 875 + (7/8 : ℚ) * 320 - 
  ((5575/10000 : ℚ) * 1280 + (1/3 : ℚ) * 960) = -2349/10 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3318_331868


namespace NUMINAMATH_CALUDE_kaleb_ferris_wheel_spend_l3318_331865

/-- Calculates the money spent on a ferris wheel ride given initial tickets, remaining tickets, and cost per ticket. -/
def money_spent_on_ride (initial_tickets remaining_tickets cost_per_ticket : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) * cost_per_ticket

/-- Proves that Kaleb spent 27 dollars on the ferris wheel ride. -/
theorem kaleb_ferris_wheel_spend :
  let initial_tickets : ℕ := 6
  let remaining_tickets : ℕ := 3
  let cost_per_ticket : ℕ := 9
  money_spent_on_ride initial_tickets remaining_tickets cost_per_ticket = 27 := by
sorry

end NUMINAMATH_CALUDE_kaleb_ferris_wheel_spend_l3318_331865


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3318_331890

/-- The eccentricity of the hyperbola 3x^2 - y^2 = 3 is 2 -/
theorem hyperbola_eccentricity : ∃ (e : ℝ), e = 2 ∧ 
  ∀ (x y : ℝ), 3 * x^2 - y^2 = 3 → 
  e = (Real.sqrt ((3 * x^2 + y^2) / 3)) / (Real.sqrt (3 * x^2 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3318_331890


namespace NUMINAMATH_CALUDE_deposit_calculation_l3318_331810

/-- Calculates the deposit amount given an initial amount -/
def calculateDeposit (initialAmount : ℚ) : ℚ :=
  initialAmount * (30 / 100) * (25 / 100) * (20 / 100)

/-- Proves that the deposit calculation for Rs. 50,000 results in Rs. 750 -/
theorem deposit_calculation :
  calculateDeposit 50000 = 750 := by
  sorry

end NUMINAMATH_CALUDE_deposit_calculation_l3318_331810


namespace NUMINAMATH_CALUDE_matching_socks_probability_l3318_331845

def gray_socks : ℕ := 10
def white_socks : ℕ := 8
def blue_socks : ℕ := 6

def total_socks : ℕ := gray_socks + white_socks + blue_socks

theorem matching_socks_probability :
  (Nat.choose gray_socks 2 + Nat.choose white_socks 2 + Nat.choose blue_socks 2) /
  Nat.choose total_socks 2 = 22 / 69 :=
by sorry

end NUMINAMATH_CALUDE_matching_socks_probability_l3318_331845


namespace NUMINAMATH_CALUDE_solution_pairs_l3318_331886

theorem solution_pairs : 
  ∃ (S : Set (ℕ × ℕ)), 
    S = {(0, 0), (1, 0)} ∧ 
    ∀ (a b : ℕ) (x : ℝ), 
      (a, b) ∈ S ↔ 
        (-2 * (a : ℝ) + (b : ℝ)^2 = Real.cos (π * (a : ℝ) + (b : ℝ)^2) - 1 ∧
         (b : ℝ)^2 = Real.cos (2 * π * (a : ℝ) + (b : ℝ)^2) - 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l3318_331886


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_square_plus_one_less_than_zero_l3318_331885

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_square_plus_one_less_than_zero :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_square_plus_one_less_than_zero_l3318_331885


namespace NUMINAMATH_CALUDE_fourth_jeweler_bags_l3318_331844

def bags : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def total_gold : ℕ := bags.sum

theorem fourth_jeweler_bags (lost_bag : ℕ) 
  (h1 : lost_bag ∈ bags)
  (h2 : lost_bag ≠ 1 ∧ lost_bag ≠ 3 ∧ lost_bag ≠ 11)
  (h3 : (total_gold - lost_bag) % 4 = 0)
  (h4 : (bags.length - 1) % 4 = 0) :
  ∃ (jeweler1 jeweler2 jeweler3 jeweler4 : List ℕ),
    jeweler1.sum = jeweler2.sum ∧
    jeweler2.sum = jeweler3.sum ∧
    jeweler3.sum = jeweler4.sum ∧
    jeweler1.length = jeweler2.length ∧
    jeweler2.length = jeweler3.length ∧
    jeweler3.length = jeweler4.length ∧
    1 ∈ jeweler1 ∧
    3 ∈ jeweler2 ∧
    11 ∈ jeweler3 ∧
    jeweler4 = [2, 9, 10] :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_jeweler_bags_l3318_331844


namespace NUMINAMATH_CALUDE_greatest_m_value_l3318_331874

theorem greatest_m_value (p m : ℕ) (hp : Nat.Prime p) 
  (heq : p * (p + m) + 2 * p = (m + 2)^3) : 
  m ≤ 28 ∧ ∃ (p' m' : ℕ), Nat.Prime p' ∧ p' * (p' + 28) + 2 * p' = (28 + 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_greatest_m_value_l3318_331874


namespace NUMINAMATH_CALUDE_symmetric_point_l3318_331809

/-- The line of symmetry -/
def line_of_symmetry (x y : ℝ) : Prop := x - y - 1 = 0

/-- Check if two points are symmetric with respect to a line -/
def is_symmetric (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  -- The line connecting the points is perpendicular to the line of symmetry
  (y₂ - y₁) / (x₂ - x₁) = -1 ∧
  -- The midpoint of the two points lies on the line of symmetry
  line_of_symmetry ((x₁ + x₂) / 2) ((y₁ + y₂) / 2)

/-- Theorem: The point (5, -4) is symmetric to (-3, 4) with respect to the line x-y-1=0 -/
theorem symmetric_point : is_symmetric (-3) 4 5 (-4) :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_l3318_331809


namespace NUMINAMATH_CALUDE_no_natural_solution_l3318_331893

theorem no_natural_solution :
  ¬∃ (n m : ℕ), (n + 1) * (2 * n + 1) = 18 * m^2 := by
sorry

end NUMINAMATH_CALUDE_no_natural_solution_l3318_331893


namespace NUMINAMATH_CALUDE_snacks_ryan_can_buy_l3318_331820

def one_way_travel_time : ℝ := 2
def snack_cost (round_trip_time : ℝ) : ℝ := 10 * round_trip_time
def ryan_budget : ℝ := 2000

theorem snacks_ryan_can_buy :
  (ryan_budget / snack_cost (2 * one_way_travel_time)) = 50 := by
  sorry

end NUMINAMATH_CALUDE_snacks_ryan_can_buy_l3318_331820


namespace NUMINAMATH_CALUDE_cookie_problem_l3318_331855

/-- Calculates the number of chocolate chips per cookie -/
def chocolate_chips_per_cookie (cookies_per_batch : ℕ) (family_members : ℕ) (batches : ℕ) (chips_per_person : ℕ) : ℕ :=
  let total_cookies := cookies_per_batch * batches
  let cookies_per_person := total_cookies / family_members
  chips_per_person / cookies_per_person

/-- Proves that the number of chocolate chips per cookie is 2 under given conditions -/
theorem cookie_problem : 
  chocolate_chips_per_cookie 12 4 3 18 = 2 := by
  sorry

end NUMINAMATH_CALUDE_cookie_problem_l3318_331855


namespace NUMINAMATH_CALUDE_bears_per_shelf_l3318_331839

theorem bears_per_shelf (initial_stock : ℕ) (new_shipment : ℕ) (num_shelves : ℕ) 
  (h1 : initial_stock = 6)
  (h2 : new_shipment = 18)
  (h3 : num_shelves = 4) :
  (initial_stock + new_shipment) / num_shelves = 6 :=
by sorry

end NUMINAMATH_CALUDE_bears_per_shelf_l3318_331839


namespace NUMINAMATH_CALUDE_joy_tonight_outcomes_l3318_331884

/-- The number of letters in mailbox A -/
def mailbox_A : Nat := 30

/-- The number of letters in mailbox B -/
def mailbox_B : Nat := 20

/-- The total number of different outcomes for selecting a lucky star and two lucky partners -/
def total_outcomes : Nat := mailbox_A * (mailbox_A - 1) * mailbox_B + mailbox_B * (mailbox_B - 1) * mailbox_A

theorem joy_tonight_outcomes : total_outcomes = 28800 := by
  sorry

end NUMINAMATH_CALUDE_joy_tonight_outcomes_l3318_331884


namespace NUMINAMATH_CALUDE_sideline_time_l3318_331801

def game_duration : ℕ := 90
def first_play_time : ℕ := 20
def second_play_time : ℕ := 35

theorem sideline_time :
  game_duration - (first_play_time + second_play_time) = 35 := by
  sorry

end NUMINAMATH_CALUDE_sideline_time_l3318_331801


namespace NUMINAMATH_CALUDE_highest_a_divisible_by_8_l3318_331888

def is_divisible_by_8 (n : ℕ) : Prop := n % 8 = 0

def last_three_digits (n : ℕ) : ℕ := n % 1000

theorem highest_a_divisible_by_8 :
  ∀ a : ℕ, a ≤ 9 →
    (is_divisible_by_8 (365 * 100 + a * 10 + 16) ↔ a ≤ 8) ∧
    (∀ b : ℕ, b > 8 ∧ b ≤ 9 → ¬ is_divisible_by_8 (365 * 100 + b * 10 + 16)) :=
sorry

end NUMINAMATH_CALUDE_highest_a_divisible_by_8_l3318_331888


namespace NUMINAMATH_CALUDE_z_min_max_in_D_l3318_331849

-- Define the function z
def z (x y : ℝ) : ℝ := 4 * x^2 + y^2 - 16 * x - 4 * y + 20

-- Define the region D
def D : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.1 - 2 * p.2 ≤ 0 ∧ p.1 + p.2 - 6 ≤ 0}

-- Theorem statement
theorem z_min_max_in_D :
  (∃ p ∈ D, ∀ q ∈ D, z p.1 p.2 ≤ z q.1 q.2) ∧
  (∃ p ∈ D, ∀ q ∈ D, z p.1 p.2 ≥ z q.1 q.2) ∧
  (∃ p ∈ D, z p.1 p.2 = 0) ∧
  (∃ p ∈ D, z p.1 p.2 = 32) :=
sorry

end NUMINAMATH_CALUDE_z_min_max_in_D_l3318_331849


namespace NUMINAMATH_CALUDE_vector_operation_proof_l3318_331857

def vector_operation : Prop :=
  let v1 : Fin 2 → ℝ := ![5, -6]
  let v2 : Fin 2 → ℝ := ![-2, 13]
  let v3 : Fin 2 → ℝ := ![1, -2]
  v1 + v2 - 3 • v3 = ![0, 13]

theorem vector_operation_proof : vector_operation := by
  sorry

end NUMINAMATH_CALUDE_vector_operation_proof_l3318_331857


namespace NUMINAMATH_CALUDE_difference_of_squares_l3318_331804

theorem difference_of_squares (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3318_331804


namespace NUMINAMATH_CALUDE_general_term_formula_l3318_331876

def S (n : ℕ) : ℤ := 1 - 2^n

def a (n : ℕ) : ℤ := -2^(n-1)

theorem general_term_formula (n : ℕ) (h : n ≥ 2) : 
  a n = S n - S (n-1) := by
  sorry

end NUMINAMATH_CALUDE_general_term_formula_l3318_331876


namespace NUMINAMATH_CALUDE_purchasing_ways_l3318_331881

/-- The number of different oreo flavors --/
def oreo_flavors : ℕ := 7

/-- The number of different milk flavors --/
def milk_flavors : ℕ := 4

/-- The total number of products they purchase --/
def total_products : ℕ := 4

/-- Charlie's purchasing strategy: no repeats, can buy both oreos and milk --/
def charlie_strategy (k : ℕ) : ℕ := Nat.choose (oreo_flavors + milk_flavors) k

/-- Delta's purchasing strategy: only oreos, can have repeats --/
def delta_strategy (k : ℕ) : ℕ :=
  if k = 0 then 1
  else if k = 1 then oreo_flavors
  else if k = 2 then Nat.choose oreo_flavors 2 + oreo_flavors
  else if k = 3 then Nat.choose oreo_flavors 3 + oreo_flavors * (oreo_flavors - 1) + oreo_flavors
  else Nat.choose oreo_flavors 4 + oreo_flavors * (oreo_flavors - 1) + 
       (oreo_flavors * (oreo_flavors - 1)) / 2 + oreo_flavors

/-- The total number of ways to purchase 4 products --/
def total_ways : ℕ := 
  (charlie_strategy 4 * delta_strategy 0) +
  (charlie_strategy 3 * delta_strategy 1) +
  (charlie_strategy 2 * delta_strategy 2) +
  (charlie_strategy 1 * delta_strategy 3) +
  (charlie_strategy 0 * delta_strategy 4)

theorem purchasing_ways : total_ways = 4054 := by
  sorry

end NUMINAMATH_CALUDE_purchasing_ways_l3318_331881


namespace NUMINAMATH_CALUDE_four_last_digit_fib_mod8_l3318_331870

/-- Fibonacci sequence modulo 8 -/
def fib_mod8 : ℕ → ℕ
| 0 => 0
| 1 => 1
| n + 2 => (fib_mod8 n + fib_mod8 (n + 1)) % 8

/-- Set of digits that have appeared in the Fibonacci sequence modulo 8 up to n -/
def digits_appeared (n : ℕ) : Finset ℕ :=
  Finset.range (n + 1).succ
    |>.filter (fun i => fib_mod8 i ∈ Finset.range 8)
    |>.image fib_mod8

/-- The proposition that 4 is the last digit to appear in the Fibonacci sequence modulo 8 -/
theorem four_last_digit_fib_mod8 :
  ∃ n : ℕ, 4 ∈ digits_appeared n ∧ digits_appeared n = Finset.range 8 :=
sorry

end NUMINAMATH_CALUDE_four_last_digit_fib_mod8_l3318_331870


namespace NUMINAMATH_CALUDE_arithmetic_mean_1_5_l3318_331882

theorem arithmetic_mean_1_5 (m : ℝ) : m = (1 + 5) / 2 → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_1_5_l3318_331882


namespace NUMINAMATH_CALUDE_chord_count_l3318_331817

/-- The number of points on the circumference of the circle -/
def n : ℕ := 9

/-- The number of points needed to form a chord -/
def r : ℕ := 2

/-- The number of different chords that can be drawn -/
def num_chords : ℕ := Nat.choose n r

theorem chord_count : num_chords = 36 := by
  sorry

end NUMINAMATH_CALUDE_chord_count_l3318_331817


namespace NUMINAMATH_CALUDE_union_problem_l3318_331843

def M : Set ℕ := {0, 1, 2}
def N (x : ℕ) : Set ℕ := {x}

theorem union_problem (x : ℕ) : M ∪ N x = {0, 1, 2, 3} → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_union_problem_l3318_331843


namespace NUMINAMATH_CALUDE_smallest_n_no_sum_of_powers_is_square_l3318_331850

theorem smallest_n_no_sum_of_powers_is_square : ∃ (n : ℕ), n > 1 ∧
  (∀ (m k : ℕ), ¬∃ (a : ℕ), n^m + n^k = a^2) ∧
  (∀ (n' : ℕ), 1 < n' ∧ n' < n →
    ∃ (m k a : ℕ), n'^m + n'^k = a^2) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_n_no_sum_of_powers_is_square_l3318_331850


namespace NUMINAMATH_CALUDE_success_permutations_l3318_331897

def word := "SUCCESS"

-- Define the counts of each letter
def s_count := 3
def c_count := 2
def u_count := 1
def e_count := 1

-- Define the total number of letters
def total_letters := s_count + c_count + u_count + e_count

-- Theorem statement
theorem success_permutations :
  (Nat.factorial total_letters) / 
  (Nat.factorial s_count * Nat.factorial c_count * Nat.factorial u_count * Nat.factorial e_count) = 420 := by
  sorry

end NUMINAMATH_CALUDE_success_permutations_l3318_331897


namespace NUMINAMATH_CALUDE_min_quotient_three_digit_number_l3318_331883

theorem min_quotient_three_digit_number (a : ℕ) :
  a ≠ 0 ∧ a ≠ 7 ∧ a ≠ 8 →
  (∀ x : ℕ, x ≠ 0 ∧ x ≠ 7 ∧ x ≠ 8 →
    (100 * a + 78 : ℚ) / (a + 15) ≤ (100 * x + 78 : ℚ) / (x + 15)) →
  (100 * a + 78 : ℚ) / (a + 15) = 11125 / 1000 :=
sorry

end NUMINAMATH_CALUDE_min_quotient_three_digit_number_l3318_331883


namespace NUMINAMATH_CALUDE_angle_terminal_side_cosine_l3318_331812

theorem angle_terminal_side_cosine (x : ℝ) (α : ℝ) : 
  (∃ (P : ℝ × ℝ), P = (-x, -6) ∧ P ∈ {p | ∃ t : ℝ, p = (t * Real.cos α, t * Real.sin α)}) →
  Real.cos α = 4/5 →
  x = -8 := by
  sorry

end NUMINAMATH_CALUDE_angle_terminal_side_cosine_l3318_331812


namespace NUMINAMATH_CALUDE_range_of_dot_product_line_passes_fixed_point_l3318_331847

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the focus and vertex
def F : ℝ × ℝ := (-1, 0)
def A : ℝ × ℝ := (-2, 0)

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the vector from a point to another
def vector_to (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

-- Theorem 1: Range of PF · PA
theorem range_of_dot_product :
  ∀ P : ℝ × ℝ, ellipse P.1 P.2 →
  0 ≤ dot_product (vector_to P F) (vector_to P A) ∧
  dot_product (vector_to P F) (vector_to P A) ≤ 12 :=
sorry

-- Define the line
def line (k m : ℝ) (x : ℝ) : ℝ := k * x + m

-- Theorem 2: Line passes through fixed point
theorem line_passes_fixed_point :
  ∀ k m : ℝ, ∀ M N : ℝ × ℝ,
  M ≠ N →
  ellipse M.1 M.2 →
  ellipse N.1 N.2 →
  M.2 = line k m M.1 →
  N.2 = line k m N.1 →
  (∃ H : ℝ × ℝ, 
    dot_product (vector_to A H) (vector_to M N) = 0 ∧
    dot_product (vector_to A H) (vector_to A H) = 
    dot_product (vector_to M H) (vector_to H N)) →
  line k m (-2/7) = 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_dot_product_line_passes_fixed_point_l3318_331847


namespace NUMINAMATH_CALUDE_exists_square_with_2018_l3318_331838

theorem exists_square_with_2018 : ∃ n : ℕ, ∃ a b : ℕ, n^2 = a * 10000 + 2018 * b :=
  sorry

end NUMINAMATH_CALUDE_exists_square_with_2018_l3318_331838


namespace NUMINAMATH_CALUDE_polynomial_factors_sum_l3318_331851

/-- Given real numbers a, b, and c, if x^2 + x + 2 is a factor of ax^3 + bx^2 + cx + 5
    and 2x - 1 is a factor of ax^3 + bx^2 + cx - 25/16, then a + b + c = 45/11 -/
theorem polynomial_factors_sum (a b c : ℝ) :
  (∃ d : ℝ, ∀ x : ℝ, a * x^3 + b * x^2 + c * x + 5 = (x^2 + x + 2) * (a * x + d)) →
  (∀ x : ℝ, (2 * x - 1) ∣ (a * x^3 + b * x^2 + c * x - 25/16)) →
  a + b + c = 45/11 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factors_sum_l3318_331851


namespace NUMINAMATH_CALUDE_parabola_directrix_l3318_331880

/-- The parabola equation -/
def parabola_eq (x y : ℝ) : Prop := x = (y^2 - 8*y - 20) / 16

/-- The directrix equation -/
def directrix_eq (x : ℝ) : Prop := x = -6.25

/-- Theorem stating that the given directrix equation is correct for the parabola -/
theorem parabola_directrix : 
  ∀ x y : ℝ, parabola_eq x y → (∃ x_d : ℝ, directrix_eq x_d ∧ 
    -- Additional conditions about the relationship between the point (x,y) on the parabola
    -- and its distance to the directrix would be specified here
    True) :=
by sorry

end NUMINAMATH_CALUDE_parabola_directrix_l3318_331880


namespace NUMINAMATH_CALUDE_swallow_pests_calculation_l3318_331860

/-- The number of pests a frog can catch per day -/
def frog_pests : ℕ := 145

/-- The multiplier for how many times more pests a swallow can eliminate compared to a frog -/
def swallow_multiplier : ℕ := 12

/-- The number of pests a swallow can eliminate per day -/
def swallow_pests : ℕ := frog_pests * swallow_multiplier

theorem swallow_pests_calculation : swallow_pests = 1740 := by
  sorry

end NUMINAMATH_CALUDE_swallow_pests_calculation_l3318_331860


namespace NUMINAMATH_CALUDE_t_equality_l3318_331896

theorem t_equality (t : ℝ) : t = 1 / (2 - Real.rpow 3 (1/3)) → t = (2 + Real.rpow 3 (1/3)) * (2 + Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_t_equality_l3318_331896


namespace NUMINAMATH_CALUDE_negative_sqrt_17_bound_l3318_331826

theorem negative_sqrt_17_bound : -5 < -Real.sqrt 17 ∧ -Real.sqrt 17 < -4 := by
  sorry

end NUMINAMATH_CALUDE_negative_sqrt_17_bound_l3318_331826


namespace NUMINAMATH_CALUDE_ant_borya_position_l3318_331875

/-- Represents a point on the coordinate plane -/
structure Point where
  x : Int
  y : Int

/-- Generates the nth point in the spiral sequence -/
def spiral_point (n : Nat) : Point :=
  sorry

/-- The starting point of the sequence -/
def P₀ : Point := { x := 0, y := 0 }

/-- The second point in the sequence -/
def P₁ : Point := { x := 1, y := 0 }

/-- The spiral sequence of points -/
def P : Nat → Point
  | 0 => P₀
  | 1 => P₁
  | n + 2 => spiral_point (n + 2)

theorem ant_borya_position : P 1557 = { x := 20, y := 17 } := by
  sorry

end NUMINAMATH_CALUDE_ant_borya_position_l3318_331875


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3318_331895

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x : ℝ | x > 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≥ -1} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3318_331895


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3318_331877

theorem inequality_solution_set : 
  {x : ℝ | 2*x + 1 > x + 2} = {x : ℝ | x > 1} := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3318_331877


namespace NUMINAMATH_CALUDE_least_tiles_required_l3318_331803

def room_length : Real := 8.16
def room_width : Real := 4.32
def recess_width : Real := 1.24
def recess_length : Real := 2
def protrusion_width : Real := 0.48
def protrusion_length : Real := 0.96

def main_area : Real := room_length * room_width
def recess_area : Real := recess_width * recess_length
def protrusion_area : Real := protrusion_width * protrusion_length
def total_area : Real := main_area + recess_area + protrusion_area

def tile_size : Real := protrusion_width

theorem least_tiles_required :
  ∃ n : ℕ, n = ⌈total_area / (tile_size * tile_size)⌉ ∧ n = 166 := by
  sorry

end NUMINAMATH_CALUDE_least_tiles_required_l3318_331803


namespace NUMINAMATH_CALUDE_inequality_proof_l3318_331856

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a * b + b * c + c * a = 1) : 
  (a + b + c ≥ Real.sqrt 3) ∧ 
  (Real.sqrt (a / (b * c)) + Real.sqrt (b / (c * a)) + Real.sqrt (c / (a * b)) ≥ 
   Real.sqrt 3 * (Real.sqrt a + Real.sqrt b + Real.sqrt c)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3318_331856


namespace NUMINAMATH_CALUDE_min_value_of_f_l3318_331889

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x-2) - 2*a

theorem min_value_of_f (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) (hf : f a 2 = 1/3) :
  ∃ (m : ℝ), IsMinOn (f a) (Set.Icc 0 3) m ∧ m = -1/3 := by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3318_331889


namespace NUMINAMATH_CALUDE_segment_length_l3318_331814

theorem segment_length : Real.sqrt 193 = Real.sqrt ((8 - 1)^2 + (14 - 2)^2) := by sorry

end NUMINAMATH_CALUDE_segment_length_l3318_331814


namespace NUMINAMATH_CALUDE_divisor_property_characterization_l3318_331878

/-- A positive integer n > 1 satisfies the divisor property if all its positive divisors
    greater than 1 are of the form a^r + 1, where a and r are positive integers and r > 1 -/
def satisfies_divisor_property (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d ∣ n → ∃ a r : ℕ, a > 0 ∧ r > 1 ∧ d = a^r + 1

/-- The main theorem stating that if n satisfies the divisor property,
    then n = 10 or n is a prime of the form a^2 + 1 -/
theorem divisor_property_characterization (n : ℕ) :
  satisfies_divisor_property n →
  (n = 10 ∨ (Nat.Prime n ∧ ∃ a : ℕ, n = a^2 + 1)) :=
by sorry


end NUMINAMATH_CALUDE_divisor_property_characterization_l3318_331878


namespace NUMINAMATH_CALUDE_last_digit_of_322_power_111569_last_digit_is_two_l3318_331891

theorem last_digit_of_322_power_111569 : ℕ → Prop :=
  fun n => (322^111569 : ℕ) % 10 = n

theorem last_digit_is_two : last_digit_of_322_power_111569 2 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_322_power_111569_last_digit_is_two_l3318_331891


namespace NUMINAMATH_CALUDE_profit_minimum_at_radius_one_l3318_331853

noncomputable def profit_function (r : ℝ) : ℝ :=
  0.2 * (4/3) * Real.pi * r^3 - 0.8 * Real.pi * r^2

theorem profit_minimum_at_radius_one :
  ∀ r : ℝ, 0 < r → r ≤ 6 →
  profit_function r ≥ profit_function 1 :=
sorry

end NUMINAMATH_CALUDE_profit_minimum_at_radius_one_l3318_331853


namespace NUMINAMATH_CALUDE_g_shifted_l3318_331854

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

-- Theorem statement
theorem g_shifted (x : ℝ) : g (x + 3) = x^2 + 3*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_g_shifted_l3318_331854


namespace NUMINAMATH_CALUDE_work_fraction_proof_l3318_331836

theorem work_fraction_proof (total_payment : ℚ) (b_payment : ℚ) 
  (h1 : total_payment = 529)
  (h2 : b_payment = 12) :
  (total_payment - b_payment) / total_payment = 517 / 529 := by
  sorry

end NUMINAMATH_CALUDE_work_fraction_proof_l3318_331836


namespace NUMINAMATH_CALUDE_x_minus_y_equals_three_l3318_331840

theorem x_minus_y_equals_three (x y : ℝ) 
  (h1 : x + y = 8) 
  (h2 : x^2 - y^2 = 24) : 
  x - y = 3 := by
sorry

end NUMINAMATH_CALUDE_x_minus_y_equals_three_l3318_331840


namespace NUMINAMATH_CALUDE_fewer_buses_on_river_road_l3318_331867

theorem fewer_buses_on_river_road (num_cars : ℕ) (num_buses : ℕ) : 
  num_cars = 60 →
  num_buses < num_cars →
  num_buses * 3 = num_cars →
  num_cars - num_buses = 40 := by
  sorry

end NUMINAMATH_CALUDE_fewer_buses_on_river_road_l3318_331867


namespace NUMINAMATH_CALUDE_three_distinct_roots_condition_l3318_331823

theorem three_distinct_roots_condition (a : ℝ) :
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (|x^3 - a^3| = x - a) ∧
    (|y^3 - a^3| = y - a) ∧
    (|z^3 - a^3| = z - a)) ↔
  (-2 / Real.sqrt 3 < a ∧ a < -1 / Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_three_distinct_roots_condition_l3318_331823


namespace NUMINAMATH_CALUDE_log3_one_third_l3318_331824

-- Define the logarithm function for base 3
noncomputable def log3 (x : ℝ) := Real.log x / Real.log 3

-- State the theorem
theorem log3_one_third : log3 (1/3) = -1 := by
  sorry

end NUMINAMATH_CALUDE_log3_one_third_l3318_331824


namespace NUMINAMATH_CALUDE_union_of_sets_l3318_331879

theorem union_of_sets : let A : Set ℕ := {2, 3}
                        let B : Set ℕ := {1, 2}
                        A ∪ B = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_sets_l3318_331879


namespace NUMINAMATH_CALUDE_task_completion_probability_l3318_331829

theorem task_completion_probability (p1 p2 p3 p4 : ℚ) 
  (h1 : p1 = 3/8) (h2 : p2 = 3/5) (h3 : p3 = 5/9) (h4 : p4 = 7/12) :
  p1 * (1 - p2) * p3 * (1 - p4) = 5/72 := by
  sorry

end NUMINAMATH_CALUDE_task_completion_probability_l3318_331829


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3318_331894

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptote x - 2y = 0 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b/a = 1/2) :
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = Real.sqrt 5 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3318_331894


namespace NUMINAMATH_CALUDE_elsas_marbles_l3318_331815

theorem elsas_marbles (initial : ℕ) (lost_breakfast : ℕ) (given_lunch : ℕ) (received_mom : ℕ) : 
  initial = 40 →
  lost_breakfast = 3 →
  given_lunch = 5 →
  received_mom = 12 →
  initial - lost_breakfast - given_lunch + received_mom + 2 * given_lunch = 54 := by
  sorry

end NUMINAMATH_CALUDE_elsas_marbles_l3318_331815


namespace NUMINAMATH_CALUDE_remaining_area_formula_l3318_331805

/-- The area of the remaining part when a small square is removed from a larger square -/
def remaining_area (x : ℝ) : ℝ := 9 - x^2

/-- Theorem stating the area of the remaining part when a small square is removed from a larger square -/
theorem remaining_area_formula (x : ℝ) (h : 0 < x ∧ x < 3) : 
  remaining_area x = 9 - x^2 := by
  sorry

end NUMINAMATH_CALUDE_remaining_area_formula_l3318_331805
