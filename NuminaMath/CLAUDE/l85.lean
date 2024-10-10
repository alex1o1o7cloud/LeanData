import Mathlib

namespace ellipse_constant_slope_l85_8545

/-- An ellipse with the given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a
  h_ecc : (a^2 - b^2) / a^2 = 3/4
  h_point : 4/a^2 + 1/b^2 = 1

/-- A point on the ellipse -/
structure PointOnEllipse (E : Ellipse) where
  x : ℝ
  y : ℝ
  h_on_ellipse : x^2/E.a^2 + y^2/E.b^2 = 1

/-- The theorem statement -/
theorem ellipse_constant_slope (E : Ellipse) 
  (h_bisector : ∀ (P Q : PointOnEllipse E), 
    (∃ (k : ℝ), k * (P.x - 2) = P.y - 1 ∧ k * (Q.x - 2) = -(Q.y - 1))) :
  ∀ (P Q : PointOnEllipse E), (Q.y - P.y) / (Q.x - P.x) = 1/2 :=
sorry

end ellipse_constant_slope_l85_8545


namespace final_temperature_of_mixed_gases_l85_8560

/-- The final temperature of mixed gases in thermally insulated vessels -/
theorem final_temperature_of_mixed_gases
  (V₁ V₂ : ℝ) (p₁ p₂ : ℝ) (T₁ T₂ : ℝ) (R : ℝ) :
  V₁ = 1 →
  V₂ = 2 →
  p₁ = 2 →
  p₂ = 3 →
  T₁ = 300 →
  T₂ = 400 →
  R > 0 →
  let n₁ := p₁ * V₁ / (R * T₁)
  let n₂ := p₂ * V₂ / (R * T₂)
  let T := (n₁ * T₁ + n₂ * T₂) / (n₁ + n₂)
  ∃ ε > 0, |T - 369| < ε :=
sorry

end final_temperature_of_mixed_gases_l85_8560


namespace quadratic_inequality_necessary_not_sufficient_l85_8555

theorem quadratic_inequality_necessary_not_sufficient :
  (∃ x : ℝ, (|x - 2| < 1 ∧ ¬(x^2 - 5*x + 4 < 0))) ∧
  (∀ x : ℝ, (x^2 - 5*x + 4 < 0 → |x - 2| < 1)) :=
by sorry

end quadratic_inequality_necessary_not_sufficient_l85_8555


namespace y_derivative_l85_8568

open Real

noncomputable def y (x : ℝ) : ℝ :=
  (6^x * (sin (4*x) * log 6 - 4 * cos (4*x))) / (16 + (log 6)^2)

theorem y_derivative (x : ℝ) : 
  deriv y x = 6^x * sin (4*x) :=
sorry

end y_derivative_l85_8568


namespace log_sqrt12_1728sqrt12_l85_8551

theorem log_sqrt12_1728sqrt12 : Real.log (1728 * Real.sqrt 12) / Real.log (Real.sqrt 12) = 7 := by
  sorry

end log_sqrt12_1728sqrt12_l85_8551


namespace inequality_property_l85_8586

theorem inequality_property (a b c : ℝ) : a * c^2 > b * c^2 → a > b := by
  sorry

end inequality_property_l85_8586


namespace horner_v₁_is_8_l85_8563

/-- Horner's method for polynomial evaluation -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 4x^5 - 12x^4 + 3.5x^3 - 2.6x^2 + 1.7x - 0.8 -/
def f : ℝ → ℝ := fun x => 4 * x^5 - 12 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

/-- The coefficients of the polynomial f in descending order of degree -/
def f_coeffs : List ℝ := [4, -12, 3.5, -2.6, 1.7, -0.8]

/-- The value of x at which we evaluate the polynomial -/
def x : ℝ := 5

/-- v₁ in Horner's method for f(x) when x = 5 -/
def v₁ : ℝ := 4 * x - 12

theorem horner_v₁_is_8 : v₁ = 8 := by sorry

end horner_v₁_is_8_l85_8563


namespace product_not_fifty_l85_8591

theorem product_not_fifty : ∃! (a b : ℚ), (a = 5 ∧ b = 11) ∧ a * b ≠ 50 ∧
  ((a = 1/2 ∧ b = 100) ∨ (a = -5 ∧ b = -10) ∨ (a = 2 ∧ b = 25) ∨ (a = 5/2 ∧ b = 20)) → a * b = 50 :=
by sorry

end product_not_fifty_l85_8591


namespace equal_edge_length_relation_l85_8596

/-- Represents a hexagonal prism -/
structure HexagonalPrism :=
  (edge_length : ℝ)
  (total_edge_length : ℝ)
  (h_total : total_edge_length = 18 * edge_length)

/-- Represents a quadrangular pyramid -/
structure QuadrangularPyramid :=
  (edge_length : ℝ)
  (total_edge_length : ℝ)
  (h_total : total_edge_length = 8 * edge_length)

/-- 
Given a hexagonal prism and a quadrangular pyramid with equal edge lengths,
if the total edge length of the hexagonal prism is 81 cm,
then the total edge length of the quadrangular pyramid is 36 cm.
-/
theorem equal_edge_length_relation 
  (prism : HexagonalPrism) 
  (pyramid : QuadrangularPyramid) 
  (h_equal_edges : prism.edge_length = pyramid.edge_length) 
  (h_prism_total : prism.total_edge_length = 81) : 
  pyramid.total_edge_length = 36 := by
  sorry

end equal_edge_length_relation_l85_8596


namespace evaluate_expression_l85_8518

theorem evaluate_expression (a : ℝ) (h : a = 3) : (5 * a^2 - 11 * a + 6) * (2 * a - 4) = 36 := by
  sorry

end evaluate_expression_l85_8518


namespace inner_set_area_of_specific_triangle_l85_8541

/-- Triangle with side lengths a, b, c -/
structure Triangle (a b c : ℝ) where
  side_a : a > 0
  side_b : b > 0
  side_c : c > 0
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Set of points inside a triangle not within distance d of any side -/
def InnerSet (T : Triangle a b c) (d : ℝ) : Set (ℝ × ℝ) :=
  sorry

/-- Area of a set in ℝ² -/
noncomputable def area (S : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Main theorem -/
theorem inner_set_area_of_specific_triangle :
  let T : Triangle 26 51 73 := ⟨sorry, sorry, sorry, sorry⟩
  let S := InnerSet T 5
  area S = 135 / 28 := by
  sorry

end inner_set_area_of_specific_triangle_l85_8541


namespace volunteer_assignment_count_l85_8572

/-- Stirling number of the second kind -/
def stirling2 (n k : ℕ) : ℕ := sorry

/-- Number of ways to assign n volunteers to k tasks, where each task must have at least one person -/
def assignVolunteers (n k : ℕ) : ℕ := (stirling2 n k) * (Nat.factorial k)

theorem volunteer_assignment_count :
  assignVolunteers 5 3 = 150 := by sorry

end volunteer_assignment_count_l85_8572


namespace five_balls_three_boxes_l85_8558

/-- The number of ways to put n distinguishable balls into k distinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 243 ways to put 5 distinguishable balls into 3 distinguishable boxes -/
theorem five_balls_three_boxes : ways_to_put_balls_in_boxes 5 3 = 243 := by
  sorry

end five_balls_three_boxes_l85_8558


namespace q_gt_one_neither_sufficient_nor_necessary_for_increasing_l85_8519

-- Define a geometric sequence
def geometric_sequence (a₀ : ℝ) (q : ℝ) : ℕ → ℝ :=
  λ n => a₀ * q^n

-- Define monotonically increasing sequence
def monotonically_increasing (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, s n ≤ s (n + 1)

-- Theorem statement
theorem q_gt_one_neither_sufficient_nor_necessary_for_increasing
  (a₀ : ℝ) (q : ℝ) :
  ¬(((q > 1) → monotonically_increasing (geometric_sequence a₀ q)) ∧
    (monotonically_increasing (geometric_sequence a₀ q) → (q > 1))) :=
by sorry

end q_gt_one_neither_sufficient_nor_necessary_for_increasing_l85_8519


namespace unique_four_digit_number_l85_8589

theorem unique_four_digit_number : ∃! n : ℕ, 
  (1000 ≤ n ∧ n < 10000) ∧ 
  (∃ d₁ d₂ : ℕ, d₁ ≠ d₂ ∧ d₁ < 10 ∧ d₂ < 10 ∧
    (∃ x y : ℕ, x < 10 ∧ y < 10 ∧ 10 + (n / 100 % 10) = x * y) ∧
    (10 + (n / 100 % 10) - (n / d₂ % 10) = 1)) ∧
  n = 1014 :=
sorry

end unique_four_digit_number_l85_8589


namespace expression_evaluation_l85_8514

theorem expression_evaluation (d : ℕ) (h : d = 4) : 
  (d^d - d*(d-2)^d + Nat.factorial (d-1))^2 = 39204 := by
  sorry

end expression_evaluation_l85_8514


namespace imaginary_power_sum_l85_8526

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^13 + i^113 = 2*i := by
  sorry

end imaginary_power_sum_l85_8526


namespace platform_length_calculation_l85_8567

/-- Calculates the length of a platform given train parameters --/
theorem platform_length_calculation (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time : ℝ) : 
  train_length = 180 ∧ 
  train_speed_kmph = 72 ∧ 
  crossing_time = 20 →
  (train_speed_kmph * 1000 / 3600) * crossing_time - train_length = 220 := by
  sorry

#check platform_length_calculation

end platform_length_calculation_l85_8567


namespace largest_three_digit_base6_l85_8553

/-- Converts a base-6 number to base-10 --/
def base6ToBase10 (d1 d2 d3 : Nat) : Nat :=
  d1 * 6^2 + d2 * 6^1 + d3 * 6^0

/-- The largest digit in base-6 --/
def maxBase6Digit : Nat := 5

theorem largest_three_digit_base6 :
  base6ToBase10 maxBase6Digit maxBase6Digit maxBase6Digit = 215 := by
  sorry

end largest_three_digit_base6_l85_8553


namespace consecutive_numbers_sum_l85_8512

theorem consecutive_numbers_sum (n : ℕ) : 
  n + (n + 1) + (n + 2) = 60 → (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end consecutive_numbers_sum_l85_8512


namespace ratio_w_y_l85_8505

-- Define the ratios
def ratio_w_x : ℚ := 5 / 4
def ratio_y_z : ℚ := 7 / 5
def ratio_z_x : ℚ := 1 / 8

-- Theorem statement
theorem ratio_w_y (w x y z : ℚ) 
  (hw : w / x = ratio_w_x)
  (hy : y / z = ratio_y_z)
  (hz : z / x = ratio_z_x) : 
  w / y = 25 / 7 := by
  sorry

end ratio_w_y_l85_8505


namespace paintings_on_last_page_paintings_on_last_page_zero_l85_8580

theorem paintings_on_last_page (initial_albums : Nat) (pages_per_album : Nat) 
  (initial_paintings_per_page : Nat) (new_paintings_per_page : Nat) 
  (filled_albums : Nat) (filled_pages_last_album : Nat) : Nat :=
  let total_paintings := initial_albums * pages_per_album * initial_paintings_per_page
  let total_pages_filled := filled_albums * pages_per_album + filled_pages_last_album
  total_paintings - (total_pages_filled * new_paintings_per_page)

theorem paintings_on_last_page_zero : 
  paintings_on_last_page 10 36 8 9 6 28 = 0 := by
  sorry

end paintings_on_last_page_paintings_on_last_page_zero_l85_8580


namespace product_97_103_l85_8535

theorem product_97_103 : 97 * 103 = 9991 := by
  sorry

end product_97_103_l85_8535


namespace repeating_decimal_85_l85_8564

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

theorem repeating_decimal_85 :
  RepeatingDecimal 8 5 = 85 / 99 := by
  sorry

end repeating_decimal_85_l85_8564


namespace root_implies_range_m_l85_8582

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem
theorem root_implies_range_m :
  ∀ m : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ f x = m) → m ∈ Set.Icc (-2) 2 :=
by sorry

end root_implies_range_m_l85_8582


namespace two_year_increase_l85_8574

def yearly_increase (amount : ℚ) : ℚ := amount * (1 + 1/8)

theorem two_year_increase (P : ℚ) (h : P = 2880) : 
  yearly_increase (yearly_increase P) = 3645 := by
  sorry

end two_year_increase_l85_8574


namespace max_value_theorem_l85_8543

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x + y + z = 3) :
  (x^2 - x*y + y^2) * (x^2 - x*z + z^2) * (y^2 - y*z + z^2) ≤ 81/4 ∧
  ∃ (a b c : ℝ), 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 3 ∧
    (a^2 - a*b + b^2) * (a^2 - a*c + c^2) * (b^2 - b*c + c^2) = 81/4 :=
by sorry

end max_value_theorem_l85_8543


namespace cube_volume_l85_8547

/-- The volume of a cube with total edge length of 60 cm is 125 cubic centimeters. -/
theorem cube_volume (total_edge_length : ℝ) (h : total_edge_length = 60) : 
  (total_edge_length / 12)^3 = 125 := by
  sorry

end cube_volume_l85_8547


namespace f_plus_three_odd_l85_8575

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be odd
def IsOdd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem f_plus_three_odd 
  (h1 : IsOdd (fun x ↦ f (x + 1)))
  (h2 : IsOdd (fun x ↦ f (x - 1))) :
  IsOdd (fun x ↦ f (x + 3)) := by
  sorry

end f_plus_three_odd_l85_8575


namespace square_difference_product_l85_8520

theorem square_difference_product : (476 + 424)^2 - 4 * 476 * 424 = 4624 := by
  sorry

end square_difference_product_l85_8520


namespace more_apples_than_pears_l85_8542

theorem more_apples_than_pears :
  let total_fruits : ℕ := 85
  let num_apples : ℕ := 48
  let num_pears : ℕ := total_fruits - num_apples
  num_apples - num_pears = 11 :=
by sorry

end more_apples_than_pears_l85_8542


namespace taco_truck_problem_l85_8554

/-- The price of a hard shell taco, given the conditions of the taco truck problem -/
def hard_shell_taco_price : ℝ := 5

theorem taco_truck_problem :
  let soft_taco_price : ℝ := 2
  let family_hard_tacos : ℕ := 4
  let family_soft_tacos : ℕ := 3
  let other_customers : ℕ := 10
  let other_customer_soft_tacos : ℕ := 2
  let total_earnings : ℝ := 66

  family_hard_tacos * hard_shell_taco_price +
  family_soft_tacos * soft_taco_price +
  other_customers * other_customer_soft_tacos * soft_taco_price = total_earnings :=
by
  sorry

#eval hard_shell_taco_price

end taco_truck_problem_l85_8554


namespace translation_theorem_l85_8537

/-- Represents a point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translates a point left by a given amount -/
def translateLeft (p : Point) (units : ℝ) : Point :=
  { x := p.x - units, y := p.y }

/-- Translates a point down by a given amount -/
def translateDown (p : Point) (units : ℝ) : Point :=
  { x := p.x, y := p.y - units }

theorem translation_theorem :
  let M : Point := { x := 5, y := 2 }
  let M' : Point := translateDown (translateLeft M 3) 2
  M'.x = 2 ∧ M'.y = 0 := by
  sorry

end translation_theorem_l85_8537


namespace third_month_sale_l85_8550

def average_sale : ℕ := 6500
def num_months : ℕ := 6
def sixth_month_sale : ℕ := 4791
def first_month_sale : ℕ := 6635
def second_month_sale : ℕ := 6927
def fourth_month_sale : ℕ := 7230
def fifth_month_sale : ℕ := 6562

theorem third_month_sale :
  let total_sales := average_sale * num_months
  let known_sales := first_month_sale + second_month_sale + fourth_month_sale + fifth_month_sale + sixth_month_sale
  total_sales - known_sales = 14085 := by
sorry

end third_month_sale_l85_8550


namespace min_value_reciprocal_sum_l85_8573

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 10) :
  (1 / x + 2 / y) ≥ (3 + 2 * Real.sqrt 2) / 10 :=
by sorry

end min_value_reciprocal_sum_l85_8573


namespace parallelogram_base_length_l85_8525

/-- Given a parallelogram with area 128 sq m and altitude twice the base, prove the base is 8 m -/
theorem parallelogram_base_length :
  ∀ (base altitude : ℝ),
  base > 0 →
  altitude > 0 →
  altitude = 2 * base →
  base * altitude = 128 →
  base = 8 :=
by
  sorry

end parallelogram_base_length_l85_8525


namespace max_r_value_l85_8566

theorem max_r_value (p q r : ℝ) (sum_eq : p + q + r = 6) (prod_sum_eq : p * q + p * r + q * r = 8) :
  r ≤ 2 + Real.sqrt (20 / 3) := by
  sorry

end max_r_value_l85_8566


namespace sin_sum_product_l85_8506

theorem sin_sum_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (9 * x) = 2 * Real.sin (6 * x) * Real.cos (3 * x) := by
  sorry

end sin_sum_product_l85_8506


namespace m_greater_than_n_l85_8588

/-- Given two quadratic functions M and N, prove that M > N for all real x. -/
theorem m_greater_than_n : ∀ x : ℝ, (x^2 - 3*x + 7) > (-x^2 + x + 1) := by
  sorry

end m_greater_than_n_l85_8588


namespace tic_tac_toe_ties_l85_8569

theorem tic_tac_toe_ties (james_win_rate mary_win_rate : ℚ)
  (h1 : james_win_rate = 4 / 9)
  (h2 : mary_win_rate = 5 / 18) :
  1 - (james_win_rate + mary_win_rate) = 5 / 18 := by
sorry

end tic_tac_toe_ties_l85_8569


namespace homework_problems_left_l85_8507

theorem homework_problems_left (math_problems science_problems finished_problems : ℕ) 
  (h1 : math_problems = 46)
  (h2 : science_problems = 9)
  (h3 : finished_problems = 40) :
  math_problems + science_problems - finished_problems = 15 :=
by sorry

end homework_problems_left_l85_8507


namespace cost_difference_is_six_l85_8515

/-- Represents the cost and consumption of a pizza -/
structure PizzaCost where
  totalSlices : ℕ
  plainCost : ℚ
  toppingCost : ℚ
  daveToppedSlices : ℕ
  davePlainSlices : ℕ

/-- Calculates the difference in cost between Dave's and Doug's portions -/
def costDifference (p : PizzaCost) : ℚ :=
  let totalCost := p.plainCost + p.toppingCost
  let costPerSlice := totalCost / p.totalSlices
  let daveCost := costPerSlice * (p.daveToppedSlices + p.davePlainSlices)
  let dougSlices := p.totalSlices - p.daveToppedSlices - p.davePlainSlices
  let dougCost := (p.plainCost / p.totalSlices) * dougSlices
  daveCost - dougCost

/-- Theorem stating that the cost difference is $6 -/
theorem cost_difference_is_six (p : PizzaCost) 
  (h1 : p.totalSlices = 12)
  (h2 : p.plainCost = 12)
  (h3 : p.toppingCost = 3)
  (h4 : p.daveToppedSlices = 6)
  (h5 : p.davePlainSlices = 2) :
  costDifference p = 6 := by
  sorry

#eval costDifference { totalSlices := 12, plainCost := 12, toppingCost := 3, daveToppedSlices := 6, davePlainSlices := 2 }

end cost_difference_is_six_l85_8515


namespace fraction_to_decimal_l85_8544

theorem fraction_to_decimal : (59 : ℚ) / 160 = (36875 : ℚ) / 100000 := by
  sorry

end fraction_to_decimal_l85_8544


namespace sqrt_two_times_sqrt_three_minus_five_l85_8500

theorem sqrt_two_times_sqrt_three_minus_five (x : ℝ) :
  x = Real.sqrt 2 * Real.sqrt 3 - 5 → x = Real.sqrt 6 - 5 := by
  sorry

end sqrt_two_times_sqrt_three_minus_five_l85_8500


namespace complex_number_problem_l85_8590

theorem complex_number_problem (z : ℂ) :
  (∃ (a : ℝ), z = Complex.I * a) →
  (∃ (b : ℝ), (z + 2)^2 - Complex.I * 8 = Complex.I * b) →
  z = -2 * Complex.I :=
by sorry

end complex_number_problem_l85_8590


namespace min_value_of_g_l85_8556

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := 2 * a^x - 2 * a^(-x)

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2 * f a x

-- Theorem statement
theorem min_value_of_g (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f a 1 = 3) :
  ∃ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, g a x ≤ g a y ∧ g a x = -2 :=
sorry

end min_value_of_g_l85_8556


namespace hoseok_multiplication_l85_8516

theorem hoseok_multiplication (x : ℚ) : x / 11 = 2 → 6 * x = 132 := by
  sorry

end hoseok_multiplication_l85_8516


namespace caravan_camels_l85_8581

theorem caravan_camels (hens goats keepers : ℕ) (camel_feet : ℕ) : 
  hens = 50 → 
  goats = 45 → 
  keepers = 15 → 
  camel_feet = (hens + goats + keepers + 224) * 2 - (hens * 2 + goats * 4 + keepers * 2) → 
  camel_feet / 4 = 6 := by
  sorry

end caravan_camels_l85_8581


namespace jack_marathon_time_l85_8529

/-- Proves that Jack's marathon time is 5 hours given the specified conditions -/
theorem jack_marathon_time
  (marathon_distance : ℝ)
  (jill_time : ℝ)
  (speed_ratio : ℝ)
  (h1 : marathon_distance = 42)
  (h2 : jill_time = 4.2)
  (h3 : speed_ratio = 0.8400000000000001)
  : ℝ :=
by
  sorry

#check jack_marathon_time

end jack_marathon_time_l85_8529


namespace quadratic_monotone_increasing_iff_l85_8593

/-- A quadratic function f(x) = x^2 + bx + c is monotonically increasing 
    on the interval [0, +∞) if and only if b ≥ 0 -/
theorem quadratic_monotone_increasing_iff (b c : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ → x₁^2 + b*x₁ + c < x₂^2 + b*x₂ + c) ↔ b ≥ 0 := by
  sorry

end quadratic_monotone_increasing_iff_l85_8593


namespace s_range_l85_8565

noncomputable def s (x : ℝ) : ℝ := 1 / (2 - x)^3

theorem s_range : Set.range s = {y : ℝ | y < 0 ∨ y > 0} := by sorry

end s_range_l85_8565


namespace total_slices_today_l85_8595

def lunch_slices : ℕ := 7
def dinner_slices : ℕ := 5

theorem total_slices_today : lunch_slices + dinner_slices = 12 := by
  sorry

end total_slices_today_l85_8595


namespace minimum_toddlers_l85_8510

theorem minimum_toddlers (total_teeth : ℕ) (max_pair_teeth : ℕ) (h1 : total_teeth = 90) (h2 : max_pair_teeth = 9) :
  ∃ (n : ℕ), n ≥ 23 ∧
  (∀ (m : ℕ), m < n →
    ¬∃ (teeth_distribution : Fin m → ℕ),
      (∀ i j : Fin m, i ≠ j → teeth_distribution i + teeth_distribution j ≤ max_pair_teeth) ∧
      (Finset.sum (Finset.univ : Finset (Fin m)) teeth_distribution = total_teeth)) :=
by sorry

end minimum_toddlers_l85_8510


namespace units_digit_of_k97_l85_8511

-- Define the modified Lucas sequence
def modifiedLucas : ℕ → ℕ
  | 0 => 3
  | 1 => 1
  | n + 2 => modifiedLucas (n + 1) + modifiedLucas n

-- Define a function to get the units digit
def unitsDigit (n : ℕ) : ℕ :=
  n % 10

-- Theorem statement
theorem units_digit_of_k97 : unitsDigit (modifiedLucas 97) = 7 := by
  sorry

end units_digit_of_k97_l85_8511


namespace food_distribution_l85_8585

/-- The number of days the food initially lasts -/
def initial_days : ℝ := 45

/-- The initial number of men in the camp -/
def initial_men : ℕ := 40

/-- The number of days the food lasts after additional men join -/
def final_days : ℝ := 32.73

/-- The number of additional men who joined the camp -/
def additional_men : ℕ := 15

theorem food_distribution (total_food : ℝ) :
  total_food = initial_men * initial_days ∧
  total_food = (initial_men + additional_men) * final_days :=
sorry

#check food_distribution

end food_distribution_l85_8585


namespace gcf_72_108_l85_8522

theorem gcf_72_108 : Nat.gcd 72 108 = 36 := by
  sorry

end gcf_72_108_l85_8522


namespace job_completion_time_l85_8557

-- Define the problem parameters
def initial_workers : ℕ := 6
def initial_days : ℕ := 8
def days_before_joining : ℕ := 3
def additional_workers : ℕ := 4

-- Define the total work as a fraction
def total_work : ℚ := 1

-- Define the work rate of one worker per day
def work_rate_per_worker : ℚ := 1 / (initial_workers * initial_days)

-- Define the work completed in the first 3 days
def work_completed_first_phase : ℚ := initial_workers * work_rate_per_worker * days_before_joining

-- Define the remaining work
def remaining_work : ℚ := total_work - work_completed_first_phase

-- Define the total number of workers after joining
def total_workers : ℕ := initial_workers + additional_workers

-- Define the work rate of all workers after joining
def work_rate_after_joining : ℚ := total_workers * work_rate_per_worker

-- State the theorem
theorem job_completion_time :
  ∃ (remaining_days : ℕ), 
    (days_before_joining : ℚ) + remaining_days = 6 ∧
    remaining_work = work_rate_after_joining * remaining_days :=
sorry

end job_completion_time_l85_8557


namespace polygon_properties_l85_8579

-- Define the polygon
structure Polygon where
  n : ℕ  -- number of sides
  h : n > 2  -- a polygon must have at least 3 sides

-- Define the ratio of interior to exterior angles
def interiorToExteriorRatio (p : Polygon) : ℚ :=
  (p.n - 2) / 2

-- Theorem statement
theorem polygon_properties (p : Polygon) 
  (h : interiorToExteriorRatio p = 13 / 2) : 
  p.n = 15 ∧ (p.n * (p.n - 3)) / 2 = 90 := by
  sorry


end polygon_properties_l85_8579


namespace harmonic_series_term_count_l85_8594

theorem harmonic_series_term_count (k : ℕ) (h : k ≥ 2) :
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k := by
  sorry

end harmonic_series_term_count_l85_8594


namespace solution_set_part1_range_of_a_part2_l85_8584

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| - 2 * |x + a|

-- Part 1: Solution set for f(x) > 1 when a = 1
theorem solution_set_part1 :
  {x : ℝ | f 1 x > 1} = {x : ℝ | -2 < x ∧ x < -2/3} :=
sorry

-- Part 2: Range of a for f(x) > 0 when x ∈ [2, 3]
theorem range_of_a_part2 :
  {a : ℝ | ∀ x ∈ Set.Icc 2 3, f a x > 0} = {a : ℝ | -5/2 < a ∧ a < -2} :=
sorry

end solution_set_part1_range_of_a_part2_l85_8584


namespace sqrt_equation_solution_l85_8513

theorem sqrt_equation_solution (x : ℝ) : Real.sqrt (5 * x + 9) = 12 → x = 27 := by
  sorry

end sqrt_equation_solution_l85_8513


namespace equation_equivalent_to_lines_l85_8527

/-- The set of points satisfying the given equation is equivalent to the union of two lines -/
theorem equation_equivalent_to_lines :
  ∀ x y : ℝ, 2 * x^2 + y^2 + 3 * x * y + 3 * x + y = 2 ↔ 
  (y = -x - 2 ∨ y = -2 * x + 1) :=
by sorry

end equation_equivalent_to_lines_l85_8527


namespace simplify_and_ratio_l85_8587

theorem simplify_and_ratio : 
  (∀ m : ℝ, (6*m + 12) / 3 = 2*m + 4) ∧ (2 / 4 : ℚ) = 1/2 := by
  sorry

end simplify_and_ratio_l85_8587


namespace sixteen_students_not_liking_sports_l85_8546

/-- The number of students who do not like basketball, cricket, or football -/
def students_not_liking_sports (total : ℕ) (basketball cricket football : ℕ) 
  (basketball_cricket cricket_football basketball_football : ℕ) (all_three : ℕ) : ℕ :=
  total - (basketball + cricket + football - basketball_cricket - cricket_football - basketball_football + all_three)

/-- Theorem stating that 16 students do not like any of the three sports -/
theorem sixteen_students_not_liking_sports : 
  students_not_liking_sports 50 20 18 12 8 6 5 3 = 16 := by
  sorry

end sixteen_students_not_liking_sports_l85_8546


namespace hash_difference_l85_8523

/-- Custom operation # -/
def hash (x y : ℤ) : ℤ := x * y - 3 * x + y

/-- Theorem stating the result of (5 # 3) - (3 # 5) -/
theorem hash_difference : hash 5 3 - hash 3 5 = -8 := by
  sorry

end hash_difference_l85_8523


namespace min_value_floor_sum_l85_8521

theorem min_value_floor_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ∃ (m : ℕ), m = 4 ∧
  (∀ (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0),
    ⌊(2*a + b) / c⌋ + ⌊(2*b + c) / a⌋ + ⌊(2*c + a) / b⌋ + ⌊(a + b + c) / (a + b)⌋ ≥ m) ∧
  (⌊(2*x + y) / z⌋ + ⌊(2*y + z) / x⌋ + ⌊(2*z + x) / y⌋ + ⌊(x + y + z) / (x + y)⌋ = m) :=
by sorry

end min_value_floor_sum_l85_8521


namespace silverware_probability_l85_8562

def total_silverware : ℕ := 24
def forks : ℕ := 8
def spoons : ℕ := 10
def knives : ℕ := 6
def pieces_removed : ℕ := 4

theorem silverware_probability :
  let total_ways := Nat.choose total_silverware pieces_removed
  let favorable_ways := Nat.choose forks 2 * Nat.choose spoons 2
  (favorable_ways : ℚ) / total_ways = 18 / 91 := by sorry

end silverware_probability_l85_8562


namespace jason_additional_manager_months_l85_8531

/-- Calculates the additional months Jason worked as a manager -/
def additional_manager_months (bartender_years : ℕ) (manager_years : ℕ) (total_months : ℕ) : ℕ :=
  total_months - (bartender_years * 12 + manager_years * 12)

/-- Proves that Jason worked 6 additional months as a manager -/
theorem jason_additional_manager_months :
  additional_manager_months 9 3 150 = 6 := by
  sorry

end jason_additional_manager_months_l85_8531


namespace arithmetic_square_root_l85_8538

theorem arithmetic_square_root (n : ℝ) (h1 : n > 0) 
  (h2 : ∃ x : ℝ, (x + 1)^2 = n ∧ (2*x - 4)^2 = n) : 
  Real.sqrt n = 2 := by
  sorry

end arithmetic_square_root_l85_8538


namespace kekai_remaining_money_l85_8504

def shirt_price : ℝ := 1
def shirt_discount : ℝ := 0.2
def pants_price : ℝ := 3
def pants_discount : ℝ := 0.1
def hat_price : ℝ := 2
def hat_discount : ℝ := 0
def shoes_price : ℝ := 10
def shoes_discount : ℝ := 0.15
def parent_contribution : ℝ := 0.35

def shirts_sold : ℕ := 5
def pants_sold : ℕ := 5
def hats_sold : ℕ := 3
def shoes_sold : ℕ := 2

def total_sales (shirt_price pants_price hat_price shoes_price : ℝ)
                (shirt_discount pants_discount hat_discount shoes_discount : ℝ)
                (shirts_sold pants_sold hats_sold shoes_sold : ℕ) : ℝ :=
  (shirt_price * (1 - shirt_discount) * shirts_sold) +
  (pants_price * (1 - pants_discount) * pants_sold) +
  (hat_price * (1 - hat_discount) * hats_sold) +
  (shoes_price * (1 - shoes_discount) * shoes_sold)

def remaining_money (total : ℝ) (contribution : ℝ) : ℝ :=
  total * (1 - contribution)

theorem kekai_remaining_money :
  remaining_money (total_sales shirt_price pants_price hat_price shoes_price
                                shirt_discount pants_discount hat_discount shoes_discount
                                shirts_sold pants_sold hats_sold shoes_sold)
                  parent_contribution = 26.32 := by
  sorry

end kekai_remaining_money_l85_8504


namespace toms_age_ratio_l85_8530

theorem toms_age_ratio (T N : ℝ) : T > 0 → N > 0 → T - N = 3 * (T - 3 * N) → T / N = 4 := by
  sorry

end toms_age_ratio_l85_8530


namespace gcd_factorial_eight_six_squared_l85_8503

theorem gcd_factorial_eight_six_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 5760 := by
  sorry

end gcd_factorial_eight_six_squared_l85_8503


namespace sphere_surface_area_from_prism_l85_8552

/-- Given a right square prism with height 4 and volume 16, 
    where all vertices are on the surface of a sphere,
    prove that the surface area of the sphere is 24π -/
theorem sphere_surface_area_from_prism (h : ℝ) (v : ℝ) (r : ℝ) : 
  h = 4 →
  v = 16 →
  v = h * r^2 →
  r^2 + h^2 / 4 + r^2 = (2 * r)^2 →
  4 * π * ((r^2 + h^2 / 4 + r^2) / 4) = 24 * π :=
by sorry

end sphere_surface_area_from_prism_l85_8552


namespace equation_represents_hyperbola_l85_8570

/-- The equation (x-3)^2 = 9(y+2)^2 - 81 represents a hyperbola -/
theorem equation_represents_hyperbola :
  ∃ (a b h k : ℝ) (A B : ℝ → ℝ → Prop),
    (∀ x y, A x y ↔ (x - 3)^2 = 9*(y + 2)^2 - 81) ∧
    (∀ x y, B x y ↔ ((x - h) / a)^2 - ((y - k) / b)^2 = 1) ∧
    (∀ x y, A x y ↔ B x y) :=
by sorry

end equation_represents_hyperbola_l85_8570


namespace smallest_integer_y_l85_8501

theorem smallest_integer_y : ∃ y : ℤ, (1 : ℚ) / 4 < (y : ℚ) / 7 ∧ (y : ℚ) / 7 < 2 / 3 ∧ ∀ z : ℤ, (1 : ℚ) / 4 < (z : ℚ) / 7 ∧ (z : ℚ) / 7 < 2 / 3 → y ≤ z :=
by sorry

end smallest_integer_y_l85_8501


namespace matrix_equation_solution_l85_8540

theorem matrix_equation_solution : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  N^3 - 3 * N^2 + 4 * N = !![8, 16; 4, 8] :=
by
  -- Define the matrix N
  let N : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; 1, 2]
  
  -- Assert that N satisfies the equation
  have h : N^3 - 3 * N^2 + 4 * N = !![8, 16; 4, 8] := by sorry
  
  -- Prove existence
  exact ⟨N, h⟩

#check matrix_equation_solution

end matrix_equation_solution_l85_8540


namespace odot_inequality_range_l85_8536

-- Define the ⊙ operation
def odot (a b : ℝ) : ℝ := a * b + 2 * a + b

-- State the theorem
theorem odot_inequality_range :
  ∀ x : ℝ, odot x (x - 2) < 0 ↔ -2 < x ∧ x < 1 := by sorry

end odot_inequality_range_l85_8536


namespace jia_candies_theorem_l85_8578

/-- Represents a configuration of lines in a plane -/
structure LineConfiguration where
  num_lines : ℕ
  num_parallel_pairs : ℕ

/-- Calculates the number of intersections for a given number of lines -/
def num_intersections (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- Calculates the total number of candies for a given line configuration -/
def total_candies (config : LineConfiguration) : ℕ :=
  num_intersections config.num_lines + config.num_parallel_pairs

/-- Theorem: Given 5 lines with one parallel pair, Jia receives 11 candies -/
theorem jia_candies_theorem (config : LineConfiguration) 
  (h1 : config.num_lines = 5)
  (h2 : config.num_parallel_pairs = 1) :
  total_candies config = 11 := by
  sorry

end jia_candies_theorem_l85_8578


namespace enclosed_area_of_special_curve_l85_8597

/-- The area enclosed by a curve consisting of 9 congruent circular arcs, 
    each of length 2π/3, with centers on the vertices of a regular hexagon 
    with side length 3, is equal to 13.5√3 + π. -/
theorem enclosed_area_of_special_curve (
  n : ℕ) (arc_length : ℝ) (hexagon_side : ℝ) (enclosed_area : ℝ) : 
  n = 9 → 
  arc_length = 2 * Real.pi / 3 → 
  hexagon_side = 3 → 
  enclosed_area = 13.5 * Real.sqrt 3 + Real.pi → 
  enclosed_area = 
    (3 * Real.sqrt 3 / 2 * hexagon_side^2) + (n * arc_length * (arc_length / (2 * Real.pi))) :=
by sorry

end enclosed_area_of_special_curve_l85_8597


namespace mary_chestnut_pick_l85_8571

/-- Given three people picking chestnuts with specific relationships between their picks,
    prove that one person picked a certain amount. -/
theorem mary_chestnut_pick (peter lucy mary : ℝ) 
  (h1 : mary = 2 * peter)
  (h2 : lucy = peter + 2)
  (h3 : peter + mary + lucy = 26) :
  mary = 12 := by
  sorry

end mary_chestnut_pick_l85_8571


namespace proportional_sum_ratio_l85_8549

theorem proportional_sum_ratio (x y z : ℝ) (h : x / 2 = y / 3 ∧ y / 3 = z / 4 ∧ z / 4 ≠ 0) : 
  (2 * x + 3 * y) / z = 13 / 4 := by
  sorry

end proportional_sum_ratio_l85_8549


namespace find_x_l85_8524

theorem find_x (x y z a b c d k : ℝ) 
  (h1 : (x * y + k) / (x + y) = a)
  (h2 : (x * z + k) / (x + z) = b)
  (h3 : (y * z + k) / (y + z) = c)
  (hk : k ≠ 0) :
  x = (2 * a * b * c * d) / (b * (a * c - k) + c * (a * b - k) - a * (b * c - k)) :=
sorry

end find_x_l85_8524


namespace sequence_fifth_term_l85_8598

/-- Given a sequence {aₙ} with the following properties:
  1) a₁ = 1
  2) aₙ - aₙ₋₁ = 2 for n ≥ 2, n ∈ ℕ*
  Prove that a₅ = 9 -/
theorem sequence_fifth_term (a : ℕ+ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ+, n ≥ 2 → a n - a (n-1) = 2) :
  a 5 = 9 := by
  sorry

end sequence_fifth_term_l85_8598


namespace hypotenuse_length_l85_8517

-- Define a right-angled triangle
structure RightTriangle where
  a : ℝ  -- Length of one side
  b : ℝ  -- Length of another side
  c : ℝ  -- Length of the hypotenuse
  right_angle : c^2 = a^2 + b^2  -- Pythagorean theorem

-- Theorem statement
theorem hypotenuse_length (t : RightTriangle) 
  (sum_of_squares : t.a^2 + t.b^2 + t.c^2 = 1450) : 
  t.c = Real.sqrt 725 := by
  sorry

end hypotenuse_length_l85_8517


namespace ellipse_and_line_equations_l85_8508

noncomputable section

-- Define the ellipse C
def ellipse_C (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Define the focal distance
def c : ℝ := Real.sqrt 3

-- Define the perimeter of triangle MF₁F₂
def triangle_perimeter : ℝ := 4 + 2 * Real.sqrt 3

-- Define point P
def P : ℝ × ℝ := (0, 2)

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem ellipse_and_line_equations :
  ∃ (a b : ℝ),
    -- Conditions
    ellipse_C a b (-c) 0 ∧
    (∀ x y, ellipse_C a b x y → 
      ∃ (m : ℝ × ℝ), Real.sqrt ((x - (-c))^2 + y^2) + Real.sqrt ((x - c)^2 + y^2) = triangle_perimeter) ∧
    -- Conclusions
    (a = 2 ∧ b = 1) ∧
    (∃ (k : ℝ), k = 2 ∨ k = -2) ∧
    (∀ k, k = 2 ∨ k = -2 →
      ∃ (x₁ y₁ x₂ y₂ : ℝ),
        ellipse_C 2 1 x₁ y₁ ∧
        ellipse_C 2 1 x₂ y₂ ∧
        y₁ = k * x₁ - 2 ∧
        y₂ = k * x₂ - 2 ∧
        perpendicular x₁ y₁ x₂ y₂) :=
by sorry

end ellipse_and_line_equations_l85_8508


namespace positive_solution_square_root_form_l85_8534

theorem positive_solution_square_root_form :
  ∃ (a' b' : ℕ+), 
    (∃ (x : ℝ), x^2 + 14*x = 96 ∧ x > 0 ∧ x = Real.sqrt a' - b') ∧
    a' = 145 ∧ 
    b' = 7 ∧
    (a' : ℕ) + (b' : ℕ) = 152 := by
  sorry

end positive_solution_square_root_form_l85_8534


namespace ratio_of_sum_and_difference_l85_8502

theorem ratio_of_sum_and_difference (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y = 6 * (x - y)) : x / y = 7 / 5 := by
  sorry

end ratio_of_sum_and_difference_l85_8502


namespace cylinder_volume_increase_l85_8533

/-- Given a cylinder whose radius is increased by a factor x and height is doubled,
    resulting in a volume 18 times the original, prove that x = 3 -/
theorem cylinder_volume_increase (r h x : ℝ) (hr : r > 0) (hh : h > 0) (hx : x > 0) :
  2 * x^2 * (π * r^2 * h) = 18 * (π * r^2 * h) → x = 3 := by
  sorry

#check cylinder_volume_increase

end cylinder_volume_increase_l85_8533


namespace football_season_games_l85_8509

/-- Calculates the total number of football games in a season -/
def total_games (months : ℕ) (games_per_month : ℕ) : ℕ :=
  months * games_per_month

theorem football_season_games :
  let season_length : ℕ := 17
  let games_per_month : ℕ := 19
  total_games season_length games_per_month = 323 := by
sorry

end football_season_games_l85_8509


namespace decimal_to_binary_15_l85_8539

theorem decimal_to_binary_15 : (15 : ℕ) = 0b1111 := by sorry

end decimal_to_binary_15_l85_8539


namespace inequality_solution_l85_8599

-- Define the polynomial function
def f (x : ℝ) := x^3 - 4*x^2 - x + 20

-- Define the set of x satisfying the inequality
def S : Set ℝ := {x | f x > 0}

-- State the theorem
theorem inequality_solution : S = Set.Ioi (-4) ∪ Set.Ioi 1 := by sorry

end inequality_solution_l85_8599


namespace polynomial_multiplication_l85_8528

theorem polynomial_multiplication (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end polynomial_multiplication_l85_8528


namespace fixed_point_exponential_function_l85_8592

theorem fixed_point_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x => a^(2*x - 1) - 2
  f (1/2) = -1 := by
  sorry

end fixed_point_exponential_function_l85_8592


namespace polynomial_expansion_l85_8577

theorem polynomial_expansion (x : ℝ) : 
  (3*x^2 + 4*x + 8)*(x - 2) - (x - 2)*(2*x^2 + 5*x - 72) + (4*x - 21)*(x - 2)*(x - 3) = 
  5*x^3 - 23*x^2 + 43*x - 34 := by
sorry

end polynomial_expansion_l85_8577


namespace inequality_and_equality_condition_l85_8548

theorem inequality_and_equality_condition (a b : ℝ) : 
  (a^2 + 4*b^2 + 4*b - 4*a + 5 ≥ 0) ∧ 
  (a^2 + 4*b^2 + 4*b - 4*a + 5 = 0 ↔ a = 2 ∧ b = -1/2) := by
  sorry

end inequality_and_equality_condition_l85_8548


namespace polynomial_factorization_l85_8559

theorem polynomial_factorization (m n : ℝ) : 
  (∀ x, x^2 + m*x + n = (x+1)*(x+3)) → m - n = 1 := by
  sorry

end polynomial_factorization_l85_8559


namespace smallest_number_l85_8532

def binary_to_decimal (n : ℕ) : ℕ := n

def base_6_to_decimal (n : ℕ) : ℕ := n

def base_4_to_decimal (n : ℕ) : ℕ := n

def base_9_to_decimal (n : ℕ) : ℕ := n

theorem smallest_number :
  let a := binary_to_decimal 111111
  let b := base_6_to_decimal 210
  let c := base_4_to_decimal 1000
  let d := base_9_to_decimal 81
  a < b ∧ a < c ∧ a < d :=
by sorry

end smallest_number_l85_8532


namespace p_sufficient_not_necessary_for_q_l85_8583

/-- Condition p: 0 < x < 2 -/
def p (x : ℝ) : Prop := 0 < x ∧ x < 2

/-- Condition q: -1 < x < 3 -/
def q (x : ℝ) : Prop := -1 < x ∧ x < 3

/-- p is sufficient but not necessary for q -/
theorem p_sufficient_not_necessary_for_q :
  (∀ x : ℝ, p x → q x) ∧ (∃ x : ℝ, q x ∧ ¬p x) := by sorry

end p_sufficient_not_necessary_for_q_l85_8583


namespace union_of_sets_l85_8561

theorem union_of_sets : 
  let A : Set ℕ := {1, 3}
  let B : Set ℕ := {1, 2, 4, 5}
  A ∪ B = {1, 2, 3, 4, 5} := by
sorry

end union_of_sets_l85_8561


namespace nonagon_diagonals_l85_8576

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def nonagon_sides : ℕ := 9

theorem nonagon_diagonals : num_diagonals nonagon_sides = 27 := by
  sorry

end nonagon_diagonals_l85_8576
