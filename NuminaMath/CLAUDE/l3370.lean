import Mathlib

namespace trigonometric_simplification_l3370_337078

theorem trigonometric_simplification (x z : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + z) ^ 2 - 2 * Real.sin x * Real.sin z * Real.sin (x + z) = Real.sin z ^ 2 := by
  sorry

end trigonometric_simplification_l3370_337078


namespace derivative_of_y_l3370_337014

noncomputable def y (x : ℝ) : ℝ := (1 + Real.cos (2 * x))^3

theorem derivative_of_y (x : ℝ) :
  deriv y x = -48 * (Real.cos x)^5 * Real.sin x := by sorry

end derivative_of_y_l3370_337014


namespace equation_solution_l3370_337024

/-- Given an equation y = a + b/x where a and b are constants, 
    if y = 3 when x = 2 and y = 2 when x = 4, then a + b = 5 -/
theorem equation_solution (a b : ℝ) : 
  (3 = a + b / 2) → (2 = a + b / 4) → (a + b = 5) := by
  sorry

end equation_solution_l3370_337024


namespace probability_red_or_green_is_13_22_l3370_337048

/-- Represents the count of jelly beans for each color -/
structure JellyBeanCounts where
  orange : ℕ
  purple : ℕ
  red : ℕ
  green : ℕ

/-- Calculates the probability of selecting either a red or green jelly bean -/
def probability_red_or_green (counts : JellyBeanCounts) : ℚ :=
  (counts.red + counts.green : ℚ) / (counts.orange + counts.purple + counts.red + counts.green)

/-- Theorem stating the probability of selecting a red or green jelly bean -/
theorem probability_red_or_green_is_13_22 :
  let counts : JellyBeanCounts := ⟨4, 5, 6, 7⟩
  probability_red_or_green counts = 13 / 22 := by
  sorry

end probability_red_or_green_is_13_22_l3370_337048


namespace constant_value_c_l3370_337070

theorem constant_value_c (b c : ℚ) :
  (∀ x : ℚ, (x + 3) * (x + b) = x^2 + c*x + 8) →
  c = 17/3 := by
sorry

end constant_value_c_l3370_337070


namespace expand_and_simplify_l3370_337094

theorem expand_and_simplify (x : ℝ) : 6 * (x - 3) * (x + 10) = 6 * x^2 + 42 * x - 180 := by
  sorry

end expand_and_simplify_l3370_337094


namespace dip_amount_is_twenty_l3370_337022

/-- Represents the amount of dip that can be made given a budget and artichoke-to-dip ratio --/
def dip_amount (budget : ℚ) (artichokes_per_batch : ℕ) (ounces_per_batch : ℚ) (total_ounces : ℚ) : ℚ :=
  let price_per_artichoke : ℚ := budget / (total_ounces / ounces_per_batch * artichokes_per_batch)
  let artichokes_bought : ℚ := budget / price_per_artichoke
  (artichokes_bought / artichokes_per_batch) * ounces_per_batch

/-- Theorem stating that under given conditions, 20 ounces of dip can be made --/
theorem dip_amount_is_twenty :
  dip_amount 15 3 5 20 = 20 := by
  sorry

end dip_amount_is_twenty_l3370_337022


namespace ellipse_equation_l3370_337076

/-- Given a circle and an ellipse with specific properties, prove the equation of the ellipse -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  (∃ (A B : ℝ × ℝ),
    -- Point (1, 1/2) is on the circle x^2 + y^2 = 1
    1^2 + (1/2)^2 = 1 ∧
    -- A and B are points on the circle
    A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧
    -- Line AB passes through (1, 0) (focus) and (0, 2) (upper vertex)
    ∃ (m c : ℝ), (m * 1 + c = 0) ∧ (m * 0 + c = 2) ∧
    (m * A.1 + c = A.2) ∧ (m * B.1 + c = B.2) ∧
    -- Ellipse equation
    ∀ (x y : ℝ), (x^2 / a^2) + (y^2 / b^2) = 1) →
  a^2 = 5 ∧ b^2 = 4 := by
sorry

end ellipse_equation_l3370_337076


namespace joan_balloon_count_l3370_337043

/-- Given an initial count of balloons and a number of lost balloons, 
    calculate the final count of balloons. -/
def final_balloon_count (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost

/-- Theorem stating that with 8 initial balloons and 2 lost balloons, 
    the final count is 6. -/
theorem joan_balloon_count : final_balloon_count 8 2 = 6 := by
  sorry

end joan_balloon_count_l3370_337043


namespace wrong_value_correction_l3370_337073

theorem wrong_value_correction (n : ℕ) (initial_mean correct_mean wrong_value : ℝ) 
  (h1 : n = 20)
  (h2 : initial_mean = 150)
  (h3 : wrong_value = 135)
  (h4 : correct_mean = 151.25) :
  (n : ℝ) * correct_mean - ((n : ℝ) * initial_mean - wrong_value) = 160 := by
  sorry

end wrong_value_correction_l3370_337073


namespace polynomial_expansion_l3370_337021

theorem polynomial_expansion (x : ℝ) : 
  (3*x^3 + x^2 - 5*x + 9)*(x + 2) - (x + 2)*(2*x^3 - 4*x + 8) + (x^2 - 6*x + 13)*(x + 2)*(x - 3) = 
  2*x^4 + x^3 + 9*x^2 + 23*x + 2 := by
sorry

end polynomial_expansion_l3370_337021


namespace equation_solution_l3370_337032

theorem equation_solution : 
  let eq (x : ℝ) := x^3 + Real.log 25 + Real.log 32 + Real.log 53 * x - Real.log 23 - Real.log 35 - Real.log 52 * x^2 - 1
  (eq (Real.log 23) = 0) ∧ (eq (Real.log 35) = 0) ∧ (eq (Real.log 52) = 0) := by
  sorry

end equation_solution_l3370_337032


namespace power_of_nine_mod_fifty_l3370_337009

theorem power_of_nine_mod_fifty : 9^1002 % 50 = 1 := by
  sorry

end power_of_nine_mod_fifty_l3370_337009


namespace min_abc_value_l3370_337062

-- Define the quadratic function P(x)
def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the condition for P(x) having exactly one real root
def has_one_root (a b c : ℝ) : Prop := ∃! x : ℝ, P a b c x = 0

-- Define the condition for P(P(P(x))) having exactly three different real roots
def triple_P_has_three_roots (a b c : ℝ) : Prop :=
  ∃! x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    P a b c (P a b c (P a b c x)) = 0 ∧
    P a b c (P a b c (P a b c y)) = 0 ∧
    P a b c (P a b c (P a b c z)) = 0

-- State the theorem
theorem min_abc_value (a b c : ℝ) :
  has_one_root a b c →
  triple_P_has_three_roots a b c →
  ∀ a' b' c' : ℝ, has_one_root a' b' c' → triple_P_has_three_roots a' b' c' →
    a * b * c ≤ a' * b' * c' →
    a * b * c = -2 :=
sorry

end min_abc_value_l3370_337062


namespace recurring_decimal_fraction_sum_l3370_337040

theorem recurring_decimal_fraction_sum (a b : ℕ+) : 
  (a.val : ℚ) / (b.val : ℚ) = 36 / 99 → 
  Nat.gcd a.val b.val = 1 → 
  a.val + b.val = 15 := by
sorry

end recurring_decimal_fraction_sum_l3370_337040


namespace original_professors_count_l3370_337061

/-- The original number of professors in the DVEU Department of Mathematical Modeling. -/
def original_professors : ℕ := 5

/-- The number of failing grades given in the first academic year. -/
def first_year_grades : ℕ := 6480

/-- The number of failing grades given in the second academic year. -/
def second_year_grades : ℕ := 11200

/-- The increase in the number of professors in the second year. -/
def professor_increase : ℕ := 3

theorem original_professors_count :
  (first_year_grades % original_professors = 0) ∧
  (second_year_grades % (original_professors + professor_increase) = 0) ∧
  (first_year_grades / original_professors < second_year_grades / (original_professors + professor_increase)) ∧
  (∀ p : ℕ, p < original_professors →
    (first_year_grades % p = 0 ∧ 
     second_year_grades % (p + professor_increase) = 0) → 
    (first_year_grades / p ≥ second_year_grades / (p + professor_increase))) :=
by sorry

end original_professors_count_l3370_337061


namespace drop_notation_l3370_337091

/-- Represents a temperature change in Celsius -/
structure TempChange where
  value : ℤ

/-- Notation for temperature changes -/
def temp_notation (change : TempChange) : ℤ :=
  change.value

/-- Given condition: A temperature rise of 3℃ is denoted as +3℃ -/
axiom rise_notation : temp_notation ⟨3⟩ = 3

/-- Theorem: A temperature drop of 8℃ is denoted as -8℃ -/
theorem drop_notation : temp_notation ⟨-8⟩ = -8 := by
  sorry

end drop_notation_l3370_337091


namespace max_sum_on_circle_max_sum_achieved_l3370_337020

theorem max_sum_on_circle (x y : ℤ) (h : x^2 + y^2 = 50) : x + y ≤ 8 := by
  sorry

theorem max_sum_achieved : ∃ (x y : ℤ), x^2 + y^2 = 50 ∧ x + y = 8 := by
  sorry

end max_sum_on_circle_max_sum_achieved_l3370_337020


namespace initial_velocity_is_three_l3370_337059

/-- The displacement function of an object moving in a straight line -/
def displacement (t : ℝ) : ℝ := 3 * t - t^2

/-- The velocity function of the object -/
def velocity (t : ℝ) : ℝ := 3 - 2 * t

/-- The theorem stating that the initial velocity is 3 -/
theorem initial_velocity_is_three : velocity 0 = 3 := by sorry

end initial_velocity_is_three_l3370_337059


namespace copy_machine_rate_copy_machine_rate_proof_l3370_337019

/-- Given two copy machines working together for 30 minutes to produce 3000 copies,
    where one machine produces 65 copies per minute, prove that the other machine
    must produce 35 copies per minute. -/
theorem copy_machine_rate : ℕ → Prop :=
  fun x =>
    -- x is the number of copies per minute for the first machine
    -- 65 is the number of copies per minute for the second machine
    -- 30 is the number of minutes they work
    -- 3000 is the total number of copies produced
    30 * x + 30 * 65 = 3000 →
    x = 35

-- The proof would go here, but we're skipping it as requested
theorem copy_machine_rate_proof : copy_machine_rate 35 := by sorry

end copy_machine_rate_copy_machine_rate_proof_l3370_337019


namespace quadratic_solution_sum_l3370_337015

theorem quadratic_solution_sum (m n p : ℤ) : 
  (∃ x : ℚ, x * (5 * x - 11) = -6) ∧
  (∃ x y : ℚ, x = (m + n.sqrt : ℚ) / p ∧ y = (m - n.sqrt : ℚ) / p ∧ 
    x * (5 * x - 11) = -6 ∧ y * (5 * y - 11) = -6) ∧
  Nat.gcd (Nat.gcd m.natAbs n.natAbs) p.natAbs = 1 →
  m + n + p = 70 :=
sorry

end quadratic_solution_sum_l3370_337015


namespace hundredth_digit_of_13_over_90_l3370_337081

theorem hundredth_digit_of_13_over_90 : 
  ∃ (d : ℕ), d = 4 ∧ 
  (∃ (a b : ℕ), (13 : ℚ) / 90 = a + (d : ℚ) / 10^100 + b / 10^101 ∧ 
                0 ≤ b ∧ b < 10) := by
  sorry

end hundredth_digit_of_13_over_90_l3370_337081


namespace school_dinner_theatre_attendance_l3370_337098

theorem school_dinner_theatre_attendance
  (total_tickets : ℕ)
  (total_revenue : ℕ)
  (child_ticket_price : ℕ)
  (adult_ticket_price : ℕ)
  (h1 : total_tickets = 225)
  (h2 : total_revenue = 1875)
  (h3 : child_ticket_price = 6)
  (h4 : adult_ticket_price = 9) :
  ∃ (children_tickets : ℕ) (adult_tickets : ℕ),
    children_tickets + adult_tickets = total_tickets ∧
    child_ticket_price * children_tickets + adult_ticket_price * adult_tickets = total_revenue ∧
    children_tickets = 50 := by
  sorry


end school_dinner_theatre_attendance_l3370_337098


namespace reference_city_hospitals_l3370_337025

/-- The number of hospitals in the reference city -/
def reference_hospitals : ℕ := sorry

/-- The number of stores in the reference city -/
def reference_stores : ℕ := 2000

/-- The number of schools in the reference city -/
def reference_schools : ℕ := 200

/-- The number of police stations in the reference city -/
def reference_police : ℕ := 20

/-- The total number of buildings in the new city -/
def new_city_total : ℕ := 2175

theorem reference_city_hospitals :
  reference_stores / 2 + 2 * reference_hospitals + (reference_schools - 50) + (reference_police + 5) = new_city_total →
  reference_hospitals = 500 := by
  sorry

end reference_city_hospitals_l3370_337025


namespace max_thursday_hours_l3370_337085

def max_video_game_hours (wednesday : ℝ) (friday : ℝ) (average : ℝ) : Prop :=
  ∃ thursday : ℝ,
    wednesday = 2 ∧
    friday > wednesday + 3 ∧
    average = 3 ∧
    (wednesday + thursday + friday) / 3 = average ∧
    thursday = 2

theorem max_thursday_hours :
  max_video_game_hours 2 5 3 :=
sorry

end max_thursday_hours_l3370_337085


namespace find_x2_l3370_337030

theorem find_x2 (x₁ x₂ x₃ : ℝ) 
  (h_order : x₁ < x₂ ∧ x₂ < x₃)
  (h_sum1 : x₁ + x₂ = 14)
  (h_sum2 : x₁ + x₃ = 17)
  (h_sum3 : x₂ + x₃ = 33) : 
  x₂ = 15 := by
sorry

end find_x2_l3370_337030


namespace base_for_256_four_digits_l3370_337001

-- Define the property of a number having exactly 4 digits in a given base
def has_four_digits (n : ℕ) (b : ℕ) : Prop :=
  b ^ 3 ≤ n ∧ n < b ^ 4

-- State the theorem
theorem base_for_256_four_digits :
  ∃! b : ℕ, has_four_digits 256 b ∧ b = 6 := by
  sorry

end base_for_256_four_digits_l3370_337001


namespace percent_of_x_is_z_l3370_337090

theorem percent_of_x_is_z (x y z : ℝ) 
  (h1 : 0.45 * z = 1.20 * y) 
  (h2 : y = 0.75 * x) : 
  z = 2 * x := by
sorry

end percent_of_x_is_z_l3370_337090


namespace solution_set_empty_implies_a_range_main_theorem_l3370_337012

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 + a * x + 3

-- State the theorem
theorem solution_set_empty_implies_a_range (a : ℝ) :
  (∀ x, f a x ≥ 0) → 0 ≤ a ∧ a ≤ 12 :=
by sorry

-- Define the range of a
def a_range : Set ℝ := {a | 0 ≤ a ∧ a ≤ 12}

-- State the main theorem
theorem main_theorem : 
  {a : ℝ | ∀ x, f a x ≥ 0} = a_range :=
by sorry

end solution_set_empty_implies_a_range_main_theorem_l3370_337012


namespace megan_finished_problems_l3370_337075

theorem megan_finished_problems (total_problems : ℕ) (remaining_pages : ℕ) (problems_per_page : ℕ) 
  (h1 : total_problems = 40)
  (h2 : remaining_pages = 2)
  (h3 : problems_per_page = 7) :
  total_problems - (remaining_pages * problems_per_page) = 26 := by
sorry

end megan_finished_problems_l3370_337075


namespace dart_board_probability_l3370_337099

/-- The probability of a dart landing within the center hexagon of a dart board -/
theorem dart_board_probability (s : ℝ) (h : s > 0) : 
  let center_area := (3 * Real.sqrt 3 / 2) * s^2
  let total_area := (3 * Real.sqrt 3 / 2) * (2*s)^2
  center_area / total_area = 1/4 := by
  sorry

end dart_board_probability_l3370_337099


namespace inscribed_sphere_radius_to_height_ratio_l3370_337018

/-- A regular tetrahedron with height H and an inscribed sphere of radius R -/
structure RegularTetrahedron where
  H : ℝ
  R : ℝ
  H_pos : H > 0
  R_pos : R > 0

/-- The ratio of the radius of the inscribed sphere to the height of a regular tetrahedron is 1:4 -/
theorem inscribed_sphere_radius_to_height_ratio (t : RegularTetrahedron) : t.R / t.H = 1 / 4 := by
  sorry

end inscribed_sphere_radius_to_height_ratio_l3370_337018


namespace paint_container_rectangle_perimeter_l3370_337002

theorem paint_container_rectangle_perimeter :
  ∀ x : ℝ,
  -- Old rectangle conditions
  x > 0 →
  let old_width := x
  let old_length := 3 * x
  let old_area := old_width * old_length
  -- New rectangle conditions
  let new_width := x + 8
  let new_length := 3 * x - 18
  let new_area := new_width * new_length
  -- Equal area condition
  old_area = new_area →
  -- Perimeter calculation
  let new_perimeter := 2 * (new_width + new_length)
  -- Theorem statement
  new_perimeter = 172 :=
by
  sorry


end paint_container_rectangle_perimeter_l3370_337002


namespace value_of_expression_l3370_337096

theorem value_of_expression (x y : ℝ) 
  (h1 : x - 2*y = -5) 
  (h2 : x*y = -2) : 
  2*x^2*y - 4*x*y^2 = 20 := by
sorry

end value_of_expression_l3370_337096


namespace sin_cos_sum_10_50_l3370_337000

theorem sin_cos_sum_10_50 : 
  Real.sin (10 * π / 180) * Real.cos (50 * π / 180) + 
  Real.cos (10 * π / 180) * Real.sin (50 * π / 180) = 
  Real.sqrt 3 / 2 := by sorry

end sin_cos_sum_10_50_l3370_337000


namespace mountain_bike_pricing_l3370_337037

/-- Represents the sales and pricing of mountain bikes over three months -/
structure MountainBikeSales where
  january_sales : ℝ
  february_price_decrease : ℝ
  february_sales : ℝ
  march_price_decrease_percentage : ℝ
  march_profit_percentage : ℝ

/-- Theorem stating the selling price in February and the cost price of each mountain bike -/
theorem mountain_bike_pricing (sales : MountainBikeSales)
  (h1 : sales.january_sales = 27000)
  (h2 : sales.february_price_decrease = 100)
  (h3 : sales.february_sales = 24000)
  (h4 : sales.march_price_decrease_percentage = 0.1)
  (h5 : sales.march_profit_percentage = 0.44) :
  ∃ (february_price cost_price : ℝ),
    february_price = 800 ∧ cost_price = 500 := by
  sorry

end mountain_bike_pricing_l3370_337037


namespace f_max_value_when_a_2_f_no_min_value_when_a_2_f_decreasing_when_a_leq_neg_quarter_f_decreasing_when_neg_quarter_lt_a_leq_zero_f_monotonicity_when_a_gt_zero_l3370_337053

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x - (1/2) * x^2

-- Theorem for the maximum value when a = 2
theorem f_max_value_when_a_2 :
  ∃ (x : ℝ), x > 0 ∧ f 2 x = -(3/2) ∧ ∀ (y : ℝ), y > 0 → f 2 y ≤ f 2 x :=
sorry

-- Theorem for no minimum value when a = 2
theorem f_no_min_value_when_a_2 :
  ∀ (M : ℝ), ∃ (x : ℝ), x > 0 ∧ f 2 x < M :=
sorry

-- Theorem for monotonicity when a ≤ -1/4
theorem f_decreasing_when_a_leq_neg_quarter (a : ℝ) (h : a ≤ -(1/4)) :
  ∀ (x y : ℝ), 0 < x → 0 < y → x < y → f a x > f a y :=
sorry

-- Theorem for monotonicity when -1/4 < a ≤ 0
theorem f_decreasing_when_neg_quarter_lt_a_leq_zero (a : ℝ) (h1 : -(1/4) < a) (h2 : a ≤ 0) :
  ∀ (x y : ℝ), 0 < x → 0 < y → x < y → f a x > f a y :=
sorry

-- Theorem for monotonicity when a > 0
theorem f_monotonicity_when_a_gt_zero (a : ℝ) (h : a > 0) :
  let x0 := (-1 + Real.sqrt (1 + 4*a)) / 2
  ∀ (x y : ℝ), 0 < x → x < y → y < x0 → f a x < f a y ∧
  ∀ (x y : ℝ), x0 < x → x < y → f a x > f a y :=
sorry

end

end f_max_value_when_a_2_f_no_min_value_when_a_2_f_decreasing_when_a_leq_neg_quarter_f_decreasing_when_neg_quarter_lt_a_leq_zero_f_monotonicity_when_a_gt_zero_l3370_337053


namespace class_average_l3370_337077

theorem class_average (total_students : ℕ) (top_scorers : ℕ) (zero_scorers : ℕ) (top_score : ℕ) (rest_average : ℕ) 
  (h1 : total_students = 20)
  (h2 : top_scorers = 2)
  (h3 : zero_scorers = 3)
  (h4 : top_score = 100)
  (h5 : rest_average = 40) :
  (top_scorers * top_score + zero_scorers * 0 + (total_students - top_scorers - zero_scorers) * rest_average) / total_students = 40 := by
  sorry

#check class_average

end class_average_l3370_337077


namespace f_satisfies_condition_l3370_337071

-- Define the function f
def f (x : ℝ) : ℝ := x - 1

-- State the theorem
theorem f_satisfies_condition : ∀ x : ℝ, f x + f (2 - x) = 0 := by
  sorry

end f_satisfies_condition_l3370_337071


namespace count_green_curlers_l3370_337064

/-- Given a total number of curlers and relationships between different types,
    prove the number of large green curlers. -/
theorem count_green_curlers (total : ℕ) (pink : ℕ) (blue : ℕ) (green : ℕ)
  (h1 : total = 16)
  (h2 : pink = total / 4)
  (h3 : blue = 2 * pink)
  (h4 : green = total - pink - blue) :
  green = 4 := by
  sorry

end count_green_curlers_l3370_337064


namespace nineteen_in_base_three_l3370_337042

theorem nineteen_in_base_three : 
  (2 * 3^2 + 0 * 3^1 + 1 * 3^0) = 19 := by
  sorry

end nineteen_in_base_three_l3370_337042


namespace prob_A_value_l3370_337068

/-- The probability of producing a grade B product -/
def prob_B : ℝ := 0.05

/-- The probability of producing a grade C product -/
def prob_C : ℝ := 0.03

/-- The probability of a randomly inspected product being grade A (non-defective) -/
def prob_A : ℝ := 1 - prob_B - prob_C

theorem prob_A_value : prob_A = 0.92 := by
  sorry

end prob_A_value_l3370_337068


namespace james_injury_timeline_l3370_337047

/-- The number of days it took for James's pain to subside -/
def pain_subsided_days : ℕ := 3

/-- The total number of days until James can lift heavy again -/
def total_days : ℕ := 39

/-- The number of additional days James waits after the injury is fully healed -/
def additional_waiting_days : ℕ := 3

/-- The number of days (3 weeks) James waits before lifting heavy -/
def heavy_lifting_wait_days : ℕ := 21

theorem james_injury_timeline : 
  pain_subsided_days * 5 + pain_subsided_days + additional_waiting_days + heavy_lifting_wait_days = total_days :=
by sorry

end james_injury_timeline_l3370_337047


namespace optimal_allocation_l3370_337013

/-- Represents an investment project --/
structure Project where
  maxProfitRate : ℝ
  maxLossRate : ℝ

/-- Represents an investment allocation --/
structure Allocation where
  projectA : ℝ
  projectB : ℝ

/-- Calculates the potential profit for a given allocation --/
def potentialProfit (projects : Project × Project) (alloc : Allocation) : ℝ :=
  alloc.projectA * projects.1.maxProfitRate + alloc.projectB * projects.2.maxProfitRate

/-- Calculates the potential loss for a given allocation --/
def potentialLoss (projects : Project × Project) (alloc : Allocation) : ℝ :=
  alloc.projectA * projects.1.maxLossRate + alloc.projectB * projects.2.maxLossRate

/-- Theorem: The optimal allocation maximizes profit while satisfying constraints --/
theorem optimal_allocation
  (projectA : Project)
  (projectB : Project)
  (totalLimit : ℝ)
  (lossLimit : ℝ)
  (h1 : projectA.maxProfitRate = 1)
  (h2 : projectB.maxProfitRate = 0.5)
  (h3 : projectA.maxLossRate = 0.3)
  (h4 : projectB.maxLossRate = 0.1)
  (h5 : totalLimit = 100000)
  (h6 : lossLimit = 18000) :
  ∃ (alloc : Allocation),
    alloc.projectA = 40000 ∧
    alloc.projectB = 60000 ∧
    alloc.projectA + alloc.projectB ≤ totalLimit ∧
    potentialLoss (projectA, projectB) alloc ≤ lossLimit ∧
    ∀ (otherAlloc : Allocation),
      otherAlloc.projectA + otherAlloc.projectB ≤ totalLimit →
      potentialLoss (projectA, projectB) otherAlloc ≤ lossLimit →
      potentialProfit (projectA, projectB) alloc ≥ potentialProfit (projectA, projectB) otherAlloc :=
by sorry

end optimal_allocation_l3370_337013


namespace lcm_18_20_l3370_337097

theorem lcm_18_20 : Nat.lcm 18 20 = 180 := by
  sorry

end lcm_18_20_l3370_337097


namespace inscribed_circle_radius_right_triangle_l3370_337016

/-- Given a right triangle with legs a and b, and hypotenuse c, 
    the radius r of its inscribed circle is (a + b - c) / 2 -/
theorem inscribed_circle_radius_right_triangle 
  (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) :
  ∃ r : ℝ, r > 0 ∧ r = (a + b - c) / 2 ∧ 
    r * (a + b + c) / 2 = a * b / 2 := by
  sorry


end inscribed_circle_radius_right_triangle_l3370_337016


namespace power_function_value_l3370_337065

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, x > 0 → f x = x ^ α

-- State the theorem
theorem power_function_value (f : ℝ → ℝ) :
  isPowerFunction f → f 2 = Real.sqrt 2 / 2 → f 9 = 1 / 3 := by
  sorry

end power_function_value_l3370_337065


namespace chess_tournament_games_l3370_337026

theorem chess_tournament_games (n : ℕ) (h : n = 12) :
  2 * n * (n - 1) / 2 = 264 := by
  sorry

end chess_tournament_games_l3370_337026


namespace largest_n_for_perfect_square_l3370_337074

theorem largest_n_for_perfect_square (n : ℕ) : 
  (∃ k : ℕ, 4^27 + 4^500 + 4^n = k^2) → n ≤ 972 :=
by sorry

end largest_n_for_perfect_square_l3370_337074


namespace candace_hiking_ratio_l3370_337028

/-- Candace's hiking scenario -/
def hiking_scenario (old_speed new_speed hike_duration blister_interval blister_slowdown : ℝ) : Prop :=
  let blisters := hike_duration / blister_interval
  let total_slowdown := blisters * blister_slowdown
  let final_new_speed := new_speed - total_slowdown
  final_new_speed / old_speed = 7 / 6

/-- The theorem representing Candace's hiking problem -/
theorem candace_hiking_ratio :
  hiking_scenario 6 11 4 2 2 :=
by
  sorry

end candace_hiking_ratio_l3370_337028


namespace circle_and_line_properties_l3370_337058

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y + 1 = 0

-- Define the line L
def line_L (x y : ℝ) : Prop := x - y = 2

-- Theorem statement
theorem circle_and_line_properties :
  -- The radius of circle C is 1
  (∃ (a b : ℝ), ∀ (x y : ℝ), circle_C x y ↔ (x - a)^2 + (y - b)^2 = 1) ∧
  -- The distance from the center of C to line L is √2
  (∃ (a b : ℝ), (a - b - 2) / Real.sqrt 2 = Real.sqrt 2) ∧
  -- The minimum distance from a point on C to line L is √2 - 1
  (∃ (x y : ℝ), circle_C x y ∧ 
    (∀ (x' y' : ℝ), circle_C x' y' →
      |x' - y' - 2| / Real.sqrt 2 ≥ Real.sqrt 2 - 1) ∧
    |x - y - 2| / Real.sqrt 2 = Real.sqrt 2 - 1) :=
sorry

end circle_and_line_properties_l3370_337058


namespace escalator_speed_l3370_337086

/-- Proves that the escalator speed is 12 feet per second given the conditions -/
theorem escalator_speed (escalator_length : ℝ) (person_speed : ℝ) (time_taken : ℝ) :
  escalator_length = 210 →
  person_speed = 2 →
  time_taken = 15 →
  (person_speed + (escalator_length / time_taken)) * time_taken = escalator_length →
  escalator_length / time_taken = 12 := by
  sorry


end escalator_speed_l3370_337086


namespace g_x_plus_3_l3370_337027

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x + 1

-- State the theorem
theorem g_x_plus_3 : ∀ x : ℝ, g (x + 3) = 3 * x + 10 := by
  sorry

end g_x_plus_3_l3370_337027


namespace part_I_part_II_part_III_l3370_337080

-- Define the function f
def f (m x : ℝ) : ℝ := m * x^2 + (1 - 3*m) * x + 2*m - 1

-- Part I
theorem part_I (a : ℝ) :
  (a > 0 ∧ {x : ℝ | f 2 x ≤ 0} ⊆ Set.Ioo a (2*a + 1)) ↔ (1/4 ≤ a ∧ a < 1) :=
sorry

-- Part II
def solution_set (m : ℝ) : Set ℝ :=
  if m < 0 then Set.Iic 1 ∪ Set.Ici (2 - 1/m)
  else if m = 0 then Set.Iic 1
  else if 0 < m ∧ m < 1 then Set.Icc (2 - 1/m) 1
  else if m = 1 then {1}
  else Set.Icc 1 (2 - 1/m)

theorem part_II (m : ℝ) :
  {x : ℝ | f m x ≤ 0} = solution_set m :=
sorry

-- Part III
theorem part_III (m : ℝ) :
  (∃ x > 0, f m x > -3*m*x + m - 1) ↔ m > -1/2 :=
sorry

end part_I_part_II_part_III_l3370_337080


namespace probability_not_paying_cash_l3370_337031

theorem probability_not_paying_cash (p_only_cash p_both : ℝ) 
  (h1 : p_only_cash = 0.45)
  (h2 : p_both = 0.15) : 
  1 - (p_only_cash + p_both) = 0.4 := by
sorry

end probability_not_paying_cash_l3370_337031


namespace sin_2x_value_l3370_337033

theorem sin_2x_value (x : Real) (h : Real.sin (x + π/4) = 1/4) : 
  Real.sin (2*x) = -7/8 := by
sorry

end sin_2x_value_l3370_337033


namespace arrangement_count_l3370_337072

-- Define the total number of people
def total_people : ℕ := 6

-- Define the number of people in the Jia-Bing-Yi group
def group_size : ℕ := 3

-- Define the number of units (group + other individuals)
def num_units : ℕ := total_people - group_size + 1

-- Theorem statement
theorem arrangement_count : 
  (num_units.factorial * group_size.factorial * 2) - (num_units.factorial * 2) = 240 := by
  sorry

end arrangement_count_l3370_337072


namespace cos_eight_arccos_one_fourth_l3370_337039

theorem cos_eight_arccos_one_fourth :
  Real.cos (8 * Real.arccos (1/4)) = 18593/32768 := by
  sorry

end cos_eight_arccos_one_fourth_l3370_337039


namespace mona_unique_players_l3370_337079

/-- The number of groups Mona joined -/
def num_groups : ℕ := 9

/-- The number of other players in each group -/
def players_per_group : ℕ := 4

/-- The number of repeat players in the first group with repeats -/
def repeat_players_group1 : ℕ := 2

/-- The number of repeat players in the second group with repeats -/
def repeat_players_group2 : ℕ := 1

/-- The total number of unique players Mona grouped with -/
def unique_players : ℕ := num_groups * players_per_group - (repeat_players_group1 + repeat_players_group2)

theorem mona_unique_players : unique_players = 33 := by
  sorry

end mona_unique_players_l3370_337079


namespace stratified_sampling_theorem_l3370_337041

theorem stratified_sampling_theorem (teachers male_students female_students : ℕ) 
  (female_sample : ℕ) (n : ℕ) : 
  teachers = 160 → 
  male_students = 960 → 
  female_students = 800 → 
  female_sample = 80 → 
  (female_students : ℚ) / (teachers + male_students + female_students : ℚ) = 
    (female_sample : ℚ) / (n : ℚ) → 
  n = 192 := by
  sorry

end stratified_sampling_theorem_l3370_337041


namespace marble_probability_l3370_337057

theorem marble_probability (total : ℕ) (blue red : ℕ) (h1 : total = 150) (h2 : blue = 24) (h3 : red = 37) :
  let white := total - blue - red
  (red + white : ℚ) / total = 21 / 25 := by sorry

end marble_probability_l3370_337057


namespace solve_exponential_equation_l3370_337049

theorem solve_exponential_equation :
  ∃ n : ℕ, (9 : ℝ)^n * (9 : ℝ)^n * (9 : ℝ)^n * (9 : ℝ)^n = (729 : ℝ)^4 ∧ n = 3 := by
  sorry

end solve_exponential_equation_l3370_337049


namespace area_of_closed_region_l3370_337063

-- Define the functions
def f₀ (x : ℝ) := |x|
def f₁ (x : ℝ) := |f₀ x - 1|
def f₂ (x : ℝ) := |f₁ x - 2|

-- Define the area function
noncomputable def area_under_curve (f : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, f x

-- Theorem statement
theorem area_of_closed_region :
  area_under_curve f₂ (-3) 3 = 7 := by
  sorry

end area_of_closed_region_l3370_337063


namespace apples_bought_l3370_337051

theorem apples_bought (initial : ℕ) (used : ℕ) (final : ℕ) (bought : ℕ) : 
  initial ≥ used →
  bought = final - (initial - used) := by
  sorry

end apples_bought_l3370_337051


namespace min_value_expression_l3370_337029

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / (a + 1) + 4 / (b + 1) ≥ 9 / 4 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + b₀ = 2 ∧ 1 / (a₀ + 1) + 4 / (b₀ + 1) = 9 / 4 :=
by sorry

end min_value_expression_l3370_337029


namespace system_solution_proof_l3370_337055

theorem system_solution_proof :
  ∃ (x y z : ℝ),
    (1/x + 1/(y+z) = 6/5) ∧
    (1/y + 1/(x+z) = 3/4) ∧
    (1/z + 1/(x+y) = 2/3) ∧
    (x = 2) ∧ (y = 3) ∧ (z = 1) :=
by sorry

end system_solution_proof_l3370_337055


namespace vector_problem_l3370_337052

def vector_a (m : ℝ) : ℝ × ℝ := (m, 3)
def vector_b (m : ℝ) : ℝ × ℝ := (1, m - 2)

def parallel (u v : ℝ × ℝ) : Prop :=
  u.1 * v.2 = u.2 * v.1

def perpendicular (u v : ℝ × ℝ) : Prop :=
  u.1 * v.1 + u.2 * v.2 = 0

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem vector_problem :
  (∀ m : ℝ, parallel (vector_a m) (vector_b m) → m = 3 ∨ m = -1) ∧
  (∀ m : ℝ, perpendicular (vector_a m) (vector_b m) →
    let a := vector_a m
    let b := vector_b m
    dot_product (a.1 + 2 * b.1, a.2 + 2 * b.2) (2 * a.1 - b.1, 2 * a.2 - b.2) = 20) :=
by sorry

end vector_problem_l3370_337052


namespace apple_distribution_l3370_337004

theorem apple_distribution (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 4) :
  (Nat.choose (n + k - 1) (k - 1)) = 3276 :=
sorry

end apple_distribution_l3370_337004


namespace special_sequence_third_term_l3370_337003

/-- A sequence S with special properties -/
def SpecialSequence (S : ℕ → ℝ) : Prop :=
  ∃ a : ℝ, a > 0 ∧
  S 0 = a^4 ∧
  (∀ n : ℕ, S (n + 1) = 4 * Real.sqrt (S n)) ∧
  (S 2 - S 1 = S 1 - S 0)

/-- The third term of the special sequence can only be 16 or 8√5 - 8 -/
theorem special_sequence_third_term (S : ℕ → ℝ) (h : SpecialSequence S) :
  S 2 = 16 ∨ S 2 = 8 * Real.sqrt 5 - 8 := by
  sorry

end special_sequence_third_term_l3370_337003


namespace odd_gon_symmetry_axis_through_vertex_l3370_337050

/-- A (2k+1)-gon is a polygon with 2k+1 vertices, where k is a positive integer. -/
structure OddGon where
  k : ℕ+
  vertices : Fin (2 * k + 1) → ℝ × ℝ

/-- An axis of symmetry for a polygon -/
structure SymmetryAxis (P : OddGon) where
  line : ℝ × ℝ → Prop

/-- A vertex lies on a line -/
def vertex_on_line (P : OddGon) (axis : SymmetryAxis P) (v : Fin (2 * P.k + 1)) : Prop :=
  axis.line (P.vertices v)

/-- The theorem stating that the axis of symmetry of a (2k+1)-gon passes through one of its vertices -/
theorem odd_gon_symmetry_axis_through_vertex (P : OddGon) (axis : SymmetryAxis P) :
  ∃ v : Fin (2 * P.k + 1), vertex_on_line P axis v := by
  sorry

end odd_gon_symmetry_axis_through_vertex_l3370_337050


namespace luke_game_rounds_l3370_337044

theorem luke_game_rounds (points_per_round : ℕ) (total_points : ℕ) (h1 : points_per_round = 146) (h2 : total_points = 22922) :
  total_points / points_per_round = 157 := by
  sorry

end luke_game_rounds_l3370_337044


namespace electric_sharpener_time_l3370_337089

/-- Proves that an electric pencil sharpener takes 20 seconds to sharpen one pencil -/
theorem electric_sharpener_time : ∀ (hand_crank_time electric_time : ℕ),
  hand_crank_time = 45 →
  (360 / hand_crank_time : ℚ) + 10 = 360 / electric_time →
  electric_time = 20 :=
by sorry

end electric_sharpener_time_l3370_337089


namespace basketball_games_count_l3370_337093

theorem basketball_games_count : ∃ (x : ℕ), 
  x > 0 ∧ 
  x ∣ 60 ∧ 
  (3 * x / 5 : ℚ) = ⌊(3 * x / 5 : ℚ)⌋ ∧
  (7 * (x + 10) / 12 : ℚ) = ⌊(7 * (x + 10) / 12 : ℚ)⌋ ∧
  (7 * (x + 10) / 12 : ℕ) = (3 * x / 5 : ℕ) + 5 ∧
  x = 60 := by
  sorry

end basketball_games_count_l3370_337093


namespace cube_surface_area_from_volume_l3370_337011

theorem cube_surface_area_from_volume (V : ℝ) (h : V = 64) :
  ∃ (a : ℝ), a > 0 ∧ a^3 = V ∧ 6 * a^2 = 96 := by
  sorry

end cube_surface_area_from_volume_l3370_337011


namespace trapezoid_area_in_isosceles_triangle_l3370_337066

/-- Represents a triangle in a plane -/
structure Triangle where
  area : ℝ

/-- Represents a trapezoid in a plane -/
structure Trapezoid where
  area : ℝ

/-- The main theorem statement -/
theorem trapezoid_area_in_isosceles_triangle 
  (PQR : Triangle) 
  (smallest : Triangle)
  (RSQT : Trapezoid) :
  PQR.area = 72 ∧ 
  smallest.area = 2 ∧ 
  (∃ n : ℕ, n = 9 ∧ n * smallest.area = PQR.area) ∧
  (∃ m : ℕ, m = 3 ∧ m * smallest.area ≤ RSQT.area) →
  RSQT.area = 39 := by
sorry

end trapezoid_area_in_isosceles_triangle_l3370_337066


namespace acme_vowel_soup_words_l3370_337056

/-- The number of different letters available -/
def num_letters : ℕ := 5

/-- The number of times each letter appears -/
def letter_count : ℕ := 5

/-- The length of words to be formed -/
def word_length : ℕ := 5

/-- The total number of words that can be formed -/
def total_words : ℕ := num_letters ^ word_length

theorem acme_vowel_soup_words : total_words = 3125 := by
  sorry

end acme_vowel_soup_words_l3370_337056


namespace correct_answers_is_120_l3370_337095

/-- Represents an exam scoring system -/
structure ExamScoring where
  totalScore : Int
  totalQuestions : Nat
  correctScore : Int
  wrongPenalty : Int

/-- Calculates the number of correct answers in an exam -/
def calculateCorrectAnswers (exam : ExamScoring) : Int :=
  (exam.totalScore + 2 * exam.totalQuestions) / (exam.correctScore - exam.wrongPenalty)

/-- Theorem: Given the exam conditions, the number of correct answers is 120 -/
theorem correct_answers_is_120 (exam : ExamScoring) 
  (h1 : exam.totalScore = 420)
  (h2 : exam.totalQuestions = 150)
  (h3 : exam.correctScore = 4)
  (h4 : exam.wrongPenalty = 2) :
  calculateCorrectAnswers exam = 120 := by
  sorry

#eval calculateCorrectAnswers { totalScore := 420, totalQuestions := 150, correctScore := 4, wrongPenalty := 2 }

end correct_answers_is_120_l3370_337095


namespace a_minus_b_and_c_linearly_dependent_l3370_337083

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (e₁ e₂ : V)

/-- e₁ and e₂ are not collinear -/
axiom not_collinear : ¬ ∃ (r : ℝ), e₁ = r • e₂

/-- Definition of vector a -/
def a : V := 2 • e₁ - e₂

/-- Definition of vector b -/
def b : V := e₁ + 2 • e₂

/-- Definition of vector c -/
def c : V := (1/2) • e₁ - (3/2) • e₂

/-- Theorem stating that (a - b) and c are linearly dependent -/
theorem a_minus_b_and_c_linearly_dependent :
  ∃ (r s : ℝ) (hs : s ≠ 0), r • (a e₁ e₂ - b e₁ e₂) + s • c e₁ e₂ = 0 :=
sorry

end a_minus_b_and_c_linearly_dependent_l3370_337083


namespace quadrilateral_area_l3370_337017

theorem quadrilateral_area (S_ABCD S_OKSL S_ONAM S_OMBK : ℝ) 
  (h1 : S_ABCD = 4 * (S_OKSL + S_ONAM))
  (h2 : S_OKSL = 6)
  (h3 : S_ONAM = 12)
  (h4 : S_OMBK = S_ABCD - S_OKSL - 24 - S_ONAM) :
  S_ABCD = 72 ∧ S_OMBK = 30 := by
  sorry

end quadrilateral_area_l3370_337017


namespace hyperbola_eccentricity_l3370_337092

/-- Given a hyperbola with real axis length 16 and imaginary axis length 12, its eccentricity is 5/4 -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a = 8) (h2 : b = 6) : 
  let c := Real.sqrt (a^2 + b^2)
  c / a = 5/4 := by sorry

end hyperbola_eccentricity_l3370_337092


namespace fraction_multiplication_l3370_337084

theorem fraction_multiplication (x : ℚ) : (3/4 : ℚ) * (1/2 : ℚ) * (2/5 : ℚ) * 5020 = 753 := by
  sorry

end fraction_multiplication_l3370_337084


namespace arctan_tan_difference_l3370_337010

theorem arctan_tan_difference (x y : Real) :
  Real.arctan (Real.tan (65 * π / 180) - 2 * Real.tan (40 * π / 180)) = 25 * π / 180 := by
  sorry

end arctan_tan_difference_l3370_337010


namespace rectangle_area_l3370_337060

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width^2 = square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end rectangle_area_l3370_337060


namespace scalar_for_coplanarity_l3370_337046

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define the points as vectors
variable (O A B C D : V)

-- Define the scalar k
variable (k : ℝ)

-- Define the equation
def equation (O A B C D : V) (k : ℝ) : Prop :=
  2 • (A - O) - 3 • (B - O) + 7 • (C - O) + k • (D - O) = 0

-- Define coplanarity
def coplanar (A B C D : V) : Prop :=
  ∃ (a b c : ℝ), a • (B - A) + b • (C - A) + c • (D - A) = 0 ∧ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

-- Theorem statement
theorem scalar_for_coplanarity (O A B C D : V) :
  ∃ (k : ℝ), equation O A B C D k ∧ coplanar A B C D ∧ k = -6 := by sorry

end scalar_for_coplanarity_l3370_337046


namespace birds_on_fence_l3370_337088

/-- The number of birds that fly away -/
def birds_flown : ℝ := 8.0

/-- The number of birds left on the fence -/
def birds_left : ℕ := 4

/-- The initial number of birds on the fence -/
def initial_birds : ℝ := birds_flown + birds_left

theorem birds_on_fence : initial_birds = 12.0 := by
  sorry

end birds_on_fence_l3370_337088


namespace sin_cos_equation_l3370_337069

theorem sin_cos_equation (x : Real) (p q : Nat) 
  (h1 : (1 + Real.sin x) * (1 + Real.cos x) = 9/4)
  (h2 : (1 - Real.sin x) * (1 - Real.cos x) = p - Real.sqrt q)
  (h3 : 0 < p) (h4 : 0 < q) : p + q = 1 := by
  sorry

end sin_cos_equation_l3370_337069


namespace intersection_of_M_and_N_l3370_337067

def M : Set Nat := {1, 3, 5, 7}
def N : Set Nat := {2, 5, 8}

theorem intersection_of_M_and_N : M ∩ N = {5} := by
  sorry

end intersection_of_M_and_N_l3370_337067


namespace uncle_bruce_chocolate_cookies_l3370_337087

theorem uncle_bruce_chocolate_cookies (total_dough : ℝ) (chocolate_percentage : ℝ) (leftover_chocolate : ℝ) :
  total_dough = 36 ∧ 
  chocolate_percentage = 0.20 ∧ 
  leftover_chocolate = 4 →
  ∃ initial_chocolate : ℝ,
    initial_chocolate = 13 ∧
    chocolate_percentage * (total_dough + initial_chocolate - leftover_chocolate) = initial_chocolate - leftover_chocolate :=
by sorry

end uncle_bruce_chocolate_cookies_l3370_337087


namespace noodles_given_to_william_l3370_337082

/-- The number of noodles Daniel initially had -/
def initial_noodles : ℕ := 66

/-- The number of noodles Daniel has now -/
def remaining_noodles : ℕ := 54

/-- The number of noodles Daniel gave to William -/
def noodles_given : ℕ := initial_noodles - remaining_noodles

theorem noodles_given_to_william : noodles_given = 12 := by
  sorry

end noodles_given_to_william_l3370_337082


namespace expression_evaluation_l3370_337038

theorem expression_evaluation : 
  let x : ℝ := 2
  let y : ℝ := -1
  (2*x - y)^2 + (x - 2*y) * (x + 2*y) = 25 := by sorry

end expression_evaluation_l3370_337038


namespace circle_tangent_origin_l3370_337036

/-- A circle in the xy-plane -/
structure Circle where
  G : ℝ
  E : ℝ
  F : ℝ

/-- Predicate to check if a circle is tangent to the x-axis at the origin -/
def isTangentAtOrigin (c : Circle) : Prop :=
  ∃ (x y : ℝ), x^2 + y^2 + c.G * x + c.E * y + c.F = 0 ∧
                (x = 0 ∧ y = 0) ∧
                ∀ (x' y' : ℝ), x' ≠ 0 → (x'^2 + y'^2 + c.G * x' + c.E * y' + c.F > 0)

theorem circle_tangent_origin (c : Circle) :
  isTangentAtOrigin c → c.G = 0 ∧ c.F = 0 ∧ c.E ≠ 0 := by
  sorry

end circle_tangent_origin_l3370_337036


namespace athletes_leaving_hours_l3370_337035

/-- The number of hours athletes left the camp -/
def hours_athletes_left : ℕ := 4

/-- The initial number of athletes in the camp -/
def initial_athletes : ℕ := 300

/-- The rate at which athletes left the camp (per hour) -/
def leaving_rate : ℕ := 28

/-- The rate at which new athletes entered the camp (per hour) -/
def entering_rate : ℕ := 15

/-- The number of hours new athletes entered the camp -/
def entering_hours : ℕ := 7

/-- The difference in the total number of athletes over the two nights -/
def athlete_difference : ℕ := 7

theorem athletes_leaving_hours :
  initial_athletes - (leaving_rate * hours_athletes_left) + 
  (entering_rate * entering_hours) = initial_athletes + athlete_difference :=
by sorry

end athletes_leaving_hours_l3370_337035


namespace f_difference_l3370_337023

/-- The function f(x) = 3x^2 - 4x + 2 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2

/-- Theorem: For all real x and h, f(x + h) - f(x) = h(6x + 3h - 4) -/
theorem f_difference (x h : ℝ) : f (x + h) - f x = h * (6 * x + 3 * h - 4) := by
  sorry

end f_difference_l3370_337023


namespace trig_expression_equals_zero_l3370_337054

theorem trig_expression_equals_zero :
  Real.cos (π / 3) - Real.tan (π / 4) + (3 / 4) * (Real.tan (π / 6))^2 - Real.sin (π / 6) + (Real.cos (π / 6))^2 = 0 := by
  sorry

end trig_expression_equals_zero_l3370_337054


namespace average_book_width_l3370_337006

def book_widths : List ℝ := [7.5, 3, 0.75, 4, 1.25, 12]

theorem average_book_width : 
  (List.sum book_widths) / (List.length book_widths) = 4.75 := by
  sorry

end average_book_width_l3370_337006


namespace ratio_a_to_c_l3370_337007

theorem ratio_a_to_c (a b c d : ℚ) 
  (hab : a / b = 5 / 2)
  (hcd : c / d = 4 / 1)
  (hdb : d / b = 1 / 4) :
  a / c = 5 / 2 := by
sorry

end ratio_a_to_c_l3370_337007


namespace perfect_square_condition_l3370_337034

theorem perfect_square_condition (a b : ℤ) : 
  (∀ m n : ℕ, ∃ k : ℕ, a * m^2 + b * n^2 = k^2) → a * b = 0 := by
  sorry

end perfect_square_condition_l3370_337034


namespace fiftieth_student_age_l3370_337045

theorem fiftieth_student_age
  (total_students : Nat)
  (average_age : ℝ)
  (group1_count : Nat)
  (group1_avg : ℝ)
  (group2_count : Nat)
  (group2_avg : ℝ)
  (group3_count : Nat)
  (group3_avg : ℝ)
  (group4_count : Nat)
  (group4_avg : ℝ)
  (h1 : total_students = 50)
  (h2 : average_age = 20)
  (h3 : group1_count = 15)
  (h4 : group1_avg = 18)
  (h5 : group2_count = 15)
  (h6 : group2_avg = 22)
  (h7 : group3_count = 10)
  (h8 : group3_avg = 25)
  (h9 : group4_count = 9)
  (h10 : group4_avg = 24)
  (h11 : group1_count + group2_count + group3_count + group4_count = total_students - 1) :
  (total_students : ℝ) * average_age - 
  (group1_count : ℝ) * group1_avg - 
  (group2_count : ℝ) * group2_avg - 
  (group3_count : ℝ) * group3_avg - 
  (group4_count : ℝ) * group4_avg = 66 := by
  sorry

end fiftieth_student_age_l3370_337045


namespace course_selection_theorem_l3370_337008

theorem course_selection_theorem (total_courses : Nat) (conflicting_courses : Nat) 
  (courses_to_choose : Nat) (h1 : total_courses = 6) (h2 : conflicting_courses = 2) 
  (h3 : courses_to_choose = 2) :
  (Nat.choose (total_courses - conflicting_courses) courses_to_choose + 
   conflicting_courses * Nat.choose (total_courses - conflicting_courses) (courses_to_choose - 1)) = 14 := by
  sorry

end course_selection_theorem_l3370_337008


namespace arithmetic_mean_of_fractions_l3370_337005

theorem arithmetic_mean_of_fractions :
  let a := (3 : ℚ) / 4
  let b := (5 : ℚ) / 8
  let c := (7 : ℚ) / 9
  (a + b + c) / 3 = 155 / 216 := by sorry

end arithmetic_mean_of_fractions_l3370_337005
