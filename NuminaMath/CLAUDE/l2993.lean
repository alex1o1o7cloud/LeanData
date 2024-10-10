import Mathlib

namespace figure_circumference_value_l2993_299387

/-- The circumference of a figure formed by one large semicircular arc and 8 identical small semicircular arcs -/
def figure_circumference (d : ℝ) (π : ℝ) : ℝ :=
  π * d

/-- Theorem stating that the circumference of the described figure is 75.36 -/
theorem figure_circumference_value :
  let d : ℝ := 24
  let π : ℝ := 3.14
  figure_circumference d π = 75.36 := by sorry

end figure_circumference_value_l2993_299387


namespace problem_2011_l2993_299301

theorem problem_2011 : (2011^2 + 2011) / 2011 = 2012 := by
  sorry

end problem_2011_l2993_299301


namespace circle_motion_speeds_l2993_299369

/-- Represents the state of two circles moving towards the vertex of a right angle -/
structure CircleMotion where
  r1 : ℝ  -- radius of first circle
  r2 : ℝ  -- radius of second circle
  d1 : ℝ  -- initial distance of first circle from vertex
  d2 : ℝ  -- initial distance of second circle from vertex
  t_external : ℝ  -- time when circles touch externally
  t_internal : ℝ  -- time when circles touch internally

/-- Represents a pair of speeds for the two circles -/
structure SpeedPair where
  s1 : ℝ  -- speed of first circle
  s2 : ℝ  -- speed of second circle

/-- Checks if a given speed pair satisfies the conditions for the circle motion -/
def satisfiesConditions (cm : CircleMotion) (sp : SpeedPair) : Prop :=
  let d1_external := cm.d1 - sp.s1 * cm.t_external
  let d2_external := cm.d2 - sp.s2 * cm.t_external
  let d1_internal := cm.d1 - sp.s1 * cm.t_internal
  let d2_internal := cm.d2 - sp.s2 * cm.t_internal
  d1_external^2 + d2_external^2 = (cm.r1 + cm.r2)^2 ∧
  d1_internal^2 + d2_internal^2 = (cm.r1 - cm.r2)^2

/-- The main theorem stating that given the conditions, only two speed pairs satisfy the motion -/
theorem circle_motion_speeds (cm : CircleMotion)
  (h_r1 : cm.r1 = 9)
  (h_r2 : cm.r2 = 4)
  (h_d1 : cm.d1 = 48)
  (h_d2 : cm.d2 = 14)
  (h_t_external : cm.t_external = 9)
  (h_t_internal : cm.t_internal = 11) :
  ∃ (sp1 sp2 : SpeedPair),
    satisfiesConditions cm sp1 ∧
    satisfiesConditions cm sp2 ∧
    ((sp1.s1 = 4 ∧ sp1.s2 = 1) ∨ (sp1.s1 = 3.9104 ∧ sp1.s2 = 1.3072)) ∧
    ((sp2.s1 = 4 ∧ sp2.s2 = 1) ∨ (sp2.s1 = 3.9104 ∧ sp2.s2 = 1.3072)) ∧
    sp1 ≠ sp2 ∧
    ∀ (sp : SpeedPair), satisfiesConditions cm sp → (sp = sp1 ∨ sp = sp2) := by
  sorry

end circle_motion_speeds_l2993_299369


namespace ellipse_and_line_equation_l2993_299396

-- Define the ellipse C₁
def C₁ (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the parabola C₂
def C₂ (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the point M
def M (x y : ℝ) : Prop := C₁ 2 (Real.sqrt 3) x y ∧ C₂ x y ∧ x > 0 ∧ y > 0

-- Define the distance between M and F₂
def MF₂_distance (x y : ℝ) : Prop := (x - 1)^2 + y^2 = (5/3)^2

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := y = Real.sqrt 6 * (x - m)

-- Define the perpendicularity condition
def perpendicular_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

theorem ellipse_and_line_equation :
  ∃ (x y : ℝ),
    M x y ∧ MF₂_distance x y ∧
    (∀ (m : ℝ),
      (∃ (x₁ y₁ x₂ y₂ : ℝ),
        C₁ 2 (Real.sqrt 3) x₁ y₁ ∧ C₁ 2 (Real.sqrt 3) x₂ y₂ ∧
        line_l m x₁ y₁ ∧ line_l m x₂ y₂ ∧
        perpendicular_condition x₁ y₁ x₂ y₂) →
      m = Real.sqrt 2 ∨ m = -Real.sqrt 2) :=
sorry

end ellipse_and_line_equation_l2993_299396


namespace student_ticket_price_l2993_299338

/-- Calculates the price of a student ticket given the total number of tickets sold,
    the total amount collected, the price of an adult ticket, and the number of student tickets sold. -/
theorem student_ticket_price
  (total_tickets : ℕ)
  (total_amount : ℚ)
  (adult_price : ℚ)
  (student_tickets : ℕ)
  (h1 : total_tickets = 59)
  (h2 : total_amount = 222.5)
  (h3 : adult_price = 4)
  (h4 : student_tickets = 9) :
  (total_amount - (adult_price * (total_tickets - student_tickets))) / student_tickets = 2.5 := by
  sorry

end student_ticket_price_l2993_299338


namespace product_sum_theorem_l2993_299331

theorem product_sum_theorem (W F : ℕ) (c : ℕ) : 
  W > 20 → F > 20 → W * F = 770 → W + F = c → c = 57 := by
  sorry

end product_sum_theorem_l2993_299331


namespace vector_difference_magnitude_l2993_299397

theorem vector_difference_magnitude (x : ℝ) : 
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (x, 4)
  (a.1 * b.1 + a.2 * b.2 = 10) → 
  Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = Real.sqrt 5 := by
  sorry

end vector_difference_magnitude_l2993_299397


namespace calculate_F_l2993_299348

-- Define the functions f and F
def f (a : ℝ) : ℝ := a^2 - 5*a + 6
def F (a b c : ℝ) : ℝ := b^2 + a*c + 1

-- State the theorem
theorem calculate_F : F 3 (f 3) (f 5) = 19 := by
  sorry

end calculate_F_l2993_299348


namespace triangle_collinearity_l2993_299379

-- Define the triangle ABC
variable (A B C : ℝ × ℝ)

-- Define the orthocenter H
variable (H : ℝ × ℝ)

-- Define points M and N
variable (M N : ℝ × ℝ)

-- Define the circumcenter O of triangle HMN
variable (O : ℝ × ℝ)

-- Define point D
variable (D : ℝ × ℝ)

-- Define the conditions
def is_acute_triangle (A B C : ℝ × ℝ) : Prop := sorry

def angle_A_greater_than_60 (A B C : ℝ × ℝ) : Prop := sorry

def is_orthocenter (H A B C : ℝ × ℝ) : Prop := sorry

def on_side (M A B : ℝ × ℝ) : Prop := sorry

def angle_equals_60 (H M B : ℝ × ℝ) : Prop := sorry

def is_circumcenter (O H M N : ℝ × ℝ) : Prop := sorry

def forms_equilateral_triangle (D B C : ℝ × ℝ) : Prop := sorry

def same_side_as_A (D A B C : ℝ × ℝ) : Prop := sorry

def are_collinear (H O D : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem triangle_collinearity 
  (h_acute : is_acute_triangle A B C)
  (h_angle_A : angle_A_greater_than_60 A B C)
  (h_orthocenter : is_orthocenter H A B C)
  (h_M_on_AB : on_side M A B)
  (h_N_on_AC : on_side N A C)
  (h_angle_HMB : angle_equals_60 H M B)
  (h_angle_HNC : angle_equals_60 H N C)
  (h_circumcenter : is_circumcenter O H M N)
  (h_equilateral : forms_equilateral_triangle D B C)
  (h_same_side : same_side_as_A D A B C) :
  are_collinear H O D :=
sorry

end triangle_collinearity_l2993_299379


namespace concentric_circles_area_l2993_299334

theorem concentric_circles_area (R r : ℝ) (h1 : R > r) (h2 : r > 0) 
  (h3 : R^2 - r^2 = 2500) : 
  π * (R^2 - r^2) = 2500 * π := by
  sorry

#check concentric_circles_area

end concentric_circles_area_l2993_299334


namespace at_least_one_greater_than_one_l2993_299314

theorem at_least_one_greater_than_one (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b > 1) :
  a > 1 ∨ b > 1 := by
  sorry

end at_least_one_greater_than_one_l2993_299314


namespace mean_is_seven_l2993_299306

def pull_up_data : List (ℕ × ℕ) := [(9, 2), (8, 3), (6, 3), (5, 2)]

def total_students : ℕ := 10

theorem mean_is_seven :
  let sum := (pull_up_data.map (λ p => p.1 * p.2)).sum
  sum / total_students = 7 := by sorry

end mean_is_seven_l2993_299306


namespace sticker_pages_l2993_299391

theorem sticker_pages (stickers_per_page : ℕ) (total_stickers : ℕ) (pages : ℕ) : 
  stickers_per_page = 10 →
  total_stickers = 220 →
  pages * stickers_per_page = total_stickers →
  pages = 22 := by
sorry

end sticker_pages_l2993_299391


namespace select_teachers_eq_140_l2993_299395

/-- The number of ways to select 6 out of 10 teachers, where two specific teachers cannot be selected together -/
def select_teachers : ℕ :=
  let total_teachers : ℕ := 10
  let teachers_to_invite : ℕ := 6
  let remaining_teachers : ℕ := 8  -- Excluding A and B
  let case1 : ℕ := 2 * Nat.choose remaining_teachers (teachers_to_invite - 1)
  let case2 : ℕ := Nat.choose remaining_teachers teachers_to_invite
  case1 + case2

theorem select_teachers_eq_140 : select_teachers = 140 := by
  sorry

end select_teachers_eq_140_l2993_299395


namespace average_weight_decrease_l2993_299349

theorem average_weight_decrease (n : ℕ) (old_weight new_weight : ℝ) :
  n = 6 →
  old_weight = 80 →
  new_weight = 62 →
  (old_weight - new_weight) / n = 3 :=
by
  sorry

end average_weight_decrease_l2993_299349


namespace number_333_less_than_600_l2993_299341

theorem number_333_less_than_600 : 600 - 333 = 267 := by sorry

end number_333_less_than_600_l2993_299341


namespace nested_inequality_l2993_299336

/-- A function is ascendant if it preserves order -/
def Ascendant (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem nested_inequality (f g φ : ℝ → ℝ)
    (hf : Ascendant f) (hg : Ascendant g) (hφ : Ascendant φ)
    (h : ∀ x, f x ≤ g x ∧ g x ≤ φ x) :
    ∀ x, f (f x) ≤ g (g x) ∧ g (g x) ≤ φ (φ x) := by
  sorry

end nested_inequality_l2993_299336


namespace john_weekly_production_l2993_299319

/-- The number of widgets John makes per week -/
def widgets_per_week (widgets_per_hour : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  widgets_per_hour * hours_per_day * days_per_week

/-- Theorem stating that John makes 800 widgets per week -/
theorem john_weekly_production : 
  widgets_per_week 20 8 5 = 800 := by
  sorry

end john_weekly_production_l2993_299319


namespace complement_A_intersect_B_l2993_299333

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3}

-- Define set A
def A : Set Nat := {0, 1}

-- Define set B
def B : Set Nat := {1, 2, 3}

-- Theorem statement
theorem complement_A_intersect_B :
  (U \ A) ∩ B = {2, 3} := by
  sorry

end complement_A_intersect_B_l2993_299333


namespace least_common_multiple_first_ten_l2993_299329

theorem least_common_multiple_first_ten : ∃ n : ℕ, n > 0 ∧ 
  (∀ k : ℕ, k > 0 ∧ k ≤ 10 → k ∣ n) ∧
  (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, k > 0 ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧
  n = 2520 := by
sorry

end least_common_multiple_first_ten_l2993_299329


namespace invalid_votes_percentage_l2993_299317

theorem invalid_votes_percentage
  (total_votes : ℕ)
  (winner_percentage : ℚ)
  (loser_votes : ℕ)
  (h_total : total_votes = 7000)
  (h_winner : winner_percentage = 55 / 100)
  (h_loser : loser_votes = 2520) :
  (total_votes - (loser_votes / (1 - winner_percentage))) / total_votes = 1 / 5 := by
  sorry

end invalid_votes_percentage_l2993_299317


namespace amoeba_count_after_week_l2993_299308

/-- The number of amoebas after a given number of days -/
def amoeba_count (days : ℕ) : ℕ :=
  3^days

/-- The theorem stating that after 7 days, there will be 2187 amoebas -/
theorem amoeba_count_after_week : amoeba_count 7 = 2187 := by
  sorry

end amoeba_count_after_week_l2993_299308


namespace fraction_simplification_l2993_299323

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3) = 1 / 39 := by
  sorry

end fraction_simplification_l2993_299323


namespace largest_increase_1993_l2993_299378

/-- Profit margin percentages for each year from 1990 to 1999 -/
def profitMargins : Fin 10 → ℝ
  | 0 => 10
  | 1 => 20
  | 2 => 30
  | 3 => 60
  | 4 => 70
  | 5 => 75
  | 6 => 80
  | 7 => 82
  | 8 => 86
  | 9 => 70

/-- Calculate the percentage increase between two years -/
def percentageIncrease (year1 year2 : Fin 10) : ℝ :=
  profitMargins year2 - profitMargins year1

/-- The year with the largest percentage increase -/
def yearWithLargestIncrease : Fin 10 :=
  3  -- Representing 1993 (index 3 corresponds to 1993)

/-- Theorem stating that 1993 (index 3) has the largest percentage increase -/
theorem largest_increase_1993 :
  ∀ (year : Fin 9), percentageIncrease year (year + 1) ≤ percentageIncrease 2 3 :=
sorry

end largest_increase_1993_l2993_299378


namespace karens_round_trip_distance_l2993_299330

/-- The total distance Karen covers for a round trip to the library -/
def total_distance (shelves : ℕ) (books_per_shelf : ℕ) : ℕ :=
  2 * (shelves * books_per_shelf)

/-- Proof that Karen's round trip distance is 3200 miles -/
theorem karens_round_trip_distance :
  total_distance 4 400 = 3200 :=
by
  -- The proof goes here
  sorry

end karens_round_trip_distance_l2993_299330


namespace segment_division_l2993_299305

theorem segment_division (AB : ℝ) (n : ℕ) (h : n > 1) :
  ∃ E : ℝ, (E = AB / (n^2 + 1) ∨ E = AB / (n^2 - 1)) ∧ 0 ≤ E ∧ E ≤ AB :=
sorry

end segment_division_l2993_299305


namespace ab_gt_b_squared_l2993_299320

theorem ab_gt_b_squared {a b : ℝ} (h1 : a < b) (h2 : b < 0) : a * b > b ^ 2 := by
  sorry

end ab_gt_b_squared_l2993_299320


namespace problem_solution_l2993_299374

theorem problem_solution (p q r s : ℕ+) 
  (h1 : p^3 = q^2) 
  (h2 : r^4 = s^3) 
  (h3 : r - p = 17) : 
  s - q = 73 := by
  sorry

end problem_solution_l2993_299374


namespace correct_payments_l2993_299375

/-- Represents the weekly payments to three employees --/
structure EmployeePayments where
  total : ℕ
  a : ℕ
  b : ℕ
  c : ℕ

/-- Checks if the given payments satisfy the problem conditions --/
def isValidPayment (p : EmployeePayments) : Prop :=
  p.total = 1500 ∧
  p.a = (150 * p.b) / 100 ∧
  p.c = (80 * p.b) / 100 ∧
  p.a + p.b + p.c = p.total

/-- The theorem stating the correct payments --/
theorem correct_payments :
  ∃ (p : EmployeePayments), isValidPayment p ∧ p.a = 682 ∧ p.b = 454 ∧ p.c = 364 :=
by
  sorry

end correct_payments_l2993_299375


namespace box_surface_area_l2993_299366

/-- Calculates the surface area of the interior of a box formed by removing squares from corners of a rectangular sheet --/
def interior_surface_area (sheet_length : ℕ) (sheet_width : ℕ) (corner_size : ℕ) : ℕ :=
  let original_area := sheet_length * sheet_width
  let corner_area := corner_size * corner_size
  let total_removed_area := 4 * corner_area
  original_area - total_removed_area

/-- Theorem stating that the surface area of the interior of the box is 1379 square units --/
theorem box_surface_area :
  interior_surface_area 35 45 7 = 1379 :=
by sorry

end box_surface_area_l2993_299366


namespace arrangements_five_not_adjacent_l2993_299315

/-- The number of permutations of n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of ways to arrange n distinct objects in a line, 
    where two specific objects are not adjacent -/
def arrangements_not_adjacent (n : ℕ) : ℕ :=
  factorial n - 2 * factorial (n - 1)

theorem arrangements_five_not_adjacent :
  arrangements_not_adjacent 5 = 72 := by
  sorry

#eval arrangements_not_adjacent 5

end arrangements_five_not_adjacent_l2993_299315


namespace least_multiplier_for_72_to_be_multiple_of_112_l2993_299380

theorem least_multiplier_for_72_to_be_multiple_of_112 :
  (∃ n : ℕ+, (72 * n : ℕ) % 112 = 0 ∧ ∀ m : ℕ+, m < n → (72 * m : ℕ) % 112 ≠ 0) ∧
  (∃ n : ℕ+, n = 14 ∧ (72 * n : ℕ) % 112 = 0 ∧ ∀ m : ℕ+, m < n → (72 * m : ℕ) % 112 ≠ 0) :=
by sorry

end least_multiplier_for_72_to_be_multiple_of_112_l2993_299380


namespace parabola_translation_l2993_299359

def parabola1 (x : ℝ) := -(x - 1)^2 + 3
def parabola2 (x : ℝ) := -x^2

def translation (f : ℝ → ℝ) (h k : ℝ) : ℝ → ℝ :=
  λ x => f (x + h) + k

theorem parabola_translation :
  ∃ h k : ℝ, (∀ x : ℝ, translation parabola1 h k x = parabola2 x) ∧ h = 1 ∧ k = -3 :=
sorry

end parabola_translation_l2993_299359


namespace function_property_l2993_299303

/-- Given two functions f and g defined on ℝ satisfying certain properties, 
    prove that g(1) + g(-1) = 1 -/
theorem function_property (f g : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x - y) = f x * g y - g x * f y)
  (h2 : f 1 = f 2)
  (h3 : f 1 ≠ 0) : 
  g 1 + g (-1) = 1 := by
  sorry

end function_property_l2993_299303


namespace two_year_inflation_rate_real_yield_bank_deposit_l2993_299318

-- Define the annual inflation rate
def annual_inflation_rate : ℝ := 0.015

-- Define the nominal annual yield of the bank deposit
def nominal_annual_yield : ℝ := 0.07

-- Theorem for two-year inflation rate
theorem two_year_inflation_rate :
  ((1 + annual_inflation_rate)^2 - 1) * 100 = 3.0225 := by sorry

-- Theorem for real yield of bank deposit
theorem real_yield_bank_deposit :
  ((1 + nominal_annual_yield)^2 / (1 + ((1 + annual_inflation_rate)^2 - 1)) - 1) * 100 = 11.13 := by sorry

end two_year_inflation_rate_real_yield_bank_deposit_l2993_299318


namespace three_numbers_equation_l2993_299340

theorem three_numbers_equation (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : x^2 - y^2 = y*z) (eq2 : y^2 - z^2 = x*z) :
  x^2 - z^2 = x*y :=
by
  sorry

end three_numbers_equation_l2993_299340


namespace symmetric_complex_product_l2993_299309

theorem symmetric_complex_product :
  ∀ (z₁ z₂ : ℂ),
  z₁ = 1 + Complex.I →
  Complex.re z₂ = -Complex.re z₁ →
  Complex.im z₂ = Complex.im z₁ →
  z₁ * z₂ = -2 := by
sorry

end symmetric_complex_product_l2993_299309


namespace solution_set_inequality_l2993_299383

theorem solution_set_inequality (x : ℝ) : (x - 2)^2 ≤ 2*x + 11 ↔ x ∈ Set.Icc (-1) 7 := by
  sorry

end solution_set_inequality_l2993_299383


namespace kats_strength_training_time_l2993_299325

/-- Given Kat's training schedule, prove that she spends 1 hour on strength training each session -/
theorem kats_strength_training_time (
  strength_sessions : ℕ) 
  (boxing_sessions : ℕ) 
  (boxing_hours_per_session : ℚ)
  (total_weekly_hours : ℕ) 
  (h1 : strength_sessions = 3)
  (h2 : boxing_sessions = 4)
  (h3 : boxing_hours_per_session = 3/2)
  (h4 : total_weekly_hours = 9) :
  (total_weekly_hours - boxing_sessions * boxing_hours_per_session) / strength_sessions = 1 := by
  sorry

end kats_strength_training_time_l2993_299325


namespace tangent_product_approximation_l2993_299352

theorem tangent_product_approximation :
  let A : Real := 30 * π / 180
  let B : Real := 40 * π / 180
  ∃ ε > 0, |(1 + Real.tan A) * (1 + Real.tan B) - 2.9| < ε :=
by
  sorry

end tangent_product_approximation_l2993_299352


namespace xiao_jun_travel_box_probability_l2993_299302

-- Define the number of digits in the password
def password_length : ℕ := 6

-- Define the number of possible digits (0-9)
def possible_digits : ℕ := 10

-- Define the probability of guessing the correct last digit
def probability_of_success : ℚ := 1 / possible_digits

-- Theorem statement
theorem xiao_jun_travel_box_probability :
  probability_of_success = 1 / 10 :=
sorry

end xiao_jun_travel_box_probability_l2993_299302


namespace divisibility_of_fifth_power_differences_l2993_299313

theorem divisibility_of_fifth_power_differences (x y z : ℤ) 
  (h_distinct_xy : x ≠ y) (h_distinct_yz : y ≠ z) (h_distinct_zx : z ≠ x) :
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = 5 * k * (x - y) * (y - z) * (z - x) := by
  sorry

end divisibility_of_fifth_power_differences_l2993_299313


namespace base_10_to_base_8_conversion_l2993_299346

theorem base_10_to_base_8_conversion : 
  (2 * 8^3 + 0 * 8^2 + 0 * 8^1 + 0 * 8^0 : ℕ) = 1024 := by
  sorry

end base_10_to_base_8_conversion_l2993_299346


namespace sixth_root_of_68968845601_l2993_299354

theorem sixth_root_of_68968845601 :
  51^6 = 68968845601 := by
  sorry

end sixth_root_of_68968845601_l2993_299354


namespace finish_in_sixteen_days_l2993_299357

/-- Represents Jack's reading pattern and book information -/
structure ReadingPattern where
  totalPages : Nat
  weekdayPages : Nat
  weekendPages : Nat
  weekdaySkip : Nat
  weekendSkip : Nat

/-- Calculates the number of days it takes to read the book -/
def daysToFinish (pattern : ReadingPattern) : Nat :=
  sorry

/-- Theorem stating that it takes 16 days to finish the book with the given reading pattern -/
theorem finish_in_sixteen_days :
  daysToFinish { totalPages := 285
                , weekdayPages := 23
                , weekendPages := 35
                , weekdaySkip := 3
                , weekendSkip := 2 } = 16 := by
  sorry

end finish_in_sixteen_days_l2993_299357


namespace solution_implies_a_value_l2993_299351

theorem solution_implies_a_value (a : ℝ) : (2 * 2 - a = 0) → a = 4 := by
  sorry

end solution_implies_a_value_l2993_299351


namespace part_one_part_two_part_three_l2993_299394

-- Define the function y
def y (a b x : ℝ) : ℝ := a * x^2 + x - b

-- Part 1
theorem part_one (a : ℝ) :
  (∃! x, y a 1 x = 0) → (a = -1/4 ∨ a = 0) :=
sorry

-- Part 2
theorem part_two (a b x : ℝ) :
  y a b x < (a-1) * x^2 + (b+2) * x - 2*b ↔
    (b < 1 ∧ b < x ∧ x < 1) ∨
    (b > 1 ∧ 1 < x ∧ x < b) :=
sorry

-- Part 3
theorem part_three (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  (∀ t > 0, ∃ x, y a b x > 0 ∧ -2-t < x ∧ x < -2+t) →
  (∃ m, m = 1/a - 1/b ∧ ∀ a' b', a' > 0 → b' > 1 →
    (∀ t > 0, ∃ x, y a' b' x > 0 ∧ -2-t < x ∧ x < -2+t) →
    1/a' - 1/b' ≤ m) ∧
  m = 1/2 :=
sorry

end part_one_part_two_part_three_l2993_299394


namespace sum_of_squares_l2993_299337

theorem sum_of_squares (a b c x y z : ℝ) 
  (h1 : x/a + y/b + z/c = 4)
  (h2 : a/x + b/y + c/z = 2) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 12 := by
  sorry

end sum_of_squares_l2993_299337


namespace fair_coin_prob_diff_l2993_299321

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ k * (1 / 2) ^ (n - k)

/-- The probability of getting exactly 3 heads in 4 flips of a fair coin -/
def prob_3_heads : ℚ := prob_k_heads 4 3

/-- The probability of getting 4 heads in 4 flips of a fair coin -/
def prob_4_heads : ℚ := prob_k_heads 4 4

/-- The positive difference between the probability of exactly 3 heads
    and the probability of 4 heads in 4 flips of a fair coin -/
theorem fair_coin_prob_diff : prob_3_heads - prob_4_heads = 7 / 16 := by
  sorry

end fair_coin_prob_diff_l2993_299321


namespace sum_first_three_terms_l2993_299355

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  /-- The common difference between consecutive terms -/
  d : ℝ
  /-- The first term of the sequence -/
  a : ℝ
  /-- The eighth term of the sequence is 20 -/
  eighth_term : a + 7 * d = 20
  /-- The common difference is 2 -/
  diff_is_two : d = 2

/-- The sum of the first three terms of the arithmetic sequence is 24 -/
theorem sum_first_three_terms (seq : ArithmeticSequence) :
  seq.a + (seq.a + seq.d) + (seq.a + 2 * seq.d) = 24 := by
  sorry


end sum_first_three_terms_l2993_299355


namespace mr_a_loss_l2993_299327

/-- Represents the house transaction between Mr. A and Mr. B -/
def house_transaction (initial_value : ℝ) (loss_percent : ℝ) (rent : ℝ) (gain_percent : ℝ) : ℝ :=
  let sale_price := initial_value * (1 - loss_percent)
  let repurchase_price := sale_price * (1 + gain_percent)
  repurchase_price - initial_value

/-- Theorem stating that Mr. A loses $144 in the transaction -/
theorem mr_a_loss :
  house_transaction 12000 0.12 1000 0.15 = 144 := by
  sorry

end mr_a_loss_l2993_299327


namespace line_slope_proof_l2993_299307

theorem line_slope_proof (x y : ℝ) :
  x + Real.sqrt 3 * y - 2 = 0 →
  ∃ (α : ℝ), α ∈ Set.Icc 0 π ∧ Real.tan α = -Real.sqrt 3 / 3 ∧ α = 5 * π / 6 :=
by sorry

end line_slope_proof_l2993_299307


namespace hat_guessing_strategy_exists_l2993_299390

/-- Represents a strategy for guessing hat numbers -/
def Strategy := (ι : Fin 2023 → ℕ) → Fin 2023 → ℕ

/-- Theorem stating that there exists a winning strategy for the hat guessing game -/
theorem hat_guessing_strategy_exists :
  ∃ (s : Strategy),
    ∀ (ι : Fin 2023 → ℕ),
      (∀ i, 1 ≤ ι i ∧ ι i ≤ 2023) →
      ∃ i, s ι i = ι i :=
sorry

end hat_guessing_strategy_exists_l2993_299390


namespace brothers_age_equation_l2993_299373

theorem brothers_age_equation (x : ℝ) (h1 : x > 0) : 
  (x - 6) + (2*x - 6) = 15 :=
by
  sorry

#check brothers_age_equation

end brothers_age_equation_l2993_299373


namespace cat_weight_l2993_299304

theorem cat_weight (num_puppies num_cats : ℕ) (puppy_weight : ℝ) (weight_difference : ℝ) :
  num_puppies = 4 →
  num_cats = 14 →
  puppy_weight = 7.5 →
  weight_difference = 5 →
  puppy_weight + weight_difference = 12.5 :=
by sorry

end cat_weight_l2993_299304


namespace infinite_parallel_lines_l2993_299364

/-- A plane in 3D space -/
structure Plane3D where
  -- Define the plane (implementation details omitted)

/-- A point in 3D space -/
structure Point3D where
  -- Define the point (implementation details omitted)

/-- A line in 3D space -/
structure Line3D where
  -- Define the line (implementation details omitted)

/-- Predicate to check if a point is not on a plane -/
def notOnPlane (p : Point3D) (plane : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a line is parallel to a plane -/
def isParallelToPlane (l : Line3D) (plane : Plane3D) : Prop :=
  sorry

/-- Predicate to check if a line passes through a point -/
def passesThroughPoint (l : Line3D) (p : Point3D) : Prop :=
  sorry

/-- The main theorem -/
theorem infinite_parallel_lines
  (plane : Plane3D) (p : Point3D) (h : notOnPlane p plane) :
  ∃ (s : Set Line3D), (∀ l ∈ s, isParallelToPlane l plane ∧ passesThroughPoint l p) ∧ Set.Infinite s :=
sorry

end infinite_parallel_lines_l2993_299364


namespace inequality_condition_not_sufficient_l2993_299360

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x + 1 > 0) → 0 ≤ a ∧ a < 4 :=
sorry

theorem not_sufficient : 
  ∃ a : ℝ, 0 ≤ a ∧ a < 4 ∧ ∃ x : ℝ, a * x^2 - a * x + 1 ≤ 0 :=
sorry

end inequality_condition_not_sufficient_l2993_299360


namespace downstream_speed_l2993_299377

/-- Calculates the downstream speed of a rower given upstream and still water speeds -/
theorem downstream_speed (upstream_speed still_water_speed : ℝ) :
  upstream_speed = 20 →
  still_water_speed = 24 →
  still_water_speed + (still_water_speed - upstream_speed) = 28 := by
  sorry


end downstream_speed_l2993_299377


namespace sum_of_squares_l2993_299392

theorem sum_of_squares (a b : ℕ+) (h : a.val^2 + 2*a.val*b.val - 3*b.val^2 - 41 = 0) : 
  a.val^2 + b.val^2 = 221 := by sorry

end sum_of_squares_l2993_299392


namespace systematic_sampling_result_l2993_299350

def systematicSampling (totalProducts : Nat) (sampleSize : Nat) (firstSample : Nat) : List Nat :=
  let interval := totalProducts / sampleSize
  List.range sampleSize |>.map (fun i => firstSample + i * interval)

theorem systematic_sampling_result :
  systematicSampling 60 5 5 = [5, 17, 29, 41, 53] := by
  sorry

end systematic_sampling_result_l2993_299350


namespace second_term_of_geometric_series_l2993_299347

theorem second_term_of_geometric_series :
  ∀ (a : ℝ) (r : ℝ) (S : ℝ),
    r = (1 : ℝ) / 4 →
    S = 16 →
    S = a / (1 - r) →
    a * r = 3 :=
by sorry

end second_term_of_geometric_series_l2993_299347


namespace book_collection_problem_l2993_299371

/-- The number of books in either Jessica's or Tina's collection, but not both -/
def unique_books (shared : ℕ) (jessica_total : ℕ) (tina_unique : ℕ) : ℕ :=
  (jessica_total - shared) + tina_unique

theorem book_collection_problem :
  unique_books 12 22 10 = 20 := by
sorry

end book_collection_problem_l2993_299371


namespace log_problem_l2993_299300

theorem log_problem (x : ℝ) (h : Real.log x / Real.log 7 - Real.log 3 / Real.log 7 = 2) :
  Real.log x / Real.log 13 = Real.log 52 / Real.log 13 := by
  sorry

end log_problem_l2993_299300


namespace candy_distribution_l2993_299322

/-- Given a total number of candy pieces and the number of pieces per student,
    calculate the number of students. -/
def number_of_students (total_candy : ℕ) (candy_per_student : ℕ) : ℕ :=
  total_candy / candy_per_student

theorem candy_distribution (total_candy : ℕ) (candy_per_student : ℕ) 
    (h1 : total_candy = 344) 
    (h2 : candy_per_student = 8) :
    number_of_students total_candy candy_per_student = 43 := by
  sorry

end candy_distribution_l2993_299322


namespace dice_probability_l2993_299398

/-- A fair 10-sided die -/
def ten_sided_die : Finset ℕ := Finset.range 10

/-- A fair 6-sided die -/
def six_sided_die : Finset ℕ := Finset.range 6

/-- The event that the number on the 10-sided die is less than or equal to the number on the 6-sided die -/
def favorable_event : Finset (ℕ × ℕ) :=
  Finset.filter (fun p => p.1 ≤ p.2) (ten_sided_die.product six_sided_die)

/-- The probability of the event -/
def probability : ℚ :=
  (favorable_event.card : ℚ) / ((ten_sided_die.card * six_sided_die.card) : ℚ)

theorem dice_probability : probability = 7 / 20 := by sorry

end dice_probability_l2993_299398


namespace line_slope_l2993_299368

/-- The slope of the line given by the equation x/4 + y/5 = 1 is -5/4 -/
theorem line_slope (x y : ℝ) : 
  (x / 4 + y / 5 = 1) → (∃ m b : ℝ, y = m * x + b ∧ m = -5/4) :=
by sorry

end line_slope_l2993_299368


namespace flight_time_sum_l2993_299344

/-- Represents time in hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : minutes < 60

def Time.toMinutes (t : Time) : ℕ := t.hours * 60 + t.minutes

theorem flight_time_sum (departure : Time) (arrival : Time) (layover : ℕ) 
  (h m : ℕ) (hm : 0 < m ∧ m < 60) :
  departure.hours = 15 ∧ departure.minutes = 45 →
  arrival.hours = 20 ∧ arrival.minutes = 2 →
  layover = 25 →
  arrival.toMinutes - departure.toMinutes - layover = h * 60 + m →
  h + m = 55 := by
sorry

end flight_time_sum_l2993_299344


namespace simplify_expression_l2993_299372

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^3)^10 = 1342177280 := by
  sorry

end simplify_expression_l2993_299372


namespace original_number_proof_l2993_299399

theorem original_number_proof : ∃ N : ℕ, 
  (N > 30) ∧ 
  (N - 30) % 87 = 0 ∧ 
  (∀ M : ℕ, M > 30 ∧ (M - 30) % 87 = 0 → M ≥ N) :=
by sorry

end original_number_proof_l2993_299399


namespace train_speed_problem_train_speed_solution_l2993_299328

theorem train_speed_problem (train1_length train2_length : ℝ) 
  (train1_speed time_to_clear : ℝ) : ℝ :=
  let total_length := train1_length + train2_length
  let total_length_km := total_length / 1000
  let time_to_clear_hours := time_to_clear / 3600
  let relative_speed := total_length_km / time_to_clear_hours
  relative_speed - train1_speed

theorem train_speed_solution :
  train_speed_problem 140 280 42 20.99832013438925 = 30 := by
  sorry

end train_speed_problem_train_speed_solution_l2993_299328


namespace factorization_equality_l2993_299385

theorem factorization_equality (a b : ℝ) : a * b^2 - 2 * a * b + a = a * (b - 1)^2 := by
  sorry

end factorization_equality_l2993_299385


namespace tetrahedron_altitude_impossibility_l2993_299370

theorem tetrahedron_altitude_impossibility : ∀ (S₁ S₂ S₃ S₄ : ℝ),
  S₁ > 0 ∧ S₂ > 0 ∧ S₃ > 0 ∧ S₄ > 0 →
  ¬ ∃ (h₁ h₂ h₃ h₄ : ℝ),
    h₁ > 0 ∧ h₂ > 0 ∧ h₃ > 0 ∧ h₄ > 0 ∧
    (S₁ * h₁ = S₂ * h₂) ∧ (S₁ * h₁ = S₃ * h₃) ∧ (S₁ * h₁ = S₄ * h₄) ∧
    h₁ = 4 ∧ h₂ = 25 * Real.sqrt 3 / 3 ∧ h₃ = 25 * Real.sqrt 3 / 3 ∧ h₄ = 25 * Real.sqrt 3 / 3 :=
by sorry


end tetrahedron_altitude_impossibility_l2993_299370


namespace proposition_false_negation_true_l2993_299362

theorem proposition_false_negation_true :
  (¬ (∀ x y : ℝ, x + y > 0 → x > 0 ∧ y > 0)) ∧
  (∃ x y : ℝ, x + y > 0 ∧ (x ≤ 0 ∨ y ≤ 0)) :=
by sorry

end proposition_false_negation_true_l2993_299362


namespace cube_root_eq_four_l2993_299312

theorem cube_root_eq_four (y : ℝ) :
  (y * (y^5)^(1/2))^(1/3) = 4 → y = 4^(6/7) := by
  sorry

end cube_root_eq_four_l2993_299312


namespace arithmetic_sequence_common_difference_l2993_299393

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 3 + a 11 = 24)
  (h_a4 : a 4 = 3) :
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by sorry

end arithmetic_sequence_common_difference_l2993_299393


namespace minimum_value_theorem_l2993_299326

theorem minimum_value_theorem (m n t : ℝ) 
  (hm : m > 0) 
  (hn : n > 0) 
  (hmn : m + n = 1) 
  (ht : t > 0) 
  (hmin : ∀ s > 0, s / m + 1 / n ≥ t / m + 1 / n) 
  (heq : t / m + 1 / n = 9) : 
  t = 4 := by
sorry

end minimum_value_theorem_l2993_299326


namespace two_integer_solutions_l2993_299367

/-- The function f(x) = x^2 + bx + 1 -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 1

/-- The condition on b -/
def valid_b (b : ℝ) : Prop :=
  abs b > 2 ∧ ∀ a : ℤ, a ≠ 0 ∧ a ≠ 1 ∧ a ≠ -1 → b ≠ a + 1/a

/-- The main theorem -/
theorem two_integer_solutions (b : ℝ) (hb : valid_b b) :
  ∃! n : ℕ, n = 2 ∧ ∃ s : Finset ℤ, s.card = n ∧
    ∀ x : ℤ, x ∈ s ↔ f b (f b x + x) < 0 :=
sorry

end two_integer_solutions_l2993_299367


namespace a_2n_is_perfect_square_l2993_299381

/-- Define a function that counts the number of natural numbers with a given digit sum,
    where each digit can only be 1, 3, or 4 -/
def a (n : ℕ) : ℕ := sorry

/-- Theorem: For all natural numbers n, a(2n) is a perfect square -/
theorem a_2n_is_perfect_square (n : ℕ) : ∃ k : ℕ, a (2 * n) = k ^ 2 := by sorry

end a_2n_is_perfect_square_l2993_299381


namespace point_in_second_quadrant_l2993_299382

theorem point_in_second_quadrant (x y : ℝ) : 
  x < 0 ∧ y > 0 →  -- point is in the second quadrant
  |y| = 4 →        -- 4 units away from x-axis
  |x| = 7 →        -- 7 units away from y-axis
  x = -7 ∧ y = 4 := by
sorry

end point_in_second_quadrant_l2993_299382


namespace circle_area_equilateral_triangle_l2993_299332

/-- The area of a circle circumscribing an equilateral triangle with side length 12 units is 48π square units. -/
theorem circle_area_equilateral_triangle : 
  ∀ (s : ℝ) (A : ℝ),
  s = 12 →  -- Side length of the equilateral triangle
  A = π * (s / Real.sqrt 3)^2 →  -- Area of the circle (using the circumradius formula)
  A = 48 * π := by
sorry

end circle_area_equilateral_triangle_l2993_299332


namespace q_coordinates_l2993_299389

/-- Triangle ABC with points G on AC and H on AB -/
structure Triangle (A B C G H : ℝ × ℝ) : Prop where
  g_on_ac : ∃ t : ℝ, 0 < t ∧ t < 1 ∧ G = t • C + (1 - t) • A
  h_on_ab : ∃ s : ℝ, 0 < s ∧ s < 1 ∧ H = s • B + (1 - s) • A
  ag_gc_ratio : (G.1 - A.1) / (C.1 - G.1) = 3 / 2 ∧ (G.2 - A.2) / (C.2 - G.2) = 3 / 2
  ah_hb_ratio : (H.1 - A.1) / (B.1 - H.1) = 2 / 3 ∧ (H.2 - A.2) / (B.2 - H.2) = 2 / 3

/-- Q is the intersection of BG and CH -/
def Q (A B C G H : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Theorem: Coordinates of Q in terms of A, B, and C -/
theorem q_coordinates (A B C G H : ℝ × ℝ) (tri : Triangle A B C G H) :
  ∃ (u v w : ℝ), u + v + w = 1 ∧ 
    Q A B C G H = (u • A.1 + v • B.1 + w • C.1, u • A.2 + v • B.2 + w • C.2) ∧
    u = 5/13 ∧ v = 11/26 ∧ w = 3/13 :=
  sorry

end q_coordinates_l2993_299389


namespace adjacent_probability_l2993_299361

/-- The number of students in the arrangement -/
def total_students : ℕ := 9

/-- The number of rows in the seating arrangement -/
def rows : ℕ := 3

/-- The number of columns in the seating arrangement -/
def columns : ℕ := 3

/-- The number of ways to arrange n students -/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of adjacent pairs in a row or column -/
def adjacent_pairs_per_line : ℕ := 2

/-- The number of ways to arrange two specific students in an adjacent pair -/
def ways_to_arrange_pair : ℕ := 2

/-- The probability of two specific students being adjacent in a 3x3 grid -/
theorem adjacent_probability :
  (((rows * adjacent_pairs_per_line + columns * adjacent_pairs_per_line) * ways_to_arrange_pair * 
    (arrangements (total_students - 2))) : ℚ) / 
  (arrangements total_students) = 1 / 3 := by
  sorry

end adjacent_probability_l2993_299361


namespace number_of_boys_l2993_299376

theorem number_of_boys (M W B : ℕ) : 
  M = W → 
  W = B → 
  M * 8 = 120 → 
  B = 15 := by sorry

end number_of_boys_l2993_299376


namespace g_1993_of_4_l2993_299335

-- Define the function g
def g (x : ℚ) : ℚ := (2 + x) / (2 - 4 * x)

-- Define the recursive function gn
def gn : ℕ → ℚ → ℚ
  | 0, x => x
  | n + 1, x => g (gn n x)

-- Theorem statement
theorem g_1993_of_4 : gn 1993 4 = 11 / 20 := by
  sorry

end g_1993_of_4_l2993_299335


namespace negative_300_coterminal_with_60_l2993_299343

/-- Two angles are coterminal if their difference is a multiple of 360° -/
def coterminal (a b : ℝ) : Prop := ∃ k : ℤ, a - b = 360 * k

/-- The theorem states that -300° is coterminal with 60° -/
theorem negative_300_coterminal_with_60 : coterminal (-300 : ℝ) 60 := by
  sorry

end negative_300_coterminal_with_60_l2993_299343


namespace inequality_solution_set_empty_implies_k_range_l2993_299386

theorem inequality_solution_set_empty_implies_k_range (k : ℝ) : 
  (∀ x : ℝ, k * x^2 - 2 * |x - 1| + 6 * k ≥ 0) → 
  k ≥ (1 + Real.sqrt 7) / 6 := by
  sorry

end inequality_solution_set_empty_implies_k_range_l2993_299386


namespace star_equal_is_diagonal_l2993_299316

/-- The star operation defined on real numbers -/
def star (a b : ℝ) : ℝ := a * b * (a - b)

/-- The set of points (x, y) where x ★ y = y ★ x -/
def star_equal_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1}

/-- The line y = x in ℝ² -/
def diagonal_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = p.2}

theorem star_equal_is_diagonal :
  star_equal_set = diagonal_line := by sorry

end star_equal_is_diagonal_l2993_299316


namespace min_MN_length_l2993_299363

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) (a b : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

/-- Definition of the circle containing point P -/
def circle_P (x y : ℝ) : Prop :=
  x^2 + y^2 = 16

/-- Theorem statement -/
theorem min_MN_length (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ellipse_C 0 1 a b) -- vertex at (0,1)
  (h4 : (a^2 - b^2) / a^2 = 3/4) -- eccentricity is √3/2
  : ∃ (x_p y_p x_m y_n : ℝ),
    circle_P x_p y_p ∧
    (∀ (x_a y_a x_b y_b : ℝ),
      ellipse_C x_a y_a a b →
      ellipse_C x_b y_b a b →
      (y_n - y_a) * (x_p - x_a) = (y_p - y_a) * (x_m - x_a) →
      (y_n - y_b) * (x_p - x_b) = (y_p - y_b) * (x_m - x_b) →
      x_m = 0 ∧ y_n = 0) →
    (x_m - 0)^2 + (0 - y_n)^2 ≥ (5/4)^2 :=
by sorry

end min_MN_length_l2993_299363


namespace sin_cos_sum_14_16_l2993_299310

theorem sin_cos_sum_14_16 : 
  Real.sin (14 * π / 180) * Real.cos (16 * π / 180) + 
  Real.cos (14 * π / 180) * Real.sin (16 * π / 180) = 1 / 2 := by
  sorry

end sin_cos_sum_14_16_l2993_299310


namespace system_solution_l2993_299365

theorem system_solution : 
  ∀ (x y : ℝ), x > 0 ∧ y > 0 → 
  (x - 3 * Real.sqrt (x * y) - 2 * Real.sqrt (x / y) + 6 = 0 ∧ 
   x^2 * y^2 + x^4 = 82) → 
  ((x = 3 ∧ y = 1/3) ∨ (x = Real.rpow 33 (1/4) ∧ y = 4 / Real.rpow 33 (1/4))) :=
by sorry

end system_solution_l2993_299365


namespace gym_down_payment_down_payment_is_50_l2993_299358

/-- Calculates the down payment for a gym membership -/
theorem gym_down_payment (monthly_fee : ℕ) (total_payment : ℕ) : ℕ :=
  let months : ℕ := 3 * 12
  let total_monthly_payments : ℕ := months * monthly_fee
  total_payment - total_monthly_payments

/-- Proves that the down payment for the gym membership is $50 -/
theorem down_payment_is_50 :
  gym_down_payment 12 482 = 50 := by
  sorry

end gym_down_payment_down_payment_is_50_l2993_299358


namespace rectangle_side_ratio_l2993_299339

theorem rectangle_side_ratio (a b c d : ℝ) (h : a / c = b / d ∧ a / c = 4 / 5) :
  a / c = 4 / 5 := by
  sorry

end rectangle_side_ratio_l2993_299339


namespace tan_seven_pi_sixths_l2993_299311

theorem tan_seven_pi_sixths : Real.tan (7 * Real.pi / 6) = Real.sqrt 3 / 3 := by
  sorry

end tan_seven_pi_sixths_l2993_299311


namespace pascal_triangle_sum_rows_7_8_l2993_299324

theorem pascal_triangle_sum_rows_7_8 : ℕ := by
  -- Define the sum of numbers in row n of Pascal's Triangle
  let sum_row (n : ℕ) := 2^n
  
  -- Sum of Row 7
  let sum_row_7 := sum_row 7
  
  -- Sum of Row 8
  let sum_row_8 := sum_row 8
  
  -- Total sum of Rows 7 and 8
  let total_sum := sum_row_7 + sum_row_8
  
  -- Prove that the total sum equals 384
  have h : total_sum = 384 := by sorry
  
  exact 384


end pascal_triangle_sum_rows_7_8_l2993_299324


namespace most_accurate_reading_l2993_299345

def scale_reading : ℝ → Prop :=
  λ x => 3.25 < x ∧ x < 3.5

def closer_to_3_3 (x : ℝ) : Prop :=
  |x - 3.3| < |x - 3.375|

def options : Set ℝ :=
  {3.05, 3.15, 3.25, 3.3, 3.6}

theorem most_accurate_reading (x : ℝ) 
  (h1 : scale_reading x) 
  (h2 : closer_to_3_3 x) : 
  ∀ y ∈ options, |x - 3.3| ≤ |x - y| :=
by sorry

end most_accurate_reading_l2993_299345


namespace geometric_sequence_fifth_term_l2993_299353

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The fifth term of a geometric sequence. -/
def FifthTerm (a : ℕ → ℝ) : ℝ := a 5

theorem geometric_sequence_fifth_term
    (a : ℕ → ℝ)
    (h_pos : ∀ n, a n > 0)
    (h_geom : IsGeometricSequence a)
    (h_prod : a 1 * a 3 = 16)
    (h_sum : a 3 + a 4 = 24) :
  FifthTerm a = 32 := by
  sorry

end geometric_sequence_fifth_term_l2993_299353


namespace min_four_digit_with_different_remainders_l2993_299342

theorem min_four_digit_with_different_remainders :
  ∃ (n : ℕ),
    1000 ≤ n ∧ n ≤ 9999 ∧
    (∀ i j, i ≠ j → n % (i + 2) ≠ n % (j + 2)) ∧
    (∀ i, n % (i + 2) ≠ 0) ∧
    (∀ m, 1000 ≤ m ∧ m < n →
      ¬(∀ i j, i ≠ j → m % (i + 2) ≠ m % (j + 2)) ∨
      ¬(∀ i, m % (i + 2) ≠ 0)) ∧
    n = 1259 :=
by sorry

end min_four_digit_with_different_remainders_l2993_299342


namespace minimum_value_theorem_l2993_299388

theorem minimum_value_theorem (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  a^2 + b^2 + c^2 + 1/a^2 + b/a + 1/c^2 ≥ Real.sqrt 3 + 2 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ ≠ 0 ∧ b₀ ≠ 0 ∧ c₀ ≠ 0 ∧
    a₀^2 + b₀^2 + c₀^2 + 1/a₀^2 + b₀/a₀ + 1/c₀^2 = Real.sqrt 3 + 2 :=
by sorry

end minimum_value_theorem_l2993_299388


namespace wedding_champagne_bottles_l2993_299356

/-- The number of wedding guests -/
def num_guests : ℕ := 120

/-- The number of glasses of champagne per guest -/
def glasses_per_guest : ℕ := 2

/-- The number of servings per bottle of champagne -/
def servings_per_bottle : ℕ := 6

/-- The number of bottles of champagne needed for the wedding toast -/
def bottles_needed : ℕ := (num_guests * glasses_per_guest) / servings_per_bottle

theorem wedding_champagne_bottles : bottles_needed = 40 := by
  sorry

end wedding_champagne_bottles_l2993_299356


namespace trapezoid_area_equality_l2993_299384

/-- Given a square ABCD with side length a, and a trapezoid EBCF inside it with BE = CF = x,
    if the area of EBCF equals the area of ABCD minus twice the area of a rectangle JKHG 
    inside the square, then x = a/2 -/
theorem trapezoid_area_equality (a : ℝ) (x : ℝ) :
  (∃ (y z : ℝ), y + z = a ∧ x * a = a^2 - 2 * y * z) →
  x = a / 2 := by
  sorry

end trapezoid_area_equality_l2993_299384
