import Mathlib

namespace second_number_difference_l3035_303545

theorem second_number_difference (first_number second_number : ℤ) : 
  first_number = 15 →
  second_number = 55 →
  first_number + second_number = 70 →
  second_number - 3 * first_number = 10 := by
sorry

end second_number_difference_l3035_303545


namespace equation_solutions_l3035_303565

theorem equation_solutions :
  (∀ x : ℝ, (x - 2)^2 = 4 ↔ x = 4 ∨ x = 0) ∧
  (∀ x : ℝ, x^2 - 3*x + 1 = 0 ↔ x = (3 + Real.sqrt 5) / 2 ∨ x = (3 - Real.sqrt 5) / 2) := by
  sorry

end equation_solutions_l3035_303565


namespace right_triangle_sets_l3035_303537

/-- A function that checks if three numbers can form a right-angled triangle -/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The theorem stating that among the given sets, only (5, 12, 13) forms a right-angled triangle -/
theorem right_triangle_sets : 
  is_right_triangle 5 12 13 ∧
  ¬is_right_triangle 2 3 4 ∧
  ¬is_right_triangle 4 5 6 ∧
  ¬is_right_triangle 3 4 6 :=
by sorry

end right_triangle_sets_l3035_303537


namespace probability_second_new_given_first_new_l3035_303580

/-- Represents the total number of balls in the box -/
def total_balls : ℕ := 10

/-- Represents the number of new balls initially in the box -/
def new_balls : ℕ := 6

/-- Represents the number of old balls initially in the box -/
def old_balls : ℕ := 4

/-- Theorem stating the probability of drawing a new ball on the second draw,
    given that the first ball drawn was new -/
theorem probability_second_new_given_first_new :
  (new_balls - 1) / (total_balls - 1) = 5 / 9 := by sorry

end probability_second_new_given_first_new_l3035_303580


namespace stickers_in_red_folder_l3035_303504

/-- The number of stickers on each sheet in the red folder -/
def red_stickers : ℕ := 3

/-- The number of sheets in each folder -/
def sheets_per_folder : ℕ := 10

/-- The number of stickers on each sheet in the green folder -/
def green_stickers : ℕ := 2

/-- The number of stickers on each sheet in the blue folder -/
def blue_stickers : ℕ := 1

/-- The total number of stickers used -/
def total_stickers : ℕ := 60

theorem stickers_in_red_folder :
  red_stickers * sheets_per_folder +
  green_stickers * sheets_per_folder +
  blue_stickers * sheets_per_folder = total_stickers :=
by sorry

end stickers_in_red_folder_l3035_303504


namespace arithmetic_sequence_500th_term_l3035_303578

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

theorem arithmetic_sequence_500th_term 
  (p q : ℝ) 
  (h1 : arithmetic_sequence p 9 2 = 9)
  (h2 : arithmetic_sequence p 9 3 = 3*p - q^3)
  (h3 : arithmetic_sequence p 9 4 = 3*p + q^3) :
  arithmetic_sequence p 9 500 = 2005 - 2 * Real.rpow 2 (1/3) :=
sorry

end arithmetic_sequence_500th_term_l3035_303578


namespace sum_and_multiply_l3035_303570

theorem sum_and_multiply : (57.6 + 1.4) * 3 = 177 := by
  sorry

end sum_and_multiply_l3035_303570


namespace sample_size_equals_selected_high_school_entrance_exam_sample_size_l3035_303557

/-- Represents a statistical sample --/
structure Sample where
  population : ℕ
  selected : ℕ

/-- Definition of sample size --/
def sampleSize (s : Sample) : ℕ := s.selected

/-- Theorem stating that the sample size is equal to the number of selected students --/
theorem sample_size_equals_selected (s : Sample) 
  (h₁ : s.population = 150000) 
  (h₂ : s.selected = 1000) : 
  sampleSize s = 1000 := by
  sorry

/-- Main theorem proving the sample size for the given problem --/
theorem high_school_entrance_exam_sample_size :
  ∃ s : Sample, s.population = 150000 ∧ s.selected = 1000 ∧ sampleSize s = 1000 := by
  sorry

end sample_size_equals_selected_high_school_entrance_exam_sample_size_l3035_303557


namespace circle_point_range_l3035_303581

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - Real.sqrt 3)^2 = 1

-- Define points A and B
def point_A (m : ℝ) : ℝ × ℝ := (0, m)
def point_B (m : ℝ) : ℝ × ℝ := (0, -m)

-- Define the condition for point P
def point_P_condition (P : ℝ × ℝ) (m : ℝ) : Prop :=
  circle_C P.1 P.2 ∧ 
  ∃ (A B : ℝ × ℝ), A = point_A m ∧ B = point_B m ∧ 
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

-- Main theorem
theorem circle_point_range (m : ℝ) :
  m > 0 → (∃ P : ℝ × ℝ, point_P_condition P m) → 1 ≤ m ∧ m ≤ 3 :=
by sorry

end circle_point_range_l3035_303581


namespace distinct_tower_heights_l3035_303527

/-- Represents the number of bricks in the tower. -/
def num_bricks : ℕ := 50

/-- Represents the minimum possible height of the tower in inches. -/
def min_height : ℕ := 250

/-- Represents the maximum possible height of the tower in inches. -/
def max_height : ℕ := 900

/-- The theorem stating the number of distinct tower heights achievable. -/
theorem distinct_tower_heights :
  ∃ (heights : Finset ℕ),
    (∀ h ∈ heights, min_height ≤ h ∧ h ≤ max_height) ∧
    (∀ h, min_height ≤ h → h ≤ max_height →
      (∃ (a b c : ℕ), a + b + c = num_bricks ∧ 5*a + 12*b + 18*c = h) ↔ h ∈ heights) ∧
    heights.card = 651 := by
  sorry

end distinct_tower_heights_l3035_303527


namespace cookie_difference_l3035_303593

def sweet_cookies_initial : ℕ := 37
def salty_cookies_initial : ℕ := 11
def sweet_cookies_eaten : ℕ := 5
def salty_cookies_eaten : ℕ := 2

theorem cookie_difference : sweet_cookies_eaten - salty_cookies_eaten = 3 := by
  sorry

end cookie_difference_l3035_303593


namespace triangle_inradius_l3035_303582

/-- Given a triangle with perimeter 32 cm and area 40 cm², prove that its inradius is 2.5 cm. -/
theorem triangle_inradius (P : ℝ) (A : ℝ) (r : ℝ) 
  (h_perimeter : P = 32) 
  (h_area : A = 40) 
  (h_inradius : A = r * (P / 2)) : 
  r = 2.5 := by
  sorry

end triangle_inradius_l3035_303582


namespace num_technicians_correct_l3035_303589

/-- The number of technicians in a workshop with given conditions. -/
def num_technicians : ℕ :=
  let total_workers : ℕ := 42
  let avg_salary_all : ℕ := 8000
  let avg_salary_technicians : ℕ := 18000
  let avg_salary_rest : ℕ := 6000
  7

/-- Theorem stating that the number of technicians is correct given the workshop conditions. -/
theorem num_technicians_correct :
  let total_workers : ℕ := 42
  let avg_salary_all : ℕ := 8000
  let avg_salary_technicians : ℕ := 18000
  let avg_salary_rest : ℕ := 6000
  let num_technicians := num_technicians
  let num_rest := total_workers - num_technicians
  (num_technicians * avg_salary_technicians + num_rest * avg_salary_rest) / total_workers = avg_salary_all :=
by
  sorry

#eval num_technicians

end num_technicians_correct_l3035_303589


namespace angle_DAE_in_special_triangle_l3035_303514

-- Define the triangle ABC
def Triangle (A B C : Point) : Prop := sorry

-- Define the angle measure in degrees
def AngleMeasure (A B C : Point) : ℝ := sorry

-- Define the foot of the perpendicular
def PerpendicularFoot (A D : Point) (B C : Point) : Prop := sorry

-- Define the center of the circumscribed circle
def CircumcenterOfTriangle (O A B C : Point) : Prop := sorry

-- Define the diameter of a circle
def DiameterOfCircle (A E O : Point) : Prop := sorry

theorem angle_DAE_in_special_triangle 
  (A B C D E O : Point) 
  (triangle_ABC : Triangle A B C)
  (angle_ACB : AngleMeasure A C B = 40)
  (angle_CBA : AngleMeasure C B A = 60)
  (D_perpendicular : PerpendicularFoot A D B C)
  (O_circumcenter : CircumcenterOfTriangle O A B C)
  (AE_diameter : DiameterOfCircle A E O) :
  AngleMeasure D A E = 20 := by
sorry

end angle_DAE_in_special_triangle_l3035_303514


namespace geometric_sequence_first_term_l3035_303583

/-- Given a geometric sequence {a_n} with specific conditions, prove that a_1 = -1/2 -/
theorem geometric_sequence_first_term 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) 
  (h_product : a 2 * a 5 * a 8 = -8) 
  (h_sum : a 1 + a 2 + a 3 = a 2 + 3 * a 1) : 
  a 1 = -1/2 := by
sorry

end geometric_sequence_first_term_l3035_303583


namespace triangle_sides_simplification_l3035_303538

theorem triangle_sides_simplification (a b c : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ a + c > b) : 
  |a + b - c| - |a - c - b| = 2*a - 2*c := by
  sorry

end triangle_sides_simplification_l3035_303538


namespace range_of_m_l3035_303548

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 3*x - 10 ≤ 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- State the theorem
theorem range_of_m (m : ℝ) : A ∩ B m = B m → m ≤ 3 := by
  sorry

end range_of_m_l3035_303548


namespace polynomial_division_theorem_l3035_303597

theorem polynomial_division_theorem (x : ℝ) : 
  x^5 - 25*x^3 + 13*x^2 - 16*x + 12 = (x - 3) * (x^4 + 3*x^3 - 16*x^2 - 35*x - 121) + (-297) := by
  sorry

end polynomial_division_theorem_l3035_303597


namespace binding_cost_per_manuscript_l3035_303586

/-- Proves that the binding cost per manuscript is $5 given the specified conditions. -/
theorem binding_cost_per_manuscript
  (num_manuscripts : ℕ)
  (pages_per_manuscript : ℕ)
  (copy_cost_per_page : ℚ)
  (total_cost : ℚ)
  (h1 : num_manuscripts = 10)
  (h2 : pages_per_manuscript = 400)
  (h3 : copy_cost_per_page = 5 / 100)
  (h4 : total_cost = 250) :
  (total_cost - (num_manuscripts * pages_per_manuscript * copy_cost_per_page)) / num_manuscripts = 5 :=
by sorry

end binding_cost_per_manuscript_l3035_303586


namespace c_investment_value_l3035_303541

/-- Represents the investment and profit distribution in a partnership business. -/
structure Partnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  c_profit : ℕ

/-- Theorem stating that given the conditions of the partnership,
    c's investment is 50,000. -/
theorem c_investment_value (p : Partnership)
  (h1 : p.a_investment = 30000)
  (h2 : p.b_investment = 45000)
  (h3 : p.total_profit = 90000)
  (h4 : p.c_profit = 36000)
  (h5 : p.c_investment * p.total_profit = p.c_profit * (p.a_investment + p.b_investment + p.c_investment)) :
  p.c_investment = 50000 := by
  sorry


end c_investment_value_l3035_303541


namespace petes_son_age_l3035_303595

/-- Given Pete's current age and the relationship between Pete's and his son's ages in 4 years,
    this theorem proves the current age of Pete's son. -/
theorem petes_son_age (pete_age : ℕ) (h : pete_age = 35) :
  ∃ (son_age : ℕ), son_age = 9 ∧ pete_age + 4 = 3 * (son_age + 4) := by
  sorry

end petes_son_age_l3035_303595


namespace digit_puzzle_l3035_303572

theorem digit_puzzle :
  ∀ (A B C D E F G H M : ℕ),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ D ≠ 0 ∧ E ≠ 0 ∧ F ≠ 0 ∧ G ≠ 0 ∧ H ≠ 0 ∧ M ≠ 0 →
    A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧ A ≠ M →
    B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧ B ≠ M →
    C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧ C ≠ M →
    D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧ D ≠ M →
    E ≠ F ∧ E ≠ G ∧ E ≠ H ∧ E ≠ M →
    F ≠ G ∧ F ≠ H ∧ F ≠ M →
    G ≠ H ∧ G ≠ M →
    H ≠ M →
    A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧ E < 10 ∧ F < 10 ∧ G < 10 ∧ H < 10 ∧ M < 10 →
    A + B = 14 →
    M / G = M - F ∧ M - F = H - C →
    D * F = 24 →
    B + E = 16 →
    H = 4 :=
by sorry

end digit_puzzle_l3035_303572


namespace indeterminate_product_l3035_303556

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the continuity of f on [-2, 2]
variable (hcont : ContinuousOn f (Set.Icc (-2) 2))

-- Define that f has at least one root in (-2, 2)
variable (hroot : ∃ x ∈ Set.Ioo (-2) 2, f x = 0)

-- Theorem statement
theorem indeterminate_product :
  ¬ (∀ (f : ℝ → ℝ) (hcont : ContinuousOn f (Set.Icc (-2) 2)) 
    (hroot : ∃ x ∈ Set.Ioo (-2) 2, f x = 0),
    (f (-2) * f 2 > 0) ∨ (f (-2) * f 2 < 0) ∨ (f (-2) * f 2 = 0)) :=
by sorry

end indeterminate_product_l3035_303556


namespace green_apples_count_l3035_303547

theorem green_apples_count (total : ℕ) (red_to_green_ratio : ℕ) 
  (h1 : total = 496) 
  (h2 : red_to_green_ratio = 3) : 
  ∃ green : ℕ, green = 124 ∧ total = green * (red_to_green_ratio + 1) :=
by sorry

end green_apples_count_l3035_303547


namespace line_equation_l3035_303573

/-- A line passing through point (1, 2) with slope √3 has the equation √3x - y + 2 - √3 = 0 -/
theorem line_equation (x y : ℝ) : 
  (y - 2 = Real.sqrt 3 * (x - 1)) ↔ (Real.sqrt 3 * x - y + 2 - Real.sqrt 3 = 0) := by
  sorry

end line_equation_l3035_303573


namespace spring_bud_cup_value_l3035_303540

theorem spring_bud_cup_value : ∃ x : ℕ, x + x = 578 ∧ x = 289 := by sorry

end spring_bud_cup_value_l3035_303540


namespace dog_purchase_cost_l3035_303553

theorem dog_purchase_cost (current_amount additional_amount : ℕ) 
  (h1 : current_amount = 34)
  (h2 : additional_amount = 13) :
  current_amount + additional_amount = 47 := by
  sorry

end dog_purchase_cost_l3035_303553


namespace yogurt_combinations_count_l3035_303536

/-- The number of combinations of one item from a set of 4 and two different items from a set of 6 -/
def yogurt_combinations (flavors : Nat) (toppings : Nat) : Nat :=
  flavors * (toppings.choose 2)

/-- Theorem stating that the number of combinations is 60 -/
theorem yogurt_combinations_count :
  yogurt_combinations 4 6 = 60 := by
  sorry

end yogurt_combinations_count_l3035_303536


namespace lcm_of_ratio_numbers_l3035_303576

theorem lcm_of_ratio_numbers (a b : ℕ) (h1 : a = 48) (h2 : b * 8 = a * 9) : 
  Nat.lcm a b = 432 := by
  sorry

end lcm_of_ratio_numbers_l3035_303576


namespace salesman_profit_l3035_303569

/-- Calculates the salesman's profit from backpack sales --/
theorem salesman_profit : 
  let initial_cost : ℚ := 1500
  let import_tax_rate : ℚ := 5 / 100
  let total_cost : ℚ := initial_cost * (1 + import_tax_rate)
  let swap_meet_sales : ℚ := 30 * 22
  let department_store_sales : ℚ := 25 * 35
  let online_sales_regular : ℚ := 10 * 28
  let online_sales_discounted : ℚ := 5 * 28 * (1 - 10 / 100)
  let local_market_sales_1 : ℚ := 10 * 33
  let local_market_sales_2 : ℚ := 5 * 40
  let local_market_sales_3 : ℚ := 15 * 25
  let shipping_expenses : ℚ := 60
  let total_revenue : ℚ := swap_meet_sales + department_store_sales + 
    online_sales_regular + online_sales_discounted + 
    local_market_sales_1 + local_market_sales_2 + local_market_sales_3
  let profit : ℚ := total_revenue - total_cost - shipping_expenses
  profit = 1211 := by sorry

end salesman_profit_l3035_303569


namespace max_d_value_l3035_303500

def a (n : ℕ+) : ℕ := 100 + n^2

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  (∃ k : ℕ+, d k = 401) ∧ (∀ n : ℕ+, d n ≤ 401) := by
  sorry

end max_d_value_l3035_303500


namespace sin_300_degrees_l3035_303568

theorem sin_300_degrees : Real.sin (300 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_300_degrees_l3035_303568


namespace choose_two_from_three_l3035_303535

theorem choose_two_from_three (n : ℕ) (k : ℕ) : n = 3 ∧ k = 2 → Nat.choose n k = 3 := by
  sorry

end choose_two_from_three_l3035_303535


namespace ratio_equality_l3035_303596

theorem ratio_equality (a b : ℝ) (h1 : 7 * a = 8 * b) (h2 : a * b ≠ 0) :
  (a / 8) / (b / 7) = 1 := by
  sorry

end ratio_equality_l3035_303596


namespace lines_intersection_l3035_303546

/-- Represents a 2D point or vector -/
structure Point2D where
  x : ℚ
  y : ℚ

/-- Represents a parametric line in 2D -/
structure ParametricLine where
  origin : Point2D
  direction : Point2D

def line1 : ParametricLine := {
  origin := { x := 2, y := 3 },
  direction := { x := 3, y := -1 }
}

def line2 : ParametricLine := {
  origin := { x := 4, y := 1 },
  direction := { x := 1, y := 5 }
}

def intersection : Point2D := {
  x := 26 / 7,
  y := 17 / 7
}

/-- 
  Theorem: The point (26/7, 17/7) is the unique intersection point of the two given lines.
-/
theorem lines_intersection (t u : ℚ) : 
  (∃! p : Point2D, 
    p.x = line1.origin.x + t * line1.direction.x ∧ 
    p.y = line1.origin.y + t * line1.direction.y ∧
    p.x = line2.origin.x + u * line2.direction.x ∧ 
    p.y = line2.origin.y + u * line2.direction.y) ∧
  (intersection.x = line1.origin.x + t * line1.direction.x) ∧
  (intersection.y = line1.origin.y + t * line1.direction.y) ∧
  (intersection.x = line2.origin.x + u * line2.direction.x) ∧
  (intersection.y = line2.origin.y + u * line2.direction.y) :=
by sorry

end lines_intersection_l3035_303546


namespace pages_left_to_read_l3035_303554

/-- Given a book with specified total pages, pages read, daily reading rate, and reading duration,
    calculate the number of pages left to read after the reading period. -/
theorem pages_left_to_read 
  (total_pages : ℕ) 
  (pages_read : ℕ) 
  (pages_per_day : ℕ) 
  (days : ℕ) 
  (h1 : total_pages = 381) 
  (h2 : pages_read = 149) 
  (h3 : pages_per_day = 20) 
  (h4 : days = 7) :
  total_pages - pages_read - (pages_per_day * days) = 92 := by
  sorry


end pages_left_to_read_l3035_303554


namespace expression_evaluation_l3035_303529

theorem expression_evaluation (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hxy : x > y) (hyz : y > z) :
  (x^z * y^x * z^y) / (z^z * y^y * x^x) = x^(z-x) * y^(x-y) * z^(y-z) := by
  sorry

end expression_evaluation_l3035_303529


namespace polygon_with_16_diagonals_has_7_sides_l3035_303518

/-- The number of sides in a regular polygon with 16 diagonals -/
def num_sides_of_polygon_with_16_diagonals : ℕ := 7

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem polygon_with_16_diagonals_has_7_sides :
  num_diagonals num_sides_of_polygon_with_16_diagonals = 16 :=
by sorry

end polygon_with_16_diagonals_has_7_sides_l3035_303518


namespace dividend_calculation_l3035_303542

theorem dividend_calculation (remainder quotient divisor : ℕ) 
  (h_remainder : remainder = 8)
  (h_quotient : quotient = 43)
  (h_divisor : divisor = 23) :
  divisor * quotient + remainder = 997 := by
  sorry

end dividend_calculation_l3035_303542


namespace circle_tangent_to_line_l3035_303574

/-- A circle with equation x^2 + y^2 = m^2 is tangent to the line x + 2y = √(3m) if and only if m = 3/5 -/
theorem circle_tangent_to_line (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 = m^2 ∧ x + 2*y = Real.sqrt (3*m) ∧ 
    ∀ (x' y' : ℝ), x'^2 + y'^2 = m^2 → x' + 2*y' ≠ Real.sqrt (3*m) ∨ (x' = x ∧ y' = y)) ↔ 
  m = 3/5 := by
sorry

end circle_tangent_to_line_l3035_303574


namespace average_side_length_of_squares_l3035_303567

theorem average_side_length_of_squares (a b c : Real) 
  (ha : a = 25) (hb : b = 64) (hc : c = 144) : 
  (Real.sqrt a + Real.sqrt b + Real.sqrt c) / 3 = 25 / 3 := by
  sorry

end average_side_length_of_squares_l3035_303567


namespace chime_2400_date_l3035_303577

/-- Represents a date in the year 2004 --/
structure Date2004 where
  month : Nat
  day : Nat

/-- Represents a time of day --/
structure Time where
  hour : Nat
  minute : Nat

/-- Calculates the number of chimes for a given hour --/
def chimesForHour (hour : Nat) : Nat :=
  3 * (hour % 12 + if hour % 12 = 0 then 12 else 0)

/-- Calculates the total chimes from the start time to midnight --/
def chimesToMidnight (startTime : Time) : Nat :=
  sorry

/-- Calculates the total chimes for a full day --/
def chimesPerDay : Nat :=
  258

/-- Determines the date when the nth chime occurs --/
def dateOfNthChime (n : Nat) (startDate : Date2004) (startTime : Time) : Date2004 :=
  sorry

theorem chime_2400_date :
  dateOfNthChime 2400 ⟨2, 28⟩ ⟨17, 45⟩ = ⟨3, 7⟩ := by sorry

end chime_2400_date_l3035_303577


namespace quadratic_factorization_l3035_303524

theorem quadratic_factorization (a : ℝ) : a^2 - a + 1/4 = (a - 1/2)^2 := by sorry

end quadratic_factorization_l3035_303524


namespace min_value_expression_l3035_303571

theorem min_value_expression (x y z : ℝ) (h : z = Real.sin x) :
  ∃ (m : ℝ), (∀ (x' y' z' : ℝ), z' = Real.sin x' →
    (y' * Real.cos x' - 2)^2 + (y' + z' + 1)^2 ≥ m) ∧
  m = (1/2 : ℝ) := by
  sorry

end min_value_expression_l3035_303571


namespace hendecagon_diagonal_intersection_probability_l3035_303539

/-- A regular hendecagon is an 11-sided regular polygon -/
def RegularHendecagon : Type := Unit

/-- The number of vertices in a regular hendecagon -/
def num_vertices : ℕ := 11

/-- The number of diagonals in a regular hendecagon -/
def num_diagonals (h : RegularHendecagon) : ℕ := 44

/-- The number of pairs of diagonals in a regular hendecagon -/
def num_diagonal_pairs (h : RegularHendecagon) : ℕ := Nat.choose (num_diagonals h) 2

/-- The number of intersecting diagonal pairs inside a regular hendecagon -/
def num_intersecting_pairs (h : RegularHendecagon) : ℕ := Nat.choose num_vertices 4

/-- The probability that two randomly chosen diagonals intersect inside the hendecagon -/
def intersection_probability (h : RegularHendecagon) : ℚ :=
  (num_intersecting_pairs h : ℚ) / (num_diagonal_pairs h : ℚ)

theorem hendecagon_diagonal_intersection_probability (h : RegularHendecagon) :
  intersection_probability h = 165 / 473 := by
  sorry

end hendecagon_diagonal_intersection_probability_l3035_303539


namespace max_product_dice_rolls_l3035_303591

theorem max_product_dice_rolls (rolls : List Nat) : 
  rolls.length = 25 → 
  (∀ x ∈ rolls, 1 ≤ x ∧ x ≤ 20) →
  rolls.sum = 70 →
  rolls.prod ≤ (List.replicate 5 2 ++ List.replicate 20 3).prod :=
sorry

end max_product_dice_rolls_l3035_303591


namespace third_set_size_l3035_303563

/-- The number of students in the third set that satisfies the given conditions -/
def third_set_students : ℕ := 60

/-- The pass percentage of the whole set -/
def total_pass_percentage : ℚ := 266 / 300

theorem third_set_size :
  let first_set := 40
  let second_set := 50
  let first_pass_rate := 1
  let second_pass_rate := 9 / 10
  let third_pass_rate := 4 / 5
  (first_set * first_pass_rate + second_set * second_pass_rate + third_set_students * third_pass_rate) /
    (first_set + second_set + third_set_students) = total_pass_percentage := by
  sorry

#check third_set_size

end third_set_size_l3035_303563


namespace cos_negative_seventeen_thirds_pi_l3035_303592

theorem cos_negative_seventeen_thirds_pi : 
  Real.cos (-17/3 * Real.pi) = 1/2 := by sorry

end cos_negative_seventeen_thirds_pi_l3035_303592


namespace completePassage_correct_l3035_303520

/-- Represents an incomplete sentence or passage -/
inductive IncompleteSentence : Type
| Wei : IncompleteSentence
| Zhuangzi : IncompleteSentence
| TaoYuanming : IncompleteSentence
| LiBai : IncompleteSentence
| SuShi : IncompleteSentence
| XinQiji : IncompleteSentence
| Analects : IncompleteSentence
| LiuYuxi : IncompleteSentence

/-- Represents the correct completion for a sentence -/
def Completion : Type := String

/-- A function that returns the correct completion for a given incomplete sentence -/
def completePassage : IncompleteSentence → Completion
| IncompleteSentence.Wei => "垝垣"
| IncompleteSentence.Zhuangzi => "水之积也不厚"
| IncompleteSentence.TaoYuanming => "仰而视之"
| IncompleteSentence.LiBai => "扶疏荫初上"
| IncompleteSentence.SuShi => "举匏樽"
| IncompleteSentence.XinQiji => "骑鲸鱼"
| IncompleteSentence.Analects => "切问而近思"
| IncompleteSentence.LiuYuxi => "莫是银屏"

/-- Theorem stating that the completePassage function returns the correct completion for each incomplete sentence -/
theorem completePassage_correct :
  ∀ (s : IncompleteSentence), 
    (s = IncompleteSentence.Wei → completePassage s = "垝垣") ∧
    (s = IncompleteSentence.Zhuangzi → completePassage s = "水之积也不厚") ∧
    (s = IncompleteSentence.TaoYuanming → completePassage s = "仰而视之") ∧
    (s = IncompleteSentence.LiBai → completePassage s = "扶疏荫初上") ∧
    (s = IncompleteSentence.SuShi → completePassage s = "举匏樽") ∧
    (s = IncompleteSentence.XinQiji → completePassage s = "骑鲸鱼") ∧
    (s = IncompleteSentence.Analects → completePassage s = "切问而近思") ∧
    (s = IncompleteSentence.LiuYuxi → completePassage s = "莫是银屏") :=
by sorry


end completePassage_correct_l3035_303520


namespace find_y_l3035_303503

theorem find_y : ∃ y : ℕ, (12^3 * 6^4) / y = 5184 ∧ y = 432 := by
  sorry

end find_y_l3035_303503


namespace complex_point_in_second_quadrant_l3035_303588

def is_in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem complex_point_in_second_quadrant (a : ℝ) (z : ℂ) 
  (h1 : z = a + Complex.I) 
  (h2 : Complex.abs z < Real.sqrt 2) : 
  is_in_second_quadrant (a - 1) 1 := by
  sorry

#check complex_point_in_second_quadrant

end complex_point_in_second_quadrant_l3035_303588


namespace number_exceeding_percentage_l3035_303584

theorem number_exceeding_percentage : 
  ∃ (x : ℝ), x = 200 ∧ x = 0.25 * x + 150 := by
  sorry

end number_exceeding_percentage_l3035_303584


namespace wood_per_sack_l3035_303516

/-- Given that 4 sacks were filled with a total of 80 pieces of wood,
    prove that each sack contains 20 pieces of wood. -/
theorem wood_per_sack (total_wood : ℕ) (num_sacks : ℕ) 
  (h1 : total_wood = 80) (h2 : num_sacks = 4) :
  total_wood / num_sacks = 20 := by
  sorry

end wood_per_sack_l3035_303516


namespace simons_blueberry_pies_l3035_303521

/-- Simon's blueberry pie problem -/
theorem simons_blueberry_pies :
  ∀ (own_blueberries nearby_blueberries blueberries_per_pie : ℕ),
    own_blueberries = 100 →
    nearby_blueberries = 200 →
    blueberries_per_pie = 100 →
    (own_blueberries + nearby_blueberries) / blueberries_per_pie = 3 :=
by
  sorry

#check simons_blueberry_pies

end simons_blueberry_pies_l3035_303521


namespace sum_of_digits_11_pow_2010_l3035_303587

/-- The sum of the tens digit and the units digit in the decimal representation of 11^2010 is 1. -/
theorem sum_of_digits_11_pow_2010 : ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 11^2010 % 100 = 10 * a + b ∧ a + b = 1 := by
  sorry

end sum_of_digits_11_pow_2010_l3035_303587


namespace rectangular_field_area_l3035_303559

/-- Given a rectangular field with perimeter 120 meters and length three times the width,
    prove that its area is 675 square meters. -/
theorem rectangular_field_area (l w : ℝ) : 
  (2 * l + 2 * w = 120) → 
  (l = 3 * w) → 
  (l * w = 675) := by
  sorry

end rectangular_field_area_l3035_303559


namespace factory_equation_correctness_l3035_303507

/-- Represents the factory worker assignment problem -/
def factory_problem (x y : ℕ) : Prop :=
  -- Total number of workers is 95
  x + y = 95 ∧
  -- Production ratio for sets (2 nuts : 1 screw)
  16 * x = 22 * y

/-- The system of linear equations correctly represents the factory problem -/
theorem factory_equation_correctness :
  ∀ x y : ℕ,
  factory_problem x y ↔ 
  (x + y = 95 ∧ 16 * x - 22 * y = 0) :=
by sorry

end factory_equation_correctness_l3035_303507


namespace absolute_value_inequality_solution_l3035_303528

theorem absolute_value_inequality_solution :
  {y : ℝ | 3 ≤ |y - 4| ∧ |y - 4| ≤ 7} = {y : ℝ | (7 ≤ y ∧ y ≤ 11) ∨ (-3 ≤ y ∧ y ≤ 1)} := by
  sorry

end absolute_value_inequality_solution_l3035_303528


namespace mango_rate_per_kg_l3035_303534

/-- The rate per kg for mangoes given the purchase details --/
theorem mango_rate_per_kg (grape_quantity grape_rate mango_quantity total_payment : ℕ) : 
  grape_quantity = 9 →
  grape_rate = 70 →
  mango_quantity = 9 →
  total_payment = 1125 →
  (total_payment - grape_quantity * grape_rate) / mango_quantity = 55 := by
sorry

end mango_rate_per_kg_l3035_303534


namespace johnny_planks_needed_l3035_303550

/-- Calculates the number of planks needed to build tables. -/
def planks_needed (num_tables : ℕ) (planks_per_leg : ℕ) (legs_per_table : ℕ) (planks_for_surface : ℕ) : ℕ :=
  num_tables * (legs_per_table * planks_per_leg + planks_for_surface)

/-- Theorem: Johnny needs 45 planks to build 5 tables. -/
theorem johnny_planks_needed : 
  planks_needed 5 1 4 5 = 45 := by
sorry

end johnny_planks_needed_l3035_303550


namespace BD_expression_A_B_D_collinear_l3035_303560

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the non-collinear vectors a and b
variable (a b : V)
variable (h_non_collinear : a ≠ 0 ∧ b ≠ 0 ∧ ¬∃ (r : ℝ), a = r • b)

-- Define the vectors AB, OB, and OD
def AB (a b : V) : V := 2 • a - 8 • b
def OB (a b : V) : V := a + 3 • b
def OD (a b : V) : V := 2 • a - b

-- Statement 1: Express BD in terms of a and b
theorem BD_expression (a b : V) : OD a b - OB a b = a - 4 • b := by sorry

-- Statement 2: Prove that A, B, and D are collinear
theorem A_B_D_collinear (a b : V) : 
  ∃ (r : ℝ), AB a b = r • (OD a b - OB a b) := by sorry

end BD_expression_A_B_D_collinear_l3035_303560


namespace graph_is_two_lines_l3035_303523

/-- The equation of the graph -/
def equation (x y : ℝ) : Prop := x^2 - 25 * y^2 - 20 * x + 100 = 0

/-- Definition of a line in slope-intercept form -/
def is_line (m b : ℝ) (x y : ℝ) : Prop := y = m * x + b

/-- The graph represents two lines -/
theorem graph_is_two_lines :
  ∃ (m₁ b₁ m₂ b₂ : ℝ), 
    (∀ x y, equation x y ↔ (is_line m₁ b₁ x y ∨ is_line m₂ b₂ x y)) ∧
    m₁ ≠ m₂ :=
sorry

end graph_is_two_lines_l3035_303523


namespace inequality_solution_set_l3035_303508

theorem inequality_solution_set : 
  {x : ℕ | 1 + x ≥ 2 * x - 1} = {0, 1, 2} := by sorry

end inequality_solution_set_l3035_303508


namespace purely_imaginary_complex_number_l3035_303512

theorem purely_imaginary_complex_number (x : ℝ) : 
  (Complex.ofReal (x^2 - 1) + Complex.I * Complex.ofReal (x + 1)).im ≠ 0 ∧
  (Complex.ofReal (x^2 - 1) + Complex.I * Complex.ofReal (x + 1)).re = 0 →
  x = 1 := by
sorry

end purely_imaginary_complex_number_l3035_303512


namespace wire_length_around_square_field_l3035_303575

theorem wire_length_around_square_field (area : ℝ) (rounds : ℕ) 
  (h1 : area = 69696) 
  (h2 : rounds = 15) : 
  Real.sqrt area * 4 * rounds = 15840 := by
  sorry

end wire_length_around_square_field_l3035_303575


namespace event_probability_l3035_303515

theorem event_probability (p : ℝ) : 
  (0 ≤ p ∧ p ≤ 1) →
  (1 - p)^3 = 1 - 63/64 →
  3 * p * (1 - p)^2 = 9/64 :=
by sorry

end event_probability_l3035_303515


namespace yellow_sweets_count_l3035_303522

theorem yellow_sweets_count (green_sweets blue_sweets total_sweets : ℕ) 
  (h1 : green_sweets = 212)
  (h2 : blue_sweets = 310)
  (h3 : total_sweets = 1024) : 
  total_sweets - (green_sweets + blue_sweets) = 502 := by
  sorry

end yellow_sweets_count_l3035_303522


namespace parallel_line_equation_l3035_303552

/-- A line passing through a point and parallel to another line -/
theorem parallel_line_equation (x y : ℝ) : 
  (x - 2*y + 7 = 0) ↔ 
  (∃ (m b : ℝ), y = m*x + b ∧ m = (1/2) ∧ y = m*(x+1) + 3) := by
  sorry

end parallel_line_equation_l3035_303552


namespace find_N_l3035_303594

theorem find_N (a b c N : ℚ) 
  (sum_eq : a + b + c = 120)
  (a_eq : a - 10 = N)
  (b_eq : 10 * b = N)
  (c_eq : c - 10 = N) :
  N = 1100 / 21 := by
sorry

end find_N_l3035_303594


namespace complement_of_M_l3035_303511

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 5}

theorem complement_of_M : 
  (U \ M) = {3, 4, 6} := by sorry

end complement_of_M_l3035_303511


namespace f_nonnegative_and_a_range_f_unique_zero_l3035_303590

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x - 1 / (x + a)

theorem f_nonnegative_and_a_range (a : ℝ) (h : a > 0) :
  (∀ x > 0, f a x ≥ 0) ∧ a ≥ 1 := by sorry

theorem f_unique_zero (a : ℝ) (h : 0 < a ∧ a ≤ 2/3) :
  ∃! x, x > -a ∧ f a x = 0 := by sorry

end f_nonnegative_and_a_range_f_unique_zero_l3035_303590


namespace fraction_zero_implies_a_equals_one_l3035_303558

theorem fraction_zero_implies_a_equals_one (a : ℝ) : 
  (|a| - 1) / (a + 1) = 0 → a = 1 := by
  sorry

end fraction_zero_implies_a_equals_one_l3035_303558


namespace jia_zi_second_occurrence_l3035_303519

/-- The number of Heavenly Stems -/
def heavenly_stems : ℕ := 10

/-- The number of Earthly Branches -/
def earthly_branches : ℕ := 12

/-- The column number when Jia and Zi are in the same column for the second time -/
def second_occurrence : ℕ := 61

/-- Proves that the column number when Jia and Zi are in the same column for the second time is 61 -/
theorem jia_zi_second_occurrence :
  second_occurrence = Nat.lcm heavenly_stems earthly_branches + 1 := by
  sorry

end jia_zi_second_occurrence_l3035_303519


namespace not_both_perfect_squares_l3035_303513

/-- For any natural numbers x and y, at least one of x^2 + y + 1 or y^2 + 4x + 3 is not a perfect square. -/
theorem not_both_perfect_squares (x y : ℕ) : 
  ¬(∃ a b : ℕ, (x^2 + y + 1 = a^2) ∧ (y^2 + 4*x + 3 = b^2)) := by
  sorry

end not_both_perfect_squares_l3035_303513


namespace system_solution_l3035_303501

theorem system_solution (x y : ℝ) (eq1 : x + 2*y = 6) (eq2 : 2*x + y = 21) : x + y = 9 := by
  sorry

end system_solution_l3035_303501


namespace four_number_average_l3035_303532

theorem four_number_average (a b c d : ℝ) 
  (h1 : b + c + d = 24)
  (h2 : a + c + d = 36)
  (h3 : a + b + d = 28)
  (h4 : a + b + c = 32) :
  (a + b + c + d) / 4 = 10 := by
sorry

end four_number_average_l3035_303532


namespace dishwasher_manager_wage_ratio_l3035_303509

/-- Proves that the ratio of dishwasher's wage to manager's wage is 0.5 -/
theorem dishwasher_manager_wage_ratio :
  ∀ (dishwasher_wage chef_wage manager_wage : ℝ),
  manager_wage = 7.5 →
  chef_wage = manager_wage - 3 →
  chef_wage = dishwasher_wage * 1.2 →
  dishwasher_wage / manager_wage = 0.5 := by
  sorry

end dishwasher_manager_wage_ratio_l3035_303509


namespace platform_length_calculation_l3035_303530

/-- Calculates the length of a platform given train parameters -/
theorem platform_length_calculation (train_length : ℝ) (time_platform : ℝ) (time_pole : ℝ) :
  train_length = 300 →
  time_platform = 39 →
  time_pole = 18 →
  ∃ platform_length : ℝ,
    (platform_length > 350.12 ∧ platform_length < 350.14) ∧
    platform_length = train_length * (time_platform / time_pole - 1) :=
by
  sorry

#check platform_length_calculation

end platform_length_calculation_l3035_303530


namespace largest_divisor_of_n_squared_div_72_l3035_303549

theorem largest_divisor_of_n_squared_div_72 (n : ℕ) (h : n > 0) (h_div : 72 ∣ n^2) :
  ∀ k : ℕ, k > 12 → ¬(∀ m : ℕ, m > 0 ∧ 72 ∣ m^2 → k ∣ m) :=
by sorry

end largest_divisor_of_n_squared_div_72_l3035_303549


namespace board_division_impossibility_l3035_303564

theorem board_division_impossibility : ¬ ∃ (triangle_area : ℚ),
  (63 : ℚ) = 17 * triangle_area ∧
  ∃ (side_length : ℚ), 
    triangle_area = (side_length * side_length * Real.sqrt 3) / 4 ∧
    0 < side_length ∧
    side_length ≤ 8 := by
  sorry

end board_division_impossibility_l3035_303564


namespace parallel_lines_distance_l3035_303599

/-- A circle intersected by three equally spaced parallel lines -/
structure CircleWithParallelLines where
  /-- The radius of the circle -/
  r : ℝ
  /-- The distance between adjacent parallel lines -/
  d : ℝ
  /-- The length of the first chord -/
  chord1 : ℝ
  /-- The length of the second chord -/
  chord2 : ℝ
  /-- The length of the third chord -/
  chord3 : ℝ
  /-- The first chord has length 42 -/
  chord1_eq : chord1 = 42
  /-- The second chord has length 42 -/
  chord2_eq : chord2 = 42
  /-- The third chord has length 40 -/
  chord3_eq : chord3 = 40

/-- The theorem stating that the distance between adjacent parallel lines is 3 3/8 -/
theorem parallel_lines_distance (c : CircleWithParallelLines) : c.d = 3 + 3 / 8 := by
  sorry

end parallel_lines_distance_l3035_303599


namespace jade_transactions_l3035_303561

theorem jade_transactions (mabel anthony cal jade : ℕ) : 
  mabel = 90 →
  anthony = mabel + mabel / 10 →
  cal = anthony * 2 / 3 →
  jade = cal + 17 →
  jade = 83 := by
  sorry

end jade_transactions_l3035_303561


namespace sum_of_solutions_l3035_303562

theorem sum_of_solutions (x y : ℝ) (h1 : x + 6 * y = 12) (h2 : 3 * x - 2 * y = 8) : x + y = 5 := by
  sorry

end sum_of_solutions_l3035_303562


namespace equation_solution_l3035_303585

theorem equation_solution (x : ℝ) : 
  (Real.sqrt (x + 15) - 7 / Real.sqrt (x + 15) = 4) ↔ 
  (x = 15 + 4 * Real.sqrt 11 ∨ x = 15 - 4 * Real.sqrt 11) := by
sorry

end equation_solution_l3035_303585


namespace sufficient_but_not_necessary_l3035_303506

theorem sufficient_but_not_necessary (p q : Prop) 
  (h : (¬p → ¬q) ∧ ¬(¬q → ¬p)) : 
  (q → p) ∧ ¬(p → q) := by
  sorry

end sufficient_but_not_necessary_l3035_303506


namespace quadratic_inequality_solution_set_l3035_303525

theorem quadratic_inequality_solution_set (m : ℝ) (h : m * (m - 1) < 0) :
  {x : ℝ | x^2 - (m + 1/m) * x + 1 < 0} = {x : ℝ | m < x ∧ x < 1/m} := by
  sorry

end quadratic_inequality_solution_set_l3035_303525


namespace min_value_reciprocal_product_l3035_303505

theorem min_value_reciprocal_product (a b : ℝ) 
  (h1 : a + a * b + 2 * b = 30) 
  (h2 : a > 0) 
  (h3 : b > 0) : 
  ∀ x y : ℝ, x > 0 → y > 0 → x + x * y + 2 * y = 30 → 1 / (a * b) ≤ 1 / (x * y) := by
  sorry

end min_value_reciprocal_product_l3035_303505


namespace seating_arrangements_count_l3035_303551

/-- Represents a seating arrangement around a round table -/
def SeatingArrangement := Fin 12 → Fin 12

/-- Checks if two positions are adjacent on a round table with 12 seats -/
def are_adjacent (a b : Fin 12) : Prop :=
  (a + 1 = b) ∨ (b + 1 = a) ∨ (a = 11 ∧ b = 0) ∨ (a = 0 ∧ b = 11)

/-- Checks if two positions are across from each other on a round table with 12 seats -/
def are_across (a b : Fin 12) : Prop := (a + 6 = b) ∨ (b + 6 = a)

/-- Checks if a seating arrangement is valid according to the problem constraints -/
def is_valid_arrangement (arr : SeatingArrangement) (couples : Fin 6 → Fin 12 × Fin 12) : Prop :=
  ∀ i j : Fin 12,
    (i ≠ j) →
    (¬are_adjacent (arr i) (arr j)) ∧
    (¬are_across (arr i) (arr j)) ∧
    (∀ k : Fin 6, (couples k).1 ≠ i ∨ (couples k).2 ≠ j)

/-- The main theorem stating the number of valid seating arrangements -/
theorem seating_arrangements_count :
  ∃ (arrangements : Finset SeatingArrangement) (couples : Fin 6 → Fin 12 × Fin 12),
    (∀ arr ∈ arrangements, is_valid_arrangement arr couples) ∧
    arrangements.card = 1440 := by
  sorry

end seating_arrangements_count_l3035_303551


namespace cubic_root_sum_cubes_l3035_303598

theorem cubic_root_sum_cubes (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) → 
  (b^3 - 2*b^2 + 3*b - 4 = 0) → 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  a^3 + b^3 + c^3 = 2 := by
  sorry

end cubic_root_sum_cubes_l3035_303598


namespace circus_tent_capacity_l3035_303517

/-- The number of sections in the circus tent -/
def num_sections : ℕ := 4

/-- The capacity of each section in the circus tent -/
def section_capacity : ℕ := 246

/-- The total capacity of the circus tent -/
def total_capacity : ℕ := num_sections * section_capacity

theorem circus_tent_capacity : total_capacity = 984 := by
  sorry

end circus_tent_capacity_l3035_303517


namespace triangle_problem_l3035_303533

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : (2 * t.b - t.c) * Real.cos t.A = t.a * Real.cos t.C)
  (h2 : t.a = Real.sqrt 13)
  (h3 : t.b + t.c = 5) :
  t.A = π / 3 ∧ 
  (1 / 2 : ℝ) * t.b * t.c * Real.sin t.A = Real.sqrt 3 := by
  sorry


end triangle_problem_l3035_303533


namespace adam_total_earnings_l3035_303502

/-- Calculates Adam's earnings given task rates, completion numbers, and exchange rates -/
def adam_earnings (dollar_per_lawn : ℝ) (euro_per_car : ℝ) (peso_per_dog : ℝ)
                  (lawns_total : ℕ) (cars_total : ℕ) (dogs_total : ℕ)
                  (lawns_forgot : ℕ) (cars_forgot : ℕ) (dogs_forgot : ℕ)
                  (euro_to_dollar : ℝ) (peso_to_dollar : ℝ) : ℝ :=
  let lawns_done := lawns_total - lawns_forgot
  let cars_done := cars_total - cars_forgot
  let dogs_done := dogs_total - dogs_forgot
  
  let lawn_earnings := dollar_per_lawn * lawns_done
  let car_earnings := euro_per_car * cars_done * euro_to_dollar
  let dog_earnings := peso_per_dog * dogs_done * peso_to_dollar
  
  lawn_earnings + car_earnings + dog_earnings

/-- Theorem stating Adam's earnings based on given conditions -/
theorem adam_total_earnings :
  adam_earnings 9 10 50 12 6 4 8 2 1 1.1 0.05 = 87.5 := by
  sorry

#eval adam_earnings 9 10 50 12 6 4 8 2 1 1.1 0.05

end adam_total_earnings_l3035_303502


namespace factorial_fraction_l3035_303510

theorem factorial_fraction (N : ℕ) :
  (Nat.factorial (N + 1)) / ((Nat.factorial (N + 2)) + (Nat.factorial N)) = 
  (N + 1) / (N^2 + 3*N + 3) := by
sorry

end factorial_fraction_l3035_303510


namespace tenby_position_l3035_303579

def letters : List Char := ['B', 'E', 'N', 'T', 'Y']

def word : String := "TENBY"

def alphabetical_position (w : String) (l : List Char) : ℕ :=
  sorry

theorem tenby_position :
  alphabetical_position word letters = 75 := by
  sorry

end tenby_position_l3035_303579


namespace point_not_in_second_quadrant_l3035_303544

theorem point_not_in_second_quadrant (n : ℝ) : ¬(n + 1 < 0 ∧ 2*n - 1 > 0) := by
  sorry

end point_not_in_second_quadrant_l3035_303544


namespace total_money_proof_l3035_303526

/-- Given the money distribution among Cecil, Catherine, and Carmela, 
    prove that their total money is $2800 -/
theorem total_money_proof (cecil_money : ℕ) 
  (h1 : cecil_money = 600)
  (catherine_money : ℕ) 
  (h2 : catherine_money = 2 * cecil_money - 250)
  (carmela_money : ℕ) 
  (h3 : carmela_money = 2 * cecil_money + 50) : 
  cecil_money + catherine_money + carmela_money = 2800 := by
  sorry

end total_money_proof_l3035_303526


namespace valid_numbers_count_l3035_303531

/-- Counts the number of valid eight-digit numbers where each digit appears exactly as many times as its value. -/
def count_valid_numbers : ℕ :=
  let single_eight := 1
  let seven_sevens_one_one := 8
  let six_sixes_two_twos := 28
  let five_fives_two_twos_one_one := 168
  let five_fives_three_threes := 56
  let four_fours_three_threes_one_one := 280
  single_eight + seven_sevens_one_one + six_sixes_two_twos + 
  five_fives_two_twos_one_one + five_fives_three_threes + 
  four_fours_three_threes_one_one

theorem valid_numbers_count : count_valid_numbers = 541 := by
  sorry

end valid_numbers_count_l3035_303531


namespace rectangle_area_diagonal_relation_l3035_303543

theorem rectangle_area_diagonal_relation :
  ∀ (length width : ℝ),
  length > 0 ∧ width > 0 →
  length / width = 5 / 2 →
  2 * (length + width) = 56 →
  ∃ (d : ℝ),
  d^2 = length^2 + width^2 ∧
  length * width = (10/29) * d^2 :=
by sorry

end rectangle_area_diagonal_relation_l3035_303543


namespace vector_norm_condition_l3035_303566

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- Given non-zero vectors a and b, a = -2b is a sufficient but not necessary condition
    for |a| - |b| = |a + b| --/
theorem vector_norm_condition (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a = -2 • b → ‖a‖ - ‖b‖ = ‖a + b‖) ∧
  ¬(‖a‖ - ‖b‖ = ‖a + b‖ → a = -2 • b) :=
sorry

end vector_norm_condition_l3035_303566


namespace max_students_is_eight_l3035_303555

/-- Represents the relation of two students knowing each other -/
def knows (n : ℕ) : (Fin n → Fin n → Prop) := sorry

/-- The property that in any group of 3 students, at least 2 know each other -/
def three_two_know (n : ℕ) (knows : Fin n → Fin n → Prop) : Prop :=
  ∀ (a b c : Fin n), a ≠ b ∧ b ≠ c ∧ a ≠ c →
    knows a b ∨ knows b c ∨ knows a c

/-- The property that in any group of 4 students, at least 2 do not know each other -/
def four_two_dont_know (n : ℕ) (knows : Fin n → Fin n → Prop) : Prop :=
  ∀ (a b c d : Fin n), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ a ≠ c ∧ a ≠ d ∧ b ≠ d →
    ¬(knows a b) ∨ ¬(knows a c) ∨ ¬(knows a d) ∨
    ¬(knows b c) ∨ ¬(knows b d) ∨ ¬(knows c d)

/-- The maximum number of students satisfying the conditions is 8 -/
theorem max_students_is_eight :
  (∃ (n : ℕ), n = 8 ∧
    three_two_know n (knows n) ∧
    four_two_dont_know n (knows n)) ∧
  (∀ (m : ℕ), m > 8 →
    ¬(three_two_know m (knows m) ∧
      four_two_dont_know m (knows m))) :=
by sorry

end max_students_is_eight_l3035_303555
