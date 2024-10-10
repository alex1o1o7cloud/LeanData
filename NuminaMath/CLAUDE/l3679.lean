import Mathlib

namespace angle_bisector_equation_l3679_367984

/-- Given three lines y = x, y = 3x, and y = -x intersecting at the origin,
    the angle bisector of the smallest acute angle that passes through (1, 1)
    has the equation y = (2 - √11/2)x -/
theorem angle_bisector_equation (x y : ℝ) :
  let line1 : ℝ → ℝ := λ t => t
  let line2 : ℝ → ℝ := λ t => 3 * t
  let line3 : ℝ → ℝ := λ t => -t
  let bisector : ℝ → ℝ := λ t => (2 - Real.sqrt 11 / 2) * t
  (∀ t, line1 t = t ∧ line2 t = 3 * t ∧ line3 t = -t) →
  (bisector 0 = 0) →
  (bisector 1 = 1) →
  (∀ t, bisector t = (2 - Real.sqrt 11 / 2) * t) :=
by sorry

end angle_bisector_equation_l3679_367984


namespace tunnel_safety_condition_l3679_367991

def height_limit : ℝ := 4.5

def can_pass_safely (h : ℝ) : Prop := h ≤ height_limit

theorem tunnel_safety_condition (h : ℝ) :
  can_pass_safely h ↔ h ≤ height_limit :=
sorry

end tunnel_safety_condition_l3679_367991


namespace solve_equation_l3679_367930

theorem solve_equation (x : ℚ) : (4 / 7) * (1 / 5) * x = 12 → x = 105 := by
  sorry

end solve_equation_l3679_367930


namespace complement_intersection_equals_set_l3679_367947

-- Define the universal set U
def U : Set Int := {-1, 1, 2, 3}

-- Define set A
def A : Set Int := {-1, 2}

-- Define set B
def B : Set Int := {x : Int | x^2 - 2*x - 3 = 0}

-- Theorem statement
theorem complement_intersection_equals_set : 
  (U \ (A ∩ B)) = {1, 2, 3} := by sorry

end complement_intersection_equals_set_l3679_367947


namespace change_in_expression_l3679_367986

theorem change_in_expression (x a : ℝ) (k : ℝ) (h : k > 0) :
  let f := fun x => 3 * x^2 - k
  (f (x + a) - f x = 6 * a * x + 3 * a^2) ∧
  (f (x - a) - f x = -6 * a * x + 3 * a^2) :=
by sorry

end change_in_expression_l3679_367986


namespace eggs_per_tray_l3679_367924

theorem eggs_per_tray (total_trays : ℕ) (total_eggs : ℕ) (eggs_per_tray : ℕ) : 
  total_trays = 7 →
  total_eggs = 70 →
  total_eggs = total_trays * eggs_per_tray →
  eggs_per_tray = 10 := by
sorry

end eggs_per_tray_l3679_367924


namespace calculation_proof_l3679_367969

theorem calculation_proof : 1 - (1/2)⁻¹ * Real.sin (π/3) + |2^0 - Real.sqrt 3| = 0 := by
  sorry

end calculation_proof_l3679_367969


namespace absolute_value_even_and_increasing_l3679_367935

def f (x : ℝ) := abs x

theorem absolute_value_even_and_increasing :
  (∀ x : ℝ, f x = f (-x)) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end absolute_value_even_and_increasing_l3679_367935


namespace cos_half_alpha_l3679_367942

theorem cos_half_alpha (α : Real) 
  (h1 : 25 * (Real.sin α)^2 + Real.sin α - 24 = 0)
  (h2 : π / 2 < α ∧ α < π) :
  Real.cos (α / 2) = 3 / 5 ∨ Real.cos (α / 2) = -3 / 5 := by
  sorry

end cos_half_alpha_l3679_367942


namespace dara_jane_age_ratio_l3679_367918

-- Define the given conditions
def minimum_employment_age : ℕ := 25
def jane_current_age : ℕ := 28
def years_until_dara_minimum_age : ℕ := 14
def years_in_future : ℕ := 6

-- Define Dara's current age
def dara_current_age : ℕ := minimum_employment_age - years_until_dara_minimum_age

-- Define Dara's and Jane's ages in 6 years
def dara_future_age : ℕ := dara_current_age + years_in_future
def jane_future_age : ℕ := jane_current_age + years_in_future

-- Theorem to prove
theorem dara_jane_age_ratio : 
  dara_future_age * 2 = jane_future_age := by sorry

end dara_jane_age_ratio_l3679_367918


namespace meeting_point_distance_l3679_367946

theorem meeting_point_distance (total_distance : ℝ) (speed1 speed2 : ℝ) 
  (h1 : total_distance = 36)
  (h2 : speed1 = 2)
  (h3 : speed2 = 4) :
  speed1 * (total_distance / (speed1 + speed2)) = 12 :=
by sorry

end meeting_point_distance_l3679_367946


namespace probability_real_roots_l3679_367956

/-- The probability that the equation x^2 - mx + 4 = 0 has real roots,
    given that m is uniformly distributed in the interval [0, 6]. -/
theorem probability_real_roots : ℝ := by
  sorry

#check probability_real_roots

end probability_real_roots_l3679_367956


namespace line_slope_intercept_product_l3679_367917

theorem line_slope_intercept_product (m b : ℝ) : m = 3/4 ∧ b = -2 → m * b < -1 := by
  sorry

end line_slope_intercept_product_l3679_367917


namespace first_complete_coverage_l3679_367949

/-- Triangular number function -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Function to check if all remainders modulo 12 have been covered -/
def allRemaindersCovered (n : ℕ) : Prop :=
  ∀ r : Fin 12, ∃ k ≤ n, triangular k % 12 = r

/-- The main theorem -/
theorem first_complete_coverage :
  (allRemaindersCovered 19 ∧ ∀ m < 19, ¬allRemaindersCovered m) :=
sorry

end first_complete_coverage_l3679_367949


namespace grunters_win_probability_l3679_367988

theorem grunters_win_probability : 
  let n_games : ℕ := 6
  let p_first_half : ℚ := 3/4
  let p_second_half : ℚ := 4/5
  let n_first_half : ℕ := 3
  let n_second_half : ℕ := 3
  
  (n_first_half + n_second_half = n_games) →
  (p_first_half ^ n_first_half * p_second_half ^ n_second_half = 27/125) :=
by sorry

end grunters_win_probability_l3679_367988


namespace problem_statement_l3679_367994

theorem problem_statement (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (abs (x + 2*y) + abs (x - y) ≤ 5/2 ↔ 1/6 ≤ x ∧ x < 1) ∧
  ((1/x^2 - 1) * (1/y^2 - 1) ≥ 9) := by sorry

end problem_statement_l3679_367994


namespace sara_wrapping_paper_l3679_367905

theorem sara_wrapping_paper (total_paper : ℚ) (num_presents : ℕ) (paper_per_present : ℚ) :
  total_paper = 1/2 →
  num_presents = 5 →
  total_paper = num_presents * paper_per_present →
  paper_per_present = 1/10 := by
  sorry

end sara_wrapping_paper_l3679_367905


namespace intersection_condition_chord_length_condition_l3679_367965

-- Define the ellipse and line
def ellipse (x y : ℝ) : Prop := 4 * x^2 + y^2 = 1
def line (x y m : ℝ) : Prop := y = x + m

-- Theorem for intersection condition
theorem intersection_condition (m : ℝ) :
  (∃ x y : ℝ, ellipse x y ∧ line x y m) ↔ -Real.sqrt 5 / 2 ≤ m ∧ m ≤ Real.sqrt 5 / 2 :=
sorry

-- Theorem for chord length condition
theorem chord_length_condition (m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧ 
    line x₁ y₁ m ∧ line x₂ y₂ m ∧
    Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4 * Real.sqrt 2 / 5) →
  m = 1/2 ∨ m = -1/2 :=
sorry

end intersection_condition_chord_length_condition_l3679_367965


namespace economics_class_question_l3679_367981

theorem economics_class_question (total_students : ℕ) 
  (q2_correct : ℕ) (not_taken : ℕ) (both_correct : ℕ) :
  total_students = 40 →
  q2_correct = 29 →
  not_taken = 10 →
  both_correct = 29 →
  ∃ (q1_correct : ℕ), q1_correct ≥ 29 :=
by sorry

end economics_class_question_l3679_367981


namespace first_digit_base_nine_of_2121122_base_three_l3679_367992

def base_three_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ i)) 0

def first_digit_base_nine (n : Nat) : Nat :=
  Nat.log 9 n

theorem first_digit_base_nine_of_2121122_base_three :
  let y : Nat := base_three_to_decimal [2, 2, 1, 1, 2, 1, 2]
  first_digit_base_nine y = 2 := by
  sorry

end first_digit_base_nine_of_2121122_base_three_l3679_367992


namespace correct_negation_of_existential_statement_l3679_367910

theorem correct_negation_of_existential_statement :
  (¬ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + 2 > 0) :=
by sorry

end correct_negation_of_existential_statement_l3679_367910


namespace purely_imaginary_value_l3679_367922

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem purely_imaginary_value (a : ℝ) :
  let z : ℂ := Complex.mk (a^2 + a - 2) (a^2 - 1)
  is_purely_imaginary z → a = -2 :=
by sorry

end purely_imaginary_value_l3679_367922


namespace pet_store_puppies_l3679_367978

/-- The number of puppies sold --/
def puppies_sold : ℕ := 39

/-- The number of cages used --/
def cages_used : ℕ := 3

/-- The number of puppies per cage --/
def puppies_per_cage : ℕ := 2

/-- The initial number of puppies in the pet store --/
def initial_puppies : ℕ := puppies_sold + cages_used * puppies_per_cage

theorem pet_store_puppies : initial_puppies = 45 := by
  sorry

end pet_store_puppies_l3679_367978


namespace group_size_proof_l3679_367944

theorem group_size_proof (total_paise : ℕ) (n : ℕ) : 
  total_paise = 7744 →
  n * n = total_paise →
  n = 88 := by
  sorry

end group_size_proof_l3679_367944


namespace ben_mm_count_l3679_367998

theorem ben_mm_count (bryan_skittles : ℕ) (difference : ℕ) (ben_mm : ℕ) : 
  bryan_skittles = 50 →
  difference = 30 →
  bryan_skittles = ben_mm + difference →
  ben_mm = 20 := by
sorry

end ben_mm_count_l3679_367998


namespace square_area_ratio_l3679_367908

/-- The ratio of the area of a square with side length 2y to the area of a square with side length 8y is 1/16 -/
theorem square_area_ratio (y : ℝ) (y_pos : y > 0) : 
  (2 * y)^2 / (8 * y)^2 = 1 / 16 := by
  sorry

end square_area_ratio_l3679_367908


namespace difference_equals_three_44ths_l3679_367979

/-- The decimal representation of 0.overline{81} -/
def repeating_decimal : ℚ := 9/11

/-- The decimal representation of 0.75 -/
def decimal_75 : ℚ := 3/4

/-- The theorem stating that the difference between 0.overline{81} and 0.75 is 3/44 -/
theorem difference_equals_three_44ths : 
  repeating_decimal - decimal_75 = 3/44 := by sorry

end difference_equals_three_44ths_l3679_367979


namespace sum_of_reciprocals_squared_l3679_367987

theorem sum_of_reciprocals_squared (a b c d : ℝ) : 
  a = 2 * Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 →
  b = -2 * Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 →
  c = 2 * Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 →
  d = -2 * Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 →
  (1/a + 1/b + 1/c + 1/d)^2 = 560 / 155432121 := by
  sorry

end sum_of_reciprocals_squared_l3679_367987


namespace complex_fraction_simplification_l3679_367934

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (6 - 3*i) / (-2 + 5*i) = (-27 : ℚ) / 29 - (24 : ℚ) / 29 * i :=
by sorry

end complex_fraction_simplification_l3679_367934


namespace triangle_area_l3679_367980

theorem triangle_area (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (b * Real.sin C + c * Real.sin B = 4 * a * Real.sin B * Real.sin C) →
  (b^2 + c^2 - a^2 = 8) →
  (1/2 * b * c * Real.sin A = 4 * Real.sqrt 3 / 3) :=
by sorry

end triangle_area_l3679_367980


namespace sum_in_base8_l3679_367961

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base-10 number to base-8 --/
def base10ToBase8 (n : ℕ) : ℕ := sorry

theorem sum_in_base8 : 
  base10ToBase8 (base8ToBase10 24 + base8ToBase10 157) = 203 := by sorry

end sum_in_base8_l3679_367961


namespace min_cut_length_40x30_paper_10x5_rect_l3679_367993

/-- Represents a rectangular piece of paper -/
structure Paper where
  width : ℕ
  height : ℕ

/-- Represents a rectangle to be cut out -/
structure CutRectangle where
  width : ℕ
  height : ℕ

/-- Calculates the minimum cut length required to extract a rectangle from a paper -/
def minCutLength (paper : Paper) (rect : CutRectangle) : ℕ :=
  sorry

/-- Theorem stating the minimum cut length for the given problem -/
theorem min_cut_length_40x30_paper_10x5_rect :
  let paper := Paper.mk 40 30
  let rect := CutRectangle.mk 10 5
  minCutLength paper rect = 40 := by sorry

end min_cut_length_40x30_paper_10x5_rect_l3679_367993


namespace final_number_is_365_l3679_367904

/-- Represents the skipping pattern for a single student -/
def skip_pattern (n : ℕ) : Bool :=
  n % 4 ≠ 2

/-- Applies the skipping pattern for a given number of students -/
def apply_skip_pattern (students : ℕ) (n : ℕ) : Bool :=
  match students with
  | 0 => true
  | k + 1 => skip_pattern n && apply_skip_pattern k (((n - 1) / 4) + 1)

/-- The main theorem stating that after 8 students apply the skipping pattern, 365 is the only number remaining -/
theorem final_number_is_365 :
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 1100 → (apply_skip_pattern 8 n ↔ n = 365) :=
by sorry

end final_number_is_365_l3679_367904


namespace circle_through_two_points_tangent_to_line_l3679_367995

-- Define the basic geometric objects
structure Point where
  x : ℝ
  y : ℝ

structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

structure Circle where
  center : Point
  radius : ℝ

-- Define the condition for a point to be on a line
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define the condition for a circle to pass through a point
def circlePassesThroughPoint (c : Circle) (p : Point) : Prop :=
  (c.center.x - p.x)^2 + (c.center.y - p.y)^2 = c.radius^2

-- Define the condition for a circle to be tangent to a line
def circleTangentToLine (c : Circle) (l : Line) : Prop :=
  ∃ p : Point, pointOnLine p l ∧ circlePassesThroughPoint c p ∧
  ∀ q : Point, pointOnLine q l → (c.center.x - q.x)^2 + (c.center.y - q.y)^2 ≥ c.radius^2

-- Theorem statement
theorem circle_through_two_points_tangent_to_line 
  (A B : Point) (l : Line) : 
  ∃ c : Circle, circlePassesThroughPoint c A ∧ 
                circlePassesThroughPoint c B ∧ 
                circleTangentToLine c l :=
sorry

end circle_through_two_points_tangent_to_line_l3679_367995


namespace kanul_raw_materials_expenditure_l3679_367960

/-- The problem of calculating Kanul's expenditure on raw materials -/
theorem kanul_raw_materials_expenditure
  (total : ℝ)
  (machinery : ℝ)
  (cash_percentage : ℝ)
  (h1 : total = 7428.57)
  (h2 : machinery = 200)
  (h3 : cash_percentage = 0.30)
  (h4 : ∃ (raw_materials : ℝ), raw_materials + machinery + cash_percentage * total = total) :
  ∃ (raw_materials : ℝ), raw_materials = 5000 :=
sorry

end kanul_raw_materials_expenditure_l3679_367960


namespace percentage_boys_playing_soccer_l3679_367912

theorem percentage_boys_playing_soccer 
  (total_students : ℕ) 
  (boys : ℕ) 
  (playing_soccer : ℕ) 
  (girls_not_playing : ℕ) 
  (h1 : total_students = 420)
  (h2 : boys = 296)
  (h3 : playing_soccer = 250)
  (h4 : girls_not_playing = 89)
  : (boys - (total_students - boys - girls_not_playing)) / playing_soccer * 100 = 86 := by
  sorry

end percentage_boys_playing_soccer_l3679_367912


namespace compute_expression_l3679_367996

theorem compute_expression : 
  18 * (140 / 2 + 30 / 4 + 12 / 20 + 2 / 3) = 1417.8 := by
  sorry

end compute_expression_l3679_367996


namespace remainder_eight_power_2002_mod_9_l3679_367901

theorem remainder_eight_power_2002_mod_9 : 8^2002 % 9 = 1 := by
  sorry

end remainder_eight_power_2002_mod_9_l3679_367901


namespace net_salary_average_is_correct_l3679_367928

/-- Represents Sharon's salary structure and performance scenarios -/
structure SalaryStructure where
  initial_salary : ℝ
  exceptional_increase : ℝ
  good_increase : ℝ
  average_increase : ℝ
  exceptional_bonus : ℝ
  good_bonus : ℝ
  federal_tax : ℝ
  state_tax : ℝ
  healthcare_deduction : ℝ

/-- Calculates the net salary for average performance -/
def net_salary_average (s : SalaryStructure) : ℝ :=
  let increased_salary := s.initial_salary * (1 + s.average_increase)
  let tax_deduction := increased_salary * (s.federal_tax + s.state_tax)
  increased_salary - tax_deduction - s.healthcare_deduction

/-- Theorem stating that the net salary for average performance is 497.40 -/
theorem net_salary_average_is_correct (s : SalaryStructure) 
    (h1 : s.initial_salary = 560)
    (h2 : s.average_increase = 0.15)
    (h3 : s.federal_tax = 0.10)
    (h4 : s.state_tax = 0.05)
    (h5 : s.healthcare_deduction = 50) :
    net_salary_average s = 497.40 := by
  sorry

#eval net_salary_average { 
  initial_salary := 560,
  exceptional_increase := 0.25,
  good_increase := 0.20,
  average_increase := 0.15,
  exceptional_bonus := 0.05,
  good_bonus := 0.03,
  federal_tax := 0.10,
  state_tax := 0.05,
  healthcare_deduction := 50
}

end net_salary_average_is_correct_l3679_367928


namespace polynomial_simplification_l3679_367900

theorem polynomial_simplification : 
  2021^4 - 4 * 2023^4 + 6 * 2025^4 - 4 * 2027^4 + 2029^4 = 384 := by
  sorry

end polynomial_simplification_l3679_367900


namespace largest_prime_divisor_test_l3679_367926

theorem largest_prime_divisor_test (n : ℕ) (h1 : 1000 ≤ n) (h2 : n ≤ 1100) :
  (∀ p : ℕ, Nat.Prime p → p ≤ 31 → ¬(p ∣ n)) →
  (∀ p : ℕ, Nat.Prime p → p < Real.sqrt (n : ℝ) → ¬(p ∣ n)) :=
by sorry

end largest_prime_divisor_test_l3679_367926


namespace time_to_earn_house_cost_l3679_367958

/-- Represents the financial situation of a man buying a house -/
structure HouseBuying where
  /-- The cost of the house -/
  houseCost : ℝ
  /-- Annual household expenses -/
  annualExpenses : ℝ
  /-- Annual savings -/
  annualSavings : ℝ
  /-- The man spends the same on expenses in 8 years as on savings in 12 years -/
  expensesSavingsRelation : 8 * annualExpenses = 12 * annualSavings
  /-- It takes 24 years to buy the house with all earnings -/
  buyingTime : houseCost = 24 * (annualExpenses + annualSavings)

/-- Theorem stating the time needed to earn the house cost -/
theorem time_to_earn_house_cost (hb : HouseBuying) :
  hb.houseCost / hb.annualSavings = 60 := by
  sorry

end time_to_earn_house_cost_l3679_367958


namespace exponent_calculation_l3679_367972

theorem exponent_calculation : (((18^15 / 18^14)^3 * 8^3) / 4^5) = 2916 := by
  sorry

end exponent_calculation_l3679_367972


namespace mod_equivalence_unique_solution_l3679_367959

theorem mod_equivalence_unique_solution : 
  ∃! n : ℤ, 0 ≤ n ∧ n < 137 ∧ 12345 ≡ n [ZMOD 137] := by sorry

end mod_equivalence_unique_solution_l3679_367959


namespace square_roots_problem_l3679_367983

theorem square_roots_problem (a : ℝ) (x : ℝ) (h1 : a > 0) 
  (h2 : (3*x - 2)^2 = a) (h3 : (5*x + 6)^2 = a) : a = 49/4 := by
  sorry

end square_roots_problem_l3679_367983


namespace circle_radius_is_sqrt_21_25_l3679_367950

-- Define the circle Ω
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define points P and Q
def P : ℝ × ℝ := (9, 17)
def Q : ℝ × ℝ := (18, 15)

-- Define the line y = 2
def line_y_2 (x : ℝ) : ℝ := 2

-- Theorem statement
theorem circle_radius_is_sqrt_21_25 (Ω : Circle) :
  P ∈ {p : ℝ × ℝ | (p.1 - Ω.center.1)^2 + (p.2 - Ω.center.2)^2 = Ω.radius^2} →
  Q ∈ {p : ℝ × ℝ | (p.1 - Ω.center.1)^2 + (p.2 - Ω.center.2)^2 = Ω.radius^2} →
  (∃ x : ℝ, (x, line_y_2 x) ∈ {p : ℝ × ℝ | ∃ t : ℝ, p = (P.1 + t * (P.2 - Ω.center.2), P.2 - t * (P.1 - Ω.center.1))} ∩
                               {p : ℝ × ℝ | ∃ t : ℝ, p = (Q.1 + t * (Q.2 - Ω.center.2), Q.2 - t * (Q.1 - Ω.center.1))}) →
  Ω.radius = Real.sqrt 21.25 :=
by sorry

end circle_radius_is_sqrt_21_25_l3679_367950


namespace blue_balloons_l3679_367932

theorem blue_balloons (total red green purple : ℕ) (h1 : total = 135) (h2 : red = 45) (h3 : green = 27) (h4 : purple = 32) :
  total - (red + green + purple) = 31 := by
  sorry

end blue_balloons_l3679_367932


namespace universal_rook_program_exists_l3679_367957

/-- Represents a position on the 8x8 chessboard --/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents a command for moving the rook --/
inductive Command
  | RIGHT
  | LEFT
  | UP
  | DOWN

/-- Represents a maze configuration on the 8x8 chessboard --/
def Maze := Set (Position × Position)

/-- Represents a program as a finite sequence of commands --/
def Program := List Command

/-- Function to determine if a square is accessible from a given position in a maze --/
def isAccessible (maze : Maze) (start finish : Position) : Prop := sorry

/-- Function to determine if a program visits all accessible squares from a given start position --/
def visitsAllAccessible (maze : Maze) (start : Position) (program : Program) : Prop := sorry

/-- The main theorem stating that there exists a program that works for all mazes and start positions --/
theorem universal_rook_program_exists :
  ∃ (program : Program),
    ∀ (maze : Maze) (start : Position),
      visitsAllAccessible maze start program := by sorry

end universal_rook_program_exists_l3679_367957


namespace cos_angle_F₁PF₂_l3679_367997

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 2 = 1

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define the foci (we don't know their exact coordinates, so we leave them abstract)
variables (F₁ F₂ : ℝ × ℝ)

-- Define point P on the ellipse
variable (P : ℝ × ℝ)

-- State that P is on the ellipse
axiom P_on_ellipse : ellipse P.1 P.2

-- Define the distance between O and P
axiom OP_distance : Real.sqrt ((P.1 - O.1)^2 + (P.2 - O.2)^2) = Real.sqrt 3

-- Theorem to prove
theorem cos_angle_F₁PF₂ : 
  ∃ (F₁ F₂ : ℝ × ℝ), 
    (F₁ ≠ F₂) ∧ 
    (∀ Q : ℝ × ℝ, ellipse Q.1 Q.2 → 
      Real.sqrt ((Q.1 - F₁.1)^2 + (Q.2 - F₁.2)^2) +
      Real.sqrt ((Q.1 - F₂.1)^2 + (Q.2 - F₂.2)^2) = 
      Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2)) →
    ((P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2)) / 
    (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * 
     Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2)) = 1/3 :=
by sorry

end cos_angle_F₁PF₂_l3679_367997


namespace domino_coloring_l3679_367999

/-- 
Given an m × n board where m and n are natural numbers and mn is even,
prove that the smallest non-negative integer V such that in each row,
the number of squares covered by red dominoes and the number of squares
covered by blue dominoes are each at most V, is equal to n.
-/
theorem domino_coloring (m n : ℕ) (h : Even (m * n)) : 
  ∃ V : ℕ, (∀ row_red row_blue : ℕ, row_red + row_blue = n → row_red ≤ V ∧ row_blue ≤ V) ∧
  (∀ W : ℕ, W < n → ∃ row_red row_blue : ℕ, row_red + row_blue = n ∧ (row_red > W ∨ row_blue > W)) :=
by sorry

end domino_coloring_l3679_367999


namespace mod_multiplication_equivalence_l3679_367967

theorem mod_multiplication_equivalence : 98 * 202 ≡ 71 [ZMOD 75] := by sorry

end mod_multiplication_equivalence_l3679_367967


namespace bookstore_revenue_theorem_l3679_367948

structure BookStore where
  total_books : ℕ
  novels : ℕ
  biographies : ℕ
  science_books : ℕ
  novel_price : ℚ
  biography_price : ℚ
  science_book_price : ℚ
  novel_discount : ℚ
  biography_discount : ℚ
  science_book_discount : ℚ
  remaining_novels : ℕ
  remaining_biographies : ℕ
  remaining_science_books : ℕ
  sales_tax : ℚ

def calculate_total_revenue (store : BookStore) : ℚ :=
  let sold_novels := store.novels - store.remaining_novels
  let sold_biographies := store.biographies - store.remaining_biographies
  let sold_science_books := store.science_books - store.remaining_science_books
  let novel_revenue := (sold_novels : ℚ) * store.novel_price * (1 - store.novel_discount)
  let biography_revenue := (sold_biographies : ℚ) * store.biography_price * (1 - store.biography_discount)
  let science_book_revenue := (sold_science_books : ℚ) * store.science_book_price * (1 - store.science_book_discount)
  let total_discounted_revenue := novel_revenue + biography_revenue + science_book_revenue
  total_discounted_revenue * (1 + store.sales_tax)

theorem bookstore_revenue_theorem (store : BookStore) 
  (h1 : store.total_books = 500)
  (h2 : store.novels + store.biographies + store.science_books = store.total_books)
  (h3 : store.novels - store.remaining_novels = (3 * store.novels) / 5)
  (h4 : store.biographies - store.remaining_biographies = (2 * store.biographies) / 3)
  (h5 : store.science_books - store.remaining_science_books = (7 * store.science_books) / 10)
  (h6 : store.novel_price = 8)
  (h7 : store.biography_price = 12)
  (h8 : store.science_book_price = 10)
  (h9 : store.novel_discount = 1/4)
  (h10 : store.biography_discount = 3/10)
  (h11 : store.science_book_discount = 1/5)
  (h12 : store.remaining_novels = 60)
  (h13 : store.remaining_biographies = 65)
  (h14 : store.remaining_science_books = 50)
  (h15 : store.sales_tax = 1/20)
  : calculate_total_revenue store = 2696.4 := by
  sorry

end bookstore_revenue_theorem_l3679_367948


namespace reduction_after_four_trials_l3679_367902

/-- The reduction factor for the 0.618 method -/
def golden_ratio_inverse : ℝ := 0.618

/-- The number of trials -/
def num_trials : ℕ := 4

/-- The reduction factor after n trials using the 0.618 method -/
def reduction_factor (n : ℕ) : ℝ := golden_ratio_inverse ^ (n - 1)

/-- Theorem stating the reduction factor after 4 trials -/
theorem reduction_after_four_trials :
  reduction_factor num_trials = golden_ratio_inverse ^ 3 := by
  sorry

end reduction_after_four_trials_l3679_367902


namespace negation_of_at_least_one_even_l3679_367940

theorem negation_of_at_least_one_even (a b c : ℕ) :
  (¬ (Even a ∨ Even b ∨ Even c)) ↔ (Odd a ∧ Odd b ∧ Odd c) := by
  sorry

end negation_of_at_least_one_even_l3679_367940


namespace inequality_proof_l3679_367968

theorem inequality_proof (k m n : ℕ+) (h1 : 1 < k) (h2 : k ≤ m) (h3 : m < n) :
  (1 + m.val : ℝ)^2 > (1 + n.val : ℝ)^m.val := by
  sorry

end inequality_proof_l3679_367968


namespace heptagon_angles_l3679_367963

/-- The number of sides in a heptagon -/
def n : ℕ := 7

/-- The measure of an interior angle of a regular heptagon -/
def interior_angle : ℚ := (5 * 180) / n

/-- The measure of an exterior angle of a regular heptagon -/
def exterior_angle : ℚ := 180 - interior_angle

theorem heptagon_angles :
  (interior_angle = (5 * 180) / n) ∧
  (exterior_angle = 180 - ((5 * 180) / n)) := by
  sorry

end heptagon_angles_l3679_367963


namespace arithmetic_sequence_sum_l3679_367907

/-- An arithmetic sequence with positive terms -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  positive : ∀ n, a n > 0
  arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The theorem statement -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h1 : seq.a 1 = 3)
  (h2 : seq.a 1 + seq.a 2 + seq.a 3 = 21) :
  seq.a 4 + seq.a 5 + seq.a 6 = 57 := by
  sorry

end arithmetic_sequence_sum_l3679_367907


namespace jamie_earnings_l3679_367916

def total_earnings (hourly_rate : ℝ) (days_per_week : ℕ) (hours_per_day : ℕ) (weeks_worked : ℕ) : ℝ :=
  hourly_rate * (days_per_week * hours_per_day * weeks_worked)

theorem jamie_earnings : 
  let hourly_rate : ℝ := 10
  let days_per_week : ℕ := 2
  let hours_per_day : ℕ := 3
  let weeks_worked : ℕ := 6
  total_earnings hourly_rate days_per_week hours_per_day weeks_worked = 360 := by
  sorry

end jamie_earnings_l3679_367916


namespace smallest_n_for_probability_condition_l3679_367945

theorem smallest_n_for_probability_condition : 
  (∃ n : ℕ+, (1 : ℚ) / (n * (n + 1)) < 1 / 2020 ∧ 
    ∀ m : ℕ+, m < n → (1 : ℚ) / (m * (m + 1)) ≥ 1 / 2020) ∧
  (∀ n : ℕ+, (1 : ℚ) / (n * (n + 1)) < 1 / 2020 ∧ 
    ∀ m : ℕ+, m < n → (1 : ℚ) / (m * (m + 1)) ≥ 1 / 2020 → n = 45) :=
by sorry

end smallest_n_for_probability_condition_l3679_367945


namespace min_value_of_expression_l3679_367951

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2) * (1 / b + 2) ≥ 16 ∧
  ((1 / a + 2) * (1 / b + 2) = 16 ↔ a = 1/2 ∧ b = 1/2) :=
by sorry

end min_value_of_expression_l3679_367951


namespace max_square_plots_for_given_field_l3679_367976

/-- Represents the dimensions of a rectangular field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available fencing and field dimensions -/
structure FencingProblem where
  field : FieldDimensions
  internalFencing : ℕ

/-- Calculates the maximum number of square plots given a fencing problem -/
def maxSquarePlots (problem : FencingProblem) : ℕ :=
  sorry

/-- The main theorem stating the solution to the specific problem -/
theorem max_square_plots_for_given_field :
  let problem : FencingProblem := {
    field := { width := 40, length := 60 },
    internalFencing := 2400
  }
  maxSquarePlots problem = 6 := by
  sorry

end max_square_plots_for_given_field_l3679_367976


namespace popcorn_profit_l3679_367974

def buying_price : ℝ := 4
def selling_price : ℝ := 8
def bags_sold : ℕ := 30

def profit_per_bag : ℝ := selling_price - buying_price
def total_profit : ℝ := bags_sold * profit_per_bag

theorem popcorn_profit : total_profit = 120 := by
  sorry

end popcorn_profit_l3679_367974


namespace problem_solution_l3679_367989

def f (a : ℝ) : ℝ → ℝ := fun x ↦ |x - a|

theorem problem_solution :
  (∃ a : ℝ, (∀ x : ℝ, f a x ≤ 2 ↔ 1 ≤ x ∧ x ≤ 5) ∧
   (∀ m : ℝ, (∀ x : ℝ, f 3 (2*x) + f 3 (x+2) ≥ m) ↔ m ≤ 1/2)) :=
by sorry

end problem_solution_l3679_367989


namespace inequality_system_solution_l3679_367985

theorem inequality_system_solution (x : ℝ) :
  (x - 1) * Real.log 2 + Real.log (2^(x + 1) + 1) < Real.log (7 * 2^x + 12) →
  Real.log (x + 2) / Real.log x > 2 →
  1 < x ∧ x < 2 := by
  sorry

end inequality_system_solution_l3679_367985


namespace binomial_expansion_sum_l3679_367952

theorem binomial_expansion_sum (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x : ℝ, (3*x - 2)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  a₀ + a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ + 6*a₆ = 82 := by
  sorry

end binomial_expansion_sum_l3679_367952


namespace inequality_properties_l3679_367913

theorem inequality_properties (a b : ℝ) (h : 1/a < 1/b ∧ 1/b < 0) :
  a^2 < b^2 ∧ a*b < b^2 ∧ a/b + b/a > 2 ∧ |a| + |b| = |a + b| := by
  sorry

end inequality_properties_l3679_367913


namespace red_balloons_count_l3679_367937

/-- Proves that the total number of red balloons after destruction is 40 -/
theorem red_balloons_count (fred_balloons sam_balloons dan_destroyed : ℝ) 
  (h1 : fred_balloons = 10)
  (h2 : sam_balloons = 46)
  (h3 : dan_destroyed = 16) :
  fred_balloons + sam_balloons - dan_destroyed = 40 := by
  sorry

#check red_balloons_count

end red_balloons_count_l3679_367937


namespace population_increase_rate_l3679_367975

def birth_rate : ℚ := 32 / 1000
def death_rate : ℚ := 11 / 1000

theorem population_increase_rate : 
  (birth_rate - death_rate) * 100 = 2.1 := by sorry

end population_increase_rate_l3679_367975


namespace local_value_of_three_is_300_l3679_367923

/-- Represents a four-digit number -/
structure FourDigitNumber where
  thousands : Nat
  hundreds : Nat
  tens : Nat
  ones : Nat
  is_valid : thousands ≥ 1 ∧ thousands ≤ 9 ∧
             hundreds ≥ 0 ∧ hundreds ≤ 9 ∧
             tens ≥ 0 ∧ tens ≤ 9 ∧
             ones ≥ 0 ∧ ones ≤ 9

/-- Calculate the local value of a digit given its place value -/
def localValue (digit : Nat) (placeValue : Nat) : Nat :=
  digit * placeValue

/-- Theorem: In the number 2345, if the sum of local values of all digits is 2345,
    then the local value of the digit 3 is 300 -/
theorem local_value_of_three_is_300 (n : FourDigitNumber)
    (h1 : n.thousands = 2 ∧ n.hundreds = 3 ∧ n.tens = 4 ∧ n.ones = 5)
    (h2 : localValue n.thousands 1000 + localValue n.hundreds 100 +
          localValue n.tens 10 + localValue n.ones 1 = 2345) :
    localValue n.hundreds 100 = 300 := by
  sorry

end local_value_of_three_is_300_l3679_367923


namespace expand_product_l3679_367906

theorem expand_product (x y : ℝ) : (3*x - 2) * (2*x + 4*y + 1) = 6*x^2 + 12*x*y - x - 8*y - 2 := by
  sorry

end expand_product_l3679_367906


namespace linear_equation_equivalence_l3679_367911

theorem linear_equation_equivalence (x y : ℝ) (h : x + 3 * y = 3) : 
  (x = 3 - 3 * y) ∧ (y = (3 - x) / 3) := by
  sorry

end linear_equation_equivalence_l3679_367911


namespace configuration_permutations_l3679_367914

/-- The number of distinct arrangements of the letters in "CONFIGURATION" -/
def configuration_arrangements : ℕ := 389188800

/-- The total number of letters in "CONFIGURATION" -/
def total_letters : ℕ := 13

/-- The number of times each of O, I, N, and U appears in "CONFIGURATION" -/
def repeated_letter_count : ℕ := 2

/-- The number of letters that repeat in "CONFIGURATION" -/
def repeating_letters : ℕ := 4

theorem configuration_permutations :
  configuration_arrangements = (Nat.factorial total_letters) / (Nat.factorial repeated_letter_count ^ repeating_letters) :=
sorry

end configuration_permutations_l3679_367914


namespace triangle_is_right_angle_l3679_367938

theorem triangle_is_right_angle (a b c : ℝ) : 
  a = 3 ∧ b = 4 ∧ c^2 - 10*c + 25 = 0 → c^2 = a^2 + b^2 := by
  sorry

end triangle_is_right_angle_l3679_367938


namespace sum_four_digit_distinct_remainder_l3679_367939

def T : ℕ := sorry

theorem sum_four_digit_distinct_remainder (T : ℕ) : T % 1000 = 960 :=
by
  sorry

end sum_four_digit_distinct_remainder_l3679_367939


namespace iphone_price_proof_l3679_367919

/-- The original price of an iPhone X -/
def original_price : ℝ := sorry

/-- The discount rate for buying at least 2 smartphones -/
def discount_rate : ℝ := 0.05

/-- The number of people buying iPhones -/
def num_buyers : ℕ := 3

/-- The amount saved by buying together -/
def amount_saved : ℝ := 90

theorem iphone_price_proof :
  (original_price * num_buyers * discount_rate = amount_saved) →
  original_price = 600 := by
  sorry

end iphone_price_proof_l3679_367919


namespace geralds_apples_count_l3679_367909

/-- Given that Pam has 10 bags of apples, 1200 apples in total, and each of her bags
    contains 3 times the number of apples in each of Gerald's bags, prove that
    each of Gerald's bags contains 40 apples. -/
theorem geralds_apples_count (pam_bags : ℕ) (pam_total_apples : ℕ) (gerald_apples : ℕ) 
  (h1 : pam_bags = 10)
  (h2 : pam_total_apples = 1200)
  (h3 : pam_total_apples = pam_bags * (3 * gerald_apples)) :
  gerald_apples = 40 := by
  sorry

end geralds_apples_count_l3679_367909


namespace solution_set_f_exp_pos_l3679_367954

-- Define the function f
noncomputable def f : ℝ → ℝ := sorry

-- Define the solution set of f(x) < 0
def solution_set_f_neg (x : ℝ) : Prop := x < -1 ∨ x > 1/3

-- Theorem statement
theorem solution_set_f_exp_pos :
  (∀ x, f x < 0 ↔ solution_set_f_neg x) →
  (∀ x, f (Real.exp x) > 0 ↔ x < -Real.log 3) :=
sorry

end solution_set_f_exp_pos_l3679_367954


namespace WXYZ_perimeter_l3679_367921

/-- Represents a rectangle with a perimeter --/
structure Rectangle where
  perimeter : ℕ

/-- Represents the large rectangle WXYZ --/
def WXYZ : Rectangle := sorry

/-- The four smaller rectangles that WXYZ is divided into --/
def smallRectangles : Fin 4 → Rectangle := sorry

/-- The sum of perimeters of diagonally opposite rectangles equals the perimeter of WXYZ --/
axiom perimeter_sum (i j : Fin 4) (h : i.val + j.val = 3) : 
  (smallRectangles i).perimeter + (smallRectangles j).perimeter = WXYZ.perimeter

/-- The perimeters of three of the smaller rectangles --/
axiom known_perimeters : 
  ∃ (i j k : Fin 4) (h : i ≠ j ∧ j ≠ k ∧ i ≠ k),
    (smallRectangles i).perimeter = 11 ∧
    (smallRectangles j).perimeter = 16 ∧
    (smallRectangles k).perimeter = 19

/-- The perimeter of the fourth rectangle is between 11 and 19 --/
axiom fourth_perimeter :
  ∃ (l : Fin 4), ∀ (i : Fin 4), 
    (smallRectangles i).perimeter ≠ 11 → 
    (smallRectangles i).perimeter ≠ 16 → 
    (smallRectangles i).perimeter ≠ 19 →
    11 < (smallRectangles l).perimeter ∧ (smallRectangles l).perimeter < 19

/-- The perimeter of WXYZ is 30 --/
theorem WXYZ_perimeter : WXYZ.perimeter = 30 := by sorry

end WXYZ_perimeter_l3679_367921


namespace circle_area_l3679_367962

theorem circle_area (r : ℝ) (h : r = 5) : π * r^2 = 25 * π := by
  sorry

end circle_area_l3679_367962


namespace right_triangle_hypotenuse_l3679_367966

/-- A right triangle with perimeter 40 and area 30 has a hypotenuse of length 74/4 -/
theorem right_triangle_hypotenuse : ∀ a b c : ℝ,
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- right triangle condition
  a + b + c = 40 →   -- perimeter condition
  a * b / 2 = 30 →   -- area condition
  c = 74 / 4 := by
    sorry

end right_triangle_hypotenuse_l3679_367966


namespace smallest_digit_for_divisibility_by_nine_l3679_367982

theorem smallest_digit_for_divisibility_by_nine :
  ∃ (d : ℕ), d < 10 ∧ 
  (∀ (x : ℕ), x < d → ¬(9 ∣ (438000 + x * 100 + 4))) ∧
  (9 ∣ (438000 + d * 100 + 4)) ∧
  d = 8 := by
  sorry

end smallest_digit_for_divisibility_by_nine_l3679_367982


namespace birthday_candles_l3679_367903

/-- Represents the ages of the seven children --/
structure ChildrenAges where
  youngest : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ
  fifth : ℕ
  sixth : ℕ
  seventh : ℕ

/-- The problem statement --/
theorem birthday_candles (ages : ChildrenAges) : 
  ages.youngest = 1 →
  ages.second = 2 →
  ages.third = 3 →
  ages.fourth = 4 →
  ages.fifth = 5 →
  ages.sixth = ages.seventh →
  (ages.youngest + ages.second + ages.third + ages.fourth + ages.fifth + ages.sixth + ages.seventh) = 
    2 * (ages.second - 1 + ages.third - 1 + ages.fourth - 1 + ages.fifth - 1 + ages.sixth - 1 + ages.seventh - 1) + 2 →
  (ages.youngest + ages.second + ages.third + ages.fourth + ages.fifth + ages.sixth + ages.seventh) = 27 := by
  sorry


end birthday_candles_l3679_367903


namespace isosceles_triangle_l3679_367970

theorem isosceles_triangle (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧          -- Sum of angles in a triangle
  c = 2 * a * Real.cos B → -- Given condition
  A = B                    -- Conclusion: triangle is isosceles
  := by sorry

end isosceles_triangle_l3679_367970


namespace edward_spending_l3679_367943

theorem edward_spending (initial : ℝ) : 
  initial > 0 →
  let after_clothes := initial - 250
  let after_food := after_clothes - (0.35 * after_clothes)
  let after_electronics := after_food - (0.5 * after_food)
  after_electronics = 200 →
  initial = 1875 := by sorry

end edward_spending_l3679_367943


namespace quadratic_rewrite_l3679_367925

theorem quadratic_rewrite (d e f : ℤ) : 
  (∀ x : ℝ, 4 * x^2 - 24 * x + 35 = (d * x + e)^2 + f) → 
  d * e = -12 := by
sorry

end quadratic_rewrite_l3679_367925


namespace smallest_num_with_digit_sum_2017_properties_first_digit_times_num_digits_l3679_367953

/-- The smallest natural number with digit sum 2017 -/
def smallest_num_with_digit_sum_2017 : ℕ :=
  1 * 10^224 + (10^224 - 1)

/-- The digit sum of a natural number -/
def digit_sum (n : ℕ) : ℕ :=
  sorry

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ :=
  sorry

theorem smallest_num_with_digit_sum_2017_properties :
  digit_sum smallest_num_with_digit_sum_2017 = 2017 ∧
  num_digits smallest_num_with_digit_sum_2017 = 225 ∧
  smallest_num_with_digit_sum_2017 < 10^225 ∧
  ∀ m : ℕ, m < smallest_num_with_digit_sum_2017 → digit_sum m ≠ 2017 :=
by sorry

theorem first_digit_times_num_digits :
  (smallest_num_with_digit_sum_2017 / 10^224) * num_digits smallest_num_with_digit_sum_2017 = 225 :=
by sorry

end smallest_num_with_digit_sum_2017_properties_first_digit_times_num_digits_l3679_367953


namespace special_ring_classification_l3679_367941

universe u

/-- A ring satisfying the given property -/
class SpecialRing (A : Type u) extends Ring A where
  special_property : ∀ (x : A), x ≠ 0 → x^(2^n + 1) = 1
  n : ℕ
  n_pos : n ≥ 1

/-- The theorem stating that any SpecialRing is isomorphic to F₂ or F₄ -/
theorem special_ring_classification (A : Type u) [SpecialRing A] :
  (∃ (f : A ≃+* Fin 2), Function.Bijective f) ∨
  (∃ (g : A ≃+* Fin 4), Function.Bijective g) :=
sorry

end special_ring_classification_l3679_367941


namespace four_integer_pairs_satisfy_equation_l3679_367927

theorem four_integer_pairs_satisfy_equation : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1 + p.2 = p.1 * p.2 - 1) ∧ 
    s.card = 4 := by
  sorry

end four_integer_pairs_satisfy_equation_l3679_367927


namespace yellow_ball_count_l3679_367920

theorem yellow_ball_count (r b g y : ℕ) : 
  r = 2 * b →
  b = 2 * g →
  y > 7 →
  r + b + g + y = 27 →
  y = 20 := by
sorry

end yellow_ball_count_l3679_367920


namespace geometric_sequence_sum_l3679_367964

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n - 1)

theorem geometric_sequence_sum (a₁ r : ℝ) (h₁ : a₁ = 1) (h₂ : r = -2) :
  let a := geometric_sequence a₁ r
  (a 1) + |a 2| + (a 3) + |a 4| = 15 := by
sorry

end geometric_sequence_sum_l3679_367964


namespace ferry_speed_difference_l3679_367971

theorem ferry_speed_difference (speed_p time_p distance_q_factor time_difference : ℝ) 
  (h1 : speed_p = 6)
  (h2 : time_p = 3)
  (h3 : distance_q_factor = 3)
  (h4 : time_difference = 3) : 
  let distance_p := speed_p * time_p
  let distance_q := distance_q_factor * distance_p
  let time_q := time_p + time_difference
  let speed_q := distance_q / time_q
  speed_q - speed_p = 3 := by sorry

end ferry_speed_difference_l3679_367971


namespace arithmetic_mean_of_fractions_l3679_367990

theorem arithmetic_mean_of_fractions :
  let a := 8 / 12
  let b := 5 / 6
  let c := 9 / 12
  c = (a + b) / 2 := by sorry

end arithmetic_mean_of_fractions_l3679_367990


namespace second_quadrant_angles_l3679_367933

-- Define a function to check if an angle is in the second quadrant
def is_in_second_quadrant (angle : ℝ) : Prop :=
  90 < angle % 360 ∧ angle % 360 ≤ 180

-- Define the given angles
def angle1 : ℝ := -120
def angle2 : ℝ := -240
def angle3 : ℝ := 180
def angle4 : ℝ := 495

-- Theorem statement
theorem second_quadrant_angles :
  is_in_second_quadrant angle2 ∧
  is_in_second_quadrant angle4 ∧
  ¬is_in_second_quadrant angle1 ∧
  ¬is_in_second_quadrant angle3 :=
sorry

end second_quadrant_angles_l3679_367933


namespace sum_maximized_at_11_or_12_l3679_367931

/-- The sequence term defined as a function of n -/
def a (n : ℕ) : ℤ := 24 - 2 * n

/-- The sum of the first n terms of the sequence -/
def S (n : ℕ) : ℤ := n * (24 - n)

/-- Theorem stating that the sum is maximized when n is 11 or 12 -/
theorem sum_maximized_at_11_or_12 :
  ∀ k : ℕ, k > 0 → S k ≤ max (S 11) (S 12) :=
sorry

end sum_maximized_at_11_or_12_l3679_367931


namespace exponent_simplification_l3679_367955

theorem exponent_simplification (a : ℝ) (h : a > 0) : 
  a^(1/2) * a^(2/3) / a^(1/6) = a := by
  sorry

end exponent_simplification_l3679_367955


namespace trigonometric_values_l3679_367929

theorem trigonometric_values (α : Real) 
  (h1 : α ∈ Set.Ioo (π/3) (π/2))
  (h2 : Real.cos (π/6 + α) * Real.cos (π/3 - α) = -1/4) : 
  Real.sin (2*α) = Real.sqrt 3 / 2 ∧ 
  Real.tan α - 1 / Real.tan α = 2 * Real.sqrt 3 / 3 := by
sorry

end trigonometric_values_l3679_367929


namespace water_bills_theorem_l3679_367936

/-- Water pricing structure -/
def water_price (usage : ℕ) : ℚ :=
  if usage ≤ 10 then 0.45 * usage
  else if usage ≤ 20 then 0.45 * 10 + 0.80 * (usage - 10)
  else 0.45 * 10 + 0.80 * 10 + 1.50 * (usage - 20)

/-- Theorem stating the water bills for households A, B, and C -/
theorem water_bills_theorem :
  ∃ (usage_A usage_B usage_C : ℕ),
    usage_A > 20 ∧ 
    10 < usage_B ∧ usage_B ≤ 20 ∧
    usage_C ≤ 10 ∧
    water_price usage_A - water_price usage_B = 7.10 ∧
    water_price usage_B - water_price usage_C = 3.75 ∧
    water_price usage_A = 14 ∧
    water_price usage_B = 6.9 ∧
    water_price usage_C = 3.15 :=
by sorry

end water_bills_theorem_l3679_367936


namespace total_spent_on_presents_l3679_367973

def leonard_wallets : ℕ := 3
def leonard_wallet_price : ℚ := 35.50
def leonard_sneakers : ℕ := 2
def leonard_sneaker_price : ℚ := 120.75
def leonard_belt_price : ℚ := 44.25
def leonard_discount_rate : ℚ := 0.10

def michael_backpack_price : ℚ := 89.50
def michael_jeans : ℕ := 3
def michael_jeans_price : ℚ := 54.50
def michael_tie_price : ℚ := 24.75
def michael_discount_rate : ℚ := 0.15

def emily_shirts : ℕ := 2
def emily_shirt_price : ℚ := 69.25
def emily_books : ℕ := 4
def emily_book_price : ℚ := 14.80
def emily_tax_rate : ℚ := 0.08

theorem total_spent_on_presents (leonard_total michael_total emily_total : ℚ) :
  leonard_total = (1 - leonard_discount_rate) * (leonard_wallets * leonard_wallet_price + leonard_sneakers * leonard_sneaker_price + leonard_belt_price) →
  michael_total = (1 - michael_discount_rate) * (michael_backpack_price + michael_jeans * michael_jeans_price + michael_tie_price) →
  emily_total = (1 + emily_tax_rate) * (emily_shirts * emily_shirt_price + emily_books * emily_book_price) →
  leonard_total + michael_total + emily_total = 802.64 :=
by sorry

end total_spent_on_presents_l3679_367973


namespace polynomial_symmetry_l3679_367915

/-- Given a polynomial function f(x) = ax^5 + bx^3 + cx + 6 where f(-3) = -12, prove that f(3) = 24 -/
theorem polynomial_symmetry (a b c : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^5 + b * x^3 + c * x + 6)
  (h2 : f (-3) = -12) : 
  f 3 = 24 := by sorry

end polynomial_symmetry_l3679_367915


namespace c_months_correct_l3679_367977

/-- The number of months c put his oxen for grazing -/
def c_months : ℝ :=
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let b_months := 5
  let c_oxen := 15
  let total_rent := 210
  let c_share := 53.99999999999999
  3

/-- Theorem stating that c_months is correct given the problem conditions -/
theorem c_months_correct :
  let a_oxen := 10
  let a_months := 7
  let b_oxen := 12
  let b_months := 5
  let c_oxen := 15
  let total_rent := 210
  let c_share := 53.99999999999999
  let total_ox_months := a_oxen * a_months + b_oxen * b_months + c_oxen * c_months
  c_share = (c_oxen * c_months / total_ox_months) * total_rent :=
by sorry

#eval c_months

end c_months_correct_l3679_367977
