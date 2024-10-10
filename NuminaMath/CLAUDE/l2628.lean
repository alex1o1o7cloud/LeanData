import Mathlib

namespace larger_number_value_l2628_262847

theorem larger_number_value (x y : ℝ) 
  (h1 : 4 * y = 5 * x) 
  (h2 : y - x = 10) : 
  y = 50 := by
  sorry

end larger_number_value_l2628_262847


namespace max_students_above_mean_l2628_262844

/-- Given a class of 150 students, proves that the maximum number of students
    who can have a score higher than the class mean is 149. -/
theorem max_students_above_mean (scores : Fin 150 → ℝ) :
  (Finset.filter (fun i => scores i > Finset.sum Finset.univ scores / 150) Finset.univ).card ≤ 149 :=
by
  sorry

end max_students_above_mean_l2628_262844


namespace min_n_constant_term_is_correct_l2628_262880

/-- The minimum natural number n for which (x^2 + 1/(2x^3))^n contains a constant term -/
def min_n_constant_term : ℕ := 5

/-- Predicate to check if the expansion contains a constant term -/
def has_constant_term (n : ℕ) : Prop :=
  ∃ r : ℕ, 2 * n = 5 * r

theorem min_n_constant_term_is_correct :
  (∀ k < min_n_constant_term, ¬ has_constant_term k) ∧
  has_constant_term min_n_constant_term := by sorry

end min_n_constant_term_is_correct_l2628_262880


namespace joe_chocolate_spending_l2628_262860

theorem joe_chocolate_spending (total : ℚ) (fruit_fraction : ℚ) (left : ℚ) 
  (h1 : total = 450)
  (h2 : fruit_fraction = 2/5)
  (h3 : left = 220) :
  (total - left - fruit_fraction * total) / total = 1/9 := by
  sorry

end joe_chocolate_spending_l2628_262860


namespace modular_inverse_of_5_mod_26_l2628_262855

theorem modular_inverse_of_5_mod_26 : ∃ x : ℕ, x ≤ 25 ∧ (5 * x) % 26 = 1 :=
by
  use 21
  sorry

end modular_inverse_of_5_mod_26_l2628_262855


namespace complex_equation_implies_real_equation_l2628_262806

theorem complex_equation_implies_real_equation (a b : ℝ) :
  (Complex.I : ℂ) ^ 2 = -1 →
  (3 + 4 * Complex.I) * (a + b * Complex.I) = 10 * Complex.I →
  3 * a - 4 * b = 0 := by
sorry

end complex_equation_implies_real_equation_l2628_262806


namespace ellipse_m_relation_l2628_262870

/-- Represents an ellipse with parameter m -/
structure Ellipse (m : ℝ) where
  eq : ∀ (x y : ℝ), x^2 / (10 - m) + y^2 / (m - 2) = 1
  major_axis_y : m - 2 > 10 - m
  focal_distance : ℝ

/-- The theorem stating the relationship between m and the focal distance -/
theorem ellipse_m_relation (m : ℝ) (e : Ellipse m) (h : e.focal_distance = 4) :
  16 = 2 * m - 12 := by
  sorry


end ellipse_m_relation_l2628_262870


namespace number_divisibility_l2628_262840

theorem number_divisibility (a b : ℕ) : 
  (∃ k : ℤ, (1001 * a + 110 * b : ℤ) = 11 * k) ∧ 
  (∃ m : ℤ, (111000 * a + 111 * b : ℤ) = 37 * m) ∧
  (∃ n : ℤ, (101010 * a + 10101 * b : ℤ) = 7 * n) ∧
  (∃ p q : ℤ, (909 * (a - b) : ℤ) = 9 * p ∧ (909 * (a - b) : ℤ) = 101 * q) :=
by sorry

end number_divisibility_l2628_262840


namespace corrected_mean_l2628_262845

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 ∧ original_mean = 36 ∧ incorrect_value = 23 ∧ correct_value = 60 →
  (n : ℚ) * original_mean + (correct_value - incorrect_value) / n = 36.74 := by
  sorry

end corrected_mean_l2628_262845


namespace arithmetic_sequence_120th_term_l2628_262830

/-- The nth term of an arithmetic sequence -/
def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 120th term of the specific arithmetic sequence -/
def term_120 : ℝ :=
  arithmetic_sequence 6 6 120

theorem arithmetic_sequence_120th_term :
  term_120 = 720 := by sorry

end arithmetic_sequence_120th_term_l2628_262830


namespace arithmetic_sequence_sum_l2628_262899

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a →
  a 1 + a 3 = 2 →
  a 2 + a 4 = 6 →
  a 1 + a 7 = 10 := by
  sorry

end arithmetic_sequence_sum_l2628_262899


namespace parallelogram_condition_inscribed_quadrilateral_condition_l2628_262889

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop := sorry

-- Define parallel sides
def parallel_sides (q : Quadrilateral) (side1 side2 : Segment) : Prop := sorry

-- Define equal sides
def equal_sides (side1 side2 : Segment) : Prop := sorry

-- Define supplementary angles
def supplementary_angles (a1 a2 : Angle) : Prop := sorry

-- Define inscribed in a circle
def inscribed_in_circle (q : Quadrilateral) : Prop := sorry

-- Theorem 1
theorem parallelogram_condition (q : Quadrilateral) 
  (side1 side2 : Segment) :
  parallel_sides q side1 side2 → 
  equal_sides side1 side2 → 
  is_parallelogram q :=
sorry

-- Theorem 2
theorem inscribed_quadrilateral_condition (q : Quadrilateral) 
  (a1 a2 a3 a4 : Angle) :
  supplementary_angles a1 a3 → 
  supplementary_angles a2 a4 → 
  inscribed_in_circle q :=
sorry

end parallelogram_condition_inscribed_quadrilateral_condition_l2628_262889


namespace quadratic_sum_of_b_and_c_l2628_262850

/-- Given a quadratic expression x^2 - 24x + 50, when written in the form (x+b)^2 + c,
    the sum of b and c is equal to -106. -/
theorem quadratic_sum_of_b_and_c : ∃ (b c : ℝ), 
  (∀ x, x^2 - 24*x + 50 = (x + b)^2 + c) ∧ (b + c = -106) := by
  sorry

end quadratic_sum_of_b_and_c_l2628_262850


namespace aaron_age_l2628_262841

/-- Proves that Aaron is 16 years old given the conditions of the problem -/
theorem aaron_age : 
  ∀ (aaron_age henry_age sister_age : ℕ),
  sister_age = 3 * aaron_age →
  henry_age = 4 * sister_age →
  henry_age + sister_age = 240 →
  aaron_age = 16 :=
by
  sorry

end aaron_age_l2628_262841


namespace total_rainfall_l2628_262803

def rainfall_problem (first_week : ℝ) (second_week : ℝ) : Prop :=
  (second_week = 1.5 * first_week) ∧
  (second_week = 12) ∧
  (first_week + second_week = 20)

theorem total_rainfall : ∃ (first_week second_week : ℝ), 
  rainfall_problem first_week second_week :=
by
  sorry

end total_rainfall_l2628_262803


namespace x_gt_one_sufficient_not_necessary_for_x_gt_zero_l2628_262865

theorem x_gt_one_sufficient_not_necessary_for_x_gt_zero :
  (∀ x : ℝ, x > 1 → x > 0) ∧ (∃ x : ℝ, x > 0 ∧ x ≤ 1) := by
  sorry

end x_gt_one_sufficient_not_necessary_for_x_gt_zero_l2628_262865


namespace unique_solution_trig_system_l2628_262816

theorem unique_solution_trig_system (x : ℝ) :
  (Real.arccos (3 * x) - Real.arcsin x = π / 6 ∧
   Real.arccos (3 * x) + Real.arcsin x = 5 * π / 6) ↔ x = 0 := by
  sorry

end unique_solution_trig_system_l2628_262816


namespace matts_work_schedule_l2628_262892

/-- Matt's work schedule problem -/
theorem matts_work_schedule (monday_minutes : ℕ) (wednesday_minutes : ℕ) : 
  monday_minutes = 450 →
  wednesday_minutes = 300 →
  wednesday_minutes - (monday_minutes / 2) = 75 := by
  sorry

end matts_work_schedule_l2628_262892


namespace train_length_l2628_262838

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 52 → time = 9 → ∃ length : ℝ, 
  (abs (length - 129.96) < 0.01) ∧ (length = speed * 1000 / 3600 * time) := by
  sorry

#check train_length

end train_length_l2628_262838


namespace knives_percentage_after_trade_l2628_262882

/-- Represents Carolyn's silverware set --/
structure SilverwareSet where
  knives : ℕ
  forks : ℕ
  spoons : ℕ

/-- Calculates the total number of pieces in a silverware set --/
def SilverwareSet.total (s : SilverwareSet) : ℕ :=
  s.knives + s.forks + s.spoons

/-- Represents the initial state of Carolyn's silverware set --/
def initial_set : SilverwareSet :=
  { knives := 6
  , forks := 12
  , spoons := 6 * 3 }

/-- Represents the trade operation --/
def trade (s : SilverwareSet) : SilverwareSet :=
  { knives := s.knives - 6
  , forks := s.forks
  , spoons := s.spoons + 6 }

/-- Theorem stating that after the trade, 0% of Carolyn's silverware is knives --/
theorem knives_percentage_after_trade :
  let final_set := trade initial_set
  (final_set.knives : ℚ) / (final_set.total : ℚ) * 100 = 0 := by
  sorry

end knives_percentage_after_trade_l2628_262882


namespace angle_measure_from_area_ratio_l2628_262868

/-- Given three concentric circles and two lines passing through their center,
    prove that the acute angle between the lines is 12π/77 radians when the
    shaded area is 3/4 of the unshaded area. -/
theorem angle_measure_from_area_ratio :
  ∀ (r₁ r₂ r₃ : ℝ) (shaded_area unshaded_area : ℝ) (θ : ℝ),
  r₁ = 4 →
  r₂ = 3 →
  r₃ = 2 →
  shaded_area = (3/4) * unshaded_area →
  shaded_area + unshaded_area = π * (r₁^2 + r₂^2 + r₃^2) →
  shaded_area = θ * (r₁^2 + r₃^2) + (π - θ) * r₂^2 →
  θ = 12 * π / 77 :=
by sorry

end angle_measure_from_area_ratio_l2628_262868


namespace seminar_attendees_l2628_262821

/-- The total number of attendees at a seminar -/
def total_attendees (company_a company_b company_c company_d other : ℕ) : ℕ :=
  company_a + company_b + company_c + company_d + other

/-- Theorem: Given the conditions, the total number of attendees is 185 -/
theorem seminar_attendees : 
  ∀ (company_a company_b company_c company_d other : ℕ),
    company_a = 30 →
    company_b = 2 * company_a →
    company_c = company_a + 10 →
    company_d = company_c - 5 →
    other = 20 →
    total_attendees company_a company_b company_c company_d other = 185 :=
by
  sorry

#eval total_attendees 30 60 40 35 20

end seminar_attendees_l2628_262821


namespace stone_growth_prevention_l2628_262817

/-- The amount of stone consumed by one warrior per day -/
def warrior_consumption : ℝ := 1

/-- The number of days it takes for the stone to pierce the sky with 14 warriors -/
def days_with_14 : ℕ := 16

/-- The number of days it takes for the stone to pierce the sky with 15 warriors -/
def days_with_15 : ℕ := 24

/-- The daily growth rate of the stone -/
def stone_growth_rate : ℝ := 17 * warrior_consumption

/-- The minimum number of warriors needed to prevent the stone from piercing the sky -/
def min_warriors : ℕ := 17

theorem stone_growth_prevention :
  (↑min_warriors * warrior_consumption = stone_growth_rate) ∧
  (∀ n : ℕ, n < min_warriors → ↑n * warrior_consumption < stone_growth_rate) := by
  sorry

#check stone_growth_prevention

end stone_growth_prevention_l2628_262817


namespace monotone_increasing_intervals_l2628_262823

/-- The function f(x) = 2x^3 - 3x^2 - 36x + 16 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 36 * x + 16

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 6 * x^2 - 6 * x - 36

theorem monotone_increasing_intervals :
  MonotoneOn f (Set.Ici (-2) ∩ Set.Iic (-2)) ∧
  MonotoneOn f (Set.Ici 3) :=
sorry

end monotone_increasing_intervals_l2628_262823


namespace complex_modulus_one_l2628_262866

theorem complex_modulus_one (z : ℂ) (h : (1 - z) / (1 + z) = Complex.I * 2) : Complex.abs z = 1 := by
  sorry

end complex_modulus_one_l2628_262866


namespace first_number_value_l2628_262810

theorem first_number_value (x y : ℝ) : 
  x - y = 88 → y = 0.2 * x → x = 110 := by sorry

end first_number_value_l2628_262810


namespace expression_lower_bound_l2628_262832

theorem expression_lower_bound (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : b ≠ 0) :
  ((a + b)^2 + (b - c)^2 + (a - c)^2) / b^2 ≥ 2.5 := by
  sorry

end expression_lower_bound_l2628_262832


namespace sin_cos_45_degrees_l2628_262896

theorem sin_cos_45_degrees : 
  let θ : Real := Real.pi / 4  -- 45 degrees in radians
  Real.sin θ = 1 / Real.sqrt 2 ∧ Real.cos θ = 1 / Real.sqrt 2 := by
  sorry

end sin_cos_45_degrees_l2628_262896


namespace circle_pair_relation_infinite_quadrilaterals_l2628_262879

/-- A structure representing a pair of circles with a quadrilateral inscribed in one and circumscribed around the other. -/
structure CirclePair where
  R : ℝ  -- Radius of the larger circle
  r : ℝ  -- Radius of the smaller circle
  d : ℝ  -- Distance between the centers of the circles
  h_positive_R : R > 0
  h_positive_r : r > 0
  h_positive_d : d > 0
  h_d_less_R : d < R

/-- The main theorem stating the relationship between the radii and distance of the circles. -/
theorem circle_pair_relation (cp : CirclePair) :
  1 / (cp.R + cp.d)^2 + 1 / (cp.R - cp.d)^2 = 1 / cp.r^2 :=
sorry

/-- There exist infinitely many quadrilaterals satisfying the conditions. -/
theorem infinite_quadrilaterals (R r d : ℝ) (h_R : R > 0) (h_r : r > 0) (h_d : d > 0) (h_d_R : d < R) :
  ∃ (cp : CirclePair), cp.R = R ∧ cp.r = r ∧ cp.d = d :=
sorry

end circle_pair_relation_infinite_quadrilaterals_l2628_262879


namespace problem_statement_l2628_262890

theorem problem_statement : (2002^3 + 4 * 2002^2 + 6006) / (2002^2 + 2002) = 2005 := by
  sorry

end problem_statement_l2628_262890


namespace smaller_solution_of_quadratic_l2628_262898

theorem smaller_solution_of_quadratic (x : ℝ) : 
  x^2 + 12*x - 64 = 0 ∧ 
  (∀ y : ℝ, y^2 + 12*y - 64 = 0 → x ≤ y) → 
  x = -16 := by
sorry

end smaller_solution_of_quadratic_l2628_262898


namespace set_operations_l2628_262805

-- Define the sets A and B
def A : Set ℝ := {x | x < 1 ∨ x > 2}
def B : Set ℝ := {x | x < -3 ∨ x ≥ 1}

-- State the theorem
theorem set_operations :
  (Set.univ \ A = {x | 1 ≤ x ∧ x ≤ 2}) ∧
  (Set.univ \ B = {x | -3 ≤ x ∧ x < 1}) ∧
  (A ∩ B = {x | x < -3 ∨ x > 2}) ∧
  (A ∪ B = Set.univ) := by
  sorry

end set_operations_l2628_262805


namespace ella_gives_one_sixth_l2628_262852

-- Define the initial cookie distribution
def initial_distribution (luke_cookies : ℚ) : ℚ × ℚ × ℚ :=
  (2 * luke_cookies, 4 * luke_cookies, luke_cookies)

-- Define the function to calculate the fraction Ella gives to Luke
def fraction_ella_gives (luke_cookies : ℚ) : ℚ :=
  let (ella_cookies, connor_cookies, luke_cookies) := initial_distribution luke_cookies
  let total_cookies := ella_cookies + connor_cookies + luke_cookies
  let equal_share := total_cookies / 3
  (equal_share - luke_cookies) / ella_cookies

-- Theorem statement
theorem ella_gives_one_sixth :
  ∀ (luke_cookies : ℚ), luke_cookies > 0 → fraction_ella_gives luke_cookies = 1/6 := by
  sorry


end ella_gives_one_sixth_l2628_262852


namespace decreasing_linear_function_not_in_third_quadrant_l2628_262836

/-- A linear function y = kx + 1 where k ≠ 0 and y decreases as x increases -/
structure DecreasingLinearFunction where
  k : ℝ
  hk_nonzero : k ≠ 0
  hk_negative : k < 0

/-- The third quadrant -/
def ThirdQuadrant : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 > 0 ∧ p.2 < 0}

/-- The graph of a linear function y = kx + 1 -/
def LinearFunctionGraph (f : DecreasingLinearFunction) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = f.k * p.1 + 1}

/-- The theorem stating that the graph of a decreasing linear function
    does not pass through the third quadrant -/
theorem decreasing_linear_function_not_in_third_quadrant
  (f : DecreasingLinearFunction) :
  LinearFunctionGraph f ∩ ThirdQuadrant = ∅ := by
  sorry

end decreasing_linear_function_not_in_third_quadrant_l2628_262836


namespace janabel_widget_sales_l2628_262820

theorem janabel_widget_sales (n : ℕ) (h : n = 15) : 
  let a₁ := 2
  let d := 3
  let aₙ := a₁ + (n - 1) * d
  n / 2 * (a₁ + aₙ) = 345 := by
  sorry

end janabel_widget_sales_l2628_262820


namespace max_lateral_area_inscribed_cylinder_l2628_262842

/-- The maximum lateral surface area of a cylinder inscribed in a sphere -/
theorem max_lateral_area_inscribed_cylinder (r : ℝ) (h : r > 0) :
  ∃ (cylinder_area : ℝ),
    (∀ (inscribed_cylinder_area : ℝ), inscribed_cylinder_area ≤ cylinder_area) ∧
    cylinder_area = 2 * Real.pi * r^2 :=
sorry

end max_lateral_area_inscribed_cylinder_l2628_262842


namespace root_sum_theorem_l2628_262861

theorem root_sum_theorem : ∃ (a b : ℝ), 
  (∃ (x y : ℝ), x ≠ y ∧ 
    ((a * x^2 - 24 * x + b) / (x^2 - 1) = x) ∧ 
    ((a * y^2 - 24 * y + b) / (y^2 - 1) = y) ∧
    x + y = 12) ∧
  ((a = 11 ∧ b = -35) ∨ (a = 35 ∧ b = -5819)) := by
sorry

end root_sum_theorem_l2628_262861


namespace square_perimeter_sum_l2628_262873

theorem square_perimeter_sum (x y : ℕ) : 
  (x : ℤ) ^ 2 - (y : ℤ) ^ 2 = 19 → 
  4 * x + 4 * y = 76 :=
by
  sorry

end square_perimeter_sum_l2628_262873


namespace train_platform_problem_l2628_262818

/-- Represents the speed of the train in km/hr -/
def train_speed : ℝ := 72

/-- Represents the time taken to cross the platform in minutes -/
def crossing_time : ℝ := 1

/-- Calculates the length of the train in meters -/
def train_length : ℝ := 600

theorem train_platform_problem :
  ∀ (train_length platform_length : ℝ),
  train_length = platform_length →
  train_length = train_speed * (1000 / 3600) * (crossing_time * 60) / 2 →
  train_length = 600 := by
  sorry

end train_platform_problem_l2628_262818


namespace unique_valid_number_l2628_262813

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100 ∧
  (n / 10 + n % 10 = 9) ∧
  (10 * (n % 10) + (n / 10) = n + 9)

theorem unique_valid_number : ∃! n : ℕ, is_valid_number n :=
  sorry

end unique_valid_number_l2628_262813


namespace work_left_theorem_l2628_262871

def work_left (p_days q_days collab_days : ℚ) : ℚ :=
  1 - collab_days * (1 / p_days + 1 / q_days)

theorem work_left_theorem (p_days q_days collab_days : ℚ) 
  (hp : p_days = 15)
  (hq : q_days = 20)
  (hc : collab_days = 4) :
  work_left p_days q_days collab_days = 8 / 15 := by
  sorry

#eval work_left 15 20 4

end work_left_theorem_l2628_262871


namespace set_b_forms_triangle_l2628_262885

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function to check if three line segments can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

/-- Theorem: The set of line segments (6, 7, 8) can form a triangle. -/
theorem set_b_forms_triangle : can_form_triangle 6 7 8 := by
  sorry

end set_b_forms_triangle_l2628_262885


namespace custom_mult_chain_l2628_262888

/-- Custom multiplication operation -/
def star_mult (a b : ℚ) : ℚ := (a - b) / (1 - a * b)

/-- Main theorem -/
theorem custom_mult_chain : star_mult 5 (star_mult 6 (star_mult 7 (star_mult 8 9))) = 3588 / 587 := by
  sorry

end custom_mult_chain_l2628_262888


namespace platform_length_l2628_262839

/-- Given a train of length 450 m that crosses a platform in 56 sec and a signal pole in 24 sec,
    the length of the platform is 600 m. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) 
    (h1 : train_length = 450)
    (h2 : platform_time = 56)
    (h3 : pole_time = 24) : 
  train_length * (platform_time / pole_time - 1) = 600 := by
  sorry

end platform_length_l2628_262839


namespace sock_distribution_l2628_262884

-- Define the total number of socks
def total_socks : ℕ := 9

-- Define the property that among any 4 socks, at least 2 belong to the same child
def at_least_two_same (socks : Finset ℕ) : Prop :=
  ∀ (s : Finset ℕ), s ⊆ socks → s.card = 4 → ∃ (child : ℕ), (s.filter (λ x => x = child)).card ≥ 2

-- Define the property that among any 5 socks, no more than 3 belong to the same child
def no_more_than_three (socks : Finset ℕ) : Prop :=
  ∀ (s : Finset ℕ), s ⊆ socks → s.card = 5 → ∀ (child : ℕ), (s.filter (λ x => x = child)).card ≤ 3

-- Theorem statement
theorem sock_distribution (socks : Finset ℕ) 
  (h_total : socks.card = total_socks)
  (h_at_least_two : at_least_two_same socks)
  (h_no_more_than_three : no_more_than_three socks) :
  ∃ (children : Finset ℕ), 
    children.card = 3 ∧ 
    (∀ child ∈ children, (socks.filter (λ x => x = child)).card = 3) :=
sorry

end sock_distribution_l2628_262884


namespace largest_three_digit_special_number_l2628_262878

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def distinct_nonzero_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  hundreds ≠ 0 ∧ hundreds ≠ tens ∧ hundreds ≠ ones ∧ tens ≠ ones ∧ tens ≠ 0 ∧ ones ≠ 0

def sum_of_digits (n : ℕ) : ℕ :=
  (n / 100) + ((n % 100) / 10) + (n % 10)

def divisible_by_digits (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n % 100) / 10
  let ones := n % 10
  n % hundreds = 0 ∧ (tens ≠ 0 → n % tens = 0) ∧ (ones ≠ 0 → n % ones = 0)

theorem largest_three_digit_special_number :
  ∀ n : ℕ, 100 ≤ n → n < 1000 →
    (distinct_nonzero_digits n ∧
     is_prime (sum_of_digits n) ∧
     divisible_by_digits n) →
    n ≤ 963 :=
sorry

end largest_three_digit_special_number_l2628_262878


namespace lord_moneybag_puzzle_l2628_262876

/-- Lord Moneybag's Christmas money puzzle -/
theorem lord_moneybag_puzzle :
  ∃! n : ℕ, 300 ≤ n ∧ n ≤ 500 ∧ 
  6 ∣ n ∧
  5 ∣ (n - 1) ∧
  4 ∣ (n - 2) ∧
  3 ∣ (n - 3) ∧
  2 ∣ (n - 4) ∧
  Nat.Prime (n - 5) ∧
  n = 426 := by
sorry

end lord_moneybag_puzzle_l2628_262876


namespace transformed_function_theorem_l2628_262894

def original_function (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

def rotate_180_degrees (f : ℝ → ℝ) : ℝ → ℝ := λ x => -f x

def translate_upwards (f : ℝ → ℝ) (units : ℝ) : ℝ → ℝ := λ x => f x + units

theorem transformed_function_theorem :
  (translate_upwards (rotate_180_degrees original_function) 3) = λ x => -2 * x^2 - 4 * x :=
by sorry

end transformed_function_theorem_l2628_262894


namespace partial_fraction_decomposition_l2628_262827

theorem partial_fraction_decomposition :
  ∃! (A B : ℝ), ∀ (x : ℝ), x ≠ 5 → x ≠ 6 →
    (5 * x - 8) / (x^2 - 11 * x + 30) = A / (x - 5) + B / (x - 6) ∧ A = -17 ∧ B = 22 := by
  sorry

end partial_fraction_decomposition_l2628_262827


namespace linear_function_fixed_point_l2628_262822

theorem linear_function_fixed_point :
  ∀ (k : ℝ), (2 * k - 3) * 2 + (k + 1) * (-3) - (k - 9) = 0 := by
sorry

end linear_function_fixed_point_l2628_262822


namespace james_profit_20_weeks_l2628_262867

/-- Calculates the profit from James' media empire over a given number of weeks. -/
def calculate_profit (movie_cost : ℕ) (dvd_cost : ℕ) (price_multiplier : ℚ) 
                     (daily_sales : ℕ) (days_per_week : ℕ) (num_weeks : ℕ) : ℕ :=
  let selling_price := dvd_cost * price_multiplier
  let profit_per_dvd := selling_price - dvd_cost
  let daily_profit := profit_per_dvd * daily_sales
  let weekly_profit := daily_profit * days_per_week
  let total_profit := weekly_profit * num_weeks
  (total_profit - movie_cost).floor.toNat

/-- Theorem stating that James' profit over 20 weeks is $448,000. -/
theorem james_profit_20_weeks : 
  calculate_profit 2000 6 (5/2) 500 5 20 = 448000 := by
  sorry

end james_profit_20_weeks_l2628_262867


namespace reflection_coordinates_sum_l2628_262856

/-- Given a point C with coordinates (3, -2) and its reflection D over the y-axis,
    the sum of their four coordinate values is -4. -/
theorem reflection_coordinates_sum :
  let C : ℝ × ℝ := (3, -2)
  let D : ℝ × ℝ := (-C.1, C.2)  -- Reflection over y-axis
  C.1 + C.2 + D.1 + D.2 = -4 :=
by sorry

end reflection_coordinates_sum_l2628_262856


namespace relationship_between_D_and_A_l2628_262863

theorem relationship_between_D_and_A (A B C D : Prop) 
  (h1 : A → B)
  (h2 : ¬(B → A))
  (h3 : B → C)
  (h4 : ¬(C → B))
  (h5 : D ↔ C) :
  (D → A) ∧ ¬(A → D) :=
sorry

end relationship_between_D_and_A_l2628_262863


namespace f_2005_equals_2_l2628_262824

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem f_2005_equals_2 
  (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_period : ∀ x, f (x + 6) = f x + f 3)
  (h_f_1 : f 1 = 2) :
  f 2005 = 2 := by
sorry

end f_2005_equals_2_l2628_262824


namespace min_value_quadratic_expression_l2628_262887

theorem min_value_quadratic_expression :
  ∃ (min_val : ℝ), min_val = -7208 ∧
  ∀ (x y : ℝ), 2*x^2 + 3*x*y + 4*y^2 - 8*x - 10*y ≥ min_val :=
by sorry

end min_value_quadratic_expression_l2628_262887


namespace three_squares_before_2300_l2628_262809

theorem three_squares_before_2300 : 
  ∃ (n : ℕ), n = 2025 ∧ 
  (∃ (a b c : ℕ), 
    n < a^2 ∧ a^2 < b^2 ∧ b^2 < c^2 ∧ c^2 ≤ 2300 ∧
    ∀ (x : ℕ), n < x^2 ∧ x^2 ≤ 2300 → x^2 = a^2 ∨ x^2 = b^2 ∨ x^2 = c^2) ∧
  ∀ (m : ℕ), m > n → 
    ¬(∃ (a b c : ℕ), 
      m < a^2 ∧ a^2 < b^2 ∧ b^2 < c^2 ∧ c^2 ≤ 2300 ∧
      ∀ (x : ℕ), m < x^2 ∧ x^2 ≤ 2300 → x^2 = a^2 ∨ x^2 = b^2 ∨ x^2 = c^2) :=
by sorry

end three_squares_before_2300_l2628_262809


namespace inequality_solution_existence_l2628_262862

theorem inequality_solution_existence (a : ℝ) (ha : a > 0) :
  (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ a > 1 := by
  sorry

end inequality_solution_existence_l2628_262862


namespace brianna_cd_purchase_l2628_262874

/-- Brianna's CD purchase problem -/
theorem brianna_cd_purchase (m : ℚ) (c : ℚ) : 
  (1 / 4 : ℚ) * m = (1 / 2 : ℚ) * c → 
  m - c = (1 / 2 : ℚ) * m := by
  sorry

end brianna_cd_purchase_l2628_262874


namespace find_m_l2628_262875

def U : Set ℕ := {1, 2, 3, 4}

def A (m : ℕ) : Set ℕ := {x ∈ U | x^2 - 5*x + m = 0}

theorem find_m : ∃ m : ℕ, (U \ A m) = {1, 4} := by
  sorry

end find_m_l2628_262875


namespace sum_abc_equals_negative_three_l2628_262846

theorem sum_abc_equals_negative_three
  (a b c : ℝ)
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_common_root1 : ∃ x : ℝ, x^2 + a*x + 1 = 0 ∧ x^2 + b*x + c = 0)
  (h_common_root2 : ∃ x : ℝ, x^2 + x + a = 0 ∧ x^2 + c*x + b = 0) :
  a + b + c = -3 :=
by sorry

end sum_abc_equals_negative_three_l2628_262846


namespace xy_greater_than_xz_l2628_262854

theorem xy_greater_than_xz (x y z : ℝ) (h1 : x > y) (h2 : y > z) (h3 : x + y + z = 0) : x * y > x * z := by
  sorry

end xy_greater_than_xz_l2628_262854


namespace inequality_proof_l2628_262801

theorem inequality_proof (a b : ℝ) (h : a < b) : -3 * a > -3 * b := by
  sorry

end inequality_proof_l2628_262801


namespace hyperbola_focal_length_l2628_262857

/-- Given a hyperbola C with equation x²/m - y² = 1 where m > 0,
    and its asymptote √3x + my = 0, the focal length of C is 4. -/
theorem hyperbola_focal_length (m : ℝ) (hm : m > 0) : 
  (∃ (C : Set (ℝ × ℝ)), 
    C = {(x, y) | x^2 / m - y^2 = 1} ∧
    (∃ (asymptote : Set (ℝ × ℝ)), 
      asymptote = {(x, y) | Real.sqrt 3 * x + m * y = 0})) →
  (∃ (focal_length : ℝ), focal_length = 4) :=
by sorry

end hyperbola_focal_length_l2628_262857


namespace cans_recycled_l2628_262891

/-- Proves the number of cans recycled given the bottle and can deposits, number of bottles, and total money earned -/
theorem cans_recycled 
  (bottle_deposit : ℚ) 
  (can_deposit : ℚ) 
  (bottles_recycled : ℕ) 
  (total_money : ℚ) 
  (h1 : bottle_deposit = 10 / 100)
  (h2 : can_deposit = 5 / 100)
  (h3 : bottles_recycled = 80)
  (h4 : total_money = 15) :
  (total_money - (bottle_deposit * bottles_recycled)) / can_deposit = 140 := by
sorry

end cans_recycled_l2628_262891


namespace solve_system_and_calculate_l2628_262835

theorem solve_system_and_calculate (x y : ℚ) 
  (eq1 : 2 * x + y = 26) 
  (eq2 : x + 2 * y = 10) : 
  (x + y) / 3 = 4 := by
sorry

end solve_system_and_calculate_l2628_262835


namespace log_equation_solution_l2628_262872

theorem log_equation_solution (x : ℝ) (h : x > 0) (eq : Real.log (729 : ℝ) / Real.log (3 * x) = x) :
  x = 3 ∧ ¬ ∃ n : ℕ, x = n^2 ∧ ¬ ∃ m : ℕ, x = m^3 ∧ ∃ k : ℕ, x = k := by
  sorry

end log_equation_solution_l2628_262872


namespace initial_population_size_l2628_262807

/-- Given a population with specific birth and death rates, prove the initial population size. -/
theorem initial_population_size 
  (P : ℕ) 
  (birth_rate : ℕ) 
  (death_rate : ℕ) 
  (net_growth_rate : ℚ) 
  (h1 : birth_rate = 52) 
  (h2 : death_rate = 16) 
  (h3 : net_growth_rate = 12/1000) 
  (h4 : (birth_rate - death_rate : ℚ) / P = net_growth_rate) : 
  P = 3000 := by
  sorry

end initial_population_size_l2628_262807


namespace polynomial_product_theorem_l2628_262859

theorem polynomial_product_theorem (p q : ℚ) : 
  (∀ x, (x^2 + p*x - 1/3) * (x^2 - 3*x + q) = x^4 + (q - 3*p - 1/3)*x^2 - q/3) → 
  (p = 3 ∧ q = -1/3 ∧ (-2*p^2*q)^2 + 3*p*q = 33) := by
  sorry

end polynomial_product_theorem_l2628_262859


namespace quadratic_inequality_solution_l2628_262897

-- Define the quadratic function
def f (a b x : ℝ) := x^2 - a*x - b

-- Define the solution set of the quadratic inequality
def solution_set (a b : ℝ) := {x : ℝ | 1 < x ∧ x < 3}

-- Theorem statement
theorem quadratic_inequality_solution 
  (a b : ℝ) 
  (h : ∀ x, x ∈ solution_set a b ↔ f a b x < 0) :
  a = 4 ∧ b = -3 ∧
  (∀ x, (2*x + a) / (x + b) > 1 ↔ x > -7 ∨ x > 3) :=
sorry

end quadratic_inequality_solution_l2628_262897


namespace tourist_tax_theorem_l2628_262800

/-- Calculates the tax paid given the total value of goods -/
def tax_paid (total_value : ℝ) : ℝ :=
  0.08 * (total_value - 600)

/-- Theorem stating that if $89.6 tax is paid, the total value of goods is $1720 -/
theorem tourist_tax_theorem (total_value : ℝ) :
  tax_paid total_value = 89.6 → total_value = 1720 := by
  sorry

end tourist_tax_theorem_l2628_262800


namespace different_color_probability_l2628_262825

def blue_chips : ℕ := 8
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4
def green_chips : ℕ := 3

def total_chips : ℕ := blue_chips + red_chips + yellow_chips + green_chips

def prob_different_colors : ℚ :=
  (blue_chips * (red_chips + yellow_chips + green_chips) +
   red_chips * (blue_chips + yellow_chips + green_chips) +
   yellow_chips * (blue_chips + red_chips + green_chips) +
   green_chips * (blue_chips + red_chips + yellow_chips)) /
  (total_chips * total_chips)

theorem different_color_probability :
  prob_different_colors = 143 / 200 := by
  sorry

end different_color_probability_l2628_262825


namespace nice_sequence_divisibility_exists_nice_sequence_not_divisible_l2628_262833

/-- Definition of a nice sequence -/
def NiceSequence (a : ℕ → ℕ) : Prop :=
  (∀ n, a n < a (n + 1)) ∧ (∀ n, a (2 * n) = 2 * a n)

theorem nice_sequence_divisibility (a : ℕ → ℕ) (p : ℕ) (hp : Prime p) (h_nice : NiceSequence a) (h_p_gt_a1 : p > a 1) :
  ∃ k, p ∣ a k := by
  sorry

theorem exists_nice_sequence_not_divisible (p : ℕ) (hp : Prime p) (h_p_gt_2 : p > 2) :
  ∃ a : ℕ → ℕ, NiceSequence a ∧ ∀ n, ¬(p ∣ a n) := by
  sorry

end nice_sequence_divisibility_exists_nice_sequence_not_divisible_l2628_262833


namespace equation_solution_l2628_262814

theorem equation_solution : ∃ y : ℝ, 
  (Real.sqrt (2 + Real.sqrt (3 + Real.sqrt y)) = (2 + Real.sqrt y) ^ (1/4)) ∧ 
  y = 81/256 := by
sorry

end equation_solution_l2628_262814


namespace intersection_points_coincide_l2628_262828

/-- Two circles in a plane -/
structure TwoCircles where
  /-- Center of the first circle -/
  center1 : ℝ × ℝ
  /-- Radius of the first circle -/
  radius1 : ℝ
  /-- Center of the second circle -/
  center2 : ℝ × ℝ
  /-- Radius of the second circle -/
  radius2 : ℝ

/-- The square of the distance between intersection points of two circles -/
def intersectionPointsDistanceSquared (circles : TwoCircles) : ℝ := sorry

/-- Theorem: The square of the distance between intersection points is zero for the given circles -/
theorem intersection_points_coincide (circles : TwoCircles) 
  (h1 : circles.center1 = (3, -2))
  (h2 : circles.radius1 = 5)
  (h3 : circles.center2 = (3, 6))
  (h4 : circles.radius2 = 3) :
  intersectionPointsDistanceSquared circles = 0 := by sorry

end intersection_points_coincide_l2628_262828


namespace blue_water_bottles_l2628_262826

theorem blue_water_bottles (red black : ℕ) (total removed remaining : ℕ) (blue : ℕ) : 
  red = 2 →
  black = 3 →
  total = red + black + blue →
  removed = 5 →
  remaining = 4 →
  total = removed + remaining →
  blue = 4 := by
sorry

end blue_water_bottles_l2628_262826


namespace repeating_decimal_as_fraction_l2628_262881

/-- Represents the repeating decimal 7.036036036... -/
def repeating_decimal : ℚ := 7 + 36 / 999

/-- The repeating decimal 7.036036036... is equal to the fraction 781/111 -/
theorem repeating_decimal_as_fraction : repeating_decimal = 781 / 111 := by
  sorry

end repeating_decimal_as_fraction_l2628_262881


namespace two_dice_same_side_probability_l2628_262802

/-- Represents a 10-sided die with specific side distributions -/
structure TenSidedDie :=
  (gold : Nat)
  (silver : Nat)
  (diamond : Nat)
  (rainbow : Nat)
  (total : Nat)
  (sides_sum : gold + silver + diamond + rainbow = total)

/-- The probability of rolling two dice and getting the same color or pattern -/
def sameSideProbability (die : TenSidedDie) : ℚ :=
  (die.gold ^ 2 + die.silver ^ 2 + die.diamond ^ 2 + die.rainbow ^ 2) / die.total ^ 2

/-- Theorem: The probability of rolling two 10-sided dice with the given distribution
    and getting the same color or pattern is 3/10 -/
theorem two_dice_same_side_probability :
  ∃ (die : TenSidedDie),
    die.gold = 3 ∧
    die.silver = 4 ∧
    die.diamond = 2 ∧
    die.rainbow = 1 ∧
    die.total = 10 ∧
    sameSideProbability die = 3 / 10 := by
  sorry

end two_dice_same_side_probability_l2628_262802


namespace shortest_chord_line_l2628_262812

/-- The circle C in the 2D plane -/
def C : Set (ℝ × ℝ) := {p | p.1^2 + (p.2 - 2)^2 = 5}

/-- The line l passing through (1,1) -/
def l : Set (ℝ × ℝ) := {p | p.1 - p.2 = 0}

/-- The point (1,1) -/
def A : ℝ × ℝ := (1, 1)

/-- Theorem: The line l intersects the circle C with the shortest chord length -/
theorem shortest_chord_line :
  A ∈ l ∧
  (∃ p q : ℝ × ℝ, p ∈ l ∧ q ∈ l ∧ p ∈ C ∧ q ∈ C ∧ p ≠ q) ∧
  (∀ m : Set (ℝ × ℝ), A ∈ m →
    (∃ p q : ℝ × ℝ, p ∈ m ∧ q ∈ m ∧ p ∈ C ∧ q ∈ C ∧ p ≠ q) →
    ∃ p q : ℝ × ℝ, p ∈ l ∧ q ∈ l ∧ p ∈ C ∧ q ∈ C ∧
    ∀ r s : ℝ × ℝ, r ∈ m ∧ s ∈ m ∧ r ∈ C ∧ s ∈ C →
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≤ Real.sqrt ((r.1 - s.1)^2 + (r.2 - s.2)^2)) :=
sorry

end shortest_chord_line_l2628_262812


namespace inequality_of_logarithms_l2628_262849

theorem inequality_of_logarithms (a b c : ℝ) 
  (ha : a = Real.log 2) 
  (hb : b = Real.log 3) 
  (hc : c = Real.log 5) : 
  c / 5 < a / 2 ∧ a / 2 < b / 3 := by sorry

end inequality_of_logarithms_l2628_262849


namespace extreme_value_derivative_l2628_262893

/-- A function has an extreme value at a point -/
def has_extreme_value (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

/-- The relationship between extreme values and derivative -/
theorem extreme_value_derivative (f : ℝ → ℝ) (x : ℝ) 
  (hf : Differentiable ℝ f) :
  (has_extreme_value f x → deriv f x = 0) ∧
  ∃ g : ℝ → ℝ, Differentiable ℝ g ∧ deriv g 0 = 0 ∧ ¬ has_extreme_value g 0 := by
  sorry

end extreme_value_derivative_l2628_262893


namespace min_distance_squared_l2628_262883

/-- Given real numbers a, b, c, d satisfying the condition,
    the minimum value of (a - c)^2 + (b - d)^2 is 25/2 -/
theorem min_distance_squared (a b c d : ℝ) 
  (h : (a - 2 * Real.exp a) / b = (2 - c) / (d - 1) ∧ (a - 2 * Real.exp a) / b = 1) :
  ∃ (min : ℝ), min = 25 / 2 ∧ ∀ (x y : ℝ), 
    (x - 2 * Real.exp x) / y = (2 - c) / (d - 1) ∧ (x - 2 * Real.exp x) / y = 1 →
    (x - c)^2 + (y - d)^2 ≥ min :=
by sorry

end min_distance_squared_l2628_262883


namespace same_color_probability_l2628_262858

/-- Represents a 12-sided die with colored sides -/
structure ColoredDie :=
  (pink : Nat)
  (green : Nat)
  (blue : Nat)
  (total : Nat)
  (h_total : pink + green + blue = total)

/-- The probability of two dice showing the same color -/
def samColorProbability (d : ColoredDie) : Rat :=
  (d.pink^2 + d.green^2 + d.blue^2) / d.total^2

/-- Two 12-sided dice with 3 pink, 4 green, and 5 blue sides each -/
def twelveSidedDie : ColoredDie :=
  { pink := 3
  , green := 4
  , blue := 5
  , total := 12
  , h_total := by rfl }

theorem same_color_probability :
  samColorProbability twelveSidedDie = 25 / 72 := by
  sorry

end same_color_probability_l2628_262858


namespace substitution_sequences_remainder_l2628_262834

/-- Represents the number of possible substitution sequences in a basketball game -/
def substitutionSequences (totalPlayers startingPlayers maxSubstitutions : ℕ) : ℕ :=
  let substitutes := totalPlayers - startingPlayers
  let a0 := 1  -- No substitutions
  let a1 := startingPlayers * substitutes  -- One substitution
  let a2 := a1 * (startingPlayers - 1) * (substitutes - 1)  -- Two substitutions
  let a3 := a2 * (startingPlayers - 2) * (substitutes - 2)  -- Three substitutions
  let a4 := a3 * (startingPlayers - 3) * (substitutes - 3)  -- Four substitutions
  a0 + a1 + a2 + a3 + a4

/-- The main theorem stating the remainder of substitution sequences divided by 100 -/
theorem substitution_sequences_remainder :
  substitutionSequences 15 5 4 % 100 = 51 := by
  sorry


end substitution_sequences_remainder_l2628_262834


namespace probability_at_least_one_heart_or_king_l2628_262895

def standard_deck : ℕ := 52
def hearts_and_kings : ℕ := 16

theorem probability_at_least_one_heart_or_king :
  let p : ℚ := 1 - (1 - hearts_and_kings / standard_deck) ^ 2
  p = 88 / 169 := by
sorry

end probability_at_least_one_heart_or_king_l2628_262895


namespace fourth_month_sale_proof_l2628_262815

/-- Calculates the sale in the fourth month given sales for other months and the average --/
def fourthMonthSale (sale1 sale2 sale3 sale5 sale6 average : ℕ) : ℕ :=
  6 * average - (sale1 + sale2 + sale3 + sale5 + sale6)

theorem fourth_month_sale_proof (sale1 sale2 sale3 sale5 sale6 average : ℕ) 
  (h1 : sale1 = 3435)
  (h2 : sale2 = 3927)
  (h3 : sale3 = 3855)
  (h5 : sale5 = 3562)
  (h6 : sale6 = 1991)
  (h_avg : average = 3500) :
  fourthMonthSale sale1 sale2 sale3 sale5 sale6 average = 4230 := by
  sorry

#eval fourthMonthSale 3435 3927 3855 3562 1991 3500

end fourth_month_sale_proof_l2628_262815


namespace prob_allison_between_brian_and_noah_l2628_262851

/-- Represents a 6-sided cube with specific face values -/
structure Cube :=
  (faces : Fin 6 → ℕ)

/-- Allison's cube with all faces showing 6 -/
def allison_cube : Cube :=
  { faces := λ _ => 6 }

/-- Brian's cube with faces numbered 1 to 6 -/
def brian_cube : Cube :=
  { faces := λ i => i.val + 1 }

/-- Noah's cube with three faces showing 4 and three faces showing 7 -/
def noah_cube : Cube :=
  { faces := λ i => if i.val < 3 then 4 else 7 }

/-- The probability of rolling a specific value or higher on a given cube -/
def prob_roll_ge (c : Cube) (n : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i ≥ n) (Finset.univ : Finset (Fin 6))).card / 6

/-- The probability of rolling a specific value or lower on a given cube -/
def prob_roll_le (c : Cube) (n : ℕ) : ℚ :=
  (Finset.filter (λ i => c.faces i ≤ n) (Finset.univ : Finset (Fin 6))).card / 6

/-- The main theorem stating the probability of Allison rolling higher than Brian but lower than Noah -/
theorem prob_allison_between_brian_and_noah :
  prob_roll_ge brian_cube 6 * prob_roll_ge noah_cube 7 = 5 / 12 := by
  sorry

end prob_allison_between_brian_and_noah_l2628_262851


namespace sam_total_dimes_l2628_262829

def initial_dimes : ℕ := 9
def received_dimes : ℕ := 7

theorem sam_total_dimes : initial_dimes + received_dimes = 16 := by
  sorry

end sam_total_dimes_l2628_262829


namespace unique_triplet_solution_l2628_262848

theorem unique_triplet_solution (a b p : ℕ+) (h_prime : Nat.Prime p) :
  (a + b : ℕ+) ^ (p : ℕ) = p ^ (a : ℕ) + p ^ (b : ℕ) ↔ a = 1 ∧ b = 1 ∧ p = 2 := by
  sorry

end unique_triplet_solution_l2628_262848


namespace both_first_prize_probability_X_distribution_l2628_262877

structure StudentPopulation where
  total : ℕ
  male : ℕ
  female : ℕ
  male_first_prize : ℕ
  male_second_prize : ℕ
  male_third_prize : ℕ
  female_first_prize : ℕ
  female_second_prize : ℕ
  female_third_prize : ℕ

def sample : StudentPopulation := {
  total := 500,
  male := 200,
  female := 300,
  male_first_prize := 10,
  male_second_prize := 15,
  male_third_prize := 15,
  female_first_prize := 25,
  female_second_prize := 25,
  female_third_prize := 40
}

def prob_both_first_prize (s : StudentPopulation) : ℚ :=
  (s.male_first_prize : ℚ) / s.male * (s.female_first_prize : ℚ) / s.female

def prob_male_award (s : StudentPopulation) : ℚ :=
  (s.male_first_prize + s.male_second_prize + s.male_third_prize : ℚ) / s.male

def prob_female_award (s : StudentPopulation) : ℚ :=
  (s.female_first_prize + s.female_second_prize + s.female_third_prize : ℚ) / s.female

def prob_X (s : StudentPopulation) : Fin 3 → ℚ
| 0 => (1 - prob_male_award s) * (1 - prob_female_award s)
| 1 => 1 - (1 - prob_male_award s) * (1 - prob_female_award s) - prob_male_award s * prob_female_award s
| 2 => prob_male_award s * prob_female_award s

theorem both_first_prize_probability :
  prob_both_first_prize sample = 1 / 240 := by sorry

theorem X_distribution :
  prob_X sample 0 = 28 / 50 ∧
  prob_X sample 1 = 19 / 50 ∧
  prob_X sample 2 = 3 / 50 := by sorry

end both_first_prize_probability_X_distribution_l2628_262877


namespace cookie_problem_l2628_262853

theorem cookie_problem (initial_cookies : ℕ) : 
  (initial_cookies : ℚ) * (1/4) * (1/2) = 8 → initial_cookies = 64 := by
  sorry

end cookie_problem_l2628_262853


namespace book_sale_loss_l2628_262837

/-- Represents the sale of two books with given conditions -/
def book_sale (total_cost cost_book1 loss_percent1 gain_percent2 : ℚ) : ℚ :=
  let cost_book2 := total_cost - cost_book1
  let selling_price1 := cost_book1 * (1 - loss_percent1 / 100)
  let selling_price2 := cost_book2 * (1 + gain_percent2 / 100)
  let total_selling_price := selling_price1 + selling_price2
  total_cost - total_selling_price

/-- Theorem stating the overall loss from the book sale -/
theorem book_sale_loss :
  book_sale 460 268.33 15 19 = 3.8322 := by
  sorry

end book_sale_loss_l2628_262837


namespace union_condition_intersection_condition_l2628_262864

-- Define sets A and B
def A : Set ℝ := {x | x^2 + 4*x = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

-- Theorem for the first question
theorem union_condition (a : ℝ) : A ∪ B a = B a → a = 1 := by sorry

-- Theorem for the second question
theorem intersection_condition (a : ℝ) : A ∩ B a = B a → a ≤ -1 ∨ a = 1 := by sorry

end union_condition_intersection_condition_l2628_262864


namespace proportion_third_number_l2628_262819

theorem proportion_third_number : 
  ∀ y : ℝ, (0.75 : ℝ) / 0.6 = y / 8 → y = 10 := by
  sorry

end proportion_third_number_l2628_262819


namespace consistency_condition_l2628_262831

theorem consistency_condition (a b c d x y z : ℝ) 
  (eq1 : y + z = a)
  (eq2 : x + y = b)
  (eq3 : x + z = c)
  (eq4 : x + y + z = d) :
  a + b + c = 2 * d := by
  sorry

end consistency_condition_l2628_262831


namespace total_age_is_22_l2628_262886

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - b is 8 years old
  Prove that the total of their ages is 22 years. -/
theorem total_age_is_22 (a b c : ℕ) : 
  b = 8 → 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 22 := by sorry

end total_age_is_22_l2628_262886


namespace square_form_existence_l2628_262811

theorem square_form_existence (a b : ℕ+) (h : a.val^3 + 4 * a.val = b.val^2) :
  ∃ t : ℕ+, a.val = 2 * t.val^2 := by
sorry

end square_form_existence_l2628_262811


namespace geometric_sequence_sum_range_l2628_262808

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_sum_range
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_prod : a 4 * a 8 = 9) :
  ∀ x : ℝ, (x ∈ Set.Iic (-6) ∪ Set.Ici 6) ↔ ∃ (a₃ a₉ : ℝ), a 3 = a₃ ∧ a 9 = a₉ ∧ a₃ + a₉ = x :=
sorry

end geometric_sequence_sum_range_l2628_262808


namespace age_problem_l2628_262869

theorem age_problem (a b c : ℕ) : 
  a = b + 2 → 
  b = 2 * c → 
  a + b + c = 32 → 
  b = 12 := by sorry

end age_problem_l2628_262869


namespace pirate_total_distance_l2628_262804

def island1_distances : List ℝ := [10, 15, 20]
def island1_increase : ℝ := 1.1
def island2_distance : ℝ := 40
def island2_increase : ℝ := 1.15
def island3_morning : ℝ := 25
def island3_afternoon : ℝ := 20
def island3_days : ℕ := 2
def island3_increase : ℝ := 1.2
def island4_distance : ℝ := 35
def island4_increase : ℝ := 1.25

theorem pirate_total_distance :
  let island1_total := (island1_distances.map (· * island1_increase)).sum
  let island2_total := island2_distance * island2_increase
  let island3_total := (island3_morning + island3_afternoon) * island3_increase * island3_days
  let island4_total := island4_distance * island4_increase
  island1_total + island2_total + island3_total + island4_total = 247.25 := by
  sorry

end pirate_total_distance_l2628_262804


namespace regular_ngon_construction_l2628_262843

/-- Theorem about the construction of points on the extensions of a regular n-gon's sides -/
theorem regular_ngon_construction (n : ℕ) (a : ℝ) (h_n : n ≥ 5) :
  let α : ℝ := π - (2 * π) / n
  ∀ (x : ℕ → ℝ), 
    (∀ k, x k = (a + x ((k + 1) % n)) * Real.cos α) →
    ∀ k, x k = (a * Real.cos α) / (1 - Real.cos α) := by
  sorry

end regular_ngon_construction_l2628_262843
