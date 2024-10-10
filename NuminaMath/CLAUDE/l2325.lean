import Mathlib

namespace age_difference_l2325_232506

def hiram_age : ℕ := 40
def allyson_age : ℕ := 28

theorem age_difference : 
  (2 * allyson_age) - (hiram_age + 12) = 4 :=
by sorry

end age_difference_l2325_232506


namespace problem_triangle_count_l2325_232558

/-- Represents a rectangle subdivided into sections with diagonal lines -/
structure SubdividedRectangle where
  vertical_sections : Nat
  horizontal_sections : Nat
  has_diagonals : Bool

/-- Counts the number of triangles in a subdivided rectangle -/
def count_triangles (rect : SubdividedRectangle) : Nat :=
  sorry

/-- The specific rectangle from the problem -/
def problem_rectangle : SubdividedRectangle :=
  { vertical_sections := 4
  , horizontal_sections := 2
  , has_diagonals := true }

/-- Theorem stating that the number of triangles in the problem rectangle is 42 -/
theorem problem_triangle_count : count_triangles problem_rectangle = 42 := by
  sorry

end problem_triangle_count_l2325_232558


namespace min_value_fraction_l2325_232578

theorem min_value_fraction (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (m : ℝ), m = 1 - Real.sqrt 2 ∧ ∀ (z : ℝ), z = x*y/(x+y-2) → m ≤ z :=
sorry

end min_value_fraction_l2325_232578


namespace shooting_probabilities_l2325_232569

/-- Probability of a person hitting a target -/
def prob_hit (p : ℝ) : ℝ := p

/-- Probability of missing at least once in n shots -/
def prob_miss_at_least_once (p : ℝ) (n : ℕ) : ℝ := 1 - p^n

/-- Probability of stopping exactly after n shots, given stopping after two consecutive misses -/
def prob_stop_after_n_shots (p : ℝ) (n : ℕ) : ℝ :=
  if n < 2 then 0
  else if n = 2 then (1 - p)^2
  else p * (prob_stop_after_n_shots p (n - 1)) + (1 - p) * p * (1 - p)^2

theorem shooting_probabilities :
  let pA := prob_hit (2/3)
  let pB := prob_hit (3/4)
  (prob_miss_at_least_once pA 4 = 65/81) ∧
  (prob_stop_after_n_shots pB 5 = 45/1024) :=
by sorry

end shooting_probabilities_l2325_232569


namespace profit_threshold_l2325_232550

/-- Represents the minimum number of workers needed for profit -/
def min_workers_for_profit (
  daily_maintenance : ℕ)
  (hourly_wage : ℕ)
  (gadgets_per_hour : ℕ)
  (gadget_price : ℕ)
  (workday_hours : ℕ) : ℕ :=
  16

theorem profit_threshold (
  daily_maintenance : ℕ)
  (hourly_wage : ℕ)
  (gadgets_per_hour : ℕ)
  (gadget_price : ℕ)
  (workday_hours : ℕ)
  (h1 : daily_maintenance = 600)
  (h2 : hourly_wage = 20)
  (h3 : gadgets_per_hour = 6)
  (h4 : gadget_price = 4)
  (h5 : workday_hours = 10) :
  ∀ n : ℕ, n ≥ min_workers_for_profit daily_maintenance hourly_wage gadgets_per_hour gadget_price workday_hours →
    n * workday_hours * gadgets_per_hour * gadget_price > daily_maintenance + n * workday_hours * hourly_wage :=
by sorry

#check profit_threshold

end profit_threshold_l2325_232550


namespace largest_valid_number_l2325_232525

def is_valid (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p ∣ n → (p^2 - 1) ∣ n

theorem largest_valid_number : 
  (1944 < 2012) ∧ 
  is_valid 1944 ∧ 
  (∀ m : ℕ, 1944 < m → m < 2012 → ¬ is_valid m) :=
sorry

end largest_valid_number_l2325_232525


namespace sin_arccos_twelve_thirteenths_l2325_232510

theorem sin_arccos_twelve_thirteenths : Real.sin (Real.arccos (12/13)) = 5/13 := by
  sorry

end sin_arccos_twelve_thirteenths_l2325_232510


namespace total_points_is_1390_l2325_232596

-- Define the points scored in each try
def first_try : ℕ := 400
def second_try : ℕ := first_try - 70
def third_try : ℕ := 2 * second_try

-- Define the total points
def total_points : ℕ := first_try + second_try + third_try

-- Theorem statement
theorem total_points_is_1390 : total_points = 1390 := by
  sorry

end total_points_is_1390_l2325_232596


namespace min_value_expression_min_value_attainable_l2325_232586

theorem min_value_expression (x y : ℝ) : 
  x^2 + 4*x*Real.sin y - 4*(Real.cos y)^2 ≥ -4 :=
by sorry

theorem min_value_attainable : 
  ∃ (x y : ℝ), x^2 + 4*x*Real.sin y - 4*(Real.cos y)^2 = -4 :=
by sorry

end min_value_expression_min_value_attainable_l2325_232586


namespace f_less_than_g_iff_m_in_range_l2325_232555

-- Define the functions f and g
def f (x m : ℝ) : ℝ := |x - 1| + |x + m|
def g (x : ℝ) : ℝ := 2 * x - 1

-- State the theorem
theorem f_less_than_g_iff_m_in_range :
  ∀ m : ℝ, (∀ x ∈ Set.Icc (-m) 1, f x m < g x) ↔ -1 < m ∧ m < -2/3 := by sorry

end f_less_than_g_iff_m_in_range_l2325_232555


namespace even_perfect_square_factors_count_l2325_232592

def num_factors (n : ℕ) : ℕ := sorry

def is_even_perfect_square (n : ℕ) : Prop := sorry

theorem even_perfect_square_factors_count : 
  ∃ (f : ℕ → ℕ), 
    (∀ x, is_even_perfect_square (f x)) ∧ 
    (∀ x, f x ∣ (2^6 * 5^3 * 7^8)) ∧ 
    (num_factors (2^6 * 5^3 * 7^8) = 30) := by
  sorry

end even_perfect_square_factors_count_l2325_232592


namespace actual_distance_walked_l2325_232590

/-- 
Given a person who walks at two different speeds for the same duration:
- At 5 km/hr, they cover a distance D
- At 15 km/hr, they would cover a distance D + 20 km
This theorem proves that the actual distance D is 10 km.
-/
theorem actual_distance_walked (D : ℝ) : 
  (D / 5 = (D + 20) / 15) → D = 10 := by sorry

end actual_distance_walked_l2325_232590


namespace cobbler_efficiency_l2325_232514

/-- Represents the cobbler's work schedule and output --/
structure CobblerSchedule where
  hours_per_day : ℕ -- Hours worked per day from Monday to Thursday
  friday_hours : ℕ  -- Hours worked on Friday
  shoes_per_week : ℕ -- Number of shoes mended in a week

/-- Calculates the number of shoes mended per hour --/
def shoes_per_hour (schedule : CobblerSchedule) : ℚ :=
  schedule.shoes_per_week / (4 * schedule.hours_per_day + schedule.friday_hours)

/-- Theorem stating that the cobbler mends 3 shoes per hour --/
theorem cobbler_efficiency (schedule : CobblerSchedule) 
  (h1 : schedule.hours_per_day = 8)
  (h2 : schedule.friday_hours = 3)
  (h3 : schedule.shoes_per_week = 105) :
  shoes_per_hour schedule = 3 := by
  sorry

#eval shoes_per_hour ⟨8, 3, 105⟩

end cobbler_efficiency_l2325_232514


namespace parabola_with_conditions_l2325_232598

/-- A parabola passing through specific points with a specific tangent line -/
theorem parabola_with_conditions (a b c : ℝ) :
  (1 : ℝ)^2 * a + b * 1 + c = 1 →  -- Parabola passes through (1, 1)
  (2 : ℝ)^2 * a + b * 2 + c = -1 →  -- Parabola passes through (2, -1)
  2 * a * 2 + b = 1 →  -- Tangent line at (2, -1) is parallel to y = x - 3
  a = 3 ∧ b = -11 ∧ c = 9 := by
sorry

end parabola_with_conditions_l2325_232598


namespace cubic_equation_solution_l2325_232515

theorem cubic_equation_solution (x y z n : ℕ+) :
  x^3 + y^3 + z^3 = n * x^2 * y^2 * z^2 ↔ n = 1 ∨ n = 3 := by
  sorry

end cubic_equation_solution_l2325_232515


namespace complement_of_B_in_A_l2325_232539

def A : Set ℕ := {0, 2, 4, 6, 8, 10}
def B : Set ℕ := {4, 8}

theorem complement_of_B_in_A :
  (A \ B) = {0, 2, 6, 10} := by sorry

end complement_of_B_in_A_l2325_232539


namespace dihedral_angle_range_l2325_232595

/-- The dihedral angle between two adjacent faces in a regular n-sided polyhedron -/
def dihedralAngle (n : ℕ) (θ : ℝ) : Prop :=
  n ≥ 3 ∧ ((n - 2 : ℝ) / n) * Real.pi < θ ∧ θ < Real.pi

/-- Theorem stating the range of the dihedral angle in a regular n-sided polyhedron -/
theorem dihedral_angle_range (n : ℕ) :
  ∃ θ : ℝ, dihedralAngle n θ :=
sorry

end dihedral_angle_range_l2325_232595


namespace pi_irrational_l2325_232565

def is_rational (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b

theorem pi_irrational :
  is_rational (-4/7) →
  is_rational 3.333333 →
  is_rational 1.010010001 →
  ¬ is_rational Real.pi :=
by sorry

end pi_irrational_l2325_232565


namespace sqrt_three_sum_product_l2325_232507

theorem sqrt_three_sum_product : Real.sqrt 3 * (Real.sqrt 3 + Real.sqrt 27) = 12 := by
  sorry

end sqrt_three_sum_product_l2325_232507


namespace sum_of_altitudes_equals_2432_div_17_l2325_232547

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 15 * x + 8 * y = 120

-- Define the triangle formed by the line and coordinate axes
def triangle : Set (ℝ × ℝ) :=
  {p | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ line_equation p.1 p.2}

-- Define the function to calculate the sum of altitudes
def sum_of_altitudes (t : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem sum_of_altitudes_equals_2432_div_17 :
  sum_of_altitudes triangle = 2432 / 17 := by sorry

end sum_of_altitudes_equals_2432_div_17_l2325_232547


namespace rectangle_area_l2325_232554

/-- The area of a rectangle with sides 1.5 meters and 0.75 meters is 1.125 square meters. -/
theorem rectangle_area : 
  let length : ℝ := 1.5
  let width : ℝ := 0.75
  length * width = 1.125 := by
  sorry

end rectangle_area_l2325_232554


namespace peyton_juice_boxes_l2325_232546

/-- Calculate the total number of juice boxes needed for Peyton's children for the school year -/
def total_juice_boxes (num_children : ℕ) (school_days_per_week : ℕ) (weeks_in_school_year : ℕ) : ℕ :=
  num_children * school_days_per_week * weeks_in_school_year

/-- Proof that Peyton needs 375 juice boxes for the entire school year for all of her children -/
theorem peyton_juice_boxes :
  total_juice_boxes 3 5 25 = 375 := by
  sorry

end peyton_juice_boxes_l2325_232546


namespace sticker_count_l2325_232583

def ryan_stickers : ℕ := 30

def steven_stickers (ryan : ℕ) : ℕ := 3 * ryan

def terry_stickers (steven : ℕ) : ℕ := steven + 20

def total_stickers (ryan steven terry : ℕ) : ℕ := ryan + steven + terry

theorem sticker_count :
  total_stickers ryan_stickers (steven_stickers ryan_stickers) (terry_stickers (steven_stickers ryan_stickers)) = 230 := by
  sorry

end sticker_count_l2325_232583


namespace range_of_a_l2325_232533

open Set Real

def A : Set ℝ := {x | 1 ≤ x ∧ x < 3}

def B (a : ℝ) : Set ℝ := {x | x^2 - a*x ≤ x - a}

theorem range_of_a :
  ∀ a : ℝ, (B a ⊆ A) ↔ (1 ≤ a ∧ a < 3) :=
sorry

end range_of_a_l2325_232533


namespace converse_proposition_l2325_232534

theorem converse_proposition : 
  (∀ x : ℝ, x > 0 → x^2 - 1 > 0) ↔ 
  (∀ x : ℝ, x^2 - 1 > 0 → x > 0) :=
by sorry

end converse_proposition_l2325_232534


namespace figure_102_squares_l2325_232588

/-- A function representing the number of non-overlapping unit squares in the nth figure -/
def g (n : ℕ) : ℕ := 2 * n^2 - 2 * n + 1

/-- Theorem stating that the 102nd figure contains 20605 non-overlapping unit squares -/
theorem figure_102_squares : g 102 = 20605 := by
  sorry

/-- Lemma verifying the given initial conditions -/
lemma initial_conditions :
  g 1 = 1 ∧ g 2 = 5 ∧ g 3 = 13 ∧ g 4 = 25 := by
  sorry

end figure_102_squares_l2325_232588


namespace solve_system_l2325_232542

theorem solve_system (a b : ℚ) 
  (h1 : -3 / (a - 3) = 3 / (a + 2))
  (h2 : (a^2 - b^2)/(a - b) = 7) :
  a = 1/2 ∧ b = 13/2 := by
sorry

end solve_system_l2325_232542


namespace sufficient_not_necessary_l2325_232508

theorem sufficient_not_necessary (x : ℝ) : 
  (x = 2 → (x - 2) * (x - 1) = 0) ∧ 
  ¬((x - 2) * (x - 1) = 0 → x = 2) :=
by sorry

end sufficient_not_necessary_l2325_232508


namespace stratified_sampling_most_appropriate_l2325_232543

/-- Represents different sampling methods --/
inductive SamplingMethod
  | Lottery
  | RandomNumber
  | Systematic
  | Stratified

/-- Represents a school population --/
structure SchoolPopulation where
  male_students : Nat
  female_students : Nat

/-- Represents a survey plan --/
structure SurveyPlan where
  population : SchoolPopulation
  sample_size : Nat
  goal : String

/-- Determines the most appropriate sampling method for a given survey plan --/
def most_appropriate_sampling_method (plan : SurveyPlan) : SamplingMethod :=
  sorry

/-- The theorem stating that stratified sampling is most appropriate for the given scenario --/
theorem stratified_sampling_most_appropriate (plan : SurveyPlan) :
  plan.population.male_students = 500 →
  plan.population.female_students = 500 →
  plan.sample_size = 100 →
  plan.goal = "investigate differences in study interests and hobbies between male and female students" →
  most_appropriate_sampling_method plan = SamplingMethod.Stratified :=
  sorry

end stratified_sampling_most_appropriate_l2325_232543


namespace fraction_sum_difference_equals_half_l2325_232527

theorem fraction_sum_difference_equals_half : 
  (3 : ℚ) / 9 + 5 / 12 - 1 / 4 = 1 / 2 := by sorry

end fraction_sum_difference_equals_half_l2325_232527


namespace functional_equation_solution_l2325_232548

-- Define the function type
def FunctionType := ℝ → ℝ

-- State the theorem
theorem functional_equation_solution (c : ℝ) (h_c : c > 1) (f : FunctionType) 
  (h_f : ∀ x y : ℝ, f (x + y) = f x * f y - c * Real.sin x * Real.sin y) :
  (∀ t : ℝ, f t = Real.sqrt (c - 1) * Real.sin t + Real.cos t) ∨ 
  (∀ t : ℝ, f t = -Real.sqrt (c - 1) * Real.sin t + Real.cos t) :=
by sorry

end functional_equation_solution_l2325_232548


namespace union_of_A_and_B_l2325_232528

def A : Set ℕ := {1, 2, 3, 5}
def B : Set ℕ := {2, 3, 6}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 5, 6} := by
  sorry

end union_of_A_and_B_l2325_232528


namespace binomial_variance_problem_l2325_232557

-- Define the binomial distribution
def binomial_distribution (n : ℕ) (p : ℝ) : ℕ → ℝ := sorry

-- Define the probability mass function for ξ = 1
def prob_xi_equals_one (n : ℕ) : ℝ := binomial_distribution n (1/2) 1

-- Define the variance of the binomial distribution
def variance_binomial (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

theorem binomial_variance_problem (n : ℕ) (h1 : 3 ≤ n) (h2 : n ≤ 8) 
  (h3 : prob_xi_equals_one n = 3/32) :
  variance_binomial n (1/2) = 3/2 := by sorry

end binomial_variance_problem_l2325_232557


namespace k_range_when_f_less_than_bound_l2325_232541

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (1/2) * x^2 + (1-k)*x - k * Real.log x

theorem k_range_when_f_less_than_bound (k : ℝ) (h_k_pos : k > 0) :
  (∃ x₀ : ℝ, f k x₀ < 3/2 - k^2) → 0 < k ∧ k < 1 := by sorry

end k_range_when_f_less_than_bound_l2325_232541


namespace min_value_theorem_l2325_232599

theorem min_value_theorem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2*a + b)/c + (2*a + c)/b + (2*b + c)/a ≥ 6 ∧
  ((2*a + b)/c + (2*a + c)/b + (2*b + c)/a = 6 ↔ 2*a = b ∧ b = c) :=
sorry

end min_value_theorem_l2325_232599


namespace angle_bisector_points_sum_l2325_232589

theorem angle_bisector_points_sum (a b : ℝ) : 
  ((-4 : ℝ) = -4 ∧ a = -4) → 
  ((-2 : ℝ) = -2 ∧ b = -2) → 
  a + b + a * b = 2 := by
  sorry

end angle_bisector_points_sum_l2325_232589


namespace candy_distribution_l2325_232591

theorem candy_distribution (C n : ℕ) 
  (h1 : C = 8 * n + 4)
  (h2 : C = 11 * (n - 1)) : 
  n = 5 := by sorry

end candy_distribution_l2325_232591


namespace equal_part_implies_a_eq_neg_two_l2325_232519

/-- A complex number is an "equal part complex number" if its real and imaginary parts are equal -/
def is_equal_part (z : ℂ) : Prop := z.re = z.im

/-- The complex number z defined in terms of a real number a -/
def z (a : ℝ) : ℂ := Complex.I * (2 + a * Complex.I)

/-- Theorem: If z(a) is an equal part complex number, then a = -2 -/
theorem equal_part_implies_a_eq_neg_two (a : ℝ) :
  is_equal_part (z a) → a = -2 := by sorry

end equal_part_implies_a_eq_neg_two_l2325_232519


namespace number_problem_l2325_232574

theorem number_problem (x : ℚ) (h : x - (3/5) * x = 62) : x = 155 := by
  sorry

end number_problem_l2325_232574


namespace unique_base_for_1024_l2325_232577

theorem unique_base_for_1024 : ∃! b : ℕ, 4 ≤ b ∧ b ≤ 12 ∧ 1024 % b = 1 := by
  sorry

end unique_base_for_1024_l2325_232577


namespace percent_of_percent_equality_l2325_232532

theorem percent_of_percent_equality (y : ℝ) : (0.3 * (0.6 * y)) = (0.18 * y) := by
  sorry

end percent_of_percent_equality_l2325_232532


namespace light_bulb_probability_l2325_232517

/-- The probability of exactly k successes in n independent trials -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

/-- The probability that a light bulb lasts more than 1000 hours -/
def p_success : ℝ := 0.2

/-- The number of light bulbs -/
def n : ℕ := 3

/-- The number of light bulbs that fail -/
def k : ℕ := 1

theorem light_bulb_probability : 
  binomial_probability n k p_success = 0.096 := by
  sorry

end light_bulb_probability_l2325_232517


namespace tangent_lines_count_l2325_232524

/-- Represents a circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in a 2D plane -/
structure Line where
  -- Add appropriate fields for a line

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Counts the number of lines tangent to both circles -/
def countTangentLines (c1 c2 : Circle) : ℕ := sorry

/-- The main theorem -/
theorem tangent_lines_count 
  (c1 c2 : Circle) 
  (h1 : c1.radius = 5) 
  (h2 : c2.radius = 8) 
  (h3 : Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2) = 13) :
  countTangentLines c1 c2 = 3 := by sorry

end tangent_lines_count_l2325_232524


namespace safe_dish_fraction_is_one_ninth_l2325_232556

/-- Represents a restaurant menu with vegan and nut-containing dishes -/
structure Menu where
  total_dishes : ℕ
  vegan_dishes : ℕ
  vegan_with_nuts : ℕ
  vegan_fraction : Rat
  h_vegan_fraction : vegan_fraction = 1 / 3
  h_vegan_dishes : vegan_dishes = 6
  h_vegan_with_nuts : vegan_with_nuts = 4

/-- The fraction of dishes that are both vegan and nut-free -/
def safe_dish_fraction (m : Menu) : Rat :=
  (m.vegan_dishes - m.vegan_with_nuts) / m.total_dishes

/-- Theorem stating that the fraction of safe dishes is 1/9 -/
theorem safe_dish_fraction_is_one_ninth (m : Menu) : safe_dish_fraction m = 1 / 9 := by
  sorry

end safe_dish_fraction_is_one_ninth_l2325_232556


namespace competition_matches_l2325_232512

theorem competition_matches (n : ℕ) (h : n = 6) : n * (n - 1) / 2 = 15 := by
  sorry

end competition_matches_l2325_232512


namespace jerry_insult_points_l2325_232520

/-- Represents the point system in Mrs. Carlton's class -/
structure PointSystem where
  interrupt_points : ℕ
  throw_points : ℕ
  office_threshold : ℕ

/-- Represents Jerry's behavior -/
structure JerryBehavior where
  interrupts : ℕ
  insults : ℕ
  throws : ℕ

/-- Calculates the points for insults given the point system and Jerry's behavior -/
def insult_points (ps : PointSystem) (jb : JerryBehavior) : ℕ :=
  (ps.office_threshold - (ps.interrupt_points * jb.interrupts + ps.throw_points * jb.throws)) / jb.insults

/-- Theorem stating that Jerry gets 10 points for insulting his classmates -/
theorem jerry_insult_points :
  let ps : PointSystem := { interrupt_points := 5, throw_points := 25, office_threshold := 100 }
  let jb : JerryBehavior := { interrupts := 2, insults := 4, throws := 2 }
  insult_points ps jb = 10 := by
  sorry

end jerry_insult_points_l2325_232520


namespace unique_positive_solution_l2325_232505

def f (x : ℝ) := x^12 + 5*x^11 + 20*x^10 + 1300*x^9 - 1105*x^8

theorem unique_positive_solution :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end unique_positive_solution_l2325_232505


namespace bell_interval_problem_l2325_232553

theorem bell_interval_problem (x : ℕ) :
  (∃ n : ℕ, n > 0 ∧ n * 5 = 1320) ∧
  (∃ n : ℕ, n > 0 ∧ n * 8 = 1320) ∧
  (∃ n : ℕ, n > 0 ∧ n * x = 1320) ∧
  (∃ n : ℕ, n > 0 ∧ n * 15 = 1320) →
  x = 11 := by
sorry

end bell_interval_problem_l2325_232553


namespace value_of_a_l2325_232593

/-- Proves that if 0.5% of a equals 70 paise, then a equals 140 rupees. -/
theorem value_of_a (a : ℝ) : (0.5 / 100) * a = 70 / 100 → a = 140 := by
  sorry

end value_of_a_l2325_232593


namespace equation_is_hyperbola_l2325_232561

/-- Represents a conic section --/
inductive ConicSection
  | Parabola
  | Circle
  | Ellipse
  | Hyperbola
  | Point
  | Line
  | TwoLines
  | Empty

/-- Determines the type of conic section for the given equation --/
def determineConicSection (a b c d e f : ℝ) : ConicSection :=
  sorry

/-- The equation x^2 - 25y^2 - 10x + 50 = 0 represents a hyperbola --/
theorem equation_is_hyperbola :
  determineConicSection 1 (-25) 0 (-10) 0 50 = ConicSection.Hyperbola :=
sorry

end equation_is_hyperbola_l2325_232561


namespace melody_reading_pages_l2325_232567

def english_pages : ℕ := 20
def science_pages : ℕ := 16
def civics_pages : ℕ := 8
def total_pages_tomorrow : ℕ := 14

def chinese_pages : ℕ := 12

theorem melody_reading_pages : 
  (english_pages / 4 + science_pages / 4 + civics_pages / 4 + chinese_pages / 4 = total_pages_tomorrow) ∧
  (chinese_pages ≥ 0) := by
  sorry

end melody_reading_pages_l2325_232567


namespace repeating_decimal_sum_l2325_232502

theorem repeating_decimal_sum : 
  (2 : ℚ) / 9 + 5 / 99 + 3 / 9999 = 910 / 3333 := by sorry

end repeating_decimal_sum_l2325_232502


namespace product_and_reciprocal_relation_l2325_232564

theorem product_and_reciprocal_relation (x y : ℝ) 
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) 
  (h_product : x * y = 16) 
  (h_reciprocal : 1 / x = 3 * (1 / y)) :
  2 * y - x = 24 - (4 * Real.sqrt 3) / 3 := by
sorry

end product_and_reciprocal_relation_l2325_232564


namespace two_recess_breaks_l2325_232526

/-- Calculates the number of 15-minute recess breaks given the total time outside class,
    lunch duration, and additional recess duration. -/
def numberOfRecessBreaks (totalTimeOutside lunchDuration additionalRecessDuration : ℕ) : ℕ :=
  ((totalTimeOutside - lunchDuration - additionalRecessDuration) / 15)

/-- Proves that given the specified conditions, students get 2 fifteen-minute recess breaks. -/
theorem two_recess_breaks :
  let totalTimeOutside : ℕ := 80
  let lunchDuration : ℕ := 30
  let additionalRecessDuration : ℕ := 20
  numberOfRecessBreaks totalTimeOutside lunchDuration additionalRecessDuration = 2 := by
sorry


end two_recess_breaks_l2325_232526


namespace speed_difference_l2325_232531

/-- Proves that the difference in average speed between two people traveling the same distance,
    where one travels at 12 miles per hour and the other completes the journey in 10 minutes,
    is 24 miles per hour. -/
theorem speed_difference (distance : ℝ) (speed_maya : ℝ) (time_naomi : ℝ) : 
  distance > 0 ∧ speed_maya = 12 ∧ time_naomi = 1/6 →
  (distance / time_naomi) - speed_maya = 24 :=
by sorry

end speed_difference_l2325_232531


namespace passing_marks_calculation_l2325_232562

theorem passing_marks_calculation (T : ℝ) (P : ℝ) : 
  (0.20 * T = P - 40) → 
  (0.30 * T = P + 20) → 
  P = 160 := by
sorry

end passing_marks_calculation_l2325_232562


namespace quadratic_equation_iff_m_eq_neg_one_l2325_232545

/-- The equation is quadratic if and only if m = -1 -/
theorem quadratic_equation_iff_m_eq_neg_one (m : ℝ) : 
  (∀ x, (m - 1) * x^(m^2 + 1) - x - 2 = 0 ↔ ∃ a b c, a ≠ 0 ∧ a * x^2 + b * x + c = 0) ↔ 
  m = -1 := by
sorry

end quadratic_equation_iff_m_eq_neg_one_l2325_232545


namespace add_three_preserves_inequality_l2325_232537

theorem add_three_preserves_inequality (a b : ℝ) (h : a > b) : a + 3 > b + 3 := by
  sorry

end add_three_preserves_inequality_l2325_232537


namespace car_race_distance_l2325_232576

theorem car_race_distance (karen_speed tom_speed : ℝ) (karen_delay : ℝ) (win_margin : ℝ) : 
  karen_speed = 75 →
  tom_speed = 50 →
  karen_delay = 7 / 60 →
  win_margin = 5 →
  (karen_speed * (tom_speed * win_margin / (karen_speed - tom_speed) + karen_delay) - 
   tom_speed * (tom_speed * win_margin / (karen_speed - tom_speed) + karen_delay)) = win_margin →
  tom_speed * (tom_speed * win_margin / (karen_speed - tom_speed) + karen_delay) = 27.5 :=
by sorry

end car_race_distance_l2325_232576


namespace circle_line_distance_l2325_232544

theorem circle_line_distance (x y : ℝ) (a : ℝ) :
  (x^2 + y^2 - 2*x - 4*y = 0) →
  ((1 - y + a) / Real.sqrt 2 = Real.sqrt 2 / 2 ∨
   (-1 + y - a) / Real.sqrt 2 = Real.sqrt 2 / 2) →
  (a = 0 ∨ a = 2) :=
sorry

end circle_line_distance_l2325_232544


namespace sequence_length_l2325_232579

/-- Given a sequence of real numbers satisfying specific conditions, prove that the length of the sequence is 455. -/
theorem sequence_length : ∃ (n : ℕ) (b : ℕ → ℝ), 
  n > 0 ∧ 
  b 0 = 28 ∧ 
  b 1 = 81 ∧ 
  b n = 0 ∧ 
  (∀ j ∈ Finset.range (n - 1), b (j + 2) = b j - 5 / b (j + 1)) ∧
  (∀ m : ℕ, m < n → 
    m > 0 → 
    b m ≠ 0 → 
    ¬(b 0 = 28 ∧ 
      b 1 = 81 ∧ 
      b m = 0 ∧ 
      (∀ j ∈ Finset.range (m - 1), b (j + 2) = b j - 5 / b (j + 1)))) ∧
  n = 455 :=
sorry

end sequence_length_l2325_232579


namespace number_problem_l2325_232581

theorem number_problem (x : ℝ) : 0.2 * x = 0.3 * 120 + 80 → x = 580 := by
  sorry

end number_problem_l2325_232581


namespace remaining_black_cards_l2325_232509

theorem remaining_black_cards (total_cards : Nat) (black_cards : Nat) (removed_cards : Nat) :
  total_cards = 52 →
  black_cards = 26 →
  removed_cards = 4 →
  black_cards - removed_cards = 22 := by
  sorry

end remaining_black_cards_l2325_232509


namespace sin_right_angle_l2325_232535

theorem sin_right_angle (D E F : ℝ) (h1 : D = 90) (h2 : DE = 12) (h3 : EF = 35) : Real.sin D = 1 := by
  sorry

end sin_right_angle_l2325_232535


namespace odd_number_factorial_not_divisible_by_square_l2325_232551

/-- A function that checks if a natural number is odd -/
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

/-- A function that checks if a natural number is prime -/
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 0 → m < p → p % m ≠ 0

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem odd_number_factorial_not_divisible_by_square (n : ℕ) :
  is_odd n → (factorial (n - 1) % (n^2) ≠ 0 ↔ is_prime n ∨ n = 9) :=
by sorry

end odd_number_factorial_not_divisible_by_square_l2325_232551


namespace expression_simplification_l2325_232500

theorem expression_simplification :
  Real.sqrt (1 + 3) * Real.sqrt (4 + Real.sqrt (1 + 3 + 5 + 7 + 9)) = 6 := by
  sorry

end expression_simplification_l2325_232500


namespace divisibility_property_l2325_232572

theorem divisibility_property (a b n : ℕ) (h : a^n ∣ b) : a^(n+1) ∣ (a+1)^b - 1 := by
  sorry

end divisibility_property_l2325_232572


namespace minsu_marbles_left_l2325_232501

/-- Calculates the number of marbles left after distribution -/
def marblesLeft (totalMarbles : ℕ) (largeBulk smallBulk : ℕ) (largeBoxes smallBoxes : ℕ) : ℕ :=
  totalMarbles - (largeBulk * largeBoxes + smallBulk * smallBoxes)

/-- Theorem stating the number of marbles left after Minsu's distribution -/
theorem minsu_marbles_left :
  marblesLeft 240 35 6 4 3 = 82 := by
  sorry

end minsu_marbles_left_l2325_232501


namespace baker_bread_rolls_l2325_232582

theorem baker_bread_rolls (regular_rolls : ℕ) (regular_flour : ℚ) 
  (new_rolls : ℕ) (new_flour : ℚ) :
  regular_rolls = 40 →
  regular_flour = 1 / 8 →
  new_rolls = 25 →
  regular_rolls * regular_flour = new_rolls * new_flour →
  new_flour = 1 / 5 := by
  sorry

end baker_bread_rolls_l2325_232582


namespace circles_are_externally_tangent_l2325_232552

/-- Circle represented by its equation in the form (x - h)^2 + (y - k)^2 = r^2 -/
structure Circle where
  h : ℝ  -- x-coordinate of the center
  k : ℝ  -- y-coordinate of the center
  r : ℝ  -- radius
  r_pos : r > 0

/-- Two circles are externally tangent if the distance between their centers
    is equal to the sum of their radii -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  (c1.h - c2.h)^2 + (c1.k - c2.k)^2 = (c1.r + c2.r)^2

theorem circles_are_externally_tangent :
  let c1 : Circle := { h := 0, k := 0, r := 1, r_pos := by norm_num }
  let c2 : Circle := { h := 0, k := 3, r := 2, r_pos := by norm_num }
  are_externally_tangent c1 c2 := by sorry

end circles_are_externally_tangent_l2325_232552


namespace complex_roots_on_circle_l2325_232549

theorem complex_roots_on_circle : ∀ z : ℂ, 
  (z + 2)^6 = 64 * z^6 → Complex.abs (z - (2/3 : ℂ)) = 2/3 := by
  sorry

end complex_roots_on_circle_l2325_232549


namespace dining_room_tiles_l2325_232559

/-- Calculates the total number of tiles needed for a rectangular room with a border --/
def total_tiles (room_length room_width border_width : ℕ) : ℕ :=
  let border_tiles := 2 * (room_length + room_width - 4 * border_width)
  let inner_length := room_length - 2 * border_width
  let inner_width := room_width - 2 * border_width
  let inner_area := inner_length * inner_width
  let inner_tiles := (inner_area + 3) / 4  -- Ceiling division by 4
  border_tiles + inner_tiles

/-- Theorem stating that a 15ft by 18ft room with a 2ft border requires 139 tiles --/
theorem dining_room_tiles : total_tiles 18 15 2 = 139 := by
  sorry

end dining_room_tiles_l2325_232559


namespace magician_earnings_l2325_232522

/-- Calculates the money earned by a magician selling card decks --/
def money_earned (price_per_deck : ℕ) (starting_decks : ℕ) (ending_decks : ℕ) : ℕ :=
  (starting_decks - ending_decks) * price_per_deck

/-- Proves that the magician earned 4 dollars --/
theorem magician_earnings : money_earned 2 5 3 = 4 := by
  sorry

end magician_earnings_l2325_232522


namespace expected_balls_in_original_position_l2325_232540

/-- The number of balls arranged in a circle -/
def n : ℕ := 7

/-- The probability of a ball being swapped twice -/
def p_twice : ℚ := 2 / (n * n)

/-- The probability of a ball never being swapped -/
def p_never : ℚ := (n - 2)^2 / (n * n)

/-- The probability of a ball being in its original position after two transpositions -/
def p_original : ℚ := p_twice + p_never

/-- The expected number of balls in their original positions after two transpositions -/
def expected_original : ℚ := n * p_original

theorem expected_balls_in_original_position :
  expected_original = 189 / 49 := by sorry

end expected_balls_in_original_position_l2325_232540


namespace smallest_n_value_l2325_232585

theorem smallest_n_value (o y v : ℝ) (ho : o > 0) (hy : y > 0) (hv : v > 0) :
  let n := Nat.lcm (Nat.lcm 10 16) 18 / 24
  ∀ m : ℕ, m > 0 → (∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ 10 * a = 16 * b ∧ 16 * b = 18 * c ∧ 18 * c = 24 * m) →
  m ≥ n :=
by sorry

end smallest_n_value_l2325_232585


namespace triangle_abc_properties_l2325_232503

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Opposite sides of angles A, B, C are a, b, c respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given equation
  Real.cos C * Real.sin (A + π/6) - Real.sin C * Real.sin (A - π/3) = 1/2 →
  -- Perimeter condition
  a + b + c = 4 →
  -- Area condition
  1/2 * a * c * Real.sin B = Real.sqrt 3 / 3 →
  -- Conclusion: B = π/3 and b = 3/2
  B = π/3 ∧ b = 3/2 := by
sorry

end triangle_abc_properties_l2325_232503


namespace intersection_probability_theorem_l2325_232575

/-- The probability that two randomly chosen diagonals intersect in a convex polygon with 2n + 1 vertices. -/
def intersection_probability (n : ℕ) : ℚ :=
  if n > 0 then
    (n * (2 * n - 1)) / (3 * (2 * n^2 - n - 2))
  else
    0

/-- Theorem: In a convex polygon with 2n + 1 vertices (n > 0), the probability that two randomly
    chosen diagonals intersect is n(2n - 1) / (3(2n^2 - n - 2)). -/
theorem intersection_probability_theorem (n : ℕ) (h : n > 0) :
  intersection_probability n = (n * (2 * n - 1)) / (3 * (2 * n^2 - n - 2)) :=
by sorry

end intersection_probability_theorem_l2325_232575


namespace repeating_decimal_subtraction_l2325_232511

def repeating_decimal_246 : ℚ := 246 / 999
def repeating_decimal_135 : ℚ := 135 / 999
def repeating_decimal_579 : ℚ := 579 / 999

theorem repeating_decimal_subtraction :
  repeating_decimal_246 - repeating_decimal_135 - repeating_decimal_579 = -24 / 51 := by
  sorry

end repeating_decimal_subtraction_l2325_232511


namespace Mp_not_perfect_square_l2325_232536

/-- A prime number p congruent to 3 modulo 4 -/
def p : ℕ := sorry

/-- Assumption that p is prime -/
axiom p_prime : Nat.Prime p

/-- Assumption that p is congruent to 3 modulo 4 -/
axiom p_mod_4 : p % 4 = 3

/-- Definition of a balanced sequence -/
def BalancedSequence (seq : List ℤ) : Prop :=
  (∀ x ∈ seq, ∃ y ∈ seq, x = -y) ∧
  (∀ x ∈ seq, |x| ≤ (p - 1) / 2) ∧
  (seq.length ≤ p - 1)

/-- The number of balanced sequences for prime p -/
def Mp : ℕ := sorry

/-- Theorem: Mp is not a perfect square -/
theorem Mp_not_perfect_square : ¬ ∃ (n : ℕ), Mp = n ^ 2 := by sorry

end Mp_not_perfect_square_l2325_232536


namespace students_in_grade_6_l2325_232504

theorem students_in_grade_6 (total : ℕ) (grade_4 : ℕ) (grade_5 : ℕ) (grade_6 : ℕ) :
  total = 100 → grade_4 = 30 → grade_5 = 35 → total = grade_4 + grade_5 + grade_6 → grade_6 = 35 := by
  sorry

end students_in_grade_6_l2325_232504


namespace remainder_of_product_mod_12_l2325_232568

theorem remainder_of_product_mod_12 : (1425 * 1427 * 1429) % 12 = 3 := by
  sorry

end remainder_of_product_mod_12_l2325_232568


namespace pipe_fill_time_l2325_232571

/-- Time to fill tank with leak (in hours) -/
def time_with_leak : ℝ := 15

/-- Time for leak to empty full tank (in hours) -/
def time_leak_empty : ℝ := 30

/-- Time to fill tank without leak (in hours) -/
def time_without_leak : ℝ := 10

theorem pipe_fill_time :
  (1 / time_without_leak) - (1 / time_leak_empty) = (1 / time_with_leak) :=
sorry

end pipe_fill_time_l2325_232571


namespace power_expression_l2325_232530

theorem power_expression (x y : ℝ) (a b : ℝ) (h1 : 10^x = a) (h2 : 10^y = b) :
  10^(3*x + 2*y) = a^3 * b^2 := by
  sorry

end power_expression_l2325_232530


namespace probability_two_black_balls_l2325_232529

/-- Probability of drawing two black balls without replacement -/
theorem probability_two_black_balls 
  (white : ℕ) 
  (black : ℕ) 
  (h1 : white = 7) 
  (h2 : black = 8) : 
  (black * (black - 1)) / ((white + black) * (white + black - 1)) = 4 / 15 :=
by sorry

end probability_two_black_balls_l2325_232529


namespace greatest_prime_factor_eleven_l2325_232570

def f (m : ℕ) : ℕ := Finset.prod (Finset.range (m/2)) (fun i => 2*(i+1))

theorem greatest_prime_factor_eleven (m : ℕ) (h1 : m > 0) (h2 : Even m) :
  (∀ p : ℕ, Prime p → p ∣ f m → p ≤ 11) ∧
  (11 ∣ f m) →
  m = 22 := by sorry

end greatest_prime_factor_eleven_l2325_232570


namespace absolute_value_sum_difference_l2325_232518

theorem absolute_value_sum_difference (x y : ℝ) : 
  (|x| = 3 ∧ |y| = 7) →
  ((x > 0 ∧ y < 0 → x + y = -4) ∧
   (x < y → (x - y = -10 ∨ x - y = -4))) := by
  sorry

end absolute_value_sum_difference_l2325_232518


namespace binary_1101_equals_base5_23_l2325_232594

-- Define a function to convert binary to decimal
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

-- Define a function to convert decimal to base-5
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

-- Theorem statement
theorem binary_1101_equals_base5_23 :
  decimal_to_base5 (binary_to_decimal [true, false, true, true]) = [2, 3] := by
  sorry

#eval binary_to_decimal [true, false, true, true]
#eval decimal_to_base5 13

end binary_1101_equals_base5_23_l2325_232594


namespace land_sections_area_l2325_232513

theorem land_sections_area (x y z : ℝ) 
  (h1 : x = (2/5) * (x + y + z))
  (h2 : y / z = (3/2) / (4/3))
  (h3 : z = x - 16) :
  x + y + z = 136 := by
  sorry

end land_sections_area_l2325_232513


namespace game_winner_l2325_232521

/-- Represents the state of the game with three balls -/
structure GameState where
  n : ℕ -- number of empty holes between one outer ball and the middle ball
  k : ℕ -- number of empty holes between the other outer ball and the middle ball

/-- Determines if a player can make a move in the given game state -/
def canMove (state : GameState) : Prop :=
  state.n > 0 ∨ state.k > 0

/-- Determines if the first player wins in the given game state -/
def firstPlayerWins (state : GameState) : Prop :=
  (state.n + state.k) % 2 = 1

theorem game_winner (state : GameState) :
  canMove state → (firstPlayerWins state ↔ ¬firstPlayerWins { n := state.k, k := state.n - 1 }) ∧
                  (¬firstPlayerWins state ↔ ¬firstPlayerWins { n := state.n - 1, k := state.k }) :=
sorry

end game_winner_l2325_232521


namespace cubic_function_derivative_l2325_232563

theorem cubic_function_derivative (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^3 + 3 * x^2 + 2
  let f' : ℝ → ℝ := λ x ↦ 3 * a * x^2 + 6 * x
  f' (-1) = 4 → a = 10/3 := by
  sorry

end cubic_function_derivative_l2325_232563


namespace circle_radius_l2325_232584

/-- Given a circle with center (0,k) where k > 5, which is tangent to the lines y=2x, y=-2x, and y=5,
    the radius of the circle is (k-5)/√5. -/
theorem circle_radius (k : ℝ) (h : k > 5) : ∃ r : ℝ,
  r > 0 ∧
  r = (k - 5) / Real.sqrt 5 ∧
  (∀ x y : ℝ, (x = 0 ∧ y = k) → (x^2 + (y - k)^2 = r^2)) ∧
  (∃ x y : ℝ, y = 2*x ∧ x^2 + (y - k)^2 = r^2) ∧
  (∃ x y : ℝ, y = -2*x ∧ x^2 + (y - k)^2 = r^2) ∧
  (∃ x : ℝ, x^2 + (5 - k)^2 = r^2) :=
by sorry

end circle_radius_l2325_232584


namespace profit_maximizing_price_l2325_232516

/-- Represents the profit function for a product -/
def profit_function (x : ℝ) : ℝ :=
  (x - 8) * (100 - 10 * (x - 10))

/-- Theorem stating that the profit-maximizing price is 14 yuan -/
theorem profit_maximizing_price :
  ∃ (x : ℝ), x = 14 ∧ ∀ (y : ℝ), profit_function y ≤ profit_function x :=
sorry

end profit_maximizing_price_l2325_232516


namespace exists_x_y_inequality_l2325_232587

theorem exists_x_y_inequality (f : ℝ → ℝ) : ∃ x y : ℝ, f (x - f y) > y * f x + x := by
  sorry

end exists_x_y_inequality_l2325_232587


namespace other_diagonal_length_l2325_232538

/-- Represents a rhombus with given diagonals and area -/
structure Rhombus where
  diagonal1 : ℝ
  diagonal2 : ℝ
  area : ℝ

/-- The area of a rhombus is half the product of its diagonals -/
axiom rhombus_area (r : Rhombus) : r.area = (r.diagonal1 * r.diagonal2) / 2

theorem other_diagonal_length :
  ∀ r : Rhombus, r.diagonal1 = 12 ∧ r.area = 60 → r.diagonal2 = 10 := by
  sorry

end other_diagonal_length_l2325_232538


namespace volume_cylindrical_wedge_with_cap_l2325_232597

/-- The volume of a solid composed of a cylindrical wedge and a conical cap -/
theorem volume_cylindrical_wedge_with_cap (d : ℝ) (h : d = 16) :
  let r := d / 2
  let wedge_volume := (π * r^2 * d) / 2
  let cone_volume := (1/3) * π * r^2 * d
  wedge_volume + cone_volume = (2560/3) * π := by
  sorry

end volume_cylindrical_wedge_with_cap_l2325_232597


namespace quadratic_properties_l2325_232560

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_properties
  (a b c : ℝ)
  (ha : a ≠ 0)
  (h1 : quadratic a b c 0 = 2)
  (h2 : quadratic a b c 1 = 2)
  (h3 : quadratic a b c (3/2) < 0)
  (h4 : ∃ x₁ x₂ : ℝ, x₁ < 0 ∧ x₂ > 0 ∧ quadratic a b c x₁ = 0 ∧ quadratic a b c x₂ = 0)
  (h5 : ∃ x : ℝ, -1/2 < x ∧ x < 0 ∧ quadratic a b c x = 0) :
  (∀ x ≤ 0, ∀ y ≤ x, quadratic a b c y ≤ quadratic a b c x) ∧
  (3 * quadratic a b c (-1) - quadratic a b c 2 < -20/3) :=
by sorry

end quadratic_properties_l2325_232560


namespace quadratic_equation_properties_l2325_232573

/-- Represents a quadratic equation of the form x^2 - (m-3)x - m = 0 -/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  x^2 - (m-3)*x - m = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  (m-3)^2 - 4*(-m)

/-- Represents the condition on the roots of the quadratic equation -/
def root_condition (x₁ x₂ : ℝ) : Prop :=
  x₁^2 + x₂^2 - x₁*x₂ = 13

theorem quadratic_equation_properties (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂) ∧
  (∀ x₁ x₂ : ℝ, quadratic_equation m x₁ ∧ quadratic_equation m x₂ ∧ root_condition x₁ x₂ →
    m = 4 ∨ m = -1) :=
by sorry

end quadratic_equation_properties_l2325_232573


namespace village_households_l2325_232566

/-- The number of households in a village where:
    - Each household uses 150 litres of water per month
    - 6000 litres of water lasts for 4 months for all households
-/
def number_of_households : ℕ := 10

/-- Water usage per household per month in litres -/
def water_per_household_per_month : ℕ := 150

/-- Total water available in litres -/
def total_water : ℕ := 6000

/-- Number of months the water lasts -/
def months : ℕ := 4

theorem village_households : 
  number_of_households * water_per_household_per_month * months = total_water :=
sorry

end village_households_l2325_232566


namespace correct_order_count_l2325_232580

/-- Represents the number of letters in the original stack -/
def n : ℕ := 10

/-- Represents the position of the letter known to be typed -/
def k : ℕ := 9

/-- Calculates the number of possible typing orders for the remaining letters -/
def possibleOrders : ℕ := 
  (List.range (k - 1)).foldl (fun acc i => acc + (Nat.choose (k - 1) i) * (i + 2)) 0

/-- Theorem stating the correct number of possible typing orders -/
theorem correct_order_count : possibleOrders = 1536 := by
  sorry

end correct_order_count_l2325_232580


namespace stamps_leftover_l2325_232523

theorem stamps_leftover (olivia parker quinn : ℕ) (album_capacity : ℕ) 
  (h1 : olivia = 52) 
  (h2 : parker = 66) 
  (h3 : quinn = 23) 
  (h4 : album_capacity = 15) : 
  (olivia + parker + quinn) % album_capacity = 6 := by
  sorry

end stamps_leftover_l2325_232523
