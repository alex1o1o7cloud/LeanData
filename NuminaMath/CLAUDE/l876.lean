import Mathlib

namespace rectangle_perimeter_l876_87625

theorem rectangle_perimeter (x y : ℝ) 
  (rachel_sum : 2 * x + y = 44)
  (heather_sum : x + 2 * y = 40) : 
  2 * (x + y) = 56 := by
  sorry

end rectangle_perimeter_l876_87625


namespace company_total_individuals_l876_87670

/-- Represents the hierarchical structure of a company -/
structure CompanyHierarchy where
  workers_per_team_lead : Nat
  team_leads_per_manager : Nat
  managers_per_supervisor : Nat

/-- Calculates the total number of individuals in the company given the hierarchy and number of supervisors -/
def total_individuals (h : CompanyHierarchy) (supervisors : Nat) : Nat :=
  let managers := supervisors * h.managers_per_supervisor
  let team_leads := managers * h.team_leads_per_manager
  let workers := team_leads * h.workers_per_team_lead
  workers + team_leads + managers + supervisors

/-- Theorem stating that given the specific hierarchy and 10 supervisors, the total number of individuals is 3260 -/
theorem company_total_individuals :
  let h : CompanyHierarchy := {
    workers_per_team_lead := 15,
    team_leads_per_manager := 4,
    managers_per_supervisor := 5
  }
  total_individuals h 10 = 3260 := by
  sorry

end company_total_individuals_l876_87670


namespace max_sum_on_circle_l876_87621

theorem max_sum_on_circle : ∀ x y : ℤ, x^2 + y^2 = 20 → x + y ≤ 6 := by sorry

end max_sum_on_circle_l876_87621


namespace lose_sector_area_l876_87654

theorem lose_sector_area (radius : ℝ) (win_probability : ℝ) 
  (h1 : radius = 12)
  (h2 : win_probability = 1/3) : 
  (1 - win_probability) * π * radius^2 = 96 * π := by
  sorry

end lose_sector_area_l876_87654


namespace simplify_expression_l876_87641

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) :
  x⁻¹ - x + 2 = (1 - (x - 1)^2) / x := by
  sorry

end simplify_expression_l876_87641


namespace line_length_problem_l876_87689

theorem line_length_problem (L : ℝ) (h : 0.75 * L - 0.4 * L = 28) : L = 80 := by
  sorry

end line_length_problem_l876_87689


namespace mnp_value_l876_87691

theorem mnp_value (a b x y : ℝ) (m n p : ℤ) 
  (h : a^8*x*y - a^7*y - a^6*x = a^5*(b^5 - 1)) 
  (h_equiv : (a^m*x - a^n)*(a^p*y - a^3) = a^5*b^5) : 
  m * n * p = 32 := by
  sorry

end mnp_value_l876_87691


namespace petes_flag_shapes_l876_87679

def us_stars : ℕ := 50
def us_stripes : ℕ := 13

def circles : ℕ := us_stars / 2 - 3
def squares : ℕ := us_stripes * 2 + 6
def triangles : ℕ := (us_stars - us_stripes) * 2
def diamonds : ℕ := (us_stars + us_stripes) / 4

theorem petes_flag_shapes :
  circles + squares + triangles + diamonds = 143 := by
  sorry

end petes_flag_shapes_l876_87679


namespace ones_digit_of_largest_power_of_2_dividing_32_factorial_l876_87615

/-- The largest power of 2 that divides n! -/
def largestPowerOf2DividingFactorial (n : ℕ) : ℕ :=
  sorry

/-- The ones digit of a natural number -/
def onesDigit (n : ℕ) : ℕ :=
  n % 10

theorem ones_digit_of_largest_power_of_2_dividing_32_factorial :
  onesDigit (2^(largestPowerOf2DividingFactorial 32)) = 8 := by
  sorry

end ones_digit_of_largest_power_of_2_dividing_32_factorial_l876_87615


namespace quadratic_roots_range_l876_87606

theorem quadratic_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + (2*m - 3)*x + m - 150 = 0 ∧
               y^2 + (2*m - 3)*y + m - 150 = 0 ∧
               x > 2 ∧ y < 2) ↔
  m > 5 :=
sorry

end quadratic_roots_range_l876_87606


namespace ed_lost_no_marbles_l876_87628

def marbles_lost (ed_initial : ℕ) (ed_now : ℕ) (doug : ℕ) : ℕ :=
  ed_initial - ed_now

theorem ed_lost_no_marbles 
  (h1 : ∃ ed_initial : ℕ, ed_initial = doug + 12)
  (h2 : ∃ ed_now : ℕ, ed_now = 17)
  (h3 : doug = 5) :
  marbles_lost (doug + 12) 17 doug = 0 := by
  sorry

end ed_lost_no_marbles_l876_87628


namespace parabola_coefficient_b_l876_87619

/-- Given a parabola y = ax^2 + bx + c with vertex at (h, -h) and y-intercept at (0, h),
    where h ≠ 0, the coefficient b equals -4. -/
theorem parabola_coefficient_b (a b c h : ℝ) : 
  h ≠ 0 →
  (∀ x, a * x^2 + b * x + c = a * (x - h)^2 - h) →
  c = h →
  b = -4 := by sorry

end parabola_coefficient_b_l876_87619


namespace red_part_length_l876_87669

/-- The length of the red part of a pencil given specific color proportions -/
theorem red_part_length (total_length : ℝ) (green_ratio : ℝ) (gold_ratio : ℝ) (red_ratio : ℝ)
  (h_total : total_length = 15)
  (h_green : green_ratio = 7/10)
  (h_gold : gold_ratio = 3/7)
  (h_red : red_ratio = 2/3) :
  red_ratio * (total_length - green_ratio * total_length - gold_ratio * (total_length - green_ratio * total_length)) =
  2/3 * (15 - 15 * 7/10 - (15 - 15 * 7/10) * 3/7) :=
by sorry

end red_part_length_l876_87669


namespace max_crates_third_trip_l876_87678

/-- The weight of each crate in kilograms -/
def crate_weight : ℝ := 1250

/-- The maximum weight capacity of the trailer in kilograms -/
def max_weight : ℝ := 6250

/-- The number of crates on the first trip -/
def first_trip_crates : ℕ := 3

/-- The number of crates on the second trip -/
def second_trip_crates : ℕ := 4

/-- Theorem: The maximum number of crates that can be carried on the third trip is 5 -/
theorem max_crates_third_trip :
  ∃ (x : ℕ), x ≤ 5 ∧
  (∀ y : ℕ, y > x → y * crate_weight > max_weight) ∧
  x * crate_weight ≤ max_weight :=
sorry

end max_crates_third_trip_l876_87678


namespace arctan_sum_three_four_l876_87653

theorem arctan_sum_three_four : Real.arctan (3/4) + Real.arctan (4/3) = π/2 := by
  sorry

end arctan_sum_three_four_l876_87653


namespace rain_and_humidity_probability_l876_87649

/-- The probability of rain in a coastal city in Zhejiang -/
def prob_rain : ℝ := 0.4

/-- The probability that the humidity exceeds 70% on rainy days -/
def prob_humidity_given_rain : ℝ := 0.6

/-- The probability that it rains and the humidity exceeds 70% -/
def prob_rain_and_humidity : ℝ := prob_rain * prob_humidity_given_rain

theorem rain_and_humidity_probability :
  prob_rain_and_humidity = 0.24 :=
sorry

end rain_and_humidity_probability_l876_87649


namespace old_supervisor_salary_l876_87643

def num_workers : ℕ := 8
def initial_total : ℕ := 9
def initial_avg_salary : ℚ := 430
def new_avg_salary : ℚ := 420
def new_supervisor_salary : ℕ := 780

theorem old_supervisor_salary :
  ∃ (workers_total_salary old_supervisor_salary : ℚ),
    (workers_total_salary + old_supervisor_salary) / initial_total = initial_avg_salary ∧
    (workers_total_salary + new_supervisor_salary) / initial_total = new_avg_salary ∧
    old_supervisor_salary = 870 := by
  sorry

end old_supervisor_salary_l876_87643


namespace diophantine_equation_solutions_l876_87683

theorem diophantine_equation_solutions (x y : ℕ+) :
  x^(y : ℕ) = y^(x : ℕ) + 1 ↔ (x = 2 ∧ y = 1) ∨ (x = 3 ∧ y = 2) := by
  sorry

end diophantine_equation_solutions_l876_87683


namespace hotel_pricing_theorem_l876_87681

/-- Hotel pricing model -/
structure HotelPricing where
  flatFee : ℝ  -- Flat fee for the first night
  nightlyFee : ℝ  -- Fixed amount for each additional night

/-- Calculate total cost for a stay -/
def totalCost (pricing : HotelPricing) (nights : ℕ) : ℝ :=
  pricing.flatFee + pricing.nightlyFee * (nights - 1)

/-- The hotel pricing theorem -/
theorem hotel_pricing_theorem (pricing : HotelPricing) :
  totalCost pricing 4 = 200 ∧ totalCost pricing 7 = 350 → pricing.flatFee = 50 := by
  sorry

#check hotel_pricing_theorem

end hotel_pricing_theorem_l876_87681


namespace solid_shapes_count_l876_87627

-- Define the set of geometric shapes
inductive GeometricShape
  | Square
  | Cuboid
  | Circle
  | Sphere
  | Cone

-- Define a function to determine if a shape is solid
def isSolid (shape : GeometricShape) : Bool :=
  match shape with
  | GeometricShape.Square => false
  | GeometricShape.Cuboid => true
  | GeometricShape.Circle => false
  | GeometricShape.Sphere => true
  | GeometricShape.Cone => true

-- Define the list of given shapes
def givenShapes : List GeometricShape :=
  [GeometricShape.Square, GeometricShape.Cuboid, GeometricShape.Circle, GeometricShape.Sphere, GeometricShape.Cone]

-- Theorem statement
theorem solid_shapes_count :
  (givenShapes.filter isSolid).length = 3 := by
  sorry

end solid_shapes_count_l876_87627


namespace power_fraction_equality_l876_87660

theorem power_fraction_equality : (2^2017 + 2^2013) / (2^2017 - 2^2013) = 17/15 := by
  sorry

end power_fraction_equality_l876_87660


namespace counterexample_exists_l876_87623

theorem counterexample_exists : ∃ n : ℕ, 
  (∃ p : ℕ, Prime p ∧ ∃ k : ℕ, n = p^k) ∧ 
  Prime (n - 2) ∧ 
  n = 25 := by
  sorry

end counterexample_exists_l876_87623


namespace eighth_term_ratio_l876_87614

def arithmetic_sequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def arithmetic_sum (a₁ d : ℚ) (n : ℕ) : ℚ := (n : ℚ) * (2 * a₁ + (n - 1 : ℚ) * d) / 2

theorem eighth_term_ratio
  (a₁ d b₁ e : ℚ)
  (h : ∀ n : ℕ, arithmetic_sum a₁ d n / arithmetic_sum b₁ e n = (5 * n + 6 : ℚ) / (3 * n + 30 : ℚ)) :
  (arithmetic_sequence a₁ d 8) / (arithmetic_sequence b₁ e 8) = 4 / 3 :=
sorry

end eighth_term_ratio_l876_87614


namespace square_side_increase_l876_87661

theorem square_side_increase (p : ℝ) : 
  (1 + p / 100)^2 = 1.96 → p = 40 := by
  sorry

end square_side_increase_l876_87661


namespace triangle_side_values_l876_87645

theorem triangle_side_values (y : ℕ+) : 
  (∃ (a b c : ℝ), a = 8 ∧ b = 11 ∧ c = y.val ^ 2 ∧ 
    a + b > c ∧ b + c > a ∧ c + a > b) ↔ 
  (y = 2 ∨ y = 3 ∨ y = 4) :=
by sorry

end triangle_side_values_l876_87645


namespace unique_circle_digits_l876_87616

theorem unique_circle_digits : ∃! (a b c d e : ℕ),
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) ∧
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10) ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
  (a + b = (c + d + e) / 7) ∧
  (a + c = (b + d + e) / 5) :=
by sorry

end unique_circle_digits_l876_87616


namespace sqrt_expression_value_l876_87600

theorem sqrt_expression_value (x y : ℝ) 
  (h : Real.sqrt (x + 5) + (2 * x - y)^2 = 0) : 
  Real.sqrt (x^2 - 2*x*y + y^2) = 5 := by
  sorry

end sqrt_expression_value_l876_87600


namespace square_area_l876_87668

theorem square_area (side_length : ℝ) (h : side_length = 19) :
  side_length * side_length = 361 := by
  sorry

end square_area_l876_87668


namespace solution_in_first_and_second_quadrants_l876_87659

-- Define the inequalities
def inequality1 (x y : ℝ) : Prop := y > 3 * x
def inequality2 (x y : ℝ) : Prop := y > 6 - 2 * x

-- Define the first quadrant
def first_quadrant (x y : ℝ) : Prop := x > 0 ∧ y > 0

-- Define the second quadrant
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem statement
theorem solution_in_first_and_second_quadrants :
  ∀ x y : ℝ, inequality1 x y ∧ inequality2 x y →
  first_quadrant x y ∨ second_quadrant x y :=
sorry

end solution_in_first_and_second_quadrants_l876_87659


namespace maple_trees_planted_l876_87667

theorem maple_trees_planted (initial : ℕ) (final : ℕ) (h1 : initial = 53) (h2 : final = 64) :
  final - initial = 11 := by
  sorry

end maple_trees_planted_l876_87667


namespace min_value_sum_l876_87638

theorem min_value_sum (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : a^2 + a*b + a*c + b*c = 4) :
  ∀ x y z, x > 0 ∧ y > 0 ∧ z > 0 ∧ x^2 + x*y + x*z + y*z = 4 →
  2*a + b + c ≤ 2*x + y + z ∧ 2*a + b + c = 4 := by
  sorry

end min_value_sum_l876_87638


namespace starting_lineup_count_l876_87634

/-- The number of ways to choose a starting lineup from a basketball team -/
def choose_lineup (team_size : ℕ) (lineup_size : ℕ) (point_guard : ℕ) : ℕ :=
  team_size * (Nat.choose (team_size - 1) (lineup_size - 1))

/-- Theorem stating the number of ways to choose the starting lineup -/
theorem starting_lineup_count :
  choose_lineup 12 5 1 = 3960 := by
  sorry

end starting_lineup_count_l876_87634


namespace roots_properties_l876_87605

def i : ℂ := Complex.I

def quadratic_equation (z : ℂ) : Prop :=
  z^2 + 2*z = -4 + 8*i

def roots (z₁ z₂ : ℂ) : Prop :=
  quadratic_equation z₁ ∧ quadratic_equation z₂ ∧ z₁ ≠ z₂

theorem roots_properties :
  ∃ z₁ z₂ : ℂ, roots z₁ z₂ ∧
  (z₁.re * z₂.re = -7) ∧
  (z₁.im + z₂.im = 0) := by sorry

end roots_properties_l876_87605


namespace point_movement_on_number_line_l876_87673

/-- Given two points A and B on a number line, where A represents -3 and B is obtained by moving 7 units to the right from A, prove that B represents 4. -/
theorem point_movement_on_number_line (A B : ℝ) : A = -3 ∧ B = A + 7 → B = 4 := by
  sorry

end point_movement_on_number_line_l876_87673


namespace square_difference_65_55_l876_87651

theorem square_difference_65_55 : 65^2 - 55^2 = 1200 := by
  sorry

end square_difference_65_55_l876_87651


namespace point_on_line_ratio_l876_87665

/-- Given six points O, A, B, C, D, E on a straight line in that order, with P between C and D,
    prove that OP = (ce - ad) / (a - c + e - d) when AP:PE = CP:PD -/
theorem point_on_line_ratio (a b c d e x : ℝ) 
  (h_order : 0 < a ∧ a < b ∧ b < c ∧ c < x ∧ x < d ∧ d < e) 
  (h_ratio : (a - x) / (x - e) = (c - x) / (x - d)) : 
  x = (c * e - a * d) / (a - c + e - d) := by
  sorry

end point_on_line_ratio_l876_87665


namespace one_man_work_time_l876_87609

-- Define the work as a unit
def total_work : ℝ := 1

-- Define the time taken by the group
def group_time : ℝ := 6

-- Define the number of men and women in the group
def num_men : ℝ := 10
def num_women : ℝ := 15

-- Define the time taken by one woman
def woman_time : ℝ := 225

-- Define the time taken by one man (to be proved)
def man_time : ℝ := 100

-- Theorem statement
theorem one_man_work_time :
  (num_men / man_time + num_women / woman_time) * group_time = total_work →
  1 / man_time = 1 / 100 :=
by
  sorry

end one_man_work_time_l876_87609


namespace simplify_expression_l876_87613

theorem simplify_expression (b c : ℝ) :
  (1 : ℝ) * (-2 * b) * (3 * b^2) * (-4 * c^3) * (5 * c^4) = -120 * b^3 * c^7 := by
  sorry

end simplify_expression_l876_87613


namespace nine_twelve_fifteen_pythagorean_triple_l876_87699

/-- A Pythagorean triple is a set of three positive integers (a, b, c) such that a² + b² = c² --/
def is_pythagorean_triple (a b c : ℕ) : Prop := a * a + b * b = c * c

/-- Prove that (9, 12, 15) is a Pythagorean triple --/
theorem nine_twelve_fifteen_pythagorean_triple : is_pythagorean_triple 9 12 15 := by
  sorry

end nine_twelve_fifteen_pythagorean_triple_l876_87699


namespace probability_second_high_given_first_inferior_is_eight_ninths_l876_87639

/-- Represents the total number of pencils -/
def total_pencils : ℕ := 10

/-- Represents the number of high-quality pencils -/
def high_quality : ℕ := 8

/-- Represents the number of inferior quality pencils -/
def inferior_quality : ℕ := 2

/-- Represents the probability of drawing a high-quality pencil on the second draw,
    given that the first draw was an inferior quality pencil -/
def probability_second_high_given_first_inferior : ℚ :=
  high_quality / (total_pencils - 1)

theorem probability_second_high_given_first_inferior_is_eight_ninths :
  probability_second_high_given_first_inferior = 8 / 9 := by
  sorry

end probability_second_high_given_first_inferior_is_eight_ninths_l876_87639


namespace max_value_at_13_l876_87690

-- Define the function f(x) = x - 5
def f (x : ℝ) : ℝ := x - 5

-- Theorem statement
theorem max_value_at_13 :
  ∃ (x : ℝ), x ≤ 13 ∧ ∀ (y : ℝ), y ≤ 13 → f y ≤ f x ∧ f x = 8 :=
by
  sorry

end max_value_at_13_l876_87690


namespace mathematics_permutations_l876_87657

def word : String := "MATHEMATICS"

theorem mathematics_permutations :
  let n : ℕ := word.length
  let m_count : ℕ := word.count 'M'
  let a_count : ℕ := word.count 'A'
  let t_count : ℕ := word.count 'T'
  (n = 11 ∧ m_count = 2 ∧ a_count = 2 ∧ t_count = 2) →
  (Nat.factorial n) / (Nat.factorial m_count * Nat.factorial a_count * Nat.factorial t_count) = 4989600 := by
sorry

end mathematics_permutations_l876_87657


namespace total_tissues_used_l876_87626

/-- The number of tissues Carol had initially -/
def initial_tissues : ℕ := 97

/-- The number of tissues Carol had after use -/
def remaining_tissues : ℕ := 58

/-- The total number of tissues used by Carol and her friends -/
def tissues_used : ℕ := initial_tissues - remaining_tissues

theorem total_tissues_used :
  tissues_used = 39 :=
by sorry

end total_tissues_used_l876_87626


namespace resort_worker_period_l876_87629

theorem resort_worker_period (average_tips : ℝ) (total_period : ℕ) : 
  (6 * average_tips = (1 / 2) * (6 * average_tips + (total_period - 1) * average_tips)) →
  total_period = 7 := by
  sorry

end resort_worker_period_l876_87629


namespace figure_50_squares_l876_87607

/-- The number of nonoverlapping unit squares in figure n -/
def g (n : ℕ) : ℕ := 2 * n^2 + 4 * n + 2

/-- The sequence of nonoverlapping unit squares follows the pattern -/
axiom pattern_holds : g 0 = 2 ∧ g 1 = 8 ∧ g 2 = 18 ∧ g 3 = 32

theorem figure_50_squares : g 50 = 5202 := by sorry

end figure_50_squares_l876_87607


namespace f_properties_l876_87672

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (a + 1) / 2 * x^2 + 1

theorem f_properties (a : ℝ) :
  -- Part 1
  (a = -1/2 →
    ∃ (max min : ℝ),
      (∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x ≤ max) ∧
      (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x = max) ∧
      (∀ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x ≥ min) ∧
      (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x = min) ∧
      max = 1/2 + (Real.exp 1)^2/4 ∧
      min = 5/4) ∧
  -- Part 2
  (∃ (mono : ℝ → Prop), mono a ↔
    (∀ x y, 0 < x → 0 < y → x < y → (f a x < f a y ∨ f a x > f a y ∨ f a x = f a y))) ∧
  -- Part 3
  (-1 < a → a < 0 →
    (∀ x, x > 0 → f a x > 1 + a/2 * Real.log (-a)) ∧
    (1/Real.exp 1 - 1 < a ∧ a < 0)) :=
by sorry

end f_properties_l876_87672


namespace conor_weekly_vegetables_l876_87652

/-- Represents Conor's vegetable chopping capacity and work schedule --/
structure VegetableChopper where
  eggplants_per_day : ℕ
  carrots_per_day : ℕ
  potatoes_per_day : ℕ
  work_days_per_week : ℕ

/-- Calculates the total number of vegetables chopped in a week --/
def total_vegetables_per_week (c : VegetableChopper) : ℕ :=
  (c.eggplants_per_day + c.carrots_per_day + c.potatoes_per_day) * c.work_days_per_week

/-- Theorem stating that Conor can chop 116 vegetables in a week --/
theorem conor_weekly_vegetables :
  ∃ c : VegetableChopper,
    c.eggplants_per_day = 12 ∧
    c.carrots_per_day = 9 ∧
    c.potatoes_per_day = 8 ∧
    c.work_days_per_week = 4 ∧
    total_vegetables_per_week c = 116 :=
by
  sorry

end conor_weekly_vegetables_l876_87652


namespace fish_value_in_honey_l876_87696

/-- Represents the value of one fish in terms of jars of honey -/
def fish_value (fish_to_bread : ℚ) (bread_to_honey : ℚ) : ℚ :=
  (3 / 4) * bread_to_honey

/-- Theorem stating the value of one fish in jars of honey -/
theorem fish_value_in_honey 
  (h1 : fish_to_bread = 3 / 4)  -- 4 fish = 3 loaves of bread
  (h2 : bread_to_honey = 3)     -- 1 loaf of bread = 3 jars of honey
  : fish_value fish_to_bread bread_to_honey = 9 / 4 := by
  sorry

#eval fish_value (3 / 4) 3  -- Should evaluate to 2.25

end fish_value_in_honey_l876_87696


namespace fiftieth_term_is_296_l876_87630

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 50th term of the specific arithmetic sequence -/
def fiftiethTerm : ℝ :=
  arithmeticSequenceTerm 2 6 50

theorem fiftieth_term_is_296 : fiftiethTerm = 296 := by
  sorry

end fiftieth_term_is_296_l876_87630


namespace triangle_theorem_l876_87642

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem about the triangle ABC -/
theorem triangle_theorem (t : Triangle) :
  (Real.sin t.B)^2 = (Real.sin t.A)^2 + (Real.sin t.C)^2 - Real.sin t.A * Real.sin t.C →
  t.B = π / 3 ∧
  (t.b = Real.sqrt 3 ∧ t.a * t.c * Real.sin t.B / 2 = Real.sqrt 3 / 2 →
    t.a + t.c = 3 ∧
    -t.a * t.c * Real.cos t.B = -1) :=
by sorry

end triangle_theorem_l876_87642


namespace river_width_l876_87684

/-- The width of a river given its depth, flow rate, and volume of water per minute. -/
theorem river_width (depth : ℝ) (flow_rate_kmph : ℝ) (volume_per_minute : ℝ) :
  depth = 3 →
  flow_rate_kmph = 2 →
  volume_per_minute = 3200 →
  ∃ (width : ℝ), abs (width - 32) < 0.1 := by
  sorry

end river_width_l876_87684


namespace nested_fraction_evaluation_l876_87693

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end nested_fraction_evaluation_l876_87693


namespace function_inequality_l876_87655

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x : ℝ, (x - 2) * deriv f x ≥ 0) : 
  f 1 + f 3 ≥ 2 * f 2 := by
  sorry

end function_inequality_l876_87655


namespace continuity_at_one_l876_87624

noncomputable def f (x : ℝ) : ℝ := (x^2 - 1) / (x^3 - 1)

theorem continuity_at_one :
  ∃ (L : ℝ), ContinuousAt (fun x => if x = 1 then L else f x) 1 ↔ L = 2/3 :=
sorry

end continuity_at_one_l876_87624


namespace uncle_james_height_difference_l876_87601

theorem uncle_james_height_difference :
  ∀ (james_initial_height uncle_height james_growth : ℝ),
    james_initial_height = (2/3) * uncle_height →
    uncle_height = 72 →
    james_growth = 10 →
    uncle_height - (james_initial_height + james_growth) = 14 :=
by
  sorry

end uncle_james_height_difference_l876_87601


namespace truck_driver_speed_l876_87697

/-- A truck driver's problem -/
theorem truck_driver_speed 
  (gas_cost : ℝ)
  (fuel_efficiency : ℝ)
  (pay_rate : ℝ)
  (total_pay : ℝ)
  (total_hours : ℝ)
  (h1 : gas_cost = 2)
  (h2 : fuel_efficiency = 10)
  (h3 : pay_rate = 0.5)
  (h4 : total_pay = 90)
  (h5 : total_hours = 10)
  : (total_pay / pay_rate) / total_hours = 18 := by
  sorry


end truck_driver_speed_l876_87697


namespace first_floor_bedrooms_l876_87631

theorem first_floor_bedrooms 
  (total : ℕ) 
  (second_floor : ℕ) 
  (third_floor : ℕ) 
  (fourth_floor : ℕ) 
  (h1 : total = 22) 
  (h2 : second_floor = 6) 
  (h3 : third_floor = 4) 
  (h4 : fourth_floor = 3) : 
  total - (second_floor + third_floor + fourth_floor) = 9 := by
  sorry

end first_floor_bedrooms_l876_87631


namespace video_game_lives_l876_87635

theorem video_game_lives (initial_lives lost_lives gained_lives : ℕ) 
  (h1 : initial_lives = 10)
  (h2 : lost_lives = 6)
  (h3 : gained_lives = 37) :
  initial_lives - lost_lives + gained_lives = 41 :=
by sorry

end video_game_lives_l876_87635


namespace unique_solution_quadratic_system_l876_87698

theorem unique_solution_quadratic_system (y : ℚ) 
  (h1 : 9 * y^2 + 8 * y - 1 = 0)
  (h2 : 27 * y^2 + 44 * y - 7 = 0) : 
  y = 1/9 := by sorry

end unique_solution_quadratic_system_l876_87698


namespace relationship_abc_l876_87676

theorem relationship_abc :
  ∀ (a b c : ℝ), a = 2 → b = 3 → c = 4 → c > b ∧ b > a := by
  sorry

end relationship_abc_l876_87676


namespace cube_edge_length_is_15_l876_87617

/-- The edge length of a cube that displaces a specific volume of water -/
def cube_edge_length (base_length base_width water_rise : ℝ) : ℝ :=
  (base_length * base_width * water_rise) ^ (1/3)

/-- Theorem stating that a cube with the given specifications has an edge length of 15 cm -/
theorem cube_edge_length_is_15 :
  cube_edge_length 20 14 12.053571428571429 = 15 := by
  sorry

end cube_edge_length_is_15_l876_87617


namespace cone_division_ratio_l876_87674

/-- Represents a right circular cone -/
structure Cone where
  height : ℝ
  baseRadius : ℝ

/-- Represents the division of a cone into two parts -/
structure ConeDivision where
  cone : Cone
  ratio : ℝ

/-- Calculates the surface area ratio of the smaller cone to the whole cone -/
def surfaceAreaRatio (d : ConeDivision) : ℝ := 
  d.ratio ^ 2

/-- Calculates the volume ratio of the smaller cone to the whole cone -/
def volumeRatio (d : ConeDivision) : ℝ := 
  d.ratio ^ 3

theorem cone_division_ratio (d : ConeDivision) 
  (h1 : d.cone.height = 4)
  (h2 : d.cone.baseRadius = 3)
  (h3 : surfaceAreaRatio d = volumeRatio d) :
  d.ratio = 125 / 387 := by
  sorry

#eval (125 : Nat) + 387

end cone_division_ratio_l876_87674


namespace decimal_to_base_five_l876_87650

theorem decimal_to_base_five : 
  (2 * 5^3 + 0 * 5^2 + 1 * 5^1 + 1 * 5^0 : ℕ) = 256 := by sorry

end decimal_to_base_five_l876_87650


namespace total_items_sold_is_727_l876_87633

/-- Represents the data for a single day of James' sales --/
structure DayData where
  houses : ℕ
  successRate : ℚ
  itemsPerHouse : ℕ

/-- Calculates the number of items sold in a day --/
def itemsSoldInDay (data : DayData) : ℚ :=
  data.houses * data.successRate * data.itemsPerHouse

/-- The week's data --/
def weekData : List DayData := [
  { houses := 20, successRate := 1, itemsPerHouse := 2 },
  { houses := 40, successRate := 4/5, itemsPerHouse := 3 },
  { houses := 50, successRate := 9/10, itemsPerHouse := 1 },
  { houses := 60, successRate := 3/4, itemsPerHouse := 4 },
  { houses := 80, successRate := 1/2, itemsPerHouse := 2 },
  { houses := 100, successRate := 7/10, itemsPerHouse := 1 },
  { houses := 120, successRate := 3/5, itemsPerHouse := 3 }
]

/-- Theorem: The total number of items sold during the week is 727 --/
theorem total_items_sold_is_727 : 
  (weekData.map itemsSoldInDay).sum = 727 := by
  sorry

end total_items_sold_is_727_l876_87633


namespace triangle_sine_sum_maximized_equilateral_maximizes_sine_sum_l876_87636

open Real

theorem triangle_sine_sum_maximized (α β γ : ℝ) : 
  0 < α ∧ 0 < β ∧ 0 < γ →
  α + β + γ = π →
  sin α + sin β + sin γ ≤ 3 * sin (π / 3) :=
sorry

theorem equilateral_maximizes_sine_sum (α β γ : ℝ) :
  0 < α ∧ 0 < β ∧ 0 < γ →
  α + β + γ = π →
  sin α + sin β + sin γ = 3 * sin (π / 3) ↔ α = β ∧ β = γ :=
sorry

end triangle_sine_sum_maximized_equilateral_maximizes_sine_sum_l876_87636


namespace smallest_integer_l876_87694

theorem smallest_integer (n : ℕ+) : 
  (Nat.lcm 36 n.val) / (Nat.gcd 36 n.val) = 24 → n.val ≥ 96 := by
  sorry

end smallest_integer_l876_87694


namespace solution_set_inequality_l876_87695

theorem solution_set_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_eq : 1/a + 2/b = 1) :
  {x : ℝ | (2 : ℝ)^(|x-1|-|x+2|) < 1} = {x : ℝ | x > -1/2} := by
  sorry

end solution_set_inequality_l876_87695


namespace existence_equivalence_l876_87610

theorem existence_equivalence : 
  (∃ (x : ℝ), x^2 + 1 < 0) ↔ (∃ (x : ℝ), x^2 + 1 < 0) := by
  sorry

end existence_equivalence_l876_87610


namespace arithmetic_mean_of_sqrt2_plus_minus_one_l876_87658

theorem arithmetic_mean_of_sqrt2_plus_minus_one :
  (((Real.sqrt 2) + 1) + ((Real.sqrt 2) - 1)) / 2 = Real.sqrt 2 := by
  sorry

end arithmetic_mean_of_sqrt2_plus_minus_one_l876_87658


namespace equilateral_parallelogram_diagonal_l876_87608

/-- A parallelogram composed of four equilateral triangles -/
structure EquilateralParallelogram where
  -- Define the vertices of the parallelogram
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  -- Ensure the parallelogram is made up of four equilateral triangles
  is_equilateral : 
    (dist A B = 2) ∧ 
    (dist B C = 2) ∧ 
    (dist C D = 2) ∧ 
    (dist D A = 2) ∧
    (dist A C = dist B D)
  -- Ensure each equilateral triangle has side length 1
  triangle_side_length : dist A B / 2 = 1

/-- The length of the diagonal in an equilateral parallelogram is √7 -/
theorem equilateral_parallelogram_diagonal 
  (p : EquilateralParallelogram) : dist p.A p.C = Real.sqrt 7 :=
sorry

end equilateral_parallelogram_diagonal_l876_87608


namespace max_product_constrained_l876_87618

theorem max_product_constrained (x y : ℕ+) (h : 7 * x + 5 * y = 140) :
  x * y ≤ 140 := by
  sorry

end max_product_constrained_l876_87618


namespace circle_portion_area_l876_87687

/-- The area of the portion of the circle x^2 - 16x + y^2 = 51 that lies above the x-axis 
    and to the left of the line y = 10 - x is equal to 8π. -/
theorem circle_portion_area : 
  ∃ (A : ℝ), 
    (∀ x y : ℝ, x^2 - 16*x + y^2 = 51 → y ≥ 0 → y ≤ 10 - x → 
      (x, y) ∈ {p : ℝ × ℝ | p.1^2 - 16*p.1 + p.2^2 = 51 ∧ p.2 ≥ 0 ∧ p.2 ≤ 10 - p.1}) ∧
    A = Real.pi * 8 := by
  sorry

end circle_portion_area_l876_87687


namespace real_sqrt_reciprocal_range_l876_87675

theorem real_sqrt_reciprocal_range (x : ℝ) : 
  (∃ y : ℝ, y = 1 / Real.sqrt (5 - x)) ↔ x < 5 := by sorry

end real_sqrt_reciprocal_range_l876_87675


namespace probability_different_digits_l876_87680

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_different_digits (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits.length = 3 ∧ digits.toFinset.card = 3

def count_valid_numbers : ℕ :=
  999 - 100 + 1

def count_different_digit_numbers : ℕ :=
  9 * 9 * 8

theorem probability_different_digits :
  (count_different_digit_numbers : ℚ) / count_valid_numbers = 18 / 25 := by
  sorry

end probability_different_digits_l876_87680


namespace binomial_linear_transform_l876_87648

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  (p_nonneg : 0 ≤ p)
  (p_le_one : p ≤ 1)

/-- Expected value of a binomial random variable -/
def expected_value (X : BinomialRV n p) : ℝ :=
  n * p

/-- Variance of a binomial random variable -/
def variance (X : BinomialRV n p) : ℝ :=
  n * p * (1 - p)

/-- Theorem: Expected value and variance of η = 5ξ, where ξ ~ B(5, 0.5) -/
theorem binomial_linear_transform :
  ∀ (ξ : BinomialRV 5 (1/2)) (η : ℝ),
  η = 5 * (expected_value ξ) →
  expected_value ξ = 5/2 ∧
  variance ξ = 5/4 ∧
  η = 25/2 ∧
  25 * (variance ξ) = 125/4 :=
sorry

end binomial_linear_transform_l876_87648


namespace equation_roots_l876_87646

/-- Given an equation with two real roots, prove the range of m and a specific case. -/
theorem equation_roots (m : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁^2 - 2*m*x₁ = -m^2 + 2*x₁ ∧ x₂^2 - 2*m*x₂ = -m^2 + 2*x₂ ∧ x₁ ≠ x₂) → 
  (m ≥ -1/2 ∧ 
   (∀ x₁ x₂ : ℝ, x₁^2 - 2*m*x₁ = -m^2 + 2*x₁ → x₂^2 - 2*m*x₂ = -m^2 + 2*x₂ → |x₁| = x₂ → m = -1/2)) :=
by sorry

end equation_roots_l876_87646


namespace sphere_expansion_l876_87602

theorem sphere_expansion (r₁ r₂ : ℝ) (h : r₁ > 0) :
  (4 / 3 * Real.pi * r₂^3) = 8 * (4 / 3 * Real.pi * r₁^3) →
  (4 * Real.pi * r₂^2) = 4 * (4 * Real.pi * r₁^2) := by
  sorry

end sphere_expansion_l876_87602


namespace some_number_value_l876_87656

theorem some_number_value (a n : ℕ) (h1 : a = 105) (h2 : a^3 = n * 25 * 315 * 7) : n = 63 := by
  sorry

end some_number_value_l876_87656


namespace difference_of_squares_a_l876_87663

theorem difference_of_squares_a (a : ℝ) : (a + 1) * (a - 1) = a^2 - 1 := by sorry

end difference_of_squares_a_l876_87663


namespace olivias_phone_pictures_l876_87604

theorem olivias_phone_pictures :
  ∀ (phone_pics camera_pics total_pics albums pics_per_album : ℕ),
    camera_pics = 35 →
    albums = 8 →
    pics_per_album = 5 →
    total_pics = albums * pics_per_album →
    total_pics = phone_pics + camera_pics →
    phone_pics = 5 := by
  sorry

end olivias_phone_pictures_l876_87604


namespace tank_depth_is_six_l876_87622

-- Define the tank dimensions and plastering cost
def tankLength : ℝ := 25
def tankWidth : ℝ := 12
def plasteringCostPerSqM : ℝ := 0.45
def totalPlasteringCost : ℝ := 334.8

-- Define the function to calculate the total surface area to be plastered
def surfaceArea (depth : ℝ) : ℝ :=
  tankLength * tankWidth + 2 * (tankLength * depth) + 2 * (tankWidth * depth)

-- Theorem statement
theorem tank_depth_is_six :
  ∃ (depth : ℝ), plasteringCostPerSqM * surfaceArea depth = totalPlasteringCost ∧ depth = 6 :=
sorry

end tank_depth_is_six_l876_87622


namespace right_handed_players_count_l876_87611

/-- Represents a football team with various player categories -/
structure FootballTeam where
  total_players : ℕ
  thrower_percentage : ℚ
  kicker_percentage : ℚ
  left_handed_remaining_percentage : ℚ
  left_handed_kicker_percentage : ℚ
  exclusive_thrower_percentage : ℚ

/-- Calculates the number of right-handed players and exclusive throwers -/
def calculate_right_handed_players (team : FootballTeam) : ℕ × ℕ :=
  sorry

/-- Theorem stating the correct number of right-handed players and exclusive throwers -/
theorem right_handed_players_count (team : FootballTeam) 
  (h1 : team.total_players = 180)
  (h2 : team.thrower_percentage = 3/10)
  (h3 : team.kicker_percentage = 9/40)
  (h4 : team.left_handed_remaining_percentage = 3/7)
  (h5 : team.left_handed_kicker_percentage = 1/4)
  (h6 : team.exclusive_thrower_percentage = 3/5) :
  calculate_right_handed_players team = (134, 32) := by
  sorry

end right_handed_players_count_l876_87611


namespace calculate_sales_professionals_l876_87644

/-- Calculates the number of sales professionals needed to sell a given number of cars
    over a specified period, with each professional selling a fixed number of cars per month. -/
theorem calculate_sales_professionals
  (total_cars : ℕ)
  (cars_per_salesperson_per_month : ℕ)
  (months_to_sell_all : ℕ)
  (h_total_cars : total_cars = 500)
  (h_cars_per_salesperson : cars_per_salesperson_per_month = 10)
  (h_months_to_sell : months_to_sell_all = 5)
  : (total_cars / months_to_sell_all) / cars_per_salesperson_per_month = 10 := by
  sorry

#check calculate_sales_professionals

end calculate_sales_professionals_l876_87644


namespace triangle_inequality_power_l876_87603

theorem triangle_inequality_power (a b c S : ℝ) (n : ℝ) : 
  a > 0 → b > 0 → c > 0 → -- triangle side lengths are positive
  a + b > c → b + c > a → c + a > b → -- triangle inequality
  2 * S = a + b + c → -- perimeter definition
  n ≥ 1 → -- condition on n
  (a^n / (b + c)) + (b^n / (c + a)) + (c^n / (a + b)) ≥ (2/3)^(n-2) * S^(n-1) := by
  sorry

end triangle_inequality_power_l876_87603


namespace sin_equality_problem_l876_87685

theorem sin_equality_problem (m : ℤ) (h1 : -90 ≤ m) (h2 : m ≤ 90) :
  Real.sin (m * Real.pi / 180) = Real.sin (710 * Real.pi / 180) → m = -10 := by
  sorry

end sin_equality_problem_l876_87685


namespace function_properties_l876_87664

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 - 3*a*x^2 + 2*b*x

-- State the theorem
theorem function_properties :
  ∃ (a b : ℝ),
  (∀ x, f a b x ≥ f a b 1) ∧ 
  (f a b 1 = -1) ∧
  (a = 1/3) ∧ 
  (b = -1/2) ∧
  (∀ x, x ≤ -1/3 ∨ x ≥ 1 → (deriv (f a b)) x ≥ 0) ∧
  (∀ x, -1/3 ≤ x ∧ x ≤ 1 → (deriv (f a b)) x ≤ 0) ∧
  (∀ α, -1 < α ∧ α < 5/27 → ∃ x y z, x < y ∧ y < z ∧ f a b x = α ∧ f a b y = α ∧ f a b z = α) :=
sorry

end function_properties_l876_87664


namespace ramp_installation_cost_l876_87647

/-- Calculates the total cost of installing a ramp given specific conditions --/
theorem ramp_installation_cost :
  let permit_base_cost : ℝ := 250
  let permit_tax_rate : ℝ := 0.1
  let contractor_labor_rate : ℝ := 150
  let raw_materials_rate : ℝ := 50
  let work_days : ℕ := 3
  let work_hours_per_day : ℝ := 5
  let tool_rental_rate : ℝ := 30
  let lunch_break_hours : ℝ := 0.5
  let raw_materials_markup : ℝ := 0.15
  let inspector_rate_discount : ℝ := 0.8
  let inspector_hours_per_day : ℝ := 2

  let permit_cost : ℝ := permit_base_cost * (1 + permit_tax_rate)
  let raw_materials_cost_with_markup : ℝ := raw_materials_rate * (1 + raw_materials_markup)
  let contractor_hourly_cost : ℝ := contractor_labor_rate + raw_materials_cost_with_markup
  let total_work_hours : ℝ := work_days * work_hours_per_day
  let total_lunch_hours : ℝ := work_days * lunch_break_hours
  let tool_rental_cost : ℝ := tool_rental_rate * work_days
  let contractor_cost : ℝ := contractor_hourly_cost * (total_work_hours - total_lunch_hours) + tool_rental_cost
  let inspector_rate : ℝ := contractor_labor_rate * (1 - inspector_rate_discount)
  let inspector_cost : ℝ := inspector_rate * inspector_hours_per_day * work_days

  let total_cost : ℝ := permit_cost + contractor_cost + inspector_cost

  total_cost = 3432.5 := by sorry

end ramp_installation_cost_l876_87647


namespace minimum_shoeing_time_l876_87677

/-- The minimum time required for blacksmiths to shoe horses -/
theorem minimum_shoeing_time
  (num_blacksmiths : ℕ)
  (num_horses : ℕ)
  (time_per_hoof : ℕ)
  (hooves_per_horse : ℕ)
  (h1 : num_blacksmiths = 48)
  (h2 : num_horses = 60)
  (h3 : time_per_hoof = 5)
  (h4 : hooves_per_horse = 4) :
  (num_horses * hooves_per_horse * time_per_hoof) / num_blacksmiths = 25 := by
  sorry

#eval (60 * 4 * 5) / 48  -- Should output 25

end minimum_shoeing_time_l876_87677


namespace no_real_roots_quadratic_l876_87682

theorem no_real_roots_quadratic (k : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + k - 1 ≠ 0) → k > 2 := by
  sorry

end no_real_roots_quadratic_l876_87682


namespace investment_return_l876_87637

/-- Given an investment scenario, calculate the percentage return -/
theorem investment_return (total_investment annual_income stock_price : ℝ)
  (h1 : total_investment = 6800)
  (h2 : stock_price = 136)
  (h3 : annual_income = 500) :
  (annual_income / total_investment) * 100 = (500 / 6800) * 100 := by
sorry

#eval (500 / 6800) * 100 -- To display the actual percentage

end investment_return_l876_87637


namespace height_difference_approx_10_inches_l876_87632

-- Define constants
def mark_height_cm : ℝ := 160
def mike_height_cm : ℝ := 185
def cm_to_m : ℝ := 0.01
def m_to_ft : ℝ := 3.28084
def ft_to_in : ℝ := 12

-- Define the height difference function
def height_difference_inches (h1 h2 : ℝ) : ℝ :=
  (h2 - h1) * cm_to_m * m_to_ft * ft_to_in

-- Theorem statement
theorem height_difference_approx_10_inches :
  ∃ ε > 0, abs (height_difference_inches mark_height_cm mike_height_cm - 10) < ε :=
sorry

end height_difference_approx_10_inches_l876_87632


namespace transformation_symmetry_l876_87612

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the transformation function
def transform (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

-- Define symmetry with respect to x-axis
def symmetricToXAxis (p1 p2 : Point2D) : Prop :=
  p1.x = p2.x ∧ p1.y = -p2.y

-- Theorem statement
theorem transformation_symmetry (p : Point2D) :
  symmetricToXAxis p (transform p) := by
  sorry


end transformation_symmetry_l876_87612


namespace odd_function_negative_x_l876_87686

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_function_negative_x
  (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_pos : ∀ x > 0, f x = 2 * x - 3) :
  ∀ x < 0, f x = 2 * x + 3 :=
by sorry

end odd_function_negative_x_l876_87686


namespace half_meter_cut_l876_87666

theorem half_meter_cut (initial_length : ℚ) (cut_length : ℚ) (result_length : ℚ) : 
  initial_length = 8/15 →
  cut_length = 1/30 →
  result_length = initial_length - cut_length →
  result_length = 1/2 :=
by
  sorry

#check half_meter_cut

end half_meter_cut_l876_87666


namespace solve_equation_l876_87671

theorem solve_equation (x : ℚ) : x / 4 * 5 + 10 - 12 = 48 → x = 40 := by
  sorry

end solve_equation_l876_87671


namespace expression_evaluation_l876_87640

theorem expression_evaluation (a b c : ℚ) (ha : a = 12) (hb : b = 15) (hc : c = 19) :
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b) + a) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b) + 1) = a + b + c - 1 := by
  sorry

end expression_evaluation_l876_87640


namespace diagonals_not_parallel_32gon_l876_87620

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- The number of diagonals parallel to sides in a regular polygon with n sides -/
def num_parallel_diagonals (n : ℕ) : ℕ := (n / 2) * ((n - 4) / 2)

/-- The number of diagonals not parallel to any side in a regular 32-sided polygon -/
theorem diagonals_not_parallel_32gon : 
  num_diagonals 32 - num_parallel_diagonals 32 = 240 := by
  sorry


end diagonals_not_parallel_32gon_l876_87620


namespace special_sequence_bijective_l876_87662

/-- An integer sequence with specific properties -/
def SpecialSequence (a : ℕ → ℤ) : Prop :=
  (∀ n : ℕ, ∃ k > n, a k > 0) ∧  -- Infinitely many positive integers
  (∀ n : ℕ, ∃ k > n, a k < 0) ∧  -- Infinitely many negative integers
  (∀ n : ℕ+, Function.Injective (fun i => a i % n))  -- Distinct remainders

/-- The main theorem -/
theorem special_sequence_bijective (a : ℕ → ℤ) (h : SpecialSequence a) :
  Function.Bijective a :=
sorry

end special_sequence_bijective_l876_87662


namespace characterize_valid_triples_l876_87692

def is_valid_triple (a b c : ℕ+) : Prop :=
  (1 : ℚ) / a + 2 / b + 3 / c = 1 ∧
  Nat.Prime a ∧
  a ≤ b ∧ b ≤ c

def valid_triples : Set (ℕ+ × ℕ+ × ℕ+) :=
  {(2, 5, 30), (2, 6, 18), (2, 7, 14), (2, 8, 12), (2, 10, 10),
   (3, 4, 18), (3, 6, 9), (5, 4, 10)}

theorem characterize_valid_triples :
  ∀ a b c : ℕ+, is_valid_triple a b c ↔ (a, b, c) ∈ valid_triples := by
  sorry

end characterize_valid_triples_l876_87692


namespace total_tree_count_l876_87688

def douglas_fir_count : ℕ := 350
def douglas_fir_cost : ℕ := 300
def ponderosa_pine_cost : ℕ := 225
def total_cost : ℕ := 217500

theorem total_tree_count : 
  ∃ (ponderosa_pine_count : ℕ),
    douglas_fir_count * douglas_fir_cost + 
    ponderosa_pine_count * ponderosa_pine_cost = total_cost ∧
    douglas_fir_count + ponderosa_pine_count = 850 :=
by sorry

end total_tree_count_l876_87688
