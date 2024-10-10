import Mathlib

namespace pyramid_cases_l1522_152270

/-- The sum of the first n triangular numbers -/
def sum_triangular (n : ℕ) : ℕ :=
  (n * (n + 1) * (n + 2)) / 6

/-- The pyramid has 6 levels -/
def pyramid_levels : ℕ := 6

theorem pyramid_cases : sum_triangular pyramid_levels = 56 := by
  sorry

end pyramid_cases_l1522_152270


namespace rectangle_square_equal_area_l1522_152223

theorem rectangle_square_equal_area : 
  ∀ (rectangle_width rectangle_length square_side : ℝ),
    rectangle_width = 2 →
    rectangle_length = 18 →
    square_side = 6 →
    rectangle_width * rectangle_length = square_side * square_side := by
  sorry

end rectangle_square_equal_area_l1522_152223


namespace sum_30_to_40_proof_l1522_152205

def sum_30_to_40 : ℕ := (List.range 11).map (· + 30) |>.sum

def even_count_30_to_40 : ℕ := (List.range 11).map (· + 30) |>.filter (· % 2 = 0) |>.length

theorem sum_30_to_40_proof : sum_30_to_40 = 385 :=
  by
  have h1 : sum_30_to_40 + even_count_30_to_40 = 391 := by sorry
  sorry

#eval sum_30_to_40
#eval even_count_30_to_40

end sum_30_to_40_proof_l1522_152205


namespace remaining_money_for_sharpeners_l1522_152262

def total_money : ℕ := 100
def notebook_price : ℕ := 5
def notebooks_bought : ℕ := 4
def eraser_price : ℕ := 4
def erasers_bought : ℕ := 10
def highlighter_cost : ℕ := 30

def heaven_notebook_cost : ℕ := notebook_price * notebooks_bought
def brother_eraser_cost : ℕ := eraser_price * erasers_bought
def brother_total_cost : ℕ := brother_eraser_cost + highlighter_cost

theorem remaining_money_for_sharpeners :
  total_money - (heaven_notebook_cost + brother_total_cost) = 10 := by
  sorry

end remaining_money_for_sharpeners_l1522_152262


namespace percentage_without_scholarship_l1522_152286

/-- Represents the percentage of students who won't get a scholarship in a school with a given ratio of boys to girls and scholarship rates. -/
theorem percentage_without_scholarship
  (boy_girl_ratio : ℚ)
  (boy_scholarship_rate : ℚ)
  (girl_scholarship_rate : ℚ)
  (h1 : boy_girl_ratio = 5 / 6)
  (h2 : boy_scholarship_rate = 1 / 4)
  (h3 : girl_scholarship_rate = 1 / 5) :
  (1 - (boy_girl_ratio * boy_scholarship_rate + girl_scholarship_rate) / (boy_girl_ratio + 1)) * 100 =
  (1 - (1.25 + 1.2) / 11) * 100 :=
by sorry

end percentage_without_scholarship_l1522_152286


namespace average_and_square_difference_l1522_152279

theorem average_and_square_difference (y : ℝ) : 
  (45 + y) / 2 = 50 → (y - 45)^2 = 100 := by
  sorry

end average_and_square_difference_l1522_152279


namespace class_average_problem_l1522_152263

theorem class_average_problem (x : ℝ) :
  (0.2 * x + 0.5 * 60 + 0.3 * 40 = 58) →
  x = 80 := by
  sorry

end class_average_problem_l1522_152263


namespace angles_equal_necessary_not_sufficient_l1522_152232

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines
variable (parallel : Line → Line → Prop)

-- Define the angle between a line and a plane
variable (angle : Line → Plane → ℝ)

-- State the theorem
theorem angles_equal_necessary_not_sufficient
  (m n : Line) (a : Plane) :
  (∀ (l₁ l₂ : Line), parallel l₁ l₂ → angle l₁ a = angle l₂ a) ∧
  ¬(∀ (l₁ l₂ : Line), angle l₁ a = angle l₂ a → parallel l₁ l₂) :=
sorry

end angles_equal_necessary_not_sufficient_l1522_152232


namespace monotone_increasing_interval_l1522_152233

/-- The function f(x) = sin(x/2) + cos(x/2) is monotonically increasing 
    on the intervals [4kπ - 3π/2, 4kπ + π/2] for all integer k. -/
theorem monotone_increasing_interval (k : ℤ) :
  StrictMonoOn (fun x => Real.sin (x/2) + Real.cos (x/2))
    (Set.Icc (4 * k * Real.pi - 3 * Real.pi / 2) (4 * k * Real.pi + Real.pi / 2)) :=
sorry

end monotone_increasing_interval_l1522_152233


namespace intersection_complement_equality_l1522_152282

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def P : Set Nat := {1, 2, 3, 4, 5}
def Q : Set Nat := {3, 4, 5, 6, 7}

theorem intersection_complement_equality : P ∩ (U \ Q) = {1, 2} := by
  sorry

end intersection_complement_equality_l1522_152282


namespace xiao_liang_arrival_time_l1522_152231

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60,
    minutes := totalMinutes % 60,
    h_valid := by sorry
    m_valid := by sorry }

theorem xiao_liang_arrival_time :
  let departure_time : Time := ⟨7, 40, by sorry, by sorry⟩
  let journey_duration : Nat := 25
  let arrival_time : Time := addMinutes departure_time journey_duration
  arrival_time = ⟨8, 5, by sorry, by sorry⟩ := by sorry

end xiao_liang_arrival_time_l1522_152231


namespace hyperbola_ac_range_l1522_152295

-- Define the hyperbola and its properties
structure Hyperbola where
  focal_distance : ℝ
  a : ℝ
  left_focus : ℝ × ℝ
  right_focus : ℝ × ℝ
  point_on_right_branch : ℝ × ℝ

-- Define the theorem
theorem hyperbola_ac_range (h : Hyperbola) : 
  h.focal_distance = 4 → 
  h.a < 2 →
  let A := h.left_focus
  let B := h.right_focus
  let C := h.point_on_right_branch
  (dist C B - dist C A = 2 * h.a) →
  (dist A C + dist B C + dist A B = 10) →
  (3 < dist A C) ∧ (dist A C < 5) := by
  sorry

-- Note: dist is assumed to be a function that calculates the distance between two points

end hyperbola_ac_range_l1522_152295


namespace polynomial_roots_problem_l1522_152255

theorem polynomial_roots_problem (c d : ℤ) (h1 : c ≠ 0) (h2 : d ≠ 0) 
  (h3 : ∃ p q : ℤ, (X - p)^2 * (X - q) = X^3 + c*X^2 + d*X + 12*c) : 
  |c * d| = 192 := by
  sorry

end polynomial_roots_problem_l1522_152255


namespace face_value_of_shares_l1522_152275

/-- Theorem: Face value of shares given investment and dividend information -/
theorem face_value_of_shares 
  (investment : ℝ) 
  (premium_rate : ℝ) 
  (dividend_rate : ℝ) 
  (dividend_amount : ℝ) 
  (h1 : investment = 14400)
  (h2 : premium_rate = 0.20)
  (h3 : dividend_rate = 0.06)
  (h4 : dividend_amount = 720) :
  ∃ (face_value : ℝ), 
    face_value = 12000 ∧ 
    investment = face_value * (1 + premium_rate) ∧
    dividend_amount = face_value * dividend_rate :=
by sorry

end face_value_of_shares_l1522_152275


namespace phone_price_reduction_l1522_152290

theorem phone_price_reduction (reduced_price : ℝ) (percentage : ℝ) 
  (h1 : reduced_price = 1800)
  (h2 : percentage = 90/100)
  (h3 : reduced_price = percentage * (reduced_price / percentage)) :
  reduced_price / percentage - reduced_price = 200 := by
  sorry

end phone_price_reduction_l1522_152290


namespace point_comparison_l1522_152220

/-- Given points A(-3,m) and B(2,n) lie on the line y = -2x + 1, prove that m > n -/
theorem point_comparison (m n : ℝ) : 
  ((-3 : ℝ), m) ∈ {p : ℝ × ℝ | p.2 = -2 * p.1 + 1} → 
  ((2 : ℝ), n) ∈ {p : ℝ × ℝ | p.2 = -2 * p.1 + 1} → 
  m > n := by
  sorry

end point_comparison_l1522_152220


namespace sweets_per_person_l1522_152210

/-- Represents the number of sweets Jennifer has of each color -/
structure Sweets :=
  (green : ℕ)
  (blue : ℕ)
  (yellow : ℕ)

/-- Calculates the total number of sweets -/
def total_sweets (s : Sweets) : ℕ := s.green + s.blue + s.yellow

/-- Represents the number of people sharing the sweets -/
def num_people : ℕ := 4

/-- Jennifer's sweets -/
def jennifer_sweets : Sweets := ⟨212, 310, 502⟩

/-- Theorem: Each person gets 256 sweets when Jennifer's sweets are shared equally -/
theorem sweets_per_person :
  (total_sweets jennifer_sweets) / num_people = 256 := by sorry

end sweets_per_person_l1522_152210


namespace min_value_of_w_min_value_achievable_l1522_152245

theorem min_value_of_w (x y : ℝ) : 3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 27 ≥ 81/4 := by
  sorry

theorem min_value_achievable : ∃ x y : ℝ, 3 * x^2 + 3 * y^2 + 9 * x - 6 * y + 27 = 81/4 := by
  sorry

end min_value_of_w_min_value_achievable_l1522_152245


namespace triple_counted_number_l1522_152297

def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ (n % 10) % 5 = 0

def sum_valid_numbers : ℕ := sorry

theorem triple_counted_number (triple_counted : ℕ) 
  (h1 : is_valid_number triple_counted)
  (h2 : sum_valid_numbers + 2 * triple_counted = 1035) :
  triple_counted = 45 := by sorry

end triple_counted_number_l1522_152297


namespace team_average_correct_l1522_152225

theorem team_average_correct (v w x y : ℝ) (h : v < w ∧ w < x ∧ x < y) : 
  ((v + w) / 2 + (x + y) / 2) / 2 = (v + w + x + y) / 4 := by
  sorry

end team_average_correct_l1522_152225


namespace mean_equality_implies_z_value_l1522_152283

theorem mean_equality_implies_z_value :
  let x₁ : ℚ := 7
  let x₂ : ℚ := 11
  let x₃ : ℚ := 23
  let y₁ : ℚ := 15
  let mean_xyz : ℚ := (x₁ + x₂ + x₃) / 3
  let mean_yz : ℚ := (y₁ + z) / 2
  mean_xyz = mean_yz → z = 37 / 3 :=
by
  sorry

end mean_equality_implies_z_value_l1522_152283


namespace revenue_decrease_65_percent_l1522_152217

/-- Represents the change in revenue when tax is reduced and consumption is increased -/
def revenue_change (tax_reduction : ℝ) (consumption_increase : ℝ) : ℝ :=
  (1 - tax_reduction) * (1 + consumption_increase) - 1

/-- Theorem stating that a 15% tax reduction and 10% consumption increase results in a 6.5% revenue decrease -/
theorem revenue_decrease_65_percent :
  revenue_change 0.15 0.10 = -0.065 := by
  sorry

end revenue_decrease_65_percent_l1522_152217


namespace average_problem_l1522_152299

theorem average_problem (c d e : ℝ) : 
  (4 + 6 + 9 + c + d + e) / 6 = 20 → (c + d + e) / 3 = 101 / 3 := by
  sorry

end average_problem_l1522_152299


namespace female_managers_count_female_managers_is_200_l1522_152235

/-- The number of female managers in a company, given certain conditions. -/
theorem female_managers_count (total_employees : ℕ) : ℕ := by
  -- Define the total number of female employees
  let female_employees : ℕ := 500

  -- Define the ratio of managers to all employees
  let manager_ratio : ℚ := 2 / 5

  -- Define the ratio of male managers to male employees
  let male_manager_ratio : ℚ := 2 / 5

  -- The number of female managers
  let female_managers : ℕ := 200

  sorry

/-- The main theorem stating that the number of female managers is 200. -/
theorem female_managers_is_200 : female_managers_count = 200 := by
  sorry

end female_managers_count_female_managers_is_200_l1522_152235


namespace rectangle_area_problem_l1522_152241

theorem rectangle_area_problem (w : ℝ) (L L' A : ℝ) : 
  w = 10 →                      -- Width is 10 m
  A = L * w →                   -- Original area
  L' * w = (4/3) * A →          -- New area is 1 1/3 times original
  2 * L' + 2 * w = 60 →         -- New perimeter is 60 m
  A = 150                       -- Original area is 150 square meters
:= by sorry

end rectangle_area_problem_l1522_152241


namespace min_value_of_xy_l1522_152260

theorem min_value_of_xy (x y : ℝ) (hx : x > 1) (hy : y > 1)
  (h_geom : (Real.log x) * (Real.log y) = 1/4) : 
  ∀ z, x * y ≥ z → z ≤ Real.exp 1 := by
sorry

end min_value_of_xy_l1522_152260


namespace power_tower_mod_2000_l1522_152208

theorem power_tower_mod_2000 : 2^(2^(2^2)) ≡ 536 [ZMOD 2000] := by
  sorry

end power_tower_mod_2000_l1522_152208


namespace solve_exponential_equation_l1522_152293

theorem solve_exponential_equation :
  ∃ x : ℝ, 3^(3*x) = Real.sqrt 81 ∧ x = 2/3 := by
  sorry

end solve_exponential_equation_l1522_152293


namespace book_chapters_l1522_152212

theorem book_chapters (total_pages : ℕ) (pages_per_chapter : ℕ) (h1 : total_pages = 555) (h2 : pages_per_chapter = 111) :
  total_pages / pages_per_chapter = 5 := by
sorry

end book_chapters_l1522_152212


namespace peach_difference_l1522_152209

theorem peach_difference (red_peaches green_peaches : ℕ) 
  (h1 : red_peaches = 5) 
  (h2 : green_peaches = 11) : 
  green_peaches - red_peaches = 6 := by
sorry

end peach_difference_l1522_152209


namespace frank_peanuts_theorem_l1522_152218

def frank_peanuts (one_dollar_bills five_dollar_bills ten_dollar_bills twenty_dollar_bills : ℕ)
  (peanut_cost_per_pound : ℚ) (change : ℚ) (days_in_week : ℕ) : Prop :=
  let initial_money : ℚ := one_dollar_bills + 5 * five_dollar_bills + 10 * ten_dollar_bills + 20 * twenty_dollar_bills
  let spent_money : ℚ := initial_money - change
  let pounds_bought : ℚ := spent_money / peanut_cost_per_pound
  pounds_bought / days_in_week = 3

theorem frank_peanuts_theorem :
  frank_peanuts 7 4 2 1 3 4 7 :=
by
  sorry

end frank_peanuts_theorem_l1522_152218


namespace parking_probability_l1522_152253

/-- The number of parking spaces -/
def total_spaces : ℕ := 20

/-- The number of cars parked -/
def parked_cars : ℕ := 15

/-- The number of adjacent spaces required -/
def required_spaces : ℕ := 3

/-- The probability of finding the required adjacent empty spaces -/
def probability_of_parking : ℚ := 12501 / 15504

theorem parking_probability :
  (total_spaces : ℕ) = 20 →
  (parked_cars : ℕ) = 15 →
  (required_spaces : ℕ) = 3 →
  probability_of_parking = 12501 / 15504 := by
  sorry

end parking_probability_l1522_152253


namespace heart_five_three_l1522_152214

-- Define the ♥ operation
def heart (x y : ℝ) : ℝ := 4 * x - 2 * y

-- Theorem statement
theorem heart_five_three : heart 5 3 = 14 := by
  sorry

end heart_five_three_l1522_152214


namespace extra_planks_count_l1522_152200

/-- The number of planks Charlie got -/
def charlie_planks : ℕ := 10

/-- The number of planks Charlie's father got -/
def father_planks : ℕ := 10

/-- The total number of wood pieces they have -/
def total_wood : ℕ := 35

/-- The number of extra planks initially in the house -/
def extra_planks : ℕ := total_wood - (charlie_planks + father_planks)

theorem extra_planks_count : extra_planks = 15 := by
  sorry

end extra_planks_count_l1522_152200


namespace imaginary_part_of_z_squared_minus_one_l1522_152268

theorem imaginary_part_of_z_squared_minus_one (z : ℂ) :
  z = 1 + Complex.I →
  Complex.im ((z + 1) * (z - 1)) = 2 := by sorry

end imaginary_part_of_z_squared_minus_one_l1522_152268


namespace intersection_distance_l1522_152234

/-- The distance between the intersection points of a circle and a line --/
theorem intersection_distance (x y : ℝ) : 
  x^2 + y^2 = 25 → 
  y = x + 3 → 
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁^2 + y₁^2 = 25 ∧ 
    y₁ = x₁ + 3 ∧ 
    x₂^2 + y₂^2 = 25 ∧ 
    y₂ = x₂ + 3 ∧ 
    (x₂ - x₁)^2 + (y₂ - y₁)^2 = 72 := by
  sorry

#check intersection_distance

end intersection_distance_l1522_152234


namespace annual_interest_rate_proof_l1522_152211

theorem annual_interest_rate_proof (investment1 investment2 interest1 interest2 : ℝ) 
  (h1 : investment1 = 5000)
  (h2 : investment2 = 20000)
  (h3 : interest1 = 250)
  (h4 : interest2 = 1000)
  (h5 : interest1 / investment1 = interest2 / investment2) :
  interest1 / investment1 = 0.05 := by
  sorry

end annual_interest_rate_proof_l1522_152211


namespace jimmy_lost_points_l1522_152258

def jimmy_problem (points_to_pass : ℕ) (points_per_exam : ℕ) (num_exams : ℕ) (extra_points : ℕ) : Prop :=
  let total_exam_points := points_per_exam * num_exams
  let current_points := points_to_pass + extra_points
  let lost_points := total_exam_points - current_points
  lost_points = 5

theorem jimmy_lost_points :
  jimmy_problem 50 20 3 5 := by
  sorry

end jimmy_lost_points_l1522_152258


namespace complex_equation_solution_l1522_152288

theorem complex_equation_solution (a : ℝ) (h : (1 + a * Complex.I) * Complex.I = 3 + Complex.I) : a = -3 := by
  sorry

end complex_equation_solution_l1522_152288


namespace alex_shirts_l1522_152239

theorem alex_shirts (alex joe ben : ℕ) 
  (h1 : joe = alex + 3) 
  (h2 : ben = joe + 8) 
  (h3 : ben = 15) : 
  alex = 4 := by
sorry

end alex_shirts_l1522_152239


namespace f_properties_l1522_152274

noncomputable def f (x : ℝ) : ℝ := x^2 / (1 - x)

theorem f_properties :
  (∃ (max_val : ℝ), max_val = -4 ∧ ∀ x ≠ 1, f x ≤ max_val) ∧
  (∀ x ≠ 1, f (1 - x) + f (1 + x) = -4) ∧
  (∀ x₁ x₂ : ℝ, x₁ > 1 → x₂ > 1 → f ((x₁ + x₂) / 2) ≥ (f x₁ + f x₂) / 2) :=
by sorry

end f_properties_l1522_152274


namespace book_distribution_l1522_152252

theorem book_distribution (n m : ℕ) (hn : n = 7) (hm : m = 3) :
  (Nat.factorial n) / (Nat.factorial (n - m)) = 210 :=
sorry

end book_distribution_l1522_152252


namespace binomial_8_choose_5_l1522_152203

theorem binomial_8_choose_5 : Nat.choose 8 5 = 56 := by
  sorry

end binomial_8_choose_5_l1522_152203


namespace jellybean_distribution_l1522_152215

/-- Proves that given 70 jellybeans divided equally among 3 nephews and 2 nieces, each child receives 14 jellybeans. -/
theorem jellybean_distribution (total_jellybeans : ℕ) (num_nephews : ℕ) (num_nieces : ℕ)
  (h1 : total_jellybeans = 70)
  (h2 : num_nephews = 3)
  (h3 : num_nieces = 2) :
  total_jellybeans / (num_nephews + num_nieces) = 14 := by
  sorry

end jellybean_distribution_l1522_152215


namespace complement_M_intersect_N_l1522_152292

def M : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x : ℝ | x < 1}

theorem complement_M_intersect_N :
  (Set.univ \ M) ∩ N = {x : ℝ | x < -2} := by sorry

end complement_M_intersect_N_l1522_152292


namespace irrational_approximation_l1522_152251

-- Define α as an irrational real number
variable (α : ℝ) (h : ¬ IsRat α)

-- State the theorem
theorem irrational_approximation :
  ∃ (p q : ℤ), -(1 : ℝ) / (q : ℝ)^2 ≤ α - (p : ℝ) / (q : ℝ) ∧ α - (p : ℝ) / (q : ℝ) ≤ 1 / (q : ℝ)^2 :=
sorry

end irrational_approximation_l1522_152251


namespace second_derivative_y_l1522_152204

noncomputable def x (t : ℝ) : ℝ := Real.log t
noncomputable def y (t : ℝ) : ℝ := Real.sin (2 * t)

theorem second_derivative_y (t : ℝ) (h : t > 0) :
  (deriv^[2] (y ∘ (x⁻¹))) (x t) = -4 * t^2 * Real.sin (2 * t) + 2 * t * Real.cos (2 * t) :=
by sorry

end second_derivative_y_l1522_152204


namespace platform_length_l1522_152247

/-- The length of a platform given train speed and crossing times -/
theorem platform_length (train_speed : ℝ) (platform_time : ℝ) (man_time : ℝ) : 
  train_speed = 72 →
  platform_time = 30 →
  man_time = 19 →
  (train_speed * 1000 / 3600) * platform_time - (train_speed * 1000 / 3600) * man_time = 220 := by
  sorry

end platform_length_l1522_152247


namespace readers_overlap_l1522_152298

theorem readers_overlap (total : ℕ) (science_fiction : ℕ) (literary : ℕ) 
  (h1 : total = 650) 
  (h2 : science_fiction = 250) 
  (h3 : literary = 550) : 
  science_fiction + literary - total = 150 := by
  sorry

end readers_overlap_l1522_152298


namespace power_product_equals_power_sum_l1522_152213

theorem power_product_equals_power_sum (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end power_product_equals_power_sum_l1522_152213


namespace johns_hats_cost_l1522_152278

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks John can wear a different hat each day -/
def weeks_of_different_hats : ℕ := 2

/-- The cost of each hat in dollars -/
def cost_per_hat : ℕ := 50

/-- The total cost of John's hats -/
def total_cost : ℕ := weeks_of_different_hats * days_per_week * cost_per_hat

theorem johns_hats_cost :
  total_cost = 700 := by sorry

end johns_hats_cost_l1522_152278


namespace max_value_not_one_l1522_152261

theorem max_value_not_one :
  let f : ℝ → ℝ := λ x ↦ Real.sin (x + π/4)
  let g : ℝ → ℝ := λ x ↦ Real.cos (x - π/4)
  let y : ℝ → ℝ := λ x ↦ f x * g x
  ∃ M : ℝ, (∀ x, y x ≤ M) ∧ M < 1 :=
by sorry

end max_value_not_one_l1522_152261


namespace carmelas_initial_money_l1522_152243

/-- Proves that Carmela's initial amount of money is $7 given the problem conditions --/
theorem carmelas_initial_money :
  ∀ x : ℕ,
  (∃ (final_amount : ℕ),
    -- Carmela's final amount after giving $1 to each of 4 cousins
    x - 4 = final_amount ∧
    -- Each cousin's final amount after receiving $1
    2 + 1 = final_amount) →
  x = 7 := by
  sorry

end carmelas_initial_money_l1522_152243


namespace solution_set_range_of_m_l1522_152207

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1|

-- Theorem for the solution set of |f(x) - 3| ≤ 4
theorem solution_set :
  {x : ℝ | |f x - 3| ≤ 4} = {x : ℝ | -6 ≤ x ∧ x ≤ 8} := by sorry

-- Theorem for the range of m
theorem range_of_m :
  {m : ℝ | ∀ x, f x + f (x + 3) ≥ m^2 - 2*m} = {m : ℝ | -1 ≤ m ∧ m ≤ 3} := by sorry

end solution_set_range_of_m_l1522_152207


namespace infinite_solutions_diophantine_equation_l1522_152229

theorem infinite_solutions_diophantine_equation :
  ∃ (S : Set (ℕ × ℕ × ℕ)), 
    (∀ (x y z : ℕ), (x, y, z) ∈ S → 
      x > 2008 ∧ y > 2008 ∧ z > 2008 ∧ 
      x^2 + y^2 + z^2 - x*y*z + 10 = 0) ∧
    Set.Infinite S :=
by sorry

end infinite_solutions_diophantine_equation_l1522_152229


namespace nested_subtraction_simplification_l1522_152284

theorem nested_subtraction_simplification (x : ℝ) : 1 - (2 - (3 - (4 - (5 - (6 - x))))) = x - 3 := by
  sorry

end nested_subtraction_simplification_l1522_152284


namespace fourth_term_of_geometric_sequence_l1522_152272

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem fourth_term_of_geometric_sequence 
  (a : ℕ → ℝ) 
  (h_geometric : geometric_sequence a) 
  (h_3 : a 3 = 2) 
  (h_5 : a 5 = 16) : 
  a 4 = 4 * Real.sqrt 2 ∨ a 4 = -4 * Real.sqrt 2 :=
sorry

end fourth_term_of_geometric_sequence_l1522_152272


namespace inequality_proof_l1522_152224

theorem inequality_proof (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  a * b * c ≤ 1 / 9 ∧ 
  a / (b + c) + b / (a + c) + c / (a + b) ≤ 1 / (2 * Real.sqrt (a * b * c)) := by
  sorry

end inequality_proof_l1522_152224


namespace polynomial_sum_l1522_152289

-- Define the polynomials f and g
def f (a b x : ℝ) := x^2 + a*x + b
def g (c d x : ℝ) := x^2 + c*x + d

-- State the theorem
theorem polynomial_sum (a b c d : ℝ) : 
  (∃ (x : ℝ), g c d x = 0 ∧ x = -a/2) →  -- x-coordinate of vertex of f is root of g
  (∃ (x : ℝ), f a b x = 0 ∧ x = -c/2) →  -- x-coordinate of vertex of g is root of f
  (f a b (-a/2) = -25) →                 -- minimum value of f is -25
  (g c d (-c/2) = -25) →                 -- minimum value of g is -25
  (f a b 50 = -50) →                     -- f and g intersect at (50, -50)
  (g c d 50 = -50) →                     -- f and g intersect at (50, -50)
  (a ≠ c ∨ b ≠ d) →                      -- f and g are distinct
  a + c = -200 := by
sorry

end polynomial_sum_l1522_152289


namespace right_triangle_acute_angle_measure_l1522_152264

theorem right_triangle_acute_angle_measure (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b = 90) ∧ (a / b = 5 / 4) → min a b = 40 := by
  sorry

end right_triangle_acute_angle_measure_l1522_152264


namespace teresa_jogging_speed_l1522_152221

theorem teresa_jogging_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 25 → time = 5 → speed = distance / time → speed = 5 :=
by sorry

end teresa_jogging_speed_l1522_152221


namespace derivative_of_f_l1522_152280

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x

-- State the theorem
theorem derivative_of_f (x : ℝ) : 
  (f ∘ Real.cos ∘ (fun x => 2 * x)) x = 1 - 2 * (Real.sin x) ^ 2 → 
  deriv f x = -2 * Real.sin (2 * x) :=
by sorry

end derivative_of_f_l1522_152280


namespace line_relationship_l1522_152228

-- Define the concept of lines in 3D space
structure Line3D where
  -- This is a placeholder definition. In a real implementation, 
  -- we might represent a line using a point and a direction vector.
  id : ℕ

-- Define the relationships between lines
def are_skew (l1 l2 : Line3D) : Prop := sorry
def are_parallel (l1 l2 : Line3D) : Prop := sorry
def are_intersecting (l1 l2 : Line3D) : Prop := sorry

-- State the theorem
theorem line_relationship (a b c : Line3D) 
  (h1 : are_skew a b) (h2 : are_parallel a c) : 
  are_intersecting b c ∨ are_skew b c := by sorry

end line_relationship_l1522_152228


namespace coefficient_x_squared_in_expansion_l1522_152250

theorem coefficient_x_squared_in_expansion : 
  (Finset.range 11).sum (fun k => (Nat.choose 10 k) * (2^k) * if k = 2 then 1 else 0) = 180 := by
  sorry

end coefficient_x_squared_in_expansion_l1522_152250


namespace quadratic_equation_roots_l1522_152226

theorem quadratic_equation_roots (k : ℝ) (a b : ℝ) : 
  (∀ x : ℝ, x^2 - 3*x + k = 0 ↔ x = a ∨ x = b) →
  (a*b + 2*a + 2*b = 1) →
  k = -5 := by
  sorry

end quadratic_equation_roots_l1522_152226


namespace equation_solution_l1522_152201

theorem equation_solution : ∃! x : ℝ, (1 : ℝ) / (x + 3) = (3 : ℝ) / (x - 1) ∧ x = -5 := by
  sorry

end equation_solution_l1522_152201


namespace chord_division_theorem_l1522_152296

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents an annulus -/
structure Annulus where
  inner : Circle
  outer : Circle

/-- Theorem: Given an annulus and a point inside it, there exists a chord passing through the point
    that divides the chord in a given ratio -/
theorem chord_division_theorem (A : Annulus) (P : Point) (m n : ℝ) 
    (h_concentric : A.inner.center = A.outer.center)
    (h_inside : (P.x - A.inner.center.x)^2 + (P.y - A.inner.center.y)^2 > A.inner.radius^2 ∧ 
                (P.x - A.outer.center.x)^2 + (P.y - A.outer.center.y)^2 < A.outer.radius^2)
    (h_positive : m > 0 ∧ n > 0) :
  ∃ (A₁ A₂ : Point),
    -- A₁ is on the inner circle
    (A₁.x - A.inner.center.x)^2 + (A₁.y - A.inner.center.y)^2 = A.inner.radius^2 ∧
    -- A₂ is on the outer circle
    (A₂.x - A.outer.center.x)^2 + (A₂.y - A.outer.center.y)^2 = A.outer.radius^2 ∧
    -- P is on the line segment A₁A₂
    ∃ (t : ℝ), 0 < t ∧ t < 1 ∧ P.x = A₁.x + t * (A₂.x - A₁.x) ∧ P.y = A₁.y + t * (A₂.y - A₁.y) ∧
    -- The ratio A₁P:PA₂ is m:n
    t / (1 - t) = m / n :=
by sorry

end chord_division_theorem_l1522_152296


namespace not_perfect_square_l1522_152291

theorem not_perfect_square (n : ℕ) (h : n > 1) : ¬ ∃ (a : ℕ), 4 * 10^n + 9 = a^2 := by
  sorry

end not_perfect_square_l1522_152291


namespace consecutive_numbers_product_l1522_152271

theorem consecutive_numbers_product (a b c d : ℤ) : 
  (b = a + 1) → (c = b + 1) → (d = c + 1) → (a + d = 109) → (b * c = 2970) := by
  sorry

end consecutive_numbers_product_l1522_152271


namespace triangle_inequality_condition_l1522_152237

theorem triangle_inequality_condition (m : ℝ) :
  (m > 0) →
  (∀ (x y : ℝ), x > 0 → y > 0 →
    (x + y + m * Real.sqrt (x * y) > Real.sqrt (x^2 + y^2 + x * y) ∧
     x + y + Real.sqrt (x^2 + y^2 + x * y) > m * Real.sqrt (x * y) ∧
     m * Real.sqrt (x * y) + Real.sqrt (x^2 + y^2 + x * y) > x + y)) ↔
  (m > 2 - Real.sqrt 3 ∧ m < 2 + Real.sqrt 3) :=
by sorry

end triangle_inequality_condition_l1522_152237


namespace fraction_simplification_l1522_152236

theorem fraction_simplification : 
  (1 / 4 - 1 / 5) / (1 / 3 - 1 / 6) = 3 / 10 := by
  sorry

end fraction_simplification_l1522_152236


namespace cos_two_theta_value_l1522_152240

theorem cos_two_theta_value (θ : ℝ) 
  (h1 : 3 * Real.sin (2 * θ) = 4 * Real.tan θ) 
  (h2 : ∀ k : ℤ, θ ≠ k * Real.pi) : 
  Real.cos (2 * θ) = 1 / 3 := by
  sorry

end cos_two_theta_value_l1522_152240


namespace max_remainder_eleven_l1522_152277

theorem max_remainder_eleven (x : ℕ+) : ∃ (q r : ℕ), x = 11 * q + r ∧ r ≤ 10 ∧ ∀ (r' : ℕ), x = 11 * q + r' → r' ≤ r :=
sorry

end max_remainder_eleven_l1522_152277


namespace function_inequality_l1522_152266

theorem function_inequality (f : ℝ → ℝ) (h1 : Differentiable ℝ f) 
  (h2 : ∀ x < -1, (deriv f) x ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end function_inequality_l1522_152266


namespace scale_division_l1522_152281

/-- Given a scale of length 80 inches divided into 5 equal parts, 
    prove that the length of each part is 16 inches. -/
theorem scale_division (total_length : ℕ) (num_parts : ℕ) (part_length : ℕ) 
  (h1 : total_length = 80) 
  (h2 : num_parts = 5) 
  (h3 : part_length * num_parts = total_length) : 
  part_length = 16 := by
  sorry

end scale_division_l1522_152281


namespace geometric_sequence_property_l1522_152248

def is_geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

theorem geometric_sequence_property (b : ℕ → ℝ) :
  is_geometric_sequence b →
  b 9 = (3 + 5) / 2 →
  b 1 * b 17 = 16 := by
  sorry

end geometric_sequence_property_l1522_152248


namespace scoop_size_l1522_152242

/-- Given the total amount of ingredients and the total number of scoops, 
    calculate the size of each scoop. -/
theorem scoop_size (total_cups : ℚ) (total_scoops : ℕ) 
  (h1 : total_cups = 5) 
  (h2 : total_scoops = 15) : 
  total_cups / total_scoops = 1 / 3 := by
  sorry

end scoop_size_l1522_152242


namespace complex_in_first_quadrant_l1522_152256

-- Define the operation
def determinant (a b c d : ℂ) : ℂ := a * d - b * c

-- Define the first quadrant
def is_in_first_quadrant (z : ℂ) : Prop := z.re > 0 ∧ z.im > 0

-- State the theorem
theorem complex_in_first_quadrant :
  ∃ z : ℂ, determinant z (1 + Complex.I) 2 1 = 0 ∧ is_in_first_quadrant z :=
sorry

end complex_in_first_quadrant_l1522_152256


namespace fraction_inequality_l1522_152238

theorem fraction_inequality (x : ℝ) (h : x ≠ 1) :
  (1 / (x - 1) ≤ 1) ↔ (x < 1 ∨ x ≥ 2) := by sorry

end fraction_inequality_l1522_152238


namespace spy_is_B_l1522_152222

-- Define the possible roles
inductive Role
| Knight
| Liar
| Spy

-- Define the defendants
inductive Defendant
| A
| B
| C

-- Define a function to represent the role of each defendant
def role : Defendant → Role := sorry

-- Define the answers given by defendants
def answer_A : Bool := sorry
def answer_B : Bool := sorry
def answer_remaining : Bool := sorry

-- Define which defendant was released
def released : Defendant := sorry

-- Define which defendant was asked the final question
def final_asked : Defendant := sorry

-- Axioms based on the problem conditions
axiom different_roles : 
  ∃! (a b c : Defendant), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    role a = Role.Knight ∧ role b = Role.Liar ∧ role c = Role.Spy

axiom judge_deduction : 
  ∃! (spy : Defendant), role spy = Role.Spy

axiom released_not_spy : 
  role released ≠ Role.Spy

axiom final_question_neighbor : 
  final_asked ≠ released ∧ 
  (final_asked = Defendant.A ∨ final_asked = Defendant.B)

-- The theorem to prove
theorem spy_is_B : 
  role Defendant.B = Role.Spy := by sorry

end spy_is_B_l1522_152222


namespace sector_area_l1522_152265

theorem sector_area (θ : Real) (s : Real) (A : Real) :
  θ = 2 ∧ s = 4 → A = 4 :=
by
  sorry

end sector_area_l1522_152265


namespace even_function_implies_a_eq_neg_one_l1522_152285

def f (a : ℝ) (x : ℝ) : ℝ := (x - 1) * (x - a)

theorem even_function_implies_a_eq_neg_one (a : ℝ) :
  (∀ x : ℝ, f a x = f a (-x)) → a = -1 := by
  sorry

end even_function_implies_a_eq_neg_one_l1522_152285


namespace power_multiplication_equality_l1522_152267

theorem power_multiplication_equality (m : ℝ) : m^2 * (-m)^4 = m^6 := by
  sorry

end power_multiplication_equality_l1522_152267


namespace river_width_l1522_152246

/-- Proves that given a river with specified depth, flow rate, and discharge volume,
    the width of the river is 25 meters. -/
theorem river_width (depth : ℝ) (flow_rate_kmph : ℝ) (discharge_volume : ℝ) :
  depth = 8 →
  flow_rate_kmph = 8 →
  discharge_volume = 26666.666666666668 →
  (discharge_volume / (depth * (flow_rate_kmph * 1000 / 60))) = 25 := by
  sorry


end river_width_l1522_152246


namespace linear_function_property_l1522_152254

/-- A linear function is a function of the form f(x) = mx + b where m and b are constants -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

theorem linear_function_property (g : ℝ → ℝ) 
  (h_linear : LinearFunction g) (h_diff : g 10 - g 5 = 20) :
  g 20 - g 5 = 60 := by
sorry

end linear_function_property_l1522_152254


namespace francies_allowance_l1522_152294

/-- Francie's allowance problem -/
theorem francies_allowance (x : ℚ) : 
  (∀ (total_saved half_spent remaining : ℚ),
    total_saved = 8 * x + 6 * 6 →
    half_spent = total_saved / 2 →
    remaining = half_spent - 35 →
    remaining = 3) →
  x = 5 := by sorry

end francies_allowance_l1522_152294


namespace not_p_or_not_q_is_true_l1522_152202

-- Define propositions p and q
variable (p q : Prop)

-- Define the conditions
axiom p_true : p
axiom q_false : ¬q

-- Theorem to prove
theorem not_p_or_not_q_is_true : ¬p ∨ ¬q := by
  sorry

end not_p_or_not_q_is_true_l1522_152202


namespace son_age_proof_l1522_152219

theorem son_age_proof (son_age father_age : ℕ) : 
  father_age = son_age + 22 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 20 := by
sorry

end son_age_proof_l1522_152219


namespace BE_length_l1522_152244

-- Define the points
variable (A B C D E F G H : Point)

-- Define the square ABCD
def is_square (A B C D : Point) : Prop := sorry

-- Define that E is on the extension of BC
def on_extension (E B C : Point) : Prop := sorry

-- Define the square AEFG
def is_square_AEFG (A E F G : Point) : Prop := sorry

-- Define that A and G are on the same side of BE
def same_side (A G B E : Point) : Prop := sorry

-- Define that H is on the extension of BD and intersects AF
def intersects_extension (H B D A F : Point) : Prop := sorry

-- Define the lengths
def length (P Q : Point) : ℝ := sorry

-- State the theorem
theorem BE_length 
  (h1 : is_square A B C D)
  (h2 : on_extension E B C)
  (h3 : is_square_AEFG A E F G)
  (h4 : same_side A G B E)
  (h5 : intersects_extension H B D A F)
  (h6 : length H D = Real.sqrt 2)
  (h7 : length F H = 5 * Real.sqrt 2) :
  length B E = 8 := by sorry

end BE_length_l1522_152244


namespace unique_root_quadratic_l1522_152276

theorem unique_root_quadratic (k : ℝ) :
  (∃! a : ℝ, (k^2 - 9) * a^2 - 2*(k + 1)*a + 1 = 0) →
  (k = 3 ∨ k = -3 ∨ k = -5) :=
by sorry

end unique_root_quadratic_l1522_152276


namespace factorization_3x_squared_minus_9x_l1522_152230

theorem factorization_3x_squared_minus_9x (x : ℝ) : 3 * x^2 - 9 * x = 3 * x * (x - 3) := by
  sorry

end factorization_3x_squared_minus_9x_l1522_152230


namespace tan_405_degrees_l1522_152206

theorem tan_405_degrees : Real.tan (405 * π / 180) = 1 := by
  sorry

end tan_405_degrees_l1522_152206


namespace carnival_spending_theorem_l1522_152227

def carnival_spending (bumper_car_rides : ℕ) (space_shuttle_rides : ℕ) (ferris_wheel_rides : ℕ)
  (bumper_car_cost : ℕ) (space_shuttle_cost : ℕ) (ferris_wheel_cost : ℕ) : ℕ :=
  bumper_car_rides * bumper_car_cost +
  space_shuttle_rides * space_shuttle_cost +
  2 * ferris_wheel_rides * ferris_wheel_cost

theorem carnival_spending_theorem :
  carnival_spending 2 4 3 2 4 5 = 50 := by
  sorry

end carnival_spending_theorem_l1522_152227


namespace expression_simplification_l1522_152273

theorem expression_simplification (a : ℝ) 
  (h1 : a ≠ -8) 
  (h2 : a ≠ 1) 
  (h3 : a ≠ -1) : 
  (9 / (a + 8) - (a^(1/3) + 2) / (a^(2/3) - 2*a^(1/3) + 4)) * 
  ((a^(4/3) + 8*a^(1/3)) / (1 - a^(2/3))) + 
  (5 - a^(2/3)) / (1 + a^(1/3)) = 5 := by
sorry

end expression_simplification_l1522_152273


namespace sum_abc_equals_eight_l1522_152216

theorem sum_abc_equals_eight (a b c : ℝ) 
  (h : (a - 5)^2 + (b - 6)^2 + (c - 7)^2 - 2*(a - 5)*(b - 6) = 0) : 
  a + b + c = 8 := by
  sorry

end sum_abc_equals_eight_l1522_152216


namespace bubble_gum_count_l1522_152249

/-- The cost of a single piece of bubble gum in cents -/
def cost_per_piece : ℕ := 18

/-- The total cost of all pieces of bubble gum in cents -/
def total_cost : ℕ := 2448

/-- The number of pieces of bubble gum -/
def num_pieces : ℕ := total_cost / cost_per_piece

theorem bubble_gum_count : num_pieces = 136 := by
  sorry

end bubble_gum_count_l1522_152249


namespace regular_polygon_with_405_diagonals_has_30_sides_l1522_152269

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular polygon with 405 diagonals has 30 sides -/
theorem regular_polygon_with_405_diagonals_has_30_sides :
  ∃ (n : ℕ), n > 2 ∧ num_diagonals n = 405 → n = 30 := by
sorry

end regular_polygon_with_405_diagonals_has_30_sides_l1522_152269


namespace cubic_equation_sum_l1522_152259

theorem cubic_equation_sum (r s t : ℝ) : 
  r^3 - 7*r^2 + 11*r = 13 →
  s^3 - 7*s^2 + 11*s = 13 →
  t^3 - 7*t^2 + 11*t = 13 →
  (r+s)/t + (s+t)/r + (t+r)/s = 38/13 := by
  sorry

end cubic_equation_sum_l1522_152259


namespace equal_roots_quadratic_l1522_152257

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0 ∧ 
   ∀ y : ℝ, y^2 - 2*y + m = 0 → y = x) → 
  m = 1 := by
  sorry

end equal_roots_quadratic_l1522_152257


namespace x_lt_1_necessary_not_sufficient_for_ln_x_lt_0_l1522_152287

theorem x_lt_1_necessary_not_sufficient_for_ln_x_lt_0 :
  (∀ x : ℝ, Real.log x < 0 → x < 1) ∧
  (∃ x : ℝ, x < 1 ∧ Real.log x ≥ 0) := by
  sorry

end x_lt_1_necessary_not_sufficient_for_ln_x_lt_0_l1522_152287
