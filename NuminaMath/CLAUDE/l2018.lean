import Mathlib

namespace scientific_notation_of_1300000000_l2018_201844

/-- Expresses 1300000000 in scientific notation -/
theorem scientific_notation_of_1300000000 :
  (1300000000 : ℝ) = 1.3 * (10 ^ 9) := by sorry

end scientific_notation_of_1300000000_l2018_201844


namespace transistors_in_1995_l2018_201867

/-- Moore's law states that the number of transistors doubles every 18 months -/
def moores_law_period : ℕ := 18

/-- Initial year when the count began -/
def initial_year : ℕ := 1985

/-- Initial number of transistors -/
def initial_transistors : ℕ := 500000

/-- Target year for calculation -/
def target_year : ℕ := 1995

/-- Calculate the number of transistors based on Moore's law -/
def calculate_transistors (initial : ℕ) (months : ℕ) : ℕ :=
  initial * (2 ^ (months / moores_law_period))

/-- Theorem stating that the number of transistors in 1995 is 32,000,000 -/
theorem transistors_in_1995 :
  calculate_transistors initial_transistors ((target_year - initial_year) * 12) = 32000000 := by
  sorry

end transistors_in_1995_l2018_201867


namespace problem_solution_l2018_201814

theorem problem_solution (m a b c d : ℚ) 
  (h1 : |m + 1| = 4)
  (h2 : a + b = 0)
  (h3 : a ≠ 0)
  (h4 : c * d = 1) :
  2*a + 2*b + (a + b - 3*c*d) - m = 2 ∨ 2*a + 2*b + (a + b - 3*c*d) - m = -6 := by
sorry

end problem_solution_l2018_201814


namespace problem_solution_l2018_201880

-- Define p₁
def p₁ : Prop := ∃ x : ℝ, x^2 + x + 1 < 0

-- Define p₂
def p₂ : Prop := ∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - 1 ≥ 0

-- Theorem to prove
theorem problem_solution : (¬p₁) ∨ (¬p₂) := by
  sorry

end problem_solution_l2018_201880


namespace smallest_common_factor_l2018_201894

theorem smallest_common_factor (m : ℕ) : m = 108 ↔ 
  (m > 0 ∧ 
   ∃ (k : ℕ), k > 1 ∧ k ∣ (11*m - 3) ∧ k ∣ (8*m + 5) ∧
   ∀ (n : ℕ), n < m → ¬(∃ (l : ℕ), l > 1 ∧ l ∣ (11*n - 3) ∧ l ∣ (8*n + 5))) :=
by sorry

end smallest_common_factor_l2018_201894


namespace quadratic_equation_properties_l2018_201846

theorem quadratic_equation_properties (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hab : a ≤ b) :
  let f : ℝ → ℝ := fun x ↦ x^2 + (a + b - 1 : ℝ) * x + (a * b - a - b : ℝ)
  -- The equation has two distinct real solutions
  ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧
  -- If one solution is an integer, then both are non-positive integers and b < 2a
  (∃ z : ℤ, f (z : ℝ) = 0 → ∃ r s : ℤ, r ≤ 0 ∧ s ≤ 0 ∧ f (r : ℝ) = 0 ∧ f (s : ℝ) = 0 ∧ b < 2 * a) :=
by sorry

end quadratic_equation_properties_l2018_201846


namespace ellipse_foci_distance_l2018_201839

/-- The distance between foci of an ellipse with given semi-major and semi-minor axes -/
theorem ellipse_foci_distance (a b : ℝ) (ha : a = 10) (hb : b = 4) :
  2 * Real.sqrt (a^2 - b^2) = 4 * Real.sqrt 21 :=
by sorry

end ellipse_foci_distance_l2018_201839


namespace total_discount_savings_l2018_201805

def mangoes_per_box : ℕ := 10 * 12 -- 10 dozen

def prices_per_dozen : List ℕ := [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

def total_boxes : ℕ := 36

def discount_rate (boxes : ℕ) : ℚ :=
  if boxes ≥ 30 then 15 / 100
  else if boxes ≥ 20 then 10 / 100
  else if boxes ≥ 10 then 5 / 100
  else 0

theorem total_discount_savings : 
  let total_cost := (prices_per_dozen.map (· * mangoes_per_box)).sum * total_boxes
  let discounted_cost := total_cost * (1 - discount_rate total_boxes)
  total_cost - discounted_cost = 5090 := by
  sorry

end total_discount_savings_l2018_201805


namespace george_red_marbles_l2018_201878

/-- The number of red marbles in George's collection --/
def red_marbles (total yellow white green : ℕ) : ℕ :=
  total - (yellow + white + green)

/-- Theorem stating the number of red marbles in George's collection --/
theorem george_red_marbles :
  ∀ (total yellow white green : ℕ),
    total = 50 →
    yellow = 12 →
    white = total / 2 →
    green = yellow / 2 →
    red_marbles total yellow white green = 7 := by
  sorry

end george_red_marbles_l2018_201878


namespace boys_to_girls_ratio_l2018_201855

theorem boys_to_girls_ratio 
  (T : ℕ) -- Total number of students
  (B : ℕ) -- Number of boys
  (G : ℕ) -- Number of girls
  (h1 : T = B + G) -- Total is sum of boys and girls
  (h2 : 2 * G = 3 * (T / 4)) -- 2/3 of girls = 1/4 of total
  : B * 3 = G * 5 := by
sorry

end boys_to_girls_ratio_l2018_201855


namespace teachers_pizza_fraction_l2018_201823

theorem teachers_pizza_fraction (teachers : ℕ) (staff : ℕ) (staff_pizza_fraction : ℚ) (non_pizza_eaters : ℕ) :
  teachers = 30 →
  staff = 45 →
  staff_pizza_fraction = 4/5 →
  non_pizza_eaters = 19 →
  (teachers : ℚ) * (2/3) + (staff : ℚ) * staff_pizza_fraction = (teachers + staff : ℚ) - non_pizza_eaters := by
  sorry

end teachers_pizza_fraction_l2018_201823


namespace consecutive_integers_sum_l2018_201809

theorem consecutive_integers_sum :
  (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ y = x + 1 ∧ z = y + 1 ∧ x + y + z = 48) ∧
  (∃ (x y z w : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ 
    Even x ∧ Even y ∧ Even z ∧ Even w ∧
    y = x + 2 ∧ z = y + 2 ∧ w = z + 2 ∧ x + y + z + w = 52) :=
by sorry

end consecutive_integers_sum_l2018_201809


namespace not_sufficient_nor_necessary_l2018_201858

theorem not_sufficient_nor_necessary (a b : ℝ) : 
  ¬(∀ a b : ℝ, a^2 > b^2 → a > b) ∧ ¬(∀ a b : ℝ, a > b → a^2 > b^2) := by
  sorry

end not_sufficient_nor_necessary_l2018_201858


namespace spherical_to_rectangular_conversion_l2018_201803

theorem spherical_to_rectangular_conversion :
  let ρ : ℝ := 8
  let θ : ℝ := 5 * π / 4
  let φ : ℝ := π / 6
  let x : ℝ := ρ * Real.sin φ * Real.cos θ
  let y : ℝ := ρ * Real.sin φ * Real.sin θ
  let z : ℝ := ρ * Real.cos φ
  (x, y, z) = (-2 * Real.sqrt 2, -2 * Real.sqrt 2, 4 * Real.sqrt 3) :=
by sorry

end spherical_to_rectangular_conversion_l2018_201803


namespace final_balance_is_correct_l2018_201830

/-- Represents a currency with its exchange rate to USD -/
structure Currency where
  name : String
  exchange_rate : Float

/-- Represents a transaction with amount, currency, and discount -/
structure Transaction where
  amount : Float
  currency : Currency
  discount : Float

def initial_balance : Float := 126.00

def gbp : Currency := { name := "GBP", exchange_rate := 1.39 }
def eur : Currency := { name := "EUR", exchange_rate := 1.18 }
def jpy : Currency := { name := "JPY", exchange_rate := 0.0091 }
def usd : Currency := { name := "USD", exchange_rate := 1.0 }

def uk_transaction : Transaction := { amount := 50.0, currency := gbp, discount := 0.1 }
def france_transaction : Transaction := { amount := 70.0, currency := eur, discount := 0.15 }
def japan_transaction : Transaction := { amount := 10000.0, currency := jpy, discount := 0.05 }
def us_gas_transaction : Transaction := { amount := 25.0, currency := gbp, discount := 0.0 }
def return_transaction : Transaction := { amount := 45.0, currency := usd, discount := 0.0 }

def monthly_interest_rate : Float := 0.015

def calculate_final_balance (initial_balance : Float) 
  (transactions : List Transaction) 
  (return_transaction : Transaction)
  (monthly_interest_rate : Float) : Float :=
  sorry

theorem final_balance_is_correct :
  calculate_final_balance initial_balance 
    [uk_transaction, france_transaction, japan_transaction, us_gas_transaction]
    return_transaction
    monthly_interest_rate = 340.00 := by
  sorry

end final_balance_is_correct_l2018_201830


namespace balloon_count_l2018_201853

def total_balloons (joan_initial : ℕ) (popped : ℕ) (jessica : ℕ) : ℕ :=
  (joan_initial - popped) + jessica

theorem balloon_count : total_balloons 9 5 2 = 6 := by
  sorry

end balloon_count_l2018_201853


namespace trapezoid_segment_length_l2018_201832

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  height : ℝ
  midpoint_ratio : ℝ
  equal_area_segment : ℝ
  base_difference : shorter_base + 120 = longer_base
  area_ratio : (shorter_base + (shorter_base + 60)) / ((shorter_base + 60) + longer_base) = 3 / 4
  equal_areas : equal_area_segment > shorter_base ∧ equal_area_segment < longer_base

/-- The theorem to be proved -/
theorem trapezoid_segment_length (t : Trapezoid) : 
  ⌊t.equal_area_segment ^ 2 / 120⌋ = 45 := by sorry

end trapezoid_segment_length_l2018_201832


namespace inequality_solution_l2018_201849

theorem inequality_solution :
  {n : ℕ+ | 25 - 5 * n.val < 15} = {n : ℕ+ | n.val > 2} := by sorry

end inequality_solution_l2018_201849


namespace yard_length_18_trees_l2018_201879

/-- The length of a yard with equally spaced trees -/
def yardLength (numTrees : ℕ) (distanceBetweenTrees : ℝ) : ℝ :=
  (numTrees - 1 : ℝ) * distanceBetweenTrees

/-- Theorem: The length of a yard with 18 equally spaced trees,
    where the distance between consecutive trees is 15 meters, is 255 meters -/
theorem yard_length_18_trees : yardLength 18 15 = 255 := by
  sorry

#eval yardLength 18 15

end yard_length_18_trees_l2018_201879


namespace smallest_square_containing_circle_l2018_201843

theorem smallest_square_containing_circle (r : ℝ) (h : r = 7) :
  (2 * r) ^ 2 = 196 := by
  sorry

end smallest_square_containing_circle_l2018_201843


namespace ratio_RN_NS_l2018_201831

/-- Square ABCD with side length 10, F is on DC 3 units from D, N is midpoint of AF,
    perpendicular bisector of AF intersects AD at R and BC at S -/
structure SquareConfiguration where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  F : ℝ × ℝ
  N : ℝ × ℝ
  R : ℝ × ℝ
  S : ℝ × ℝ
  h_square : A = (0, 10) ∧ B = (10, 10) ∧ C = (10, 0) ∧ D = (0, 0)
  h_F : F = (3, 0)
  h_N : N = (3/2, 5)
  h_R : R.1 = 57/3 ∧ R.2 = 10
  h_S : S.1 = -43/3 ∧ S.2 = 0

/-- The ratio of RN to NS is 1:1 -/
theorem ratio_RN_NS (cfg : SquareConfiguration) : 
  dist cfg.R cfg.N = dist cfg.N cfg.S :=
by sorry


end ratio_RN_NS_l2018_201831


namespace inequality_proof_l2018_201866

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (1 + a + a * b)) + (b / (1 + b + b * c)) + (c / (1 + c + c * a)) ≤ 1 := by
  sorry

end inequality_proof_l2018_201866


namespace video_library_space_per_hour_l2018_201888

/-- Given a video library with the following properties:
  * Contains 15 days of videos
  * Each day consists of 18 hours of videos
  * The entire library takes up 45,000 megabytes of disk space
  This theorem proves that the disk space required for one hour of video,
  when rounded to the nearest whole number, is 167 megabytes. -/
theorem video_library_space_per_hour :
  ∀ (days hours_per_day total_space : ℕ),
  days = 15 →
  hours_per_day = 18 →
  total_space = 45000 →
  round ((total_space : ℝ) / (days * hours_per_day : ℝ)) = 167 := by
  sorry

end video_library_space_per_hour_l2018_201888


namespace largest_difference_l2018_201892

-- Define the type for our table
def Table := Fin 20 → Fin 20 → Fin 400

-- Define a property that checks if a table is valid
def is_valid_table (t : Table) : Prop :=
  ∀ i j k l, (i ≠ k ∨ j ≠ l) → t i j ≠ t k l

-- Define the property we want to prove
theorem largest_difference (t : Table) (h : is_valid_table t) :
  (∃ i j k, (i = k ∨ j = k) ∧ 
    (t i j : ℕ).succ.pred - (t i k : ℕ).succ.pred ≥ 209) ∧
  ∀ m, m > 209 → 
    ∃ t', is_valid_table t' ∧ 
      ∀ i j k, (i = k ∨ j = k) → 
        (t' i j : ℕ).succ.pred - (t' i k : ℕ).succ.pred < m :=
sorry

end largest_difference_l2018_201892


namespace weight_difference_is_one_black_dog_weight_conditions_are_consistent_l2018_201827

-- Define the weights of the dogs
def brown_weight : ℝ := 4
def black_weight : ℝ := 5  -- This is derived from the solution, but we'll prove it
def white_weight : ℝ := 2 * brown_weight
def grey_weight : ℝ := black_weight - 2

-- Define the average weight
def average_weight : ℝ := 5

-- Define the number of dogs
def num_dogs : ℕ := 4

-- Theorem to prove
theorem weight_difference_is_one :
  black_weight - brown_weight = 1 :=
by
  -- The proof would go here
  sorry

-- Additional theorem to prove the black dog's weight
theorem black_dog_weight :
  black_weight = 5 :=
by
  -- This proof would use the given conditions to show that black_weight must be 5
  sorry

-- Theorem to show that the conditions are consistent
theorem conditions_are_consistent :
  (brown_weight + black_weight + white_weight + grey_weight) / num_dogs = average_weight :=
by
  -- This proof would show that the given weights satisfy the average weight condition
  sorry

end weight_difference_is_one_black_dog_weight_conditions_are_consistent_l2018_201827


namespace orangeade_price_day2_l2018_201890

/-- Represents the price and volume of orangeade on a given day -/
structure OrangeadeDay where
  juice : ℝ
  water : ℝ
  price : ℝ

/-- Calculates the revenue for a given day -/
def revenue (day : OrangeadeDay) : ℝ :=
  (day.juice + day.water) * day.price

theorem orangeade_price_day2 (day1 day2 : OrangeadeDay) :
  day1.juice = day1.water →
  day2.juice = day1.juice →
  day2.water = 2 * day1.water →
  day1.price = 0.9 →
  revenue day1 = revenue day2 →
  day2.price = 0.6 := by
  sorry

end orangeade_price_day2_l2018_201890


namespace three_thousandths_decimal_l2018_201802

theorem three_thousandths_decimal : (3 : ℚ) / 1000 = (0.003 : ℚ) := by
  sorry

end three_thousandths_decimal_l2018_201802


namespace mode_better_representation_l2018_201854

/-- Represents the salary distribution of employees in a company -/
structure SalaryDistribution where
  manager_salary : ℕ
  deputy_manager_salary : ℕ
  employee_salary : ℕ
  manager_count : ℕ
  deputy_manager_count : ℕ
  employee_count : ℕ

/-- Calculates the mean salary -/
def mean_salary (sd : SalaryDistribution) : ℚ :=
  (sd.manager_salary * sd.manager_count +
   sd.deputy_manager_salary * sd.deputy_manager_count +
   sd.employee_salary * sd.employee_count) /
  (sd.manager_count + sd.deputy_manager_count + sd.employee_count)

/-- Finds the mode salary -/
def mode_salary (sd : SalaryDistribution) : ℕ :=
  if sd.employee_count > sd.manager_count ∧ sd.employee_count > sd.deputy_manager_count then
    sd.employee_salary
  else if sd.deputy_manager_count > sd.manager_count then
    sd.deputy_manager_salary
  else
    sd.manager_salary

/-- Represents how well a measure describes the concentration trend -/
def concentration_measure (salary : ℕ) (sd : SalaryDistribution) : ℚ :=
  (sd.manager_count * (if salary = sd.manager_salary then 1 else 0) +
   sd.deputy_manager_count * (if salary = sd.deputy_manager_salary then 1 else 0) +
   sd.employee_count * (if salary = sd.employee_salary then 1 else 0)) /
  (sd.manager_count + sd.deputy_manager_count + sd.employee_count)

/-- Theorem stating that the mode better represents the concentration trend than the mean -/
theorem mode_better_representation (sd : SalaryDistribution)
  (h1 : sd.manager_salary = 12000)
  (h2 : sd.deputy_manager_salary = 8000)
  (h3 : sd.employee_salary = 3000)
  (h4 : sd.manager_count = 1)
  (h5 : sd.deputy_manager_count = 1)
  (h6 : sd.employee_count = 8) :
  concentration_measure (mode_salary sd) sd > concentration_measure (Nat.floor (mean_salary sd)) sd :=
  sorry

end mode_better_representation_l2018_201854


namespace diplomats_conference_l2018_201856

theorem diplomats_conference (D : ℕ) : 
  (20 : ℕ) ≤ D ∧  -- Number of diplomats who spoke Japanese
  (32 : ℕ) ≤ D ∧  -- Number of diplomats who did not speak Russian
  (D - (20 + (D - 32) - (D / 10 : ℕ)) : ℤ) = (D / 5 : ℕ) ∧  -- 20% spoke neither Japanese nor Russian
  (D / 10 : ℕ) ≤ 20  -- 10% spoke both Japanese and Russian (this must be ≤ 20)
  → D = 40 := by
sorry

end diplomats_conference_l2018_201856


namespace parabola_focus_is_correct_l2018_201828

/-- The focus of a parabola given by y = ax^2 + bx + c -/
def parabola_focus (a b c : ℝ) : ℝ × ℝ := sorry

/-- The equation of the parabola -/
def parabola_equation (x : ℝ) : ℝ := -3 * x^2 - 6 * x

theorem parabola_focus_is_correct :
  parabola_focus (-3) (-6) 0 = (-1, 35/12) := by sorry

end parabola_focus_is_correct_l2018_201828


namespace distance_from_point_to_line_l2018_201834

def point : ℝ × ℝ × ℝ := (2, 4, 5)

def line_point : ℝ × ℝ × ℝ := (4, 6, 8)
def line_direction : ℝ × ℝ × ℝ := (1, 1, -1)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ :=
  sorry

theorem distance_from_point_to_line :
  distance_to_line point line_point line_direction = 2 * Real.sqrt 33 / 3 := by
  sorry

end distance_from_point_to_line_l2018_201834


namespace zachary_needs_money_l2018_201829

/-- The additional amount of money Zachary needs to buy football equipment -/
def additional_money_needed (football_price : ℝ) (shorts_price : ℝ) (shoes_price : ℝ) 
  (socks_price : ℝ) (water_bottle_price : ℝ) (eur_to_usd : ℝ) (gbp_to_usd : ℝ) 
  (jpy_to_usd : ℝ) (krw_to_usd : ℝ) (discount_rate : ℝ) (current_money : ℝ) : ℝ :=
  let total_cost := football_price * eur_to_usd + 2 * shorts_price * gbp_to_usd + 
    shoes_price + 4 * socks_price * jpy_to_usd + water_bottle_price * krw_to_usd
  let discounted_cost := total_cost * (1 - discount_rate)
  discounted_cost - current_money

/-- Theorem stating the additional amount Zachary needs -/
theorem zachary_needs_money : 
  additional_money_needed 3.756 2.498 11.856 135.29 7834 1.19 1.38 0.0088 0.00085 0.1 24.042 = 7.127214 := by
  sorry

end zachary_needs_money_l2018_201829


namespace circle_symmetry_l2018_201817

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = -x

-- Define the symmetric point
def symmetric_point (x y x' y' : ℝ) : Prop := x' = -y ∧ y' = -x

-- Theorem statement
theorem circle_symmetry (x y : ℝ) :
  (∃ (x' y' : ℝ), symmetric_point x y x' y' ∧ 
   symmetry_line x y ∧ 
   original_circle x' y') →
  x^2 + (y + 1)^2 = 1 :=
sorry

end circle_symmetry_l2018_201817


namespace largest_six_digit_divisible_by_five_l2018_201841

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

theorem largest_six_digit_divisible_by_five :
  ∃ (n : ℕ), is_six_digit n ∧ n % 5 = 0 ∧ ∀ (m : ℕ), is_six_digit m ∧ m % 5 = 0 → m ≤ n :=
by sorry

end largest_six_digit_divisible_by_five_l2018_201841


namespace sticker_pages_calculation_l2018_201893

/-- Given a total number of stickers and the number of stickers per page,
    calculate the number of pages. -/
def calculate_pages (total_stickers : ℕ) (stickers_per_page : ℕ) : ℕ :=
  total_stickers / stickers_per_page

/-- Theorem stating that with 220 total stickers and 10 stickers per page,
    the number of pages is 22. -/
theorem sticker_pages_calculation :
  calculate_pages 220 10 = 22 := by
  sorry

end sticker_pages_calculation_l2018_201893


namespace lesser_fraction_l2018_201875

theorem lesser_fraction (x y : ℚ) 
  (sum_eq : x + y = 13 / 14)
  (prod_eq : x * y = 1 / 8) :
  min x y = (13 - Real.sqrt 113) / 28 := by
  sorry

end lesser_fraction_l2018_201875


namespace thursday_steps_l2018_201804

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The target average steps per day -/
def target_average : ℕ := 9000

/-- Steps walked on Sunday -/
def sunday_steps : ℕ := 9400

/-- Steps walked on Monday -/
def monday_steps : ℕ := 9100

/-- Steps walked on Tuesday -/
def tuesday_steps : ℕ := 8300

/-- Steps walked on Wednesday -/
def wednesday_steps : ℕ := 9200

/-- Average steps for Friday and Saturday -/
def friday_saturday_average : ℕ := 9050

/-- Theorem: Given the conditions, Toby must have walked 8900 steps on Thursday to meet his weekly goal -/
theorem thursday_steps : 
  (days_in_week * target_average) - 
  (sunday_steps + monday_steps + tuesday_steps + wednesday_steps + 2 * friday_saturday_average) = 8900 := by
  sorry

end thursday_steps_l2018_201804


namespace line_intercept_l2018_201822

/-- A line passing through two points (x₁, y₁) and (x₂, y₂) -/
structure Line where
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ

/-- The x-intercept of a line is the x-coordinate where the line crosses the x-axis -/
def x_intercept (l : Line) : ℝ := sorry

/-- Theorem: The line passing through (7, -3) and (3, 1) intersects the x-axis at (4, 0) -/
theorem line_intercept : 
  let l : Line := { x₁ := 7, y₁ := -3, x₂ := 3, y₂ := 1 }
  x_intercept l = 4 := by sorry

end line_intercept_l2018_201822


namespace total_milk_count_l2018_201826

theorem total_milk_count (chocolate : ℕ) (strawberry : ℕ) (regular : ℕ)
  (h1 : chocolate = 2)
  (h2 : strawberry = 15)
  (h3 : regular = 3) :
  chocolate + strawberry + regular = 20 := by
  sorry

end total_milk_count_l2018_201826


namespace M_equals_P_l2018_201800

-- Define set M
def M : Set ℝ := {x | ∃ a : ℝ, x = a^2 + 1}

-- Define set P
def P : Set ℝ := {y | ∃ b : ℝ, y = b^2 - 4*b + 5}

-- Theorem stating that M equals P
theorem M_equals_P : M = P := by sorry

end M_equals_P_l2018_201800


namespace james_total_spent_l2018_201835

def milk_price : ℚ := 3
def bananas_price : ℚ := 2
def bread_price : ℚ := 3/2
def cereal_price : ℚ := 4

def milk_tax_rate : ℚ := 1/5
def bananas_tax_rate : ℚ := 3/20
def bread_tax_rate : ℚ := 1/10
def cereal_tax_rate : ℚ := 1/4

def total_spent : ℚ := milk_price * (1 + milk_tax_rate) + 
                       bananas_price * (1 + bananas_tax_rate) + 
                       bread_price * (1 + bread_tax_rate) + 
                       cereal_price * (1 + cereal_tax_rate)

theorem james_total_spent : total_spent = 251/20 := by
  sorry

end james_total_spent_l2018_201835


namespace zoey_reading_schedule_l2018_201865

def days_to_read (n : ℕ) : ℕ := n

def total_days (n : ℕ) : ℕ := n * (n + 1) / 2

def day_of_week (start_day : ℕ) (days_passed : ℕ) : ℕ :=
  (start_day + days_passed) % 7

theorem zoey_reading_schedule (num_books : ℕ) (start_day : ℕ) 
  (h1 : num_books = 20)
  (h2 : start_day = 5) -- Friday is represented as 5 (0 is Sunday)
  : day_of_week start_day (total_days num_books) = start_day := by
  sorry

#check zoey_reading_schedule

end zoey_reading_schedule_l2018_201865


namespace zero_exponent_rule_l2018_201899

theorem zero_exponent_rule (a : ℝ) (h : a ≠ 0) : a ^ 0 = 1 := by
  sorry

end zero_exponent_rule_l2018_201899


namespace correct_regression_equation_l2018_201818

-- Define the sample means
def x_mean : ℝ := 2
def y_mean : ℝ := 1.5

-- Define the linear regression equation
def linear_regression (x : ℝ) : ℝ := -2 * x + 5.5

-- State the theorem
theorem correct_regression_equation :
  -- Condition: x and y are negatively correlated
  (∃ k : ℝ, k < 0 ∧ ∀ x y : ℝ, y = k * x + linear_regression x_mean - k * x_mean) →
  -- The linear regression equation passes through the point (x_mean, y_mean)
  linear_regression x_mean = y_mean := by
  sorry

end correct_regression_equation_l2018_201818


namespace monotonicity_depends_on_a_l2018_201877

/-- The function f(x) = x³ + ax² + 1 where a is a real number -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + 1

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*a*x

/-- Theorem stating that the monotonicity of f depends on the value of a -/
theorem monotonicity_depends_on_a :
  ∀ a : ℝ, ∃ x y : ℝ, x < y ∧
    ((f_derivative a x > 0 ∧ f_derivative a y < 0) ∨
     (f_derivative a x < 0 ∧ f_derivative a y > 0) ∨
     (∀ z : ℝ, f_derivative a z ≥ 0) ∨
     (∀ z : ℝ, f_derivative a z ≤ 0)) :=
by sorry

end monotonicity_depends_on_a_l2018_201877


namespace both_games_count_l2018_201860

/-- The number of people who play both kabadi and kho kho -/
def both_games : ℕ := sorry

/-- The total number of players -/
def total_players : ℕ := 45

/-- The number of people who play kabadi (including those who play both) -/
def kabadi_players : ℕ := 10

/-- The number of people who play only kho kho -/
def only_kho_kho : ℕ := 35

theorem both_games_count : both_games = 10 := by sorry

end both_games_count_l2018_201860


namespace sixteen_point_sphere_half_circumscribed_sphere_l2018_201816

/-- A tetrahedron with its associated spheres -/
structure Tetrahedron where
  /-- The radius of the circumscribed sphere -/
  R : ℝ
  /-- The radius of the sixteen-point sphere -/
  r : ℝ

/-- Theorem: There exists a tetrahedron for which the radius of its sixteen-point sphere 
    is equal to half the radius of its circumscribed sphere -/
theorem sixteen_point_sphere_half_circumscribed_sphere : 
  ∃ (t : Tetrahedron), t.r = t.R / 2 := by
  sorry


end sixteen_point_sphere_half_circumscribed_sphere_l2018_201816


namespace divisible_by_five_l2018_201862

theorem divisible_by_five (n : ℕ) : 
  (∃ B : ℕ, B < 10 ∧ n = 5270 + B) → (n % 5 = 0 ↔ B = 0 ∨ B = 5) :=
by
  sorry

end divisible_by_five_l2018_201862


namespace cos_four_theta_value_l2018_201882

theorem cos_four_theta_value (θ : ℝ) (h : ∑' n, (Real.cos θ)^(2*n) = 8) :
  Real.cos (4 * θ) = 1/8 := by
  sorry

end cos_four_theta_value_l2018_201882


namespace sparkling_juice_bottles_l2018_201870

def total_guests : ℕ := 120
def champagne_percentage : ℚ := 60 / 100
def wine_percentage : ℚ := 30 / 100
def juice_percentage : ℚ := 10 / 100

def champagne_glasses_per_guest : ℕ := 2
def wine_glasses_per_guest : ℕ := 1
def juice_glasses_per_guest : ℕ := 1

def champagne_servings_per_bottle : ℕ := 6
def wine_servings_per_bottle : ℕ := 5
def juice_servings_per_bottle : ℕ := 4

theorem sparkling_juice_bottles (
  total_guests : ℕ)
  (juice_percentage : ℚ)
  (juice_glasses_per_guest : ℕ)
  (juice_servings_per_bottle : ℕ)
  (h1 : total_guests = 120)
  (h2 : juice_percentage = 10 / 100)
  (h3 : juice_glasses_per_guest = 1)
  (h4 : juice_servings_per_bottle = 4)
  : ℕ := by
  sorry

end sparkling_juice_bottles_l2018_201870


namespace triple_tangent_identity_l2018_201861

theorem triple_tangent_identity (x y z : ℝ) 
  (hx : |x| ≠ 1 / Real.sqrt 3) 
  (hy : |y| ≠ 1 / Real.sqrt 3) 
  (hz : |z| ≠ 1 / Real.sqrt 3) 
  (h_sum : x + y + z = x * y * z) : 
  (3 * x - x^3) / (1 - 3 * x^2) + (3 * y - y^3) / (1 - 3 * y^2) + (3 * z - z^3) / (1 - 3 * z^2) = 
  (3 * x - x^3) / (1 - 3 * x^2) * (3 * y - y^3) / (1 - 3 * y^2) * (3 * z - z^3) / (1 - 3 * z^2) := by
  sorry

end triple_tangent_identity_l2018_201861


namespace range_of_a_l2018_201836

-- Define the sets A, B, and C
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 3) - 1 / Real.sqrt (7 - x)}
def B : Set ℝ := {y | ∃ x, y = -x^2 + 2*x + 8}
def C (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}

-- Define the theorem
theorem range_of_a (a : ℝ) : A ∪ C a = C a → a ≥ 7 ∨ a + 1 < 3 := by
  sorry

end range_of_a_l2018_201836


namespace polynomial_expansion_l2018_201848

theorem polynomial_expansion (x : ℝ) :
  (7 * x - 3) * (2 * x^3 + 5 * x^2 - 4) = 14 * x^4 + 29 * x^3 - 15 * x^2 - 28 * x + 12 := by
  sorry

end polynomial_expansion_l2018_201848


namespace worker_completion_time_l2018_201891

/-- Given workers A and B, where A can complete a job in 15 days, works for 5 days,
    and B finishes the remaining work in 18 days, prove that B can complete the
    entire job alone in 27 days. -/
theorem worker_completion_time
  (total_days_A : ℕ)
  (work_days_A : ℕ)
  (remaining_days_B : ℕ)
  (h1 : total_days_A = 15)
  (h2 : work_days_A = 5)
  (h3 : remaining_days_B = 18) :
  (total_days_A * remaining_days_B) / (total_days_A - work_days_A) = 27 := by
  sorry

end worker_completion_time_l2018_201891


namespace abc_inequality_l2018_201885

noncomputable def e : ℝ := Real.exp 1

theorem abc_inequality (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1)
  (eqa : a^2 - 2 * Real.log a + 1 = e)
  (eqb : b^2 - 2 * Real.log b + 2 = e^2)
  (eqc : c^2 - 2 * Real.log c + 3 = e^3) : 
  c < b ∧ b < a := by
  sorry

end abc_inequality_l2018_201885


namespace committee_formation_l2018_201815

theorem committee_formation (n m : ℕ) (hn : n = 8) (hm : m = 4) :
  Nat.choose n m = 70 := by
  sorry

end committee_formation_l2018_201815


namespace class_average_mark_l2018_201847

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) (remaining_students : ℕ)
  (excluded_avg : ℚ) (remaining_avg : ℚ) :
  total_students = 25 →
  excluded_students = 5 →
  remaining_students = 20 →
  excluded_avg = 40 →
  remaining_avg = 90 →
  (total_students : ℚ) * ((excluded_students : ℚ) * excluded_avg + 
    (remaining_students : ℚ) * remaining_avg) / total_students = 80 := by
  sorry

end class_average_mark_l2018_201847


namespace no_solution_iff_n_eq_zero_l2018_201895

/-- The system of equations has no solution if and only if n = 0 -/
theorem no_solution_iff_n_eq_zero (n : ℝ) :
  (∃ (x y z : ℝ), 2*n*x + y = 2 ∧ 3*n*y + z = 3 ∧ x + 2*n*z = 2) ↔ n ≠ 0 :=
sorry

end no_solution_iff_n_eq_zero_l2018_201895


namespace not_perfect_power_probability_l2018_201872

/-- A function that determines if a number is a perfect power --/
def isPerfectPower (n : ℕ) : Prop :=
  ∃ (x y : ℕ), y > 1 ∧ x^y = n

/-- The count of numbers from 1 to 200 that are perfect powers --/
def perfectPowerCount : ℕ := 22

/-- The probability of selecting a number that is not a perfect power --/
def probabilityNotPerfectPower : ℚ := 89 / 100

theorem not_perfect_power_probability :
  (200 - perfectPowerCount : ℚ) / 200 = probabilityNotPerfectPower :=
sorry

end not_perfect_power_probability_l2018_201872


namespace bird_feet_count_l2018_201807

theorem bird_feet_count (num_birds : ℕ) (feet_per_bird : ℕ) (h1 : num_birds = 46) (h2 : feet_per_bird = 2) :
  num_birds * feet_per_bird = 92 := by
  sorry

end bird_feet_count_l2018_201807


namespace unique_triple_solution_l2018_201887

theorem unique_triple_solution : 
  ∃! (a b c : ℝ), 
    a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ 
    a^2 + b^2 + c^2 = 3 ∧ 
    (a + b + c) * (a^2*b + b^2*c + c^2*a) = 9 := by
  sorry

end unique_triple_solution_l2018_201887


namespace smallest_three_digit_non_divisor_l2018_201869

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_three_digit_non_divisor : 
  ∀ n : ℕ, is_three_digit n → (n - 1) ∣ factorial n → n ≥ 1004 :=
by sorry

end smallest_three_digit_non_divisor_l2018_201869


namespace initial_amount_calculation_l2018_201825

theorem initial_amount_calculation (deposit : ℚ) (initial : ℚ) : 
  deposit = 750 → 
  deposit = initial * (20 / 100) * (25 / 100) * (30 / 100) → 
  initial = 50000 := by
sorry

end initial_amount_calculation_l2018_201825


namespace log_equation_solution_l2018_201874

theorem log_equation_solution : 
  ∃! x : ℝ, (1 : ℝ) + Real.log x = Real.log (1 + x) :=
by
  use (1 : ℝ) / 9
  sorry

end log_equation_solution_l2018_201874


namespace square_difference_l2018_201842

theorem square_difference (x y : ℝ) 
  (h1 : (x + y)^2 = 81) 
  (h2 : x * y = 18) : 
  (x - y)^2 = 9 := by
  sorry

end square_difference_l2018_201842


namespace parallel_vectors_m_value_l2018_201833

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.2 = v1.2 * v2.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (m, -1)
  are_parallel a b → m = -2 := by
  sorry

end parallel_vectors_m_value_l2018_201833


namespace tan_product_eighths_of_pi_l2018_201876

theorem tan_product_eighths_of_pi : 
  Real.tan (π / 8) * Real.tan (3 * π / 8) * Real.tan (5 * π / 8) * Real.tan (7 * π / 8) = 1 := by
  sorry

end tan_product_eighths_of_pi_l2018_201876


namespace ellipse_properties_l2018_201824

/-- Properties of the ellipse y²/25 + x²/16 = 1 -/
theorem ellipse_properties :
  let ellipse := (fun (x y : ℝ) => y^2 / 25 + x^2 / 16 = 1)
  ∃ (a b c : ℝ),
    -- Major and minor axis lengths
    a = 5 ∧ b = 4 ∧
    -- Vertices
    ellipse (-4) 0 ∧ ellipse 4 0 ∧ ellipse 0 5 ∧ ellipse 0 (-5) ∧
    -- Foci
    c = 3 ∧
    -- Eccentricity
    c / a = 3 / 5 := by
  sorry

end ellipse_properties_l2018_201824


namespace initial_student_count_l2018_201850

/-- Proves that the initial number of students is 29 given the conditions of the problem -/
theorem initial_student_count (initial_avg : ℝ) (new_avg : ℝ) (new_student_weight : ℝ) :
  initial_avg = 28 →
  new_avg = 27.5 →
  new_student_weight = 13 →
  ∃ n : ℕ, n = 29 ∧
    (n : ℝ) * initial_avg + new_student_weight = (n + 1 : ℝ) * new_avg :=
by
  sorry


end initial_student_count_l2018_201850


namespace weight_of_replaced_person_l2018_201810

/-- Given a group of 10 persons, prove that if replacing one person with a new person
    weighing 110 kg increases the average weight by 4 kg, then the weight of the
    replaced person is 70 kg. -/
theorem weight_of_replaced_person
  (initial_avg : ℝ)
  (h1 : initial_avg > 0)
  (h2 : (10 * (initial_avg + 4) - 10 * initial_avg) = (110 - 70)) :
  70 = 110 - (10 * (initial_avg + 4) - 10 * initial_avg) :=
by sorry

end weight_of_replaced_person_l2018_201810


namespace christines_speed_l2018_201896

theorem christines_speed (distance : ℝ) (time : ℝ) (speed : ℝ) : 
  distance = 20 ∧ time = 5 ∧ speed = distance / time → speed = 4 :=
by sorry

end christines_speed_l2018_201896


namespace max_value_on_interval_l2018_201857

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5

-- State the theorem
theorem max_value_on_interval :
  ∃ (M : ℝ), M = 5 ∧ 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → f x ≤ M) ∧
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ f x = M) :=
by sorry

end max_value_on_interval_l2018_201857


namespace average_cost_theorem_l2018_201808

/-- The average cost per marker in cents, rounded to the nearest whole number -/
def average_cost_per_marker (num_markers : ℕ) (package_cost : ℚ) (handling_fee : ℚ) : ℕ :=
  let total_cost_cents := (package_cost + handling_fee) * 100
  let avg_cost_cents := total_cost_cents / num_markers
  (avg_cost_cents + 1/2).floor.toNat

theorem average_cost_theorem (num_markers : ℕ) (package_cost : ℚ) (handling_fee : ℚ)
  (h1 : num_markers = 150)
  (h2 : package_cost = 24.75)
  (h3 : handling_fee = 5.25) :
  average_cost_per_marker num_markers package_cost handling_fee = 20 := by
  sorry

end average_cost_theorem_l2018_201808


namespace probability_of_valid_pair_l2018_201873

def is_odd (n : ℤ) : Prop := n % 2 ≠ 0

def is_divisible_by_5 (n : ℤ) : Prop := n % 5 = 0

def valid_pair (a b : ℤ) : Prop :=
  1 ≤ a ∧ a ≤ 20 ∧
  1 ≤ b ∧ b ≤ 20 ∧
  a ≠ b ∧
  is_odd (a * b) ∧
  is_divisible_by_5 (a + b)

def total_pairs : ℕ := 190

def valid_pairs : ℕ := 18

theorem probability_of_valid_pair :
  (valid_pairs : ℚ) / total_pairs = 9 / 95 :=
sorry

end probability_of_valid_pair_l2018_201873


namespace three_intersections_iff_l2018_201838

/-- The number of intersection points between the curves x^2 + y^2 = a^2 and y = x^2 + a -/
def num_intersections (a : ℝ) : ℕ :=
  sorry

/-- Theorem stating the condition for exactly 3 intersection points -/
theorem three_intersections_iff (a : ℝ) : 
  num_intersections a = 3 ↔ a < -1/2 := by
  sorry

end three_intersections_iff_l2018_201838


namespace grid_toothpicks_l2018_201871

/-- Calculates the number of toothpicks required for a rectangular grid. -/
def toothpicks_in_grid (height width : ℕ) : ℕ :=
  (height + 1) * width + (width + 1) * height

/-- Theorem: A rectangular grid with height 15 and width 12 requires 387 toothpicks. -/
theorem grid_toothpicks : toothpicks_in_grid 15 12 = 387 := by
  sorry

#eval toothpicks_in_grid 15 12

end grid_toothpicks_l2018_201871


namespace equation_equivalence_l2018_201821

theorem equation_equivalence (x : ℝ) : 1 - (x + 3) / 3 = x / 2 ↔ 6 - 2 * x - 6 = 3 * x := by
  sorry

end equation_equivalence_l2018_201821


namespace smallest_y_l2018_201852

theorem smallest_y (x y : ℕ+) (h1 : x.val - y.val = 8) 
  (h2 : Nat.gcd ((x.val^3 + y.val^3) / (x.val + y.val)) (x.val * y.val) = 16) :
  ∀ z : ℕ+, z.val < y.val → 
    Nat.gcd ((z.val^3 + (z.val + 8)^3) / (z.val + (z.val + 8))) (z.val * (z.val + 8)) ≠ 16 :=
by sorry

#check smallest_y

end smallest_y_l2018_201852


namespace library_shelf_count_l2018_201801

theorem library_shelf_count (notebooks : ℕ) (pen_difference : ℕ) (pencil_difference : ℕ) 
  (h1 : notebooks = 40)
  (h2 : pen_difference = 80)
  (h3 : pencil_difference = 45) :
  notebooks + (notebooks + pen_difference) + (notebooks + pencil_difference) = 245 :=
by
  sorry

end library_shelf_count_l2018_201801


namespace sufficient_not_necessary_l2018_201864

theorem sufficient_not_necessary : 
  (∃ x : ℝ, x^2 - 3*x + 2 = 0 ∧ x ≠ 1) ∧ 
  (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) := by
  sorry

end sufficient_not_necessary_l2018_201864


namespace smallest_sum_mn_smallest_sum_value_unique_smallest_sum_l2018_201859

theorem smallest_sum_mn (m n : ℕ) (h : 3 * n^3 = 5 * m^2) : 
  ∀ (a b : ℕ), (3 * a^3 = 5 * b^2) → m + n ≤ a + b :=
by
  sorry

theorem smallest_sum_value : 
  ∃ (m n : ℕ), (3 * n^3 = 5 * m^2) ∧ (m + n = 60) :=
by
  sorry

theorem unique_smallest_sum : 
  ∀ (m n : ℕ), (3 * n^3 = 5 * m^2) → m + n ≥ 60 :=
by
  sorry

end smallest_sum_mn_smallest_sum_value_unique_smallest_sum_l2018_201859


namespace rectangle_ratio_l2018_201897

/-- Given a rectangle with length 40 cm, if reducing the length by 5 cm and
    increasing the width by 5 cm results in an area increase of 75 cm²,
    then the ratio of the original length to the original width is 2:1. -/
theorem rectangle_ratio (w : ℝ) : 
  (40 - 5) * (w + 5) = 40 * w + 75 → 40 / w = 2 := by
  sorry

end rectangle_ratio_l2018_201897


namespace frequency_below_70kg_l2018_201813

def total_students : ℕ := 50

def weight_groups : List (ℝ × ℝ) := [(40, 50), (50, 60), (60, 70), (70, 80), (80, 90)]

def frequencies : List ℕ := [6, 8, 15, 18, 3]

def students_below_70kg : ℕ := 29

theorem frequency_below_70kg :
  (students_below_70kg : ℝ) / total_students = 0.58 :=
sorry

end frequency_below_70kg_l2018_201813


namespace pipe_length_difference_l2018_201819

theorem pipe_length_difference (total_length shorter_length : ℕ) : 
  total_length = 68 → 
  shorter_length = 28 → 
  shorter_length < total_length - shorter_length →
  total_length - shorter_length - shorter_length = 12 :=
by sorry

end pipe_length_difference_l2018_201819


namespace absolute_value_inequality_l2018_201886

theorem absolute_value_inequality (x : ℝ) : 
  |((3 * x - 2) / (2 * x - 3))| > 3 ↔ 11/9 < x ∧ x < 7/3 :=
by sorry

end absolute_value_inequality_l2018_201886


namespace max_value_vx_minus_yz_l2018_201881

def A : Set Int := {-3, -2, -1, 0, 1, 2, 3}

theorem max_value_vx_minus_yz :
  ∃ (v x y z : Int), v ∈ A ∧ x ∈ A ∧ y ∈ A ∧ z ∈ A ∧
    v * x - y * z = 6 ∧
    ∀ (v' x' y' z' : Int), v' ∈ A → x' ∈ A → y' ∈ A → z' ∈ A →
      v' * x' - y' * z' ≤ 6 :=
sorry

end max_value_vx_minus_yz_l2018_201881


namespace center_is_eight_l2018_201868

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Nat

/-- Check if two positions are adjacent in the grid --/
def isAdjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- Check if a grid is valid according to the problem conditions --/
def isValidGrid (g : Grid) : Prop :=
  (∀ n : Fin 9, ∃! p : Fin 3 × Fin 3, g p.1 p.2 = n.val + 1) ∧
  (∀ n : Fin 8, ∃ p1 p2 : Fin 3 × Fin 3, 
    g p1.1 p1.2 = n.val + 1 ∧ 
    g p2.1 p2.2 = n.val + 2 ∧ 
    isAdjacent p1 p2) ∧
  (g 0 0 + g 0 2 + g 2 0 + g 2 2 = 24)

theorem center_is_eight (g : Grid) (h : isValidGrid g) : 
  g 1 1 = 8 := by sorry

end center_is_eight_l2018_201868


namespace rectangle_area_theorem_l2018_201889

theorem rectangle_area_theorem (L W : ℝ) (h : 2 * L * (3 * W) = 1800) : L * W = 300 := by
  sorry

end rectangle_area_theorem_l2018_201889


namespace max_value_of_function_l2018_201884

open Real

theorem max_value_of_function (x : ℝ) : 
  ∃ (M : ℝ), M = 2 - sqrt 3 ∧ ∀ y : ℝ, sin (2 * y) - 2 * sqrt 3 * sin y ^ 2 ≤ M :=
sorry

end max_value_of_function_l2018_201884


namespace sum_of_digits_of_10_pow_95_minus_107_l2018_201837

-- Define the number
def n : ℕ := 95

-- Define the function to calculate the number
def f (n : ℕ) : ℤ := 10^n - 107

-- Define the function to calculate the sum of digits
def sum_of_digits (z : ℤ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_10_pow_95_minus_107 :
  sum_of_digits (f n) = 849 := by sorry

end sum_of_digits_of_10_pow_95_minus_107_l2018_201837


namespace proposition_and_variants_l2018_201851

theorem proposition_and_variants (x y : ℝ) :
  (∀ x y, xy = 0 → x = 0 ∨ y = 0) ∧
  (∀ x y, x = 0 ∨ y = 0 → xy = 0) ∧
  (∀ x y, xy ≠ 0 → x ≠ 0 ∧ y ≠ 0) ∧
  (∀ x y, x ≠ 0 ∧ y ≠ 0 → xy ≠ 0) :=
by sorry

end proposition_and_variants_l2018_201851


namespace vector_magnitude_l2018_201845

/-- Given two vectors a and b in ℝ², prove that |2a + b| = 2√21 -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  a = (3, -4) → 
  ‖b‖ = 2 → 
  a • b = -5 → 
  ‖2 • a + b‖ = 2 * Real.sqrt 21 :=
by sorry

end vector_magnitude_l2018_201845


namespace defective_product_probability_l2018_201883

theorem defective_product_probability
  (p_first : ℝ)
  (p_second : ℝ)
  (h1 : p_first = 0.65)
  (h2 : p_second = 0.3)
  (h3 : p_first + p_second + p_defective = 1)
  (h4 : p_first ≥ 0 ∧ p_second ≥ 0 ∧ p_defective ≥ 0)
  : p_defective = 0.05 := by
  sorry

end defective_product_probability_l2018_201883


namespace problem_statement_l2018_201811

theorem problem_statement (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : 2 * (Real.log x / Real.log y) + 2 * (Real.log y / Real.log x) = 8) 
  (h4 : x * y = 256) : (x + y) / 2 = 16 := by
  sorry

end problem_statement_l2018_201811


namespace equation_solution_l2018_201863

theorem equation_solution (x : ℝ) : x ≠ 1 →
  ((3 * x + 6) / (x^2 + 5*x - 6) = (3 - x) / (x - 1)) ↔ (x = -4 ∨ x = -2) :=
by sorry

end equation_solution_l2018_201863


namespace train_meeting_time_l2018_201806

/-- Calculates the time for two trains to meet given their speeds, lengths, and the platform length --/
theorem train_meeting_time (length_A length_B platform_length : ℝ)
                           (speed_A speed_B : ℝ)
                           (h1 : length_A = 120)
                           (h2 : length_B = 150)
                           (h3 : platform_length = 180)
                           (h4 : speed_A = 90 * 1000 / 3600)
                           (h5 : speed_B = 72 * 1000 / 3600) :
  (length_A + length_B + platform_length) / (speed_A + speed_B) = 10 := by
  sorry

#check train_meeting_time

end train_meeting_time_l2018_201806


namespace bookkeeper_arrangements_l2018_201840

/-- The number of distinct arrangements of letters in a word with the given letter distribution -/
def distinctArrangements (totalLetters : Nat) (repeatedLetters : Nat) (repeatCount : Nat) : Nat :=
  Nat.factorial totalLetters / (Nat.factorial repeatCount ^ repeatedLetters)

/-- Theorem stating the number of distinct arrangements for the specific word structure -/
theorem bookkeeper_arrangements :
  distinctArrangements 10 4 2 = 226800 := by
  sorry

end bookkeeper_arrangements_l2018_201840


namespace perpendicular_line_through_point_l2018_201812

/-- Given a line L1 with equation 3x - y = 7 and a point P (2, -3),
    this theorem proves that the line L2 with equation y = -1/3x - 7/3
    is perpendicular to L1 and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 3 * x - y = 7
  let L2 : ℝ → ℝ → Prop := λ x y => y = -1/3 * x - 7/3
  let P : ℝ × ℝ := (2, -3)
  (∀ x y, L1 x y → L2 x y → (3 * (-1/3) = -1)) ∧ 
  (L2 P.1 P.2) := by
  sorry

end perpendicular_line_through_point_l2018_201812


namespace michael_anna_ratio_is_500_251_l2018_201898

/-- Sum of odd integers from 1 to n -/
def sumOddIntegers (n : ℕ) : ℕ := n^2

/-- Sum of integers from 1 to n -/
def sumIntegers (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The ratio of Michael's sum to Anna's sum -/
def michaelAnnaRatio : ℚ :=
  (sumOddIntegers 500 : ℚ) / (sumIntegers 500 : ℚ)

theorem michael_anna_ratio_is_500_251 :
  michaelAnnaRatio = 500 / 251 := by
  sorry

end michael_anna_ratio_is_500_251_l2018_201898


namespace complex_sixth_power_l2018_201820

theorem complex_sixth_power (z : ℂ) : z = (-Real.sqrt 3 - I) / 2 → z^6 = -1 := by
  sorry

end complex_sixth_power_l2018_201820
