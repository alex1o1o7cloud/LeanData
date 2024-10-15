import Mathlib

namespace NUMINAMATH_CALUDE_food_weight_l3563_356334

/-- Given a bowl with 14 pieces of food, prove that each piece weighs 0.76 kg -/
theorem food_weight (total_weight : ℝ) (empty_bowl_weight : ℝ) (num_pieces : ℕ) :
  total_weight = 11.14 ∧ 
  empty_bowl_weight = 0.5 ∧ 
  num_pieces = 14 →
  (total_weight - empty_bowl_weight) / num_pieces = 0.76 := by
  sorry

end NUMINAMATH_CALUDE_food_weight_l3563_356334


namespace NUMINAMATH_CALUDE_decreasing_functions_a_range_l3563_356319

/-- Given two functions f and g, prove that if they are both decreasing on [1,2],
    then the parameter a is in the interval (0,1]. -/
theorem decreasing_functions_a_range 
  (f g : ℝ → ℝ) 
  (hf : f = fun x ↦ -x^2 + 2*a*x) 
  (hg : g = fun x ↦ a / (x + 1)) 
  (hf_decreasing : ∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x < y → f x > f y) 
  (hg_decreasing : ∀ x y, x ∈ Set.Icc 1 2 → y ∈ Set.Icc 1 2 → x < y → g x > g y) 
  : a ∈ Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_functions_a_range_l3563_356319


namespace NUMINAMATH_CALUDE_harold_marbles_l3563_356370

/-- Given that Harold had 100 marbles, shared them evenly among 5 friends,
    and each friend received 16 marbles, prove that Harold kept 20 marbles. -/
theorem harold_marbles :
  ∀ (total_marbles friends_count marbles_per_friend marbles_kept : ℕ),
    total_marbles = 100 →
    friends_count = 5 →
    marbles_per_friend = 16 →
    marbles_kept + (friends_count * marbles_per_friend) = total_marbles →
    marbles_kept = 20 :=
by sorry

end NUMINAMATH_CALUDE_harold_marbles_l3563_356370


namespace NUMINAMATH_CALUDE_interest_rate_problem_l3563_356371

/-- Calculates simple interest given principal, rate, and time -/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem interest_rate_problem (principal : ℝ) (time : ℝ) (rate : ℝ) 
    (h1 : principal = 15000)
    (h2 : time = 2)
    (h3 : simpleInterest principal rate time = simpleInterest principal 0.12 time + 900) :
  rate = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_problem_l3563_356371


namespace NUMINAMATH_CALUDE_car_efficiency_before_modification_l3563_356312

/-- Represents the fuel efficiency of a car before and after modification -/
structure CarEfficiency where
  pre_mod : ℝ  -- Fuel efficiency before modification (miles per gallon)
  post_mod : ℝ  -- Fuel efficiency after modification (miles per gallon)
  fuel_capacity : ℝ  -- Fuel tank capacity in gallons
  extra_distance : ℝ  -- Additional distance traveled after modification (miles)

/-- Theorem stating the car's fuel efficiency before modification -/
theorem car_efficiency_before_modification (car : CarEfficiency)
  (h1 : car.post_mod = car.pre_mod / 0.8)
  (h2 : car.fuel_capacity = 15)
  (h3 : car.fuel_capacity * car.post_mod = car.fuel_capacity * car.pre_mod + car.extra_distance)
  (h4 : car.extra_distance = 105) :
  car.pre_mod = 28 := by
  sorry

end NUMINAMATH_CALUDE_car_efficiency_before_modification_l3563_356312


namespace NUMINAMATH_CALUDE_shop_monthly_rent_l3563_356376

/-- Calculates the monthly rent of a shop given its dimensions and annual rent per square foot. -/
def monthly_rent (length width annual_rent_per_sqft : ℕ) : ℕ :=
  let area := length * width
  let annual_rent := area * annual_rent_per_sqft
  annual_rent / 12

/-- Theorem stating that for a shop with given dimensions and annual rent per square foot,
    the monthly rent is 3600. -/
theorem shop_monthly_rent :
  monthly_rent 20 15 144 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_shop_monthly_rent_l3563_356376


namespace NUMINAMATH_CALUDE_no_eight_term_ap_with_r_ap_l3563_356338

/-- r(n) is the odd positive integer whose binary representation is the reverse of n's binary representation -/
def r (n : ℕ) : ℕ :=
  sorry

/-- Theorem: There does not exist a strictly increasing eight-term arithmetic progression of odd positive integers such that their r values form an arithmetic progression -/
theorem no_eight_term_ap_with_r_ap :
  ¬ ∃ (a : ℕ → ℕ) (d : ℕ),
    (∀ i, i ∈ Finset.range 8 → Odd (a i)) ∧
    (∀ i j, i < j → i ∈ Finset.range 8 → j ∈ Finset.range 8 → a i < a j) ∧
    (∀ i, i ∈ Finset.range 7 → a (i + 1) - a i = d) ∧
    (∃ d' : ℕ, ∀ i, i ∈ Finset.range 7 → r (a (i + 1)) - r (a i) = d') :=
  sorry

end NUMINAMATH_CALUDE_no_eight_term_ap_with_r_ap_l3563_356338


namespace NUMINAMATH_CALUDE_two_numbers_with_given_means_l3563_356341

theorem two_numbers_with_given_means (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (Real.sqrt (a * b) = Real.sqrt 5) → 
  (2 / (1/a + 1/b) = 5/3) → 
  ((a = 1 ∧ b = 5) ∨ (a = 5 ∧ b = 1)) := by
sorry

end NUMINAMATH_CALUDE_two_numbers_with_given_means_l3563_356341


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l3563_356382

-- Define an isosceles triangle with one angle of 80 degrees
structure IsoscelesTriangle :=
  (angle1 : ℝ)
  (angle2 : ℝ)
  (angle3 : ℝ)
  (is_isosceles : (angle1 = angle2) ∨ (angle1 = angle3) ∨ (angle2 = angle3))
  (has_80_degree : angle1 = 80 ∨ angle2 = 80 ∨ angle3 = 80)
  (sum_180 : angle1 + angle2 + angle3 = 180)

-- Theorem statement
theorem isosceles_triangle_base_angle (t : IsoscelesTriangle) :
  (t.angle1 = 50 ∨ t.angle1 = 80) ∨
  (t.angle2 = 50 ∨ t.angle2 = 80) ∨
  (t.angle3 = 50 ∨ t.angle3 = 80) :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_angle_l3563_356382


namespace NUMINAMATH_CALUDE_arctanSum_implies_powerSum_l3563_356342

theorem arctanSum_implies_powerSum (x y z : ℝ) (n : ℕ) 
  (h1 : x + y + z = 1) 
  (h2 : Real.arctan x + Real.arctan y + Real.arctan z = π / 4) 
  (h3 : n > 0) : 
  x^(2*n+1) + y^(2*n+1) + z^(2*n+1) = 1 := by
sorry

end NUMINAMATH_CALUDE_arctanSum_implies_powerSum_l3563_356342


namespace NUMINAMATH_CALUDE_polyhedron_volume_l3563_356380

-- Define the polygons
def right_triangle (a b c : ℝ) := a^2 + b^2 = c^2
def rectangle (l w : ℝ) := l > 0 ∧ w > 0
def equilateral_triangle (s : ℝ) := s > 0

-- Define the polyhedron
def polyhedron (A E F : ℝ → ℝ → ℝ → Prop) 
               (B C D : ℝ → ℝ → Prop) 
               (G : ℝ → Prop) := 
  A 1 2 (Real.sqrt 5) ∧ 
  E 1 2 (Real.sqrt 5) ∧ 
  F 1 2 (Real.sqrt 5) ∧ 
  B 1 2 ∧ 
  C 2 3 ∧ 
  D 1 3 ∧ 
  G (Real.sqrt 5)

-- State the theorem
theorem polyhedron_volume 
  (A E F : ℝ → ℝ → ℝ → Prop) 
  (B C D : ℝ → ℝ → Prop) 
  (G : ℝ → Prop) : 
  polyhedron right_triangle right_triangle right_triangle 
              rectangle rectangle rectangle 
              equilateral_triangle → 
  ∃ v : ℝ, v = 6 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_volume_l3563_356380


namespace NUMINAMATH_CALUDE_B_when_a_is_3_range_of_a_when_A_equals_B_l3563_356324

-- Define the set B
def B (a : ℝ) : Set ℝ := {x | (a - 2) * x^2 + 2 * (a - 2) * x - 3 < 0}

-- Theorem 1: When a = 3, B = (-3, 1)
theorem B_when_a_is_3 : B 3 = Set.Ioo (-3) 1 := by sorry

-- Theorem 2: When A = B = ℝ, a ∈ (-1, 2]
theorem range_of_a_when_A_equals_B :
  (∀ x, x ∈ B a) ↔ a ∈ Set.Ioc (-1) 2 := by sorry

end NUMINAMATH_CALUDE_B_when_a_is_3_range_of_a_when_A_equals_B_l3563_356324


namespace NUMINAMATH_CALUDE_lines_parallel_if_planes_parallel_and_coplanar_l3563_356357

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the necessary relations
variable (subset : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (coplanar : Line → Line → Prop)

-- State the theorem
theorem lines_parallel_if_planes_parallel_and_coplanar
  (m n : Line) (α β : Plane)
  (h_diff_lines : m ≠ n)
  (h_diff_planes : α ≠ β)
  (h_m_in_α : subset m α)
  (h_n_in_β : subset n β)
  (h_planes_parallel : parallel α β)
  (h_coplanar : coplanar m n) :
  line_parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_if_planes_parallel_and_coplanar_l3563_356357


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3563_356392

/-- The line mx-y+3+m=0 passes through the point (-1, 3) for any real number m -/
theorem line_passes_through_fixed_point (m : ℝ) : m * (-1) - 3 + 3 + m = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3563_356392


namespace NUMINAMATH_CALUDE_next_simultaneous_ringing_l3563_356375

def town_hall_period : ℕ := 18
def university_tower_period : ℕ := 24
def fire_station_period : ℕ := 30

def minutes_in_hour : ℕ := 60

theorem next_simultaneous_ringing :
  ∃ (n : ℕ), n > 0 ∧ 
    n % town_hall_period = 0 ∧
    n % university_tower_period = 0 ∧
    n % fire_station_period = 0 ∧
    n / minutes_in_hour = 6 :=
sorry

end NUMINAMATH_CALUDE_next_simultaneous_ringing_l3563_356375


namespace NUMINAMATH_CALUDE_mean_of_remaining_numbers_l3563_356310

theorem mean_of_remaining_numbers (a b c d : ℝ) :
  (a + b + c + d + 105) / 5 = 92 →
  (a + b + c + d) / 4 = 88.75 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_remaining_numbers_l3563_356310


namespace NUMINAMATH_CALUDE_koolaid_percentage_is_four_percent_l3563_356322

def koolaid_percentage (initial_powder initial_water evaporation water_multiplier : ℕ) : ℚ :=
  let remaining_water := initial_water - evaporation
  let final_water := remaining_water * water_multiplier
  let total_liquid := initial_powder + final_water
  (initial_powder : ℚ) / total_liquid * 100

theorem koolaid_percentage_is_four_percent :
  koolaid_percentage 2 16 4 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_koolaid_percentage_is_four_percent_l3563_356322


namespace NUMINAMATH_CALUDE_pencil_arrangement_theorem_l3563_356317

def total_pencils (total_rows : ℕ) (pattern_length : ℕ) (pencils_second_row : ℕ) : ℕ :=
  let pattern_repeats := total_rows / pattern_length
  let pencils_fifth_row := pencils_second_row + pencils_second_row / 2
  let pencil_rows_per_pattern := 2
  pattern_repeats * pencil_rows_per_pattern * (pencils_second_row + pencils_fifth_row)

theorem pencil_arrangement_theorem :
  total_pencils 30 6 76 = 950 := by
  sorry

end NUMINAMATH_CALUDE_pencil_arrangement_theorem_l3563_356317


namespace NUMINAMATH_CALUDE_function_inequality_implies_k_range_l3563_356350

theorem function_inequality_implies_k_range (k : ℝ) :
  (∀ x : ℝ, (k * x + 1 > 0) ∨ (x^2 - 1 > 0)) →
  k ∈ Set.Ioo (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_implies_k_range_l3563_356350


namespace NUMINAMATH_CALUDE_general_ticket_price_is_six_l3563_356344

/-- Represents the ticket sales and pricing scenario -/
structure TicketSale where
  student_price : ℕ
  total_tickets : ℕ
  total_revenue : ℕ
  general_tickets : ℕ

/-- Calculates the price of a general admission ticket -/
def general_price (sale : TicketSale) : ℚ :=
  (sale.total_revenue - sale.student_price * (sale.total_tickets - sale.general_tickets)) / sale.general_tickets

/-- Theorem stating that the general admission ticket price is 6 dollars -/
theorem general_ticket_price_is_six (sale : TicketSale) 
  (h1 : sale.student_price = 4)
  (h2 : sale.total_tickets = 525)
  (h3 : sale.total_revenue = 2876)
  (h4 : sale.general_tickets = 388) :
  general_price sale = 6 := by
  sorry

#eval general_price {
  student_price := 4,
  total_tickets := 525,
  total_revenue := 2876,
  general_tickets := 388
}

end NUMINAMATH_CALUDE_general_ticket_price_is_six_l3563_356344


namespace NUMINAMATH_CALUDE_coin_toss_sequences_coin_toss_theorem_l3563_356394

/-- The number of ways to place n indistinguishable balls into k distinguishable urns -/
def stars_and_bars (n k : ℕ) : ℕ := Nat.choose (n + k - 1) n

/-- The number of different sequences of 20 coin tosses with specific subsequence counts -/
theorem coin_toss_sequences : ℕ := 
  let hh_placements := stars_and_bars 3 4
  let tt_placements := stars_and_bars 7 5
  hh_placements * tt_placements

/-- The main theorem stating the number of valid sequences -/
theorem coin_toss_theorem : coin_toss_sequences = 6600 := by sorry

end NUMINAMATH_CALUDE_coin_toss_sequences_coin_toss_theorem_l3563_356394


namespace NUMINAMATH_CALUDE_matilda_age_is_35_l3563_356373

-- Define the ages as natural numbers
def louis_age : ℕ := 14
def jerica_age : ℕ := 2 * louis_age
def matilda_age : ℕ := jerica_age + 7

-- Theorem statement
theorem matilda_age_is_35 : matilda_age = 35 := by
  sorry

end NUMINAMATH_CALUDE_matilda_age_is_35_l3563_356373


namespace NUMINAMATH_CALUDE_product_prs_is_27_l3563_356364

theorem product_prs_is_27 (p r s : ℕ) 
  (eq1 : 4^p + 4^3 = 272)
  (eq2 : 3^r + 27 = 54)
  (eq3 : 2^(s+2) + 10 = 42) :
  p * r * s = 27 := by
  sorry

end NUMINAMATH_CALUDE_product_prs_is_27_l3563_356364


namespace NUMINAMATH_CALUDE_boxes_with_neither_l3563_356395

theorem boxes_with_neither (total : ℕ) (markers : ℕ) (crayons : ℕ) (both : ℕ)
  (h1 : total = 15)
  (h2 : markers = 10)
  (h3 : crayons = 8)
  (h4 : both = 4) :
  total - (markers + crayons - both) = 1 := by
  sorry

end NUMINAMATH_CALUDE_boxes_with_neither_l3563_356395


namespace NUMINAMATH_CALUDE_investment_calculation_l3563_356396

/-- Given two investors p and q with an investment ratio of 4:5, 
    where q invests 65000, prove that p's investment is 52000. -/
theorem investment_calculation (p q : ℕ) : 
  (p : ℚ) / q = 4 / 5 → q = 65000 → p = 52000 := by
  sorry

end NUMINAMATH_CALUDE_investment_calculation_l3563_356396


namespace NUMINAMATH_CALUDE_canoe_production_sum_l3563_356388

theorem canoe_production_sum : ∀ (a₁ r n : ℕ), 
  a₁ = 5 → r = 3 → n = 4 → 
  a₁ * (r^n - 1) / (r - 1) = 200 := by sorry

end NUMINAMATH_CALUDE_canoe_production_sum_l3563_356388


namespace NUMINAMATH_CALUDE_internet_speed_calculation_l3563_356362

/-- Represents the internet speed calculation problem -/
theorem internet_speed_calculation 
  (file1 : ℝ) 
  (file2 : ℝ) 
  (file3 : ℝ) 
  (download_time : ℝ) 
  (h1 : file1 = 80) 
  (h2 : file2 = 90) 
  (h3 : file3 = 70) 
  (h4 : download_time = 2) :
  (file1 + file2 + file3) / download_time = 120 := by
  sorry

#check internet_speed_calculation

end NUMINAMATH_CALUDE_internet_speed_calculation_l3563_356362


namespace NUMINAMATH_CALUDE_running_competition_sample_l3563_356333

/-- Given a school with 2000 students, where 3/5 participate in a running competition
    with grade ratios of 2:3:5, and a sample of 200 students is taken, 
    the number of 2nd grade students in the running competition sample is 36. -/
theorem running_competition_sample (total_students : ℕ) (sample_size : ℕ) 
  (running_ratio : ℚ) (grade_ratios : Fin 3 → ℚ) :
  total_students = 2000 →
  sample_size = 200 →
  running_ratio = 3/5 →
  grade_ratios 0 = 2/10 ∧ grade_ratios 1 = 3/10 ∧ grade_ratios 2 = 5/10 →
  ↑sample_size * running_ratio * grade_ratios 1 = 36 := by
  sorry

#check running_competition_sample

end NUMINAMATH_CALUDE_running_competition_sample_l3563_356333


namespace NUMINAMATH_CALUDE_book_distribution_l3563_356385

theorem book_distribution (total : ℕ) (books_A books_B : ℕ) : 
  total = 282 → 
  4 * books_A = 3 * total → 
  9 * books_B = 5 * total → 
  books_A + books_B = total → 
  books_A = 120 ∧ books_B = 162 := by
  sorry

end NUMINAMATH_CALUDE_book_distribution_l3563_356385


namespace NUMINAMATH_CALUDE_fraction_equality_l3563_356398

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 21)
  (h2 : p / n = 7)
  (h3 : p / q = 1 / 9) :
  m / q = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_fraction_equality_l3563_356398


namespace NUMINAMATH_CALUDE_harvard_attendance_l3563_356315

theorem harvard_attendance 
  (total_applicants : ℕ) 
  (acceptance_rate : ℚ) 
  (attendance_rate : ℚ) :
  total_applicants = 20000 →
  acceptance_rate = 5 / 100 →
  attendance_rate = 90 / 100 →
  (total_applicants : ℚ) * acceptance_rate * attendance_rate = 900 :=
by sorry

end NUMINAMATH_CALUDE_harvard_attendance_l3563_356315


namespace NUMINAMATH_CALUDE_crayon_cost_proof_l3563_356328

/-- The cost of one pack of crayons -/
def pack_cost : ℚ := 5/2

/-- The number of packs Michael initially has -/
def initial_packs : ℕ := 4

/-- The number of packs Michael buys -/
def bought_packs : ℕ := 2

/-- The total value of all packs after purchase -/
def total_value : ℚ := 15

theorem crayon_cost_proof :
  (initial_packs + bought_packs : ℚ) * pack_cost = total_value := by
  sorry

end NUMINAMATH_CALUDE_crayon_cost_proof_l3563_356328


namespace NUMINAMATH_CALUDE_recycling_points_l3563_356327

/-- Calculates the points earned from recycling paper --/
def points_earned (pounds_per_point : ℕ) (chloe_pounds : ℕ) (friends_pounds : ℕ) : ℕ :=
  (chloe_pounds + friends_pounds) / pounds_per_point

/-- Theorem: Given the recycling conditions, the total points earned is 5 --/
theorem recycling_points : points_earned 6 28 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_recycling_points_l3563_356327


namespace NUMINAMATH_CALUDE_commodity_profit_optimization_l3563_356347

/-- Represents the monthly sales quantity as a function of price -/
def sales_quantity (x : ℝ) : ℝ := -30 * x + 960

/-- Represents the monthly profit as a function of price -/
def monthly_profit (x : ℝ) : ℝ := (sales_quantity x) * (x - 10)

theorem commodity_profit_optimization (cost_price : ℝ) 
  (h1 : cost_price = 10)
  (h2 : sales_quantity 20 = 360)
  (h3 : sales_quantity 30 = 60) :
  (∃ (optimal_price max_profit : ℝ),
    (∀ x, monthly_profit x ≤ monthly_profit optimal_price) ∧
    monthly_profit optimal_price = max_profit ∧
    optimal_price = 21 ∧
    max_profit = 3630) := by
  sorry

end NUMINAMATH_CALUDE_commodity_profit_optimization_l3563_356347


namespace NUMINAMATH_CALUDE_prob_different_plants_l3563_356337

/-- The number of distinct plant options available -/
def num_options : ℕ := 4

/-- The probability of two employees choosing different plants -/
def prob_different_choices : ℚ := 3/4

/-- Theorem stating that the probability of two employees choosing different plants
    from four options is 3/4 -/
theorem prob_different_plants :
  (num_options : ℚ)^2 - num_options = (prob_different_choices * num_options^2 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_prob_different_plants_l3563_356337


namespace NUMINAMATH_CALUDE_stock_sale_cash_realization_l3563_356339

/-- The cash realized on selling a stock, given the brokerage rate and total amount including brokerage -/
theorem stock_sale_cash_realization (brokerage_rate : ℚ) (total_with_brokerage : ℚ) :
  brokerage_rate = 1 / 400 →
  total_with_brokerage = 106 →
  ∃ cash_realized : ℚ, cash_realized + cash_realized * brokerage_rate = total_with_brokerage ∧
                    cash_realized = 42400 / 401 := by
  sorry

end NUMINAMATH_CALUDE_stock_sale_cash_realization_l3563_356339


namespace NUMINAMATH_CALUDE_locus_of_midpoints_is_single_point_l3563_356336

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ
  h : r > 0

/-- A point P inside the circle on its diameter -/
structure InteriorPointOnDiameter (K : Circle) where
  P : ℝ × ℝ
  h₁ : dist P K.O < K.r
  h₂ : ∃ (t : ℝ), P = (K.O.1 + t * K.r, K.O.2) ∨ P = (K.O.1, K.O.2 + t * K.r)

/-- The midpoint of a chord passing through P -/
def midpoint_of_chord (K : Circle) (P : InteriorPointOnDiameter K) (θ : ℝ) : ℝ × ℝ :=
  P.P

/-- The theorem stating that the locus of midpoints is a single point -/
theorem locus_of_midpoints_is_single_point (K : Circle) (P : InteriorPointOnDiameter K) :
  ∀ θ : ℝ, midpoint_of_chord K P θ = P.P :=
sorry

end NUMINAMATH_CALUDE_locus_of_midpoints_is_single_point_l3563_356336


namespace NUMINAMATH_CALUDE_complex_real_condition_l3563_356318

theorem complex_real_condition (m : ℝ) : 
  let z : ℂ := (m - 2 : ℂ) + (m^2 - 3*m + 2 : ℂ) * Complex.I
  (z ≠ 0 ∧ z.im = 0) → m = 1 :=
by sorry

end NUMINAMATH_CALUDE_complex_real_condition_l3563_356318


namespace NUMINAMATH_CALUDE_equality_or_power_relation_l3563_356349

theorem equality_or_power_relation (x y : ℝ) (hx : x > 1) (hy : y > 1) (h : x^y = y^x) :
  x = y ∨ ∃ m : ℝ, m > 0 ∧ m ≠ 1 ∧ x = m^(1/(m-1)) ∧ y = m^(m/(m-1)) := by
  sorry

end NUMINAMATH_CALUDE_equality_or_power_relation_l3563_356349


namespace NUMINAMATH_CALUDE_display_rows_l3563_356355

/-- The number of cans in the nth row of the display -/
def cans_in_row (n : ℕ) : ℕ := 2 + 3 * (n - 1)

/-- The total number of cans in the first n rows of the display -/
def total_cans (n : ℕ) : ℕ := n * (cans_in_row 1 + cans_in_row n) / 2

/-- The number of rows in the display -/
def num_rows : ℕ := 12

theorem display_rows :
  total_cans num_rows = 225 ∧
  cans_in_row 1 = 2 ∧
  ∀ n : ℕ, n > 1 → cans_in_row n = cans_in_row (n - 1) + 3 :=
sorry

end NUMINAMATH_CALUDE_display_rows_l3563_356355


namespace NUMINAMATH_CALUDE_five_integers_sum_20_product_420_l3563_356309

theorem five_integers_sum_20_product_420 :
  ∃! (a b c d e : ℕ+),
    a + b + c + d + e = 20 ∧
    a * b * c * d * e = 420 :=
by sorry

end NUMINAMATH_CALUDE_five_integers_sum_20_product_420_l3563_356309


namespace NUMINAMATH_CALUDE_gcd_digits_bound_l3563_356314

theorem gcd_digits_bound (a b : ℕ) : 
  10000 ≤ a ∧ a < 100000 ∧ 
  10000 ≤ b ∧ b < 100000 ∧ 
  100000000 ≤ Nat.lcm a b ∧ Nat.lcm a b < 1000000000 →
  Nat.gcd a b < 100 :=
by sorry

end NUMINAMATH_CALUDE_gcd_digits_bound_l3563_356314


namespace NUMINAMATH_CALUDE_imaginary_number_condition_l3563_356358

theorem imaginary_number_condition (a : ℝ) : 
  let z : ℂ := (a - 2*I) / (2 + I)
  (∃ b : ℝ, z = b * I ∧ b ≠ 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_number_condition_l3563_356358


namespace NUMINAMATH_CALUDE_rational_equation_solution_l3563_356379

theorem rational_equation_solution (A B : ℚ) :
  (∀ x : ℚ, x ≠ 3 ∧ x ≠ 5 → (B * x - 13) / (x^2 - 8*x + 15) = A / (x - 3) + 4 / (x - 5)) →
  A + B = 22 / 5 := by
sorry

end NUMINAMATH_CALUDE_rational_equation_solution_l3563_356379


namespace NUMINAMATH_CALUDE_inverse_function_point_l3563_356387

/-- Given a function f(x) = 2^x + m, prove that if its inverse passes through (3,1), then m = 1 -/
theorem inverse_function_point (m : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = 2^x + m) ∧ (∃ g : ℝ → ℝ, Function.LeftInverse g f ∧ g 3 = 1)) → 
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_function_point_l3563_356387


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_l3563_356301

theorem inequality_and_minimum_value 
  (a b m n : ℝ) (x : ℝ) 
  (ha : a > 0) (hb : b > 0) (hm : m > 0) (hn : n > 0)
  (hx : 0 < x ∧ x < 1/2) : 
  (m^2 / a + n^2 / b ≥ (m + n)^2 / (a + b)) ∧
  (2 / x + 9 / (1 - 2*x) ≥ 25) ∧
  (∀ y, 0 < y ∧ y < 1/2 → 2 / y + 9 / (1 - 2*y) ≥ 2 / x + 9 / (1 - 2*x)) ∧
  (x = 1/5) := by
sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_l3563_356301


namespace NUMINAMATH_CALUDE_min_value_collinear_points_l3563_356378

/-- Given points A(3,-1), B(x,y), and C(0,1) are collinear, and x > 0, y > 0, 
    the minimum value of (3/x + 2/y) is 8 -/
theorem min_value_collinear_points (x y : ℝ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : (y + 1) / (x - 3) = 2 / (-3)) : 
  (∀ a b : ℝ, a > 0 → b > 0 → (y + 1) / (x - 3) = 2 / (-3) → 3 / x + 2 / y ≤ 3 / a + 2 / b) → 
  3 / x + 2 / y = 8 := by
sorry

end NUMINAMATH_CALUDE_min_value_collinear_points_l3563_356378


namespace NUMINAMATH_CALUDE_rice_profit_l3563_356356

/-- Calculates the profit from selling a sack of rice -/
def calculate_profit (weight : ℝ) (cost : ℝ) (price_per_kg : ℝ) : ℝ :=
  weight * price_per_kg - cost

/-- Theorem: The profit from selling a 50kg sack of rice that costs $50 at $1.20 per kg is $10 -/
theorem rice_profit : calculate_profit 50 50 1.20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_rice_profit_l3563_356356


namespace NUMINAMATH_CALUDE_cos_double_angle_special_l3563_356377

/-- Given an angle θ formed by the positive x-axis and a line passing through
    the origin and the point (-3, 4), prove that cos(2θ) = -7/25 -/
theorem cos_double_angle_special (θ : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos θ = -3 ∧ r * Real.sin θ = 4) → 
  Real.cos (2 * θ) = -7/25 := by
sorry

end NUMINAMATH_CALUDE_cos_double_angle_special_l3563_356377


namespace NUMINAMATH_CALUDE_root_relation_l3563_356389

theorem root_relation (a b c : ℂ) (p q u : ℂ) : 
  (a^3 + 2*a^2 + 5*a - 8 = 0) → 
  (b^3 + 2*b^2 + 5*b - 8 = 0) → 
  (c^3 + 2*c^2 + 5*c - 8 = 0) → 
  ((a+b)^3 + p*(a+b)^2 + q*(a+b) + u = 0) → 
  ((b+c)^3 + p*(b+c)^2 + q*(b+c) + u = 0) → 
  ((c+a)^3 + p*(c+a)^2 + q*(c+a) + u = 0) → 
  u = 18 := by
sorry

end NUMINAMATH_CALUDE_root_relation_l3563_356389


namespace NUMINAMATH_CALUDE_a_divisibility_a_specific_cases_l3563_356300

def a (n : ℕ) : ℕ := (10^(3^n) - 1) / 9

theorem a_divisibility (n : ℕ) (h : n > 0) :
  (3^n ∣ a n) ∧ ¬(3^(n+1) ∣ a n) :=
sorry

theorem a_specific_cases :
  (3 ∣ a 1) ∧ ¬(9 ∣ a 1) ∧
  (9 ∣ a 2) ∧ ¬(27 ∣ a 2) ∧
  (27 ∣ a 3) ∧ ¬(81 ∣ a 3) :=
sorry

end NUMINAMATH_CALUDE_a_divisibility_a_specific_cases_l3563_356300


namespace NUMINAMATH_CALUDE_enrollment_system_correct_l3563_356359

/-- Represents the enrollment plan and actual enrollment of a school -/
structure EnrollmentPlan where
  planned_total : ℕ
  actual_total : ℕ
  boys_exceed_percent : ℚ
  girls_exceed_percent : ℚ

/-- The correct system of equations for the enrollment plan -/
def correct_system (plan : EnrollmentPlan) (x y : ℚ) : Prop :=
  x + y = plan.planned_total ∧
  (1 + plan.boys_exceed_percent) * x + (1 + plan.girls_exceed_percent) * y = plan.actual_total

/-- Theorem stating that the given system of equations is correct for the enrollment plan -/
theorem enrollment_system_correct (plan : EnrollmentPlan)
  (h1 : plan.planned_total = 1000)
  (h2 : plan.actual_total = 1240)
  (h3 : plan.boys_exceed_percent = 1/5)
  (h4 : plan.girls_exceed_percent = 3/10) :
  ∀ x y : ℚ, correct_system plan x y ↔ 
    (x + y = 1000 ∧ 6/5 * x + 13/10 * y = 1240) :=
by sorry

end NUMINAMATH_CALUDE_enrollment_system_correct_l3563_356359


namespace NUMINAMATH_CALUDE_book_purchase_change_l3563_356367

/-- The change received when buying two books with given prices and paying with a fixed amount. -/
theorem book_purchase_change (book1_price book2_price payment : ℝ) 
  (h1 : book1_price = 5.5)
  (h2 : book2_price = 6.5)
  (h3 : payment = 20) : 
  payment - (book1_price + book2_price) = 8 := by
sorry

end NUMINAMATH_CALUDE_book_purchase_change_l3563_356367


namespace NUMINAMATH_CALUDE_complex_simplification_l3563_356330

theorem complex_simplification (i : ℂ) (h : i^2 = -1) :
  7 * (4 - 2*i) + 4*i * (7 - 2*i) = 36 + 14*i := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l3563_356330


namespace NUMINAMATH_CALUDE_problem_solution_l3563_356311

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a * x - a - Real.log x

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * g a x

theorem problem_solution :
  (∃ a : ℝ, ∀ x > 0, g a x ≥ 0) ∧
  (∃ a : ℝ, ∃ x₀ : ℝ, 0 < x₀ ∧ x₀ < 1 ∧
    (∀ x > 0, (deriv (f a)) x₀ = 0 ∧ f a x ≤ f a x₀)) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3563_356311


namespace NUMINAMATH_CALUDE_lighter_cost_difference_l3563_356390

/-- Calculates the cost of buying lighters at the gas station with a "buy 4 get 1 free" offer -/
def gas_station_cost (price_per_lighter : ℚ) (num_lighters : ℕ) : ℚ :=
  let sets := (num_lighters + 4) / 5
  let lighters_to_pay := sets * 4
  lighters_to_pay * price_per_lighter

/-- Calculates the cost of buying lighters on Amazon including tax and shipping -/
def amazon_cost (price_per_pack : ℚ) (lighters_per_pack : ℕ) (num_lighters : ℕ) 
                (tax_rate : ℚ) (shipping_cost : ℚ) : ℚ :=
  let packs_needed := (num_lighters + lighters_per_pack - 1) / lighters_per_pack
  let subtotal := packs_needed * price_per_pack
  let tax := subtotal * tax_rate
  subtotal + tax + shipping_cost

theorem lighter_cost_difference : 
  gas_station_cost (175/100) 24 - amazon_cost 5 12 24 (5/100) (7/2) = 1925/100 := by
  sorry

end NUMINAMATH_CALUDE_lighter_cost_difference_l3563_356390


namespace NUMINAMATH_CALUDE_sin_sum_identity_l3563_356368

theorem sin_sum_identity (x : ℝ) (h : Real.sin (2 * x + π / 5) = Real.sqrt 3 / 3) :
  Real.sin (4 * π / 5 - 2 * x) + Real.sin (3 * π / 10 - 2 * x)^2 = (2 + Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_identity_l3563_356368


namespace NUMINAMATH_CALUDE_calculation_proof_l3563_356384

theorem calculation_proof :
  ((-3)^2 - 60 / 10 * (1 / 10) - |(-2)|) = 32 / 5 ∧
  (-4 / 5 * (9 / 4) + (-1 / 4) * (4 / 5) - (3 / 2) * (-4 / 5)) = -4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l3563_356384


namespace NUMINAMATH_CALUDE_max_min_x_plus_y_l3563_356303

noncomputable def f (x y : ℝ) : Prop :=
  1 - Real.sqrt (x - 1) = Real.sqrt (y - 1) ∧ x ≥ 1 ∧ y ≥ 1

theorem max_min_x_plus_y :
  ∀ x y : ℝ, f x y →
    (∀ a b : ℝ, f a b → x + y ≤ a + b) ∧
    (∃ a b : ℝ, f a b ∧ a + b = 3) ∧
    (∀ a b : ℝ, f a b → a + b ≥ 5/2) ∧
    (∃ a b : ℝ, f a b ∧ a + b = 5/2) :=
by sorry

end NUMINAMATH_CALUDE_max_min_x_plus_y_l3563_356303


namespace NUMINAMATH_CALUDE_whole_number_between_values_l3563_356381

theorem whole_number_between_values (N : ℤ) : 
  (6.75 < (N : ℚ) / 4 ∧ (N : ℚ) / 4 < 7.25) → N = 28 := by
  sorry

end NUMINAMATH_CALUDE_whole_number_between_values_l3563_356381


namespace NUMINAMATH_CALUDE_units_digit_base8_product_l3563_356332

/-- The units digit of a number in base 8 -/
def unitsDigitBase8 (n : ℕ) : ℕ := n % 8

/-- The product of 348 and 76 -/
def product : ℕ := 348 * 76

theorem units_digit_base8_product : unitsDigitBase8 product = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_base8_product_l3563_356332


namespace NUMINAMATH_CALUDE_jessica_calculation_l3563_356397

theorem jessica_calculation (y : ℝ) : (y - 8) / 4 = 22 → (y - 4) / 8 = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_jessica_calculation_l3563_356397


namespace NUMINAMATH_CALUDE_quadratic_equation_1_l3563_356360

theorem quadratic_equation_1 : 
  ∀ x : ℝ, x^2 - 2*x + 1 = 0 → x = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_1_l3563_356360


namespace NUMINAMATH_CALUDE_smallest_sum_of_consecutive_multiples_l3563_356369

theorem smallest_sum_of_consecutive_multiples : ∃ (a b c : ℕ),
  (b = a + 1) ∧
  (c = a + 2) ∧
  (a % 9 = 0) ∧
  (b % 8 = 0) ∧
  (c % 7 = 0) ∧
  (a + b + c = 1488) ∧
  (∀ (x y z : ℕ), (y = x + 1) ∧ (z = x + 2) ∧ (x % 9 = 0) ∧ (y % 8 = 0) ∧ (z % 7 = 0) → (x + y + z ≥ 1488)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_consecutive_multiples_l3563_356369


namespace NUMINAMATH_CALUDE_square_difference_formula_l3563_356329

-- Define the expressions
def expr_A (x : ℝ) := (x + 1) * (x - 1)
def expr_B (x : ℝ) := (-x + 1) * (-x - 1)
def expr_C (x : ℝ) := (x + 1) * (-x + 1)
def expr_D (x : ℝ) := (x + 1) * (1 + x)

-- Define a predicate for expressions that can be written as a difference of squares
def is_diff_of_squares (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ → ℝ), ∀ x, f x = (a x)^2 - (b x)^2

-- State the theorem
theorem square_difference_formula :
  (is_diff_of_squares expr_A) ∧
  (is_diff_of_squares expr_B) ∧
  (is_diff_of_squares expr_C) ∧
  ¬(is_diff_of_squares expr_D) := by
  sorry

end NUMINAMATH_CALUDE_square_difference_formula_l3563_356329


namespace NUMINAMATH_CALUDE_bus_passengers_l3563_356393

theorem bus_passengers (initial : ℕ) (difference : ℕ) (final : ℕ) : 
  initial = 38 → difference = 9 → final = initial - difference → final = 29 := by sorry

end NUMINAMATH_CALUDE_bus_passengers_l3563_356393


namespace NUMINAMATH_CALUDE_train_speed_l3563_356306

def train_length : Real := 250.00000000000003
def crossing_time : Real := 15

theorem train_speed : 
  (train_length / 1000) / (crossing_time / 3600) = 60 := by sorry

end NUMINAMATH_CALUDE_train_speed_l3563_356306


namespace NUMINAMATH_CALUDE_sumata_family_driving_l3563_356391

/-- The Sumata family's driving problem -/
theorem sumata_family_driving (days : ℝ) (miles_per_day : ℝ) 
  (h1 : days = 5.0)
  (h2 : miles_per_day = 50) :
  days * miles_per_day = 250 := by
  sorry

end NUMINAMATH_CALUDE_sumata_family_driving_l3563_356391


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3563_356304

/-- A geometric sequence with specified terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_fifth_term 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_3 : a 3 = -4) 
  (h_7 : a 7 = -16) : 
  a 5 = -8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l3563_356304


namespace NUMINAMATH_CALUDE_intersection_complement_l3563_356383

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3, 4}

theorem intersection_complement : M ∩ (U \ N) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_complement_l3563_356383


namespace NUMINAMATH_CALUDE_complex_parts_of_z_l3563_356353

theorem complex_parts_of_z : ∃ (z : ℂ), z = 2 - 3 * I ∧ z.re = 2 ∧ z.im = -3 := by
  sorry

end NUMINAMATH_CALUDE_complex_parts_of_z_l3563_356353


namespace NUMINAMATH_CALUDE_cos_75_degrees_l3563_356374

theorem cos_75_degrees : 
  let cos_75 := Real.cos (75 * π / 180)
  let cos_60 := Real.cos (60 * π / 180)
  let sin_60 := Real.sin (60 * π / 180)
  let cos_15 := Real.cos (15 * π / 180)
  let sin_15 := Real.sin (15 * π / 180)
  cos_60 = 1/2 ∧ sin_60 = Real.sqrt 3 / 2 →
  cos_75 = cos_60 * cos_15 - sin_60 * sin_15 →
  cos_75 = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
sorry

end NUMINAMATH_CALUDE_cos_75_degrees_l3563_356374


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3563_356331

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  (∃ d : ℝ, 3 * a 1 + d = (1/2) * a 3 ∧ (1/2) * a 3 + d = 2 * a 2) →  -- Arithmetic sequence condition
  (∃ q : ℝ, ∀ n, a (n+1) = q * a n) →  -- Geometric sequence definition
  (a 8 + a 9) / (a 6 + a 7) = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3563_356331


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3563_356345

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 2 * p^2 + 5 * p + 15 = 0) →
  (3 * q^3 - 2 * q^2 + 5 * q + 15 = 0) →
  (3 * r^3 - 2 * r^2 + 5 * r + 15 = 0) →
  p^2 + q^2 + r^2 = -26/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3563_356345


namespace NUMINAMATH_CALUDE_rabbit_walk_prob_l3563_356343

/-- A random walk on a rectangular grid. -/
structure RandomWalk where
  width : ℕ
  height : ℕ
  start_x : ℕ
  start_y : ℕ

/-- The probability of ending on the top or bottom edge for a given random walk. -/
noncomputable def prob_top_bottom (walk : RandomWalk) : ℚ :=
  sorry

/-- The specific random walk described in the problem. -/
def rabbit_walk : RandomWalk :=
  { width := 6
    height := 5
    start_x := 2
    start_y := 3 }

/-- The main theorem stating the probability for the specific random walk. -/
theorem rabbit_walk_prob : prob_top_bottom rabbit_walk = 17 / 24 := by
  sorry

end NUMINAMATH_CALUDE_rabbit_walk_prob_l3563_356343


namespace NUMINAMATH_CALUDE_circle_symmetry_l3563_356307

/-- The equation of a circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Symmetry of a circle with respect to the origin -/
def symmetric_to_origin (c1 c2 : Circle) : Prop :=
  c1.center.1 = -c2.center.1 ∧ c1.center.2 = -c2.center.2 ∧ c1.radius = c2.radius

theorem circle_symmetry (c : Circle) :
  symmetric_to_origin c ⟨(-2, 1), 1⟩ →
  c = ⟨(2, -1), 1⟩ := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetry_l3563_356307


namespace NUMINAMATH_CALUDE_range_of_m_l3563_356325

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | 2*m - 1 ≤ x ∧ x ≤ m + 3}

-- Theorem statement
theorem range_of_m (m : ℝ) : B m ⊆ A → m < -4 ∨ m > 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3563_356325


namespace NUMINAMATH_CALUDE_equality_preservation_l3563_356308

theorem equality_preservation (x y : ℝ) : x = y → x - 2 = y - 2 := by
  sorry

end NUMINAMATH_CALUDE_equality_preservation_l3563_356308


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l3563_356316

/-- A positive geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_pos : ∀ n, a n > 0)
  (h_cond : a 7 = a 6 + 2 * a 5)
  (h_exist : ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ a m * a n = 16 * (a 1)^2) :
  ∃ m n : ℕ, m > 0 ∧ n > 0 ∧ a m * a n = 16 * (a 1)^2 ∧
    ∀ k l : ℕ, k > 0 → l > 0 → a k * a l = 16 * (a 1)^2 →
      1 / m + 4 / n ≤ 1 / k + 4 / l :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l3563_356316


namespace NUMINAMATH_CALUDE_surface_area_of_rmon_l3563_356366

/-- Right prism with equilateral triangle base -/
structure RightPrism :=
  (height : ℝ)
  (baseSideLength : ℝ)

/-- Point on an edge of the prism -/
structure EdgePoint :=
  (position : ℝ)

/-- The solid RMON created by slicing the prism -/
structure SlicedSolid :=
  (prism : RightPrism)
  (m : EdgePoint)
  (n : EdgePoint)
  (o : EdgePoint)

/-- Calculate the surface area of the sliced solid -/
noncomputable def surfaceArea (solid : SlicedSolid) : ℝ :=
  sorry

/-- Main theorem: The surface area of RMON is 30.62 square units -/
theorem surface_area_of_rmon (solid : SlicedSolid) 
  (h1 : solid.prism.height = 10)
  (h2 : solid.prism.baseSideLength = 10)
  (h3 : solid.m.position = 1/4)
  (h4 : solid.n.position = 1/4)
  (h5 : solid.o.position = 1/4) :
  surfaceArea solid = 30.62 := by
  sorry

end NUMINAMATH_CALUDE_surface_area_of_rmon_l3563_356366


namespace NUMINAMATH_CALUDE_ratio_equality_l3563_356365

theorem ratio_equality (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a / 2 = b / 3) (h5 : b / 3 = c / 5) : (a + b) / (c - a) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_equality_l3563_356365


namespace NUMINAMATH_CALUDE_perpendicular_slope_l3563_356326

/-- The slope of a line perpendicular to a line passing through (2, 3) and (7, 8) is -1 -/
theorem perpendicular_slope : 
  let p1 : ℝ × ℝ := (2, 3)
  let p2 : ℝ × ℝ := (7, 8)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  (-1 : ℝ) * m = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_slope_l3563_356326


namespace NUMINAMATH_CALUDE_sally_tuesday_shirts_l3563_356335

/-- The number of shirts Sally sewed on Monday -/
def monday_shirts : ℕ := 4

/-- The number of shirts Sally sewed on Wednesday -/
def wednesday_shirts : ℕ := 2

/-- The number of buttons required for each shirt -/
def buttons_per_shirt : ℕ := 5

/-- The total number of buttons needed for all shirts -/
def total_buttons : ℕ := 45

/-- The number of shirts Sally sewed on Tuesday -/
def tuesday_shirts : ℕ := 3

theorem sally_tuesday_shirts :
  tuesday_shirts = (total_buttons - (monday_shirts + wednesday_shirts) * buttons_per_shirt) / buttons_per_shirt :=
by sorry

end NUMINAMATH_CALUDE_sally_tuesday_shirts_l3563_356335


namespace NUMINAMATH_CALUDE_simple_interest_principal_calculation_l3563_356340

/-- Simple interest calculation -/
theorem simple_interest_principal_calculation
  (rate : ℝ) (time : ℝ) (interest : ℝ) (principal : ℝ)
  (h_rate : rate = 0.08)
  (h_time : time = 1)
  (h_interest : interest = 800)
  (h_formula : interest = principal * rate * time) :
  principal = 10000 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_principal_calculation_l3563_356340


namespace NUMINAMATH_CALUDE_digging_project_breadth_l3563_356313

/-- The breadth of the first digging project -/
def breadth_project1 : ℝ := 30

/-- The depth of the first digging project in meters -/
def depth_project1 : ℝ := 100

/-- The length of the first digging project in meters -/
def length_project1 : ℝ := 25

/-- The depth of the second digging project in meters -/
def depth_project2 : ℝ := 75

/-- The length of the second digging project in meters -/
def length_project2 : ℝ := 20

/-- The breadth of the second digging project in meters -/
def breadth_project2 : ℝ := 50

/-- The number of days to complete each project -/
def days_to_complete : ℝ := 12

theorem digging_project_breadth :
  depth_project1 * length_project1 * breadth_project1 =
  depth_project2 * length_project2 * breadth_project2 :=
by sorry

end NUMINAMATH_CALUDE_digging_project_breadth_l3563_356313


namespace NUMINAMATH_CALUDE_x_one_value_l3563_356320

theorem x_one_value (x₁ x₂ x₃ : Real) 
  (h1 : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 0.8)
  (h2 : (1-x₁)^2 + (x₁-x₂)^2 + (x₂-x₃)^2 + x₃^2 = 1/3) :
  x₁ = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_x_one_value_l3563_356320


namespace NUMINAMATH_CALUDE_chord_length_unit_circle_l3563_356305

/-- The length of the chord cut by a line on a unit circle -/
theorem chord_length_unit_circle (a b c : ℝ) (h : a ≠ 0 ∨ b ≠ 0) :
  let line := {p : ℝ × ℝ | a * p.1 + b * p.2 + c = 0}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1}
  let d := |c| / Real.sqrt (a^2 + b^2)
  2 * Real.sqrt (1 - d^2) = 8/5 ↔ 
    a = 3 ∧ b = -4 ∧ c = 3 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_unit_circle_l3563_356305


namespace NUMINAMATH_CALUDE_last_two_digits_zero_l3563_356348

theorem last_two_digits_zero (x y : ℕ) : 
  (x^2 + x*y + y^2) % 10 = 0 → (x^2 + x*y + y^2) % 100 = 0 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_zero_l3563_356348


namespace NUMINAMATH_CALUDE_max_delta_ratio_l3563_356321

/-- Represents a contestant's score in a two-day competition -/
structure ContestantScore where
  day1_score : ℕ
  day1_total : ℕ
  day2_score : ℕ
  day2_total : ℕ

/-- Calculate the two-day success ratio -/
def two_day_ratio (score : ContestantScore) : ℚ :=
  (score.day1_score + score.day2_score : ℚ) / (score.day1_total + score.day2_total)

/-- Charlie's score in the competition -/
def charlie : ContestantScore :=
  { day1_score := 210, day1_total := 400, day2_score := 150, day2_total := 200 }

theorem max_delta_ratio :
  ∀ delta : ContestantScore,
    delta.day1_score > 0 ∧ 
    delta.day2_score > 0 ∧
    delta.day1_total + delta.day2_total = 600 ∧
    (delta.day1_score : ℚ) / delta.day1_total < 210 / 400 ∧
    (delta.day2_score : ℚ) / delta.day2_total < 3 / 4 →
    two_day_ratio delta ≤ 349 / 600 :=
  sorry

end NUMINAMATH_CALUDE_max_delta_ratio_l3563_356321


namespace NUMINAMATH_CALUDE_sum_of_squares_unique_l3563_356363

theorem sum_of_squares_unique (x y z : ℕ+) : 
  (x : ℕ) + y + z = 24 → 
  Nat.gcd x y + Nat.gcd y z + Nat.gcd z x = 10 → 
  x^2 + y^2 + z^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_unique_l3563_356363


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3563_356372

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: For a geometric sequence {a_n}, if a_2 * a_4 = 1/2, then a_1 * a_3^2 * a_5 = 1/4 -/
theorem geometric_sequence_property (a : ℕ → ℝ) 
    (h_geo : geometric_sequence a) (h_cond : a 2 * a 4 = 1/2) : 
    a 1 * (a 3)^2 * a 5 = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3563_356372


namespace NUMINAMATH_CALUDE_circumscribed_circle_area_l3563_356323

/-- The area of a circle circumscribed about an equilateral triangle with side length 12 units is 48π square units. -/
theorem circumscribed_circle_area (s : ℝ) (h : s = 12) : 
  let R := s / Real.sqrt 3
  π * R^2 = 48 * π := by sorry

end NUMINAMATH_CALUDE_circumscribed_circle_area_l3563_356323


namespace NUMINAMATH_CALUDE_log_problem_l3563_356399

theorem log_problem (x k : ℝ) : 
  (Real.log 3 / Real.log 9 = x) → 
  (Real.log 81 / Real.log 3 = k * x) → 
  k = 8 := by
sorry

end NUMINAMATH_CALUDE_log_problem_l3563_356399


namespace NUMINAMATH_CALUDE_number_difference_l3563_356302

theorem number_difference (x y : ℝ) (h1 : x + y = 50) (h2 : 3 * y - 4 * x = 10) :
  |y - x| = 10 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l3563_356302


namespace NUMINAMATH_CALUDE_sum_of_three_at_least_fifty_l3563_356354

theorem sum_of_three_at_least_fifty (S : Finset ℕ) (h1 : S.card = 7) 
  (h2 : ∀ x ∈ S, x > 0) (h3 : S.sum id = 100) :
  ∃ T ⊆ S, T.card = 3 ∧ T.sum id ≥ 50 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_three_at_least_fifty_l3563_356354


namespace NUMINAMATH_CALUDE_max_closable_companies_l3563_356346

/-- The number of planets (vertices) in the Intergalactic empire -/
def n : ℕ := 10^2015

/-- The number of travel companies (colors) -/
def m : ℕ := 2015

/-- A function that determines if a graph remains connected after removing k colors -/
def remains_connected (k : ℕ) : Prop :=
  ∀ (removed_colors : Finset (Fin m)),
    removed_colors.card = k →
    ∃ (remaining_graph : SimpleGraph (Fin n)),
      remaining_graph.Connected

/-- The theorem stating the maximum number of companies that can be closed -/
theorem max_closable_companies :
  (∀ k ≤ 1007, remains_connected k) ∧
  ¬(remains_connected 1008) :=
sorry

end NUMINAMATH_CALUDE_max_closable_companies_l3563_356346


namespace NUMINAMATH_CALUDE_integer_sum_problem_l3563_356352

theorem integer_sum_problem (x y : ℤ) : 
  (x = 15 ∨ y = 15) → (4 * x + 3 * y = 150) → (x = 30 ∨ y = 30) := by
sorry

end NUMINAMATH_CALUDE_integer_sum_problem_l3563_356352


namespace NUMINAMATH_CALUDE_color_combination_count_l3563_356361

/-- The number of colors available -/
def total_colors : ℕ := 9

/-- The number of colors to be chosen -/
def colors_to_choose : ℕ := 2

/-- The number of forbidden combinations (red and pink) -/
def forbidden_combinations : ℕ := 1

/-- The number of ways to choose colors, excluding forbidden combinations -/
def valid_combinations : ℕ := (total_colors.choose colors_to_choose) - forbidden_combinations

theorem color_combination_count : valid_combinations = 35 := by
  sorry

end NUMINAMATH_CALUDE_color_combination_count_l3563_356361


namespace NUMINAMATH_CALUDE_perpendicular_lines_n_value_l3563_356386

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A point lies on a line if it satisfies the line's equation -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem perpendicular_lines_n_value (m n : ℝ) (p : ℝ) :
  let l₁ : Line := ⟨m, 4, -2⟩
  let l₂ : Line := ⟨2, -5, n⟩
  let foot : Point := ⟨1, p⟩
  perpendicular (m / -4) (2 / 5) →
  point_on_line foot l₁ →
  point_on_line foot l₂ →
  n = -12 := by
  sorry

#check perpendicular_lines_n_value

end NUMINAMATH_CALUDE_perpendicular_lines_n_value_l3563_356386


namespace NUMINAMATH_CALUDE_circle_area_increase_l3563_356351

theorem circle_area_increase (r : ℝ) (h : r > 0) : 
  let new_radius := 1.5 * r
  let original_area := π * r^2
  let new_area := π * new_radius^2
  (new_area - original_area) / original_area = 1.25 := by
sorry

end NUMINAMATH_CALUDE_circle_area_increase_l3563_356351
