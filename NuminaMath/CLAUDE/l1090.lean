import Mathlib

namespace NUMINAMATH_CALUDE_right_triangle_with_specific_perimeter_l1090_109025

theorem right_triangle_with_specific_perimeter :
  ∃ (b c : ℤ), 
    b = 7 ∧ 
    c = 5 ∧ 
    (b : ℝ)^2 + (b + c : ℝ)^2 = (b + 2*c : ℝ)^2 ∧ 
    b + (b + c) + (b + 2*c) = 36 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_with_specific_perimeter_l1090_109025


namespace NUMINAMATH_CALUDE_second_number_is_72_l1090_109072

theorem second_number_is_72 
  (sum : ℝ) 
  (first : ℝ) 
  (second : ℝ) 
  (third : ℝ) 
  (h1 : sum = 264) 
  (h2 : first = 2 * second) 
  (h3 : third = (1/3) * first) 
  (h4 : first + second + third = sum) : second = 72 := by
sorry

end NUMINAMATH_CALUDE_second_number_is_72_l1090_109072


namespace NUMINAMATH_CALUDE_claire_shirts_count_l1090_109067

theorem claire_shirts_count :
  ∀ (brian_shirts andrew_shirts steven_shirts claire_shirts : ℕ),
    brian_shirts = 3 →
    andrew_shirts = 6 * brian_shirts →
    steven_shirts = 4 * andrew_shirts →
    claire_shirts = 5 * steven_shirts →
    claire_shirts = 360 := by
  sorry

end NUMINAMATH_CALUDE_claire_shirts_count_l1090_109067


namespace NUMINAMATH_CALUDE_lizzy_final_amount_l1090_109020

/-- Calculates the total amount Lizzy will have after loans are returned with interest -/
def lizzys_total_amount (initial_amount : ℝ) (alice_loan : ℝ) (bob_loan : ℝ) 
  (alice_interest_rate : ℝ) (bob_interest_rate : ℝ) : ℝ :=
  initial_amount - alice_loan - bob_loan + 
  alice_loan * (1 + alice_interest_rate) + 
  bob_loan * (1 + bob_interest_rate)

/-- Theorem stating that Lizzy will have $52.75 after loans are returned -/
theorem lizzy_final_amount : 
  lizzys_total_amount 50 25 20 0.15 0.20 = 52.75 := by
  sorry

end NUMINAMATH_CALUDE_lizzy_final_amount_l1090_109020


namespace NUMINAMATH_CALUDE_hans_room_options_l1090_109080

/-- Represents a hotel with floors and rooms -/
structure Hotel where
  total_floors : ℕ
  rooms_per_floor : ℕ
  available_rooms_on_odd_floor : ℕ

/-- Calculates the number of available rooms in the hotel -/
def available_rooms (h : Hotel) : ℕ :=
  (h.total_floors / 2) * h.available_rooms_on_odd_floor

/-- The specific hotel in the problem -/
def problem_hotel : Hotel :=
  { total_floors := 20
    rooms_per_floor := 15
    available_rooms_on_odd_floor := 10 }

/-- Theorem stating that the number of available rooms in the problem hotel is 100 -/
theorem hans_room_options : available_rooms problem_hotel = 100 := by
  sorry

end NUMINAMATH_CALUDE_hans_room_options_l1090_109080


namespace NUMINAMATH_CALUDE_f_bound_iff_m_range_l1090_109005

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.exp x + x^2 / m^2 - x

theorem f_bound_iff_m_range (m : ℝ) (hm : m ≠ 0) :
  (∀ a b : ℝ, a ∈ Set.Icc (-1) 1 → b ∈ Set.Icc (-1) 1 → |f m a - f m b| ≤ Real.exp 1) ↔
  m ∈ Set.Iic (-Real.sqrt 2 / 2) ∪ Set.Ici (Real.sqrt 2 / 2) :=
sorry

end NUMINAMATH_CALUDE_f_bound_iff_m_range_l1090_109005


namespace NUMINAMATH_CALUDE_point_M_theorem_l1090_109021

-- Define the point M
def M (a : ℝ) : ℝ × ℝ := (3*a - 9, 10 - 2*a)

-- Define the conditions
def second_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 > 0

def equal_distance_to_axes (p : ℝ × ℝ) : Prop := abs p.1 = abs p.2

-- Theorem statement
theorem point_M_theorem (a : ℝ) :
  second_quadrant (M a) ∧ equal_distance_to_axes (M a) →
  (a + 2)^2023 - 1 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_point_M_theorem_l1090_109021


namespace NUMINAMATH_CALUDE_johns_age_l1090_109002

theorem johns_age (j d : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 := by
  sorry

end NUMINAMATH_CALUDE_johns_age_l1090_109002


namespace NUMINAMATH_CALUDE_b_investment_value_l1090_109073

/-- Calculates the investment of partner B in a partnership business --/
def calculate_b_investment (a_investment b_investment c_investment total_profit a_profit : ℚ) : Prop :=
  let total_investment := a_investment + b_investment + c_investment
  (a_investment / total_investment = a_profit / total_profit) ∧
  b_investment = 13650

/-- Theorem stating B's investment given the problem conditions --/
theorem b_investment_value :
  calculate_b_investment 6300 13650 10500 12500 3750 := by
  sorry

end NUMINAMATH_CALUDE_b_investment_value_l1090_109073


namespace NUMINAMATH_CALUDE_no_good_points_iff_a_in_range_l1090_109034

def f (a x : ℝ) : ℝ := x^2 + 2*a*x + 1

def has_no_good_points (a : ℝ) : Prop :=
  ∀ x : ℝ, f a x ≠ x

theorem no_good_points_iff_a_in_range :
  ∀ a : ℝ, has_no_good_points a ↔ -1/2 < a ∧ a < 3/2 := by
  sorry

end NUMINAMATH_CALUDE_no_good_points_iff_a_in_range_l1090_109034


namespace NUMINAMATH_CALUDE_N_not_cube_l1090_109052

/-- Represents a number of the form 10...050...01 with 100 zeros in each group -/
def N : ℕ := 10^201 + 5 * 10^100 + 1

/-- Theorem stating that N is not a perfect cube -/
theorem N_not_cube : ¬ ∃ (m : ℕ), m^3 = N := by
  sorry

end NUMINAMATH_CALUDE_N_not_cube_l1090_109052


namespace NUMINAMATH_CALUDE_least_xy_value_l1090_109031

theorem least_xy_value (x y : ℕ+) (h : (1 : ℚ) / x + (1 : ℚ) / (3 * y) = (1 : ℚ) / 8) :
  (x * y : ℕ) ≥ 96 ∧ ∃ (a b : ℕ+), (a : ℚ) / b + (1 : ℚ) / (3 * b) = (1 : ℚ) / 8 ∧ (a * b : ℕ) = 96 :=
sorry

end NUMINAMATH_CALUDE_least_xy_value_l1090_109031


namespace NUMINAMATH_CALUDE_chess_tournament_games_l1090_109016

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) : 
  n = 24 → 
  total_games = 276 → 
  total_games = n * (n - 1) / 2 → 
  ∃ (games_per_participant : ℕ), 
    games_per_participant = n - 1 ∧ 
    games_per_participant = 23 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l1090_109016


namespace NUMINAMATH_CALUDE_total_wheels_at_station_l1090_109038

/-- The number of trains at the station -/
def num_trains : ℕ := 4

/-- The number of carriages per train -/
def carriages_per_train : ℕ := 4

/-- The number of wheel rows per carriage -/
def wheel_rows_per_carriage : ℕ := 3

/-- The number of wheels per row -/
def wheels_per_row : ℕ := 5

/-- The total number of wheels at the train station -/
def total_wheels : ℕ := num_trains * carriages_per_train * wheel_rows_per_carriage * wheels_per_row

theorem total_wheels_at_station :
  total_wheels = 240 :=
by sorry

end NUMINAMATH_CALUDE_total_wheels_at_station_l1090_109038


namespace NUMINAMATH_CALUDE_min_stamps_for_39_cents_l1090_109037

theorem min_stamps_for_39_cents : 
  ∃ (c f : ℕ), 3 * c + 5 * f = 39 ∧ 
  c + f = 9 ∧ 
  ∀ (c' f' : ℕ), 3 * c' + 5 * f' = 39 → c + f ≤ c' + f' :=
by sorry

end NUMINAMATH_CALUDE_min_stamps_for_39_cents_l1090_109037


namespace NUMINAMATH_CALUDE_remainder_of_2857916_div_4_l1090_109033

theorem remainder_of_2857916_div_4 : 2857916 % 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_2857916_div_4_l1090_109033


namespace NUMINAMATH_CALUDE_derivative_f_at_neg_one_l1090_109039

noncomputable def f (x : ℝ) : ℝ := (1 + x) * (2 + x^2)^(1/2) * (3 + x^3)^(1/3)

theorem derivative_f_at_neg_one :
  deriv f (-1) = Real.sqrt 3 * 2^(1/3) :=
sorry

end NUMINAMATH_CALUDE_derivative_f_at_neg_one_l1090_109039


namespace NUMINAMATH_CALUDE_cube_paint_theorem_l1090_109049

/-- Given a cube of side length n, prove that if exactly one-third of the total number of faces
    of the n^3 unit cubes (obtained by cutting the original cube) are red, then n = 3. -/
theorem cube_paint_theorem (n : ℕ) (h : n > 0) :
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_paint_theorem_l1090_109049


namespace NUMINAMATH_CALUDE_min_sum_inequality_l1090_109075

theorem min_sum_inequality (a b μ : ℝ) (ha : a > 0) (hb : b > 0) (hμ : μ > 0) 
  (h : 1/a + 9/b = 1) : 
  (∀ a b, a > 0 → b > 0 → 1/a + 9/b = 1 → a + b ≥ μ) ↔ μ ∈ Set.Ioo 0 16 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_inequality_l1090_109075


namespace NUMINAMATH_CALUDE_amanda_ticket_sales_l1090_109003

/-- The number of days Amanda needs to sell tickets -/
def days_to_sell : ℕ := 3

/-- The total number of tickets Amanda needs to sell -/
def total_tickets : ℕ := 80

/-- The number of tickets sold on day 1 -/
def day1_sales : ℕ := 20

/-- The number of tickets sold on day 2 -/
def day2_sales : ℕ := 32

/-- The number of tickets sold on day 3 -/
def day3_sales : ℕ := 28

/-- Theorem stating that Amanda needs 3 days to sell all tickets -/
theorem amanda_ticket_sales : 
  days_to_sell = 3 ∧ 
  total_tickets = day1_sales + day2_sales + day3_sales := by
  sorry

end NUMINAMATH_CALUDE_amanda_ticket_sales_l1090_109003


namespace NUMINAMATH_CALUDE_wednesday_to_monday_ratio_l1090_109095

/-- Represents the number of cars passing through a toll booth on each day of the week -/
structure TollBoothWeek where
  total : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ
  sunday : ℕ

/-- The ratio of cars on Wednesday to cars on Monday is 2:1 -/
theorem wednesday_to_monday_ratio (week : TollBoothWeek)
  (h1 : week.total = 450)
  (h2 : week.monday = 50)
  (h3 : week.tuesday = 50)
  (h4 : week.wednesday = week.thursday)
  (h5 : week.friday = 50)
  (h6 : week.saturday = 50)
  (h7 : week.sunday = 50)
  (h8 : week.total = week.monday + week.tuesday + week.wednesday + week.thursday + 
                     week.friday + week.saturday + week.sunday) :
  week.wednesday = 2 * week.monday := by
  sorry

#check wednesday_to_monday_ratio

end NUMINAMATH_CALUDE_wednesday_to_monday_ratio_l1090_109095


namespace NUMINAMATH_CALUDE_max_inscribed_right_triangles_l1090_109060

/-- Represents an ellipse with equation x^2 + a^2 * y^2 = a^2 where a > 1 -/
structure Ellipse where
  a : ℝ
  h_a_gt_one : a > 1

/-- Represents a right triangle inscribed in the ellipse with C(0, 1) as the right angle -/
structure InscribedRightTriangle (e : Ellipse) where
  A : ℝ × ℝ
  B : ℝ × ℝ
  h_on_ellipse_A : A.1^2 + e.a^2 * A.2^2 = e.a^2
  h_on_ellipse_B : B.1^2 + e.a^2 * B.2^2 = e.a^2
  h_right_angle : (A.1 - 0) * (B.1 - 0) + (A.2 - 1) * (B.2 - 1) = 0

/-- The theorem stating the maximum number of inscribed right triangles -/
theorem max_inscribed_right_triangles (e : Ellipse) : 
  (∃ (n : ℕ), ∀ (m : ℕ), (∃ (f : Fin m → InscribedRightTriangle e), Function.Injective f) → m ≤ n) ∧ 
  (∃ (f : Fin 3 → InscribedRightTriangle e), Function.Injective f) := by
  sorry

end NUMINAMATH_CALUDE_max_inscribed_right_triangles_l1090_109060


namespace NUMINAMATH_CALUDE_valid_numbers_l1090_109048

def is_valid_number (n : ℕ) : Prop :=
  ∃ (A B C : ℕ),
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A ≥ 1 ∧ A ≤ 9 ∧ B ≥ 0 ∧ B ≤ 9 ∧ C ≥ 0 ∧ C ≤ 9 ∧
    n = 100001 * A + 10010 * B + 1100 * C ∧
    n % 7 = 0 ∧
    (100 * A + 10 * B + C) % 7 = 0

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {168861, 259952, 861168, 952259} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l1090_109048


namespace NUMINAMATH_CALUDE_herb_leaf_difference_l1090_109071

theorem herb_leaf_difference : 
  ∀ (basil sage verbena : ℕ),
  basil = 2 * sage →
  basil + sage + verbena = 29 →
  basil = 12 →
  verbena - sage = 5 := by
sorry

end NUMINAMATH_CALUDE_herb_leaf_difference_l1090_109071


namespace NUMINAMATH_CALUDE_common_divisors_sum_l1090_109065

theorem common_divisors_sum (numbers : List Int) (divisors : List Nat) : 
  numbers = [48, 144, -24, 180, 192] →
  divisors.length = 4 →
  (∀ d ∈ divisors, d > 0) →
  (∀ n ∈ numbers, ∀ d ∈ divisors, n % d = 0) →
  divisors.sum = 12 :=
by sorry

end NUMINAMATH_CALUDE_common_divisors_sum_l1090_109065


namespace NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l1090_109096

theorem smallest_common_multiple_of_8_and_6 : 
  ∃ (n : ℕ), n > 0 ∧ 8 ∣ n ∧ 6 ∣ n ∧ ∀ (m : ℕ), m > 0 → 8 ∣ m → 6 ∣ m → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_common_multiple_of_8_and_6_l1090_109096


namespace NUMINAMATH_CALUDE_expression_simplification_l1090_109058

theorem expression_simplification (x y : ℚ) (hx : x = -4) (hy : y = -1/2) :
  x^2 - (x^2 - 2*x*y + 3*(x*y - 1/3*y^2)) = -7/4 := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1090_109058


namespace NUMINAMATH_CALUDE_rachel_assembly_time_l1090_109007

/-- Calculates the total time taken to assemble furniture -/
def total_assembly_time (num_chairs : ℕ) (num_tables : ℕ) (time_per_piece : ℕ) : ℕ :=
  (num_chairs + num_tables) * time_per_piece

/-- Proves that the total assembly time for Rachel's furniture is 40 minutes -/
theorem rachel_assembly_time :
  total_assembly_time 7 3 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_rachel_assembly_time_l1090_109007


namespace NUMINAMATH_CALUDE_solve_system_l1090_109074

theorem solve_system (x y : ℚ) (eq1 : 2 * x - 3 * y = 15) (eq2 : x + 2 * y = 8) : x = 54 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l1090_109074


namespace NUMINAMATH_CALUDE_fencing_cost_per_meter_l1090_109035

/-- Proves that for a rectangular field with sides in the ratio 3:4, area of 7500 sq. m,
    and a total fencing cost of 87.5, the cost per metre of fencing is 0.25. -/
theorem fencing_cost_per_meter (length width : ℝ) (h1 : width / length = 4 / 3)
    (h2 : length * width = 7500) (h3 : 87.5 = 2 * (length + width) * cost_per_meter) :
  cost_per_meter = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_fencing_cost_per_meter_l1090_109035


namespace NUMINAMATH_CALUDE_credit_card_balance_proof_l1090_109059

/-- Calculates the new credit card balance after purchases and returns -/
def new_credit_card_balance (initial_balance groceries_cost towels_return : ℚ) : ℚ :=
  initial_balance + groceries_cost + (groceries_cost / 2) - towels_return

/-- Proves that the new credit card balance is correct given the initial conditions -/
theorem credit_card_balance_proof :
  new_credit_card_balance 126 60 45 = 171 := by
  sorry

#eval new_credit_card_balance 126 60 45

end NUMINAMATH_CALUDE_credit_card_balance_proof_l1090_109059


namespace NUMINAMATH_CALUDE_set_P_definition_l1090_109086

def U : Set ℕ := {1, 2, 3, 4, 5}
def C_UP : Set ℕ := {4, 5}

theorem set_P_definition : 
  ∃ P : Set ℕ, P = U \ C_UP ∧ P = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_set_P_definition_l1090_109086


namespace NUMINAMATH_CALUDE_bills_steps_correct_l1090_109040

/-- The length of each step Bill takes, in metres -/
def step_length : ℚ := 1/2

/-- The total distance Bill walks, in metres -/
def total_distance : ℚ := 12

/-- The number of steps Bill takes to walk the total distance -/
def number_of_steps : ℕ := 24

/-- Theorem stating that the number of steps Bill takes is correct -/
theorem bills_steps_correct : 
  (step_length * number_of_steps : ℚ) = total_distance :=
by sorry

end NUMINAMATH_CALUDE_bills_steps_correct_l1090_109040


namespace NUMINAMATH_CALUDE_raccoon_lock_ratio_l1090_109044

/-- Proves that the ratio of time both locks stall raccoons to time second lock alone stalls raccoons is 5 -/
theorem raccoon_lock_ratio : 
  let first_lock_time : ℕ := 5
  let second_lock_time : ℕ := 3 * first_lock_time - 3
  let both_locks_time : ℕ := 60
  both_locks_time / second_lock_time = 5 := by
  sorry

end NUMINAMATH_CALUDE_raccoon_lock_ratio_l1090_109044


namespace NUMINAMATH_CALUDE_bread_per_sandwich_proof_l1090_109070

/-- The number of sandwiches Sally eats on Saturday -/
def saturday_sandwiches : ℕ := 2

/-- The number of sandwiches Sally eats on Sunday -/
def sunday_sandwiches : ℕ := 1

/-- The total number of pieces of bread Sally eats across Saturday and Sunday -/
def total_bread : ℕ := 6

/-- The number of pieces of bread per sandwich -/
def bread_per_sandwich : ℕ := 2

theorem bread_per_sandwich_proof :
  saturday_sandwiches * bread_per_sandwich + sunday_sandwiches * bread_per_sandwich = total_bread :=
by sorry

end NUMINAMATH_CALUDE_bread_per_sandwich_proof_l1090_109070


namespace NUMINAMATH_CALUDE_shelly_money_theorem_l1090_109029

/-- Calculates the total amount of money Shelly has given the number of $10 and $5 bills -/
def total_money (ten_dollar_bills : ℕ) (five_dollar_bills : ℕ) : ℕ :=
  10 * ten_dollar_bills + 5 * five_dollar_bills

/-- Proves that Shelly has $130 in total -/
theorem shelly_money_theorem :
  let ten_dollar_bills : ℕ := 10
  let five_dollar_bills : ℕ := ten_dollar_bills - 4
  total_money ten_dollar_bills five_dollar_bills = 130 := by
  sorry

#check shelly_money_theorem

end NUMINAMATH_CALUDE_shelly_money_theorem_l1090_109029


namespace NUMINAMATH_CALUDE_dealer_purchase_fraction_l1090_109030

/-- Represents the pricing details of an article sold by a dealer -/
structure ArticlePricing where
  listPrice : ℝ
  sellingPrice : ℝ
  purchasePrice : ℝ

/-- Conditions for the dealer's pricing strategy -/
def validPricing (a : ArticlePricing) : Prop :=
  a.sellingPrice = 1.5 * a.listPrice ∧ 
  a.sellingPrice = 2 * a.purchasePrice ∧
  a.listPrice > 0

/-- The theorem to be proved -/
theorem dealer_purchase_fraction (a : ArticlePricing) 
  (h : validPricing a) : 
  a.purchasePrice = (3/8 : ℝ) * a.listPrice :=
sorry

end NUMINAMATH_CALUDE_dealer_purchase_fraction_l1090_109030


namespace NUMINAMATH_CALUDE_inequality_implication_l1090_109028

theorem inequality_implication (a b x y : ℝ) (h1 : 1 < a) (h2 : a < b)
  (h3 : a^x + b^y ≤ a^(-x) + b^(-y)) : x + y ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l1090_109028


namespace NUMINAMATH_CALUDE_vectors_not_coplanar_l1090_109077

def a : ℝ × ℝ × ℝ := (1, -1, 4)
def b : ℝ × ℝ × ℝ := (1, 0, 3)
def c : ℝ × ℝ × ℝ := (1, -3, 8)

def scalar_triple_product (v1 v2 v3 : ℝ × ℝ × ℝ) : ℝ :=
  let (x1, y1, z1) := v1
  let (x2, y2, z2) := v2
  let (x3, y3, z3) := v3
  x1 * (y2 * z3 - y3 * z2) - y1 * (x2 * z3 - x3 * z2) + z1 * (x2 * y3 - x3 * y2)

theorem vectors_not_coplanar : scalar_triple_product a b c ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_vectors_not_coplanar_l1090_109077


namespace NUMINAMATH_CALUDE_divisibility_by_66_l1090_109013

theorem divisibility_by_66 : ∃ k : ℤ, 43^23 + 23^43 = 66 * k := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_66_l1090_109013


namespace NUMINAMATH_CALUDE_rectangle_area_value_l1090_109097

theorem rectangle_area_value (y : ℝ) : 
  y > 1 → 
  (3 : ℝ) * (y - 1) = 36 → 
  y = 13 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_value_l1090_109097


namespace NUMINAMATH_CALUDE_contractor_daily_wage_l1090_109015

/-- Represents the contractor's payment scenario -/
structure ContractorPayment where
  totalDays : ℕ
  finePerAbsence : ℚ
  totalPayment : ℚ
  absentDays : ℕ

/-- Calculates the daily wage of the contractor -/
def dailyWage (c : ContractorPayment) : ℚ :=
  (c.totalPayment + c.finePerAbsence * c.absentDays) / (c.totalDays - c.absentDays)

/-- Theorem stating the contractor's daily wage is 25 -/
theorem contractor_daily_wage :
  let c := ContractorPayment.mk 30 (7.5) 425 10
  dailyWage c = 25 := by sorry

end NUMINAMATH_CALUDE_contractor_daily_wage_l1090_109015


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1090_109043

/-- The eccentricity of a hyperbola with equation x²/4 - y²/2 = 1 is √6/2 -/
theorem hyperbola_eccentricity : ∃ e : ℝ, e = Real.sqrt 6 / 2 ∧
  ∀ x y : ℝ, x^2 / 4 - y^2 / 2 = 1 → 
  e = Real.sqrt ((x / 2)^2 + (y / Real.sqrt 2)^2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1090_109043


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1090_109089

theorem partial_fraction_decomposition (M₁ M₂ : ℝ) : 
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → (27 * x - 19) / (x^2 - 5*x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) →
  M₁ * M₂ = -2170 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1090_109089


namespace NUMINAMATH_CALUDE_max_exterior_elements_sum_l1090_109017

/-- A shape formed by adding a pyramid to a rectangular prism -/
structure PrismWithPyramid where
  prism_faces : ℕ
  prism_edges : ℕ
  prism_vertices : ℕ
  pyramid_base_edges : ℕ

/-- Calculate the total number of exterior elements after fusion -/
def total_exterior_elements (shape : PrismWithPyramid) : ℕ :=
  let new_faces := shape.prism_faces - 1 + shape.pyramid_base_edges
  let new_edges := shape.prism_edges + shape.pyramid_base_edges
  let new_vertices := shape.prism_vertices + 1
  new_faces + new_edges + new_vertices

/-- Theorem stating the maximum sum of exterior elements -/
theorem max_exterior_elements_sum :
  ∀ shape : PrismWithPyramid,
  shape.prism_faces = 6 →
  shape.prism_edges = 12 →
  shape.prism_vertices = 8 →
  shape.pyramid_base_edges = 4 →
  total_exterior_elements shape = 34 := by
  sorry


end NUMINAMATH_CALUDE_max_exterior_elements_sum_l1090_109017


namespace NUMINAMATH_CALUDE_extreme_value_cubic_l1090_109078

/-- Given a cubic function f(x) = x^3 + ax^2 + bx with an extreme value of -2 at x = 1,
    prove that a + 2b = -6 -/
theorem extreme_value_cubic (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + b*x
  (f 1 = -2) ∧ (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≥ f 1 ∨ f x ≤ f 1) →
  a + 2*b = -6 := by
sorry

end NUMINAMATH_CALUDE_extreme_value_cubic_l1090_109078


namespace NUMINAMATH_CALUDE_village_population_l1090_109085

theorem village_population (P : ℕ) : 
  (P : ℝ) * (1 - 0.1) * (1 - 0.2) = 3168 → P = 4400 := by
  sorry

end NUMINAMATH_CALUDE_village_population_l1090_109085


namespace NUMINAMATH_CALUDE_cost_price_calculation_l1090_109042

/-- Proves that if an article is sold for Rs. 400 with a 60% profit, its cost price is Rs. 250. -/
theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) : 
  selling_price = 400 →
  profit_percentage = 60 →
  selling_price = (1 + profit_percentage / 100) * 250 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_calculation_l1090_109042


namespace NUMINAMATH_CALUDE_muffin_packs_per_case_l1090_109062

/-- Proves the number of packs per case for Nora's muffin sale -/
theorem muffin_packs_per_case 
  (total_amount : ℕ) 
  (price_per_muffin : ℕ) 
  (num_cases : ℕ) 
  (muffins_per_pack : ℕ) 
  (h1 : total_amount = 120)
  (h2 : price_per_muffin = 2)
  (h3 : num_cases = 5)
  (h4 : muffins_per_pack = 4) :
  (total_amount / price_per_muffin) / num_cases / muffins_per_pack = 3 := by
  sorry

#check muffin_packs_per_case

end NUMINAMATH_CALUDE_muffin_packs_per_case_l1090_109062


namespace NUMINAMATH_CALUDE_distance_to_point_l1090_109046

theorem distance_to_point : Real.sqrt ((8 - 0)^2 + (-15 - 0)^2) = 17 := by sorry

end NUMINAMATH_CALUDE_distance_to_point_l1090_109046


namespace NUMINAMATH_CALUDE_ratio_sum_last_number_l1090_109090

theorem ratio_sum_last_number (a b c : ℕ) : 
  a + b + c = 1000 → 
  5 * b = a → 
  4 * b = c → 
  c = 400 := by
sorry

end NUMINAMATH_CALUDE_ratio_sum_last_number_l1090_109090


namespace NUMINAMATH_CALUDE_distance_to_line_mn_l1090_109056

/-- The distance from the origin to the line MN, where M is on the hyperbola 2x² - y² = 1
    and N is on the ellipse 4x² + y² = 1, with OM perpendicular to ON. -/
theorem distance_to_line_mn (M N : ℝ × ℝ) : 
  (2 * M.1^2 - M.2^2 = 1) →  -- M is on the hyperbola
  (4 * N.1^2 + N.2^2 = 1) →  -- N is on the ellipse
  (M.1 * N.1 + M.2 * N.2 = 0) →  -- OM ⟂ ON
  let d := Real.sqrt 3 / 3
  ∃ (t : ℝ), t * M.1 + (1 - t) * N.1 = d * (N.2 - M.2) ∧
             t * M.2 + (1 - t) * N.2 = d * (M.1 - N.1) :=
by sorry

end NUMINAMATH_CALUDE_distance_to_line_mn_l1090_109056


namespace NUMINAMATH_CALUDE_median_of_100_numbers_l1090_109024

def is_median (s : Finset ℕ) (m : ℕ) : Prop :=
  2 * (s.filter (· < m)).card ≤ s.card ∧ 2 * (s.filter (· > m)).card ≤ s.card

theorem median_of_100_numbers (s : Finset ℕ) (h_card : s.card = 100) :
  (∃ x ∈ s, is_median (s.erase x) 78) →
  (∃ y ∈ s, y ≠ x → is_median (s.erase y) 66) →
  is_median s 72 :=
sorry

end NUMINAMATH_CALUDE_median_of_100_numbers_l1090_109024


namespace NUMINAMATH_CALUDE_maya_lift_increase_l1090_109006

/-- Given America's peak lift and Maya's relative lift capacities, calculate the increase in Maya's lift capacity. -/
theorem maya_lift_increase (america_peak : ℝ) (maya_initial_ratio : ℝ) (maya_peak_ratio : ℝ) 
  (h1 : america_peak = 300)
  (h2 : maya_initial_ratio = 1/4)
  (h3 : maya_peak_ratio = 1/2) :
  maya_peak_ratio * america_peak - maya_initial_ratio * america_peak = 75 := by
  sorry

end NUMINAMATH_CALUDE_maya_lift_increase_l1090_109006


namespace NUMINAMATH_CALUDE_triangle_area_l1090_109057

theorem triangle_area (a b : ℝ) (θ : Real) (h1 : a = 30) (h2 : b = 24) (h3 : θ = π/3) :
  (1/2) * a * b * Real.sin θ = 180 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1090_109057


namespace NUMINAMATH_CALUDE_circle_area_equality_l1090_109009

theorem circle_area_equality (r₁ r₂ : ℝ) (h₁ : r₁ = 35) (h₂ : r₂ = 25) :
  ∃ r₃ : ℝ, r₃ = 10 * Real.sqrt 6 ∧ π * r₃^2 = π * (r₁^2 - r₂^2) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_equality_l1090_109009


namespace NUMINAMATH_CALUDE_max_t_for_tangent_slope_l1090_109019

/-- Given t > 0 and f(x) = x²(x - t), prove that the maximum value of t for which
    the slope of the tangent line to f(x) is always greater than or equal to -1
    when x is in (0, 1] is 3/2. -/
theorem max_t_for_tangent_slope (t : ℝ) (h_t : t > 0) :
  (∀ x : ℝ, x ∈ (Set.Ioo 0 1) → (3 * x^2 - 2 * t * x) ≥ -1) ↔ t ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_max_t_for_tangent_slope_l1090_109019


namespace NUMINAMATH_CALUDE_discount_calculation_l1090_109087

/-- Proves that a 20% discount followed by a 15% discount on an item 
    originally priced at 450 results in a final price of 306 -/
theorem discount_calculation (original_price : ℝ) (first_discount second_discount final_price : ℝ) :
  original_price = 450 ∧ 
  first_discount = 20 ∧ 
  second_discount = 15 ∧ 
  final_price = 306 →
  original_price * (1 - first_discount / 100) * (1 - second_discount / 100) = final_price :=
by sorry

end NUMINAMATH_CALUDE_discount_calculation_l1090_109087


namespace NUMINAMATH_CALUDE_seniors_physical_books_l1090_109055

/-- A survey on book preferences --/
structure BookSurvey where
  total_physical : ℕ
  adults_physical : ℕ
  seniors_ebook : ℕ

/-- The number of seniors preferring physical books --/
def seniors_physical (survey : BookSurvey) : ℕ :=
  survey.total_physical - survey.adults_physical

/-- Theorem: In the given survey, 100 seniors prefer physical books --/
theorem seniors_physical_books (survey : BookSurvey)
  (h1 : survey.total_physical = 180)
  (h2 : survey.adults_physical = 80)
  (h3 : survey.seniors_ebook = 130) :
  seniors_physical survey = 100 := by
  sorry

end NUMINAMATH_CALUDE_seniors_physical_books_l1090_109055


namespace NUMINAMATH_CALUDE_cycle_price_calculation_l1090_109079

/-- Proves that given a cycle sold for Rs. 1080 with a 60% gain, the original price of the cycle was Rs. 675. -/
theorem cycle_price_calculation (selling_price : ℝ) (gain_percent : ℝ) 
  (h1 : selling_price = 1080)
  (h2 : gain_percent = 60) :
  ∃ (original_price : ℝ), 
    selling_price = original_price * (1 + gain_percent / 100) ∧ 
    original_price = 675 :=
by sorry

end NUMINAMATH_CALUDE_cycle_price_calculation_l1090_109079


namespace NUMINAMATH_CALUDE_cos_negative_420_degrees_l1090_109022

theorem cos_negative_420_degrees : Real.cos ((-420 : ℝ) * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_negative_420_degrees_l1090_109022


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l1090_109082

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 6 = 4 ∧ ∀ m : ℕ, m < 100 ∧ m % 6 = 4 → m ≤ n := by
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l1090_109082


namespace NUMINAMATH_CALUDE_milk_water_mixture_volume_l1090_109091

/-- Proves that given a mixture of milk and water with an initial ratio of 3:2, 
    if adding 62 liters of water changes the ratio to 3:4, 
    then the initial volume of the mixture was 155 liters. -/
theorem milk_water_mixture_volume 
  (initial_milk : ℝ) 
  (initial_water : ℝ) 
  (h1 : initial_milk / initial_water = 3 / 2) 
  (h2 : initial_milk / (initial_water + 62) = 3 / 4) : 
  initial_milk + initial_water = 155 := by
  sorry

#check milk_water_mixture_volume

end NUMINAMATH_CALUDE_milk_water_mixture_volume_l1090_109091


namespace NUMINAMATH_CALUDE_men_in_room_l1090_109061

theorem men_in_room (initial_men : ℕ) (initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →
  (2 * (initial_women - 3) = 24) →
  (initial_men + 2 = 14) :=
by
  sorry

#check men_in_room

end NUMINAMATH_CALUDE_men_in_room_l1090_109061


namespace NUMINAMATH_CALUDE_distance_is_35_over_13_l1090_109012

def point : ℝ × ℝ × ℝ := (0, -1, 4)
def line_point : ℝ × ℝ × ℝ := (-3, 2, 5)
def line_direction : ℝ × ℝ × ℝ := (4, 1, -3)

def distance_to_line (p : ℝ × ℝ × ℝ) (l_point : ℝ × ℝ × ℝ) (l_dir : ℝ × ℝ × ℝ) : ℝ := 
  sorry

theorem distance_is_35_over_13 : 
  distance_to_line point line_point line_direction = 35 / 13 := by sorry

end NUMINAMATH_CALUDE_distance_is_35_over_13_l1090_109012


namespace NUMINAMATH_CALUDE_bus_distance_ratio_l1090_109023

theorem bus_distance_ratio (total_distance : ℝ) (foot_fraction : ℝ) (car_distance : ℝ) :
  total_distance = 40 →
  foot_fraction = 1 / 4 →
  car_distance = 10 →
  (total_distance - (foot_fraction * total_distance + car_distance)) / total_distance = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_bus_distance_ratio_l1090_109023


namespace NUMINAMATH_CALUDE_no_four_distinct_real_roots_l1090_109093

theorem no_four_distinct_real_roots (a b : ℝ) : 
  ¬ (∃ (r₁ r₂ r₃ r₄ : ℝ), 
    (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₃ ≠ r₄) ∧
    (r₁^4 - 4*r₁^3 + 6*r₁^2 + a*r₁ + b = 0) ∧
    (r₂^4 - 4*r₂^3 + 6*r₂^2 + a*r₂ + b = 0) ∧
    (r₃^4 - 4*r₃^3 + 6*r₃^2 + a*r₃ + b = 0) ∧
    (r₄^4 - 4*r₄^3 + 6*r₄^2 + a*r₄ + b = 0)) :=
by sorry

end NUMINAMATH_CALUDE_no_four_distinct_real_roots_l1090_109093


namespace NUMINAMATH_CALUDE_imaginary_complex_implies_m_conditions_l1090_109041

theorem imaginary_complex_implies_m_conditions (m : ℝ) : 
  (∃ (z : ℂ), z = Complex.mk (m^2 - 3*m - 4) (m^2 - 5*m - 6) ∧ z.re = 0 ∧ z.im ≠ 0) →
  (m ≠ -1 ∧ m ≠ 6) := by
  sorry

end NUMINAMATH_CALUDE_imaginary_complex_implies_m_conditions_l1090_109041


namespace NUMINAMATH_CALUDE_max_value_theorem_l1090_109076

theorem max_value_theorem (a b : ℝ) : 
  a^2 = (1 + 2*b) * (1 - 2*b) →
  ∃ (x : ℝ), x = (2*a*b)/(|a| + 2*|b|) ∧ 
             ∀ (y : ℝ), y = (2*a*b)/(|a| + 2*|b|) → y ≤ x ∧
             x = Real.sqrt 2 / 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1090_109076


namespace NUMINAMATH_CALUDE_equality_of_pairs_l1090_109094

theorem equality_of_pairs (a b x y : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ x > 0 ∧ y > 0)
  (h_sum : a + b + x + y < 2)
  (h_eq1 : a + b^2 = x + y^2)
  (h_eq2 : a^2 + b = x^2 + y) :
  a = x ∧ b = y := by
  sorry

end NUMINAMATH_CALUDE_equality_of_pairs_l1090_109094


namespace NUMINAMATH_CALUDE_equation_condition_l1090_109032

theorem equation_condition (a b c : ℕ) 
  (ha : 0 < a ∧ a < 20) 
  (hb : 0 < b ∧ b < 20) 
  (hc : 0 < c ∧ c < 20) : 
  (20 * a + b) * (20 * a + c) = 400 * a^2 + 200 * a + b * c ↔ b + c = 10 := by
sorry

end NUMINAMATH_CALUDE_equation_condition_l1090_109032


namespace NUMINAMATH_CALUDE_product_trailing_zeros_l1090_109081

def max_num : ℕ := 2020
def multiples_of_5 : ℕ := 404

-- Function to calculate the number of trailing zeros
def trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

-- Theorem statement
theorem product_trailing_zeros :
  trailing_zeros max_num = 503 :=
sorry

end NUMINAMATH_CALUDE_product_trailing_zeros_l1090_109081


namespace NUMINAMATH_CALUDE_profit_equation_l1090_109045

/-- Represents the profit equation for a product with given cost and selling prices,
    initial quantity sold, and price reduction effects. -/
theorem profit_equation
  (cost_price : ℝ)
  (initial_selling_price : ℝ)
  (initial_quantity : ℝ)
  (additional_units_per_reduction : ℝ)
  (target_profit : ℝ)
  (h1 : cost_price = 40)
  (h2 : initial_selling_price = 60)
  (h3 : initial_quantity = 200)
  (h4 : additional_units_per_reduction = 8)
  (h5 : target_profit = 8450)
  (x : ℝ) :
  (initial_selling_price - cost_price - x) * (initial_quantity + additional_units_per_reduction * x) = target_profit :=
by sorry

end NUMINAMATH_CALUDE_profit_equation_l1090_109045


namespace NUMINAMATH_CALUDE_largest_package_size_l1090_109068

/-- The largest possible number of markers in a package given that Lucy bought 30 markers, 
    Mia bought 45 markers, and Noah bought 75 markers. -/
theorem largest_package_size : Nat.gcd 30 (Nat.gcd 45 75) = 15 := by
  sorry

end NUMINAMATH_CALUDE_largest_package_size_l1090_109068


namespace NUMINAMATH_CALUDE_exercise_books_count_l1090_109063

/-- Given a shop with pencils, pens, and exercise books in the ratio 10 : 2 : 3,
    prove that if there are 120 pencils, then there are 36 exercise books. -/
theorem exercise_books_count (pencils pens books : ℕ) : 
  pencils = 120 →
  pencils / 10 = pens / 2 →
  pencils / 10 = books / 3 →
  books = 36 := by
sorry

end NUMINAMATH_CALUDE_exercise_books_count_l1090_109063


namespace NUMINAMATH_CALUDE_gain_percent_problem_l1090_109054

/-- 
If the cost price of 50 articles is equal to the selling price of 25 articles, 
then the gain percent is 100%.
-/
theorem gain_percent_problem (C S : ℝ) (hpos : C > 0) : 
  50 * C = 25 * S → (S - C) / C * 100 = 100 :=
by
  sorry

#check gain_percent_problem

end NUMINAMATH_CALUDE_gain_percent_problem_l1090_109054


namespace NUMINAMATH_CALUDE_sin_4_arcsin_l1090_109066

theorem sin_4_arcsin (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  Real.sin (4 * Real.arcsin x) = 4 * x * (1 - 2 * x^2) * Real.sqrt (1 - x^2) := by
  sorry

end NUMINAMATH_CALUDE_sin_4_arcsin_l1090_109066


namespace NUMINAMATH_CALUDE_factor_iff_t_eq_neg_six_or_one_l1090_109098

/-- The polynomial in question -/
def f (x : ℝ) : ℝ := 4 * x^2 + 20 * x - 24

/-- Theorem stating that x - t is a factor of f(x) if and only if t is -6 or 1 -/
theorem factor_iff_t_eq_neg_six_or_one :
  ∀ t : ℝ, (∃ g : ℝ → ℝ, ∀ x, f x = (x - t) * g x) ↔ (t = -6 ∨ t = 1) := by
  sorry

end NUMINAMATH_CALUDE_factor_iff_t_eq_neg_six_or_one_l1090_109098


namespace NUMINAMATH_CALUDE_cube_root_comparison_l1090_109004

theorem cube_root_comparison : 2 + Real.rpow 7 (1/3) < Real.rpow 60 (1/3) := by
  sorry

end NUMINAMATH_CALUDE_cube_root_comparison_l1090_109004


namespace NUMINAMATH_CALUDE_other_communities_count_l1090_109026

theorem other_communities_count (total_boys : ℕ) (muslim_percent hindu_percent sikh_percent : ℚ) : 
  total_boys = 850 →
  muslim_percent = 44/100 →
  hindu_percent = 14/100 →
  sikh_percent = 10/100 →
  (total_boys : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 272 := by
  sorry

end NUMINAMATH_CALUDE_other_communities_count_l1090_109026


namespace NUMINAMATH_CALUDE_sum_even_integers_102_to_200_l1090_109083

/-- The sum of even integers from 102 to 200 inclusive -/
def sum_even_102_to_200 : ℕ := 7550

/-- The sum of the first 50 positive even integers -/
def sum_first_50_even : ℕ := 2550

/-- The number of even integers from 102 to 200 inclusive -/
def num_even_102_to_200 : ℕ := 50

theorem sum_even_integers_102_to_200 :
  sum_even_102_to_200 = (num_even_102_to_200 / 2) * (102 + 200) :=
by sorry

end NUMINAMATH_CALUDE_sum_even_integers_102_to_200_l1090_109083


namespace NUMINAMATH_CALUDE_triangle_sum_zero_l1090_109008

theorem triangle_sum_zero (a b c : ℝ) 
  (ha : |a| ≥ |b + c|) 
  (hb : |b| ≥ |c + a|) 
  (hc : |c| ≥ |a + b|) : 
  a + b + c = 0 := by
sorry

end NUMINAMATH_CALUDE_triangle_sum_zero_l1090_109008


namespace NUMINAMATH_CALUDE_ending_number_proof_l1090_109001

theorem ending_number_proof (start : ℕ) (multiples : ℚ) (end_number : ℕ) : 
  start = 81 → 
  multiples = 93.33333333333333 → 
  end_number = (start + 3 * (multiples.floor - 1)) → 
  end_number = 357 := by
sorry

end NUMINAMATH_CALUDE_ending_number_proof_l1090_109001


namespace NUMINAMATH_CALUDE_quadrilateral_area_l1090_109069

/-- The area of a quadrilateral ABCD with given diagonal and offsets -/
theorem quadrilateral_area (BD AC : ℝ) (offset_A offset_C : ℝ) :
  BD = 28 →
  offset_A = 8 →
  offset_C = 2 →
  (1/2 * BD * offset_A) + (1/2 * BD * offset_C) = 140 :=
by sorry

end NUMINAMATH_CALUDE_quadrilateral_area_l1090_109069


namespace NUMINAMATH_CALUDE_probability_one_boy_one_girl_l1090_109018

def num_boys : ℕ := 3
def num_girls : ℕ := 2
def num_participants : ℕ := 2

def total_combinations : ℕ := (num_boys + num_girls).choose num_participants

def favorable_outcomes : ℕ := num_boys.choose 1 * num_girls.choose 1

theorem probability_one_boy_one_girl :
  (favorable_outcomes : ℚ) / total_combinations = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_one_boy_one_girl_l1090_109018


namespace NUMINAMATH_CALUDE_average_increase_is_three_l1090_109092

/-- Represents a batsman's statistics -/
structure Batsman where
  total_runs : ℕ
  innings : ℕ
  average : ℚ

/-- Calculates the new average after an inning -/
def new_average (b : Batsman) (runs : ℕ) : ℚ :=
  (b.total_runs + runs) / (b.innings + 1)

/-- Theorem: The increase in average is 3 for the given conditions -/
theorem average_increase_is_three (b : Batsman) (h1 : b.innings = 16) 
    (h2 : new_average b 92 = 44) : 
    new_average b 92 - b.average = 3 := by
  sorry

#check average_increase_is_three

end NUMINAMATH_CALUDE_average_increase_is_three_l1090_109092


namespace NUMINAMATH_CALUDE_neil_fraction_of_packs_l1090_109084

def total_marbles : ℕ := 400
def marbles_per_pack : ℕ := 10
def packs_kept_by_leo : ℕ := 25
def fraction_to_manny : ℚ := 1/4

theorem neil_fraction_of_packs (total_packs : ℕ) (packs_given_away : ℕ) 
  (packs_to_manny : ℕ) (packs_to_neil : ℕ) :
  total_packs = total_marbles / marbles_per_pack →
  packs_given_away = total_packs - packs_kept_by_leo →
  packs_to_manny = ⌊(fraction_to_manny * packs_given_away : ℚ)⌋ →
  packs_to_neil = packs_given_away - packs_to_manny →
  (packs_to_neil : ℚ) / packs_given_away = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_neil_fraction_of_packs_l1090_109084


namespace NUMINAMATH_CALUDE_a_8_equals_14_l1090_109050

/-- The sum of the first n terms of the sequence a_n -/
def S (n : ℕ) : ℕ := n^2 - n

/-- The nth term of the sequence a_n -/
def a (n : ℕ) : ℕ := S n - S (n-1)

theorem a_8_equals_14 : a 8 = 14 := by sorry

end NUMINAMATH_CALUDE_a_8_equals_14_l1090_109050


namespace NUMINAMATH_CALUDE_total_cost_trick_decks_l1090_109051

/-- Calculates the cost of decks based on tiered pricing and promotion --/
def calculate_cost (num_decks : ℕ) : ℚ :=
  let base_price := if num_decks ≤ 3 then 8
                    else if num_decks ≤ 6 then 7
                    else 6
  let full_price_decks := num_decks / 2
  let discounted_decks := num_decks - full_price_decks
  (full_price_decks * base_price + discounted_decks * base_price / 2 : ℚ)

/-- The total cost of trick decks for Victor and his friend --/
theorem total_cost_trick_decks : 
  calculate_cost 6 + calculate_cost 2 = 43.5 := by
  sorry

#eval calculate_cost 6 + calculate_cost 2

end NUMINAMATH_CALUDE_total_cost_trick_decks_l1090_109051


namespace NUMINAMATH_CALUDE_linear_equation_exponents_l1090_109088

/-- A function to represent the linearity of an equation in two variables -/
def is_linear_two_var (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (m n c : ℝ), ∀ x y, f x y = m * x + n * y + c

/-- The main theorem -/
theorem linear_equation_exponents :
  ∀ a b : ℝ,
  (is_linear_two_var (λ x y => x^(a-3) + y^(b-1))) →
  (a = 4 ∧ b = 2) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_exponents_l1090_109088


namespace NUMINAMATH_CALUDE_point_not_in_region_l1090_109000

def plane_region (x y : ℝ) : Prop := 3 * x + 2 * y < 6

theorem point_not_in_region : ¬ plane_region 2 0 := by
  sorry

end NUMINAMATH_CALUDE_point_not_in_region_l1090_109000


namespace NUMINAMATH_CALUDE_negation_existence_gt_one_l1090_109036

theorem negation_existence_gt_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_existence_gt_one_l1090_109036


namespace NUMINAMATH_CALUDE_product_without_x_terms_l1090_109014

theorem product_without_x_terms (m n : ℝ) : 
  (∀ x : ℝ, (x + 2*m) * (x^2 - x + 1/2*n) = x^3 + 2*m*n) → 
  m^2023 * n^2022 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_product_without_x_terms_l1090_109014


namespace NUMINAMATH_CALUDE_range_of_reciprocal_sum_l1090_109011

theorem range_of_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 1) :
  (1/x + 1/y ≥ 3 + 2*Real.sqrt 2) ∧ 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a + 2*b = 1 ∧ 1/a + 1/b = 3 + 2*Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_reciprocal_sum_l1090_109011


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l1090_109053

theorem divisibility_implies_equality (a b n : ℕ) :
  (∀ k : ℕ, k ≠ 0 → ∃ q : ℤ, a - k^n = (b - k) * q) →
  a = b^n := by sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l1090_109053


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l1090_109099

theorem complex_fraction_simplification : 
  (((12^4 + 500) * (24^4 + 500) * (36^4 + 500) * (48^4 + 500) * (60^4 + 500)) / 
   ((6^4 + 500) * (18^4 + 500) * (30^4 + 500) * (42^4 + 500) * (54^4 + 500))) = -182 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l1090_109099


namespace NUMINAMATH_CALUDE_relationship_abc_l1090_109064

theorem relationship_abc (a b c : ℝ) 
  (ha : a = Real.rpow 0.9 0.3)
  (hb : b = Real.rpow 1.2 0.3)
  (hc : c = Real.rpow 0.5 (-0.3)) :
  c > b ∧ b > a :=
by sorry

end NUMINAMATH_CALUDE_relationship_abc_l1090_109064


namespace NUMINAMATH_CALUDE_quadratic_properties_l1090_109010

-- Define the quadratic function
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the conditions
def intersects_x_axis (a b c : ℝ) : Prop :=
  quadratic a b c (-2) = 0 ∧ ∃ x1, 1 < x1 ∧ x1 < 2 ∧ quadratic a b c x1 = 0

def intersects_y_axis (a b c : ℝ) : Prop :=
  0 < quadratic a b c 0 ∧ quadratic a b c 0 < 2

-- Theorem statement
theorem quadratic_properties (a b c : ℝ) 
  (h1 : intersects_x_axis a b c) 
  (h2 : intersects_y_axis a b c) :
  4*a - 2*b + c = 0 ∧ 
  2*a - b < 0 ∧ 
  2*a - b > -1 ∧ 
  b > a := by
  sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1090_109010


namespace NUMINAMATH_CALUDE_correct_sampling_methods_l1090_109047

/-- Represents a sampling method -/
inductive SamplingMethod
  | Systematic
  | SimpleRandom
  | Stratified

/-- Represents a sampling scenario -/
structure SamplingScenario where
  method : SamplingMethod
  description : String

/-- The milk production line sampling scenario -/
def milkProductionScenario : SamplingScenario :=
  { method := SamplingMethod.Systematic,
    description := "Sampling a bag every 30 minutes on a milk production line" }

/-- The math enthusiasts sampling scenario -/
def mathEnthusiastsScenario : SamplingScenario :=
  { method := SamplingMethod.SimpleRandom,
    description := "Selecting 3 individuals from 30 math enthusiasts in a middle school" }

/-- Theorem stating that the sampling methods are correctly identified -/
theorem correct_sampling_methods :
  (milkProductionScenario.method = SamplingMethod.Systematic) ∧
  (mathEnthusiastsScenario.method = SamplingMethod.SimpleRandom) :=
sorry

end NUMINAMATH_CALUDE_correct_sampling_methods_l1090_109047


namespace NUMINAMATH_CALUDE_soda_price_increase_l1090_109027

/-- Proves that the percentage increase in the price of a can of soda is 50% given the specified conditions. -/
theorem soda_price_increase (initial_combined_price new_candy_price new_soda_price candy_increase : ℝ) 
  (h1 : initial_combined_price = 16)
  (h2 : new_candy_price = 15)
  (h3 : new_soda_price = 6)
  (h4 : candy_increase = 25)
  : (new_soda_price - (initial_combined_price - new_candy_price / (1 + candy_increase / 100))) / 
    (initial_combined_price - new_candy_price / (1 + candy_increase / 100)) * 100 = 50 := by
  sorry

#check soda_price_increase

end NUMINAMATH_CALUDE_soda_price_increase_l1090_109027
