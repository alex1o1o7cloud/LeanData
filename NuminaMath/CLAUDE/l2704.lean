import Mathlib

namespace alice_ice_cream_l2704_270486

/-- The number of pints of ice cream Alice bought on Sunday -/
def sunday_pints : ℕ := sorry

/-- The number of pints Alice had on Wednesday after returning expired ones -/
def wednesday_pints : ℕ := 18

theorem alice_ice_cream :
  sunday_pints = 4 ∧
  3 * sunday_pints + sunday_pints + sunday_pints - sunday_pints / 2 = wednesday_pints :=
by sorry

end alice_ice_cream_l2704_270486


namespace cube_product_inequality_l2704_270453

theorem cube_product_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  x^3 * y^3 * (x^3 + y^3) ≤ 2 := by
  sorry

end cube_product_inequality_l2704_270453


namespace john_payment_first_year_l2704_270406

/- Define the family members -/
inductive FamilyMember
| John
| Wife
| Son
| Daughter

/- Define whether a family member is extended or not -/
def isExtended : FamilyMember → Bool
  | FamilyMember.Wife => true
  | _ => false

/- Define the initial membership fee -/
def initialMembershipFee : ℕ := 4000

/- Define the monthly cost for each family member -/
def monthlyCost : FamilyMember → ℕ
  | FamilyMember.John => 1000
  | FamilyMember.Wife => 1200
  | FamilyMember.Son => 800
  | FamilyMember.Daughter => 900

/- Define the membership fee discount rate for extended family members -/
def membershipDiscountRate : ℚ := 1/5

/- Define the monthly fee discount rate for extended family members -/
def monthlyDiscountRate : ℚ := 1/10

/- Define the number of months in a year -/
def monthsInYear : ℕ := 12

/- Define John's payment fraction -/
def johnPaymentFraction : ℚ := 1/2

/- Theorem statement -/
theorem john_payment_first_year :
  let totalCost := (FamilyMember.John :: FamilyMember.Wife :: FamilyMember.Son :: FamilyMember.Daughter :: []).foldl
    (fun acc member =>
      let membershipFee := if isExtended member then initialMembershipFee * (1 - membershipDiscountRate) else initialMembershipFee
      let monthlyFee := if isExtended member then monthlyCost member * (1 - monthlyDiscountRate) else monthlyCost member
      acc + membershipFee + monthlyFee * monthsInYear)
    0
  johnPaymentFraction * totalCost = 30280 := by
  sorry

end john_payment_first_year_l2704_270406


namespace rainfall_difference_l2704_270416

def monday_count : ℕ := 10
def tuesday_count : ℕ := 12
def wednesday_count : ℕ := 8
def thursday_count : ℕ := 6

def monday_rain : ℝ := 1.25
def tuesday_rain : ℝ := 2.15
def wednesday_rain : ℝ := 1.60
def thursday_rain : ℝ := 2.80

theorem rainfall_difference :
  (tuesday_count * tuesday_rain + thursday_count * thursday_rain) -
  (monday_count * monday_rain + wednesday_count * wednesday_rain) = 17.3 := by
  sorry

end rainfall_difference_l2704_270416


namespace range_of_a_for_always_positive_quadratic_l2704_270498

theorem range_of_a_for_always_positive_quadratic :
  {a : ℝ | ∀ x : ℝ, 2 * x^2 + (a - 1) * x + (1/2 : ℝ) > 0} = {a : ℝ | -1 < a ∧ a < 3} := by
  sorry

end range_of_a_for_always_positive_quadratic_l2704_270498


namespace false_conjunction_implication_l2704_270432

theorem false_conjunction_implication : ¬(∀ (p q : Prop), (¬(p ∧ q)) → (¬p ∧ ¬q)) := by
  sorry

end false_conjunction_implication_l2704_270432


namespace right_triangle_side_length_l2704_270483

theorem right_triangle_side_length (A B C : ℝ × ℝ) (AB AC BC : ℝ) :
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = AB^2 →
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = AC^2 →
  (B.1 - C.1)^2 + (B.2 - C.2)^2 = BC^2 →
  AB^2 + AC^2 = BC^2 →
  Real.cos (30 * π / 180) = (BC^2 + AC^2 - AB^2) / (2 * BC * AC) →
  AC = 18 →
  AB = 18 * Real.sqrt 3 := by
sorry

end right_triangle_side_length_l2704_270483


namespace tom_has_sixteen_robots_l2704_270431

/-- The number of animal robots Michael has -/
def michael_robots : ℕ := 8

/-- The number of animal robots Tom has -/
def tom_robots : ℕ := 2 * michael_robots

/-- Theorem stating that Tom has 16 animal robots -/
theorem tom_has_sixteen_robots : tom_robots = 16 := by
  sorry

end tom_has_sixteen_robots_l2704_270431


namespace friend_candy_purchase_l2704_270450

def feeding_allowance : ℚ := 4
def fraction_given : ℚ := 1/4
def candy_cost : ℚ := 1/5  -- 20 cents = 1/5 dollar

theorem friend_candy_purchase :
  (feeding_allowance * fraction_given) / candy_cost = 5 := by
  sorry

end friend_candy_purchase_l2704_270450


namespace initial_water_percentage_l2704_270475

/-- 
Given a mixture of 180 liters, if adding 12 liters of water results in a new mixture 
where water is 25% of the total, then the initial percentage of water in the mixture was 20%.
-/
theorem initial_water_percentage (initial_volume : ℝ) (added_water : ℝ) (final_water_percentage : ℝ) :
  initial_volume = 180 →
  added_water = 12 →
  final_water_percentage = 25 →
  (initial_volume * (20 / 100) + added_water) / (initial_volume + added_water) = final_water_percentage / 100 :=
by sorry

end initial_water_percentage_l2704_270475


namespace junior_prom_attendance_l2704_270470

theorem junior_prom_attendance :
  ∀ (total_kids : ℕ),
    (total_kids / 4 : ℕ) = 25 + 10 →
    total_kids = 140 :=
by
  sorry

end junior_prom_attendance_l2704_270470


namespace circle_probability_l2704_270404

def total_figures : ℕ := 10
def triangle_count : ℕ := 4
def circle_count : ℕ := 3
def square_count : ℕ := 3

theorem circle_probability : 
  (circle_count : ℚ) / total_figures = 3 / 10 := by sorry

end circle_probability_l2704_270404


namespace johns_donation_l2704_270479

theorem johns_donation (
  initial_contributions : ℕ) 
  (new_average : ℚ)
  (increase_percentage : ℚ) :
  initial_contributions = 3 →
  new_average = 75 →
  increase_percentage = 50 / 100 →
  ∃ (johns_donation : ℚ),
    johns_donation = 150 ∧
    new_average = (initial_contributions * (new_average / (1 + increase_percentage)) + johns_donation) / (initial_contributions + 1) :=
by sorry

end johns_donation_l2704_270479


namespace calculation_part1_sum_first_25_odd_numbers_l2704_270424

-- Part 1
theorem calculation_part1 : 0.45 * 2.5 + 4.5 * 0.65 + 0.45 = 4.5 := by
  sorry

-- Part 2
def first_n_odd_numbers (n : ℕ) : List ℕ :=
  List.range n |>.map (fun i => 2 * i + 1)

theorem sum_first_25_odd_numbers :
  (first_n_odd_numbers 25).sum = 625 := by
  sorry

end calculation_part1_sum_first_25_odd_numbers_l2704_270424


namespace total_rounded_to_nearest_dollar_l2704_270401

def purchase1 : ℚ := 1.98
def purchase2 : ℚ := 5.04
def purchase3 : ℚ := 9.89

def roundToNearestInteger (x : ℚ) : ℤ :=
  if x - ↑(Int.floor x) < 1/2 then Int.floor x else Int.ceil x

theorem total_rounded_to_nearest_dollar :
  roundToNearestInteger (purchase1 + purchase2 + purchase3) = 17 := by sorry

end total_rounded_to_nearest_dollar_l2704_270401


namespace reciprocal_of_one_fifth_l2704_270429

theorem reciprocal_of_one_fifth (x : ℚ) : 
  (x * (1 / x) = 1) → ((1 / (1 / 5)) = 5) := by
  sorry

end reciprocal_of_one_fifth_l2704_270429


namespace mary_spends_five_l2704_270451

/-- Proves that Mary spends $5 given the initial conditions and final state -/
theorem mary_spends_five (marco_initial : ℕ) (mary_initial : ℕ) 
  (h1 : marco_initial = 24)
  (h2 : mary_initial = 15)
  (marco_gives : ℕ := marco_initial / 2)
  (marco_final : ℕ := marco_initial - marco_gives)
  (mary_after_receiving : ℕ := mary_initial + marco_gives)
  (mary_final : ℕ)
  (h3 : mary_final = marco_final + 10) :
  mary_after_receiving - mary_final = 5 := by
sorry

end mary_spends_five_l2704_270451


namespace bills_age_l2704_270423

theorem bills_age (bill eric : ℕ) 
  (h1 : bill = eric + 4) 
  (h2 : bill + eric = 28) : 
  bill = 16 := by
  sorry

end bills_age_l2704_270423


namespace lindas_cookies_l2704_270441

theorem lindas_cookies (classmates : ℕ) (cookies_per_student : ℕ) (cookies_per_batch : ℕ) 
  (chocolate_chip_batches : ℕ) (remaining_batches : ℕ) :
  classmates = 24 →
  cookies_per_student = 10 →
  cookies_per_batch = 48 →
  chocolate_chip_batches = 2 →
  remaining_batches = 2 →
  (classmates * cookies_per_student - chocolate_chip_batches * cookies_per_batch) / cookies_per_batch - remaining_batches = 1 :=
by sorry

end lindas_cookies_l2704_270441


namespace point_n_coordinates_l2704_270477

/-- Given point M(5, -6) and vector a = (1, -2), if MN = -3a, then N has coordinates (2, 0) -/
theorem point_n_coordinates (M N : ℝ × ℝ) (a : ℝ × ℝ) :
  M = (5, -6) →
  a = (1, -2) →
  N.1 - M.1 = -3 * a.1 ∧ N.2 - M.2 = -3 * a.2 →
  N = (2, 0) := by
  sorry

end point_n_coordinates_l2704_270477


namespace collinear_vectors_y_value_l2704_270407

theorem collinear_vectors_y_value (y : ℝ) : 
  let a : Fin 2 → ℝ := ![(-3), 1]
  let b : Fin 2 → ℝ := ![6, y]
  (∃ (k : ℝ), k ≠ 0 ∧ (∀ i, b i = k * a i)) → y = -2 := by
  sorry

end collinear_vectors_y_value_l2704_270407


namespace john_pills_per_week_l2704_270452

/-- The number of pills John takes in a week -/
def pills_per_week (hours_between_pills : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) : ℕ :=
  (hours_per_day / hours_between_pills) * days_per_week

/-- Theorem: John takes 28 pills in a week -/
theorem john_pills_per_week : 
  pills_per_week 6 24 7 = 28 := by
  sorry

end john_pills_per_week_l2704_270452


namespace expression_values_l2704_270436

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0) :
  let e := a / |a| + b / |b| + c / |c| + d / |d| + (a * b * c * d) / |a * b * c * d|
  e = 5 ∨ e = 1 ∨ e = -1 ∨ e = -5 :=
by sorry

end expression_values_l2704_270436


namespace equidistant_point_y_coordinate_l2704_270466

/-- The y-coordinate of the point on the y-axis equidistant from A(3, 0) and B(4, -3) is -8/3 -/
theorem equidistant_point_y_coordinate : 
  ∃ y : ℝ, 
    (3 - 0)^2 + (0 - y)^2 = (4 - 0)^2 + (-3 - y)^2 ∧ 
    y = -8/3 := by
  sorry

end equidistant_point_y_coordinate_l2704_270466


namespace neil_initial_games_neil_had_two_games_l2704_270444

theorem neil_initial_games (henry_initial : ℕ) (games_given : ℕ) (henry_neil_ratio : ℕ) : ℕ :=
  let henry_final := henry_initial - games_given
  let neil_final := henry_final / henry_neil_ratio
  neil_final - games_given

theorem neil_had_two_games : neil_initial_games 33 5 4 = 2 := by
  sorry

end neil_initial_games_neil_had_two_games_l2704_270444


namespace finance_club_probability_l2704_270496

theorem finance_club_probability (total_members : ℕ) (interested_ratio : ℚ) : 
  total_members = 20 →
  interested_ratio = 3 / 4 →
  let interested_members := (interested_ratio * total_members).num
  let not_interested_members := total_members - interested_members
  let prob_neither_interested := (not_interested_members / total_members) * ((not_interested_members - 1) / (total_members - 1))
  1 - prob_neither_interested = 18 / 19 := by
sorry

end finance_club_probability_l2704_270496


namespace line_parallel_perpendicular_l2704_270418

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Define the theorem
theorem line_parallel_perpendicular 
  (m n : Line) (α : Plane) :
  m ≠ n →
  parallel n m →
  perpendicular n α →
  perpendicular m α :=
sorry

end line_parallel_perpendicular_l2704_270418


namespace bottle_production_l2704_270482

/-- Given that 6 identical machines produce 300 bottles per minute at a constant rate,
    10 such machines will produce 2000 bottles in 4 minutes. -/
theorem bottle_production (machines : ℕ) (bottles_per_minute : ℕ) (time : ℕ) : 
  machines = 6 → bottles_per_minute = 300 → time = 4 →
  (10 : ℕ) * bottles_per_minute * time / machines = 2000 :=
by sorry

end bottle_production_l2704_270482


namespace min_value_inequality_l2704_270421

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : 1/a + 1/b = Real.sqrt 6) : 1/a^2 + 2/b^2 ≥ 4 := by
  sorry

end min_value_inequality_l2704_270421


namespace geometric_sequence_sum_l2704_270408

/-- Represents the sum of the first n terms of a geometric sequence -/
def S (n : ℕ) (a : ℕ → ℝ) : ℝ := sorry

/-- The common ratio of the geometric sequence -/
def q (a : ℕ → ℝ) : ℝ := sorry

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  S 4 a = -5 →
  S 6 a = 21 * S 2 a →
  S 8 a = -85 := by sorry

end geometric_sequence_sum_l2704_270408


namespace brad_books_this_month_l2704_270438

theorem brad_books_this_month (william_last_month : ℕ) (brad_last_month : ℕ) (william_total : ℕ) (brad_total : ℕ) :
  william_last_month = 6 →
  brad_last_month = 3 * william_last_month →
  william_total = brad_total + 4 →
  william_total = william_last_month + 2 * (brad_total - brad_last_month) →
  brad_total - brad_last_month = 16 := by
  sorry

end brad_books_this_month_l2704_270438


namespace definite_integral_reciprocal_cosine_squared_l2704_270469

theorem definite_integral_reciprocal_cosine_squared (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  ∫ x in (0)..(2 * Real.pi), 1 / (a + b * Real.cos x)^2 = (2 * Real.pi * a) / (a^2 - b^2)^(3/2) := by
  sorry

end definite_integral_reciprocal_cosine_squared_l2704_270469


namespace gumballs_per_package_l2704_270492

theorem gumballs_per_package (total_gumballs : ℕ) (total_boxes : ℕ) 
  (h1 : total_gumballs = 20) 
  (h2 : total_boxes = 4) 
  (h3 : total_gumballs > 0) 
  (h4 : total_boxes > 0) : 
  (total_gumballs / total_boxes : ℕ) = 5 := by
  sorry

end gumballs_per_package_l2704_270492


namespace sufficient_condition_range_l2704_270445

theorem sufficient_condition_range (a x : ℝ) : 
  (∀ x, (a ≤ x ∧ x < a + 2) → x ≤ -1) ∧ 
  (∃ x, x ≤ -1 ∧ ¬(a ≤ x ∧ x < a + 2)) →
  a ≤ -3 :=
sorry

end sufficient_condition_range_l2704_270445


namespace sarah_weed_pulling_l2704_270442

def tuesday_weeds : ℕ := 25

def wednesday_weeds : ℕ := 3 * tuesday_weeds

def thursday_weeds : ℕ := wednesday_weeds / 5

def friday_weeds : ℕ := thursday_weeds - 10

def total_weeds : ℕ := tuesday_weeds + wednesday_weeds + thursday_weeds + friday_weeds

theorem sarah_weed_pulling :
  total_weeds = 120 :=
sorry

end sarah_weed_pulling_l2704_270442


namespace average_of_one_eighth_and_one_sixth_l2704_270460

theorem average_of_one_eighth_and_one_sixth :
  (1 / 8 + 1 / 6) / 2 = 7 / 48 := by sorry

end average_of_one_eighth_and_one_sixth_l2704_270460


namespace calligraphy_supplies_problem_l2704_270412

/-- Represents the unit price of a brush in yuan -/
def brush_price : ℝ := 6

/-- Represents the unit price of rice paper in yuan -/
def paper_price : ℝ := 0.4

/-- Represents the maximum number of brushes that can be purchased -/
def max_brushes : ℕ := 50

/-- Theorem stating the solution to the calligraphy supplies problem -/
theorem calligraphy_supplies_problem :
  /- Given conditions -/
  (40 * brush_price + 100 * paper_price = 280) ∧
  (30 * brush_price + 200 * paper_price = 260) ∧
  (∀ m : ℕ, m ≤ 200 → 
    m * brush_price + (200 - m) * paper_price ≤ 360 → 
    m ≤ max_brushes) →
  /- Conclusion -/
  brush_price = 6 ∧ paper_price = 0.4 ∧ max_brushes = 50 :=
by sorry

end calligraphy_supplies_problem_l2704_270412


namespace janes_bagels_l2704_270471

theorem janes_bagels (b m : ℕ) : 
  b + m = 6 →
  (55 * b + 80 * m) % 100 = 0 →
  b = 0 := by
sorry

end janes_bagels_l2704_270471


namespace factorial_ratio_equals_fraction_l2704_270430

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_ratio_equals_fraction : 
  (factorial 6)^2 / (factorial 5 * factorial 7) = 100 / 101 := by
  sorry

end factorial_ratio_equals_fraction_l2704_270430


namespace complex_magnitude_for_specific_quadratic_l2704_270499

theorem complex_magnitude_for_specific_quadratic : 
  ∀ z : ℂ, z^2 - 6*z + 20 = 0 → Complex.abs z = Real.sqrt 20 := by
  sorry

end complex_magnitude_for_specific_quadratic_l2704_270499


namespace complex_multiplication_l2704_270410

theorem complex_multiplication : (1 + Complex.I) * (2 + Complex.I) * (3 + Complex.I) = 10 * Complex.I := by
  sorry

end complex_multiplication_l2704_270410


namespace baker_usual_bread_sales_l2704_270413

/-- Represents the baker's sales and pricing information -/
structure BakerSales where
  usual_pastries : ℕ
  usual_bread : ℕ
  today_pastries : ℕ
  today_bread : ℕ
  pastry_price : ℕ
  bread_price : ℕ

/-- Calculates the difference between usual sales and today's sales -/
def sales_difference (s : BakerSales) : ℤ :=
  (s.usual_pastries * s.pastry_price + s.usual_bread * s.bread_price) -
  (s.today_pastries * s.pastry_price + s.today_bread * s.bread_price)

/-- Theorem stating that given the conditions, the baker usually sells 34 loaves of bread -/
theorem baker_usual_bread_sales :
  ∀ (s : BakerSales),
    s.usual_pastries = 20 ∧
    s.today_pastries = 14 ∧
    s.today_bread = 25 ∧
    s.pastry_price = 2 ∧
    s.bread_price = 4 ∧
    sales_difference s = 48 →
    s.usual_bread = 34 :=
by
  sorry


end baker_usual_bread_sales_l2704_270413


namespace sum_of_roots_l2704_270439

theorem sum_of_roots (p q r : ℝ) 
  (hp : p^3 - 18*p^2 + 27*p - 72 = 0)
  (hq : 27*q^3 - 243*q^2 + 729*q - 972 = 0)
  (hr : 3*r = 9) : 
  p + q + r = 18 := by
  sorry

end sum_of_roots_l2704_270439


namespace shooting_probabilities_l2704_270459

/-- Represents the probability of hitting a specific ring -/
structure RingProbability where
  ring : Nat
  probability : Real

/-- Calculates the probability of hitting either the 10-ring or 9-ring -/
def prob_10_or_9 (probs : List RingProbability) : Real :=
  (probs.filter (fun p => p.ring == 10 || p.ring == 9)).map (fun p => p.probability) |>.sum

/-- Calculates the probability of hitting below the 7-ring -/
def prob_below_7 (probs : List RingProbability) : Real :=
  1 - (probs.map (fun p => p.probability) |>.sum)

/-- Theorem stating the probabilities for the given shooting scenario -/
theorem shooting_probabilities (probs : List RingProbability) 
  (h10 : RingProbability.mk 10 0.21 ∈ probs)
  (h9 : RingProbability.mk 9 0.23 ∈ probs)
  (h8 : RingProbability.mk 8 0.25 ∈ probs)
  (h7 : RingProbability.mk 7 0.28 ∈ probs)
  (h_no_other : ∀ p ∈ probs, p.ring ∈ [7, 8, 9, 10]) :
  prob_10_or_9 probs = 0.44 ∧ prob_below_7 probs = 0.03 := by
  sorry


end shooting_probabilities_l2704_270459


namespace cubic_function_extremum_l2704_270478

theorem cubic_function_extremum (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^3 - a*x^2 - b*x + a^2
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 - 2*a*x - b
  (f 1 = 10 ∧ f' 1 = 0) → ((a = -4 ∧ b = 11) ∨ (a = 3 ∧ b = -3)) := by
  sorry

end cubic_function_extremum_l2704_270478


namespace carl_watermelons_left_l2704_270467

/-- Calculates the number of watermelons left after a day of selling -/
def watermelons_left (price : ℕ) (profit : ℕ) (initial : ℕ) : ℕ :=
  initial - (profit / price)

/-- Theorem: Given the conditions, Carl has 18 watermelons left -/
theorem carl_watermelons_left :
  let price : ℕ := 3
  let profit : ℕ := 105
  let initial : ℕ := 53
  watermelons_left price profit initial = 18 := by
  sorry

end carl_watermelons_left_l2704_270467


namespace polynomial_division_theorem_l2704_270468

theorem polynomial_division_theorem (x : ℝ) : 
  ∃ (q r : ℝ), x^5 - 24*x^3 + 12*x^2 - x + 20 = (x - 3) * (x^4 + 3*x^3 - 15*x^2 - 33*x - 100) + (-280) := by
  sorry

end polynomial_division_theorem_l2704_270468


namespace dima_grade_and_instrument_l2704_270474

-- Define the students
inductive Student : Type
| Vasya : Student
| Dima : Student
| Kolya : Student
| Sergey : Student

-- Define the grades
inductive Grade : Type
| Fifth : Grade
| Sixth : Grade
| Seventh : Grade
| Eighth : Grade

-- Define the instruments
inductive Instrument : Type
| Saxophone : Instrument
| Keyboard : Instrument
| Drums : Instrument
| Guitar : Instrument

-- Define the assignment of grades and instruments to students
def grade_assignment : Student → Grade := sorry
def instrument_assignment : Student → Instrument := sorry

-- State the theorem
theorem dima_grade_and_instrument :
  (instrument_assignment Student.Vasya = Instrument.Saxophone) ∧
  (grade_assignment Student.Vasya ≠ Grade.Eighth) ∧
  (∃ s, grade_assignment s = Grade.Sixth ∧ instrument_assignment s = Instrument.Keyboard) ∧
  (∀ s, instrument_assignment s = Instrument.Drums → s ≠ Student.Dima) ∧
  (instrument_assignment Student.Sergey ≠ Instrument.Keyboard) ∧
  (grade_assignment Student.Sergey ≠ Grade.Fifth) ∧
  (grade_assignment Student.Dima ≠ Grade.Sixth) ∧
  (∀ s, instrument_assignment s = Instrument.Drums → grade_assignment s ≠ Grade.Eighth) →
  (grade_assignment Student.Dima = Grade.Eighth ∧ instrument_assignment Student.Dima = Instrument.Guitar) :=
by sorry


end dima_grade_and_instrument_l2704_270474


namespace expression_evaluation_l2704_270426

theorem expression_evaluation :
  let x : ℚ := -1/2
  (3 * x^4 - 2 * x^3) / (-x) - (x - x^2) * 3 * x = -1/4 := by
  sorry

end expression_evaluation_l2704_270426


namespace toys_problem_toys_problem_unique_l2704_270491

/-- Given the number of toys for Kamari, calculates the total number of toys for all three children. -/
def total_toys (kamari_toys : ℕ) : ℕ :=
  kamari_toys + (kamari_toys + 30) + (2 * kamari_toys)

/-- Theorem stating that given the conditions, the total number of toys is 290. -/
theorem toys_problem : ∃ (k : ℕ), k + (k + 30) = 160 ∧ total_toys k = 290 :=
  sorry

/-- Corollary: The solution to the problem exists and is unique. -/
theorem toys_problem_unique : ∃! (k : ℕ), k + (k + 30) = 160 ∧ total_toys k = 290 :=
  sorry

end toys_problem_toys_problem_unique_l2704_270491


namespace polynomial_roots_l2704_270417

theorem polynomial_roots : 
  let p (x : ℝ) := 3 * x^4 - 2 * x^3 - 4 * x^2 - 2 * x + 3
  ∀ x : ℝ, p x = 0 ↔ x = 1 ∨ x = -2 ∨ x = (-1/2 : ℝ) :=
by sorry

end polynomial_roots_l2704_270417


namespace least_positive_integer_with_remainders_l2704_270427

theorem least_positive_integer_with_remainders : ∃! b : ℕ+, 
  (b : ℕ) % 3 = 2 ∧ 
  (b : ℕ) % 4 = 3 ∧ 
  (b : ℕ) % 5 = 4 ∧ 
  (b : ℕ) % 6 = 5 ∧ 
  ∀ x : ℕ+, 
    (x : ℕ) % 3 = 2 → 
    (x : ℕ) % 4 = 3 → 
    (x : ℕ) % 5 = 4 → 
    (x : ℕ) % 6 = 5 → 
    b ≤ x :=
by sorry

end least_positive_integer_with_remainders_l2704_270427


namespace total_mass_of_water_l2704_270415

/-- The total mass of water in two glasses on an unequal-arm scale -/
theorem total_mass_of_water (L m l : ℝ) (hL : L > 0) (hm : m > 0) (hl : l ≠ 0) : ∃ total_mass : ℝ,
  (∃ m₁ m₂ l₁ : ℝ, 
    -- Initial balance condition
    m₁ * l₁ = m₂ * (L - l₁) ∧
    -- Balance condition after transfer
    (m₁ - m) * (l₁ + l) = (m₂ + m) * (L - l₁ - l) ∧
    -- Total mass definition
    total_mass = m₁ + m₂) ∧
  total_mass = m * L / l :=
sorry

end total_mass_of_water_l2704_270415


namespace remaining_money_l2704_270490

def initial_amount : ℕ := 100
def roast_cost : ℕ := 17
def vegetable_cost : ℕ := 11

theorem remaining_money :
  initial_amount - (roast_cost + vegetable_cost) = 72 := by
  sorry

end remaining_money_l2704_270490


namespace parallel_vectors_x_value_l2704_270465

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (x, -9)
  parallel a b → x = -6 := by
  sorry

end parallel_vectors_x_value_l2704_270465


namespace ant_climb_floors_l2704_270461

-- Define the problem parameters
def time_per_floor : ℕ := 15
def total_time : ℕ := 105
def start_floor : ℕ := 1

-- State the theorem
theorem ant_climb_floors :
  ∃ (final_floor : ℕ),
    final_floor = (total_time / time_per_floor) + start_floor ∧
    final_floor = 8 :=
by
  sorry

end ant_climb_floors_l2704_270461


namespace parentheses_value_l2704_270454

theorem parentheses_value : (6 : ℝ) / Real.sqrt 18 = Real.sqrt 2 := by
  sorry

end parentheses_value_l2704_270454


namespace triangle_side_altitude_sum_l2704_270419

theorem triangle_side_altitude_sum (x y : ℝ) : 
  x < 75 →
  y < 28 →
  x * 60 = 75 * 28 →
  100 * y = 75 * 28 →
  x + y = 56 := by
sorry

end triangle_side_altitude_sum_l2704_270419


namespace lisa_flight_time_l2704_270446

/-- Given a distance of 256 miles and a speed of 32 miles per hour, 
    the time taken to travel this distance is 8 hours. -/
theorem lisa_flight_time : 
  ∀ (distance speed time : ℝ), 
    distance = 256 → 
    speed = 32 → 
    time = distance / speed → 
    time = 8 := by sorry

end lisa_flight_time_l2704_270446


namespace constant_c_value_l2704_270437

theorem constant_c_value (b c : ℝ) : 
  (∀ x : ℝ, (x + 4) * (x + b) = x^2 + c*x + 12) → c = 7 := by
  sorry

end constant_c_value_l2704_270437


namespace sector_area_l2704_270457

theorem sector_area (arc_length : ℝ) (central_angle : ℝ) (h1 : arc_length = 6) (h2 : central_angle = 2) : 
  (1/2) * arc_length * (arc_length / central_angle) = 9 := by
sorry

end sector_area_l2704_270457


namespace quadratic_always_real_roots_implies_b_bound_l2704_270448

theorem quadratic_always_real_roots_implies_b_bound (b : ℝ) :
  (∀ a : ℝ, ∃ x : ℝ, x^2 - 2*a*x - a + 2*b = 0) →
  b ≤ -1/8 := by
sorry

end quadratic_always_real_roots_implies_b_bound_l2704_270448


namespace curve_fixed_point_l2704_270402

/-- The curve C: x^2 + y^2 + 2kx + (4k+10)y + 10k + 20 = 0 passes through the fixed point (1, -3) for all k ≠ -1 -/
theorem curve_fixed_point (k : ℝ) (h : k ≠ -1) :
  let C (x y : ℝ) := x^2 + y^2 + 2*k*x + (4*k+10)*y + 10*k + 20
  C 1 (-3) = 0 := by sorry

end curve_fixed_point_l2704_270402


namespace quadratic_equation_k_value_l2704_270464

theorem quadratic_equation_k_value (x₁ x₂ k : ℝ) : 
  x₁^2 - 3*x₁ + k = 0 →
  x₂^2 - 3*x₂ + k = 0 →
  x₁*x₂ + 2*x₁ + 2*x₂ = 1 →
  k = -5 :=
by sorry

end quadratic_equation_k_value_l2704_270464


namespace salt_concentration_proof_l2704_270480

/-- Proves that adding 66.67 gallons of 25% salt solution to 100 gallons of pure water results in a 10% salt solution -/
theorem salt_concentration_proof (initial_water : ℝ) (saline_volume : ℝ) (salt_percentage : ℝ) :
  initial_water = 100 →
  saline_volume = 66.67 →
  salt_percentage = 0.25 →
  (salt_percentage * saline_volume) / (initial_water + saline_volume) = 0.1 := by
  sorry

end salt_concentration_proof_l2704_270480


namespace solve_equation_l2704_270447

theorem solve_equation (a : ℚ) (h : a + a/4 - 1/2 = 10/5) : a = 2 := by
  sorry

end solve_equation_l2704_270447


namespace repeating_decimal_sum_l2704_270488

theorem repeating_decimal_sum : ∃ (a b : ℚ), 
  (∀ n : ℕ, a = 2 / 10^n + a / 10^n) ∧ 
  (∀ m : ℕ, b = 3 / 100^m + b / 100^m) ∧ 
  (a + b = 25 / 99) := by
sorry

end repeating_decimal_sum_l2704_270488


namespace two_a_minus_two_d_is_zero_l2704_270484

/-- Given a function g and constants a, b, c, d, prove that 2a - 2d = 0 -/
theorem two_a_minus_two_d_is_zero
  (a b c d : ℝ)
  (h_abcd : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (g : ℝ → ℝ)
  (h_g : ∀ x, g x = (2*a*x - b) / (c*x - 2*d))
  (h_inv : ∀ x, g (g x) = x) :
  2*a - 2*d = 0 := by
  sorry

end two_a_minus_two_d_is_zero_l2704_270484


namespace max_x_value_l2704_270456

theorem max_x_value (x : ℝ) : 
  ((4*x - 16)/(3*x - 4))^2 + (4*x - 16)/(3*x - 4) = 12 → x ≤ 2 :=
by sorry

end max_x_value_l2704_270456


namespace function_composition_equality_l2704_270495

theorem function_composition_equality (c : ℝ) : 
  let p : ℝ → ℝ := λ x => 4 * x - 9
  let q : ℝ → ℝ := λ x => 5 * x - c
  p (q 3) = 14 → c = 9.25 := by
sorry

end function_composition_equality_l2704_270495


namespace constant_term_of_product_l2704_270476

/-- The constant term in the expansion of (x^6 + x^2 + 3)(x^4 + x^3 + 20) is 60 -/
theorem constant_term_of_product (x : ℝ) : 
  (x^6 + x^2 + 3) * (x^4 + x^3 + 20) = x^10 + x^9 + 20*x^6 + x^7 + x^6 + 20*x^2 + 3*x^4 + 3*x^3 + 60 :=
by sorry

end constant_term_of_product_l2704_270476


namespace largest_y_floor_div_l2704_270497

theorem largest_y_floor_div : 
  ∀ y : ℝ, (↑(Int.floor y) / y = 8 / 9) → y ≤ 63 / 8 := by
  sorry

end largest_y_floor_div_l2704_270497


namespace part_one_part_two_l2704_270462

/-- Given c > 0 and c ≠ 1, define p and q as follows:
    p: The function y = c^x is monotonically decreasing
    q: The function f(x) = x^2 - 2cx + 1 is increasing on the interval (1/2, +∞) -/
def p (c : ℝ) : Prop := ∀ x y : ℝ, x < y → c^x > c^y
def q (c : ℝ) : Prop := ∀ x y : ℝ, 1/2 < x ∧ x < y → x^2 - 2*c*x + 1 < y^2 - 2*c*y + 1

/-- Part 1: If p is true and ¬q is false, then 0 < c ≤ 1/2 -/
theorem part_one (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1) (h3 : p c) (h4 : ¬¬(q c)) :
  0 < c ∧ c ≤ 1/2 := by sorry

/-- Part 2: If "p AND q" is false and "p OR q" is true, then 1/2 < c < 1 -/
theorem part_two (c : ℝ) (h1 : c > 0) (h2 : c ≠ 1) (h3 : ¬(p c ∧ q c)) (h4 : p c ∨ q c) :
  1/2 < c ∧ c < 1 := by sorry

end part_one_part_two_l2704_270462


namespace bob_show_dogs_count_l2704_270403

/-- The number of show dogs Bob bought -/
def num_show_dogs : ℕ := 2

/-- The cost of each show dog in dollars -/
def cost_per_show_dog : ℕ := 250

/-- The number of puppies -/
def num_puppies : ℕ := 6

/-- The selling price of each puppy in dollars -/
def price_per_puppy : ℕ := 350

/-- The total profit in dollars -/
def total_profit : ℕ := 1600

theorem bob_show_dogs_count :
  num_puppies * price_per_puppy - num_show_dogs * cost_per_show_dog = total_profit :=
by sorry

end bob_show_dogs_count_l2704_270403


namespace no_prime_between_100_110_congruent_3_mod_6_l2704_270489

theorem no_prime_between_100_110_congruent_3_mod_6 : ¬ ∃ n : ℕ, 
  Nat.Prime n ∧ 100 < n ∧ n < 110 ∧ n % 6 = 3 := by
  sorry

end no_prime_between_100_110_congruent_3_mod_6_l2704_270489


namespace zachary_purchase_l2704_270487

/-- The cost of items at a store -/
structure StorePrices where
  pencil : ℕ
  notebook : ℕ
  eraser : ℕ

/-- The conditions of the problem -/
def store_conditions (p : StorePrices) : Prop :=
  p.pencil + p.notebook = 80 ∧
  p.notebook + p.eraser = 85 ∧
  3 * p.pencil + 3 * p.notebook + 3 * p.eraser = 315

/-- The theorem to prove -/
theorem zachary_purchase (p : StorePrices) (h : store_conditions p) : 
  p.pencil + p.eraser = 45 := by
  sorry

end zachary_purchase_l2704_270487


namespace servant_service_duration_l2704_270422

/-- Calculates the number of months served given the total yearly payment and the received payment -/
def months_served (total_yearly_payment : ℚ) (received_payment : ℚ) : ℚ :=
  (received_payment / (total_yearly_payment / 12))

/-- Theorem stating that for the given payment conditions, the servant served approximately 6 months -/
theorem servant_service_duration :
  let total_yearly_payment : ℚ := 800
  let received_payment : ℚ := 400
  abs (months_served total_yearly_payment received_payment - 6) < 0.1 := by
sorry

end servant_service_duration_l2704_270422


namespace arithmetic_sequence_length_l2704_270405

/-- Given an arithmetic sequence with first term 6, last term 154, and common difference 4,
    prove that the number of terms is 38. -/
theorem arithmetic_sequence_length :
  ∀ (a : ℕ) (d : ℕ) (last : ℕ) (n : ℕ),
    a = 6 →
    d = 4 →
    last = 154 →
    last = a + (n - 1) * d →
    n = 38 := by
  sorry

end arithmetic_sequence_length_l2704_270405


namespace total_students_is_59_l2704_270481

/-- Represents a group of students with subgroups taking history and statistics -/
structure StudentGroup where
  total : ℕ
  history : ℕ
  statistics : ℕ
  both : ℕ
  history_only : ℕ
  history_or_statistics : ℕ

/-- The properties of the student group as described in the problem -/
def problem_group : StudentGroup where
  history := 36
  statistics := 32
  history_or_statistics := 59
  history_only := 27
  both := 36 - 27  -- Derived from history - history_only
  total := 59  -- This is what we want to prove

/-- Theorem stating that the total number of students in the group is 59 -/
theorem total_students_is_59 (g : StudentGroup) 
  (h1 : g.history = problem_group.history)
  (h2 : g.statistics = problem_group.statistics)
  (h3 : g.history_or_statistics = problem_group.history_or_statistics)
  (h4 : g.history_only = problem_group.history_only)
  (h5 : g.both = g.history - g.history_only)
  (h6 : g.history_or_statistics = g.history + g.statistics - g.both) :
  g.total = problem_group.total := by
  sorry

end total_students_is_59_l2704_270481


namespace golden_retriever_age_problem_l2704_270443

/-- The age of a golden retriever given its weight gain per year and current weight -/
def golden_retriever_age (weight_gain_per_year : ℕ) (current_weight : ℕ) : ℕ :=
  current_weight / weight_gain_per_year

/-- Theorem: The age of a golden retriever that gains 11 pounds each year and currently weighs 88 pounds is 8 years -/
theorem golden_retriever_age_problem :
  golden_retriever_age 11 88 = 8 := by
  sorry

end golden_retriever_age_problem_l2704_270443


namespace toy_cost_price_l2704_270434

/-- The cost price of a toy -/
def cost_price : ℕ := sorry

/-- The number of toys sold -/
def toys_sold : ℕ := 18

/-- The total selling price of all toys -/
def total_selling_price : ℕ := 23100

/-- The number of toys whose cost price equals the gain -/
def gain_equivalent_toys : ℕ := 3

theorem toy_cost_price : 
  (toys_sold + gain_equivalent_toys) * cost_price = total_selling_price ∧ 
  cost_price = 1100 := by
  sorry

end toy_cost_price_l2704_270434


namespace no_consecutive_sum_for_2004_l2704_270435

theorem no_consecutive_sum_for_2004 :
  ¬ ∃ (n : ℕ) (a : ℕ), n > 1 ∧ n * (2 * a + n - 1) = 4008 := by
  sorry

end no_consecutive_sum_for_2004_l2704_270435


namespace apple_preference_percentage_l2704_270449

def fruit_survey (apples bananas cherries oranges grapes : ℕ) : Prop :=
  let total := apples + bananas + cherries + oranges + grapes
  let apple_percentage := (apples : ℚ) / (total : ℚ) * 100
  apple_percentage = 26.67

theorem apple_preference_percentage :
  fruit_survey 80 90 50 40 40 := by
  sorry

end apple_preference_percentage_l2704_270449


namespace autumn_sales_l2704_270455

/-- Ice cream sales data for a city --/
structure IceCreamSales where
  spring : ℝ
  summer : ℝ
  autumn : ℝ
  winter : ℝ

/-- Theorem: Autumn ice cream sales calculation --/
theorem autumn_sales (sales : IceCreamSales) 
  (h1 : sales.spring = 3)
  (h2 : sales.summer = 6)
  (h3 : sales.winter = 5)
  (h4 : sales.spring = 0.2 * (sales.spring + sales.summer + sales.autumn + sales.winter)) :
  sales.autumn = 1 := by
  sorry

#check autumn_sales

end autumn_sales_l2704_270455


namespace sum_of_simplified_fraction_l2704_270473

-- Define the repeating decimal 0.̅4̅5̅
def repeating_decimal : ℚ := 45 / 99

-- Define the function to simplify a fraction
def simplify (q : ℚ) : ℚ := q

-- Define the function to sum numerator and denominator
def sum_num_denom (q : ℚ) : ℕ := q.num.natAbs + q.den

-- Theorem statement
theorem sum_of_simplified_fraction :
  sum_num_denom (simplify repeating_decimal) = 16 := by sorry

end sum_of_simplified_fraction_l2704_270473


namespace lcm_gcd_sum_inequality_l2704_270409

theorem lcm_gcd_sum_inequality (a b k : ℕ+) (hk : k > 1) 
  (h : Nat.lcm a b + Nat.gcd a b = k * (a + b)) : 
  a + b ≥ 4 * k := by
  sorry

end lcm_gcd_sum_inequality_l2704_270409


namespace race_speed_ratio_l2704_270400

/-- Race problem statement -/
theorem race_speed_ratio :
  ∀ (vA vB : ℝ) (d : ℝ),
  d > 0 →
  d / vA = 2 →
  d / vB = 1.5 →
  vA / vB = 3 / 4 :=
by
  sorry

end race_speed_ratio_l2704_270400


namespace coin_drawing_probability_l2704_270428

/-- The number of shiny coins in the box -/
def shiny_coins : ℕ := 3

/-- The number of dull coins in the box -/
def dull_coins : ℕ := 4

/-- The total number of coins in the box -/
def total_coins : ℕ := shiny_coins + dull_coins

/-- The probability of needing more than 4 draws to select all shiny coins -/
def prob_more_than_four_draws : ℚ := 31 / 35

theorem coin_drawing_probability :
  let p := 1 - (Nat.choose shiny_coins shiny_coins * 
    (Nat.choose dull_coins 1 * Nat.choose shiny_coins shiny_coins + 
    Nat.choose (total_coins - 1) 3)) / Nat.choose total_coins 4
  p = prob_more_than_four_draws := by sorry

end coin_drawing_probability_l2704_270428


namespace infinitely_many_increasing_largest_prime_factors_l2704_270494

/-- h(n) denotes the largest prime factor of the natural number n -/
def h (n : ℕ) : ℕ := sorry

/-- There exist infinitely many natural numbers n such that 
    the largest prime factor of n is less than the largest prime factor of n+1, 
    which is less than the largest prime factor of n+2 -/
theorem infinitely_many_increasing_largest_prime_factors :
  ∃ (S : Set ℕ), Set.Infinite S ∧ ∀ n ∈ S, h n < h (n + 1) ∧ h (n + 1) < h (n + 2) := by sorry

end infinitely_many_increasing_largest_prime_factors_l2704_270494


namespace redistribution_impossible_l2704_270411

/-- Represents the distribution of balls in boxes -/
structure BallDistribution where
  white_boxes : ℕ
  black_boxes : ℕ
  balls_per_white : ℕ
  balls_per_black : ℕ

/-- The initial distribution of balls -/
def initial_distribution : BallDistribution :=
  { white_boxes := 0,  -- We don't know the exact number, so we use 0
    black_boxes := 0,  -- We don't know the exact number, so we use 0
    balls_per_white := 31,
    balls_per_black := 26 }

/-- The distribution after adding 3 boxes -/
def new_distribution : BallDistribution :=
  { white_boxes := initial_distribution.white_boxes + 3,  -- Total boxes increased by 3
    black_boxes := initial_distribution.black_boxes,      -- Assuming all new boxes are white
    balls_per_white := 21,
    balls_per_black := 16 }

/-- The desired final distribution -/
def desired_distribution : BallDistribution :=
  { white_boxes := 0,  -- We don't know the exact number
    black_boxes := 0,  -- We don't know the exact number
    balls_per_white := 15,
    balls_per_black := 10 }

theorem redistribution_impossible :
  ∀ (final_distribution : BallDistribution),
  (final_distribution.balls_per_white = desired_distribution.balls_per_white ∧
   final_distribution.balls_per_black = desired_distribution.balls_per_black) →
  (final_distribution.white_boxes * final_distribution.balls_per_white +
   final_distribution.black_boxes * final_distribution.balls_per_black =
   new_distribution.white_boxes * new_distribution.balls_per_white +
   new_distribution.black_boxes * new_distribution.balls_per_black) →
  False :=
sorry

end redistribution_impossible_l2704_270411


namespace problem_statement_l2704_270485

theorem problem_statement (x y : ℚ) (hx : x = 5/6) (hy : y = 6/5) : 
  (1/3) * x^8 * y^9 = 2/5 := by sorry

end problem_statement_l2704_270485


namespace value_of_x_l2704_270425

theorem value_of_x (x y z : ℤ) 
  (eq1 : 4*x + y + z = 80) 
  (eq2 : 2*x - y - z = 40) 
  (eq3 : 3*x + y - z = 20) : 
  x = 20 := by
sorry

end value_of_x_l2704_270425


namespace mean_equality_problem_l2704_270414

theorem mean_equality_problem (y : ℚ) : 
  (5 + 10 + 20) / 3 = (15 + y) / 2 → y = 25 / 3 := by
  sorry

end mean_equality_problem_l2704_270414


namespace greatest_3digit_base8_divisible_by_7_l2704_270420

/-- Converts a base 8 number to decimal --/
def base8ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a decimal number to base 8 --/
def decimalToBase8 (n : ℕ) : ℕ := sorry

/-- Checks if a number is a 3-digit base 8 number --/
def isThreeDigitBase8 (n : ℕ) : Prop := 
  100 ≤ n ∧ n ≤ 777

theorem greatest_3digit_base8_divisible_by_7 :
  ∀ n : ℕ, isThreeDigitBase8 n → n ≤ 774 ∨ ¬(7 ∣ base8ToDecimal n) :=
by sorry

#check greatest_3digit_base8_divisible_by_7

end greatest_3digit_base8_divisible_by_7_l2704_270420


namespace plates_problem_l2704_270493

theorem plates_problem (total_days : ℕ) (plates_two_people : ℕ) (plates_four_people : ℕ) (total_plates : ℕ) :
  total_days = 7 →
  plates_two_people = 2 →
  plates_four_people = 8 →
  total_plates = 38 →
  ∃ (days_two_people : ℕ),
    days_two_people * plates_two_people + (total_days - days_two_people) * plates_four_people = total_plates ∧
    days_two_people = 3 :=
by sorry

end plates_problem_l2704_270493


namespace extreme_value_implies_a_minus_b_l2704_270463

-- Define the function f(x)
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 + b*x + a^2

-- State the theorem
theorem extreme_value_implies_a_minus_b (a b : ℝ) :
  (f a b (-1) = 0) ∧ 
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a b x ≥ f a b (-1)) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1 - ε) (-1 + ε), f a b x ≤ f a b (-1)) →
  a - b = -7 := by
  sorry


end extreme_value_implies_a_minus_b_l2704_270463


namespace percentage_problem_l2704_270433

theorem percentage_problem (x : ℝ) (h : 45 = 25 / 100 * x) : x = 180 := by
  sorry

end percentage_problem_l2704_270433


namespace range_of_n_over_m_l2704_270472

def A (m n : ℝ) := {z : ℂ | Complex.abs (z + n * Complex.I) + Complex.abs (z - m * Complex.I) = n}
def B (m n : ℝ) := {z : ℂ | Complex.abs (z + n * Complex.I) - Complex.abs (z - m * Complex.I) = -m}

theorem range_of_n_over_m (m n : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) 
  (hA : Set.Nonempty (A m n)) (hB : Set.Nonempty (B m n)) : 
  n / m ≤ -2 ∧ ∀ k : ℝ, ∃ m n : ℝ, m ≠ 0 ∧ n ≠ 0 ∧ Set.Nonempty (A m n) ∧ Set.Nonempty (B m n) ∧ n / m < k :=
sorry

end range_of_n_over_m_l2704_270472


namespace willie_initial_stickers_l2704_270440

/-- The number of stickers Willie gave to Emily -/
def stickers_given : ℕ := 7

/-- The number of stickers Willie had left after giving some to Emily -/
def stickers_left : ℕ := 29

/-- The initial number of stickers Willie had -/
def initial_stickers : ℕ := stickers_given + stickers_left

theorem willie_initial_stickers : initial_stickers = 36 := by
  sorry

end willie_initial_stickers_l2704_270440


namespace ellipse_and_triangle_properties_l2704_270458

/-- Represents an ellipse -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h : a > b ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line with slope 1 -/
def Line := { m : ℝ // ∀ x y, y = x + m }

theorem ellipse_and_triangle_properties
  (G : Ellipse)
  (e : ℝ)
  (F : Point)
  (l : Line)
  (P : Point)
  (he : e = Real.sqrt 6 / 3)
  (hF : F.x = 2 * Real.sqrt 2 ∧ F.y = 0)
  (hP : P.x = -3 ∧ P.y = 2)
  (h_isosceles : ∃ A B : Point, A ≠ B ∧
    (A.x - P.x)^2 + (A.y - P.y)^2 = (B.x - P.x)^2 + (B.y - P.y)^2 ∧
    ∃ t : ℝ, A.y = A.x + l.val ∧ B.y = B.x + l.val ∧
    A.x^2 / G.a^2 + A.y^2 / G.b^2 = 1 ∧
    B.x^2 / G.a^2 + B.y^2 / G.b^2 = 1) :
  G.a^2 = 12 ∧ G.b^2 = 4 ∧
  ∃ A B : Point, A ≠ B ∧
    (A.x - P.x)^2 + (A.y - P.y)^2 = (B.x - P.x)^2 + (B.y - P.y)^2 ∧
    ∃ t : ℝ, A.y = A.x + l.val ∧ B.y = B.x + l.val ∧
    A.x^2 / G.a^2 + A.y^2 / G.b^2 = 1 ∧
    B.x^2 / G.a^2 + B.y^2 / G.b^2 = 1 ∧
    (B.x - A.x) * (B.y - A.y) / 2 = 9/2 :=
by sorry

end ellipse_and_triangle_properties_l2704_270458
