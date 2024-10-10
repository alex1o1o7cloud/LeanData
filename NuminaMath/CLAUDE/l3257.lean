import Mathlib

namespace triangle_existence_l3257_325777

theorem triangle_existence (a b c d e : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) : 
  ∃ (x y z : ℝ), 
    x = Real.sqrt (b^2 + c^2 + d^2) ∧ 
    y = Real.sqrt (a^2 + b^2 + c^2 + e^2 + 2*a*c) ∧ 
    z = Real.sqrt (a^2 + d^2 + e^2 + 2*d*e) ∧ 
    x + y > z ∧ y + z > x ∧ z + x > y :=
sorry

end triangle_existence_l3257_325777


namespace trapezoid_max_segment_length_l3257_325780

/-- Given a trapezoid with sum of bases equal to 4, the maximum length of a segment
    passing through the intersection of diagonals and parallel to bases is 2. -/
theorem trapezoid_max_segment_length (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 4) :
  ∃ (s : ℝ), s ≤ 2 ∧ 
  ∀ (t : ℝ), (∃ (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 4),
    t = (2 * x * y) / (x + y)) → t ≤ s :=
by sorry

end trapezoid_max_segment_length_l3257_325780


namespace march_production_l3257_325739

/-- Represents the monthly production function -/
def production_function (x : ℝ) : ℝ := x + 1

/-- March is represented by the number 3 -/
def march : ℝ := 3

/-- Theorem stating that the estimated production for March is 4 -/
theorem march_production :
  production_function march = 4 := by sorry

end march_production_l3257_325739


namespace room_length_calculation_l3257_325703

theorem room_length_calculation (area : ℝ) (width : ℝ) (length : ℝ) :
  area = 10 ∧ width = 2 ∧ area = length * width → length = 5 := by
  sorry

end room_length_calculation_l3257_325703


namespace common_points_on_line_l3257_325731

-- Define the circles and line
def circle1 (a : ℝ) (x y : ℝ) : Prop := x^2 + (y - 1)^2 = a^2 ∧ a > 0
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4
def line (x y : ℝ) : Prop := y = 2*x

-- Define the theorem
theorem common_points_on_line (a : ℝ) : 
  (∀ x y : ℝ, circle1 a x y ∧ circle2 x y → line x y) → a = 1 := by
  sorry

end common_points_on_line_l3257_325731


namespace units_digit_of_n_l3257_325744

-- Define a function to get the units digit of a natural number
def unitsDigit (n : ℕ) : ℕ := n % 10

-- Define the problem statement
theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 21^6) (h2 : unitsDigit m = 7) :
  unitsDigit n = 3 := by
sorry

end units_digit_of_n_l3257_325744


namespace initial_marbles_equals_sum_l3257_325732

/-- The number of marbles Connie initially had -/
def initial_marbles : ℕ := sorry

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 73

/-- The number of marbles Connie has left -/
def marbles_left : ℕ := 70

/-- Theorem stating that the initial number of marbles equals the sum of marbles given away and marbles left -/
theorem initial_marbles_equals_sum : initial_marbles = marbles_given + marbles_left := by sorry

end initial_marbles_equals_sum_l3257_325732


namespace multiply_and_distribute_l3257_325789

theorem multiply_and_distribute (a b : ℝ) : -a * b * (-b + 1) = a * b^2 - a * b := by
  sorry

end multiply_and_distribute_l3257_325789


namespace largest_multiple_of_11_below_negative_85_l3257_325795

theorem largest_multiple_of_11_below_negative_85 :
  ∀ n : ℤ, n % 11 = 0 → n < -85 → n ≤ -88 :=
by
  sorry

end largest_multiple_of_11_below_negative_85_l3257_325795


namespace race_distance_proof_l3257_325718

/-- The length of the race track in feet -/
def track_length : ℕ := 5000

/-- The distance Alex and Max run evenly at the start -/
def even_start : ℕ := 200

/-- The distance Alex gets ahead after the even start -/
def alex_first_lead : ℕ := 300

/-- The distance Alex gets ahead at the end -/
def alex_final_lead : ℕ := 440

/-- The distance left for Max to catch up at the end -/
def max_remaining : ℕ := 3890

/-- The unknown distance Max gets ahead of Alex -/
def max_lead : ℕ := 170

theorem race_distance_proof :
  even_start + alex_first_lead + max_lead + alex_final_lead = track_length - max_remaining :=
by sorry

end race_distance_proof_l3257_325718


namespace f_2007_equals_negative_two_l3257_325792

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

theorem f_2007_equals_negative_two (f : ℝ → ℝ) 
  (h1 : isEven f) 
  (h2 : ∀ x, f (2 + x) = f (2 - x)) 
  (h3 : f (-3) = -2) : 
  f 2007 = -2 := by
  sorry

end f_2007_equals_negative_two_l3257_325792


namespace floor_plus_double_eq_15_4_l3257_325768

theorem floor_plus_double_eq_15_4 :
  ∃! r : ℝ, ⌊r⌋ + 2 * r = 15.4 ∧ r = 5.2 :=
by sorry

end floor_plus_double_eq_15_4_l3257_325768


namespace box_height_is_55cm_l3257_325730

/-- The height of the box Bob needs to reach the light fixture -/
def box_height (ceiling_height light_fixture_distance bob_height bob_reach : ℝ) : ℝ :=
  ceiling_height - light_fixture_distance - (bob_height + bob_reach)

/-- Theorem stating the height of the box Bob needs -/
theorem box_height_is_55cm :
  let ceiling_height : ℝ := 300
  let light_fixture_distance : ℝ := 15
  let bob_height : ℝ := 180
  let bob_reach : ℝ := 50
  box_height ceiling_height light_fixture_distance bob_height bob_reach = 55 := by
  sorry

#eval box_height 300 15 180 50

end box_height_is_55cm_l3257_325730


namespace cosine_value_l3257_325712

theorem cosine_value (α : Real) 
  (h : Real.sin (π / 6 - α) = 5 / 13) : 
  Real.cos (π / 3 + α) = 5 / 13 := by
  sorry

end cosine_value_l3257_325712


namespace expression_evaluation_equation_solutions_l3257_325741

-- Part 1
theorem expression_evaluation :
  |Real.sqrt 3 - 1| - 2 * Real.cos (60 * π / 180) + (Real.sqrt 3 - 2)^2 + Real.sqrt 12 = 5 - Real.sqrt 3 := by
  sorry

-- Part 2
theorem equation_solutions (x : ℝ) :
  2 * (x - 3)^2 = x^2 - 9 ↔ x = 3 ∨ x = 9 := by
  sorry

end expression_evaluation_equation_solutions_l3257_325741


namespace mod_eight_thirteen_fourth_l3257_325769

theorem mod_eight_thirteen_fourth (m : ℕ) : 
  13^4 ≡ m [ZMOD 8] → 0 ≤ m → m < 8 → m = 1 := by
  sorry

end mod_eight_thirteen_fourth_l3257_325769


namespace fraction_of_loss_example_l3257_325711

/-- Calculates the fraction of loss given the cost price and selling price -/
def fractionOfLoss (costPrice sellingPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice

/-- Theorem: The fraction of loss for an item with cost price 18 and selling price 17 is 1/18 -/
theorem fraction_of_loss_example : fractionOfLoss 18 17 = 1 / 18 := by
  sorry

end fraction_of_loss_example_l3257_325711


namespace star_card_probability_l3257_325735

theorem star_card_probability (total_cards : ℕ) (num_ranks : ℕ) (num_suits : ℕ) 
  (h1 : total_cards = 65)
  (h2 : num_ranks = 13)
  (h3 : num_suits = 5)
  (h4 : total_cards = num_ranks * num_suits) :
  (num_ranks : ℚ) / total_cards = 1 / 5 := by
  sorry

end star_card_probability_l3257_325735


namespace original_number_proof_l3257_325771

theorem original_number_proof : 
  ∃ (x : ℕ), x = 6 ∧ 
  (∀ (y : ℕ), y < x → ¬(25 ∣ (y + 19))) ∧ 
  (25 ∣ (x + 19)) := by
  sorry

end original_number_proof_l3257_325771


namespace polynomial_roots_l3257_325785

theorem polynomial_roots (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^4 + 2*p*x^3 - x^2 + 2*p*x + 1 = 0 ∧ 
    y^4 + 2*p*y^3 - y^2 + 2*p*y + 1 = 0) ↔ 
  -3/4 ≤ p ∧ p ≤ -1/4 :=
by sorry

end polynomial_roots_l3257_325785


namespace total_figures_is_44_l3257_325714

/-- The number of action figures that can fit on each shelf. -/
def figures_per_shelf : ℕ := 11

/-- The number of shelves in Adam's room. -/
def number_of_shelves : ℕ := 4

/-- The total number of action figures that can fit on all shelves. -/
def total_figures : ℕ := figures_per_shelf * number_of_shelves

/-- Theorem stating that the total number of action figures is 44. -/
theorem total_figures_is_44 : total_figures = 44 := by
  sorry

end total_figures_is_44_l3257_325714


namespace f_properties_l3257_325793

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x < 0 then -x^2 - 4*x - 3
  else if x = 0 then 0
  else x^2 - 4*x + 3

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x > 0, f x = x^2 - 4*x + 3) →  -- given condition for x > 0
  (f (f (-1)) = 0) ∧  -- part 1
  (∀ x, f x = if x < 0 then -x^2 - 4*x - 3
              else if x = 0 then 0
              else x^2 - 4*x + 3) :=  -- part 2
by sorry

end f_properties_l3257_325793


namespace tan_alpha_2_implies_ratio_3_l3257_325737

theorem tan_alpha_2_implies_ratio_3 (α : Real) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 := by
  sorry

end tan_alpha_2_implies_ratio_3_l3257_325737


namespace triangle_centroid_inequality_l3257_325738

/-- Given a triangle ABC with side lengths a, b, and c, centroid G, and an arbitrary point P,
    prove that a⋅PA³ + b⋅PB³ + c⋅PC³ ≥ 3abc⋅PG -/
theorem triangle_centroid_inequality (A B C P : ℝ × ℝ) 
    (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  let G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  let PA := Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2)
  let PB := Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2)
  let PC := Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2)
  let PG := Real.sqrt ((P.1 - G.1)^2 + (P.2 - G.2)^2)
  a * PA^3 + b * PB^3 + c * PC^3 ≥ 3 * a * b * c * PG := by
sorry


end triangle_centroid_inequality_l3257_325738


namespace modular_inverse_34_mod_35_l3257_325728

theorem modular_inverse_34_mod_35 : ∃ x : ℕ, x ≤ 34 ∧ (34 * x) % 35 = 1 := by
  sorry

end modular_inverse_34_mod_35_l3257_325728


namespace other_integer_17_or_21_l3257_325721

/-- Two consecutive odd integers with a sum of at least 36, one being 19 -/
structure ConsecutiveOddIntegers where
  n : ℤ
  sum_at_least_36 : n + (n + 2) ≥ 36
  one_is_19 : n = 19 ∨ n + 2 = 19

/-- The other integer is either 17 or 21 -/
theorem other_integer_17_or_21 (x : ConsecutiveOddIntegers) : 
  x.n = 21 ∨ x.n = 17 := by
  sorry


end other_integer_17_or_21_l3257_325721


namespace tyrone_nickels_l3257_325765

/-- Represents the contents of Tyrone's piggy bank -/
structure PiggyBank where
  one_dollar_bills : Nat
  five_dollar_bills : Nat
  quarters : Nat
  dimes : Nat
  nickels : Nat
  pennies : Nat

/-- Calculates the total value in dollars of the contents of the piggy bank -/
def total_value (pb : PiggyBank) : Rat :=
  pb.one_dollar_bills + 
  5 * pb.five_dollar_bills + 
  (1/4) * pb.quarters + 
  (1/10) * pb.dimes + 
  (1/20) * pb.nickels + 
  (1/100) * pb.pennies

/-- Tyrone's piggy bank contents -/
def tyrone_piggy_bank : PiggyBank :=
  { one_dollar_bills := 2
  , five_dollar_bills := 1
  , quarters := 13
  , dimes := 20
  , nickels := 8  -- This is what we want to prove
  , pennies := 35 }

theorem tyrone_nickels : 
  total_value tyrone_piggy_bank = 13 := by sorry

end tyrone_nickels_l3257_325765


namespace mailman_delivery_l3257_325764

/-- Represents the different types of mail delivered by the mailman -/
structure MailDelivery where
  junkMail : ℕ
  magazines : ℕ
  newspapers : ℕ
  bills : ℕ
  postcards : ℕ

/-- Calculates the total number of mail pieces delivered -/
def totalMail (delivery : MailDelivery) : ℕ :=
  delivery.junkMail + delivery.magazines + delivery.newspapers + delivery.bills + delivery.postcards

/-- Theorem stating that the total mail delivered is 20 pieces -/
theorem mailman_delivery :
  ∃ (delivery : MailDelivery),
    delivery.junkMail = 6 ∧
    delivery.magazines = 5 ∧
    delivery.newspapers = 3 ∧
    delivery.bills = 4 ∧
    delivery.postcards = 2 ∧
    totalMail delivery = 20 := by
  sorry

end mailman_delivery_l3257_325764


namespace stop_signs_per_mile_l3257_325710

-- Define the distance traveled
def distance : ℝ := 5 + 2

-- Define the number of stop signs encountered
def stop_signs : ℕ := 17 - 3

-- Theorem to prove
theorem stop_signs_per_mile : (stop_signs : ℝ) / distance = 2 := by
  sorry

end stop_signs_per_mile_l3257_325710


namespace notebook_cost_l3257_325727

/-- Given the following conditions:
  * Total spent on school supplies is $32
  * A backpack costs $15
  * A pack of pens costs $1
  * A pack of pencils costs $1
  * 5 multi-subject notebooks were bought
Prove that each notebook costs $3 -/
theorem notebook_cost (total_spent : ℚ) (backpack_cost : ℚ) (pen_cost : ℚ) (pencil_cost : ℚ) (notebook_count : ℕ) :
  total_spent = 32 →
  backpack_cost = 15 →
  pen_cost = 1 →
  pencil_cost = 1 →
  notebook_count = 5 →
  (total_spent - backpack_cost - pen_cost - pencil_cost) / notebook_count = 3 := by
  sorry

#check notebook_cost

end notebook_cost_l3257_325727


namespace table_height_l3257_325781

/-- Given three rectangular boxes (blue, red, and green) and their height relationships
    with a table, prove that the height of the table is 91 cm. -/
theorem table_height
  (h b r g : ℝ)
  (eq1 : h + b - g = 111)
  (eq2 : h + r - b = 80)
  (eq3 : h + g - r = 82) :
  h = 91 := by
sorry

end table_height_l3257_325781


namespace carters_dog_height_l3257_325753

-- Define heights in inches
def betty_height : ℕ := 3 * 12  -- 3 feet converted to inches
def carter_height : ℕ := betty_height + 12
def dog_height : ℕ := carter_height / 2

-- Theorem statement
theorem carters_dog_height : dog_height = 24 := by
  sorry

end carters_dog_height_l3257_325753


namespace plain_pancakes_count_l3257_325763

theorem plain_pancakes_count (total : ℕ) (blueberry : ℕ) (banana : ℕ) 
  (h1 : total = 67) (h2 : blueberry = 20) (h3 : banana = 24) : 
  total - (blueberry + banana) = 23 := by
  sorry

end plain_pancakes_count_l3257_325763


namespace rings_per_game_l3257_325778

theorem rings_per_game (total_rings : ℕ) (num_games : ℕ) (rings_per_game : ℕ) 
  (h1 : total_rings = 48) 
  (h2 : num_games = 8) 
  (h3 : total_rings = num_games * rings_per_game) : 
  rings_per_game = 6 := by
  sorry

end rings_per_game_l3257_325778


namespace r_fraction_of_total_l3257_325729

theorem r_fraction_of_total (total : ℚ) (r_amount : ℚ) 
  (h1 : total = 4000)
  (h2 : r_amount = 1600) :
  r_amount / total = 2 / 5 := by
sorry

end r_fraction_of_total_l3257_325729


namespace fifteen_times_fifteen_l3257_325782

theorem fifteen_times_fifteen : 
  ∀ n : ℕ, n = 15 → 15 * n = 225 := by
  sorry

end fifteen_times_fifteen_l3257_325782


namespace points_deducted_for_incorrect_l3257_325779

def test_questions : ℕ := 30
def correct_answer_points : ℕ := 20
def maria_final_score : ℕ := 325
def maria_correct_answers : ℕ := 19

theorem points_deducted_for_incorrect (deducted_points : ℕ) : 
  (maria_correct_answers * correct_answer_points) - 
  ((test_questions - maria_correct_answers) * deducted_points) = 
  maria_final_score → 
  deducted_points = 5 := by
sorry

end points_deducted_for_incorrect_l3257_325779


namespace coefficient_x_squared_is_80_l3257_325762

/-- The coefficient of x^2 in the expansion of (2x + 1/x^2)^5 -/
def coefficient_x_squared : ℕ :=
  (Nat.choose 5 1) * (2^4)

/-- Theorem stating that the coefficient of x^2 in the expansion of (2x + 1/x^2)^5 is 80 -/
theorem coefficient_x_squared_is_80 : coefficient_x_squared = 80 := by
  sorry

end coefficient_x_squared_is_80_l3257_325762


namespace greatest_x_value_l3257_325724

theorem greatest_x_value (a b c d : ℤ) (x : ℝ) :
  x = (a + b * Real.sqrt c) / d →
  (7 * x) / 9 + 1 = 3 / x →
  (∀ y : ℝ, (7 * y) / 9 + 1 = 3 / y → y ≤ x) →
  a * c * d / b = -4158 := by
  sorry

end greatest_x_value_l3257_325724


namespace f_is_locally_odd_l3257_325757

/-- Definition of a locally odd function -/
def LocallyOdd (f : ℝ → ℝ) : Prop :=
  ∃ x : ℝ, f (-x) = -f x

/-- The quadratic function we're examining -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 2 * x - 4 * a

/-- Theorem: The function f is locally odd for any real a -/
theorem f_is_locally_odd (a : ℝ) : LocallyOdd (f a) := by
  sorry


end f_is_locally_odd_l3257_325757


namespace ratio_p_to_r_l3257_325749

theorem ratio_p_to_r (p q r s : ℚ) 
  (h1 : p / q = 5 / 4)
  (h2 : r / s = 4 / 3)
  (h3 : s / q = 1 / 5) :
  p / r = 75 / 16 := by
sorry

end ratio_p_to_r_l3257_325749


namespace joes_pocket_money_l3257_325783

theorem joes_pocket_money (initial_money : ℚ) : 
  (initial_money * (1 - (1/9 + 2/5)) = 220) → initial_money = 450 := by
  sorry

end joes_pocket_money_l3257_325783


namespace difference_of_squares_example_l3257_325748

theorem difference_of_squares_example : (17 + 10)^2 - (17 - 10)^2 = 680 := by
  sorry

end difference_of_squares_example_l3257_325748


namespace cody_reading_time_l3257_325747

def read_series (total_books : ℕ) (first_week : ℕ) (second_week : ℕ) (subsequent_weeks : ℕ) : ℕ :=
  let books_first_two_weeks := first_week + second_week
  let remaining_books := total_books - books_first_two_weeks
  let additional_weeks := (remaining_books + subsequent_weeks - 1) / subsequent_weeks
  2 + additional_weeks

theorem cody_reading_time :
  read_series 54 6 3 9 = 7 := by
  sorry

end cody_reading_time_l3257_325747


namespace charging_time_is_112_5_l3257_325743

/-- Represents the charging time for each device type -/
structure ChargingTimes where
  smartphone : ℝ
  tablet : ℝ
  laptop : ℝ

/-- Represents the charging percentages for each device -/
structure ChargingPercentages where
  smartphone : ℝ
  tablet : ℝ
  laptop : ℝ

/-- Calculates the total charging time given the full charging times and charging percentages -/
def totalChargingTime (times : ChargingTimes) (percentages : ChargingPercentages) : ℝ :=
  times.tablet * percentages.tablet +
  times.smartphone * percentages.smartphone +
  times.laptop * percentages.laptop

/-- Theorem stating that the total charging time is 112.5 minutes -/
theorem charging_time_is_112_5 (times : ChargingTimes) (percentages : ChargingPercentages) :
  times.smartphone = 26 →
  times.tablet = 53 →
  times.laptop = 80 →
  percentages.smartphone = 0.75 →
  percentages.tablet = 1 →
  percentages.laptop = 0.5 →
  totalChargingTime times percentages = 112.5 := by
  sorry

end charging_time_is_112_5_l3257_325743


namespace simplify_polynomial_l3257_325752

theorem simplify_polynomial (w : ℝ) : 
  3*w + 4 - 6*w - 5 + 7*w + 8 - 9*w - 10 + 2*w^2 = 2*w^2 - 5*w - 3 := by
  sorry

end simplify_polynomial_l3257_325752


namespace not_perfect_square_l3257_325726

theorem not_perfect_square (n : ℤ) (h : n > 11) :
  ¬ ∃ m : ℤ, n^2 - 19*n + 89 = m^2 := by
  sorry

end not_perfect_square_l3257_325726


namespace marble_probability_l3257_325740

/-- The probability of drawing a red, blue, or green marble from a bag -/
theorem marble_probability (red blue green yellow : ℕ) : 
  red = 4 → blue = 3 → green = 2 → yellow = 6 → 
  (red + blue + green : ℚ) / (red + blue + green + yellow) = 0.6 := by
sorry

end marble_probability_l3257_325740


namespace quadratic_root_difference_sum_l3257_325717

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := 5 * x^2 - 7 * x - 10 = 0

-- Define the condition for m (positive integer not divisible by the square of any prime)
def is_squarefree (m : ℕ) : Prop :=
  m > 0 ∧ ∀ p : ℕ, Prime p → (p^2 ∣ m → False)

-- Main theorem
theorem quadratic_root_difference_sum (m n : ℤ) : 
  (∃ r₁ r₂ : ℝ, quadratic_equation r₁ ∧ quadratic_equation r₂ ∧ |r₁ - r₂| = (Real.sqrt (m : ℝ)) / (n : ℝ)) →
  is_squarefree (m.natAbs) →
  m + n = 254 := by
  sorry


end quadratic_root_difference_sum_l3257_325717


namespace cone_volume_from_circle_sector_l3257_325723

theorem cone_volume_from_circle_sector (r : ℝ) (h : r = 6) :
  let sector_fraction : ℝ := 5 / 8
  let base_radius : ℝ := r * sector_fraction / 2
  let height : ℝ := Real.sqrt (r^2 - base_radius^2)
  let volume : ℝ := (1 / 3) * Real.pi * base_radius^2 * height
  volume = 4.6875 * Real.pi * Real.sqrt 21.9375 := by
sorry

end cone_volume_from_circle_sector_l3257_325723


namespace carls_flowerbed_area_l3257_325734

/-- Represents a rectangular flowerbed with fencing --/
structure Flowerbed where
  short_posts : ℕ  -- Number of posts on the shorter side (including corners)
  long_posts : ℕ   -- Number of posts on the longer side (including corners)
  post_spacing : ℕ -- Spacing between posts in yards

/-- Calculates the area of the flowerbed --/
def Flowerbed.area (fb : Flowerbed) : ℕ :=
  (fb.short_posts - 1) * (fb.long_posts - 1) * fb.post_spacing * fb.post_spacing

/-- Theorem stating the area of Carl's flowerbed --/
theorem carls_flowerbed_area :
  ∃ fb : Flowerbed,
    fb.short_posts + fb.long_posts = 13 ∧
    fb.long_posts = 3 * fb.short_posts - 2 ∧
    fb.post_spacing = 3 ∧
    fb.area = 144 := by
  sorry

end carls_flowerbed_area_l3257_325734


namespace umbrella_count_l3257_325713

theorem umbrella_count (y b r : ℕ) 
  (h1 : b = (y + r) / 2)
  (h2 : r = (y + b) / 3)
  (h3 : y = 45) :
  b = 36 ∧ r = 27 := by
  sorry

end umbrella_count_l3257_325713


namespace max_distance_is_25km_l3257_325775

def car_position (t : ℝ) : ℝ := 40 * t

def motorcycle_position (t : ℝ) : ℝ := 16 * t^2 + 9

def distance (t : ℝ) : ℝ := |motorcycle_position t - car_position t|

theorem max_distance_is_25km :
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 2 ∧
  ∀ s : ℝ, 0 ≤ s ∧ s ≤ 2 → distance t ≥ distance s ∧
  distance t = 25 := by
  sorry

end max_distance_is_25km_l3257_325775


namespace max_value_sum_of_fractions_l3257_325756

theorem max_value_sum_of_fractions (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a + b + c = 3) :
  (a * b) / (a + b) + (a * c) / (a + c) + (b * c) / (b + c) ≤ 9 / 4 :=
by sorry

end max_value_sum_of_fractions_l3257_325756


namespace neg_three_at_neg_two_l3257_325754

-- Define the "@" operation
def at_op (x y : ℤ) : ℤ := x * y - y

-- Theorem statement
theorem neg_three_at_neg_two : at_op (-3) (-2) = 8 := by
  sorry

end neg_three_at_neg_two_l3257_325754


namespace sum_of_numbers_with_given_hcf_and_lcm_factors_l3257_325787

theorem sum_of_numbers_with_given_hcf_and_lcm_factors
  (a b : ℕ+)
  (h_hcf : Nat.gcd a b = 23)
  (h_lcm : Nat.lcm a b = 81328) :
  a + b = 667 := by
  sorry

end sum_of_numbers_with_given_hcf_and_lcm_factors_l3257_325787


namespace complex_number_in_first_quadrant_l3257_325708

theorem complex_number_in_first_quadrant (z : ℂ) : 
  z / (z - Complex.I) = Complex.I → 
  (z.re > 0 ∧ z.im > 0) := by
  sorry

end complex_number_in_first_quadrant_l3257_325708


namespace yellow_beans_percentage_approx_32_percent_l3257_325701

def bag1_total : ℕ := 24
def bag2_total : ℕ := 32
def bag3_total : ℕ := 34

def bag1_yellow_percent : ℚ := 40 / 100
def bag2_yellow_percent : ℚ := 30 / 100
def bag3_yellow_percent : ℚ := 25 / 100

def total_beans : ℕ := bag1_total + bag2_total + bag3_total

def yellow_beans : ℚ := 
  bag1_total * bag1_yellow_percent + 
  bag2_total * bag2_yellow_percent + 
  bag3_total * bag3_yellow_percent

def mixed_yellow_percent : ℚ := yellow_beans / total_beans

theorem yellow_beans_percentage_approx_32_percent :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/100 ∧ |mixed_yellow_percent - 32/100| < ε :=
sorry

end yellow_beans_percentage_approx_32_percent_l3257_325701


namespace sum_of_first_five_terms_l3257_325722

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_of_first_five_terms
  (a : ℕ → ℤ)
  (h_seq : arithmetic_sequence a)
  (h_4th : a 4 = 11)
  (h_5th : a 5 = 15)
  (h_6th : a 6 = 19) :
  a 1 + a 2 + a 3 + a 4 + a 5 = 35 :=
sorry

end sum_of_first_five_terms_l3257_325722


namespace bowen_purchase_ratio_l3257_325733

/-- Represents the purchase of pens and pencils -/
structure Purchase where
  pen_price : ℚ
  pencil_price : ℚ
  num_pens : ℕ
  total_spent : ℚ

/-- Calculates the ratio of pencils to pens for a given purchase -/
def pencil_to_pen_ratio (p : Purchase) : ℚ × ℚ :=
  let pencil_cost := p.total_spent - p.pen_price * p.num_pens
  let num_pencils := pencil_cost / p.pencil_price
  let gcd := Nat.gcd (Nat.floor num_pencils) p.num_pens
  ((num_pencils / gcd), (p.num_pens / gcd))

/-- Theorem stating that for the given purchase conditions, the ratio of pencils to pens is 7:5 -/
theorem bowen_purchase_ratio : 
  let p : Purchase := {
    pen_price := 15/100,
    pencil_price := 25/100,
    num_pens := 40,
    total_spent := 20
  }
  pencil_to_pen_ratio p = (7, 5) := by sorry

end bowen_purchase_ratio_l3257_325733


namespace jimmy_passing_points_l3257_325760

def points_per_exam : ℕ := 20
def number_of_exams : ℕ := 3
def points_lost_for_behavior : ℕ := 5
def additional_points_can_lose : ℕ := 5

def points_to_pass : ℕ := 50

theorem jimmy_passing_points :
  points_to_pass = 
    points_per_exam * number_of_exams - 
    points_lost_for_behavior - 
    additional_points_can_lose :=
by
  sorry

end jimmy_passing_points_l3257_325760


namespace greatest_valid_integer_l3257_325709

def is_valid (n : ℕ) : Prop :=
  n < 150 ∧ Nat.gcd n 30 = 5

theorem greatest_valid_integer : 
  (∀ m : ℕ, is_valid m → m ≤ 125) ∧ is_valid 125 :=
by sorry

end greatest_valid_integer_l3257_325709


namespace range_of_product_l3257_325705

def f (x : ℝ) := |x^2 + 2*x - 1|

theorem range_of_product (a b : ℝ) 
  (h1 : a < b) (h2 : b < -1) (h3 : f a = f b) :
  ∃ y, y ∈ Set.Ioo 0 2 ∧ y = (a + 1) * (b + 1) :=
sorry

end range_of_product_l3257_325705


namespace favorite_song_probability_l3257_325786

/-- Represents a digital music player with a collection of songs. -/
structure MusicPlayer where
  numSongs : Nat
  shortestSongDuration : Nat
  durationIncrement : Nat
  favoriteSongDuration : Nat
  playbackDuration : Nat

/-- Calculates the probability of not hearing the favorite song in full 
    within the given playback duration. -/
def probabilityNoFavoriteSong (player : MusicPlayer) : Rat :=
  sorry

/-- Theorem stating the probability of not hearing the favorite song in full
    for the specific music player configuration. -/
theorem favorite_song_probability (player : MusicPlayer) 
  (h1 : player.numSongs = 12)
  (h2 : player.shortestSongDuration = 40)
  (h3 : player.durationIncrement = 40)
  (h4 : player.favoriteSongDuration = 300)
  (h5 : player.playbackDuration = 360) :
  probabilityNoFavoriteSong player = 43 / 48 := by
  sorry

end favorite_song_probability_l3257_325786


namespace parabola_directrix_l3257_325716

/-- Given a parabola with equation y² = 6x, its directrix equation is x = -3/2 -/
theorem parabola_directrix (x y : ℝ) : 
  (y^2 = 6*x) → (∃ (k : ℝ), k = -3/2 ∧ x = k) := by
  sorry

end parabola_directrix_l3257_325716


namespace family_suitcases_l3257_325715

theorem family_suitcases (num_siblings : ℕ) (suitcases_per_sibling : ℕ) (parent_suitcases : ℕ) : 
  num_siblings = 4 →
  suitcases_per_sibling = 2 →
  parent_suitcases = 3 →
  num_siblings * suitcases_per_sibling + parent_suitcases * 2 = 14 := by
  sorry

end family_suitcases_l3257_325715


namespace triangle_max_area_l3257_325772

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions and theorem
theorem triangle_max_area (t : Triangle) 
  (h1 : Real.sin t.A + Real.sqrt 2 * Real.sin t.B = 2 * Real.sin t.C)
  (h2 : t.b = 3) :
  ∃ (max_area : ℝ), max_area = (9 + 3 * Real.sqrt 3) / 4 ∧ 
    ∀ (area : ℝ), area ≤ max_area := by
  sorry

end triangle_max_area_l3257_325772


namespace swan_population_after_ten_years_l3257_325784

/-- The number of swans after a given number of years, given an initial population and a doubling period. -/
def swan_population (initial_population : ℕ) (doubling_period : ℕ) (years : ℕ) : ℕ :=
  initial_population * (2 ^ (years / doubling_period))

/-- Theorem stating that given an initial population of 15 swans, if the population doubles every 2 years, then after 10 years, the population will be 480 swans. -/
theorem swan_population_after_ten_years :
  swan_population 15 2 10 = 480 := by
  sorry

#eval swan_population 15 2 10

end swan_population_after_ten_years_l3257_325784


namespace rogers_new_crayons_l3257_325798

/-- Given that Roger has 4 used crayons, 8 broken crayons, and a total of 14 crayons,
    prove that the number of new crayons is 2. -/
theorem rogers_new_crayons (used : ℕ) (broken : ℕ) (total : ℕ) (new : ℕ) :
  used = 4 →
  broken = 8 →
  total = 14 →
  new + used + broken = total →
  new = 2 := by
  sorry

end rogers_new_crayons_l3257_325798


namespace yacht_weight_excess_excess_weight_l3257_325746

/-- Represents the weight of an animal in sheep equivalents -/
structure AnimalWeight where
  sheep : ℕ

/-- Represents the count of each animal type -/
structure AnimalCounts where
  cows : ℕ
  foxes : ℕ
  zebras : ℕ

/-- Defines the weight equivalents for each animal type -/
def animalWeights : AnimalWeight :=
  { sheep := 1 }

def cowWeight : AnimalWeight :=
  { sheep := 3 }

def foxWeight : AnimalWeight :=
  { sheep := 2 }

def zebraWeight : AnimalWeight :=
  { sheep := 5 }

/-- Calculates the total weight of all animals in sheep equivalents -/
def totalWeight (counts : AnimalCounts) : ℕ :=
  counts.cows * cowWeight.sheep +
  counts.foxes * foxWeight.sheep +
  counts.zebras * zebraWeight.sheep

/-- The theorem to be proved -/
theorem yacht_weight_excess (counts : AnimalCounts)
  (h1 : counts.cows = 20)
  (h2 : counts.foxes = 15)
  (h3 : counts.zebras = 3 * counts.foxes)
  : totalWeight counts = 315 := by
  sorry

/-- The main theorem stating the excess weight -/
theorem excess_weight (counts : AnimalCounts)
  (h1 : counts.cows = 20)
  (h2 : counts.foxes = 15)
  (h3 : counts.zebras = 3 * counts.foxes)
  : totalWeight counts - 300 = 15 := by
  sorry

end yacht_weight_excess_excess_weight_l3257_325746


namespace first_meeting_turns_l3257_325706

/-- The number of points on the circle -/
def n : ℕ := 15

/-- Alice's clockwise movement per turn -/
def alice_move : ℕ := 7

/-- Bob's counterclockwise movement per turn -/
def bob_move : ℕ := 11

/-- The relative clockwise movement per turn -/
def relative_move : ℕ := alice_move - (n - bob_move)

theorem first_meeting_turns : 
  (∃ k : ℕ, k > 0 ∧ (k * relative_move) % n = 0) → 
  (∃ m : ℕ, m > 0 ∧ (m * relative_move) % n = 0 ∧ 
    ∀ l : ℕ, l > 0 → (l * relative_move) % n = 0 → l ≥ m) →
  (∃ k : ℕ, k > 0 ∧ (k * relative_move) % n = 0 ∧ 
    ∀ l : ℕ, l > 0 → (l * relative_move) % n = 0 → k ≤ l) →
  (∀ k : ℕ, k > 0 ∧ (k * relative_move) % n = 0 ∧ 
    ∀ l : ℕ, l > 0 → (l * relative_move) % n = 0 → k ≤ l) → k = 5 := by
  sorry

#eval relative_move -- Should output 3

end first_meeting_turns_l3257_325706


namespace ben_game_probability_l3257_325759

theorem ben_game_probability (p_lose p_tie : ℚ) 
  (h_lose : p_lose = 5 / 11)
  (h_tie : p_tie = 1 / 11)
  (h_total : p_lose + p_tie + (1 - p_lose - p_tie) = 1) :
  1 - p_lose - p_tie = 5 / 11 := by
sorry

end ben_game_probability_l3257_325759


namespace andy_socks_difference_l3257_325773

theorem andy_socks_difference (black_socks : ℕ) (white_socks : ℕ) : 
  black_socks = 6 →
  white_socks = 4 * black_socks →
  (white_socks / 2) - black_socks = 6 := by
  sorry

end andy_socks_difference_l3257_325773


namespace arnold_protein_consumption_l3257_325776

/-- Calculates the total protein consumed given the protein content of different food items. -/
def total_protein_consumed (collagen_protein_per_2_scoops : ℕ) (protein_powder_per_scoop : ℕ) (steak_protein : ℕ) : ℕ :=
  let collagen_protein := collagen_protein_per_2_scoops / 2
  collagen_protein + protein_powder_per_scoop + steak_protein

/-- Proves that the total protein consumed is 86 grams given the specific food items. -/
theorem arnold_protein_consumption : 
  total_protein_consumed 18 21 56 = 86 := by
  sorry

end arnold_protein_consumption_l3257_325776


namespace lcm_18_35_l3257_325788

theorem lcm_18_35 : Nat.lcm 18 35 = 630 := by
  sorry

end lcm_18_35_l3257_325788


namespace power_inequality_l3257_325770

theorem power_inequality (a x y : ℝ) (ha : 0 < a) (ha1 : a < 1) (h : a^x < a^y) : x^3 > y^3 := by
  sorry

end power_inequality_l3257_325770


namespace rectangular_solid_depth_l3257_325745

/-- The surface area of a rectangular solid given its dimensions -/
def surface_area (length width depth : ℝ) : ℝ :=
  2 * (length * width + length * depth + width * depth)

/-- Theorem: A rectangular solid with length 10, width 9, and surface area 408 has depth 6 -/
theorem rectangular_solid_depth :
  ∃ (depth : ℝ), surface_area 10 9 depth = 408 ∧ depth = 6 := by
  sorry

end rectangular_solid_depth_l3257_325745


namespace jamies_shoes_cost_l3257_325758

/-- The cost of Jamie's shoes given the total cost and James' items -/
theorem jamies_shoes_cost (total_cost : ℕ) (coat_cost : ℕ) (jeans_cost : ℕ) : 
  total_cost = 110 →
  coat_cost = 40 →
  jeans_cost = 20 →
  total_cost = coat_cost + 2 * jeans_cost + (total_cost - (coat_cost + 2 * jeans_cost)) →
  (total_cost - (coat_cost + 2 * jeans_cost)) = 30 := by
sorry

end jamies_shoes_cost_l3257_325758


namespace bowtie_equation_solution_l3257_325700

-- Define the operation ⋈
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ y : ℝ, bowtie 7 y = 15 ∧ y = 56 := by
  sorry

end bowtie_equation_solution_l3257_325700


namespace sum_first_15_odd_from_5_l3257_325766

/-- The sum of the first n odd positive integers starting from a given odd number -/
def sumOddIntegers (start : ℕ) (n : ℕ) : ℕ :=
  let lastTerm := start + 2 * (n - 1)
  (start + lastTerm) * n / 2

/-- The proposition that the sum of the first 15 odd positive integers starting from 5 is 315 -/
theorem sum_first_15_odd_from_5 : sumOddIntegers 5 15 = 315 := by
  sorry

end sum_first_15_odd_from_5_l3257_325766


namespace four_digit_sum_problem_l3257_325767

theorem four_digit_sum_problem (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
  6 * (a + b + c + d) * 1111 = 73326 →
  ({a, b, c, d} : Finset ℕ) = {1, 2, 3, 5} :=
by sorry

end four_digit_sum_problem_l3257_325767


namespace quadratic_inequality_l3257_325707

/-- Quadratic function -/
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Definition of N -/
def N (a b c : ℝ) : ℝ := |a + b + c| + |2*a - b|

/-- Definition of M -/
def M (a b c : ℝ) : ℝ := |a - b + c| + |2*a + b|

theorem quadratic_inequality (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : -b/(2*a) > 1) 
  (h3 : f a b c 0 = c) 
  (h4 : ∃ x, f a b c x > 0) : 
  M a b c < N a b c := by
  sorry

end quadratic_inequality_l3257_325707


namespace gcd_process_max_rows_l3257_325761

/-- Represents the GCD process described in the problem -/
def gcd_process (initial_sequence : List Nat) : Nat :=
  sorry

/-- The maximum number of rows in the GCD process -/
def max_rows : Nat := 501

/-- Theorem stating that the maximum number of rows in the GCD process is 501 -/
theorem gcd_process_max_rows :
  ∀ (seq : List Nat),
    (∀ n ∈ seq, 500 ≤ n ∧ n ≤ 1499) →
    seq.length = 1000 →
    gcd_process seq ≤ max_rows :=
  sorry

end gcd_process_max_rows_l3257_325761


namespace arithmetic_sequence_problem_l3257_325719

/-- Given a sequence a_n where a_2 = 2, a_6 = 0, and {1 / (a_n + 1)} is an arithmetic sequence,
    prove that a_4 = 1/2 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : a 2 = 2)
  (h2 : a 6 = 0)
  (h3 : ∃ d : ℚ, ∀ n : ℕ, 1 / (a (n + 1) + 1) - 1 / (a n + 1) = d) :
  a 4 = 1/2 := by
  sorry

end arithmetic_sequence_problem_l3257_325719


namespace midpoint_polar_coordinates_l3257_325704

/-- The polar coordinates of the midpoint of the chord intercepted by two curves -/
theorem midpoint_polar_coordinates (ρ θ : ℝ) :
  (ρ * (Real.cos θ - Real.sin θ) + 2 = 0) →  -- Curve C₁
  (ρ = 2) →  -- Curve C₂
  ∃ (r θ' : ℝ), (r = Real.sqrt 2 ∧ θ' = 3 * Real.pi / 4 ∧
    r * Real.cos θ' = -1 ∧ r * Real.sin θ' = 1) :=
by sorry

end midpoint_polar_coordinates_l3257_325704


namespace expand_expression_l3257_325702

theorem expand_expression (x : ℝ) : (7*x^2 + 5*x + 8) * 3*x = 21*x^3 + 15*x^2 + 24*x := by
  sorry

end expand_expression_l3257_325702


namespace quadrilateral_area_relations_integer_areas_perfect_square_product_l3257_325750

/-- Given a convex quadrilateral ABCD with diagonals intersecting at point P,
    S_ABP, S_BCP, S_CDP, and S_ADP are the areas of triangles ABP, BCP, CDP, and ADP respectively. -/
def QuadrilateralAreas (S_ABP S_BCP S_CDP S_ADP : ℝ) : Prop :=
  S_ABP > 0 ∧ S_BCP > 0 ∧ S_CDP > 0 ∧ S_ADP > 0

theorem quadrilateral_area_relations
  (S_ABP S_BCP S_CDP S_ADP : ℝ)
  (h : QuadrilateralAreas S_ABP S_BCP S_CDP S_ADP) :
  S_ADP = (S_ABP * S_CDP) / S_BCP ∧
  S_ABP * S_BCP * S_CDP * S_ADP = (S_ADP * S_BCP)^2 := by
  sorry

/-- If the areas of the four triangles are integers, their product is a perfect square. -/
theorem integer_areas_perfect_square_product
  (S_ABP S_BCP S_CDP S_ADP : ℤ)
  (h : QuadrilateralAreas (S_ABP : ℝ) (S_BCP : ℝ) (S_CDP : ℝ) (S_ADP : ℝ)) :
  ∃ (n : ℤ), S_ABP * S_BCP * S_CDP * S_ADP = n^2 := by
  sorry

end quadrilateral_area_relations_integer_areas_perfect_square_product_l3257_325750


namespace is_14th_term_l3257_325796

/-- The sequence term for a given index -/
def sequenceTerm (n : ℕ) : ℚ := (n + 3 : ℚ) / (n + 1 : ℚ)

/-- Theorem stating that 17/15 is the 14th term of the sequence -/
theorem is_14th_term : sequenceTerm 14 = 17 / 15 := by
  sorry

end is_14th_term_l3257_325796


namespace set_intersection_union_theorem_l3257_325736

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 > 0}
def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

-- State the theorem
theorem set_intersection_union_theorem (a b : ℝ) :
  A ∪ B a b = Set.univ ∧ A ∩ B a b = Set.Ioc 3 4 → a = -3 ∧ b = -4 := by
  sorry

end set_intersection_union_theorem_l3257_325736


namespace quadratic_roots_property_l3257_325755

theorem quadratic_roots_property (p q : ℝ) : 
  (3 * p ^ 2 + 9 * p - 21 = 0) → 
  (3 * q ^ 2 + 9 * q - 21 = 0) → 
  (3 * p - 4) * (6 * q - 8) = 14 := by
sorry

end quadratic_roots_property_l3257_325755


namespace square_distance_sum_l3257_325791

theorem square_distance_sum (s : Real) (h : s = 4) : 
  let midpoint_distance := 2 * s / 2
  let diagonal_distance := s * Real.sqrt 2
  let side_distance := s
  2 * midpoint_distance + 2 * Real.sqrt (midpoint_distance^2 + (s/2)^2) + diagonal_distance + side_distance = 10 + 4 * Real.sqrt 5 + 4 * Real.sqrt 2 :=
by sorry

end square_distance_sum_l3257_325791


namespace village_population_is_100_l3257_325799

/-- Represents the number of people in a youth summer village with specific characteristics. -/
def village_population (total : ℕ) (not_working : ℕ) (with_families : ℕ) (shower_singers : ℕ) (working_no_family_singers : ℕ) : Prop :=
  not_working = 50 ∧
  with_families = 25 ∧
  shower_singers = 75 ∧
  working_no_family_singers = 50 ∧
  total = not_working + with_families + shower_singers - working_no_family_singers

theorem village_population_is_100 :
  ∃ (total : ℕ), village_population total 50 25 75 50 ∧ total = 100 := by
  sorry

end village_population_is_100_l3257_325799


namespace H_surjective_l3257_325725

def H (x : ℝ) : ℝ := |3 * x + 1| - |x - 2|

theorem H_surjective : Function.Surjective H := by sorry

end H_surjective_l3257_325725


namespace grains_in_gray_areas_l3257_325790

/-- Given two circles with equal total grains, prove that the sum of their non-overlapping parts is 61 grains -/
theorem grains_in_gray_areas (total_circle1 total_circle2 overlap : ℕ) 
  (h1 : total_circle1 = 110)
  (h2 : total_circle2 = 87)
  (h3 : overlap = 68)
  (h4 : total_circle1 = total_circle2) : 
  (total_circle1 - overlap) + (total_circle2 - overlap) = 61 := by
  sorry

#check grains_in_gray_areas

end grains_in_gray_areas_l3257_325790


namespace right_triangle_leg_sum_equals_circle_diameters_sum_l3257_325720

/-- A right-angled triangle with its inscribed and circumscribed circles -/
structure RightTriangle where
  /-- The length of one leg of the right triangle -/
  leg1 : ℝ
  /-- The length of the other leg of the right triangle -/
  leg2 : ℝ
  /-- The radius of the inscribed circle -/
  inradius : ℝ
  /-- The radius of the circumscribed circle -/
  circumradius : ℝ
  /-- All lengths are positive -/
  leg1_pos : 0 < leg1
  leg2_pos : 0 < leg2
  inradius_pos : 0 < inradius
  circumradius_pos : 0 < circumradius

/-- 
In a right-angled triangle, the sum of the lengths of the two legs 
is equal to the sum of the diameters of the inscribed and circumscribed circles
-/
theorem right_triangle_leg_sum_equals_circle_diameters_sum (t : RightTriangle) :
  t.leg1 + t.leg2 = 2 * t.inradius + 2 * t.circumradius := by
  sorry

end right_triangle_leg_sum_equals_circle_diameters_sum_l3257_325720


namespace fifteenth_term_of_sequence_l3257_325794

def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

theorem fifteenth_term_of_sequence (a₁ a₂ : ℚ) (h₁ : a₁ = 12) (h₂ : a₂ = 4) :
  geometric_sequence a₁ (a₂ / a₁) 15 = 12 / 4782969 := by
  sorry

end fifteenth_term_of_sequence_l3257_325794


namespace sams_mystery_books_l3257_325797

/-- The number of mystery books Sam bought at the school's book fair -/
def mystery_books : ℕ := sorry

/-- The number of adventure books Sam bought -/
def adventure_books : ℕ := 13

/-- The number of used books Sam bought -/
def used_books : ℕ := 15

/-- The number of new books Sam bought -/
def new_books : ℕ := 15

/-- The total number of books Sam bought -/
def total_books : ℕ := used_books + new_books

theorem sams_mystery_books : 
  mystery_books = total_books - adventure_books ∧ 
  mystery_books = 17 := by sorry

end sams_mystery_books_l3257_325797


namespace fruit_basket_combinations_l3257_325751

/-- The number of ways to choose apples for a fruit basket -/
def apple_choices : ℕ := 3

/-- The number of ways to choose oranges for a fruit basket -/
def orange_choices : ℕ := 8

/-- The total number of fruit basket combinations -/
def total_combinations : ℕ := apple_choices * orange_choices

/-- Theorem stating the number of possible fruit baskets -/
theorem fruit_basket_combinations :
  total_combinations = 36 :=
sorry

end fruit_basket_combinations_l3257_325751


namespace roots_sum_of_squares_l3257_325774

theorem roots_sum_of_squares (p q r s : ℝ) : 
  (r^2 - p*r + q = 0) → (s^2 - p*s + q = 0) → r^2 + s^2 = p^2 - 2*q :=
by sorry

end roots_sum_of_squares_l3257_325774


namespace cinnamon_amount_l3257_325742

/-- The amount of nutmeg used in tablespoons -/
def nutmeg : ℝ := 0.5

/-- The difference in tablespoons between cinnamon and nutmeg -/
def difference : ℝ := 0.17

/-- The amount of cinnamon used in tablespoons -/
def cinnamon : ℝ := nutmeg + difference

theorem cinnamon_amount : cinnamon = 0.67 := by
  sorry

end cinnamon_amount_l3257_325742
