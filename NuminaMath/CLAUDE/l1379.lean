import Mathlib

namespace inverse_matrix_problem_l1379_137961

theorem inverse_matrix_problem (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A⁻¹ = !![1, 0; 0, 2] → A = !![1, 0; 0, (1/2)] := by
  sorry

end inverse_matrix_problem_l1379_137961


namespace eighty_six_million_scientific_notation_l1379_137951

/-- Expresses 86 million in scientific notation -/
theorem eighty_six_million_scientific_notation :
  (86000000 : ℝ) = 8.6 * 10^7 := by
  sorry

end eighty_six_million_scientific_notation_l1379_137951


namespace positive_root_irrational_l1379_137960

-- Define the equation
def f (x : ℝ) : ℝ := x^5 + x

-- Define the property of being a solution to the equation
def is_solution (x : ℝ) : Prop := f x = 10

-- State the theorem
theorem positive_root_irrational :
  ∃ x > 0, is_solution x ∧ ¬ (∃ (p q : ℤ), q ≠ 0 ∧ x = p / q) :=
by sorry

end positive_root_irrational_l1379_137960


namespace complex_equation_solution_l1379_137971

open Complex

theorem complex_equation_solution (a : ℝ) : 
  (1 - I)^3 / (1 + I) = a + 3*I → a = -2 := by
  sorry

end complex_equation_solution_l1379_137971


namespace purely_imaginary_complex_number_l1379_137987

theorem purely_imaginary_complex_number (m : ℝ) :
  (2 * m^2 - 3 * m - 2 : ℂ) + (6 * m^2 + 5 * m + 1 : ℂ) * Complex.I = Complex.I * ((6 * m^2 + 5 * m + 1 : ℝ) : ℂ) →
  m = -1 ∨ m = 2 :=
by sorry

end purely_imaginary_complex_number_l1379_137987


namespace tangent_chord_existence_l1379_137923

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a line in 2D space -/
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

/-- Checks if a line is tangent to a circle -/
def isTangent (l : Line) (c : Circle) : Prop := sorry

/-- Checks if a line intersects a circle to form a chord of given length -/
def formsChord (l : Line) (c : Circle) (length : ℝ) : Prop := sorry

/-- Main theorem: Given two circles and a length, there exists a tangent to the larger circle
    that forms a chord of the given length in the smaller circle -/
theorem tangent_chord_existence (largeCircle smallCircle : Circle) (chordLength : ℝ) :
  ∃ (tangentLine : Line),
    isTangent tangentLine largeCircle ∧
    formsChord tangentLine smallCircle chordLength :=
  sorry

end tangent_chord_existence_l1379_137923


namespace intersection_empty_implies_a_nonnegative_l1379_137992

theorem intersection_empty_implies_a_nonnegative 
  (A : Set ℝ) (B : Set ℝ) (a : ℝ) 
  (h1 : A = {x : ℝ | x - a > 0})
  (h2 : B = {x : ℝ | x ≤ 0})
  (h3 : A ∩ B = ∅) :
  a ≥ 0 := by
sorry

end intersection_empty_implies_a_nonnegative_l1379_137992


namespace dodecagon_diagonals_l1379_137934

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

/-- Theorem: A convex dodecagon has 54 diagonals -/
theorem dodecagon_diagonals : num_diagonals dodecagon_sides = 54 := by
  sorry

end dodecagon_diagonals_l1379_137934


namespace profit_maximizing_price_profit_function_increase_current_state_verification_cost_price_verification_l1379_137984

/-- Represents the profit function for a product with given pricing and demand characteristics. -/
def profit_function (x : ℝ) : ℝ :=
  (60 + x - 40) * (300 - 10 * x)

/-- Theorem stating that the profit-maximizing price is 65 yuan. -/
theorem profit_maximizing_price :
  ∃ (max_profit : ℝ), 
    (∀ (x : ℝ), profit_function x ≤ profit_function 5) ∧ 
    (profit_function 5 = max_profit) ∧
    (60 + 5 = 65) := by
  sorry

/-- Verifies that the profit function behaves as expected for price increases. -/
theorem profit_function_increase (x : ℝ) :
  profit_function x = -10 * x^2 + 100 * x + 6000 := by
  sorry

/-- Verifies that the current price and sales volume are consistent with the problem statement. -/
theorem current_state_verification :
  profit_function 0 = (60 - 40) * 300 := by
  sorry

/-- Ensures that the cost price is correctly represented in the profit function. -/
theorem cost_price_verification (x : ℝ) :
  (60 + x - 40) = (profit_function x) / (300 - 10 * x) := by
  sorry

end profit_maximizing_price_profit_function_increase_current_state_verification_cost_price_verification_l1379_137984


namespace skt_lineup_count_l1379_137995

/-- The total number of StarCraft programmers -/
def total_programmers : ℕ := 111

/-- The number of programmers in SKT's initial team -/
def initial_team_size : ℕ := 11

/-- The number of programmers needed for the lineup -/
def lineup_size : ℕ := 5

/-- The number of different lineups for SKT's second season opening match -/
def number_of_lineups : ℕ := 
  initial_team_size * (total_programmers - initial_team_size + 1) * 
  (Nat.choose initial_team_size lineup_size) * (Nat.factorial lineup_size)

theorem skt_lineup_count : number_of_lineups = 61593840 := by
  sorry

end skt_lineup_count_l1379_137995


namespace olivias_cans_l1379_137925

/-- The number of bags Olivia had -/
def num_bags : ℕ := 4

/-- The number of cans in each bag -/
def cans_per_bag : ℕ := 5

/-- The total number of cans Olivia had -/
def total_cans : ℕ := num_bags * cans_per_bag

theorem olivias_cans : total_cans = 20 := by
  sorry

end olivias_cans_l1379_137925


namespace probability_letter_in_mathematics_l1379_137936

def alphabet_size : ℕ := 26
def unique_letters_in_mathematics : ℕ := 8

theorem probability_letter_in_mathematics :
  (unique_letters_in_mathematics : ℚ) / (alphabet_size : ℚ) = 4 / 13 := by
  sorry

end probability_letter_in_mathematics_l1379_137936


namespace tony_grocery_distance_l1379_137945

/-- Represents the distance Tony needs to drive for his errands -/
structure TonyErrands where
  halfway_distance : ℝ
  haircut_distance : ℝ
  doctor_distance : ℝ

/-- Calculates the distance Tony needs to drive for groceries -/
def grocery_distance (e : TonyErrands) : ℝ :=
  2 * e.halfway_distance - (e.haircut_distance + e.doctor_distance)

/-- Theorem stating that Tony needs to drive 10 miles for groceries -/
theorem tony_grocery_distance :
  ∀ (e : TonyErrands),
    e.halfway_distance = 15 →
    e.haircut_distance = 15 →
    e.doctor_distance = 5 →
    grocery_distance e = 10 :=
by
  sorry

end tony_grocery_distance_l1379_137945


namespace cubic_roots_sum_l1379_137947

theorem cubic_roots_sum (p q r : ℝ) : 
  (3 * p^3 + 4 * p^2 - 5 * p - 6 = 0) →
  (3 * q^3 + 4 * q^2 - 5 * q - 6 = 0) →
  (3 * r^3 + 4 * r^2 - 5 * r - 6 = 0) →
  (p + q + 1)^3 + (q + r + 1)^3 + (r + p + 1)^3 = 2/3 := by
sorry

end cubic_roots_sum_l1379_137947


namespace guitar_sales_proof_l1379_137933

/-- Calculates the total amount earned from selling guitars -/
def total_guitar_sales (total_guitars : ℕ) (electric_guitars : ℕ) (electric_price : ℕ) (acoustic_price : ℕ) : ℕ :=
  let acoustic_guitars := total_guitars - electric_guitars
  electric_guitars * electric_price + acoustic_guitars * acoustic_price

/-- Proves that the total amount earned from selling 9 guitars, 
    consisting of 4 electric guitars at $479 each and 5 acoustic guitars at $339 each, is $3611 -/
theorem guitar_sales_proof : 
  total_guitar_sales 9 4 479 339 = 3611 := by
  sorry

#eval total_guitar_sales 9 4 479 339

end guitar_sales_proof_l1379_137933


namespace expense_recording_l1379_137913

/-- Represents the recording of a financial transaction -/
inductive FinancialRecord
  | income (amount : ℤ)
  | expense (amount : ℤ)

/-- Records an income of 5 yuan as +5 -/
def record_income : FinancialRecord := FinancialRecord.income 5

/-- Theorem: If income of 5 yuan is recorded as +5, then expenses of 5 yuan should be recorded as -5 -/
theorem expense_recording (h : record_income = FinancialRecord.income 5) :
  FinancialRecord.expense 5 = FinancialRecord.expense (-5) :=
sorry

end expense_recording_l1379_137913


namespace chastity_final_money_is_16_49_l1379_137964

/-- Calculates the final amount of money Chastity has after buying candies and giving some to a friend --/
def chastity_final_money (
  lollipop_price : ℚ)
  (gummies_price : ℚ)
  (chips_price : ℚ)
  (chocolate_price : ℚ)
  (discount_rate : ℚ)
  (tax_rate : ℚ)
  (initial_money : ℚ) : ℚ :=
  let total_cost := 4 * lollipop_price + gummies_price + 3 * chips_price + chocolate_price
  let discounted_cost := total_cost * (1 - discount_rate)
  let taxed_cost := discounted_cost * (1 + tax_rate)
  let money_after_purchase := initial_money - taxed_cost
  let friend_payback := 2 * lollipop_price + chips_price
  money_after_purchase + friend_payback

/-- Theorem stating that Chastity's final amount of money is $16.49 --/
theorem chastity_final_money_is_16_49 :
  chastity_final_money 1.5 2 1.25 1.75 0.1 0.05 25 = 16.49 := by
  sorry

end chastity_final_money_is_16_49_l1379_137964


namespace expression_evaluation_l1379_137998

theorem expression_evaluation :
  let a : ℚ := -1/3
  let expr := (a / (a - 1)) / ((a + 1) / (a^2 - 1)) - (1 - 2*a)
  expr = -2 := by sorry

end expression_evaluation_l1379_137998


namespace committee_selection_problem_l1379_137985

/-- The number of ways to select a committee under specific constraints -/
def committeeSelections (n : ℕ) (k : ℕ) (pairTogether : Fin n → Fin n → Prop) (pairApart : Fin n → Fin n → Prop) : ℕ :=
  sorry

/-- The specific problem setup -/
theorem committee_selection_problem :
  let n : ℕ := 9
  let k : ℕ := 5
  let a : Fin n := 0
  let b : Fin n := 1
  let c : Fin n := 2
  let d : Fin n := 3
  let pairTogether (i j : Fin n) := (i = a ∧ j = b) ∨ (i = b ∧ j = a)
  let pairApart (i j : Fin n) := (i = c ∧ j = d) ∨ (i = d ∧ j = c)
  committeeSelections n k pairTogether pairApart = 41 :=
sorry

end committee_selection_problem_l1379_137985


namespace lowest_price_calculation_l1379_137965

/-- Calculates the lowest price per component to avoid loss --/
def lowest_price_per_component (production_cost shipping_cost : ℚ) 
  (fixed_monthly_cost : ℚ) (production_volume : ℕ) : ℚ :=
  let total_cost := production_volume * (production_cost + shipping_cost) + fixed_monthly_cost
  total_cost / production_volume

/-- Theorem: The lowest price per component is the total cost divided by the number of components --/
theorem lowest_price_calculation (production_cost shipping_cost : ℚ) 
  (fixed_monthly_cost : ℚ) (production_volume : ℕ) :
  lowest_price_per_component production_cost shipping_cost fixed_monthly_cost production_volume = 
  (production_volume * (production_cost + shipping_cost) + fixed_monthly_cost) / production_volume :=
by
  sorry

#eval lowest_price_per_component 80 4 16500 150

end lowest_price_calculation_l1379_137965


namespace classroom_average_l1379_137949

theorem classroom_average (class_size : ℕ) (class_avg : ℚ) (two_thirds_avg : ℚ) :
  class_size > 0 →
  class_avg = 55 →
  two_thirds_avg = 60 →
  ∃ (one_third_avg : ℚ),
    (1 : ℚ) / 3 * one_third_avg + (2 : ℚ) / 3 * two_thirds_avg = class_avg ∧
    one_third_avg = 45 :=
by sorry

end classroom_average_l1379_137949


namespace positive_real_product_and_sum_squares_l1379_137928

theorem positive_real_product_and_sum_squares (m n : ℝ) 
  (hm : m > 0) (hn : n > 0) (h_sum : m + n = 2 * m * n) : 
  m * n ≥ 1 ∧ m^2 + n^2 ≥ 2 := by
  sorry

end positive_real_product_and_sum_squares_l1379_137928


namespace cube_root_equation_solutions_l1379_137954

theorem cube_root_equation_solutions :
  let f (x : ℝ) := (18 * x - 3)^(1/3) + (12 * x + 3)^(1/3) - 5 * x^(1/3)
  { x : ℝ | f x = 0 } = 
    { 0, (-27 + Real.sqrt 18477) / 1026, (-27 - Real.sqrt 18477) / 1026 } := by
  sorry

end cube_root_equation_solutions_l1379_137954


namespace geometric_sequence_sum_l1379_137912

/-- A geometric sequence with common ratio q < 0 -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q < 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  a 2 = 1 - a 1 →
  a 4 = 4 - a 3 →
  a 4 + a 5 = -8 := by
sorry

end geometric_sequence_sum_l1379_137912


namespace rockham_soccer_league_members_l1379_137907

theorem rockham_soccer_league_members : 
  let sock_cost : ℕ := 6
  let tshirt_cost : ℕ := sock_cost + 7
  let member_cost : ℕ := 2 * (sock_cost + tshirt_cost)
  let custom_fee : ℕ := 200
  let total_cost : ℕ := 2892
  ∃ (n : ℕ), n * member_cost + custom_fee = total_cost ∧ n = 70 :=
by sorry

end rockham_soccer_league_members_l1379_137907


namespace remove_500th_digit_of_3_7_is_greater_l1379_137956

/-- Represents a decimal expansion with a finite number of digits -/
def DecimalExpansion := List Nat

/-- Converts a rational number to its decimal expansion with a given number of digits -/
def rationalToDecimal (n d : Nat) (digits : Nat) : DecimalExpansion :=
  sorry

/-- Removes the nth digit from a decimal expansion -/
def removeNthDigit (n : Nat) (d : DecimalExpansion) : DecimalExpansion :=
  sorry

/-- Converts a decimal expansion back to a rational number -/
def decimalToRational (d : DecimalExpansion) : Rat :=
  sorry

theorem remove_500th_digit_of_3_7_is_greater :
  let original := (3 : Rat) / 7
  let decimalExp := rationalToDecimal 3 7 1000
  let modified := removeNthDigit 500 decimalExp
  decimalToRational modified > original := by
  sorry

end remove_500th_digit_of_3_7_is_greater_l1379_137956


namespace rectangle_length_l1379_137926

/-- Proves that a rectangle with length 2 cm more than width and perimeter 20 cm has length 6 cm -/
theorem rectangle_length (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = width + 2 →
  perimeter = 2 * length + 2 * width →
  perimeter = 20 →
  length = 6 := by
sorry

end rectangle_length_l1379_137926


namespace conference_support_percentage_l1379_137957

theorem conference_support_percentage
  (total_attendees : ℕ)
  (male_attendees : ℕ)
  (female_attendees : ℕ)
  (male_support_rate : ℚ)
  (female_support_rate : ℚ)
  (h1 : total_attendees = 1000)
  (h2 : male_attendees = 150)
  (h3 : female_attendees = 850)
  (h4 : male_support_rate = 70 / 100)
  (h5 : female_support_rate = 75 / 100)
  (h6 : total_attendees = male_attendees + female_attendees) :
  let total_supporters : ℚ :=
    male_support_rate * male_attendees + female_support_rate * female_attendees
  (total_supporters / total_attendees) * 100 = 74.2 := by
  sorry


end conference_support_percentage_l1379_137957


namespace profit_per_meter_cloth_l1379_137963

theorem profit_per_meter_cloth (cloth_length : ℝ) (selling_price : ℝ) (cost_price_per_meter : ℝ)
  (h1 : cloth_length = 80)
  (h2 : selling_price = 6900)
  (h3 : cost_price_per_meter = 66.25) :
  (selling_price - cloth_length * cost_price_per_meter) / cloth_length = 20 := by
  sorry

end profit_per_meter_cloth_l1379_137963


namespace ac_less_than_bc_l1379_137977

theorem ac_less_than_bc (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) : a * c < b * c := by
  sorry

end ac_less_than_bc_l1379_137977


namespace minimum_games_for_percentage_l1379_137972

theorem minimum_games_for_percentage (N : ℕ) : N = 7 ↔ 
  (N ≥ 0) ∧ 
  (∀ k : ℕ, k ≥ 0 → (2 : ℚ) / (3 + k) ≥ (9 : ℚ) / 10 → k ≥ N) ∧
  ((2 : ℚ) / (3 + N) ≥ (9 : ℚ) / 10) :=
by sorry

end minimum_games_for_percentage_l1379_137972


namespace monomial_count_l1379_137927

/-- An algebraic expression is a monomial if it consists of a single term. -/
def isMonomial (expr : String) : Bool := sorry

/-- The set of given algebraic expressions. -/
def expressions : List String := [
  "3a^2 + b",
  "-2",
  "3xy^3/5",
  "a^2b/3 + 1",
  "a^2 - 3b^2",
  "2abc"
]

/-- Counts the number of monomials in a list of expressions. -/
def countMonomials (exprs : List String) : Nat :=
  exprs.filter isMonomial |>.length

theorem monomial_count :
  countMonomials expressions = 3 := by sorry

end monomial_count_l1379_137927


namespace mean_height_of_players_l1379_137981

def heights : List ℕ := [47, 48, 50, 51, 51, 54, 55, 56, 56, 57, 61, 63, 64, 64, 65, 67]

theorem mean_height_of_players : 
  (heights.sum : ℚ) / heights.length = 56.8125 := by
  sorry

end mean_height_of_players_l1379_137981


namespace same_day_of_week_l1379_137902

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Given a year and a day number, returns the day of the week -/
def dayOfWeek (year : Nat) (dayNumber : Nat) : DayOfWeek := sorry

theorem same_day_of_week (year : Nat) :
  dayOfWeek year 15 = DayOfWeek.Monday →
  dayOfWeek year 197 = DayOfWeek.Monday :=
by
  sorry

end same_day_of_week_l1379_137902


namespace parabola_intersection_right_angle_l1379_137920

-- Define the line equation
def line (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
def point_on_parabola (t : ℝ) : ℝ × ℝ := (t^2, 2*t)

-- Define the angle between three points
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem parabola_intersection_right_angle :
  ∃ (A B C : ℝ × ℝ),
    A ≠ B ∧
    line A.1 A.2 ∧
    line B.1 B.2 ∧
    parabola A.1 A.2 ∧
    parabola B.1 B.2 ∧
    parabola C.1 C.2 ∧
    angle A C B = π/2 →
    C = (1, -2) ∨ C = (9, -6) :=
sorry

end parabola_intersection_right_angle_l1379_137920


namespace weight_loss_difference_l1379_137958

-- Define weight loss patterns
def barbi_loss_year1 : ℝ := 1.5 * 12
def barbi_loss_year2_3 : ℝ := 2.2 * 12 * 2

def luca_loss_year1 : ℝ := 9
def luca_loss_year2 : ℝ := 12
def luca_loss_year3_7 : ℝ := (12 + 3 * 5)

def kim_loss_year1 : ℝ := 2 * 12
def kim_loss_year2_3 : ℝ := 3 * 12 * 2
def kim_loss_year4_6 : ℝ := 1 * 12 * 3

-- Calculate total weight loss for each person
def barbi_total_loss : ℝ := barbi_loss_year1 + barbi_loss_year2_3
def luca_total_loss : ℝ := luca_loss_year1 + luca_loss_year2 + 5 * luca_loss_year3_7
def kim_total_loss : ℝ := kim_loss_year1 + kim_loss_year2_3 + kim_loss_year4_6

-- Theorem to prove
theorem weight_loss_difference :
  luca_total_loss + kim_total_loss - barbi_total_loss = 217.2 := by
  sorry

end weight_loss_difference_l1379_137958


namespace max_volume_after_dilutions_l1379_137994

/-- The maximum volume of a bucket that satisfies the given dilution conditions -/
theorem max_volume_after_dilutions : 
  ∃ (V : ℝ), V > 0 ∧ 
  (V - 10 - 8 * (V - 10) / V) / V ≤ 0.6 ∧
  ∀ (W : ℝ), W > 0 → (W - 10 - 8 * (W - 10) / W) / W ≤ 0.6 → W ≤ V ∧
  V = 40 :=
sorry

end max_volume_after_dilutions_l1379_137994


namespace area_ratio_is_459_625_l1379_137943

/-- Triangle XYZ with points P and Q -/
structure TriangleXYZ where
  /-- Side length XY -/
  xy : ℝ
  /-- Side length YZ -/
  yz : ℝ
  /-- Side length XZ -/
  xz : ℝ
  /-- Length XP -/
  xp : ℝ
  /-- Length XQ -/
  xq : ℝ
  /-- xy is positive -/
  xy_pos : 0 < xy
  /-- yz is positive -/
  yz_pos : 0 < yz
  /-- xz is positive -/
  xz_pos : 0 < xz
  /-- xp is positive and less than xy -/
  xp_bounds : 0 < xp ∧ xp < xy
  /-- xq is positive and less than xz -/
  xq_bounds : 0 < xq ∧ xq < xz

/-- The ratio of areas in the triangle -/
def areaRatio (t : TriangleXYZ) : ℝ := sorry

/-- Theorem stating the ratio of areas -/
theorem area_ratio_is_459_625 (t : TriangleXYZ) 
  (h1 : t.xy = 30) (h2 : t.yz = 45) (h3 : t.xz = 51) 
  (h4 : t.xp = 18) (h5 : t.xq = 15) : 
  areaRatio t = 459 / 625 := by sorry

end area_ratio_is_459_625_l1379_137943


namespace min_decimal_digits_l1379_137983

theorem min_decimal_digits (n : ℕ) (d : ℕ) : 
  n = 987654321 ∧ d = 2^30 * 5^6 → 
  (∃ (k : ℕ), k = 30 ∧ 
    ∀ (m : ℕ), (∃ (q r : ℚ), q * 10^m = n / d ∧ r = 0) → m ≥ k) ∧
  (∀ (l : ℕ), l < 30 → 
    ∃ (q r : ℚ), q * 10^l = n / d ∧ r ≠ 0) :=
sorry

end min_decimal_digits_l1379_137983


namespace optimal_tax_and_revenue_l1379_137950

-- Define the market supply function
def supply_function (P : ℝ) : ℝ := 6 * P - 312

-- Define the market demand function
def demand_function (P : ℝ) (a : ℝ) : ℝ := a - 4 * P

-- Define the tax revenue function
def tax_revenue (t : ℝ) : ℝ := 288 * t - 2.4 * t^2

-- State the theorem
theorem optimal_tax_and_revenue 
  (elasticity_ratio : ℝ) 
  (consumer_price_after_tax : ℝ) 
  (initial_tax_rate : ℝ) :
  elasticity_ratio = 1.5 →
  consumer_price_after_tax = 118 →
  initial_tax_rate = 30 →
  ∃ (optimal_tax : ℝ) (max_revenue : ℝ),
    optimal_tax = 60 ∧
    max_revenue = 8640 ∧
    ∀ (t : ℝ), tax_revenue t ≤ max_revenue :=
by sorry

end optimal_tax_and_revenue_l1379_137950


namespace min_value_expression_min_value_achieved_l1379_137989

theorem min_value_expression (x₁ x₂ : ℝ) (h1 : x₁ + x₂ = 16) (h2 : x₁ > x₂) :
  (x₁^2 + x₂^2) / (x₁ - x₂) ≥ 16 :=
by sorry

theorem min_value_achieved (x₁ x₂ : ℝ) (h1 : x₁ + x₂ = 16) (h2 : x₁ > x₂) :
  ∃ x₁' x₂' : ℝ, x₁' + x₂' = 16 ∧ x₁' > x₂' ∧ (x₁'^2 + x₂'^2) / (x₁' - x₂') = 16 :=
by sorry

end min_value_expression_min_value_achieved_l1379_137989


namespace expression_equals_36_l1379_137988

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end expression_equals_36_l1379_137988


namespace inequality_and_minimum_l1379_137980

theorem inequality_and_minimum (a b x y : ℝ) (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) :
  let f := fun (t : ℝ) => 2 / t + 9 / (1 - 2 * t)
  -- Part I: Inequality and equality condition
  (a^2 / x + b^2 / y ≥ (a + b)^2 / (x + y)) ∧
  (a^2 / x + b^2 / y = (a + b)^2 / (x + y) ↔ a * y = b * x) ∧
  -- Part II: Minimum value and x value for minimum
  (∀ t ∈ Set.Ioo 0 (1/2), f t ≥ 25) ∧
  (f (1/5) = 25) := by
  sorry

end inequality_and_minimum_l1379_137980


namespace visibility_time_correct_l1379_137968

/-- The time when Steve and Laura can see each other again -/
def visibility_time : ℝ := 45

/-- Steve's walking speed in feet per second -/
def steve_speed : ℝ := 3

/-- Laura's walking speed in feet per second -/
def laura_speed : ℝ := 1

/-- Distance between Steve and Laura's parallel paths in feet -/
def path_distance : ℝ := 240

/-- Diameter of the circular art installation in feet -/
def installation_diameter : ℝ := 80

/-- Initial separation between Steve and Laura when hidden by the art installation in feet -/
def initial_separation : ℝ := 230

/-- Theorem stating that the visibility time is correct given the problem conditions -/
theorem visibility_time_correct :
  ∃ (steve_pos laura_pos : ℝ × ℝ),
    let steve_final := (steve_pos.1 + steve_speed * visibility_time, steve_pos.2)
    let laura_final := (laura_pos.1 + laura_speed * visibility_time, laura_pos.2)
    (steve_pos.2 - laura_pos.2 = path_distance) ∧
    ((steve_pos.1 - laura_pos.1)^2 + (steve_pos.2 - laura_pos.2)^2 = initial_separation^2) ∧
    (∃ (center : ℝ × ℝ), 
      (center.1 - steve_pos.1)^2 + ((center.2 - steve_pos.2) - path_distance/2)^2 = (installation_diameter/2)^2 ∧
      (center.1 - laura_pos.1)^2 + ((center.2 - laura_pos.2) + path_distance/2)^2 = (installation_diameter/2)^2) ∧
    ((steve_final.1 - laura_final.1)^2 + (steve_final.2 - laura_final.2)^2 > 
     (steve_pos.1 - laura_pos.1)^2 + (steve_pos.2 - laura_pos.2)^2) ∧
    (∀ t : ℝ, 0 < t → t < visibility_time →
      ∃ (x y : ℝ), 
        x^2 + y^2 = (installation_diameter/2)^2 ∧
        (y - steve_pos.2) * (steve_pos.1 + steve_speed * t - x) = 
        (x - steve_pos.1 - steve_speed * t) * (steve_pos.2 - y) ∧
        (y - laura_pos.2) * (laura_pos.1 + laura_speed * t - x) = 
        (x - laura_pos.1 - laura_speed * t) * (laura_pos.2 - y)) :=
by sorry

end visibility_time_correct_l1379_137968


namespace stating_plane_landing_time_l1379_137973

/-- Represents the scenario of a mail delivery between a post office and an airfield -/
structure MailDeliveryScenario where
  usual_travel_time : ℕ  -- Usual one-way travel time in minutes
  early_arrival : ℕ      -- How many minutes earlier the Moskvich arrived
  truck_travel_time : ℕ  -- How long the truck traveled before meeting Moskvich

/-- 
Theorem stating that under the given conditions, the plane must have landed 40 minutes early.
-/
theorem plane_landing_time (scenario : MailDeliveryScenario) 
  (h1 : scenario.early_arrival = 20)
  (h2 : scenario.truck_travel_time = 30) :
  40 = (scenario.truck_travel_time + (scenario.early_arrival / 2)) :=
by sorry

end stating_plane_landing_time_l1379_137973


namespace simplify_fraction_l1379_137921

theorem simplify_fraction :
  (140 : ℚ) / 2100 = 1 / 15 := by sorry

end simplify_fraction_l1379_137921


namespace paul_strawberries_l1379_137937

/-- The number of strawberries Paul has after picking more -/
def total_strawberries (initial : ℕ) (picked : ℕ) : ℕ := initial + picked

/-- Theorem: Paul has 63 strawberries after picking more -/
theorem paul_strawberries : total_strawberries 28 35 = 63 := by
  sorry

end paul_strawberries_l1379_137937


namespace f_one_third_bounds_l1379_137996

def f_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x ∧ f x ≤ 1) ∧
  f 0 = 0 ∧
  f 1 = 1 ∧
  ∀ x y z, 0 ≤ x ∧ x < y ∧ y < z ∧ z ≤ 1 ∧ z - y = y - x →
    (1/2 : ℝ) ≤ (f z - f y) / (f y - f x) ∧ (f z - f y) / (f y - f x) ≤ 2

theorem f_one_third_bounds (f : ℝ → ℝ) (h : f_conditions f) :
  (1/7 : ℝ) ≤ f (1/3) ∧ f (1/3) ≤ 4/7 := by
  sorry

end f_one_third_bounds_l1379_137996


namespace gcd_of_198_and_286_l1379_137914

theorem gcd_of_198_and_286 : Nat.gcd 198 286 = 22 := by
  sorry

end gcd_of_198_and_286_l1379_137914


namespace hall_to_cube_edge_l1379_137946

/-- Represents the dimensions of a rectangular hall --/
structure HallDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular hall --/
def hallVolume (d : HallDimensions) : ℝ := d.length * d.width * d.height

/-- Theorem: Given a rectangular hall with specific wall areas, 
    the edge of a cube with the same volume is the cube root of 40 --/
theorem hall_to_cube_edge 
  (d : HallDimensions) 
  (floor_area : d.length * d.width = 20)
  (long_wall_area : d.width * d.height = 10)
  (short_wall_area : d.length * d.height = 8) :
  ∃ (edge : ℝ), edge^3 = hallVolume d ∧ edge^3 = 40 := by
  sorry

end hall_to_cube_edge_l1379_137946


namespace divisibility_of_power_plus_one_l1379_137915

theorem divisibility_of_power_plus_one (n : ℕ) :
  ∃ k : ℤ, 2^(3^n) + 1 = k * 3^(n + 1) := by sorry

end divisibility_of_power_plus_one_l1379_137915


namespace chinese_character_multiplication_l1379_137909

theorem chinese_character_multiplication : ∃! (x y : Nat), 
  x ≠ y ∧ x ≠ 3 ∧ x ≠ 0 ∧ y ≠ 3 ∧ y ≠ 0 ∧
  (3000 + 100 * x + y) * (3000 + 100 * x + y) ≥ 10000000 ∧
  (3000 + 100 * x + y) * (3000 + 100 * x + y) < 100000000 :=
by sorry

#check chinese_character_multiplication

end chinese_character_multiplication_l1379_137909


namespace highest_score_is_96_l1379_137982

def standard_score : ℝ := 85

def deviations : List ℝ := [-9, -4, 11, -7, 0]

def actual_scores : List ℝ := deviations.map (λ x => x + standard_score)

theorem highest_score_is_96 : 
  ∀ (score : ℝ), score ∈ actual_scores → score ≤ 96 :=
by sorry

end highest_score_is_96_l1379_137982


namespace triangle_max_perimeter_l1379_137922

theorem triangle_max_perimeter :
  ∀ x : ℕ,
  x > 0 →
  x ≤ 20 →
  x + 4*x > 20 →
  x + 20 > 4*x →
  4*x + 20 > x →
  ∀ y : ℕ,
  y > 0 →
  y ≤ 20 →
  y + 4*y > 20 →
  y + 20 > 4*y →
  4*y + 20 > y →
  x + 4*x + 20 ≥ y + 4*y + 20 →
  x + 4*x + 20 ≤ 50 :=
by sorry

end triangle_max_perimeter_l1379_137922


namespace shekar_average_marks_l1379_137910

/-- Represents Shekar's scores in different subjects -/
structure ShekarScores where
  mathematics : ℕ
  science : ℕ
  social_studies : ℕ
  english : ℕ
  biology : ℕ

/-- Calculates the average of Shekar's scores -/
def average_marks (scores : ShekarScores) : ℚ :=
  (scores.mathematics + scores.science + scores.social_studies + scores.english + scores.biology) / 5

/-- Theorem stating that Shekar's average marks are 77 -/
theorem shekar_average_marks :
  let scores : ShekarScores := {
    mathematics := 76,
    science := 65,
    social_studies := 82,
    english := 67,
    biology := 95
  }
  average_marks scores = 77 := by sorry

end shekar_average_marks_l1379_137910


namespace mistaken_calculation_system_l1379_137900

theorem mistaken_calculation_system (x y : ℝ) : 
  (5/4 * x = 4/5 * x + 36) ∧ 
  (7/3 * y = 3/7 * y + 28) → 
  x = 80 ∧ y = 14.7 := by
sorry

end mistaken_calculation_system_l1379_137900


namespace stratified_sampling_school_a_l1379_137999

theorem stratified_sampling_school_a (school_a : ℕ) (school_b : ℕ) (school_c : ℕ) (sample_size : ℕ) :
  school_a = 3600 →
  school_b = 5400 →
  school_c = 1800 →
  sample_size = 90 →
  (school_a * sample_size) / (school_a + school_b + school_c) = 30 := by
  sorry

end stratified_sampling_school_a_l1379_137999


namespace water_tank_capacity_l1379_137908

theorem water_tank_capacity (initial_fraction : Rat) (final_fraction : Rat) (added_volume : Rat) (total_capacity : Rat) : 
  initial_fraction = 1/3 →
  final_fraction = 2/5 →
  added_volume = 5 →
  initial_fraction * total_capacity + added_volume = final_fraction * total_capacity →
  total_capacity = 37.5 := by
  sorry

end water_tank_capacity_l1379_137908


namespace quadratic_discriminant_l1379_137939

/-- The discriminant of a quadratic equation ax^2 + bx + c = 0 is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation x^2 - 4x - 11 = 0 has discriminant 60 -/
theorem quadratic_discriminant :
  discriminant 1 (-4) (-11) = 60 := by
  sorry

end quadratic_discriminant_l1379_137939


namespace price_after_two_reductions_l1379_137918

-- Define the price reductions
def first_reduction : ℝ := 0.1  -- 10%
def second_reduction : ℝ := 0.14  -- 14%

-- Define the theorem
theorem price_after_two_reductions :
  let original_price : ℝ := 100
  let price_after_first_reduction := original_price * (1 - first_reduction)
  let final_price := price_after_first_reduction * (1 - second_reduction)
  final_price / original_price = 0.774 :=
by sorry

end price_after_two_reductions_l1379_137918


namespace linear_equation_k_value_l1379_137953

theorem linear_equation_k_value (k : ℤ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (k - 3) * x^(abs k - 2) + 5 = a * x + b) → 
  k = -3 := by
sorry

end linear_equation_k_value_l1379_137953


namespace problem_1_l1379_137997

theorem problem_1 (x y : ℝ) (hx : x = Real.sqrt 3 + Real.sqrt 5) (hy : y = Real.sqrt 3 - Real.sqrt 5) :
  2 * x^2 - 4 * x * y + 2 * y^2 = 40 := by
  sorry

end problem_1_l1379_137997


namespace problem_solution_l1379_137970

theorem problem_solution (x y : ℝ) (h1 : x + y = 20) (h2 : x - y = 4) :
  x^2 - y^2 = 80 ∧ x * y = 96 := by
  sorry

end problem_solution_l1379_137970


namespace circle_tangent_line_l1379_137930

/-- Given a circle x^2 + y^2 = r^2 and a point P(x₀, y₀) on the circle,
    the tangent line at P has the equation x₀x + y₀y = r^2 -/
theorem circle_tangent_line (r x₀ y₀ : ℝ) (h : x₀^2 + y₀^2 = r^2) :
  ∀ x y : ℝ, (x - x₀)^2 + (y - y₀)^2 = 0 ↔ x₀*x + y₀*y = r^2 := by
sorry

end circle_tangent_line_l1379_137930


namespace sum_of_a_and_b_l1379_137966

theorem sum_of_a_and_b (a b : ℝ) : (2*a + 2*b - 1) * (2*a + 2*b + 1) = 99 → a + b = 5 ∨ a + b = -5 := by
  sorry

end sum_of_a_and_b_l1379_137966


namespace pasture_feeding_theorem_l1379_137919

/-- Represents a pasture with growing grass -/
structure Pasture where
  dailyGrowthRate : ℕ
  initialGrass : ℕ

/-- Calculates the number of days a pasture can feed a given number of cows -/
def feedingDays (p : Pasture) (cows : ℕ) : ℕ :=
  (p.initialGrass + p.dailyGrowthRate * cows) / cows

theorem pasture_feeding_theorem (p : Pasture) : 
  feedingDays p 10 = 20 → 
  feedingDays p 15 = 10 → 
  p.dailyGrowthRate = 5 ∧ 
  feedingDays p 30 = 4 := by
  sorry

#check pasture_feeding_theorem

end pasture_feeding_theorem_l1379_137919


namespace smallest_integer_above_sqrt5_plus_sqrt3_to_6th_l1379_137962

theorem smallest_integer_above_sqrt5_plus_sqrt3_to_6th (x : ℝ) :
  x = (Real.sqrt 5 + Real.sqrt 3)^6 → ⌈x⌉ = 3323 :=
by sorry

end smallest_integer_above_sqrt5_plus_sqrt3_to_6th_l1379_137962


namespace number_of_petunia_flats_l1379_137904

/-- The number of petunias per flat of petunias -/
def petunias_per_flat : ℕ := 8

/-- The amount of fertilizer needed for each petunia in ounces -/
def fertilizer_per_petunia : ℕ := 8

/-- The number of flats of roses -/
def rose_flats : ℕ := 3

/-- The number of roses per flat of roses -/
def roses_per_flat : ℕ := 6

/-- The amount of fertilizer needed for each rose in ounces -/
def fertilizer_per_rose : ℕ := 3

/-- The number of Venus flytraps -/
def venus_flytraps : ℕ := 2

/-- The amount of fertilizer needed for each Venus flytrap in ounces -/
def fertilizer_per_venus_flytrap : ℕ := 2

/-- The total amount of fertilizer needed in ounces -/
def total_fertilizer : ℕ := 314

/-- The theorem stating that the number of flats of petunias is 4 -/
theorem number_of_petunia_flats : 
  ∃ (P : ℕ), P * (petunias_per_flat * fertilizer_per_petunia) + 
             (rose_flats * roses_per_flat * fertilizer_per_rose) + 
             (venus_flytraps * fertilizer_per_venus_flytrap) = total_fertilizer ∧ 
             P = 4 :=
by sorry

end number_of_petunia_flats_l1379_137904


namespace homework_students_l1379_137955

theorem homework_students (total : ℕ) (reading : ℕ) (games : ℕ) (homework : ℕ) : 
  total = 24 ∧ 
  reading = total / 2 ∧ 
  games = total / 3 ∧ 
  homework = total - (reading + games) →
  homework = 4 := by
sorry

end homework_students_l1379_137955


namespace dislike_food_count_problem_solution_l1379_137944

/-- Calculates the number of students who didn't like the food -/
def students_dislike (total participants : ℕ) (students_like : ℕ) : ℕ :=
  total - students_like

/-- Proves that the number of students who didn't like the food is correct -/
theorem dislike_food_count (total : ℕ) (like : ℕ) (h : total ≥ like) :
  students_dislike total like = total - like :=
by sorry

/-- Verifies the solution for the specific problem -/
theorem problem_solution :
  students_dislike 814 383 = 431 :=
by sorry

end dislike_food_count_problem_solution_l1379_137944


namespace geometric_series_sum_l1379_137979

theorem geometric_series_sum : 
  let a : ℚ := 2/3
  let r : ℚ := -1/2
  let n : ℕ := 6
  let S : ℚ := a * (1 - r^n) / (1 - r)
  S = 7/16 := by sorry

end geometric_series_sum_l1379_137979


namespace incorrect_reasoning_l1379_137991

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the belonging relation
variable (belongs_to : Point → Line → Prop)
variable (belongs_to_plane : Point → Plane → Prop)

-- Define the subset relation for a line and a plane
variable (line_subset_plane : Line → Plane → Prop)

-- State the theorem
theorem incorrect_reasoning 
  (l : Line) (α : Plane) (A : Point) :
  ¬(∀ (l : Line) (α : Plane) (A : Point), 
    (¬(line_subset_plane l α) ∧ belongs_to A l) → ¬(belongs_to_plane A α)) :=
sorry

end incorrect_reasoning_l1379_137991


namespace largest_prime_factor_of_T_l1379_137986

/-- Represents a three-digit integer -/
structure ThreeDigitInt where
  value : Nat
  is_three_digit : 100 ≤ value ∧ value < 1000

/-- Generates the next term in the sequence based on the current term -/
def next_term (n : ThreeDigitInt) : ThreeDigitInt :=
  { value := (n.value % 100) * 10 + (n.value / 100),
    is_three_digit := sorry }

/-- Generates a sequence of three terms starting with the given number -/
def generate_sequence (start : ThreeDigitInt) : Fin 3 → ThreeDigitInt
| 0 => start
| 1 => next_term start
| 2 => next_term (next_term start)

/-- Calculates the sum of all terms in a sequence -/
def sequence_sum (start : ThreeDigitInt) : Nat :=
  (generate_sequence start 0).value +
  (generate_sequence start 1).value +
  (generate_sequence start 2).value

/-- The starting number for the first sequence -/
def start1 : ThreeDigitInt :=
  { value := 312,
    is_three_digit := sorry }

/-- The starting number for the second sequence -/
def start2 : ThreeDigitInt :=
  { value := 231,
    is_three_digit := sorry }

/-- The sum of all terms from both sequences -/
def T : Nat := sequence_sum start1 + sequence_sum start2

theorem largest_prime_factor_of_T :
  ∃ (p : Nat), Nat.Prime p ∧ p ∣ T ∧ ∀ (q : Nat), Nat.Prime q → q ∣ T → q ≤ p :=
by sorry

end largest_prime_factor_of_T_l1379_137986


namespace chessboard_pawn_placement_l1379_137978

/-- The number of columns on the chessboard -/
def n : ℕ := 8

/-- The number of rows on the chessboard -/
def m : ℕ := 8

/-- The number of ways to place a pawn in a single row -/
def ways_per_row : ℕ := n + 1

/-- The total number of ways to place pawns on the chessboard -/
def total_ways : ℕ := ways_per_row ^ m

theorem chessboard_pawn_placement :
  total_ways = 3^16 :=
sorry

end chessboard_pawn_placement_l1379_137978


namespace sum_of_coefficients_l1379_137932

theorem sum_of_coefficients (b₀ b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) :
  (∀ x : ℝ, (5*x - 2)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 729 := by
sorry

end sum_of_coefficients_l1379_137932


namespace min_distance_line_circle_l1379_137976

/-- The minimum distance between a point on the line x - y + 1 = 0 
    and a point on the circle (x - 1)² + y² = 1 is √2 - 1 -/
theorem min_distance_line_circle :
  let line := {(x, y) : ℝ × ℝ | x - y + 1 = 0}
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + y^2 = 1}
  ∃ (d : ℝ), d = Real.sqrt 2 - 1 ∧ 
    (∀ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line → q ∈ circle → 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d) ∧
    (∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ circle ∧ 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = d) :=
by
  sorry


end min_distance_line_circle_l1379_137976


namespace pythagoras_field_planted_fraction_l1379_137929

theorem pythagoras_field_planted_fraction :
  ∀ (a b c x : ℝ),
  a = 5 ∧ b = 12 ∧ c^2 = a^2 + b^2 →
  (a - x)^2 + (b - x)^2 = 4^2 →
  (a * b / 2 - x^2) / (a * b / 2) = 734 / 750 := by
sorry

end pythagoras_field_planted_fraction_l1379_137929


namespace benzoic_acid_molecular_weight_l1379_137924

/-- The molecular weight of Benzoic acid -/
def molecular_weight_benzoic_acid : ℝ := 122

/-- The number of moles given in the problem -/
def moles_given : ℝ := 4

/-- The total molecular weight for the given number of moles -/
def total_molecular_weight : ℝ := 488

/-- Theorem stating that the molecular weight of Benzoic acid is correct -/
theorem benzoic_acid_molecular_weight :
  molecular_weight_benzoic_acid = total_molecular_weight / moles_given :=
sorry

end benzoic_acid_molecular_weight_l1379_137924


namespace prop_logic_l1379_137901

theorem prop_logic (p q : Prop) (h1 : ¬p) (h2 : p ∨ q) : ¬p ∧ q := by
  sorry

end prop_logic_l1379_137901


namespace final_expression_l1379_137935

theorem final_expression (x y : ℕ) : x + 2*y + x + 3*y + x + 4*y + x + y = 4*x + 10*y := by
  sorry

end final_expression_l1379_137935


namespace linear_function_x_intercept_l1379_137942

/-- A linear function f(x) = -x + 2 -/
def f (x : ℝ) : ℝ := -x + 2

/-- The x-coordinate of the intersection point with the x-axis -/
def x_intercept : ℝ := 2

theorem linear_function_x_intercept :
  f x_intercept = 0 ∧ x_intercept = 2 := by
  sorry

end linear_function_x_intercept_l1379_137942


namespace solution_set_of_inequality_l1379_137967

theorem solution_set_of_inequality (x : ℝ) :
  Set.Ioo (-1 : ℝ) 2 = {x | |x^2 - x| < 2} := by sorry

end solution_set_of_inequality_l1379_137967


namespace solution_pairs_l1379_137948

theorem solution_pairs : ∀ x y : ℝ,
  (x^2 + y^2 - 48*x - 29*y + 714 = 0 ∧
   2*x*y - 29*x - 48*y + 756 = 0) ↔
  ((x = 31.5 ∧ y = 10.5) ∨
   (x = 20 ∧ y = 22) ∨
   (x = 28 ∧ y = 7) ∨
   (x = 16.5 ∧ y = 18.5)) := by
sorry

end solution_pairs_l1379_137948


namespace airplane_seats_proof_l1379_137969

-- Define the total number of seats
def total_seats : ℕ := 540

-- Define the number of First Class seats
def first_class_seats : ℕ := 54

-- Define the proportion of Business Class seats
def business_class_proportion : ℚ := 3 / 10

-- Define the proportion of Economy Class seats
def economy_class_proportion : ℚ := 6 / 10

-- Theorem statement
theorem airplane_seats_proof :
  (first_class_seats : ℚ) + 
  (business_class_proportion * total_seats) + 
  (economy_class_proportion * total_seats) = total_seats ∧
  economy_class_proportion = 2 * business_class_proportion :=
by sorry


end airplane_seats_proof_l1379_137969


namespace sierpinski_carpet_area_sum_l1379_137938

/-- Sierpinski carpet area calculation -/
theorem sierpinski_carpet_area_sum (n : ℕ) : 
  let initial_area : ℝ := Real.sqrt 3 / 4
  let removed_area_sum : ℝ → ℕ → ℝ := λ a k => a * (1 - (3/4)^k)
  removed_area_sum initial_area n = (Real.sqrt 3 / 4) * (1 - (3/4)^n) := by
  sorry

end sierpinski_carpet_area_sum_l1379_137938


namespace quadratic_root_relation_l1379_137952

/-- Given two quadratic equations with a specific relationship between their roots,
    prove that the ratio of certain coefficients is 27. -/
theorem quadratic_root_relation (k n p : ℝ) : 
  k ≠ 0 → n ≠ 0 → p ≠ 0 →
  (∃ r₁ r₂ : ℝ, r₁ + r₂ = -p ∧ r₁ * r₂ = k) →
  (∃ s₁ s₂ : ℝ, s₁ + s₂ = -k ∧ s₁ * s₂ = n ∧ s₁ = 3*r₁ ∧ s₂ = 3*r₂) →
  n / p = 27 := by
sorry

end quadratic_root_relation_l1379_137952


namespace max_rectangle_area_garden_max_area_l1379_137905

/-- The maximum area of a rectangle with a fixed perimeter -/
theorem max_rectangle_area (p : ℝ) (h : p > 0) : 
  (∃ l w : ℝ, l > 0 ∧ w > 0 ∧ 2 * (l + w) = p ∧ 
    ∀ l' w' : ℝ, l' > 0 → w' > 0 → 2 * (l' + w') = p → l * w ≥ l' * w') →
  ∃ l w : ℝ, l > 0 ∧ w > 0 ∧ 2 * (l + w) = p ∧ l * w = (p / 4) ^ 2 :=
by sorry

/-- The maximum area of a rectangle with perimeter 400 feet is 10000 square feet -/
theorem garden_max_area : 
  ∃ l w : ℝ, l > 0 ∧ w > 0 ∧ 2 * (l + w) = 400 ∧ l * w = 10000 :=
by sorry

end max_rectangle_area_garden_max_area_l1379_137905


namespace kris_age_l1379_137959

/-- Herbert's age next year -/
def herbert_next_year : ℕ := 15

/-- Age difference between Kris and Herbert -/
def age_difference : ℕ := 10

/-- Herbert's current age -/
def herbert_current : ℕ := herbert_next_year - 1

/-- Kris's current age -/
def kris_current : ℕ := herbert_current + age_difference

theorem kris_age : kris_current = 24 := by
  sorry

end kris_age_l1379_137959


namespace cos_sin_power_relation_l1379_137993

theorem cos_sin_power_relation (x a : Real) (h : Real.cos x ^ 6 + Real.sin x ^ 6 = a) :
  Real.cos x ^ 4 + Real.sin x ^ 4 = (1 + 2 * a) / 3 := by
  sorry

end cos_sin_power_relation_l1379_137993


namespace min_value_product_l1379_137903

theorem min_value_product (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h_pos : θ₁ > 0 ∧ θ₂ > 0 ∧ θ₃ > 0 ∧ θ₄ > 0) 
  (h_sum : θ₁ + θ₂ + θ₃ + θ₄ = π) : 
  (2 * Real.sin θ₁ ^ 2 + 1 / Real.sin θ₁ ^ 2) *
  (2 * Real.sin θ₂ ^ 2 + 1 / Real.sin θ₂ ^ 2) *
  (2 * Real.sin θ₃ ^ 2 + 1 / Real.sin θ₃ ^ 2) *
  (2 * Real.sin θ₄ ^ 2 + 1 / Real.sin θ₄ ^ 2) ≥ 81 := by
  sorry

end min_value_product_l1379_137903


namespace least_distinct_values_is_184_l1379_137906

/-- Represents a list of positive integers with a unique mode -/
structure IntegerList where
  elements : List Nat
  size : elements.length = 2023
  mode_frequency : Nat
  mode_unique : mode_frequency = 12
  is_mode : elements.count mode_frequency = mode_frequency
  other_frequencies : ∀ n, n ≠ mode_frequency → elements.count n < mode_frequency

/-- The least number of distinct values in the list -/
def leastDistinctValues (list : IntegerList) : Nat :=
  list.elements.toFinset.card

/-- Theorem: The least number of distinct values in the list is 184 -/
theorem least_distinct_values_is_184 (list : IntegerList) :
  leastDistinctValues list = 184 := by
  sorry


end least_distinct_values_is_184_l1379_137906


namespace prairie_area_l1379_137974

/-- The total area of a prairie given the dust-covered and untouched areas -/
theorem prairie_area (dust_covered : ℕ) (untouched : ℕ) 
  (h1 : dust_covered = 64535) 
  (h2 : untouched = 522) : 
  dust_covered + untouched = 65057 := by
  sorry

end prairie_area_l1379_137974


namespace kite_area_in_square_l1379_137916

/-- Given a 10 cm by 10 cm square with diagonals and a vertical line segment from
    the midpoint of the bottom side to the top side, the area of the kite-shaped
    region formed around the vertical line segment is 25 cm². -/
theorem kite_area_in_square (square_side : ℝ) (kite_area : ℝ) : 
  square_side = 10 → kite_area = 25 :=
by
  sorry

end kite_area_in_square_l1379_137916


namespace square_of_complex_is_pure_imaginary_l1379_137917

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem square_of_complex_is_pure_imaginary (a : ℝ) :
  is_pure_imaginary ((1 + a * Complex.I) ^ 2) → a = 1 ∨ a = -1 := by
  sorry

end square_of_complex_is_pure_imaginary_l1379_137917


namespace rotate_minus_six_minus_three_i_l1379_137975

/-- Rotate a complex number by 180 degrees counter-clockwise around the origin -/
def rotate180 (z : ℂ) : ℂ := -z

/-- The theorem stating that rotating -6 - 3i by 180 degrees results in 6 + 3i -/
theorem rotate_minus_six_minus_three_i :
  rotate180 (-6 - 3*I) = (6 + 3*I) := by
  sorry

end rotate_minus_six_minus_three_i_l1379_137975


namespace square_from_relation_l1379_137911

theorem square_from_relation (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : b^2 = a^2 + a*b + b) : 
  ∃ k : ℕ, k > 0 ∧ b = k^2 := by
  sorry

end square_from_relation_l1379_137911


namespace tangent_line_at_one_monotone_condition_equivalent_to_range_l1379_137931

noncomputable def f (a x : ℝ) : ℝ := a * x^2 - (a + 2) * x + Real.log x

theorem tangent_line_at_one (a : ℝ) (h : a = 1) :
  ∃ (k b : ℝ), ∀ x, k * x + b = f a x + (f a 1 - f a x) * (x - 1) / (1 - x) ∧ k = 0 ∧ b = -2 :=
sorry

theorem monotone_condition_equivalent_to_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 0 < x₁ → x₁ < x₂ → f a x₁ + 2*x₁ < f a x₂ + 2*x₂) ↔ 0 ≤ a ∧ a ≤ 8 :=
sorry

end tangent_line_at_one_monotone_condition_equivalent_to_range_l1379_137931


namespace expression_evaluation_l1379_137941

theorem expression_evaluation (m : ℤ) : 
  m = -1 → (6 * m^2 - m + 3) + (-5 * m^2 + 2 * m + 1) = 4 := by
sorry

end expression_evaluation_l1379_137941


namespace arithmetic_sequence_property_l1379_137990

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 7th term of the sequence -/
def a_7 (a : ℕ → ℝ) (m : ℝ) : Prop := a 7 = m

/-- The 14th term of the sequence -/
def a_14 (a : ℕ → ℝ) (n : ℝ) : Prop := a 14 = n

/-- Theorem: In an arithmetic sequence, if a₇ = m and a₁₄ = n, then a₂₁ = 2n - m -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) (m n : ℝ) 
  (h1 : arithmetic_sequence a) (h2 : a_7 a m) (h3 : a_14 a n) : 
  a 21 = 2 * n - m := by
  sorry

end arithmetic_sequence_property_l1379_137990


namespace even_odd_function_sum_l1379_137940

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem even_odd_function_sum (f g : ℝ → ℝ) 
  (hf : IsEven f) (hg : IsOdd g) 
  (h : ∀ x, f x + g x = x^2 + 3*x + 1) : 
  ∀ x, f x = x^2 + 1 := by
sorry

end even_odd_function_sum_l1379_137940
