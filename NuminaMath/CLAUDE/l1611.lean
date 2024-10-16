import Mathlib

namespace NUMINAMATH_CALUDE_longest_side_of_triangle_l1611_161122

theorem longest_side_of_triangle (x : ℝ) : 
  5 + (2*x + 3) + (3*x - 2) = 41 →
  max 5 (max (2*x + 3) (3*x - 2)) = 19 := by
sorry

end NUMINAMATH_CALUDE_longest_side_of_triangle_l1611_161122


namespace NUMINAMATH_CALUDE_square_area_multiple_l1611_161150

theorem square_area_multiple (a p m : ℝ) : 
  a > 0 → 
  p > 0 → 
  p = 36 → 
  a = (p / 4)^2 → 
  m * a = 10 * p + 45 → 
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_square_area_multiple_l1611_161150


namespace NUMINAMATH_CALUDE_specific_pairing_probability_l1611_161188

/-- The probability of a specific pairing in a class of 50 students -/
theorem specific_pairing_probability (n : ℕ) (h : n = 50) :
  (1 : ℚ) / (n - 1) = 1 / 49 := by
  sorry

#check specific_pairing_probability

end NUMINAMATH_CALUDE_specific_pairing_probability_l1611_161188


namespace NUMINAMATH_CALUDE_gym_occupancy_l1611_161118

theorem gym_occupancy (initial_people : ℕ) (people_came_in : ℕ) (people_left : ℕ) 
  (h1 : initial_people = 16) 
  (h2 : people_came_in = 5) 
  (h3 : people_left = 2) : 
  initial_people + people_came_in - people_left = 19 :=
by sorry

end NUMINAMATH_CALUDE_gym_occupancy_l1611_161118


namespace NUMINAMATH_CALUDE_negative_plus_abs_neg_l1611_161113

theorem negative_plus_abs_neg (a : ℝ) (h : a < 0) : a + |-a| = 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_plus_abs_neg_l1611_161113


namespace NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l1611_161134

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmerSpeeds where
  manSpeed : ℝ  -- Speed of the man in still water (km/h)
  streamSpeed : ℝ  -- Speed of the stream (km/h)

/-- Calculates the effective speed given the swimmer's speed and stream speed. -/
def effectiveSpeed (s : SwimmerSpeeds) (downstream : Bool) : ℝ :=
  if downstream then s.manSpeed + s.streamSpeed else s.manSpeed - s.streamSpeed

/-- Theorem stating that given the conditions, the man's speed in still water is 5 km/h. -/
theorem swimmer_speed_in_still_water :
  ∀ s : SwimmerSpeeds,
    (effectiveSpeed s true * 4 = 24) →  -- Downstream condition
    (effectiveSpeed s false * 4 = 16) →  -- Upstream condition
    s.manSpeed = 5 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_speed_in_still_water_l1611_161134


namespace NUMINAMATH_CALUDE_decimal_to_binary_2008_l1611_161141

theorem decimal_to_binary_2008 :
  ∃ (binary : List Bool),
    binary.length = 11 ∧
    (binary.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0 = 2008) ∧
    binary = [true, true, true, true, true, false, true, true, false, false, false] := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_binary_2008_l1611_161141


namespace NUMINAMATH_CALUDE_ab_not_always_negative_l1611_161143

theorem ab_not_always_negative (a b : ℚ) 
  (h1 : (a - b)^2 + (b - a) * |a - b| = a * b) 
  (h2 : a * b ≠ 0) : 
  ¬(∀ a b : ℚ, (a - b)^2 + (b - a) * |a - b| = a * b → a * b < 0) := by
sorry

end NUMINAMATH_CALUDE_ab_not_always_negative_l1611_161143


namespace NUMINAMATH_CALUDE_workshop_nobel_laureates_l1611_161104

theorem workshop_nobel_laureates
  (total_scientists : ℕ)
  (wolf_laureates : ℕ)
  (wolf_and_nobel : ℕ)
  (h_total : total_scientists = 50)
  (h_wolf : wolf_laureates = 31)
  (h_both : wolf_and_nobel = 16)
  (h_diff : ∃ (non_nobel : ℕ), 
    wolf_laureates + non_nobel + (non_nobel + 3) = total_scientists) :
  ∃ (nobel_laureates : ℕ), 
    nobel_laureates = 27 ∧ 
    nobel_laureates ≤ total_scientists ∧
    wolf_and_nobel ≤ nobel_laureates ∧
    wolf_and_nobel ≤ wolf_laureates :=
by
  sorry


end NUMINAMATH_CALUDE_workshop_nobel_laureates_l1611_161104


namespace NUMINAMATH_CALUDE_sum_3x_4y_equals_60_l1611_161152

theorem sum_3x_4y_equals_60 
  (x y N : ℝ) 
  (h1 : 3 * x + 4 * y = N) 
  (h2 : 6 * x - 4 * y = 12) 
  (h3 : x * y = 72) : 
  3 * x + 4 * y = 60 := by
sorry

end NUMINAMATH_CALUDE_sum_3x_4y_equals_60_l1611_161152


namespace NUMINAMATH_CALUDE_simplify_T_l1611_161133

theorem simplify_T (x : ℝ) : 9*(x+2)^2 - 12*(x+2) + 4 = 4*(1.5*x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_T_l1611_161133


namespace NUMINAMATH_CALUDE_three_distinct_values_l1611_161145

/-- The number of distinct values possible when evaluating 3^(3^(3^3)) with different parenthesizations -/
def num_distinct_values : ℕ := 3

/-- The original expression 3^(3^(3^3)) -/
def original_expr : ℕ := 3^(3^(3^3))

theorem three_distinct_values :
  ∃ (a b : ℕ), a ≠ b ∧ a ≠ original_expr ∧ b ≠ original_expr ∧
  (∀ (x : ℕ), x ≠ a ∧ x ≠ b ∧ x ≠ original_expr →
    ¬∃ (e₁ e₂ e₃ : ℕ → ℕ → ℕ), x = e₁ 3 (e₂ 3 (e₃ 3 3))) ∧
  num_distinct_values = 3 :=
sorry

end NUMINAMATH_CALUDE_three_distinct_values_l1611_161145


namespace NUMINAMATH_CALUDE_tangent_perpendicular_to_y_axis_l1611_161190

/-- The curve function -/
def f (a : ℝ) (x : ℝ) : ℝ := x^4 + a*x^2 + 1

/-- The derivative of the curve function -/
def f_prime (a : ℝ) (x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_perpendicular_to_y_axis (a : ℝ) :
  (f a (-1) = a + 2) →
  (f_prime a (-1) = 0) →
  a = -2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_perpendicular_to_y_axis_l1611_161190


namespace NUMINAMATH_CALUDE_darwin_money_problem_l1611_161167

theorem darwin_money_problem (initial_money : ℝ) : 
  (3/4 * (2/3 * initial_money) = 300) → initial_money = 600 := by
  sorry

end NUMINAMATH_CALUDE_darwin_money_problem_l1611_161167


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l1611_161196

theorem merry_go_round_revolutions 
  (outer_radius inner_radius : ℝ) 
  (outer_revolutions : ℕ) 
  (h1 : outer_radius = 40)
  (h2 : inner_radius = 10)
  (h3 : outer_revolutions = 15) :
  ∃ inner_revolutions : ℕ,
    inner_revolutions = 60 ∧
    outer_radius * outer_revolutions = inner_radius * inner_revolutions :=
by sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_l1611_161196


namespace NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l1611_161183

/-- The ratio of the volume of a sphere with radius 3q to the volume of a hemisphere with radius q is 54 -/
theorem sphere_hemisphere_volume_ratio (q : ℝ) (q_pos : 0 < q) : 
  (4 / 3 * Real.pi * (3 * q)^3) / ((1 / 2) * (4 / 3 * Real.pi * q^3)) = 54 := by
  sorry

end NUMINAMATH_CALUDE_sphere_hemisphere_volume_ratio_l1611_161183


namespace NUMINAMATH_CALUDE_max_students_equal_distribution_l1611_161112

theorem max_students_equal_distribution (pens pencils : ℕ) 
  (h_pens : pens = 1001) (h_pencils : pencils = 910) : 
  (∃ (students : ℕ), 
    students > 0 ∧ 
    pens % students = 0 ∧ 
    pencils % students = 0 ∧ 
    ∀ (n : ℕ), n > students → (pens % n ≠ 0 ∨ pencils % n ≠ 0)) ↔ 
  (∃ (max_students : ℕ), max_students = Nat.gcd pens pencils) :=
sorry

end NUMINAMATH_CALUDE_max_students_equal_distribution_l1611_161112


namespace NUMINAMATH_CALUDE_mans_speed_with_current_l1611_161155

/-- 
Given a man's speed against a current and the speed of the current,
this theorem proves the man's speed with the current.
-/
theorem mans_speed_with_current 
  (speed_against_current : ℝ) 
  (current_speed : ℝ) 
  (h1 : speed_against_current = 12) 
  (h2 : current_speed = 5) : 
  speed_against_current + 2 * current_speed = 22 :=
by sorry

end NUMINAMATH_CALUDE_mans_speed_with_current_l1611_161155


namespace NUMINAMATH_CALUDE_age_sum_problem_l1611_161173

theorem age_sum_problem (a b c : ℕ+) : 
  a = b ∧ 
  a > c ∧ 
  c < 10 ∧ 
  a * b * c = 162 → 
  a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_age_sum_problem_l1611_161173


namespace NUMINAMATH_CALUDE_min_sum_squares_l1611_161187

theorem min_sum_squares (a b c t : ℝ) (h : a + b + c = t) :
  a^2 + b^2 + c^2 ≥ t^2 / 3 ∧
  (a^2 + b^2 + c^2 = t^2 / 3 ↔ a = t/3 ∧ b = t/3 ∧ c = t/3) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l1611_161187


namespace NUMINAMATH_CALUDE_supermarket_prices_theorem_l1611_161120

/-- Represents the prices and discounts at supermarkets -/
structure SupermarketPrices where
  english_machine : ℕ
  backpack : ℕ
  discount_a : ℚ
  voucher_b : ℕ
  voucher_threshold : ℕ

/-- Theorem stating the correct prices and most cost-effective supermarket -/
theorem supermarket_prices_theorem (prices : SupermarketPrices)
    (h1 : prices.english_machine + prices.backpack = 452)
    (h2 : prices.english_machine = 4 * prices.backpack - 8)
    (h3 : prices.discount_a = 75 / 100)
    (h4 : prices.voucher_b = 30)
    (h5 : prices.voucher_threshold = 100)
    (h6 : 400 ≥ prices.english_machine + prices.backpack) :
    prices.english_machine = 360 ∧ 
    prices.backpack = 92 ∧ 
    (prices.english_machine + prices.backpack) * prices.discount_a < 
      prices.english_machine + prices.backpack - prices.voucher_b := by
  sorry


end NUMINAMATH_CALUDE_supermarket_prices_theorem_l1611_161120


namespace NUMINAMATH_CALUDE_tan_sum_45_deg_l1611_161154

theorem tan_sum_45_deg (A B : Real) (h : A + B = Real.pi / 4) :
  (1 + Real.tan A) * (1 + Real.tan B) = 2 := by
sorry

end NUMINAMATH_CALUDE_tan_sum_45_deg_l1611_161154


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l1611_161101

theorem solution_set_implies_a_value (a : ℝ) :
  ({x : ℝ | |x - a| < 1} = {x : ℝ | 2 < x ∧ x < 4}) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l1611_161101


namespace NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_65_l1611_161171

theorem right_triangle_with_hypotenuse_65 :
  ∃! (a b : ℕ), a < b ∧ a^2 + b^2 = 65^2 ∧ a = 25 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_with_hypotenuse_65_l1611_161171


namespace NUMINAMATH_CALUDE_probability_spade_or_king_is_4_13_l1611_161186

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (suits : Nat)
  (cards_per_suit : Nat)
  (kings_per_suit : Nat)

/-- Calculates the probability of drawing a spade or a king -/
def probability_spade_or_king (d : Deck) : Rat :=
  let spades := d.cards_per_suit
  let kings := d.suits * d.kings_per_suit
  let overlap := d.kings_per_suit
  let favorable_outcomes := spades + kings - overlap
  favorable_outcomes / d.total_cards

/-- Theorem stating the probability of drawing a spade or a king is 4/13 -/
theorem probability_spade_or_king_is_4_13 (d : Deck) 
    (h1 : d.total_cards = 52)
    (h2 : d.suits = 4)
    (h3 : d.cards_per_suit = 13)
    (h4 : d.kings_per_suit = 1) : 
  probability_spade_or_king d = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_spade_or_king_is_4_13_l1611_161186


namespace NUMINAMATH_CALUDE_million_to_scientific_notation_two_point_684_million_scientific_notation_l1611_161160

theorem million_to_scientific_notation (n : ℝ) : 
  n * 1000000 = n * (10 : ℝ) ^ 6 := by sorry

-- Define 2.684 million
def two_point_684_million : ℝ := 2.684 * 1000000

-- Theorem to prove
theorem two_point_684_million_scientific_notation : 
  two_point_684_million = 2.684 * (10 : ℝ) ^ 6 := by sorry

end NUMINAMATH_CALUDE_million_to_scientific_notation_two_point_684_million_scientific_notation_l1611_161160


namespace NUMINAMATH_CALUDE_profit_difference_theorem_l1611_161106

/-- Represents the profit distribution for a business partnership --/
structure ProfitDistribution where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  b_profit : ℕ

/-- Calculates the difference between profit shares of a and c --/
def profit_difference (pd : ProfitDistribution) : ℕ :=
  let total_investment := pd.a_investment + pd.b_investment + pd.c_investment
  let total_profit := pd.b_profit * total_investment / pd.b_investment
  let a_profit := total_profit * pd.a_investment / total_investment
  let c_profit := total_profit * pd.c_investment / total_investment
  c_profit - a_profit

/-- Theorem stating the difference between profit shares of a and c --/
theorem profit_difference_theorem (pd : ProfitDistribution) 
  (h1 : pd.a_investment = 8000)
  (h2 : pd.b_investment = 10000)
  (h3 : pd.c_investment = 12000)
  (h4 : pd.b_profit = 3500) :
  profit_difference pd = 1400 := by
  sorry

end NUMINAMATH_CALUDE_profit_difference_theorem_l1611_161106


namespace NUMINAMATH_CALUDE_stream_speed_l1611_161156

/-- Proves that given a boat with a speed of 57 km/h in still water, 
    if the time taken to row upstream is twice the time taken to row downstream 
    for the same distance, then the speed of the stream is 19 km/h. -/
theorem stream_speed (d : ℝ) (h : d > 0) : 
  let boat_speed := 57
  let stream_speed := 19
  (d / (boat_speed - stream_speed) = 2 * (d / (boat_speed + stream_speed))) →
  stream_speed = 19 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_l1611_161156


namespace NUMINAMATH_CALUDE_corn_acreage_l1611_161110

theorem corn_acreage (total_land : ℕ) (bean_ratio wheat_ratio corn_ratio : ℕ) 
  (h1 : total_land = 1034)
  (h2 : bean_ratio = 5)
  (h3 : wheat_ratio = 2)
  (h4 : corn_ratio = 4) : 
  (total_land * corn_ratio) / (bean_ratio + wheat_ratio + corn_ratio) = 376 := by
  sorry

end NUMINAMATH_CALUDE_corn_acreage_l1611_161110


namespace NUMINAMATH_CALUDE_expense_increase_percentage_l1611_161123

def monthly_salary : ℝ := 5750
def initial_savings_rate : ℝ := 0.20
def new_savings : ℝ := 230

theorem expense_increase_percentage :
  let initial_savings := monthly_salary * initial_savings_rate
  let initial_expenses := monthly_salary - initial_savings
  let expense_increase := initial_savings - new_savings
  (expense_increase / initial_expenses) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_expense_increase_percentage_l1611_161123


namespace NUMINAMATH_CALUDE_conference_handshakes_l1611_161199

/-- Represents a conference with two groups of people -/
structure Conference :=
  (total : ℕ)
  (group_a : ℕ)
  (group_b : ℕ)
  (h_total : total = group_a + group_b)

/-- Calculates the number of handshakes in a conference -/
def handshakes (c : Conference) : ℕ :=
  c.group_a * c.group_b + (c.group_b.choose 2)

/-- Theorem stating the number of handshakes in the specific conference -/
theorem conference_handshakes :
  ∃ (c : Conference), c.total = 40 ∧ c.group_a = 25 ∧ c.group_b = 15 ∧ handshakes c = 480 :=
by sorry

end NUMINAMATH_CALUDE_conference_handshakes_l1611_161199


namespace NUMINAMATH_CALUDE_lion_population_l1611_161116

/-- Given a population of lions that increases by 4 each month for 12 months,
    prove that if the final population is 148, the initial population was 100. -/
theorem lion_population (initial_population final_population : ℕ) 
  (monthly_increase : ℕ) (months : ℕ) : 
  monthly_increase = 4 →
  months = 12 →
  final_population = 148 →
  final_population = initial_population + monthly_increase * months →
  initial_population = 100 := by
sorry

end NUMINAMATH_CALUDE_lion_population_l1611_161116


namespace NUMINAMATH_CALUDE_roundness_of_1728_l1611_161114

/-- Roundness of a number is defined as the sum of the exponents in its prime factorization -/
def roundness (n : Nat) : Nat :=
  sorry

/-- 1728 can be expressed as 2^6 * 3^3 -/
axiom factorization_1728 : 1728 = 2^6 * 3^3

theorem roundness_of_1728 : roundness 1728 = 9 := by
  sorry

end NUMINAMATH_CALUDE_roundness_of_1728_l1611_161114


namespace NUMINAMATH_CALUDE_cos_two_theta_value_l1611_161144

theorem cos_two_theta_value (θ : Real) 
  (h : Real.exp (Real.log 2 * (1 - 3/2 + 3 * Real.cos θ)) + 3 = Real.exp (Real.log 2 * (2 + Real.cos θ))) :
  Real.cos (2 * θ) = -1/2 := by
sorry

end NUMINAMATH_CALUDE_cos_two_theta_value_l1611_161144


namespace NUMINAMATH_CALUDE_number_of_tenths_l1611_161125

theorem number_of_tenths (n : ℚ) : (375 : ℚ) * (1 / 10 : ℚ) = n → n = (37.5 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_number_of_tenths_l1611_161125


namespace NUMINAMATH_CALUDE_valeria_apartment_number_l1611_161180

def is_not_multiple_of_5 (n : ℕ) : Prop := n % 5 ≠ 0

def is_odd (n : ℕ) : Prop := n % 2 = 1

def sum_of_digits_less_than_8 (n : ℕ) : Prop :=
  (n / 10 + n % 10) < 8

def units_digit_is_6 (n : ℕ) : Prop := n % 10 = 6

theorem valeria_apartment_number (n : ℕ) :
  n ≥ 10 ∧ n < 100 →
  (is_not_multiple_of_5 n ∧ is_odd n ∧ units_digit_is_6 n) ∨
  (is_not_multiple_of_5 n ∧ is_odd n ∧ sum_of_digits_less_than_8 n) ∨
  (is_not_multiple_of_5 n ∧ sum_of_digits_less_than_8 n ∧ units_digit_is_6 n) ∨
  (is_odd n ∧ sum_of_digits_less_than_8 n ∧ units_digit_is_6 n) →
  units_digit_is_6 n :=
by sorry

end NUMINAMATH_CALUDE_valeria_apartment_number_l1611_161180


namespace NUMINAMATH_CALUDE_equation_solution_l1611_161197

theorem equation_solution : ∃ x : ℝ, 5 * (x - 4) = 2 * (3 - 2 * x) + 10 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1611_161197


namespace NUMINAMATH_CALUDE_choose_officers_specific_club_l1611_161142

/-- Represents a club with boys and girls -/
structure Club where
  total_members : ℕ
  boys : ℕ
  girls : ℕ

/-- Calculates the number of ways to choose officers in a club -/
def choose_officers (c : Club) : ℕ :=
  c.total_members * (c.boys - 1 + c.girls - 1) * (c.total_members - 2)

/-- Theorem: The number of ways to choose officers in a specific club configuration -/
theorem choose_officers_specific_club :
  let c : Club := { total_members := 30, boys := 15, girls := 15 }
  choose_officers c = 11760 := by
  sorry

#eval choose_officers { total_members := 30, boys := 15, girls := 15 }

end NUMINAMATH_CALUDE_choose_officers_specific_club_l1611_161142


namespace NUMINAMATH_CALUDE_square_field_area_l1611_161135

/-- Given a square field with barbed wire drawn around it, if the total cost of the wire
    at a specific rate per meter is a certain amount, then we can determine the area of the field. -/
theorem square_field_area (wire_cost_per_meter : ℝ) (gate_width : ℝ) (num_gates : ℕ) (total_cost : ℝ) :
  wire_cost_per_meter = 1 →
  gate_width = 1 →
  num_gates = 2 →
  total_cost = 666 →
  ∃ (side_length : ℝ), 
    side_length > 0 ∧
    (4 * side_length - num_gates * gate_width) * wire_cost_per_meter = total_cost ∧
    side_length^2 = 27889 :=
by sorry

end NUMINAMATH_CALUDE_square_field_area_l1611_161135


namespace NUMINAMATH_CALUDE_rectangle_area_stage_5_l1611_161136

/-- The area of a rectangle formed by aligning squares with increasing side lengths -/
def rectangle_area (n : ℕ) : ℕ :=
  let width := 3 + n - 1
  let length := (3 + n) * n / 2 + 3
  width * length

/-- Theorem: The area of the rectangle at Stage 5 is 175 square inches -/
theorem rectangle_area_stage_5 : rectangle_area 5 = 175 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_stage_5_l1611_161136


namespace NUMINAMATH_CALUDE_fraction_sum_equality_l1611_161165

theorem fraction_sum_equality : (2 : ℚ) / 5 - 1 / 10 + 3 / 5 = 9 / 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_equality_l1611_161165


namespace NUMINAMATH_CALUDE_probability_two_females_l1611_161192

/-- The probability of selecting two female contestants out of 7 total contestants 
    (4 female, 3 male) when choosing 2 contestants at random -/
theorem probability_two_females (total : Nat) (females : Nat) (chosen : Nat) 
    (h1 : total = 7) 
    (h2 : females = 4) 
    (h3 : chosen = 2) : 
    (Nat.choose females chosen : Rat) / (Nat.choose total chosen : Rat) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_females_l1611_161192


namespace NUMINAMATH_CALUDE_fish_tank_problem_l1611_161172

theorem fish_tank_problem (tank1_goldfish tank2 tank3 : ℕ) : 
  tank1_goldfish = 7 →
  tank3 = 10 →
  tank2 = 3 * tank3 →
  tank2 = 2 * (tank1_goldfish + (tank1_beta : ℕ)) →
  tank1_beta = 8 := by
  sorry

end NUMINAMATH_CALUDE_fish_tank_problem_l1611_161172


namespace NUMINAMATH_CALUDE_cyclist_return_speed_l1611_161146

theorem cyclist_return_speed 
  (total_distance : ℝ) 
  (first_segment : ℝ) 
  (second_segment : ℝ) 
  (first_speed : ℝ) 
  (second_speed : ℝ) 
  (total_time : ℝ) :
  total_distance = first_segment + second_segment →
  first_segment = 12 →
  second_segment = 24 →
  first_speed = 8 →
  second_speed = 12 →
  total_time = 7.5 →
  (total_distance / first_speed + second_segment / second_speed + 
   (total_distance / ((total_time - (total_distance / first_speed + second_segment / second_speed))))) = total_time →
  (total_distance / (total_time - (total_distance / first_speed + second_segment / second_speed))) = 9 := by
sorry

end NUMINAMATH_CALUDE_cyclist_return_speed_l1611_161146


namespace NUMINAMATH_CALUDE_triangle_problem_l1611_161169

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : t.a = 2)
  (h2 : t.c = 3)
  (h3 : Real.cos t.B = 1/4) :
  t.b = Real.sqrt 10 ∧ Real.sin (2 * t.C) = (3 * Real.sqrt 15) / 16 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l1611_161169


namespace NUMINAMATH_CALUDE_unique_c_value_l1611_161140

theorem unique_c_value : ∃! (c : ℝ), c ≠ 0 ∧
  (∃! (b : ℝ), b > 0 ∧
    (∃! (x : ℝ), x^2 + (b + 1/b + 1) * x + c = 0)) ∧
  c = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_unique_c_value_l1611_161140


namespace NUMINAMATH_CALUDE_div_ratio_problem_l1611_161177

theorem div_ratio_problem (a b c d : ℚ) 
  (h1 : a / b = 3)
  (h2 : b / c = 3 / 4)
  (h3 : c / d = 2 / 3) :
  d / a = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_div_ratio_problem_l1611_161177


namespace NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l1611_161161

theorem polar_to_rectangular_equivalence (ρ θ x y : ℝ) :
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (3 * ρ * Real.cos θ + 4 * ρ * Real.sin θ = 2) →
  (3 * x + 4 * y - 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_equivalence_l1611_161161


namespace NUMINAMATH_CALUDE_min_sum_of_product_72_l1611_161193

theorem min_sum_of_product_72 (a b : ℤ) (h : a * b = 72) : 
  (∀ x y : ℤ, x * y = 72 → a + b ≤ x + y) ∧ (a + b = -17) :=
sorry

end NUMINAMATH_CALUDE_min_sum_of_product_72_l1611_161193


namespace NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l1611_161117

/-- The coordinates of a point in a 2D Cartesian coordinate system. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Theorem: The coordinates of the point (1, -2) with respect to the origin
    in a Cartesian coordinate system are (1, -2). -/
theorem point_coordinates_wrt_origin (p : Point2D) (h : p = ⟨1, -2⟩) :
  p.x = 1 ∧ p.y = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_coordinates_wrt_origin_l1611_161117


namespace NUMINAMATH_CALUDE_equivalence_theorem_l1611_161148

-- Define f as a differentiable function on ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define the condition that f(x) + f'(x) > 0 for all x ∈ ℝ
variable (h : ∀ x : ℝ, f x + (deriv f) x > 0)

-- State the theorem
theorem equivalence_theorem (a b : ℝ) :
  a > b ↔ Real.exp a * f a > Real.exp b * f b :=
sorry

end NUMINAMATH_CALUDE_equivalence_theorem_l1611_161148


namespace NUMINAMATH_CALUDE_expected_red_pairs_value_l1611_161157

/-- Represents a standard 104-card deck -/
structure Deck :=
  (cards : Finset (Fin 104))
  (size : cards.card = 104)

/-- Represents the color of a card -/
inductive Color
| Red
| Black

/-- Function to determine the color of a card -/
def color (card : Fin 104) : Color :=
  if card.val ≤ 51 then Color.Red else Color.Black

/-- Number of red cards in the deck -/
def num_red_cards : Nat := 52

/-- Calculates the expected number of adjacent red card pairs in a 104-card deck -/
def expected_red_pairs (d : Deck) : ℚ :=
  (num_red_cards : ℚ) * ((num_red_cards - 1) / (d.cards.card - 1))

/-- Theorem: The expected number of adjacent red card pairs is 2652/103 -/
theorem expected_red_pairs_value (d : Deck) :
  expected_red_pairs d = 2652 / 103 := by
  sorry

end NUMINAMATH_CALUDE_expected_red_pairs_value_l1611_161157


namespace NUMINAMATH_CALUDE_arccos_arcsin_equation_l1611_161195

theorem arccos_arcsin_equation : ∃ x : ℝ, Real.arccos (3 * x) - Real.arcsin (2 * x) = π / 6 := by
  sorry

end NUMINAMATH_CALUDE_arccos_arcsin_equation_l1611_161195


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1611_161178

theorem complex_equation_solution (z : ℂ) : z * (1 + Complex.I) = 2 * Complex.I → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1611_161178


namespace NUMINAMATH_CALUDE_solution_check_l1611_161168

theorem solution_check (x : ℝ) : x = 2 →
  (2 * x - 4 = 0) ∧ 
  (3 * x + 6 ≠ 0) ∧ 
  (2 * x + 4 ≠ 0) ∧ 
  (1/2 * x ≠ -4) := by
sorry

end NUMINAMATH_CALUDE_solution_check_l1611_161168


namespace NUMINAMATH_CALUDE_salary_increase_l1611_161137

theorem salary_increase (num_employees : ℕ) (initial_avg : ℚ) (manager_salary : ℚ) :
  num_employees = 20 →
  initial_avg = 1600 →
  manager_salary = 3700 →
  let total_salary := num_employees * initial_avg
  let new_total := total_salary + manager_salary
  let new_avg := new_total / (num_employees + 1)
  new_avg - initial_avg = 100 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l1611_161137


namespace NUMINAMATH_CALUDE_power_negative_multiply_l1611_161100

theorem power_negative_multiply (m : ℝ) : (-m)^2 * m^5 = m^7 := by
  sorry

end NUMINAMATH_CALUDE_power_negative_multiply_l1611_161100


namespace NUMINAMATH_CALUDE_max_parts_formula_max_parts_special_cases_l1611_161128

/-- The maximum number of parts a plane can be divided into by n lines -/
def max_parts (n : ℕ) : ℕ := (n^2 + n + 2) / 2

/-- Theorem: The maximum number of parts a plane can be divided into by n lines is (n^2 + n + 2) / 2 -/
theorem max_parts_formula (n : ℕ) : max_parts n = (n^2 + n + 2) / 2 := by
  sorry

/-- Corollary: Special cases for n = 1, 2, 3, and 4 -/
theorem max_parts_special_cases :
  (max_parts 1 = 2) ∧
  (max_parts 2 = 4) ∧
  (max_parts 3 = 7) ∧
  (max_parts 4 = 11) := by
  sorry

end NUMINAMATH_CALUDE_max_parts_formula_max_parts_special_cases_l1611_161128


namespace NUMINAMATH_CALUDE_exists_n_power_half_eq_ten_l1611_161162

theorem exists_n_power_half_eq_ten :
  ∃ n : ℝ, n > 0 ∧ n ^ (n / 2) = 10 := by
sorry

end NUMINAMATH_CALUDE_exists_n_power_half_eq_ten_l1611_161162


namespace NUMINAMATH_CALUDE_gcf_90_150_l1611_161105

theorem gcf_90_150 : Nat.gcd 90 150 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_90_150_l1611_161105


namespace NUMINAMATH_CALUDE_log_difference_equals_two_l1611_161132

theorem log_difference_equals_two :
  (Real.log 80 / Real.log 2) / (Real.log 2 / Real.log 40) -
  (Real.log 160 / Real.log 2) / (Real.log 2 / Real.log 20) = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_difference_equals_two_l1611_161132


namespace NUMINAMATH_CALUDE_cricket_team_matches_l1611_161159

/-- Proves that the total number of matches played by a cricket team in August is 250,
    given the initial and final winning percentages and the number of matches won during a winning streak. -/
theorem cricket_team_matches : 
  ∀ (initial_win_percent : ℝ) (final_win_percent : ℝ) (streak_wins : ℕ),
    initial_win_percent = 0.20 →
    final_win_percent = 0.52 →
    streak_wins = 80 →
    ∃ (total_matches : ℕ),
      total_matches = 250 ∧
      (initial_win_percent * total_matches + streak_wins) / total_matches = final_win_percent :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_matches_l1611_161159


namespace NUMINAMATH_CALUDE_snooker_tournament_revenue_l1611_161115

theorem snooker_tournament_revenue :
  let total_tickets : ℕ := 320
  let vip_price : ℚ := 40
  let general_price : ℚ := 10
  let vip_tickets : ℕ := (total_tickets - 148) / 2
  let general_tickets : ℕ := (total_tickets + 148) / 2
  (vip_price * vip_tickets + general_price * general_tickets : ℚ) = 5780 :=
by sorry

end NUMINAMATH_CALUDE_snooker_tournament_revenue_l1611_161115


namespace NUMINAMATH_CALUDE_solution_set_implies_a_and_b_l1611_161121

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * x + 3

-- Define the theorem
theorem solution_set_implies_a_and_b :
  ∀ a b : ℝ, 
  (∀ x : ℝ, f a x > 0 ↔ b < x ∧ x < 1) →
  (a = -7 ∧ b = -3/7) := by
sorry

-- Note: The second part of the problem is not included in the Lean statement
-- as it relies on the solution of the first part, which should not be assumed
-- in the theorem statement according to the given criteria.

end NUMINAMATH_CALUDE_solution_set_implies_a_and_b_l1611_161121


namespace NUMINAMATH_CALUDE_second_hand_large_division_time_l1611_161181

/-- The number of large divisions on a clock face -/
def large_divisions : ℕ := 12

/-- The number of small divisions in each large division -/
def small_divisions_per_large : ℕ := 5

/-- The time (in seconds) it takes for the second hand to move one small division -/
def time_per_small_division : ℕ := 1

/-- The time it takes for the second hand to move one large division -/
def time_for_large_division : ℕ := small_divisions_per_large * time_per_small_division

theorem second_hand_large_division_time :
  time_for_large_division = 5 := by sorry

end NUMINAMATH_CALUDE_second_hand_large_division_time_l1611_161181


namespace NUMINAMATH_CALUDE_equation_solution_l1611_161184

theorem equation_solution : 
  ∃! x : ℝ, (2 : ℝ) / (x + 3) + (3 * x) / (x + 3) - (5 : ℝ) / (x + 3) = 4 ∧ x = -15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1611_161184


namespace NUMINAMATH_CALUDE_savings_percentage_l1611_161189

theorem savings_percentage (income_year1 : ℝ) (savings_year1 : ℝ) 
  (h1 : savings_year1 > 0)
  (h2 : income_year1 > savings_year1)
  (h3 : (income_year1 - savings_year1) + (1.35 * income_year1 - 2 * savings_year1) = 
        2 * (income_year1 - savings_year1)) : 
  savings_year1 / income_year1 = 0.35 := by
sorry

end NUMINAMATH_CALUDE_savings_percentage_l1611_161189


namespace NUMINAMATH_CALUDE_sum_of_dot_products_l1611_161163

/-- Given three points A, B, C on a plane, prove that the sum of their vector dot products is -25 -/
theorem sum_of_dot_products (A B C : ℝ × ℝ) : 
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  let CA := (A.1 - C.1, A.2 - C.2)
  (AB.1^2 + AB.2^2 = 3^2) →
  (BC.1^2 + BC.2^2 = 4^2) →
  (CA.1^2 + CA.2^2 = 5^2) →
  (AB.1 * BC.1 + AB.2 * BC.2) + (BC.1 * CA.1 + BC.2 * CA.2) + (CA.1 * AB.1 + CA.2 * AB.2) = -25 :=
by
  sorry


end NUMINAMATH_CALUDE_sum_of_dot_products_l1611_161163


namespace NUMINAMATH_CALUDE_simplify_fraction_l1611_161126

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1611_161126


namespace NUMINAMATH_CALUDE_exactly_two_integers_l1611_161138

/-- Define the function that we want to check for integrality --/
def f (n : ℕ) : ℚ :=
  (Nat.factorial (n^3 - 1)) / ((Nat.factorial n)^(n + 2))

/-- Predicate to check if a number is in the range [1, 50] --/
def in_range (n : ℕ) : Prop :=
  1 ≤ n ∧ n ≤ 50

/-- Predicate to check if f(n) is an integer --/
def is_integer (n : ℕ) : Prop :=
  ∃ k : ℤ, f n = k

/-- Main theorem statement --/
theorem exactly_two_integers :
  (∃ (S : Finset ℕ), S.card = 2 ∧ 
    (∀ n, n ∈ S ↔ (in_range n ∧ is_integer n)) ∧
    (∀ n, in_range n → is_integer n → n ∈ S)) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_integers_l1611_161138


namespace NUMINAMATH_CALUDE_vector_calculation_l1611_161109

def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-1, -2)

theorem vector_calculation : 
  (2 : ℝ) • vector_a - vector_b = (5, 8) := by sorry

end NUMINAMATH_CALUDE_vector_calculation_l1611_161109


namespace NUMINAMATH_CALUDE_triangle_area_and_side_l1611_161179

theorem triangle_area_and_side (a b c : ℝ) (A B C : ℝ) :
  b = 2 →
  c = Real.sqrt 3 →
  A = π / 6 →
  (1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) ∧
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →
  (1/2 * b * c * Real.sin A = Real.sqrt 3 / 2) ∧ (a = 1) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_and_side_l1611_161179


namespace NUMINAMATH_CALUDE_salt_concentration_change_l1611_161166

/-- Proves that adding 1.25 kg of pure salt to 20 kg of 15% saltwater results in 20% saltwater -/
theorem salt_concentration_change (initial_water : ℝ) (initial_concentration : ℝ) 
  (added_salt : ℝ) (final_concentration : ℝ) 
  (h1 : initial_water = 20)
  (h2 : initial_concentration = 0.15)
  (h3 : added_salt = 1.25)
  (h4 : final_concentration = 0.2) :
  initial_water * initial_concentration + added_salt = 
  (initial_water + added_salt) * final_concentration :=
by sorry

end NUMINAMATH_CALUDE_salt_concentration_change_l1611_161166


namespace NUMINAMATH_CALUDE_max_segment_length_l1611_161174

-- Define the circles and line
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_C (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 4
def line (x y : ℝ) : Prop := x + Real.sqrt 3 * y - 2 = 0

-- Define the condition PB ≥ 2PA
def condition (x y : ℝ) : Prop :=
  ((x - 4)^2 + y^2 - 4) ≥ 4 * (x^2 + y^2 - 1)

-- Theorem statement
theorem max_segment_length :
  ∃ (E F : ℝ × ℝ),
    line E.1 E.2 ∧ line F.1 F.2 ∧
    (∀ (P : ℝ × ℝ), line P.1 P.2 →
      (E.1 ≤ P.1 ∧ P.1 ≤ F.1) → condition P.1 P.2) ∧
    Real.sqrt ((E.1 - F.1)^2 + (E.2 - F.2)^2) = 2 * Real.sqrt 39 / 3 :=
sorry

end NUMINAMATH_CALUDE_max_segment_length_l1611_161174


namespace NUMINAMATH_CALUDE_max_pieces_20x24_cake_l1611_161130

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Represents a piece of cake -/
structure CakePiece where
  size : Dimensions

/-- Represents the whole cake -/
structure Cake where
  size : Dimensions

/-- Calculates the maximum number of pieces that can be cut from a cake -/
def maxPieces (cake : Cake) (piece : CakePiece) : ℕ :=
  let horizontal := (cake.size.length / piece.size.length) * (cake.size.width / piece.size.width)
  let vertical := (cake.size.length / piece.size.width) * (cake.size.width / piece.size.length)
  max horizontal vertical

theorem max_pieces_20x24_cake (cake : Cake) (piece : CakePiece) :
  cake.size = Dimensions.mk 20 24 →
  piece.size = Dimensions.mk 4 4 →
  maxPieces cake piece = 30 := by
  sorry

#eval maxPieces (Cake.mk (Dimensions.mk 20 24)) (CakePiece.mk (Dimensions.mk 4 4))

end NUMINAMATH_CALUDE_max_pieces_20x24_cake_l1611_161130


namespace NUMINAMATH_CALUDE_log_inequality_equivalence_l1611_161102

-- Define the logarithm with base 1/3
noncomputable def log_one_third (x : ℝ) : ℝ := Real.log x / Real.log (1/3)

-- State the theorem
theorem log_inequality_equivalence :
  ∀ x : ℝ, log_one_third (2*x - 1) > 1 ↔ 1/2 < x ∧ x < 2/3 :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_equivalence_l1611_161102


namespace NUMINAMATH_CALUDE_no_integer_square_root_l1611_161151

/-- The polynomial p(x) = x^4 + 6x^3 + 11x^2 + 13x + 37 -/
def p (x : ℤ) : ℤ := x^4 + 6*x^3 + 11*x^2 + 13*x + 37

/-- Theorem stating that there are no integer values of x such that p(x) is a perfect square -/
theorem no_integer_square_root : ∀ x : ℤ, ¬∃ y : ℤ, p x = y^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_square_root_l1611_161151


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l1611_161185

theorem greatest_divisor_four_consecutive_integers :
  ∃ (d : ℕ), d = 12 ∧ 
  (∀ (n : ℕ), n > 0 → d ∣ (n * (n + 1) * (n + 2) * (n + 3))) ∧
  (∀ (k : ℕ), k > d → ∃ (m : ℕ), m > 0 ∧ ¬(k ∣ (m * (m + 1) * (m + 2) * (m + 3)))) :=
by sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l1611_161185


namespace NUMINAMATH_CALUDE_correct_statement_l1611_161147

-- Define proposition P
def P : Prop := ∀ x : ℝ, x^2 - 4*x + 5 > 0

-- Define proposition q
def q : Prop := ∃ x : ℝ, x > 0 ∧ Real.cos x > 1

-- Theorem to prove
theorem correct_statement : P ∨ ¬q := by
  sorry

end NUMINAMATH_CALUDE_correct_statement_l1611_161147


namespace NUMINAMATH_CALUDE_room_tiling_l1611_161149

-- Define the room dimensions in centimeters
def room_length : ℕ := 544
def room_width : ℕ := 374

-- Define the function to calculate the least number of square tiles
def least_number_of_tiles (length width : ℕ) : ℕ :=
  let tile_size := Nat.gcd length width
  (length / tile_size) * (width / tile_size)

-- Theorem statement
theorem room_tiling :
  least_number_of_tiles room_length room_width = 176 := by
  sorry

end NUMINAMATH_CALUDE_room_tiling_l1611_161149


namespace NUMINAMATH_CALUDE_parabola_unique_values_l1611_161158

/-- Parabola passing through (1, 1) and tangent to y = x - 3 at (2, -1) -/
def parabola_conditions (a b c : ℝ) : Prop :=
  -- Passes through (1, 1)
  a + b + c = 1 ∧
  -- Passes through (2, -1)
  4*a + 2*b + c = -1 ∧
  -- Derivative at x = 2 equals slope of y = x - 3
  4*a + b = 1

/-- Theorem stating the unique values of a, b, and c satisfying the conditions -/
theorem parabola_unique_values :
  ∃! (a b c : ℝ), parabola_conditions a b c ∧ a = 3 ∧ b = -11 ∧ c = 9 :=
by sorry

end NUMINAMATH_CALUDE_parabola_unique_values_l1611_161158


namespace NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l1611_161119

theorem circle_tangent_to_x_axis (x y : ℝ) :
  let center : ℝ × ℝ := (-3, 4)
  let equation := (x + 3)^2 + (y - 4)^2 = 16
  let is_tangent_to_x_axis := ∃ (x₀ : ℝ), (x₀ + 3)^2 + 4^2 = 16 ∧ ∀ (y : ℝ), y ≠ 0 → (x₀ + 3)^2 + (y - 4)^2 > 16
  equation ∧ is_tangent_to_x_axis :=
by
  sorry


end NUMINAMATH_CALUDE_circle_tangent_to_x_axis_l1611_161119


namespace NUMINAMATH_CALUDE_union_complement_problem_l1611_161191

theorem union_complement_problem (U A B : Set Char) : 
  U = {'a', 'b', 'c', 'd', 'e'} →
  A = {'b', 'c', 'd'} →
  B = {'b', 'e'} →
  B ∪ (U \ A) = {'a', 'b', 'e'} := by
sorry

end NUMINAMATH_CALUDE_union_complement_problem_l1611_161191


namespace NUMINAMATH_CALUDE_complex_angle_in_second_quadrant_l1611_161164

theorem complex_angle_in_second_quadrant 
  (z : ℂ) (θ : ℝ) 
  (h1 : z = Complex.exp (θ * Complex.I))
  (h2 : Real.cos θ < 0)
  (h3 : Real.sin θ > 0) : 
  π / 2 < θ ∧ θ < π :=
by sorry

end NUMINAMATH_CALUDE_complex_angle_in_second_quadrant_l1611_161164


namespace NUMINAMATH_CALUDE_negative_integer_sum_square_twelve_l1611_161131

theorem negative_integer_sum_square_twelve (M : ℤ) : 
  M < 0 → M^2 + M = 12 → M = -4 := by sorry

end NUMINAMATH_CALUDE_negative_integer_sum_square_twelve_l1611_161131


namespace NUMINAMATH_CALUDE_distinct_feeding_sequences_l1611_161108

def number_of_pairs : ℕ := 5

def feeding_sequence (n : ℕ) : ℕ := 
  match n with
  | 0 => 1  -- The first animal (male lion) is fixed
  | 1 => number_of_pairs  -- First choice of female
  | k => if k % 2 = 0 then number_of_pairs - k / 2 else number_of_pairs - (k - 1) / 2

theorem distinct_feeding_sequences :
  (List.range (2 * number_of_pairs)).foldl (fun acc i => acc * feeding_sequence i) 1 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_distinct_feeding_sequences_l1611_161108


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_eccentricity_is_four_l1611_161103

/-- The eccentricity of a hyperbola with given conditions -/
theorem hyperbola_eccentricity : ℝ → ℝ → ℝ → Prop :=
  fun a b e =>
    -- Hyperbola equation
    (∀ x y, x^2 / a^2 - y^2 / b^2 = 1 →
    -- Parabola equation
    ∃ x₀, ∀ y, y^2 = 16 * x₀ →
    -- Right focus of hyperbola coincides with focus of parabola
    4 = (a^2 + b^2).sqrt →
    -- Eccentricity definition
    e = (a^2 + b^2).sqrt / a →
    -- Prove eccentricity is 4
    e = 4)

/-- The main theorem stating the eccentricity is 4 -/
theorem eccentricity_is_four :
  ∃ a b e, hyperbola_eccentricity a b e :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_eccentricity_is_four_l1611_161103


namespace NUMINAMATH_CALUDE_smallest_sum_is_337_dice_sum_theorem_l1611_161198

/-- Represents a set of symmetrical dice --/
structure DiceSet where
  num_dice : ℕ
  max_sum : ℕ
  min_sum : ℕ

/-- The property that the dice set can achieve a sum of 2022 --/
def can_sum_2022 (d : DiceSet) : Prop :=
  d.max_sum = 2022

/-- The property that each die is symmetrical (6-sided) --/
def symmetrical_dice (d : DiceSet) : Prop :=
  d.max_sum = 6 * d.num_dice ∧ d.min_sum = d.num_dice

/-- The theorem stating that the smallest possible sum is 337 --/
theorem smallest_sum_is_337 (d : DiceSet) 
  (h1 : can_sum_2022 d) 
  (h2 : symmetrical_dice d) : 
  d.min_sum = 337 := by
  sorry

/-- The main theorem combining all conditions --/
theorem dice_sum_theorem (d : DiceSet) 
  (h1 : can_sum_2022 d) 
  (h2 : symmetrical_dice d) : 
  ∃ (p : ℝ), p > 0 ∧ 
    (∃ (sum : ℕ), sum = 2022 ∧ sum ≤ d.max_sum) ∧
    (∃ (min_sum : ℕ), min_sum = 337 ∧ min_sum = d.min_sum) := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_is_337_dice_sum_theorem_l1611_161198


namespace NUMINAMATH_CALUDE_expression_value_l1611_161182

theorem expression_value (a b c : ℝ) : 
  a * (-2)^5 + b * (-2)^3 + c * (-2) - 5 = 7 →
  a * 2^5 + b * 2^3 + c * 2 - 5 = -17 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l1611_161182


namespace NUMINAMATH_CALUDE_tiling_coverage_l1611_161176

/-- Represents a tiling of a plane with hexagons and triangles -/
structure PlaneTiling where
  /-- The number of smaller triangles in each hexagon -/
  triangles_per_hexagon : ℕ
  /-- The number of smaller triangles that form larger triangles in each hexagon -/
  triangles_in_larger : ℕ

/-- Calculates the percentage of the plane covered by larger triangles -/
def percentage_covered (tiling : PlaneTiling) : ℚ :=
  (tiling.triangles_in_larger : ℚ) / (tiling.triangles_per_hexagon : ℚ) * 100

/-- Theorem stating that for the given tiling, 56% of the plane is covered by larger triangles -/
theorem tiling_coverage : 
  ∀ (t : PlaneTiling), 
  t.triangles_per_hexagon = 16 → 
  t.triangles_in_larger = 9 → 
  percentage_covered t = 56 := by
  sorry

end NUMINAMATH_CALUDE_tiling_coverage_l1611_161176


namespace NUMINAMATH_CALUDE_no_rational_solutions_for_positive_k_l1611_161194

theorem no_rational_solutions_for_positive_k : ¬ ∃ (k : ℕ+), ∃ (x : ℚ), k.val * x^2 + 16 * x + k.val = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solutions_for_positive_k_l1611_161194


namespace NUMINAMATH_CALUDE_final_S_value_l1611_161153

def sequence_A : ℕ → ℕ
  | 0 => 1
  | n + 1 => sequence_A n + 1

def sequence_S : ℕ → ℕ
  | 0 => 0
  | n + 1 => sequence_S n + sequence_A (n + 1)

theorem final_S_value :
  ∃ n : ℕ, sequence_S n ≤ 36 ∧ sequence_S (n + 1) > 36 ∧ sequence_S (n + 1) = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_final_S_value_l1611_161153


namespace NUMINAMATH_CALUDE_gcd_lcm_300_462_l1611_161175

theorem gcd_lcm_300_462 : 
  (Nat.gcd 300 462 = 6) ∧ (Nat.lcm 300 462 = 46200) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_300_462_l1611_161175


namespace NUMINAMATH_CALUDE_triangle_property_l1611_161170

open Real

theorem triangle_property (a b c : ℝ) (A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  a * sin B - Real.sqrt 3 * b * cos B * cos C = Real.sqrt 3 * c * (cos B)^2 →
  (B = π / 3 ∧
   (0 < C ∧ C < π / 2 → 1 < a^2 + b^2 ∧ a^2 + b^2 < 7)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l1611_161170


namespace NUMINAMATH_CALUDE_time_differences_not_constant_l1611_161127

/-- Represents the relationship between height and time for the sliding car experiment -/
def slide_data : List (ℝ × ℝ) :=
  [(10, 4.23), (20, 3.00), (30, 2.45), (40, 2.13), (50, 1.89), (60, 1.71), (70, 1.59)]

/-- Calculates the time difference between two consecutive measurements -/
def time_diff (data : List (ℝ × ℝ)) (i : ℕ) : ℝ :=
  match data.get? i, data.get? (i+1) with
  | some (_, t1), some (_, t2) => t1 - t2
  | _, _ => 0

/-- Theorem stating that time differences are not constant -/
theorem time_differences_not_constant : ∃ i j, i ≠ j ∧ i < slide_data.length - 1 ∧ j < slide_data.length - 1 ∧ time_diff slide_data i ≠ time_diff slide_data j :=
sorry

end NUMINAMATH_CALUDE_time_differences_not_constant_l1611_161127


namespace NUMINAMATH_CALUDE_triangle_area_equalities_l1611_161111

theorem triangle_area_equalities (S r R A B C : ℝ) 
  (h_positive : S > 0 ∧ r > 0 ∧ R > 0)
  (h_angles : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h_sum : A + B + C = π)
  (h_area : S = r * R * (Real.sin A + Real.sin B + Real.sin C)) :
  S = r * R * (Real.sin A + Real.sin B + Real.sin C) ∧
  S = 4 * r * R * Real.cos (A/2) * Real.cos (B/2) * Real.cos (C/2) ∧
  S = (R^2 / 2) * (Real.sin (2*A) + Real.sin (2*B) + Real.sin (2*C)) ∧
  S = 2 * R^2 * Real.sin A * Real.sin B * Real.sin C := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_equalities_l1611_161111


namespace NUMINAMATH_CALUDE_meaningful_fraction_l1611_161124

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = x / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_meaningful_fraction_l1611_161124


namespace NUMINAMATH_CALUDE_problem_solution_l1611_161129

noncomputable def f (a x : ℝ) : ℝ :=
  if x < a then 2 * a - (x + 4 / x) else x - 4 / x

theorem problem_solution (a : ℝ) :
  (a = 1 → ∃! x, f a x = 3 ∧ x = 4) ∧
  (a ≤ -1 →
    (∃ x₁ x₂ x₃, x₁ < x₂ ∧ x₂ < x₃ ∧
      f a x₁ = 3 ∧ f a x₂ = 3 ∧ f a x₃ = 3 ∧
      x₃ - x₂ = x₂ - x₁) →
    a = -11/6) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l1611_161129


namespace NUMINAMATH_CALUDE_determinant_equality_l1611_161107

theorem determinant_equality (p q r s : ℝ) : 
  Matrix.det !![p, q; r, s] = 3 → 
  Matrix.det !![p + 2*r, q + 2*s; r, s] = 3 := by
sorry

end NUMINAMATH_CALUDE_determinant_equality_l1611_161107


namespace NUMINAMATH_CALUDE_remaining_time_for_finger_exerciser_l1611_161139

theorem remaining_time_for_finger_exerciser (total_time piano_time writing_time history_time : ℕ) :
  total_time = 120 ∧ piano_time = 30 ∧ writing_time = 25 ∧ history_time = 38 →
  total_time - (piano_time + writing_time + history_time) = 27 := by
sorry

end NUMINAMATH_CALUDE_remaining_time_for_finger_exerciser_l1611_161139
