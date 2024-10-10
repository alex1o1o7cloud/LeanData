import Mathlib

namespace child_running_speed_l3203_320341

/-- The child's running speed in meters per minute -/
def child_speed : ℝ := 74

/-- The sidewalk's speed in meters per minute -/
def sidewalk_speed : ℝ := child_speed - 55

theorem child_running_speed 
  (h1 : (child_speed + sidewalk_speed) * 4 = 372)
  (h2 : (child_speed - sidewalk_speed) * 3 = 165) :
  child_speed = 74 := by sorry

end child_running_speed_l3203_320341


namespace kathleen_savings_problem_l3203_320314

/-- Kathleen's savings and spending problem -/
theorem kathleen_savings_problem (june july august clothes_cost remaining : ℕ)
  (h_june : june = 21)
  (h_july : july = 46)
  (h_august : august = 45)
  (h_clothes : clothes_cost = 54)
  (h_remaining : remaining = 46) :
  ∃ (school_supplies : ℕ),
    june + july + august = clothes_cost + school_supplies + remaining ∧ 
    school_supplies = 12 := by
  sorry

end kathleen_savings_problem_l3203_320314


namespace mean_of_remaining_two_l3203_320387

def numbers : List ℕ := [1870, 1996, 2022, 2028, 2112, 2124]

theorem mean_of_remaining_two (four_numbers : List ℕ) 
  (h1 : four_numbers.length = 4)
  (h2 : four_numbers.all (· ∈ numbers))
  (h3 : (four_numbers.sum : ℚ) / 4 = 2011) :
  let remaining_two := numbers.filter (λ x => x ∉ four_numbers)
  (remaining_two.sum : ℚ) / 2 = 2054 := by
sorry

end mean_of_remaining_two_l3203_320387


namespace sector_area_l3203_320347

/-- Given a sector with perimeter 8 and central angle 2 radians, its area is 4 -/
theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (radius : ℝ) (arc_length : ℝ) :
  perimeter = 8 →
  central_angle = 2 →
  perimeter = arc_length + 2 * radius →
  arc_length = radius * central_angle →
  (1 / 2) * radius * arc_length = 4 := by
  sorry

end sector_area_l3203_320347


namespace oil_price_reduction_l3203_320350

/-- Given a 20% reduction in the price of oil, if a housewife can obtain 5 kg more for Rs. 800 after the reduction, then the reduced price per kg is Rs. 32. -/
theorem oil_price_reduction (P : ℝ) (h1 : P > 0) :
  let R := 0.8 * P
  800 / R - 800 / P = 5 →
  R = 32 := by sorry

end oil_price_reduction_l3203_320350


namespace laptop_price_theorem_l3203_320301

/-- The sticker price of the laptop -/
def stickerPrice : ℝ := 900

/-- The price at Store P -/
def storePPrice (price : ℝ) : ℝ := 0.8 * price - 120

/-- The price at Store Q -/
def storeQPrice (price : ℝ) : ℝ := 0.7 * price

/-- Theorem stating that the sticker price satisfies the given conditions -/
theorem laptop_price_theorem :
  storeQPrice stickerPrice - storePPrice stickerPrice = 30 := by
  sorry

#check laptop_price_theorem

end laptop_price_theorem_l3203_320301


namespace total_pencils_l3203_320325

/-- Given the ages and pencil counts of Asaf and Alexander, prove their total pencil count -/
theorem total_pencils (asaf_age alexander_age asaf_pencils alexander_pencils : ℕ) : 
  asaf_age = 50 →
  asaf_age + alexander_age = 140 →
  alexander_age - asaf_age = asaf_pencils / 2 →
  alexander_pencils = asaf_pencils + 60 →
  asaf_pencils + alexander_pencils = 220 := by
sorry


end total_pencils_l3203_320325


namespace find_m_l3203_320388

theorem find_m (A B : Set ℕ) (m : ℕ) : 
  A = {1, 2, 3} →
  B = {2, m, 4} →
  A ∩ B = {2, 3} →
  m = 3 := by
sorry

end find_m_l3203_320388


namespace factorial_ratio_squared_l3203_320315

theorem factorial_ratio_squared (M : ℕ) : 
  (Nat.factorial (M + 1) : ℚ) / (Nat.factorial (M + 2) : ℚ)^2 = 1 / ((M + 2 : ℚ)^2) := by
  sorry

end factorial_ratio_squared_l3203_320315


namespace fraction_division_multiplication_l3203_320304

theorem fraction_division_multiplication : 
  (5 : ℚ) / 6 / (2 / 3) * (4 / 9) = 5 / 9 := by sorry

end fraction_division_multiplication_l3203_320304


namespace charms_per_necklace_is_10_l3203_320311

/-- The number of charms used to make each necklace -/
def charms_per_necklace : ℕ := sorry

/-- The cost of each charm in dollars -/
def charm_cost : ℕ := 15

/-- The selling price of each necklace in dollars -/
def necklace_price : ℕ := 200

/-- The number of necklaces sold -/
def necklaces_sold : ℕ := 30

/-- The total profit in dollars -/
def total_profit : ℕ := 1500

theorem charms_per_necklace_is_10 :
  charms_per_necklace = 10 ∧
  charm_cost = 15 ∧
  necklace_price = 200 ∧
  necklaces_sold = 30 ∧
  total_profit = 1500 ∧
  necklaces_sold * (necklace_price - charms_per_necklace * charm_cost) = total_profit :=
sorry

end charms_per_necklace_is_10_l3203_320311


namespace product_expansion_sum_l3203_320348

theorem product_expansion_sum (a b c d : ℝ) : 
  (∀ x, (2*x^2 - 3*x + 5)*(8 - 3*x) = a*x^3 + b*x^2 + c*x + d) →
  9*a + 3*b + 6*c + d = -173 := by
sorry

end product_expansion_sum_l3203_320348


namespace fixed_point_parabola_l3203_320339

theorem fixed_point_parabola :
  ∀ (k : ℝ), 225 = 9 * (5 : ℝ)^2 + k * 5 - 5 * k := by
  sorry

end fixed_point_parabola_l3203_320339


namespace sum_of_solutions_l3203_320307

theorem sum_of_solutions (x : ℝ) : (|3 * x - 9| = 6) → (∃ y : ℝ, (|3 * y - 9| = 6) ∧ x + y = 6) :=
by sorry

end sum_of_solutions_l3203_320307


namespace smallest_x_absolute_value_equation_l3203_320319

theorem smallest_x_absolute_value_equation : 
  (∃ x : ℝ, 2 * |x - 10| = 24) ∧ 
  (∀ x : ℝ, 2 * |x - 10| = 24 → x ≥ -2) ∧
  (2 * |-2 - 10| = 24) := by
  sorry

end smallest_x_absolute_value_equation_l3203_320319


namespace fourth_coefficient_equals_five_l3203_320305

theorem fourth_coefficient_equals_five (x a₁ a₂ a₃ a₄ a₅ aₙ : ℝ) :
  (∀ x, x^5 = aₙ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + a₅*(x-1)^5) →
  a₄ = 5 := by
sorry

end fourth_coefficient_equals_five_l3203_320305


namespace max_xy_on_line_segment_l3203_320337

/-- The maximum value of xy for a point P(x,y) on the line segment AB, where A(3,0) and B(0,4) -/
theorem max_xy_on_line_segment : ∀ x y : ℝ, 
  (x / 3 + y / 4 = 1) → -- Point P(x,y) is on the line segment AB
  (x ≥ 0 ∧ y ≥ 0) →    -- P is between A and B (non-negative coordinates)
  (x ≤ 3 ∧ y ≤ 4) →    -- P is between A and B (upper bounds)
  x * y ≤ 3 :=         -- The maximum value of xy is 3
by
  sorry

#check max_xy_on_line_segment

end max_xy_on_line_segment_l3203_320337


namespace wheel_spinner_probability_l3203_320323

theorem wheel_spinner_probability (p_E p_F p_G p_H p_I : ℚ) : 
  p_E = 1/5 →
  p_F = 1/4 →
  p_G = p_H →
  p_E + p_F + p_G + p_H + p_I = 1 →
  p_H = 9/40 := by
sorry

end wheel_spinner_probability_l3203_320323


namespace circle_center_coordinate_sum_l3203_320398

theorem circle_center_coordinate_sum :
  ∀ (x y : ℝ), x^2 + y^2 = 6*x - 10*y + 24 →
  ∃ (center_x center_y : ℝ),
    (∀ (p_x p_y : ℝ), (p_x - center_x)^2 + (p_y - center_y)^2 = (x - center_x)^2 + (y - center_y)^2) ∧
    center_x + center_y = -2 :=
by sorry

end circle_center_coordinate_sum_l3203_320398


namespace unknown_blanket_rate_l3203_320312

theorem unknown_blanket_rate (price1 price2 avg_price : ℚ) 
  (count1 count2 count_unknown : ℕ) : 
  price1 = 100 → 
  price2 = 150 → 
  avg_price = 150 → 
  count1 = 4 → 
  count2 = 5 → 
  count_unknown = 2 → 
  (count1 * price1 + count2 * price2 + count_unknown * 
    ((count1 + count2 + count_unknown) * avg_price - count1 * price1 - count2 * price2) / count_unknown) / 
    (count1 + count2 + count_unknown) = avg_price → 
  ((count1 + count2 + count_unknown) * avg_price - count1 * price1 - count2 * price2) / count_unknown = 250 :=
by sorry

end unknown_blanket_rate_l3203_320312


namespace fifth_pythagorean_triple_l3203_320346

/-- Generates the nth Pythagorean triple based on the given pattern -/
def pythagoreanTriple (n : ℕ) : ℕ × ℕ × ℕ :=
  let a := 2 * n + 1
  let b := 2 * n * (n + 1)
  let c := 2 * n * (n + 1) + 1
  (a, b, c)

/-- Checks if a triple of natural numbers forms a Pythagorean triple -/
def isPythagoreanTriple (triple : ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c) := triple
  a * a + b * b = c * c

theorem fifth_pythagorean_triple :
  let triple := pythagoreanTriple 5
  triple = (11, 60, 61) ∧ isPythagoreanTriple triple :=
by sorry

end fifth_pythagorean_triple_l3203_320346


namespace reema_loan_interest_l3203_320389

/-- Calculates simple interest given principal, rate, and time -/
def simple_interest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem reema_loan_interest :
  let principal : ℚ := 1500
  let rate : ℚ := 7
  let time : ℚ := rate
  simple_interest principal rate time = 735 := by sorry

end reema_loan_interest_l3203_320389


namespace triangle_angle_inequality_l3203_320331

theorem triangle_angle_inequality (A B C : ℝ) (h_triangle : A + B + C = π) (h_positive : A > 0 ∧ B > 0 ∧ C > 0) :
  2 * (Real.sin A / A + Real.sin B / B + Real.sin C / C) ≤
  (1/B + 1/C) * Real.sin A + (1/C + 1/A) * Real.sin B + (1/A + 1/B) * Real.sin C :=
by sorry

end triangle_angle_inequality_l3203_320331


namespace maximize_product_l3203_320300

theorem maximize_product (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_sum : x + y = 40) :
  x^6 * y^3 ≤ 24^6 * 16^3 ∧
  (x^6 * y^3 = 24^6 * 16^3 ↔ x = 24 ∧ y = 16) :=
sorry

end maximize_product_l3203_320300


namespace geometric_sum_problem_l3203_320338

/-- Sum of a finite geometric series -/
def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The problem statement -/
theorem geometric_sum_problem : 
  let a : ℚ := 1/4
  let r : ℚ := 1/4
  let n : ℕ := 6
  geometric_sum a r n = 4095/12288 := by
sorry

end geometric_sum_problem_l3203_320338


namespace inequality_and_equality_condition_l3203_320308

theorem inequality_and_equality_condition 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℕ) 
  (ha₁ : a₁ > 0) (ha₂ : a₂ > 0) (ha₃ : a₃ > 0) 
  (hb₁ : b₁ > 0) (hb₂ : b₂ > 0) (hb₃ : b₃ > 0) : 
  (((a₁ * b₂ + a₁ * b₃ + a₂ * b₁ + a₂ * b₃ + a₃ * b₁ + a₃ * b₂) : ℚ) ^ 2 ≥ 
   4 * ((a₁ * a₂ + a₂ * a₃ + a₃ * a₁) : ℚ) * (b₁ * b₂ + b₂ * b₃ + b₃ * b₁)) ∧
  (((a₁ * b₂ + a₁ * b₃ + a₂ * b₁ + a₂ * b₃ + a₃ * b₁ + a₃ * b₂) : ℚ) ^ 2 = 
   4 * ((a₁ * a₂ + a₂ * a₃ + a₃ * a₁) : ℚ) * (b₁ * b₂ + b₂ * b₃ + b₃ * b₁) ↔ 
   (a₁ : ℚ) / b₁ = (a₂ : ℚ) / b₂ ∧ (a₂ : ℚ) / b₂ = (a₃ : ℚ) / b₃) := by
  sorry

end inequality_and_equality_condition_l3203_320308


namespace right_triangle_height_properties_l3203_320335

/-- Properties of a right-angled triangle with height to hypotenuse --/
theorem right_triangle_height_properties
  (a b c h p q : ℝ)
  (right_triangle : a^2 + b^2 = c^2)
  (height_divides_hypotenuse : p + q = c)
  (height_forms_similar_triangles : h^2 = a * b) :
  h^2 = p * q ∧ a^2 = p * c ∧ b^2 = q * c ∧ p / q = (a / b)^2 := by
  sorry

end right_triangle_height_properties_l3203_320335


namespace h_zero_at_seven_fifths_l3203_320395

/-- The function h(x) = 5x - 7 -/
def h (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: The value of b that satisfies h(b) = 0 is 7/5 -/
theorem h_zero_at_seven_fifths : ∃ b : ℝ, h b = 0 ∧ b = 7/5 := by
  sorry

end h_zero_at_seven_fifths_l3203_320395


namespace document_typing_time_l3203_320377

theorem document_typing_time (barbara_speed jim_speed : ℕ) (document_length : ℕ) (jim_time : ℕ) :
  barbara_speed = 172 →
  jim_speed = 100 →
  document_length = 3440 →
  jim_time = 20 →
  ∃ t : ℕ, t < jim_time ∧ t * (barbara_speed + jim_speed) ≥ document_length :=
by sorry

end document_typing_time_l3203_320377


namespace unique_pizza_combinations_l3203_320344

def number_of_toppings : ℕ := 8
def toppings_per_pizza : ℕ := 5

theorem unique_pizza_combinations : 
  (number_of_toppings.choose toppings_per_pizza) = 56 := by
  sorry

end unique_pizza_combinations_l3203_320344


namespace triangle_ABC_properties_l3203_320385

theorem triangle_ABC_properties (A B C : ℝ) (p : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π → -- Triangle ABC exists
  (∃ x y : ℝ, x^2 + (x+1)*p + 1 = 0 ∧ y^2 + (y+1)*p + 1 = 0 ∧ x = Real.tan A ∧ y = Real.tan B) →
  C = 3*π/4 ∧ p ∈ Set.Ioo (-2 : ℝ) (2 - 2*Real.sqrt 2) := by
  sorry

end triangle_ABC_properties_l3203_320385


namespace abs_five_minus_e_equals_five_minus_e_l3203_320322

theorem abs_five_minus_e_equals_five_minus_e :
  |5 - Real.exp 1| = 5 - Real.exp 1 :=
by sorry

end abs_five_minus_e_equals_five_minus_e_l3203_320322


namespace pentagon_angle_problem_l3203_320399

def pentagon_largest_angle (P Q R S T : ℝ) : Prop :=
  -- Sum of angles in a pentagon is 540°
  P + Q + R + S + T = 540 ∧
  -- Given conditions
  P = 55 ∧
  Q = 120 ∧
  R = S ∧
  T = 2 * R + 20 ∧
  -- Largest angle is 192.5°
  max P (max Q (max R (max S T))) = 192.5

theorem pentagon_angle_problem :
  ∃ (P Q R S T : ℝ), pentagon_largest_angle P Q R S T := by
  sorry

end pentagon_angle_problem_l3203_320399


namespace frog_jump_probability_l3203_320318

/-- Represents a jump in a random direction -/
structure Jump where
  length : ℝ
  direction : ℝ × ℝ

/-- Represents the frog's journey -/
def FrogJourney := List Jump

/-- Calculate the final position of the frog after a series of jumps -/
def finalPosition (journey : FrogJourney) : ℝ × ℝ := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Generate a random journey for the frog -/
def randomJourney : FrogJourney := sorry

/-- Probability that the frog's final position is within 2 meters of the start -/
def probabilityWithinTwoMeters : ℝ := sorry

/-- Theorem stating the probability is approximately 1/10 -/
theorem frog_jump_probability :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |probabilityWithinTwoMeters - 1/10| < ε := by sorry

end frog_jump_probability_l3203_320318


namespace simplify_and_evaluate_l3203_320357

theorem simplify_and_evaluate (x : ℝ) (h : x = -2) :
  -(2 * x^2 + 3 * x) + 2 * (4 * x + x^2) = -10 := by
  sorry

end simplify_and_evaluate_l3203_320357


namespace value_of_expression_l3203_320384

theorem value_of_expression (a : ℝ) (h : a - 1/a = 1) : a^2 - a + 2 = 3 := by
  sorry

end value_of_expression_l3203_320384


namespace train_cross_platform_time_l3203_320371

def train_length : ℝ := 300
def platform_length : ℝ := 300
def time_cross_pole : ℝ := 18

theorem train_cross_platform_time :
  let train_speed := train_length / time_cross_pole
  let total_distance := train_length + platform_length
  total_distance / train_speed = 36 := by
  sorry

end train_cross_platform_time_l3203_320371


namespace binomial_p_value_l3203_320332

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  mean : ℝ
  variance : ℝ
  mean_eq : mean = n * p
  variance_eq : variance = n * p * (1 - p)

/-- Theorem stating the value of p for a binomial random variable with given mean and variance -/
theorem binomial_p_value (ξ : BinomialRV) 
  (h_mean : ξ.mean = 300)
  (h_var : ξ.variance = 200) :
  ξ.p = 1/3 := by
  sorry

#check binomial_p_value

end binomial_p_value_l3203_320332


namespace towels_used_is_285_towels_used_le_total_towels_l3203_320326

/-- Calculates the total number of towels used in a gym over 4 hours -/
def totalTowelsUsed (firstHourGuests : ℕ) : ℕ :=
  let secondHourGuests := firstHourGuests + (firstHourGuests * 20 / 100)
  let thirdHourGuests := secondHourGuests + (secondHourGuests * 25 / 100)
  let fourthHourGuests := thirdHourGuests + (thirdHourGuests * 1 / 3)
  firstHourGuests + secondHourGuests + thirdHourGuests + fourthHourGuests

/-- Theorem stating that the total number of towels used is 285 -/
theorem towels_used_is_285 :
  totalTowelsUsed 50 = 285 := by
  sorry

/-- The number of towels laid out daily -/
def totalTowels : ℕ := 300

/-- Theorem stating that the number of towels used is less than or equal to the total towels -/
theorem towels_used_le_total_towels :
  totalTowelsUsed 50 ≤ totalTowels := by
  sorry

end towels_used_is_285_towels_used_le_total_towels_l3203_320326


namespace isosceles_triangle_on_hyperbola_x_coordinate_range_l3203_320313

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 6 - y^2 / 3 = 1

-- Define the isosceles triangle
structure IsoscelesTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  isIsosceles : (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

-- Theorem statement
theorem isosceles_triangle_on_hyperbola_x_coordinate_range 
  (triangle : IsoscelesTriangle)
  (hA : hyperbola triangle.A.1 triangle.A.2)
  (hB : hyperbola triangle.B.1 triangle.B.2)
  (hC : triangle.C.2 = 0)
  (hAB_not_perpendicular : (triangle.A.2 - triangle.B.2) ≠ 0) :
  triangle.C.1 > (3/2) * Real.sqrt 6 :=
by sorry

end isosceles_triangle_on_hyperbola_x_coordinate_range_l3203_320313


namespace log_64_4_l3203_320320

theorem log_64_4 : Real.log 4 / Real.log 64 = 1 / 3 := by
  sorry

end log_64_4_l3203_320320


namespace exactly_one_two_black_mutually_exclusive_not_contradictory_l3203_320302

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Black

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The set of all possible outcomes when drawing two balls from the pocket -/
def allOutcomes : Finset DrawOutcome := sorry

/-- The event of drawing exactly one black ball -/
def exactlyOneBlack (outcome : DrawOutcome) : Prop :=
  (outcome.first = BallColor.Black ∧ outcome.second = BallColor.Red) ∨
  (outcome.first = BallColor.Red ∧ outcome.second = BallColor.Black)

/-- The event of drawing exactly two black balls -/
def exactlyTwoBlack (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.Black ∧ outcome.second = BallColor.Black

/-- The theorem stating that "Exactly one black ball" and "Exactly two black balls" are mutually exclusive but not contradictory -/
theorem exactly_one_two_black_mutually_exclusive_not_contradictory :
  (∀ outcome : DrawOutcome, ¬(exactlyOneBlack outcome ∧ exactlyTwoBlack outcome)) ∧
  (∃ outcome : DrawOutcome, exactlyOneBlack outcome) ∧
  (∃ outcome : DrawOutcome, exactlyTwoBlack outcome) :=
sorry

end exactly_one_two_black_mutually_exclusive_not_contradictory_l3203_320302


namespace square_diff_ratio_equals_one_third_l3203_320360

theorem square_diff_ratio_equals_one_third :
  (2025^2 - 2018^2) / (2032^2 - 2011^2) = 1/3 := by
sorry

end square_diff_ratio_equals_one_third_l3203_320360


namespace imaginary_unit_equation_l3203_320355

theorem imaginary_unit_equation : Complex.I ^ 3 - 2 / Complex.I = Complex.I := by
  sorry

end imaginary_unit_equation_l3203_320355


namespace updated_mean_after_decrement_l3203_320368

theorem updated_mean_after_decrement (n : ℕ) (original_mean decrement : ℝ) :
  n > 0 →
  n = 50 →
  original_mean = 200 →
  decrement = 15 →
  (n * original_mean - n * decrement) / n = 185 := by
  sorry

end updated_mean_after_decrement_l3203_320368


namespace solution_set_when_a_is_one_f_always_greater_equal_g_iff_l3203_320392

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |2*x + 1|
def g (x : ℝ) : ℝ := x + 2

-- Statement for part (1)
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ g x} = {x : ℝ | 0 ≤ x ∧ x ≤ 2/3} := by sorry

-- Statement for part (2)
theorem f_always_greater_equal_g_iff (a : ℝ) :
  (∀ x : ℝ, f a x ≥ g x) ↔ a ≥ 2 := by sorry

-- Condition that a > 0
axiom a_positive : ∀ a : ℝ, a > 0

end solution_set_when_a_is_one_f_always_greater_equal_g_iff_l3203_320392


namespace problem_solution_l3203_320363

theorem problem_solution (x : ℝ) (h : x = 6) : (x^6 - 17*x^3 + 72) / (x^3 - 8) = 207 := by
  sorry

end problem_solution_l3203_320363


namespace wrong_mark_calculation_l3203_320336

theorem wrong_mark_calculation (n : ℕ) (correct_mark : ℝ) (average_increase : ℝ) : 
  n = 56 → 
  correct_mark = 45 → 
  average_increase = 1/2 → 
  ∃ x : ℝ, x - correct_mark = n * average_increase ∧ x = 73 := by
sorry

end wrong_mark_calculation_l3203_320336


namespace triangle_circles_area_sum_l3203_320328

/-- The sum of the areas of three mutually externally tangent circles 
    centered at the vertices of a 6-8-10 right triangle is 56π. -/
theorem triangle_circles_area_sum : 
  ∀ (r s t : ℝ), 
    r + s = 6 →
    r + t = 8 →
    s + t = 10 →
    r > 0 → s > 0 → t > 0 →
    π * (r^2 + s^2 + t^2) = 56 * π := by
  sorry

end triangle_circles_area_sum_l3203_320328


namespace integer_between_sqrt3_plus_1_and_sqrt11_l3203_320374

theorem integer_between_sqrt3_plus_1_and_sqrt11 :
  ∃! n : ℤ, (Real.sqrt 3 + 1 < n) ∧ (n < Real.sqrt 11) :=
by
  -- We assume the following inequalities as given:
  have h1 : 1 < Real.sqrt 3 ∧ Real.sqrt 3 < 2 := sorry
  have h2 : 3 < Real.sqrt 11 ∧ Real.sqrt 11 < 4 := sorry

  -- The proof goes here
  sorry

end integer_between_sqrt3_plus_1_and_sqrt11_l3203_320374


namespace import_tax_problem_l3203_320382

/-- Calculates the import tax percentage given the total value, tax-free portion, and tax amount -/
def import_tax_percentage (total_value tax_free_portion tax_amount : ℚ) : ℚ :=
  (tax_amount / (total_value - tax_free_portion)) * 100

/-- Proves that the import tax percentage is 7% given the problem conditions -/
theorem import_tax_problem :
  let total_value : ℚ := 2610
  let tax_free_portion : ℚ := 1000
  let tax_amount : ℚ := 1127/10
by
  sorry


end import_tax_problem_l3203_320382


namespace f_increasing_on_interval_l3203_320394

-- Define the function f(x) = x^2 + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 2 → f x < f y :=
sorry

end f_increasing_on_interval_l3203_320394


namespace constant_volume_l3203_320378

/-- Represents a line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  vertices : Fin 4 → ℝ × ℝ × ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ := sorry

/-- Represents the configuration of the tetrahedron with moving vertices -/
structure MovingTetrahedron where
  fixedEdge : Line3D
  movingVertex1 : Line3D
  movingVertex2 : Line3D
  initialTetrahedron : Tetrahedron

/-- Checks if three lines are parallel -/
def areLinesParallel (l1 l2 l3 : Line3D) : Prop := sorry

/-- Calculates the tetrahedron at a given time t -/
def tetrahedronAtTime (mt : MovingTetrahedron) (t : ℝ) : Tetrahedron := sorry

/-- Theorem stating that the volume remains constant -/
theorem constant_volume (mt : MovingTetrahedron) 
  (h : areLinesParallel mt.fixedEdge mt.movingVertex1 mt.movingVertex2) :
  ∀ t : ℝ, tetrahedronVolume (tetrahedronAtTime mt t) = tetrahedronVolume mt.initialTetrahedron :=
sorry

end constant_volume_l3203_320378


namespace coin_division_theorem_l3203_320373

theorem coin_division_theorem :
  let sum_20 := (20 * 21) / 2
  let sum_20_plus_100 := sum_20 + 100
  (sum_20 % 3 = 0) ∧ (sum_20_plus_100 % 3 ≠ 0) := by
  sorry

end coin_division_theorem_l3203_320373


namespace complex_number_in_fourth_quadrant_l3203_320309

/-- The point corresponding to the complex number (a^2 - 4a + 5) + (-b^2 + 2b - 6)i 
    is in the fourth quadrant for all real a and b. -/
theorem complex_number_in_fourth_quadrant (a b : ℝ) : 
  (a^2 - 4*a + 5 > 0) ∧ (-b^2 + 2*b - 6 < 0) := by
  sorry

end complex_number_in_fourth_quadrant_l3203_320309


namespace total_apples_collected_l3203_320391

def apples_per_day : ℕ := 4
def days : ℕ := 30
def remaining_apples : ℕ := 230

theorem total_apples_collected :
  apples_per_day * days + remaining_apples = 350 := by
  sorry

end total_apples_collected_l3203_320391


namespace odd_function_sum_condition_l3203_320364

-- Define an odd function
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_sum_condition (f : ℝ → ℝ) (h : OddFunction f) :
  (∀ x₁ x₂ : ℝ, x₁ + x₂ = 0 → f x₁ + f x₂ = 0) ∧
  ¬(∀ x₁ x₂ : ℝ, f x₁ + f x₂ = 0 → x₁ + x₂ = 0) :=
by sorry

end odd_function_sum_condition_l3203_320364


namespace subcommittee_count_l3203_320354

/-- The number of members in the planning committee -/
def totalMembers : ℕ := 12

/-- The number of professors in the planning committee -/
def professorCount : ℕ := 5

/-- The size of the subcommittee -/
def subcommitteeSize : ℕ := 4

/-- The minimum number of professors required in the subcommittee -/
def minProfessors : ℕ := 2

/-- Calculates the number of valid subcommittees -/
def validSubcommittees : ℕ := sorry

theorem subcommittee_count :
  validSubcommittees = 285 := by sorry

end subcommittee_count_l3203_320354


namespace polynomial_degree_bound_l3203_320353

theorem polynomial_degree_bound (m n k : ℕ) (P : Polynomial ℤ) :
  m > 0 →
  n > 0 →
  k ≥ 2 →
  (∀ i, Odd (P.coeff i)) →
  P.degree = n →
  (X - 1 : Polynomial ℤ) ^ m ∣ P →
  m ≥ 2^k →
  n ≥ 2^(k+1) - 1 := by
  sorry

end polynomial_degree_bound_l3203_320353


namespace parabola_directrix_l3203_320324

/-- The directrix of the parabola y = -1/4 * x^2 is y = 1 -/
theorem parabola_directrix (x y : ℝ) : 
  y = -1/4 * x^2 → 
  (∀ (x₀ y₀ : ℝ), y₀ = -1/4 * x₀^2 → 
    (x₀ - x)^2 + (y₀ - y)^2 = (y₀ - 1)^2) → 
  y = 1 := by sorry

end parabola_directrix_l3203_320324


namespace max_books_is_eight_l3203_320329

/-- Represents the maximum number of books borrowed by a single student in a class with the given conditions. -/
def max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) (avg_books : ℕ) : ℕ :=
  let rest_students := total_students - (zero_books + one_book + two_books)
  let total_books := total_students * avg_books
  let known_books := one_book + 2 * two_books
  let rest_books := total_books - known_books
  let min_rest_books := (rest_students - 1) * 3
  rest_books - min_rest_books

/-- Theorem stating that under the given conditions, the maximum number of books borrowed by a single student is 8. -/
theorem max_books_is_eight :
  max_books_borrowed 20 2 8 3 2 = 8 := by
  sorry

end max_books_is_eight_l3203_320329


namespace sourball_candies_count_l3203_320334

/-- The number of sourball candies in the bucket initially -/
def initial_candies : ℕ := 30

/-- The number of candies Nellie can eat before crying -/
def nellie_candies : ℕ := 12

/-- The number of candies Jacob can eat before crying -/
def jacob_candies : ℕ := nellie_candies / 2

/-- The number of candies Lana can eat before crying -/
def lana_candies : ℕ := jacob_candies - 3

/-- The number of candies each person gets after division -/
def remaining_per_person : ℕ := 3

/-- The number of people -/
def num_people : ℕ := 3

theorem sourball_candies_count :
  initial_candies = nellie_candies + jacob_candies + lana_candies + remaining_per_person * num_people :=
by sorry

end sourball_candies_count_l3203_320334


namespace triangle_inequality_minimum_l3203_320366

theorem triangle_inequality_minimum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (htri : a + b ≥ c ∧ b + c ≥ a ∧ c + a ≥ b) :
  c / (a + b) + b / c ≥ Real.sqrt 2 - 1 / 2 :=
by sorry

end triangle_inequality_minimum_l3203_320366


namespace ornamental_rings_ratio_l3203_320383

theorem ornamental_rings_ratio (initial_purchase : ℕ) (mother_purchase : ℕ) (sold_after : ℕ) (remaining : ℕ) :
  initial_purchase = 200 →
  mother_purchase = 300 →
  sold_after = 150 →
  remaining = 225 →
  ∃ (original_stock : ℕ),
    initial_purchase + original_stock > 0 ∧
    (1 / 4 : ℚ) * (initial_purchase + original_stock : ℚ) + (mother_purchase : ℚ) - (sold_after : ℚ) = (remaining : ℚ) ∧
    (initial_purchase : ℚ) / (original_stock : ℚ) = 2 := by
  sorry

end ornamental_rings_ratio_l3203_320383


namespace unique_consecutive_set_l3203_320316

/-- Represents a set of consecutive positive integers -/
structure ConsecutiveSet where
  start : Nat
  length : Nat

/-- The sum of a set of consecutive positive integers -/
def sum_consecutive (s : ConsecutiveSet) : Nat :=
  s.length * (2 * s.start + s.length - 1) / 2

/-- A set is valid if it contains at least two integers and sums to 20 -/
def is_valid_set (s : ConsecutiveSet) : Prop :=
  s.length ≥ 2 ∧ sum_consecutive s = 20

theorem unique_consecutive_set : ∃! s : ConsecutiveSet, is_valid_set s :=
sorry

end unique_consecutive_set_l3203_320316


namespace count_congruent_integers_l3203_320376

theorem count_congruent_integers (n : ℕ) : 
  (Finset.filter (fun x => x > 0 ∧ x < 500 ∧ x % 14 = 9) (Finset.range 500)).card = 36 := by
  sorry

end count_congruent_integers_l3203_320376


namespace inequality_proof_l3203_320365

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2)/(x^2 + y*z) + (y^2 + 2*z^2 + 2*x^2)/(y^2 + z*x) + (z^2 + 2*x^2 + 2*y^2)/(z^2 + x*y) > 6 :=
by sorry

end inequality_proof_l3203_320365


namespace flippy_divisible_by_four_l3203_320340

/-- A four-digit number is flippy if its digits alternate between two distinct digits from the set {4, 6} -/
def is_flippy (n : ℕ) : Prop :=
  (n ≥ 1000 ∧ n < 10000) ∧
  (∃ a b : ℕ, (a = 4 ∨ a = 6) ∧ (b = 4 ∨ b = 6) ∧ a ≠ b ∧
   ((n = 1000 * a + 100 * b + 10 * a + b) ∨
    (n = 1000 * b + 100 * a + 10 * b + a)))

theorem flippy_divisible_by_four :
  ∃! n : ℕ, is_flippy n ∧ n % 4 = 0 :=
sorry

end flippy_divisible_by_four_l3203_320340


namespace emily_sixth_score_l3203_320362

def emily_scores : List ℕ := [91, 94, 88, 90, 101]
def target_mean : ℕ := 95
def num_quizzes : ℕ := 6

theorem emily_sixth_score :
  ∃ (sixth_score : ℕ),
    (emily_scores.sum + sixth_score) / num_quizzes = target_mean ∧
    sixth_score = 106 := by
  sorry

end emily_sixth_score_l3203_320362


namespace drain_time_for_specific_pumps_l3203_320327

/-- Represents the time taken to drain a lake with three pumps working together -/
def drain_time (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

/-- Theorem stating the time taken to drain a lake with three specific pumps -/
theorem drain_time_for_specific_pumps :
  drain_time (1/9) (1/6) (1/12) = 36/13 := by
  sorry

end drain_time_for_specific_pumps_l3203_320327


namespace parallelogram_area_l3203_320358

theorem parallelogram_area (side1 side2 : ℝ) (angle : ℝ) :
  side1 = 7 →
  side2 = 12 →
  angle = Real.pi / 3 →
  side2 * side1 * Real.sin angle = 12 * 7 * Real.sin (Real.pi / 3) :=
by sorry

end parallelogram_area_l3203_320358


namespace meaningful_expression_range_l3203_320369

theorem meaningful_expression_range (x : ℝ) :
  (∃ y : ℝ, y = (1 : ℝ) / Real.sqrt (x - 2)) ↔ x > 2 := by
  sorry

end meaningful_expression_range_l3203_320369


namespace mem_not_zeige_l3203_320367

-- Define our universes
variable (U : Type)

-- Define our sets
variable (Mem Enform Zeige : Set U)

-- State our premises
variable (h1 : Mem ⊆ Enform)
variable (h2 : Enform ∩ Zeige = ∅)

-- State our theorem
theorem mem_not_zeige :
  (∀ x, x ∈ Mem → x ∉ Zeige) ∧
  (Mem ∩ Zeige = ∅) :=
sorry

end mem_not_zeige_l3203_320367


namespace line_perp_plane_implies_planes_perp_l3203_320351

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the subset relation
variable (subset : Line → Plane → Prop)

-- Define the perpendicular relation
variable (perp : Line → Plane → Prop)
variable (perpPlanes : Plane → Plane → Prop)

-- State the theorem
theorem line_perp_plane_implies_planes_perp
  (l : Line) (α β : Plane)
  (h1 : subset l α)
  (h2 : perp l β) :
  perpPlanes α β :=
sorry

end line_perp_plane_implies_planes_perp_l3203_320351


namespace equation_solution_l3203_320361

theorem equation_solution :
  ∃ x : ℚ, (8 * x^2 + 150 * x + 2) / (3 * x + 50) = 4 * x + 2 ∧ x = -7/2 := by
  sorry

end equation_solution_l3203_320361


namespace continuous_piecewise_function_sum_l3203_320386

-- Define the piecewise function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x > 2 then a * x + 5
  else if x ≥ -2 ∧ x ≤ 2 then x - 7
  else 3 * x - b

-- State the theorem
theorem continuous_piecewise_function_sum (a b : ℝ) :
  Continuous (f a b) → a + b = -2 := by sorry

end continuous_piecewise_function_sum_l3203_320386


namespace orange_distribution_l3203_320345

/-- Given a number of oranges, calories per orange, and calories per person,
    calculate the number of people who can receive an equal share of the total calories. -/
def people_fed (num_oranges : ℕ) (calories_per_orange : ℕ) (calories_per_person : ℕ) : ℕ :=
  (num_oranges * calories_per_orange) / calories_per_person

/-- Prove that with 5 oranges, 80 calories per orange, and 100 calories per person,
    the number of people fed is 4. -/
theorem orange_distribution :
  people_fed 5 80 100 = 4 := by
  sorry

end orange_distribution_l3203_320345


namespace total_ways_eq_2501_l3203_320396

/-- The number of different types of cookies --/
def num_cookie_types : ℕ := 6

/-- The number of different types of milk --/
def num_milk_types : ℕ := 4

/-- The total number of item types --/
def total_item_types : ℕ := num_cookie_types + num_milk_types

/-- The number of items they purchase collectively --/
def total_items : ℕ := 4

/-- Represents a purchase combination for Charlie and Delta --/
structure Purchase where
  charlie_items : ℕ
  delta_items : ℕ
  charlie_items_le_total_types : charlie_items ≤ total_item_types
  delta_items_le_cookies : delta_items ≤ num_cookie_types
  sum_eq_total : charlie_items + delta_items = total_items

/-- The number of ways to choose items for a given purchase combination --/
def ways_to_choose (p : Purchase) : ℕ := sorry

/-- The total number of ways Charlie and Delta can purchase items --/
def total_ways : ℕ := sorry

/-- The main theorem: proving the total number of ways is 2501 --/
theorem total_ways_eq_2501 : total_ways = 2501 := sorry

end total_ways_eq_2501_l3203_320396


namespace complex_power_four_l3203_320393

theorem complex_power_four (i : ℂ) : i^2 = -1 → (1 - i)^4 = -4 := by sorry

end complex_power_four_l3203_320393


namespace senior_junior_ratio_l3203_320349

theorem senior_junior_ratio (S J : ℕ) (k : ℕ+) :
  S = k * J →
  (1 / 8 : ℚ) * S + (3 / 4 : ℚ) * J = (1 / 3 : ℚ) * (S + J) →
  k = 2 := by
  sorry

end senior_junior_ratio_l3203_320349


namespace trig_expression_equals_one_l3203_320372

theorem trig_expression_equals_one : 
  (Real.tan (30 * π / 180))^2 - (Real.sin (30 * π / 180))^2 = 
  (Real.tan (30 * π / 180))^2 * (Real.sin (30 * π / 180))^2 := by
  sorry

end trig_expression_equals_one_l3203_320372


namespace interest_rate_multiple_l3203_320359

theorem interest_rate_multiple (P r m : ℝ) 
  (h1 : P * r^2 = 40)
  (h2 : P * (m * r)^2 = 360) :
  m = 3 := by
  sorry

end interest_rate_multiple_l3203_320359


namespace shopping_expenditure_theorem_l3203_320397

theorem shopping_expenditure_theorem (x : ℝ) : 
  x ≥ 0 ∧ x ≤ 100 ∧
  x / 100 + 0.3 + 0.3 = 1 ∧
  0.04 * (x / 100) + 0.08 * 0.3 = 0.04 →
  x = 40 := by sorry

end shopping_expenditure_theorem_l3203_320397


namespace prob_A_misses_at_least_once_prob_A_hits_twice_B_hits_thrice_l3203_320379

-- Define the probabilities of hitting the target for A and B
def prob_A_hit : ℚ := 2/3
def prob_B_hit : ℚ := 3/4

-- Define the number of shots
def num_shots : ℕ := 4

-- Theorem for the first question
theorem prob_A_misses_at_least_once :
  1 - prob_A_hit ^ num_shots = 65/81 :=
sorry

-- Theorem for the second question
theorem prob_A_hits_twice_B_hits_thrice :
  (Nat.choose num_shots 2 : ℚ) * prob_A_hit^2 * (1 - prob_A_hit)^(num_shots - 2) *
  (Nat.choose num_shots 3 : ℚ) * prob_B_hit^3 * (1 - prob_B_hit)^(num_shots - 3) = 1/8 :=
sorry

end prob_A_misses_at_least_once_prob_A_hits_twice_B_hits_thrice_l3203_320379


namespace log_range_theorem_l3203_320390

-- Define the set of valid 'a' values
def validA : Set ℝ := {a | a ∈ (Set.Ioo 2 3) ∪ (Set.Ioo 3 5)}

-- Define the conditions for a meaningful logarithmic expression
def isValidLog (a : ℝ) : Prop :=
  a - 2 > 0 ∧ 5 - a > 0 ∧ a - 2 ≠ 1

-- Theorem statement
theorem log_range_theorem :
  ∀ a : ℝ, isValidLog a ↔ a ∈ validA :=
by sorry

end log_range_theorem_l3203_320390


namespace total_apples_is_sixteen_l3203_320356

/-- The number of apples picked by Mike -/
def mike_apples : ℕ := 7

/-- The number of apples picked by Nancy -/
def nancy_apples : ℕ := 3

/-- The number of apples picked by Keith -/
def keith_apples : ℕ := 6

/-- The total number of apples picked -/
def total_apples : ℕ := mike_apples + nancy_apples + keith_apples

theorem total_apples_is_sixteen : total_apples = 16 := by
  sorry

end total_apples_is_sixteen_l3203_320356


namespace area_DEF_value_l3203_320310

/-- Triangle ABC with sides 5, 12, and 13 -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (side_a : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 5)
  (side_b : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 12)
  (side_c : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 13)

/-- Parabola with focus F and directrix L -/
structure Parabola :=
  (F : ℝ × ℝ)
  (L : Set (ℝ × ℝ))

/-- Intersection points of parabolas with triangle sides -/
structure Intersections (t : Triangle) :=
  (A1 A2 B1 B2 C1 C2 : ℝ × ℝ)
  (on_parabola_A : Parabola → Prop)
  (on_parabola_B : Parabola → Prop)
  (on_parabola_C : Parabola → Prop)

/-- The area of triangle DEF formed by A1C2, B1A2, and C1B2 -/
def area_DEF (t : Triangle) (i : Intersections t) : ℝ := sorry

/-- Main theorem: The area of triangle DEF is 6728/3375 -/
theorem area_DEF_value (t : Triangle) (i : Intersections t) : 
  area_DEF t i = 6728 / 3375 := by sorry

end area_DEF_value_l3203_320310


namespace complementary_angles_theorem_l3203_320333

theorem complementary_angles_theorem (x : ℝ) : 
  (2 * x + 3 * x = 90) → x = 18 := by
  sorry

end complementary_angles_theorem_l3203_320333


namespace tennis_tournament_matches_l3203_320343

theorem tennis_tournament_matches (total_players : ℕ) (bye_players : ℕ) (first_round_players : ℕ) (first_round_matches : ℕ) :
  total_players = 128 →
  bye_players = 36 →
  first_round_players = 92 →
  first_round_matches = 46 →
  first_round_players = 2 * first_round_matches →
  total_players = bye_players + first_round_players →
  (∃ (total_matches : ℕ), total_matches = first_round_matches + (total_players - 1) ∧ total_matches = 127) :=
by sorry

end tennis_tournament_matches_l3203_320343


namespace dalton_needs_four_more_l3203_320380

def jump_rope_cost : ℕ := 7
def board_game_cost : ℕ := 12
def playground_ball_cost : ℕ := 4
def saved_allowance : ℕ := 6
def money_from_uncle : ℕ := 13

theorem dalton_needs_four_more :
  jump_rope_cost + board_game_cost + playground_ball_cost - (saved_allowance + money_from_uncle) = 4 := by
  sorry

end dalton_needs_four_more_l3203_320380


namespace solution_implies_a_value_l3203_320381

theorem solution_implies_a_value (a x y : ℝ) : 
  x = 2 → y = 1 → a * x - 3 * y = 1 → a = 2 := by
  sorry

end solution_implies_a_value_l3203_320381


namespace integral_inequality_l3203_320303

-- Define a non-decreasing function on [0,∞)
def NonDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y

-- State the theorem
theorem integral_inequality
  (f : ℝ → ℝ)
  (h_nondec : NonDecreasing f)
  {x y z : ℝ}
  (h_x : 0 ≤ x)
  (h_xy : x < y)
  (h_yz : y < z) :
  (z - x) * ∫ u in y..z, f u ≥ (z - y) * ∫ u in x..z, f u :=
sorry

end integral_inequality_l3203_320303


namespace trajectory_of_moving_circle_l3203_320342

structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def C1 : Circle := { center := (0, -1), radius := 5 }
def C2 : Circle := { center := (0, 2), radius := 1 }

def is_tangent_inside (M : ℝ × ℝ) (C : Circle) : Prop :=
  (M.1 - C.center.1)^2 + (M.2 - C.center.2)^2 = (C.radius - 1)^2

def is_tangent_outside (M : ℝ × ℝ) (C : Circle) : Prop :=
  (M.1 - C.center.1)^2 + (M.2 - C.center.2)^2 = (C.radius + 1)^2

theorem trajectory_of_moving_circle (x y : ℝ) :
  is_tangent_inside (x, y) C1 → is_tangent_outside (x, y) C2 →
  y ≠ 3 → y^2/9 + x^2/5 = 1 :=
by sorry

end trajectory_of_moving_circle_l3203_320342


namespace gcf_of_16_and_24_l3203_320352

theorem gcf_of_16_and_24 : Nat.gcd 16 24 = 8 :=
by
  have h1 : Nat.lcm 16 24 = 48 := by sorry
  sorry

end gcf_of_16_and_24_l3203_320352


namespace arithmetic_sequence_middle_term_l3203_320306

theorem arithmetic_sequence_middle_term (y : ℝ) :
  y > 0 ∧ 
  (∃ (d : ℝ), y^2 - 2^2 = d ∧ 5^2 - y^2 = d) →
  y = Real.sqrt 14.5 :=
by sorry

end arithmetic_sequence_middle_term_l3203_320306


namespace largest_divisor_of_five_consecutive_integers_l3203_320321

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k * 120 = (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) ∧
  ∀ m : ℤ, m > 120 → ∃ l : ℤ, l * m ≠ (l * (l + 1) * (l + 2) * (l + 3) * (l + 4)) :=
by sorry

end largest_divisor_of_five_consecutive_integers_l3203_320321


namespace extremum_point_implies_k_range_l3203_320375

noncomputable def f (x k : ℝ) : ℝ := (Real.exp x) / (x^2) - k * (2/x + Real.log x)

theorem extremum_point_implies_k_range :
  (∀ x : ℝ, x > 0 → (∀ y : ℝ, y > 0 → f x k = f y k → x = y ∨ x = 2)) →
  k ∈ Set.Iic (Real.exp 1) :=
sorry

end extremum_point_implies_k_range_l3203_320375


namespace pirate_treasure_l3203_320370

theorem pirate_treasure (m : ℕ) : 
  (m / 3 + 1) + (m / 4 + 5) + (m / 5 + 20) = m → m = 120 :=
by
  sorry

end pirate_treasure_l3203_320370


namespace octahedron_sum_l3203_320317

/-- Represents an octahedron -/
structure Octahedron where
  edges : ℕ
  vertices : ℕ
  faces : ℕ

/-- The sum of edges, vertices, and faces of an octahedron is 26 -/
theorem octahedron_sum : ∀ (o : Octahedron), o.edges + o.vertices + o.faces = 26 := by
  sorry

end octahedron_sum_l3203_320317


namespace friends_carrying_bananas_l3203_320330

theorem friends_carrying_bananas (total_friends : ℕ) (pears oranges apples : ℕ) : 
  total_friends = 35 →
  pears = 14 →
  oranges = 8 →
  apples = 5 →
  total_friends = pears + oranges + apples + (total_friends - (pears + oranges + apples)) →
  total_friends - (pears + oranges + apples) = 8 :=
by sorry

end friends_carrying_bananas_l3203_320330
