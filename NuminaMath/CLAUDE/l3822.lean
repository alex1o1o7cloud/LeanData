import Mathlib

namespace birthday_gift_cost_l3822_382225

def boss_contribution : ℕ := 15
def todd_contribution : ℕ := 2 * boss_contribution
def remaining_employees : ℕ := 5
def employee_contribution : ℕ := 11

theorem birthday_gift_cost :
  boss_contribution + todd_contribution + (remaining_employees * employee_contribution) = 100 := by
  sorry

end birthday_gift_cost_l3822_382225


namespace girls_to_boys_ratio_camp_cedar_ratio_l3822_382202

/-- Represents a summer camp with boys, girls, and counselors -/
structure SummerCamp where
  boys : ℕ
  girls : ℕ
  counselors : ℕ
  children_per_counselor : ℕ

/-- Camp Cedar with given conditions -/
def camp_cedar : SummerCamp :=
  { boys := 40,
    girls := 120,  -- This is derived, not given directly
    counselors := 20,
    children_per_counselor := 8 }

/-- The theorem stating the ratio of girls to boys in Camp Cedar -/
theorem girls_to_boys_ratio (c : SummerCamp) (h1 : c = camp_cedar) :
  c.girls / c.boys = 3 := by
  sorry

/-- The main theorem proving the ratio of girls to boys in Camp Cedar -/
theorem camp_cedar_ratio :
  (camp_cedar.girls : ℚ) / camp_cedar.boys = 3 := by
  sorry

end girls_to_boys_ratio_camp_cedar_ratio_l3822_382202


namespace linear_system_solution_l3822_382288

theorem linear_system_solution (a b : ℚ) 
  (eq1 : 2 * a + 5 * b = 47)
  (eq2 : 4 * a + 3 * b = 39) :
  2 * a + 2 * b = 164 / 7 := by
  sorry

end linear_system_solution_l3822_382288


namespace volunteers_2008_l3822_382296

/-- The expected number of volunteers after a given number of years, 
    given an initial number and annual increase rate. -/
def expected_volunteers (initial : ℕ) (rate : ℝ) (years : ℕ) : ℝ :=
  initial * (1 + rate) ^ years

/-- Theorem: Given 500 initial volunteers in 2005 and a 20% annual increase,
    the expected number of volunteers in 2008 is 864. -/
theorem volunteers_2008 : 
  ⌊expected_volunteers 500 0.2 3⌋ = 864 := by
  sorry

end volunteers_2008_l3822_382296


namespace sodium_chloride_dilution_l3822_382232

/-- Proves that adding 30 ounces of pure water to 50 ounces of a 40% sodium chloride solution
    results in a 25% concentration. -/
theorem sodium_chloride_dilution (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_water : ℝ) (final_concentration : ℝ) :
  initial_volume = 50 ∧
  initial_concentration = 0.4 ∧
  added_water = 30 ∧
  final_concentration = 0.25 →
  (initial_volume * initial_concentration) / (initial_volume + added_water) = final_concentration :=
by
  sorry

#check sodium_chloride_dilution

end sodium_chloride_dilution_l3822_382232


namespace probability_of_selecting_seven_l3822_382223

-- Define the fraction
def fraction : ℚ := 3 / 8

-- Define the decimal representation as a list of digits
def decimal_representation : List ℕ := [3, 7, 5]

-- Define the target digit
def target_digit : ℕ := 7

-- Theorem statement
theorem probability_of_selecting_seven :
  (decimal_representation.filter (· = target_digit)).length / decimal_representation.length = 1 / 3 := by
  sorry

end probability_of_selecting_seven_l3822_382223


namespace cubic_root_sum_reciprocal_squares_l3822_382282

theorem cubic_root_sum_reciprocal_squares : 
  ∀ (α β γ : ℝ), 
    (α^3 - 15*α^2 + 26*α - 8 = 0) → 
    (β^3 - 15*β^2 + 26*β - 8 = 0) → 
    (γ^3 - 15*γ^2 + 26*γ - 8 = 0) → 
    α ≠ β → β ≠ γ → γ ≠ α →
    1/α^2 + 1/β^2 + 1/γ^2 = 916/64 := by
  sorry

end cubic_root_sum_reciprocal_squares_l3822_382282


namespace min_value_expression_l3822_382219

theorem min_value_expression (x y z : ℝ) (hpos : x > 0 ∧ y > 0 ∧ z > 0) (hsum : x^2 + y^2 + z^2 = 1) :
  (z + 1)^2 / (2 * x * y * z) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_expression_l3822_382219


namespace parabola_tangent_intersection_l3822_382285

/-- Two points on a parabola with tangents intersecting at 45° -/
structure ParabolaPoints where
  a : ℝ
  b : ℝ

/-- The parabola y = 4x^2 -/
def parabola (x : ℝ) : ℝ := 4 * x^2

/-- Tangent slope at a point on the parabola -/
def tangentSlope (x : ℝ) : ℝ := 8 * x

/-- Condition for tangents intersecting at 45° -/
def tangentAngle45 (p : ParabolaPoints) : Prop :=
  |((tangentSlope p.a - tangentSlope p.b) / (1 + tangentSlope p.a * tangentSlope p.b))| = 1

/-- Y-coordinate of the intersection point of tangents -/
def intersectionY (p : ParabolaPoints) : ℝ := 4 * p.a * p.b

theorem parabola_tangent_intersection
  (p : ParabolaPoints)
  (h1 : parabola p.a = 4 * p.a^2)
  (h2 : parabola p.b = 4 * p.b^2)
  (h3 : tangentAngle45 p) :
  intersectionY p = -1/16 := by
  sorry

end parabola_tangent_intersection_l3822_382285


namespace distance_difference_after_three_hours_l3822_382224

/-- Represents a cyclist with a constant cycling rate -/
structure Cyclist where
  name : String
  rate : ℝ  -- cycling rate in miles per hour

/-- Calculates the distance traveled by a cyclist in a given time -/
def distanceTraveled (cyclist : Cyclist) (time : ℝ) : ℝ :=
  cyclist.rate * time

/-- Proves that the difference in distance traveled between Carlos and Diana after 3 hours is 15 miles -/
theorem distance_difference_after_three_hours 
  (carlos : Cyclist)
  (diana : Cyclist)
  (h1 : carlos.rate = 20)
  (h2 : diana.rate = 15)
  : distanceTraveled carlos 3 - distanceTraveled diana 3 = 15 := by
  sorry

#check distance_difference_after_three_hours

end distance_difference_after_three_hours_l3822_382224


namespace sum_multiple_of_three_l3822_382261

theorem sum_multiple_of_three (a b : ℤ) 
  (ha : ∃ k : ℤ, a = 3 * k) 
  (hb : ∃ m : ℤ, b = 9 * m) : 
  ∃ n : ℤ, a + b = 3 * n := by
sorry

end sum_multiple_of_three_l3822_382261


namespace dividend_calculation_l3822_382269

theorem dividend_calculation (quotient divisor k : ℕ) 
  (h1 : quotient = 4)
  (h2 : divisor = k)
  (h3 : k = 4) :
  quotient * divisor = 16 := by
  sorry

end dividend_calculation_l3822_382269


namespace expression_evaluation_l3822_382270

theorem expression_evaluation :
  (((3^0 : ℝ) - 1 + 4^2 - 3)^(-1 : ℝ)) * 4 = 4/13 := by sorry

end expression_evaluation_l3822_382270


namespace no_real_roots_iff_m_less_than_neg_one_l3822_382279

theorem no_real_roots_iff_m_less_than_neg_one (m : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - m ≠ 0) ↔ m < -1 := by
  sorry

end no_real_roots_iff_m_less_than_neg_one_l3822_382279


namespace sufficient_not_necessary_l3822_382277

theorem sufficient_not_necessary : 
  let A := {x : ℝ | 1 < x ∧ x < 2}
  let B := {x : ℝ | x < 2}
  (A ⊂ B) ∧ (B \ A).Nonempty := by sorry

end sufficient_not_necessary_l3822_382277


namespace total_distance_calculation_l3822_382266

/-- Calculates the total distance driven by Darius, Julia, and Thomas in miles and kilometers -/
def total_distance (darius_miles : ℝ) (julia_miles : ℝ) (thomas_miles : ℝ) (detour_miles : ℝ) (km_per_mile : ℝ) : ℝ × ℝ :=
  let darius_total := darius_miles * 2 + detour_miles
  let julia_total := julia_miles * 2 + detour_miles
  let thomas_total := thomas_miles * 2
  let total_miles := darius_total + julia_total + thomas_total
  let total_km := total_miles * km_per_mile
  (total_miles, total_km)

theorem total_distance_calculation :
  total_distance 679 998 1205 120 1.60934 = (6004, 9665.73616) := by
  sorry

end total_distance_calculation_l3822_382266


namespace circle_intersection_and_origin_l3822_382238

/-- Given line -/
def given_line (x y : ℝ) : Prop := 2 * x - y + 1 = 0

/-- Given circle -/
def given_circle (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 15 = 0

/-- New circle -/
def new_circle (x y : ℝ) : Prop := x^2 + y^2 + 28*x - 15*y = 0

theorem circle_intersection_and_origin :
  (∀ x y : ℝ, given_line x y ∧ given_circle x y → new_circle x y) ∧
  new_circle 0 0 :=
sorry

end circle_intersection_and_origin_l3822_382238


namespace f_properties_l3822_382253

-- Define the function f
def f (x : ℝ) : ℝ := -x - x^3

-- State the theorem
theorem f_properties (x₁ x₂ : ℝ) (h : x₁ + x₂ ≤ 0) :
  (f x₁ * f (-x₁) ≤ 0) ∧ (f x₁ + f x₂ ≥ f (-x₁) + f (-x₂)) := by
  sorry

end f_properties_l3822_382253


namespace sum_of_numbers_l3822_382255

theorem sum_of_numbers (a b c : ℝ) (ha : a = 0.8) (hb : b = 1/2) (hc : c = 0.9) :
  (if a ≥ 0.3 then a else 0) + (if b ≥ 0.3 then b else 0) + (if c ≥ 0.3 then c else 0) = 2.2 := by
  sorry

end sum_of_numbers_l3822_382255


namespace extremum_implies_a_equals_three_l3822_382227

/-- Given a function f(x) = (x^2 + a) / (x + 1), prove that if f has an extremum at x = 1, then a = 3. -/
theorem extremum_implies_a_equals_three (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (x^2 + a) / (x + 1)
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  a = 3 := by
sorry


end extremum_implies_a_equals_three_l3822_382227


namespace parallel_vectors_l3822_382278

def a (n : ℝ) : ℝ × ℝ := (n, -1)
def b : ℝ × ℝ := (-1, 1)
def c : ℝ × ℝ := (-1, 2)

theorem parallel_vectors (n : ℝ) : 
  (∃ k : ℝ, a n + b = k • c) → n = 1 := by
  sorry

end parallel_vectors_l3822_382278


namespace unique_solution_l3822_382201

theorem unique_solution (x y z : ℝ) 
  (hx : x > 4) (hy : y > 4) (hz : z > 4)
  (heq : (x + 3)^2 / (y + z - 3) + (y + 5)^2 / (z + x - 5) + (z + 7)^2 / (x + y - 7) = 45) :
  x = 11 ∧ y = 10 ∧ z = 9 := by
  sorry

end unique_solution_l3822_382201


namespace programmer_is_odd_one_out_l3822_382291

-- Define the set of professions
inductive Profession
| Dentist
| ElementarySchoolTeacher
| Programmer

-- Define a predicate for having special pension benefits
def has_special_pension_benefits (p : Profession) : Prop :=
  match p with
  | Profession.Dentist => true
  | Profession.ElementarySchoolTeacher => true
  | Profession.Programmer => false

-- Define the odd one out
def is_odd_one_out (p : Profession) : Prop :=
  ¬(has_special_pension_benefits p) ∧
  ∀ q : Profession, q ≠ p → has_special_pension_benefits q

-- Theorem statement
theorem programmer_is_odd_one_out :
  is_odd_one_out Profession.Programmer :=
sorry

end programmer_is_odd_one_out_l3822_382291


namespace sum_of_specific_series_l3822_382263

def arithmetic_series (a₁ : ℕ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |>.map (λ i => a₁ + i * d)

def alternating_sum (l : List ℤ) : ℤ :=
  l.enum.foldl (λ acc (i, x) => acc + (if i % 2 == 0 then x else -x)) 0

theorem sum_of_specific_series :
  let series := arithmetic_series 100 (-2) 50
  alternating_sum series = 50 := by
  sorry

end sum_of_specific_series_l3822_382263


namespace extremum_condition_l3822_382239

/-- A function f: ℝ → ℝ has an extremum at x₀ if f(x₀) is either a maximum or minimum value of f in some neighborhood of x₀ -/
def HasExtremumAt (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∃ ε > 0, ∀ x, |x - x₀| < ε → f x₀ ≤ f x ∨ f x ≤ f x₀

theorem extremum_condition (f : ℝ → ℝ) (x₀ : ℝ) (hf : Differentiable ℝ f) :
  (HasExtremumAt f x₀ → (deriv f) x₀ = 0) ∧
  ¬(((deriv f) x₀ = 0) → HasExtremumAt f x₀) :=
sorry

end extremum_condition_l3822_382239


namespace part_i_part_ii_l3822_382231

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 3| + |2*x - 4| - a

-- Theorem for Part I
theorem part_i :
  ∀ x : ℝ, f x 6 > 0 ↔ x < (1:ℝ)/3 ∨ x > (13:ℝ)/3 := by sorry

-- Theorem for Part II
theorem part_ii :
  ∀ a : ℝ, (∃ x : ℝ, f x a < 0) ↔ a > 1 := by sorry

end part_i_part_ii_l3822_382231


namespace building_volume_l3822_382207

/-- Represents the dimensions of a rectangular room -/
structure RoomDimensions where
  length : ℝ
  breadth : ℝ
  height : ℝ

/-- Calculates the volume of a rectangular room -/
def volume (d : RoomDimensions) : ℝ := d.length * d.breadth * d.height

/-- Calculates the surface area of the walls of a rectangular room -/
def wallArea (d : RoomDimensions) : ℝ := 2 * (d.length * d.height + d.breadth * d.height)

/-- Calculates the floor area of a rectangular room -/
def floorArea (d : RoomDimensions) : ℝ := d.length * d.breadth

theorem building_volume (firstFloor secondFloor : RoomDimensions)
  (h1 : firstFloor.length = 15)
  (h2 : firstFloor.breadth = 12)
  (h3 : secondFloor.length = 20)
  (h4 : secondFloor.breadth = 10)
  (h5 : secondFloor.height = firstFloor.height)
  (h6 : 2 * floorArea firstFloor = wallArea firstFloor) :
  volume firstFloor + volume secondFloor = 2534.6 := by
  sorry

end building_volume_l3822_382207


namespace distance_between_points_on_lines_l3822_382264

/-- The distance between two points on specific lines -/
theorem distance_between_points_on_lines (a c m k : ℝ) :
  let b := 2 * m * a + k
  let d := -m * c + k
  (((c - a)^2 + (d - b)^2) : ℝ).sqrt = ((1 + m^2 * (c + 2*a)^2) * (c - a)^2 : ℝ).sqrt :=
by sorry

end distance_between_points_on_lines_l3822_382264


namespace cosine_sum_simplification_l3822_382244

theorem cosine_sum_simplification :
  let x := Real.cos (2 * Real.pi / 17) + Real.cos (6 * Real.pi / 17) + Real.cos (8 * Real.pi / 17)
  x = (Real.sqrt 13 - 1) / 4 := by
  sorry

end cosine_sum_simplification_l3822_382244


namespace road_length_l3822_382216

theorem road_length (repaired : ℚ) (remaining_extra : ℚ) : 
  repaired = 7/15 → remaining_extra = 2/5 → repaired + (repaired + remaining_extra) = 4/3 := by
  sorry

end road_length_l3822_382216


namespace sqrt_sum_equality_l3822_382251

theorem sqrt_sum_equality : Real.sqrt (49 + 81) + Real.sqrt (36 + 49) = Real.sqrt 130 + Real.sqrt 85 := by
  sorry

end sqrt_sum_equality_l3822_382251


namespace max_sum_of_squares_l3822_382248

theorem max_sum_of_squares (p q r s : ℝ) : 
  p + q = 18 →
  p * q + r + s = 85 →
  p * r + q * s = 190 →
  r * s = 120 →
  p^2 + q^2 + r^2 + s^2 ≤ 886 :=
by sorry

end max_sum_of_squares_l3822_382248


namespace geometric_and_arithmetic_sequences_l3822_382226

/-- A geometric sequence with a₁ = 2 and a₄ = 16 -/
def geometric_sequence (n : ℕ) : ℝ :=
  2 * (2 : ℝ) ^ (n - 1)

/-- An arithmetic sequence with b₃ = a₃ and b₅ = a₅ -/
def arithmetic_sequence (n : ℕ) : ℝ :=
  -16 + 12 * (n - 1)

theorem geometric_and_arithmetic_sequences :
  (∀ n : ℕ, geometric_sequence n = 2^n) ∧
  (arithmetic_sequence 3 = geometric_sequence 3 ∧
   arithmetic_sequence 5 = geometric_sequence 5) ∧
  (∀ n : ℕ, arithmetic_sequence n = 12*n - 28) := by
  sorry

end geometric_and_arithmetic_sequences_l3822_382226


namespace sum_of_fourth_powers_l3822_382265

theorem sum_of_fourth_powers (a b c : ℝ) 
  (sum_zero : a + b + c = 0)
  (sum_squares : a^2 + b^2 + c^2 = 0.1) :
  a^4 + b^4 + c^4 = 0.005 := by
  sorry

end sum_of_fourth_powers_l3822_382265


namespace exponent_multiplication_l3822_382274

theorem exponent_multiplication (x : ℝ) : x^8 * x^2 = x^10 := by
  sorry

end exponent_multiplication_l3822_382274


namespace pretzel_price_is_two_l3822_382212

/-- Represents the revenue and quantity information for a candy store --/
structure CandyStore where
  fudgePounds : ℕ
  fudgePrice : ℚ
  trufflesDozens : ℕ
  trufflePrice : ℚ
  pretzelsDozens : ℕ
  totalRevenue : ℚ

/-- Calculates the price of each chocolate-covered pretzel --/
def pretzelPrice (store : CandyStore) : ℚ :=
  let fudgeRevenue := store.fudgePounds * store.fudgePrice
  let trufflesRevenue := store.trufflesDozens * 12 * store.trufflePrice
  let pretzelsRevenue := store.totalRevenue - fudgeRevenue - trufflesRevenue
  let pretzelsCount := store.pretzelsDozens * 12
  pretzelsRevenue / pretzelsCount

/-- Theorem stating that the price of each chocolate-covered pretzel is $2 --/
theorem pretzel_price_is_two (store : CandyStore)
  (h1 : store.fudgePounds = 20)
  (h2 : store.fudgePrice = 5/2)
  (h3 : store.trufflesDozens = 5)
  (h4 : store.trufflePrice = 3/2)
  (h5 : store.pretzelsDozens = 3)
  (h6 : store.totalRevenue = 212) :
  pretzelPrice store = 2 := by
  sorry


end pretzel_price_is_two_l3822_382212


namespace fast_food_order_cost_correct_l3822_382272

/-- Calculates the total cost of a fast food order with discount and tax --/
def fastFoodOrderCost (burgerPrice sandwichPrice smoothiePrice : ℚ)
                      (smoothieQuantity : ℕ)
                      (discountRate taxRate : ℚ)
                      (discountThreshold : ℚ)
                      (orderTime : ℕ) : ℚ :=
  let totalBeforeDiscount := burgerPrice + sandwichPrice + smoothiePrice * smoothieQuantity
  let discountedPrice := if totalBeforeDiscount > discountThreshold ∧ orderTime ≥ 1400 ∧ orderTime ≤ 1600
                         then totalBeforeDiscount * (1 - discountRate)
                         else totalBeforeDiscount
  let finalPrice := discountedPrice * (1 + taxRate)
  finalPrice

theorem fast_food_order_cost_correct :
  fastFoodOrderCost 5.75 4.50 4.25 2 0.20 0.12 15 1545 = 16.80 := by
  sorry

end fast_food_order_cost_correct_l3822_382272


namespace quadratic_solution_sum_l3822_382258

theorem quadratic_solution_sum (a b c : ℝ) : a ≠ 0 → (∀ x, a * x^2 + b * x + c = 0 ↔ x = 1) → a + b + c = 0 := by
  sorry

end quadratic_solution_sum_l3822_382258


namespace rotate_point_A_l3822_382246

/-- Rotates a point 90 degrees clockwise about the origin -/
def rotate90Clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.2, -p.1)

theorem rotate_point_A :
  let A : ℝ × ℝ := (-4, 1)
  rotate90Clockwise A = (1, 4) := by sorry

end rotate_point_A_l3822_382246


namespace right_triangle_hypotenuse_l3822_382257

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 9 → b = 12 → c^2 = a^2 + b^2 → c = 15 := by sorry

end right_triangle_hypotenuse_l3822_382257


namespace number_order_l3822_382221

/-- Represents a number in a given base -/
structure BaseNumber where
  digits : List Nat
  base : Nat

/-- Convert a BaseNumber to its decimal representation -/
def toDecimal (n : BaseNumber) : Nat :=
  n.digits.enum.foldl (fun acc (i, d) => acc + d * n.base ^ i) 0

/-- Define the given numbers -/
def a : BaseNumber := ⟨[14, 3], 16⟩
def b : BaseNumber := ⟨[0, 1, 2], 6⟩
def c : BaseNumber := ⟨[0, 0, 0, 1], 4⟩
def d : BaseNumber := ⟨[1, 1, 0, 1, 1, 1], 2⟩

/-- Theorem stating the order of the given numbers -/
theorem number_order :
  toDecimal b > toDecimal c ∧ toDecimal c > toDecimal a ∧ toDecimal a > toDecimal d := by
  sorry

end number_order_l3822_382221


namespace triangle_angle_c_l3822_382220

theorem triangle_angle_c (A B C : Real) :
  -- ABC is a triangle
  A + B + C = π →
  -- Given condition
  |Real.cos A - Real.sqrt 3 / 2| + (1 - Real.tan B)^2 = 0 →
  -- Conclusion
  C = π * 7 / 12 := by
sorry

end triangle_angle_c_l3822_382220


namespace problem_1_l3822_382240

theorem problem_1 (x : ℝ) : x^4 * x^3 * x - (x^4)^2 + (-2*x)^3 = -8*x^3 := by
  sorry

end problem_1_l3822_382240


namespace infinitely_many_winning_positions_l3822_382290

/-- The pebble game where players remove square numbers of pebbles -/
def PebbleGame (n : ℕ) : Prop :=
  ∀ (move : ℕ → ℕ), 
    (∀ k, ∃ m : ℕ, move k = m * m) → 
    (∀ k, move k ≤ n * n) →
    (n + 1 ≤ n * n + n + 1 - move (n * n + n + 1))

/-- There are infinitely many winning positions for the second player -/
theorem infinitely_many_winning_positions :
  ∀ n : ℕ, PebbleGame n :=
sorry

end infinitely_many_winning_positions_l3822_382290


namespace impossible_61_cents_l3822_382287

/-- Represents the types of coins available in the piggy bank -/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter

/-- Returns the value of a coin in cents -/
def coin_value (c : Coin) : Nat :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25

/-- Represents a combination of coins -/
def CoinCombination := List Coin

/-- Calculates the total value of a coin combination in cents -/
def total_value (comb : CoinCombination) : Nat :=
  comb.map coin_value |>.sum

/-- Theorem: It's impossible to make 61 cents with exactly 6 coins -/
theorem impossible_61_cents :
  ¬∃ (comb : CoinCombination), comb.length = 6 ∧ total_value comb = 61 := by
  sorry


end impossible_61_cents_l3822_382287


namespace f_max_min_on_interval_l3822_382222

noncomputable def f (x : ℝ) : ℝ := Real.exp x * (Real.sin x - 1)

theorem f_max_min_on_interval :
  let a : ℝ := -3 * Real.pi / 4
  let b : ℝ := 3 * Real.pi / 4
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc a b, f x ≤ max) ∧
    (∃ x ∈ Set.Icc a b, f x = max) ∧
    (∀ x ∈ Set.Icc a b, min ≤ f x) ∧
    (∃ x ∈ Set.Icc a b, f x = min) ∧
    max = 0 ∧
    min = -(Real.sqrt 2 / 2) * Real.exp (3 * Real.pi / 4) :=
by sorry

end f_max_min_on_interval_l3822_382222


namespace integral_3_minus_7x_squared_cos_2x_l3822_382286

theorem integral_3_minus_7x_squared_cos_2x (π : ℝ) :
  (∫ x in (0 : ℝ)..(2 * π), (3 - 7 * x^2) * Real.cos (2 * x)) = -7 * π := by
  sorry

end integral_3_minus_7x_squared_cos_2x_l3822_382286


namespace sqrt_equation_solution_l3822_382262

theorem sqrt_equation_solution (y : ℝ) : Real.sqrt (4 - 5*y) = 8 → y = -12 := by
  sorry

end sqrt_equation_solution_l3822_382262


namespace imaginary_part_of_z_l3822_382243

theorem imaginary_part_of_z (z : ℂ) : 
  z * (1 - Complex.I) = Complex.abs (1 - Complex.I) + Complex.I →
  z.im = (Real.sqrt 2 + 1) / 2 := by
sorry

end imaginary_part_of_z_l3822_382243


namespace median_determines_top_five_l3822_382260

/-- A list of 9 distinct real numbers representing scores -/
def Scores := List ℝ

/-- Predicate to check if a list has exactly 9 distinct elements -/
def has_nine_distinct (s : Scores) : Prop :=
  s.length = 9 ∧ s.Nodup

/-- The median of a list of 9 distinct real numbers -/
def median (s : Scores) : ℝ := sorry

/-- Predicate to check if a score is in the top 5 of a list of scores -/
def in_top_five (score : ℝ) (s : Scores) : Prop := sorry

theorem median_determines_top_five (s : Scores) (score : ℝ) 
  (h : has_nine_distinct s) :
  in_top_five score s ↔ score > median s := by sorry

end median_determines_top_five_l3822_382260


namespace total_flowers_in_two_weeks_l3822_382242

/-- Represents the flowers Miriam takes care of in a day -/
structure DailyFlowers where
  roses : ℕ
  tulips : ℕ
  daisies : ℕ
  lilies : ℕ
  sunflowers : ℕ

/-- Calculates the total number of flowers for a day -/
def totalFlowers (df : DailyFlowers) : ℕ :=
  df.roses + df.tulips + df.daisies + df.lilies + df.sunflowers

/-- Represents Miriam's work schedule for a week -/
structure WeekSchedule where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  saturday : ℕ

/-- Miriam's work hours in the first week -/
def firstWeekSchedule : WeekSchedule :=
  { monday := 4, tuesday := 5, wednesday := 3, thursday := 6, saturday := 5 }

/-- Flowers taken care of in the first week -/
def firstWeekFlowers : DailyFlowers :=
  { roses := 40, tulips := 50, daisies := 36, lilies := 48, sunflowers := 55 }

/-- Calculates the improved number of flowers with 20% increase -/
def improvePerformance (n : ℕ) : ℕ :=
  n + (n / 5)

/-- Theorem stating that the total number of flowers Miriam takes care of in two weeks is 504 -/
theorem total_flowers_in_two_weeks :
  let secondWeekFlowers : DailyFlowers :=
    { roses := improvePerformance firstWeekFlowers.roses,
      tulips := improvePerformance firstWeekFlowers.tulips,
      daisies := improvePerformance firstWeekFlowers.daisies,
      lilies := improvePerformance firstWeekFlowers.lilies,
      sunflowers := improvePerformance firstWeekFlowers.sunflowers }
  totalFlowers firstWeekFlowers + totalFlowers secondWeekFlowers = 504 := by
  sorry

end total_flowers_in_two_weeks_l3822_382242


namespace binomial_square_coefficient_l3822_382233

theorem binomial_square_coefficient (a : ℝ) : 
  (∃ r s : ℝ, ∀ x : ℝ, a * x^2 + 16 * x + 16 = (r * x + s)^2) → a = 4 := by
  sorry

end binomial_square_coefficient_l3822_382233


namespace z_in_first_quadrant_l3822_382292

/-- Given a complex number z satisfying (2+i)z = 1+3i, prove that z is in the first quadrant -/
theorem z_in_first_quadrant (z : ℂ) (h : (2 + Complex.I) * z = 1 + 3 * Complex.I) :
  0 < z.re ∧ 0 < z.im :=
by sorry

end z_in_first_quadrant_l3822_382292


namespace not_always_valid_proof_from_untrue_prop_l3822_382237

-- Define the concept of a valid proof
def ValidProof (premises : Prop) (conclusion : Prop) : Prop :=
  premises → conclusion

-- Define the concept of an untrue proposition
def UntrueProp (p : Prop) : Prop :=
  ¬p

-- Theorem stating that it's not generally true that a valid proof
-- can be constructed from an untrue proposition to reach a true conclusion
theorem not_always_valid_proof_from_untrue_prop :
  ¬∀ (p q : Prop), UntrueProp p → ValidProof p q → q :=
sorry

end not_always_valid_proof_from_untrue_prop_l3822_382237


namespace catch_time_correct_l3822_382267

/-- Represents the pursuit scenario between a smuggler and coast guard -/
structure Pursuit where
  initial_distance : ℝ
  initial_smuggler_speed : ℝ
  initial_coast_guard_speed : ℝ
  speed_change_time : ℝ
  new_speed_ratio : ℝ

/-- Calculates the time when the coast guard catches the smuggler -/
def catch_time (p : Pursuit) : ℝ :=
  sorry

/-- Theorem stating that the coast guard catches the smuggler after 6 hours and 36 minutes -/
theorem catch_time_correct (p : Pursuit) : 
  p.initial_distance = 15 ∧ 
  p.initial_smuggler_speed = 13 ∧ 
  p.initial_coast_guard_speed = 15 ∧
  p.speed_change_time = 3 ∧
  p.new_speed_ratio = 18/15 →
  catch_time p = 6 + 36/60 := by
  sorry

end catch_time_correct_l3822_382267


namespace quadratic_equation_d_has_two_distinct_roots_l3822_382211

/-- Discriminant of a quadratic equation ax^2 + bx + c = 0 -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Predicate for a quadratic equation having two distinct real roots -/
def has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ discriminant a b c > 0

theorem quadratic_equation_d_has_two_distinct_roots :
  has_two_distinct_real_roots 1 2 (-1) ∧
  ¬has_two_distinct_real_roots 1 0 4 ∧
  ¬has_two_distinct_real_roots 4 (-4) 1 ∧
  ¬has_two_distinct_real_roots 1 (-1) 3 :=
by sorry

end quadratic_equation_d_has_two_distinct_roots_l3822_382211


namespace linear_function_through_points_l3822_382234

/-- A linear function passing through two points -/
def linear_function (k b : ℝ) (x : ℝ) : ℝ := k * x + b

theorem linear_function_through_points :
  ∃ k b : ℝ, 
    (linear_function k b 3 = 5) ∧ 
    (linear_function k b (-4) = -9) ∧
    (∀ x : ℝ, linear_function k b x = 2 * x - 1) := by
  sorry

end linear_function_through_points_l3822_382234


namespace complex_power_eight_l3822_382228

theorem complex_power_eight :
  let z : ℂ := 3 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180))
  z^8 = -3280.5 - 3280.5 * Complex.I * Real.sqrt 3 := by
  sorry

end complex_power_eight_l3822_382228


namespace intersection_A_B_l3822_382208

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

-- Define set B
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x : ℝ | x = 2} := by sorry

end intersection_A_B_l3822_382208


namespace tom_fruit_purchase_amount_l3822_382294

/-- Represents a fruit purchase with quantity and rate --/
structure FruitPurchase where
  quantity : ℝ
  rate : ℝ

/-- Calculates the total cost of purchases before discount --/
def totalCost (purchases : List FruitPurchase) : ℝ :=
  purchases.foldl (fun acc p => acc + p.quantity * p.rate) 0

/-- Calculates the final amount after discount and tax --/
def finalAmount (purchases : List FruitPurchase) (discountRate : ℝ) (taxRate : ℝ) : ℝ :=
  let total := totalCost purchases
  let discountedPrice := total * (1 - discountRate)
  discountedPrice * (1 + taxRate)

theorem tom_fruit_purchase_amount :
  let purchases := [
    ⟨8, 70⟩,  -- Apples
    ⟨9, 55⟩,  -- Mangoes
    ⟨5, 40⟩,  -- Oranges
    ⟨12, 30⟩, -- Bananas
    ⟨7, 45⟩,  -- Grapes
    ⟨4, 80⟩   -- Cherries
  ]
  finalAmount purchases 0.1 0.05 = 2126.25 := by
  sorry


end tom_fruit_purchase_amount_l3822_382294


namespace arrangements_count_l3822_382241

/-- Represents the number of liberal arts classes -/
def liberal_arts_classes : ℕ := 2

/-- Represents the number of science classes -/
def science_classes : ℕ := 3

/-- Represents the total number of classes -/
def total_classes : ℕ := liberal_arts_classes + science_classes

/-- Function to calculate the number of arrangements -/
def arrangements : ℕ :=
  (science_classes.choose liberal_arts_classes) *
  (liberal_arts_classes.factorial) *
  (science_classes - liberal_arts_classes) *
  (liberal_arts_classes.factorial)

/-- Theorem stating that the number of arrangements is 24 -/
theorem arrangements_count : arrangements = 24 := by sorry

end arrangements_count_l3822_382241


namespace min_value_of_function_l3822_382235

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  ∃ y : ℝ, y = x + 4 / x ∧ y ≥ 4 ∧ (∀ z : ℝ, z = x + 4 / x → z ≥ y) :=
sorry

end min_value_of_function_l3822_382235


namespace dot_product_problem_l3822_382200

/-- Given vectors a and b in ℝ², prove that the dot product of (2a + b) and a is 6. -/
theorem dot_product_problem (a b : ℝ × ℝ) (h1 : a = (2, -1)) (h2 : b = (-1, 2)) :
  (2 • a + b) • a = 6 := by
  sorry

end dot_product_problem_l3822_382200


namespace equation_solution_l3822_382213

theorem equation_solution :
  ∃! y : ℚ, 7 * (2 * y - 3) + 4 = 3 * (5 - 9 * y) ∧ y = 32 / 41 := by
sorry

end equation_solution_l3822_382213


namespace total_peaches_l3822_382281

theorem total_peaches (red yellow green : ℕ) 
  (h1 : red = 7) 
  (h2 : yellow = 15) 
  (h3 : green = 8) : 
  red + yellow + green = 30 := by
  sorry

end total_peaches_l3822_382281


namespace triangle_area_with_median_l3822_382247

/-- The area of a triangle with two sides 1 and √15, and a median to the third side equal to 2, is √15/2. -/
theorem triangle_area_with_median (a b c : ℝ) (m : ℝ) (h1 : a = 1) (h2 : b = Real.sqrt 15) (h3 : m = 2) :
  (1/2 : ℝ) * a * b = (Real.sqrt 15) / 2 := by
  sorry

end triangle_area_with_median_l3822_382247


namespace square_of_binomial_constant_l3822_382289

/-- If 16x^2 + 32x + a is the square of a binomial, then a = 16 -/
theorem square_of_binomial_constant (a : ℝ) : 
  (∃ b c : ℝ, ∀ x, 16 * x^2 + 32 * x + a = (b * x + c)^2) → a = 16 := by
  sorry

end square_of_binomial_constant_l3822_382289


namespace xy_sum_greater_than_two_l3822_382295

theorem xy_sum_greater_than_two (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : x + y > 2 := by
  sorry

end xy_sum_greater_than_two_l3822_382295


namespace freshman_percentage_l3822_382256

theorem freshman_percentage (total_students : ℝ) (freshmen : ℝ) 
  (h1 : freshmen > 0) (h2 : total_students > 0) :
  let liberal_arts_fraction : ℝ := 0.6
  let psychology_fraction : ℝ := 0.5
  let freshmen_psych_liberal_fraction : ℝ := 0.24
  (liberal_arts_fraction * psychology_fraction * (freshmen / total_students) = 
    freshmen_psych_liberal_fraction) →
  freshmen / total_students = 0.8 := by
sorry

end freshman_percentage_l3822_382256


namespace nonnegative_real_inequality_l3822_382250

theorem nonnegative_real_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  x^2 * y^2 + x^2 * y + x * y^2 ≤ x^4 * y + x + y^4 := by
  sorry

end nonnegative_real_inequality_l3822_382250


namespace marching_band_total_weight_l3822_382298

def trumpet_weight : ℕ := 5
def clarinet_weight : ℕ := 5
def trombone_weight : ℕ := 10
def tuba_weight : ℕ := 20
def drum_weight : ℕ := 15

def trumpet_count : ℕ := 6
def clarinet_count : ℕ := 9
def trombone_count : ℕ := 8
def tuba_count : ℕ := 3
def drum_count : ℕ := 2

theorem marching_band_total_weight :
  trumpet_weight * trumpet_count +
  clarinet_weight * clarinet_count +
  trombone_weight * trombone_count +
  tuba_weight * tuba_count +
  drum_weight * drum_count = 245 := by
  sorry

end marching_band_total_weight_l3822_382298


namespace hillary_climbing_rate_l3822_382276

/-- Hillary's climbing rate in ft/hr -/
def hillary_rate : ℝ := 800

/-- Eddy's climbing rate in ft/hr -/
def eddy_rate : ℝ := hillary_rate - 500

/-- Distance from base camp to summit in ft -/
def summit_distance : ℝ := 5000

/-- Distance Hillary climbs before stopping in ft -/
def hillary_climb_distance : ℝ := summit_distance - 1000

/-- Hillary's descent rate in ft/hr -/
def hillary_descent_rate : ℝ := 1000

/-- Total time from departure to meeting in hours -/
def total_time : ℝ := 6

theorem hillary_climbing_rate :
  hillary_rate = 800 ∧
  eddy_rate = hillary_rate - 500 ∧
  summit_distance = 5000 ∧
  hillary_climb_distance = summit_distance - 1000 ∧
  hillary_descent_rate = 1000 ∧
  total_time = 6 →
  hillary_rate * (total_time - hillary_climb_distance / hillary_descent_rate) = hillary_climb_distance ∧
  eddy_rate * total_time = hillary_climb_distance - hillary_descent_rate * (total_time - hillary_climb_distance / hillary_descent_rate) :=
by sorry

end hillary_climbing_rate_l3822_382276


namespace union_of_A_and_B_l3822_382293

def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

theorem union_of_A_and_B : A ∪ B = {1, 2, 3} := by
  sorry

end union_of_A_and_B_l3822_382293


namespace min_common_roots_quadratic_trinomials_l3822_382245

theorem min_common_roots_quadratic_trinomials 
  (n : ℕ) 
  (f : Fin n → ℝ → ℝ) 
  (h1 : n = 1004)
  (h2 : ∀ i : Fin n, ∃ a b c : ℝ, ∀ x, f i x = x^2 + a*x + b)
  (h3 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ 2007 → ∃ i : Fin n, ∃ x : ℝ, f i x = 0 ∧ x = k)
  : (∀ i j : Fin n, i ≠ j → ∀ x : ℝ, f i x ≠ f j x) :=
sorry

end min_common_roots_quadratic_trinomials_l3822_382245


namespace g_has_no_zeros_l3822_382230

noncomputable section

open Real

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := x - a * log x - x + exp (x - 1)

-- State the theorem
theorem g_has_no_zeros (a : ℝ) (h : 0 ≤ a ∧ a ≤ exp 1) :
  ∀ x > 0, g a x ≠ 0 := by
  sorry

end

end g_has_no_zeros_l3822_382230


namespace exists_max_volume_l3822_382217

/-- A rectangular prism with specific diagonal lengths --/
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ
  h_space_diagonal : a^2 + b^2 + c^2 = 1
  h_face_diagonal : b^2 + c^2 = 2
  h_positive : a > 0 ∧ b > 0 ∧ c > 0

/-- The volume of a rectangular prism --/
def volume (prism : RectangularPrism) : ℝ :=
  prism.a * prism.b * prism.c

/-- There exists a value p that maximizes the volume of the rectangular prism --/
theorem exists_max_volume : 
  ∃ p : ℝ, p > 0 ∧ 
  ∃ prism : RectangularPrism, 
    prism.a = p ∧
    ∀ other : RectangularPrism, volume prism ≥ volume other := by
  sorry


end exists_max_volume_l3822_382217


namespace star_calculation_l3822_382210

-- Define the ★ operation
def star (a b : ℚ) : ℚ := (a + b) / (a - b)

-- State the theorem
theorem star_calculation : star (star 3 5) 8 = -1/3 := by sorry

end star_calculation_l3822_382210


namespace files_remaining_l3822_382283

theorem files_remaining (m v d : ℕ) (hm : m = 4) (hv : v = 21) (hd : d = 23) :
  (m + v) - d = 2 :=
by sorry

end files_remaining_l3822_382283


namespace total_notes_count_l3822_382203

/-- Proves that the total number of notes is 126 given the conditions -/
theorem total_notes_count (total_amount : ℕ) (note_50_count : ℕ) (note_50_value : ℕ) (note_500_value : ℕ) :
  total_amount = 10350 ∧
  note_50_count = 117 ∧
  note_50_value = 50 ∧
  note_500_value = 500 ∧
  total_amount = note_50_count * note_50_value + (total_amount - note_50_count * note_50_value) / note_500_value * note_500_value →
  note_50_count + (total_amount - note_50_count * note_50_value) / note_500_value = 126 :=
by sorry

end total_notes_count_l3822_382203


namespace vertical_complementary_implies_perpendicular_l3822_382214

/-- Two angles are vertical if they are opposite each other when two lines intersect. -/
def are_vertical_angles (α β : Real) : Prop := sorry

/-- Two angles are complementary if their sum is 90 degrees. -/
def are_complementary (α β : Real) : Prop := α + β = 90

/-- Two lines are perpendicular if they form a right angle (90 degrees) at their intersection. -/
def are_perpendicular_lines (l1 l2 : Line) : Prop := sorry

theorem vertical_complementary_implies_perpendicular (α β : Real) (l1 l2 : Line) :
  are_vertical_angles α β → are_complementary α β → are_perpendicular_lines l1 l2 := by
  sorry

end vertical_complementary_implies_perpendicular_l3822_382214


namespace complex_quadrant_l3822_382229

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the conditions
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

-- Define the equation
def equation (z a : ℂ) : Prop := (2 + i) * z = 1 + a * i^3

-- Theorem statement
theorem complex_quadrant (z a : ℂ) 
  (h1 : is_pure_imaginary z) 
  (h2 : equation z a) : 
  (a + z).re > 0 ∧ (a + z).im < 0 := by
  sorry

end complex_quadrant_l3822_382229


namespace angle_A_is_60_degrees_side_c_equation_l3822_382273

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The cosine law for triangle ABC --/
axiom cosine_law (t : Triangle) : t.c^2 = t.a^2 + t.b^2 - 2*t.a*t.b*Real.cos t.C

/-- The given condition c/2 = b - a cos(C) --/
def condition (t : Triangle) : Prop := t.c/2 = t.b - t.a * Real.cos t.C

theorem angle_A_is_60_degrees (t : Triangle) (h : condition t) : Real.cos t.A = 1/2 := by sorry

theorem side_c_equation (t : Triangle) (h : condition t) (ha : t.a = Real.sqrt 15) (hb : t.b = 4) :
  t.c^2 - 4*t.c + 1 = 0 := by sorry

end angle_A_is_60_degrees_side_c_equation_l3822_382273


namespace gcd_3_powers_l3822_382284

theorem gcd_3_powers : Nat.gcd (3^1001 - 1) (3^1012 - 1) = 177146 := by
  sorry

end gcd_3_powers_l3822_382284


namespace conic_section_eccentricity_l3822_382259

/-- A conic section curve with two foci -/
structure ConicSection where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- The eccentricity of a conic section -/
def eccentricity (C : ConicSection) : ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- Theorem: Given a conic section curve C with foci F₁ and F₂, and a point P on C
    such that |PF₁| : |F₁F₂| : |PF₂| = 4 : 3 : 2, the eccentricity of C is either 1/2 or 3/2 -/
theorem conic_section_eccentricity (C : ConicSection) (P : ℝ × ℝ) :
  distance P C.F₁ / distance C.F₁ C.F₂ = 4/3 ∧
  distance C.F₁ C.F₂ / distance P C.F₂ = 3/2 →
  eccentricity C = 1/2 ∨ eccentricity C = 3/2 := by
  sorry

end conic_section_eccentricity_l3822_382259


namespace imaginary_part_of_2_minus_i_l3822_382275

theorem imaginary_part_of_2_minus_i :
  Complex.im (2 - Complex.I) = -1 := by sorry

end imaginary_part_of_2_minus_i_l3822_382275


namespace smallest_part_of_proportional_division_l3822_382204

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (h_total : total = 120) (h_props : a = 3 ∧ b = 5 ∧ c = 7) :
  let x := total / (a + b + c)
  min (a * x) (min (b * x) (c * x)) = 24 := by
sorry

end smallest_part_of_proportional_division_l3822_382204


namespace fraction_of_larger_part_l3822_382206

theorem fraction_of_larger_part (total : ℝ) (larger : ℝ) (f : ℝ) : 
  total = 66 →
  larger = 50 →
  f * larger = 0.625 * (total - larger) + 10 →
  f = 0.4 := by
sorry

end fraction_of_larger_part_l3822_382206


namespace total_balloons_l3822_382280

theorem total_balloons (gold : ℕ) (silver : ℕ) (black : ℕ) : 
  gold = 141 → 
  silver = 2 * gold → 
  black = 150 → 
  gold + silver + black = 573 := by
sorry

end total_balloons_l3822_382280


namespace greatest_value_quadratic_inequality_l3822_382236

theorem greatest_value_quadratic_inequality :
  ∃ (x_max : ℝ), (∀ x : ℝ, -x^2 + 9*x - 18 ≥ 0 → x ≤ x_max) ∧ (-x_max^2 + 9*x_max - 18 ≥ 0) ∧ x_max = 6 := by
  sorry

end greatest_value_quadratic_inequality_l3822_382236


namespace frank_candy_total_l3822_382252

/-- Given that Frank put candy equally into 2 bags and there are 8 pieces of candy in each bag,
    prove that the total number of pieces of candy is 16. -/
theorem frank_candy_total (num_bags : ℕ) (pieces_per_bag : ℕ) 
    (h1 : num_bags = 2) 
    (h2 : pieces_per_bag = 8) : 
  num_bags * pieces_per_bag = 16 := by
  sorry

end frank_candy_total_l3822_382252


namespace negative_cube_squared_l3822_382249

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end negative_cube_squared_l3822_382249


namespace roots_in_unit_interval_l3822_382215

noncomputable def f (q : ℕ → ℝ) : ℕ → ℝ → ℝ
| 0, x => 1
| 1, x => x
| (n + 2), x => (1 + q n) * x * f q (n + 1) x - q n * f q n x

theorem roots_in_unit_interval (q : ℕ → ℝ) (h : ∀ n, q n > 0) :
  ∀ n : ℕ, ∀ x : ℝ, |x| > 1 → |f q (n + 1) x| > |f q n x| :=
sorry

end roots_in_unit_interval_l3822_382215


namespace order_of_expressions_l3822_382268

theorem order_of_expressions : 
  let a : ℝ := (1/2)^(3/2)
  let b : ℝ := Real.log π
  let c : ℝ := Real.log (3/2) / Real.log (1/2)
  c < a ∧ a < b := by sorry

end order_of_expressions_l3822_382268


namespace zero_of_f_l3822_382209

def f (x : ℝ) := 2 * x - 3

theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ∧ x = 3/2 :=
sorry

end zero_of_f_l3822_382209


namespace certain_number_value_l3822_382271

theorem certain_number_value : ∃ x : ℝ, (0.60 * 50 = 0.42 * x + 17.4) ∧ x = 30 := by
  sorry

end certain_number_value_l3822_382271


namespace solve_equation_l3822_382218

theorem solve_equation (x y : ℝ) : y = 2 / (5 * x + 2) → y = 2 → x = -1 / 5 := by
  sorry

end solve_equation_l3822_382218


namespace smallest_ratio_of_equation_l3822_382205

theorem smallest_ratio_of_equation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : 18 * x - 4 * x^2 + 2 * x^3 - 9 * y - 10 * x * y - x^2 * y + 6 * y^2 + 2 * x * y^2 - y^3 = 0) :
  ∃ (k : ℝ), k = y / x ∧ k ≥ 4/3 ∧ (∀ (k' : ℝ), k' = y / x → k' ≥ k) := by
  sorry

end smallest_ratio_of_equation_l3822_382205


namespace units_digit_of_F_F10_l3822_382254

-- Define the sequence F_n
def F : ℕ → ℕ
| 0 => 3
| 1 => 2
| (n + 2) => F (n + 1) + F n

-- Theorem statement
theorem units_digit_of_F_F10 : ∃ k : ℕ, F (F 10) = 10 * k + 1 := by
  sorry

end units_digit_of_F_F10_l3822_382254


namespace circle_equation_l3822_382299

/-- Given a circle with center (1,2) and a point (-2,6) on the circle,
    prove that its standard equation is (x-1)^2 + (y-2)^2 = 25 -/
theorem circle_equation (x y : ℝ) :
  let center := (1, 2)
  let point := (-2, 6)
  let on_circle := (point.1 - center.1)^2 + (point.2 - center.2)^2 = (x - center.1)^2 + (y - center.2)^2
  on_circle → (x - 1)^2 + (y - 2)^2 = 25 := by
  sorry

end circle_equation_l3822_382299


namespace fast_food_fries_sales_l3822_382297

theorem fast_food_fries_sales (S M L XL : ℕ) : 
  S + M + L + XL = 123 →
  Odd (S + M) →
  XL = 2 * M →
  L = S + M + 7 →
  S = 4 ∧ M = 27 ∧ L = 38 ∧ XL = 54 ∧ XL * 41 = 18 * (S + M + L + XL) :=
by sorry

end fast_food_fries_sales_l3822_382297
