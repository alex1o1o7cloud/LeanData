import Mathlib

namespace NUMINAMATH_CALUDE_percent_of_double_is_eighteen_l92_9260

theorem percent_of_double_is_eighteen (y : ℝ) (h1 : y > 0) (h2 : (y / 100) * (2 * y) = 18) : y = 30 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_double_is_eighteen_l92_9260


namespace NUMINAMATH_CALUDE_angle4_value_l92_9229

-- Define the angles
def angle1 : ℝ := 50
def angle2 : ℝ := 110
def angle3 : ℝ := 35
def angle4 : ℝ := 35
def angle5 : ℝ := 60
def angle6 : ℝ := 70

-- State the theorem
theorem angle4_value :
  angle1 + angle2 = 180 ∧
  angle3 = angle4 ∧
  angle1 = 50 ∧
  angle5 = 60 ∧
  angle1 + angle5 + angle6 = 180 ∧
  angle2 + angle6 = 180 ∧
  angle3 + angle4 = 180 - angle2 →
  angle4 = 35 := by sorry

end NUMINAMATH_CALUDE_angle4_value_l92_9229


namespace NUMINAMATH_CALUDE_sum_factorials_mod_30_l92_9232

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

theorem sum_factorials_mod_30 :
  sum_factorials 10 % 30 = 3 := by sorry

end NUMINAMATH_CALUDE_sum_factorials_mod_30_l92_9232


namespace NUMINAMATH_CALUDE_min_sum_exponents_520_l92_9220

/-- Given a natural number n, returns the sum of exponents in its binary representation -/
def sumOfExponents (n : ℕ) : ℕ := sorry

/-- Expresses a natural number as a sum of distinct powers of 2 -/
def expressAsPowersOf2 (n : ℕ) : List ℕ := sorry

theorem min_sum_exponents_520 :
  let powers := expressAsPowersOf2 520
  powers.length ≥ 2 ∧ sumOfExponents 520 = 12 :=
sorry

end NUMINAMATH_CALUDE_min_sum_exponents_520_l92_9220


namespace NUMINAMATH_CALUDE_kenneth_remaining_money_l92_9269

def remaining_money (initial_amount baguette_cost water_cost baguette_count water_count : ℕ) : ℕ :=
  initial_amount - (baguette_cost * baguette_count + water_cost * water_count)

theorem kenneth_remaining_money :
  remaining_money 50 2 1 2 2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_kenneth_remaining_money_l92_9269


namespace NUMINAMATH_CALUDE_downstream_speed_l92_9264

theorem downstream_speed (upstream_speed : ℝ) (average_speed : ℝ) (downstream_speed : ℝ) :
  upstream_speed = 6 →
  average_speed = 60 / 11 →
  (1 / upstream_speed + 1 / downstream_speed) / 2 = 1 / average_speed →
  downstream_speed = 5 := by
sorry

end NUMINAMATH_CALUDE_downstream_speed_l92_9264


namespace NUMINAMATH_CALUDE_rationalize_denominator_l92_9211

theorem rationalize_denominator : 35 / Real.sqrt 35 = Real.sqrt 35 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l92_9211


namespace NUMINAMATH_CALUDE_jana_height_l92_9288

/-- Given the heights of Jana, Kelly, and Jess, prove Jana's height -/
theorem jana_height (jana_height kelly_height jess_height : ℕ) 
  (h1 : jana_height = kelly_height + 5)
  (h2 : kelly_height = jess_height - 3)
  (h3 : jess_height = 72) : 
  jana_height = 74 := by
  sorry

end NUMINAMATH_CALUDE_jana_height_l92_9288


namespace NUMINAMATH_CALUDE_equal_sums_exist_l92_9261

/-- A 3x3 table with entries of 1, 0, or -1 -/
def Table := Fin 3 → Fin 3 → Int

/-- Predicate to check if a table is valid (contains only 1, 0, or -1) -/
def isValidTable (t : Table) : Prop :=
  ∀ i j, t i j = 1 ∨ t i j = 0 ∨ t i j = -1

/-- Sum of a row in the table -/
def rowSum (t : Table) (i : Fin 3) : Int :=
  (t i 0) + (t i 1) + (t i 2)

/-- Sum of a column in the table -/
def colSum (t : Table) (j : Fin 3) : Int :=
  (t 0 j) + (t 1 j) + (t 2 j)

/-- List of all row and column sums -/
def allSums (t : Table) : List Int :=
  (List.range 3).map (rowSum t) ++ (List.range 3).map (colSum t)

/-- Theorem: In a valid 3x3 table, there exist at least two equal sums among row and column sums -/
theorem equal_sums_exist (t : Table) (h : isValidTable t) :
  ∃ (x y : Fin 6), x ≠ y ∧ (allSums t).get x = (allSums t).get y :=
sorry

end NUMINAMATH_CALUDE_equal_sums_exist_l92_9261


namespace NUMINAMATH_CALUDE_dissimilar_terms_eq_distribution_ways_l92_9237

/-- The number of dissimilar terms in the expansion of (a + b + c + d)^7 -/
def dissimilar_terms : ℕ := sorry

/-- The number of ways to distribute 7 indistinguishable objects into 4 distinguishable boxes -/
def distribution_ways : ℕ := sorry

/-- Theorem stating that the number of dissimilar terms in (a + b + c + d)^7 
    is equal to the number of ways to distribute 7 objects into 4 boxes -/
theorem dissimilar_terms_eq_distribution_ways : 
  dissimilar_terms = distribution_ways := by sorry

end NUMINAMATH_CALUDE_dissimilar_terms_eq_distribution_ways_l92_9237


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l92_9247

/-- A geometric sequence with common ratio q satisfying certain conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  is_geometric : ∀ n, a (n + 1) = a n * q
  condition1 : a 5 - a 1 = 15
  condition2 : a 4 - a 2 = 6

/-- The common ratio of a geometric sequence satisfying the given conditions is either 1/2 or 2 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) : 
  seq.q = 1/2 ∨ seq.q = 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l92_9247


namespace NUMINAMATH_CALUDE_gas_refill_amount_l92_9273

def gas_problem (initial_gas tank_capacity gas_to_store gas_to_doctor : ℕ) : ℕ :=
  tank_capacity - (initial_gas - gas_to_store - gas_to_doctor)

theorem gas_refill_amount :
  gas_problem 10 12 6 2 = 10 := by sorry

end NUMINAMATH_CALUDE_gas_refill_amount_l92_9273


namespace NUMINAMATH_CALUDE_debt_payment_average_l92_9267

theorem debt_payment_average (total_payments : ℕ) (first_payment_amount : ℕ) 
  (first_payment_count : ℕ) (payment_increase : ℕ) :
  total_payments = 52 →
  first_payment_count = 12 →
  first_payment_amount = 410 →
  payment_increase = 65 →
  (first_payment_count * first_payment_amount + 
   (total_payments - first_payment_count) * (first_payment_amount + payment_increase)) / 
   total_payments = 460 :=
by sorry

end NUMINAMATH_CALUDE_debt_payment_average_l92_9267


namespace NUMINAMATH_CALUDE_no_valid_sequences_for_420_l92_9214

/-- Represents a sequence of consecutive positive integers -/
structure ConsecutiveSequence where
  start : ℕ
  length : ℕ
  h_length : length ≥ 2

/-- The sum of a consecutive sequence -/
def sum_consecutive_sequence (seq : ConsecutiveSequence) : ℕ :=
  seq.length * seq.start + seq.length * (seq.length - 1) / 2

/-- Predicate for a natural number being a perfect square -/
def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

/-- Theorem stating that there are no valid sequences summing to 420 -/
theorem no_valid_sequences_for_420 :
  ¬∃ (seq : ConsecutiveSequence), 
    sum_consecutive_sequence seq = 420 ∧ 
    is_perfect_square seq.start :=
sorry

end NUMINAMATH_CALUDE_no_valid_sequences_for_420_l92_9214


namespace NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l92_9234

/-- The volume of a cylinder formed by rotating a rectangle about its longer side. -/
theorem cylinder_volume_from_rectangle (width length : ℝ) (h_width : width = 8) (h_length : length = 20) :
  let radius : ℝ := width / 2
  let height : ℝ := length
  let volume : ℝ := π * radius^2 * height
  volume = 320 * π := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_from_rectangle_l92_9234


namespace NUMINAMATH_CALUDE_friends_bill_calculation_l92_9253

/-- Represents a restaurant order --/
structure Order where
  tacos : ℕ
  enchiladas : ℕ

/-- Represents the cost of an order --/
def cost (o : Order) (taco_price enchilada_price : ℚ) : ℚ :=
  o.tacos * taco_price + o.enchiladas * enchilada_price

theorem friends_bill_calculation (enchilada_price : ℚ) 
  (your_order friend_order : Order) (your_bill : ℚ) 
  (h1 : enchilada_price = 2)
  (h2 : your_order = ⟨2, 3⟩)
  (h3 : friend_order = ⟨3, 5⟩)
  (h4 : your_bill = 39/5) : 
  ∃ (taco_price : ℚ), cost friend_order taco_price enchilada_price = 127/10 := by
  sorry

#eval 127/10  -- Should output 12.7

end NUMINAMATH_CALUDE_friends_bill_calculation_l92_9253


namespace NUMINAMATH_CALUDE_emerald_count_l92_9289

/-- Represents the count of gemstones in a box -/
def GemCount := Nat

/-- Represents a box of gemstones -/
structure Box where
  count : GemCount

/-- Represents the collection of all boxes -/
structure JewelryBox where
  boxes : List Box
  diamond_boxes : List Box
  ruby_boxes : List Box
  emerald_boxes : List Box

/-- The total count of gemstones in a list of boxes -/
def total_gems (boxes : List Box) : Nat :=
  boxes.map (λ b => b.count) |>.sum

theorem emerald_count (jb : JewelryBox) 
  (h1 : jb.boxes.length = 6)
  (h2 : jb.diamond_boxes.length = 2)
  (h3 : jb.ruby_boxes.length = 2)
  (h4 : jb.emerald_boxes.length = 2)
  (h5 : jb.boxes = jb.diamond_boxes ++ jb.ruby_boxes ++ jb.emerald_boxes)
  (h6 : total_gems jb.ruby_boxes = total_gems jb.diamond_boxes + 15)
  (h7 : total_gems jb.boxes = 39) :
  total_gems jb.emerald_boxes = 12 := by
  sorry

end NUMINAMATH_CALUDE_emerald_count_l92_9289


namespace NUMINAMATH_CALUDE_rhombus_perimeter_l92_9205

/-- The perimeter of a rhombus with diagonals of 12 inches and 30 inches is 4√261 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 30) :
  4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = 4 * Real.sqrt 261 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_perimeter_l92_9205


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l92_9240

/-- Given a triangle ABC with side lengths a, b, and c satisfying (a+b+c)(b+c-a) = bc,
    prove that the measure of angle A is 120 degrees. -/
theorem triangle_angle_measure (a b c : ℝ) (h : (a + b + c) * (b + c - a) = b * c) :
  let A := Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c))
  A = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l92_9240


namespace NUMINAMATH_CALUDE_smallest_divisor_divisor_is_four_l92_9270

theorem smallest_divisor (d : ℕ) : d > 0 ∧ d > 3 ∧
  (∃ n : ℤ, n % d = 1 ∧ (3 * n) % d = 3) →
  d ≥ 4 :=
sorry

theorem divisor_is_four : ∃ d : ℕ, d > 0 ∧ d > 3 ∧
  (∃ n : ℤ, n % d = 1 ∧ (3 * n) % d = 3) ∧
  ∀ k : ℕ, k > 0 ∧ k > 3 ∧ (∃ m : ℤ, m % k = 1 ∧ (3 * m) % k = 3) →
  k ≥ d :=
sorry

end NUMINAMATH_CALUDE_smallest_divisor_divisor_is_four_l92_9270


namespace NUMINAMATH_CALUDE_proposition_false_implies_a_equals_one_l92_9250

theorem proposition_false_implies_a_equals_one (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + (1 - a) * x < 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_proposition_false_implies_a_equals_one_l92_9250


namespace NUMINAMATH_CALUDE_solutions_of_equation_l92_9292

theorem solutions_of_equation (x : ℝ) : 
  (3 * x^2 = Real.sqrt 3 * x) ↔ (x = 0 ∨ x = Real.sqrt 3 / 3) := by
sorry

end NUMINAMATH_CALUDE_solutions_of_equation_l92_9292


namespace NUMINAMATH_CALUDE_spider_count_l92_9279

/-- The number of webs made by some spiders in 5 days -/
def total_webs : ℕ := 5

/-- The number of days it takes for some spiders to make 5 webs -/
def days_for_total_webs : ℕ := 5

/-- The number of days it takes for 1 spider to make 1 web -/
def days_for_one_web : ℕ := 10

/-- The number of spiders -/
def num_spiders : ℕ := total_webs * days_for_one_web / days_for_total_webs

theorem spider_count : num_spiders = 10 := by
  sorry

end NUMINAMATH_CALUDE_spider_count_l92_9279


namespace NUMINAMATH_CALUDE_difference_quotient_of_f_l92_9227

-- Define the function f
def f (x : ℝ) : ℝ := 2 * x^2 - 1

-- State the theorem
theorem difference_quotient_of_f (Δx : ℝ) :
  let y1 := f 1
  let y2 := f (1 + Δx)
  (y2 - y1) / Δx = 4 + 2 * Δx := by
  sorry

end NUMINAMATH_CALUDE_difference_quotient_of_f_l92_9227


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_circles_l92_9200

-- Define the circles
def C₁ (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 7 = 0
def C₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*y - 27 = 0

-- Define the line
def L (x y : ℝ) : Prop := x + y - 3 = 0

-- Theorem statement
theorem perpendicular_bisector_of_intersecting_circles :
  ∃ (A B : ℝ × ℝ), 
    C₁ A.1 A.2 ∧ C₁ B.1 B.2 ∧ 
    C₂ A.1 A.2 ∧ C₂ B.1 B.2 ∧ 
    A ≠ B ∧
    L ((A.1 + B.1) / 2) ((A.2 + B.2) / 2) ∧
    (B.2 - A.2) * ((A.1 + B.1) / 2 - A.1) = (A.1 - B.1) * ((A.2 + B.2) / 2 - A.2) :=
by sorry


end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersecting_circles_l92_9200


namespace NUMINAMATH_CALUDE_cookies_per_bag_l92_9224

theorem cookies_per_bag (total_cookies : ℕ) (num_bags : ℕ) (h1 : total_cookies = 2173) (h2 : num_bags = 53) :
  total_cookies / num_bags = 41 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_bag_l92_9224


namespace NUMINAMATH_CALUDE_prob_four_ones_l92_9296

/-- The number of sides on a standard die -/
def die_sides : ℕ := 6

/-- The probability of rolling a specific number on a standard die -/
def prob_single_roll : ℚ := 1 / die_sides

/-- The number of dice rolled -/
def num_dice : ℕ := 4

/-- Theorem: The probability of rolling four 1s on four standard dice is 1/1296 -/
theorem prob_four_ones (die_sides : ℕ) (prob_single_roll : ℚ) (num_dice : ℕ) :
  die_sides = 6 →
  prob_single_roll = 1 / die_sides →
  num_dice = 4 →
  prob_single_roll ^ num_dice = 1 / 1296 := by
  sorry

#check prob_four_ones

end NUMINAMATH_CALUDE_prob_four_ones_l92_9296


namespace NUMINAMATH_CALUDE_maddie_tshirt_cost_l92_9248

-- Define the number of packs of white and blue T-shirts
def white_packs : ℕ := 2
def blue_packs : ℕ := 4

-- Define the number of T-shirts per pack for white and blue
def white_per_pack : ℕ := 5
def blue_per_pack : ℕ := 3

-- Define the cost per T-shirt
def cost_per_shirt : ℕ := 3

-- Define the total number of T-shirts
def total_shirts : ℕ := white_packs * white_per_pack + blue_packs * blue_per_pack

-- Define the total cost
def total_cost : ℕ := total_shirts * cost_per_shirt

-- Theorem to prove
theorem maddie_tshirt_cost : total_cost = 66 := by
  sorry

end NUMINAMATH_CALUDE_maddie_tshirt_cost_l92_9248


namespace NUMINAMATH_CALUDE_function_growth_l92_9215

open Real

theorem function_growth (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_growth : ∀ x, (deriv f) x > f x ∧ f x > 0) : 
  f 8 > 2022 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_growth_l92_9215


namespace NUMINAMATH_CALUDE_total_tax_percentage_l92_9221

/-- Calculate the total tax percentage given spending percentages and tax rates -/
theorem total_tax_percentage
  (clothing_percent : ℝ)
  (food_percent : ℝ)
  (other_percent : ℝ)
  (clothing_tax_rate : ℝ)
  (food_tax_rate : ℝ)
  (other_tax_rate : ℝ)
  (h1 : clothing_percent = 0.60)
  (h2 : food_percent = 0.10)
  (h3 : other_percent = 0.30)
  (h4 : clothing_percent + food_percent + other_percent = 1)
  (h5 : clothing_tax_rate = 0.04)
  (h6 : food_tax_rate = 0)
  (h7 : other_tax_rate = 0.08) :
  clothing_percent * clothing_tax_rate +
  food_percent * food_tax_rate +
  other_percent * other_tax_rate = 0.048 := by
sorry

end NUMINAMATH_CALUDE_total_tax_percentage_l92_9221


namespace NUMINAMATH_CALUDE_angle_in_quadrant_four_l92_9274

/-- If cos(π - α) < 0 and tan(α) < 0, then α is in Quadrant IV -/
theorem angle_in_quadrant_four (α : Real) 
  (h1 : Real.cos (Real.pi - α) < 0) 
  (h2 : Real.tan α < 0) : 
  0 < α ∧ α < Real.pi/2 ∧ Real.sin α < 0 ∧ Real.cos α > 0 := by
  sorry

end NUMINAMATH_CALUDE_angle_in_quadrant_four_l92_9274


namespace NUMINAMATH_CALUDE_sum_zero_inequality_l92_9203

theorem sum_zero_inequality (a b c d : ℝ) (h : a + b + c + d = 0) :
  (a*b + a*c + a*d + b*c + b*d + c*d)^2 + 12 ≥ 6*(a*b*c + a*b*d + a*c*d + b*c*d) := by
  sorry

end NUMINAMATH_CALUDE_sum_zero_inequality_l92_9203


namespace NUMINAMATH_CALUDE_impossible_number_composition_l92_9286

def is_base_five_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 45

def compose_number (base_numbers : List ℕ) : ℕ := sorry

theorem impossible_number_composition :
  ¬ ∃ (x : ℕ) (base_numbers : List ℕ) (p q : ℕ),
    (base_numbers.length = 2021) ∧
    (∀ n ∈ base_numbers, is_base_five_two_digit n) ∧
    (∀ i, i < 2021 → i % 2 = 0 →
      base_numbers.get! i = base_numbers.get! (i + 1) - 1) ∧
    (x = compose_number base_numbers) ∧
    (Nat.Prime p ∧ Nat.Prime q) ∧
    (p * q = x) ∧
    (q = p + 2) :=
  sorry

end NUMINAMATH_CALUDE_impossible_number_composition_l92_9286


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l92_9283

noncomputable def g (x : ℝ) (A B C : ℤ) : ℝ := x^2 / (A * x^2 + B * x + C)

theorem sum_of_coefficients 
  (A B C : ℤ) 
  (h1 : ∀ x > 5, g x A B C > 0.5)
  (h2 : (A : ℝ) * (-3)^2 + B * (-3) + C = 0)
  (h3 : (A : ℝ) * 4^2 + B * 4 + C = 0) :
  A + B + C = -24 := by sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l92_9283


namespace NUMINAMATH_CALUDE_area_difference_l92_9298

/-- Represents the construction of squares and triangles as described in the problem -/
structure Construction where
  square_side : ℝ
  triangle_side : ℝ
  first_level_triangles : ℕ
  second_level_triangles : ℕ

/-- The region R covered by the square and all triangles -/
def R (c : Construction) : ℝ := sorry

/-- The smallest convex polygon S encompassing R -/
def S (c : Construction) : ℝ := sorry

/-- The theorem to be proved -/
theorem area_difference (c : Construction) 
  (h1 : c.square_side = 2)
  (h2 : c.triangle_side = 2)
  (h3 : c.first_level_triangles = 8)
  (h4 : c.second_level_triangles = 16)
  (h5 : R c < S c) :
  S c - R c = 24 * Real.sqrt 3 - 4 := by sorry

end NUMINAMATH_CALUDE_area_difference_l92_9298


namespace NUMINAMATH_CALUDE_dave_new_cards_l92_9285

/-- Calculates the number of new baseball cards given the total pages used,
    cards per page, and number of old cards. -/
def new_cards (pages : ℕ) (cards_per_page : ℕ) (old_cards : ℕ) : ℕ :=
  pages * cards_per_page - old_cards

theorem dave_new_cards :
  new_cards 2 8 13 = 3 := by
  sorry

end NUMINAMATH_CALUDE_dave_new_cards_l92_9285


namespace NUMINAMATH_CALUDE_multiply_by_twenty_l92_9287

theorem multiply_by_twenty (x : ℝ) (h : 10 * x = 40) : 20 * x = 80 := by
  sorry

end NUMINAMATH_CALUDE_multiply_by_twenty_l92_9287


namespace NUMINAMATH_CALUDE_horner_method_v2_l92_9236

def f (x : ℝ) : ℝ := x^6 - 8*x^5 + 60*x^4 + 16*x^3 + 96*x^2 + 240*x + 64

def horner_v2 (a : ℝ) : ℝ :=
  let v0 := 1
  let v1 := v0 * a - 8
  v1 * a + 60

theorem horner_method_v2 :
  horner_v2 2 = 48 :=
by sorry

end NUMINAMATH_CALUDE_horner_method_v2_l92_9236


namespace NUMINAMATH_CALUDE_bryden_receive_amount_l92_9277

/-- The face value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The number of quarters Bryden has -/
def bryden_quarters : ℕ := 7

/-- The percentage the collector offers for state quarters -/
def collector_offer_percent : ℕ := 2500

/-- Calculate the amount Bryden will receive from the collector -/
def bryden_receive : ℚ :=
  (quarter_value * bryden_quarters) * (collector_offer_percent / 100)

/-- Theorem stating that Bryden will receive $43.75 from the collector -/
theorem bryden_receive_amount :
  bryden_receive = 43.75 := by sorry

end NUMINAMATH_CALUDE_bryden_receive_amount_l92_9277


namespace NUMINAMATH_CALUDE_parallel_perpendicular_to_plane_l92_9246

/-- Two lines are parallel -/
def parallel (a b : Line3) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perpendicular_to_plane (l : Line3) (p : Plane3) : Prop := sorry

/-- The theorem statement -/
theorem parallel_perpendicular_to_plane 
  (a b : Line3) (α : Plane3) 
  (h1 : parallel a b) 
  (h2 : perpendicular_to_plane a α) : 
  perpendicular_to_plane b α := by sorry

end NUMINAMATH_CALUDE_parallel_perpendicular_to_plane_l92_9246


namespace NUMINAMATH_CALUDE_average_daily_allowance_l92_9291

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The weekly calorie allowance -/
def weekly_allowance : ℕ := 10500

/-- The average daily calorie allowance -/
def daily_allowance : ℕ := weekly_allowance / days_in_week

theorem average_daily_allowance :
  daily_allowance = 1500 :=
sorry

end NUMINAMATH_CALUDE_average_daily_allowance_l92_9291


namespace NUMINAMATH_CALUDE_number_of_gigs_played_l92_9256

-- Define the earnings for each band member
def lead_singer_earnings : ℕ := 30
def guitarist_earnings : ℕ := 25
def bassist_earnings : ℕ := 20
def drummer_earnings : ℕ := 25
def keyboardist_earnings : ℕ := 20
def backup_singer_earnings : ℕ := 15

-- Define the total earnings per gig
def earnings_per_gig : ℕ := lead_singer_earnings + guitarist_earnings + bassist_earnings + 
                            drummer_earnings + keyboardist_earnings + backup_singer_earnings

-- Define the total earnings from all gigs
def total_earnings : ℕ := 2055

-- Theorem: The number of gigs played is 15
theorem number_of_gigs_played : 
  ⌊(total_earnings : ℚ) / (earnings_per_gig : ℚ)⌋ = 15 := by sorry

end NUMINAMATH_CALUDE_number_of_gigs_played_l92_9256


namespace NUMINAMATH_CALUDE_geometric_transformation_of_arithmetic_sequence_exists_unique_x_l92_9278

def arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem geometric_transformation_of_arithmetic_sequence 
  (a : ℕ → ℝ) (x : ℝ) : Prop :=
  arithmetic_sequence a ∧ 
  a 1 = -2 ∧ 
  a 2 = 0 ∧
  (a 1 + x) * (a 5 + x) = (a 4 + x)^2

theorem exists_unique_x (a : ℕ → ℝ) : 
  ∃! x : ℝ, geometric_transformation_of_arithmetic_sequence a x := by
sorry

end NUMINAMATH_CALUDE_geometric_transformation_of_arithmetic_sequence_exists_unique_x_l92_9278


namespace NUMINAMATH_CALUDE_library_crates_needed_l92_9209

theorem library_crates_needed (novels : ℕ) (comics : ℕ) (documentaries : ℕ) (albums : ℕ) 
  (crate_capacity : ℕ) (h1 : novels = 145) (h2 : comics = 271) (h3 : documentaries = 419) 
  (h4 : albums = 209) (h5 : crate_capacity = 9) : 
  (novels + comics + documentaries + albums + crate_capacity - 1) / crate_capacity = 116 := by
  sorry

end NUMINAMATH_CALUDE_library_crates_needed_l92_9209


namespace NUMINAMATH_CALUDE_son_age_proof_l92_9258

theorem son_age_proof (father_age son_age : ℕ) : 
  father_age = son_age + 29 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 27 := by
sorry

end NUMINAMATH_CALUDE_son_age_proof_l92_9258


namespace NUMINAMATH_CALUDE_inequality_theorem_l92_9212

theorem inequality_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^5 - a^2 + 3) * (b^5 - b^2 + 3) * (c^5 - c^2 + 3) ≥ (a + b + c)^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l92_9212


namespace NUMINAMATH_CALUDE_no_solution_exists_l92_9249

theorem no_solution_exists : ¬∃ x : ℝ, (16 : ℝ)^(3*x - 1) = (64 : ℝ)^(2*x + 3) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_exists_l92_9249


namespace NUMINAMATH_CALUDE_luna_bus_cost_l92_9213

/-- The distance from city X to city Y in kilometers -/
def distance_XY : ℝ := 4500

/-- The cost per kilometer for bus travel in dollars -/
def bus_cost_per_km : ℝ := 0.20

/-- The total cost for Luna to bus from city X to city Y -/
def total_bus_cost : ℝ := distance_XY * bus_cost_per_km

/-- Theorem stating that the total bus cost for Luna to travel from X to Y is $900 -/
theorem luna_bus_cost : total_bus_cost = 900 := by
  sorry

end NUMINAMATH_CALUDE_luna_bus_cost_l92_9213


namespace NUMINAMATH_CALUDE_linear_function_property_l92_9259

/-- A linear function is a function of the form f(x) = mx + b where m and b are constants. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ m b : ℝ, ∀ x, f x = m * x + b

theorem linear_function_property (g : ℝ → ℝ) 
  (hlinear : LinearFunction g) 
  (hg : g 8 - g 4 = 16) : 
  g 16 - g 4 = 48 := by
sorry

end NUMINAMATH_CALUDE_linear_function_property_l92_9259


namespace NUMINAMATH_CALUDE_zhuoma_combinations_l92_9238

/-- The number of different styles of backpacks -/
def num_backpack_styles : ℕ := 2

/-- The number of different styles of pencil cases -/
def num_pencil_case_styles : ℕ := 2

/-- The number of different combinations of backpack and pencil case styles -/
def num_combinations : ℕ := num_backpack_styles * num_pencil_case_styles

theorem zhuoma_combinations :
  num_combinations = 4 :=
by sorry

end NUMINAMATH_CALUDE_zhuoma_combinations_l92_9238


namespace NUMINAMATH_CALUDE_correct_average_weight_l92_9252

/-- Given a class of boys with an incorrect average weight due to a misread measurement,
    calculate the correct average weight. -/
theorem correct_average_weight
  (n : ℕ) -- number of boys
  (initial_avg : ℝ) -- initial (incorrect) average weight
  (misread_weight : ℝ) -- weight that was misread
  (correct_weight : ℝ) -- correct weight for the misread value
  (h1 : n = 20) -- there are 20 boys
  (h2 : initial_avg = 58.4) -- initial average was 58.4 kg
  (h3 : misread_weight = 56) -- misread weight was 56 kg
  (h4 : correct_weight = 60) -- correct weight is 60 kg
  : (n * initial_avg + correct_weight - misread_weight) / n = 58.6 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_weight_l92_9252


namespace NUMINAMATH_CALUDE_power_division_equality_l92_9210

theorem power_division_equality : (10 ^ 7) / (10 ^ 3 / 10 ^ 2) = 10 ^ 6 := by sorry

end NUMINAMATH_CALUDE_power_division_equality_l92_9210


namespace NUMINAMATH_CALUDE_min_value_theorem_l92_9219

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  (3 / x + 4 / y) ≥ 7 + 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l92_9219


namespace NUMINAMATH_CALUDE_cost_of_750_candies_l92_9202

/-- The cost of buying a specific number of chocolate candies given the following conditions:
  * A box contains a fixed number of candies
  * A box costs a fixed amount
  * There is a discount percentage for buying more than a certain number of boxes
  * We need to buy a specific number of candies -/
def cost_of_candies (candies_per_box : ℕ) (cost_per_box : ℚ) (discount_percentage : ℚ) 
  (discount_threshold : ℕ) (total_candies : ℕ) : ℚ :=
  let boxes_needed : ℕ := (total_candies + candies_per_box - 1) / candies_per_box
  let total_cost : ℚ := boxes_needed * cost_per_box
  if boxes_needed > discount_threshold
  then total_cost * (1 - discount_percentage)
  else total_cost

theorem cost_of_750_candies :
  cost_of_candies 30 (7.5) (1/10) 20 750 = (168.75) := by
  sorry

end NUMINAMATH_CALUDE_cost_of_750_candies_l92_9202


namespace NUMINAMATH_CALUDE_brian_running_time_l92_9206

theorem brian_running_time (todd_time brian_time : ℕ) : 
  todd_time = 88 → 
  brian_time = todd_time + 8 → 
  brian_time = 96 := by
sorry

end NUMINAMATH_CALUDE_brian_running_time_l92_9206


namespace NUMINAMATH_CALUDE_paving_stone_width_l92_9217

/-- Theorem: Width of paving stones in a rectangular courtyard -/
theorem paving_stone_width
  (courtyard_length : ℝ)
  (courtyard_width : ℝ)
  (stone_length : ℝ)
  (stone_count : ℕ)
  (h1 : courtyard_length = 20)
  (h2 : courtyard_width = 16.5)
  (h3 : stone_length = 2.5)
  (h4 : stone_count = 66)
  : ∃ (stone_width : ℝ),
    courtyard_length * courtyard_width = stone_count * (stone_length * stone_width) ∧
    stone_width = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_paving_stone_width_l92_9217


namespace NUMINAMATH_CALUDE_senate_committee_arrangements_l92_9280

/-- The number of ways to arrange n distinguishable people around a circular table -/
def circularArrangements (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of politicians on the Senate committee -/
def numPoliticians : ℕ := 12

theorem senate_committee_arrangements :
  circularArrangements numPoliticians = 39916800 := by
  sorry

end NUMINAMATH_CALUDE_senate_committee_arrangements_l92_9280


namespace NUMINAMATH_CALUDE_inequality_proof_l92_9263

def f (x : ℝ) : ℝ := |x + 2| + |x - 2|

def A : Set ℝ := {x | f x ≤ 6}

theorem inequality_proof (m n : ℝ) (hm : m ∈ A) (hn : n ∈ A) :
  |1/3 * m - 1/2 * n| ≤ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l92_9263


namespace NUMINAMATH_CALUDE_intersection_M_N_l92_9299

def M : Set ℤ := {-1, 1}
def N : Set ℤ := {x | x^2 + x = 0}

theorem intersection_M_N : M ∩ N = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l92_9299


namespace NUMINAMATH_CALUDE_janet_savings_l92_9257

theorem janet_savings (monthly_rent : ℕ) (advance_months : ℕ) (deposit : ℕ) (additional_needed : ℕ) : 
  monthly_rent = 1250 →
  advance_months = 2 →
  deposit = 500 →
  additional_needed = 775 →
  monthly_rent * advance_months + deposit - additional_needed = 2225 :=
by sorry

end NUMINAMATH_CALUDE_janet_savings_l92_9257


namespace NUMINAMATH_CALUDE_positive_integer_triplet_solution_l92_9222

theorem positive_integer_triplet_solution (x y z : ℕ+) :
  (x + y)^2 + 3*x + y + 1 = z^2 → x = y ∧ z = 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_triplet_solution_l92_9222


namespace NUMINAMATH_CALUDE_total_snowballs_l92_9233

def lucy_snowballs : ℕ := 19
def peter_snowballs : ℕ := 47
def charlie_snowballs : ℕ := lucy_snowballs + 31

theorem total_snowballs : lucy_snowballs + charlie_snowballs + peter_snowballs = 116 := by
  sorry

end NUMINAMATH_CALUDE_total_snowballs_l92_9233


namespace NUMINAMATH_CALUDE_inequality_proof_l92_9204

theorem inequality_proof (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_condition : a + b + c = 1) : 
  (1 + a) * (1 + b) * (1 + c) ≥ 8 * (1 - a) * (1 - b) * (1 - c) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l92_9204


namespace NUMINAMATH_CALUDE_trapezoid_area_l92_9239

-- Define the rectangle ABCD
structure Rectangle where
  width : ℝ
  height : ℝ
  area : ℝ
  area_eq : area = width * height

-- Define the trapezoid DEFG
structure Trapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ

-- Define the problem setup
def problem_setup (rect : Rectangle) (trap : Trapezoid) : Prop :=
  rect.area = 108 ∧
  trap.base1 = rect.height / 2 ∧
  trap.base2 = rect.width / 2 ∧
  trap.height = rect.height / 2

-- Theorem to prove
theorem trapezoid_area 
  (rect : Rectangle) 
  (trap : Trapezoid) 
  (h : problem_setup rect trap) : 
  (trap.base1 + trap.base2) / 2 * trap.height = 27 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_area_l92_9239


namespace NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l92_9208

theorem greatest_integer_fraction_inequality :
  ∀ x : ℤ, (8 : ℚ) / 11 > (x : ℚ) / 17 ↔ x ≤ 12 :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_fraction_inequality_l92_9208


namespace NUMINAMATH_CALUDE_remainder_problem_l92_9230

theorem remainder_problem : (5^7 + 9^6 + 3^5) % 7 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l92_9230


namespace NUMINAMATH_CALUDE_triangle_angle_C_l92_9297

theorem triangle_angle_C (A B C : Real) (a b c : Real) :
  a + b + c = Real.sqrt 2 + 1 →
  (1/2) * a * b * Real.sin C = (1/6) * Real.sin C →
  Real.sin A + Real.sin B = Real.sqrt 2 * Real.sin C →
  C = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_C_l92_9297


namespace NUMINAMATH_CALUDE_max_sum_of_digits_is_24_l92_9228

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours < 24
  minute_valid : minutes < 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a Time24 -/
def sumOfDigitsTime24 (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits in a 24-hour format digital watch display -/
def maxSumOfDigits : Nat := 24

theorem max_sum_of_digits_is_24 :
  ∀ t : Time24, sumOfDigitsTime24 t ≤ maxSumOfDigits :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_is_24_l92_9228


namespace NUMINAMATH_CALUDE_power_equation_solution_l92_9271

theorem power_equation_solution : ∃ n : ℤ, (5 : ℝ) ^ (4 * n) = (1 / 5 : ℝ) ^ (n - 30) ∧ n = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l92_9271


namespace NUMINAMATH_CALUDE_juan_tricycles_l92_9281

theorem juan_tricycles (cars bicycles pickups : ℕ) (total_tires : ℕ) : 
  cars = 15 → 
  bicycles = 3 → 
  pickups = 8 → 
  total_tires = 101 → 
  ∃ (tricycles : ℕ), 
    cars * 4 + bicycles * 2 + pickups * 4 + tricycles * 3 = total_tires ∧ 
    tricycles = 1 := by
  sorry

end NUMINAMATH_CALUDE_juan_tricycles_l92_9281


namespace NUMINAMATH_CALUDE_intersection_M_N_l92_9282

def M : Set (ℝ × ℝ) := {p | (p.1^2 / 9) + (p.2^2 / 4) = 1}

def N : Set (ℝ × ℝ) := {p | (p.1 / 3) + (p.2 / 2) = 1}

theorem intersection_M_N : M ∩ N = {(3, 0), (0, 2)} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l92_9282


namespace NUMINAMATH_CALUDE_perfect_square_factors_of_2880_l92_9294

def prime_factorization (n : ℕ) : List (ℕ × ℕ) := sorry

def is_perfect_square (n : ℕ) : Prop := sorry

def count_perfect_square_factors (n : ℕ) : ℕ := sorry

theorem perfect_square_factors_of_2880 :
  let factorization := prime_factorization 2880
  (factorization = [(2, 6), (3, 2), (5, 1)]) →
  count_perfect_square_factors 2880 = 8 := by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_of_2880_l92_9294


namespace NUMINAMATH_CALUDE_paths_from_A_to_D_l92_9262

/-- Represents a point in the graph -/
inductive Point
| A
| B
| C
| D

/-- Represents the number of paths between two points -/
def num_paths (start finish : Point) : ℕ :=
  match start, finish with
  | Point.A, Point.B => 2
  | Point.B, Point.C => 2
  | Point.C, Point.D => 2
  | Point.A, Point.C => 1
  | _, _ => 0

/-- The total number of paths from A to D -/
def total_paths : ℕ :=
  (num_paths Point.A Point.B) * (num_paths Point.B Point.C) * (num_paths Point.C Point.D) +
  (num_paths Point.A Point.C) * (num_paths Point.C Point.D)

theorem paths_from_A_to_D : total_paths = 10 := by
  sorry

end NUMINAMATH_CALUDE_paths_from_A_to_D_l92_9262


namespace NUMINAMATH_CALUDE_sun_division_l92_9272

theorem sun_division (x y z total : ℚ) : 
  (y = (45/100) * x) →  -- y gets 45 paisa for each rupee x gets
  (z = (30/100) * x) →  -- z gets 30 paisa for each rupee x gets
  (y = 63) →            -- y's share is Rs. 63
  (total = x + y + z) → -- total is the sum of all shares
  (total = 245) :=      -- prove that the total is Rs. 245
by
  sorry

end NUMINAMATH_CALUDE_sun_division_l92_9272


namespace NUMINAMATH_CALUDE_bus_students_problem_l92_9251

theorem bus_students_problem (initial_students final_students : ℕ) 
  (h1 : initial_students = 28)
  (h2 : final_students = 58) :
  (0.4 : ℝ) * (final_students - initial_students) = 12 := by
  sorry

end NUMINAMATH_CALUDE_bus_students_problem_l92_9251


namespace NUMINAMATH_CALUDE_cheaper_feed_cost_l92_9266

/-- Proves that the cost of the cheaper feed is $0.18 per pound given the problem conditions --/
theorem cheaper_feed_cost (total_mix : ℝ) (mix_price : ℝ) (expensive_price : ℝ) (cheaper_amount : ℝ) 
  (h1 : total_mix = 35)
  (h2 : mix_price = 0.36)
  (h3 : expensive_price = 0.53)
  (h4 : cheaper_amount = 17) :
  ∃ (cheaper_price : ℝ), 
    cheaper_price * cheaper_amount + expensive_price * (total_mix - cheaper_amount) = mix_price * total_mix ∧ 
    cheaper_price = 0.18 := by
  sorry

end NUMINAMATH_CALUDE_cheaper_feed_cost_l92_9266


namespace NUMINAMATH_CALUDE_decimal_sum_to_fraction_l92_9276

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0005 + 0.00006 = 1466 / 6250 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_to_fraction_l92_9276


namespace NUMINAMATH_CALUDE_linear_equation_solution_l92_9268

theorem linear_equation_solution (m : ℝ) : (2 * m + 2 = 0) → m = -1 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l92_9268


namespace NUMINAMATH_CALUDE_four_digit_permutations_l92_9223

-- Define the multiset
def digit_multiset : Multiset ℕ := {3, 3, 7, 7}

-- Define the function to calculate permutations of a multiset
noncomputable def multiset_permutations (m : Multiset ℕ) : ℕ := sorry

-- Theorem statement
theorem four_digit_permutations :
  multiset_permutations digit_multiset = 6 := by sorry

end NUMINAMATH_CALUDE_four_digit_permutations_l92_9223


namespace NUMINAMATH_CALUDE_not_sufficient_for_geometric_sequence_l92_9207

theorem not_sufficient_for_geometric_sequence 
  (a : ℕ → ℝ) 
  (h : ∀ n : ℕ, n ≥ 2 → a (n + 1) * a (n - 1) = a n ^ 2) :
  ¬ (∀ n : ℕ, ∃ r : ℝ, ∀ k : ℕ, a (n + k) = a n * r ^ k) :=
sorry

end NUMINAMATH_CALUDE_not_sufficient_for_geometric_sequence_l92_9207


namespace NUMINAMATH_CALUDE_average_weight_problem_l92_9201

/-- Given the weights of three people with specific relationships, prove their average weight. -/
theorem average_weight_problem (jalen_weight ponce_weight ishmael_weight : ℕ) : 
  jalen_weight = 160 ∧ 
  ponce_weight = jalen_weight - 10 ∧ 
  ishmael_weight = ponce_weight + 20 → 
  (jalen_weight + ponce_weight + ishmael_weight) / 3 = 160 := by
  sorry


end NUMINAMATH_CALUDE_average_weight_problem_l92_9201


namespace NUMINAMATH_CALUDE_correct_arrangements_no_adjacent_women_correct_arrangements_no_adjacent_spouses_or_same_gender_l92_9242

/-- Represents a couple consisting of a man and a woman -/
structure Couple :=
  (man : ℕ)
  (woman : ℕ)

/-- The number of couples -/
def num_couples : ℕ := 4

/-- The number of ways to arrange four couples around a round table 
    such that no two women sit next to each other -/
def arrangements_no_adjacent_women : ℕ := 144

/-- The number of arrangements possible if no two spouses, 
    no two women, and no two men sit next to each other -/
def arrangements_no_adjacent_spouses_or_same_gender : ℕ := 12

/-- Theorem stating the correct number of arrangements for the first question -/
theorem correct_arrangements_no_adjacent_women :
  arrangements_no_adjacent_women = 144 :=
by sorry

/-- Theorem stating the correct number of arrangements for the second question -/
theorem correct_arrangements_no_adjacent_spouses_or_same_gender :
  arrangements_no_adjacent_spouses_or_same_gender = 12 :=
by sorry

end NUMINAMATH_CALUDE_correct_arrangements_no_adjacent_women_correct_arrangements_no_adjacent_spouses_or_same_gender_l92_9242


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l92_9241

theorem lcm_hcf_problem (n : ℕ) 
  (h1 : Nat.lcm 12 n = 60) 
  (h2 : Nat.gcd 12 n = 3) : 
  n = 15 := by
sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l92_9241


namespace NUMINAMATH_CALUDE_union_implies_m_equals_two_l92_9243

theorem union_implies_m_equals_two (A B : Set ℕ) (m : ℕ) : 
  A = {1, 3, m} → 
  B = {3, 4} → 
  A ∪ B = {1, 2, 3, 4} → 
  m = 2 := by
sorry

end NUMINAMATH_CALUDE_union_implies_m_equals_two_l92_9243


namespace NUMINAMATH_CALUDE_two_boys_three_girls_probability_l92_9231

-- Define the number of children
def n : ℕ := 5

-- Define the number of boys
def k : ℕ := 2

-- Define the probability of having a boy
def p : ℚ := 1/2

-- Define the binomial coefficient function
def binomial_coefficient (n k : ℕ) : ℕ := sorry

-- Define the probability function
def probability (n k : ℕ) (p : ℚ) : ℚ := sorry

-- Theorem statement
theorem two_boys_three_girls_probability :
  probability n k p = 0.3125 := by sorry

end NUMINAMATH_CALUDE_two_boys_three_girls_probability_l92_9231


namespace NUMINAMATH_CALUDE_parallelepiped_arrangement_exists_l92_9255

/-- Represents a parallelepiped in 3D space -/
structure Parallelepiped where
  -- Define the parallelepiped structure (simplified for this example)
  dummy : Unit

/-- Represents a point in 3D space -/
structure Point where
  -- Define the point structure (simplified for this example)
  dummy : Unit

/-- Checks if two parallelepipeds intersect -/
def intersects (p1 p2 : Parallelepiped) : Prop :=
  sorry

/-- Checks if a point is inside a parallelepiped -/
def isInside (point : Point) (p : Parallelepiped) : Prop :=
  sorry

/-- Checks if a vertex of a parallelepiped is visible from a point -/
def isVertexVisible (point : Point) (p : Parallelepiped) : Prop :=
  sorry

/-- Theorem stating the existence of the required arrangement -/
theorem parallelepiped_arrangement_exists : 
  ∃ (parallelepipeds : Fin 6 → Parallelepiped) (observationPoint : Point),
    (∀ i j : Fin 6, i ≠ j → ¬intersects (parallelepipeds i) (parallelepipeds j)) ∧
    (∀ i : Fin 6, ¬isInside observationPoint (parallelepipeds i)) ∧
    (∀ i : Fin 6, ¬isVertexVisible observationPoint (parallelepipeds i)) :=
  sorry

end NUMINAMATH_CALUDE_parallelepiped_arrangement_exists_l92_9255


namespace NUMINAMATH_CALUDE_line_plane_perpendicularity_l92_9265

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m n : Line) (β : Plane) 
  (h1 : m ≠ n) 
  (h2 : parallel m n) 
  (h3 : perpendicular m β) : 
  perpendicular n β :=
sorry

end NUMINAMATH_CALUDE_line_plane_perpendicularity_l92_9265


namespace NUMINAMATH_CALUDE_triangle_angle_and_max_area_l92_9295

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and vectors m and n, prove the measure of angle C and the maximum area. -/
theorem triangle_angle_and_max_area 
  (A B C : ℝ) (a b c : ℝ) (m n : ℝ × ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →  -- Triangle angle sum
  m = (Real.sin A, Real.sin B) →           -- Definition of m
  n = (Real.cos B, Real.cos A) →           -- Definition of n
  m.1 * n.1 + m.2 * n.2 = -Real.sin (2 * C) →  -- Dot product condition
  c = 2 * Real.sqrt 3 →                    -- Given value of c
  C = 2 * π / 3 ∧                          -- Angle C
  (∃ (S : ℝ), S ≤ Real.sqrt 3 ∧            -- Maximum area
    ∀ (S' : ℝ), S' = 1/2 * a * b * Real.sin C → S' ≤ S) :=
by sorry

end NUMINAMATH_CALUDE_triangle_angle_and_max_area_l92_9295


namespace NUMINAMATH_CALUDE_vector_collinearity_l92_9225

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem vector_collinearity (m : ℝ) :
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (m, 2*m - 3)
  collinear a b → m = -3 := by sorry

end NUMINAMATH_CALUDE_vector_collinearity_l92_9225


namespace NUMINAMATH_CALUDE_stratified_sampling_group_c_l92_9275

/-- Represents the number of cities to be selected from a group in a stratified sampling -/
def citiesSelected (totalSelected : ℕ) (totalCities : ℕ) (groupCities : ℕ) : ℕ :=
  (totalSelected * groupCities) / totalCities

theorem stratified_sampling_group_c (totalCities : ℕ) (groupACities : ℕ) (groupBCities : ℕ) 
    (totalSelected : ℕ) (hTotal : totalCities = 48) (hA : groupACities = 8) (hB : groupBCities = 24) 
    (hSelected : totalSelected = 12) :
    citiesSelected totalSelected totalCities (totalCities - groupACities - groupBCities) = 4 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_group_c_l92_9275


namespace NUMINAMATH_CALUDE_part1_part2_l92_9284

-- Define the quadratic function f(x)
def f (q : ℝ) (x : ℝ) : ℝ := x^2 - 16*x + q + 3

-- Part 1: Prove that if f has a root in [-1, 1], then q ∈ [-20, 12]
theorem part1 (q : ℝ) : 
  (∃ x ∈ Set.Icc (-1) 1, f q x = 0) → q ∈ Set.Icc (-20) 12 :=
by sorry

-- Part 2: Prove that if f(x) + 51 ≥ 0 for all x ∈ [q, 10], then q ∈ [9, 10)
theorem part2 (q : ℝ) :
  (∀ x ∈ Set.Icc q 10, f q x + 51 ≥ 0) → q ∈ Set.Ici 9 ∩ Set.Iio 10 :=
by sorry

end NUMINAMATH_CALUDE_part1_part2_l92_9284


namespace NUMINAMATH_CALUDE_video_game_lives_l92_9244

theorem video_game_lives (initial_lives won_lives gained_lives : Float) 
  (h1 : initial_lives = 43.0)
  (h2 : won_lives = 14.0)
  (h3 : gained_lives = 27.0) :
  initial_lives + won_lives + gained_lives = 84.0 := by
  sorry

end NUMINAMATH_CALUDE_video_game_lives_l92_9244


namespace NUMINAMATH_CALUDE_min_distance_and_line_equation_l92_9293

-- Define the line l: x - y + 3 = 0
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- Define the circle C: (x - 1)^2 + y^2 = 1
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define the distance PA
def distance_PA (x y : ℝ) : ℝ := sorry

-- Define the equation of line AB
def line_AB (x y : ℝ) : Prop := 2*x - 2*y - 1 = 0

-- Theorem statement
theorem min_distance_and_line_equation :
  (∃ (x y : ℝ), point_P x y ∧ 
    (∀ (x' y' : ℝ), point_P x' y' → distance_PA x y ≤ distance_PA x' y')) ∧
  (∀ (x y : ℝ), point_P x y ∧ distance_PA x y = Real.sqrt 7 → line_AB x y) :=
sorry

end NUMINAMATH_CALUDE_min_distance_and_line_equation_l92_9293


namespace NUMINAMATH_CALUDE_quarter_percentage_approx_l92_9216

/-- Represents the number and value of coins -/
structure Coins where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ
  dime_value : ℕ
  quarter_value : ℕ
  nickel_value : ℕ

/-- Calculates the percentage of quarters in the total value -/
def quarter_percentage (c : Coins) : ℚ :=
  let total_value := c.dimes * c.dime_value + c.quarters * c.quarter_value + c.nickels * c.nickel_value
  let quarter_value := c.quarters * c.quarter_value
  (quarter_value : ℚ) / (total_value : ℚ) * 100

/-- Theorem stating that the percentage of quarters is approximately 51.28% -/
theorem quarter_percentage_approx (c : Coins) 
  (h1 : c.dimes = 80) (h2 : c.quarters = 40) (h3 : c.nickels = 30)
  (h4 : c.dime_value = 10) (h5 : c.quarter_value = 25) (h6 : c.nickel_value = 5) : 
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 : ℚ) / 100 ∧ |quarter_percentage c - (5128 : ℚ) / 100| < ε := by
  sorry

end NUMINAMATH_CALUDE_quarter_percentage_approx_l92_9216


namespace NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l92_9254

theorem quadratic_rewrite_ratio : ∃ (c r s : ℚ),
  (∀ k, 8 * k^2 - 12 * k + 20 = c * (k + r)^2 + s) ∧
  s / r = -62 / 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l92_9254


namespace NUMINAMATH_CALUDE_stock_percentage_l92_9235

/-- Calculate the percentage of a stock given income, stock price, and total investment. -/
theorem stock_percentage (income : ℚ) (stock_price : ℚ) (total_investment : ℚ) :
  income = 450 →
  stock_price = 108 →
  total_investment = 4860 →
  (income / total_investment) * 100 = (450 : ℚ) / 4860 * 100 := by
  sorry

end NUMINAMATH_CALUDE_stock_percentage_l92_9235


namespace NUMINAMATH_CALUDE_min_value_abc_l92_9226

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 1/a + 1/b + 1/c = 9) :
  a^4 * b^3 * c^2 ≥ 1/1152 ∧
  ∃ (a₀ b₀ c₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧
    1/a₀ + 1/b₀ + 1/c₀ = 9 ∧ a₀^4 * b₀^3 * c₀^2 = 1/1152 :=
by sorry

end NUMINAMATH_CALUDE_min_value_abc_l92_9226


namespace NUMINAMATH_CALUDE_roses_per_decoration_correct_l92_9290

/-- The number of white roses in each table decoration -/
def roses_per_decoration : ℕ := 12

/-- The number of bouquets -/
def num_bouquets : ℕ := 5

/-- The number of table decorations -/
def num_decorations : ℕ := 7

/-- The number of white roses in each bouquet -/
def roses_per_bouquet : ℕ := 5

/-- The total number of white roses used -/
def total_roses : ℕ := 109

theorem roses_per_decoration_correct :
  roses_per_decoration * num_decorations + roses_per_bouquet * num_bouquets = total_roses :=
by sorry

end NUMINAMATH_CALUDE_roses_per_decoration_correct_l92_9290


namespace NUMINAMATH_CALUDE_distance_between_foci_l92_9218

/-- The ellipse equation -/
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 2)^2 + (y + 3)^2) + Real.sqrt ((x + 6)^2 + (y - 7)^2) = 24

/-- The first focus of the ellipse -/
def F₁ : ℝ × ℝ := (2, -3)

/-- The second focus of the ellipse -/
def F₂ : ℝ × ℝ := (-6, 7)

/-- The theorem stating the distance between the foci -/
theorem distance_between_foci :
  Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2) = 2 * Real.sqrt 41 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_foci_l92_9218


namespace NUMINAMATH_CALUDE_fraction_sum_mixed_number_equality_main_theorem_l92_9245

theorem fraction_sum : (3 : ℚ) / 4 + (5 : ℚ) / 6 + (4 : ℚ) / 3 = (35 : ℚ) / 12 := by
  sorry

theorem mixed_number_equality : (35 : ℚ) / 12 = 2 + (11 : ℚ) / 12 := by
  sorry

theorem main_theorem : (3 : ℚ) / 4 + (5 : ℚ) / 6 + (1 + (1 : ℚ) / 3) = 2 + (11 : ℚ) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_mixed_number_equality_main_theorem_l92_9245
