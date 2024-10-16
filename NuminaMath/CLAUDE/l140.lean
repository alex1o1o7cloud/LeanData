import Mathlib

namespace NUMINAMATH_CALUDE_symmetry_xoy_plane_l140_14076

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xoy plane in 3D space -/
def xoy_plane : Set Point3D := {p : Point3D | p.z = 0}

/-- Symmetry with respect to the xoy plane -/
def symmetric_xoy (p : Point3D) : Point3D :=
  ⟨p.x, p.y, -p.z⟩

theorem symmetry_xoy_plane :
  let P : Point3D := ⟨1, 3, -5⟩
  symmetric_xoy P = ⟨1, 3, 5⟩ := by
  sorry


end NUMINAMATH_CALUDE_symmetry_xoy_plane_l140_14076


namespace NUMINAMATH_CALUDE_colored_points_segment_existence_l140_14092

/-- Represents a color --/
inductive Color
  | Red
  | Blue
  | Green
  | Yellow

/-- Represents a colored point on a line --/
structure ColoredPoint where
  position : ℝ
  color : Color

/-- The main theorem --/
theorem colored_points_segment_existence
  (n : ℕ)
  (h_n : n ≥ 4)
  (points : Fin n → ColoredPoint)
  (h_distinct : ∀ i j, i ≠ j → (points i).position ≠ (points j).position)
  (h_all_colors : ∀ c : Color, ∃ i, (points i).color = c) :
  ∃ (a b : ℝ), a < b ∧
    (∃ (c₁ c₂ : Color), c₁ ≠ c₂ ∧
      (∃! i, a ≤ (points i).position ∧ (points i).position ≤ b ∧ (points i).color = c₁) ∧
      (∃! j, a ≤ (points j).position ∧ (points j).position ≤ b ∧ (points j).color = c₂)) ∧
    (∃ (c₃ c₄ : Color), c₃ ≠ c₄ ∧ c₃ ≠ c₁ ∧ c₃ ≠ c₂ ∧ c₄ ≠ c₁ ∧ c₄ ≠ c₂ ∧
      (∃ i, a ≤ (points i).position ∧ (points i).position ≤ b ∧ (points i).color = c₃) ∧
      (∃ j, a ≤ (points j).position ∧ (points j).position ≤ b ∧ (points j).color = c₄)) :=
by
  sorry


end NUMINAMATH_CALUDE_colored_points_segment_existence_l140_14092


namespace NUMINAMATH_CALUDE_initial_average_height_l140_14003

/-- The initially calculated average height of students in a class with measurement error -/
theorem initial_average_height (n : ℕ) (incorrect_height actual_height : ℝ) (actual_average : ℝ) 
  (hn : n = 20)
  (h_incorrect : incorrect_height = 151)
  (h_actual : actual_height = 136)
  (h_average : actual_average = 174.25) :
  ∃ (initial_average : ℝ), 
    initial_average * n = actual_average * n - (incorrect_height - actual_height) ∧ 
    initial_average = 173.5 := by
  sorry


end NUMINAMATH_CALUDE_initial_average_height_l140_14003


namespace NUMINAMATH_CALUDE_square_of_negative_double_product_l140_14071

theorem square_of_negative_double_product (x y : ℝ) : (-2 * x * y)^2 = 4 * x^2 * y^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_double_product_l140_14071


namespace NUMINAMATH_CALUDE_roots_sum_reciprocal_l140_14018

theorem roots_sum_reciprocal (x₁ x₂ : ℝ) : 
  x₁^2 + 8*x₁ + 4 = 0 → x₂^2 + 8*x₂ + 4 = 0 → x₁ ≠ x₂ → 
  (1 / x₁) + (1 / x₂) = -2 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_reciprocal_l140_14018


namespace NUMINAMATH_CALUDE_parabola_translation_l140_14053

-- Define the initial parabola
def initial_parabola (x : ℝ) : ℝ := (x - 1)^2 - 4

-- Define the translated parabola
def translated_parabola (x : ℝ) : ℝ := (x - 4)^2 - 2

-- Define the translation
def translation_right : ℝ := 3
def translation_up : ℝ := 2

-- Theorem statement
theorem parabola_translation :
  ∀ x : ℝ, translated_parabola x = initial_parabola (x - translation_right) + translation_up :=
by sorry

end NUMINAMATH_CALUDE_parabola_translation_l140_14053


namespace NUMINAMATH_CALUDE_expression_evaluation_l140_14000

theorem expression_evaluation : 2 + (0 * 2^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l140_14000


namespace NUMINAMATH_CALUDE_hyperbola_iff_m_negative_l140_14011

/-- A conic section defined by the equation x^2 + my^2 = 1 -/
structure Conic (m : ℝ) where
  x : ℝ
  y : ℝ
  eq : x^2 + m*y^2 = 1

/-- Definition of a hyperbola -/
def IsHyperbola (m : ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ (x y : ℝ), x^2/a^2 - y^2/b^2 = 1 ↔ x^2 + m*y^2 = 1

/-- Theorem: The equation x^2 + my^2 = 1 represents a hyperbola if and only if m < 0 -/
theorem hyperbola_iff_m_negative (m : ℝ) : IsHyperbola m ↔ m < 0 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_iff_m_negative_l140_14011


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_problem_l140_14096

/-- The 'Crazy Silly School' series problem -/
theorem crazy_silly_school_series_problem 
  (total_books : ℕ) 
  (total_movies : ℕ) 
  (books_read : ℕ) 
  (movies_watched : ℕ) 
  (h1 : total_books = 25) 
  (h2 : total_movies = 35) 
  (h3 : books_read = 15) 
  (h4 : movies_watched = 29) :
  movies_watched - books_read = 14 := by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_problem_l140_14096


namespace NUMINAMATH_CALUDE_existence_of_even_odd_composition_l140_14058

theorem existence_of_even_odd_composition :
  ∃ (p q : ℝ → ℝ),
    (∀ x, p x = p (-x)) ∧
    (∀ x, p (q x) = -(p (q (-x)))) ∧
    (∃ x, p (q x) ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_even_odd_composition_l140_14058


namespace NUMINAMATH_CALUDE_bridge_length_l140_14013

/-- The length of a bridge given train parameters -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 150 ∧ 
  train_speed_kmh = 45 ∧ 
  crossing_time = 30 →
  ∃ (bridge_length : ℝ), bridge_length = 225 :=
by
  sorry


end NUMINAMATH_CALUDE_bridge_length_l140_14013


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l140_14079

/-- An arithmetic sequence with its partial sums -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- S_n is the sum of the first n terms

/-- Given an arithmetic sequence with S_10 = 10 and S_20 = 40, prove S_30 = 90 -/
theorem arithmetic_sequence_sum (a : ArithmeticSequence) 
  (h1 : a.S 10 = 10) (h2 : a.S 20 = 40) : a.S 30 = 90 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l140_14079


namespace NUMINAMATH_CALUDE_cos_thirteen_pi_thirds_l140_14090

theorem cos_thirteen_pi_thirds : Real.cos (13 * Real.pi / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_thirteen_pi_thirds_l140_14090


namespace NUMINAMATH_CALUDE_tax_rate_problem_l140_14004

/-- The tax rate problem in Country X -/
theorem tax_rate_problem (income : ℝ) (total_tax : ℝ) (tax_rate_above_40k : ℝ) :
  income = 50000 →
  total_tax = 8000 →
  tax_rate_above_40k = 0.2 →
  ∃ (tax_rate_below_40k : ℝ),
    tax_rate_below_40k * 40000 + tax_rate_above_40k * (income - 40000) = total_tax ∧
    tax_rate_below_40k = 0.15 := by
  sorry

end NUMINAMATH_CALUDE_tax_rate_problem_l140_14004


namespace NUMINAMATH_CALUDE_two_sector_area_l140_14001

/-- The area of a figure formed by two sectors of a circle -/
theorem two_sector_area (r : ℝ) (θ : ℝ) : 
  r = 15 → θ = 90 → 2 * (θ / 360) * π * r^2 = 112.5 * π := by sorry

end NUMINAMATH_CALUDE_two_sector_area_l140_14001


namespace NUMINAMATH_CALUDE_no_real_roots_l140_14043

theorem no_real_roots : ¬∃ x : ℝ, x + 2 * Real.sqrt (x - 5) = 6 := by sorry

end NUMINAMATH_CALUDE_no_real_roots_l140_14043


namespace NUMINAMATH_CALUDE_lateral_edge_length_for_specific_pyramid_l140_14093

/-- A regular quadrilateral pyramid with given base side length and volume -/
structure RegularQuadPyramid where
  base_side : ℝ
  volume : ℝ

/-- The length of the lateral edge of a regular quadrilateral pyramid -/
def lateral_edge_length (p : RegularQuadPyramid) : ℝ :=
  sorry

theorem lateral_edge_length_for_specific_pyramid :
  let p : RegularQuadPyramid := { base_side := 2, volume := 4 * Real.sqrt 3 / 3 }
  lateral_edge_length p = Real.sqrt 5 := by
    sorry

end NUMINAMATH_CALUDE_lateral_edge_length_for_specific_pyramid_l140_14093


namespace NUMINAMATH_CALUDE_total_money_calculation_l140_14026

def hundred_bills : ℕ := 2
def fifty_bills : ℕ := 5
def ten_bills : ℕ := 10

def hundred_value : ℕ := 100
def fifty_value : ℕ := 50
def ten_value : ℕ := 10

theorem total_money_calculation : 
  (hundred_bills * hundred_value) + (fifty_bills * fifty_value) + (ten_bills * ten_value) = 550 := by
  sorry

end NUMINAMATH_CALUDE_total_money_calculation_l140_14026


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l140_14086

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (21/16, 9/8)

/-- First line equation -/
def line1 (x y : ℚ) : Prop := 3 * y = -2 * x + 6

/-- Second line equation -/
def line2 (x y : ℚ) : Prop := 2 * y = 4 * x - 3

theorem intersection_point_is_unique :
  ∀ x y : ℚ, line1 x y ∧ line2 x y ↔ (x, y) = intersection_point :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l140_14086


namespace NUMINAMATH_CALUDE_profit_achieved_l140_14069

/-- Calculates the number of pens needed to be sold to achieve a specific profit --/
def pens_to_sell (num_purchased : ℕ) (purchase_price : ℚ) (sell_price : ℚ) (desired_profit : ℚ) : ℕ :=
  let total_cost := num_purchased * purchase_price
  let revenue_needed := total_cost + desired_profit
  (revenue_needed / sell_price).ceil.toNat

/-- Theorem stating that selling 1500 pens achieves the desired profit --/
theorem profit_achieved (num_purchased : ℕ) (purchase_price sell_price desired_profit : ℚ) :
  num_purchased = 2000 →
  purchase_price = 15/100 →
  sell_price = 30/100 →
  desired_profit = 150 →
  pens_to_sell num_purchased purchase_price sell_price desired_profit = 1500 := by
  sorry

end NUMINAMATH_CALUDE_profit_achieved_l140_14069


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l140_14010

theorem triangle_side_lengths 
  (α : Real) 
  (r R : Real) 
  (hr : r > 0) 
  (hR : R > 0) 
  (ha : ∃ a, a = Real.sqrt (r * R)) :
  ∃ b c : Real,
    b^2 - (Real.sqrt (r * R) * (5 + 4 * Real.cos α)) * b + 4 * r * R * (3 + 2 * Real.cos α) = 0 ∧
    c^2 - (Real.sqrt (r * R) * (5 + 4 * Real.cos α)) * c + 4 * r * R * (3 + 2 * Real.cos α) = 0 ∧
    b ≠ c :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l140_14010


namespace NUMINAMATH_CALUDE_p_difference_qr_l140_14008

theorem p_difference_qr (p q r : ℕ) : 
  p = 56 → 
  q = p / 8 →
  r = p / 8 →
  p - (q + r) = 42 := by
sorry

end NUMINAMATH_CALUDE_p_difference_qr_l140_14008


namespace NUMINAMATH_CALUDE_square_value_l140_14012

theorem square_value (square : ℚ) (h : (1:ℚ)/9 + (1:ℚ)/18 = (1:ℚ)/square) : square = 6 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l140_14012


namespace NUMINAMATH_CALUDE_mari_made_64_buttons_l140_14072

/-- Given the number of buttons Sue made -/
def sue_buttons : ℕ := 6

/-- Kendra's buttons in terms of Sue's -/
def kendra_buttons : ℕ := 2 * sue_buttons

/-- Mari's buttons in terms of Kendra's -/
def mari_buttons : ℕ := 4 + 5 * kendra_buttons

/-- Theorem stating that Mari made 64 buttons -/
theorem mari_made_64_buttons : mari_buttons = 64 := by
  sorry

end NUMINAMATH_CALUDE_mari_made_64_buttons_l140_14072


namespace NUMINAMATH_CALUDE_square_side_length_l140_14087

-- Define the perimeter of the square
def perimeter : ℝ := 34.8

-- Theorem: The length of one side of a square with perimeter 34.8 cm is 8.7 cm
theorem square_side_length : 
  perimeter / 4 = 8.7 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l140_14087


namespace NUMINAMATH_CALUDE_probability_of_choosing_quarter_l140_14015

def quarter_value : ℚ := 25 / 100
def nickel_value : ℚ := 5 / 100
def penny_value : ℚ := 1 / 100

def total_value_per_coin_type : ℚ := 10

theorem probability_of_choosing_quarter :
  let num_quarters := total_value_per_coin_type / quarter_value
  let num_nickels := total_value_per_coin_type / nickel_value
  let num_pennies := total_value_per_coin_type / penny_value
  let total_coins := num_quarters + num_nickels + num_pennies
  (num_quarters / total_coins : ℚ) = 1 / 31 := by
sorry

end NUMINAMATH_CALUDE_probability_of_choosing_quarter_l140_14015


namespace NUMINAMATH_CALUDE_carries_tshirt_purchase_l140_14024

/-- The cost of a single t-shirt in dollars -/
def tshirt_cost : ℝ := 9.65

/-- The number of t-shirts Carrie bought -/
def num_tshirts : ℕ := 12

/-- The total cost of Carrie's t-shirt purchase -/
def total_cost : ℝ := tshirt_cost * (num_tshirts : ℝ)

/-- Theorem: The total cost of Carrie's t-shirt purchase is $115.80 -/
theorem carries_tshirt_purchase : total_cost = 115.80 := by
  sorry

end NUMINAMATH_CALUDE_carries_tshirt_purchase_l140_14024


namespace NUMINAMATH_CALUDE_unique_number_with_properties_l140_14074

theorem unique_number_with_properties : ∃! n : ℕ, 
  50 < n ∧ n < 70 ∧ 
  ∃ k : ℤ, n - 3 = 5 * k ∧
  ∃ l : ℤ, n - 2 = 7 * l :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_number_with_properties_l140_14074


namespace NUMINAMATH_CALUDE_complete_collection_probability_l140_14050

def total_stickers : ℕ := 18
def selected_stickers : ℕ := 10
def uncollected_stickers : ℕ := 6
def collected_stickers : ℕ := 12

theorem complete_collection_probability :
  (Nat.choose uncollected_stickers uncollected_stickers * Nat.choose collected_stickers (selected_stickers - uncollected_stickers)) /
  (Nat.choose total_stickers selected_stickers) = 5 / 442 := by
  sorry

end NUMINAMATH_CALUDE_complete_collection_probability_l140_14050


namespace NUMINAMATH_CALUDE_total_apple_and_cherry_pies_l140_14083

def apple_pies : ℕ := 6
def pecan_pies : ℕ := 9
def pumpkin_pies : ℕ := 8
def cherry_pies : ℕ := 5
def blueberry_pies : ℕ := 3

theorem total_apple_and_cherry_pies : apple_pies + cherry_pies = 11 := by
  sorry

end NUMINAMATH_CALUDE_total_apple_and_cherry_pies_l140_14083


namespace NUMINAMATH_CALUDE_car_sale_price_l140_14039

/-- The final sale price of a car after multiple discounts and tax --/
theorem car_sale_price (original_price : ℝ) (discount1 discount2 discount3 tax_rate : ℝ) :
  original_price = 20000 ∧
  discount1 = 0.12 ∧
  discount2 = 0.10 ∧
  discount3 = 0.05 ∧
  tax_rate = 0.08 →
  (original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) * (1 + tax_rate)) = 16251.84 := by
  sorry

#eval (20000 : ℝ) * (1 - 0.12) * (1 - 0.10) * (1 - 0.05) * (1 + 0.08)

end NUMINAMATH_CALUDE_car_sale_price_l140_14039


namespace NUMINAMATH_CALUDE_justine_paper_usage_l140_14082

theorem justine_paper_usage 
  (total_sheets : ℕ) 
  (num_binders : ℕ) 
  (sheets_per_binder : ℕ) 
  (justine_binder : ℕ) 
  (h1 : total_sheets = 2450)
  (h2 : num_binders = 5)
  (h3 : sheets_per_binder = total_sheets / num_binders)
  (h4 : justine_binder = sheets_per_binder / 2) :
  justine_binder = 245 := by
  sorry

end NUMINAMATH_CALUDE_justine_paper_usage_l140_14082


namespace NUMINAMATH_CALUDE_cades_remaining_marbles_l140_14006

/-- Represents the number of marbles Cade has left after giving some away. -/
def marblesLeft (initial : ℕ) (givenAway : ℕ) : ℕ :=
  initial - givenAway

/-- Theorem stating that Cade's remaining marbles is the difference between his initial marbles and those given away. -/
theorem cades_remaining_marbles (initial : ℕ) (givenAway : ℕ) 
  (h : givenAway ≤ initial) : 
  marblesLeft initial givenAway = initial - givenAway :=
by
  sorry

#eval marblesLeft 87 8  -- Should output 79

end NUMINAMATH_CALUDE_cades_remaining_marbles_l140_14006


namespace NUMINAMATH_CALUDE_jeff_travel_distance_l140_14075

def speed1 : ℝ := 80
def time1 : ℝ := 6
def speed2 : ℝ := 60
def time2 : ℝ := 4
def speed3 : ℝ := 40
def time3 : ℝ := 2

def total_distance : ℝ := speed1 * time1 + speed2 * time2 + speed3 * time3

theorem jeff_travel_distance : total_distance = 800 := by
  sorry

end NUMINAMATH_CALUDE_jeff_travel_distance_l140_14075


namespace NUMINAMATH_CALUDE_days_for_C_alone_is_8_l140_14017

/-- The number of days it takes for C to finish the work alone, given that:
    - A, B, and C together can finish the work in 4 days
    - A alone can finish the work in 12 days
    - B alone can finish the work in 24 days
-/
def days_for_C_alone (days_together days_A_alone days_B_alone : ℚ) : ℚ :=
  let work_rate_together := 1 / days_together
  let work_rate_A := 1 / days_A_alone
  let work_rate_B := 1 / days_B_alone
  let work_rate_C := work_rate_together - work_rate_A - work_rate_B
  1 / work_rate_C

theorem days_for_C_alone_is_8 :
  days_for_C_alone 4 12 24 = 8 := by
  sorry

end NUMINAMATH_CALUDE_days_for_C_alone_is_8_l140_14017


namespace NUMINAMATH_CALUDE_negation_of_implication_l140_14045

theorem negation_of_implication (a b : ℝ) :
  ¬(a > b → 2*a > 2*b) ↔ (a ≤ b → 2*a ≤ 2*b) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l140_14045


namespace NUMINAMATH_CALUDE_jamies_flyer_delivery_l140_14066

/-- Jamie's flyer delivery problem -/
theorem jamies_flyer_delivery 
  (hourly_rate : ℝ) 
  (hours_per_delivery : ℝ) 
  (total_weeks : ℕ) 
  (total_earnings : ℝ) 
  (h1 : hourly_rate = 10)
  (h2 : hours_per_delivery = 3)
  (h3 : total_weeks = 6)
  (h4 : total_earnings = 360) : 
  (total_earnings / hourly_rate / total_weeks / hours_per_delivery : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_jamies_flyer_delivery_l140_14066


namespace NUMINAMATH_CALUDE_total_oil_leaked_equals_11687_l140_14031

/-- The amount of oil leaked before repairs, in liters -/
def oil_leaked_before : ℕ := 6522

/-- The amount of oil leaked during repairs, in liters -/
def oil_leaked_during : ℕ := 5165

/-- The total amount of oil leaked, in liters -/
def total_oil_leaked : ℕ := oil_leaked_before + oil_leaked_during

theorem total_oil_leaked_equals_11687 : total_oil_leaked = 11687 := by
  sorry

end NUMINAMATH_CALUDE_total_oil_leaked_equals_11687_l140_14031


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l140_14095

theorem binomial_expansion_properties :
  let n : ℕ := 15
  let last_three_sum := (n.choose (n-2)) + (n.choose (n-1)) + (n.choose n)
  let term (r : ℕ) := (n.choose r) * (3^r)
  ∃ (r₁ r₂ : ℕ),
    (last_three_sum = 121) ∧
    (∀ k, 0 ≤ k ∧ k ≤ n → (n.choose k) ≤ (n.choose r₁) ∧ (n.choose k) ≤ (n.choose r₂)) ∧
    (∀ k, 0 ≤ k ∧ k ≤ n → term k ≤ term r₁ ∧ term k ≤ term r₂) ∧
    r₁ = 11 ∧ r₂ = 12 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l140_14095


namespace NUMINAMATH_CALUDE_exactly_one_greater_than_one_l140_14089

theorem exactly_one_greater_than_one
  (a b c : ℝ)
  (pos_a : 0 < a)
  (pos_b : 0 < b)
  (pos_c : 0 < c)
  (prod_one : a * b * c = 1)
  (ineq : a + b + c > 1/a + 1/b + 1/c) :
  (a > 1 ∧ b ≤ 1 ∧ c ≤ 1) ∨
  (a ≤ 1 ∧ b > 1 ∧ c ≤ 1) ∨
  (a ≤ 1 ∧ b ≤ 1 ∧ c > 1) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_greater_than_one_l140_14089


namespace NUMINAMATH_CALUDE_modular_inverse_11_mod_1000_l140_14002

theorem modular_inverse_11_mod_1000 : ∃ x : ℕ, x < 1000 ∧ (11 * x) % 1000 = 1 :=
  by
  use 91
  sorry

end NUMINAMATH_CALUDE_modular_inverse_11_mod_1000_l140_14002


namespace NUMINAMATH_CALUDE_greatest_power_of_200_dividing_100_factorial_l140_14027

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem greatest_power_of_200_dividing_100_factorial :
  (∃ k : ℕ, 200^k ∣ factorial 100 ∧ ∀ m : ℕ, m > k → ¬(200^m ∣ factorial 100)) ∧
  (∀ k : ℕ, 200^k ∣ factorial 100 → k ≤ 12) ∧
  (200^12 ∣ factorial 100) :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_200_dividing_100_factorial_l140_14027


namespace NUMINAMATH_CALUDE_partner_investment_duration_l140_14067

/-- Given two partners P and Q with investments and profits, calculate Q's investment duration -/
theorem partner_investment_duration
  (investment_ratio_p investment_ratio_q : ℕ)
  (profit_ratio_p profit_ratio_q : ℕ)
  (p_duration : ℕ)
  (h_investment : investment_ratio_p = 7 ∧ investment_ratio_q = 5)
  (h_profit : profit_ratio_p = 7 ∧ profit_ratio_q = 14)
  (h_p_duration : p_duration = 5) :
  ∃ q_duration : ℕ,
    q_duration = 14 ∧
    (investment_ratio_p * p_duration) / (investment_ratio_q * q_duration) =
    profit_ratio_p / profit_ratio_q :=
by sorry

end NUMINAMATH_CALUDE_partner_investment_duration_l140_14067


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l140_14094

/-- Given a line y = mx + 5 intersecting the ellipse 9x^2 + 16y^2 = 144,
    prove that the possible slopes m satisfy m ∈ (-∞,-1] ∪ [1,∞). -/
theorem line_ellipse_intersection_slopes (m : ℝ) : 
  (∃ x y : ℝ, 9 * x^2 + 16 * y^2 = 144 ∧ y = m * x + 5) ↔ m ≤ -1 ∨ m ≥ 1 := by
  sorry

#check line_ellipse_intersection_slopes

end NUMINAMATH_CALUDE_line_ellipse_intersection_slopes_l140_14094


namespace NUMINAMATH_CALUDE_goldfish_red_balls_l140_14038

/-- Given a fish tank with goldfish and platyfish, prove the number of red balls each goldfish plays with -/
theorem goldfish_red_balls 
  (total_balls : ℕ) 
  (num_goldfish : ℕ) 
  (num_platyfish : ℕ) 
  (white_balls_per_platyfish : ℕ) 
  (h1 : total_balls = 80) 
  (h2 : num_goldfish = 3) 
  (h3 : num_platyfish = 10) 
  (h4 : white_balls_per_platyfish = 5) : 
  (total_balls - num_platyfish * white_balls_per_platyfish) / num_goldfish = 10 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_red_balls_l140_14038


namespace NUMINAMATH_CALUDE_union_implies_a_zero_l140_14049

theorem union_implies_a_zero (a : ℝ) : 
  let A : Set ℝ := {1, a^2}
  let B : Set ℝ := {a, -1}
  A ∪ B = {-1, a, 1} → a = 0 := by
sorry

end NUMINAMATH_CALUDE_union_implies_a_zero_l140_14049


namespace NUMINAMATH_CALUDE_unique_solutions_l140_14077

def factorial : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem unique_solutions (x y n : ℕ) : 
  (factorial x + factorial y) / factorial n = 3^n ↔ 
  ((x = 0 ∧ y = 2 ∧ n = 1) ∨ (x = 1 ∧ y = 2 ∧ n = 1)) :=
by sorry

end NUMINAMATH_CALUDE_unique_solutions_l140_14077


namespace NUMINAMATH_CALUDE_max_a_value_l140_14042

theorem max_a_value (e : ℝ) (h_e : e = Real.exp 1) : 
  (∃ (a : ℕ), a > 0 ∧ 
    (∀ t ∈ Set.Icc 1 a, ∃ m ∈ Set.Icc 0 5, e^t * (t^3 - 6*t^2 + 3*t + m) ≤ t) ∧
    (∀ a' > a, ∃ t ∈ Set.Icc 1 a', ∀ m ∈ Set.Icc 0 5, e^t * (t^3 - 6*t^2 + 3*t + m) > t)) ∧
  (∀ a : ℕ, a > 0 ∧ 
    (∀ t ∈ Set.Icc 1 a, ∃ m ∈ Set.Icc 0 5, e^t * (t^3 - 6*t^2 + 3*t + m) ≤ t) →
    a ≤ 5) :=
by sorry

end NUMINAMATH_CALUDE_max_a_value_l140_14042


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l140_14036

/-- The line kx + 3y + k - 9 = 0 passes through the point (-1, 3) for all values of k -/
theorem line_passes_through_fixed_point (k : ℝ) : k * (-1) + 3 * 3 + k - 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l140_14036


namespace NUMINAMATH_CALUDE_theater_seats_tom_wants_500_seats_l140_14005

/-- Calculates the number of seats in Tom's theater based on given conditions --/
theorem theater_seats (cost_per_sqft : ℝ) (sqft_per_seat : ℝ) (partner_share : ℝ) (tom_spend : ℝ) : ℝ :=
  let cost_per_seat := cost_per_sqft * sqft_per_seat
  let total_cost := tom_spend / (1 - partner_share)
  total_cost / (3 * cost_per_seat)

/-- Proves that Tom wants 500 seats in his theater --/
theorem tom_wants_500_seats :
  theater_seats 5 12 0.4 54000 = 500 := by
  sorry

end NUMINAMATH_CALUDE_theater_seats_tom_wants_500_seats_l140_14005


namespace NUMINAMATH_CALUDE_large_pizza_slices_correct_l140_14046

/-- The number of slices a small pizza gives -/
def small_pizza_slices : ℕ := 4

/-- The number of small pizzas purchased -/
def small_pizzas_bought : ℕ := 3

/-- The number of large pizzas purchased -/
def large_pizzas_bought : ℕ := 2

/-- The number of slices George eats -/
def george_slices : ℕ := 3

/-- The number of slices Bob eats -/
def bob_slices : ℕ := george_slices + 1

/-- The number of slices Susie eats -/
def susie_slices : ℕ := bob_slices / 2

/-- The number of slices Bill, Fred, and Mark each eat -/
def others_slices : ℕ := 3

/-- The number of slices left over -/
def leftover_slices : ℕ := 10

/-- The number of slices a large pizza gives -/
def large_pizza_slices : ℕ := 8

theorem large_pizza_slices_correct : 
  small_pizza_slices * small_pizzas_bought + large_pizza_slices * large_pizzas_bought = 
  george_slices + bob_slices + susie_slices + 3 * others_slices + leftover_slices :=
by sorry

end NUMINAMATH_CALUDE_large_pizza_slices_correct_l140_14046


namespace NUMINAMATH_CALUDE_symmetric_point_and_line_l140_14081

-- Define the point A
def A : ℝ × ℝ := (0, 1)

-- Define the line l₁
def l₁ (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the line l₂
def l₂ (x y : ℝ) : Prop := x - 2*y + 2 = 0

-- Define the symmetric point B
def B : ℝ × ℝ := (2, -1)

-- Define the symmetric line l
def l (x y : ℝ) : Prop := 2*x - y - 5 = 0

-- Theorem statement
theorem symmetric_point_and_line :
  (∀ x y : ℝ, l₁ x y ↔ x - y - 1 = 0) ∧ 
  (∀ x y : ℝ, l₂ x y ↔ x - 2*y + 2 = 0) →
  (B = (2, -1) ∧ (∀ x y : ℝ, l x y ↔ 2*x - y - 5 = 0)) :=
sorry

end NUMINAMATH_CALUDE_symmetric_point_and_line_l140_14081


namespace NUMINAMATH_CALUDE_quadratic_factorization_l140_14084

theorem quadratic_factorization (p q : ℤ) :
  (∀ x, 20 * x^2 - 110 * x - 120 = (5 * x + p) * (4 * x + q)) →
  p + 2 * q = -8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l140_14084


namespace NUMINAMATH_CALUDE_bread_distribution_l140_14029

theorem bread_distribution (a d : ℚ) : 
  d > 0 ∧ 
  (a - 2*d) + (a - d) + a + (a + d) + (a + 2*d) = 100 ∧ 
  (a + (a + d) + (a + 2*d)) = (1/7) * ((a - 2*d) + (a - d)) →
  a - 2*d = 5/3 := by
sorry

end NUMINAMATH_CALUDE_bread_distribution_l140_14029


namespace NUMINAMATH_CALUDE_line_transformation_l140_14061

/-- Given a line ax + y - 7 = 0 transformed by matrix A to 9x + y - 91 = 0, prove a = 2 and b = 13 -/
theorem line_transformation (a b : ℝ) :
  (∀ x y : ℝ, a * x + y - 7 = 0 →
    ∃ x' y' : ℝ, x' = 3 * x ∧ y' = -x + b * y ∧ 9 * x' + y' - 91 = 0) →
  a = 2 ∧ b = 13 := by
  sorry


end NUMINAMATH_CALUDE_line_transformation_l140_14061


namespace NUMINAMATH_CALUDE_jordans_sister_jars_l140_14009

def total_plums : ℕ := 240
def ripe_ratio : ℚ := 1/4
def unripe_ratio : ℚ := 3/4
def kept_unripe : ℕ := 46
def plums_per_mango : ℕ := 7
def mangoes_per_jar : ℕ := 5

theorem jordans_sister_jars : 
  ⌊(total_plums * unripe_ratio - kept_unripe + total_plums * ripe_ratio) / plums_per_mango / mangoes_per_jar⌋ = 5 := by
  sorry

end NUMINAMATH_CALUDE_jordans_sister_jars_l140_14009


namespace NUMINAMATH_CALUDE_job_interviews_comprehensive_l140_14099

/-- Represents a scenario that may or may not require comprehensive investigation. -/
inductive Scenario
| AirQuality
| VisionStatus
| JobInterviews
| FishCount

/-- Determines if a scenario requires comprehensive investigation. -/
def requiresComprehensiveInvestigation (s : Scenario) : Prop :=
  match s with
  | Scenario.JobInterviews => True
  | _ => False

/-- Theorem stating that job interviews is the only scenario requiring comprehensive investigation. -/
theorem job_interviews_comprehensive :
  ∀ s : Scenario, requiresComprehensiveInvestigation s ↔ s = Scenario.JobInterviews :=
by sorry

end NUMINAMATH_CALUDE_job_interviews_comprehensive_l140_14099


namespace NUMINAMATH_CALUDE_divide_fractions_l140_14041

theorem divide_fractions : (3 : ℚ) / 4 / ((7 : ℚ) / 8) = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_divide_fractions_l140_14041


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l140_14047

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation (1-i)z = 2i
def equation (z : ℂ) : Prop := (1 - i) * z = 2 * i

-- Define what it means for a complex number to be in the second quadrant
def in_second_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im > 0

-- State the theorem
theorem z_in_second_quadrant :
  ∃ z : ℂ, equation z ∧ in_second_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l140_14047


namespace NUMINAMATH_CALUDE_negation_of_implication_l140_14057

theorem negation_of_implication (a b : ℝ) :
  ¬(a^2 > b^2 → a > b) ↔ (a^2 ≤ b^2 → a ≤ b) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l140_14057


namespace NUMINAMATH_CALUDE_equation_solution_l140_14085

theorem equation_solution :
  let f (x : ℝ) := 5 * (x^2)^2 + 3 * x^2 + 2 - 4 * (4 * x^2 + x^2 + 1)
  ∀ x : ℝ, f x = 0 ↔ x = Real.sqrt ((17 + Real.sqrt 329) / 10) ∨ x = -Real.sqrt ((17 + Real.sqrt 329) / 10) :=
by
  sorry


end NUMINAMATH_CALUDE_equation_solution_l140_14085


namespace NUMINAMATH_CALUDE_prob_exactly_one_of_two_independent_l140_14048

/-- The probability of exactly one of two independent events occurring -/
theorem prob_exactly_one_of_two_independent (p₁ p₂ : ℝ) 
  (h₁ : 0 ≤ p₁ ∧ p₁ ≤ 1) (h₂ : 0 ≤ p₂ ∧ p₂ ≤ 1) : 
  p₁ * (1 - p₂) + p₂ * (1 - p₁) = 
  (p₁ + p₂) - (p₁ * p₂) := by sorry

end NUMINAMATH_CALUDE_prob_exactly_one_of_two_independent_l140_14048


namespace NUMINAMATH_CALUDE_average_age_of_new_joiners_l140_14016

/-- Given a group of people going for a picnic, prove the average age of new joiners -/
theorem average_age_of_new_joiners
  (initial_count : ℕ)
  (initial_avg_age : ℝ)
  (new_count : ℕ)
  (new_total_avg_age : ℝ)
  (h1 : initial_count = 12)
  (h2 : initial_avg_age = 16)
  (h3 : new_count = 12)
  (h4 : new_total_avg_age = 15.5) :
  let total_count := initial_count + new_count
  let new_joiners_avg_age := (total_count * new_total_avg_age - initial_count * initial_avg_age) / new_count
  new_joiners_avg_age = 15 := by
sorry

end NUMINAMATH_CALUDE_average_age_of_new_joiners_l140_14016


namespace NUMINAMATH_CALUDE_system_solution_l140_14033

theorem system_solution : 
  ∃! (x y : ℚ), (4 * x - 3 * y = -17) ∧ (5 * x + 6 * y = -4) ∧ 
  (x = -74/13) ∧ (y = -25/13) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l140_14033


namespace NUMINAMATH_CALUDE_simone_finish_time_l140_14055

-- Define the start time
def start_time : Nat := 8 * 60  -- 8:00 AM in minutes since midnight

-- Define the duration of the first two tasks
def first_two_tasks_duration : Nat := 2 * 45

-- Define the break duration
def break_duration : Nat := 15

-- Define the duration of the third task
def third_task_duration : Nat := 2 * 45

-- Define the total duration
def total_duration : Nat := first_two_tasks_duration + break_duration + third_task_duration

-- Define the finish time in minutes since midnight
def finish_time : Nat := start_time + total_duration

-- Theorem to prove
theorem simone_finish_time : 
  finish_time = 11 * 60 + 15  -- 11:15 AM in minutes since midnight
  := by sorry

end NUMINAMATH_CALUDE_simone_finish_time_l140_14055


namespace NUMINAMATH_CALUDE_first_player_wins_or_draws_l140_14062

/-- Represents a game where two players take turns picking bills from a sequence. -/
structure BillGame where
  n : ℕ
  bills : List ℕ
  turn : ℕ

/-- Represents a move in the game, either taking from the left or right end. -/
inductive Move
  | Left
  | Right

/-- Represents the result of the game. -/
inductive GameResult
  | FirstPlayerWins
  | SecondPlayerWins
  | Draw

/-- Defines an optimal strategy for the first player. -/
def optimalStrategy : BillGame → Move
  | _ => sorry

/-- Simulates the game with both players following the optimal strategy. -/
def playGame : BillGame → GameResult
  | _ => sorry

/-- Theorem stating that the first player can always ensure a win or draw. -/
theorem first_player_wins_or_draws (n : ℕ) :
  ∀ (game : BillGame),
    game.n = n ∧
    game.bills = List.range (2*n) ∧
    game.turn = 0 →
    playGame game ≠ GameResult.SecondPlayerWins :=
  sorry

end NUMINAMATH_CALUDE_first_player_wins_or_draws_l140_14062


namespace NUMINAMATH_CALUDE_solution_set_f_max_m_value_l140_14052

-- Define the functions f and g
def f (x : ℝ) : ℝ := |x - 3| - 5
def g (x : ℝ) : ℝ := |x + 2| - 2

-- Theorem for the solution set of f(x) ≤ 2
theorem solution_set_f (x : ℝ) : f x ≤ 2 ↔ -4 ≤ x ∧ x ≤ 10 := by sorry

-- Theorem for the maximum value of m
theorem max_m_value : 
  ∃ (m : ℝ), (∃ (x : ℝ), f x - g x ≥ m - 3) ∧ 
  ∀ (m' : ℝ), m' > m → ¬∃ (x : ℝ), f x - g x ≥ m' - 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_max_m_value_l140_14052


namespace NUMINAMATH_CALUDE_triangle_third_side_l140_14097

theorem triangle_third_side (a b : ℝ) (h1 : a = 3.14) (h2 : b = 0.67) : 
  ∃ c : ℕ, c = 3 ∧ 
    a + b > c ∧
    a + c > b ∧
    b + c > a := by
  sorry

end NUMINAMATH_CALUDE_triangle_third_side_l140_14097


namespace NUMINAMATH_CALUDE_solution_difference_l140_14007

/-- The quadratic equation from the problem -/
def quadratic_equation (x : ℝ) : Prop :=
  x^2 - 3*x + 9 = x + 41

/-- The two solutions of the quadratic equation -/
def solutions : Set ℝ :=
  {x : ℝ | quadratic_equation x}

/-- Theorem stating that the positive difference between the two solutions is 12 -/
theorem solution_difference : 
  ∃ (x y : ℝ), x ∈ solutions ∧ y ∈ solutions ∧ x ≠ y ∧ |x - y| = 12 :=
sorry

end NUMINAMATH_CALUDE_solution_difference_l140_14007


namespace NUMINAMATH_CALUDE_new_person_weight_l140_14080

theorem new_person_weight (n : Nat) (original_weight replaced_weight increase : ℝ) :
  n = 8 ∧ 
  replaced_weight = 50 ∧ 
  increase = 2.5 →
  (n : ℝ) * increase + replaced_weight = 70 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l140_14080


namespace NUMINAMATH_CALUDE_jenny_distance_difference_l140_14037

theorem jenny_distance_difference : 
  ∀ (run_distance walk_distance : ℝ),
    run_distance = 0.6 →
    walk_distance = 0.4 →
    run_distance - walk_distance = 0.2 :=
by
  sorry

end NUMINAMATH_CALUDE_jenny_distance_difference_l140_14037


namespace NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l140_14014

theorem smallest_solution_quartic_equation :
  ∃ (x : ℝ), x^4 - 50*x^2 + 625 = 0 ∧ 
  (∀ y : ℝ, y^4 - 50*y^2 + 625 = 0 → y ≥ x) ∧
  x = -5 := by
sorry

end NUMINAMATH_CALUDE_smallest_solution_quartic_equation_l140_14014


namespace NUMINAMATH_CALUDE_divisibility_by_24_l140_14091

theorem divisibility_by_24 (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) :
  24 ∣ (p^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_24_l140_14091


namespace NUMINAMATH_CALUDE_greatest_divisor_XYXY_l140_14070

/-- A four-digit palindrome of the pattern XYXY -/
def XYXY (X Y : Nat) : Nat := 1000 * X + 100 * Y + 10 * X + Y

/-- Predicate to check if a number is a single digit -/
def is_single_digit (n : Nat) : Prop := n ≥ 0 ∧ n ≤ 9

/-- The theorem stating that 11 is the greatest divisor of all XYXY palindromes -/
theorem greatest_divisor_XYXY :
  ∀ X Y : Nat, is_single_digit X → is_single_digit Y →
  (∀ d : Nat, d > 11 → ¬(d ∣ XYXY X Y)) ∧
  (11 ∣ XYXY X Y) :=
sorry

end NUMINAMATH_CALUDE_greatest_divisor_XYXY_l140_14070


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_product_l140_14023

theorem consecutive_even_numbers_product : 
  ∃! (a b c : ℕ), 
    (b = a + 2 ∧ c = b + 2) ∧ 
    (a % 2 = 0) ∧
    (800000 ≤ a * b * c) ∧ 
    (a * b * c < 900000) ∧
    (a * b * c % 10 = 2) ∧
    (a = 94 ∧ b = 96 ∧ c = 98) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_product_l140_14023


namespace NUMINAMATH_CALUDE_pages_ratio_l140_14035

theorem pages_ratio (lana_initial : ℕ) (duane_initial : ℕ) (lana_final : ℕ)
  (h1 : lana_initial = 8)
  (h2 : duane_initial = 42)
  (h3 : lana_final = 29) :
  (lana_final - lana_initial) * 2 = duane_initial :=
by sorry

end NUMINAMATH_CALUDE_pages_ratio_l140_14035


namespace NUMINAMATH_CALUDE_octal_subtraction_l140_14019

/-- Converts a base 8 number represented as a list of digits to a natural number -/
def octalToNat (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => 8 * acc + d) 0

/-- Converts a natural number to its base 8 representation as a list of digits -/
def natToOctal (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec go (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else go (m / 8) ((m % 8) :: acc)
    go n []

theorem octal_subtraction :
  let a := [1, 3, 5, 2]
  let b := [0, 6, 7, 4]
  let result := [1, 4, 5, 6]
  octalToNat a - octalToNat b = octalToNat result := by
  sorry

end NUMINAMATH_CALUDE_octal_subtraction_l140_14019


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l140_14056

theorem complex_arithmetic_equality : 
  908 * 501 - (731 * 1389 - (547 * 236 + 842 * 731 - 495 * 361)) = 5448 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l140_14056


namespace NUMINAMATH_CALUDE_special_collection_loans_l140_14030

theorem special_collection_loans (initial_count : ℕ) (return_rate : ℚ) (final_count : ℕ) 
  (h1 : initial_count = 75)
  (h2 : return_rate = 70 / 100)
  (h3 : final_count = 60) :
  ∃ (loaned_out : ℕ), loaned_out = 50 ∧ 
    initial_count - (1 - return_rate) * loaned_out = final_count :=
by sorry

end NUMINAMATH_CALUDE_special_collection_loans_l140_14030


namespace NUMINAMATH_CALUDE_largest_box_volume_l140_14025

/-- The volume of the largest rectangular parallelopiped that can be enclosed in a cylindrical container with a hemispherical lid. -/
theorem largest_box_volume (total_height radius : ℝ) (h_total_height : total_height = 60) (h_radius : radius = 30) :
  let cylinder_height : ℝ := total_height - radius
  let box_base_side : ℝ := 2 * radius
  let box_height : ℝ := cylinder_height
  let box_volume : ℝ := box_base_side^2 * box_height
  box_volume = 108000 := by
  sorry

#check largest_box_volume

end NUMINAMATH_CALUDE_largest_box_volume_l140_14025


namespace NUMINAMATH_CALUDE_problem_statement_l140_14064

theorem problem_statement (x y : ℝ) (h : (x + y - 2020) * (2023 - x - y) = 2) :
  (x + y - 2020)^2 * (2023 - x - y)^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l140_14064


namespace NUMINAMATH_CALUDE_quadratic_root_sum_product_l140_14068

theorem quadratic_root_sum_product (x₁ x₂ : ℝ) : 
  x₁^2 + 3*x₁ - 2023 = 0 →
  x₂^2 + 3*x₂ - 2023 = 0 →
  x₁^2 * x₂ + x₁ * x₂^2 = 6069 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_product_l140_14068


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l140_14065

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := x = y^2 / 10 + 5 / 2

/-- The focus of the parabola -/
def parabola_focus : ℝ × ℝ := (5, 0)

/-- The directrix of the parabola is the x-axis -/
def parabola_directrix (x : ℝ) : ℝ × ℝ := (x, 0)

theorem hyperbola_parabola_intersection :
  ∃ (x₁ y₁ x₂ y₂ : ℝ), 
    hyperbola x₁ y₁ ∧ parabola x₁ y₁ ∧
    hyperbola x₂ y₂ ∧ parabola x₂ y₂ ∧
    (x₁, y₁) ≠ (x₂, y₂) :=
  sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l140_14065


namespace NUMINAMATH_CALUDE_gcf_of_75_and_105_l140_14020

theorem gcf_of_75_and_105 : Nat.gcd 75 105 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_105_l140_14020


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l140_14098

theorem hot_dogs_remainder : 16789537 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l140_14098


namespace NUMINAMATH_CALUDE_append_nine_to_two_digit_number_l140_14073

/-- Given a two-digit number with tens digit t and units digit u,
    appending 9 to the right results in 100t + 10u + 9 -/
theorem append_nine_to_two_digit_number (t u : ℕ) 
  (h1 : t ≥ 1 ∧ t ≤ 9) (h2 : u ≥ 0 ∧ u ≤ 9) :
  (10 * t + u) * 10 + 9 = 100 * t + 10 * u + 9 := by
  sorry


end NUMINAMATH_CALUDE_append_nine_to_two_digit_number_l140_14073


namespace NUMINAMATH_CALUDE_square_divisibility_l140_14051

theorem square_divisibility (a b : ℕ+) (h : (a * b + 1) ∣ (a ^ 2 + b ^ 2)) :
  ∃ k : ℕ, (a ^ 2 + b ^ 2) / (a * b + 1) = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_divisibility_l140_14051


namespace NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l140_14088

theorem min_value_expression (a b c : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ 5) :
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 ≥ 2 * Real.sqrt 5 - 4 * 5^(1/4) + 4 :=
by sorry

theorem min_value_achievable :
  ∃ a b c : ℝ, 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ c ≤ 5 ∧
  (a - 1)^2 + (b/a - 1)^2 + (c/b - 1)^2 + (5/c - 1)^2 = 2 * Real.sqrt 5 - 4 * 5^(1/4) + 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_min_value_achievable_l140_14088


namespace NUMINAMATH_CALUDE_sufficiency_not_necessity_l140_14044

theorem sufficiency_not_necessity (a b : ℝ) :
  (a < b ∧ b < 0 → 1 / a > 1 / b) ∧
  ∃ a b : ℝ, 1 / a > 1 / b ∧ ¬(a < b ∧ b < 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficiency_not_necessity_l140_14044


namespace NUMINAMATH_CALUDE_sum_289_37_base4_l140_14060

/-- Converts a natural number to its base 4 representation as a list of digits -/
def toBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- Checks if a list of natural numbers represents a valid base 4 number -/
def isValidBase4 (l : List ℕ) : Prop :=
  ∀ d ∈ l, d < 4

theorem sum_289_37_base4 :
  let sum := 289 + 37
  let base4Sum := toBase4 sum
  isValidBase4 base4Sum ∧ base4Sum = [1, 1, 0, 1, 2] := by sorry

end NUMINAMATH_CALUDE_sum_289_37_base4_l140_14060


namespace NUMINAMATH_CALUDE_restaurant_bill_proof_l140_14059

theorem restaurant_bill_proof : 
  ∀ (total_friends : ℕ) (paying_friends : ℕ) (extra_payment : ℚ),
    total_friends = 12 →
    paying_friends = 10 →
    extra_payment = 3 →
    ∃ (bill : ℚ), 
      bill = paying_friends * (bill / total_friends + extra_payment) ∧
      bill = 180 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_proof_l140_14059


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l140_14022

theorem fractional_equation_solution :
  ∃ x : ℝ, (x * (x - 2) ≠ 0) ∧ (4 / (x - 2) = 2 / x) ∧ (x = -2) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l140_14022


namespace NUMINAMATH_CALUDE_worm_length_difference_l140_14034

def worm_lengths : List ℝ := [0.8, 0.1, 1.2, 0.4, 0.7]

theorem worm_length_difference : 
  let max_length := worm_lengths.maximum?
  let min_length := worm_lengths.minimum?
  ∀ max min, max_length = some max → min_length = some min →
    max - min = 1.1 := by sorry

end NUMINAMATH_CALUDE_worm_length_difference_l140_14034


namespace NUMINAMATH_CALUDE_product_sum_of_digits_l140_14032

def repeat_digit (d : Nat) (n : Nat) : Nat :=
  d * ((10^n - 1) / 9)

def sum_of_digits (n : Nat) : Nat :=
  if n = 0 then 0 else n % 10 + sum_of_digits (n / 10)

theorem product_sum_of_digits :
  sum_of_digits (repeat_digit 4 2012 * repeat_digit 9 2012) = 18108 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_of_digits_l140_14032


namespace NUMINAMATH_CALUDE_average_weight_problem_l140_14040

theorem average_weight_problem (a b c : ℝ) : 
  (a + b) / 2 = 40 →
  (b + c) / 2 = 41 →
  b = 27 →
  (a + b + c) / 3 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_weight_problem_l140_14040


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l140_14028

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

-- Define the conditions of the problem
def problem_conditions (a : ℕ → ℝ) (n : ℕ) : Prop :=
  geometric_sequence a ∧
  a 1 * a 2 * a 3 = 4 ∧
  a 4 * a 5 * a 6 = 12 ∧
  a (n - 1) * a n * a (n + 1) = 324

-- Theorem statement
theorem geometric_sequence_problem (a : ℕ → ℝ) (n : ℕ) :
  problem_conditions a n → n = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l140_14028


namespace NUMINAMATH_CALUDE_abs_sum_inequality_l140_14054

theorem abs_sum_inequality (k : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 3| > k) ↔ k < 4 :=
sorry

end NUMINAMATH_CALUDE_abs_sum_inequality_l140_14054


namespace NUMINAMATH_CALUDE_max_profit_l140_14078

/-- Profit function for price greater than 120 yuan -/
def profit_above (x : ℝ) : ℝ := -10 * x^2 + 2500 * x - 150000

/-- Profit function for price between 100 and 120 yuan -/
def profit_below (x : ℝ) : ℝ := -30 * x^2 + 6900 * x - 390000

/-- The maximum profit occurs at 115 yuan and equals 6750 yuan -/
theorem max_profit :
  ∃ (x : ℝ), x = 115 ∧ 
  profit_below x = 6750 ∧
  ∀ (y : ℝ), y > 100 → profit_above y ≤ profit_below x ∧ profit_below y ≤ profit_below x :=
sorry

end NUMINAMATH_CALUDE_max_profit_l140_14078


namespace NUMINAMATH_CALUDE_odd_function_properties_l140_14021

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem odd_function_properties (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = 1 ∧
   ∀ x, f a x = 1 - 2 / (2^x + 1) ∧
   StrictMono (f a)) :=
by sorry

end NUMINAMATH_CALUDE_odd_function_properties_l140_14021


namespace NUMINAMATH_CALUDE_math_quiz_items_l140_14063

theorem math_quiz_items (score_percentage : ℝ) (mistakes : ℕ) (total_items : ℕ) : 
  score_percentage = 0.80 → 
  mistakes = 5 → 
  (total_items - mistakes : ℝ) / total_items = score_percentage → 
  total_items = 25 := by
sorry

end NUMINAMATH_CALUDE_math_quiz_items_l140_14063
