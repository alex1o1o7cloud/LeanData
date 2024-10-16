import Mathlib

namespace NUMINAMATH_CALUDE_total_pens_l1841_184112

def red_pens : ℕ := 65
def blue_pens : ℕ := 45
def black_pens : ℕ := 58

theorem total_pens : red_pens + blue_pens + black_pens = 168 := by
  sorry

end NUMINAMATH_CALUDE_total_pens_l1841_184112


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1841_184186

theorem sqrt_equation_solution (x y : ℝ) :
  Real.sqrt (x^2 + y^2 - 1) = x + y - 1 ↔ (x = 1 ∧ y ≥ 0) ∨ (y = 1 ∧ x ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1841_184186


namespace NUMINAMATH_CALUDE_product_calculation_l1841_184128

theorem product_calculation : 200 * 19.9 * 1.99 * 100 = 791620 := by
  sorry

end NUMINAMATH_CALUDE_product_calculation_l1841_184128


namespace NUMINAMATH_CALUDE_total_luggage_pieces_is_142_l1841_184103

/-- Represents the different classes of service on an international flight. -/
inductive ServiceClass
  | Economy
  | Business
  | FirstClass

/-- Returns the number of luggage pieces allowed for a given service class. -/
def luggageAllowance (c : ServiceClass) : ℕ :=
  match c with
  | ServiceClass.Economy => 5
  | ServiceClass.Business => 8
  | ServiceClass.FirstClass => 12

/-- Calculates the total number of luggage pieces for a given service class and number of passengers. -/
def totalLuggageForClass (c : ServiceClass) (passengers : ℕ) : ℕ :=
  (luggageAllowance c) * passengers

/-- Theorem: The total number of luggage pieces allowed onboard is 142. -/
theorem total_luggage_pieces_is_142 :
  (totalLuggageForClass ServiceClass.Economy 10) +
  (totalLuggageForClass ServiceClass.Business 7) +
  (totalLuggageForClass ServiceClass.FirstClass 3) = 142 := by
  sorry

end NUMINAMATH_CALUDE_total_luggage_pieces_is_142_l1841_184103


namespace NUMINAMATH_CALUDE_min_distance_to_locus_l1841_184166

open Complex

theorem min_distance_to_locus (z : ℂ) :
  (abs (z - 1) = abs (z + 2*I)) →
  ∃ min_val : ℝ, (min_val = (9 * Real.sqrt 5) / 10) ∧
  (∀ w : ℂ, abs (z - 1) = abs (z + 2*I) → abs (w - 1 - I) ≥ min_val) ∧
  (∃ z₀ : ℂ, abs (z₀ - 1) = abs (z₀ + 2*I) ∧ abs (z₀ - 1 - I) = min_val) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_locus_l1841_184166


namespace NUMINAMATH_CALUDE_dividend_calculation_l1841_184108

theorem dividend_calculation (q d r x : ℝ) 
  (hq : q = -427.86)
  (hd : d = 52.7)
  (hr : r = -14.5)
  (hx : x = q * d + r) : 
  x = -22571.002 := by
sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1841_184108


namespace NUMINAMATH_CALUDE_square_difference_identity_l1841_184193

theorem square_difference_identity : 287 * 287 + 269 * 269 - 2 * 287 * 269 = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l1841_184193


namespace NUMINAMATH_CALUDE_five_spiders_make_five_webs_l1841_184172

/-- The number of webs made by a given number of spiders in 5 days -/
def webs_made (num_spiders : ℕ) : ℕ :=
  num_spiders * 1

/-- Theorem stating that 5 spiders make 5 webs in 5 days -/
theorem five_spiders_make_five_webs :
  webs_made 5 = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_spiders_make_five_webs_l1841_184172


namespace NUMINAMATH_CALUDE_complex_magnitude_l1841_184194

theorem complex_magnitude (a b : ℝ) (h : (1 + 2*a*Complex.I) * Complex.I = 1 - b*Complex.I) :
  Complex.abs (a + b*Complex.I) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_l1841_184194


namespace NUMINAMATH_CALUDE_three_digit_congruence_solutions_l1841_184109

theorem three_digit_congruence_solutions : 
  let count := Finset.filter (fun x => 100 ≤ x ∧ x ≤ 999 ∧ (4573 * x + 502) % 23 = 1307 % 23) (Finset.range 1000)
  Finset.card count = 39 := by
  sorry

end NUMINAMATH_CALUDE_three_digit_congruence_solutions_l1841_184109


namespace NUMINAMATH_CALUDE_half_dollars_in_tip_jar_l1841_184187

def nickel_value : ℚ := 5 / 100
def dime_value : ℚ := 10 / 100
def half_dollar_value : ℚ := 50 / 100

def nickels_shining : ℕ := 3
def dimes_shining : ℕ := 13
def dimes_tip : ℕ := 7
def total_amount : ℚ := 665 / 100

theorem half_dollars_in_tip_jar :
  ∃ (half_dollars : ℕ),
    (nickels_shining : ℚ) * nickel_value +
    (dimes_shining : ℚ) * dime_value +
    (dimes_tip : ℚ) * dime_value +
    (half_dollars : ℚ) * half_dollar_value = total_amount ∧
    half_dollars = 9 :=
by sorry

end NUMINAMATH_CALUDE_half_dollars_in_tip_jar_l1841_184187


namespace NUMINAMATH_CALUDE_sphere_quarter_sphere_radius_l1841_184159

theorem sphere_quarter_sphere_radius (r : ℝ) (h : r = 4 * Real.rpow 2 (1/3)) :
  ∃ R : ℝ, (4/3 * Real.pi * R^3 = 1/3 * Real.pi * r^3) ∧ R = 2 * Real.rpow 2 (1/3) :=
sorry

end NUMINAMATH_CALUDE_sphere_quarter_sphere_radius_l1841_184159


namespace NUMINAMATH_CALUDE_bottle_and_beverage_weight_l1841_184158

/-- Given a bottle and some beverage, prove the weight of the original beverage and the bottle. -/
theorem bottle_and_beverage_weight 
  (original_beverage : ℝ) 
  (bottle : ℝ) 
  (h1 : 2 * original_beverage + bottle = 5) 
  (h2 : 4 * original_beverage + bottle = 9) : 
  original_beverage = 2 ∧ bottle = 1 := by
sorry

end NUMINAMATH_CALUDE_bottle_and_beverage_weight_l1841_184158


namespace NUMINAMATH_CALUDE_sum_squared_equals_129_l1841_184113

theorem sum_squared_equals_129 (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a^2 + a*b + b^2 = 25)
  (h2 : b^2 + b*c + c^2 = 49)
  (h3 : c^2 + c*a + a^2 = 64) :
  (a + b + c)^2 = 129 := by
  sorry

end NUMINAMATH_CALUDE_sum_squared_equals_129_l1841_184113


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l1841_184157

theorem simplify_fraction_product :
  (1 : ℝ) / (1 + Real.sqrt 3) * (1 / (1 - Real.sqrt 5)) = 1 / (1 - Real.sqrt 5 + Real.sqrt 3 - Real.sqrt 15) := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l1841_184157


namespace NUMINAMATH_CALUDE_tank_filling_ratio_l1841_184184

/-- Proves that the ratio of time B works alone to total time is 0.5 -/
theorem tank_filling_ratio : 
  ∀ (t_A t_B t_total : ℝ),
  t_A > 0 → t_B > 0 → t_total > 0 →
  (1 / t_A + 1 / t_B = 1 / 24) →
  t_total = 29.999999999999993 →
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ t_total ∧ 
    t / t_B + (t_total - t) / 24 = 1) →
  ∃ t : ℝ, t / t_total = 0.5 := by
sorry

end NUMINAMATH_CALUDE_tank_filling_ratio_l1841_184184


namespace NUMINAMATH_CALUDE_ratio_and_equation_imply_c_value_l1841_184190

theorem ratio_and_equation_imply_c_value 
  (a b c : ℝ) 
  (h1 : ∃ (k : ℝ), a = 2*k ∧ b = 3*k ∧ c = 7*k) 
  (h2 : a - b + 3 = c - 2*b) : 
  c = 21/2 := by
sorry

end NUMINAMATH_CALUDE_ratio_and_equation_imply_c_value_l1841_184190


namespace NUMINAMATH_CALUDE_range_of_a_l1841_184185

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_increasing : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

-- State the theorem
theorem range_of_a (a : ℝ) (h : f a ≤ f 2) : a ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1841_184185


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l1841_184125

open Set

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 3*x - 4 ≤ 0}
def N : Set ℝ := {x | Real.log x ≥ 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x | 1 ≤ x ∧ x ≤ 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l1841_184125


namespace NUMINAMATH_CALUDE_no_solution_for_specific_k_l1841_184198

theorem no_solution_for_specific_k (p : ℕ) (hp : Prime p) (hp_mod : p % 4 = 3) :
  ¬ ∃ (n m : ℕ+), (n.val^2 + m.val^2 : ℚ) / (m.val^4 + n.val) = p^2 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_specific_k_l1841_184198


namespace NUMINAMATH_CALUDE_tree_planting_l1841_184133

theorem tree_planting (path_length : ℕ) (tree_distance : ℕ) (total_trees : ℕ) : 
  path_length = 50 →
  tree_distance = 2 →
  total_trees = 2 * (path_length / tree_distance + 1) →
  total_trees = 52 := by
sorry

end NUMINAMATH_CALUDE_tree_planting_l1841_184133


namespace NUMINAMATH_CALUDE_event_ratio_l1841_184100

theorem event_ratio (total : ℕ) (children : ℕ) (adults : ℕ) : 
  total = 42 → children = 28 → adults = total - children → 
  (children : ℚ) / (adults : ℚ) = 2 / 1 := by
  sorry

end NUMINAMATH_CALUDE_event_ratio_l1841_184100


namespace NUMINAMATH_CALUDE_no_line_exists_l1841_184199

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line passing through the focus
def line_through_focus (m : ℝ) (x y : ℝ) : Prop := x = m*y + 1

-- Define the intersection points of the line and the parabola
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ parabola x y ∧ line_through_focus m x y}

-- Define the distance from a point to the line x = -2
def distance_to_line (x y : ℝ) : ℝ := x + 2

-- Statement to prove
theorem no_line_exists :
  ¬ ∃ m : ℝ, ∃ A B : ℝ × ℝ,
    A ∈ intersection_points m ∧
    B ∈ intersection_points m ∧
    A ≠ B ∧
    distance_to_line A.1 A.2 + distance_to_line B.1 B.2 = 5 :=
sorry

end NUMINAMATH_CALUDE_no_line_exists_l1841_184199


namespace NUMINAMATH_CALUDE_rectangle_24_60_parts_l1841_184192

/-- The number of parts a rectangle is divided into when split into unit squares and its diagonal is drawn -/
def rectangle_parts (width : ℕ) (length : ℕ) : ℕ :=
  width * length + width + length - Nat.gcd width length

/-- Theorem stating that a 24 × 60 rectangle divided into unit squares and with its diagonal drawn is divided into 1512 parts -/
theorem rectangle_24_60_parts :
  rectangle_parts 24 60 = 1512 := by
  sorry

#eval rectangle_parts 24 60

end NUMINAMATH_CALUDE_rectangle_24_60_parts_l1841_184192


namespace NUMINAMATH_CALUDE_johns_butterfly_jars_l1841_184121

/-- The number of caterpillars in each jar -/
def caterpillars_per_jar : ℕ := 10

/-- The percentage of caterpillars that fail to become butterflies -/
def failure_rate : ℚ := 40 / 100

/-- The price of each butterfly in dollars -/
def price_per_butterfly : ℕ := 3

/-- The total amount made from selling butterflies in dollars -/
def total_amount : ℕ := 72

/-- The number of jars John has -/
def number_of_jars : ℕ := 4

theorem johns_butterfly_jars :
  let butterflies_per_jar := caterpillars_per_jar * (1 - failure_rate)
  let revenue_per_jar := butterflies_per_jar * price_per_butterfly
  total_amount / revenue_per_jar = number_of_jars := by sorry

end NUMINAMATH_CALUDE_johns_butterfly_jars_l1841_184121


namespace NUMINAMATH_CALUDE_deal_or_no_deal_probability_l1841_184189

/-- The total number of boxes in the game --/
def total_boxes : ℕ := 26

/-- The number of boxes containing at least $250,000 --/
def high_value_boxes : ℕ := 6

/-- The number of boxes to eliminate --/
def boxes_to_eliminate : ℕ := 8

/-- The probability of selecting a high-value box after elimination --/
def probability_high_value : ℚ := 1 / 3

theorem deal_or_no_deal_probability :
  (high_value_boxes : ℚ) / (total_boxes - boxes_to_eliminate : ℚ) = probability_high_value :=
sorry

end NUMINAMATH_CALUDE_deal_or_no_deal_probability_l1841_184189


namespace NUMINAMATH_CALUDE_parabola_circle_theorem_l1841_184139

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define point E
def E : ℝ × ℝ := (2, 2)

-- Define line l
def line_l (y : ℝ) : ℝ := 2 * y + 2

-- Define points A and B on the parabola and line l
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- Define points M and N
def M : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry

-- Origin
def O : ℝ × ℝ := (0, 0)

theorem parabola_circle_theorem :
  parabola E.1 E.2 →
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  A.1 = line_l A.2 →
  B.1 = line_l B.2 →
  A ≠ E →
  B ≠ E →
  M.2 = -2 →
  N.2 = -2 →
  (∃ t : ℝ, M = (1 - t) • E + t • A) →
  (∃ s : ℝ, N = (1 - s) • E + s • B) →
  (O.1 - M.1) * (O.1 - N.1) + (O.2 - M.2) * (O.2 - N.2) = 0 :=
sorry

end NUMINAMATH_CALUDE_parabola_circle_theorem_l1841_184139


namespace NUMINAMATH_CALUDE_set_operations_l1841_184122

-- Define the universal set U
def U : Set ℕ := {n : ℕ | n % 2 = 0 ∧ n ≤ 10}

-- Define set A
def A : Set ℕ := {0, 2, 4, 6}

-- Define set B
def B : Set ℕ := {x : ℕ | x ∈ A ∧ x < 4}

theorem set_operations :
  (Set.compl A) = {8, 10} ∧
  (A ∩ (Set.compl B)) = {4, 6} := by
  sorry

#check set_operations

end NUMINAMATH_CALUDE_set_operations_l1841_184122


namespace NUMINAMATH_CALUDE_inequality_proof_l1841_184171

theorem inequality_proof (a b c : ℝ) (n : ℕ) (p q r : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hn : p + q + r = n) :
  a^n + b^n + c^n ≥ a^p * b^q * c^r + a^q * b^r * c^p + a^r * b^p * c^q :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1841_184171


namespace NUMINAMATH_CALUDE_pavan_total_distance_l1841_184102

/-- Represents a segment of a journey -/
structure Segment where
  speed : ℝ
  time : ℝ

/-- Calculates the distance traveled in a segment -/
def distance_traveled (s : Segment) : ℝ := s.speed * s.time

/-- Represents Pavan's journey -/
def pavan_journey : List Segment := [
  { speed := 30, time := 4 },
  { speed := 35, time := 5 },
  { speed := 25, time := 6 },
  { speed := 40, time := 5 }
]

/-- The total travel time -/
def total_time : ℝ := 20

/-- Theorem stating the total distance traveled by Pavan -/
theorem pavan_total_distance :
  (pavan_journey.map distance_traveled).sum = 645 := by
  sorry

end NUMINAMATH_CALUDE_pavan_total_distance_l1841_184102


namespace NUMINAMATH_CALUDE_range_of_a_l1841_184179

-- Define the custom operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- State the theorem
theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, otimes (x - a) (x + 1) < 1) →
  -2 < a ∧ a < 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1841_184179


namespace NUMINAMATH_CALUDE_magnitude_of_z_l1841_184144

theorem magnitude_of_z : ∀ z : ℂ, z = (Complex.abs (2 + Complex.I) + 2 * Complex.I) / Complex.I → Complex.abs z = 3 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l1841_184144


namespace NUMINAMATH_CALUDE_min_value_ab_l1841_184148

theorem min_value_ab (a b : ℝ) (h1 : a > 1) (h2 : b > 1)
  (h3 : (1/4 * Real.log a) * (Real.log b) = (1/4)^2) : 
  (∀ x y : ℝ, x > 1 → y > 1 → (1/4 * Real.log x) * (Real.log y) = (1/4)^2 → x * y ≥ a * b) →
  a * b = Real.exp 1 := by
sorry


end NUMINAMATH_CALUDE_min_value_ab_l1841_184148


namespace NUMINAMATH_CALUDE_ticket_sales_ratio_l1841_184141

/-- Proves that the ratio of full-price to reduced-price tickets sold during the remaining weeks is 1:1 -/
theorem ticket_sales_ratio (total_tickets : ℕ) (first_week_reduced : ℕ) (total_full_price : ℕ)
  (h1 : total_tickets = 25200)
  (h2 : first_week_reduced = 5400)
  (h3 : total_full_price = 16500)
  (h4 : total_full_price = total_tickets - first_week_reduced - total_full_price) :
  total_full_price = total_tickets - first_week_reduced - total_full_price :=
by sorry

end NUMINAMATH_CALUDE_ticket_sales_ratio_l1841_184141


namespace NUMINAMATH_CALUDE_scarlett_fruit_salad_berries_l1841_184134

/-- The weight of berries in Scarlett's fruit salad -/
def weight_of_berries (total_weight melon_weight : ℚ) : ℚ :=
  total_weight - melon_weight

/-- Proof that the weight of berries in Scarlett's fruit salad is 0.38 pounds -/
theorem scarlett_fruit_salad_berries :
  weight_of_berries (63/100) (1/4) = 38/100 := by
  sorry

end NUMINAMATH_CALUDE_scarlett_fruit_salad_berries_l1841_184134


namespace NUMINAMATH_CALUDE_new_man_weight_l1841_184183

theorem new_man_weight (n : ℕ) (avg_increase : ℝ) (replaced_weight : ℝ) :
  n = 10 →
  avg_increase = 2.5 →
  replaced_weight = 68 →
  (n : ℝ) * avg_increase + replaced_weight = 93 :=
by
  sorry

end NUMINAMATH_CALUDE_new_man_weight_l1841_184183


namespace NUMINAMATH_CALUDE_lcm_gcd_product_equality_l1841_184167

theorem lcm_gcd_product_equality (a b : ℕ) (h : Nat.lcm a b + Nat.gcd a b = a * b) : a = 2 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_equality_l1841_184167


namespace NUMINAMATH_CALUDE_difference_of_squares_l1841_184120

theorem difference_of_squares : 72^2 - 48^2 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1841_184120


namespace NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l1841_184127

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by sorry

theorem increase_80_by_150_percent :
  80 * (1 + 150 / 100) = 200 := by sorry

end NUMINAMATH_CALUDE_increase_by_percentage_increase_80_by_150_percent_l1841_184127


namespace NUMINAMATH_CALUDE_correct_stickers_calculation_l1841_184188

/-- The number of cat stickers each girl received from their grandparents -/
def stickers_from_grandparents (june_initial : ℕ) (bonnie_initial : ℕ) (total_after : ℕ) : ℕ :=
  (total_after - (june_initial + bonnie_initial)) / 2

theorem correct_stickers_calculation :
  stickers_from_grandparents 76 63 189 = 25 := by
  sorry

end NUMINAMATH_CALUDE_correct_stickers_calculation_l1841_184188


namespace NUMINAMATH_CALUDE_rectangle_length_calculation_l1841_184110

theorem rectangle_length_calculation (rectangle_width square_width area_difference : ℝ) : 
  rectangle_width = 6 →
  square_width = 5 →
  rectangle_width * (32 / rectangle_width) - square_width * square_width = area_difference →
  area_difference = 7 →
  32 / rectangle_width = 32 / 6 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_calculation_l1841_184110


namespace NUMINAMATH_CALUDE_problem_solution_l1841_184170

def p : Prop := 0 % 2 = 0
def q : Prop := ∃ k : ℤ, 3 = 2 * k

theorem problem_solution : p ∨ q := by sorry

end NUMINAMATH_CALUDE_problem_solution_l1841_184170


namespace NUMINAMATH_CALUDE_droid_coffee_ratio_l1841_184124

/-- The ratio of afternoon to morning coffee bean usage in Droid's coffee shop --/
def afternoon_to_morning_ratio (morning_bags : ℕ) (total_weekly_bags : ℕ) : ℚ :=
  let afternoon_ratio := (total_weekly_bags / 7 - morning_bags - 2 * morning_bags) / morning_bags
  afternoon_ratio

/-- Theorem stating that the ratio of afternoon to morning coffee bean usage is 3 --/
theorem droid_coffee_ratio :
  afternoon_to_morning_ratio 3 126 = 3 := by sorry

end NUMINAMATH_CALUDE_droid_coffee_ratio_l1841_184124


namespace NUMINAMATH_CALUDE_estimate_larger_than_original_l1841_184173

theorem estimate_larger_than_original 
  (u v δ γ : ℝ) 
  (hu_pos : u > 0) 
  (hv_pos : v > 0) 
  (huv : u > v) 
  (hδγ : δ > γ) 
  (hγ_pos : γ > 0) : 
  (u + δ) - (v - γ) > u - v := by
sorry

end NUMINAMATH_CALUDE_estimate_larger_than_original_l1841_184173


namespace NUMINAMATH_CALUDE_circle_radius_l1841_184119

theorem circle_radius (x y : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, x^2 + y^2 + 4*x - 4*y - 1 = 0 ↔ ∃ h k : ℝ, (x - h)^2 + (y - k)^2 = r^2) →
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, x^2 + y^2 + 4*x - 4*y - 1 = 0 ↔ (x + 2)^2 + (y - 2)^2 = r^2) →
  ∃ r : ℝ, r = 3 ∧ ∀ x y : ℝ, x^2 + y^2 + 4*x - 4*y - 1 = 0 ↔ (x + 2)^2 + (y - 2)^2 = r^2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l1841_184119


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1841_184132

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (∀ a b : ℝ, a > |b| → a^3 > b^3) ∧
  (∃ a b : ℝ, a^3 > b^3 ∧ a ≤ |b|) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1841_184132


namespace NUMINAMATH_CALUDE_storage_tub_cost_l1841_184104

/-- The cost of storage tubs problem -/
theorem storage_tub_cost (total_cost : ℕ) (num_large : ℕ) (num_small : ℕ) (small_cost : ℕ) :
  total_cost = 48 →
  num_large = 3 →
  num_small = 6 →
  small_cost = 5 →
  ∃ (large_cost : ℕ), num_large * large_cost + num_small * small_cost = total_cost ∧ large_cost = 6 :=
by sorry

end NUMINAMATH_CALUDE_storage_tub_cost_l1841_184104


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l1841_184150

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum_property
  (a : ℕ → ℝ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 2 + a 8 = 10) :
  a 3 + a 7 = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_property_l1841_184150


namespace NUMINAMATH_CALUDE_sandy_comic_books_l1841_184142

theorem sandy_comic_books (initial : ℕ) : 
  (initial / 2 + 6 = 13) → initial = 14 := by
  sorry

end NUMINAMATH_CALUDE_sandy_comic_books_l1841_184142


namespace NUMINAMATH_CALUDE_solve_for_A_l1841_184181

theorem solve_for_A : ∃ A : ℕ, 3 + 68 * A = 691 ∧ 100 ≤ 68 * A ∧ 68 * A < 1000 ∧ A = 8 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_A_l1841_184181


namespace NUMINAMATH_CALUDE_square_side_length_l1841_184164

/-- Given an arrangement of rectangles and squares forming a larger rectangle,
    this theorem proves that the side length of square S2 is 900 units. -/
theorem square_side_length (r : ℕ) : 
  (2 * r + 900 = 2800) ∧ (2 * r + 3 * 900 = 4600) → 900 = 900 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l1841_184164


namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l1841_184115

theorem triangle_cosine_theorem (A B C : ℝ) (h1 : A + C = 2 * B) 
  (h2 : 1 / Real.cos A + 1 / Real.cos C = -Real.sqrt 2 / Real.cos B) :
  Real.cos ((A - C) / 2) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l1841_184115


namespace NUMINAMATH_CALUDE_complex_magnitude_equation_l1841_184162

theorem complex_magnitude_equation (z : ℂ) :
  Complex.abs z * (3 * z + 2 * Complex.I) = 2 * (Complex.I * z - 6) →
  Complex.abs z = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equation_l1841_184162


namespace NUMINAMATH_CALUDE_problem_solution_l1841_184152

theorem problem_solution (a b : ℝ) : 
  |a + 1| + (b - 2)^2 = 0 → 
  a = -1 ∧ b = 2 ∧ (a + b)^2020 + a^2019 = 0 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1841_184152


namespace NUMINAMATH_CALUDE_negation_of_sum_squares_l1841_184180

theorem negation_of_sum_squares (x y : ℝ) : -(x^2 + y^2) = -x^2 - y^2 := by
  sorry

end NUMINAMATH_CALUDE_negation_of_sum_squares_l1841_184180


namespace NUMINAMATH_CALUDE_first_supplier_cars_l1841_184135

theorem first_supplier_cars (total_production : ℕ) 
  (second_supplier_extra : ℕ) (fourth_fifth_supplier : ℕ) : 
  total_production = 5650000 →
  second_supplier_extra = 500000 →
  fourth_fifth_supplier = 325000 →
  ∃ (first_supplier : ℕ),
    first_supplier + 
    (first_supplier + second_supplier_extra) + 
    (first_supplier + (first_supplier + second_supplier_extra)) + 
    (2 * fourth_fifth_supplier) = total_production ∧
    first_supplier = 1000000 :=
by sorry

end NUMINAMATH_CALUDE_first_supplier_cars_l1841_184135


namespace NUMINAMATH_CALUDE_system_solution_system_solution_values_no_solution_fractional_equation_l1841_184149

/-- Proves that the system of equations 2x - 7y = 5 and 3x - 8y = 10 has a unique solution -/
theorem system_solution : ∃! (x y : ℝ), 2*x - 7*y = 5 ∧ 3*x - 8*y = 10 := by sorry

/-- Proves that x = 6 and y = 1 is the solution to the system of equations -/
theorem system_solution_values : 
  ∀ (x y : ℝ), (2*x - 7*y = 5 ∧ 3*x - 8*y = 10) → (x = 6 ∧ y = 1) := by sorry

/-- Proves that the equation 3/(x-1) - (x+2)/(x(x-1)) = 0 has no solution -/
theorem no_solution_fractional_equation :
  ¬∃ (x : ℝ), x ≠ 0 ∧ x ≠ 1 ∧ 3/(x-1) - (x+2)/(x*(x-1)) = 0 := by sorry

end NUMINAMATH_CALUDE_system_solution_system_solution_values_no_solution_fractional_equation_l1841_184149


namespace NUMINAMATH_CALUDE_wall_height_is_600_l1841_184156

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculate the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- The dimensions of a single brick -/
def brick_dim : Dimensions := ⟨80, 11.25, 6⟩

/-- The known dimensions of the wall (length and width) -/
def wall_dim (h : ℝ) : Dimensions := ⟨800, 22.5, h⟩

/-- The number of bricks required to build the wall -/
def num_bricks : ℕ := 2000

/-- Theorem stating that if 2000 bricks of given dimensions are required to build a wall
    with known length and width, then the height of the wall is 600 cm -/
theorem wall_height_is_600 :
  volume (wall_dim 600) = (volume brick_dim) * num_bricks := by sorry

end NUMINAMATH_CALUDE_wall_height_is_600_l1841_184156


namespace NUMINAMATH_CALUDE_max_value_theorem_l1841_184146

theorem max_value_theorem (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) 
  (h4 : x^2 + y^2 + z^2 = 1) : 
  2 * x^2 * y * Real.sqrt 6 + 8 * y^2 * z ≤ Real.sqrt (144/35) + Real.sqrt (88/35) := by
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l1841_184146


namespace NUMINAMATH_CALUDE_factorial_simplification_l1841_184195

theorem factorial_simplification : 
  Nat.factorial 15 / (Nat.factorial 11 + 3 * Nat.factorial 10) = 25740 := by
  sorry

end NUMINAMATH_CALUDE_factorial_simplification_l1841_184195


namespace NUMINAMATH_CALUDE_problem_solution_l1841_184145

theorem problem_solution : ∃! n : ℕ, n > 1 ∧ Nat.Prime n ∧ Even n ∧ n ≠ 9 ∧ ¬(15 ∣ n) :=
by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1841_184145


namespace NUMINAMATH_CALUDE_four_possible_ones_digits_l1841_184191

-- Define a function to check if a number is divisible by 6
def divisible_by_six (n : ℕ) : Prop := n % 6 = 0

-- Define a function to get the ones digit of a number
def ones_digit (n : ℕ) : ℕ := n % 10

-- Theorem statement
theorem four_possible_ones_digits :
  ∃! (s : Finset ℕ), 
    (∀ n ∈ s, ones_digit n ∈ s ∧ divisible_by_six n) ∧
    (∀ d, d ∈ s ↔ ∃ n, ones_digit n = d ∧ divisible_by_six n) ∧
    Finset.card s = 4 :=
sorry

end NUMINAMATH_CALUDE_four_possible_ones_digits_l1841_184191


namespace NUMINAMATH_CALUDE_travel_problem_solution_l1841_184165

/-- Represents the speeds and distance in the problem -/
structure TravelData where
  pedestrian_speed : ℝ
  cyclist_speed : ℝ
  rider_speed : ℝ
  distance_AB : ℝ

/-- The conditions of the problem -/
def problem_conditions (data : TravelData) : Prop :=
  data.cyclist_speed = 2 * data.pedestrian_speed ∧
  2 * data.cyclist_speed + 2 * data.rider_speed = data.distance_AB ∧
  2.8 * data.pedestrian_speed + 2.8 * data.rider_speed = data.distance_AB ∧
  2 * data.rider_speed = data.distance_AB / 2 - 3 ∧
  2 * data.cyclist_speed = data.distance_AB / 2 + 3

/-- The theorem to prove -/
theorem travel_problem_solution :
  ∃ (data : TravelData),
    problem_conditions data ∧
    data.pedestrian_speed = 6 ∧
    data.cyclist_speed = 12 ∧
    data.rider_speed = 9 ∧
    data.distance_AB = 42 :=
by
  sorry

end NUMINAMATH_CALUDE_travel_problem_solution_l1841_184165


namespace NUMINAMATH_CALUDE_xy_squared_l1841_184116

theorem xy_squared (x y : ℝ) (h1 : x + y = 20) (h2 : 2*x + y = 27) : (x + y)^2 = 400 := by
  sorry

end NUMINAMATH_CALUDE_xy_squared_l1841_184116


namespace NUMINAMATH_CALUDE_grid_path_problem_l1841_184163

/-- The number of paths on a grid from (0,0) to (m,n) with exactly k steps -/
def grid_paths (m n k : ℕ) : ℕ := Nat.choose k m

/-- The problem statement -/
theorem grid_path_problem :
  let m : ℕ := 7  -- width of the grid
  let n : ℕ := 5  -- height of the grid
  let k : ℕ := 10 -- total number of steps
  grid_paths m n k = 120 := by sorry

end NUMINAMATH_CALUDE_grid_path_problem_l1841_184163


namespace NUMINAMATH_CALUDE_quadrilateral_to_parallelogram_l1841_184178

-- Define the points
variable (A B C D E F O : ℝ × ℝ)

-- Define the conditions
def is_convex_quadrilateral (A B C D : ℝ × ℝ) : Prop := sorry

def is_midpoint (M X Y : ℝ × ℝ) : Prop := sorry

def segments_intersect (P Q R S : ℝ × ℝ) (I : ℝ × ℝ) : Prop := sorry

def divides_into_three_equal_parts (P Q R S : ℝ × ℝ) : Prop := sorry

def is_parallelogram (A B C D : ℝ × ℝ) : Prop := sorry

-- State the theorem
theorem quadrilateral_to_parallelogram 
  (h_convex : is_convex_quadrilateral A B C D)
  (h_midpoint_E : is_midpoint E A D)
  (h_midpoint_F : is_midpoint F B C)
  (h_intersect : segments_intersect C E D F O)
  (h_divide_AO : divides_into_three_equal_parts A O C D)
  (h_divide_BO : divides_into_three_equal_parts B O C D) :
  is_parallelogram A B C D :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_to_parallelogram_l1841_184178


namespace NUMINAMATH_CALUDE_yoojungs_initial_candies_l1841_184137

/-- The number of candies Yoojung gave to her older sister -/
def candies_to_older_sister : ℕ := 7

/-- The number of candies Yoojung gave to her younger sister -/
def candies_to_younger_sister : ℕ := 6

/-- The number of candies Yoojung had left after giving candies to her sisters -/
def candies_left : ℕ := 15

/-- The initial number of candies Yoojung had -/
def initial_candies : ℕ := candies_to_older_sister + candies_to_younger_sister + candies_left

theorem yoojungs_initial_candies : initial_candies = 28 := by
  sorry

end NUMINAMATH_CALUDE_yoojungs_initial_candies_l1841_184137


namespace NUMINAMATH_CALUDE_digit_sum_property_l1841_184114

/-- A function that returns true if a number has no zero digits -/
def has_no_zero_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d ≠ 0

/-- A function that returns all digit permutations of a number -/
def digit_permutations (n : ℕ) : Finset ℕ :=
  sorry

/-- A function that checks if a number is composed entirely of ones -/
def all_ones (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 1

/-- A function that checks if a number has a digit 5 or greater -/
def has_digit_ge_5 (n : ℕ) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ d ≥ 5

theorem digit_sum_property (n : ℕ) (h1 : has_no_zero_digits n) 
  (h2 : all_ones (n + (Finset.sum (digit_permutations n) id))) :
  has_digit_ge_5 n :=
sorry

end NUMINAMATH_CALUDE_digit_sum_property_l1841_184114


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_fraction_l1841_184174

theorem simplify_and_rationalize_fraction :
  (5 : ℝ) / (Real.sqrt 50 + 3 * Real.sqrt 8 + Real.sqrt 18 + Real.sqrt 32) = (5 * Real.sqrt 2) / 36 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_fraction_l1841_184174


namespace NUMINAMATH_CALUDE_cycle_selling_price_l1841_184153

/-- Calculates the final selling price of a cycle given the original price and various discounts and losses. -/
def final_selling_price (original_price : ℝ) (initial_discount : ℝ) (loss_on_sale : ℝ) (exchange_discount : ℝ) : ℝ :=
  let price_after_initial_discount := original_price * (1 - initial_discount)
  let intended_selling_price := price_after_initial_discount * (1 - loss_on_sale)
  intended_selling_price * (1 - exchange_discount)

/-- Theorem stating that the final selling price of the cycle is 897.75 given the specified conditions. -/
theorem cycle_selling_price :
  final_selling_price 1400 0.05 0.25 0.10 = 897.75 := by
  sorry

end NUMINAMATH_CALUDE_cycle_selling_price_l1841_184153


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_plus_i_l1841_184154

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Theorem statement
theorem imaginary_part_of_i_times_one_plus_i :
  (i * (1 + i)).im = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_times_one_plus_i_l1841_184154


namespace NUMINAMATH_CALUDE_meal_combinations_l1841_184147

/-- The number of items on the menu -/
def menu_items : ℕ := 15

/-- The number of dishes Camille avoids -/
def avoided_dishes : ℕ := 2

/-- The number of dishes Camille can choose from -/
def camille_choices : ℕ := menu_items - avoided_dishes

theorem meal_combinations : menu_items * camille_choices = 195 := by
  sorry

end NUMINAMATH_CALUDE_meal_combinations_l1841_184147


namespace NUMINAMATH_CALUDE_lara_today_cans_l1841_184169

/-- The number of cans collected by Sarah and Lara over two days -/
structure CanCollection where
  sarah_yesterday : ℕ
  lara_yesterday : ℕ
  sarah_today : ℕ
  lara_today : ℕ

/-- The conditions of the can collection problem -/
def can_collection_problem (c : CanCollection) : Prop :=
  c.sarah_yesterday = 50 ∧
  c.lara_yesterday = c.sarah_yesterday + 30 ∧
  c.sarah_today = 40 ∧
  c.sarah_today + c.lara_today = c.sarah_yesterday + c.lara_yesterday - 20

/-- The theorem stating Lara collected 70 cans today -/
theorem lara_today_cans (c : CanCollection) :
  can_collection_problem c → c.lara_today = 70 := by
  sorry

end NUMINAMATH_CALUDE_lara_today_cans_l1841_184169


namespace NUMINAMATH_CALUDE_train_length_calculation_l1841_184111

theorem train_length_calculation (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) : 
  jogger_speed = 9 →
  train_speed = 45 →
  initial_distance = 240 →
  passing_time = 37 →
  let relative_speed := (train_speed - jogger_speed) * (5 / 18)
  let distance_traveled := relative_speed * passing_time
  let train_length := distance_traveled - initial_distance
  train_length = 130 := by
sorry

end NUMINAMATH_CALUDE_train_length_calculation_l1841_184111


namespace NUMINAMATH_CALUDE_tan_2715_degrees_l1841_184161

theorem tan_2715_degrees : Real.tan (2715 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_2715_degrees_l1841_184161


namespace NUMINAMATH_CALUDE_max_y_value_l1841_184130

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -2) : 
  ∃ (max_y : ℤ), (∀ (y' : ℤ), ∃ (x' : ℤ), x' * y' + 3 * x' + 2 * y' = -2 → y' ≤ max_y) ∧ max_y = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l1841_184130


namespace NUMINAMATH_CALUDE_complex_equation_l1841_184118

theorem complex_equation (z : ℂ) (h : Complex.abs z = 1 + 3*I - z) :
  ((1 + I)^2 * (3 + 4*I)^2) / (2 * z) = 3 + 4*I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_l1841_184118


namespace NUMINAMATH_CALUDE_root_transformation_l1841_184126

/-- Given a nonzero constant k and roots a, b, c, d of the equation kx^4 - 5kx - 12 = 0,
    the polynomial with roots (b+c+d)/(ka^2), (a+c+d)/(kb^2), (a+b+d)/(kc^2), (a+b+c)/(kd^2)
    is 12k^3x^4 - 5k^3x^3 - 1 = 0 -/
theorem root_transformation (k : ℝ) (a b c d : ℝ) : k ≠ 0 →
  (k * a^4 - 5*k*a - 12 = 0) →
  (k * b^4 - 5*k*b - 12 = 0) →
  (k * c^4 - 5*k*c - 12 = 0) →
  (k * d^4 - 5*k*d - 12 = 0) →
  ∃ (x : ℝ), 12*k^3*x^4 - 5*k^3*x^3 - 1 = 0 ∧
    (x = (b+c+d)/(k*a^2) ∨ x = (a+c+d)/(k*b^2) ∨ x = (a+b+d)/(k*c^2) ∨ x = (a+b+c)/(k*d^2)) :=
by sorry

end NUMINAMATH_CALUDE_root_transformation_l1841_184126


namespace NUMINAMATH_CALUDE_big_stack_orders_l1841_184168

/-- The number of pancakes in a big stack -/
def big_stack : ℕ := 5

/-- The number of pancakes in a short stack -/
def short_stack : ℕ := 3

/-- The number of customers who ordered the short stack -/
def short_stack_orders : ℕ := 9

/-- The total number of pancakes needed -/
def total_pancakes : ℕ := 57

/-- Theorem stating that the number of customers who ordered the big stack is 6 -/
theorem big_stack_orders : ℕ := by
  sorry

end NUMINAMATH_CALUDE_big_stack_orders_l1841_184168


namespace NUMINAMATH_CALUDE_class_representation_ratio_l1841_184136

theorem class_representation_ratio :
  ∀ (num_boys num_girls : ℕ),
  num_boys > 0 →
  num_girls > 0 →
  (num_boys : ℚ) / (num_boys + num_girls : ℚ) = 3 / 5 * (num_girls : ℚ) / (num_boys + num_girls : ℚ) →
  (num_boys : ℚ) / (num_boys + num_girls : ℚ) = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_class_representation_ratio_l1841_184136


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_81_l1841_184106

theorem arithmetic_square_root_of_81 : Real.sqrt 81 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_81_l1841_184106


namespace NUMINAMATH_CALUDE_square_plus_one_geq_double_l1841_184155

theorem square_plus_one_geq_double (x : ℝ) : x^2 + 1 ≥ 2*x := by
  sorry

end NUMINAMATH_CALUDE_square_plus_one_geq_double_l1841_184155


namespace NUMINAMATH_CALUDE_box_counting_l1841_184176

theorem box_counting (initial_boxes : ℕ) (boxes_per_filled : ℕ) (non_empty_boxes : ℕ) : 
  initial_boxes = 7 →
  boxes_per_filled = 7 →
  non_empty_boxes = 10 →
  initial_boxes + (boxes_per_filled * non_empty_boxes) = 77 :=
by sorry

end NUMINAMATH_CALUDE_box_counting_l1841_184176


namespace NUMINAMATH_CALUDE_brothers_age_ratio_l1841_184138

theorem brothers_age_ratio :
  ∀ (rick_age oldest_age middle_age smallest_age youngest_age : ℕ),
    rick_age = 15 →
    oldest_age = 2 * rick_age →
    middle_age = oldest_age / 3 →
    youngest_age = 3 →
    smallest_age = youngest_age + 2 →
    (smallest_age : ℚ) / (middle_age : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_ratio_l1841_184138


namespace NUMINAMATH_CALUDE_three_true_propositions_l1841_184101

theorem three_true_propositions :
  (∀ (x : ℝ), x^2 + 1 > 0) ∧
  (∃ (x : ℤ), x^3 < 1) ∧
  (∀ (x : ℚ), x^2 ≠ 2) ∧
  ¬(∀ (x : ℕ), x^4 ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_three_true_propositions_l1841_184101


namespace NUMINAMATH_CALUDE_base_conversion_theorem_l1841_184129

def base_7_to_10 (n : Nat) : Nat :=
  5 * 7^5 + 4 * 7^4 + 3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0

def base_10_to_4 (n : Nat) : List Nat :=
  let rec convert (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else convert (m / 4) ((m % 4) :: acc)
  convert n []

theorem base_conversion_theorem :
  (base_7_to_10 543210 = 94773) ∧
  (base_10_to_4 94773 = [1, 1, 3, 2, 3, 0, 1, 1]) := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_theorem_l1841_184129


namespace NUMINAMATH_CALUDE_share_ratio_l1841_184105

theorem share_ratio (total c b a : ℕ) (h1 : total = 406) (h2 : total = a + b + c) 
  (h3 : b = c / 2) (h4 : c = 232) : a / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_share_ratio_l1841_184105


namespace NUMINAMATH_CALUDE_monomial_same_type_l1841_184123

/-- A structure representing a monomial with coefficients in ℤ -/
structure Monomial :=
  (coeff : ℤ)
  (m_exp : ℕ)
  (n_exp : ℕ)

/-- Two monomials are of the same type if they have the same variables with the same exponents -/
def same_type (a b : Monomial) : Prop :=
  a.m_exp = b.m_exp ∧ a.n_exp = b.n_exp

/-- The monomial -2mn^2 -/
def monomial1 : Monomial :=
  { coeff := -2, m_exp := 1, n_exp := 2 }

/-- The monomial mn^2 -/
def monomial2 : Monomial :=
  { coeff := 1, m_exp := 1, n_exp := 2 }

theorem monomial_same_type : same_type monomial1 monomial2 := by
  sorry

end NUMINAMATH_CALUDE_monomial_same_type_l1841_184123


namespace NUMINAMATH_CALUDE_natural_pythagorean_triples_real_circle_equation_l1841_184182

-- Part 1: Natural numbers
def natural_solutions : Set (ℕ × ℕ) :=
  {(0, 5), (5, 0), (3, 4), (4, 3)}

theorem natural_pythagorean_triples :
  ∀ (x y : ℕ), x^2 + y^2 = 25 ↔ (x, y) ∈ natural_solutions :=
sorry

-- Part 2: Real numbers
def real_solutions : Set (ℝ × ℝ) :=
  {(x, y) | -5 ≤ x ∧ x ≤ 5 ∧ (y = Real.sqrt (25 - x^2) ∨ y = -Real.sqrt (25 - x^2))}

theorem real_circle_equation :
  ∀ (x y : ℝ), x^2 + y^2 = 25 ↔ (x, y) ∈ real_solutions :=
sorry

end NUMINAMATH_CALUDE_natural_pythagorean_triples_real_circle_equation_l1841_184182


namespace NUMINAMATH_CALUDE_consecutive_odd_numbers_l1841_184117

theorem consecutive_odd_numbers (a b c d e : ℕ) : 
  (∃ k : ℕ, a = 2*k + 1) ∧ 
  b = a + 2 ∧ 
  c = b + 2 ∧ 
  d = c + 2 ∧ 
  e = d + 2 ∧ 
  a + c = 146 ∧ 
  e = 79 →
  a = 71 := by
sorry

end NUMINAMATH_CALUDE_consecutive_odd_numbers_l1841_184117


namespace NUMINAMATH_CALUDE_recycling_drive_target_l1841_184177

/-- The recycling drive problem -/
theorem recycling_drive_target (num_sections : ℕ) (kilos_per_section : ℕ) (kilos_needed : ℕ) : 
  num_sections = 6 → 
  kilos_per_section = 280 → 
  kilos_needed = 320 → 
  num_sections * kilos_per_section + kilos_needed = 2000 := by
sorry

end NUMINAMATH_CALUDE_recycling_drive_target_l1841_184177


namespace NUMINAMATH_CALUDE_consecutive_triples_divisible_by_1001_l1841_184107

def is_valid_triple (a b c : ℕ) : Prop :=
  a < 101 ∧ b < 101 ∧ c < 101 ∧
  b = a + 1 ∧ c = b + 1 ∧
  (a * b * c) % 1001 = 0

theorem consecutive_triples_divisible_by_1001 :
  ∀ a b c : ℕ,
    is_valid_triple a b c ↔ (a = 76 ∧ b = 77 ∧ c = 78) ∨ (a = 77 ∧ b = 78 ∧ c = 79) :=
by sorry

end NUMINAMATH_CALUDE_consecutive_triples_divisible_by_1001_l1841_184107


namespace NUMINAMATH_CALUDE_new_supervisor_salary_l1841_184175

/-- Proves that the new supervisor's salary must be $870 to maintain the same average salary --/
theorem new_supervisor_salary
  (num_workers : ℕ)
  (num_total : ℕ)
  (initial_average : ℚ)
  (old_supervisor_salary : ℚ)
  (new_average : ℚ)
  (h_num_workers : num_workers = 8)
  (h_num_total : num_total = num_workers + 1)
  (h_initial_average : initial_average = 430)
  (h_old_supervisor_salary : old_supervisor_salary = 870)
  (h_new_average : new_average = initial_average)
  : ∃ (new_supervisor_salary : ℚ),
    new_supervisor_salary = 870 ∧
    (num_workers : ℚ) * initial_average + old_supervisor_salary = num_total * initial_average ∧
    (num_workers : ℚ) * new_average + new_supervisor_salary = num_total * new_average :=
by
  sorry


end NUMINAMATH_CALUDE_new_supervisor_salary_l1841_184175


namespace NUMINAMATH_CALUDE_equation_solution_l1841_184140

theorem equation_solution (x y : ℝ) :
  x / (x - 1) = (y^2 + 3*y + 2) / (y^2 + 3*y - 1) →
  x = (y^2 + 3*y + 2) / 3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1841_184140


namespace NUMINAMATH_CALUDE_square_field_area_proof_l1841_184197

/-- The time taken to cross the square field diagonally in hours -/
def crossing_time : ℝ := 6.0008333333333335

/-- The speed of the person crossing the field in km/hr -/
def crossing_speed : ℝ := 1.2

/-- The area of the square field in square meters -/
def field_area : ℝ := 25939744.8

/-- Theorem stating that the area of the square field is approximately 25939744.8 square meters -/
theorem square_field_area_proof :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  |field_area - (crossing_speed * 1000 * crossing_time)^2 / 2| < ε :=
sorry

end NUMINAMATH_CALUDE_square_field_area_proof_l1841_184197


namespace NUMINAMATH_CALUDE_jenny_ate_65_squares_l1841_184131

-- Define the number of chocolate squares Mike ate
def mike_squares : ℕ := 20

-- Define the number of chocolate squares Jenny ate
def jenny_squares : ℕ := 3 * mike_squares + 5

-- Theorem to prove
theorem jenny_ate_65_squares : jenny_squares = 65 := by
  sorry

end NUMINAMATH_CALUDE_jenny_ate_65_squares_l1841_184131


namespace NUMINAMATH_CALUDE_shopkeeper_profit_l1841_184160

/-- The number of markers bought by the shopkeeper -/
def total_markers : ℕ := 2000

/-- The cost price of each marker in dollars -/
def cost_price : ℚ := 3/10

/-- The selling price of each marker in dollars -/
def selling_price : ℚ := 11/20

/-- The target profit in dollars -/
def target_profit : ℚ := 150

/-- The number of markers that need to be sold to achieve the target profit -/
def markers_to_sell : ℕ := 1364

theorem shopkeeper_profit :
  (markers_to_sell : ℚ) * selling_price - (total_markers : ℚ) * cost_price = target_profit :=
sorry

end NUMINAMATH_CALUDE_shopkeeper_profit_l1841_184160


namespace NUMINAMATH_CALUDE_libor_lucky_numbers_l1841_184143

theorem libor_lucky_numbers :
  {n : ℕ | n < 1000 ∧ 7 ∣ n^2 ∧ 8 ∣ n^2 ∧ 9 ∣ n^2 ∧ 10 ∣ n^2} = {420, 840} :=
by sorry

end NUMINAMATH_CALUDE_libor_lucky_numbers_l1841_184143


namespace NUMINAMATH_CALUDE_solve_composite_function_equation_l1841_184151

theorem solve_composite_function_equation (a : ℝ) 
  (h : ℝ → ℝ) (f : ℝ → ℝ) (g : ℝ → ℝ)
  (h_def : ∀ x, h x = x + 2)
  (f_def : ∀ x, f x = 2 * x + 3)
  (g_def : ∀ x, g x = x^2 - 5)
  (a_pos : a > 0)
  (eq : h (f (g a)) = 12) :
  a = Real.sqrt (17 / 2) := by
sorry

end NUMINAMATH_CALUDE_solve_composite_function_equation_l1841_184151


namespace NUMINAMATH_CALUDE_map_distance_to_real_distance_l1841_184196

/-- Proves that for a map with scale 1:500,000, a 4 cm distance on the map represents 20 km in reality -/
theorem map_distance_to_real_distance 
  (scale : ℚ) 
  (map_distance : ℚ) 
  (h_scale : scale = 1 / 500000)
  (h_map_distance : map_distance = 4)
  : map_distance * scale * 100000 = 20 := by
  sorry

end NUMINAMATH_CALUDE_map_distance_to_real_distance_l1841_184196
