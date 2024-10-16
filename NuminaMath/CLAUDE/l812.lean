import Mathlib

namespace NUMINAMATH_CALUDE_incorrect_number_calculation_l812_81279

theorem incorrect_number_calculation (n : ℕ) (initial_avg correct_avg correct_num incorrect_num : ℚ) : 
  n = 10 ∧ 
  initial_avg = 16 ∧ 
  correct_avg = 18 ∧ 
  correct_num = 46 ∧ 
  n * initial_avg + (correct_num - incorrect_num) = n * correct_avg → 
  incorrect_num = 26 := by
sorry

end NUMINAMATH_CALUDE_incorrect_number_calculation_l812_81279


namespace NUMINAMATH_CALUDE_point_transformation_l812_81283

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the transformation function
def transform (p : Point2D) : Point2D :=
  { x := p.x + 3, y := p.y + 5 }

theorem point_transformation :
  ∀ (x y : ℝ),
  let A : Point2D := { x := x, y := -2 }
  let B : Point2D := transform A
  B.x = 1 → x = -2 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l812_81283


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l812_81245

-- Define the quadratic function
def f (a c x : ℝ) : ℝ := a * x^2 - 4*x + c

-- State the theorem
theorem quadratic_function_properties :
  ∀ a c : ℝ,
  (∀ x : ℝ, f a c x < 0 ↔ -1 < x ∧ x < 5) →
  (a = 1 ∧ c = -5) ∧
  (∀ x : ℝ, x ∈ Set.Icc 0 3 → f a c x ∈ Set.Icc (-9) (-5)) ∧
  (∃ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 0 3 ∧ x₂ ∈ Set.Icc 0 3 ∧ f a c x₁ = -9 ∧ f a c x₂ = -5) :=
by sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l812_81245


namespace NUMINAMATH_CALUDE_function_properties_l812_81225

noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
  if x < c then c * x + 1 else 3 * x^4 + x^2 * c

theorem function_properties (c : ℝ) 
  (h1 : 0 < c) (h2 : c < 1) (h3 : f c c^2 = 9/8) :
  c = 1/2 ∧ ∀ x, f (1/2) x < 2 ↔ 0 < x ∧ x < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l812_81225


namespace NUMINAMATH_CALUDE_frog_reaches_boundary_in_three_hops_l812_81257

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Represents the possible directions of movement -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Defines whether a position is on the boundary of the grid -/
def is_boundary (p : Position) : Bool :=
  p.x = 0 || p.x = 3 || p.y = 0 || p.y = 3

/-- Defines a single hop movement on the grid -/
def hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨min 3 (p.x + 1), p.y⟩
  | Direction.Down => ⟨max 0 (p.x - 1), p.y⟩
  | Direction.Left => ⟨p.x, max 0 (p.y - 1)⟩
  | Direction.Right => ⟨p.x, min 3 (p.y + 1)⟩

/-- Calculates the probability of reaching the boundary within n hops -/
def prob_reach_boundary (start : Position) (n : Nat) : ℝ :=
  sorry

theorem frog_reaches_boundary_in_three_hops :
  prob_reach_boundary ⟨1, 1⟩ 3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_frog_reaches_boundary_in_three_hops_l812_81257


namespace NUMINAMATH_CALUDE_cathys_initial_amount_l812_81210

/-- The amount of money Cathy had before her parents sent her money -/
def initial_amount : ℕ := sorry

/-- The amount of money Cathy's dad sent her -/
def dad_amount : ℕ := 25

/-- The amount of money Cathy's mom sent her -/
def mom_amount : ℕ := 2 * dad_amount

/-- The total amount Cathy has now -/
def total_amount : ℕ := 87

theorem cathys_initial_amount : 
  initial_amount = total_amount - (dad_amount + mom_amount) ∧ 
  initial_amount = 12 := by sorry

end NUMINAMATH_CALUDE_cathys_initial_amount_l812_81210


namespace NUMINAMATH_CALUDE_right_triangle_height_radius_ratio_l812_81220

theorem right_triangle_height_radius_ratio (a b c h r : ℝ) :
  a > 0 → b > 0 → c > 0 → h > 0 → r > 0 →
  a^2 + b^2 = c^2 →  -- Right triangle condition
  (a + b + c) * r = c * h →  -- Area equality condition
  2 < h / r ∧ h / r ≤ 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_height_radius_ratio_l812_81220


namespace NUMINAMATH_CALUDE_simple_interest_problem_l812_81281

/-- Simple interest calculation -/
theorem simple_interest_problem (interest : ℝ) (rate : ℝ) (time : ℝ) (principal : ℝ) :
  interest = 4016.25 →
  rate = 11 →
  time = 5 →
  principal = interest / (rate * time / 100) →
  principal = 7302.27 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_problem_l812_81281


namespace NUMINAMATH_CALUDE_some_number_value_l812_81274

theorem some_number_value : ∃ (n : ℚ), n = 10/3 ∧ 
  (3 + 2 * (3/2 : ℚ))^5 = (1 + n * (3/2 : ℚ))^4 := by
  sorry

end NUMINAMATH_CALUDE_some_number_value_l812_81274


namespace NUMINAMATH_CALUDE_teal_sales_theorem_l812_81280

/-- Represents the types of pies sold in the bakery -/
inductive PieType
| Pumpkin
| Custard

/-- Represents the properties of a pie -/
structure Pie where
  pieType : PieType
  slicesPerPie : Nat
  pricePerSlice : Nat
  piesCount : Nat

/-- Calculates the total sales for a given pie -/
def totalSales (pie : Pie) : Nat :=
  pie.slicesPerPie * pie.pricePerSlice * pie.piesCount

/-- Theorem: Teal's total sales from pumpkin and custard pies equal $340 -/
theorem teal_sales_theorem (pumpkinPie custardPie : Pie)
    (h_pumpkin : pumpkinPie.pieType = PieType.Pumpkin ∧
                 pumpkinPie.slicesPerPie = 8 ∧
                 pumpkinPie.pricePerSlice = 5 ∧
                 pumpkinPie.piesCount = 4)
    (h_custard : custardPie.pieType = PieType.Custard ∧
                 custardPie.slicesPerPie = 6 ∧
                 custardPie.pricePerSlice = 6 ∧
                 custardPie.piesCount = 5) :
    totalSales pumpkinPie + totalSales custardPie = 340 := by
  sorry

end NUMINAMATH_CALUDE_teal_sales_theorem_l812_81280


namespace NUMINAMATH_CALUDE_average_ticket_cost_l812_81289

/-- Calculates the average cost of tickets per person given the specified conditions --/
theorem average_ticket_cost (full_price : ℕ) (total_people : ℕ) (half_price_tickets : ℕ) (free_tickets : ℕ) (full_price_tickets : ℕ) :
  full_price = 150 →
  total_people = 5 →
  half_price_tickets = 2 →
  free_tickets = 1 →
  full_price_tickets = 2 →
  (full_price * full_price_tickets + (full_price / 2) * half_price_tickets) / total_people = 90 :=
by sorry

end NUMINAMATH_CALUDE_average_ticket_cost_l812_81289


namespace NUMINAMATH_CALUDE_min_sum_a_c_l812_81282

theorem min_sum_a_c (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h1 : (a - c) * (b - d) = -4)
  (h2 : (a + c) / 2 ≥ (a^2 + b^2 + c^2 + d^2) / (a + b + c + d)) :
  ∀ ε > 0, a + c ≥ 4 * Real.sqrt 2 - ε :=
by sorry

end NUMINAMATH_CALUDE_min_sum_a_c_l812_81282


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l812_81288

theorem quadratic_distinct_roots (p q : ℚ) : 
  (∃ x y : ℚ, x ≠ y ∧ 
    x^2 + p*x + q = 0 ∧ 
    y^2 + p*y + q = 0 ∧ 
    x = 2*p ∧ 
    y = p + q) → 
  (p = 2/3 ∧ q = -8/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l812_81288


namespace NUMINAMATH_CALUDE_simple_interest_difference_l812_81213

/-- Calculate the simple interest and prove that it's Rs. 306 less than the principal -/
theorem simple_interest_difference (principal rate time : ℝ) : 
  principal = 450 → 
  rate = 4 → 
  time = 8 → 
  principal - (principal * rate * time / 100) = 306 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_difference_l812_81213


namespace NUMINAMATH_CALUDE_total_hours_is_fifty_l812_81261

/-- Calculates the total hours needed to make dresses from two types of fabric -/
def total_hours_for_dresses (fabric_a_total : ℕ) (fabric_a_per_dress : ℕ) (fabric_a_hours : ℕ)
                            (fabric_b_total : ℕ) (fabric_b_per_dress : ℕ) (fabric_b_hours : ℕ) : ℕ :=
  let dresses_a := fabric_a_total / fabric_a_per_dress
  let dresses_b := fabric_b_total / fabric_b_per_dress
  dresses_a * fabric_a_hours + dresses_b * fabric_b_hours

/-- Theorem stating that the total hours needed to make dresses from the given fabrics is 50 -/
theorem total_hours_is_fifty :
  total_hours_for_dresses 40 4 3 28 5 4 = 50 := by
  sorry

end NUMINAMATH_CALUDE_total_hours_is_fifty_l812_81261


namespace NUMINAMATH_CALUDE_max_value_cube_root_sum_l812_81234

theorem max_value_cube_root_sum (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) : 
  (a^2 * b^2 * c^2)^(1/3) + ((1 - a^2) * (1 - b^2) * (1 - c^2))^(1/3) ≤ 1 ∧
  ∃ x y z, 0 ≤ x ∧ x ≤ 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧ 0 ≤ z ∧ z ≤ 1 ∧
    (x^2 * y^2 * z^2)^(1/3) + ((1 - x^2) * (1 - y^2) * (1 - z^2))^(1/3) = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_cube_root_sum_l812_81234


namespace NUMINAMATH_CALUDE_complex_roots_of_quadratic_l812_81235

theorem complex_roots_of_quadratic : 
  let z₁ : ℂ := -1 + Real.sqrt 2 - Complex.I * Real.sqrt 2
  let z₂ : ℂ := -1 - Real.sqrt 2 + Complex.I * Real.sqrt 2
  (z₁^2 + 2*z₁ = 3 - 4*Complex.I) ∧ (z₂^2 + 2*z₂ = 3 - 4*Complex.I) := by
  sorry


end NUMINAMATH_CALUDE_complex_roots_of_quadratic_l812_81235


namespace NUMINAMATH_CALUDE_fraction_value_l812_81215

theorem fraction_value : (900 ^ 2 : ℚ) / (153 ^ 2 - 147 ^ 2) = 450 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l812_81215


namespace NUMINAMATH_CALUDE_hotpot_expenditure_theorem_l812_81223

/-- Represents the expenditure of three people on hotpot base materials. -/
structure HotpotExpenditure where
  a : ℕ  -- number of brands
  m : ℕ  -- price of clear soup flavor
  n : ℕ  -- price of mushroom soup flavor
  spicy_price : ℕ := 25  -- price of spicy flavor

/-- Conditions for the hotpot expenditure problem -/
def valid_expenditure (h : HotpotExpenditure) : Prop :=
  h.a * (h.spicy_price + h.m + h.n) = 1900 ∧
  33 ≤ h.m ∧ h.m < h.n ∧ h.n ≤ 37

/-- The maximum amount Xiao Li could have spent on clear soup and mushroom soup flavors -/
def max_non_spicy_expenditure (h : HotpotExpenditure) : ℕ :=
  700 - h.spicy_price

/-- The main theorem stating the maximum amount Xiao Li could have spent on non-spicy flavors -/
theorem hotpot_expenditure_theorem (h : HotpotExpenditure) 
  (h_valid : valid_expenditure h) : 
  max_non_spicy_expenditure h = 675 := by
  sorry

end NUMINAMATH_CALUDE_hotpot_expenditure_theorem_l812_81223


namespace NUMINAMATH_CALUDE_polar_coordinates_of_point_l812_81233

theorem polar_coordinates_of_point (x y : ℝ) (h : (x, y) = (-1, Real.sqrt 3)) :
  ∃ (ρ θ : ℝ), ρ = 2 ∧ θ = 2 * Real.pi / 3 ∧ 
  x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ :=
by sorry

end NUMINAMATH_CALUDE_polar_coordinates_of_point_l812_81233


namespace NUMINAMATH_CALUDE_number_equation_l812_81201

theorem number_equation : ∃ x : ℚ, (5 + 4/9) / 7 = 5 * x ∧ x = 49/315 := by
  sorry

end NUMINAMATH_CALUDE_number_equation_l812_81201


namespace NUMINAMATH_CALUDE_jills_lavender_candles_l812_81214

/-- Represents the number of candles of each scent Jill made -/
structure CandleCounts where
  lavender : ℕ
  coconut : ℕ
  almond : ℕ
  jasmine : ℕ

/-- Represents the amount of scent (in ml) required for each type of candle -/
def scentAmounts : CandleCounts where
  lavender := 10
  coconut := 8
  almond := 12
  jasmine := 9

/-- The total number of almond candles Jill made -/
def totalAlmondCandles : ℕ := 12

/-- The ratio of coconut scent to almond scent Jill had -/
def coconutToAlmondRatio : ℚ := 5/2

theorem jills_lavender_candles (counts : CandleCounts) : counts.lavender = 135 :=
  by
  have h1 : counts.lavender = 3 * counts.coconut := by sorry
  have h2 : counts.almond = 2 * counts.jasmine := by sorry
  have h3 : counts.almond = totalAlmondCandles := by sorry
  have h4 : counts.coconut * scentAmounts.coconut = 
            coconutToAlmondRatio * (counts.almond * scentAmounts.almond) := by sorry
  have h5 : counts.jasmine * scentAmounts.jasmine = 
            counts.jasmine * scentAmounts.jasmine := by sorry
  sorry

end NUMINAMATH_CALUDE_jills_lavender_candles_l812_81214


namespace NUMINAMATH_CALUDE_triangle_properties_l812_81241

-- Define the triangle ABC
structure Triangle :=
  (A B C : Real)  -- Angles
  (a b c : Real)  -- Sides opposite to angles A, B, C respectively
  (S : Real)      -- Area

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  Real.sqrt 3 * (t.c * t.a * Real.cos t.C) = 2 * t.S

def condition2 (t : Triangle) : Prop :=
  (Real.sin t.C + Real.sin t.A) * (Real.sin t.C - Real.sin t.A) = 
  Real.sin t.B * (Real.sin t.B - Real.sin t.A)

def condition3 (t : Triangle) : Prop :=
  (2 * t.a - t.b) * Real.cos t.C = t.c * Real.cos t.B

-- Theorem statement
theorem triangle_properties (t : Triangle) :
  (condition1 t ∨ condition2 t ∨ condition3 t) →
  t.C = Real.pi / 3 ∧
  (t.c = 2 → t.S ≤ Real.sqrt 3 ∧ 
   ∃ (t' : Triangle), t'.c = 2 ∧ t'.S = Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l812_81241


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l812_81250

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0, b > 0, 
    and eccentricity e = 2, prove that its asymptotes are y = ± √3 * x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := 2
  let hyperbola := fun (x y : ℝ) ↦ x^2 / a^2 - y^2 / b^2 = 1
  let eccentricity := fun (c : ℝ) ↦ c / a = e
  let asymptotes := fun (x y : ℝ) ↦ y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x
  ∃ c, eccentricity c ∧ (∀ x y, asymptotes x y ↔ (hyperbola x y ∧ x ≠ 0)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l812_81250


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l812_81277

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h_sum : a + 3*b = 1) :
  (1/a + 1/b) ≥ 4*Real.sqrt 3 + 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l812_81277


namespace NUMINAMATH_CALUDE_sunzi_wood_measurement_l812_81253

theorem sunzi_wood_measurement 
  (x y : ℝ) 
  (h1 : y - x = 4.5) 
  (h2 : y / 2 = x - 1) : 
  (y - x = 4.5) ∧ (y / 2 = x - 1) := by
sorry

end NUMINAMATH_CALUDE_sunzi_wood_measurement_l812_81253


namespace NUMINAMATH_CALUDE_solve_for_b_l812_81273

theorem solve_for_b (y : ℝ) (b : ℝ) (h1 : y > 0) 
  (h2 : (6 * y) / b + (3 * y) / 10 = 0.60 * y) : b = 20 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_b_l812_81273


namespace NUMINAMATH_CALUDE_part_one_part_two_l812_81249

-- Define the sets A, B, and C
def A : Set ℝ := {x | x^2 + 2*x - 8 ≥ 0}
def B : Set ℝ := {x | Real.sqrt (9 - 3*x) ≤ Real.sqrt (2*x + 19)}
def C (a : ℝ) : Set ℝ := {x | x^2 + 2*a*x + 2 ≤ 0}

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Theorem for part (1)
theorem part_one (b c : ℝ) : 
  (A ∩ B = {x | b*x^2 + 10*x + c ≥ 0}) → (b = -2 ∧ c = -12) := by sorry

-- Theorem for part (2)
theorem part_two (a : ℝ) : 
  (C a ⊆ B ∪ (U \ A)) → (a ≥ -11/6 ∧ a ≤ 9/4) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l812_81249


namespace NUMINAMATH_CALUDE_imaginary_sum_equals_negative_one_l812_81295

theorem imaginary_sum_equals_negative_one (i : ℂ) (hi : i^2 = -1) : 
  i^10 + i^20 + i^34 = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_sum_equals_negative_one_l812_81295


namespace NUMINAMATH_CALUDE_hair_extension_length_l812_81266

def original_length : ℕ := 18

def extension_factor : ℕ := 2

theorem hair_extension_length : 
  original_length * extension_factor = 36 := by sorry

end NUMINAMATH_CALUDE_hair_extension_length_l812_81266


namespace NUMINAMATH_CALUDE_cereal_eating_time_l812_81251

/-- The time it takes for two people to eat a certain amount of cereal together -/
def eat_time (swift_rate : ℚ) (slow_rate : ℚ) (total_amount : ℚ) : ℚ :=
  total_amount / (swift_rate + slow_rate)

/-- Theorem: Mr. Swift and Mr. Slow will take 45 minutes to eat 4 pounds of cereal together -/
theorem cereal_eating_time :
  let swift_rate : ℚ := 1 / 15  -- Mr. Swift's eating rate in pounds per minute
  let slow_rate : ℚ := 1 / 45   -- Mr. Slow's eating rate in pounds per minute
  let total_amount : ℚ := 4     -- Total amount of cereal in pounds
  eat_time swift_rate slow_rate total_amount = 45 := by
sorry

end NUMINAMATH_CALUDE_cereal_eating_time_l812_81251


namespace NUMINAMATH_CALUDE_block_count_is_eight_l812_81297

/-- Represents the orthographic views of a geometric body -/
structure OrthographicViews where
  front : Nat
  top : Nat
  side : Nat

/-- Calculates the number of blocks in a geometric body based on its orthographic views -/
def countBlocks (views : OrthographicViews) : Nat :=
  sorry

/-- The specific orthographic views for the given problem -/
def problemViews : OrthographicViews :=
  { front := 6, top := 6, side := 4 }

/-- Theorem stating that the number of blocks for the given views is 8 -/
theorem block_count_is_eight :
  countBlocks problemViews = 8 := by
  sorry

end NUMINAMATH_CALUDE_block_count_is_eight_l812_81297


namespace NUMINAMATH_CALUDE_stan_playlist_sufficient_stan_playlist_sufficient_proof_l812_81258

theorem stan_playlist_sufficient (total_run_time : ℕ) 
  (songs_3min songs_4min songs_6min : ℕ) 
  (max_songs_per_category : ℕ) 
  (min_favorite_songs : ℕ) 
  (favorite_song_length : ℕ) : Prop :=
  total_run_time = 90 ∧
  songs_3min ≥ 10 ∧
  songs_4min ≥ 12 ∧
  songs_6min ≥ 15 ∧
  max_songs_per_category = 7 ∧
  min_favorite_songs = 3 ∧
  favorite_song_length = 4 →
  ∃ (playlist_3min playlist_4min playlist_6min : ℕ),
    playlist_3min ≤ max_songs_per_category ∧
    playlist_4min ≤ max_songs_per_category ∧
    playlist_6min ≤ max_songs_per_category ∧
    playlist_4min ≥ min_favorite_songs ∧
    playlist_3min * 3 + playlist_4min * 4 + playlist_6min * 6 ≥ total_run_time

theorem stan_playlist_sufficient_proof : stan_playlist_sufficient 90 10 12 15 7 3 4 := by
  sorry

end NUMINAMATH_CALUDE_stan_playlist_sufficient_stan_playlist_sufficient_proof_l812_81258


namespace NUMINAMATH_CALUDE_bag_original_price_l812_81299

theorem bag_original_price (sale_price : ℝ) (discount_percentage : ℝ) : 
  sale_price = 135 → discount_percentage = 10 → 
  sale_price = (1 - discount_percentage / 100) * 150 := by
  sorry

end NUMINAMATH_CALUDE_bag_original_price_l812_81299


namespace NUMINAMATH_CALUDE_double_counted_page_l812_81239

theorem double_counted_page (n : ℕ) (m : ℕ) : 
  n > 0 ∧ m > 0 ∧ m ≤ n ∧ (n * (n + 1)) / 2 + m = 2040 → m = 24 := by
  sorry

end NUMINAMATH_CALUDE_double_counted_page_l812_81239


namespace NUMINAMATH_CALUDE_pitcher_juice_distribution_l812_81292

theorem pitcher_juice_distribution (C : ℝ) (h : C > 0) : 
  let total_juice := (2/3) * C
  let cups := 6
  let juice_per_cup := total_juice / cups
  (juice_per_cup / C) * 100 = 100/9 := by sorry

end NUMINAMATH_CALUDE_pitcher_juice_distribution_l812_81292


namespace NUMINAMATH_CALUDE_sum_of_f_always_positive_l812_81265

/-- A monotonically increasing odd function -/
def MonoIncreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) = -f x)

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 2) - a (n + 1) = a (n + 1) - a n

/-- Main theorem -/
theorem sum_of_f_always_positive
  (f : ℝ → ℝ) (a : ℕ → ℝ)
  (hf : MonoIncreasingOddFunction f)
  (ha : ArithmeticSequence a)
  (ha3_pos : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end NUMINAMATH_CALUDE_sum_of_f_always_positive_l812_81265


namespace NUMINAMATH_CALUDE_fractional_equation_simplification_l812_81226

theorem fractional_equation_simplification (x : ℝ) (h : x ≠ 1 ∧ x ≠ 1) : 
  (x / (x - 1) = 3 / (2 * x - 2) - 3) ↔ (2 * x = 3 - 6 * x + 6) :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_simplification_l812_81226


namespace NUMINAMATH_CALUDE_odd_function_and_inequality_l812_81291

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + a * x^2) + 2 * x)

theorem odd_function_and_inequality (a m : ℝ) : 
  (∀ x, f a x = -f a (-x)) ∧ 
  (∀ x, f a (2 * m - m * Real.sin x) + f a (Real.cos x)^2 ≥ 0) →
  a = 4 ∧ m ≥ 0 := by sorry

end NUMINAMATH_CALUDE_odd_function_and_inequality_l812_81291


namespace NUMINAMATH_CALUDE_smoking_lung_disease_relation_l812_81293

/-- Represents the Chi-square statistic -/
def K_squared : ℝ := 5.231

/-- The probability that K² is greater than or equal to 3.841 -/
def p_value_95 : ℝ := 0.05

/-- The probability that K² is greater than or equal to 6.635 -/
def p_value_99 : ℝ := 0.01

/-- Confidence level for the relationship between smoking and lung disease -/
def confidence_level : ℝ := 1 - p_value_95

theorem smoking_lung_disease_relation :
  K_squared ≥ 3.841 ∧ K_squared < 6.635 →
  confidence_level > 0.95 :=
sorry

end NUMINAMATH_CALUDE_smoking_lung_disease_relation_l812_81293


namespace NUMINAMATH_CALUDE_thirty_blocks_placeable_l812_81224

/-- Represents a chessboard with two opposite corners removed -/
structure ModifiedChessboard :=
  (size : Nat)
  (cornersRemoved : Nat)

/-- Represents a rectangular block -/
structure Block :=
  (length : Nat)
  (width : Nat)

/-- Calculates the number of blocks that can be placed on the modified chessboard -/
def countPlaceableBlocks (board : ModifiedChessboard) (block : Block) : Nat :=
  sorry

/-- Theorem stating that 30 blocks can be placed on the modified 8x8 chessboard -/
theorem thirty_blocks_placeable :
  ∀ (board : ModifiedChessboard) (block : Block),
    board.size = 8 ∧ 
    board.cornersRemoved = 2 ∧ 
    block.length = 2 ∧ 
    block.width = 1 →
    countPlaceableBlocks board block = 30 :=
  sorry

end NUMINAMATH_CALUDE_thirty_blocks_placeable_l812_81224


namespace NUMINAMATH_CALUDE_bobby_candy_count_l812_81238

/-- The number of candy pieces Bobby had initially -/
def initial_candy : ℕ := 22

/-- The number of candy pieces Bobby ate at the start -/
def eaten_start : ℕ := 9

/-- The number of additional candy pieces Bobby ate -/
def eaten_additional : ℕ := 5

/-- The number of candy pieces Bobby has left -/
def remaining_candy : ℕ := 8

/-- Theorem stating that Bobby's initial candy count is correct -/
theorem bobby_candy_count : 
  initial_candy = eaten_start + eaten_additional + remaining_candy :=
by sorry

end NUMINAMATH_CALUDE_bobby_candy_count_l812_81238


namespace NUMINAMATH_CALUDE_event_A_not_random_l812_81296

-- Define the type for events
inductive Event
| A : Event  -- The sun rises in the east and it rains in the west
| B : Event  -- It's not cold when it snows but cold when the snow melts
| C : Event  -- It often rains during the Qingming festival
| D : Event  -- It's sunny every day when the plums turn yellow

-- Define what it means for an event to be random
def isRandomEvent (e : Event) : Prop := sorry

-- Define what it means for an event to be based on natural laws
def isBasedOnNaturalLaws (e : Event) : Prop := sorry

-- Axiom: Events based on natural laws are not random
axiom natural_law_not_random : ∀ (e : Event), isBasedOnNaturalLaws e → ¬isRandomEvent e

-- Theorem: Event A is not a random event
theorem event_A_not_random : ¬isRandomEvent Event.A := by
  sorry

end NUMINAMATH_CALUDE_event_A_not_random_l812_81296


namespace NUMINAMATH_CALUDE_b_four_lt_b_seven_l812_81222

def b (α : ℕ → ℕ) : ℕ → ℚ
  | 0 => 1
  | n + 1 => 1 + 1 / (b α n + 1 / α (n + 1))

theorem b_four_lt_b_seven (α : ℕ → ℕ) : b α 4 < b α 7 := by
  sorry

end NUMINAMATH_CALUDE_b_four_lt_b_seven_l812_81222


namespace NUMINAMATH_CALUDE_probability_prime_sum_two_dice_l812_81205

def die_sides : ℕ := 8

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def sum_is_prime (a b : ℕ) : Prop := is_prime (a + b)

def favorable_outcomes : ℕ := 23

def total_outcomes : ℕ := die_sides * die_sides

theorem probability_prime_sum_two_dice :
  (favorable_outcomes : ℚ) / total_outcomes = 23 / 64 := by sorry

end NUMINAMATH_CALUDE_probability_prime_sum_two_dice_l812_81205


namespace NUMINAMATH_CALUDE_point_transformation_final_coordinates_l812_81284

/-- Point in 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Function to find the symmetric point about the origin -/
def symmetricAboutOrigin (p : Point) : Point :=
  { x := -p.x, y := -p.y }

/-- Function to move a point to the left -/
def moveLeft (p : Point) (units : ℝ) : Point :=
  { x := p.x - units, y := p.y }

/-- Theorem stating that the given transformations result in the expected point -/
theorem point_transformation (initialPoint : Point) :
  initialPoint.x = -2 ∧ initialPoint.y = 3 →
  (moveLeft (symmetricAboutOrigin initialPoint) 2).x = 0 ∧
  (moveLeft (symmetricAboutOrigin initialPoint) 2).y = -3 := by
  sorry

/-- Main theorem proving the final coordinates -/
theorem final_coordinates : ∃ (p : Point),
  p.x = -2 ∧ p.y = 3 ∧
  (moveLeft (symmetricAboutOrigin p) 2).x = 0 ∧
  (moveLeft (symmetricAboutOrigin p) 2).y = -3 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_final_coordinates_l812_81284


namespace NUMINAMATH_CALUDE_supplement_triple_angle_l812_81231

theorem supplement_triple_angle : ∃ (x : ℝ), x > 0 ∧ x < 180 ∧ x = 3 * (180 - x) ∧ x = 135 := by
  sorry

end NUMINAMATH_CALUDE_supplement_triple_angle_l812_81231


namespace NUMINAMATH_CALUDE_triangle_angle_measure_l812_81240

/-- Given a triangle ABD where angle ABC is a straight angle (180°),
    angle CBD is 133°, and one angle in triangle ABD is 31°,
    prove that the measure of the remaining angle y in triangle ABD is 102°. -/
theorem triangle_angle_measure (angle_CBD : ℝ) (angle_in_ABD : ℝ) :
  angle_CBD = 133 →
  angle_in_ABD = 31 →
  let angle_ABD : ℝ := 180 - angle_CBD
  180 - (angle_ABD + angle_in_ABD) = 102 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_measure_l812_81240


namespace NUMINAMATH_CALUDE_simplify_sqrt_2_simplify_complex_sqrt_l812_81276

-- Part 1
theorem simplify_sqrt_2 : 2 * Real.sqrt 2 - Real.sqrt 2 = Real.sqrt 2 := by
  sorry

-- Part 2
theorem simplify_complex_sqrt : 
  Real.sqrt 2 * Real.sqrt 10 / (1 / Real.sqrt 5) = 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_2_simplify_complex_sqrt_l812_81276


namespace NUMINAMATH_CALUDE_zachary_did_19_pushups_l812_81263

/-- The number of push-ups Zachary did -/
def zachary_pushups : ℕ := 19

/-- The number of push-ups David did -/
def david_pushups : ℕ := 58

/-- The difference between David's and Zachary's push-ups -/
def difference : ℕ := 39

/-- Theorem stating that Zachary did 19 push-ups given the conditions -/
theorem zachary_did_19_pushups : zachary_pushups = 19 := by sorry

end NUMINAMATH_CALUDE_zachary_did_19_pushups_l812_81263


namespace NUMINAMATH_CALUDE_shaded_area_of_semicircles_l812_81259

/-- The shaded area of semicircles in a pattern --/
theorem shaded_area_of_semicircles (d : ℝ) (l : ℝ) : 
  d = 3 → l = 24 → (l / d) * (π * d^2 / 8) = 18 * π := by sorry

end NUMINAMATH_CALUDE_shaded_area_of_semicircles_l812_81259


namespace NUMINAMATH_CALUDE_textbook_weight_difference_l812_81298

theorem textbook_weight_difference :
  let chemistry_weight : Float := 7.12
  let geometry_weight : Float := 0.62
  (chemistry_weight - geometry_weight) = 6.50 := by
  sorry

end NUMINAMATH_CALUDE_textbook_weight_difference_l812_81298


namespace NUMINAMATH_CALUDE_power_of_1307_squared_cubed_l812_81206

theorem power_of_1307_squared_cubed : (1307 * 1307)^3 = 4984209203082045649 := by
  sorry

end NUMINAMATH_CALUDE_power_of_1307_squared_cubed_l812_81206


namespace NUMINAMATH_CALUDE_profit_and_max_profit_l812_81207

/-- The daily sales quantity as a function of selling price -/
def sales_quantity (x : ℝ) : ℝ := -10 * x + 300

/-- The daily profit as a function of selling price -/
def profit (x : ℝ) : ℝ := (x - 10) * sales_quantity x

theorem profit_and_max_profit :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ profit x₁ = 750 ∧ profit x₂ = 750 ∧ 
    ((∀ x : ℝ, profit x = 750 → x = x₁ ∨ x = x₂) ∧ 
    (x₁ = 15 ∨ x₁ = 25) ∧ (x₂ = 15 ∨ x₂ = 25))) ∧
  (∃ max_profit : ℝ, max_profit = 1000 ∧ ∀ x : ℝ, profit x ≤ max_profit) :=
by sorry

end NUMINAMATH_CALUDE_profit_and_max_profit_l812_81207


namespace NUMINAMATH_CALUDE_hiking_problem_l812_81227

/-- Hiking problem statement -/
theorem hiking_problem (endpoint_distance : ℝ) (speed_ratio : ℝ) (head_start : ℝ) (meet_time : ℝ) 
  (planned_time : ℝ) (early_arrival : ℝ) :
  endpoint_distance = 7.5 →
  speed_ratio = 1.5 →
  head_start = 0.75 →
  meet_time = 0.5 →
  planned_time = 1 →
  early_arrival = 1/6 →
  ∃ (speed_a speed_b actual_time : ℝ),
    speed_a = 4.5 ∧
    speed_b = 3 ∧
    actual_time = 4/3 ∧
    speed_a = speed_ratio * speed_b ∧
    (speed_a - speed_b) * meet_time = head_start ∧
    endpoint_distance / speed_b - early_arrival = planned_time + (endpoint_distance - speed_b * planned_time) / speed_a :=
by sorry


end NUMINAMATH_CALUDE_hiking_problem_l812_81227


namespace NUMINAMATH_CALUDE_sets_properties_l812_81229

-- Define the sets A, B, and C
def A : Set ℝ := {x : ℝ | ∃ y, y = x^2 + 1}
def B : Set ℝ := {y : ℝ | ∃ x, y = x^2 + 1}
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1^2 + 1}

-- Theorem stating the properties of A, B, and C
theorem sets_properties :
  (A = Set.univ) ∧
  (B = {y : ℝ | y ≥ 1}) ∧
  (C = {p : ℝ × ℝ | p.2 = p.1^2 + 1}) :=
by sorry

end NUMINAMATH_CALUDE_sets_properties_l812_81229


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l812_81209

theorem contrapositive_equivalence (x y : ℝ) :
  (¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) ↔ (x^2 + y^2 = 0 → x = 0 ∧ y = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l812_81209


namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_l812_81242

theorem consecutive_numbers_sum (n : ℕ) : 
  n + (n + 1) + (n + 2) = 60 → 
  (n + 2) + (n + 3) + (n + 4) = 66 := by
sorry

end NUMINAMATH_CALUDE_consecutive_numbers_sum_l812_81242


namespace NUMINAMATH_CALUDE_tiger_count_l812_81270

/-- Given a zoo where the ratio of lions to tigers is 3:4 and there are 21 lions, 
    prove that the number of tigers is 28. -/
theorem tiger_count (lion_count : ℕ) (tiger_count : ℕ) : 
  (lion_count : ℚ) / tiger_count = 3 / 4 → 
  lion_count = 21 → 
  tiger_count = 28 := by
  sorry

end NUMINAMATH_CALUDE_tiger_count_l812_81270


namespace NUMINAMATH_CALUDE_refrigerator_price_l812_81290

theorem refrigerator_price (P : ℝ) 
  (selling_price : P + 0.1 * P = 23100)
  (discount : ℝ := 0.2)
  (transport_cost : ℝ := 125)
  (installation_cost : ℝ := 250) :
  P * (1 - discount) + transport_cost + installation_cost = 17175 := by
sorry

end NUMINAMATH_CALUDE_refrigerator_price_l812_81290


namespace NUMINAMATH_CALUDE_favorite_pet_dog_l812_81267

theorem favorite_pet_dog (total : ℕ) (cat fish bird other : ℕ) 
  (h_total : total = 90)
  (h_cat : cat = 25)
  (h_fish : fish = 10)
  (h_bird : bird = 15)
  (h_other : other = 5) :
  total - (cat + fish + bird + other) = 35 := by
  sorry

end NUMINAMATH_CALUDE_favorite_pet_dog_l812_81267


namespace NUMINAMATH_CALUDE_horner_method_polynomial_evaluation_l812_81218

def f (x : ℝ) : ℝ := 2*x^5 - 5*x^4 - 4*x^3 + 3*x^2 - 6*x + 7

theorem horner_method_polynomial_evaluation :
  f 5 = 2677 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_polynomial_evaluation_l812_81218


namespace NUMINAMATH_CALUDE_days_to_reach_goal_chris_breath_holding_days_l812_81272

/-- Given Chris's breath-holding capacity and improvement rate, calculate the number of days to reach his goal. -/
theorem days_to_reach_goal (start_capacity : ℕ) (daily_improvement : ℕ) (goal : ℕ) : ℕ :=
  let days := (goal - start_capacity) / daily_improvement
  days

/-- Prove that Chris needs 6 more days to reach his goal. -/
theorem chris_breath_holding_days : days_to_reach_goal 30 10 90 = 6 := by
  sorry

end NUMINAMATH_CALUDE_days_to_reach_goal_chris_breath_holding_days_l812_81272


namespace NUMINAMATH_CALUDE_projection_problem_l812_81217

def v (z : ℝ) : Fin 3 → ℝ := ![4, -1, z]
def u : Fin 3 → ℝ := ![6, -2, 3]

theorem projection_problem (z : ℝ) : 
  (v z • u) / (u • u) = 20 / 49 → z = -2 := by sorry

end NUMINAMATH_CALUDE_projection_problem_l812_81217


namespace NUMINAMATH_CALUDE_chess_match_outcomes_count_l812_81244

/-- The number of different possible outcomes for a chess match draw -/
def chessMatchOutcomes : ℕ :=
  2^8 * Nat.factorial 8

/-- Theorem stating the number of different possible outcomes for a chess match draw -/
theorem chess_match_outcomes_count :
  chessMatchOutcomes = 2^8 * Nat.factorial 8 := by
  sorry

#eval chessMatchOutcomes

end NUMINAMATH_CALUDE_chess_match_outcomes_count_l812_81244


namespace NUMINAMATH_CALUDE_unique_solution_l812_81203

/-- A quadratic polynomial with exactly one root -/
structure UniqueRootQuadratic where
  a : ℝ
  b : ℝ
  has_unique_root : ∃! x : ℝ, x^2 + a * x + b = 0

/-- The composite polynomial with exactly one root -/
def composite_poly (g : UniqueRootQuadratic) (x : ℝ) : ℝ :=
  g.a * (x^5 + 2*x - 1) + g.b + g.a * (x^5 + 3*x + 1) + g.b

/-- Theorem stating the unique solution for a and b -/
theorem unique_solution (g : UniqueRootQuadratic) 
  (h : ∃! x : ℝ, composite_poly g x = 0) : 
  g.a = 74 ∧ g.b = 1369 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l812_81203


namespace NUMINAMATH_CALUDE_max_students_distribution_l812_81262

theorem max_students_distribution (pens pencils : ℕ) 
  (h1 : pens = 2500) (h2 : pencils = 1575) : 
  Nat.gcd pens pencils = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_students_distribution_l812_81262


namespace NUMINAMATH_CALUDE_circle_area_from_polar_equation_l812_81212

/-- The area of the circle described by the polar equation r = -4 cos θ + 8 sin θ is equal to 20π. -/
theorem circle_area_from_polar_equation :
  let r : ℝ → ℝ := fun θ ↦ -4 * Real.cos θ + 8 * Real.sin θ
  ∃ c : ℝ × ℝ, ∃ radius : ℝ,
    (∀ θ : ℝ, (r θ * Real.cos θ - c.1)^2 + (r θ * Real.sin θ - c.2)^2 = radius^2) ∧
    Real.pi * radius^2 = 20 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_circle_area_from_polar_equation_l812_81212


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_k_l812_81202

/-- A polynomial is a perfect square trinomial if it can be expressed as (ax + b)^2 -/
def IsPerfectSquareTrinomial (p : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, p x = (a * x + b)^2

/-- The given polynomial -/
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 8*x + k

theorem perfect_square_trinomial_k (k : ℝ) :
  IsPerfectSquareTrinomial (f k) → k = 16 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_k_l812_81202


namespace NUMINAMATH_CALUDE_max_value_of_a_l812_81255

theorem max_value_of_a (a : ℝ) : 
  (∀ x > 1, a - x + Real.log (x * (x + 1)) ≤ 0) →
  a ≤ (1 + Real.sqrt 3) / 2 - Real.log ((3 / 2) + Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l812_81255


namespace NUMINAMATH_CALUDE_smallest_k_for_polynomial_division_l812_81219

theorem smallest_k_for_polynomial_division : 
  ∃ (k : ℕ), k > 0 ∧ 
  (∀ (z : ℂ), (z^10 + z^9 + z^6 + z^5 + z^4 + z + 1) ∣ (z^k - 1)) ∧
  (∀ (m : ℕ), m > 0 → m < k → 
    ¬(∀ (z : ℂ), (z^10 + z^9 + z^6 + z^5 + z^4 + z + 1) ∣ (z^m - 1))) ∧
  k = 84 :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_for_polynomial_division_l812_81219


namespace NUMINAMATH_CALUDE_dog_yelling_problem_l812_81247

theorem dog_yelling_problem (obedient_yells stubborn_ratio : ℕ) : 
  obedient_yells = 12 →
  stubborn_ratio = 4 →
  obedient_yells + stubborn_ratio * obedient_yells = 60 := by
  sorry

end NUMINAMATH_CALUDE_dog_yelling_problem_l812_81247


namespace NUMINAMATH_CALUDE_smallest_value_for_x_5_l812_81252

theorem smallest_value_for_x_5 (x : ℝ) (h : x = 5) :
  let a := 8 / x
  let b := 8 / (x + 2)
  let c := 8 / (x - 2)
  let d := x / 8
  let e := (x + 2) / 8
  d ≤ a ∧ d ≤ b ∧ d ≤ c ∧ d ≤ e := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_for_x_5_l812_81252


namespace NUMINAMATH_CALUDE_folded_paper_area_l812_81230

/-- The area of a folded rectangular paper -/
theorem folded_paper_area (length width : ℝ) (h_length : length = 17) (h_width : width = 8) :
  let original_area := length * width
  let folded_triangle_area := (1/2) * width * width
  original_area - folded_triangle_area = 104 :=
by
  sorry


end NUMINAMATH_CALUDE_folded_paper_area_l812_81230


namespace NUMINAMATH_CALUDE_two_numbers_sum_diff_product_l812_81286

theorem two_numbers_sum_diff_product : ∃ (x y : ℝ), 
  x + y = 24 ∧ x - y = 8 ∧ x * y > 100 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_diff_product_l812_81286


namespace NUMINAMATH_CALUDE_square_root_of_16_l812_81243

theorem square_root_of_16 : Real.sqrt 16 = 4 ∧ Real.sqrt 16 = -4 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_16_l812_81243


namespace NUMINAMATH_CALUDE_unknown_score_is_66_l812_81204

def scores : List ℕ := [65, 70, 78, 85, 92]

def is_integer (n : ℚ) : Prop := ∃ m : ℤ, n = m

theorem unknown_score_is_66 (x : ℕ) 
  (h1 : is_integer ((scores.sum + x) / 6))
  (h2 : x % 6 = 0)
  (h3 : x ≥ 60 ∧ x ≤ 100) :
  x = 66 := by sorry

end NUMINAMATH_CALUDE_unknown_score_is_66_l812_81204


namespace NUMINAMATH_CALUDE_students_between_positions_l812_81200

theorem students_between_positions (n : ℕ) (h : n = 9) : 
  (n - 2) - (3 + 1) = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_students_between_positions_l812_81200


namespace NUMINAMATH_CALUDE_simplest_fraction_sum_l812_81294

theorem simplest_fraction_sum (a b : ℕ+) : 
  (a : ℚ) / (b : ℚ) = 0.375 ∧ 
  ∀ (c d : ℕ+), (c : ℚ) / (d : ℚ) = 0.375 → a ≤ c ∧ b ≤ d → 
  a + b = 11 := by
sorry

end NUMINAMATH_CALUDE_simplest_fraction_sum_l812_81294


namespace NUMINAMATH_CALUDE_oil_remaining_l812_81221

theorem oil_remaining (x₁ x₂ x₃ : ℕ) : 
  x₁ > 0 → x₂ > 0 → x₃ > 0 →
  3 * x₁ = 2 * x₂ →
  5 * x₁ = 3 * x₃ →
  30 - (x₁ + x₂ + x₃) = 5 :=
by sorry

end NUMINAMATH_CALUDE_oil_remaining_l812_81221


namespace NUMINAMATH_CALUDE_value_of_3x_minus_y_l812_81285

-- Define the augmented matrix
def augmented_matrix : Matrix (Fin 2) (Fin 3) ℚ := !![2, 1, 5; 1, -2, 0]

-- Define the system of equations
def system_equations (x y : ℚ) : Prop :=
  2 * x + y = 5 ∧ x - 2 * y = 0

-- Theorem statement
theorem value_of_3x_minus_y :
  ∃ x y : ℚ, system_equations x y → 3 * x - y = 5 := by
  sorry

end NUMINAMATH_CALUDE_value_of_3x_minus_y_l812_81285


namespace NUMINAMATH_CALUDE_absolute_value_equality_l812_81216

theorem absolute_value_equality (m : ℝ) : |m| = |-7| → m = 7 ∨ m = -7 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_l812_81216


namespace NUMINAMATH_CALUDE_base_conversion_l812_81256

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem base_conversion :
  (binary_to_decimal [true, true, false, true, false, true]) = 43 ∧
  (decimal_to_base7 85) = [1, 5, 1] :=
sorry

end NUMINAMATH_CALUDE_base_conversion_l812_81256


namespace NUMINAMATH_CALUDE_discount_calculation_l812_81260

/-- Given a 25% discount on a purchase where the final price paid is $120, prove that the discount amount is $40. -/
theorem discount_calculation (original_price : ℝ) : 
  (original_price * 0.75 = 120) → (original_price - 120 = 40) := by
  sorry

end NUMINAMATH_CALUDE_discount_calculation_l812_81260


namespace NUMINAMATH_CALUDE_subtraction_preserves_inequality_l812_81237

theorem subtraction_preserves_inequality (a b c : ℝ) (h : a > b) : b - c < a - c := by
  sorry

end NUMINAMATH_CALUDE_subtraction_preserves_inequality_l812_81237


namespace NUMINAMATH_CALUDE_fish_count_l812_81232

theorem fish_count (total_pets dogs cats : ℕ) (h1 : total_pets = 149) (h2 : dogs = 43) (h3 : cats = 34) :
  total_pets - (dogs + cats) = 72 := by
sorry

end NUMINAMATH_CALUDE_fish_count_l812_81232


namespace NUMINAMATH_CALUDE_range_of_a_l812_81254

-- Define the sets A, B, and C
def A : Set ℝ := {x | -1 < x ∧ x < 3}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 4}
def C (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 1}

-- Theorem statement
theorem range_of_a (a : ℝ) : C a ⊆ (A ∩ B) → 0 ≤ a ∧ a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l812_81254


namespace NUMINAMATH_CALUDE_nelly_friends_count_l812_81264

def pizza_cost : ℕ := 12
def people_per_pizza : ℕ := 3
def nightly_earnings : ℕ := 4
def nights_worked : ℕ := 15

def total_earnings : ℕ := nightly_earnings * nights_worked
def pizzas_bought : ℕ := total_earnings / pizza_cost
def total_people_fed : ℕ := pizzas_bought * people_per_pizza

theorem nelly_friends_count : total_people_fed - 1 = 14 := by
  sorry

end NUMINAMATH_CALUDE_nelly_friends_count_l812_81264


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l812_81275

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A * B * C = 3003 →
  A + B + C ≤ 49 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l812_81275


namespace NUMINAMATH_CALUDE_inequality_solution_range_l812_81228

theorem inequality_solution_range :
  ∀ a : ℝ, (∃ x : ℝ, |x + 1| - |x - 3| < a) ↔ a > -4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l812_81228


namespace NUMINAMATH_CALUDE_max_ant_path_theorem_l812_81211

/-- Represents a cube with edge length 12 cm -/
structure Cube where
  edge_length : ℝ
  edge_length_eq : edge_length = 12

/-- Represents a path on the cube's edges -/
structure CubePath where
  length : ℝ
  no_repeat : Bool

/-- The maximum distance an ant can walk on the cube's edges without repetition -/
def max_ant_path (c : Cube) : ℝ := 108

/-- Theorem stating the maximum distance an ant can walk on the cube -/
theorem max_ant_path_theorem (c : Cube) :
  ∀ (path : CubePath), path.no_repeat → path.length ≤ max_ant_path c :=
by sorry

end NUMINAMATH_CALUDE_max_ant_path_theorem_l812_81211


namespace NUMINAMATH_CALUDE_max_common_roots_and_coefficients_l812_81208

/-- A polynomial of degree 2020 with non-zero coefficients -/
def Polynomial2020 : Type := { p : Polynomial ℝ // p.degree = some 2020 ∧ ∀ i, p.coeff i ≠ 0 }

/-- The number of common real roots (counting multiplicity) of two polynomials -/
noncomputable def commonRoots (P Q : Polynomial2020) : ℕ := sorry

/-- The number of common coefficients of two polynomials -/
def commonCoefficients (P Q : Polynomial2020) : ℕ := sorry

/-- The main theorem: the maximum possible value of r + s is 3029 -/
theorem max_common_roots_and_coefficients (P Q : Polynomial2020) (h : P ≠ Q) :
  commonRoots P Q + commonCoefficients P Q ≤ 3029 := by sorry

end NUMINAMATH_CALUDE_max_common_roots_and_coefficients_l812_81208


namespace NUMINAMATH_CALUDE_mowing_time_ab_l812_81246

/-- The time (in days) taken by a and b together to mow the field -/
def time_ab : ℝ := 28

/-- The time (in days) taken by a, b, and c together to mow the field -/
def time_abc : ℝ := 21

/-- The time (in days) taken by c alone to mow the field -/
def time_c : ℝ := 84

/-- Theorem stating that the time taken by a and b to mow the field together is 28 days -/
theorem mowing_time_ab :
  time_ab = 28 ∧
  (1 / time_ab + 1 / time_c = 1 / time_abc) :=
sorry

end NUMINAMATH_CALUDE_mowing_time_ab_l812_81246


namespace NUMINAMATH_CALUDE_min_value_theorem_l812_81278

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^2 + a*b + a*c + b*c = 4) :
  ∀ x y z, x > 0 → y > 0 → z > 0 → x^2 + x*y + x*z + y*z = 4 →
  2*a + b + c ≤ 2*x + y + z :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l812_81278


namespace NUMINAMATH_CALUDE_circle_equation_radius_l812_81236

theorem circle_equation_radius (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 12*x + y^2 + 8*y - k = 0 ↔ (x + 6)^2 + (y + 4)^2 = 49) → 
  k = -3 := by
sorry

end NUMINAMATH_CALUDE_circle_equation_radius_l812_81236


namespace NUMINAMATH_CALUDE_trajectory_of_point_m_l812_81271

/-- The trajectory of point M on a line segment AB with given conditions -/
theorem trajectory_of_point_m (a : ℝ) (x y : ℝ) :
  (∃ (m b : ℝ),
    -- AB has length 2a
    m^2 + b^2 = (2*a)^2 ∧
    -- M divides AB in ratio 1:2
    x = (2/3) * m ∧
    y = (1/3) * b) →
  x^2 / ((4/3 * a)^2) + y^2 / ((2/3 * a)^2) = 1 :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_point_m_l812_81271


namespace NUMINAMATH_CALUDE_max_daily_revenue_l812_81287

def sales_price (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 70
  else 0

def sales_volume (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

def daily_revenue (t : ℕ) : ℝ :=
  sales_price t * sales_volume t

theorem max_daily_revenue :
  ∃ (t : ℕ), t = 25 ∧ daily_revenue t = 1125 ∧
  ∀ (s : ℕ), 0 < s ∧ s ≤ 30 → daily_revenue s ≤ daily_revenue t :=
by sorry

end NUMINAMATH_CALUDE_max_daily_revenue_l812_81287


namespace NUMINAMATH_CALUDE_probability_at_least_one_success_l812_81268

theorem probability_at_least_one_success (p : ℝ) (n : ℕ) (h1 : p = 3/10) (h2 : n = 2) :
  1 - (1 - p)^n = 51/100 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_success_l812_81268


namespace NUMINAMATH_CALUDE_exists_x_in_interval_equivalence_of_statements_l812_81248

-- Define the logarithm function with base 1/2
noncomputable def log_half (x : ℝ) := Real.log x / Real.log (1/2)

-- Theorem 1
theorem exists_x_in_interval : ∃ x : ℝ, x ∈ Set.Ioo 0 1 ∧ (1/2)^x > log_half x := by sorry

-- Theorem 2
theorem equivalence_of_statements :
  (∀ x : ℝ, x ∈ Set.Ioo 1 5 → x + 1/x ≥ 2) ↔
  (∀ x : ℝ, x ∈ Set.Iic 1 ∪ Set.Ici 5 → x + 1/x < 2) := by sorry

end NUMINAMATH_CALUDE_exists_x_in_interval_equivalence_of_statements_l812_81248


namespace NUMINAMATH_CALUDE_salary_left_at_month_end_l812_81269

/-- Represents the fraction of salary left after each step of the month --/
structure SalaryFraction where
  value : ℝ
  is_fraction : 0 ≤ value ∧ value ≤ 1

/-- Calculates the remaining salary fraction after tax deduction --/
def after_tax (tax_rate : ℝ) : SalaryFraction :=
  { value := 1 - tax_rate,
    is_fraction := by sorry }

/-- Calculates the remaining salary fraction after spending --/
def after_spending (s : SalaryFraction) (spend_rate : ℝ) : SalaryFraction :=
  { value := s.value * (1 - spend_rate),
    is_fraction := by sorry }

/-- Calculates the remaining salary fraction after an expense based on original salary --/
def after_expense (s : SalaryFraction) (expense_rate : ℝ) : SalaryFraction :=
  { value := s.value - expense_rate,
    is_fraction := by sorry }

/-- Theorem stating the fraction of salary left at the end of the month --/
theorem salary_left_at_month_end :
  let initial_salary := SalaryFraction.mk 1 (by sorry)
  let after_tax := after_tax 0.15
  let after_week1 := after_spending after_tax 0.25
  let after_week2 := after_spending after_week1 0.3
  let after_week3 := after_expense after_week2 0.2
  let final_salary := after_spending after_week3 0.1
  final_salary.value = 0.221625 := by sorry

end NUMINAMATH_CALUDE_salary_left_at_month_end_l812_81269
