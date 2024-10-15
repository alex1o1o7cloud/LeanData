import Mathlib

namespace NUMINAMATH_CALUDE_caterpillar_to_scorpion_ratio_l2510_251059

/-- Represents Calvin's bug collection -/
structure BugCollection where
  roaches : ℕ
  scorpions : ℕ
  crickets : ℕ
  caterpillars : ℕ

/-- Calvin's bug collection satisfies the given conditions -/
def calvins_collection : BugCollection where
  roaches := 12
  scorpions := 3
  crickets := 6  -- half as many crickets as roaches
  caterpillars := 6  -- to be proven

theorem caterpillar_to_scorpion_ratio (c : BugCollection) 
  (h1 : c.roaches = 12)
  (h2 : c.scorpions = 3)
  (h3 : c.crickets = c.roaches / 2)
  (h4 : c.roaches + c.scorpions + c.crickets + c.caterpillars = 27) :
  c.caterpillars / c.scorpions = 2 := by
  sorry

#check caterpillar_to_scorpion_ratio calvins_collection

end NUMINAMATH_CALUDE_caterpillar_to_scorpion_ratio_l2510_251059


namespace NUMINAMATH_CALUDE_triangle_with_perimeter_7_l2510_251026

def is_valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_with_perimeter_7 :
  ∀ a b c : ℕ,
  a + b + c = 7 →
  is_valid_triangle a b c →
  (a = 1 ∨ a = 2 ∨ a = 3) ∧
  (b = 1 ∨ b = 2 ∨ b = 3) ∧
  (c = 1 ∨ c = 2 ∨ c = 3) :=
by sorry

end NUMINAMATH_CALUDE_triangle_with_perimeter_7_l2510_251026


namespace NUMINAMATH_CALUDE_ratio_arithmetic_sequence_property_l2510_251078

/-- Definition of a ratio arithmetic sequence -/
def is_ratio_arithmetic (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → a (n + 2) / a (n + 1) - a (n + 1) / a n = d

/-- Theorem about the specific ratio arithmetic sequence -/
theorem ratio_arithmetic_sequence_property (a : ℕ → ℝ) (d : ℝ) :
  is_ratio_arithmetic a d →
  a 1 = 1 →
  a 2 = 1 →
  a 3 = 2 →
  a 2009 / a 2006 = 2006 := by
sorry

end NUMINAMATH_CALUDE_ratio_arithmetic_sequence_property_l2510_251078


namespace NUMINAMATH_CALUDE_max_m_value_max_m_is_maximum_l2510_251056

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- State the theorem
theorem max_m_value (m : ℝ) : 
  (∃ t : ℝ, ∀ x ∈ Set.Icc 2 m, f (x + t) ≤ 2 * x) → 
  m ≤ 8 ∧ ∃ t : ℝ, ∀ x ∈ Set.Icc 2 8, f (x + t) ≤ 2 * x :=
sorry

-- Define the maximum value of m
def max_m : ℝ := 8

-- Prove that max_m is indeed the maximum value
theorem max_m_is_maximum :
  (∃ t : ℝ, ∀ x ∈ Set.Icc 2 max_m, f (x + t) ≤ 2 * x) ∧
  ∀ m > max_m, ¬(∃ t : ℝ, ∀ x ∈ Set.Icc 2 m, f (x + t) ≤ 2 * x) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_max_m_is_maximum_l2510_251056


namespace NUMINAMATH_CALUDE_cartons_packed_l2510_251084

theorem cartons_packed (total_cups : ℕ) (cups_per_box : ℕ) (boxes_per_carton : ℕ) 
  (h1 : total_cups = 768) 
  (h2 : cups_per_box = 12) 
  (h3 : boxes_per_carton = 8) : 
  total_cups / (cups_per_box * boxes_per_carton) = 8 := by
  sorry

end NUMINAMATH_CALUDE_cartons_packed_l2510_251084


namespace NUMINAMATH_CALUDE_tangent_slope_at_pi_over_six_l2510_251025

noncomputable def f (x : ℝ) : ℝ := (1/2) * x - 2 * Real.cos x

theorem tangent_slope_at_pi_over_six :
  deriv f (π/6) = 3/2 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_at_pi_over_six_l2510_251025


namespace NUMINAMATH_CALUDE_wages_theorem_l2510_251054

/-- 
Given:
- A sum of money can pay A's wages for 20 days
- The same sum of money can pay B's wages for 30 days

Prove:
The same sum of money can pay both A and B's wages together for 12 days
-/
theorem wages_theorem (A B : ℝ) (h1 : 20 * A = 30 * B) : 
  12 * (A + B) = 20 * A := by sorry

end NUMINAMATH_CALUDE_wages_theorem_l2510_251054


namespace NUMINAMATH_CALUDE_sqrt_115_between_consecutive_integers_product_l2510_251020

theorem sqrt_115_between_consecutive_integers_product :
  ∃ (n : ℕ), n > 0 ∧ (n : ℝ) < Real.sqrt 115 ∧ Real.sqrt 115 < (n + 1) ∧ n * (n + 1) = 110 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_115_between_consecutive_integers_product_l2510_251020


namespace NUMINAMATH_CALUDE_circle_C_properties_l2510_251080

-- Define the circle C
def circle_C (x y k : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y - k = 0

-- Define the center of the circle
def center_of_circle (h k : ℝ) : Prop := 
  ∀ x y k, circle_C x y k ↔ (x - h)^2 + (y - k)^2 = k + 5

-- Define the radius of the circle
def radius_of_circle (r k : ℝ) : Prop := 
  ∀ x y, circle_C x y k ↔ (x - 1)^2 + (y + 2)^2 = r^2

-- Theorem statements
theorem circle_C_properties :
  (∀ k, (∃ x y, circle_C x y k) → k > -5) ∧
  center_of_circle 1 (-2) ∧
  radius_of_circle 3 4 ∧
  (∀ k, (∃ x, circle_C x 0 k) ∧ (∀ y, y ≠ 0 → ¬circle_C x y k) → k = -1) :=
sorry

end NUMINAMATH_CALUDE_circle_C_properties_l2510_251080


namespace NUMINAMATH_CALUDE_quadratic_decreasing_before_vertex_l2510_251092

def f (x : ℝ) : ℝ := 2 * (x - 3)^2 + 1

theorem quadratic_decreasing_before_vertex :
  ∀ (x1 x2 : ℝ), x1 < x2 → x2 < 3 → f x1 > f x2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_decreasing_before_vertex_l2510_251092


namespace NUMINAMATH_CALUDE_group_distribution_methods_l2510_251058

theorem group_distribution_methods (total_boys : ℕ) (total_girls : ℕ)
  (group_size : ℕ) (boys_per_group : ℕ) (girls_per_group : ℕ) :
  total_boys = 6 →
  total_girls = 4 →
  group_size = 5 →
  boys_per_group = 3 →
  girls_per_group = 2 →
  (Nat.choose total_boys boys_per_group * Nat.choose total_girls girls_per_group) / 2 = 60 :=
by sorry

end NUMINAMATH_CALUDE_group_distribution_methods_l2510_251058


namespace NUMINAMATH_CALUDE_framing_for_enlarged_picture_l2510_251057

/-- Calculates the minimum number of linear feet of framing needed for an enlarged picture with border -/
def minimum_framing_feet (original_width original_height enlargement_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlargement_factor
  let enlarged_height := original_height * enlargement_factor
  let final_width := enlarged_width + 2 * border_width
  let final_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (final_width + final_height)
  (perimeter_inches + 11) / 12  -- Dividing by 12 and rounding up

/-- Theorem stating that the minimum framing needed for the given picture is 10 feet -/
theorem framing_for_enlarged_picture :
  minimum_framing_feet 5 7 4 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_framing_for_enlarged_picture_l2510_251057


namespace NUMINAMATH_CALUDE_product_of_place_values_l2510_251053

def numeral : ℚ := 8712480.83

theorem product_of_place_values :
  let millions := 8000000
  let thousands := 8000
  let tenths := 0.8
  millions * thousands * tenths = 51200000000 :=
by sorry

end NUMINAMATH_CALUDE_product_of_place_values_l2510_251053


namespace NUMINAMATH_CALUDE_shoes_discount_percentage_l2510_251099

/-- Given the original price and sale price of an item, calculate the discount percentage. -/
def discount_percentage (original_price sale_price : ℚ) : ℚ :=
  (original_price - sale_price) / original_price * 100

/-- Theorem: The discount percentage for shoes with original price $204 and sale price $51 is 75%. -/
theorem shoes_discount_percentage :
  discount_percentage 204 51 = 75 := by sorry

end NUMINAMATH_CALUDE_shoes_discount_percentage_l2510_251099


namespace NUMINAMATH_CALUDE_fourth_vertex_of_complex_rectangle_l2510_251014

/-- A rectangle in the complex plane --/
structure ComplexRectangle where
  a : ℂ
  b : ℂ
  c : ℂ
  d : ℂ
  is_rectangle : (b - a).arg.cos * (c - b).arg.cos + (b - a).arg.sin * (c - b).arg.sin = 0

/-- The theorem stating that given three vertices of a rectangle in the complex plane,
    we can determine the fourth vertex --/
theorem fourth_vertex_of_complex_rectangle (r : ComplexRectangle)
  (h1 : r.a = 3 + 2*I)
  (h2 : r.b = 1 + I)
  (h3 : r.c = -1 - 2*I) :
  r.d = -3 - 3*I := by
  sorry

#check fourth_vertex_of_complex_rectangle

end NUMINAMATH_CALUDE_fourth_vertex_of_complex_rectangle_l2510_251014


namespace NUMINAMATH_CALUDE_power_of_sum_equals_225_l2510_251083

theorem power_of_sum_equals_225 : (3^2 + 6)^(4/2) = 225 := by sorry

end NUMINAMATH_CALUDE_power_of_sum_equals_225_l2510_251083


namespace NUMINAMATH_CALUDE_ambiguous_date_and_longest_periods_l2510_251048

/-- Represents a date in DD/MM format -/
structure Date :=
  (day : Nat)
  (month : Nat)

/-- Checks if a date is valid in both DD/MM and MM/DD formats -/
def Date.isAmbiguous (d : Date) : Prop :=
  d.day ≤ 12 ∧ d.month ≤ 12 ∧ d.day ≠ d.month

/-- Checks if a date is within the range of January 2nd to January 12th or December 2nd to December 12th -/
def Date.isInLongestAmbiguousPeriod (d : Date) : Prop :=
  (d.month = 1 ∧ d.day ≥ 2 ∧ d.day ≤ 12) ∨ (d.month = 12 ∧ d.day ≥ 2 ∧ d.day ≤ 12)

theorem ambiguous_date_and_longest_periods :
  (∃ d : Date, d.day = 3 ∧ d.month = 12 ∧ d.isAmbiguous) ∧
  (∀ d : Date, d.isAmbiguous → d.isInLongestAmbiguousPeriod ∨ ¬d.isInLongestAmbiguousPeriod) ∧
  (∀ d : Date, d.isInLongestAmbiguousPeriod → d.isAmbiguous) :=
sorry

end NUMINAMATH_CALUDE_ambiguous_date_and_longest_periods_l2510_251048


namespace NUMINAMATH_CALUDE_sum_of_ten_and_thousand_cube_equals_1010_scientific_notation_of_1010_sum_equals_scientific_notation_l2510_251028

theorem sum_of_ten_and_thousand_cube_equals_1010 : 10 + 10^3 = 1010 := by
  sorry

theorem scientific_notation_of_1010 : 1010 = 1.01 * 10^3 := by
  sorry

theorem sum_equals_scientific_notation : 10 + 10^3 = 1.01 * 10^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ten_and_thousand_cube_equals_1010_scientific_notation_of_1010_sum_equals_scientific_notation_l2510_251028


namespace NUMINAMATH_CALUDE_total_cost_is_correct_l2510_251072

def cement_bags : ℕ := 500
def cement_price_per_bag : ℚ := 10
def cement_discount_rate : ℚ := 5 / 100
def sand_lorries : ℕ := 20
def sand_tons_per_lorry : ℕ := 10
def sand_price_per_ton : ℚ := 40
def tax_rate_first_half : ℚ := 7 / 100
def tax_rate_second_half : ℚ := 5 / 100

def total_cost : ℚ := sorry

theorem total_cost_is_correct : 
  total_cost = 13230 := by sorry

end NUMINAMATH_CALUDE_total_cost_is_correct_l2510_251072


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_through_point_implies_b_and_eccentricity_l2510_251049

/-- A hyperbola with equation x^2 - y^2/b^2 = 1 where b > 0 -/
structure Hyperbola where
  b : ℝ
  h_pos : b > 0

/-- The asymptote of the hyperbola -/
def asymptote (h : Hyperbola) (x : ℝ) : ℝ := h.b * x

theorem hyperbola_asymptote_through_point_implies_b_and_eccentricity
  (h : Hyperbola)
  (h_asymptote : asymptote h 1 = 2) :
  h.b = 2 ∧ Real.sqrt ((1 : ℝ)^2 + h.b^2) / 1 = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_through_point_implies_b_and_eccentricity_l2510_251049


namespace NUMINAMATH_CALUDE_common_terms_count_l2510_251016

theorem common_terms_count : 
  (Finset.filter (fun k => 15 * k + 8 ≤ 2018) (Finset.range (2019 / 15 + 1))).card = 135 :=
by sorry

end NUMINAMATH_CALUDE_common_terms_count_l2510_251016


namespace NUMINAMATH_CALUDE_drilled_cube_surface_area_l2510_251037

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with a drilled tunnel -/
structure DrilledCube where
  edgeLength : ℝ
  tunnelStartDistance : ℝ

/-- Calculates the surface area of a drilled cube -/
noncomputable def surfaceArea (cube : DrilledCube) : ℝ :=
  sorry

theorem drilled_cube_surface_area :
  let cube : DrilledCube := { edgeLength := 10, tunnelStartDistance := 3 }
  surfaceArea cube = 582 + 42 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_drilled_cube_surface_area_l2510_251037


namespace NUMINAMATH_CALUDE_quadratic_max_value_l2510_251042

/-- A quadratic function that takes specific values at consecutive natural numbers -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ n : ℕ, f n = 6 ∧ f (n + 1) = 14 ∧ f (n + 2) = 14

/-- The theorem stating the maximum value of the quadratic function -/
theorem quadratic_max_value (f : ℝ → ℝ) (h : QuadraticFunction f) :
  ∃ c : ℝ, c = 15 ∧ ∀ x : ℝ, f x ≤ c :=
sorry

end NUMINAMATH_CALUDE_quadratic_max_value_l2510_251042


namespace NUMINAMATH_CALUDE_triangle_problem_l2510_251011

/-- Triangle with side lengths a, b, c and inradius r -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ

/-- The main theorem stating the properties of the three triangles -/
theorem triangle_problem (H₁ H₂ H₃ : Triangle) : 
  (∃ (d : ℝ), H₁.b = (H₁.a + H₁.c) / 2 ∧ 
               H₁.a = H₁.b - d ∧ 
               H₁.c = H₁.b + d) →
  (H₂.a = H₁.a - 10 ∧ H₂.b = H₁.b - 10 ∧ H₂.c = H₁.c - 10) →
  (H₃.a = H₁.a + 14 ∧ H₃.b = H₁.b + 14 ∧ H₃.c = H₁.c + 14) →
  (H₂.r = H₁.r - 5) →
  (H₃.r = H₁.r + 5) →
  (H₁.a = 25 ∧ H₁.b = 38 ∧ H₁.c = 51) := by
sorry

end NUMINAMATH_CALUDE_triangle_problem_l2510_251011


namespace NUMINAMATH_CALUDE_train_passing_time_l2510_251038

theorem train_passing_time 
  (L : ℝ) 
  (v₁ v₂ : ℝ) 
  (h₁ : L > 0) 
  (h₂ : v₁ > 0) 
  (h₃ : v₂ > 0) : 
  (2 * L) / ((v₁ + v₂) * (1000 / 3600)) = 
  (2 * L) / ((v₁ + v₂) * (1000 / 3600)) :=
by sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l2510_251038


namespace NUMINAMATH_CALUDE_original_average_l2510_251065

theorem original_average (n : ℕ) (a : ℝ) (b : ℝ) (c : ℝ) :
  n > 0 →
  n = 15 →
  b = 13 →
  c = 53 →
  (a + b = c) →
  a = 40 := by
sorry

end NUMINAMATH_CALUDE_original_average_l2510_251065


namespace NUMINAMATH_CALUDE_cargo_volume_maximized_l2510_251069

/-- Represents the number of round trips as a function of the number of small boats towed -/
def roundTrips (x : ℝ) : ℝ := -2 * x + 24

/-- Represents the total cargo volume as a function of the number of small boats towed -/
def cargoVolume (x : ℝ) (M : ℝ) : ℝ := M * x * roundTrips x

theorem cargo_volume_maximized :
  ∀ M : ℝ, M > 0 →
  ∀ x : ℝ, x > 0 →
  cargoVolume 6 M ≥ cargoVolume x M ∧
  roundTrips 4 = 16 ∧
  roundTrips 7 = 10 :=
sorry

end NUMINAMATH_CALUDE_cargo_volume_maximized_l2510_251069


namespace NUMINAMATH_CALUDE_four_digit_integer_transformation_l2510_251018

theorem four_digit_integer_transformation (A : ℕ) (n : ℕ) :
  (A ≥ 1000 ∧ A < 10000) →
  (∃ a b c d : ℕ,
    A = 1000 * a + 100 * b + 10 * c + d ∧
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧
    1000 * (a + n) + 100 * (b - n) + 10 * (c + n) + (d - n) = n * A) →
  A = 1818 :=
by sorry

end NUMINAMATH_CALUDE_four_digit_integer_transformation_l2510_251018


namespace NUMINAMATH_CALUDE_max_cards_jasmine_can_buy_l2510_251097

/-- The maximum number of cards Jasmine can buy given her budget and the pricing conditions --/
theorem max_cards_jasmine_can_buy :
  let initial_price : ℚ := 95 / 100  -- $0.95 per card
  let discounted_price : ℚ := 85 / 100  -- $0.85 per card
  let budget : ℚ := 9  -- $9.00 budget
  let discount_threshold : ℕ := 6  -- Discount applies after 6 cards

  ∃ (n : ℕ), 
    (n ≤ discount_threshold ∧ n * initial_price ≤ budget) ∨
    (n > discount_threshold ∧ 
     discount_threshold * initial_price + (n - discount_threshold) * discounted_price ≤ budget) ∧
    ∀ (m : ℕ), m > n → 
      (m ≤ discount_threshold → m * initial_price > budget) ∧
      (m > discount_threshold → 
       discount_threshold * initial_price + (m - discount_threshold) * discounted_price > budget) ∧
    n = 9 :=
by
  sorry


end NUMINAMATH_CALUDE_max_cards_jasmine_can_buy_l2510_251097


namespace NUMINAMATH_CALUDE_flower_purchase_analysis_l2510_251010

/-- Represents the number and cost of different flower types --/
structure FlowerPurchase where
  roses : ℕ
  lilies : ℕ
  sunflowers : ℕ
  daisies : ℕ
  rose_cost : ℚ
  lily_cost : ℚ
  sunflower_cost : ℚ
  daisy_cost : ℚ

/-- Calculates the total cost of the flower purchase --/
def total_cost (purchase : FlowerPurchase) : ℚ :=
  purchase.roses * purchase.rose_cost +
  purchase.lilies * purchase.lily_cost +
  purchase.sunflowers * purchase.sunflower_cost +
  purchase.daisies * purchase.daisy_cost

/-- Calculates the total number of flowers --/
def total_flowers (purchase : FlowerPurchase) : ℕ :=
  purchase.roses + purchase.lilies + purchase.sunflowers + purchase.daisies

/-- Calculates the percentage of a specific flower type --/
def flower_percentage (count : ℕ) (total : ℕ) : ℚ :=
  (count : ℚ) / (total : ℚ) * 100

/-- Theorem stating the total cost and percentages of flowers --/
theorem flower_purchase_analysis (purchase : FlowerPurchase)
  (h1 : purchase.roses = 50)
  (h2 : purchase.lilies = 40)
  (h3 : purchase.sunflowers = 30)
  (h4 : purchase.daisies = 20)
  (h5 : purchase.rose_cost = 2)
  (h6 : purchase.lily_cost = 3/2)
  (h7 : purchase.sunflower_cost = 1)
  (h8 : purchase.daisy_cost = 3/4) :
  total_cost purchase = 205 ∧
  flower_percentage purchase.roses (total_flowers purchase) = 35.71 ∧
  flower_percentage purchase.lilies (total_flowers purchase) = 28.57 ∧
  flower_percentage purchase.sunflowers (total_flowers purchase) = 21.43 ∧
  flower_percentage purchase.daisies (total_flowers purchase) = 14.29 := by
  sorry

end NUMINAMATH_CALUDE_flower_purchase_analysis_l2510_251010


namespace NUMINAMATH_CALUDE_max_profit_at_0_032_l2510_251073

-- Define the bank's profit function
def bankProfit (k : ℝ) (x : ℝ) : ℝ := 0.048 * k * x^2 - k * x^3

-- State the theorem
theorem max_profit_at_0_032 (k : ℝ) (h_k : k > 0) :
  ∃ (x : ℝ), x ∈ Set.Ioo 0 0.048 ∧
  ∀ (y : ℝ), y ∈ Set.Ioo 0 0.048 → bankProfit k x ≥ bankProfit k y :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_0_032_l2510_251073


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l2510_251063

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

/-- The main theorem -/
theorem collinear_points_x_value :
  let A : Point := ⟨-1, 1⟩
  let B : Point := ⟨2, -4⟩
  let C : Point := ⟨x, -9⟩
  collinear A B C → x = 5 := by
  sorry


end NUMINAMATH_CALUDE_collinear_points_x_value_l2510_251063


namespace NUMINAMATH_CALUDE_debate_team_grouping_l2510_251008

theorem debate_team_grouping (boys girls groups : ℕ) (h1 : boys = 28) (h2 : girls = 4) (h3 : groups = 8) :
  (boys + girls) / groups = 4 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_grouping_l2510_251008


namespace NUMINAMATH_CALUDE_binomial_sum_odd_terms_l2510_251085

theorem binomial_sum_odd_terms (n : ℕ) (h : n > 0) (h_equal : Nat.choose n 4 = Nat.choose n 6) :
  (Finset.range ((n + 1) / 2)).sum (fun k => Nat.choose n (2 * k)) = 2^(n - 1) :=
sorry

end NUMINAMATH_CALUDE_binomial_sum_odd_terms_l2510_251085


namespace NUMINAMATH_CALUDE_power_product_equals_two_l2510_251089

theorem power_product_equals_two :
  (1/2)^2016 * (-2)^2017 * (-1)^2017 = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equals_two_l2510_251089


namespace NUMINAMATH_CALUDE_not_parabola_l2510_251043

theorem not_parabola (α : Real) (h : α ∈ Set.Icc 0 Real.pi) :
  ¬∃ (a b c : Real), ∀ (x y : Real),
    x^2 * Real.sin α + y^2 * Real.cos α = 1 ↔ y = a*x^2 + b*x + c :=
sorry

end NUMINAMATH_CALUDE_not_parabola_l2510_251043


namespace NUMINAMATH_CALUDE_johns_remaining_money_is_135_l2510_251090

/-- Calculates John's remaining money after dog walking and expenses in April --/
def johns_remaining_money : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ :=
  fun (total_days : ℕ) (sundays : ℕ) (weekday_rate : ℕ) (weekend_rate : ℕ)
      (mark_help_days : ℕ) (book_cost : ℕ) (book_discount : ℕ)
      (sister_percentage : ℕ) (gift_cost : ℕ) =>
    let working_days := total_days - sundays
    let weekends := sundays
    let weekdays := working_days - weekends
    let weekday_earnings := weekdays * weekday_rate
    let weekend_earnings := weekends * weekend_rate
    let mark_split_earnings := (mark_help_days * weekday_rate) / 2
    let total_earnings := weekday_earnings + weekend_earnings + mark_split_earnings
    let discounted_book_cost := book_cost - (book_cost * book_discount / 100)
    let after_books := total_earnings - discounted_book_cost
    let sister_share := after_books * sister_percentage / 100
    let after_sister := after_books - sister_share
    let after_gift := after_sister - gift_cost
    let food_cost := weekends * 10
    after_gift - food_cost

theorem johns_remaining_money_is_135 :
  johns_remaining_money 30 4 10 15 3 50 10 20 25 = 135 := by
  sorry

end NUMINAMATH_CALUDE_johns_remaining_money_is_135_l2510_251090


namespace NUMINAMATH_CALUDE_min_sum_squares_reciprocal_inequality_l2510_251030

-- Define the set D
def D : Set (ℝ × ℝ) := {p | p.1 + p.2 = 2 ∧ p.1 > 0 ∧ p.2 > 0}

-- Theorem 1: Minimum value of x₁² + x₂²
theorem min_sum_squares (p : ℝ × ℝ) (h : p ∈ D) : p.1^2 + p.2^2 ≥ 2 := by
  sorry

-- Theorem 2: Inequality for reciprocals
theorem reciprocal_inequality (p : ℝ × ℝ) (h : p ∈ D) :
  1 / (p.1 + 2*p.2) + 1 / (2*p.1 + p.2) ≥ 2/3 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_squares_reciprocal_inequality_l2510_251030


namespace NUMINAMATH_CALUDE_ice_water_volume_change_l2510_251081

theorem ice_water_volume_change (v : ℝ) (h : v > 0) :
  let ice_volume := v * (1 + 1/11)
  (ice_volume - v) / ice_volume = 1/12 := by
sorry

end NUMINAMATH_CALUDE_ice_water_volume_change_l2510_251081


namespace NUMINAMATH_CALUDE_cakes_left_with_brenda_l2510_251036

def cakes_per_day : ℕ := 20
def days_baking : ℕ := 9
def fraction_sold : ℚ := 1/2

theorem cakes_left_with_brenda : 
  (cakes_per_day * days_baking) * (1 - fraction_sold) = 90 := by
  sorry

end NUMINAMATH_CALUDE_cakes_left_with_brenda_l2510_251036


namespace NUMINAMATH_CALUDE_P_in_fourth_quadrant_l2510_251044

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the fourth quadrant -/
def fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- The given point P -/
def P : Point :=
  { x := 2023, y := -2024 }

/-- Theorem stating that P is in the fourth quadrant -/
theorem P_in_fourth_quadrant : fourth_quadrant P := by
  sorry

end NUMINAMATH_CALUDE_P_in_fourth_quadrant_l2510_251044


namespace NUMINAMATH_CALUDE_positive_expression_l2510_251000

theorem positive_expression (a b c : ℝ) 
  (ha : 0 < a ∧ a < 2) 
  (hb : -2 < b ∧ b < 0) 
  (hc : 0 < c ∧ c < 3) : 
  0 < b + 3 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l2510_251000


namespace NUMINAMATH_CALUDE_garden_expenses_l2510_251019

/-- Calculate the total expenses for flowers in a garden --/
theorem garden_expenses (tulips carnations roses : ℕ) (price : ℚ) : 
  tulips = 250 → 
  carnations = 375 → 
  roses = 320 → 
  price = 2 → 
  (tulips + carnations + roses : ℚ) * price = 1890 := by
sorry

end NUMINAMATH_CALUDE_garden_expenses_l2510_251019


namespace NUMINAMATH_CALUDE_perpendicular_lines_planes_l2510_251051

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation for lines and planes
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_line_line : Line → Line → Prop)
variable (perp_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_lines_planes 
  (a b : Line) (α β : Plane) 
  (h_non_coincident : a ≠ b) 
  (h_a_perp_α : perp_line_plane a α) 
  (h_b_perp_β : perp_line_plane b β) : 
  (perp_line_line a b ↔ perp_plane_plane α β) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_planes_l2510_251051


namespace NUMINAMATH_CALUDE_fraction_equality_l2510_251088

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (4 * x + y) / (x - 4 * y) = 3) : 
  (x + 4 * y) / (4 * x - y) = 9 / 53 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l2510_251088


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_l2510_251045

/-- Given a hyperbola with equation x²/m² - y² = 1 where m > 0,
    if one of its asymptote equations is x + √3 * y = 0, then m = √3 -/
theorem hyperbola_asymptote (m : ℝ) (hm : m > 0) :
  (∃ x y : ℝ, x^2 / m^2 - y^2 = 1) →
  (∃ x y : ℝ, x + Real.sqrt 3 * y = 0) →
  m = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_l2510_251045


namespace NUMINAMATH_CALUDE_rational_root_l2510_251001

theorem rational_root (x : ℝ) (hx : x ≠ 0) 
  (h1 : ∃ r : ℚ, x^5 = r) 
  (h2 : ∃ p : ℚ, 20*x + 19/x = p) : 
  ∃ q : ℚ, x = q := by
sorry

end NUMINAMATH_CALUDE_rational_root_l2510_251001


namespace NUMINAMATH_CALUDE_sequence_arithmetic_progression_l2510_251094

theorem sequence_arithmetic_progression
  (s : ℕ → ℕ)
  (h_increasing : ∀ n, s n < s (n + 1))
  (h_positive : ∀ n, s n > 0)
  (h_subseq1 : ∃ a d : ℕ, ∀ n, s (s n) = a + n * d)
  (h_subseq2 : ∃ b e : ℕ, ∀ n, s (s n + 1) = b + n * e) :
  ∃ c f : ℕ, ∀ n, s n = c + n * f := by
sorry

end NUMINAMATH_CALUDE_sequence_arithmetic_progression_l2510_251094


namespace NUMINAMATH_CALUDE_extended_triangle_theorem_l2510_251003

-- Define the triangle ABC
variable (A B C : Point) (ABC : Triangle A B C)

-- Define the condition BC = 2AC
variable (h1 : BC = 2 * AC)

-- Define point D such that AD = 1/3 * AB
variable (D : Point)
variable (h2 : AD = (1/3) * AB)

-- Theorem statement
theorem extended_triangle_theorem : CD = 2 * AD := by
  sorry

end NUMINAMATH_CALUDE_extended_triangle_theorem_l2510_251003


namespace NUMINAMATH_CALUDE_cookie_pattern_proof_l2510_251062

def cookie_sequence (n : ℕ) : ℕ := 
  match n with
  | 1 => 5
  | 2 => 5  -- This is what we want to prove
  | 3 => 10
  | 4 => 14
  | 5 => 19
  | 6 => 25
  | _ => 0  -- For other values, we don't care in this problem

theorem cookie_pattern_proof : 
  (cookie_sequence 1 = 5) ∧ 
  (cookie_sequence 3 = 10) ∧ 
  (cookie_sequence 4 = 14) ∧ 
  (cookie_sequence 5 = 19) ∧ 
  (cookie_sequence 6 = 25) ∧ 
  (∀ n : ℕ, n > 2 → cookie_sequence n - cookie_sequence (n-1) = 
    if n % 2 = 0 then 4 else 5) →
  cookie_sequence 2 = 5 := by
sorry

end NUMINAMATH_CALUDE_cookie_pattern_proof_l2510_251062


namespace NUMINAMATH_CALUDE_evaluate_expression_l2510_251041

theorem evaluate_expression : -(16 / 4 * 7 + 25 - 2 * 7) = -39 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2510_251041


namespace NUMINAMATH_CALUDE_projection_x_coordinate_l2510_251029

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- Theorem: The x-coordinate of the projection of a point on a circle onto the x-axis -/
theorem projection_x_coordinate 
  (circle : Circle)
  (start : Point)
  (B : Point)
  (angle : ℝ) :
  circle.center = Point.mk 0 0 →
  circle.radius = 4 →
  start = Point.mk 4 0 →
  B.x = 4 * Real.cos angle →
  B.y = 4 * Real.sin angle →
  angle ≥ 0 →
  4 * Real.cos angle = (Point.mk (B.x) 0).x :=
by sorry

end NUMINAMATH_CALUDE_projection_x_coordinate_l2510_251029


namespace NUMINAMATH_CALUDE_parabola_focus_distance_l2510_251095

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus F
def focus : ℝ × ℝ := (2, 0)

-- Define the directrix l
def directrix : ℝ → Prop := λ x => x = -2

-- Define a point on the parabola
def point_on_parabola (P : ℝ × ℝ) : Prop :=
  parabola P.1 P.2

-- Define perpendicularity of PA to directrix
def perpendicular_to_directrix (P A : ℝ × ℝ) : Prop :=
  A.1 = -2 ∧ P.2 = A.2

-- Define the slope of AF
def slope_AF (A : ℝ × ℝ) : Prop :=
  (A.2 - 0) / (A.1 - 2) = -Real.sqrt 3

-- Theorem statement
theorem parabola_focus_distance 
  (P : ℝ × ℝ) 
  (A : ℝ × ℝ) 
  (h1 : point_on_parabola P) 
  (h2 : perpendicular_to_directrix P A) 
  (h3 : slope_AF A) : 
  Real.sqrt ((P.1 - focus.1)^2 + (P.2 - focus.2)^2) = 8 :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_distance_l2510_251095


namespace NUMINAMATH_CALUDE_most_accurate_value_for_given_K_l2510_251021

/-- Given a scientific constant K and its error margin, 
    returns the most accurate value with all digits significant -/
def most_accurate_value (K : ℝ) (error : ℝ) : ℝ :=
  sorry

theorem most_accurate_value_for_given_K :
  let K : ℝ := 3.68547
  let error : ℝ := 0.00256
  most_accurate_value K error = 3.7 := by sorry

end NUMINAMATH_CALUDE_most_accurate_value_for_given_K_l2510_251021


namespace NUMINAMATH_CALUDE_kids_savings_l2510_251068

/-- The total amount saved by three kids given their coin collections -/
def total_savings (teagan_pennies rex_nickels toni_dimes : ℕ) : ℚ :=
  (teagan_pennies : ℚ) * (1 / 100) +
  (rex_nickels : ℚ) * (5 / 100) +
  (toni_dimes : ℚ) * (10 / 100)

/-- Theorem stating that the total savings of the three kids is $40 -/
theorem kids_savings : total_savings 200 100 330 = 40 := by
  sorry

end NUMINAMATH_CALUDE_kids_savings_l2510_251068


namespace NUMINAMATH_CALUDE_seagrass_study_l2510_251015

/-- Represents the sample statistics for a town -/
structure TownSample where
  size : ℕ
  mean : ℝ
  variance : ℝ

/-- Represents the competition probabilities for town A -/
structure CompetitionProbs where
  win_in_A : ℝ
  win_in_B : ℝ

theorem seagrass_study (town_A : TownSample) (town_B : TownSample) (probs : CompetitionProbs)
  (h_A_size : town_A.size = 12)
  (h_A_mean : town_A.mean = 18)
  (h_A_var : town_A.variance = 19)
  (h_B_size : town_B.size = 18)
  (h_B_mean : town_B.mean = 36)
  (h_B_var : town_B.variance = 70)
  (h_prob_A : probs.win_in_A = 3/5)
  (h_prob_B : probs.win_in_B = 1/2) :
  let total_mean := (town_A.size * town_A.mean + town_B.size * town_B.mean) / (town_A.size + town_B.size)
  let total_variance := (1 / (town_A.size + town_B.size)) *
    (town_A.size * town_A.variance + town_A.size * (town_A.mean - total_mean)^2 +
     town_B.size * town_B.variance + town_B.size * (town_B.mean - total_mean)^2)
  let expected_score := 0 * (1 - probs.win_in_A)^2 + 1 * (2 * probs.win_in_A * (1 - probs.win_in_B) * (1 - probs.win_in_A)) +
    2 * (1 - (1 - probs.win_in_A)^2 - 2 * probs.win_in_A * (1 - probs.win_in_B) * (1 - probs.win_in_A))
  total_mean = 28.8 ∧ total_variance = 127.36 ∧ expected_score = 36/25 := by
  sorry

end NUMINAMATH_CALUDE_seagrass_study_l2510_251015


namespace NUMINAMATH_CALUDE_peter_is_18_l2510_251023

-- Define Peter's current age
def peter_current_age : ℕ := sorry

-- Define Ivan's current age
def ivan_current_age : ℕ := sorry

-- Define Peter's past age when Ivan was Peter's current age
def peter_past_age : ℕ := sorry

-- Condition 1: Ivan's current age is twice Peter's past age
axiom ivan_age_relation : ivan_current_age = 2 * peter_past_age

-- Condition 2: Sum of their ages will be 54 when Peter reaches Ivan's current age
axiom future_age_sum : ivan_current_age + ivan_current_age = 54

-- Condition 3: The time difference between Peter's current age and past age
-- is equal to the time difference between Ivan's current age and Peter's current age
axiom age_difference_relation : ivan_current_age - peter_current_age = peter_current_age - peter_past_age

theorem peter_is_18 : peter_current_age = 18 := by
  sorry

end NUMINAMATH_CALUDE_peter_is_18_l2510_251023


namespace NUMINAMATH_CALUDE_green_bay_high_relay_race_length_l2510_251098

/-- The length of a relay race given the number of team members and distance per member -/
def relay_race_length (team_members : ℕ) (distance_per_member : ℕ) : ℕ :=
  team_members * distance_per_member

/-- Theorem: The relay race length for 5 team members running 30 meters each is 150 meters -/
theorem green_bay_high_relay_race_length :
  relay_race_length 5 30 = 150 := by
  sorry

end NUMINAMATH_CALUDE_green_bay_high_relay_race_length_l2510_251098


namespace NUMINAMATH_CALUDE_derivative_of_exp_2x_l2510_251031

theorem derivative_of_exp_2x (x : ℝ) :
  deriv (fun x => Real.exp (2 * x)) x = 2 * Real.exp (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_exp_2x_l2510_251031


namespace NUMINAMATH_CALUDE_intersection_tangent_negative_x_l2510_251052

theorem intersection_tangent_negative_x (x₀ y₀ : ℝ) : 
  x₀ > 0 → y₀ = Real.tan x₀ → y₀ = -x₀ → 
  (x₀^2 + 1) * (Real.cos (2 * x₀) + 1) = 2 := by sorry

end NUMINAMATH_CALUDE_intersection_tangent_negative_x_l2510_251052


namespace NUMINAMATH_CALUDE_cube_root_increasing_l2510_251050

/-- The cube root function is increasing on the real numbers. -/
theorem cube_root_increasing :
  ∀ x y : ℝ, x < y → x^(1/3) < y^(1/3) := by sorry

end NUMINAMATH_CALUDE_cube_root_increasing_l2510_251050


namespace NUMINAMATH_CALUDE_marbles_problem_l2510_251076

theorem marbles_problem (initial_marbles given_marbles remaining_marbles : ℕ) : 
  given_marbles = 8 → remaining_marbles = 24 → initial_marbles = given_marbles + remaining_marbles →
  initial_marbles = 32 := by
sorry

end NUMINAMATH_CALUDE_marbles_problem_l2510_251076


namespace NUMINAMATH_CALUDE_cube_sum_of_sqrt_equals_24_l2510_251024

theorem cube_sum_of_sqrt_equals_24 :
  (Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3))^3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_of_sqrt_equals_24_l2510_251024


namespace NUMINAMATH_CALUDE_average_weight_abc_l2510_251012

theorem average_weight_abc (a b c : ℝ) 
  (h1 : (a + b) / 2 = 40)
  (h2 : (b + c) / 2 = 43)
  (h3 : b = 31) :
  (a + b + c) / 3 = 45 := by
sorry

end NUMINAMATH_CALUDE_average_weight_abc_l2510_251012


namespace NUMINAMATH_CALUDE_cookies_per_box_l2510_251047

/-- The number of cookies in each box, given the collection amounts of Abigail, Grayson, and Olivia, and the total number of cookies. -/
theorem cookies_per_box (abigail_boxes : ℚ) (grayson_boxes : ℚ) (olivia_boxes : ℚ) (total_cookies : ℕ) :
  abigail_boxes = 2 →
  grayson_boxes = 3 / 4 →
  olivia_boxes = 3 →
  total_cookies = 276 →
  total_cookies / (abigail_boxes + grayson_boxes + olivia_boxes) = 48 := by
sorry

end NUMINAMATH_CALUDE_cookies_per_box_l2510_251047


namespace NUMINAMATH_CALUDE_wen_family_movie_cost_l2510_251055

def ticket_cost (regular_price : ℚ) (discount : ℚ) : ℚ :=
  regular_price * (1 - discount)

theorem wen_family_movie_cost :
  let senior_price : ℚ := 6
  let senior_discount : ℚ := 1/4
  let children_discount : ℚ := 1/2
  let regular_price : ℚ := senior_price / (1 - senior_discount)
  let num_people_per_generation : ℕ := 2
  
  num_people_per_generation * senior_price +
  num_people_per_generation * regular_price +
  num_people_per_generation * (ticket_cost regular_price children_discount) = 36
  := by sorry

end NUMINAMATH_CALUDE_wen_family_movie_cost_l2510_251055


namespace NUMINAMATH_CALUDE_special_sequence_has_repeats_l2510_251040

/-- A sequence of rational numbers satisfying the given property -/
def SpecialSequence := ℕ → ℚ

/-- The property that defines our special sequence -/
def HasSpecialProperty (a : SpecialSequence) : Prop :=
  ∀ m n : ℕ, a m + a n = a (m * n)

/-- The theorem stating that a sequence with the special property has repeated elements -/
theorem special_sequence_has_repeats (a : SpecialSequence) (h : HasSpecialProperty a) :
  ∃ i j : ℕ, i ≠ j ∧ a i = a j := by sorry

end NUMINAMATH_CALUDE_special_sequence_has_repeats_l2510_251040


namespace NUMINAMATH_CALUDE_kennel_cats_dogs_difference_l2510_251032

/-- Proves that in a kennel with a 2:3 ratio of cats to dogs and 18 dogs, there are 6 fewer cats than dogs -/
theorem kennel_cats_dogs_difference :
  ∀ (num_cats num_dogs : ℕ),
  num_dogs = 18 →
  num_cats * 3 = num_dogs * 2 →
  num_cats < num_dogs →
  num_dogs - num_cats = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_kennel_cats_dogs_difference_l2510_251032


namespace NUMINAMATH_CALUDE_incorrect_absolute_value_expression_l2510_251060

theorem incorrect_absolute_value_expression : 
  ((-|5|)^2 = 25) ∧ 
  (|((-5)^2)| = 25) ∧ 
  ((-|5|)^2 = 25) ∧ 
  ¬((|(-5)|)^2 = 25) := by sorry

end NUMINAMATH_CALUDE_incorrect_absolute_value_expression_l2510_251060


namespace NUMINAMATH_CALUDE_tank_volume_l2510_251034

/-- Given a cube-shaped tank constructed from metal sheets, calculate its volume in liters -/
theorem tank_volume (sheet_length : ℝ) (sheet_width : ℝ) (num_sheets : ℕ) : 
  sheet_length = 2 →
  sheet_width = 3 →
  num_sheets = 100 →
  (((num_sheets * sheet_length * sheet_width / 6) ^ (1/2 : ℝ)) ^ 3) * 1000 = 1000000 := by
  sorry

#check tank_volume

end NUMINAMATH_CALUDE_tank_volume_l2510_251034


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l2510_251007

theorem complex_magnitude_problem (z : ℂ) (h : z * (1 - Complex.I) = 1 + Complex.I) :
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l2510_251007


namespace NUMINAMATH_CALUDE_roof_area_l2510_251035

theorem roof_area (width length : ℝ) : 
  width > 0 →
  length > 0 →
  length = 4 * width →
  length - width = 48 →
  width * length = 1024 := by
  sorry

end NUMINAMATH_CALUDE_roof_area_l2510_251035


namespace NUMINAMATH_CALUDE_system_solutions_l2510_251039

theorem system_solutions :
  ∃! (S : Set (ℝ × ℝ × ℝ)), 
    S = {(1, 1, 1), (-2, -2, -2)} ∧
    ∀ (x y z : ℝ), (x, y, z) ∈ S ↔ 
      (x + y * z = 2 ∧ y + z * x = 2 ∧ z + x * y = 2) :=
by sorry

end NUMINAMATH_CALUDE_system_solutions_l2510_251039


namespace NUMINAMATH_CALUDE_escalator_solution_l2510_251066

/-- Represents the escalator problem with given conditions -/
structure EscalatorProblem where
  total_steps : ℕ
  escalator_speed : ℚ
  walking_speed : ℚ
  first_condition : 26 + 30 * escalator_speed = total_steps
  second_condition : 34 + 18 * escalator_speed = total_steps

/-- The solution to the escalator problem -/
theorem escalator_solution (problem : EscalatorProblem) : problem.total_steps = 46 := by
  sorry

#check escalator_solution

end NUMINAMATH_CALUDE_escalator_solution_l2510_251066


namespace NUMINAMATH_CALUDE_students_with_both_calculation_l2510_251064

/-- The number of students who brought both apples and bananas -/
def students_with_both : ℕ := sorry

/-- The number of students who brought apples -/
def students_with_apples : ℕ := 12

/-- The number of students who brought bananas -/
def students_with_bananas : ℕ := 8

/-- The number of students who brought only one type of fruit -/
def students_with_one_fruit : ℕ := 10

theorem students_with_both_calculation : 
  students_with_both = students_with_apples + students_with_bananas - students_with_one_fruit :=
by sorry

end NUMINAMATH_CALUDE_students_with_both_calculation_l2510_251064


namespace NUMINAMATH_CALUDE_seating_arrangements_eq_48_num_seating_arrangements_l2510_251091

/- Define the number of teams -/
def num_teams : ℕ := 3

/- Define the number of athletes per team -/
def athletes_per_team : ℕ := 2

/- Define the total number of athletes -/
def total_athletes : ℕ := num_teams * athletes_per_team

/- Function to calculate the number of seating arrangements -/
def seating_arrangements : ℕ :=
  (Nat.factorial num_teams) * (Nat.factorial athletes_per_team)^num_teams

/- Theorem stating that the number of seating arrangements is 48 -/
theorem seating_arrangements_eq_48 :
  seating_arrangements = 48 := by
  sorry

/- Main theorem to prove -/
theorem num_seating_arrangements :
  ∀ (n m : ℕ), n = num_teams → m = athletes_per_team →
  (Nat.factorial n) * (Nat.factorial m)^n = 48 := by
  sorry

end NUMINAMATH_CALUDE_seating_arrangements_eq_48_num_seating_arrangements_l2510_251091


namespace NUMINAMATH_CALUDE_dog_grouping_theorem_l2510_251087

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of dogs -/
def total_dogs : ℕ := 12

/-- The number of dogs in Fluffy's group -/
def fluffy_group_size : ℕ := 3

/-- The number of dogs in Nipper's group -/
def nipper_group_size : ℕ := 5

/-- The number of dogs in the third group -/
def third_group_size : ℕ := 4

theorem dog_grouping_theorem :
  choose (total_dogs - 2) (fluffy_group_size - 1) *
  choose (total_dogs - fluffy_group_size - 1) (nipper_group_size - 1) = 3150 := by
  sorry

end NUMINAMATH_CALUDE_dog_grouping_theorem_l2510_251087


namespace NUMINAMATH_CALUDE_method_doubles_method_power_of_two_l2510_251017

/-- Represents the state of a coin (Heads or Tails) -/
inductive CoinState
| Heads
| Tails

/-- Represents a row of coins -/
def CoinRow (N : ℕ) := Fin N → CoinState

/-- Represents a method for the magician to guess the number -/
structure GuessMethod (N : ℕ) :=
(guess : CoinRow N → Fin N)

/-- States that if a method exists for N coins, it exists for 2N coins -/
theorem method_doubles {N : ℕ} (h : GuessMethod N) : GuessMethod (2 * N) :=
sorry

/-- States that the method only works for powers of 2 -/
theorem method_power_of_two {N : ℕ} : GuessMethod N → ∃ k : ℕ, N = 2^k :=
sorry

end NUMINAMATH_CALUDE_method_doubles_method_power_of_two_l2510_251017


namespace NUMINAMATH_CALUDE_sin_double_plus_sin_squared_l2510_251086

theorem sin_double_plus_sin_squared (α : Real) (h : Real.tan α = 1/2) :
  Real.sin (2 * α) + Real.sin α ^ 2 = 1 := by sorry

end NUMINAMATH_CALUDE_sin_double_plus_sin_squared_l2510_251086


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l2510_251067

def total_employees : ℕ := 10
def men : ℕ := 6
def women : ℕ := 4
def unavailable_men : ℕ := 1
def unavailable_women : ℕ := 1
def selection_size : ℕ := 3

def available_men : ℕ := men - unavailable_men
def available_women : ℕ := women - unavailable_women
def total_available : ℕ := available_men + available_women

theorem probability_at_least_one_woman :
  (1 - (Nat.choose available_men selection_size : ℚ) / (Nat.choose total_available selection_size : ℚ)) = 23/28 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l2510_251067


namespace NUMINAMATH_CALUDE_ramanujan_number_l2510_251013

theorem ramanujan_number (h r : ℂ) : 
  h * r = 40 + 24 * I ∧ h = 3 + 7 * I → r = 4 - (104 / 29) * I :=
by sorry

end NUMINAMATH_CALUDE_ramanujan_number_l2510_251013


namespace NUMINAMATH_CALUDE_power_function_property_l2510_251093

/-- A power function is a function of the form f(x) = x^α for some real α -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x : ℝ, f x = x ^ α

theorem power_function_property (f : ℝ → ℝ) (h1 : isPowerFunction f) (h2 : f 2 = 4) :
  f 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_power_function_property_l2510_251093


namespace NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_equation_l2510_251077

-- Define the intersection point
def intersection_point : ℝ × ℝ := (3, 2)

-- Define the parallel line
def parallel_line (x y : ℝ) : Prop := 3 * x - 2 * y + 4 = 0

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := 4 * x - 3 * y - 7 = 0

-- Theorem for the parallel case
theorem parallel_line_equation :
  ∃ (m : ℝ), ∀ (x y : ℝ),
    (x, y) = intersection_point →
    (3 * x - 2 * y + m = 0 ↔ parallel_line x y) :=
sorry

-- Theorem for the perpendicular case
theorem perpendicular_line_equation :
  ∃ (n : ℝ), ∀ (x y : ℝ),
    (x, y) = intersection_point →
    (3 * x + 4 * y + n = 0 ↔ perpendicular_line x y) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_equation_perpendicular_line_equation_l2510_251077


namespace NUMINAMATH_CALUDE_polygon_area_l2510_251079

-- Define the polygon
structure Polygon :=
  (sides : ℕ)
  (perimeter : ℝ)
  (num_squares : ℕ)
  (congruent_sides : Bool)
  (perpendicular_sides : Bool)

-- Define the properties of our specific polygon
def special_polygon : Polygon :=
  { sides := 28,
    perimeter := 56,
    num_squares := 25,
    congruent_sides := true,
    perpendicular_sides := true }

-- Theorem statement
theorem polygon_area (p : Polygon) (h1 : p = special_polygon) : 
  (p.perimeter / p.sides)^2 * p.num_squares = 100 := by
  sorry

end NUMINAMATH_CALUDE_polygon_area_l2510_251079


namespace NUMINAMATH_CALUDE_inequality_proof_l2510_251074

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : c < 0) : a * (c - 1) < b * (c - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2510_251074


namespace NUMINAMATH_CALUDE_elvis_studio_time_l2510_251009

/-- Calculates the total time spent in the studio for Elvis's album production -/
def total_studio_time (num_songs : ℕ) (record_time : ℕ) (edit_time : ℕ) (write_time : ℕ) : ℚ :=
  let total_minutes := num_songs * (record_time + write_time) + edit_time
  total_minutes / 60

/-- Proves that Elvis spent 5 hours in the studio given the specified conditions -/
theorem elvis_studio_time :
  total_studio_time 10 12 30 15 = 5 := by
sorry

end NUMINAMATH_CALUDE_elvis_studio_time_l2510_251009


namespace NUMINAMATH_CALUDE_m_range_theorem_l2510_251075

-- Define the conditions
def p (x : ℝ) : Prop := -2 < x ∧ x < 10
def q (x m : ℝ) : Prop := (x - 1)^2 - m^2 ≤ 0

-- Define the theorem
theorem m_range_theorem :
  (∀ x, p x → q x m) ∧ 
  (∃ x, q x m ∧ ¬p x) ∧ 
  (m > 0) →
  m ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l2510_251075


namespace NUMINAMATH_CALUDE_distinct_tetrahedrons_count_l2510_251022

/-- The number of vertices in a cube -/
def cube_vertices : ℕ := 8

/-- The number of vertices required to form a tetrahedron -/
def tetrahedron_vertices : ℕ := 4

/-- The number of non-tetrahedral configurations -/
def non_tetrahedral_configurations : ℕ := 12

/-- The number of distinct tetrahedrons that can be formed using the vertices of a cube -/
def distinct_tetrahedrons : ℕ :=
  Nat.choose cube_vertices tetrahedron_vertices - non_tetrahedral_configurations

theorem distinct_tetrahedrons_count : distinct_tetrahedrons = 58 := by
  sorry

end NUMINAMATH_CALUDE_distinct_tetrahedrons_count_l2510_251022


namespace NUMINAMATH_CALUDE_intersection_M_N_l2510_251006

def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 + 1}
def N : Set ℝ := {y | ∃ x : ℝ, y = x + 1}

theorem intersection_M_N : M ∩ N = {y : ℝ | y ≥ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2510_251006


namespace NUMINAMATH_CALUDE_derivative_f_at_pi_l2510_251082

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) / (x^2)

theorem derivative_f_at_pi : 
  deriv f π = -1 / (π^2) := by sorry

end NUMINAMATH_CALUDE_derivative_f_at_pi_l2510_251082


namespace NUMINAMATH_CALUDE_dollar_symmetric_sum_l2510_251061

def dollar (a b : ℝ) : ℝ := (a - b)^2

theorem dollar_symmetric_sum (x y : ℝ) : dollar (x + y) (y + x) = 0 := by
  sorry

end NUMINAMATH_CALUDE_dollar_symmetric_sum_l2510_251061


namespace NUMINAMATH_CALUDE_escalator_speed_l2510_251071

theorem escalator_speed (escalator_speed : ℝ) (escalator_length : ℝ) (time_taken : ℝ) 
  (h1 : escalator_speed = 12)
  (h2 : escalator_length = 150)
  (h3 : time_taken = 10) :
  let person_speed := (escalator_length / time_taken) - escalator_speed
  person_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_escalator_speed_l2510_251071


namespace NUMINAMATH_CALUDE_quadratic_roots_properties_l2510_251002

theorem quadratic_roots_properties (a b m : ℝ) : 
  m > 0 → 
  2 * a^2 - 8 * a + m = 0 → 
  2 * b^2 - 8 * b + m = 0 → 
  (a^2 + b^2 ≥ 8) ∧ 
  (Real.sqrt a + Real.sqrt b ≤ 2 * Real.sqrt 2) ∧ 
  (1 / (a + 2) + 1 / (2 * b) ≥ (3 + 2 * Real.sqrt 2) / 12) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_properties_l2510_251002


namespace NUMINAMATH_CALUDE_exactly_two_cubic_polynomials_satisfy_l2510_251004

/-- A polynomial function of degree 3 or less -/
def CubicPolynomial (a b c d : ℝ) : ℝ → ℝ := λ x ↦ a*x^3 + b*x^2 + c*x + d

/-- The condition that f(x)f(-x) = f(x^3) for all x -/
def SatisfiesCondition (f : ℝ → ℝ) : Prop :=
  ∀ x, f x * f (-x) = f (x^3)

/-- The main theorem stating that exactly two cubic polynomials satisfy the condition -/
theorem exactly_two_cubic_polynomials_satisfy :
  ∃! (s : Finset (ℝ → ℝ)),
    (∀ f ∈ s, ∃ a b c d, f = CubicPolynomial a b c d) ∧
    (∀ f ∈ s, SatisfiesCondition f) ∧
    s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_cubic_polynomials_satisfy_l2510_251004


namespace NUMINAMATH_CALUDE_parking_spot_difference_l2510_251005

/-- Represents the number of open parking spots on each level of a 4-story parking area -/
structure ParkingArea where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Theorem stating the difference in open spots between second and first levels -/
theorem parking_spot_difference (p : ParkingArea) : 
  p.first = 4 → 
  p.third = p.second + 6 → 
  p.fourth = 14 → 
  p.first + p.second + p.third + p.fourth = 46 → 
  p.second - p.first = 7 := by
  sorry

#check parking_spot_difference

end NUMINAMATH_CALUDE_parking_spot_difference_l2510_251005


namespace NUMINAMATH_CALUDE_distance_to_line_l2510_251033

/-- Given a triangle ABC with sides AB = 3, BC = 4, and CA = 5,
    the distance from point B to line AC is 12/5 -/
theorem distance_to_line (A B C : ℝ × ℝ) : 
  let d := (λ P Q : ℝ × ℝ => Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2))
  d A B = 3 ∧ d B C = 4 ∧ d C A = 5 → 
  (let area := (1/2) * d A B * d B C
   area / d C A) = 12/5 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_line_l2510_251033


namespace NUMINAMATH_CALUDE_probability_three_different_suits_l2510_251027

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of suits in a standard deck -/
def NumberOfSuits : ℕ := 4

/-- Represents the number of cards in each suit -/
def CardsPerSuit : ℕ := 13

/-- The probability of selecting three cards of different suits from a standard deck without replacement -/
def probabilityDifferentSuits : ℚ :=
  (CardsPerSuit * (StandardDeck - CardsPerSuit) * (StandardDeck - 2 * CardsPerSuit)) /
  (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2))

theorem probability_three_different_suits :
  probabilityDifferentSuits = 169 / 425 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_different_suits_l2510_251027


namespace NUMINAMATH_CALUDE_bumper_car_line_problem_l2510_251046

theorem bumper_car_line_problem (initial_people : ℕ) : 
  (initial_people - 10 + 15 = 17) → initial_people = 12 := by
  sorry

end NUMINAMATH_CALUDE_bumper_car_line_problem_l2510_251046


namespace NUMINAMATH_CALUDE_stratified_sampling_l2510_251070

theorem stratified_sampling (total_students : ℕ) (sample_size : ℕ) (first_grade : ℕ) (second_grade : ℕ) :
  total_students = 2000 →
  sample_size = 100 →
  first_grade = 30 →
  second_grade = 30 →
  sample_size = first_grade + second_grade + (sample_size - first_grade - second_grade) →
  (sample_size - first_grade - second_grade) = 40 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l2510_251070


namespace NUMINAMATH_CALUDE_midpoint_trajectory_l2510_251096

/-- The trajectory of the midpoint of a line segment connecting a point on a unit circle to a fixed point -/
theorem midpoint_trajectory (a b x y : ℝ) : 
  a^2 + b^2 = 1 →  -- P(a,b) is on the unit circle
  x = (a + 3) / 2 ∧ y = b / 2 →  -- M(x,y) is the midpoint of PQ
  (2*x - 3)^2 + 4*y^2 = 1 := by sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_l2510_251096
