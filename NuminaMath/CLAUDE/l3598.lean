import Mathlib

namespace fishing_trips_l3598_359890

theorem fishing_trips (shelly_catch : ℕ) (sam_catch : ℕ) (total_catch : ℕ) :
  shelly_catch = 3 →
  sam_catch = 2 →
  total_catch = 25 →
  (total_catch / (shelly_catch + sam_catch) : ℕ) = 5 := by
sorry

end fishing_trips_l3598_359890


namespace square_arrangement_exists_l3598_359844

/-- A square in the plane --/
structure Square where
  sideLength : ℕ
  position : ℝ × ℝ

/-- An arrangement of squares --/
def Arrangement (n : ℕ) := Fin n → Square

/-- Two squares touch if they share a vertex --/
def touches (s1 s2 : Square) : Prop := sorry

/-- An arrangement is valid if no two squares overlap --/
def validArrangement (arr : Arrangement n) : Prop := sorry

/-- An arrangement satisfies the touching condition if each square touches exactly two others --/
def satisfiesTouchingCondition (arr : Arrangement n) : Prop := sorry

/-- Main theorem: For n ≥ 5, there exists a valid arrangement where each square touches exactly two others --/
theorem square_arrangement_exists (n : ℕ) (h : n ≥ 5) :
  ∃ (arr : Arrangement n), validArrangement arr ∧ satisfiesTouchingCondition arr := by
  sorry

end square_arrangement_exists_l3598_359844


namespace parabola_focus_distance_l3598_359817

/-- A point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 4*x

/-- The distance between a point and a vertical line -/
def dist_to_vertical_line (p : ℝ × ℝ) (line_x : ℝ) : ℝ :=
  |p.1 - line_x|

theorem parabola_focus_distance (P : ParabolaPoint) 
  (h : dist_to_vertical_line (P.x, P.y) (-2) = 5) :
  dist_to_vertical_line (P.x, P.y) 1 = 4 := by sorry

end parabola_focus_distance_l3598_359817


namespace quadratic_real_roots_l3598_359810

theorem quadratic_real_roots (a : ℝ) :
  (∃ x : ℝ, a * x^2 + 2 * x - 1 = 0) ↔ (a ≥ -1 ∧ a ≠ 0) :=
by sorry

end quadratic_real_roots_l3598_359810


namespace largest_angle_in_pentagon_l3598_359847

theorem largest_angle_in_pentagon (F G H I J : ℝ) : 
  F = 70 → 
  G = 110 → 
  H = I → 
  J = 2 * H + 25 → 
  F + G + H + I + J = 540 → 
  J = 192.5 ∧ J = max F (max G (max H (max I J))) := by
  sorry

end largest_angle_in_pentagon_l3598_359847


namespace prob_at_least_two_same_carriage_l3598_359851

/-- The number of carriages in the train -/
def num_carriages : ℕ := 10

/-- The number of acquaintances boarding the train -/
def num_people : ℕ := 3

/-- The probability that at least two people board the same carriage -/
def prob_same_carriage : ℚ := 7/25

/-- Theorem stating the probability of at least two people boarding the same carriage -/
theorem prob_at_least_two_same_carriage : 
  1 - (num_carriages.descFactorial num_people : ℚ) / (num_carriages ^ num_people : ℚ) = prob_same_carriage := by
  sorry

end prob_at_least_two_same_carriage_l3598_359851


namespace s_4_equals_14916_l3598_359869

-- Define s(n) as a function that attaches the first n perfect squares
def s (n : ℕ) : ℕ :=
  -- Implementation details omitted
  sorry

-- Theorem to prove
theorem s_4_equals_14916 : s 4 = 14916 := by
  sorry

end s_4_equals_14916_l3598_359869


namespace sales_function_satisfies_data_profit_192_implies_price_18_max_profit_at_19_l3598_359822

-- Define the linear function
def sales_function (x : ℝ) : ℝ := -2 * x + 60

-- Define the profit function
def profit_function (x : ℝ) : ℝ := (x - 10) * (sales_function x)

-- Theorem 1: The sales function satisfies the given data points
theorem sales_function_satisfies_data : 
  sales_function 12 = 36 ∧ sales_function 13 = 34 := by sorry

-- Theorem 2: When the daily profit is 192 yuan, the selling price is 18 yuan
theorem profit_192_implies_price_18 :
  ∀ x, 10 ≤ x ∧ x ≤ 19 → profit_function x = 192 → x = 18 := by sorry

-- Theorem 3: The maximum daily profit occurs at a selling price of 19 yuan and equals 198 yuan
theorem max_profit_at_19 :
  (∀ x, 10 ≤ x ∧ x ≤ 19 → profit_function x ≤ profit_function 19) ∧
  profit_function 19 = 198 := by sorry

end sales_function_satisfies_data_profit_192_implies_price_18_max_profit_at_19_l3598_359822


namespace diameter_difference_explanation_l3598_359861

/-- Represents the system of a polar bear walking on an ice floe -/
structure BearIceSystem where
  bear_mass : ℝ
  ice_mass : ℝ
  instrument_diameter : ℝ
  photo_diameter : ℝ

/-- The observed diameters differ due to relative motion -/
theorem diameter_difference_explanation (system : BearIceSystem)
  (h1 : system.instrument_diameter = 8.5)
  (h2 : system.photo_diameter = 9)
  (h3 : system.ice_mass > system.bear_mass)
  (h4 : system.ice_mass ≤ 100 * system.bear_mass) :
  ∃ (center_of_mass_shift : ℝ),
    center_of_mass_shift > 0 ∧
    center_of_mass_shift < 0.5 ∧
    system.photo_diameter = system.instrument_diameter + 2 * center_of_mass_shift :=
by sorry

#check diameter_difference_explanation

end diameter_difference_explanation_l3598_359861


namespace gcd_105_88_l3598_359883

theorem gcd_105_88 : Nat.gcd 105 88 = 1 := by
  sorry

end gcd_105_88_l3598_359883


namespace planes_perpendicular_l3598_359805

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation
variable (perp : Line → Line → Prop)
variable (perpLP : Line → Plane → Prop)
variable (perpPP : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (a b : Line) (α β : Plane) 
  (hab : a ≠ b) (hαβ : α ≠ β)
  (h1 : perp a b) 
  (h2 : perpLP a α) 
  (h3 : perpLP b β) : 
  perpPP α β :=
sorry

end planes_perpendicular_l3598_359805


namespace helmet_pricing_and_purchase_l3598_359824

/-- Represents the types of helmets -/
inductive HelmetType
  | A
  | B

/-- Represents the wholesale and retail prices of helmets -/
structure HelmetPrices where
  wholesale : HelmetType → ℕ
  retail : HelmetType → ℕ

/-- Represents the sales data for helmets -/
structure SalesData where
  revenue : HelmetType → ℕ
  volume_ratio : ℕ  -- B's volume relative to A's

/-- Represents the purchase plan for helmets -/
structure PurchasePlan where
  total_helmets : ℕ
  budget : ℕ

/-- Main theorem statement -/
theorem helmet_pricing_and_purchase
  (prices : HelmetPrices)
  (sales : SalesData)
  (plan : PurchasePlan)
  (h1 : prices.wholesale HelmetType.A = 30)
  (h2 : prices.wholesale HelmetType.B = 20)
  (h3 : prices.retail HelmetType.A = prices.retail HelmetType.B + 15)
  (h4 : sales.revenue HelmetType.A = 450)
  (h5 : sales.revenue HelmetType.B = 600)
  (h6 : sales.volume_ratio = 2)
  (h7 : plan.total_helmets = 100)
  (h8 : plan.budget = 2350) :
  (prices.retail HelmetType.A = 45 ∧
   prices.retail HelmetType.B = 30) ∧
  (∀ m : ℕ, m * prices.wholesale HelmetType.A +
   (plan.total_helmets - m) * prices.wholesale HelmetType.B ≤ plan.budget →
   m ≤ 35) :=
sorry

end helmet_pricing_and_purchase_l3598_359824


namespace basketball_team_girls_l3598_359896

theorem basketball_team_girls (total : ℕ) (attended : ℕ) (boys : ℕ) (girls : ℕ) : 
  total = 30 →
  attended = 18 →
  boys + girls = total →
  boys + (girls / 3) = attended →
  boys = total - girls →
  girls = 18 := by
sorry

end basketball_team_girls_l3598_359896


namespace product_of_3_6_and_0_25_l3598_359894

theorem product_of_3_6_and_0_25 : (3.6 : ℝ) * 0.25 = 0.9 := by
  sorry

end product_of_3_6_and_0_25_l3598_359894


namespace colored_pencil_drawings_l3598_359809

theorem colored_pencil_drawings (total : ℕ) (blending_markers : ℕ) (charcoal : ℕ) 
  (h1 : total = 25)
  (h2 : blending_markers = 7)
  (h3 : charcoal = 4) :
  total - (blending_markers + charcoal) = 14 := by
  sorry

end colored_pencil_drawings_l3598_359809


namespace josh_marbles_l3598_359891

/-- The number of marbles Josh has after losing some, given the initial conditions. -/
theorem josh_marbles (colors : Nat) (initial_per_color : Nat) (lost_per_color : Nat) :
  colors = 5 →
  initial_per_color = 16 →
  lost_per_color = 7 →
  colors * (initial_per_color - lost_per_color) = 45 := by
  sorry

end josh_marbles_l3598_359891


namespace sqrt_nine_is_rational_l3598_359821

-- Define rationality
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = (p : ℝ) / (q : ℝ)

-- State the theorem
theorem sqrt_nine_is_rational : IsRational (Real.sqrt 9) := by
  sorry

end sqrt_nine_is_rational_l3598_359821


namespace function_not_in_first_quadrant_l3598_359804

theorem function_not_in_first_quadrant
  (a b : ℝ) (ha : 0 < a) (ha' : a < 1) (hb : b < -1) :
  ∀ x > 0, a^x + b < 0 := by
sorry

end function_not_in_first_quadrant_l3598_359804


namespace quadratic_form_sum_l3598_359878

theorem quadratic_form_sum (x : ℝ) : ∃ (a h k : ℝ),
  (5 * x^2 - 20 * x + 8 = a * (x - h)^2 + k) ∧ (a + h + k = -5) := by
  sorry

end quadratic_form_sum_l3598_359878


namespace card_ratio_l3598_359871

theorem card_ratio (total : ℕ) (difference : ℕ) (ellis : ℕ) (orion : ℕ) : 
  total = 500 → 
  difference = 50 → 
  ellis = orion + difference → 
  total = ellis + orion → 
  (ellis : ℚ) / (orion : ℚ) = 11 / 9 := by
sorry

end card_ratio_l3598_359871


namespace binary_conversion_and_sum_l3598_359874

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

def binary1 : List Bool := [true, true, false, true, false, true, true]
def binary2 : List Bool := [false, true, true, false, true, false, true]

theorem binary_conversion_and_sum :
  (binary_to_decimal binary1 = 107) ∧
  (binary_to_decimal binary2 = 86) ∧
  (binary_to_decimal binary1 + binary_to_decimal binary2 = 193) := by
  sorry

end binary_conversion_and_sum_l3598_359874


namespace simplify_expression_l3598_359801

theorem simplify_expression : 
  (Real.sqrt 450 / Real.sqrt 250) + (Real.sqrt 294 / Real.sqrt 147) = (3 * Real.sqrt 10 + 5 * Real.sqrt 2) / 5 := by
  sorry

end simplify_expression_l3598_359801


namespace triangle_properties_l3598_359850

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem -/
theorem triangle_properties (t : Triangle) 
  (h1 : (t.b - t.c) / t.a = (sin t.A - sin t.C) / (sin t.B + sin t.C))
  (h2 : t.a + t.c = 5)
  (h3 : 1/2 * t.a * t.c * sin t.B = 3 * sqrt 3 / 2) :
  t.B = π/3 ∧ t.b = sqrt 7 := by
  sorry

end triangle_properties_l3598_359850


namespace congruence_2023_mod_10_l3598_359879

theorem congruence_2023_mod_10 :
  ∀ n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] → n = 7 := by
  sorry

end congruence_2023_mod_10_l3598_359879


namespace beadshop_profit_l3598_359845

theorem beadshop_profit (total_profit : ℝ) (monday_fraction : ℝ) (tuesday_fraction : ℝ)
  (h_total : total_profit = 1200)
  (h_monday : monday_fraction = 1/3)
  (h_tuesday : tuesday_fraction = 1/4) :
  total_profit - (monday_fraction * total_profit + tuesday_fraction * total_profit) = 500 := by
  sorry

end beadshop_profit_l3598_359845


namespace rowing_current_velocity_l3598_359868

/-- Proves that the velocity of the current is 2 kmph given the conditions of the rowing problem. -/
theorem rowing_current_velocity
  (still_water_speed : ℝ)
  (distance : ℝ)
  (total_time : ℝ)
  (h1 : still_water_speed = 8)
  (h2 : distance = 7.5)
  (h3 : total_time = 2)
  : ∃ v : ℝ, v = 2 ∧ 
    (distance / (still_water_speed + v) + distance / (still_water_speed - v) = total_time) :=
by sorry

end rowing_current_velocity_l3598_359868


namespace exists_counterexample_l3598_359862

/-- A function is strictly monotonically increasing -/
def StrictlyIncreasing (f : ℚ → ℚ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The range of a function is the entire set of rationals -/
def SurjectiveOnRationals (f : ℚ → ℚ) : Prop :=
  ∀ y, ∃ x, f x = y

/-- The main theorem -/
theorem exists_counterexample : ∃ (f g : ℚ → ℚ),
  StrictlyIncreasing f ∧ StrictlyIncreasing g ∧
  SurjectiveOnRationals f ∧ SurjectiveOnRationals g ∧
  ¬SurjectiveOnRationals (λ x => f x + g x) := by
  sorry

end exists_counterexample_l3598_359862


namespace stacys_height_l3598_359807

/-- Stacy's height problem -/
theorem stacys_height (stacy_last_year : ℕ) (stacy_growth_diff : ℕ) (brother_growth : ℕ) :
  stacy_last_year = 50 →
  stacy_growth_diff = 6 →
  brother_growth = 1 →
  stacy_last_year + (stacy_growth_diff + brother_growth) = 57 :=
by sorry

end stacys_height_l3598_359807


namespace tangent_lines_to_unit_circle_l3598_359899

/-- The equation of a circle with radius 1 centered at the origin -/
def unitCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- A point on the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A line is tangent to the unit circle at a given point -/
def isTangent (l : Line) (p : Point) : Prop :=
  unitCircle p.x p.y ∧
  l.a * p.x + l.b * p.y + l.c = 0 ∧
  ∀ (x y : ℝ), unitCircle x y → (l.a * x + l.b * y + l.c = 0 → x = p.x ∧ y = p.y)

theorem tangent_lines_to_unit_circle :
  let p1 : Point := ⟨-1, 0⟩
  let p2 : Point := ⟨-1, 2⟩
  let l1 : Line := ⟨1, 0, 1⟩  -- Represents x = -1
  let l2 : Line := ⟨3, 4, -5⟩  -- Represents 3x + 4y - 5 = 0
  isTangent l1 p1 ∧ isTangent l2 p2 := by sorry

end tangent_lines_to_unit_circle_l3598_359899


namespace square_fold_perimeter_l3598_359852

/-- Given a square ABCD with side length 2, where C is folded to meet AD at point C' 
    such that C'D = 2/3, and BC intersects AB at point E, 
    the perimeter of triangle AEC' is (17√10 + √37) / 12 -/
theorem square_fold_perimeter (A B C D C' E : ℝ × ℝ) : 
  let square_side : ℝ := 2
  let C'D : ℝ := 2/3
  -- Define the square
  square_side = ‖A - B‖ ∧ square_side = ‖B - C‖ ∧ 
  square_side = ‖C - D‖ ∧ square_side = ‖D - A‖ ∧
  -- C' is on AD
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ C' = t • A + (1 - t) • D ∧
  -- E is on AB and BC
  ∃ s r : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ 0 ≤ r ∧ r ≤ 1 ∧ 
  E = s • A + (1 - s) • B ∧ E = r • B + (1 - r) • C ∧
  -- C'D condition
  ‖C' - D‖ = C'D
  →
  ‖A - E‖ + ‖E - C'‖ + ‖C' - A‖ = (17 * Real.sqrt 10 + Real.sqrt 37) / 12 :=
by sorry


end square_fold_perimeter_l3598_359852


namespace middle_digit_is_zero_l3598_359811

/-- Represents a three-digit number in a given base -/
structure ThreeDigitNumber (base : ℕ) where
  hundreds : ℕ
  tens : ℕ
  ones : ℕ
  valid_digits : hundreds < base ∧ tens < base ∧ ones < base

/-- The value of a three-digit number in decimal -/
def ThreeDigitNumber.value {base : ℕ} (n : ThreeDigitNumber base) : ℕ :=
  n.hundreds * base^2 + n.tens * base + n.ones

/-- A number M that satisfies the problem conditions -/
structure M where
  base5 : ThreeDigitNumber 5
  base8 : ThreeDigitNumber 8
  reversed_in_base8 : base8.hundreds = base5.ones ∧
                      base8.tens = base5.tens ∧
                      base8.ones = base5.hundreds
  same_value : base5.value = base8.value

theorem middle_digit_is_zero (m : M) : m.base5.tens = 0 := by
  sorry

end middle_digit_is_zero_l3598_359811


namespace quadratic_real_roots_condition_l3598_359888

theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) := by
  sorry

end quadratic_real_roots_condition_l3598_359888


namespace fiona_reaches_food_l3598_359859

-- Define the number of lily pads
def num_pads : ℕ := 16

-- Define the predator pads
def predator_pads : Set ℕ := {4, 7}

-- Define the food pad
def food_pad : ℕ := 12

-- Define Fiona's starting pad
def start_pad : ℕ := 0

-- Define the probability of hopping to the next pad
def hop_prob : ℚ := 1/2

-- Define the probability of jumping two pads
def jump_prob : ℚ := 1/2

-- Define a function to represent the probability of reaching a pad safely
def safe_prob : ℕ → ℚ := sorry

-- State the theorem
theorem fiona_reaches_food : safe_prob food_pad = 1/32 := by sorry

end fiona_reaches_food_l3598_359859


namespace ellipse_properties_l3598_359870

-- Define the ellipse C₁
def ellipse_C₁ (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the hyperbola C₂
def hyperbola_C₂ (x y : ℝ) : Prop := x^2 - y^2 / 2 = 1

-- Define the line that intersects C₁
def intersecting_line (x y : ℝ) : Prop := x + y = 1

-- Define the perimeter of triangle PF₁F₂
def triangle_perimeter (a c : ℝ) : Prop := 2 * a + 2 * c = 2 * Real.sqrt 3 + 2

-- Define the eccentricity range
def eccentricity_range (e : ℝ) : Prop := Real.sqrt 3 / 3 ≤ e ∧ e ≤ Real.sqrt 2 / 2

-- Define the condition for circle passing through origin
def circle_through_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁ * x₂ + y₁ * y₂ = 0

-- Main theorem
theorem ellipse_properties (a b c : ℝ) (h₁ : a > b) (h₂ : b > 0) 
  (h₃ : c = 1) -- Foci of C₁ are vertices of C₂
  (h₄ : triangle_perimeter a c) :
  -- 1. Equation of C₁
  (∀ x y, ellipse_C₁ x y a b ↔ x^2 / 3 + y^2 / 2 = 1) ∧
  -- 2. Length of chord AB
  (∃ x₁ y₁ x₂ y₂, 
    ellipse_C₁ x₁ y₁ a b ∧ 
    ellipse_C₁ x₂ y₂ a b ∧ 
    intersecting_line x₁ y₁ ∧ 
    intersecting_line x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (8 * Real.sqrt 3 / 5)^2) ∧
  -- 3. Range of major axis length
  (∀ e, eccentricity_range e →
    (∃ x₁ y₁ x₂ y₂, 
      ellipse_C₁ x₁ y₁ a b ∧ 
      ellipse_C₁ x₂ y₂ a b ∧ 
      intersecting_line x₁ y₁ ∧ 
      intersecting_line x₂ y₂ ∧
      circle_through_origin x₁ y₁ x₂ y₂) →
    Real.sqrt 5 ≤ 2 * a ∧ 2 * a ≤ Real.sqrt 6) :=
by sorry

end ellipse_properties_l3598_359870


namespace order_of_magnitude_l3598_359860

theorem order_of_magnitude (x : ℝ) (hx : 0.95 < x ∧ x < 1.05) :
  x < x^(x^(x^x)) ∧ x^(x^(x^x)) < x^x := by sorry

end order_of_magnitude_l3598_359860


namespace trigonometric_expression_evaluation_l3598_359812

open Real

theorem trigonometric_expression_evaluation (x : ℝ) 
  (f : ℝ → ℝ) 
  (hf : f = fun x ↦ sin x - cos x) 
  (hf' : deriv f = fun x ↦ 2 * f x) : 
  (1 + sin x ^ 2) / (cos x ^ 2 - sin (2 * x)) = -19/5 := by
sorry

end trigonometric_expression_evaluation_l3598_359812


namespace mass_NaHCO3_required_mass_NaHCO3_proof_l3598_359808

/-- The mass of NaHCO3 required to neutralize H2SO4 -/
theorem mass_NaHCO3_required (volume_H2SO4 : Real) (concentration_H2SO4 : Real) 
  (molar_mass_NaHCO3 : Real) (stoichiometric_ratio : Real) : Real :=
  let moles_H2SO4 := volume_H2SO4 * concentration_H2SO4
  let moles_NaHCO3 := moles_H2SO4 * stoichiometric_ratio
  let mass_NaHCO3 := moles_NaHCO3 * molar_mass_NaHCO3
  mass_NaHCO3

/-- Proof that the mass of NaHCO3 required is 0.525 g -/
theorem mass_NaHCO3_proof :
  mass_NaHCO3_required 0.025 0.125 84 2 = 0.525 := by
  sorry

end mass_NaHCO3_required_mass_NaHCO3_proof_l3598_359808


namespace two_year_growth_at_fifty_percent_l3598_359814

/-- The growth factor for a principal amount over a given number of years,
    given an annual interest rate and annual compounding. -/
def growthFactor (rate : ℝ) (years : ℕ) : ℝ :=
  (1 + rate) ^ years

/-- Theorem stating that with a 50% annual interest rate and 2 years of growth,
    the principal will grow by a factor of 2.25 -/
theorem two_year_growth_at_fifty_percent :
  growthFactor 0.5 2 = 2.25 := by
  sorry

end two_year_growth_at_fifty_percent_l3598_359814


namespace barbara_age_when_16_l3598_359839

theorem barbara_age_when_16 (mike_current_age barbara_current_age : ℕ) : 
  mike_current_age = 16 →
  barbara_current_age = mike_current_age / 2 →
  ∃ (mike_future_age : ℕ), mike_future_age = 24 ∧ mike_future_age - (mike_current_age - barbara_current_age) = 16 := by
  sorry

end barbara_age_when_16_l3598_359839


namespace geometric_sequence_from_arithmetic_l3598_359816

/-- Given three positive numbers in arithmetic sequence summing to 12,
    if adding 1, 4, and 11 to these numbers respectively results in terms b₂, b₃, and b₄
    of a geometric sequence, then the general term of this sequence is bₙ = 2ⁿ -/
theorem geometric_sequence_from_arithmetic (a d : ℝ) (h1 : 0 < a - d ∧ 0 < a ∧ 0 < a + d)
  (h2 : (a - d) + a + (a + d) = 12)
  (h3 : ∃ r : ℝ, (a - d + 1) * r = a + 4 ∧ (a + 4) * r = a + d + 11) :
  ∃ b : ℕ → ℝ, (∀ n : ℕ, b (n + 1) = 2 * b n) ∧ b 2 = a - d + 1 ∧ b 3 = a + 4 ∧ b 4 = a + d + 11 :=
sorry

end geometric_sequence_from_arithmetic_l3598_359816


namespace geometric_sequence_sum_l3598_359802

/-- A geometric sequence {a_n} -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 > 0 →
  a 4 + a 7 = 2 →
  a 5 * a 6 = -8 →
  a 1 + a 4 + a 7 + a 10 = -5 := by
  sorry

end geometric_sequence_sum_l3598_359802


namespace complement_intersection_theorem_necessary_not_sufficient_condition_l3598_359840

-- Define the sets A and B
def A : Set ℝ := {x | (x - 5) / (x - 2) ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}

-- Part 1
theorem complement_intersection_theorem :
  (Aᶜ ∩ (B 2)ᶜ) = {x | x ≤ 1 ∨ x > 5} := by sorry

-- Part 2
theorem necessary_not_sufficient_condition :
  (∀ x, x ∈ A → x ∈ B a) ∧ ¬(∀ x, x ∈ B a → x ∈ A) →
  3 ≤ a ∧ a ≤ 4 := by sorry

end complement_intersection_theorem_necessary_not_sufficient_condition_l3598_359840


namespace rectangle_area_calculation_l3598_359823

/-- Rectangle with known side and area -/
structure Rectangle1 where
  side : ℝ
  area : ℝ

/-- Rectangle similar to Rectangle1 with known diagonal -/
structure Rectangle2 where
  diagonal : ℝ

/-- The theorem to be proved -/
theorem rectangle_area_calculation
  (R1 : Rectangle1)
  (R2 : Rectangle2)
  (h1 : R1.side = 4)
  (h2 : R1.area = 32)
  (h3 : R2.diagonal = 10 * Real.sqrt 2)
  : ∃ (a : ℝ), a * (2 * a) = 80 := by
  sorry

end rectangle_area_calculation_l3598_359823


namespace amc_length_sum_amc_length_sum_value_l3598_359833

/-- The sum of lengths of line segments forming AMC on a unit-spaced grid --/
theorem amc_length_sum : ℝ := by
  -- Define the grid spacing
  let grid_spacing : ℝ := 1

  -- Define the lengths of different segments
  let a_diagonal : ℝ := Real.sqrt 2
  let a_horizontal : ℝ := 2
  let m_vertical : ℝ := 3
  let m_diagonal : ℝ := Real.sqrt 2
  let c_horizontal_long : ℝ := 3
  let c_horizontal_short : ℝ := 2

  -- Calculate the total length
  let total_length : ℝ := 
    2 * a_diagonal + a_horizontal + 
    2 * m_vertical + 2 * m_diagonal + 
    2 * c_horizontal_long + c_horizontal_short

  -- Prove that the total length equals 13 + 4√2
  sorry

/-- The result of amc_length_sum is equal to 13 + 4√2 --/
theorem amc_length_sum_value : amc_length_sum = 13 + 4 * Real.sqrt 2 := by
  sorry

end amc_length_sum_amc_length_sum_value_l3598_359833


namespace michelle_taxi_ride_cost_l3598_359828

/-- Calculate the total cost of a taxi ride given the initial fee, distance, and per-mile charge. -/
def taxiRideCost (initialFee : ℝ) (distance : ℝ) (chargePerMile : ℝ) : ℝ :=
  initialFee + distance * chargePerMile

/-- Theorem stating that Michelle's taxi ride cost $12 -/
theorem michelle_taxi_ride_cost :
  taxiRideCost 2 4 2.5 = 12 := by
  sorry

end michelle_taxi_ride_cost_l3598_359828


namespace vector_relations_l3598_359826

-- Define the vectors
def a : ℝ × ℝ := (3, -2)
def b (y : ℝ) : ℝ × ℝ := (-1, y)
def c (x : ℝ) : ℝ × ℝ := (x, 5)

-- Define perpendicularity for 2D vectors
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

-- Define parallelism for 2D vectors
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

theorem vector_relations :
  (∀ y : ℝ, perpendicular a (b y) → y = 3/2) ∧
  (∀ x : ℝ, parallel a (c x) → x = 15/2) := by sorry

end vector_relations_l3598_359826


namespace stating_three_pairs_l3598_359800

/-- 
A function that returns the number of ordered pairs (m,n) of positive integers 
satisfying m ≥ n and m^2 - n^2 = 72
-/
def count_pairs : ℕ := 
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ p.1 ≥ p.2 ∧ p.1^2 - p.2^2 = 72) 
    (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- 
Theorem stating that there are exactly 3 ordered pairs (m,n) of positive integers 
satisfying m ≥ n and m^2 - n^2 = 72
-/
theorem three_pairs : count_pairs = 3 := by
  sorry

end stating_three_pairs_l3598_359800


namespace arithmetic_geometric_inequality_l3598_359893

theorem arithmetic_geometric_inequality (n : ℕ) (a b : ℕ → ℝ) 
  (h1 : a 1 = b 1) 
  (h2 : a 1 > 0)
  (h3 : a (2*n+1) = b (2*n+1))
  (h4 : ∀ k, a (k+1) - a k = a 2 - a 1)  -- arithmetic sequence condition
  (h5 : ∀ k, b (k+1) / b k = b 2 / b 1)  -- geometric sequence condition
  : a (n+1) ≥ b (n+1) := by
  sorry

end arithmetic_geometric_inequality_l3598_359893


namespace cubes_in_box_l3598_359837

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular box -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Calculates the number of cubes that can fit in each dimension -/
def cubesPerDimension (boxDim : ℕ) (cubeDim : ℕ) : ℕ :=
  boxDim / cubeDim

/-- Calculates the total number of cubes that can fit in the box -/
def totalCubes (box : BoxDimensions) (cubeDim : ℕ) : ℕ :=
  (cubesPerDimension box.length cubeDim) *
  (cubesPerDimension box.width cubeDim) *
  (cubesPerDimension box.height cubeDim)

/-- Calculates the volume of a cube -/
def cubeVolume (cubeDim : ℕ) : ℕ :=
  cubeDim ^ 3

/-- Calculates the total volume of all cubes in the box -/
def totalCubesVolume (box : BoxDimensions) (cubeDim : ℕ) : ℕ :=
  (totalCubes box cubeDim) * (cubeVolume cubeDim)

/-- Theorem: The number of 4-inch cubes that can fit in the box is 6,
    and they occupy 100% of the box's volume -/
theorem cubes_in_box (box : BoxDimensions)
    (h1 : box.length = 8)
    (h2 : box.width = 4)
    (h3 : box.height = 12)
    (cubeDim : ℕ)
    (h4 : cubeDim = 4) :
    totalCubes box cubeDim = 6 ∧
    totalCubesVolume box cubeDim = boxVolume box := by
  sorry


end cubes_in_box_l3598_359837


namespace pyramid_apex_distance_l3598_359864

/-- Pyramid structure with a square base -/
structure Pyramid :=
  (base_side : ℝ)
  (height : ℝ)
  (sphere_radius : ℝ)

/-- The distance between the center of the base and the apex of the pyramid -/
def apex_distance (p : Pyramid) : ℝ := sorry

theorem pyramid_apex_distance (p : Pyramid) 
  (h1 : p.base_side = 2 * Real.sqrt 2)
  (h2 : p.height = 1)
  (h3 : p.sphere_radius = 2 * Real.sqrt 2) :
  apex_distance p = 2 * Real.sqrt 2 := by sorry

end pyramid_apex_distance_l3598_359864


namespace necessary_sufficient_condition_for_x_squared_lt_4_l3598_359818

theorem necessary_sufficient_condition_for_x_squared_lt_4 :
  ∀ x : ℝ, x^2 < 4 ↔ -2 ≤ x ∧ x ≤ 2 := by
sorry

end necessary_sufficient_condition_for_x_squared_lt_4_l3598_359818


namespace min_value_expression_l3598_359854

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 4) :
  (2 * x / y) + (3 * y / z) + (4 * z / x) ≥ 6 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a * b * c = 4 ∧
    (2 * a / b) + (3 * b / c) + (4 * c / a) = 6 :=
by sorry

end min_value_expression_l3598_359854


namespace sum_of_fractions_equals_one_l3598_359842

theorem sum_of_fractions_equals_one 
  (p q r u v w : ℝ) 
  (eq1 : 17 * u + q * v + r * w = 0)
  (eq2 : p * u + 29 * v + r * w = 0)
  (eq3 : p * u + q * v + 56 * w = 0)
  (h_p : p ≠ 17)
  (h_u : u ≠ 0) :
  p / (p - 17) + q / (q - 29) + r / (r - 56) = 1 := by
  sorry

end sum_of_fractions_equals_one_l3598_359842


namespace downstream_distance_man_downstream_distance_l3598_359866

/-- Calculates the downstream distance given swimming conditions -/
theorem downstream_distance (v_m : ℝ) (t : ℝ) (d_upstream : ℝ) : ℝ :=
  let v_s := v_m - d_upstream / t
  let v_downstream := v_m + v_s
  v_downstream * t

/-- Proves the downstream distance for the given problem conditions -/
theorem man_downstream_distance :
  downstream_distance 4 6 18 = 30 := by
  sorry

end downstream_distance_man_downstream_distance_l3598_359866


namespace point_on_line_l3598_359884

/-- Given a point P(2, m) lying on the line 3x + y = 2, prove that m = -4 -/
theorem point_on_line (m : ℝ) : (3 * 2 + m = 2) → m = -4 := by
  sorry

end point_on_line_l3598_359884


namespace max_actors_is_five_l3598_359834

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  -- The number of actors in the tournament
  num_actors : ℕ
  -- The results of games between actors
  results : Fin num_actors → Fin num_actors → ℝ
  -- Each actor plays every other actor exactly once
  played_once : ∀ i j : Fin num_actors, i ≠ j → results i j + results j i = 1
  -- Scores are either 0, 0.5, or 1
  valid_scores : ∀ i j : Fin num_actors, results i j ∈ ({0, 0.5, 1} : Set ℝ)
  -- For any three actors, one earned exactly 1.5 against the other two
  trio_condition : ∀ i j k : Fin num_actors, i ≠ j ∧ j ≠ k ∧ i ≠ k →
    (results i j + results i k = 1.5) ∨
    (results j i + results j k = 1.5) ∨
    (results k i + results k j = 1.5)

/-- The maximum number of actors in a valid chess tournament is 5 -/
theorem max_actors_is_five :
  (∃ t : ChessTournament, t.num_actors = 5) ∧
  (∀ t : ChessTournament, t.num_actors ≤ 5) :=
sorry

end max_actors_is_five_l3598_359834


namespace sum_of_integers_with_product_seven_cubed_l3598_359881

theorem sum_of_integers_with_product_seven_cubed :
  ∃ (a b c : ℕ+),
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    (a * b * c : ℕ) = 7^3 →
    (a : ℕ) + (b : ℕ) + (c : ℕ) = 57 := by
  sorry

end sum_of_integers_with_product_seven_cubed_l3598_359881


namespace uncle_joe_parking_probability_l3598_359887

/-- The number of parking spaces -/
def total_spaces : ℕ := 18

/-- The number of cars that have parked -/
def parked_cars : ℕ := 14

/-- The number of adjacent spaces required for Uncle Joe's truck -/
def required_spaces : ℕ := 2

/-- The probability of finding two adjacent empty spaces -/
def probability_of_parking : ℚ := 113 / 204

theorem uncle_joe_parking_probability :
  probability_of_parking = 1 - (Nat.choose (total_spaces - parked_cars + required_spaces - 1) (required_spaces - 1)) / (Nat.choose total_spaces parked_cars) :=
by sorry

end uncle_joe_parking_probability_l3598_359887


namespace inequality_holds_l3598_359876

theorem inequality_holds (x y z : ℝ) : 4 * x * (x + y) * (x + z) * (x + y + z) + y^2 * z^2 ≥ 0 := by
  sorry

end inequality_holds_l3598_359876


namespace andy_wrong_answers_l3598_359885

/-- Represents the number of wrong answers for each person -/
structure WrongAnswers where
  andy : ℕ
  beth : ℕ
  charlie : ℕ
  daniel : ℕ

/-- The test conditions -/
def testConditions (w : WrongAnswers) : Prop :=
  w.andy + w.beth = w.charlie + w.daniel ∧
  w.andy + w.daniel = w.beth + w.charlie + 6 ∧
  w.charlie = 7

theorem andy_wrong_answers (w : WrongAnswers) :
  testConditions w → w.andy = 20 := by
  sorry

#check andy_wrong_answers

end andy_wrong_answers_l3598_359885


namespace new_person_weight_l3598_359820

/-- Given a group of 5 people where one person weighing 40 kg is replaced,
    resulting in an average weight increase of 10 kg, prove that the new
    person weighs 90 kg. -/
theorem new_person_weight (initial_count : Nat) (weight_replaced : Real) 
  (avg_increase : Real) (new_weight : Real) : 
  initial_count = 5 → 
  weight_replaced = 40 → 
  avg_increase = 10 → 
  new_weight = weight_replaced + (initial_count * avg_increase) → 
  new_weight = 90 := by
  sorry

end new_person_weight_l3598_359820


namespace rhombus_perimeter_l3598_359898

/-- The perimeter of a rhombus given its diagonals -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 10) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 52 := by
  sorry

end rhombus_perimeter_l3598_359898


namespace inequality_holds_iff_a_in_range_l3598_359831

/-- The inequality holds for all real x and θ ∈ [0, π/2] if and only if a is in the specified range -/
theorem inequality_holds_iff_a_in_range (a : ℝ) : 
  (∀ (x : ℝ) (θ : ℝ), 0 ≤ θ ∧ θ ≤ Real.pi / 2 → 
    (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8) ↔ 
  (a ≤ Real.sqrt 6 ∨ a ≥ 7/2) := by sorry

end inequality_holds_iff_a_in_range_l3598_359831


namespace power_sum_equality_l3598_359897

theorem power_sum_equality : (-2)^2004 + 3 * (-2)^2003 = -2^2003 := by
  sorry

end power_sum_equality_l3598_359897


namespace cost_price_calculation_l3598_359858

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) (cost_price : ℝ) : 
  selling_price = 648 →
  profit_percentage = 0.08 →
  selling_price = cost_price * (1 + profit_percentage) →
  cost_price = 600 := by
  sorry

end cost_price_calculation_l3598_359858


namespace right_triangle_sine_roots_l3598_359857

theorem right_triangle_sine_roots (n : ℤ) (A B C : ℝ) : 
  A + B = π / 2 →
  (∃ (x y : ℝ), x = Real.sin A ∧ y = Real.sin B ∧ 
    (5 * n + 8 : ℝ) * x^2 - (7 * n - 20 : ℝ) * x + 120 = 0 ∧
    (5 * n + 8 : ℝ) * y^2 - (7 * n - 20 : ℝ) * y + 120 = 0) →
  n = 66 := by
sorry

end right_triangle_sine_roots_l3598_359857


namespace simplify_fraction_l3598_359880

theorem simplify_fraction (x y : ℝ) (hx : x = 3) (hy : y = 4) :
  (15 * x^2 * y^3) / (9 * x * y^2) = 20 := by
  sorry

end simplify_fraction_l3598_359880


namespace rectangle_diagonal_problem_l3598_359819

theorem rectangle_diagonal_problem (w l : ℝ) 
  (h1 : w^2 + l^2 = 400) 
  (h2 : 4*w^2 + l^2 = 484) : 
  w^2 = 28 := by
sorry

end rectangle_diagonal_problem_l3598_359819


namespace girls_percentage_in_class_l3598_359841

theorem girls_percentage_in_class (total_students : ℕ) 
  (boys_basketball_percentage : ℚ)
  (girls_basketball_ratio : ℚ)
  (girls_basketball_percentage : ℚ) :
  total_students = 25 →
  boys_basketball_percentage = 2/5 →
  girls_basketball_ratio = 2 →
  girls_basketball_percentage = 4/5 →
  (∃ (girls_percentage : ℚ), girls_percentage = 3/5) :=
by sorry

end girls_percentage_in_class_l3598_359841


namespace circumcenter_minimizes_max_distance_l3598_359836

/-- A point in a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A triangle in a 2D plane. -/
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

/-- The distance between two points. -/
def distance (p1 p2 : Point2D) : ℝ := sorry

/-- Checks if a point is inside or on the boundary of a triangle. -/
def isInsideOrOnBoundary (p : Point2D) (t : Triangle) : Prop := sorry

/-- Checks if a triangle is acute or right. -/
def isAcuteOrRight (t : Triangle) : Prop := sorry

/-- The circumcenter of a triangle. -/
def circumcenter (t : Triangle) : Point2D := sorry

/-- Theorem: The point that minimizes the maximum distance to the vertices of an acute or right triangle is its circumcenter. -/
theorem circumcenter_minimizes_max_distance (t : Triangle) (h : isAcuteOrRight t) :
  ∀ p, isInsideOrOnBoundary p t →
    distance p t.A ≤ distance (circumcenter t) t.A ∧
    distance p t.B ≤ distance (circumcenter t) t.B ∧
    distance p t.C ≤ distance (circumcenter t) t.C :=
  sorry

end circumcenter_minimizes_max_distance_l3598_359836


namespace rain_probability_l3598_359895

theorem rain_probability (p : ℝ) (n : ℕ) (hp : p = 3 / 4) (hn : n = 4) :
  1 - (1 - p)^n = 255 / 256 := by
  sorry

end rain_probability_l3598_359895


namespace harkamal_fruit_purchase_l3598_359873

/-- Calculates the final amount paid after discount for a fruit purchase --/
def final_amount_paid (grapes_kg : ℝ) (grapes_price : ℝ) (mangoes_kg : ℝ) (mangoes_price : ℝ)
  (oranges_kg : ℝ) (oranges_price : ℝ) (bananas_kg : ℝ) (bananas_price : ℝ) (discount_rate : ℝ) : ℝ :=
  let total_cost := grapes_kg * grapes_price + mangoes_kg * mangoes_price +
                    oranges_kg * oranges_price + bananas_kg * bananas_price
  let discount := discount_rate * total_cost
  total_cost - discount

/-- Theorem stating the final amount paid for Harkamal's fruit purchase --/
theorem harkamal_fruit_purchase :
  final_amount_paid 3 70 9 55 5 40 7 20 0.1 = 940.5 := by
  sorry

end harkamal_fruit_purchase_l3598_359873


namespace hyperbola_eccentricity_l3598_359835

/-- A hyperbola with foci at (-1,0) and (1,0) -/
structure Hyperbola where
  leftFocus : ℝ × ℝ := (-1, 0)
  rightFocus : ℝ × ℝ := (1, 0)

/-- The parabola y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Vector from point a to point b -/
def vector (a b : ℝ × ℝ) : ℝ × ℝ := (b.1 - a.1, b.2 - a.2)

theorem hyperbola_eccentricity (C : Hyperbola) (P : ℝ × ℝ) :
  parabola P.1 P.2 →
  let F₁ := C.leftFocus
  let F₂ := C.rightFocus
  dot_product (vector F₂ P + vector F₂ F₁) (vector F₂ P - vector F₂ F₁) = 0 →
  eccentricity C = 1 + Real.sqrt 2 := by sorry

end hyperbola_eccentricity_l3598_359835


namespace complex_sum_powers_l3598_359892

theorem complex_sum_powers (z : ℂ) (hz : z^5 + z + 1 = 0) :
  z^103 + z^104 + z^105 + z^106 + z^107 + z^108 = 0 := by
  sorry

end complex_sum_powers_l3598_359892


namespace cubic_sum_problem_l3598_359872

theorem cubic_sum_problem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 12) / a = (b^3 + 12) / b ∧ (b^3 + 12) / b = (c^3 + 12) / c) : 
  a^3 + b^3 + c^3 = -36 := by
  sorry

end cubic_sum_problem_l3598_359872


namespace first_robber_guarantee_l3598_359846

/-- Represents the coin division game between two robbers --/
structure CoinGame where
  totalCoins : ℕ
  maxBags : ℕ

/-- Represents the outcome of the game for the first robber --/
def FirstRobberOutcome (game : CoinGame) : ℕ := 
  min game.totalCoins (game.totalCoins - (game.maxBags - 1) * (game.totalCoins / (2 * game.maxBags - 1)))

/-- Theorem stating the guaranteed minimum coins for the first robber --/
theorem first_robber_guarantee (game : CoinGame) 
  (h1 : game.totalCoins = 300) 
  (h2 : game.maxBags = 11) : 
  FirstRobberOutcome game ≥ 146 := by
  sorry

#eval FirstRobberOutcome { totalCoins := 300, maxBags := 11 }

end first_robber_guarantee_l3598_359846


namespace problem_statement_l3598_359830

theorem problem_statement (x y z w : ℝ) 
  (h : (x - y) * (z - w) / ((y - z) * (w - x)) = 1 / 3) : 
  (x - z) * (y - w) / ((x - y) * (z - w)) = 2 := by
  sorry

end problem_statement_l3598_359830


namespace factorial_17_digit_sum_l3598_359813

theorem factorial_17_digit_sum : ∃ (T M H : ℕ),
  (T < 10 ∧ M < 10 ∧ H < 10) ∧
  H = 0 ∧
  (T + M + 35) % 3 = 0 ∧
  (T - M - 2) % 11 = 0 ∧
  T + M + H = 10 :=
by
  sorry

end factorial_17_digit_sum_l3598_359813


namespace sixth_root_of_unity_product_l3598_359827

theorem sixth_root_of_unity_product (s : ℂ) (h1 : s^6 = 1) (h2 : s ≠ 1) :
  (s - 1) * (s^2 - 1) * (s^3 - 1) * (s^4 - 1) * (s^5 - 1) = -6 := by
  sorry

end sixth_root_of_unity_product_l3598_359827


namespace cube_inequality_l3598_359865

theorem cube_inequality (a b : ℝ) : a > b → a^3 > b^3 := by
  sorry

end cube_inequality_l3598_359865


namespace max_snacks_is_14_l3598_359815

/-- Represents the maximum number of snacks that can be purchased with a given budget and pricing options. -/
def max_snacks (budget : ℕ) (single_price : ℕ) (pack4_price : ℕ) (pack6_price : ℕ) : ℕ :=
  sorry

/-- Theorem: Given the specific pricing and budget, the maximum number of snacks is 14. -/
theorem max_snacks_is_14 :
  max_snacks 20 2 6 8 = 14 := by
  sorry

end max_snacks_is_14_l3598_359815


namespace polygon_line_theorem_l3598_359853

/-- A polygon is represented as a set of points in the plane. -/
def Polygon : Type := Set (ℝ × ℝ)

/-- A line in the plane. -/
def Line : Type := Set (ℝ × ℝ)

/-- The number of sides in a polygon. -/
def numSides (p : Polygon) : ℕ := sorry

/-- A side of a polygon is contained in a line. -/
def sideInLine (p : Polygon) (l : Line) : Prop := sorry

/-- A line contains exactly one side of a polygon. -/
def lineContainsExactlyOneSide (p : Polygon) (l : Line) : Prop := sorry

theorem polygon_line_theorem :
  (∀ p : Polygon, numSides p = 13 → ∃ l : Line, lineContainsExactlyOneSide p l) ∧
  (∀ n : ℕ, n > 13 → ∃ p : Polygon, numSides p = n ∧ 
    ∀ l : Line, sideInLine p l → ∃ l' : Line, l ≠ l' ∧ sideInLine p l') :=
sorry

end polygon_line_theorem_l3598_359853


namespace line_circle_no_intersection_l3598_359829

/-- The line 3x + 4y = 12 and the circle x^2 + y^2 = 4 have no points of intersection in the real plane. -/
theorem line_circle_no_intersection :
  ¬ ∃ (x y : ℝ), (3 * x + 4 * y = 12) ∧ (x^2 + y^2 = 4) := by
  sorry

end line_circle_no_intersection_l3598_359829


namespace binomial_distribution_n_l3598_359886

/-- A binomial distribution with parameters n and p -/
structure BinomialDistribution where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial distribution -/
def expectation (X : BinomialDistribution) : ℝ :=
  X.n * X.p

/-- The variance of a binomial distribution -/
def variance (X : BinomialDistribution) : ℝ :=
  X.n * X.p * (1 - X.p)

theorem binomial_distribution_n (X : BinomialDistribution) 
  (h_exp : expectation X = 15)
  (h_var : variance X = 11.25) :
  X.n = 60 := by
  sorry

end binomial_distribution_n_l3598_359886


namespace arithmetic_sequence_count_l3598_359867

theorem arithmetic_sequence_count (a₁ aₙ d : ℤ) (h₁ : a₁ = 150) (h₂ : aₙ = 42) (h₃ : d = -4) :
  (a₁ - aₙ) / d + 1 = 54 := by
  sorry

end arithmetic_sequence_count_l3598_359867


namespace intersection_A_B_l3598_359863

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B : Set ℝ := {2, 3}

theorem intersection_A_B : A ∩ B = {2} := by sorry

end intersection_A_B_l3598_359863


namespace total_frisbees_sold_l3598_359843

/-- Represents the number of frisbees sold at $3 each -/
def x : ℕ := sorry

/-- Represents the number of frisbees sold at $4 each -/
def y : ℕ := sorry

/-- The total receipts from frisbee sales -/
def total_receipts : ℕ := 200

/-- The minimum number of $4 frisbees sold -/
def min_four_dollar_frisbees : ℕ := 20

/-- Theorem stating the total number of frisbees sold -/
theorem total_frisbees_sold :
  (3 * x + 4 * y = total_receipts) →
  (y ≥ min_four_dollar_frisbees) →
  (x + y = 60) := by
  sorry

end total_frisbees_sold_l3598_359843


namespace relay_race_arrangements_l3598_359889

/-- The number of people in the class -/
def total_people : ℕ := 5

/-- The number of people to be selected for the race -/
def selected_people : ℕ := 4

/-- The set of possible first runners -/
inductive FirstRunner
| A
| B
| C

/-- The set of possible last runners -/
inductive LastRunner
| A
| B

/-- The total number of different arrangements for the order of runners -/
def total_arrangements : ℕ := 24

theorem relay_race_arrangements :
  total_arrangements = 24 :=
sorry

end relay_race_arrangements_l3598_359889


namespace functional_equation_solution_l3598_359877

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x^2 - y^2) = x * f x - y * f y) :
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x :=
by sorry

end functional_equation_solution_l3598_359877


namespace exists_valid_arrangement_l3598_359855

-- Define a structure for the circle arrangement
structure CircleArrangement where
  numbers : List ℕ
  connections : List (ℕ × ℕ)

-- Define the property of valid ratios for connected circles
def validConnectedRatio (a b : ℕ) : Prop :=
  a / b = 3 ∨ a / b = 9 ∨ b / a = 3 ∨ b / a = 9

-- Define the property of invalid ratios for unconnected circles
def invalidUnconnectedRatio (a b : ℕ) : Prop :=
  a / b ≠ 3 ∧ a / b ≠ 9 ∧ b / a ≠ 3 ∧ b / a ≠ 9

-- Define the property of a valid circle arrangement
def validArrangement (arr : CircleArrangement) : Prop :=
  (∀ (a b : ℕ), (a, b) ∈ arr.connections → validConnectedRatio a b) ∧
  (∀ (a b : ℕ), a ∈ arr.numbers ∧ b ∈ arr.numbers ∧ (a, b) ∉ arr.connections → invalidUnconnectedRatio a b)

-- Theorem stating the existence of a valid arrangement
theorem exists_valid_arrangement : ∃ (arr : CircleArrangement), validArrangement arr :=
  sorry

end exists_valid_arrangement_l3598_359855


namespace y_derivative_l3598_359849

noncomputable def y (x : ℝ) : ℝ := (1 - x^2) / Real.sin x

theorem y_derivative (x : ℝ) (h : Real.sin x ≠ 0) : 
  deriv y x = (-2*x*Real.sin x - (1 - x^2)*Real.cos x) / (Real.sin x)^2 :=
by sorry

end y_derivative_l3598_359849


namespace ellipse_standard_equation_l3598_359806

def ellipse_equation (x y : ℝ) : Prop := x^2 / 36 + y^2 / 20 = 1

theorem ellipse_standard_equation 
  (major_axis : ℝ) (eccentricity : ℝ) (foci_on_x_axis : Prop) :
  major_axis = 12 → eccentricity = 2/3 → foci_on_x_axis →
  ∀ x y : ℝ, ellipse_equation x y ↔ 
    x^2 / ((major_axis/2)^2) + y^2 / ((major_axis/2)^2 * (1 - eccentricity^2)) = 1 :=
sorry

end ellipse_standard_equation_l3598_359806


namespace clock_angle_at_3_30_clock_angle_at_3_30_is_75_l3598_359882

/-- The smaller angle between clock hands at 3:30 -/
theorem clock_angle_at_3_30 : ℝ :=
  let total_hours : ℕ := 12
  let total_degrees : ℝ := 360
  let degrees_per_hour : ℝ := total_degrees / total_hours
  let minute_hand_position : ℝ := 180
  let hour_hand_position : ℝ := 3 * degrees_per_hour + degrees_per_hour / 2
  let angle_difference : ℝ := |minute_hand_position - hour_hand_position|
  min angle_difference (total_degrees - angle_difference)

/-- Proof that the smaller angle between clock hands at 3:30 is 75 degrees -/
theorem clock_angle_at_3_30_is_75 : clock_angle_at_3_30 = 75 := by
  sorry

end clock_angle_at_3_30_clock_angle_at_3_30_is_75_l3598_359882


namespace binary_111100_is_even_l3598_359856

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem binary_111100_is_even :
  let binary := [false, false, true, true, true, true]
  is_even (binary_to_decimal binary) := by
  sorry

end binary_111100_is_even_l3598_359856


namespace long_jump_competition_l3598_359848

theorem long_jump_competition (first second third fourth : ℝ) : 
  first = 22 →
  second > first →
  third = second - 2 →
  fourth = third + 3 →
  fourth = 24 →
  second - first = 1 := by
sorry

end long_jump_competition_l3598_359848


namespace function_properties_l3598_359875

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x^2 - a*x + 1)

theorem function_properties (a : ℝ) :
  (∀ x y : ℝ, y = f a 0 → 3*x + y - 1 = 0 → x = 0) →
  a = 4 ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-1-ε) (-1+ε), f 4 (-1) ≥ f 4 x) ∧
  (∃ ε > 0, ∀ x ∈ Set.Ioo (3-ε) (3+ε), f 4 3 ≤ f 4 x) ∧
  f 4 (-1) = 6 / Real.exp 1 ∧
  f 4 3 = -2 * Real.exp 3 :=
sorry

end function_properties_l3598_359875


namespace monotonic_increasing_range_l3598_359803

/-- A function f(x) = e^x(x^2 + 2ax + 2) is monotonically increasing on R if and only if a is in the range [-1, 1] -/
theorem monotonic_increasing_range (a : ℝ) :
  (∀ x : ℝ, Monotone (fun x => Real.exp x * (x^2 + 2*a*x + 2))) ↔ a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry

end monotonic_increasing_range_l3598_359803


namespace smallest_domain_size_l3598_359825

-- Define the function f
def f : ℕ → ℕ
| 7 => 22
| n => if n % 2 = 1 then 3 * n + 1 else n / 2

-- Define the sequence of f applications starting from 7
def fSequence : ℕ → ℕ
| 0 => 7
| n + 1 => f (fSequence n)

-- Define the set of unique elements in the sequence
def uniqueElements (n : ℕ) : Finset ℕ :=
  (Finset.range (n + 1)).image fSequence

-- Theorem statement
theorem smallest_domain_size :
  ∃ n : ℕ, (uniqueElements n).card = 13 ∧
  ∀ m : ℕ, m < n → (uniqueElements m).card < 13 :=
sorry

end smallest_domain_size_l3598_359825


namespace banana_orange_equivalence_l3598_359832

/-- The value of one banana in terms of oranges -/
def banana_value : ℚ := 1

/-- The number of bananas that are worth as much as 9 oranges -/
def bananas_equal_to_9_oranges : ℚ := (3/4) * 12

/-- The number of bananas we want to find the orange equivalent for -/
def bananas_to_convert : ℚ := (1/3) * 6

theorem banana_orange_equivalence : 
  banana_value * bananas_to_convert = 2 := by sorry

end banana_orange_equivalence_l3598_359832


namespace seven_balls_three_boxes_l3598_359838

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distributeBalls (n : ℕ) (k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 36 ways to distribute 7 indistinguishable balls into 3 distinguishable boxes -/
theorem seven_balls_three_boxes : distributeBalls 7 3 = 36 := by
  sorry

end seven_balls_three_boxes_l3598_359838
