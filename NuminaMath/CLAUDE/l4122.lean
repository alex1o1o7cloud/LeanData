import Mathlib

namespace NUMINAMATH_CALUDE_sprained_wrist_frosting_time_l4122_412224

/-- The time it takes Ann to frost a cake with her sprained wrist -/
def sprained_wrist_time : ℝ := 8

/-- The normal time it takes Ann to frost a cake -/
def normal_time : ℝ := 5

/-- The additional time it takes to frost 10 cakes with a sprained wrist -/
def additional_time : ℝ := 30

theorem sprained_wrist_frosting_time :
  sprained_wrist_time = (10 * normal_time + additional_time) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sprained_wrist_frosting_time_l4122_412224


namespace NUMINAMATH_CALUDE_parabola_intersection_locus_locus_nature_l4122_412282

/-- Given a parabola and a point in its plane, this theorem describes the locus of 
    intersection points formed by certain lines related to the parabola. -/
theorem parabola_intersection_locus 
  (p : ℝ) -- Parameter of the parabola
  (α β : ℝ) -- Coordinates of point A
  (x y : ℝ) -- Coordinates of the locus point M
  (h_parabola : y^2 = 2*p*x) -- Equation of the parabola
  : 2*p*x^2 - β*x*y + α*y^2 - 2*p*α*x = 0 := by
  sorry

/-- This theorem characterizes the nature of the locus based on the position of point A 
    relative to the parabola. -/
theorem locus_nature 
  (p : ℝ) -- Parameter of the parabola
  (α β : ℝ) -- Coordinates of point A
  : (β^2 = 8*p*α → IsParabola) ∧ 
    (β^2 < 8*p*α → IsEllipse) ∧ 
    (β^2 > 8*p*α → IsHyperbola) := by
  sorry

-- We need to define these predicates
axiom IsParabola : Prop
axiom IsEllipse : Prop
axiom IsHyperbola : Prop

end NUMINAMATH_CALUDE_parabola_intersection_locus_locus_nature_l4122_412282


namespace NUMINAMATH_CALUDE_third_quadrant_condition_l4122_412223

def is_in_third_quadrant (z : ℂ) : Prop :=
  z.re < 0 ∧ z.im < 0

theorem third_quadrant_condition (a : ℝ) :
  is_in_third_quadrant ((1 + I) * (a - I)) ↔ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_third_quadrant_condition_l4122_412223


namespace NUMINAMATH_CALUDE_arc_length_of_sector_l4122_412200

/-- Given a circular sector with radius 5 cm and area 11.25 cm², 
    prove that the length of the arc is 4.5 cm. -/
theorem arc_length_of_sector (r : ℝ) (area : ℝ) (arc_length : ℝ) : 
  r = 5 → 
  area = 11.25 → 
  arc_length = r * (2 * area / (r * r)) → 
  arc_length = 4.5 := by
sorry

end NUMINAMATH_CALUDE_arc_length_of_sector_l4122_412200


namespace NUMINAMATH_CALUDE_max_notebooks_purchasable_l4122_412260

def available_funds : ℚ := 21.45
def notebook_cost : ℚ := 2.75

theorem max_notebooks_purchasable :
  ∀ n : ℕ, (n : ℚ) * notebook_cost ≤ available_funds ↔ n ≤ 7 :=
by sorry

end NUMINAMATH_CALUDE_max_notebooks_purchasable_l4122_412260


namespace NUMINAMATH_CALUDE_rick_cheese_servings_l4122_412277

/-- Calculates the number of cheese servings eaten given the remaining calories -/
def servingsEaten (caloriesPerServing : ℕ) (servingsPerBlock : ℕ) (remainingCalories : ℕ) : ℕ :=
  (caloriesPerServing * servingsPerBlock - remainingCalories) / caloriesPerServing

theorem rick_cheese_servings :
  servingsEaten 110 16 1210 = 5 := by
  sorry

end NUMINAMATH_CALUDE_rick_cheese_servings_l4122_412277


namespace NUMINAMATH_CALUDE_no_point_M_exists_line_EF_exists_l4122_412272

-- Define the ellipse C
def C (x y : ℝ) : Prop := (x - 2)^2 + y^2/4 = 1

-- Define the line l
def l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define point R
def R : ℝ × ℝ := (1, 4)

-- Theorem 1: No point M exists inside C satisfying the given condition
theorem no_point_M_exists : ¬ ∃ M : ℝ × ℝ, 
  C M.1 M.2 ∧ 
  (∀ Q A B : ℝ × ℝ, 
    l Q.1 Q.2 → 
    C A.1 A.2 → 
    C B.1 B.2 → 
    (∃ t : ℝ, M.1 = t * (Q.1 - M.1) + M.1 ∧ M.2 = t * (Q.2 - M.2) + M.2) →
    (A.1 - M.1)^2 + (A.2 - M.2)^2 = (B.1 - M.1)^2 + (B.2 - M.2)^2 →
    (A.1 - M.1)^2 + (A.2 - M.2)^2 = (A.1 - Q.1)^2 + (A.2 - Q.2)^2) :=
sorry

-- Theorem 2: Line EF exists and has the given equations
theorem line_EF_exists : ∃ E F : ℝ × ℝ,
  C E.1 E.2 ∧ 
  C F.1 F.2 ∧
  (R.1 - E.1)^2 + (R.2 - E.2)^2 = (E.1 - F.1)^2 + (E.2 - F.2)^2 ∧
  (2 * E.1 + E.2 - 6 = 0 ∨ 14 * E.1 + E.2 - 18 = 0) ∧
  (2 * F.1 + F.2 - 6 = 0 ∨ 14 * F.1 + F.2 - 18 = 0) :=
sorry

end NUMINAMATH_CALUDE_no_point_M_exists_line_EF_exists_l4122_412272


namespace NUMINAMATH_CALUDE_tan_alpha_values_l4122_412236

theorem tan_alpha_values (α : ℝ) (h : Real.sin (2 * α) = -Real.sin α) : 
  Real.tan α = 0 ∨ Real.tan α = Real.sqrt 3 ∨ Real.tan α = -Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_tan_alpha_values_l4122_412236


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l4122_412222

/-- Given a geometric sequence {a_n} where a_4 = 4, prove that a_3 * a_5 = 16 -/
theorem geometric_sequence_property (a : ℕ → ℝ) (h : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h4 : a 4 = 4) : a 3 * a 5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l4122_412222


namespace NUMINAMATH_CALUDE_building_heights_sum_l4122_412250

/-- The sum of heights of four buildings with specific height relationships -/
theorem building_heights_sum : 
  let tallest : ℝ := 100
  let second : ℝ := tallest / 2
  let third : ℝ := second / 2
  let fourth : ℝ := third / 5
  tallest + second + third + fourth = 180 := by sorry

end NUMINAMATH_CALUDE_building_heights_sum_l4122_412250


namespace NUMINAMATH_CALUDE_problem_statement_l4122_412291

theorem problem_statement (a b c m n : ℝ) 
  (h1 : a - b = m) 
  (h2 : b - c = n) : 
  a^2 + b^2 + c^2 - a*b - b*c - c*a = m^2 + n^2 + m*n := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l4122_412291


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l4122_412285

theorem z_in_fourth_quadrant : 
  ∀ z : ℂ, (1 - I) / (z - 2) = 1 + I → 
  (z.re > 0 ∧ z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l4122_412285


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l4122_412287

theorem more_girls_than_boys (total_students : ℕ) (boy_ratio girl_ratio : ℕ) 
  (h1 : total_students = 42)
  (h2 : boy_ratio = 3)
  (h3 : girl_ratio = 4) :
  ∃ (boys girls : ℕ), 
    boys + girls = total_students ∧ 
    boy_ratio * girls = girl_ratio * boys ∧
    girls - boys = 6 := by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l4122_412287


namespace NUMINAMATH_CALUDE_soda_discount_percentage_l4122_412288

-- Define the given constants
def full_price : ℚ := 30
def group_size : ℕ := 10
def num_children : ℕ := 4
def soda_price : ℚ := 5
def total_paid : ℚ := 197

-- Define the calculation functions
def adult_price := full_price
def child_price := full_price / 2

def total_price_without_discount : ℚ :=
  (group_size - num_children) * adult_price + num_children * child_price

def price_paid_for_tickets : ℚ := total_paid - soda_price

def discount_amount : ℚ := total_price_without_discount - price_paid_for_tickets

def discount_percentage : ℚ := (discount_amount / total_price_without_discount) * 100

-- State the theorem
theorem soda_discount_percentage : discount_percentage = 20 := by sorry

end NUMINAMATH_CALUDE_soda_discount_percentage_l4122_412288


namespace NUMINAMATH_CALUDE_division_multiplication_result_l4122_412227

theorem division_multiplication_result : (2 : ℚ) / 3 * (-1/3) = -2/9 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_result_l4122_412227


namespace NUMINAMATH_CALUDE_g_of_2_l4122_412265

/-- Given functions f and g, prove the value of g(2) -/
theorem g_of_2 (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = 2 * x^2 + 4 * x - 6)
  (hg : ∀ x, g (f x) = 3 * x^3 + 2 * x - 5) :
  g 2 = 3 * (-1 + Real.sqrt 5)^3 + 2 * (-1 + Real.sqrt 5) - 5 := by
sorry

end NUMINAMATH_CALUDE_g_of_2_l4122_412265


namespace NUMINAMATH_CALUDE_cafe_chairs_count_l4122_412206

/-- Calculates the total number of chairs in a cafe given the number of indoor and outdoor tables
    and the number of chairs per table type. -/
def total_chairs (indoor_tables : ℕ) (outdoor_tables : ℕ) (chairs_per_indoor_table : ℕ) (chairs_per_outdoor_table : ℕ) : ℕ :=
  indoor_tables * chairs_per_indoor_table + outdoor_tables * chairs_per_outdoor_table

/-- Theorem stating that the total number of chairs in the cafe is 123. -/
theorem cafe_chairs_count :
  total_chairs 9 11 10 3 = 123 := by
  sorry

#eval total_chairs 9 11 10 3

end NUMINAMATH_CALUDE_cafe_chairs_count_l4122_412206


namespace NUMINAMATH_CALUDE_rectangle_area_difference_l4122_412268

/-- Given a real number x, this theorem states that the area of a rectangle with 
    dimensions (x+8) and (x+6), minus the area of a rectangle with dimensions (2x-1) 
    and (x-1), plus the area of a rectangle with dimensions (x-3) and (x-5), 
    equals 25x + 62. -/
theorem rectangle_area_difference (x : ℝ) : 
  (x + 8) * (x + 6) - (2*x - 1) * (x - 1) + (x - 3) * (x - 5) = 25*x + 62 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_difference_l4122_412268


namespace NUMINAMATH_CALUDE_hyperbola_line_slope_l4122_412246

/-- Given two points on a hyperbola with a specific midpoint, prove that the slope of the line connecting them is 9/4 -/
theorem hyperbola_line_slope (A B : ℝ × ℝ) : 
  (A.1^2 - A.2^2/9 = 1) →  -- A is on the hyperbola
  (B.1^2 - B.2^2/9 = 1) →  -- B is on the hyperbola
  ((A.1 + B.1)/2 = -1) →   -- x-coordinate of midpoint
  ((A.2 + B.2)/2 = -4) →   -- y-coordinate of midpoint
  (B.2 - A.2)/(B.1 - A.1) = 9/4 :=  -- slope of line AB
by sorry

end NUMINAMATH_CALUDE_hyperbola_line_slope_l4122_412246


namespace NUMINAMATH_CALUDE_lou_fine_shoes_pricing_l4122_412289

/-- Calculates the price of shoes after Lou's Fine Shoes pricing strategy --/
theorem lou_fine_shoes_pricing (initial_price : ℝ) : 
  initial_price = 50 →
  (initial_price * (1 + 0.2)) * (1 - 0.2) = 48 := by
sorry

end NUMINAMATH_CALUDE_lou_fine_shoes_pricing_l4122_412289


namespace NUMINAMATH_CALUDE_tan_330_degrees_l4122_412237

theorem tan_330_degrees : Real.tan (330 * π / 180) = -Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_330_degrees_l4122_412237


namespace NUMINAMATH_CALUDE_count_perfect_square_factors_l4122_412299

/-- The number of factors of 12000 that are perfect squares -/
def num_perfect_square_factors : ℕ :=
  sorry

/-- 12000 expressed as its prime factorization -/
def twelve_thousand_factorization : ℕ :=
  2^5 * 3 * 5^3

theorem count_perfect_square_factors :
  num_perfect_square_factors = 6 ∧ twelve_thousand_factorization = 12000 :=
sorry

end NUMINAMATH_CALUDE_count_perfect_square_factors_l4122_412299


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l4122_412247

/-- A circle in the xy-plane -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- A line in the xy-plane -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- The property of a circle being symmetric with respect to a line -/
def isSymmetric (c : Circle) (l : Line) : Prop := sorry

theorem circle_symmetry_line (m : ℝ) :
  let c : Circle := { equation := fun x y => x^2 + y^2 + 2*x - 4*y = 0 }
  let l : Line := { equation := fun x y => 3*x + y + m = 0 }
  isSymmetric c l → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetry_line_l4122_412247


namespace NUMINAMATH_CALUDE_total_cookies_is_16000_l4122_412229

/-- The number of church members volunteering to bake cookies. -/
def num_members : ℕ := 100

/-- The number of sheets of cookies each member bakes. -/
def sheets_per_member : ℕ := 10

/-- The number of cookies on each sheet. -/
def cookies_per_sheet : ℕ := 16

/-- The total number of cookies baked by all church members. -/
def total_cookies : ℕ := num_members * sheets_per_member * cookies_per_sheet

/-- Theorem stating that the total number of cookies baked is 16,000. -/
theorem total_cookies_is_16000 : total_cookies = 16000 := by
  sorry

end NUMINAMATH_CALUDE_total_cookies_is_16000_l4122_412229


namespace NUMINAMATH_CALUDE_empty_solution_set_iff_a_range_l4122_412231

theorem empty_solution_set_iff_a_range (a : ℝ) :
  (∀ x : ℝ, |x - 1| + |x - 3| > a^2 - 2*a - 1) ↔ (-1 < a ∧ a < 3) :=
by sorry

end NUMINAMATH_CALUDE_empty_solution_set_iff_a_range_l4122_412231


namespace NUMINAMATH_CALUDE_S_intersections_empty_l4122_412261

def S (n : ℕ) : Set ℕ :=
  {x | ∃ g : ℕ, g ≥ 2 ∧ x = (g^n - 1) / (g - 1)}

theorem S_intersections_empty :
  (S 3 ∩ S 4 = ∅) ∧ (S 3 ∩ S 5 = ∅) := by
  sorry

end NUMINAMATH_CALUDE_S_intersections_empty_l4122_412261


namespace NUMINAMATH_CALUDE_not_lucky_1982_1983_l4122_412208

/-- Checks if a given year is a lucky year -/
def isLuckyYear (year : Nat) : Prop :=
  ∃ (month day : Nat),
    1 ≤ month ∧ month ≤ 12 ∧
    1 ≤ day ∧ day ≤ 31 ∧
    month * day = year % 100

theorem not_lucky_1982_1983 :
  ¬(isLuckyYear 1982) ∧ ¬(isLuckyYear 1983) :=
by sorry

end NUMINAMATH_CALUDE_not_lucky_1982_1983_l4122_412208


namespace NUMINAMATH_CALUDE_equilateral_triangle_locus_l4122_412244

-- Define an equilateral triangle ABC
def EquilateralTriangle (A B C : ℝ × ℝ) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B

-- Define the reflection of a point over a line
def ReflectPointOverLine (P Q R : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the set of points P satisfying PA^2 = PB^2 + PC^2
def SatisfyingPoints (A B C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P : ℝ × ℝ | dist P A ^ 2 = dist P B ^ 2 + dist P C ^ 2}

-- Theorem statement
theorem equilateral_triangle_locus 
  (A B C : ℝ × ℝ) 
  (h : EquilateralTriangle A B C) :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    center = ReflectPointOverLine A B C ∧ 
    radius = dist A B ∧
    SatisfyingPoints A B C = {P : ℝ × ℝ | dist P center = radius} :=
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_locus_l4122_412244


namespace NUMINAMATH_CALUDE_closest_to_zero_l4122_412270

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1) * d

theorem closest_to_zero (a₁ d : ℤ) (h₁ : a₁ = 81) (h₂ : d = -7) :
  ∀ n : ℕ, n ≠ 13 → |arithmetic_sequence a₁ d 13| ≤ |arithmetic_sequence a₁ d n| :=
by sorry

end NUMINAMATH_CALUDE_closest_to_zero_l4122_412270


namespace NUMINAMATH_CALUDE_base_conversion_sum_l4122_412275

def base_11_to_10 (n : ℕ) : ℕ := 3224

def base_5_to_10 (n : ℕ) : ℕ := 36

def base_7_to_10 (n : ℕ) : ℕ := 1362

def base_8_to_10 (n : ℕ) : ℕ := 3008

theorem base_conversion_sum :
  (base_11_to_10 2471 / base_5_to_10 121) - base_7_to_10 3654 + base_8_to_10 5680 = 1736 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_sum_l4122_412275


namespace NUMINAMATH_CALUDE_prob_four_red_cards_standard_deck_l4122_412210

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)

/-- Probability of drawing n red cards in a row from a deck -/
def prob_n_red_cards (d : Deck) (n : ℕ) : ℚ :=
  sorry

theorem prob_four_red_cards_standard_deck :
  let standard_deck : Deck := ⟨52, 26, 26⟩
  prob_n_red_cards standard_deck 4 = 276 / 9801 := by
  sorry

end NUMINAMATH_CALUDE_prob_four_red_cards_standard_deck_l4122_412210


namespace NUMINAMATH_CALUDE_smallest_whole_number_with_odd_factors_l4122_412205

theorem smallest_whole_number_with_odd_factors : ∃ n : ℕ, 
  n > 100 ∧ 
  (∀ m : ℕ, m > 100 → (∃ k : ℕ, k * k = m) → m ≥ n) ∧
  (∃ k : ℕ, k * k = n) :=
by sorry

end NUMINAMATH_CALUDE_smallest_whole_number_with_odd_factors_l4122_412205


namespace NUMINAMATH_CALUDE_amount_ratio_l4122_412296

/-- Prove that the ratio of A's amount to B's amount is 1:3 given the conditions -/
theorem amount_ratio (total amount_B amount_C : ℚ) (h1 : total = 1440)
  (h2 : amount_B = 270) (h3 : amount_B = (1/4) * amount_C) :
  ∃ amount_A : ℚ, amount_A + amount_B + amount_C = total ∧ amount_A = (1/3) * amount_B := by
  sorry

end NUMINAMATH_CALUDE_amount_ratio_l4122_412296


namespace NUMINAMATH_CALUDE_lino_shell_collection_l4122_412293

/-- Theorem: Lino's shell collection
  Given:
  - Lino put 292 shells back in the afternoon
  - She has 32 shells in total at the end
  Prove that Lino picked up 324 shells in the morning
-/
theorem lino_shell_collection (shells_put_back shells_remaining : ℕ) 
  (h1 : shells_put_back = 292)
  (h2 : shells_remaining = 32) :
  shells_put_back + shells_remaining = 324 := by
  sorry

end NUMINAMATH_CALUDE_lino_shell_collection_l4122_412293


namespace NUMINAMATH_CALUDE_unique_solution_tan_cos_equation_l4122_412281

theorem unique_solution_tan_cos_equation : 
  ∃! (n : ℕ), n > 0 ∧ Real.tan (π / (2 * n)) + Real.cos (π / (2 * n)) = n / 4 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_tan_cos_equation_l4122_412281


namespace NUMINAMATH_CALUDE_xyz_ratio_l4122_412225

theorem xyz_ratio (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (diff_xy : x ≠ y) (diff_xz : x ≠ z) (diff_yz : y ≠ z)
  (eq1 : y / (x - z) = (x + y) / z)
  (eq2 : (x + y) / z = x / y) : 
  x / y = 2 := by sorry

end NUMINAMATH_CALUDE_xyz_ratio_l4122_412225


namespace NUMINAMATH_CALUDE_negative_five_greater_than_negative_sqrt_26_l4122_412204

theorem negative_five_greater_than_negative_sqrt_26 :
  -5 > -Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_negative_five_greater_than_negative_sqrt_26_l4122_412204


namespace NUMINAMATH_CALUDE_average_weight_of_all_boys_l4122_412254

theorem average_weight_of_all_boys (group1_count : ℕ) (group1_avg : ℝ) 
  (group2_count : ℕ) (group2_avg : ℝ) : 
  group1_count = 16 → 
  group1_avg = 50.25 → 
  group2_count = 8 → 
  group2_avg = 45.15 → 
  let total_weight := group1_count * group1_avg + group2_count * group2_avg
  let total_count := group1_count + group2_count
  (total_weight / total_count) = 48.55 := by
sorry

end NUMINAMATH_CALUDE_average_weight_of_all_boys_l4122_412254


namespace NUMINAMATH_CALUDE_area_of_three_arc_region_sum_of_coefficients_l4122_412245

/-- The area of a region bounded by three circular arcs -/
theorem area_of_three_arc_region :
  let r : ℝ := 5  -- radius of each circle
  let θ : ℝ := π / 2  -- central angle of each arc (90 degrees in radians)
  let sector_area : ℝ := (θ / (2 * π)) * π * r^2  -- area of one sector
  let triangle_side : ℝ := r * Real.sqrt 2  -- side length of the equilateral triangle
  let triangle_area : ℝ := (Real.sqrt 3 / 4) * triangle_side^2  -- area of the equilateral triangle
  let region_area : ℝ := 3 * sector_area - triangle_area  -- area of the bounded region
  region_area = -125 * Real.sqrt 3 / 4 + 75 * π / 4 := by
    sorry

/-- The sum of coefficients in the area expression -/
theorem sum_of_coefficients :
  let a : ℝ := -125 / 4
  let b : ℝ := 3
  let c : ℝ := 75 / 4
  ⌊a + b + c⌋ = -9 := by
    sorry

end NUMINAMATH_CALUDE_area_of_three_arc_region_sum_of_coefficients_l4122_412245


namespace NUMINAMATH_CALUDE_solution_set_implies_b_power_a_l4122_412234

theorem solution_set_implies_b_power_a (a b : ℝ) : 
  (∀ x : ℝ, (1 < x ∧ x < 3) ↔ x^2 < a*x + b) → 
  b^a = 81 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_b_power_a_l4122_412234


namespace NUMINAMATH_CALUDE_quadratic_minimum_value_l4122_412284

/-- A quadratic function f(x) = x^2 - 2x + m with a minimum value of 1 on [3, +∞) -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m

/-- The domain of the function -/
def domain : Set ℝ := {x : ℝ | x ≥ 3}

theorem quadratic_minimum_value (m : ℝ) :
  (∀ x ∈ domain, f m x ≥ 1) ∧ (∃ x ∈ domain, f m x = 1) → m = -2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_minimum_value_l4122_412284


namespace NUMINAMATH_CALUDE_trigonometric_simplification_trigonometric_evaluation_l4122_412255

-- Part 1
theorem trigonometric_simplification (α : ℝ) : 
  (Real.cos (α - π/2)) / (Real.sin (5*π/2 + α)) * Real.sin (α - 2*π) * Real.cos (2*π - α) = Real.sin α ^ 2 := by
  sorry

-- Part 2
theorem trigonometric_evaluation : 
  Real.sin (25*π/6) + Real.cos (25*π/3) + Real.tan (-25*π/4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_trigonometric_evaluation_l4122_412255


namespace NUMINAMATH_CALUDE_locus_of_N_l4122_412240

/-- The locus of point N in an equilateral triangle with a moving point on the unit circle -/
theorem locus_of_N (M N : ℂ) (t : ℝ) : 
  (∀ t, M = Complex.exp (Complex.I * t)) →  -- M is on the unit circle
  (N - 3 = Complex.exp (Complex.I * (5 * Real.pi / 3)) * (M - 3)) →  -- N forms equilateral triangle with A(3,0) and M
  (Complex.abs (N - (3/2 + Complex.I * (3 * Real.sqrt 3 / 2))) = 1) :=  -- Locus of N is a circle
by sorry

end NUMINAMATH_CALUDE_locus_of_N_l4122_412240


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l4122_412211

theorem smallest_m_for_integral_solutions : 
  ∃ (m : ℕ), m > 0 ∧ 
  (∃ (x : ℤ), 12 * x^2 - m * x + 360 = 0) ∧
  (∀ (k : ℕ), 0 < k ∧ k < m → ¬∃ (x : ℤ), 12 * x^2 - k * x + 360 = 0) ∧
  m = 132 := by
sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_solutions_l4122_412211


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l4122_412241

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a_n where a_3 + a_11 = 22, prove that a_7 = 11 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) (h : is_arithmetic_sequence a) 
  (h_sum : a 3 + a 11 = 22) : a 7 = 11 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l4122_412241


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l4122_412233

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - 2 * a * x + 1 > 0) ↔ (0 ≤ a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l4122_412233


namespace NUMINAMATH_CALUDE_third_year_sample_size_l4122_412283

/-- Calculates the number of students to be sampled from the third year in a stratified sampling -/
theorem third_year_sample_size 
  (total_students : ℕ) 
  (first_year : ℕ) 
  (second_year : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 900) 
  (h2 : first_year = 240) 
  (h3 : second_year = 260) 
  (h4 : sample_size = 45) :
  (sample_size * (total_students - first_year - second_year)) / total_students = 20 := by
  sorry

#check third_year_sample_size

end NUMINAMATH_CALUDE_third_year_sample_size_l4122_412283


namespace NUMINAMATH_CALUDE_parallel_lines_bisect_circle_perimeter_l4122_412271

-- Define the lines and circle
def line_l (a : ℝ) (x y : ℝ) : Prop := a * x - 2 * y + 2 = 0
def line_m (a : ℝ) (x y : ℝ) : Prop := x + (a - 3) * y + 1 = 0
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 2

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, line_l a x y ↔ line_m a x y) ↔ a = 1 :=
sorry

-- Theorem for bisecting circle's perimeter
theorem bisect_circle_perimeter (a : ℝ) :
  (∃ x y : ℝ, line_l a x y ∧ x = 1 ∧ y = 0) ↔ a = -2 :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_bisect_circle_perimeter_l4122_412271


namespace NUMINAMATH_CALUDE_pigeon_difference_l4122_412248

theorem pigeon_difference (total_pigeons : ℕ) (black_ratio : ℚ) (male_ratio : ℚ) : 
  total_pigeons = 70 →
  black_ratio = 1/2 →
  male_ratio = 1/5 →
  (black_ratio * total_pigeons : ℚ) * (1 - male_ratio) - (black_ratio * total_pigeons : ℚ) * male_ratio = 21 := by
  sorry

end NUMINAMATH_CALUDE_pigeon_difference_l4122_412248


namespace NUMINAMATH_CALUDE_swim_meet_cars_l4122_412273

theorem swim_meet_cars (num_vans : ℕ) (people_per_car : ℕ) (people_per_van : ℕ) 
  (max_per_car : ℕ) (max_per_van : ℕ) (extra_capacity : ℕ) :
  num_vans = 3 →
  people_per_car = 5 →
  people_per_van = 3 →
  max_per_car = 6 →
  max_per_van = 8 →
  extra_capacity = 17 →
  ∃ (num_cars : ℕ), 
    num_cars * people_per_car + num_vans * people_per_van + extra_capacity = 
    num_cars * max_per_car + num_vans * max_per_van ∧
    num_cars = 2 :=
by sorry

end NUMINAMATH_CALUDE_swim_meet_cars_l4122_412273


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l4122_412279

-- Define the conditions
def p (a : ℝ) : Prop := (a - 1)^2 ≤ 1

def q (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 - a*x + 1 ≥ 0

-- Theorem stating that p is a sufficient but not necessary condition for q
theorem p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬(p a)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l4122_412279


namespace NUMINAMATH_CALUDE_find_a_l4122_412226

def A (a : ℝ) : Set ℝ := {4, a^2}
def B (a : ℝ) : Set ℝ := {a-6, a+1, 9}

theorem find_a : ∀ a : ℝ, A a ∩ B a = {9} → a = -3 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l4122_412226


namespace NUMINAMATH_CALUDE_triangle_area_l4122_412213

/-- Given a triangle with sides in ratio 5:12:13 and perimeter 300, prove its area is 3000 -/
theorem triangle_area (a b c : ℝ) (h_ratio : (a, b, c) = (5 * (300 / 30), 12 * (300 / 30), 13 * (300 / 30))) 
  (h_perimeter : a + b + c = 300) : 
  let s := (a + b + c) / 2
  Real.sqrt (s * (s - a) * (s - b) * (s - c)) = 3000 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l4122_412213


namespace NUMINAMATH_CALUDE_wayne_blocks_total_l4122_412230

theorem wayne_blocks_total (initial_blocks additional_blocks : ℕ) 
  (h1 : initial_blocks = 9)
  (h2 : additional_blocks = 6) :
  initial_blocks + additional_blocks = 15 := by
  sorry

end NUMINAMATH_CALUDE_wayne_blocks_total_l4122_412230


namespace NUMINAMATH_CALUDE_system_solution_l4122_412264

theorem system_solution (x y k : ℚ) 
  (eq1 : 3 * x + 2 * y = k + 1)
  (eq2 : 2 * x + 3 * y = k)
  (sum_condition : x + y = 2) :
  k = 9 / 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l4122_412264


namespace NUMINAMATH_CALUDE_textbook_weight_difference_l4122_412274

theorem textbook_weight_difference :
  let chemistry_weight : ℝ := 7.125
  let geometry_weight : ℝ := 0.625
  chemistry_weight - geometry_weight = 6.5 := by
  sorry

end NUMINAMATH_CALUDE_textbook_weight_difference_l4122_412274


namespace NUMINAMATH_CALUDE_square_less_than_triple_l4122_412278

theorem square_less_than_triple (x : ℤ) : x^2 < 3*x ↔ x = 1 ∨ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_less_than_triple_l4122_412278


namespace NUMINAMATH_CALUDE_basketball_probability_l4122_412219

/-- The number of basketballs -/
def total_balls : ℕ := 8

/-- The number of new basketballs -/
def new_balls : ℕ := 4

/-- The number of old basketballs -/
def old_balls : ℕ := 4

/-- The number of balls selected in each training session -/
def selected_balls : ℕ := 2

/-- The probability of selecting exactly one new ball in the second training session -/
def prob_one_new_second : ℚ := 51 / 98

theorem basketball_probability :
  total_balls = new_balls + old_balls →
  prob_one_new_second = (
    (Nat.choose old_balls selected_balls * Nat.choose new_balls 1 * Nat.choose old_balls 1 +
     Nat.choose new_balls 1 * Nat.choose old_balls 1 * Nat.choose (new_balls - 1) 1 * Nat.choose (old_balls + 1) 1 +
     Nat.choose new_balls selected_balls * Nat.choose (new_balls - selected_balls) 1 * Nat.choose (old_balls + selected_balls) 1) /
    (Nat.choose total_balls selected_balls * Nat.choose total_balls selected_balls)
  ) := by sorry

end NUMINAMATH_CALUDE_basketball_probability_l4122_412219


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4122_412228

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (3 + 2 * Real.sqrt x) = 4 → x = 169 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4122_412228


namespace NUMINAMATH_CALUDE_min_rows_required_l4122_412216

/-- The number of seats in each row -/
def seats_per_row : ℕ := 168

/-- The total number of students -/
def total_students : ℕ := 2016

/-- The maximum number of students from each school -/
def max_students_per_school : ℕ := 40

/-- Represents the seating arrangement in the arena -/
structure Arena where
  rows : ℕ
  students_seated : ℕ
  school_integrity : Bool  -- True if students from each school are in a single row

/-- A function to check if a seating arrangement is valid -/
def is_valid_arrangement (a : Arena) : Prop :=
  a.students_seated = total_students ∧
  a.school_integrity ∧
  a.rows * seats_per_row ≥ total_students

/-- The main theorem stating the minimum number of rows required -/
theorem min_rows_required : 
  ∀ a : Arena, is_valid_arrangement a → a.rows ≥ 15 :=
sorry

end NUMINAMATH_CALUDE_min_rows_required_l4122_412216


namespace NUMINAMATH_CALUDE_fraction_simplification_l4122_412214

theorem fraction_simplification :
  (1 : ℚ) / 462 + 17 / 42 = 94 / 231 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l4122_412214


namespace NUMINAMATH_CALUDE_circle_and_tangent_lines_l4122_412243

-- Define the circle C
def circle_C (x y : ℝ) := x^2 + (y + 1)^2 = 2

-- Define the parabola
def parabola (x y : ℝ) := y^2 = 4 * x

-- Define the line y = x
def line_y_eq_x (x y : ℝ) := y = x

-- Define the point P
def point_P : ℝ × ℝ := (-1, 2)

-- Define the two tangent lines
def tangent_line_1 (x y : ℝ) := x + y - 1 = 0
def tangent_line_2 (x y : ℝ) := 7 * x - y + 9 = 0

theorem circle_and_tangent_lines :
  -- Circle C is symmetric about the y-axis
  (∀ x y : ℝ, circle_C x y ↔ circle_C (-x) y) →
  -- Circle C passes through the focus of the parabola y^2 = 4x
  (circle_C 1 0) →
  -- Circle C is divided into two arc lengths with a ratio of 1:2 by the line y = x
  (∃ r : ℝ, r > 0 ∧ ∀ x y : ℝ, circle_C x y → line_y_eq_x x y → 
    (x^2 + y^2)^(1/2) = r ∧ ((x - 0)^2 + (y - (-1))^2)^(1/2) = 2 * r) →
  -- The center of circle C is below the x-axis
  (∃ a : ℝ, a < 0 ∧ ∀ x y : ℝ, circle_C x y ↔ x^2 + (y - a)^2 = 2) →
  -- The equation of circle C is x^2 + (y + 1)^2 = 2
  (∀ x y : ℝ, circle_C x y ↔ x^2 + (y + 1)^2 = 2) ∧
  -- The equations of the tangent lines passing through P(-1, 2) are x + y - 1 = 0 and 7x - y + 9 = 0
  (∀ x y : ℝ, (tangent_line_1 x y ∨ tangent_line_2 x y) ↔
    (∃ t : ℝ, circle_C (point_P.1 + t * (x - point_P.1)) (point_P.2 + t * (y - point_P.2)) ∧
      (∀ s : ℝ, s ≠ t → ¬ circle_C (point_P.1 + s * (x - point_P.1)) (point_P.2 + s * (y - point_P.2))))) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_tangent_lines_l4122_412243


namespace NUMINAMATH_CALUDE_charles_paint_area_l4122_412298

/-- 
Given a wall that requires 320 square feet to be painted and a work ratio of 2:6 between Allen and Charles,
prove that Charles paints 240 square feet.
-/
theorem charles_paint_area (total_area : ℝ) (allen_ratio charles_ratio : ℕ) : 
  total_area = 320 →
  allen_ratio = 2 →
  charles_ratio = 6 →
  (charles_ratio / (allen_ratio + charles_ratio)) * total_area = 240 := by
  sorry

end NUMINAMATH_CALUDE_charles_paint_area_l4122_412298


namespace NUMINAMATH_CALUDE_eesha_late_arrival_l4122_412221

/-- Eesha's commute problem -/
theorem eesha_late_arrival (usual_time : ℕ) (late_start : ℕ) (speed_reduction : ℚ) : 
  usual_time = 60 → late_start = 30 → speed_reduction = 1/4 →
  (usual_time : ℚ) / (1 - speed_reduction) + late_start - usual_time = 15 := by
  sorry

#check eesha_late_arrival

end NUMINAMATH_CALUDE_eesha_late_arrival_l4122_412221


namespace NUMINAMATH_CALUDE_leg_head_difference_l4122_412252

/-- Represents a group of ducks and cows -/
structure AnimalGroup where
  ducks : ℕ
  cows : ℕ

/-- Calculates the total number of legs in the group -/
def totalLegs (g : AnimalGroup) : ℕ := 2 * g.ducks + 4 * g.cows

/-- Calculates the total number of heads in the group -/
def totalHeads (g : AnimalGroup) : ℕ := g.ducks + g.cows

/-- The main theorem -/
theorem leg_head_difference (g : AnimalGroup) 
  (h1 : g.cows = 20)
  (h2 : ∃ k : ℕ, totalLegs g = 2 * totalHeads g + k) :
  ∃ k : ℕ, k = 40 ∧ totalLegs g = 2 * totalHeads g + k := by
  sorry


end NUMINAMATH_CALUDE_leg_head_difference_l4122_412252


namespace NUMINAMATH_CALUDE_chicken_pizza_menu_combinations_l4122_412297

theorem chicken_pizza_menu_combinations : 
  let chicken_types : ℕ := 4
  let pizza_types : ℕ := 3
  let same_chicken_diff_pizza := chicken_types * (pizza_types * (pizza_types - 1))
  let same_pizza_diff_chicken := pizza_types * (chicken_types * (chicken_types - 1))
  same_chicken_diff_pizza + same_pizza_diff_chicken = 60 :=
by sorry

end NUMINAMATH_CALUDE_chicken_pizza_menu_combinations_l4122_412297


namespace NUMINAMATH_CALUDE_problem_statement_l4122_412253

open Real

theorem problem_statement :
  ∃ a : ℝ,
    (∀ x : ℝ, x > 0 → exp x - log x ≥ exp a - log a) ∧
    exp a * log a = -1 ∧
    ∀ x₁ x₂ : ℝ,
      1 < x₁ → x₁ < x₂ →
        (∃ x₀ : ℝ, x₁ < x₀ ∧ x₀ < x₂ ∧
          ((exp x₁ - exp x₂) / (x₁ - x₂)) / ((log x₁ - log x₂) / (x₁ - x₂)) = x₀ * exp x₀) ∧
        (exp x₁ - exp x₂) / (x₁ - x₂) - (log x₁ - log x₂) / (x₁ - x₂) <
          (exp x₁ + exp x₂) / 2 - 1 / sqrt (x₁ * x₂) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4122_412253


namespace NUMINAMATH_CALUDE_equal_roots_condition_l4122_412280

/-- A quadratic equation of the form x(x+1) + ax = 0 has two equal real roots if and only if a = -1 -/
theorem equal_roots_condition (a : ℝ) : 
  (∃ x : ℝ, x * (x + 1) + a * x = 0 ∧ 
   ∀ y : ℝ, y * (y + 1) + a * y = 0 → y = x) ↔ 
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_equal_roots_condition_l4122_412280


namespace NUMINAMATH_CALUDE_team_selection_ways_l4122_412235

def num_boys : ℕ := 10
def num_girls : ℕ := 12
def team_size : ℕ := 8
def boys_in_team : ℕ := 4
def girls_in_team : ℕ := 4

theorem team_selection_ways :
  (Nat.choose num_boys boys_in_team) * (Nat.choose num_girls girls_in_team) = 103950 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_ways_l4122_412235


namespace NUMINAMATH_CALUDE_subtraction_property_l4122_412218

theorem subtraction_property : 12.56 - (5.56 - 2.63) = 12.56 - 5.56 + 2.63 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_property_l4122_412218


namespace NUMINAMATH_CALUDE_fraction_to_decimal_plus_two_l4122_412267

theorem fraction_to_decimal_plus_two : (7 : ℚ) / 16 + 2 = (2.4375 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_plus_two_l4122_412267


namespace NUMINAMATH_CALUDE_fraction_equality_l4122_412295

theorem fraction_equality (a b c : ℝ) :
  (|a^2 + b^2|^3 + |b^2 + c^2|^3 + |c^2 + a^2|^3) / (|a + b|^3 + |b + c|^3 + |c + a|^3) = 1 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l4122_412295


namespace NUMINAMATH_CALUDE_triangle_area_l4122_412202

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if A = 2π/3, a = 7, and b = 3, then the area of the triangle S_ABC = 15√3/4 -/
theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  A = 2 * Real.pi / 3 →
  a = 7 →
  b = 3 →
  (∃ (S_ABC : ℝ), S_ABC = (15 * Real.sqrt 3) / 4 ∧ S_ABC = (1/2) * b * c * Real.sin A) :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_l4122_412202


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l4122_412238

theorem solution_set_of_inequality (x : ℝ) :
  {x | x^2 - 2*x + 1 ≤ 0} = {1} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l4122_412238


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l4122_412294

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  x^4 + 2*x^3 = (x^2 + 6*x + 2) * q + (22*x^2 + 8*x) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l4122_412294


namespace NUMINAMATH_CALUDE_johns_journey_distance_l4122_412263

/-- Calculates the total distance traveled given two journey segments -/
def total_distance (speed1 : ℝ) (time1 : ℝ) (speed2 : ℝ) (time2 : ℝ) : ℝ :=
  speed1 * time1 + speed2 * time2

/-- Theorem: The total distance traveled in John's journey is 255 miles -/
theorem johns_journey_distance :
  total_distance 45 2 55 3 = 255 := by
  sorry

end NUMINAMATH_CALUDE_johns_journey_distance_l4122_412263


namespace NUMINAMATH_CALUDE_rainy_days_count_l4122_412286

theorem rainy_days_count (n : ℕ) : 
  (∃ (R NR : ℕ), 
    R + NR = 7 ∧ 
    n * R + 4 * NR = 26 ∧ 
    4 * NR - n * R = 14) → 
  (∃ (R : ℕ), R = 2 ∧ 
    (∃ (NR : ℕ), R + NR = 7 ∧ 
      n * R + 4 * NR = 26 ∧ 
      4 * NR - n * R = 14)) :=
by sorry

end NUMINAMATH_CALUDE_rainy_days_count_l4122_412286


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l4122_412220

-- Problem 1
theorem factorization_problem_1 (y : ℝ) : y^3 - y^2 + (1/4)*y = y*(y - 1/2)^2 := by sorry

-- Problem 2
theorem factorization_problem_2 (m n : ℝ) : m^4 - n^4 = (m - n)*(m + n)*(m^2 + n^2) := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l4122_412220


namespace NUMINAMATH_CALUDE_P_in_first_quadrant_l4122_412203

/-- A point in the 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The point P with coordinates (2,1) -/
def P : Point :=
  { x := 2, y := 1 }

/-- Theorem stating that P lies in the first quadrant -/
theorem P_in_first_quadrant : isInFirstQuadrant P := by
  sorry

end NUMINAMATH_CALUDE_P_in_first_quadrant_l4122_412203


namespace NUMINAMATH_CALUDE_visitors_scientific_notation_l4122_412276

-- Define 1.12 million
def visitors : ℝ := 1.12 * 1000000

-- Define scientific notation
def scientific_notation (x : ℝ) (base : ℝ) (exponent : ℤ) : Prop :=
  x = base * (10 : ℝ) ^ exponent ∧ 1 ≤ base ∧ base < 10

-- Theorem statement
theorem visitors_scientific_notation :
  scientific_notation visitors 1.12 6 := by
  sorry

end NUMINAMATH_CALUDE_visitors_scientific_notation_l4122_412276


namespace NUMINAMATH_CALUDE_crab_price_proof_l4122_412262

/-- Proves that the price per crab is $3 given the conditions of John's crab selling business -/
theorem crab_price_proof (baskets_per_week : ℕ) (crabs_per_basket : ℕ) (collection_frequency : ℕ) (total_revenue : ℕ) :
  baskets_per_week = 3 →
  crabs_per_basket = 4 →
  collection_frequency = 2 →
  total_revenue = 72 →
  (total_revenue : ℚ) / (baskets_per_week * crabs_per_basket * collection_frequency) = 3 := by
  sorry

#check crab_price_proof

end NUMINAMATH_CALUDE_crab_price_proof_l4122_412262


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l4122_412269

theorem simplify_and_rationalize :
  (Real.sqrt 6 / Real.sqrt 10) * (Real.sqrt 5 / Real.sqrt 15) * (Real.sqrt 8 / Real.sqrt 14) = 2 * Real.sqrt 35 / 35 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l4122_412269


namespace NUMINAMATH_CALUDE_rhombus_area_l4122_412249

/-- The area of a rhombus given its perimeter and one diagonal -/
theorem rhombus_area (perimeter : ℝ) (diagonal : ℝ) : 
  perimeter > 0 → diagonal > 0 → diagonal < perimeter → 
  (perimeter * diagonal) / 8 = 96 → 
  (perimeter / 4) * (((perimeter / 4)^2 - (diagonal / 2)^2).sqrt) = 96 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l4122_412249


namespace NUMINAMATH_CALUDE_waiter_problem_l4122_412212

/-- Calculates the number of men at tables given the number of tables, women, and average customers per table. -/
def number_of_men (tables : Float) (women : Float) (avg_customers : Float) : Float :=
  tables * avg_customers - women

/-- Theorem stating that given 9.0 tables, 7.0 women, and an average of 1.111111111 customers per table, the number of men at the tables is 3.0. -/
theorem waiter_problem :
  number_of_men 9.0 7.0 1.111111111 = 3.0 := by
  sorry

end NUMINAMATH_CALUDE_waiter_problem_l4122_412212


namespace NUMINAMATH_CALUDE_sum_and_double_l4122_412217

theorem sum_and_double : 2 * (1324 + 4231 + 3124 + 2413) = 22184 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_double_l4122_412217


namespace NUMINAMATH_CALUDE_number_of_coaches_l4122_412215

theorem number_of_coaches (pouches_per_pack : ℕ) (packs_bought : ℕ) (team_members : ℕ) (helpers : ℕ) :
  pouches_per_pack = 6 →
  packs_bought = 3 →
  team_members = 13 →
  helpers = 2 →
  packs_bought * pouches_per_pack = team_members + helpers + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_number_of_coaches_l4122_412215


namespace NUMINAMATH_CALUDE_sports_day_participation_l4122_412232

/-- Given that the number of participants in a school sports day this year (m) 
    is a 10% increase from last year, prove that the number of participants 
    last year was m / (1 + 10%). -/
theorem sports_day_participation (m : ℝ) : 
  let last_year := m / (1 + 10 / 100)
  let increase_rate := 10 / 100
  m = last_year * (1 + increase_rate) → 
  last_year = m / (1 + increase_rate) := by
sorry


end NUMINAMATH_CALUDE_sports_day_participation_l4122_412232


namespace NUMINAMATH_CALUDE_evaluate_expression_l4122_412259

theorem evaluate_expression (a : ℝ) : 
  let x : ℝ := a + 9
  (x - a + 5) = 14 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l4122_412259


namespace NUMINAMATH_CALUDE_sara_quarters_sum_l4122_412209

-- Define the initial number of quarters Sara had
def initial_quarters : ℝ := 783.0

-- Define the number of quarters Sara's dad gave her
def dad_quarters : ℝ := 271.0

-- Define the total number of quarters Sara has now
def total_quarters : ℝ := initial_quarters + dad_quarters

-- Theorem to prove
theorem sara_quarters_sum :
  total_quarters = 1054.0 := by sorry

end NUMINAMATH_CALUDE_sara_quarters_sum_l4122_412209


namespace NUMINAMATH_CALUDE_division_result_l4122_412292

theorem division_result (n : ℕ) (h : n = 2011) : 
  (4 * 10^n - 1) / (4 * ((10^n - 1) / 3) + 1) = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l4122_412292


namespace NUMINAMATH_CALUDE_unique_real_solution_l4122_412290

theorem unique_real_solution :
  ∃! x : ℝ, (x^12 + 1) * (x^10 + x^8 + x^6 + x^4 + x^2 + 1) = 12 * x^11 :=
by sorry

end NUMINAMATH_CALUDE_unique_real_solution_l4122_412290


namespace NUMINAMATH_CALUDE_no_larger_subdivision_exists_max_subdivision_exists_max_triangles_is_correct_l4122_412257

/-- Represents a triangular subdivision of a triangle T -/
structure TriangularSubdivision where
  numTriangles : ℕ
  numSegmentsPerVertex : ℕ
  verticesDontSplitSides : Bool

/-- The maximum number of triangles in a valid subdivision -/
def maxTriangles : ℕ := 19

/-- Checks if a triangular subdivision is valid according to the problem conditions -/
def isValidSubdivision (s : TriangularSubdivision) : Prop :=
  s.numSegmentsPerVertex > 1 ∧ s.verticesDontSplitSides

/-- States that no valid subdivision can have more than maxTriangles triangles -/
theorem no_larger_subdivision_exists (s : TriangularSubdivision) :
  isValidSubdivision s → s.numTriangles ≤ maxTriangles :=
sorry

/-- States that there exists a valid subdivision with exactly maxTriangles triangles -/
theorem max_subdivision_exists :
  ∃ s : TriangularSubdivision, isValidSubdivision s ∧ s.numTriangles = maxTriangles :=
sorry

/-- The main theorem stating that maxTriangles is indeed the maximum -/
theorem max_triangles_is_correct :
  (∀ s : TriangularSubdivision, isValidSubdivision s → s.numTriangles ≤ maxTriangles) ∧
  (∃ s : TriangularSubdivision, isValidSubdivision s ∧ s.numTriangles = maxTriangles) :=
sorry

end NUMINAMATH_CALUDE_no_larger_subdivision_exists_max_subdivision_exists_max_triangles_is_correct_l4122_412257


namespace NUMINAMATH_CALUDE_division_problem_l4122_412251

theorem division_problem (dividend : ℕ) (divisor : ℝ) (remainder : ℕ) (quotient : ℕ) : 
  dividend = 13698 →
  divisor = 153.75280898876406 →
  remainder = 14 →
  quotient = 89 →
  (dividend : ℝ) = divisor * quotient + remainder := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l4122_412251


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4122_412256

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^3 - 3*x > 0) ↔ (∃ x : ℝ, x^3 - 3*x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4122_412256


namespace NUMINAMATH_CALUDE_beverage_mix_ratio_l4122_412207

theorem beverage_mix_ratio : 
  ∀ (x y : ℝ), 
  x > 0 → y > 0 →
  (5 * x + 4 * y = 5.5 * x + 3.6 * y) →
  (x / y = 4 / 5) := by
sorry

end NUMINAMATH_CALUDE_beverage_mix_ratio_l4122_412207


namespace NUMINAMATH_CALUDE_factor_expression_l4122_412201

theorem factor_expression (x : ℝ) : 6 * x^3 - 54 * x = 6 * x * (x + 3) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l4122_412201


namespace NUMINAMATH_CALUDE_top_square_is_five_l4122_412239

/-- Represents a square on the grid --/
structure Square :=
  (number : Nat)
  (row : Nat)
  (col : Nat)

/-- Represents the grid of squares --/
def Grid := List Square

/-- Creates the initial 5x5 grid --/
def initialGrid : Grid :=
  sorry

/-- Performs the first diagonal fold --/
def foldDiagonal (g : Grid) : Grid :=
  sorry

/-- Performs the second fold (bottom half up) --/
def foldBottomUp (g : Grid) : Grid :=
  sorry

/-- Performs the third fold (left half behind) --/
def foldLeftBehind (g : Grid) : Grid :=
  sorry

/-- Returns the top square after all folds --/
def topSquareAfterFolds (g : Grid) : Square :=
  sorry

theorem top_square_is_five :
  let finalGrid := foldLeftBehind (foldBottomUp (foldDiagonal initialGrid))
  (topSquareAfterFolds finalGrid).number = 5 := by
  sorry

end NUMINAMATH_CALUDE_top_square_is_five_l4122_412239


namespace NUMINAMATH_CALUDE_range_of_a_l4122_412258

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {0, 1, a}
def B : Set ℝ := {x : ℝ | 0 < x ∧ x < 2}

-- State the theorem
theorem range_of_a (a : ℝ) : A a ∩ B = {1, a} → a ∈ Set.Ioo 0 1 ∪ Set.Ioo 1 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l4122_412258


namespace NUMINAMATH_CALUDE_total_stones_l4122_412266

/-- The number of stones in each pile -/
structure StonePiles where
  pile1 : ℕ
  pile2 : ℕ
  pile3 : ℕ
  pile4 : ℕ
  pile5 : ℕ

/-- The conditions for the stone distribution -/
def validDistribution (p : StonePiles) : Prop :=
  p.pile5 = 6 * p.pile3 ∧
  p.pile2 = 2 * (p.pile3 + p.pile5) ∧
  p.pile1 = p.pile5 / 3 ∧
  p.pile1 = p.pile4 - 10 ∧
  p.pile4 = p.pile2 / 2

/-- The theorem stating that the total number of stones is 60 -/
theorem total_stones (p : StonePiles) (h : validDistribution p) : 
  p.pile1 + p.pile2 + p.pile3 + p.pile4 + p.pile5 = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_stones_l4122_412266


namespace NUMINAMATH_CALUDE_function_value_at_zero_l4122_412242

theorem function_value_at_zero 
  (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f (x + 2) = f (x + 1) - f x) 
  (h2 : f 1 = Real.log (3/2)) 
  (h3 : f 2 = Real.log 15) : 
  f 0 = -1 := by sorry

end NUMINAMATH_CALUDE_function_value_at_zero_l4122_412242
