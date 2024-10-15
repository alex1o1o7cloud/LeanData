import Mathlib

namespace NUMINAMATH_CALUDE_complex_fraction_equality_l3752_375260

theorem complex_fraction_equality : (1 - 2*I) / (2 + I) = -I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l3752_375260


namespace NUMINAMATH_CALUDE_remainder_of_12345678_div_10_l3752_375283

theorem remainder_of_12345678_div_10 :
  ∃ q : ℕ, 12345678 = 10 * q + 8 ∧ 8 < 10 := by sorry

end NUMINAMATH_CALUDE_remainder_of_12345678_div_10_l3752_375283


namespace NUMINAMATH_CALUDE_units_digit_of_n_l3752_375234

/-- Given two natural numbers m and n, returns true if m has a units digit of 3 -/
def has_units_digit_3 (m : ℕ) : Prop :=
  m % 10 = 3

/-- Given a natural number n, returns its units digit -/
def units_digit (n : ℕ) : ℕ :=
  n % 10

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 14^8) (h2 : has_units_digit_3 m) :
  units_digit n = 2 := by
sorry

end NUMINAMATH_CALUDE_units_digit_of_n_l3752_375234


namespace NUMINAMATH_CALUDE_b_spending_percentage_l3752_375212

/-- Proves that B spends 85% of her salary given the conditions of the problem -/
theorem b_spending_percentage (total_salary : ℕ) (a_spending_rate : ℚ) (b_salary : ℕ) :
  total_salary = 14000 →
  a_spending_rate = 4/5 →
  b_salary = 8000 →
  let a_salary := total_salary - b_salary
  let a_savings := a_salary * (1 - a_spending_rate)
  let b_savings := a_savings
  let b_spending_rate := 1 - (b_savings / b_salary)
  b_spending_rate = 17/20 := by
sorry

#eval (17 : ℚ) / 20  -- Should output 0.85

end NUMINAMATH_CALUDE_b_spending_percentage_l3752_375212


namespace NUMINAMATH_CALUDE_gcd_triple_characterization_l3752_375292

theorem gcd_triple_characterization (a b c : ℕ+) :
  Nat.gcd a.val 20 = b.val ∧
  Nat.gcd b.val 15 = c.val ∧
  Nat.gcd a.val c.val = 5 →
  ∃ k : ℕ+, (a = 5 * k ∧ b = 5 ∧ c = 5) ∨
            (a = 5 * k ∧ b = 10 ∧ c = 5) ∨
            (a = 5 * k ∧ b = 20 ∧ c = 5) := by
  sorry


end NUMINAMATH_CALUDE_gcd_triple_characterization_l3752_375292


namespace NUMINAMATH_CALUDE_bhupathi_abhinav_fraction_l3752_375222

theorem bhupathi_abhinav_fraction : 
  ∀ (abhinav bhupathi : ℚ),
  abhinav + bhupathi = 1210 →
  bhupathi = 484 →
  ∃ (x : ℚ), (4 / 15) * abhinav = x * bhupathi ∧ x = 2 / 5 :=
by
  sorry

end NUMINAMATH_CALUDE_bhupathi_abhinav_fraction_l3752_375222


namespace NUMINAMATH_CALUDE_workshop_salary_problem_l3752_375231

theorem workshop_salary_problem (total_workers : Nat) (avg_salary : ℝ) 
  (num_technicians : Nat) (avg_salary_technicians : ℝ) :
  total_workers = 28 →
  avg_salary = 8000 →
  num_technicians = 7 →
  avg_salary_technicians = 14000 →
  let remaining_workers := total_workers - num_technicians
  let total_salary := total_workers * avg_salary
  let total_salary_technicians := num_technicians * avg_salary_technicians
  let total_salary_remaining := total_salary - total_salary_technicians
  let avg_salary_remaining := total_salary_remaining / remaining_workers
  avg_salary_remaining = 6000 := by
sorry

end NUMINAMATH_CALUDE_workshop_salary_problem_l3752_375231


namespace NUMINAMATH_CALUDE_inequality_proof_l3752_375262

theorem inequality_proof (a₁ a₂ b₁ b₂ : ℝ) 
  (h1 : 0 < a₁) (h2 : 0 < a₂) (h3 : a₁ > a₂) (h4 : b₁ ≥ a₁) (h5 : b₁ * b₂ ≥ a₁ * a₂) : 
  b₁ + b₂ ≥ a₁ + a₂ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3752_375262


namespace NUMINAMATH_CALUDE_complex_power_sum_l3752_375206

theorem complex_power_sum (z : ℂ) (h : z^2 - z + 1 = 0) : 
  z^98 + z^99 + z^100 + z^101 + z^102 = -z := by
  sorry

end NUMINAMATH_CALUDE_complex_power_sum_l3752_375206


namespace NUMINAMATH_CALUDE_fgh_supermarkets_count_l3752_375213

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 49

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - 14

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := 84

theorem fgh_supermarkets_count :
  us_supermarkets = 49 ∧
  us_supermarkets + canada_supermarkets = total_supermarkets ∧
  us_supermarkets = canada_supermarkets + 14 :=
by sorry

end NUMINAMATH_CALUDE_fgh_supermarkets_count_l3752_375213


namespace NUMINAMATH_CALUDE_largest_five_digit_code_l3752_375228

def is_power_of_5 (n : Nat) : Prop :=
  ∃ k : Nat, n = 5^k

def is_power_of_2 (n : Nat) : Prop :=
  ∃ k : Nat, n = 2^k

def is_multiple_of_3 (n : Nat) : Prop :=
  ∃ k : Nat, n = 3 * k

def digits_sum (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.sum

def has_unique_digits (n : Nat) : Prop :=
  let digits := n.digits 10
  digits.length = digits.toFinset.card

theorem largest_five_digit_code : 
  ∀ n : Nat,
  n ≤ 99999 ∧
  n ≥ 10000 ∧
  (∀ d : Nat, d ∈ n.digits 10 → d ≠ 0) ∧
  is_power_of_5 (n / 1000) ∧
  is_power_of_2 (n % 100) ∧
  is_multiple_of_3 ((n / 100) % 10) ∧
  Odd (digits_sum n) ∧
  has_unique_digits n
  →
  n ≤ 25916 :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_code_l3752_375228


namespace NUMINAMATH_CALUDE_father_age_equals_sum_of_brothers_ages_l3752_375261

/-- Represents the current ages of the family members -/
structure FamilyAges where
  ivan : ℕ
  vincent : ℕ
  jakub : ℕ
  father : ℕ

/-- The conditions given in the problem -/
def problem_conditions (ages : FamilyAges) : Prop :=
  ages.vincent = 11 ∧
  ages.jakub = 9 ∧
  ages.ivan = 5 * (ages.jakub / 3) ∧
  ages.father = 3 * ages.ivan

/-- The theorem to be proved -/
theorem father_age_equals_sum_of_brothers_ages (ages : FamilyAges) 
  (h : problem_conditions ages) : 
  ∃ (n : ℕ), n = 5 ∧ 
  ages.father + n = ages.ivan + ages.vincent + ages.jakub + 3 * n :=
sorry

end NUMINAMATH_CALUDE_father_age_equals_sum_of_brothers_ages_l3752_375261


namespace NUMINAMATH_CALUDE_max_intersections_convex_polygons_l3752_375249

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  isConvex : Bool

/-- Represents the state of two polygons after rotation -/
structure RotatedPolygons (Q1 Q2 : ConvexPolygon) where
  canIntersect : Bool

/-- Calculates the maximum number of intersections between two rotated polygons -/
def maxIntersections (Q1 Q2 : ConvexPolygon) (state : RotatedPolygons Q1 Q2) : ℕ :=
  if state.canIntersect then Q1.sides * Q2.sides else 0

theorem max_intersections_convex_polygons :
  ∀ (Q1 Q2 : ConvexPolygon) (state : RotatedPolygons Q1 Q2),
    Q1.sides = 5 →
    Q2.sides = 7 →
    Q1.isConvex = true →
    Q2.isConvex = true →
    state.canIntersect = true →
    maxIntersections Q1 Q2 state = 35 := by
  sorry

end NUMINAMATH_CALUDE_max_intersections_convex_polygons_l3752_375249


namespace NUMINAMATH_CALUDE_halloween_candy_theorem_l3752_375202

-- Define the number of candy pieces collected by each sibling
def maggies_candy : ℕ := 50
def harpers_candy : ℕ := maggies_candy + (maggies_candy * 3 / 10)
def neils_candy : ℕ := harpers_candy + (harpers_candy * 2 / 5)
def liams_candy : ℕ := neils_candy + (neils_candy * 1 / 5)

-- Define the total candy collected
def total_candy : ℕ := maggies_candy + harpers_candy + neils_candy + liams_candy

-- Theorem statement
theorem halloween_candy_theorem : total_candy = 315 := by
  sorry

end NUMINAMATH_CALUDE_halloween_candy_theorem_l3752_375202


namespace NUMINAMATH_CALUDE_five_cubic_yards_to_cubic_feet_l3752_375268

/-- Conversion factor from yards to feet -/
def yards_to_feet : ℝ := 3

/-- Volume in cubic feet for 5 cubic yards -/
def volume_cubic_feet : ℝ := 135

/-- Theorem stating that 5 cubic yards is equal to 135 cubic feet -/
theorem five_cubic_yards_to_cubic_feet :
  (5 : ℝ) * yards_to_feet^3 = volume_cubic_feet := by sorry

end NUMINAMATH_CALUDE_five_cubic_yards_to_cubic_feet_l3752_375268


namespace NUMINAMATH_CALUDE_power_of_product_l3752_375252

theorem power_of_product (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by sorry

end NUMINAMATH_CALUDE_power_of_product_l3752_375252


namespace NUMINAMATH_CALUDE_fourth_root_is_four_l3752_375265

/-- The polynomial with coefficients c and d -/
def polynomial (c d x : ℝ) : ℝ :=
  c * x^4 + (c + 3*d) * x^3 + (d - 4*c) * x^2 + (10 - c) * x + (5 - 2*d)

/-- The theorem stating that if -1, 2, and -3 are roots of the polynomial,
    then 4 is the fourth root -/
theorem fourth_root_is_four (c d : ℝ) :
  polynomial c d (-1) = 0 →
  polynomial c d 2 = 0 →
  polynomial c d (-3) = 0 →
  ∃ x : ℝ, x ≠ -1 ∧ x ≠ 2 ∧ x ≠ -3 ∧ polynomial c d x = 0 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_is_four_l3752_375265


namespace NUMINAMATH_CALUDE_chair_arrangement_l3752_375263

theorem chair_arrangement (total_chairs : Nat) (h1 : total_chairs = 49) :
  (∃! (rows columns : Nat), rows ≥ 2 ∧ columns ≥ 2 ∧ rows * columns = total_chairs) :=
by sorry

end NUMINAMATH_CALUDE_chair_arrangement_l3752_375263


namespace NUMINAMATH_CALUDE_train_length_l3752_375246

/-- The length of a train given its speed and time to pass a fixed point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 63 → time = 16 → speed * time * (1000 / 3600) = 280 :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_l3752_375246


namespace NUMINAMATH_CALUDE_reflect_c_twice_l3752_375295

/-- Reflects a point over the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

/-- Reflects a point over the y-axis -/
def reflect_y (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- Theorem: Reflecting point C(2,2) over x-axis then y-axis results in C''(-2,-2) -/
theorem reflect_c_twice :
  let c : ℝ × ℝ := (2, 2)
  reflect_y (reflect_x c) = (-2, -2) := by
sorry

end NUMINAMATH_CALUDE_reflect_c_twice_l3752_375295


namespace NUMINAMATH_CALUDE_fence_perimeter_is_200_l3752_375269

/-- A rectangular fence with evenly spaced posts -/
structure RectangularFence where
  num_posts : ℕ
  post_width : ℝ
  post_spacing : ℝ
  length_width_ratio : ℝ

/-- Calculate the outer perimeter of a rectangular fence -/
def outer_perimeter (fence : RectangularFence) : ℝ :=
  sorry

/-- Theorem: The outer perimeter of the specified fence is 200 feet -/
theorem fence_perimeter_is_200 :
  let fence : RectangularFence :=
    { num_posts := 36
    , post_width := 0.5  -- 6 inches = 0.5 feet
    , post_spacing := 4
    , length_width_ratio := 2 }
  outer_perimeter fence = 200 :=
by sorry

end NUMINAMATH_CALUDE_fence_perimeter_is_200_l3752_375269


namespace NUMINAMATH_CALUDE_max_area_central_angle_l3752_375204

/-- The circumference of the sector -/
def circumference : ℝ := 40

/-- The radius of the sector -/
noncomputable def radius : ℝ := sorry

/-- The arc length of the sector -/
noncomputable def arc_length : ℝ := sorry

/-- The area of the sector -/
noncomputable def area (r : ℝ) : ℝ := 20 * r - r^2

/-- The central angle of the sector -/
noncomputable def central_angle : ℝ := sorry

/-- Theorem: The central angle that maximizes the area of a sector with circumference 40 is 2 radians -/
theorem max_area_central_angle :
  circumference = 2 * radius + arc_length →
  arc_length = central_angle * radius →
  central_angle = 2 ∧ IsLocalMax area radius :=
sorry

end NUMINAMATH_CALUDE_max_area_central_angle_l3752_375204


namespace NUMINAMATH_CALUDE_triple_base_and_exponent_l3752_375214

variable (a b x : ℝ)
variable (r : ℝ)

theorem triple_base_and_exponent (h1 : b ≠ 0) (h2 : r = (3 * a) ^ (3 * b)) (h3 : r = a ^ b * x ^ b) : x = 27 * a ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_triple_base_and_exponent_l3752_375214


namespace NUMINAMATH_CALUDE_average_temperature_l3752_375272

def temperatures : List ℚ := [73, 76, 75, 78, 74]

theorem average_temperature : 
  (temperatures.sum / temperatures.length : ℚ) = 75.2 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_l3752_375272


namespace NUMINAMATH_CALUDE_sculpture_height_l3752_375242

theorem sculpture_height (base_height : ℝ) (total_height_feet : ℝ) (h1 : base_height = 10) (h2 : total_height_feet = 3.6666666666666665) : 
  total_height_feet * 12 - base_height = 34 := by
sorry

end NUMINAMATH_CALUDE_sculpture_height_l3752_375242


namespace NUMINAMATH_CALUDE_similar_triangles_perimeter_possibilities_l3752_375278

/-- Two triangles are similar if their corresponding angles are equal and the ratios of the lengths of corresponding sides are equal. -/
def similar_triangles (t1 t2 : Triangle) : Prop := sorry

/-- A triangle is defined by its three side lengths. -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- The perimeter of a triangle is the sum of its side lengths. -/
def perimeter (t : Triangle) : ℝ := t.side1 + t.side2 + t.side3

theorem similar_triangles_perimeter_possibilities :
  ∀ (t1 t2 : Triangle),
    similar_triangles t1 t2 →
    t1.side1 = 4 ∧ t1.side2 = 6 ∧ t1.side3 = 8 →
    (t2.side1 = 2 ∨ t2.side2 = 2 ∨ t2.side3 = 2) →
    (perimeter t2 = 4.5 ∨ perimeter t2 = 6 ∨ perimeter t2 = 9) :=
by sorry

end NUMINAMATH_CALUDE_similar_triangles_perimeter_possibilities_l3752_375278


namespace NUMINAMATH_CALUDE_paulas_friends_l3752_375247

/-- Given the initial number of candies, additional candies bought, and candies per friend,
    prove that the number of friends is equal to the total number of candies divided by the number of candies per friend. -/
theorem paulas_friends (initial_candies additional_candies candies_per_friend : ℕ) 
  (h1 : initial_candies = 20)
  (h2 : additional_candies = 4)
  (h3 : candies_per_friend = 4)
  : (initial_candies + additional_candies) / candies_per_friend = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_paulas_friends_l3752_375247


namespace NUMINAMATH_CALUDE_power_of_625_four_fifths_l3752_375230

theorem power_of_625_four_fifths :
  (625 : ℝ) ^ (4/5 : ℝ) = 125 * (5 : ℝ) ^ (1/5 : ℝ) :=
by
  sorry

end NUMINAMATH_CALUDE_power_of_625_four_fifths_l3752_375230


namespace NUMINAMATH_CALUDE_inequality_proof_l3752_375236

theorem inequality_proof (a b x : ℝ) (n : ℕ) (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hn : 2 ≤ n) 
  (h : x^n ≤ a*x + b) : 
  x < (2*a)^(1/(n-1 : ℝ)) + (2*b)^(1/n) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3752_375236


namespace NUMINAMATH_CALUDE_backpack_profit_analysis_l3752_375210

/-- Represents the daily profit function for backpack sales -/
def daily_profit (x : ℝ) : ℝ := -x^2 + 90*x - 1800

/-- Represents the daily sales quantity function -/
def sales_quantity (x : ℝ) : ℝ := -x + 60

theorem backpack_profit_analysis 
  (cost_price : ℝ) 
  (price_range : Set ℝ) 
  (max_price : ℝ) 
  (target_profit : ℝ) :
  cost_price = 30 →
  price_range = {x : ℝ | 30 ≤ x ∧ x ≤ 60} →
  max_price = 48 →
  target_profit = 200 →
  (∀ x ∈ price_range, daily_profit x = (x - cost_price) * sales_quantity x) ∧
  (∃ x ∈ price_range, x ≤ max_price ∧ daily_profit x = target_profit ∧ x = 40) ∧
  (∃ x ∈ price_range, ∀ y ∈ price_range, daily_profit x ≥ daily_profit y ∧ 
    x = 45 ∧ daily_profit x = 225) :=
by sorry

end NUMINAMATH_CALUDE_backpack_profit_analysis_l3752_375210


namespace NUMINAMATH_CALUDE_connect_four_games_total_l3752_375243

/-- Given that Kaleb's ratio of won to lost games is 3:2 and he won 18 games,
    prove that the total number of games played is 30. -/
theorem connect_four_games_total (won lost total : ℕ) : 
  won = 18 → 
  3 * lost = 2 * won → 
  total = won + lost → 
  total = 30 := by
sorry

end NUMINAMATH_CALUDE_connect_four_games_total_l3752_375243


namespace NUMINAMATH_CALUDE_range_of_m_l3752_375276

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, (x^2 - 4*x + 3 < 0 ∧ x^2 - 6*x + 8 < 0) → 2*x^2 - 9*x + m < 0) → 
  m < 9 := by
sorry

end NUMINAMATH_CALUDE_range_of_m_l3752_375276


namespace NUMINAMATH_CALUDE_cloth_selling_price_l3752_375239

/-- Calculates the total selling price of cloth given its length, cost price per meter, and profit per meter. -/
def total_selling_price (length : ℝ) (cost_price_per_meter : ℝ) (profit_per_meter : ℝ) : ℝ :=
  length * (cost_price_per_meter + profit_per_meter)

/-- The total selling price of 78 meters of cloth with a cost price of Rs. 58.02564102564102 per meter
    and a profit of Rs. 29 per meter is approximately Rs. 6788.00. -/
theorem cloth_selling_price :
  let length : ℝ := 78
  let cost_price_per_meter : ℝ := 58.02564102564102
  let profit_per_meter : ℝ := 29
  abs (total_selling_price length cost_price_per_meter profit_per_meter - 6788) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_cloth_selling_price_l3752_375239


namespace NUMINAMATH_CALUDE_smallest_valid_number_l3752_375290

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ),
    n = 1000 + 100 * a + b ∧
    n = (10 * a + b) ^ 2 ∧
    0 ≤ a ∧ a ≤ 99 ∧
    0 ≤ b ∧ b ≤ 99

theorem smallest_valid_number :
  is_valid_number 2025 ∧ ∀ n, is_valid_number n → n ≥ 2025 :=
sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l3752_375290


namespace NUMINAMATH_CALUDE_point_on_line_l3752_375200

theorem point_on_line (x₁ n : ℝ) : 
  (x₁ = n / 5 - 2 / 5 ∧ x₁ + 3 = (n + 15) / 5 - 2 / 5) → 
  x₁ = n / 5 - 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l3752_375200


namespace NUMINAMATH_CALUDE_sin_45_degrees_l3752_375285

theorem sin_45_degrees :
  Real.sin (π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l3752_375285


namespace NUMINAMATH_CALUDE_arccos_one_equals_zero_l3752_375277

theorem arccos_one_equals_zero : Real.arccos 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arccos_one_equals_zero_l3752_375277


namespace NUMINAMATH_CALUDE_expression_equality_l3752_375232

theorem expression_equality : 3 * 2020 + 2 * 2020 - 4 * 2020 = 2020 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3752_375232


namespace NUMINAMATH_CALUDE_paperclips_exceed_250_l3752_375271

def paperclips (n : ℕ) : ℕ := 5 * 2^(n - 1)

theorem paperclips_exceed_250 : 
  ∀ k : ℕ, k < 7 → paperclips k ≤ 250 ∧ paperclips 7 > 250 :=
by sorry

end NUMINAMATH_CALUDE_paperclips_exceed_250_l3752_375271


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3752_375274

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I : ℂ) * (1 + 2 * Complex.I) = (a + b * Complex.I) * (1 + Complex.I) → 
  a = 3/2 ∧ b = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3752_375274


namespace NUMINAMATH_CALUDE_fraction_equals_zero_l3752_375281

theorem fraction_equals_zero (x : ℝ) (h : x + 1 ≠ 0) :
  x = 1 → (x^2 - 1) / (x + 1) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_zero_l3752_375281


namespace NUMINAMATH_CALUDE_vector_decomposition_l3752_375264

/-- Given vectors in ℝ³ -/
def x : Fin 3 → ℝ := ![2, -1, 11]
def p : Fin 3 → ℝ := ![1, 1, 0]
def q : Fin 3 → ℝ := ![0, 1, -2]
def r : Fin 3 → ℝ := ![1, 0, 3]

/-- Theorem stating that x can be expressed as a linear combination of p, q, and r -/
theorem vector_decomposition :
  x = (-3 : ℝ) • p + 2 • q + 5 • r := by
  sorry

end NUMINAMATH_CALUDE_vector_decomposition_l3752_375264


namespace NUMINAMATH_CALUDE_jackson_souvenirs_l3752_375215

theorem jackson_souvenirs :
  let hermit_crabs : ℕ := 45
  let shells_per_crab : ℕ := 3
  let starfish_per_shell : ℕ := 2
  let total_shells : ℕ := hermit_crabs * shells_per_crab
  let total_starfish : ℕ := total_shells * starfish_per_shell
  let total_souvenirs : ℕ := hermit_crabs + total_shells + total_starfish
  total_souvenirs = 450 := by
sorry

end NUMINAMATH_CALUDE_jackson_souvenirs_l3752_375215


namespace NUMINAMATH_CALUDE_sum_9_equals_126_l3752_375248

/-- An arithmetic sequence with the given property -/
structure ArithmeticSequence where
  a : ℕ → ℤ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  property : a 2 + a 8 = 15 + a 5

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  (n * (seq.a 1 + seq.a n)) / 2

/-- The theorem to be proved -/
theorem sum_9_equals_126 (seq : ArithmeticSequence) : sum_n seq 9 = 126 := by
  sorry

end NUMINAMATH_CALUDE_sum_9_equals_126_l3752_375248


namespace NUMINAMATH_CALUDE_missing_mark_calculation_l3752_375298

def calculate_missing_mark (english math physics chemistry average : ℕ) : ℕ :=
  5 * average - (english + math + physics + chemistry)

theorem missing_mark_calculation (english math physics chemistry average biology : ℕ)
  (h1 : english = 76)
  (h2 : math = 65)
  (h3 : physics = 82)
  (h4 : chemistry = 67)
  (h5 : average = 73)
  (h6 : biology = calculate_missing_mark english math physics chemistry average) :
  biology = 75 := by
  sorry

end NUMINAMATH_CALUDE_missing_mark_calculation_l3752_375298


namespace NUMINAMATH_CALUDE_fish_tournament_ratio_l3752_375255

def fish_tournament (jacob_initial : ℕ) (alex_lost : ℕ) (jacob_needed : ℕ) : Prop :=
  ∃ (alex_initial : ℕ) (n : ℕ),
    alex_initial = n * jacob_initial ∧
    jacob_initial + jacob_needed = (alex_initial - alex_lost) + 1 ∧
    alex_initial / jacob_initial = 7

theorem fish_tournament_ratio :
  fish_tournament 8 23 26 := by
  sorry

end NUMINAMATH_CALUDE_fish_tournament_ratio_l3752_375255


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3752_375296

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 = 7*x - 12) → (∃ y : ℝ, y^2 = 7*y - 12 ∧ x + y = 7) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l3752_375296


namespace NUMINAMATH_CALUDE_blue_tiles_in_45th_row_l3752_375216

/-- Calculates the total number of tiles in a row given the row number. -/
def totalTiles (n : ℕ) : ℕ := 2 * n - 1

/-- Calculates the number of blue tiles in a row given the total number of tiles. -/
def blueTiles (total : ℕ) : ℕ := (total - 1) / 2

theorem blue_tiles_in_45th_row :
  blueTiles (totalTiles 45) = 44 := by
  sorry

end NUMINAMATH_CALUDE_blue_tiles_in_45th_row_l3752_375216


namespace NUMINAMATH_CALUDE_back_seat_ticket_cost_l3752_375219

/-- Proves that the cost of back seat tickets is $45 given the concert conditions -/
theorem back_seat_ticket_cost
  (total_seats : ℕ)
  (main_seat_cost : ℕ)
  (total_revenue : ℕ)
  (back_seat_sold : ℕ)
  (h_total_seats : total_seats = 20000)
  (h_main_seat_cost : main_seat_cost = 55)
  (h_total_revenue : total_revenue = 955000)
  (h_back_seat_sold : back_seat_sold = 14500) :
  (total_revenue - (total_seats - back_seat_sold) * main_seat_cost) / back_seat_sold = 45 :=
by sorry

end NUMINAMATH_CALUDE_back_seat_ticket_cost_l3752_375219


namespace NUMINAMATH_CALUDE_computer_table_markup_l3752_375208

/-- Calculate the percentage markup given the selling price and cost price -/
def percentage_markup (selling_price cost_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem: The percentage markup for a computer table with selling price 3000 and cost price 2500 is 20% -/
theorem computer_table_markup :
  percentage_markup 3000 2500 = 20 := by
  sorry

end NUMINAMATH_CALUDE_computer_table_markup_l3752_375208


namespace NUMINAMATH_CALUDE_triangle_side_b_value_l3752_375218

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  2 * Real.sin t.B = Real.sin t.A + Real.sin t.C ∧
  Real.cos t.B = 3/5 ∧
  1/2 * t.a * t.c * Real.sin t.B = 4

-- Theorem statement
theorem triangle_side_b_value (t : Triangle) 
  (h : triangle_conditions t) : t.b = 4 * Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_b_value_l3752_375218


namespace NUMINAMATH_CALUDE_min_value_expression_l3752_375233

theorem min_value_expression (x : ℝ) (h : x > 4) :
  (x + 18) / Real.sqrt (x - 4) ≥ 2 * Real.sqrt 22 ∧
  ∃ x₀ > 4, (x₀ + 18) / Real.sqrt (x₀ - 4) = 2 * Real.sqrt 22 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3752_375233


namespace NUMINAMATH_CALUDE_vector_simplification_l3752_375279

variable {V : Type*} [AddCommGroup V]

theorem vector_simplification (A B C : V) : 
  (B - A) - (C - A) + (C - B) = (0 : V) := by sorry

end NUMINAMATH_CALUDE_vector_simplification_l3752_375279


namespace NUMINAMATH_CALUDE_ring_toss_total_l3752_375203

/-- Calculates the total number of rings used in a ring toss game -/
def total_rings (rings_per_game : ℕ) (games_played : ℕ) : ℕ :=
  rings_per_game * games_played

/-- Theorem: Given 6 rings per game and 8 games played, the total rings used is 48 -/
theorem ring_toss_total :
  total_rings 6 8 = 48 := by
  sorry

end NUMINAMATH_CALUDE_ring_toss_total_l3752_375203


namespace NUMINAMATH_CALUDE_quadratic_equation_condition_l3752_375280

theorem quadratic_equation_condition (m : ℝ) : (|m| = 2 ∧ m + 2 ≠ 0) ↔ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_condition_l3752_375280


namespace NUMINAMATH_CALUDE_kim_money_amount_l3752_375205

theorem kim_money_amount (sal phil : ℝ) (h1 : sal = 0.8 * phil) (h2 : sal + phil = 1.8) : 
  1.4 * sal = 1.12 := by
  sorry

end NUMINAMATH_CALUDE_kim_money_amount_l3752_375205


namespace NUMINAMATH_CALUDE_total_students_presentation_l3752_375237

/-- The total number of students presenting given Eunjeong's position and students after her -/
def total_students (eunjeong_position : Nat) (students_after : Nat) : Nat :=
  (eunjeong_position - 1) + 1 + students_after

/-- Theorem stating the total number of students presenting -/
theorem total_students_presentation : total_students 6 7 = 13 := by
  sorry

end NUMINAMATH_CALUDE_total_students_presentation_l3752_375237


namespace NUMINAMATH_CALUDE_evaluate_expression_l3752_375299

theorem evaluate_expression : Real.sqrt 5 * 5^(1/2 : ℝ) + 20 / 4 * 3 - 9^(3/2 : ℝ) = -7 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3752_375299


namespace NUMINAMATH_CALUDE_icosahedral_die_expected_digits_l3752_375273

/-- The expected number of digits when rolling a fair icosahedral die -/
def expected_digits : ℝ := 1.55

/-- The number of faces on an icosahedral die -/
def num_faces : ℕ := 20

/-- The number of one-digit faces on the die -/
def one_digit_faces : ℕ := 9

/-- The number of two-digit faces on the die -/
def two_digit_faces : ℕ := 11

theorem icosahedral_die_expected_digits :
  expected_digits = (one_digit_faces : ℝ) / num_faces + 2 * (two_digit_faces : ℝ) / num_faces :=
sorry

end NUMINAMATH_CALUDE_icosahedral_die_expected_digits_l3752_375273


namespace NUMINAMATH_CALUDE_expression_evaluation_l3752_375253

theorem expression_evaluation :
  let a : ℤ := 1
  let b : ℤ := 10
  let c : ℤ := 100
  let d : ℤ := 1000
  (a + b + c - d) + (a + b - c + d) + (a - b + c + d) + (-a + b + c + d) = 2222 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3752_375253


namespace NUMINAMATH_CALUDE_part_one_part_two_l3752_375266

-- Define the sets A, B, and U
def A (a : ℝ) : Set ℝ := {x | x - 3 ≤ x ∧ x ≤ 2*a + 1}
def B : Set ℝ := {x | x^2 + 2*x - 15 ≤ 0}
def U : Set ℝ := Set.univ

-- Part I: Prove the intersection of complement of A and B when a = 1
theorem part_one : (Set.compl (A 1) ∩ B) = {x : ℝ | -5 ≤ x ∧ x < -2} := by sorry

-- Part II: Prove the condition for A to be a subset of B
theorem part_two : ∀ a : ℝ, A a ⊆ B ↔ (a < -4 ∨ (-2 ≤ a ∧ a ≤ 1)) := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l3752_375266


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_l3752_375287

def M (m : ℤ) : Set ℤ := {m, -3}

def N : Set ℤ := {x : ℤ | 2*x^2 + 7*x + 3 < 0}

theorem intersection_implies_m_value (m : ℤ) :
  (M m ∩ N).Nonempty → m = -2 ∨ m = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_l3752_375287


namespace NUMINAMATH_CALUDE_initial_female_percentage_calculation_l3752_375286

/-- Represents a company's workforce statistics -/
structure Company where
  initial_employees : ℕ
  initial_female_percentage : ℚ
  hired_male_workers : ℕ
  final_employees : ℕ
  final_female_percentage : ℚ

/-- Theorem stating the relationship between initial and final workforce statistics -/
theorem initial_female_percentage_calculation (c : Company) 
  (h1 : c.hired_male_workers = 20)
  (h2 : c.final_employees = 240)
  (h3 : c.final_female_percentage = 55/100)
  (h4 : c.initial_employees + c.hired_male_workers = c.final_employees)
  (h5 : c.initial_employees * c.initial_female_percentage = 
        c.final_employees * c.final_female_percentage) :
  c.initial_female_percentage = 60/100 := by
  sorry

end NUMINAMATH_CALUDE_initial_female_percentage_calculation_l3752_375286


namespace NUMINAMATH_CALUDE_dennis_teaching_years_l3752_375275

theorem dennis_teaching_years 
  (V A D E N : ℕ) -- Years taught by Virginia, Adrienne, Dennis, Elijah, and Nadine
  (h1 : V + A + D + E + N = 225) -- Total years taught
  (h2 : (V + A + D + E + N) * 5 = (V + A + D + E + N + 150) * 3) -- Total years is 3/5 of age sum
  (h3 : V = A + 9) -- Virginia vs Adrienne
  (h4 : V = D - 15) -- Virginia vs Dennis
  (h5 : E = A - 3) -- Elijah vs Adrienne
  (h6 : E = 2 * N) -- Elijah vs Nadine
  : D = 101 := by
  sorry

end NUMINAMATH_CALUDE_dennis_teaching_years_l3752_375275


namespace NUMINAMATH_CALUDE_sydney_texts_total_l3752_375223

/-- The number of texts Sydney sends to each person on Monday -/
def monday_texts : ℕ := 5

/-- The number of texts Sydney sends to each person on Tuesday -/
def tuesday_texts : ℕ := 15

/-- The number of people Sydney sends texts to -/
def num_recipients : ℕ := 2

/-- The total number of texts Sydney sent over both days -/
def total_texts : ℕ := (monday_texts * num_recipients) + (tuesday_texts * num_recipients)

theorem sydney_texts_total : total_texts = 40 := by
  sorry

end NUMINAMATH_CALUDE_sydney_texts_total_l3752_375223


namespace NUMINAMATH_CALUDE_parabola_sum_l3752_375224

/-- A parabola with equation x = ay² + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_sum (p : Parabola) : 
  p.x_coord (-6) = 8 → p.x_coord (-4) = 10 → p.a + p.b + p.c = 32.5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_l3752_375224


namespace NUMINAMATH_CALUDE_derivative_of_f_l3752_375227

-- Define the function
def f (x : ℝ) : ℝ := 2 * x^5 - 3 * x^2 - 4

-- State the theorem
theorem derivative_of_f (x : ℝ) : 
  deriv f x = 10 * x^4 - 6 * x := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_f_l3752_375227


namespace NUMINAMATH_CALUDE_sum_always_positive_l3752_375211

-- Define a monotonically increasing odd function
def MonoIncreasingOddFunction (f : ℝ → ℝ) : Prop :=
  (∀ x y, x < y → f x < f y) ∧ (∀ x, f (-x) = -f x)

-- Define an arithmetic sequence
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_always_positive
  (f : ℝ → ℝ)
  (a : ℕ → ℝ)
  (hf : MonoIncreasingOddFunction f)
  (ha : ArithmeticSequence a)
  (ha3_pos : a 3 > 0) :
  f (a 1) + f (a 3) + f (a 5) > 0 :=
sorry

end NUMINAMATH_CALUDE_sum_always_positive_l3752_375211


namespace NUMINAMATH_CALUDE_carl_index_cards_cost_l3752_375244

/-- Calculates the total cost of index cards for Carl's classes -/
def total_cost (cards_per_student : ℕ) (periods : ℕ) (students_per_class : ℕ) (cards_per_pack : ℕ) (cost_per_pack : ℕ) : ℕ :=
  let total_students := periods * students_per_class
  let total_cards := total_students * cards_per_student
  let packs_needed := (total_cards + cards_per_pack - 1) / cards_per_pack  -- Ceiling division
  packs_needed * cost_per_pack

/-- Proves that the total cost of index cards for Carl's classes is $108 -/
theorem carl_index_cards_cost : 
  total_cost 10 6 30 50 3 = 108 := by
  sorry

end NUMINAMATH_CALUDE_carl_index_cards_cost_l3752_375244


namespace NUMINAMATH_CALUDE_census_suitable_for_electricity_usage_l3752_375229

/-- Represents a survey method -/
inductive SurveyMethod
| Census
| Sampling

/-- Represents a survey population -/
structure Population where
  size : ℕ
  is_small : Bool
  is_manageable : Bool

/-- Represents a survey -/
structure Survey where
  population : Population
  method : SurveyMethod
  is_practical : Bool

/-- Theorem: A census method is most suitable for investigating the monthly average 
    electricity usage of 10 households in a residential building -/
theorem census_suitable_for_electricity_usage : 
  ∀ (p : Population) (s : Survey),
  p.size = 10 → 
  p.is_small = true → 
  p.is_manageable = true → 
  s.population = p → 
  s.is_practical = true → 
  s.method = SurveyMethod.Census :=
by sorry

end NUMINAMATH_CALUDE_census_suitable_for_electricity_usage_l3752_375229


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocal_line_l3752_375259

theorem min_sum_of_reciprocal_line (a b : ℝ) : 
  a > 0 → b > 0 → (1 : ℝ) / a + (1 : ℝ) / b = 1 → (a + b) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocal_line_l3752_375259


namespace NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l3752_375256

theorem multiplication_table_odd_fraction :
  let table_size : ℕ := 16
  let total_products : ℕ := table_size * table_size
  let odd_numbers : ℕ := (table_size + 1) / 2
  let odd_products : ℕ := odd_numbers * odd_numbers
  (odd_products : ℚ) / total_products = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_multiplication_table_odd_fraction_l3752_375256


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l3752_375201

theorem quadratic_equation_root (k : ℝ) : 
  (2 : ℝ) ∈ {x : ℝ | 2 * x^2 - 8 * x + k = 0} → k = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l3752_375201


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3752_375209

-- Define propositions p and q
def p (x : ℝ) : Prop := |x| < 2
def q (x : ℝ) : Prop := x^2 - x - 2 < 0

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem stating that ¬p is a sufficient but not necessary condition for ¬q
theorem not_p_sufficient_not_necessary_for_not_q :
  (∀ x : ℝ, not_p x → not_q x) ∧ 
  (∃ x : ℝ, not_q x ∧ ¬(not_p x)) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_for_not_q_l3752_375209


namespace NUMINAMATH_CALUDE_quadratic_equation_integer_solutions_l3752_375297

theorem quadratic_equation_integer_solutions :
  ∀ (x n : ℤ), x^2 + 3*x + 9 - 9*n^2 = 0 → (x = 0 ∨ x = -3) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_integer_solutions_l3752_375297


namespace NUMINAMATH_CALUDE_f_is_increasing_on_reals_l3752_375258

-- Define the function
def f (x : ℝ) : ℝ := x

-- State the theorem
theorem f_is_increasing_on_reals :
  (∀ x, x ∈ Set.univ → f x ∈ Set.univ) ∧
  (∀ x y, x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_is_increasing_on_reals_l3752_375258


namespace NUMINAMATH_CALUDE_gummy_worm_fraction_l3752_375284

theorem gummy_worm_fraction (initial_count : ℕ) (days : ℕ) (final_count : ℕ) (f : ℚ) :
  initial_count = 64 →
  days = 4 →
  final_count = 4 →
  0 < f →
  f < 1 →
  (1 - f) ^ days * initial_count = final_count →
  f = 1/2 := by
sorry

end NUMINAMATH_CALUDE_gummy_worm_fraction_l3752_375284


namespace NUMINAMATH_CALUDE_more_girls_than_boys_l3752_375225

theorem more_girls_than_boys 
  (total_pupils : ℕ) 
  (girls : ℕ) 
  (h1 : total_pupils = 1455)
  (h2 : girls = 868)
  (h3 : girls > total_pupils - girls) : 
  girls - (total_pupils - girls) = 281 :=
by
  sorry

end NUMINAMATH_CALUDE_more_girls_than_boys_l3752_375225


namespace NUMINAMATH_CALUDE_unique_configuration_l3752_375235

-- Define the type for statements
inductive Statement
| one_false : Statement
| two_false : Statement
| three_false : Statement
| four_false : Statement
| one_true : Statement

-- Define a function to evaluate the truth value of a statement
def evaluate (s : Statement) (true_count : Nat) : Prop :=
  match s with
  | Statement.one_false => true_count = 4
  | Statement.two_false => true_count = 3
  | Statement.three_false => true_count = 2
  | Statement.four_false => true_count = 1
  | Statement.one_true => true_count = 1

-- Define the card as a list of statements
def card : List Statement := [
  Statement.one_false,
  Statement.two_false,
  Statement.three_false,
  Statement.four_false,
  Statement.one_true
]

-- Theorem: There exists a unique configuration with exactly one true statement
theorem unique_configuration :
  ∃! true_count : Nat,
    true_count ≤ 5 ∧
    true_count > 0 ∧
    (∀ s ∈ card, evaluate s true_count ↔ s = Statement.one_true) :=
by sorry

end NUMINAMATH_CALUDE_unique_configuration_l3752_375235


namespace NUMINAMATH_CALUDE_unique_solution_condition_l3752_375238

theorem unique_solution_condition (A B : ℝ) :
  (∀ x y : ℝ, A * x + B * ⌊x⌋ = A * y + B * ⌊y⌋ → x = y) ↔ 
  (A = 0 ∨ -2 < B / A ∧ B / A < 0) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_condition_l3752_375238


namespace NUMINAMATH_CALUDE_base6_two_distinct_primes_l3752_375293

/-- Represents a number in base 6 formed by appending fives to 1200 -/
def base6Number (n : ℕ) : ℕ :=
  288 * 6^(10*n + 2) + (6^(10*n + 2) - 1)

/-- Counts the number of distinct prime factors of a natural number -/
noncomputable def countDistinctPrimeFactors (x : ℕ) : ℕ := sorry

/-- Theorem stating that the base 6 number has exactly two distinct prime factors iff n = 0 -/
theorem base6_two_distinct_primes (n : ℕ) : 
  countDistinctPrimeFactors (base6Number n) = 2 ↔ n = 0 := by sorry

end NUMINAMATH_CALUDE_base6_two_distinct_primes_l3752_375293


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l3752_375257

/-- If p and q are nonzero real numbers and (3 - 4i)(p + qi) is pure imaginary, then p/q = -4/3 -/
theorem pure_imaginary_ratio (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) 
  (h : ∃ (y : ℝ), (3 - 4 * Complex.I) * (p + q * Complex.I) = y * Complex.I) : 
  p / q = -4 / 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l3752_375257


namespace NUMINAMATH_CALUDE_fixed_intersection_point_l3752_375291

/-- Represents a circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents an angle with two sides -/
structure Angle where
  vertex : ℝ × ℝ
  side1 : ℝ × ℝ → Prop
  side2 : ℝ × ℝ → Prop

/-- Predicate to check if two circles are non-overlapping -/
def non_overlapping (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 > (c1.radius + c2.radius)^2

/-- Predicate to check if a point is on a circle -/
def on_circle (p : ℝ × ℝ) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

/-- Predicate to check if the angle touches both circles -/
def touches_circles (a : Angle) (c1 c2 : Circle) : Prop :=
  ∃ p1 p2 : ℝ × ℝ,
    a.side1 p1 ∧ on_circle p1 c1 ∧
    a.side2 p2 ∧ on_circle p2 c2 ∧
    p1 ≠ a.vertex ∧ p2 ≠ a.vertex

/-- The main theorem -/
theorem fixed_intersection_point
  (c1 c2 : Circle)
  (h_non_overlapping : non_overlapping c1 c2) :
  ∃ p : ℝ × ℝ,
    ∀ a : Angle,
      touches_circles a c1 c2 →
      ∃ t : ℝ,
        p.1 = a.vertex.1 + t * (p.1 - a.vertex.1) ∧
        p.2 = a.vertex.2 + t * (p.2 - a.vertex.2) :=
  sorry

end NUMINAMATH_CALUDE_fixed_intersection_point_l3752_375291


namespace NUMINAMATH_CALUDE_total_weight_in_kg_l3752_375245

-- Define the weights in grams
def monosodium_glutamate : ℕ := 80
def salt : ℕ := 500
def laundry_detergent : ℕ := 420

-- Define the conversion factor from grams to kilograms
def grams_per_kg : ℕ := 1000

-- Theorem statement
theorem total_weight_in_kg :
  (monosodium_glutamate + salt + laundry_detergent) / grams_per_kg = 1 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_in_kg_l3752_375245


namespace NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l3752_375240

theorem or_necessary_not_sufficient_for_and (p q : Prop) :
  (p ∧ q → p ∨ q) ∧ ∃ (p q : Prop), (p ∨ q) ∧ ¬(p ∧ q) :=
by sorry

end NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l3752_375240


namespace NUMINAMATH_CALUDE_square_area_ratio_l3752_375288

theorem square_area_ratio (a b : ℝ) (h : 4 * a = 16 * b) : a^2 / b^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3752_375288


namespace NUMINAMATH_CALUDE_sticker_distribution_l3752_375289

/-- The number of ways to distribute n indistinguishable objects among k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of stickers -/
def num_stickers : ℕ := 10

/-- The number of sheets of paper -/
def num_sheets : ℕ := 5

/-- Theorem stating that there are 935 ways to distribute 10 stickers among 5 sheets -/
theorem sticker_distribution : distribute num_stickers num_sheets = 935 := by sorry

end NUMINAMATH_CALUDE_sticker_distribution_l3752_375289


namespace NUMINAMATH_CALUDE_pre_bought_ticket_price_l3752_375220

/-- The price of pre-bought plane tickets is $155 -/
theorem pre_bought_ticket_price :
  ∀ (pre_bought_price : ℕ) (pre_bought_quantity : ℕ) (gate_price : ℕ) (gate_quantity : ℕ) (price_difference : ℕ),
  pre_bought_quantity = 20 →
  gate_quantity = 30 →
  gate_price = 200 →
  gate_quantity * gate_price = pre_bought_quantity * pre_bought_price + price_difference →
  price_difference = 2900 →
  pre_bought_price = 155 :=
by sorry

end NUMINAMATH_CALUDE_pre_bought_ticket_price_l3752_375220


namespace NUMINAMATH_CALUDE_square_difference_l3752_375270

theorem square_difference (x y : ℝ) (h1 : (x + y)^2 = 49) (h2 : x * y = 8) : 
  (x - y)^2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l3752_375270


namespace NUMINAMATH_CALUDE_digit_values_divisible_by_99_l3752_375217

theorem digit_values_divisible_by_99 (x y : Nat) : 
  (0 ≤ x ∧ x ≤ 9) → 
  (0 ≤ y ∧ y ≤ 9) → 
  (99 ∣ (141000 + 10000*x + 280 + 10*y + 3)) → 
  (x = 4 ∧ y = 4) := by
sorry

end NUMINAMATH_CALUDE_digit_values_divisible_by_99_l3752_375217


namespace NUMINAMATH_CALUDE_max_product_sum_2024_l3752_375241

theorem max_product_sum_2024 : 
  ∃ (x : ℤ), x * (2024 - x) = 1024144 ∧ 
  ∀ (y : ℤ), y * (2024 - y) ≤ 1024144 := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2024_l3752_375241


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3752_375254

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - 2*x)^9 = a₉*x^9 + a₈*x^8 + a₇*x^7 + a₆*x^6 + a₅*x^5 + a₄*x^4 + a₃*x^3 + a₂*x^2 + a₁*x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ = -2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3752_375254


namespace NUMINAMATH_CALUDE_problem_solution_l3752_375207

theorem problem_solution (x y : ℝ) (some_number : ℝ) 
  (h1 : x + 3 * y = some_number) 
  (h2 : y = 10) 
  (h3 : x = 3) : 
  some_number = 33 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3752_375207


namespace NUMINAMATH_CALUDE_largest_n_divisible_by_seven_largest_n_is_49999_l3752_375267

theorem largest_n_divisible_by_seven (n : ℕ) : 
  n < 50000 →
  (3 * (n - 3)^2 - 4 * n + 28) % 7 = 0 →
  n ≤ 49999 :=
by sorry

theorem largest_n_is_49999 : 
  (3 * (49999 - 3)^2 - 4 * 49999 + 28) % 7 = 0 ∧
  ∀ m : ℕ, m > 49999 → m < 50000 → (3 * (m - 3)^2 - 4 * m + 28) % 7 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_divisible_by_seven_largest_n_is_49999_l3752_375267


namespace NUMINAMATH_CALUDE_complex_fraction_difference_l3752_375250

theorem complex_fraction_difference (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (3 + Complex.I) / (1 - Complex.I) = a + b * Complex.I →
  a - b = -1 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_difference_l3752_375250


namespace NUMINAMATH_CALUDE_rational_triplet_problem_l3752_375282

theorem rational_triplet_problem (m n p : ℚ) : 
  m > 0 ∧ n > 0 ∧ p > 0 →
  (∃ (a b c : ℤ), m + 1 / (n * p) = a ∧ n + 1 / (p * m) = b ∧ p + 1 / (m * n) = c) →
  ((m = 1/2 ∧ n = 1/2 ∧ p = 4) ∨ 
   (m = 1/2 ∧ n = 1 ∧ p = 2) ∨ 
   (m = 1 ∧ n = 1 ∧ p = 1) ∨
   (m = 1/2 ∧ n = 4 ∧ p = 1/2) ∨
   (m = 1 ∧ n = 2 ∧ p = 1/2) ∨
   (m = 4 ∧ n = 1/2 ∧ p = 1/2) ∨
   (m = 2 ∧ n = 1/2 ∧ p = 1) ∨
   (m = 2 ∧ n = 1 ∧ p = 1/2) ∨
   (m = 1/2 ∧ n = 2 ∧ p = 1)) :=
by sorry

end NUMINAMATH_CALUDE_rational_triplet_problem_l3752_375282


namespace NUMINAMATH_CALUDE_largest_prime_factor_87_l3752_375251

def numbers : List Nat := [65, 87, 143, 169, 187]

def largest_prime_factor (n : Nat) : Nat :=
  Nat.factors n |>.foldl max 0

theorem largest_prime_factor_87 :
  ∀ n ∈ numbers, n ≠ 87 → largest_prime_factor n < largest_prime_factor 87 :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_factor_87_l3752_375251


namespace NUMINAMATH_CALUDE_factor_tree_problem_l3752_375221

theorem factor_tree_problem (X Y Z W : ℕ) : 
  X = Y * Z ∧ 
  Y = 7 * 11 ∧ 
  Z = 2 * W ∧ 
  W = 3 * 2 → 
  X = 924 := by sorry

end NUMINAMATH_CALUDE_factor_tree_problem_l3752_375221


namespace NUMINAMATH_CALUDE_gcd_78_36_l3752_375226

theorem gcd_78_36 : Nat.gcd 78 36 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_78_36_l3752_375226


namespace NUMINAMATH_CALUDE_tangent_product_special_angles_l3752_375294

theorem tangent_product_special_angles :
  let A : Real := 30 * π / 180
  let B : Real := 40 * π / 180
  let C : Real := 5 * π / 180
  (1 + Real.tan A) * (1 + Real.tan B) * (1 + Real.tan C) = 3 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_product_special_angles_l3752_375294
