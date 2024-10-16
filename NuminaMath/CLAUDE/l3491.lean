import Mathlib

namespace NUMINAMATH_CALUDE_geometric_progression_ratio_condition_l3491_349197

theorem geometric_progression_ratio_condition 
  (x y z w r : ℝ) 
  (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ w ≠ 0)
  (h2 : x * (y - z) ≠ y * (z - x) ∧ 
        y * (z - x) ≠ z * (x - y) ∧ 
        z * (x - y) ≠ w * (y - x))
  (h3 : ∃ (a : ℝ), a ≠ 0 ∧ 
        x * (y - z) = a ∧ 
        y * (z - x) = a * r ∧ 
        z * (x - y) = a * r^2 ∧ 
        w * (y - x) = a * r^3) :
  r^3 + r^2 + r + 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_condition_l3491_349197


namespace NUMINAMATH_CALUDE_dave_candy_bars_l3491_349185

/-- Proves that Dave paid for 6 candy bars given the problem conditions -/
theorem dave_candy_bars (total_bars : ℕ) (cost_per_bar : ℚ) (john_paid : ℚ) : 
  total_bars = 20 →
  cost_per_bar = 3/2 →
  john_paid = 21 →
  (total_bars : ℚ) * cost_per_bar - john_paid = 6 * cost_per_bar :=
by sorry

end NUMINAMATH_CALUDE_dave_candy_bars_l3491_349185


namespace NUMINAMATH_CALUDE_women_meeting_point_l3491_349172

/-- Represents the distance walked by the woman starting from point B -/
def distance_B (h : ℕ) : ℚ :=
  h * (h + 3) / 2

/-- Represents the total distance walked by both women -/
def total_distance (h : ℕ) : ℚ :=
  3 * h + distance_B h

theorem women_meeting_point :
  ∃ (h : ℕ), h > 0 ∧ total_distance h = 60 ∧ distance_B h - 3 * h = 6 := by
  sorry

end NUMINAMATH_CALUDE_women_meeting_point_l3491_349172


namespace NUMINAMATH_CALUDE_actual_payment_calculation_l3491_349182

/-- Represents the restaurant's voucher system and discount policy -/
structure Restaurant where
  voucher_cost : ℕ := 25
  voucher_value : ℕ := 50
  max_vouchers : ℕ := 3
  hotpot_base_cost : ℕ := 50
  other_dishes_discount : ℚ := 0.4

/-- Represents a family's dining experience -/
structure DiningExperience where
  restaurant : Restaurant
  total_bill : ℕ
  voucher_savings : ℕ
  onsite_discount_savings : ℕ

/-- The theorem to be proved -/
theorem actual_payment_calculation (d : DiningExperience) :
  d.restaurant.hotpot_base_cost = 50 ∧
  d.restaurant.voucher_cost = 25 ∧
  d.restaurant.voucher_value = 50 ∧
  d.restaurant.max_vouchers = 3 ∧
  d.restaurant.other_dishes_discount = 0.4 ∧
  d.onsite_discount_savings = d.voucher_savings + 15 →
  d.total_bill - d.onsite_discount_savings = 185 := by
  sorry


end NUMINAMATH_CALUDE_actual_payment_calculation_l3491_349182


namespace NUMINAMATH_CALUDE_m_range_l3491_349125

-- Define the function f on the interval [-2, 2]
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom f_domain : ∀ x, -2 ≤ x ∧ x ≤ 2 → f x ≠ 0
axiom f_even : ∀ x, -2 ≤ x ∧ x ≤ 2 → f (-x) = f x
axiom f_decreasing : ∀ a b, 0 ≤ a ∧ a ≤ 2 → 0 ≤ b ∧ b ≤ 2 → a ≠ b → (f a - f b) / (a - b) < 0

-- Define the theorem
theorem m_range (m : ℝ) (h : f (1 - m) < f m) : -1 ≤ m ∧ m < 1/2 :=
sorry

end NUMINAMATH_CALUDE_m_range_l3491_349125


namespace NUMINAMATH_CALUDE_dark_tile_fraction_is_one_third_l3491_349192

-- Define the properties of the tiled floor
structure TiledFloor :=
  (section_width : ℕ)
  (section_height : ℕ)
  (dark_tiles_per_section : ℕ)

-- Define the fraction of dark tiles
def dark_tile_fraction (floor : TiledFloor) : ℚ :=
  floor.dark_tiles_per_section / (floor.section_width * floor.section_height)

-- Theorem statement
theorem dark_tile_fraction_is_one_third 
  (floor : TiledFloor) 
  (h1 : floor.section_width = 6) 
  (h2 : floor.section_height = 4) 
  (h3 : floor.dark_tiles_per_section = 8) : 
  dark_tile_fraction floor = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_dark_tile_fraction_is_one_third_l3491_349192


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3491_349198

theorem quadratic_function_property (a b c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (f 0 = f 4 ∧ f 0 > f 1) → (a > 0 ∧ 4 * a + b = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3491_349198


namespace NUMINAMATH_CALUDE_triangulated_square_interior_points_l3491_349114

/-- Represents a square divided into triangles -/
structure TriangulatedSquare where
  /-- The number of triangles in the square -/
  num_triangles : ℕ
  /-- The number of interior points (vertices of triangles) -/
  num_interior_points : ℕ
  /-- Condition: No vertex lies on sides or inside other triangles -/
  no_overlap : Prop
  /-- Condition: Sides of square are sides of some triangles -/
  square_sides_are_triangle_sides : Prop

/-- Theorem: A square divided into 2016 triangles has 1007 interior points -/
theorem triangulated_square_interior_points
  (ts : TriangulatedSquare)
  (h_num_triangles : ts.num_triangles = 2016) :
  ts.num_interior_points = 1007 := by
  sorry

#check triangulated_square_interior_points

end NUMINAMATH_CALUDE_triangulated_square_interior_points_l3491_349114


namespace NUMINAMATH_CALUDE_differential_equation_solution_l3491_349108

/-- Given a differential equation y = x * y' + a / (2 * y'), where a is a constant,
    prove that the solutions are:
    1. y = C * x + a / (2 * C), where C is a constant
    2. y^2 = 2 * a * x
-/
theorem differential_equation_solution (a : ℝ) (x y : ℝ → ℝ) (y' : ℝ → ℝ) :
  (∀ t, y t = t * y' t + a / (2 * y' t)) →
  (∃ C : ℝ, ∀ t, y t = C * t + a / (2 * C)) ∨
  (∀ t, (y t)^2 = 2 * a * t) := by
  sorry

end NUMINAMATH_CALUDE_differential_equation_solution_l3491_349108


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l3491_349106

-- Define the number of balls of each color
def white_balls : ℕ := 6
def yellow_balls : ℕ := 5
def red_balls : ℕ := 4

-- Define the total number of balls
def total_balls : ℕ := white_balls + yellow_balls + red_balls

-- Define the probability of drawing a white ball
def prob_white : ℚ := white_balls / total_balls

-- Theorem statement
theorem probability_of_white_ball : prob_white = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l3491_349106


namespace NUMINAMATH_CALUDE_prop_p_true_prop_q_false_prop_2_true_prop_3_true_l3491_349142

-- Define propositions p and q
def p : Prop := ∀ x y : ℝ, x > y → -x < -y
def q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- Theorem statements
theorem prop_p_true : p := by sorry

theorem prop_q_false : ¬q := by sorry

theorem prop_2_true : p ∨ q := by sorry

theorem prop_3_true : p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_prop_p_true_prop_q_false_prop_2_true_prop_3_true_l3491_349142


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3491_349118

/-- An arithmetic sequence with first term a₁ and common difference d -/
def arithmeticSequence (a₁ d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem tenth_term_of_sequence :
  let a₁ := (3 : ℚ) / 4
  let a₂ := (5 : ℚ) / 4
  let a₃ := (7 : ℚ) / 4
  let d := a₂ - a₁
  arithmeticSequence a₁ d 10 = (21 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3491_349118


namespace NUMINAMATH_CALUDE_dealer_cheating_dealer_fair_iff_l3491_349199

theorem dealer_cheating (a w : ℝ) (ha : a > 0) (hw : w > 0) (hna : a ≠ 1) :
  (a * w + w / a) / 2 ≥ w :=
by sorry

theorem dealer_fair_iff (a w : ℝ) (ha : a > 0) (hw : w > 0) :
  (a * w + w / a) / 2 = w ↔ a = 1 :=
by sorry

end NUMINAMATH_CALUDE_dealer_cheating_dealer_fair_iff_l3491_349199


namespace NUMINAMATH_CALUDE_probability_theorem_l3491_349171

-- Define the total number of children
def total_children : ℕ := 9

-- Define the number of children with green hats
def green_hats : ℕ := 3

-- Define the function to calculate the probability
def probability_no_adjacent_green_hats (n : ℕ) (k : ℕ) : ℚ :=
  -- The actual calculation would go here, but we'll use sorry to skip the proof
  5 / 14

-- State the theorem
theorem probability_theorem :
  probability_no_adjacent_green_hats total_children green_hats = 5 / 14 := by
  sorry

end NUMINAMATH_CALUDE_probability_theorem_l3491_349171


namespace NUMINAMATH_CALUDE_locus_of_D_l3491_349115

-- Define the basic structure for points in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define a function to calculate the area of a triangle
def triangleArea (A B C : Point) : ℝ := sorry

-- Define a function to calculate the area of a quadrilateral
def quadArea (A B C D : Point) : ℝ := sorry

-- Define a function to check if three points are collinear
def collinear (A B C : Point) : Prop := sorry

-- Define a function to calculate the distance from a point to a line
def distanceToLine (P : Point) (A B : Point) : ℝ := sorry

-- Define a function to check if a point is on a line
def onLine (P : Point) (A B : Point) : Prop := sorry

-- Theorem statement
theorem locus_of_D (A B C D : Point) :
  ¬collinear A B C →
  quadArea A B C D = 3 * triangleArea A B C →
  ∃ (k : ℝ), distanceToLine D A C = 4 * distanceToLine B A C ∧
             ¬onLine D A B ∧
             ¬onLine D B C :=
sorry

end NUMINAMATH_CALUDE_locus_of_D_l3491_349115


namespace NUMINAMATH_CALUDE_fraction_equality_l3491_349184

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5 * x + 2 * y) / (x - 5 * y) = 3) : 
  (x + 5 * y) / (5 * x - y) = 7 / 87 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3491_349184


namespace NUMINAMATH_CALUDE_sin_theta_plus_7pi_6_l3491_349160

theorem sin_theta_plus_7pi_6 (θ : ℝ) 
  (h : Real.cos (θ - π/6) + Real.sin θ = 4 * Real.sqrt 3 / 5) : 
  Real.sin (θ + 7*π/6) = -4/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_theta_plus_7pi_6_l3491_349160


namespace NUMINAMATH_CALUDE_greatest_gcd_4Tn_n_minus_1_l3491_349113

/-- The nth triangular number -/
def T (n : ℕ+) : ℕ := (n * (n + 1)) / 2

/-- The statement to be proved -/
theorem greatest_gcd_4Tn_n_minus_1 :
  ∃ (k : ℕ+), ∀ (n : ℕ+), Nat.gcd (4 * T n) (n - 1) ≤ 4 ∧
  Nat.gcd (4 * T k) (k - 1) = 4 :=
sorry

end NUMINAMATH_CALUDE_greatest_gcd_4Tn_n_minus_1_l3491_349113


namespace NUMINAMATH_CALUDE_math_club_challenge_l3491_349176

theorem math_club_challenge : ((7^2 - 5^2) - 2)^3 = 10648 := by
  sorry

end NUMINAMATH_CALUDE_math_club_challenge_l3491_349176


namespace NUMINAMATH_CALUDE_units_digit_sum_factorials_49_l3491_349161

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sum_factorials (n : ℕ) : ℕ := (List.range n).map factorial |>.sum

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_sum_factorials_49 :
  units_digit (sum_factorials 49) = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_sum_factorials_49_l3491_349161


namespace NUMINAMATH_CALUDE_second_number_is_22_l3491_349193

theorem second_number_is_22 (x y : ℝ) (h1 : x + y = 33) (h2 : y = 2 * x) : y = 22 := by
  sorry

end NUMINAMATH_CALUDE_second_number_is_22_l3491_349193


namespace NUMINAMATH_CALUDE_triangle_probability_is_2ln2_minus_1_l3491_349145

-- Define the rod breaking process
def rod_break (total_length : ℝ) : ℝ × ℝ × ℝ :=
  sorry

-- Define the condition for triangle formation
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the probability of forming a triangle
def triangle_probability : ℝ :=
  sorry

-- Theorem statement
theorem triangle_probability_is_2ln2_minus_1 :
  triangle_probability = 2 * Real.log 2 - 1 :=
sorry

end NUMINAMATH_CALUDE_triangle_probability_is_2ln2_minus_1_l3491_349145


namespace NUMINAMATH_CALUDE_twentieth_triangular_number_l3491_349194

/-- Definition of triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 20th triangular number is 210 -/
theorem twentieth_triangular_number : triangular_number 20 = 210 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_triangular_number_l3491_349194


namespace NUMINAMATH_CALUDE_x_pos_sufficient_not_necessary_for_abs_x_pos_l3491_349126

theorem x_pos_sufficient_not_necessary_for_abs_x_pos :
  (∃ (x : ℝ), |x| > 0 ∧ x ≤ 0) ∧
  (∀ (x : ℝ), x > 0 → |x| > 0) :=
by sorry

end NUMINAMATH_CALUDE_x_pos_sufficient_not_necessary_for_abs_x_pos_l3491_349126


namespace NUMINAMATH_CALUDE_oil_depths_in_elliptical_tank_l3491_349102

/-- Represents an elliptical oil tank lying horizontally -/
structure EllipticalTank where
  length : ℝ
  majorAxis : ℝ
  minorAxis : ℝ

/-- Calculates the possible oil depths in an elliptical tank -/
def calculateOilDepths (tank : EllipticalTank) (oilSurfaceArea : ℝ) : Set ℝ :=
  sorry

/-- The theorem stating the correct oil depths for the given tank and oil surface area -/
theorem oil_depths_in_elliptical_tank :
  let tank : EllipticalTank := { length := 10, majorAxis := 8, minorAxis := 6 }
  let oilSurfaceArea : ℝ := 48
  calculateOilDepths tank oilSurfaceArea = {1.2, 4.8} := by
  sorry

end NUMINAMATH_CALUDE_oil_depths_in_elliptical_tank_l3491_349102


namespace NUMINAMATH_CALUDE_divisible_by_18_sqrt_between_30_and_30_5_l3491_349122

theorem divisible_by_18_sqrt_between_30_and_30_5 : 
  ∀ n : ℕ, 
    n > 0 ∧ 
    n % 18 = 0 ∧ 
    30 < Real.sqrt n ∧ 
    Real.sqrt n < 30.5 → 
    n = 900 ∨ n = 918 :=
by sorry

end NUMINAMATH_CALUDE_divisible_by_18_sqrt_between_30_and_30_5_l3491_349122


namespace NUMINAMATH_CALUDE_inequality_proof_l3491_349154

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_sum : a + b + c = 1) : 
  (1/a - 1) * (1/b - 1) * (1/c - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3491_349154


namespace NUMINAMATH_CALUDE_first_group_size_l3491_349189

/-- The amount of work done by one person in one day -/
def work_unit : ℝ := 1

/-- The number of days -/
def days : ℕ := 3

/-- The number of people in the second group -/
def people_second_group : ℕ := 8

/-- The amount of work done by the first group -/
def work_first_group : ℝ := 3

/-- The amount of work done by the second group -/
def work_second_group : ℝ := 8

/-- The number of people in the first group -/
def people_first_group : ℕ := 3

theorem first_group_size :
  (people_first_group : ℝ) * days * work_unit = work_first_group ∧
  (people_second_group : ℝ) * days * work_unit = work_second_group →
  people_first_group = 3 := by
  sorry

end NUMINAMATH_CALUDE_first_group_size_l3491_349189


namespace NUMINAMATH_CALUDE_inequality_solution_existence_l3491_349128

theorem inequality_solution_existence (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ x^2 + |x + a| < 2) ↔ a ∈ Set.Icc (-9/4 : ℝ) 2 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_existence_l3491_349128


namespace NUMINAMATH_CALUDE_ellipse_equation_l3491_349121

/-- The equation of an ellipse with foci at (-2,0) and (2,0) passing through (2, 5/3) -/
theorem ellipse_equation (x y : ℝ) : 
  (∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
    (∀ (x₀ y₀ : ℝ), x₀^2/a^2 + y₀^2/b^2 = 1 ↔ 
      (Real.sqrt ((x₀ + 2)^2 + y₀^2) + Real.sqrt ((x₀ - 2)^2 + y₀^2) = 2*a)) ∧
    (2^2/a^2 + (5/3)^2/b^2 = 1)) →
  x^2/9 + y^2/5 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3491_349121


namespace NUMINAMATH_CALUDE_s_3_equals_149_l3491_349168

-- Define the function s(n)
def s (n : ℕ) : ℕ :=
  let squares := List.range n |>.map (λ i => (i + 1) ^ 2)
  squares.foldl (λ acc x => acc * 10^(Nat.digits 10 x).length + x) 0

-- State the theorem
theorem s_3_equals_149 : s 3 = 149 := by
  sorry

end NUMINAMATH_CALUDE_s_3_equals_149_l3491_349168


namespace NUMINAMATH_CALUDE_degree_of_g_is_two_l3491_349110

/-- The degree of a polynomial -/
noncomputable def degree (p : Polynomial ℝ) : ℕ := sorry

/-- Composition of polynomials -/
def compose (f g : Polynomial ℝ) : Polynomial ℝ := sorry

theorem degree_of_g_is_two
  (f g : Polynomial ℝ)
  (h : Polynomial ℝ)
  (h_def : h = compose f g + g)
  (deg_h : degree h = 8)
  (deg_f : degree f = 3) :
  degree g = 2 := by sorry

end NUMINAMATH_CALUDE_degree_of_g_is_two_l3491_349110


namespace NUMINAMATH_CALUDE_problem_solution_l3491_349140

theorem problem_solution : (2010^2 - 2010 + 1) / (2010 + 1) = 4040091 / 2011 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3491_349140


namespace NUMINAMATH_CALUDE_accessory_production_equation_l3491_349170

theorem accessory_production_equation 
  (initial_production : ℕ) 
  (total_production : ℕ) 
  (x : ℝ) 
  (h1 : initial_production = 600000) 
  (h2 : total_production = 2180000) :
  (600 : ℝ) + 600 * (1 + x) + 600 * (1 + x)^2 = 2180 :=
by sorry

end NUMINAMATH_CALUDE_accessory_production_equation_l3491_349170


namespace NUMINAMATH_CALUDE_product_of_successive_numbers_l3491_349151

theorem product_of_successive_numbers :
  let n : ℝ := 51.49757275833493
  let product := n * (n + 1)
  ∃ ε > 0, |product - 2703| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_successive_numbers_l3491_349151


namespace NUMINAMATH_CALUDE_susan_menu_fraction_l3491_349178

theorem susan_menu_fraction (total_dishes : ℕ) (vegan_dishes : ℕ) (vegan_with_nuts : ℕ) : 
  vegan_dishes = total_dishes / 3 →
  vegan_dishes = 6 →
  vegan_with_nuts = 4 →
  (vegan_dishes - vegan_with_nuts : ℚ) / total_dishes = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_susan_menu_fraction_l3491_349178


namespace NUMINAMATH_CALUDE_equal_tuesdays_thursdays_30_days_l3491_349186

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- A function that determines if a given day is a valid starting day for a 30-day month with equal Tuesdays and Thursdays -/
def isValidStartDay (d : DayOfWeek) : Prop := sorry

/-- The number of valid starting days for a 30-day month with equal Tuesdays and Thursdays -/
def numValidStartDays : ℕ := sorry

theorem equal_tuesdays_thursdays_30_days :
  numValidStartDays = 5 := by sorry

end NUMINAMATH_CALUDE_equal_tuesdays_thursdays_30_days_l3491_349186


namespace NUMINAMATH_CALUDE_sam_carrots_l3491_349146

/-- Given that Sandy grew 6 carrots and the total number of carrots grown is 9,
    prove that Sam grew 3 carrots. -/
theorem sam_carrots (sandy_carrots : ℕ) (total_carrots : ℕ) (sam_carrots : ℕ) :
  sandy_carrots = 6 → total_carrots = 9 → sam_carrots = total_carrots - sandy_carrots →
  sam_carrots = 3 := by
  sorry

#check sam_carrots

end NUMINAMATH_CALUDE_sam_carrots_l3491_349146


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l3491_349136

theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) : 
  let base_radius : ℝ := r / 2
  let cone_height : ℝ := Real.sqrt (r^2 - base_radius^2)
  let cone_volume : ℝ := (1/3) * π * base_radius^2 * cone_height
  cone_volume = 9 * π * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l3491_349136


namespace NUMINAMATH_CALUDE_smallest_xy_value_smallest_xy_is_172_min_xy_value_l3491_349150

theorem smallest_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 200) :
  ∀ (a b : ℕ+), 7 * a + 4 * b = 200 → x * y ≤ a * b :=
by sorry

theorem smallest_xy_is_172 :
  ∃ (x y : ℕ+), 7 * x + 4 * y = 200 ∧ x * y = 172 :=
by sorry

theorem min_xy_value (x y : ℕ+) (h : 7 * x + 4 * y = 200) :
  x * y ≥ 172 :=
by sorry

end NUMINAMATH_CALUDE_smallest_xy_value_smallest_xy_is_172_min_xy_value_l3491_349150


namespace NUMINAMATH_CALUDE_pascals_triangle_ratio_l3491_349137

theorem pascals_triangle_ratio (n : ℕ) (r : ℕ) : n = 84 →
  ∃ r, r + 2 ≤ n ∧
    (Nat.choose n r : ℚ) / (Nat.choose n (r + 1)) = 5 / 6 ∧
    (Nat.choose n (r + 1) : ℚ) / (Nat.choose n (r + 2)) = 6 / 7 :=
by
  sorry


end NUMINAMATH_CALUDE_pascals_triangle_ratio_l3491_349137


namespace NUMINAMATH_CALUDE_largest_base_for_twelve_cubed_l3491_349101

/-- Given a natural number n and a base b, returns the sum of digits of n when represented in base b -/
def sumOfDigits (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Returns true if b is the largest base such that the sum of digits of 12^3 in base b is not 3^2 -/
def isLargestBase (b : ℕ) : Prop :=
  (sumOfDigits (12^3) b ≠ 3^2) ∧
  ∀ k > b, sumOfDigits (12^3) k = 3^2

theorem largest_base_for_twelve_cubed :
  isLargestBase 9 := by sorry

end NUMINAMATH_CALUDE_largest_base_for_twelve_cubed_l3491_349101


namespace NUMINAMATH_CALUDE_locus_of_T_l3491_349129

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a rectangle
structure Rectangle where
  m : Point
  k : Point
  t : Point
  p : Point

-- Main theorem
theorem locus_of_T (c : Circle) (m : Point) 
  (h1 : (m.x - c.center.1)^2 + (m.y - c.center.2)^2 < c.radius^2) :
  ∃ (c_locus : Circle),
    c_locus.center = c.center ∧
    c_locus.radius = Real.sqrt (2 * c.radius^2 - (m.x^2 + m.y^2)) ∧
    ∀ (rect : Rectangle),
      (rect.m = m) →
      ((rect.k.x - c.center.1)^2 + (rect.k.y - c.center.2)^2 = c.radius^2) →
      ((rect.p.x - c.center.1)^2 + (rect.p.y - c.center.2)^2 = c.radius^2) →
      (rect.m.x - rect.t.x = rect.k.x - rect.p.x) →
      (rect.m.y - rect.t.y = rect.k.y - rect.p.y) →
      ((rect.t.x - c.center.1)^2 + (rect.t.y - c.center.2)^2 = c_locus.radius^2) :=
by
  sorry


end NUMINAMATH_CALUDE_locus_of_T_l3491_349129


namespace NUMINAMATH_CALUDE_total_loaves_served_l3491_349134

theorem total_loaves_served (wheat_bread : Real) (white_bread : Real)
  (h1 : wheat_bread = 0.2)
  (h2 : white_bread = 0.4) :
  wheat_bread + white_bread = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_total_loaves_served_l3491_349134


namespace NUMINAMATH_CALUDE_min_value_ab_min_value_is_2sqrt2_exists_min_value_l3491_349174

theorem min_value_ab (a b : ℝ) (h : a > 0 ∧ b > 0) (eq : 1/a + 2/b = Real.sqrt (a*b)) :
  ∀ x y : ℝ, x > 0 → y > 0 → 1/x + 2/y = Real.sqrt (x*y) → a*b ≤ x*y :=
by
  sorry

theorem min_value_is_2sqrt2 (a b : ℝ) (h : a > 0 ∧ b > 0) (eq : 1/a + 2/b = Real.sqrt (a*b)) :
  a*b ≥ 2*Real.sqrt 2 :=
by
  sorry

theorem exists_min_value (a b : ℝ) (h : a > 0 ∧ b > 0) (eq : 1/a + 2/b = Real.sqrt (a*b)) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 2/y = Real.sqrt (x*y) ∧ x*y = 2*Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_min_value_ab_min_value_is_2sqrt2_exists_min_value_l3491_349174


namespace NUMINAMATH_CALUDE_points_on_line_l3491_349177

/-- Given a line with points A and B, where A is inside 50 segments and B is inside 56 segments,
    prove that the total number of points marked on the line is 16. -/
theorem points_on_line (n : ℕ) (A B : Set ℕ) : 
  (∃ a₁ a₂ b₁ b₂ : ℕ, 
    a₁ * a₂ = 50 ∧ 
    b₁ * b₂ = 56 ∧ 
    a₁ + a₂ = b₁ + b₂ ∧ 
    n = a₁ + a₂ + 1) →
  n = 16 := by sorry

end NUMINAMATH_CALUDE_points_on_line_l3491_349177


namespace NUMINAMATH_CALUDE_tangent_line_parabola_hyperbola_eccentricity_l3491_349139

/-- Given a line y = kx - 1 tangent to the parabola x² = 8y, 
    the eccentricity of the hyperbola x² - k²y² = 1 is equal to √3 -/
theorem tangent_line_parabola_hyperbola_eccentricity :
  ∀ k : ℝ,
  (∃ x y : ℝ, y = k * x - 1 ∧ x^2 = 8 * y ∧ 
   ∀ x' y' : ℝ, y' = k * x' - 1 → x'^2 ≠ 8 * y' ∨ (x' = x ∧ y' = y)) →
  Real.sqrt 3 = (Real.sqrt (1 + (1 / k^2))) / 1 :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_parabola_hyperbola_eccentricity_l3491_349139


namespace NUMINAMATH_CALUDE_cindys_calculation_l3491_349131

theorem cindys_calculation (x : ℝ) : (x - 7) / 5 = 57 → (x - 5) / 7 = 41 := by
  sorry

end NUMINAMATH_CALUDE_cindys_calculation_l3491_349131


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l3491_349167

theorem quadratic_equation_root (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x - 5 = 0 ∧ x = 1) → 
  (∃ y : ℝ, 3 * y^2 + m * y - 5 = 0 ∧ y = -5/3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l3491_349167


namespace NUMINAMATH_CALUDE_debbys_water_consumption_l3491_349116

/-- Given Debby's beverage consumption pattern, prove the number of water bottles she drank per day. -/
theorem debbys_water_consumption 
  (total_soda : ℕ) 
  (total_water : ℕ) 
  (soda_per_day : ℕ) 
  (soda_days : ℕ) 
  (water_days : ℕ) 
  (h1 : total_soda = 360)
  (h2 : total_water = 162)
  (h3 : soda_per_day = 9)
  (h4 : soda_days = 40)
  (h5 : water_days = 30)
  (h6 : total_soda = soda_per_day * soda_days) :
  (total_water : ℚ) / water_days = 5.4 := by
  sorry

end NUMINAMATH_CALUDE_debbys_water_consumption_l3491_349116


namespace NUMINAMATH_CALUDE_g_x_minus_3_l3491_349180

/-- The function g(x) = x^2 -/
def g (x : ℝ) : ℝ := x^2

/-- Theorem: For the function g(x) = x^2, g(x-3) = x^2 - 6x + 9 -/
theorem g_x_minus_3 (x : ℝ) : g (x - 3) = x^2 - 6*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_g_x_minus_3_l3491_349180


namespace NUMINAMATH_CALUDE_distance_from_origin_to_point_l3491_349183

theorem distance_from_origin_to_point : ∀ (x y : ℝ), 
  x = 12 ∧ y = 9 → Real.sqrt (x^2 + y^2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_origin_to_point_l3491_349183


namespace NUMINAMATH_CALUDE_alligator_population_after_one_year_l3491_349148

/-- The number of alligators after a given number of doubling periods -/
def alligator_population (initial_population : ℕ) (doubling_periods : ℕ) : ℕ :=
  initial_population * 2^doubling_periods

/-- Theorem: After one year (two doubling periods), 
    the alligator population will be 16 given an initial population of 4 -/
theorem alligator_population_after_one_year :
  alligator_population 4 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_alligator_population_after_one_year_l3491_349148


namespace NUMINAMATH_CALUDE_square_to_octagon_triangle_to_icosagon_l3491_349169

-- Define a square
structure Square :=
  (side : ℝ)
  (side_positive : side > 0)

-- Define a triangle
structure Triangle :=
  (side : ℝ)
  (side_positive : side > 0)

-- Define an octagon
structure Octagon :=
  (side : ℝ)
  (side_positive : side > 0)

-- Define a 20-sided polygon (icosagon)
structure Icosagon :=
  (side : ℝ)
  (side_positive : side > 0)

-- Function to cut a square into two parts
def cut_square (s : Square) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Function to form an octagon from two parts
def form_octagon (parts : (ℝ × ℝ) × (ℝ × ℝ)) : Octagon := sorry

-- Function to cut a triangle into two parts
def cut_triangle (t : Triangle) : (ℝ × ℝ) × (ℝ × ℝ) := sorry

-- Function to form an icosagon from two parts
def form_icosagon (parts : (ℝ × ℝ) × (ℝ × ℝ)) : Icosagon := sorry

-- Theorem stating that a square can be cut into two parts to form an octagon
theorem square_to_octagon (s : Square) :
  ∃ (o : Octagon), form_octagon (cut_square s) = o := sorry

-- Theorem stating that a triangle can be cut into two parts to form an icosagon
theorem triangle_to_icosagon (t : Triangle) :
  ∃ (i : Icosagon), form_icosagon (cut_triangle t) = i := sorry

end NUMINAMATH_CALUDE_square_to_octagon_triangle_to_icosagon_l3491_349169


namespace NUMINAMATH_CALUDE_exactly_two_solutions_l3491_349147

/-- The number of solutions to the system of equations -/
def num_solutions : ℕ := 2

/-- A solution to the system of equations is a triple of positive integers (x, y, z) -/
def is_solution (x y z : ℕ+) : Prop :=
  x * y + x * z = 255 ∧ x * z - y * z = 224

/-- The theorem stating that there are exactly two solutions -/
theorem exactly_two_solutions :
  (∃! (s : Finset (ℕ+ × ℕ+ × ℕ+)), s.card = num_solutions ∧ 
    ∀ (x y z : ℕ+), (x, y, z) ∈ s ↔ is_solution x y z) :=
sorry

end NUMINAMATH_CALUDE_exactly_two_solutions_l3491_349147


namespace NUMINAMATH_CALUDE_line_segment_parameter_sum_of_squares_l3491_349107

/-- A line segment parameterized by t, connecting two points in 2D space. -/
structure LineSegment where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The point on the line segment at a given parameter t. -/
def LineSegment.point_at (l : LineSegment) (t : ℝ) : ℝ × ℝ :=
  (l.a * t + l.b, l.c * t + l.d)

theorem line_segment_parameter_sum_of_squares :
  ∀ l : LineSegment,
  (l.point_at 0 = (-3, 5)) →
  (l.point_at 0.5 = (0.5, 7.5)) →
  (l.point_at 1 = (4, 10)) →
  l.a^2 + l.b^2 + l.c^2 + l.d^2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_parameter_sum_of_squares_l3491_349107


namespace NUMINAMATH_CALUDE_car_price_difference_l3491_349162

/-- Proves the difference in price between the old and new car -/
theorem car_price_difference
  (sale_percentage : ℝ)
  (additional_amount : ℝ)
  (new_car_price : ℝ)
  (h1 : sale_percentage = 0.8)
  (h2 : additional_amount = 4000)
  (h3 : new_car_price = 30000)
  (h4 : sale_percentage * (new_car_price - additional_amount) + additional_amount = new_car_price) :
  (new_car_price - additional_amount) / sale_percentage - new_car_price = 2500 := by
sorry

end NUMINAMATH_CALUDE_car_price_difference_l3491_349162


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l3491_349135

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 2210 → n + (n + 1) = -95 := by sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l3491_349135


namespace NUMINAMATH_CALUDE_perpendicular_line_through_B_l3491_349130

-- Define the given line
def given_line (x y : ℝ) : Prop := 2 * x + y - 5 = 0

-- Define the point B
def point_B : ℝ × ℝ := (3, 0)

-- Define the perpendicular line
def perpendicular_line (x y : ℝ) : Prop := x - 2 * y + 3 = 0

-- Theorem statement
theorem perpendicular_line_through_B :
  (perpendicular_line point_B.1 point_B.2) ∧
  (∀ (x y : ℝ), perpendicular_line x y → given_line x y → 
    (x - point_B.1) * (x - point_B.1) + (y - point_B.2) * (y - point_B.2) ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_B_l3491_349130


namespace NUMINAMATH_CALUDE_actual_time_greater_than_planned_l3491_349163

/-- Proves that the actual running time is greater than the planned time under given conditions -/
theorem actual_time_greater_than_planned (a V : ℝ) (h1 : a > 0) (h2 : V > 0) : 
  (a / (1.25 * V) / 2 + a / (0.8 * V) / 2) > a / V := by
  sorry

#check actual_time_greater_than_planned

end NUMINAMATH_CALUDE_actual_time_greater_than_planned_l3491_349163


namespace NUMINAMATH_CALUDE_power_congruence_l3491_349109

theorem power_congruence (h : 2^200 ≡ 1 [MOD 800]) : 2^6000 ≡ 1 [MOD 800] := by
  sorry

end NUMINAMATH_CALUDE_power_congruence_l3491_349109


namespace NUMINAMATH_CALUDE_phone_service_cost_per_minute_l3491_349144

/-- Calculates the cost per minute for a phone service given the total bill, monthly fee, and minutes used. -/
def cost_per_minute (total_bill monthly_fee : ℚ) (minutes_used : ℕ) : ℚ :=
  (total_bill - monthly_fee) / minutes_used

/-- Theorem stating that given the specific conditions, the cost per minute is $0.12. -/
theorem phone_service_cost_per_minute :
  let total_bill : ℚ := 23.36
  let monthly_fee : ℚ := 2
  let minutes_used : ℕ := 178
  cost_per_minute total_bill monthly_fee minutes_used = 0.12 := by
  sorry

end NUMINAMATH_CALUDE_phone_service_cost_per_minute_l3491_349144


namespace NUMINAMATH_CALUDE_thirteen_to_six_mod_eight_l3491_349127

theorem thirteen_to_six_mod_eight (m : ℕ) : 
  13^6 ≡ m [ZMOD 8] → 0 ≤ m → m < 8 → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_thirteen_to_six_mod_eight_l3491_349127


namespace NUMINAMATH_CALUDE_reciprocal_counterexample_l3491_349123

theorem reciprocal_counterexample : ∃ (a b : ℚ), a ≠ 0 ∧ b ≠ 0 ∧ a > b ∧ a⁻¹ ≥ b⁻¹ := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_counterexample_l3491_349123


namespace NUMINAMATH_CALUDE_shortest_path_on_right_angle_polyhedron_l3491_349112

/-- A polyhedron with all dihedral angles as right angles -/
structure RightAnglePolyhedron where
  -- We don't need to define the full structure, just what we need for the theorem
  edge_length : ℝ
  all_dihedral_angles_right : True  -- placeholder for the condition

/-- The shortest path between two vertices on the surface of the polyhedron -/
def shortest_surface_path (p : RightAnglePolyhedron) (X Y : ℝ × ℝ × ℝ) : ℝ :=
  sorry  -- The actual implementation would depend on how we represent the polyhedron

theorem shortest_path_on_right_angle_polyhedron 
  (p : RightAnglePolyhedron) 
  (X Y : ℝ × ℝ × ℝ) 
  (h_adjacent : True)  -- placeholder for the condition that X and Y are on adjacent faces
  (h_diagonal : True)  -- placeholder for the condition that X and Y are diagonally opposite
  (h_unit_edge : p.edge_length = 1) :
  shortest_surface_path p X Y = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_shortest_path_on_right_angle_polyhedron_l3491_349112


namespace NUMINAMATH_CALUDE_max_n_geometric_sequence_l3491_349181

-- Define the geometric sequence
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n-1)

theorem max_n_geometric_sequence (a : ℕ → ℝ) :
  (∀ n, a n > 0) →  -- All terms are positive
  a 2 * a 4 = 4 →  -- a₂ · a₄ = 4
  a 1 + a 2 + a 3 = 14 →  -- a₁ + a₂ + a₃ = 14
  (∃ a₁ q, ∀ n, a n = geometric_sequence a₁ q n) →  -- {a_n} is a geometric sequence
  (∀ n > 4, a n * a (n+1) * a (n+2) ≤ 1/9) ∧  -- For all n > 4, the product is ≤ 1/9
  (a 4 * a 5 * a 6 > 1/9) :=  -- For n = 4, the product is > 1/9
by sorry

end NUMINAMATH_CALUDE_max_n_geometric_sequence_l3491_349181


namespace NUMINAMATH_CALUDE_two_numbers_sum_product_l3491_349156

theorem two_numbers_sum_product (n : ℕ) (hn : n = 40) :
  ∃ a b : ℕ,
    1 ≤ a ∧ a < b ∧ b ≤ n ∧
    (n * (n + 1)) / 2 - a - b = a * b + 2 ∧
    b - a = 50 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_product_l3491_349156


namespace NUMINAMATH_CALUDE_number_wall_x_value_l3491_349141

/-- Represents a simplified number wall with given conditions --/
structure NumberWall where
  x : ℤ
  y : ℤ
  -- Define the wall structure based on given conditions
  bottom_row : Vector ℤ 5 := ⟨[x, 7, y, 14, 9], rfl⟩
  second_row_right : Vector ℤ 2 := ⟨[y + 14, 23], rfl⟩
  third_row_right : ℤ := 37
  top : ℤ := 80

/-- The main theorem stating that x must be 12 in the given number wall --/
theorem number_wall_x_value (wall : NumberWall) : wall.x = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_wall_x_value_l3491_349141


namespace NUMINAMATH_CALUDE_valid_colorings_2x9_board_l3491_349196

/-- Represents the number of columns in the board -/
def n : ℕ := 9

/-- Represents the number of colors available -/
def num_colors : ℕ := 3

/-- Represents the number of ways to color the first column -/
def first_column_colorings : ℕ := num_colors * (num_colors - 1)

/-- Represents the number of ways to color each subsequent column -/
def subsequent_column_colorings : ℕ := num_colors - 1

/-- Theorem stating the number of valid colorings for a 2 × 9 board -/
theorem valid_colorings_2x9_board :
  first_column_colorings * subsequent_column_colorings^(n - 1) = 39366 := by
  sorry

end NUMINAMATH_CALUDE_valid_colorings_2x9_board_l3491_349196


namespace NUMINAMATH_CALUDE_sum_of_solutions_eq_neg_three_fourths_l3491_349143

theorem sum_of_solutions_eq_neg_three_fourths :
  let f : ℝ → ℝ := λ x => 243^(x + 1) - 81^(x^2 + 2*x)
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = -3/4 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_eq_neg_three_fourths_l3491_349143


namespace NUMINAMATH_CALUDE_parabola_line_intersection_properties_l3491_349173

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Theorem about properties of intersections between a line through the focus and a parabola -/
theorem parabola_line_intersection_properties (par : Parabola) 
  (A B : ParabolaPoint) (h_on_parabola : A.y^2 = 2*par.p*A.x ∧ B.y^2 = 2*par.p*B.x) 
  (h_through_focus : ∃ (k : ℝ), A.y = k*(A.x - par.p/2) ∧ B.y = k*(B.x - par.p/2)) :
  A.x * B.x = (par.p^2)/4 ∧ 
  1/(A.x + par.p/2) + 1/(B.x + par.p/2) = 2/par.p := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_properties_l3491_349173


namespace NUMINAMATH_CALUDE_triangle_transformation_exists_l3491_349190

-- Define a point in the 2D plane
structure Point :=
  (x : Int) (y : Int)

-- Define a triangle as a set of three points
structure Triangle :=
  (a : Point) (b : Point) (c : Point)

-- Define the 90° counterclockwise rotation transformation
def rotate90 (center : Point) (p : Point) : Point :=
  let dx := p.x - center.x
  let dy := p.y - center.y
  Point.mk (center.x - dy) (center.y + dx)

-- Define the initial and target triangles
def initialTriangle : Triangle :=
  Triangle.mk (Point.mk 0 0) (Point.mk 1 0) (Point.mk 0 1)

def targetTriangle : Triangle :=
  Triangle.mk (Point.mk 0 0) (Point.mk 1 0) (Point.mk 1 1)

-- Theorem statement
theorem triangle_transformation_exists :
  ∃ (rotationCenter : Point),
    rotate90 rotationCenter initialTriangle.a = targetTriangle.a ∧
    rotate90 rotationCenter initialTriangle.b = targetTriangle.b ∧
    rotate90 rotationCenter initialTriangle.c = targetTriangle.c :=
by sorry

end NUMINAMATH_CALUDE_triangle_transformation_exists_l3491_349190


namespace NUMINAMATH_CALUDE_katies_flour_amount_l3491_349120

theorem katies_flour_amount (katie_flour : ℕ) (sheila_flour : ℕ) : 
  sheila_flour = katie_flour + 2 →
  katie_flour + sheila_flour = 8 →
  katie_flour = 3 := by
sorry

end NUMINAMATH_CALUDE_katies_flour_amount_l3491_349120


namespace NUMINAMATH_CALUDE_faye_candy_count_l3491_349152

/-- Calculates the final candy count for Faye after eating some and receiving more. -/
def final_candy_count (initial : ℕ) (eaten : ℕ) (received : ℕ) : ℕ :=
  initial - eaten + received

/-- Proves that Faye's final candy count is 62 pieces. -/
theorem faye_candy_count :
  final_candy_count 47 25 40 = 62 := by
  sorry

end NUMINAMATH_CALUDE_faye_candy_count_l3491_349152


namespace NUMINAMATH_CALUDE_expression_value_l3491_349155

theorem expression_value (x : ℝ) : 
  let a := 2005 * x + 2009
  let b := 2005 * x + 2010
  let c := 2005 * x + 2011
  a^2 + b^2 + c^2 - a*b - b*c - c*a = 3 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l3491_349155


namespace NUMINAMATH_CALUDE_egyptian_art_pieces_l3491_349119

theorem egyptian_art_pieces (total : ℕ) (asian : ℕ) (egyptian : ℕ) : 
  total = 992 → asian = 465 → egyptian = total - asian → egyptian = 527 := by
sorry

end NUMINAMATH_CALUDE_egyptian_art_pieces_l3491_349119


namespace NUMINAMATH_CALUDE_number_of_bowls_l3491_349195

theorem number_of_bowls (n : ℕ) : n > 0 → (96 : ℝ) / n = 6 → n = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_of_bowls_l3491_349195


namespace NUMINAMATH_CALUDE_actual_distance_calculation_l3491_349104

/-- Given a map distance and scale, calculate the actual distance between two towns. -/
theorem actual_distance_calculation (map_distance : ℝ) (scale_distance : ℝ) (scale_miles : ℝ) : 
  map_distance = 20 → scale_distance = 0.5 → scale_miles = 10 → 
  (map_distance * scale_miles / scale_distance) = 400 := by
sorry

end NUMINAMATH_CALUDE_actual_distance_calculation_l3491_349104


namespace NUMINAMATH_CALUDE_largest_number_in_set_l3491_349158

theorem largest_number_in_set (a : ℝ) (h : a = -3) :
  -3 * a = max (-3 * a) (max (5 * a) (max (24 / a) (max (a ^ 2) 1))) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_in_set_l3491_349158


namespace NUMINAMATH_CALUDE_movie_theater_child_price_l3491_349157

/-- Proves that the price for children is $4.5 given the conditions of the movie theater problem -/
theorem movie_theater_child_price 
  (adult_price : ℝ) 
  (num_children : ℕ) 
  (child_adult_diff : ℕ) 
  (total_receipts : ℝ) 
  (h1 : adult_price = 6.75)
  (h2 : num_children = 48)
  (h3 : child_adult_diff = 20)
  (h4 : total_receipts = 405) :
  ∃ (child_price : ℝ), 
    child_price = 4.5 ∧ 
    (num_children : ℝ) * child_price + ((num_children : ℝ) - (child_adult_diff : ℝ)) * adult_price = total_receipts :=
by
  sorry

end NUMINAMATH_CALUDE_movie_theater_child_price_l3491_349157


namespace NUMINAMATH_CALUDE_compare_fractions_l3491_349132

theorem compare_fractions : -3/4 > -|-(4/5)| := by
  sorry

end NUMINAMATH_CALUDE_compare_fractions_l3491_349132


namespace NUMINAMATH_CALUDE_smallest_staircase_steps_l3491_349103

theorem smallest_staircase_steps (n : ℕ) : 
  n > 20 ∧ 
  n % 6 = 4 ∧ 
  n % 7 = 3 ∧ 
  (∀ m : ℕ, m > 20 ∧ m % 6 = 4 ∧ m % 7 = 3 → m ≥ n) → 
  n = 52 := by
sorry

end NUMINAMATH_CALUDE_smallest_staircase_steps_l3491_349103


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l3491_349166

/-- A geometric sequence with common ratio q > 1 -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q > 1 ∧ ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q →
  (4 * (a 2005)^2 - 8 * (a 2005) + 3 = 0) →
  (4 * (a 2006)^2 - 8 * (a 2006) + 3 = 0) →
  a 2007 + a 2008 = 18 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l3491_349166


namespace NUMINAMATH_CALUDE_spring_length_formula_l3491_349159

/-- Spring scale properties -/
structure SpringScale where
  initialLength : ℝ
  extensionRate : ℝ

/-- The analytical expression for the total length of a spring -/
def totalLength (s : SpringScale) (mass : ℝ) : ℝ :=
  s.initialLength + s.extensionRate * mass

/-- Theorem: The analytical expression for the total length of the spring is y = 10 + 2x -/
theorem spring_length_formula (s : SpringScale) (mass : ℝ) :
  s.initialLength = 10 ∧ s.extensionRate = 2 →
  totalLength s mass = 10 + 2 * mass := by
  sorry

end NUMINAMATH_CALUDE_spring_length_formula_l3491_349159


namespace NUMINAMATH_CALUDE_subset_with_unique_sum_representation_l3491_349165

theorem subset_with_unique_sum_representation :
  ∃ (X : Set ℤ), ∀ (n : ℤ), ∃! (p : ℤ × ℤ), p.1 ∈ X ∧ p.2 ∈ X ∧ p.1 + 2 * p.2 = n :=
sorry

end NUMINAMATH_CALUDE_subset_with_unique_sum_representation_l3491_349165


namespace NUMINAMATH_CALUDE_total_tickets_used_l3491_349164

/-- The cost of the shooting game in tickets -/
def shooting_game_cost : ℕ := 5

/-- The cost of the carousel in tickets -/
def carousel_cost : ℕ := 3

/-- The number of times Jen played the shooting game -/
def jen_games : ℕ := 2

/-- The number of times Russel rode the carousel -/
def russel_rides : ℕ := 3

/-- Theorem stating the total number of tickets used -/
theorem total_tickets_used : 
  shooting_game_cost * jen_games + carousel_cost * russel_rides = 19 := by
  sorry

end NUMINAMATH_CALUDE_total_tickets_used_l3491_349164


namespace NUMINAMATH_CALUDE_extra_cat_food_l3491_349124

theorem extra_cat_food (food_one_cat food_two_cats : ℝ)
  (h1 : food_one_cat = 0.5)
  (h2 : food_two_cats = 0.9) :
  food_two_cats - food_one_cat = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_extra_cat_food_l3491_349124


namespace NUMINAMATH_CALUDE_shaded_area_value_l3491_349117

theorem shaded_area_value (d : ℝ) : 
  (3 * (2 - (1/2 * π * 1^2))) = 6 + d * π → d = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_value_l3491_349117


namespace NUMINAMATH_CALUDE_function_value_at_negative_a_l3491_349179

/-- Given a function f(x) = ax³ + bx, prove that if f(a) = 8, then f(-a) = -8 -/
theorem function_value_at_negative_a (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x^3 + b * x
  f a = 8 → f (-a) = -8 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_negative_a_l3491_349179


namespace NUMINAMATH_CALUDE_max_min_m_values_l3491_349100

/-- Given conditions p and q, find the maximum and minimum values of m -/
theorem max_min_m_values (m : ℝ) (h_m_pos : m > 0) : 
  (∀ x : ℝ, |x| ≤ m → -1 ≤ x ∧ x ≤ 4) ∧ 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 4 → |x| ≤ m) → 
  m = 4 := by sorry

end NUMINAMATH_CALUDE_max_min_m_values_l3491_349100


namespace NUMINAMATH_CALUDE_expo_min_rental_fee_l3491_349111

/-- Represents a bus type with its seat capacity and rental fee -/
structure BusType where
  seats : ℕ
  fee : ℕ

/-- Calculates the minimum rental fee for transporting people using two types of buses -/
def minRentalFee (people : ℕ) (typeA typeB : BusType) : ℕ :=
  sorry

/-- Theorem stating the minimum rental fee for the given problem -/
theorem expo_min_rental_fee :
  let typeA : BusType := ⟨40, 400⟩
  let typeB : BusType := ⟨50, 480⟩
  minRentalFee 360 typeA typeB = 3520 := by
  sorry

end NUMINAMATH_CALUDE_expo_min_rental_fee_l3491_349111


namespace NUMINAMATH_CALUDE_ellipse_equation_through_points_l3491_349105

/-- The standard equation of an ellipse passing through (-3, 0) and (0, -2) -/
theorem ellipse_equation_through_points :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  (∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ↔ 
    (x^2 / 9 + y^2 / 4 = 1)) ∧
  (-3^2 / a^2 + 0^2 / b^2 = 1) ∧
  (0^2 / a^2 + (-2)^2 / b^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_through_points_l3491_349105


namespace NUMINAMATH_CALUDE_simplify_expression_l3491_349138

theorem simplify_expression (a : ℝ) : a + 1 + a - 2 + a + 3 + a - 4 = 4*a - 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3491_349138


namespace NUMINAMATH_CALUDE_fraction_equality_l3491_349133

theorem fraction_equality (a b : ℝ) (h : b ≠ 0) : (3 * a) / (3 * b) = a / b := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3491_349133


namespace NUMINAMATH_CALUDE_polynomial_simplification_l3491_349188

theorem polynomial_simplification (y : ℝ) :
  (3 * y - 2) * (5 * y^12 + 3 * y^11 + 5 * y^9 + 3 * y^8) =
  15 * y^13 - y^12 + 3 * y^11 + 15 * y^10 - y^9 - 6 * y^8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l3491_349188


namespace NUMINAMATH_CALUDE_chess_team_boys_count_l3491_349153

theorem chess_team_boys_count (total_members : ℕ) (meeting_attendees : ℕ) : 
  total_members = 30 →
  meeting_attendees = 20 →
  ∃ (girls : ℕ) (boys : ℕ),
    girls + boys = total_members ∧
    (2 * girls / 3 : ℚ) + boys = meeting_attendees ∧
    boys = 0 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_boys_count_l3491_349153


namespace NUMINAMATH_CALUDE_average_of_xyz_is_one_l3491_349191

theorem average_of_xyz_is_one (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0)
  (h_diff : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (h_prod : x * y * z = 1)
  (h_sum : x + y + z = 1/x + 1/y + 1/z) :
  (x + y + z) / 3 = 1 := by
sorry

end NUMINAMATH_CALUDE_average_of_xyz_is_one_l3491_349191


namespace NUMINAMATH_CALUDE_medication_price_reduction_l3491_349187

theorem medication_price_reduction (a : ℝ) :
  let new_price := a
  let reduction_rate := 0.4
  let original_price := (5 / 3) * a
  (1 - reduction_rate) * original_price = new_price :=
by sorry

end NUMINAMATH_CALUDE_medication_price_reduction_l3491_349187


namespace NUMINAMATH_CALUDE_theatre_seating_l3491_349175

theorem theatre_seating (total_seats : ℕ) (row_size : ℕ) (expected_attendance : ℕ) : 
  total_seats = 225 → 
  row_size = 15 → 
  expected_attendance = 160 → 
  (total_seats - (((expected_attendance + row_size - 1) / row_size) * row_size)) = 60 :=
by sorry

end NUMINAMATH_CALUDE_theatre_seating_l3491_349175


namespace NUMINAMATH_CALUDE_subset_of_sqrt_two_in_sqrt_three_set_l3491_349149

theorem subset_of_sqrt_two_in_sqrt_three_set :
  {Real.sqrt 2} ⊆ {x : ℝ | x ≤ Real.sqrt 3} := by sorry

end NUMINAMATH_CALUDE_subset_of_sqrt_two_in_sqrt_three_set_l3491_349149
