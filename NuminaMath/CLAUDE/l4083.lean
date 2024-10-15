import Mathlib

namespace NUMINAMATH_CALUDE_mame_probability_theorem_l4083_408318

/-- Represents a piece of paper with 8 possible surfaces (4 on each side) -/
structure Paper :=
  (surfaces : Fin 8)

/-- The probability of a specific surface being on top -/
def probability_on_top (paper : Paper) : ℚ := 1 / 8

/-- The surface with "MAME" written on it -/
def mame_surface : Fin 8 := 0

theorem mame_probability_theorem :
  probability_on_top { surfaces := mame_surface } = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_mame_probability_theorem_l4083_408318


namespace NUMINAMATH_CALUDE_gcd_cube_plus_27_and_plus_3_l4083_408333

theorem gcd_cube_plus_27_and_plus_3 (n : ℕ) (h : n > 27) :
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_cube_plus_27_and_plus_3_l4083_408333


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l4083_408368

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 2*x - 8 = 0 ↔ (x - 1)^2 = 9 := by
sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l4083_408368


namespace NUMINAMATH_CALUDE_fair_cost_calculation_l4083_408317

/-- Calculate the total cost for Joe and the twins at the fair -/
theorem fair_cost_calculation (entrance_fee_under_18 : ℚ) 
  (entrance_fee_over_18_multiplier : ℚ) (group_discount : ℚ) 
  (low_thrill_under_18 : ℚ) (low_thrill_over_18 : ℚ)
  (medium_thrill_under_18 : ℚ) (medium_thrill_over_18 : ℚ)
  (high_thrill_under_18 : ℚ) (high_thrill_over_18 : ℚ)
  (joe_age : ℕ) (twin_age : ℕ)
  (joe_low : ℕ) (joe_medium : ℕ) (joe_high : ℕ)
  (twin_a_low : ℕ) (twin_a_medium : ℕ)
  (twin_b_low : ℕ) (twin_b_high : ℕ) :
  entrance_fee_under_18 = 5 →
  entrance_fee_over_18_multiplier = 1.2 →
  group_discount = 0.85 →
  low_thrill_under_18 = 0.5 →
  low_thrill_over_18 = 0.7 →
  medium_thrill_under_18 = 1 →
  medium_thrill_over_18 = 1.2 →
  high_thrill_under_18 = 1.5 →
  high_thrill_over_18 = 1.7 →
  joe_age = 30 →
  twin_age = 6 →
  joe_low = 2 →
  joe_medium = 1 →
  joe_high = 1 →
  twin_a_low = 2 →
  twin_a_medium = 1 →
  twin_b_low = 3 →
  twin_b_high = 2 →
  (entrance_fee_under_18 * entrance_fee_over_18_multiplier * group_discount + 
   2 * entrance_fee_under_18 * group_discount +
   joe_low * low_thrill_over_18 + joe_medium * medium_thrill_over_18 + 
   joe_high * high_thrill_over_18 +
   twin_a_low * low_thrill_under_18 + twin_a_medium * medium_thrill_under_18 +
   twin_b_low * low_thrill_under_18 + twin_b_high * high_thrill_under_18) = 24.4 := by
  sorry

end NUMINAMATH_CALUDE_fair_cost_calculation_l4083_408317


namespace NUMINAMATH_CALUDE_hostel_provision_days_l4083_408356

/-- Calculates the initial number of days provisions were planned for in a hostel. -/
def initial_provision_days (initial_men : ℕ) (men_left : ℕ) (days_after_leaving : ℕ) : ℕ :=
  ((initial_men - men_left) * days_after_leaving) / initial_men

/-- Theorem stating that given the conditions, the initial provision days is 32. -/
theorem hostel_provision_days :
  initial_provision_days 250 50 40 = 32 := by
  sorry

#eval initial_provision_days 250 50 40

end NUMINAMATH_CALUDE_hostel_provision_days_l4083_408356


namespace NUMINAMATH_CALUDE_prob_at_least_one_red_l4083_408328

/-- Represents a box containing red and white balls -/
structure Box where
  red_balls : ℕ
  white_balls : ℕ

/-- Calculates the probability of drawing a specific color ball from a box -/
def prob_draw (b : Box) (color : String) : ℚ :=
  if color = "red" then
    b.red_balls / (b.red_balls + b.white_balls)
  else if color = "white" then
    b.white_balls / (b.red_balls + b.white_balls)
  else
    0

/-- Theorem: The probability of drawing at least one red ball from two boxes,
    each containing 2 red balls and 1 white ball, is equal to 8/9 -/
theorem prob_at_least_one_red (box_a box_b : Box) 
  (ha : box_a.red_balls = 2 ∧ box_a.white_balls = 1)
  (hb : box_b.red_balls = 2 ∧ box_b.white_balls = 1) : 
  1 - (prob_draw box_a "white" * prob_draw box_b "white") = 8/9 :=
sorry

end NUMINAMATH_CALUDE_prob_at_least_one_red_l4083_408328


namespace NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l4083_408366

theorem sum_of_solutions_quadratic (x : ℝ) : 
  (x^2 - 6*x + 5 = 2*x - 11) → (x + x = 8) :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_solutions_quadratic_l4083_408366


namespace NUMINAMATH_CALUDE_tunnel_length_l4083_408311

/-- The length of a tunnel given train parameters -/
theorem tunnel_length (train_length : ℝ) (time_diff : ℝ) (train_speed : ℝ) :
  train_length = 2 →
  time_diff = 4 →
  train_speed = 30 →
  train_length = train_speed * time_diff / 60 := by
  sorry

#check tunnel_length

end NUMINAMATH_CALUDE_tunnel_length_l4083_408311


namespace NUMINAMATH_CALUDE_circle_trajectory_and_intersection_line_l4083_408300

-- Define the circles E and F
def circle_E (x y : ℝ) : Prop := (x + Real.sqrt 3)^2 + y^2 = 25
def circle_F (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 - y^2 = 1

-- Define the curve C (trajectory of center of circle P)
def curve_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x + 4*y - 5 = 0

-- Define the point M
def point_M : ℝ × ℝ := (1, 1)

-- Theorem statement
theorem circle_trajectory_and_intersection_line :
  ∀ (x_A y_A x_B y_B : ℝ),
  -- Circle P is internally tangent to both E and F
  (∃ (r : ℝ), r > 0 ∧ 
    (∀ (x_P y_P : ℝ), (x_P - x_A)^2 + (y_P - y_A)^2 = r^2 →
      (∃ (x_E y_E : ℝ), circle_E x_E y_E ∧ (x_P - x_E)^2 + (y_P - y_E)^2 = (5 - r)^2) ∧
      (∃ (x_F y_F : ℝ), circle_F x_F y_F ∧ (x_P - x_F)^2 + (y_P - y_F)^2 = (r - 1)^2))) →
  -- A and B are on curve C
  curve_C x_A y_A →
  curve_C x_B y_B →
  -- M is the midpoint of AB
  point_M = ((x_A + x_B)/2, (y_A + y_B)/2) →
  -- A, B are on line l
  line_l x_A y_A →
  line_l x_B y_B →
  -- The equation of curve C is correct
  (∀ (x y : ℝ), curve_C x y ↔ x^2/4 + y^2 = 1) ∧
  -- The equation of line l is correct
  (∀ (x y : ℝ), line_l x y ↔ x + 4*y - 5 = 0) :=
by sorry


end NUMINAMATH_CALUDE_circle_trajectory_and_intersection_line_l4083_408300


namespace NUMINAMATH_CALUDE_total_distance_is_151_l4083_408352

/-- Calculates the total distance Amy biked in a week -/
def total_distance_biked : ℝ :=
  let monday_distance : ℝ := 12
  let tuesday_distance : ℝ := 2 * monday_distance - 3
  let wednesday_distance : ℝ := 2 * 11
  let thursday_distance : ℝ := wednesday_distance + 2
  let friday_distance : ℝ := thursday_distance + 2
  let saturday_distance : ℝ := friday_distance + 2
  let sunday_distance : ℝ := 3 * 6
  monday_distance + tuesday_distance + wednesday_distance + thursday_distance + 
  friday_distance + saturday_distance + sunday_distance

theorem total_distance_is_151 : total_distance_biked = 151 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_is_151_l4083_408352


namespace NUMINAMATH_CALUDE_smallest_non_even_units_digit_l4083_408359

def EvenUnitsDigits : Set Nat := {0, 2, 4, 6, 8}

theorem smallest_non_even_units_digit : 
  (∀ d : Nat, d < 10 → d ∉ EvenUnitsDigits → 1 ≤ d) ∧ 1 ∉ EvenUnitsDigits := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_even_units_digit_l4083_408359


namespace NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l4083_408392

def N : Matrix (Fin 4) (Fin 4) ℝ := !![3, -1, 8, 1; 4, 6, -2, 0; -9, -3, 5, 7; 1, 2, 0, -1]

def i : Fin 4 → ℝ := ![1, 0, 0, 0]
def j : Fin 4 → ℝ := ![0, 1, 0, 0]
def k : Fin 4 → ℝ := ![0, 0, 1, 0]
def l : Fin 4 → ℝ := ![0, 0, 0, 1]

theorem matrix_N_satisfies_conditions :
  N.mulVec i = ![3, 4, -9, 1] ∧
  N.mulVec j = ![-1, 6, -3, 2] ∧
  N.mulVec k = ![8, -2, 5, 0] ∧
  N.mulVec l = ![1, 0, 7, -1] := by
  sorry

end NUMINAMATH_CALUDE_matrix_N_satisfies_conditions_l4083_408392


namespace NUMINAMATH_CALUDE_max_intersection_faces_l4083_408394

def W : Set (Fin 4 → ℝ) := {x | ∀ i, 0 ≤ x i ∧ x i ≤ 1}

def isParallelHyperplane (h : ℝ → ℝ → ℝ → ℝ → Prop) : Prop :=
  ∃ (k : ℝ), ∀ x₁ x₂ x₃ x₄, h x₁ x₂ x₃ x₄ ↔ x₁ + x₂ + x₃ + x₄ = k

def intersectionFaces (h : ℝ → ℝ → ℝ → ℝ → Prop) : ℕ :=
  sorry

theorem max_intersection_faces :
  ∀ h, isParallelHyperplane h →
    (∃ x ∈ W, h (x 0) (x 1) (x 2) (x 3)) →
    intersectionFaces h ≤ 8 ∧
    (∃ h', isParallelHyperplane h' ∧
      (∃ x ∈ W, h' (x 0) (x 1) (x 2) (x 3)) ∧
      intersectionFaces h' = 8) :=
by sorry

end NUMINAMATH_CALUDE_max_intersection_faces_l4083_408394


namespace NUMINAMATH_CALUDE_ellipse_focus_k_value_l4083_408354

/-- An ellipse with equation 5x^2 - ky^2 = 5 and one focus at (0, 2) has k = -1 -/
theorem ellipse_focus_k_value (k : ℝ) :
  (∀ x y : ℝ, 5 * x^2 - k * y^2 = 5) →  -- Ellipse equation
  (∃ x : ℝ, (x, 2) ∈ {p : ℝ × ℝ | 5 * p.1^2 - k * p.2^2 = 5}) →  -- Focus at (0, 2)
  k = -1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_k_value_l4083_408354


namespace NUMINAMATH_CALUDE_larger_number_with_given_hcf_lcm_factors_l4083_408323

theorem larger_number_with_given_hcf_lcm_factors (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  Nat.gcd a b = 120 ∧
  ∃ k : ℕ, Nat.lcm a b = 120 * 13 * 17 * 23 * k ∧ k = 1 →
  max a b = 26520 :=
by sorry

end NUMINAMATH_CALUDE_larger_number_with_given_hcf_lcm_factors_l4083_408323


namespace NUMINAMATH_CALUDE_length_MN_circle_P_equation_l4083_408347

-- Define the line l
def line_l (x y : ℝ) : Prop := y = x - 1

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define the intersection points M and N
def intersection_points (M N : ℝ × ℝ) : Prop :=
  line_l M.1 M.2 ∧ circle_C M.1 M.2 ∧
  line_l N.1 N.2 ∧ circle_C N.1 N.2 ∧
  M ≠ N

-- Theorem for the length of MN
theorem length_MN (M N : ℝ × ℝ) (h : intersection_points M N) :
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 2 :=
sorry

-- Define the circle P
def circle_P (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 1

-- Theorem for the equation of circle P
theorem circle_P_equation (M N : ℝ × ℝ) (h : intersection_points M N) :
  ∀ x y : ℝ, circle_P x y ↔ 
    ((x - (M.1 + N.1) / 2)^2 + (y - (M.2 + N.2) / 2)^2 = 
     ((M.1 - N.1)^2 + (M.2 - N.2)^2) / 4) :=
sorry

end NUMINAMATH_CALUDE_length_MN_circle_P_equation_l4083_408347


namespace NUMINAMATH_CALUDE_thirty_thousand_squared_l4083_408383

theorem thirty_thousand_squared :
  (30000 : ℕ) ^ 2 = 900000000 := by
  sorry

end NUMINAMATH_CALUDE_thirty_thousand_squared_l4083_408383


namespace NUMINAMATH_CALUDE_missing_number_l4083_408351

theorem missing_number (x z : ℕ) 
  (h1 : x * 2 = 8)
  (h2 : 2 * z = 16)
  (h3 : 8 * 7 = 56)
  (h4 : 16 * 7 = 112) :
  x = 4 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_l4083_408351


namespace NUMINAMATH_CALUDE_min_colors_theorem_l4083_408303

def is_multiple (m n : ℕ) : Prop := ∃ k : ℕ, m = k * n

def valid_coloring (f : ℕ → ℕ) : Prop :=
  ∀ m n, 2 ≤ n ∧ n < m ∧ m ≤ 31 → is_multiple m n → f m ≠ f n

theorem min_colors_theorem :
  ∃ (k : ℕ) (f : ℕ → ℕ),
    (∀ n, 2 ≤ n ∧ n ≤ 31 → f n < k) ∧
    valid_coloring f ∧
    (∀ k' < k, ¬∃ f', (∀ n, 2 ≤ n ∧ n ≤ 31 → f' n < k') ∧ valid_coloring f') ∧
    k = 4 :=
sorry

end NUMINAMATH_CALUDE_min_colors_theorem_l4083_408303


namespace NUMINAMATH_CALUDE_fifteen_percent_of_600_is_90_l4083_408304

theorem fifteen_percent_of_600_is_90 :
  ∀ x : ℝ, (15 / 100) * x = 90 → x = 600 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_600_is_90_l4083_408304


namespace NUMINAMATH_CALUDE_scientific_notation_13000_l4083_408378

theorem scientific_notation_13000 : 13000 = 1.3 * (10 ^ 4) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_13000_l4083_408378


namespace NUMINAMATH_CALUDE_only_parallelogram_not_axially_symmetric_l4083_408390

-- Define the shapes
inductive Shape
  | Rectangle
  | IsoscelesTrapezoid
  | Parallelogram
  | EquilateralTriangle

-- Define axial symmetry
def is_axially_symmetric (s : Shape) : Prop :=
  match s with
  | Shape.Rectangle => true
  | Shape.IsoscelesTrapezoid => true
  | Shape.Parallelogram => false
  | Shape.EquilateralTriangle => true

-- Theorem statement
theorem only_parallelogram_not_axially_symmetric :
  ∀ s : Shape, ¬(is_axially_symmetric s) ↔ s = Shape.Parallelogram :=
by sorry

end NUMINAMATH_CALUDE_only_parallelogram_not_axially_symmetric_l4083_408390


namespace NUMINAMATH_CALUDE_difference_of_fractions_numerator_l4083_408310

theorem difference_of_fractions_numerator : 
  let a := 2024
  let b := 2023
  let diff := a / b - b / a
  let p := (a^2 - b^2) / (a * b)
  p = 4047 := by sorry

end NUMINAMATH_CALUDE_difference_of_fractions_numerator_l4083_408310


namespace NUMINAMATH_CALUDE_smallest_number_l4083_408371

theorem smallest_number (s : Finset ℚ) (hs : s = {-5, 1, -1, 0}) : 
  ∀ x ∈ s, -5 ≤ x :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_number_l4083_408371


namespace NUMINAMATH_CALUDE_log_problem_l4083_408358

theorem log_problem (x y : ℝ) (h1 : Real.log (x * y^4) = 1) (h2 : Real.log (x^3 * y) = 1) :
  Real.log (x^3 * y^2) = 13/11 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l4083_408358


namespace NUMINAMATH_CALUDE_magic_balls_theorem_l4083_408361

theorem magic_balls_theorem :
  ∃ (n : ℕ), 5 + 4 * n = 2005 :=
by sorry

end NUMINAMATH_CALUDE_magic_balls_theorem_l4083_408361


namespace NUMINAMATH_CALUDE_cake_mass_proof_l4083_408391

/-- The initial mass of the cake in grams -/
def initial_mass : ℝ := 750

/-- The mass of cake Karlson ate for breakfast -/
def karlson_ate : ℝ := 0.4 * initial_mass

/-- The mass of cake Malish ate for breakfast -/
def malish_ate : ℝ := 150

/-- The percentage of remaining cake Freken Bok ate for lunch -/
def freken_bok_percent : ℝ := 0.3

/-- The additional mass of cake Freken Bok ate for lunch -/
def freken_bok_additional : ℝ := 120

/-- The mass of cake crumbs Matilda licked -/
def matilda_licked : ℝ := 90

theorem cake_mass_proof :
  initial_mass = karlson_ate + malish_ate +
  (freken_bok_percent * (initial_mass - karlson_ate - malish_ate) + freken_bok_additional) +
  matilda_licked := by sorry

end NUMINAMATH_CALUDE_cake_mass_proof_l4083_408391


namespace NUMINAMATH_CALUDE_ring_price_is_7_10_l4083_408376

/-- Represents the sales at a craft fair -/
structure CraftFairSales where
  necklace_price : ℝ
  earrings_price : ℝ
  ring_price : ℝ
  necklaces_sold : ℕ
  rings_sold : ℕ
  earrings_sold : ℕ
  bracelets_sold : ℕ
  total_sales : ℝ

/-- The cost of a bracelet is twice the cost of a ring -/
def bracelet_price (sales : CraftFairSales) : ℝ := 2 * sales.ring_price

/-- Theorem stating that the ring price is $7.10 given the conditions -/
theorem ring_price_is_7_10 (sales : CraftFairSales) 
  (h1 : sales.necklace_price = 12)
  (h2 : sales.earrings_price = 10)
  (h3 : sales.necklaces_sold = 4)
  (h4 : sales.rings_sold = 8)
  (h5 : sales.earrings_sold = 5)
  (h6 : sales.bracelets_sold = 6)
  (h7 : sales.total_sales = 240)
  (h8 : sales.necklace_price * sales.necklaces_sold + 
        sales.ring_price * sales.rings_sold + 
        sales.earrings_price * sales.earrings_sold + 
        bracelet_price sales * sales.bracelets_sold = sales.total_sales) :
  sales.ring_price = 7.1 := by
  sorry


end NUMINAMATH_CALUDE_ring_price_is_7_10_l4083_408376


namespace NUMINAMATH_CALUDE_community_cleaning_event_l4083_408341

theorem community_cleaning_event (total : ℝ) : 
  (0.3 * total = total * 0.3) →
  (0.6 * total = 2 * (total * 0.3)) →
  (total - (total * 0.3 + 0.6 * total) = 200) →
  total = 2000 := by
sorry

end NUMINAMATH_CALUDE_community_cleaning_event_l4083_408341


namespace NUMINAMATH_CALUDE_total_savings_calculation_l4083_408384

def initial_savings : ℕ := 849400
def monthly_income : ℕ := 110000
def monthly_expenses : ℕ := 58500
def months : ℕ := 5

theorem total_savings_calculation :
  initial_savings + months * monthly_income - months * monthly_expenses = 1106900 := by
  sorry

end NUMINAMATH_CALUDE_total_savings_calculation_l4083_408384


namespace NUMINAMATH_CALUDE_book_cost_calculation_l4083_408379

theorem book_cost_calculation (num_books : ℕ) (money_have : ℕ) (money_save : ℕ) :
  num_books = 8 ∧ money_have = 13 ∧ money_save = 27 →
  (money_have + money_save) / num_books = 5 := by
sorry

end NUMINAMATH_CALUDE_book_cost_calculation_l4083_408379


namespace NUMINAMATH_CALUDE_man_mass_on_boat_l4083_408319

/-- The mass of a man causing a boat to sink by a certain depth --/
def mass_of_man (boat_length boat_breadth sinking_depth water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * sinking_depth * water_density

/-- Theorem stating the mass of the man in the given problem --/
theorem man_mass_on_boat : 
  let boat_length : ℝ := 3
  let boat_breadth : ℝ := 2
  let sinking_depth : ℝ := 0.018  -- 1.8 cm converted to meters
  let water_density : ℝ := 1000
  mass_of_man boat_length boat_breadth sinking_depth water_density = 108 := by
  sorry

end NUMINAMATH_CALUDE_man_mass_on_boat_l4083_408319


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_specific_root_condition_l4083_408325

theorem quadratic_equation_roots (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x + k + 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ↔ k ≤ 3 :=
by sorry

theorem specific_root_condition (k : ℝ) (x₁ x₂ : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4*x + k + 1
  (f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂ ∧ 3/x₁ + 3/x₂ = x₁*x₂ - 4) → k = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_specific_root_condition_l4083_408325


namespace NUMINAMATH_CALUDE_symmetry_implies_values_l4083_408302

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are negatives of each other and their y-coordinates are equal. -/
def symmetric_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

/-- The theorem states that if point P with coordinates (a-b, 2a+b) is symmetric to point Q (3, 2) with respect to the y-axis, then a = -1/3 and b = 8/3. -/
theorem symmetry_implies_values (a b : ℝ) :
  symmetric_y_axis (a - b, 2 * a + b) (3, 2) →
  a = -1/3 ∧ b = 8/3 := by
sorry

end NUMINAMATH_CALUDE_symmetry_implies_values_l4083_408302


namespace NUMINAMATH_CALUDE_max_profit_is_850_l4083_408306

def fruit_problem (m : ℝ) : Prop :=
  let total_weight : ℝ := 200
  let profit_A : ℝ := 20 - 16
  let profit_B : ℝ := 25 - 20
  let total_profit : ℝ := m * profit_A + (total_weight - m) * profit_B
  0 ≤ m ∧ m ≤ total_weight ∧ m ≥ 3 * (total_weight - m) →
  total_profit ≤ 850

theorem max_profit_is_850 :
  ∃ m : ℝ, fruit_problem m ∧
  (∀ n : ℝ, fruit_problem n → 
    m * (20 - 16) + (200 - m) * (25 - 20) ≥ n * (20 - 16) + (200 - n) * (25 - 20)) :=
sorry

end NUMINAMATH_CALUDE_max_profit_is_850_l4083_408306


namespace NUMINAMATH_CALUDE_seal_earnings_l4083_408373

/-- Calculates the total earnings of a musician over a given number of years -/
def musician_earnings (songs_per_month : ℕ) (earnings_per_song : ℕ) (years : ℕ) : ℕ :=
  songs_per_month * earnings_per_song * 12 * years

/-- Proves that Seal's earnings over 3 years, given the specified conditions, equal $216,000 -/
theorem seal_earnings : musician_earnings 3 2000 3 = 216000 := by
  sorry

end NUMINAMATH_CALUDE_seal_earnings_l4083_408373


namespace NUMINAMATH_CALUDE_scientific_notation_of_1050000_l4083_408330

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  norm : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_1050000 :
  toScientificNotation 1050000 = ScientificNotation.mk 1.05 6 (by norm_num) :=
sorry

end NUMINAMATH_CALUDE_scientific_notation_of_1050000_l4083_408330


namespace NUMINAMATH_CALUDE_travis_cereal_cost_l4083_408381

/-- The amount Travis spends on cereal in a year -/
def cereal_cost (boxes_per_week : ℕ) (cost_per_box : ℚ) (weeks_per_year : ℕ) : ℚ :=
  (boxes_per_week : ℚ) * cost_per_box * (weeks_per_year : ℚ)

/-- Proof that Travis spends $312.00 on cereal in a year -/
theorem travis_cereal_cost :
  cereal_cost 2 3 52 = 312 := by
  sorry

end NUMINAMATH_CALUDE_travis_cereal_cost_l4083_408381


namespace NUMINAMATH_CALUDE_root_difference_l4083_408357

/-- The difference between the larger and smaller roots of the quadratic equation
    x^2 - 2px + (p^2 - 4p + 4) = 0, where p is a real number. -/
theorem root_difference (p : ℝ) : 
  let a := 1
  let b := -2*p
  let c := p^2 - 4*p + 4
  let discriminant := b^2 - 4*a*c
  let larger_root := (-b + Real.sqrt discriminant) / (2*a)
  let smaller_root := (-b - Real.sqrt discriminant) / (2*a)
  larger_root - smaller_root = 4 * Real.sqrt (p - 1) := by
sorry

end NUMINAMATH_CALUDE_root_difference_l4083_408357


namespace NUMINAMATH_CALUDE_trisha_annual_take_home_pay_l4083_408301

/-- Calculates the annual take-home pay given hourly rate, weekly hours, weeks worked, and withholding rate. -/
def annual_take_home_pay (hourly_rate : ℝ) (weekly_hours : ℝ) (weeks_worked : ℝ) (withholding_rate : ℝ) : ℝ :=
  let gross_pay := hourly_rate * weekly_hours * weeks_worked
  let withheld_amount := withholding_rate * gross_pay
  gross_pay - withheld_amount

/-- Proves that given the specified conditions, Trisha's annual take-home pay is $24,960. -/
theorem trisha_annual_take_home_pay :
  annual_take_home_pay 15 40 52 0.2 = 24960 := by
  sorry

#eval annual_take_home_pay 15 40 52 0.2

end NUMINAMATH_CALUDE_trisha_annual_take_home_pay_l4083_408301


namespace NUMINAMATH_CALUDE_eight_p_plus_one_composite_l4083_408305

theorem eight_p_plus_one_composite (p : ℕ) (h1 : Nat.Prime p) (h2 : Nat.Prime (8 * p - 1)) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ a * b = 8 * p + 1 :=
by sorry

end NUMINAMATH_CALUDE_eight_p_plus_one_composite_l4083_408305


namespace NUMINAMATH_CALUDE_unsafe_trip_probability_775km_l4083_408336

/-- The probability of not completing a trip safely given the probability of an accident per km and the total distance. -/
def unsafe_trip_probability (p : ℝ) (distance : ℕ) : ℝ :=
  1 - (1 - p) ^ distance

/-- Theorem stating that the probability of not completing a 775 km trip safely
    is equal to 1 - (1 - p)^775, where p is the probability of an accident per km. -/
theorem unsafe_trip_probability_775km (p : ℝ) (h1 : 0 ≤ p) (h2 : p ≤ 1) :
  unsafe_trip_probability p 775 = 1 - (1 - p)^775 := by
  sorry

#check unsafe_trip_probability_775km

end NUMINAMATH_CALUDE_unsafe_trip_probability_775km_l4083_408336


namespace NUMINAMATH_CALUDE_original_price_calculation_l4083_408389

theorem original_price_calculation (decreased_price : ℝ) (decrease_percentage : ℝ) 
  (h1 : decreased_price = 1064)
  (h2 : decrease_percentage = 24) : 
  decreased_price / (1 - decrease_percentage / 100) = 1400 := by
  sorry

end NUMINAMATH_CALUDE_original_price_calculation_l4083_408389


namespace NUMINAMATH_CALUDE_prime_condition_characterization_l4083_408399

def satisfies_condition (p : Nat) : Prop :=
  Nat.Prime p ∧
  ∀ q, Nat.Prime q → q < p →
    ∀ k r, p = k * q + r → 0 ≤ r → r < q →
      ∀ a, a > 1 → ¬(a^2 ∣ r)

theorem prime_condition_characterization :
  {p : Nat | satisfies_condition p} = {2, 3, 5, 7, 13} := by sorry

end NUMINAMATH_CALUDE_prime_condition_characterization_l4083_408399


namespace NUMINAMATH_CALUDE_falcons_win_percentage_l4083_408388

theorem falcons_win_percentage (initial_games : ℕ) (initial_falcon_wins : ℕ) (win_percentage : ℚ) :
  let additional_games : ℕ := 42
  initial_games = 8 ∧ 
  initial_falcon_wins = 3 ∧ 
  win_percentage = 9/10 →
  (initial_falcon_wins + additional_games : ℚ) / (initial_games + additional_games) ≥ win_percentage ∧
  ∀ n : ℕ, n < additional_games → 
    (initial_falcon_wins + n : ℚ) / (initial_games + n) < win_percentage :=
by sorry

end NUMINAMATH_CALUDE_falcons_win_percentage_l4083_408388


namespace NUMINAMATH_CALUDE_num_valid_codes_correct_num_valid_codes_positive_l4083_408327

/-- The number of possible 5-digit codes where no digit is used more than twice. -/
def num_valid_codes : ℕ := 102240

/-- A function that calculates the number of valid 5-digit codes. -/
def calculate_valid_codes : ℕ :=
  let all_different := 10 * 9 * 8 * 7 * 6
  let one_digit_repeated := 10 * (5 * 4 / 2) * 9 * 8 * 7
  let two_digits_repeated := 10 * 9 * (5 * 4 / 2) * (3 * 2 / 2) * 8
  all_different + one_digit_repeated + two_digits_repeated

/-- Theorem stating that the number of valid codes is correct. -/
theorem num_valid_codes_correct : calculate_valid_codes = num_valid_codes := by
  sorry

/-- Theorem stating that the calculated number of valid codes is positive. -/
theorem num_valid_codes_positive : 0 < num_valid_codes := by
  sorry

end NUMINAMATH_CALUDE_num_valid_codes_correct_num_valid_codes_positive_l4083_408327


namespace NUMINAMATH_CALUDE_quadrilateral_is_parallelogram_l4083_408367

-- Define a quadrilateral in 2D space
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define a function to check if a quadrilateral is a parallelogram
def is_parallelogram (q : Quadrilateral) : Prop :=
  let BA := (q.B.1 - q.A.1, q.B.2 - q.A.2)
  let DC := (q.D.1 - q.C.1, q.D.2 - q.C.2)
  BA = DC

-- Theorem statement
theorem quadrilateral_is_parallelogram (q : Quadrilateral) (O : ℝ × ℝ) :
  (q.A.1 - O.1, q.A.2 - O.2) + (q.C.1 - O.1, q.C.2 - O.2) =
  (q.B.1 - O.1, q.B.2 - O.2) + (q.D.1 - O.1, q.D.2 - O.2) →
  is_parallelogram q :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_is_parallelogram_l4083_408367


namespace NUMINAMATH_CALUDE_original_group_size_l4083_408397

/-- Proves that the original number of men in a group is 22, given the conditions of the problem. -/
theorem original_group_size (initial_days : ℕ) (absent_men : ℕ) (final_days : ℕ) : 
  initial_days = 20 → absent_men = 2 → final_days = 22 → 
  ∃ (original_men : ℕ), 
    original_men * initial_days = (original_men - absent_men) * final_days ∧ 
    original_men = 22 := by
  sorry

end NUMINAMATH_CALUDE_original_group_size_l4083_408397


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_bound_l4083_408364

/-- A convex polygon with area 1 -/
structure ConvexPolygon where
  area : ℝ
  isConvex : Bool
  area_eq_one : area = 1
  is_convex : isConvex = true

/-- A triangle inscribed in a convex polygon -/
structure InscribedTriangle (p : ConvexPolygon) where
  area : ℝ
  is_inscribed : Bool

/-- Theorem: Any convex polygon with area 1 contains a triangle with area at least 3/8 -/
theorem inscribed_triangle_area_bound (p : ConvexPolygon) : 
  ∃ (t : InscribedTriangle p), t.area ≥ 3/8 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_bound_l4083_408364


namespace NUMINAMATH_CALUDE_ratio_problem_l4083_408369

theorem ratio_problem (a b : ℤ) : 
  (a : ℚ) / b = 1 / 4 → 
  (a + 6 : ℚ) / b = 1 / 2 → 
  b = 24 := by
sorry

end NUMINAMATH_CALUDE_ratio_problem_l4083_408369


namespace NUMINAMATH_CALUDE_skaters_meeting_time_l4083_408339

/-- The time it takes for two skaters to meet on a circular rink -/
theorem skaters_meeting_time 
  (rink_circumference : ℝ) 
  (speed_skater1 : ℝ) 
  (speed_skater2 : ℝ) 
  (h1 : rink_circumference = 3000) 
  (h2 : speed_skater1 = 100) 
  (h3 : speed_skater2 = 150) : 
  rink_circumference / (speed_skater1 + speed_skater2) = 12 := by
  sorry

#check skaters_meeting_time

end NUMINAMATH_CALUDE_skaters_meeting_time_l4083_408339


namespace NUMINAMATH_CALUDE_trouser_discount_proof_l4083_408370

/-- The final percent decrease in price for a trouser with given original price and discount -/
def final_percent_decrease (original_price discount_percent : ℝ) : ℝ :=
  discount_percent

theorem trouser_discount_proof (original_price discount_percent : ℝ) 
  (h1 : original_price = 100)
  (h2 : discount_percent = 30) :
  final_percent_decrease original_price discount_percent = 30 := by
  sorry

#eval final_percent_decrease 100 30

end NUMINAMATH_CALUDE_trouser_discount_proof_l4083_408370


namespace NUMINAMATH_CALUDE_raft_sticks_total_l4083_408307

def simon_sticks : ℕ := 36

def gerry_sticks : ℕ := (2 * simon_sticks) / 3

def micky_sticks : ℕ := simon_sticks + gerry_sticks + 9

def darryl_sticks : ℕ := simon_sticks + gerry_sticks + micky_sticks + 1

def total_sticks : ℕ := simon_sticks + gerry_sticks + micky_sticks + darryl_sticks

theorem raft_sticks_total : total_sticks = 259 := by
  sorry

end NUMINAMATH_CALUDE_raft_sticks_total_l4083_408307


namespace NUMINAMATH_CALUDE_lemonade_sales_difference_l4083_408380

/-- Anna's lemonade sales problem -/
theorem lemonade_sales_difference :
  let plain_glasses : ℕ := 36
  let plain_price : ℚ := 3/4  -- $0.75 represented as a rational number
  let strawberry_earnings : ℚ := 16
  let plain_earnings := plain_glasses * plain_price
  plain_earnings - strawberry_earnings = 11 := by sorry

end NUMINAMATH_CALUDE_lemonade_sales_difference_l4083_408380


namespace NUMINAMATH_CALUDE_train_speed_problem_l4083_408343

/-- Proves that given a train journey of 3x km, where x km is traveled at 50 kmph
    and 2x km is traveled at speed v, and the average speed for the entire journey
    is 25 kmph, the speed v must be 20 kmph. -/
theorem train_speed_problem (x : ℝ) (v : ℝ) (h_x_pos : x > 0) :
  (x / 50 + 2 * x / v = 3 * x / 25) → v = 20 := by
  sorry

#check train_speed_problem

end NUMINAMATH_CALUDE_train_speed_problem_l4083_408343


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l4083_408386

/-- Given a geometric sequence {aₙ} with common ratio q = √2 and a₁ · a₃ · a₅ = 4,
    prove that a₄ · a₅ · a₆ = 32. -/
theorem geometric_sequence_product (a : ℕ → ℝ) :
  (∀ n, a (n + 1) = a n * Real.sqrt 2) →  -- geometric sequence with ratio √2
  a 1 * a 3 * a 5 = 4 →                   -- given condition
  a 4 * a 5 * a 6 = 32 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l4083_408386


namespace NUMINAMATH_CALUDE_complex_sum_modulus_l4083_408377

theorem complex_sum_modulus (z₁ z₂ : ℂ) 
  (h1 : Complex.abs z₁ = 1) 
  (h2 : Complex.abs z₂ = 1) 
  (h3 : Complex.abs (z₁ - z₂) = Real.sqrt 3) : 
  Complex.abs (z₁ + z₂) = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_modulus_l4083_408377


namespace NUMINAMATH_CALUDE_space_diagonals_specific_polyhedron_l4083_408346

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  sorry

/-- Theorem stating the number of space diagonals in the specific polyhedron -/
theorem space_diagonals_specific_polyhedron :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 70,
    faces := 42,
    triangular_faces := 30,
    quadrilateral_faces := 12
  }
  space_diagonals Q = 341 := by
  sorry

end NUMINAMATH_CALUDE_space_diagonals_specific_polyhedron_l4083_408346


namespace NUMINAMATH_CALUDE_range_of_a_l4083_408382

theorem range_of_a (a : ℝ) : 
  (∃ b : ℝ, b ∈ Set.Icc 1 2 ∧ 2^b * (b + a) ≥ 4) ↔ a ∈ Set.Ici (-1) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l4083_408382


namespace NUMINAMATH_CALUDE_mary_added_candy_l4083_408313

/-- Proof that Mary added 10 pieces of candy to her collection --/
theorem mary_added_candy (megan_candy : ℕ) (mary_total : ℕ) (h1 : megan_candy = 5) (h2 : mary_total = 25) :
  mary_total - (3 * megan_candy) = 10 := by
  sorry

end NUMINAMATH_CALUDE_mary_added_candy_l4083_408313


namespace NUMINAMATH_CALUDE_jessica_seashells_l4083_408362

/-- The number of seashells Jessica gave to Joan -/
def seashells_given : ℕ := 6

/-- The number of seashells Jessica kept -/
def seashells_kept : ℕ := 2

/-- The initial number of seashells Jessica found -/
def initial_seashells : ℕ := seashells_given + seashells_kept

theorem jessica_seashells : initial_seashells = 8 := by
  sorry

end NUMINAMATH_CALUDE_jessica_seashells_l4083_408362


namespace NUMINAMATH_CALUDE_valid_selections_count_l4083_408350

/-- The number of ways to select 4 out of 6 people to visit 4 distinct places -/
def total_selections : ℕ := 360

/-- The number of ways where person A visits the restricted place -/
def a_restricted : ℕ := 60

/-- The number of ways where person B visits the restricted place -/
def b_restricted : ℕ := 60

/-- The number of valid selection schemes -/
def valid_selections : ℕ := total_selections - a_restricted - b_restricted

theorem valid_selections_count : valid_selections = 240 := by sorry

end NUMINAMATH_CALUDE_valid_selections_count_l4083_408350


namespace NUMINAMATH_CALUDE_oranges_remaining_l4083_408322

theorem oranges_remaining (initial_oranges removed_oranges : ℕ) 
  (h1 : initial_oranges = 96)
  (h2 : removed_oranges = 45) :
  initial_oranges - removed_oranges = 51 := by
sorry

end NUMINAMATH_CALUDE_oranges_remaining_l4083_408322


namespace NUMINAMATH_CALUDE_beverly_bottle_caps_l4083_408329

/-- The number of groups of bottle caps in Beverly's collection -/
def num_groups : ℕ := 7

/-- The number of bottle caps in each group -/
def caps_per_group : ℕ := 5

/-- The total number of bottle caps in Beverly's collection -/
def total_caps : ℕ := num_groups * caps_per_group

theorem beverly_bottle_caps : total_caps = 35 := by
  sorry

end NUMINAMATH_CALUDE_beverly_bottle_caps_l4083_408329


namespace NUMINAMATH_CALUDE_aruns_weight_lower_limit_l4083_408360

theorem aruns_weight_lower_limit 
  (lower_bound : ℝ) 
  (upper_bound : ℝ) 
  (h1 : lower_bound > 65)
  (h2 : upper_bound ≤ 68)
  (h3 : (lower_bound + upper_bound) / 2 = 67) :
  lower_bound = 66 := by
sorry

end NUMINAMATH_CALUDE_aruns_weight_lower_limit_l4083_408360


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l4083_408326

def total_students : ℕ := 470
def football_players : ℕ := 325
def cricket_players : ℕ := 175
def neither_players : ℕ := 50

theorem students_playing_both_sports : ℕ := by
  sorry

#check students_playing_both_sports = 80

end NUMINAMATH_CALUDE_students_playing_both_sports_l4083_408326


namespace NUMINAMATH_CALUDE_correct_equation_l4083_408312

theorem correct_equation (a b : ℝ) : 3 * a^2 * b - 4 * b * a^2 = -a^2 * b := by
  sorry

end NUMINAMATH_CALUDE_correct_equation_l4083_408312


namespace NUMINAMATH_CALUDE_roots_relation_l4083_408365

theorem roots_relation (n r : ℝ) (c d : ℝ) : 
  (c^2 - n*c + 3 = 0) → 
  (d^2 - n*d + 3 = 0) → 
  ((c + 1/d)^2 - r*(c + 1/d) + s = 0) → 
  ((d + 1/c)^2 - r*(d + 1/c) + s = 0) → 
  s = 16/3 := by sorry

end NUMINAMATH_CALUDE_roots_relation_l4083_408365


namespace NUMINAMATH_CALUDE_adam_ferris_wheel_cost_l4083_408340

/-- The amount of money Adam spent on the ferris wheel ride -/
def ferris_wheel_cost (initial_tickets : ℕ) (remaining_tickets : ℕ) (ticket_price : ℕ) : ℕ :=
  (initial_tickets - remaining_tickets) * ticket_price

/-- Theorem: Adam spent 81 dollars on the ferris wheel ride -/
theorem adam_ferris_wheel_cost :
  ferris_wheel_cost 13 4 9 = 81 := by
  sorry

end NUMINAMATH_CALUDE_adam_ferris_wheel_cost_l4083_408340


namespace NUMINAMATH_CALUDE_son_work_time_l4083_408315

/-- Given a task that can be completed by a man in 7 days or by the man and his son together in 3 days, 
    this theorem proves that the son can complete the task alone in 5.25 days. -/
theorem son_work_time (man_time : ℝ) (combined_time : ℝ) (son_time : ℝ) : 
  man_time = 7 → combined_time = 3 → son_time = 21 / 4 := by
  sorry

end NUMINAMATH_CALUDE_son_work_time_l4083_408315


namespace NUMINAMATH_CALUDE_motorcycle_license_count_l4083_408314

/-- The number of possible letters for a motorcycle license -/
def num_letters : ℕ := 3

/-- The number of digits in a motorcycle license -/
def num_digits : ℕ := 6

/-- The number of possible choices for each digit -/
def choices_per_digit : ℕ := 10

/-- The total number of possible motorcycle licenses -/
def total_licenses : ℕ := num_letters * (choices_per_digit ^ num_digits)

theorem motorcycle_license_count :
  total_licenses = 3000000 := by
  sorry

end NUMINAMATH_CALUDE_motorcycle_license_count_l4083_408314


namespace NUMINAMATH_CALUDE_cos_alpha_for_point_four_neg_three_l4083_408387

theorem cos_alpha_for_point_four_neg_three (α : Real) :
  (∃ (x y : Real), x = 4 ∧ y = -3 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.cos α = 4/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_for_point_four_neg_three_l4083_408387


namespace NUMINAMATH_CALUDE_positive_roots_range_l4083_408308

theorem positive_roots_range (m : ℝ) :
  (∀ x : ℝ, x^2 + (m+2)*x + m+5 = 0 → x > 0) ↔ -5 < m ∧ m ≤ -4 := by
  sorry

end NUMINAMATH_CALUDE_positive_roots_range_l4083_408308


namespace NUMINAMATH_CALUDE_symmetry_implies_coordinates_l4083_408363

/-- Two points are symmetric with respect to the y-axis if their x-coordinates are opposites and their y-coordinates are the same. -/
def symmetric_wrt_y_axis (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = q.2

theorem symmetry_implies_coordinates (a b : ℝ) :
  symmetric_wrt_y_axis (a, 3) (-2, b) → a = 2 ∧ b = 3 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_coordinates_l4083_408363


namespace NUMINAMATH_CALUDE_sequence_properties_l4083_408385

/-- Sequence sum function -/
def S (n : ℕ) : ℤ := -n^2 + 7*n

/-- Sequence term function -/
def a (n : ℕ) : ℤ := -2*n + 8

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a n = S n - S (n-1)) ∧
  (∃ m : ℕ, m ≥ 1 ∧ ∀ n : ℕ, n ≥ 1 → S n ≤ S m) ∧
  (∀ n : ℕ, n ≥ 1 → S n ≤ 12) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l4083_408385


namespace NUMINAMATH_CALUDE_combined_average_marks_l4083_408349

theorem combined_average_marks (n1 n2 : ℕ) (avg1 avg2 : ℝ) : 
  n1 = 26 → 
  n2 = 50 → 
  avg1 = 40 → 
  avg2 = 60 → 
  let total_students := n1 + n2
  let total_marks := n1 * avg1 + n2 * avg2
  abs ((total_marks / total_students) - 53.16) < 0.01 := by
sorry

end NUMINAMATH_CALUDE_combined_average_marks_l4083_408349


namespace NUMINAMATH_CALUDE_graph_is_two_parabolas_l4083_408375

/-- The equation of the conic sections -/
def equation (x y : ℝ) : Prop :=
  y^6 - 9*x^6 = 3*y^3 - 1

/-- A parabola in cubic form -/
def cubic_parabola (x y : ℝ) (a b : ℝ) : Prop :=
  y^3 + a*x^3 = b

/-- The graph consists of two parabolas -/
theorem graph_is_two_parabolas :
  ∃ (a₁ b₁ a₂ b₂ : ℝ), 
    (∀ x y, equation x y ↔ (cubic_parabola x y a₁ b₁ ∨ cubic_parabola x y a₂ b₂)) ∧
    a₁ ≠ a₂ :=
  sorry

end NUMINAMATH_CALUDE_graph_is_two_parabolas_l4083_408375


namespace NUMINAMATH_CALUDE_initial_cards_l4083_408332

theorem initial_cards (x : ℚ) : 
  (x ≥ 0) → 
  (3 * (1/2) * ((x/3) + (4/3)) = 34) → 
  (x = 64) := by
sorry

end NUMINAMATH_CALUDE_initial_cards_l4083_408332


namespace NUMINAMATH_CALUDE_certain_number_addition_l4083_408395

theorem certain_number_addition (x : ℝ) (h : 5 * x = 60) : x + 34 = 46 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_addition_l4083_408395


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l4083_408309

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt x / 15 = 4 → x = 3600 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l4083_408309


namespace NUMINAMATH_CALUDE_last_digit_largest_power_of_3_dividing_27_factorial_l4083_408320

/-- The largest power of 3 that divides n! -/
def largestPowerOf3DividingFactorial (n : ℕ) : ℕ :=
  sorry

/-- The last digit of a natural number -/
def lastDigit (n : ℕ) : ℕ :=
  n % 10

theorem last_digit_largest_power_of_3_dividing_27_factorial :
  lastDigit (3^(largestPowerOf3DividingFactorial 27)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_largest_power_of_3_dividing_27_factorial_l4083_408320


namespace NUMINAMATH_CALUDE_polynomial_factorization_l4083_408345

theorem polynomial_factorization (x : ℝ) :
  (x^2 + 2*x + 1) * (x^2 + 8*x + 15) + (x^2 + 6*x + 5) = (x + 1) * (x + 5) * (x + 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l4083_408345


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l4083_408344

/-- The perimeter of a semicircle with radius 20 is approximately 102.83 -/
theorem semicircle_perimeter_approx :
  let r : ℝ := 20
  let perimeter : ℝ := π * r + 2 * r
  ∃ ε > 0, abs (perimeter - 102.83) < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l4083_408344


namespace NUMINAMATH_CALUDE_aaron_earnings_l4083_408398

/-- Represents the work hours for each day of the week -/
structure WorkHours :=
  (monday : Real)
  (tuesday : Real)
  (wednesday : Real)
  (friday : Real)

/-- Calculates the total earnings for the week given work hours and hourly rate -/
def calculateEarnings (hours : WorkHours) (hourlyRate : Real) : Real :=
  (hours.monday + hours.tuesday + hours.wednesday + hours.friday) * hourlyRate

/-- Theorem stating that Aaron's earnings for the week are $38.75 -/
theorem aaron_earnings :
  let hours : WorkHours := {
    monday := 2,
    tuesday := 1.25,
    wednesday := 2.833,
    friday := 0.667
  }
  let hourlyRate : Real := 5
  calculateEarnings hours hourlyRate = 38.75 := by
  sorry

#check aaron_earnings

end NUMINAMATH_CALUDE_aaron_earnings_l4083_408398


namespace NUMINAMATH_CALUDE_jennas_profit_l4083_408331

/-- Calculates the total profit for Jenna's wholesale business --/
def calculate_profit (buy_price sell_price rent tax_rate worker_salary num_workers num_widgets : ℝ) : ℝ :=
  let total_revenue := sell_price * num_widgets
  let total_cost := buy_price * num_widgets
  let gross_profit := total_revenue - total_cost
  let total_expenses := rent + (worker_salary * num_workers)
  let net_profit_before_tax := gross_profit - total_expenses
  let taxes := tax_rate * net_profit_before_tax
  net_profit_before_tax - taxes

/-- Theorem stating that Jenna's total profit is $4,000 given the specified conditions --/
theorem jennas_profit :
  calculate_profit 3 8 10000 0.2 2500 4 5000 = 4000 := by
  sorry

end NUMINAMATH_CALUDE_jennas_profit_l4083_408331


namespace NUMINAMATH_CALUDE_fraction_sum_l4083_408355

theorem fraction_sum : (3 : ℚ) / 9 + (5 : ℚ) / 12 = (3 : ℚ) / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l4083_408355


namespace NUMINAMATH_CALUDE_vector_addition_scalar_mult_l4083_408335

/-- Given plane vectors a and b, prove that 3a + b equals (-2, 6) -/
theorem vector_addition_scalar_mult 
  (a b : ℝ × ℝ) 
  (ha : a = (-1, 2)) 
  (hb : b = (1, 0)) : 
  (3 : ℝ) • a + b = (-2, 6) := by
  sorry

end NUMINAMATH_CALUDE_vector_addition_scalar_mult_l4083_408335


namespace NUMINAMATH_CALUDE_cos_270_degrees_l4083_408338

theorem cos_270_degrees : Real.cos (270 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_cos_270_degrees_l4083_408338


namespace NUMINAMATH_CALUDE_ice_cream_distribution_l4083_408393

theorem ice_cream_distribution (total_sandwiches : ℕ) (num_nieces : ℕ) 
  (h1 : total_sandwiches = 143) (h2 : num_nieces = 11) :
  ∃ (sandwiches_per_niece : ℕ), 
    sandwiches_per_niece * num_nieces = total_sandwiches ∧ 
    sandwiches_per_niece = 13 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_distribution_l4083_408393


namespace NUMINAMATH_CALUDE_floor_paving_cost_l4083_408374

theorem floor_paving_cost 
  (length : ℝ) 
  (width : ℝ) 
  (cost_per_sqm : ℝ) 
  (h1 : length = 5.5)
  (h2 : width = 3.75)
  (h3 : cost_per_sqm = 600) : 
  length * width * cost_per_sqm = 12375 := by
sorry

end NUMINAMATH_CALUDE_floor_paving_cost_l4083_408374


namespace NUMINAMATH_CALUDE_susan_spending_l4083_408337

def carnival_spending (initial_budget : ℝ) (food_cost : ℝ) : Prop :=
  let ride_cost : ℝ := 3 * food_cost
  let game_cost : ℝ := (1 / 4) * initial_budget
  let total_spent : ℝ := food_cost + ride_cost + game_cost
  let remaining : ℝ := initial_budget - total_spent
  remaining = 0

theorem susan_spending :
  carnival_spending 80 15 := by
  sorry

end NUMINAMATH_CALUDE_susan_spending_l4083_408337


namespace NUMINAMATH_CALUDE_negative_three_times_two_l4083_408334

theorem negative_three_times_two : (-3) * 2 = -6 := by
  sorry

end NUMINAMATH_CALUDE_negative_three_times_two_l4083_408334


namespace NUMINAMATH_CALUDE_rectangle_area_change_l4083_408348

theorem rectangle_area_change (original_area : ℝ) : 
  original_area = 540 →
  (0.9 * 1.2 * original_area : ℝ) = 583.2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_change_l4083_408348


namespace NUMINAMATH_CALUDE_cube_neg_iff_neg_l4083_408372

theorem cube_neg_iff_neg (x : ℝ) : x^3 < 0 ↔ x < 0 := by sorry

end NUMINAMATH_CALUDE_cube_neg_iff_neg_l4083_408372


namespace NUMINAMATH_CALUDE_largest_among_three_l4083_408396

theorem largest_among_three (sin2 log132 log1213 : ℝ) 
  (h1 : 0 < sin2 ∧ sin2 < 1)
  (h2 : log132 < 0)
  (h3 : log1213 > 1) :
  log1213 = max sin2 (max log132 log1213) :=
sorry

end NUMINAMATH_CALUDE_largest_among_three_l4083_408396


namespace NUMINAMATH_CALUDE_f_84_value_l4083_408324

def is_increasing (f : ℕ+ → ℕ+) : Prop :=
  ∀ n : ℕ+, f (n + 1) > f n

def multiplicative (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, f (m * n) = f m * f n

def special_condition (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n : ℕ+, m ≠ n → m ^ (n : ℕ) = n ^ (m : ℕ) → f m = n ∨ f n = m

theorem f_84_value (f : ℕ+ → ℕ+)
  (h_inc : is_increasing f)
  (h_mult : multiplicative f)
  (h_special : special_condition f) :
  f 84 = 1764 := by
  sorry

end NUMINAMATH_CALUDE_f_84_value_l4083_408324


namespace NUMINAMATH_CALUDE_frog_walk_probability_l4083_408353

/-- Represents a point on the grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- Represents the 6x6 grid -/
def Grid := {p : Point // p.x ≤ 6 ∧ p.y ≤ 6}

/-- Defines the possible jump directions -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Defines a random walk on the grid -/
def RandomWalk := List Direction

/-- Checks if a point is on the boundary of the grid -/
def isBoundary (p : Point) : Bool :=
  p.x = 0 ∨ p.x = 6 ∨ p.y = 0 ∨ p.y = 6

/-- Checks if a point is on the top or bottom horizontal side of the grid -/
def isHorizontalSide (p : Point) : Bool :=
  p.y = 0 ∨ p.y = 6

/-- Calculates the probability of ending on a horizontal side -/
noncomputable def probabilityHorizontalSide (start : Point) : ℚ :=
  sorry

/-- The main theorem to prove -/
theorem frog_walk_probability :
  probabilityHorizontalSide ⟨2, 3⟩ = 8 / 25 := by sorry

end NUMINAMATH_CALUDE_frog_walk_probability_l4083_408353


namespace NUMINAMATH_CALUDE_bike_license_count_l4083_408316

/-- The number of possible letters for a bike license -/
def num_letters : ℕ := 3

/-- The number of digits in a bike license -/
def num_digits : ℕ := 4

/-- The number of possible digits for each position (0-9) -/
def digits_per_position : ℕ := 10

/-- The total number of possible bike licenses -/
def total_licenses : ℕ := num_letters * (digits_per_position ^ num_digits)

theorem bike_license_count : total_licenses = 30000 := by
  sorry

end NUMINAMATH_CALUDE_bike_license_count_l4083_408316


namespace NUMINAMATH_CALUDE_larger_ssr_not_better_fit_l4083_408342

/-- Represents a simple linear regression model -/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ
  x : List ℝ
  y : List ℝ

/-- Calculates the sum of squared residuals for a given model -/
def sumSquaredResiduals (model : LinearRegression) : ℝ :=
  sorry

/-- Represents the goodness of fit of a model -/
def goodnessOfFit (model : LinearRegression) : ℝ :=
  sorry

theorem larger_ssr_not_better_fit (model1 model2 : LinearRegression) :
  sumSquaredResiduals model1 > sumSquaredResiduals model2 →
  goodnessOfFit model1 ≤ goodnessOfFit model2 :=
sorry

end NUMINAMATH_CALUDE_larger_ssr_not_better_fit_l4083_408342


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4083_408321

theorem complex_fraction_simplification :
  let z₁ : ℂ := 3 + 3*I
  let z₂ : ℂ := -1 + 3*I
  z₁ / z₂ = (-1.2 : ℝ) - 1.2*I := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4083_408321
