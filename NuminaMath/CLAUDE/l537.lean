import Mathlib

namespace NUMINAMATH_CALUDE_min_value_T_l537_53765

/-- Given a quadratic inequality with no real solutions and constraints on its coefficients,
    prove that a certain expression T has a minimum value of 4. -/
theorem min_value_T (a b c : ℝ) : 
  (∀ x, (1/a) * x^2 + b*x + c ≥ 0) →  -- No real solutions to the inequality
  a > 0 →
  a * b > 1 → 
  (∀ T, T = 1/(2*(a*b-1)) + (a*(b+2*c))/(a*b-1) → T ≥ 4) ∧ 
  (∃ T, T = 1/(2*(a*b-1)) + (a*(b+2*c))/(a*b-1) ∧ T = 4) :=
by sorry


end NUMINAMATH_CALUDE_min_value_T_l537_53765


namespace NUMINAMATH_CALUDE_expand_expression_l537_53712

theorem expand_expression (x : ℝ) : -2 * (x + 3) * (x - 2) * (x + 1) = -2*x^3 - 4*x^2 + 10*x + 12 := by
  sorry

end NUMINAMATH_CALUDE_expand_expression_l537_53712


namespace NUMINAMATH_CALUDE_complex_exp_form_l537_53704

/-- For the complex number z = 1 + i√3, when expressed in the form re^(iθ), θ = π/3 -/
theorem complex_exp_form (z : ℂ) : z = 1 + Complex.I * Real.sqrt 3 → 
  ∃ (r : ℝ), z = r * Complex.exp (Complex.I * (π / 3)) :=
by sorry

end NUMINAMATH_CALUDE_complex_exp_form_l537_53704


namespace NUMINAMATH_CALUDE_triangle_properties_l537_53776

/-- Triangle ABC with sides a, b, c opposite angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_properties (t : Triangle) 
  (h1 : t.c = 6)
  (h2 : Real.sin t.A - Real.sin t.C = Real.sin (t.A - t.B))
  (h3 : t.b = 2 * Real.sqrt 7) :
  t.B = π / 3 ∧ 
  (t.a * t.c * Real.sin t.B / 2 = 3 * Real.sqrt 3 ∨ 
   t.a * t.c * Real.sin t.B / 2 = 6 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l537_53776


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l537_53770

theorem arctan_equation_solution :
  ∃ x : ℝ, Real.arctan (2 / x) + Real.arctan (3 / x^3) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l537_53770


namespace NUMINAMATH_CALUDE_fred_has_ten_balloons_l537_53792

/-- The number of red balloons Fred has -/
def fred_balloons (total sam dan : ℕ) : ℕ := total - (sam + dan)

/-- Theorem stating that Fred has 10 red balloons -/
theorem fred_has_ten_balloons (total sam dan : ℕ) 
  (h_total : total = 72)
  (h_sam : sam = 46)
  (h_dan : dan = 16) :
  fred_balloons total sam dan = 10 := by
  sorry

end NUMINAMATH_CALUDE_fred_has_ten_balloons_l537_53792


namespace NUMINAMATH_CALUDE_mean_proportional_problem_l537_53784

theorem mean_proportional_problem (x : ℝ) :
  (Real.sqrt (x * 100) = 90.5) → x = 81.9025 := by
  sorry

end NUMINAMATH_CALUDE_mean_proportional_problem_l537_53784


namespace NUMINAMATH_CALUDE_money_distribution_l537_53772

-- Define the variables
variable (A B C : ℕ)

-- State the theorem
theorem money_distribution (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 320) : C = 20 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l537_53772


namespace NUMINAMATH_CALUDE_exponent_multiplication_l537_53728

theorem exponent_multiplication (a : ℝ) : a^2 * a^3 = a^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l537_53728


namespace NUMINAMATH_CALUDE_alex_shirts_l537_53716

theorem alex_shirts (alex joe ben : ℕ) 
  (h1 : joe = alex + 3)
  (h2 : ben = joe + 8)
  (h3 : ben = 15) : 
  alex = 4 := by
  sorry

end NUMINAMATH_CALUDE_alex_shirts_l537_53716


namespace NUMINAMATH_CALUDE_triangle_circles_theorem_l537_53720

/-- Represents a triangular arrangement of circles -/
structure TriangularArrangement where
  total_circles : ℕ
  longest_side_length : ℕ
  shorter_side_rows : List ℕ

/-- Calculates the number of ways to choose three consecutive circles along the longest side -/
def longest_side_choices (arr : TriangularArrangement) : ℕ :=
  (arr.longest_side_length * (arr.longest_side_length + 1)) / 2

/-- Calculates the number of ways to choose three consecutive circles along a shorter side -/
def shorter_side_choices (arr : TriangularArrangement) : ℕ :=
  arr.shorter_side_rows.sum

/-- Calculates the total number of ways to choose three consecutive circles in any direction -/
def total_choices (arr : TriangularArrangement) : ℕ :=
  longest_side_choices arr + 2 * shorter_side_choices arr

/-- The main theorem stating that for the given arrangement, there are 57 ways to choose three consecutive circles -/
theorem triangle_circles_theorem (arr : TriangularArrangement) 
  (h1 : arr.total_circles = 33)
  (h2 : arr.longest_side_length = 6)
  (h3 : arr.shorter_side_rows = [4, 4, 4, 3, 2, 1]) :
  total_choices arr = 57 := by
  sorry


end NUMINAMATH_CALUDE_triangle_circles_theorem_l537_53720


namespace NUMINAMATH_CALUDE_min_distance_between_curves_l537_53709

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x + 1)
noncomputable def g (x : ℝ) : ℝ := Real.sqrt (2 * x - 1)

theorem min_distance_between_curves :
  ∃ (min_dist : ℝ), 
    (∀ (x₁ x₂ : ℝ), f x₁ = g x₂ → |x₂ - x₁| ≥ min_dist) ∧
    (∃ (x₁ x₂ : ℝ), f x₁ = g x₂ ∧ |x₂ - x₁| = min_dist) ∧
    min_dist = (5 + Real.log 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_between_curves_l537_53709


namespace NUMINAMATH_CALUDE_exactly_three_numbers_l537_53786

/-- A two-digit number -/
def TwoDigitNumber := { n : ℕ // 10 ≤ n ∧ n < 100 }

/-- The tens digit of a two-digit number -/
def tens_digit (n : TwoDigitNumber) : ℕ := n.val / 10

/-- The units digit of a two-digit number -/
def units_digit (n : TwoDigitNumber) : ℕ := n.val % 10

/-- The sum of digits of a two-digit number -/
def sum_of_digits (n : TwoDigitNumber) : ℕ := tens_digit n + units_digit n

/-- Predicate for numbers satisfying the given conditions -/
def satisfies_conditions (n : TwoDigitNumber) : Prop :=
  (n.val - sum_of_digits n) % 10 = 2 ∧ n.val % 3 = 0

/-- The main theorem stating there are exactly 3 numbers satisfying the conditions -/
theorem exactly_three_numbers :
  ∃! (s : Finset TwoDigitNumber), (∀ n ∈ s, satisfies_conditions n) ∧ s.card = 3 :=
sorry

end NUMINAMATH_CALUDE_exactly_three_numbers_l537_53786


namespace NUMINAMATH_CALUDE_geometric_progression_solution_l537_53796

theorem geometric_progression_solution :
  ∃! x : ℚ, ((-10 + x)^2 = (-30 + x) * (40 + x)) ∧ x = 130/3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_solution_l537_53796


namespace NUMINAMATH_CALUDE_wire_connections_count_l537_53778

/-- The number of wire segments --/
def n : ℕ := 5

/-- The number of possible orientations for each segment --/
def orientations : ℕ := 2

/-- The total number of ways to connect the wire segments --/
def total_connections : ℕ := n.factorial * orientations ^ n

theorem wire_connections_count : total_connections = 3840 := by
  sorry

end NUMINAMATH_CALUDE_wire_connections_count_l537_53778


namespace NUMINAMATH_CALUDE_initial_sony_games_l537_53724

/-- The number of Sony games Kelly gives away -/
def games_given_away : ℕ := 101

/-- The number of Sony games Kelly has left after giving away games -/
def games_left : ℕ := 31

/-- The initial number of Sony games Kelly has -/
def initial_games : ℕ := games_given_away + games_left

theorem initial_sony_games : initial_games = 132 := by sorry

end NUMINAMATH_CALUDE_initial_sony_games_l537_53724


namespace NUMINAMATH_CALUDE_condition_p_necessary_not_sufficient_l537_53787

theorem condition_p_necessary_not_sufficient :
  (∀ a : ℝ, (|a| ≤ 1 → a ≤ 1)) ∧
  (∃ a : ℝ, a ≤ 1 ∧ ¬(|a| ≤ 1)) :=
by sorry

end NUMINAMATH_CALUDE_condition_p_necessary_not_sufficient_l537_53787


namespace NUMINAMATH_CALUDE_arctan_equation_solution_l537_53719

theorem arctan_equation_solution :
  ∀ x : ℝ, Real.arctan (1 / x) + Real.arctan (1 / x^2) = π / 4 → x = 2 := by
sorry

end NUMINAMATH_CALUDE_arctan_equation_solution_l537_53719


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l537_53734

/-- The eccentricity of a hyperbola with equation x^2 - y^2/m = 1 is 2 if and only if m = 3 -/
theorem hyperbola_eccentricity (m : ℝ) :
  (∀ x y : ℝ, x^2 - y^2/m = 1) →
  (∃ e : ℝ, e = 2 ∧ e^2 = 1 + 1/m) →
  m = 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l537_53734


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l537_53703

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 2) + 3
  f 2 = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l537_53703


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l537_53737

theorem concentric_circles_radii_difference
  (r R : ℝ)
  (h_positive : r > 0)
  (h_ratio : π * R^2 = 4 * π * r^2) :
  R - r = r :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l537_53737


namespace NUMINAMATH_CALUDE_cody_additional_tickets_l537_53705

/-- Calculates the number of additional tickets won given initial tickets, tickets spent, and final tickets. -/
def additional_tickets_won (initial_tickets spent_tickets final_tickets : ℕ) : ℕ :=
  final_tickets - (initial_tickets - spent_tickets)

/-- Proves that Cody won 6 additional tickets given the problem conditions. -/
theorem cody_additional_tickets :
  let initial_tickets := 49
  let spent_tickets := 25
  let final_tickets := 30
  additional_tickets_won initial_tickets spent_tickets final_tickets = 6 := by
  sorry

end NUMINAMATH_CALUDE_cody_additional_tickets_l537_53705


namespace NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l537_53742

theorem x_eq_one_sufficient_not_necessary_for_x_squared_eq_one :
  (∃ x : ℝ, x = 1 → x^2 = 1) ∧
  (∃ x : ℝ, x^2 = 1 ∧ x ≠ 1) := by
  sorry

end NUMINAMATH_CALUDE_x_eq_one_sufficient_not_necessary_for_x_squared_eq_one_l537_53742


namespace NUMINAMATH_CALUDE_jane_apple_purchase_l537_53717

/-- The price of one apple in dollars -/
def apple_price : ℝ := 2

/-- The amount Jane has to spend in dollars -/
def jane_budget : ℝ := 2

/-- There is no bulk discount -/
axiom no_bulk_discount : ∀ (n : ℕ), n * apple_price = jane_budget → n = 1

/-- The number of apples Jane can buy with her budget -/
def apples_bought : ℕ := 1

theorem jane_apple_purchase :
  apples_bought * apple_price = jane_budget :=
sorry

end NUMINAMATH_CALUDE_jane_apple_purchase_l537_53717


namespace NUMINAMATH_CALUDE_min_value_theorem_l537_53743

theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (b * (a - b)) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l537_53743


namespace NUMINAMATH_CALUDE_cube_root_125_fourth_root_256_square_root_16_l537_53732

theorem cube_root_125_fourth_root_256_square_root_16 : 
  (125 : ℝ) ^ (1/3) * (256 : ℝ) ^ (1/4) * (16 : ℝ) ^ (1/2) = 80 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_125_fourth_root_256_square_root_16_l537_53732


namespace NUMINAMATH_CALUDE_total_cost_beef_vegetables_l537_53769

/-- The total cost of beef and vegetables given their weights and prices -/
theorem total_cost_beef_vegetables 
  (beef_weight : ℝ) 
  (vegetable_weight : ℝ) 
  (vegetable_price : ℝ) 
  (beef_price_multiplier : ℝ) : 
  beef_weight = 4 →
  vegetable_weight = 6 →
  vegetable_price = 2 →
  beef_price_multiplier = 3 →
  beef_weight * (vegetable_price * beef_price_multiplier) + vegetable_weight * vegetable_price = 36 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_beef_vegetables_l537_53769


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l537_53767

theorem geometric_sequence_sum (a r : ℝ) : 
  (a + a * r = 15) →
  (a * (1 - r^6) / (1 - r) = 195) →
  (a * (1 - r^4) / (1 - r) = 82) :=
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l537_53767


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l537_53751

theorem binomial_expansion_coefficient (x : ℝ) (x_ne_zero : x ≠ 0) :
  let expansion := (x^2 - 1/x)^5
  let second_term_coefficient := Finset.sum (Finset.range 6) (fun k => 
    if k = 1 then (-1)^k * (Nat.choose 5 k) * x^(10 - 3*k)
    else 0)
  second_term_coefficient = -5 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l537_53751


namespace NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l537_53749

theorem ratio_of_sum_to_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_sum_to_difference_l537_53749


namespace NUMINAMATH_CALUDE_zhang_slower_than_li_l537_53793

theorem zhang_slower_than_li :
  let zhang_efficiency : ℚ := 5 / 8
  let li_efficiency : ℚ := 3 / 4
  zhang_efficiency < li_efficiency :=
by
  sorry

end NUMINAMATH_CALUDE_zhang_slower_than_li_l537_53793


namespace NUMINAMATH_CALUDE_percentage_of_percentage_l537_53722

theorem percentage_of_percentage (amount : ℝ) : (5 / 100) * ((25 / 100) * amount) = 20 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_percentage_of_percentage_l537_53722


namespace NUMINAMATH_CALUDE_art_supplies_problem_l537_53713

/-- Cost of one box of brushes in yuan -/
def brush_cost : ℕ := 17

/-- Cost of one canvas in yuan -/
def canvas_cost : ℕ := 15

/-- Total number of items to purchase -/
def total_items : ℕ := 10

/-- Maximum total cost in yuan -/
def max_total_cost : ℕ := 157

/-- Cost of 2 boxes of brushes and 4 canvases in yuan -/
def cost_2b_4c : ℕ := 94

/-- Cost of 4 boxes of brushes and 2 canvases in yuan -/
def cost_4b_2c : ℕ := 98

theorem art_supplies_problem :
  (2 * brush_cost + 4 * canvas_cost = cost_2b_4c) ∧
  (4 * brush_cost + 2 * canvas_cost = cost_4b_2c) ∧
  (∀ m : ℕ, m ≥ 7 → brush_cost * (total_items - m) + canvas_cost * m ≤ max_total_cost) ∧
  (brush_cost * 2 + canvas_cost * 8 < brush_cost * 3 + canvas_cost * 7) := by
  sorry

end NUMINAMATH_CALUDE_art_supplies_problem_l537_53713


namespace NUMINAMATH_CALUDE_sqrt_calculation_l537_53781

theorem sqrt_calculation : 
  Real.sqrt 27 / (Real.sqrt 3 / 2) * (2 * Real.sqrt 2) - 6 * Real.sqrt 2 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_calculation_l537_53781


namespace NUMINAMATH_CALUDE_two_intersection_points_l537_53747

-- Define the lines
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 12 = 0
def line2 (x y : ℝ) : Prop := 5 * x - 2 * y - 10 = 0
def line3 (x : ℝ) : Prop := x = 3
def line4 (y : ℝ) : Prop := y = 1

-- Define an intersection point
def is_intersection_point (x y : ℝ) : Prop :=
  (line1 x y ∧ line2 x y) ∨
  (line1 x y ∧ line3 x) ∨
  (line1 x y ∧ line4 y) ∨
  (line2 x y ∧ line3 x) ∨
  (line2 x y ∧ line4 y) ∨
  (line3 x ∧ line4 y)

-- Theorem: There are exactly two distinct intersection points
theorem two_intersection_points :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    is_intersection_point x₁ y₁ ∧
    is_intersection_point x₂ y₂ ∧
    (x₁ ≠ x₂ ∨ y₁ ≠ y₂) ∧
    (∀ (x y : ℝ), is_intersection_point x y → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂)) :=
sorry

end NUMINAMATH_CALUDE_two_intersection_points_l537_53747


namespace NUMINAMATH_CALUDE_wire_length_proof_l537_53759

theorem wire_length_proof (piece1 piece2 : ℝ) : 
  piece1 = 14 → 
  piece2 = 16 → 
  piece2 = piece1 + 2 → 
  piece1 + piece2 = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_wire_length_proof_l537_53759


namespace NUMINAMATH_CALUDE_cost_price_calculation_l537_53750

/-- Represents a type of cloth with its sales information -/
structure ClothType where
  quantity : ℕ      -- Quantity sold in meters
  totalPrice : ℕ    -- Total selling price in Rs.
  profitPerMeter : ℕ -- Profit per meter in Rs.

/-- Calculates the cost price per meter for a given cloth type -/
def costPricePerMeter (cloth : ClothType) : ℕ :=
  cloth.totalPrice / cloth.quantity - cloth.profitPerMeter

/-- The trader's cloth inventory -/
def traderInventory : List ClothType :=
  [
    { quantity := 85, totalPrice := 8500, profitPerMeter := 15 },  -- Type A
    { quantity := 120, totalPrice := 10200, profitPerMeter := 12 }, -- Type B
    { quantity := 60, totalPrice := 4200, profitPerMeter := 10 }   -- Type C
  ]

theorem cost_price_calculation (inventory : List ClothType) :
  ∀ cloth ∈ inventory,
    costPricePerMeter cloth =
      cloth.totalPrice / cloth.quantity - cloth.profitPerMeter :=
by
  sorry

#eval traderInventory.map costPricePerMeter

end NUMINAMATH_CALUDE_cost_price_calculation_l537_53750


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l537_53739

theorem imaginary_part_of_z (z : ℂ) (h : z / (1 - Complex.I) = 3 + Complex.I) : 
  z.im = -2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l537_53739


namespace NUMINAMATH_CALUDE_total_school_supplies_cost_l537_53738

-- Define the quantities and prices
def haley_paper_reams : ℕ := 2
def haley_paper_price : ℚ := 3.5
def sister_paper_reams : ℕ := 3
def sister_paper_price : ℚ := 4.25
def haley_pens : ℕ := 5
def haley_pen_price : ℚ := 1.25
def sister_pens : ℕ := 8
def sister_pen_price : ℚ := 1.5

-- Define the theorem
theorem total_school_supplies_cost :
  (haley_paper_reams : ℚ) * haley_paper_price +
  (sister_paper_reams : ℚ) * sister_paper_price +
  (haley_pens : ℚ) * haley_pen_price +
  (sister_pens : ℚ) * sister_pen_price = 38 :=
by sorry

end NUMINAMATH_CALUDE_total_school_supplies_cost_l537_53738


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l537_53771

/-- A regular polygon with an exterior angle of 18 degrees has 20 sides and interior angles of 162 degrees. -/
theorem regular_polygon_properties :
  ∀ (n : ℕ) (exterior_angle interior_angle : ℝ),
  exterior_angle = 18 →
  n * exterior_angle = 360 →
  interior_angle = (n - 2 : ℝ) * 180 / n →
  n = 20 ∧ interior_angle = 162 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l537_53771


namespace NUMINAMATH_CALUDE_ring_toss_total_earnings_l537_53777

theorem ring_toss_total_earnings (first_44_days : ℕ) (remaining_10_days : ℕ) (total : ℕ) :
  first_44_days = 382 →
  remaining_10_days = 374 →
  total = first_44_days + remaining_10_days →
  total = 756 := by sorry

end NUMINAMATH_CALUDE_ring_toss_total_earnings_l537_53777


namespace NUMINAMATH_CALUDE_gcd_15n_plus_4_9n_plus_2_max_2_l537_53788

theorem gcd_15n_plus_4_9n_plus_2_max_2 :
  (∃ n : ℕ+, Nat.gcd (15 * n + 4) (9 * n + 2) = 2) ∧
  (∀ n : ℕ+, Nat.gcd (15 * n + 4) (9 * n + 2) ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_gcd_15n_plus_4_9n_plus_2_max_2_l537_53788


namespace NUMINAMATH_CALUDE_abs_sum_lt_sum_abs_when_product_negative_l537_53744

theorem abs_sum_lt_sum_abs_when_product_negative (a b : ℝ) :
  a * b < 0 → |a + b| < |a| + |b| := by sorry

end NUMINAMATH_CALUDE_abs_sum_lt_sum_abs_when_product_negative_l537_53744


namespace NUMINAMATH_CALUDE_bob_corn_harvest_l537_53753

/-- Calculates the number of whole bushels of corn harvested given the number of rows, stalks per row, and stalks per bushel. -/
def cornHarvest (rows : ℕ) (stalksPerRow : ℕ) (stalksPerBushel : ℕ) : ℕ :=
  (rows * stalksPerRow) / stalksPerBushel

theorem bob_corn_harvest :
  cornHarvest 7 92 9 = 71 := by
  sorry

end NUMINAMATH_CALUDE_bob_corn_harvest_l537_53753


namespace NUMINAMATH_CALUDE_clock_initial_time_l537_53774

/-- Represents a time of day with hours and minutes -/
structure Time where
  hours : ℕ
  minutes : ℕ
  valid : hours < 24 ∧ minutes < 60

/-- Calculates the total minutes from midnight for a given time -/
def totalMinutes (t : Time) : ℕ :=
  t.hours * 60 + t.minutes

/-- Represents the properties of the clock in the problem -/
structure Clock where
  initialTime : Time
  gainPerHour : ℕ
  totalGainBy6PM : ℕ

/-- The theorem to be proved -/
theorem clock_initial_time (c : Clock)
  (morning : c.initialTime.hours < 12)
  (gain_rate : c.gainPerHour = 5)
  (total_gain : c.totalGainBy6PM = 35) :
  c.initialTime.hours = 11 ∧ c.initialTime.minutes = 55 := by
  sorry


end NUMINAMATH_CALUDE_clock_initial_time_l537_53774


namespace NUMINAMATH_CALUDE_percent_of_y_l537_53764

theorem percent_of_y (y : ℝ) (h : y > 0) : ((8 * y) / 20 + (3 * y) / 10) / y = 0.7 := by
  sorry

end NUMINAMATH_CALUDE_percent_of_y_l537_53764


namespace NUMINAMATH_CALUDE_correct_calculation_l537_53711

theorem correct_calculation (a b : ℝ) : 3 * a * b + 2 * a * b = 5 * a * b := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l537_53711


namespace NUMINAMATH_CALUDE_cylinder_surface_area_and_volume_l537_53766

/-- Represents a right circular cylinder -/
structure RightCircularCylinder where
  radius : ℝ
  height : ℝ

/-- Properties of the cylinder -/
def CylinderProperties (c : RightCircularCylinder) : Prop :=
  let lateral_area := 2 * Real.pi * c.radius * c.height
  let base_area := Real.pi * c.radius ^ 2
  lateral_area / base_area = 5 / 3 ∧
  (4 * c.radius ^ 2 + c.height ^ 2) = 39 ^ 2

/-- Theorem statement -/
theorem cylinder_surface_area_and_volume 
  (c : RightCircularCylinder) 
  (h : CylinderProperties c) : 
  (2 * Real.pi * c.radius * c.height + 2 * Real.pi * c.radius ^ 2 = 1188 * Real.pi) ∧
  (Real.pi * c.radius ^ 2 * c.height = 4860 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_and_volume_l537_53766


namespace NUMINAMATH_CALUDE_square_perimeter_inequality_l537_53773

theorem square_perimeter_inequality (t₁ t₂ t₃ t₄ k₁ k₂ k₃ k₄ : ℝ) 
  (h₁ : t₁ > 0) (h₂ : t₂ > 0) (h₃ : t₃ > 0) (h₄ : t₄ > 0)
  (hk₁ : k₁ = 4 * Real.sqrt t₁)
  (hk₂ : k₂ = 4 * Real.sqrt t₂)
  (hk₃ : k₃ = 4 * Real.sqrt t₃)
  (hk₄ : k₄ = 4 * Real.sqrt t₄)
  (ht : t₁ + t₂ + t₃ = t₄) :
  k₁ + k₂ + k₃ ≤ k₄ * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_inequality_l537_53773


namespace NUMINAMATH_CALUDE_product_inspection_probabilities_l537_53701

def total_units : ℕ := 6
def inspected_units : ℕ := 2
def first_grade_units : ℕ := 3
def second_grade_units : ℕ := 2
def defective_units : ℕ := 1

def probability_both_first_grade : ℚ := 1 / 5
def probability_one_second_grade : ℚ := 8 / 15

def probability_at_most_one_defective (x : ℕ) : ℚ :=
  (Nat.choose x 1 * Nat.choose (total_units - x) 1 + Nat.choose (total_units - x) 2) /
  Nat.choose total_units inspected_units

theorem product_inspection_probabilities :
  (probability_both_first_grade = 1 / 5) ∧
  (probability_one_second_grade = 8 / 15) ∧
  (∀ x : ℕ, x ≤ total_units →
    (probability_at_most_one_defective x ≥ 4 / 5 → x ≤ 3)) :=
by sorry

end NUMINAMATH_CALUDE_product_inspection_probabilities_l537_53701


namespace NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l537_53740

/-- A random variable following a binomial distribution B(n,p) -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

theorem binomial_distribution_unique_parameters
  (X : BinomialRV)
  (h_expectation : expectation X = 1.6)
  (h_variance : variance X = 1.28) :
  X.n = 8 ∧ X.p = 0.2 := by sorry

end NUMINAMATH_CALUDE_binomial_distribution_unique_parameters_l537_53740


namespace NUMINAMATH_CALUDE_solution_uniqueness_l537_53791

theorem solution_uniqueness (x y : ℝ) : x^2 - 2*x + y^2 + 6*y + 10 = 0 → x = 1 ∧ y = -3 := by
  sorry

end NUMINAMATH_CALUDE_solution_uniqueness_l537_53791


namespace NUMINAMATH_CALUDE_y_increases_with_x_l537_53726

theorem y_increases_with_x (m : ℝ) (x y : ℝ → ℝ) :
  (∀ t, y t = (m^2 + 2) * x t) →
  StrictMono y :=
sorry

end NUMINAMATH_CALUDE_y_increases_with_x_l537_53726


namespace NUMINAMATH_CALUDE_scarves_per_box_chloes_scarves_l537_53700

theorem scarves_per_box (num_boxes : ℕ) (mittens_per_box : ℕ) (total_clothing : ℕ) : ℕ :=
  let total_mittens := num_boxes * mittens_per_box
  let total_scarves := total_clothing - total_mittens
  let scarves_per_box := total_scarves / num_boxes
  scarves_per_box

theorem chloes_scarves :
  scarves_per_box 4 6 32 = 2 := by
  sorry

end NUMINAMATH_CALUDE_scarves_per_box_chloes_scarves_l537_53700


namespace NUMINAMATH_CALUDE_rain_probability_l537_53762

theorem rain_probability (p : ℝ) (h : p = 1 / 2) : 
  1 - (1 - p)^4 = 15 / 16 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l537_53762


namespace NUMINAMATH_CALUDE_soccer_balls_with_holes_l537_53718

theorem soccer_balls_with_holes (total_soccer : ℕ) (total_basketball : ℕ) (basketball_with_holes : ℕ) (total_without_holes : ℕ) :
  total_soccer = 40 →
  total_basketball = 15 →
  basketball_with_holes = 7 →
  total_without_holes = 18 →
  total_soccer - (total_without_holes - (total_basketball - basketball_with_holes)) = 30 := by
sorry

end NUMINAMATH_CALUDE_soccer_balls_with_holes_l537_53718


namespace NUMINAMATH_CALUDE_increase_by_percentage_l537_53710

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (result : ℝ) : 
  initial = 80 → percentage = 150 → result = initial * (1 + percentage / 100) → result = 200 := by
  sorry

end NUMINAMATH_CALUDE_increase_by_percentage_l537_53710


namespace NUMINAMATH_CALUDE_f_properties_l537_53741

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.exp x) / (1 + Real.exp x) - 1/2

-- Theorem statement
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x y, x < y → f x < f y)  -- f is increasing
  := by sorry

end NUMINAMATH_CALUDE_f_properties_l537_53741


namespace NUMINAMATH_CALUDE_gcd_lcm_equalities_l537_53735

/-- Define * as the greatest common divisor operation -/
def gcd_op (a b : ℕ) : ℕ := Nat.gcd a b

/-- Define ∘ as the least common multiple operation -/
def lcm_op (a b : ℕ) : ℕ := Nat.lcm a b

/-- The main theorem stating the equalities for gcd and lcm operations -/
theorem gcd_lcm_equalities (a b c : ℕ) :
  (gcd_op a (lcm_op b c) = lcm_op (gcd_op a b) (gcd_op a c)) ∧
  (lcm_op a (gcd_op b c) = gcd_op (lcm_op a b) (lcm_op a c)) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_equalities_l537_53735


namespace NUMINAMATH_CALUDE_cupcakes_left_over_l537_53798

/-- The number of cupcakes Quinton brought to school -/
def total_cupcakes : ℕ := 80

/-- The number of students in Ms. Delmont's class -/
def ms_delmont_students : ℕ := 18

/-- The number of students in Mrs. Donnelly's class -/
def mrs_donnelly_students : ℕ := 16

/-- The number of school custodians -/
def custodians : ℕ := 3

/-- The number of Quinton's favorite teachers -/
def favorite_teachers : ℕ := 5

/-- The number of other classmates who received cupcakes -/
def other_classmates : ℕ := 10

/-- The number of cupcakes given to each favorite teacher -/
def cupcakes_per_favorite_teacher : ℕ := 2

/-- The total number of cupcakes given away -/
def cupcakes_given_away : ℕ :=
  ms_delmont_students + mrs_donnelly_students + 2 + 1 + 1 + custodians +
  (favorite_teachers * cupcakes_per_favorite_teacher) + other_classmates

/-- Theorem stating the number of cupcakes Quinton has left over -/
theorem cupcakes_left_over : total_cupcakes - cupcakes_given_away = 19 := by
  sorry

end NUMINAMATH_CALUDE_cupcakes_left_over_l537_53798


namespace NUMINAMATH_CALUDE_orchard_difference_l537_53760

/-- Represents the number of trees of each type in an orchard -/
structure Orchard where
  orange : ℕ
  lemon : ℕ
  apple : ℕ
  apricot : ℕ

/-- Calculates the total number of trees in an orchard -/
def totalTrees (o : Orchard) : ℕ :=
  o.orange + o.lemon + o.apple + o.apricot

theorem orchard_difference : 
  let ahmed : Orchard := { orange := 8, lemon := 6, apple := 4, apricot := 0 }
  let hassan : Orchard := { orange := 2, lemon := 5, apple := 1, apricot := 3 }
  totalTrees ahmed - totalTrees hassan = 7 := by
  sorry

end NUMINAMATH_CALUDE_orchard_difference_l537_53760


namespace NUMINAMATH_CALUDE_ice_cream_flavors_count_l537_53702

/-- The number of ways to distribute n indistinguishable items into k distinguishable containers -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of flavors that can be created by combining 4 scoops of 3 basic flavors -/
def ice_cream_flavors : ℕ := distribute 4 3

theorem ice_cream_flavors_count : ice_cream_flavors = 15 := by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_count_l537_53702


namespace NUMINAMATH_CALUDE_amp_composition_l537_53797

-- Define the & operation (postfix)
def postAmp (x : ℝ) : ℝ := 9 - x

-- Define the & operation (prefix)
def preAmp (x : ℝ) : ℝ := x - 9

-- Theorem statement
theorem amp_composition : preAmp (postAmp 15) = -15 := by
  sorry

end NUMINAMATH_CALUDE_amp_composition_l537_53797


namespace NUMINAMATH_CALUDE_f_of_g_10_l537_53782

/-- The function g(x) = 4x + 10 -/
def g (x : ℝ) : ℝ := 4 * x + 10

/-- The function f(x) = 6x - 12 -/
def f (x : ℝ) : ℝ := 6 * x - 12

/-- Theorem: f(g(10)) = 288 -/
theorem f_of_g_10 : f (g 10) = 288 := by sorry

end NUMINAMATH_CALUDE_f_of_g_10_l537_53782


namespace NUMINAMATH_CALUDE_floor_paving_cost_l537_53775

/-- Calculates the total cost of paving a floor with different types of slabs -/
theorem floor_paving_cost (room_length room_width : ℝ)
  (square_slab_side square_slab_cost square_slab_percentage : ℝ)
  (rect_slab_length rect_slab_width rect_slab_cost rect_slab_percentage : ℝ)
  (tri_slab_height tri_slab_base tri_slab_cost tri_slab_percentage : ℝ) :
  room_length = 5.5 →
  room_width = 3.75 →
  square_slab_side = 1 →
  square_slab_cost = 800 →
  square_slab_percentage = 0.4 →
  rect_slab_length = 1.5 →
  rect_slab_width = 1 →
  rect_slab_cost = 1000 →
  rect_slab_percentage = 0.35 →
  tri_slab_height = 1 →
  tri_slab_base = 1 →
  tri_slab_cost = 1200 →
  tri_slab_percentage = 0.25 →
  square_slab_percentage + rect_slab_percentage + tri_slab_percentage = 1 →
  (room_length * room_width) * 
    (square_slab_percentage * square_slab_cost + 
     rect_slab_percentage * rect_slab_cost + 
     tri_slab_percentage * tri_slab_cost) = 20006.25 := by
  sorry

end NUMINAMATH_CALUDE_floor_paving_cost_l537_53775


namespace NUMINAMATH_CALUDE_alpha_value_l537_53723

theorem alpha_value (α β : Real) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2)
  (h3 : Real.tan (α - β) = 1/2)
  (h4 : Real.tan β = 1/3) : 
  α = π/4 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l537_53723


namespace NUMINAMATH_CALUDE_total_sodas_sold_restaurant_soda_sales_l537_53725

/-- Theorem: Total sodas sold given diet soda count and ratio of regular to diet --/
theorem total_sodas_sold (diet_count : ℕ) (regular_ratio diet_ratio : ℕ) : ℕ :=
  let regular_count := (regular_ratio * diet_count) / diet_ratio
  diet_count + regular_count

/-- Proof of the specific problem --/
theorem restaurant_soda_sales : total_sodas_sold 28 9 7 = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_sodas_sold_restaurant_soda_sales_l537_53725


namespace NUMINAMATH_CALUDE_jeff_truck_count_l537_53763

/-- The number of trucks Jeff has -/
def num_trucks : ℕ := sorry

/-- The number of cars Jeff has -/
def num_cars : ℕ := sorry

/-- The total number of vehicles Jeff has -/
def total_vehicles : ℕ := 60

theorem jeff_truck_count :
  (num_cars = 2 * num_trucks) ∧
  (num_cars + num_trucks = total_vehicles) →
  num_trucks = 20 := by sorry

end NUMINAMATH_CALUDE_jeff_truck_count_l537_53763


namespace NUMINAMATH_CALUDE_original_data_set_properties_l537_53780

/-- Represents a data set with its average and variance -/
structure DataSet where
  average : ℝ
  variance : ℝ

/-- The transformation applied to the original data set -/
def decrease_by_80 (d : DataSet) : DataSet :=
  { average := d.average - 80, variance := d.variance }

/-- Theorem stating the relationship between the original and transformed data sets -/
theorem original_data_set_properties (transformed : DataSet)
  (h1 : transformed = decrease_by_80 { average := 81.2, variance := 4.4 })
  (h2 : transformed.average = 1.2)
  (h3 : transformed.variance = 4.4) :
  ∃ (original : DataSet), original.average = 81.2 ∧ original.variance = 4.4 :=
sorry

end NUMINAMATH_CALUDE_original_data_set_properties_l537_53780


namespace NUMINAMATH_CALUDE_olympiad_sheet_distribution_l537_53731

theorem olympiad_sheet_distribution (n : ℕ) :
  let initial_total := 2 + 3 + 1 + 1
  let final_total := initial_total + 2 * n
  ¬ ∃ (k : ℕ), final_total = 4 * k := by
  sorry

end NUMINAMATH_CALUDE_olympiad_sheet_distribution_l537_53731


namespace NUMINAMATH_CALUDE_grade10_sample_size_l537_53761

/-- Calculates the number of students to be selected from a specific grade in a stratified random sample. -/
def stratifiedSampleSize (gradeSize : ℕ) (totalSize : ℕ) (sampleSize : ℕ) : ℕ :=
  (gradeSize * sampleSize) / totalSize

/-- The number of students to be selected from grade 10 in a stratified random sample is 40. -/
theorem grade10_sample_size :
  stratifiedSampleSize 1200 3000 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_grade10_sample_size_l537_53761


namespace NUMINAMATH_CALUDE_max_value_theorem_l537_53714

theorem max_value_theorem (a b c : ℝ) (h : a^2 + b^2 + c^2 = 1) :
  ∃ (M : ℝ), M = 3 ∧ ∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → 3*x*y - 3*y*z + 2*z^2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_l537_53714


namespace NUMINAMATH_CALUDE_intersection_sum_l537_53794

-- Define the lines l₁ and l₂
def l₁ (a : ℝ) (x y : ℝ) : Prop := a * x + y + 1 = 0
def l₂ (b : ℝ) (x y : ℝ) : Prop := 2 * x - b * y - 1 = 0

-- Theorem statement
theorem intersection_sum (a b : ℝ) : 
  (l₁ a 1 1 ∧ l₂ b 1 1) → a + b = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l537_53794


namespace NUMINAMATH_CALUDE_arithmetic_progression_divisibility_l537_53706

/-- Given three integers a, b, c that form an arithmetic progression with a common difference of 7,
    and one of them is divisible by 7, their product abc is divisible by 294. -/
theorem arithmetic_progression_divisibility (a b c : ℤ) 
  (h1 : b - a = 7)
  (h2 : c - b = 7)
  (h3 : (∃ k : ℤ, a = 7 * k) ∨ (∃ k : ℤ, b = 7 * k) ∨ (∃ k : ℤ, c = 7 * k)) :
  ∃ m : ℤ, a * b * c = 294 * m := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_progression_divisibility_l537_53706


namespace NUMINAMATH_CALUDE_count_tuples_divisible_sum_l537_53756

theorem count_tuples_divisible_sum : 
  let n := 2012
  let f : Fin n → ℕ → ℕ := fun i x => (i.val + 1) * x
  (Finset.univ.filter (fun t : Fin n → Fin n => 
    (Finset.sum Finset.univ (fun i => f i (t i).val)) % n = 0)).card = n^(n-1) := by
  sorry

end NUMINAMATH_CALUDE_count_tuples_divisible_sum_l537_53756


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l537_53795

def scores : List ℝ := [93, 87, 90, 96, 88, 94]

theorem arithmetic_mean_of_scores :
  (scores.sum / scores.length : ℝ) = 91.333 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_scores_l537_53795


namespace NUMINAMATH_CALUDE_family_size_problem_l537_53790

theorem family_size_problem (avg_age_before avg_age_now baby_age : ℝ) 
  (h1 : avg_age_before = 17)
  (h2 : avg_age_now = 17)
  (h3 : baby_age = 2) : 
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℝ) * avg_age_before + (n : ℝ) * 3 + baby_age = (n + 1 : ℝ) * avg_age_now ∧
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_family_size_problem_l537_53790


namespace NUMINAMATH_CALUDE_expression_proof_l537_53733

/-- An expression that, when divided by (3x + 29), equals 2 -/
def E (x : ℝ) : ℝ := 6 * x + 58

theorem expression_proof (x : ℝ) : E x / (3 * x + 29) = 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_proof_l537_53733


namespace NUMINAMATH_CALUDE_stock_market_investment_l537_53779

theorem stock_market_investment (x : ℝ) (h : x > 0) : 
  x * (1 + 0.8) * (1 - 0.3) > x := by
  sorry

end NUMINAMATH_CALUDE_stock_market_investment_l537_53779


namespace NUMINAMATH_CALUDE_function_characterization_l537_53752

/-- A function from positive integers to non-negative integers -/
def PositiveToNonNegative := ℕ+ → ℕ

/-- The p-adic valuation of a positive integer -/
noncomputable def vp (p : ℕ+) (n : ℕ+) : ℕ := sorry

theorem function_characterization 
  (f : PositiveToNonNegative) 
  (h1 : ∃ n, f n ≠ 0)
  (h2 : ∀ x y, f (x * y) = f x + f y)
  (h3 : ∃ S : Set ℕ+, Set.Infinite S ∧ ∀ n ∈ S, ∀ k < n, f k = f (n - k)) :
  ∃ (N : ℕ+) (p : ℕ+), Nat.Prime p ∧ ∀ n, f n = N * vp p n :=
sorry

end NUMINAMATH_CALUDE_function_characterization_l537_53752


namespace NUMINAMATH_CALUDE_x_value_proof_l537_53746

theorem x_value_proof : ∃ X : ℝ, 
  X * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 1200.0000000000002 ∧ 
  abs (X - 0.6) < 0.0000000000000001 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l537_53746


namespace NUMINAMATH_CALUDE_afternoon_sales_l537_53715

/-- Represents the sales of pears by a salesman in a day -/
structure PearSales where
  morning : ℕ
  afternoon : ℕ
  total : ℕ

/-- Theorem stating the afternoon sales given the conditions -/
theorem afternoon_sales (sales : PearSales) 
  (h1 : sales.afternoon = 2 * sales.morning)
  (h2 : sales.total = sales.morning + sales.afternoon)
  (h3 : sales.total = 420) :
  sales.afternoon = 280 := by
  sorry

#check afternoon_sales

end NUMINAMATH_CALUDE_afternoon_sales_l537_53715


namespace NUMINAMATH_CALUDE_weeds_never_cover_entire_field_l537_53755

/-- Represents a 10x10 grid -/
def Grid := Fin 10 → Fin 10 → Bool

/-- The initial state of the grid with 9 occupied cells -/
def initial_state : Grid := sorry

/-- Checks if a cell is adjacent to at least two occupied cells -/
def has_two_adjacent_occupied (g : Grid) (i j : Fin 10) : Bool := sorry

/-- The next state of the grid after one step of spreading -/
def next_state (g : Grid) : Grid := sorry

/-- The state of the grid after n steps of spreading -/
def state_after_n_steps (n : ℕ) : Grid := sorry

/-- Counts the number of occupied cells in the grid -/
def count_occupied (g : Grid) : ℕ := sorry

theorem weeds_never_cover_entire_field :
  ∀ n : ℕ, count_occupied (state_after_n_steps n) < 100 := by sorry

end NUMINAMATH_CALUDE_weeds_never_cover_entire_field_l537_53755


namespace NUMINAMATH_CALUDE_beautiful_point_coordinates_l537_53768

/-- A point (x,y) is called a "beautiful point" if x + y = x * y -/
def is_beautiful_point (x y : ℝ) : Prop := x + y = x * y

/-- The distance of a point (x,y) from the y-axis is the absolute value of x -/
def distance_from_y_axis (x : ℝ) : ℝ := |x|

theorem beautiful_point_coordinates :
  ∀ x y : ℝ, is_beautiful_point x y → distance_from_y_axis x = 2 →
  ((x = 2 ∧ y = 2) ∨ (x = -2 ∧ y = 2/3)) :=
sorry

end NUMINAMATH_CALUDE_beautiful_point_coordinates_l537_53768


namespace NUMINAMATH_CALUDE_seashells_given_l537_53745

theorem seashells_given (initial_seashells current_seashells : ℕ) 
  (h1 : initial_seashells = 5)
  (h2 : current_seashells = 3) : 
  initial_seashells - current_seashells = 2 := by
  sorry

end NUMINAMATH_CALUDE_seashells_given_l537_53745


namespace NUMINAMATH_CALUDE_unique_valid_sequence_l537_53729

/-- Represents a sequence of 5 missile numbers. -/
def MissileSequence := Fin 5 → Nat

/-- The total number of missiles. -/
def totalMissiles : Nat := 50

/-- Checks if a sequence is valid according to the problem conditions. -/
def isValidSequence (seq : MissileSequence) : Prop :=
  ∀ i j : Fin 5, i < j →
    (seq i < seq j) ∧
    (seq j ≤ totalMissiles) ∧
    (∃ k : Nat, seq j - seq i = k * (j - i))

/-- The specific sequence given in the correct answer. -/
def correctSequence : MissileSequence :=
  fun i => [3, 13, 23, 33, 43].get i

/-- Theorem stating that the correct sequence is the only valid sequence. -/
theorem unique_valid_sequence :
  (isValidSequence correctSequence) ∧
  (∀ seq : MissileSequence, isValidSequence seq → seq = correctSequence) := by
  sorry


end NUMINAMATH_CALUDE_unique_valid_sequence_l537_53729


namespace NUMINAMATH_CALUDE_race_finishing_orders_eq_twelve_l537_53730

/-- Represents the number of possible finishing orders in a race with three participants,
    allowing for a tie only in the first place. -/
def race_finishing_orders : ℕ := 12

/-- Theorem stating that the number of possible finishing orders in a race with three participants,
    allowing for a tie only in the first place, is 12. -/
theorem race_finishing_orders_eq_twelve : race_finishing_orders = 12 := by
  sorry

end NUMINAMATH_CALUDE_race_finishing_orders_eq_twelve_l537_53730


namespace NUMINAMATH_CALUDE_third_year_cost_l537_53789

def total_first_year_cost : ℝ := 10000
def tuition_percentage : ℝ := 0.40
def room_and_board_percentage : ℝ := 0.35
def tuition_increase_rate : ℝ := 0.06
def room_and_board_increase_rate : ℝ := 0.03
def initial_financial_aid_percentage : ℝ := 0.25
def financial_aid_increase_rate : ℝ := 0.02

def tuition (year : ℕ) : ℝ :=
  total_first_year_cost * tuition_percentage * (1 + tuition_increase_rate) ^ (year - 1)

def room_and_board (year : ℕ) : ℝ :=
  total_first_year_cost * room_and_board_percentage * (1 + room_and_board_increase_rate) ^ (year - 1)

def textbooks_and_transportation : ℝ :=
  total_first_year_cost * (1 - tuition_percentage - room_and_board_percentage)

def financial_aid (year : ℕ) : ℝ :=
  tuition year * (initial_financial_aid_percentage + financial_aid_increase_rate * (year - 1))

def total_cost (year : ℕ) : ℝ :=
  tuition year + room_and_board year + textbooks_and_transportation - financial_aid year

theorem third_year_cost :
  total_cost 3 = 9404.17 := by
  sorry

end NUMINAMATH_CALUDE_third_year_cost_l537_53789


namespace NUMINAMATH_CALUDE_tv_price_increase_l537_53727

theorem tv_price_increase (P : ℝ) (x : ℝ) (h : P > 0) :
  (0.80 * P + x / 100 * (0.80 * P) = 1.16 * P) → x = 45 :=
by
  sorry

end NUMINAMATH_CALUDE_tv_price_increase_l537_53727


namespace NUMINAMATH_CALUDE_probability_of_marked_section_on_top_l537_53754

theorem probability_of_marked_section_on_top (n : ℕ) (h : n = 8) : 
  (1 : ℚ) / n = (1 : ℚ) / 8 := by
  sorry

#check probability_of_marked_section_on_top

end NUMINAMATH_CALUDE_probability_of_marked_section_on_top_l537_53754


namespace NUMINAMATH_CALUDE_mixture_ratio_correct_l537_53758

def initial_alcohol : ℚ := 4
def initial_water : ℚ := 4
def added_water : ℚ := 8/3

def final_alcohol : ℚ := initial_alcohol
def final_water : ℚ := initial_water + added_water
def final_total : ℚ := final_alcohol + final_water

def desired_alcohol_ratio : ℚ := 3/8
def desired_water_ratio : ℚ := 5/8

theorem mixture_ratio_correct :
  (final_alcohol / final_total = desired_alcohol_ratio) ∧
  (final_water / final_total = desired_water_ratio) :=
sorry

end NUMINAMATH_CALUDE_mixture_ratio_correct_l537_53758


namespace NUMINAMATH_CALUDE_theorem_1_theorem_2_l537_53707

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- Theorem 1: If a is parallel to α and b is perpendicular to α, then a is perpendicular to b
theorem theorem_1 (a b : Line) (α : Plane) :
  parallel a α → perpendicular b α → perpendicular_lines a b :=
by sorry

-- Theorem 2: If a is perpendicular to α and a is parallel to β, then α is perpendicular to β
theorem theorem_2 (a : Line) (α β : Plane) :
  perpendicular a α → parallel a β → perpendicular_planes α β :=
by sorry

end NUMINAMATH_CALUDE_theorem_1_theorem_2_l537_53707


namespace NUMINAMATH_CALUDE_intersection_with_complement_l537_53748

def U : Set Nat := {1, 2, 3, 4, 5, 6}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem intersection_with_complement : A ∩ (U \ B) = {1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_with_complement_l537_53748


namespace NUMINAMATH_CALUDE_similar_triangle_perimeter_l537_53757

/-- Represents a triangle with side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.side1 = t.side2 ∨ t.side1 = t.side3 ∨ t.side2 = t.side3

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.side1 + t.side2 + t.side3

/-- Checks if two triangles are similar -/
def areTrianglesSimilar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧
    t2.side1 = k * t1.side1 ∧
    t2.side2 = k * t1.side2 ∧
    t2.side3 = k * t1.side3

theorem similar_triangle_perimeter
  (small large : Triangle)
  (h_small_isosceles : small.isIsosceles)
  (h_small_sides : small.side1 = 12 ∧ small.side2 = 24)
  (h_similar : areTrianglesSimilar small large)
  (h_large_shortest : min large.side1 (min large.side2 large.side3) = 30) :
  large.perimeter = 150 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangle_perimeter_l537_53757


namespace NUMINAMATH_CALUDE_wednesday_pages_proof_l537_53783

def total_pages : ℕ := 158
def monday_pages : ℕ := 23
def tuesday_pages : ℕ := 38
def thursday_pages : ℕ := 12

def friday_pages : ℕ := 2 * thursday_pages

theorem wednesday_pages_proof :
  total_pages - (monday_pages + tuesday_pages + thursday_pages + friday_pages) = 61 := by
  sorry

end NUMINAMATH_CALUDE_wednesday_pages_proof_l537_53783


namespace NUMINAMATH_CALUDE_students_not_playing_sports_l537_53721

theorem students_not_playing_sports (total : ℕ) (football : ℕ) (volleyball : ℕ) (one_sport : ℕ)
  (h_total : total = 40)
  (h_football : football = 20)
  (h_volleyball : volleyball = 19)
  (h_one_sport : one_sport = 15) :
  total - (football + volleyball - (football + volleyball - one_sport)) = 13 := by
  sorry

end NUMINAMATH_CALUDE_students_not_playing_sports_l537_53721


namespace NUMINAMATH_CALUDE_hyperbola_center_l537_53736

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 + 54 * x - 16 * y^2 - 128 * y - 200 = 0

/-- The center of a hyperbola -/
def is_center (h k : ℝ) : Prop :=
  ∀ x y : ℝ, hyperbola_equation x y ↔ 
    ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ((x - h)^2 / a^2) - ((y - k)^2 / b^2) = 1

/-- Theorem: The center of the given hyperbola is (-3, -4) -/
theorem hyperbola_center : is_center (-3) (-4) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_center_l537_53736


namespace NUMINAMATH_CALUDE_manuscript_typing_cost_is_1400_l537_53799

/-- Calculates the total cost of typing a manuscript with given parameters. -/
def manuscriptTypingCost (totalPages : ℕ) (firstTypeCost : ℕ) (revisionCost : ℕ) 
  (pagesRevisedOnce : ℕ) (pagesRevisedTwice : ℕ) : ℕ :=
  totalPages * firstTypeCost + 
  pagesRevisedOnce * revisionCost + 
  pagesRevisedTwice * revisionCost * 2

/-- Proves that the total cost of typing the manuscript is $1400. -/
theorem manuscript_typing_cost_is_1400 : 
  manuscriptTypingCost 100 10 5 20 30 = 1400 := by
  sorry

#eval manuscriptTypingCost 100 10 5 20 30

end NUMINAMATH_CALUDE_manuscript_typing_cost_is_1400_l537_53799


namespace NUMINAMATH_CALUDE_smallest_number_in_sample_l537_53785

/-- Systematic sampling function that returns the smallest number given the parameters -/
def systematicSampling (totalSchools : ℕ) (sampleSize : ℕ) (highestDrawn : ℕ) : ℕ :=
  let interval := totalSchools / sampleSize
  highestDrawn - (sampleSize - 1) * interval

/-- Theorem stating the smallest number drawn in the specific scenario -/
theorem smallest_number_in_sample (totalSchools : ℕ) (sampleSize : ℕ) (highestDrawn : ℕ) 
    (h1 : totalSchools = 32)
    (h2 : sampleSize = 8)
    (h3 : highestDrawn = 31) :
  systematicSampling totalSchools sampleSize highestDrawn = 3 := by
  sorry

#eval systematicSampling 32 8 31

end NUMINAMATH_CALUDE_smallest_number_in_sample_l537_53785


namespace NUMINAMATH_CALUDE_function_inequality_l537_53708

open Real

-- Define the function F
noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f x / (Real.exp x)

-- State the theorem
theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (hf' : Differentiable ℝ (deriv f))
  (h : ∀ x, deriv (deriv f) x < f x) :
  f 2 < Real.exp 2 * f 0 ∧ f 2017 < Real.exp 2017 * f 0 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l537_53708
