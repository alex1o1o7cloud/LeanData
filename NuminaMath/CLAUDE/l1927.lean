import Mathlib

namespace NUMINAMATH_CALUDE_slope_angle_of_line_l1927_192788

theorem slope_angle_of_line (x y : ℝ) (α : ℝ) :
  x * Real.sin (2 * π / 5) + y * Real.cos (2 * π / 5) = 0 →
  α = 3 * π / 5 := by
  sorry

end NUMINAMATH_CALUDE_slope_angle_of_line_l1927_192788


namespace NUMINAMATH_CALUDE_sine_function_midline_l1927_192776

theorem sine_function_midline (A B C D : ℝ) (h1 : A > 0) (h2 : B > 0) (h3 : C > 0) (h4 : D > 0) :
  (∀ x, 1 ≤ A * Real.sin (B * x + C) + D ∧ A * Real.sin (B * x + C) + D ≤ 5) → D = 3 := by
sorry

end NUMINAMATH_CALUDE_sine_function_midline_l1927_192776


namespace NUMINAMATH_CALUDE_intersecting_chords_theorem_l1927_192703

theorem intersecting_chords_theorem (chord1_segment1 chord1_segment2 chord2_ratio1 chord2_ratio2 : ℝ) :
  chord1_segment1 = 12 →
  chord1_segment2 = 18 →
  chord2_ratio1 = 3 →
  chord2_ratio2 = 8 →
  ∃ (chord2_length : ℝ),
    chord2_length = chord2_ratio1 / (chord2_ratio1 + chord2_ratio2) * chord2_length +
                    chord2_ratio2 / (chord2_ratio1 + chord2_ratio2) * chord2_length ∧
    chord1_segment1 * chord1_segment2 = (chord2_ratio1 / (chord2_ratio1 + chord2_ratio2) * chord2_length) *
                                        (chord2_ratio2 / (chord2_ratio1 + chord2_ratio2) * chord2_length) →
    chord2_length = 33 := by
  sorry

end NUMINAMATH_CALUDE_intersecting_chords_theorem_l1927_192703


namespace NUMINAMATH_CALUDE_monday_miles_proof_l1927_192717

def weekly_miles : ℕ := 30
def wednesday_miles : ℕ := 12

theorem monday_miles_proof (monday_miles : ℕ) 
  (h1 : monday_miles + wednesday_miles + 2 * monday_miles = weekly_miles) : 
  monday_miles = 6 := by
  sorry

end NUMINAMATH_CALUDE_monday_miles_proof_l1927_192717


namespace NUMINAMATH_CALUDE_reflection_property_l1927_192758

/-- A reflection in 2D space -/
structure Reflection2D where
  /-- The function that performs the reflection -/
  reflect : Fin 2 → ℝ → Fin 2 → ℝ

/-- Theorem: If a reflection takes (2, -3) to (8, 1), then it takes (1, 4) to (-18/13, -50/13) -/
theorem reflection_property (r : Reflection2D) 
  (h1 : r.reflect 0 2 = 8) 
  (h2 : r.reflect 1 (-3) = 1) 
  : r.reflect 0 1 = -18/13 ∧ r.reflect 1 4 = -50/13 := by
  sorry


end NUMINAMATH_CALUDE_reflection_property_l1927_192758


namespace NUMINAMATH_CALUDE_expression_evaluation_l1927_192722

theorem expression_evaluation :
  -5^2 + 2 * (-3)^2 - (-8) / (-1 - 1/3) = -13 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1927_192722


namespace NUMINAMATH_CALUDE_paint_cost_per_quart_l1927_192750

/-- The cost of paint per quart for a cube with given dimensions and total cost -/
theorem paint_cost_per_quart (edge_length : ℝ) (total_cost : ℝ) (coverage_per_quart : ℝ) : 
  edge_length = 10 →
  total_cost = 192 →
  coverage_per_quart = 10 →
  (total_cost / (6 * edge_length^2 / coverage_per_quart)) = 3.2 := by
sorry

end NUMINAMATH_CALUDE_paint_cost_per_quart_l1927_192750


namespace NUMINAMATH_CALUDE_platform_length_l1927_192793

/-- The length of a platform given train speed, crossing time, and train length -/
theorem platform_length
  (train_speed : ℝ)
  (crossing_time : ℝ)
  (train_length : ℝ)
  (h1 : train_speed = 72)  -- km/hr
  (h2 : crossing_time = 26)  -- seconds
  (h3 : train_length = 250)  -- meters
  : (train_speed * (5/18) * crossing_time) - train_length = 270 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l1927_192793


namespace NUMINAMATH_CALUDE_no_two_right_angles_l1927_192783

-- Define a triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define properties of a triangle
def Triangle.sumOfAngles (t : Triangle) : ℝ := t.angle1 + t.angle2 + t.angle3
def Triangle.isRight (t : Triangle) : Prop := t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

-- Theorem: A triangle cannot have two right angles
theorem no_two_right_angles (t : Triangle) : 
  t.sumOfAngles = 180 → ¬(t.angle1 = 90 ∧ t.angle2 = 90) :=
by
  sorry


end NUMINAMATH_CALUDE_no_two_right_angles_l1927_192783


namespace NUMINAMATH_CALUDE_tens_digit_of_36_pow_12_l1927_192709

theorem tens_digit_of_36_pow_12 : ∃ n : ℕ, 36^12 ≡ 10*n + 1 [MOD 100] :=
sorry

end NUMINAMATH_CALUDE_tens_digit_of_36_pow_12_l1927_192709


namespace NUMINAMATH_CALUDE_parabola_coefficient_l1927_192759

/-- Given a parabola y = ax^2 + bx + c with vertex (q, 2q) and y-intercept (0, -3q), where q ≠ 0, 
    the value of b is 10. -/
theorem parabola_coefficient (a b c q : ℝ) : q ≠ 0 →
  (∀ x y, y = a * x^2 + b * x + c ↔ 
    (x = q ∧ y = 2*q) ∨ (x = 0 ∧ y = -3*q)) →
  b = 10 := by sorry

end NUMINAMATH_CALUDE_parabola_coefficient_l1927_192759


namespace NUMINAMATH_CALUDE_hexagon_to_square_area_equality_l1927_192729

/-- Proves that a square with side length s = √(3√3/2) * a has the same area as a regular hexagon with side length a -/
theorem hexagon_to_square_area_equality (a : ℝ) (h : a > 0) :
  let s := Real.sqrt (3 * Real.sqrt 3 / 2) * a
  s^2 = (3 * Real.sqrt 3 / 2) * a^2 := by
  sorry

#check hexagon_to_square_area_equality

end NUMINAMATH_CALUDE_hexagon_to_square_area_equality_l1927_192729


namespace NUMINAMATH_CALUDE_prob_at_least_one_two_l1927_192713

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The total number of possible outcomes when rolling two dice -/
def totalOutcomes : ℕ := numSides * numSides

/-- The number of outcomes where neither die shows a 2 -/
def neitherShowsTwo : ℕ := (numSides - 1) * (numSides - 1)

/-- The number of outcomes where at least one die shows a 2 -/
def atLeastOneShowsTwo : ℕ := totalOutcomes - neitherShowsTwo

/-- The probability of at least one die showing a 2 when two fair 6-sided dice are rolled -/
theorem prob_at_least_one_two : 
  (atLeastOneShowsTwo : ℚ) / totalOutcomes = 11 / 36 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_two_l1927_192713


namespace NUMINAMATH_CALUDE_chip_division_percentage_l1927_192772

theorem chip_division_percentage (total_chips : ℕ) (ratio_small : ℕ) (ratio_large : ℕ) 
  (h_total : total_chips = 100)
  (h_ratio : ratio_small + ratio_large = 10)
  (h_ratio_order : ratio_large > ratio_small)
  (h_ratio_large : ratio_large = 6) :
  (ratio_large : ℚ) / (ratio_small + ratio_large : ℚ) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_chip_division_percentage_l1927_192772


namespace NUMINAMATH_CALUDE_fourth_altitude_is_six_times_radius_l1927_192796

/-- A tetrahedron with an inscribed sphere -/
structure Tetrahedron :=
  (r : ℝ)  -- radius of the inscribed sphere
  (h₁ h₂ h₃ h₄ : ℝ)  -- altitudes of the tetrahedron
  (h₁_eq : h₁ = 3 * r)
  (h₂_eq : h₂ = 4 * r)
  (h₃_eq : h₃ = 4 * r)
  (sum_reciprocals : 1 / h₁ + 1 / h₂ + 1 / h₃ + 1 / h₄ = 1 / r)

/-- The fourth altitude of the tetrahedron is 6 times the radius of its inscribed sphere -/
theorem fourth_altitude_is_six_times_radius (T : Tetrahedron) : T.h₄ = 6 * T.r := by
  sorry

end NUMINAMATH_CALUDE_fourth_altitude_is_six_times_radius_l1927_192796


namespace NUMINAMATH_CALUDE_anne_solo_time_l1927_192787

-- Define the cleaning rates
def bruce_rate : ℝ := sorry
def anne_rate : ℝ := sorry

-- Define the conditions
axiom clean_together : bruce_rate + anne_rate = 1 / 4
axiom clean_anne_double : bruce_rate + 2 * anne_rate = 1 / 3

-- Theorem to prove
theorem anne_solo_time : 1 / anne_rate = 12 := by sorry

end NUMINAMATH_CALUDE_anne_solo_time_l1927_192787


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l1927_192738

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = π → A = π * r^2 → d = 2 * r → d = 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l1927_192738


namespace NUMINAMATH_CALUDE_least_with_eight_factors_l1927_192785

/-- A function that returns the number of distinct positive factors of a positive integer -/
def number_of_factors (n : ℕ+) : ℕ := sorry

/-- A function that returns the set of all distinct positive factors of a positive integer -/
def factors (n : ℕ+) : Finset ℕ+ := sorry

/-- The theorem stating that 24 is the least positive integer with exactly eight distinct positive factors -/
theorem least_with_eight_factors :
  ∀ n : ℕ+, number_of_factors n = 8 → n ≥ 24 ∧ 
  (n = 24 → number_of_factors 24 = 8 ∧ factors 24 = {1, 2, 3, 4, 6, 8, 12, 24}) := by
  sorry

end NUMINAMATH_CALUDE_least_with_eight_factors_l1927_192785


namespace NUMINAMATH_CALUDE_prob_at_least_one_2_or_4_is_5_9_l1927_192730

/-- The probability of rolling a 2 or 4 on a single fair 6-sided die -/
def prob_2_or_4 : ℚ := 1/3

/-- The probability of not rolling a 2 or 4 on a single fair 6-sided die -/
def prob_not_2_or_4 : ℚ := 2/3

/-- The probability of at least one die showing 2 or 4 when rolling two fair 6-sided dice -/
def prob_at_least_one_2_or_4 : ℚ := 1 - (prob_not_2_or_4 * prob_not_2_or_4)

theorem prob_at_least_one_2_or_4_is_5_9 : 
  prob_at_least_one_2_or_4 = 5/9 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_2_or_4_is_5_9_l1927_192730


namespace NUMINAMATH_CALUDE_floor_area_calculation_l1927_192755

/-- The total area of a floor covered by square stone slabs -/
def floor_area (num_slabs : ℕ) (slab_side_length : ℝ) : ℝ :=
  (num_slabs : ℝ) * slab_side_length * slab_side_length

/-- Theorem: The total area of a floor covered by 50 square stone slabs, 
    each with a side length of 140 cm, is 980000 cm² -/
theorem floor_area_calculation :
  floor_area 50 140 = 980000 := by
  sorry

end NUMINAMATH_CALUDE_floor_area_calculation_l1927_192755


namespace NUMINAMATH_CALUDE_teresa_black_pencils_l1927_192769

/-- Given Teresa's pencil distribution problem, prove she has 35 black pencils. -/
theorem teresa_black_pencils : 
  (colored_pencils : ℕ) →
  (siblings : ℕ) →
  (pencils_per_sibling : ℕ) →
  (pencils_kept : ℕ) →
  colored_pencils = 14 →
  siblings = 3 →
  pencils_per_sibling = 13 →
  pencils_kept = 10 →
  (siblings * pencils_per_sibling + pencils_kept) - colored_pencils = 35 := by
sorry

end NUMINAMATH_CALUDE_teresa_black_pencils_l1927_192769


namespace NUMINAMATH_CALUDE_drowned_ratio_l1927_192725

/-- Proves the ratio of drowned cows to drowned sheep given the initial conditions -/
theorem drowned_ratio (initial_sheep initial_cows initial_dogs : ℕ)
  (drowned_sheep : ℕ) (total_survived : ℕ) :
  initial_sheep = 20 →
  initial_cows = 10 →
  initial_dogs = 14 →
  drowned_sheep = 3 →
  total_survived = 35 →
  (initial_cows - (total_survived - (initial_sheep - drowned_sheep) - initial_dogs)) /
  drowned_sheep = 2 := by
  sorry

end NUMINAMATH_CALUDE_drowned_ratio_l1927_192725


namespace NUMINAMATH_CALUDE_expression_evaluation_l1927_192786

theorem expression_evaluation (a b c : ℝ) 
  (ha : a = 12) (hb : b = 14) (hc : c = 19) : 
  (a^2 * (1/b - 1/c) + b^2 * (1/c - 1/a) + c^2 * (1/a - 1/b)) / 
  (a * (1/b - 1/c) + b * (1/c - 1/a) + c * (1/a - 1/b)) = 45 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1927_192786


namespace NUMINAMATH_CALUDE_probability_of_B_is_one_fourth_l1927_192726

/-- The probability of choosing a specific letter from a bag of letters -/
def probability_of_letter (total_letters : ℕ) (target_letters : ℕ) : ℚ :=
  target_letters / total_letters

/-- The bag contains 8 letters in total -/
def total_letters : ℕ := 8

/-- The bag contains 2 B's -/
def number_of_Bs : ℕ := 2

/-- The probability of choosing a B is 1/4 -/
theorem probability_of_B_is_one_fourth :
  probability_of_letter total_letters number_of_Bs = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_B_is_one_fourth_l1927_192726


namespace NUMINAMATH_CALUDE_wheat_yield_problem_l1927_192770

theorem wheat_yield_problem (plot1_area plot2_area : ℝ) 
  (h1 : plot2_area = plot1_area + 0.5)
  (h2 : 210 / plot1_area = 210 / plot2_area + 1) :
  210 / plot1_area = 21 ∧ 210 / plot2_area = 20 := by
sorry

end NUMINAMATH_CALUDE_wheat_yield_problem_l1927_192770


namespace NUMINAMATH_CALUDE_total_spaces_eq_spaces_to_win_susan_game_total_spaces_l1927_192771

/-- Represents the board game Susan is playing --/
structure BoardGame where
  first_turn : ℕ
  second_turn_forward : ℕ
  second_turn_backward : ℕ
  third_turn : ℕ
  spaces_to_win : ℕ

/-- Calculates the total number of spaces in the game --/
def total_spaces (game : BoardGame) : ℕ :=
  game.first_turn + game.second_turn_forward - game.second_turn_backward + game.third_turn +
  (game.spaces_to_win - (game.first_turn + game.second_turn_forward - game.second_turn_backward + game.third_turn))

/-- Theorem stating that the total number of spaces in the game is equal to the spaces to win --/
theorem total_spaces_eq_spaces_to_win (game : BoardGame) :
  total_spaces game = game.spaces_to_win := by
  sorry

/-- Susan's specific game instance --/
def susan_game : BoardGame :=
  { first_turn := 8
    second_turn_forward := 2
    second_turn_backward := 5
    third_turn := 6
    spaces_to_win := 37 }

/-- Theorem proving that Susan's game has 37 spaces in total --/
theorem susan_game_total_spaces :
  total_spaces susan_game = 37 := by
  sorry

end NUMINAMATH_CALUDE_total_spaces_eq_spaces_to_win_susan_game_total_spaces_l1927_192771


namespace NUMINAMATH_CALUDE_rectangle_area_l1927_192791

theorem rectangle_area (x y : ℝ) (h_perimeter : x + y = 6) (h_diagonal : x^2 + y^2 = 25) :
  x * y = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l1927_192791


namespace NUMINAMATH_CALUDE_product_of_numbers_l1927_192754

theorem product_of_numbers (x y : ℝ) (h1 : x^2 + y^2 = 289) (h2 : x + y = 23) : x * y = 120 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l1927_192754


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l1927_192782

-- Define the operation ⋈
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt (b^2))))

-- State the theorem
theorem bowtie_equation_solution :
  ∀ x : ℝ, bowtie 3 x = 15 → x = 2 * Real.sqrt 33 ∨ x = -2 * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l1927_192782


namespace NUMINAMATH_CALUDE_preimage_of_3_1_l1927_192757

-- Define the mapping f
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + 2*p.2, 2*p.1 - p.2)

-- Theorem statement
theorem preimage_of_3_1 : f (1, 1) = (3, 1) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_3_1_l1927_192757


namespace NUMINAMATH_CALUDE_combined_savings_equals_individual_savings_l1927_192702

-- Define the regular price of a window
def regular_price : ℕ := 120

-- Define the offer: for every 5 windows purchased, 2 are free
def offer (n : ℕ) : ℕ := (n / 5) * 2

-- Calculate the cost for a given number of windows with the offer
def cost_with_offer (n : ℕ) : ℕ :=
  regular_price * (n - offer n)

-- Calculate savings for a given number of windows
def savings (n : ℕ) : ℕ :=
  regular_price * n - cost_with_offer n

-- Dave's required windows
def dave_windows : ℕ := 9

-- Doug's required windows
def doug_windows : ℕ := 10

-- Combined windows
def combined_windows : ℕ := dave_windows + doug_windows

-- Theorem: Combined savings equals sum of individual savings
theorem combined_savings_equals_individual_savings :
  savings combined_windows = savings dave_windows + savings doug_windows :=
sorry

end NUMINAMATH_CALUDE_combined_savings_equals_individual_savings_l1927_192702


namespace NUMINAMATH_CALUDE_binary_1101100_eq_108_l1927_192774

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (binary : List Bool) : ℕ :=
  binary.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1101100₂ -/
def binary_1101100 : List Bool := [false, false, true, true, false, true, true]

/-- Theorem stating that 1101100₂ is equal to 108 in decimal -/
theorem binary_1101100_eq_108 : binary_to_decimal binary_1101100 = 108 := by
  sorry

end NUMINAMATH_CALUDE_binary_1101100_eq_108_l1927_192774


namespace NUMINAMATH_CALUDE_angle_sum_l1927_192740

/-- A configuration of angles in a geometric figure -/
structure AngleConfiguration where
  -- Pentagon angles
  a : Real
  b : Real
  x_ext : Real
  -- Right triangle angle
  c : Real
  -- Additional angle
  y : Real
  -- Constraints
  a_value : a = 36
  b_value : b = 80
  c_value : c = 24
  pentagon_sum : a + b + (360 - x_ext) + 90 + (114 - y) = 540

/-- The sum of the exterior angle x and the angle y is 140° -/
theorem angle_sum (config : AngleConfiguration) : config.x_ext + config.y = 140 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_l1927_192740


namespace NUMINAMATH_CALUDE_calendar_cost_l1927_192746

/-- Proves that the cost of each calendar is 2/3 dollars given the problem conditions --/
theorem calendar_cost (total_items : ℕ) (num_calendars : ℕ) (num_date_books : ℕ) 
  (total_spent : ℚ) (date_book_cost : ℚ) :
  total_items = 500 →
  num_calendars = 300 →
  num_date_books = 200 →
  total_spent = 300 →
  date_book_cost = 1/2 →
  (total_spent - num_date_books * date_book_cost) / num_calendars = 2/3 := by
  sorry

#check calendar_cost

end NUMINAMATH_CALUDE_calendar_cost_l1927_192746


namespace NUMINAMATH_CALUDE_triangle_property_l1927_192765

-- Define a triangle
structure Triangle where
  α : Real
  β : Real
  γ : Real
  sum_angles : α + β + γ = Real.pi

-- Define the condition
def condition (t : Triangle) : Prop :=
  Real.tan t.β * (Real.sin t.γ)^2 = Real.tan t.γ * (Real.sin t.β)^2

-- Define isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  t.α = t.β ∨ t.β = t.γ ∨ t.γ = t.α

-- Define right-angled triangle
def is_right_angled (t : Triangle) : Prop :=
  t.α = Real.pi/2 ∨ t.β = Real.pi/2 ∨ t.γ = Real.pi/2

-- Theorem statement
theorem triangle_property (t : Triangle) :
  condition t → is_isosceles t ∨ is_right_angled t :=
by sorry

end NUMINAMATH_CALUDE_triangle_property_l1927_192765


namespace NUMINAMATH_CALUDE_parallelogram_angles_l1927_192714

/-- Represents the angles of a parallelogram -/
structure ParallelogramAngles where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ
  angle4 : ℝ

/-- Properties of a parallelogram with one angle 50° less than the other -/
def is_valid_parallelogram (p : ParallelogramAngles) : Prop :=
  p.angle1 = p.angle3 ∧
  p.angle2 = p.angle4 ∧
  p.angle1 + p.angle2 = 180 ∧
  p.angle2 = p.angle1 + 50

/-- Theorem: The angles of a parallelogram with one angle 50° less than the other are 65°, 115°, 65°, and 115° -/
theorem parallelogram_angles :
  ∃ (p : ParallelogramAngles), is_valid_parallelogram p ∧
    p.angle1 = 65 ∧ p.angle2 = 115 ∧ p.angle3 = 65 ∧ p.angle4 = 115 :=
by
  sorry

end NUMINAMATH_CALUDE_parallelogram_angles_l1927_192714


namespace NUMINAMATH_CALUDE_mooncake_sales_properties_l1927_192764

/-- Represents the mooncake sales scenario -/
structure MooncakeSales where
  initial_purchase : ℕ
  purchase_price : ℕ
  initial_selling_price : ℕ
  price_reduction : ℕ
  sales_increase_per_yuan : ℕ

/-- Calculates the profit per box in the second sale -/
def profit_per_box (s : MooncakeSales) : ℤ :=
  s.initial_selling_price - s.purchase_price - s.price_reduction

/-- Calculates the expected sales volume in the second sale -/
def expected_sales_volume (s : MooncakeSales) : ℕ :=
  s.initial_purchase + s.sales_increase_per_yuan * s.price_reduction

/-- Theorem stating the properties of the mooncake sales scenario -/
theorem mooncake_sales_properties (s : MooncakeSales) 
  (h1 : s.initial_purchase = 180)
  (h2 : s.purchase_price = 40)
  (h3 : s.initial_selling_price = 52)
  (h4 : s.sales_increase_per_yuan = 10) :
  (∃ a : ℕ, 
    profit_per_box { initial_purchase := s.initial_purchase,
                     purchase_price := s.purchase_price,
                     initial_selling_price := s.initial_selling_price,
                     price_reduction := a,
                     sales_increase_per_yuan := s.sales_increase_per_yuan } = 12 - a ∧
    expected_sales_volume { initial_purchase := s.initial_purchase,
                            purchase_price := s.purchase_price,
                            initial_selling_price := s.initial_selling_price,
                            price_reduction := a,
                            sales_increase_per_yuan := s.sales_increase_per_yuan } = 180 + 10 * a ∧
    (profit_per_box { initial_purchase := s.initial_purchase,
                      purchase_price := s.purchase_price,
                      initial_selling_price := s.initial_selling_price,
                      price_reduction := a,
                      sales_increase_per_yuan := s.sales_increase_per_yuan } *
     expected_sales_volume { initial_purchase := s.initial_purchase,
                             purchase_price := s.purchase_price,
                             initial_selling_price := s.initial_selling_price,
                             price_reduction := a,
                             sales_increase_per_yuan := s.sales_increase_per_yuan } = 2000 →
     a = 2)) :=
by sorry

end NUMINAMATH_CALUDE_mooncake_sales_properties_l1927_192764


namespace NUMINAMATH_CALUDE_line_slope_l1927_192721

/-- A line that returns to its original position after moving 4 units left and 1 unit up has a slope of -1/4 -/
theorem line_slope (l : ℝ → ℝ) (b : ℝ) (h : ∀ x, l x = l (x + 4) - 1) : 
  ∃ k, k = -1/4 ∧ ∀ x, l x = k * x + b := by
sorry

end NUMINAMATH_CALUDE_line_slope_l1927_192721


namespace NUMINAMATH_CALUDE_one_in_A_l1927_192798

def A : Set ℕ := {1, 2}

theorem one_in_A : 1 ∈ A := by sorry

end NUMINAMATH_CALUDE_one_in_A_l1927_192798


namespace NUMINAMATH_CALUDE_meeting_point_l1927_192768

/-- Represents the walking speed of a person -/
structure WalkingSpeed where
  speed : ℝ
  speed_pos : speed > 0

/-- Represents a person walking around the block -/
structure Walker where
  name : String
  speed : WalkingSpeed

/-- The scenario of Jane and Hector walking around the block -/
structure WalkingScenario where
  jane : Walker
  hector : Walker
  block_size : ℝ
  jane_speed_ratio : ℝ
  start_point : ℝ
  jane_speed_twice_hector : jane.speed.speed = 2 * hector.speed.speed
  block_size_positive : block_size > 0
  jane_speed_ratio_half : jane_speed_ratio = 1/2
  start_point_zero : start_point = 0

/-- The theorem stating where Jane and Hector meet -/
theorem meeting_point (scenario : WalkingScenario) : 
  ∃ t : ℝ, t > 0 ∧ 
  (scenario.hector.speed.speed * t + scenario.jane.speed.speed * t = scenario.block_size) ∧
  (scenario.hector.speed.speed * t = 12) := by
  sorry

end NUMINAMATH_CALUDE_meeting_point_l1927_192768


namespace NUMINAMATH_CALUDE_exp_15pi_over_2_l1927_192716

theorem exp_15pi_over_2 : Complex.exp (15 * Real.pi * Complex.I / 2) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_exp_15pi_over_2_l1927_192716


namespace NUMINAMATH_CALUDE_tangerines_left_l1927_192797

theorem tangerines_left (initial : ℕ) (eaten : ℕ) (h1 : initial = 12) (h2 : eaten = 7) :
  initial - eaten = 5 := by
  sorry

end NUMINAMATH_CALUDE_tangerines_left_l1927_192797


namespace NUMINAMATH_CALUDE_initial_bottles_correct_l1927_192747

/-- The number of water bottles initially in Samira's box -/
def initial_bottles : ℕ := 48

/-- The number of players on the field -/
def num_players : ℕ := 11

/-- The number of bottles each player takes in the first break -/
def bottles_first_break : ℕ := 2

/-- The number of bottles each player takes at the end of the game -/
def bottles_end_game : ℕ := 1

/-- The number of bottles remaining after the game -/
def remaining_bottles : ℕ := 15

/-- Theorem stating that the initial number of bottles is correct -/
theorem initial_bottles_correct :
  initial_bottles = num_players * (bottles_first_break + bottles_end_game) + remaining_bottles :=
by sorry

end NUMINAMATH_CALUDE_initial_bottles_correct_l1927_192747


namespace NUMINAMATH_CALUDE_zebra_stripes_l1927_192766

theorem zebra_stripes (w n b : ℕ) : 
  w + n = b + 1 →  -- Total black stripes is one more than white stripes
  b = w + 7 →      -- White stripes are 7 more than wide black stripes
  n = 8            -- Number of narrow black stripes is 8
:= by sorry

end NUMINAMATH_CALUDE_zebra_stripes_l1927_192766


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l1927_192718

theorem no_solution_absolute_value_equation :
  (∀ x : ℝ, (x - 4)^2 ≠ 0 → ∃ y : ℝ, (y - 4)^2 = 0) ∧
  (∀ x : ℝ, |(-5 : ℝ) * x| + 10 ≠ 0) ∧
  (∀ x : ℝ, Real.sqrt (-x) - 3 ≠ 0 → ∃ y : ℝ, Real.sqrt (-y) - 3 = 0) ∧
  (∀ x : ℝ, Real.sqrt x - 7 ≠ 0 → ∃ y : ℝ, Real.sqrt y - 7 = 0) ∧
  (∀ x : ℝ, |(-5 : ℝ) * x| - 6 ≠ 0 → ∃ y : ℝ, |(-5 : ℝ) * y| - 6 = 0) :=
by sorry


end NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l1927_192718


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1927_192745

theorem rationalize_denominator : (1 : ℝ) / (2 - Real.sqrt 3) = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1927_192745


namespace NUMINAMATH_CALUDE_difference_of_squares_l1927_192784

theorem difference_of_squares (x : ℝ) : x^2 - 36 = (x + 6) * (x - 6) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l1927_192784


namespace NUMINAMATH_CALUDE_girls_in_college_l1927_192737

theorem girls_in_college (total_students : ℕ) (ratio_boys : ℕ) (ratio_girls : ℕ) :
  total_students = 455 →
  ratio_boys = 8 →
  ratio_girls = 5 →
  ∃ (num_girls : ℕ), num_girls * (ratio_boys + ratio_girls) = total_students * ratio_girls ∧ num_girls = 175 :=
by
  sorry

end NUMINAMATH_CALUDE_girls_in_college_l1927_192737


namespace NUMINAMATH_CALUDE_max_x_minus_y_l1927_192744

theorem max_x_minus_y (x y : ℝ) (h : 2 * (x^2 + y^2) = x + y) : x - y ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l1927_192744


namespace NUMINAMATH_CALUDE_isosceles_triangles_count_l1927_192790

/-- A right hexagonal prism with height 2 and regular hexagonal bases of side length 1 -/
structure HexagonalPrism where
  height : ℝ
  base_side_length : ℝ
  height_eq : height = 2
  side_eq : base_side_length = 1

/-- A triangle formed by three vertices of the hexagonal prism -/
structure PrismTriangle where
  prism : HexagonalPrism
  v1 : Fin 12
  v2 : Fin 12
  v3 : Fin 12
  distinct : v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3

/-- Predicate to determine if a triangle is isosceles -/
def is_isosceles (t : PrismTriangle) : Prop :=
  sorry

/-- The number of isosceles triangles in the hexagonal prism -/
def num_isosceles_triangles (p : HexagonalPrism) : ℕ :=
  sorry

/-- Theorem stating that the number of isosceles triangles is 24 -/
theorem isosceles_triangles_count (p : HexagonalPrism) :
  num_isosceles_triangles p = 24 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangles_count_l1927_192790


namespace NUMINAMATH_CALUDE_ben_car_payment_l1927_192749

/-- Ben's monthly finances -/
structure BenFinances where
  gross_income : ℝ
  tax_rate : ℝ
  car_expense_rate : ℝ

/-- Calculate Ben's car payment given his financial structure -/
def car_payment (bf : BenFinances) : ℝ :=
  bf.gross_income * (1 - bf.tax_rate) * bf.car_expense_rate

/-- Theorem: Ben's car payment is $400 given the specified conditions -/
theorem ben_car_payment :
  let bf : BenFinances := {
    gross_income := 3000,
    tax_rate := 1/3,
    car_expense_rate := 0.20
  }
  car_payment bf = 400 := by
  sorry


end NUMINAMATH_CALUDE_ben_car_payment_l1927_192749


namespace NUMINAMATH_CALUDE_intersection_of_sets_l1927_192739

open Set

theorem intersection_of_sets : 
  let A : Set ℝ := {x | 2 < x ∧ x < 4}
  let B : Set ℝ := {x | x > 5/3}
  A ∩ B = {x | 2 < x ∧ x < 3} := by
sorry

end NUMINAMATH_CALUDE_intersection_of_sets_l1927_192739


namespace NUMINAMATH_CALUDE_calculation_proof_l1927_192706

theorem calculation_proof : 8 * 2.25 - 5 * 0.85 / 2.5 = 16.3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1927_192706


namespace NUMINAMATH_CALUDE_max_triangle_side_length_l1927_192752

theorem max_triangle_side_length :
  ∀ a b c : ℕ,
    a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- Three different integer side lengths
    a + b + c = 30 →        -- Perimeter is 30 units
    a > 0 ∧ b > 0 ∧ c > 0 → -- Positive side lengths
    a + b > c ∧ b + c > a ∧ a + c > b → -- Triangle inequality
    a ≤ 14 ∧ b ≤ 14 ∧ c ≤ 14 -- Maximum side length is 14
  := by sorry

end NUMINAMATH_CALUDE_max_triangle_side_length_l1927_192752


namespace NUMINAMATH_CALUDE_unique_five_digit_square_last_five_l1927_192756

theorem unique_five_digit_square_last_five : ∃! (A : ℕ), 
  10000 ≤ A ∧ A < 100000 ∧ A^2 % 100000 = A :=
by
  use 90625
  sorry

end NUMINAMATH_CALUDE_unique_five_digit_square_last_five_l1927_192756


namespace NUMINAMATH_CALUDE_altitude_equation_median_equation_l1927_192795

/-- Triangle ABC with vertices A(-2,-1), B(2,1), and C(1,3) -/
structure Triangle :=
  (A : ℝ × ℝ)
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)

/-- The specific triangle given in the problem -/
def given_triangle : Triangle :=
  { A := (-2, -1),
    B := (2, 1),
    C := (1, 3) }

/-- Equation of a line in point-slope form -/
structure PointSlopeLine :=
  (m : ℝ)  -- slope
  (x₀ : ℝ) -- x-coordinate of point
  (y₀ : ℝ) -- y-coordinate of point

/-- Equation of a line in general form -/
structure GeneralLine :=
  (a : ℝ)
  (b : ℝ)
  (c : ℝ)

/-- The altitude from side AB of the triangle -/
def altitude (t : Triangle) : PointSlopeLine :=
  { m := -2,
    x₀ := 1,
    y₀ := 3 }

/-- The median from side AB of the triangle -/
def median (t : Triangle) : GeneralLine :=
  { a := 3,
    b := -1,
    c := 0 }

theorem altitude_equation (t : Triangle) :
  t = given_triangle →
  altitude t = { m := -2, x₀ := 1, y₀ := 3 } :=
by sorry

theorem median_equation (t : Triangle) :
  t = given_triangle →
  median t = { a := 3, b := -1, c := 0 } :=
by sorry

end NUMINAMATH_CALUDE_altitude_equation_median_equation_l1927_192795


namespace NUMINAMATH_CALUDE_conic_sections_l1927_192767

-- Hyperbola
def hyperbola_equation (e : ℝ) (c : ℝ) : Prop :=
  e = Real.sqrt 3 ∧ c = 5 * Real.sqrt 3 →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧
    (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 25 - y^2 / 50 = 1)

-- Ellipse
def ellipse_equation (e : ℝ) (d : ℝ) : Prop :=
  e = 1/2 ∧ d = 4 * Real.sqrt 3 →
  ∃ a b : ℝ, a > b ∧ b > 0 ∧
    (∀ x y : ℝ, y^2 / a^2 + x^2 / b^2 = 1 ↔ y^2 / 12 + x^2 / 9 = 1)

-- Parabola
def parabola_equation (p : ℝ) : Prop :=
  p = 4 →
  ∀ x y : ℝ, x^2 = 4 * p * y ↔ x^2 = 8 * y

theorem conic_sections :
  ∀ (e_hyp e_ell c d p : ℝ),
    hyperbola_equation e_hyp c ∧
    ellipse_equation e_ell d ∧
    parabola_equation p :=
by sorry

end NUMINAMATH_CALUDE_conic_sections_l1927_192767


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1927_192799

/-- If 1, 3, and x form a geometric sequence, then x = 9 -/
theorem geometric_sequence_third_term (x : ℝ) : 
  (∃ r : ℝ, r ≠ 0 ∧ 3 = 1 * r ∧ x = 3 * r) → x = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1927_192799


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l1927_192760

theorem simplify_and_evaluate_expression :
  let x : ℝ := Real.sqrt 2
  (1 + x) / (1 - x) / (x - 2 * x / (1 - x)) = - (Real.sqrt 2 + 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l1927_192760


namespace NUMINAMATH_CALUDE_exponent_calculation_l1927_192712

theorem exponent_calculation : (3^5 / 3^2) * 5^6 = 421875 := by
  sorry

end NUMINAMATH_CALUDE_exponent_calculation_l1927_192712


namespace NUMINAMATH_CALUDE_remainder_theorem_l1927_192701

/-- The dividend polynomial -/
def f (x : ℝ) : ℝ := 3*x^5 - 2*x^3 + 5*x - 8

/-- The divisor polynomial -/
def g (x : ℝ) : ℝ := x^2 - 3*x + 2

/-- The proposed remainder -/
def r (x : ℝ) : ℝ := 84*x - 84

/-- Theorem stating that r is the remainder when f is divided by g -/
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x, f x = g x * q x + r x :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1927_192701


namespace NUMINAMATH_CALUDE_production_average_l1927_192753

theorem production_average (n : ℕ) : 
  (n * 50 + 110) / (n + 1) = 55 → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_production_average_l1927_192753


namespace NUMINAMATH_CALUDE_balloon_difference_l1927_192707

theorem balloon_difference (your_balloons friend_balloons : ℕ) 
  (h1 : your_balloons = 7) 
  (h2 : friend_balloons = 5) : 
  your_balloons - friend_balloons = 2 := by
sorry

end NUMINAMATH_CALUDE_balloon_difference_l1927_192707


namespace NUMINAMATH_CALUDE_inverse_trig_sum_l1927_192723

theorem inverse_trig_sum : 
  Real.arcsin (-1/2) + Real.arccos (-Real.sqrt 3/2) + Real.arctan (-Real.sqrt 3) = π/3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_trig_sum_l1927_192723


namespace NUMINAMATH_CALUDE_average_temperature_problem_l1927_192751

/-- The average temperature problem -/
theorem average_temperature_problem 
  (temp_mon : ℝ) 
  (temp_tue : ℝ) 
  (temp_wed : ℝ) 
  (temp_thu : ℝ) 
  (temp_fri : ℝ) 
  (h1 : (temp_mon + temp_tue + temp_wed + temp_thu) / 4 = 48)
  (h2 : temp_mon = 42)
  (h3 : temp_fri = 10) :
  (temp_tue + temp_wed + temp_thu + temp_fri) / 4 = 40 := by
  sorry


end NUMINAMATH_CALUDE_average_temperature_problem_l1927_192751


namespace NUMINAMATH_CALUDE_parabola_circle_tangency_l1927_192741

theorem parabola_circle_tangency (r : ℝ) : 
  (∃ (x : ℝ), x^2 + r = x) ∧ 
  (∀ (x : ℝ), x^2 + r = x → (x - 1/2)^2 = 1/4 - r) → 
  r = 1/4 := by
sorry

end NUMINAMATH_CALUDE_parabola_circle_tangency_l1927_192741


namespace NUMINAMATH_CALUDE_log_equality_implies_equal_bases_l1927_192708

/-- Proves that for x, y ∈ (0,1) and a > 0, a ≠ 1, if log_x(a) + log_y(a) = 4 log_xy(a), then x = y -/
theorem log_equality_implies_equal_bases
  (x y a : ℝ)
  (h_x : 0 < x ∧ x < 1)
  (h_y : 0 < y ∧ y < 1)
  (h_a : a > 0 ∧ a ≠ 1)
  (h_log : Real.log a / Real.log x + Real.log a / Real.log y = 4 * Real.log a / Real.log (x * y)) :
  x = y :=
by sorry

end NUMINAMATH_CALUDE_log_equality_implies_equal_bases_l1927_192708


namespace NUMINAMATH_CALUDE_evaluate_expression_l1927_192724

theorem evaluate_expression : 2000^3 - 1999 * 2000^2 - 1999^2 * 2000 + 1999^3 = 3999 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1927_192724


namespace NUMINAMATH_CALUDE_sqrt_12_minus_sqrt_3_between_1_and_2_l1927_192735

theorem sqrt_12_minus_sqrt_3_between_1_and_2 : 1 < Real.sqrt 12 - Real.sqrt 3 ∧ Real.sqrt 12 - Real.sqrt 3 < 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_12_minus_sqrt_3_between_1_and_2_l1927_192735


namespace NUMINAMATH_CALUDE_homework_check_probability_l1927_192720

/-- Represents the days of the week when math lessons occur -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday

/-- The probability space for the homework checking scenario -/
structure HomeworkProbability where
  /-- The probability that the teacher will not check homework at all during the week -/
  p_no_check : ℝ
  /-- The probability that the teacher will check homework exactly once during the week -/
  p_check_once : ℝ
  /-- The number of math lessons per week -/
  num_lessons : ℕ
  /-- Assumption: probabilities sum to 1 -/
  sum_to_one : p_no_check + p_check_once = 1
  /-- Assumption: probabilities are non-negative -/
  non_negative : 0 ≤ p_no_check ∧ 0 ≤ p_check_once
  /-- Assumption: there are 5 math lessons per week -/
  five_lessons : num_lessons = 5

/-- The main theorem to prove -/
theorem homework_check_probability (hp : HomeworkProbability) :
  hp.p_no_check = 1/2 →
  hp.p_check_once = 1/2 →
  (1/hp.num_lessons : ℝ) * hp.p_check_once / (hp.p_no_check + (1/hp.num_lessons) * hp.p_check_once) = 1/6 :=
by sorry

end NUMINAMATH_CALUDE_homework_check_probability_l1927_192720


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l1927_192704

theorem smallest_solution_of_equation (x : ℝ) :
  x^4 - 50*x^2 + 576 = 0 ∧ (∀ y : ℝ, y^4 - 50*y^2 + 576 = 0 → x ≤ y) →
  x = -4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l1927_192704


namespace NUMINAMATH_CALUDE_towel_area_decrease_l1927_192781

theorem towel_area_decrease (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let new_length := 0.7 * L
  let new_breadth := 0.75 * B
  let original_area := L * B
  let new_area := new_length * new_breadth
  (original_area - new_area) / original_area = 0.475
  := by sorry

end NUMINAMATH_CALUDE_towel_area_decrease_l1927_192781


namespace NUMINAMATH_CALUDE_only_d_is_odd_l1927_192778

theorem only_d_is_odd : ∀ n : ℤ,
  (n = 3 * 5 + 1 ∨ n = 2 * (3 + 5) ∨ n = 3 * (3 + 5) ∨ n = (3 + 5) / 2) → ¬(Odd n) ∧
  Odd (3 + 5 + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_only_d_is_odd_l1927_192778


namespace NUMINAMATH_CALUDE_first_player_wins_l1927_192715

/-- Represents the state of the game -/
structure GameState where
  chips : Nat
  lastMove : Option Nat

/-- Checks if a move is valid -/
def isValidMove (state : GameState) (move : Nat) : Prop :=
  1 ≤ move ∧ move ≤ 9 ∧ move ≤ state.chips ∧ state.lastMove ≠ some move

/-- Represents a winning strategy for the first player -/
def hasWinningStrategy (initialChips : Nat) : Prop :=
  ∃ (strategy : GameState → Nat),
    ∀ (state : GameState),
      state.chips ≤ initialChips →
      (isValidMove state (strategy state) →
        ¬∃ (opponentMove : Nat), isValidMove { chips := state.chips - strategy state, lastMove := some (strategy state) } opponentMove)

/-- The main theorem stating that the first player has a winning strategy for 110 chips -/
theorem first_player_wins : hasWinningStrategy 110 := by
  sorry

end NUMINAMATH_CALUDE_first_player_wins_l1927_192715


namespace NUMINAMATH_CALUDE_pete_walked_7430_miles_l1927_192700

/-- Represents a pedometer with a maximum value before flipping --/
structure Pedometer where
  max_value : ℕ
  flip_count : ℕ
  final_reading : ℕ

/-- Calculates the total steps recorded by a pedometer --/
def total_steps (p : Pedometer) : ℕ :=
  p.max_value * p.flip_count + p.final_reading

/-- Represents Pete's walking data for the year --/
structure WalkingData where
  pedometer1 : Pedometer
  pedometer2 : Pedometer
  steps_per_mile : ℕ

/-- Theorem stating that Pete walked 7430 miles during the year --/
theorem pete_walked_7430_miles (data : WalkingData)
  (h1 : data.pedometer1.max_value = 100000)
  (h2 : data.pedometer1.flip_count = 50)
  (h3 : data.pedometer1.final_reading = 25000)
  (h4 : data.pedometer2.max_value = 400000)
  (h5 : data.pedometer2.flip_count = 15)
  (h6 : data.pedometer2.final_reading = 120000)
  (h7 : data.steps_per_mile = 1500) :
  (total_steps data.pedometer1 + total_steps data.pedometer2) / data.steps_per_mile = 7430 := by
  sorry

end NUMINAMATH_CALUDE_pete_walked_7430_miles_l1927_192700


namespace NUMINAMATH_CALUDE_chernomor_max_coins_l1927_192792

/-- Represents the problem of distributing coins among bogatyrs --/
structure BogatyrProblem where
  total_bogatyrs : Nat
  total_coins : Nat

/-- Represents a distribution of bogatyrs into groups --/
structure Distribution where
  groups : List Nat
  coins_per_group : List Nat

/-- Calculates the remainder for Chernomor given a distribution --/
def remainder (d : Distribution) : Nat :=
  d.groups.zip d.coins_per_group
    |> List.map (fun (g, c) => c % g)
    |> List.sum

/-- The maximum remainder Chernomor can get with arbitrary distribution --/
def max_remainder_arbitrary (p : BogatyrProblem) : Nat :=
  sorry

/-- The maximum remainder Chernomor can get with equal distribution --/
def max_remainder_equal (p : BogatyrProblem) : Nat :=
  sorry

theorem chernomor_max_coins (p : BogatyrProblem) 
  (h1 : p.total_bogatyrs = 33) (h2 : p.total_coins = 240) : 
  max_remainder_arbitrary p = 31 ∧ max_remainder_equal p = 30 := by
  sorry

end NUMINAMATH_CALUDE_chernomor_max_coins_l1927_192792


namespace NUMINAMATH_CALUDE_no_valid_operation_l1927_192733

-- Define the set of standard arithmetic operations
inductive ArithOp
  | Add
  | Sub
  | Mul
  | Div

-- Define a function to apply an arithmetic operation
def applyOp (op : ArithOp) (a b : Int) : Option Int :=
  match op with
  | ArithOp.Add => some (a + b)
  | ArithOp.Sub => some (a - b)
  | ArithOp.Mul => some (a * b)
  | ArithOp.Div => if b ≠ 0 then some (a / b) else none

-- Theorem statement
theorem no_valid_operation :
  ∀ (op : ArithOp), (applyOp op 7 4).map (λ x => x + 5 - (3 - 2)) ≠ some 4 := by
  sorry


end NUMINAMATH_CALUDE_no_valid_operation_l1927_192733


namespace NUMINAMATH_CALUDE_two_thousand_twelfth_digit_l1927_192705

def digit_sequence (n : ℕ) : ℕ :=
  sorry

theorem two_thousand_twelfth_digit :
  digit_sequence 2012 = 0 :=
sorry

end NUMINAMATH_CALUDE_two_thousand_twelfth_digit_l1927_192705


namespace NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_eq_198_l1927_192779

/-- The polynomial for which we calculate the sum of squares of coefficients -/
def p (x : ℝ) : ℝ := 3 * (x^5 + 4*x^3 + 2*x + 1)

/-- The sum of squares of coefficients of the polynomial p -/
def sum_of_squares_of_coefficients : ℝ :=
  (3^2) + (12^2) + (6^2) + (3^2) + (0^2) + (0^2)

theorem sum_of_squares_of_coefficients_eq_198 :
  sum_of_squares_of_coefficients = 198 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_coefficients_eq_198_l1927_192779


namespace NUMINAMATH_CALUDE_latest_sixty_degree_time_l1927_192736

/-- Temperature model as a function of time -/
def T (t : ℝ) : ℝ := -2 * t^2 + 16 * t + 40

/-- The statement to prove -/
theorem latest_sixty_degree_time :
  ∃ t_max : ℝ, t_max = 5 ∧ 
  T t_max = 60 ∧ 
  ∀ t : ℝ, T t = 60 → t ≤ t_max :=
sorry

end NUMINAMATH_CALUDE_latest_sixty_degree_time_l1927_192736


namespace NUMINAMATH_CALUDE_closest_integer_to_35_4_l1927_192734

theorem closest_integer_to_35_4 : ∀ n : ℤ, |n - (35 : ℚ) / 4| ≥ |9 - (35 : ℚ) / 4| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_35_4_l1927_192734


namespace NUMINAMATH_CALUDE_first_black_ace_most_likely_at_first_position_l1927_192794

/-- Probability of drawing the first black ace at position k in a shuffled 52-card deck --/
def probability_first_black_ace (k : ℕ) : ℚ :=
  if k ≥ 1 ∧ k ≤ 51 then (52 - k : ℚ) / 1326 else 0

/-- The position where the probability of drawing the first black ace is maximized --/
def max_probability_position : ℕ := 1

/-- Theorem stating that the probability of drawing the first black ace is maximized at position 1 --/
theorem first_black_ace_most_likely_at_first_position :
  ∀ k, k ≥ 1 → k ≤ 51 → probability_first_black_ace max_probability_position ≥ probability_first_black_ace k :=
by
  sorry


end NUMINAMATH_CALUDE_first_black_ace_most_likely_at_first_position_l1927_192794


namespace NUMINAMATH_CALUDE_complex_unit_vector_l1927_192732

theorem complex_unit_vector (z : ℂ) (h : z = 3 + 4*I) : z / Complex.abs z = 3/5 + 4/5*I := by
  sorry

end NUMINAMATH_CALUDE_complex_unit_vector_l1927_192732


namespace NUMINAMATH_CALUDE_product_digit_sum_theorem_l1927_192742

/-- The number of digits in the second factor of the product -/
def k : ℕ := 55

/-- The second factor of the product -/
def second_factor (k : ℕ) : ℕ := (10^k - 1) / 9

/-- The product before subtraction -/
def product (k : ℕ) : ℕ := 9 * second_factor k

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem product_digit_sum_theorem :
  sum_of_digits (product k - 1) = 500 ∧
  ∀ m : ℕ, m ≠ k → sum_of_digits (product m - 1) ≠ 500 :=
sorry

end NUMINAMATH_CALUDE_product_digit_sum_theorem_l1927_192742


namespace NUMINAMATH_CALUDE_quadratic_polynomial_inequality_l1927_192780

/-- A quadratic polynomial with non-negative coefficients -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonneg : 0 ≤ a
  b_nonneg : 0 ≤ b
  c_nonneg : 0 ≤ c

/-- The value of a quadratic polynomial at a given point -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Theorem: For any quadratic polynomial with non-negative coefficients and any real numbers x and y,
    the square of the polynomial evaluated at xy is less than or equal to 
    the product of the polynomial evaluated at x^2 and y^2 -/
theorem quadratic_polynomial_inequality (p : QuadraticPolynomial) (x y : ℝ) :
  (p.eval (x * y))^2 ≤ (p.eval (x^2)) * (p.eval (y^2)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_inequality_l1927_192780


namespace NUMINAMATH_CALUDE_unique_relation_sum_l1927_192719

theorem unique_relation_sum (a b c : ℕ) : 
  ({a, b, c} : Set ℕ) = {1, 2, 3} →
  (((a ≠ 3 ∧ b ≠ 3 ∧ c = 3) ∨ (a ≠ 3 ∧ b = 3 ∧ c ≠ 3) ∨ (a = 3 ∧ b ≠ 3 ∧ c ≠ 3)) ∧
   ¬((a ≠ 3 ∧ b ≠ 3 ∧ c = 3) ∧ (a ≠ 3 ∧ b = 3 ∧ c ≠ 3)) ∧
   ¬((a ≠ 3 ∧ b ≠ 3 ∧ c = 3) ∧ (a = 3 ∧ b ≠ 3 ∧ c ≠ 3)) ∧
   ¬((a ≠ 3 ∧ b = 3 ∧ c ≠ 3) ∧ (a = 3 ∧ b ≠ 3 ∧ c ≠ 3))) →
  100 * a + 10 * b + c = 312 := by
sorry

end NUMINAMATH_CALUDE_unique_relation_sum_l1927_192719


namespace NUMINAMATH_CALUDE_quiz_goal_achievement_l1927_192711

theorem quiz_goal_achievement (total_quizzes : ℕ) (goal_percentage : ℚ)
  (completed_quizzes : ℕ) (as_scored : ℕ) (remaining_quizzes : ℕ)
  (h1 : total_quizzes = 60)
  (h2 : goal_percentage = 85 / 100)
  (h3 : completed_quizzes = 40)
  (h4 : as_scored = 32)
  (h5 : remaining_quizzes = 20)
  (h6 : completed_quizzes + remaining_quizzes = total_quizzes) :
  (total_quizzes * goal_percentage).floor - as_scored = remaining_quizzes - 1 :=
sorry

end NUMINAMATH_CALUDE_quiz_goal_achievement_l1927_192711


namespace NUMINAMATH_CALUDE_prime_remainder_30_l1927_192743

theorem prime_remainder_30 (a : ℕ) (h_prime : Nat.Prime a) :
  ∃ (q r : ℕ), a = 30 * q + r ∧ 0 ≤ r ∧ r < 30 ∧ (Nat.Prime r ∨ r = 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_remainder_30_l1927_192743


namespace NUMINAMATH_CALUDE_point_coordinates_on_angle_l1927_192731

theorem point_coordinates_on_angle (α : Real) (P : Real × Real) :
  α = π / 4 →
  (P.1^2 + P.2^2 = 2) →
  P = (1, 1) := by
sorry

end NUMINAMATH_CALUDE_point_coordinates_on_angle_l1927_192731


namespace NUMINAMATH_CALUDE_matrix_cube_computation_l1927_192789

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -2; 2, 0]

theorem matrix_cube_computation :
  A ^ 3 = !![(-8), 0; 0, (-8)] := by sorry

end NUMINAMATH_CALUDE_matrix_cube_computation_l1927_192789


namespace NUMINAMATH_CALUDE_x27x_divisible_by_36_l1927_192728

def is_single_digit (n : ℕ) : Prop := n ≥ 0 ∧ n ≤ 9

def four_digit_number (x : ℕ) : ℕ := x * 1000 + 270 + x

theorem x27x_divisible_by_36 : 
  ∃! x : ℕ, is_single_digit x ∧ (four_digit_number x) % 36 = 0 :=
sorry

end NUMINAMATH_CALUDE_x27x_divisible_by_36_l1927_192728


namespace NUMINAMATH_CALUDE_range_of_a_l1927_192775

-- Define the propositions p and q
def p (x : ℝ) : Prop := x^2 + 2*x - 3 > 0
def q (x a : ℝ) : Prop := x > a

-- Define the theorem
theorem range_of_a :
  (∃ a : ℝ, (∀ x : ℝ, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x : ℝ, ¬(q x a) ∧ p x)) →
  (∀ a : ℝ, (∀ x : ℝ, ¬(p x) → ¬(q x a)) ∧ 
  (∃ x : ℝ, ¬(q x a) ∧ p x) → a ≥ 1) :=
by sorry

-- The range of a is [1, +∞)

end NUMINAMATH_CALUDE_range_of_a_l1927_192775


namespace NUMINAMATH_CALUDE_lottery_expected_months_l1927_192727

/-- Represents the lottery system for car permits -/
structure LotterySystem where
  initial_participants : ℕ
  permits_per_month : ℕ
  new_participants_per_month : ℕ

/-- Calculates the expected number of months to win a permit with constant probability -/
def expected_months_constant (system : LotterySystem) : ℝ :=
  10 -- The actual calculation is omitted

/-- Calculates the expected number of months to win a permit with quarterly variable probabilities -/
def expected_months_variable (system : LotterySystem) : ℝ :=
  10 -- The actual calculation is omitted

/-- The main theorem stating that both lottery systems result in an expected 10 months wait -/
theorem lottery_expected_months (system : LotterySystem) 
    (h1 : system.initial_participants = 300000)
    (h2 : system.permits_per_month = 30000)
    (h3 : system.new_participants_per_month = 30000) :
    expected_months_constant system = 10 ∧ expected_months_variable system = 10 := by
  sorry

#check lottery_expected_months

end NUMINAMATH_CALUDE_lottery_expected_months_l1927_192727


namespace NUMINAMATH_CALUDE_initial_deposit_proof_l1927_192773

/-- Calculates the final amount after simple interest --/
def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that the initial deposit was 6200, given the problem conditions --/
theorem initial_deposit_proof (rate : ℝ) : 
  (∃ (principal : ℝ), 
    simpleInterest principal rate 5 = 7200 ∧ 
    simpleInterest principal (rate + 0.03) 5 = 8130) → 
  (∃ (principal : ℝ), 
    simpleInterest principal rate 5 = 7200 ∧ 
    simpleInterest principal (rate + 0.03) 5 = 8130 ∧ 
    principal = 6200) := by
  sorry

#check initial_deposit_proof

end NUMINAMATH_CALUDE_initial_deposit_proof_l1927_192773


namespace NUMINAMATH_CALUDE_wall_bricks_count_l1927_192777

/-- Represents the wall construction scenario --/
structure WallConstruction where
  /-- Total number of bricks in the wall --/
  total_bricks : ℕ
  /-- Time taken by the first bricklayer alone (in hours) --/
  time_worker1 : ℕ
  /-- Time taken by the second bricklayer alone (in hours) --/
  time_worker2 : ℕ
  /-- Reduction in combined output when working together (in bricks per hour) --/
  output_reduction : ℕ
  /-- Actual time taken to complete the wall (in hours) --/
  actual_time : ℕ

/-- Theorem stating the number of bricks in the wall --/
theorem wall_bricks_count (w : WallConstruction) 
  (h1 : w.time_worker1 = 8)
  (h2 : w.time_worker2 = 12)
  (h3 : w.output_reduction = 15)
  (h4 : w.actual_time = 6) :
  w.total_bricks = 360 :=
sorry

end NUMINAMATH_CALUDE_wall_bricks_count_l1927_192777


namespace NUMINAMATH_CALUDE_intersects_x_axis_once_iff_a_eq_zero_or_one_or_nine_l1927_192762

/-- A function f(x) intersects the x-axis at only one point if and only if
    it has exactly one real root or it is a non-constant linear function. -/
def intersects_x_axis_once (f : ℝ → ℝ) : Prop :=
  (∃! x, f x = 0) ∨ (∃ m b, m ≠ 0 ∧ ∀ x, f x = m * x + b)

/-- The quadratic function f(x) = ax² + (a-3)x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 3) * x + 1

theorem intersects_x_axis_once_iff_a_eq_zero_or_one_or_nine :
  ∀ a : ℝ, intersects_x_axis_once (f a) ↔ a = 0 ∨ a = 1 ∨ a = 9 :=
sorry

end NUMINAMATH_CALUDE_intersects_x_axis_once_iff_a_eq_zero_or_one_or_nine_l1927_192762


namespace NUMINAMATH_CALUDE_zoo_bus_children_l1927_192761

/-- The number of children taking the bus to the zoo -/
def children_count : ℕ := 58

/-- The number of seats needed -/
def seats_needed : ℕ := 29

/-- The number of children per seat -/
def children_per_seat : ℕ := 2

/-- Theorem: The number of children taking the bus to the zoo is 58,
    given that they sit 2 children in every seat and need 29 seats in total. -/
theorem zoo_bus_children :
  children_count = seats_needed * children_per_seat :=
by sorry

end NUMINAMATH_CALUDE_zoo_bus_children_l1927_192761


namespace NUMINAMATH_CALUDE_tax_calculation_l1927_192748

/-- Calculate tax given income and tax rate -/
def calculate_tax (income : ℝ) (rate : ℝ) : ℝ :=
  income * rate

/-- Calculate total tax for given gross pay and tax brackets -/
def total_tax (gross_pay : ℝ) : ℝ :=
  let tax1 := calculate_tax 1500 0.10
  let tax2 := calculate_tax 2000 0.15
  let tax3 := calculate_tax (gross_pay - 1500 - 2000) 0.20
  tax1 + tax2 + tax3

/-- Apply standard deduction to total tax -/
def tax_after_deduction (total_tax : ℝ) (deduction : ℝ) : ℝ :=
  total_tax - deduction

theorem tax_calculation (gross_pay : ℝ) (deduction : ℝ) 
  (h1 : gross_pay = 4500)
  (h2 : deduction = 100) :
  tax_after_deduction (total_tax gross_pay) deduction = 550 := by
  sorry

#eval tax_after_deduction (total_tax 4500) 100

end NUMINAMATH_CALUDE_tax_calculation_l1927_192748


namespace NUMINAMATH_CALUDE_q_div_p_equals_48_l1927_192763

/-- The number of cards in the deck -/
def total_cards : ℕ := 52

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 13

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The number of cards for each number -/
def cards_per_number : ℕ := 4

/-- The probability of drawing all 5 cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The probability of drawing 4 cards of one number and 1 of another -/
def q : ℚ := (624 : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The theorem stating the ratio of q to p -/
theorem q_div_p_equals_48 : q / p = 48 := by sorry

end NUMINAMATH_CALUDE_q_div_p_equals_48_l1927_192763


namespace NUMINAMATH_CALUDE_annes_journey_l1927_192710

/-- Calculates the distance traveled given time and speed -/
def distance_traveled (time : ℝ) (speed : ℝ) : ℝ := time * speed

/-- Proves that wandering for 3 hours at 2 miles per hour results in a 6-mile journey -/
theorem annes_journey : distance_traveled 3 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_annes_journey_l1927_192710
