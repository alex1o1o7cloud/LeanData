import Mathlib

namespace NUMINAMATH_CALUDE_dans_initial_money_l1370_137049

/-- Given that Dan bought a candy bar for $7 and a chocolate for $6,
    and spent $13 in total, prove that his initial amount was $13. -/
theorem dans_initial_money :
  ∀ (candy_price chocolate_price total_spent initial_amount : ℕ),
    candy_price = 7 →
    chocolate_price = 6 →
    total_spent = 13 →
    total_spent = candy_price + chocolate_price →
    initial_amount = total_spent →
    initial_amount = 13 :=
by
  sorry

end NUMINAMATH_CALUDE_dans_initial_money_l1370_137049


namespace NUMINAMATH_CALUDE_school_supplies_expenditure_l1370_137013

theorem school_supplies_expenditure (winnings : ℚ) : 
  (winnings / 2 : ℚ) + -- Amount spent on supplies
  ((winnings - winnings / 2) * 3 / 8 : ℚ) + -- Amount saved
  (2500 : ℚ) -- Remaining amount
  = winnings →
  (winnings / 2 : ℚ) = 4000 := by sorry

end NUMINAMATH_CALUDE_school_supplies_expenditure_l1370_137013


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l1370_137060

theorem quadratic_equation_root (b : ℝ) : 
  (2 * (4 : ℝ)^2 + b * 4 - 44 = 0) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l1370_137060


namespace NUMINAMATH_CALUDE_absolute_difference_of_opposite_signs_l1370_137096

theorem absolute_difference_of_opposite_signs (m n : ℤ) : 
  (abs m = 5) → (abs n = 2) → (m * n < 0) → abs (m - n) = 7 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_of_opposite_signs_l1370_137096


namespace NUMINAMATH_CALUDE_fraction_inequality_solution_set_l1370_137020

theorem fraction_inequality_solution_set (x : ℝ) : 
  (1 / x > 1) ↔ (0 < x ∧ x < 1) :=
sorry

end NUMINAMATH_CALUDE_fraction_inequality_solution_set_l1370_137020


namespace NUMINAMATH_CALUDE_leak_drain_time_l1370_137086

-- Define the pump fill rate
def pump_rate : ℚ := 1 / 2

-- Define the time it takes to fill the tank with the leak
def fill_time_with_leak : ℚ := 17 / 8

-- Define the leak rate
def leak_rate : ℚ := pump_rate - (1 / fill_time_with_leak)

-- Theorem to prove
theorem leak_drain_time (pump_rate : ℚ) (fill_time_with_leak : ℚ) (leak_rate : ℚ) :
  pump_rate = 1 / 2 →
  fill_time_with_leak = 17 / 8 →
  leak_rate = pump_rate - (1 / fill_time_with_leak) →
  (1 / leak_rate) = 34 := by
  sorry

end NUMINAMATH_CALUDE_leak_drain_time_l1370_137086


namespace NUMINAMATH_CALUDE_blue_cards_count_l1370_137088

theorem blue_cards_count (red_cards : ℕ) (blue_prob : ℚ) (blue_cards : ℕ) : 
  red_cards = 8 →
  blue_prob = 6/10 →
  (blue_cards : ℚ) / (blue_cards + red_cards) = blue_prob →
  blue_cards = 12 := by
sorry

end NUMINAMATH_CALUDE_blue_cards_count_l1370_137088


namespace NUMINAMATH_CALUDE_sin_45_is_proposition_l1370_137023

-- Define what a proposition is in this context
def is_proposition (s : String) : Prop := 
  ∃ (truth_value : Bool), (s ≠ "") ∧ (truth_value = true ∨ truth_value = false)

-- State the theorem
theorem sin_45_is_proposition : 
  is_proposition "sin(45°) = 1" := by
  sorry

end NUMINAMATH_CALUDE_sin_45_is_proposition_l1370_137023


namespace NUMINAMATH_CALUDE_functional_equation_solution_l1370_137048

theorem functional_equation_solution 
  (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * Real.cos y) :
  ∀ t : ℝ, f t = f 0 * Real.cos t + f (Real.pi / 2) * Real.sin t :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l1370_137048


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1370_137077

theorem modulus_of_complex_fraction : 
  let z : ℂ := (2 - I) / (1 + 2*I)
  Complex.abs z = 1 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l1370_137077


namespace NUMINAMATH_CALUDE_equation_solution_l1370_137063

theorem equation_solution : ∃ (x : ℚ), 5*x - 3*x = 420 - 10*(x + 2) ∧ x = 100/3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1370_137063


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1370_137053

theorem quadratic_roots_sum_product (k p : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - k * x + p = 0 ∧ 3 * y^2 - k * y + p = 0 ∧ x + y = -3 ∧ x * y = -6) →
  k + p = -27 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l1370_137053


namespace NUMINAMATH_CALUDE_wendy_scholarship_amount_l1370_137081

theorem wendy_scholarship_amount 
  (wendy kelly nina : ℕ)  -- Scholarship amounts for each person
  (h1 : nina = kelly - 8000)  -- Nina's scholarship is $8000 less than Kelly's
  (h2 : kelly = 2 * wendy)    -- Kelly's scholarship is twice Wendy's
  (h3 : wendy + kelly + nina = 92000)  -- Total scholarship amount
  : wendy = 20000 := by
  sorry

end NUMINAMATH_CALUDE_wendy_scholarship_amount_l1370_137081


namespace NUMINAMATH_CALUDE_bells_sync_theorem_l1370_137083

/-- The time in minutes when all bells ring together -/
def bell_sync_time : ℕ := 360

/-- Periods of bell ringing for each institution in minutes -/
def museum_period : ℕ := 18
def library_period : ℕ := 24
def town_hall_period : ℕ := 30
def hospital_period : ℕ := 36

theorem bells_sync_theorem :
  bell_sync_time = Nat.lcm museum_period (Nat.lcm library_period (Nat.lcm town_hall_period hospital_period)) ∧
  bell_sync_time % museum_period = 0 ∧
  bell_sync_time % library_period = 0 ∧
  bell_sync_time % town_hall_period = 0 ∧
  bell_sync_time % hospital_period = 0 :=
by sorry

end NUMINAMATH_CALUDE_bells_sync_theorem_l1370_137083


namespace NUMINAMATH_CALUDE_focus_line_dot_product_fixed_point_existence_l1370_137011

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define a line that intersects the parabola at two distinct points
def intersecting_line (t b : ℝ) (x y : ℝ) : Prop := x = t*y + b

-- Define the dot product of two vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1*x2 + y1*y2

-- Part I: Theorem for line passing through focus
theorem focus_line_dot_product (t : ℝ) (x1 y1 x2 y2 : ℝ) :
  parabola x1 y1 ∧ parabola x2 y2 ∧
  intersecting_line t 1 x1 y1 ∧ intersecting_line t 1 x2 y2 ∧
  (x1 ≠ x2 ∨ y1 ≠ y2) →
  dot_product x1 y1 x2 y2 = -3 :=
sorry

-- Part II: Theorem for fixed point
theorem fixed_point_existence (t b : ℝ) (x1 y1 x2 y2 : ℝ) :
  parabola x1 y1 ∧ parabola x2 y2 ∧
  intersecting_line t b x1 y1 ∧ intersecting_line t b x2 y2 ∧
  (x1 ≠ x2 ∨ y1 ≠ y2) ∧
  dot_product x1 y1 x2 y2 = -4 →
  b = 2 :=
sorry

end NUMINAMATH_CALUDE_focus_line_dot_product_fixed_point_existence_l1370_137011


namespace NUMINAMATH_CALUDE_find_A_l1370_137029

theorem find_A : ∃ A : ℤ, A + 10 = 15 → A = 5 := by
  sorry

end NUMINAMATH_CALUDE_find_A_l1370_137029


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l1370_137093

theorem rectangle_perimeter (area : ℝ) (side : ℝ) (h1 : area = 108) (h2 : side = 12) :
  2 * (side + area / side) = 42 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l1370_137093


namespace NUMINAMATH_CALUDE_residue_of_neg1237_mod37_l1370_137037

theorem residue_of_neg1237_mod37 : ∃ (k : ℤ), -1237 = 37 * k + 21 ∧ (0 ≤ 21 ∧ 21 < 37) := by
  sorry

end NUMINAMATH_CALUDE_residue_of_neg1237_mod37_l1370_137037


namespace NUMINAMATH_CALUDE_comprehensive_score_example_l1370_137016

/-- Calculates the comprehensive score given regular assessment and final exam scores and their weightings -/
def comprehensive_score (regular_score : ℝ) (final_score : ℝ) (regular_weight : ℝ) (final_weight : ℝ) : ℝ :=
  regular_score * regular_weight + final_score * final_weight

/-- Proves that the comprehensive score is 91 given the specified scores and weightings -/
theorem comprehensive_score_example : 
  comprehensive_score 95 90 0.2 0.8 = 91 := by
  sorry

end NUMINAMATH_CALUDE_comprehensive_score_example_l1370_137016


namespace NUMINAMATH_CALUDE_constant_term_expansion_l1370_137045

theorem constant_term_expansion :
  let f := fun (x : ℝ) => (x - 1/x)^6
  ∃ (c : ℝ), c = -20 ∧ 
    ∀ (x : ℝ), x ≠ 0 → (∃ (g : ℝ → ℝ), f x = c + x * g x + (1/x) * g (1/x)) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l1370_137045


namespace NUMINAMATH_CALUDE_european_stamps_count_l1370_137068

/-- Represents the number of stamps from Asian countries -/
def asian_stamps : ℕ := sorry

/-- Represents the number of stamps from European countries -/
def european_stamps : ℕ := sorry

/-- The total number of stamps Jesse has -/
def total_stamps : ℕ := 444

/-- European stamps are three times the number of Asian stamps -/
axiom european_triple_asian : european_stamps = 3 * asian_stamps

/-- The sum of Asian and European stamps equals the total stamps -/
axiom sum_equals_total : asian_stamps + european_stamps = total_stamps

/-- Theorem stating that the number of European stamps is 333 -/
theorem european_stamps_count : european_stamps = 333 := by sorry

end NUMINAMATH_CALUDE_european_stamps_count_l1370_137068


namespace NUMINAMATH_CALUDE_crane_sling_diameter_l1370_137019

/-- Represents the problem of determining the smallest safe rope diameter for a crane sling. -/
theorem crane_sling_diameter
  (M : ℝ) -- Mass of the load in tons
  (n : ℕ) -- Number of slings
  (α : ℝ) -- Angle of each sling with vertical in radians
  (k : ℝ) -- Safety factor
  (q : ℝ) -- Maximum load per thread in N/mm²
  (g : ℝ) -- Free fall acceleration in m/s²
  (h : M = 20)
  (hn : n = 3)
  (hα : α = Real.pi / 6) -- 30° in radians
  (hk : k = 6)
  (hq : q = 1000)
  (hg : g = 10)
  : ∃ (D : ℕ), D = 26 ∧ 
    D = ⌈(4 * (k * M * g) / (n * q * π * Real.cos α))^(1/2) * 1000⌉ ∧
    ∀ (D' : ℕ), D' < D → 
      D' < ⌈(4 * (k * M * g) / (n * q * π * Real.cos α))^(1/2) * 1000⌉ :=
sorry

end NUMINAMATH_CALUDE_crane_sling_diameter_l1370_137019


namespace NUMINAMATH_CALUDE_semicircle_chord_length_l1370_137094

theorem semicircle_chord_length (d : ℝ) (h : d > 0) :
  let r := d / 2
  let remaining_area := π * r^2 / 2 - π * (d/4)^2
  remaining_area = 16 * π^3 →
  2 * Real.sqrt (r^2 - (d/4)^2) = 32 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_chord_length_l1370_137094


namespace NUMINAMATH_CALUDE_fraction_zero_implies_a_equals_two_l1370_137018

theorem fraction_zero_implies_a_equals_two (a : ℝ) : 
  (a^2 - 4) / (a + 2) = 0 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_implies_a_equals_two_l1370_137018


namespace NUMINAMATH_CALUDE_diana_video_game_time_l1370_137069

def video_game_time_per_hour_read : ℕ := 30
def raise_percentage : ℚ := 0.2
def chores_for_bonus_time : ℕ := 2
def bonus_time_per_chore_set : ℕ := 10
def max_bonus_time_from_chores : ℕ := 60
def hours_read : ℕ := 8
def chores_completed : ℕ := 10

theorem diana_video_game_time : 
  let base_time := hours_read * video_game_time_per_hour_read
  let raised_time := base_time + (base_time * raise_percentage).floor
  let chore_bonus_time := min (chores_completed / chores_for_bonus_time * bonus_time_per_chore_set) max_bonus_time_from_chores
  raised_time + chore_bonus_time = 338 := by
sorry

end NUMINAMATH_CALUDE_diana_video_game_time_l1370_137069


namespace NUMINAMATH_CALUDE_intersection_when_a_2_B_subset_A_condition_l1370_137000

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

-- Part 1: Intersection when a = 2
theorem intersection_when_a_2 : 
  A 2 ∩ B 2 = {x : ℝ | 4 < x ∧ x < 5} := by sorry

-- Part 2: Condition for B to be a subset of A
theorem B_subset_A_condition (a : ℝ) :
  a ≠ 1 →
  (B a ⊆ A a ↔ (1 < a ∧ a ≤ 3) ∨ a = -1) := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_2_B_subset_A_condition_l1370_137000


namespace NUMINAMATH_CALUDE_hoseok_workbook_days_l1370_137008

/-- The number of days Hoseok solved the workbook -/
def days_solved : ℕ := 12

/-- The number of pages Hoseok solves per day -/
def pages_per_day : ℕ := 4

/-- The total number of pages Hoseok has solved -/
def total_pages : ℕ := 48

/-- Theorem stating that the number of days Hoseok solved the workbook is correct -/
theorem hoseok_workbook_days : 
  days_solved = total_pages / pages_per_day :=
by sorry

end NUMINAMATH_CALUDE_hoseok_workbook_days_l1370_137008


namespace NUMINAMATH_CALUDE_min_value_of_f_l1370_137047

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 + a*x - 1) * Real.exp (x - 1)

theorem min_value_of_f (a : ℝ) :
  (∃ (h : ℝ), ∀ x, f a x ≥ f a (-2)) →
  (∃ (m : ℝ), ∀ x, f a x ≥ m ∧ ∃ y, f a y = m) →
  (∃ (m : ℝ), ∀ x, f a x ≥ m ∧ ∃ y, f a y = m ∧ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1370_137047


namespace NUMINAMATH_CALUDE_trigonometric_equality_l1370_137003

theorem trigonometric_equality (α β : ℝ) :
  (Real.cos α)^6 / (Real.cos β)^3 + (Real.sin α)^6 / (Real.sin β)^3 = 2 →
  (Real.sin β)^6 / (Real.sin α)^3 + (Real.cos β)^6 / (Real.cos α)^3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l1370_137003


namespace NUMINAMATH_CALUDE_saras_weekly_savings_l1370_137054

/-- Sara's weekly savings to match Jim's savings after 820 weeks -/
theorem saras_weekly_savings (sara_initial : ℕ) (jim_weekly : ℕ) (weeks : ℕ) : 
  sara_initial = 4100 → jim_weekly = 15 → weeks = 820 →
  ∃ (sara_weekly : ℕ), sara_initial + weeks * sara_weekly = weeks * jim_weekly := by
  sorry

#check saras_weekly_savings

end NUMINAMATH_CALUDE_saras_weekly_savings_l1370_137054


namespace NUMINAMATH_CALUDE_scarlett_fruit_salad_l1370_137025

theorem scarlett_fruit_salad (melon_weight berries_weight : ℚ) 
  (h1 : melon_weight = 0.25)
  (h2 : berries_weight = 0.38) :
  melon_weight + berries_weight = 0.63 := by
sorry

end NUMINAMATH_CALUDE_scarlett_fruit_salad_l1370_137025


namespace NUMINAMATH_CALUDE_geometric_sequence_third_term_l1370_137032

/-- Given a geometric sequence {aₙ} where a₁ = 1 and a₄ = 27, prove that a₃ = 9 -/
theorem geometric_sequence_third_term
  (a : ℕ → ℝ)  -- The geometric sequence
  (h1 : a 1 = 1)  -- First term is 1
  (h4 : a 4 = 27)  -- Fourth term is 27
  (h_geom : ∀ n : ℕ, n > 0 → ∃ q : ℝ, a n = a 1 * q^(n-1))  -- Definition of geometric sequence
  : a 3 = 9 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_third_term_l1370_137032


namespace NUMINAMATH_CALUDE_max_side_length_of_triangle_l1370_137040

theorem max_side_length_of_triangle (a b c : ℕ) : 
  a < b ∧ b < c ∧  -- Three different integer side lengths
  a + b + c = 24 ∧ -- Perimeter is 24 units
  a + b > c →      -- Triangle inequality
  c ≤ 10 :=        -- Maximum length of any side is 10
by sorry

end NUMINAMATH_CALUDE_max_side_length_of_triangle_l1370_137040


namespace NUMINAMATH_CALUDE_initial_seashell_count_l1370_137036

theorem initial_seashell_count (henry paul leo : ℕ) : 
  henry = 11 →
  paul = 24 →
  henry + paul + (3/4 * leo) = 53 →
  henry + paul + leo = 59 :=
by sorry

end NUMINAMATH_CALUDE_initial_seashell_count_l1370_137036


namespace NUMINAMATH_CALUDE_canDisplay_totalCans_l1370_137043

/-- The number of cans in each layer forms an arithmetic sequence -/
def canSequence (n : ℕ) : ℕ := 35 - 3 * n

/-- The total number of layers in the display -/
def numLayers : ℕ := 12

/-- The total number of cans in the display -/
def totalCans : ℕ := (numLayers * (canSequence 0 + canSequence (numLayers - 1))) / 2

theorem canDisplay_totalCans : totalCans = 216 := by
  sorry

end NUMINAMATH_CALUDE_canDisplay_totalCans_l1370_137043


namespace NUMINAMATH_CALUDE_sequence_general_formula_l1370_137056

theorem sequence_general_formula (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) :
  a 1 = 3 ∧
  (∀ n : ℕ+, S n = 2 * n * a (n + 1) - 3 * n^2 - 4 * n) →
  ∀ n : ℕ+, a n = 2 * n + 1 :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_formula_l1370_137056


namespace NUMINAMATH_CALUDE_factorization_sum_l1370_137012

theorem factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 17*x + 72 = (x + a)*(x + b)) →
  (∀ x : ℝ, x^2 + 8*x - 63 = (x + b)*(x - c)) →
  a + b + c = 24 := by
  sorry

end NUMINAMATH_CALUDE_factorization_sum_l1370_137012


namespace NUMINAMATH_CALUDE_cone_volume_from_half_sector_l1370_137039

theorem cone_volume_from_half_sector (r : ℝ) (h : r = 6) :
  let base_radius := r / 2
  let cone_height := Real.sqrt (r^2 - base_radius^2)
  (1/3) * π * base_radius^2 * cone_height = 9 * π * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_from_half_sector_l1370_137039


namespace NUMINAMATH_CALUDE_market_qualified_product_probability_l1370_137070

theorem market_qualified_product_probability :
  let market_share_A : ℝ := 0.8
  let market_share_B : ℝ := 0.2
  let qualification_rate_A : ℝ := 0.75
  let qualification_rate_B : ℝ := 0.8
  market_share_A * qualification_rate_A + market_share_B * qualification_rate_B = 0.76 :=
by sorry

end NUMINAMATH_CALUDE_market_qualified_product_probability_l1370_137070


namespace NUMINAMATH_CALUDE_sqrt_equality_implies_y_value_l1370_137084

theorem sqrt_equality_implies_y_value (y : ℝ) :
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 8 → y = 40 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equality_implies_y_value_l1370_137084


namespace NUMINAMATH_CALUDE_rice_A_more_stable_than_B_l1370_137046

/-- Represents a rice variety with its yield variance -/
structure RiceVariety where
  name : String
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines when a rice variety is considered more stable than another -/
def more_stable (a b : RiceVariety) : Prop :=
  a.variance < b.variance

/-- The main theorem stating that rice variety A is more stable than B -/
theorem rice_A_more_stable_than_B (A B : RiceVariety) 
  (hA : A.name = "A" ∧ A.variance = 794)
  (hB : B.name = "B" ∧ B.variance = 958) : 
  more_stable A B := by
  sorry

end NUMINAMATH_CALUDE_rice_A_more_stable_than_B_l1370_137046


namespace NUMINAMATH_CALUDE_smallest_square_pieces_l1370_137024

/-- Represents the area of a single piece -/
def piece_area : ℝ := sorry

/-- Represents the shape of a single piece -/
structure Piece where
  area : ℝ := piece_area
  -- Additional properties to define the shape could be added here

/-- Predicate to check if a number of pieces can form a complete square -/
def can_form_square (n : ℕ) : Prop :=
  ∃ (side_length : ℝ), (n : ℝ) * piece_area = side_length * side_length ∧
  -- Additional condition to ensure pieces fit without gaps or overlaps
  sorry

/-- The smallest number of pieces that can form a square -/
def min_pieces : ℕ := 20

theorem smallest_square_pieces :
  can_form_square min_pieces ∧
  ∀ n < min_pieces, ¬(can_form_square n) :=
sorry

end NUMINAMATH_CALUDE_smallest_square_pieces_l1370_137024


namespace NUMINAMATH_CALUDE_tan_x_minus_pi_4_eq_one_third_l1370_137090

theorem tan_x_minus_pi_4_eq_one_third (x : ℝ) 
  (h1 : x ∈ Set.Ioo 0 Real.pi) 
  (h2 : Real.cos (2 * x - Real.pi / 2) = Real.sin x ^ 2) : 
  Real.tan (x - Real.pi / 4) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_x_minus_pi_4_eq_one_third_l1370_137090


namespace NUMINAMATH_CALUDE_total_subjects_is_41_l1370_137058

/-- The number of subjects taken by Monica -/
def monica_subjects : ℕ := 10

/-- The number of subjects taken by Marius -/
def marius_subjects : ℕ := monica_subjects + 4

/-- The number of subjects taken by Millie -/
def millie_subjects : ℕ := marius_subjects + 3

/-- The total number of subjects taken by all three students -/
def total_subjects : ℕ := monica_subjects + marius_subjects + millie_subjects

/-- Theorem stating that the total number of subjects is 41 -/
theorem total_subjects_is_41 : total_subjects = 41 := by
  sorry

end NUMINAMATH_CALUDE_total_subjects_is_41_l1370_137058


namespace NUMINAMATH_CALUDE_derivative_x_minus_reciprocal_l1370_137099

/-- The derivative of f(x) = x - 1/x is f'(x) = 1 + 1/x^2 -/
theorem derivative_x_minus_reciprocal (x : ℝ) (hx : x ≠ 0) :
  deriv (fun x => x - 1 / x) x = 1 + 1 / x^2 := by
  sorry

end NUMINAMATH_CALUDE_derivative_x_minus_reciprocal_l1370_137099


namespace NUMINAMATH_CALUDE_floor_sum_sqrt_equals_floor_sqrt_9n_plus_8_l1370_137067

theorem floor_sum_sqrt_equals_floor_sqrt_9n_plus_8 (n : ℕ) :
  ⌊Real.sqrt n + Real.sqrt (n + 1) + Real.sqrt (n + 2)⌋ = ⌊Real.sqrt (9 * n + 8)⌋ :=
sorry

end NUMINAMATH_CALUDE_floor_sum_sqrt_equals_floor_sqrt_9n_plus_8_l1370_137067


namespace NUMINAMATH_CALUDE_no_divisors_between_30_and_40_l1370_137002

theorem no_divisors_between_30_and_40 : ∀ n : ℕ, 30 < n → n < 40 → ¬(2^28 - 1) % n = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_divisors_between_30_and_40_l1370_137002


namespace NUMINAMATH_CALUDE_triangle_area_l1370_137065

open Real

/-- Given a triangle ABC where angle A is π/6 and the dot product of vectors AB and AC
    equals the tangent of angle A, prove that the area of the triangle is 1/6. -/
theorem triangle_area (A B C : ℝ × ℝ) : 
  let angle_A : ℝ := π / 6
  let AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)
  let AC : ℝ × ℝ := (C.1 - A.1, C.2 - A.2)
  AB.1 * AC.1 + AB.2 * AC.2 = tan angle_A →
  abs ((B.1 - A.1) * (C.2 - A.2) - (B.2 - A.2) * (C.1 - A.1)) / 2 = 1/6 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l1370_137065


namespace NUMINAMATH_CALUDE_matrix_product_AB_l1370_137042

def A : Matrix (Fin 2) (Fin 2) ℝ := !![3, -2; 4, 0]
def B : Matrix (Fin 2) (Fin 1) ℝ := !![5; -1]

theorem matrix_product_AB :
  A * B = !![17; 20] := by sorry

end NUMINAMATH_CALUDE_matrix_product_AB_l1370_137042


namespace NUMINAMATH_CALUDE_unique_solution_l1370_137038

-- Define the two equations
def equation1 (x y : ℝ) : Prop := (x + y - 5) * (2 * x - 3 * y + 5) = 0
def equation2 (x y : ℝ) : Prop := (x - y + 1) * (3 * x + 2 * y - 12) = 0

-- Define a solution as a point satisfying both equations
def is_solution (p : ℝ × ℝ) : Prop :=
  equation1 p.1 p.2 ∧ equation2 p.1 p.2

-- Theorem stating that there is exactly one solution
theorem unique_solution : ∃! p : ℝ × ℝ, is_solution p :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l1370_137038


namespace NUMINAMATH_CALUDE_kyle_pe_laps_l1370_137021

/-- The number of laps Kyle jogged during track practice -/
def track_laps : ℝ := 2.12

/-- The total number of laps Kyle jogged -/
def total_laps : ℝ := 3.25

/-- The number of laps Kyle jogged in P.E. class -/
def pe_laps : ℝ := total_laps - track_laps

theorem kyle_pe_laps : pe_laps = 1.13 := by
  sorry

end NUMINAMATH_CALUDE_kyle_pe_laps_l1370_137021


namespace NUMINAMATH_CALUDE_channels_taken_away_proof_l1370_137097

/-- Calculates the number of channels initially taken away --/
def channels_taken_away (initial_channels : ℕ) 
  (replaced_channels : ℕ) (reduced_channels : ℕ) 
  (sports_package : ℕ) (supreme_sports : ℕ) (final_channels : ℕ) : ℕ :=
  initial_channels + replaced_channels - reduced_channels + sports_package + supreme_sports - final_channels

/-- Proves that 20 channels were initially taken away --/
theorem channels_taken_away_proof : 
  channels_taken_away 150 12 10 8 7 147 = 20 := by sorry

end NUMINAMATH_CALUDE_channels_taken_away_proof_l1370_137097


namespace NUMINAMATH_CALUDE_complex_product_magnitude_l1370_137041

theorem complex_product_magnitude (c d : ℂ) (x : ℝ) :
  Complex.abs c = 3 →
  Complex.abs d = 5 →
  c * d = x - 3 * Complex.I →
  x = 6 * Real.sqrt 6 :=
by
  sorry

end NUMINAMATH_CALUDE_complex_product_magnitude_l1370_137041


namespace NUMINAMATH_CALUDE_smallest_percent_increase_l1370_137091

-- Define the values for the relevant questions
def value : Nat → ℕ
| 1 => 100
| 2 => 300
| 3 => 400
| 4 => 700
| 12 => 180000
| 13 => 360000
| 14 => 720000
| 15 => 1440000
| _ => 0  -- Default case, not used in our problem

-- Define the percent increase function
def percent_increase (a b : ℕ) : ℚ :=
  (b - a : ℚ) / a * 100

-- Theorem statement
theorem smallest_percent_increase :
  let increase_1_2 := percent_increase (value 1) (value 2)
  let increase_2_3 := percent_increase (value 2) (value 3)
  let increase_3_4 := percent_increase (value 3) (value 4)
  let increase_12_13 := percent_increase (value 12) (value 13)
  let increase_14_15 := percent_increase (value 14) (value 15)
  increase_2_3 < increase_1_2 ∧
  increase_2_3 < increase_3_4 ∧
  increase_2_3 < increase_12_13 ∧
  increase_2_3 < increase_14_15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_percent_increase_l1370_137091


namespace NUMINAMATH_CALUDE_expression_evaluation_l1370_137064

theorem expression_evaluation :
  (45 + 15)^2 - (45^2 + 15^2 + 2 * 45 * 5) = 900 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1370_137064


namespace NUMINAMATH_CALUDE_value_of_x_l1370_137078

theorem value_of_x (w v u x : ℤ) 
  (hw : w = 50)
  (hv : v = 3 * w + 30)
  (hu : u = v - 15)
  (hx : x = 2 * u + 12) : x = 342 := by
  sorry

end NUMINAMATH_CALUDE_value_of_x_l1370_137078


namespace NUMINAMATH_CALUDE_total_votes_l1370_137076

theorem total_votes (votes_for votes_against total : ℕ) : 
  votes_for = votes_against + 66 →
  votes_against = (40 * total) / 100 →
  votes_for + votes_against = total →
  total = 330 := by
sorry

end NUMINAMATH_CALUDE_total_votes_l1370_137076


namespace NUMINAMATH_CALUDE_tourist_attraction_arrangements_l1370_137034

def total_attractions : ℕ := 10
def daytime_attractions : ℕ := 8
def nighttime_attractions : ℕ := 2
def selected_attractions : ℕ := 5
def day1_slots : ℕ := 3
def day2_slots : ℕ := 2

theorem tourist_attraction_arrangements :
  (∃ (arrangements_with_A_or_B : ℕ) 
      (arrangements_A_and_B_same_day : ℕ) 
      (arrangements_without_A_and_B_together : ℕ),
    arrangements_with_A_or_B = 2352 ∧
    arrangements_A_and_B_same_day = 28560 ∧
    arrangements_without_A_and_B_together = 2352) := by
  sorry

end NUMINAMATH_CALUDE_tourist_attraction_arrangements_l1370_137034


namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l1370_137062

theorem existence_of_special_sequence : ∃ (a b c : ℝ), 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
  (a + b + c = 6) ∧
  (b - a = c - b) ∧
  ((a^2 = b * c) ∨ (b^2 = a * c) ∨ (c^2 = a * b)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l1370_137062


namespace NUMINAMATH_CALUDE_sequence_reappearance_l1370_137072

def letter_cycle_length : ℕ := 7
def digit_cycle_length : ℕ := 4

theorem sequence_reappearance :
  Nat.lcm letter_cycle_length digit_cycle_length = 28 := by
  sorry

#check sequence_reappearance

end NUMINAMATH_CALUDE_sequence_reappearance_l1370_137072


namespace NUMINAMATH_CALUDE_absent_students_sum_l1370_137031

/-- Proves that the sum of absent students over three days equals 200 --/
theorem absent_students_sum (T : ℕ) (A1 A2 A3 : ℕ) : 
  T = 280 →
  A3 = T / 7 →
  A2 = 2 * A3 →
  T - A2 + 40 = T - A1 →
  A1 + A2 + A3 = 200 := by
  sorry

end NUMINAMATH_CALUDE_absent_students_sum_l1370_137031


namespace NUMINAMATH_CALUDE_minimum_economic_loss_l1370_137009

def repair_times : List Nat := [12, 17, 8, 18, 23, 30, 14]
def num_workers : Nat := 3
def loss_per_minute : Nat := 2

def optimal_allocation (times : List Nat) (workers : Nat) : List (List Nat) :=
  sorry

def total_waiting_time (allocation : List (List Nat)) : Nat :=
  sorry

theorem minimum_economic_loss :
  let allocation := optimal_allocation repair_times num_workers
  let total_wait := total_waiting_time allocation
  total_wait * loss_per_minute = 358 := by
  sorry

end NUMINAMATH_CALUDE_minimum_economic_loss_l1370_137009


namespace NUMINAMATH_CALUDE_proposition_p_negation_and_range_l1370_137080

theorem proposition_p_negation_and_range (a : ℝ) :
  (¬∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) ↔ 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) ∧ 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0 → 0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_proposition_p_negation_and_range_l1370_137080


namespace NUMINAMATH_CALUDE_mary_juan_income_ratio_l1370_137004

theorem mary_juan_income_ratio (juan tim mary : ℝ) 
  (h1 : mary = 1.4 * tim) 
  (h2 : tim = 0.6 * juan) : 
  mary = 0.84 * juan := by
  sorry

end NUMINAMATH_CALUDE_mary_juan_income_ratio_l1370_137004


namespace NUMINAMATH_CALUDE_stratified_sampling_equal_allocation_l1370_137055

/-- Represents a group of workers -/
structure WorkerGroup where
  total : Nat
  female : Nat

/-- Represents a stratified sampling scenario -/
structure StratifiedSample where
  groupA : WorkerGroup
  groupB : WorkerGroup
  totalSamples : Nat

/-- Theorem: In a stratified sampling scenario with two equal-sized strata,
    the number of samples drawn from each stratum is equal to half of the total sample size -/
theorem stratified_sampling_equal_allocation 
  (sample : StratifiedSample) 
  (h1 : sample.groupA.total = sample.groupB.total)
  (h2 : sample.totalSamples % 2 = 0) :
  ∃ (n : Nat), n = sample.totalSamples / 2 ∧ 
               n = sample.totalSamples - n :=
sorry

#check stratified_sampling_equal_allocation

end NUMINAMATH_CALUDE_stratified_sampling_equal_allocation_l1370_137055


namespace NUMINAMATH_CALUDE_set_c_forms_triangle_set_a_not_triangle_set_b_not_triangle_set_d_not_triangle_triangle_formation_result_l1370_137022

/-- A function that checks if three line segments can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that the set (4, 5, 7) can form a triangle --/
theorem set_c_forms_triangle : can_form_triangle 4 5 7 := by sorry

/-- Theorem stating that the set (1, 3, 4) cannot form a triangle --/
theorem set_a_not_triangle : ¬ can_form_triangle 1 3 4 := by sorry

/-- Theorem stating that the set (2, 2, 7) cannot form a triangle --/
theorem set_b_not_triangle : ¬ can_form_triangle 2 2 7 := by sorry

/-- Theorem stating that the set (3, 3, 6) cannot form a triangle --/
theorem set_d_not_triangle : ¬ can_form_triangle 3 3 6 := by sorry

/-- Main theorem combining all results --/
theorem triangle_formation_result :
  can_form_triangle 4 5 7 ∧
  ¬ can_form_triangle 1 3 4 ∧
  ¬ can_form_triangle 2 2 7 ∧
  ¬ can_form_triangle 3 3 6 := by sorry

end NUMINAMATH_CALUDE_set_c_forms_triangle_set_a_not_triangle_set_b_not_triangle_set_d_not_triangle_triangle_formation_result_l1370_137022


namespace NUMINAMATH_CALUDE_det_linear_combination_zero_l1370_137061

open Matrix

theorem det_linear_combination_zero
  (A B : Matrix (Fin 3) (Fin 3) ℝ)
  (h : A ^ 2 + B ^ 2 = 0) :
  ∀ (a b : ℝ), det (a • A + b • B) = 0 := by
sorry

end NUMINAMATH_CALUDE_det_linear_combination_zero_l1370_137061


namespace NUMINAMATH_CALUDE_fruiting_plants_given_away_l1370_137050

/-- Represents the number of plants in Roxy's garden -/
structure GardenState where
  flowering : ℕ
  fruiting : ℕ

/-- Calculates the total number of plants -/
def GardenState.total (s : GardenState) : ℕ := s.flowering + s.fruiting

def initial_state : GardenState :=
  { flowering := 7,
    fruiting := 2 * 7 }

def after_buying : GardenState :=
  { flowering := initial_state.flowering + 3,
    fruiting := initial_state.fruiting + 2 }

def plants_remaining : ℕ := 21

def flowering_given_away : ℕ := 1

theorem fruiting_plants_given_away :
  ∃ (x : ℕ), 
    after_buying.fruiting - x = plants_remaining - (after_buying.flowering - flowering_given_away) ∧
    x = 4 := by
  sorry

end NUMINAMATH_CALUDE_fruiting_plants_given_away_l1370_137050


namespace NUMINAMATH_CALUDE_dice_probability_l1370_137098

/-- The number of sides on each die -/
def sides : ℕ := 6

/-- The number of dice being rolled -/
def num_dice : ℕ := 6

/-- The probability of rolling all the same numbers -/
def prob_all_same : ℚ := 1 / (sides ^ (num_dice - 1))

/-- The probability of not rolling all the same numbers -/
def prob_not_all_same : ℚ := 1 - prob_all_same

theorem dice_probability :
  prob_not_all_same = 7775 / 7776 :=
sorry

end NUMINAMATH_CALUDE_dice_probability_l1370_137098


namespace NUMINAMATH_CALUDE_binomial_17_9_l1370_137074

theorem binomial_17_9 (h1 : Nat.choose 15 6 = 5005) (h2 : Nat.choose 15 8 = 6435) :
  Nat.choose 17 9 = 24310 := by
  sorry

end NUMINAMATH_CALUDE_binomial_17_9_l1370_137074


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l1370_137092

theorem quadratic_rewrite (k : ℝ) :
  ∃ (d r s : ℝ), 9 * k^2 - 6 * k + 15 = d * (k + r)^2 + s ∧ s / r = -42 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l1370_137092


namespace NUMINAMATH_CALUDE_correct_production_matching_equation_l1370_137066

/-- Represents a workshop producing bolts and nuts -/
structure Workshop where
  total_workers : ℕ
  bolt_production_rate : ℕ
  nut_production_rate : ℕ
  nuts_per_bolt : ℕ

/-- The equation for matching bolt and nut production in the workshop -/
def production_matching_equation (w : Workshop) (x : ℕ) : Prop :=
  2 * w.bolt_production_rate * x = w.nut_production_rate * (w.total_workers - x)

/-- Theorem stating the correct equation for matching bolt and nut production -/
theorem correct_production_matching_equation (w : Workshop) 
  (h1 : w.total_workers = 28)
  (h2 : w.bolt_production_rate = 12)
  (h3 : w.nut_production_rate = 18)
  (h4 : w.nuts_per_bolt = 2) :
  ∀ x, production_matching_equation w x ↔ 2 * 12 * x = 18 * (28 - x) :=
by
  sorry


end NUMINAMATH_CALUDE_correct_production_matching_equation_l1370_137066


namespace NUMINAMATH_CALUDE_binomial_150_150_l1370_137057

theorem binomial_150_150 : (150 : ℕ).choose 150 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_150_150_l1370_137057


namespace NUMINAMATH_CALUDE_not_prime_sum_product_l1370_137001

theorem not_prime_sum_product (a b c d : ℕ) 
  (h_pos : 0 < d ∧ d < c ∧ c < b ∧ b < a) 
  (h_eq : a * c + b * d = (b + d + a - c) * (b + d - a + c)) : 
  ¬ Nat.Prime (a * b + c * d) := by
  sorry

end NUMINAMATH_CALUDE_not_prime_sum_product_l1370_137001


namespace NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_intersection_A_and_B_l1370_137014

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x < 4}
def B : Set ℝ := {x | x - 3 ≥ 0}

-- Theorem for A ∪ B
theorem union_of_A_and_B : A ∪ B = {x | x ≥ -2} := by sorry

-- Theorem for Ā ∩ B
theorem complement_of_intersection_A_and_B : (A ∩ B)ᶜ = {x | x < 3 ∨ x ≥ 4} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_complement_of_intersection_A_and_B_l1370_137014


namespace NUMINAMATH_CALUDE_max_identical_bathrooms_l1370_137033

theorem max_identical_bathrooms (toilet_paper soap towels shower_gel shampoo toothpaste : ℕ) 
  (h1 : toilet_paper = 45)
  (h2 : soap = 30)
  (h3 : towels = 36)
  (h4 : shower_gel = 18)
  (h5 : shampoo = 27)
  (h6 : toothpaste = 24) :
  ∃ (max_bathrooms : ℕ), 
    max_bathrooms = 3 ∧ 
    (toilet_paper % max_bathrooms = 0) ∧
    (soap % max_bathrooms = 0) ∧
    (towels % max_bathrooms = 0) ∧
    (shower_gel % max_bathrooms = 0) ∧
    (shampoo % max_bathrooms = 0) ∧
    (toothpaste % max_bathrooms = 0) ∧
    ∀ (n : ℕ), n > max_bathrooms → 
      (toilet_paper % n ≠ 0) ∨
      (soap % n ≠ 0) ∨
      (towels % n ≠ 0) ∨
      (shower_gel % n ≠ 0) ∨
      (shampoo % n ≠ 0) ∨
      (toothpaste % n ≠ 0) :=
by
  sorry

end NUMINAMATH_CALUDE_max_identical_bathrooms_l1370_137033


namespace NUMINAMATH_CALUDE_mailing_cost_correct_l1370_137087

/-- The cost function for mailing a document -/
def mailing_cost (P : ℕ) : ℕ :=
  if P ≤ 5 then
    15 + 5 * (P - 1)
  else
    15 + 5 * (P - 1) + 2

/-- Theorem stating the correctness of the mailing cost function -/
theorem mailing_cost_correct (P : ℕ) :
  mailing_cost P =
    if P ≤ 5 then
      15 + 5 * (P - 1)
    else
      15 + 5 * (P - 1) + 2 :=
by
  sorry

/-- Lemma: The cost for the first kilogram is 15 cents -/
lemma first_kg_cost (P : ℕ) (h : P > 0) : mailing_cost P ≥ 15 :=
by
  sorry

/-- Lemma: Each subsequent kilogram costs 5 cents -/
lemma subsequent_kg_cost (P : ℕ) (h : P > 1) :
  mailing_cost P - mailing_cost (P - 1) = 5 :=
by
  sorry

/-- Lemma: Additional handling fee of 2 cents for documents over 5 kg -/
lemma handling_fee (P : ℕ) (h : P > 5) :
  mailing_cost P - mailing_cost 5 = 5 * (P - 5) + 2 :=
by
  sorry

end NUMINAMATH_CALUDE_mailing_cost_correct_l1370_137087


namespace NUMINAMATH_CALUDE_distance_between_intersection_points_l1370_137026

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 9

-- Define the line that intersects the circle
def intersecting_line (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | ∃ x y, p = (x, y) ∧ circle_C x y ∧ intersecting_line x y}

-- State the theorem
theorem distance_between_intersection_points :
  ∃ A B : ℝ × ℝ, A ∈ intersection_points ∧ B ∈ intersection_points ∧ A ≠ B ∧
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_intersection_points_l1370_137026


namespace NUMINAMATH_CALUDE_nine_pointed_star_angle_sum_l1370_137089

/-- A star polygon with n points, skipping k points between connections -/
structure StarPolygon where
  n : ℕ  -- number of points
  k : ℕ  -- number of points skipped

/-- The sum of angles at the tips of a star polygon -/
def sumOfTipAngles (star : StarPolygon) : ℝ :=
  sorry

/-- Theorem: The sum of angles at the tips of a 9-pointed star, skipping 3 points, is 720° -/
theorem nine_pointed_star_angle_sum :
  let star : StarPolygon := { n := 9, k := 3 }
  sumOfTipAngles star = 720 := by
  sorry

end NUMINAMATH_CALUDE_nine_pointed_star_angle_sum_l1370_137089


namespace NUMINAMATH_CALUDE_fraction_simplification_l1370_137030

theorem fraction_simplification :
  1 / (1 / (1/3)^1 + 1 / (1/3)^2 + 1 / (1/3)^3 + 1 / (1/3)^4) = 1 / 120 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1370_137030


namespace NUMINAMATH_CALUDE_units_digit_difference_l1370_137010

def is_positive_even_integer (p : ℕ) : Prop := p > 0 ∧ p % 2 = 0

def has_positive_units_digit (p : ℕ) : Prop := p % 10 > 0

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_difference (p : ℕ) 
  (h1 : is_positive_even_integer p) 
  (h2 : has_positive_units_digit p) 
  (h3 : units_digit (p + 5) = 1) : 
  units_digit (p^3) - units_digit (p^2) = 0 := by
sorry

end NUMINAMATH_CALUDE_units_digit_difference_l1370_137010


namespace NUMINAMATH_CALUDE_fundraiser_total_l1370_137005

/-- Calculates the total amount raised from cake sales and donations --/
def total_raised (num_cakes : ℕ) (slices_per_cake : ℕ) (price_per_slice : ℚ) 
                 (donation1_per_slice : ℚ) (donation2_per_slice : ℚ) : ℚ :=
  let total_slices := num_cakes * slices_per_cake
  let sales := total_slices * price_per_slice
  let donation1 := total_slices * donation1_per_slice
  let donation2 := total_slices * donation2_per_slice
  sales + donation1 + donation2

/-- Theorem stating that under given conditions, the total amount raised is $140 --/
theorem fundraiser_total : 
  total_raised 10 8 1 (1/2) (1/4) = 140 := by
  sorry

end NUMINAMATH_CALUDE_fundraiser_total_l1370_137005


namespace NUMINAMATH_CALUDE_triangle_perimeter_is_twelve_l1370_137007

/-- The line equation in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The triangle formed by a line and the coordinate axes -/
structure Triangle where
  line : Line

def Triangle.perimeter (t : Triangle) : ℝ :=
  sorry

theorem triangle_perimeter_is_twelve (t : Triangle) :
  t.line = { a := 1/3, b := 1/4, c := 1 } →
  t.perimeter = 12 :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_is_twelve_l1370_137007


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1370_137006

-- Define the ellipse
structure Ellipse where
  isTangentToXAxis : Bool
  isTangentToYAxis : Bool
  focus1 : ℝ × ℝ
  focus2 : ℝ × ℝ

-- Define the theorem
theorem ellipse_major_axis_length 
  (e : Ellipse) 
  (h1 : e.isTangentToXAxis = true) 
  (h2 : e.isTangentToYAxis = true)
  (h3 : e.focus1 = (2, -3 + Real.sqrt 13))
  (h4 : e.focus2 = (2, -3 - Real.sqrt 13)) :
  ∃ (majorAxisLength : ℝ), majorAxisLength = 6 :=
sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1370_137006


namespace NUMINAMATH_CALUDE_total_schedules_l1370_137052

/-- Represents the number of periods in a day -/
def total_periods : ℕ := 6

/-- Represents the number of morning periods -/
def morning_periods : ℕ := 3

/-- Represents the number of afternoon periods -/
def afternoon_periods : ℕ := 3

/-- Represents the total number of subjects -/
def total_subjects : ℕ := 6

/-- Represents the number of ways to schedule Mathematics in the morning and Art in the afternoon -/
def math_art_schedules : ℕ := morning_periods * afternoon_periods

/-- Represents the number of remaining subjects to be scheduled -/
def remaining_subjects : ℕ := total_subjects - 2

/-- Represents the number of remaining periods to schedule the remaining subjects -/
def remaining_periods : ℕ := total_periods - 2

/-- The main theorem stating the total number of possible schedules -/
theorem total_schedules : 
  math_art_schedules * (Nat.factorial remaining_subjects) = 216 :=
sorry

end NUMINAMATH_CALUDE_total_schedules_l1370_137052


namespace NUMINAMATH_CALUDE_intersection_equality_condition_l1370_137079

theorem intersection_equality_condition (p : ℝ) : 
  let A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
  let B : Set ℝ := {x | p + 1 ≤ x ∧ x ≤ 2*p - 1}
  (A ∩ B = B) ↔ p ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_intersection_equality_condition_l1370_137079


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l1370_137044

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l1370_137044


namespace NUMINAMATH_CALUDE_player_current_average_l1370_137095

/-- Represents a cricket player's statistics -/
structure PlayerStats where
  matches_played : ℕ
  current_average : ℝ
  desired_increase : ℝ
  next_match_runs : ℕ

/-- Theorem stating the player's current average given the conditions -/
theorem player_current_average (player : PlayerStats)
  (h1 : player.matches_played = 10)
  (h2 : player.desired_increase = 4)
  (h3 : player.next_match_runs = 78) :
  player.current_average = 34 := by
  sorry

#check player_current_average

end NUMINAMATH_CALUDE_player_current_average_l1370_137095


namespace NUMINAMATH_CALUDE_squares_not_always_congruent_l1370_137073

-- Define a square
structure Square where
  side_length : ℝ
  side_length_pos : side_length > 0

-- Define properties of squares
def Square.is_equiangular (s : Square) : Prop := True
def Square.is_rectangle (s : Square) : Prop := True
def Square.is_regular_polygon (s : Square) : Prop := True
def Square.is_similar_to (s1 s2 : Square) : Prop := True

-- Define congruence for squares
def Square.is_congruent_to (s1 s2 : Square) : Prop :=
  s1.side_length = s2.side_length

-- Theorem statement
theorem squares_not_always_congruent :
  ∃ (s1 s2 : Square),
    s1.is_equiangular ∧
    s1.is_rectangle ∧
    s1.is_regular_polygon ∧
    s2.is_equiangular ∧
    s2.is_rectangle ∧
    s2.is_regular_polygon ∧
    Square.is_similar_to s1 s2 ∧
    ¬ Square.is_congruent_to s1 s2 :=
by
  sorry

end NUMINAMATH_CALUDE_squares_not_always_congruent_l1370_137073


namespace NUMINAMATH_CALUDE_problem_statement_l1370_137085

theorem problem_statement (a b q r : ℕ) (ha : a > 0) (hb : b > 0) 
  (h_division : a^2 + b^2 = q * (a + b) + r) (h_constraint : q^2 + r = 2010) :
  a * b = 1643 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1370_137085


namespace NUMINAMATH_CALUDE_allans_balloons_prove_allans_balloons_l1370_137035

theorem allans_balloons (jake_balloons : ℕ) (difference : ℕ) : ℕ :=
  jake_balloons + difference

theorem prove_allans_balloons :
  allans_balloons 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_allans_balloons_prove_allans_balloons_l1370_137035


namespace NUMINAMATH_CALUDE_disk_space_remaining_l1370_137059

/-- Calculates the remaining disk space given total space and used space -/
def remaining_space (total : ℕ) (used : ℕ) : ℕ :=
  total - used

/-- Theorem: Given 28 GB total space and 26 GB used space, the remaining space is 2 GB -/
theorem disk_space_remaining :
  remaining_space 28 26 = 2 := by
  sorry

end NUMINAMATH_CALUDE_disk_space_remaining_l1370_137059


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1370_137027

theorem geometric_sequence_sum (a b c q : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c - a) = (a + b + c) * q ∧
  (c + a - b) = (a + b + c) * q^2 ∧
  (a + b - c) = (a + b + c) * q^3 →
  q^3 + q^2 + q = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1370_137027


namespace NUMINAMATH_CALUDE_extended_line_segment_vector_representation_l1370_137017

/-- Given a line segment AB extended to point P such that AP:PB = 7:5,
    prove that the position vector of P can be expressed as 
    P = (5/12)A + (7/12)B -/
theorem extended_line_segment_vector_representation 
  (A B P : ℝ × ℝ) -- A, B, and P are points in 2D space
  (h : (dist A P) / (dist P B) = 7 / 5) : -- AP:PB = 7:5
  ∃ (t u : ℝ), t = 5/12 ∧ u = 7/12 ∧ P = t • A + u • B :=
by sorry

end NUMINAMATH_CALUDE_extended_line_segment_vector_representation_l1370_137017


namespace NUMINAMATH_CALUDE_picture_distribution_l1370_137082

theorem picture_distribution (total : ℕ) (first_album : ℕ) (num_albums : ℕ) :
  total = 35 →
  first_album = 14 →
  num_albums = 3 →
  (total - first_album) % num_albums = 0 →
  (total - first_album) / num_albums = 7 := by
  sorry

end NUMINAMATH_CALUDE_picture_distribution_l1370_137082


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l1370_137015

theorem largest_integer_with_remainder : ∃ n : ℕ, 
  n < 200 ∧ 
  n % 7 = 4 ∧ 
  ∀ m : ℕ, m < 200 → m % 7 = 4 → m ≤ n :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l1370_137015


namespace NUMINAMATH_CALUDE_range_of_a_l1370_137071

-- Define the propositions p and q as functions of a
def p (a : ℝ) : Prop := ∀ x ∈ Set.Icc (-1) 1, 
  ∀ y ∈ Set.Icc (-1) 1, x ≤ y → (x^2 + 8*a*x + 1) ≤ (y^2 + 8*a*y + 1)

def q (a : ℝ) : Prop := ∃ x y : ℝ, x^2 / (a + 2) + y^2 / (a - 1) = 1

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  (¬(p a ∧ q a) ∧ (p a ∨ q a)) → ((-2 < a ∧ a < 1/4) ∨ a ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1370_137071


namespace NUMINAMATH_CALUDE_jose_maria_age_difference_jose_maria_age_difference_proof_l1370_137028

theorem jose_maria_age_difference : ℕ → ℕ → Prop :=
  fun jose_age maria_age =>
    (jose_age > maria_age) →
    (jose_age + maria_age = 40) →
    (maria_age = 14) →
    (jose_age - maria_age = 12)

-- The proof would go here, but we'll skip it as requested
theorem jose_maria_age_difference_proof : ∃ (j m : ℕ), jose_maria_age_difference j m :=
  sorry

end NUMINAMATH_CALUDE_jose_maria_age_difference_jose_maria_age_difference_proof_l1370_137028


namespace NUMINAMATH_CALUDE_solve_for_q_l1370_137075

theorem solve_for_q (p q : ℝ) (hp : p > 1) (hq : q > 1) 
  (h1 : 1/p + 1/q = 1) (h2 : p * q = 9) : q = (9 + 3 * Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_q_l1370_137075


namespace NUMINAMATH_CALUDE_inequality_proof_l1370_137051

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / Real.sqrt (a^2 + 8*b*c)) + (b / Real.sqrt (b^2 + 8*c*a)) + (c / Real.sqrt (c^2 + 8*a*b)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1370_137051
