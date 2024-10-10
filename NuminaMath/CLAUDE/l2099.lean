import Mathlib

namespace compound_bar_chart_must_have_legend_l2099_209926

/-- Represents a compound bar chart -/
structure CompoundBarChart where
  distinguishes_two_quantities : Bool
  uses_different_colors_or_patterns : Bool

/-- Theorem: A compound bar chart must have a clearly indicated legend -/
theorem compound_bar_chart_must_have_legend (chart : CompoundBarChart) 
  (h1 : chart.distinguishes_two_quantities = true)
  (h2 : chart.uses_different_colors_or_patterns = true) : 
  ∃ legend : Bool, legend = true :=
sorry

end compound_bar_chart_must_have_legend_l2099_209926


namespace boat_speed_in_still_water_l2099_209986

/-- Given a boat traveling downstream with a stream rate of 5 km/hr and covering 84 km in 4 hours,
    the speed of the boat in still water is 16 km/hr. -/
theorem boat_speed_in_still_water :
  ∀ (stream_rate : ℝ) (distance : ℝ) (time : ℝ) (boat_speed : ℝ),
    stream_rate = 5 →
    distance = 84 →
    time = 4 →
    distance = (boat_speed + stream_rate) * time →
    boat_speed = 16 := by
  sorry

end boat_speed_in_still_water_l2099_209986


namespace sales_composition_l2099_209934

/-- The percentage of sales that are not pens, pencils, or erasers -/
def other_sales_percentage (pen_sales pencil_sales eraser_sales : ℝ) : ℝ :=
  100 - (pen_sales + pencil_sales + eraser_sales)

/-- Theorem stating that the percentage of sales not consisting of pens, pencils, or erasers is 25% -/
theorem sales_composition 
  (pen_sales : ℝ) 
  (pencil_sales : ℝ) 
  (eraser_sales : ℝ) 
  (h1 : pen_sales = 25)
  (h2 : pencil_sales = 30)
  (h3 : eraser_sales = 20) :
  other_sales_percentage pen_sales pencil_sales eraser_sales = 25 :=
by
  sorry

end sales_composition_l2099_209934


namespace independence_test_incorrect_judgment_l2099_209972

/-- The chi-squared test statistic -/
def K_squared : ℝ := 4.05

/-- The significance level (α) for the test -/
def significance_level : ℝ := 0.05

/-- The critical value for the chi-squared distribution with 1 degree of freedom at 0.05 significance level -/
def critical_value : ℝ := 3.841

/-- The probability of incorrect judgment in an independence test -/
def probability_incorrect_judgment : ℝ := significance_level

theorem independence_test_incorrect_judgment :
  K_squared > critical_value →
  probability_incorrect_judgment = significance_level :=
sorry

end independence_test_incorrect_judgment_l2099_209972


namespace winter_sales_l2099_209946

/-- Proves that the number of pastries sold in winter is 3 million -/
theorem winter_sales (spring summer fall : ℕ) (total : ℝ) : 
  spring = 3 → 
  summer = 6 → 
  fall = 3 → 
  fall = (1/5 : ℝ) * total → 
  total - (spring + summer + fall : ℝ) = 3 := by
sorry

end winter_sales_l2099_209946


namespace sphere_between_inclined_planes_l2099_209939

/-- The distance from the center of a sphere to the horizontal plane when placed between two inclined planes -/
theorem sphere_between_inclined_planes 
  (r : ℝ) 
  (angle1 : ℝ) 
  (angle2 : ℝ) 
  (h_r : r = 2) 
  (h_angle1 : angle1 = π / 3)  -- 60 degrees in radians
  (h_angle2 : angle2 = π / 6)  -- 30 degrees in radians
  : ∃ (d : ℝ), d = Real.sqrt 3 + 1 ∧ d = 
    r * Real.sin ((π / 2 - angle1 - angle2) / 2 + angle2) :=
by sorry

end sphere_between_inclined_planes_l2099_209939


namespace dichromate_molecular_weight_l2099_209974

/-- Given that the molecular weight of 9 moles of Dichromate is 2664 g/mol,
    prove that the molecular weight of one mole of Dichromate is 296 g/mol. -/
theorem dichromate_molecular_weight :
  let mw_9_moles : ℝ := 2664 -- molecular weight of 9 moles in g/mol
  let num_moles : ℝ := 9 -- number of moles
  mw_9_moles / num_moles = 296 := by sorry

end dichromate_molecular_weight_l2099_209974


namespace third_chapter_pages_l2099_209997

theorem third_chapter_pages (total_pages first_chapter second_chapter : ℕ) 
  (h1 : total_pages = 125)
  (h2 : first_chapter = 66)
  (h3 : second_chapter = 35) :
  total_pages - (first_chapter + second_chapter) = 24 :=
by
  sorry

end third_chapter_pages_l2099_209997


namespace product_mod_eight_l2099_209904

theorem product_mod_eight : (55 * 57) % 8 = 7 := by
  sorry

end product_mod_eight_l2099_209904


namespace jerry_skit_first_character_lines_l2099_209963

/-- Represents the number of lines for each character in Jerry's skit script -/
structure SkitScript where
  first : ℕ
  second : ℕ
  third : ℕ

/-- The conditions of Jerry's skit script -/
def validScript (s : SkitScript) : Prop :=
  s.first = s.second + 8 ∧
  s.third = 2 ∧
  s.second = 3 * s.third + 6

theorem jerry_skit_first_character_lines :
  ∀ s : SkitScript, validScript s → s.first = 20 := by
  sorry

#check jerry_skit_first_character_lines

end jerry_skit_first_character_lines_l2099_209963


namespace rectangular_yard_area_l2099_209961

theorem rectangular_yard_area (w : ℝ) (l : ℝ) : 
  l = 2 * w + 30 →
  2 * w + 2 * l = 700 →
  w * l = 233600 / 9 :=
by
  sorry

end rectangular_yard_area_l2099_209961


namespace decimal_sum_to_fraction_l2099_209907

theorem decimal_sum_to_fraction :
  (0.2 : ℚ) + 0.03 + 0.004 + 0.0006 + 0.00007 = 23467 / 100000 := by
  sorry

end decimal_sum_to_fraction_l2099_209907


namespace right_triangle_area_l2099_209971

/-- The area of a right triangle with one leg of length 3 and hypotenuse of length 5 is 6. -/
theorem right_triangle_area (a b c : ℝ) (h1 : a = 3) (h2 : c = 5) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 6 := by
  sorry

end right_triangle_area_l2099_209971


namespace tank_bucket_ratio_l2099_209933

theorem tank_bucket_ratio : 
  ∀ (tank_capacity bucket_capacity : ℚ),
  tank_capacity > 0 → bucket_capacity > 0 →
  ∃ (water_transferred : ℚ),
  (3/5 * tank_capacity - water_transferred = 2/3 * tank_capacity) ∧
  (1/4 * bucket_capacity + water_transferred = 1/2 * bucket_capacity) →
  tank_capacity / bucket_capacity = 15/4 := by
sorry

end tank_bucket_ratio_l2099_209933


namespace hurdle_distance_l2099_209910

theorem hurdle_distance (total_distance : ℕ) (num_hurdles : ℕ) (start_distance : ℕ) (end_distance : ℕ) 
  (h1 : total_distance = 600)
  (h2 : num_hurdles = 12)
  (h3 : start_distance = 50)
  (h4 : end_distance = 55) :
  ∃ d : ℕ, d = 45 ∧ total_distance = start_distance + (num_hurdles - 1) * d + end_distance :=
by sorry

end hurdle_distance_l2099_209910


namespace vector_a_solution_l2099_209903

theorem vector_a_solution (a b : ℝ × ℝ) : 
  b = (1, 2) → 
  (a.1 * b.1 + a.2 * b.2 = 0) → 
  (a.1^2 + a.2^2 = 20) → 
  (a = (4, -2) ∨ a = (-4, 2)) := by
sorry

end vector_a_solution_l2099_209903


namespace nonzero_digits_count_l2099_209975

def original_fraction : ℚ := 120 / (2^5 * 5^10)

def decimal_result : ℝ := (original_fraction : ℝ) - 0.000001

def count_nonzero_digits (x : ℝ) : ℕ :=
  sorry -- Implementation of counting non-zero digits after decimal point

theorem nonzero_digits_count :
  count_nonzero_digits decimal_result = 3 := by
  sorry

end nonzero_digits_count_l2099_209975


namespace four_X_three_l2099_209918

/-- The operation X defined for any two real numbers -/
def X (a b : ℝ) : ℝ := b + 7*a - a^3 + 2*b

/-- Theorem stating that 4 X 3 = -27 -/
theorem four_X_three : X 4 3 = -27 := by
  sorry

end four_X_three_l2099_209918


namespace triangle_abc_properties_l2099_209913

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_abc_properties (t : Triangle) 
  (h_acute : 0 < t.C ∧ t.C < Real.pi / 2)
  (h_sine_relation : Real.sqrt 15 * t.a * Real.sin t.A = t.b * Real.sin t.B * Real.sin t.C)
  (h_b_twice_a : t.b = 2 * t.a)
  (h_a_c_sum : t.a + t.c = 6) :
  Real.tan t.C = Real.sqrt 15 ∧ 
  (1 / 2) * t.a * t.b * Real.sin t.C = Real.sqrt 15 := by
  sorry

end triangle_abc_properties_l2099_209913


namespace hayden_pants_ironing_time_l2099_209941

/-- Represents the ironing routine of Hayden --/
structure IroningRoutine where
  shirt_time : ℕ  -- Time spent ironing shirt per day (in minutes)
  days_per_week : ℕ  -- Number of days Hayden irons per week
  total_time : ℕ  -- Total time spent ironing over 4 weeks (in minutes)

/-- Calculates the time spent ironing pants per day --/
def pants_ironing_time (routine : IroningRoutine) : ℕ :=
  let total_per_week := routine.total_time / 4
  let shirt_per_week := routine.shirt_time * routine.days_per_week
  let pants_per_week := total_per_week - shirt_per_week
  pants_per_week / routine.days_per_week

/-- Theorem stating that Hayden spends 3 minutes ironing his pants each day --/
theorem hayden_pants_ironing_time :
  pants_ironing_time ⟨5, 5, 160⟩ = 3 := by
  sorry


end hayden_pants_ironing_time_l2099_209941


namespace total_cutlery_after_addition_l2099_209955

/-- Represents the number of each type of cutlery in a drawer -/
structure Cutlery :=
  (forks : ℕ)
  (knives : ℕ)
  (spoons : ℕ)
  (teaspoons : ℕ)

/-- Calculates the total number of cutlery pieces -/
def totalCutlery (c : Cutlery) : ℕ :=
  c.forks + c.knives + c.spoons + c.teaspoons

/-- Represents the initial state of the cutlery drawer -/
def initialCutlery : Cutlery :=
  { forks := 6
  , knives := 6 + 9
  , spoons := 2 * (6 + 9)
  , teaspoons := 6 / 2 }

/-- Represents the final state of the cutlery drawer after adding 2 of each type -/
def finalCutlery : Cutlery :=
  { forks := initialCutlery.forks + 2
  , knives := initialCutlery.knives + 2
  , spoons := initialCutlery.spoons + 2
  , teaspoons := initialCutlery.teaspoons + 2 }

/-- Theorem: The total number of cutlery pieces after adding 2 of each type is 62 -/
theorem total_cutlery_after_addition : totalCutlery finalCutlery = 62 := by
  sorry

end total_cutlery_after_addition_l2099_209955


namespace cake_sale_theorem_l2099_209931

/-- Represents the pricing and sales model for small cakes in a charity sale event -/
structure CakeSaleModel where
  initial_price : ℝ
  initial_sales : ℕ
  price_increase : ℝ
  sales_decrease : ℕ
  max_price : ℝ

/-- Calculates the new price after two equal percentage increases -/
def price_after_two_increases (model : CakeSaleModel) (percent : ℝ) : ℝ :=
  model.initial_price * (1 + percent) ^ 2

/-- Calculates the total sales per hour given a price increase -/
def total_sales (model : CakeSaleModel) (price_increase : ℝ) : ℝ :=
  (model.initial_price + price_increase) * 
  (model.initial_sales - model.sales_decrease * price_increase)

/-- The main theorem stating the correct percentage increase and optimal selling price -/
theorem cake_sale_theorem (model : CakeSaleModel) 
  (h1 : model.initial_price = 6)
  (h2 : model.initial_sales = 30)
  (h3 : model.price_increase = 1)
  (h4 : model.sales_decrease = 2)
  (h5 : model.max_price = 10) :
  ∃ (percent : ℝ) (optimal_price : ℝ),
    price_after_two_increases model percent = 8.64 ∧
    percent = 0.2 ∧
    total_sales model (optimal_price - model.initial_price) = 216 ∧
    optimal_price = 9 ∧
    optimal_price ≤ model.max_price :=
by sorry

end cake_sale_theorem_l2099_209931


namespace absolute_value_sum_zero_l2099_209914

theorem absolute_value_sum_zero (a b : ℝ) :
  |3 + a| + |b - 2| = 0 → a + b = -1 := by
  sorry

end absolute_value_sum_zero_l2099_209914


namespace point_distance_from_x_axis_l2099_209958

/-- A point in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance of a point from the x-axis -/
def distanceFromXAxis (p : Point) : ℝ := |p.y|

theorem point_distance_from_x_axis (a : ℝ) :
  let p : Point := ⟨2, a⟩
  distanceFromXAxis p = 3 → a = 3 ∨ a = -3 := by
  sorry

end point_distance_from_x_axis_l2099_209958


namespace ten_thousandths_place_of_5_32_l2099_209930

theorem ten_thousandths_place_of_5_32 : 
  (5 : ℚ) / 32 * 10000 - ((5 : ℚ) / 32 * 10000).floor = 0.2 := by
  sorry

end ten_thousandths_place_of_5_32_l2099_209930


namespace sum_50_to_75_l2099_209959

/-- Sum of integers from a to b, inclusive -/
def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

/-- Theorem: The sum of all integers from 50 through 75, inclusive, is 1625 -/
theorem sum_50_to_75 : sum_integers 50 75 = 1625 := by
  sorry

end sum_50_to_75_l2099_209959


namespace root_ratio_equality_l2099_209942

theorem root_ratio_equality (a : ℝ) (h_pos : a > 0) : 
  (∃ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ 
   x₁^3 + 1 = a*x₁ ∧ x₂^3 + 1 = a*x₂ ∧
   x₂ / x₁ = 2018 ∧
   (∀ x : ℝ, x^3 + 1 = a*x → x = x₁ ∨ x = x₂ ∨ x ≤ 0)) →
  (∃ y₁ y₂ : ℝ, 0 < y₁ ∧ y₁ < y₂ ∧ 
   y₁^3 + 1 = a*y₁^2 ∧ y₂^3 + 1 = a*y₂^2 ∧
   y₂ / y₁ = 2018 ∧
   (∀ y : ℝ, y^3 + 1 = a*y^2 → y = y₁ ∨ y = y₂ ∨ y ≤ 0)) := by
sorry

end root_ratio_equality_l2099_209942


namespace relationship_abc_l2099_209944

theorem relationship_abc :
  let a : ℝ := Real.sqrt 5
  let b : ℝ := 2
  let c : ℝ := Real.sqrt 3
  a > b ∧ b > c := by sorry

end relationship_abc_l2099_209944


namespace truncated_tetrahedron_volume_squared_l2099_209953

/-- A truncated tetrahedron is a solid with 4 triangular faces and 4 hexagonal faces. --/
structure TruncatedTetrahedron where
  side_length : ℝ
  triangular_faces : Fin 4
  hexagonal_faces : Fin 4

/-- The volume of a truncated tetrahedron. --/
noncomputable def volume (t : TruncatedTetrahedron) : ℝ := sorry

/-- Theorem: The square of the volume of a truncated tetrahedron with side length 1 is 529/72. --/
theorem truncated_tetrahedron_volume_squared :
  ∀ (t : TruncatedTetrahedron), t.side_length = 1 → (volume t)^2 = 529/72 := by sorry

end truncated_tetrahedron_volume_squared_l2099_209953


namespace smallest_three_digit_multiple_of_17_l2099_209992

theorem smallest_three_digit_multiple_of_17 : ∀ n : ℕ, 
  n ≥ 100 ∧ n < 1000 ∧ 17 ∣ n → n ≥ 102 :=
by
  sorry

end smallest_three_digit_multiple_of_17_l2099_209992


namespace problem_solution_l2099_209905

/-- The condition p: x^2 - 5ax + 4a^2 < 0 -/
def p (x a : ℝ) : Prop := x^2 - 5*a*x + 4*a^2 < 0

/-- The condition q: 3 < x ≤ 4 -/
def q (x : ℝ) : Prop := 3 < x ∧ x ≤ 4

theorem problem_solution (a : ℝ) (h : a > 0) :
  (a = 1 → ∀ x, p x a ∧ q x ↔ 3 < x ∧ x < 4) ∧
  (∀ x, q x → p x a) ∧ (∃ x, p x a ∧ ¬q x) ↔ 1 < a ∧ a ≤ 3 :=
sorry

end problem_solution_l2099_209905


namespace chess_players_per_game_l2099_209954

/-- The number of combinations of n items taken k at a time -/
def combinations (n k : ℕ) : ℕ := n.choose k

/-- The total number of games played when each player plays each other once -/
def totalGames (n : ℕ) (k : ℕ) : ℕ := combinations n k

theorem chess_players_per_game (n k : ℕ) (h1 : n = 30) (h2 : totalGames n k = 435) : k = 2 := by
  sorry

end chess_players_per_game_l2099_209954


namespace whitney_bookmarks_l2099_209937

/-- Proves that Whitney bought 2 bookmarks given the conditions of the problem --/
theorem whitney_bookmarks :
  ∀ (initial_amount : ℕ) 
    (poster_cost notebook_cost bookmark_cost : ℕ)
    (posters_bought notebooks_bought : ℕ)
    (amount_left : ℕ),
  initial_amount = 2 * 20 →
  poster_cost = 5 →
  notebook_cost = 4 →
  bookmark_cost = 2 →
  posters_bought = 2 →
  notebooks_bought = 3 →
  amount_left = 14 →
  ∃ (bookmarks_bought : ℕ),
    initial_amount = 
      poster_cost * posters_bought + 
      notebook_cost * notebooks_bought + 
      bookmark_cost * bookmarks_bought + 
      amount_left ∧
    bookmarks_bought = 2 :=
by sorry

end whitney_bookmarks_l2099_209937


namespace complex_magnitude_problem_l2099_209987

theorem complex_magnitude_problem (z : ℂ) (h : Complex.I * Real.sqrt 2 * z = 1 + Complex.I) : Complex.abs z = 1 := by
  sorry

end complex_magnitude_problem_l2099_209987


namespace remaining_wire_length_l2099_209984

-- Define the initial wire length in centimeters
def initial_length_cm : ℝ := 23.3

-- Define the first cut in millimeters
def first_cut_mm : ℝ := 105

-- Define the second cut in centimeters
def second_cut_cm : ℝ := 4.6

-- Define the conversion factor from cm to mm
def cm_to_mm : ℝ := 10

-- Theorem statement
theorem remaining_wire_length :
  (initial_length_cm * cm_to_mm - first_cut_mm - second_cut_cm * cm_to_mm) = 82 := by
  sorry

end remaining_wire_length_l2099_209984


namespace nonagon_diagonals_l2099_209966

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A nonagon is a polygon with 9 sides -/
def is_nonagon (n : ℕ) : Prop := n = 9

theorem nonagon_diagonals :
  ∀ n : ℕ, is_nonagon n → num_diagonals n = 27 := by sorry

end nonagon_diagonals_l2099_209966


namespace number_division_problem_l2099_209949

theorem number_division_problem (x : ℝ) : (x - 5) / 7 = 7 → (x - 2) / 13 = 4 := by
  sorry

end number_division_problem_l2099_209949


namespace triple_overlap_is_six_l2099_209909

/-- Represents a rectangular carpet with width and height -/
structure Carpet where
  width : ℝ
  height : ℝ

/-- Represents the hall and the arrangement of carpets -/
structure CarpetArrangement where
  hallWidth : ℝ
  hallHeight : ℝ
  carpet1 : Carpet
  carpet2 : Carpet
  carpet3 : Carpet

/-- Calculates the area of triple overlap in the carpet arrangement -/
def tripleOverlapArea (arrangement : CarpetArrangement) : ℝ :=
  sorry

/-- Theorem stating that the triple overlap area is 6 square meters -/
theorem triple_overlap_is_six (arrangement : CarpetArrangement) 
  (h1 : arrangement.hallWidth = 10 ∧ arrangement.hallHeight = 10)
  (h2 : arrangement.carpet1 = ⟨6, 8⟩)
  (h3 : arrangement.carpet2 = ⟨6, 6⟩)
  (h4 : arrangement.carpet3 = ⟨5, 7⟩) :
  tripleOverlapArea arrangement = 6 := by
  sorry

end triple_overlap_is_six_l2099_209909


namespace salary_increase_20_percent_l2099_209969

-- Define Sharon's original weekly salary
variable (S : ℝ)

-- Define the condition that a 16% increase results in $406
axiom increase_16_percent : S * 1.16 = 406

-- Define the target salary of $420
def target_salary : ℝ := 420

-- Theorem to prove
theorem salary_increase_20_percent : 
  S * 1.20 = target_salary := by
  sorry

end salary_increase_20_percent_l2099_209969


namespace three_roots_iff_b_in_range_l2099_209990

/-- The function f(x) = 2x³ - 3x² + 1 -/
def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + 1

/-- The statement that f(x) + b = 0 has three distinct real roots iff -1 < b < 0 -/
theorem three_roots_iff_b_in_range (b : ℝ) :
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f x + b = 0 ∧ f y + b = 0 ∧ f z + b = 0) ↔ 
  -1 < b ∧ b < 0 :=
sorry

end three_roots_iff_b_in_range_l2099_209990


namespace first_part_value_l2099_209911

theorem first_part_value (x y : ℝ) (h1 : x + y = 36) (h2 : 8 * x + 3 * y = 203) : x = 19 := by
  sorry

end first_part_value_l2099_209911


namespace chess_game_draw_probability_l2099_209919

theorem chess_game_draw_probability (prob_A_win prob_A_not_lose : ℝ) 
  (h1 : prob_A_win = 0.4)
  (h2 : prob_A_not_lose = 0.9) : 
  prob_A_not_lose - prob_A_win = 0.5 := by
  sorry

end chess_game_draw_probability_l2099_209919


namespace number_difference_l2099_209981

theorem number_difference (a b c d : ℝ) : 
  a = 2 * b ∧ 
  a = 3 * c ∧ 
  (a + b + c + d) / 4 = 110 ∧ 
  d = a + b + c 
  → a - c = 80 := by
sorry

end number_difference_l2099_209981


namespace sphere_surface_area_with_rectangular_solid_l2099_209957

/-- The surface area of a sphere containing a rectangular solid -/
theorem sphere_surface_area_with_rectangular_solid :
  ∀ (a b c : ℝ) (S : ℝ),
    a = 3 →
    b = 4 →
    c = 5 →
    S = 4 * Real.pi * ((a^2 + b^2 + c^2) / 4) →
    S = 50 * Real.pi :=
by
  sorry

#check sphere_surface_area_with_rectangular_solid

end sphere_surface_area_with_rectangular_solid_l2099_209957


namespace problem_solution_l2099_209948

/-- Calculates the number of songs per album given the initial number of albums,
    the number of albums removed, and the total number of songs bought. -/
def songs_per_album (initial_albums : ℕ) (removed_albums : ℕ) (total_songs : ℕ) : ℕ :=
  total_songs / (initial_albums - removed_albums)

/-- Proves that given the specific conditions in the problem,
    the number of songs per album is 7. -/
theorem problem_solution :
  songs_per_album 8 2 42 = 7 := by
  sorry

end problem_solution_l2099_209948


namespace cube_arrangement_exists_l2099_209938

/-- Represents the arrangement of numbers on a cube's edges -/
def CubeArrangement := Fin 12 → Fin 12

/-- Checks if the given arrangement is valid (uses all numbers from 1 to 12 exactly once) -/
def is_valid_arrangement (arr : CubeArrangement) : Prop :=
  (∀ i : Fin 12, ∃ j : Fin 12, arr j = i) ∧ 
  (∀ i j : Fin 12, arr i = arr j → i = j)

/-- Returns the product of numbers on the top face -/
def top_face_product (arr : CubeArrangement) : ℕ :=
  (arr 0 + 1) * (arr 1 + 1) * (arr 2 + 1) * (arr 3 + 1)

/-- Returns the product of numbers on the bottom face -/
def bottom_face_product (arr : CubeArrangement) : ℕ :=
  (arr 4 + 1) * (arr 5 + 1) * (arr 6 + 1) * (arr 7 + 1)

/-- Theorem stating that there exists a valid arrangement with equal products on top and bottom faces -/
theorem cube_arrangement_exists : 
  ∃ (arr : CubeArrangement), 
    is_valid_arrangement arr ∧ 
    top_face_product arr = bottom_face_product arr :=
by sorry

end cube_arrangement_exists_l2099_209938


namespace double_burger_cost_l2099_209980

/-- The cost of a double burger given the total spent, number of burgers, single burger cost, and number of double burgers. -/
theorem double_burger_cost 
  (total_spent : ℚ) 
  (total_burgers : ℕ) 
  (single_burger_cost : ℚ) 
  (double_burger_count : ℕ) 
  (h1 : total_spent = 66.5)
  (h2 : total_burgers = 50)
  (h3 : single_burger_cost = 1)
  (h4 : double_burger_count = 33) :
  (total_spent - single_burger_cost * (total_burgers - double_burger_count)) / double_burger_count = 1.5 := by
  sorry

end double_burger_cost_l2099_209980


namespace sin_330_degrees_l2099_209901

theorem sin_330_degrees : Real.sin (330 * π / 180) = -(1 / 2) := by
  sorry

end sin_330_degrees_l2099_209901


namespace snow_probability_value_l2099_209916

/-- The probability of snow occurring at least once in a week, where the first 4 days 
    have a 1/4 chance of snow each day and the next 3 days have a 1/3 chance of snow each day. -/
def snow_probability : ℚ := 1 - (3/4)^4 * (2/3)^3

/-- Theorem stating that the probability of snow occurring at least once in the described week
    is equal to 125/128. -/
theorem snow_probability_value : snow_probability = 125/128 := by
  sorry

end snow_probability_value_l2099_209916


namespace tetrahedron_volume_in_cube_l2099_209924

/-- Given a cube with side length 6, the volume of the tetrahedron formed by any vertex
    and the three vertices connected to that vertex by edges of the cube is 36. -/
theorem tetrahedron_volume_in_cube (cube_side_length : ℝ) (tetrahedron_volume : ℝ) :
  cube_side_length = 6 →
  tetrahedron_volume = (1 / 3) * (1 / 2 * cube_side_length * cube_side_length) * cube_side_length →
  tetrahedron_volume = 36 := by
  sorry


end tetrahedron_volume_in_cube_l2099_209924


namespace sphere_carved_cube_surface_area_l2099_209956

theorem sphere_carved_cube_surface_area :
  let sphere_diameter : ℝ := Real.sqrt 3
  let cube_side_length : ℝ := 1
  let cube_diagonal : ℝ := cube_side_length * Real.sqrt 3
  cube_diagonal = sphere_diameter →
  (6 : ℝ) * cube_side_length ^ 2 = 6 :=
by sorry

end sphere_carved_cube_surface_area_l2099_209956


namespace alex_fourth_test_score_l2099_209929

theorem alex_fourth_test_score :
  ∀ (s1 s2 s3 s4 s5 : ℕ),
  (85 ≤ s1 ∧ s1 ≤ 95) ∧
  (85 ≤ s2 ∧ s2 ≤ 95) ∧
  (85 ≤ s3 ∧ s3 ≤ 95) ∧
  (85 ≤ s4 ∧ s4 ≤ 95) ∧
  (s5 = 90) ∧
  (s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s1 ≠ s5 ∧
   s2 ≠ s3 ∧ s2 ≠ s4 ∧ s2 ≠ s5 ∧
   s3 ≠ s4 ∧ s3 ≠ s5 ∧
   s4 ≠ s5) ∧
  (∃ (k : ℕ), (s1 + s2) = 2 * k) ∧
  (∃ (k : ℕ), (s1 + s2 + s3) = 3 * k) ∧
  (∃ (k : ℕ), (s1 + s2 + s3 + s4) = 4 * k) ∧
  (∃ (k : ℕ), (s1 + s2 + s3 + s4 + s5) = 5 * k) →
  s4 = 95 :=
by sorry

end alex_fourth_test_score_l2099_209929


namespace arcsin_one_half_l2099_209985

theorem arcsin_one_half : Real.arcsin (1/2) = π/6 := by
  sorry

end arcsin_one_half_l2099_209985


namespace not_p_and_p_or_q_implies_q_l2099_209943

theorem not_p_and_p_or_q_implies_q (p q : Prop) : (¬p ∧ (p ∨ q)) → q := by
  sorry

end not_p_and_p_or_q_implies_q_l2099_209943


namespace gcd_3570_4840_l2099_209968

theorem gcd_3570_4840 : Nat.gcd 3570 4840 = 10 := by
  sorry

end gcd_3570_4840_l2099_209968


namespace towel_folding_theorem_l2099_209999

-- Define the folding rates for each person
def jane_rate : ℚ := 5 / 5
def kyla_rate : ℚ := 9 / 10
def anthony_rate : ℚ := 14 / 20
def david_rate : ℚ := 6 / 15

-- Define the total number of towels folded in one hour
def total_towels : ℕ := 180

-- Theorem statement
theorem towel_folding_theorem :
  (jane_rate + kyla_rate + anthony_rate + david_rate) * 60 = total_towels := by
  sorry

end towel_folding_theorem_l2099_209999


namespace solve_exponential_equation_l2099_209960

theorem solve_exponential_equation : ∃ x : ℝ, (1000 : ℝ)^5 = 40^x ∧ x = 15 := by sorry

end solve_exponential_equation_l2099_209960


namespace digit_move_correctness_l2099_209967

theorem digit_move_correctness : 
  let original_number := 102
  let moved_digit := 2
  let base := 10
  let new_left_term := original_number - moved_digit
  let new_right_term := base ^ moved_digit
  (new_left_term - new_right_term = 1) = True
  := by sorry

end digit_move_correctness_l2099_209967


namespace sum_product_inequality_l2099_209928

theorem sum_product_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x*y + y*z + z*x) * (1/(x+y)^2 + 1/(y+z)^2 + 1/(z+x)^2) ≥ 9/4 := by
  sorry

end sum_product_inequality_l2099_209928


namespace certain_number_equation_l2099_209906

theorem certain_number_equation (x : ℝ) : 28 = (4/5) * x + 8 ↔ x = 25 := by
  sorry

end certain_number_equation_l2099_209906


namespace arrangement_problem_l2099_209927

theorem arrangement_problem (n : ℕ) (h1 : n ≥ 2) : 
  ((n - 1) * (n - 1) = 25) → n = 6 := by
  sorry

end arrangement_problem_l2099_209927


namespace no_three_distinct_squares_sum_to_100_l2099_209940

/-- A function that checks if a natural number is a perfect square --/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

/-- The proposition that there are no three distinct positive perfect squares that sum to 100 --/
theorem no_three_distinct_squares_sum_to_100 : 
  ¬ ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    isPerfectSquare a ∧ isPerfectSquare b ∧ isPerfectSquare c ∧
    a + b + c = 100 :=
sorry

end no_three_distinct_squares_sum_to_100_l2099_209940


namespace krishans_money_l2099_209900

theorem krishans_money (x y : ℝ) (ram gopal krishan : ℝ) : 
  ram = 1503 →
  ram + gopal + krishan = 15000 →
  ram / (7 * x) = gopal / (17 * x) →
  ram / (7 * x) = krishan / (17 * y) →
  ∃ ε > 0, |krishan - 9845| < ε :=
by sorry

end krishans_money_l2099_209900


namespace rectangle_area_equation_l2099_209932

theorem rectangle_area_equation : ∃! x : ℝ, x > 3 ∧ (x - 3) * (3 * x + 4) = 10 * x := by
  sorry

end rectangle_area_equation_l2099_209932


namespace perfect_square_condition_l2099_209917

theorem perfect_square_condition (x y k : ℝ) : 
  (∃ a : ℝ, x^2 + k*x*y + 81*y^2 = a^2) ↔ k = 18 ∨ k = -18 := by
  sorry

end perfect_square_condition_l2099_209917


namespace product_equality_l2099_209921

theorem product_equality (x y : ℝ) :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 := by
  sorry

end product_equality_l2099_209921


namespace centerIsSeven_l2099_209925

-- Define the type for our 3x3 array
def Array3x3 := Fin 3 → Fin 3 → Fin 9

-- Define what it means for two positions to be adjacent
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2.val + 1 = p2.2.val ∨ p2.2.val + 1 = p1.2.val)) ∨
  (p1.2 = p2.2 ∧ (p1.1.val + 1 = p2.1.val ∨ p2.1.val + 1 = p1.1.val))

-- Define the property of consecutive numbers
def consecutive (n m : Fin 9) : Prop :=
  n.val + 1 = m.val ∨ m.val + 1 = n.val

-- Define the property that consecutive numbers are in adjacent squares
def consecutiveAdjacent (arr : Array3x3) : Prop :=
  ∀ i j k l, consecutive (arr i j) (arr k l) → adjacent (i, j) (k, l)

-- Define the property that corner numbers sum to 20
def cornerSum20 (arr : Array3x3) : Prop :=
  (arr 0 0).val + (arr 0 2).val + (arr 2 0).val + (arr 2 2).val = 20

-- Define the property that all numbers from 1 to 9 are used
def allNumbersUsed (arr : Array3x3) : Prop :=
  ∀ n : Fin 9, ∃ i j, arr i j = n

-- The main theorem
theorem centerIsSeven (arr : Array3x3) 
  (h1 : consecutiveAdjacent arr) 
  (h2 : cornerSum20 arr) 
  (h3 : allNumbersUsed arr) : 
  arr 1 1 = 7 := by
  sorry

end centerIsSeven_l2099_209925


namespace expression_evaluation_l2099_209976

theorem expression_evaluation :
  let a : ℚ := -3/2
  (a - 2) * (a + 2) - (a + 2)^2 = -2 := by
  sorry

end expression_evaluation_l2099_209976


namespace max_first_term_l2099_209996

/-- A sequence of positive real numbers satisfying the given conditions -/
def SpecialSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, 0 < a n) ∧ 
  (∀ n, n > 0 → (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0) ∧
  (a 1 = a 10)

/-- The theorem stating the maximum possible value of the first term -/
theorem max_first_term (a : ℕ → ℝ) (h : SpecialSequence a) : 
  a 1 ≤ 16 := by
  sorry

end max_first_term_l2099_209996


namespace aluminum_weight_in_compound_l2099_209982

/-- The molecular weight of the aluminum part in Al2(CO3)3 -/
def aluminum_weight : ℝ := 2 * 26.98

/-- Proof that the molecular weight of the aluminum part in Al2(CO3)3 is 53.96 g/mol -/
theorem aluminum_weight_in_compound : aluminum_weight = 53.96 := by
  sorry

end aluminum_weight_in_compound_l2099_209982


namespace set_operation_proof_l2099_209935

def U : Finset Int := {-2, -1, 0, 1, 2}
def A : Finset Int := {1, 2}
def B : Finset Int := {-2, 1, 2}

theorem set_operation_proof :
  A ∪ (U \ B) = {-1, 0, 1, 2} := by sorry

end set_operation_proof_l2099_209935


namespace mean_of_added_numbers_l2099_209912

theorem mean_of_added_numbers (original_mean original_count new_mean new_count : ℝ) 
  (h1 : original_mean = 65)
  (h2 : original_count = 7)
  (h3 : new_mean = 80)
  (h4 : new_count = 10) :
  let added_count := new_count - original_count
  let added_sum := new_mean * new_count - original_mean * original_count
  added_sum / added_count = 115 := by
sorry

end mean_of_added_numbers_l2099_209912


namespace specific_combination_probability_is_one_eighth_l2099_209947

/-- A regular tetrahedron with numbers on its faces -/
structure NumberedTetrahedron :=
  (faces : Fin 4 → Fin 4)

/-- The probability of a specific face showing on a regular tetrahedron -/
def face_probability : ℚ := 1 / 4

/-- The number of ways to choose which tetrahedron shows a specific number -/
def ways_to_choose : ℕ := 2

/-- The probability of getting a specific combination of numbers when throwing two tetrahedra -/
def specific_combination_probability (t1 t2 : NumberedTetrahedron) : ℚ :=
  ↑ways_to_choose * face_probability * face_probability

theorem specific_combination_probability_is_one_eighth (t1 t2 : NumberedTetrahedron) :
  specific_combination_probability t1 t2 = 1 / 8 := by
  sorry

end specific_combination_probability_is_one_eighth_l2099_209947


namespace speeding_fine_lawyer_hours_mark_speeding_fine_l2099_209998

theorem speeding_fine_lawyer_hours 
  (base_fine : ℕ) 
  (fine_increase_per_mph : ℕ) 
  (actual_speed : ℕ) 
  (speed_limit : ℕ) 
  (court_costs : ℕ) 
  (lawyer_hourly_rate : ℕ) 
  (total_owed : ℕ) : ℕ :=
  let speed_over_limit := actual_speed - speed_limit
  let speed_penalty := speed_over_limit * fine_increase_per_mph
  let initial_fine := base_fine + speed_penalty
  let doubled_fine := initial_fine * 2
  let fine_with_court_costs := doubled_fine + court_costs
  let lawyer_fees := total_owed - fine_with_court_costs
  lawyer_fees / lawyer_hourly_rate

theorem mark_speeding_fine 
  (h1 : speeding_fine_lawyer_hours 50 2 75 30 300 80 820 = 3) : 
  speeding_fine_lawyer_hours 50 2 75 30 300 80 820 = 3 := by
  sorry

end speeding_fine_lawyer_hours_mark_speeding_fine_l2099_209998


namespace gcd_of_B_is_two_l2099_209983

def B : Set ℕ := {n | ∃ x : ℕ, x > 0 ∧ n = 4*x + 2}

theorem gcd_of_B_is_two : 
  ∃ (d : ℕ), d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ 
  (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end gcd_of_B_is_two_l2099_209983


namespace absolute_value_plus_exponent_l2099_209945

theorem absolute_value_plus_exponent : |-8| + 3^0 = 9 := by
  sorry

end absolute_value_plus_exponent_l2099_209945


namespace regular_polygon_interior_angle_sum_l2099_209977

/-- For a regular polygon with exterior angles of 45 degrees, the sum of interior angles is 1080 degrees. -/
theorem regular_polygon_interior_angle_sum : 
  ∀ (n : ℕ), n > 2 → (360 / n = 45) → (n - 2) * 180 = 1080 :=
by sorry

end regular_polygon_interior_angle_sum_l2099_209977


namespace total_fleas_is_40_l2099_209970

/-- The number of fleas on Gertrude the chicken -/
def gertrudeFleas : ℕ := 10

/-- The number of fleas on Olive the chicken -/
def oliveFleas : ℕ := gertrudeFleas / 2

/-- The number of fleas on Maud the chicken -/
def maudFleas : ℕ := 5 * oliveFleas

/-- The total number of fleas on all three chickens -/
def totalFleas : ℕ := gertrudeFleas + oliveFleas + maudFleas

theorem total_fleas_is_40 : totalFleas = 40 := by
  sorry

end total_fleas_is_40_l2099_209970


namespace sum_of_qp_is_zero_l2099_209936

def p (x : ℝ) : ℝ := x^2 - 4

def q (x : ℝ) : ℝ := -abs x

def evaluation_points : List ℝ := [-3, -2, -1, 0, 1, 2, 3]

theorem sum_of_qp_is_zero :
  (evaluation_points.map (λ x => q (p x))).sum = 0 := by
  sorry

end sum_of_qp_is_zero_l2099_209936


namespace all_non_negative_l2099_209952

theorem all_non_negative (a b c d : ℤ) (h : (2 : ℝ)^a + (2 : ℝ)^b = (3 : ℝ)^c + (3 : ℝ)^d) :
  a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 := by
  sorry

end all_non_negative_l2099_209952


namespace canoe_capacity_fraction_l2099_209922

/-- The fraction of people that can fit in a canoe with a dog compared to without --/
theorem canoe_capacity_fraction :
  let max_people : ℕ := 6  -- Maximum number of people without dog
  let person_weight : ℕ := 140  -- Weight of each person in pounds
  let dog_weight : ℕ := person_weight / 4  -- Weight of the dog
  let total_weight : ℕ := 595  -- Total weight with dog and people
  let people_with_dog : ℕ := (total_weight - dog_weight) / person_weight  -- Number of people with dog
  (people_with_dog : ℚ) / max_people = 2 / 3 :=
by sorry

end canoe_capacity_fraction_l2099_209922


namespace a_2018_mod_49_l2099_209988

def a (n : ℕ) : ℕ := 6^n + 8^n

theorem a_2018_mod_49 : a 2018 % 49 = 0 := by
  sorry

end a_2018_mod_49_l2099_209988


namespace calculation_difference_l2099_209994

theorem calculation_difference : 
  let correct_calculation := 10 - (3 * 4)
  let incorrect_calculation := 10 - 3 + 4
  correct_calculation - incorrect_calculation = -13 := by
sorry

end calculation_difference_l2099_209994


namespace pierre_cake_consumption_l2099_209902

theorem pierre_cake_consumption (cake_weight : ℝ) (num_parts : ℕ) 
  (nathalie_parts : ℝ) (pierre_multiplier : ℝ) : 
  cake_weight = 400 → 
  num_parts = 8 → 
  nathalie_parts = 1 / 8 → 
  pierre_multiplier = 2 → 
  pierre_multiplier * (nathalie_parts * cake_weight) = 100 := by
  sorry

end pierre_cake_consumption_l2099_209902


namespace subset_condition_implies_p_range_l2099_209989

open Set

theorem subset_condition_implies_p_range (p : ℝ) : 
  let A : Set ℝ := {x | 4 * x + p < 0}
  let B : Set ℝ := {x | x < -1 ∨ x > 2}
  A.Nonempty → B.Nonempty → A ⊆ B → p ≥ 4 :=
by sorry

end subset_condition_implies_p_range_l2099_209989


namespace inequality_proof_l2099_209964

theorem inequality_proof (x y z : ℝ) : 
  (x^2 + 2*y^2 + 2*z^2)/(x^2 + y*z) + (y^2 + 2*z^2 + 2*x^2)/(y^2 + z*x) + (z^2 + 2*x^2 + 2*y^2)/(z^2 + x*y) > 6 :=
by sorry

end inequality_proof_l2099_209964


namespace equation_solution_l2099_209991

theorem equation_solution : 
  {x : ℝ | (x + 2)^4 + (x - 4)^4 = 272} = {0, 2} := by sorry

end equation_solution_l2099_209991


namespace min_value_abc_l2099_209950

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a^2 + 2*a*b + 2*a*c + 4*b*c = 12) : 
  ∀ x y z, x > 0 → y > 0 → z > 0 → x^2 + 2*x*y + 2*x*z + 4*y*z = 12 → 
  a + b + c ≤ x + y + z ∧ a + b + c ≥ 2 * Real.sqrt 3 := by
  sorry

end min_value_abc_l2099_209950


namespace oplus_calculation_l2099_209908

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := a * b + a + b + 1

-- State the theorem
theorem oplus_calculation : oplus (-3) (oplus 4 2) = -32 := by
  sorry

end oplus_calculation_l2099_209908


namespace linear_function_properties_l2099_209962

def f (x : ℝ) : ℝ := x + 2

theorem linear_function_properties :
  (f 1 = 3) ∧
  (f (-2) = 0) ∧
  (∀ x y, f x = y → x ≥ 0 ∧ y ≤ 0 → x = 0 ∧ y = 2) ∧
  (∃ x, x > 2 ∧ f x ≥ 4) := by
  sorry

end linear_function_properties_l2099_209962


namespace average_age_combined_l2099_209923

theorem average_age_combined (n_students : ℕ) (n_parents : ℕ) 
  (avg_age_students : ℚ) (avg_age_parents : ℚ) :
  n_students = 40 →
  n_parents = 50 →
  avg_age_students = 13 →
  avg_age_parents = 40 →
  (n_students * avg_age_students + n_parents * avg_age_parents) / (n_students + n_parents : ℚ) = 28 := by
  sorry

end average_age_combined_l2099_209923


namespace mans_speed_in_still_water_l2099_209915

/-- The speed of a man rowing a boat in still water, given the speed of the stream
    and the time taken to row a certain distance downstream. -/
theorem mans_speed_in_still_water
  (stream_speed : ℝ)
  (downstream_distance : ℝ)
  (downstream_time : ℝ)
  (h1 : stream_speed = 8)
  (h2 : downstream_distance = 90)
  (h3 : downstream_time = 5)
  : ∃ (mans_speed : ℝ), mans_speed = 10 ∧ 
    (mans_speed + stream_speed) * downstream_time = downstream_distance :=
by sorry

end mans_speed_in_still_water_l2099_209915


namespace horner_v2_equals_16_l2099_209965

/-- Horner's method for evaluating a polynomial -/
def horner_v2 (x : ℝ) : ℝ :=
  let v0 : ℝ := x
  let v1 : ℝ := 3 * v0 + 1
  v1 * v0 + 2

/-- The polynomial f(x) = 3x^4 + x^3 + 2x^2 + x + 4 -/
def f (x : ℝ) : ℝ := 3*x^4 + x^3 + 2*x^2 + x + 4

theorem horner_v2_equals_16 : horner_v2 2 = 16 := by
  sorry


end horner_v2_equals_16_l2099_209965


namespace dam_building_time_with_reduced_workers_l2099_209973

/-- The time taken to build a dam given the number of workers and their work rate -/
def build_time (workers : ℕ) (rate : ℚ) : ℚ :=
  1 / (workers * rate)

/-- The work rate of a single worker -/
def worker_rate (initial_workers : ℕ) (initial_time : ℚ) : ℚ :=
  1 / (initial_workers * initial_time)

theorem dam_building_time_with_reduced_workers 
  (initial_workers : ℕ) 
  (initial_time : ℚ) 
  (new_workers : ℕ) : 
  initial_workers = 60 → 
  initial_time = 5 → 
  new_workers = 40 → 
  build_time new_workers (worker_rate initial_workers initial_time) = 7.5 := by
sorry

end dam_building_time_with_reduced_workers_l2099_209973


namespace sheilas_extra_flour_l2099_209979

/-- Given that Katie needs 3 pounds of flour and the total flour needed is 8 pounds,
    prove that Sheila needs 2 pounds more flour than Katie. -/
theorem sheilas_extra_flour (katie_flour sheila_flour total_flour : ℕ) : 
  katie_flour = 3 → 
  total_flour = 8 → 
  sheila_flour = total_flour - katie_flour →
  sheila_flour - katie_flour = 2 := by
  sorry

end sheilas_extra_flour_l2099_209979


namespace first_square_length_is_correct_l2099_209978

/-- The length of the first square of fabric -/
def first_square_length : ℝ := 8

/-- The height of the first square of fabric -/
def first_square_height : ℝ := 5

/-- The length of the second square of fabric -/
def second_square_length : ℝ := 10

/-- The height of the second square of fabric -/
def second_square_height : ℝ := 7

/-- The length of the third square of fabric -/
def third_square_length : ℝ := 5

/-- The height of the third square of fabric -/
def third_square_height : ℝ := 5

/-- The desired length of the flag -/
def flag_length : ℝ := 15

/-- The desired height of the flag -/
def flag_height : ℝ := 9

theorem first_square_length_is_correct : 
  first_square_length * first_square_height + 
  second_square_length * second_square_height + 
  third_square_length * third_square_height = 
  flag_length * flag_height := by
  sorry

end first_square_length_is_correct_l2099_209978


namespace imaginary_sum_zero_l2099_209920

theorem imaginary_sum_zero (i : ℂ) (h : i^2 = -1) :
  i^15732 + i^15733 + i^15734 + i^15735 = 0 := by sorry

end imaginary_sum_zero_l2099_209920


namespace square_of_binomial_l2099_209993

/-- If ax^2 + 18x + 16 is the square of a binomial, then a = 81/16 -/
theorem square_of_binomial (a : ℚ) : 
  (∃ r s : ℚ, ∀ x : ℚ, a * x^2 + 18 * x + 16 = (r * x + s)^2) → 
  a = 81 / 16 := by
sorry

end square_of_binomial_l2099_209993


namespace library_visitors_on_sunday_l2099_209951

/-- Proves that the average number of visitors on Sundays is 660 given the specified conditions --/
theorem library_visitors_on_sunday (total_days : Nat) (non_sunday_avg : Nat) (overall_avg : Nat) : 
  total_days = 30 →
  non_sunday_avg = 240 →
  overall_avg = 310 →
  (5 * (total_days * overall_avg - 25 * non_sunday_avg)) / 5 = 660 := by
  sorry

#check library_visitors_on_sunday

end library_visitors_on_sunday_l2099_209951


namespace min_value_x_plus_y_l2099_209995

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : (x + 1)⁻¹ + y⁻¹ = (1 : ℝ) / 2) :
  ∀ a b : ℝ, a > 0 → b > 0 → (a + 1)⁻¹ + b⁻¹ = (1 : ℝ) / 2 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x + 1)⁻¹ + y⁻¹ = (1 : ℝ) / 2 ∧ x + y = 7 := by
  sorry

end min_value_x_plus_y_l2099_209995
