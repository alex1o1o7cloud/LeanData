import Mathlib

namespace tiles_on_floor_l369_36914

/-- Represents a rectangular floor covered with square tiles. -/
structure TiledFloor where
  width : ℕ
  length : ℕ
  diagonal_tiles : ℕ

/-- Calculates the total number of tiles on the floor. -/
def total_tiles (floor : TiledFloor) : ℕ :=
  floor.width * floor.length

/-- Theorem: For a rectangular floor with length twice the width and 25 tiles along the diagonal,
    the total number of tiles is 242. -/
theorem tiles_on_floor (floor : TiledFloor) 
    (h1 : floor.length = 2 * floor.width)
    (h2 : floor.diagonal_tiles = 25) :
    total_tiles floor = 242 := by
  sorry

#eval total_tiles { width := 11, length := 22, diagonal_tiles := 25 }

end tiles_on_floor_l369_36914


namespace problem_solution_l369_36920

theorem problem_solution (d : ℝ) (a b c : ℤ) (h1 : d ≠ 0) 
  (h2 : (18 * d + 19 + 20 * d^2) + (4 * d + 3 - 2 * d^2) = a * d + b + c * d^2) : 
  a + b + c = 62 := by
  sorry

end problem_solution_l369_36920


namespace car_sales_prediction_l369_36984

theorem car_sales_prediction (sports_cars : ℕ) (sedans : ℕ) (other_cars : ℕ) : 
  sports_cars = 35 →
  5 * sedans = 8 * sports_cars →
  sedans = 2 * other_cars →
  other_cars = 28 := by
sorry

end car_sales_prediction_l369_36984


namespace subtraction_of_decimals_l369_36910

theorem subtraction_of_decimals : 5.18 - 3.45 = 1.73 := by
  sorry

end subtraction_of_decimals_l369_36910


namespace rectangular_plot_breadth_l369_36980

/-- The breadth of a rectangular plot with specific conditions -/
theorem rectangular_plot_breadth :
  ∀ (b l : ℝ),
  (l * b + (1/2 * (b/2) * (l/3)) = 24 * b) →  -- Area condition
  (l - b = 10) →                              -- Length-breadth difference
  (b = 158/13) :=                             -- Breadth of the plot
by
  sorry

end rectangular_plot_breadth_l369_36980


namespace b_fourth_zero_implies_b_squared_zero_l369_36982

theorem b_fourth_zero_implies_b_squared_zero (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : B ^ 2 = 0 := by
  sorry

end b_fourth_zero_implies_b_squared_zero_l369_36982


namespace sum_to_n_432_l369_36959

theorem sum_to_n_432 : ∃ n : ℕ, (∀ m : ℕ, m > n → (m * (m + 1)) / 2 > 432) ∧ (n * (n + 1)) / 2 ≤ 432 := by
  sorry

end sum_to_n_432_l369_36959


namespace cafe_outdoor_tables_l369_36948

/-- The number of indoor tables -/
def indoor_tables : ℕ := 9

/-- The number of chairs per indoor table -/
def chairs_per_indoor_table : ℕ := 10

/-- The number of chairs per outdoor table -/
def chairs_per_outdoor_table : ℕ := 3

/-- The total number of chairs -/
def total_chairs : ℕ := 123

/-- The number of outdoor tables -/
def outdoor_tables : ℕ := (total_chairs - indoor_tables * chairs_per_indoor_table) / chairs_per_outdoor_table

theorem cafe_outdoor_tables : outdoor_tables = 11 := by
  sorry

end cafe_outdoor_tables_l369_36948


namespace gracies_height_l369_36945

/-- Proves that Gracie's height is 56 inches given the relationships between Gracie, Grayson, and Griffin's heights. -/
theorem gracies_height (griffin_height grayson_height gracie_height : ℕ) : 
  griffin_height = 61 →
  grayson_height = griffin_height + 2 →
  gracie_height = grayson_height - 7 →
  gracie_height = 56 :=
by sorry

end gracies_height_l369_36945


namespace one_true_related_proposition_l369_36901

theorem one_true_related_proposition :
  let P : ℝ → Prop := λ b => b = 3
  let Q : ℝ → Prop := λ b => b^2 = 9
  let converse := ∀ b, Q b → P b
  let negation := ∀ b, ¬(P b) → ¬(Q b)
  let inverse := ∀ b, ¬(Q b) → ¬(P b)
  (converse ∨ negation ∨ inverse) ∧ ¬(converse ∧ negation) ∧ ¬(converse ∧ inverse) ∧ ¬(negation ∧ inverse) :=
by
  sorry

#check one_true_related_proposition

end one_true_related_proposition_l369_36901


namespace function_properties_l369_36974

def f (abc : ℝ) (x : ℝ) : ℝ := x^3 - 6*x^2 + 9*x - abc

theorem function_properties (a b c abc : ℝ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : f abc a = 0) (h4 : f abc b = 0) (h5 : f abc c = 0) :
  (f abc 0) * (f abc 1) < 0 ∧ (f abc 0) * (f abc 3) > 0 := by
  sorry

end function_properties_l369_36974


namespace quadratic_function_properties_l369_36936

def f (x : ℝ) : ℝ := -2.5 * x^2 + 15 * x - 12.5

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 10 := by sorry

end quadratic_function_properties_l369_36936


namespace complex_power_500_l369_36911

theorem complex_power_500 : ((1 + 2 * Complex.I) / (1 - 2 * Complex.I)) ^ 500 = 1 := by
  sorry

end complex_power_500_l369_36911


namespace dye_arrangement_count_l369_36958

/-- The number of ways to arrange 3 organic dyes, 2 inorganic dyes, and 2 additives -/
def total_arrangements : ℕ := sorry

/-- The condition that no two organic dyes are adjacent -/
def organic_not_adjacent (arrangement : List (Fin 7)) : Prop := sorry

/-- The number of valid arrangements where no two organic dyes are adjacent -/
def valid_arrangements : ℕ := sorry

theorem dye_arrangement_count :
  valid_arrangements = 1440 := by sorry

end dye_arrangement_count_l369_36958


namespace bread_flour_calculation_l369_36981

theorem bread_flour_calculation (x : ℝ) : 
  x > 0 ∧ 
  x + 10 > 0 ∧ 
  x * (1 + x / 100) + (x + 10) * (1 + (x + 10) / 100) = 112.5 → 
  x = 35 :=
by sorry

end bread_flour_calculation_l369_36981


namespace tan_product_values_l369_36975

theorem tan_product_values (a b : ℝ) :
  7 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 2) = 0 →
  Real.tan (a / 2) * Real.tan (b / 2) = 4 ∨ Real.tan (a / 2) * Real.tan (b / 2) = -4 := by
  sorry

end tan_product_values_l369_36975


namespace area_MNKP_l369_36999

/-- The area of quadrilateral MNKP given the area of quadrilateral ABCD -/
theorem area_MNKP (S_ABCD : ℝ) (h1 : S_ABCD = (180 + 50 * Real.sqrt 3) / 6)
  (h2 : ∃ S_MNKP : ℝ, S_MNKP = S_ABCD / 2) :
  ∃ S_MNKP : ℝ, S_MNKP = (90 + 25 * Real.sqrt 3) / 6 := by
  sorry

end area_MNKP_l369_36999


namespace concert_cost_theorem_l369_36926

/-- Calculates the total cost for two people to attend a concert -/
def concert_cost (ticket_price : ℝ) (processing_fee_rate : ℝ) (parking_fee : ℝ) (entrance_fee : ℝ) : ℝ :=
  let total_ticket_cost := 2 * ticket_price
  let processing_fee := total_ticket_cost * processing_fee_rate
  let total_entrance_fee := 2 * entrance_fee
  total_ticket_cost + processing_fee + parking_fee + total_entrance_fee

/-- Theorem stating that the total cost for two people to attend the concert is $135.00 -/
theorem concert_cost_theorem : 
  concert_cost 50 0.15 10 5 = 135 := by
  sorry

end concert_cost_theorem_l369_36926


namespace cake_mix_distribution_l369_36956

theorem cake_mix_distribution (tray1 tray2 : ℕ) : 
  tray2 = tray1 - 20 → 
  tray1 + tray2 = 500 → 
  tray1 = 260 := by
sorry

end cake_mix_distribution_l369_36956


namespace cubic_square_fraction_inequality_l369_36931

theorem cubic_square_fraction_inequality {s r : ℝ} (hs : 0 < s) (hr : 0 < r) (hsr : r < s) :
  (s^3 - r^3) / (s^3 + r^3) > (s^2 - r^2) / (s^2 + r^2) := by
  sorry

end cubic_square_fraction_inequality_l369_36931


namespace min_radius_circle_line_intersection_l369_36913

theorem min_radius_circle_line_intersection (θ : Real) (h1 : θ ∈ Set.Icc 0 (2 * Real.pi)) :
  let circle := fun (x y : Real) => (x - Real.cos θ)^2 + (y - Real.sin θ)^2
  let line := fun (x y : Real) => 2 * x - y - 10
  ∃ (r : Real), r > 0 ∧ ∃ (x y : Real), circle x y = r^2 ∧ line x y = 0 →
  ∀ (r' : Real), (∃ (x y : Real), circle x y = r'^2 ∧ line x y = 0) → r' ≥ 2 * Real.sqrt 5 - 1 :=
sorry

end min_radius_circle_line_intersection_l369_36913


namespace ad_transmission_cost_l369_36902

/-- The cost of transmitting advertisements during a race -/
theorem ad_transmission_cost
  (num_ads : ℕ)
  (ad_duration : ℕ)
  (cost_per_minute : ℕ)
  (h1 : num_ads = 5)
  (h2 : ad_duration = 3)
  (h3 : cost_per_minute = 4000) :
  num_ads * ad_duration * cost_per_minute = 60000 :=
by sorry

end ad_transmission_cost_l369_36902


namespace circles_internally_tangent_l369_36940

-- Define the circles
def circle_C1 (m : ℝ) (x y : ℝ) : Prop := (x - m)^2 + (y + 2)^2 = 9
def circle_C2 (m : ℝ) (x y : ℝ) : Prop := (x + 1)^2 + (y - m)^2 = 4

-- Define the condition for internal tangency
def internally_tangent (m : ℝ) : Prop :=
  ∃ x y : ℝ, circle_C1 m x y ∧ circle_C2 m x y

-- Theorem statement
theorem circles_internally_tangent :
  ∀ m : ℝ, internally_tangent m ↔ m = -2 ∨ m = -1 :=
by sorry

end circles_internally_tangent_l369_36940


namespace part1_part2_l369_36928

/-- The function f(x) = x³ - x² --/
def f (x : ℝ) : ℝ := x^3 - x^2

/-- Part 1: At least one of f(m) and f(n) is not less than zero --/
theorem part1 (m n : ℝ) (hm : m > 0) (hn : n > 0) (hmn : m * n > 1) :
  max (f m) (f n) ≥ 0 := by sorry

/-- Part 2: a + b < 4/3 --/
theorem part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (heq : f a = f b) :
  a + b < 4/3 := by sorry

end part1_part2_l369_36928


namespace fraction_unchanged_l369_36987

theorem fraction_unchanged (a b : ℝ) : (2 * (7 * a)) / ((7 * a) + (7 * b)) = (2 * a) / (a + b) := by
  sorry

end fraction_unchanged_l369_36987


namespace part_one_part_two_l369_36919

-- Define the sets A and B
def A : Set ℝ := {x | -2 + 3*x - x^2 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a < x ∧ x < a + 2}

-- Part 1: Prove that when a = 1, (∁A) ∩ B = (1, 2)
theorem part_one : (Set.compl A) ∩ (B 1) = Set.Ioo 1 2 := by sorry

-- Part 2: Prove that (∁A) ∩ B = ∅ if and only if a ≤ -1 or a ≥ 2
theorem part_two (a : ℝ) : (Set.compl A) ∩ (B a) = ∅ ↔ a ≤ -1 ∨ a ≥ 2 := by sorry

end part_one_part_two_l369_36919


namespace tetrahedron_volume_is_16_l369_36942

/-- Represents a tetrahedron PQRS with given side lengths and base area -/
structure Tetrahedron where
  PQ : ℝ
  PR : ℝ
  PS : ℝ
  QR : ℝ
  QS : ℝ
  RS : ℝ
  area_PQR : ℝ

/-- Calculates the volume of a tetrahedron -/
def tetrahedron_volume (t : Tetrahedron) : ℝ :=
  sorry

/-- Theorem stating that the volume of the given tetrahedron is 16 -/
theorem tetrahedron_volume_is_16 (t : Tetrahedron) 
  (h1 : t.PQ = 6)
  (h2 : t.PR = 4)
  (h3 : t.PS = 5)
  (h4 : t.QR = 5)
  (h5 : t.QS = 6)
  (h6 : t.RS = 15/2)
  (h7 : t.area_PQR = 12) :
  tetrahedron_volume t = 16 :=
sorry

end tetrahedron_volume_is_16_l369_36942


namespace machine_value_calculation_l369_36917

theorem machine_value_calculation (initial_value : ℝ) : 
  initial_value * (0.75 ^ 2) = 4000 → initial_value = 7111.11111111111 := by
  sorry

end machine_value_calculation_l369_36917


namespace no_snow_probability_l369_36951

/-- The probability of no snow for five consecutive days, given the probability of snow each day is 2/3 -/
theorem no_snow_probability (p : ℚ) (h : p = 2/3) : (1 - p)^5 = 1/243 := by
  sorry

end no_snow_probability_l369_36951


namespace cone_cylinder_volume_ratio_l369_36990

/-- The ratio of the total volume of two cones to the volume of a cylinder -/
theorem cone_cylinder_volume_ratio :
  let r : ℝ := 4 -- radius of cylinder and cones
  let h_cyl : ℝ := 18 -- height of cylinder
  let h_cone1 : ℝ := 6 -- height of first cone
  let h_cone2 : ℝ := 9 -- height of second cone
  let v_cyl := π * r^2 * h_cyl -- volume of cylinder
  let v_cone1 := (1/3) * π * r^2 * h_cone1 -- volume of first cone
  let v_cone2 := (1/3) * π * r^2 * h_cone2 -- volume of second cone
  let v_cones := v_cone1 + v_cone2 -- total volume of cones
  v_cones / v_cyl = 5 / 18 := by
sorry


end cone_cylinder_volume_ratio_l369_36990


namespace quadratic_minimum_l369_36935

/-- Given a quadratic function f(x) = x^2 + 2px + r, 
    if the minimum value of f(x) is 1, then r = p^2 + 1 -/
theorem quadratic_minimum (p r : ℝ) : 
  (∃ (f : ℝ → ℝ), (∀ x, f x = x^2 + 2*p*x + r) ∧ 
   (∃ (m : ℝ), ∀ x, f x ≥ m ∧ (∃ y, f y = m)) ∧
   (∃ x, f x = 1)) →
  r = p^2 + 1 := by
  sorry

end quadratic_minimum_l369_36935


namespace complex_product_theorem_l369_36941

theorem complex_product_theorem (y : ℂ) (h : y = Complex.exp (4 * Real.pi * Complex.I / 9)) :
  (3 * y^2 + y^4) * (3 * y^4 + y^8) * (3 * y^6 + y^12) * 
  (3 * y^8 + y^16) * (3 * y^10 + y^20) * (3 * y^12 + y^24) = -8 := by
  sorry

end complex_product_theorem_l369_36941


namespace even_number_induction_step_l369_36996

theorem even_number_induction_step (P : ℕ → Prop) (k : ℕ) 
  (h_even : Even k) (h_ge_2 : k ≥ 2) (h_base : P 2) (h_k : P k) :
  (∀ n, Even n → n ≥ 2 → P n) ↔ 
  (P k → P (k + 2)) :=
sorry

end even_number_induction_step_l369_36996


namespace right_triangle_complex_roots_l369_36973

theorem right_triangle_complex_roots : 
  ∃! (s : Finset ℂ), 
    (∀ z ∈ s, z ≠ 0 ∧ (z.re * (z^3).re + z.im * (z^3).im = 0)) ∧ 
    s.card = 2 :=
sorry

end right_triangle_complex_roots_l369_36973


namespace sqrt_not_arithmetic_if_geometric_not_arithmetic_l369_36991

theorem sqrt_not_arithmetic_if_geometric_not_arithmetic
  (a b c : ℝ)
  (positive_a : 0 < a)
  (positive_b : 0 < b)
  (positive_c : 0 < c)
  (geometric_sequence : b^2 = a * c)
  (not_arithmetic_sequence : ¬(a + c = 2 * b)) :
  ¬(Real.sqrt a + Real.sqrt c = 2 * Real.sqrt b) :=
by sorry

end sqrt_not_arithmetic_if_geometric_not_arithmetic_l369_36991


namespace geometric_sequence_ratio_l369_36957

/-- Given a geometric sequence {a_n} where (a_5 - a_1) / (a_3 - a_1) = 3,
    prove that (a_10 - a_2) / (a_6 + a_2) = 3 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (h : (a 5 - a 1) / (a 3 - a 1) = 3)
  (h_geom : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1) :
  (a 10 - a 2) / (a 6 + a 2) = 3 := by
sorry

end geometric_sequence_ratio_l369_36957


namespace cost_price_calculation_l369_36967

theorem cost_price_calculation (selling_price : ℝ) (profit_percentage : ℝ) 
  (hp : selling_price = 1170)
  (hq : profit_percentage = 20) : 
  ∃ cost_price : ℝ, 
    cost_price * (1 + profit_percentage / 100) = selling_price ∧ 
    cost_price = 975 := by
  sorry

end cost_price_calculation_l369_36967


namespace hyperbola_focal_length_specific_hyperbola_focal_length_l369_36909

/-- The focal length of a hyperbola with equation x²/a² - y²/b² = 1 is 2√(a² + b²) -/
theorem hyperbola_focal_length (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let focal_length := 2 * Real.sqrt (a^2 + b^2)
  focal_length = 2 * Real.sqrt 41 :=
by
  sorry

/-- The focal length of the hyperbola x²/16 - y²/25 = 1 is 2√41 -/
theorem specific_hyperbola_focal_length :
  let focal_length := 2 * Real.sqrt (16 + 25)
  focal_length = 2 * Real.sqrt 41 :=
by
  sorry

end hyperbola_focal_length_specific_hyperbola_focal_length_l369_36909


namespace remaining_trip_time_l369_36933

/-- Proves the remaining time of a trip given specific conditions -/
theorem remaining_trip_time 
  (total_time : ℝ) 
  (original_speed : ℝ) 
  (first_part_time : ℝ) 
  (first_part_speed : ℝ) 
  (remaining_speed : ℝ) 
  (h1 : total_time = 7.25)
  (h2 : original_speed = 50)
  (h3 : first_part_time = 2)
  (h4 : first_part_speed = 80)
  (h5 : remaining_speed = 40) :
  let total_distance := total_time * original_speed
  let first_part_distance := first_part_time * first_part_speed
  let remaining_distance := total_distance - first_part_distance
  remaining_distance / remaining_speed = 5.0625 := by
  sorry

end remaining_trip_time_l369_36933


namespace win_sector_area_l369_36985

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 12) (h2 : p = 1/4) :
  p * (π * r^2) = 36 * π := by
  sorry

end win_sector_area_l369_36985


namespace increasing_function_parameter_range_l369_36904

/-- Given that f(x) = x^3 + ax + 1/x is an increasing function on (1/2, +∞),
    prove that a ∈ [13/4, +∞) -/
theorem increasing_function_parameter_range
  (f : ℝ → ℝ)
  (a : ℝ)
  (h1 : ∀ x > 1/2, f x = x^3 + a*x + 1/x)
  (h2 : StrictMono f) :
  a ∈ Set.Ici (13/4) :=
sorry

end increasing_function_parameter_range_l369_36904


namespace triangle_perimeter_l369_36949

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that the perimeter is 7 + √19 under the following conditions:
    1) a² - c² + 3b = 0
    2) The area of the triangle is 5√3/2
    3) Angle A = 60° -/
theorem triangle_perimeter (a b c : ℝ) (A : ℝ) (S : ℝ) : 
  a^2 - c^2 + 3*b = 0 → 
  S = (5 * Real.sqrt 3) / 2 →
  A = π / 3 →
  a + b + c = 7 + Real.sqrt 19 := by
  sorry

end triangle_perimeter_l369_36949


namespace exponential_inequality_l369_36943

theorem exponential_inequality (a b : ℝ) (h : a > b) : (0.9 : ℝ) ^ a < (0.9 : ℝ) ^ b := by
  sorry

end exponential_inequality_l369_36943


namespace points_collinear_l369_36965

/-- Prove that points A(-1, -2), B(2, -1), and C(8, 1) are collinear. -/
theorem points_collinear : 
  let A : ℝ × ℝ := (-1, -2)
  let B : ℝ × ℝ := (2, -1)
  let C : ℝ × ℝ := (8, 1)
  ∃ (t : ℝ), C - A = t • (B - A) :=
by sorry

end points_collinear_l369_36965


namespace solve_for_x_l369_36952

def U (x : ℝ) : Set ℝ := {2, 3, x^2 + 2*x - 3}
def A (x : ℝ) : Set ℝ := {2, |x + 7|}

theorem solve_for_x : 
  ∃ x : ℝ, (U x \ A x = {5}) ∧ x = -4 :=
sorry

end solve_for_x_l369_36952


namespace marks_weekly_reading_time_l369_36947

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- Mark's current daily reading time in hours -/
def daily_reading_time : ℕ := 3

/-- Mark's planned weekly increase in reading time in hours -/
def weekly_increase : ℕ := 6

/-- Theorem: Mark's total weekly reading time after the increase will be 27 hours -/
theorem marks_weekly_reading_time :
  daily_reading_time * days_in_week + weekly_increase = 27 := by
  sorry

end marks_weekly_reading_time_l369_36947


namespace union_of_A_and_B_l369_36961

-- Define set A
def A : Set ℝ := {x | -1 ≤ 2*x + 1 ∧ 2*x + 1 ≤ 3}

-- Define set B
def B : Set ℝ := {x | x ≠ 0 ∧ (x - 2) / x ≤ 0}

-- Theorem statement
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x ≤ 2} := by sorry

end union_of_A_and_B_l369_36961


namespace function_properties_l369_36929

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / x

theorem function_properties (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ > 0 ∧ x₁ < x₂ → f a x₁ < f a x₂) ∧
  (∀ m n : ℝ, m > 0 ∧ n > 0 ∧ m < n ∧ f a m = 2*m ∧ f a n = 2*n → a > 2 * Real.sqrt 2) ∧
  ((∀ x : ℝ, x ∈ Set.Icc (1/3) (1/2) → x^2 * |f a x| ≤ 1) → a ∈ Set.Icc (-2) 6) :=
by sorry

end function_properties_l369_36929


namespace ferris_wheel_rides_l369_36922

/-- The number of times Will rode the Ferris wheel during the day -/
def daytime_rides : ℕ := 7

/-- The number of times Will rode the Ferris wheel at night -/
def nighttime_rides : ℕ := 6

/-- The total number of times Will rode the Ferris wheel -/
def total_rides : ℕ := daytime_rides + nighttime_rides

theorem ferris_wheel_rides : total_rides = 13 := by
  sorry

end ferris_wheel_rides_l369_36922


namespace odd_not_div_by_three_square_plus_five_div_by_six_l369_36993

theorem odd_not_div_by_three_square_plus_five_div_by_six (n : ℤ) 
  (h_odd : Odd n) (h_not_div_three : ¬(3 ∣ n)) : 
  6 ∣ (n^2 + 5) := by
  sorry

end odd_not_div_by_three_square_plus_five_div_by_six_l369_36993


namespace log_101600_value_l369_36900

-- Define the logarithm function (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_101600_value (h : log 102 = 0.3010) : log 101600 = 2.3010 := by
  sorry

end log_101600_value_l369_36900


namespace gcd_of_72_120_168_l369_36955

theorem gcd_of_72_120_168 : Nat.gcd 72 (Nat.gcd 120 168) = 24 := by
  sorry

end gcd_of_72_120_168_l369_36955


namespace collinear_vectors_m_value_l369_36946

def a : Fin 2 → ℝ := ![2, 3]
def b : Fin 2 → ℝ := ![-1, 2]

theorem collinear_vectors_m_value :
  ∃ (m : ℝ), ∃ (k : ℝ),
    (k ≠ 0) ∧
    (∀ i : Fin 2, k * (m * a i + 4 * b i) = (a i - 2 * b i)) →
    m = -2 :=
sorry

end collinear_vectors_m_value_l369_36946


namespace max_binomial_coeff_x_minus_2_pow_5_l369_36932

theorem max_binomial_coeff_x_minus_2_pow_5 :
  (Finset.range 6).sup (fun k => Nat.choose 5 k) = 10 := by
  sorry

end max_binomial_coeff_x_minus_2_pow_5_l369_36932


namespace equation_solution_l369_36986

noncomputable def f (x a : ℝ) : ℝ := Real.sqrt ((x + 2)^2 + 4 * a^2 - 4) + Real.sqrt ((x - 2)^2 + 4 * a^2 - 4)

theorem equation_solution (a b : ℝ) (h : b ≥ 0) :
  (∀ x, f x a = 4 * b → (b ∈ Set.Icc 0 1 ∪ Set.Ioi 1 → x = 0) ∧
                        (b = 1 → x ∈ Set.Icc (-2) 2)) :=
by sorry

end equation_solution_l369_36986


namespace trains_at_initial_positions_l369_36937

/-- Represents a metro line with a given number of stations -/
structure MetroLine where
  stations : ℕ
  roundTripTime : ℕ

/-- Theorem: After 2016 minutes, all trains are at their initial positions -/
theorem trains_at_initial_positions 
  (red : MetroLine) 
  (blue : MetroLine) 
  (green : MetroLine)
  (h_red : red.stations = 7 ∧ red.roundTripTime = 14)
  (h_blue : blue.stations = 8 ∧ blue.roundTripTime = 16)
  (h_green : green.stations = 9 ∧ green.roundTripTime = 18) :
  2016 % red.roundTripTime = 0 ∧ 
  2016 % blue.roundTripTime = 0 ∧ 
  2016 % green.roundTripTime = 0 := by
  sorry

#eval 2016 % 14  -- Should output 0
#eval 2016 % 16  -- Should output 0
#eval 2016 % 18  -- Should output 0

end trains_at_initial_positions_l369_36937


namespace anne_weight_proof_l369_36916

/-- Anne's weight in pounds -/
def anne_weight : ℕ := 67

/-- Douglas's weight in pounds -/
def douglas_weight : ℕ := 52

/-- The difference in weight between Anne and Douglas in pounds -/
def weight_difference : ℕ := 15

/-- Theorem: Anne's weight is 67 pounds, given Douglas's weight and the weight difference -/
theorem anne_weight_proof : anne_weight = douglas_weight + weight_difference := by
  sorry

end anne_weight_proof_l369_36916


namespace strip_width_problem_l369_36977

theorem strip_width_problem (width1 width2 : ℕ) 
  (h1 : width1 = 44) (h2 : width2 = 33) : 
  Nat.gcd width1 width2 = 11 := by
  sorry

end strip_width_problem_l369_36977


namespace unique_positive_integer_solution_l369_36930

theorem unique_positive_integer_solution : ∃! (x : ℕ), x > 0 ∧ x - x^2 + 29 = 526 := by
  sorry

end unique_positive_integer_solution_l369_36930


namespace three_digit_square_ends_with_itself_l369_36997

theorem three_digit_square_ends_with_itself (A : ℕ) : 
  (100 ≤ A ∧ A ≤ 999) → (A^2 ≡ A [ZMOD 1000]) ↔ (A = 376 ∨ A = 625) := by
  sorry

end three_digit_square_ends_with_itself_l369_36997


namespace sqrt_three_subtraction_l369_36995

theorem sqrt_three_subtraction : 2 * Real.sqrt 3 - Real.sqrt 3 = Real.sqrt 3 := by
  sorry

end sqrt_three_subtraction_l369_36995


namespace geometric_sequence_ratio_l369_36918

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a n > 0) →
  (∀ n, a (n + 1) = q * a n) →
  q > 0 →
  a 3 + a 4 = a 5 →
  (a 3 + a 5) / (a 4 + a 6) = (Real.sqrt 5 - 1) / 2 := by
  sorry

end geometric_sequence_ratio_l369_36918


namespace parallel_vectors_magnitude_l369_36963

/-- Given vectors a and b where a is parallel to b, prove that |3a + 2b| = √5 -/
theorem parallel_vectors_magnitude (y : ℝ) :
  let a : Fin 2 → ℝ := ![1, -2]
  let b : Fin 2 → ℝ := ![-2, y]
  (∃ (k : ℝ), a = k • b) →
  ‖(3 : ℝ) • a + (2 : ℝ) • b‖ = Real.sqrt 5 := by
  sorry

end parallel_vectors_magnitude_l369_36963


namespace count_squares_in_H_l369_36939

/-- The set of points (x,y) with integer coordinates satisfying 2 ≤ |x| ≤ 8 and 2 ≤ |y| ≤ 8 -/
def H : Set (ℤ × ℤ) :=
  {p | 2 ≤ |p.1| ∧ |p.1| ≤ 8 ∧ 2 ≤ |p.2| ∧ |p.2| ≤ 8}

/-- A square with vertices in H -/
structure SquareInH where
  vertices : Fin 4 → ℤ × ℤ
  in_H : ∀ i, vertices i ∈ H
  is_square : ∃ (side : ℤ), side ≥ 5 ∧
    (vertices 1).1 - (vertices 0).1 = side ∧
    (vertices 2).1 - (vertices 1).1 = side ∧
    (vertices 3).1 - (vertices 2).1 = -side ∧
    (vertices 0).1 - (vertices 3).1 = -side ∧
    (vertices 1).2 - (vertices 0).2 = side ∧
    (vertices 2).2 - (vertices 1).2 = -side ∧
    (vertices 3).2 - (vertices 2).2 = -side ∧
    (vertices 0).2 - (vertices 3).2 = side

/-- The number of squares with side length at least 5 whose vertices are in H -/
def numSquaresInH : ℕ := sorry

theorem count_squares_in_H : numSquaresInH = 14 := by sorry

end count_squares_in_H_l369_36939


namespace max_triangle_area_l369_36950

/-- The maximum area of a triangle ABC with side AB = 13 and BC:AC ratio of 60:61 is 3634 -/
theorem max_triangle_area (A B C : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (AB + BC + AC) / 2
  let area := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))
  AB = 13 ∧ BC / AC = 60 / 61 → area ≤ 3634 :=
by sorry


end max_triangle_area_l369_36950


namespace polynomial_division_quotient_l369_36923

theorem polynomial_division_quotient :
  ∀ (x : ℝ), x ≠ 1 →
  (x^6 + 8) / (x - 1) = x^5 + x^4 + x^3 + x^2 + x + 1 := by
  sorry

end polynomial_division_quotient_l369_36923


namespace vectors_collinear_l369_36915

/-- The problem setup -/
structure GeometrySetup where
  -- The coordinate system
  P : ℝ × ℝ
  Q : ℝ × ℝ
  S : ℝ × ℝ
  T : ℝ × ℝ
  N : ℝ × ℝ
  M : ℝ × ℝ
  -- Conditions
  hl : S.1 = -1
  hT : T = (3, 0)
  hPl : S.2 = P.2
  hOP_ST : P.1 * 4 - P.2 * S.2 = 0
  hC : Q.2^2 = 4 * Q.1
  hPQ : ∃ (t : ℝ), (1 - P.1) * t + P.1 = Q.1 ∧ (0 - P.2) * t + P.2 = Q.2
  hM : M = ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  hN : N = (-1, 0)

/-- The theorem to be proved -/
theorem vectors_collinear (g : GeometrySetup) : 
  ∃ (k : ℝ), (g.M.1 - g.S.1, g.M.2 - g.S.2) = k • (g.Q.1 - g.N.1, g.Q.2 - g.N.2) := by
  sorry

end vectors_collinear_l369_36915


namespace balloon_difference_l369_36970

theorem balloon_difference (your_balloons friend_balloons : ℕ) 
  (h1 : your_balloons = 7) 
  (h2 : friend_balloons = 5) : 
  your_balloons - friend_balloons = 2 := by
sorry

end balloon_difference_l369_36970


namespace solve_for_q_l369_36905

theorem solve_for_q (p q : ℝ) 
  (h1 : 1 < p) 
  (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) 
  (h4 : p * q = 16 / 3) : 
  q = 4 := by
sorry

end solve_for_q_l369_36905


namespace reflection_sum_l369_36925

/-- Given a line y = mx + b, if the reflection of point (-3, -1) across this line is (5, 3), then m + b = 1 -/
theorem reflection_sum (m b : ℝ) : 
  (∃ (x y : ℝ), 
    (x = ((-3) + 5) / 2 ∧ y = ((-1) + 3) / 2) ∧ 
    (y = m * x + b) ∧
    (m = -(5 - (-3)) / (3 - (-1))))
  → m + b = 1 := by sorry

end reflection_sum_l369_36925


namespace luke_fish_fillets_l369_36912

/-- Calculates the number of fillets per fish given the total number of fish caught and total fillets obtained. -/
def filletsPerFish (fishPerDay : ℕ) (days : ℕ) (totalFillets : ℕ) : ℚ :=
  totalFillets / (fishPerDay * days)

/-- Proves that the number of fillets per fish is 2 given the problem conditions. -/
theorem luke_fish_fillets : filletsPerFish 2 30 120 = 2 := by
  sorry

end luke_fish_fillets_l369_36912


namespace tan_105_degrees_l369_36976

theorem tan_105_degrees :
  Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end tan_105_degrees_l369_36976


namespace parallel_linear_function_through_point_l369_36908

-- Define a linear function
def linear_function (k b : ℝ) : ℝ → ℝ := λ x => k * x + b

-- State the theorem
theorem parallel_linear_function_through_point :
  ∀ (k b : ℝ),
  -- The linear function is parallel to y = 2x + 1
  k = 2 →
  -- The linear function passes through the point (-1, 1)
  linear_function k b (-1) = 1 →
  -- The linear function is equal to y = 2x + 3
  linear_function k b = linear_function 2 3 := by
sorry


end parallel_linear_function_through_point_l369_36908


namespace fish_pond_population_l369_36960

theorem fish_pond_population (initial_tagged : ℕ) (second_catch : ℕ) (tagged_in_second : ℕ) :
  initial_tagged = 80 →
  second_catch = 80 →
  tagged_in_second = 2 →
  (tagged_in_second : ℚ) / second_catch = initial_tagged / (initial_tagged * second_catch / tagged_in_second) :=
by
  sorry

#check fish_pond_population

end fish_pond_population_l369_36960


namespace harvest_time_calculation_l369_36994

theorem harvest_time_calculation (initial_harvesters initial_days initial_area final_harvesters final_area : ℕ) 
  (h1 : initial_harvesters = 2)
  (h2 : initial_days = 3)
  (h3 : initial_area = 450)
  (h4 : final_harvesters = 7)
  (h5 : final_area = 2100) :
  (initial_harvesters * initial_days * final_area) / (initial_area * final_harvesters) = 4 := by
  sorry

end harvest_time_calculation_l369_36994


namespace evening_pages_read_l369_36972

/-- Given a person who reads books with the following conditions:
  * Reads twice a day (morning and evening)
  * Reads 5 pages in the morning
  * Reads at this rate for a week (7 days)
  * Reads a total of 105 pages in a week
This theorem proves that the number of pages read in the evening is 10. -/
theorem evening_pages_read (morning_pages : ℕ) (total_pages : ℕ) (days : ℕ) :
  morning_pages = 5 →
  days = 7 →
  total_pages = 105 →
  ∃ (evening_pages : ℕ), 
    days * (morning_pages + evening_pages) = total_pages ∧ 
    evening_pages = 10 := by
  sorry

end evening_pages_read_l369_36972


namespace jelly_bean_probability_l369_36979

/-- The probability of selecting either a blue or yellow jelly bean from a bag -/
theorem jelly_bean_probability : 
  let red : ℕ := 6
  let green : ℕ := 7
  let yellow : ℕ := 8
  let blue : ℕ := 9
  let total : ℕ := red + green + yellow + blue
  let target : ℕ := yellow + blue
  (target : ℚ) / total = 17 / 30 := by
  sorry

end jelly_bean_probability_l369_36979


namespace complex_modulus_example_l369_36978

theorem complex_modulus_example : Complex.abs (3 - 10 * Complex.I * Real.sqrt 3) = Real.sqrt 309 := by
  sorry

end complex_modulus_example_l369_36978


namespace solve_system_of_equations_solve_system_of_inequalities_l369_36968

-- Part 1: System of Equations
def system_of_equations (x y : ℝ) : Prop :=
  (2 * x - y = 3) ∧ (x + y = 6)

theorem solve_system_of_equations :
  ∃ x y : ℝ, system_of_equations x y ∧ x = 3 ∧ y = 3 :=
sorry

-- Part 2: System of Inequalities
def system_of_inequalities (x : ℝ) : Prop :=
  (3 * x > x - 4) ∧ ((4 + x) / 3 > x + 2)

theorem solve_system_of_inequalities :
  ∀ x : ℝ, system_of_inequalities x ↔ -2 < x ∧ x < -1 :=
sorry

end solve_system_of_equations_solve_system_of_inequalities_l369_36968


namespace max_radius_of_circle_l369_36992

-- Define the circle C
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the two given points
def point1 : ℝ × ℝ := (4, 0)
def point2 : ℝ × ℝ := (-4, 0)

-- Theorem statement
theorem max_radius_of_circle (C : ℝ × ℝ → ℝ → Set (ℝ × ℝ)) 
  (h1 : point1 ∈ C center radius) (h2 : point2 ∈ C center radius) :
  ∃ (center : ℝ × ℝ) (radius : ℝ), radius ≤ 4 ∧ 
  (∀ (center' : ℝ × ℝ) (radius' : ℝ), 
    point1 ∈ C center' radius' → point2 ∈ C center' radius' → radius' ≤ radius) :=
sorry

end max_radius_of_circle_l369_36992


namespace ramanujan_number_proof_l369_36938

def hardy_number : ℂ := 4 + 6 * Complex.I

theorem ramanujan_number_proof (product : ℂ) (h : product = 40 - 24 * Complex.I) :
  ∃ (ramanujan_number : ℂ), 
    ramanujan_number * hardy_number = product ∧ 
    ramanujan_number = 76/13 - 36/13 * Complex.I :=
by
  sorry

end ramanujan_number_proof_l369_36938


namespace inequality_solution_set_l369_36998

theorem inequality_solution_set : 
  {x : ℝ | x + 5 > -1} = {x : ℝ | x > -6} := by sorry

end inequality_solution_set_l369_36998


namespace function_positive_interval_implies_m_range_l369_36969

theorem function_positive_interval_implies_m_range 
  (F : ℝ → ℝ) (m : ℝ) 
  (h_def : ∀ x, F x = -x^2 - m*x + 1) 
  (h_pos : ∀ x ∈ Set.Icc m (m+1), F x > 0) : 
  m > -Real.sqrt 2 / 2 ∧ m < 0 := by
sorry

end function_positive_interval_implies_m_range_l369_36969


namespace midline_leg_relation_l369_36962

/-- A right triangle with legs a and b, and midlines K₁ and K₂. -/
structure RightTriangle where
  a : ℝ
  b : ℝ
  K₁ : ℝ
  K₂ : ℝ
  a_positive : 0 < a
  b_positive : 0 < b
  K₁_eq : K₁^2 = (a/2)^2 + b^2
  K₂_eq : K₂^2 = a^2 + (b/2)^2

/-- The main theorem about the relationship between midlines and leg in a right triangle. -/
theorem midline_leg_relation (t : RightTriangle) : 16 * t.K₂^2 - 4 * t.K₁^2 = 15 * t.a^2 := by
  sorry

end midline_leg_relation_l369_36962


namespace triangle_inequality_l369_36903

theorem triangle_inequality (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ A < π/2 ∧
  0 < B ∧ B < π/2 ∧
  0 < C ∧ C < π/2 ∧
  A + B + C = π ∧
  a = 1 ∧
  b * Real.cos A - Real.cos B = 1 →
  Real.sqrt 3 < Real.sin B + 2 * Real.sqrt 3 * Real.sin A * Real.sin A ∧
  Real.sin B + 2 * Real.sqrt 3 * Real.sin A * Real.sin A < 1 + Real.sqrt 3 :=
by sorry

end triangle_inequality_l369_36903


namespace equality_of_polynomials_l369_36983

theorem equality_of_polynomials (a b c : ℝ) :
  (∀ x : ℝ, (x^2 + a*x - 3)*(x + 1) = x^3 + b*x^2 + c*x - 3) →
  b - c = 4 := by
  sorry

end equality_of_polynomials_l369_36983


namespace cube_volume_ratio_l369_36971

theorem cube_volume_ratio (edge_q : ℝ) (edge_p : ℝ) (h : edge_p = 3 * edge_q) :
  (edge_q ^ 3) / (edge_p ^ 3) = 1 / 27 := by
sorry

end cube_volume_ratio_l369_36971


namespace sqrt_product_simplification_l369_36924

theorem sqrt_product_simplification (y : ℝ) (hy : y > 0) :
  Real.sqrt (48 * y) * Real.sqrt (18 * y) * Real.sqrt (50 * y) = 120 * y * Real.sqrt (3 * y) := by
  sorry

end sqrt_product_simplification_l369_36924


namespace frustum_cone_altitude_l369_36953

theorem frustum_cone_altitude (h : ℝ) (A_lower A_upper : ℝ) :
  h = 24 →
  A_lower = 225 * Real.pi →
  A_upper = 25 * Real.pi →
  ∃ x : ℝ, x = 12 ∧ x = (1/3) * (3/2 * h) :=
by sorry

end frustum_cone_altitude_l369_36953


namespace gcd_4288_9277_l369_36934

theorem gcd_4288_9277 : Int.gcd 4288 9277 = 1 := by sorry

end gcd_4288_9277_l369_36934


namespace fixed_point_satisfies_line_equation_fixed_point_is_unique_l369_36988

/-- The line equation as a function of m, x, and y -/
def line_equation (m x y : ℝ) : ℝ := (3*m + 4)*x + (5 - 2*m)*y + 7*m - 6

/-- The fixed point through which all lines pass -/
def fixed_point : ℝ × ℝ := (-1, 2)

/-- Theorem stating that the fixed point satisfies the line equation for all real m -/
theorem fixed_point_satisfies_line_equation :
  ∀ (m : ℝ), line_equation m fixed_point.1 fixed_point.2 = 0 := by sorry

/-- Theorem stating that the fixed point is unique -/
theorem fixed_point_is_unique :
  ∀ (x y : ℝ), (∀ (m : ℝ), line_equation m x y = 0) → (x, y) = fixed_point := by sorry

end fixed_point_satisfies_line_equation_fixed_point_is_unique_l369_36988


namespace quadratic_coefficient_sum_l369_36927

/-- A quadratic function passing through (1,0) and (5,0) with minimum value 36 -/
def quadratic (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_coefficient_sum (a b c : ℝ) :
  (quadratic a b c 1 = 0) →
  (quadratic a b c 5 = 0) →
  (∃ x, ∀ y, quadratic a b c y ≥ quadratic a b c x) →
  (∃ x, quadratic a b c x = 36) →
  a + b + c = 0 := by
  sorry

end quadratic_coefficient_sum_l369_36927


namespace fraction_equality_l369_36964

theorem fraction_equality (p q s u : ℚ) 
  (h1 : p / q = 5 / 6) 
  (h2 : s / u = 7 / 15) : 
  (5 * p * s - 3 * q * u) / (6 * q * u - 5 * p * s) = -19 / 73 := by
  sorry

end fraction_equality_l369_36964


namespace coplanar_iff_k_eq_neg_eight_l369_36907

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [FiniteDimensional ℝ V]

-- Define points as vectors
variable (O A B C D E : V)

-- Define the scalar k
variable (k : ℝ)

-- Define the condition from the problem
def vector_equation (O A B C D E : V) (k : ℝ) : Prop :=
  4 • (A - O) - 3 • (B - O) + 6 • (C - O) + k • (D - O) + (E - O) = 0

-- Define coplanarity
def coplanar (A B C D E : V) : Prop :=
  ∃ (a b c d : ℝ), a • (B - A) + b • (C - A) + c • (D - A) + d • (E - A) = 0

-- State the theorem
theorem coplanar_iff_k_eq_neg_eight
  (O A B C D E : V) (k : ℝ) :
  vector_equation O A B C D E k →
  (coplanar A B C D E ↔ k = -8) :=
sorry

end coplanar_iff_k_eq_neg_eight_l369_36907


namespace max_additional_spheres_is_two_l369_36989

/-- Represents a truncated cone -/
structure TruncatedCone where
  height : ℝ

/-- Represents a sphere -/
structure Sphere where
  radius : ℝ
  center : ℝ × ℝ × ℝ

/-- Represents the configuration of spheres in the truncated cone -/
structure SphereConfiguration where
  cone : TruncatedCone
  O₁ : Sphere
  O₂ : Sphere

/-- Calculates the maximum number of additional spheres that can be placed in the cone -/
def maxAdditionalSpheres (config : SphereConfiguration) : ℕ :=
  sorry

/-- The main theorem stating the maximum number of additional spheres -/
theorem max_additional_spheres_is_two (config : SphereConfiguration) :
  config.cone.height = 8 ∧
  config.O₁.radius = 2 ∧
  config.O₂.radius = 3 →
  maxAdditionalSpheres config = 2 :=
by sorry

end max_additional_spheres_is_two_l369_36989


namespace fiftieth_ring_squares_l369_36966

/-- The number of squares in the nth ring around a 3x3 centered square -/
def ring_squares (n : ℕ) : ℕ :=
  if n = 1 then 16
  else if n = 2 then 24
  else 33 + 24 * (n - 1)

/-- The 50th ring contains 1209 unit squares -/
theorem fiftieth_ring_squares :
  ring_squares 50 = 1209 := by
  sorry

end fiftieth_ring_squares_l369_36966


namespace length_of_AD_rhombus_condition_l369_36921

-- Define the quadratic equation
def quadratic_equation (a x : ℝ) : ℝ := x^2 - (a - 4) * x + a - 1

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (AB AD : ℝ)
  (a : ℝ)
  (eq_AB : quadratic_equation a AB = 0)
  (eq_AD : quadratic_equation a AD = 0)

-- Theorem 1: Length of AD
theorem length_of_AD (ABCD : Quadrilateral) (h : ABCD.AB = 2) : ABCD.AD = 5 := by
  sorry

-- Theorem 2: Condition for rhombus
theorem rhombus_condition (ABCD : Quadrilateral) : ABCD.AB = ABCD.AD ↔ ABCD.a = 10 := by
  sorry

end length_of_AD_rhombus_condition_l369_36921


namespace smallest_prime_divisor_of_sum_l369_36944

theorem smallest_prime_divisor_of_sum (p : ℕ → ℕ → ℕ) :
  (∃ (d : ℕ), d ∣ (p 3 15 + p 11 21) ∧ Prime d) →
  (∀ (d : ℕ), d ∣ (p 3 15 + p 11 21) ∧ Prime d → d ≥ 2) →
  (∃ (d : ℕ), d ∣ (p 3 15 + p 11 21) ∧ Prime d ∧ d = 2) :=
by sorry

end smallest_prime_divisor_of_sum_l369_36944


namespace white_mice_count_l369_36954

theorem white_mice_count (total : ℕ) (white : ℕ) (brown : ℕ) : 
  (white = 2 * total / 3) →  -- 2/3 of the mice are white
  (brown = 7) →              -- There are 7 brown mice
  (total = white + brown) →  -- Total mice is the sum of white and brown mice
  (white > 0) →              -- There are some white mice
  (white = 14) :=            -- The number of white mice is 14
by
  sorry

end white_mice_count_l369_36954


namespace folded_paper_distance_l369_36906

theorem folded_paper_distance (area : ℝ) (h_area : area = 12) : ℝ :=
  let side_length := Real.sqrt area
  let folded_side_length := Real.sqrt (area / 2)
  let distance := Real.sqrt (2 * folded_side_length ^ 2)
  
  have h_distance : distance = 2 * Real.sqrt 6 := by sorry
  
  distance

end folded_paper_distance_l369_36906
