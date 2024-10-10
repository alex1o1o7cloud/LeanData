import Mathlib

namespace constant_term_expansion_l432_43221

/-- The constant term in the expansion of (x^2 + a/sqrt(x))^5 -/
def constantTerm (a : ℝ) : ℝ := 5 * a^4

theorem constant_term_expansion (a : ℝ) (h1 : a > 0) (h2 : constantTerm a = 80) : a = 2 := by
  sorry

#check constant_term_expansion

end constant_term_expansion_l432_43221


namespace car_race_distance_l432_43248

theorem car_race_distance (v_A v_B : ℝ) (d : ℝ) :
  v_A > 0 ∧ v_B > 0 ∧ d > 0 →
  (v_A / v_B = (2 * v_A) / (2 * v_B)) →
  (d / v_A = (d/2) / (2 * v_A)) →
  15 = 15 := by sorry

end car_race_distance_l432_43248


namespace first_level_teachers_selected_l432_43250

/-- Represents the number of teachers selected in a stratified sample -/
def stratified_sample (total : ℕ) (senior : ℕ) (first_level : ℕ) (second_level : ℕ) (sample_size : ℕ) : ℕ :=
  (first_level * sample_size) / (senior + first_level + second_level)

/-- Theorem stating that the number of first-level teachers selected in the given scenario is 12 -/
theorem first_level_teachers_selected :
  stratified_sample 380 90 120 170 38 = 12 := by
  sorry

end first_level_teachers_selected_l432_43250


namespace zero_has_square_root_l432_43272

theorem zero_has_square_root : ∃ x : ℝ, x^2 = 0 := by
  sorry

end zero_has_square_root_l432_43272


namespace diamond_two_seven_l432_43213

-- Define the diamond operation
def diamond (a b : ℤ) : ℤ := a^2 * b - a * b^2

-- Theorem statement
theorem diamond_two_seven : diamond 2 7 = -70 := by
  sorry

end diamond_two_seven_l432_43213


namespace distance_to_origin_l432_43205

theorem distance_to_origin : let M : ℝ × ℝ := (-3, 4)
                             Real.sqrt ((M.1 - 0)^2 + (M.2 - 0)^2) = 5 := by
  sorry

end distance_to_origin_l432_43205


namespace pat_initial_stickers_l432_43227

/-- The number of stickers Pat had at the end of the week -/
def end_stickers : ℕ := 61

/-- The number of stickers Pat earned during the week -/
def earned_stickers : ℕ := 22

/-- The number of stickers Pat had on the first day of the week -/
def initial_stickers : ℕ := end_stickers - earned_stickers

theorem pat_initial_stickers : initial_stickers = 39 := by
  sorry

end pat_initial_stickers_l432_43227


namespace sandwich_cookie_cost_l432_43231

theorem sandwich_cookie_cost (s c : ℝ) 
  (eq1 : 3 * s + 4 * c = 4.20)
  (eq2 : 4 * s + 3 * c = 4.50) : 
  4 * s + 5 * c = 5.44 := by
sorry

end sandwich_cookie_cost_l432_43231


namespace female_students_count_l432_43262

/-- Given a class with n total students and m male students, 
    prove that the number of female students is n - m. -/
theorem female_students_count (n m : ℕ) : ℕ :=
  n - m

#check female_students_count

end female_students_count_l432_43262


namespace find_k_value_l432_43292

theorem find_k_value (k : ℝ) (h1 : k ≠ 0) : 
  (∀ x : ℝ, (x^2 - k) * (x + 3*k) = x^3 + k*(x^2 - 2*x - 8)) → k = 8/3 := by
  sorry

end find_k_value_l432_43292


namespace g_at_one_l432_43281

theorem g_at_one (a b c d : ℝ) (h₁ : 1 < a) (h₂ : a < b) (h₃ : b < c) (h₄ : c < d) :
  let f : ℝ → ℝ := λ x => x^4 + a*x^3 + b*x^2 + c*x + d
  ∃ g : ℝ → ℝ,
    (∀ x, g x = 0 → ∃ y, f y = 0 ∧ x * y = 1) ∧
    (g 0 = 1) ∧
    (g 1 = (1 + a + b + c + d) / d) :=
by sorry

end g_at_one_l432_43281


namespace junior_girls_count_l432_43291

theorem junior_girls_count (total_players : ℕ) (boy_percentage : ℚ) : 
  total_players = 50 → 
  boy_percentage = 60 / 100 → 
  (total_players : ℚ) * (1 - boy_percentage) / 2 = 10 := by
  sorry

end junior_girls_count_l432_43291


namespace min_throws_for_repeated_sum_l432_43273

theorem min_throws_for_repeated_sum (n : ℕ) (d : ℕ) (s : ℕ) : 
  n = 4 →  -- number of dice
  d = 6 →  -- number of sides on each die
  s = (n * d - n * 1 + 1) →  -- number of possible sums
  s + 1 = 22 →  -- minimum number of throws
  ∀ (throws : ℕ), throws ≥ s + 1 → 
    ∃ (sum1 sum2 : ℕ) (i j : ℕ), 
      i ≠ j ∧ i < throws ∧ j < throws ∧ sum1 = sum2 :=
by sorry

end min_throws_for_repeated_sum_l432_43273


namespace smallest_n_for_probability_l432_43257

theorem smallest_n_for_probability (n : ℕ) : n ≥ 11 ↔ (n - 4 : ℝ)^3 / (n - 2 : ℝ)^3 > 1/3 :=
by sorry

end smallest_n_for_probability_l432_43257


namespace complex_power_215_36_l432_43200

theorem complex_power_215_36 : (Complex.exp (215 * π / 180 * Complex.I))^36 = -1 := by
  sorry

end complex_power_215_36_l432_43200


namespace solution_set_for_even_monotonic_function_l432_43238

-- Define the properties of the function f
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_monotonic_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x → 0 < y → x < y → f x < f y

-- Define the set of solutions
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f (x + 1) = f (2 * x)}

-- Theorem statement
theorem solution_set_for_even_monotonic_function
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_monotonic : is_monotonic_on_positive f) :
  solution_set f = {1, -1/3} := by
sorry

end solution_set_for_even_monotonic_function_l432_43238


namespace cost_of_cherries_l432_43246

/-- Given Sally's purchase of peaches and cherries, prove the cost of cherries. -/
theorem cost_of_cherries
  (peaches_after_coupon : ℝ)
  (coupon_value : ℝ)
  (total_cost : ℝ)
  (h1 : peaches_after_coupon = 12.32)
  (h2 : coupon_value = 3)
  (h3 : total_cost = 23.86) :
  total_cost - (peaches_after_coupon + coupon_value) = 8.54 := by
  sorry

end cost_of_cherries_l432_43246


namespace absolute_value_expression_l432_43270

theorem absolute_value_expression : 
  |(-2)| * (|(-Real.sqrt 25)| - |Real.sin (5 * Real.pi / 2)|) = 8 := by
  sorry

end absolute_value_expression_l432_43270


namespace constant_remainder_iff_a_eq_neg_35_l432_43276

/-- The dividend polynomial -/
def dividend (a : ℚ) (x : ℚ) : ℚ := 10 * x^3 - 7 * x^2 + a * x + 10

/-- The divisor polynomial -/
def divisor (x : ℚ) : ℚ := 2 * x^2 - 5 * x + 2

/-- The remainder when dividend is divided by divisor -/
def remainder (a : ℚ) (x : ℚ) : ℚ := dividend a x - divisor x * (5 * x + 15/2)

theorem constant_remainder_iff_a_eq_neg_35 :
  (∃ (c : ℚ), ∀ (x : ℚ), remainder a x = c) ↔ a = -35 := by sorry

end constant_remainder_iff_a_eq_neg_35_l432_43276


namespace complex_sum_of_sixth_powers_l432_43230

theorem complex_sum_of_sixth_powers : 
  (((1 : ℂ) + Complex.I * Real.sqrt 3) / 2) ^ 6 + 
  (((1 : ℂ) - Complex.I * Real.sqrt 3) / 2) ^ 6 = 
  (1 : ℂ) / 2 := by sorry

end complex_sum_of_sixth_powers_l432_43230


namespace arithmetic_geometric_ratio_l432_43203

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h_arithmetic : ∀ n, a (n + 1) = a n + d
  h_nonzero : d ≠ 0

/-- Theorem: If a_1, a_3, and a_7 of an arithmetic sequence form a geometric sequence,
    then the common ratio of this geometric sequence is 2 -/
theorem arithmetic_geometric_ratio
  (seq : ArithmeticSequence)
  (h_geometric : (seq.a 3) ^ 2 = (seq.a 1) * (seq.a 7)) :
  (seq.a 3) / (seq.a 1) = 2 :=
sorry

end arithmetic_geometric_ratio_l432_43203


namespace cardinality_of_P_l432_43211

def M : Finset ℕ := {0, 1, 2, 3, 4}
def N : Finset ℕ := {1, 3, 5}
def P : Finset ℕ := M ∩ N

theorem cardinality_of_P : Finset.card P = 2 := by sorry

end cardinality_of_P_l432_43211


namespace circle_ellipse_tangent_l432_43298

-- Define the circle M
def circle_M (m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + 2*m*x - 3 = 0

-- Define the ellipse C
def ellipse_C (a : ℝ) (x y : ℝ) : Prop :=
  x^2/a^2 + y^2/3 = 1

-- Define the line l
def line_l (c : ℝ) (x y : ℝ) : Prop :=
  x = -c

theorem circle_ellipse_tangent (m c a : ℝ) :
  m < 0 →  -- m is negative
  (∀ x y, circle_M m x y → (x + m)^2 + y^2 = 4) →  -- radius of M is 2
  (∃ x y, ellipse_C a x y ∧ x = -c ∧ y = 0) →  -- left focus of C is F(-c, 0)
  (∀ x y, line_l c x y → (x - 1)^2 = 4) →  -- l is tangent to M
  a = 2 := by
  sorry

end circle_ellipse_tangent_l432_43298


namespace tomato_soup_cans_l432_43249

/-- Proves the number of cans of tomato soup sold for every 4 cans of chili beans -/
theorem tomato_soup_cans (total_cans : ℕ) (chili_beans_cans : ℕ) 
  (h1 : total_cans = 12)
  (h2 : chili_beans_cans = 8)
  (h3 : ∃ (n : ℕ), n * 4 = total_cans - chili_beans_cans) :
  4 = total_cans - chili_beans_cans :=
by sorry

end tomato_soup_cans_l432_43249


namespace bond_paper_cost_l432_43260

/-- Represents the cost of bond paper for an office. -/
structure BondPaperCost where
  sheets_per_ream : ℕ
  sheets_needed : ℕ
  total_cost : ℚ

/-- Calculates the cost of one ream of bond paper. -/
def cost_per_ream (bpc : BondPaperCost) : ℚ :=
  bpc.total_cost / (bpc.sheets_needed / bpc.sheets_per_ream)

/-- Theorem stating that the cost of one ream of bond paper is $27. -/
theorem bond_paper_cost (bpc : BondPaperCost)
  (h1 : bpc.sheets_per_ream = 500)
  (h2 : bpc.sheets_needed = 5000)
  (h3 : bpc.total_cost = 270) :
  cost_per_ream bpc = 27 := by
  sorry

end bond_paper_cost_l432_43260


namespace range_of_a_for_monotone_decreasing_f_l432_43242

/-- A piecewise function f(x) defined on ℝ -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * x^2 - 6*x + a^2 + 1
  else x^(5 - 2*a)

/-- The theorem stating the range of a for which f is monotonically decreasing -/
theorem range_of_a_for_monotone_decreasing_f :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) ↔ a ∈ Set.Ioo (5/2) 3 :=
sorry

end range_of_a_for_monotone_decreasing_f_l432_43242


namespace rainfall_ratio_is_two_l432_43226

-- Define the parameters
def total_rainfall : ℝ := 180
def first_half_daily_rainfall : ℝ := 4
def days_in_november : ℕ := 30
def first_half_days : ℕ := 15

-- Define the theorem
theorem rainfall_ratio_is_two :
  let first_half_total := first_half_daily_rainfall * first_half_days
  let second_half_total := total_rainfall - first_half_total
  let second_half_days := days_in_november - first_half_days
  let second_half_daily_rainfall := second_half_total / second_half_days
  (second_half_daily_rainfall / first_half_daily_rainfall) = 2 := by
  sorry

end rainfall_ratio_is_two_l432_43226


namespace strawberry_picking_problem_l432_43268

/-- The number of times Kimberly picked more strawberries than her brother -/
def kimberlyMultiplier : ℕ → ℕ → ℕ → ℕ → ℕ
| brother_baskets, strawberries_per_basket, parents_difference, equal_share =>
  let brother_strawberries := brother_baskets * strawberries_per_basket
  let total_strawberries := equal_share * 4
  2 * total_strawberries / brother_strawberries - 2

theorem strawberry_picking_problem :
  kimberlyMultiplier 3 15 93 168 = 8 := by
  sorry

end strawberry_picking_problem_l432_43268


namespace violets_family_size_l432_43215

/-- Proves the number of children in Violet's family given ticket prices and total cost -/
theorem violets_family_size (adult_ticket : ℕ) (child_ticket : ℕ) (total_cost : ℕ) :
  adult_ticket = 35 →
  child_ticket = 20 →
  total_cost = 155 →
  ∃ (num_children : ℕ), adult_ticket + num_children * child_ticket = total_cost ∧ num_children = 6 :=
by sorry

end violets_family_size_l432_43215


namespace hyperbola_foci_distance_l432_43297

/-- Given a hyperbola with specified asymptotes and a point it passes through,
    prove that the distance between its foci is 2√(13.5). -/
theorem hyperbola_foci_distance (x₀ y₀ : ℝ) :
  let asymptote1 : ℝ → ℝ := λ x => 2 * x + 2
  let asymptote2 : ℝ → ℝ := λ x => -2 * x + 4
  let point : ℝ × ℝ := (2, 6)
  (∀ x, y₀ = asymptote1 x ∨ y₀ = asymptote2 x → x₀ = x) →
  (y₀ = asymptote1 x₀ ∨ y₀ = asymptote2 x₀) →
  point.1 = 2 ∧ point.2 = 6 →
  ∃ (center : ℝ × ℝ) (a b : ℝ),
    (∀ x y, ((y - center.2)^2 / a^2) - ((x - center.1)^2 / b^2) = 1 →
      y = asymptote1 x ∨ y = asymptote2 x) ∧
    ((point.2 - center.2)^2 / a^2) - ((point.1 - center.1)^2 / b^2) = 1 ∧
    2 * Real.sqrt (a^2 + b^2) = 2 * Real.sqrt 13.5 := by
  sorry

end hyperbola_foci_distance_l432_43297


namespace jerome_solution_l432_43237

def jerome_problem (initial_money : ℕ) (remaining_money : ℕ) (meg_amount : ℕ) : Prop :=
  initial_money = 2 * 43 ∧
  remaining_money = 54 ∧
  initial_money = remaining_money + meg_amount + 3 * meg_amount ∧
  meg_amount = 8

theorem jerome_solution : ∃ initial_money remaining_money meg_amount, jerome_problem initial_money remaining_money meg_amount :=
  sorry

end jerome_solution_l432_43237


namespace clock_rings_eight_times_l432_43295

/-- A clock that rings every 3 hours, starting at 1 A.M. -/
structure Clock :=
  (ring_interval : ℕ := 3)
  (first_ring : ℕ := 1)

/-- The number of times the clock rings in a 24-hour period -/
def rings_per_day (c : Clock) : ℕ :=
  ((24 - c.first_ring) / c.ring_interval) + 1

theorem clock_rings_eight_times (c : Clock) : rings_per_day c = 8 := by
  sorry

end clock_rings_eight_times_l432_43295


namespace gcf_68_92_l432_43251

theorem gcf_68_92 : Nat.gcd 68 92 = 4 := by
  sorry

end gcf_68_92_l432_43251


namespace cylinder_cone_sphere_volume_l432_43266

/-- Given a cylinder with volume 54π cm³ and height three times its radius,
    prove that the total volume of a cone and a sphere both having the same radius
    as the cylinder is 42π cm³ -/
theorem cylinder_cone_sphere_volume (r : ℝ) (h : ℝ) : 
  (π * r^2 * h = 54 * π) →
  (h = 3 * r) →
  (π * r^2 * r / 3 + 4 * π * r^3 / 3 = 42 * π) := by
  sorry

end cylinder_cone_sphere_volume_l432_43266


namespace assignment_schemes_eq_240_l432_43218

/-- The number of ways to assign 4 out of 6 students to tasks A, B, C, and D,
    given that two specific students can perform task A. -/
def assignment_schemes : ℕ :=
  Nat.descFactorial 6 4 - 2 * Nat.descFactorial 5 3

/-- Theorem stating that the number of assignment schemes is 240. -/
theorem assignment_schemes_eq_240 : assignment_schemes = 240 := by
  sorry

end assignment_schemes_eq_240_l432_43218


namespace square_perimeter_from_rectangle_l432_43293

theorem square_perimeter_from_rectangle (rectangle_length rectangle_width : ℝ) 
  (h1 : rectangle_length = 32)
  (h2 : rectangle_width = 10)
  (h3 : rectangle_length > 0)
  (h4 : rectangle_width > 0) :
  ∃ (square_side : ℝ), 
    square_side^2 = 5 * (rectangle_length * rectangle_width) ∧ 
    4 * square_side = 160 := by
sorry

end square_perimeter_from_rectangle_l432_43293


namespace june_1_2014_is_sunday_l432_43236

def is_leap_year (year : ℕ) : Bool :=
  year % 4 = 0 && (year % 100 ≠ 0 || year % 400 = 0)

def days_in_month (year : ℕ) (month : ℕ) : ℕ :=
  if month = 2 then
    if is_leap_year year then 29 else 28
  else if month ∈ [4, 6, 9, 11] then 30
  else 31

def days_between (start_year start_month start_day : ℕ) (end_year end_month end_day : ℕ) : ℕ :=
  sorry

theorem june_1_2014_is_sunday :
  let start_date := (2013, 12, 31)
  let end_date := (2014, 6, 1)
  let start_day_of_week := 2  -- Tuesday
  let days_passed := days_between start_date.1 start_date.2.1 start_date.2.2 end_date.1 end_date.2.1 end_date.2.2
  (start_day_of_week + days_passed) % 7 = 0 :=
by sorry

end june_1_2014_is_sunday_l432_43236


namespace ten_spheres_melted_l432_43285

/-- The radius of a sphere formed by melting multiple smaller spheres -/
def large_sphere_radius (n : ℕ) (r : ℝ) : ℝ :=
  (n * r ^ 3) ^ (1/3)

/-- Theorem: The radius of a sphere formed by melting 10 smaller spheres 
    with radius 3 inches is equal to the cube root of 270 inches -/
theorem ten_spheres_melted (n : ℕ) (r : ℝ) : 
  n = 10 ∧ r = 3 → large_sphere_radius n r = 270 ^ (1/3) := by
  sorry

end ten_spheres_melted_l432_43285


namespace water_drinking_ratio_l432_43223

/-- Proof of the water drinking ratio problem -/
theorem water_drinking_ratio :
  let morning_water : ℝ := 1.5
  let total_water : ℝ := 6
  let afternoon_water : ℝ := total_water - morning_water
  afternoon_water / morning_water = 3 := by
  sorry

end water_drinking_ratio_l432_43223


namespace surface_area_ratio_l432_43287

/-- The ratio of the surface area of a cube to the surface area of a rectangular solid
    with dimensions 2L, 3W, and 4H, where L, W, H are the cube's dimensions. -/
theorem surface_area_ratio (s : ℝ) (h : s > 0) : 
  (6 * s^2) / (2 * (2*s) * (3*s) + 2 * (2*s) * (4*s) + 2 * (3*s) * (4*s)) = 3 / 26 := by
  sorry

#check surface_area_ratio

end surface_area_ratio_l432_43287


namespace planet_coloring_l432_43232

/-- Given 3 people coloring planets with 24 total colors, prove each person uses 8 colors --/
theorem planet_coloring (total_colors : ℕ) (num_people : ℕ) (h1 : total_colors = 24) (h2 : num_people = 3) :
  total_colors / num_people = 8 := by
  sorry

end planet_coloring_l432_43232


namespace fox_bridge_crossing_fox_initial_money_unique_l432_43240

/-- The function that doubles the money and subtracts the toll -/
def f (x : ℝ) : ℝ := 2 * x - 40

/-- Theorem stating that applying f three times to 35 results in 0 -/
theorem fox_bridge_crossing :
  f (f (f 35)) = 0 := by sorry

/-- Theorem proving that 35 is the only initial value that results in 0 after three crossings -/
theorem fox_initial_money_unique (x : ℝ) :
  f (f (f x)) = 0 → x = 35 := by sorry

end fox_bridge_crossing_fox_initial_money_unique_l432_43240


namespace zero_lite_soda_bottles_l432_43269

/-- The number of bottles of lite soda in a grocery store -/
def lite_soda_bottles (regular_soda diet_soda total_regular_and_diet : ℕ) : ℕ :=
  total_regular_and_diet - (regular_soda + diet_soda)

/-- Theorem: The number of lite soda bottles is 0 -/
theorem zero_lite_soda_bottles :
  lite_soda_bottles 49 40 89 = 0 := by
  sorry

end zero_lite_soda_bottles_l432_43269


namespace price_change_effect_l432_43296

theorem price_change_effect (a : ℝ) (h : a > 0) : a * 1.02 * 0.98 < a := by
  sorry

end price_change_effect_l432_43296


namespace mean_of_added_numbers_l432_43234

theorem mean_of_added_numbers (original_count : ℕ) (original_mean : ℚ) 
  (new_count : ℕ) (new_mean : ℚ) (x y z : ℚ) : 
  original_count = 7 →
  original_mean = 40 →
  new_count = original_count + 3 →
  new_mean = 50 →
  (original_count * original_mean + x + y + z) / new_count = new_mean →
  (x + y + z) / 3 = 220 / 3 := by
  sorry

end mean_of_added_numbers_l432_43234


namespace smallest_c_value_l432_43202

theorem smallest_c_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : ∀ x, a * Real.sin (b * x + c) ≤ a * Real.sin (b * (-π/4) + c))
  (h5 : a = 3) :
  c ≥ 3 * π / 4 := by
  sorry

end smallest_c_value_l432_43202


namespace area_of_specific_triangle_l432_43264

/-- Configuration of hexagons with a central hexagon of side length 2,
    surrounded by hexagons of side length 2 and 1 -/
structure HexagonConfiguration where
  centralHexagonSide : ℝ
  firstLevelSide : ℝ
  secondLevelSide : ℝ

/-- The triangle formed by connecting centers of three specific hexagons
    at the second surrounding level -/
def TriangleAtSecondLevel (config : HexagonConfiguration) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a triangle -/
def triangleArea (triangle : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem area_of_specific_triangle (config : HexagonConfiguration) 
  (h1 : config.centralHexagonSide = 2)
  (h2 : config.firstLevelSide = 2)
  (h3 : config.secondLevelSide = 1) :
  triangleArea (TriangleAtSecondLevel config) = 48 * Real.sqrt 3 :=
sorry

end area_of_specific_triangle_l432_43264


namespace gardener_hourly_rate_l432_43206

/-- Gardening project cost calculation -/
theorem gardener_hourly_rate 
  (num_rose_bushes : ℕ) 
  (cost_per_rose_bush : ℚ) 
  (hours_per_day : ℕ) 
  (num_days : ℕ) 
  (soil_volume : ℕ) 
  (cost_per_cubic_foot : ℚ) 
  (total_project_cost : ℚ) : 
  num_rose_bushes = 20 →
  cost_per_rose_bush = 150 →
  hours_per_day = 5 →
  num_days = 4 →
  soil_volume = 100 →
  cost_per_cubic_foot = 5 →
  total_project_cost = 4100 →
  (total_project_cost - (num_rose_bushes * cost_per_rose_bush + soil_volume * cost_per_cubic_foot)) / (hours_per_day * num_days) = 30 :=
by
  sorry


end gardener_hourly_rate_l432_43206


namespace investment_ratio_from_profit_ratio_and_time_l432_43294

/-- Given two partners with investments and profits, prove their investment ratio -/
theorem investment_ratio_from_profit_ratio_and_time (p q : ℝ) (h1 : p > 0) (h2 : q > 0) :
  (p * 20) / (q * 40) = 7 / 10 → p / q = 7 / 5 := by
  sorry

end investment_ratio_from_profit_ratio_and_time_l432_43294


namespace butterfly_collection_l432_43239

theorem butterfly_collection (total : ℕ) (black : ℕ) : 
  total = 19 → 
  black = 10 → 
  ∃ (blue yellow : ℕ), 
    blue = 2 * yellow ∧ 
    blue + yellow + black = total ∧ 
    blue = 6 := by
  sorry

end butterfly_collection_l432_43239


namespace max_cubes_in_box_l432_43216

/-- The volume of a rectangular box -/
def box_volume (length width height : ℕ) : ℕ :=
  length * width * height

/-- The volume of a cube -/
def cube_volume (side : ℕ) : ℕ :=
  side ^ 3

/-- The maximum number of cubes that can fit in a box -/
def max_cubes (box_length box_width box_height cube_side : ℕ) : ℕ :=
  (box_volume box_length box_width box_height) / (cube_volume cube_side)

theorem max_cubes_in_box :
  max_cubes 8 9 12 3 = 32 :=
by sorry

end max_cubes_in_box_l432_43216


namespace regular_polygon_radius_inequality_l432_43214

/-- For a regular polygon with n sides, n ≥ 3, the circumradius R is at most twice the inradius r. -/
theorem regular_polygon_radius_inequality (n : ℕ) (r R : ℝ) 
  (h_n : n ≥ 3) 
  (h_r : r > 0) 
  (h_R : R > 0) 
  (h_relation : r / R = Real.cos (π / n)) : 
  R ≤ 2 * r := by
  sorry

end regular_polygon_radius_inequality_l432_43214


namespace parabola_one_x_intercept_l432_43212

/-- A parabola defined by x = -3y^2 + 2y + 2 has exactly one x-intercept. -/
theorem parabola_one_x_intercept :
  ∃! x : ℝ, ∃ y : ℝ, x = -3 * y^2 + 2 * y + 2 ∧ y = 0 :=
by sorry

end parabola_one_x_intercept_l432_43212


namespace ceiling_neg_sqrt_64_over_9_l432_43261

theorem ceiling_neg_sqrt_64_over_9 : 
  ⌈-Real.sqrt (64/9)⌉ = -2 := by sorry

end ceiling_neg_sqrt_64_over_9_l432_43261


namespace product_remainder_is_one_l432_43252

def sequence1 : List Nat := List.range 10 |>.map (fun n => 3 + 10 * n)
def sequence2 : List Nat := List.range 10 |>.map (fun n => 7 + 10 * n)

theorem product_remainder_is_one :
  (sequence1.prod * sequence2.prod) % 8 = 1 := by
  sorry

end product_remainder_is_one_l432_43252


namespace five_letter_words_with_consonant_l432_43286

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Finset Char := {'B', 'C', 'D', 'F'}
def vowels : Finset Char := {'A', 'E'}

def word_length : Nat := 5

theorem five_letter_words_with_consonant :
  (alphabet.card ^ word_length) - (vowels.card ^ word_length) = 7744 :=
by sorry

end five_letter_words_with_consonant_l432_43286


namespace two_roots_condition_l432_43219

open Real

theorem two_roots_condition (k : ℝ) : 
  (∃ x y, x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1) ∧ 
          y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1) ∧ 
          x ≠ y ∧ 
          x * log x - k * x + 1 = 0 ∧ 
          y * log y - k * y + 1 = 0) ↔ 
  k ∈ Set.Ioo 1 (1 + 1/Real.exp 1) :=
sorry

end two_roots_condition_l432_43219


namespace thirteen_step_staircase_l432_43222

/-- 
Represents a staircase where each step is made of toothpicks following an arithmetic sequence.
The first step uses 3 toothpicks, and each subsequent step uses 2 more toothpicks than the previous one.
-/
def Staircase (n : ℕ) : ℕ := n * (n + 2)

/-- A staircase with 5 steps uses 55 toothpicks -/
axiom five_step_staircase : Staircase 5 = 55

/-- Theorem: A staircase with 13 steps uses 210 toothpicks -/
theorem thirteen_step_staircase : Staircase 13 = 210 := by
  sorry

end thirteen_step_staircase_l432_43222


namespace polynomial_value_theorem_l432_43217

theorem polynomial_value_theorem (P : ℤ → ℤ) : 
  (∃ a b c d e : ℤ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ 
                     b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ 
                     c ≠ d ∧ c ≠ e ∧ 
                     d ≠ e ∧
                     P a = 5 ∧ P b = 5 ∧ P c = 5 ∧ P d = 5 ∧ P e = 5) →
  (∀ x : ℤ, P x ≠ 9) :=
by sorry

end polynomial_value_theorem_l432_43217


namespace equation_representations_l432_43224

-- Define the equations
def equation1 (x y : ℝ) : Prop := x * (x^2 + y^2 - 4) = 0
def equation2 (x y : ℝ) : Prop := x^2 + (x^2 + y^2 - 4)^2 = 0

-- Define what it means for an equation to represent a line and a circle
def represents_line_and_circle (f : ℝ → ℝ → Prop) : Prop :=
  (∃ (a : ℝ), ∀ y, f a y) ∧ 
  (∃ (h k r : ℝ), ∀ x y, f x y ↔ (x - h)^2 + (y - k)^2 = r^2)

-- Define what it means for an equation to represent two points
def represents_two_points (f : ℝ → ℝ → Prop) : Prop :=
  ∃ (x1 y1 x2 y2 : ℝ), x1 ≠ x2 ∨ y1 ≠ y2 ∧ 
    (∀ x y, f x y ↔ (x = x1 ∧ y = y1) ∨ (x = x2 ∧ y = y2))

-- State the theorem
theorem equation_representations : 
  represents_line_and_circle equation1 ∧ represents_two_points equation2 := by
  sorry

end equation_representations_l432_43224


namespace polynomial_monomial_degree_l432_43282

/-- Given a sixth-degree polynomial and a monomial, prove the values of m and n -/
theorem polynomial_monomial_degree (m n : ℕ) : 
  (2 + (m + 1) = 6) ∧ (2*n + (5 - m) = 6) → m = 3 ∧ n = 2 := by
  sorry

end polynomial_monomial_degree_l432_43282


namespace joes_honey_purchase_l432_43220

theorem joes_honey_purchase (orange_price : ℚ) (juice_price : ℚ) (honey_price : ℚ) 
  (plant_price : ℚ) (total_spent : ℚ) (orange_count : ℕ) (juice_count : ℕ) 
  (plant_count : ℕ) :
  orange_price = 9/2 →
  juice_price = 1/2 →
  honey_price = 5 →
  plant_price = 9 →
  total_spent = 68 →
  orange_count = 3 →
  juice_count = 7 →
  plant_count = 4 →
  (total_spent - (orange_price * orange_count + juice_price * juice_count + 
    plant_price * plant_count)) / honey_price = 3 := by
  sorry

end joes_honey_purchase_l432_43220


namespace increase_by_percentage_increase_800_by_110_percent_l432_43244

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) :
  initial * (1 + percentage / 100) = initial + initial * (percentage / 100) := by
  sorry

theorem increase_800_by_110_percent :
  800 * (1 + 110 / 100) = 1680 := by
  sorry

end increase_by_percentage_increase_800_by_110_percent_l432_43244


namespace exam_score_calculation_l432_43254

theorem exam_score_calculation (total_questions : ℕ) (correct_score : ℕ) (total_score : ℕ) (correct_answers : ℕ) :
  total_questions = 75 →
  correct_score = 4 →
  total_score = 125 →
  correct_answers = 40 →
  (total_questions - correct_answers) * (correct_score - (correct_score * correct_answers - total_score) / (total_questions - correct_answers)) = total_score :=
by sorry

end exam_score_calculation_l432_43254


namespace inequality_proof_l432_43207

theorem inequality_proof (x a b c : ℝ) 
  (h1 : x ≠ a) (h2 : x ≠ b) (h3 : x ≠ c) 
  (h4 : (a < c ∧ c < b) ∨ (b < c ∧ c < a)) 
  (h5 : (x - a) * (x - b) * (x - c) > 0) : 
  1 / (x - a) + 1 / (x - b) > 1 / (x - c) := by
  sorry

end inequality_proof_l432_43207


namespace toy_cars_in_first_box_l432_43290

theorem toy_cars_in_first_box 
  (total_boxes : Nat)
  (total_cars : Nat)
  (cars_in_second : Nat)
  (cars_in_third : Nat)
  (h1 : total_boxes = 3)
  (h2 : total_cars = 71)
  (h3 : cars_in_second = 31)
  (h4 : cars_in_third = 19) :
  total_cars - cars_in_second - cars_in_third = 21 :=
by sorry

end toy_cars_in_first_box_l432_43290


namespace date_book_cost_date_book_cost_value_l432_43277

/-- Given the conditions of a real estate salesperson's promotional item purchase,
    prove that the cost of each date book is $0.375. -/
theorem date_book_cost (total_items : ℕ) (calendars : ℕ) (date_books : ℕ) 
                       (calendar_cost : ℚ) (total_spent : ℚ) : ℚ :=
  let date_book_cost := (total_spent - (calendar_cost * calendars)) / date_books
  by
    have h1 : total_items = calendars + date_books := by sorry
    have h2 : total_items = 500 := by sorry
    have h3 : calendars = 300 := by sorry
    have h4 : date_books = 200 := by sorry
    have h5 : calendar_cost = 3/4 := by sorry
    have h6 : total_spent = 300 := by sorry
    
    -- Prove that date_book_cost = 3/8
    sorry

#eval (300 : ℚ) - (3/4 * 300)
#eval ((300 : ℚ) - (3/4 * 300)) / 200

/-- The cost of each date book is $0.375. -/
theorem date_book_cost_value : 
  date_book_cost 500 300 200 (3/4) 300 = 3/8 := by sorry

end date_book_cost_date_book_cost_value_l432_43277


namespace angle_in_fourth_quadrant_l432_43253

def angle : Int := -1120

def is_coterminal (a b : Int) : Prop :=
  ∃ k : Int, a = b + k * 360

def in_fourth_quadrant (a : Int) : Prop :=
  ∃ b : Int, is_coterminal a b ∧ 270 ≤ b ∧ b < 360

theorem angle_in_fourth_quadrant : in_fourth_quadrant angle := by
  sorry

end angle_in_fourth_quadrant_l432_43253


namespace jolene_bicycle_purchase_l432_43204

structure Income where
  babysitting : Nat
  babysittingRate : Nat
  carWashing : Nat
  carWashingRate : Nat
  dogWalking : Nat
  dogWalkingRate : Nat
  cashGift : Nat

structure BicycleOption where
  price : Nat
  discount : Nat

def calculateTotalIncome (income : Income) : Nat :=
  income.babysitting * income.babysittingRate +
  income.carWashing * income.carWashingRate +
  income.dogWalking * income.dogWalkingRate +
  income.cashGift

def calculateDiscountedPrice (option : BicycleOption) : Nat :=
  option.price - (option.price * option.discount / 100)

def canAfford (income : Nat) (price : Nat) : Prop :=
  income ≥ price

theorem jolene_bicycle_purchase (income : Income)
  (optionA optionB optionC : BicycleOption) :
  income.babysitting = 4 ∧
  income.babysittingRate = 30 ∧
  income.carWashing = 5 ∧
  income.carWashingRate = 12 ∧
  income.dogWalking = 3 ∧
  income.dogWalkingRate = 15 ∧
  income.cashGift = 40 ∧
  optionA.price = 250 ∧
  optionA.discount = 0 ∧
  optionB.price = 300 ∧
  optionB.discount = 10 ∧
  optionC.price = 350 ∧
  optionC.discount = 15 →
  canAfford (calculateTotalIncome income) (calculateDiscountedPrice optionA) ∧
  calculateTotalIncome income - calculateDiscountedPrice optionA = 15 :=
by sorry


end jolene_bicycle_purchase_l432_43204


namespace sector_arc_length_l432_43278

/-- Given a circular sector with perimeter 12 and central angle 4 radians,
    the length of its arc is 8. -/
theorem sector_arc_length (p : ℝ) (θ : ℝ) (l : ℝ) (r : ℝ) :
  p = 12 →  -- perimeter of the sector
  θ = 4 →   -- central angle in radians
  p = l + 2 * r →  -- perimeter formula for a sector
  l = θ * r →  -- arc length formula
  l = 8 :=
by sorry

end sector_arc_length_l432_43278


namespace range_of_a_l432_43258

/-- Given sets A and B, prove that if A ∪ B = A, then 0 < a ≤ 9/5 -/
theorem range_of_a (a : ℝ) : 
  let A : Set ℝ := { x | 0 < x ∧ x ≤ 3 }
  let B : Set ℝ := { x | x^2 - 2*a*x + a ≤ 0 }
  (A ∪ B = A) → (0 < a ∧ a ≤ 9/5) := by
sorry

end range_of_a_l432_43258


namespace pyramid_volume_l432_43267

/-- The volume of a pyramid with a square base of side length 10 and edges of length 15 from apex to base corners is 500√7 / 3 -/
theorem pyramid_volume : 
  ∀ (base_side edge_length : ℝ) (volume : ℝ),
  base_side = 10 →
  edge_length = 15 →
  volume = (1/3) * base_side^2 * (edge_length^2 - (base_side^2/2))^(1/2) →
  volume = 500 * Real.sqrt 7 / 3 :=
by
  sorry

end pyramid_volume_l432_43267


namespace triangle_value_l432_43243

theorem triangle_value (q : ℤ) (h1 : ∃ triangle : ℤ, triangle + q = 59) 
  (h2 : ∃ triangle : ℤ, (triangle + q) + q = 106) : 
  ∃ triangle : ℤ, triangle = 12 := by
  sorry

end triangle_value_l432_43243


namespace sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l432_43279

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m : ℝ) : Prop :=
  2 * (m + 1) * (m - 3) + 2 * (m - 3) = 0

/-- m = 3 is a sufficient condition for the lines to be perpendicular -/
theorem sufficient_condition : perpendicular 3 := by sorry

/-- m = 3 is not a necessary condition for the lines to be perpendicular -/
theorem not_necessary_condition : ∃ m ≠ 3, perpendicular m := by sorry

/-- m = 3 is a sufficient but not necessary condition for the lines to be perpendicular -/
theorem sufficient_but_not_necessary :
  (perpendicular 3) ∧ (∃ m ≠ 3, perpendicular m) := by sorry

end sufficient_condition_not_necessary_condition_sufficient_but_not_necessary_l432_43279


namespace geometric_sequence_product_l432_43299

theorem geometric_sequence_product (a b : ℝ) : 
  2 < a ∧ a < b ∧ b < 16 ∧ 
  (∃ r : ℝ, r > 0 ∧ a = 2 * r ∧ b = 2 * r^2 ∧ 16 = 2 * r^3) →
  a * b = 32 := by
sorry

end geometric_sequence_product_l432_43299


namespace constant_term_expansion_l432_43209

theorem constant_term_expansion : 
  let f (x : ℝ) := (x^2 + 2) * (x - 1/x)^6
  ∃ (g : ℝ → ℝ), (∀ x ≠ 0, f x = g x) ∧ g 0 = -25 :=
by sorry

end constant_term_expansion_l432_43209


namespace rectangle_area_l432_43245

theorem rectangle_area (square_area : ℝ) (rectangle_width : ℝ) (rectangle_length : ℝ) : 
  square_area = 36 →
  rectangle_width = Real.sqrt square_area →
  rectangle_length = 3 * rectangle_width →
  rectangle_width * rectangle_length = 108 := by
sorry

end rectangle_area_l432_43245


namespace isosceles_trapezoid_area_l432_43263

/-- Given a circle with radius 5 and an isosceles trapezoid circumscribed around it,
    where the distance between the points of tangency of its lateral sides is 8,
    prove that the area of the trapezoid is 125. -/
theorem isosceles_trapezoid_area (r : ℝ) (d : ℝ) (A : ℝ) :
  r = 5 →
  d = 8 →
  A = (5 * d) * 2.5 →
  A = 125 :=
by sorry

end isosceles_trapezoid_area_l432_43263


namespace factorization_x3y_minus_xy_l432_43208

theorem factorization_x3y_minus_xy (x y : ℝ) : x^3*y - x*y = x*y*(x - 1)*(x + 1) := by
  sorry

end factorization_x3y_minus_xy_l432_43208


namespace rectangular_to_polar_equivalence_l432_43225

/-- Given a curve C in the xy-plane, prove that its rectangular coordinate equation
    x^2 + y^2 - 2x = 0 is equivalent to the polar coordinate equation ρ = 2cosθ. -/
theorem rectangular_to_polar_equivalence :
  ∀ (x y ρ θ : ℝ),
  (x = ρ * Real.cos θ) →
  (y = ρ * Real.sin θ) →
  (x^2 + y^2 - 2*x = 0) ↔ (ρ = 2 * Real.cos θ) :=
by sorry

end rectangular_to_polar_equivalence_l432_43225


namespace no_solutions_fibonacci_equation_l432_43201

def fib : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n+2) => fib (n+1) + fib n

theorem no_solutions_fibonacci_equation :
  ∀ n : ℕ, n * (fib n) * (fib (n+1)) ≠ (fib (n+2) - 1)^2 :=
by
  sorry

end no_solutions_fibonacci_equation_l432_43201


namespace andrea_lauren_bike_problem_l432_43275

/-- The problem of Andrea and Lauren biking towards each other --/
theorem andrea_lauren_bike_problem 
  (initial_distance : ℝ) 
  (andrea_speed_ratio : ℝ) 
  (initial_closing_rate : ℝ) 
  (lauren_stop_time : ℝ) 
  (h1 : initial_distance = 30) 
  (h2 : andrea_speed_ratio = 2) 
  (h3 : initial_closing_rate = 2) 
  (h4 : lauren_stop_time = 10) :
  ∃ (total_time : ℝ), 
    total_time = 17.5 ∧ 
    (∃ (lauren_speed : ℝ),
      lauren_speed > 0 ∧
      andrea_speed_ratio * lauren_speed + lauren_speed = initial_closing_rate ∧
      total_time = lauren_stop_time + (initial_distance - lauren_stop_time * initial_closing_rate) / (andrea_speed_ratio * lauren_speed)) :=
by sorry

end andrea_lauren_bike_problem_l432_43275


namespace absolute_value_inequality_l432_43259

theorem absolute_value_inequality (x : ℝ) : |x| ≠ 3 → x ≠ 3 := by sorry

end absolute_value_inequality_l432_43259


namespace f_monotone_range_l432_43247

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem f_monotone_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 < a ∧ a ≤ 3 := by
  sorry

end f_monotone_range_l432_43247


namespace domain_relation_l432_43210

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f(x+2)
def domain_f_plus_2 : Set ℝ := Set.Ioo (-2) 2

-- Theorem stating the relationship between the domains
theorem domain_relation (h : ∀ x, f (x + 2) ∈ domain_f_plus_2 ↔ x ∈ domain_f_plus_2) :
  ∀ x, f (x - 3) ∈ Set.Ioo 3 7 ↔ x ∈ Set.Ioo 3 7 :=
sorry

end domain_relation_l432_43210


namespace prob_less_than_8_prob_at_least_7_l432_43229

-- Define the probabilities
def p_9_or_above : ℝ := 0.56
def p_8 : ℝ := 0.22
def p_7 : ℝ := 0.12

-- Theorem for the first question
theorem prob_less_than_8 : 1 - p_9_or_above - p_8 = 0.22 := by sorry

-- Theorem for the second question
theorem prob_at_least_7 : p_9_or_above + p_8 + p_7 = 0.9 := by sorry

end prob_less_than_8_prob_at_least_7_l432_43229


namespace darnells_average_yards_is_11_l432_43265

/-- Calculates Darnell's average yards rushed per game given the total yards and other players' yards. -/
def darnells_average_yards (total_yards : ℕ) (malik_yards_per_game : ℕ) (josiah_yards_per_game : ℕ) (num_games : ℕ) : ℕ := 
  (total_yards - (malik_yards_per_game * num_games + josiah_yards_per_game * num_games)) / num_games

/-- Proves that Darnell's average yards rushed per game is 11 yards given the problem conditions. -/
theorem darnells_average_yards_is_11 : 
  darnells_average_yards 204 18 22 4 = 11 := by
  sorry

#eval darnells_average_yards 204 18 22 4

end darnells_average_yards_is_11_l432_43265


namespace coastal_analysis_uses_gis_l432_43283

-- Define the available technologies
inductive CoastalAnalysisTechnology
  | GPS
  | GIS
  | RemoteSensing
  | GeographicInformationTechnology

-- Define the properties of the analysis
structure CoastalAnalysis where
  involves_sea_level_changes : Bool
  available_technologies : List CoastalAnalysisTechnology

-- Define the main technology used for the analysis
def main_technology_for_coastal_analysis (analysis : CoastalAnalysis) : CoastalAnalysisTechnology :=
  CoastalAnalysisTechnology.GIS

-- Theorem statement
theorem coastal_analysis_uses_gis (analysis : CoastalAnalysis) 
  (h1 : analysis.involves_sea_level_changes = true)
  (h2 : analysis.available_technologies.length ≥ 2) :
  main_technology_for_coastal_analysis analysis = CoastalAnalysisTechnology.GIS := by
  sorry

end coastal_analysis_uses_gis_l432_43283


namespace inequality_proof_l432_43289

theorem inequality_proof (a b c : ℝ) 
  (h1 : 0 < a ∧ a < 1) 
  (h2 : 0 < b ∧ b < 1) 
  (h3 : 0 < c ∧ c < 1) 
  (h4 : a * b * c = Real.sqrt 3 / 9) : 
  a / (1 - a^2) + b / (1 - b^2) + c / (1 - c^2) ≥ 3 * Real.sqrt 3 / 2 := by
  sorry

end inequality_proof_l432_43289


namespace playground_area_l432_43256

theorem playground_area (perimeter width length : ℝ) : 
  perimeter = 120 →
  length = 3 * width →
  2 * length + 2 * width = perimeter →
  length * width = 675 :=
by
  sorry

end playground_area_l432_43256


namespace exists_k_divisible_by_power_of_three_l432_43280

theorem exists_k_divisible_by_power_of_three : 
  ∃ k : ℤ, (3 : ℤ)^2008 ∣ (k^3 - 36*k^2 + 51*k - 97) := by
  sorry

end exists_k_divisible_by_power_of_three_l432_43280


namespace fourth_test_score_l432_43288

theorem fourth_test_score (first_three_average : ℝ) (desired_increase : ℝ) : 
  first_three_average = 85 → 
  desired_increase = 2 → 
  (3 * first_three_average + 93) / 4 = first_three_average + desired_increase := by
sorry

end fourth_test_score_l432_43288


namespace soccer_ball_selling_price_l432_43274

theorem soccer_ball_selling_price 
  (num_balls : ℕ) 
  (cost_per_ball : ℚ) 
  (total_profit : ℚ) 
  (h1 : num_balls = 50)
  (h2 : cost_per_ball = 60)
  (h3 : total_profit = 1950) :
  (total_profit / num_balls + cost_per_ball : ℚ) = 99 := by
sorry

end soccer_ball_selling_price_l432_43274


namespace monday_is_42_l432_43228

/-- Represents the temperature on each day of the week --/
structure WeekTemperatures where
  monday : ℝ
  tuesday : ℝ
  wednesday : ℝ
  thursday : ℝ
  friday : ℝ

/-- The average temperature for Monday to Thursday is 48 degrees --/
def avg_mon_to_thu (w : WeekTemperatures) : Prop :=
  (w.monday + w.tuesday + w.wednesday + w.thursday) / 4 = 48

/-- The average temperature for Tuesday to Friday is 46 degrees --/
def avg_tue_to_fri (w : WeekTemperatures) : Prop :=
  (w.tuesday + w.wednesday + w.thursday + w.friday) / 4 = 46

/-- The temperature on Friday is 34 degrees --/
def friday_temp (w : WeekTemperatures) : Prop :=
  w.friday = 34

/-- Some day has a temperature of 42 degrees --/
def some_day_42 (w : WeekTemperatures) : Prop :=
  w.monday = 42 ∨ w.tuesday = 42 ∨ w.wednesday = 42 ∨ w.thursday = 42 ∨ w.friday = 42

theorem monday_is_42 (w : WeekTemperatures) 
  (h1 : avg_mon_to_thu w) 
  (h2 : avg_tue_to_fri w) 
  (h3 : friday_temp w) 
  (h4 : some_day_42 w) : 
  w.monday = 42 := by
  sorry

end monday_is_42_l432_43228


namespace child_cost_age_18_l432_43241

/-- Represents the cost of raising a child --/
structure ChildCost where
  initialYearlyCost : ℕ
  initialYears : ℕ
  laterYearlyCost : ℕ
  tuitionCost : ℕ
  totalCost : ℕ

/-- Calculates the age at which the child stops incurring yearly cost --/
def ageStopCost (c : ChildCost) : ℕ :=
  let initialCost := c.initialYears * c.initialYearlyCost
  let laterYears := (c.totalCost - initialCost - c.tuitionCost) / c.laterYearlyCost
  c.initialYears + laterYears

/-- Theorem stating that given the specific costs, the child stops incurring yearly cost at age 18 --/
theorem child_cost_age_18 :
  let c := ChildCost.mk 5000 8 10000 125000 265000
  ageStopCost c = 18 := by
  sorry

#eval ageStopCost (ChildCost.mk 5000 8 10000 125000 265000)

end child_cost_age_18_l432_43241


namespace bus_speed_problem_l432_43255

theorem bus_speed_problem (distance : ℝ) (speed_increase : ℝ) (time_reduction : ℝ) :
  distance = 660 ∧ 
  speed_increase = 5 ∧ 
  time_reduction = 1 →
  ∃ (v : ℝ), 
    v > 0 ∧
    distance / v - time_reduction = distance / (v + speed_increase) ∧
    v = 55 := by
  sorry

end bus_speed_problem_l432_43255


namespace fraction_zero_implies_x_negative_two_l432_43235

theorem fraction_zero_implies_x_negative_two (x : ℝ) : 
  (x^2 - 4) / (x^2 - 4*x + 4) = 0 → x = -2 := by
  sorry

end fraction_zero_implies_x_negative_two_l432_43235


namespace same_group_probability_l432_43271

/-- The probability that two randomly selected people from a group of 16 divided into two equal subgroups are from the same subgroup is 7/15. -/
theorem same_group_probability (n : ℕ) (h1 : n = 16) (h2 : n % 2 = 0) : 
  (Nat.choose (n / 2) 2 * 2) / Nat.choose n 2 = 7 / 15 := by
  sorry

end same_group_probability_l432_43271


namespace johnny_savings_l432_43233

/-- The amount Johnny saved in September -/
def september_savings : ℕ := 30

/-- The amount Johnny saved in October -/
def october_savings : ℕ := 49

/-- The amount Johnny saved in November -/
def november_savings : ℕ := 46

/-- The amount Johnny spent on a video game -/
def video_game_cost : ℕ := 58

/-- The amount Johnny has left after all transactions -/
def remaining_money : ℕ := 67

theorem johnny_savings : 
  september_savings + october_savings + november_savings - video_game_cost = remaining_money := by
  sorry

end johnny_savings_l432_43233


namespace smallest_three_digit_palindrome_times_103_not_six_digit_palindrome_l432_43284

/-- Checks if a number is a three-digit palindrome -/
def isThreeDigitPalindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ (n / 100 = n % 10)

/-- Checks if a number is a six-digit palindrome -/
def isSixDigitPalindrome (n : ℕ) : Prop :=
  100000 ≤ n ∧ n ≤ 999999 ∧ 
  (n / 100000 = n % 10) ∧ 
  ((n / 10000) % 10 = (n / 10) % 10) ∧
  ((n / 1000) % 10 = (n / 100) % 10)

/-- The main theorem -/
theorem smallest_three_digit_palindrome_times_103_not_six_digit_palindrome :
  isThreeDigitPalindrome 131 ∧
  ¬(isSixDigitPalindrome (131 * 103)) ∧
  ∀ n : ℕ, isThreeDigitPalindrome n ∧ n < 131 → isSixDigitPalindrome (n * 103) :=
by sorry

end smallest_three_digit_palindrome_times_103_not_six_digit_palindrome_l432_43284
