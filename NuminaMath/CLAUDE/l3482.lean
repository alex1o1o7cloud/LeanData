import Mathlib

namespace NUMINAMATH_CALUDE_f_min_value_inequality_proof_l3482_348296

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Theorem for the minimum value of f(x)
theorem f_min_value : ∃ (a : ℝ), ∀ (x : ℝ), f x ≥ a ∧ ∃ (y : ℝ), f y = a :=
sorry

-- Theorem for the inequality
theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m^3 + n^3 = 2) : m + n ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_f_min_value_inequality_proof_l3482_348296


namespace NUMINAMATH_CALUDE_greatest_n_value_l3482_348205

theorem greatest_n_value (n : ℤ) (h : 102 * n^2 ≤ 8100) : n ≤ 8 ∧ ∃ (m : ℤ), m = 8 ∧ 102 * m^2 ≤ 8100 := by
  sorry

end NUMINAMATH_CALUDE_greatest_n_value_l3482_348205


namespace NUMINAMATH_CALUDE_soccer_ball_cost_l3482_348217

theorem soccer_ball_cost (total_cost : ℕ) (num_soccer_balls : ℕ) (num_volleyballs : ℕ) (volleyball_cost : ℕ) :
  total_cost = 980 ∧ num_soccer_balls = 5 ∧ num_volleyballs = 4 ∧ volleyball_cost = 65 →
  ∃ (soccer_ball_cost : ℕ), soccer_ball_cost = 144 ∧ 
    total_cost = num_soccer_balls * soccer_ball_cost + num_volleyballs * volleyball_cost :=
by
  sorry

end NUMINAMATH_CALUDE_soccer_ball_cost_l3482_348217


namespace NUMINAMATH_CALUDE_mikes_shirt_cost_l3482_348297

/-- The cost of Mike's shirt given the profit sharing between Mike and Johnson -/
theorem mikes_shirt_cost (total_profit : ℚ) (mikes_share johnson_share : ℚ) : 
  mikes_share / johnson_share = 2 / 5 →
  johnson_share = 2500 →
  mikes_share - 800 = 200 :=
by sorry

end NUMINAMATH_CALUDE_mikes_shirt_cost_l3482_348297


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_city_stratified_sampling_l3482_348201

/-- Represents the types of schools in the city -/
inductive SchoolType
  | University
  | MiddleSchool
  | PrimarySchool

/-- Represents the distribution of schools in the city -/
structure SchoolDistribution where
  total : ℕ
  universities : ℕ
  middleSchools : ℕ
  primarySchools : ℕ

/-- Represents the sample size and distribution in stratified sampling -/
structure StratifiedSample where
  sampleSize : ℕ
  universitiesSample : ℕ
  middleSchoolsSample : ℕ
  primarySchoolsSample : ℕ

def citySchools : SchoolDistribution :=
  { total := 500
  , universities := 10
  , middleSchools := 200
  , primarySchools := 290 }

def sampleSize : ℕ := 50

theorem stratified_sampling_theorem (d : SchoolDistribution) (s : ℕ) :
  d.total = d.universities + d.middleSchools + d.primarySchools →
  s ≤ d.total →
  ∃ (sample : StratifiedSample),
    sample.sampleSize = s ∧
    sample.universitiesSample = (s * d.universities) / d.total ∧
    sample.middleSchoolsSample = (s * d.middleSchools) / d.total ∧
    sample.primarySchoolsSample = (s * d.primarySchools) / d.total ∧
    sample.sampleSize = sample.universitiesSample + sample.middleSchoolsSample + sample.primarySchoolsSample :=
by sorry

theorem city_stratified_sampling :
  ∃ (sample : StratifiedSample),
    sample.sampleSize = sampleSize ∧
    sample.universitiesSample = 1 ∧
    sample.middleSchoolsSample = 20 ∧
    sample.primarySchoolsSample = 29 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_city_stratified_sampling_l3482_348201


namespace NUMINAMATH_CALUDE_area_of_specific_quadrilateral_l3482_348256

/-- Represents a convex quadrilateral ABCD with specific side lengths and a right angle -/
structure ConvexQuadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  angle_CDA : ℝ
  convex : AB > 0 ∧ BC > 0 ∧ CD > 0 ∧ DA > 0
  right_angle : angle_CDA = 90

/-- The area of the specific convex quadrilateral ABCD is 62 -/
theorem area_of_specific_quadrilateral (ABCD : ConvexQuadrilateral)
    (h1 : ABCD.AB = 8)
    (h2 : ABCD.BC = 4)
    (h3 : ABCD.CD = 10)
    (h4 : ABCD.DA = 10) :
    Real.sqrt 0 + 62 * Real.sqrt 1 = 62 := by
  sorry

end NUMINAMATH_CALUDE_area_of_specific_quadrilateral_l3482_348256


namespace NUMINAMATH_CALUDE_girls_with_tablets_l3482_348202

/-- Proves that the number of girls who brought tablets is 13 -/
theorem girls_with_tablets (total_boys : ℕ) (students_with_tablets : ℕ) (boys_with_tablets : ℕ)
  (h1 : total_boys = 20)
  (h2 : students_with_tablets = 24)
  (h3 : boys_with_tablets = 11) :
  students_with_tablets - boys_with_tablets = 13 := by
  sorry

end NUMINAMATH_CALUDE_girls_with_tablets_l3482_348202


namespace NUMINAMATH_CALUDE_profit_maximum_l3482_348236

/-- Represents the daily sales profit function -/
def profit (x : ℕ) : ℝ := -10 * (x : ℝ)^2 + 90 * (x : ℝ) + 1900

/-- The maximum daily profit -/
def max_profit : ℝ := 2100

theorem profit_maximum :
  ∃ x : ℕ, profit x = max_profit ∧
  ∀ y : ℕ, profit y ≤ max_profit :=
sorry

end NUMINAMATH_CALUDE_profit_maximum_l3482_348236


namespace NUMINAMATH_CALUDE_tenth_diagram_shading_l3482_348269

/-- Represents a square grid with a specific shading pattern -/
structure ShadedGrid (n : ℕ) where
  size : ℕ
  shaded_squares : ℕ
  h_size : size = n * n
  h_shaded : shaded_squares = (n - 1) * (n / 2) + n

/-- The fraction of shaded squares in the grid -/
def shaded_fraction (grid : ShadedGrid n) : ℚ :=
  grid.shaded_squares / grid.size

theorem tenth_diagram_shading :
  ∃ (grid : ShadedGrid 10), shaded_fraction grid = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_tenth_diagram_shading_l3482_348269


namespace NUMINAMATH_CALUDE_matrix_equation_solution_l3482_348240

theorem matrix_equation_solution :
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, -3; 4, -1]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![-12, 5; 8, -3]
  let M : Matrix (Fin 2) (Fin 2) ℝ := !![-0.8, -2.6; -2.0, 1.8]
  M * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_solution_l3482_348240


namespace NUMINAMATH_CALUDE_softball_opponent_score_l3482_348210

theorem softball_opponent_score :
  let team_scores : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let games_lost_by_one : Nat := 5
  let other_games_score_ratio : Nat := 2
  let opponent_scores : List Nat := 
    team_scores.map (fun score => 
      if score % 2 = 1 then score + 1
      else score / other_games_score_ratio)
  opponent_scores.sum = 45 := by
  sorry

end NUMINAMATH_CALUDE_softball_opponent_score_l3482_348210


namespace NUMINAMATH_CALUDE_equal_area_segment_property_l3482_348222

/-- Represents a trapezoid with the given properties -/
structure Trapezoid where
  b : ℝ  -- Length of the shorter base
  h : ℝ  -- Height of the trapezoid
  midline_ratio : (b + 75) / (b + 150) = 3 / 4  -- Area ratio condition for midline
  h_pos : h > 0
  b_pos : b > 0

/-- The length of the segment that divides the trapezoid into two equal areas -/
def equal_area_segment (t : Trapezoid) : ℝ :=
  225  -- This is the value of x we found in the solution

/-- The theorem to be proved -/
theorem equal_area_segment_property (t : Trapezoid) :
  ⌊(equal_area_segment t)^2 / 100⌋ = 506 := by
  sorry

end NUMINAMATH_CALUDE_equal_area_segment_property_l3482_348222


namespace NUMINAMATH_CALUDE_chicken_cost_problem_l3482_348271

/-- A problem about calculating the cost of chickens given various expenses --/
theorem chicken_cost_problem (land_acres : ℕ) (land_cost_per_acre : ℕ) 
  (house_cost : ℕ) (cow_count : ℕ) (cow_cost : ℕ) (chicken_count : ℕ) 
  (solar_install_hours : ℕ) (solar_install_rate : ℕ) (solar_equipment_cost : ℕ) 
  (total_cost : ℕ) : 
  land_acres = 30 →
  land_cost_per_acre = 20 →
  house_cost = 120000 →
  cow_count = 20 →
  cow_cost = 1000 →
  chicken_count = 100 →
  solar_install_hours = 6 →
  solar_install_rate = 100 →
  solar_equipment_cost = 6000 →
  total_cost = 147700 →
  (total_cost - (land_acres * land_cost_per_acre + house_cost + cow_count * cow_cost + 
    solar_install_hours * solar_install_rate + solar_equipment_cost)) / chicken_count = 5 :=
by sorry

end NUMINAMATH_CALUDE_chicken_cost_problem_l3482_348271


namespace NUMINAMATH_CALUDE_inequality_of_positive_reals_l3482_348253

theorem inequality_of_positive_reals (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^x * y^y * z^z ≥ (x*y*z)^((x+y+z)/3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_positive_reals_l3482_348253


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_quadratic_inequality_range_l3482_348251

-- Problem 1
theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x : ℝ, x^2 + a*x + b < 0 ↔ -1 < x ∧ x < 2) →
  a = 1 ∧ b = -2 :=
sorry

-- Problem 2
theorem quadratic_inequality_range (c : ℝ) :
  (∀ x : ℝ, x ≥ 1 → x^2 + 3*x - c > 0) →
  c < 4 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_quadratic_inequality_range_l3482_348251


namespace NUMINAMATH_CALUDE_min_extracted_tablets_l3482_348272

/-- Represents the contents of a medicine box -/
structure MedicineBox where
  tabletA : Nat
  tabletB : Nat

/-- Represents the minimum number of tablets extracted -/
structure ExtractedTablets where
  minA : Nat
  minB : Nat

/-- Given a medicine box with 10 tablets of each kind and a minimum extraction of 12 tablets,
    proves that the minimum number of tablets of each kind among the extracted is 2 for A and 1 for B -/
theorem min_extracted_tablets (box : MedicineBox) (min_extraction : Nat) :
  box.tabletA = 10 → box.tabletB = 10 → min_extraction = 12 →
  ∃ (extracted : ExtractedTablets),
    extracted.minA = 2 ∧ extracted.minB = 1 ∧
    extracted.minA + extracted.minB ≤ min_extraction ∧
    extracted.minA ≤ box.tabletA ∧ extracted.minB ≤ box.tabletB := by
  sorry

end NUMINAMATH_CALUDE_min_extracted_tablets_l3482_348272


namespace NUMINAMATH_CALUDE_minimum_value_of_expression_l3482_348235

theorem minimum_value_of_expression (x : ℝ) (h : x > 0) :
  x + 4 / x ≥ 4 ∧ (x + 4 / x = 4 ↔ x = 2) := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_of_expression_l3482_348235


namespace NUMINAMATH_CALUDE_long_tennis_players_l3482_348298

theorem long_tennis_players (total : ℕ) (football : ℕ) (both : ℕ) (neither : ℕ) :
  total = 36 →
  football = 26 →
  both = 17 →
  neither = 7 →
  ∃ (long_tennis : ℕ), long_tennis = 20 ∧ 
    total = football + long_tennis - both + neither :=
by sorry

end NUMINAMATH_CALUDE_long_tennis_players_l3482_348298


namespace NUMINAMATH_CALUDE_cosine_of_angle_l3482_348245

/-- Given two vectors a and b in ℝ², prove that the cosine of the angle between them is (3√10) / 10 -/
theorem cosine_of_angle (a b : ℝ × ℝ) (h1 : a = (3, 3)) (h2 : (2 • b) - a = (-1, 1)) :
  let θ := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  Real.cos θ = 3 * Real.sqrt 10 / 10 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_angle_l3482_348245


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_three_l3482_348260

theorem reciprocal_of_negative_three :
  (1 : ℝ) / (-3 : ℝ) = -1/3 := by sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_three_l3482_348260


namespace NUMINAMATH_CALUDE_lizzie_has_27_crayons_l3482_348282

def billie_crayons : ℕ := 18

def bobbie_crayons (billie : ℕ) : ℕ := 3 * billie

def lizzie_crayons (bobbie : ℕ) : ℕ := bobbie / 2

theorem lizzie_has_27_crayons :
  lizzie_crayons (bobbie_crayons billie_crayons) = 27 :=
by sorry

end NUMINAMATH_CALUDE_lizzie_has_27_crayons_l3482_348282


namespace NUMINAMATH_CALUDE_two_plus_three_eq_eight_is_proposition_l3482_348229

-- Define what a proposition is
def is_proposition (s : String) : Prop := ∃ (b : Bool), (s = "true" ∨ s = "false")

-- State the theorem
theorem two_plus_three_eq_eight_is_proposition :
  is_proposition "2+3=8" :=
sorry

end NUMINAMATH_CALUDE_two_plus_three_eq_eight_is_proposition_l3482_348229


namespace NUMINAMATH_CALUDE_complement_of_A_l3482_348239

def U : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | x^2 - 2*x - 3 > 0}

theorem complement_of_A : Set.compl A = Set.Icc (-1) 3 := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l3482_348239


namespace NUMINAMATH_CALUDE_well_depth_specific_well_depth_l3482_348206

/-- The depth of a cylindrical well given its diameter, cost per cubic meter, and total cost -/
theorem well_depth (diameter : ℝ) (cost_per_cubic_meter : ℝ) (total_cost : ℝ) : ℝ :=
  let radius := diameter / 2
  let volume := total_cost / cost_per_cubic_meter
  let depth := volume / (Real.pi * radius^2)
  depth

/-- The depth of a specific well with given parameters is approximately 14 meters -/
theorem specific_well_depth : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0000001 ∧ 
  |well_depth 3 16 1583.3626974092558 - 14| < ε :=
sorry

end NUMINAMATH_CALUDE_well_depth_specific_well_depth_l3482_348206


namespace NUMINAMATH_CALUDE_max_pies_without_ingredients_l3482_348277

theorem max_pies_without_ingredients (total_pies : ℕ) 
  (chocolate_pies marshmallow_pies cayenne_pies walnut_pies : ℕ) :
  total_pies = 48 →
  chocolate_pies ≥ 16 →
  marshmallow_pies = 24 →
  cayenne_pies = 36 →
  walnut_pies ≥ 6 →
  ∃ (pies_without_ingredients : ℕ),
    pies_without_ingredients ≤ 12 ∧
    pies_without_ingredients + chocolate_pies + marshmallow_pies + cayenne_pies + walnut_pies ≥ total_pies :=
by sorry

end NUMINAMATH_CALUDE_max_pies_without_ingredients_l3482_348277


namespace NUMINAMATH_CALUDE_divisibility_by_33_l3482_348267

def five_digit_number (n : ℕ) : ℕ := 70000 + 1000 * n + 933

theorem divisibility_by_33 (n : ℕ) : 
  n < 10 → (five_digit_number n % 33 = 0 ↔ n = 5) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_33_l3482_348267


namespace NUMINAMATH_CALUDE_paint_ratio_circular_signs_l3482_348220

theorem paint_ratio_circular_signs (d : ℝ) (h : d > 0) : 
  let D := 7 * d
  (π * (D / 2)^2) / (π * (d / 2)^2) = 49 := by sorry

end NUMINAMATH_CALUDE_paint_ratio_circular_signs_l3482_348220


namespace NUMINAMATH_CALUDE_parallel_condition_l3482_348281

/-- Two lines in the real plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Determine if two lines are parallel -/
def are_parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a ≠ 0 ∨ l1.b ≠ 0

/-- The lines l₁ and l₂ parameterized by m -/
def l1 (m : ℝ) : Line := ⟨1, 2*m, -1⟩
def l2 (m : ℝ) : Line := ⟨3*m+1, -m, -1⟩

/-- The statement to be proved -/
theorem parallel_condition :
  (∀ m : ℝ, are_parallel (l1 m) (l2 m) → m = -1/2 ∨ m = 0) ∧
  (∃ m : ℝ, m ≠ -1/2 ∧ m ≠ 0 ∧ ¬are_parallel (l1 m) (l2 m)) :=
sorry

end NUMINAMATH_CALUDE_parallel_condition_l3482_348281


namespace NUMINAMATH_CALUDE_task_completion_probability_l3482_348295

theorem task_completion_probability 
  (p1 : ℚ) (p2 : ℚ) 
  (h1 : p1 = 3 / 8) 
  (h2 : p2 = 3 / 5) : 
  p1 * (1 - p2) = 3 / 20 := by
sorry

end NUMINAMATH_CALUDE_task_completion_probability_l3482_348295


namespace NUMINAMATH_CALUDE_vector_midpoint_dot_product_l3482_348227

def problem (a b : ℝ × ℝ) : Prop :=
  let m : ℝ × ℝ := (4, 10)
  m = ((a.1 + b.1) / 2, (a.2 + b.2) / 2) ∧
  a.1 * b.1 + a.2 * b.2 = 10 →
  a.1^2 + a.2^2 + b.1^2 + b.2^2 = 444

theorem vector_midpoint_dot_product :
  ∀ a b : ℝ × ℝ, problem a b :=
by
  sorry

end NUMINAMATH_CALUDE_vector_midpoint_dot_product_l3482_348227


namespace NUMINAMATH_CALUDE_prime_remainder_30_l3482_348208

theorem prime_remainder_30 (p : ℕ) (h : Prime p) : 
  let r := p % 30
  Prime r ∨ r = 1 := by
sorry

end NUMINAMATH_CALUDE_prime_remainder_30_l3482_348208


namespace NUMINAMATH_CALUDE_fourth_month_sale_is_7230_l3482_348275

/-- Represents the sales data for a grocer over 6 months -/
structure GrocerSales where
  month1 : ℕ
  month2 : ℕ
  month3 : ℕ
  month5 : ℕ
  month6 : ℕ
  average : ℕ

/-- Calculates the sale in the fourth month given the sales data -/
def fourthMonthSale (sales : GrocerSales) : ℕ :=
  sales.average * 6 - (sales.month1 + sales.month2 + sales.month3 + sales.month5 + sales.month6)

/-- Theorem stating that the fourth month sale is 7230 given the provided sales data -/
theorem fourth_month_sale_is_7230 (sales : GrocerSales) 
  (h1 : sales.month1 = 6435)
  (h2 : sales.month2 = 6927)
  (h3 : sales.month3 = 6855)
  (h5 : sales.month5 = 6562)
  (h6 : sales.month6 = 6791)
  (ha : sales.average = 6800) :
  fourthMonthSale sales = 7230 := by
  sorry

#eval fourthMonthSale {
  month1 := 6435,
  month2 := 6927,
  month3 := 6855,
  month5 := 6562,
  month6 := 6791,
  average := 6800
}

end NUMINAMATH_CALUDE_fourth_month_sale_is_7230_l3482_348275


namespace NUMINAMATH_CALUDE_number_puzzle_l3482_348293

theorem number_puzzle (x : ℚ) : (x / 4) * 12 = 9 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l3482_348293


namespace NUMINAMATH_CALUDE_reflect_P_across_y_axis_l3482_348234

/-- Reflects a point across the y-axis in a 2D Cartesian coordinate system -/
def reflect_across_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, p.2)

/-- The original point P -/
def P : ℝ × ℝ := (-2, 3)

/-- Theorem stating that reflecting P(-2,3) across the y-axis results in (2,3) -/
theorem reflect_P_across_y_axis :
  reflect_across_y_axis P = (2, 3) := by
  sorry

end NUMINAMATH_CALUDE_reflect_P_across_y_axis_l3482_348234


namespace NUMINAMATH_CALUDE_xiao_jun_pictures_xiao_jun_pictures_proof_l3482_348283

theorem xiao_jun_pictures : ℕ → Prop :=
  fun original : ℕ =>
    let half := original / 2
    let given_away := half - 1
    let remaining := original - given_away
    remaining = 25 → original = 48

-- The proof is omitted
theorem xiao_jun_pictures_proof : xiao_jun_pictures 48 := by
  sorry

end NUMINAMATH_CALUDE_xiao_jun_pictures_xiao_jun_pictures_proof_l3482_348283


namespace NUMINAMATH_CALUDE_max_value_expression_l3482_348243

theorem max_value_expression (x : ℝ) :
  (Real.exp (2 * x) + Real.exp (-2 * x) + 1) / (Real.exp x + Real.exp (-x) + 2) ≤ 2 * (1 - Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_max_value_expression_l3482_348243


namespace NUMINAMATH_CALUDE_multiples_between_2000_and_3000_l3482_348247

def count_multiples (lower upper lcm : ℕ) : ℕ :=
  (upper / lcm) - ((lower - 1) / lcm)

theorem multiples_between_2000_and_3000 : count_multiples 2000 3000 72 = 14 := by
  sorry

end NUMINAMATH_CALUDE_multiples_between_2000_and_3000_l3482_348247


namespace NUMINAMATH_CALUDE_inverse_g_equals_two_l3482_348265

/-- Given nonzero constants a and b, and a function g defined as g(x) = 1 / (2ax + b),
    prove that the inverse of g evaluated at 1 / (4a + b) is equal to 2. -/
theorem inverse_g_equals_two (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let g := fun x => 1 / (2 * a * x + b)
  Function.invFun g (1 / (4 * a + b)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_inverse_g_equals_two_l3482_348265


namespace NUMINAMATH_CALUDE_six_valid_cuts_l3482_348216

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  vertex : Point3D
  base : (Point3D × Point3D × Point3D)

/-- Represents a plane -/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Represents an isosceles right triangle -/
structure IsoscelesRightTriangle where
  vertex1 : Point3D
  vertex2 : Point3D
  vertex3 : Point3D

/-- Function to check if a plane cuts a tetrahedron such that 
    the first projection is an isosceles right triangle -/
def validCut (t : Tetrahedron) (p : Plane) : Bool :=
  sorry

/-- Function to count the number of valid cutting planes -/
def countValidCuts (t : Tetrahedron) : Nat :=
  sorry

/-- Theorem stating that there are exactly 6 valid cutting planes -/
theorem six_valid_cuts (t : Tetrahedron) : 
  countValidCuts t = 6 := by sorry

end NUMINAMATH_CALUDE_six_valid_cuts_l3482_348216


namespace NUMINAMATH_CALUDE_triangle_side_length_l3482_348254

/-- Given a right-angled triangle XYZ where angle XZY is 30° and XZ = 12, prove XY = 12√3 -/
theorem triangle_side_length (X Y Z : ℝ × ℝ) :
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = ((X.1 - Z.1)^2 + (X.2 - Z.2)^2) + ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) →  -- right-angled triangle
  Real.cos (Real.arccos ((Y.1 - Z.1) * (X.1 - Z.1) + (Y.2 - Z.2) * (X.2 - Z.2)) / 
    (Real.sqrt ((Y.1 - Z.1)^2 + (Y.2 - Z.2)^2) * Real.sqrt ((X.1 - Z.1)^2 + (X.2 - Z.2)^2))) = 1/2 →  -- angle XZY is 30°
  (X.1 - Z.1)^2 + (X.2 - Z.2)^2 = 144 →  -- XZ = 12
  (X.1 - Y.1)^2 + (X.2 - Y.2)^2 = 432  -- XY = 12√3
  := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3482_348254


namespace NUMINAMATH_CALUDE_raviraj_journey_l3482_348261

def journey (initial_south distance_after_first_turn second_north final_west distance_to_home : ℝ) : Prop :=
  initial_south = 20 ∧
  second_north = 20 ∧
  final_west = 20 ∧
  distance_to_home = 30 ∧
  distance_after_first_turn + final_west = distance_to_home

theorem raviraj_journey :
  ∀ initial_south distance_after_first_turn second_north final_west distance_to_home,
    journey initial_south distance_after_first_turn second_north final_west distance_to_home →
    distance_after_first_turn = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_raviraj_journey_l3482_348261


namespace NUMINAMATH_CALUDE_integer_inequalities_result_l3482_348289

theorem integer_inequalities_result (n m : ℤ) 
  (h1 : 3*n - m < 5)
  (h2 : n + m > 26)
  (h3 : 3*m - 2*n < 46) :
  2*n + m = 36 := by
  sorry

end NUMINAMATH_CALUDE_integer_inequalities_result_l3482_348289


namespace NUMINAMATH_CALUDE_simple_interest_ratio_l3482_348278

/-- Simple interest calculation --/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

/-- The ratio of final amount to initial amount after simple interest --/
def final_to_initial_ratio (rate : ℝ) (time : ℝ) : ℝ :=
  1 + rate * time

theorem simple_interest_ratio :
  let rate : ℝ := 0.1
  let time : ℝ := 10
  final_to_initial_ratio rate time = 2 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_ratio_l3482_348278


namespace NUMINAMATH_CALUDE_expand_and_simplify_l3482_348230

theorem expand_and_simplify (a b : ℝ) : (3*a - b) * (-3*a - b) = b^2 - 9*a^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l3482_348230


namespace NUMINAMATH_CALUDE_square_greater_than_self_for_x_greater_than_one_l3482_348244

theorem square_greater_than_self_for_x_greater_than_one (x : ℝ) : x > 1 → x^2 > x := by
  sorry

end NUMINAMATH_CALUDE_square_greater_than_self_for_x_greater_than_one_l3482_348244


namespace NUMINAMATH_CALUDE_olga_aquarium_fish_count_l3482_348242

/-- The number of fish in Olga's aquarium -/
def fish_count : ℕ := 76

/-- The colors of fish in the aquarium -/
inductive FishColor
| Yellow | Blue | Green | Orange | Purple | Pink | Grey | Other

/-- The count of fish for each color -/
def fish_by_color (color : FishColor) : ℕ :=
  match color with
  | FishColor.Yellow => 12
  | FishColor.Blue => 6
  | FishColor.Green => 24
  | FishColor.Purple => 3
  | FishColor.Pink => 8
  | _ => 0  -- We don't have exact numbers for Orange, Grey, and Other

theorem olga_aquarium_fish_count :
  fish_count = 76 ∧
  fish_by_color FishColor.Yellow = 12 ∧
  fish_by_color FishColor.Blue = fish_by_color FishColor.Yellow / 2 ∧
  fish_by_color FishColor.Green = 2 * fish_by_color FishColor.Yellow ∧
  fish_by_color FishColor.Purple = fish_by_color FishColor.Blue / 2 ∧
  fish_by_color FishColor.Pink = fish_by_color FishColor.Green / 3 ∧
  (fish_count : ℚ) = (fish_by_color FishColor.Yellow +
                      fish_by_color FishColor.Blue +
                      fish_by_color FishColor.Green +
                      fish_by_color FishColor.Purple +
                      fish_by_color FishColor.Pink) / 0.7 :=
by sorry

#check olga_aquarium_fish_count

end NUMINAMATH_CALUDE_olga_aquarium_fish_count_l3482_348242


namespace NUMINAMATH_CALUDE_square_difference_theorem_l3482_348212

theorem square_difference_theorem : (13 + 8)^2 - (13 - 8)^2 = 416 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l3482_348212


namespace NUMINAMATH_CALUDE_factoring_expression_l3482_348203

theorem factoring_expression (x : ℝ) : 60 * x + 45 = 15 * (4 * x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factoring_expression_l3482_348203


namespace NUMINAMATH_CALUDE_photo_arrangements_eq_24_l3482_348211

/-- The number of different arrangements for a teacher and two boys and two girls standing in a row,
    with the requirement that the two girls must stand together and the teacher cannot stand at either end. -/
def photo_arrangements : ℕ :=
  let n_people : ℕ := 5
  let n_boys : ℕ := 2
  let n_girls : ℕ := 2
  let n_teacher : ℕ := 1
  let girls_together : ℕ := 1  -- Treat the two girls as one unit
  let teacher_positions : ℕ := n_people - 2  -- Teacher can't be at either end
  Nat.factorial n_people / (Nat.factorial n_boys * Nat.factorial girls_together * Nat.factorial n_teacher)
    * teacher_positions

theorem photo_arrangements_eq_24 : photo_arrangements = 24 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangements_eq_24_l3482_348211


namespace NUMINAMATH_CALUDE_whatsapp_message_count_l3482_348280

/-- Calculate the total number of messages sent in a Whatsapp group over 6 days -/
theorem whatsapp_message_count : 
  let monday : ℕ := 300
  let tuesday : ℕ := 200
  let wednesday : ℕ := tuesday + 300
  let thursday : ℕ := 2 * wednesday
  let friday : ℕ := thursday + thursday / 5
  let saturday : ℕ := friday - friday / 10
  monday + tuesday + wednesday + thursday + friday + saturday = 4280 :=
by sorry

end NUMINAMATH_CALUDE_whatsapp_message_count_l3482_348280


namespace NUMINAMATH_CALUDE_sculpture_height_proof_l3482_348228

/-- The height of the sculpture in inches -/
def sculpture_height : ℝ := 34

/-- The height of the base in inches -/
def base_height : ℝ := 8

/-- The total height of the sculpture and base in feet -/
def total_height_feet : ℝ := 3.5

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℝ := 12

theorem sculpture_height_proof :
  sculpture_height = total_height_feet * feet_to_inches - base_height := by
  sorry

end NUMINAMATH_CALUDE_sculpture_height_proof_l3482_348228


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3482_348274

/-- Two vectors in ℝ² are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (x, 4)
  let b : ℝ × ℝ := (4, x)
  parallel a b → x = 4 ∨ x = -4 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3482_348274


namespace NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l3482_348218

theorem integer_pairs_satisfying_equation :
  {(x, y) : ℤ × ℤ | 8 * x^2 * y^2 + x^2 + y^2 = 10 * x * y} =
  {(0, 0), (1, 1), (-1, -1)} :=
by sorry

end NUMINAMATH_CALUDE_integer_pairs_satisfying_equation_l3482_348218


namespace NUMINAMATH_CALUDE_twelve_hour_clock_chimes_90_l3482_348209

/-- Calculates the sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents a clock that chimes on the hour and half-hour -/
structure ChimingClock where
  hours : ℕ
  chimes_on_hour : ℕ → ℕ
  chimes_on_half_hour : ℕ

/-- Calculates the total number of chimes for a ChimingClock over its set hours -/
def total_chimes (clock : ChimingClock) : ℕ :=
  sum_to_n clock.hours + clock.hours * clock.chimes_on_half_hour

/-- Theorem stating that a clock chiming the hour count on the hour and once on the half-hour,
    over 12 hours, will chime 90 times in total -/
theorem twelve_hour_clock_chimes_90 :
  ∃ (clock : ChimingClock),
    clock.hours = 12 ∧
    clock.chimes_on_hour = id ∧
    clock.chimes_on_half_hour = 1 ∧
    total_chimes clock = 90 := by
  sorry

end NUMINAMATH_CALUDE_twelve_hour_clock_chimes_90_l3482_348209


namespace NUMINAMATH_CALUDE_min_sum_distances_to_four_points_l3482_348266

/-- The minimum sum of distances from a point to four fixed points -/
theorem min_sum_distances_to_four_points :
  let A : ℝ × ℝ := (-3, 0)
  let B : ℝ × ℝ := (1, -1)
  let C : ℝ × ℝ := (0, 3)
  let D : ℝ × ℝ := (-1, 3)
  ∀ P : ℝ × ℝ,
    Real.sqrt ((P.1 - A.1)^2 + (P.2 - A.2)^2) +
    Real.sqrt ((P.1 - B.1)^2 + (P.2 - B.2)^2) +
    Real.sqrt ((P.1 - C.1)^2 + (P.2 - C.2)^2) +
    Real.sqrt ((P.1 - D.1)^2 + (P.2 - D.2)^2) ≥
    3 * Real.sqrt 2 + 2 * Real.sqrt 5 :=
by
  sorry

end NUMINAMATH_CALUDE_min_sum_distances_to_four_points_l3482_348266


namespace NUMINAMATH_CALUDE_pace_difference_is_one_minute_l3482_348207

-- Define the race parameters
def square_length : ℚ := 3/4
def num_laps : ℕ := 7
def this_year_time : ℚ := 42
def last_year_time : ℚ := 47.25

-- Define the total race distance
def race_distance : ℚ := square_length * num_laps

-- Define the average pace for this year and last year
def this_year_pace : ℚ := this_year_time / race_distance
def last_year_pace : ℚ := last_year_time / race_distance

-- Theorem statement
theorem pace_difference_is_one_minute :
  last_year_pace - this_year_pace = 1 := by sorry

end NUMINAMATH_CALUDE_pace_difference_is_one_minute_l3482_348207


namespace NUMINAMATH_CALUDE_students_present_l3482_348263

theorem students_present (total : ℕ) (absent_percent : ℚ) : 
  total = 100 → absent_percent = 14/100 → 
  (total : ℚ) * (1 - absent_percent) = 86 := by
  sorry

end NUMINAMATH_CALUDE_students_present_l3482_348263


namespace NUMINAMATH_CALUDE_shirt_tie_combinations_l3482_348237

theorem shirt_tie_combinations (num_shirts num_ties : ℕ) : 
  num_shirts = 8 → num_ties = 7 → num_shirts * num_ties = 56 := by
  sorry

end NUMINAMATH_CALUDE_shirt_tie_combinations_l3482_348237


namespace NUMINAMATH_CALUDE_flower_pot_cost_difference_flower_pot_cost_difference_proof_l3482_348255

theorem flower_pot_cost_difference 
  (num_pots : ℕ) 
  (total_cost : ℚ) 
  (largest_pot_cost : ℚ) 
  (cost_difference : ℚ) : Prop :=
  num_pots = 6 ∧ 
  total_cost = 33/4 ∧ 
  largest_pot_cost = 13/8 ∧
  (∀ i : ℕ, i < num_pots - 1 → 
    (largest_pot_cost - i * cost_difference) > 
    (largest_pot_cost - (i + 1) * cost_difference)) ∧
  total_cost = (num_pots : ℚ) / 2 * 
    (2 * largest_pot_cost - (num_pots - 1 : ℚ) * cost_difference) →
  cost_difference = 1/10

theorem flower_pot_cost_difference_proof : 
  flower_pot_cost_difference 6 (33/4) (13/8) (1/10) :=
sorry

end NUMINAMATH_CALUDE_flower_pot_cost_difference_flower_pot_cost_difference_proof_l3482_348255


namespace NUMINAMATH_CALUDE_ellipse_properties_l3482_348219

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_minor_axis : b = Real.sqrt 3
  h_eccentricity : a / Real.sqrt (a^2 - b^2) = 2

/-- The standard form of the ellipse -/
def standard_form (e : Ellipse) : Prop :=
  e.a^2 = 4 ∧ e.b^2 = 3

/-- The maximum area of triangle F₁AB -/
def max_triangle_area (e : Ellipse) : ℝ := 3

/-- Main theorem stating the properties of the ellipse -/
theorem ellipse_properties (e : Ellipse) :
  standard_form e ∧ max_triangle_area e = 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_properties_l3482_348219


namespace NUMINAMATH_CALUDE_big_boxes_count_l3482_348284

/-- The number of big boxes given the conditions of the problem -/
def number_of_big_boxes (small_boxes_per_big_box : ℕ) 
                        (candles_per_small_box : ℕ) 
                        (total_candles : ℕ) : ℕ :=
  total_candles / (small_boxes_per_big_box * candles_per_small_box)

theorem big_boxes_count :
  number_of_big_boxes 4 40 8000 = 50 := by
  sorry

#eval number_of_big_boxes 4 40 8000

end NUMINAMATH_CALUDE_big_boxes_count_l3482_348284


namespace NUMINAMATH_CALUDE_troll_count_l3482_348226

theorem troll_count (P B T : ℕ) : 
  P = 6 → 
  B = 4 * P - 6 → 
  T = B / 2 → 
  P + B + T = 33 := by
sorry

end NUMINAMATH_CALUDE_troll_count_l3482_348226


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3482_348200

open Real

theorem inequality_solution_set (x : ℝ) :
  x ∈ Set.Ioo (-1 : ℝ) 1 →
  (abs (sin x) + abs (log (1 - x^2)) > abs (sin x + log (1 - x^2))) ↔ x ∈ Set.Ioo 0 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3482_348200


namespace NUMINAMATH_CALUDE_opposite_player_no_aces_l3482_348224

/-- The number of cards in the deck -/
def deck_size : ℕ := 32

/-- The number of players -/
def num_players : ℕ := 4

/-- The number of cards each player receives -/
def cards_per_player : ℕ := deck_size / num_players

/-- The number of aces in the deck -/
def num_aces : ℕ := 4

/-- The probability that the opposite player has no aces given that one player has no aces -/
def opposite_player_no_aces_prob : ℚ := 130 / 759

theorem opposite_player_no_aces (h1 : deck_size = 32) 
                                (h2 : num_players = 4) 
                                (h3 : cards_per_player = deck_size / num_players) 
                                (h4 : num_aces = 4) : 
  opposite_player_no_aces_prob = 130 / 759 := by
  sorry

end NUMINAMATH_CALUDE_opposite_player_no_aces_l3482_348224


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3482_348259

theorem rectangle_dimensions : ∀ x y : ℝ,
  y = 2 * x →
  2 * (x + y) = 2 * (x * y) →
  x = (3 : ℝ) / 2 ∧ y = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3482_348259


namespace NUMINAMATH_CALUDE_minimum_race_distance_l3482_348225

/-- The minimum distance a runner must travel in a race with given constraints -/
theorem minimum_race_distance (wall_length : ℝ) (a_to_wall : ℝ) (wall_to_b : ℝ) :
  wall_length = 1500 ∧ a_to_wall = 400 ∧ wall_to_b = 600 →
  ⌊Real.sqrt (wall_length ^ 2 + (a_to_wall + wall_to_b) ^ 2) + 0.5⌋ = 1803 := by
  sorry

end NUMINAMATH_CALUDE_minimum_race_distance_l3482_348225


namespace NUMINAMATH_CALUDE_determinant_2x2_l3482_348270

open Matrix

theorem determinant_2x2 (a b c d : ℝ) : 
  det ![![a, c], ![b, d]] = a * d - b * c := by
  sorry

end NUMINAMATH_CALUDE_determinant_2x2_l3482_348270


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l3482_348238

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l3482_348238


namespace NUMINAMATH_CALUDE_loan_division_l3482_348291

/-- Given a total sum of 2730 divided into two parts, where the interest on the first part
    for 8 years at 3% per annum equals the interest on the second part for 3 years at 5% per annum,
    prove that the second part is 1680. -/
theorem loan_division (total : ℝ) (part1 part2 : ℝ) : 
  total = 2730 →
  part1 + part2 = total →
  (part1 * 3 * 8) / 100 = (part2 * 5 * 3) / 100 →
  part2 = 1680 := by
  sorry

end NUMINAMATH_CALUDE_loan_division_l3482_348291


namespace NUMINAMATH_CALUDE_power_minus_ten_over_nine_equals_ten_l3482_348250

theorem power_minus_ten_over_nine_equals_ten : (10^2 - 10) / 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_power_minus_ten_over_nine_equals_ten_l3482_348250


namespace NUMINAMATH_CALUDE_triangle_perimeter_l3482_348231

/-- Given a triangle with sides of lengths 3, 6, and x, where x is a solution to x^2 - 7x + 12 = 0
    and satisfies the triangle inequality, prove that the perimeter of the triangle is 13. -/
theorem triangle_perimeter (x : ℝ) : 
  x^2 - 7*x + 12 = 0 →
  x + 3 > 6 →
  x + 6 > 3 →
  3 + 6 + x = 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l3482_348231


namespace NUMINAMATH_CALUDE_max_value_when_a_zero_one_zero_iff_a_positive_l3482_348279

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x - 1 / x - (a + 1) * Real.log x

-- Theorem for the maximum value when a = 0
theorem max_value_when_a_zero :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f 0 y ≤ f 0 x ∧ f 0 x = -1 :=
sorry

-- Theorem for the range of a when f(x) has exactly one zero
theorem one_zero_iff_a_positive :
  ∀ (a : ℝ), (∃! (x : ℝ), x > 0 ∧ f a x = 0) ↔ a > 0 :=
sorry

end

end NUMINAMATH_CALUDE_max_value_when_a_zero_one_zero_iff_a_positive_l3482_348279


namespace NUMINAMATH_CALUDE_triangle_radii_relation_l3482_348249

/-- Given a triangle with side lengths a, b, c, semi-perimeter p, area S, and circumradius R,
    prove the relationship between the inradius τ, exradii τa, τb, τc, and other triangle properties. -/
theorem triangle_radii_relation
  (a b c p S R τ τa τb τc : ℝ)
  (h1 : S = τ * p)
  (h2 : S = τa * (p - a))
  (h3 : S = τb * (p - b))
  (h4 : S = τc * (p - c))
  (h5 : a * b * c / S = 4 * R) :
  1 / τ^3 - 1 / τa^3 - 1 / τb^3 - 1 / τc^3 = 12 * R / S^2 := by
  sorry


end NUMINAMATH_CALUDE_triangle_radii_relation_l3482_348249


namespace NUMINAMATH_CALUDE_final_red_probability_zero_l3482_348223

/-- Represents the color of a marble -/
inductive Color
| Red
| Blue

/-- Represents the state of the jar -/
structure JarState :=
  (red : Nat)
  (blue : Nat)

/-- Represents the result of drawing two marbles -/
inductive DrawResult
| SameColor (c : Color)
| DifferentColors

/-- Simulates drawing two marbles from the jar -/
def draw (state : JarState) : DrawResult := sorry

/-- Updates the jar state based on the draw result -/
def updateJar (state : JarState) (result : DrawResult) : JarState := sorry

/-- Simulates the entire process of drawing and updating three times -/
def process (initialState : JarState) : JarState := sorry

/-- The probability of the final marble being red -/
def finalRedProbability (initialState : JarState) : Real := sorry

/-- Theorem stating that the probability of the final marble being red is 0 -/
theorem final_red_probability_zero :
  finalRedProbability ⟨2, 2⟩ = 0 := by sorry

end NUMINAMATH_CALUDE_final_red_probability_zero_l3482_348223


namespace NUMINAMATH_CALUDE_integer_power_sum_l3482_348285

theorem integer_power_sum (a : ℝ) (h1 : a ≠ 0) (h2 : ∃ k : ℤ, a + 1/a = k) :
  ∀ n : ℕ, 2 ≤ n ∧ n ≤ 7 → ∃ m : ℤ, a^n + 1/(a^n) = m :=
sorry

end NUMINAMATH_CALUDE_integer_power_sum_l3482_348285


namespace NUMINAMATH_CALUDE_grandpas_initial_tomatoes_l3482_348232

-- Define the number of tomatoes that grew during vacation
def tomatoes_grown : ℕ := 3564

-- Define the multiplication factor for tomato growth
def growth_factor : ℕ := 100

-- Define the function to calculate the initial number of tomatoes
def initial_tomatoes : ℕ := (tomatoes_grown + growth_factor - 1) / growth_factor

-- Theorem statement
theorem grandpas_initial_tomatoes :
  initial_tomatoes = 36 :=
sorry

end NUMINAMATH_CALUDE_grandpas_initial_tomatoes_l3482_348232


namespace NUMINAMATH_CALUDE_minimum_box_cost_greenville_box_cost_l3482_348233

/-- The minimum amount spent on boxes for packaging a fine arts collection -/
theorem minimum_box_cost (box_length box_width box_height : ℝ) 
  (box_cost : ℝ) (collection_volume : ℝ) : ℝ :=
  let box_volume := box_length * box_width * box_height
  let num_boxes := collection_volume / box_volume
  num_boxes * box_cost

/-- The specific case for Greenville State University -/
theorem greenville_box_cost : 
  minimum_box_cost 20 20 12 0.40 2400000 = 200 := by
  sorry

end NUMINAMATH_CALUDE_minimum_box_cost_greenville_box_cost_l3482_348233


namespace NUMINAMATH_CALUDE_students_in_section_A_l3482_348290

/-- The number of students in section A -/
def students_A : ℕ := 26

/-- The number of students in section B -/
def students_B : ℕ := 34

/-- The average weight of students in section A (in kg) -/
def avg_weight_A : ℚ := 50

/-- The average weight of students in section B (in kg) -/
def avg_weight_B : ℚ := 30

/-- The average weight of the whole class (in kg) -/
def avg_weight_total : ℚ := 38.67

theorem students_in_section_A : 
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = avg_weight_total := by
  sorry

end NUMINAMATH_CALUDE_students_in_section_A_l3482_348290


namespace NUMINAMATH_CALUDE_unique_positive_solution_l3482_348204

theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ (x - 5) / 12 = 5 / (x - 12) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l3482_348204


namespace NUMINAMATH_CALUDE_expected_draws_no_ugly_l3482_348213

def bag_total : ℕ := 20
def blue_marbles : ℕ := 9
def ugly_marbles : ℕ := 10
def special_marbles : ℕ := 1

def prob_blue : ℚ := blue_marbles / bag_total
def prob_special : ℚ := special_marbles / bag_total

theorem expected_draws_no_ugly : 
  let p := prob_blue
  let q := prob_special
  (∑' k : ℕ, k * (1 / (1 - p)) * p^(k-1) * q) = 20 / 11 :=
sorry

end NUMINAMATH_CALUDE_expected_draws_no_ugly_l3482_348213


namespace NUMINAMATH_CALUDE_cynthia_potato_harvest_l3482_348268

theorem cynthia_potato_harvest :
  ∀ (P : ℕ),
  (P ≥ 13) →
  (P - 13) % 2 = 0 →
  ((P - 13) / 2 - 13 = 436) →
  P = 911 :=
by
  sorry

end NUMINAMATH_CALUDE_cynthia_potato_harvest_l3482_348268


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3482_348276

theorem complex_number_quadrant : 
  let z : ℂ := (1/2 + (Real.sqrt 3 / 2) * Complex.I)^2
  z.re < 0 ∧ z.im > 0 := by sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3482_348276


namespace NUMINAMATH_CALUDE_train_speed_and_length_l3482_348287

/-- Given a train passing a stationary observer in 7 seconds and taking 25 seconds to pass a 378-meter platform at constant speed, prove that the train's speed is 21 m/s and its length is 147 m. -/
theorem train_speed_and_length :
  ∀ (V l : ℝ),
  (7 * V = l) →
  (25 * V = 378 + l) →
  (V = 21 ∧ l = 147) :=
by sorry

end NUMINAMATH_CALUDE_train_speed_and_length_l3482_348287


namespace NUMINAMATH_CALUDE_function_properties_l3482_348288

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a * x^2 - 3

-- State the theorem
theorem function_properties (a : ℝ) :
  (∀ x > 0, HasDerivAt (f a) ((-1 : ℝ)) 1) →
  (∃ m : ℝ, m = -1 ∧ 
    (∀ x > 0, f (-1) x - m * x ≤ -3) ∧
    (∀ m' < m, ∃ x > 0, f (-1) x - m' * x > -3)) ∧
  (∀ x > 0, x * Real.log x - x^2 - 3 - x * Real.exp x + x^2 < -2 * x - 3) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3482_348288


namespace NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3482_348252

theorem greatest_power_of_two_factor (n : ℕ) : 
  2^1200 ∣ (15^600 - 3^600) ∧ 
  ∀ k > 1200, ¬(2^k ∣ (15^600 - 3^600)) :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_two_factor_l3482_348252


namespace NUMINAMATH_CALUDE_red_peaches_count_l3482_348264

theorem red_peaches_count (green_peaches : ℕ) (red_peaches : ℕ) : 
  green_peaches = 16 → red_peaches = green_peaches + 1 → red_peaches = 17 := by
  sorry

end NUMINAMATH_CALUDE_red_peaches_count_l3482_348264


namespace NUMINAMATH_CALUDE_certain_number_proof_l3482_348246

theorem certain_number_proof (original : ℝ) (certain : ℝ) : 
  original = 50 → (1/5 : ℝ) * original - 5 = certain → certain = 5 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3482_348246


namespace NUMINAMATH_CALUDE_problem_solution_l3482_348258

/-- The function f as defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 else x^2 + a*x

/-- Theorem stating the solution to the problem -/
theorem problem_solution (a : ℝ) : f a (f a 0) = 4 * a → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3482_348258


namespace NUMINAMATH_CALUDE_farmer_apples_l3482_348248

theorem farmer_apples (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : 
  initial = 924 → given_away = 639 → remaining = initial - given_away → remaining = 285 := by
  sorry

end NUMINAMATH_CALUDE_farmer_apples_l3482_348248


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l3482_348286

/-- The number of ways to arrange books on a shelf -/
def arrange_books (num_math_books : Nat) (num_history_books : Nat) : Nat :=
  if num_math_books ≥ 2 then
    num_math_books * (num_math_books - 1) * Nat.factorial (num_math_books + num_history_books - 2)
  else
    0

/-- Theorem: The number of ways to arrange 3 math books and 5 history books with math books on both ends is 4320 -/
theorem book_arrangement_theorem :
  arrange_books 3 5 = 4320 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l3482_348286


namespace NUMINAMATH_CALUDE_sum_first_five_eq_l3482_348241

/-- A geometric progression with given fourth and fifth terms -/
structure GeometricProgression where
  b₄ : ℚ  -- Fourth term
  b₅ : ℚ  -- Fifth term
  h₄ : b₄ = 1 / 25
  h₅ : b₅ = 1 / 125

/-- The sum of the first five terms of a geometric progression -/
def sum_first_five (gp : GeometricProgression) : ℚ :=
  -- Definition of the sum (to be proved)
  781 / 125

/-- Theorem stating that the sum of the first five terms is 781/125 -/
theorem sum_first_five_eq (gp : GeometricProgression) :
  sum_first_five gp = 781 / 125 := by
  sorry

#eval sum_first_five ⟨1/25, 1/125, rfl, rfl⟩

end NUMINAMATH_CALUDE_sum_first_five_eq_l3482_348241


namespace NUMINAMATH_CALUDE_ellipse_equation_hyperbola_equation_l3482_348299

-- Define the ellipse properties
def ellipse_axis_sum : ℝ := 18
def ellipse_focal_length : ℝ := 6

-- Define the reference ellipse for the hyperbola
def reference_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the point Q
def point_Q : ℝ × ℝ := (2, 1)

-- Theorem for the ellipse equation
theorem ellipse_equation (x y : ℝ) :
  (x^2 / 25 + y^2 / 16 = 1) ∨ (x^2 / 16 + y^2 / 25 = 1) :=
sorry

-- Theorem for the hyperbola equation
theorem hyperbola_equation (x y : ℝ) :
  x^2 / 2 - y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_hyperbola_equation_l3482_348299


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l3482_348257

theorem ratio_of_numbers (x y : ℝ) (h : x > y) (h' : (x + y) / (x - y) = 4 / 3) :
  x / y = 7 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l3482_348257


namespace NUMINAMATH_CALUDE_elephant_arrangements_l3482_348214

theorem elephant_arrangements (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 2) :
  (n! / k!) = 20160 := by
  sorry

end NUMINAMATH_CALUDE_elephant_arrangements_l3482_348214


namespace NUMINAMATH_CALUDE_factory_non_defective_percentage_l3482_348221

/-- Represents a machine in the factory -/
structure Machine where
  production_percentage : Real
  defective_rate : Real

/-- Calculates the percentage of non-defective products given a list of machines -/
def non_defective_percentage (machines : List Machine) : Real :=
  100 - (machines.map (λ m => m.production_percentage * m.defective_rate)).sum

/-- The theorem stating that the percentage of non-defective products is 95.25% -/
theorem factory_non_defective_percentage : 
  let machines : List Machine := [
    ⟨20, 2⟩,
    ⟨25, 4⟩,
    ⟨30, 5⟩,
    ⟨15, 7⟩,
    ⟨10, 8⟩
  ]
  non_defective_percentage machines = 95.25 := by
  sorry

end NUMINAMATH_CALUDE_factory_non_defective_percentage_l3482_348221


namespace NUMINAMATH_CALUDE_store_optimal_plan_l3482_348294

/-- Represents the types of soccer balls -/
inductive BallType
| A
| B

/-- Represents the store's inventory and pricing -/
structure Store where
  cost_price : BallType → ℕ
  selling_price : BallType → ℕ
  budget : ℕ

/-- Represents the purchase plan -/
structure PurchasePlan where
  num_A : ℕ
  num_B : ℕ

def Store.is_valid (s : Store) : Prop :=
  s.cost_price BallType.A = s.cost_price BallType.B + 40 ∧
  480 / s.cost_price BallType.A = 240 / s.cost_price BallType.B ∧
  s.budget = 4000 ∧
  s.selling_price BallType.A = 100 ∧
  s.selling_price BallType.B = 55

def PurchasePlan.is_valid (p : PurchasePlan) (s : Store) : Prop :=
  p.num_A ≥ p.num_B ∧
  p.num_A * s.cost_price BallType.A + p.num_B * s.cost_price BallType.B ≤ s.budget

def PurchasePlan.profit (p : PurchasePlan) (s : Store) : ℤ :=
  (s.selling_price BallType.A - s.cost_price BallType.A) * p.num_A +
  (s.selling_price BallType.B - s.cost_price BallType.B) * p.num_B

theorem store_optimal_plan (s : Store) (h : s.is_valid) :
  ∃ (p : PurchasePlan), 
    p.is_valid s ∧ 
    s.cost_price BallType.A = 80 ∧ 
    s.cost_price BallType.B = 40 ∧
    p.num_A = 34 ∧ 
    p.num_B = 32 ∧
    ∀ (p' : PurchasePlan), p'.is_valid s → p.profit s ≥ p'.profit s :=
sorry

end NUMINAMATH_CALUDE_store_optimal_plan_l3482_348294


namespace NUMINAMATH_CALUDE_complex_magnitude_equals_five_l3482_348292

theorem complex_magnitude_equals_five (t : ℝ) (ht : t > 0) :
  Complex.abs (-3 + t * Complex.I) = 5 → t = 4 := by
sorry

end NUMINAMATH_CALUDE_complex_magnitude_equals_five_l3482_348292


namespace NUMINAMATH_CALUDE_star_equation_solution_l3482_348215

-- Define the star operation
noncomputable def star (a b : ℝ) : ℝ := a + Real.sqrt (b + Real.sqrt (b + Real.sqrt b))

-- Theorem statement
theorem star_equation_solution (h : ℝ) :
  star 8 h = 11 → h = 6 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l3482_348215


namespace NUMINAMATH_CALUDE_plane_distance_ratio_l3482_348273

/-- Proves the ratio of plane distance to total distance -/
theorem plane_distance_ratio (total : ℝ) (bus : ℝ) (train : ℝ) (plane : ℝ) :
  total = 900 →
  train = (2/3) * bus →
  bus = 360 →
  plane = total - (bus + train) →
  plane / total = 1/3 := by
sorry

end NUMINAMATH_CALUDE_plane_distance_ratio_l3482_348273


namespace NUMINAMATH_CALUDE_expression_equality_l3482_348262

theorem expression_equality (x : ℝ) (h : x > 0) : 
  (∃! e : ℕ, e = 1) ∧ 
  (6^x * x^3 = 3^x * x^3 + 3^x * x^3) ∧ 
  ((3*x)^(3*x) ≠ 3^x * x^3 + 3^x * x^3) ∧ 
  (3^x * x^6 ≠ 3^x * x^3 + 3^x * x^3) ∧ 
  ((6*x)^x ≠ 3^x * x^3 + 3^x * x^3) :=
sorry

end NUMINAMATH_CALUDE_expression_equality_l3482_348262
