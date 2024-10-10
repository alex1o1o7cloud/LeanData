import Mathlib

namespace parallelogram_line_theorem_l1612_161273

/-- A parallelogram with vertices at (15,35), (15,95), (27,122), and (27,62) -/
structure Parallelogram :=
  (v1 : ℝ × ℝ) (v2 : ℝ × ℝ) (v3 : ℝ × ℝ) (v4 : ℝ × ℝ)

/-- A line that passes through the origin -/
structure Line :=
  (slope : ℚ)

/-- The line cuts the parallelogram into two congruent polygons -/
def cuts_into_congruent_polygons (p : Parallelogram) (l : Line) : Prop := sorry

/-- m and n are relatively prime integers -/
def are_relatively_prime (m n : ℕ) : Prop := sorry

theorem parallelogram_line_theorem (p : Parallelogram) (l : Line) 
  (h1 : p.v1 = (15, 35)) (h2 : p.v2 = (15, 95)) (h3 : p.v3 = (27, 122)) (h4 : p.v4 = (27, 62))
  (h5 : cuts_into_congruent_polygons p l)
  (h6 : ∃ (m n : ℕ), l.slope = m / n ∧ are_relatively_prime m n) :
  ∃ (m n : ℕ), l.slope = m / n ∧ are_relatively_prime m n ∧ m + n = 71 := by sorry

end parallelogram_line_theorem_l1612_161273


namespace curve_length_is_pi_l1612_161275

/-- A closed convex curve in the plane -/
structure ClosedConvexCurve where
  -- Add necessary fields here
  -- This is just a placeholder definition

/-- The length of a curve -/
noncomputable def curve_length (c : ClosedConvexCurve) : ℝ :=
  sorry

/-- The length of the projection of a curve onto a line -/
noncomputable def projection_length (c : ClosedConvexCurve) (l : Line) : ℝ :=
  sorry

/-- A line in the plane -/
structure Line where
  -- Add necessary fields here
  -- This is just a placeholder definition

theorem curve_length_is_pi (c : ClosedConvexCurve) 
  (h : ∀ l : Line, projection_length c l = 1) : 
  curve_length c = π :=
sorry

end curve_length_is_pi_l1612_161275


namespace probability_ratio_l1612_161282

def num_balls : ℕ := 20
def num_bins : ℕ := 5

def distribution_A : List ℕ := [2, 4, 4, 3, 7]
def distribution_B : List ℕ := [3, 3, 4, 4, 4]

def probability_A : ℚ := (Nat.choose 5 1) * (Nat.choose 4 2) * (Nat.choose 2 1) * (Nat.factorial 20) / 
  ((Nat.factorial 2) * (Nat.factorial 4) * (Nat.factorial 4) * (Nat.factorial 3) * (Nat.factorial 7) * (Nat.choose (num_balls + num_bins - 1) (num_bins - 1)))

def probability_B : ℚ := (Nat.choose 5 2) * (Nat.choose 3 3) * (Nat.factorial 20) / 
  ((Nat.factorial 3) * (Nat.factorial 3) * (Nat.factorial 4) * (Nat.factorial 4) * (Nat.factorial 4) * (Nat.choose (num_balls + num_bins - 1) (num_bins - 1)))

theorem probability_ratio : probability_A / probability_B = 12 := by
  sorry

end probability_ratio_l1612_161282


namespace elisa_math_books_l1612_161234

theorem elisa_math_books :
  ∀ (total math lit : ℕ),
  total < 100 →
  total = 24 + math + lit →
  (math + 1) * 9 = total + 1 →
  lit * 4 = total + 1 →
  math = 7 := by
sorry

end elisa_math_books_l1612_161234


namespace monday_polygons_tuesday_segments_wednesday_polygons_l1612_161245

/-- Represents the types of polygons Miky can draw -/
inductive Polygon
| Square
| Pentagon
| Hexagon

/-- Number of sides for each polygon type -/
def sides (p : Polygon) : Nat :=
  match p with
  | .Square => 4
  | .Pentagon => 5
  | .Hexagon => 6

/-- Number of diagonals for each polygon type -/
def diagonals (p : Polygon) : Nat :=
  match p with
  | .Square => 2
  | .Pentagon => 5
  | .Hexagon => 9

/-- Total number of line segments (sides + diagonals) for each polygon type -/
def totalSegments (p : Polygon) : Nat :=
  sides p + diagonals p

theorem monday_polygons :
  ∃ p : Polygon, sides p = diagonals p ∧ p = Polygon.Pentagon :=
sorry

theorem tuesday_segments (n : Nat) (h : n * sides Polygon.Hexagon = 18) :
  n * diagonals Polygon.Hexagon = 27 :=
sorry

theorem wednesday_polygons (n : Nat) (h : n * totalSegments Polygon.Pentagon = 70) :
  n = 7 :=
sorry

end monday_polygons_tuesday_segments_wednesday_polygons_l1612_161245


namespace smartphone_cost_l1612_161224

theorem smartphone_cost (selling_price : ℝ) (loss_percentage : ℝ) (initial_cost : ℝ) : 
  selling_price = 255 ∧ 
  loss_percentage = 15 ∧ 
  selling_price = initial_cost * (1 - loss_percentage / 100) →
  initial_cost = 300 :=
by sorry

end smartphone_cost_l1612_161224


namespace paper_folding_volumes_l1612_161200

/-- Given a square paper with side length 1, prove the volume of a cone and max volume of a rectangular prism --/
theorem paper_folding_volumes (ε : ℝ) (hε : ε = 0.0001) :
  ∃ (V_cone V_prism : ℝ),
    (abs (V_cone - (π / 6)) < ε) ∧
    (abs (V_prism - (1 / (3 * Real.sqrt 3))) < ε) ∧
    (∀ (a b c : ℝ), 2 * (a * b + b * c + c * a) = 1 → a * b * c ≤ V_prism) := by
  sorry

end paper_folding_volumes_l1612_161200


namespace square_perimeter_l1612_161217

theorem square_perimeter (area : ℝ) (side : ℝ) (perimeter : ℝ) : 
  area = 200 → 
  side^2 = area → 
  perimeter = 4 * side → 
  perimeter = 40 * Real.sqrt 2 := by
  sorry

end square_perimeter_l1612_161217


namespace alice_sold_120_oranges_l1612_161255

/-- The number of oranges Emily sold -/
def emily_oranges : ℕ := sorry

/-- The number of oranges Alice sold -/
def alice_oranges : ℕ := 2 * emily_oranges

/-- The total number of oranges sold -/
def total_oranges : ℕ := 180

/-- Theorem stating that Alice sold 120 oranges -/
theorem alice_sold_120_oranges : alice_oranges = 120 :=
  by
    sorry

/-- The condition that the total oranges sold is the sum of Alice's and Emily's oranges -/
axiom total_is_sum : total_oranges = alice_oranges + emily_oranges

end alice_sold_120_oranges_l1612_161255


namespace reciprocal_expression_l1612_161240

theorem reciprocal_expression (m n : ℝ) (h : m * n = 1) :
  (2 * m - 2 / n) * (1 / m + n) = 0 := by sorry

end reciprocal_expression_l1612_161240


namespace p_amount_l1612_161236

theorem p_amount (p q r : ℝ) : 
  p = (1/8 * p) + (1/8 * p) + 42 → p = 56 := by sorry

end p_amount_l1612_161236


namespace new_weighted_average_age_l1612_161238

/-- The new weighted average age of a class after new students join -/
theorem new_weighted_average_age
  (n₁ : ℕ) (a₁ : ℝ)
  (n₂ : ℕ) (a₂ : ℝ)
  (n₃ : ℕ) (a₃ : ℝ)
  (n₄ : ℕ) (a₄ : ℝ)
  (n₅ : ℕ) (a₅ : ℝ)
  (h₁ : n₁ = 15) (h₂ : a₁ = 42)
  (h₃ : n₂ = 20) (h₄ : a₂ = 35)
  (h₅ : n₃ = 10) (h₆ : a₃ = 50)
  (h₇ : n₄ = 7)  (h₈ : a₄ = 30)
  (h₉ : n₅ = 11) (h₁₀ : a₅ = 45) :
  (n₁ * a₁ + n₂ * a₂ + n₃ * a₃ + n₄ * a₄ + n₅ * a₅) / (n₁ + n₂ + n₃ + n₄ + n₅) = 2535 / 63 := by
  sorry

#eval (2535 : Float) / 63

end new_weighted_average_age_l1612_161238


namespace cows_per_herd_l1612_161226

theorem cows_per_herd (total_cows : ℕ) (num_herds : ℕ) (h1 : total_cows = 320) (h2 : num_herds = 8) :
  total_cows / num_herds = 40 := by
  sorry

end cows_per_herd_l1612_161226


namespace bella_age_l1612_161260

theorem bella_age (bella_age : ℕ) (brother_age : ℕ) : 
  brother_age = bella_age + 9 →
  bella_age + brother_age = 19 →
  bella_age = 5 := by
sorry

end bella_age_l1612_161260


namespace workshop_salary_problem_l1612_161222

/-- Workshop salary problem -/
theorem workshop_salary_problem 
  (total_workers : ℕ) 
  (num_technicians : ℕ) 
  (avg_salary_technicians : ℕ) 
  (avg_salary_others : ℕ) 
  (h1 : total_workers = 14)
  (h2 : num_technicians = 7)
  (h3 : avg_salary_technicians = 10000)
  (h4 : avg_salary_others = 6000) :
  (num_technicians * avg_salary_technicians + 
   (total_workers - num_technicians) * avg_salary_others) / total_workers = 8000 := by
  sorry


end workshop_salary_problem_l1612_161222


namespace exists_sequence_expectation_sum_neq_sum_expectation_l1612_161251

/-- A sequence of random variables -/
def RandomSequence := ℕ → MeasurableSpace ℝ

/-- Expected value of a random variable -/
noncomputable def expectation (X : MeasurableSpace ℝ) : ℝ := sorry

/-- Infinite sum of random variables -/
noncomputable def infiniteSum (ξ : RandomSequence) : MeasurableSpace ℝ := sorry

/-- Theorem: There exists a sequence of random variables where the expectation of the sum
    is not equal to the sum of expectations -/
theorem exists_sequence_expectation_sum_neq_sum_expectation :
  ∃ ξ : RandomSequence,
    expectation (infiniteSum ξ) ≠ ∑' n, expectation (ξ n) := by sorry

end exists_sequence_expectation_sum_neq_sum_expectation_l1612_161251


namespace ice_cream_flavors_l1612_161205

def num_flavors : ℕ := 4
def num_scoops : ℕ := 5

def total_distributions : ℕ := (num_scoops + num_flavors - 1).choose (num_flavors - 1)
def non_mint_distributions : ℕ := (num_scoops + (num_flavors - 1) - 1).choose ((num_flavors - 1) - 1)

theorem ice_cream_flavors :
  total_distributions - non_mint_distributions = 35 := by sorry

end ice_cream_flavors_l1612_161205


namespace sibling_ages_l1612_161286

/-- A family with 4 siblings: Richard, David, Scott, and Jane. -/
structure Family :=
  (Richard David Scott Jane : ℕ)

/-- The conditions and question of the problem -/
theorem sibling_ages (f : Family) : 
  f.Richard = f.David + 6 →
  f.David = f.Scott + 8 →
  f.Jane = f.Richard - 5 →
  f.Richard + 8 = 2 * (f.Scott + 8) →
  f.Jane + 10 = (f.David + 10) / 2 + 4 →
  f.Scott + 12 + f.Jane + 12 = 60 →
  f.Richard - 3 + f.David - 3 + f.Scott - 3 + f.Jane - 3 = 43 := by
  sorry

#check sibling_ages

end sibling_ages_l1612_161286


namespace matilda_jellybeans_l1612_161230

/-- Given that:
    1. Matilda has half as many jellybeans as Matt
    2. Matt has ten times as many jellybeans as Steve
    3. Steve has 84 jellybeans
    Prove that Matilda has 420 jellybeans. -/
theorem matilda_jellybeans (steve_jellybeans : ℕ) (matt_jellybeans : ℕ) (matilda_jellybeans : ℕ)
  (h1 : steve_jellybeans = 84)
  (h2 : matt_jellybeans = 10 * steve_jellybeans)
  (h3 : matilda_jellybeans = matt_jellybeans / 2) :
  matilda_jellybeans = 420 := by
  sorry

end matilda_jellybeans_l1612_161230


namespace custom_mul_neg_three_two_l1612_161262

/-- Custom multiplication operation -/
def custom_mul (a b : ℝ) := a * b - a^2

/-- Theorem: The custom multiplication of -3 and 2 equals -15 -/
theorem custom_mul_neg_three_two :
  custom_mul (-3) 2 = -15 := by
  sorry

end custom_mul_neg_three_two_l1612_161262


namespace ninas_toys_l1612_161270

theorem ninas_toys (toy_price : ℕ) (card_packs : ℕ) (card_price : ℕ) (shirts : ℕ) (shirt_price : ℕ) (total_spent : ℕ) : 
  toy_price = 10 →
  card_packs = 2 →
  card_price = 5 →
  shirts = 5 →
  shirt_price = 6 →
  total_spent = 70 →
  ∃ (num_toys : ℕ), num_toys * toy_price + card_packs * card_price + shirts * shirt_price = total_spent ∧ num_toys = 3 :=
by
  sorry

end ninas_toys_l1612_161270


namespace min_y_value_l1612_161288

theorem min_y_value (x y : ℝ) (h : x^2 + y^2 = 20*x + 64*y) : 
  ∃ (y_min : ℝ), y_min = 32 - 2 * Real.sqrt 281 ∧ 
  ∀ (x' y' : ℝ), x'^2 + y'^2 = 20*x' + 64*y' → y' ≥ y_min :=
by sorry

end min_y_value_l1612_161288


namespace f_monotonicity_and_minimum_l1612_161247

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 2

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x - 9

-- Theorem for intervals of monotonicity and minimum value
theorem f_monotonicity_and_minimum :
  (∀ x y, x < y ∧ x < -1 ∧ y < -1 → f x < f y) ∧
  (∀ x y, x < y ∧ x > 3 ∧ y > 3 → f x < f y) ∧
  (∀ x y, x < y ∧ x > -1 ∧ y < 3 → f x > f y) ∧
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≥ -20) ∧
  (f 2 = -20) :=
sorry

end f_monotonicity_and_minimum_l1612_161247


namespace basketball_score_total_l1612_161203

theorem basketball_score_total (tim joe ken : ℕ) : 
  tim = joe + 20 →
  tim = ken / 2 →
  tim = 30 →
  tim + joe + ken = 100 := by
sorry

end basketball_score_total_l1612_161203


namespace similar_triangles_perimeter_l1612_161221

theorem similar_triangles_perimeter (h_small h_large : ℝ) (p_small p_large : ℝ) :
  h_small / h_large = 3 / 5 →
  p_small = 12 →
  p_small / p_large = h_small / h_large →
  p_large = 20 := by
  sorry

end similar_triangles_perimeter_l1612_161221


namespace opposite_solutions_imply_a_value_l1612_161274

theorem opposite_solutions_imply_a_value (a x y : ℚ) : 
  (x - y = 3 * a + 1) → 
  (x + y = 9 - 5 * a) → 
  (x = -y) → 
  (a = 9 / 5) := by
sorry

end opposite_solutions_imply_a_value_l1612_161274


namespace puss_in_boots_pikes_l1612_161243

theorem puss_in_boots_pikes (x : ℚ) : x = 4 + (1/2) * x → x = 8 := by
  sorry

end puss_in_boots_pikes_l1612_161243


namespace parking_lot_search_time_l1612_161241

/-- Calculates the time spent searching a parking lot given the layout and walking speed. -/
theorem parking_lot_search_time
  (section_g_rows : ℕ)
  (section_g_cars_per_row : ℕ)
  (section_h_rows : ℕ)
  (section_h_cars_per_row : ℕ)
  (cars_passed_per_minute : ℕ)
  (h_section_g_rows : section_g_rows = 15)
  (h_section_g_cars : section_g_cars_per_row = 10)
  (h_section_h_rows : section_h_rows = 20)
  (h_section_h_cars : section_h_cars_per_row = 9)
  (h_cars_passed : cars_passed_per_minute = 11)
  : (section_g_rows * section_g_cars_per_row + section_h_rows * section_h_cars_per_row) / cars_passed_per_minute = 30 := by
  sorry


end parking_lot_search_time_l1612_161241


namespace age_problem_solution_l1612_161208

/-- Given three people a, b, and c, with their ages represented as natural numbers. -/
def age_problem (a b c : ℕ) : Prop :=
  -- The average age of a, b, and c is 27 years
  (a + b + c) / 3 = 27 ∧
  -- The average age of a and c is 29 years
  (a + c) / 2 = 29 →
  -- The age of b is 23 years
  b = 23

/-- Theorem stating that under the given conditions, b's age is 23 years -/
theorem age_problem_solution :
  ∀ a b c : ℕ, age_problem a b c :=
by
  sorry

end age_problem_solution_l1612_161208


namespace hypotenuse_length_l1612_161266

/-- A right triangle with specific medians -/
structure RightTriangleWithMedians where
  /-- First leg of the triangle -/
  a : ℝ
  /-- Second leg of the triangle -/
  b : ℝ
  /-- First median (from vertex of acute angle) -/
  m₁ : ℝ
  /-- Second median (from vertex of acute angle) -/
  m₂ : ℝ
  /-- The first median is 6 -/
  h₁ : m₁ = 6
  /-- The second median is 3√13 -/
  h₂ : m₂ = 3 * Real.sqrt 13
  /-- Relationship between first leg and first median -/
  h₃ : m₁^2 = a^2 + (3*b/2)^2
  /-- Relationship between second leg and second median -/
  h₄ : m₂^2 = b^2 + (3*a/2)^2

/-- The theorem stating that the hypotenuse of the triangle is 3√23 -/
theorem hypotenuse_length (t : RightTriangleWithMedians) : 
  Real.sqrt (9 * (t.a^2 + t.b^2)) = 3 * Real.sqrt 23 := by
  sorry

end hypotenuse_length_l1612_161266


namespace weight_replacement_l1612_161229

theorem weight_replacement (W : ℝ) (original_weight replaced_weight : ℝ) : 
  W / 10 + 2.5 = (W - replaced_weight + 75) / 10 → replaced_weight = 50 :=
by sorry

end weight_replacement_l1612_161229


namespace cube_sum_reciprocal_l1612_161231

theorem cube_sum_reciprocal (a : ℝ) (h : (a + 1/a)^2 = 12) : 
  a^3 + 1/a^3 = 18 * Real.sqrt 3 := by
  sorry

end cube_sum_reciprocal_l1612_161231


namespace adrianna_gum_l1612_161212

/-- Calculates the remaining pieces of gum after sharing with friends -/
def remaining_gum (initial : ℕ) (additional : ℕ) (friends : ℕ) : ℕ :=
  initial + additional - friends

/-- Proves that Adrianna has 2 pieces of gum left -/
theorem adrianna_gum : remaining_gum 10 3 11 = 2 := by
  sorry

end adrianna_gum_l1612_161212


namespace circumscribed_parallelepiped_surface_area_l1612_161276

/-- A right parallelepiped circumscribed by a sphere -/
structure CircumscribedParallelepiped where
  /-- The first base diagonal of the parallelepiped -/
  a : ℝ
  /-- The second base diagonal of the parallelepiped -/
  b : ℝ
  /-- The parallelepiped is circumscribed by a sphere -/
  is_circumscribed : True

/-- The surface area of a circumscribed parallelepiped -/
def surface_area (p : CircumscribedParallelepiped) : ℝ :=
  6 * p.a * p.b

/-- Theorem: The surface area of a right parallelepiped circumscribed by a sphere,
    with base diagonals a and b, is equal to 6ab -/
theorem circumscribed_parallelepiped_surface_area
  (p : CircumscribedParallelepiped) :
  surface_area p = 6 * p.a * p.b := by
  sorry

end circumscribed_parallelepiped_surface_area_l1612_161276


namespace simplify_expression_l1612_161287

theorem simplify_expression (b : ℝ) : (1 : ℝ) * (2 * b) * (3 * b^2) * (4 * b^3) * (5 * b^4) * (6 * b^5) = 720 * b^15 := by
  sorry

end simplify_expression_l1612_161287


namespace max_a_inequality_l1612_161295

theorem max_a_inequality (a : ℝ) : 
  (∀ x : ℝ, Real.sqrt (2 * x) - a ≥ Real.sqrt (9 - 5 * x)) → 
  a ≤ -3 :=
sorry

end max_a_inequality_l1612_161295


namespace m_range_theorem_l1612_161279

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x^2 + m*x + 1 = 0 ∧ y^2 + m*y + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the range of m
def m_range (m : ℝ) : Prop := (1 < m ∧ m ≤ 2) ∨ (m ≥ 3)

-- State the theorem
theorem m_range_theorem : 
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m_range m :=
by sorry

end m_range_theorem_l1612_161279


namespace hcf_from_lcm_and_product_l1612_161214

theorem hcf_from_lcm_and_product (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 600) 
  (h_product : a * b = 18000) : 
  Nat.gcd a b = 30 := by
  sorry

end hcf_from_lcm_and_product_l1612_161214


namespace degree_of_g_l1612_161209

def f (x : ℝ) : ℝ := -7 * x^4 + 3 * x^3 + x - 5

theorem degree_of_g (g : ℝ → ℝ) :
  (∃ (a b : ℝ), ∀ x, f x + g x = a * x + b) →
  (∃ (a b c d e : ℝ), a ≠ 0 ∧ ∀ x, g x = a * x^4 + b * x^3 + c * x^2 + d * x + e) :=
by sorry

end degree_of_g_l1612_161209


namespace ten_people_seating_arrangement_l1612_161235

/-- The number of ways to seat n people around a round table -/
def roundTableArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- The number of ways to arrange 3 people in a block where one person is fixed between the other two -/
def fixedBlockArrangements : ℕ := 2

theorem ten_people_seating_arrangement :
  roundTableArrangements 9 * fixedBlockArrangements = 80640 := by
  sorry

end ten_people_seating_arrangement_l1612_161235


namespace circle_reflection_translation_l1612_161281

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the x-axis -/
def reflectX (p : Point) : Point :=
  { x := p.x, y := -p.y }

/-- Translates a point vertically -/
def translateY (p : Point) (dy : ℝ) : Point :=
  { x := p.x, y := p.y + dy }

/-- The main theorem -/
theorem circle_reflection_translation :
  let Q : Point := { x := 3, y := -4 }
  let Q' := translateY (reflectX Q) 5
  Q'.x = 3 ∧ Q'.y = 9 := by sorry

end circle_reflection_translation_l1612_161281


namespace max_inscribed_sphere_volume_l1612_161202

/-- The maximum volume of a sphere inscribed in a right triangular prism -/
theorem max_inscribed_sphere_volume (a b h : ℝ) (ha : 0 < a) (hb : 0 < b) (hh : 0 < h) :
  let r := min (a * b / (a + b + Real.sqrt (a^2 + b^2))) (h / 2)
  (4 / 3) * Real.pi * r^3 = (9 * Real.pi) / 2 :=
by sorry

end max_inscribed_sphere_volume_l1612_161202


namespace line_with_acute_inclination_l1612_161259

/-- Given a line passing through points A(2,1) and B(1,m) with an acute angle of inclination, 
    the value of m must be less than 1. -/
theorem line_with_acute_inclination (m : ℝ) : 
  let A : ℝ × ℝ := (2, 1)
  let B : ℝ × ℝ := (1, m)
  let slope : ℝ := (m - A.2) / (B.1 - A.1)
  (0 < slope) ∧ (slope < 1) → m < 1 := by sorry

end line_with_acute_inclination_l1612_161259


namespace chess_tournament_games_l1612_161253

theorem chess_tournament_games (n : ℕ) (h : n = 10) : 
  (n * (n - 1)) / 2 = 45 := by
  sorry

end chess_tournament_games_l1612_161253


namespace current_speed_l1612_161215

/-- Calculates the speed of the current given boat travel information -/
theorem current_speed (boat_speed : ℝ) (distance : ℝ) (time_against : ℝ) (time_with : ℝ) :
  boat_speed = 15.6 →
  distance = 96 →
  time_against = 8 →
  time_with = 5 →
  ∃ (current_speed : ℝ),
    distance = time_against * (boat_speed - current_speed) ∧
    distance = time_with * (boat_speed + current_speed) ∧
    current_speed = 3.6 := by
  sorry

end current_speed_l1612_161215


namespace james_nickels_l1612_161291

/-- Represents the number of nickels in James' jar -/
def n : ℕ := sorry

/-- Represents the number of quarters in James' jar -/
def q : ℕ := sorry

/-- The total value in cents -/
def total_cents : ℕ := 685

/-- Theorem stating the number of nickels in James' jar -/
theorem james_nickels : 
  (5 * n + 25 * q = total_cents) ∧ 
  (n = q + 11) → 
  n = 32 := by sorry

end james_nickels_l1612_161291


namespace paths_through_B_l1612_161294

/-- The number of paths between two points on a grid -/
def grid_paths (right : ℕ) (down : ℕ) : ℕ := Nat.choose (right + down) down

/-- The theorem stating the number of 11-step paths from A to C passing through B -/
theorem paths_through_B : 
  let paths_A_to_B := grid_paths 4 2
  let paths_B_to_C := grid_paths 3 3
  paths_A_to_B * paths_B_to_C = 300 := by sorry

end paths_through_B_l1612_161294


namespace A_inter_B_eq_B_l1612_161265

-- Define set A
def A : Set ℝ := {y | ∃ x, y = |x| - 1}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem A_inter_B_eq_B : A ∩ B = B := by sorry

end A_inter_B_eq_B_l1612_161265


namespace ellipse_equation_l1612_161277

/-- The standard equation of an ellipse with given properties -/
theorem ellipse_equation (c a b : ℝ) (h1 : c = 3) (h2 : a = 5) (h3 : b = 4) 
  (h4 : c / a = 3 / 5) (h5 : a^2 = b^2 + c^2) :
  ∀ x y : ℝ, (x^2 / 25 + y^2 / 16 = 1) ↔ 
  (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end ellipse_equation_l1612_161277


namespace pet_store_cages_l1612_161223

theorem pet_store_cages (initial_puppies : ℕ) (sold_puppies : ℕ) (puppies_per_cage : ℕ) 
  (h1 : initial_puppies = 102)
  (h2 : sold_puppies = 21)
  (h3 : puppies_per_cage = 9) :
  (initial_puppies - sold_puppies) / puppies_per_cage = 9 :=
by sorry

end pet_store_cages_l1612_161223


namespace xy_sum_values_l1612_161244

theorem xy_sum_values (x y : ℕ) (hx : x > 0) (hy : y > 0) (hx_lt : x < 25) (hy_lt : y < 25) 
  (h_eq : x + y + x * y = 119) : 
  x + y = 27 ∨ x + y = 24 ∨ x + y = 21 ∨ x + y = 20 :=
by sorry

end xy_sum_values_l1612_161244


namespace problem_solution_l1612_161254

theorem problem_solution (x y : ℝ) : 
  y - Real.sqrt (x - 2022) = Real.sqrt (2022 - x) - 2023 →
  (x + y) ^ 2023 = -1 := by
sorry

end problem_solution_l1612_161254


namespace xyz_sum_l1612_161210

theorem xyz_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 12)
  (eq2 : y^2 + y*z + z^2 = 16)
  (eq3 : z^2 + x*z + x^2 = 28) :
  x*y + y*z + x*z = 16 := by
sorry

end xyz_sum_l1612_161210


namespace stationery_box_cost_l1612_161228

/-- The cost of a single stationery box in yuan -/
def unit_price : ℕ := 23

/-- The number of stationery boxes to be purchased -/
def quantity : ℕ := 3

/-- The total cost of purchasing the stationery boxes -/
def total_cost : ℕ := unit_price * quantity

theorem stationery_box_cost : total_cost = 69 := by
  sorry

end stationery_box_cost_l1612_161228


namespace acme_cheaper_at_min_shirts_l1612_161242

/-- Acme T-Shirt Company's pricing function -/
def acme_cost (x : ℕ) : ℚ := 75 + 12 * x

/-- Gamma T-shirt Company's pricing function -/
def gamma_cost (x : ℕ) : ℚ := 16 * x

/-- The minimum number of shirts for which Acme is cheaper than Gamma -/
def min_shirts_for_acme : ℕ := 19

theorem acme_cheaper_at_min_shirts :
  acme_cost min_shirts_for_acme < gamma_cost min_shirts_for_acme ∧
  ∀ n : ℕ, n < min_shirts_for_acme → acme_cost n ≥ gamma_cost n :=
by sorry

end acme_cheaper_at_min_shirts_l1612_161242


namespace even_odd_sum_difference_l1612_161237

/-- Sum of the first n even integers -/
def sum_even (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of the first n odd integers -/
def sum_odd (n : ℕ) : ℕ := n^2

/-- Count of odd integers divisible by 5 up to 2n-1 -/
def count_odd_div_5 (n : ℕ) : ℕ := (2*n - 1) / 10 + 1

/-- Sum of odd integers divisible by 5 up to 2n-1 -/
def sum_odd_div_5 (n : ℕ) : ℕ := 5 * (count_odd_div_5 n) * (count_odd_div_5 n)

/-- Sum of odd integers not divisible by 5 up to 2n-1 -/
def sum_odd_not_div_5 (n : ℕ) : ℕ := sum_odd n - sum_odd_div_5 n

theorem even_odd_sum_difference (n : ℕ) : 
  sum_even n - sum_odd_not_div_5 n = 51000 := by sorry

end even_odd_sum_difference_l1612_161237


namespace composition_of_convex_increasing_and_convex_is_convex_l1612_161219

def IsConvex (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ) (t : ℝ), 0 ≤ t ∧ t ≤ 1 →
    f (t * x + (1 - t) * y) ≤ t * f x + (1 - t) * f y

def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x ≤ y → f x ≤ f y

theorem composition_of_convex_increasing_and_convex_is_convex
  (f g : ℝ → ℝ) (hf : IsConvex f) (hg : IsConvex g) (hf_inc : IsIncreasing f) :
  IsConvex (f ∘ g) := by
  sorry

end composition_of_convex_increasing_and_convex_is_convex_l1612_161219


namespace tea_customers_l1612_161278

/-- Proves that the number of tea customers is 8 given the conditions of the problem -/
theorem tea_customers (coffee_price : ℕ) (tea_price : ℕ) (coffee_customers : ℕ) (total_revenue : ℕ) :
  coffee_price = 5 →
  tea_price = 4 →
  coffee_customers = 7 →
  total_revenue = 67 →
  ∃ tea_customers : ℕ, 
    tea_customers = 8 ∧ 
    coffee_price * coffee_customers + tea_price * tea_customers = total_revenue :=
by
  sorry

end tea_customers_l1612_161278


namespace average_sale_per_month_l1612_161267

def sales : List ℝ := [2435, 2920, 2855, 3230, 2560, 1000]

theorem average_sale_per_month :
  (sales.sum / sales.length : ℝ) = 2500 := by sorry

end average_sale_per_month_l1612_161267


namespace first_player_wins_l1612_161283

/-- Represents a game played on a regular polygon -/
structure PolygonGame where
  sides : ℕ
  is_regular : sides > 2

/-- Represents a move in the game -/
inductive Move
| connect (v1 v2 : ℕ)

/-- Represents the state of the game -/
structure GameState where
  game : PolygonGame
  moves : List Move

/-- Checks if a move is valid given the current game state -/
def is_valid_move (state : GameState) (move : Move) : Prop :=
  sorry

/-- Checks if the game is over (no more valid moves) -/
def is_game_over (state : GameState) : Prop :=
  sorry

/-- Determines the winner of the game -/
def winner (state : GameState) : Option Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Move

/-- Checks if a strategy is winning for the first player -/
def is_winning_strategy (strat : Strategy) (game : PolygonGame) : Prop :=
  sorry

/-- The main theorem: there exists a winning strategy for the first player in a 1968-sided polygon game -/
theorem first_player_wins :
  ∃ (strat : Strategy), is_winning_strategy strat ⟨1968, by norm_num⟩ :=
sorry

end first_player_wins_l1612_161283


namespace solve_x_and_y_l1612_161272

-- Define the universal set I
def I (x : ℝ) : Set ℝ := {2, 3, x^2 + 2*x - 3}

-- Define set A
def A : Set ℝ := {5}

-- Define the complement of A with respect to I
def complement_A (x : ℝ) (y : ℝ) : Set ℝ := {2, y}

-- Theorem statement
theorem solve_x_and_y (x : ℝ) (y : ℝ) :
  (5 ∈ I x) ∧ (complement_A x y = (I x) \ A) →
  ((x = -4 ∨ x = 2) ∧ y = 3) :=
by sorry

end solve_x_and_y_l1612_161272


namespace area_circle_circumscribed_equilateral_triangle_l1612_161258

/-- The area of a circle circumscribed about an equilateral triangle with side length 15 units is 75π square units. -/
theorem area_circle_circumscribed_equilateral_triangle :
  let s : ℝ := 15  -- Side length of the equilateral triangle
  let r : ℝ := s * Real.sqrt 3 / 3  -- Radius of the circumscribed circle
  let area : ℝ := π * r^2  -- Area of the circle
  area = 75 * π := by
  sorry

end area_circle_circumscribed_equilateral_triangle_l1612_161258


namespace geometric_sequences_theorem_l1612_161296

/-- Two geometric sequences satisfying given conditions -/
structure GeometricSequences where
  a : ℝ
  q : ℝ
  r : ℝ
  a_pos : a > 0
  b1_minus_a1 : a * r - a = 1
  b2_minus_a2 : a * r * r - a * q = 2
  b3_minus_a3 : a * r^3 - a * q^2 = 3

/-- The general term of the sequence a_n -/
def a_n (gs : GeometricSequences) (n : ℕ) : ℝ := gs.a * gs.q^(n-1)

/-- The general term of the sequence b_n -/
def b_n (gs : GeometricSequences) (n : ℕ) : ℝ := gs.a * gs.r^(n-1)

theorem geometric_sequences_theorem (gs : GeometricSequences) :
  (gs.a = 1 → (∀ n : ℕ, a_n gs n = (2 + Real.sqrt 2)^(n-1) ∨ a_n gs n = (2 - Real.sqrt 2)^(n-1))) ∧
  ((∃! q : ℝ, ∀ n : ℕ, a_n gs n = gs.a * q^(n-1)) → gs.a = 1/3) :=
sorry

end geometric_sequences_theorem_l1612_161296


namespace farmer_problem_l1612_161293

theorem farmer_problem (total_cost : ℕ) (rabbit_cost chicken_cost : ℕ) 
  (h_total : total_cost = 1125)
  (h_rabbit : rabbit_cost = 30)
  (h_chicken : chicken_cost = 45) :
  ∃! (r c : ℕ), 
    r > 0 ∧ c > 0 ∧ 
    r * rabbit_cost + c * chicken_cost = total_cost :=
by
  sorry

end farmer_problem_l1612_161293


namespace english_only_students_l1612_161298

/-- Represents the number of students in each language class -/
structure LanguageClasses where
  english : ℕ
  french : ℕ
  spanish : ℕ

/-- The conditions of the problem -/
def language_class_conditions (c : LanguageClasses) : Prop :=
  c.english + c.french + c.spanish = 40 ∧
  c.english = 3 * c.french ∧
  c.english = 2 * c.spanish

/-- The theorem to prove -/
theorem english_only_students (c : LanguageClasses) 
  (h : language_class_conditions c) : 
  c.english - (c.french + c.spanish) = 30 := by
  sorry


end english_only_students_l1612_161298


namespace unique_scores_count_l1612_161292

/-- Represents the number of baskets made by the player -/
def total_baskets : ℕ := 7

/-- Represents the possible point values for each basket -/
inductive BasketType
| two_point : BasketType
| three_point : BasketType

/-- Calculates the total score given a list of basket types -/
def calculate_score (baskets : List BasketType) : ℕ :=
  baskets.foldl (fun acc b => acc + match b with
    | BasketType.two_point => 2
    | BasketType.three_point => 3) 0

/-- Generates all possible combinations of basket types -/
def generate_combinations : List (List BasketType) :=
  sorry

/-- Theorem stating that the number of unique possible scores is 8 -/
theorem unique_scores_count :
  (generate_combinations.map calculate_score).toFinset.card = 8 := by sorry

end unique_scores_count_l1612_161292


namespace total_flowers_and_sticks_l1612_161201

theorem total_flowers_and_sticks (num_pots : ℕ) (flowers_per_pot : ℕ) (sticks_per_pot : ℕ) 
  (h1 : num_pots = 466) 
  (h2 : flowers_per_pot = 53) 
  (h3 : sticks_per_pot = 181) : 
  num_pots * flowers_per_pot + num_pots * sticks_per_pot = 109044 :=
by sorry

end total_flowers_and_sticks_l1612_161201


namespace unit_digit_product_l1612_161261

theorem unit_digit_product : ∃ n : ℕ, (3^68 * 6^59 * 7^71) % 10 = 8 ∧ n = (3^68 * 6^59 * 7^71) := by
  sorry

end unit_digit_product_l1612_161261


namespace steps_climbed_proof_l1612_161252

/-- Calculates the total number of steps climbed given the number of steps and climbs for two ladders -/
def total_steps_climbed (full_ladder_steps : ℕ) (full_ladder_climbs : ℕ) 
                        (small_ladder_steps : ℕ) (small_ladder_climbs : ℕ) : ℕ :=
  full_ladder_steps * full_ladder_climbs + small_ladder_steps * small_ladder_climbs

/-- Proves that the total number of steps climbed is 152 given the specific ladder configurations -/
theorem steps_climbed_proof :
  total_steps_climbed 11 10 6 7 = 152 := by
  sorry

end steps_climbed_proof_l1612_161252


namespace complex_product_theorem_l1612_161290

theorem complex_product_theorem (Q E D : ℂ) : 
  Q = 3 + 4*I ∧ E = 2*I ∧ D = 3 - 4*I → 2 * Q * E * D = 100 * I :=
by sorry

end complex_product_theorem_l1612_161290


namespace min_sum_first_two_terms_l1612_161239

def is_valid_sequence (b : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → b (n + 2) * (b (n + 1) + 1) = b n + 2210

theorem min_sum_first_two_terms (b : ℕ → ℕ) (h : is_valid_sequence b) :
  ∃ b₁ b₂ : ℕ, b 1 = b₁ ∧ b 2 = b₂ ∧ b₁ + b₂ = 147 ∧
  ∀ b₁' b₂' : ℕ, b 1 = b₁' ∧ b 2 = b₂' → b₁' + b₂' ≥ 147 :=
sorry

end min_sum_first_two_terms_l1612_161239


namespace boat_speed_calculation_l1612_161285

/-- Given the downstream speed and upstream speed of a boat, 
    calculate the stream speed and the man's rowing speed. -/
theorem boat_speed_calculation (R S : ℝ) :
  ∃ (x y : ℝ), 
    (R = y + x) ∧ 
    (S = y - x) ∧ 
    (x = (R - S) / 2) ∧ 
    (y = (R + S) / 2) := by
  sorry

end boat_speed_calculation_l1612_161285


namespace roxy_daily_consumption_l1612_161289

/-- Represents the daily water consumption of the siblings --/
structure WaterConsumption where
  theo : ℕ
  mason : ℕ
  roxy : ℕ

/-- Represents the total weekly water consumption of the siblings --/
def weekly_total (wc : WaterConsumption) : ℕ :=
  7 * (wc.theo + wc.mason + wc.roxy)

/-- Theorem stating that given the conditions, Roxy drinks 9 cups of water daily --/
theorem roxy_daily_consumption (wc : WaterConsumption) :
  wc.theo = 8 → wc.mason = 7 → weekly_total wc = 168 → wc.roxy = 9 := by
  sorry

#check roxy_daily_consumption

end roxy_daily_consumption_l1612_161289


namespace prime_mod_8_not_sum_of_three_squares_l1612_161299

theorem prime_mod_8_not_sum_of_three_squares (p : ℕ) (hp : Nat.Prime p) (hmod : p % 8 = 7) :
  ¬ ∃ (a b c : ℤ), (a ^ 2 + b ^ 2 + c ^ 2 : ℤ) = p := by
  sorry

end prime_mod_8_not_sum_of_three_squares_l1612_161299


namespace log2_derivative_l1612_161204

theorem log2_derivative (x : ℝ) (h : x > 0) : 
  deriv (fun x => Real.log x / Real.log 2) x = 1 / (x * Real.log 2) := by
  sorry

end log2_derivative_l1612_161204


namespace simplify_polynomial_l1612_161256

theorem simplify_polynomial (x : ℝ) : (3 * x^2 + 9 * x - 5) - (2 * x^2 + 3 * x - 10) = x^2 + 6 * x + 5 := by
  sorry

end simplify_polynomial_l1612_161256


namespace find_number_l1612_161257

theorem find_number : ∃ x : ℝ, x / 2 = 9 ∧ x = 18 := by
  sorry

end find_number_l1612_161257


namespace quadratic_inequality_range_l1612_161268

theorem quadratic_inequality_range (k : ℝ) : 
  (∀ x : ℝ, x^2 - k*x + 1 > 0) → -2 < k ∧ k < 2 := by
  sorry

end quadratic_inequality_range_l1612_161268


namespace percentage_problem_l1612_161263

theorem percentage_problem (P : ℝ) : 
  (0.15 * P * (0.5 * 4000) = 90) → P = 0.3 := by
  sorry

end percentage_problem_l1612_161263


namespace percentage_to_pass_l1612_161206

/-- The percentage needed to pass an exam, given the achieved score, shortfall, and maximum possible marks. -/
theorem percentage_to_pass 
  (achieved_score : ℕ) 
  (shortfall : ℕ) 
  (max_marks : ℕ) 
  (h1 : achieved_score = 212)
  (h2 : shortfall = 28)
  (h3 : max_marks = 800) : 
  (achieved_score + shortfall) / max_marks * 100 = 30 := by
sorry

end percentage_to_pass_l1612_161206


namespace quadratic_inequality_always_nonpositive_l1612_161269

theorem quadratic_inequality_always_nonpositive :
  ∀ x : ℝ, -8 * x^2 + 4 * x - 7 ≤ 0 := by
  sorry

end quadratic_inequality_always_nonpositive_l1612_161269


namespace frog_jump_theorem_l1612_161250

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Function to calculate the number of final frog positions -/
def numFrogPositions (ℓ : ℕ) : ℕ :=
  ((ℓ + 2) / 2) * ((ℓ + 4) / 2) * (((ℓ + 1) / 2) * ((ℓ + 3) / 2))^2 / 8

/-- Main theorem statement -/
theorem frog_jump_theorem (abc : Triangle) (ℓ : ℕ) (m n : Point) :
  (abc.a.x = 0 ∧ abc.a.y = 0) →  -- A at origin
  (abc.b.x = 1 ∧ abc.b.y = 0) →  -- B at (1,0)
  (abc.c.x = 1/2 ∧ abc.c.y = Real.sqrt 3 / 2) →  -- C at (1/2, √3/2)
  (m.x = ℓ ∧ m.y = 0) →  -- M on AB
  (n.x = ℓ/2 ∧ n.y = ℓ * Real.sqrt 3 / 2) →  -- N on AC
  ∃ (finalPositions : ℕ), finalPositions = numFrogPositions ℓ :=
by sorry

end frog_jump_theorem_l1612_161250


namespace find_a_l1612_161233

theorem find_a (a b : ℚ) (h1 : a / 3 = b / 2) (h2 : a + b = 10) : a = 6 := by
  sorry

end find_a_l1612_161233


namespace interval_of_decrease_l1612_161280

/-- Given a function f with derivative f'(x) = 2x - 4, 
    prove that the interval of decrease for f(x-1) is (-∞, 3) -/
theorem interval_of_decrease (f : ℝ → ℝ) (h : ∀ x, deriv f x = 2 * x - 4) :
  ∀ x, x < 3 ↔ deriv (fun y ↦ f (y - 1)) x < 0 := by
  sorry

end interval_of_decrease_l1612_161280


namespace train_encounters_l1612_161216

/-- Represents the number of hours in the journey -/
def journey_duration : ℕ := 5

/-- Represents the number of trains already on the route when the journey begins -/
def initial_trains : ℕ := 4

/-- Calculates the number of trains encountered during the journey -/
def trains_encountered (duration : ℕ) (initial : ℕ) : ℕ :=
  initial + duration

theorem train_encounters :
  trains_encountered journey_duration initial_trains = 9 := by
  sorry

end train_encounters_l1612_161216


namespace fourth_grade_students_l1612_161207

theorem fourth_grade_students (initial : ℕ) (left : ℕ) (new : ℕ) (final : ℕ) : 
  initial = 8 → left = 5 → new = 8 → final = initial - left + new → final = 11 := by
  sorry

end fourth_grade_students_l1612_161207


namespace restaurant_change_l1612_161297

/-- Calculates the change received after a restaurant meal -/
theorem restaurant_change 
  (lee_money : ℕ) 
  (friend_money : ℕ) 
  (wings_cost : ℕ) 
  (salad_cost : ℕ) 
  (soda_cost : ℕ) 
  (soda_quantity : ℕ) 
  (tax : ℕ) 
  (h1 : lee_money = 10) 
  (h2 : friend_money = 8) 
  (h3 : wings_cost = 6) 
  (h4 : salad_cost = 4) 
  (h5 : soda_cost = 1) 
  (h6 : soda_quantity = 2) 
  (h7 : tax = 3) : 
  lee_money + friend_money - (wings_cost + salad_cost + soda_cost * soda_quantity + tax) = 3 :=
by
  sorry

end restaurant_change_l1612_161297


namespace rational_cube_root_sum_implies_rational_inverse_sum_l1612_161218

theorem rational_cube_root_sum_implies_rational_inverse_sum 
  (p q r : ℚ) 
  (h : ∃ (x : ℚ), x = (p^2*q)^(1/3) + (q^2*r)^(1/3) + (r^2*p)^(1/3)) : 
  ∃ (y : ℚ), y = 1/(p^2*q)^(1/3) + 1/(q^2*r)^(1/3) + 1/(r^2*p)^(1/3) := by
  sorry

end rational_cube_root_sum_implies_rational_inverse_sum_l1612_161218


namespace diamond_example_l1612_161246

/-- The diamond operation -/
def diamond (a b : ℤ) : ℤ := a * b^2 - b + 1

/-- Theorem stating the result of (3 ◇ 4) ◇ 2 -/
theorem diamond_example : diamond (diamond 3 4) 2 = 179 := by
  sorry

end diamond_example_l1612_161246


namespace sqrt_difference_inequality_l1612_161227

theorem sqrt_difference_inequality (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  Real.sqrt a - Real.sqrt b < Real.sqrt (a - b) := by
  sorry

end sqrt_difference_inequality_l1612_161227


namespace expression_equivalence_l1612_161225

theorem expression_equivalence : 
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * 
  (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 5^256 - 4^256 := by
  sorry

end expression_equivalence_l1612_161225


namespace errand_time_is_110_minutes_l1612_161264

def driving_time_one_way : ℕ := 20
def parent_teacher_night_time : ℕ := 70

def total_errand_time : ℕ :=
  2 * driving_time_one_way + parent_teacher_night_time

theorem errand_time_is_110_minutes :
  total_errand_time = 110 := by
  sorry

end errand_time_is_110_minutes_l1612_161264


namespace smallest_angle_60_implies_n_3_or_4_l1612_161220

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A line in 3D space determined by two points -/
structure Line3D where
  p1 : Point3D
  p2 : Point3D

/-- The angle between two lines in 3D space -/
def angle (l1 l2 : Line3D) : ℝ := sorry

/-- A configuration of n points in 3D space -/
def Configuration (n : ℕ) := Fin n → Point3D

/-- The smallest angle formed by any pair of lines in a configuration -/
def smallestAngle (config : Configuration n) : ℝ := sorry

theorem smallest_angle_60_implies_n_3_or_4 (n : ℕ) (h1 : n > 2) 
  (config : Configuration n) (h2 : smallestAngle config = 60) :
  n = 3 ∨ n = 4 := by sorry

end smallest_angle_60_implies_n_3_or_4_l1612_161220


namespace quadratic_real_roots_range_l1612_161211

def quadratic_equation (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 - 2 * x + 1

def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, quadratic_equation m x = 0

theorem quadratic_real_roots_range (m : ℝ) :
  has_real_roots m ↔ m ≤ 2 ∧ m ≠ 1 :=
sorry

end quadratic_real_roots_range_l1612_161211


namespace simple_interest_problem_l1612_161284

/-- Given a principal amount P, an unknown interest rate R, and a 10-year period,
    if increasing the interest rate by 5% results in Rs. 400 more interest,
    then P must equal Rs. 800. -/
theorem simple_interest_problem (P R : ℝ) (h : P > 0) (h_R : R > 0) :
  (P * (R + 5) * 10) / 100 = (P * R * 10) / 100 + 400 →
  P = 800 := by
sorry

end simple_interest_problem_l1612_161284


namespace john_profit_l1612_161271

def calculate_profit (woodburning_qty : ℕ) (woodburning_price : ℚ)
                     (metal_qty : ℕ) (metal_price : ℚ)
                     (painting_qty : ℕ) (painting_price : ℚ)
                     (glass_qty : ℕ) (glass_price : ℚ)
                     (wood_cost : ℚ) (metal_cost : ℚ)
                     (paint_cost : ℚ) (glass_cost : ℚ)
                     (woodburning_discount : ℚ) (glass_discount : ℚ)
                     (sales_tax : ℚ) : ℚ :=
  sorry

theorem john_profit :
  calculate_profit 20 15 15 25 10 40 5 30 100 150 120 90 (10/100) (15/100) (5/100) = 771.13 :=
sorry

end john_profit_l1612_161271


namespace no_real_roots_l1612_161248

theorem no_real_roots : ¬∃ (x : ℝ), Real.sqrt (x + 9) - Real.sqrt (x - 6) + 2 = 0 := by
  sorry

end no_real_roots_l1612_161248


namespace log_expression_equals_zero_l1612_161232

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_expression_equals_zero (x : ℝ) (h : x > 10) :
  (log x) ^ (log (log (log x))) - (log (log x)) ^ (log (log x)) = 0 := by
  sorry

end log_expression_equals_zero_l1612_161232


namespace sin_sum_arcsin_arctan_l1612_161213

theorem sin_sum_arcsin_arctan :
  Real.sin (Real.arcsin (4/5) + Real.arctan (1/2)) = 11 * Real.sqrt 5 / 25 := by
  sorry

end sin_sum_arcsin_arctan_l1612_161213


namespace problem_solution_l1612_161249

-- Define the functions f and g
def f (x : ℝ) : ℝ := 3 * x^2 + 12
def g (x : ℝ) : ℝ := x^2 - 6

-- State the theorem
theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : f (g a) = 12) : a = Real.sqrt 6 := by
  sorry

end problem_solution_l1612_161249
