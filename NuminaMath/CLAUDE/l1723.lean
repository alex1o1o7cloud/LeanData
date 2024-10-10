import Mathlib

namespace map_distance_theorem_l1723_172337

/-- Given a map scale and an actual distance, calculate the distance on the map --/
def map_distance (scale : ℚ) (actual_distance_km : ℚ) : ℚ :=
  (actual_distance_km * 100000) / (1 / scale)

/-- Theorem: The distance between two points on a map with scale 1/250000 and actual distance 5 km is 2 cm --/
theorem map_distance_theorem :
  map_distance (1 / 250000) 5 = 2 := by
  sorry

#eval map_distance (1 / 250000) 5

end map_distance_theorem_l1723_172337


namespace sin_transformation_l1723_172380

theorem sin_transformation (x : ℝ) : 
  2 * Real.sin (3 * x + π / 6) = 2 * Real.sin ((x + π / 6) / 3) := by
  sorry

end sin_transformation_l1723_172380


namespace right_angled_iff_sum_radii_right_angled_iff_sum_squared_radii_l1723_172300

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  r : ℝ
  r_a : ℝ
  r_b : ℝ
  r_c : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < r ∧ 0 < r_a ∧ 0 < r_b ∧ 0 < r_c
  h_triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

def is_right_angled (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2 ∨ t.b^2 + t.c^2 = t.a^2 ∨ t.c^2 + t.a^2 = t.b^2

theorem right_angled_iff_sum_radii (t : Triangle) :
  is_right_angled t ↔ t.r + t.r_a + t.r_b + t.r_c = t.a + t.b + t.c :=
sorry

theorem right_angled_iff_sum_squared_radii (t : Triangle) :
  is_right_angled t ↔ t.r^2 + t.r_a^2 + t.r_b^2 + t.r_c^2 = t.a^2 + t.b^2 + t.c^2 :=
sorry

end right_angled_iff_sum_radii_right_angled_iff_sum_squared_radii_l1723_172300


namespace exactly_two_non_congruent_triangles_l1723_172317

/-- A triangle with integer side lengths --/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ

/-- The perimeter of a triangle --/
def perimeter (t : IntTriangle) : ℕ := t.a + t.b + t.c

/-- Triangle inequality --/
def is_valid_triangle (t : IntTriangle) : Prop :=
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

/-- Non-congruent triangles --/
def are_non_congruent (t1 t2 : IntTriangle) : Prop :=
  ¬(t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∧
  ¬(t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∧
  ¬(t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of valid triangles with perimeter 12 --/
def valid_triangles : Set IntTriangle :=
  {t : IntTriangle | perimeter t = 12 ∧ is_valid_triangle t}

/-- The theorem to be proved --/
theorem exactly_two_non_congruent_triangles :
  ∃ (t1 t2 : IntTriangle),
    t1 ∈ valid_triangles ∧
    t2 ∈ valid_triangles ∧
    are_non_congruent t1 t2 ∧
    ∀ (t3 : IntTriangle),
      t3 ∈ valid_triangles →
      (t3 = t1 ∨ t3 = t2 ∨ ¬(are_non_congruent t1 t3 ∧ are_non_congruent t2 t3)) :=
by sorry

end exactly_two_non_congruent_triangles_l1723_172317


namespace sequence_properties_l1723_172341

def geometric_sequence (i : ℕ) : ℕ := 7 * 3^(16 - i) * 5^(i - 1)

theorem sequence_properties :
  (∀ i ∈ Finset.range 16, geometric_sequence (i + 1) > 0) ∧
  (∀ i ∈ Finset.range 5, geometric_sequence (i + 1) ≥ 10^8 ∧ geometric_sequence (i + 1) < 10^9) ∧
  (∀ i ∈ Finset.range 5, geometric_sequence (i + 6) ≥ 10^9 ∧ geometric_sequence (i + 6) < 10^10) ∧
  (∀ i ∈ Finset.range 4, geometric_sequence (i + 11) ≥ 10^10 ∧ geometric_sequence (i + 11) < 10^11) ∧
  (∀ i ∈ Finset.range 2, geometric_sequence (i + 15) ≥ 10^11 ∧ geometric_sequence (i + 15) < 10^12) ∧
  (∀ i ∈ Finset.range 15, geometric_sequence (i + 2) / geometric_sequence (i + 1) = geometric_sequence 2 / geometric_sequence 1) :=
by sorry

end sequence_properties_l1723_172341


namespace larger_integer_problem_l1723_172309

theorem larger_integer_problem :
  ∃ (x : ℕ+) (y : ℕ+), (4 * x)^2 - 2 * x = 8100 ∧ x + 10 = 2 * y ∧ x = 22 := by
  sorry

end larger_integer_problem_l1723_172309


namespace system_solution_l1723_172308

theorem system_solution :
  ∃ (x y : ℤ), 
    (x + 9773 = 13200) ∧
    (2 * x - 3 * y = 1544) ∧
    (x = 3427) ∧
    (y = 1770) := by
  sorry

end system_solution_l1723_172308


namespace system_solution_l1723_172340

theorem system_solution : 
  ∃! (x y : ℝ), x = 4 * y ∧ x + 2 * y = -12 ∧ x = -8 ∧ y = -2 := by
  sorry

end system_solution_l1723_172340


namespace hyperbola_eccentricity_l1723_172394

/-- A hyperbola with center at the origin -/
structure Hyperbola where
  center : ℝ × ℝ
  foci_on_axes : Bool
  asymptote_angle : ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Theorem: The eccentricity of the given hyperbola is either 2 or 2√3/3 -/
theorem hyperbola_eccentricity (C : Hyperbola) 
  (h1 : C.center = (0, 0))
  (h2 : C.foci_on_axes = true)
  (h3 : C.asymptote_angle = π / 3) :
  eccentricity C = 2 ∨ eccentricity C = 2 * Real.sqrt 3 / 3 := by sorry

end hyperbola_eccentricity_l1723_172394


namespace sequence_equality_l1723_172328

theorem sequence_equality (a : ℕ → ℝ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, (a (n + 1))^2 + (a n)^2 + 1 = 2 * ((a (n + 1)) * (a n) + (a (n + 1)) - (a n))) :
  ∀ n : ℕ, a n = n := by
sorry

end sequence_equality_l1723_172328


namespace revenue_calculation_l1723_172303

/-- The total revenue from selling apples and oranges -/
def total_revenue (z t : ℕ) (a b : ℚ) : ℚ :=
  z * a + t * b

/-- Theorem: The total revenue from selling 200 apples at $0.50 each and 75 oranges at $0.75 each is $156.25 -/
theorem revenue_calculation :
  total_revenue 200 75 (1/2) (3/4) = 156.25 := by
  sorry

end revenue_calculation_l1723_172303


namespace geometric_sequence_sum_l1723_172377

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The theorem states that for a geometric sequence satisfying given conditions, 
    the sum of the 5th and 6th terms equals 48. -/
theorem geometric_sequence_sum (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a)
  (h_sum1 : a 1 + a 2 = 3)
  (h_sum2 : a 3 + a 4 = 12) :
  a 5 + a 6 = 48 := by
  sorry


end geometric_sequence_sum_l1723_172377


namespace orthocenter_centroid_perpendicular_l1723_172368

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- The centroid of a triangle -/
def centroid (t : Triangle) : ℝ × ℝ := sorry

/-- The area of a triangle given its vertices -/
def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

/-- Checks if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop := sorry

/-- Checks if two points are not equal -/
def notEqual (p q : ℝ × ℝ) : Prop := p ≠ q

/-- Calculates the angle between three points -/
def angle (p q r : ℝ × ℝ) : ℝ := sorry

theorem orthocenter_centroid_perpendicular (t : Triangle) :
  isAcuteAngled t →
  notEqual t.A t.B →
  notEqual t.A t.C →
  let H := orthocenter t
  let G := centroid t
  1 / triangleArea H t.A t.B + 1 / triangleArea H t.A t.C = 1 / triangleArea H t.B t.C →
  angle t.A G H = 90 := by sorry

end orthocenter_centroid_perpendicular_l1723_172368


namespace extreme_points_imply_a_range_and_negative_min_l1723_172311

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x - (a / 2) * x^2 - x

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := Real.log x - a * x

-- Theorem statement
theorem extreme_points_imply_a_range_and_negative_min 
  (a : ℝ) (x₁ x₂ : ℝ) (h_extreme : x₁ < x₂ ∧ 
    f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0 ∧
    (∀ x, x₁ < x → x < x₂ → f_deriv a x ≠ 0)) :
  (0 < a ∧ a < Real.exp (-1)) ∧ f a x₁ < 0 := by
  sorry

end extreme_points_imply_a_range_and_negative_min_l1723_172311


namespace sum_of_coefficients_l1723_172346

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ : ℝ) :
  (∀ x : ℝ, (x^2 - 3*x + 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + 
                                a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9 + a₁₀*x^10) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ = -2 :=
by
  sorry

end sum_of_coefficients_l1723_172346


namespace sin_cos_product_l1723_172362

theorem sin_cos_product (α : Real) (h : Real.tan α = 3) : 
  Real.sin α * Real.cos α = 3 / 10 := by
  sorry

end sin_cos_product_l1723_172362


namespace solids_of_revolution_l1723_172359

-- Define the type for geometric solids
inductive GeometricSolid
  | Cylinder
  | HexagonalPyramid
  | Cube
  | Sphere
  | Tetrahedron

-- Define the property of being a solid of revolution
def isSolidOfRevolution : GeometricSolid → Prop :=
  fun solid => match solid with
    | GeometricSolid.Cylinder => True
    | GeometricSolid.Sphere => True
    | _ => False

-- Theorem statement
theorem solids_of_revolution :
  ∀ s : GeometricSolid,
    isSolidOfRevolution s ↔ (s = GeometricSolid.Cylinder ∨ s = GeometricSolid.Sphere) :=
by
  sorry

#check solids_of_revolution

end solids_of_revolution_l1723_172359


namespace target_breaking_orders_l1723_172392

theorem target_breaking_orders : 
  (Nat.factorial 8) / (Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 2) = 560 := by
  sorry

end target_breaking_orders_l1723_172392


namespace sum_of_fractions_zero_l1723_172331

theorem sum_of_fractions_zero (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
    (h : a + b + 2*c = 0) : 
  1 / (b^2 + 4*c^2 - a^2) + 1 / (a^2 + 4*c^2 - b^2) + 1 / (a^2 + b^2 - 4*c^2) = 0 := by
  sorry

end sum_of_fractions_zero_l1723_172331


namespace different_winning_scores_l1723_172304

def cross_country_meet (n : ℕ) : Prop :=
  n = 12 ∧ ∃ (team_size : ℕ), team_size = 6

def sum_of_positions (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def winning_score (total_score : ℕ) (score : ℕ) : Prop :=
  score ≤ total_score / 2

def min_winning_score (team_size : ℕ) : ℕ :=
  sum_of_positions team_size

theorem different_winning_scores (total_runners : ℕ) (team_size : ℕ) : 
  cross_country_meet total_runners →
  (winning_score (sum_of_positions total_runners) (sum_of_positions total_runners / 2) ∧
   min_winning_score team_size = sum_of_positions team_size) →
  (sum_of_positions total_runners / 2 - min_winning_score team_size + 1 = 19) :=
by sorry

end different_winning_scores_l1723_172304


namespace bowtie_equation_solution_l1723_172321

-- Define the bowties operation
noncomputable def bowtie (a b : ℝ) : ℝ := a + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt b^2)))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ h : ℝ, bowtie 5 h = 10 ∧ h = 2 * Real.sqrt 5 := by
  sorry

end bowtie_equation_solution_l1723_172321


namespace gcd_of_324_243_270_l1723_172373

theorem gcd_of_324_243_270 : Nat.gcd 324 (Nat.gcd 243 270) = 27 := by
  sorry

end gcd_of_324_243_270_l1723_172373


namespace pure_imaginary_fraction_l1723_172336

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (Complex.I : ℂ) * b = (a - 2 * Complex.I) / (1 + 2 * Complex.I)) → a = 4 := by
  sorry

end pure_imaginary_fraction_l1723_172336


namespace closer_to_cottage_l1723_172324

theorem closer_to_cottage (c m p : ℝ) 
  (hc : c > 0)
  (hm : m + 3/2 * (1/2 * m) = c)
  (hp : 2*p + 1/3 * (2*p) = c) : 
  m/c > p/c := by
sorry

end closer_to_cottage_l1723_172324


namespace vaccine_effective_l1723_172386

/-- Represents the contingency table for vaccine effectiveness study -/
structure VaccineStudy where
  total_mice : ℕ
  infected_mice : ℕ
  not_infected_mice : ℕ
  prob_infected_not_vaccinated : ℚ

/-- Calculates the chi-square statistic for the vaccine study -/
def chi_square (study : VaccineStudy) : ℚ :=
  let a := study.not_infected_mice - (study.total_mice / 2 - study.infected_mice * study.prob_infected_not_vaccinated)
  let b := study.not_infected_mice - a
  let c := study.total_mice / 2 - study.infected_mice * study.prob_infected_not_vaccinated
  let d := study.infected_mice - c
  let n := study.total_mice
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

/-- The critical value for 95% confidence in the chi-square test -/
def chi_square_critical : ℚ := 3841 / 1000

/-- Theorem stating that the vaccine is effective with 95% confidence -/
theorem vaccine_effective (study : VaccineStudy) 
  (h1 : study.total_mice = 200)
  (h2 : study.infected_mice = 100)
  (h3 : study.not_infected_mice = 100)
  (h4 : study.prob_infected_not_vaccinated = 3/5) :
  chi_square study > chi_square_critical := by
  sorry

end vaccine_effective_l1723_172386


namespace arithmetic_24_l1723_172374

def numbers : List ℕ := [8, 8, 8, 10]

inductive ArithExpr
  | Num : ℕ → ArithExpr
  | Add : ArithExpr → ArithExpr → ArithExpr
  | Sub : ArithExpr → ArithExpr → ArithExpr
  | Mul : ArithExpr → ArithExpr → ArithExpr
  | Div : ArithExpr → ArithExpr → ArithExpr

def eval : ArithExpr → ℕ
  | ArithExpr.Num n => n
  | ArithExpr.Add e1 e2 => eval e1 + eval e2
  | ArithExpr.Sub e1 e2 => eval e1 - eval e2
  | ArithExpr.Mul e1 e2 => eval e1 * eval e2
  | ArithExpr.Div e1 e2 => eval e1 / eval e2

def uses_all_numbers (expr : ArithExpr) (nums : List ℕ) : Prop := sorry

theorem arithmetic_24 : 
  ∃ (expr : ArithExpr), uses_all_numbers expr numbers ∧ eval expr = 24 :=
sorry

end arithmetic_24_l1723_172374


namespace mary_seashells_l1723_172352

theorem mary_seashells (jessica_seashells : ℕ) (total_seashells : ℕ) 
  (h1 : jessica_seashells = 41)
  (h2 : total_seashells = 59) :
  total_seashells - jessica_seashells = 18 :=
by sorry

end mary_seashells_l1723_172352


namespace custom_mul_result_l1723_172384

/-- Custom multiplication operation -/
def custom_mul (a b c : ℚ) (x y : ℚ) : ℚ := a * x + b * y + c

theorem custom_mul_result (a b c : ℚ) :
  custom_mul a b c 1 2 = 9 →
  custom_mul a b c (-3) 3 = 6 →
  custom_mul a b c 0 1 = 2 →
  custom_mul a b c (-2) 5 = 18 := by
sorry

end custom_mul_result_l1723_172384


namespace f_inequality_l1723_172355

open Real

/-- The function f(x) = x ln x -/
noncomputable def f (x : ℝ) : ℝ := x * log x

/-- The derivative of f(x) -/
noncomputable def f_deriv (x : ℝ) : ℝ := 1 + log x

theorem f_inequality (x₁ x₂ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : x₁ ≠ x₂) :
  (f x₂ - f x₁) / (x₂ - x₁) < f_deriv ((x₁ + x₂) / 2) :=
sorry

end f_inequality_l1723_172355


namespace ned_bomb_diffusion_l1723_172349

/-- Represents the problem of Ned racing to deactivate a time bomb --/
def bomb_diffusion_problem (total_flights : ℕ) (seconds_per_flight : ℕ) (bomb_timer : ℕ) (time_spent : ℕ) : Prop :=
  let total_time := total_flights * seconds_per_flight
  let remaining_time := total_time - time_spent
  let time_left := bomb_timer - remaining_time
  time_left = 84

/-- Theorem stating that Ned will have 84 seconds to diffuse the bomb --/
theorem ned_bomb_diffusion :
  bomb_diffusion_problem 40 13 58 273 := by
  sorry

#check ned_bomb_diffusion

end ned_bomb_diffusion_l1723_172349


namespace billy_brad_weight_difference_l1723_172350

-- Define the weights as natural numbers
def carl_weight : ℕ := 145
def billy_weight : ℕ := 159

-- Define Brad's weight in terms of Carl's
def brad_weight : ℕ := carl_weight + 5

-- State the theorem
theorem billy_brad_weight_difference :
  billy_weight - brad_weight = 9 :=
by sorry

end billy_brad_weight_difference_l1723_172350


namespace total_time_knife_and_vegetables_l1723_172323

/-- Proves that the total time spent on knife sharpening and vegetable peeling is 40 minutes -/
theorem total_time_knife_and_vegetables (knife_time vegetable_time total_time : ℕ) : 
  knife_time = 10 →
  vegetable_time = 3 * knife_time →
  total_time = knife_time + vegetable_time →
  total_time = 40 := by
  sorry

end total_time_knife_and_vegetables_l1723_172323


namespace m_range_l1723_172322

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, |x - 1| > m - 1
def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(5 - 2*m))^x > (-(5 - 2*m))^y

-- Define the theorem
theorem m_range (m : ℝ) : 
  ((p m ∨ q m) ∧ ¬(p m ∧ q m)) → (1 ≤ m ∧ m < 2) :=
by sorry

end m_range_l1723_172322


namespace yellow_flags_count_l1723_172387

/-- Represents the number of yellow flags in a cycle -/
def yellow_per_cycle : ℕ := 2

/-- Represents the length of the repeating cycle -/
def cycle_length : ℕ := 5

/-- Represents the total number of flags we're considering -/
def total_flags : ℕ := 200

/-- Theorem: The number of yellow flags in the first 200 flags is 80 -/
theorem yellow_flags_count : 
  (total_flags / cycle_length) * yellow_per_cycle = 80 := by
sorry

end yellow_flags_count_l1723_172387


namespace tourist_travel_time_l1723_172314

theorem tourist_travel_time (boat_distance : ℝ) (walk_distance : ℝ) 
  (h1 : boat_distance = 90) 
  (h2 : walk_distance = 10) : ∃ (walk_time boat_time : ℝ),
  walk_time + 4 = boat_time ∧ 
  walk_distance / walk_time * boat_time = boat_distance / boat_time * walk_time ∧
  walk_time = 2 ∧ 
  boat_time = 6 := by
  sorry

end tourist_travel_time_l1723_172314


namespace factorial_of_factorial_l1723_172343

theorem factorial_of_factorial (n : ℕ) : (n.factorial.factorial) / n.factorial = (n.factorial - 1).factorial := by
  sorry

end factorial_of_factorial_l1723_172343


namespace line_circle_intersection_l1723_172358

/-- A circle with center O and radius r -/
structure Circle where
  O : ℝ × ℝ
  r : ℝ

/-- A line l -/
structure Line where
  l : ℝ × ℝ → Prop

/-- The distance from a point to a line -/
def distancePointToLine (p : ℝ × ℝ) (l : Line) : ℝ :=
  sorry

/-- Determines if a line intersects a circle -/
def intersects (c : Circle) (l : Line) : Prop :=
  distancePointToLine c.O l < c.r

theorem line_circle_intersection (c : Circle) (l : Line) :
  distancePointToLine c.O l < c.r → intersects c l :=
by sorry

end line_circle_intersection_l1723_172358


namespace simplify_expression_l1723_172335

theorem simplify_expression (w : ℝ) : 
  4*w + 6*w + 8*w + 10*w + 12*w + 14*w + 16 = 54*w + 16 := by
  sorry

end simplify_expression_l1723_172335


namespace ball_distribution_problem_l1723_172393

/-- The number of ways to distribute n identical objects into k distinct boxes --/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of ways to distribute n identical balls into 3 distinct boxes,
    where box i must contain at least i balls (for i = 1, 2, 3) --/
def distributeWithMinimum (n : ℕ) : ℕ :=
  distribute (n - (1 + 2 + 3)) 3

theorem ball_distribution_problem :
  distributeWithMinimum 10 = 15 := by sorry

end ball_distribution_problem_l1723_172393


namespace chocolate_bars_in_box_l1723_172325

/-- The weight of a single chocolate bar in grams -/
def bar_weight : ℕ := 125

/-- The weight of the box in kilograms -/
def box_weight : ℕ := 2

/-- The number of chocolate bars in the box -/
def num_bars : ℕ := (box_weight * 1000) / bar_weight

/-- Theorem stating that the number of chocolate bars in the box is 16 -/
theorem chocolate_bars_in_box : num_bars = 16 := by
  sorry

end chocolate_bars_in_box_l1723_172325


namespace sine_cosine_relation_l1723_172320

theorem sine_cosine_relation (x : ℝ) : 
  Real.sin (2 * x + π / 6) = -1 / 3 → Real.cos (π / 3 - 2 * x) = -1 / 3 := by
  sorry

end sine_cosine_relation_l1723_172320


namespace wednesday_earnings_l1723_172370

/-- Represents the earnings from selling cabbage over three days -/
structure CabbageEarnings where
  wednesday : ℝ
  friday : ℝ
  today : ℝ
  total_kg : ℝ
  price_per_kg : ℝ

/-- Theorem stating that given the conditions, Johannes earned $30 on Wednesday -/
theorem wednesday_earnings (e : CabbageEarnings) 
  (h1 : e.friday = 24)
  (h2 : e.today = 42)
  (h3 : e.total_kg = 48)
  (h4 : e.price_per_kg = 2)
  (h5 : e.wednesday + e.friday + e.today = e.total_kg * e.price_per_kg) :
  e.wednesday = 30 := by
  sorry

end wednesday_earnings_l1723_172370


namespace imaginary_power_l1723_172369

theorem imaginary_power (i : ℂ) : i^2 = -1 → i^2015 = -i := by
  sorry

end imaginary_power_l1723_172369


namespace square_cut_perimeter_l1723_172312

/-- The perimeter of a figure formed by cutting a square and rearranging it -/
theorem square_cut_perimeter (s : ℝ) (h : s = 100) : 
  let rect_length : ℝ := s
  let rect_width : ℝ := s / 2
  let perimeter : ℝ := 3 * rect_length + 4 * rect_width
  perimeter = 500 := by sorry

end square_cut_perimeter_l1723_172312


namespace max_value_expression_l1723_172334

theorem max_value_expression (x y : ℝ) :
  (3 * x + 4 * y + 5) / Real.sqrt (x^2 + y^2 + 4) ≤ Real.sqrt 50 := by
  sorry

end max_value_expression_l1723_172334


namespace pure_imaginary_solution_l1723_172315

/-- A complex number is pure imaginary if its real part is zero -/
def IsPureImaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_solution (z : ℂ) (a : ℝ) 
  (h1 : IsPureImaginary z) 
  (h2 : (1 - Complex.I) * z = 1 + a * Complex.I) : 
  a = 1 := by
  sorry

end pure_imaginary_solution_l1723_172315


namespace bunny_burrow_exits_l1723_172375

/-- Represents the total number of bunnies in the community -/
def total_bunnies : ℕ := 100

/-- Represents the number of bunnies in Group A -/
def group_a_bunnies : ℕ := 40

/-- Represents the number of bunnies in Group B -/
def group_b_bunnies : ℕ := 30

/-- Represents the number of bunnies in Group C -/
def group_c_bunnies : ℕ := 30

/-- Represents how many times a bunny in Group A comes out per minute -/
def group_a_frequency : ℚ := 3

/-- Represents how many times a bunny in Group B comes out per minute -/
def group_b_frequency : ℚ := 5 / 2

/-- Represents how many times a bunny in Group C comes out per minute -/
def group_c_frequency : ℚ := 8 / 5

/-- Represents the reduction factor in burrow-exiting behavior after environmental change -/
def reduction_factor : ℚ := 1 / 2

/-- Represents the number of weeks before environmental change -/
def weeks_before_change : ℕ := 1

/-- Represents the number of weeks after environmental change -/
def weeks_after_change : ℕ := 2

/-- Represents the total number of weeks -/
def total_weeks : ℕ := 3

/-- Represents the number of minutes in a day -/
def minutes_per_day : ℕ := 1440

/-- Represents the number of days in a week -/
def days_per_week : ℕ := 7

/-- Theorem stating that the combined number of times all bunnies come out during 3 weeks is 4,897,920 -/
theorem bunny_burrow_exits : 
  (group_a_bunnies * group_a_frequency + 
   group_b_bunnies * group_b_frequency + 
   group_c_bunnies * group_c_frequency) * 
  minutes_per_day * days_per_week * weeks_before_change +
  (group_a_bunnies * group_a_frequency + 
   group_b_bunnies * group_b_frequency + 
   group_c_bunnies * group_c_frequency) * 
  reduction_factor * 
  minutes_per_day * days_per_week * weeks_after_change = 4897920 := by
  sorry

end bunny_burrow_exits_l1723_172375


namespace pizza_combinations_l1723_172316

def num_toppings : ℕ := 8

def one_topping_pizzas (n : ℕ) : ℕ := n

def two_topping_pizzas (n : ℕ) : ℕ := n.choose 2

def three_topping_pizzas (n : ℕ) : ℕ := n.choose 3

theorem pizza_combinations :
  one_topping_pizzas num_toppings +
  two_topping_pizzas num_toppings +
  three_topping_pizzas num_toppings = 92 := by
  sorry

end pizza_combinations_l1723_172316


namespace cow_horse_ratio_l1723_172379

theorem cow_horse_ratio (total : ℕ) (cows : ℕ) (horses : ℕ) (h1 : total = 168) (h2 : cows = 140) (h3 : total = cows + horses) (h4 : ∃ r : ℕ, cows = r * horses) : 
  cows / horses = 5 := by
  sorry

end cow_horse_ratio_l1723_172379


namespace sphere_radius_ratio_l1723_172391

theorem sphere_radius_ratio (V_large V_small : ℝ) (h1 : V_large = 500 * Real.pi) (h2 : V_small = 40 * Real.pi) :
  (((3 * V_small) / (4 * Real.pi)) ^ (1/3)) / (((3 * V_large) / (4 * Real.pi)) ^ (1/3)) = (10 ^ (1/3)) / 5 := by
  sorry

end sphere_radius_ratio_l1723_172391


namespace absolute_value_equation_l1723_172344

theorem absolute_value_equation (x : ℝ) :
  |x - 25| + |x - 15| = |2*x - 40| ↔ x ≤ 15 ∨ x ≥ 25 := by
sorry

end absolute_value_equation_l1723_172344


namespace complex_purely_imaginary_l1723_172383

theorem complex_purely_imaginary (z : ℂ) : 
  (∃ y : ℝ, z = y * I) →  -- z is purely imaginary
  (∃ w : ℝ, (z + 2)^2 - 8*I = w * I) →  -- (z + 2)² - 8i is purely imaginary
  z = -2 * I :=  -- z = -2i
by sorry

end complex_purely_imaginary_l1723_172383


namespace f_properties_l1723_172342

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| - |x - 2|

-- Theorem stating the properties of f and the inequality
theorem f_properties :
  (∃ (t : ℝ), t = 3 ∧ ∀ x, f x ≤ t) ∧
  (∀ x, x ≥ 2 → f x = 3) ∧
  (∀ a b : ℝ, a^2 + 2*b = 1 → 2*a^2 + b^2 ≥ 1/4) :=
by sorry

end f_properties_l1723_172342


namespace complex_magnitude_theorem_l1723_172338

theorem complex_magnitude_theorem (r : ℝ) (z : ℂ) 
  (h1 : |r| < 6) 
  (h2 : z + 9 / z = r) : 
  Complex.abs z = 3 := by
sorry

end complex_magnitude_theorem_l1723_172338


namespace cab_driver_income_theorem_l1723_172318

/-- Represents the weather condition for a day --/
inductive Weather
  | Sunny
  | Rainy
  | Cloudy

/-- Represents a day's income data --/
structure DayData where
  income : ℝ
  weather : Weather
  isPeakHours : Bool

/-- Calculates the adjusted income for a day based on weather and peak hours --/
def adjustedIncome (day : DayData) : ℝ :=
  match day.weather with
  | Weather.Rainy => day.income * 1.1
  | Weather.Cloudy => day.income * 0.95
  | Weather.Sunny => 
    if day.isPeakHours then day.income * 1.2
    else day.income

/-- The income data for 12 days --/
def incomeData : List DayData := [
  ⟨200, Weather.Rainy, false⟩,
  ⟨150, Weather.Sunny, false⟩,
  ⟨750, Weather.Sunny, false⟩,
  ⟨400, Weather.Sunny, false⟩,
  ⟨500, Weather.Cloudy, false⟩,
  ⟨300, Weather.Rainy, false⟩,
  ⟨650, Weather.Sunny, false⟩,
  ⟨350, Weather.Cloudy, false⟩,
  ⟨600, Weather.Sunny, true⟩,
  ⟨450, Weather.Sunny, false⟩,
  ⟨530, Weather.Sunny, false⟩,
  ⟨480, Weather.Cloudy, false⟩
]

theorem cab_driver_income_theorem :
  let totalIncome := (incomeData.map adjustedIncome).sum
  let averageIncome := totalIncome / incomeData.length
  totalIncome = 4963.5 ∧ averageIncome = 413.625 := by
  sorry


end cab_driver_income_theorem_l1723_172318


namespace parallel_line_through_point_line_equation_proof_l1723_172307

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space in the form ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point2D) (l : Line2D) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def parallelLines (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point 
  (givenLine : Line2D) 
  (point : Point2D) 
  (resultLine : Line2D) : Prop :=
  parallelLines givenLine resultLine ∧ 
  pointOnLine point resultLine

/-- The main theorem to prove -/
theorem line_equation_proof : 
  let givenLine : Line2D := { a := 2, b := 3, c := 5 }
  let point : Point2D := { x := 1, y := -4 }
  let resultLine : Line2D := { a := 2, b := 3, c := 10 }
  parallel_line_through_point givenLine point resultLine := by
  sorry

end parallel_line_through_point_line_equation_proof_l1723_172307


namespace crayon_calculation_l1723_172302

/-- Calculates the final number of crayons and their percentage of the total items -/
theorem crayon_calculation (initial_crayons : ℕ) (initial_pencils : ℕ) 
  (removed_crayons : ℕ) (added_crayons : ℕ) (increase_percentage : ℚ) :
  initial_crayons = 41 →
  initial_pencils = 26 →
  removed_crayons = 8 →
  added_crayons = 12 →
  increase_percentage = 1/10 →
  let intermediate_crayons := initial_crayons - removed_crayons + added_crayons
  let final_crayons := (intermediate_crayons : ℚ) * (1 + increase_percentage)
  let rounded_final_crayons := round final_crayons
  let total_items := rounded_final_crayons + initial_pencils
  let percentage_crayons := (rounded_final_crayons : ℚ) / (total_items : ℚ) * 100
  rounded_final_crayons = 50 ∧ abs (percentage_crayons - 65.79) < 0.01 := by
  sorry

end crayon_calculation_l1723_172302


namespace volunteers_in_2002_l1723_172301

/-- The number of volunteers after n years, given an initial number and annual increase rate -/
def volunteers (initial : ℕ) (rate : ℚ) (years : ℕ) : ℚ :=
  initial * (1 + rate) ^ years

/-- Theorem: The number of volunteers in 2002 will be 6075, given the initial conditions -/
theorem volunteers_in_2002 :
  volunteers 1200 (1/2) 4 = 6075 := by
  sorry

#eval volunteers 1200 (1/2) 4

end volunteers_in_2002_l1723_172301


namespace clock_malfunction_l1723_172327

/-- Represents a time in HH:MM format -/
structure Time where
  hours : Nat
  minutes : Nat
  hh_valid : hours < 24
  mm_valid : minutes < 60

/-- Represents a malfunctioning clock where each digit either increases or decreases by 1 -/
def is_malfunctioned (original : Time) (displayed : Time) : Prop :=
  (displayed.hours / 10 = original.hours / 10 + 1 ∨ displayed.hours / 10 = original.hours / 10 - 1) ∧
  (displayed.hours % 10 = (original.hours % 10 + 1) % 10 ∨ displayed.hours % 10 = (original.hours % 10 - 1 + 10) % 10) ∧
  (displayed.minutes / 10 = original.minutes / 10 + 1 ∨ displayed.minutes / 10 = original.minutes / 10 - 1) ∧
  (displayed.minutes % 10 = (original.minutes % 10 + 1) % 10 ∨ displayed.minutes % 10 = (original.minutes % 10 - 1 + 10) % 10)

theorem clock_malfunction (displayed : Time) (h_displayed : displayed.hours = 20 ∧ displayed.minutes = 9) :
  ∃ (original : Time), is_malfunctioned original displayed ∧ original.hours = 11 ∧ original.minutes = 18 := by
  sorry

end clock_malfunction_l1723_172327


namespace women_per_table_women_per_table_solution_l1723_172396

theorem women_per_table (num_tables : ℕ) (men_per_table : ℕ) (total_customers : ℕ) : ℕ :=
  let total_men := num_tables * men_per_table
  let total_women := total_customers - total_men
  total_women / num_tables

theorem women_per_table_solution :
  women_per_table 9 3 90 = 7 := by
  sorry

end women_per_table_women_per_table_solution_l1723_172396


namespace simplify_expression_l1723_172330

theorem simplify_expression (x y z : ℚ) (hx : x = 3) (hy : y = 2) (hz : z = 4) :
  18 * x^3 * y^2 * z^2 / (9 * x^2 * y * z^3) = 3 := by
  sorry

end simplify_expression_l1723_172330


namespace subtraction_inequality_l1723_172399

theorem subtraction_inequality (a b c : ℝ) (h : a > b) : c - a < c - b := by
  sorry

end subtraction_inequality_l1723_172399


namespace quadratic_power_function_l1723_172378

/-- A function is a power function if it's of the form f(x) = x^a for some real number a -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, ∀ x : ℝ, f x = x^a

/-- A function is quadratic if it's of the form f(x) = ax^2 + bx + c for some real numbers a, b, c with a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x : ℝ, f x = a * x^2 + b * x + c

theorem quadratic_power_function (f : ℝ → ℝ) :
  IsQuadratic f ∧ IsPowerFunction f → ∀ x : ℝ, f x = x^2 := by
  sorry

end quadratic_power_function_l1723_172378


namespace min_distinct_lines_for_31_segments_l1723_172382

/-- Represents a non-self-intersecting open polyline on a plane -/
structure OpenPolyline where
  segments : ℕ
  non_self_intersecting : Bool
  consecutive_segments_not_collinear : Bool

/-- The minimum number of distinct lines needed to contain all segments of the polyline -/
def min_distinct_lines (p : OpenPolyline) : ℕ :=
  sorry

/-- Theorem stating the minimum number of distinct lines for a specific polyline -/
theorem min_distinct_lines_for_31_segments (p : OpenPolyline) :
  p.segments = 31 ∧ p.non_self_intersecting ∧ p.consecutive_segments_not_collinear →
  min_distinct_lines p = 9 :=
sorry

end min_distinct_lines_for_31_segments_l1723_172382


namespace trapezoid_area_sum_properties_l1723_172356

/-- Represents a trapezoid with four side lengths -/
structure Trapezoid where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- Calculates the sum of all possible areas of a trapezoid -/
def sum_of_areas (t : Trapezoid) : ℝ := sorry

/-- Checks if a number is not divisible by the square of any prime -/
def not_divisible_by_square_prime (n : ℕ) : Prop := sorry

/-- Theorem stating the properties of the sum of areas for the given trapezoid -/
theorem trapezoid_area_sum_properties :
  ∃ (r₁ r₂ r₃ : ℚ) (n₁ n₂ : ℕ),
    let t := Trapezoid.mk 4 6 8 10
    sum_of_areas t = r₁ * Real.sqrt n₁ + r₂ * Real.sqrt n₂ + r₃ ∧
    not_divisible_by_square_prime n₁ ∧
    not_divisible_by_square_prime n₂ ∧
    ⌊r₁ + r₂ + r₃ + n₁ + n₂⌋ = 742 := by
  sorry

end trapezoid_area_sum_properties_l1723_172356


namespace two_suits_cost_l1723_172365

def off_the_rack_cost : ℕ := 300
def tailoring_cost : ℕ := 200

def total_cost (off_the_rack : ℕ) (tailoring : ℕ) : ℕ :=
  off_the_rack + (3 * off_the_rack + tailoring)

theorem two_suits_cost :
  total_cost off_the_rack_cost tailoring_cost = 1400 := by
  sorry

end two_suits_cost_l1723_172365


namespace pond_volume_l1723_172395

/-- The volume of a rectangular prism given its length, width, and height -/
def volume (length width height : ℝ) : ℝ := length * width * height

/-- Theorem: The volume of a rectangular prism with dimensions 20 m × 10 m × 8 m is 1600 cubic meters -/
theorem pond_volume : volume 20 10 8 = 1600 := by
  sorry

end pond_volume_l1723_172395


namespace books_total_is_54_l1723_172376

/-- The total number of books Darla, Katie, and Gary have -/
def total_books (darla_books katie_books gary_books : ℕ) : ℕ :=
  darla_books + katie_books + gary_books

/-- Theorem stating the total number of books is 54 -/
theorem books_total_is_54 :
  ∀ (darla_books katie_books gary_books : ℕ),
    darla_books = 6 →
    katie_books = darla_books / 2 →
    gary_books = 5 * (darla_books + katie_books) →
    total_books darla_books katie_books gary_books = 54 :=
by
  sorry

end books_total_is_54_l1723_172376


namespace sum_of_max_min_g_l1723_172398

def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |2*x - 8| + 3

theorem sum_of_max_min_g : 
  ∃ (max min : ℝ), 
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 7 → g x ≤ max) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 7 ∧ g x = max) ∧
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 7 → min ≤ g x) ∧
    (∃ x : ℝ, 1 ≤ x ∧ x ≤ 7 ∧ g x = min) ∧
    max + min = 18 :=
by sorry

end sum_of_max_min_g_l1723_172398


namespace tom_marble_groups_l1723_172353

/-- Represents the types of marbles Tom has --/
inductive MarbleType
  | Red
  | Blue
  | Green
  | Yellow

/-- Represents Tom's marble collection --/
structure MarbleCollection where
  red : Nat
  blue : Nat
  green : Nat
  yellow : Nat

/-- Counts the number of different groups of 3 marbles that can be chosen --/
def countDifferentGroups (collection : MarbleCollection) : Nat :=
  sorry

/-- Theorem stating that Tom can choose 8 different groups of 3 marbles --/
theorem tom_marble_groups (tom_marbles : MarbleCollection) 
  (h_red : tom_marbles.red = 1)
  (h_blue : tom_marbles.blue = 1)
  (h_green : tom_marbles.green = 2)
  (h_yellow : tom_marbles.yellow = 3) :
  countDifferentGroups tom_marbles = 8 := by
  sorry

end tom_marble_groups_l1723_172353


namespace smallest_four_digit_palindrome_div_by_3_proof_l1723_172347

/-- A function that checks if a number is a four-digit palindrome -/
def is_four_digit_palindrome (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ (n / 1000 = n % 10) ∧ ((n / 100) % 10 = (n / 10) % 10)

/-- The smallest four-digit palindrome divisible by 3 -/
def smallest_four_digit_palindrome_div_by_3 : ℕ := 2112

theorem smallest_four_digit_palindrome_div_by_3_proof :
  is_four_digit_palindrome smallest_four_digit_palindrome_div_by_3 ∧
  smallest_four_digit_palindrome_div_by_3 % 3 = 0 ∧
  ∀ n : ℕ, is_four_digit_palindrome n ∧ n % 3 = 0 → n ≥ smallest_four_digit_palindrome_div_by_3 := by
  sorry

end smallest_four_digit_palindrome_div_by_3_proof_l1723_172347


namespace probability_is_one_fourth_l1723_172333

/-- The number of pennies Keiko tosses -/
def keiko_pennies : ℕ := 1

/-- The number of pennies Ephraim tosses -/
def ephraim_pennies : ℕ := 3

/-- The total number of possible outcomes when tossing n pennies -/
def total_outcomes (n : ℕ) : ℕ := 2^n

/-- The number of favorable outcomes where Ephraim gets the same number of heads as Keiko -/
def favorable_outcomes : ℕ := 4

/-- The probability that Ephraim gets the same number of heads as Keiko -/
def probability : ℚ := favorable_outcomes / (total_outcomes keiko_pennies * total_outcomes ephraim_pennies)

theorem probability_is_one_fourth : probability = 1/4 := by
  sorry

end probability_is_one_fourth_l1723_172333


namespace equal_area_division_l1723_172364

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a quadrilateral -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Calculates the area of a quadrilateral using the shoelace formula -/
def area (q : Quadrilateral) : ℚ :=
  let det := q.A.x * q.B.y + q.B.x * q.C.y + q.C.x * q.D.y + q.D.x * q.A.y -
             (q.B.x * q.A.y + q.C.x * q.B.y + q.D.x * q.C.y + q.A.x * q.D.y)
  (1/2) * abs det

/-- Represents the intersection point of the dividing line with CD -/
structure IntersectionPoint where
  p : ℤ
  q : ℤ
  r : ℤ
  s : ℤ

/-- The main theorem -/
theorem equal_area_division (q : Quadrilateral) (i : IntersectionPoint) :
  q.A = ⟨0, 0⟩ →
  q.B = ⟨0, 3⟩ →
  q.C = ⟨4, 4⟩ →
  q.D = ⟨5, 0⟩ →
  area { A := q.A, B := q.B, C := ⟨i.p / i.q, i.r / i.s⟩, D := q.D } = 
  area { A := q.A, B := ⟨i.p / i.q, i.r / i.s⟩, C := q.C, D := q.D } →
  i.p + i.q + i.r + i.s = 13 := by sorry

end equal_area_division_l1723_172364


namespace shaded_area_circle_and_tangents_l1723_172332

theorem shaded_area_circle_and_tangents (r : ℝ) (θ : ℝ) :
  r = 3 →
  θ = Real.pi / 3 →
  let circle_area := π * r^2
  let sector_angle := 2 * θ
  let sector_area := (sector_angle / (2 * Real.pi)) * circle_area
  let triangle_area := r^2 * Real.tan θ
  sector_area + 2 * triangle_area = 6 * π + 9 * Real.sqrt 3 :=
by sorry

end shaded_area_circle_and_tangents_l1723_172332


namespace necessary_but_not_sufficient_condition_l1723_172381

theorem necessary_but_not_sufficient_condition :
  ∀ x : ℝ, (2 * x^2 - 5 * x - 3 ≥ 0) → (x < 0 ∨ x > 2) ∧
  ∃ y : ℝ, (y < 0 ∨ y > 2) ∧ ¬(2 * y^2 - 5 * y - 3 ≥ 0) :=
by sorry

end necessary_but_not_sufficient_condition_l1723_172381


namespace subtraction_error_correction_l1723_172389

theorem subtraction_error_correction (x y : ℕ) 
  (h1 : x - y = 8008)
  (h2 : x - 10 * y = 88) :
  x = 8888 := by
  sorry

end subtraction_error_correction_l1723_172389


namespace x_shape_is_line_segments_l1723_172306

/-- The shape defined by θ = π/4 or θ = 5π/4 within 2 units of the origin -/
def X_shape : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1^2 + p.2^2 ≤ 4) ∧ 
    (p.2 = p.1 ∨ p.2 = -p.1) ∧ 
    (p.1 ≠ 0 ∨ p.2 ≠ 0)}

theorem x_shape_is_line_segments : 
  ∃ (a b c d : ℝ × ℝ), 
    a ≠ b ∧ c ≠ d ∧
    X_shape = {p : ℝ × ℝ | ∃ (t : ℝ), (0 ≤ t ∧ t ≤ 1 ∧ 
      ((p = (1 - t) • a + t • b) ∨ (p = (1 - t) • c + t • d)))} :=
sorry

end x_shape_is_line_segments_l1723_172306


namespace initial_machines_l1723_172329

/-- The number of machines working initially -/
def N : ℕ := sorry

/-- The number of units produced by N machines in 5 days -/
def x : ℝ := sorry

/-- Machines work at a constant rate -/
axiom constant_rate : ∀ (m : ℕ) (u t : ℝ), m ≠ 0 → t ≠ 0 → u / (m * t) = x / (N * 5)

theorem initial_machines :
  N * (x / 5) = 12 * (x / 30) → N = 2 :=
sorry

end initial_machines_l1723_172329


namespace sqrt_224_range_l1723_172366

theorem sqrt_224_range : 14 < Real.sqrt 224 ∧ Real.sqrt 224 < 15 := by
  sorry

end sqrt_224_range_l1723_172366


namespace cos_theta_equals_sqrt2_over_2_l1723_172367

/-- Given vectors a and b with an angle θ between them, 
    if a = (1,1) and b - a = (-1,1), then cos θ = √2/2 -/
theorem cos_theta_equals_sqrt2_over_2 (a b : ℝ × ℝ) (θ : ℝ) :
  a = (1, 1) →
  b - a = (-1, 1) →
  let cos_theta := (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))
  cos_theta = Real.sqrt 2 / 2 := by
sorry

end cos_theta_equals_sqrt2_over_2_l1723_172367


namespace right_triangle_tan_y_l1723_172348

theorem right_triangle_tan_y (X Y Z : ℝ × ℝ) :
  -- Right triangle condition
  (Y.1 - X.1) * (Z.2 - X.2) = (Z.1 - X.1) * (Y.2 - X.2) →
  -- XY = 30 condition
  Real.sqrt ((Y.1 - X.1)^2 + (Y.2 - X.2)^2) = 30 →
  -- XZ = 40 condition (derived from the solution)
  Real.sqrt ((Z.1 - X.1)^2 + (Z.2 - X.2)^2) = 40 →
  -- Conclusion: tan Y = 4/3
  (Z.2 - X.2) / (Y.1 - X.1) = 4 / 3 :=
by
  sorry


end right_triangle_tan_y_l1723_172348


namespace complex_power_4_30_degrees_l1723_172305

theorem complex_power_4_30_degrees : 
  (2 * Complex.cos (π / 6) + 2 * Complex.I * Complex.sin (π / 6)) ^ 4 = -8 + 8 * Complex.I * Real.sqrt 3 := by
  sorry

end complex_power_4_30_degrees_l1723_172305


namespace smallest_upper_bound_l1723_172397

theorem smallest_upper_bound (x : ℤ) 
  (h1 : 5 < x ∧ x < 21)
  (h2 : 7 < x)
  (h3 : 13 > x ∧ x > 2)
  (h4 : 12 > x ∧ x > 9)
  (h5 : x + 1 < 13) :
  ∀ y : ℤ, x < y → 12 ≤ y := by
  sorry

#check smallest_upper_bound

end smallest_upper_bound_l1723_172397


namespace rectangle_area_l1723_172310

theorem rectangle_area (a b c : ℝ) (h : a > 0) (h' : b > 0) (h'' : c > 0) 
  (h_pythagorean : a^2 + b^2 = c^2) : a * b = a * b :=
by sorry

end rectangle_area_l1723_172310


namespace min_value_on_interval_l1723_172345

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- State the theorem
theorem min_value_on_interval (a : ℝ) :
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≥ f a y) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f a x = 20) →
  ∃ x ∈ Set.Icc (-2 : ℝ) 2, ∀ y ∈ Set.Icc (-2 : ℝ) 2, f a x ≤ f a y ∧ f a x = -7 :=
by sorry

end min_value_on_interval_l1723_172345


namespace range_of_a_l1723_172357

theorem range_of_a (e : ℝ) (h_e : e = Real.exp 1) :
  ∀ a : ℝ, (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 2 * x + a * (y - 2 * e * x) * (Real.log y - Real.log x) = 0) ↔
  a < 0 ∨ a ≥ 2 / e := by
sorry

end range_of_a_l1723_172357


namespace bryan_pushups_l1723_172372

/-- The number of push-up sets Bryan does -/
def total_sets : ℕ := 15

/-- The number of push-ups Bryan intends to do in each set -/
def pushups_per_set : ℕ := 18

/-- The number of push-ups Bryan doesn't do in the last set due to exhaustion -/
def missed_pushups : ℕ := 12

/-- The actual number of push-ups Bryan does in the last set -/
def last_set_pushups : ℕ := pushups_per_set - missed_pushups

/-- The total number of push-ups Bryan does -/
def total_pushups : ℕ := (total_sets - 1) * pushups_per_set + last_set_pushups

theorem bryan_pushups : total_pushups = 258 := by
  sorry

end bryan_pushups_l1723_172372


namespace problem_statement_l1723_172371

theorem problem_statement : 
  let N := (Real.sqrt (Real.sqrt 6 + 3) + Real.sqrt (Real.sqrt 6 - 3)) / Real.sqrt (Real.sqrt 6 + 2) - Real.sqrt (4 - 2 * Real.sqrt 3)
  N = -1 := by sorry

end problem_statement_l1723_172371


namespace janessa_baseball_cards_l1723_172351

/-- Janessa's baseball card collection problem -/
theorem janessa_baseball_cards
  (initial_cards : ℕ)
  (father_cards : ℕ)
  (ebay_cards : ℕ)
  (bad_cards : ℕ)
  (cards_given_to_dexter : ℕ)
  (h1 : initial_cards = 4)
  (h2 : father_cards = 13)
  (h3 : ebay_cards = 36)
  (h4 : bad_cards = 4)
  (h5 : cards_given_to_dexter = 29) :
  initial_cards + father_cards + ebay_cards - bad_cards - cards_given_to_dexter = 20 := by
  sorry

#check janessa_baseball_cards

end janessa_baseball_cards_l1723_172351


namespace distance_home_to_school_l1723_172388

/-- Represents the scenario of a boy traveling between home and school. -/
structure TravelScenario where
  speed : ℝ  -- Speed in km/hr
  time_diff : ℝ  -- Time difference in hours (positive for late, negative for early)

/-- The distance between home and school satisfies the given travel scenarios. -/
def distance_satisfies (d : ℝ) (s1 s2 : TravelScenario) : Prop :=
  ∃ t : ℝ, 
    d = s1.speed * (t + s1.time_diff) ∧
    d = s2.speed * (t - s2.time_diff)

/-- The theorem stating the distance between home and school. -/
theorem distance_home_to_school : 
  ∃ d : ℝ, d = 1.5 ∧ 
    distance_satisfies d 
      { speed := 3, time_diff := 7/60 }
      { speed := 6, time_diff := -8/60 } := by
  sorry

end distance_home_to_school_l1723_172388


namespace total_notes_l1723_172319

/-- Calculates the total number of notes on a communal board -/
theorem total_notes (red_rows : Nat) (red_per_row : Nat) (blue_per_red : Nat) (extra_blue : Nat) :
  red_rows = 5 →
  red_per_row = 6 →
  blue_per_red = 2 →
  extra_blue = 10 →
  red_rows * red_per_row + red_rows * red_per_row * blue_per_red + extra_blue = 100 := by
  sorry


end total_notes_l1723_172319


namespace parabola_intersection_midpoint_l1723_172313

/-- Given two parabolas that intersect at points A and B, prove that if the sum of the x-coordinate
    and y-coordinate of the midpoint of AB is 2017, then c = 4031. -/
theorem parabola_intersection_midpoint (c : ℝ) : 
  let f (x : ℝ) := x^2 - 2*x - 3
  let g (x : ℝ) := -x^2 + 4*x + c
  ∃ A B : ℝ × ℝ, 
    (f A.1 = A.2 ∧ g A.1 = A.2) ∧ 
    (f B.1 = B.2 ∧ g B.1 = B.2) ∧
    ((A.1 + B.1) / 2 + (A.2 + B.2) / 2 = 2017) →
  c = 4031 := by
  sorry

end parabola_intersection_midpoint_l1723_172313


namespace sqrt_equation_solution_l1723_172326

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (6 - 5 * z) = 7 :=
by
  -- The unique solution is z = -43/5
  use -43/5
  constructor
  · -- Prove that -43/5 satisfies the equation
    sorry
  · -- Prove that any solution must equal -43/5
    sorry

#check sqrt_equation_solution

end sqrt_equation_solution_l1723_172326


namespace unit_digit_of_15_power_l1723_172339

theorem unit_digit_of_15_power (X : ℕ+) : ∃ n : ℕ, 15^(X : ℕ) ≡ 5 [MOD 10] :=
sorry

end unit_digit_of_15_power_l1723_172339


namespace palindrome_square_base_l1723_172354

theorem palindrome_square_base (r : ℕ) (x : ℕ) : 
  x = r^3 + r^2 + r + 1 →
  Even r →
  ∃ (a b c d : ℕ), 
    (x^2 = a*r^7 + b*r^6 + c*r^5 + d*r^4 + d*r^3 + c*r^2 + b*r + a) ∧
    (b + c = 24) →
  r = 26 :=
sorry

end palindrome_square_base_l1723_172354


namespace rectangle_difference_l1723_172385

theorem rectangle_difference (x y : ℝ) : 
  y = x / 3 →
  2 * x + 2 * y = 32 →
  x^2 + y^2 = 17^2 →
  x - y = 8 := by
sorry

end rectangle_difference_l1723_172385


namespace contractor_problem_l1723_172363

/-- Represents the problem of calculating the number of days to complete 1/4 of the work --/
theorem contractor_problem (total_days : ℕ) (initial_workers : ℕ) (remaining_days : ℕ) (fired_workers : ℕ) :
  total_days = 100 →
  initial_workers = 10 →
  remaining_days = 75 →
  fired_workers = 2 →
  let remaining_workers := initial_workers - fired_workers
  let work_per_day := (1 : ℚ) / initial_workers
  let days_to_quarter := (1 / 4 : ℚ) / work_per_day
  let remaining_work := (3 / 4 : ℚ) / (remaining_workers : ℚ) / (remaining_days : ℚ)
  (1 : ℚ) = days_to_quarter * work_per_day + remaining_work * (remaining_workers : ℚ) * (remaining_days : ℚ) →
  days_to_quarter = 20 := by
  sorry


end contractor_problem_l1723_172363


namespace females_with_advanced_degrees_l1723_172361

/-- Proves the number of females with advanced degrees in a company --/
theorem females_with_advanced_degrees 
  (total_employees : ℕ)
  (total_females : ℕ)
  (employees_with_advanced_degrees : ℕ)
  (males_with_college_only : ℕ)
  (h1 : total_employees = 180)
  (h2 : total_females = 110)
  (h3 : employees_with_advanced_degrees = 90)
  (h4 : males_with_college_only = 35) :
  total_females - (total_employees - employees_with_advanced_degrees - males_with_college_only) = 55 :=
by sorry

end females_with_advanced_degrees_l1723_172361


namespace simplify_and_evaluate_l1723_172360

theorem simplify_and_evaluate (a : ℝ) (h : a = 2023) :
  (a + 1) / a / (a - 1 / a) = 1 / 2022 := by
sorry

end simplify_and_evaluate_l1723_172360


namespace successive_discounts_l1723_172390

theorem successive_discounts (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) :
  let price_after_discount1 := initial_price * (1 - discount1)
  let final_price := price_after_discount1 * (1 - discount2)
  discount1 = 0.1 ∧ discount2 = 0.2 →
  final_price / initial_price = 0.72 := by
sorry

end successive_discounts_l1723_172390
