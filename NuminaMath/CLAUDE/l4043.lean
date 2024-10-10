import Mathlib

namespace perpendicular_bisector_equation_l4043_404323

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Define the centers of the circles
def center1 : ℝ × ℝ := (1, -2)
def center2 : ℝ × ℝ := (2, 0)

-- Define the equation of the line connecting the centers
def connecting_line (x y : ℝ) : Prop := 2*x - y - 4 = 0

-- Theorem statement
theorem perpendicular_bisector_equation :
  connecting_line (Prod.fst center1) (Prod.snd center1) ∧
  connecting_line (Prod.fst center2) (Prod.snd center2) :=
sorry

end perpendicular_bisector_equation_l4043_404323


namespace hemisphere_properties_l4043_404378

/-- Properties of a hemisphere with base area 144π -/
theorem hemisphere_properties :
  ∀ (r : ℝ),
  r > 0 →
  π * r^2 = 144 * π →
  (2 * π * r^2 + π * r^2 = 432 * π) ∧
  ((2 / 3) * π * r^3 = 1152 * π) := by
  sorry

end hemisphere_properties_l4043_404378


namespace nesbitt_like_inequality_l4043_404374

theorem nesbitt_like_inequality (a b x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  x / (a * y + b * z) + y / (a * z + b * x) + z / (a * x + b * y) ≥ 3 / (a + b) := by
  sorry

end nesbitt_like_inequality_l4043_404374


namespace imaginary_part_of_pure_imaginary_l4043_404394

theorem imaginary_part_of_pure_imaginary (a : ℝ) : 
  let z : ℂ := (1 + Complex.I) / (a - Complex.I)
  (z.re = 0) → z.im = 1 := by
  sorry

end imaginary_part_of_pure_imaginary_l4043_404394


namespace exactly_one_two_black_mutually_exclusive_not_complementary_l4043_404373

/-- A bag containing red and black balls -/
structure Bag where
  red : Nat
  black : Nat

/-- The outcome of drawing two balls -/
inductive DrawResult
  | TwoRed
  | OneRedOneBlack
  | TwoBlack

/-- Event representing exactly one black ball drawn -/
def ExactlyOneBlack (result : DrawResult) : Prop :=
  result = DrawResult.OneRedOneBlack

/-- Event representing exactly two black balls drawn -/
def ExactlyTwoBlack (result : DrawResult) : Prop :=
  result = DrawResult.TwoBlack

/-- The sample space of all possible outcomes -/
def SampleSpace (bag : Bag) : Set DrawResult :=
  {DrawResult.TwoRed, DrawResult.OneRedOneBlack, DrawResult.TwoBlack}

/-- Two events are mutually exclusive if their intersection is empty -/
def MutuallyExclusive (E₁ E₂ : Set DrawResult) : Prop :=
  E₁ ∩ E₂ = ∅

/-- Two events are complementary if their union is the entire sample space -/
def Complementary (E₁ E₂ : Set DrawResult) (S : Set DrawResult) : Prop :=
  E₁ ∪ E₂ = S

/-- Main theorem: ExactlyOneBlack and ExactlyTwoBlack are mutually exclusive but not complementary -/
theorem exactly_one_two_black_mutually_exclusive_not_complementary (bag : Bag) :
  let S := SampleSpace bag
  let E₁ := {r : DrawResult | ExactlyOneBlack r}
  let E₂ := {r : DrawResult | ExactlyTwoBlack r}
  MutuallyExclusive E₁ E₂ ∧ ¬Complementary E₁ E₂ S :=
by sorry

end exactly_one_two_black_mutually_exclusive_not_complementary_l4043_404373


namespace equation_is_linear_one_variable_l4043_404359

/-- Represents a polynomial equation --/
structure PolynomialEquation where
  lhs : ℝ → ℝ
  rhs : ℝ → ℝ

/-- Checks if a polynomial equation is linear with one variable --/
def is_linear_one_variable (eq : PolynomialEquation) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, eq.lhs x = a * x + b ∧ eq.rhs x = 0

/-- The equation y + 3 = 0 --/
def equation : PolynomialEquation :=
  { lhs := λ y => y + 3
    rhs := λ _ => 0 }

/-- Theorem stating that the equation y + 3 = 0 is a linear equation with one variable --/
theorem equation_is_linear_one_variable : is_linear_one_variable equation := by
  sorry

#check equation_is_linear_one_variable

end equation_is_linear_one_variable_l4043_404359


namespace peter_drew_age_difference_l4043_404387

/-- Proves that Peter is 4 years older than Drew given the conditions in the problem --/
theorem peter_drew_age_difference : 
  ∀ (maya drew peter john jacob : ℕ),
  drew = maya + 5 →
  peter > drew →
  john = 30 →
  john = 2 * maya →
  jacob + 2 = (peter + 2) / 2 →
  jacob = 11 →
  peter - drew = 4 := by
  sorry

end peter_drew_age_difference_l4043_404387


namespace dividend_calculation_l4043_404370

theorem dividend_calculation (divisor quotient remainder : ℕ) (h1 : divisor = 36) (h2 : quotient = 20) (h3 : remainder = 5) :
  divisor * quotient + remainder = 725 := by
  sorry

end dividend_calculation_l4043_404370


namespace oliver_card_collection_l4043_404389

theorem oliver_card_collection (monster_club : ℕ) (alien_baseball : ℕ) (battle_gremlins : ℕ) 
  (h1 : monster_club = 2 * alien_baseball)
  (h2 : battle_gremlins = 48)
  (h3 : battle_gremlins = 3 * alien_baseball) :
  monster_club = 32 := by
  sorry

end oliver_card_collection_l4043_404389


namespace sales_volume_estimate_l4043_404338

/-- Represents the linear regression equation for sales volume and price -/
def regression_equation (x : ℝ) : ℝ := -10 * x + 200

/-- The selling price in yuan -/
def selling_price : ℝ := 10

/-- Theorem stating that the estimated sales volume is approximately 100 pieces when the selling price is 10 yuan -/
theorem sales_volume_estimate :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ |regression_equation selling_price - 100| < ε :=
sorry

end sales_volume_estimate_l4043_404338


namespace polynomial_division_proof_l4043_404360

theorem polynomial_division_proof (x : ℝ) :
  6 * x^3 + 12 * x^2 - 5 * x + 3 = 
  (3 * x + 4) * (2 * x^2 + (4/3) * x - 31/9) + 235/9 :=
by sorry

end polynomial_division_proof_l4043_404360


namespace f_max_value_l4043_404397

/-- The function f(x) defined as |x+2017| - |x-2016| -/
def f (x : ℝ) := |x + 2017| - |x - 2016|

/-- Theorem stating that the maximum value of f(x) is 4033 -/
theorem f_max_value : ∀ x : ℝ, f x ≤ 4033 := by sorry

end f_max_value_l4043_404397


namespace smallest_number_with_2_and_4_l4043_404369

def smallest_two_digit_number (a b : ℕ) : ℕ := 
  if a ≤ b then 10 * a + b else 10 * b + a

theorem smallest_number_with_2_and_4 : 
  smallest_two_digit_number 2 4 = 24 := by sorry

end smallest_number_with_2_and_4_l4043_404369


namespace modulo_equivalence_unique_l4043_404334

theorem modulo_equivalence_unique : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 15 ∧ n ≡ 15725 [MOD 16] ∧ n = 13 := by
  sorry

end modulo_equivalence_unique_l4043_404334


namespace polygon_properties_l4043_404381

/-- A polygon with interior angle sum of 1080 degrees has 8 sides and exterior angle sum of 360 degrees -/
theorem polygon_properties (n : ℕ) (interior_sum : ℝ) (h : interior_sum = 1080) :
  (n - 2) * 180 = interior_sum ∧ n = 8 ∧ 360 = (n : ℝ) * (360 / n) := by
  sorry

end polygon_properties_l4043_404381


namespace union_of_sets_l4043_404332

theorem union_of_sets (A B : Set ℝ) : 
  (A = {x : ℝ | x ≥ 0}) → 
  (B = {x : ℝ | x < 1}) → 
  A ∪ B = Set.univ := by
sorry

end union_of_sets_l4043_404332


namespace fran_required_speed_l4043_404366

/-- Calculates the required average speed for Fran to cover the same distance as Joann -/
theorem fran_required_speed (joann_speed : ℝ) (joann_time : ℝ) (fran_time : ℝ) 
  (h1 : joann_speed = 15)
  (h2 : joann_time = 4)
  (h3 : fran_time = 3.5) : 
  (joann_speed * joann_time) / fran_time = 120 / 7 := by
  sorry

#check fran_required_speed

end fran_required_speed_l4043_404366


namespace complex_fourth_power_l4043_404361

theorem complex_fourth_power (i : ℂ) (h : i^2 = -1) : (1 + i)^4 = -4 := by
  sorry

end complex_fourth_power_l4043_404361


namespace car_average_speed_l4043_404349

/-- Given a car traveling at different speeds for 5 hours, prove that its average speed is 94 km/h -/
theorem car_average_speed (v1 v2 v3 v4 v5 : ℝ) (h1 : v1 = 120) (h2 : v2 = 70) (h3 : v3 = 90) (h4 : v4 = 110) (h5 : v5 = 80) :
  (v1 + v2 + v3 + v4 + v5) / 5 = 94 := by
  sorry

#check car_average_speed

end car_average_speed_l4043_404349


namespace total_cost_is_correct_l4043_404391

def chicken_soup_quantity : ℕ := 6
def chicken_soup_price : ℚ := 3/2

def tomato_soup_quantity : ℕ := 3
def tomato_soup_price : ℚ := 5/4

def vegetable_soup_quantity : ℕ := 4
def vegetable_soup_price : ℚ := 7/4

def clam_chowder_quantity : ℕ := 2
def clam_chowder_price : ℚ := 2

def french_onion_soup_quantity : ℕ := 1
def french_onion_soup_price : ℚ := 9/5

def minestrone_soup_quantity : ℕ := 5
def minestrone_soup_price : ℚ := 17/10

def total_cost : ℚ := 
  chicken_soup_quantity * chicken_soup_price +
  tomato_soup_quantity * tomato_soup_price +
  vegetable_soup_quantity * vegetable_soup_price +
  clam_chowder_quantity * clam_chowder_price +
  french_onion_soup_quantity * french_onion_soup_price +
  minestrone_soup_quantity * minestrone_soup_price

theorem total_cost_is_correct : total_cost = 3405/100 := by
  sorry

end total_cost_is_correct_l4043_404391


namespace contractor_job_problem_l4043_404340

/-- A contractor's job problem -/
theorem contractor_job_problem
  (total_days : ℕ) (initial_workers : ℕ) (first_period : ℕ) (remaining_days : ℕ)
  (h1 : total_days = 100)
  (h2 : initial_workers = 10)
  (h3 : first_period = 20)
  (h4 : remaining_days = 75)
  (h5 : first_period * initial_workers = (total_days * initial_workers) / 4) :
  ∃ (fired : ℕ), 
    fired = 2 ∧
    remaining_days * (initial_workers - fired) = 
      (total_days * initial_workers) - (first_period * initial_workers) :=
by sorry

end contractor_job_problem_l4043_404340


namespace rectangle_perimeter_l4043_404304

theorem rectangle_perimeter (length width : ℝ) (h1 : length = 3 * width) (h2 : length * width = 147) :
  2 * (length + width) = 56 := by
  sorry

end rectangle_perimeter_l4043_404304


namespace solution_set_equivalence_l4043_404321

theorem solution_set_equivalence (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := by
  sorry

end solution_set_equivalence_l4043_404321


namespace tangent_point_segment_difference_l4043_404314

/-- A cyclic quadrilateral with an inscribed circle -/
structure CyclicQuadrilateral where
  /-- The lengths of the four sides of the quadrilateral -/
  sides : Fin 4 → ℝ
  /-- The radius of the inscribed circle -/
  inradius : ℝ
  /-- The semiperimeter of the quadrilateral -/
  semiperimeter : ℝ
  /-- The area of the quadrilateral -/
  area : ℝ

/-- The theorem about the difference of segments created by the point of tangency -/
theorem tangent_point_segment_difference
  (Q : CyclicQuadrilateral)
  (h1 : Q.sides 0 = 80)
  (h2 : Q.sides 1 = 100)
  (h3 : Q.sides 2 = 140)
  (h4 : Q.sides 3 = 120)
  (h5 : Q.semiperimeter = (Q.sides 0 + Q.sides 1 + Q.sides 2 + Q.sides 3) / 2)
  (h6 : Q.area = Real.sqrt ((Q.semiperimeter - Q.sides 0) *
                            (Q.semiperimeter - Q.sides 1) *
                            (Q.semiperimeter - Q.sides 2) *
                            (Q.semiperimeter - Q.sides 3)))
  (h7 : Q.inradius * Q.semiperimeter = Q.area) :
  ∃ (x y : ℝ), x + y = 140 ∧ |x - y| = 5 := by
  sorry


end tangent_point_segment_difference_l4043_404314


namespace equation_solution_l4043_404351

theorem equation_solution : ∃ x : ℝ, (x - 5)^3 = (1/27)⁻¹ ∧ x = 8 := by sorry

end equation_solution_l4043_404351


namespace certain_number_proof_l4043_404362

theorem certain_number_proof (x : ℝ) : (((x + 10) * 2) / 2) - 2 = 88 / 2 → x = 36 := by
  sorry

end certain_number_proof_l4043_404362


namespace water_leaked_l4043_404336

/-- Calculates the amount of water leaked from a bucket given the initial and remaining amounts. -/
theorem water_leaked (initial : ℚ) (remaining : ℚ) (h1 : initial = 0.75) (h2 : remaining = 0.5) :
  initial - remaining = 0.25 := by
  sorry

#check water_leaked

end water_leaked_l4043_404336


namespace circle_relations_l4043_404322

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Given three circles P, Q, R with radii p, q, r respectively, where p > q > r,
    and distances between centers d_PQ, d_PR, d_QR, prove that the following
    statements can all be true simultaneously:
    1. p + q can be equal to d_PQ
    2. q + r can be equal to d_QR
    3. p + r can be less than d_PR
    4. p - q can be less than d_PQ -/
theorem circle_relations (P Q R : Circle) 
    (h_p_gt_q : P.radius > Q.radius)
    (h_q_gt_r : Q.radius > R.radius)
    (d_PQ : ℝ) (d_PR : ℝ) (d_QR : ℝ) :
    ∃ (p q r : ℝ),
      p = P.radius ∧ q = Q.radius ∧ r = R.radius ∧
      (p + q = d_PQ ∨ p + q ≠ d_PQ) ∧
      (q + r = d_QR ∨ q + r ≠ d_QR) ∧
      (p + r < d_PR ∨ p + r ≥ d_PR) ∧
      (p - q < d_PQ ∨ p - q ≥ d_PQ) :=
by sorry

end circle_relations_l4043_404322


namespace emily_caught_four_trout_l4043_404388

def fishing_problem (num_trout : ℕ) : Prop :=
  let num_catfish : ℕ := 3
  let num_bluegill : ℕ := 5
  let weight_trout : ℝ := 2
  let weight_catfish : ℝ := 1.5
  let weight_bluegill : ℝ := 2.5
  let total_weight : ℝ := 25
  (num_trout : ℝ) * weight_trout + 
  (num_catfish : ℝ) * weight_catfish + 
  (num_bluegill : ℝ) * weight_bluegill = total_weight

theorem emily_caught_four_trout : 
  ∃ (n : ℕ), fishing_problem n ∧ n = 4 := by
  sorry

end emily_caught_four_trout_l4043_404388


namespace quadratic_roots_property_l4043_404317

theorem quadratic_roots_property (m n : ℝ) : 
  (m^2 - 2*m - 2025 = 0) → 
  (n^2 - 2*n - 2025 = 0) → 
  (m^2 - 3*m - n = 2023) := by
sorry

end quadratic_roots_property_l4043_404317


namespace sin_405_plus_cos_neg_270_l4043_404348

theorem sin_405_plus_cos_neg_270 : 
  Real.sin (405 * π / 180) + Real.cos (-270 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end sin_405_plus_cos_neg_270_l4043_404348


namespace sum_difference_even_odd_1000_l4043_404396

def first_n_odd (n : ℕ) : List ℕ := List.range n |> List.map (fun i => 2 * i + 1)
def first_n_even (n : ℕ) : List ℕ := List.range n |> List.map (fun i => 2 * (i + 1))

theorem sum_difference_even_odd_1000 : 
  (first_n_even 1000).sum - (first_n_odd 1000).sum = 1000 := by
  sorry

end sum_difference_even_odd_1000_l4043_404396


namespace max_value_theorem_l4043_404339

theorem max_value_theorem (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) 
  (h_sum : x₁ + x₂ + x₃ = 1) : 
  x₁ * x₂^2 * x₃ + x₁ * x₂ * x₃^2 ≤ 27/1024 := by
  sorry

end max_value_theorem_l4043_404339


namespace sally_pens_home_l4043_404383

/-- The number of pens Sally takes home given the initial conditions -/
def pens_taken_home (total_pens : ℕ) (num_students : ℕ) (pens_per_student : ℕ) : ℕ :=
  let pens_given := num_students * pens_per_student
  let pens_remaining := total_pens - pens_given
  pens_remaining / 2

theorem sally_pens_home :
  pens_taken_home 342 44 7 = 17 := by
  sorry

end sally_pens_home_l4043_404383


namespace trapezoid_area_l4043_404316

/-- Trapezoid ABCD with given properties -/
structure Trapezoid where
  -- Length of AD
  ad : ℝ
  -- Length of BC
  bc : ℝ
  -- Length of CD
  cd : ℝ
  -- BC is parallel to AD
  parallel : True
  -- Ratio of BC to AD is 5:7
  ratio_bc_ad : bc / ad = 5 / 7
  -- AF:FD = 4:3
  ratio_af_fd : (4 / 7 * ad) / (3 / 7 * ad) = 4 / 3
  -- CE:ED = 2:3
  ratio_ce_ed : (2 / 5 * cd) / (3 / 5 * cd) = 2 / 3
  -- Area of ABEF is 123
  area_abef : (ad * cd - (3 / 7 * ad) * (3 / 5 * cd) - bc * (2 / 5 * cd)) / 2 = 123

/-- The area of trapezoid ABCD is 180 -/
theorem trapezoid_area (t : Trapezoid) : (t.ad + t.bc) * t.cd / 2 = 180 := by
  sorry

end trapezoid_area_l4043_404316


namespace ben_savings_problem_l4043_404327

/-- Ben's daily starting amount -/
def daily_start : ℕ := 50

/-- Ben's daily spending -/
def daily_spend : ℕ := 15

/-- Ben's daily savings -/
def daily_savings : ℕ := daily_start - daily_spend

/-- Ben's final amount after mom's doubling and dad's addition -/
def final_amount : ℕ := 500

/-- Additional amount from dad -/
def dad_addition : ℕ := 10

/-- The number of days elapsed -/
def days_elapsed : ℕ := 7

theorem ben_savings_problem :
  final_amount = 2 * (daily_savings * days_elapsed) + dad_addition := by
  sorry

end ben_savings_problem_l4043_404327


namespace algebraic_expression_value_l4043_404333

theorem algebraic_expression_value (p q : ℤ) :
  p * 3^3 + 3 * q + 1 = 2015 →
  p * (-3)^3 - 3 * q + 1 = -2013 := by
  sorry

end algebraic_expression_value_l4043_404333


namespace min_value_of_reciprocal_sum_l4043_404313

theorem min_value_of_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2 * b = 1) :
  1 / a + 2 / b ≥ 9 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 2 * b₀ = 1 ∧ 1 / a₀ + 2 / b₀ = 9 :=
by sorry

end min_value_of_reciprocal_sum_l4043_404313


namespace max_factors_upper_bound_max_factors_achievable_max_factors_is_maximum_l4043_404306

def max_factors (b n : ℕ+) : ℕ :=
  sorry

theorem max_factors_upper_bound (b n : ℕ+) (hb : b ≤ 15) (hn : n ≤ 20) :
  max_factors b n ≤ 861 :=
sorry

theorem max_factors_achievable :
  ∃ (b n : ℕ+), b ≤ 15 ∧ n ≤ 20 ∧ max_factors b n = 861 :=
sorry

theorem max_factors_is_maximum :
  ∀ (b n : ℕ+), b ≤ 15 → n ≤ 20 → max_factors b n ≤ 861 :=
sorry

end max_factors_upper_bound_max_factors_achievable_max_factors_is_maximum_l4043_404306


namespace gem_stone_necklaces_sold_l4043_404386

/-- Proves that the number of gem stone necklaces sold is 3 -/
theorem gem_stone_necklaces_sold (bead_necklaces : ℕ) (price_per_necklace : ℕ) (total_earnings : ℕ) :
  bead_necklaces = 4 →
  price_per_necklace = 3 →
  total_earnings = 21 →
  total_earnings = price_per_necklace * (bead_necklaces + 3) :=
by sorry

end gem_stone_necklaces_sold_l4043_404386


namespace min_distance_point_l4043_404390

/-- A triangle in a 2D plane --/
structure Triangle :=
  (A B C : ℝ × ℝ)

/-- The Fermat point of a triangle --/
def fermatPoint (t : Triangle) : ℝ × ℝ := sorry

/-- The sum of distances from a point to the vertices of a triangle --/
def sumOfDistances (t : Triangle) (p : ℝ × ℝ) : ℝ := sorry

/-- The largest angle in a triangle --/
def largestAngle (t : Triangle) : ℝ := sorry

/-- The vertex corresponding to the largest angle in a triangle --/
def largestAngleVertex (t : Triangle) : ℝ × ℝ := sorry

/-- Theorem: The point that minimizes the sum of distances to the vertices of a triangle --/
theorem min_distance_point (t : Triangle) :
  ∃ (M : ℝ × ℝ), (∀ (p : ℝ × ℝ), sumOfDistances t M ≤ sumOfDistances t p) ∧
  ((largestAngle t < 2 * Real.pi / 3 ∧ M = fermatPoint t) ∨
   (largestAngle t ≥ 2 * Real.pi / 3 ∧ M = largestAngleVertex t)) :=
sorry

end min_distance_point_l4043_404390


namespace janes_skirts_l4043_404398

/-- Proves that Jane bought 2 skirts given the problem conditions -/
theorem janes_skirts :
  let skirt_price : ℕ := 13
  let blouse_price : ℕ := 6
  let num_blouses : ℕ := 3
  let paid : ℕ := 100
  let change : ℕ := 56
  let total_spent : ℕ := paid - change
  ∃ (num_skirts : ℕ), num_skirts * skirt_price + num_blouses * blouse_price = total_spent ∧ num_skirts = 2 :=
by
  sorry

end janes_skirts_l4043_404398


namespace angle_c_measure_l4043_404363

theorem angle_c_measure (A B C : ℝ) (h : A + B = 90) : C = 90 :=
by
  sorry

end angle_c_measure_l4043_404363


namespace third_number_in_nth_row_l4043_404352

/-- Represents the function that gives the third number from the left in the nth row
    of a triangular array of positive odd numbers. -/
def thirdNumber (n : ℕ) : ℕ := n^2 - n + 5

/-- Theorem stating that for n ≥ 3, the third number from the left in the nth row
    of a triangular array of positive odd numbers is n^2 - n + 5. -/
theorem third_number_in_nth_row (n : ℕ) (h : n ≥ 3) :
  thirdNumber n = n^2 - n + 5 := by
  sorry

end third_number_in_nth_row_l4043_404352


namespace arithmetic_geometric_k4_l4043_404379

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  d_nonzero : d ≠ 0
  is_arithmetic : ∀ n, a (n + 1) = a n + d

/-- A subsequence of an arithmetic sequence that forms a geometric sequence -/
structure GeometricSubsequence (as : ArithmeticSequence) where
  k : ℕ → ℕ
  q : ℝ
  q_nonzero : q ≠ 0
  is_geometric : ∀ n, as.a (k (n + 1)) = q * as.a (k n)
  k1_not_1 : k 1 ≠ 1
  k2_not_2 : k 2 ≠ 2
  k3_not_6 : k 3 ≠ 6

/-- The main theorem -/
theorem arithmetic_geometric_k4 (as : ArithmeticSequence) (gs : GeometricSubsequence as) :
  gs.k 4 = 22 := by
  sorry

end arithmetic_geometric_k4_l4043_404379


namespace base5_132_to_base10_l4043_404329

/-- Converts a base-5 digit to its base-10 equivalent --/
def base5ToBase10Digit (d : Nat) : Nat :=
  if d < 5 then d else 0

/-- Converts a 3-digit base-5 number to its base-10 equivalent --/
def base5ToBase10 (d2 d1 d0 : Nat) : Nat :=
  (base5ToBase10Digit d2) * 25 + (base5ToBase10Digit d1) * 5 + (base5ToBase10Digit d0)

/-- Theorem stating that the base-10 representation of the base-5 number 132 is 42 --/
theorem base5_132_to_base10 : base5ToBase10 1 3 2 = 42 := by
  sorry

end base5_132_to_base10_l4043_404329


namespace first_group_students_l4043_404310

theorem first_group_students (total : ℕ) (group2 group3 group4 : ℕ) 
  (h1 : total = 24)
  (h2 : group2 = 8)
  (h3 : group3 = 7)
  (h4 : group4 = 4) :
  total - (group2 + group3 + group4) = 5 := by
  sorry

end first_group_students_l4043_404310


namespace tan_value_for_special_condition_l4043_404384

theorem tan_value_for_special_condition (α : Real) 
  (h1 : α > 0) (h2 : α < Real.pi / 2) 
  (h3 : Real.sin α ^ 2 + Real.cos (2 * α) = 1 / 4) : 
  Real.tan α = Real.sqrt 3 := by sorry

end tan_value_for_special_condition_l4043_404384


namespace pentagon_perimeter_l4043_404337

/-- The perimeter of pentagon FGHIJ is 6, given that FG = GH = HI = IJ = 1 -/
theorem pentagon_perimeter (F G H I J : ℝ × ℝ) : 
  (dist F G = 1) → (dist G H = 1) → (dist H I = 1) → (dist I J = 1) →
  dist F G + dist G H + dist H I + dist I J + dist J F = 6 :=
by sorry


end pentagon_perimeter_l4043_404337


namespace sufficient_not_necessary_condition_l4043_404302

theorem sufficient_not_necessary_condition :
  (∃ a b : ℝ, a < 0 ∧ -1 < b ∧ b < 0 → a + a * b < 0) ∧
  (∃ a b : ℝ, a + a * b < 0 ∧ ¬(a < 0 ∧ -1 < b ∧ b < 0)) :=
by sorry

end sufficient_not_necessary_condition_l4043_404302


namespace min_value_is_four_l4043_404307

/-- The line passing through points A(3, 0) and B(1, 1) -/
def line_AB (x y : ℝ) : Prop := y = (x - 3) / (-2)

/-- The objective function to be minimized -/
def objective_function (x y : ℝ) : ℝ := 2 * x + 4 * y

/-- Theorem stating that the minimum value of the objective function is 4 -/
theorem min_value_is_four :
  ∀ x y : ℝ, line_AB x y → objective_function x y ≥ 4 :=
sorry

end min_value_is_four_l4043_404307


namespace sum_of_factors_l4043_404326

theorem sum_of_factors (d e f : ℤ) : 
  (∀ x : ℝ, x^2 + 21*x + 110 = (x + d)*(x + e)) → 
  (∀ x : ℝ, x^2 - 19*x + 88 = (x - e)*(x - f)) → 
  d + e + f = 30 := by
sorry

end sum_of_factors_l4043_404326


namespace max_a_equals_min_f_l4043_404346

theorem max_a_equals_min_f : 
  let f (x : ℝ) := x^2 + 2*x - 6
  (∃ (a_max : ℝ), (∀ (a : ℝ), (∀ (x : ℝ), f x ≥ a) → a ≤ a_max) ∧ 
    (∀ (x : ℝ), f x ≥ a_max)) ∧ 
  (∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x) →
  ∃ (a_max x_min : ℝ), (∀ (a : ℝ), (∀ (x : ℝ), f x ≥ a) → a ≤ a_max) ∧ 
    (∀ (x : ℝ), f x ≥ a_max) ∧ 
    (∀ (x : ℝ), f x_min ≤ f x) ∧ 
    a_max = f x_min :=
by sorry

end max_a_equals_min_f_l4043_404346


namespace quadratic_equation_equivalence_l4043_404325

theorem quadratic_equation_equivalence (m : ℝ) : 
  (∀ x, x^2 - m*x + 6 = 0 ↔ (x - 3)^2 = 3) → m = 6 := by
  sorry

end quadratic_equation_equivalence_l4043_404325


namespace least_positive_integer_to_multiple_of_five_l4043_404318

theorem least_positive_integer_to_multiple_of_five : 
  ∃ (n : ℕ), n > 0 ∧ (∀ m : ℕ, m > 0 → (624 + m) % 5 = 0 → m ≥ n) ∧ (624 + n) % 5 = 0 :=
by sorry

end least_positive_integer_to_multiple_of_five_l4043_404318


namespace sixth_term_is_half_l4043_404382

/-- Geometric sequence with first term 16 and common ratio 1/2 -/
def geometricSequence : ℕ → ℚ
  | 0 => 16
  | n + 1 => (geometricSequence n) / 2

/-- The sixth term of the geometric sequence is 1/2 -/
theorem sixth_term_is_half : geometricSequence 5 = 1 / 2 := by
  sorry

end sixth_term_is_half_l4043_404382


namespace all_triangles_congruent_l4043_404315

/-- Represents a square tablecloth with hanging triangles -/
structure Tablecloth where
  -- Side length of the square tablecloth
  side : ℝ
  -- Heights of the hanging triangles
  hA : ℝ
  hB : ℝ
  hC : ℝ
  hD : ℝ
  -- Condition that all heights are positive
  hA_pos : hA > 0
  hB_pos : hB > 0
  hC_pos : hC > 0
  hD_pos : hD > 0
  -- Condition that △A and △B are congruent (given)
  hA_eq_hB : hA = hB

/-- Theorem stating that if △A and △B are congruent, then all hanging triangles are congruent -/
theorem all_triangles_congruent (t : Tablecloth) :
  t.hA = t.hB ∧ t.hA = t.hC ∧ t.hA = t.hD :=
sorry

end all_triangles_congruent_l4043_404315


namespace arithmetic_and_geometric_sequence_l4043_404300

theorem arithmetic_and_geometric_sequence (a b c : ℝ) :
  (b - a = c - b) → -- arithmetic sequence condition
  (b / a = c / b) → -- geometric sequence condition
  (a ≠ 0) →         -- non-zero condition for geometric sequence
  (a = b ∧ b = c ∧ a ≠ 0) := by
sorry

end arithmetic_and_geometric_sequence_l4043_404300


namespace min_value_of_f_l4043_404305

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- State the theorem that the minimum value of f(x) is -2
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ (x : ℝ), f x ≥ m) ∧ (∃ (x : ℝ), f x = m) ∧ m = -2 := by
  sorry

end min_value_of_f_l4043_404305


namespace inequality_proof_l4043_404377

def f (a x : ℝ) : ℝ := |x - a| + 1

theorem inequality_proof (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  let a := 1 / m + 1 / n
  (∀ x, f a x ≤ 2 ↔ 0 ≤ x ∧ x ≤ 2) →
  m + 2 * n ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end inequality_proof_l4043_404377


namespace field_trip_probability_l4043_404372

/-- The number of vehicles available for the field trip -/
def num_vehicles : ℕ := 3

/-- The number of students in the specific group we're considering -/
def group_size : ℕ := 4

/-- The probability that all students in the group ride in the same vehicle -/
def same_vehicle_probability : ℚ := 1 / 27

theorem field_trip_probability :
  (num_vehicles : ℚ) / (num_vehicles ^ group_size) = same_vehicle_probability :=
sorry

end field_trip_probability_l4043_404372


namespace system_solution_existence_l4043_404367

/-- The system of equations has at least one solution if and only if b ≥ -2√2 - 1/4 -/
theorem system_solution_existence (b : ℝ) :
  (∃ a x y : ℝ, y = b - x^2 ∧ x^2 + y^2 + 2*a^2 = 4 - 2*a*(x + y)) ↔
  b ≥ -2 * Real.sqrt 2 - 1/4 := by
  sorry

end system_solution_existence_l4043_404367


namespace quadratic_monotone_decreasing_condition_l4043_404330

/-- A quadratic function f(x) = x^2 + ax + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

/-- The function is monotonically decreasing in the interval (-∞, 3) -/
def is_monotone_decreasing (a : ℝ) : Prop :=
  ∀ x y, x < y → x < 3 → y < 3 → f a x > f a y

/-- The theorem states that if f is monotonically decreasing in (-∞, 3), then a ∈ (-∞, -6] -/
theorem quadratic_monotone_decreasing_condition (a : ℝ) :
  is_monotone_decreasing a → a ≤ -6 :=
sorry

end quadratic_monotone_decreasing_condition_l4043_404330


namespace jaxon_toys_count_l4043_404368

/-- The number of toys Jaxon has -/
def jaxon_toys : ℕ := 15

/-- The number of toys Gabriel has -/
def gabriel_toys : ℕ := 2 * jaxon_toys

/-- The number of toys Jerry has -/
def jerry_toys : ℕ := gabriel_toys + 8

theorem jaxon_toys_count :
  jaxon_toys + gabriel_toys + jerry_toys = 83 ∧ jaxon_toys = 15 :=
by sorry

end jaxon_toys_count_l4043_404368


namespace polynomial_division_identity_l4043_404353

/-- The polynomial to be divided -/
def f (x : ℝ) : ℝ := x^6 - 5*x^4 + 3*x^3 - 7*x^2 + 2*x - 8

/-- The divisor polynomial -/
def g (x : ℝ) : ℝ := x - 3

/-- The quotient polynomial -/
def q (x : ℝ) : ℝ := x^5 + 3*x^4 + 4*x^3 + 15*x^2 + 38*x + 116

/-- The remainder -/
def r : ℝ := 340

/-- Theorem stating the polynomial division identity -/
theorem polynomial_division_identity : 
  ∀ x : ℝ, f x = g x * q x + r := by sorry

end polynomial_division_identity_l4043_404353


namespace dividend_calculation_l4043_404343

/-- Proves that given a divisor of -4 2/3, a quotient of -57 1/5, and a remainder of 2 1/9, the dividend is equal to 269 2/45. -/
theorem dividend_calculation (divisor quotient remainder dividend : ℚ) : 
  divisor = -14/3 →
  quotient = -286/5 →
  remainder = 19/9 →
  dividend = divisor * quotient + remainder →
  dividend = 12107/45 :=
by sorry

end dividend_calculation_l4043_404343


namespace system_solution_l4043_404375

theorem system_solution (x y z : ℝ) : 
  (x + y + x * y = 19 ∧ 
   y + z + y * z = 11 ∧ 
   z + x + z * x = 14) ↔ 
  ((x = 4 ∧ y = 3 ∧ z = 2) ∨ 
   (x = -6 ∧ y = -5 ∧ z = -4)) :=
by sorry

end system_solution_l4043_404375


namespace symmetry_of_point_l4043_404392

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line y = -x -/
def symmetryLine (p : Point) : Prop := p.y = -p.x

/-- Definition of symmetry with respect to y = -x -/
def isSymmetric (p1 p2 : Point) : Prop :=
  p2.x = -p1.y ∧ p2.y = -p1.x

theorem symmetry_of_point :
  let p1 : Point := ⟨1, 4⟩
  let p2 : Point := ⟨-4, -1⟩
  isSymmetric p1 p2 := by sorry

end symmetry_of_point_l4043_404392


namespace gp_ratio_is_four_l4043_404309

theorem gp_ratio_is_four (x : ℝ) :
  (∀ r : ℝ, (40 + x) = (10 + x) * r ∧ (160 + x) = (40 + x) * r) →
  r = 4 :=
by sorry

end gp_ratio_is_four_l4043_404309


namespace mystery_number_l4043_404301

theorem mystery_number : ∃ x : ℤ, x + 45 = 92 ∧ x = 47 := by
  sorry

end mystery_number_l4043_404301


namespace sqrt_19992000_floor_l4043_404341

theorem sqrt_19992000_floor : ⌊Real.sqrt 19992000⌋ = 4471 := by sorry

end sqrt_19992000_floor_l4043_404341


namespace certain_number_exists_and_is_one_l4043_404331

theorem certain_number_exists_and_is_one : 
  ∃ (x : ℕ), x > 0 ∧ (57 * x) % 8 = 7 ∧ ∀ (y : ℕ), y > 0 ∧ (57 * y) % 8 = 7 → x ≤ y := by
  sorry

end certain_number_exists_and_is_one_l4043_404331


namespace cat_food_finished_l4043_404365

/-- Represents the days of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the day after a given number of days -/
def dayAfter (d : Day) (n : ℕ) : Day :=
  match n with
  | 0 => d
  | n + 1 => dayAfter (match d with
    | Day.Monday => Day.Tuesday
    | Day.Tuesday => Day.Wednesday
    | Day.Wednesday => Day.Thursday
    | Day.Thursday => Day.Friday
    | Day.Friday => Day.Saturday
    | Day.Saturday => Day.Sunday
    | Day.Sunday => Day.Monday) n

/-- The amount of food consumed by the cat per day -/
def dailyConsumption : ℚ := 1/5 + 1/6

/-- The total amount of food in the box -/
def totalFood : ℚ := 10

/-- Theorem stating when the cat will finish the food -/
theorem cat_food_finished :
  ∃ (n : ℕ), n * dailyConsumption > totalFood ∧
  (n - 1) * dailyConsumption ≤ totalFood ∧
  dayAfter Day.Monday (n - 1) = Day.Wednesday :=
by sorry


end cat_food_finished_l4043_404365


namespace log_equation_sum_of_squares_l4043_404324

theorem log_equation_sum_of_squares (x y : ℝ) (hx : x > 1) (hy : y > 1) 
  (h : (Real.log x / Real.log 4)^3 + (Real.log y / Real.log 5)^3 + 27 = 9 * (Real.log x / Real.log 4) * (Real.log y / Real.log 5)) :
  x^2 + y^2 = 189 := by
sorry

end log_equation_sum_of_squares_l4043_404324


namespace smallest_four_digit_number_l4043_404328

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def T (n : ℕ) : ℕ := (n / 1000) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def S (n : ℕ) : ℕ := (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem smallest_four_digit_number (p : ℕ) (k : ℕ) (h_prime : Nat.Prime p) :
  ∃ (x : ℕ), is_four_digit x ∧ T x = p^k ∧ S x = p^p - 5 ∧
  ∀ (y : ℕ), is_four_digit y ∧ T y = p^k ∧ S y = p^p - 5 → x ≤ y :=
sorry

end smallest_four_digit_number_l4043_404328


namespace divisibility_theorem_l4043_404335

theorem divisibility_theorem (n : ℕ) (x : ℤ) (h : n ≥ 1) :
  ∃ k : ℤ, x^(2*n+1) - (2*n+1)*x^(n+1) + (2*n+1)*x^n - 1 = k * (x-1)^3 := by
  sorry

end divisibility_theorem_l4043_404335


namespace fair_spending_l4043_404380

def money_at_arrival : ℕ := 87
def money_at_departure : ℕ := 16

theorem fair_spending : money_at_arrival - money_at_departure = 71 := by
  sorry

end fair_spending_l4043_404380


namespace cups_count_l4043_404342

-- Define the cost of a single paper plate and cup
variable (plate_cost cup_cost : ℝ)

-- Define the number of cups in the second purchase
variable (cups_in_second_purchase : ℕ)

-- First condition: 100 plates and 200 cups cost $7.50
axiom first_purchase : 100 * plate_cost + 200 * cup_cost = 7.50

-- Second condition: 20 plates and cups_in_second_purchase cups cost $1.50
axiom second_purchase : 20 * plate_cost + cups_in_second_purchase * cup_cost = 1.50

-- Theorem to prove
theorem cups_count : cups_in_second_purchase = 40 := by
  sorry

end cups_count_l4043_404342


namespace gcd_of_324_243_135_l4043_404319

theorem gcd_of_324_243_135 : Nat.gcd 324 (Nat.gcd 243 135) = 27 := by
  sorry

end gcd_of_324_243_135_l4043_404319


namespace equal_roots_quadratic_l4043_404399

/-- 
Given a quadratic equation x^2 + 2x + k = 0 with two equal real roots,
prove that k = 1.
-/
theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, x^2 + 2*x + k = 0 ∧ 
   ∀ y : ℝ, y^2 + 2*y + k = 0 → y = x) →
  k = 1 := by
sorry

end equal_roots_quadratic_l4043_404399


namespace maximal_colored_squares_correct_l4043_404345

/-- Given positive integers n and k where n > k^2 > 4, maximal_colored_squares
    returns the maximal number of unit squares that can be colored in an n × n grid,
    such that in any k-group there are two squares with the same color and
    two squares with different colors. -/
def maximal_colored_squares (n k : ℕ+) (h1 : n > k^2) (h2 : k^2 > 4) : ℕ :=
  n * (k - 1)^2

/-- Theorem stating that maximal_colored_squares gives the correct result -/
theorem maximal_colored_squares_correct (n k : ℕ+) (h1 : n > k^2) (h2 : k^2 > 4) :
  maximal_colored_squares n k h1 h2 = n * (k - 1)^2 := by
  sorry

#check maximal_colored_squares
#check maximal_colored_squares_correct

end maximal_colored_squares_correct_l4043_404345


namespace triangle_height_calculation_l4043_404303

/-- Given a triangle with area 615 m² and one side of 123 meters, 
    the length of the perpendicular dropped on this side from the opposite vertex is 10 meters. -/
theorem triangle_height_calculation (A : ℝ) (b h : ℝ) 
    (h_area : A = 615)
    (h_base : b = 123)
    (h_triangle_area : A = (b * h) / 2) : h = 10 := by
  sorry

end triangle_height_calculation_l4043_404303


namespace square_side_length_l4043_404347

theorem square_side_length (s : ℝ) (h : s > 0) : s^2 = 6 * (4 * s) → s = 24 := by
  sorry

end square_side_length_l4043_404347


namespace problem_solution_l4043_404364

theorem problem_solution (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 119) : x = 39 := by
  sorry

end problem_solution_l4043_404364


namespace allan_total_balloons_l4043_404311

def initial_balloons : Nat := 5
def additional_balloons : Nat := 3

theorem allan_total_balloons : 
  initial_balloons + additional_balloons = 8 := by
  sorry

end allan_total_balloons_l4043_404311


namespace sqrt_49_times_sqrt_25_l4043_404395

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end sqrt_49_times_sqrt_25_l4043_404395


namespace pencils_left_ashtons_pencils_l4043_404350

/-- Given two boxes of pencils with fourteen pencils each, after giving away six pencils, the number of pencils left is 22. -/
theorem pencils_left (boxes : Nat) (pencils_per_box : Nat) (pencils_given_away : Nat) : Nat :=
  boxes * pencils_per_box - pencils_given_away

/-- Ashton's pencil problem -/
theorem ashtons_pencils : pencils_left 2 14 6 = 22 := by
  sorry

end pencils_left_ashtons_pencils_l4043_404350


namespace students_taking_history_or_statistics_l4043_404357

theorem students_taking_history_or_statistics (total : ℕ) (history : ℕ) (statistics : ℕ) (history_not_statistics : ℕ) : 
  total = 90 → history = 36 → statistics = 32 → history_not_statistics = 25 →
  ∃ (both : ℕ), history - both = history_not_statistics ∧ history + statistics - both = 57 := by
sorry

end students_taking_history_or_statistics_l4043_404357


namespace sqrt_fifth_power_sixth_l4043_404356

theorem sqrt_fifth_power_sixth : (Real.sqrt ((Real.sqrt 5)^4))^6 = 15625 := by
  sorry

end sqrt_fifth_power_sixth_l4043_404356


namespace lunch_cost_theorem_l4043_404376

/-- Calculates the amount each paying student contributes for lunch -/
def lunch_cost_per_paying_student (total_students : ℕ) (free_lunch_percentage : ℚ) (total_cost : ℚ) : ℚ :=
  let paying_students := total_students * (1 - free_lunch_percentage)
  total_cost / paying_students

theorem lunch_cost_theorem (total_students : ℕ) (free_lunch_percentage : ℚ) (total_cost : ℚ) 
  (h1 : total_students = 50)
  (h2 : free_lunch_percentage = 2/5)
  (h3 : total_cost = 210) :
  lunch_cost_per_paying_student total_students free_lunch_percentage total_cost = 7 := by
  sorry

#eval lunch_cost_per_paying_student 50 (2/5) 210

end lunch_cost_theorem_l4043_404376


namespace smallest_square_partition_l4043_404385

theorem smallest_square_partition : ∃ (n : ℕ),
  (n > 0) ∧ 
  (∃ (a b c : ℕ), 
    (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
    (a + b + c = 12) ∧ 
    (a ≥ 9) ∧
    (n^2 = a * 1^2 + b * 2^2 + c * 3^2)) ∧
  (∀ (m : ℕ), m < n → 
    ¬(∃ (a b c : ℕ),
      (a > 0) ∧ (b > 0) ∧ (c > 0) ∧
      (a + b + c = 12) ∧
      (a ≥ 9) ∧
      (m^2 = a * 1^2 + b * 2^2 + c * 3^2))) ∧
  n = 5 :=
by sorry

end smallest_square_partition_l4043_404385


namespace f_properties_l4043_404371

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 0 else 1 - 1/x

theorem f_properties :
  (∀ x, x ≥ 0 → f x ≥ 0) ∧
  (f 1 = 0) ∧
  (∀ x, x > 1 → f x > 0) ∧
  (∀ x y, x ≥ 0 → y ≥ 0 → x + y > 0 → f (x * f y) * f y = f (x * y / (x + y))) :=
by sorry

end f_properties_l4043_404371


namespace prob_2012_higher_than_2011_l4043_404312

/-- The probability of guessing the correct answer to a single question -/
def p : ℝ := 0.25

/-- The probability of guessing incorrectly -/
def q : ℝ := 1 - p

/-- Calculate the probability of passing the exam given the total number of questions and the minimum required correct answers -/
def prob_pass (n : ℕ) (k : ℕ) : ℝ :=
  1 - (Finset.sum (Finset.range k) (λ i => Nat.choose n i * p^i * q^(n - i)))

/-- The probability of passing the exam in 2011 -/
def prob_2011 : ℝ := prob_pass 20 3

/-- The probability of passing the exam in 2012 -/
def prob_2012 : ℝ := prob_pass 40 6

/-- Theorem stating that the probability of passing in 2012 is higher than in 2011 -/
theorem prob_2012_higher_than_2011 : prob_2012 > prob_2011 := by
  sorry

end prob_2012_higher_than_2011_l4043_404312


namespace isosceles_diagonal_probability_l4043_404320

/-- The probability of selecting two diagonals from a regular pentagon 
    such that they form the two legs of an isosceles triangle -/
theorem isosceles_diagonal_probability (n m : ℕ) : 
  n = 10 → m = 5 → (m : ℚ) / n = 1 / 2 := by sorry

end isosceles_diagonal_probability_l4043_404320


namespace face_washing_unit_is_liters_l4043_404358

/-- Represents units of volume measurement -/
inductive VolumeUnit
  | Liters
  | Milliliters
  | Grams

/-- Represents the amount of water used for face washing -/
def face_washing_amount : ℝ := 2

/-- Determines if a given volume unit is appropriate for face washing -/
def is_appropriate_unit (unit : VolumeUnit) : Prop :=
  match unit with
  | VolumeUnit.Liters => true
  | _ => false

theorem face_washing_unit_is_liters :
  is_appropriate_unit VolumeUnit.Liters = true :=
sorry

end face_washing_unit_is_liters_l4043_404358


namespace quadratic_expression_value_l4043_404355

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 3 * x + y = 10) 
  (eq2 : x + 3 * y = 14) : 
  10 * x^2 + 12 * x * y + 10 * y^2 = 296 := by
  sorry

end quadratic_expression_value_l4043_404355


namespace divisibility_implies_one_or_seven_l4043_404308

theorem divisibility_implies_one_or_seven (a n : ℤ) 
  (ha : a ≥ 1) 
  (h1 : a ∣ n + 2) 
  (h2 : a ∣ n^2 + n + 5) : 
  a = 1 ∨ a = 7 := by
  sorry

end divisibility_implies_one_or_seven_l4043_404308


namespace coefficient_a2_l4043_404344

/-- Given z = 1 + i and (z+x)^4 = a_4x^4 + a_3x^3 + a_2x^2 + a_1x + a_0, prove that a_2 = 12i -/
theorem coefficient_a2 (z : ℂ) (a_4 a_3 a_2 a_1 a_0 : ℂ) :
  z = 1 + Complex.I →
  (∀ x : ℂ, (z + x)^4 = a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) →
  a_2 = 12 * Complex.I :=
by sorry

end coefficient_a2_l4043_404344


namespace elmwood_population_l4043_404393

/-- The number of cities in the County of Elmwood -/
def num_cities : ℕ := 25

/-- The lower bound of the average population per city -/
def avg_pop_lower : ℕ := 3200

/-- The upper bound of the average population per city -/
def avg_pop_upper : ℕ := 3700

/-- The total population of the County of Elmwood -/
def total_population : ℕ := 86250

theorem elmwood_population :
  ∃ (avg_pop : ℚ),
    avg_pop > avg_pop_lower ∧
    avg_pop < avg_pop_upper ∧
    (num_cities : ℚ) * avg_pop = total_population :=
sorry

end elmwood_population_l4043_404393


namespace smallest_integers_difference_l4043_404354

theorem smallest_integers_difference : ∃ (n₁ n₂ : ℕ), 
  (∀ k : ℕ, 2 ≤ k → k ≤ 12 → n₁ % k = 1) ∧
  (∀ k : ℕ, 2 ≤ k → k ≤ 12 → n₂ % k = 1) ∧
  n₁ > 1 ∧ n₂ > n₁ ∧
  (∀ m : ℕ, m > 1 → (∀ k : ℕ, 2 ≤ k → k ≤ 12 → m % k = 1) → m ≥ n₁) ∧
  n₂ - n₁ = 4620 :=
by sorry

end smallest_integers_difference_l4043_404354
