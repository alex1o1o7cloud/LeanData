import Mathlib

namespace NUMINAMATH_CALUDE_number_division_problem_l3419_341929

theorem number_division_problem (x : ℝ) : (x / 5 = 70 + x / 6) → x = 2100 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l3419_341929


namespace NUMINAMATH_CALUDE_complex_fraction_equals_i_l3419_341963

theorem complex_fraction_equals_i : (1 + Complex.I * Real.sqrt 3) / (Real.sqrt 3 - Complex.I) = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_i_l3419_341963


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l3419_341935

theorem sum_of_a_and_b (a b : ℚ) 
  (eq1 : 3 * a + 5 * b = 47) 
  (eq2 : 7 * a + 2 * b = 52) : 
  a + b = 35 / 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l3419_341935


namespace NUMINAMATH_CALUDE_gcd_of_squares_sum_l3419_341916

theorem gcd_of_squares_sum : Nat.gcd (125^2 + 235^2 + 349^2) (124^2 + 234^2 + 350^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_squares_sum_l3419_341916


namespace NUMINAMATH_CALUDE_shifted_proportional_function_l3419_341913

/-- Given a proportional function y = -2x that is shifted up by 3 units,
    the resulting function is y = -2x + 3. -/
theorem shifted_proportional_function :
  let f : ℝ → ℝ := λ x ↦ -2 * x
  let shift : ℝ := 3
  let g : ℝ → ℝ := λ x ↦ f x + shift
  ∀ x : ℝ, g x = -2 * x + 3 := by
  sorry

end NUMINAMATH_CALUDE_shifted_proportional_function_l3419_341913


namespace NUMINAMATH_CALUDE_soda_can_ratio_l3419_341911

theorem soda_can_ratio :
  let initial_cans : ℕ := 22
  let taken_cans : ℕ := 6
  let final_cans : ℕ := 24
  let remaining_cans := initial_cans - taken_cans
  let bought_cans := final_cans - remaining_cans
  (bought_cans : ℚ) / remaining_cans = 1 / 2 :=
by
  sorry

end NUMINAMATH_CALUDE_soda_can_ratio_l3419_341911


namespace NUMINAMATH_CALUDE_total_students_l3419_341928

/-- Proves that in a college with a boy-to-girl ratio of 8:5 and 400 girls, the total number of students is 1040 -/
theorem total_students (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 400) : boys + girls = 1040 := by
  sorry

end NUMINAMATH_CALUDE_total_students_l3419_341928


namespace NUMINAMATH_CALUDE_x_value_when_one_in_set_l3419_341952

theorem x_value_when_one_in_set (x : ℝ) : 1 ∈ ({x, x^2} : Set ℝ) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_x_value_when_one_in_set_l3419_341952


namespace NUMINAMATH_CALUDE_parametric_to_general_plane_equation_l3419_341907

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a parametric equation of a plane -/
structure ParametricPlane where
  origin : Point3D
  dir1 : Point3D
  dir2 : Point3D

/-- Represents the equation of a plane in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a plane equation satisfies the required conditions -/
def isValidPlaneEquation (eq : PlaneEquation) : Prop :=
  eq.A > 0 ∧ Int.gcd (Int.natAbs eq.A) (Int.gcd (Int.natAbs eq.B) (Int.gcd (Int.natAbs eq.C) (Int.natAbs eq.D))) = 1

/-- The main theorem stating the equivalence of the parametric and general forms of the plane -/
theorem parametric_to_general_plane_equation 
  (plane : ParametricPlane)
  (h_plane : plane = { 
    origin := { x := 3, y := 4, z := 1 },
    dir1 := { x := 1, y := -2, z := -1 },
    dir2 := { x := 2, y := 0, z := 1 }
  }) :
  ∃ (eq : PlaneEquation), 
    isValidPlaneEquation eq ∧
    (∀ (p : Point3D), 
      (∃ (s t : ℝ), 
        p.x = plane.origin.x + s * plane.dir1.x + t * plane.dir2.x ∧
        p.y = plane.origin.y + s * plane.dir1.y + t * plane.dir2.y ∧
        p.z = plane.origin.z + s * plane.dir1.z + t * plane.dir2.z) ↔
      eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0) ∧
    eq = { A := 2, B := 3, C := -4, D := -14 } :=
  sorry

end NUMINAMATH_CALUDE_parametric_to_general_plane_equation_l3419_341907


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3419_341948

/-- A quadratic function f(x) = ax^2 + (b-2)x + 3 where a ≠ 0 -/
def f (a b x : ℝ) : ℝ := a * x^2 + (b - 2) * x + 3

theorem quadratic_function_properties (a b : ℝ) (ha : a ≠ 0) :
  /- If the solution set of f(x) > 0 is (-1, 3), then a = -1 and b = 4 -/
  (∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) →
  (a = -1 ∧ b = 4) ∧
  /- If f(1) = 2, a > 0, and b > 0, then the minimum value of 1/a + 4/b is 9 -/
  (f a b 1 = 2 ∧ a > 0 ∧ b > 0 →
   ∀ a' b', a' > 0 → b' > 0 → f a' b' 1 = 2 → 1/a' + 4/b' ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3419_341948


namespace NUMINAMATH_CALUDE_problem_solution_l3419_341938

-- Define proposition p
def p : Prop := ∀ (x a : ℝ), x^2 + a*x + a^2 ≥ 0

-- Define proposition q
def q : Prop := ∃ (x : ℕ), x > 0 ∧ 2*x^2 - 1 ≤ 0

-- Theorem to prove
theorem problem_solution :
  p ∧ ¬q ∧ (p ∨ q) :=
sorry

end NUMINAMATH_CALUDE_problem_solution_l3419_341938


namespace NUMINAMATH_CALUDE_percent_relation_l3419_341982

theorem percent_relation (x y : ℝ) (h : (1/2) * (x - y) = (1/5) * (x + y)) :
  y = (3/7) * x := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l3419_341982


namespace NUMINAMATH_CALUDE_range_of_z_l3419_341964

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  ∃ (z : ℝ), z = x + y ∧ -Real.sqrt 6 ≤ z ∧ z ≤ Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_range_of_z_l3419_341964


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3419_341994

theorem quadratic_solution_sum (p q : ℝ) : 
  (∀ x : ℂ, 5 * x^2 - 7 * x + 20 = 0 ↔ x = p + q * I ∨ x = p - q * I) → 
  p + q^2 = 421 / 100 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3419_341994


namespace NUMINAMATH_CALUDE_intersection_implies_a_in_range_l3419_341999

def set_A (a : ℝ) : Set (ℝ × ℝ) := {p | p.2 = a * |p.1|}

def set_B (a : ℝ) : Set (ℝ × ℝ) := {p | p.2 = p.1 + a}

theorem intersection_implies_a_in_range (a : ℝ) :
  (∃! p, p ∈ set_A a ∩ set_B a) → a ∈ Set.Icc (-1 : ℝ) 1 := by
  sorry


end NUMINAMATH_CALUDE_intersection_implies_a_in_range_l3419_341999


namespace NUMINAMATH_CALUDE_union_A_complement_B_l3419_341934

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define set A
def A : Set ℕ := {1, 3}

-- Define set B
def B : Set ℕ := {2, 3, 4}

-- Theorem to prove
theorem union_A_complement_B : A ∪ (U \ B) = {1, 3, 5} := by
  sorry

end NUMINAMATH_CALUDE_union_A_complement_B_l3419_341934


namespace NUMINAMATH_CALUDE_james_money_theorem_l3419_341903

/-- The amount of money James has after finding some bills -/
def jamesTotal (billsFound : ℕ) (billValue : ℕ) (walletAmount : ℕ) : ℕ :=
  billsFound * billValue + walletAmount

/-- Theorem stating that James has $135 after finding 3 $20 bills -/
theorem james_money_theorem :
  jamesTotal 3 20 75 = 135 := by
  sorry

end NUMINAMATH_CALUDE_james_money_theorem_l3419_341903


namespace NUMINAMATH_CALUDE_inequality_proof_l3419_341941

theorem inequality_proof (a b c d : ℝ) 
  (non_neg_a : a ≥ 0) (non_neg_b : b ≥ 0) (non_neg_c : c ≥ 0) (non_neg_d : d ≥ 0)
  (sum_condition : a * b + b * c + c * d + d * a = 1) :
  a^3 / (b + c + d) + b^3 / (a + c + d) + c^3 / (a + b + d) + d^3 / (a + b + c) ≥ 1/3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3419_341941


namespace NUMINAMATH_CALUDE_mosaic_tile_size_l3419_341960

theorem mosaic_tile_size (height width : ℝ) (num_tiles : ℕ) (tile_side : ℝ) : 
  height = 10 → width = 15 → num_tiles = 21600 → 
  (height * width * 144) / num_tiles = tile_side^2 → tile_side = 1 := by
sorry

end NUMINAMATH_CALUDE_mosaic_tile_size_l3419_341960


namespace NUMINAMATH_CALUDE_most_suitable_sampling_method_l3419_341915

/-- Represents the age groups in the population --/
inductive AgeGroup
  | Elderly
  | MiddleAged
  | Young

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified
  | ExcludeOneElderlyThenStratified

/-- Represents the population composition --/
structure Population where
  elderly : Nat
  middleAged : Nat
  young : Nat

/-- Determines if a sampling method is suitable for a given population and sample size --/
def isSuitableMethod (pop : Population) (sampleSize : Nat) (method : SamplingMethod) : Prop :=
  sorry

/-- The theorem stating that excluding one elderly person and then using stratified sampling
    is the most suitable method for the given population and sample size --/
theorem most_suitable_sampling_method
  (pop : Population)
  (h1 : pop.elderly = 28)
  (h2 : pop.middleAged = 54)
  (h3 : pop.young = 81)
  (sampleSize : Nat)
  (h4 : sampleSize = 36) :
  isSuitableMethod pop sampleSize SamplingMethod.ExcludeOneElderlyThenStratified ∧
  ∀ m : SamplingMethod,
    isSuitableMethod pop sampleSize m →
    m = SamplingMethod.ExcludeOneElderlyThenStratified :=
  sorry


end NUMINAMATH_CALUDE_most_suitable_sampling_method_l3419_341915


namespace NUMINAMATH_CALUDE_double_windows_count_l3419_341957

/-- Represents the number of glass panels in each window -/
def panels_per_window : ℕ := 4

/-- Represents the number of single windows upstairs -/
def single_windows : ℕ := 8

/-- Represents the total number of glass panels in the house -/
def total_panels : ℕ := 80

/-- Represents the number of double windows downstairs -/
def double_windows : ℕ := 12

/-- Theorem stating that the number of double windows downstairs is 12 -/
theorem double_windows_count : 
  panels_per_window * double_windows + panels_per_window * single_windows = total_panels :=
by sorry

end NUMINAMATH_CALUDE_double_windows_count_l3419_341957


namespace NUMINAMATH_CALUDE_possible_values_of_a_l3419_341977

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 4 = 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | a * x - 2 = 0}

-- Define the condition that x ∈ A is necessary but not sufficient for x ∈ B
def necessary_not_sufficient (a : ℝ) : Prop :=
  B a ⊆ A ∧ B a ≠ A

-- Theorem statement
theorem possible_values_of_a :
  ∀ a : ℝ, necessary_not_sufficient a ↔ a ∈ ({-1, 0, 1} : Set ℝ) :=
by sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l3419_341977


namespace NUMINAMATH_CALUDE_potato_price_proof_l3419_341996

/-- The cost of one bag of potatoes from the farmer in rubles -/
def farmer_price : ℝ := sorry

/-- The number of bags each trader bought -/
def bags_bought : ℕ := 60

/-- Andrey's price increase percentage -/
def andrey_increase : ℝ := 1

/-- Boris's first price increase percentage -/
def boris_first_increase : ℝ := 0.6

/-- Boris's second price increase percentage -/
def boris_second_increase : ℝ := 0.4

/-- Number of bags Boris sold at first price -/
def boris_first_sale : ℕ := 15

/-- Number of bags Boris sold at second price -/
def boris_second_sale : ℕ := 45

/-- The difference in earnings between Boris and Andrey in rubles -/
def earnings_difference : ℝ := 1200

theorem potato_price_proof : 
  farmer_price = 250 :=
by
  have h1 : bags_bought * farmer_price * (1 + andrey_increase) = 
            bags_bought * farmer_price * (1 + boris_first_increase) * (boris_first_sale / bags_bought) + 
            bags_bought * farmer_price * (1 + boris_first_increase) * (1 + boris_second_increase) * (boris_second_sale / bags_bought) - 
            earnings_difference := by sorry
  sorry

end NUMINAMATH_CALUDE_potato_price_proof_l3419_341996


namespace NUMINAMATH_CALUDE_solve_for_a_l3419_341908

theorem solve_for_a (A : Set ℝ) (a : ℝ) 
  (h1 : A = {a - 2, 2 * a^2 + 5 * a, 12})
  (h2 : -3 ∈ A) : 
  a = -3/2 := by sorry

end NUMINAMATH_CALUDE_solve_for_a_l3419_341908


namespace NUMINAMATH_CALUDE_sin_cos_cube_sum_l3419_341909

theorem sin_cos_cube_sum (θ : ℝ) (h : Real.sin θ + Real.cos θ = 1/2) :
  Real.sin θ ^ 3 + Real.cos θ ^ 3 = 11/16 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_cube_sum_l3419_341909


namespace NUMINAMATH_CALUDE_valid_selections_count_l3419_341990

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  vertices : Fin n → Point

/-- Represents a 3600-gon -/
def BigPolygon : RegularPolygon 3600 := sorry

/-- Represents a 72-gon formed by red vertices -/
def RedPolygon : RegularPolygon 72 := sorry

/-- Predicate to check if a vertex is red -/
def isRed (v : Fin 3600) : Prop := sorry

/-- Represents a selection of 40 vertices -/
def Selection : Finset (Fin 3600) := sorry

/-- Predicate to check if a selection forms a regular 40-gon -/
def isRegular40gon (s : Finset (Fin 3600)) : Prop := sorry

/-- The number of ways to select 40 non-red vertices forming a regular 40-gon -/
def validSelections : ℕ := sorry

theorem valid_selections_count : validSelections = 81 := by sorry

end NUMINAMATH_CALUDE_valid_selections_count_l3419_341990


namespace NUMINAMATH_CALUDE_polynomial_roots_product_l3419_341937

theorem polynomial_roots_product (p q r s : ℝ) : 
  let Q : ℝ → ℝ := λ x => x^4 + p*x^3 + q*x^2 + r*x + s
  (Q (Real.cos (π/8)) = 0) ∧ 
  (Q (Real.cos (3*π/8)) = 0) ∧ 
  (Q (Real.cos (5*π/8)) = 0) ∧ 
  (Q (Real.cos (7*π/8)) = 0) →
  p * q * r * s = 0 := by
sorry

end NUMINAMATH_CALUDE_polynomial_roots_product_l3419_341937


namespace NUMINAMATH_CALUDE_correlation_theorem_l3419_341940

-- Define the types for our quantities
def Time := ℝ
def Displacement := ℝ
def Grade := ℝ
def Weight := ℝ
def DrunkDrivers := ℕ
def TrafficAccidents := ℕ
def Volume := ℝ

-- Define a type for pairs of quantities
structure QuantityPair where
  first : Type
  second : Type

-- Define our pairs
def uniformMotionPair : QuantityPair := ⟨Time, Displacement⟩
def gradeWeightPair : QuantityPair := ⟨Grade, Weight⟩
def drunkDriverAccidentPair : QuantityPair := ⟨DrunkDrivers, TrafficAccidents⟩
def volumeWeightPair : QuantityPair := ⟨Volume, Weight⟩

-- Define a predicate for correlation
def hasCorrelation (pair : QuantityPair) : Prop := sorry

-- Theorem statement
theorem correlation_theorem :
  ¬ hasCorrelation uniformMotionPair ∧
  ¬ hasCorrelation gradeWeightPair ∧
  hasCorrelation drunkDriverAccidentPair ∧
  ¬ hasCorrelation volumeWeightPair :=
sorry

end NUMINAMATH_CALUDE_correlation_theorem_l3419_341940


namespace NUMINAMATH_CALUDE_circle_equation_from_diameter_endpoints_l3419_341914

theorem circle_equation_from_diameter_endpoints (x y : ℝ) :
  let p₁ : ℝ × ℝ := (0, 0)
  let p₂ : ℝ × ℝ := (6, 8)
  let center : ℝ × ℝ := ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  let radius : ℝ := Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2) / 2
  (x - center.1)^2 + (y - center.2)^2 = radius^2 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_equation_from_diameter_endpoints_l3419_341914


namespace NUMINAMATH_CALUDE_set_difference_equals_singleton_l3419_341946

def M : Set ℕ := {x | 1 ≤ x ∧ x ≤ 2002}

def N : Set ℕ := {y | 2 ≤ y ∧ y ≤ 2003}

theorem set_difference_equals_singleton : N \ M = {2003} := by
  sorry

end NUMINAMATH_CALUDE_set_difference_equals_singleton_l3419_341946


namespace NUMINAMATH_CALUDE_complex_magnitude_product_l3419_341984

theorem complex_magnitude_product : 
  Complex.abs ((7 - 24 * Complex.I) * (-5 + 10 * Complex.I)) = 125 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_product_l3419_341984


namespace NUMINAMATH_CALUDE_intersection_M_N_l3419_341919

def M : Set ℤ := {m | -3 < m ∧ m < 2}
def N : Set ℤ := {n | -1 ≤ n ∧ n ≤ 3}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3419_341919


namespace NUMINAMATH_CALUDE_pens_to_sell_for_profit_l3419_341967

theorem pens_to_sell_for_profit (total_pens : ℕ) (cost_per_pen sell_price : ℚ) (desired_profit : ℚ) :
  total_pens = 2000 →
  cost_per_pen = 15/100 →
  sell_price = 30/100 →
  desired_profit = 150 →
  ∃ (pens_to_sell : ℕ), 
    pens_to_sell * sell_price - total_pens * cost_per_pen = desired_profit ∧
    pens_to_sell = 1500 :=
by sorry

end NUMINAMATH_CALUDE_pens_to_sell_for_profit_l3419_341967


namespace NUMINAMATH_CALUDE_pie_not_crust_percentage_l3419_341962

/-- Given a pie weighing 200 grams with 50 grams of crust, 
    the percentage of the pie that is not crust is 75%. -/
theorem pie_not_crust_percentage 
  (total_weight : ℝ) 
  (crust_weight : ℝ) 
  (h1 : total_weight = 200) 
  (h2 : crust_weight = 50) : 
  (total_weight - crust_weight) / total_weight * 100 = 75 := by
sorry

end NUMINAMATH_CALUDE_pie_not_crust_percentage_l3419_341962


namespace NUMINAMATH_CALUDE_range_of_a_l3419_341956

/-- Proposition p: The equation x^2 + ax + 1 = 0 has solutions -/
def p (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + a*x + 1 = 0

/-- Proposition q: For all x ∈ ℝ, e^(2x) - 2e^x + a ≥ 0 always holds -/
def q (a : ℝ) : Prop :=
  ∀ x : ℝ, Real.exp (2*x) - 2*(Real.exp x) + a ≥ 0

/-- The range of a given p ∧ q is true -/
theorem range_of_a (a : ℝ) (h : p a ∧ q a) : a ∈ Set.Ici (0 : ℝ) := by
  sorry

#check range_of_a

end NUMINAMATH_CALUDE_range_of_a_l3419_341956


namespace NUMINAMATH_CALUDE_inequality_proof_l3419_341981

theorem inequality_proof (a b : ℝ) : a^2 + b^2 + 2*(a-1)*(b-1) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3419_341981


namespace NUMINAMATH_CALUDE_f_2018_value_l3419_341920

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.exp x + x^2017

noncomputable def f_n : ℕ → (ℝ → ℝ)
| 0 => f
| (n+1) => deriv (f_n n)

theorem f_2018_value (x : ℝ) :
  f_n 2018 x = -Real.sin x + Real.exp x := by sorry

end NUMINAMATH_CALUDE_f_2018_value_l3419_341920


namespace NUMINAMATH_CALUDE_first_half_speed_l3419_341955

/-- Given a trip with the following properties:
  - Total distance is 50 km
  - First half (25 km) is traveled at speed v km/h
  - Second half (25 km) is traveled at 30 km/h
  - Average speed of the entire trip is 40 km/h
  Then the speed v of the first half of the trip is 100/3 km/h. -/
theorem first_half_speed (v : ℝ) : 
  v > 0 → -- Ensure v is positive
  (25 / v + 25 / 30) * 40 = 50 → -- Average speed equation
  v = 100 / 3 := by
sorry


end NUMINAMATH_CALUDE_first_half_speed_l3419_341955


namespace NUMINAMATH_CALUDE_austin_surfboard_length_l3419_341930

/-- Austin's surfing problem -/
theorem austin_surfboard_length 
  (H : ℝ) -- Austin's height
  (S : ℝ) -- Austin's surfboard length
  (highest_wave : 4 * H + 2 = 26) -- Highest wave is 2 feet higher than 4 times Austin's height
  (shortest_wave_height : H + 4 = S + 3) -- Shortest wave is 4 feet higher than Austin's height and 3 feet higher than surfboard length
  : S = 7 := by
  sorry


end NUMINAMATH_CALUDE_austin_surfboard_length_l3419_341930


namespace NUMINAMATH_CALUDE_exponential_problem_l3419_341983

theorem exponential_problem (a x y : ℝ) (h1 : a^x = 2) (h2 : a^y = 3) :
  a^(2*x + 3*y) = 108 := by
  sorry

end NUMINAMATH_CALUDE_exponential_problem_l3419_341983


namespace NUMINAMATH_CALUDE_max_volume_container_l3419_341905

/-- Represents the dimensions and volume of a rectangular container --/
structure Container where
  shorter_side : ℝ
  longer_side : ℝ
  height : ℝ
  volume : ℝ

/-- Calculates the volume of a container given its dimensions --/
def calculate_volume (c : Container) : ℝ :=
  c.shorter_side * c.longer_side * c.height

/-- Defines the constraints for the container based on the problem --/
def is_valid_container (c : Container) : Prop :=
  c.longer_side = c.shorter_side + 0.5 ∧
  c.height = 3.2 - 2 * c.shorter_side ∧
  4 * (c.shorter_side + c.longer_side + c.height) = 14.8 ∧
  c.volume = calculate_volume c

/-- Theorem stating the maximum volume and corresponding height --/
theorem max_volume_container :
  ∃ (c : Container), is_valid_container c ∧
    c.volume = 1.8 ∧
    c.height = 1.2 ∧
    ∀ (c' : Container), is_valid_container c' → c'.volume ≤ c.volume :=
  sorry

end NUMINAMATH_CALUDE_max_volume_container_l3419_341905


namespace NUMINAMATH_CALUDE_product_ratio_theorem_l3419_341971

theorem product_ratio_theorem (a b c d e f : ℚ) 
  (h1 : a * b * c = 65)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : d * e * f = 250) :
  (a * f) / (c * d) = 1/4 := by
sorry

end NUMINAMATH_CALUDE_product_ratio_theorem_l3419_341971


namespace NUMINAMATH_CALUDE_correct_calculation_l3419_341988

theorem correct_calculation (x : ℤ) : 63 - x = 70 → 36 + x = 29 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l3419_341988


namespace NUMINAMATH_CALUDE_lunch_solution_l3419_341980

def lunch_problem (total_spent : ℕ) (friend_spent : ℕ) : Prop :=
  friend_spent > total_spent - friend_spent ∧
  friend_spent - (total_spent - friend_spent) = 3

theorem lunch_solution :
  lunch_problem 19 11 := by sorry

end NUMINAMATH_CALUDE_lunch_solution_l3419_341980


namespace NUMINAMATH_CALUDE_taehyung_candies_l3419_341906

def total_candies : ℕ := 6
def seokjin_eats : ℕ := 4

theorem taehyung_candies : total_candies - seokjin_eats = 2 := by
  sorry

end NUMINAMATH_CALUDE_taehyung_candies_l3419_341906


namespace NUMINAMATH_CALUDE_complex_modulus_l3419_341947

theorem complex_modulus (z : ℂ) : i * z = Real.sqrt 2 - i → Complex.abs z = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3419_341947


namespace NUMINAMATH_CALUDE_image_and_preimage_of_f_l3419_341953

def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 - p.2, p.1 + p.2)

theorem image_and_preimage_of_f :
  (f (3, 5) = (-2, 8)) ∧ (f (4, 1) = (-2, 8)) := by sorry

end NUMINAMATH_CALUDE_image_and_preimage_of_f_l3419_341953


namespace NUMINAMATH_CALUDE_soda_bottle_difference_l3419_341969

theorem soda_bottle_difference (regular_soda : ℕ) (diet_soda : ℕ)
  (h1 : regular_soda = 81)
  (h2 : diet_soda = 60) :
  regular_soda - diet_soda = 21 := by
  sorry

end NUMINAMATH_CALUDE_soda_bottle_difference_l3419_341969


namespace NUMINAMATH_CALUDE_arithmetic_associativity_l3419_341961

theorem arithmetic_associativity (a b c : ℚ) : 
  ((a + b) + c = a + (b + c)) ∧
  ((a - b) - c ≠ a - (b - c)) ∧
  ((a * b) * c = a * (b * c)) ∧
  (a / b / c ≠ a / (b / c)) := by
  sorry

#check arithmetic_associativity

end NUMINAMATH_CALUDE_arithmetic_associativity_l3419_341961


namespace NUMINAMATH_CALUDE_quadratic_always_has_real_roots_k_range_for_positive_root_less_than_one_l3419_341989

variable (k : ℝ)

def quadratic_equation (x : ℝ) : Prop :=
  x^2 - (k+3)*x + 2*k + 2 = 0

theorem quadratic_always_has_real_roots :
  ∃ x₁ x₂ : ℝ, quadratic_equation k x₁ ∧ quadratic_equation k x₂ :=
sorry

theorem k_range_for_positive_root_less_than_one :
  (∃ x : ℝ, quadratic_equation k x ∧ 0 < x ∧ x < 1) → -1 < k ∧ k < 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_always_has_real_roots_k_range_for_positive_root_less_than_one_l3419_341989


namespace NUMINAMATH_CALUDE_quadratic_single_solution_l3419_341965

theorem quadratic_single_solution (b : ℝ) (hb : b ≠ 0) :
  (∃! x : ℝ, b * x^2 + 16 * x + 5 = 0) →
  (∃ x : ℝ, b * x^2 + 16 * x + 5 = 0 ∧ x = -5/8) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_single_solution_l3419_341965


namespace NUMINAMATH_CALUDE_four_inequalities_l3419_341943

theorem four_inequalities :
  (∃ (x : ℝ), x = Real.sqrt (2 * Real.sqrt (2 * Real.sqrt (2 * Real.sqrt 2))) ∧ x < 2) ∧
  (∃ (y : ℝ), y = Real.sqrt (2 + Real.sqrt (2 + Real.sqrt (2 + Real.sqrt 2))) ∧ y < 2) ∧
  (∃ (z : ℝ), z = Real.sqrt (3 * Real.sqrt (3 * Real.sqrt (3 * Real.sqrt 3))) ∧ z < 3) ∧
  (∃ (w : ℝ), w = Real.sqrt (3 + Real.sqrt (3 + Real.sqrt (3 + Real.sqrt 3))) ∧ w < 3) :=
by sorry

end NUMINAMATH_CALUDE_four_inequalities_l3419_341943


namespace NUMINAMATH_CALUDE_prob_two_tails_two_heads_proof_l3419_341900

/-- The probability of getting exactly two tails and two heads when four fair coins are tossed simultaneously -/
def prob_two_tails_two_heads : ℚ := 3/8

/-- The number of ways to choose 2 items from 4 items -/
def choose_two_from_four : ℕ := 6

/-- The probability of a specific sequence of two tails and two heads -/
def prob_specific_sequence : ℚ := 1/16

theorem prob_two_tails_two_heads_proof :
  prob_two_tails_two_heads = choose_two_from_four * prob_specific_sequence :=
by sorry

end NUMINAMATH_CALUDE_prob_two_tails_two_heads_proof_l3419_341900


namespace NUMINAMATH_CALUDE_water_bill_calculation_l3419_341933

/-- Water bill calculation for a household --/
theorem water_bill_calculation 
  (a : ℝ) -- Base rate for water usage up to 20 cubic meters
  (usage : ℝ) -- Total water usage
  (h1 : usage = 25) -- The household used 25 cubic meters
  (h2 : usage > 20) -- Usage exceeds 20 cubic meters
  : 
  (min usage 20) * a + (usage - 20) * (a + 3) = 25 * a + 15 :=
by sorry

end NUMINAMATH_CALUDE_water_bill_calculation_l3419_341933


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3419_341974

theorem square_plus_reciprocal_square (x : ℝ) (h : x ≠ 0) :
  x^2 + 1/x^2 = 2 → x^4 + 1/x^4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l3419_341974


namespace NUMINAMATH_CALUDE_largest_number_in_set_l3419_341912

/-- Given a = -3, -4a is the largest number in the set {-4a, 3a, 36/a, a^3, 2} -/
theorem largest_number_in_set (a : ℝ) (h : a = -3) :
  (-4 * a) = max (-4 * a) (max (3 * a) (max (36 / a) (max (a ^ 3) 2))) :=
by sorry

end NUMINAMATH_CALUDE_largest_number_in_set_l3419_341912


namespace NUMINAMATH_CALUDE_yunhwan_water_consumption_l3419_341968

/-- Yunhwan's yearly water consumption in liters -/
def yearly_water_consumption (monthly_consumption : ℝ) (months_per_year : ℕ) : ℝ :=
  monthly_consumption * months_per_year

/-- Proof that Yunhwan's yearly water consumption is 2194.56 liters -/
theorem yunhwan_water_consumption : 
  yearly_water_consumption 182.88 12 = 2194.56 := by
  sorry

end NUMINAMATH_CALUDE_yunhwan_water_consumption_l3419_341968


namespace NUMINAMATH_CALUDE_A9_coordinates_l3419_341973

/-- Define a sequence of points in a Cartesian coordinate system -/
def A (n : ℕ) : ℝ × ℝ := (n, n^2)

/-- Theorem: The 9th point in the sequence has coordinates (9, 81) -/
theorem A9_coordinates : A 9 = (9, 81) := by
  sorry

end NUMINAMATH_CALUDE_A9_coordinates_l3419_341973


namespace NUMINAMATH_CALUDE_composite_form_l3419_341921

theorem composite_form (x : ℤ) (m n : ℕ) (hm : m > 0) (hn : n ≥ 0) :
  x^(4*m) + 2^(4*n + 2) = (x^(2*m) + 2^(2*n + 1) + 2^(n + 1) * x^m) * ((x^m - 2^n)^2 + 2^(2*n)) :=
by sorry

end NUMINAMATH_CALUDE_composite_form_l3419_341921


namespace NUMINAMATH_CALUDE_volume_of_T_l3419_341902

/-- The solid T in ℝ³ -/
def T : Set (ℝ × ℝ × ℝ) :=
  {p : ℝ × ℝ × ℝ | let (x, y, z) := p
                   (|x| + |y| ≤ 2) ∧ (|x| + |z| ≤ 2) ∧ (|y| + |z| ≤ 2)}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (S : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating the volume of T -/
theorem volume_of_T : volume T = 32 * Real.sqrt 3 / 9 := by sorry

end NUMINAMATH_CALUDE_volume_of_T_l3419_341902


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3419_341972

theorem quadratic_inequality_solution_set (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - (m + 3) * x - 1 < 0) ↔ (-9 < m ∧ m < -1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3419_341972


namespace NUMINAMATH_CALUDE_prime_triplets_theorem_l3419_341998

/-- A prime triplet (a, b, c) satisfying the given conditions -/
structure PrimeTriplet where
  a : Nat
  b : Nat
  c : Nat
  h1 : a < b
  h2 : b < c
  h3 : c < 100
  h4 : Nat.Prime a
  h5 : Nat.Prime b
  h6 : Nat.Prime c
  h7 : (b + 1 - (a + 1)) * (c + 1 - (b + 1)) = (b + 1) * (b + 1 - (a + 1))

/-- The set of all valid prime triplets -/
def validTriplets : Set PrimeTriplet := {
  ⟨2, 5, 11, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩,
  ⟨5, 11, 23, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩,
  ⟨7, 11, 23, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩,
  ⟨11, 23, 47, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩
}

/-- The main theorem -/
theorem prime_triplets_theorem :
  ∀ t : PrimeTriplet, t ∈ validTriplets := by
  sorry

end NUMINAMATH_CALUDE_prime_triplets_theorem_l3419_341998


namespace NUMINAMATH_CALUDE_bottle_caps_per_box_l3419_341931

theorem bottle_caps_per_box (total_caps : ℕ) (num_boxes : ℚ) (caps_per_box : ℕ) :
  total_caps = 245 →
  num_boxes = 7 →
  caps_per_box * num_boxes = total_caps →
  caps_per_box = 35 := by
  sorry

end NUMINAMATH_CALUDE_bottle_caps_per_box_l3419_341931


namespace NUMINAMATH_CALUDE_parallel_vectors_sum_l3419_341992

/-- Given two vectors a and b in R³, if they are parallel and have specific components,
    then the sum of their unknown components is -7. -/
theorem parallel_vectors_sum (x y : ℝ) :
  let a : ℝ × ℝ × ℝ := (2, x, 3)
  let b : ℝ × ℝ × ℝ := (-4, 2, y)
  (∃ (k : ℝ), a = k • b) →
  x + y = -7 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_sum_l3419_341992


namespace NUMINAMATH_CALUDE_multiples_of_seven_square_l3419_341910

theorem multiples_of_seven_square (a b : ℕ) : 
  (∀ k : ℕ, k ≤ a → (7 * k < 50)) ∧ 
  (∀ k : ℕ, k > a → (7 * k ≥ 50)) ∧
  (∀ k : ℕ, k ≤ b → (k * 7 < 50 ∧ k > 0)) ∧
  (∀ k : ℕ, k > b → (k * 7 ≥ 50 ∨ k ≤ 0)) →
  (a + b)^2 = 196 := by
sorry

end NUMINAMATH_CALUDE_multiples_of_seven_square_l3419_341910


namespace NUMINAMATH_CALUDE_chord_length_squared_l3419_341944

/-- Two circles with radii 10 and 7, centers 15 units apart, intersecting at P.
    A line through P creates equal chords QP and PR. -/
structure IntersectingCircles where
  r₁ : ℝ
  r₂ : ℝ
  center_distance : ℝ
  chord_length : ℝ
  h₁ : r₁ = 10
  h₂ : r₂ = 7
  h₃ : center_distance = 15
  h₄ : chord_length > 0

/-- The square of the length of chord QP in the given configuration is 289. -/
theorem chord_length_squared (c : IntersectingCircles) : c.chord_length ^ 2 = 289 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_squared_l3419_341944


namespace NUMINAMATH_CALUDE_right_triangle_from_medians_l3419_341924

theorem right_triangle_from_medians (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 
    x^2 = (16 * b^2 - 4 * a^2) / 15 ∧
    y^2 = (16 * a^2 - 4 * b^2) / 15 ∧
    x^2 + y^2 = a^2 + b^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_from_medians_l3419_341924


namespace NUMINAMATH_CALUDE_exists_positive_decreasing_function_l3419_341987

theorem exists_positive_decreasing_function :
  ∃ f : ℝ → ℝ, (∀ x y : ℝ, x < y → f y < f x) ∧ (∀ x : ℝ, f x > 0) := by
  sorry

end NUMINAMATH_CALUDE_exists_positive_decreasing_function_l3419_341987


namespace NUMINAMATH_CALUDE_quadratic_solution_difference_l3419_341945

theorem quadratic_solution_difference : ∃ (x₁ x₂ : ℝ), 
  (x₁^2 - 5*x₁ + 15 = x₁ + 55) ∧ 
  (x₂^2 - 5*x₂ + 15 = x₂ + 55) ∧ 
  (x₁ ≠ x₂) ∧ 
  (|x₁ - x₂| = 14) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_difference_l3419_341945


namespace NUMINAMATH_CALUDE_infinitely_many_coprime_phi_m_root_l3419_341995

/-- m-th iteration of Euler's totient function -/
def phi_m (m : ℕ) : ℕ → ℕ :=
  match m with
  | 0 => id
  | m + 1 => phi_m m ∘ Nat.totient

/-- Main theorem -/
theorem infinitely_many_coprime_phi_m_root (a b m k : ℕ) (hk : k ≥ 2) :
  ∃ S : Set ℕ, S.Infinite ∧ ∀ n ∈ S, Nat.gcd (phi_m m n) (Nat.floor ((a * n + b : ℝ) ^ (1 / k))) = 1 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_coprime_phi_m_root_l3419_341995


namespace NUMINAMATH_CALUDE_total_sandwiches_l3419_341958

/-- Represents the number of sandwiches of each type -/
structure Sandwiches where
  cheese : ℕ
  bologna : ℕ
  peanutButter : ℕ

/-- The ratio of sandwich types -/
def sandwichRatio : Sandwiches :=
  { cheese := 1
    bologna := 7
    peanutButter := 8 }

/-- Theorem: Given the sandwich ratio and the number of bologna sandwiches,
    prove the total number of sandwiches -/
theorem total_sandwiches
    (ratio : Sandwiches)
    (bologna_count : ℕ)
    (h1 : ratio = sandwichRatio)
    (h2 : bologna_count = 35) :
    ratio.cheese * (bologna_count / ratio.bologna) +
    bologna_count +
    ratio.peanutButter * (bologna_count / ratio.bologna) = 80 := by
  sorry

#check total_sandwiches

end NUMINAMATH_CALUDE_total_sandwiches_l3419_341958


namespace NUMINAMATH_CALUDE_square_sum_equality_l3419_341917

theorem square_sum_equality (p q r a b c : ℝ) 
  (h1 : p + q + r = 1) 
  (h2 : 1/p + 1/q + 1/r = 0) : 
  a^2 + b^2 + c^2 = (p*a + q*b + r*c)^2 + (q*a + r*b + p*c)^2 + (r*a + p*b + q*c)^2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equality_l3419_341917


namespace NUMINAMATH_CALUDE_cooperation_is_best_l3419_341979

/-- Represents a factory with its daily processing capacity and fee -/
structure Factory where
  capacity : ℕ
  fee : ℕ

/-- Represents a processing plan with its duration and total cost -/
structure Plan where
  duration : ℕ
  cost : ℕ

/-- Calculates the plan for a single factory -/
def single_factory_plan (f : Factory) (total_products : ℕ) (engineer_fee : ℕ) : Plan :=
  let duration := total_products / f.capacity
  { duration := duration
  , cost := duration * (f.fee + engineer_fee) }

/-- Calculates the plan for two factories cooperating -/
def cooperation_plan (f1 f2 : Factory) (total_products : ℕ) (engineer_fee : ℕ) : Plan :=
  let duration := total_products / (f1.capacity + f2.capacity)
  { duration := duration
  , cost := duration * (f1.fee + f2.fee + engineer_fee) }

/-- Checks if one plan is better than another -/
def is_better_plan (p1 p2 : Plan) : Prop :=
  p1.duration < p2.duration ∧ p1.cost < p2.cost

theorem cooperation_is_best (total_products engineer_fee : ℕ) :
  let factory_a : Factory := { capacity := 16, fee := 80 }
  let factory_b : Factory := { capacity := 24, fee := 120 }
  let plan_a := single_factory_plan factory_a total_products engineer_fee
  let plan_b := single_factory_plan factory_b total_products engineer_fee
  let plan_coop := cooperation_plan factory_a factory_b total_products engineer_fee
  total_products = 960 ∧
  engineer_fee = 10 ∧
  factory_a.capacity * 3 = factory_b.capacity * 2 ∧
  factory_a.capacity + factory_b.capacity = 40 →
  is_better_plan plan_coop plan_a ∧ is_better_plan plan_coop plan_b :=
by sorry

end NUMINAMATH_CALUDE_cooperation_is_best_l3419_341979


namespace NUMINAMATH_CALUDE_probability_face_then_number_standard_deck_l3419_341942

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : ℕ)
  (ranks : ℕ)
  (suits : ℕ)
  (face_cards_per_suit : ℕ)
  (number_cards_per_suit : ℕ)

/-- The probability of drawing a face card first and a number card second from a standard deck -/
def probability_face_then_number (d : Deck) : ℚ :=
  let total_face_cards := d.face_cards_per_suit * d.suits
  let total_number_cards := d.number_cards_per_suit * d.suits
  (total_face_cards * total_number_cards : ℚ) / (d.total_cards * (d.total_cards - 1))

/-- Theorem stating the probability of drawing a face card first and a number card second from a standard deck -/
theorem probability_face_then_number_standard_deck :
  let d : Deck := {
    total_cards := 52,
    ranks := 13,
    suits := 4,
    face_cards_per_suit := 3,
    number_cards_per_suit := 9
  }
  probability_face_then_number d = 8 / 49 := by sorry

end NUMINAMATH_CALUDE_probability_face_then_number_standard_deck_l3419_341942


namespace NUMINAMATH_CALUDE_min_value_of_z_l3419_341985

/-- Given a system of linear inequalities, prove that the minimum value of z = 2x + y is 4 -/
theorem min_value_of_z (x y : ℝ) 
  (h1 : 2 * x - y ≥ 0) 
  (h2 : x + y - 3 ≥ 0) 
  (h3 : y - x ≥ 0) : 
  ∃ (z : ℝ), z = 2 * x + y ∧ z ≥ 4 ∧ ∀ (w : ℝ), w = 2 * x + y → w ≥ z :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_z_l3419_341985


namespace NUMINAMATH_CALUDE_valid_coloring_iff_even_product_l3419_341951

/-- Represents a chessboard coloring where each small square not on the perimeter has exactly two sides colored. -/
def ValidColoring (m n : ℕ) := True  -- Placeholder definition

/-- Theorem stating that a valid coloring exists if and only if m * n is even -/
theorem valid_coloring_iff_even_product (m n : ℕ) :
  ValidColoring m n ↔ Even (m * n) :=
by sorry

end NUMINAMATH_CALUDE_valid_coloring_iff_even_product_l3419_341951


namespace NUMINAMATH_CALUDE_jude_bottle_cap_trading_l3419_341927

/-- Jude's bottle cap trading problem -/
theorem jude_bottle_cap_trading
  (initial_caps : ℕ)
  (car_cost : ℕ)
  (truck_cost : ℕ)
  (trucks_bought : ℕ)
  (total_vehicles : ℕ)
  (h1 : initial_caps = 100)
  (h2 : car_cost = 5)
  (h3 : truck_cost = 6)
  (h4 : trucks_bought = 10)
  (h5 : total_vehicles = 16) :
  (car_cost * (total_vehicles - trucks_bought) : ℚ) / (initial_caps - truck_cost * trucks_bought) = 3/4 := by
  sorry


end NUMINAMATH_CALUDE_jude_bottle_cap_trading_l3419_341927


namespace NUMINAMATH_CALUDE_number_ratio_l3419_341918

theorem number_ratio (A B C : ℝ) : 
  A + B + C = 110 → A = 2 * B → B = 30 → C / A = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_number_ratio_l3419_341918


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3419_341923

theorem quadratic_inequality_solution_set (a : ℝ) (ha : a > 0) :
  {x : ℝ | x^2 - 4*a*x - 5*a^2 < 0} = {x : ℝ | -a < x ∧ x < 5*a} := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3419_341923


namespace NUMINAMATH_CALUDE_set_operations_l3419_341950

open Set

-- Define the sets A and B
def A : Set ℝ := {x | -5 < x ∧ x < 2}
def B : Set ℝ := {x | -3 < x ∧ x ≤ 3}

-- Define the theorem
theorem set_operations :
  (A ∩ B = Ioc (-3) 2) ∧
  (A ∪ B = Ioc (-5) 3) ∧
  (Aᶜ = Iic (-5) ∪ Ici 2) ∧
  ((A ∩ B)ᶜ = Iic (-3) ∪ Ioi 2) ∧
  (Aᶜ ∩ B = Icc 2 3) :=
by sorry

end NUMINAMATH_CALUDE_set_operations_l3419_341950


namespace NUMINAMATH_CALUDE_sin_cos_cube_difference_l3419_341975

theorem sin_cos_cube_difference (α : ℝ) (n : ℝ) (h : Real.sin α - Real.cos α = n) :
  Real.sin α ^ 3 - Real.cos α ^ 3 = (3 * n - n^3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_cube_difference_l3419_341975


namespace NUMINAMATH_CALUDE_problem_statement_l3419_341939

theorem problem_statement (x : ℝ) (Q : ℝ) (h : 5 * (6 * x - 3 * Real.pi) = Q) :
  15 * (18 * x - 9 * Real.pi) = 9 * Q := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3419_341939


namespace NUMINAMATH_CALUDE_box_packing_problem_l3419_341976

theorem box_packing_problem (x y : ℤ) : 
  (3 * x + 4 * y = 108) → 
  (2 * x + 3 * y = 76) → 
  (x = 20 ∧ y = 12) := by
sorry

end NUMINAMATH_CALUDE_box_packing_problem_l3419_341976


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3419_341959

theorem expression_simplification_and_evaluation (x : ℝ) (h : x = 1 / (Real.sqrt 2 - 1)) :
  (1 - 4 / (x + 3)) / ((x^2 - 2*x + 1) / (2*x + 6)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3419_341959


namespace NUMINAMATH_CALUDE_quadratic_inequalities_l3419_341993

theorem quadratic_inequalities :
  (∀ y : ℝ, y^2 + 4*y + 8 ≥ 4) ∧
  (∀ m : ℝ, m^2 + 2*m + 3 ≥ 2) ∧
  (∀ m : ℝ, -m^2 + 2*m + 3 ≤ 4) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l3419_341993


namespace NUMINAMATH_CALUDE_rectangle_count_l3419_341978

/-- Given a rectangle with dimensions a and b where a < b, this theorem states that
    the number of rectangles with dimensions x and y satisfying the specified conditions
    is either 0 or 1. -/
theorem rectangle_count (a b x y : ℝ) (h1 : 0 < a) (h2 : a < b) : 
  (x < a ∧ y < a ∧ 
   2*(x + y) = (1/2)*(a + b) ∧ 
   x*y = (1/4)*a*b) → 
  (∃! p : ℝ × ℝ, p.1 < a ∧ p.2 < a ∧ 
                 2*(p.1 + p.2) = (1/2)*(a + b) ∧ 
                 p.1*p.2 = (1/4)*a*b) ∨
  (¬ ∃ p : ℝ × ℝ, p.1 < a ∧ p.2 < a ∧ 
                  2*(p.1 + p.2) = (1/2)*(a + b) ∧ 
                  p.1*p.2 = (1/4)*a*b) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_count_l3419_341978


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3419_341970

theorem geometric_sequence_ratio (q : ℝ) (S : ℕ → ℝ) (a : ℕ → ℝ) :
  q = 1/2 →
  (∀ n, S n = a 1 * (1 - q^n) / (1 - q)) →
  (∀ n, a (n + 1) = a n * q) →
  S 4 / a 3 = 15/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3419_341970


namespace NUMINAMATH_CALUDE_roots_expression_l3419_341932

theorem roots_expression (p q α β γ δ : ℝ) : 
  (α^2 - p*α + 1 = 0) → 
  (β^2 - p*β + 1 = 0) → 
  (γ^2 - q*γ + 1 = 0) → 
  (δ^2 - q*δ + 1 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = p^2 - q^2 := by
  sorry

end NUMINAMATH_CALUDE_roots_expression_l3419_341932


namespace NUMINAMATH_CALUDE_min_S_19_l3419_341904

/-- Given an arithmetic sequence {a_n} where S_8 ≤ 6 and S_11 ≥ 27, 
    the minimum value of S_19 is 133. -/
theorem min_S_19 (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, S n = (n * (a 1 + a n)) / 2) →  -- Definition of S_n
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →  -- Definition of arithmetic sequence
  S 8 ≤ 6 →                                 -- Given condition
  S 11 ≥ 27 →                               -- Given condition
  ∀ S_19 : ℝ, (S_19 = S 19 → S_19 ≥ 133) :=
by sorry

end NUMINAMATH_CALUDE_min_S_19_l3419_341904


namespace NUMINAMATH_CALUDE_exists_a_for_even_f_l3419_341922

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x

theorem exists_a_for_even_f : ∃ a : ℝ, ∀ x : ℝ, f a x = f a (-x) := by
  sorry

end NUMINAMATH_CALUDE_exists_a_for_even_f_l3419_341922


namespace NUMINAMATH_CALUDE_spiral_strip_length_l3419_341925

/-- The length of a spiral strip on a right circular cylinder -/
theorem spiral_strip_length (base_circumference height : ℝ) 
  (h_base : base_circumference = 18)
  (h_height : height = 8) :
  Real.sqrt (height^2 + base_circumference^2) = Real.sqrt 388 := by
  sorry

end NUMINAMATH_CALUDE_spiral_strip_length_l3419_341925


namespace NUMINAMATH_CALUDE_dan_marbles_l3419_341986

/-- The number of marbles Dan has after giving some away -/
def remaining_marbles (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that Dan has 96 marbles after giving away 32 from his initial 128 -/
theorem dan_marbles : remaining_marbles 128 32 = 96 := by
  sorry

end NUMINAMATH_CALUDE_dan_marbles_l3419_341986


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3419_341954

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 - 6*x + 8 = 0) → (∃ r₁ r₂ : ℝ, r₁ + r₂ = 6 ∧ x^2 - 6*x + 8 = (x - r₁) * (x - r₂)) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l3419_341954


namespace NUMINAMATH_CALUDE_circle_point_range_l3419_341991

theorem circle_point_range (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) → (-1 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_circle_point_range_l3419_341991


namespace NUMINAMATH_CALUDE_ice_cream_sundaes_l3419_341949

theorem ice_cream_sundaes (n : ℕ) (h : n = 8) : 
  Nat.choose n 2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_sundaes_l3419_341949


namespace NUMINAMATH_CALUDE_books_printed_count_l3419_341936

def pages_per_book : ℕ := 600
def pages_per_sheet : ℕ := 8  -- 4 pages per side, double-sided
def sheets_used : ℕ := 150

theorem books_printed_count :
  (sheets_used * pages_per_sheet) / pages_per_book = 2 :=
by sorry

end NUMINAMATH_CALUDE_books_printed_count_l3419_341936


namespace NUMINAMATH_CALUDE_cycling_distance_l3419_341966

theorem cycling_distance (rate : ℝ) (time : ℝ) (distance : ℝ) : 
  rate = 8 → time = 2.25 → distance = rate * time → distance = 18 := by
sorry

end NUMINAMATH_CALUDE_cycling_distance_l3419_341966


namespace NUMINAMATH_CALUDE_eighty_sixth_word_ends_with_E_l3419_341901

-- Define the set of letters
inductive Letter : Type
| A | H | S | M | E

-- Define a permutation as a list of letters
def Permutation := List Letter

-- Define the dictionary order for permutations
def dict_order (p1 p2 : Permutation) : Prop := sorry

-- Define a function to get the nth permutation in dictionary order
def nth_permutation (n : Nat) : Permutation := sorry

-- Define a function to get the last letter of a permutation
def last_letter (p : Permutation) : Letter := sorry

-- State the theorem
theorem eighty_sixth_word_ends_with_E : 
  last_letter (nth_permutation 86) = Letter.E := by sorry

end NUMINAMATH_CALUDE_eighty_sixth_word_ends_with_E_l3419_341901


namespace NUMINAMATH_CALUDE_committee_selection_count_l3419_341926

theorem committee_selection_count : Nat.choose 30 5 = 142506 := by sorry

end NUMINAMATH_CALUDE_committee_selection_count_l3419_341926


namespace NUMINAMATH_CALUDE_pipe_filling_time_l3419_341997

theorem pipe_filling_time (fill_rate_A fill_rate_B : ℝ) : 
  fill_rate_A = 2 / 75 →
  9 * (fill_rate_A + fill_rate_B) + 21 * fill_rate_A = 1 →
  fill_rate_B = 1 / 45 :=
by sorry

end NUMINAMATH_CALUDE_pipe_filling_time_l3419_341997
