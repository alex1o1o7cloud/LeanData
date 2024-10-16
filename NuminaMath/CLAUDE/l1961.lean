import Mathlib

namespace NUMINAMATH_CALUDE_equivalent_operation_l1961_196186

theorem equivalent_operation (x : ℚ) : 
  (x * (2 / 3)) / (4 / 7) = x * (7 / 6) :=
by sorry

end NUMINAMATH_CALUDE_equivalent_operation_l1961_196186


namespace NUMINAMATH_CALUDE_inequality_proof_l1961_196195

theorem inequality_proof (a b c : ℝ) 
  (ha : a = (1/6) * Real.log 8)
  (hb : b = (1/2) * Real.log 5)
  (hc : c = Real.log (Real.sqrt 6) - Real.log (Real.sqrt 2)) :
  a < c ∧ c < b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1961_196195


namespace NUMINAMATH_CALUDE_problem_one_problem_two_l1961_196177

-- Problem 1
theorem problem_one (x n : ℝ) (h : x^n = 2) : 
  (3*x^n)^2 - 4*(x^2)^n = 20 := by sorry

-- Problem 2
theorem problem_two (x y n : ℝ) (h1 : x = 2^n - 1) (h2 : y = 3 + 8^n) :
  y = 3 + (x + 1)^3 := by sorry

end NUMINAMATH_CALUDE_problem_one_problem_two_l1961_196177


namespace NUMINAMATH_CALUDE_ellipse_equation_l1961_196174

/-- Given an ellipse with foci at (-2,0) and (2,0) passing through the point (2√3, √3),
    its standard equation is x²/16 + y²/12 = 1 -/
theorem ellipse_equation (x y : ℝ) :
  let f1 : ℝ × ℝ := (-2, 0)
  let f2 : ℝ × ℝ := (2, 0)
  let p : ℝ × ℝ := (2 * Real.sqrt 3, Real.sqrt 3)
  let d1 := Real.sqrt ((x - f1.1)^2 + (y - f1.2)^2)
  let d2 := Real.sqrt ((x - f2.1)^2 + (y - f2.2)^2)
  let passing_point := d1 + d2 = Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2) +
                                 Real.sqrt ((p.1 - f2.1)^2 + (p.2 - f2.2)^2)
  passing_point → x^2 / 16 + y^2 / 12 = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1961_196174


namespace NUMINAMATH_CALUDE_different_color_probability_l1961_196167

def red : ℕ := 4
def green : ℕ := 5
def white : ℕ := 12
def blue : ℕ := 3

def total : ℕ := red + green + white + blue

theorem different_color_probability :
  let prob_diff_color := (red * green + red * white + red * blue +
                          green * white + green * blue + white * blue) /
                         (total * (total - 1))
  prob_diff_color = 191 / 552 := by sorry

end NUMINAMATH_CALUDE_different_color_probability_l1961_196167


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l1961_196182

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem planes_parallel_if_perpendicular_to_same_line 
  (m : Line) (α β : Plane) (hα : α ≠ β) :
  perpendicular m α → perpendicular m β → parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l1961_196182


namespace NUMINAMATH_CALUDE_smores_marshmallows_needed_l1961_196176

def graham_crackers : ℕ := 48
def marshmallows : ℕ := 6
def crackers_per_smore : ℕ := 2
def marshmallows_per_smore : ℕ := 1

theorem smores_marshmallows_needed : 
  (graham_crackers / crackers_per_smore) - marshmallows = 18 :=
by sorry

end NUMINAMATH_CALUDE_smores_marshmallows_needed_l1961_196176


namespace NUMINAMATH_CALUDE_star_divided_by_square_equals_sixteen_l1961_196101

-- Define symbols as natural numbers
variable (triangle circle square star : ℕ)

-- Define the conditions from the problem
axiom condition1 : triangle + triangle = star
axiom condition2 : circle = square + square
axiom condition3 : triangle = circle + circle + circle + circle

-- The theorem to prove
theorem star_divided_by_square_equals_sixteen : star / square = 16 := by
  sorry

end NUMINAMATH_CALUDE_star_divided_by_square_equals_sixteen_l1961_196101


namespace NUMINAMATH_CALUDE_tom_dance_years_l1961_196103

/-- The number of years Tom danced -/
def years_danced (
  dances_per_week : ℕ
) (
  hours_per_dance : ℕ
) (
  weeks_per_year : ℕ
) (
  total_hours_danced : ℕ
) : ℕ :=
  total_hours_danced / (dances_per_week * hours_per_dance * weeks_per_year)

theorem tom_dance_years :
  years_danced 4 2 52 4160 = 10 := by
  sorry

end NUMINAMATH_CALUDE_tom_dance_years_l1961_196103


namespace NUMINAMATH_CALUDE_circle_equation_with_diameter_PQ_l1961_196127

def P : ℝ × ℝ := (4, 0)
def Q : ℝ × ℝ := (0, 2)

theorem circle_equation_with_diameter_PQ :
  let center := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let radius_squared := ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) / 4
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = radius_squared ↔
    (x - 2)^2 + (y - 1)^2 = 5 := by sorry

end NUMINAMATH_CALUDE_circle_equation_with_diameter_PQ_l1961_196127


namespace NUMINAMATH_CALUDE_expressions_equality_l1961_196126

theorem expressions_equality (a b c : ℝ) :
  a + 2 * b * c = (a + b) * (a + 2 * c) ↔ a + b + 2 * c = 0 := by
  sorry

end NUMINAMATH_CALUDE_expressions_equality_l1961_196126


namespace NUMINAMATH_CALUDE_max_abs_sum_on_circle_l1961_196118

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) :
  ∃ (M : ℝ), M = 2 * Real.sqrt 2 ∧ |x| + |y| ≤ M ∧ ∃ (x₀ y₀ : ℝ), x₀^2 + y₀^2 = 4 ∧ |x₀| + |y₀| = M :=
by sorry

end NUMINAMATH_CALUDE_max_abs_sum_on_circle_l1961_196118


namespace NUMINAMATH_CALUDE_union_and_complement_find_a_l1961_196171

-- Part 1
def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_and_complement : 
  (A ∪ B = {x | 2 < x ∧ x < 10}) ∧ 
  ((Set.univ \ A) ∩ B = {x | 2 < x ∧ x < 3} ∪ {x | 7 ≤ x ∧ x < 10}) := by sorry

-- Part 2
def A' (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B' : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C' : Set ℝ := {x | x^2 + 2*x - 8 = 0}

theorem find_a : 
  ∃ a : ℝ, (A' a ∩ B' ≠ ∅) ∧ (A' a ∩ C' = ∅) ∧ (a = -2) := by sorry

end NUMINAMATH_CALUDE_union_and_complement_find_a_l1961_196171


namespace NUMINAMATH_CALUDE_natural_number_equality_integer_absolute_equality_l1961_196156

-- Part a
theorem natural_number_equality (x y n : ℕ) 
  (h : x^3 + 2^n * y = y^3 + 2^n * x) : x = y := by sorry

-- Part b
theorem integer_absolute_equality (x y : ℤ) (n : ℕ) 
  (hx : x ≠ 0) (hy : y ≠ 0)
  (h : x^3 + 2^n * y = y^3 + 2^n * x) : |x| = |y| := by sorry

end NUMINAMATH_CALUDE_natural_number_equality_integer_absolute_equality_l1961_196156


namespace NUMINAMATH_CALUDE_largest_four_digit_divisible_by_4_with_digit_sum_20_l1961_196161

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem largest_four_digit_divisible_by_4_with_digit_sum_20 :
  ∃ (n : ℕ), is_four_digit n ∧ n % 4 = 0 ∧ digit_sum n = 20 ∧
  ∀ (m : ℕ), is_four_digit m → m % 4 = 0 → digit_sum m = 20 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_four_digit_divisible_by_4_with_digit_sum_20_l1961_196161


namespace NUMINAMATH_CALUDE_triangle_properties_l1961_196143

-- Define the triangle ABC
def triangle_ABC (A B C : Real) (a b c : Real) : Prop :=
  a = 2 * Real.sqrt 2 ∧ b = 5 ∧ c = Real.sqrt 13

-- Theorem to prove the three parts of the problem
theorem triangle_properties {A B C a b c : Real} 
  (h : triangle_ABC A B C a b c) : 
  C = π / 4 ∧ 
  Real.sin A = 2 * Real.sqrt 13 / 13 ∧ 
  Real.sin (2 * A + π / 4) = 17 * Real.sqrt 2 / 26 := by
  sorry


end NUMINAMATH_CALUDE_triangle_properties_l1961_196143


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1961_196139

theorem pure_imaginary_condition (a : ℝ) : 
  let i : ℂ := Complex.I
  let z : ℂ := (a + i) / (1 + i)
  (z.re = 0) → a = -1 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1961_196139


namespace NUMINAMATH_CALUDE_same_side_line_range_l1961_196170

theorem same_side_line_range (a : ℝ) : 
  (∀ x y : ℝ, (x = 3 ∧ y = -1) ∨ (x = -1 ∧ y = 2) → 
    (a * x + 2 * y - 1) * (a * 3 + 2 * (-1) - 1) > 0) ↔ 
  a ∈ Set.Ioo 1 3 :=
sorry

end NUMINAMATH_CALUDE_same_side_line_range_l1961_196170


namespace NUMINAMATH_CALUDE_polyhedron_vertices_l1961_196144

/-- A polyhedron is a three-dimensional geometric object with flat polygonal faces, straight edges and sharp corners or vertices --/
structure Polyhedron where
  faces : Nat
  edges : Nat
  vertices : Nat

/-- Euler's formula for polyhedra states that for a polyhedron with F faces, E edges, and V vertices, F - E + V = 2 --/
axiom euler_formula (p : Polyhedron) : p.faces - p.edges + p.vertices = 2

/-- Theorem: A polyhedron with 6 faces and 12 edges has 8 vertices --/
theorem polyhedron_vertices (p : Polyhedron) (h1 : p.faces = 6) (h2 : p.edges = 12) : p.vertices = 8 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_vertices_l1961_196144


namespace NUMINAMATH_CALUDE_hot_dogs_remainder_l1961_196125

theorem hot_dogs_remainder : 35867413 % 6 = 1 := by
  sorry

end NUMINAMATH_CALUDE_hot_dogs_remainder_l1961_196125


namespace NUMINAMATH_CALUDE_eggplant_basket_weight_l1961_196107

def cucumber_baskets : ℕ := 25
def eggplant_baskets : ℕ := 32
def total_weight : ℕ := 1870
def cucumber_basket_weight : ℕ := 30

theorem eggplant_basket_weight :
  (total_weight - cucumber_baskets * cucumber_basket_weight) / eggplant_baskets =
  (1870 - 25 * 30) / 32 := by
  sorry

end NUMINAMATH_CALUDE_eggplant_basket_weight_l1961_196107


namespace NUMINAMATH_CALUDE_b_current_age_l1961_196183

/-- Given two people A and B, proves B's current age is 39 years -/
theorem b_current_age (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →  -- A's age in 10 years equals twice B's age 10 years ago
  (a = b + 9) →              -- A is currently 9 years older than B
  b = 39 := by               -- B's current age is 39 years
sorry


end NUMINAMATH_CALUDE_b_current_age_l1961_196183


namespace NUMINAMATH_CALUDE_peanuts_in_box_l1961_196150

/-- Given a box with an initial number of peanuts and an additional number of peanuts added,
    calculate the total number of peanuts in the box. -/
def total_peanuts (initial : Nat) (added : Nat) : Nat :=
  initial + added

/-- Theorem stating that if there are initially 4 peanuts in a box and 8 more are added,
    the total number of peanuts in the box is 12. -/
theorem peanuts_in_box : total_peanuts 4 8 = 12 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l1961_196150


namespace NUMINAMATH_CALUDE_pens_given_to_sharon_l1961_196196

/-- The number of pens given to Sharon -/
def pens_to_sharon (initial : ℕ) (from_mike : ℕ) (final : ℕ) : ℕ :=
  2 * (initial + from_mike) - final

theorem pens_given_to_sharon :
  pens_to_sharon 5 20 40 = 10 := by
  sorry

end NUMINAMATH_CALUDE_pens_given_to_sharon_l1961_196196


namespace NUMINAMATH_CALUDE_water_added_third_hour_is_one_l1961_196128

/-- Calculates the amount of water added in the third hour -/
def water_added_third_hour (initial_water : ℝ) (loss_rate : ℝ) (fourth_hour_addition : ℝ) (final_water : ℝ) : ℝ :=
  final_water - (initial_water - 3 * loss_rate + fourth_hour_addition)

theorem water_added_third_hour_is_one :
  let initial_water : ℝ := 40
  let loss_rate : ℝ := 2
  let fourth_hour_addition : ℝ := 3
  let final_water : ℝ := 36
  water_added_third_hour initial_water loss_rate fourth_hour_addition final_water = 1 := by
  sorry

#eval water_added_third_hour 40 2 3 36

end NUMINAMATH_CALUDE_water_added_third_hour_is_one_l1961_196128


namespace NUMINAMATH_CALUDE_rationalize_denominator_l1961_196194

theorem rationalize_denominator : 
  ∃ (A B C D E F : ℤ), 
    F > 0 ∧
    (1 : ℝ) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 11) = 
      (A * Real.sqrt 3 + B * Real.sqrt 5 + C * Real.sqrt 11 + D * Real.sqrt E) / F ∧
    A = 13 ∧ B = 9 ∧ C = -3 ∧ D = -2 ∧ E = 165 ∧ F = 51 :=
by sorry

end NUMINAMATH_CALUDE_rationalize_denominator_l1961_196194


namespace NUMINAMATH_CALUDE_model_c_sample_size_l1961_196190

/-- Represents the total number of units produced -/
def total_population : ℕ := 1000

/-- Represents the number of units of model C -/
def model_c_population : ℕ := 300

/-- Represents the total sample size -/
def total_sample_size : ℕ := 60

/-- Calculates the number of units to be sampled from model C using stratified sampling -/
def stratified_sample_size (total_pop : ℕ) (model_pop : ℕ) (sample_size : ℕ) : ℕ :=
  (model_pop * sample_size) / total_pop

/-- Theorem stating that the stratified sample size for model C is 18 -/
theorem model_c_sample_size :
  stratified_sample_size total_population model_c_population total_sample_size = 18 := by
  sorry


end NUMINAMATH_CALUDE_model_c_sample_size_l1961_196190


namespace NUMINAMATH_CALUDE_geometric_sequence_term_number_l1961_196198

theorem geometric_sequence_term_number (a : ℕ → ℚ) :
  (∀ n : ℕ, a (n + 1) = a n * (1/2))  -- geometric sequence with q = 1/2
  → a 1 = 1/2                         -- a₁ = 1/2
  → (∃ n : ℕ, a n = 1/32)             -- aₙ = 1/32 for some n
  → (∃ n : ℕ, a n = 1/32 ∧ n = 5) :=  -- prove that this n is 5
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_term_number_l1961_196198


namespace NUMINAMATH_CALUDE_contingency_fund_amount_l1961_196114

def total_donation : ℚ := 240

def community_pantry_ratio : ℚ := 1/3
def crisis_fund_ratio : ℚ := 1/2
def livelihood_ratio : ℚ := 1/4

def community_pantry : ℚ := total_donation * community_pantry_ratio
def crisis_fund : ℚ := total_donation * crisis_fund_ratio

def remaining_after_main : ℚ := total_donation - community_pantry - crisis_fund
def livelihood_fund : ℚ := remaining_after_main * livelihood_ratio

def contingency_fund : ℚ := remaining_after_main - livelihood_fund

theorem contingency_fund_amount : contingency_fund = 30 := by
  sorry

end NUMINAMATH_CALUDE_contingency_fund_amount_l1961_196114


namespace NUMINAMATH_CALUDE_range_of_a_l1961_196133

/-- Two circles intersect at exactly two points if and only if 
the distance between their centers is greater than the absolute difference 
of their radii and less than the sum of their radii. -/
axiom circle_intersection_condition (r₁ r₂ d : ℝ) : 
  (∃! (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ 
    ((p₁.1 - d)^2 + p₁.2^2 = r₁^2) ∧ (p₁.1^2 + p₁.2^2 = r₂^2) ∧
    ((p₂.1 - d)^2 + p₂.2^2 = r₁^2) ∧ (p₂.1^2 + p₂.2^2 = r₂^2)) ↔ 
  (abs (r₁ - r₂) < d ∧ d < r₁ + r₂)

/-- The main theorem stating the range of a given the intersection condition. -/
theorem range_of_a : 
  ∀ a : ℝ, (∃! (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ 
    ((p₁.1 - a)^2 + (p₁.2 - a)^2 = 4) ∧ (p₁.1^2 + p₁.2^2 = 4) ∧
    ((p₂.1 - a)^2 + (p₂.2 - a)^2 = 4) ∧ (p₂.1^2 + p₂.2^2 = 4)) → 
  (-2 * Real.sqrt 2 < a ∧ a < 2 * Real.sqrt 2 ∧ a ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l1961_196133


namespace NUMINAMATH_CALUDE_marta_worked_19_hours_l1961_196111

/-- Calculates the number of hours worked given total money collected, hourly rate, and tips received. -/
def hoursWorked (totalCollected : ℚ) (hourlyRate : ℚ) (tips : ℚ) : ℚ :=
  (totalCollected - tips) / hourlyRate

/-- Proves that Marta has worked 19 hours on the farm. -/
theorem marta_worked_19_hours (totalCollected : ℚ) (hourlyRate : ℚ) (tips : ℚ)
    (h1 : totalCollected = 240)
    (h2 : hourlyRate = 10)
    (h3 : tips = 50) :
    hoursWorked totalCollected hourlyRate tips = 19 := by
  sorry

end NUMINAMATH_CALUDE_marta_worked_19_hours_l1961_196111


namespace NUMINAMATH_CALUDE_ice_melting_problem_l1961_196160

theorem ice_melting_problem (original_volume : ℝ) : 
  (original_volume > 0) →
  (original_volume * (1/4) * (1/4) = 0.2) → 
  (original_volume = 3.2) :=
by
  sorry

end NUMINAMATH_CALUDE_ice_melting_problem_l1961_196160


namespace NUMINAMATH_CALUDE_wheelbarrow_sale_ratio_l1961_196185

def duck_price : ℕ := 10
def chicken_price : ℕ := 8
def ducks_sold : ℕ := 2
def chickens_sold : ℕ := 5
def additional_earnings : ℕ := 60

def total_earnings : ℕ := duck_price * ducks_sold + chicken_price * chickens_sold

def wheelbarrow_cost : ℕ := total_earnings / 2

theorem wheelbarrow_sale_ratio :
  (wheelbarrow_cost + additional_earnings) / wheelbarrow_cost = 3 := by
  sorry

end NUMINAMATH_CALUDE_wheelbarrow_sale_ratio_l1961_196185


namespace NUMINAMATH_CALUDE_douglas_fir_price_l1961_196151

theorem douglas_fir_price (total_trees : ℕ) (douglas_fir_count : ℕ) (ponderosa_price : ℕ) (total_paid : ℕ) :
  total_trees = 850 →
  douglas_fir_count = 350 →
  ponderosa_price = 225 →
  total_paid = 217500 →
  ∃ (douglas_price : ℕ),
    douglas_price * douglas_fir_count + ponderosa_price * (total_trees - douglas_fir_count) = total_paid ∧
    douglas_price = 300 :=
by sorry

end NUMINAMATH_CALUDE_douglas_fir_price_l1961_196151


namespace NUMINAMATH_CALUDE_valid_lineup_count_l1961_196136

/- Define the total number of players -/
def total_players : ℕ := 18

/- Define the number of quadruplets -/
def quadruplets : ℕ := 4

/- Define the number of starters to select -/
def starters : ℕ := 8

/- Define the function to calculate combinations -/
def combination (n k : ℕ) : ℕ := (Nat.choose n k)

/- Theorem statement -/
theorem valid_lineup_count :
  combination total_players starters - combination (total_players - quadruplets) (starters - quadruplets) =
  42757 := by sorry

end NUMINAMATH_CALUDE_valid_lineup_count_l1961_196136


namespace NUMINAMATH_CALUDE_main_project_time_l1961_196180

def total_days : ℕ := 4
def hours_per_day : ℝ := 8
def time_on_smaller_tasks : ℝ := 9
def time_on_naps : ℝ := 13.5

theorem main_project_time :
  total_days * hours_per_day - time_on_smaller_tasks - time_on_naps = 9.5 := by
sorry

end NUMINAMATH_CALUDE_main_project_time_l1961_196180


namespace NUMINAMATH_CALUDE_min_value_quadratic_l1961_196197

theorem min_value_quadratic (a b : ℝ) : 
  let f := fun x : ℝ ↦ x^2 - a*x + b
  (∃ r₁ ∈ Set.Icc (-1) 1, f r₁ = 0) →
  (∃ r₂ ∈ Set.Icc 1 2, f r₂ = 0) →
  ∃ m : ℝ, m = -1 ∧ ∀ x : ℝ, a - 2*b ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_quadratic_l1961_196197


namespace NUMINAMATH_CALUDE_cyclic_sum_extrema_l1961_196130

def cyclic_sum (a : List ℕ) : ℕ :=
  (List.zip a (a.rotate 1)).map (fun (x, y) => x * y) |>.sum

def is_permutation (a : List ℕ) (n : ℕ) : Prop :=
  a.length = n ∧ a.toFinset = Finset.range n

def max_permutation (n : ℕ) : List ℕ :=
  (List.range ((n + 1) / 2)).map (fun i => 2 * i + 1) ++
  (List.range (n / 2)).reverse.map (fun i => 2 * (i + 1))

def min_permutation (n : ℕ) : List ℕ :=
  if n % 2 = 0 then
    (List.range (n / 2)).reverse.map (fun i => n - 2 * i) ++
    (List.range (n / 2)).map (fun i => 2 * i + 1)
  else
    (List.range ((n + 1) / 2)).reverse.map (fun i => n - 2 * i) ++
    (List.range (n / 2)).map (fun i => 2 * i + 2)

theorem cyclic_sum_extrema (n : ℕ) (a : List ℕ) (h : is_permutation a n) :
  cyclic_sum a ≤ cyclic_sum (max_permutation n) ∧
  cyclic_sum (min_permutation n) ≤ cyclic_sum a := by sorry

end NUMINAMATH_CALUDE_cyclic_sum_extrema_l1961_196130


namespace NUMINAMATH_CALUDE_window_treatment_cost_l1961_196100

/-- The number of windows that need treatment -/
def num_windows : ℕ := 3

/-- The cost of a pair of sheers in dollars -/
def sheer_cost : ℚ := 40

/-- The cost of a pair of drapes in dollars -/
def drape_cost : ℚ := 60

/-- The total cost of window treatments for all windows -/
def total_cost : ℚ := num_windows * (sheer_cost + drape_cost)

theorem window_treatment_cost : total_cost = 300 := by
  sorry

end NUMINAMATH_CALUDE_window_treatment_cost_l1961_196100


namespace NUMINAMATH_CALUDE_matrix_value_example_l1961_196117

def matrix_value (p q r s : ℤ) : ℤ := p * s - q * r

theorem matrix_value_example : matrix_value 4 5 2 3 = 2 := by sorry

end NUMINAMATH_CALUDE_matrix_value_example_l1961_196117


namespace NUMINAMATH_CALUDE_distinct_and_no_real_solutions_l1961_196131

theorem distinct_and_no_real_solutions : 
  ∀ b c : ℕ+, 
    (∃ x y : ℝ, x ≠ y ∧ x^2 + b*x + c = 0 ∧ y^2 + b*y + c = 0) ∧ 
    (∀ z : ℝ, z^2 + c*z + b ≠ 0) → 
    ((b = 3 ∧ c = 1) ∨ (b = 3 ∧ c = 2)) := by
  sorry

end NUMINAMATH_CALUDE_distinct_and_no_real_solutions_l1961_196131


namespace NUMINAMATH_CALUDE_triangle_pairs_lower_bound_l1961_196165

/-- Given n points in a plane and l line segments, this theorem proves a lower bound
    for the number of triangle pairs formed. -/
theorem triangle_pairs_lower_bound
  (n : ℕ) (l : ℕ) (h_n : n ≥ 4) (h_l : l ≥ n^2 / 4 + 1)
  (no_three_collinear : sorry) -- Hypothesis for no three points being collinear
  (T : ℕ) (h_T : T = sorry) -- Definition of T as the number of triangle pairs
  : T ≥ (l * (4 * l - n^2) * (4 * l - n^2 - n)) / (2 * n^2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_pairs_lower_bound_l1961_196165


namespace NUMINAMATH_CALUDE_lindas_bakery_profit_l1961_196109

/-- Calculate Linda's total profit for the day given her bread sales strategy -/
theorem lindas_bakery_profit :
  let total_loaves : ℕ := 60
  let morning_price : ℚ := 3
  let afternoon_price : ℚ := 3/2
  let evening_price : ℚ := 1
  let production_cost : ℚ := 1
  let morning_sales : ℕ := total_loaves / 3
  let afternoon_sales : ℕ := (total_loaves - morning_sales) / 2
  let evening_sales : ℕ := total_loaves - morning_sales - afternoon_sales
  let total_revenue : ℚ := morning_sales * morning_price + 
                           afternoon_sales * afternoon_price + 
                           evening_sales * evening_price
  let total_cost : ℚ := total_loaves * production_cost
  let profit : ℚ := total_revenue - total_cost
  profit = 50 := by
sorry

end NUMINAMATH_CALUDE_lindas_bakery_profit_l1961_196109


namespace NUMINAMATH_CALUDE_boys_in_class_l1961_196124

/-- Given a class with a 4:3 ratio of girls to boys and 49 total students,
    prove that the number of boys is 21. -/
theorem boys_in_class (girls boys : ℕ) : 
  4 * boys = 3 * girls →  -- ratio of girls to boys is 4:3
  girls + boys = 49 →     -- total number of students is 49
  boys = 21 :=            -- prove that the number of boys is 21
by sorry

end NUMINAMATH_CALUDE_boys_in_class_l1961_196124


namespace NUMINAMATH_CALUDE_second_replaced_man_age_is_35_l1961_196166

/-- The age of the second replaced man in a group replacement scenario -/
def second_replaced_man_age (initial_count : ℕ) (age_increase : ℕ) 
  (replaced_count : ℕ) (first_replaced_age : ℕ) (new_men_avg_age : ℕ) : ℕ :=
  47 - (initial_count * age_increase)

/-- Theorem stating the age of the second replaced man is 35 -/
theorem second_replaced_man_age_is_35 :
  second_replaced_man_age 12 1 2 21 34 = 35 := by
  sorry

end NUMINAMATH_CALUDE_second_replaced_man_age_is_35_l1961_196166


namespace NUMINAMATH_CALUDE_second_bag_kernels_l1961_196148

/-- Represents the number of kernels in a bag of popcorn -/
structure PopcornBag where
  total : ℕ
  popped : ℕ

/-- Calculates the percentage of popped kernels in a bag -/
def poppedPercentage (bag : PopcornBag) : ℚ :=
  (bag.popped : ℚ) / (bag.total : ℚ) * 100

theorem second_bag_kernels (bag1 bag2 bag3 : PopcornBag)
  (h1 : bag1.total = 75 ∧ bag1.popped = 60)
  (h2 : bag2.popped = 42)
  (h3 : bag3.total = 100 ∧ bag3.popped = 82)
  (h_avg : (poppedPercentage bag1 + poppedPercentage bag2 + poppedPercentage bag3) / 3 = 82) :
  bag2.total = 50 := by
  sorry


end NUMINAMATH_CALUDE_second_bag_kernels_l1961_196148


namespace NUMINAMATH_CALUDE_figure_division_l1961_196115

/-- A figure consisting of 24 cells can be divided into equal parts of specific sizes. -/
theorem figure_division (n : ℕ) : n ∣ 24 ∧ n ≠ 1 ↔ n ∈ ({2, 3, 4, 6, 8, 12, 24} : Finset ℕ) :=
sorry

end NUMINAMATH_CALUDE_figure_division_l1961_196115


namespace NUMINAMATH_CALUDE_preimage_of_three_l1961_196105

def f (x : ℝ) : ℝ := 2 * x - 1

theorem preimage_of_three (x : ℝ) : f x = 3 ↔ x = 2 := by sorry

end NUMINAMATH_CALUDE_preimage_of_three_l1961_196105


namespace NUMINAMATH_CALUDE_b_work_time_l1961_196121

-- Define the work completion time for A and the combined team
def a_time : ℝ := 6
def combined_time : ℝ := 3

-- Define the total payment and C's payment
def total_payment : ℝ := 5000
def c_payment : ℝ := 625.0000000000002

-- Define B's work completion time (to be proved)
def b_time : ℝ := 8

-- Theorem statement
theorem b_work_time : 
  (1 / a_time + 1 / b_time + c_payment / total_payment / combined_time = 1 / combined_time) → 
  b_time = 8 :=
by sorry

end NUMINAMATH_CALUDE_b_work_time_l1961_196121


namespace NUMINAMATH_CALUDE_andrew_final_stickers_l1961_196193

def total_stickers : ℕ := 1500
def ratio_sum : ℕ := 5

def initial_shares (i : Fin 3) : ℕ := 
  if i = 0 ∨ i = 1 then total_stickers / ratio_sum else 3 * (total_stickers / ratio_sum)

theorem andrew_final_stickers : 
  initial_shares 1 + (2/3 : ℚ) * initial_shares 2 = 900 := by sorry

end NUMINAMATH_CALUDE_andrew_final_stickers_l1961_196193


namespace NUMINAMATH_CALUDE_car_arrival_delay_l1961_196179

/-- Proves that a car traveling 225 km at 50 kmph instead of 60 kmph arrives 45 minutes later -/
theorem car_arrival_delay (distance : ℝ) (speed1 speed2 : ℝ) :
  distance = 225 →
  speed1 = 60 →
  speed2 = 50 →
  (distance / speed2 - distance / speed1) * 60 = 45 := by
sorry

end NUMINAMATH_CALUDE_car_arrival_delay_l1961_196179


namespace NUMINAMATH_CALUDE_obtain_a_to_six_l1961_196141

/-- Given a^4 and a^6 - 1, prove that a^6 can be obtained using +, -, and · operations -/
theorem obtain_a_to_six (a : ℝ) : ∃ f : ℝ → ℝ → ℝ → ℝ, 
  f (a^4) (a^6 - 1) 1 = a^6 ∧ 
  (∀ x y z, f x y z = x + y ∨ f x y z = x - y ∨ f x y z = x * y ∨ 
            f x y z = y + z ∨ f x y z = y - z ∨ f x y z = y * z ∨
            f x y z = z + x ∨ f x y z = z - x ∨ f x y z = z * x) :=
by
  sorry

end NUMINAMATH_CALUDE_obtain_a_to_six_l1961_196141


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1961_196116

-- Define the quadratic function
def f (x : ℝ) : ℝ := 4 * x^2 - x - 5

-- Define the solution set
def S : Set ℝ := { x | -1 ≤ x ∧ x ≤ 5/4 }

-- Theorem statement
theorem solution_set_of_inequality :
  { x : ℝ | f x ≤ 0 } = S :=
sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1961_196116


namespace NUMINAMATH_CALUDE_find_p_value_l1961_196158

-- Define the polynomial (x+y)^9
def polynomial (x y : ℝ) : ℝ := (x + y)^9

-- Define the second term of the expansion
def second_term (x y : ℝ) : ℝ := 9 * x^8 * y

-- Define the third term of the expansion
def third_term (x y : ℝ) : ℝ := 36 * x^7 * y^2

-- Theorem statement
theorem find_p_value (p q : ℝ) : 
  p > 0 ∧ q > 0 ∧ p + q = 1 ∧ second_term p q = third_term p q → p = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_find_p_value_l1961_196158


namespace NUMINAMATH_CALUDE_paige_recycled_amount_l1961_196192

/-- The number of pounds recycled per point earned -/
def pounds_per_point : ℕ := 4

/-- The number of pounds recycled by Paige's friends -/
def friends_recycled : ℕ := 2

/-- The total number of points earned -/
def total_points : ℕ := 4

/-- The number of pounds Paige recycled -/
def paige_recycled : ℕ := 14

theorem paige_recycled_amount :
  paige_recycled = total_points * pounds_per_point - friends_recycled := by
  sorry

end NUMINAMATH_CALUDE_paige_recycled_amount_l1961_196192


namespace NUMINAMATH_CALUDE_inequality_proof_l1961_196142

theorem inequality_proof (x y z : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : 0 ≤ z) (h4 : x^2 + y^2 + z^2 = 1) :
  1 ≤ (x / (1 + y*z)) + (y / (1 + z*x)) + (z / (1 + x*y)) ∧
  (x / (1 + y*z)) + (y / (1 + z*x)) + (z / (1 + x*y)) ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1961_196142


namespace NUMINAMATH_CALUDE_tournament_schools_l1961_196108

theorem tournament_schools (n : ℕ) : 
  (∀ (school : ℕ), school ≤ n → ∃ (team : Fin 4 → ℕ), 
    (∀ i j, i ≠ j → team i ≠ team j) ∧ 
    (∃ (theo leah mark nora : ℕ), 
      theo = (4 * n + 1) / 2 ∧
      leah = 48 ∧ 
      mark = 75 ∧ 
      nora = 97 ∧
      theo < leah ∧ theo < mark ∧ theo < nora ∧
      (∀ k, k ∈ [theo, leah, mark, nora] → k ≤ 4 * n) ∧
      (∀ k, k ∉ [theo, leah, mark, nora] → 
        (k < theo ∧ k ≤ 4 * n - 3) ∨ (k > theo ∧ k ≤ 4 * n)))) → 
  n = 36 := by
sorry

end NUMINAMATH_CALUDE_tournament_schools_l1961_196108


namespace NUMINAMATH_CALUDE_binomial_coefficient_divisibility_binomial_coefficient_extremes_l1961_196169

theorem binomial_coefficient_divisibility (p k : ℕ) (hp : Nat.Prime p) (hk : 0 < k ∧ k < p) :
  ∃ m : ℕ, (Nat.choose p k) = m * p :=
sorry

theorem binomial_coefficient_extremes (p : ℕ) (hp : Nat.Prime p) :
  Nat.choose p 0 = 1 ∧ Nat.choose p p = 1 :=
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_divisibility_binomial_coefficient_extremes_l1961_196169


namespace NUMINAMATH_CALUDE_solution_set_x_squared_leq_one_l1961_196173

theorem solution_set_x_squared_leq_one :
  ∀ x : ℝ, x^2 ≤ 1 ↔ -1 ≤ x ∧ x ≤ 1 := by sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_leq_one_l1961_196173


namespace NUMINAMATH_CALUDE_correct_articles_for_problem_l1961_196159

/-- Represents the possible articles that can be used before a noun -/
inductive Article
  | A
  | An
  | The
  | None

/-- Represents a noun with its properties -/
structure Noun where
  word : String
  startsWithSilentH : Bool
  isCountable : Bool

/-- Represents a fixed phrase -/
structure FixedPhrase where
  phrase : String
  meaning : String

/-- Function to determine the correct article for a noun -/
def correctArticle (n : Noun) : Article := sorry

/-- Function to determine the correct article for a fixed phrase -/
def correctPhraseArticle (fp : FixedPhrase) : Article := sorry

/-- Theorem stating the correct articles for the given problem -/
theorem correct_articles_for_problem 
  (hour : Noun)
  (out_of_question : FixedPhrase)
  (h1 : hour.word = "hour")
  (h2 : hour.startsWithSilentH = true)
  (h3 : hour.isCountable = true)
  (h4 : out_of_question.phrase = "out of __ question")
  (h5 : out_of_question.meaning = "impossible") :
  correctArticle hour = Article.An ∧ correctPhraseArticle out_of_question = Article.The := by
  sorry

end NUMINAMATH_CALUDE_correct_articles_for_problem_l1961_196159


namespace NUMINAMATH_CALUDE_polly_total_tweets_l1961_196188

/-- Represents an emotional state or activity of Polly the parakeet -/
structure State where
  name : String
  tweets_per_minute : ℕ
  duration : ℕ

/-- Calculates the total number of tweets for a given state -/
def tweets_for_state (s : State) : ℕ := s.tweets_per_minute * s.duration

/-- The list of Polly's states during the day -/
def polly_states : List State := [
  { name := "Happy", tweets_per_minute := 18, duration := 50 },
  { name := "Hungry", tweets_per_minute := 4, duration := 35 },
  { name := "Watching reflection", tweets_per_minute := 45, duration := 30 },
  { name := "Sad", tweets_per_minute := 6, duration := 20 },
  { name := "Playing with toys", tweets_per_minute := 25, duration := 75 }
]

/-- Calculates the total number of tweets for all states -/
def total_tweets (states : List State) : ℕ :=
  states.map tweets_for_state |>.sum

/-- Theorem: The total number of tweets Polly makes during the day is 4385 -/
theorem polly_total_tweets : total_tweets polly_states = 4385 := by
  sorry

end NUMINAMATH_CALUDE_polly_total_tweets_l1961_196188


namespace NUMINAMATH_CALUDE_symmetry_origin_symmetry_y_eq_x_vertex_2_neg_2_l1961_196175

-- Define the curve E
def E (x y : ℝ) : Prop := x^2 + x*y + y^2 = 4

-- Symmetry with respect to the origin
theorem symmetry_origin : ∀ x y : ℝ, E x y ↔ E (-x) (-y) := by sorry

-- Symmetry with respect to the line y = x
theorem symmetry_y_eq_x : ∀ x y : ℝ, E x y ↔ E y x := by sorry

-- (2, -2) is a vertex of E
theorem vertex_2_neg_2 : E 2 (-2) ∧ (∃ ε > 0, ∀ x y : ℝ, 
  (x - 2)^2 + (y + 2)^2 < ε^2 → E x y → x^2 + y^2 ≥ 2^2 + (-2)^2) := by sorry

end NUMINAMATH_CALUDE_symmetry_origin_symmetry_y_eq_x_vertex_2_neg_2_l1961_196175


namespace NUMINAMATH_CALUDE_line_intercepts_sum_l1961_196113

/-- Given a line with equation y + 3 = -2(x + 5), 
    the sum of its x-intercept and y-intercept is -39/2 -/
theorem line_intercepts_sum (x y : ℝ) : 
  (y + 3 = -2*(x + 5)) → 
  (∃ x_int y_int : ℝ, 
    (y_int + 3 = -2*(x_int + 5)) ∧ 
    (0 + 3 = -2*(x_int + 5)) ∧ 
    (y_int + 3 = -2*(0 + 5)) ∧ 
    (x_int + y_int = -39/2)) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_sum_l1961_196113


namespace NUMINAMATH_CALUDE_quadratic_real_roots_l1961_196102

/-- The quadratic equation x^2 - 6x + m = 0 has real roots if and only if m ≤ 9 -/
theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + m = 0) ↔ m ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_l1961_196102


namespace NUMINAMATH_CALUDE_bookcase_organization_l1961_196129

/-- The number of books on each shelf -/
def books_per_shelf : ℕ := 9

/-- The number of shelves containing mystery books -/
def mystery_shelves : ℕ := 3

/-- The number of shelves containing picture books -/
def picture_shelves : ℕ := 5

/-- The total number of books in the bookcase -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem bookcase_organization :
  total_books = 72 := by sorry

end NUMINAMATH_CALUDE_bookcase_organization_l1961_196129


namespace NUMINAMATH_CALUDE_monotone_increasing_implies_a_geq_5_l1961_196168

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2

-- State the theorem
theorem monotone_increasing_implies_a_geq_5 :
  ∀ a : ℝ, (∀ x y : ℝ, -5 ≤ x ∧ x < y ∧ y ≤ 5 → f a x < f a y) → a ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_monotone_increasing_implies_a_geq_5_l1961_196168


namespace NUMINAMATH_CALUDE_max_backyard_area_l1961_196172

/-- Represents a rectangular backyard with given constraints -/
structure Backyard where
  length : ℝ
  width : ℝ
  fencing : ℝ
  length_min : ℝ
  fence_constraint : fencing = length + 2 * width
  length_constraint : length ≥ length_min
  proportion_constraint : length ≤ 2 * width

/-- The area of a backyard -/
def area (b : Backyard) : ℝ := b.length * b.width

/-- Theorem stating the maximum area of a backyard with given constraints -/
theorem max_backyard_area (b : Backyard) (h1 : b.fencing = 400) (h2 : b.length_min = 100) :
  ∃ (max_area : ℝ), max_area = 20000 ∧ ∀ (other : Backyard), 
    other.fencing = 400 → other.length_min = 100 → area other ≤ max_area :=
sorry

end NUMINAMATH_CALUDE_max_backyard_area_l1961_196172


namespace NUMINAMATH_CALUDE_sin_plus_two_cos_alpha_l1961_196122

theorem sin_plus_two_cos_alpha (α : Real) :
  (∃ P : Real × Real, P.1 = 3/5 ∧ P.2 = 4/5 ∧ P.1^2 + P.2^2 = 1 ∧ 
   P.1 = Real.cos α ∧ P.2 = Real.sin α) →
  Real.sin α + 2 * Real.cos α = 2 := by
sorry

end NUMINAMATH_CALUDE_sin_plus_two_cos_alpha_l1961_196122


namespace NUMINAMATH_CALUDE_tower_arrangement_count_l1961_196145

def red_cubes : ℕ := 3
def blue_cubes : ℕ := 3
def green_cubes : ℕ := 4
def tower_height : ℕ := 8
def left_out_cubes : ℕ := 2

def total_cubes : ℕ := red_cubes + blue_cubes + green_cubes

def tower_arrangements : ℕ := 
  (tower_height.factorial / (red_cubes.factorial * blue_cubes.factorial * (green_cubes - 2).factorial)) +
  (tower_height.factorial / ((red_cubes - 1).factorial * (blue_cubes - 1).factorial * green_cubes.factorial))

theorem tower_arrangement_count : tower_arrangements = 980 := by
  sorry

end NUMINAMATH_CALUDE_tower_arrangement_count_l1961_196145


namespace NUMINAMATH_CALUDE_power_of_seven_mod_twelve_l1961_196152

theorem power_of_seven_mod_twelve : 7^135 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_twelve_l1961_196152


namespace NUMINAMATH_CALUDE_train_meeting_distance_l1961_196154

/-- Proves that given two trains starting 450 miles apart and traveling towards each other
    at 50 miles per hour each, train A will have traveled 225 miles when they meet. -/
theorem train_meeting_distance (distance_between_stations : ℝ) (speed_a : ℝ) (speed_b : ℝ)
  (h1 : distance_between_stations = 450)
  (h2 : speed_a = 50)
  (h3 : speed_b = 50) :
  speed_a * (distance_between_stations / (speed_a + speed_b)) = 225 := by
  sorry

end NUMINAMATH_CALUDE_train_meeting_distance_l1961_196154


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_seven_l1961_196178

theorem least_three_digit_multiple_of_seven : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 7 ∣ n → 105 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_seven_l1961_196178


namespace NUMINAMATH_CALUDE_f_behavior_l1961_196153

-- Define the function f(x) = 2x^3 - 7
def f (x : ℝ) : ℝ := 2 * x^3 - 7

-- State the theorem about the behavior of f(x) as x approaches infinity
theorem f_behavior :
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x > N → f x > M) ∧
  (∀ M : ℝ, ∃ N : ℝ, ∀ x : ℝ, x < N → f x < M) :=
sorry

end NUMINAMATH_CALUDE_f_behavior_l1961_196153


namespace NUMINAMATH_CALUDE_product_inequality_l1961_196146

theorem product_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : a + b + c + a * b * c = 4) : 
  (1 + a / b + c * a) * (1 + b / c + a * b) * (1 + c / a + b * c) ≥ 27 := by
  sorry

end NUMINAMATH_CALUDE_product_inequality_l1961_196146


namespace NUMINAMATH_CALUDE_cubic_root_inequality_l1961_196163

theorem cubic_root_inequality (R : ℚ) (h : R ≥ 0) : 
  let a : ℤ := 1
  let b : ℤ := 1
  let c : ℤ := 2
  let d : ℤ := 1
  let e : ℤ := 1
  let f : ℤ := 1
  |((a * R^2 + b * R + c) / (d * R^2 + e * R + f) : ℚ) - (2 : ℚ)^(1/3)| < |R - (2 : ℚ)^(1/3)| :=
by
  sorry

#check cubic_root_inequality

end NUMINAMATH_CALUDE_cubic_root_inequality_l1961_196163


namespace NUMINAMATH_CALUDE_worker_completion_time_l1961_196189

/-- Given that two workers a and b can complete a job together in 8 days,
    and worker a alone can complete the job in 12 days,
    prove that worker b alone can complete the job in 24 days. -/
theorem worker_completion_time (a b : ℝ) 
  (h1 : a + b = 1 / 8)  -- a and b together complete 1/8 of the work per day
  (h2 : a = 1 / 12)     -- a alone completes 1/12 of the work per day
  : b = 1 / 24 :=       -- b alone completes 1/24 of the work per day
by sorry

end NUMINAMATH_CALUDE_worker_completion_time_l1961_196189


namespace NUMINAMATH_CALUDE_simplify_expression_l1961_196119

theorem simplify_expression (a : ℝ) (h : 2 < a ∧ a < 3) :
  (2 - a)^(1/3) + (3 - a)^(1/4) = 5 - 2*a :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1961_196119


namespace NUMINAMATH_CALUDE_cuboid_surface_area_formula_l1961_196137

/-- The surface area of a cuboid with edges of length a, b, and c. -/
def cuboidSurfaceArea (a b c : ℝ) : ℝ := 2 * a * b + 2 * b * c + 2 * a * c

/-- Theorem: The surface area of a cuboid with edges of length a, b, and c
    is equal to 2ab + 2bc + 2ac. -/
theorem cuboid_surface_area_formula (a b c : ℝ) :
  cuboidSurfaceArea a b c = 2 * a * b + 2 * b * c + 2 * a * c := by
  sorry

end NUMINAMATH_CALUDE_cuboid_surface_area_formula_l1961_196137


namespace NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l1961_196164

-- Define the equations
def equation1 (x : ℝ) : Prop := 2 * (x - 1) = 2 - 5 * (x + 2)
def equation2 (x : ℝ) : Prop := (5 * x + 1) / 2 - (6 * x + 2) / 4 = 1

-- Theorem for the first equation
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = -6/7 := by sorry

-- Theorem for the second equation
theorem solution_equation2 : ∃ x : ℝ, equation2 x ∧ x = 1 := by sorry

end NUMINAMATH_CALUDE_solution_equation1_solution_equation2_l1961_196164


namespace NUMINAMATH_CALUDE_missing_data_point_l1961_196134

def linear_regression (x y : ℝ) := 0.28 * x + 0.16 = y

def data_points : List (ℝ × ℝ) := [(1, 0.5), (3, 1), (4, 1.4), (5, 1.5)]

theorem missing_data_point : 
  ∀ (a : ℝ), 
  (∀ (point : ℝ × ℝ), point ∈ data_points → linear_regression point.1 point.2) →
  linear_regression 2 a →
  linear_regression 3 ((0.5 + a + 1 + 1.4 + 1.5) / 5) →
  a = 0.6 := by sorry

end NUMINAMATH_CALUDE_missing_data_point_l1961_196134


namespace NUMINAMATH_CALUDE_triangle_area_l1961_196147

/-- The area of a triangle with vertices at (0,0), (0,5), and (7,12) is 17.5 square units. -/
theorem triangle_area : 
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (0, 5)
  let v3 : ℝ × ℝ := (7, 12)
  (1/2 : ℝ) * |v2.2 - v1.2| * |v3.1 - v1.1| = 17.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1961_196147


namespace NUMINAMATH_CALUDE_hannah_running_difference_l1961_196132

/-- Hannah's running distances for different days of the week -/
structure RunningDistances where
  monday : ℕ     -- Distance in kilometers
  wednesday : ℕ  -- Distance in meters
  friday : ℕ     -- Distance in meters

/-- Calculates the difference in meters between Monday's run and the combined Wednesday and Friday runs -/
def run_difference (distances : RunningDistances) : ℕ :=
  distances.monday * 1000 - (distances.wednesday + distances.friday)

/-- Theorem stating the difference in Hannah's running distances -/
theorem hannah_running_difference : 
  let distances : RunningDistances := { monday := 9, wednesday := 4816, friday := 2095 }
  run_difference distances = 2089 := by
  sorry

end NUMINAMATH_CALUDE_hannah_running_difference_l1961_196132


namespace NUMINAMATH_CALUDE_gas_station_candy_boxes_l1961_196138

theorem gas_station_candy_boxes : 
  let chocolate : Real := 3.5
  let sugar : Real := 5.25
  let gum : Real := 2.75
  let licorice : Real := 4.5
  let sour : Real := 7.125
  chocolate + sugar + gum + licorice + sour = 23.125 := by
  sorry

end NUMINAMATH_CALUDE_gas_station_candy_boxes_l1961_196138


namespace NUMINAMATH_CALUDE_grassy_plot_length_l1961_196140

/-- Represents the dimensions and cost of a rectangular grassy plot with a gravel path. -/
structure GrassyPlot where
  width : ℝ  -- Width of the grassy plot in meters
  pathWidth : ℝ  -- Width of the gravel path in meters
  gravelCost : ℝ  -- Cost of gravelling in rupees
  gravelRate : ℝ  -- Cost of gravelling per square meter in rupees

/-- Calculates the length of the grassy plot given its specifications. -/
def calculateLength (plot : GrassyPlot) : ℝ :=
  -- Implementation not provided as per instructions
  sorry

/-- Theorem stating that given the specified conditions, the length of the grassy plot is 100 meters. -/
theorem grassy_plot_length 
  (plot : GrassyPlot) 
  (h1 : plot.width = 65) 
  (h2 : plot.pathWidth = 2.5) 
  (h3 : plot.gravelCost = 425) 
  (h4 : plot.gravelRate = 0.5) : 
  calculateLength plot = 100 := by
  sorry

end NUMINAMATH_CALUDE_grassy_plot_length_l1961_196140


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l1961_196162

-- Define the motion equation
def s (t : ℝ) : ℝ := 1 - t + t^2

-- Define the instantaneous velocity (derivative of s)
def v (t : ℝ) : ℝ := -1 + 2*t

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 5 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l1961_196162


namespace NUMINAMATH_CALUDE_no_solution_for_equation_l1961_196112

theorem no_solution_for_equation : ¬∃ (x : ℝ), x - 7 / (x - 3) = 3 - 7 / (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_equation_l1961_196112


namespace NUMINAMATH_CALUDE_centroid_circle_area_l1961_196135

/-- Given a circle with diameter 'd', the area of the circle traced by the centroid of a triangle
    formed by the diameter and a point on the circumference is (25/900) times the area of the original circle. -/
theorem centroid_circle_area (d : ℝ) (h : d > 0) :
  ∃ (A_centroid A_circle : ℝ),
    A_circle = π * (d/2)^2 ∧
    A_centroid = π * (d/6)^2 ∧
    A_centroid = (25/900) * A_circle :=
by sorry

end NUMINAMATH_CALUDE_centroid_circle_area_l1961_196135


namespace NUMINAMATH_CALUDE_equation_solution_l1961_196104

theorem equation_solution (x : ℝ) : 
  (1 - 2 * Real.sin (x / 2) * Real.cos (x / 2) = 
   (Real.sin (x / 2) - Real.cos (x / 2)) / Real.cos (x / 2)) ↔ 
  (∃ k : ℤ, x = π / 2 + 2 * k * π) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l1961_196104


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_l1961_196106

def first_six_odd_numbers : List ℕ := [1, 3, 5, 7, 9, 11]

def rectangle_areas (width : ℕ) (lengths : List ℕ) : List ℕ :=
  lengths.map (λ l => width * l)

theorem sum_of_rectangle_areas :
  let width := 2
  let lengths := first_six_odd_numbers.map (λ n => n * n)
  let areas := rectangle_areas width lengths
  areas.sum = 572 := by sorry

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_l1961_196106


namespace NUMINAMATH_CALUDE_floor_of_negative_two_point_seven_l1961_196184

theorem floor_of_negative_two_point_seven : ⌊(-2.7 : ℝ)⌋ = -3 := by
  sorry

end NUMINAMATH_CALUDE_floor_of_negative_two_point_seven_l1961_196184


namespace NUMINAMATH_CALUDE_replacement_solution_percentage_l1961_196155

theorem replacement_solution_percentage
  (original_percentage : ℝ)
  (replaced_portion : ℝ)
  (final_percentage : ℝ)
  (h1 : original_percentage = 85)
  (h2 : replaced_portion = 0.6923076923076923)
  (h3 : final_percentage = 40)
  (x : ℝ) :
  (original_percentage * (1 - replaced_portion) + x * replaced_portion = final_percentage) →
  x = 20 := by
  sorry

end NUMINAMATH_CALUDE_replacement_solution_percentage_l1961_196155


namespace NUMINAMATH_CALUDE_remainder_theorem_l1961_196123

theorem remainder_theorem (T E N S E' N' S' : ℤ)
  (h1 : T = N * E + S)
  (h2 : N = N' * E' + S')
  : T % (E + E') = E * S' + S := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1961_196123


namespace NUMINAMATH_CALUDE_smallest_n_for_fraction_l1961_196191

def fraction (n : ℕ) : ℚ :=
  (5^(n+1) + 2^(n+1)) / (5^n + 2^n)

theorem smallest_n_for_fraction :
  (∀ k : ℕ, k < 7 → fraction k ≤ 4.99) ∧
  fraction 7 > 4.99 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_fraction_l1961_196191


namespace NUMINAMATH_CALUDE_roots_sum_bound_l1961_196149

theorem roots_sum_bound (z x : ℂ) : 
  z ≠ x → 
  z^2017 = 1 → 
  x^2017 = 1 → 
  Complex.abs (z + x) < Real.sqrt (2 + Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_roots_sum_bound_l1961_196149


namespace NUMINAMATH_CALUDE_bob_bought_four_candies_l1961_196110

/-- The number of candies bought by each person -/
structure CandyPurchase where
  emily : ℕ
  jennifer : ℕ
  bob : ℕ

/-- The conditions of the candy purchase scenario -/
def candy_scenario (p : CandyPurchase) : Prop :=
  p.emily = 6 ∧
  p.jennifer = 2 * p.emily ∧
  p.jennifer = 3 * p.bob

/-- Theorem stating that Bob bought 4 candies -/
theorem bob_bought_four_candies :
  ∀ p : CandyPurchase, candy_scenario p → p.bob = 4 := by
  sorry

end NUMINAMATH_CALUDE_bob_bought_four_candies_l1961_196110


namespace NUMINAMATH_CALUDE_friends_with_oranges_l1961_196181

theorem friends_with_oranges (total_friends : ℕ) (friends_with_pears : ℕ) : 
  total_friends = 15 → friends_with_pears = 9 → total_friends - friends_with_pears = 6 := by
  sorry

end NUMINAMATH_CALUDE_friends_with_oranges_l1961_196181


namespace NUMINAMATH_CALUDE_bus_performance_analysis_l1961_196199

structure BusCompany where
  name : String
  onTime : ℕ
  notOnTime : ℕ

def totalBuses (company : BusCompany) : ℕ := company.onTime + company.notOnTime

def onTimeProbability (company : BusCompany) : ℚ :=
  company.onTime / totalBuses company

def kSquared (companyA companyB : BusCompany) : ℚ :=
  let n := totalBuses companyA + totalBuses companyB
  let a := companyA.onTime
  let b := companyA.notOnTime
  let c := companyB.onTime
  let d := companyB.notOnTime
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

def companyA : BusCompany := ⟨"A", 240, 20⟩
def companyB : BusCompany := ⟨"B", 210, 30⟩

theorem bus_performance_analysis :
  (onTimeProbability companyA = 12/13) ∧ 
  (onTimeProbability companyB = 7/8) ∧ 
  (kSquared companyA companyB > 2706/1000) := by
  sorry

end NUMINAMATH_CALUDE_bus_performance_analysis_l1961_196199


namespace NUMINAMATH_CALUDE_cubic_equation_implications_l1961_196120

theorem cubic_equation_implications (x y z : ℝ) 
  (h_pos : x > 0 ∧ y > 0 ∧ z > 0) 
  (h_not_equal : ¬(x = y ∧ y = z))
  (h_equation : x^3 + y^3 + z^3 - 3*x*y*z - 3*(x^2 + y^2 + z^2 - x*y - y*z - z*x) = 0) :
  (x + y + z = 3) ∧ 
  (x^2*(1+y) + y^2*(1+z) + z^2*(1+x) > 6) := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_implications_l1961_196120


namespace NUMINAMATH_CALUDE_total_children_l1961_196187

theorem total_children (happy sad neutral boys girls happy_boys sad_girls : ℕ) :
  happy = 30 →
  sad = 10 →
  neutral = 20 →
  boys = 18 →
  girls = 42 →
  happy_boys = 6 →
  sad_girls = 4 →
  happy + sad + neutral = boys + girls :=
by sorry

end NUMINAMATH_CALUDE_total_children_l1961_196187


namespace NUMINAMATH_CALUDE_square_area_on_parabola_and_line_l1961_196157

theorem square_area_on_parabola_and_line : ∃ (a : ℝ), a > 0 ∧ 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    (x₁^2 + 2*x₁ + 1 = 8) ∧ 
    (x₂^2 + 2*x₂ + 1 = 8) ∧ 
    a = (x₂ - x₁)^2) ∧ 
  a = 36 := by
sorry

end NUMINAMATH_CALUDE_square_area_on_parabola_and_line_l1961_196157
