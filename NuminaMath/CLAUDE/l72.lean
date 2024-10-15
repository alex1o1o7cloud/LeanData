import Mathlib

namespace NUMINAMATH_CALUDE_expected_votes_for_candidate_a_l72_7228

theorem expected_votes_for_candidate_a (total_voters : ℕ) 
  (democrat_percentage : ℚ) (republican_percentage : ℚ)
  (democrat_support : ℚ) (republican_support : ℚ) :
  democrat_percentage + republican_percentage = 1 →
  democrat_percentage = 3/5 →
  democrat_support = 3/4 →
  republican_support = 1/5 →
  (democrat_percentage * democrat_support + 
   republican_percentage * republican_support) * total_voters = 
  (53/100) * total_voters :=
by sorry

end NUMINAMATH_CALUDE_expected_votes_for_candidate_a_l72_7228


namespace NUMINAMATH_CALUDE_cube_sum_equals_sum_l72_7200

theorem cube_sum_equals_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : a / (1 + b) + b / (1 + a) = 1) : a^3 + b^3 = a + b := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equals_sum_l72_7200


namespace NUMINAMATH_CALUDE_chinese_character_sum_l72_7224

theorem chinese_character_sum : ∃! (a b c d e f g : ℕ),
  (a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧ g < 10) ∧
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧
   d ≠ e ∧ d ≠ f ∧ d ≠ g ∧
   e ≠ f ∧ e ≠ g ∧
   f ≠ g) ∧
  (1000 * a + 100 * b + 10 * c + d + 100 * e + 10 * f + g = 2013) ∧
  (a + b + c + d + e + f + g = 24) :=
by sorry

end NUMINAMATH_CALUDE_chinese_character_sum_l72_7224


namespace NUMINAMATH_CALUDE_rectangular_to_cubic_block_l72_7247

/-- The edge length of a cube with the same volume as a rectangular block -/
def cube_edge_length (l w h : ℝ) : ℝ :=
  (l * w * h) ^ (1/3)

/-- Theorem stating that a 50cm x 8cm x 20cm rectangular block forged into a cube has an edge length of 20cm -/
theorem rectangular_to_cubic_block :
  cube_edge_length 50 8 20 = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_to_cubic_block_l72_7247


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l72_7208

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a_1 + a_7 = 10, then a_3 + a_5 = 10 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : arithmetic_sequence a) (h1 : a 1 + a 7 = 10) :
  a 3 + a 5 = 10 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l72_7208


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l72_7238

theorem sqrt_equation_solution (x : ℝ) :
  Real.sqrt (1 - 3 * x) = 7 → x = -16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l72_7238


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l72_7259

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop :=
  (y - 1)^2 / 16 - (x + 2)^2 / 9 = 1

/-- The asymptote equation -/
def asymptote_eq (x y m : ℝ) : Prop :=
  y - 1 = m * x + 2 * m ∨ y - 1 = -m * x - 2 * m

/-- Theorem stating that the value of m for the given hyperbola is 4/3 -/
theorem hyperbola_asymptote_slope :
  ∃ m : ℝ, (∀ x y : ℝ, hyperbola_eq x y → asymptote_eq x y m) ∧ m = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l72_7259


namespace NUMINAMATH_CALUDE_eighteenth_is_sunday_l72_7203

/-- Represents days of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a date in a month -/
structure Date where
  day : Nat
  dayOfWeek : DayOfWeek

/-- A month with three Fridays on even dates -/
structure SpecialMonth where
  dates : List Date
  three_even_fridays : ∃ (d1 d2 d3 : Date),
    d1 ∈ dates ∧ d2 ∈ dates ∧ d3 ∈ dates ∧
    d1.day ≠ d2.day ∧ d1.day ≠ d3.day ∧ d2.day ≠ d3.day ∧
    d1.dayOfWeek = DayOfWeek.Friday ∧
    d2.dayOfWeek = DayOfWeek.Friday ∧
    d3.dayOfWeek = DayOfWeek.Friday ∧
    d1.day % 2 = 0 ∧ d2.day % 2 = 0 ∧ d3.day % 2 = 0

/-- The 18th of a special month is a Sunday -/
theorem eighteenth_is_sunday (m : SpecialMonth) :
  ∃ (d : Date), d ∈ m.dates ∧ d.day = 18 ∧ d.dayOfWeek = DayOfWeek.Sunday :=
sorry

end NUMINAMATH_CALUDE_eighteenth_is_sunday_l72_7203


namespace NUMINAMATH_CALUDE_board_zeros_l72_7244

theorem board_zeros (n : ℕ) (pos neg zero : ℕ) : 
  n = 10 → 
  pos + neg + zero = n → 
  pos * neg = 15 → 
  zero = 2 := by sorry

end NUMINAMATH_CALUDE_board_zeros_l72_7244


namespace NUMINAMATH_CALUDE_expenditure_increase_l72_7278

theorem expenditure_increase (income : ℝ) (expenditure : ℝ) (savings : ℝ) 
  (new_income : ℝ) (new_expenditure : ℝ) (new_savings : ℝ) :
  expenditure = 0.75 * income →
  savings = income - expenditure →
  new_income = 1.2 * income →
  new_savings = 1.5 * savings →
  new_savings = new_income - new_expenditure →
  new_expenditure = 1.1 * expenditure :=
by sorry

end NUMINAMATH_CALUDE_expenditure_increase_l72_7278


namespace NUMINAMATH_CALUDE_town_trash_cans_l72_7243

theorem town_trash_cans (street_cans : ℕ) (store_cans : ℕ) : 
  street_cans = 14 →
  store_cans = 2 * street_cans →
  street_cans + store_cans = 42 := by
sorry

end NUMINAMATH_CALUDE_town_trash_cans_l72_7243


namespace NUMINAMATH_CALUDE_union_complement_eq_specific_set_l72_7239

open Set

def U : Finset ℕ := {0, 1, 2, 4, 6, 8}
def M : Finset ℕ := {0, 4, 6}
def N : Finset ℕ := {0, 1, 6}

theorem union_complement_eq_specific_set :
  M ∪ (U \ N) = {0, 2, 4, 6, 8} :=
sorry

end NUMINAMATH_CALUDE_union_complement_eq_specific_set_l72_7239


namespace NUMINAMATH_CALUDE_impossibleTransformation_l72_7257

/-- Represents a pile of stones -/
structure Pile :=
  (count : ℕ)

/-- Represents the state of all piles -/
structure PileState :=
  (piles : List Pile)

/-- Allowed operations on piles -/
inductive Operation
  | Combine : Pile → Pile → Operation
  | Split : Pile → Operation

/-- Applies an operation to a pile state -/
def applyOperation (state : PileState) (op : Operation) : PileState :=
  sorry

/-- Checks if a pile state is the desired final state -/
def isFinalState (state : PileState) : Prop :=
  state.piles.length = 105 ∧ state.piles.all (fun p => p.count = 1)

/-- The main theorem -/
theorem impossibleTransformation :
  ∀ (operations : List Operation),
    let initialState : PileState := ⟨[⟨51⟩, ⟨49⟩, ⟨5⟩]⟩
    let finalState := operations.foldl applyOperation initialState
    ¬(isFinalState finalState) := by
  sorry

end NUMINAMATH_CALUDE_impossibleTransformation_l72_7257


namespace NUMINAMATH_CALUDE_f_2011_equals_6_l72_7261

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)

theorem f_2011_equals_6 (f : ℝ → ℝ) 
    (h_even : is_even_function f)
    (h_sym : symmetric_about f 2)
    (h_sum : f 2011 + 2 * f 1 = 18) :
  f 2011 = 6 := by
sorry

end NUMINAMATH_CALUDE_f_2011_equals_6_l72_7261


namespace NUMINAMATH_CALUDE_cost_price_percentage_l72_7213

theorem cost_price_percentage (marked_price cost_price selling_price : ℝ) :
  selling_price = 0.88 * marked_price →
  selling_price = 1.375 * cost_price →
  cost_price / marked_price = 0.64 := by
  sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l72_7213


namespace NUMINAMATH_CALUDE_hcl_moles_formed_l72_7211

/-- Represents the chemical reaction NH4Cl + H2O → NH4OH + HCl -/
structure ChemicalReaction where
  nh4cl_mass : ℝ
  h2o_moles : ℝ
  nh4oh_moles : ℝ
  hcl_moles : ℝ

/-- The molar mass of NH4Cl in g/mol -/
def nh4cl_molar_mass : ℝ := 53.49

/-- Theorem stating that in the given reaction, 1 mole of HCl is formed -/
theorem hcl_moles_formed (reaction : ChemicalReaction) 
  (h1 : reaction.nh4cl_mass = 53)
  (h2 : reaction.h2o_moles = 1)
  (h3 : reaction.nh4oh_moles = 1) :
  reaction.hcl_moles = 1 := by
sorry

end NUMINAMATH_CALUDE_hcl_moles_formed_l72_7211


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l72_7293

theorem geometric_sequence_minimum (a : ℕ → ℝ) (m n : ℕ) :
  (∀ k, a k > 0) →  -- positive sequence
  (∃ r > 0, ∀ k, a (k + 1) = r * a k) →  -- geometric sequence
  a 9 = 9 * a 7 →  -- given condition
  a m * a n = 9 * (a 1)^2 →  -- given condition
  (∀ i j : ℕ, (a i * a j = 9 * (a 1)^2) → 1/i + 9/j ≥ 1/m + 9/n) →  -- minimum condition
  1/m + 9/n = 4 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l72_7293


namespace NUMINAMATH_CALUDE_max_value_theorem_l72_7212

-- Define the constraint function
def constraint (x y : ℝ) : Prop := x^2 + y^2 = 2

-- Define the objective function
def objective (x y : ℝ) : ℝ := 3*x + 4*y

-- Theorem statement
theorem max_value_theorem :
  ∃ (max : ℝ), max = 5 * Real.sqrt 2 ∧
  (∀ x y : ℝ, constraint x y → objective x y ≤ max) ∧
  (∃ x y : ℝ, constraint x y ∧ objective x y = max) :=
sorry

end NUMINAMATH_CALUDE_max_value_theorem_l72_7212


namespace NUMINAMATH_CALUDE_vacation_miles_theorem_l72_7227

/-- Calculates the total miles driven during a vacation -/
def total_miles_driven (days : ℕ) (miles_per_day : ℕ) : ℕ :=
  days * miles_per_day

/-- Proves that a 5-day vacation driving 250 miles per day results in 1250 total miles -/
theorem vacation_miles_theorem : 
  total_miles_driven 5 250 = 1250 := by
  sorry

end NUMINAMATH_CALUDE_vacation_miles_theorem_l72_7227


namespace NUMINAMATH_CALUDE_slope_intercept_sum_l72_7282

/-- Given points X, Y, Z, and G as the midpoint of XY, prove that the sum of the slope
    and y-intercept of the line passing through Z and G is 18/5 -/
theorem slope_intercept_sum (X Y Z G : ℝ × ℝ) : 
  X = (0, 8) → Y = (0, 0) → Z = (10, 0) → 
  G = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2) →
  let m := (G.2 - Z.2) / (G.1 - Z.1)
  let b := G.2
  m + b = 18 / 5 := by
sorry

end NUMINAMATH_CALUDE_slope_intercept_sum_l72_7282


namespace NUMINAMATH_CALUDE_lcm_gcd_product_l72_7285

theorem lcm_gcd_product (a b : ℕ) (ha : a = 12) (hb : b = 15) :
  Nat.lcm a b * Nat.gcd a b = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcd_product_l72_7285


namespace NUMINAMATH_CALUDE_pool_problem_l72_7217

/-- Given a pool with humans and dogs, calculate the number of dogs -/
def number_of_dogs (total_legs_paws : ℕ) (num_humans : ℕ) (human_legs : ℕ) (dog_paws : ℕ) : ℕ :=
  ((total_legs_paws - (num_humans * human_legs)) / dog_paws)

theorem pool_problem :
  let total_legs_paws : ℕ := 24
  let num_humans : ℕ := 2
  let human_legs : ℕ := 2
  let dog_paws : ℕ := 4
  number_of_dogs total_legs_paws num_humans human_legs dog_paws = 5 := by
  sorry

end NUMINAMATH_CALUDE_pool_problem_l72_7217


namespace NUMINAMATH_CALUDE_time_expression_l72_7264

/-- Given V = 3gt + V₀ and S = (3/2)gt² + V₀t + (1/2)at², where a is another constant acceleration,
    prove that t = 9gS / (2(V - V₀)² + 3V₀(V - V₀)) -/
theorem time_expression (V V₀ g a S t : ℝ) 
  (hV : V = 3 * g * t + V₀)
  (hS : S = (3/2) * g * t^2 + V₀ * t + (1/2) * a * t^2) :
  t = (9 * g * S) / (2 * (V - V₀)^2 + 3 * V₀ * (V - V₀)) :=
by sorry

end NUMINAMATH_CALUDE_time_expression_l72_7264


namespace NUMINAMATH_CALUDE_product_of_roots_l72_7297

theorem product_of_roots (x : ℝ) : 
  (24 * x^2 - 72 * x + 200 = 0) → 
  (∃ r₁ r₂ : ℝ, (24 * r₁^2 - 72 * r₁ + 200 = 0) ∧ 
                (24 * r₂^2 - 72 * r₂ + 200 = 0) ∧ 
                (r₁ * r₂ = 25 / 3)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l72_7297


namespace NUMINAMATH_CALUDE_complement_of_120_degrees_l72_7284

-- Define the angle in degrees
def given_angle : ℝ := 120

-- Define the complement of an angle
def complement (angle : ℝ) : ℝ := 180 - angle

-- Theorem statement
theorem complement_of_120_degrees :
  complement given_angle = 60 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_120_degrees_l72_7284


namespace NUMINAMATH_CALUDE_wall_bricks_count_l72_7241

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 720

/-- Time taken by the first bricklayer to complete the wall alone (in hours) -/
def time_worker1 : ℕ := 12

/-- Time taken by the second bricklayer to complete the wall alone (in hours) -/
def time_worker2 : ℕ := 15

/-- Productivity decrease when working together (in bricks per hour) -/
def productivity_decrease : ℕ := 12

/-- Time taken when both workers work together (in hours) -/
def time_together : ℕ := 6

/-- Theorem stating that the number of bricks in the wall is 720 -/
theorem wall_bricks_count :
  (total_bricks / time_worker1 + total_bricks / time_worker2 - productivity_decrease) * time_together = total_bricks := by
  sorry

end NUMINAMATH_CALUDE_wall_bricks_count_l72_7241


namespace NUMINAMATH_CALUDE_percentage_sum_problem_l72_7249

theorem percentage_sum_problem : (0.2 * 40) + (0.25 * 60) = 23 := by
  sorry

end NUMINAMATH_CALUDE_percentage_sum_problem_l72_7249


namespace NUMINAMATH_CALUDE_probability_mame_on_top_l72_7288

/-- Represents a section of a folded paper -/
structure PaperSection :=
  (side : Fin 2)
  (quadrant : Fin 4)

/-- The total number of sections on a paper folded in quarters -/
def total_sections : ℕ := 8

/-- The probability of a specific section being on top when randomly refolded -/
def probability_on_top : ℚ := 1 / total_sections

theorem probability_mame_on_top :
  probability_on_top = 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_probability_mame_on_top_l72_7288


namespace NUMINAMATH_CALUDE_altitudes_sum_lt_perimeter_l72_7276

/-- A triangle with side lengths and altitudes -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : ℝ
  hb : ℝ
  hc : ℝ
  side_positive : 0 < a ∧ 0 < b ∧ 0 < c
  altitude_positive : 0 < ha ∧ 0 < hb ∧ 0 < hc
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The sum of altitudes is less than the perimeter in any triangle -/
theorem altitudes_sum_lt_perimeter (t : Triangle) : t.ha + t.hb + t.hc < t.a + t.b + t.c := by
  sorry

end NUMINAMATH_CALUDE_altitudes_sum_lt_perimeter_l72_7276


namespace NUMINAMATH_CALUDE_orange_pricing_theorem_l72_7216

/-- Represents the price in cents for a pack of oranges -/
structure PackPrice :=
  (quantity : ℕ)
  (price : ℕ)

/-- Calculates the total cost for a given number of packs -/
def totalCost (pack : PackPrice) (numPacks : ℕ) : ℕ :=
  pack.price * numPacks

/-- Calculates the total number of oranges for a given number of packs -/
def totalOranges (pack : PackPrice) (numPacks : ℕ) : ℕ :=
  pack.quantity * numPacks

theorem orange_pricing_theorem (pack1 pack2 : PackPrice) 
    (h1 : pack1 = ⟨4, 15⟩)
    (h2 : pack2 = ⟨6, 25⟩)
    (h3 : totalOranges pack1 5 + totalOranges pack2 5 = 20) :
  (totalCost pack1 5 + totalCost pack2 5) / 20 = 10 := by
  sorry

end NUMINAMATH_CALUDE_orange_pricing_theorem_l72_7216


namespace NUMINAMATH_CALUDE_cylinder_volume_relation_l72_7233

/-- Given two cylinders C and D with the specified properties, 
    prove that the volume of D is 9πh³ --/
theorem cylinder_volume_relation (h r : ℝ) : 
  h > 0 → r > 0 → 
  (π * h^2 * r) * 3 = π * r^2 * h → 
  π * r^2 * h = 9 * π * h^3 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_relation_l72_7233


namespace NUMINAMATH_CALUDE_log_problem_l72_7214

-- Define the logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_problem :
  4 * lg 2 + 3 * lg 5 - lg (1/5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l72_7214


namespace NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l72_7290

theorem smallest_sum_of_perfect_squares (x y : ℕ) : 
  x^2 - y^2 = 143 → ∀ a b : ℕ, a^2 - b^2 = 143 → x^2 + y^2 ≤ a^2 + b^2 → x^2 + y^2 = 145 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_perfect_squares_l72_7290


namespace NUMINAMATH_CALUDE_product_of_integers_l72_7291

theorem product_of_integers (p q r : ℤ) : 
  p ≠ 0 → q ≠ 0 → r ≠ 0 →
  p + q + r = 30 →
  1 / p + 1 / q + 1 / r + 450 / (p * q * r) = 1 →
  p * q * r = 1920 := by
sorry

end NUMINAMATH_CALUDE_product_of_integers_l72_7291


namespace NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficient_l72_7207

theorem quadratic_rational_root_even_coefficient 
  (a b c : ℤ) (hα : a ≠ 0) : 
  (∃ (x : ℚ), a * x^2 + b * x + c = 0) → 
  (Even a ∨ Even b ∨ Even c) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rational_root_even_coefficient_l72_7207


namespace NUMINAMATH_CALUDE_parallel_line_m_value_l72_7263

/-- Given two points A(-3,m) and B(m,5), and a line parallel to 3x+y-1=0, prove m = -7 -/
theorem parallel_line_m_value :
  ∀ m : ℝ,
  let A : ℝ × ℝ := (-3, m)
  let B : ℝ × ℝ := (m, 5)
  let parallel_line_slope : ℝ := -3
  (B.2 - A.2) / (B.1 - A.1) = parallel_line_slope →
  m = -7 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_line_m_value_l72_7263


namespace NUMINAMATH_CALUDE_odd_function_negative_domain_l72_7251

-- Define the function f for x > 0
def f_pos (x : ℝ) : ℝ := x^2 - x - 1

-- Define the property of an odd function
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- State the theorem
theorem odd_function_negative_domain 
  (f : ℝ → ℝ) 
  (h_odd : is_odd_function f) 
  (h_pos : ∀ x > 0, f x = f_pos x) : 
  ∀ x < 0, f x = -x^2 - x + 1 := by
sorry

end NUMINAMATH_CALUDE_odd_function_negative_domain_l72_7251


namespace NUMINAMATH_CALUDE_meeting_point_distance_l72_7272

/-- A problem about two people meeting on a road --/
theorem meeting_point_distance
  (total_distance : ℝ)
  (distance_B_to_C : ℝ)
  (h1 : total_distance = 1000)
  (h2 : distance_B_to_C = 400) :
  total_distance - distance_B_to_C = 600 :=
by sorry

end NUMINAMATH_CALUDE_meeting_point_distance_l72_7272


namespace NUMINAMATH_CALUDE_oatmeal_cookie_baggies_l72_7222

theorem oatmeal_cookie_baggies 
  (total_cookies : ℝ) 
  (chocolate_chip_cookies : ℝ) 
  (cookies_per_bag : ℝ) 
  (h1 : total_cookies = 41.0) 
  (h2 : chocolate_chip_cookies = 13.0) 
  (h3 : cookies_per_bag = 9.0) :
  ⌊(total_cookies - chocolate_chip_cookies) / cookies_per_bag⌋ = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_oatmeal_cookie_baggies_l72_7222


namespace NUMINAMATH_CALUDE_problem_statement_l72_7246

theorem problem_statement (x y : ℤ) (h1 : x = 7) (h2 : y = x + 5) :
  (x - y) * (x + y) = -95 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l72_7246


namespace NUMINAMATH_CALUDE_train_station_distance_l72_7275

theorem train_station_distance (speed1 speed2 : ℝ) (time_diff : ℝ) : 
  speed1 = 4 →
  speed2 = 5 →
  time_diff = 12 / 60 →
  (∃ d : ℝ, d / speed1 - d / speed2 = time_diff ∧ d = 4) :=
by sorry

end NUMINAMATH_CALUDE_train_station_distance_l72_7275


namespace NUMINAMATH_CALUDE_astronaut_distribution_l72_7220

/-- The number of ways to distribute n astronauts among k distinct modules,
    with each module containing at least min and at most max astronauts. -/
def distribute_astronauts (n k min max : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 450 ways to distribute
    6 astronauts among 3 distinct modules, with each module containing
    at least 1 and at most 3 astronauts. -/
theorem astronaut_distribution :
  distribute_astronauts 6 3 1 3 = 450 :=
sorry

end NUMINAMATH_CALUDE_astronaut_distribution_l72_7220


namespace NUMINAMATH_CALUDE_factor_of_expression_l72_7298

theorem factor_of_expression (x y z : ℝ) :
  ∃ (f : ℝ → ℝ → ℝ → ℝ), (x^2 - y^2 - z^2 + 2*y*z + x + y - z) = (x - y + z + 1) * f x y z := by
  sorry

end NUMINAMATH_CALUDE_factor_of_expression_l72_7298


namespace NUMINAMATH_CALUDE_even_number_decomposition_theorem_l72_7206

def is_perfect_square (n : ℤ) : Prop := ∃ m : ℤ, n = m * m ∧ n > 0

def even_number_decomposition (k : ℤ) : Prop :=
  (∃ a b : ℤ, 2 * k = a + b ∧ is_perfect_square (a * b)) ∨
  (∃ c d : ℤ, 2 * k = c - d ∧ is_perfect_square (c * d))

theorem even_number_decomposition_theorem :
  ∃ S : Set ℤ, S.Finite ∧ ∀ k : ℤ, k ∉ S → even_number_decomposition k :=
sorry

end NUMINAMATH_CALUDE_even_number_decomposition_theorem_l72_7206


namespace NUMINAMATH_CALUDE_polynomial_real_root_iff_b_negative_l72_7205

/-- The polynomial x^3 + bx^2 - x + b = 0 has at least one real root if and only if b < 0 -/
theorem polynomial_real_root_iff_b_negative :
  ∀ b : ℝ, (∃ x : ℝ, x^3 + b*x^2 - x + b = 0) ↔ b < 0 := by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_iff_b_negative_l72_7205


namespace NUMINAMATH_CALUDE_garden_area_l72_7204

theorem garden_area (total_distance : ℝ) (length_walks : ℕ) (perimeter_walks : ℕ) :
  total_distance = 1500 →
  length_walks = 30 →
  perimeter_walks = 12 →
  ∃ (length width : ℝ),
    length > 0 ∧
    width > 0 ∧
    length * length_walks = total_distance ∧
    2 * (length + width) * perimeter_walks = total_distance ∧
    length * width = 625 := by
  sorry


end NUMINAMATH_CALUDE_garden_area_l72_7204


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l72_7226

/-- Given an arithmetic sequence {a_n} where a_5 = 8 and a_9 = 24, prove that a_4 = 4 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
  (h_a5 : a 5 = 8)
  (h_a9 : a 9 = 24) : 
  a 4 = 4 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l72_7226


namespace NUMINAMATH_CALUDE_system_of_equations_l72_7277

theorem system_of_equations (x y a : ℝ) : 
  (2 * x + y = 2 * a + 1) → 
  (x + 2 * y = a - 1) → 
  (x - y = 4) → 
  (a = 2) := by
sorry

end NUMINAMATH_CALUDE_system_of_equations_l72_7277


namespace NUMINAMATH_CALUDE_closed_polyline_theorem_l72_7265

/-- Represents a rectangle on a unit grid --/
structure Rectangle where
  m : ℕ  -- Width
  n : ℕ  -- Height

/-- Determines if a closed polyline exists for a given rectangle --/
def closedPolylineExists (rect : Rectangle) : Prop :=
  Odd rect.m ∨ Odd rect.n

/-- Calculates the length of the closed polyline if it exists --/
def polylineLength (rect : Rectangle) : ℕ :=
  (rect.n + 1) * (rect.m + 1)

/-- Main theorem about the existence and length of the closed polyline --/
theorem closed_polyline_theorem (rect : Rectangle) :
  closedPolylineExists rect ↔ 
    ∃ (length : ℕ), length = polylineLength rect ∧ 
      (∀ (i j : ℕ), i ≤ rect.m ∧ j ≤ rect.n → 
        ∃ (unique_visit : Prop), unique_visit) :=
by sorry

end NUMINAMATH_CALUDE_closed_polyline_theorem_l72_7265


namespace NUMINAMATH_CALUDE_probability_second_shiny_penny_l72_7253

def total_pennies : ℕ := 7
def shiny_pennies : ℕ := 4
def dull_pennies : ℕ := 3

def probability_more_than_three_draws : ℚ :=
  (Nat.choose 3 1 * Nat.choose 4 1 + Nat.choose 3 0 * Nat.choose 4 2) / Nat.choose total_pennies shiny_pennies

theorem probability_second_shiny_penny :
  probability_more_than_three_draws = 18 / 35 := by sorry

end NUMINAMATH_CALUDE_probability_second_shiny_penny_l72_7253


namespace NUMINAMATH_CALUDE_p_sequence_constant_difference_l72_7258

/-- A P-sequence is a geometric sequence {a_n} where (a_1 + 1, a_2 + 2, a_3 + 3) also forms a geometric sequence -/
def is_p_sequence (a : ℝ) (a_n : ℕ → ℝ) : Prop :=
  (∀ n, a_n (n + 1) = a_n n * a_n 1) ∧
  (∃ r, (a_n 1 + 1) * r = a_n 2 + 2 ∧ (a_n 2 + 2) * r = a_n 3 + 3)

theorem p_sequence_constant_difference (a : ℝ) (h1 : 1/2 < a) (h2 : a < 1) :
  let a_n : ℕ → ℝ := λ n => a^(2*n - 1)
  let x_n : ℕ → ℝ := λ n => a_n n - 1 / (a_n n)
  is_p_sequence a a_n →
  ∀ n ≥ 2, x_n n^2 - x_n (n-1) * x_n (n+1) = 5 := by
sorry

end NUMINAMATH_CALUDE_p_sequence_constant_difference_l72_7258


namespace NUMINAMATH_CALUDE_customers_in_other_countries_l72_7225

/-- Given a cell phone company with a total of 7422 customers,
    of which 723 live in the United States,
    prove that 6699 customers live in other countries. -/
theorem customers_in_other_countries
  (total : ℕ)
  (usa : ℕ)
  (h1 : total = 7422)
  (h2 : usa = 723) :
  total - usa = 6699 := by
  sorry

end NUMINAMATH_CALUDE_customers_in_other_countries_l72_7225


namespace NUMINAMATH_CALUDE_not_always_true_parallel_intersection_l72_7280

-- Define the types for planes and lines
variable (Plane : Type) (Line : Type)

-- Define the relations
variable (parallel_lines : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- Define the planes and lines
variable (α β : Plane)
variable (m n : Line)

-- State the theorem
theorem not_always_true_parallel_intersection
  (h_diff_planes : α ≠ β)
  (h_diff_lines : m ≠ n)
  (h_m_parallel_α : parallel_line_plane m α)
  (h_intersect : intersect α β n) :
  ¬ (parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_not_always_true_parallel_intersection_l72_7280


namespace NUMINAMATH_CALUDE_credit_card_interest_rate_l72_7215

theorem credit_card_interest_rate 
  (initial_balance : ℝ) 
  (payment : ℝ) 
  (new_balance : ℝ) 
  (h1 : initial_balance = 150)
  (h2 : payment = 50)
  (h3 : new_balance = 120) :
  (new_balance - (initial_balance - payment)) / initial_balance * 100 = 13.33 := by
sorry

end NUMINAMATH_CALUDE_credit_card_interest_rate_l72_7215


namespace NUMINAMATH_CALUDE_nyc_streetlights_l72_7281

/-- Given the total number of streetlights bought, the number of squares, and the number of streetlights per square, 
    calculate the number of unused streetlights. -/
def unused_streetlights (total : ℕ) (squares : ℕ) (per_square : ℕ) : ℕ :=
  total - squares * per_square

/-- Theorem stating that with 200 total streetlights, 15 squares, and 12 streetlights per square, 
    there will be 20 unused streetlights. -/
theorem nyc_streetlights : unused_streetlights 200 15 12 = 20 := by
  sorry

end NUMINAMATH_CALUDE_nyc_streetlights_l72_7281


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l72_7232

theorem polynomial_divisibility (p q : ℤ) : 
  (∀ x : ℝ, (x + 3) * (x - 2) ∣ (x^5 - 2*x^4 + 3*x^3 - p*x^2 + q*x + 12)) →
  p = -28 ∧ q = -74 := by
sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l72_7232


namespace NUMINAMATH_CALUDE_max_q_plus_r_for_1051_l72_7269

theorem max_q_plus_r_for_1051 :
  ∀ q r : ℕ+,
  1051 = 23 * q + r →
  ∀ q' r' : ℕ+,
  1051 = 23 * q' + r' →
  q + r ≤ 61 :=
by sorry

end NUMINAMATH_CALUDE_max_q_plus_r_for_1051_l72_7269


namespace NUMINAMATH_CALUDE_leo_caught_40_l72_7299

/-- The number of fish Leo caught -/
def leo_fish : ℕ := sorry

/-- The number of fish Agrey caught -/
def agrey_fish : ℕ := sorry

/-- Agrey caught 20 more fish than Leo -/
axiom agrey_more : agrey_fish = leo_fish + 20

/-- They caught a total of 100 fish together -/
axiom total_fish : leo_fish + agrey_fish = 100

/-- Prove that Leo caught 40 fish -/
theorem leo_caught_40 : leo_fish = 40 := by sorry

end NUMINAMATH_CALUDE_leo_caught_40_l72_7299


namespace NUMINAMATH_CALUDE_platform_length_l72_7268

/-- Given a train with speed 72 km/h and length 290.04 m, crossing a platform in 26 seconds,
    prove that the length of the platform is 229.96 m. -/
theorem platform_length (train_speed : ℝ) (train_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 →
  train_length = 290.04 →
  crossing_time = 26 →
  ∃ platform_length : ℝ,
    platform_length = 229.96 ∧
    platform_length = train_speed * (1000 / 3600) * crossing_time - train_length :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l72_7268


namespace NUMINAMATH_CALUDE_helmet_sales_theorem_l72_7296

/-- Represents the helmet sales scenario -/
structure HelmetSales where
  originalPrice : ℝ
  originalSales : ℝ
  costPrice : ℝ
  salesIncrease : ℝ

/-- Calculates the monthly profit given a price reduction -/
def monthlyProfit (hs : HelmetSales) (priceReduction : ℝ) : ℝ :=
  (hs.originalPrice - priceReduction - hs.costPrice) * (hs.originalSales + hs.salesIncrease * priceReduction)

/-- The main theorem about helmet sales -/
theorem helmet_sales_theorem (hs : HelmetSales) 
    (h1 : hs.originalPrice = 80)
    (h2 : hs.originalSales = 200)
    (h3 : hs.costPrice = 50)
    (h4 : hs.salesIncrease = 10) :
  (∃ (x : ℝ), x ≥ 10 ∧ monthlyProfit hs x = 5250 ∧ hs.originalPrice - x = 65) ∧
  (∃ (maxProfit : ℝ), maxProfit = 6000 ∧ 
    ∀ (y : ℝ), y ≥ 10 → monthlyProfit hs y ≤ maxProfit ∧
    monthlyProfit hs 10 = maxProfit) := by
  sorry

end NUMINAMATH_CALUDE_helmet_sales_theorem_l72_7296


namespace NUMINAMATH_CALUDE_circular_track_speed_l72_7202

/-- The speed of person A in rounds per hour -/
def speed_A : ℝ := 7

/-- The speed of person B in rounds per hour -/
def speed_B : ℝ := 3

/-- The number of times A and B cross each other in 1 hour -/
def crossings : ℕ := 10

/-- The time period in hours -/
def time_period : ℝ := 1

theorem circular_track_speed :
  speed_A + speed_B = crossings / time_period :=
by sorry

end NUMINAMATH_CALUDE_circular_track_speed_l72_7202


namespace NUMINAMATH_CALUDE_rectangle_y_value_l72_7295

/-- Given a rectangle with vertices at (-2, y), (10, y), (-2, 4), and (10, 4),
    where y is positive and the area is 108 square units, prove that y = 13. -/
theorem rectangle_y_value (y : ℝ) (h1 : y > 0) (h2 : (10 - (-2)) * (y - 4) = 108) : y = 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_y_value_l72_7295


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l72_7260

-- Define the quadratic function f
def f : ℝ → ℝ := λ x => x^2 - 2*x - 1

-- State the theorem
theorem quadratic_function_properties :
  (∀ x, f x ≥ -2) ∧  -- minimum value is -2
  f 3 = 2 ∧ f (-1) = 2 ∧  -- given conditions
  (∀ t, f (2*t^2 - 4*t + 3) > f (t^2 + t + 3) ↔ t > 5 ∨ t < 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l72_7260


namespace NUMINAMATH_CALUDE_container_problem_l72_7273

theorem container_problem (x y : ℝ) :
  (∃ (large_capacity small_capacity : ℝ),
    large_capacity = x ∧
    small_capacity = y ∧
    5 * large_capacity + small_capacity = 3 ∧
    large_capacity + 5 * small_capacity = 2) →
  (5 * x + y = 3 ∧ x + 5 * y = 2) :=
by sorry

end NUMINAMATH_CALUDE_container_problem_l72_7273


namespace NUMINAMATH_CALUDE_pyramid_layers_l72_7221

/-- Represents a pyramid with layers of sandstone blocks -/
structure Pyramid where
  total_blocks : ℕ
  layer_ratio : ℕ
  top_layer_blocks : ℕ

/-- Calculates the number of layers in a pyramid -/
def num_layers (p : Pyramid) : ℕ :=
  sorry

/-- Theorem stating that a pyramid with 40 blocks, 3:1 layer ratio, and single top block has 4 layers -/
theorem pyramid_layers (p : Pyramid) 
  (h1 : p.total_blocks = 40)
  (h2 : p.layer_ratio = 3)
  (h3 : p.top_layer_blocks = 1) :
  num_layers p = 4 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_layers_l72_7221


namespace NUMINAMATH_CALUDE_parametric_to_standard_equation_l72_7210

theorem parametric_to_standard_equation (x y t : ℝ) :
  (∃ θ : ℝ, x = (1/2) * (Real.exp t + Real.exp (-t)) * Real.cos θ ∧
             y = (1/2) * (Real.exp t - Real.exp (-t)) * Real.sin θ) →
  x^2 * (Real.exp (2*t) - 2 + Real.exp (-2*t)) + 
  y^2 * (Real.exp (2*t) + 2 + Real.exp (-2*t)) = 
  Real.exp (6*t) - 2 * Real.exp (4*t) + Real.exp (2*t) + 
  2 * Real.exp (4*t) - 4 * Real.exp (2*t) + 2 + 
  Real.exp (2*t) - 2 * Real.exp (-2*t) + Real.exp (-4*t) :=
by sorry

end NUMINAMATH_CALUDE_parametric_to_standard_equation_l72_7210


namespace NUMINAMATH_CALUDE_count_two_digit_S_equal_l72_7245

def S (n : ℕ) : ℕ :=
  (n % 2) + (n % 3) + (n % 4) + (n % 5)

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem count_two_digit_S_equal : 
  ∃ (l : List ℕ), (∀ n ∈ l, is_two_digit n ∧ S n = S (n + 1)) ∧ 
                  (∀ n, is_two_digit n → S n = S (n + 1) → n ∈ l) ∧
                  l.length = 6 :=
sorry

end NUMINAMATH_CALUDE_count_two_digit_S_equal_l72_7245


namespace NUMINAMATH_CALUDE_colins_class_girls_l72_7283

theorem colins_class_girls (total : ℕ) (boys : ℕ) (girls : ℕ) :
  total = 35 →
  boys > 15 →
  girls + boys = total →
  4 * girls = 3 * boys →
  girls = 15 :=
by sorry

end NUMINAMATH_CALUDE_colins_class_girls_l72_7283


namespace NUMINAMATH_CALUDE_possible_values_of_a_l72_7236

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 19*x^3) 
  (h3 : a - b = x) : 
  a = 3*x ∨ a = -2*x := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l72_7236


namespace NUMINAMATH_CALUDE_juan_cars_count_l72_7292

theorem juan_cars_count (num_bicycles num_pickup_trucks num_tricycles total_tires : ℕ)
  (h1 : num_bicycles = 3)
  (h2 : num_pickup_trucks = 8)
  (h3 : num_tricycles = 1)
  (h4 : total_tires = 101)
  (h5 : ∀ (num_cars : ℕ), total_tires = 4 * num_cars + 2 * num_bicycles + 4 * num_pickup_trucks + 3 * num_tricycles) :
  ∃ (num_cars : ℕ), num_cars = 15 ∧ total_tires = 4 * num_cars + 2 * num_bicycles + 4 * num_pickup_trucks + 3 * num_tricycles :=
by
  sorry

end NUMINAMATH_CALUDE_juan_cars_count_l72_7292


namespace NUMINAMATH_CALUDE_colberts_treehouse_l72_7229

theorem colberts_treehouse (total : ℕ) (storage : ℕ) (parents : ℕ) (store : ℕ) (friends : ℕ) : 
  total = 200 →
  storage = total / 4 →
  parents = total / 2 →
  store = 30 →
  total = storage + parents + store + friends →
  friends = 20 := by
sorry

end NUMINAMATH_CALUDE_colberts_treehouse_l72_7229


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l72_7231

theorem complex_arithmetic_equality : (90 + 5) * (12 / (180 / (3^2))) = 57 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l72_7231


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l72_7237

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define the concept of axis of symmetry
def AxisOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- State the theorem
theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) (h : EvenFunction f) :
  AxisOfSymmetry (fun x ↦ f (x + 1)) (-1) :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l72_7237


namespace NUMINAMATH_CALUDE_sector_area_l72_7235

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 4) (h2 : central_angle = 2) :
  let radius := perimeter / (2 + central_angle)
  let arc_length := radius * central_angle
  let area := (1 / 2) * radius * arc_length
  area = 1 := by sorry

end NUMINAMATH_CALUDE_sector_area_l72_7235


namespace NUMINAMATH_CALUDE_division_problem_l72_7286

theorem division_problem (dividend quotient remainder : ℕ) 
  (h1 : dividend = 690) 
  (h2 : quotient = 19) 
  (h3 : remainder = 6) :
  ∃ divisor : ℕ, 
    dividend = divisor * quotient + remainder ∧ 
    divisor = 36 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l72_7286


namespace NUMINAMATH_CALUDE_square_construction_theorem_l72_7209

/-- A line in a plane -/
structure Line :=
  (point : ℝ × ℝ)
  (direction : ℝ × ℝ)

/-- A square in a plane -/
structure Square :=
  (center : ℝ × ℝ)
  (side_length : ℝ)
  (rotation : ℝ)

/-- Check if a point lies on a line -/
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop := sorry

/-- Check if a line intersects a square (including its extensions) -/
def line_intersects_square (l : Line) (s : Square) : Prop := sorry

/-- The main theorem -/
theorem square_construction_theorem 
  (L : Line) 
  (A B C D : ℝ × ℝ) 
  (h_distinct : A ≠ B ∧ B ≠ C ∧ C ≠ D)
  (h_order : point_on_line A L ∧ point_on_line B L ∧ point_on_line C L ∧ point_on_line D L) :
  ∃ (S : Square), 
    (∃ (p q : ℝ × ℝ), line_intersects_square L S ∧ p ≠ q ∧ 
      ((p = A ∧ q = B) ∨ (p = B ∧ q = A))) ∧
    (∃ (r s : ℝ × ℝ), line_intersects_square L S ∧ r ≠ s ∧ 
      ((r = C ∧ s = D) ∨ (r = D ∧ s = C))) :=
sorry

end NUMINAMATH_CALUDE_square_construction_theorem_l72_7209


namespace NUMINAMATH_CALUDE_flag_design_count_l72_7266

/-- The number of color choices for each stripe -/
def num_colors : ℕ := 3

/-- The number of stripes on the flag -/
def num_stripes : ℕ := 3

/-- The total number of possible flag designs -/
def total_flag_designs : ℕ := num_colors ^ num_stripes

/-- Theorem: The number of possible flag designs is 27 -/
theorem flag_design_count : total_flag_designs = 27 := by
  sorry

end NUMINAMATH_CALUDE_flag_design_count_l72_7266


namespace NUMINAMATH_CALUDE_exercise_time_distribution_l72_7267

theorem exercise_time_distribution (total_time : ℕ) (aerobics_ratio : ℕ) (weight_ratio : ℕ) 
  (h1 : total_time = 250)
  (h2 : aerobics_ratio = 3)
  (h3 : weight_ratio = 2) :
  ∃ (aerobics_time weight_time : ℕ),
    aerobics_time + weight_time = total_time ∧
    aerobics_time * weight_ratio = weight_time * aerobics_ratio ∧
    aerobics_time = 150 ∧
    weight_time = 100 := by
  sorry

end NUMINAMATH_CALUDE_exercise_time_distribution_l72_7267


namespace NUMINAMATH_CALUDE_dolls_count_l72_7271

/-- The number of dolls Hannah's sister has -/
def sister_dolls : ℝ := 8.5

/-- The ratio of Hannah's dolls to her sister's dolls -/
def hannah_ratio : ℝ := 5.5

/-- The ratio of cousin's dolls to Hannah and her sister's combined dolls -/
def cousin_ratio : ℝ := 7

/-- The total number of dolls Hannah, her sister, and their cousin have -/
def total_dolls : ℝ := 
  sister_dolls + (hannah_ratio * sister_dolls) + (cousin_ratio * (sister_dolls + hannah_ratio * sister_dolls))

theorem dolls_count : total_dolls = 442 := by
  sorry

end NUMINAMATH_CALUDE_dolls_count_l72_7271


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l72_7294

/-- Given a geometric sequence with 10 terms, first term 6, and last term 93312,
    prove that the 7th term is 279936 -/
theorem seventh_term_of_geometric_sequence :
  ∀ (a : ℕ → ℝ),
    (∀ i j, a (i + 1) / a i = a (j + 1) / a j) →  -- geometric sequence condition
    a 1 = 6 →                                     -- first term is 6
    a 10 = 93312 →                                -- last term is 93312
    a 7 = 279936 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l72_7294


namespace NUMINAMATH_CALUDE_blueberry_pie_count_l72_7223

theorem blueberry_pie_count (total_pies : ℕ) (apple_ratio blueberry_ratio cherry_ratio : ℕ) :
  total_pies = 36 →
  apple_ratio = 2 →
  blueberry_ratio = 5 →
  cherry_ratio = 3 →
  blueberry_ratio * total_pies / (apple_ratio + blueberry_ratio + cherry_ratio) = 18 :=
by sorry

end NUMINAMATH_CALUDE_blueberry_pie_count_l72_7223


namespace NUMINAMATH_CALUDE_defective_rate_is_twenty_percent_l72_7254

variable (n : ℕ)  -- number of defective items among 10 products

-- Define the probability of selecting one defective item out of two random selections
def prob_one_defective (n : ℕ) : ℚ :=
  (n * (10 - n)) / (10 * 9)

-- Theorem statement
theorem defective_rate_is_twenty_percent :
  n ≤ 10 ∧                     -- n is at most 10 (total number of products)
  prob_one_defective n = 16/45 ∧ -- probability of selecting one defective item is 16/45
  n ≤ 4 →                      -- defective rate does not exceed 40%
  n = 2                        -- implies that n = 2, which means 20% defective rate
  := by sorry

end NUMINAMATH_CALUDE_defective_rate_is_twenty_percent_l72_7254


namespace NUMINAMATH_CALUDE_percentage_problem_l72_7240

theorem percentage_problem (P : ℝ) : (P / 100 * 1265) / 6 = 543.95 ↔ P = 258 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l72_7240


namespace NUMINAMATH_CALUDE_brothers_catch_up_time_l72_7287

/-- The time taken for the older brother to catch up with the younger brother -/
def catchUpTime (olderTime youngerTime delay : ℚ) : ℚ :=
  let relativeSpeed := 1 / olderTime - 1 / youngerTime
  let distanceCovered := delay / youngerTime
  delay + distanceCovered / relativeSpeed

/-- Theorem stating the catch-up time for the given problem -/
theorem brothers_catch_up_time :
  catchUpTime 12 20 5 = 25/2 := by
  sorry

#eval catchUpTime 12 20 5

end NUMINAMATH_CALUDE_brothers_catch_up_time_l72_7287


namespace NUMINAMATH_CALUDE_condition_for_reciprocal_less_than_one_l72_7219

theorem condition_for_reciprocal_less_than_one (a : ℝ) :
  (a > 1 → (1 / a) < 1) ∧ (∃ b : ℝ, (1 / b) < 1 ∧ b ≤ 1) := by sorry

end NUMINAMATH_CALUDE_condition_for_reciprocal_less_than_one_l72_7219


namespace NUMINAMATH_CALUDE_triangle_trig_expression_l72_7218

theorem triangle_trig_expression (D E F : Real) (h1 : 0 < D) (h2 : 0 < E) (h3 : 0 < F)
  (h4 : D + E + F = Real.pi) 
  (h5 : Real.sin F * 8 = 7) (h6 : Real.sin D * 5 = 8) (h7 : Real.sin E * 7 = 5) :
  (Real.cos ((D - E) / 2) / Real.sin (F / 2)) - (Real.sin ((D - E) / 2) / Real.cos (F / 2)) = 
  1 / Real.sqrt ((1 + Real.sqrt (15 / 64)) / 2) := by
  sorry

end NUMINAMATH_CALUDE_triangle_trig_expression_l72_7218


namespace NUMINAMATH_CALUDE_vertex_segments_different_colors_l72_7279

structure ColoredTriangle where
  n : ℕ  -- number of marked points inside the triangle
  k₁ : ℕ  -- number of segments of first color connected to vertices
  k₂ : ℕ  -- number of segments of second color connected to vertices
  k₃ : ℕ  -- number of segments of third color connected to vertices
  sum_k : k₁ + k₂ + k₃ = 3
  valid_k : 0 ≤ k₁ ∧ k₁ ≤ 3 ∧ 0 ≤ k₂ ∧ k₂ ≤ 3 ∧ 0 ≤ k₃ ∧ k₃ ≤ 3
  even_sum : Even (n + k₁) ∧ Even (n + k₂) ∧ Even (n + k₃)

theorem vertex_segments_different_colors (t : ColoredTriangle) : t.k₁ = 1 ∧ t.k₂ = 1 ∧ t.k₃ = 1 := by
  sorry

end NUMINAMATH_CALUDE_vertex_segments_different_colors_l72_7279


namespace NUMINAMATH_CALUDE_right_triangle_area_l72_7248

theorem right_triangle_area (a b c : ℝ) (h_right : a^2 + b^2 = c^2)
  (h_ratio : a / b = 7 / 24)
  (h_distance : (c / 2) * ((c / 2) - 2 * ((a + b - c) / 2)) = 1) :
  (1 / 2) * a * b = 336 / 325 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l72_7248


namespace NUMINAMATH_CALUDE_shirts_returned_l72_7234

/-- Given that Haley bought 11 shirts initially and ended up with 5 shirts,
    prove that she returned 6 shirts. -/
theorem shirts_returned (initial_shirts : ℕ) (final_shirts : ℕ) (h1 : initial_shirts = 11) (h2 : final_shirts = 5) :
  initial_shirts - final_shirts = 6 := by
  sorry

end NUMINAMATH_CALUDE_shirts_returned_l72_7234


namespace NUMINAMATH_CALUDE_constant_ratio_problem_l72_7201

theorem constant_ratio_problem (x y : ℝ) (k : ℝ) : 
  (∀ x y, k = (5 * x - 3) / (y + 10)) →  -- The ratio is constant for all x and y
  (k = (5 * 3 - 3) / (4 + 10)) →         -- When x = 3, y = 4
  (19 + 10 = (5 * (39 / 7) - 3) / k) →   -- When y = 19, x = 39/7
  x = 39 / 7                             -- Conclusion
  := by sorry

end NUMINAMATH_CALUDE_constant_ratio_problem_l72_7201


namespace NUMINAMATH_CALUDE_orange_groups_count_l72_7255

/-- The number of oranges in Philip's collection -/
def total_oranges : ℕ := 356

/-- The number of oranges in each group -/
def oranges_per_group : ℕ := 2

/-- The number of groups of oranges -/
def orange_groups : ℕ := total_oranges / oranges_per_group

theorem orange_groups_count : orange_groups = 178 := by
  sorry

end NUMINAMATH_CALUDE_orange_groups_count_l72_7255


namespace NUMINAMATH_CALUDE_marie_cash_register_cost_l72_7230

/-- A bakery's daily sales and expenses -/
structure BakeryFinances where
  bread_quantity : ℕ
  bread_price : ℝ
  cake_quantity : ℕ
  cake_price : ℝ
  rent : ℝ
  electricity : ℝ

/-- Calculate the cost of a cash register based on daily sales and expenses -/
def cash_register_cost (finances : BakeryFinances) (days : ℕ) : ℝ :=
  let daily_sales := finances.bread_quantity * finances.bread_price + 
                     finances.cake_quantity * finances.cake_price
  let daily_expenses := finances.rent + finances.electricity
  let daily_profit := daily_sales - daily_expenses
  days * daily_profit

/-- Marie's bakery finances -/
def marie_finances : BakeryFinances :=
  { bread_quantity := 40
  , bread_price := 2
  , cake_quantity := 6
  , cake_price := 12
  , rent := 20
  , electricity := 2 }

/-- Theorem: The cost of Marie's cash register is $1040 -/
theorem marie_cash_register_cost :
  cash_register_cost marie_finances 8 = 1040 := by
  sorry

end NUMINAMATH_CALUDE_marie_cash_register_cost_l72_7230


namespace NUMINAMATH_CALUDE_vaishalis_total_stripes_l72_7242

/-- Represents the types of stripes on hats --/
inductive StripeType
  | Solid
  | Zigzag
  | Wavy
  | Other

/-- Represents a hat with its stripe count and type --/
structure Hat :=
  (stripeCount : ℕ)
  (stripeType : StripeType)

/-- Determines if a stripe type should be counted --/
def countStripe (st : StripeType) : Bool :=
  match st with
  | StripeType.Solid => true
  | StripeType.Zigzag => true
  | StripeType.Wavy => true
  | _ => false

/-- Calculates the total number of counted stripes for a list of hats --/
def totalCountedStripes (hats : List Hat) : ℕ :=
  hats.foldl (fun acc hat => 
    if countStripe hat.stripeType then
      acc + hat.stripeCount
    else
      acc
  ) 0

/-- Vaishali's hat collection --/
def vaishalisHats : List Hat := [
  { stripeCount := 3, stripeType := StripeType.Solid },
  { stripeCount := 3, stripeType := StripeType.Solid },
  { stripeCount := 3, stripeType := StripeType.Solid },
  { stripeCount := 3, stripeType := StripeType.Solid },
  { stripeCount := 5, stripeType := StripeType.Zigzag },
  { stripeCount := 5, stripeType := StripeType.Zigzag },
  { stripeCount := 2, stripeType := StripeType.Wavy },
  { stripeCount := 2, stripeType := StripeType.Wavy },
  { stripeCount := 2, stripeType := StripeType.Wavy }
]

theorem vaishalis_total_stripes :
  totalCountedStripes vaishalisHats = 28 := by
  sorry

end NUMINAMATH_CALUDE_vaishalis_total_stripes_l72_7242


namespace NUMINAMATH_CALUDE_reciprocal_of_negative_one_l72_7250

theorem reciprocal_of_negative_one :
  ∃ x : ℚ, x * (-1) = 1 ∧ x = -1 :=
sorry

end NUMINAMATH_CALUDE_reciprocal_of_negative_one_l72_7250


namespace NUMINAMATH_CALUDE_curve_symmetrical_about_y_axis_l72_7289

/-- A curve defined by an equation f(x, y) = 0 is symmetrical about the y-axis
    if f(-x, y) = f(x, y) for all x and y. -/
def is_symmetrical_about_y_axis (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y, f (-x) y = f x y

/-- The equation x^2 - y^2 = 1 can be represented as a function f(x, y) = x^2 - y^2 - 1. -/
def f (x y : ℝ) : ℝ := x^2 - y^2 - 1

/-- Theorem: The curve defined by x^2 - y^2 = 1 is symmetrical about the y-axis. -/
theorem curve_symmetrical_about_y_axis : is_symmetrical_about_y_axis f := by
  sorry

end NUMINAMATH_CALUDE_curve_symmetrical_about_y_axis_l72_7289


namespace NUMINAMATH_CALUDE_perpendicular_parallel_relations_l72_7274

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpPlane : Line → Plane → Prop)
variable (para : Line → Line → Prop)
variable (paraPlane : Line → Plane → Prop)

-- Given: l is perpendicular to α
variable (l : Line) (α : Plane)
variable (h : perpPlane l α)

-- Theorem to prove
theorem perpendicular_parallel_relations :
  (∀ m : Line, perpPlane m α → para m l) ∧
  (∀ m : Line, paraPlane m α → perp m l) ∧
  (∀ m : Line, para m l → perpPlane m α) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_relations_l72_7274


namespace NUMINAMATH_CALUDE_square_perimeter_from_area_l72_7262

theorem square_perimeter_from_area (area : ℝ) (perimeter : ℝ) :
  area = 225 → perimeter = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_from_area_l72_7262


namespace NUMINAMATH_CALUDE_no_consecutive_squares_l72_7256

/-- The number of positive integer divisors of n -/
def tau (n : ℕ) : ℕ := (Nat.divisors n).card

/-- The sequence defined by a_(n+1) = a_n + τ(n) -/
def a : ℕ → ℕ
  | 0 => 1  -- Arbitrary starting value
  | n + 1 => a n + tau n

/-- Two consecutive terms of the sequence cannot both be perfect squares -/
theorem no_consecutive_squares (n : ℕ) : ¬(∃ k m : ℕ, a n = k^2 ∧ a (n + 1) = m^2) := by
  sorry


end NUMINAMATH_CALUDE_no_consecutive_squares_l72_7256


namespace NUMINAMATH_CALUDE_seat_to_right_of_xiaofang_l72_7270

/-- Represents a seat position as an ordered pair of integers -/
structure SeatPosition :=
  (column : ℤ)
  (row : ℤ)

/-- Returns the seat position to the right of a given seat -/
def seatToRight (seat : SeatPosition) : SeatPosition :=
  { column := seat.column + 1, row := seat.row }

/-- Xiaofang's seat position -/
def xiaofangSeat : SeatPosition := { column := 3, row := 5 }

theorem seat_to_right_of_xiaofang :
  seatToRight xiaofangSeat = { column := 4, row := 5 } := by sorry

end NUMINAMATH_CALUDE_seat_to_right_of_xiaofang_l72_7270


namespace NUMINAMATH_CALUDE_adjacent_cells_difference_l72_7252

/-- A type representing a cell in an n × n grid --/
structure Cell (n : ℕ) where
  row : Fin n
  col : Fin n

/-- A function representing the placement of integers in the grid --/
def GridPlacement (n : ℕ) := Cell n → Fin (n^2)

/-- Two cells are adjacent if they share a side or a corner --/
def adjacent {n : ℕ} (c1 c2 : Cell n) : Prop :=
  (c1.row = c2.row ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row = c2.row ∧ c2.col.val + 1 = c1.col.val) ∨
  (c1.col = c2.col ∧ c1.row.val + 1 = c2.row.val) ∨
  (c1.col = c2.col ∧ c2.row.val + 1 = c1.row.val) ∨
  (c1.row.val + 1 = c2.row.val ∧ c1.col.val + 1 = c2.col.val) ∨
  (c1.row.val + 1 = c2.row.val ∧ c2.col.val + 1 = c1.col.val) ∨
  (c2.row.val + 1 = c1.row.val ∧ c1.col.val + 1 = c2.col.val) ∨
  (c2.row.val + 1 = c1.row.val ∧ c2.col.val + 1 = c1.col.val)

/-- The main theorem --/
theorem adjacent_cells_difference {n : ℕ} (h : n > 0) (g : GridPlacement n) :
  ∃ (c1 c2 : Cell n), adjacent c1 c2 ∧ 
    ((g c1).val + n + 1 ≤ (g c2).val ∨ (g c2).val + n + 1 ≤ (g c1).val) :=
sorry

end NUMINAMATH_CALUDE_adjacent_cells_difference_l72_7252
