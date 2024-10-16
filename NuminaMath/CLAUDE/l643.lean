import Mathlib

namespace NUMINAMATH_CALUDE_midpoint_intersection_l643_64340

/-- Given a line segment from (1,3) to (5,11), if the line x + y = b
    intersects this segment at its midpoint, then b = 10. -/
theorem midpoint_intersection (b : ℝ) : 
  (∃ (x y : ℝ), x + y = b ∧ 
   x = (1 + 5) / 2 ∧ 
   y = (3 + 11) / 2) → 
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_midpoint_intersection_l643_64340


namespace NUMINAMATH_CALUDE_greatest_integer_solution_l643_64362

theorem greatest_integer_solution (x : ℝ) : 
  (((|x^2 - 2| - 7) * (|x + 3| - 5)) / (|x - 3| - |x - 1|) > 0) → 
  (∃ (n : ℤ), n ≤ x ∧ n ≤ 1 ∧ ∀ (m : ℤ), m ≤ x → m ≤ n) :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_solution_l643_64362


namespace NUMINAMATH_CALUDE_price_ratio_theorem_l643_64348

theorem price_ratio_theorem (cost_price : ℝ) (price1 price2 : ℝ) 
  (h1 : price1 = cost_price * (1 + 0.32))
  (h2 : price2 = cost_price * (1 - 0.12)) :
  price2 / price1 = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_price_ratio_theorem_l643_64348


namespace NUMINAMATH_CALUDE_number_and_square_relationship_l643_64318

theorem number_and_square_relationship (n : ℕ) (h : n = 8) : n^2 + n = 72 := by
  sorry

end NUMINAMATH_CALUDE_number_and_square_relationship_l643_64318


namespace NUMINAMATH_CALUDE_inscribed_circle_diameter_is_six_l643_64300

/-- The diameter of the inscribed circle in a right triangle with sides 9, 12, and 15 -/
def inscribed_circle_diameter (a b c : ℝ) : ℝ :=
  a + b - c

/-- Theorem: The diameter of the inscribed circle in a right triangle with sides 9, 12, and 15 is 6 -/
theorem inscribed_circle_diameter_is_six :
  let a : ℝ := 9
  let b : ℝ := 12
  let c : ℝ := 15
  a^2 + b^2 = c^2 →  -- Pythagorean theorem to ensure it's a right triangle
  inscribed_circle_diameter a b c = 6 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_diameter_is_six_l643_64300


namespace NUMINAMATH_CALUDE_sqrt_29_between_5_and_6_l643_64317

theorem sqrt_29_between_5_and_6 : 5 < Real.sqrt 29 ∧ Real.sqrt 29 < 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_29_between_5_and_6_l643_64317


namespace NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l643_64306

theorem product_of_numbers_with_given_sum_and_difference :
  ∀ x y : ℝ, x + y = 30 ∧ x - y = 10 → x * y = 200 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_with_given_sum_and_difference_l643_64306


namespace NUMINAMATH_CALUDE_inverse_proportion_l643_64345

theorem inverse_proportion (x y : ℝ) (h : x ≠ 0) : 
  (3 * x * y = 1) ↔ ∃ k : ℝ, k ≠ 0 ∧ y = k / x := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l643_64345


namespace NUMINAMATH_CALUDE_circle_line_intersection_and_min_chord_l643_64369

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 25

-- Define the line L
def L (m x y : ℝ) : Prop := (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

-- Theorem statement
theorem circle_line_intersection_and_min_chord :
  -- Part 1: L always intersects C at two points for any real m
  (∀ m : ℝ, ∃! (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ C p₁.1 p₁.2 ∧ C p₂.1 p₂.2 ∧ L m p₁.1 p₁.2 ∧ L m p₂.1 p₂.2) ∧
  -- Part 2: The equation of L for minimum chord length
  (∃ m : ℝ, ∀ x y : ℝ, L m x y ↔ y = 2*x - 5) :=
sorry

end NUMINAMATH_CALUDE_circle_line_intersection_and_min_chord_l643_64369


namespace NUMINAMATH_CALUDE_x_squared_mod_25_l643_64335

theorem x_squared_mod_25 (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 25]) 
  (h2 : 4 * x ≡ 21 [ZMOD 25]) : 
  x^2 ≡ 21 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_25_l643_64335


namespace NUMINAMATH_CALUDE_log_equality_l643_64395

theorem log_equality (y : ℝ) (k : ℝ) 
  (h1 : Real.log 3 / Real.log 8 = y)
  (h2 : Real.log 243 / Real.log 2 = k * y) : 
  k = 15 := by
sorry

end NUMINAMATH_CALUDE_log_equality_l643_64395


namespace NUMINAMATH_CALUDE_chewing_gum_revenue_comparison_l643_64356

theorem chewing_gum_revenue_comparison 
  (last_year_revenue : ℝ) 
  (projected_increase_rate : ℝ) 
  (actual_decrease_rate : ℝ) 
  (h1 : projected_increase_rate = 0.25)
  (h2 : actual_decrease_rate = 0.25) :
  (last_year_revenue * (1 - actual_decrease_rate)) / 
  (last_year_revenue * (1 + projected_increase_rate)) = 0.6 := by
sorry

end NUMINAMATH_CALUDE_chewing_gum_revenue_comparison_l643_64356


namespace NUMINAMATH_CALUDE_max_M_min_N_equals_two_thirds_l643_64357

theorem max_M_min_N_equals_two_thirds (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  let M := x / (2 * x + y) + y / (x + 2 * y)
  let N := x / (x + 2 * y) + y / (2 * x + y)
  (∀ a b : ℝ, a > 0 → b > 0 → M ≤ (a / (2 * a + b) + b / (a + 2 * b))) ∧
  (∀ a b : ℝ, a > 0 → b > 0 → N ≥ (a / (a + 2 * b) + b / (2 * a + b))) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ M = 2/3) ∧
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ N = 2/3) :=
by sorry


end NUMINAMATH_CALUDE_max_M_min_N_equals_two_thirds_l643_64357


namespace NUMINAMATH_CALUDE_total_revenue_is_628_l643_64383

/-- Represents the characteristics of a pie type -/
structure PieType where
  slices_per_pie : ℕ
  price_per_slice : ℕ
  pies_sold : ℕ

/-- Calculates the revenue for a single pie type -/
def revenue_for_pie_type (pie : PieType) : ℕ :=
  pie.slices_per_pie * pie.price_per_slice * pie.pies_sold

/-- Defines the pumpkin pie -/
def pumpkin_pie : PieType :=
  { slices_per_pie := 8, price_per_slice := 5, pies_sold := 4 }

/-- Defines the custard pie -/
def custard_pie : PieType :=
  { slices_per_pie := 6, price_per_slice := 6, pies_sold := 5 }

/-- Defines the apple pie -/
def apple_pie : PieType :=
  { slices_per_pie := 10, price_per_slice := 4, pies_sold := 3 }

/-- Defines the pecan pie -/
def pecan_pie : PieType :=
  { slices_per_pie := 12, price_per_slice := 7, pies_sold := 2 }

/-- Theorem stating that the total revenue from all pie sales is $628 -/
theorem total_revenue_is_628 :
  revenue_for_pie_type pumpkin_pie +
  revenue_for_pie_type custard_pie +
  revenue_for_pie_type apple_pie +
  revenue_for_pie_type pecan_pie = 628 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_is_628_l643_64383


namespace NUMINAMATH_CALUDE_concave_probability_l643_64377

/-- A bottle cap with two sides -/
structure BottleCap where
  convex : ℝ
  concave : ℝ
  sum_to_one : convex + concave = 1
  non_negative : 0 ≤ convex ∧ 0 ≤ concave

/-- Theorem: If the probability of the convex side being up is 0.44,
    then the probability of the concave side being up is 0.56 -/
theorem concave_probability (cap : BottleCap) (h : cap.convex = 0.44) :
  cap.concave = 0.56 := by
sorry

end NUMINAMATH_CALUDE_concave_probability_l643_64377


namespace NUMINAMATH_CALUDE_tea_brewing_time_proof_l643_64309

/-- The time needed to wash the kettle and fill it with cold water -/
def wash_kettle_time : ℕ := 2

/-- The time needed to wash the teapot and cups -/
def wash_teapot_cups_time : ℕ := 2

/-- The time needed to get tea leaves -/
def get_tea_leaves_time : ℕ := 1

/-- The time needed to boil water -/
def boil_water_time : ℕ := 15

/-- The time needed to brew the tea -/
def brew_tea_time : ℕ := 1

/-- The shortest operation time for brewing a pot of tea -/
def shortest_operation_time : ℕ := 18

theorem tea_brewing_time_proof :
  shortest_operation_time = max wash_kettle_time (max boil_water_time brew_tea_time) :=
by sorry

end NUMINAMATH_CALUDE_tea_brewing_time_proof_l643_64309


namespace NUMINAMATH_CALUDE_remainder_three_power_2023_mod_5_l643_64386

theorem remainder_three_power_2023_mod_5 : 3^2023 % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_power_2023_mod_5_l643_64386


namespace NUMINAMATH_CALUDE_lcm_gcd_12_15_l643_64368

theorem lcm_gcd_12_15 :
  let a := 12
  let b := 15
  (Nat.lcm a b * Nat.gcd a b = 180) ∧
  (Nat.lcm a b + Nat.gcd a b = 63) := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_12_15_l643_64368


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_conditions_l643_64361

theorem sufficient_not_necessary_conditions (a b : ℝ) :
  (∀ (a b : ℝ), a + b > 2 → a + b > 0) ∧
  (∀ (a b : ℝ), (a > 0 ∧ b > 0) → a + b > 0) ∧
  (∃ (a b : ℝ), a + b > 0 ∧ ¬(a + b > 2)) ∧
  (∃ (a b : ℝ), a + b > 0 ∧ ¬(a > 0 ∧ b > 0)) ∧
  (∃ (a b : ℝ), ¬(ab > 0) ∧ a + b > 0) ∧
  (∃ (a b : ℝ), ¬(a > 0 ∨ b > 0) ∧ a + b > 0) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_conditions_l643_64361


namespace NUMINAMATH_CALUDE_min_n_for_inequality_l643_64338

-- Define the notation x[n] for repeated exponentiation
def repeated_exp (x : ℕ) : ℕ → ℕ
| 0 => x
| n + 1 => x ^ (repeated_exp x n)

-- Define the specific case for 3[n]
def three_exp (n : ℕ) : ℕ := repeated_exp 3 n

-- Theorem statement
theorem min_n_for_inequality : 
  ∀ n : ℕ, three_exp n > 3^(2^9) ↔ n ≥ 10 :=
sorry

end NUMINAMATH_CALUDE_min_n_for_inequality_l643_64338


namespace NUMINAMATH_CALUDE_perfect_square_base9_l643_64312

/-- Represents a number in base 9 of the form ac7b -/
structure Base9Number where
  a : ℕ
  c : ℕ
  b : ℕ
  a_nonzero : a ≠ 0
  b_less_than_9 : b < 9
  c_less_than_9 : c < 9

/-- Converts a Base9Number to its decimal representation -/
def toDecimal (n : Base9Number) : ℕ :=
  729 * n.a + 81 * n.c + 63 + n.b

/-- Theorem stating that if a Base9Number is a perfect square, then b must be 0 -/
theorem perfect_square_base9 (n : Base9Number) :
  ∃ (k : ℕ), toDecimal n = k^2 → n.b = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_base9_l643_64312


namespace NUMINAMATH_CALUDE_simplify_expression_l643_64341

theorem simplify_expression : ∃ (a b c : ℕ+), 
  (Real.sqrt 3 + 2 / Real.sqrt 3 + Real.sqrt 11 + 3 / Real.sqrt 11 = (a * Real.sqrt 3 + b * Real.sqrt 11) / c) ∧
  (∀ (a' b' c' : ℕ+), 
    Real.sqrt 3 + 2 / Real.sqrt 3 + Real.sqrt 11 + 3 / Real.sqrt 11 = (a' * Real.sqrt 3 + b' * Real.sqrt 11) / c' →
    c ≤ c') ∧
  a = 39 ∧ b = 42 ∧ c = 33 := by
sorry

end NUMINAMATH_CALUDE_simplify_expression_l643_64341


namespace NUMINAMATH_CALUDE_water_ice_mixture_theorem_l643_64303

/-- Represents the properties of water and ice mixture -/
structure WaterIceMixture where
  total_mass : ℝ
  water_mass : ℝ
  ice_mass : ℝ
  water_mass_added : ℝ
  initial_temp : ℝ
  final_temp : ℝ
  latent_heat_fusion : ℝ

/-- Calculates the heat balance for the water-ice mixture -/
def heat_balance (m : WaterIceMixture) : ℝ :=
  m.water_mass_added * (m.initial_temp - m.final_temp) -
  (m.ice_mass * m.latent_heat_fusion + m.total_mass * (m.final_temp - 0))

/-- Theorem stating that the original water mass in the mixture is 90.625g -/
theorem water_ice_mixture_theorem (m : WaterIceMixture) 
  (h1 : m.total_mass = 250)
  (h2 : m.water_mass_added = 1000)
  (h3 : m.initial_temp = 20)
  (h4 : m.final_temp = 5)
  (h5 : m.latent_heat_fusion = 80)
  (h6 : m.water_mass + m.ice_mass = m.total_mass)
  (h7 : heat_balance m = 0) :
  m.water_mass = 90.625 :=
sorry

end NUMINAMATH_CALUDE_water_ice_mixture_theorem_l643_64303


namespace NUMINAMATH_CALUDE_student_count_l643_64308

theorem student_count (total_eggs : ℕ) (eggs_per_student : ℕ) (h1 : total_eggs = 56) (h2 : eggs_per_student = 8) :
  total_eggs / eggs_per_student = 7 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l643_64308


namespace NUMINAMATH_CALUDE_other_focus_coordinates_l643_64336

/-- An ellipse with specific properties -/
structure Ellipse where
  /-- The ellipse is tangent to the y-axis at the origin -/
  tangent_at_origin : True
  /-- The length of the major axis -/
  major_axis_length : ℝ
  /-- The coordinates of one focus -/
  focus1 : ℝ × ℝ

/-- Theorem: Given an ellipse with specific properties, the other focus has coordinates (-3, -4) -/
theorem other_focus_coordinates (e : Ellipse) 
  (h1 : e.major_axis_length = 20)
  (h2 : e.focus1 = (3, 4)) :
  ∃ (other_focus : ℝ × ℝ), other_focus = (-3, -4) := by
  sorry

end NUMINAMATH_CALUDE_other_focus_coordinates_l643_64336


namespace NUMINAMATH_CALUDE_third_offense_percentage_increase_l643_64384

/-- Calculates the percentage increase for a third offense in a burglary case -/
theorem third_offense_percentage_increase
  (base_rate : ℚ)  -- Base sentence rate in years per $5000
  (stolen_value : ℚ)  -- Total value of stolen goods in dollars
  (additional_penalty : ℚ)  -- Additional penalty in years
  (total_sentence : ℚ)  -- Total sentence in years
  (h1 : base_rate = 1)  -- Base rate is 1 year per $5000
  (h2 : stolen_value = 40000)  -- $40,000 worth of goods stolen
  (h3 : additional_penalty = 2)  -- 2 years additional penalty
  (h4 : total_sentence = 12)  -- Total sentence is 12 years
  : (total_sentence - additional_penalty - (stolen_value / 5000 * base_rate)) / (stolen_value / 5000 * base_rate) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_third_offense_percentage_increase_l643_64384


namespace NUMINAMATH_CALUDE_remainder_two_power_33_mod_9_l643_64359

theorem remainder_two_power_33_mod_9 : 2^33 % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_two_power_33_mod_9_l643_64359


namespace NUMINAMATH_CALUDE_binary_matrix_sum_theorem_l643_64310

/-- A 5x5 matrix with entries 0 or 1 -/
def BinaryMatrix := Matrix (Fin 5) (Fin 5) Bool

/-- Get the 24 sequences from a BinaryMatrix as specified in the problem -/
def getSequences (X : BinaryMatrix) : Finset (List Bool) := sorry

/-- The sum of all entries in a BinaryMatrix -/
def matrixSum (X : BinaryMatrix) : ℕ := sorry

/-- Main theorem -/
theorem binary_matrix_sum_theorem (X : BinaryMatrix) :
  (getSequences X).card = 24 → matrixSum X = 12 ∨ matrixSum X = 13 := by sorry

end NUMINAMATH_CALUDE_binary_matrix_sum_theorem_l643_64310


namespace NUMINAMATH_CALUDE_squares_characterization_l643_64360

class MyGroup (G : Type) extends Group G where
  g : G
  h : G
  g_four : g ^ 4 = 1
  g_two_ne_one : g ^ 2 ≠ 1
  h_seven : h ^ 7 = 1
  h_ne_one : h ≠ 1
  gh_relation : g * h * g⁻¹ * h = 1
  subgroup_condition : ∀ (H : Subgroup G), g ∈ H → h ∈ H → H = ⊤

variable {G : Type} [MyGroup G]

def squares (G : Type) [MyGroup G] : Set G :=
  {x : G | ∃ y : G, y ^ 2 = x}

theorem squares_characterization :
  squares G = {1, (MyGroup.g : G) ^ 2, MyGroup.h, MyGroup.h ^ 2, MyGroup.h ^ 3, MyGroup.h ^ 4, MyGroup.h ^ 5, MyGroup.h ^ 6} := by
  sorry

end NUMINAMATH_CALUDE_squares_characterization_l643_64360


namespace NUMINAMATH_CALUDE_symbol_set_has_14_plus_l643_64301

/-- A set of symbols consisting of plus and minus signs -/
structure SymbolSet where
  total : ℕ
  plus : ℕ
  minus : ℕ
  sum_eq : total = plus + minus
  plus_constraint : ∀ (n : ℕ), n ≤ total - plus → n < 10
  minus_constraint : ∀ (n : ℕ), n ≤ total - minus → n < 15

/-- The theorem stating that a SymbolSet with 23 total symbols has 14 plus signs -/
theorem symbol_set_has_14_plus (s : SymbolSet) (h : s.total = 23) : s.plus = 14 := by
  sorry

end NUMINAMATH_CALUDE_symbol_set_has_14_plus_l643_64301


namespace NUMINAMATH_CALUDE_samias_walking_distance_l643_64374

/-- Represents the problem of calculating Samia's walking distance --/
theorem samias_walking_distance
  (total_time : ℝ)
  (bike_speed : ℝ)
  (walk_speed : ℝ)
  (wait_time : ℝ)
  (h_total_time : total_time = 1.25)  -- 1 hour and 15 minutes
  (h_bike_speed : bike_speed = 20)
  (h_walk_speed : walk_speed = 4)
  (h_wait_time : wait_time = 0.25)  -- 15 minutes
  : ∃ (total_distance : ℝ),
    let bike_distance := total_distance / 3
    let walk_distance := 2 * total_distance / 3
    bike_distance / bike_speed + wait_time + walk_distance / walk_speed = total_time ∧
    (walk_distance ≥ 3.55 ∧ walk_distance ≤ 3.65) :=
by
  sorry

#check samias_walking_distance

end NUMINAMATH_CALUDE_samias_walking_distance_l643_64374


namespace NUMINAMATH_CALUDE_integral_reciprocal_plus_x_l643_64388

theorem integral_reciprocal_plus_x : ∫ x in (1:ℝ)..(2:ℝ), (1/x + x) = Real.log 2 + 3/2 := by
  sorry

end NUMINAMATH_CALUDE_integral_reciprocal_plus_x_l643_64388


namespace NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_win_l643_64372

/-- Represents the voting structure in the giraffe beauty contest -/
structure VotingStructure where
  total_voters : Nat
  num_districts : Nat
  precincts_per_district : Nat
  voters_per_precinct : Nat

/-- Calculates the minimum number of voters required to win -/
def min_voters_to_win (vs : VotingStructure) : Nat :=
  let districts_to_win := (vs.num_districts + 1) / 2
  let precincts_to_win_per_district := (vs.precincts_per_district + 1) / 2
  let voters_to_win_per_precinct := (vs.voters_per_precinct + 1) / 2
  districts_to_win * precincts_to_win_per_district * voters_to_win_per_precinct

/-- The theorem stating the minimum number of voters required for the Tall giraffe to win -/
theorem min_voters_for_tall_giraffe_win (vs : VotingStructure) 
  (h1 : vs.total_voters = 135)
  (h2 : vs.num_districts = 5)
  (h3 : vs.precincts_per_district = 9)
  (h4 : vs.voters_per_precinct = 3)
  (h5 : vs.total_voters = vs.num_districts * vs.precincts_per_district * vs.voters_per_precinct) :
  min_voters_to_win vs = 30 := by
  sorry

#eval min_voters_to_win ⟨135, 5, 9, 3⟩

end NUMINAMATH_CALUDE_min_voters_for_tall_giraffe_win_l643_64372


namespace NUMINAMATH_CALUDE_total_households_l643_64304

/-- Represents the number of households in each category -/
structure HouseholdCounts where
  both : ℕ
  gasOnly : ℕ
  elecOnly : ℕ
  neither : ℕ

/-- The conditions of the survey -/
def surveyCounts : HouseholdCounts where
  both := 120
  gasOnly := 60
  elecOnly := 4 * 24
  neither := 24

/-- The theorem stating the total number of households surveyed -/
theorem total_households : 
  surveyCounts.both + surveyCounts.gasOnly + surveyCounts.elecOnly + surveyCounts.neither = 300 := by
  sorry


end NUMINAMATH_CALUDE_total_households_l643_64304


namespace NUMINAMATH_CALUDE_no_triangle_tangent_and_inscribed_l643_64332

/-- The problem statement as a theorem -/
theorem no_triangle_tangent_and_inscribed (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  let C₁ : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 1}
  let C₂ : Set (ℝ × ℝ) := {p | p.1^2 / a^2 + p.2^2 / b^2 = 1}
  (1 : ℝ)^2 / a^2 + (1 : ℝ)^2 / b^2 = 1 →
  ¬ ∃ (A B C : ℝ × ℝ),
    (A ∈ C₂ ∧ B ∈ C₂ ∧ C ∈ C₂) ∧
    (∀ p : ℝ × ℝ, p ∈ C₁ → (dist p A ≥ dist A B ∧ dist p B ≥ dist A B ∧ dist p C ≥ dist A B)) :=
by
  sorry


end NUMINAMATH_CALUDE_no_triangle_tangent_and_inscribed_l643_64332


namespace NUMINAMATH_CALUDE_leader_assistant_selection_l643_64307

theorem leader_assistant_selection (n : ℕ) (h : n = 8) : n * (n - 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_leader_assistant_selection_l643_64307


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l643_64323

/-- Given a triangle XYZ where the measure of ∠X is 78 degrees and 
    the measure of ∠Y is 14 degrees less than four times the measure of ∠Z,
    prove that the measure of ∠Z is 23.2 degrees. -/
theorem angle_measure_in_triangle (X Y Z : ℝ) 
  (h1 : X = 78)
  (h2 : Y = 4 * Z - 14)
  (h3 : X + Y + Z = 180) :
  Z = 23.2 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l643_64323


namespace NUMINAMATH_CALUDE_abigail_fence_count_l643_64378

/-- The number of fences Abigail has already built -/
def initial_fences : ℕ := 10

/-- The time it takes to build one fence, in minutes -/
def time_per_fence : ℕ := 30

/-- The number of additional hours Abigail will work -/
def additional_hours : ℕ := 8

/-- The number of minutes in an hour -/
def minutes_per_hour : ℕ := 60

/-- Calculate the total number of fences Abigail will have built -/
def total_fences : ℕ :=
  initial_fences + (additional_hours * (minutes_per_hour / time_per_fence))

theorem abigail_fence_count : total_fences = 26 := by
  sorry

end NUMINAMATH_CALUDE_abigail_fence_count_l643_64378


namespace NUMINAMATH_CALUDE_fraction_unchanged_l643_64389

theorem fraction_unchanged (x y : ℝ) (h : x ≠ y) :
  (3 * x) / (x - y) = (3 * (2 * x)) / ((2 * x) - (2 * y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_unchanged_l643_64389


namespace NUMINAMATH_CALUDE_first_grade_students_l643_64320

theorem first_grade_students (total : ℕ) (difference : ℕ) (first_grade : ℕ) : 
  total = 1256 → 
  difference = 408 →
  first_grade + difference = total - first_grade →
  first_grade = 424 := by
sorry

end NUMINAMATH_CALUDE_first_grade_students_l643_64320


namespace NUMINAMATH_CALUDE_deepak_age_l643_64391

theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 2 →
  rahul_age + 10 = 26 →
  deepak_age = 8 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l643_64391


namespace NUMINAMATH_CALUDE_fifteen_percent_of_number_l643_64371

theorem fifteen_percent_of_number (x : ℝ) : 12 = 0.15 * x → x = 80 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_percent_of_number_l643_64371


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l643_64325

/-- Represents a repeating decimal with a two-digit repetend -/
def RepeatingDecimal (a b : ℕ) : ℚ :=
  (10 * a + b : ℚ) / 99

theorem sum_of_repeating_decimals :
  RepeatingDecimal 25 + RepeatingDecimal 87 = 112 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l643_64325


namespace NUMINAMATH_CALUDE_transitivity_of_greater_than_l643_64390

theorem transitivity_of_greater_than {a b c : ℝ} (h1 : a > b) (h2 : b > c) : a > c := by
  sorry

end NUMINAMATH_CALUDE_transitivity_of_greater_than_l643_64390


namespace NUMINAMATH_CALUDE_set_equality_implies_sum_l643_64354

theorem set_equality_implies_sum (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a+b, 0} → a^2013 + b^2013 = -1 := by
  sorry

end NUMINAMATH_CALUDE_set_equality_implies_sum_l643_64354


namespace NUMINAMATH_CALUDE_non_negative_integer_solutions_of_inequality_l643_64311

theorem non_negative_integer_solutions_of_inequality : 
  {x : ℕ | -2 * (x : ℤ) > -4} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_non_negative_integer_solutions_of_inequality_l643_64311


namespace NUMINAMATH_CALUDE_jane_earnings_l643_64358

/-- Represents the number of bulbs planted for each flower type -/
structure BulbCounts where
  tulip : ℕ
  iris : ℕ
  hyacinth : ℕ
  daffodil : ℕ
  crocus : ℕ
  gladiolus : ℕ

/-- Represents the price per bulb for each flower type -/
structure BulbPrices where
  tulip : ℚ
  iris : ℚ
  hyacinth : ℚ
  daffodil : ℚ
  crocus : ℚ
  gladiolus : ℚ

def calculateEarnings (counts : BulbCounts) (prices : BulbPrices) : ℚ :=
  counts.tulip * prices.tulip +
  counts.iris * prices.iris +
  counts.hyacinth * prices.hyacinth +
  counts.daffodil * prices.daffodil +
  counts.crocus * prices.crocus +
  counts.gladiolus * prices.gladiolus

theorem jane_earnings (counts : BulbCounts) (prices : BulbPrices) :
  counts.tulip = 20 ∧
  counts.iris = counts.tulip / 2 ∧
  counts.hyacinth = counts.iris + counts.iris / 3 ∧
  counts.daffodil = 30 ∧
  counts.crocus = 3 * counts.daffodil ∧
  counts.gladiolus = 2 * (counts.crocus - counts.daffodil) + (15 * counts.daffodil / 100) ∧
  prices.tulip = 1/2 ∧
  prices.iris = 2/5 ∧
  prices.hyacinth = 3/4 ∧
  prices.daffodil = 1/4 ∧
  prices.crocus = 3/5 ∧
  prices.gladiolus = 3/10
  →
  calculateEarnings counts prices = 12245/100 := by
  sorry


end NUMINAMATH_CALUDE_jane_earnings_l643_64358


namespace NUMINAMATH_CALUDE_power_division_l643_64366

theorem power_division (x : ℝ) : x^3 / x^2 = x := by
  sorry

end NUMINAMATH_CALUDE_power_division_l643_64366


namespace NUMINAMATH_CALUDE_horner_method_f_2_l643_64376

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℚ) (x : ℚ) : ℚ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 3x^5 - 5x^4 + 3x^3 - 2x^2 + x -/
def f (x : ℚ) : ℚ :=
  horner [1, 0, -2, 3, -5, 3] x

theorem horner_method_f_2 :
  f 2 = 34 := by sorry

end NUMINAMATH_CALUDE_horner_method_f_2_l643_64376


namespace NUMINAMATH_CALUDE_x_value_l643_64330

theorem x_value : ∃ x : ℝ, (0.5 * x = 0.05 * 500 - 20) ∧ (x = 10) := by
  sorry

end NUMINAMATH_CALUDE_x_value_l643_64330


namespace NUMINAMATH_CALUDE_investment_timing_l643_64396

/-- Given two investments A and B, where A invests for the full year and B invests for part of the year,
    prove that B's investment starts 6 months after A's if their total investment-months are equal. -/
theorem investment_timing (a_amount : ℝ) (b_amount : ℝ) (total_months : ℕ) (x : ℝ) :
  a_amount > 0 →
  b_amount > 0 →
  total_months = 12 →
  a_amount * total_months = b_amount * (total_months - x) →
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_investment_timing_l643_64396


namespace NUMINAMATH_CALUDE_C₂_is_symmetric_to_C₁_l643_64352

/-- Circle C₁ with equation (x+1)²+(y-1)²=1 -/
def C₁ (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 1

/-- The line of symmetry x-y-1=0 -/
def symmetry_line (x y : ℝ) : Prop := x - y - 1 = 0

/-- The symmetric point of (x, y) with respect to the line x-y-1=0 -/
def symmetric_point (x y : ℝ) : ℝ × ℝ := (y + 1, x - 1)

/-- Circle C₂, symmetric to C₁ with respect to the line x-y-1=0 -/
def C₂ (x y : ℝ) : Prop := (x - 2)^2 + (y + 2)^2 = 1

/-- Theorem stating that C₂ is indeed symmetric to C₁ with respect to the given line -/
theorem C₂_is_symmetric_to_C₁ :
  ∀ x y : ℝ, C₂ x y ↔ C₁ (symmetric_point x y).1 (symmetric_point x y).2 :=
sorry

end NUMINAMATH_CALUDE_C₂_is_symmetric_to_C₁_l643_64352


namespace NUMINAMATH_CALUDE_perp_planes_sufficient_not_necessary_l643_64364

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and between a line and a plane
variable (perp_planes : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the relation of a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- State the theorem
theorem perp_planes_sufficient_not_necessary 
  (α β : Plane) (m : Line) 
  (h_m_in_α : line_in_plane m α) :
  (∀ α β m, perp_planes α β → line_in_plane m α → perp_line_plane m β) ∧ 
  (∃ α β m, line_in_plane m α ∧ perp_line_plane m β ∧ ¬perp_planes α β) :=
sorry

end NUMINAMATH_CALUDE_perp_planes_sufficient_not_necessary_l643_64364


namespace NUMINAMATH_CALUDE_equation_solutions_l643_64347

theorem equation_solutions : 
  (∃ x : ℝ, 2 * x + 62 = 248 ∧ x = 93) ∧
  (∃ x : ℝ, x - 12.7 = 2.7 ∧ x = 15.4) ∧
  (∃ x : ℝ, x / 5 = 0.16 ∧ x = 0.8) ∧
  (∃ x : ℝ, 7 * x + 2 * x = 6.3 ∧ x = 0.7) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l643_64347


namespace NUMINAMATH_CALUDE_original_denominator_proof_l643_64392

theorem original_denominator_proof (d : ℚ) : 
  (2 : ℚ) / d ≠ (1 : ℚ) / 2 ∧ (2 + 5 : ℚ) / (d + 5) = (1 : ℚ) / 2 → d = 9 := by
  sorry

end NUMINAMATH_CALUDE_original_denominator_proof_l643_64392


namespace NUMINAMATH_CALUDE_apex_angle_of_identical_cones_l643_64314

/-- The apex angle of a cone is the angle between its generatrices in the axial section. -/
def apex_angle (cone : Type) : ℝ := sorry

/-- A cone with apex at point A -/
structure Cone (A : Type) where
  apex : A
  angle : ℝ

/-- Three cones touch each other externally -/
def touch_externally (c1 c2 c3 : Cone A) : Prop := sorry

/-- A cone touches another cone internally -/
def touch_internally (c1 c2 : Cone A) : Prop := sorry

theorem apex_angle_of_identical_cones 
  (A : Type) 
  (c1 c2 c3 c4 : Cone A) 
  (h1 : touch_externally c1 c2 c3)
  (h2 : c1.angle = c2.angle)
  (h3 : c3.angle = π / 3)
  (h4 : touch_internally c1 c4)
  (h5 : touch_internally c2 c4)
  (h6 : touch_internally c3 c4)
  (h7 : c4.angle = 5 * π / 6) :
  c1.angle = 2 * Real.arctan (Real.sqrt 3 - 1) := by sorry

end NUMINAMATH_CALUDE_apex_angle_of_identical_cones_l643_64314


namespace NUMINAMATH_CALUDE_junior_score_l643_64399

theorem junior_score (total_students : ℕ) (junior_percentage senior_percentage : ℚ)
  (class_average senior_average : ℚ) (h1 : junior_percentage = 1/5)
  (h2 : senior_percentage = 4/5) (h3 : junior_percentage + senior_percentage = 1)
  (h4 : class_average = 85) (h5 : senior_average = 84) :
  let junior_count := (junior_percentage * total_students).num
  let senior_count := (senior_percentage * total_students).num
  let total_score := class_average * total_students
  let senior_total_score := senior_average * senior_count
  let junior_total_score := total_score - senior_total_score
  junior_total_score / junior_count = 89 := by
sorry


end NUMINAMATH_CALUDE_junior_score_l643_64399


namespace NUMINAMATH_CALUDE_mod_sum_powers_l643_64351

theorem mod_sum_powers (n : ℕ) : (44^1234 + 99^567) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_mod_sum_powers_l643_64351


namespace NUMINAMATH_CALUDE_slope_condition_l643_64355

-- Define the function
def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x

-- Define the theorem
theorem slope_condition (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (h1 : f m x₁ = y₁)
  (h2 : f m x₂ = y₂)
  (h3 : x₁ > x₂)
  (h4 : y₁ > y₂) :
  m > 2 := by
  sorry

end NUMINAMATH_CALUDE_slope_condition_l643_64355


namespace NUMINAMATH_CALUDE_probability_odd_divisor_15_factorial_l643_64398

theorem probability_odd_divisor_15_factorial (n : ℕ) (h : n = 15) :
  let factorial := n.factorial
  let total_divisors := (factorial.divisors.filter (λ x => x > 0)).card
  let odd_divisors := (factorial.divisors.filter (λ x => x > 0 ∧ x % 2 ≠ 0)).card
  (odd_divisors : ℚ) / total_divisors = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_probability_odd_divisor_15_factorial_l643_64398


namespace NUMINAMATH_CALUDE_infinite_series_equality_l643_64385

theorem infinite_series_equality (p q : ℝ) 
  (h : ∑' n, p / q^n = 5) :
  ∑' n, p / (p^2 + q)^n = 5 * (q - 1) / (25 * q^2 - 50 * q + 26) := by
  sorry

end NUMINAMATH_CALUDE_infinite_series_equality_l643_64385


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_three_l643_64370

theorem sum_of_fractions_equals_three (a b c x : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : (x - a - b) / c + (x - b - c) / a + (x - c - a) / b = 3) : 
  x = a + b + c := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_three_l643_64370


namespace NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_l643_64346

/-- The nth positive integer that is both odd and a multiple of 5 -/
def oddMultipleOf5 (n : ℕ) : ℕ := 10 * n - 5

/-- Prove that the 15th positive integer that is both odd and a multiple of 5 is 145 -/
theorem fifteenth_odd_multiple_of_5 : oddMultipleOf5 15 = 145 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_odd_multiple_of_5_l643_64346


namespace NUMINAMATH_CALUDE_subtraction_grouping_l643_64363

theorem subtraction_grouping (a b c d : ℝ) : a - b + c - d = a + c - (b + d) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_grouping_l643_64363


namespace NUMINAMATH_CALUDE_gold_coins_count_l643_64343

theorem gold_coins_count (gold_value : ℕ) (silver_value : ℕ) (silver_count : ℕ) (cash : ℕ) (total : ℕ) :
  gold_value = 50 →
  silver_value = 25 →
  silver_count = 5 →
  cash = 30 →
  total = 305 →
  ∃ (gold_count : ℕ), gold_count * gold_value + silver_count * silver_value + cash = total ∧ gold_count = 3 :=
by sorry

end NUMINAMATH_CALUDE_gold_coins_count_l643_64343


namespace NUMINAMATH_CALUDE_unique_teammate_d_score_l643_64326

-- Define the scoring system
def single_points : ℕ := 1
def double_points : ℕ := 2
def triple_points : ℕ := 3
def home_run_points : ℕ := 4

-- Define the total team score
def total_team_score : ℕ := 68

-- Define Faye's score
def faye_score : ℕ := 28

-- Define Teammate A's score components
def teammate_a_singles : ℕ := 1
def teammate_a_doubles : ℕ := 3
def teammate_a_home_runs : ℕ := 1

-- Define Teammate B's score components
def teammate_b_singles : ℕ := 4
def teammate_b_doubles : ℕ := 2
def teammate_b_triples : ℕ := 1

-- Define Teammate C's score components
def teammate_c_singles : ℕ := 2
def teammate_c_doubles : ℕ := 1
def teammate_c_triples : ℕ := 2
def teammate_c_home_runs : ℕ := 1

-- Theorem: There must be exactly one more player (Teammate D) who scored 4 points
theorem unique_teammate_d_score : 
  ∃! teammate_d_score : ℕ, 
    faye_score + 
    (teammate_a_singles * single_points + teammate_a_doubles * double_points + teammate_a_home_runs * home_run_points) +
    (teammate_b_singles * single_points + teammate_b_doubles * double_points + teammate_b_triples * triple_points) +
    (teammate_c_singles * single_points + teammate_c_doubles * double_points + teammate_c_triples * triple_points + teammate_c_home_runs * home_run_points) +
    teammate_d_score = total_team_score ∧ 
    teammate_d_score = 4 := by sorry

end NUMINAMATH_CALUDE_unique_teammate_d_score_l643_64326


namespace NUMINAMATH_CALUDE_unique_values_a_k_l643_64339

-- Define the sets A and B
def A (k : ℕ) : Set ℕ := {1, 2, 3, k}
def B (a : ℕ) : Set ℕ := {4, 7, a^4, a^2 + 3*a}

-- Define the function f
def f (x : ℕ) : ℕ := 3*x + 1

-- Theorem statement
theorem unique_values_a_k :
  ∃! (a k : ℕ), 
    a > 0 ∧ 
    (∀ x ∈ A k, ∃ y ∈ B a, f x = y) ∧ 
    (∀ y ∈ B a, ∃ x ∈ A k, f x = y) ∧
    a = 2 ∧ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_values_a_k_l643_64339


namespace NUMINAMATH_CALUDE_constant_function_l643_64394

theorem constant_function (α : ℝ) (x : ℝ) : 
  let f : ℝ → ℝ := λ x => Real.cos x ^ 2 + Real.cos (x + α) ^ 2 - 2 * Real.cos α * Real.cos x * Real.cos (x + α)
  f x = (1 - Real.cos (2 * α)) / 2 :=
by sorry

end NUMINAMATH_CALUDE_constant_function_l643_64394


namespace NUMINAMATH_CALUDE_arithmetic_sequence_proof_l643_64393

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_proof (k b : ℝ) :
  let a : ℕ → ℝ := λ n => k * n + b
  is_arithmetic_sequence a ∧ 
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d) ∧ 
  (∀ n : ℕ, a (n + 1) - a n = k) :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_proof_l643_64393


namespace NUMINAMATH_CALUDE_fifteen_degrees_to_radians_l643_64333

theorem fifteen_degrees_to_radians :
  ∀ (π : ℝ), 180 * (π / 12) = π → 15 * (π / 180) = π / 12 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_degrees_to_radians_l643_64333


namespace NUMINAMATH_CALUDE_square_side_increase_l643_64344

theorem square_side_increase (a : ℝ) (h : a > 0) : 
  let b := 2 * a
  let c := b * (1 + 80 / 100)
  c^2 = (a^2 + b^2) * (1 + 159.20000000000002 / 100) := by
  sorry

end NUMINAMATH_CALUDE_square_side_increase_l643_64344


namespace NUMINAMATH_CALUDE_right_triangle_cosine_l643_64387

theorem right_triangle_cosine (X Y Z : ℝ) (h1 : X = 90) (h2 : Real.sin Z = 4/5) : 
  Real.cos Z = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_cosine_l643_64387


namespace NUMINAMATH_CALUDE_fixed_point_of_logarithmic_function_l643_64334

theorem fixed_point_of_logarithmic_function (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 1 + Real.log x / Real.log a
  f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_logarithmic_function_l643_64334


namespace NUMINAMATH_CALUDE_sum_between_bounds_l643_64382

theorem sum_between_bounds : 
  (21/2 : ℚ) < (15/7 : ℚ) + (7/2 : ℚ) + (96/19 : ℚ) ∧ 
  (15/7 : ℚ) + (7/2 : ℚ) + (96/19 : ℚ) < 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_between_bounds_l643_64382


namespace NUMINAMATH_CALUDE_fruit_card_probability_l643_64316

theorem fruit_card_probability (total_cards : ℕ) (fruit_cards : ℕ) 
  (h1 : total_cards = 6)
  (h2 : fruit_cards = 2) :
  (fruit_cards : ℚ) / total_cards = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fruit_card_probability_l643_64316


namespace NUMINAMATH_CALUDE_geometric_sum_first_seven_l643_64381

-- Define the geometric sequence
def geometric_sequence (a₀ : ℚ) (r : ℚ) (n : ℕ) : ℚ := a₀ * r^n

-- Define the sum of the first n terms of the geometric sequence
def geometric_sum (a₀ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  if r = 1 then n * a₀ else a₀ * (1 - r^n) / (1 - r)

theorem geometric_sum_first_seven :
  let a₀ : ℚ := 1/3
  let r : ℚ := 1/3
  let n : ℕ := 7
  geometric_sum a₀ r n = 1093/2187 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sum_first_seven_l643_64381


namespace NUMINAMATH_CALUDE_twenty_apples_fourteen_cucumbers_l643_64365

/-- Represents the cost of a single apple -/
def apple_cost : ℝ := sorry

/-- Represents the cost of a single banana -/
def banana_cost : ℝ := sorry

/-- Represents the cost of a single cucumber -/
def cucumber_cost : ℝ := sorry

/-- The cost of 10 apples equals the cost of 5 bananas -/
axiom ten_apples_five_bananas : 10 * apple_cost = 5 * banana_cost

/-- The cost of 5 bananas equals the cost of 7 cucumbers -/
axiom five_bananas_seven_cucumbers : 5 * banana_cost = 7 * cucumber_cost

/-- Theorem: The cost of 20 apples equals the cost of 14 cucumbers -/
theorem twenty_apples_fourteen_cucumbers : 20 * apple_cost = 14 * cucumber_cost := by
  sorry

end NUMINAMATH_CALUDE_twenty_apples_fourteen_cucumbers_l643_64365


namespace NUMINAMATH_CALUDE_negation_of_implication_l643_64322

theorem negation_of_implication (x y : ℝ) :
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l643_64322


namespace NUMINAMATH_CALUDE_trevor_ride_cost_l643_64397

/-- The total cost of Trevor's taxi ride downtown including the tip -/
def total_cost (uber_cost lyft_cost taxi_cost : ℚ) : ℚ :=
  taxi_cost + 0.2 * taxi_cost

theorem trevor_ride_cost :
  ∀ (uber_cost lyft_cost taxi_cost : ℚ),
    uber_cost = lyft_cost + 3 →
    lyft_cost = taxi_cost + 4 →
    uber_cost = 22 →
    total_cost uber_cost lyft_cost taxi_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_trevor_ride_cost_l643_64397


namespace NUMINAMATH_CALUDE_second_round_votes_l643_64380

/-- Represents the total number of votes in the second round of an election. -/
def total_votes : ℕ := sorry

/-- Represents the percentage of votes received by Candidate A in the second round. -/
def candidate_a_percentage : ℚ := 50 / 100

/-- Represents the percentage of votes received by Candidate B in the second round. -/
def candidate_b_percentage : ℚ := 30 / 100

/-- Represents the percentage of votes received by Candidate C in the second round. -/
def candidate_c_percentage : ℚ := 20 / 100

/-- Represents the majority of votes by which Candidate A won over Candidate B. -/
def majority : ℕ := 1350

theorem second_round_votes : 
  (candidate_a_percentage - candidate_b_percentage) * total_votes = majority ∧
  total_votes = 6750 := by sorry

end NUMINAMATH_CALUDE_second_round_votes_l643_64380


namespace NUMINAMATH_CALUDE_sum_of_2001_numbers_positive_l643_64350

theorem sum_of_2001_numbers_positive 
  (a : Fin 2001 → ℝ) 
  (h : ∀ (i j k l : Fin 2001), i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l → 
    a i + a j + a k + a l > 0) : 
  Finset.sum Finset.univ a > 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_2001_numbers_positive_l643_64350


namespace NUMINAMATH_CALUDE_donald_oranges_l643_64367

def final_oranges (initial found given_away : ℕ) : ℕ :=
  initial + found - given_away

theorem donald_oranges : 
  final_oranges 4 5 3 = 6 := by sorry

end NUMINAMATH_CALUDE_donald_oranges_l643_64367


namespace NUMINAMATH_CALUDE_sum_equals_three_or_seven_l643_64313

theorem sum_equals_three_or_seven (x y z : ℝ) 
  (eq1 : x + y / z = 2)
  (eq2 : y + z / x = 2)
  (eq3 : z + x / y = 2) :
  x + y + z = 3 ∨ x + y + z = 7 := by
sorry

end NUMINAMATH_CALUDE_sum_equals_three_or_seven_l643_64313


namespace NUMINAMATH_CALUDE_solve_iterated_f_equation_l643_64324

def f (x : ℝ) : ℝ := x^2 + 6*x + 6

def iterate_f (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n+1 => f (iterate_f n x)

theorem solve_iterated_f_equation :
  ∃ x : ℝ, iterate_f 2017 x = 2017 ∧
  x = -3 + (2020 : ℝ)^(1/(2^2017)) ∨
  x = -3 - (2020 : ℝ)^(1/(2^2017)) :=
sorry

end NUMINAMATH_CALUDE_solve_iterated_f_equation_l643_64324


namespace NUMINAMATH_CALUDE_inverse_97_mod_98_l643_64319

theorem inverse_97_mod_98 : ∃ x : ℕ, x ≥ 0 ∧ x ≤ 97 ∧ (97 * x) % 98 = 1 := by
  sorry

end NUMINAMATH_CALUDE_inverse_97_mod_98_l643_64319


namespace NUMINAMATH_CALUDE_moon_permutations_eq_12_l643_64373

/-- The number of distinct permutations of the letters in "MOON" -/
def moon_permutations : ℕ :=
  Nat.factorial 4 / Nat.factorial 2

theorem moon_permutations_eq_12 : moon_permutations = 12 := by
  sorry

end NUMINAMATH_CALUDE_moon_permutations_eq_12_l643_64373


namespace NUMINAMATH_CALUDE_product_equals_square_l643_64375

theorem product_equals_square : 500 * 49.95 * 4.995 * 5000 = (24975 : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_product_equals_square_l643_64375


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l643_64337

/-- A circle in the 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point lies on a line -/
def Line.contains (l : Line) (p : ℝ × ℝ) : Prop :=
  l.a * p.1 + l.b * p.2 + l.c = 0

/-- Check if a line is tangent to a circle -/
def is_tangent (l : Line) (c : Circle) : Prop :=
  let (x₀, y₀) := c.center
  (l.a * x₀ + l.b * y₀ + l.c)^2 = (l.a^2 + l.b^2) * c.radius^2

theorem tangent_line_to_circle (c : Circle) (l : Line) (p : ℝ × ℝ) :
  c.center = (0, 0) ∧ c.radius = 5 ∧
  l.a = 3 ∧ l.b = 4 ∧ l.c = -25 ∧
  p = (3, 4) →
  is_tangent l c ∧ l.contains p :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l643_64337


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l643_64328

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 2 * Complex.I) :
  z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l643_64328


namespace NUMINAMATH_CALUDE_no_general_rational_solution_l643_64331

theorem no_general_rational_solution (k : ℚ) : 
  ¬ ∃ (S : Set ℝ), ∀ (x : ℝ), x ∈ S → 
    ∃ (q : ℚ), x + k * Real.sqrt (x^2 + 1) - 1 / (x + k * Real.sqrt (x^2 + 1)) = q :=
by sorry

end NUMINAMATH_CALUDE_no_general_rational_solution_l643_64331


namespace NUMINAMATH_CALUDE_blended_tea_selling_price_l643_64302

/-- Calculates the selling price of a blended tea variety -/
theorem blended_tea_selling_price
  (cost1 : ℝ) (cost2 : ℝ) (ratio1 : ℝ) (ratio2 : ℝ) (gain_percent : ℝ)
  (h1 : cost1 = 18)
  (h2 : cost2 = 20)
  (h3 : ratio1 = 5)
  (h4 : ratio2 = 3)
  (h5 : gain_percent = 12)
  : (cost1 * ratio1 + cost2 * ratio2) / (ratio1 + ratio2) * (1 + gain_percent / 100) = 21 := by
  sorry

#check blended_tea_selling_price

end NUMINAMATH_CALUDE_blended_tea_selling_price_l643_64302


namespace NUMINAMATH_CALUDE_expression_value_l643_64342

theorem expression_value (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 1)
  (h2 : a / x + b / y + c / z = 0)
  : x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l643_64342


namespace NUMINAMATH_CALUDE_digits_of_product_l643_64315

theorem digits_of_product : ∃ n : ℕ, n = 3^4 * 6^8 ∧ (Nat.log 10 n).succ = 9 := by sorry

end NUMINAMATH_CALUDE_digits_of_product_l643_64315


namespace NUMINAMATH_CALUDE_symmetric_points_ab_power_l643_64321

/-- Given two points M(2a, 2) and N(-8, a+b) that are symmetric with respect to the y-axis,
    prove that a^b = 1/16 -/
theorem symmetric_points_ab_power (a b : ℝ) : 
  (∃ (M N : ℝ × ℝ), 
    M = (2*a, 2) ∧ 
    N = (-8, a+b) ∧ 
    (M.1 = -N.1) ∧  -- x-coordinates are opposite
    (M.2 = N.2))    -- y-coordinates are equal
  → a^b = 1/16 := by
sorry

end NUMINAMATH_CALUDE_symmetric_points_ab_power_l643_64321


namespace NUMINAMATH_CALUDE_jacoby_lottery_winnings_l643_64305

theorem jacoby_lottery_winnings :
  let trip_cost : ℕ := 5000
  let hourly_wage : ℕ := 20
  let hours_worked : ℕ := 10
  let cookie_price : ℕ := 4
  let cookies_sold : ℕ := 24
  let lottery_ticket_cost : ℕ := 10
  let remaining_needed : ℕ := 3214
  let sister_gift : ℕ := 500
  let num_sisters : ℕ := 2

  let job_earnings := hourly_wage * hours_worked
  let cookie_earnings := cookie_price * cookies_sold
  let total_earnings := job_earnings + cookie_earnings - lottery_ticket_cost
  let total_gifts := sister_gift * num_sisters
  let current_funds := total_earnings + total_gifts
  let lottery_winnings := trip_cost - remaining_needed - current_funds

  lottery_winnings = 500 := by
    sorry

end NUMINAMATH_CALUDE_jacoby_lottery_winnings_l643_64305


namespace NUMINAMATH_CALUDE_sasha_sticker_problem_l643_64353

theorem sasha_sticker_problem (m n : ℕ) (t : ℝ) : 
  0 < m ∧ m < n ∧ 1 < t ∧ 
  m * t + n = 100 ∧ 
  m + n * t = 101 → 
  n = 34 ∨ n = 66 := by
sorry

end NUMINAMATH_CALUDE_sasha_sticker_problem_l643_64353


namespace NUMINAMATH_CALUDE_floor_ceiling_sum_l643_64327

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.7 : ℝ)⌉ = 31 := by sorry

end NUMINAMATH_CALUDE_floor_ceiling_sum_l643_64327


namespace NUMINAMATH_CALUDE_cubic_roots_sum_squares_l643_64329

theorem cubic_roots_sum_squares (a b c : ℝ) : 
  (3 * a^3 - 4 * a^2 + 100 * a - 3 = 0) →
  (3 * b^3 - 4 * b^2 + 100 * b - 3 = 0) →
  (3 * c^3 - 4 * c^2 + 100 * c - 3 = 0) →
  (a + b + 2)^2 + (b + c + 2)^2 + (c + a + 2)^2 = 1079/9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_squares_l643_64329


namespace NUMINAMATH_CALUDE_inequality_proof_l643_64349

theorem inequality_proof (a b : ℝ) (n : ℤ) (ha : a > 0) (hb : b > 0) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l643_64349


namespace NUMINAMATH_CALUDE_unique_valid_number_l643_64379

def shares_one_digit (n m : Nat) : Bool :=
  let n_digits := n.digits 10
  let m_digits := m.digits 10
  (n_digits.filter (fun d => m_digits.contains d)).length = 1

def is_valid_number (n : Nat) : Bool :=
  n ≥ 100 ∧ n < 1000 ∧
  shares_one_digit n 543 ∧
  shares_one_digit n 142 ∧
  shares_one_digit n 562

theorem unique_valid_number : 
  ∀ n : Nat, is_valid_number n ↔ n = 163 :=
sorry

end NUMINAMATH_CALUDE_unique_valid_number_l643_64379
