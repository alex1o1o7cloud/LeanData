import Mathlib

namespace marys_investment_l802_80288

theorem marys_investment (mary_investment : ℕ) (mike_investment : ℕ) (total_profit : ℕ) : 
  mike_investment = 400 →
  total_profit = 7500 →
  let equal_share := total_profit / 3 / 2
  let remaining_profit := total_profit - total_profit / 3
  let mary_share := equal_share + remaining_profit * mary_investment / (mary_investment + mike_investment)
  let mike_share := equal_share + remaining_profit * mike_investment / (mary_investment + mike_investment)
  mary_share = mike_share + 1000 →
  mary_investment = 600 :=
by sorry

end marys_investment_l802_80288


namespace max_y_value_l802_80202

theorem max_y_value (x y : ℤ) (h : x * y + 7 * x + 6 * y = -14) : y ≤ 21 := by
  sorry

end max_y_value_l802_80202


namespace inequality_proof_l802_80235

theorem inequality_proof (a b x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  (x / (a*y + b*z)) + (y / (a*z + b*x)) + (z / (a*x + b*y)) ≥ 3 / (a + b) := by
  sorry

end inequality_proof_l802_80235


namespace nonagon_triangles_l802_80250

/-- The number of triangles formed by vertices of a regular nonagon -/
def triangles_in_nonagon : ℕ := Nat.choose 9 3

/-- Theorem stating that the number of triangles in a regular nonagon is 84 -/
theorem nonagon_triangles : triangles_in_nonagon = 84 := by
  sorry

end nonagon_triangles_l802_80250


namespace chess_tournament_points_inequality_l802_80201

theorem chess_tournament_points_inequality (boys girls : ℕ) (boys_points girls_points : ℚ) : 
  boys = 9 → 
  girls = 3 → 
  boys_points = 36 + (9 * 3 - boys_points) → 
  girls_points = 3 + (9 * 3 - girls_points) → 
  boys_points ≠ girls_points :=
by sorry

end chess_tournament_points_inequality_l802_80201


namespace one_fourth_x_equals_nine_l802_80269

theorem one_fourth_x_equals_nine (x : ℝ) (h : (1 / 3) * x = 12) : (1 / 4) * x = 9 := by
  sorry

end one_fourth_x_equals_nine_l802_80269


namespace divisibility_theorem_l802_80254

theorem divisibility_theorem (a b c x y z : ℝ) :
  (a * y - b * x)^2 + (b * z - c * y)^2 + (c * x - a * z)^2 + (a * x + b * y + c * z)^2 =
  (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) := by
  sorry

end divisibility_theorem_l802_80254


namespace function_inequality_l802_80240

-- Define the function F
noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f x / Real.exp x

-- State the theorem
theorem function_inequality
  (f : ℝ → ℝ)
  (f_diff : Differentiable ℝ f)
  (h : ∀ x, deriv f x < f x) :
  f 2 < Real.exp 2 * f 0 ∧ f 2012 < Real.exp 2012 * f 0 :=
sorry

end function_inequality_l802_80240


namespace curtis_farm_chickens_l802_80229

/-- The number of chickens on Mr. Curtis's farm -/
theorem curtis_farm_chickens :
  let roosters : ℕ := 28
  let non_egg_laying_hens : ℕ := 20
  let egg_laying_hens : ℕ := 277
  roosters + non_egg_laying_hens + egg_laying_hens = 325 :=
by sorry

end curtis_farm_chickens_l802_80229


namespace sum_of_absolute_coefficients_l802_80233

theorem sum_of_absolute_coefficients (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - 3*x)^9 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  |a| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| + |a₉| = 4^9 :=
by
  sorry

end sum_of_absolute_coefficients_l802_80233


namespace abigail_fence_count_l802_80249

/-- The number of fences Abigail builds in total -/
def total_fences (initial_fences : ℕ) (build_time_per_fence : ℕ) (additional_hours : ℕ) : ℕ :=
  initial_fences + (60 / build_time_per_fence) * additional_hours

theorem abigail_fence_count :
  total_fences 10 30 8 = 26 := by
  sorry

end abigail_fence_count_l802_80249


namespace storage_box_length_l802_80292

/-- Calculates the length of cubic storage boxes given total volume, cost per box, and total cost -/
theorem storage_box_length (total_volume : ℝ) (cost_per_box : ℝ) (total_cost : ℝ) :
  total_volume = 1080000 ∧ cost_per_box = 0.5 ∧ total_cost = 300 →
  ∃ (length : ℝ), abs (length - (total_volume / (total_cost / cost_per_box))^(1/3)) < 0.1 := by
sorry

#eval (1080000 / (300 / 0.5))^(1/3)

end storage_box_length_l802_80292


namespace at_least_ten_same_weight_l802_80256

/-- Represents the weight of a coin as measured by the scale -/
structure MeasuredWeight where
  value : ℝ
  is_valid : value > 11

/-- Represents the actual weight of a coin -/
structure ActualWeight where
  value : ℝ
  is_valid : value > 10

/-- The scale's measurement is always off by exactly 1 gram -/
def scale_error (actual : ActualWeight) (measured : MeasuredWeight) : Prop :=
  (measured.value = actual.value + 1) ∨ (measured.value = actual.value - 1)

/-- A collection of 12 coin measurements -/
def CoinMeasurements := Fin 12 → MeasuredWeight

/-- The actual weights corresponding to the measurements -/
def ActualWeights := Fin 12 → ActualWeight

theorem at_least_ten_same_weight 
  (measurements : CoinMeasurements) 
  (actual_weights : ActualWeights) 
  (h : ∀ i, scale_error (actual_weights i) (measurements i)) :
  ∃ (w : ℝ) (s : Finset (Fin 12)), s.card ≥ 10 ∧ ∀ i ∈ s, (actual_weights i).value = w :=
sorry

end at_least_ten_same_weight_l802_80256


namespace trig_simplification_l802_80285

theorem trig_simplification (x y : ℝ) :
  Real.sin x ^ 2 + Real.sin (x + y) ^ 2 - 2 * Real.sin x * Real.sin y * Real.sin (x + y) = Real.sin y ^ 2 := by
  sorry

end trig_simplification_l802_80285


namespace abc_inequality_l802_80287

theorem abc_inequality (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0) (h_prod : a * b * c = 1000)
  (h_sum : b * c * (1 - a) + a * (b + c) = 110) (h_a_lt_1 : a < 1) :
  10 < c ∧ c < 100 := by
  sorry

end abc_inequality_l802_80287


namespace right_triangle_partition_l802_80216

-- Define the set of points on the sides of an equilateral triangle
def TrianglePoints : Type := Set (ℝ × ℝ)

-- Define a property that a set of points contains a right triangle
def ContainsRightTriangle (S : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ × ℝ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧
    (a.1 - b.1) * (a.1 - c.1) + (a.2 - b.2) * (a.2 - c.2) = 0

-- State the theorem
theorem right_triangle_partition (T : TrianglePoints) :
  ∀ (S₁ S₂ : Set (ℝ × ℝ)), S₁ ∪ S₂ = T ∧ S₁ ∩ S₂ = ∅ →
    ContainsRightTriangle S₁ ∨ ContainsRightTriangle S₂ :=
sorry

end right_triangle_partition_l802_80216


namespace randy_theorem_l802_80245

def randy_problem (initial_amount : ℕ) (smith_contribution : ℕ) (sally_gift : ℕ) : Prop :=
  let total := initial_amount + smith_contribution
  let remaining := total - sally_gift
  remaining = 2000

theorem randy_theorem : randy_problem 3000 200 1200 := by
  sorry

end randy_theorem_l802_80245


namespace quotient_problem_l802_80278

theorem quotient_problem (dividend : ℕ) (divisor : ℕ) (remainder : ℕ) (quotient : ℕ) 
  (h1 : dividend = 139)
  (h2 : divisor = 19)
  (h3 : remainder = 6)
  (h4 : dividend = divisor * quotient + remainder) :
  quotient = 7 := by
  sorry

end quotient_problem_l802_80278


namespace cos_330_degrees_l802_80282

theorem cos_330_degrees : Real.cos (330 * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_degrees_l802_80282


namespace perimeter_of_figure_C_l802_80214

/-- Given a large rectangle composed of 20 identical small rectangles,
    prove that the perimeter of figure C is 40 cm given the perimeters of figures A and B. -/
theorem perimeter_of_figure_C (x y : ℝ) : 
  (x > 0) → 
  (y > 0) → 
  (6 * x + 2 * y = 56) →  -- Perimeter of figure A
  (4 * x + 6 * y = 56) →  -- Perimeter of figure B
  (2 * x + 6 * y = 40)    -- Perimeter of figure C
  := by sorry

end perimeter_of_figure_C_l802_80214


namespace remainder_71_73_div_9_l802_80274

theorem remainder_71_73_div_9 : (71 * 73) % 9 = 8 := by
  sorry

end remainder_71_73_div_9_l802_80274


namespace genetic_events_in_both_divisions_l802_80230

-- Define cell division processes
inductive CellDivision
| mitosis
| meiosis

-- Define genetic events
inductive GeneticEvent
| mutation
| chromosomalVariation

-- Define cellular processes during division
structure CellularProcess where
  chromosomeReplication : Bool
  centromereSplitting : Bool

-- Define the occurrence of genetic events during cell division
def geneticEventOccurs (event : GeneticEvent) (division : CellDivision) : Prop :=
  ∃ (process : CellularProcess), 
    process.chromosomeReplication ∧ 
    process.centromereSplitting

-- Theorem statement
theorem genetic_events_in_both_divisions :
  (∀ (event : GeneticEvent) (division : CellDivision), 
    geneticEventOccurs event division) :=
sorry

end genetic_events_in_both_divisions_l802_80230


namespace perimeter_marbles_12_l802_80257

/-- A square made of marbles -/
structure MarbleSquare where
  side_length : ℕ
  
/-- The number of marbles on the perimeter of a square -/
def perimeter_marbles (square : MarbleSquare) : ℕ :=
  4 * square.side_length - 4

theorem perimeter_marbles_12 :
  ∀ (square : MarbleSquare),
    square.side_length = 12 →
    perimeter_marbles square = 44 := by
  sorry

end perimeter_marbles_12_l802_80257


namespace printer_price_ratio_l802_80236

/-- Given the price of a basic computer and printer setup, prove the ratio of the printer price
    to the total price of an enhanced computer and printer setup. -/
theorem printer_price_ratio (basic_computer_price printer_price enhanced_computer_price : ℕ) : 
  basic_computer_price + printer_price = 2500 →
  enhanced_computer_price = basic_computer_price + 500 →
  basic_computer_price = 2125 →
  printer_price / (enhanced_computer_price + printer_price) = 1 / 8 := by
  sorry

#check printer_price_ratio

end printer_price_ratio_l802_80236


namespace train_late_speed_l802_80211

/-- Proves that the late average speed is 35 kmph given the conditions of the train problem -/
theorem train_late_speed (distance : ℝ) (on_time_speed : ℝ) (late_time : ℝ) :
  distance = 70 →
  on_time_speed = 40 →
  late_time = (distance / on_time_speed) + (15 / 60) →
  ∃ (late_speed : ℝ), late_speed = distance / late_time ∧ late_speed = 35 := by
  sorry

end train_late_speed_l802_80211


namespace arithmetic_sequence_general_term_l802_80264

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_general_term
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : a 1 = 2)
  (h3 : ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  ∀ n : ℕ, a n = 2 * n :=
sorry

end arithmetic_sequence_general_term_l802_80264


namespace quadratic_root_reciprocal_relation_l802_80271

/-- Given two quadratic equations ax² + bx + c = 0 and cx² + bx + a = 0,
    this theorem states that the roots of the second equation
    are the reciprocals of the roots of the first equation. -/
theorem quadratic_root_reciprocal_relation (a b c : ℝ) (x₁ x₂ : ℝ) :
  (a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) →
  (c * (1/x₁)^2 + b * (1/x₁) + a = 0 ∧ c * (1/x₂)^2 + b * (1/x₂) + a = 0) :=
by sorry

end quadratic_root_reciprocal_relation_l802_80271


namespace quadrilateral_with_parallel_sides_and_congruent_diagonals_is_rectangle_l802_80296

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define properties of a quadrilateral
def has_parallel_sides (q : Quadrilateral) : Prop := 
  sorry

def has_congruent_diagonals (q : Quadrilateral) : Prop := 
  sorry

def is_rectangle (q : Quadrilateral) : Prop := 
  sorry

-- Theorem statement
theorem quadrilateral_with_parallel_sides_and_congruent_diagonals_is_rectangle 
  (q : Quadrilateral) : 
  has_parallel_sides q → has_congruent_diagonals q → is_rectangle q :=
sorry

end quadrilateral_with_parallel_sides_and_congruent_diagonals_is_rectangle_l802_80296


namespace team_C_most_uniform_l802_80206

-- Define the teams
inductive Team : Type
| A : Team
| B : Team
| C : Team
| D : Team

-- Define the variance for each team
def variance : Team → ℝ
| Team.A => 0.13
| Team.B => 0.11
| Team.C => 0.09
| Team.D => 0.15

-- Define a function to determine if a team has the most uniform height
def has_most_uniform_height (t : Team) : Prop :=
  ∀ other : Team, variance t ≤ variance other

-- Theorem: Team C has the most uniform height
theorem team_C_most_uniform : has_most_uniform_height Team.C := by
  sorry


end team_C_most_uniform_l802_80206


namespace origami_distribution_l802_80239

theorem origami_distribution (total_papers : ℕ) (num_cousins : ℕ) (papers_per_cousin : ℕ) : 
  total_papers = 48 → 
  num_cousins = 6 → 
  total_papers = num_cousins * papers_per_cousin → 
  papers_per_cousin = 8 := by
sorry

end origami_distribution_l802_80239


namespace radical_simplification_l802_80237

theorem radical_simplification (q : ℝ) (hq : q ≥ 0) :
  Real.sqrt (15 * q) * Real.sqrt (10 * q^3) * Real.sqrt (14 * q^5) = 10 * q^4 * Real.sqrt (21 * q) :=
by sorry

end radical_simplification_l802_80237


namespace abs_sum_inequalities_l802_80277

theorem abs_sum_inequalities (a b : ℝ) (h : a * b > 0) : 
  (abs (a + b) > abs a) ∧ (abs (a + b) > abs (a - b)) := by sorry

end abs_sum_inequalities_l802_80277


namespace min_value_of_product_l802_80260

theorem min_value_of_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a/b + b/c + c/a + b/a + c/b + a/c = 10) :
  (a/b + b/c + c/a) * (b/a + c/b + a/c) ≥ 47 :=
sorry

end min_value_of_product_l802_80260


namespace inequality_equivalent_to_interval_l802_80222

-- Define the inequality
def inequality (x : ℝ) : Prop := |8 - x| / 4 < 3

-- Define the interval
def interval (x : ℝ) : Prop := -4 < x ∧ x < 20

-- Theorem statement
theorem inequality_equivalent_to_interval :
  ∀ x : ℝ, inequality x ↔ interval x :=
by sorry

end inequality_equivalent_to_interval_l802_80222


namespace inequality_solution_set_l802_80279

theorem inequality_solution_set :
  ∀ x : ℝ, abs (2*x - 1) - abs (x - 2) < 0 ↔ -1 < x ∧ x < 1 :=
sorry

end inequality_solution_set_l802_80279


namespace commutative_property_demonstration_l802_80258

theorem commutative_property_demonstration :
  (2 + 1 + 5 - 1 = 2 - 1 + 1 + 5) →
  ∃ (a b c d : ℤ), a + b + c + d = b + c + d + a :=
by sorry

end commutative_property_demonstration_l802_80258


namespace sin_sum_equality_l802_80248

theorem sin_sum_equality : 
  Real.sin (30 * π / 180) * Real.sin (75 * π / 180) + 
  Real.sin (60 * π / 180) * Real.sin (15 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end sin_sum_equality_l802_80248


namespace jack_book_loss_l802_80234

/-- Calculates the amount of money Jack lost in a year buying and selling books. -/
theorem jack_book_loss (books_per_month : ℕ) (book_cost : ℕ) (selling_price : ℕ) (months_per_year : ℕ) : 
  books_per_month = 3 →
  book_cost = 20 →
  selling_price = 500 →
  months_per_year = 12 →
  (books_per_month * months_per_year * book_cost) - selling_price = 220 := by
sorry

end jack_book_loss_l802_80234


namespace existence_of_basis_vectors_l802_80286

-- Define the set of points
variable (n : ℕ)
variable (O : ℝ × ℝ)
variable (A : Fin n → ℝ × ℝ)

-- Define the distance condition
variable (h : ∀ (i j : Fin n), ∃ (m : ℕ), ‖A i - A j‖ = Real.sqrt m)
variable (h' : ∀ (i : Fin n), ∃ (m : ℕ), ‖A i - O‖ = Real.sqrt m)

-- The theorem to be proved
theorem existence_of_basis_vectors :
  ∃ (x y : ℝ × ℝ), ∀ (i : Fin n), ∃ (k l : ℤ), A i - O = k • x + l • y :=
sorry

end existence_of_basis_vectors_l802_80286


namespace four_groups_four_spots_l802_80210

/-- The number of ways to arrange tour groups among scenic spots with one spot unvisited -/
def tourArrangements (numGroups numSpots : ℕ) : ℕ :=
  (numGroups.choose 2) * (numSpots.factorial / (numSpots - 3).factorial)

/-- Theorem stating the number of arrangements for 4 groups and 4 spots -/
theorem four_groups_four_spots :
  tourArrangements 4 4 = 144 := by
  sorry

end four_groups_four_spots_l802_80210


namespace min_m_plus_n_l802_80293

noncomputable def f (m n x : ℝ) : ℝ := Real.log x - 2 * m * x^2 - n

theorem min_m_plus_n (m n : ℝ) :
  (∀ x > 0, f m n x ≤ -Real.log 2) →
  (∃ x > 0, f m n x = -Real.log 2) →
  m + n ≥ (1/2) * Real.log 2 :=
by sorry

end min_m_plus_n_l802_80293


namespace ratio_equality_l802_80267

theorem ratio_equality (a b c d : ℚ) 
  (h1 : a = 4 * b) 
  (h2 : b = 5 * c) 
  (h3 : c = 3 * d) : 
  a * d / (b * c) = 4 / 3 := by
sorry

end ratio_equality_l802_80267


namespace b_power_a_equals_negative_one_l802_80290

theorem b_power_a_equals_negative_one (a b : ℝ) : 
  (a - 5)^2 + |2*b + 2| = 0 → b^a = -1 := by sorry

end b_power_a_equals_negative_one_l802_80290


namespace game_points_sum_l802_80252

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allieRolls : List ℕ := [6, 3, 2, 4]
def charlieRolls : List ℕ := [5, 3, 1, 6]

theorem game_points_sum : 
  (List.sum (List.map g allieRolls)) + (List.sum (List.map g charlieRolls)) = 38 := by
  sorry

end game_points_sum_l802_80252


namespace cards_found_l802_80200

def initial_cards : ℕ := 7
def final_cards : ℕ := 54

theorem cards_found (initial : ℕ) (final : ℕ) (h1 : initial = initial_cards) (h2 : final = final_cards) :
  final - initial = 47 := by sorry

end cards_found_l802_80200


namespace gcd_490_910_l802_80243

theorem gcd_490_910 : Nat.gcd 490 910 = 70 := by
  sorry

end gcd_490_910_l802_80243


namespace inequality_theorem_l802_80299

theorem inequality_theorem (a b c d : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
  (h_product : a * b * c * d = 1) :
  1 / (1 + a)^2 + 1 / (1 + b)^2 + 1 / (1 + c)^2 + 1 / (1 + d)^2 ≥ 1 := by
  sorry

end inequality_theorem_l802_80299


namespace sum_with_abs_zero_implies_triple_l802_80253

theorem sum_with_abs_zero_implies_triple (a : ℝ) : a + |a| = 0 → a - |2*a| = 3*a := by
  sorry

end sum_with_abs_zero_implies_triple_l802_80253


namespace total_campers_is_71_l802_80204

/-- The total number of campers who went rowing and hiking -/
def total_campers (morning_rowing : ℕ) (morning_hiking : ℕ) (afternoon_rowing : ℕ) : ℕ :=
  morning_rowing + morning_hiking + afternoon_rowing

/-- Theorem stating that the total number of campers who went rowing and hiking is 71 -/
theorem total_campers_is_71 :
  total_campers 41 4 26 = 71 := by
  sorry

end total_campers_is_71_l802_80204


namespace inverse_proportion_ordering_l802_80215

/-- Prove that for points on an inverse proportion function with k < 0,
    the y-coordinates have a specific ordering. -/
theorem inverse_proportion_ordering (k : ℝ) (y₁ y₂ y₃ : ℝ) 
  (hk : k < 0)
  (h1 : y₁ = k / (-2))
  (h2 : y₂ = k / 1)
  (h3 : y₃ = k / 2) :
  y₂ < y₃ ∧ y₃ < y₁ :=
by sorry

end inverse_proportion_ordering_l802_80215


namespace quadratic_other_x_intercept_l802_80247

/-- A quadratic function with vertex (5, 8) and one x-intercept at (1, 0) has its other x-intercept at x = 9 -/
theorem quadratic_other_x_intercept (a b c : ℝ) : 
  (∀ x, a * x^2 + b * x + c = 8 - a * (x - 5)^2) →  -- vertex form of quadratic with vertex (5, 8)
  (a * 1^2 + b * 1 + c = 0) →                       -- (1, 0) is an x-intercept
  (∃ x, x ≠ 1 ∧ a * x^2 + b * x + c = 0 ∧ x = 9) :=
by sorry

end quadratic_other_x_intercept_l802_80247


namespace jump_rope_cost_is_seven_l802_80276

/-- The cost of Dalton's jump rope --/
def jump_rope_cost (board_game_cost playground_ball_cost allowance_savings uncle_gift additional_needed : ℕ) : ℕ :=
  (allowance_savings + uncle_gift + additional_needed) - (board_game_cost + playground_ball_cost)

/-- Theorem stating that the jump rope costs $7 --/
theorem jump_rope_cost_is_seven :
  jump_rope_cost 12 4 6 13 4 = 7 := by
  sorry

end jump_rope_cost_is_seven_l802_80276


namespace mowing_problem_l802_80224

/-- Represents the time it takes to mow a lawn together -/
def mowing_time (mary_rate tom_rate : ℚ) (tom_alone_time : ℚ) : ℚ :=
  let remaining_lawn := 1 - tom_rate * tom_alone_time
  remaining_lawn / (mary_rate + tom_rate)

theorem mowing_problem :
  let mary_rate : ℚ := 1 / 3  -- Mary's mowing rate (lawn per hour)
  let tom_rate : ℚ := 1 / 6   -- Tom's mowing rate (lawn per hour)
  let tom_alone_time : ℚ := 2 -- Time Tom mows alone (hours)
  mowing_time mary_rate tom_rate tom_alone_time = 4 / 3 := by
sorry

end mowing_problem_l802_80224


namespace max_p_value_l802_80217

theorem max_p_value (p q r s : ℕ+) 
  (h1 : p < 3 * q)
  (h2 : q < 4 * r)
  (h3 : r < 5 * s)
  (h4 : s < 90) :
  p ≤ 5324 ∧ ∃ (p' q' r' s' : ℕ+), 
    p' = 5324 ∧ 
    p' < 3 * q' ∧ 
    q' < 4 * r' ∧ 
    r' < 5 * s' ∧ 
    s' < 90 :=
by sorry

end max_p_value_l802_80217


namespace set_B_characterization_l802_80275

def U : Set ℕ := {x | x > 0 ∧ Real.log x < 1}

def A : Set ℕ := {x | x ∈ U ∧ ∃ n : ℕ, n ≤ 4 ∧ x = 2*n + 1}

def B : Set ℕ := {x | x ∈ U ∧ x % 2 = 0}

theorem set_B_characterization :
  B = {2, 4, 6, 8} :=
sorry

end set_B_characterization_l802_80275


namespace kennel_problem_l802_80208

/-- Represents the number of dogs in a kennel that don't like either watermelon or salmon. -/
def dogs_not_liking_either (total : ℕ) (watermelon : ℕ) (salmon : ℕ) (both : ℕ) : ℕ :=
  total - (watermelon + salmon - both)

/-- Theorem stating that in a kennel of 60 dogs, where 9 like watermelon, 
    48 like salmon, and 5 like both, 8 dogs don't like either. -/
theorem kennel_problem : dogs_not_liking_either 60 9 48 5 = 8 := by
  sorry

end kennel_problem_l802_80208


namespace procedure_cost_l802_80225

theorem procedure_cost (insurance_coverage : Real) (amount_saved : Real) :
  insurance_coverage = 0.80 →
  amount_saved = 3520 →
  ∃ (cost : Real), cost = 4400 ∧ insurance_coverage * cost = amount_saved :=
by sorry

end procedure_cost_l802_80225


namespace fraction_valid_for_all_reals_l802_80241

theorem fraction_valid_for_all_reals :
  ∀ x : ℝ, (x^2 + 1 ≠ 0) :=
by sorry

end fraction_valid_for_all_reals_l802_80241


namespace total_paths_a_to_d_l802_80218

/-- The number of paths between two adjacent points -/
def paths_between_adjacent : ℕ := 2

/-- The number of direct paths from A to D -/
def direct_paths : ℕ := 1

/-- Theorem: The total number of paths from A to D is 9 -/
theorem total_paths_a_to_d : 
  paths_between_adjacent^3 + direct_paths = 9 := by sorry

end total_paths_a_to_d_l802_80218


namespace contrapositive_example_l802_80205

theorem contrapositive_example : 
  (∀ x : ℝ, x ≤ -3 → x < 0) ↔ (∀ x : ℝ, x ≥ 0 → x > -3) :=
by sorry

end contrapositive_example_l802_80205


namespace integral_identity_l802_80228

theorem integral_identity : ∫ x in (2 * Real.arctan 2)..(2 * Real.arctan 3), 
  1 / (Real.cos x * (1 - Real.cos x)) = 1/6 + Real.log 2 - Real.log 3 := by
  sorry

end integral_identity_l802_80228


namespace division_multiplication_equality_l802_80207

theorem division_multiplication_equality : (1100 / 25) * 4 / 11 = 16 := by
  sorry

end division_multiplication_equality_l802_80207


namespace mike_tire_change_l802_80221

def total_tires_changed (num_motorcycles num_cars tires_per_motorcycle tires_per_car : ℕ) : ℕ :=
  num_motorcycles * tires_per_motorcycle + num_cars * tires_per_car

theorem mike_tire_change :
  let num_motorcycles : ℕ := 12
  let num_cars : ℕ := 10
  let tires_per_motorcycle : ℕ := 2
  let tires_per_car : ℕ := 4
  total_tires_changed num_motorcycles num_cars tires_per_motorcycle tires_per_car = 64 := by
  sorry

end mike_tire_change_l802_80221


namespace three_quarters_of_48_minus_12_l802_80251

theorem three_quarters_of_48_minus_12 : (3 / 4 : ℚ) * 48 - 12 = 24 := by
  sorry

end three_quarters_of_48_minus_12_l802_80251


namespace tenth_number_in_sixteenth_group_l802_80284

/-- The sequence a_n defined by a_n = 2n - 3 -/
def a (n : ℕ) : ℤ := 2 * n - 3

/-- The first number in the kth group -/
def first_in_group (k : ℕ) : ℤ := k^2 - k - 1

/-- The mth number in the kth group -/
def number_in_group (k m : ℕ) : ℤ := first_in_group k + 2 * (m - 1)

theorem tenth_number_in_sixteenth_group :
  number_in_group 16 10 = 257 := by sorry

end tenth_number_in_sixteenth_group_l802_80284


namespace half_of_expression_l802_80263

theorem half_of_expression : (2^12 + 3 * 2^10) / 2 = 2^9 * 7 := by sorry

end half_of_expression_l802_80263


namespace two_digit_number_decimal_sum_l802_80261

theorem two_digit_number_decimal_sum (a b : ℕ) (h1 : a ≥ 1 ∧ a ≤ 9) (h2 : b ≥ 0 ∧ b ≤ 9) :
  let n := 10 * a + b
  (n : ℚ) + (a : ℚ) + (b : ℚ) / 10 = 869 / 10 → n = 79 := by
  sorry

end two_digit_number_decimal_sum_l802_80261


namespace willow_peach_tree_count_l802_80212

/-- Represents the dimensions of a rectangular playground -/
structure Playground where
  length : ℕ
  width : ℕ

/-- Calculates the perimeter of a rectangular playground -/
def perimeter (p : Playground) : ℕ := 2 * (p.length + p.width)

/-- Represents the spacing between trees -/
def treeSpacing : ℕ := 10

/-- Calculates the total number of tree positions along the perimeter -/
def totalTreePositions (p : Playground) : ℕ := perimeter p / treeSpacing

/-- Theorem: The number of willow trees (or peach trees) is half of the total tree positions -/
theorem willow_peach_tree_count (p : Playground) (h1 : p.length = 150) (h2 : p.width = 60) :
  totalTreePositions p / 2 = 21 := by
  sorry

#check willow_peach_tree_count

end willow_peach_tree_count_l802_80212


namespace probability_three_girls_in_six_children_l802_80281

theorem probability_three_girls_in_six_children :
  let n : ℕ := 6  -- Total number of children
  let k : ℕ := 3  -- Number of girls we're interested in
  let p : ℚ := 1/2  -- Probability of having a girl
  Nat.choose n k * p^k * (1-p)^(n-k) = 5/16 := by
  sorry

end probability_three_girls_in_six_children_l802_80281


namespace monomial_count_l802_80266

/-- An algebraic expression is a monomial if it is a single number, a single variable, or a product of numbers and variables without variables in the denominator. -/
def is_monomial (expr : String) : Bool :=
  sorry

/-- The set of given algebraic expressions -/
def expressions : List String := ["2x^2", "-3", "x-2y", "t", "6m^2/π", "1/a", "m^3+2m^2-m"]

/-- Count the number of monomials in a list of expressions -/
def count_monomials (exprs : List String) : Nat :=
  sorry

/-- The main theorem: The number of monomials in the given set of expressions is 4 -/
theorem monomial_count : count_monomials expressions = 4 := by
  sorry

end monomial_count_l802_80266


namespace haris_joining_time_l802_80280

theorem haris_joining_time (praveen_investment hari_investment : ℝ) 
  (profit_ratio_praveen profit_ratio_hari : ℕ) (x : ℝ) :
  praveen_investment = 3780 →
  hari_investment = 9720 →
  profit_ratio_praveen = 2 →
  profit_ratio_hari = 3 →
  (praveen_investment * 12) / (hari_investment * (12 - x)) = 
    profit_ratio_praveen / profit_ratio_hari →
  x = 5 := by sorry

end haris_joining_time_l802_80280


namespace charlie_age_when_jenny_twice_bobby_l802_80294

theorem charlie_age_when_jenny_twice_bobby (jenny charlie bobby : ℕ) : 
  jenny = charlie + 5 → 
  charlie = bobby + 3 → 
  ∃ x : ℕ, jenny + x = 2 * (bobby + x) ∧ charlie + x = 11 :=
by sorry

end charlie_age_when_jenny_twice_bobby_l802_80294


namespace largest_absolute_value_l802_80289

theorem largest_absolute_value : 
  let S : Finset ℤ := {4, -5, 0, -1}
  ∀ x ∈ S, |(-5 : ℤ)| ≥ |x| := by
  sorry

end largest_absolute_value_l802_80289


namespace max_fruits_bought_l802_80295

/-- Represents the cost of each fruit in RM -/
structure FruitCost where
  apple : ℕ
  mango : ℕ
  papaya : ℕ

/-- Represents the number of each fruit bought -/
structure FruitCount where
  apple : ℕ
  mango : ℕ
  papaya : ℕ

/-- Calculates the total cost of fruits bought -/
def totalCost (cost : FruitCost) (count : FruitCount) : ℕ :=
  cost.apple * count.apple + cost.mango * count.mango + cost.papaya * count.papaya

/-- Calculates the total number of fruits bought -/
def totalFruits (count : FruitCount) : ℕ :=
  count.apple + count.mango + count.papaya

/-- Theorem stating the maximum number of fruits that can be bought under given conditions -/
theorem max_fruits_bought (cost : FruitCost) (count : FruitCount) 
    (h_apple_cost : cost.apple = 3)
    (h_mango_cost : cost.mango = 4)
    (h_papaya_cost : cost.papaya = 5)
    (h_at_least_one : count.apple ≥ 1 ∧ count.mango ≥ 1 ∧ count.papaya ≥ 1)
    (h_total_cost : totalCost cost count = 50) :
    totalFruits count ≤ 15 ∧ ∃ (max_count : FruitCount), totalFruits max_count = 15 ∧ totalCost cost max_count = 50 :=
  sorry


end max_fruits_bought_l802_80295


namespace next_friday_birthday_l802_80259

/-- Represents the day of the week --/
inductive DayOfWeek
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Checks if a given year is a leap year --/
def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)

/-- Calculates the day of the week for May 27 in a given year, 
    assuming May 27, 2013 was a Monday --/
def dayOfWeekMay27 (year : Nat) : DayOfWeek :=
  sorry

/-- Theorem: The next year after 2013 when May 27 falls on a Friday is 2016 --/
theorem next_friday_birthday : 
  (dayOfWeekMay27 2013 = DayOfWeek.Monday) → 
  (∀ y : Nat, 2013 < y ∧ y < 2016 → dayOfWeekMay27 y ≠ DayOfWeek.Friday) ∧
  (dayOfWeekMay27 2016 = DayOfWeek.Friday) :=
sorry

end next_friday_birthday_l802_80259


namespace probability_of_red_ball_l802_80255

/-- The probability of drawing a red ball from a bag with white and red balls -/
theorem probability_of_red_ball (total_balls : ℕ) (white_balls : ℕ) (red_balls : ℕ) 
  (h1 : total_balls = white_balls + red_balls)
  (h2 : white_balls = 3)
  (h3 : red_balls = 7) :
  (red_balls : ℚ) / total_balls = 7 / 10 := by
sorry

end probability_of_red_ball_l802_80255


namespace mans_rate_in_still_water_l802_80203

/-- Given a man who can row with the stream at 16 km/h and against the stream at 8 km/h,
    his rate in still water is 12 km/h. -/
theorem mans_rate_in_still_water
  (speed_with_stream : ℝ)
  (speed_against_stream : ℝ)
  (h_with : speed_with_stream = 16)
  (h_against : speed_against_stream = 8) :
  (speed_with_stream + speed_against_stream) / 2 = 12 := by
  sorry

#check mans_rate_in_still_water

end mans_rate_in_still_water_l802_80203


namespace unique_pair_l802_80220

def is_valid_pair (a b : ℕ) : Prop :=
  a ≠ b ∧ a > 11 ∧ b > 11 ∧ a + b = 28 ∧ (Even a ∨ Even b)

theorem unique_pair : ∀ a b : ℕ, is_valid_pair a b → (a = 12 ∧ b = 16) ∨ (a = 16 ∧ b = 12) :=
by sorry

end unique_pair_l802_80220


namespace inequality_proof_l802_80219

theorem inequality_proof (a b c : ℝ) 
  (h1 : 0 ≤ a ∧ a ≤ 1) 
  (h2 : 0 ≤ b ∧ b ≤ 1) 
  (h3 : 0 ≤ c ∧ c ≤ 1) 
  (h4 : a + b + c = 1 + Real.sqrt (2 * (1 - a) * (1 - b) * (1 - c))) :
  Real.sqrt (1 - a^2) + Real.sqrt (1 - b^2) + Real.sqrt (1 - c^2) ≤ 3 * Real.sqrt 3 / 2 := by
  sorry

end inequality_proof_l802_80219


namespace coordinates_of_D_l802_80238

-- Define the points
def C : ℝ × ℝ := (5, -1)
def M : ℝ × ℝ := (3, 7)

-- Define D as a point that satisfies the midpoint condition
def D : ℝ × ℝ := (2 * M.1 - C.1, 2 * M.2 - C.2)

-- Theorem statement
theorem coordinates_of_D :
  D.1 * D.2 = 15 ∧ D.1 + D.2 = 16 := by
  sorry

end coordinates_of_D_l802_80238


namespace remainder_274_pow_274_mod_13_l802_80246

theorem remainder_274_pow_274_mod_13 : 274^274 % 13 = 1 := by
  sorry

end remainder_274_pow_274_mod_13_l802_80246


namespace min_value_sum_reciprocals_l802_80226

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_7 : a + b + c + d + e + f = 7) :
  (1/a) + (4/b) + (9/c) + (16/d) + (25/e) + (36/f) ≥ 63 ∧
  ((1/a) + (4/b) + (9/c) + (16/d) + (25/e) + (36/f) = 63 ↔ 
   a = 1/3 ∧ b = 2/3 ∧ c = 1 ∧ d = 4/3 ∧ e = 5/3 ∧ f = 2) :=
by sorry

end min_value_sum_reciprocals_l802_80226


namespace triangle_inequality_1_triangle_inequality_2_l802_80223

-- Define a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  pos_A : A > 0
  pos_B : B > 0
  pos_C : C > 0
  angle_sum : A + B + C = π

-- State the theorems
theorem triangle_inequality_1 (t : Triangle) :
  1 / t.a^3 + 1 / t.b^3 + 1 / t.c^3 + t.a * t.b * t.c ≥ 2 * Real.sqrt 3 := by
  sorry

theorem triangle_inequality_2 (t : Triangle) :
  1 / t.A + 1 / t.B + 1 / t.C ≥ 9 / π := by
  sorry

end triangle_inequality_1_triangle_inequality_2_l802_80223


namespace zephyrian_word_count_l802_80209

/-- The number of letters in the Zephyrian alphabet -/
def zephyrian_alphabet_size : ℕ := 8

/-- The maximum word length in the Zephyrian language -/
def max_word_length : ℕ := 3

/-- Calculate the number of possible words in the Zephyrian language -/
def count_zephyrian_words : ℕ :=
  zephyrian_alphabet_size +
  zephyrian_alphabet_size ^ 2 +
  zephyrian_alphabet_size ^ 3

theorem zephyrian_word_count :
  count_zephyrian_words = 584 :=
sorry

end zephyrian_word_count_l802_80209


namespace population_change_l802_80298

theorem population_change (P : ℝ) : 
  P * 1.12 * 0.88 = 14784 → P = 15000 := by
  sorry

end population_change_l802_80298


namespace email_sending_ways_l802_80262

/-- The number of ways to send emails given the number of email addresses and the number of emails to be sent. -/
def number_of_ways (num_addresses : ℕ) (num_emails : ℕ) : ℕ :=
  num_addresses ^ num_emails

/-- Theorem stating that the number of ways to send 5 emails using 3 email addresses is 3^5. -/
theorem email_sending_ways : number_of_ways 3 5 = 3^5 := by
  sorry

end email_sending_ways_l802_80262


namespace inequality_transformation_l802_80232

theorem inequality_transformation (a b : ℝ) (h : a > b) : 2 * a + 1 > 2 * b + 1 := by
  sorry

end inequality_transformation_l802_80232


namespace trig_function_properties_l802_80227

open Real

theorem trig_function_properties :
  (∀ x, cos (x + π/3) = cos (π/3 - x)) ∧
  (∀ x, 3 * sin (2 * (x - π/6) + π/3) = 3 * sin (2 * x)) := by
  sorry

end trig_function_properties_l802_80227


namespace expression_evaluation_l802_80268

theorem expression_evaluation : 6 * 5 * ((-1) ^ (2 ^ (3 ^ 5))) + ((-1) ^ (5 ^ (3 ^ 2))) = 29 := by
  sorry

end expression_evaluation_l802_80268


namespace hyperbola_eccentricity_l802_80231

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 and asymptote y = 2x -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b = 2*a) :
  let e := Real.sqrt (1 + b^2 / a^2)
  e = Real.sqrt 5 := by
sorry

end hyperbola_eccentricity_l802_80231


namespace inequality_proof_l802_80242

theorem inequality_proof (x y z : ℝ) : 
  x^2 / (x^2 + 2*y*z) + y^2 / (y^2 + 2*z*x) + z^2 / (z^2 + 2*x*y) ≥ 1 := by
  sorry

end inequality_proof_l802_80242


namespace square_sum_divisibility_problem_l802_80291

theorem square_sum_divisibility_problem :
  ∃ a b : ℕ, a^2 + b^2 = 2018 ∧ 7 ∣ (a + b) ∧
  ((a = 43 ∧ b = 13) ∨ (a = 13 ∧ b = 43)) ∧
  (∀ x y : ℕ, x^2 + y^2 = 2018 ∧ 7 ∣ (x + y) → (x = 43 ∧ y = 13) ∨ (x = 13 ∧ y = 43)) :=
by sorry

end square_sum_divisibility_problem_l802_80291


namespace coffee_decaf_percentage_l802_80244

/-- Given initial coffee stock, percentages, and additional purchase,
    calculate the percentage of decaffeinated coffee in the new batch. -/
theorem coffee_decaf_percentage
  (initial_stock : ℝ)
  (initial_decaf_percent : ℝ)
  (additional_purchase : ℝ)
  (final_decaf_percent : ℝ)
  (h1 : initial_stock = 400)
  (h2 : initial_decaf_percent = 0.3)
  (h3 : additional_purchase = 100)
  (h4 : final_decaf_percent = 0.36)
  (h5 : initial_stock > 0)
  (h6 : additional_purchase > 0) :
  let total_stock := initial_stock + additional_purchase
  let initial_decaf := initial_stock * initial_decaf_percent
  let total_decaf := total_stock * final_decaf_percent
  let new_decaf := total_decaf - initial_decaf
  new_decaf / additional_purchase = 0.6 := by
  sorry

end coffee_decaf_percentage_l802_80244


namespace simplify_expression1_simplify_expression2_l802_80297

-- First expression
theorem simplify_expression1 (a b : ℝ) : (a - 2*b) - (2*b - 5*a) = 6*a - 4*b := by sorry

-- Second expression
theorem simplify_expression2 (m n : ℝ) : -m^2*n + (4*m*n^2 - 3*m*n) - 2*(m*n^2 - 3*m^2*n) = 5*m^2*n + 2*m*n^2 - 3*m*n := by sorry

end simplify_expression1_simplify_expression2_l802_80297


namespace wire_forms_perpendicular_segments_l802_80283

/-- Represents a wire configuration -/
structure WireConfiguration where
  semicircles : ℕ
  straight_segments : ℕ
  segment_length : ℝ

/-- Represents a figure formed by the wire -/
inductive Figure
  | TwoPerpendicularSegments
  | Other

/-- Checks if a wire configuration can form two perpendicular segments -/
def can_form_perpendicular_segments (w : WireConfiguration) : Prop :=
  w.semicircles = 3 ∧ w.straight_segments = 4

/-- Theorem stating that a specific wire configuration can form two perpendicular segments -/
theorem wire_forms_perpendicular_segments (w : WireConfiguration) 
  (h : can_form_perpendicular_segments w) : 
  ∃ (f : Figure), f = Figure.TwoPerpendicularSegments :=
sorry

end wire_forms_perpendicular_segments_l802_80283


namespace pet_store_cages_l802_80272

/-- The number of bird cages in a pet store -/
def num_cages : ℕ := 9

/-- The number of parrots in each cage -/
def parrots_per_cage : ℕ := 2

/-- The number of parakeets in each cage -/
def parakeets_per_cage : ℕ := 6

/-- The total number of birds in the pet store -/
def total_birds : ℕ := 72

/-- Theorem stating that the number of bird cages is correct -/
theorem pet_store_cages :
  num_cages * (parrots_per_cage + parakeets_per_cage) = total_birds :=
by sorry

end pet_store_cages_l802_80272


namespace last_day_of_month_l802_80213

/-- Days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to advance a day by n days -/
def advanceDay (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | Nat.succ m => nextDay (advanceDay d m)

/-- Theorem: If the 24th day of a 31-day month is a Wednesday, 
    then the last day of the month (31st) is also a Wednesday -/
theorem last_day_of_month (d : DayOfWeek) (h : d = DayOfWeek.Wednesday) :
  advanceDay d 7 = DayOfWeek.Wednesday :=
by
  sorry

end last_day_of_month_l802_80213


namespace albert_pizza_consumption_l802_80270

/-- The number of large pizzas Albert buys -/
def large_pizzas : ℕ := 2

/-- The number of small pizzas Albert buys -/
def small_pizzas : ℕ := 2

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := 16

/-- The number of slices in a small pizza -/
def small_pizza_slices : ℕ := 8

/-- The total number of pizza slices Albert eats in one day -/
def total_slices : ℕ := large_pizzas * large_pizza_slices + small_pizzas * small_pizza_slices

theorem albert_pizza_consumption :
  total_slices = 48 := by
  sorry

end albert_pizza_consumption_l802_80270


namespace theater_eye_color_ratio_l802_80265

theorem theater_eye_color_ratio :
  let total_people : ℕ := 100
  let blue_eyes : ℕ := 19
  let brown_eyes : ℕ := total_people / 2
  let green_eyes : ℕ := 6
  let black_eyes : ℕ := total_people - (blue_eyes + brown_eyes + green_eyes)
  (black_eyes : ℚ) / total_people = 1 / 4 := by
  sorry

end theater_eye_color_ratio_l802_80265


namespace junior_score_l802_80273

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


end junior_score_l802_80273
