import Mathlib

namespace only_set_C_is_right_triangle_l782_78220

-- Define a function to check if three numbers satisfy the Pythagorean theorem
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

-- Theorem statement
theorem only_set_C_is_right_triangle :
  (¬ isPythagoreanTriple 3 4 2) ∧
  (¬ isPythagoreanTriple 5 12 15) ∧
  (isPythagoreanTriple 8 15 17) ∧
  (¬ isPythagoreanTriple 9 16 25) :=
by sorry


end only_set_C_is_right_triangle_l782_78220


namespace square_area_given_circle_l782_78216

-- Define the circle's area
def circle_area : ℝ := 39424

-- Define the relationship between square perimeter and circle radius
def square_perimeter_equals_circle_radius (square_side : ℝ) (circle_radius : ℝ) : Prop :=
  4 * square_side = circle_radius

-- Theorem statement
theorem square_area_given_circle (square_side : ℝ) (circle_radius : ℝ) :
  circle_area = Real.pi * circle_radius^2 →
  square_perimeter_equals_circle_radius square_side circle_radius →
  square_side^2 = 784 := by
  sorry

end square_area_given_circle_l782_78216


namespace all_radii_equal_l782_78256

/-- A circle with radius 2 cm -/
structure Circle :=
  (radius : ℝ)
  (h : radius = 2)

/-- Any radius of the circle is 2 cm -/
theorem all_radii_equal (c : Circle) (r : ℝ) (h : r = c.radius) : r = 2 := by
  sorry

end all_radii_equal_l782_78256


namespace base_b_not_divisible_by_five_l782_78291

theorem base_b_not_divisible_by_five (b : ℤ) : b ∈ ({5, 6, 7, 8, 10} : Set ℤ) →
  (¬(5 ∣ (b * (3 * b^2 - 3 * b - 1))) ↔ (b = 6 ∨ b = 8)) := by
  sorry

end base_b_not_divisible_by_five_l782_78291


namespace triangle_properties_l782_78213

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.c = (2 * Real.sqrt 3 / 3) * t.b * Real.sin (t.A + π/3))
  (h2 : t.a + t.c = 4) :
  t.B = π/3 ∧ 4 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 6 := by
  sorry

end triangle_properties_l782_78213


namespace parabola_triangle_area_l782_78282

-- Define the parabolas
def M1 (a c x : ℝ) : ℝ := a * x^2 + c
def M2 (a c x : ℝ) : ℝ := a * (x - 2)^2 + c - 5

-- Define the theorem
theorem parabola_triangle_area (a c : ℝ) :
  -- M2 passes through the vertex of M1
  (M2 a c 0 = M1 a c 0) →
  -- Point C on M2 has coordinates (2, c-5)
  (M2 a c 2 = c - 5) →
  -- The area of triangle ABC is 10
  ∃ (x_B y_B : ℝ), 
    x_B = 2 ∧ 
    y_B = M1 a c x_B ∧ 
    (1/2 * |x_B - 0| * |y_B - (c - 5)| = 10) :=
by sorry

end parabola_triangle_area_l782_78282


namespace max_trailing_zeros_l782_78201

/-- Given three natural numbers whose sum is 1003, the maximum number of trailing zeros in their product is 7 -/
theorem max_trailing_zeros (a b c : ℕ) (h_sum : a + b + c = 1003) : 
  (∃ (n : ℕ), a * b * c = n * 10^7 ∧ n % 10 ≠ 0) ∧ 
  ¬(∃ (m : ℕ), a * b * c = m * 10^8) :=
sorry

end max_trailing_zeros_l782_78201


namespace parabola_directrix_l782_78229

/-- The equation of the directrix of the parabola y^2 = 6x is x = -3/2 -/
theorem parabola_directrix (x y : ℝ) : y^2 = 6*x → (∃ (k : ℝ), k = -3/2 ∧ x = k) :=
sorry

end parabola_directrix_l782_78229


namespace division_equations_for_26_l782_78217

theorem division_equations_for_26 : 
  {(x, y) : ℕ × ℕ | 1 ≤ x ∧ x ≤ 9 ∧ 1 ≤ y ∧ y ≤ 9 ∧ 26 = x * y + 2} = 
  {(3, 8), (4, 6), (6, 4), (8, 3)} := by sorry

end division_equations_for_26_l782_78217


namespace B_equals_set_l782_78266

def A : Set ℤ := {-1, 2, 3, 4}

def B : Set ℤ := {y | ∃ x ∈ A, y = x^2 - 2*x + 2}

theorem B_equals_set : B = {2, 5, 10} := by sorry

end B_equals_set_l782_78266


namespace fraction_division_addition_l782_78202

theorem fraction_division_addition : (3 / 7 : ℚ) / 4 + 2 / 7 = 11 / 28 := by
  sorry

end fraction_division_addition_l782_78202


namespace sum_of_coefficients_l782_78247

theorem sum_of_coefficients (b₀ b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) :
  (∀ x : ℝ, (2*x + 3)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 15625 := by
  sorry

end sum_of_coefficients_l782_78247


namespace guppies_count_l782_78269

/-- The number of guppies Rick bought -/
def guppies : ℕ := sorry

/-- The number of clowns Tim bought -/
def clowns : ℕ := sorry

/-- The number of tetras bought -/
def tetras : ℕ := sorry

/-- The total number of animals bought -/
def total_animals : ℕ := 330

theorem guppies_count :
  (tetras = 4 * clowns) →
  (clowns = 2 * guppies) →
  (guppies + clowns + tetras = total_animals) →
  guppies = 30 := by sorry

end guppies_count_l782_78269


namespace colonization_combinations_eq_77056_l782_78289

/-- The number of Earth-like planets -/
def earth_like_planets : ℕ := 8

/-- The number of Mars-like planets -/
def mars_like_planets : ℕ := 12

/-- The resource cost to colonize an Earth-like planet -/
def earth_cost : ℕ := 3

/-- The resource cost to colonize a Mars-like planet -/
def mars_cost : ℕ := 1

/-- The total available resources -/
def total_resources : ℕ := 18

/-- The function to calculate the number of combinations -/
def colonization_combinations : ℕ :=
  (Nat.choose earth_like_planets 2 * Nat.choose mars_like_planets 12) +
  (Nat.choose earth_like_planets 4 * Nat.choose mars_like_planets 6) +
  (Nat.choose earth_like_planets 5 * Nat.choose mars_like_planets 3) +
  (Nat.choose earth_like_planets 6 * Nat.choose mars_like_planets 0)

/-- The theorem stating that the number of colonization combinations is 77056 -/
theorem colonization_combinations_eq_77056 : colonization_combinations = 77056 := by
  sorry

end colonization_combinations_eq_77056_l782_78289


namespace complex_fraction_simplification_l782_78252

theorem complex_fraction_simplification :
  let i : ℂ := Complex.I
  (3 + 4*i) / (1 + i) = (7:ℂ)/2 + (1:ℂ)/2 * i :=
by sorry

end complex_fraction_simplification_l782_78252


namespace project_contribution_balance_l782_78275

/-- The contribution of the first worker to the project -/
def first_worker_contribution : ℚ := 1/3

/-- The contribution of the second worker to the project -/
def second_worker_contribution : ℚ := 1/3

/-- The contribution of the third worker to the project -/
def third_worker_contribution : ℚ := 1/3

/-- The total full-time equivalent (FTE) for the project -/
def total_fte : ℚ := 1

theorem project_contribution_balance :
  first_worker_contribution + second_worker_contribution + third_worker_contribution = total_fte :=
sorry

end project_contribution_balance_l782_78275


namespace creeping_jennies_count_l782_78255

/-- The number of creeping jennies per planter -/
def creeping_jennies : ℕ := sorry

/-- The cost of a palm fern -/
def palm_fern_cost : ℚ := 15

/-- The cost of a creeping jenny -/
def creeping_jenny_cost : ℚ := 4

/-- The cost of a geranium -/
def geranium_cost : ℚ := 3.5

/-- The number of geraniums per planter -/
def geraniums_per_planter : ℕ := 4

/-- The number of planters -/
def num_planters : ℕ := 4

/-- The total cost for all planters -/
def total_cost : ℚ := 180

theorem creeping_jennies_count : 
  creeping_jennies = 4 ∧ 
  (num_planters : ℚ) * (palm_fern_cost + creeping_jenny_cost * (creeping_jennies : ℚ) + 
    geranium_cost * (geraniums_per_planter : ℚ)) = total_cost :=
sorry

end creeping_jennies_count_l782_78255


namespace day_shift_percentage_l782_78260

theorem day_shift_percentage
  (excel_percentage : ℝ)
  (excel_and_night_percentage : ℝ)
  (h1 : excel_percentage = 0.20)
  (h2 : excel_and_night_percentage = 0.06) :
  1 - (excel_and_night_percentage / excel_percentage) = 0.70 := by
  sorry

end day_shift_percentage_l782_78260


namespace exactly_two_classical_models_l782_78281

/-- Represents a random event model -/
structure RandomEventModel where
  is_finite : Bool
  has_equal_likelihood : Bool

/-- Checks if a random event model is a classical probability model -/
def is_classical_probability_model (model : RandomEventModel) : Bool :=
  model.is_finite && model.has_equal_likelihood

/-- The list of random event models given in the problem -/
def models : List RandomEventModel := [
  ⟨false, true⟩,   -- Model 1
  ⟨true, false⟩,   -- Model 2
  ⟨true, true⟩,    -- Model 3
  ⟨false, false⟩,  -- Model 4
  ⟨true, true⟩     -- Model 5
]

theorem exactly_two_classical_models : 
  (models.filter is_classical_probability_model).length = 2 := by
  sorry

end exactly_two_classical_models_l782_78281


namespace number_division_problem_l782_78244

theorem number_division_problem (x y : ℝ) 
  (h1 : (x - 5) / y = 7)
  (h2 : (x - 4) / 10 = 5) : 
  y = 7 := by
sorry

end number_division_problem_l782_78244


namespace students_taking_one_subject_l782_78284

theorem students_taking_one_subject (total_geometry : ℕ) (both_subjects : ℕ) (science_only : ℕ)
  (h1 : both_subjects = 15)
  (h2 : total_geometry = 30)
  (h3 : science_only = 18) :
  total_geometry - both_subjects + science_only = 33 := by
sorry

end students_taking_one_subject_l782_78284


namespace average_score_three_subjects_l782_78261

theorem average_score_three_subjects 
  (math_score : ℝ)
  (korean_english_avg : ℝ)
  (h1 : math_score = 100)
  (h2 : korean_english_avg = 88) : 
  (math_score + 2 * korean_english_avg) / 3 = 92 := by
  sorry

end average_score_three_subjects_l782_78261


namespace first_ten_digits_of_expression_l782_78274

theorem first_ten_digits_of_expression (ε : ℝ) (h : ε > 0) :
  ∃ n : ℤ, (5 + Real.sqrt 26) ^ 100 = n - ε ∧ 0 < ε ∧ ε < 1e-10 :=
sorry

end first_ten_digits_of_expression_l782_78274


namespace restock_is_mode_l782_78272

def shoe_sizes : List ℝ := [22, 22.5, 23, 23.5, 24, 24.5, 25]
def quantities : List ℕ := [3, 5, 10, 15, 8, 3, 2]
def restock_size : ℝ := 23.5

def mode (sizes : List ℝ) (quants : List ℕ) : ℝ :=
  let paired := List.zip sizes quants
  let max_quant := paired.map (λ p => p.2) |>.maximum?
  match paired.find? (λ p => p.2 = max_quant) with
  | some (size, _) => size
  | none => 0  -- This case should not occur if the lists are non-empty

theorem restock_is_mode :
  mode shoe_sizes quantities = restock_size :=
sorry

end restock_is_mode_l782_78272


namespace parabola_focus_directrix_distance_l782_78236

/-- A parabola in the Cartesian plane -/
structure Parabola where
  /-- The parameter p of the parabola -/
  p : ℝ
  /-- The vertex is at the origin -/
  vertex_at_origin : True
  /-- The focus is on the x-axis -/
  focus_on_x_axis : True
  /-- The parabola passes through the point (1, 2) -/
  passes_through_point : p = 2

/-- The distance from the focus to the directrix of a parabola -/
def focus_directrix_distance (c : Parabola) : ℝ := c.p

theorem parabola_focus_directrix_distance :
  ∀ c : Parabola, focus_directrix_distance c = 2 := by
  sorry

end parabola_focus_directrix_distance_l782_78236


namespace geometric_sequence_constant_l782_78231

/-- A geometric sequence with positive terms. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q ∧ a n > 0

/-- The theorem stating that if a geometric sequence satisfies the given condition,
    then it is a constant sequence. -/
theorem geometric_sequence_constant
  (a : ℕ → ℝ)
  (h_geometric : GeometricSequence a)
  (h_condition : (a 3 + a 11) / a 7 ≤ 2) :
  ∃ c : ℝ, ∀ n : ℕ, a n = c :=
sorry

end geometric_sequence_constant_l782_78231


namespace modulus_of_neg_one_plus_i_l782_78257

theorem modulus_of_neg_one_plus_i :
  Complex.abs (-1 + Complex.I) = Real.sqrt 2 := by
  sorry

end modulus_of_neg_one_plus_i_l782_78257


namespace ac_price_l782_78237

/-- Given a car and an AC with prices in the ratio 3:2, where the car costs $500 more than the AC,
    prove that the price of the AC is $1000. -/
theorem ac_price (car_price ac_price : ℕ) : 
  car_price = 3 * (car_price / 5) ∧ 
  ac_price = 2 * (car_price / 5) ∧ 
  car_price = ac_price + 500 → 
  ac_price = 1000 := by
sorry

end ac_price_l782_78237


namespace sqrt_66_greater_than_8_l782_78280

theorem sqrt_66_greater_than_8 : Real.sqrt 66 > 8 := by
  sorry

end sqrt_66_greater_than_8_l782_78280


namespace matrix_N_computation_l782_78265

theorem matrix_N_computation (N : Matrix (Fin 2) (Fin 2) ℝ) 
  (h1 : N.mulVec (![3, -2]) = ![5, 1])
  (h2 : N.mulVec (![(-2), 4]) = ![0, -2]) :
  N.mulVec (![7, 0]) = ![17.5, 0] := by
sorry

end matrix_N_computation_l782_78265


namespace cupcake_distribution_l782_78267

def dozen : ℕ := 12

theorem cupcake_distribution (total_dozens : ℕ) (cupcakes_per_cousin : ℕ) : 
  total_dozens = 4 → cupcakes_per_cousin = 3 → (dozen * total_dozens) / cupcakes_per_cousin = 16 := by
  sorry

end cupcake_distribution_l782_78267


namespace base_conversion_l782_78273

/-- Given that 132 in base k is equal to 42 in base 10, prove that k = 5 -/
theorem base_conversion (k : ℕ) : k ^ 2 + 3 * k + 2 = 42 → k = 5 := by
  sorry

end base_conversion_l782_78273


namespace at_least_one_fraction_less_than_two_l782_78222

theorem at_least_one_fraction_less_than_two (x y : ℝ) 
  (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

end at_least_one_fraction_less_than_two_l782_78222


namespace solution_set_equivalence_l782_78225

-- Define the quadratic function f(x) = ax^2 + 2x + c
def f (a c x : ℝ) : ℝ := a * x^2 + 2 * x + c

-- Define the quadratic function g(x) = -cx^2 + 2x - a
def g (a c x : ℝ) : ℝ := -c * x^2 + 2 * x - a

-- Theorem statement
theorem solution_set_equivalence (a c : ℝ) :
  (∀ x, -1/3 < x ∧ x < 1/2 → f a c x > 0) →
  (∀ x, g a c x > 0 ↔ -2 < x ∧ x < 3) :=
by sorry

end solution_set_equivalence_l782_78225


namespace power_value_from_equation_l782_78279

theorem power_value_from_equation (a b : ℝ) : 
  |a - 2| + (b + 3)^2 = 0 → b^a = 9 := by
sorry

end power_value_from_equation_l782_78279


namespace smallest_odd_island_has_nine_counties_l782_78235

/-- A rectangular county (graphstum) -/
structure County where
  width : ℕ
  height : ℕ

/-- A rectangular island composed of counties -/
structure Island where
  counties : List County
  isRectangular : Bool
  hasDiagonalRoads : Bool
  hasClosedPath : Bool

/-- The property of having an odd number of counties -/
def hasOddCounties (i : Island) : Prop :=
  Odd (List.length i.counties)

/-- The property of being a valid island configuration -/
def isValidIsland (i : Island) : Prop :=
  i.isRectangular ∧ i.hasDiagonalRoads ∧ i.hasClosedPath ∧ hasOddCounties i

/-- The theorem stating that the smallest valid odd-county island has 9 counties -/
theorem smallest_odd_island_has_nine_counties :
  ∀ i : Island, isValidIsland i → List.length i.counties ≥ 9 :=
sorry

end smallest_odd_island_has_nine_counties_l782_78235


namespace unique_solution_l782_78243

/-- A four-digit number is a natural number between 1000 and 9999, inclusive. -/
def FourDigitNumber (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- The equation that the four-digit number must satisfy. -/
def SatisfiesEquation (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  a * (a + b + c + d) * (a^2 + b^2 + c^2 + d^2) * (a^6 + 2*b^6 + 3*c^6 + 4*d^6) = n

/-- The main theorem stating that 2010 is the only four-digit number satisfying the equation. -/
theorem unique_solution :
  ∀ n : ℕ, FourDigitNumber n → SatisfiesEquation n → n = 2010 :=
sorry

end unique_solution_l782_78243


namespace unique_positive_n_l782_78218

/-- A quadratic equation has exactly one real root if and only if its discriminant is zero -/
axiom discriminant_zero_iff_one_root {a b c : ℝ} (ha : a ≠ 0) :
  b^2 - 4*a*c = 0 ↔ ∃! x, a*x^2 + b*x + c = 0

/-- The quadratic equation y^2 + 6ny + 9n has exactly one real root -/
def has_one_root (n : ℝ) : Prop :=
  ∃! y, y^2 + 6*n*y + 9*n = 0

theorem unique_positive_n :
  ∃! n : ℝ, n > 0 ∧ has_one_root n :=
sorry

end unique_positive_n_l782_78218


namespace congruence_problem_l782_78211

theorem congruence_problem (n : ℤ) : 
  0 ≤ n ∧ n < 151 ∧ (100 * n) % 151 = 93 % 151 → n % 151 = 29 := by
  sorry

end congruence_problem_l782_78211


namespace tucker_tissues_used_l782_78290

/-- The number of tissues Tucker used while sick -/
def tissues_used (tissues_per_box : ℕ) (boxes_bought : ℕ) (tissues_left : ℕ) : ℕ :=
  boxes_bought * tissues_per_box - tissues_left

/-- Theorem stating the number of tissues Tucker used while sick -/
theorem tucker_tissues_used :
  tissues_used 160 3 270 = 210 := by
  sorry

end tucker_tissues_used_l782_78290


namespace area_of_region_t_l782_78239

/-- Represents a rhombus PQRS -/
structure Rhombus where
  side_length : ℝ
  angle_q : ℝ

/-- Represents the region T inside the rhombus -/
def region_t (r : Rhombus) : Set (ℝ × ℝ) :=
  sorry

/-- Calculates the area of a given set in ℝ² -/
def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Theorem stating the area of region T in the given rhombus -/
theorem area_of_region_t (r : Rhombus) 
  (h1 : r.side_length = 4) 
  (h2 : r.angle_q = 150 * π / 180) : 
  abs (area (region_t r) - 1.034) < 0.001 := by
  sorry

end area_of_region_t_l782_78239


namespace min_value_expression_l782_78215

theorem min_value_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y) * (1 / x + 4 / y) ≥ 9 ∧
  ((x + y) * (1 / x + 4 / y) = 9 ↔ y = 2 * x) :=
sorry

end min_value_expression_l782_78215


namespace farm_legs_count_l782_78298

/-- Calculates the total number of legs in a farm with ducks and horses -/
def total_legs (total_animals : ℕ) (num_ducks : ℕ) : ℕ :=
  let num_horses := total_animals - num_ducks
  let duck_legs := 2 * num_ducks
  let horse_legs := 4 * num_horses
  duck_legs + horse_legs

/-- Proves that in a farm with 11 animals, including 7 ducks and the rest horses, 
    the total number of legs is 30 -/
theorem farm_legs_count : total_legs 11 7 = 30 := by
  sorry

end farm_legs_count_l782_78298


namespace shop_owner_profit_l782_78204

/-- Calculates the percentage profit of a shop owner who cheats while buying and selling -/
theorem shop_owner_profit (buy_cheat : ℝ) (sell_cheat : ℝ) : 
  buy_cheat = 0.12 → sell_cheat = 0.3 → 
  (((1 + buy_cheat) / (1 - sell_cheat) - 1) * 100 : ℝ) = 60 := by
  sorry

end shop_owner_profit_l782_78204


namespace age_ratio_l782_78240

def arun_future_age : ℕ := 30
def years_to_future : ℕ := 10
def deepak_age : ℕ := 50

theorem age_ratio :
  (arun_future_age - years_to_future) / deepak_age = 2 / 5 := by
  sorry

end age_ratio_l782_78240


namespace max_roads_in_graphia_l782_78212

/-- A graph representing the towns and roads in Graphia. -/
structure Graphia where
  towns : Finset Nat
  roads : Finset (Nat × Nat)

/-- The number of towns in Graphia. -/
def num_towns : Nat := 100

/-- A function representing Peter's travel pattern. -/
def peter_travel (g : Graphia) (start : Nat) : List Nat := sorry

/-- The condition that each town is visited exactly twice. -/
def all_towns_visited_twice (g : Graphia) : Prop :=
  ∀ t ∈ g.towns, (List.count t (List.join (List.map (peter_travel g) (List.range num_towns)))) = 2

/-- The theorem stating the maximum number of roads in Graphia. -/
theorem max_roads_in_graphia :
  ∀ g : Graphia,
    g.towns.card = num_towns →
    all_towns_visited_twice g →
    g.roads.card ≤ 4851 :=
sorry

end max_roads_in_graphia_l782_78212


namespace line_equation_proof_l782_78293

theorem line_equation_proof (m b k : ℝ) : 
  (∃! k, ∀ y₁ y₂, y₁ = k^2 + 6*k + 5 ∧ y₂ = m*k + b → |y₁ - y₂| = 7) →
  (8 = 2*m + b) →
  (b ≠ 0) →
  (m = 10 ∧ b = -12) :=
sorry

end line_equation_proof_l782_78293


namespace ball_box_arrangements_l782_78210

/-- The number of distinct balls -/
def num_balls : ℕ := 4

/-- The number of distinct boxes -/
def num_boxes : ℕ := 4

/-- The number of arrangements when exactly one box remains empty -/
def arrangements_one_empty : ℕ := 144

/-- The number of arrangements when exactly two boxes remain empty -/
def arrangements_two_empty : ℕ := 84

/-- Theorem stating the correct number of arrangements for each case -/
theorem ball_box_arrangements :
  (∀ (n : ℕ), n = num_balls → n = num_boxes) →
  (arrangements_one_empty = 144 ∧ arrangements_two_empty = 84) := by
  sorry

end ball_box_arrangements_l782_78210


namespace sum_of_fractions_equals_sixteen_l782_78241

theorem sum_of_fractions_equals_sixteen :
  let fractions : List ℚ := [2/10, 4/10, 6/10, 8/10, 10/10, 15/10, 20/10, 25/10, 30/10, 40/10]
  fractions.sum = 16 := by
  sorry

end sum_of_fractions_equals_sixteen_l782_78241


namespace max_correct_answers_l782_78233

theorem max_correct_answers (total_questions : ℕ) (correct_score : ℤ) (incorrect_score : ℤ) (total_score : ℤ) :
  total_questions = 25 →
  correct_score = 4 →
  incorrect_score = -3 →
  total_score = 57 →
  ∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    correct_score * correct + incorrect_score * incorrect = total_score ∧
    correct ≤ 18 ∧
    ∀ c i u : ℕ,
      c + i + u = total_questions →
      correct_score * c + incorrect_score * i = total_score →
      c ≤ correct :=
by sorry

end max_correct_answers_l782_78233


namespace surface_area_of_S_l782_78250

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Represents the solid S' formed by removing a tunnel from the cube -/
structure Solid where
  cube : Cube
  tunnelStart : Point3D
  tunnelEnd : Point3D

/-- Calculate the surface area of the solid S' -/
def surfaceAreaS' (s : Solid) : ℝ :=
  sorry

theorem surface_area_of_S' (c : Cube) (e i j k : Point3D) :
  c.sideLength = 12 ∧
  e.x = 12 ∧ e.y = 12 ∧ e.z = 12 ∧
  i.x = 9 ∧ i.y = 12 ∧ i.z = 12 ∧
  j.x = 12 ∧ j.y = 9 ∧ j.z = 12 ∧
  k.x = 12 ∧ k.y = 12 ∧ k.z = 9 →
  surfaceAreaS' { cube := c, tunnelStart := i, tunnelEnd := k } = 840 + 45 * Real.sqrt 2 := by
  sorry

end surface_area_of_S_l782_78250


namespace problem_statement_l782_78214

theorem problem_statement : Real.rpow 81 0.25 * Real.rpow 81 0.2 = 9 := by sorry

end problem_statement_l782_78214


namespace inscribed_prism_properties_l782_78295

/-- Regular triangular pyramid with inscribed regular triangular prism -/
structure PyramidWithPrism where
  pyramid_height : ℝ
  pyramid_base_side : ℝ
  prism_lateral_area : ℝ

/-- Possible solutions for the inscribed prism -/
structure PrismSolution where
  prism_height : ℝ
  lateral_area_ratio : ℝ

/-- Theorem stating the properties of the inscribed prism -/
theorem inscribed_prism_properties (p : PyramidWithPrism) 
  (h1 : p.pyramid_height = 15)
  (h2 : p.pyramid_base_side = 12)
  (h3 : p.prism_lateral_area = 120) :
  ∃ (s1 s2 : PrismSolution),
    (s1.prism_height = 10 ∧ s1.lateral_area_ratio = 1/9) ∧
    (s2.prism_height = 5 ∧ s2.lateral_area_ratio = 4/9) :=
sorry

end inscribed_prism_properties_l782_78295


namespace fifteen_blue_points_l782_78227

/-- Represents the configuration of points on a line -/
structure LineConfiguration where
  red_points : Fin 2 → ℕ
  blue_left : Fin 2 → ℕ
  blue_right : Fin 2 → ℕ

/-- The number of segments containing a red point with blue endpoints -/
def segments_count (config : LineConfiguration) (i : Fin 2) : ℕ :=
  config.blue_left i * config.blue_right i

/-- The total number of blue points -/
def total_blue_points (config : LineConfiguration) : ℕ :=
  config.blue_left 0 + config.blue_right 0

/-- Theorem stating that there are exactly 15 blue points -/
theorem fifteen_blue_points (config : LineConfiguration) 
  (h1 : segments_count config 0 = 56)
  (h2 : segments_count config 1 = 50)
  (h3 : config.blue_left 0 + config.blue_right 0 = config.blue_left 1 + config.blue_right 1) :
  total_blue_points config = 15 := by
  sorry


end fifteen_blue_points_l782_78227


namespace quadratic_solution_sum_l782_78287

theorem quadratic_solution_sum (a b : ℕ+) (x : ℝ) :
  x^2 + 16*x = 100 ∧ x > 0 ∧ x = Real.sqrt a - b → a + b = 172 := by
  sorry

end quadratic_solution_sum_l782_78287


namespace min_value_at_seven_l782_78286

/-- The quadratic function f(x) = x^2 - 14x + 40 -/
def f (x : ℝ) : ℝ := x^2 - 14*x + 40

theorem min_value_at_seven :
  ∀ x : ℝ, f 7 ≤ f x :=
sorry

end min_value_at_seven_l782_78286


namespace sequence_sum_equals_29_l782_78245

def sequence_term (n : ℕ) : ℤ :=
  if n % 2 = 0 then 2 + 3 * (n - 1) else -(5 + 3 * (n - 2))

def sequence_length : ℕ := 19

theorem sequence_sum_equals_29 :
  (Finset.range sequence_length).sum (λ i => sequence_term i) = 29 :=
by sorry

end sequence_sum_equals_29_l782_78245


namespace triangle_inequality_l782_78262

/-- The inequality for triangle sides and area -/
theorem triangle_inequality (a b c : ℝ) (Δ : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_area : Δ > 0) : 
  a^2 + b^2 + c^2 ≥ 4 * Real.sqrt 3 * Δ + (a - b)^2 + (b - c)^2 + (c - a)^2 ∧
  (a^2 + b^2 + c^2 = 4 * Real.sqrt 3 * Δ + (a - b)^2 + (b - c)^2 + (c - a)^2 ↔ a = b ∧ b = c) := by
  sorry


end triangle_inequality_l782_78262


namespace expand_expression_l782_78223

theorem expand_expression (x y : ℝ) :
  5 * (3 * x^3 - 4 * x * y + x^2 - y^2) = 15 * x^3 - 20 * x * y + 5 * x^2 - 5 * y^2 := by
  sorry

end expand_expression_l782_78223


namespace trig_identity_l782_78248

theorem trig_identity : Real.sin (63 * π / 180) * Real.cos (18 * π / 180) + 
  Real.cos (63 * π / 180) * Real.cos (108 * π / 180) = Real.sqrt 2 / 2 := by
  sorry

end trig_identity_l782_78248


namespace g_of_4_equals_26_l782_78238

-- Define the function g
def g (x : ℝ) : ℝ := 5 * x + 6

-- Theorem statement
theorem g_of_4_equals_26 : g 4 = 26 := by
  sorry

end g_of_4_equals_26_l782_78238


namespace lcm_45_75_180_l782_78285

theorem lcm_45_75_180 : Nat.lcm 45 (Nat.lcm 75 180) = 900 := by
  sorry

end lcm_45_75_180_l782_78285


namespace lucius_weekly_earnings_l782_78288

/-- Lucius's small business model --/
structure Business where
  daily_ingredient_cost : ℝ
  french_fries_price : ℝ
  poutine_price : ℝ
  tax_rate : ℝ
  daily_french_fries_sold : ℝ
  daily_poutine_sold : ℝ

/-- Calculate weekly earnings after taxes and expenses --/
def weekly_earnings_after_taxes_and_expenses (b : Business) : ℝ :=
  let daily_revenue := b.french_fries_price * b.daily_french_fries_sold + b.poutine_price * b.daily_poutine_sold
  let weekly_revenue := daily_revenue * 7
  let weekly_expenses := b.daily_ingredient_cost * 7
  let taxable_income := weekly_revenue
  let tax := taxable_income * b.tax_rate
  weekly_revenue - weekly_expenses - tax

/-- Theorem stating Lucius's weekly earnings --/
theorem lucius_weekly_earnings :
  ∃ (b : Business),
    b.daily_ingredient_cost = 10 ∧
    b.french_fries_price = 12 ∧
    b.poutine_price = 8 ∧
    b.tax_rate = 0.1 ∧
    b.daily_french_fries_sold = 1 ∧
    b.daily_poutine_sold = 1 ∧
    weekly_earnings_after_taxes_and_expenses b = 56 := by
  sorry


end lucius_weekly_earnings_l782_78288


namespace house_sale_profit_rate_l782_78221

/-- The profit rate calculation for a house sale with discount, price increase, and inflation -/
theorem house_sale_profit_rate 
  (list_price : ℝ) 
  (discount_rate : ℝ) 
  (price_increase_rate : ℝ) 
  (inflation_rate : ℝ) 
  (h1 : discount_rate = 0.05)
  (h2 : price_increase_rate = 0.60)
  (h3 : inflation_rate = 0.40) : 
  ∃ (profit_rate : ℝ), 
    abs (profit_rate - ((1 + price_increase_rate) / ((1 - discount_rate) * (1 + inflation_rate)) - 1)) < 0.001 ∧ 
    abs (profit_rate - 0.203) < 0.001 := by
  sorry

end house_sale_profit_rate_l782_78221


namespace smallest_n_doughnuts_l782_78207

theorem smallest_n_doughnuts : ∃ n : ℕ, n > 0 ∧ 
  (∀ m : ℕ, m > 0 → (15 * m - 1) % 5 = 0 → m ≥ n) ∧
  (15 * n - 1) % 5 = 0 :=
by sorry

end smallest_n_doughnuts_l782_78207


namespace not_suff_not_nec_condition_l782_78232

theorem not_suff_not_nec_condition (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  ¬(∀ a b, a > b → 1/a < 1/b) ∧ ¬(∀ a b, 1/a < 1/b → a > b) := by
  sorry

end not_suff_not_nec_condition_l782_78232


namespace smallest_m_no_real_roots_l782_78263

theorem smallest_m_no_real_roots : 
  ∃ (m : ℤ), (∀ (n : ℤ), n < m → ∃ (x : ℝ), 3*x*(n*x-5) - x^2 + 8 = 0) ∧
             (∀ (x : ℝ), 3*x*(m*x-5) - x^2 + 8 ≠ 0) ∧
             m = 3 := by
  sorry

end smallest_m_no_real_roots_l782_78263


namespace sum_of_roots_l782_78205

theorem sum_of_roots (r s : ℝ) : 
  (r ≠ s) → 
  (2 * (r^2 + 1/r^2) - 3 * (r + 1/r) = 1) → 
  (2 * (s^2 + 1/s^2) - 3 * (s + 1/s) = 1) → 
  (r + s = -5/2) := by
sorry

end sum_of_roots_l782_78205


namespace double_discount_l782_78246

/-- Calculates the final price as a percentage of the original price after applying two consecutive discounts -/
theorem double_discount (initial_discount coupon_discount : ℝ) :
  initial_discount = 0.4 →
  coupon_discount = 0.25 →
  (1 - initial_discount) * (1 - coupon_discount) = 0.45 :=
by
  sorry

#check double_discount

end double_discount_l782_78246


namespace smallest_k_for_inequality_l782_78228

theorem smallest_k_for_inequality : ∃ k : ℕ, k = 8 ∧ 
  (∀ w x y z : ℝ, (w^2 + x^2 + y^2 + z^2)^3 ≤ k * (w^6 + x^6 + y^6 + z^6)) ∧
  (∀ k' : ℕ, k' < k → 
    ∃ w x y z : ℝ, (w^2 + x^2 + y^2 + z^2)^3 > k' * (w^6 + x^6 + y^6 + z^6)) :=
by sorry

end smallest_k_for_inequality_l782_78228


namespace raffle_ticket_sales_difference_l782_78278

theorem raffle_ticket_sales_difference :
  ∀ (friday_sales saturday_sales sunday_sales : ℕ),
    friday_sales = 181 →
    saturday_sales = 2 * friday_sales →
    sunday_sales = 78 →
    saturday_sales - sunday_sales = 284 :=
by
  sorry

end raffle_ticket_sales_difference_l782_78278


namespace quadratic_equation_m_value_l782_78230

/-- 
Given that (m+2)x^(m^2-2) + 2x + 1 = 0 is a quadratic equation in x and m+2 ≠ 0, 
prove that m = 2.
-/
theorem quadratic_equation_m_value (m : ℝ) : 
  (∀ x, ∃ a b c : ℝ, (m + 2) * x^(m^2 - 2) + 2*x + 1 = a*x^2 + b*x + c) →
  (m + 2 ≠ 0) →
  m = 2 := by
sorry

end quadratic_equation_m_value_l782_78230


namespace eggs_needed_is_84_l782_78226

/-- Represents the number of eggs in an omelette -/
inductive OmeletteType
  | threeEgg
  | fourEgg

/-- Represents an hour of operation at the cafe -/
structure Hour where
  customers : Nat
  omeletteType : OmeletteType

/-- Represents a day of operation at Theo's cafe -/
structure CafeDay where
  hours : List Hour

/-- Calculates the total number of eggs needed for a given hour -/
def eggsNeededForHour (hour : Hour) : Nat :=
  match hour.omeletteType with
  | OmeletteType.threeEgg => 3 * hour.customers
  | OmeletteType.fourEgg => 4 * hour.customers

/-- Calculates the total number of eggs needed for the entire day -/
def totalEggsNeeded (day : CafeDay) : Nat :=
  day.hours.foldl (fun acc hour => acc + eggsNeededForHour hour) 0

/-- Theorem stating that the total number of eggs needed is 84 -/
theorem eggs_needed_is_84 (day : CafeDay) 
    (h1 : day.hours = [
      { customers := 5, omeletteType := OmeletteType.threeEgg },
      { customers := 7, omeletteType := OmeletteType.fourEgg },
      { customers := 3, omeletteType := OmeletteType.threeEgg },
      { customers := 8, omeletteType := OmeletteType.fourEgg }
    ]) : 
    totalEggsNeeded day = 84 := by
  sorry


end eggs_needed_is_84_l782_78226


namespace right_triangle_area_l782_78283

/-- Given a right triangle with perimeter 4 + √26 and median length 2 on the hypotenuse, its area is 5/2 -/
theorem right_triangle_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- Positive side lengths
  a^2 + b^2 = c^2 →  -- Right triangle (Pythagorean theorem)
  a + b + c = 4 + Real.sqrt 26 →  -- Perimeter condition
  c / 2 = 2 →  -- Median length condition
  (1/2) * a * b = 5/2 := by  -- Area of the triangle
sorry

end right_triangle_area_l782_78283


namespace probability_both_red_probability_different_colors_l782_78254

/-- Represents the color of a ball -/
inductive Color
| Red
| Black

/-- Represents a ball with a label -/
structure Ball where
  color : Color
  label : String

/-- The set of all balls in the bag -/
def bag : Finset Ball := sorry

/-- The number of balls in the bag -/
def total_balls : Nat := 6

/-- The number of red balls -/
def red_balls : Nat := 4

/-- The number of black balls -/
def black_balls : Nat := 2

/-- The set of all possible combinations when drawing 2 balls -/
def all_combinations : Finset (Ball × Ball) := sorry

/-- The set of combinations where both balls are red -/
def both_red : Finset (Ball × Ball) := sorry

/-- The set of combinations where the balls have different colors -/
def different_colors : Finset (Ball × Ball) := sorry

theorem probability_both_red :
  (Finset.card both_red : ℚ) / (Finset.card all_combinations : ℚ) = 2 / 5 := by sorry

theorem probability_different_colors :
  (Finset.card different_colors : ℚ) / (Finset.card all_combinations : ℚ) = 8 / 15 := by sorry

end probability_both_red_probability_different_colors_l782_78254


namespace jordan_rectangle_width_l782_78292

theorem jordan_rectangle_width (carol_length carol_width jordan_length : ℝ)
  (h1 : carol_length = 15)
  (h2 : carol_width = 24)
  (h3 : jordan_length = 8)
  (h4 : carol_length * carol_width = jordan_length * jordan_width) :
  jordan_width = 45 :=
by
  sorry


end jordan_rectangle_width_l782_78292


namespace hillarys_weekend_reading_l782_78258

/-- The total reading time assigned for a weekend, given the reading times for Friday, Saturday, and Sunday. -/
def weekend_reading_time (friday_time saturday_time sunday_time : ℕ) : ℕ :=
  friday_time + saturday_time + sunday_time

/-- Theorem stating that the total reading time for Hillary's weekend assignment is 60 minutes. -/
theorem hillarys_weekend_reading : weekend_reading_time 16 28 16 = 60 := by
  sorry

end hillarys_weekend_reading_l782_78258


namespace tangent_line_equation_l782_78200

/-- The equation of the tangent line to the curve y = x³ + 2x at the point (1, 3) is 5x - y - 2 = 0 -/
theorem tangent_line_equation (f : ℝ → ℝ) (x₀ y₀ : ℝ) :
  f x = x^3 + 2*x →
  f x₀ = y₀ →
  x₀ = 1 →
  y₀ = 3 →
  ∃ (m b : ℝ), ∀ x y, y = m*x + b ↔ 5*x - y - 2 = 0 :=
by sorry

end tangent_line_equation_l782_78200


namespace nth_equation_l782_78259

theorem nth_equation (n : ℕ+) : 9 * (n - 1) + n = 10 * n - 9 := by
  sorry

end nth_equation_l782_78259


namespace horner_V₁_value_l782_78203

/-- Horner's method for polynomial evaluation -/
def horner_step (a : ℝ) (x : ℝ) (v : ℝ) : ℝ := v * x + a

/-- The polynomial f(x) = 3x⁴ + 2x² + x + 4 -/
def f (x : ℝ) : ℝ := 3 * x^4 + 2 * x^2 + x + 4

/-- V₁ in Horner's method for f(x) at x = 10 -/
def V₁ : ℝ := horner_step 2 10 3

theorem horner_V₁_value : V₁ = 32 := by sorry

end horner_V₁_value_l782_78203


namespace integral_fractional_parts_sum_l782_78268

theorem integral_fractional_parts_sum (x y : ℝ) : 
  (x = ⌊5 - 2 * Real.sqrt 3⌋) → 
  (y = (5 - 2 * Real.sqrt 3) - ⌊5 - 2 * Real.sqrt 3⌋) → 
  (x + y + 4 / y = 9) := by
  sorry

end integral_fractional_parts_sum_l782_78268


namespace new_average_after_joining_l782_78219

theorem new_average_after_joining (initial_count : ℕ) (initial_average : ℚ) (new_member_amount : ℚ) :
  initial_count = 7 →
  initial_average = 14 →
  new_member_amount = 56 →
  (initial_count * initial_average + new_member_amount) / (initial_count + 1) = 19.25 := by
  sorry

end new_average_after_joining_l782_78219


namespace negation_of_universal_statement_l782_78277

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 + x + 1 < 0) ↔ (∃ x : ℝ, x^2 + x + 1 ≥ 0) :=
by sorry

end negation_of_universal_statement_l782_78277


namespace jasons_shopping_l782_78234

theorem jasons_shopping (total_spent jacket_cost : ℝ) 
  (h1 : total_spent = 14.28)
  (h2 : jacket_cost = 4.74) :
  total_spent - jacket_cost = 9.54 := by
sorry

end jasons_shopping_l782_78234


namespace similar_triangles_height_l782_78206

/-- Given two similar triangles with an area ratio of 1:9 and the height of the smaller triangle is 5 cm,
    prove that the corresponding height of the larger triangle is 15 cm. -/
theorem similar_triangles_height (h_small : ℝ) (area_ratio : ℝ) :
  h_small = 5 →
  area_ratio = 9 →
  ∃ h_large : ℝ, h_large = 15 ∧ h_large / h_small = Real.sqrt area_ratio :=
by sorry

end similar_triangles_height_l782_78206


namespace arithmetic_sequence_difference_l782_78208

/-- Given an arithmetic sequence with first term -3 and second term 5,
    the positive difference between the 1010th and 1000th terms is 80. -/
theorem arithmetic_sequence_difference : ∀ (a : ℕ → ℤ),
  (a 1 = -3) →
  (a 2 = 5) →
  (∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) →
  |a 1010 - a 1000| = 80 := by
  sorry

end arithmetic_sequence_difference_l782_78208


namespace hypotenuse_length_16_l782_78297

/-- A right triangle with one angle of 30 degrees -/
structure RightTriangle30 where
  /-- The length of the side opposite to the 30° angle -/
  short_side : ℝ
  /-- The short side is positive -/
  short_side_pos : 0 < short_side

/-- The length of the hypotenuse in a right triangle with a 30° angle -/
def hypotenuse (t : RightTriangle30) : ℝ := 2 * t.short_side

/-- Theorem: In a right triangle with a 30° angle, if the short side is 8, then the hypotenuse is 16 -/
theorem hypotenuse_length_16 (t : RightTriangle30) (h : t.short_side = 8) : 
  hypotenuse t = 16 := by sorry

end hypotenuse_length_16_l782_78297


namespace distance_to_origin_l782_78276

def z₁ : ℂ := Complex.I
def z₂ : ℂ := 1 + Complex.I

theorem distance_to_origin (z : ℂ := z₁ * z₂) :
  Complex.abs z = Real.sqrt 2 := by sorry

end distance_to_origin_l782_78276


namespace max_k_value_l782_78264

theorem max_k_value (k : ℤ) : 
  (∃ x : ℕ+, k * x.val - 5 = 2021 * x.val + 2 * k) → k ≤ 6068 :=
by sorry

end max_k_value_l782_78264


namespace smallest_x_absolute_value_equation_l782_78253

theorem smallest_x_absolute_value_equation :
  ∃ x : ℚ, (∀ y : ℚ, |5 * y - 3| = 45 → x ≤ y) ∧ |5 * x - 3| = 45 :=
by sorry

end smallest_x_absolute_value_equation_l782_78253


namespace pizza_area_increase_l782_78296

theorem pizza_area_increase (d₁ d₂ d₃ : ℝ) (h₁ : d₁ = 8) (h₂ : d₂ = 10) (h₃ : d₃ = 14) :
  let area (d : ℝ) := Real.pi * (d / 2)^2
  let percent_increase (a₁ a₂ : ℝ) := (a₂ - a₁) / a₁ * 100
  (percent_increase (area d₁) (area d₂) = 56.25) ∧
  (percent_increase (area d₂) (area d₃) = 96) := by
  sorry

end pizza_area_increase_l782_78296


namespace salary_increase_percentage_l782_78271

theorem salary_increase_percentage (x : ℝ) : 
  (((100 + x) / 100) * 0.8 = 1.04) → x = 30 := by
  sorry

end salary_increase_percentage_l782_78271


namespace ellipse_equation_correct_l782_78224

/-- An ellipse with foci (-1,0) and (1,0), passing through (2,0) -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

/-- The foci of the ellipse -/
def foci : (ℝ × ℝ) × (ℝ × ℝ) := ((-1, 0), (1, 0))

/-- A point on the ellipse -/
def P : ℝ × ℝ := (2, 0)

theorem ellipse_equation_correct :
  (∀ p ∈ Ellipse, 
    (Real.sqrt ((p.1 - foci.1.1)^2 + (p.2 - foci.1.2)^2) + 
     Real.sqrt ((p.1 - foci.2.1)^2 + (p.2 - foci.2.2)^2)) = 
    (Real.sqrt ((P.1 - foci.1.1)^2 + (P.2 - foci.1.2)^2) + 
     Real.sqrt ((P.1 - foci.2.1)^2 + (P.2 - foci.2.2)^2))) ∧
  P ∈ Ellipse := by
  sorry

#check ellipse_equation_correct

end ellipse_equation_correct_l782_78224


namespace total_athletes_l782_78299

/-- Given the ratio of players and the number of basketball players, 
    calculate the total number of athletes -/
theorem total_athletes (football baseball soccer basketball : ℕ) 
  (h_ratio : football = 10 ∧ baseball = 7 ∧ soccer = 5 ∧ basketball = 4)
  (h_basketball_players : basketball * 4 = 16) : 
  football * 4 + baseball * 4 + soccer * 4 + basketball * 4 = 104 := by
  sorry

#check total_athletes

end total_athletes_l782_78299


namespace n_pointed_star_angle_sum_l782_78270

/-- Represents an n-pointed star created from a convex n-gon. -/
structure NPointedStar where
  n : ℕ
  n_ge_7 : n ≥ 7

/-- The sum of interior angles at the n intersection points of an n-pointed star. -/
def interior_angle_sum (star : NPointedStar) : ℝ :=
  180 * (star.n - 2)

/-- Theorem stating that the sum of interior angles at the n intersection points
    of an n-pointed star is 180°(n-2). -/
theorem n_pointed_star_angle_sum (star : NPointedStar) :
  interior_angle_sum star = 180 * (star.n - 2) := by
  sorry

end n_pointed_star_angle_sum_l782_78270


namespace linear_function_properties_l782_78249

-- Define the linear function
def f (x : ℝ) : ℝ := -2 * x + 3

-- Theorem statement
theorem linear_function_properties :
  (f 1 = 1) ∧ 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ f x = y) ∧
  (∃ (x y : ℝ), x < 0 ∧ y > 0 ∧ f x = y) ∧
  (∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ f x = y) ∧
  (f (3/2) = 0) ∧
  (∀ (x₁ x₂ : ℝ), x₁ < x₂ → f x₁ > f x₂) :=
by sorry

end linear_function_properties_l782_78249


namespace circled_plus_two_three_four_l782_78209

/-- The operation ⊕ is defined for real numbers a, b, and c. -/
def CircledPlus (a b c : ℝ) : ℝ := b^2 - 3*a*c

/-- Theorem: The value of ⊕(2, 3, 4) is -15. -/
theorem circled_plus_two_three_four :
  CircledPlus 2 3 4 = -15 := by
  sorry

end circled_plus_two_three_four_l782_78209


namespace fifth_term_value_l782_78294

/-- A geometric sequence with common ratio 2 and positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (∀ n, a (n + 1) = 2 * a n)

theorem fifth_term_value (a : ℕ → ℝ) :
  GeometricSequence a → a 3 * a 11 = 16 → a 5 = 1 := by
  sorry


end fifth_term_value_l782_78294


namespace quadratic_inequality_theorem_l782_78242

def solution_set (a b c : ℝ) : Set ℝ :=
  {x : ℝ | x ≤ -3 ∨ x ≥ 4}

theorem quadratic_inequality_theorem (a b c : ℝ) 
  (h : solution_set a b c = {x : ℝ | a * x^2 + b * x + c ≥ 0}) :
  (a > 0) ∧ 
  ({x : ℝ | c * x^2 - b * x + a < 0} = {x : ℝ | x < -1/4 ∨ x > 1/3}) :=
sorry

end quadratic_inequality_theorem_l782_78242


namespace product_of_sums_equals_difference_of_powers_l782_78251

theorem product_of_sums_equals_difference_of_powers : 
  (5 + 2) * (5^3 + 2^3) * (5^9 + 2^9) * (5^27 + 2^27) * (5^81 + 2^81) = 5^128 - 2^128 := by
  sorry

end product_of_sums_equals_difference_of_powers_l782_78251
