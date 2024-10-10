import Mathlib

namespace complement_not_always_greater_l3092_309235

def complement (θ : ℝ) : ℝ := 90 - θ

theorem complement_not_always_greater : ∃ θ : ℝ, complement θ ≤ θ := by
  sorry

end complement_not_always_greater_l3092_309235


namespace area_equality_l3092_309260

-- Define the areas of various shapes
variable (S_Quadrilateral_BHCG : ℝ)
variable (S_Quadrilateral_AGDH : ℝ)
variable (S_Triangle_ABG : ℝ)
variable (S_Triangle_DCG : ℝ)
variable (S_Triangle_DEH : ℝ)
variable (S_Triangle_AFH : ℝ)
variable (S_Triangle_AOG : ℝ)
variable (S_Triangle_DOG : ℝ)
variable (S_Triangle_DOH : ℝ)
variable (S_Triangle_AOH : ℝ)
variable (S_Shaded : ℝ)
variable (S_Triangle_EFH : ℝ)
variable (S_Triangle_BCG : ℝ)

-- State the theorem
theorem area_equality 
  (h1 : S_Quadrilateral_BHCG / S_Quadrilateral_AGDH = 1 / 4)
  (h2 : S_Triangle_ABG + S_Triangle_DCG + S_Triangle_DEH + S_Triangle_AFH = 
        S_Triangle_AOG + S_Triangle_DOG + S_Triangle_DOH + S_Triangle_AOH)
  (h3 : S_Triangle_ABG + S_Triangle_DCG + S_Triangle_DEH + S_Triangle_AFH = S_Shaded)
  (h4 : S_Triangle_EFH + S_Triangle_BCG = S_Quadrilateral_BHCG)
  (h5 : S_Quadrilateral_BHCG = 1/4 * S_Shaded) :
  S_Quadrilateral_AGDH = S_Shaded :=
by sorry

end area_equality_l3092_309260


namespace last_term_is_one_l3092_309239

/-- A sequence is k-th order repeatable if there exist two sets of consecutive k terms that match in order. -/
def kth_order_repeatable (a : ℕ → Fin 2) (m k : ℕ) : Prop :=
  ∃ i j, i ≠ j ∧ i + k ≤ m ∧ j + k ≤ m ∧ ∀ t, t < k → a (i + t) = a (j + t)

theorem last_term_is_one
  (a : ℕ → Fin 2)
  (m : ℕ)
  (h_m : m ≥ 3)
  (h_not_5th : ¬ kth_order_repeatable a m 5)
  (h_5th_after : ∀ b : Fin 2, kth_order_repeatable (Function.update a m b) (m + 1) 5)
  (h_a4 : a 4 ≠ 1) :
  a m = 1 :=
sorry

end last_term_is_one_l3092_309239


namespace fraction_multiplication_result_l3092_309294

theorem fraction_multiplication_result : (3 / 4 : ℚ) * (1 / 2 : ℚ) * (2 / 5 : ℚ) * 5000 = 750 := by
  sorry

end fraction_multiplication_result_l3092_309294


namespace line_passes_through_point_with_slope_l3092_309276

/-- The slope of the line -/
def m : ℝ := 2

/-- The x-coordinate of the point P -/
def x₀ : ℝ := 3

/-- The y-coordinate of the point P -/
def y₀ : ℝ := 4

/-- The equation of the line passing through (x₀, y₀) with slope m -/
def line_equation (x y : ℝ) : Prop := 2 * x - y - 2 = 0

theorem line_passes_through_point_with_slope :
  line_equation x₀ y₀ ∧ 
  ∀ x y : ℝ, line_equation x y → (y - y₀) = m * (x - x₀) :=
sorry

end line_passes_through_point_with_slope_l3092_309276


namespace arithmetic_mean_of_fractions_l3092_309244

theorem arithmetic_mean_of_fractions :
  (3 : ℚ) / 7 + (5 : ℚ) / 9 = (31 : ℚ) / 63 := by
  sorry

end arithmetic_mean_of_fractions_l3092_309244


namespace florist_bouquets_l3092_309211

/-- The number of flower colors --/
def num_colors : ℕ := 4

/-- The number of flowers in each bouquet --/
def flowers_per_bouquet : ℕ := 9

/-- The number of seeds planted for each color --/
def seeds_per_color : ℕ := 125

/-- The number of red flowers killed by fungus --/
def red_killed : ℕ := 45

/-- The number of yellow flowers killed by fungus --/
def yellow_killed : ℕ := 61

/-- The number of orange flowers killed by fungus --/
def orange_killed : ℕ := 30

/-- The number of purple flowers killed by fungus --/
def purple_killed : ℕ := 40

/-- Theorem: The florist can make 36 bouquets --/
theorem florist_bouquets :
  (num_colors * seeds_per_color - (red_killed + yellow_killed + orange_killed + purple_killed)) / flowers_per_bouquet = 36 :=
by sorry

end florist_bouquets_l3092_309211


namespace consumption_decrease_l3092_309226

/-- Represents a country with its production capabilities -/
structure Country where
  zucchini : ℕ
  cauliflower : ℕ

/-- Calculates the total consumption of each crop under free trade -/
def freeTradeTotalConsumption (a b : Country) : ℕ := by
  sorry

/-- Calculates the total consumption of each crop under autarky -/
def autarkyTotalConsumption (a b : Country) : ℕ := by
  sorry

/-- Theorem stating that consumption decreases by 4 tons when countries merge and trade is banned -/
theorem consumption_decrease (a b : Country) 
  (h1 : a.zucchini = 20 ∧ a.cauliflower = 16)
  (h2 : b.zucchini = 36 ∧ b.cauliflower = 24) :
  freeTradeTotalConsumption a b - autarkyTotalConsumption a b = 4 := by
  sorry

end consumption_decrease_l3092_309226


namespace cubic_polynomial_with_irrational_product_of_roots_l3092_309283

theorem cubic_polynomial_with_irrational_product_of_roots :
  ∃ (a b c : ℚ) (u v : ℝ),
    (u^3 + a*u^2 + b*u + c = 0) ∧
    (v^3 + a*v^2 + b*v + c = 0) ∧
    ((u*v)^3 + a*(u*v)^2 + b*(u*v) + c = 0) ∧
    ¬(∃ (q : ℚ), u*v = q) := by
  sorry

end cubic_polynomial_with_irrational_product_of_roots_l3092_309283


namespace triangle_side_relation_l3092_309265

theorem triangle_side_relation (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) (h_angle : Real.cos (2 * Real.pi / 3) = -1/2) :
  a^2 + a*c + c^2 - b^2 = 0 := by sorry

end triangle_side_relation_l3092_309265


namespace ellipse_equation_l3092_309286

/-- An ellipse with a line passing through its vertex and focus -/
structure EllipseWithLine where
  /-- The semi-major axis length of the ellipse -/
  a : ℝ
  /-- The semi-minor axis length of the ellipse -/
  b : ℝ
  /-- Condition that a > b > 0 -/
  h1 : a > b ∧ b > 0
  /-- The line equation x - 2y + 4 = 0 -/
  line_eq : ℝ → ℝ → Prop := fun x y => x - 2*y + 4 = 0
  /-- The line passes through a vertex and focus of the ellipse -/
  line_through_vertex_focus : ∃ (x₁ y₁ x₂ y₂ : ℝ),
    line_eq x₁ y₁ ∧ line_eq x₂ y₂ ∧
    ((x₁^2 / a^2 + y₁^2 / b^2 = 1 ∧ (x₁ = a ∨ x₁ = -a ∨ y₁ = b ∨ y₁ = -b)) ∨
     (x₂^2 / a^2 + y₂^2 / b^2 > 1 ∧ x₂^2 - y₂^2 = a^2 - b^2))

/-- The theorem stating the standard equation of the ellipse -/
theorem ellipse_equation (e : EllipseWithLine) :
  ∀ (x y : ℝ), x^2/20 + y^2/4 = 1 ↔ x^2/e.a^2 + y^2/e.b^2 = 1 :=
sorry

end ellipse_equation_l3092_309286


namespace sum_transformed_sequence_formula_l3092_309216

/-- Given a sequence {aₙ} where the sum of its first n terms Sₙ satisfies 3Sₙ = 4^(n+1) - 4,
    this function computes the sum of the first n terms of the sequence {(3n-2)aₙ}. -/
def sumTransformedSequence (n : ℕ) (S : ℕ → ℝ) (h : ∀ n, 3 * S n = 4^(n+1) - 4) : ℝ :=
  4 + (n - 1 : ℝ) * 4^(n+1)

/-- Theorem stating that the sum of the first n terms of {(3n-2)aₙ} is 4 + (n-1) * 4^(n+1),
    given that the sum of the first n terms of {aₙ} satisfies 3Sₙ = 4^(n+1) - 4. -/
theorem sum_transformed_sequence_formula (n : ℕ) (S : ℕ → ℝ) (h : ∀ n, 3 * S n = 4^(n+1) - 4) :
  sumTransformedSequence n S h = 4 + (n - 1 : ℝ) * 4^(n+1) := by
  sorry

end sum_transformed_sequence_formula_l3092_309216


namespace remaining_time_is_three_l3092_309246

/-- Represents the time needed to finish plowing a field with two tractors -/
def time_to_finish (time_a time_b worked_time : ℚ) : ℚ :=
  let rate_a : ℚ := 1 / time_a
  let rate_b : ℚ := 1 / time_b
  let remaining_work : ℚ := 1 - (rate_a * worked_time)
  let combined_rate : ℚ := rate_a + rate_b
  remaining_work / combined_rate

/-- Theorem stating that the remaining time to finish plowing is 3 hours -/
theorem remaining_time_is_three :
  time_to_finish 20 15 13 = 3 := by
  sorry

end remaining_time_is_three_l3092_309246


namespace circle_area_from_polar_equation_l3092_309274

/-- The area of the circle described by the polar equation r = 3 cos θ - 4 sin θ is 25π/4 -/
theorem circle_area_from_polar_equation :
  let r : ℝ → ℝ := λ θ => 3 * Real.cos θ - 4 * Real.sin θ
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ θ, (r θ * Real.cos θ - center.1)^2 + (r θ * Real.sin θ - center.2)^2 = radius^2) ∧
    π * radius^2 = 25 * π / 4 := by
  sorry

end circle_area_from_polar_equation_l3092_309274


namespace all_paths_end_at_z_l3092_309230

-- Define the graph structure
structure DirectedGraph (V : Type) where
  edge : V → V → Prop
  distinct_edge : ∀ a b : V, edge a b → a ≠ b
  at_most_one : ∀ a b : V, Unique (edge a b)

-- Define the property mentioned in the problem
def has_common_target {V : Type} (G : DirectedGraph V) (x u v w : V) : Prop :=
  x ≠ u ∧ x ≠ v ∧ u ≠ v ∧ G.edge x u ∧ G.edge x v → ∃ w, G.edge u w ∧ G.edge v w

-- Define a path in the graph
def is_path {V : Type} (G : DirectedGraph V) : List V → Prop
  | [] => True
  | [_] => True
  | (a::b::rest) => G.edge a b ∧ is_path G (b::rest)

-- Define the length of a path
def path_length {V : Type} : List V → Nat
  | [] => 0
  | [_] => 0
  | (_::rest) => 1 + path_length rest

-- The main theorem
theorem all_paths_end_at_z {V : Type} (G : DirectedGraph V) (x z : V) (n : Nat) :
  (∀ a b c w : V, has_common_target G a b c w) →
  (∃ path : List V, is_path G path ∧ path.head? = some x ∧ path.getLast? = some z ∧ path_length path = n) →
  (∀ v : V, ¬G.edge z v) →
  (∀ path : List V, is_path G path ∧ path.head? = some x → path_length path = n ∧ path.getLast? = some z) :=
by sorry

end all_paths_end_at_z_l3092_309230


namespace smallest_Y_for_binary_multiple_of_15_l3092_309284

def is_binary_number (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 0 ∨ d = 1

theorem smallest_Y_for_binary_multiple_of_15 :
  (∃ U : ℕ, is_binary_number U ∧ U % 15 = 0 ∧ U = 15 * 74) ∧
  (∀ Y : ℕ, Y < 74 → ¬∃ U : ℕ, is_binary_number U ∧ U % 15 = 0 ∧ U = 15 * Y) :=
by sorry

end smallest_Y_for_binary_multiple_of_15_l3092_309284


namespace grid_routes_equal_binomial_coefficient_l3092_309202

def grid_width : ℕ := 10
def grid_height : ℕ := 5

def num_routes : ℕ := Nat.choose (grid_width + grid_height) grid_height

theorem grid_routes_equal_binomial_coefficient :
  num_routes = Nat.choose (grid_width + grid_height) grid_height :=
by sorry

end grid_routes_equal_binomial_coefficient_l3092_309202


namespace convention_handshakes_specific_l3092_309209

/-- Calculates the number of handshakes in a convention with representatives from multiple companies. -/
def convention_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Proves that in a convention with 5 representatives from each of 5 companies, 
    where representatives only shake hands with people from other companies, 
    the total number of handshakes is 250. -/
theorem convention_handshakes_specific : convention_handshakes 5 5 = 250 := by
  sorry

end convention_handshakes_specific_l3092_309209


namespace fraction_less_than_mode_l3092_309298

def data_list : List ℕ := [3, 4, 5, 5, 5, 5, 7, 11, 21]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

def count_less_than_mode (l : List ℕ) : ℕ :=
  l.filter (· < mode l) |>.length

theorem fraction_less_than_mode :
  (count_less_than_mode data_list : ℚ) / data_list.length = 2 / 9 := by
  sorry

end fraction_less_than_mode_l3092_309298


namespace triangle_inradius_l3092_309242

/-- The inradius of a triangle with side lengths 7, 11, and 14 is 3√10 / 4 -/
theorem triangle_inradius (a b c : ℝ) (ha : a = 7) (hb : b = 11) (hc : c = 14) :
  let s := (a + b + c) / 2
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  A / s = (3 * Real.sqrt 10) / 4 := by sorry

end triangle_inradius_l3092_309242


namespace candy_bar_sales_earnings_candy_bar_sales_proof_l3092_309249

/-- Calculates the total amount earned from candy bar sales given the specified conditions --/
theorem candy_bar_sales_earnings (num_members : ℕ) (type_a_price type_b_price : ℚ) 
  (avg_total_bars avg_type_a avg_type_b : ℕ) : ℚ :=
  let total_bars := num_members * avg_total_bars
  let total_type_a := num_members * avg_type_a
  let total_type_b := num_members * avg_type_b
  let earnings_type_a := total_type_a * type_a_price
  let earnings_type_b := total_type_b * type_b_price
  earnings_type_a + earnings_type_b

/-- Proves that the group earned $95 from their candy bar sales --/
theorem candy_bar_sales_proof :
  candy_bar_sales_earnings 20 (1/2) (3/4) 8 5 3 = 95 := by
  sorry

end candy_bar_sales_earnings_candy_bar_sales_proof_l3092_309249


namespace angle_PDO_is_45_degrees_l3092_309228

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A square defined by its side length -/
structure Square where
  sideLength : ℝ

/-- Represents the configuration described in the problem -/
structure Configuration where
  outerSquare : Square
  L : Point
  P : Point
  O : Point

/-- Predicate to check if a point is on the diagonal of a square -/
def isOnDiagonal (s : Square) (p : Point) : Prop :=
  p.x = p.y ∧ 0 ≤ p.x ∧ p.x ≤ s.sideLength

/-- Predicate to check if a point is on the side of a square -/
def isOnSide (s : Square) (p : Point) : Prop :=
  p.y = 0 ∧ 0 ≤ p.x ∧ p.x ≤ s.sideLength

/-- Calculate the angle between three points in degrees -/
def angleBetween (p1 p2 p3 : Point) : ℝ := sorry

/-- The main theorem -/
theorem angle_PDO_is_45_degrees (c : Configuration) 
  (h1 : isOnDiagonal c.outerSquare c.L)
  (h2 : isOnSide c.outerSquare c.P)
  (h3 : c.O.x = (c.L.x + c.outerSquare.sideLength) / 2)
  (h4 : c.O.y = (c.L.y + c.outerSquare.sideLength) / 2) :
  angleBetween c.P (Point.mk 0 c.outerSquare.sideLength) c.O = 45 := by
  sorry

end angle_PDO_is_45_degrees_l3092_309228


namespace binary_110011_equals_51_l3092_309272

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

theorem binary_110011_equals_51 :
  binary_to_decimal [true, true, false, false, true, true] = 51 := by sorry

end binary_110011_equals_51_l3092_309272


namespace smallest_square_factor_l3092_309206

theorem smallest_square_factor (n : ℕ) (hn : n = 4410) :
  (∃ (y : ℕ), y > 0 ∧ ∃ (k : ℕ), n * y = k^2) ∧
  (∀ (z : ℕ), z > 0 → (∃ (k : ℕ), n * z = k^2) → z ≥ 10) := by
  sorry

end smallest_square_factor_l3092_309206


namespace probability_at_least_one_from_subset_l3092_309248

def total_elements : ℕ := 4
def selected_elements : ℕ := 2
def subset_size : ℕ := 2

theorem probability_at_least_one_from_subset :
  (1 : ℚ) - (Nat.choose (total_elements - subset_size) selected_elements : ℚ) / 
  (Nat.choose total_elements selected_elements : ℚ) = 5/6 := by sorry

end probability_at_least_one_from_subset_l3092_309248


namespace total_flowers_l3092_309289

theorem total_flowers (num_pots : ℕ) (flowers_per_pot : ℕ) 
  (h1 : num_pots = 544) (h2 : flowers_per_pot = 32) : 
  num_pots * flowers_per_pot = 17408 := by
  sorry

end total_flowers_l3092_309289


namespace dog_food_cans_per_package_adam_dog_food_cans_l3092_309254

theorem dog_food_cans_per_package (cat_packages : Nat) (dog_packages : Nat) 
  (cat_cans_per_package : Nat) (extra_cat_cans : Nat) : Nat :=
  let total_cat_cans := cat_packages * cat_cans_per_package
  let dog_cans_per_package := (total_cat_cans - extra_cat_cans) / dog_packages
  dog_cans_per_package

/-- The number of cans in each package of dog food is 5. -/
theorem adam_dog_food_cans : dog_food_cans_per_package 9 7 10 55 = 5 := by
  sorry

end dog_food_cans_per_package_adam_dog_food_cans_l3092_309254


namespace first_equation_is_midpoint_second_equation_is_midpoint_iff_l3092_309207

/-- Definition of a midpoint equation -/
def is_midpoint_equation (a b : ℚ) : Prop :=
  a ≠ 0 ∧ ((-b) / a = (a + b) / 2)

/-- First part of the problem -/
theorem first_equation_is_midpoint : is_midpoint_equation 4 (-8/3) := by
  sorry

/-- Second part of the problem -/
theorem second_equation_is_midpoint_iff (m : ℚ) : 
  is_midpoint_equation 5 (m - 1) ↔ m = -18/7 := by
  sorry

end first_equation_is_midpoint_second_equation_is_midpoint_iff_l3092_309207


namespace simplify_expression_l3092_309221

theorem simplify_expression : 2^3 * 2^2 * 3^3 * 3^2 = 6^5 := by
  sorry

end simplify_expression_l3092_309221


namespace permutation_distinct_differences_l3092_309295

theorem permutation_distinct_differences (n : ℕ+) :
  (∃ (a : Fin n → Fin n), Function.Bijective a ∧
    (∀ (i j : Fin n), i ≠ j → |a i - i| ≠ |a j - j|)) ↔
  (∃ (k : ℕ), n = 4 * k ∨ n = 4 * k + 1) :=
by sorry

end permutation_distinct_differences_l3092_309295


namespace simplify_and_evaluate_l3092_309293

theorem simplify_and_evaluate (m n : ℝ) :
  (m + n)^2 - 2*m*(m + n) = n^2 - m^2 ∧
  (let m := 2; let n := -3; (m + n)^2 - 2*m*(m + n) = 5) :=
by sorry

end simplify_and_evaluate_l3092_309293


namespace softball_players_count_l3092_309291

theorem softball_players_count (cricket hockey football total : ℕ) 
  (h1 : cricket = 22)
  (h2 : hockey = 15)
  (h3 : football = 21)
  (h4 : total = 77) :
  total - (cricket + hockey + football) = 19 := by
  sorry

end softball_players_count_l3092_309291


namespace unique_sums_count_l3092_309229

def bag_C : Finset ℕ := {1, 3, 7, 9}
def bag_D : Finset ℕ := {4, 6, 8}

theorem unique_sums_count : 
  Finset.card ((bag_C.product bag_D).image (fun p => p.1 + p.2)) = 7 := by
  sorry

end unique_sums_count_l3092_309229


namespace smallest_four_digit_multiple_of_17_l3092_309204

theorem smallest_four_digit_multiple_of_17 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 17 ∣ n → n ≥ 1013 := by
  sorry

end smallest_four_digit_multiple_of_17_l3092_309204


namespace largest_band_size_l3092_309236

/-- Represents a rectangular band formation --/
structure BandFormation where
  rows : ℕ
  membersPerRow : ℕ

/-- The total number of members in a formation --/
def totalMembers (f : BandFormation) : ℕ := f.rows * f.membersPerRow

/-- The condition that the band has less than 150 members --/
def lessThan150 (f : BandFormation) : Prop := totalMembers f < 150

/-- The condition that there are 3 members left over in the original formation --/
def hasThreeLeftOver (f : BandFormation) (totalBandMembers : ℕ) : Prop :=
  totalMembers f + 3 = totalBandMembers

/-- The new formation with 2 more members per row and 3 fewer rows --/
def newFormation (f : BandFormation) : BandFormation :=
  { rows := f.rows - 3, membersPerRow := f.membersPerRow + 2 }

/-- The condition that the new formation fits all members exactly --/
def newFormationFitsExactly (f : BandFormation) (totalBandMembers : ℕ) : Prop :=
  totalMembers (newFormation f) = totalBandMembers

/-- The theorem stating that the largest possible number of band members is 108 --/
theorem largest_band_size :
  ∃ (f : BandFormation) (totalBandMembers : ℕ),
    lessThan150 f ∧
    hasThreeLeftOver f totalBandMembers ∧
    newFormationFitsExactly f totalBandMembers ∧
    totalBandMembers = 108 ∧
    (∀ (g : BandFormation) (m : ℕ),
      lessThan150 g →
      hasThreeLeftOver g m →
      newFormationFitsExactly g m →
      m ≤ 108) :=
  sorry


end largest_band_size_l3092_309236


namespace concrete_density_l3092_309213

/-- Concrete density problem -/
theorem concrete_density (num_homes : ℕ) (length width height : ℝ) (cost_per_pound : ℝ) (total_cost : ℝ)
  (h1 : num_homes = 3)
  (h2 : length = 100)
  (h3 : width = 100)
  (h4 : height = 0.5)
  (h5 : cost_per_pound = 0.02)
  (h6 : total_cost = 45000) :
  (total_cost / cost_per_pound) / (num_homes * length * width * height) = 150 := by
  sorry

end concrete_density_l3092_309213


namespace rectangle_area_is_144_l3092_309280

-- Define the radius of the circles
def circle_radius : ℝ := 3

-- Define the rectangle
structure Rectangle where
  length : ℝ
  width : ℝ

-- Define the property that circles touch the sides of the rectangle
def circles_touch_sides (r : Rectangle) : Prop :=
  r.length = 4 * circle_radius ∧ r.width = 4 * circle_radius

-- Define the area of the rectangle
def rectangle_area (r : Rectangle) : ℝ :=
  r.length * r.width

-- Theorem statement
theorem rectangle_area_is_144 (r : Rectangle) 
  (h : circles_touch_sides r) : rectangle_area r = 144 := by
  sorry

end rectangle_area_is_144_l3092_309280


namespace computation_proof_l3092_309267

theorem computation_proof : (143 + 29) * 2 + 25 + 13 = 382 := by
  sorry

end computation_proof_l3092_309267


namespace magnitude_of_vector_AB_l3092_309212

theorem magnitude_of_vector_AB (OA OB : ℝ × ℝ) : 
  OA = (Real.cos (15 * π / 180), Real.sin (15 * π / 180)) →
  OB = (Real.cos (75 * π / 180), Real.sin (75 * π / 180)) →
  Real.sqrt ((OB.1 - OA.1)^2 + (OB.2 - OA.2)^2) = 1 :=
by sorry

end magnitude_of_vector_AB_l3092_309212


namespace arithmetic_sequence_general_term_l3092_309238

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific arithmetic sequence from the problem
def specific_sequence (a : ℕ → ℝ) : Prop :=
  is_arithmetic_sequence a ∧ a 3 = 7 ∧ a 7 = 3

-- Theorem statement
theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) (h : specific_sequence a) : 
  ∀ n : ℕ, a n = -n + 10 :=
sorry

end arithmetic_sequence_general_term_l3092_309238


namespace diagonal_path_crosses_12_tiles_l3092_309240

/-- Represents a rectangular floor tiled with 1x2 foot tiles -/
structure TiledFloor where
  width : ℕ
  length : ℕ

/-- Calculates the number of tiles crossed by a diagonal path on a tiled floor -/
def tilesCrossed (floor : TiledFloor) : ℕ :=
  floor.width / Nat.gcd floor.width floor.length +
  floor.length / Nat.gcd floor.width floor.length - 1

/-- Theorem stating that a diagonal path on an 8x18 foot floor crosses 12 tiles -/
theorem diagonal_path_crosses_12_tiles :
  let floor : TiledFloor := { width := 8, length := 18 }
  tilesCrossed floor = 12 := by sorry

end diagonal_path_crosses_12_tiles_l3092_309240


namespace circle_equation_proof_l3092_309203

-- Define the given circle
def givenCircle (x y : ℝ) : Prop := x^2 + y^2 - 2*x + 4*y + 1 = 0

-- Define the given line
def givenLine (x y : ℝ) : Prop := 2*x - y + 1 = 0

-- Define the result circle
def resultCircle (x y : ℝ) : Prop := (x - 1)^2 + (y + 2)^2 = 5

-- Theorem statement
theorem circle_equation_proof :
  ∀ x y : ℝ, 
    (∃ c : ℝ, c > 0 ∧ 
      (∀ x' y' : ℝ, givenCircle x' y' ↔ (x' - 1)^2 + (y' + 2)^2 = c)) ∧ 
    (∃ x₀ y₀ : ℝ, givenLine x₀ y₀ ∧ resultCircle x₀ y₀) ∧
    (∀ x' y' : ℝ, givenLine x' y' → ¬(resultCircle x' y' ∧ ¬(x' = x₀ ∧ y' = y₀))) :=
by sorry

end circle_equation_proof_l3092_309203


namespace negation_of_exists_exponential_nonpositive_l3092_309296

theorem negation_of_exists_exponential_nonpositive :
  (¬ ∃ x : ℝ, Real.exp x ≤ 0) ↔ (∀ x : ℝ, Real.exp x > 0) := by sorry

end negation_of_exists_exponential_nonpositive_l3092_309296


namespace parallel_planes_condition_l3092_309273

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallelLines : Line → Line → Prop)
variable (parallelPlanes : Plane → Plane → Prop)
variable (perpendicularLineToPlane : Line → Plane → Prop)

-- State the theorem
theorem parallel_planes_condition 
  (m n : Line) (α β : Plane) :
  perpendicularLineToPlane m α →
  perpendicularLineToPlane n β →
  parallelLines m n →
  parallelPlanes α β :=
sorry

end parallel_planes_condition_l3092_309273


namespace digit_placement_ways_l3092_309225

/-- The number of corner boxes in a 3x3 grid -/
def num_corners : ℕ := 4

/-- The total number of boxes in a 3x3 grid -/
def total_boxes : ℕ := 9

/-- The number of digits to be placed -/
def num_digits : ℕ := 4

/-- The number of ways to place digits 1, 2, 3, and 4 in a 3x3 grid -/
def num_ways : ℕ := num_corners * (total_boxes - 1) * (total_boxes - 2) * (total_boxes - 3)

theorem digit_placement_ways :
  num_ways = 1344 :=
sorry

end digit_placement_ways_l3092_309225


namespace basketball_height_data_field_survey_l3092_309253

def HeightData := List Nat

def isFieldSurveyMethod (data : HeightData) : Prop :=
  data.all (λ h => h ≥ 150 ∧ h ≤ 200) ∧ 
  data.length > 0 ∧
  data.length ≤ 20

def basketballTeamHeights : HeightData :=
  [167, 168, 167, 164, 168, 168, 163, 168, 167, 160]

theorem basketball_height_data_field_survey :
  isFieldSurveyMethod basketballTeamHeights := by
  sorry

end basketball_height_data_field_survey_l3092_309253


namespace smallest_difference_fraction_l3092_309245

theorem smallest_difference_fraction :
  ∀ p q : ℕ, 
    0 < q → q < 1001 → 
    |123 / 1001 - (p : ℚ) / q| ≥ |123 / 1001 - 94 / 765| := by
  sorry

end smallest_difference_fraction_l3092_309245


namespace equation_and_inequality_solution_l3092_309227

theorem equation_and_inequality_solution :
  (∃ x : ℝ, 3 * (x - 2) - (1 - 2 * x) = 3 ∧ x = 2) ∧
  (∀ x : ℝ, 2 * x - 1 < 4 * x + 3 ↔ x > -2) := by
  sorry

end equation_and_inequality_solution_l3092_309227


namespace cot_thirteen_pi_fourths_l3092_309257

theorem cot_thirteen_pi_fourths : Real.cos (13 * π / 4) / Real.sin (13 * π / 4) = -1 := by
  sorry

end cot_thirteen_pi_fourths_l3092_309257


namespace vector_parallel_implies_m_l3092_309232

def vector_a (m : ℝ) : ℝ × ℝ := (1, m)
def vector_b : ℝ × ℝ := (-3, 1)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_parallel_implies_m (m : ℝ) :
  parallel ((2 * vector_a m).1 + vector_b.1, (2 * vector_a m).2 + vector_b.2) vector_b →
  m = -1/3 := by
sorry

end vector_parallel_implies_m_l3092_309232


namespace quadratic_expression_value_l3092_309231

/-- Given that 2a - b = -3, prove that 4a - 2b = -6 --/
theorem quadratic_expression_value (a b : ℝ) (h : 2 * a - b = -3) :
  4 * a - 2 * b = -6 := by
  sorry

end quadratic_expression_value_l3092_309231


namespace arithmetic_sequence_problem_l3092_309252

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h1 : is_arithmetic_sequence a) 
  (h2 : a 4 + a 7 + a 10 = 30) : 
  a 3 - 2 * a 5 = -10 := by
  sorry

end arithmetic_sequence_problem_l3092_309252


namespace first_month_bill_l3092_309217

/-- Represents the telephone bill structure -/
structure TelephoneBill where
  callCharge : ℝ
  internetCharge : ℝ
  totalCharge : ℝ
  totalCharge_eq : totalCharge = callCharge + internetCharge

/-- The telephone bill problem -/
theorem first_month_bill (
  firstMonth secondMonth : TelephoneBill
) (h1 : firstMonth.totalCharge = 46)
  (h2 : secondMonth.totalCharge = 76)
  (h3 : secondMonth.callCharge = 2 * firstMonth.callCharge)
  (h4 : firstMonth.internetCharge = secondMonth.internetCharge) :
  firstMonth.totalCharge = 46 := by
sorry

end first_month_bill_l3092_309217


namespace luke_trivia_game_l3092_309268

/-- Given a trivia game where a player gains a constant number of points per round
    and achieves a total score, calculate the number of rounds played. -/
def rounds_played (points_per_round : ℕ) (total_points : ℕ) : ℕ :=
  total_points / points_per_round

/-- Luke's trivia game scenario -/
theorem luke_trivia_game : rounds_played 3 78 = 26 := by
  sorry

end luke_trivia_game_l3092_309268


namespace fort_soldiers_count_l3092_309292

/-- The initial number of soldiers in the fort -/
def initial_soldiers : ℕ := 480

/-- The number of additional soldiers joining the fort -/
def additional_soldiers : ℕ := 528

/-- The number of days provisions last with initial soldiers -/
def initial_days : ℕ := 30

/-- The number of days provisions last with additional soldiers -/
def new_days : ℕ := 25

/-- The daily consumption per soldier initially (in kg) -/
def initial_consumption : ℚ := 3

/-- The daily consumption per soldier after additional soldiers join (in kg) -/
def new_consumption : ℚ := 5/2

theorem fort_soldiers_count :
  initial_soldiers * initial_consumption * initial_days =
  (initial_soldiers + additional_soldiers) * new_consumption * new_days :=
sorry

end fort_soldiers_count_l3092_309292


namespace darren_tshirts_l3092_309223

/-- The number of packs of white t-shirts Darren bought -/
def white_packs : ℕ := 5

/-- The number of packs of blue t-shirts Darren bought -/
def blue_packs : ℕ := 3

/-- The number of t-shirts in each pack of white t-shirts -/
def white_per_pack : ℕ := 6

/-- The number of t-shirts in each pack of blue t-shirts -/
def blue_per_pack : ℕ := 9

/-- The total number of t-shirts Darren bought -/
def total_tshirts : ℕ := white_packs * white_per_pack + blue_packs * blue_per_pack

theorem darren_tshirts : total_tshirts = 57 := by
  sorry

end darren_tshirts_l3092_309223


namespace fundraiser_result_l3092_309218

/-- Represents the fundraiser scenario with students bringing brownies, cookies, and donuts. -/
structure Fundraiser where
  brownie_students : ℕ
  brownies_per_student : ℕ
  cookie_students : ℕ
  cookies_per_student : ℕ
  donut_students : ℕ
  donuts_per_student : ℕ
  price_per_item : ℚ

/-- Calculates the total amount of money raised in the fundraiser. -/
def total_money_raised (f : Fundraiser) : ℚ :=
  ((f.brownie_students * f.brownies_per_student +
    f.cookie_students * f.cookies_per_student +
    f.donut_students * f.donuts_per_student) : ℚ) * f.price_per_item

/-- Theorem stating that the fundraiser with given conditions raises $2040.00. -/
theorem fundraiser_result : 
  let f : Fundraiser := {
    brownie_students := 30,
    brownies_per_student := 12,
    cookie_students := 20,
    cookies_per_student := 24,
    donut_students := 15,
    donuts_per_student := 12,
    price_per_item := 2
  }
  total_money_raised f = 2040 := by
  sorry


end fundraiser_result_l3092_309218


namespace max_value_sin_cos_sum_l3092_309222

theorem max_value_sin_cos_sum (a b : ℝ) :
  ∃ (M : ℝ), M = Real.sqrt (a^2 + b^2) ∧
  (∀ t : ℝ, 0 < t ∧ t < 2 * Real.pi → a * Real.sin t + b * Real.cos t ≤ M) ∧
  (∃ t : ℝ, 0 < t ∧ t < 2 * Real.pi ∧ a * Real.sin t + b * Real.cos t = M) :=
by sorry

end max_value_sin_cos_sum_l3092_309222


namespace unfair_coin_expected_value_l3092_309282

/-- Given an unfair coin with the following properties:
  * Probability of heads: 2/3
  * Probability of tails: 1/3
  * Gain on heads: $5
  * Loss on tails: $12
  This theorem proves that the expected value of a single coin flip
  is -2/3 dollars. -/
theorem unfair_coin_expected_value :
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let gain_heads : ℚ := 5
  let loss_tails : ℚ := 12
  p_heads * gain_heads + p_tails * (-loss_tails) = -2/3 := by
sorry

end unfair_coin_expected_value_l3092_309282


namespace max_sum_abc_l3092_309237

theorem max_sum_abc (a b c : ℤ) 
  (h1 : a + b = 2006) 
  (h2 : c - a = 2005) 
  (h3 : a < b) : 
  ∃ (m : ℤ), m = 5013 ∧ a + b + c ≤ m ∧ ∃ (a' b' c' : ℤ), a' + b' = 2006 ∧ c' - a' = 2005 ∧ a' < b' ∧ a' + b' + c' = m :=
sorry

end max_sum_abc_l3092_309237


namespace regular_polygon_inscribed_circle_l3092_309281

theorem regular_polygon_inscribed_circle (n : ℕ) (R : ℝ) (h : R > 0) :
  (1 / 2 : ℝ) * n * R^2 * Real.sin (2 * Real.pi / n) = 4 * R^2 → n = 8 := by
  sorry

end regular_polygon_inscribed_circle_l3092_309281


namespace hyperbola_intersection_theorem_l3092_309241

/-- Hyperbola C with equation x²/16 - y²/4 = 1 -/
def hyperbola_C (x y : ℝ) : Prop := x^2 / 16 - y^2 / 4 = 1

/-- Point P with coordinates (0, 3) -/
def point_P : ℝ × ℝ := (0, 3)

/-- Line l passing through point P -/
def line_l (k : ℝ) (x : ℝ) : ℝ := k * x + 3

/-- Condition for point A to be on the hyperbola C and line l -/
def point_A_condition (k : ℝ) (x y : ℝ) : Prop :=
  hyperbola_C x y ∧ y = line_l k x ∧ x > 0

/-- Condition for point B to be on the hyperbola C and line l -/
def point_B_condition (k : ℝ) (x y : ℝ) : Prop :=
  hyperbola_C x y ∧ y = line_l k x ∧ x > 0

/-- Condition for point D to be on line l -/
def point_D_condition (k : ℝ) (x y : ℝ) : Prop :=
  y = line_l k x ∧ (x, y) ≠ point_P

/-- Condition for the cross ratio equality |PA| * |DB| = |PB| * |DA| -/
def cross_ratio_condition (xa ya xb yb xd yd : ℝ) : Prop :=
  (xa - 0) * (xd - xb) = (xb - 0) * (xd - xa)

theorem hyperbola_intersection_theorem (k : ℝ) 
  (xa ya xb yb xd yd : ℝ) :
  point_A_condition k xa ya →
  point_B_condition k xb yb →
  point_D_condition k xd yd →
  (xa ≠ xb) →
  cross_ratio_condition xa ya xb yb xd yd →
  yd = -4/3 := by sorry


end hyperbola_intersection_theorem_l3092_309241


namespace smallest_number_of_weights_l3092_309277

/-- A function that determines if a given number of weights can measure all masses -/
def can_measure_all (n : ℕ) : Prop :=
  ∃ (weights : Fin n → ℝ), 
    (∀ i, weights i ≥ 0.01) ∧ 
    (∀ m : ℝ, 0 ≤ m ∧ m ≤ 20.2 → 
      ∃ (subset : Fin n → Bool), 
        abs (m - (Finset.sum (Finset.filter (λ i => subset i = true) Finset.univ) weights)) ≤ 0.01)

theorem smallest_number_of_weights : 
  (∀ k < 2020, ¬ can_measure_all k) ∧ can_measure_all 2020 :=
sorry

end smallest_number_of_weights_l3092_309277


namespace turner_rides_l3092_309285

theorem turner_rides (rollercoaster_rides : ℕ) (ferris_wheel_rides : ℕ) 
  (rollercoaster_cost : ℕ) (catapult_cost : ℕ) (ferris_wheel_cost : ℕ) 
  (total_tickets : ℕ) :
  rollercoaster_rides = 3 →
  ferris_wheel_rides = 1 →
  rollercoaster_cost = 4 →
  catapult_cost = 4 →
  ferris_wheel_cost = 1 →
  total_tickets = 21 →
  ∃ catapult_rides : ℕ, 
    catapult_rides * catapult_cost + 
    rollercoaster_rides * rollercoaster_cost + 
    ferris_wheel_rides * ferris_wheel_cost = total_tickets ∧
    catapult_rides = 2 :=
by sorry

end turner_rides_l3092_309285


namespace fixed_point_of_exponential_function_l3092_309262

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 :=
by sorry

end fixed_point_of_exponential_function_l3092_309262


namespace relationship_order_l3092_309205

theorem relationship_order (a b : ℝ) (h1 : a + b > 0) (h2 : b < 0) : a > -b ∧ -b > b ∧ b > -a := by
  sorry

end relationship_order_l3092_309205


namespace oil_production_theorem_l3092_309264

/-- Oil production per person for different regions --/
def oil_production_problem : Prop :=
  let west_production := 55.084
  let non_west_production := 214.59
  let russia_production := 1038.33
  let total_production := 13737.1
  let russia_percentage := 0.09
  let russia_population := 147000000

  let russia_total_production := total_production * russia_percentage
  let russia_per_person := russia_total_production / russia_population

  (west_production = 55.084) ∧
  (non_west_production = 214.59) ∧
  (russia_per_person = 1038.33)

theorem oil_production_theorem : oil_production_problem := by
  sorry

end oil_production_theorem_l3092_309264


namespace min_b_for_q_half_or_more_l3092_309299

def q (b : ℕ) : ℚ :=
  (Nat.choose (40 - b) 2 + Nat.choose (b - 1) 2) / 1225

theorem min_b_for_q_half_or_more : 
  ∀ b : ℕ, 1 ≤ b ∧ b ≤ 41 → (q b ≥ 1/2 ↔ b ≥ 7) :=
sorry

end min_b_for_q_half_or_more_l3092_309299


namespace points_form_circle_l3092_309278

theorem points_form_circle :
  ∀ (t : ℝ), ∃ (x y : ℝ), x = Real.cos t ∧ y = Real.sin t → x^2 + y^2 = 1 :=
by sorry

end points_form_circle_l3092_309278


namespace quadratic_function_properties_l3092_309219

/-- A quadratic function satisfying specific conditions -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  h1 : a ≠ 0
  h2 : ∀ x, a*(x-1)^2 + b*(x-1) = a*(3-x)^2 + b*(3-x)
  h3 : ∃! x, a*x^2 + b*x = 2*x

/-- The main theorem about the quadratic function -/
theorem quadratic_function_properties (f : QuadraticFunction) :
  (∀ x, f.a*x^2 + f.b*x = -x^2 + 2*x) ∧
  ∃ m n, m < n ∧
    (∀ x, f.a*x^2 + f.b*x ∈ Set.Icc m n ↔ x ∈ Set.Icc (-1) 0) ∧
    (∀ y, y ∈ Set.Icc (4*(-1)) (4*0) ↔ ∃ x, f.a*x^2 + f.b*x = y) :=
sorry

end quadratic_function_properties_l3092_309219


namespace lucca_ball_count_l3092_309201

theorem lucca_ball_count :
  ∀ (lucca_balls : ℕ) (lucca_basketballs : ℕ) (lucien_basketballs : ℕ),
    lucca_basketballs = lucca_balls / 10 →
    lucien_basketballs = 40 →
    lucca_basketballs + lucien_basketballs = 50 →
    lucca_balls = 100 :=
by
  sorry

end lucca_ball_count_l3092_309201


namespace total_fireworks_is_1188_l3092_309214

/-- Calculates the total number of fireworks used in the New Year's Eve display -/
def total_fireworks : ℕ :=
  let fireworks_per_number : ℕ := 6
  let fireworks_per_regular_letter : ℕ := 5
  let fireworks_for_H : ℕ := 8
  let fireworks_for_E : ℕ := 7
  let fireworks_for_L : ℕ := 6
  let fireworks_for_O : ℕ := 9
  let fireworks_for_square : ℕ := 4
  let fireworks_for_triangle : ℕ := 3
  let fireworks_for_circle : ℕ := 12
  let additional_boxes : ℕ := 100
  let fireworks_per_box : ℕ := 10

  let years_fireworks := fireworks_per_number * 4 * 3
  let happy_new_year_fireworks := fireworks_per_regular_letter * 11 + fireworks_per_number
  let geometric_shapes_fireworks := fireworks_for_square + fireworks_for_triangle + fireworks_for_circle
  let hello_fireworks := fireworks_for_H + fireworks_for_E + fireworks_for_L * 2 + fireworks_for_O
  let additional_fireworks := additional_boxes * fireworks_per_box

  years_fireworks + happy_new_year_fireworks + geometric_shapes_fireworks + hello_fireworks + additional_fireworks

theorem total_fireworks_is_1188 : total_fireworks = 1188 := by
  sorry

end total_fireworks_is_1188_l3092_309214


namespace equation_equals_24_l3092_309261

theorem equation_equals_24 : (2 + 2 / 11) * 11 = 24 := by
  sorry

#check equation_equals_24

end equation_equals_24_l3092_309261


namespace duck_profit_is_170_l3092_309210

/-- Calculates the profit from selling ducks given the specified conditions -/
def duck_profit : ℕ :=
  let first_group_weight := 10 * 3
  let second_group_weight := 10 * 4
  let third_group_weight := 10 * 5
  let total_weight := first_group_weight + second_group_weight + third_group_weight

  let first_group_cost := 10 * 9
  let second_group_cost := 10 * 10
  let third_group_cost := 10 * 12
  let total_cost := first_group_cost + second_group_cost + third_group_cost

  let selling_price_per_pound := 5
  let total_selling_price := total_weight * selling_price_per_pound
  let discount_rate := 20
  let discount_amount := total_selling_price * discount_rate / 100
  let final_selling_price := total_selling_price - discount_amount

  final_selling_price - total_cost

theorem duck_profit_is_170 : duck_profit = 170 := by
  sorry

end duck_profit_is_170_l3092_309210


namespace range_of_negative_two_a_plus_three_l3092_309270

theorem range_of_negative_two_a_plus_three (a : ℝ) : 
  a < 1 → -2*a + 3 > 1 := by
sorry

end range_of_negative_two_a_plus_three_l3092_309270


namespace joe_cars_l3092_309255

theorem joe_cars (initial_cars additional_cars : ℕ) :
  initial_cars = 50 → additional_cars = 12 → initial_cars + additional_cars = 62 := by
  sorry

end joe_cars_l3092_309255


namespace remainder_of_x_50_divided_by_x2_minus_4x_plus_3_l3092_309271

theorem remainder_of_x_50_divided_by_x2_minus_4x_plus_3 :
  ∀ (x : ℝ), ∃ (Q : ℝ → ℝ) (R : ℝ → ℝ),
    x^50 = (x^2 - 4*x + 3) * Q x + R x ∧
    (∀ (y : ℝ), R y = (3^50 - 1)/2 * y + (3 - 3^50)/2) ∧
    (∀ (y : ℝ), ∃ (a b : ℝ), R y = a * y + b) :=
by sorry

end remainder_of_x_50_divided_by_x2_minus_4x_plus_3_l3092_309271


namespace estimate_sqrt_expression_l3092_309275

theorem estimate_sqrt_expression :
  7 < Real.sqrt 36 * Real.sqrt (1/2) + Real.sqrt 8 ∧
  Real.sqrt 36 * Real.sqrt (1/2) + Real.sqrt 8 < 8 :=
by sorry

end estimate_sqrt_expression_l3092_309275


namespace count_four_digit_divisible_by_13_l3092_309251

theorem count_four_digit_divisible_by_13 : 
  (Finset.filter (fun n => n % 13 = 0) (Finset.range 9000)).card = 689 := by
  sorry

end count_four_digit_divisible_by_13_l3092_309251


namespace fermat_numbers_coprime_l3092_309297

theorem fermat_numbers_coprime (m n : ℕ) (h : m ≠ n) : 
  Nat.gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 := by
  sorry

end fermat_numbers_coprime_l3092_309297


namespace divisibility_of_n_l3092_309243

theorem divisibility_of_n : ∀ (n : ℕ),
  n = (2^4 - 1) * (3^6 - 1) * (5^10 - 1) * (7^12 - 1) →
  5 ∣ n ∧ 7 ∣ n ∧ 11 ∣ n ∧ 13 ∣ n :=
by
  sorry

end divisibility_of_n_l3092_309243


namespace function_equality_l3092_309234

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x - 5

-- Theorem statement
theorem function_equality (x : ℝ) : 
  (2 * f x - 10 = f (x - 2)) ↔ x = 3 := by
  sorry

end function_equality_l3092_309234


namespace final_sum_theorem_l3092_309259

/-- The number of participants in the game --/
def participants : ℕ := 53

/-- The initial value of the first calculator --/
def calc1_initial : ℤ := 2

/-- The initial value of the second calculator --/
def calc2_initial : ℤ := -2

/-- The initial value of the third calculator --/
def calc3_initial : ℕ := 5

/-- The operation applied to the first calculator --/
def op1 (n : ℤ) : ℤ := n ^ 2

/-- The operation applied to the second calculator --/
def op2 (n : ℤ) : ℤ := n ^ 3

/-- The operation applied to the third calculator --/
def op3 (n : ℕ) : ℕ := n + 2

/-- The final value of the first calculator after all participants --/
def calc1_final : ℤ := calc1_initial ^ (2 ^ participants)

/-- The final value of the second calculator after all participants --/
def calc2_final : ℤ := calc2_initial ^ (3 ^ participants)

/-- The final value of the third calculator after all participants --/
def calc3_final : ℕ := calc3_initial + 2 * participants

/-- The theorem stating the final sum of all calculators --/
theorem final_sum_theorem : 
  calc1_final + calc2_final + calc3_final = 
  calc1_initial ^ (2 ^ participants) + calc2_initial ^ (3 ^ participants) + (calc3_initial + 2 * participants) := by
  sorry

end final_sum_theorem_l3092_309259


namespace arithmetic_sequence_problem_l3092_309247

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence {a_n}, if a_1 + 3a_8 + a_15 = 120, then a_8 = 24 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + 3 * a 8 + a 15 = 120) : 
  a 8 = 24 := by
  sorry

end arithmetic_sequence_problem_l3092_309247


namespace trigonometric_inequality_l3092_309263

theorem trigonometric_inequality (h : 3 * Real.pi / 8 ∈ Set.Ioo 0 (Real.pi / 2)) :
  Real.sin (Real.cos (3 * Real.pi / 8)) < Real.cos (Real.sin (3 * Real.pi / 8)) ∧
  Real.cos (Real.sin (3 * Real.pi / 8)) < Real.sin (Real.sin (3 * Real.pi / 8)) ∧
  Real.sin (Real.sin (3 * Real.pi / 8)) < Real.cos (Real.cos (3 * Real.pi / 8)) := by
  sorry

end trigonometric_inequality_l3092_309263


namespace solve_equation_l3092_309290

theorem solve_equation (x y : ℝ) : y = 2 / (5 * x + 3) → y = 2 → x = -2 / 5 := by
  sorry

end solve_equation_l3092_309290


namespace system_solution_ratio_l3092_309269

theorem system_solution_ratio (x y c d : ℝ) 
  (eq1 : 8 * x - 6 * y = c)
  (eq2 : 9 * y - 12 * x = d)
  (x_nonzero : x ≠ 0)
  (y_nonzero : y ≠ 0)
  (d_nonzero : d ≠ 0) :
  c / d = -2 / 3 := by
sorry

end system_solution_ratio_l3092_309269


namespace specific_quadrilateral_area_l3092_309288

/-- Represents a quadrilateral ABCD with given side lengths and a right angle at C -/
structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  AD : ℝ
  right_angle_at_C : Bool

/-- Calculates the area of the quadrilateral ABCD -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  sorry

/-- Theorem stating that the area of the specific quadrilateral is 106 -/
theorem specific_quadrilateral_area :
  ∃ (q : Quadrilateral),
    q.AB = 15 ∧
    q.BC = 5 ∧
    q.CD = 12 ∧
    q.AD = 13 ∧
    q.right_angle_at_C = true ∧
    area q = 106 := by
  sorry

end specific_quadrilateral_area_l3092_309288


namespace bales_stored_l3092_309233

theorem bales_stored (initial_bales final_bales : ℕ) 
  (h1 : initial_bales = 73)
  (h2 : final_bales = 96) :
  final_bales - initial_bales = 23 := by
sorry

end bales_stored_l3092_309233


namespace gregs_shopping_expenditure_l3092_309287

/-- Greg's shopping expenditure theorem -/
theorem gregs_shopping_expenditure (shirt_cost shoes_cost : ℕ) : 
  shirt_cost + shoes_cost = 300 →
  shoes_cost = 2 * shirt_cost + 9 →
  shirt_cost = 97 := by
  sorry

end gregs_shopping_expenditure_l3092_309287


namespace tetrahedron_volume_ratio_l3092_309220

/-- Represents a tetrahedron ABCD with specific properties -/
structure Tetrahedron where
  a : ℝ  -- Length of edge AB
  b : ℝ  -- Length of edge CD
  d : ℝ  -- Distance between lines AB and CD
  w : ℝ  -- Angle between lines AB and CD
  k : ℝ  -- Ratio of distances from plane π to AB and CD
  h_a : a > 0
  h_b : b > 0
  h_d : d > 0
  h_w : 0 < w ∧ w < π
  h_k : k > 0

/-- Calculates the volume ratio of the two parts of the tetrahedron divided by plane π -/
noncomputable def volumeRatio (t : Tetrahedron) : ℝ :=
  (t.k^3 + 3*t.k^2) / (3*t.k + 1)

/-- Theorem stating the volume ratio of the two parts of the tetrahedron -/
theorem tetrahedron_volume_ratio (t : Tetrahedron) :
  ∃ (v1 v2 : ℝ), v1 > 0 ∧ v2 > 0 ∧ v1 / v2 = volumeRatio t :=
by sorry

end tetrahedron_volume_ratio_l3092_309220


namespace work_completion_time_l3092_309200

/-- If a group can complete a task in 12 days, then twice that group can complete half the task in 3 days. -/
theorem work_completion_time 
  (people : ℕ) 
  (work : ℝ) 
  (h : people > 0) 
  (completion_time : ℝ := 12) 
  (h_completion : work = people * completion_time) : 
  work / 2 = (2 * people) * 3 := by
sorry

end work_completion_time_l3092_309200


namespace problem_solution_l3092_309215

-- Define the variables
variable (a b c : ℝ)

-- Define the conditions
def condition1 : Prop := (5 * a + 2) ^ (1/3 : ℝ) = 3
def condition2 : Prop := (3 * a + b - 1) ^ (1/2 : ℝ) = 4
def condition3 : Prop := c = ⌊Real.sqrt 13⌋

-- Define the theorem
theorem problem_solution (h1 : condition1 a) (h2 : condition2 a b) (h3 : condition3 c) :
  a = 5 ∧ b = 2 ∧ c = 3 ∧ (3 * a - b + c) ^ (1/2 : ℝ) = 4 ∨ (3 * a - b + c) ^ (1/2 : ℝ) = -4 :=
sorry

end problem_solution_l3092_309215


namespace quadratic_inequality_solution_sets_l3092_309208

theorem quadratic_inequality_solution_sets 
  (a b c : ℝ) 
  (h : ∀ x, ax^2 + b*x + c > 0 ↔ 3 < x ∧ x < 6) :
  ∀ x, c*x^2 + b*x + a < 0 ↔ x < 1/6 ∨ x > 1/3 :=
by sorry

end quadratic_inequality_solution_sets_l3092_309208


namespace eds_remaining_money_l3092_309266

/-- Calculates the remaining money after a hotel stay -/
def remaining_money (initial_amount : ℝ) (night_rate : ℝ) (morning_rate : ℝ) 
  (night_hours : ℝ) (morning_hours : ℝ) : ℝ :=
  initial_amount - (night_rate * night_hours + morning_rate * morning_hours)

/-- Theorem: Ed's remaining money after his hotel stay -/
theorem eds_remaining_money :
  remaining_money 80 1.5 2 6 4 = 63 := by
  sorry

end eds_remaining_money_l3092_309266


namespace find_d_value_l3092_309258

theorem find_d_value (x y d : ℝ) : 
  7^(3*x - 1) * 3^(4*y - 3) = 49^x * d^y ∧ x + y = 4 → d = 27 := by
  sorry

end find_d_value_l3092_309258


namespace polynomial_evaluation_l3092_309256

theorem polynomial_evaluation (x : ℝ) (h : x = 2) : 3 * x^2 + 5 * x - 2 = 20 := by
  sorry

end polynomial_evaluation_l3092_309256


namespace karthik_weight_upper_bound_l3092_309279

-- Define the lower and upper bounds for Karthik's weight according to different opinions
def karthik_lower_bound : ℝ := 55
def brother_lower_bound : ℝ := 50
def brother_upper_bound : ℝ := 60
def father_upper_bound : ℝ := 58

-- Define the average weight
def average_weight : ℝ := 56.5

-- Define Karthik's upper bound as a variable
def karthik_upper_bound : ℝ := sorry

-- Theorem statement
theorem karthik_weight_upper_bound :
  karthik_lower_bound < karthik_upper_bound ∧
  brother_lower_bound < karthik_upper_bound ∧
  karthik_upper_bound ≤ brother_upper_bound ∧
  karthik_upper_bound ≤ father_upper_bound ∧
  average_weight = (karthik_lower_bound + karthik_upper_bound) / 2 →
  karthik_upper_bound = 58 := by sorry

end karthik_weight_upper_bound_l3092_309279


namespace max_garden_area_l3092_309224

/-- Represents the dimensions of a rectangular garden -/
structure GardenDimensions where
  width : ℝ
  length : ℝ

/-- Calculates the area of a rectangular garden -/
def gardenArea (d : GardenDimensions) : ℝ := d.width * d.length

/-- Calculates the perimeter of a rectangular garden (excluding the side adjacent to the house) -/
def gardenPerimeter (d : GardenDimensions) : ℝ := d.length + 2 * d.width

/-- Theorem stating the maximum area of the garden under given constraints -/
theorem max_garden_area :
  ∃ (d : GardenDimensions),
    d.length = 2 * d.width ∧
    gardenPerimeter d = 480 ∧
    ∀ (d' : GardenDimensions),
      d'.length = 2 * d'.width →
      gardenPerimeter d' = 480 →
      gardenArea d' ≤ 28800 :=
by
  sorry

end max_garden_area_l3092_309224


namespace quadratic_solution_sum_l3092_309250

theorem quadratic_solution_sum (x : ℝ) (m n p : ℕ) : 
  x * (4 * x - 9) = -4 ∧ 
  (∃ (r : ℝ), r * r = n ∧ 
    (x = (m + r) / p ∨ x = (m - r) / p)) ∧
  Nat.gcd m (Nat.gcd n p) = 1 →
  m + n + p = 34 := by
sorry

end quadratic_solution_sum_l3092_309250
