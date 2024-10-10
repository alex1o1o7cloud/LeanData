import Mathlib

namespace sum_is_positive_l1432_143211

theorem sum_is_positive (x y : ℝ) (h1 : x * y < 0) (h2 : x > abs y) : x + y > 0 := by
  sorry

end sum_is_positive_l1432_143211


namespace least_equal_bulbs_l1432_143207

def tulip_pack_size : ℕ := 15
def daffodil_pack_size : ℕ := 16

theorem least_equal_bulbs :
  ∃ (n : ℕ), n > 0 ∧ n % tulip_pack_size = 0 ∧ n % daffodil_pack_size = 0 ∧
  ∀ (m : ℕ), (m > 0 ∧ m % tulip_pack_size = 0 ∧ m % daffodil_pack_size = 0) → m ≥ n :=
by
  use 240
  sorry

end least_equal_bulbs_l1432_143207


namespace number_of_students_is_five_l1432_143208

/-- The number of students who will receive stickers from Miss Walter -/
def number_of_students : ℕ :=
  let gold_stickers : ℕ := 50
  let silver_stickers : ℕ := 2 * gold_stickers
  let bronze_stickers : ℕ := silver_stickers - 20
  let total_stickers : ℕ := gold_stickers + silver_stickers + bronze_stickers
  let stickers_per_student : ℕ := 46
  total_stickers / stickers_per_student

/-- Theorem stating that the number of students who will receive stickers is 5 -/
theorem number_of_students_is_five : number_of_students = 5 := by
  sorry

end number_of_students_is_five_l1432_143208


namespace correct_subtraction_l1432_143203

theorem correct_subtraction (x : ℤ) (h : x - 63 = 8) : x - 36 = 35 := by
  sorry

end correct_subtraction_l1432_143203


namespace class_cans_collection_l1432_143297

/-- Calculates the total number of cans collected by a class given specific conditions -/
def totalCansCollected (totalStudents : ℕ) (cansPerHalf : ℕ) (nonCollectingStudents : ℕ) 
  (remainingStudents : ℕ) (cansPerRemaining : ℕ) : ℕ :=
  let halfStudents := totalStudents / 2
  let cansFromHalf := halfStudents * cansPerHalf
  let cansFromRemaining := remainingStudents * cansPerRemaining
  cansFromHalf + cansFromRemaining

/-- Theorem stating that under given conditions, the class collects 232 cans in total -/
theorem class_cans_collection : 
  totalCansCollected 30 12 2 13 4 = 232 := by
  sorry

end class_cans_collection_l1432_143297


namespace min_length_shared_side_l1432_143245

/-- Given two triangles ABC and DBC sharing side BC, with known side lengths,
    prove that the length of BC must be greater than 14. -/
theorem min_length_shared_side (AB AC DC BD BC : ℝ) : 
  AB = 7 → AC = 15 → DC = 9 → BD = 23 → BC > 14 := by
  sorry

end min_length_shared_side_l1432_143245


namespace exists_lcm_sum_for_non_power_of_two_l1432_143273

/-- Represents the least common multiple of two positive integers -/
def lcm (a b : ℕ+) : ℕ := (a * b) / Nat.gcd a b

/-- Theorem: For any natural number n that is not a power of 2,
    there exist positive integers a, b, and c such that
    n = lcm a b + lcm b c + lcm c a -/
theorem exists_lcm_sum_for_non_power_of_two (n : ℕ) 
    (h : ∀ k : ℕ, n ≠ 2^k) :
    ∃ (a b c : ℕ+), n = lcm a b + lcm b c + lcm c a := by
  sorry

end exists_lcm_sum_for_non_power_of_two_l1432_143273


namespace tangent_to_circumcircle_l1432_143263

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the necessary relations and functions
variable (midpoint : Circle → Point)
variable (intersect : Circle → Circle → Set Point)
variable (line_intersect : Point → Point → Circle → Set Point)
variable (on_line : Point → Point → Point → Prop)
variable (circumcircle : Point → Point → Point → Circle)
variable (tangent_to : Point → Point → Circle → Prop)

-- State the theorem
theorem tangent_to_circumcircle
  (Γ₁ Γ₂ Γ₃ : Circle)
  (O₁ O₂ A C D S E F : Point) :
  (midpoint Γ₁ = O₁) →
  (midpoint Γ₂ = O₂) →
  (A ∈ intersect Γ₂ (circumcircle O₁ O₂ A)) →
  ({C, D} ⊆ intersect Γ₁ Γ₂) →
  (S ∈ line_intersect A D Γ₁) →
  (on_line C S F) →
  (on_line O₁ O₂ F) →
  (Γ₃ = circumcircle A D E) →
  (E ∈ intersect Γ₁ Γ₃) →
  (E ≠ D) →
  tangent_to O₁ E Γ₃ :=
sorry

end tangent_to_circumcircle_l1432_143263


namespace garden_yield_mr_green_garden_yield_l1432_143228

/-- Calculates the expected potato yield from a rectangular garden after applying fertilizer -/
theorem garden_yield (length_steps width_steps feet_per_step : ℕ) 
  (original_yield_per_sqft : ℚ) (yield_increase_percent : ℕ) : ℚ :=
  let length_feet := length_steps * feet_per_step
  let width_feet := width_steps * feet_per_step
  let area := length_feet * width_feet
  let original_yield := area * original_yield_per_sqft
  let yield_increase_factor := 1 + yield_increase_percent / 100
  original_yield * yield_increase_factor

/-- Proves that Mr. Green's garden will yield 2227.5 pounds of potatoes after fertilizer -/
theorem mr_green_garden_yield :
  garden_yield 18 25 3 (1/2) 10 = 2227.5 := by
  sorry

end garden_yield_mr_green_garden_yield_l1432_143228


namespace bottle_cap_distance_difference_l1432_143257

/-- Calculates the total distance traveled by Jenny's bottle cap -/
def jennys_distance : ℝ := 18 + 6 + 7.2 + 3.6 + 3.96

/-- Calculates the total distance traveled by Mark's bottle cap -/
def marks_distance : ℝ := 15 + 30 + 34.5 + 25.875 + 24.58125 + 7.374375 + 9.21796875

/-- The difference in distance between Mark's and Jenny's bottle caps -/
def distance_difference : ℝ := marks_distance - jennys_distance

theorem bottle_cap_distance_difference :
  distance_difference = 107.78959375 := by sorry

end bottle_cap_distance_difference_l1432_143257


namespace common_divisors_90_105_l1432_143259

theorem common_divisors_90_105 : Finset.card (Finset.filter (· ∣ 105) (Nat.divisors 90)) = 8 := by
  sorry

end common_divisors_90_105_l1432_143259


namespace transformed_equation_solutions_l1432_143289

theorem transformed_equation_solutions
  (h : ∀ x : ℝ, x^2 + 2*x - 3 = 0 ↔ x = 1 ∨ x = -3) :
  ∀ x : ℝ, (x + 3)^2 + 2*(x + 3) - 3 = 0 ↔ x = -2 ∨ x = -6 := by
sorry

end transformed_equation_solutions_l1432_143289


namespace puppy_weight_l1432_143260

/-- Given the weights of animals satisfying certain conditions, prove the puppy's weight is √2 -/
theorem puppy_weight (p s l r : ℝ) 
  (h1 : p + s + l + r = 40)
  (h2 : p^2 + l^2 = 4*s)
  (h3 : p^2 + s^2 = l^2) :
  p = Real.sqrt 2 := by
  sorry

end puppy_weight_l1432_143260


namespace sequence_length_l1432_143231

/-- The sequence defined by a(n) = 2 + 5(n-1) for n ≥ 1 -/
def a : ℕ → ℕ := λ n => 2 + 5 * (n - 1)

/-- The last term of the sequence -/
def last_term : ℕ := 57

theorem sequence_length :
  ∃ n : ℕ, n > 0 ∧ a n = last_term ∧ n = 12 := by sorry

end sequence_length_l1432_143231


namespace symmetric_circles_and_common_chord_l1432_143266

-- Define the symmetry relation with respect to line l
def symmetric_line (x y : ℝ) : Prop := ∃ (x' y' : ℝ), x' = y + 1 ∧ y' = x - 1

-- Define circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 2*y = 0

-- Define circle C'
def circle_C' (x y : ℝ) : Prop := (x-2)^2 + (y-2)^2 = 10

-- Theorem statement
theorem symmetric_circles_and_common_chord :
  (∀ x y : ℝ, symmetric_line x y → (circle_C x y ↔ circle_C' y x)) ∧
  (∃ a b c d : ℝ, 
    circle_C a b ∧ circle_C c d ∧ 
    circle_C' a b ∧ circle_C' c d ∧
    (a - c)^2 + (b - d)^2 = 38) :=
sorry

end symmetric_circles_and_common_chord_l1432_143266


namespace fraction_value_l1432_143250

theorem fraction_value (a b c d : ℝ) 
  (h1 : a = 3 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 4 * d) : 
  a * c / (b * d) = 12 := by
sorry

end fraction_value_l1432_143250


namespace simplify_fraction_multiplication_l1432_143258

theorem simplify_fraction_multiplication :
  (175 : ℚ) / 1225 * 25 = 25 / 7 := by
sorry

end simplify_fraction_multiplication_l1432_143258


namespace equation_solutions_l1432_143252

/-- Definition of matrix expression -/
def matrix_expr (a b c d : ℝ) : ℝ := a * b - c * d

/-- The equation to be solved -/
def equation (x : ℝ) : Prop :=
  matrix_expr (3 * x) (2 * x + 1) 1 (2 * x) = 5

/-- Theorem stating the solutions of the equation -/
theorem equation_solutions :
  ∃ (x₁ x₂ : ℝ), x₁ = -1 ∧ x₂ = 5/6 ∧ equation x₁ ∧ equation x₂ ∧
  ∀ (x : ℝ), equation x → (x = x₁ ∨ x = x₂) :=
sorry

end equation_solutions_l1432_143252


namespace cubic_equation_integer_solutions_l1432_143232

theorem cubic_equation_integer_solutions :
  ∀ x y : ℤ, x^3 + 2*x*y - 7 = 0 ↔ 
    (x = -7 ∧ y = -25) ∨ 
    (x = -1 ∧ y = -4) ∨ 
    (x = 1 ∧ y = 3) ∨ 
    (x = 7 ∧ y = -24) := by
  sorry

end cubic_equation_integer_solutions_l1432_143232


namespace investment_with_interest_l1432_143225

def total_investment : ℝ := 1000
def amount_at_3_percent : ℝ := 199.99999999999983
def interest_rate_3_percent : ℝ := 0.03
def interest_rate_5_percent : ℝ := 0.05

theorem investment_with_interest :
  let amount_at_5_percent := total_investment - amount_at_3_percent
  let interest_at_3_percent := amount_at_3_percent * interest_rate_3_percent
  let interest_at_5_percent := amount_at_5_percent * interest_rate_5_percent
  let total_with_interest := total_investment + interest_at_3_percent + interest_at_5_percent
  total_with_interest = 1046 := by sorry

end investment_with_interest_l1432_143225


namespace leftover_tarts_l1432_143293

theorem leftover_tarts (cherry_tarts blueberry_tarts peach_tarts : ℝ) 
  (h1 : cherry_tarts = 0.08)
  (h2 : blueberry_tarts = 0.75)
  (h3 : peach_tarts = 0.08) :
  cherry_tarts + blueberry_tarts + peach_tarts = 0.91 := by
  sorry

end leftover_tarts_l1432_143293


namespace base8_digit_product_l1432_143218

/-- Converts a natural number from base 10 to base 8 --/
def toBase8 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers --/
def productOfList (l : List ℕ) : ℕ :=
  sorry

theorem base8_digit_product :
  productOfList (toBase8 8127) = 1764 :=
sorry

end base8_digit_product_l1432_143218


namespace combined_weight_of_new_men_weight_problem_l1432_143286

/-- The combined weight of two new men replacing one man in a group, given certain conditions -/
theorem combined_weight_of_new_men (initial_count : ℕ) (weight_increase : ℝ) 
  (replaced_weight : ℝ) (new_count : ℕ) : ℝ :=
  let total_weight_increase := weight_increase * new_count
  let combined_weight := total_weight_increase + replaced_weight
  combined_weight

/-- The theorem statement matching the original problem -/
theorem weight_problem : 
  combined_weight_of_new_men 10 2.5 68 11 = 95.5 := by
  sorry

end combined_weight_of_new_men_weight_problem_l1432_143286


namespace min_angle_BFE_l1432_143274

-- Define the triangle ABC
structure Triangle :=
  (A B C : Point)

-- Define the incenter of a triangle
def incenter (t : Triangle) : Point := sorry

-- Define the angle between three points
def angle (A B C : Point) : ℝ := sorry

-- Main theorem
theorem min_angle_BFE (ABC : Triangle) :
  let D := incenter ABC
  let ABD := Triangle.mk ABC.A ABC.B D
  let E := incenter ABD
  let BDE := Triangle.mk ABC.B D E
  let F := incenter BDE
  ∀ θ : ℕ, (θ : ℝ) = angle B F E → θ ≥ 113 :=
sorry

end min_angle_BFE_l1432_143274


namespace min_plates_matching_pair_l1432_143272

/-- Represents the colors of plates -/
inductive PlateColor
  | White
  | Green
  | Red
  | Pink
  | Purple

/-- The minimum number of plates needed to guarantee a matching pair -/
def min_plates_for_match : ℕ := 6

/-- Theorem stating that given at least one plate of each of 5 colors,
    the minimum number of plates needed to guarantee a matching pair is 6 -/
theorem min_plates_matching_pair
  (white_count : ℕ) (green_count : ℕ) (red_count : ℕ) (pink_count : ℕ) (purple_count : ℕ)
  (h_white : white_count ≥ 1)
  (h_green : green_count ≥ 1)
  (h_red : red_count ≥ 1)
  (h_pink : pink_count ≥ 1)
  (h_purple : purple_count ≥ 1) :
  min_plates_for_match = 6 := by
  sorry

#check min_plates_matching_pair

end min_plates_matching_pair_l1432_143272


namespace quadrilateral_perimeter_l1432_143212

/-- Perimeter of a quadrilateral EFGH with specific properties -/
theorem quadrilateral_perimeter (EF HG FG : ℝ) (h1 : EF = 15) (h2 : HG = 6) (h3 : FG = 20) :
  ∃ (EH : ℝ), EF + FG + HG + EH = 41 + Real.sqrt 481 := by
  sorry

end quadrilateral_perimeter_l1432_143212


namespace bill_toilet_paper_supply_l1432_143215

/-- The number of days Bill's toilet paper supply will last -/
def toilet_paper_days (bathroom_visits_per_day : ℕ) (squares_per_visit : ℕ) (rolls : ℕ) (squares_per_roll : ℕ) : ℕ :=
  (rolls * squares_per_roll) / (bathroom_visits_per_day * squares_per_visit)

/-- Theorem stating that Bill's toilet paper supply will last for 20,000 days -/
theorem bill_toilet_paper_supply : toilet_paper_days 3 5 1000 300 = 20000 := by
  sorry

end bill_toilet_paper_supply_l1432_143215


namespace constant_term_of_f_composition_l1432_143220

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (x - 1/x)^8 else -Real.sqrt x

theorem constant_term_of_f_composition (x : ℝ) (h : x > 0) :
  ∃ (expansion : ℝ → ℝ),
    (∀ y, y > 0 → f (f y) = expansion y) ∧
    (∃ c, ∀ ε > 0, |expansion x - c| < ε) ∧
    (∀ c, (∃ ε > 0, |expansion x - c| < ε) → c = 70) :=
sorry

end constant_term_of_f_composition_l1432_143220


namespace shielas_drawings_l1432_143253

/-- The number of neighbors Shiela has -/
def num_neighbors : ℕ := 6

/-- The number of drawings each neighbor would receive -/
def drawings_per_neighbor : ℕ := 9

/-- The total number of animal drawings Shiela drew -/
def total_drawings : ℕ := num_neighbors * drawings_per_neighbor

theorem shielas_drawings : total_drawings = 54 := by
  sorry

end shielas_drawings_l1432_143253


namespace franks_savings_l1432_143299

/-- The amount of money Frank had saved initially -/
def initial_savings : ℕ := sorry

/-- The cost of one toy -/
def toy_cost : ℕ := 8

/-- The number of toys Frank could buy -/
def num_toys : ℕ := 5

/-- The additional allowance Frank received -/
def additional_allowance : ℕ := 37

/-- Theorem stating that Frank's initial savings is $3 -/
theorem franks_savings : 
  (initial_savings + additional_allowance = num_toys * toy_cost) → 
  initial_savings = 3 := by sorry

end franks_savings_l1432_143299


namespace probability_after_removing_pairs_l1432_143254

/-- Represents a deck of cards -/
structure Deck :=
  (total : ℕ)
  (numbers : ℕ)
  (cards_per_number : ℕ)

/-- Represents the state after removing pairs -/
structure RemovedPairs :=
  (pairs_removed : ℕ)

/-- Calculates the probability of selecting a pair from the remaining deck -/
def probability_of_pair (d : Deck) (r : RemovedPairs) : ℚ :=
  sorry

/-- The main theorem -/
theorem probability_after_removing_pairs :
  let d : Deck := ⟨80, 20, 4⟩
  let r : RemovedPairs := ⟨3⟩
  probability_of_pair d r = 105 / 2701 :=
sorry

end probability_after_removing_pairs_l1432_143254


namespace absolute_value_equation_product_l1432_143279

theorem absolute_value_equation_product (x : ℝ) : 
  (|20 / x + 4| = 3) → (∃ y : ℝ, (|20 / y + 4| = 3) ∧ (x * y = 400 / 7)) :=
by
  sorry

end absolute_value_equation_product_l1432_143279


namespace grid_solution_l1432_143222

/-- Represents the possible values in the grid -/
inductive GridValue
  | Two
  | Zero
  | One
  | Five
  | Blank

/-- Represents a 5x5 grid -/
def Grid := Fin 5 → Fin 5 → GridValue

/-- Check if a grid satisfies the row and column constraints -/
def isValidGrid (g : Grid) : Prop :=
  (∀ i : Fin 5, ∃! j : Fin 5, g i j = GridValue.Two) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g i j = GridValue.Zero) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g i j = GridValue.One) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g i j = GridValue.Five) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g j i = GridValue.Two) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g j i = GridValue.Zero) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g j i = GridValue.One) ∧
  (∀ i : Fin 5, ∃! j : Fin 5, g j i = GridValue.Five)

/-- Check if the diagonal constraint is satisfied -/
def validDiagonal (g : Grid) : Prop :=
  ∀ i j : Fin 5, i ≠ j → g i i ≠ g j j

/-- The main theorem stating the solution -/
theorem grid_solution (g : Grid) 
  (hvalid : isValidGrid g) 
  (hdiag : validDiagonal g) : 
  g 4 0 = GridValue.One ∧ 
  g 4 1 = GridValue.Five ∧ 
  g 4 2 = GridValue.Blank ∧ 
  g 4 3 = GridValue.Zero ∧ 
  g 4 4 = GridValue.Two :=
sorry

end grid_solution_l1432_143222


namespace M_properties_l1432_143276

-- Define the set M
def M : Set (ℝ × ℝ) := {p | Real.sqrt 2 * p.1 - 1 < p.2 ∧ p.2 < Real.sqrt 2 * p.1}

-- Define what it means for a point to have integer coordinates
def hasIntegerCoordinates (p : ℝ × ℝ) : Prop := ∃ (i j : ℤ), p = (↑i, ↑j)

-- Statement of the theorem
theorem M_properties :
  Convex ℝ M ∧
  (∃ (S : Set (ℝ × ℝ)), S ⊆ M ∧ Set.Infinite S ∧ ∀ p ∈ S, hasIntegerCoordinates p) ∧
  ∀ (a b : ℝ), let L := {p : ℝ × ℝ | p.2 = a * p.1 + b}
    (∃ (S : Set (ℝ × ℝ)), S ⊆ (M ∩ L) ∧ Set.Finite S ∧
      ∀ p ∈ (M ∩ L), hasIntegerCoordinates p → p ∈ S) :=
by
  sorry

end M_properties_l1432_143276


namespace arccos_one_equals_zero_l1432_143295

theorem arccos_one_equals_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_equals_zero_l1432_143295


namespace common_internal_tangent_length_l1432_143202

theorem common_internal_tangent_length
  (center_distance : ℝ)
  (small_radius : ℝ)
  (large_radius : ℝ)
  (h1 : center_distance = 50)
  (h2 : small_radius = 7)
  (h3 : large_radius = 10) :
  Real.sqrt (center_distance ^ 2 - (small_radius + large_radius) ^ 2) = Real.sqrt 2211 :=
by sorry

end common_internal_tangent_length_l1432_143202


namespace unique_hyperdeficient_number_l1432_143290

/-- Sum of divisors function -/
def f (n : ℕ) : ℕ := sorry

/-- A number is hyperdeficient if f(f(n)) = n + 3 -/
def is_hyperdeficient (n : ℕ) : Prop := f (f n) = n + 3

theorem unique_hyperdeficient_number : 
  ∃! n : ℕ, n > 0 ∧ is_hyperdeficient n :=
sorry

end unique_hyperdeficient_number_l1432_143290


namespace points_in_quadrants_I_and_II_l1432_143248

-- Define the set of points satisfying the inequalities
def S : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 > -1/2 * p.1 + 6 ∧ p.2 > 3 * p.1 - 4}

-- Define the first quadrant
def Q1 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 > 0 ∧ p.2 > 0}

-- Define the second quadrant
def Q2 : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0}

-- Theorem statement
theorem points_in_quadrants_I_and_II : S ⊆ Q1 ∪ Q2 := by
  sorry


end points_in_quadrants_I_and_II_l1432_143248


namespace quadratic_equation_properties_l1432_143292

/-- The quadratic equation x^2 - 2mx + m^2 + m - 3 = 0 has real roots -/
def has_real_roots (m : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - 2*m*x + m^2 + m - 3 = 0

/-- The range of m for which the equation has real roots -/
def m_range : Set ℝ :=
  {m : ℝ | m ≤ 3}

/-- The product of the roots of the equation -/
def root_product (m : ℝ) : ℝ := m^2 + m - 3

theorem quadratic_equation_properties :
  (∀ m : ℝ, has_real_roots m ↔ m ∈ m_range) ∧
  (∃ m : ℝ, root_product m = 17 ∧ m = -5) :=
sorry

end quadratic_equation_properties_l1432_143292


namespace unique_five_digit_number_l1432_143256

/-- Function to transform a digit according to the problem rules -/
def transformDigit (d : Nat) : Nat :=
  match d with
  | 2 => 5
  | 5 => 2
  | _ => d

/-- Function to transform a five-digit number according to the problem rules -/
def transformNumber (n : Nat) : Nat :=
  let d1 := n / 10000
  let d2 := (n / 1000) % 10
  let d3 := (n / 100) % 10
  let d4 := (n / 10) % 10
  let d5 := n % 10
  10000 * (transformDigit d1) + 1000 * (transformDigit d2) + 100 * (transformDigit d3) + 10 * (transformDigit d4) + (transformDigit d5)

/-- The main theorem statement -/
theorem unique_five_digit_number :
  ∃! x : Nat, 
    10000 ≤ x ∧ x < 100000 ∧  -- x is a five-digit number
    x % 2 = 1 ∧               -- x is odd
    transformNumber x = 2 * (x + 1) ∧ -- y = 2(x+1)
    x = 29995 := by
  sorry

end unique_five_digit_number_l1432_143256


namespace divisibility_rule_l1432_143241

theorem divisibility_rule (x y : ℕ+) (h : (1000 * y + x : ℕ) > 0) :
  (((x : ℤ) - (y : ℤ)) % 7 = 0 ∨ ((x : ℤ) - (y : ℤ)) % 11 = 0) →
  ((1000 * y + x : ℕ) % 7 = 0 ∨ (1000 * y + x : ℕ) % 11 = 0) := by
  sorry

end divisibility_rule_l1432_143241


namespace prob_spade_or_king_l1432_143298

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of spades in a standard deck -/
def num_spades : ℕ := 13

/-- The number of kings in a standard deck -/
def num_kings : ℕ := 4

/-- The number of cards that are both spades and kings -/
def overlap : ℕ := 1

/-- The probability of drawing a spade or a king from a standard 52-card deck -/
theorem prob_spade_or_king : 
  (num_spades + num_kings - overlap : ℚ) / deck_size = 4 / 13 := by
  sorry

end prob_spade_or_king_l1432_143298


namespace square_sum_greater_than_product_l1432_143243

theorem square_sum_greater_than_product {a b : ℝ} (h : a > b) : a^2 + b^2 > a*b := by
  sorry

end square_sum_greater_than_product_l1432_143243


namespace cookies_left_to_take_home_l1432_143236

def initial_cookies : ℕ := 120
def dozen : ℕ := 12
def morning_sales : ℕ := 3 * dozen
def lunch_sales : ℕ := 57
def afternoon_sales : ℕ := 16

theorem cookies_left_to_take_home : 
  initial_cookies - morning_sales - lunch_sales - afternoon_sales = 11 := by
  sorry

end cookies_left_to_take_home_l1432_143236


namespace solution_is_rhombus_l1432_143270

def is_solution (x y : ℝ) : Prop :=
  max (|x + y|) (|x - y|) = 1

def rhombus_vertices : Set (ℝ × ℝ) :=
  {(-1, 0), (1, 0), (0, -1), (0, 1)}

theorem solution_is_rhombus :
  {p : ℝ × ℝ | is_solution p.1 p.2} = rhombus_vertices := by sorry

end solution_is_rhombus_l1432_143270


namespace a_142_equals_1995_and_unique_l1432_143237

def p (n : ℕ) : ℕ := sorry

def q (n : ℕ) : ℕ := sorry

def a : ℕ → ℕ
  | 0 => 1
  | n + 1 => a n * p (a n) / q (a n)

theorem a_142_equals_1995_and_unique :
  a 142 = 1995 ∧ ∀ n : ℕ, n ≠ 142 → a n ≠ 1995 := by sorry

end a_142_equals_1995_and_unique_l1432_143237


namespace right_triangle_third_side_l1432_143264

theorem right_triangle_third_side : ∀ (a b c : ℝ),
  (a = 3 ∧ b = 2) ∨ (a = 2 ∧ b = 3) →
  (c^2 = a^2 + b^2 ∨ a^2 = b^2 + c^2 ∨ b^2 = a^2 + c^2) →
  c = Real.sqrt 13 ∨ c = Real.sqrt 5 := by
  sorry

end right_triangle_third_side_l1432_143264


namespace lattice_points_on_segment_l1432_143219

/-- The number of lattice points on a line segment with given endpoints -/
def latticePointCount (x₁ y₁ x₂ y₂ : ℤ) : ℕ :=
  sorry

/-- Theorem: The number of lattice points on the line segment from (7,23) to (61,353) is 7 -/
theorem lattice_points_on_segment : latticePointCount 7 23 61 353 = 7 := by
  sorry

end lattice_points_on_segment_l1432_143219


namespace constant_t_value_l1432_143282

theorem constant_t_value : ∃ t : ℝ, 
  (∀ x : ℝ, (3*x^2 - 4*x + 5) * (5*x^2 + t*x + 15) = 15*x^4 - 47*x^3 + 115*x^2 - 110*x + 75) ∧ 
  t = -10 := by
  sorry

end constant_t_value_l1432_143282


namespace smallest_prime_divisor_of_sum_l1432_143210

theorem smallest_prime_divisor_of_sum (p : ℕ) : 
  (p.Prime ∧ p ∣ (7^15 + 9^7)) → p = 2 := by
  sorry

end smallest_prime_divisor_of_sum_l1432_143210


namespace company_fund_proof_l1432_143285

theorem company_fund_proof (n : ℕ) (initial_fund : ℕ) : 
  (80 * n - 20 = initial_fund) →  -- Planned $80 bonus, $20 short
  (70 * n + 75 = initial_fund) →  -- Actual $70 bonus, $75 left
  initial_fund = 700 := by
sorry

end company_fund_proof_l1432_143285


namespace steves_pool_filling_time_l1432_143277

/-- The time required to fill Steve's pool -/
theorem steves_pool_filling_time :
  let pool_capacity : ℝ := 30000  -- gallons
  let num_hoses : ℕ := 5
  let flow_rate_per_hose : ℝ := 3  -- gallons per minute
  let minutes_per_hour : ℕ := 60
  
  let total_flow_rate : ℝ := num_hoses * flow_rate_per_hose  -- gallons per minute
  let hourly_flow_rate : ℝ := total_flow_rate * minutes_per_hour  -- gallons per hour
  let filling_time : ℝ := pool_capacity / hourly_flow_rate  -- hours
  
  ⌈filling_time⌉ = 34 := by
  sorry

end steves_pool_filling_time_l1432_143277


namespace first_half_speed_l1432_143278

theorem first_half_speed
  (total_distance : ℝ)
  (total_time : ℝ)
  (second_half_speed : ℝ)
  (h1 : total_distance = 300)
  (h2 : total_time = 11)
  (h3 : second_half_speed = 25)
  : ∃ (first_half_speed : ℝ),
    first_half_speed = 30 ∧
    total_distance / 2 / first_half_speed +
    total_distance / 2 / second_half_speed = total_time :=
by sorry

end first_half_speed_l1432_143278


namespace bobs_corn_field_efficiency_l1432_143281

/-- Given a corn field with a certain number of rows and stalks per row,
    and a total harvest in bushels, calculate the number of stalks needed per bushel. -/
def stalks_per_bushel (rows : ℕ) (stalks_per_row : ℕ) (total_bushels : ℕ) : ℕ :=
  (rows * stalks_per_row) / total_bushels

/-- Theorem stating that for Bob's corn field, 8 stalks are needed per bushel. -/
theorem bobs_corn_field_efficiency :
  stalks_per_bushel 5 80 50 = 8 := by
  sorry

end bobs_corn_field_efficiency_l1432_143281


namespace bike_ride_distance_l1432_143216

/-- Calculates the total distance of a 3-hour bike ride given specific conditions -/
theorem bike_ride_distance (second_hour_distance : ℝ) 
  (h1 : second_hour_distance = 18)
  (h2 : second_hour_distance = 1.2 * (second_hour_distance / 1.2))
  (h3 : 1.25 * second_hour_distance = 22.5) :
  (second_hour_distance / 1.2) + second_hour_distance + (1.25 * second_hour_distance) = 55.5 := by
sorry

end bike_ride_distance_l1432_143216


namespace terrier_to_poodle_groom_ratio_l1432_143291

def poodle_groom_time : ℕ := 30
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8
def total_groom_time : ℕ := 210

theorem terrier_to_poodle_groom_ratio :
  ∃ (terrier_groom_time : ℕ),
    terrier_groom_time * num_terriers + poodle_groom_time * num_poodles = total_groom_time ∧
    2 * terrier_groom_time = poodle_groom_time :=
by sorry

end terrier_to_poodle_groom_ratio_l1432_143291


namespace original_curve_equation_l1432_143206

/-- Given a curve C in a Cartesian coordinate system that undergoes a stretching transformation,
    this theorem proves the equation of the original curve C. -/
theorem original_curve_equation
  (C : Set (ℝ × ℝ)) -- The original curve C
  (stretching : ℝ × ℝ → ℝ × ℝ) -- The stretching transformation
  (h_stretching : ∀ (x y : ℝ), stretching (x, y) = (3 * x, y)) -- Definition of the stretching
  (h_transformed : ∀ (x y : ℝ), (x, y) ∈ C → x^2 + 9*y^2 = 9) -- Equation of the transformed curve
  : ∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + y^2 = 1 := by sorry

end original_curve_equation_l1432_143206


namespace fraction_zero_implies_negative_one_l1432_143239

theorem fraction_zero_implies_negative_one (x : ℝ) :
  (x^2 - 1) / (x - 1) = 0 ∧ x - 1 ≠ 0 → x = -1 := by
  sorry

end fraction_zero_implies_negative_one_l1432_143239


namespace normal_distribution_probability_l1432_143244

/-- A random variable following a normal distribution -/
structure NormalRV where
  μ : ℝ
  σ : ℝ
  σ_pos : σ > 0

/-- The probability that a normal random variable is less than a given value -/
noncomputable def prob_less (ξ : NormalRV) (x : ℝ) : ℝ := sorry

/-- The probability that a normal random variable is greater than a given value -/
noncomputable def prob_greater (ξ : NormalRV) (x : ℝ) : ℝ := sorry

/-- The probability that a normal random variable is between two given values -/
noncomputable def prob_between (ξ : NormalRV) (a b : ℝ) : ℝ := sorry

/-- Theorem: For a normally distributed random variable ξ with 
    P(ξ < -2) = P(ξ > 2) = 0.3, P(-2 < ξ < 0) = 0.2 -/
theorem normal_distribution_probability (ξ : NormalRV) 
    (h1 : prob_less ξ (-2) = 0.3)
    (h2 : prob_greater ξ 2 = 0.3) :
    prob_between ξ (-2) 0 = 0.2 := by sorry

end normal_distribution_probability_l1432_143244


namespace initial_ball_count_l1432_143229

theorem initial_ball_count (initial_blue : ℕ) (removed_blue : ℕ) (final_probability : ℚ) : 
  initial_blue = 7 → 
  removed_blue = 3 → 
  final_probability = 1/3 → 
  ∃ (total : ℕ), total = 15 ∧ 
    (initial_blue - removed_blue : ℚ) / (total - removed_blue : ℚ) = final_probability :=
by sorry

end initial_ball_count_l1432_143229


namespace f_f_zero_l1432_143284

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then -2 * Real.exp x else Real.log x

theorem f_f_zero (x : ℝ) : f (f x) = 0 ↔ x = Real.exp 1 := by
  sorry

end f_f_zero_l1432_143284


namespace wedge_volume_l1432_143280

/-- The volume of a wedge formed by two planar cuts in a cylindrical log. -/
theorem wedge_volume (d : ℝ) (angle : ℝ) (h : ℝ) (m : ℕ) : 
  d = 16 →
  angle = 60 →
  h = d →
  (1 / 6) * π * (d / 2)^2 * h = m * π →
  m = 171 := by
  sorry

end wedge_volume_l1432_143280


namespace oplus_neg_two_three_oplus_inequality_l1432_143287

-- Define the ⊕ operation
def oplus (a b : ℝ) : ℝ := 2 * a - 3 * b

-- Theorem 1: (-2) ⊕ 3 = -13
theorem oplus_neg_two_three : oplus (-2) 3 = -13 := by sorry

-- Theorem 2: For all real x, ((-3/2x+1) ⊕ (-1-2x)) > ((3x-2) ⊕ (x+1))
theorem oplus_inequality (x : ℝ) : oplus (-3/2*x+1) (-1-2*x) > oplus (3*x-2) (x+1) := by sorry

end oplus_neg_two_three_oplus_inequality_l1432_143287


namespace binary_to_decimal_l1432_143267

theorem binary_to_decimal : 
  (1 * 2^0 + 1 * 2^1 + 0 * 2^2 + 1 * 2^3 + 1 * 2^4 : ℕ) = 27 := by
  sorry

end binary_to_decimal_l1432_143267


namespace total_groom_time_is_210_l1432_143268

/-- Time to groom a poodle in minutes -/
def poodle_groom_time : ℕ := 30

/-- Time to groom a terrier in minutes -/
def terrier_groom_time : ℕ := poodle_groom_time / 2

/-- Number of poodles to groom -/
def num_poodles : ℕ := 3

/-- Number of terriers to groom -/
def num_terriers : ℕ := 8

/-- Total grooming time for all dogs -/
def total_groom_time : ℕ := num_poodles * poodle_groom_time + num_terriers * terrier_groom_time

theorem total_groom_time_is_210 : total_groom_time = 210 := by
  sorry

end total_groom_time_is_210_l1432_143268


namespace quadratic_no_solution_l1432_143226

theorem quadratic_no_solution (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * x - 1 ≠ 0) → a < -1 := by
  sorry

end quadratic_no_solution_l1432_143226


namespace vector_subtraction_l1432_143255

/-- Given plane vectors a and b, prove that a - 2b equals (7, 3) -/
theorem vector_subtraction (a b : ℝ × ℝ) (ha : a = (3, 5)) (hb : b = (-2, 1)) :
  a - 2 • b = (7, 3) := by sorry

end vector_subtraction_l1432_143255


namespace min_value_sum_l1432_143251

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 / x + 1 / y = 1) :
  x + y ≥ 4 + 2 * Real.sqrt 3 := by
  sorry

end min_value_sum_l1432_143251


namespace batsman_average_l1432_143214

/-- Calculates the average runs for a batsman given two sets of matches --/
def calculate_average (matches1 : ℕ) (average1 : ℕ) (matches2 : ℕ) (average2 : ℕ) : ℚ :=
  let total_runs := matches1 * average1 + matches2 * average2
  let total_matches := matches1 + matches2
  (total_runs : ℚ) / total_matches

/-- Proves that the batsman's average for 30 matches is 31 runs --/
theorem batsman_average : calculate_average 20 40 10 13 = 31 := by
  sorry

#eval calculate_average 20 40 10 13

end batsman_average_l1432_143214


namespace total_average_marks_specific_classes_l1432_143240

/-- The total average marks of students in three classes -/
def total_average_marks (class1_students : ℕ) (class1_avg : ℚ)
                        (class2_students : ℕ) (class2_avg : ℚ)
                        (class3_students : ℕ) (class3_avg : ℚ) : ℚ :=
  (class1_avg * class1_students + class2_avg * class2_students + class3_avg * class3_students) /
  (class1_students + class2_students + class3_students)

/-- Theorem stating the total average marks of students in three specific classes -/
theorem total_average_marks_specific_classes :
  total_average_marks 47 52 33 68 40 75 = 7688 / 120 :=
by sorry

end total_average_marks_specific_classes_l1432_143240


namespace correct_cookies_in_partial_bag_edgars_cookies_l1432_143296

/-- Represents the number of cookies in a paper bag that is not full. -/
def cookiesInPartialBag (totalCookies bagCapacity : ℕ) : ℕ :=
  totalCookies % bagCapacity

/-- Proves that the number of cookies in a partial bag is correct. -/
theorem correct_cookies_in_partial_bag (totalCookies bagCapacity : ℕ) 
    (h1 : bagCapacity > 0) (h2 : totalCookies ≥ bagCapacity) :
  cookiesInPartialBag totalCookies bagCapacity = 
    totalCookies - bagCapacity * (totalCookies / bagCapacity) :=
by sorry

/-- The specific problem instance. -/
theorem edgars_cookies :
  cookiesInPartialBag 292 16 = 4 :=
by sorry

end correct_cookies_in_partial_bag_edgars_cookies_l1432_143296


namespace factoring_transformation_l1432_143200

-- Define the concept of factoring
def is_factored (f g : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x ∧ ∃ (p q : ℝ → ℝ), g x = p x * q x

-- Define the specific expression
def left_expr : ℝ → ℝ := λ x ↦ x^2 - 4
def right_expr : ℝ → ℝ := λ x ↦ (x + 2) * (x - 2)

-- Theorem statement
theorem factoring_transformation :
  is_factored left_expr right_expr :=
sorry

end factoring_transformation_l1432_143200


namespace triangle_ratio_sqrt_two_l1432_143213

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a * sin(A) * sin(B) + b * cos²(A) = √2 * a, then b/a = √2 -/
theorem triangle_ratio_sqrt_two (a b c : ℝ) (A B C : ℝ) 
    (h : a * Real.sin A * Real.sin B + b * Real.cos A ^ 2 = Real.sqrt 2 * a) 
    (h_positive : a > 0) : b / a = Real.sqrt 2 := by
  sorry

end triangle_ratio_sqrt_two_l1432_143213


namespace career_d_degrees_l1432_143201

/-- Represents the ratio of male to female students -/
def maleToFemaleRatio : Rat := 2 / 3

/-- Represents the percentage of males preferring each career -/
def malePreference : Fin 6 → Rat
| 0 => 25 / 100  -- Career A
| 1 => 15 / 100  -- Career B
| 2 => 30 / 100  -- Career C
| 3 => 40 / 100  -- Career D
| 4 => 20 / 100  -- Career E
| 5 => 35 / 100  -- Career F

/-- Represents the percentage of females preferring each career -/
def femalePreference : Fin 6 → Rat
| 0 => 50 / 100  -- Career A
| 1 => 40 / 100  -- Career B
| 2 => 10 / 100  -- Career C
| 3 => 20 / 100  -- Career D
| 4 => 30 / 100  -- Career E
| 5 => 25 / 100  -- Career F

/-- Calculates the degrees in a circle graph for a given career -/
def careerDegrees (careerIndex : Fin 6) : ℚ :=
  let totalStudents := maleToFemaleRatio + 1
  let maleStudents := maleToFemaleRatio
  let femaleStudents := 1
  let studentsPreferringCareer := 
    maleStudents * malePreference careerIndex + femaleStudents * femalePreference careerIndex
  (studentsPreferringCareer / totalStudents) * 360

/-- Theorem stating that Career D should be represented by 100.8 degrees in the circle graph -/
theorem career_d_degrees : careerDegrees 3 = 100.8 := by sorry

end career_d_degrees_l1432_143201


namespace fraction_sum_non_negative_l1432_143224

theorem fraction_sum_non_negative (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  1 / (a - b) + 1 / (b - c) + 4 / (c - a) ≥ 0 := by sorry

end fraction_sum_non_negative_l1432_143224


namespace cos_angle_minus_pi_half_l1432_143288

/-- 
Given an angle α in a plane rectangular coordinate system whose terminal side 
passes through the point (4, -3), prove that cos(α - π/2) = -3/5.
-/
theorem cos_angle_minus_pi_half (α : Real) : 
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 4 ∧ r * Real.sin α = -3) →
  Real.cos (α - π/2) = -3/5 := by
  sorry

end cos_angle_minus_pi_half_l1432_143288


namespace triangle_altitude_sum_perfect_square_l1432_143221

theorem triangle_altitude_sum_perfect_square (x y z : ℤ) :
  x > 0 ∧ y > 0 ∧ z > 0 →
  (∃ (h_x h_y h_z : ℝ), 
    h_x > 0 ∧ h_y > 0 ∧ h_z > 0 ∧
    (h_x = h_y + h_z ∨ h_y = h_x + h_z ∨ h_z = h_x + h_y) ∧
    x * h_x = y * h_y ∧ y * h_y = z * h_z) →
  ∃ (n : ℤ), x^2 + y^2 + z^2 = n^2 :=
by sorry

end triangle_altitude_sum_perfect_square_l1432_143221


namespace functional_equation_solution_l1432_143234

-- Define the property that f must satisfy
def satisfies_property (f : ℝ → ℝ) : Prop :=
  ∀ x y u v : ℝ, (f x + f y) * (f u + f v) = f (x*u - y*v) + f (x*v + y*u)

-- State the theorem
theorem functional_equation_solution :
  ∀ f : ℝ → ℝ, satisfies_property f →
    (∀ x : ℝ, f x = x^2) ∨ (∀ x : ℝ, f x = (1/2 : ℝ)) :=
sorry

end functional_equation_solution_l1432_143234


namespace min_odd_integers_l1432_143269

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum_ab : a + b = 32)
  (sum_abcd : a + b + c + d = 47)
  (sum_abcdef : a + b + c + d + e + f = 66) :
  ∃ (odds : Finset ℤ), odds ⊆ {a, b, c, d, e, f} ∧ 
    odds.card = 2 ∧ 
    (∀ x ∈ odds, Odd x) ∧
    (∀ y ∈ {a, b, c, d, e, f} \ odds, Even y) :=
sorry

end min_odd_integers_l1432_143269


namespace algebraic_expression_value_l1432_143230

theorem algebraic_expression_value (a b : ℝ) (h : a - b - 2 = 0) :
  a^2 - b^2 - 4*a = -4 := by sorry

end algebraic_expression_value_l1432_143230


namespace valid_choices_count_l1432_143261

/-- The number of objects placed along a circle -/
def n : ℕ := 32

/-- The number of objects to be chosen -/
def k : ℕ := 3

/-- The number of ways to choose k objects from n objects -/
def total_ways : ℕ := n.choose k

/-- The number of pairs of adjacent objects -/
def adjacent_pairs : ℕ := n

/-- The number of pairs of diametrically opposite objects -/
def opposite_pairs : ℕ := n / 2

/-- The number of remaining objects after choosing two adjacent or opposite objects -/
def remaining_objects : ℕ := n - 4

/-- The theorem stating the number of valid ways to choose objects -/
theorem valid_choices_count : 
  total_ways - adjacent_pairs * remaining_objects - opposite_pairs * remaining_objects + n = 3648 := by
  sorry

end valid_choices_count_l1432_143261


namespace fuse_length_safety_l1432_143242

theorem fuse_length_safety (safe_distance : ℝ) (fuse_speed : ℝ) (operator_speed : ℝ) 
  (h1 : safe_distance = 400)
  (h2 : fuse_speed = 1.2)
  (h3 : operator_speed = 5) :
  ∃ (min_length : ℝ), min_length > 96 ∧ 
  ∀ (fuse_length : ℝ), fuse_length > min_length → 
  (fuse_length / fuse_speed) > (safe_distance / operator_speed) := by
  sorry

end fuse_length_safety_l1432_143242


namespace megan_popsicle_consumption_l1432_143205

/-- The number of Popsicles consumed in a given time period -/
def popsicles_consumed (rate_minutes : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / rate_minutes

theorem megan_popsicle_consumption :
  popsicles_consumed 20 340 = 17 := by
  sorry

#eval popsicles_consumed 20 340

end megan_popsicle_consumption_l1432_143205


namespace combined_weight_is_63_l1432_143271

/-- The combined weight of candles made by Ethan -/
def combined_weight : ℕ :=
  let beeswax_per_candle : ℕ := 8
  let coconut_oil_per_candle : ℕ := 1
  let total_candles : ℕ := 10 - 3
  let weight_per_candle : ℕ := beeswax_per_candle + coconut_oil_per_candle
  total_candles * weight_per_candle

/-- Theorem stating that the combined weight of candles is 63 ounces -/
theorem combined_weight_is_63 : combined_weight = 63 := by
  sorry

end combined_weight_is_63_l1432_143271


namespace smallest_integer_m_l1432_143283

theorem smallest_integer_m (x y m : ℝ) : 
  (2 * x + y = 4) →
  (x + 2 * y = -3 * m + 2) →
  (x - y > -3/2) →
  (∀ k : ℤ, k < m → ¬(∃ x y : ℝ, 2 * x + y = 4 ∧ x + 2 * y = -3 * (k : ℝ) + 2 ∧ x - y > -3/2)) →
  m = -1 :=
by sorry

end smallest_integer_m_l1432_143283


namespace smallest_angle_in_345_ratio_triangle_l1432_143262

theorem smallest_angle_in_345_ratio_triangle (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a + b + c = 180 →
  b = (4/3) * a →
  c = (5/3) * a →
  a = 45 := by
sorry

end smallest_angle_in_345_ratio_triangle_l1432_143262


namespace two_n_squares_implies_n_squares_l1432_143217

theorem two_n_squares_implies_n_squares (n : ℕ) 
  (h : ∃ (k m : ℤ), 2 * n = k^2 + m^2) : 
  ∃ (a b : ℚ), n = a^2 + b^2 := by
sorry

end two_n_squares_implies_n_squares_l1432_143217


namespace tileable_rectangle_divisibility_l1432_143223

/-- A rectangle is (a, b)-tileable if it can be covered by non-overlapping a × b tiles -/
def is_tileable (m n a b : ℕ) : Prop := sorry

/-- Main theorem: If k divides a and b, and an m × n rectangle is (a, b)-tileable, 
    then 2k divides m or 2k divides n -/
theorem tileable_rectangle_divisibility 
  (k a b m n : ℕ) 
  (h1 : k ∣ a) 
  (h2 : k ∣ b) 
  (h3 : is_tileable m n a b) : 
  (2 * k) ∣ m ∨ (2 * k) ∣ n :=
sorry

end tileable_rectangle_divisibility_l1432_143223


namespace square_perimeter_from_circle_l1432_143235

theorem square_perimeter_from_circle (circle_perimeter : ℝ) : 
  circle_perimeter = 52.5 → 
  ∃ (square_perimeter : ℝ), square_perimeter = 210 / Real.pi := by
  sorry

end square_perimeter_from_circle_l1432_143235


namespace square_sum_equality_l1432_143233

theorem square_sum_equality (a b : ℕ) (h1 : a^2 = 225) (h2 : b^2 = 25) :
  a^2 + 2*a*b + b^2 = 400 := by
  sorry

end square_sum_equality_l1432_143233


namespace points_and_lines_l1432_143204

theorem points_and_lines (n : ℕ) : 
  (n * (n - 1)) / 2 ≤ 45 → n = 10 := by
  sorry

end points_and_lines_l1432_143204


namespace permutations_not_divisible_by_three_l1432_143246

/-- The number of permutations of 1 to n where 1 is fixed and each number differs from its neighbors by at most 2 -/
def p (n : ℕ) : ℕ :=
  if n ≤ 2 then 1
  else if n = 3 then 2
  else p (n - 1) + p (n - 3) + 1

/-- The theorem stating that the number of permutations for 1996 is not divisible by 3 -/
theorem permutations_not_divisible_by_three :
  ¬ (3 ∣ p 1996) :=
sorry

end permutations_not_divisible_by_three_l1432_143246


namespace rectangle_cover_theorem_l1432_143249

/-- An increasing function from [0, 1] to [0, 1] -/
def IncreasingFunction := {f : ℝ → ℝ | Monotone f ∧ Set.range f ⊆ Set.Icc 0 1}

/-- A rectangle with sides parallel to the coordinate axes -/
structure Rectangle where
  x : ℝ
  y : ℝ
  width : ℝ
  height : ℝ

/-- The area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- A set of rectangles covers the graph of a function -/
def covers (rs : Set Rectangle) (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ Set.Icc 0 1, ∃ r ∈ rs, x ∈ Set.Icc r.x (r.x + r.width) ∧ f x ∈ Set.Icc r.y (r.y + r.height)

/-- Main theorem -/
theorem rectangle_cover_theorem (f : IncreasingFunction) (n : ℕ) :
  ∃ (rs : Set Rectangle), (∀ r ∈ rs, r.area = 1 / (2 * n)) ∧ covers rs f := by sorry

end rectangle_cover_theorem_l1432_143249


namespace unique_solution_l1432_143294

def A : Nat := 89252525 -- ... (200 digits total)

def B (x y : Nat) : Nat := 444 * x * 100000 + 18 * 1000 + y * 10 + 27

def digit_at (n : Nat) (pos : Nat) : Nat :=
  (n / (10 ^ (pos - 1))) % 10

theorem unique_solution :
  ∃! (x y : Nat),
    x < 10 ∧ y < 10 ∧
    digit_at (A * B x y) 53 = 1 ∧
    digit_at (A * B x y) 54 = 0 ∧
    x = 4 ∧ y = 6 := by sorry

end unique_solution_l1432_143294


namespace circular_garden_radius_l1432_143227

theorem circular_garden_radius (r : ℝ) (h : r > 0) :
  2 * Real.pi * r = (1 / 4) * Real.pi * r^2 → r = 8 := by
  sorry

end circular_garden_radius_l1432_143227


namespace total_cakes_eaten_l1432_143275

def monday_cakes : ℕ := 6
def friday_cakes : ℕ := 9
def saturday_cakes : ℕ := 3 * monday_cakes

theorem total_cakes_eaten : monday_cakes + friday_cakes + saturday_cakes = 33 := by
  sorry

end total_cakes_eaten_l1432_143275


namespace triangle_ABC_is_right_angled_l1432_143265

/-- Triangle ABC is defined by points A(5, -2), B(1, 5), and C(-1, 2) in a 2D Euclidean space -/
def A : ℝ × ℝ := (5, -2)
def B : ℝ × ℝ := (1, 5)
def C : ℝ × ℝ := (-1, 2)

/-- Distance squared between two points -/
def dist_squared (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

/-- Triangle ABC is right-angled -/
theorem triangle_ABC_is_right_angled : 
  dist_squared A B = dist_squared B C + dist_squared C A :=
by sorry

end triangle_ABC_is_right_angled_l1432_143265


namespace running_program_weekly_increase_l1432_143247

theorem running_program_weekly_increase 
  (initial_distance : ℝ) 
  (final_distance : ℝ) 
  (program_duration : ℕ) 
  (increase_duration : ℕ) 
  (h1 : initial_distance = 3)
  (h2 : final_distance = 7)
  (h3 : program_duration = 5)
  (h4 : increase_duration = 4)
  : (final_distance - initial_distance) / increase_duration = 1 := by
  sorry

end running_program_weekly_increase_l1432_143247


namespace parabola_focus_l1432_143238

/-- A parabola is defined by the equation x^2 = 4y -/
def is_parabola (f : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, f (x, y) ↔ x^2 = 4*y

/-- The focus of a parabola is a point (h, k) -/
def is_focus (f : ℝ × ℝ → Prop) (h k : ℝ) : Prop :=
  is_parabola f ∧ (h, k) = (0, 1)

/-- Theorem: The focus of the parabola x^2 = 4y is (0, 1) -/
theorem parabola_focus :
  ∀ f : ℝ × ℝ → Prop, is_parabola f → is_focus f 0 1 := by
  sorry

end parabola_focus_l1432_143238


namespace sqrt_500_simplification_l1432_143209

theorem sqrt_500_simplification : Real.sqrt 500 = 10 * Real.sqrt 5 := by
  sorry

end sqrt_500_simplification_l1432_143209
