import Mathlib

namespace NUMINAMATH_CALUDE_complex_statements_l482_48255

open Complex

theorem complex_statements :
  (∃ z : ℂ, z = 1 - I ∧ Complex.abs (2 / z + z^2) = Real.sqrt 2) ∧
  (∃ z : ℂ, z = 1 / I ∧ (z^5 + 1).re > 0 ∧ (z^5 + 1).im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_statements_l482_48255


namespace NUMINAMATH_CALUDE_carton_height_calculation_l482_48275

/-- Calculates the height of a carton given its base dimensions, soap box dimensions, and maximum capacity -/
theorem carton_height_calculation (carton_length carton_width : ℕ) 
  (box_length box_width box_height : ℕ) (max_boxes : ℕ) : 
  carton_length = 25 ∧ carton_width = 42 ∧ 
  box_length = 7 ∧ box_width = 6 ∧ box_height = 5 ∧
  max_boxes = 300 →
  (max_boxes / ((carton_length / box_length) * (carton_width / box_width))) * box_height = 70 :=
by sorry

end NUMINAMATH_CALUDE_carton_height_calculation_l482_48275


namespace NUMINAMATH_CALUDE_right_triangle_area_l482_48278

/-- A right triangle ABC in the xy-plane with specific properties -/
structure RightTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_right_angle_at_C : (A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2) = 0
  hypotenuse_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 900
  median_A_on_y_eq_x : A.1 = A.2
  median_B_on_y_eq_x_plus_1 : B.2 = B.1 + 1

/-- The area of the right triangle ABC is 448 -/
theorem right_triangle_area (t : RightTriangle) : 
  (1/2) * abs ((t.A.1 * (t.B.2 - t.C.2) + t.B.1 * (t.C.2 - t.A.2) + t.C.1 * (t.A.2 - t.B.2))) = 448 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l482_48278


namespace NUMINAMATH_CALUDE_same_number_on_four_dice_l482_48236

theorem same_number_on_four_dice (n : ℕ) (h : n = 8) :
  (1 : ℚ) / (n ^ 3) = 1 / 512 :=
by sorry

end NUMINAMATH_CALUDE_same_number_on_four_dice_l482_48236


namespace NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l482_48221

theorem smallest_prime_dividing_sum : ∃ (p : ℕ), 
  Nat.Prime p ∧ 
  p ∣ (2^11 + 7^13) ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ (2^11 + 7^13) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_dividing_sum_l482_48221


namespace NUMINAMATH_CALUDE_distance_between_points_l482_48284

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (-5, 2)
  let p2 : ℝ × ℝ := (7, 7)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 13 := by
sorry

end NUMINAMATH_CALUDE_distance_between_points_l482_48284


namespace NUMINAMATH_CALUDE_log_problem_l482_48243

-- Define the logarithm function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_problem :
  4 * lg 2 + 3 * lg 5 - lg (1/5) = 4 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l482_48243


namespace NUMINAMATH_CALUDE_min_value_problem_l482_48289

theorem min_value_problem (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 9) 
  (h2 : e * f * g * h = 4) : 
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 24 := by
  sorry

end NUMINAMATH_CALUDE_min_value_problem_l482_48289


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l482_48215

theorem at_least_one_not_less_than_two 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_three : a + b + c = 3) : 
  (a + 1/b ≥ 2) ∨ (b + 1/c ≥ 2) ∨ (c + 1/a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l482_48215


namespace NUMINAMATH_CALUDE_classmates_not_invited_l482_48257

/-- A simple graph representing friendships among classmates -/
structure FriendshipGraph where
  vertices : Finset Nat
  edges : Finset (Nat × Nat)
  symm : ∀ {a b}, (a, b) ∈ edges → (b, a) ∈ edges
  irrefl : ∀ a, (a, a) ∉ edges

/-- The set of vertices reachable within n steps from a given vertex -/
def reachableWithin (G : FriendshipGraph) (start : Nat) (n : Nat) : Finset Nat :=
  sorry

/-- The main theorem -/
theorem classmates_not_invited (G : FriendshipGraph) (mark : Nat) : 
  G.vertices.card = 25 →
  mark ∈ G.vertices →
  (G.vertices \ reachableWithin G mark 3).card = 5 := by
  sorry

end NUMINAMATH_CALUDE_classmates_not_invited_l482_48257


namespace NUMINAMATH_CALUDE_quadratic_factorization_l482_48287

theorem quadratic_factorization (x : ℝ) : x^2 - 30*x + 225 = (x - 15)^2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l482_48287


namespace NUMINAMATH_CALUDE_sum_of_edge_lengths_specific_prism_l482_48244

/-- Regular hexagonal prism with given base side length and height -/
structure RegularHexagonalPrism where
  base_side_length : ℝ
  height : ℝ

/-- Calculate the sum of the lengths of all edges of a regular hexagonal prism -/
def sum_of_edge_lengths (prism : RegularHexagonalPrism) : ℝ :=
  12 * prism.base_side_length + 6 * prism.height

/-- Theorem: The sum of edge lengths for a regular hexagonal prism with base side 6 cm and height 11 cm is 138 cm -/
theorem sum_of_edge_lengths_specific_prism :
  let prism : RegularHexagonalPrism := ⟨6, 11⟩
  sum_of_edge_lengths prism = 138 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_edge_lengths_specific_prism_l482_48244


namespace NUMINAMATH_CALUDE_valid_x_values_l482_48209

def is_valid_x (x : ℕ) : Prop :=
  13 ≤ x ∧ x ≤ 20 ∧
  (132 + x) % 3 = 0 ∧
  ∃ (s : ℕ), 3 * s = 132 + 3 * x

theorem valid_x_values :
  ∀ x : ℕ, is_valid_x x ↔ (x = 15 ∨ x = 18) :=
sorry

end NUMINAMATH_CALUDE_valid_x_values_l482_48209


namespace NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l482_48273

theorem smaller_solution_of_quadratic (x : ℝ) : 
  x^2 + 8*x - 48 = 0 → (∃ y : ℝ, y^2 + 8*y - 48 = 0 ∧ y ≠ x) → x ≥ -12 :=
by sorry

end NUMINAMATH_CALUDE_smaller_solution_of_quadratic_l482_48273


namespace NUMINAMATH_CALUDE_brothers_catch_up_time_l482_48205

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

end NUMINAMATH_CALUDE_brothers_catch_up_time_l482_48205


namespace NUMINAMATH_CALUDE_divisor_power_equation_l482_48262

/-- The number of positive divisors of n -/
def d (n : ℕ+) : ℕ := sorry

/-- The statement of the problem -/
theorem divisor_power_equation :
  ∀ n k : ℕ+, ∀ p : ℕ,
  Prime p →
  (n : ℕ) ^ (d n) - 1 = p ^ (k : ℕ) →
  ((n = 2 ∧ k = 1 ∧ p = 3) ∨ (n = 3 ∧ k = 3 ∧ p = 2)) :=
by sorry

end NUMINAMATH_CALUDE_divisor_power_equation_l482_48262


namespace NUMINAMATH_CALUDE_cat_whiskers_problem_l482_48228

/-- Represents the number of whiskers for each cat -/
structure CatWhiskers where
  juniper : ℕ
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ
  whisper : ℕ
  bella : ℕ
  max : ℕ
  felix : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem cat_whiskers_problem (c : CatWhiskers) : 
  c.juniper = 12 ∧
  c.puffy = 3 * c.juniper ∧
  c.scruffy = 2 * c.puffy ∧
  c.buffy = (c.puffy + c.scruffy + c.juniper) / 3 ∧
  c.whisper = 2 * c.puffy ∧
  c.whisper = c.scruffy / 3 ∧
  c.bella = c.juniper + c.puffy - 4 ∧
  c.max = c.scruffy + c.buffy ∧
  c.felix = min c.juniper (min c.puffy (min c.scruffy (min c.buffy (min c.whisper (min c.bella c.max)))))
  →
  c.max = 112 := by
sorry

end NUMINAMATH_CALUDE_cat_whiskers_problem_l482_48228


namespace NUMINAMATH_CALUDE_banana_price_reduction_l482_48270

/-- Given a 50% reduction in banana prices allows buying 80 more dozens for 60000.25 rupees,
    prove the reduced price per dozen is 375.0015625 rupees. -/
theorem banana_price_reduction (original_price : ℝ) : 
  (2 * 60000.25 / original_price - 60000.25 / original_price = 80) → 
  (original_price / 2 = 375.0015625) :=
by
  sorry

end NUMINAMATH_CALUDE_banana_price_reduction_l482_48270


namespace NUMINAMATH_CALUDE_expression_factorization_l482_48232

theorem expression_factorization (b : ℝ) :
  (8 * b^3 + 104 * b^2 - 9) - (-9 * b^3 + b^2 - 9) = b^2 * (17 * b + 103) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l482_48232


namespace NUMINAMATH_CALUDE_probability_two_red_shoes_l482_48224

def total_shoes : ℕ := 8
def red_shoes : ℕ := 4
def green_shoes : ℕ := 4

theorem probability_two_red_shoes :
  (red_shoes : ℚ) / total_shoes * (red_shoes - 1) / (total_shoes - 1) = 3 / 14 :=
sorry

end NUMINAMATH_CALUDE_probability_two_red_shoes_l482_48224


namespace NUMINAMATH_CALUDE_convex_quadrilateral_probability_l482_48272

/-- The probability of forming a convex quadrilateral by selecting 4 chords at random from 8 points on a circle -/
theorem convex_quadrilateral_probability (n : ℕ) (k : ℕ) : 
  n = 8 → k = 4 → (Nat.choose n 2).choose k / (Nat.choose n k) = 2 / 585 :=
by sorry

end NUMINAMATH_CALUDE_convex_quadrilateral_probability_l482_48272


namespace NUMINAMATH_CALUDE_alpha_value_l482_48264

/-- Given that α is an acute angle and sin(α - 10°) = √3/2, prove that α = 70°. -/
theorem alpha_value (α : Real) (h1 : 0 < α ∧ α < 90) (h2 : Real.sin (α - 10) = Real.sqrt 3 / 2) : 
  α = 70 := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l482_48264


namespace NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l482_48201

/-- Given a geometric sequence with 10 terms, first term 6, and last term 93312,
    prove that the 7th term is 279936 -/
theorem seventh_term_of_geometric_sequence :
  ∀ (a : ℕ → ℝ),
    (∀ i j, a (i + 1) / a i = a (j + 1) / a j) →  -- geometric sequence condition
    a 1 = 6 →                                     -- first term is 6
    a 10 = 93312 →                                -- last term is 93312
    a 7 = 279936 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_of_geometric_sequence_l482_48201


namespace NUMINAMATH_CALUDE_lcm_of_coprime_integers_l482_48254

def is_lcm (a b l : ℕ) : Prop := 
  l ∣ a ∧ l ∣ b ∧ ∀ m, m ∣ a → m ∣ b → m ∣ l

def is_hcf (a b h : ℕ) : Prop :=
  h ∣ a ∧ h ∣ b ∧ ∀ m, m ∣ a → m ∣ b → m ∣ h

theorem lcm_of_coprime_integers (a b : ℕ) (h1 : a > 0) (h2 : b > 0) :
  (is_hcf a b 1) →
  (∃ L, is_lcm a b L) →
  (∃ M m : ℤ, (M - m = 38) ∧ 
    (∀ d : ℤ, (∃ a' b' : ℕ, a' > 0 ∧ b' > 0 ∧ is_hcf a' b' 1 ∧ (a' : ℤ) - (b' : ℤ) = d) → 
      m ≤ d ∧ d ≤ M)) →
  (∃ L, is_lcm a b L ∧ L = 40) :=
by sorry

end NUMINAMATH_CALUDE_lcm_of_coprime_integers_l482_48254


namespace NUMINAMATH_CALUDE_sin_cos_sum_13_17_l482_48299

theorem sin_cos_sum_13_17 : 
  Real.sin (13 * π / 180) * Real.cos (17 * π / 180) + 
  Real.cos (13 * π / 180) * Real.sin (17 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_sum_13_17_l482_48299


namespace NUMINAMATH_CALUDE_average_temperature_l482_48280

def temperature_data : List ℝ := [90, 90, 90, 79, 71]
def num_years : ℕ := 5

theorem average_temperature : 
  (List.sum temperature_data) / num_years = 84 := by
  sorry

end NUMINAMATH_CALUDE_average_temperature_l482_48280


namespace NUMINAMATH_CALUDE_train_departure_time_difference_l482_48217

/-- Proves that Train A leaves 40 minutes before Train B, given their speeds and overtake time --/
theorem train_departure_time_difference 
  (speed_A : ℝ) 
  (speed_B : ℝ) 
  (overtake_time : ℝ) 
  (h1 : speed_A = 60) 
  (h2 : speed_B = 80) 
  (h3 : overtake_time = 120) :
  ∃ (time_diff : ℝ), 
    time_diff = 40 ∧ 
    speed_A * (time_diff / 60 + overtake_time / 60) = speed_B * (overtake_time / 60) := by
  sorry


end NUMINAMATH_CALUDE_train_departure_time_difference_l482_48217


namespace NUMINAMATH_CALUDE_division_problem_l482_48204

theorem division_problem (dividend quotient remainder : ℕ) 
  (h1 : dividend = 690) 
  (h2 : quotient = 19) 
  (h3 : remainder = 6) :
  ∃ divisor : ℕ, 
    dividend = divisor * quotient + remainder ∧ 
    divisor = 36 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l482_48204


namespace NUMINAMATH_CALUDE_initial_boarders_l482_48212

theorem initial_boarders (initial_ratio_boarders initial_ratio_day_scholars : ℕ)
  (new_ratio_boarders new_ratio_day_scholars : ℕ)
  (new_boarders : ℕ) :
  initial_ratio_boarders = 7 →
  initial_ratio_day_scholars = 16 →
  new_ratio_boarders = 1 →
  new_ratio_day_scholars = 2 →
  new_boarders = 80 →
  ∃ (x : ℕ),
    x * initial_ratio_boarders + new_boarders = x * initial_ratio_day_scholars * new_ratio_boarders / new_ratio_day_scholars →
    x * initial_ratio_boarders = 560 :=
by sorry

end NUMINAMATH_CALUDE_initial_boarders_l482_48212


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_third_l482_48203

theorem tan_alpha_plus_pi_third (α β : Real) 
  (h1 : Real.tan (α + β) = 3/5)
  (h2 : Real.tan (β - Real.pi/3) = 1/4) : 
  Real.tan (α + Real.pi/3) = 7/23 := by sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_third_l482_48203


namespace NUMINAMATH_CALUDE_number_of_pupils_l482_48298

theorem number_of_pupils (total_people : ℕ) (parents : ℕ) (pupils : ℕ) : 
  total_people = 676 → parents = 22 → pupils = total_people - parents → pupils = 654 := by
  sorry

end NUMINAMATH_CALUDE_number_of_pupils_l482_48298


namespace NUMINAMATH_CALUDE_triangle_inequality_sum_negative_l482_48286

theorem triangle_inequality_sum_negative 
  (a b c x y z : ℝ) 
  (h1 : 0 < b - c) 
  (h2 : b - c < a) 
  (h3 : a < b + c) 
  (h4 : a * x + b * y + c * z = 0) : 
  a * y * z + b * z * x + c * x * y < 0 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_sum_negative_l482_48286


namespace NUMINAMATH_CALUDE_gnomes_distribution_l482_48290

/-- Given a street with houses and gnomes, calculates the number of gnomes in each of the first few houses -/
def gnomes_per_house (total_houses : ℕ) (total_gnomes : ℕ) (last_house_gnomes : ℕ) : ℕ :=
  (total_gnomes - last_house_gnomes) / (total_houses - 1)

/-- Theorem stating that under given conditions, each of the first few houses has 3 gnomes -/
theorem gnomes_distribution (total_houses : ℕ) (total_gnomes : ℕ) (last_house_gnomes : ℕ)
  (h1 : total_houses = 5)
  (h2 : total_gnomes = 20)
  (h3 : last_house_gnomes = 8) :
  gnomes_per_house total_houses total_gnomes last_house_gnomes = 3 := by
  sorry

end NUMINAMATH_CALUDE_gnomes_distribution_l482_48290


namespace NUMINAMATH_CALUDE_redistribution_amount_l482_48227

def earnings : List ℕ := [18, 22, 26, 32, 47]

theorem redistribution_amount (earnings : List ℕ) (h1 : earnings = [18, 22, 26, 32, 47]) :
  let total := earnings.sum
  let equalShare := total / earnings.length
  let maxEarning := earnings.maximum?
  maxEarning.map (λ max => max - equalShare) = some 18 := by
  sorry

end NUMINAMATH_CALUDE_redistribution_amount_l482_48227


namespace NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_l482_48245

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  s : Point
  t : Point
  u : Point
  v : Point

/-- Represents a triangle -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Calculate the area of a triangle -/
def triangleArea (t : Triangle) : ℝ := sorry

/-- Calculate the area of a square -/
def squareArea (s : Square) : ℝ := sorry

/-- Check if a point is the midpoint of two other points -/
def isMidpoint (m : Point) (a : Point) (b : Point) : Prop := sorry

/-- Check if two line segments are parallel -/
def isParallel (a : Point) (b : Point) (c : Point) (d : Point) : Prop := sorry

/-- Check if two line segments are equal in length -/
def segmentEqual (a : Point) (b : Point) (c : Point) (d : Point) : Prop := sorry

theorem shaded_to_unshaded_ratio 
  (s : Square) 
  (q p r o : Point) 
  (t1 t2 t3 : Triangle) :
  isMidpoint q s.s s.t →
  isMidpoint p s.u s.v →
  segmentEqual p r q r →
  isParallel s.v q p r →
  t1 = Triangle.mk q o r →
  t2 = Triangle.mk p o r →
  t3 = Triangle.mk q p s.v →
  (triangleArea t1 + triangleArea t2 + triangleArea t3) / 
  (squareArea s - (triangleArea t1 + triangleArea t2 + triangleArea t3)) = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_l482_48245


namespace NUMINAMATH_CALUDE_regression_change_l482_48283

/-- Represents a linear regression equation of the form ŷ = a + bx -/
structure LinearRegression where
  a : ℝ  -- intercept
  b : ℝ  -- slope

/-- Calculates the change in y when x increases by one unit -/
def change_in_y (regression : LinearRegression) : ℝ := -regression.b

/-- Theorem: For the given regression equation ŷ = 2 - 1.5x, 
    when x increases by one unit, y decreases by 1.5 units -/
theorem regression_change (regression : LinearRegression) 
  (h1 : regression.a = 2) 
  (h2 : regression.b = -1.5) : 
  change_in_y regression = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_regression_change_l482_48283


namespace NUMINAMATH_CALUDE_arc_length_from_sector_area_l482_48213

/-- Given a circle with radius 5 cm and a sector with area 10 cm², 
    prove that the length of the arc forming the sector is 4 cm. -/
theorem arc_length_from_sector_area (r : ℝ) (area : ℝ) (arc_length : ℝ) : 
  r = 5 → 
  area = 10 → 
  area = (arc_length / (2 * r)) * r^2 → 
  arc_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_from_sector_area_l482_48213


namespace NUMINAMATH_CALUDE_subset_sum_property_l482_48259

theorem subset_sum_property (n : ℕ) (A B C : Finset ℕ) :
  (∀ i ∈ A ∪ B ∪ C, i ≤ 3*n) →
  A.card = n →
  B.card = n →
  C.card = n →
  (A ∩ B ∩ C).card = 0 →
  (A ∪ B ∪ C).card = 3*n →
  ∃ (a b c : ℕ), a ∈ A ∧ b ∈ B ∧ c ∈ C ∧ a + b = c :=
by sorry

end NUMINAMATH_CALUDE_subset_sum_property_l482_48259


namespace NUMINAMATH_CALUDE_arccos_cos_three_l482_48247

theorem arccos_cos_three : Real.arccos (Real.cos 3) = 3 - 2 * Real.pi := by sorry

end NUMINAMATH_CALUDE_arccos_cos_three_l482_48247


namespace NUMINAMATH_CALUDE_unique_solution_for_all_y_l482_48276

theorem unique_solution_for_all_y :
  ∃! x : ℚ, ∀ y : ℚ, 10 * x * y - 15 * y + 3 * x - 9 / 2 = 0 :=
by
  -- The unique solution is x = 3/2
  use 3 / 2
  sorry

end NUMINAMATH_CALUDE_unique_solution_for_all_y_l482_48276


namespace NUMINAMATH_CALUDE_vector_simplification_1_vector_simplification_2_vector_simplification_3_l482_48266

variable {V : Type*} [AddCommGroup V]

-- Define vector between two points
def vec (A B : V) : V := B - A

-- Theorem 1
theorem vector_simplification_1 (A B C D : V) :
  vec A B + vec B C - vec A D = vec D C := by sorry

-- Theorem 2
theorem vector_simplification_2 (A B C D : V) :
  (vec A B - vec C D) - (vec A C - vec B D) = 0 := by sorry

-- Theorem 3
theorem vector_simplification_3 (A B C D O : V) :
  (vec A C + vec B O + vec O A) - (vec D C - vec D O - vec O B) = 0 := by sorry

end NUMINAMATH_CALUDE_vector_simplification_1_vector_simplification_2_vector_simplification_3_l482_48266


namespace NUMINAMATH_CALUDE_optimal_strategy_highest_hunter_l482_48219

/-- Represents a hunter in the treasure division game -/
structure Hunter :=
  (id : Nat)
  (coins : Nat)

/-- Represents the state of the game -/
structure GameState :=
  (n : Nat)  -- Total number of hunters
  (m : Nat)  -- Total number of coins
  (hunters : List Hunter)

/-- Checks if a proposal is accepted by majority vote -/
def isProposalAccepted (state : GameState) (proposal : List Hunter) : Prop :=
  2 * (proposal.filter (fun h => h.coins > 0)).length > state.hunters.length

/-- Generates the optimal proposal for a given hunter -/
def optimalProposal (state : GameState) (hunterId : Nat) : List Hunter :=
  sorry

/-- Theorem: The optimal strategy for the highest-numbered hunter is to propose
    m - (n ÷ 2) coins for themselves and 1 coin each for the even-numbered
    hunters below them, until they secure a majority vote -/
theorem optimal_strategy_highest_hunter (state : GameState) :
  let proposal := optimalProposal state state.n
  isProposalAccepted state proposal ∧
  (proposal.head?.map Hunter.coins).getD 0 = state.m - (state.n / 2) ∧
  (proposal.tail.filter (fun h => h.coins > 0)).all (fun h => h.coins = 1 ∧ h.id % 2 = 0) :=
  sorry


end NUMINAMATH_CALUDE_optimal_strategy_highest_hunter_l482_48219


namespace NUMINAMATH_CALUDE_ratio_subtraction_l482_48234

theorem ratio_subtraction (a b : ℚ) (h : a / b = 4 / 7) :
  (a - b) / b = -3 / 7 := by sorry

end NUMINAMATH_CALUDE_ratio_subtraction_l482_48234


namespace NUMINAMATH_CALUDE_percentage_not_sold_approx_l482_48256

def initial_stock : ℕ := 1400
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135

theorem percentage_not_sold_approx (ε : ℝ) (ε_pos : ε > 0) :
  ∃ (p : ℝ), abs (p - ((initial_stock - (monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales)) / initial_stock * 100)) < ε ∧
             abs (p - 71.29) < ε := by
  sorry

end NUMINAMATH_CALUDE_percentage_not_sold_approx_l482_48256


namespace NUMINAMATH_CALUDE_circle_symmetry_line_l482_48297

/-- A circle in the xy-plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A line in the xy-plane of the form y = mx + b -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Predicate to check if a circle is symmetric with respect to a line -/
def is_symmetric (c : Circle) (l : Line) : Prop :=
  c.center.1 + c.center.2 = l.slope * c.center.1 + l.intercept + c.center.2

theorem circle_symmetry_line (b : ℝ) : 
  let c : Circle := { center := (1, 2), radius := 1 }
  let l : Line := { slope := 1, intercept := b }
  is_symmetric c l → b = 1 := by
  sorry

#check circle_symmetry_line

end NUMINAMATH_CALUDE_circle_symmetry_line_l482_48297


namespace NUMINAMATH_CALUDE_nabla_calculation_l482_48263

-- Define the nabla operation
def nabla (a b : ℕ) : ℕ := 3 + b^a

-- State the theorem
theorem nabla_calculation : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end NUMINAMATH_CALUDE_nabla_calculation_l482_48263


namespace NUMINAMATH_CALUDE_water_used_l482_48218

theorem water_used (total_liquid oil : ℝ) (h1 : total_liquid = 1.33) (h2 : oil = 0.17) :
  total_liquid - oil = 1.16 := by
  sorry

end NUMINAMATH_CALUDE_water_used_l482_48218


namespace NUMINAMATH_CALUDE_waiter_customers_l482_48208

theorem waiter_customers : ∃ x : ℕ, x = 33 ∧ (x - 31 + 26 = 28) := by
  sorry

end NUMINAMATH_CALUDE_waiter_customers_l482_48208


namespace NUMINAMATH_CALUDE_angelina_speed_l482_48248

/-- Proves that Angelina's speed from the grocery to the gym is 3 meters per second --/
theorem angelina_speed (home_to_grocery : ℝ) (grocery_to_gym : ℝ) (v : ℝ) :
  home_to_grocery = 180 →
  grocery_to_gym = 240 →
  (home_to_grocery / v) - (grocery_to_gym / (2 * v)) = 40 →
  2 * v = 3 := by
  sorry

end NUMINAMATH_CALUDE_angelina_speed_l482_48248


namespace NUMINAMATH_CALUDE_half_product_uniqueness_l482_48223

theorem half_product_uniqueness (x : ℕ) :
  (∃ n : ℕ, x = n * (n + 1) / 2) →
  ∀ k m : ℕ, x = k * (k + 1) / 2 ∧ x = m * (m + 1) / 2 → k = m := by
  sorry

end NUMINAMATH_CALUDE_half_product_uniqueness_l482_48223


namespace NUMINAMATH_CALUDE_triangle_properties_l482_48281

theorem triangle_properties (A B C : Real) (a b c : Real) (D : Real × Real) :
  -- Given conditions
  (c / Real.cos C = (a + b) / (Real.cos A + Real.cos B)) →
  (Real.cos A + Real.cos B ≠ 0) →
  (D.1 = (B + C) / 2) →
  (D.2 = 0) →
  (Real.sqrt ((A - D.1)^2 + D.2^2) = 2) →
  (Real.sqrt ((A - C)^2 + 0^2) = Real.sqrt 7) →
  -- Conclusions
  (C = Real.pi / 3) ∧
  (Real.sqrt ((B - A)^2 + 0^2) = 2 * Real.sqrt 7) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l482_48281


namespace NUMINAMATH_CALUDE_magnitude_of_z_l482_48216

theorem magnitude_of_z (i : ℂ) (h : i^2 = -1) : Complex.abs ((1 + i) / i) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_z_l482_48216


namespace NUMINAMATH_CALUDE_expression_defined_iff_l482_48206

theorem expression_defined_iff (x : ℝ) : 
  (∃ y : ℝ, y = Real.sqrt (x - 2) / Real.sqrt (x - 1)) ↔ x ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_defined_iff_l482_48206


namespace NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l482_48237

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from one base to another -/
def baseConvert (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

/-- Number of digits in a number in a given base -/
def numDigits (n : ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_dual_base_palindrome :
  let n := 10001 -- In base 3
  ∀ m : ℕ,
    (numDigits n 3 = 5) →
    (isPalindrome n 3) →
    (∃ b : ℕ, b > 3 ∧ isPalindrome (baseConvert n 3 b) b ∧ numDigits (baseConvert n 3 b) b = 4) →
    (numDigits m 3 = 5) →
    (isPalindrome m 3) →
    (∃ b : ℕ, b > 3 ∧ isPalindrome (baseConvert m 3 b) b ∧ numDigits (baseConvert m 3 b) b = 4) →
    m ≥ n :=
by sorry

end NUMINAMATH_CALUDE_smallest_dual_base_palindrome_l482_48237


namespace NUMINAMATH_CALUDE_complex_fraction_problem_l482_48239

theorem complex_fraction_problem (x y : ℂ) (k : ℝ) 
  (h : (x + k * y) / (x - k * y) + (x - k * y) / (x + k * y) = 1) :
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 41 / 20 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_problem_l482_48239


namespace NUMINAMATH_CALUDE_factorization_of_4x_cubed_minus_x_l482_48226

theorem factorization_of_4x_cubed_minus_x (x : ℝ) : 4 * x^3 - x = x * (2*x + 1) * (2*x - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_4x_cubed_minus_x_l482_48226


namespace NUMINAMATH_CALUDE_intersection_when_a_is_two_proper_subset_condition_l482_48230

def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < 2 * a + 3}
def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}

theorem intersection_when_a_is_two :
  A 2 ∩ B = {x | 1 < x ∧ x ≤ 4} := by sorry

theorem proper_subset_condition (a : ℝ) :
  A a ⊂ B ↔ a ≤ -4 ∨ (-1 ≤ a ∧ a ≤ 1/2) := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_two_proper_subset_condition_l482_48230


namespace NUMINAMATH_CALUDE_fraction_simplification_l482_48251

theorem fraction_simplification (x y : ℝ) : 
  (2*x + y)/4 + (5*y - 4*x)/6 - y/12 = (-x + 6*y)/6 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l482_48251


namespace NUMINAMATH_CALUDE_tax_percentage_proof_l482_48250

/-- 
Given:
- total_income: The total annual income
- after_tax_income: The income left after paying taxes

Prove that the percentage of income paid in taxes is 18%
-/
theorem tax_percentage_proof (total_income after_tax_income : ℝ) 
  (h1 : total_income = 60000)
  (h2 : after_tax_income = 49200) :
  (total_income - after_tax_income) / total_income * 100 = 18 := by
  sorry


end NUMINAMATH_CALUDE_tax_percentage_proof_l482_48250


namespace NUMINAMATH_CALUDE_triangle_and_line_properties_l482_48261

-- Define the points
def A : ℝ × ℝ := (4, 1)
def B : ℝ × ℝ := (-1, -6)
def C : ℝ × ℝ := (-3, 2)

-- Define the triangular region D
def D : Set (ℝ × ℝ) := {(x, y) | 7*x - 5*y - 23 ≤ 0 ∧ x + 7*y - 11 ≤ 0 ∧ 4*x + y + 10 ≥ 0}

-- Define the line 4x - 3y - a = 0
def line (a : ℝ) : Set (ℝ × ℝ) := {(x, y) | 4*x - 3*y - a = 0}

-- Theorem statement
theorem triangle_and_line_properties :
  -- B and C are on opposite sides of the line 4x - 3y - a = 0
  ∀ a : ℝ, (4 * B.1 - 3 * B.2 - a) * (4 * C.1 - 3 * C.2 - a) < 0 →
  -- The system of inequalities correctly represents region D
  (∀ p : ℝ × ℝ, p ∈ D ↔ 7*p.1 - 5*p.2 - 23 ≤ 0 ∧ p.1 + 7*p.2 - 11 ≤ 0 ∧ 4*p.1 + p.2 + 10 ≥ 0) ∧
  -- The range of values for a is (-18, 14)
  (∀ a : ℝ, (4 * B.1 - 3 * B.2 - a) * (4 * C.1 - 3 * C.2 - a) < 0 ↔ -18 < a ∧ a < 14) :=
sorry

end NUMINAMATH_CALUDE_triangle_and_line_properties_l482_48261


namespace NUMINAMATH_CALUDE_girls_in_class_l482_48211

theorem girls_in_class (total : ℕ) (boy_ratio girl_ratio : ℕ) (h1 : total = 260) (h2 : boy_ratio = 5) (h3 : girl_ratio = 8) :
  (girl_ratio * total) / (boy_ratio + girl_ratio) = 160 := by
sorry

end NUMINAMATH_CALUDE_girls_in_class_l482_48211


namespace NUMINAMATH_CALUDE_hyperbola_vertex_distance_l482_48229

/-- The distance between the vertices of a hyperbola with equation y²/48 - x²/16 = 1 is 8√3 -/
theorem hyperbola_vertex_distance :
  let hyperbola := {(x, y) : ℝ × ℝ | y^2 / 48 - x^2 / 16 = 1}
  ∃ v₁ v₂ : ℝ × ℝ, v₁ ∈ hyperbola ∧ v₂ ∈ hyperbola ∧ 
    ∀ p ∈ hyperbola, (p.1 = v₁.1 ∨ p.1 = v₂.1) → p.2 = 0 ∧
    ‖v₁ - v₂‖ = 8 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_vertex_distance_l482_48229


namespace NUMINAMATH_CALUDE_height_weight_regression_properties_l482_48253

/-- Represents the linear regression model for female students' height and weight -/
structure HeightWeightRegression where
  slope : ℝ
  intercept : ℝ

/-- Defines the sample data points -/
structure SampleData where
  x : List ℝ
  y : List ℝ

/-- Theorem stating properties of the linear regression model -/
theorem height_weight_regression_properties
  (model : HeightWeightRegression)
  (data : SampleData)
  (h_model : model.slope = 0.85 ∧ model.intercept = -85.71)
  (h_data : data.x.length = data.y.length ∧ data.x.length > 0) :
  let x_mean := data.x.sum / data.x.length
  let y_mean := data.y.sum / data.y.length
  -- 1. Positive correlation
  model.slope > 0 ∧
  -- 2. Regression line passes through (x̄, ȳ)
  y_mean = model.slope * x_mean + model.intercept ∧
  -- 3. Unit increase in x corresponds to 0.85 increase in y
  ∀ x₁ x₂, model.slope * (x₂ - x₁) = 0.85 * (x₂ - x₁) ∧
  -- 4. Equation provides an estimate, not exact value
  ∀ x y, y = model.slope * x + model.intercept → 
    ∃ ε > 0, ∀ actual_y, |actual_y - y| < ε := by
  sorry

end NUMINAMATH_CALUDE_height_weight_regression_properties_l482_48253


namespace NUMINAMATH_CALUDE_rectangle_length_l482_48285

theorem rectangle_length (square_side : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  square_side = 9 →
  rect_width = 3 →
  square_side * square_side = rect_width * rect_length →
  rect_length = 27 := by
sorry

end NUMINAMATH_CALUDE_rectangle_length_l482_48285


namespace NUMINAMATH_CALUDE_remainder_seven_n_mod_three_l482_48240

theorem remainder_seven_n_mod_three (n : ℤ) (h : n % 3 = 2) : (7 * n) % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_seven_n_mod_three_l482_48240


namespace NUMINAMATH_CALUDE_modulus_of_complex_fraction_l482_48294

theorem modulus_of_complex_fraction : 
  let z : ℂ := (4 - 3*I) / (2 - I)
  Complex.abs z = Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_fraction_l482_48294


namespace NUMINAMATH_CALUDE_second_hand_movement_l482_48258

/-- Represents the movement of clock hands -/
def ClockMovement : Type :=
  { minutes : ℕ // minutes > 0 }

/-- Converts minutes to seconds -/
def minutesToSeconds (m : ClockMovement) : ℕ :=
  m.val * 60

/-- Calculates the number of circles the second hand moves -/
def secondHandCircles (m : ClockMovement) : ℕ :=
  minutesToSeconds m / 60

/-- The theorem to be proved -/
theorem second_hand_movement (m : ClockMovement) (h : m.val = 2) :
  secondHandCircles m = 2 := by
  sorry

end NUMINAMATH_CALUDE_second_hand_movement_l482_48258


namespace NUMINAMATH_CALUDE_square_cube_sum_condition_l482_48282

theorem square_cube_sum_condition (n : ℕ) : 
  (∃ a b : ℤ, n^2 = a + b ∧ n^3 = a^2 + b^2) ↔ n = 0 ∨ n = 1 ∨ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_cube_sum_condition_l482_48282


namespace NUMINAMATH_CALUDE_absent_students_count_l482_48269

/-- The number of classes at Webster Middle School -/
def num_classes : ℕ := 18

/-- The number of students in each class -/
def students_per_class : ℕ := 28

/-- The number of students present on Monday -/
def students_present : ℕ := 496

/-- The number of absent students -/
def absent_students : ℕ := num_classes * students_per_class - students_present

theorem absent_students_count : absent_students = 8 := by
  sorry

end NUMINAMATH_CALUDE_absent_students_count_l482_48269


namespace NUMINAMATH_CALUDE_original_number_proof_l482_48279

theorem original_number_proof (n : ℕ) (k : ℕ) : 
  (∃ m : ℕ, n + k = 5 * m) ∧ 
  (n + k = 2500) ∧ 
  (∀ j : ℕ, j < k → ¬∃ m : ℕ, n + j = 5 * m) →
  n = 2500 :=
by sorry

end NUMINAMATH_CALUDE_original_number_proof_l482_48279


namespace NUMINAMATH_CALUDE_sum_greatest_odd_divisors_formula_l482_48246

/-- The greatest odd divisor of a positive integer -/
def greatest_odd_divisor (k : ℕ+) : ℕ+ :=
  sorry

/-- The sum of greatest odd divisors from 1 to 2^n -/
def sum_greatest_odd_divisors (n : ℕ+) : ℕ+ :=
  sorry

theorem sum_greatest_odd_divisors_formula (n : ℕ+) :
  (sum_greatest_odd_divisors n : ℚ) = (4^(n : ℕ) + 5) / 3 :=
sorry

end NUMINAMATH_CALUDE_sum_greatest_odd_divisors_formula_l482_48246


namespace NUMINAMATH_CALUDE_clover_total_distance_l482_48210

/-- Clover's daily morning walk distance in miles -/
def morning_walk : ℝ := 1.5

/-- Clover's daily evening walk distance in miles -/
def evening_walk : ℝ := 1.5

/-- Number of days Clover walks -/
def days : ℕ := 30

/-- Theorem stating the total distance Clover walks in 30 days -/
theorem clover_total_distance : 
  (morning_walk + evening_walk) * days = 90 := by
  sorry

end NUMINAMATH_CALUDE_clover_total_distance_l482_48210


namespace NUMINAMATH_CALUDE_initial_journey_speed_l482_48233

/-- Proves that the speed of the initial journey is 63 mph given the conditions -/
theorem initial_journey_speed (d : ℝ) (v : ℝ) (h1 : v > 0) : 
  (2 * d) / (d / v + 2 * (d / v)) = 42 → v = 63 := by
  sorry

end NUMINAMATH_CALUDE_initial_journey_speed_l482_48233


namespace NUMINAMATH_CALUDE_value_of_c_l482_48252

theorem value_of_c (a b c : ℝ) 
  (h1 : 12 = 0.06 * a) 
  (h2 : 6 = 0.12 * b) 
  (h3 : c = b / a) : 
  c = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_value_of_c_l482_48252


namespace NUMINAMATH_CALUDE_fixed_point_quadratic_l482_48225

theorem fixed_point_quadratic (k : ℝ) : 
  228 = 9 * (5 : ℝ)^2 + k * 5 - 5 * k + 3 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_quadratic_l482_48225


namespace NUMINAMATH_CALUDE_alyssas_allowance_l482_48265

theorem alyssas_allowance (allowance : ℝ) : 
  (allowance / 2 + 8 = 12) → allowance = 8 := by
  sorry

end NUMINAMATH_CALUDE_alyssas_allowance_l482_48265


namespace NUMINAMATH_CALUDE_petya_friend_count_l482_48295

/-- Represents the number of friends a student has -/
def FriendCount := Fin 29

/-- Represents a student in the class -/
def Student := Fin 29

/-- The function that maps each student to their friend count -/
def friendCount : Student → FriendCount := sorry

/-- Petya is represented by the last student in the enumeration -/
def petya : Student := ⟨28, sorry⟩

theorem petya_friend_count :
  (∀ (s1 s2 : Student), s1 ≠ s2 → friendCount s1 ≠ friendCount s2) →
  friendCount petya = ⟨14, sorry⟩ := by
  sorry

end NUMINAMATH_CALUDE_petya_friend_count_l482_48295


namespace NUMINAMATH_CALUDE_max_donated_cookies_l482_48249

def distribute_cookies (total : Nat) (employees : Nat) : Nat :=
  total - (employees * (total / employees))

theorem max_donated_cookies :
  distribute_cookies 120 7 = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_donated_cookies_l482_48249


namespace NUMINAMATH_CALUDE_terminating_decimal_of_19_80_l482_48291

theorem terminating_decimal_of_19_80 : ∃ (n : ℕ), (19 : ℚ) / 80 = (2375 : ℚ) / 10^n :=
sorry

end NUMINAMATH_CALUDE_terminating_decimal_of_19_80_l482_48291


namespace NUMINAMATH_CALUDE_spade_heart_eval_l482_48296

/-- Operation ♠ for real numbers -/
def spade (x y : ℝ) : ℝ := (x + y) * (x - y)

/-- Operation ♥ for real numbers -/
def heart (x y : ℝ) : ℝ := x^2 - y^2

/-- Theorem stating that 5 ♠ (3 ♥ 2) = 0 -/
theorem spade_heart_eval : spade 5 (heart 3 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_spade_heart_eval_l482_48296


namespace NUMINAMATH_CALUDE_preimage_of_neg_one_two_l482_48293

/-- A mapping f from ℝ² to ℝ² defined as f(x, y) = (2x, x - y) -/
def f : ℝ × ℝ → ℝ × ℝ := fun (x, y) ↦ (2 * x, x - y)

/-- Theorem stating that f(-1/2, -5/2) = (-1, 2) -/
theorem preimage_of_neg_one_two :
  f (-1/2, -5/2) = (-1, 2) := by
  sorry

end NUMINAMATH_CALUDE_preimage_of_neg_one_two_l482_48293


namespace NUMINAMATH_CALUDE_angle_east_southwest_is_135_l482_48200

/-- Represents a circle with 8 equally spaced rays --/
structure EightRayCircle where
  /-- The measure of the angle between adjacent rays in degrees --/
  angle_between_rays : ℝ
  /-- The angle between adjacent rays is 45° --/
  angle_is_45 : angle_between_rays = 45

/-- The measure of the smaller angle between East and Southwest rays in degrees --/
def angle_east_southwest (circle : EightRayCircle) : ℝ :=
  3 * circle.angle_between_rays

theorem angle_east_southwest_is_135 (circle : EightRayCircle) :
  angle_east_southwest circle = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_east_southwest_is_135_l482_48200


namespace NUMINAMATH_CALUDE_vincent_earnings_l482_48222

/-- Represents Vincent's bookstore earnings over a period of days -/
def bookstore_earnings (fantasy_price : ℕ) (fantasy_sold : ℕ) (literature_sold : ℕ) (days : ℕ) : ℕ :=
  let literature_price := fantasy_price / 2
  let daily_earnings := fantasy_price * fantasy_sold + literature_price * literature_sold
  daily_earnings * days

/-- Theorem stating that Vincent's earnings after 5 days will be $180 -/
theorem vincent_earnings : bookstore_earnings 4 5 8 5 = 180 := by
  sorry

end NUMINAMATH_CALUDE_vincent_earnings_l482_48222


namespace NUMINAMATH_CALUDE_ten_point_six_trillion_scientific_notation_l482_48268

-- Define a trillion
def trillion : ℝ := 10^12

-- State the theorem
theorem ten_point_six_trillion_scientific_notation :
  (10.6 * trillion) = 1.06 * 10^13 := by sorry

end NUMINAMATH_CALUDE_ten_point_six_trillion_scientific_notation_l482_48268


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l482_48235

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x - 1) + 1
  f 1 = 2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l482_48235


namespace NUMINAMATH_CALUDE_artichokey_invested_seven_l482_48220

/-- Represents the investment and payout of earthworms -/
structure EarthwormInvestment where
  total_earthworms : ℕ
  okeydokey_apples : ℕ
  okeydokey_earthworms : ℕ

/-- Calculates the number of apples Artichokey invested -/
def artichokey_investment (e : EarthwormInvestment) : ℕ :=
  sorry

/-- Theorem stating that Artichokey invested 7 apples -/
theorem artichokey_invested_seven (e : EarthwormInvestment)
  (h1 : e.total_earthworms = 60)
  (h2 : e.okeydokey_apples = 5)
  (h3 : e.okeydokey_earthworms = 25)
  (h4 : e.okeydokey_earthworms * e.total_earthworms = e.okeydokey_apples * (e.total_earthworms + e.okeydokey_earthworms)) :
  artichokey_investment e = 7 :=
sorry

end NUMINAMATH_CALUDE_artichokey_invested_seven_l482_48220


namespace NUMINAMATH_CALUDE_jellybean_problem_l482_48214

theorem jellybean_problem (initial_count : ℕ) : 
  (initial_count : ℝ) * (3/4)^3 = 27 → initial_count = 64 :=
by
  sorry

end NUMINAMATH_CALUDE_jellybean_problem_l482_48214


namespace NUMINAMATH_CALUDE_expression_value_l482_48238

theorem expression_value (a : ℝ) (h1 : 1 ≤ a) (h2 : a < 2) :
  Real.sqrt (a + 2 * Real.sqrt (a - 1)) + Real.sqrt (a - 2 * Real.sqrt (a - 1)) = 2 :=
by sorry

end NUMINAMATH_CALUDE_expression_value_l482_48238


namespace NUMINAMATH_CALUDE_shaded_rectangle_perimeter_l482_48241

theorem shaded_rectangle_perimeter
  (total_perimeter : ℝ)
  (square_area : ℝ)
  (h_total_perimeter : total_perimeter = 30)
  (h_square_area : square_area = 9) :
  let square_side := Real.sqrt square_area
  let remaining_sum := (total_perimeter / 2) - 2 * square_side
  2 * remaining_sum = 18 :=
by sorry

end NUMINAMATH_CALUDE_shaded_rectangle_perimeter_l482_48241


namespace NUMINAMATH_CALUDE_bag_equals_two_balls_l482_48271

/-- Represents the weight of an object -/
structure Weight : Type :=
  (value : ℝ)

/-- Represents a balanced scale -/
structure BalancedScale : Type :=
  (left_bags : ℕ)
  (left_balls : ℕ)
  (right_bags : ℕ)
  (right_balls : ℕ)
  (bag_weight : Weight)
  (ball_weight : Weight)

/-- Predicate to check if a scale is balanced -/
def is_balanced (s : BalancedScale) : Prop :=
  s.left_bags * s.bag_weight.value + s.left_balls * s.ball_weight.value =
  s.right_bags * s.bag_weight.value + s.right_balls * s.ball_weight.value

theorem bag_equals_two_balls (s : BalancedScale) :
  s.left_bags = 5 ∧ s.left_balls = 4 ∧ s.right_bags = 2 ∧ s.right_balls = 10 ∧
  is_balanced s →
  s.bag_weight.value = 2 * s.ball_weight.value :=
sorry

end NUMINAMATH_CALUDE_bag_equals_two_balls_l482_48271


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l482_48207

theorem concentric_circles_radii_difference
  (r R : ℝ)
  (h_positive : r > 0)
  (h_ratio : R^2 / r^2 = 4) :
  R - r = r :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l482_48207


namespace NUMINAMATH_CALUDE_quadratic_minimum_l482_48277

theorem quadratic_minimum (x : ℝ) :
  ∃ (min : ℝ), ∀ y : ℝ, y = 4 * x^2 + 8 * x + 16 → y ≥ min ∧ ∃ x₀ : ℝ, 4 * x₀^2 + 8 * x₀ + 16 = min :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l482_48277


namespace NUMINAMATH_CALUDE_range_of_a_for_inequality_l482_48260

theorem range_of_a_for_inequality : 
  {a : ℝ | ∀ x : ℝ, |x + 3| + |x - 1| ≥ a^2 - 3*a} = {a : ℝ | -1 ≤ a ∧ a ≤ 4} := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_for_inequality_l482_48260


namespace NUMINAMATH_CALUDE_seventieth_even_positive_integer_seventieth_even_positive_integer_is_140_l482_48274

theorem seventieth_even_positive_integer : ℕ → ℕ := 
  fun n => 2 * n

#check seventieth_even_positive_integer 70 = 140

theorem seventieth_even_positive_integer_is_140 : 
  seventieth_even_positive_integer 70 = 140 := by
  sorry

end NUMINAMATH_CALUDE_seventieth_even_positive_integer_seventieth_even_positive_integer_is_140_l482_48274


namespace NUMINAMATH_CALUDE_crayon_difference_l482_48202

theorem crayon_difference (willy_crayons lucy_crayons : ℕ) 
  (hw : willy_crayons = 5092) 
  (hl : lucy_crayons = 3971) : 
  willy_crayons - lucy_crayons = 1121 := by
  sorry

end NUMINAMATH_CALUDE_crayon_difference_l482_48202


namespace NUMINAMATH_CALUDE_tangent_point_abscissa_l482_48292

noncomputable section

-- Define the function f(x) = x^2 + x - ln x
def f (x : ℝ) : ℝ := x^2 + x - Real.log x

-- Define the derivative of f(x)
def f_deriv (x : ℝ) : ℝ := 2*x + 1 - 1/x

-- Theorem statement
theorem tangent_point_abscissa (t : ℝ) (h : t > 0) :
  (f t / t = f_deriv t) → t = 1 :=
sorry


end NUMINAMATH_CALUDE_tangent_point_abscissa_l482_48292


namespace NUMINAMATH_CALUDE_distance_walked_l482_48267

theorem distance_walked (x t d : ℝ) 
  (h1 : d = x * t) 
  (h2 : d = (x + 1/2) * (4/5 * t))
  (h3 : d = (x - 1/2) * (t + 5/2)) :
  d = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_walked_l482_48267


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l482_48288

def M : Set ℕ := {1, 2, 3, 4}
def N : Set ℕ := {2, 4, 6, 8}

theorem intersection_of_M_and_N :
  M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l482_48288


namespace NUMINAMATH_CALUDE_largest_of_five_consecutive_integers_l482_48231

theorem largest_of_five_consecutive_integers (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧  -- all positive
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧  -- consecutive
  a * b * c * d * e = 15120 →  -- product is 15120
  e = 10 :=  -- largest is 10
by sorry

end NUMINAMATH_CALUDE_largest_of_five_consecutive_integers_l482_48231


namespace NUMINAMATH_CALUDE_rent_distribution_l482_48242

/-- Represents an individual renting the pasture -/
structure Renter where
  name : String
  oxen : ℕ
  months : ℕ

/-- Calculates the share of rent for a renter -/
def calculateShare (r : Renter) (totalRent : ℚ) (totalOxMonths : ℕ) : ℚ :=
  (r.oxen * r.months : ℚ) * totalRent / totalOxMonths

/-- The main theorem stating the properties of rent distribution -/
theorem rent_distribution
  (renters : List Renter)
  (totalRent : ℚ)
  (h_positive_rent : totalRent > 0)
  (h_renters : renters = [
    ⟨"A", 10, 7⟩,
    ⟨"B", 12, 5⟩,
    ⟨"C", 15, 3⟩,
    ⟨"D", 8, 6⟩,
    ⟨"E", 20, 2⟩
  ])
  (h_total_rent : totalRent = 385) :
  let totalOxMonths := (renters.map (fun r => r.oxen * r.months)).sum
  let shares := renters.map (fun r => calculateShare r totalRent totalOxMonths)
  (∀ (r : Renter), r ∈ renters → 
    calculateShare r totalRent totalOxMonths = 
    (r.oxen * r.months : ℚ) * totalRent / totalOxMonths) ∧
  shares.sum = totalRent :=
sorry

end NUMINAMATH_CALUDE_rent_distribution_l482_48242
