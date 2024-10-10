import Mathlib

namespace lateral_to_base_area_ratio_l997_99706

/-- A cone with its lateral surface unfolded into a sector with a 90° central angle -/
structure UnfoldedCone where
  r : ℝ  -- radius of the base circle
  R : ℝ  -- radius of the unfolded sector (lateral surface)
  h : R = 4 * r  -- condition from the 90° central angle

/-- The ratio of lateral surface area to base area for an UnfoldedCone is 4:1 -/
theorem lateral_to_base_area_ratio (cone : UnfoldedCone) :
  (π * cone.r * cone.R) / (π * cone.r^2) = 4 := by
  sorry

#check lateral_to_base_area_ratio

end lateral_to_base_area_ratio_l997_99706


namespace binomial_30_3_l997_99711

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l997_99711


namespace ellipse_equation_l997_99731

/-- Given an ellipse with equation x²/a² + y²/2 = 1 and one focus at (2,0),
    prove that its specific equation is x²/6 + y²/2 = 1 -/
theorem ellipse_equation (a : ℝ) :
  (∃ (x y : ℝ), x^2/a^2 + y^2/2 = 1) ∧ 
  (∃ (c : ℝ), c = 2 ∧ c^2 = a^2 - 2) →
  a^2 = 6 := by
  sorry

end ellipse_equation_l997_99731


namespace symmetric_point_coordinates_l997_99772

/-- Given a point P in polar coordinates, find its symmetric point with respect to the pole -/
theorem symmetric_point_coordinates (r : ℝ) (θ : ℝ) :
  let P : ℝ × ℝ := (r, θ)
  let symmetric_polar : ℝ × ℝ := (r, θ + π)
  let symmetric_cartesian : ℝ × ℝ := (r * Real.cos (θ + π), r * Real.sin (θ + π))
  P = (2, -5 * π / 3) →
  symmetric_polar = (2, -2 * π / 3) ∧
  symmetric_cartesian = (-1, -Real.sqrt 3) := by
  sorry

#check symmetric_point_coordinates

end symmetric_point_coordinates_l997_99772


namespace apple_distribution_l997_99789

theorem apple_distribution (total_apples : ℕ) (num_babies : ℕ) (min_apples : ℕ) (max_apples : ℕ) :
  total_apples = 30 →
  num_babies = 7 →
  min_apples = 3 →
  max_apples = 6 →
  ∃ (removed : ℕ), 
    (total_apples - removed) % num_babies = 0 ∧
    (total_apples - removed) / num_babies ≥ min_apples ∧
    (total_apples - removed) / num_babies ≤ max_apples ∧
    removed = 2 :=
by sorry

end apple_distribution_l997_99789


namespace seven_digit_palindromes_count_l997_99799

/-- Represents a multiset of digits --/
def DigitMultiset := Multiset Nat

/-- Checks if a number is a palindrome --/
def isPalindrome (n : Nat) : Bool := sorry

/-- Counts the number of 7-digit palindromes that can be formed from a given multiset of digits --/
def countSevenDigitPalindromes (digits : DigitMultiset) : Nat := sorry

/-- The specific multiset of digits given in the problem --/
def givenDigits : DigitMultiset := sorry

theorem seven_digit_palindromes_count :
  countSevenDigitPalindromes givenDigits = 18 := by sorry

end seven_digit_palindromes_count_l997_99799


namespace special_triangle_area_l997_99719

/-- A triangle with specific properties -/
structure SpecialTriangle where
  -- The angle between the two longest sides (in degrees)
  x : ℝ
  -- The perimeter of the triangle (in cm)
  perimeter : ℝ
  -- The inradius of the triangle (in cm)
  inradius : ℝ
  -- Constraint on the angle x
  angle_constraint : 60 < x ∧ x < 120
  -- Given perimeter value
  perimeter_value : perimeter = 48
  -- Given inradius value
  inradius_value : inradius = 2.5

/-- The area of a triangle given its perimeter and inradius -/
def triangleArea (t : SpecialTriangle) : ℝ := t.perimeter * t.inradius

/-- Theorem stating that the area of the special triangle is 120 cm² -/
theorem special_triangle_area (t : SpecialTriangle) : triangleArea t = 120 := by
  sorry

end special_triangle_area_l997_99719


namespace jake_has_nine_peaches_l997_99713

/-- Jake has 7 fewer peaches than Steven and 9 more peaches than Jill. Steven has 16 peaches. -/
def peach_problem (jake steven jill : ℕ) : Prop :=
  jake + 7 = steven ∧ jake = jill + 9 ∧ steven = 16

/-- Prove that Jake has 9 peaches. -/
theorem jake_has_nine_peaches :
  ∀ jake steven jill : ℕ, peach_problem jake steven jill → jake = 9 := by
  sorry

end jake_has_nine_peaches_l997_99713


namespace circle_rectangle_area_relation_l997_99795

theorem circle_rectangle_area_relation (x : ℝ) :
  let circle_radius : ℝ := x - 2
  let rectangle_length : ℝ := x - 3
  let rectangle_width : ℝ := x + 4
  let circle_area : ℝ := π * circle_radius ^ 2
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  rectangle_area = 3 * circle_area →
  (12 * π + 1) / (3 * π - 1) = x + (-(12 * π + 1) / (2 * (1 - 3 * π)) + (12 * π + 1) / (2 * (1 - 3 * π))) :=
by
  sorry

end circle_rectangle_area_relation_l997_99795


namespace symmetric_polynomial_property_l997_99736

theorem symmetric_polynomial_property (p q r : ℝ) :
  let f := λ x : ℝ => p * x^7 + q * x^3 + r * x - 5
  f (-6) = 3 → f 6 = -13 := by
sorry

end symmetric_polynomial_property_l997_99736


namespace new_player_weight_l997_99777

theorem new_player_weight (n : ℕ) (old_avg new_avg new_weight : ℝ) : 
  n = 20 →
  old_avg = 180 →
  new_avg = 181.42857142857142 →
  (n * old_avg + new_weight) / (n + 1) = new_avg →
  new_weight = 210 := by
sorry

end new_player_weight_l997_99777


namespace cubic_inequality_l997_99767

theorem cubic_inequality (x : ℝ) : x^3 - 16*x^2 + 73*x > 84 ↔ x > 13 := by
  sorry

end cubic_inequality_l997_99767


namespace polynomial_simplification_l997_99716

theorem polynomial_simplification (r : ℝ) :
  (2 * r^3 + 4 * r^2 + 5 * r - 3) - (r^3 + 5 * r^2 + 9 * r - 6) = r^3 - r^2 - 4 * r + 3 := by
  sorry

end polynomial_simplification_l997_99716


namespace bob_payment_bob_acorn_payment_l997_99758

theorem bob_payment (alice_acorns : ℕ) (alice_price_per_acorn : ℚ) (alice_bob_price_ratio : ℕ) : ℚ :=
  let alice_total_payment := alice_acorns * alice_price_per_acorn
  alice_total_payment / alice_bob_price_ratio

theorem bob_acorn_payment : bob_payment 3600 15 9 = 6000 := by
  sorry

end bob_payment_bob_acorn_payment_l997_99758


namespace inscribed_sphere_sum_l997_99756

/-- A right cone with a sphere inscribed in it -/
structure InscribedSphere where
  baseRadius : ℝ
  height : ℝ
  sphereRadius : ℝ
  b : ℝ
  d : ℝ
  base_radius_positive : 0 < baseRadius
  height_positive : 0 < height
  sphere_radius_formula : sphereRadius = b * Real.sqrt d - b

/-- The theorem stating that b + d = 20 for the given conditions -/
theorem inscribed_sphere_sum (cone : InscribedSphere)
  (h1 : cone.baseRadius = 15)
  (h2 : cone.height = 30) :
  cone.b + cone.d = 20 := by
  sorry

end inscribed_sphere_sum_l997_99756


namespace functional_equation_solution_l997_99793

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (2 * x * y) + f (f (x + y)) = x * f y + y * f x + f (x + y)) →
  (∀ x : ℝ, f x = 0 ∨ f x = x ∨ f x = 2 - x) :=
by sorry

end functional_equation_solution_l997_99793


namespace ducks_in_marsh_l997_99742

/-- The number of ducks in a marsh, given the total number of birds and the number of geese. -/
def number_of_ducks (total_birds geese : ℕ) : ℕ := total_birds - geese

/-- Theorem stating that there are 37 ducks in the marsh. -/
theorem ducks_in_marsh : number_of_ducks 95 58 = 37 := by
  sorry

end ducks_in_marsh_l997_99742


namespace wire_circle_square_ratio_l997_99729

theorem wire_circle_square_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (π * (a / (2 * π))^2 = (b / 4)^2) → (a / b = 2 / Real.sqrt π) := by
  sorry

end wire_circle_square_ratio_l997_99729


namespace fraction_sum_and_complex_fraction_l997_99750

theorem fraction_sum_and_complex_fraction (a b m : ℝ) 
  (h1 : a ≠ b) (h2 : m ≠ 1) (h3 : m ≠ 2) : 
  (a / (a - b) + b / (b - a) = 1) ∧ 
  ((m^2 - 4) / (4 + 4*m + m^2) / ((m - 2) / (2*m - 2)) * ((m + 2) / (m - 1)) = 2) := by
  sorry

end fraction_sum_and_complex_fraction_l997_99750


namespace height_lateral_edge_ratio_is_correct_l997_99782

/-- Regular quadrilateral pyramid with vertex P and square base ABCD -/
structure RegularQuadPyramid where
  base_side : ℝ
  height : ℝ

/-- Plane intersecting the pyramid -/
structure IntersectingPlane where
  pyramid : RegularQuadPyramid

/-- The ratio of height to lateral edge in a regular quadrilateral pyramid
    where the intersecting plane creates a cross-section with half the area of the base -/
def height_to_lateral_edge_ratio (p : RegularQuadPyramid) (plane : IntersectingPlane) : ℝ :=
  sorry

/-- Theorem stating the ratio of height to lateral edge -/
theorem height_lateral_edge_ratio_is_correct (p : RegularQuadPyramid) (plane : IntersectingPlane) :
  height_to_lateral_edge_ratio p plane = (1 + Real.sqrt 33) / 8 :=
sorry

end height_lateral_edge_ratio_is_correct_l997_99782


namespace base8_perfect_square_b_zero_l997_99761

/-- Represents a number in base 8 of the form a1b4 -/
structure Base8Number where
  a : ℕ
  b : ℕ
  h_a_nonzero : a ≠ 0

/-- Converts a Base8Number to its decimal representation -/
def toDecimal (n : Base8Number) : ℕ :=
  512 * n.a + 64 + 8 * n.b + 4

/-- Predicate to check if a natural number is a perfect square -/
def isPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

theorem base8_perfect_square_b_zero (n : Base8Number) :
  isPerfectSquare (toDecimal n) → n.b = 0 := by
  sorry

end base8_perfect_square_b_zero_l997_99761


namespace sum_of_unique_areas_l997_99704

-- Define a structure for right triangles with integer leg lengths
structure SuperCoolTriangle where
  a : ℕ
  b : ℕ
  h : (a * b) / 2 = 3 * (a + b)

-- Define a function to calculate the area of a triangle
def triangleArea (t : SuperCoolTriangle) : ℕ := (t.a * t.b) / 2

-- Define a function to get all unique areas of super cool triangles
def uniqueAreas : List ℕ := sorry

-- Theorem statement
theorem sum_of_unique_areas : (uniqueAreas.sum) = 471 := by sorry

end sum_of_unique_areas_l997_99704


namespace geometric_sequence_first_term_determination_l997_99705

/-- A geometric sequence is defined by its first term and common ratio -/
structure GeometricSequence where
  first_term : ℝ
  common_ratio : ℝ

/-- The nth term of a geometric sequence -/
def nth_term (seq : GeometricSequence) (n : ℕ) : ℝ :=
  seq.first_term * seq.common_ratio ^ (n - 1)

theorem geometric_sequence_first_term_determination 
  (seq : GeometricSequence) 
  (h5 : nth_term seq 5 = 72)
  (h8 : nth_term seq 8 = 576) : 
  seq.first_term = 4.5 := by
sorry

end geometric_sequence_first_term_determination_l997_99705


namespace twenty_people_handshakes_l997_99770

/-- The number of unique handshakes in a group where each person shakes hands once with every other person -/
def number_of_handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a group of 20 people, where each person shakes hands once with every other person, there are 190 unique handshakes -/
theorem twenty_people_handshakes :
  number_of_handshakes 20 = 190 := by
  sorry

#eval number_of_handshakes 20

end twenty_people_handshakes_l997_99770


namespace vector_coordinates_proof_l997_99786

theorem vector_coordinates_proof (a : ℝ × ℝ) (b : ℝ × ℝ) :
  let x := a.1
  let y := a.2
  b = (1, 2) →
  Real.sqrt (x^2 + y^2) = 3 →
  x * b.1 + y * b.2 = 0 →
  (x = -6 * Real.sqrt 5 / 5 ∧ y = 3 * Real.sqrt 5 / 5) ∨
  (x = 6 * Real.sqrt 5 / 5 ∧ y = -3 * Real.sqrt 5 / 5) :=
by sorry

end vector_coordinates_proof_l997_99786


namespace line_through_M_parallel_to_line1_line_through_N_perpendicular_to_line2_l997_99753

-- Define the points M and N
def M : ℝ × ℝ := (1, -2)
def N : ℝ × ℝ := (2, -3)

-- Define the lines given in the conditions
def line1 (x y : ℝ) : Prop := 2*x - y + 5 = 0
def line2 (x y : ℝ) : Prop := x - 2*y - 3 = 0

-- Define the parallel and perpendicular conditions
def parallel (m₁ m₂ : ℝ) : Prop := m₁ = m₂
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

-- Theorem for the first line
theorem line_through_M_parallel_to_line1 :
  ∃ (a b c : ℝ), 
    (a * M.1 + b * M.2 + c = 0) ∧ 
    (∀ (x y : ℝ), a * x + b * y + c = 0 ↔ 2 * x - y - 4 = 0) ∧
    parallel (a / b) 2 :=
sorry

-- Theorem for the second line
theorem line_through_N_perpendicular_to_line2 :
  ∃ (a b c : ℝ), 
    (a * N.1 + b * N.2 + c = 0) ∧ 
    (∀ (x y : ℝ), a * x + b * y + c = 0 ↔ 2 * x + y - 1 = 0) ∧
    perpendicular (a / b) (1 / 2) :=
sorry

end line_through_M_parallel_to_line1_line_through_N_perpendicular_to_line2_l997_99753


namespace y_change_when_x_increases_y_decreases_by_1_5_l997_99700

-- Define the regression equation
def regression_equation (x : ℝ) : ℝ := 2 - 1.5 * x

-- Theorem stating the change in y when x increases by one unit
theorem y_change_when_x_increases (x : ℝ) :
  regression_equation (x + 1) = regression_equation x - 1.5 := by
  sorry

-- Theorem stating that y decreases by 1.5 units when x increases by one unit
theorem y_decreases_by_1_5 (x : ℝ) :
  regression_equation (x + 1) - regression_equation x = -1.5 := by
  sorry

end y_change_when_x_increases_y_decreases_by_1_5_l997_99700


namespace triangle_area_l997_99766

def a : ℝ × ℝ := (4, -1)
def b : ℝ × ℝ := (2, 3)

theorem triangle_area : 
  let doubled_a : ℝ × ℝ := (2 * a.1, 2 * a.2)
  (1/2) * |doubled_a.1 * b.2 - doubled_a.2 * b.1| = 14 := by sorry

end triangle_area_l997_99766


namespace units_digit_of_product_l997_99708

theorem units_digit_of_product (n : ℕ) : (4^101 * 5^204 * 9^303 * 11^404) % 10 = 0 := by
  sorry

end units_digit_of_product_l997_99708


namespace gcd_459_357_l997_99776

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l997_99776


namespace cookie_count_theorem_l997_99797

/-- Represents a pack of cookies with a specific number of cookies -/
structure CookiePack where
  cookies : ℕ

/-- Represents a person's purchase of cookie packs -/
structure Purchase where
  packA : ℕ
  packB : ℕ
  packC : ℕ
  packD : ℕ

def packA : CookiePack := ⟨15⟩
def packB : CookiePack := ⟨30⟩
def packC : CookiePack := ⟨45⟩
def packD : CookiePack := ⟨60⟩

def paulPurchase : Purchase := ⟨1, 2, 0, 0⟩
def paulaPurchase : Purchase := ⟨1, 0, 1, 0⟩

def totalCookies (p : Purchase) : ℕ :=
  p.packA * packA.cookies + p.packB * packB.cookies + p.packC * packC.cookies + p.packD * packD.cookies

theorem cookie_count_theorem :
  totalCookies paulPurchase + totalCookies paulaPurchase = 135 := by
  sorry


end cookie_count_theorem_l997_99797


namespace yellow_block_weight_l997_99764

theorem yellow_block_weight (green_weight : ℝ) (weight_difference : ℝ) 
  (h1 : green_weight = 0.4)
  (h2 : weight_difference = 0.2) :
  green_weight + weight_difference = 0.6 := by
  sorry

end yellow_block_weight_l997_99764


namespace point_on_line_l997_99759

/-- The value of m for which the point (m + 1, 3) lies on the line x + y + 1 = 0 -/
theorem point_on_line (m : ℝ) : (m + 1) + 3 + 1 = 0 ↔ m = -5 := by sorry

end point_on_line_l997_99759


namespace reassembled_prism_surface_area_l997_99721

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  width : ℝ
  height : ℝ
  length : ℝ

/-- Represents the cuts made to the prism -/
structure PrismCuts where
  first_cut : ℝ
  second_cut : ℝ
  third_cut : ℝ

/-- Calculates the surface area of the reassembled prism -/
def surface_area_reassembled (dim : PrismDimensions) (cuts : PrismCuts) : ℝ :=
  sorry

/-- Theorem stating that the surface area of the reassembled prism is 16 square feet -/
theorem reassembled_prism_surface_area 
  (dim : PrismDimensions) 
  (cuts : PrismCuts) 
  (h1 : dim.width = 1) 
  (h2 : dim.height = 1) 
  (h3 : dim.length = 2) 
  (h4 : cuts.first_cut = 1/4) 
  (h5 : cuts.second_cut = 1/5) 
  (h6 : cuts.third_cut = 1/6) : 
  surface_area_reassembled dim cuts = 16 := by
  sorry

end reassembled_prism_surface_area_l997_99721


namespace prob_not_six_is_five_sevenths_l997_99727

/-- A specially designed six-sided die -/
structure SpecialDie :=
  (sides : Nat)
  (odds_six : Rat)
  (is_valid : sides = 6 ∧ odds_six = 2/5)

/-- The probability of rolling a number other than six -/
def prob_not_six (d : SpecialDie) : Rat :=
  1 - (d.odds_six / (1 + d.odds_six))

theorem prob_not_six_is_five_sevenths (d : SpecialDie) :
  prob_not_six d = 5/7 := by
  sorry

end prob_not_six_is_five_sevenths_l997_99727


namespace integer_root_count_theorem_l997_99757

/-- A polynomial of degree 5 with integer coefficients -/
structure IntPolynomial5 where
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ

/-- The number of integer roots of a polynomial, counting multiplicity -/
def numIntegerRoots (p : IntPolynomial5) : ℕ := sorry

/-- The set of possible values for the number of integer roots -/
def possibleRootCounts : Set ℕ := {0, 1, 2, 3, 5}

/-- Theorem: The number of integer roots of a degree 5 polynomial 
    with integer coefficients is always in the set {0, 1, 2, 3, 5} -/
theorem integer_root_count_theorem (p : IntPolynomial5) : 
  numIntegerRoots p ∈ possibleRootCounts := by sorry

end integer_root_count_theorem_l997_99757


namespace geometric_sequence_constant_l997_99722

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_constant
  (a : ℕ → ℝ)
  (h_geo : GeometricSequence a)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_eq : (a 1 + a 3) * (a 5 + a 7) = 4 * (a 4)^2) :
  ∃ c : ℝ, ∀ n : ℕ, a n = c :=
sorry

end geometric_sequence_constant_l997_99722


namespace infinite_solutions_condition_l997_99730

theorem infinite_solutions_condition (k : ℝ) : 
  (∀ x : ℝ, 4 * (3 * x - k) = 3 * (4 * x + 10)) ↔ k = -7.5 := by
  sorry

end infinite_solutions_condition_l997_99730


namespace quadratic_factorization_l997_99714

/-- Given a quadratic equation x^2 + px + q = 0 with roots 2 and -3,
    prove that it can be factored as (x - 2)(x + 3) = 0 -/
theorem quadratic_factorization (p q : ℝ) :
  (∀ x, x^2 + p*x + q = 0 ↔ x = 2 ∨ x = -3) →
  ∀ x, x^2 + p*x + q = 0 ↔ (x - 2) * (x + 3) = 0 :=
by sorry

end quadratic_factorization_l997_99714


namespace fishing_moratorium_purpose_l997_99775

/-- Represents a fishing moratorium period -/
structure FishingMoratorium where
  start_date : Nat
  end_date : Nat
  regulations : String

/-- Represents the purpose of a fishing moratorium -/
inductive MoratoriumPurpose
  | ProtectEndangeredSpecies
  | ReducePollution
  | ProtectFishermen
  | SustainableUse

/-- The main purpose of the fishing moratorium -/
def main_purpose (moratorium : FishingMoratorium) : MoratoriumPurpose := sorry

/-- Theorem stating the main purpose of the fishing moratorium -/
theorem fishing_moratorium_purpose 
  (moratorium : FishingMoratorium)
  (h1 : moratorium.start_date = 20150516)
  (h2 : moratorium.end_date = 20150801)
  (h3 : moratorium.regulations = "Ministry of Agriculture regulations") :
  main_purpose moratorium = MoratoriumPurpose.SustainableUse := by sorry

end fishing_moratorium_purpose_l997_99775


namespace corrected_mean_calculation_l997_99709

/-- Calculate the corrected mean of a dataset with misrecorded observations -/
theorem corrected_mean_calculation (n : ℕ) (incorrect_mean : ℚ) 
  (actual_values : List ℚ) (recorded_values : List ℚ) : 
  n = 25 ∧ 
  incorrect_mean = 50 ∧ 
  actual_values = [20, 35, 70] ∧
  recorded_values = [40, 55, 80] →
  (n * incorrect_mean - (recorded_values.sum - actual_values.sum)) / n = 48 := by
  sorry

end corrected_mean_calculation_l997_99709


namespace ocean_depth_for_specific_mountain_l997_99796

/-- Represents a cone-shaped mountain partially submerged in water -/
structure SubmergedMountain where
  height : ℝ
  aboveWaterVolumeFraction : ℝ

/-- Calculates the depth of the ocean at the base of a submerged mountain -/
def oceanDepth (m : SubmergedMountain) : ℝ :=
  m.height * (1 - (m.aboveWaterVolumeFraction ^ (1/3)))

/-- The theorem stating the ocean depth for a specific mountain -/
theorem ocean_depth_for_specific_mountain : 
  let m : SubmergedMountain := { height := 12000, aboveWaterVolumeFraction := 1/5 }
  oceanDepth m = 864 := by
  sorry

end ocean_depth_for_specific_mountain_l997_99796


namespace jump_distance_difference_l997_99784

theorem jump_distance_difference (grasshopper_jump frog_jump : ℕ) 
  (h1 : grasshopper_jump = 13)
  (h2 : frog_jump = 11) :
  grasshopper_jump - frog_jump = 2 := by
  sorry

end jump_distance_difference_l997_99784


namespace quadratic_factorization_l997_99707

theorem quadratic_factorization (a : ℕ+) :
  (∃ m n p q : ℤ, (21 : ℤ) * x^2 + (a : ℤ) * x + 21 = (m * x + n) * (p * x + q)) →
  ∃ k : ℕ+, a = 2 * k := by
  sorry

end quadratic_factorization_l997_99707


namespace stratified_sampling_problem_l997_99760

theorem stratified_sampling_problem (teachers : ℕ) (male_students : ℕ) (female_students : ℕ) 
  (female_sample : ℕ) (total_sample : ℕ) : 
  teachers = 200 → 
  male_students = 1200 → 
  female_students = 1000 → 
  female_sample = 80 → 
  (female_students : ℚ) / (teachers + male_students + female_students : ℚ) * total_sample = female_sample →
  total_sample = 192 := by
sorry

end stratified_sampling_problem_l997_99760


namespace largest_divisor_is_60_l997_99765

def is_largest_divisor (n : ℕ) : Prop :=
  n ∣ 540 ∧ n < 80 ∧ n ∣ 180 ∧
  ∀ m : ℕ, m ∣ 540 → m < 80 → m ∣ 180 → m ≤ n

theorem largest_divisor_is_60 : is_largest_divisor 60 := by
  sorry

end largest_divisor_is_60_l997_99765


namespace l_shape_area_is_58_l997_99744

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ := r.length * r.width

/-- Represents the "L" shaped figure -/
structure LShape where
  outerRectangle : Rectangle
  innerRectangle : Rectangle

/-- Calculates the area of the "L" shaped figure -/
def lShapeArea (l : LShape) : ℝ :=
  rectangleArea l.outerRectangle - rectangleArea l.innerRectangle

/-- Theorem: The area of the "L" shaped figure is 58 square units -/
theorem l_shape_area_is_58 :
  let outer := Rectangle.mk 10 7
  let inner := Rectangle.mk 4 3
  let l := LShape.mk outer inner
  lShapeArea l = 58 := by sorry

end l_shape_area_is_58_l997_99744


namespace union_A_B_when_m_half_B_subset_A_iff_m_range_l997_99791

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {x | (x + m) * (x - 2*m - 1) < 0}
def B : Set ℝ := {x | (1 - x) / (x + 2) > 0}

-- Statement 1
theorem union_A_B_when_m_half : 
  A (1/2) ∪ B = {x | -2 < x ∧ x < 2} := by sorry

-- Statement 2
theorem B_subset_A_iff_m_range :
  ∀ m : ℝ, B ⊆ A m ↔ m ≤ -3/2 ∨ m ≥ 2 := by sorry

end union_A_B_when_m_half_B_subset_A_iff_m_range_l997_99791


namespace pie_crust_flour_calculation_l997_99749

theorem pie_crust_flour_calculation :
  let original_crusts : ℕ := 30
  let new_crusts : ℕ := 40
  let flour_per_original_crust : ℚ := 1 / 5
  let total_flour : ℚ := original_crusts * flour_per_original_crust
  let flour_per_new_crust : ℚ := total_flour / new_crusts
  flour_per_new_crust = 3 / 20 := by sorry

end pie_crust_flour_calculation_l997_99749


namespace area_bounded_by_curve_l997_99703

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.sqrt (4 - x^2)

theorem area_bounded_by_curve : ∫ x in (0)..(2), f x = π := by sorry

end area_bounded_by_curve_l997_99703


namespace upstream_downstream_time_ratio_l997_99785

def boat_speed : ℝ := 18
def stream_speed : ℝ := 6

def upstream_speed : ℝ := boat_speed - stream_speed
def downstream_speed : ℝ := boat_speed + stream_speed

theorem upstream_downstream_time_ratio :
  upstream_speed / downstream_speed = 1 / 2 := by
  sorry

end upstream_downstream_time_ratio_l997_99785


namespace expression_simplification_l997_99740

theorem expression_simplification (x : ℝ) (hx : x ≠ 0) :
  ((2 * x^2)^3 - 6 * x^3 * (x^3 - 2 * x^2)) / (2 * x^4) = x^2 + 6 * x := by
  sorry

end expression_simplification_l997_99740


namespace sqrt_8_div_7_same_type_as_sqrt_2_l997_99779

-- Define what it means for two quadratic radicals to be of the same type
def same_type (a b : ℝ) : Prop :=
  ∃ (q : ℚ), a = q * b

-- State the theorem
theorem sqrt_8_div_7_same_type_as_sqrt_2 :
  same_type (Real.sqrt 8 / 7) (Real.sqrt 2) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 2) ∧
  ¬ same_type (Real.sqrt (1/3)) (Real.sqrt 2) ∧
  ¬ same_type (Real.sqrt 12) (Real.sqrt 2) :=
by sorry

end sqrt_8_div_7_same_type_as_sqrt_2_l997_99779


namespace horner_v1_for_f_at_neg_two_l997_99783

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = x^6 - 5x^5 + 6x^4 + x^2 + 0.3x + 2 -/
def f : List ℝ := [2, 0.3, 1, 0, 6, -5, 1]

/-- Theorem: v1 in Horner's method for f(x) at x = -2 is -7 -/
theorem horner_v1_for_f_at_neg_two :
  (horner (f.tail) (-2) : ℝ) = -7 := by
  sorry

end horner_v1_for_f_at_neg_two_l997_99783


namespace arithmetic_progression_pairs_l997_99724

/-- A pair of real numbers (a, b) forms an arithmetic progression with 10 and ab if
    the differences between consecutive terms are equal. -/
def is_arithmetic_progression (a b : ℝ) : Prop :=
  (a - 10 = b - a) ∧ (b - a = a * b - b)

/-- The only pairs (a, b) of real numbers such that 10, a, b, ab form an arithmetic progression
    are (4, -2) and (2.5, -5). -/
theorem arithmetic_progression_pairs :
  ∀ a b : ℝ, is_arithmetic_progression a b ↔ (a = 4 ∧ b = -2) ∨ (a = 2.5 ∧ b = -5) := by
  sorry

end arithmetic_progression_pairs_l997_99724


namespace vector_operation_l997_99781

theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (2, 1)) (h2 : b = (-3, 4)) :
  2 • a - b = (7, -2) := by
  sorry

end vector_operation_l997_99781


namespace quadratic_polynomial_condition_l997_99787

/-- A quadratic polynomial of the form ax^2 + bx + c -/
structure QuadraticPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Evaluate a quadratic polynomial at a given x -/
def QuadraticPolynomial.evaluate (p : QuadraticPolynomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- A quadratic polynomial satisfies the given condition if
    f(a) = a, f(b) = b, and f(c) = c -/
def satisfies_condition (p : QuadraticPolynomial) : Prop :=
  p.evaluate p.a = p.a ∧
  p.evaluate p.b = p.b ∧
  p.evaluate p.c = p.c

/-- The theorem stating that only x^2 + x - 1 and x - 2 satisfy the condition -/
theorem quadratic_polynomial_condition :
  ∀ p : QuadraticPolynomial,
    satisfies_condition p →
      (p = ⟨1, 1, -1⟩ ∨ p = ⟨0, 1, -2⟩) :=
sorry

end quadratic_polynomial_condition_l997_99787


namespace ellipse_perpendicular_points_product_l997_99734

/-- Given an ellipse with semi-major axis a and semi-minor axis b, 
    prove that the product of the distances from the origin to any two 
    perpendicular points on the ellipse is at least 2a²b²/(a² + b²) -/
theorem ellipse_perpendicular_points_product (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  ∀ (P Q : ℝ × ℝ), 
    (P.1^2 / a^2 + P.2^2 / b^2 = 1) →
    (Q.1^2 / a^2 + Q.2^2 / b^2 = 1) →
    (P.1 * Q.1 + P.2 * Q.2 = 0) →
    (P.1^2 + P.2^2) * (Q.1^2 + Q.2^2) ≥ (2 * a^2 * b^2) / (a^2 + b^2) := by
  sorry

end ellipse_perpendicular_points_product_l997_99734


namespace square_region_perimeter_l997_99747

/-- Given a region formed by eight congruent squares arranged in a vertical rectangle
    with a total area of 512 square centimeters, the perimeter of the region is 160 centimeters. -/
theorem square_region_perimeter : 
  ∀ (side_length : ℝ),
  side_length > 0 →
  8 * side_length^2 = 512 →
  2 * (7 * side_length + 3 * side_length) = 160 :=
by sorry

end square_region_perimeter_l997_99747


namespace not_complete_residue_sum_l997_99737

/-- For an even number n, if a and b are complete residue systems modulo n,
    then their pairwise sum is not a complete residue system modulo n. -/
theorem not_complete_residue_sum
  (n : ℕ) (hn : Even n) 
  (a b : Fin n → ℕ)
  (ha : Function.Surjective (λ i => a i % n))
  (hb : Function.Surjective (λ i => b i % n)) :
  ¬ Function.Surjective (λ i => (a i + b i) % n) :=
sorry

end not_complete_residue_sum_l997_99737


namespace specific_polyhedron_space_diagonals_l997_99769

/-- A convex polyhedron with specified properties -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ
  pentagonal_faces : ℕ

/-- Calculate the number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  Nat.choose Q.vertices 2 - Q.edges - (2 * Q.quadrilateral_faces + 5 * Q.pentagonal_faces)

/-- Theorem: A specific convex polyhedron Q has 323 space diagonals -/
theorem specific_polyhedron_space_diagonals :
  let Q : ConvexPolyhedron := {
    vertices := 30,
    edges := 72,
    faces := 44,
    triangular_faces := 30,
    quadrilateral_faces := 10,
    pentagonal_faces := 4
  }
  space_diagonals Q = 323 := by
  sorry


end specific_polyhedron_space_diagonals_l997_99769


namespace segment_length_product_l997_99701

theorem segment_length_product (a : ℝ) : 
  (((3 * a - 7)^2 + (a - 7)^2) = 90) → 
  (∃ b : ℝ, (((3 * b - 7)^2 + (b - 7)^2) = 90) ∧ (a * b = 0.8)) :=
by sorry

end segment_length_product_l997_99701


namespace inscribed_squares_ratio_l997_99754

/-- A square inscribed in a right triangle with one vertex at the right angle -/
def square_in_triangle_vertex (a b c : ℝ) (x : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ (c - x) / x = a / b

/-- A square inscribed in a right triangle with one side on the hypotenuse -/
def square_in_triangle_hypotenuse (a b c : ℝ) (y : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∧ (a - y) / y = (b - y) / y

theorem inscribed_squares_ratio :
  ∀ x y : ℝ,
    square_in_triangle_vertex 5 12 13 x →
    square_in_triangle_hypotenuse 6 8 10 y →
    x / y = 37 / 35 := by
  sorry

end inscribed_squares_ratio_l997_99754


namespace candy_mix_equations_correct_l997_99780

/-- Represents the candy mixing problem -/
structure CandyMix where
  x : ℝ  -- quantity of 36 yuan/kg candy
  y : ℝ  -- quantity of 20 yuan/kg candy
  total_weight : ℝ  -- total weight of mixed candy
  mixed_price : ℝ  -- price of mixed candy per kg
  high_price : ℝ  -- price of more expensive candy per kg
  low_price : ℝ  -- price of less expensive candy per kg

/-- The system of equations correctly describes the candy mixing problem -/
theorem candy_mix_equations_correct (mix : CandyMix) 
  (h1 : mix.total_weight = 100)
  (h2 : mix.mixed_price = 28)
  (h3 : mix.high_price = 36)
  (h4 : mix.low_price = 20) :
  (mix.x + mix.y = mix.total_weight) ∧ 
  (mix.high_price * mix.x + mix.low_price * mix.y = mix.mixed_price * mix.total_weight) :=
sorry

end candy_mix_equations_correct_l997_99780


namespace no_primes_divisible_by_45_l997_99741

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem no_primes_divisible_by_45 : ¬∃ p : ℕ, is_prime p ∧ 45 ∣ p :=
sorry

end no_primes_divisible_by_45_l997_99741


namespace average_of_five_quantities_l997_99762

theorem average_of_five_quantities (q1 q2 q3 q4 q5 : ℝ) 
  (h1 : (q1 + q2 + q3) / 3 = 4)
  (h2 : (q4 + q5) / 2 = 19) :
  (q1 + q2 + q3 + q4 + q5) / 5 = 10 := by
  sorry

end average_of_five_quantities_l997_99762


namespace emily_walk_distance_l997_99702

/-- Calculates the total distance walked given the number of blocks and block length -/
def total_distance (blocks_west : ℕ) (blocks_south : ℕ) (miles_per_block : ℚ) : ℚ :=
  (blocks_west + blocks_south : ℚ) * miles_per_block

/-- Proves that walking 8 blocks west and 10 blocks south, with each block being 1/4 mile, results in a total distance of 4.5 miles -/
theorem emily_walk_distance : total_distance 8 10 (1/4) = 4.5 := by
  sorry

end emily_walk_distance_l997_99702


namespace rectangle_max_area_l997_99726

/-- Given a rectangle with perimeter 60 and one side 5 units longer than the other,
    the maximum area is 218.75 square units. -/
theorem rectangle_max_area :
  ∀ x y : ℝ,
  x > 0 → y > 0 →
  2 * (x + y) = 60 →
  y = x + 5 →
  x * y ≤ 218.75 :=
by sorry

end rectangle_max_area_l997_99726


namespace curve_length_of_right_square_prism_l997_99768

/-- Represents a right square prism -/
structure RightSquarePrism where
  sideEdge : ℝ
  baseEdge : ℝ

/-- Calculates the total length of curves on the surface of a right square prism
    formed by points at a given distance from a vertex -/
def totalCurveLength (prism : RightSquarePrism) (distance : ℝ) : ℝ :=
  sorry

/-- The theorem statement -/
theorem curve_length_of_right_square_prism :
  let prism : RightSquarePrism := ⟨4, 4⟩
  totalCurveLength prism 3 = 6 * Real.pi :=
by sorry

end curve_length_of_right_square_prism_l997_99768


namespace opposite_of_eight_l997_99723

theorem opposite_of_eight :
  ∀ x : ℤ, x + 8 = 0 ↔ x = -8 := by sorry

end opposite_of_eight_l997_99723


namespace total_photos_lisa_robert_l997_99773

def claire_photos : ℕ := 8
def lisa_photos : ℕ := 3 * claire_photos
def robert_photos : ℕ := claire_photos + 16

theorem total_photos_lisa_robert : lisa_photos + robert_photos = 48 := by
  sorry

end total_photos_lisa_robert_l997_99773


namespace exam_correct_percentage_l997_99746

/-- Given an exam with two sections, calculate the percentage of correctly solved problems. -/
theorem exam_correct_percentage (y : ℕ) : 
  let total_problems := 10 * y
  let section1_problems := 6 * y
  let section2_problems := 4 * y
  let missed_section1 := 2 * y
  let missed_section2 := y
  let correct_problems := (section1_problems - missed_section1) + (section2_problems - missed_section2)
  (correct_problems : ℚ) / total_problems * 100 = 70 := by
  sorry

end exam_correct_percentage_l997_99746


namespace exponential_comparison_l997_99792

theorem exponential_comparison (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  a^(-1 : ℝ) > a^(2 : ℝ) := by
  sorry

end exponential_comparison_l997_99792


namespace cat_food_sale_revenue_l997_99739

/-- Calculates the total revenue from cat food sales during a promotion --/
theorem cat_food_sale_revenue : 
  let original_price : ℚ := 25
  let first_group_size : ℕ := 8
  let first_group_cases : ℕ := 3
  let first_group_discount : ℚ := 15/100
  let second_group_size : ℕ := 4
  let second_group_cases : ℕ := 2
  let second_group_discount : ℚ := 10/100
  let third_group_size : ℕ := 8
  let third_group_cases : ℕ := 1
  let third_group_discount : ℚ := 0

  let first_group_revenue := (first_group_size * first_group_cases : ℚ) * 
    (original_price * (1 - first_group_discount))
  let second_group_revenue := (second_group_size * second_group_cases : ℚ) * 
    (original_price * (1 - second_group_discount))
  let third_group_revenue := (third_group_size * third_group_cases : ℚ) * 
    (original_price * (1 - third_group_discount))

  let total_revenue := first_group_revenue + second_group_revenue + third_group_revenue

  total_revenue = 890 := by
  sorry

end cat_food_sale_revenue_l997_99739


namespace exists_same_answer_question_l997_99748

/-- A person who either always tells the truth or always lies -/
inductive Person
| TruthTeller
| Liar

/-- The answer a person gives to a question -/
inductive Answer
| Yes
| No

/-- A question that can be asked to a person -/
def Question := Person → Answer

/-- The actual answer to a question about a person -/
def actualAnswer (p : Person) (q : Question) : Answer :=
  match p with
  | Person.TruthTeller => q Person.TruthTeller
  | Person.Liar => match q Person.Liar with
    | Answer.Yes => Answer.No
    | Answer.No => Answer.Yes

/-- There exists a question that makes both a truth-teller and a liar give the same answer -/
theorem exists_same_answer_question : ∃ (q : Question),
  actualAnswer Person.TruthTeller q = actualAnswer Person.Liar q :=
sorry

end exists_same_answer_question_l997_99748


namespace balls_in_bins_probability_ratio_l997_99743

def number_of_balls : ℕ := 20
def number_of_bins : ℕ := 5

def p' : ℚ := (number_of_bins * (number_of_bins - 1) * (Nat.factorial 11) / 
  (Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 4 * Nat.factorial 3)) / 
  (Nat.choose (number_of_balls + number_of_bins - 1) (number_of_bins - 1))

def q : ℚ := (Nat.factorial number_of_balls) / 
  ((Nat.factorial 4)^number_of_bins * Nat.factorial number_of_bins) / 
  (Nat.choose (number_of_balls + number_of_bins - 1) (number_of_bins - 1))

theorem balls_in_bins_probability_ratio : 
  p' / q = 8 / 57 := by sorry

end balls_in_bins_probability_ratio_l997_99743


namespace total_problems_practiced_l997_99755

def marvin_yesterday : ℕ := 40
def marvin_today : ℕ := 3 * marvin_yesterday
def arvin_yesterday : ℕ := 2 * marvin_yesterday
def arvin_today : ℕ := 2 * marvin_today
def kevin_yesterday : ℕ := 30
def kevin_today : ℕ := kevin_yesterday ^ 2

theorem total_problems_practiced :
  marvin_yesterday + marvin_today + arvin_yesterday + arvin_today + kevin_yesterday + kevin_today = 1410 :=
by sorry

end total_problems_practiced_l997_99755


namespace range_of_m_l997_99788

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 - 2*x - 3

-- Define the necessary condition
def necessary_condition (m : ℝ) (x : ℝ) : Prop :=
  x < m - 1 ∨ x > m + 1

theorem range_of_m :
  ∀ m : ℝ,
    (∀ x : ℝ, f x > 0 → necessary_condition m x) ∧
    (∃ x : ℝ, necessary_condition m x ∧ f x ≤ 0) →
    0 ≤ m ∧ m ≤ 2 :=
sorry

end range_of_m_l997_99788


namespace coin_move_termination_uniqueness_l997_99725

-- Define the coin configuration as a function from integers to natural numbers
def CoinConfiguration := ℤ → ℕ

-- Define a legal move
def is_legal_move (c₁ c₂ : CoinConfiguration) : Prop :=
  ∃ i : ℤ, c₁ i ≥ 2 ∧
    c₂ i = c₁ i - 2 ∧
    c₂ (i - 1) = c₁ (i - 1) + 1 ∧
    c₂ (i + 1) = c₁ (i + 1) + 1 ∧
    ∀ j : ℤ, j ≠ i ∧ j ≠ (i - 1) ∧ j ≠ (i + 1) → c₂ j = c₁ j

-- Define a legal sequence of moves
def legal_sequence (c₀ : CoinConfiguration) (n : ℕ) (c : ℕ → CoinConfiguration) : Prop :=
  c 0 = c₀ ∧
  ∀ i : ℕ, i < n → is_legal_move (c i) (c (i + 1))

-- Define a terminal configuration
def is_terminal (c : CoinConfiguration) : Prop :=
  ∀ i : ℤ, c i ≤ 1

-- The main theorem
theorem coin_move_termination_uniqueness
  (c₀ : CoinConfiguration)
  (n₁ n₂ : ℕ)
  (c₁ : ℕ → CoinConfiguration)
  (c₂ : ℕ → CoinConfiguration)
  (h₁ : legal_sequence c₀ n₁ c₁)
  (h₂ : legal_sequence c₀ n₂ c₂)
  (t₁ : is_terminal (c₁ n₁))
  (t₂ : is_terminal (c₂ n₂)) :
  n₁ = n₂ ∧ c₁ n₁ = c₂ n₂ :=
sorry

end coin_move_termination_uniqueness_l997_99725


namespace cost_price_correct_l997_99745

/-- The cost price of an eye-protection lamp -/
def cost_price : ℝ := 150

/-- The original selling price of the lamp -/
def original_price : ℝ := 200

/-- The discount rate during the special period -/
def discount_rate : ℝ := 0.1

/-- The profit rate after the discount -/
def profit_rate : ℝ := 0.2

/-- Theorem stating that the cost price is correct given the conditions -/
theorem cost_price_correct : 
  original_price * (1 - discount_rate) = cost_price * (1 + profit_rate) :=
sorry

end cost_price_correct_l997_99745


namespace projectile_speed_problem_l997_99752

theorem projectile_speed_problem (initial_distance : ℝ) (second_projectile_speed : ℝ) (time_to_meet : ℝ) :
  initial_distance = 1182 →
  second_projectile_speed = 525 →
  time_to_meet = 1.2 →
  ∃ (first_projectile_speed : ℝ),
    first_projectile_speed = 460 ∧
    (first_projectile_speed + second_projectile_speed) * time_to_meet = initial_distance :=
by sorry

end projectile_speed_problem_l997_99752


namespace inverse_as_linear_combination_l997_99728

def N : Matrix (Fin 2) (Fin 2) ℝ := !![3, 0; 2, -4]

theorem inverse_as_linear_combination :
  ∃ (c d : ℝ), N⁻¹ = c • N + d • (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧ c = 1/12 ∧ d = 1/12 := by
  sorry

end inverse_as_linear_combination_l997_99728


namespace intersection_complement_theorem_l997_99794

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def P : Set ℕ := {1, 2, 3, 4}
def Q : Set ℕ := {3, 4, 5}

theorem intersection_complement_theorem :
  P ∩ (U \ Q) = {1, 2} := by
  sorry

end intersection_complement_theorem_l997_99794


namespace geometric_sequence_problem_l997_99790

theorem geometric_sequence_problem (a : ℝ) (h1 : a > 0) 
  (h2 : ∃ r : ℝ, r ≠ 0 ∧ a = 25 * r ∧ 7/9 = a * r) : 
  a = 5 * Real.sqrt 7 / 3 := by
  sorry

end geometric_sequence_problem_l997_99790


namespace place_value_comparison_l997_99763

theorem place_value_comparison (n : Real) (h : n = 85376.4201) : 
  (10 : Real) / (1 / 10 : Real) = 100 := by
  sorry

end place_value_comparison_l997_99763


namespace two_from_three_l997_99751

/-- The number of combinations of k items from a set of n items -/
def combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Theorem: There are 3 ways to choose 2 items from a set of 3 items -/
theorem two_from_three : combinations 3 2 = 3 := by sorry

end two_from_three_l997_99751


namespace exists_specific_polyhedron_l997_99771

/-- A face of a polyhedron -/
structure Face where
  sides : ℕ

/-- A polyhedron -/
structure Polyhedron where
  faces : List Face

/-- Counts the number of faces with a given number of sides -/
def countFaces (p : Polyhedron) (n : ℕ) : ℕ :=
  p.faces.filter (λ f => f.sides = n) |>.length

/-- Theorem: There exists a polyhedron with exactly 6 faces,
    where 2 faces are triangles, 2 faces are quadrilaterals, and 2 faces are pentagons -/
theorem exists_specific_polyhedron :
  ∃ p : Polyhedron,
    p.faces.length = 6 ∧
    countFaces p 3 = 2 ∧
    countFaces p 4 = 2 ∧
    countFaces p 5 = 2 :=
  sorry

end exists_specific_polyhedron_l997_99771


namespace unique_solution_condition_l997_99717

/-- Given real numbers m, n, p, q, and functions f and g,
    prove that f(g(x)) = g(f(x)) has a unique solution
    if and only if mq = p and q = n -/
theorem unique_solution_condition (m n p q : ℝ)
  (f : ℝ → ℝ) (g : ℝ → ℝ)
  (hf : ∀ x, f x = m * x^2 + n)
  (hg : ∀ x, g x = p * x + q) :
  (∃! x, f (g x) = g (f x)) ↔ (m * q = p ∧ q = n) :=
sorry

end unique_solution_condition_l997_99717


namespace contrapositive_equivalence_l997_99732

theorem contrapositive_equivalence (x y : ℝ) :
  (¬(x = 0 ∧ y = 0) → x^2 + y^2 ≠ 0) ↔
  (x^2 + y^2 = 0 → x = 0 ∧ y = 0) :=
by sorry

end contrapositive_equivalence_l997_99732


namespace same_point_on_bisector_l997_99738

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the angle bisector of the first and third quadrants
def firstThirdQuadrantBisector : Set Point2D :=
  { p : Point2D | p.x = p.y }

theorem same_point_on_bisector (a b : ℝ) :
  (Point2D.mk a b = Point2D.mk b a) →
  Point2D.mk a b ∈ firstThirdQuadrantBisector := by
  sorry

end same_point_on_bisector_l997_99738


namespace mod_equivalence_unique_solution_l997_99720

theorem mod_equivalence_unique_solution : 
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -1774 [ZMOD 7] ∧ n = 2 := by
sorry

end mod_equivalence_unique_solution_l997_99720


namespace arithmetic_sequence_a10_l997_99733

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a10 (a : ℕ → ℤ) :
  arithmetic_sequence a → a 7 = 4 → a 8 = 1 → a 10 = -5 := by
  sorry

end arithmetic_sequence_a10_l997_99733


namespace arthur_sword_problem_l997_99715

theorem arthur_sword_problem (A B : ℕ) : 
  5 * A + 7 * B = 49 → A - B = 5 := by
sorry

end arthur_sword_problem_l997_99715


namespace simplify_A_plus_2B_value_A_plus_2B_at_1_neg1_l997_99718

-- Define polynomials A and B
def A (a b : ℝ) : ℝ := 3*a^2 - 6*a*b + b^2
def B (a b : ℝ) : ℝ := -2*a^2 + 3*a*b - 5*b^2

-- Theorem for the simplified form of A + 2B
theorem simplify_A_plus_2B (a b : ℝ) : A a b + 2 * B a b = -a^2 - 9*b^2 := by sorry

-- Theorem for the value of A + 2B when a = 1 and b = -1
theorem value_A_plus_2B_at_1_neg1 : A 1 (-1) + 2 * B 1 (-1) = -10 := by sorry

end simplify_A_plus_2B_value_A_plus_2B_at_1_neg1_l997_99718


namespace total_books_correct_l997_99710

/-- Calculates the total number of books after a purchase -/
def total_books (initial : Real) (bought : Real) : Real :=
  initial + bought

/-- Theorem: The total number of books is the sum of initial and bought books -/
theorem total_books_correct (initial : Real) (bought : Real) :
  total_books initial bought = initial + bought := by
  sorry

end total_books_correct_l997_99710


namespace central_sum_theorem_l997_99778

/-- Represents a 4x4 matrix of integers -/
def Matrix4x4 := Fin 4 → Fin 4 → ℕ

/-- Checks if two positions in the matrix are adjacent -/
def isAdjacent (a b : Fin 4 × Fin 4) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ b.2 = a.2 + 1)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ b.1 = a.1 + 1))

/-- Checks if the matrix contains all numbers from 1 to 16 -/
def containsAllNumbers (m : Matrix4x4) : Prop :=
  ∀ n : Fin 16, ∃ i j : Fin 4, m i j = n.val + 1

/-- Checks if consecutive numbers are adjacent in the matrix -/
def consecutiveAdjacent (m : Matrix4x4) : Prop :=
  ∀ n : Fin 15, ∃ i₁ j₁ i₂ j₂ : Fin 4,
    m i₁ j₁ = n.val + 1 ∧ m i₂ j₂ = n.val + 2 ∧ isAdjacent (i₁, j₁) (i₂, j₂)

/-- Calculates the sum of corner numbers in the matrix -/
def cornerSum (m : Matrix4x4) : ℕ :=
  m 0 0 + m 0 3 + m 3 0 + m 3 3

/-- Calculates the sum of central numbers in the matrix -/
def centerSum (m : Matrix4x4) : ℕ :=
  m 1 1 + m 1 2 + m 2 1 + m 2 2

theorem central_sum_theorem (m : Matrix4x4)
  (h1 : containsAllNumbers m)
  (h2 : consecutiveAdjacent m)
  (h3 : cornerSum m = 34) :
  centerSum m = 34 := by
  sorry

end central_sum_theorem_l997_99778


namespace pen_pencil_length_difference_l997_99712

theorem pen_pencil_length_difference :
  ∀ (rubber pen pencil : ℝ),
  pen = rubber + 3 →
  pencil = 12 →
  rubber + pen + pencil = 29 →
  pencil - pen = 2 :=
by
  sorry

end pen_pencil_length_difference_l997_99712


namespace iron_conducts_electricity_l997_99735

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Metal : U → Prop)
variable (ConductsElectricity : U → Prop)

-- Define iron as a constant in our universe
variable (iron : U)

-- Theorem statement
theorem iron_conducts_electricity 
  (h1 : ∀ x, Metal x → ConductsElectricity x) 
  (h2 : Metal iron) : 
  ConductsElectricity iron := by
  sorry


end iron_conducts_electricity_l997_99735


namespace high_octane_half_cost_l997_99774

/-- Represents the composition and cost of a fuel mixture -/
structure FuelMixture where
  high_octane_units : ℕ
  regular_octane_units : ℕ
  high_octane_cost_multiplier : ℕ

/-- Calculates the fraction of the total cost due to high octane fuel -/
def high_octane_cost_fraction (fuel : FuelMixture) : ℚ :=
  let high_octane_cost := fuel.high_octane_units * fuel.high_octane_cost_multiplier
  let regular_octane_cost := fuel.regular_octane_units
  let total_cost := high_octane_cost + regular_octane_cost
  high_octane_cost / total_cost

/-- Theorem: The fraction of the cost due to high octane is 1/2 for the given fuel mixture -/
theorem high_octane_half_cost (fuel : FuelMixture) 
    (h1 : fuel.high_octane_units = 1515)
    (h2 : fuel.regular_octane_units = 4545)
    (h3 : fuel.high_octane_cost_multiplier = 3) :
  high_octane_cost_fraction fuel = 1/2 := by
  sorry

end high_octane_half_cost_l997_99774


namespace expression_evaluation_l997_99798

theorem expression_evaluation : 
  (0.66 : ℝ)^3 - (0.1 : ℝ)^3 / (0.66 : ℝ)^2 + 0.066 + (0.1 : ℝ)^2 = 0.3612 := by
  sorry

end expression_evaluation_l997_99798
