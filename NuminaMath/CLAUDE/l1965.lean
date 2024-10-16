import Mathlib

namespace NUMINAMATH_CALUDE_latus_rectum_of_parabola_l1965_196585

/-- Given a parabola with equation y^2 = 8x, prove that its latus rectum has equation x = -2 -/
theorem latus_rectum_of_parabola (x y : ℝ) :
  y^2 = 8*x → (∃ (a : ℝ), a = -2 ∧ ∀ (x₀ y₀ : ℝ), y₀^2 = 8*x₀ → x₀ = a → 
    (x₀, y₀) ∈ {p : ℝ × ℝ | p.1 = a ∧ p.2^2 = 8*p.1}) :=
by sorry

end NUMINAMATH_CALUDE_latus_rectum_of_parabola_l1965_196585


namespace NUMINAMATH_CALUDE_line_and_symmetric_point_l1965_196578

/-- Given a line with inclination angle 135° passing through (1,1), 
    prove its equation and find the symmetric point of (3,4) with respect to it. -/
theorem line_and_symmetric_point :
  let l : Set (ℝ × ℝ) := {(x, y) | x + y - 2 = 0}
  let P : ℝ × ℝ := (1, 1)
  let A : ℝ × ℝ := (3, 4)
  let inclination_angle : ℝ := 135 * (π / 180)
  -- Line l passes through P
  (P ∈ l) →
  -- The slope of l is tan(135°)
  (∀ (x y : ℝ), (x, y) ∈ l → y - P.2 = Real.tan inclination_angle * (x - P.1)) →
  -- The equation of l is x + y - 2 = 0
  (∀ (x y : ℝ), (x, y) ∈ l ↔ x + y - 2 = 0) ∧
  -- The symmetric point A' of A with respect to l has coordinates (-2, -1)
  (∃ (A' : ℝ × ℝ), 
    -- A' is on the opposite side of l from A
    (A'.1 + A'.2 - 2) * (A.1 + A.2 - 2) < 0 ∧
    -- The midpoint of AA' is on l
    ((A.1 + A'.1) / 2 + (A.2 + A'.2) / 2 - 2 = 0) ∧
    -- AA' is perpendicular to l
    ((A'.2 - A.2) / (A'.1 - A.1)) * Real.tan inclination_angle = -1 ∧
    -- A' has coordinates (-2, -1)
    A' = (-2, -1)) := by
  sorry

end NUMINAMATH_CALUDE_line_and_symmetric_point_l1965_196578


namespace NUMINAMATH_CALUDE_unique_quadruple_existence_l1965_196567

theorem unique_quadruple_existence : 
  ∃! (a b c d : ℝ), 
    0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧
    a^2 + b^2 + c^2 + d^2 = 4 ∧
    (a + b + c + d) * (a^2 + b^2 + c^2 + d^2) = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadruple_existence_l1965_196567


namespace NUMINAMATH_CALUDE_smallest_sum_of_product_l1965_196572

theorem smallest_sum_of_product (a b c d e : ℕ+) : 
  a * b * c * d * e = Nat.factorial 12 → 
  (∀ w x y z v : ℕ+, w * x * y * z * v = Nat.factorial 12 → 
    a + b + c + d + e ≤ w + x + y + z + v) →
  a + b + c + d + e = 501 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_of_product_l1965_196572


namespace NUMINAMATH_CALUDE_line_intersection_area_ratio_l1965_196509

/-- Given a line y = b - 2x where 0 < b < 6, intersecting the y-axis at P and the line x=6 at S,
    if the ratio of the area of triangle QRS to the area of triangle QOP is 4:9,
    then b = √(1296/11). -/
theorem line_intersection_area_ratio (b : ℝ) : 
  0 < b → b < 6 → 
  let line := fun x => b - 2 * x
  let P := (0, b)
  let S := (6, line 6)
  let Q := (b / 2, 0)
  let R := (6, 0)
  let area_QOP := (1 / 2) * (b / 2) * b
  let area_QRS := (1 / 2) * (6 - b / 2) * |b - 12|
  area_QRS / area_QOP = 4 / 9 →
  b = Real.sqrt (1296 / 11) := by
sorry

end NUMINAMATH_CALUDE_line_intersection_area_ratio_l1965_196509


namespace NUMINAMATH_CALUDE_wooden_block_length_l1965_196583

-- Define the conversion factor from meters to centimeters
def meters_to_cm : ℝ := 100

-- Define the initial length in meters
def initial_length_m : ℝ := 31

-- Define the additional length in centimeters
def additional_length_cm : ℝ := 30

-- Theorem to prove
theorem wooden_block_length :
  (initial_length_m * meters_to_cm + additional_length_cm) = 3130 := by
  sorry

end NUMINAMATH_CALUDE_wooden_block_length_l1965_196583


namespace NUMINAMATH_CALUDE_expansion_coefficient_sum_l1965_196563

theorem expansion_coefficient_sum (m : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (m * x - 1)^5 = a₅ * x^5 + a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀) →
  a₁ + a₂ + a₃ + a₄ + a₅ = 33 →
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_expansion_coefficient_sum_l1965_196563


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l1965_196504

/-- Represents the percentage of employees who are men -/
def percentMen : ℝ := 35

/-- Represents the percentage of employees who are women -/
def percentWomen : ℝ := 100 - percentMen

/-- Represents the percentage of all employees who attended the picnic -/
def percentAttended : ℝ := 33

/-- Represents the percentage of women who attended the picnic -/
def percentWomenAttended : ℝ := 40

/-- Represents the percentage of men who attended the picnic -/
def percentMenAttended : ℝ := 20

theorem company_picnic_attendance :
  percentMenAttended * (percentMen / 100) + percentWomenAttended * (percentWomen / 100) = percentAttended / 100 :=
by sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l1965_196504


namespace NUMINAMATH_CALUDE_remainder_17_45_mod_5_l1965_196561

theorem remainder_17_45_mod_5 : 17^45 % 5 = 2 := by sorry

end NUMINAMATH_CALUDE_remainder_17_45_mod_5_l1965_196561


namespace NUMINAMATH_CALUDE_complex_power_difference_l1965_196520

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_power_difference (h : i^2 = -1) : (1 + i)^12 - (1 - i)^12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_difference_l1965_196520


namespace NUMINAMATH_CALUDE_field_ratio_is_two_to_one_l1965_196508

/-- Proves that the ratio of length to width of a rectangular field is 2:1 given specific conditions --/
theorem field_ratio_is_two_to_one (field_length field_width pond_side : ℝ) : 
  field_length = 80 →
  pond_side = 8 →
  field_length * field_width = 50 * (pond_side * pond_side) →
  field_length / field_width = 2 := by
sorry

end NUMINAMATH_CALUDE_field_ratio_is_two_to_one_l1965_196508


namespace NUMINAMATH_CALUDE_women_in_room_l1965_196538

theorem women_in_room (initial_men : ℕ) (initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →
  (initial_men + 2) = 14 →
  (2 * (initial_women - 3)) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_women_in_room_l1965_196538


namespace NUMINAMATH_CALUDE_final_price_percentage_l1965_196569

/-- Given a suggested retail price, store discount, and additional discount,
    calculates the percentage of the original price paid. -/
def percentage_paid (suggested_retail_price : ℝ) (store_discount : ℝ) (additional_discount : ℝ) : ℝ :=
  (1 - store_discount) * (1 - additional_discount) * 100

/-- Theorem stating that with a 20% store discount and 10% additional discount,
    the final price paid is 72% of the suggested retail price. -/
theorem final_price_percentage (suggested_retail_price : ℝ) 
  (h1 : suggested_retail_price > 0)
  (h2 : store_discount = 0.2)
  (h3 : additional_discount = 0.1) :
  percentage_paid suggested_retail_price store_discount additional_discount = 72 := by
  sorry

end NUMINAMATH_CALUDE_final_price_percentage_l1965_196569


namespace NUMINAMATH_CALUDE_solution_replacement_fraction_l1965_196524

theorem solution_replacement_fraction (Q : ℝ) (h : Q > 0) :
  let initial_conc : ℝ := 0.70
  let replacement_conc : ℝ := 0.25
  let new_conc : ℝ := 0.35
  let x : ℝ := (new_conc * Q - initial_conc * Q) / (replacement_conc * Q - initial_conc * Q)
  x = 7 / 9 := by
sorry

end NUMINAMATH_CALUDE_solution_replacement_fraction_l1965_196524


namespace NUMINAMATH_CALUDE_banana_pear_weight_equivalence_l1965_196551

theorem banana_pear_weight_equivalence (banana_weight pear_weight : ℝ) 
  (h1 : 9 * banana_weight = 6 * pear_weight) :
  36 * banana_weight = 24 * pear_weight := by
  sorry

end NUMINAMATH_CALUDE_banana_pear_weight_equivalence_l1965_196551


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l1965_196527

theorem min_value_squared_sum (a b t k : ℝ) (hk : k > 0) (ht : a + k * b = t) :
  a^2 + k^2 * b^2 ≥ ((1 + k^2) * t^2) / (1 + k)^2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l1965_196527


namespace NUMINAMATH_CALUDE_triangle_area_specific_l1965_196589

/-- The area of a triangle given the coordinates of its vertices -/
def triangleArea (x1 y1 x2 y2 x3 y3 : ℤ) : ℚ :=
  (1 / 2 : ℚ) * |x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)|

/-- Theorem: The area of a triangle with vertices at (-1,-1), (2,3), and (-4,0) is 8.5 square units -/
theorem triangle_area_specific : triangleArea (-1) (-1) 2 3 (-4) 0 = 17/2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_specific_l1965_196589


namespace NUMINAMATH_CALUDE_four_digit_sum_problem_l1965_196531

theorem four_digit_sum_problem (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 →
  6 * (a + b + c + d) * 1111 = 73326 →
  ({a, b, c, d} : Finset ℕ) = {1, 2, 3, 5} :=
by sorry

end NUMINAMATH_CALUDE_four_digit_sum_problem_l1965_196531


namespace NUMINAMATH_CALUDE_least_value_ba_l1965_196532

/-- Given a number in the form 11,0ab that is divisible by 115, 
    the least possible value of b × a is 0 -/
theorem least_value_ba (a b : ℕ) : 
  a < 10 → b < 10 → (11000 + 100 * a + b) % 115 = 0 → 
  ∀ (c d : ℕ), c < 10 → d < 10 → (11000 + 100 * c + d) % 115 = 0 → 
  b * a ≤ d * c := by
  sorry

end NUMINAMATH_CALUDE_least_value_ba_l1965_196532


namespace NUMINAMATH_CALUDE_complex_additive_inverse_l1965_196594

theorem complex_additive_inverse (b : ℝ) : 
  let z : ℂ := (4 + b * Complex.I) / (1 + Complex.I)
  (z.re = -z.im) → b = 0 := by
sorry

end NUMINAMATH_CALUDE_complex_additive_inverse_l1965_196594


namespace NUMINAMATH_CALUDE_condition1_condition2_f_satisfies_conditions_l1965_196558

/-- A function satisfying the given conditions -/
def f (x : ℝ) := -3 * x

/-- The first condition: f(x) + f(-x) = 0 for all x ∈ ℝ -/
theorem condition1 : ∀ x : ℝ, f x + f (-x) = 0 := by
  sorry

/-- The second condition: f(x + t) - f(x) < 0 for all x ∈ ℝ and t > 0 -/
theorem condition2 : ∀ x t : ℝ, t > 0 → f (x + t) - f x < 0 := by
  sorry

/-- The main theorem: f satisfies both conditions -/
theorem f_satisfies_conditions : 
  (∀ x : ℝ, f x + f (-x) = 0) ∧ 
  (∀ x t : ℝ, t > 0 → f (x + t) - f x < 0) := by
  sorry

end NUMINAMATH_CALUDE_condition1_condition2_f_satisfies_conditions_l1965_196558


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1965_196519

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h1 : r = 1 / 4)
  (h2 : S = 50)
  (h3 : S = a / (1 - r)) :
  a = 75 / 2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l1965_196519


namespace NUMINAMATH_CALUDE_min_attempts_to_open_safe_l1965_196505

/-- Represents a sequence of 7 digits -/
def Code := Fin 7 → Fin 10

/-- Checks if all digits in a code are different -/
def all_different (c : Code) : Prop :=
  ∀ i j : Fin 7, i ≠ j → c i ≠ c j

/-- Checks if at least one digit in the attempt matches the secret code in the same position -/
def has_match (secret : Code) (attempt : Code) : Prop :=
  ∃ i : Fin 7, secret i = attempt i

/-- Represents a sequence of attempts to open the safe -/
def AttemptSequence (n : ℕ) := Fin n → Code

/-- Checks if a sequence of attempts guarantees opening the safe for any possible secret code -/
def guarantees_opening (attempts : AttemptSequence n) : Prop :=
  ∀ secret : Code, all_different secret →
    ∃ attempt ∈ Set.range attempts, all_different attempt ∧ has_match secret attempt

/-- The main theorem: 6 attempts are sufficient and necessary to guarantee opening the safe -/
theorem min_attempts_to_open_safe :
  (∃ attempts : AttemptSequence 6, guarantees_opening attempts) ∧
  (∀ n < 6, ¬∃ attempts : AttemptSequence n, guarantees_opening attempts) :=
sorry

end NUMINAMATH_CALUDE_min_attempts_to_open_safe_l1965_196505


namespace NUMINAMATH_CALUDE_elliot_reading_rate_l1965_196513

/-- Given a book with a certain number of pages, the number of pages read before a week,
    and the number of pages left after a week of reading, calculate the number of pages read per day. -/
def pages_per_day (total_pages : ℕ) (pages_read_before : ℕ) (pages_left : ℕ) : ℕ :=
  ((total_pages - pages_left) - pages_read_before) / 7

/-- Theorem stating that for Elliot's specific reading scenario, he reads 20 pages per day. -/
theorem elliot_reading_rate : pages_per_day 381 149 92 = 20 := by
  sorry

end NUMINAMATH_CALUDE_elliot_reading_rate_l1965_196513


namespace NUMINAMATH_CALUDE_other_factor_in_product_l1965_196511

theorem other_factor_in_product (w : ℕ+) (x : ℕ+) : 
  w = 156 →
  (∃ k : ℕ+, k * w * x = 2^5 * 3^3 * 13^2) →
  x = 936 := by
sorry

end NUMINAMATH_CALUDE_other_factor_in_product_l1965_196511


namespace NUMINAMATH_CALUDE_albert_more_than_joshua_l1965_196533

/-- The number of rocks collected by Joshua, Jose, and Albert -/
def rock_collection (joshua jose albert : ℕ) : Prop :=
  (jose = joshua - 14) ∧ 
  (albert = jose + 20) ∧ 
  (joshua = 80)

/-- Theorem stating that Albert collected 6 more rocks than Joshua -/
theorem albert_more_than_joshua {joshua jose albert : ℕ} 
  (h : rock_collection joshua jose albert) : albert - joshua = 6 := by
  sorry

end NUMINAMATH_CALUDE_albert_more_than_joshua_l1965_196533


namespace NUMINAMATH_CALUDE_expression_evaluation_l1965_196560

theorem expression_evaluation (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (x^2 + 2*x + 2) / x * (y^2 + 2*y + 2) / y + (x^2 - 3*x + 2) / y * (y^2 - 3*y + 2) / x =
  2*x*y - x/y - y/x + 13 + 10/x + 4/y + 8/(x*y) := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1965_196560


namespace NUMINAMATH_CALUDE_fraction_simplification_l1965_196586

theorem fraction_simplification :
  (1 - 1/3) / (1 - 1/2) = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l1965_196586


namespace NUMINAMATH_CALUDE_shaded_area_is_16pi_l1965_196503

/-- Represents the pattern of semicircles as described in the problem -/
structure SemicirclePattern where
  diameter : ℝ
  length : ℝ

/-- Calculates the area of the shaded region in the semicircle pattern -/
def shaded_area (pattern : SemicirclePattern) : ℝ :=
  sorry

/-- Theorem stating that the shaded area of the given pattern is 16π square inches -/
theorem shaded_area_is_16pi (pattern : SemicirclePattern) 
  (h1 : pattern.diameter = 4)
  (h2 : pattern.length = 18) : 
  shaded_area pattern = 16 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_16pi_l1965_196503


namespace NUMINAMATH_CALUDE_basketball_rim_height_l1965_196534

/-- Represents the height of a basketball rim above the ground -/
def rim_height : ℕ := sorry

/-- Represents the player's height in feet -/
def player_height_feet : ℕ := 6

/-- Represents the player's reach above their head in inches -/
def player_reach : ℕ := 22

/-- Represents the player's jump height in inches -/
def player_jump : ℕ := 32

/-- Represents how far above the rim the player can reach when jumping, in inches -/
def above_rim : ℕ := 6

/-- Conversion factor from feet to inches -/
def feet_to_inches : ℕ := 12

theorem basketball_rim_height : 
  rim_height = player_height_feet * feet_to_inches + player_reach + player_jump - above_rim :=
by sorry

end NUMINAMATH_CALUDE_basketball_rim_height_l1965_196534


namespace NUMINAMATH_CALUDE_larger_cross_section_distance_l1965_196548

/-- Represents a right octagonal pyramid -/
structure RightOctagonalPyramid where
  /-- Height of the pyramid -/
  height : ℝ
  /-- Side length of the base octagon -/
  base_side : ℝ

/-- Represents a cross section of the pyramid -/
structure CrossSection where
  /-- Distance from the apex of the pyramid -/
  distance_from_apex : ℝ
  /-- Area of the cross section -/
  area : ℝ

/-- Main theorem about the distance of the larger cross section from the apex -/
theorem larger_cross_section_distance
  (pyramid : RightOctagonalPyramid)
  (cs1 cs2 : CrossSection)
  (h_areas : cs1.area = 300 * Real.sqrt 2 ∧ cs2.area = 675 * Real.sqrt 2)
  (h_distance : |cs1.distance_from_apex - cs2.distance_from_apex| = 10)
  (h_order : cs1.area < cs2.area) :
  cs2.distance_from_apex = 30 := by
sorry

end NUMINAMATH_CALUDE_larger_cross_section_distance_l1965_196548


namespace NUMINAMATH_CALUDE_min_distance_sum_parabola_to_lines_l1965_196581

/-- The minimum sum of distances from a point on the parabola y^2 = 4x to two lines -/
theorem min_distance_sum_parabola_to_lines : 
  let l₁ := {(x, y) : ℝ × ℝ | 4 * x - 3 * y + 6 = 0}
  let l₂ := {(x, y) : ℝ × ℝ | x = -1}
  let parabola := {(x, y) : ℝ × ℝ | y^2 = 4 * x}
  let dist_to_l₁ (a : ℝ) := |4 * a^2 - 6 * a + 6| / 5
  let dist_to_l₂ (a : ℝ) := |a^2 + 1|
  ∃ (min_dist : ℝ), min_dist = 2 ∧ 
    ∀ (a : ℝ), (dist_to_l₁ a + dist_to_l₂ a) ≥ min_dist :=
by sorry


end NUMINAMATH_CALUDE_min_distance_sum_parabola_to_lines_l1965_196581


namespace NUMINAMATH_CALUDE_binomial_expansion_term_sum_l1965_196577

theorem binomial_expansion_term_sum (n : ℕ) (b : ℝ) : 
  n ≥ 2 → 
  b ≠ 0 → 
  (Nat.choose n 3 : ℝ) * b^(n-3) + (Nat.choose n 4 : ℝ) * b^(n-4) = 0 → 
  n = 4 := by
sorry

end NUMINAMATH_CALUDE_binomial_expansion_term_sum_l1965_196577


namespace NUMINAMATH_CALUDE_circle_ratio_l1965_196525

theorem circle_ratio (R r : ℝ) (h1 : R > 0) (h2 : r > 0) (h3 : R > r) :
  (π * R^2 - π * r^2) = 3 * (π * r^2) → R = 2 * r := by
  sorry

end NUMINAMATH_CALUDE_circle_ratio_l1965_196525


namespace NUMINAMATH_CALUDE_count_propositions_l1965_196514

-- Define a function to check if a statement is a proposition
def isProposition (s : String) : Bool :=
  match s with
  | "|x+2|" => false
  | "-5 ∈ ℤ" => true
  | "π ∉ ℝ" => true
  | "{0} ∈ ℕ" => true
  | _ => false

-- Define the list of statements
def statements : List String := ["|x+2|", "-5 ∈ ℤ", "π ∉ ℝ", "{0} ∈ ℕ"]

-- Theorem to prove
theorem count_propositions :
  (statements.filter isProposition).length = 3 := by sorry

end NUMINAMATH_CALUDE_count_propositions_l1965_196514


namespace NUMINAMATH_CALUDE_tommy_family_size_l1965_196570

/-- The number of family members given the steak requirements -/
def family_members (ounces_per_member : ℕ) (ounces_per_steak : ℕ) (num_steaks : ℕ) : ℕ :=
  (ounces_per_steak * num_steaks) / ounces_per_member

/-- Theorem stating that Tommy's family has 5 members -/
theorem tommy_family_size :
  family_members 16 20 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_tommy_family_size_l1965_196570


namespace NUMINAMATH_CALUDE_quadratic_root_relation_l1965_196502

theorem quadratic_root_relation (a b c : ℝ) (ha : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  (x₁ + x₂ = 2*(x₁ - x₂)) → (3*b^2 = 16*a*c) := by
  sorry

#check quadratic_root_relation

end NUMINAMATH_CALUDE_quadratic_root_relation_l1965_196502


namespace NUMINAMATH_CALUDE_min_k_plus_l_l1965_196588

theorem min_k_plus_l (k l : ℕ+) (h : 120 * k = l ^ 3) : 
  ∀ (k' l' : ℕ+), 120 * k' = l' ^ 3 → k + l ≤ k' + l' :=
by sorry

end NUMINAMATH_CALUDE_min_k_plus_l_l1965_196588


namespace NUMINAMATH_CALUDE_f_properties_l1965_196507

-- Define the function f
noncomputable def f : ℝ → ℝ := fun x =>
  if x < 0 then -x^2 - 4*x - 3
  else if x = 0 then 0
  else x^2 - 4*x + 3

-- State the theorem
theorem f_properties :
  (∀ x, f (-x) = -f x) ∧  -- f is odd
  (∀ x > 0, f x = x^2 - 4*x + 3) →  -- given condition for x > 0
  (f (f (-1)) = 0) ∧  -- part 1
  (∀ x, f x = if x < 0 then -x^2 - 4*x - 3
              else if x = 0 then 0
              else x^2 - 4*x + 3) :=  -- part 2
by sorry

end NUMINAMATH_CALUDE_f_properties_l1965_196507


namespace NUMINAMATH_CALUDE_more_stable_performance_l1965_196584

/-- Given two students A and B with their respective variances, 
    proves that the student with lower variance has more stable performance -/
theorem more_stable_performance (S_A_squared S_B_squared : ℝ) 
  (h1 : S_A_squared = 0.3)
  (h2 : S_B_squared = 0.1) : 
  S_B_squared < S_A_squared := by sorry


end NUMINAMATH_CALUDE_more_stable_performance_l1965_196584


namespace NUMINAMATH_CALUDE_parallelogram_side_ge_altitude_l1965_196556

/-- A parallelogram with side lengths and altitudes. -/
structure Parallelogram where
  side_a : ℝ
  side_b : ℝ
  altitude_a : ℝ
  altitude_b : ℝ
  side_a_pos : 0 < side_a
  side_b_pos : 0 < side_b
  altitude_a_pos : 0 < altitude_a
  altitude_b_pos : 0 < altitude_b

/-- 
Theorem: For any parallelogram, there exists a side length that is 
greater than or equal to the altitude perpendicular to that side.
-/
theorem parallelogram_side_ge_altitude (p : Parallelogram) :
  (p.side_a ≥ p.altitude_a) ∨ (p.side_b ≥ p.altitude_b) := by
  sorry


end NUMINAMATH_CALUDE_parallelogram_side_ge_altitude_l1965_196556


namespace NUMINAMATH_CALUDE_binomial_18_10_l1965_196596

theorem binomial_18_10 (h1 : Nat.choose 16 7 = 11440) (h2 : Nat.choose 16 9 = 11440) :
  Nat.choose 18 10 = 45760 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_10_l1965_196596


namespace NUMINAMATH_CALUDE_simplify_expression_l1965_196543

theorem simplify_expression : 0.4 * 0.5 + 0.3 * 0.2 = 0.26 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1965_196543


namespace NUMINAMATH_CALUDE_sum_first_15_odd_from_5_l1965_196530

/-- The sum of the first n odd positive integers starting from a given odd number -/
def sumOddIntegers (start : ℕ) (n : ℕ) : ℕ :=
  let lastTerm := start + 2 * (n - 1)
  (start + lastTerm) * n / 2

/-- The proposition that the sum of the first 15 odd positive integers starting from 5 is 315 -/
theorem sum_first_15_odd_from_5 : sumOddIntegers 5 15 = 315 := by
  sorry

end NUMINAMATH_CALUDE_sum_first_15_odd_from_5_l1965_196530


namespace NUMINAMATH_CALUDE_opposite_reciprocal_fraction_l1965_196547

theorem opposite_reciprocal_fraction (a b c d : ℝ) 
  (h1 : a + b = 0) -- a and b are opposite numbers
  (h2 : c * d = 1) -- c and d are reciprocals
  : (5*a + 5*b - 7*c*d) / ((-c*d)^3) = 7 := by sorry

end NUMINAMATH_CALUDE_opposite_reciprocal_fraction_l1965_196547


namespace NUMINAMATH_CALUDE_prob_at_least_one_black_without_replacement_prob_exactly_one_black_with_replacement_l1965_196559

/- Define the number of white and black balls -/
def num_white : ℕ := 4
def num_black : ℕ := 2
def total_balls : ℕ := num_white + num_black

/- Define the number of draws -/
def num_draws : ℕ := 3

/- Theorem for drawing without replacement -/
theorem prob_at_least_one_black_without_replacement :
  let total_ways := Nat.choose total_balls num_draws
  let all_white_ways := Nat.choose num_white num_draws
  (1 : ℚ) - (all_white_ways : ℚ) / (total_ways : ℚ) = 4/5 := by sorry

/- Theorem for drawing with replacement -/
theorem prob_exactly_one_black_with_replacement :
  let total_ways := total_balls ^ num_draws
  let one_black_ways := num_draws * num_black * (num_white ^ (num_draws - 1))
  (one_black_ways : ℚ) / (total_ways : ℚ) = 4/9 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_black_without_replacement_prob_exactly_one_black_with_replacement_l1965_196559


namespace NUMINAMATH_CALUDE_jamies_shoes_cost_l1965_196540

/-- The cost of Jamie's shoes given the total cost and James' items -/
theorem jamies_shoes_cost (total_cost : ℕ) (coat_cost : ℕ) (jeans_cost : ℕ) : 
  total_cost = 110 →
  coat_cost = 40 →
  jeans_cost = 20 →
  total_cost = coat_cost + 2 * jeans_cost + (total_cost - (coat_cost + 2 * jeans_cost)) →
  (total_cost - (coat_cost + 2 * jeans_cost)) = 30 := by
sorry

end NUMINAMATH_CALUDE_jamies_shoes_cost_l1965_196540


namespace NUMINAMATH_CALUDE_claire_gift_card_value_l1965_196522

/-- The value of Claire's gift card -/
def gift_card_value : ℚ := 100

/-- Cost of a latte -/
def latte_cost : ℚ := 3.75

/-- Cost of a croissant -/
def croissant_cost : ℚ := 3.50

/-- Cost of a cookie -/
def cookie_cost : ℚ := 1.25

/-- Number of days Claire buys coffee and pastry -/
def days : ℕ := 7

/-- Number of cookies Claire buys -/
def num_cookies : ℕ := 5

/-- Amount left on the gift card after spending -/
def amount_left : ℚ := 43

/-- Theorem stating the value of Claire's gift card -/
theorem claire_gift_card_value :
  gift_card_value = 
    (latte_cost + croissant_cost) * days + 
    cookie_cost * num_cookies + 
    amount_left :=
by sorry

end NUMINAMATH_CALUDE_claire_gift_card_value_l1965_196522


namespace NUMINAMATH_CALUDE_linear_function_proof_l1965_196597

/-- A linear function passing through two points -/
def LinearFunction (k b : ℝ) : ℝ → ℝ := fun x ↦ k * x + b

/-- The linear function passes through the point (3,1) -/
def PassesThrough3_1 (k b : ℝ) : Prop := LinearFunction k b 3 = 1

/-- The linear function passes through the point (2,0) -/
def PassesThrough2_0 (k b : ℝ) : Prop := LinearFunction k b 2 = 0

theorem linear_function_proof (k b : ℝ) 
  (h1 : PassesThrough3_1 k b) (h2 : PassesThrough2_0 k b) :
  (∀ x, LinearFunction k b x = x - 2) ∧ (LinearFunction k b 6 = 4) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_proof_l1965_196597


namespace NUMINAMATH_CALUDE_sinusoidal_period_l1965_196512

/-- 
Given a sinusoidal function y = a * sin(b * x + c) + d where a, b, c, and d are positive constants,
if the function completes five periods over an interval of 2π, then b = 5.
-/
theorem sinusoidal_period (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) 
  (h_periods : (2 * Real.pi) / b = (2 * Real.pi) / 5) : b = 5 := by
  sorry

end NUMINAMATH_CALUDE_sinusoidal_period_l1965_196512


namespace NUMINAMATH_CALUDE_social_logistics_turnover_scientific_notation_l1965_196528

/-- Given that one trillion is 10^12, prove that 347.6 trillion yuan is equal to 3.476 × 10^14 yuan -/
theorem social_logistics_turnover_scientific_notation :
  let trillion : ℝ := 10^12
  347.6 * trillion = 3.476 * 10^14 := by
  sorry

end NUMINAMATH_CALUDE_social_logistics_turnover_scientific_notation_l1965_196528


namespace NUMINAMATH_CALUDE_focus_directrix_distance_l1965_196529

/-- The parabola equation y^2 = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- The focus of the parabola y^2 = 4x -/
def focus : ℝ × ℝ := (1, 0)

/-- The directrix of the parabola y^2 = 4x -/
def directrix (x : ℝ) : Prop := x = -1

/-- The distance from the focus to the directrix of the parabola y^2 = 4x is 2 -/
theorem focus_directrix_distance : 
  ∃ (d : ℝ), d = 2 ∧ d = |focus.1 - (-1)| :=
sorry

end NUMINAMATH_CALUDE_focus_directrix_distance_l1965_196529


namespace NUMINAMATH_CALUDE_journey_problem_l1965_196598

theorem journey_problem (total_distance : ℝ) (days : ℕ) (ratio : ℝ) 
  (h1 : total_distance = 378)
  (h2 : days = 6)
  (h3 : ratio = 1/2) :
  let first_day := total_distance * (1 - ratio) / (1 - ratio^days)
  first_day * ratio = 96 := by
  sorry

end NUMINAMATH_CALUDE_journey_problem_l1965_196598


namespace NUMINAMATH_CALUDE_green_hats_count_l1965_196574

theorem green_hats_count (total_hats : ℕ) (blue_cost green_cost total_cost : ℕ) 
  (h1 : total_hats = 85)
  (h2 : blue_cost = 6)
  (h3 : green_cost = 7)
  (h4 : total_cost = 550) :
  ∃ (blue_hats green_hats : ℕ),
    blue_hats + green_hats = total_hats ∧
    blue_cost * blue_hats + green_cost * green_hats = total_cost ∧
    green_hats = 40 :=
by sorry

end NUMINAMATH_CALUDE_green_hats_count_l1965_196574


namespace NUMINAMATH_CALUDE_probability_exactly_one_instrument_l1965_196552

/-- The probability of playing exactly one instrument in a group -/
theorem probability_exactly_one_instrument 
  (total_people : ℕ) 
  (at_least_one_fraction : ℚ) 
  (two_or_more : ℕ) 
  (h1 : total_people = 800) 
  (h2 : at_least_one_fraction = 1 / 5) 
  (h3 : two_or_more = 64) : 
  (↑((at_least_one_fraction * ↑total_people).num - two_or_more) / ↑total_people : ℚ) = 3 / 25 := by
  sorry

end NUMINAMATH_CALUDE_probability_exactly_one_instrument_l1965_196552


namespace NUMINAMATH_CALUDE_complex_sum_real_necessary_not_sufficient_l1965_196550

theorem complex_sum_real_necessary_not_sufficient (z₁ z₂ : ℂ) :
  (∃ (a b : ℝ), z₁ = a + b * I ∧ z₂ = a - b * I) → (z₁ + z₂).im = 0 ∧
  ¬(∀ z₁ z₂ : ℂ, (z₁ + z₂).im = 0 → ∃ (a b : ℝ), z₁ = a + b * I ∧ z₂ = a - b * I) :=
by sorry

end NUMINAMATH_CALUDE_complex_sum_real_necessary_not_sufficient_l1965_196550


namespace NUMINAMATH_CALUDE_inductive_reasoning_not_comparison_l1965_196506

/-- Represents different types of reasoning --/
inductive ReasoningType
| Deductive
| Inductive
| Analogical
| Plausibility

/-- Represents the process of reasoning --/
structure Reasoning where
  type : ReasoningType
  process : String
  conclusion_certainty : Bool

/-- Definition of deductive reasoning --/
def deductive_reasoning : Reasoning :=
  { type := ReasoningType.Deductive,
    process := "from general to specific",
    conclusion_certainty := true }

/-- Definition of inductive reasoning --/
def inductive_reasoning : Reasoning :=
  { type := ReasoningType.Inductive,
    process := "from specific to general",
    conclusion_certainty := false }

/-- Definition of analogical reasoning --/
def analogical_reasoning : Reasoning :=
  { type := ReasoningType.Analogical,
    process := "comparing characteristics of different things",
    conclusion_certainty := false }

/-- Theorem stating that inductive reasoning is not about comparing characteristics of two types of things --/
theorem inductive_reasoning_not_comparison : 
  inductive_reasoning.process ≠ "reasoning between the characteristics of two types of things" := by
  sorry


end NUMINAMATH_CALUDE_inductive_reasoning_not_comparison_l1965_196506


namespace NUMINAMATH_CALUDE_cos_two_x_value_l1965_196537

theorem cos_two_x_value (x : ℝ) (h : Real.sin (-x) = Real.sqrt 3 / 2) : 
  Real.cos (2 * x) = -(1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_cos_two_x_value_l1965_196537


namespace NUMINAMATH_CALUDE_snack_eaters_final_count_l1965_196593

/-- Calculates the final number of snack eaters after a series of events --/
def finalSnackEaters (initialGathering : ℕ) (initialSnackEaters : ℕ)
  (firstNewGroup : ℕ) (secondNewGroup : ℕ) (thirdLeaving : ℕ) : ℕ :=
  let afterFirst := initialSnackEaters + firstNewGroup
  let afterHalfLeft := afterFirst / 2
  let afterSecondNew := afterHalfLeft + secondNewGroup
  let afterThirdLeft := afterSecondNew - thirdLeaving
  afterThirdLeft / 2

/-- Theorem stating that given the initial conditions and sequence of events,
    the final number of snack eaters is 20 --/
theorem snack_eaters_final_count :
  finalSnackEaters 200 100 20 10 30 = 20 := by
  sorry

#eval finalSnackEaters 200 100 20 10 30

end NUMINAMATH_CALUDE_snack_eaters_final_count_l1965_196593


namespace NUMINAMATH_CALUDE_arithmetic_mean_equidistant_l1965_196599

/-- The arithmetic mean of two real numbers is equidistant from both numbers. -/
theorem arithmetic_mean_equidistant (a b : ℝ) : 
  |((a + b) / 2) - a| = |b - ((a + b) / 2)| := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_equidistant_l1965_196599


namespace NUMINAMATH_CALUDE_cube_sum_reciprocal_l1965_196580

theorem cube_sum_reciprocal (x : ℝ) (h : x + 1/x = -3) : x^3 + 1/x^3 = -18 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_reciprocal_l1965_196580


namespace NUMINAMATH_CALUDE_expression_simplification_l1965_196544

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2) :
  (x - 1) / x / (x - 1 / x) = Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1965_196544


namespace NUMINAMATH_CALUDE_last_two_digits_of_seven_power_l1965_196565

theorem last_two_digits_of_seven_power (n : ℕ) : 7^(5^6) ≡ 7 [MOD 100] := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_seven_power_l1965_196565


namespace NUMINAMATH_CALUDE_triangle_max_area_l1965_196590

-- Define a structure for a triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions and theorem
theorem triangle_max_area (t : Triangle) 
  (h1 : Real.sin t.A + Real.sqrt 2 * Real.sin t.B = 2 * Real.sin t.C)
  (h2 : t.b = 3) :
  ∃ (max_area : ℝ), max_area = (9 + 3 * Real.sqrt 3) / 4 ∧ 
    ∀ (area : ℝ), area ≤ max_area := by
  sorry

end NUMINAMATH_CALUDE_triangle_max_area_l1965_196590


namespace NUMINAMATH_CALUDE_sum_prime_factors_2_pow_22_minus_4_l1965_196526

/-- SPF(n) denotes the sum of the prime factors of n, where the prime factors are not necessarily distinct. -/
def SPF (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of prime factors of 2^22 - 4 is 100. -/
theorem sum_prime_factors_2_pow_22_minus_4 : SPF (2^22 - 4) = 100 := by sorry

end NUMINAMATH_CALUDE_sum_prime_factors_2_pow_22_minus_4_l1965_196526


namespace NUMINAMATH_CALUDE_smallest_integer_in_consecutive_set_l1965_196500

theorem smallest_integer_in_consecutive_set (n : ℤ) : 
  (n + 6 < 3 * ((n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6)) / 7)) →
  n = 0 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_in_consecutive_set_l1965_196500


namespace NUMINAMATH_CALUDE_drama_club_subject_distribution_l1965_196557

theorem drama_club_subject_distribution (total : ℕ) (math physics chem : ℕ) 
  (math_physics math_chem physics_chem : ℕ) (all_three : ℕ) :
  total = 100 ∧ 
  math = 50 ∧ 
  physics = 40 ∧ 
  chem = 30 ∧ 
  math_physics = 20 ∧ 
  physics_chem = 10 ∧ 
  all_three = 5 →
  total - (math + physics + chem - math_physics - physics_chem - math_chem + all_three) = 20 :=
by sorry

end NUMINAMATH_CALUDE_drama_club_subject_distribution_l1965_196557


namespace NUMINAMATH_CALUDE_figure_50_squares_l1965_196555

def square_count (n : ℕ) : ℕ := 2 * n^2 + 3 * n + 1

theorem figure_50_squares :
  square_count 0 = 1 ∧
  square_count 1 = 6 ∧
  square_count 2 = 15 ∧
  square_count 3 = 28 →
  square_count 50 = 5151 := by
  sorry

end NUMINAMATH_CALUDE_figure_50_squares_l1965_196555


namespace NUMINAMATH_CALUDE_maurice_rides_before_is_10_l1965_196549

/-- The number of times Maurice had been horseback riding before visiting Matt -/
def maurice_rides_before : ℕ := 10

/-- The number of different horses Maurice rode before his visit -/
def maurice_horses_before : ℕ := 2

/-- The number of different horses Matt has ridden -/
def matt_horses : ℕ := 4

/-- The number of times Maurice rode during his visit -/
def maurice_rides_visit : ℕ := 8

/-- The number of additional times Matt rode on his other horses -/
def matt_additional_rides : ℕ := 16

/-- The number of horses Matt rode each time with Maurice -/
def matt_horses_per_ride : ℕ := 2

theorem maurice_rides_before_is_10 :
  maurice_rides_before = 10 ∧
  maurice_horses_before = 2 ∧
  matt_horses = 4 ∧
  maurice_rides_visit = 8 ∧
  matt_additional_rides = 16 ∧
  matt_horses_per_ride = 2 ∧
  maurice_rides_visit = maurice_rides_before ∧
  (maurice_rides_visit * matt_horses_per_ride + matt_additional_rides) = 3 * maurice_rides_before :=
by sorry

end NUMINAMATH_CALUDE_maurice_rides_before_is_10_l1965_196549


namespace NUMINAMATH_CALUDE_second_year_sample_size_l1965_196545

/-- Represents the distribution of students across four years -/
structure StudentDistribution where
  first : ℕ
  second : ℕ
  third : ℕ
  fourth : ℕ

/-- Calculates the total number of students -/
def total_students (d : StudentDistribution) : ℕ :=
  d.first + d.second + d.third + d.fourth

/-- Calculates the number of students to sample from a specific year -/
def sample_size_for_year (total_population : ℕ) (year_population : ℕ) (sample_size : ℕ) : ℕ :=
  (year_population * sample_size) / total_population

theorem second_year_sample_size 
  (total_population : ℕ) 
  (distribution : StudentDistribution) 
  (sample_size : ℕ) :
  total_population = 5000 →
  distribution = { first := 5, second := 4, third := 3, fourth := 1 } →
  sample_size = 260 →
  sample_size_for_year total_population distribution.second sample_size = 80 := by
  sorry

#check second_year_sample_size

end NUMINAMATH_CALUDE_second_year_sample_size_l1965_196545


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l1965_196542

theorem meaningful_expression_range (x : ℝ) : 
  (∃ y : ℝ, y = (Real.sqrt (x + 1)) / (x - 2)) ↔ (x ≥ -1 ∧ x ≠ 2) :=
by sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l1965_196542


namespace NUMINAMATH_CALUDE_chessboard_impossible_l1965_196568

/-- Represents a 6x6 chessboard filled with numbers -/
def Chessboard := Fin 6 → Fin 6 → Fin 36

/-- The sum of numbers from 1 to 36 -/
def total_sum : Nat := (36 * 37) / 2

/-- The required sum for each row, column, and diagonal -/
def required_sum : Nat := total_sum / 6

/-- Checks if a number appears exactly once on the chessboard -/
def appears_once (board : Chessboard) (n : Fin 36) : Prop :=
  ∃! (i j : Fin 6), board i j = n

/-- Checks if a row has the required sum -/
def row_sum_correct (board : Chessboard) (i : Fin 6) : Prop :=
  (Finset.sum (Finset.univ : Finset (Fin 6)) fun j => (board i j).val + 1) = required_sum

/-- Checks if a column has the required sum -/
def col_sum_correct (board : Chessboard) (j : Fin 6) : Prop :=
  (Finset.sum (Finset.univ : Finset (Fin 6)) fun i => (board i j).val + 1) = required_sum

/-- Checks if a northeast diagonal has the required sum -/
def diag_sum_correct (board : Chessboard) (k : Fin 6) : Prop :=
  (Finset.sum (Finset.univ : Finset (Fin 6)) fun i =>
    (board i ((i.val - k.val + 6) % 6 : Fin 6)).val + 1) = required_sum

/-- The main theorem stating that it's impossible to fill the chessboard with the given conditions -/
theorem chessboard_impossible : ¬∃ (board : Chessboard),
  (∀ n : Fin 36, appears_once board n) ∧
  (∀ i : Fin 6, row_sum_correct board i) ∧
  (∀ j : Fin 6, col_sum_correct board j) ∧
  (∀ k : Fin 6, diag_sum_correct board k) :=
sorry

end NUMINAMATH_CALUDE_chessboard_impossible_l1965_196568


namespace NUMINAMATH_CALUDE_intersection_point_l1965_196516

/-- The quadratic function f(x) = x^2 - 4x + 4 -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 4

/-- Theorem: The point (2,0) is the only intersection point of y = x^2 - 4x + 4 with the x-axis -/
theorem intersection_point : 
  (∃! x : ℝ, f x = 0) ∧ (f 2 = 0) := by sorry

end NUMINAMATH_CALUDE_intersection_point_l1965_196516


namespace NUMINAMATH_CALUDE_f_intersects_x_axis_l1965_196554

-- Define the function f(x) = x + 1
def f (x : ℝ) : ℝ := x + 1

-- Theorem stating that f intersects the x-axis
theorem f_intersects_x_axis : ∃ x : ℝ, f x = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_intersects_x_axis_l1965_196554


namespace NUMINAMATH_CALUDE_special_function_properties_l1965_196576

def I : Set ℝ := Set.Icc (-1) 1

structure SpecialFunction (f : ℝ → ℝ) : Prop where
  domain : ∀ x, x ∈ I → f x ≠ 0 → True
  additive : ∀ x y, x ∈ I → y ∈ I → f (x + y) = f x + f y
  positive : ∀ x, x > 0 → x ∈ I → f x > 0

theorem special_function_properties (f : ℝ → ℝ) (h : SpecialFunction f) :
  (∀ x, x ∈ I → f (-x) = -f x) ∧
  (∀ x y, x ∈ I → y ∈ I → x < y → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_special_function_properties_l1965_196576


namespace NUMINAMATH_CALUDE_stamps_theorem_l1965_196591

/-- Given denominations 3, n, and n+1, this function checks if k cents can be formed -/
def can_form (n : ℕ) (k : ℕ) : Prop :=
  ∃ (a b c : ℕ), k = 3 * a + n * b + (n + 1) * c

/-- The main theorem -/
theorem stamps_theorem :
  ∃! (n : ℕ), 
    n > 0 ∧ 
    (∀ (k : ℕ), k ≤ 115 → ¬(can_form n k)) ∧
    (∀ (k : ℕ), k > 115 → can_form n k) ∧
    n = 59 := by
  sorry

end NUMINAMATH_CALUDE_stamps_theorem_l1965_196591


namespace NUMINAMATH_CALUDE_exponential_equation_solution_l1965_196562

theorem exponential_equation_solution :
  ∃ x : ℝ, (64 : ℝ) ^ (3 * x) = (16 : ℝ) ^ (4 * x - 5) ↔ x = -10 := by
  sorry

end NUMINAMATH_CALUDE_exponential_equation_solution_l1965_196562


namespace NUMINAMATH_CALUDE_liquid_film_radius_l1965_196501

/-- The radius of a circular film formed by a liquid --/
theorem liquid_film_radius (tank_length tank_width tank_height film_thickness : ℝ)
  (tank_length_pos : 0 < tank_length)
  (tank_width_pos : 0 < tank_width)
  (tank_height_pos : 0 < tank_height)
  (film_thickness_pos : 0 < film_thickness)
  (h_length : tank_length = 8)
  (h_width : tank_width = 4)
  (h_height : tank_height = 10)
  (h_thickness : film_thickness = 0.2) :
  ∃ (r : ℝ), r > 0 ∧ r^2 * π * film_thickness = tank_length * tank_width * tank_height ∧
    r = Real.sqrt (1600 / π) :=
by sorry

end NUMINAMATH_CALUDE_liquid_film_radius_l1965_196501


namespace NUMINAMATH_CALUDE_central_angle_from_arc_length_l1965_196566

/-- Given a circle with radius 2 and arc length 4, prove that the central angle is 2 radians -/
theorem central_angle_from_arc_length (r : ℝ) (l : ℝ) (h1 : r = 2) (h2 : l = 4) :
  l / r = 2 := by
  sorry

end NUMINAMATH_CALUDE_central_angle_from_arc_length_l1965_196566


namespace NUMINAMATH_CALUDE_rational_function_with_infinite_integer_values_is_polynomial_l1965_196579

/-- A rational function is a quotient of two real polynomials -/
def RationalFunction (f : ℝ → ℝ) : Prop :=
  ∃ p q : Polynomial ℝ, q ≠ 0 ∧ ∀ x, f x = (p.eval x) / (q.eval x)

/-- A function that takes integer values at infinitely many integer points -/
def IntegerValuesAtInfinitelyManyPoints (f : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, ∃ m > n, ∃ k : ℤ, f k = m

/-- Main theorem: If f is a rational function and takes integer values at infinitely many
    integer points, then f is a polynomial -/
theorem rational_function_with_infinite_integer_values_is_polynomial
  (f : ℝ → ℝ) (hf : RationalFunction f) (hi : IntegerValuesAtInfinitelyManyPoints f) :
  ∃ p : Polynomial ℝ, ∀ x, f x = p.eval x :=
sorry

end NUMINAMATH_CALUDE_rational_function_with_infinite_integer_values_is_polynomial_l1965_196579


namespace NUMINAMATH_CALUDE_greatest_x_given_lcm_l1965_196592

theorem greatest_x_given_lcm (x : ℕ+) : 
  (Nat.lcm x (Nat.lcm 12 18) = 180) → x ≤ 180 ∧ ∃ y : ℕ+, y > 180 → Nat.lcm y (Nat.lcm 12 18) > 180 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_given_lcm_l1965_196592


namespace NUMINAMATH_CALUDE_number_problem_l1965_196575

theorem number_problem (x : ℝ) : 0.4 * x - 30 = 50 → x = 200 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1965_196575


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1965_196523

/-- The line ax+by-2=0 passes through the point (4,2) for all a and b that satisfy 2a+b=1 -/
theorem line_passes_through_fixed_point (a b : ℝ) (h : 2*a + b = 1) :
  a*4 + b*2 - 2 = 0 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l1965_196523


namespace NUMINAMATH_CALUDE_square_difference_given_system_l1965_196553

theorem square_difference_given_system (x y : ℝ) 
  (eq1 : 3 * x + 2 * y = 20) 
  (eq2 : 4 * x + 3 * y = 29) : 
  x^2 - y^2 = -45 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_given_system_l1965_196553


namespace NUMINAMATH_CALUDE_class_overlap_difference_l1965_196539

theorem class_overlap_difference (total : ℕ) (geometry : ℕ) (biology : ℕ)
  (h1 : total = 232)
  (h2 : geometry = 144)
  (h3 : biology = 119)
  (h4 : geometry ≤ total)
  (h5 : biology ≤ total) :
  min geometry biology - max 0 (geometry + biology - total) = 88 :=
by sorry

end NUMINAMATH_CALUDE_class_overlap_difference_l1965_196539


namespace NUMINAMATH_CALUDE_variance_binomial_4_half_l1965_196517

/-- The variance of a binomial distribution with n trials and probability p -/
def binomialVariance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

/-- Theorem: The variance of a binomial distribution B(4, 1/2) is 1 -/
theorem variance_binomial_4_half :
  binomialVariance 4 (1/2 : ℝ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_variance_binomial_4_half_l1965_196517


namespace NUMINAMATH_CALUDE_worker_wage_increase_l1965_196518

theorem worker_wage_increase (original_wage : ℝ) : 
  (original_wage * 1.5 = 42) → original_wage = 28 := by
  sorry

end NUMINAMATH_CALUDE_worker_wage_increase_l1965_196518


namespace NUMINAMATH_CALUDE_bowtie_equation_solution_l1965_196573

-- Define the ⋈ operation
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ h : ℝ, bowtie 5 h = 11 ∧ h = 30 :=
sorry

end NUMINAMATH_CALUDE_bowtie_equation_solution_l1965_196573


namespace NUMINAMATH_CALUDE_cube_root_inequality_l1965_196536

theorem cube_root_inequality (x : ℝ) : 
  x > 0 → (x^(1/3) < 3*x ↔ x > 1/(3*Real.sqrt 3)) := by sorry

end NUMINAMATH_CALUDE_cube_root_inequality_l1965_196536


namespace NUMINAMATH_CALUDE_f_at_negative_one_l1965_196546

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x^2 + 1

-- Theorem statement
theorem f_at_negative_one : f (-1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_f_at_negative_one_l1965_196546


namespace NUMINAMATH_CALUDE_gain_percent_when_cost_equals_sell_l1965_196582

/-- Proves that if the cost price of 50 articles equals the selling price of 25 articles, 
    then the gain percent is 100%. -/
theorem gain_percent_when_cost_equals_sell (C S : ℝ) 
  (h : 50 * C = 25 * S) : (S - C) / C * 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_when_cost_equals_sell_l1965_196582


namespace NUMINAMATH_CALUDE_minimum_selling_price_chocolate_manufacturer_l1965_196541

/-- Calculates the minimum selling price per unit to achieve a desired monthly profit -/
def minimum_selling_price (units : ℕ) (cost_per_unit : ℚ) (desired_profit : ℚ) : ℚ :=
  (units * cost_per_unit + desired_profit) / units

theorem minimum_selling_price_chocolate_manufacturer :
  let units : ℕ := 400
  let cost_per_unit : ℚ := 40
  let desired_profit : ℚ := 40000
  minimum_selling_price units cost_per_unit desired_profit = 140 := by
  sorry

end NUMINAMATH_CALUDE_minimum_selling_price_chocolate_manufacturer_l1965_196541


namespace NUMINAMATH_CALUDE_f_properties_l1965_196571

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def f_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≠ Real.pi * ↑(Int.floor (x / Real.pi)) →
         y ≠ Real.pi * ↑(Int.floor (y / Real.pi)) →
         f (x - y) = (f x * f y + 1) / (f y - f x)

theorem f_properties (f : ℝ → ℝ) 
  (h_eq : f_equation f)
  (h_f1 : f 1 = 1)
  (h_pos : ∀ x, 0 < x → x < 2 → f x > 0) :
  is_odd f ∧ 
  f 2 = 0 ∧ 
  f 3 = -1 ∧
  (∀ x, 2 ≤ x → x ≤ 3 → f x ≤ 0) ∧
  (∀ x, 2 ≤ x → x ≤ 3 → f x ≥ -1) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l1965_196571


namespace NUMINAMATH_CALUDE_twin_prime_divisibility_l1965_196595

theorem twin_prime_divisibility (p q : ℕ) : 
  Prime p → Prime q → q = p + 2 → (p + q) ∣ (p^q + q^p) := by
sorry

end NUMINAMATH_CALUDE_twin_prime_divisibility_l1965_196595


namespace NUMINAMATH_CALUDE_number_division_problem_l1965_196521

theorem number_division_problem (x : ℝ) : 
  (x - 6) / 8 = 6 → (x - 5) / 7 = 7 := by
sorry

end NUMINAMATH_CALUDE_number_division_problem_l1965_196521


namespace NUMINAMATH_CALUDE_tan_15_identity_l1965_196564

theorem tan_15_identity : (1 + Real.tan (15 * π / 180)) / (1 - Real.tan (15 * π / 180)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_15_identity_l1965_196564


namespace NUMINAMATH_CALUDE_smallest_m_for_divisibility_l1965_196510

theorem smallest_m_for_divisibility : 
  ∃ (n : ℕ), 
    n % 2 = 1 ∧ 
    (55^n + 436 * 32^n) % 2001 = 0 ∧ 
    ∀ (m : ℕ), m < 436 → 
      ∀ (k : ℕ), k % 2 = 1 → (55^k + m * 32^k) % 2001 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_divisibility_l1965_196510


namespace NUMINAMATH_CALUDE_final_sum_after_transformation_l1965_196515

theorem final_sum_after_transformation (x y S : ℝ) (h : x + y = S) :
  3 * (x + 4) + 3 * (y + 4) = 3 * S + 24 := by
  sorry

end NUMINAMATH_CALUDE_final_sum_after_transformation_l1965_196515


namespace NUMINAMATH_CALUDE_f_sum_theorem_l1965_196587

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_sum_theorem (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_period : ∀ x, f (x + 3) = -f x)
  (h_f1 : f 1 = -1) : 
  f 5 + f 13 = -2 := by
sorry

end NUMINAMATH_CALUDE_f_sum_theorem_l1965_196587


namespace NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1965_196535

/-- An isosceles triangle with congruent sides of length 8 cm and perimeter of 26 cm has a base of length 10 cm. -/
theorem isosceles_triangle_base_length :
  ∀ (base congruent_side perimeter : ℝ),
  congruent_side = 8 →
  perimeter = 26 →
  perimeter = 2 * congruent_side + base →
  base = 10 := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_base_length_l1965_196535
