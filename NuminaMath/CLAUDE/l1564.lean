import Mathlib

namespace min_value_fraction_l1564_156418

theorem min_value_fraction (x : ℝ) (h : x > 2) : (x^2 - 4*x + 5) / (x - 2) ≥ 2 := by
  sorry

end min_value_fraction_l1564_156418


namespace geometric_sequence_min_value_l1564_156411

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n, a (n + 1) = a n * q

theorem geometric_sequence_min_value
  (a : ℕ → ℝ) (q : ℝ) (h_pos : ∀ n, a n > 0) (h_geom : geometric_sequence a q)
  (h_relation : a 7 = a 6 + 2 * a 5)
  (h_exist : ∃ m n, Real.sqrt (a m * a n) = 4 * a 1) :
  (∀ m n, Real.sqrt (a m * a n) = 4 * a 1 → 1 / m + 5 / n ≥ 7 / 4) ∧
  (∃ m n, Real.sqrt (a m * a n) = 4 * a 1 ∧ 1 / m + 5 / n = 7 / 4) :=
by sorry

end geometric_sequence_min_value_l1564_156411


namespace tight_sequence_x_range_arithmetic_sequence_is_tight_geometric_sequence_tight_condition_l1564_156445

def is_tight_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → (1/2 : ℝ) ≤ a (n+1) / a n ∧ a (n+1) / a n ≤ 2

theorem tight_sequence_x_range (a : ℕ → ℝ) (h : is_tight_sequence a)
  (h1 : a 1 = 1) (h2 : a 2 = 3/2) (h3 : a 4 = 4) :
  2 ≤ a 3 ∧ a 3 ≤ 3 := by sorry

theorem arithmetic_sequence_is_tight (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 1 > 0) (h2 : 0 < d) (h3 : d ≤ a 1)
  (h4 : ∀ n : ℕ, n > 0 → a (n+1) = a n + d) :
  is_tight_sequence a := by sorry

def partial_sum (a : ℕ → ℝ) : ℕ → ℝ
| 0 => 0
| n+1 => partial_sum a n + a (n+1)

theorem geometric_sequence_tight_condition (a : ℕ → ℝ) (q : ℝ)
  (h : ∀ n : ℕ, n > 0 → a (n+1) = q * a n) :
  (is_tight_sequence a ∧ is_tight_sequence (partial_sum a)) ↔ 1/2 ≤ q ∧ q ≤ 1 := by sorry

end tight_sequence_x_range_arithmetic_sequence_is_tight_geometric_sequence_tight_condition_l1564_156445


namespace employed_males_percentage_l1564_156442

theorem employed_males_percentage (population : ℝ) 
  (h1 : population > 0) 
  (employed_percentage : ℝ) 
  (h2 : employed_percentage = 0.64) 
  (employed_females_percentage : ℝ) 
  (h3 : employed_females_percentage = 0.140625) : 
  (employed_percentage * (1 - employed_females_percentage)) * population / population = 0.5496 := by
sorry

end employed_males_percentage_l1564_156442


namespace complex_magnitude_l1564_156448

theorem complex_magnitude (z : ℂ) (h : z * (1 + Complex.I) = Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by sorry

end complex_magnitude_l1564_156448


namespace tank_water_volume_l1564_156471

/-- Calculates the final volume of water in a tank after evaporation, draining, and rainfall. -/
theorem tank_water_volume 
  (initial_volume : ℕ) 
  (evaporated_volume : ℕ) 
  (drained_volume : ℕ) 
  (rain_duration : ℕ) 
  (rain_rate : ℕ) 
  (rain_interval : ℕ) 
  (h1 : initial_volume = 6000)
  (h2 : evaporated_volume = 2000)
  (h3 : drained_volume = 3500)
  (h4 : rain_duration = 30)
  (h5 : rain_rate = 350)
  (h6 : rain_interval = 10) :
  initial_volume - evaporated_volume - drained_volume + 
  (rain_duration / rain_interval) * rain_rate = 1550 :=
by
  sorry

#check tank_water_volume

end tank_water_volume_l1564_156471


namespace magazine_boxes_l1564_156461

theorem magazine_boxes (total_magazines : ℕ) (magazines_per_box : ℚ) : 
  total_magazines = 150 → magazines_per_box = 11.5 → 
  ⌈(total_magazines : ℚ) / magazines_per_box⌉ = 14 := by
  sorry

end magazine_boxes_l1564_156461


namespace product_of_three_numbers_l1564_156419

theorem product_of_three_numbers (x y z : ℝ) 
  (sum_eq : x + y + z = 30)
  (first_eq : x = 3 * (y + z))
  (second_eq : y = 5 * z) :
  x * y * z = 175.78125 := by
  sorry

end product_of_three_numbers_l1564_156419


namespace age_difference_l1564_156470

/-- Given the ages of Mehki, Jordyn, and Zrinka, prove that Mehki is 10 years older than Jordyn. -/
theorem age_difference (mehki_age jordyn_age zrinka_age : ℕ) 
  (h1 : jordyn_age = 2 * zrinka_age)
  (h2 : zrinka_age = 6)
  (h3 : mehki_age = 22) :
  mehki_age - jordyn_age = 10 := by
sorry

end age_difference_l1564_156470


namespace min_value_theorem_l1564_156405

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : ∀ x : ℝ, 1 ≤ x → x ≤ 4 → a * x + b - 3 ≤ 0) : 
  1 / a - b ≥ 1 := by
sorry

end min_value_theorem_l1564_156405


namespace composite_sum_of_squares_l1564_156439

theorem composite_sum_of_squares (a b : ℤ) : 
  (∃ x₁ x₂ : ℤ, x₁ ≠ 0 ∧ x₂ ≠ 0 ∧ 
   x₁^2 + a*x₁ + 1 = b ∧ x₂^2 + a*x₂ + 1 = b) →
  ∃ m n : ℤ, m > 1 ∧ n > 1 ∧ a^2 + b^2 = m * n :=
by sorry

end composite_sum_of_squares_l1564_156439


namespace tangent_to_exponential_l1564_156481

theorem tangent_to_exponential (k : ℝ) :
  (∃ x : ℝ, k * x = Real.exp x ∧ k = Real.exp x) → k = Real.exp 1 := by
sorry

end tangent_to_exponential_l1564_156481


namespace unique_prime_product_l1564_156449

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem unique_prime_product :
  ∀ n : ℕ,
  n ≠ 2103 →
  (∃ p q r : ℕ, is_prime p ∧ is_prime q ∧ is_prime r ∧ distinct p q r ∧ n = p * q * r) →
  ¬(∃ p1 p2 p3 : ℕ, is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ distinct p1 p2 p3 ∧ p1 + p2 + p3 = 59) :=
by sorry

end unique_prime_product_l1564_156449


namespace percentage_men_correct_l1564_156453

/-- The percentage of men in a college class. -/
def percentage_men : ℝ := 40

theorem percentage_men_correct :
  let women_science_percentage : ℝ := 30
  let non_science_percentage : ℝ := 60
  let men_science_percentage : ℝ := 55.00000000000001
  let women_percentage : ℝ := 100 - percentage_men
  let science_percentage : ℝ := 100 - non_science_percentage
  (women_science_percentage / 100 * women_percentage + 
   men_science_percentage / 100 * percentage_men = science_percentage) ∧
  (percentage_men ≥ 0 ∧ percentage_men ≤ 100) :=
by sorry

end percentage_men_correct_l1564_156453


namespace probability_two_number_cards_sum_15_l1564_156409

/-- The number of cards in a standard deck -/
def standardDeckSize : ℕ := 52

/-- The number of number cards (2 through 10) in each suit -/
def numberCardsPerSuit : ℕ := 9

/-- The number of suits in a standard deck -/
def numberOfSuits : ℕ := 4

/-- The total number of number cards (2 through 10) in a standard deck -/
def totalNumberCards : ℕ := numberCardsPerSuit * numberOfSuits

/-- The possible first card values that can sum to 15 with another number card -/
def validFirstCards : List ℕ := [5, 6, 7, 8, 9]

/-- The number of ways to choose two number cards that sum to 15 -/
def waysToSum15 : ℕ := validFirstCards.length * numberOfSuits

theorem probability_two_number_cards_sum_15 :
  (waysToSum15 : ℚ) / (standardDeckSize * (standardDeckSize - 1)) = 100 / 663 := by
  sorry

end probability_two_number_cards_sum_15_l1564_156409


namespace power_equation_solution_l1564_156408

theorem power_equation_solution : ∃ k : ℕ, 3 * 2^2001 - 3 * 2^2000 - 2^1999 + 2^1998 = k * 2^1998 ∧ k = 11 := by
  sorry

end power_equation_solution_l1564_156408


namespace quadratic_polynomial_special_roots_l1564_156467

theorem quadratic_polynomial_special_roots (p q : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 + p*x + q
  (∃ α β : ℝ, (f α = 0 ∧ f β = 0) ∧ 
   ((α = f 0 ∧ β = f 1) ∨ (α = f 1 ∧ β = f 0))) →
  f 6 = 71/2 - p := by
sorry

end quadratic_polynomial_special_roots_l1564_156467


namespace email_difference_l1564_156404

def morning_emails : ℕ := 10
def afternoon_emails : ℕ := 7
def evening_emails : ℕ := 17

theorem email_difference : morning_emails - afternoon_emails = 3 := by
  sorry

end email_difference_l1564_156404


namespace fourth_intersection_point_l1564_156420

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The curve xy = 2 -/
def on_curve (p : Point) : Prop :=
  p.x * p.y = 2

/-- The circle that intersects the curve at four points -/
structure Circle where
  center : Point
  radius : ℝ

/-- A point lies on the circle -/
def on_circle (c : Circle) (p : Point) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The theorem stating the fourth intersection point -/
theorem fourth_intersection_point (c : Circle) 
    (h1 : on_curve ⟨4, 1/2⟩ ∧ on_circle c ⟨4, 1/2⟩)
    (h2 : on_curve ⟨-2, -1⟩ ∧ on_circle c ⟨-2, -1⟩)
    (h3 : on_curve ⟨1/4, 8⟩ ∧ on_circle c ⟨1/4, 8⟩)
    (h4 : ∃ p, on_curve p ∧ on_circle c p ∧ p ≠ ⟨4, 1/2⟩ ∧ p ≠ ⟨-2, -1⟩ ∧ p ≠ ⟨1/4, 8⟩) :
    ∃ p, p = ⟨-1/8, -16⟩ ∧ on_curve p ∧ on_circle c p :=
sorry

end fourth_intersection_point_l1564_156420


namespace equal_area_trapezoid_result_l1564_156427

/-- 
A trapezoid with bases differing by 150 units, where x is the length of the segment 
parallel to the bases that divides the trapezoid into two equal-area regions.
-/
structure EqualAreaTrapezoid where
  base_diff : ℝ := 150
  x : ℝ
  divides_equally : x > 0

/-- 
The greatest integer not exceeding x^2/120 for an EqualAreaTrapezoid is 3000.
-/
theorem equal_area_trapezoid_result (t : EqualAreaTrapezoid) : 
  ⌊(t.x^2 / 120)⌋ = 3000 := by
  sorry

end equal_area_trapezoid_result_l1564_156427


namespace point_coordinates_l1564_156488

-- Define the point P
def P : ℝ × ℝ := sorry

-- Define the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Define the distance to x-axis
def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

-- Define the distance to y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  |p.1|

-- State the theorem
theorem point_coordinates :
  in_fourth_quadrant P ∧
  distance_to_x_axis P = 1 ∧
  distance_to_y_axis P = 2 →
  P = (2, -1) :=
by sorry

end point_coordinates_l1564_156488


namespace solution_verification_l1564_156435

-- Define the system of equations
def equation1 (x : ℝ) : Prop := 0.05 * x + 0.07 * (30 + x) = 14.9
def equation2 (x y : ℝ) : Prop := 0.03 * y - 5.6 = 0.07 * x

-- Theorem statement
theorem solution_verification :
  ∃ (x y : ℝ), equation1 x ∧ equation2 x y ∧ x = 106.67 ∧ y = 435.567 := by
  sorry

end solution_verification_l1564_156435


namespace isosceles_triangle_leg_range_l1564_156486

theorem isosceles_triangle_leg_range (x : ℝ) : 
  (∃ (base : ℝ), base > 0 ∧ x + x + base = 10 ∧ x + x > base ∧ x + base > x) ↔ 
  (5/2 < x ∧ x < 5) :=
by sorry

end isosceles_triangle_leg_range_l1564_156486


namespace min_sum_squares_with_real_root_l1564_156444

theorem min_sum_squares_with_real_root (a b : ℝ) : 
  (∃ x : ℝ, x^4 + a*x^3 + b*x^2 + a*x + 1 = 0) → a^2 + b^2 ≥ 4 :=
by sorry

end min_sum_squares_with_real_root_l1564_156444


namespace symmetric_curve_correct_l1564_156450

/-- The equation of a curve symmetric to y^2 = 4x with respect to the line x = 2 -/
def symmetric_curve_equation (x y : ℝ) : Prop :=
  y^2 = 16 - 4*x

/-- The original curve equation -/
def original_curve_equation (x y : ℝ) : Prop :=
  y^2 = 4*x

/-- The line of symmetry -/
def symmetry_line : ℝ := 2

/-- Theorem stating that the symmetric curve equation is correct -/
theorem symmetric_curve_correct :
  ∀ x y : ℝ, symmetric_curve_equation x y ↔ 
  original_curve_equation (2*symmetry_line - x) y :=
by sorry

end symmetric_curve_correct_l1564_156450


namespace largest_prime_factor_of_expression_l1564_156437

theorem largest_prime_factor_of_expression : 
  let n : ℤ := 16^4 + 3*16^2 + 2 - 17^4
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ n.natAbs ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ n.natAbs → q ≤ p ∧ p = 547 := by
  sorry

end largest_prime_factor_of_expression_l1564_156437


namespace young_inequality_l1564_156491

theorem young_inequality (p q a b : ℝ) : 
  0 < p → 0 < q → 1 / p + 1 / q = 1 → 0 < a → 0 < b →
  a * b ≤ a^p / p + b^q / q := by
  sorry

end young_inequality_l1564_156491


namespace inequality_proof_l1564_156452

theorem inequality_proof (a b θ : Real) 
  (h1 : a > b) (h2 : b > 1) (h3 : 0 < θ) (h4 : θ < π / 2) :
  a * Real.log (Real.sin θ) / Real.log b < b * Real.log (Real.sin θ) / Real.log a :=
sorry

end inequality_proof_l1564_156452


namespace employed_females_percentage_l1564_156462

theorem employed_females_percentage (total_population : ℝ) 
  (employed_percentage : ℝ) (employed_males_percentage : ℝ) :
  employed_percentage = 96 →
  employed_males_percentage = 24 →
  (employed_percentage - employed_males_percentage) / employed_percentage * 100 = 75 := by
  sorry

end employed_females_percentage_l1564_156462


namespace sum_of_two_squares_condition_l1564_156474

theorem sum_of_two_squares_condition (p : ℕ) (hp : Nat.Prime p) :
  (∃ a b : ℤ, p = a^2 + b^2) ↔ p % 4 = 1 ∨ p = 2 :=
sorry

end sum_of_two_squares_condition_l1564_156474


namespace uniform_cost_calculation_l1564_156485

/-- Calculates the total cost of uniforms for a student --/
def uniformCost (
  numUniforms : ℕ
) (
  pantsCost : ℚ
) (
  shirtCostMultiplier : ℚ
) (
  tieCostFraction : ℚ
) (
  socksCost : ℚ
) (
  jacketCostMultiplier : ℚ
) (
  shoesCost : ℚ
) (
  discountRate : ℚ
) (
  discountThreshold : ℕ
) : ℚ :=
  sorry

theorem uniform_cost_calculation :
  uniformCost 5 20 2 (1/5) 3 3 40 (1/10) 3 = 1039.5 := by
  sorry

end uniform_cost_calculation_l1564_156485


namespace geometric_increasing_condition_l1564_156499

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

def is_increasing_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, a n < a (n + 1)

theorem geometric_increasing_condition (a : ℕ → ℝ) :
  is_geometric_sequence a →
  (is_increasing_sequence a ↔ a 1 < a 2 ∧ a 2 < a 3) :=
sorry

end geometric_increasing_condition_l1564_156499


namespace remainder_plus_3255_l1564_156415

theorem remainder_plus_3255 (n : ℤ) (h : n % 5 = 2) : (n + 3255) % 5 = 2 := by
  sorry

end remainder_plus_3255_l1564_156415


namespace solve_for_y_l1564_156412

theorem solve_for_y (x y : ℝ) (h1 : 3 * x^2 = y - 6) (h2 : x = 4) : y = 54 := by
  sorry

end solve_for_y_l1564_156412


namespace sequence_periodicity_l1564_156475

/-- A cubic polynomial with rational coefficients -/
def CubicPolynomial (α : Type) [Field α] := α → α

/-- A sequence of rational numbers -/
def RationalSequence := ℕ → ℚ

/-- The statement that a sequence satisfies q_n = p(q_{n+1}) for all positive n -/
def SatisfiesRelation (p : CubicPolynomial ℚ) (q : RationalSequence) :=
  ∀ n : ℕ, q n = p (q (n + 1))

/-- The theorem stating the existence of a period for the sequence -/
theorem sequence_periodicity
  (p : CubicPolynomial ℚ)
  (q : RationalSequence)
  (h : SatisfiesRelation p q) :
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, q (n + k) = q n :=
sorry

end sequence_periodicity_l1564_156475


namespace total_pure_acid_in_mixture_l1564_156443

def solution1_concentration : ℝ := 0.20
def solution1_volume : ℝ := 8
def solution2_concentration : ℝ := 0.35
def solution2_volume : ℝ := 5

theorem total_pure_acid_in_mixture :
  let pure_acid1 := solution1_concentration * solution1_volume
  let pure_acid2 := solution2_concentration * solution2_volume
  pure_acid1 + pure_acid2 = 3.35 := by sorry

end total_pure_acid_in_mixture_l1564_156443


namespace room_area_l1564_156458

theorem room_area (breadth length : ℝ) : 
  length = 3 * breadth →
  2 * (length + breadth) = 16 →
  length * breadth = 12 := by
sorry

end room_area_l1564_156458


namespace average_discount_rate_proof_l1564_156413

theorem average_discount_rate_proof (bag_marked bag_sold shoes_marked shoes_sold jacket_marked jacket_sold : ℝ) 
  (h1 : bag_marked = 80)
  (h2 : bag_sold = 68)
  (h3 : shoes_marked = 120)
  (h4 : shoes_sold = 96)
  (h5 : jacket_marked = 150)
  (h6 : jacket_sold = 135) :
  let bag_discount := (bag_marked - bag_sold) / bag_marked
  let shoes_discount := (shoes_marked - shoes_sold) / shoes_marked
  let jacket_discount := (jacket_marked - jacket_sold) / jacket_marked
  (bag_discount + shoes_discount + jacket_discount) / 3 = 0.15 := by
  sorry

end average_discount_rate_proof_l1564_156413


namespace cloth_cost_theorem_l1564_156402

/-- The total cost of cloth given the length and price per meter -/
def total_cost (length : ℝ) (price_per_meter : ℝ) : ℝ :=
  length * price_per_meter

/-- Theorem: The total cost of 9.25 meters of cloth at $44 per meter is $407 -/
theorem cloth_cost_theorem :
  total_cost 9.25 44 = 407 := by
  sorry

end cloth_cost_theorem_l1564_156402


namespace find_other_number_l1564_156494

theorem find_other_number (A B : ℕ+) (h1 : Nat.lcm A B = 2310) (h2 : Nat.gcd A B = 83) (h3 : A = 210) : B = 913 := by
  sorry

end find_other_number_l1564_156494


namespace sum_of_solutions_l1564_156431

-- Define the equation
def equation (M : ℝ) : Prop := M * (M - 8) = -8

-- Theorem statement
theorem sum_of_solutions : 
  ∃ (M₁ M₂ : ℝ), equation M₁ ∧ equation M₂ ∧ M₁ + M₂ = 8 := by
  sorry

end sum_of_solutions_l1564_156431


namespace melody_cutouts_l1564_156438

/-- Given that Melody planned to paste 4 cut-outs on each card and made 6 cards in total,
    prove that the total number of cut-outs she made is 24. -/
theorem melody_cutouts (cutouts_per_card : ℕ) (total_cards : ℕ) 
  (h1 : cutouts_per_card = 4) 
  (h2 : total_cards = 6) : 
  cutouts_per_card * total_cards = 24 := by
  sorry

end melody_cutouts_l1564_156438


namespace quadratic_inequality_solution_l1564_156483

theorem quadratic_inequality_solution (c : ℝ) : 
  (∀ x : ℝ, x^2 + x - c < 0 ↔ -2 < x ∧ x < 1) → c = 2 := by
  sorry

end quadratic_inequality_solution_l1564_156483


namespace triangle_condition_implies_right_angle_l1564_156498

-- Define a triangle with side lengths a, b, and c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the condition from the problem
def satisfiesCondition (t : Triangle) : Prop :=
  (t.a - 3)^2 + Real.sqrt (t.b - 4) + |t.c - 5| = 0

-- Define what it means for a triangle to be right-angled
def isRightTriangle (t : Triangle) : Prop :=
  t.a^2 + t.b^2 = t.c^2

-- Theorem statement
theorem triangle_condition_implies_right_angle (t : Triangle) :
  satisfiesCondition t → isRightTriangle t :=
by sorry

end triangle_condition_implies_right_angle_l1564_156498


namespace arithmetic_sequence_properties_l1564_156459

/-- An arithmetic sequence with its sum sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  S : ℕ → ℝ  -- Sum sequence
  arithmetic_seq : ∀ n, a (n + 1) = a n + d
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * d) / 2

/-- Main theorem about properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) 
    (h1 : seq.S 6 < seq.S 7) (h2 : seq.S 7 > seq.S 8) :
  seq.d < 0 ∧ seq.S 9 < seq.S 6 ∧ ∀ n, seq.S n ≤ seq.S 7 := by
  sorry


end arithmetic_sequence_properties_l1564_156459


namespace complex_sum_nonzero_components_l1564_156440

theorem complex_sum_nonzero_components (a b : ℝ) :
  (a : ℂ) + b * Complex.I = (1 - Complex.I)^10 + (1 + Complex.I)^10 →
  a ≠ 0 ∧ b ≠ 0 :=
by sorry

end complex_sum_nonzero_components_l1564_156440


namespace expression_value_l1564_156428

theorem expression_value (a b : ℝ) (h : a + b = 1) :
  a^3 + b^3 + 3*(a^3*b + a*b^3) + 6*(a^3*b^2 + a^2*b^3) = 1 := by
  sorry

end expression_value_l1564_156428


namespace tolu_pencils_tolu_wants_three_pencils_l1564_156456

/-- The problem of determining the number of pencils Tolu wants -/
theorem tolu_pencils (pencil_price : ℚ) (robert_pencils melissa_pencils : ℕ) 
  (total_spent : ℚ) : ℕ :=
  let tolu_pencils := (total_spent - pencil_price * (robert_pencils + melissa_pencils)) / pencil_price
  3

/-- The main theorem stating that Tolu wants 3 pencils -/
theorem tolu_wants_three_pencils : 
  tolu_pencils (20 / 100) 5 2 2 = 3 := by
  sorry

end tolu_pencils_tolu_wants_three_pencils_l1564_156456


namespace digit_sum_problem_l1564_156416

theorem digit_sum_problem (a b c d : ℕ) : 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧  -- Digits are less than 10
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧  -- All digits are different
  100 * a + 10 * b + c + 100 * d + 10 * c + a = 1100  -- The equation
  → a + b + c + d = 19 := by
sorry

end digit_sum_problem_l1564_156416


namespace sum_of_powers_l1564_156460

theorem sum_of_powers (x : ℝ) (h1 : x^2020 - 3*x + 2 = 0) (h2 : x ≠ 1) :
  x^2019 + x^2018 + x^2017 + x^2016 + x^2015 + x^2014 + x^2013 + x^2012 + x^2011 + x^2010 +
  x^2009 + x^2008 + x^2007 + x^2006 + x^2005 + x^2004 + x^2003 + x^2002 + x^2001 + x^2000 +
  x^1999 + x^1998 + x^1997 + x^1996 + x^1995 + x^1994 + x^1993 + x^1992 + x^1991 + x^1990 +
  x^1989 + x^1988 + x^1987 + x^1986 + x^1985 + x^1984 + x^1983 + x^1982 + x^1981 + x^1980 +
  x^1979 + x^1978 + x^1977 + x^1976 + x^1975 + x^1974 + x^1973 + x^1972 + x^1971 + x^1970 +
  -- ... (continue for all powers from 1969 to 2)
  x^2 + x + 1 - 2 = 2 := by
  sorry

end sum_of_powers_l1564_156460


namespace minimize_resistance_l1564_156479

/-- Represents the resistance of a component assembled using six resistors. -/
noncomputable def totalResistance (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) (R₁ R₂ R₃ R₄ R₅ R₆ : ℝ) : ℝ :=
  sorry -- Definition of total resistance based on the given configuration

/-- Theorem stating the condition for minimizing the total resistance of the component. -/
theorem minimize_resistance
  (a₁ a₂ a₃ a₄ a₅ a₆ : ℝ)
  (h₁ : a₁ > a₂) (h₂ : a₂ > a₃) (h₃ : a₃ > a₄) (h₄ : a₄ > a₅) (h₅ : a₅ > a₆)
  (h₆ : a₁ > 0) (h₇ : a₂ > 0) (h₈ : a₃ > 0) (h₉ : a₄ > 0) (h₁₀ : a₅ > 0) (h₁₁ : a₆ > 0) :
  ∃ (R₁ R₂ : ℝ), 
    (R₁ = a₁ ∧ R₂ = a₂) ∨ (R₁ = a₂ ∧ R₂ = a₁) ∧
    ∀ (S₁ S₂ S₃ S₄ S₅ S₆ : ℝ),
      totalResistance a₁ a₂ a₃ a₄ a₅ a₆ R₁ R₂ a₃ a₄ a₅ a₆ ≤ 
      totalResistance a₁ a₂ a₃ a₄ a₅ a₆ S₁ S₂ S₃ S₄ S₅ S₆ :=
by
  sorry

end minimize_resistance_l1564_156479


namespace scott_runs_84_miles_per_month_l1564_156455

/-- Scott's weekly running schedule -/
structure RunningSchedule where
  mon_to_wed : ℕ  -- Miles run Monday through Wednesday (daily)
  thu_fri : ℕ     -- Miles run Thursday and Friday (daily)

/-- Calculate total miles run in a week -/
def weekly_miles (schedule : RunningSchedule) : ℕ :=
  schedule.mon_to_wed * 3 + schedule.thu_fri * 2

/-- Calculate total miles run in a month -/
def monthly_miles (schedule : RunningSchedule) (weeks : ℕ) : ℕ :=
  weekly_miles schedule * weeks

/-- Scott's actual running schedule -/
def scotts_schedule : RunningSchedule :=
  { mon_to_wed := 3, thu_fri := 6 }

/-- Theorem: Scott runs 84 miles in a month with 4 weeks -/
theorem scott_runs_84_miles_per_month : 
  monthly_miles scotts_schedule 4 = 84 := by sorry

end scott_runs_84_miles_per_month_l1564_156455


namespace duplicated_page_number_l1564_156451

/-- The sum of natural numbers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The theorem statement -/
theorem duplicated_page_number :
  ∀ k : ℕ,
  (k ≤ 70) →
  (sum_to_n 70 + k = 2550) →
  (k = 65) :=
by sorry

end duplicated_page_number_l1564_156451


namespace farm_animals_difference_l1564_156446

theorem farm_animals_difference : 
  ∀ (pigs dogs sheep : ℕ), 
    pigs = 42 → 
    sheep = 48 → 
    pigs = dogs → 
    pigs + dogs - sheep = 36 := by
  sorry

end farm_animals_difference_l1564_156446


namespace two_digit_number_digit_difference_l1564_156421

/-- Given a two-digit number where the difference between the original number
    and the number with interchanged digits is 45, prove that the difference
    between its two digits is 5. -/
theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 45 → x - y = 5 := by
  sorry

end two_digit_number_digit_difference_l1564_156421


namespace inequality_proof_l1564_156423

theorem inequality_proof (α x y z : ℝ) (hα : α > 0) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x * y + y * z + z * x = α) :
  (1 + α / x^2) * (1 + α / y^2) * (1 + α / z^2) ≥ (1/3) * (x / z + z / x + 2) :=
by sorry

end inequality_proof_l1564_156423


namespace phillip_initial_vinegar_l1564_156463

/-- The number of jars Phillip has -/
def num_jars : ℕ := 4

/-- The number of cucumbers Phillip has -/
def num_cucumbers : ℕ := 10

/-- The number of pickles each cucumber makes -/
def pickles_per_cucumber : ℕ := 6

/-- The number of pickles each jar can hold -/
def pickles_per_jar : ℕ := 12

/-- The amount of vinegar (in ounces) needed per jar of pickles -/
def vinegar_per_jar : ℕ := 10

/-- The amount of vinegar (in ounces) left after making pickles -/
def vinegar_left : ℕ := 60

/-- Theorem stating that Phillip started with 100 ounces of vinegar -/
theorem phillip_initial_vinegar : 
  (min num_jars ((num_cucumbers * pickles_per_cucumber) / pickles_per_jar)) * vinegar_per_jar + vinegar_left = 100 := by
  sorry

end phillip_initial_vinegar_l1564_156463


namespace workshop_salary_problem_l1564_156466

theorem workshop_salary_problem (total_workers : ℕ) (avg_salary : ℝ) 
  (technicians : ℕ) (technician_avg_salary : ℝ) :
  total_workers = 28 →
  avg_salary = 8000 →
  technicians = 7 →
  technician_avg_salary = 14000 →
  (total_workers * avg_salary - technicians * technician_avg_salary) / (total_workers - technicians) = 6000 :=
by
  sorry

end workshop_salary_problem_l1564_156466


namespace exists_winning_strategy_l1564_156424

/-- Represents a card in the game -/
structure Card where
  id : Nat
  deriving Repr

/-- Represents the state of the game -/
structure GameState where
  player1_cards : List Card
  player2_cards : List Card
  deriving Repr

/-- Represents the strength relationship between cards -/
def beats (card1 card2 : Card) : Bool := sorry

/-- Represents a single turn in the game -/
def play_turn (state : GameState) : GameState := sorry

/-- Represents the strategy chosen by the players -/
def strategy (state : GameState) : GameState := sorry

/-- Theorem stating that there exists a strategy to end the game -/
theorem exists_winning_strategy 
  (n : Nat) 
  (initial_state : GameState) 
  (h1 : initial_state.player1_cards.length + initial_state.player2_cards.length = n) 
  (h2 : ∀ c1 c2 : Card, c1 ≠ c2 → (beats c1 c2 ∨ beats c2 c1)) :
  ∃ (final_state : GameState), 
    (final_state.player1_cards.length = 0 ∨ final_state.player2_cards.length = 0) ∧
    (∃ k : Nat, (strategy^[k]) initial_state = final_state) :=
sorry

end exists_winning_strategy_l1564_156424


namespace pet_shop_total_l1564_156436

-- Define the number of each type of animal
def num_kittens : ℕ := 32
def num_hamsters : ℕ := 15
def num_birds : ℕ := 30

-- Define the total number of animals
def total_animals : ℕ := num_kittens + num_hamsters + num_birds

-- Theorem to prove
theorem pet_shop_total : total_animals = 77 := by
  sorry

end pet_shop_total_l1564_156436


namespace sum_of_reciprocals_l1564_156400

theorem sum_of_reciprocals (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 15) (h4 : a * b = 225) : 
  1 / a + 1 / b = 1 / 15 := by
sorry

end sum_of_reciprocals_l1564_156400


namespace triangle_side_length_l1564_156426

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if the area is √3, B = 60°, and a² + c² = 3ac, then b = 2√2 -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2 * a * c * Real.sin B = Real.sqrt 3) →  -- Area condition
  (B = π/3) →  -- 60° in radians
  (a^2 + c^2 = 3*a*c) →  -- Given condition
  (b = 2 * Real.sqrt 2) := by
sorry


end triangle_side_length_l1564_156426


namespace pears_for_20_apples_l1564_156477

/-- The number of apples that cost the same as 5 oranges -/
def apples_per_5_oranges : ℕ := 10

/-- The number of oranges that cost the same as 4 pears -/
def oranges_per_4_pears : ℕ := 3

/-- The number of apples we want to find the equivalent pears for -/
def target_apples : ℕ := 20

/-- The function to calculate the number of pears equivalent to a given number of apples -/
def pears_for_apples (n : ℕ) : ℚ :=
  (n : ℚ) * 5 / apples_per_5_oranges * 4 / oranges_per_4_pears

theorem pears_for_20_apples :
  pears_for_apples target_apples = 40 / 3 := by
  sorry

end pears_for_20_apples_l1564_156477


namespace equation_solution_l1564_156425

theorem equation_solution : 
  ∃! x : ℚ, (x + 10) / (x - 4) = (x - 3) / (x + 6) ∧ x = -48/23 := by
  sorry

end equation_solution_l1564_156425


namespace coin_value_proof_l1564_156496

/-- The value of a penny in dollars -/
def penny_value : ℚ := 1 / 100

/-- The value of a nickel in dollars -/
def nickel_value : ℚ := 5 / 100

/-- The value of a dime in dollars -/
def dime_value : ℚ := 10 / 100

/-- The number of pennies -/
def num_pennies : ℕ := 9

/-- The number of nickels -/
def num_nickels : ℕ := 4

/-- The number of dimes -/
def num_dimes : ℕ := 3

/-- The total value of the coins in dollars -/
def total_value : ℚ := num_pennies * penny_value + num_nickels * nickel_value + num_dimes * dime_value

theorem coin_value_proof : total_value = 59 / 100 := by
  sorry

end coin_value_proof_l1564_156496


namespace square_between_500_600_l1564_156468

theorem square_between_500_600 : ∃ n : ℕ, 
  500 < n^2 ∧ n^2 ≤ 600 ∧ (n-1)^2 < 500 := by
  sorry

end square_between_500_600_l1564_156468


namespace cistern_filling_time_l1564_156414

theorem cistern_filling_time (x : ℝ) : 
  x > 0 ∧                            -- x is positive (time can't be negative or zero)
  (1 / x + 1 / 12 - 1 / 20 = 1 / 7.5) -- combined rate equation
  → x = 10 := by
sorry

end cistern_filling_time_l1564_156414


namespace island_population_theorem_l1564_156480

theorem island_population_theorem (a b c d : ℝ) 
  (h1 : a / (a + b) = 0.65)  -- 65% of blue-eyed are brunettes
  (h2 : b / (b + c) = 0.7)   -- 70% of blondes have blue eyes
  (h3 : c / (c + d) = 0.1)   -- 10% of green-eyed are blondes
  (h4 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) -- All populations are positive
  : d / (a + b + c + d) = 0.54 := by
  sorry

#check island_population_theorem

end island_population_theorem_l1564_156480


namespace hyperbola_equation_hyperbola_eccentricity_l1564_156401

/-- Represents a hyperbola with equation (x^2 / a^2) - (y^2 / b^2) = 1 -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a_pos : 0 < a
  h_b_pos : 0 < b
  h_c_eq : c = 2
  h_asymptote : b = a

/-- The equation of the hyperbola is (x^2 / 2) - (y^2 / 2) = 1 -/
theorem hyperbola_equation (h : Hyperbola) : 
  h.a^2 = 2 ∧ h.b^2 = 2 :=
sorry

/-- The eccentricity of the hyperbola is √2 -/
theorem hyperbola_eccentricity (h : Hyperbola) : 
  Real.sqrt (h.c^2 / h.a^2) = Real.sqrt 2 :=
sorry

end hyperbola_equation_hyperbola_eccentricity_l1564_156401


namespace race_elimination_proof_l1564_156447

/-- The number of racers at the start of the race -/
def initial_racers : ℕ := 100

/-- The number of racers in the final section -/
def final_racers : ℕ := 30

/-- The fraction of racers remaining after the second segment -/
def second_segment_fraction : ℚ := 2/3

/-- The fraction of racers remaining after the third segment -/
def third_segment_fraction : ℚ := 1/2

/-- The number of racers eliminated after the first segment -/
def eliminated_first_segment : ℕ := 10

theorem race_elimination_proof :
  (↑final_racers : ℚ) = third_segment_fraction * second_segment_fraction * (initial_racers - eliminated_first_segment) :=
sorry

end race_elimination_proof_l1564_156447


namespace candy_bar_cost_l1564_156497

/-- Given that the total cost of 2 candy bars is $4 and each candy bar costs the same amount,
    prove that the cost of each candy bar is $2. -/
theorem candy_bar_cost (total_cost : ℝ) (num_bars : ℕ) (cost_per_bar : ℝ) : 
  total_cost = 4 → num_bars = 2 → total_cost = num_bars * cost_per_bar → cost_per_bar = 2 := by
  sorry

end candy_bar_cost_l1564_156497


namespace quadratic_with_property_has_negative_root_l1564_156441

/-- A quadratic polynomial with the given property has at least one negative root -/
theorem quadratic_with_property_has_negative_root (f : ℝ → ℝ) 
  (h1 : ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0) 
  (h2 : ∀ (a b : ℝ), f (a^2 + b^2) ≥ f (2*a*b)) :
  ∃ (x : ℝ), x < 0 ∧ f x = 0 :=
sorry

end quadratic_with_property_has_negative_root_l1564_156441


namespace impossible_equal_tokens_l1564_156407

/-- Represents the state of tokens --/
structure TokenState where
  green : ℕ
  red : ℕ

/-- Represents a token exchange operation --/
inductive Exchange
  | GreenToRed
  | RedToGreen

/-- Applies an exchange to a token state --/
def applyExchange (state : TokenState) (ex : Exchange) : TokenState :=
  match ex with
  | Exchange.GreenToRed => 
      if state.green ≥ 1 then ⟨state.green - 1, state.red + 5⟩ else state
  | Exchange.RedToGreen => 
      if state.red ≥ 1 then ⟨state.green + 5, state.red - 1⟩ else state

/-- A sequence of exchanges --/
def ExchangeSequence := List Exchange

/-- Applies a sequence of exchanges to a token state --/
def applyExchangeSequence (state : TokenState) (seq : ExchangeSequence) : TokenState :=
  seq.foldl applyExchange state

/-- The theorem to be proved --/
theorem impossible_equal_tokens : 
  ∀ (seq : ExchangeSequence), 
  let finalState := applyExchangeSequence ⟨1, 0⟩ seq
  finalState.green ≠ finalState.red :=
sorry

end impossible_equal_tokens_l1564_156407


namespace max_value_theorem_l1564_156478

theorem max_value_theorem (u v : ℝ) 
  (h1 : 2 * u + 3 * v ≤ 10) 
  (h2 : 4 * u + v ≤ 9) : 
  u + 2 * v ≤ 6.1 ∧ ∃ (u₀ v₀ : ℝ), 2 * u₀ + 3 * v₀ ≤ 10 ∧ 4 * u₀ + v₀ ≤ 9 ∧ u₀ + 2 * v₀ = 6.1 :=
by sorry

end max_value_theorem_l1564_156478


namespace job_completion_time_l1564_156469

theorem job_completion_time (a_time b_time : ℕ) (remaining_fraction : ℚ) : 
  a_time = 15 → b_time = 20 → remaining_fraction = 8/15 →
  ∃ (days_worked : ℕ), days_worked = 4 ∧
    (1 - remaining_fraction) = days_worked * (1/a_time + 1/b_time) :=
by sorry

end job_completion_time_l1564_156469


namespace sqrt_40_div_sqrt_5_l1564_156472

theorem sqrt_40_div_sqrt_5 : Real.sqrt 40 / Real.sqrt 5 = 2 * Real.sqrt 2 := by
  sorry

end sqrt_40_div_sqrt_5_l1564_156472


namespace cost_mms_in_snickers_l1564_156417

/-- The cost of a pack of M&M's in terms of Snickers pieces -/
theorem cost_mms_in_snickers 
  (snickers_quantity : ℕ)
  (mms_quantity : ℕ)
  (snickers_price : ℚ)
  (total_paid : ℚ)
  (change_received : ℚ)
  (h1 : snickers_quantity = 2)
  (h2 : mms_quantity = 3)
  (h3 : snickers_price = 3/2)
  (h4 : total_paid = 20)
  (h5 : change_received = 8) :
  (total_paid - change_received - snickers_quantity * snickers_price) / mms_quantity = 2 * snickers_price :=
by sorry

end cost_mms_in_snickers_l1564_156417


namespace smallest_constant_inequality_l1564_156482

theorem smallest_constant_inequality (D : ℝ) : 
  (∀ x y z : ℝ, x^2 + y^2 + 4 ≥ D * (x + y + z)) ↔ D ≤ 2 * Real.sqrt 2 :=
sorry

end smallest_constant_inequality_l1564_156482


namespace parabola_equation_l1564_156484

-- Define the line on which the focus lies
def focus_line (x y : ℝ) : Prop := x + 2 * y + 3 = 0

-- Define the two possible standard equations for the parabola
def vertical_parabola (x y : ℝ) : Prop := x^2 = -6 * y
def horizontal_parabola (x y : ℝ) : Prop := y^2 = -12 * x

-- Theorem statement
theorem parabola_equation (C : Set (ℝ × ℝ)) :
  (∃ (x y : ℝ), (x, y) ∈ C ∧ focus_line x y) →
  (∀ (x y : ℝ), (x, y) ∈ C → vertical_parabola x y ∨ horizontal_parabola x y) :=
by sorry

end parabola_equation_l1564_156484


namespace min_value_implies_a_equals_two_l1564_156490

theorem min_value_implies_a_equals_two (x y a : ℝ) :
  x + 3*y + 5 ≥ 0 →
  x + y - 1 ≤ 0 →
  x + a ≥ 0 →
  (∀ x' y', x' + 3*y' + 5 ≥ 0 → x' + y' - 1 ≤ 0 → x' + 2*y' ≥ x + 2*y) →
  x + 2*y = -4 →
  a = 2 := by
sorry

end min_value_implies_a_equals_two_l1564_156490


namespace original_average_age_proof_l1564_156422

theorem original_average_age_proof (initial_avg : ℝ) (new_students : ℕ) (new_students_avg : ℝ) (avg_decrease : ℝ) :
  initial_avg = 40 →
  new_students = 12 →
  new_students_avg = 34 →
  avg_decrease = 4 →
  initial_avg = 40 := by
sorry

end original_average_age_proof_l1564_156422


namespace recommendation_plans_count_l1564_156464

/-- The number of universities --/
def num_universities : ℕ := 3

/-- The number of students to be recommended --/
def num_students : ℕ := 4

/-- The maximum number of students a university can accept --/
def max_students_per_university : ℕ := 2

/-- The function that calculates the number of recommendation plans --/
noncomputable def num_recommendation_plans : ℕ := sorry

/-- Theorem stating that the number of recommendation plans is 54 --/
theorem recommendation_plans_count : num_recommendation_plans = 54 := by sorry

end recommendation_plans_count_l1564_156464


namespace circle_origin_inside_l1564_156465

theorem circle_origin_inside (m : ℝ) : 
  (∀ x y : ℝ, (x - m)^2 + (y + m)^2 < 4 → x^2 + y^2 = 0) → 
  -Real.sqrt 2 < m ∧ m < Real.sqrt 2 := by
  sorry

end circle_origin_inside_l1564_156465


namespace factorization_equality_l1564_156432

theorem factorization_equality (x : ℝ) : 
  32 * x^4 - 48 * x^7 + 16 * x^2 = 16 * x^2 * (2 * x^2 - 3 * x^5 + 1) := by
  sorry

end factorization_equality_l1564_156432


namespace completing_square_result_l1564_156495

theorem completing_square_result (x : ℝ) : 
  x^2 - 4*x - 1 = 0 → (x - 2)^2 = 5 := by
  sorry

end completing_square_result_l1564_156495


namespace f_min_at_neg_three_l1564_156493

/-- The function f(x) = x^2 + 6x + 1 -/
def f (x : ℝ) : ℝ := x^2 + 6*x + 1

/-- Theorem stating that f(x) is minimized when x = -3 -/
theorem f_min_at_neg_three :
  ∀ x : ℝ, f (-3) ≤ f x :=
by sorry

end f_min_at_neg_three_l1564_156493


namespace fraction_power_product_l1564_156434

theorem fraction_power_product : (9/8 : ℚ)^4 * (8/9 : ℚ)^4 = 1 := by
  sorry

end fraction_power_product_l1564_156434


namespace smallest_in_set_l1564_156430

theorem smallest_in_set : 
  let S : Set ℤ := {0, -1, 1, 2}
  ∀ x ∈ S, -1 ≤ x :=
by sorry

end smallest_in_set_l1564_156430


namespace count_valid_numbers_l1564_156473

def is_valid_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ n = 34 * (n / 100 + (n / 10 % 10) + (n % 10))

theorem count_valid_numbers :
  ∃ (S : Finset ℕ), (∀ n ∈ S, is_valid_number n) ∧ S.card = 4 ∧
  (∀ m : ℕ, is_valid_number m → m ∈ S) :=
sorry

end count_valid_numbers_l1564_156473


namespace eightieth_digit_is_one_l1564_156487

def sequence_digit (n : ℕ) : ℕ :=
  if n ≤ 102 then
    let num := 60 - ((n - 1) / 2)
    if n % 2 = 0 then num % 10 else (num / 10) % 10
  else
    sorry -- Handle single-digit numbers if needed

theorem eightieth_digit_is_one :
  sequence_digit 80 = 1 := by
  sorry

end eightieth_digit_is_one_l1564_156487


namespace probability_at_least_three_white_is_550_715_l1564_156410

def white_balls : ℕ := 8
def black_balls : ℕ := 7
def total_balls : ℕ := white_balls + black_balls
def drawn_balls : ℕ := 6

def probability_at_least_three_white : ℚ :=
  (Nat.choose white_balls 3 * Nat.choose black_balls 3 +
   Nat.choose white_balls 4 * Nat.choose black_balls 2 +
   Nat.choose white_balls 5 * Nat.choose black_balls 1 +
   Nat.choose white_balls 6 * Nat.choose black_balls 0) /
  Nat.choose total_balls drawn_balls

theorem probability_at_least_three_white_is_550_715 :
  probability_at_least_three_white = 550 / 715 :=
by sorry

end probability_at_least_three_white_is_550_715_l1564_156410


namespace unique_quadratic_solution_l1564_156454

theorem unique_quadratic_solution (c : ℝ) : 
  (c ≠ 0 ∧ 
   ∃! b : ℝ, b > 0 ∧ 
   ∃! x : ℝ, x^2 + (b + 2/b) * x + c = 0) ↔ 
  c = 2 :=
sorry

end unique_quadratic_solution_l1564_156454


namespace max_glow_count_max_glow_count_for_given_conditions_l1564_156492

/-- The maximum number of times a light can glow in a given time range -/
theorem max_glow_count (total_duration : ℕ) (glow_interval : ℕ) : ℕ :=
  (total_duration / glow_interval : ℕ)

/-- Proof that the maximum number of glows is 236 for the given conditions -/
theorem max_glow_count_for_given_conditions :
  max_glow_count 4969 21 = 236 := by
  sorry

end max_glow_count_max_glow_count_for_given_conditions_l1564_156492


namespace abcd_equation_solutions_l1564_156489

theorem abcd_equation_solutions :
  ∀ (A B C D : ℕ),
    0 ≤ A ∧ A ≤ 9 ∧
    0 ≤ B ∧ B ≤ 9 ∧
    0 ≤ C ∧ C ≤ 9 ∧
    0 ≤ D ∧ D ≤ 9 ∧
    1000 ≤ 1000 * A + 100 * B + 10 * C + D ∧
    1000 * A + 100 * B + 10 * C + D ≤ 9999 ∧
    1000 * A + 100 * B + 10 * C + D = (10 * A + D) * (101 * A + 10 * D) →
    (A = 1 ∧ B = 0 ∧ C = 1 ∧ D = 0) ∨
    (A = 1 ∧ B = 2 ∧ C = 2 ∧ D = 1) ∨
    (A = 1 ∧ B = 4 ∧ C = 5 ∧ D = 2) ∨
    (A = 1 ∧ B = 7 ∧ C = 0 ∧ D = 3) ∨
    (A = 1 ∧ B = 9 ∧ C = 7 ∧ D = 4) :=
by sorry

end abcd_equation_solutions_l1564_156489


namespace certain_number_minus_32_l1564_156433

theorem certain_number_minus_32 (x : ℤ) (h : x - 48 = 22) : x - 32 = 38 := by
  sorry

end certain_number_minus_32_l1564_156433


namespace arithmetic_mean_after_removal_l1564_156457

theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y : ℝ) :
  S.card = 60 →
  x ∈ S →
  y ∈ S →
  x = 50 →
  y = 65 →
  (S.sum id) / S.card = 42 →
  ((S.sum id) - x - y) / (S.card - 2) = 41.5 := by
  sorry

end arithmetic_mean_after_removal_l1564_156457


namespace distribute_negative_three_l1564_156476

theorem distribute_negative_three (x y : ℝ) : -3 * (x - x * y) = -3 * x + 3 * x * y := by
  sorry

end distribute_negative_three_l1564_156476


namespace quadratic_difference_theorem_l1564_156429

theorem quadratic_difference_theorem (a b : ℝ) :
  (∀ x y : ℝ, (a*x^2 + 2*x*y - x) - (3*x^2 - 2*b*x*y + 3*y) = (-x + 3*y)) →
  a^2 - 4*b = 13 := by
sorry

end quadratic_difference_theorem_l1564_156429


namespace angle_after_rotation_l1564_156406

def rotation_result (initial_angle rotation : ℕ) : ℕ :=
  (rotation - initial_angle) % 360

theorem angle_after_rotation (initial_angle : ℕ) (h1 : initial_angle = 70) (rotation : ℕ) (h2 : rotation = 960) :
  rotation_result initial_angle rotation = 170 := by
  sorry

end angle_after_rotation_l1564_156406


namespace evaluate_expression_l1564_156403

/-- Given x, y, and z are variables, prove that (25x³y) · (4xy²z) · (1/(5xyz)²) = 4x²y/z -/
theorem evaluate_expression (x y z : ℝ) (h : z ≠ 0) :
  (25 * x^3 * y) * (4 * x * y^2 * z) * (1 / (5 * x * y * z)^2) = 4 * x^2 * y / z := by
  sorry

end evaluate_expression_l1564_156403
